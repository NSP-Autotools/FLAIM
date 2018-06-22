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

#include "jniftk.h"
#include "jnirestore.h"

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreClient::openBackupSet( void)
{
	RCODE				rc = NE_XFLM_OK;
	JNIEnv *			pEnv;
	jclass			Cls;
	jmethodID		MId;
	FLMBOOL			bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jClient);
	MId = pEnv->GetMethodID( Cls, "openBackupSet", "()I");
	flmAssert( MId);
		
	if( RC_BAD( rc = (RCODE)pEnv->CallIntMethod( m_jClient, MId)))
	{
		goto Exit;
	}
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreClient::openRflFile(
	FLMUINT			uiFileNum)
{
	RCODE				rc = NE_XFLM_OK;
	JNIEnv *			pEnv;
	jclass			Cls;
	jmethodID		MId;
	FLMBOOL			bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jClient);
	MId = pEnv->GetMethodID( Cls, "opeRflFile", "(I)I");
	flmAssert( MId);
		
	if( RC_BAD( rc = (RCODE)pEnv->CallIntMethod( 
		m_jClient, MId, (jint)uiFileNum)))
	{
		goto Exit;
	}
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}
	}

	return( rc);	
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreClient::openIncFile(
	FLMUINT			uiFileNum)
{
	RCODE				rc = NE_XFLM_OK;
	JNIEnv *			pEnv;
	jclass			Cls;
	jmethodID		MId;
	FLMBOOL			bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jClient);
	MId = pEnv->GetMethodID( Cls, "openIncFile", "(I)I");
	flmAssert( MId);
		
	if( RC_BAD( rc = (RCODE)pEnv->CallIntMethod( 
		m_jClient, MId, (jint)uiFileNum)))
	{
		goto Exit;
	}
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreClient::read(
	FLMUINT			uiLength,
	void *			pvBuffer,
	FLMUINT *		puiBytesRead)
{
	RCODE				rc = NE_XFLM_OK;
	JNIEnv *			pEnv;
	jclass			Cls;
	jmethodID		MId;
	jbyteArray		ByteBuffer;
	jintArray		IntBuffer;
	jint				iBytesRead;
	void *			pvTemp;
	FLMBOOL			bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET(NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}
	
	Cls = pEnv->GetObjectClass( m_jClient);
	MId = pEnv->GetMethodID( Cls, "read", "([B[I)I");
	flmAssert( MId);
	
	ByteBuffer = pEnv->NewByteArray( (jsize)uiLength);
	IntBuffer = pEnv->NewIntArray( 1);
	
	rc = (RCODE)pEnv->CallIntMethod( m_jClient, MId, 
						ByteBuffer, IntBuffer);

	// Get the value out of IntBuffer
	
	pEnv->GetIntArrayRegion( IntBuffer, 0, 1, &iBytesRead);
	flmAssert( iBytesRead > 0);
	*puiBytesRead = (FLMUINT)iBytesRead;
					
	// Get the data out of ByteBuffer
	
	pvTemp = pEnv->GetPrimitiveArrayCritical( ByteBuffer, NULL);
	f_memcpy( pvBuffer, pvTemp, (FLMUINT)iBytesRead);
	pEnv->ReleasePrimitiveArrayCritical( ByteBuffer, pvTemp, JNI_ABORT);

Exit:

	if (m_pJvm->DetachCurrentThread() != 0)
	{
		flmAssert( 0);
		rc = RC_SET( NE_XFLM_FAILURE);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreClient::close( void)
{
	RCODE				rc = NE_XFLM_OK;
	JNIEnv *			pEnv;
	jclass			Cls;
	jmethodID		MId;
	FLMBOOL			bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jClient);
	MId = pEnv->GetMethodID( Cls, "close", "()I");
	flmAssert( MId);
		
	if( RC_BAD( rc = (RCODE)pEnv->CallIntMethod( m_jClient, MId)))
	{
		goto Exit;
	}
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreClient::abortFile( void)
{
	RCODE				rc = NE_XFLM_OK;
	JNIEnv *			pEnv;
	jclass			Cls;
	jmethodID		MId;
	FLMBOOL			bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jClient);
	MId = pEnv->GetMethodID( Cls, "abortFile", "()I");
	flmAssert( MId);
		
	if( RC_BAD( rc = (RCODE)pEnv->CallIntMethod( m_jClient, MId)))
	{
		goto Exit;
	}
								  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportProgress(
	eRestoreAction *		peAction,
	FLMUINT64				ui64BytesToDo,
	FLMUINT64				ui64BytesDone)
{
	RCODE				rc = NE_XFLM_OK;
	JNIEnv *			pEnv;
	jclass			Cls;
	jmethodID		MId;
	FLMBOOL			bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET(NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportProgress", "(JJ)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId,
													 (jlong)ui64BytesToDo,
													 (jlong)ui64BytesDone);

Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportError(
	eRestoreAction *		peAction,
	RCODE						rcErr)
{
	RCODE				rc = NE_XFLM_OK;
	JNIEnv *			pEnv;
	jclass			Cls;
	jmethodID		MId;
	FLMBOOL			bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET(NE_XFLM_FAILURE);	
			goto Exit;
		}
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportError", "(I)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId,
													 (jint)rcErr);
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);	
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportBeginTrans(
	eRestoreAction *		peAction,
	FLMUINT64				ui64TransId)
{
	RCODE			rc = NE_XFLM_OK;
	JNIEnv *		pEnv;
	jclass		Cls;
	jmethodID	MId;
	FLMBOOL		bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET(NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportBeginTrans", "(J)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId,
													 (jlong)ui64TransId);
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportCommitTrans(
	eRestoreAction *		peAction,
	FLMUINT64				ui64TransId)
{
	RCODE				rc = NE_XFLM_OK;
	JNIEnv *			pEnv;
	jclass			Cls;
	jmethodID		MId;
	FLMBOOL			bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET(NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportCommitTrans", "(J)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId,
													 (jlong)ui64TransId);
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportAbortTrans(
	eRestoreAction *		peAction,
	FLMUINT64				ui64TransId)
{
	RCODE			rc = NE_XFLM_OK;
	JNIEnv *		pEnv;
	jclass		Cls;
	jmethodID	MId;
	FLMBOOL		bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET(NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportAbortTrans", "(J)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId,
													 (jlong)ui64TransId);

Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportBlockChainFree(
	eRestoreAction *		peAction,
	FLMUINT64				ui64TransId,
	FLMUINT64				ui64MaintDocNum,
	FLMUINT					uiStartBlkAddr,
	FLMUINT					uiEndBlkAddr,
	FLMUINT					uiCount)
{
	RCODE				rc = NE_XFLM_OK;
	JNIEnv *			pEnv;
	jclass			Cls;
	jmethodID		MId;
	FLMBOOL			bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET(NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportBlockChainFree", "(JJIII)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId,
							(jlong)ui64TransId, (jlong)ui64MaintDocNum, 
							(jint)uiStartBlkAddr, (jint)uiEndBlkAddr, (jint)uiCount);
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportIndexSuspend(
	eRestoreAction *		peAction,
	FLMUINT64				ui64TransId,
	FLMUINT					uiIndexNum)
{
	RCODE				rc = NE_XFLM_OK;
	JNIEnv *			pEnv;
	jclass			Cls;
	jmethodID		MId;
	FLMBOOL			bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportIndexSuspend", "(JI)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId,
						(jlong)ui64TransId, (jint)uiIndexNum);
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportIndexResume(
	eRestoreAction *		peAction,
	FLMUINT64				ui64TransId,
	FLMUINT					uiIndexNum)
{
	RCODE				rc = NE_XFLM_OK;
	JNIEnv *			pEnv;
	jclass			Cls;
	jmethodID		MId;
	FLMBOOL			bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportIndexResume", "(JI)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId,
						(jlong)ui64TransId, (jint)uiIndexNum);

Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportReduce(
	eRestoreAction *		peAction,
	FLMUINT64				ui64TransId,
	FLMUINT					uiCount)
{
	RCODE				rc = NE_XFLM_OK;
	JNIEnv *			pEnv;
	jclass			Cls;
	jmethodID		MId;
	FLMBOOL			bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET(NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportReduce", "(JI)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId,
						(jlong)ui64TransId, (jint)uiCount);

Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);	
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportUpgrade(
	eRestoreAction *		peAction,
	FLMUINT64				ui64TransId,
	FLMUINT					uiOldDbVersion,
	FLMUINT					uiNewDbVersion)
{
	RCODE				rc = NE_XFLM_OK;
	JNIEnv *			pEnv;
	jclass			Cls;
	jmethodID		MId;
	FLMBOOL			bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportUpgrade", "(JII)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId,
						(jlong)ui64TransId, (jint)uiOldDbVersion,
						(jint)uiNewDbVersion);
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);	
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportOpenRflFile(
	eRestoreAction *		peAction,
	FLMUINT					uiFileNum)
{
	RCODE				rc = NE_XFLM_OK;
	JNIEnv *			pEnv;
	jclass			Cls;
	jmethodID		MId;
	FLMBOOL			bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportOpenRflFile", "(I)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId, 
						(jint)uiFileNum);
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);	
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportRflRead(
	eRestoreAction *		peAction,
	FLMUINT					uiFileNum,
	FLMUINT					uiBytesRead)
{
	RCODE				rc = NE_XFLM_OK;
	JNIEnv *			pEnv;
	jclass			Cls;
	jmethodID		MId;
	FLMBOOL			bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportRflRead", "(II)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId,
						(jint)uiFileNum, (jint)uiBytesRead);
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);	
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportEnableEncryption(
	eRestoreAction *		peAction,
	FLMUINT64				ui64TransId)
{
	RCODE				rc = NE_XFLM_OK;
	JNIEnv *			pEnv;
	jclass			Cls;
	jmethodID		MId;
	FLMBOOL			bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportEnableEncryption", "(J)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId,
						(jlong)ui64TransId);
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);	
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportWrapKey(
	eRestoreAction *		peAction,
	FLMUINT64				ui64TransId)
{
	RCODE			rc = NE_XFLM_OK;
	JNIEnv *		pEnv;
	jclass		Cls;
	jmethodID	MId;
	FLMBOOL		bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportWrapKey", "(J)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId,
						(jlong)ui64TransId);
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);	
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportSetNextNodeId(
	eRestoreAction *	peAction,
	FLMUINT64			ui64TransId,
	FLMUINT				uiCollection,
	FLMUINT64			ui64NextNodeId)
{
	RCODE				rc = NE_XFLM_OK;
	JNIEnv *			pEnv;
	jclass			Cls;
	jmethodID		MId;
	FLMBOOL			bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportSetNextNodeId", "(JIJ)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId,
						(jlong)ui64TransId, (jint)uiCollection, 
						(jlong)ui64NextNodeId);
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);	
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportNodeSetMetaValue(
	eRestoreAction *	peAction,
	FLMUINT64			ui64TransId,
	FLMUINT				uiCollection,
	FLMUINT64			ui64NodeId,
	FLMUINT64			ui64MetaValue)
{
	RCODE				rc = NE_XFLM_OK;
	JNIEnv *			pEnv;
	jclass			Cls;
	jmethodID		MId;
	FLMBOOL			bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportNodeSetMetaValue", "(JIJJ)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId,
						(jlong)ui64TransId, (jint)uiCollection, (jlong)ui64NodeId,
						(jlong)ui64MetaValue);
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);	
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportNodeSetPrefixId(
	eRestoreAction *	peAction,
	FLMUINT64			ui64TransId,
	FLMUINT				uiCollection,
	FLMUINT64			ui64NodeId,
	FLMUINT				uiAttrNameId,
	FLMUINT				uiPrefixId)
{
	RCODE					rc = NE_XFLM_OK;
	JNIEnv *				pEnv;
	jclass				Cls;
	jmethodID			MId;
	FLMBOOL				bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportNodeSetPrefixId", "(JIJII)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId,
						(jlong)ui64TransId, (jint)uiCollection, (jlong)ui64NodeId,
						(jint)uiAttrNameId, (jint)uiPrefixId);
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);	
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportNodeFlagsUpdate(
	eRestoreAction *	peAction,
	FLMUINT64			ui64TransId,
	FLMUINT				uiCollection,
	FLMUINT64			ui64NodeId,
	FLMUINT				uiFlags,
	FLMBOOL				bAdd)
{
	RCODE					rc = NE_XFLM_OK;
	JNIEnv *				pEnv;
	jclass				Cls;
	jmethodID			MId;
	FLMBOOL				bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportNodeFlagsUpdate", "(JIJIZ)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId,
						(jlong)ui64TransId, (jint)uiCollection, (jlong)ui64NodeId,
						(jint)uiFlags, (jboolean)bAdd);
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);	
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportAttributeSetValue(
	eRestoreAction *	peAction,
	FLMUINT64			ui64TransId,
	FLMUINT				uiCollection,
	FLMUINT64			ui64ElementNodeId,
	FLMUINT				uiAttrNameId)
{
	RCODE					rc = NE_XFLM_OK;
	JNIEnv *				pEnv;
	jclass				Cls;
	jmethodID			MId;
	FLMBOOL				bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportAttributeSetValue", "(JIJI)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId,
						(jlong)ui64TransId, (jint)uiCollection, 
						(jlong)ui64ElementNodeId, (jint)uiAttrNameId);
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);	
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportNodeSetValue(
	eRestoreAction *	peAction,
	FLMUINT64			ui64TransId,
	FLMUINT				uiCollection,
	FLMUINT64			ui64NodeId)
{
	RCODE					rc = NE_XFLM_OK;
	JNIEnv *				pEnv;
	jclass				Cls;
	jmethodID			MId;
	FLMBOOL				bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportNodeSetValue", "(JIJ)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId,
						(jlong)ui64TransId, (jint)uiCollection, (jlong)ui64NodeId);
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);	
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportNodeUpdate(
	eRestoreAction *	peAction,
	FLMUINT64			ui64TransId,
	FLMUINT				uiCollection,
	FLMUINT64			ui64NodeId)
{
	RCODE					rc = NE_XFLM_OK;
	JNIEnv *				pEnv;
	jclass				Cls;
	jmethodID			MId;
	FLMBOOL				bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportNodeUpdate", "(JIJ)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId,
						(jlong)ui64TransId, (jint)uiCollection, (jlong)ui64NodeId);
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);	
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportInsertBefore(
	eRestoreAction *	peAction,
	FLMUINT64			ui64TransId,
	FLMUINT				uiCollection,
	FLMUINT64			ui64ParentId,
	FLMUINT64			ui64NewChildId,
	FLMUINT64			ui64RefChildId)
{
	RCODE					rc = NE_XFLM_OK;
	JNIEnv *				pEnv;
	jclass				Cls;
	jmethodID			MId;
	FLMBOOL				bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportInsertBefore", "(JIJJJ)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId,
						(jlong)ui64TransId, (jint)uiCollection, (jlong)ui64ParentId,
						(jlong)ui64NewChildId, (jlong)ui64RefChildId);
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);	
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportNodeCreate(
	eRestoreAction *	peAction,
	FLMUINT64			ui64TransId,
	FLMUINT				uiCollection,
	FLMUINT64			ui64RefNodeId,
	eDomNodeType		eNodeType,
	FLMUINT				uiNameId,
	eNodeInsertLoc		eLocation)
{
	RCODE					rc = NE_XFLM_OK;
	JNIEnv *				pEnv;
	jclass				Cls;
	jmethodID			MId;
	FLMBOOL				bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportNodeCreate", "(JIJIII)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId,
						(jlong)ui64TransId, (jint)uiCollection, (jlong)ui64RefNodeId,
						(jint)eNodeType, (jint)uiNameId, (jint)eLocation);
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);	
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportNodeChildrenDelete(
	eRestoreAction *	peAction,
	FLMUINT64			ui64TransId,
	FLMUINT				uiCollection,
	FLMUINT64			ui64NodeId,
	FLMUINT				uiNameId)
{
	RCODE					rc = NE_XFLM_OK;
	JNIEnv *				pEnv;
	jclass				Cls;
	jmethodID			MId;
	FLMBOOL				bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportNodeChildrenDelete", "(JIJI)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId,
						(jlong)ui64TransId, (jint)uiCollection, (jlong)ui64NodeId,
						(jint)uiNameId);
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);	
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportAttributeDelete(
	eRestoreAction *	peAction,
	FLMUINT64			ui64TransId,
	FLMUINT				uiCollection,
	FLMUINT64			ui64ElementId,
	FLMUINT				uiAttrNameId)
{
	RCODE					rc = NE_XFLM_OK;
	JNIEnv *				pEnv;
	jclass				Cls;
	jmethodID			MId;
	FLMBOOL				bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportAttributeDelete", "(JIJI)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId,
						(jlong)ui64TransId, (jint)uiCollection, (jlong)ui64ElementId,
						(jint)uiAttrNameId);
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);	
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportNodeDelete(
	eRestoreAction *	peAction,
	FLMUINT64			ui64TransId,
	FLMUINT				uiCollection,
	FLMUINT64			ui64NodeId)
{
	RCODE					rc = NE_XFLM_OK;
	JNIEnv *				pEnv;
	jclass				Cls;
	jmethodID			MId;
	FLMBOOL				bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportNodeDelete", "(JIJ)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId,
						(jlong)ui64TransId, (jint)uiCollection, (jlong)ui64NodeId);
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);	
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportDocumentDone(
	eRestoreAction *	peAction,
	FLMUINT64			ui64TransId,
	FLMUINT				uiCollection,
	FLMUINT64			ui64NodeId)
{
	RCODE					rc = NE_XFLM_OK;
	JNIEnv *				pEnv;
	jclass				Cls;
	jmethodID			MId;
	FLMBOOL				bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportDocumentDone", "(JIJ)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId,
						(jlong)ui64TransId, (jint)uiCollection, (jlong)ui64NodeId);

Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);	
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI JNIRestoreStatus::reportRollOverDbKey(
	eRestoreAction *	peAction,
	FLMUINT64			ui64TransId)
{
	RCODE					rc = NE_XFLM_OK;
	JNIEnv *				pEnv;
	jclass				Cls;
	jmethodID			MId;
	FLMBOOL				bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportRollOverDbKey", "(J)I");
	flmAssert( MId);
		
	*peAction = (eRestoreAction)pEnv->CallIntMethod( m_jStatus, MId,
						(jlong)ui64TransId);
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}		
	}

	return( rc);	
}
