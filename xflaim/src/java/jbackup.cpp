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
#include "xflaim_Backup.h"

#define THIS_BACKUP() ((IF_Backup *)(FLMUINT)lThis)

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Backup__1release(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	THIS_BACKUP()->Release();	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Backup__1getBackupTransId(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	return( THIS_BACKUP()->getBackupTransId());
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Backup__1getLastBackupTransId(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	return( THIS_BACKUP()->getLastBackupTransId());
}

/****************************************************************************
Desc:
****************************************************************************/
class JNIBackupClient : public IF_BackupClient
{
public:

	JNIBackupClient(
		jobject		jClient,
		JavaVM *		pJvm)
	{
		flmAssert( jClient);
		flmAssert( pJvm);
		m_jClient = jClient;
		m_pJvm = pJvm;
	}
	
	RCODE XFLAPI WriteData(
		const void *	pvBuffer,
		FLMUINT			uiBytesToWrite);
		
	FINLINE FLMINT XFLAPI getRefCount( void)
	{
		return( IF_BackupClient::getRefCount());
	}

	virtual FINLINE FLMINT XFLAPI AddRef( void)
	{
		return( IF_BackupClient::AddRef());
	}

	virtual FINLINE FLMINT XFLAPI Release( void)
	{
		return( IF_BackupClient::Release());
	}

private:

	jobject		m_jClient;
	JavaVM *		m_pJvm;
};

/****************************************************************************
Desc:
****************************************************************************/
class JNIBackupStatus : public IF_BackupStatus
{
public:

	JNIBackupStatus(
		jobject		jStatus,
		JavaVM *		pJvm)
	{
		flmAssert(jStatus);
		flmAssert(pJvm);
		m_jStatus = jStatus;
		m_pJvm = pJvm;
	}
	
	RCODE XFLAPI backupStatus(
		FLMUINT64	ui64BytesToDo,
		FLMUINT64	ui64BytesDone);
	
	FINLINE FLMINT XFLAPI getRefCount( void)
	{
		return( IF_BackupStatus::getRefCount());
	}

	virtual FINLINE FLMINT XFLAPI AddRef( void)
	{
		return( IF_BackupStatus::AddRef());
	}

	virtual FINLINE FLMINT XFLAPI Release( void)
	{
		return( IF_BackupStatus::Release());
	}

private:

	jobject			m_jStatus;
	JavaVM *			m_pJvm;
};

/****************************************************************************
Desc:
****************************************************************************/
RCODE JNIBackupClient::WriteData(
	const void *	pvBuffer,
	FLMUINT			uiBytesToWrite)
{
	RCODE				rc = NE_XFLM_OK;
	JNIEnv *			pEnv;
	jclass			Cls;
	jmethodID		MId;
	jbyteArray		jBuff = NULL;
	void *			pvBuff;
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
	MId = pEnv->GetMethodID( Cls, "WriteData", "([B)I");
	
	flmAssert( MId);
	
	if ((jBuff = pEnv->NewByteArray( (jsize)uiBytesToWrite)) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	pvBuff = pEnv->GetPrimitiveArrayCritical(jBuff, NULL);
	f_memcpy(pvBuff, pvBuffer, uiBytesToWrite);
	pEnv->ReleasePrimitiveArrayCritical( jBuff, pvBuff, 0);
	
	if( RC_BAD( rc = (RCODE)pEnv->CallIntMethod( m_jClient, MId, jBuff)))
	{
		goto Exit;
	}
		
Exit:

	if (jBuff)
	{
		pEnv->DeleteLocalRef( jBuff);
	}

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
RCODE JNIBackupStatus::backupStatus(
	FLMUINT64		ui64BytesToDo,
	FLMUINT64		ui64BytesDone)
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
	MId = pEnv->GetMethodID( Cls, "backupStatus", "(JJ)I");
	flmAssert( MId);
		
	rc = (RCODE)pEnv->CallIntMethod( m_jStatus, MId, (jlong)ui64BytesToDo,
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
JNIEXPORT jlong JNICALL Java_xflaim_Backup__1backup(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sBackupPath,
	jstring			sPassword,
	jobject			backupClient,
	jobject			backupStatus)
{
	RCODE						rc = NE_XFLM_OK;
	IF_Backup *				pBackup = THIS_BACKUP();
	FLMUINT					uiSeqNum = 0;
	JavaVM *					pJvm;
	JNIBackupClient *		pClient = NULL;
	JNIBackupStatus *		pStatus = NULL;
	FLMBYTE					ucBackupPath [F_PATH_MAX_SIZE];
	F_DynaBuf				backupPathBuf( ucBackupPath, sizeof( ucBackupPath));
	FLMBYTE					ucPassword [100];
	F_DynaBuf				passwordBuf( ucPassword, sizeof( ucPassword));
	
	if (RC_BAD( rc = getUTF8String( pEnv, sBackupPath, &backupPathBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sPassword, &passwordBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	flmAssert( backupClient);
	
	pEnv->GetJavaVM( &pJvm);
	if( (pClient = f_new JNIBackupClient( backupClient, pJvm)) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (backupStatus)
	{
		if( (pStatus = f_new JNIBackupStatus( backupStatus, pJvm)) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			ThrowError( rc, pEnv);
			goto Exit;
		}
	}
	
	if (RC_BAD( rc = pBackup->backup(
							(const char *)(backupPathBuf.getDataLength() > 1
											   ? (const char *)backupPathBuf.getBufferPtr()
												: (const char *)NULL),
							(const char *)(passwordBuf.getDataLength() > 1
											   ? (const char *)passwordBuf.getBufferPtr()
												: (const char *)NULL),
							pClient, pStatus, &uiSeqNum)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	if (pClient)
	{
		pClient->Release();
	}
	
	if (pStatus)
	{
		pStatus->Release();
	}
	
	return( uiSeqNum);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Backup__1endBackup(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Backup *		pThisBackup = THIS_BACKUP();

	if (RC_BAD( rc = pThisBackup->endBackup()))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;	
}

