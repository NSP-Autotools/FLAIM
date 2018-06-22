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

/****************************************************************************
Desc:
****************************************************************************/
class JNIRestoreClient : public IF_RestoreClient
{
public:

	JNIRestoreClient(
		jobject			jClient,
		JavaVM *			pJvm)
	{
		flmAssert( jClient);
		flmAssert( pJvm);
		m_jClient = jClient;
		m_pJvm = pJvm;
	}

	RCODE XFLAPI openBackupSet( void);

	RCODE XFLAPI openRflFile(
		FLMUINT			uiFileNum);

	RCODE XFLAPI openIncFile(
		FLMUINT			uiFileNum);

	RCODE XFLAPI read(
		FLMUINT			uiLength,
		void *			pvBuffer,
		FLMUINT *		puiBytesRead);

	RCODE XFLAPI close( void);

	RCODE XFLAPI abortFile( void);
	
	FINLINE FLMINT XFLAPI getRefCount( void)
	{
		return( IF_RestoreClient::getRefCount());
	}

	virtual FINLINE FLMINT XFLAPI AddRef( void)
	{
		return( IF_RestoreClient::AddRef());
	}

	virtual FINLINE FLMINT XFLAPI Release( void)
	{
		return( IF_RestoreClient::Release());
	}

private:

	jobject		m_jClient;
	JavaVM *		m_pJvm;
};

/****************************************************************************
Desc:
****************************************************************************/
class JNIRestoreStatus : public IF_RestoreStatus
{
public:

	JNIRestoreStatus(
		jobject				jStatus,
		JavaVM *				pJvm)
	{
		flmAssert( jStatus);
		flmAssert( pJvm);
		m_jStatus = jStatus;
		m_pJvm = pJvm;
	}
	
	RCODE XFLAPI reportProgress(
		eRestoreAction *	peAction,
		FLMUINT64			ui64BytesToDo,
		FLMUINT64			ui64BytesDone);

	RCODE XFLAPI reportError(
		eRestoreAction *	peAction,
		RCODE					rcErr);

	RCODE XFLAPI reportOpenRflFile(
		eRestoreAction *	peAction,
		FLMUINT				uiFileNum);

	RCODE XFLAPI reportRflRead(
		eRestoreAction *	peAction,
		FLMUINT				uiFileNum,
		FLMUINT				uiBytesRead);

	RCODE XFLAPI reportBeginTrans(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId);

	RCODE XFLAPI reportCommitTrans(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId);

	RCODE XFLAPI reportAbortTrans(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId);

	RCODE XFLAPI reportBlockChainFree(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT64			ui64MaintDocNum,
		FLMUINT				uiStartBlkAddr,
		FLMUINT				uiEndBlkAddr,
		FLMUINT				uiCount);

	RCODE XFLAPI reportIndexSuspend(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiIndexNum);

	RCODE XFLAPI reportIndexResume(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiIndexNum);

	RCODE XFLAPI reportReduce(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCount);

	RCODE XFLAPI reportUpgrade(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiOldDbVersion,
		FLMUINT				uiNewDbVersion);

	RCODE XFLAPI reportEnableEncryption(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId);

	RCODE XFLAPI reportWrapKey(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId);
		
	RCODE XFLAPI reportRollOverDbKey(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId);
		
	RCODE XFLAPI reportDocumentDone(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId);
		
	RCODE XFLAPI reportNodeDelete(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId);
		
	RCODE XFLAPI reportAttributeDelete(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64ElementId,
		FLMUINT				uiAttrNameId);
			
	RCODE XFLAPI reportNodeChildrenDelete(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId,
		FLMUINT				uiNameId);
		
	RCODE XFLAPI reportNodeCreate(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64RefNodeId,
		eDomNodeType		eNodeType,
		FLMUINT				uiNameId,
		eNodeInsertLoc		eLocation);
		
	RCODE XFLAPI reportInsertBefore(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64ParentId,
		FLMUINT64			ui64NewChildId,
		FLMUINT64			ui64RefChildId);
		
	RCODE XFLAPI reportNodeUpdate(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId);
		
	RCODE XFLAPI reportNodeSetValue(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId);
		
	RCODE XFLAPI reportAttributeSetValue(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64ElementNodeId,
		FLMUINT				uiAttrNameId);
		
	RCODE XFLAPI reportNodeFlagsUpdate(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId,
		FLMUINT				uiFlags,
		FLMBOOL				bAdd);
		
	RCODE XFLAPI reportNodeSetPrefixId(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId,
		FLMUINT				uiAttrNameId,
		FLMUINT				uiPrefixId);
			
	RCODE XFLAPI reportNodeSetMetaValue(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId,
		FLMUINT64			ui64MetaValue);
		
	RCODE XFLAPI reportSetNextNodeId(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NextNodeId);

	FINLINE FLMINT XFLAPI getRefCount( void)
	{
		return( IF_RestoreStatus::getRefCount());
	}

	virtual FINLINE FLMINT XFLAPI AddRef( void)
	{
		return( IF_RestoreStatus::AddRef());
	}

	virtual FINLINE FLMINT XFLAPI Release( void)
	{
		return( IF_RestoreStatus::Release());
	}

private:

	jobject			m_jStatus;
	JavaVM *			m_pJvm;
};
