//------------------------------------------------------------------------------
// Desc: Native C routines to support C# DbSystem class
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

FSTATIC void copyCacheUsage(
	CS_XFLM_CACHE_USAGE *	pDest,
	XFLM_CACHE_USAGE *		pSrc);

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_createDbSystem(
	IF_DbSystem **	ppDbSystem)
{
	return( FlmAllocDbSystem( ppDbSystem));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DbSystem_Release(
	IF_DbSystem *	pDbSystem)
{
	if (pDbSystem)
	{
		pDbSystem->Release();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_dbCreate(
	IF_DbSystem *	pDbSystem,
	const char *			pszDbPath,
	const char *			pszDataDir,
	const char *			pszRflDir,
	const char *			pszDictFileName,
	const char *			pszDictBuf,
	XFLM_CREATE_OPTS *	pCreateOpts,
	IF_Db **					ppDb)
{
	return( pDbSystem->dbCreate( pszDbPath, pszDataDir, pszRflDir,
								pszDictFileName, pszDictBuf,
								pCreateOpts, ppDb));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_dbOpen(
	IF_DbSystem *	pDbSystem,
	const char *			pszDbPath,
	const char *			pszDataDir,
	const char *			pszRflDir,
	const char *			pszPassword,
	FLMBOOL					bAllowLimited,
	IF_Db **					ppDb)
{
	return( pDbSystem->dbOpen( pszDbPath, pszDataDir, pszRflDir,
		pszPassword, bAllowLimited, ppDb));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_dbRemove(
	IF_DbSystem *	pDbSystem,
	const char *			pszDbPath,
	const char *			pszDataDir,
	const char *			pszRflDir,
	FLMBOOL					bRemoveRflFiles)
{
	return( pDbSystem->dbRemove( pszDbPath, pszDataDir, pszRflDir, bRemoveRflFiles));
}

// WARNING NOTE: Any changes to this enum should also be reflected in DbSystem.cs
typedef enum
{
	RESTORE_OPEN_BACKUP_SET		= 1,
	RESTORE_OPEN_RFL_FILE		= 2,
	RESTORE_OPEN_INC_FILE		= 3,
	RESTORE_READ					= 4,
	RESTORE_CLOSE					= 5,
	RESTORE_ABORT_FILE			= 6
} eRestoreClientAction;

typedef RCODE (XFLAPI * RESTORE_CLIENT)(
	FLMINT32					iAction,
	FLMUINT32				ui32FileNum,
	FLMUINT32				ui32BytesRequested,
	void *					pvBuffer,
	FLMUINT32 *				pui32BytesRead);

/****************************************************************************
Desc:
****************************************************************************/
class CS_RestoreClient : public IF_RestoreClient
{
public:

	CS_RestoreClient(
		RESTORE_CLIENT	fnRestoreClient)
	{
		m_fnRestoreClient = fnRestoreClient;
	}

	virtual ~CS_RestoreClient()
	{
	}

	RCODE XFLAPI openBackupSet( void)
	{
		return( m_fnRestoreClient( (FLMINT32)RESTORE_OPEN_BACKUP_SET, 0, 0, NULL, NULL));
	}

	RCODE XFLAPI openRflFile(
		FLMUINT	uiFileNum)
	{
		return( m_fnRestoreClient( (FLMINT32)RESTORE_OPEN_RFL_FILE, (FLMUINT32)uiFileNum, 0, NULL, NULL));
	}

	RCODE XFLAPI openIncFile(
		FLMUINT	uiFileNum)
	{
		return( m_fnRestoreClient( (FLMINT32)RESTORE_OPEN_INC_FILE, (FLMUINT32)uiFileNum, 0, NULL, NULL));
	}

	RCODE XFLAPI read(
		FLMUINT		uiBytesRequested,
		void *		pvBuffer,
		FLMUINT *	puiBytesRead)
	{
		RCODE			rc;
		FLMUINT32	ui32BytesRead;

		rc = m_fnRestoreClient( (FLMINT32)RESTORE_READ, 0, (FLMUINT32)uiBytesRequested,
					pvBuffer, &ui32BytesRead);

		if (puiBytesRead)
		{
			*puiBytesRead = (FLMUINT)ui32BytesRead;
		}
		return( rc);
	}

	
	RCODE XFLAPI close( void)
	{
		return( m_fnRestoreClient( (FLMINT32)RESTORE_CLOSE, 0, 0, NULL, NULL));
	}

	RCODE XFLAPI abortFile( void)
	{
		return( m_fnRestoreClient( (FLMINT32)RESTORE_ABORT_FILE, 0, 0, NULL, NULL));
	}

private:

	RESTORE_CLIENT	m_fnRestoreClient;
};

// WARNING NOTE: Any changes to this enum should also be reflected in DbSystem.cs
typedef enum
{
	REPORT_PROGRESS					= 1,
	REPORT_ERROR						= 2,
	REPORT_BEGIN_TRANS				= 3,
	REPORT_COMMIT_TRANS				= 4,
	REPORT_ABORT_TRANS				= 5,
	REPORT_BLOCK_CHAIN_FREE			= 6,
	REPORT_INDEX_SUSPEND				= 7,
	REPORT_INDEX_RESUME				= 8,
	REPORT_REDUCE						= 9,
	REPORT_UPGRADE						= 10,
	REPORT_OPEN_RFL_FILE				= 11,
	REPORT_RFL_READ					= 12,
	REPORT_ENABLE_ENCRYPTION		= 13,
	REPORT_WRAP_KEY					= 14,
	REPORT_SET_NEXT_NODE_ID			= 15,
	REPORT_NODE_SET_META_VALUE		= 16,
	REPORT_NODE_SET_PREFIX_ID		= 17,
	REPORT_NODE_FLAGS_UPDATE		= 18,
	REPORT_ATTRIBUTE_SET_VALUE		= 19,
	REPORT_NODE_SET_VALUE			= 20,
	REPORT_NODE_UPDATE				= 21,
	REPORT_INSERT_BEFORE				= 22,
	REPORT_NODE_CREATE				= 23,
	REPORT_NODE_CHILDREN_DELETE	= 24,
	REPORT_ATTRIBUTE_DELETE			= 25,
	REPORT_NODE_DELETE				= 26,
	REPORT_DOCUMENT_DONE				= 27,
	REPORT_ROLL_OVER_DB_KEY			= 28
} eRestoreStatusAction;

typedef RCODE (XFLAPI * RESTORE_STATUS)(
	FLMINT32					iAction,
	FLMINT32 *				piRestoreAction,
	FLMUINT64				ui64TransId,
	FLMUINT64				ui64LongNum1,
	FLMUINT64				ui64LongNum2,
	FLMUINT64				ui64LongNum3,
	FLMUINT32				ui32ShortNum1,
	FLMUINT32				ui32ShortNum2,
	FLMUINT32				ui32ShortNum3,
	FLMUINT32				ui32ShortNum4);

/****************************************************************************
Desc:
****************************************************************************/
class CS_RestoreStatus : public IF_RestoreStatus
{
public:

	CS_RestoreStatus(
		RESTORE_STATUS	fnRestoreStatus)
	{
		m_fnRestoreStatus = fnRestoreStatus;
	}

	virtual ~CS_RestoreStatus()
	{
	}

	RCODE XFLAPI reportProgress(
		eRestoreAction *		peAction,
		FLMUINT64				ui64BytesToDo,
		FLMUINT64				ui64BytesDone)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_PROGRESS, &iAction, 0,
					ui64BytesToDo, ui64BytesDone, 0,
					0, 0, 0, 0);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}

	RCODE XFLAPI reportError(
		eRestoreAction *		peAction,
		RCODE						rcErr)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_ERROR, &iAction, 0,
					0, 0, 0,
					(FLMUINT32)rcErr, 0, 0, 0);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}

	RCODE XFLAPI reportOpenRflFile(
		eRestoreAction *		peAction,
		FLMUINT					uiFileNum)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_OPEN_RFL_FILE, &iAction, 0,
					0, 0, 0,
					(FLMUINT32)uiFileNum, 0, 0, 0);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}

	RCODE XFLAPI reportRflRead(
		eRestoreAction *		peAction,
		FLMUINT					uiFileNum,
		FLMUINT					uiBytesRead)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_RFL_READ, &iAction, 0,
					0, 0, 0,
					(FLMUINT32)uiFileNum, (FLMUINT32)uiBytesRead, 0, 0);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}

	RCODE XFLAPI reportBeginTrans(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_BEGIN_TRANS, &iAction, ui64TransId,
					0, 0, 0,
					0, 0, 0, 0);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}

	RCODE XFLAPI reportCommitTrans(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_COMMIT_TRANS, &iAction, ui64TransId,
					0, 0, 0,
					0, 0, 0, 0);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}

	RCODE XFLAPI reportAbortTrans(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_ABORT_TRANS, &iAction, ui64TransId,
					0, 0, 0,
					0, 0, 0, 0);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}

	RCODE XFLAPI reportBlockChainFree(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT64				ui64MaintDocNum,
		FLMUINT					uiStartBlkAddr,
		FLMUINT					uiEndBlkAddr,
		FLMUINT					uiCount)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_BLOCK_CHAIN_FREE, &iAction, ui64TransId,
					ui64MaintDocNum, 0, 0,
					(FLMUINT32)uiStartBlkAddr, (FLMUINT32)uiEndBlkAddr, (FLMUINT32)uiCount, 0);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}

	RCODE XFLAPI reportIndexSuspend(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiIndexNum)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_INDEX_SUSPEND, &iAction, ui64TransId,
					0, 0, 0,
					(FLMUINT32)uiIndexNum, 0, 0, 0);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}

	RCODE XFLAPI reportIndexResume(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiIndexNum)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_INDEX_RESUME, &iAction, ui64TransId,
					0, 0, 0,
					(FLMUINT32)uiIndexNum, 0, 0, 0);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}

	RCODE XFLAPI reportReduce(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCount)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_REDUCE, &iAction, ui64TransId,
					0, 0, 0,
					(FLMUINT32)uiCount, 0, 0, 0);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}

	RCODE XFLAPI reportUpgrade(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiOldDbVersion,
		FLMUINT					uiNewDbVersion)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_UPGRADE, &iAction, ui64TransId,
					0, 0, 0,
					(FLMUINT32)uiOldDbVersion, (FLMUINT32)uiNewDbVersion, 0, 0);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}

	RCODE XFLAPI reportEnableEncryption(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_ENABLE_ENCRYPTION, &iAction, ui64TransId,
					0, 0, 0,
					0, 0, 0, 0);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}

	RCODE XFLAPI reportWrapKey(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_WRAP_KEY, &iAction, ui64TransId,
					0, 0, 0,
					0, 0, 0, 0);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}
		
	RCODE XFLAPI reportRollOverDbKey(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_ROLL_OVER_DB_KEY, &iAction, ui64TransId,
					0, 0, 0,
					0, 0, 0, 0);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}
		
	RCODE XFLAPI reportDocumentDone(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64DocumentId)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_DOCUMENT_DONE, &iAction, ui64TransId,
					ui64DocumentId, 0, 0,
					(FLMUINT32)uiCollection, 0, 0, 0);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}
		
	RCODE XFLAPI reportNodeDelete(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64NodeId)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_NODE_DELETE, &iAction, ui64TransId,
					ui64NodeId, 0, 0,
					(FLMUINT32)uiCollection, 0, 0, 0);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}
		
	RCODE XFLAPI reportAttributeDelete(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64ElementId,
		FLMUINT					uiAttrNameId)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_ATTRIBUTE_DELETE, &iAction, ui64TransId,
					ui64ElementId, 0, 0,
					(FLMUINT32)uiCollection, (FLMUINT32)uiAttrNameId, 0, 0);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}
		
	RCODE XFLAPI reportNodeChildrenDelete(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64ParentNodeId,
		FLMUINT					uiNameId)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_NODE_CHILDREN_DELETE, &iAction, ui64TransId,
					ui64ParentNodeId, 0, 0,
					(FLMUINT32)uiCollection, (FLMUINT32)uiNameId, 0, 0);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}
		
	RCODE XFLAPI reportNodeCreate(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64RefNodeId,
		eDomNodeType			eNodeType,
		FLMUINT					uiNameId,
		eNodeInsertLoc			eLocation)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_NODE_CREATE, &iAction, ui64TransId,
					ui64RefNodeId, 0, 0,
					(FLMUINT32)uiCollection, (FLMUINT32)eNodeType, (FLMUINT32)uiNameId, (FLMUINT32)eLocation);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}
		
	RCODE XFLAPI reportInsertBefore(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64ParentNodeId,
		FLMUINT64				ui64NewChildNodeId,
		FLMUINT64				ui64RefChildNodeId)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_INSERT_BEFORE, &iAction, ui64TransId,
					ui64ParentNodeId, ui64NewChildNodeId, ui64RefChildNodeId,
					(FLMUINT32)uiCollection, 0, 0, 0);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}
		
	RCODE XFLAPI reportNodeUpdate(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64NodeId)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_NODE_UPDATE, &iAction, ui64TransId,
					ui64NodeId, 0, 0,
					(FLMUINT32)uiCollection, 0, 0, 0);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}
		
	RCODE XFLAPI reportNodeSetValue(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64NodeId)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_NODE_SET_VALUE, &iAction, ui64TransId,
					ui64NodeId, 0, 0,
					(FLMUINT32)uiCollection, 0, 0, 0);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}
		
	RCODE XFLAPI reportAttributeSetValue(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64ElementNodeId,
		FLMUINT					uiAttrNameId)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_ATTRIBUTE_SET_VALUE, &iAction, ui64TransId,
					ui64ElementNodeId, 0, 0,
					(FLMUINT32)uiCollection, (FLMUINT32)uiAttrNameId, 0, 0);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}
		
	RCODE XFLAPI reportNodeFlagsUpdate(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64NodeId,
		FLMUINT					uiFlags,
		FLMBOOL					bAdd)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_NODE_FLAGS_UPDATE, &iAction, ui64TransId,
					ui64NodeId, 0, 0,
					(FLMUINT32)uiCollection, (FLMUINT32)uiFlags, (FLMUINT32)bAdd, 0);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}
		
	RCODE XFLAPI reportNodeSetPrefixId(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64NodeId,
		FLMUINT					uiAttrNameId,
		FLMUINT					uiPrefixId)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_NODE_SET_PREFIX_ID, &iAction, ui64TransId,
					ui64NodeId, 0, 0,
					(FLMUINT32)uiCollection, (FLMUINT32)uiAttrNameId, (FLMUINT32)uiPrefixId, 0);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}
		
	RCODE XFLAPI reportNodeSetMetaValue(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64NodeId,
		FLMUINT64				ui64MetaValue)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_NODE_SET_META_VALUE, &iAction, ui64TransId,
					ui64NodeId, ui64MetaValue, 0,
					(FLMUINT32)uiCollection, 0, 0, 0);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}
		
	RCODE XFLAPI reportSetNextNodeId(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64NextNodeId)
	{
		RCODE		rc;
		FLMINT32	iAction;

		rc = m_fnRestoreStatus( (FLMINT32)REPORT_SET_NEXT_NODE_ID, &iAction, ui64TransId,
					ui64NextNodeId, 0, 0,
					(FLMUINT32)uiCollection, 0, 0, 0);
		*peAction = (eRestoreAction)iAction;
		return( rc);
	}


private:

	RESTORE_STATUS	m_fnRestoreStatus;
};

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_dbRestore(
	IF_DbSystem *	pDbSystem,
	const char *		pszDbFileName,
	const char *		pszDataDir,
	const char *		pszRflDir,
	const char *		pszBackupPath,
	const char *		pszPassword,
	RESTORE_CLIENT		fnRestoreClient,
	RESTORE_STATUS		fnRestoreStatus)
{
	RCODE						rc = NE_XFLM_OK;
	IF_RestoreClient *	pRestoreClient = NULL;
	IF_RestoreStatus *	pRestoreStatus = NULL;

	if (fnRestoreClient)
	{
		if ((pRestoreClient = f_new CS_RestoreClient( fnRestoreClient)) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}
	}
	if (fnRestoreStatus)
	{
		if ((pRestoreStatus = f_new CS_RestoreStatus( fnRestoreStatus)) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}
	}
 
	if (RC_BAD( rc = pDbSystem->dbRestore( pszDbFileName, pszDataDir, pszRflDir, pszBackupPath,
								pszPassword, pRestoreClient, pRestoreStatus)))
	{
		goto Exit;
	}

Exit:

	if (pRestoreClient)
	{
		pRestoreClient->Release();
	}
	if (pRestoreStatus)
	{
		pRestoreStatus->Release();
	}

	return( rc);
}

typedef RCODE (XFLAPI * DB_CHECK_STATUS)(
	FLMBOOL							bHaveProgressInfo,
	XFLM_PROGRESS_CHECK_INFO *	pProgressInfo,
	XFLM_CORRUPT_INFO *			pCorruptInfo);

/****************************************************************************
Desc:
****************************************************************************/
class CS_DbCheckStatus : public IF_DbCheckStatus
{
public:

	CS_DbCheckStatus(
		DB_CHECK_STATUS	fnDbCheckStatus)
	{
		m_fnDbCheckStatus = fnDbCheckStatus;
	}

	virtual ~CS_DbCheckStatus()
	{
	}

	RCODE XFLAPI reportProgress(
		XFLM_PROGRESS_CHECK_INFO *	pProgCheck)
	{
		return( m_fnDbCheckStatus( TRUE, pProgCheck, NULL));
	}
	
	RCODE XFLAPI reportCheckErr(
		XFLM_CORRUPT_INFO *	pCorruptInfo,
		FLMBOOL *				pbFix)
	{
		if (pbFix)
		{
			*pbFix = TRUE;
		}
		return( m_fnDbCheckStatus( FALSE, NULL, pCorruptInfo));
	}

private:

	DB_CHECK_STATUS	m_fnDbCheckStatus;
};

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_dbCheck(
	IF_DbSystem *	pDbSystem,
	const char *		pszDbName,
	const char *		pszDataDir,
	const char *		pszRflDir,
	const char *		pszPassword,
	FLMUINT32			ui32Flags,
	DB_CHECK_STATUS	fnCheckStatus,
	IF_DbInfo **		ppDbInfo)
{
	RCODE						rc = NE_XFLM_OK;
	IF_DbCheckStatus *	pDbCheckStatus = NULL;

	*ppDbInfo = NULL;
	if (fnCheckStatus)
	{
		if ((pDbCheckStatus = f_new CS_DbCheckStatus( fnCheckStatus)) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}
	}
 
	if (RC_BAD( rc = pDbSystem->dbCheck( pszDbName, pszDataDir, pszRflDir, pszPassword,
								(FLMUINT)ui32Flags, ppDbInfo, pDbCheckStatus)))
	{
		goto Exit;
	}

Exit:

	if (pDbCheckStatus)
	{
		pDbCheckStatus->Release();
	}

	return( rc);
}

typedef RCODE (XFLAPI * DB_COPY_STATUS)(
	FLMUINT64		ui64BytesToCopy,
	FLMUINT64		ui64BytesCopied,
	FLMBOOL			bNewSrcFile,
	const char *	pszSrcFileName,
	const char *	pszDestFileName);

/****************************************************************************
Desc:
****************************************************************************/
class CS_DbCopyStatus : public IF_DbCopyStatus
{
public:

	CS_DbCopyStatus(
		DB_COPY_STATUS	fnCopyStatus)
	{
		m_fnCopyStatus = fnCopyStatus;
	}

	virtual ~CS_DbCopyStatus()
	{
	}
	
	RCODE XFLAPI dbCopyStatus(
		FLMUINT64		ui64BytesToCopy,
		FLMUINT64		ui64BytesCopied,
		FLMBOOL			bNewSrcFile,
		const char *	pszSrcFileName,
		const char *	pszDestFileName)
	{
		return( m_fnCopyStatus( ui64BytesToCopy, ui64BytesCopied, bNewSrcFile,
									pszSrcFileName, pszDestFileName));
	}
		
private:

	DB_COPY_STATUS	m_fnCopyStatus;

};

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_dbCopy(
	IF_DbSystem *	pDbSystem,
	const char *		pszSrcDbName,
	const char *		pszSrcDataDir,
	const char *		pszSrcRflDir,
	const char *		pszDestDbName,
	const char *		pszDestDataDir,
	const char *		pszDestRflDir,
	DB_COPY_STATUS		fnCopyStatus)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DbCopyStatus *	pDbCopyStatus = NULL;

	if (fnCopyStatus)
	{
		if ((pDbCopyStatus = f_new CS_DbCopyStatus( fnCopyStatus)) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}
	}
 
	if (RC_BAD( rc = pDbSystem->dbCopy( pszSrcDbName, pszSrcDataDir, pszSrcRflDir, pszDestDbName,
								pszDestDataDir, pszDestRflDir, pDbCopyStatus)))
	{
		goto Exit;
	}

Exit:

	if (pDbCopyStatus)
	{
		pDbCopyStatus->Release();
	}

	return( rc);
}

typedef RCODE (XFLAPI * DB_RENAME_STATUS)(
	const char *	pszSrcFileName,
	const char *	pszDestFileName);

/****************************************************************************
Desc:
****************************************************************************/
class CS_DbRenameStatus : public IF_DbRenameStatus
{
public:

	CS_DbRenameStatus(
		DB_RENAME_STATUS	fnRenameStatus)
	{
		m_fnRenameStatus = fnRenameStatus;
	}

	virtual ~CS_DbRenameStatus()
	{
	}
	
	RCODE XFLAPI dbRenameStatus(
		const char *	pszSrcFileName,
		const char *	pszDestFileName)
	{
		return( m_fnRenameStatus( pszSrcFileName, pszDestFileName));
	}
		
private:

	DB_RENAME_STATUS	m_fnRenameStatus;

};

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_dbRename(
	IF_DbSystem *	pDbSystem,
	const char *		pszSrcDbName,
	const char *		pszSrcDataDir,
	const char *		pszSrcRflDir,
	const char *		pszDestDbName,
	FLMBOOL				bOverwriteDestOk,
	DB_RENAME_STATUS	fnRenameStatus)
{
	RCODE						rc = NE_XFLM_OK;
	IF_DbRenameStatus *	pDbRenameStatus = NULL;

	if (fnRenameStatus)
	{
		if ((pDbRenameStatus = f_new CS_DbRenameStatus( fnRenameStatus)) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}
	}
 
	if (RC_BAD( rc = pDbSystem->dbRename( pszSrcDbName, pszSrcDataDir, pszSrcRflDir, pszDestDbName,
								bOverwriteDestOk, pDbRenameStatus)))
	{
		goto Exit;
	}

Exit:

	if (pDbRenameStatus)
	{
		pDbRenameStatus->Release();
	}

	return( rc);
}

typedef RCODE (XFLAPI * DB_REBUILD_STATUS)(
	FLMBOOL					bHaveRebuildInfo,
	XFLM_REBUILD_INFO *	pRebuildInfo,
	XFLM_CORRUPT_INFO *	pCorruptInfo);

/****************************************************************************
Desc:
****************************************************************************/
class CS_DbRebuildStatus : public IF_DbRebuildStatus
{
public:

	CS_DbRebuildStatus(
		DB_REBUILD_STATUS	fnDbRebuildStatus)
	{
		m_fnDbRebuildStatus = fnDbRebuildStatus;
	}

	virtual ~CS_DbRebuildStatus()
	{
	}

	RCODE XFLAPI reportRebuild(
		XFLM_REBUILD_INFO *	pRebuildInfo)
	{
		return( m_fnDbRebuildStatus( TRUE, pRebuildInfo, NULL));
	}
	
	RCODE XFLAPI reportRebuildErr(
		XFLM_CORRUPT_INFO *	pCorruptInfo)
	{
		return( m_fnDbRebuildStatus( FALSE, NULL, pCorruptInfo));
	}

private:

	DB_REBUILD_STATUS	m_fnDbRebuildStatus;
};

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_dbRebuild(
	IF_DbSystem *	pDbSystem,
	const char *			pszSourceDbPath,
	const char *			pszSourceDataDir,
	const char *			pszDestDbPath,
	const char *			pszDestDataDir,
	const char *			pszDestRflDir,
	const char *			pszDictPath,
	const char *			pszPassword,
	XFLM_CREATE_OPTS *	pCreateOpts,
	DB_REBUILD_STATUS		fnRebuildStatus)
{
	RCODE						rc = NE_XFLM_OK;
	IF_DbRebuildStatus *	pDbRebuildStatus = NULL;
	FLMUINT64				ui64TotNodes;
	FLMUINT64				ui64NodesRecov;
	FLMUINT64				ui64DiscardedDocs;

	if (fnRebuildStatus)
	{
		if ((pDbRebuildStatus = f_new CS_DbRebuildStatus( fnRebuildStatus)) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}
	}
 
	if (RC_BAD( rc = pDbSystem->dbRebuild( pszSourceDbPath, pszSourceDataDir,
								pszDestDbPath, pszDestDataDir, pszDestRflDir,
								pszDictPath, pszPassword, pCreateOpts,
								&ui64TotNodes, &ui64NodesRecov, &ui64DiscardedDocs,
								pDbRebuildStatus)))
	{
		goto Exit;
	}

Exit:

	if (pDbRebuildStatus)
	{
		pDbRebuildStatus->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_openBufferIStream(
	const char *			pszBuffer,
	F_BufferIStream **	ppIStream)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiStrCharCount = f_strlen( pszBuffer);
	F_BufferIStream *	pIStream = NULL;
	char *				pszAllocBuffer = NULL;
	
	// Create the buffer stream object.
	
	if ((pIStream = f_new F_BufferIStream) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	// Call the openStream method so that it will allocate a buffer internally.
	
	if (RC_BAD( rc = pIStream->openStream( NULL, uiStrCharCount, &pszAllocBuffer)))
	{
		goto Exit;
	}
	
	// Copy the data from the passed in string into pucBuffer, excluding the null
	// terminating byte.
	
	f_memcpy( pszAllocBuffer, pszBuffer, uiStrCharCount);
	
Exit:

	*ppIStream = pIStream;

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_openFileIStream(
	IF_DbSystem *		pDbSystem,
	const char *		pszFileName,
	IF_PosIStream **	ppIStream)
{
	return( pDbSystem->openFileIStream( pszFileName, ppIStream));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_openMultiFileIStream(
	IF_DbSystem *	pDbSystem,
	const char *	pszDirectory,
	const char *	pszBaseName,
	IF_IStream **	ppIStream)
{
	return( pDbSystem->openMultiFileIStream( pszDirectory, pszBaseName, ppIStream));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_openBufferedIStream(
	IF_DbSystem *	pDbSystem,
	IF_IStream *	pInputIStream,
	FLMUINT32		ui32BufferSize,
	IF_IStream **	ppIStream)
{
	return( pDbSystem->openBufferedIStream( pInputIStream,
							(FLMUINT)ui32BufferSize, ppIStream));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_openUncompressingIStream(
	IF_DbSystem *	pDbSystem,
	IF_IStream *	pInputIStream,
	IF_IStream **	ppIStream)
{
	return( pDbSystem->openUncompressingIStream( pInputIStream, ppIStream));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_openBase64Encoder(
	IF_DbSystem *	pDbSystem,
	IF_IStream *	pInputIStream,
	FLMBOOL			bInsertLineBreaks,
	IF_IStream **	ppIStream)
{
	return( pDbSystem->openBase64Encoder( pInputIStream, bInsertLineBreaks, ppIStream));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_openBase64Decoder(
	IF_DbSystem *	pDbSystem,
	IF_IStream *	pInputIStream,
	IF_IStream **	ppIStream)
{
	return( pDbSystem->openBase64Decoder( pInputIStream, ppIStream));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_openFileOStream(
	IF_DbSystem *	pDbSystem,
	const char *	pszFileName,
	FLMBOOL			bTruncateIfExists,
	IF_OStream **	ppOStream)
{
	return( pDbSystem->openFileOStream( pszFileName, bTruncateIfExists, ppOStream));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_openMultiFileOStream(
	IF_DbSystem *	pDbSystem,
	const char *	pszDirectory,
	const char *	pszBaseName,
	FLMUINT32		ui32MaxFileSize,
	FLMBOOL			bOkToOverwrite,
	IF_OStream **	ppOStream)
{
	return( pDbSystem->openMultiFileOStream( pszDirectory, pszBaseName,
								(FLMUINT)ui32MaxFileSize, bOkToOverwrite, ppOStream));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_removeMultiFileStream(
	IF_DbSystem *	pDbSystem,
	const char *	pszDirectory,
	const char *	pszBaseName)
{
	return( pDbSystem->removeMultiFileStream( pszDirectory, pszBaseName));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_openBufferedOStream(
	IF_DbSystem *	pDbSystem,
	IF_OStream *	pInputOStream,
	FLMUINT32		ui32BufferSize,
	IF_OStream **	ppOStream)
{
	return( pDbSystem->openBufferedOStream( pInputOStream,
								(FLMUINT)ui32BufferSize, ppOStream));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_openCompressingOStream(
	IF_DbSystem *	pDbSystem,
	IF_OStream *	pInputOStream,
	IF_OStream **	ppOStream)
{
	return( pDbSystem->openCompressingOStream( pInputOStream, ppOStream));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_writeToOStream(
	IF_DbSystem *	pDbSystem,
	IF_IStream *	pIStream,
	IF_OStream *	pOStream)
{
	return( pDbSystem->writeToOStream( pIStream, pOStream));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_createDataVector(
	IF_DbSystem *		pDbSystem,
	IF_DataVector **	ppDataVector)
{
	return( pDbSystem->createIFDataVector( ppDataVector));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DbSystem_freeUnmanagedMem(
	IF_DbSystem *	pDbSystem,
	void *			pvMem)
{
	pDbSystem->freeMem( &pvMem);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_updateIniFile(
	IF_DbSystem *	pDbSystem,
	const char *	pszParamName,
	const char *	pszValue)
{
	return( pDbSystem->updateIniFile( pszParamName, pszValue));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_dbDup(
	IF_DbSystem *	pDbSystem,
	IF_Db *			pDbToDup,
	IF_Db **			ppDupDb)
{
	return( pDbSystem->dbDup( pDbToDup, ppDupDb));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_setDynamicMemoryLimit(
	IF_DbSystem *	pDbSystem,
	FLMUINT32		ui32CacheAdjustPercent,
	FLMUINT64		ui64CacheAdjustMin,
	FLMUINT64		ui64CacheAdjustMax,
	FLMUINT64		ui64CacheAdjustMinToLeave)
{
	return( pDbSystem->setDynamicMemoryLimit( (FLMUINT)ui32CacheAdjustPercent,
			(FLMUINT)ui64CacheAdjustMin, (FLMUINT)ui64CacheAdjustMax,
			(FLMUINT)ui64CacheAdjustMinToLeave));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_setHardMemoryLimit(
	IF_DbSystem *	pDbSystem,
	FLMUINT32		ui32Percent,
	FLMBOOL			bPercentOfAvail,
	FLMUINT64		ui64Min,
	FLMUINT64		ui64Max,
	FLMUINT64		ui64MinToLeave,
	FLMBOOL			bPreallocate)
{
	return( pDbSystem->setHardMemoryLimit( (FLMUINT)ui32Percent, bPercentOfAvail,
			(FLMUINT)ui64Min, (FLMUINT)ui64Max, (FLMUINT)ui64MinToLeave, bPreallocate));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMBOOL XFLAPI xflaim_DbSystem_getDynamicCacheSupported(
	IF_DbSystem *	pDbSystem)
{
	return( pDbSystem->getDynamicCacheSupported());
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC void copyCacheUsage(
	CS_XFLM_CACHE_USAGE *	pDest,
	XFLM_CACHE_USAGE *		pSrc)
{
	pDest->ui64ByteCount = (FLMUINT64)pSrc->uiByteCount;
	pDest->ui64Count = (FLMUINT64)pSrc->uiCount;
	pDest->ui64OldVerCount = (FLMUINT64)pSrc->uiOldVerCount;
	pDest->ui64OldVerBytes = (FLMUINT64)pSrc->uiOldVerBytes;
	pDest->ui32CacheHits = (FLMUINT32)pSrc->uiCacheHits;
	pDest->ui32CacheHitLooks = (FLMUINT32)pSrc->uiCacheHitLooks;
	pDest->ui32CacheFaults = (FLMUINT32)pSrc->uiCacheFaults;
	pDest->ui32CacheFaultLooks = (FLMUINT32)pSrc->uiCacheFaultLooks;
	f_memcpy( &pDest->slabUsage, &pSrc->slabUsage, sizeof( FLM_SLAB_USAGE));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DbSystem_getCacheInfo(
	IF_DbSystem *			pDbSystem,
	CS_XFLM_CACHE_INFO *	pCacheInfo)
{
	XFLM_CACHE_INFO	cacheInfo;

	pDbSystem->getCacheInfo( &cacheInfo);

	pCacheInfo->ui64MaxBytes = (FLMUINT64)cacheInfo.uiMaxBytes;
	pCacheInfo->ui64TotalBytesAllocated = (FLMUINT64)cacheInfo.uiTotalBytesAllocated;
	pCacheInfo->bDynamicCacheAdjust = cacheInfo.bDynamicCacheAdjust;
	pCacheInfo->ui32CacheAdjustPercent = (FLMUINT32)cacheInfo.uiCacheAdjustPercent;
	pCacheInfo->ui64CacheAdjustMin = (FLMUINT64)cacheInfo.uiCacheAdjustMin;
	pCacheInfo->ui64CacheAdjustMax = (FLMUINT64)cacheInfo.uiCacheAdjustMax;
	pCacheInfo->ui64CacheAdjustMinToLeave = (FLMUINT64)cacheInfo.uiCacheAdjustMinToLeave;
	pCacheInfo->ui64DirtyCount = (FLMUINT64)cacheInfo.uiDirtyCount;
	pCacheInfo->ui64DirtyBytes = (FLMUINT64)cacheInfo.uiDirtyBytes;
	pCacheInfo->ui64NewCount = (FLMUINT64)cacheInfo.uiNewCount;
	pCacheInfo->ui64NewBytes = (FLMUINT64)cacheInfo.uiNewBytes;
	pCacheInfo->ui64LogCount = (FLMUINT64)cacheInfo.uiLogCount;
	pCacheInfo->ui64LogBytes = (FLMUINT64)cacheInfo.uiLogBytes;
	pCacheInfo->ui64FreeCount = (FLMUINT64)cacheInfo.uiFreeCount;
	pCacheInfo->ui64FreeBytes = (FLMUINT64)cacheInfo.uiFreeBytes;
	pCacheInfo->ui64ReplaceableCount = (FLMUINT64)cacheInfo.uiReplaceableCount;
	pCacheInfo->ui64ReplaceableBytes = (FLMUINT64)cacheInfo.uiReplaceableBytes;
	copyCacheUsage( &pCacheInfo->BlockCache, &cacheInfo.BlockCache);
	copyCacheUsage( &pCacheInfo->NodeCache, &cacheInfo.NodeCache);
	pCacheInfo->bPreallocatedCache = cacheInfo.bPreallocatedCache;
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_closeUnusedFiles(
	IF_DbSystem *	pDbSystem,
	FLMUINT32		ui32Seconds)
{
	return( pDbSystem->closeUnusedFiles( (FLMUINT)ui32Seconds));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DbSystem_startStats(
	IF_DbSystem *	pDbSystem)
{
	pDbSystem->startStats();
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DbSystem_stopStats(
	IF_DbSystem *	pDbSystem)
{
	pDbSystem->stopStats();
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DbSystem_resetStats(
	IF_DbSystem *	pDbSystem)
{
	pDbSystem->resetStats();
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_getStats(
	IF_DbSystem *	pDbSystem,
	XFLM_STATS **	ppStats)
{
	RCODE				rc = NE_XFLM_OK;
	XFLM_STATS *	pStats = NULL;

	if (RC_BAD( rc = f_calloc( sizeof( XFLM_STATS), &pStats)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pDbSystem->getStats( pStats)))
	{
		f_free( &pStats);
		goto Exit;
	}

Exit:

	*ppStats = pStats;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_setTempDir(
	IF_DbSystem *	pDbSystem,
	const char *	pszTempDir)
{
	return( pDbSystem->setTempDir( pszTempDir));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_getTempDir(
	IF_DbSystem *	pDbSystem,
	char **			ppszTempDir)
{
	RCODE				rc = NE_XFLM_OK;
	char				szPath [F_PATH_MAX_SIZE];
	FLMUINT			uiLen;
	
	if (RC_BAD( rc = pDbSystem->getTempDir( szPath)))
	{
		goto Exit;
	}
	uiLen = f_strlen( szPath) + 1;
	if (RC_BAD( rc = f_alloc( uiLen, ppszTempDir)))
	{
		goto Exit;
	}
	f_memcpy( *ppszTempDir, szPath, uiLen);

Exit:
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DbSystem_setCheckpointInterval(
	IF_DbSystem *	pDbSystem,
	FLMUINT32		ui32Seconds)
{
	pDbSystem->setCheckpointInterval( (FLMUINT)ui32Seconds);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMUINT32 XFLAPI xflaim_DbSystem_getCheckpointInterval(
	IF_DbSystem *	pDbSystem)
{
	return( (FLMUINT32)pDbSystem->getCheckpointInterval());
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DbSystem_setCacheAdjustInterval(
	IF_DbSystem *	pDbSystem,
	FLMUINT32		ui32Seconds)
{
	pDbSystem->setCacheAdjustInterval( (FLMUINT)ui32Seconds);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMUINT32 XFLAPI xflaim_DbSystem_getCacheAdjustInterval(
	IF_DbSystem *	pDbSystem)
{
	return( (FLMUINT32)pDbSystem->getCacheAdjustInterval());
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DbSystem_setCacheCleanupInterval(
	IF_DbSystem *	pDbSystem,
	FLMUINT32		ui32Seconds)
{
	pDbSystem->setCacheCleanupInterval( (FLMUINT)ui32Seconds);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMUINT32 XFLAPI xflaim_DbSystem_getCacheCleanupInterval(
	IF_DbSystem *	pDbSystem)
{
	return( (FLMUINT32)pDbSystem->getCacheCleanupInterval());
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DbSystem_setUnusedCleanupInterval(
	IF_DbSystem *	pDbSystem,
	FLMUINT32		ui32Seconds)
{
	pDbSystem->setUnusedCleanupInterval( (FLMUINT)ui32Seconds);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMUINT32 XFLAPI xflaim_DbSystem_getUnusedCleanupInterval(
	IF_DbSystem *	pDbSystem)
{
	return( (FLMUINT32)pDbSystem->getUnusedCleanupInterval());
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DbSystem_setMaxUnusedTime(
	IF_DbSystem *	pDbSystem,
	FLMUINT32		ui32Seconds)
{
	pDbSystem->setMaxUnusedTime( (FLMUINT)ui32Seconds);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMUINT32 XFLAPI xflaim_DbSystem_getMaxUnusedTime(
	IF_DbSystem *	pDbSystem)
{
	return( (FLMUINT32)pDbSystem->getMaxUnusedTime());
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DbSystem_deactivateOpenDb(
	IF_DbSystem *	pDbSystem,
	const char *	pszDbFileName,
	const char *	pszDataDir)
{
	pDbSystem->deactivateOpenDb( pszDbFileName, pszDataDir);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DbSystem_setQuerySaveMax(
	IF_DbSystem *	pDbSystem,
	FLMUINT32		ui32MaxToSave)
{
	pDbSystem->setQuerySaveMax( (FLMUINT)ui32MaxToSave);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMUINT32 XFLAPI xflaim_DbSystem_getQuerySaveMax(
	IF_DbSystem *	pDbSystem)
{
	return( (FLMUINT32)pDbSystem->getQuerySaveMax());
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DbSystem_setDirtyCacheLimits(
	IF_DbSystem *	pDbSystem,
	FLMUINT64		ui64MaxDirty,
	FLMUINT64		ui64LowDirty)
{
	pDbSystem->setDirtyCacheLimits( (FLMUINT)ui64MaxDirty, (FLMUINT)ui64LowDirty);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DbSystem_getDirtyCacheLimits(
	IF_DbSystem *	pDbSystem,
	FLMUINT64 *		pui64MaxDirty,
	FLMUINT64 *		pui64LowDirty)
{
	FLMUINT			uiMaxDirty;
	FLMUINT			uiLowDirty;

	pDbSystem->getDirtyCacheLimits( &uiMaxDirty, &uiLowDirty);
	*pui64MaxDirty = (FLMUINT64)uiMaxDirty;
	*pui64LowDirty = (FLMUINT64)uiLowDirty;
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_compareStrings(
	IF_DbSystem *	pDbSystem,
	const FLMUNICODE *	puzLeftString,
	FLMBOOL					bLeftWild,
	const FLMUNICODE *	puzRightString,
	FLMBOOL					bRightWild,
	FLMUINT32				ui32CompareRules,
	FLMUINT32				ui32Language,
	FLMINT32 *				pi32Result)
{
	RCODE				rc = NE_XFLM_OK;
	FLMINT			iResult;

	if (RC_BAD( rc = pDbSystem->compareUnicodeStrings(
							puzLeftString, f_unilen( puzLeftString) * 2, bLeftWild,
							puzRightString, f_unilen( puzRightString) * 2, bRightWild,
							(FLMUINT)ui32CompareRules, (FLMUINT)ui32Language,
							&iResult)))
	{
		goto Exit;
	}
	*pi32Result = (FLMINT32)iResult;

Exit:
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMBOOL XFLAPI xflaim_DbSystem_uniIsUpper(
	IF_DbSystem *	pDbSystem,
	FLMUNICODE		uzChar)
{
	return( pDbSystem->uniIsUpper( uzChar));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMBOOL XFLAPI xflaim_DbSystem_uniIsLower(
	IF_DbSystem *	pDbSystem,
	FLMUNICODE		uzChar)
{
	return( pDbSystem->uniIsLower( uzChar));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMBOOL XFLAPI xflaim_DbSystem_uniIsAlpha(
	IF_DbSystem *	pDbSystem,
	FLMUNICODE		uzChar)
{
	return( pDbSystem->uniIsAlpha( uzChar));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMBOOL XFLAPI xflaim_DbSystem_uniIsDecimalDigit(
	IF_DbSystem *	pDbSystem,
	FLMUNICODE		uzChar)
{
	return( pDbSystem->uniIsDecimalDigit( uzChar));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMUNICODE XFLAPI xflaim_DbSystem_uniToLower(
	IF_DbSystem *	pDbSystem,
	FLMUNICODE		uzChar)
{
	return( pDbSystem->uniToLower( uzChar));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_waitToClose(
	IF_DbSystem *	pDbSystem,
	const char *	pszDbFileName)
{
	return( pDbSystem->waitToClose( pszDbFileName));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystem_clearCache(
	IF_DbSystem *	pDbSystem,
	IF_Db *			pDb)
{
	return( pDbSystem->clearCache( pDb));
}
