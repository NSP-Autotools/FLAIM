//------------------------------------------------------------------------------
// Desc: Native C routines to support C# Db class
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

// IMPORTANT NOTE: This needs to be kept in sync with the
// corresponding definition in xflaim.h and C#.  In xflaim.h, we need
// to use the XFLM_IMPORT_STATS structure.
typedef struct
{
	FLMUINT32	ui32Lines;
	FLMUINT32	ui32Chars;
	FLMUINT32	ui32Attributes;
	FLMUINT32	ui32Elements;
	FLMUINT32	ui32Text;
	FLMUINT32	ui32Documents;
	FLMUINT32	ui32ErrLineNum;
	FLMUINT32	ui32ErrLineOffset;
	FLMUINT32	ui32ErrorType;
	FLMUINT32	ui32ErrLineFilePos;
	FLMUINT32	ui32ErrLineBytes;
	FLMUINT32	ui32XMLEncoding;
} CS_XFLM_IMPORT_STATS;

FSTATIC RCODE CS_getDictName(
	IF_Db *			pDb,
	FLMUINT			uiDictType,
	FLMUINT			uiDictNumber,
	FLMBOOL			bGetNamespace,
	FLMUNICODE **	ppuzName);

FSTATIC void CS_copyImportStats(
	CS_XFLM_IMPORT_STATS *	pDestStats,
	XFLM_IMPORT_STATS *		pSrcStats);

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_Db_Release(
	IF_Db *	pDb)
{
	if (pDb)
	{
		pDb->Release();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_transBegin(
	IF_Db *		pDb,
	FLMUINT32	ui32TransType,
	FLMUINT32	ui32MaxLockWait,
	FLMUINT32	ui32Flags)
{
	return( pDb->transBegin( (eDbTransType)ui32TransType, 
		(FLMUINT)ui32MaxLockWait, (FLMUINT)ui32Flags));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_transBeginClone(
	IF_Db *	pDb,
	IF_Db *	pDbToClone)
{
	return( pDb->transBegin( pDbToClone));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_transCommit(
	IF_Db *	pDb)
{
	return( pDb->transCommit());
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_transAbort(
	IF_Db *	pDb)
{
	return( pDb->transAbort());
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMUINT32 XFLAPI xflaim_Db_getTransType(
	IF_Db *	pDb)
{
	return( (FLMUINT32)pDb->getTransType());
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_doCheckpoint(
	IF_Db *		pDb,
	FLMUINT32	ui32Timeout)
{
	return( pDb->doCheckpoint( ui32Timeout));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_dbLock(
	IF_Db *		pDb,
	FLMUINT32	ui32LockType,
	FLMINT32		i32Priority,
	FLMUINT32	ui32Timeout)
{
	return( pDb->dbLock( (eLockType)ui32LockType, (FLMINT)i32Priority,
						(FLMUINT)ui32Timeout));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_dbUnlock(
	IF_Db *		pDb)
{
	return( pDb->dbUnlock());
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getLockType(
	IF_Db *		pDb,
	FLMUINT32 *	pui32LockType,
	FLMBOOL *	pbImplicitLock)
{
	RCODE			rc;
	eLockType	eLckType;
	
	rc = pDb->getLockType( &eLckType, pbImplicitLock);
	*pui32LockType = (FLMUINT32)eLckType;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getLockInfo(
	IF_Db *			pDb,
	FLMINT32			i32Priority,
	FLMUINT32 *		pui32LockType,
	FLMUINT32 *		pui32ThreadId,
	FLMUINT32 *		pui32NumExclQueued,
	FLMUINT32 *		pui32NumSharedQueued,
	FLMUINT32 *		pui32PriorityCount)
{
	RCODE			rc;
	eLockType	lockType;
	FLMUINT		uiThreadId;
	FLMUINT		uiNumExclQueued;
	FLMUINT		uiNumSharedQueued;
	FLMUINT		uiPriorityCount;
	
	if (RC_BAD( rc = pDb->getLockInfo( (FLMINT)i32Priority, &lockType,
										&uiThreadId, &uiNumExclQueued,
										&uiNumSharedQueued, &uiPriorityCount)))
	{
		goto Exit;
	}

	*pui32LockType = (FLMUINT32)lockType;
	*pui32ThreadId = (FLMUINT32)uiThreadId;
	*pui32NumExclQueued = (FLMUINT32)uiNumExclQueued;
	*pui32NumSharedQueued = (FLMUINT32)uiNumSharedQueued;
	*pui32PriorityCount = (FLMUINT32)uiPriorityCount;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_indexSuspend(
	IF_Db *		pDb,
	FLMUINT32	ui32Index)
{
	return( pDb->indexSuspend( (FLMUINT)ui32Index));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_indexResume(
	IF_Db *		pDb,
	FLMUINT32	ui32Index)
{
	return( pDb->indexResume( (FLMUINT)ui32Index));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_indexGetNext(
	IF_Db *		pDb,
	FLMUINT32 *	pui32Index)
{
	RCODE		rc;
	FLMUINT	uiIndex = (FLMUINT)(*pui32Index);

	rc = pDb->indexGetNext( &uiIndex);
	*pui32Index = (FLMUINT32)uiIndex;
	return( rc);
}

// IMPORTANT NOTE: This structure needs to be kept in sync with the
// XFLM_INDEX_STATUS structure in xflaim.h, as well as the corresponding
// structure in C#.
typedef struct
{
	FLMUINT64	ui64LastDocumentIndexed;
	FLMUINT64	ui64KeysProcessed;
	FLMUINT64	ui64DocumentsProcessed;
	FLMUINT64	ui64Transactions;
	FLMUINT32	ui32IndexNum;
	FLMUINT32	ui32StartTime;
	FLMUINT32	ui32State;			// This is the only member with a different
											// type than is found in XFLM_INDEX_STATUS.
											// It needs to correspond to the C# type.
} CS_XFLM_INDEX_STATUS;

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_indexStatus(
	IF_Db *						pDb,
	FLMUINT32					ui32Index,
	CS_XFLM_INDEX_STATUS *	pIndexStatus)
{
	RCODE					rc;
	XFLM_INDEX_STATUS	indexStatus;

	if (RC_BAD( rc = pDb->indexStatus( (FLMUINT)ui32Index, &indexStatus)))
	{
		goto Exit;
	}

	pIndexStatus->ui64LastDocumentIndexed = indexStatus.ui64LastDocumentIndexed;
	pIndexStatus->ui64KeysProcessed = indexStatus.ui64KeysProcessed;
	pIndexStatus->ui64DocumentsProcessed = indexStatus.ui64DocumentsProcessed;
	pIndexStatus->ui64Transactions = indexStatus.ui64Transactions;
	pIndexStatus->ui32IndexNum = indexStatus.ui32IndexNum;
	pIndexStatus->ui32StartTime = indexStatus.ui32StartTime;
	pIndexStatus->ui32State = (FLMUINT32)indexStatus.eState;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_reduceSize(
	IF_Db *		pDb,
	FLMUINT32	ui32Count,
	FLMUINT32 *	pui32NumReduced)
{
	RCODE		rc;
	FLMUINT	uiCount = 0;
	
	rc = pDb->reduceSize( (FLMUINT)ui32Count, &uiCount);
	*pui32NumReduced = (FLMUINT32)uiCount;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_keyRetrieve(
	IF_Db *				pDb,
	FLMUINT32			ui32Index,
	IF_DataVector *	pSearchKey,
	FLMUINT32			ui32SearchFlags,
	IF_DataVector *	pFoundKey)
{
	return( pDb->keyRetrieve( (FLMUINT)ui32Index,
							pSearchKey, (FLMUINT)ui32SearchFlags, pFoundKey));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_createDocument(
	IF_Db *				pDb,
	FLMUINT32			ui32Collection,
	IF_DOMNode **		ppNode)
{
	return( pDb->createDocument( (FLMUINT)ui32Collection, ppNode));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_createRootElement(
	IF_Db *				pDb,
	FLMUINT32			ui32Collection,
	FLMUINT32			ui32ElementNameId,
	IF_DOMNode **		ppNode)
{
	return( pDb->createRootElement( (FLMUINT)ui32Collection,
								(FLMUINT)ui32ElementNameId, ppNode));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getFirstDocument(
	IF_Db *				pDb,
	FLMUINT32			ui32Collection,
	IF_DOMNode **		ppNode)
{
	return( pDb->getFirstDocument( (FLMUINT)ui32Collection, ppNode));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getLastDocument(
	IF_Db *				pDb,
	FLMUINT32			ui32Collection,
	IF_DOMNode **		ppNode)
{
	return( pDb->getLastDocument( (FLMUINT)ui32Collection, ppNode));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getDocument(
	IF_Db *				pDb,
	FLMUINT32			ui32Collection,
	FLMUINT32			ui32RetrieveFlags,
	FLMUINT64			ui64DocumentId,
	IF_DOMNode **		ppNode)
{
	return( pDb->getDocument( (FLMUINT)ui32Collection, (FLMUINT)ui32RetrieveFlags,
									ui64DocumentId, ppNode));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_documentDone(
	IF_Db *				pDb,
	FLMUINT32			ui32Collection,
	FLMUINT64			ui64DocumentId)
{
	return( pDb->documentDone( (FLMUINT)ui32Collection, ui64DocumentId));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_documentDone2(
	IF_Db *			pDb,
	IF_DOMNode *	pDocument)
{
	return( pDb->documentDone( pDocument));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_createElementDef(
	IF_Db *					pDb,
	const FLMUNICODE *	puzNamespaceURI,
	const FLMUNICODE *	puzElementName,
	FLMUINT32				ui32DataType,
	FLMUINT32 *				pui32NameId)
{
	RCODE		rc;
	FLMUINT	uiNameId = (FLMUINT)(*pui32NameId);

	rc = pDb->createElementDef( puzNamespaceURI, puzElementName,
					(FLMUINT)ui32DataType, &uiNameId, NULL);
	*pui32NameId = (FLMUINT32)uiNameId;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_createUniqueElmDef(
	IF_Db *					pDb,
	const FLMUNICODE *	puzNamespaceURI,
	const FLMUNICODE *	puzElementName,
	FLMUINT32 *				pui32NameId)
{
	RCODE		rc;
	FLMUINT	uiNameId = (FLMUINT)(*pui32NameId);

	rc = pDb->createUniqueElmDef( puzNamespaceURI, puzElementName,
					&uiNameId, NULL);
	*pui32NameId = (FLMUINT32)uiNameId;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getElementNameId(
	IF_Db *					pDb,
	const FLMUNICODE *	puzNamespaceURI,
	const FLMUNICODE *	puzElementName,
	FLMUINT32 *				pui32NameId)
{
	RCODE		rc;
	FLMUINT	uiNameId;

	rc = pDb->getElementNameId( puzNamespaceURI, puzElementName, &uiNameId);
	*pui32NameId = (FLMUINT32)uiNameId;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_createAttributeDef(
	IF_Db *					pDb,
	const FLMUNICODE *	puzNamespaceURI,
	const FLMUNICODE *	puzAttributeName,
	FLMUINT32				ui32DataType,
	FLMUINT32 *				pui32NameId)
{
	RCODE		rc;
	FLMUINT	uiNameId = (FLMUINT)(*pui32NameId);

	rc = pDb->createAttributeDef( puzNamespaceURI, puzAttributeName,
					(FLMUINT)ui32DataType, &uiNameId, NULL);
	*pui32NameId = (FLMUINT32)uiNameId;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getAttributeNameId(
	IF_Db *					pDb,
	const FLMUNICODE *	puzNamespaceURI,
	const FLMUNICODE *	puzAttributeName,
	FLMUINT32 *				pui32NameId)
{
	RCODE		rc;
	FLMUINT	uiNameId;

	rc = pDb->getAttributeNameId( puzNamespaceURI, puzAttributeName, &uiNameId);
	*pui32NameId = (FLMUINT32)uiNameId;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_createPrefixDef(
	IF_Db *					pDb,
	const FLMUNICODE *	puzPrefixName,
	FLMUINT32 *				pui32NameId)
{
	RCODE		rc;
	FLMUINT	uiNameId = (FLMUINT)(*pui32NameId);

	rc = pDb->createPrefixDef( puzPrefixName, &uiNameId);
	*pui32NameId = (FLMUINT32)uiNameId;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getPrefixId(
	IF_Db *					pDb,
	const FLMUNICODE *	puzPrefixName,
	FLMUINT32 *				pui32NameId)
{
	RCODE		rc;
	FLMUINT	uiNameId;

	rc = pDb->getPrefixId( puzPrefixName, &uiNameId);
	*pui32NameId = (FLMUINT32)uiNameId;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_createEncDef(
	IF_Db *					pDb,
	const FLMUNICODE *	puzEncName,
	const FLMUNICODE *	puzEncType,
	FLMUINT32				ui32KeySize,
	FLMUINT32 *				pui32NameId)
{
	RCODE		rc;
	FLMUINT	uiNameId = (FLMUINT)(*pui32NameId);

	rc = pDb->createEncDef( puzEncType, puzEncName, (FLMUINT)ui32KeySize, &uiNameId);
	*pui32NameId = (FLMUINT32)uiNameId;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getEncDefId(
	IF_Db *					pDb,
	const FLMUNICODE *	puzEncName,
	FLMUINT32 *				pui32NameId)
{
	RCODE		rc;
	FLMUINT	uiNameId;

	rc = pDb->getEncDefId( puzEncName, &uiNameId);
	*pui32NameId = (FLMUINT32)uiNameId;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_createCollectionDef(
	IF_Db *					pDb,
	const FLMUNICODE *	puzCollectionName,
	FLMUINT32				ui32EncDefId,
	FLMUINT32 *				pui32NameId)
{
	RCODE		rc;
	FLMUINT	uiNameId = (FLMUINT)(*pui32NameId);

	rc = pDb->createCollectionDef( puzCollectionName, &uiNameId, (FLMUINT)ui32EncDefId);
	*pui32NameId = (FLMUINT32)uiNameId;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getCollectionNumber(
	IF_Db *					pDb,
	const FLMUNICODE *	puzCollectionName,
	FLMUINT32 *				pui32NameId)
{
	RCODE		rc;
	FLMUINT	uiNameId;

	rc = pDb->getCollectionNumber( puzCollectionName, &uiNameId);
	*pui32NameId = (FLMUINT32)uiNameId;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getIndexNumber(
	IF_Db *					pDb,
	const FLMUNICODE *	puzIndexName,
	FLMUINT32 *				pui32NameId)
{
	RCODE		rc;
	FLMUINT	uiNameId;

	rc = pDb->getIndexNumber( puzIndexName, &uiNameId);
	*pui32NameId = (FLMUINT32)uiNameId;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getDictionaryDef(
	IF_Db *			pDb,
	FLMUINT32		ui32DictType,
	FLMUINT32		ui32DictNumber,
	IF_DOMNode **	ppNode)
{
	return( pDb->getDictionaryDef( (FLMUINT)ui32DictType, (FLMUINT)ui32DictNumber,
						ppNode));
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE CS_getDictName(
	IF_Db *			pDb,
	FLMUINT			uiDictType,
	FLMUINT			uiDictNumber,
	FLMBOOL			bGetNamespace,
	FLMUNICODE **	ppuzName)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiNameSize = 0;

	*ppuzName = NULL;

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

	if (RC_BAD( rc = f_alloc( uiNameSize, ppuzName)))
	{
		goto Exit;
	}
	
	// Now get the name.
	
	if (bGetNamespace)
	{
		if (RC_BAD( rc = pDb->getDictionaryName( uiDictType, uiDictNumber,
											(FLMUNICODE *)NULL, NULL,
											*ppuzName, &uiNameSize)))
		{
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = pDb->getDictionaryName( uiDictType, uiDictNumber,
											*ppuzName, &uiNameSize,
											(FLMUNICODE *)NULL, NULL)))
		{
			goto Exit;
		}
	}
	
Exit:

	if (RC_BAD( rc) && *ppuzName)
	{
		f_free( ppuzName);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getDictionaryName(
	IF_Db *			pDb,
	FLMUINT32		ui32DictType,
	FLMUINT32		ui32DictNumber,
	FLMUNICODE **	ppuzName)
{
	return( CS_getDictName( pDb, (FLMUINT)ui32DictType, (FLMUINT)ui32DictNumber,
							FALSE, ppuzName));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getElementNamespace(
	IF_Db *			pDb,
	FLMUINT32		ui32DictNumber,
	FLMUNICODE **	ppuzName)
{
	return( CS_getDictName( pDb, (FLMUINT)ELM_ELEMENT_TAG, (FLMUINT)ui32DictNumber,
							TRUE, ppuzName));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getAttributeNamespace(
	IF_Db *			pDb,
	FLMUINT32		ui32DictNumber,
	FLMUNICODE **	ppuzName)
{
	return( CS_getDictName( pDb, (FLMUINT)ELM_ATTRIBUTE_TAG, (FLMUINT)ui32DictNumber,
							TRUE, ppuzName));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getNode(
	IF_Db *			pDb,
	FLMUINT32		ui32Collection,
	FLMUINT64		ui64NodeId,
	IF_DOMNode **	ppNode)
{
	return( pDb->getNode( (FLMUINT)ui32Collection, ui64NodeId, ppNode));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getAttribute(
	IF_Db *			pDb,
	FLMUINT32		ui32Collection,
	FLMUINT64		ui64ElementNodeId,
	FLMUINT32		ui32AttrNameId,
	IF_DOMNode **	ppNode)
{
	return( pDb->getAttribute( (FLMUINT)ui32Collection, ui64ElementNodeId,
							(FLMUINT)ui32AttrNameId, ppNode));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getDataType(
	IF_Db *		pDb,
	FLMUINT32	ui32DictType,
	FLMUINT32	ui3DictNumber,
	FLMUINT32 *	pui32DataType)
{
	RCODE		rc;
	FLMUINT	uiDataType;

	rc = pDb->getDataType( (FLMUINT)ui32DictType, (FLMUINT)ui3DictNumber, &uiDataType);
	*pui32DataType = (FLMUINT32)uiDataType;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_backupBegin(
	IF_Db *			pDb,
	FLMBOOL			bFullBackup,
	FLMBOOL			bLockDb,
	FLMUINT32		ui32MaxLockWait,
	IF_Backup **	ppBackup)
{
	return( pDb->backupBegin(
								(eDbBackupType)(bFullBackup
														? XFLM_FULL_BACKUP
														: XFLM_INCREMENTAL_BACKUP),
								(eDbTransType)(bLockDb
													? XFLM_UPDATE_TRANS
													: XFLM_READ_TRANS),
								(FLMUINT)ui32MaxLockWait, ppBackup));
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC void CS_copyImportStats(
	CS_XFLM_IMPORT_STATS *	pDestStats,
	XFLM_IMPORT_STATS *		pSrcStats)
{
	pDestStats->ui32Lines = (FLMUINT32)pSrcStats->uiLines;
	pDestStats->ui32Chars = (FLMUINT32)pSrcStats->uiChars;
	pDestStats->ui32Attributes = (FLMUINT32)pSrcStats->uiAttributes;
	pDestStats->ui32Elements = (FLMUINT32)pSrcStats->uiElements;
	pDestStats->ui32Text = (FLMUINT32)pSrcStats->uiText;
	pDestStats->ui32Documents = (FLMUINT32)pSrcStats->uiDocuments;
	pDestStats->ui32ErrLineNum = (FLMUINT32)pSrcStats->uiErrLineNum;
	pDestStats->ui32ErrLineOffset = (FLMUINT32)pSrcStats->uiErrLineOffset;
	pDestStats->ui32ErrorType = (FLMUINT32)pSrcStats->eErrorType;
	pDestStats->ui32ErrLineFilePos = (FLMUINT32)pSrcStats->uiErrLineFilePos;
	pDestStats->ui32ErrLineBytes = (FLMUINT32)pSrcStats->uiErrLineBytes;
	pDestStats->ui32XMLEncoding = (FLMUINT32)pSrcStats->eXMLEncoding;
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_importDocument(
	IF_Db *						pDb,
	IF_IStream *				pIStream,
	FLMUINT32					ui32Collection,
	IF_DOMNode **				ppDocument,
	CS_XFLM_IMPORT_STATS *	pImportStats)
{
	RCODE					rc;
	XFLM_IMPORT_STATS	importStats;
	rc = pDb->importDocument( pIStream, (FLMUINT)ui32Collection, ppDocument,
							&importStats);

	CS_copyImportStats( pImportStats, &importStats);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_importIntoDocument(
	IF_Db *						pDb,
	IF_IStream *				pIStream,
	IF_DOMNode *				pNodeToLinkTo,
	FLMUINT32					ui32InsertLocation,
	CS_XFLM_IMPORT_STATS *	pImportStats)
{
	RCODE					rc;
	XFLM_IMPORT_STATS	importStats;
	FLMUINT				uiCollection;

	if (RC_BAD( rc = pNodeToLinkTo->getCollection( pDb, &uiCollection)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pDb->import( pIStream, uiCollection,
									pNodeToLinkTo, (eNodeInsertLoc)ui32InsertLocation,
									&importStats)))
	{
		goto Exit;
	}

	CS_copyImportStats( pImportStats, &importStats);

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_changeItemState(
	IF_Db *			pDb,
	FLMUINT32		ui32DictType,
	FLMUINT32		ui32DictNumber,
	const char *	pszState)
{
	return( pDb->changeItemState( (FLMUINT)ui32DictType, (FLMUINT)ui32DictNumber,
				pszState));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getRflFileName(
	IF_Db *			pDb,
	FLMUINT32		ui32FileNum,
	FLMBOOL			bBaseOnly,
	char **			ppszFileName)
{
	RCODE		rc = NE_XFLM_OK;
	FLMUINT	uiFileNameBufSize = F_PATH_MAX_SIZE + 1;

	*ppszFileName = NULL;
	if (RC_BAD( rc = f_alloc( uiFileNameBufSize, ppszFileName)))
	{
		goto Exit;
	}
	pDb->getRflFileName( (FLMUINT)ui32FileNum, bBaseOnly, *ppszFileName,
									&uiFileNameBufSize, NULL);

Exit:

	if (RC_BAD( rc) && *ppszFileName)
	{
		f_free( ppszFileName);
	}
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_setNextNodeId(
	IF_Db *			pDb,
	FLMUINT32		ui32Collection,
	FLMUINT64		ui64NextNodeId)
{
	return( pDb->setNextNodeId( (FLMUINT)ui32Collection, ui64NextNodeId));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_setNextDictNum(
	IF_Db *			pDb,
	FLMUINT32		ui32DictType,
	FLMUINT32		ui32DictNum)
{
	return( pDb->setNextDictNum( (FLMUINT)ui32DictType, (FLMUINT)ui32DictNum));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_setRflKeepFilesFlag(
	IF_Db *		pDb,
	FLMBOOL		bKeep)
{
	return( pDb->setRflKeepFilesFlag( bKeep));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getRflKeepFlag(
	IF_Db *		pDb,
	FLMBOOL *	pbKeep)
{
	return( pDb->getRflKeepFlag( pbKeep));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_setRflDir(
	IF_Db *			pDb,
	const char *	pszRflDir)
{
	return( pDb->setRflDir( pszRflDir));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getRflDir(
	IF_Db *	pDb,
	char **	ppszRflDir)
{
	RCODE		rc = NE_XFLM_OK;

	*ppszRflDir = NULL;
	if (RC_BAD( rc = f_alloc( F_PATH_MAX_SIZE + 1, ppszRflDir)))
	{
		goto Exit;
	}
	pDb->getRflDir( *ppszRflDir);

Exit:

	if (RC_BAD( rc) && *ppszRflDir)
	{
		f_free( ppszRflDir);
	}
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getRflFileNum(
	IF_Db *		pDb,
	FLMUINT32 *	pui32RflFileNum)
{
	RCODE		rc;
	FLMUINT	uiRflFileNum = 0;

	rc = pDb->getRflFileNum( &uiRflFileNum);
	*pui32RflFileNum = (FLMUINT32)uiRflFileNum;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getHighestNotUsedRflFileNum(
	IF_Db *		pDb,
	FLMUINT32 *	pui32RflFileNum)
{
	RCODE		rc;
	FLMUINT	uiRflFileNum = 0;

	rc = pDb->getHighestNotUsedRflFileNum( &uiRflFileNum);
	*pui32RflFileNum = (FLMUINT32)uiRflFileNum;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_setRflFileSizeLimits(
	IF_Db *		pDb,
	FLMUINT32	ui32MinRflSize,
	FLMUINT32	ui32MaxRflSize)
{
	return( pDb->setRflFileSizeLimits( (FLMUINT)ui32MinRflSize, (FLMUINT)ui32MaxRflSize));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getRflFileSizeLimits(
	IF_Db *		pDb,
	FLMUINT32 *	pui32MinRflSize,
	FLMUINT32 *	pui32MaxRflSize)
{
	RCODE		rc;
	FLMUINT	uiMinRflSize = 0;
	FLMUINT	uiMaxRflSize = 0;

	rc = pDb->getRflFileSizeLimits( &uiMinRflSize, &uiMaxRflSize);
	*pui32MinRflSize = (FLMUINT32)uiMinRflSize;
	*pui32MaxRflSize = (FLMUINT32)uiMaxRflSize;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_rflRollToNextFile(
	IF_Db *		pDb)
{
	return( pDb->rflRollToNextFile());
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_setKeepAbortedTransInRflFlag(
	IF_Db *		pDb,
	FLMBOOL		bKeep)
{
	return( pDb->setKeepAbortedTransInRflFlag( bKeep));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getKeepAbortedTransInRflFlag(
	IF_Db *		pDb,
	FLMBOOL *	pbKeep)
{
	return( pDb->getKeepAbortedTransInRflFlag( pbKeep));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_setAutoTurnOffKeepRflFlag(
	IF_Db *		pDb,
	FLMBOOL		bAutoTurnOff)
{
	return( pDb->setAutoTurnOffKeepRflFlag( bAutoTurnOff));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getAutoTurnOffKeepRflFlag(
	IF_Db *		pDb,
	FLMBOOL *	pbAutoTurnOff)
{
	return( pDb->getAutoTurnOffKeepRflFlag( pbAutoTurnOff));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_Db_setFileExtendSize(
	IF_Db *		pDb,
	FLMUINT32	ui32FileExtendSize)
{
	pDb->setFileExtendSize( (FLMUINT)ui32FileExtendSize);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMUINT32 XFLAPI xflaim_Db_getFileExtendSize(
	IF_Db *		pDb)
{
	return( (FLMUINT32)pDb->getFileExtendSize());
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMUINT32 XFLAPI xflaim_Db_getDbVersion(
	IF_Db *		pDb)
{
	return( (FLMUINT32)pDb->getDbVersion());
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMUINT32 XFLAPI xflaim_Db_getBlockSize(
	IF_Db *		pDb)
{
	return( (FLMUINT32)pDb->getBlockSize());
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMUINT32 XFLAPI xflaim_Db_getDefaultLanguage(
	IF_Db *		pDb)
{
	return( (FLMUINT32)pDb->getDefaultLanguage());
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMUINT64 XFLAPI xflaim_Db_getTransID(
	IF_Db *		pDb)
{
	return( pDb->getTransID());
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getDbControlFileName(
	IF_Db *	pDb,
	char **	ppszControlFileName)
{
	RCODE		rc = NE_XFLM_OK;

	*ppszControlFileName = NULL;
	if (RC_BAD( rc = f_alloc( F_PATH_MAX_SIZE + 1, ppszControlFileName)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pDb->getDbControlFileName( *ppszControlFileName,
								F_PATH_MAX_SIZE + 1)))
	{
		goto Exit;
	}

Exit:

	if (RC_BAD( rc) && *ppszControlFileName)
	{
		f_free( ppszControlFileName);
	}
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getLastBackupTransID(
	IF_Db *		pDb,
	FLMUINT64 *	pui64LastBackupTransId)
{
	return( pDb->getLastBackupTransID( pui64LastBackupTransId));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getBlocksChangedSinceBackup(
	IF_Db *		pDb,
	FLMUINT32 *	pui32BlocksChanged)
{
	RCODE		rc;
	FLMUINT	uiBlocksChanged;

	rc = pDb->getBlocksChangedSinceBackup( &uiBlocksChanged);
	*pui32BlocksChanged = (FLMUINT32)uiBlocksChanged;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getNextIncBackupSequenceNum(
	IF_Db *		pDb,
	FLMUINT32 *	pui32NextIncBackupSequenceNum)
{
	RCODE		rc;
	FLMUINT	uiNextIncBackupSequenceNum;

	rc = pDb->getNextIncBackupSequenceNum( &uiNextIncBackupSequenceNum);
	*pui32NextIncBackupSequenceNum = (FLMUINT32)uiNextIncBackupSequenceNum;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getDiskSpaceUsage(
	IF_Db *		pDb,
	FLMUINT64 *	pui64DataSize,
	FLMUINT64 *	pui64RollbackSize,
	FLMUINT64 *	pui64RflSize)
{
	return( pDb->getDiskSpaceUsage( pui64DataSize, pui64RollbackSize, pui64RflSize));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getMustCloseRC(
	IF_Db *		pDb)
{
	return( pDb->getMustCloseRC());
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getAbortRC(
	IF_Db *		pDb)
{
	return( pDb->getAbortRC());
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_Db_setMustAbortTrans(
	IF_Db *		pDb,
	RCODE			rc)
{
	pDb->setMustAbortTrans( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_enableEncryption(
	IF_Db *		pDb)
{
	return( pDb->enableEncryption());
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_wrapKey(
	IF_Db *			pDb,
	const char *	pszPassword)
{
	return( pDb->wrapKey( pszPassword));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_rollOverDbKey(
	IF_Db *	pDb)
{
	return( pDb->rollOverDbKey());
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_Db_getSerialNumber(
	IF_Db *	pDb,
	char *	pucSerialNum)
{
	pDb->getSerialNumber( pucSerialNum);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_Db_getCheckpointInfo(
	IF_Db *						pDb,
	XFLM_CHECKPOINT_INFO *	pCheckpointInfo)
{
	pDb->getCheckpointInfo( pCheckpointInfo);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_exportXML(
	IF_Db *			pDb,
	IF_DOMNode *	pStartNode,
	const char *	pszFileName,
	FLMUINT32		ui32Format)
{
	RCODE				rc = NE_XFLM_OK;
	IF_OStream *	pOStream = NULL;

	if (RC_BAD( rc = FlmOpenFileOStream( pszFileName, TRUE, &pOStream)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->exportXML( pStartNode, pOStream, (eExportFormatType)ui32Format)))
	{
		goto Exit;
	}
	
Exit:

	if (pOStream)
	{
		pOStream->Release();
	}
	return( rc);
}

/****************************************************************************
Desc:	Output stream that writes to a dynamic buffer.
****************************************************************************/
class CS_DynaBufOStream : public IF_OStream
{
public:

	CS_DynaBufOStream(
		F_DynaBuf *	pDynaBuf)
	{
		m_pDynaBuf = pDynaBuf;
		m_pDynaBuf->truncateData( 0);
	}
	
	virtual ~CS_DynaBufOStream()
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
XFLXPC RCODE XFLAPI xflaim_Db_exportXMLToString(
	IF_Db *			pDb,
	IF_DOMNode *	pStartNode,
	FLMUINT32		ui32Format,
	char **			ppszStr)
{
	RCODE					rc = NE_XFLM_OK;
	FLMBYTE				ucBuffer [512];
	F_DynaBuf			dynaBuf( ucBuffer, sizeof( ucBuffer));
	CS_DynaBufOStream	dynaOStream( &dynaBuf);
	
	if (RC_BAD( rc = pDb->exportXML( pStartNode, &dynaOStream, (eExportFormatType)ui32Format)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = f_alloc( dynaBuf.getDataLength(), ppszStr)))
	{
		goto Exit;
	}
	f_memcpy( *ppszStr, dynaBuf.getBufferPtr(), dynaBuf.getDataLength());
	
Exit:

	return( rc);
}

typedef FLMBOOL (XFLAPI * LOCK_INFO_CLIENT)(
	FLMBOOL		bSetTotalLocks,
	FLMUINT32	ui32TotalLocks,
	FLMUINT32	ui32LockNum,
	FLMUINT32	ui32ThreadId,
	FLMUINT32	ui32Time);

/****************************************************************************
Desc:
****************************************************************************/
class CS_LockInfoClient : public IF_LockInfoClient
{
public:

	CS_LockInfoClient(
		LOCK_INFO_CLIENT	fnLockInfoClient)
	{
		m_fnLockInfoClient = fnLockInfoClient;
	}

	virtual ~CS_LockInfoClient()
	{
	}

	FLMBOOL XFLAPI setLockCount(
		FLMUINT	uiTotalLocks)
	{
		return( m_fnLockInfoClient( TRUE, (FLMUINT32)uiTotalLocks, 0, 0, 0));
	}

	FLMBOOL XFLAPI addLockInfo(
		FLMUINT	uiLockNum,
		FLMUINT	uiThreadId,
		FLMUINT	uiTime)
	{
		return( m_fnLockInfoClient( FALSE, 0, (FLMUINT32)uiLockNum, (FLMUINT32)uiThreadId,
						(FLMUINT32)uiTime));
	}

private:

	LOCK_INFO_CLIENT	m_fnLockInfoClient;
};

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Db_getLockUsers(
	IF_Db *				pDb,
	LOCK_INFO_CLIENT	fnLockInfoClient)
{
	RCODE						rc = NE_XFLM_OK;
	IF_LockInfoClient *	pLockInfoClient = NULL;

	if (fnLockInfoClient)
	{
		if ((pLockInfoClient = f_new CS_LockInfoClient( fnLockInfoClient)) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}
	}
 
	if (RC_BAD( rc = pDb->getLockWaiters( pLockInfoClient)))
	{
		goto Exit;
	}

Exit:

	if (pLockInfoClient)
	{
		pLockInfoClient->Release();
	}

	return( rc);
}
