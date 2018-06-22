//------------------------------------------------------------------------------
// Desc:	This is the header file that contains the FSIndexCursor class.
// Tabs:	3
//
// Copyright (c) 2000-2007 Novell, Inc. All Rights Reserved.
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

#ifndef FSCURSOR_H
#define FSCURSOR_H

typedef struct KeyPosition
{
	FLMBYTE	ucKey [XFLM_MAX_KEY_SIZE];
	FLMUINT	uiKeyLen;
} KEYPOS;

/*============================================================================
Desc: 	File system implementation of a cursor for an index.
============================================================================*/
class FSIndexCursor : public F_Object
{
public:

	// Constructors & Destructor

	FSIndexCursor();
	~FSIndexCursor();

	void resetCursor( void);

	RCODE resetTransaction( 
		F_Db *			pDb);

	RCODE	setupKeys(
		F_Db *			pDb,
		IXD *				pIxd,
		PATH_PRED *		pPred,
		FLMBOOL *		pbDoNodeMatch,
		FLMBOOL *		pbCanCompareOnKey,
		FLMUINT *		puiLeafBlocksBetween,
		FLMUINT *		puiTotalRefs,	
		FLMBOOL *		pbTotalsEstimated);

	RCODE currentKey(
		F_Db *			pDb,
		F_DataVector *	pKey);
	
	RCODE firstKey(
		F_Db *			pDb,
		F_DataVector *	pKey);

	RCODE lastKey(
		F_Db *			pDb,
		F_DataVector *	pKey);

	RCODE	nextKey(
		F_Db *			pDb,
		F_DataVector *	pKey,
		FLMBOOL			bSkipCurrKey);

	RCODE	prevKey(
		F_Db *			pDb,
		F_DataVector *	pKey,
		FLMBOOL			bSkipCurrKey);

private:

	RCODE allocDupCheckSet( void);

	RCODE checkIfDup(
		FLMUINT64	ui64NodeId,
		FLMBOOL *	pbDup);

	RCODE useNewDb( 
		F_Db *	pDb);

	RCODE openBTree(
		F_Db *	pDb);

	// Does this index support native absolute positioning?

	FLMBOOL isAbsolutePositionable()
	{
		return (m_pIxd->uiFlags & IXD_ABS_POS) ? TRUE : FALSE;
	}

	RCODE getKeyData(
		F_Btree *	pBTree,
		FLMUINT		uiDataLen);

	RCODE setKeyPosition(
		F_Db *			pDb,
		FLMBOOL			bGoingForward,
		FLMBOOL			bExcludeKey,
		F_DataVector *	pExtSrchKey,
		KEYPOS *			pSearchKey,
		KEYPOS *			pFoundKey,
		FLMBOOL			bGetKeyData,
		FLMUINT *		puiDataLen,
		F_Btree *		pBTree,
		FLMUINT *		puiAbsolutePos);

	FINLINE void closeBTree( void)
	{
		if (m_bTreeOpen)
		{
			m_pbTree->btClose();
			m_bTreeOpen = FALSE;
			m_pDb = NULL;
			m_eTransType = XFLM_NO_TRANS;
		}
	}

	FINLINE RCODE checkTransaction(
		F_Db *	pDb)
	{
		RCODE	rc = NE_XFLM_OK;
		if (RC_OK( rc = pDb->flushKeys()))
		{
			rc = (RCODE)((m_ui64CurrTransId != pDb->m_ui64CurrTransID ||
								m_uiBlkChangeCnt != pDb->m_uiBlkChangeCnt)
							  ? resetTransaction( pDb) 
							  : NE_XFLM_OK);
		}
		return( rc);
	}

	RCODE populateKey(
		F_DataVector *	pKey);

	RCODE checkIfKeyInRange(
		FLMBOOL	bPositionForward);

	FINLINE void getCurrKey(
		KEYPOS *	pKey
		)
	{
		f_memcpy( pKey->ucKey, m_curKey.ucKey, m_curKey.uiKeyLen);
		pKey->uiKeyLen = m_curKey.uiKeyLen;
	}

	// Database information

	FLMUINT64			m_ui64CurrTransId;
	FLMUINT				m_uiBlkChangeCnt;
	FLMUINT				m_uiIndexNum;
	LFILE	*				m_pLFile;
	IXD *					m_pIxd;
	F_Db *				m_pDb;
	eDbTransType		m_eTransType;

	// Key range information

	FLMBOOL				m_bSetup;
	KEYPOS				m_fromKey;
	KEYPOS				m_untilKey;
	
	// State information.

	FLMBOOL				m_bAtBOF;			// Before the first key.
	FLMBOOL				m_bAtEOF;			// After the last key.
	KEYPOS				m_curKey;			// Current key
	FLMBYTE *			m_pucCurKeyDataBuf;
	FLMUINT				m_uiCurKeyDataBufSize;
	FLMUINT				m_uiCurKeyDataLen;
	F_Btree *			m_pbTree;
	FLMBOOL				m_bTreeOpen;
	F_DynSearchSet *	m_pNodeIdSet;
	FLMBOOL				m_bElimDups;
	FLMBOOL				m_bMovingForward;
	IXKeyCompare		m_ixCompare;
	F_DataVector		m_fromExtKey;
	F_DataVector		m_untilExtKey;

	friend class F_Query;
};

/*============================================================================
Desc:	File system implementation of a cursor for a collection.
============================================================================*/
class FSCollectionCursor : public F_Object
{
public:

	// Constructors & Destructor

	FSCollectionCursor();
	~FSCollectionCursor();

	void resetCursor();

	RCODE resetTransaction( 
		F_Db *	pDb);

	RCODE	setupRange(
		F_Db *			pDb,
		FLMUINT			uiCollection,
		FLMBOOL			bDocumentIds,
		FLMUINT64		ui64LowNodeId,
		FLMUINT64		ui64HighNodeId,
		FLMUINT *		puiLeafBlocksBetween,
		FLMUINT *		puiTotalNodes,
		FLMBOOL *		pbTotalsEstimated);

	RCODE currentNode(
		F_Db *			pDb,
		IF_DOMNode **	ppNode,
		FLMUINT64 *		pui64NodeId);

	RCODE firstNode(
		F_Db *			pDb,
		IF_DOMNode **	ppNode,
		FLMUINT64 *		pui64NodeId);

	RCODE lastNode(
		F_Db *			pDb,
		IF_DOMNode **	ppNode,
		FLMUINT64 *		pui64NodeId);

	RCODE nextNode(
		F_Db *			pDb,
		IF_DOMNode **	ppNode,
		FLMUINT64 *		pui64NodeId);

	RCODE prevNode(
		F_Db *			pDb,
		IF_DOMNode **	ppNode,
		FLMUINT64 *		pui64NodeId);

private:

	RCODE setNodePosition(
		F_Db *			pDb,
		FLMBOOL			bGoingForward,
		FLMUINT64		ui64NodeId,
		FLMUINT64 *		pui64FoundNodeId,
		F_Btree *		pBTree);

	RCODE openBTree(
		F_Db *	pDb);

	FINLINE void closeBTree( void)
	{
		if (m_bTreeOpen)
		{
			m_pbTree->btClose();
			m_bTreeOpen = FALSE;
			m_pDb = NULL;
			m_eTransType = XFLM_NO_TRANS;
		}
	}

	FINLINE RCODE checkTransaction(
		F_Db *	pDb)
	{
		RCODE	rc = NE_XFLM_OK;

		if (pDb->m_uiDirtyNodeCount)
		{
			if (RC_BAD( rc = pDb->flushDirtyNodes()))
			{
				goto Exit;
			}
		}
		rc = (RCODE)((m_pDb != pDb ||
						  m_ui64CurrTransId != pDb->m_ui64CurrTransID ||
						  m_uiBlkChangeCnt != pDb->m_uiBlkChangeCnt)
						 ? resetTransaction( pDb) 
						 : NE_XFLM_OK);
	Exit:
		return( rc);
	}

	FINLINE RCODE populateNode(
		F_Db *			pDb,
		IF_DOMNode **	ppNode,
		FLMUINT64 *		pui64NodeId
		)
	{
		if (pui64NodeId)
		{
			*pui64NodeId = m_ui64CurNodeId;
		}
		if (ppNode)
		{
			return( pDb->getNode( m_uiCollection, m_ui64CurNodeId, ppNode));
		}
		return( NE_XFLM_OK);
	}

	RCODE checkIfNodeInRange(
		FLMBOOL	bPositionForward);

	// Database Information

	FLMUINT64			m_ui64CurrTransId;
	FLMUINT				m_uiBlkChangeCnt;
	FLMUINT				m_uiCollection;
	F_COLLECTION *		m_pCollection;
	FLMBOOL				m_bDocumentIds;
	LFILE	*				m_pLFile;
	F_Db *				m_pDb;
	eDbTransType		m_eTransType;

	// Key range information

	FLMBOOL				m_bSetup;
	FLMUINT64			m_ui64FromNodeId;
	FLMUINT64			m_ui64UntilNodeId;
	
	// State information.

	FLMBOOL				m_bAtBOF;			// Before the first key.
	FLMBOOL				m_bAtEOF;			// After the last key.
	FLMUINT64			m_ui64CurNodeId;	// Current node
	F_Btree *			m_pbTree;
	FLMBOOL				m_bTreeOpen;
};


#endif
