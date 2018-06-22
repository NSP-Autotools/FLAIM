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
	FLMBYTE	ucKey [SFLM_MAX_KEY_SIZE];
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
	virtual ~FSIndexCursor();

	void resetCursor( void);

	RCODE resetTransaction( 
		F_Db *			pDb);

	RCODE calculateCost( void);
	
	RCODE	setupKeys(
		F_Db *			pDb,
		F_INDEX *		pIndex,
		F_TABLE *		pTable,
		SQL_PRED **		ppKeyComponents);
		
	RCODE unionKeys(
		F_Db *				pDb,
		FSIndexCursor *	pFSIndexCursor,
		FLMBOOL *			pbUnioned,
		FLMINT *				piCompare);

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

	FINLINE FLMUINT64 getCost( void)
	{
		return( m_ui64Cost);
	}

	FINLINE FLMBOOL doRowMatch( void)
	{
		return( m_bDoRowMatch);
	}
	
	FINLINE FLMBOOL canCompareOnKey( void)
	{
		return( m_bCanCompareOnKey);
	}
	
private:

	RCODE useNewDb( 
		F_Db *	pDb);

	RCODE openBTree(
		F_Db *	pDb);

	// Does this index support native absolute positioning?

	FLMBOOL isAbsolutePositionable()
	{
		return (m_pIndex->uiFlags & IXD_ABS_POS) ? TRUE : FALSE;
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
			m_eTransType = SFLM_NO_TRANS;
		}
	}

	FINLINE RCODE checkTransaction(
		F_Db *	pDb)
	{
		RCODE	rc = NE_SFLM_OK;
		if (RC_OK( rc = pDb->flushKeys()))
		{
			rc = (RCODE)((m_ui64CurrTransId != pDb->m_ui64CurrTransID ||
								m_uiBlkChangeCnt != pDb->m_uiBlkChangeCnt)
							  ? resetTransaction( pDb) 
							  : NE_SFLM_OK);
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
	F_INDEX *			m_pIndex;
	F_Db *				m_pDb;
	eDbTransType		m_eTransType;

	// Key range information

	FLMBOOL				m_bSetup;
	KEYPOS				m_fromKey;
	KEYPOS				m_untilKey;
	FLMUINT64			m_ui64Cost;
	FLMUINT64			m_ui64LeafBlocksBetween;
	FLMUINT64			m_ui64TotalRefs;
	FLMBOOL				m_bTotalsEstimated;
	FLMBOOL				m_bDoRowMatch;
	FLMBOOL				m_bCanCompareOnKey;
	
	// State information.

	FLMBOOL				m_bAtBOF;			// Before the first key.
	FLMBOOL				m_bAtEOF;			// After the last key.
	KEYPOS				m_curKey;			// Current key
	FLMBYTE *			m_pucCurKeyDataBuf;
	FLMUINT				m_uiCurKeyDataBufSize;
	FLMUINT				m_uiCurKeyDataLen;
	F_Btree *			m_pbTree;
	FLMBOOL				m_bTreeOpen;
	IXKeyCompare		m_ixCompare;
	F_DataVector		m_fromExtKey;
	F_DataVector		m_untilExtKey;

	friend class F_Query;
};

/*============================================================================
Desc:	File system implementation of a cursor for a collection.
============================================================================*/
class FSTableCursor : public F_Object
{
public:

	// Constructors & Destructor

	FSTableCursor();
	virtual ~FSTableCursor();

	void resetCursor();

	RCODE resetTransaction( 
		F_Db *	pDb);

	RCODE	setupRange(
		F_Db *			pDb,
		FLMUINT			uiTableNum,
		FLMUINT64		ui64LowRowId,
		FLMUINT64		ui64HighRowId,
		FLMBOOL			bEstimateCost);

	RCODE currentRow(
		F_Db *				pDb,
		F_Row **				ppRow,
		FLMUINT64 *			pui64RowId);

	RCODE firstRow(
		F_Db *				pDb,
		F_Row **				ppRow,
		FLMUINT64 *			pui64RowId);

	RCODE lastRow(
		F_Db *				pDb,
		F_Row **				ppRow,
		FLMUINT64 *			pui64RowId);

	RCODE nextRow(
		F_Db *				pDb,
		F_Row **				ppRow,
		FLMUINT64 *			pui64RowId);

	RCODE prevRow(
		F_Db *				pDb,
		F_Row **				ppRow,
		FLMUINT64 *			pui64RowId);
		
	FINLINE FLMUINT64 getCost( void)
	{
		return( m_ui64Cost);
	}

private:

	RCODE setRowPosition(
		F_Db *			pDb,
		FLMBOOL			bGoingForward,
		FLMUINT64		ui64RowId,
		FLMBOOL			bPopulateCurRowId,
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
			m_eTransType = SFLM_NO_TRANS;
		}
	}

	FINLINE RCODE checkTransaction(
		F_Db *	pDb)
	{
		RCODE	rc = NE_SFLM_OK;
		
		if (pDb->m_uiDirtyRowCount)
		{
			if (RC_BAD( rc = pDb->flushDirtyRows()))
			{
				goto Exit;
			}
		}
		rc = (RCODE)((m_pDb != pDb ||
						  m_ui64CurrTransId != pDb->m_ui64CurrTransID ||
						  m_uiBlkChangeCnt != pDb->m_uiBlkChangeCnt)
						 ? resetTransaction( pDb) 
						 : NE_SFLM_OK);
	Exit:
		return( rc);
	}

	RCODE populateRow(
		F_Db *			pDb,
		F_Row **			ppRow,
		FLMUINT64 *		pui64RowId);

	RCODE checkIfRowInRange(
		FLMBOOL	bPositionForward);

	// Database Information

	FLMUINT64			m_ui64CurrTransId;
	FLMUINT				m_uiBlkChangeCnt;
	FLMUINT				m_uiTableNum;
	F_TABLE *			m_pTable;
	LFILE	*				m_pLFile;
	F_Db *				m_pDb;
	eDbTransType		m_eTransType;

	// Key range information

	FLMBOOL				m_bSetup;
	FLMUINT64			m_ui64FromRowId;
	FLMUINT64			m_ui64UntilRowId;
	FLMUINT64			m_ui64Cost;
	FLMUINT64			m_ui64LeafBlocksBetween;
	FLMUINT64			m_ui64TotalRows;
	FLMBOOL				m_bTotalsEstimated;
	
	// State information.

	FLMBOOL				m_bAtBOF;			// Before the first row.
	FLMBOOL				m_bAtEOF;			// After the last row.
	FLMUINT64			m_ui64CurRowId;	// Current row id
	FLMBYTE				m_ucCurRowKey [FLM_MAX_NUM_BUF_SIZE];
	FLMUINT				m_uiCurRowKeyLen;
	F_Btree *			m_pbTree;
	FLMBOOL				m_bTreeOpen;
};

RCODE flmBuildFromAndUntilKeys(
	F_Dict *			pDict,
	F_INDEX *		pIndex,
	F_TABLE *		pTable,
	SQL_PRED **		ppKeyComponents,
	F_DataVector *	pFromSearchKey,
	FLMBYTE *		pucFromKey,
	FLMUINT *		puiFromKeyLen,
	F_DataVector *	pUntilSearchKey,
	FLMBYTE *		pucUntilKey,
	FLMUINT *		puiUntilKeyLen,
	FLMBOOL *		pbDoRowMatch,
	FLMBOOL *		pbCanCompareOnKey);

#endif
