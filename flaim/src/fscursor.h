//-------------------------------------------------------------------------
// Desc:	Routines used during query to traverse through index b-trees - definitions.
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

#include "fpackon.h"

// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

/****************************************************************************
Desc:
****************************************************************************/
typedef struct KEYPOS
{
	FLMUINT			uiKeyLen;
	FLMUINT			uiRecordId;
	FLMBOOL			bExclusiveKey;

	// State information
	
	FLMUINT			uiRefPosition;
	FLMUINT			uiDomain;
	FLMUINT			uiBlockTransId;
	FLMUINT			uiBlockAddr;
	FLMUINT			uiCurElm;
	DIN_STATE		DinState;

	// Stack and key information
	
	BTSK *			pStack;
	FLMBOOL			bStackInUse;
	BTSK				Stack [BH_MAX_LEVELS];
	FLMBYTE			pKey [MAX_KEY_SIZ + 4];	// + 4 is for safety
} KEYPOS;

/****************************************************************************
Desc:
****************************************************************************/
typedef struct KEYSET
{
	KEYPOS			fromKey;
	KEYPOS			untilKey;
	KEYSET *			pNext;
	KEYSET *			pPrev;
} KEYSET;

/****************************************************************************
Desc:	File system implementation of a cursor for an index.
****************************************************************************/
class FSIndexCursor : public F_Object
{
public:

	FSIndexCursor();
	
	virtual ~FSIndexCursor();

	// Reset the cursor back to an empty state.
	
	void reset( void);

	// Reset the transaction on this cursor.
	
	RCODE resetTransaction( 
		FDB *					pDb);

	// Release all b-tree blocks back to the cache.
	
	void releaseBlocks( void);

	RCODE	setupKeys(
		FDB *					pDb,
		IXD *					pIxd,
		QPREDICATE ** 		ppQPredicateList,
		FLMBOOL *			pbDoRecMatch,
		FLMBOOL *			pbDoKeyMatch,
		FLMUINT *			puiLeafBlocksBetween,
		FLMUINT *			puiTotalKeys,	
		FLMUINT *			puiTotalRefs,	
		FLMBOOL *			pbTotalsEstimated);

	RCODE	setupKeys(
		FDB *					pDb,
		IXD *       		pIxd,
		FLMBYTE *			pFromKey,
		FLMUINT				uiFromKeyLen,
		FLMUINT				uiFromRecordId,
		FLMBYTE *			pUntilKey,
		FLMUINT				uiUntilKeyLen,
		FLMUINT				uiUntilRecordId,
		FLMBOOL				bExclusiveUntil);

	RCODE unionKeys(
		FSIndexCursor * 	pFSCursor);

	RCODE intersectKeys(
		FDB *					pDb,
		FSIndexCursor * 	pFSCursor);

	FLMBOOL compareKeyRange(
		FLMBYTE *			pFromKey,
		FLMUINT				uiFromKeyLen,
		FLMBOOL				bExclusiveFrom,
		FLMBYTE *			pUntilKey,
		FLMUINT				uiUntilKeyLen,
		FLMBOOL				bExclusiveUntil,
		FLMBOOL *			pbUntilKeyInSet,
		FLMBOOL *			pbUntilGreaterThan);
	
	RCODE currentKey(
		FDB *					pDb,
		FlmRecord **		pPrecordKey,
		FLMUINT *			puiRecordId);
	
	RCODE currentKeyBuf(
		FDB *					pDb,
		F_Pool *				pPool,
		FLMBYTE **			ppKeyBuf,
		FLMUINT *			puiKeyLen,
		FLMUINT *			puiRecordId,
		FLMUINT *			puiContainerId);
	
	RCODE firstKey(
		FDB *					pDb,
		FlmRecord **		pPrecordKey,
		FLMUINT *			puiRecordId);

	RCODE lastKey(
		FDB *					pDb,
		FlmRecord **		pPrecordKey,
		FLMUINT *			puiRecordId);

	RCODE	nextKey(
		FDB *					pDb,
		FlmRecord **		pPrecordKey,
		FLMUINT *			puiRecordId);

	RCODE	prevKey(
		FDB *					pDb,
		FlmRecord **		pPrecordKey,
		FLMUINT *			puiRecordId);

	RCODE	nextRef(
		FDB *					pDb,
		FLMUINT *			puiRecordId);

	RCODE	prevRef(
		FDB *					pDb,
		FLMUINT *			puiRecordId);

	RCODE positionTo(
		FDB *					pDb,
		FLMBYTE *			pKey,
		FLMUINT				uiKeyLen,
		FLMUINT				uiRecordId = 0);

	RCODE positionToDomain(
		FDB *					pDb,
		FLMBYTE *			pKey,
		FLMUINT				uiKeyLen,
		FLMUINT				uiDomain);

	FLMBOOL isAbsolutePositionable( void)
	{
		return (m_pIxd->uiFlags & IXD_POSITIONING) ? TRUE : FALSE;
	}

	// Set absolute position (if not supported returns FERR_FAILURE).
	// uiPosition of zero positions to BOF, ~0 to EOF, one based value.
	
	RCODE setAbsolutePosition(
		FDB *					pDb,
		FLMUINT				uiRefPosition);

	// Get absolute position (if not supported returns FERR_FAILURE).
	// uiPosition of zero positions to BOF, ~0 to EOF, one based value.
	
	RCODE getAbsolutePosition(
		FDB *					pDb,
		FLMUINT *			puiRefPosition);

	// Get the total number of reference with all from/until sets.
	// Does not have to support absolute positioning.
	
	RCODE getTotalReferences(
		FDB *					pDb,
		FLMUINT *			puiTotalRefs,
		FLMBOOL *			pbTotalEstimated);

	RCODE savePosition( void);

	RCODE restorePosition( void);

	RCODE	getFirstLastKeys(
		FLMBYTE **			ppFirstKey,
		FLMUINT *			puiFirstKeyLen,
		FLMBYTE **			ppLastKey,
		FLMUINT *			puiLastKeyLen,
		FLMBOOL *			pbLastKeyExclusive);

protected:

	KEYSET *	getFromUntilSets( void) 
	{
		return m_pFirstSet;
	}

private:

	void freeSets( void);

	RCODE useNewDb( 
		FDB *	pDb);

	FLMBOOL FSCompareKeyPos(
		KEYSET *				pSet1,
		KEYSET *				pSet2,
		FLMBOOL *			pbFromKeysLessThan,
		FLMBOOL *			pbUntilKeysGreaterThan);

	RCODE setKeyPosition(
		FDB *					pDb,
		FLMBOOL				bGoingForward,
		KEYPOS *				pInKeyPos,
		KEYPOS *				pOutKeyPos);

	RCODE reposition(
		FDB *					pDb,
		FLMBOOL				bCanPosToNextKey,
		FLMBOOL				bCanPosToPrevKey,
		FLMBOOL *			pbKeyGone,
		FLMBOOL				bCanPosToNextRef,
		FLMBOOL				bCanPosToPrevRef,
		FLMBOOL *			pbRefGone);

	void releaseKeyBlocks( 
		KEYPOS *				pKeyPos)
	{
		if( pKeyPos->bStackInUse)
		{
			FSReleaseStackCache( pKeyPos->Stack, BH_MAX_LEVELS, FALSE);
			pKeyPos->bStackInUse = FALSE;
		}
	}

	RCODE checkTransaction(
		FDB *					pDb)
	{
		return (RCODE) ((m_uiCurrTransId != pDb->LogHdr.uiCurrTransID ||
			m_uiBlkChangeCnt != pDb->uiBlkChangeCnt)
				? resetTransaction( pDb) 
				: FERR_OK);
	}

	RCODE	setupForPositioning(
		FDB *					pDb);

	// Save the current key position into pSaveKeyPos
	
	void saveCurrKeyPos(
		KEYPOS *				pSaveKeyPos);

	// Restore the current key position from pSaveKeyPos
	
	void restoreCurrKeyPos(
		KEYPOS *				pSaveKeyPos);

	RCODE getKeySet(
		FLMBYTE *			pKey,
		FLMUINT				uiKeyLen,
		KEYSET **			ppKeySet);

	FLMUINT					m_uiCurrTransId;
	FLMUINT					m_uiBlkChangeCnt;
	FLMBOOL					m_bIsUpdateTrans;
	FLMUINT					m_uiIndexNum;
	LFILE	*					m_pLFile;
	IXD *						m_pIxd;
	KEYSET *					m_pFirstSet;
	KEYSET *					m_pCurSet;
	FLMBOOL					m_bAtBOF;
	FLMBOOL					m_bAtEOF;
	KEYPOS					m_curKeyPos;
	KEYPOS *					m_pSavedPos;
	KEYSET					m_DefaultSet;
};

/****************************************************************************
Desc:
****************************************************************************/
typedef struct RECPOS
{
	FLMUINT			uiRecordId;
	FLMUINT			uiBlockTransId;
	FLMUINT			uiBlockAddr;
	BTSK *			pStack;
	FLMBOOL			bStackInUse;
	FLMBOOL			bExclusiveKey;
	BTSK				Stack [BH_MAX_LEVELS];
	FLMBYTE			pKey [DIN_KEY_SIZ];
} RECPOS;

/****************************************************************************
Desc:	The record set will always have inclusive FROM/UNTIL values.
****************************************************************************/
typedef struct RECSET
{
	RECPOS			fromKey;
	RECPOS			untilKey;
	RECSET *			pNext;
	RECSET *			pPrev;
} RECSET;

/****************************************************************************
Desc:	File system implementation of a cursor for a data container.
****************************************************************************/
class FSDataCursor: public F_Object
{
public:

	FSDataCursor();
	
	virtual ~FSDataCursor();

	// Reset this cursor back to an initial state.
	
	void reset( void);

	// Reset the transaction on this cursor.
	
	RCODE resetTransaction( 
		FDB *				pDb);

	void releaseBlocks( void);

	void setContainer( FLMUINT uiContainer)
	{
		m_uiContainer = uiContainer;
	}

	RCODE	setupRange(
		FDB *				pDb,
		FLMUINT			uiContainer,
		FLMUINT			uiLowRecordId,
		FLMUINT			uiHighRecordId,
		FLMUINT *		puiLeafBlocksBetween,
		FLMUINT *		puiTotalRecords,		
		FLMBOOL *		pbTotalsEstimated);

	RCODE unionRange(
		FSDataCursor * pFSCursor);

	RCODE intersectRange(
		FSDataCursor * pFSCursor);

	RCODE currentRec(
		FDB *				pDb,
		FlmRecord **	pPrecord,
		FLMUINT *		puiRecordId);
	
	RCODE firstRec(
		FDB *				pDb,
		FlmRecord **	pPrecord,
		FLMUINT *		puiRecordId);

	RCODE lastRec(
		FDB *				pDb,
		FlmRecord **	pPrecord,
		FLMUINT *		puiRecordId);

	RCODE	nextRec(
		FDB *				pDb,
		FlmRecord **	pPrecord,
		FLMUINT *		puiRecordId);

	RCODE	prevRec(
		FDB *				pDb,
		FlmRecord **	pPrecord,
		FLMUINT *		puiRecordId);

	RCODE positionTo(
		FDB *				pDb,
		FLMUINT			uiRecordId);

	RCODE positionToOrAfter(
		FDB *				pDb,
		FLMUINT *		puiRecordId);

	RCODE savePosition( void);

	RCODE restorePosition( void);

protected:

	RECSET *	getFromUntilSets( void) 
	{
		return m_pFirstSet;
	}

private:

	void freeSets( void);

	void releaseRecBlocks( 
		RECPOS *			pRecPos)
	{
		if( pRecPos->bStackInUse)
		{
			FSReleaseStackCache( pRecPos->Stack, BH_MAX_LEVELS, FALSE);
			pRecPos->bStackInUse = FALSE;
		}
	}
	
	RCODE setRecPosition(
		FDB *				pDb,
		FLMBOOL			bGoingForward,
		RECPOS *			pInRecPos,
		RECPOS *			pOutRecPos);

	RCODE reposition(
		FDB *				pDb,
		FLMBOOL			bCanPosToNextRec,
		FLMBOOL			bCanPosToPrevRec,
		FLMBOOL *		pbRecordGone);

	FLMBOOL FSCompareRecPos(
		RECSET  *		pSet1,
		RECSET  *		pSet2,
		FLMBOOL *		pbFromKeysLessThan,
		FLMBOOL *		pbUntilKeysGreaterThan);

	RCODE checkTransaction(
		FDB *				pDb)
	{
		return (RCODE) ((m_uiCurrTransId != pDb->LogHdr.uiCurrTransID ||
			m_uiBlkChangeCnt != pDb->uiBlkChangeCnt)
				? resetTransaction( pDb) 
				: FERR_OK);
	}
	
	FLMUINT				m_uiCurrTransId;
	FLMUINT				m_uiBlkChangeCnt;
	FLMBOOL				m_bIsUpdateTrans;
	FLMUINT				m_uiContainer;
	LFILE	*				m_pLFile;
	RECSET *				m_pFirstSet;
	RECSET *				m_pCurSet;
	FLMBOOL				m_bAtBOF;
	FLMBOOL				m_bAtEOF;
	RECPOS				m_curRecPos;
	RECPOS *				m_pSavedPos;
	RECSET				m_DefaultSet;
};

#include "fpackoff.h"

#endif
