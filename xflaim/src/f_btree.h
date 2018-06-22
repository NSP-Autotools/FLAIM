//------------------------------------------------------------------------------
// Desc:	Header file for the B-Tree class definitions
// Tabs:	3
//
// Copyright (c) 2002-2007 Novell, Inc. All Rights Reserved.
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

#ifndef F_BTREE_H
#define F_BTREE_H

// Represent the maximum size for data & key before needing two bytes to
// store the length.

#define ONE_BYTE_SIZE				0xFF

// Flag definitions - BT_LEAF_DATA

#define BTE_LEAF_DATA_OVHD			7		// Offset (2) Flags (1) OA Data (4)

#define BTE_FLAG						0		// Offset to the FLAGS field
#define BTE_FLAG_LAST_ELEMENT		0x04
#define BTE_FLAG_FIRST_ELEMENT	0x08
#define BTE_FLAG_DATA_BLOCK		0x10	// Data is stored in a Data-only Block
#define BTE_FLAG_OA_DATA_LEN		0x20	// Overall data length
#define BTE_FLAG_DATA_LEN			0x40
#define BTE_FLAG_KEY_LEN			0x80

// BT_LEAF (no data)

#define BTE_LEAF_OVHD				4		// Offset (2) KeyLen (2)
#define BTE_KEY_LEN					0
#define BTE_KEY_START				2

// BT_NON_LEAF_DATA

#define BTE_NON_LEAF_OVHD			8		// Offset (2) Child Blk Addr (4) KeyLen (2)
#define BTE_NL_CHILD_BLOCK_ADDR	0
#define BTE_NL_KEY_LEN				4
#define BTE_NL_KEY_START			6

// BT_NON_LEAF_COUNTS

#define BTE_NON_LEAF_COUNTS_OVHD	12		// Offset (2) Child Blk Addr (4) Counts (4) KeyLen (2)
#define BTE_NLC_CHILD_BLOCK_ADDR	0
#define BTE_NLC_COUNTS				4
#define BTE_NLC_KEY_LEN				8
#define BTE_NLC_KEY_START			10

// Low water mark for coalescing blocks (as a percentage)

#define BT_LOW_WATER_MARK			65

FINLINE FLMBOOL bteKeyLenFlag( 
	FLMBYTE *			pucEntry)
{
	return( (pucEntry[ BTE_FLAG] & BTE_FLAG_KEY_LEN) ? TRUE : FALSE);
}

FINLINE FLMBOOL bteDataLenFlag( 
	FLMBYTE *			pucEntry)
{
	return( (pucEntry[ BTE_FLAG] & BTE_FLAG_DATA_LEN) ? TRUE : FALSE);
}

FINLINE FLMBOOL bteOADataLenFlag( 
	FLMBYTE *			pucEntry)
{
	return( (pucEntry[ BTE_FLAG] & BTE_FLAG_OA_DATA_LEN) ? TRUE : FALSE);
}

FINLINE FLMBOOL bteDataBlockFlag( 
	FLMBYTE *			pucEntry)
{
	return( (pucEntry[ BTE_FLAG] & BTE_FLAG_DATA_BLOCK) ? TRUE : FALSE);
}

FINLINE FLMBOOL bteFirstElementFlag(
	FLMBYTE *			pucEntry)
{
	return( (pucEntry[ BTE_FLAG] & BTE_FLAG_FIRST_ELEMENT) ? TRUE : FALSE);
}

FINLINE FLMBOOL bteLastElementFlag(
	FLMBYTE *			pucEntry)
{
	return( (pucEntry[ BTE_FLAG] & BTE_FLAG_LAST_ELEMENT) ? TRUE : FALSE);
}

FINLINE FLMUINT32 bteGetBlkAddr(
	const FLMBYTE *	pucEntry)
{
	return( FB2UD( pucEntry));
}

FINLINE void bteSetEntryOffset(
	FLMUINT16 *			pui16OffsetArray,
	FLMUINT				uiOffsetIndex,
	FLMUINT16			ui16Offset)
{
	UW2FBA( ui16Offset, (FLMBYTE *)&pui16OffsetArray[ uiOffsetIndex]);
}

FINLINE FLMUINT16 bteGetEntryOffset( 
	const FLMUINT16 *	pui16OffsetArray,
	FLMUINT				uiOffsetIndex)
{
	return( FB2UW( (FLMBYTE *)&pui16OffsetArray[ uiOffsetIndex]));
}

typedef struct
{
	F_BTREE_BLK_HDR *			pBlkHdr;
	F_CachedBlock *			pSCache;
	const FLMBYTE *			pucKeyBuf;
	FLMUINT						uiKeyLen;
	FLMUINT						uiCurOffset;
	FLMUINT						uiLevel;
	FLMUINT16 *					pui16OffsetArray;
	FLMUINT32					ui32BlkAddr;
} F_BTSK, * F_BTSK_p;

typedef enum
{
	ELM_INSERT_DO,
	ELM_INSERT,
	ELM_REPLACE_DO,
	ELM_REPLACE,
	ELM_REMOVE,
	ELM_BLK_MERGE,
	ELM_DONE
} F_ELM_UPD_ACTION;


FINLINE FLMBYTE * BtEntry(
	FLMBYTE *		pBlk,
	FLMUINT			uiIndex)
{
	FLMBYTE *		pucOffsetArray =
							pBlk + sizeofBTreeBlkHdr( (F_BTREE_BLK_HDR *)pBlk) +
							(uiIndex * 2);  // 2 byte offset entries.

	return( pBlk + FB2UW( pucOffsetArray));
}

FINLINE FLMBYTE * BtLastEntry(
	FLMBYTE *				pBlk
	)
{
	return BtEntry( pBlk, ((F_BTREE_BLK_HDR *)pBlk)->ui16NumKeys - 1);
}

FINLINE FLMUINT getBlkType(
	FLMBYTE *	pBlk)
{
	return (FLMUINT)((F_BLK_HDR *)pBlk)->ui8BlkType;
}

FINLINE FLMUINT16 * BtOffsetArray(
	FLMBYTE *		pBlk,
	FLMUINT			uiIndex)
{
	return (FLMUINT16 *)
		(pBlk + sizeofBTreeBlkHdr( (F_BTREE_BLK_HDR *)pBlk) + (uiIndex * sizeof( FLMUINT16)));
}

// Returns the address of the first entry in the block.  i.e. the first non-blank
// address after the offset array.

FINLINE FLMBYTE * getBlockEnd(
	F_BTREE_BLK_HDR *		pBlkHdr)
{
	return ((FLMBYTE *)pBlkHdr + sizeofBTreeBlkHdr( pBlkHdr) +
			  (pBlkHdr->ui16NumKeys * 2) +
				pBlkHdr->ui16HeapSize);
}

// This inline function takes the parameter returned from getEntrySize which
// adds 2 for the offset.

FINLINE FLMUINT actualEntrySize(
	FLMUINT uiEntrySize)
{
	return uiEntrySize - 2;
}

typedef struct
{
	FLMUINT		uiParentLevel;
	FLMUINT		uiParentKeyLen;
	FLMUINT		uiParentChildBlkAddr;
	FLMUINT		uiNewKeyLen;
	FLMUINT		uiChildBlkAddr;
	FLMUINT		uiCounts;
	void *		pPrev;
	FLMBYTE		pucParentKey[ XFLM_MAX_KEY_SIZE];
	FLMBYTE		pucNewKey[ XFLM_MAX_KEY_SIZE];
} BTREE_REPLACE_STRUCT;

class F_BtPool;

class F_Btree : public F_Object
{
public:


	F_Btree( void);
	~F_Btree( void);

	RCODE btCreate(
		F_Db *					pDb,
		LFILE *					pLFile,
		FLMBOOL					bCounts,
		FLMBOOL					bData);

	RCODE btOpen(
		F_Db *						pDb,
		LFILE *						pLFile,
		FLMBOOL						bCounts,
		FLMBOOL						bData,
		IF_ResultSetCompare *	pCompare = NULL);

	void btClose( void);

	RCODE btDeleteTree(
		IF_DeleteStatus *		ifpDeleteStatus);

	RCODE btGetBlockChains(
		FLMUINT *				puiBlockChains,
		FLMUINT *				puiNumLevels);

	RCODE btRemoveEntry(
		const FLMBYTE *		pucKey,
		FLMUINT					uiKeyLen);

	RCODE btInsertEntry(
		const FLMBYTE *		pucKey,
		FLMUINT					uiKeyLen,
		const FLMBYTE *		pucData,
		FLMUINT					uiDataLen,
		FLMBOOL					bFirst,
		FLMBOOL					bLast,
		FLMUINT32 *				pui32BlkAddr = NULL,
		FLMUINT *				puiOffsetIndex = NULL);

	RCODE btReplaceEntry(
		const FLMBYTE *		pucKey,
		FLMUINT					uiKeyLen,
		const FLMBYTE *		pucData,
		FLMUINT					uiDataLen,
		FLMBOOL					bFirst,
		FLMBOOL					bLast,
		FLMBOOL					bTruncate = TRUE,
		FLMUINT32 *				pui32BlkAddr = NULL,
		FLMUINT *				puiOffsetIndex = NULL);

	RCODE btLocateEntry(
		FLMBYTE *				pucKey,
		FLMUINT					uiKeyBufSize,
		FLMUINT *				puiKeyLen,
		FLMUINT					uiMatch,
		FLMUINT *				puiPosition = NULL,
		FLMUINT *				puiDataLength = NULL,
		FLMUINT32 *				pui32BlkAddr = NULL,
		FLMUINT *				puiOffsetIndex = NULL);

	RCODE btGetEntry(
		FLMBYTE *				pucKey,
		FLMUINT					uiKeyBufSize,
		FLMUINT					uiKeyLen,
		FLMBYTE *				pucData,
		FLMUINT					uiDataBufSize,
		FLMUINT *				puiDataLen);

	RCODE btNextEntry(
		FLMBYTE *				pucKey,
		FLMUINT					uiKeyBufSize,
		FLMUINT *				puiKeyLen,
		FLMUINT *				puiDataLength = NULL,
		FLMUINT32 *				pui32BlkAddr = NULL,
		FLMUINT *				puiOffsetIndex = NULL);


	RCODE btPrevEntry(
		FLMBYTE *				pucKey,
		FLMUINT					uiKeyBufSize,
		FLMUINT *				puiKeyLen,
		FLMUINT *				puiDataLength = NULL,
		FLMUINT32 *				pui32BlkAddr = NULL,
		FLMUINT *				puiOffsetIndex = NULL);

	RCODE btFirstEntry(
		FLMBYTE *				pucKey,
		FLMUINT					uiKeyBufSize,
		FLMUINT *				puiKeyLen,
		FLMUINT *				puiDataLength = NULL,
		FLMUINT32 *				pui32BlkAddr = NULL,
		FLMUINT *				puiOffsetIndex = NULL);

	RCODE btLastEntry(
		FLMBYTE *				pucKey,
		FLMUINT					uiKeyBufSize,
		FLMUINT *				puiKeyLen,
		FLMUINT *				puiDataLength = NULL,
		FLMUINT32 *				pui32BlkAddr = NULL,
		FLMUINT *				puiOffsetIndex = NULL);

	RCODE btSetReadPosition(
		FLMBYTE *				pucKey,
		FLMUINT					uiKeyLen,
		FLMUINT					uiPosition);

	RCODE btGetReadPosition(
		FLMBYTE *				pucKey,
		FLMUINT					uiKeyLen,
		FLMUINT *				puiPosition);

	RCODE btPositionTo(
		FLMUINT					uiPosition,
		FLMBYTE *				pucKey,
		FLMUINT					uiKeyBufSize,
		FLMUINT *				puiKeyLen);

	RCODE btGetPosition(
		FLMBYTE *				pucKey,
		FLMUINT					uiKeyBufSize,
		FLMUINT *				puiPosition);

	RCODE btCheck(
		BTREE_ERR_INFO *		pErrStruct);

	RCODE btRewind(
		FLMBYTE *				pucKey,
		FLMUINT					uiKeyBufSize,
		FLMUINT *				puiKeyLen);

	FINLINE void btGetTransInfo(
		FLMUINT64 *	pui64LowTransId,
		FLMBOOL *	pbMostCurrent)
	{
		*pui64LowTransId = m_ui64LowTransId;
		*pbMostCurrent = m_bMostCurrent;
	}

	FINLINE void btRelease( void)
	{
		releaseBlocks( TRUE);
	}

	FINLINE void btResetBtree( void)
	{
		releaseBlocks( TRUE);
		m_bSetupForRead = FALSE;
		m_bSetupForWrite = FALSE;
		m_bSetupForReplace = FALSE;
		m_bOrigInDOBlocks = FALSE;
		m_bDataOnlyBlock = FALSE;
		m_ui32PrimaryBlkAddr = 0;
		m_ui32CurBlkAddr = 0;
		m_uiPrimaryOffset = 0;
		m_uiCurOffset = 0;
		m_uiDataLength = 0;
		m_uiPrimaryDataLen = 0;
		m_uiOADataLength = 0;
		m_uiDataRemaining = 0;
		m_uiOADataRemaining = 0;
		m_uiOffsetAtStart = 0;
		m_ui64CurrTransID = 0;
		m_ui64LastBlkTransId = 0;
		m_ui64PrimaryBlkTransId = 0;
		m_uiBlkChangeCnt = 0;
		m_uiSearchLevel = BH_MAX_LEVELS;
	}

	RCODE btComputeCounts(
		F_Btree *		pUntilBtree,
		FLMUINT *		puiBlkCount,
		FLMUINT *		puiKeyCount,
		FLMBOOL *		pbTotalsEstimated,
		FLMUINT			uiAvgBlkFullness);

	FINLINE void btSetSearchLevel(
		FLMUINT			uiSearchLevel)
	{
		flmAssert( uiSearchLevel <= BH_MAX_LEVELS);

		btResetBtree();

		m_uiSearchLevel = uiSearchLevel;
	}

	RCODE btMoveBlock(
		FLMUINT32			ui32FromBlkAddr,
		FLMUINT32			ui32ToBlkAddr);

	FINLINE FLMBOOL	btHasCounts( void)
	{
		return m_bCounts;
	}

	FINLINE FLMBOOL	btHasData( void)
	{
		return m_bData;
	}

	FINLINE FLMBOOL	btDbIsOpen( void)
	{
		return m_bOpened;
	}

	FINLINE FLMBOOL btIsSetupForRead( void)
	{
		return m_bSetupForRead;
	}

	FINLINE FLMBOOL btIsSetupForWrite( void)
	{
		return m_bSetupForWrite;
	}

	FINLINE FLMBOOL btIsSetupForReplace( void)
	{
		return m_bSetupForReplace;
	}
	
private:

	FINLINE FLMUINT calcEntrySize(
		FLMUINT		uiBlkType,
		FLMUINT		uiFlags,
		FLMUINT		uiKeyLen,
		FLMUINT		uiDataLen,
		FLMUINT		uiOADataLen)
	{
		switch( uiBlkType)
		{
			case BT_LEAF:
			{
				return( uiKeyLen + 2);
			}

			case BT_LEAF_DATA:
			{
				return( 1 +															// Flags
								(uiKeyLen > ONE_BYTE_SIZE ? 2 : 1) +		// KeyLen
								(uiDataLen > ONE_BYTE_SIZE ? 2 : 1) +		// DataLen
								(uiOADataLen &&									// OA DataLen
									(uiFlags & BTE_FLAG_FIRST_ELEMENT) ? 4 : 0) +
								uiKeyLen + uiDataLen);
			}

			case BT_NON_LEAF:
			case BT_NON_LEAF_COUNTS:
			{
				return( 4 +															// Child block address
						  (uiBlkType == BT_NON_LEAF_COUNTS ? 4 : 0) +	// Counts
							2 +														// Key length
							uiKeyLen);
			}
		}

		return( 0);
	}

	RCODE computeCounts(
		F_BTSK_p			pFromStack,
		F_BTSK_p			pUntilStack,
		FLMUINT *		puiBlockCount,
		FLMUINT *		puiKeyCount,
		FLMBOOL *		pbTotalsEstimated,
		FLMUINT			uiAvgBlkFullness);

	RCODE blockCounts(
		F_BTSK_p			pStack,
		FLMUINT			uiFirstOffset,
		FLMUINT			uiLastOffset,
		FLMUINT *		puiKeyCount,
		FLMUINT *		puiElementCount);

	RCODE getStoredCounts(
		F_BTSK_p			pFromStack,
		F_BTSK_p			pUntilStack,
		FLMUINT *		puiBlockCount,
		FLMUINT *		puiKeyCount,
		FLMBOOL *		pbTotalsEstimated,
		FLMUINT			uiAvgBlkFullness);

	RCODE getCacheBlocks(
		F_BTSK_p			pStack1,
		F_BTSK_p			pStack2);

	FINLINE FLMUINT getAvgKeyCount(
		F_BTSK_p			pFromStack,
		F_BTSK_p			pUntilStack,
		FLMUINT			uiAvgBlkFullness);

	FINLINE void updateTransInfo(
		FLMUINT64	ui64LowTransID,
		FLMUINT64	ui64HighTransID
		)
	{
		if (m_ui64LowTransId > ui64LowTransID)
		{
			m_ui64LowTransId = ui64LowTransID;
		}

		if (!m_bMostCurrent)
		{
			m_bMostCurrent = (ui64HighTransID == FLM_MAX_UINT64)
									? TRUE
									: FALSE;
		}
	}

	FINLINE FLMUINT getBlkEntryCount(
		FLMBYTE *		pBlk
		)
	{
		return ((F_BTREE_BLK_HDR *)pBlk)->ui16NumKeys;
	}

	FINLINE FLMUINT getBlkAvailSpace(
		FLMBYTE *		pBlk
		)
	{
		return ((F_BLK_HDR *)pBlk)->ui16BlkBytesAvail;
	}

	FLMUINT getEntryKeyLength(
		FLMBYTE *			pucEntry,
		FLMUINT				uiBlockType,
		const FLMBYTE **	ppucKeyRV);

	FLMUINT getEntrySize(
		FLMBYTE *			pBlk,
		FLMUINT				uiOffset,
		FLMBYTE **			ppucEntry = NULL);

	RCODE calcNewEntrySize(
		FLMUINT			uiKeyLen,
		FLMUINT			uiDataLen,
		FLMUINT *		puiEntrySize,
		FLMBOOL *		pbHaveRoom,
		FLMBOOL *		pbDefragBlk);

	RCODE extractEntryData(
		FLMBYTE *			pucKey,
		FLMUINT				uiKeyLen,
		FLMBYTE *			pucBuffer,
		FLMUINT				uiBufSiz,
		FLMUINT *			puiDataLen);

	RCODE updateEntry(
		const FLMBYTE *		pucKey,
		FLMUINT					uiKeyLen,
		const FLMBYTE *		pucValue,
		FLMUINT					uiLen,
		F_ELM_UPD_ACTION		eAction,
		FLMBOOL					bTruncate = TRUE);

	RCODE insertEntry(
		const FLMBYTE **		ppucKey,
		FLMUINT *				puiKeyLen,
		const FLMBYTE *		pucValue,
		FLMUINT					uiLen,
		FLMUINT					uiFlags,
		FLMUINT *				puiChildBlkAddr,
		FLMUINT *				puiCounts,
		const FLMBYTE **		ppucRemainingValue,
		FLMUINT *				puiRemainingLen,
		F_ELM_UPD_ACTION *	peAction);

	RCODE storeEntry(
		const FLMBYTE *		pucKey,
		FLMUINT					uiKeyLen,
		const FLMBYTE *		pucValue,
		FLMUINT					uiLen,
		FLMUINT					uiFlags,
		FLMUINT					uiOADataLen,
		FLMUINT					uiChildBlkAddr,
		FLMUINT					uiCounts,
		FLMUINT					uiEntrySize,
		FLMBOOL *				pbLastEntry);

	RCODE removeEntry(
		const FLMBYTE **		ppucKey,
		FLMUINT *				puiKeyLen,
		FLMUINT *				puiChildBlkAddr,
		FLMUINT *				puiCounts,
		FLMBOOL *				pbMoreToRemove,
		F_ELM_UPD_ACTION *	peAction);

	RCODE remove(
		FLMBOOL				bDeleteDOBlocks);

	RCODE removeRange(
		FLMUINT				uiStartElm,
		FLMUINT				uiEndElm,
		FLMBOOL				bDeleteDOBlocks);

	RCODE findEntry(
		const FLMBYTE *	pucKey,
		FLMUINT				uiKeyLen,
		FLMUINT				uiMatch,
		FLMUINT *			puiPosition = NULL,
		FLMUINT32 *			pui32BlkAddr = NULL,
		FLMUINT *			puiOffsetIndex = NULL);

	RCODE findInBlock(
		const FLMBYTE *	pucKey,
		FLMUINT				uiKeyLen,
		FLMUINT				uiMatch,
		FLMUINT *			uiPosition,
		FLMUINT32 *			ui32BlkAddr,
		FLMUINT *			uiOffsetIndex);

	RCODE scanBlock(
		F_BTSK_p				pStack,
		FLMUINT				uiMatch);

	RCODE compareKeys(
		const FLMBYTE *	pucKey1,
		FLMUINT				uiKeyLen1,
		const FLMBYTE *	pucKey2,
		FLMUINT				uiKeyLen2,
		FLMINT *				piCompare);

	FINLINE RCODE compareBlkKeys(
		const FLMBYTE *	pucBlockKey,
		FLMUINT				uiBlockKeyLen,
		const FLMBYTE *	pucTargetKey,
		FLMUINT				uiTargetKeyLen,
		FLMINT *				piCompare)
	{
		flmAssert( uiBlockKeyLen);

		if( !m_pCompare && uiBlockKeyLen == uiTargetKeyLen)
		{
			*piCompare = f_memcmp( pucBlockKey, pucTargetKey, uiBlockKeyLen);
									
			return( NE_XFLM_OK);
		}

		return( compareKeys( pucBlockKey, uiBlockKeyLen,
							pucTargetKey, uiTargetKeyLen, piCompare));
	}

	RCODE positionToEntry(
		FLMUINT			uiPosition);

	RCODE searchBlock(
		F_BTREE_BLK_HDR *		pBlkHdr,
		FLMUINT *				puiPrevCounts,
		FLMUINT					uiPosition,
		FLMUINT *				puiOffset);

	RCODE defragmentBlock(
		F_CachedBlock **	ppSCache);

	RCODE advanceToNextElement(
		FLMBOOL			bAdvanceStack);

	RCODE backupToPrevElement(
		FLMBOOL			bBackupStack);

	RCODE replaceEntry(
		const FLMBYTE **		ppucKey,
		FLMUINT *				puiKeyLen,
		const FLMBYTE *		pucValue,
		FLMUINT					uiLen,
		FLMUINT					uiFlags,
		FLMUINT *				puiChildBlkAddr,
		FLMUINT *				puiCounts,
		const FLMBYTE **		ppucRemainingValue,
		FLMUINT *				puiRemainingLen,
		F_ELM_UPD_ACTION *	peAction,
		FLMBOOL					bTruncate = TRUE);

	RCODE replaceOldEntry(
		const FLMBYTE **		ppucKey,
		FLMUINT *				puiKeyLen,
		const FLMBYTE *		pucValue,
		FLMUINT					uiLen,
		FLMUINT					uiFlags,
		FLMUINT					uiOADataLen,
		FLMUINT *				puiChildBlkAddr,
		FLMUINT *				puiCounts,
		const FLMBYTE **		ppucRemainingValue,
		FLMUINT *				puiRemainingLen,
		F_ELM_UPD_ACTION *	peAction,
		FLMBOOL					bTruncate = TRUE);

	RCODE replaceByInsert(
		const FLMBYTE **		ppucKey,
		FLMUINT *				puiKeyLen,
		const FLMBYTE *		pucDataValue,
		FLMUINT					uiDataLen,
		FLMUINT					uiOADataLen,
		FLMUINT					uiFlags,
		FLMUINT *				puiChildBlkAddr,
		FLMUINT *				puiCounts,
		const FLMBYTE **		ppucRemainingValue,
		FLMUINT *				puiRemainingLen,
		F_ELM_UPD_ACTION *	peAction);

	RCODE replace(
		FLMBYTE *			pucEntry,
		FLMUINT				uiEntrySize,
		FLMBOOL *			pbLastEntry);

	RCODE buildAndStoreEntry(
		FLMUINT				uiBlkType,
		FLMUINT				uiFlags,
		const FLMBYTE *	pucKey,
		FLMUINT				uiKeyLen,
		const FLMBYTE *	pucData,
		FLMUINT				uiDataLen,
		FLMUINT				uiOADataLen,
		FLMUINT				uiChildBlkAddr,
		FLMUINT				uiCounts,
		FLMBYTE *			pucBuffer,
		FLMUINT				uiBufferSize,
		FLMUINT *			puiEntrySize);

	RCODE moveEntriesToPrevBlk(
		FLMUINT				uiNewEntrySize,
		F_CachedBlock **	ppPrevSCache,
		FLMBOOL *			pbEntriesWereMoved);

	RCODE moveEntriesToNextBlk(
		FLMUINT		uiEntrySize,
		FLMBOOL *	pbEntriesWereMoved);

	RCODE splitBlock(
		const FLMBYTE *	pucKey,
		FLMUINT				uiKeyLen,
		const FLMBYTE *	pucValue,
		FLMUINT				uiLen,
		FLMUINT				uiFlags,
		FLMUINT				uiOADataLen,
		FLMUINT 				uiChildBlkAddr,
		FLMUINT				uiCounts,
		const FLMBYTE **	ppucRemainingValue,
		FLMUINT *			puiRemainingLen,
		FLMBOOL *			pbBlockSplit);

	RCODE createNewLevel( void);

	RCODE storeDataOnlyBlocks(
		const FLMBYTE *	pucKey,
		FLMUINT				uiKeyLen,
		FLMBOOL				bSaveKey,
		const FLMBYTE *	pucData,
		FLMUINT				uiDataLen);

	RCODE replaceDataOnlyBlocks(
		const FLMBYTE *	pucKey,
		FLMUINT				uiKeyLen,
		FLMBOOL				bSaveKey,
		const FLMBYTE *	pucData,
		FLMUINT				uiDataLen,
		FLMBOOL				bLast,
		FLMBOOL				bTruncate = TRUE);

	RCODE moveToPrev(
		FLMUINT				uiStart,
		FLMUINT				uiFinish,
		F_CachedBlock **	ppPrevSCache);

	RCODE moveToNext(
		FLMUINT				uiStart,
		FLMUINT				uiFinish,
		F_CachedBlock **	ppNextSCache);

	RCODE updateParentCounts(
		F_CachedBlock *	pChildSCache,
		F_CachedBlock **	ppParentSCache,
		FLMUINT				uiParentElm);

	FLMUINT countKeys(
		FLMBYTE *			pBlk);

	FLMUINT countRangeOfKeys(
		F_BTSK_p				pFromStack,
		FLMUINT				uiFromOffset,
		FLMUINT				uiUntilOffset);

	RCODE moveStackToPrev(
		F_CachedBlock *	pPrevSCache);

	RCODE moveStackToNext(
		F_CachedBlock *	pSCache,
		FLMBOOL				bReleaseCurrent = TRUE);

	RCODE calcOptimalDataLength(
		FLMUINT				uiKeyLen,
		FLMUINT				uiDataLen,
		FLMUINT				uiBytesAvail,
		FLMUINT *			puiNewDataLen);

	// Performs an integrity check on a chain of data-only blocks
	
	RCODE verifyDOBlkChain(
		FLMUINT					uiDOAddr,
		FLMUINT					uiDataLength,
		BTREE_ERR_INFO *		plErrStruct);

	// Performs a check to verify that the counts in the DB match.
	RCODE verifyCounts(
		BTREE_ERR_INFO *		pErrStruct);

	void releaseBlocks(
		FLMBOOL				bResetStack);

	void releaseBtree( void);

	RCODE saveReplaceInfo(
		const FLMBYTE *	pucKey,
		FLMUINT				uiKeyLen);

	RCODE restoreReplaceInfo(
		const FLMBYTE **	ppucKey,
		FLMUINT *			puiKeyLen,
		FLMUINT *			puiChildBlkAddr,
		FLMUINT *			puiCounts);

	FINLINE RCODE setReturnKey(
		FLMBYTE *			pucEntry,
		FLMUINT				uiBlockType,
		FLMBYTE *			pucKey,
		FLMUINT *			puiKeyLen,
		FLMUINT				uiKeyBufSize);

	RCODE setupReadState(
		F_BLK_HDR *			pBlkHdr,
		FLMBYTE *			pucEntry);

	RCODE removeRemainingEntries(
		const FLMBYTE *	pucKey,
		FLMUINT				uiKeyLen);

	RCODE deleteEmptyBlock( void);

	RCODE removeDOBlocks(
		FLMUINT32		ui32OrigDOAddr);

	RCODE replaceMultiples(
		const FLMBYTE **		ppucKey,
		FLMUINT *				puiKeyLen,
		const FLMBYTE *		pucDataValue,
		FLMUINT					uiLen,
		FLMUINT					uiFlags,
		FLMUINT *				puiChildBlkAddr,
		FLMUINT *				puiCounts,
		const FLMBYTE **		ppucRemainingValue,
		FLMUINT *				puiRemainingLen,
		F_ELM_UPD_ACTION *	peAction);

	RCODE replaceMultiNoTruncate(
		const FLMBYTE **		ppucKey,
		FLMUINT *				puiKeyLen,
		const FLMBYTE *		pucDataValue,
		FLMUINT					uiLen,
		FLMUINT					uiFlags,
		FLMUINT *				puiChildBlkAddr,
		FLMUINT *				puiCounts,
		const FLMBYTE **		ppucRemainingValue,
		FLMUINT *				puiRemainingLen,
		F_ELM_UPD_ACTION *	peAction);

	FINLINE RCODE getNextBlock(
		F_CachedBlock **		ppSCache);

	FINLINE RCODE getPrevBlock(
		F_CachedBlock **		ppSCache);

	FLMBOOL checkContinuedEntry(
		const FLMBYTE *		pucKey,
		FLMUINT					uiKeyLen,
		FLMBOOL *				pbLastElement,
		FLMBYTE *				pucEntry,
		FLMUINT					uiBlkType);

	RCODE updateCounts( void);

	RCODE storePartialEntry(
		const FLMBYTE *		pucKey,
		FLMUINT					uiKeyLen,
		const FLMBYTE *		pucValue,
		FLMUINT					uiLen,
		FLMUINT					uiFlags,
		FLMUINT					uiChildBlkAddr,
		FLMUINT 					uiCounts,
		const FLMBYTE **		ppucRemainingValue,
		FLMUINT *				puiRemainingLen,
		FLMBOOL					bNewBlock = FALSE);

	RCODE mergeBlocks(
		FLMBOOL					bLastEntry,
		FLMBOOL *				pbMergedWithPrev,
		FLMBOOL *				pbMergedWithNext,
		F_ELM_UPD_ACTION *	peAction);

	RCODE merge(
		F_CachedBlock **		ppFromSCache,
		F_CachedBlock **		ppToSCache);

	RCODE checkDownLinks( void);

	RCODE verifyChildLinks(
		F_CachedBlock *		pParentSCache);

	RCODE combineEntries(
		F_BTREE_BLK_HDR *		pSrcBlkHdr,
		FLMUINT					uiSrcOffset,
		F_BTREE_BLK_HDR *		pDstBlkHdr,
		FLMUINT					uiDstOffset,
		FLMBOOL *				pbEntriesCombined,
		FLMUINT *				puiEntrySize);

	RCODE moveBtreeBlock(
		FLMUINT32			ui32FromBlkAddr,
		FLMUINT32			ui32ToBlkAddr);

	RCODE moveDOBlock(
		FLMUINT32			ui32FromBlkAddr,
		FLMUINT32			ui32ToBlkAddr);

// Member variables
	FLMBOOL						m_bCounts;				// BT_NON_LEAF_COUNTS
	FLMBOOL						m_bData;					// BT_LEAF_DATA
	FLMBOOL						m_bSetupForRead;
	FLMBOOL						m_bSetupForWrite;
	FLMBOOL						m_bSetupForReplace;
	FLMBOOL						m_bOpened;
	FLMBOOL						m_bMostCurrent;
	FLMBOOL						m_bDataOnlyBlock;
	FLMBOOL						m_bOrigInDOBlocks;
	FLMBOOL						m_bFirstRead;
	FLMBOOL						m_bStackSetup;
	LFILE *						m_pLFile;
	F_Db *						m_pDb;
	FLMBOOL						m_bTempDb;
	F_BTSK_p						m_pStack;				// Used for traversing the B-Tree
	FLMBYTE *					m_pucTempBlk;
	FLMBYTE *					m_pucTempDefragBlk;
	BTREE_REPLACE_STRUCT *	m_pReplaceInfo;
	BTREE_REPLACE_STRUCT *	m_pReplaceStruct;
	const FLMBYTE *			m_pucDataPtr;
	F_CachedBlock *			m_pSCache;
	F_BTREE_BLK_HDR *			m_pBlkHdr;
	FLMBYTE *					m_pucBuffer;			// Buffer used during moves
	FLMUINT						m_uiBufferSize;		// Size of the buffer
	FLMUINT						m_uiBlockSize;
	FLMUINT						m_uiDefragThreshold;
	FLMUINT						m_uiOverflowThreshold;
	FLMUINT						m_uiStackLevels;
	FLMUINT						m_uiRootLevel;
	FLMUINT						m_uiReplaceLevels;
	FLMUINT						m_uiBlkChangeCnt;
	FLMUINT						m_uiDataLength;
	FLMUINT						m_uiPrimaryDataLen;
	FLMUINT						m_uiOADataLength;
	FLMUINT						m_uiDataRemaining;
	FLMUINT						m_uiOADataRemaining;
	FLMUINT						m_uiPrimaryOffset;	// Offset into primary block
	FLMUINT						m_uiCurOffset;			// Offset into current block
	FLMUINT						m_uiSearchLevel;
	FLMUINT						m_uiOffsetAtStart;	// Offset into the current
																// element at the beginning of
																// the entry. An element may
																// span multiple entries.
	FLMUINT32					m_ui32PrimaryBlkAddr;// Primary block address
	FLMUINT32					m_ui32DOBlkAddr;		// Address of first DO Block
	FLMUINT32					m_ui32CurBlkAddr;		// Current block being read
	FLMUINT64					m_ui64LowTransId;
	FLMUINT64					m_ui64LastBlkTransId;
	FLMUINT64					m_ui64PrimaryBlkTransId;
	FLMUINT64					m_ui64CurrTransID;
	F_BTSK						m_Stack[ BH_MAX_LEVELS];
	// The m_pNext field is only used by the btPool object.
	F_Btree *					m_pNext;
	IF_ResultSetCompare *	m_pCompare;

friend class F_BtPool;
friend class F_Db;
friend class F_Rfl;
};

RCODE btFreeBlockChain(
	F_Db *					pDb,
	LFILE *					pLFile,
	FLMUINT					uiStartAddr,
	FLMUINT					uiBlocksToFree,
	FLMUINT *				puiBlocksFreed,
	FLMUINT *				puiEndAddr,
	IF_DeleteStatus *		ifpDeleteStatus);

#endif
