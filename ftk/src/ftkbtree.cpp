//------------------------------------------------------------------------------
// Desc:	This class handles all of operations on a given B-Tree.
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

#include "ftksys.h"

FSTATIC FLMUINT fbtGetEntryDataLength(
	FLMBYTE *			pucEntry,
	const FLMBYTE **	ppucDataRV,
	FLMUINT *			puiOADataLengthRV,
	FLMBOOL *			pbDOBlockRV);

FSTATIC RCODE fbtGetEntryData(
	FLMBYTE *			pucEntry,
	FLMBYTE *			pucBufferRV,
	FLMUINT				uiBufferSize,
	FLMUINT *			puiLenDataRV);

/****************************************************************************
Desc:	Standard block header
****************************************************************************/
typedef struct
{
	FLMUINT32			ui32BlockAddr;
	FLMUINT32			ui32PrevBlockInChain;
	FLMUINT32			ui32NextBlockInChain;
	FLMUINT32			ui32PriorBlockImgAddr;
	FLMUINT64			ui64TransId;
	FLMUINT32			ui32BlockChecksum;
	FLMUINT16			ui16BlockBytesAvail;
	FLMUINT8				ui8BlockFlags;
	FLMUINT8				ui8BlockType;
} F_STD_BLK_HDR;

#define F_STD_BLK_HDR_ui32BlockAddr_OFFSET				0
#define F_STD_BLK_HDR_ui32PrevBlockInChain_OFFSET		4
#define F_STD_BLK_HDR_ui32NextBlockInChain_OFFSET		8
#define F_STD_BLK_HDR_ui32PriorBlockImgAddr_OFFSET		12
#define F_STD_BLK_HDR_ui64TransId_OFFSET					16
#define F_STD_BLK_HDR_ui32BlockChecksum_OFFSET			24
#define F_STD_BLK_HDR_ui16BlockBytesAvail_OFFSET		28
#define F_STD_BLK_HDR_ui8BlockFlags_OFFSET				30
#define F_STD_BLK_HDR_ui8BlockType_OFFSET					31

#define F_BLK_TYPE_FREE 										0
#define F_BLK_TYPE_1_RESERVED									1
#define F_BLK_TYPE_BT_LEAF										2
#define F_BLK_TYPE_BT_NON_LEAF								3
#define F_BLK_TYPE_BT_NON_LEAF_COUNTS						4
#define F_BLK_TYPE_BT_LEAF_DATA								5
#define F_BLK_TYPE_BT_DATA_ONLY								6

#define F_BLK_FORMAT_IS_LITTLE_ENDIAN						0x01
#define F_BLK_IS_BEFORE_IMAGE									0x02
#define F_BLK_IS_ENCRYPTED										0x04

/****************************************************************************
Desc:	B-Tree block header
****************************************************************************/
typedef struct F_BTREE_BLK_HDR : F_STD_BLK_HDR
{
	FLMUINT16			ui16BtreeId;
	FLMUINT16			ui16NumKeys;
	FLMUINT8				ui8BlockLevel;
	FLMUINT8				ui8BTreeFlags;
	FLMUINT16			ui16HeapSize;
} F_BTREE_BLK_HDR;

#define F_BTREE_BLK_HDR_ui16BtreeId_OFFSET				32
#define F_BTREE_BLK_HDR_ui16NumKeys_OFFSET				34
#define F_BTREE_BLK_HDR_ui8BlockLevel_OFFSET				36
#define F_BTREE_BLK_HDR_ui8BTreeFlags_OFFSET				37
#define F_BTREE_BLK_HDR_ui16HeapSize_OFFSET				38

#define F_BTREE_BLK_IS_ROOT									0x01
#define SIZEOF_STD_BLK_HDR										sizeof( F_STD_BLK_HDR)

/****************************************************************************
Desc:	 	This is a union of all block header types - so that we can have
			something that gives us the largest block header type in one
			structure.
****************************************************************************/
typedef struct
{
	union
	{
		F_STD_BLK_HDR		stdBlockHdr;
		F_BTREE_BLK_HDR	BTreeBlockHdr;
	} all;
} F_LARGEST_BLK_HDR;

#define SIZEOF_LARGEST_BLK_HDR								sizeof( F_LARGEST_BLK_HDR)

/****************************************************************************
Desc:	Encrypted B-Tree block header
****************************************************************************/
typedef struct F_ENC_BTREE_BLK_HDR : F_BTREE_BLK_HDR
{
	FLMUINT64			ui64Reserved;
} F_ENC_BTREE_BLK_HDR;

/****************************************************************************
Desc:	Encrypted data-only block header
****************************************************************************/
typedef struct F_ENC_DO_BLK_HDR : F_STD_BLK_HDR
{
	FLMUINT32			ui32EncId;
	FLMBYTE				ucReserved[ 12];
} F_ENC_DO_BLK_HDR;

/****************************************************************************
Desc:
****************************************************************************/
typedef struct
{
	FLMUINT				uiParentLevel;
	FLMUINT				uiParentKeyLen;
	FLMUINT				uiParentChildBlockAddr;
	FLMUINT				uiNewKeyLen;
	FLMUINT				uiChildBlockAddr;
	FLMUINT				uiCounts;
	void *				pPrev;
	FLMBYTE				pucParentKey[ FLM_MAX_KEY_SIZE];
	FLMBYTE				pucNewKey[ FLM_MAX_KEY_SIZE];
} BTREE_REPLACE_STRUCT;

/****************************************************************************
Desc:
****************************************************************************/
typedef struct
{
	IF_Block *			pBlock;
	FLMBYTE *			pucBlock;
	const FLMBYTE *	pucKeyBuf;
	FLMUINT				uiKeyBufSize;
	FLMUINT				uiKeyLen;
	FLMUINT				uiCurOffset;
	FLMUINT				uiLevel;
	FLMUINT16 *			pui16OffsetArray;
	FLMUINT32			ui32BlockAddr;
} F_BTSK;

/****************************************************************************
Desc:
****************************************************************************/
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

#define BTE_NON_LEAF_OVHD			8		// Offset (2) Child Block Addr (4) KeyLen (2)
#define BTE_NL_CHILD_BLOCK_ADDR	0
#define BTE_NL_KEY_LEN				4
#define BTE_NL_KEY_START			6

// BT_NON_LEAF_COUNTS

#define BTE_NON_LEAF_COUNTS_OVHD	12		// Offset (2) Child Block Addr (4) Counts (4) KeyLen (2)
#define BTE_NLC_CHILD_BLOCK_ADDR	0
#define BTE_NLC_COUNTS				4
#define BTE_NLC_KEY_LEN				8
#define BTE_NLC_KEY_START			10

// Low water mark for coalescing blocks (as a percentage)

#define BT_LOW_WATER_MARK			65
	
/****************************************************************************
Desc:	Block
****************************************************************************/
class F_Block : public IF_Block
{
public:

	F_Block()
	{
		m_pucBlock = NULL;
		m_pPrevInBucket = NULL;
		m_pNextInBucket = NULL;
		m_ui32BlockAddr = 0;
	}
	
	virtual ~F_Block()
	{
		f_assert( !m_pPrevInBucket);
		f_assert( !m_pNextInBucket);
		
		if( m_pucBlock)
		{
			f_free( &m_pucBlock);
		}
	}
	
private:

	FLMBYTE *			m_pucBlock;
	F_Block *			m_pPrevInBucket;
	F_Block *			m_pNextInBucket;
	FLMUINT32			m_ui32BlockAddr;

friend class F_BlockMgr;
};

/****************************************************************************
Desc:	Block manager
****************************************************************************/
class F_BlockMgr : public IF_BlockMgr
{
public:

	F_BlockMgr()
	{
		m_pHashTbl = NULL;
		m_uiBuckets = 0;
		m_ui32NextBlockAddr = 1;
	}
	
	virtual ~F_BlockMgr();
	
	RCODE setup(
		FLMUINT					uiBlockSize);
	
	FLMUINT FTKAPI getBlockSize( void);
	
	RCODE FTKAPI getBlock(
		FLMUINT32				ui32BlockAddr,
		IF_Block **				ppBlock,
		FLMBYTE **				ppucBlock);
		
	RCODE FTKAPI createBlock(
		IF_Block **				ppBlock,
		FLMBYTE **				ppucBlock,
		FLMUINT32 *				pui32BlockAddr);
	
	RCODE FTKAPI freeBlock(
		IF_Block **				ppBlock,
		FLMBYTE **				ppucBlock);
	
	RCODE FTKAPI prepareForUpdate(
		IF_Block **				ppBlock,
		FLMBYTE **				ppucBlock);
		
private:

	void freeAllBlocks( void);

	F_Block **					m_pHashTbl;
	FLMUINT						m_uiBuckets;
	FLMUINT						m_uiBlockSize;
	FLMUINT32					m_ui32NextBlockAddr;
};
	
/****************************************************************************
Desc:
****************************************************************************/
class F_BTree : public IF_BTree
{
public:

	F_BTree(
		IF_BlockMgr *				pBlockMgr);
	
	virtual ~F_BTree( void);

	RCODE FTKAPI btCreate(
		FLMUINT16					ui16BtreeId,
		FLMBOOL						bCounts,
		FLMBOOL						bData,
		FLMUINT32 *					pui32RootBlockAddr,
		IF_ResultSetCompare *	pCompare = NULL);

	RCODE FTKAPI btOpen(
		FLMUINT32					ui32RootBlockAddr,
		FLMBOOL						bCounts,
		FLMBOOL						bData,
		IF_ResultSetCompare *	pCompare = NULL);

	void FTKAPI btClose( void);

	RCODE FTKAPI btDeleteTree(
		IF_DeleteStatus *			ifpDeleteStatus);

	RCODE FTKAPI btGetBlockChains(
		FLMUINT *					puiBlockChains,
		FLMUINT *					puiNumLevels);

	RCODE FTKAPI btRemoveEntry(
		const FLMBYTE *			pucKey,
		FLMUINT						uiKeyBufSize,
		FLMUINT						uiKeyLen);

	RCODE FTKAPI btInsertEntry(
		const FLMBYTE *			pucKey,
		FLMUINT						uiKeyBufSize,
		FLMUINT						uiKeyLen,
		const FLMBYTE *			pucData,
		FLMUINT						uiDataLen,
		FLMBOOL						bFirst, // VISIT: This isn't really needed.  We can track state internally
		FLMBOOL						bLast,
		FLMUINT32 *					pui32BlockAddr = NULL,
		FLMUINT *					puiOffsetIndex = NULL);

	RCODE FTKAPI btReplaceEntry(
		const FLMBYTE *			pucKey,
		FLMUINT						uiKeyBufSize,
		FLMUINT						uiKeyLen,
		const FLMBYTE *			pucData,
		FLMUINT						uiDataLen,
		FLMBOOL						bFirst,
		FLMBOOL						bLast,
		FLMBOOL						bTruncate = TRUE,
		FLMUINT32 *					pui32BlockAddr = NULL,
		FLMUINT *					puiOffsetIndex = NULL);

	RCODE FTKAPI btLocateEntry(
		FLMBYTE *					pucKey,
		FLMUINT						uiKeyBufSize,
		FLMUINT *					puiKeyLen,
		FLMUINT						uiMatch,
		FLMUINT *					puiPosition = NULL,
		FLMUINT *					puiDataLength = NULL,
		FLMUINT32 *					pui32BlockAddr = NULL,
		FLMUINT *					puiOffsetIndex = NULL);

	RCODE FTKAPI btGetEntry(
		FLMBYTE *					pucKey,
		FLMUINT						uiKeyLen,
		FLMBYTE *					pucData,
		FLMUINT						uiDataBufSize,
		FLMUINT *					puiDataLen);

	RCODE FTKAPI btNextEntry(
		FLMBYTE *					pucKey,
		FLMUINT						uiKeyBufSize,
		FLMUINT *					puiKeyLen,
		FLMUINT *					puiDataLength = NULL,
		FLMUINT32 *					pui32BlockAddr = NULL,
		FLMUINT *					puiOffsetIndex = NULL);

	RCODE FTKAPI btPrevEntry(
		FLMBYTE *					pucKey,
		FLMUINT						uiKeyBufSize,
		FLMUINT *					puiKeyLen,
		FLMUINT *					puiDataLength = NULL,
		FLMUINT32 *					pui32BlockAddr = NULL,
		FLMUINT *					puiOffsetIndex = NULL);

	RCODE FTKAPI btFirstEntry(
		FLMBYTE *					pucKey,
		FLMUINT						uiKeyBufSize,
		FLMUINT *					puiKeyLen,
		FLMUINT *					puiDataLength = NULL,
		FLMUINT32 *					pui32BlockAddr = NULL,
		FLMUINT *					puiOffsetIndex = NULL);

	RCODE FTKAPI btLastEntry(
		FLMBYTE *					pucKey,
		FLMUINT						uiKeyBufSize,
		FLMUINT *					puiKeyLen,
		FLMUINT *					puiDataLength = NULL,
		FLMUINT32 *					pui32BlockAddr = NULL,
		FLMUINT *					puiOffsetIndex = NULL);

	RCODE FTKAPI btSetReadPosition(
		FLMBYTE *					pucKey,
		FLMUINT						uiKeyLen,
		FLMUINT						uiPosition);

	RCODE FTKAPI btGetReadPosition(
		FLMUINT *					puiPosition);

	RCODE FTKAPI btPositionTo(
		FLMUINT						uiPosition,
		FLMBYTE *					pucKey,
		FLMUINT						uiKeyBufSize,
		FLMUINT *					puiKeyLen);

	RCODE FTKAPI btGetPosition(
		FLMUINT *					puiPosition);

//	RCODE FTKAPI btComputeCounts(
//		IF_BTree *					pUntilBtree,
//		FLMUINT *					puiBlockCount,
//		FLMUINT *					puiKeyCount,
//		FLMBOOL *					pbTotalsEstimated,
//		FLMUINT						uiAvgBlockFullness);

	RCODE FTKAPI btRewind( void);

	FINLINE FLMBOOL FTKAPI btHasCounts( void)
	{
		return( m_bCounts);
	}

	FINLINE FLMBOOL FTKAPI btHasData( void)
	{
		return( m_bTreeHoldsData);
	}

	FINLINE void FTKAPI btResetBtree( void)
	{
		releaseBlocks( TRUE);
		m_bSetupForRead = FALSE;
		m_bSetupForWrite = FALSE;
		m_bSetupForReplace = FALSE;
		m_bOrigInDOBlocks = FALSE;
		m_bDataOnlyBlock = FALSE;
		m_ui32PrimaryBlockAddr = 0;
		m_ui32CurBlockAddr = 0;
		m_uiPrimaryOffset = 0;
		m_uiCurOffset = 0;
		m_uiDataLength = 0;
		m_uiPrimaryDataLen = 0;
		m_uiOADataLength = 0;
		m_uiDataRemaining = 0;
		m_uiOADataRemaining = 0;
		m_uiOffsetAtStart = 0;
		m_uiSearchLevel = F_BTREE_MAX_LEVELS;
	}

	FLMUINT32 FTKAPI getRootBlockAddr( void)
	{
		return( m_ui32RootBlockAddr);
	}

	RCODE btCheck(
		BTREE_ERR_INFO *			pErrInfo);

private:
	
	F_BTree()
	{
		f_assert( 0);
	}

	FINLINE void btRelease( void)
	{
		releaseBlocks( TRUE);
	}

	FINLINE void btSetSearchLevel(
		FLMUINT						uiSearchLevel)
	{
		f_assert( uiSearchLevel <= F_BTREE_MAX_LEVELS);

		btResetBtree();

		m_uiSearchLevel = uiSearchLevel;
	}

	RCODE btMoveBlock(
		FLMUINT32					ui32FromBlockAddr,
		FLMUINT32					ui32ToBlockAddr);

	FINLINE FLMBOOL btDbIsOpen( void)
	{
		return( m_bOpened);
	}

	FINLINE FLMBOOL btIsSetupForRead( void)
	{
		return( m_bSetupForRead);
	}

	FINLINE FLMBOOL btIsSetupForWrite( void)
	{
		return( m_bSetupForWrite);
	}

	FINLINE FLMBOOL btIsSetupForReplace( void)
	{
		return( m_bSetupForReplace);
	}
	
	RCODE btFreeBlockChain(
		FLMUINT					uiStartAddr,
		FLMUINT					uiBlocksToFree,
		FLMUINT *				puiBlocksFreed,
		FLMUINT *				puiEndAddr,
		IF_DeleteStatus *		ifpDeleteStatus);
		
	FINLINE FLMUINT calcEntrySize(
		FLMUINT					uiBlockType,
		FLMUINT					uiFlags,
		FLMUINT					uiKeyLen,
		FLMUINT					uiDataLen,
		FLMUINT					uiOADataLen)
	{
		switch( uiBlockType)
		{
			case F_BLK_TYPE_BT_LEAF:
			{
				return( uiKeyLen + 2);
			}

			case F_BLK_TYPE_BT_LEAF_DATA:
			{
				return( 1 + (uiKeyLen > ONE_BYTE_SIZE ? 2 : 1) +
								(uiDataLen > ONE_BYTE_SIZE ? 2 : 1) +
								(uiOADataLen && (uiFlags & BTE_FLAG_FIRST_ELEMENT) ? 4 : 0) +
								uiKeyLen + uiDataLen);
			}

			case F_BLK_TYPE_BT_NON_LEAF:
			case F_BLK_TYPE_BT_NON_LEAF_COUNTS:
			{
				return( 4 + (uiBlockType == F_BLK_TYPE_BT_NON_LEAF_COUNTS ? 4 : 0) +
						  2 + uiKeyLen);
			}
		}

		return( 0);
	}

	RCODE computeCounts(
		F_BTSK *					pFromStack,
		F_BTSK *					pUntilStack,
		FLMUINT *				puiBlockCount,
		FLMUINT *				puiKeyCount,
		FLMBOOL *				pbTotalsEstimated,
		FLMUINT					uiAvgBlockFullness);

	RCODE blockCounts(
		F_BTSK *					pStack,
		FLMUINT					uiFirstOffset,
		FLMUINT					uiLastOffset,
		FLMUINT *				puiKeyCount,
		FLMUINT *				puiElementCount);

	RCODE getStoredCounts(
		F_BTSK *					pFromStack,
		F_BTSK *					pUntilStack,
		FLMUINT *				puiBlockCount,
		FLMUINT *				puiKeyCount,
		FLMBOOL *				pbTotalsEstimated,
		FLMUINT					uiAvgBlockFullness);

	RCODE getBlocks(
		F_BTSK *					pStack1,
		F_BTSK *					pStack2);

	FINLINE FLMUINT getAvgKeyCount(
		F_BTSK *					pFromStack,
		F_BTSK *					pUntilStack,
		FLMUINT					uiAvgBlockFullness);

	FLMUINT getEntryKeyLength(
		FLMBYTE *				pucEntry,
		FLMUINT					uiBlockType,
		const FLMBYTE **		ppucKeyRV);

	FLMUINT getEntrySize(
		FLMBYTE *				pucBlock,
		FLMUINT					uiOffset,
		FLMBYTE **				ppucEntry = NULL);

	RCODE calcNewEntrySize(
		FLMUINT					uiKeyLen,
		FLMUINT					uiDataLen,
		FLMUINT *				puiEntrySize,
		FLMBOOL *				pbHaveRoom,
		FLMBOOL *				pbDefragBlock);

	RCODE extractEntryData(
		FLMBYTE *				pucKey,
		FLMUINT					uiKeyLen,
		FLMBYTE *				pucBuffer,
		FLMUINT					uiBufSiz,
		FLMUINT *				puiDataLen,
		FLMBYTE **				ppucDataPtr);

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
		FLMUINT *				puiChildBlockAddr,
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
		FLMUINT					uiChildBlockAddr,
		FLMUINT					uiCounts,
		FLMUINT					uiEntrySize,
		FLMBOOL *				pbLastEntry);

	RCODE removeEntry(
		const FLMBYTE **		ppucKey,
		FLMUINT *				puiKeyLen,
		FLMUINT *				puiChildBlockAddr,
		FLMUINT *				puiCounts,
		FLMBOOL *				pbMoreToRemove,
		F_ELM_UPD_ACTION *	peAction);

	RCODE remove(
		FLMBOOL					bDeleteDOBlocks);

	RCODE removeRange(
		FLMUINT					uiStartElm,
		FLMUINT					uiEndElm,
		FLMBOOL					bDeleteDOBlocks);

	RCODE findEntry(
		const FLMBYTE *		pucKey,
		FLMUINT					uiKeyLen,
		FLMUINT					uiMatch,
		FLMUINT *				puiPosition = NULL,
		FLMUINT32 *				pui32BlockAddr = NULL,
		FLMUINT *				puiOffsetIndex = NULL);

	RCODE findInBlock(
		const FLMBYTE *		pucKey,
		FLMUINT					uiKeyLen,
		FLMUINT					uiMatch,
		FLMUINT *				uiPosition,
		FLMUINT32 *				ui32BlockAddr,
		FLMUINT *				uiOffsetIndex);

	RCODE scanBlock(
		F_BTSK *					pStack,
		FLMUINT					uiMatch);

	RCODE compareKeys(
		const FLMBYTE *		pucKey1,
		FLMUINT					uiKeyLen1,
		const FLMBYTE *		pucKey2,
		FLMUINT					uiKeyLen2,
		FLMINT *					piCompare);

	FINLINE RCODE compareBlockKeys(
		const FLMBYTE *		pucBlockKey,
		FLMUINT					uiBlockKeyLen,
		const FLMBYTE *		pucTargetKey,
		FLMUINT					uiTargetKeyLen,
		FLMINT *					piCompare)
	{
		f_assert( uiBlockKeyLen);

		if( !m_pCompare && uiBlockKeyLen == uiTargetKeyLen)
		{
			*piCompare = f_memcmp( pucBlockKey, pucTargetKey, uiBlockKeyLen);
									
			return( NE_FLM_OK);
		}

		return( compareKeys( pucBlockKey, uiBlockKeyLen,
							pucTargetKey, uiTargetKeyLen, piCompare));
	}

	RCODE positionToEntry(
		FLMUINT					uiPosition);

	RCODE searchBlock(
		FLMBYTE *				pucBlock,
		FLMUINT *				puiPrevCounts,
		FLMUINT					uiPosition,
		FLMUINT *				puiOffset);

	RCODE defragmentBlock(
		IF_Block **				ppBlock,
		FLMBYTE **				ppucBlock);

	RCODE advanceToNextElement(
		FLMBOOL					bAdvanceStack);

	RCODE backupToPrevElement(
		FLMBOOL					bBackupStack);

	RCODE replaceEntry(
		const FLMBYTE **		ppucKey,
		FLMUINT *				puiKeyLen,
		const FLMBYTE *		pucValue,
		FLMUINT					uiLen,
		FLMUINT					uiFlags,
		FLMUINT *				puiChildBlockAddr,
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
		FLMUINT *				puiChildBlockAddr,
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
		FLMUINT *				puiChildBlockAddr,
		FLMUINT *				puiCounts,
		const FLMBYTE **		ppucRemainingValue,
		FLMUINT *				puiRemainingLen,
		F_ELM_UPD_ACTION *	peAction);

	RCODE replace(
		FLMBYTE *				pucEntry,
		FLMUINT					uiEntrySize,
		FLMBOOL *				pbLastEntry);

	RCODE buildAndStoreEntry(
		FLMUINT					uiBlockType,
		FLMUINT					uiFlags,
		const FLMBYTE *		pucKey,
		FLMUINT					uiKeyLen,
		const FLMBYTE *		pucData,
		FLMUINT					uiDataLen,
		FLMUINT					uiOADataLen,
		FLMUINT					uiChildBlockAddr,
		FLMUINT					uiCounts,
		FLMBYTE *				pucBuffer,
		FLMUINT					uiBufferSize,
		FLMUINT *				puiEntrySize);

	RCODE moveEntriesToPrevBlock(
		FLMUINT					uiNewEntrySize,
		IF_Block **				ppPrevBlock,
		FLMBYTE **				ppucPrevBlock,
		FLMBOOL *				pbEntriesWereMoved);

	RCODE moveEntriesToNextBlock(
		FLMUINT					uiEntrySize,
		FLMBOOL *				pbEntriesWereMoved);

	RCODE splitBlock(
		const FLMBYTE *		pucKey,
		FLMUINT					uiKeyLen,
		const FLMBYTE *		pucValue,
		FLMUINT					uiLen,
		FLMUINT					uiFlags,
		FLMUINT					uiOADataLen,
		FLMUINT 					uiChildBlockAddr,
		FLMUINT					uiCounts,
		const FLMBYTE **		ppucRemainingValue,
		FLMUINT *				puiRemainingLen,
		FLMBOOL *				pbBlockSplit);

	RCODE createNewLevel( void);

	RCODE storeDataOnlyBlocks(
		const FLMBYTE *		pucKey,
		FLMUINT					uiKeyLen,
		FLMBOOL					bSaveKey,
		const FLMBYTE *		pucData,
		FLMUINT					uiDataLen);

	RCODE replaceDataOnlyBlocks(
		const FLMBYTE *		pucKey,
		FLMUINT					uiKeyLen,
		FLMBOOL					bSaveKey,
		const FLMBYTE *		pucData,
		FLMUINT					uiDataLen,
		FLMBOOL					bLast,
		FLMBOOL					bTruncate = TRUE);

	RCODE moveToPrev(
		FLMUINT					uiStart,
		FLMUINT					uiFinish,
		IF_Block **				ppPrevBlock,
		FLMBYTE **				ppucPrevBlock);

	RCODE moveToNext(
		FLMUINT					uiStart,
		FLMUINT					uiFinish,
		IF_Block **				ppNextBlock,
		FLMBYTE **				ppucNextBlock);

	RCODE updateParentCounts(
		FLMBYTE *				pucChildBlock,
		IF_Block **				ppParentBlock,
		FLMBYTE **				ppucParentBlock,
		FLMUINT					uiParentElm);

	FLMUINT countKeys(
		FLMBYTE *				pucBlock);

	FLMUINT countRangeOfKeys(
		F_BTSK *					pFromStack,
		FLMUINT					uiFromOffset,
		FLMUINT					uiUntilOffset);

	RCODE moveStackToPrev(
		IF_Block *				pPrevBlock,
		FLMBYTE *				pucPrevBlock);

	RCODE moveStackToNext(
		IF_Block *				pBlock,
		FLMBYTE *				pucBlock);

	RCODE calcOptimalDataLength(
		FLMUINT					uiKeyLen,
		FLMUINT					uiDataLen,
		FLMUINT					uiBytesAvail,
		FLMUINT *				puiNewDataLen);

	RCODE verifyDOBlockChain(
		FLMUINT					uiDOAddr,
		FLMUINT					uiDataLength,
		BTREE_ERR_INFO *		localErrInfo);

	RCODE verifyCounts(
		BTREE_ERR_INFO *		pErrInfo);

	void releaseBlocks(
		FLMBOOL					bResetStack);

	void releaseBtree( void);

	RCODE saveReplaceInfo(
		const FLMBYTE *		pucKey,
		FLMUINT					uiKeyLen);

	RCODE restoreReplaceInfo(
		const FLMBYTE **		ppucKey,
		FLMUINT *				puiKeyLen,
		FLMUINT *				puiChildBlockAddr,
		FLMUINT *				puiCounts);

	FINLINE RCODE setReturnKey(
		FLMBYTE *				pucEntry,
		FLMUINT					uiBlockType,
		FLMBYTE *				pucKey,
		FLMUINT *				puiKeyLen,
		FLMUINT					uiKeyBufSize);

	RCODE setupReadState(
		FLMBYTE *				pucBlock,
		FLMBYTE *				pucEntry);

	RCODE removeRemainingEntries(
		const FLMBYTE *		pucKey,
		FLMUINT					uiKeyLen);

	RCODE deleteEmptyBlock( void);

	RCODE removeDOBlocks(
		FLMUINT32				ui32OrigDOAddr);

	RCODE replaceMultiples(
		const FLMBYTE **		ppucKey,
		FLMUINT *				puiKeyLen,
		const FLMBYTE *		pucDataValue,
		FLMUINT					uiLen,
		const FLMBYTE **		ppucRemainingValue,
		FLMUINT *				puiRemainingLen,
		F_ELM_UPD_ACTION *	peAction);

	RCODE replaceMultiNoTruncate(
		const FLMBYTE **		ppucKey,
		FLMUINT *				puiKeyLen,
		const FLMBYTE *		pucDataValue,
		FLMUINT					uiLen,
		const FLMBYTE **		ppucRemainingValue,
		FLMUINT *				puiRemainingLen,
		F_ELM_UPD_ACTION *	peAction);

	RCODE getNextBlock(
		IF_Block **				ppBlock,
		FLMBYTE **				ppucBlock);

	RCODE getPrevBlock(
		IF_Block **				ppBlock,
		FLMBYTE **				ppucBlock);

	FLMBOOL checkContinuedEntry(
		const FLMBYTE *		pucKey,
		FLMUINT					uiKeyLen,
		FLMBOOL *				pbLastElement,
		FLMBYTE *				pucEntry,
		FLMUINT					uiBlockType);

	RCODE updateCounts( void);

	RCODE storePartialEntry(
		const FLMBYTE *		pucKey,
		FLMUINT					uiKeyLen,
		const FLMBYTE *		pucValue,
		FLMUINT					uiLen,
		FLMUINT					uiFlags,
		FLMUINT					uiChildBlockAddr,
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
		IF_Block **				ppFromBlock,
		FLMBYTE **				ppucFromBlock,
		IF_Block **				ppToBlock,
		FLMBYTE **				ppucToBlock);

	RCODE checkDownLinks( void);

	RCODE verifyChildLinks(
		FLMBYTE *				pucParentBlock);

	RCODE combineEntries(
		FLMBYTE *				pucSrcBlock,
		FLMUINT					uiSrcOffset,
		FLMBYTE *				pucDstBlock,
		FLMUINT					uiDstOffset,
		FLMBOOL *				pbEntriesCombined,
		FLMUINT *				puiEntrySize,
		FLMBYTE *				pucTempBlock);

	RCODE moveBtreeBlock(
		FLMUINT32				ui32FromBlockAddr,
		FLMUINT32				ui32ToBlockAddr);

	RCODE moveDOBlock(
		FLMUINT32				ui32FromBlockAddr,
		FLMUINT32				ui32ToBlockAddr);

	IF_BlockMgr *				m_pBlockMgr;
	F_Pool						m_pool;
	FLMBOOL						m_bCounts;
	FLMBOOL						m_bTreeHoldsData;
	FLMBOOL						m_bSetupForRead;
	FLMBOOL						m_bSetupForWrite;
	FLMBOOL						m_bSetupForReplace;
	FLMBOOL						m_bOpened;
	FLMBOOL						m_bDataOnlyBlock;
	FLMBOOL						m_bOrigInDOBlocks;
	FLMBOOL						m_bFirstRead;
	FLMBOOL						m_bStackSetup;
	F_BTSK *						m_pStack;
	BTREE_REPLACE_STRUCT *	m_pReplaceInfo;
	BTREE_REPLACE_STRUCT *	m_pReplaceStruct;
	IF_Block *					m_pBlock;
	FLMBYTE *					m_pucBlock;
	FLMUINT						m_uiBlockSize;
	FLMUINT						m_uiDefragThreshold;
	FLMUINT						m_uiOverflowThreshold;
	FLMUINT						m_uiStackLevels;
	FLMUINT						m_uiRootLevel;
	FLMUINT						m_uiReplaceLevels;
	FLMUINT						m_uiDataLength;
	FLMUINT						m_uiPrimaryDataLen;
	FLMUINT						m_uiOADataLength;
	FLMUINT						m_uiDataRemaining;
	FLMUINT						m_uiOADataRemaining;
	FLMUINT						m_uiPrimaryOffset;
	FLMUINT						m_uiCurOffset;
	FLMUINT						m_uiSearchLevel;
	FLMUINT						m_uiOffsetAtStart;
	FLMUINT32					m_ui32RootBlockAddr;
	FLMUINT32					m_ui32PrimaryBlockAddr;
	FLMUINT32					m_ui32DOBlockAddr;
	FLMUINT32					m_ui32CurBlockAddr;
	F_BTSK						m_Stack[ F_BTREE_MAX_LEVELS];
	IF_ResultSetCompare *	m_pCompare;
};

/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMBOOL bteKeyLenFlag( 
	FLMBYTE *			pucEntry)
{
	return( (pucEntry[ BTE_FLAG] & BTE_FLAG_KEY_LEN) ? TRUE : FALSE);
}
	
/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMBOOL bteDataLenFlag( 
	FLMBYTE *			pucEntry)
{
	return( (pucEntry[ BTE_FLAG] & BTE_FLAG_DATA_LEN) ? TRUE : FALSE);
}
	
/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMBOOL bteOADataLenFlag( 
	FLMBYTE *			pucEntry)
{
	return( (pucEntry[ BTE_FLAG] & BTE_FLAG_OA_DATA_LEN) ? TRUE : FALSE);
}
	
/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMBOOL bteDataBlockFlag( 
	FLMBYTE *			pucEntry)
{
	return( (pucEntry[ BTE_FLAG] & BTE_FLAG_DATA_BLOCK) ? TRUE : FALSE);
}
	
/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMBOOL bteFirstElementFlag(
	FLMBYTE *			pucEntry)
{
	return( (pucEntry[ BTE_FLAG] & BTE_FLAG_FIRST_ELEMENT) ? TRUE : FALSE);
}
	
/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMBOOL bteLastElementFlag(
	FLMBYTE *			pucEntry)
{
	return( (pucEntry[ BTE_FLAG] & BTE_FLAG_LAST_ELEMENT) ? TRUE : FALSE);
}
	
/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMUINT32 bteGetBlockAddr(
	const FLMBYTE *	pucEntry)
{
	return( FB2UD( pucEntry));
}
	
/***************************************************************************
Desc:
****************************************************************************/
FINLINE void bteSetEntryOffset(
	FLMUINT16 *			pui16OffsetArray,
	FLMUINT				uiOffsetIndex,
	FLMUINT				uiOffset)
{
	UW2FBA( (FLMUINT16)uiOffset, 
		(FLMBYTE *)&pui16OffsetArray[ uiOffsetIndex]);
}
	
/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMUINT16 bteGetEntryOffset( 
	const FLMUINT16 *	pui16OffsetArray,
	FLMUINT				uiOffsetIndex)
{
	return( FB2UW( (FLMBYTE *)&pui16OffsetArray[ uiOffsetIndex]));
}
	
/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMBOOL isRootBlock(
	FLMBYTE *			pucBlock)
{
	return( (((F_BTREE_BLK_HDR *)pucBlock)->ui8BTreeFlags & F_BTREE_BLK_IS_ROOT) 
						? TRUE 
						: FALSE);
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE void setRootBlock(
	FLMBYTE *			pucBlock)
{
	((F_BTREE_BLK_HDR *)pucBlock)->ui8BTreeFlags |= F_BTREE_BLK_IS_ROOT;
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE void unsetRootBlock(
	FLMBYTE *			pucBlock)
{
	((F_BTREE_BLK_HDR *)pucBlock)->ui8BTreeFlags &= (~(F_BTREE_BLK_IS_ROOT));
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMUINT16 getNumKeys(
	FLMBYTE *			pucBlock)
{
	return( (((F_BTREE_BLK_HDR *)pucBlock)->ui16NumKeys));
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE void setNumKeys(
	FLMBYTE *			pucBlock,
	FLMUINT16			ui16NumKeys)
{
	((F_BTREE_BLK_HDR *)pucBlock)->ui16NumKeys = ui16NumKeys;
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE void incNumKeys(
	FLMBYTE *			pucBlock)
{
	((F_BTREE_BLK_HDR *)pucBlock)->ui16NumKeys++;
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE void decNumKeys(
	FLMBYTE *			pucBlock)
{
	((F_BTREE_BLK_HDR *)pucBlock)->ui16NumKeys--;
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMUINT16 getHeapSize(
	FLMBYTE *			pucBlock)
{
	return( (((F_BTREE_BLK_HDR *)pucBlock)->ui16HeapSize));
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE void setHeapSize(
	FLMBYTE *			pucBlock,
	FLMUINT16			ui16HeapSize)
{
	((F_BTREE_BLK_HDR *)pucBlock)->ui16HeapSize = ui16HeapSize;
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE void incHeapSize(
	FLMBYTE *			pucBlock,
	FLMUINT				uiIncAmount)
{
	((F_BTREE_BLK_HDR *)pucBlock)->ui16HeapSize += (FLMUINT16)uiIncAmount;
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE void decHeapSize(
	FLMBYTE *			pucBlock,
	FLMUINT				uiDecAmount)
{
	((F_BTREE_BLK_HDR *)pucBlock)->ui16HeapSize -= (FLMUINT16)uiDecAmount;
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMUINT8 getBTreeFlags(
	FLMBYTE *			pucBlock)
{
	return( (((F_BTREE_BLK_HDR *)pucBlock)->ui8BTreeFlags));
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE void setBTreeFlags(
	FLMBYTE *			pucBlock,
	FLMUINT8				ui8BTreeFlags)
{
	((F_BTREE_BLK_HDR *)pucBlock)->ui8BTreeFlags = ui8BTreeFlags;
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMUINT16 getBTreeId(
	FLMBYTE *			pucBlock)
{
	return( (((F_BTREE_BLK_HDR *)pucBlock)->ui16BtreeId));
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE void setBTreeId(
	FLMBYTE *			pucBlock,
	FLMUINT16			ui16BtreeId)
{
	((F_BTREE_BLK_HDR *)pucBlock)->ui16BtreeId = ui16BtreeId;
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMUINT8 getBlockLevel(
	FLMBYTE *			pucBlock)
{
	return( ((F_BTREE_BLK_HDR *)pucBlock)->ui8BlockLevel);
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE void setBlockLevel(
	FLMBYTE *			pucBlock,
	FLMUINT8				ui8BlockLevel)
{
	((F_BTREE_BLK_HDR *)pucBlock)->ui8BlockLevel = ui8BlockLevel;
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE void incBlockLevel(
	FLMBYTE *			pucBlock)
{
	((F_BTREE_BLK_HDR *)pucBlock)->ui8BlockLevel++;
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMUINT8 getBlockFlags(
	FLMBYTE *			pucBlock)
{
	return( ((F_STD_BLK_HDR *)pucBlock)->ui8BlockFlags);
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE void setBlockFlags(
	FLMBYTE *			pucBlock,
	FLMUINT8				ui8BlockFlags)
{
	((F_STD_BLK_HDR *)pucBlock)->ui8BlockFlags = ui8BlockFlags;
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE void setBlockEncrypted(
	FLMBYTE *			pucBlock)
{
	((F_STD_BLK_HDR *)pucBlock)->ui8BlockFlags |= F_BLK_IS_ENCRYPTED;
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE void unsetBlockEncrypted(
	FLMBYTE *			pucBlock)
{
	((F_STD_BLK_HDR *)pucBlock)->ui8BlockFlags &= (~(F_BLK_IS_ENCRYPTED));
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMBOOL isEncryptedBlock(
	FLMBYTE *			pucBlock)
{
	return( (((F_STD_BLK_HDR *)pucBlock)->ui8BlockFlags & F_BLK_IS_ENCRYPTED) 
						? TRUE 
						: FALSE);
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMBYTE getBlockType(
	FLMBYTE *			pucBlock)
{
	return( (((F_STD_BLK_HDR *)pucBlock)->ui8BlockType));
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE void setBlockType(
	FLMBYTE *			pucBlock,
	FLMUINT8				ui8BlockType)
{
	((F_STD_BLK_HDR *)pucBlock)->ui8BlockType = ui8BlockType;
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMUINT32 getBlockAddr(
	FLMBYTE *			pucBlock)
{
	return( (((F_STD_BLK_HDR *)pucBlock)->ui32BlockAddr));
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE void setBlockAddr(
	FLMBYTE *			pucBlock,
	FLMUINT32			ui32BlockAddr)
{
	((F_STD_BLK_HDR *)pucBlock)->ui32BlockAddr = ui32BlockAddr;
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMUINT32 getPrevInChain(
	FLMBYTE *			pucBlock)
{
	return( (((F_STD_BLK_HDR *)pucBlock)->ui32PrevBlockInChain));
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE void setPrevInChain(
	FLMBYTE *			pucBlock,
	FLMUINT32			ui32PrevInChain)
{
	((F_STD_BLK_HDR *)pucBlock)->ui32PrevBlockInChain = ui32PrevInChain;
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMUINT32 getNextInChain(
	FLMBYTE *			pucBlock)
{
	return( (((F_STD_BLK_HDR *)pucBlock)->ui32NextBlockInChain));
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE void setNextInChain(
	FLMBYTE *			pucBlock,
	FLMUINT32			ui32NextInChain)
{
	((F_STD_BLK_HDR *)pucBlock)->ui32NextBlockInChain = ui32NextInChain;
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMUINT16 getBytesAvail(
	FLMBYTE *			pucBlock)
{
	return( (((F_STD_BLK_HDR *)pucBlock)->ui16BlockBytesAvail));
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE void setBytesAvail(
	FLMBYTE *			pucBlock,
	FLMUINT16			ui16BytesAvail)
{
	((F_STD_BLK_HDR *)pucBlock)->ui16BlockBytesAvail = ui16BytesAvail;
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE void incBytesAvail(
	FLMBYTE *			pucBlock,
	FLMUINT				uiIncAmount)
{
	((F_STD_BLK_HDR *)pucBlock)->ui16BlockBytesAvail += 
		(FLMUINT16)uiIncAmount;
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE void decBytesAvail(
	FLMBYTE *			pucBlock,
	FLMUINT				uiDecAmount)
{
	((F_STD_BLK_HDR *)pucBlock)->ui16BlockBytesAvail -= 
		(FLMUINT16)uiDecAmount;
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMUINT sizeofDOBlockHdr(
	FLMBYTE *			pucBlock)
{
	if( !isEncryptedBlock( pucBlock))
	{
		return( sizeof( F_STD_BLK_HDR));
	}
	else
	{
		return( sizeof( F_ENC_DO_BLK_HDR));
	}
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMUINT sizeofBTreeBlockHdr(
	FLMBYTE *			pucBlock)
{
	if( !isEncryptedBlock( pucBlock))
	{
		return( sizeof( F_BTREE_BLK_HDR));
	}
	else
	{
		return( sizeof( F_ENC_BTREE_BLK_HDR));
	}
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMBYTE * BtEntry(
	FLMBYTE *			pucBlock,
	FLMUINT				uiIndex)
{
	FLMBYTE *	pucOffsetArray;
	
	pucOffsetArray = pucBlock + sizeofBTreeBlockHdr( pucBlock) + (uiIndex * 2);
	return( pucBlock + FB2UW( pucOffsetArray));
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMBYTE * BtLastEntry(
	FLMBYTE *			pucBlock)
{
	return( BtEntry( pucBlock, getNumKeys( pucBlock) - 1));
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMUINT16 * BtOffsetArray(
	FLMBYTE *			pucBlock,
	FLMUINT				uiIndex)
{
	return( (FLMUINT16 *)(pucBlock + sizeofBTreeBlockHdr( pucBlock) + 
			(uiIndex * sizeof( FLMUINT16))));
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMUINT actualEntrySize(
	FLMUINT 			uiEntrySize)
{
	return( uiEntrySize - 2);
}

/***************************************************************************
Desc:
****************************************************************************/
FINLINE FLMBYTE * getBlockEnd(
	FLMBYTE *			pucBlock)
{
	return( pucBlock + sizeofBTreeBlockHdr( pucBlock) +
			  (getNumKeys( pucBlock) * 2) + getHeapSize( pucBlock));
}

/****************************************************************************
Desc:	Test to see if a block type is a B-Tree block type.
****************************************************************************/
FINLINE FLMBOOL blkIsBTree(
	FLMBYTE *			pucBlock)
{
	FLMBYTE		ucBlockType = getBlockType( pucBlock);
	
	if( ucBlockType == F_BLK_TYPE_BT_LEAF ||
		 ucBlockType == F_BLK_TYPE_BT_NON_LEAF ||
		 ucBlockType == F_BLK_TYPE_BT_NON_LEAF_COUNTS ||
		 ucBlockType == F_BLK_TYPE_BT_LEAF_DATA ||
		 ucBlockType == F_BLK_TYPE_BT_DATA_ONLY)
	{
		return( TRUE);
	}
	
	return( FALSE);
}

/****************************************************************************
Desc:
****************************************************************************/
FINLINE FLMBOOL blkIsLeaf(
	FLMBYTE *			pucBlock)
{
	FLMBYTE		ucBlockType = getBlockType( pucBlock);

	if( ucBlockType == F_BLK_TYPE_BT_LEAF ||
		 ucBlockType == F_BLK_TYPE_BT_LEAF_DATA)
	{
		return( TRUE);
	}

	return( FALSE);
}

/***************************************************************************
Desc:
****************************************************************************/
F_BTree::F_BTree(
	IF_BlockMgr *		pBlockMgr)
{
	m_pBlockMgr = pBlockMgr;
	pBlockMgr->AddRef();
	
	m_pool.poolInit( 4096);
	m_bOpened = FALSE;
	m_ui32RootBlockAddr = 0;
	m_pStack = NULL;
	m_uiStackLevels = 0;
	m_uiRootLevel = 0;
	f_memset( m_Stack, 0, sizeof(m_Stack));
	m_bCounts = FALSE;
	m_bTreeHoldsData = TRUE;
	m_bSetupForRead = FALSE;
	m_bSetupForWrite = FALSE;
	m_bSetupForReplace = FALSE;
	m_uiBlockSize = 0;
	m_uiDefragThreshold = 0;
	m_uiOverflowThreshold = 0;
	m_pReplaceInfo = NULL;
	m_pReplaceStruct = NULL;
	m_uiReplaceLevels = 0;
	m_uiDataLength = 0;
	m_uiPrimaryDataLen = 0;
	m_uiOADataLength = 0;
	m_uiDataRemaining = 0;
	m_uiOADataRemaining = 0;
	m_uiOffsetAtStart = 0;
	m_bDataOnlyBlock = FALSE;
	m_bOrigInDOBlocks = FALSE;
	m_ui32PrimaryBlockAddr = 0;
	m_uiPrimaryOffset = 0;
	m_ui32DOBlockAddr = 0;
	m_ui32CurBlockAddr = 0;
	m_uiCurOffset = 0;
	m_bFirstRead = FALSE;
	m_pBlock = NULL;
	m_pucBlock = NULL;
	m_uiSearchLevel = F_BTREE_MAX_LEVELS;
	m_pCompare = NULL;
}

/***************************************************************************
Desc:
****************************************************************************/
F_BTree::~F_BTree( void)
{
	if( m_bOpened)
	{
		btClose();
	}
	
	if( m_pBlockMgr)
	{
		m_pBlockMgr->Release();
	}
	
	m_pool.poolFree();
}

/***************************************************************************
Desc: Function to create a new (empty) B-Tree.  To do this, we create the
		root block.
****************************************************************************/
RCODE F_BTree::btCreate(
	FLMUINT16					ui16BtreeId,
	FLMBOOL						bCounts,
	FLMBOOL						bData,
	FLMUINT32 *					pui32RootBlockAddr,
	IF_ResultSetCompare *	pCompare)
{
	RCODE						rc = NE_FLM_OK;
	IF_Block *				pBlock = NULL;
	FLMBYTE *				pucBlock = NULL;
	FLMUINT16 *				pui16Offset;
	FLMBYTE *				pucEntry;
	FLMBYTE					ucLEMEntry[ 3];
	FLMUINT					uiFlags = 0;
	FLMUINT					uiLEMSize;
	FLMUINT32				ui32RootBlockAddr = 0;

	// We can't create a new Btree if we have already been initialized.
	
	if (m_bOpened)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_BAD_STATE);
		goto Exit;
	}

	// Call createBlock to create a new block
	
	if (RC_BAD( rc = m_pBlockMgr->createBlock( &pBlock, 
		&pucBlock, &ui32RootBlockAddr)))
	{
		goto Exit;
	}

	setBlockAddr( pucBlock, ui32RootBlockAddr);

	// Save the block address and identify the block as the root block.
	
	if( RC_BAD( rc = btOpen( ui32RootBlockAddr, bCounts, bData, pCompare)))
	{
		goto Exit;
	}
	
	setRootBlock( pucBlock);
	setBTreeId( pucBlock, ui16BtreeId);
	setBlockLevel( pucBlock, 0);
	setBlockType( pucBlock, (bData ? F_BLK_TYPE_BT_LEAF_DATA : F_BLK_TYPE_BT_LEAF));
	setPrevInChain( pucBlock, 0);
	setNextInChain( pucBlock, 0);
	
//	if (pLFile->uiEncId)
//	{
//		setBlockEncrypted( (F_STD_BLK_HDR *)pBlockHdr);
//	}

	// Insert a LEM into the block
	
	uiFlags = BTE_FLAG_FIRST_ELEMENT | BTE_FLAG_LAST_ELEMENT;

	if (RC_BAD( rc = buildAndStoreEntry( 
		(bData ? F_BLK_TYPE_BT_LEAF_DATA : F_BLK_TYPE_BT_LEAF),
		uiFlags, NULL, 0, NULL, 0, 0, 0, 0, &ucLEMEntry[0],
		3, &uiLEMSize)))
	{
		goto Exit;
	}

	pui16Offset = BtOffsetArray( pucBlock, 0);
	pucEntry = pucBlock + m_uiBlockSize - uiLEMSize;

	bteSetEntryOffset( pui16Offset, 0, (FLMUINT16)(pucEntry - pucBlock));
	f_memcpy( pucEntry, ucLEMEntry, uiLEMSize);

	// Offset Entry and 2 byte LEM
	
	setBytesAvail( pucBlock, 
		(FLMUINT16)(m_uiBlockSize - sizeofBTreeBlockHdr( pucBlock) -
		uiLEMSize - 2));
	setHeapSize( pucBlock, getBytesAvail( pucBlock));

	// There is one entry now.
	
	setNumKeys( pucBlock, 1);
	
	// Return the root block address
	
	if( pui32RootBlockAddr)
	{
		*pui32RootBlockAddr = ui32RootBlockAddr;
	}

Exit:

	if( pBlock)
	{
		pBlock->Release();
	}

	return( rc);
}

/***************************************************************************
Desc: Btree initialization function.
****************************************************************************/
RCODE F_BTree::btOpen(
	FLMUINT32					ui32RootBlockAddr,
 	FLMBOOL						bCounts,
	FLMBOOL						bData,
	IF_ResultSetCompare *	pCompare)
{
	RCODE				rc = NE_FLM_OK;

	if( m_bOpened)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_BAD_STATE);
		goto Exit;
	}

	if( !ui32RootBlockAddr)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_FAILURE);
		goto Exit;
	}

	m_uiBlockSize = m_pBlockMgr->getBlockSize();
	m_ui32RootBlockAddr = ui32RootBlockAddr;
	m_uiDefragThreshold = m_uiBlockSize / 20;
	m_uiOverflowThreshold = (m_uiBlockSize * 8) / 5;
	m_bCounts = bCounts;
	m_bTreeHoldsData = bData;
	m_pReplaceInfo = NULL;
	m_uiReplaceLevels = 0;
	m_uiSearchLevel = F_BTREE_MAX_LEVELS;

	m_bSetupForRead = FALSE;
	m_bSetupForWrite = FALSE;
	m_bSetupForReplace = FALSE;
	
	m_pool.poolFree();
	m_pool.poolInit( m_uiBlockSize);
	
	if( RC_BAD( rc = m_pool.poolAlloc( 
		sizeof( BTREE_REPLACE_STRUCT) * F_BTREE_MAX_LEVELS, 
		(void **)&m_pReplaceStruct)))
	{
		goto Exit;
	}

	f_assert( !m_pCompare);
	if ((m_pCompare = pCompare) != NULL)
	{
		m_pCompare->AddRef();
	}

	m_bOpened = TRUE;
	
Exit:

	return( rc);
}

/***************************************************************************
Desc: Btree close function
****************************************************************************/
void F_BTree::btClose()
{
	FLMUINT			uiLoop;

	if( !m_bOpened)
	{
		return;
	}

	for (uiLoop = 0; uiLoop < F_BTREE_MAX_LEVELS; uiLoop++)
	{
		m_Stack[ uiLoop].pucKeyBuf = NULL;
		m_Stack[ uiLoop].uiKeyBufSize = 0;
	}

	btRelease();

	if( m_pBlock)
	{
		f_assert( 0);
		m_pBlock->Release();
		m_pBlock = NULL;
	}
	
	m_pucBlock = NULL;
	
	if( m_pCompare)
	{
		m_pCompare->Release();
		m_pCompare = NULL;
	}
	
	m_pool.poolFree();
	m_pool.poolInit( 4096);	

	m_ui32RootBlockAddr = 0;
	m_bOpened = FALSE;
}

/***************************************************************************
Desc: Delete the entire tree
****************************************************************************/
RCODE F_BTree::btDeleteTree(
	IF_DeleteStatus *		ifpDeleteStatus)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiNumLevels;
	FLMUINT		puiBlockAddrs[ F_BTREE_MAX_LEVELS];
	FLMUINT		uiLoop;

	f_assert( m_bOpened);

	// Fill up uiBlockAddrs and calculate the number of levels.

	if( RC_BAD( rc = btGetBlockChains( puiBlockAddrs, &uiNumLevels)))
	{
		goto Exit;
	}

	// Iterate over the list of block chains and free all of the blocks

	for( uiLoop = 0; uiLoop < uiNumLevels; uiLoop++)
	{
		if( RC_BAD( rc = btFreeBlockChain( 
			puiBlockAddrs[ uiLoop], 0, NULL, NULL, ifpDeleteStatus)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE F_BTree::btFreeBlockChain(
	FLMUINT					uiStartAddr,
	FLMUINT					uiBlocksToFree,
	FLMUINT *				puiBlocksFreed,
	FLMUINT *				puiEndAddr,
	IF_DeleteStatus *		ifpDeleteStatus)
{
	RCODE					rc = NE_FLM_OK;
	IF_Block *			pCurrentBlock = NULL;
	IF_Block *			pDOBlock = NULL;
	FLMBYTE *			pucCurrentBlock = NULL;
	FLMBYTE *			pucDOBlock;
	FLMBYTE *			pucEntry;
	FLMUINT				uiEntryNum;
	FLMUINT				uiDOBlockAddr;
	FLMBYTE				ucDOBlockAddr[ 4];
	FLMUINT				uiStatusCounter = 0;
	FLMUINT				uiNextBlockAddr = 0;
	FLMUINT				uiCurrentBlockAddr = 0;
	FLMUINT				uiTreeBlocksFreed = 0;
	FLMUINT				uiDataBlocksFreed = 0;
	FLMBOOL				bFreeAll = FALSE;

	if( !uiBlocksToFree)
	{
		bFreeAll = TRUE;
	}

	// Now, go through the chain and delete the blocks...

	uiCurrentBlockAddr = uiStartAddr;
	while( uiCurrentBlockAddr)
	{
		if( !bFreeAll && uiTreeBlocksFreed >= uiBlocksToFree)
		{
			break;
		}

		if( RC_BAD( m_pBlockMgr->getBlock( (FLMUINT32)uiCurrentBlockAddr, 
			&pCurrentBlock, &pucCurrentBlock)))
		{
			goto Exit;
		}

		uiNextBlockAddr = getNextInChain( pucCurrentBlock);

		// If this is a leaf block, then there may be entries 
		// with data-only references that will need to be cleaned up too.

		if( getBlockType( pucCurrentBlock) == F_BLK_TYPE_BT_LEAF_DATA)
		{
			for( uiEntryNum = 0;
				  uiEntryNum < getNumKeys( pucCurrentBlock);
				  uiEntryNum++)
			{
				pucEntry = BtEntry( pucCurrentBlock, uiEntryNum);

				if( bteDataBlockFlag( pucEntry))
				{
					// Get the data-only block address

					if( RC_BAD( rc = fbtGetEntryData( 
						pucEntry, &ucDOBlockAddr[ 0], 4, NULL)))
					{
						goto Exit;
					}

					uiDOBlockAddr = bteGetBlockAddr( &ucDOBlockAddr[ 0]);
					while( uiDOBlockAddr)
					{
						if( RC_BAD( rc = m_pBlockMgr->getBlock( 
							(FLMUINT32)uiDOBlockAddr, &pDOBlock, &pucDOBlock)))
						{
							goto Exit;
						}

						uiDOBlockAddr = getNextInChain( pucDOBlock);
						if( RC_BAD( rc = m_pBlockMgr->freeBlock( 
							&pDOBlock, &pucDOBlock)))
						{
							goto Exit;
						}
						
						uiDataBlocksFreed++;
					}
				}
			}
		}

		if( RC_BAD( rc = m_pBlockMgr->freeBlock( &pCurrentBlock, &pucCurrentBlock)))
		{
			goto Exit;
		}
		
		if( ifpDeleteStatus && ++uiStatusCounter >= 25)
		{
			uiStatusCounter = 0;
			if( RC_BAD( rc = ifpDeleteStatus->reportDelete( 
					uiTreeBlocksFreed + uiDataBlocksFreed, m_uiBlockSize)))
			{
				goto Exit;
			}
		}

		uiTreeBlocksFreed++;
		uiCurrentBlockAddr = uiNextBlockAddr;
	}

	if( puiBlocksFreed)
	{
		*puiBlocksFreed = uiTreeBlocksFreed;
	}

	if( puiEndAddr)
	{
		*puiEndAddr = uiCurrentBlockAddr;
	}

Exit:

	if( pDOBlock)
	{
		pDOBlock->Release();
	}

	if( pCurrentBlock)
	{
		pCurrentBlock->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:	Returns the address of the first block at each level of the tree
Note:	puiBlockAddrs is assumed to point to a buffer that can store
		F_BTREE_MAX_LEVELS FLMUINT values
****************************************************************************/
RCODE F_BTree::btGetBlockChains(
	FLMUINT *	puiBlockAddrs,
	FLMUINT *	puiNumLevels)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT				uiNumLevels = 0;
	FLMUINT32			ui32NextBlockAddr;
	IF_Block *			pCurrentBlock = NULL;
	FLMBYTE *			pucCurrentBlock = NULL;
	FLMBYTE *			pucEntry;

	f_assert( m_bOpened);

	// Fill puiBlockAddrs and calculate the number of levels.
	// NOTE: Normally, level 0 is the leaf level.  In this function,
	// puiBlockAddrs[ 0] is the ROOT and puiBlockAddrs[ uiNumLevels - 1] 
	// is the LEAF!

	ui32NextBlockAddr = m_ui32RootBlockAddr;

	while( ui32NextBlockAddr)
	{
		puiBlockAddrs[ uiNumLevels++] = ui32NextBlockAddr;

		if( RC_BAD( m_pBlockMgr->getBlock( ui32NextBlockAddr, 
			&pCurrentBlock, &pucCurrentBlock)))
		{
			goto Exit;
		}

		if( getBlockType( pucCurrentBlock) == F_BLK_TYPE_BT_LEAF || 
			 getBlockType( pucCurrentBlock) == F_BLK_TYPE_BT_LEAF_DATA)
		{
			ui32NextBlockAddr = 0;
		}
		else
		{
			// The child block address is the first part of the entry

			pucEntry = BtEntry( pucCurrentBlock, 0);
			ui32NextBlockAddr = bteGetBlockAddr( pucEntry);
		}

		// Release the current block

		pCurrentBlock->Release();
		pCurrentBlock = NULL;
		pucCurrentBlock = NULL;
	}

	*puiNumLevels = uiNumLevels;

Exit:

	if( pCurrentBlock)
	{
		pCurrentBlock->Release();
	}

	return( rc);
}

/***************************************************************************
Desc: Function to insert an entry into the Btree.
****************************************************************************/
RCODE F_BTree::btInsertEntry(
	const FLMBYTE *	pucKey,
	FLMUINT				uiKeyBufSize,
	FLMUINT				uiKeyLen,
	const FLMBYTE *	pucData,
	FLMUINT				uiDataLen,
	FLMBOOL				bFirst,
	FLMBOOL				bLast,
	FLMUINT32 *			pui32BlockAddr,
	FLMUINT *			puiOffsetIndex)
{
	RCODE					rc = NE_FLM_OK;
	FLMBYTE				pucDOAddr[ 4];
	FLMUINT32			ui32TmpBlockAddr;
	
	if( !m_bOpened || m_bSetupForRead || m_bSetupForReplace ||
		  (m_bSetupForWrite && bFirst))
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_BAD_STATE);
		goto Exit;
	}

	f_assert( m_uiSearchLevel >= F_BTREE_MAX_LEVELS);

	if( !uiKeyLen)
	{
		rc = RC_SET( NE_FLM_ILLEGAL_OP);
		goto Exit;
	}

	// Be sure to clear the Data Only flag.
	
	if( bFirst)
	{
		m_bDataOnlyBlock = FALSE;
	}

	if( bLast)
	{
		m_Stack[ 0].uiKeyBufSize = uiKeyBufSize;

		// We need to locate where we should insert the new entry.
		
		rc = findEntry( pucKey, uiKeyLen, FLM_EXACT);

		// Should not find this entry.  If we get back anything other than
		// an NE_FLM_NOT_FOUND, then there is a problem.
		
		if( rc != NE_FLM_NOT_FOUND)
		{
			if( RC_OK( rc))
			{
				rc = RC_SET( NE_FLM_NOT_UNIQUE);
			}
			goto Exit;
		}
	}

	if( bFirst && (!bLast || (uiKeyLen + uiDataLen > m_uiOverflowThreshold)))
	{
		// If bLast is not set, then we will setup to store the data in
		// data only blocks.  The assumption is that whenever we don't see bLast
		// set when starting an insert, then the data is so large that it must
		// be placed in a chain of Data Only blocks.  There is no way for me to
		// check the final size of the data ahead of time, so I rely on the
		// calling routine to figure this out for me.

		// Get one empty block to begin with.

		f_assert( !m_pBlock && !m_pucBlock);
		
		if( RC_BAD( rc = m_pBlockMgr->createBlock( &m_pBlock, &m_pucBlock, 
			&ui32TmpBlockAddr)))
		{
			goto Exit;
		}
		
		setBlockAddr( m_pucBlock, ui32TmpBlockAddr);

		// The data going in will be stored in Data-only blocks.
		// Setup the block header...

		setBlockType( m_pucBlock, F_BLK_TYPE_BT_DATA_ONLY);
		setPrevInChain( m_pucBlock, 0);
		setNextInChain( m_pucBlock, 0);

//		if( m_pLFile->uiEncId)
//		{
//			((F_ENC_DO_BLK_HDR *)pBlockHdr)->ui32EncId = (FLMUINT32)m_pLFile->uiEncId;
//			setBlockEncrypted( pBlockHdr);
//		}

		setBytesAvail( m_pucBlock, 
			(FLMUINT16)(m_uiBlockSize - sizeofDOBlockHdr( m_pucBlock)));

		m_uiDataRemaining = m_uiBlockSize - sizeofDOBlockHdr( m_pucBlock);
		m_uiDataLength = 0;
		m_uiOADataLength = 0;
		m_bDataOnlyBlock = TRUE;
		m_bSetupForWrite = TRUE;
		m_ui32DOBlockAddr = getBlockAddr( m_pucBlock);
		m_ui32CurBlockAddr = m_ui32DOBlockAddr;
	}

	if( m_bDataOnlyBlock)
	{
		if( RC_BAD( rc = storeDataOnlyBlocks( pucKey, uiKeyLen, bFirst,
			pucData, uiDataLen)))
		{
			goto Exit;
		}
	}

	if( bLast)
	{
		const FLMBYTE *		pucLocalData;
		FLMUINT					uiLocalDataLen;
		F_ELM_UPD_ACTION		eAction;

		if( m_bDataOnlyBlock)
		{
			// build an entry that points to the DO block.

			UD2FBA( m_ui32DOBlockAddr, pucDOAddr);
			pucLocalData = &pucDOAddr[0];
			uiLocalDataLen = m_uiOADataLength;
			eAction = ELM_INSERT_DO;
		}
		else
		{
			pucLocalData = pucData;
			uiLocalDataLen = uiDataLen;
			eAction = ELM_INSERT;
		}

		if( RC_BAD( rc = updateEntry( pucKey, uiKeyLen, pucLocalData,
			uiLocalDataLen, eAction)))
		{
			goto Exit;
		}

		if( pui32BlockAddr)
		{
			*pui32BlockAddr = m_ui32PrimaryBlockAddr;
		}

		if( puiOffsetIndex)
		{
			*puiOffsetIndex = m_uiCurOffset;
		}

		m_bSetupForWrite = FALSE;
	}

Exit:

	if( m_pBlock)
	{
		m_pBlock->Release();
		m_pBlock = NULL;
		m_pucBlock = NULL;
	}

	releaseBlocks( TRUE);
	return( rc);
}

/***************************************************************************
Desc: Function to remove an entry into the Btree.
****************************************************************************/
RCODE F_BTree::btRemoveEntry(
	const FLMBYTE *	pucKey,
	FLMUINT				uiKeyBufSize,
	FLMUINT				uiKeyLen)
{
	RCODE				rc = NE_FLM_OK;
	FLMBYTE *		pucValue = NULL;
	FLMUINT			uiLen = 0;

	if ( !m_bOpened)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_BAD_STATE);
		goto Exit;
	}

	btResetBtree();

	m_Stack[ 0].uiKeyBufSize = uiKeyBufSize;

	// We need to locate where we should remove the entry.
	
	if (RC_BAD( rc = findEntry( pucKey, uiKeyLen, FLM_EXACT)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = updateEntry( pucKey, uiKeyLen, pucValue, uiLen,
			ELM_REMOVE)))
	{
		goto Exit;
	}

Exit:

	releaseBlocks( TRUE);
	return( rc);
}

/***************************************************************************
Desc:	Function to provide a streaming interface for replacing large 
		data elements.
****************************************************************************/
RCODE F_BTree::btReplaceEntry(
	const FLMBYTE *	pucKey,
	FLMUINT				uiKeyBufSize,
	FLMUINT				uiKeyLen,
	const FLMBYTE *	pucData,
	FLMUINT				uiDataLen,
	FLMBOOL				bFirst,
	FLMBOOL				bLast,
	FLMBOOL				bTruncate,
	FLMUINT32 *			pui32BlockAddr,
	FLMUINT *			puiOffsetIndex)
{
	RCODE					rc = NE_FLM_OK;
	FLMBYTE *			pucEntry;
	const FLMBYTE *	pucLocalData;
	FLMUINT				uiOADataLength = 0;
	FLMBYTE				pucDOAddr[ 4];
	FLMUINT32			ui32TmpBlockAddr;

	if( !m_bOpened || m_bSetupForRead || m_bSetupForWrite ||
		  (m_bSetupForReplace && bFirst))
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_BAD_STATE);
		goto Exit;
	}

	f_assert( m_uiSearchLevel >= F_BTREE_MAX_LEVELS);

	if (!uiKeyLen)
	{
		rc = RC_SET( NE_FLM_ILLEGAL_OP);
		goto Exit;
	}

	// Be sure to clear the Data Only flags
	
	if( bFirst)
	{
		m_bDataOnlyBlock = FALSE;
		m_bOrigInDOBlocks = FALSE;
	}

	if( bFirst || bLast)
	{
		m_Stack[ 0].uiKeyBufSize = uiKeyBufSize;

		// We need to locate the entry we want to replace
		
		if( RC_BAD( rc = findEntry( pucKey, uiKeyLen, FLM_EXACT, NULL,
			pui32BlockAddr, puiOffsetIndex)))
		{
			goto Exit;
		}

		// We must first determine if the existing entry is stored
		// in data only blocks.
		
		pucEntry = BtEntry( m_pStack->pucBlock, m_pStack->uiCurOffset);

		fbtGetEntryDataLength( pucEntry, &pucLocalData,
			&uiOADataLength, &m_bOrigInDOBlocks);

	}

	if( bFirst && (!bLast || (bLast && !bTruncate && m_bOrigInDOBlocks) ||
			(uiKeyLen + uiDataLen > m_uiOverflowThreshold)))
	{
		// If bLast is not set, then we will setup to store the data in
		// data only blocks.
		
		m_bDataOnlyBlock = TRUE;
		f_assert( !m_pBlock && !m_pucBlock);
		
		if( m_bOrigInDOBlocks)
		{
			// Need to get the first DO block, and work from there.
			
			m_ui32DOBlockAddr = bteGetBlockAddr( pucLocalData);
			
			if( RC_BAD( rc = m_pBlockMgr->getBlock( m_ui32DOBlockAddr, 
				&m_pBlock, &m_pucBlock)))
			{
				goto Exit;
			}
		}
		else
		{
			// Get one empty block to begin with
			
			if( RC_BAD( rc = m_pBlockMgr->createBlock( &m_pBlock, 
				&m_pucBlock, &ui32TmpBlockAddr)))
			{
				goto Exit;
			}

			// Setup the block header
			
			setBlockAddr( m_pucBlock, ui32TmpBlockAddr);
			setBlockType( m_pucBlock, F_BLK_TYPE_BT_DATA_ONLY);
			setPrevInChain( m_pucBlock, 0);
			setNextInChain( m_pucBlock, 0);

//			if (m_pLFile->uiEncId)
//			{
//				((F_ENC_DO_BLK_HDR *)pBlockHdr)->ui32EncId = (FLMUINT32)m_pLFile->uiEncId;
//				setBlockEncrypted( pBlockHdr);
//			}

			setBytesAvail( m_pucBlock,
				(FLMUINT16)(m_uiBlockSize - sizeofDOBlockHdr( m_pucBlock)));
		}

		m_uiDataRemaining = m_uiBlockSize - sizeofDOBlockHdr( m_pucBlock);
		m_uiDataLength = 0;
		m_uiOADataLength = 0;
		m_bDataOnlyBlock = TRUE;
		m_bSetupForReplace = TRUE;
		m_ui32DOBlockAddr = getBlockAddr( m_pucBlock);
		m_ui32CurBlockAddr = m_ui32DOBlockAddr;
	}

	if( m_bDataOnlyBlock)
	{
		if( !bTruncate && !m_bOrigInDOBlocks)
		{
			bTruncate = TRUE;
		}

		// May need to skip over the key that is stored in the first DO block.
		// We only want to do this the first time in here.  The test to determine
		// if this is our first time in this block is to see if the m_uiDataLength
		// is equal to the m_uiDataRemaining.  They would only be the same on the
		// first time for each DO block.
		
		if( m_bOrigInDOBlocks && m_pBlock &&
			getPrevInChain( m_pucBlock) == 0 && !m_uiDataLength)
		{
			m_uiDataRemaining -= (uiKeyLen + 2);
		}

		if( RC_BAD( rc = replaceDataOnlyBlocks( pucKey, uiKeyLen,
			!m_bOrigInDOBlocks && bFirst, pucData, uiDataLen, bLast,
			bTruncate)))
		{
			goto Exit;
		}
	}

	// If we were writing to Data Only Blocks and we are not truncating the
	// data, then we are done here.
	
	if( m_bDataOnlyBlock && !bTruncate)
	{
		if( bLast && (uiOADataLength <= m_uiOADataLength))
		{
			bTruncate = TRUE;
		}
		else
		{
			goto Exit;
		}
	}

	// Only replace the entry on the last call.
	
	if( bLast)
	{
		FLMUINT					uiLocalDataLen;
		F_ELM_UPD_ACTION		eAction;

		if (m_bDataOnlyBlock)
		{
			// build an entry that points to the DO block.

			UD2FBA( m_ui32DOBlockAddr, pucDOAddr);

			pucLocalData = &pucDOAddr[0];
			uiLocalDataLen = m_uiOADataLength;
			eAction = ELM_REPLACE_DO;
		}
		else
		{
			pucLocalData = pucData;
			uiLocalDataLen = uiDataLen;
			eAction = ELM_REPLACE;
		}

		if( RC_BAD( rc = updateEntry( pucKey, uiKeyLen, pucLocalData,
			uiLocalDataLen, eAction, bTruncate)))
		{
			goto Exit;
		}
	}

Exit:

	if (RC_OK( rc))
	{
		if (pui32BlockAddr)
		{
			*pui32BlockAddr = m_ui32PrimaryBlockAddr;
		}

		if (puiOffsetIndex)
		{
			*puiOffsetIndex = m_uiCurOffset;
		}
	}

	if( bLast)
	{
		m_bSetupForReplace = FALSE;
	}

	if (m_pBlock)
	{
		m_pBlock->Release();
		m_pBlock = NULL;
		m_pucBlock = NULL;
	}

	releaseBlocks( TRUE);
	return( rc);
}

/***************************************************************************
Desc: Function to search the Btree for a specific key.
****************************************************************************/
RCODE F_BTree::btLocateEntry(
	FLMBYTE *			pucKey,
	FLMUINT				uiKeyBufSize,
	FLMUINT *			puiKeyLen,
	FLMUINT				uiMatch,
	FLMUINT *			puiPosition,		// May be NULL
	FLMUINT *			puiDataLength,
	FLMUINT32 *			pui32BlockAddr,
	FLMUINT *			puiOffsetIndex)
{
	RCODE					rc = NE_FLM_OK;
	FLMBYTE *			pucEntry = NULL;

	f_assert( pucKey && uiKeyBufSize && puiKeyLen);

	if (!m_bOpened || m_bSetupForWrite || m_bSetupForReplace)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_BAD_STATE);
		goto Exit;
	}

	m_bSetupForRead = FALSE;
	m_Stack[ 0].uiKeyBufSize = uiKeyBufSize;

	// Find the entry we are interested in.
	
	if (RC_BAD(rc = findEntry( pucKey, *puiKeyLen, uiMatch, puiPosition,
		pui32BlockAddr, puiOffsetIndex)))
	{
		goto Exit;
	}

	m_ui32PrimaryBlockAddr = m_pStack->ui32BlockAddr;
	m_uiPrimaryOffset = m_pStack->uiCurOffset;
	m_ui32CurBlockAddr = m_ui32PrimaryBlockAddr;
	m_uiCurOffset = m_uiPrimaryOffset;

	// Point to the entry...
	
	pucEntry = BtEntry( m_pStack->pucBlock, m_pStack->uiCurOffset);
                                                       
	// Return the optional data length - get the overall data length only.
	
	if( puiDataLength && 
		 getBlockType( m_pStack->pucBlock) == F_BLK_TYPE_BT_LEAF_DATA)
	{
		fbtGetEntryDataLength( pucEntry, NULL, puiDataLength, NULL);
	}                                               
	else if (puiDataLength)
	{
		*puiDataLength = 0;
	}

	if( RC_BAD( rc = setupReadState( m_pStack->pucBlock, pucEntry)))
	{
		goto Exit;
	}

	// In case the returning key is not what was originally requested, such as
	// in the case of FLM_FIRST, FLM_LAST, FLM_EXCL and possibly FLM_INCL, 
	// we will pass back the key we actually found.

	if( uiMatch != FLM_EXACT)
	{
		if( RC_BAD( rc = setReturnKey( pucEntry, getBlockType( m_pStack->pucBlock),
			pucKey, puiKeyLen, uiKeyBufSize)))
		{
			goto Exit;
		}
	}

	m_bFirstRead =		FALSE;
	m_bSetupForRead = TRUE;

Exit:

	releaseBlocks( FALSE);
	return( rc);
}

/***************************************************************************
Desc: Method to get the data after a call to btLocateEntry, btNextEntry,
		btPrevEntry, btFirstEntry or btLastEntry.
****************************************************************************/
RCODE F_BTree::btGetEntry(
	FLMBYTE *				pucKey,
	FLMUINT 					uiKeyLen,
	FLMBYTE *				pucData,
	FLMUINT					uiDataBufSize,
	FLMUINT *				puiDataLen)
{
	RCODE					rc = NE_FLM_OK;
	FLMBYTE *			pucEntry;
	FLMBYTE *			pucDataPtr = NULL;
	
	if( !m_bOpened || !m_bSetupForRead || 
		 m_bSetupForWrite || m_bSetupForReplace)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_BAD_STATE);
		goto Exit;
	}

	if( puiDataLen)
	{
		*puiDataLen = 0;
	}

	// Is there anything there to get?
	
	if( m_uiOADataRemaining == 0)
	{
		rc = RC_SET( NE_FLM_EOF_HIT);
		goto Exit;
	}

	// Get the current block.  It is either a DO or a Btree block.
	
	if( !m_pBlock)
	{
		if( RC_BAD( rc = m_pBlockMgr->getBlock( m_ui32CurBlockAddr, 
			&m_pBlock, &m_pucBlock)))
		{
			goto Exit;
		}
	}

	// Now to find where we were the last time through.
	
	if( !m_bDataOnlyBlock)
	{
		pucEntry = BtEntry( m_pucBlock, m_uiCurOffset);
		fbtGetEntryDataLength( pucEntry, 
			(const FLMBYTE **)&pucDataPtr, NULL, NULL);
	}
	else
	{
		pucDataPtr = m_pucBlock + sizeofDOBlockHdr( m_pucBlock);

		// May need to skip over the key that is stored in the first DO block.
		// We only want to do this the first time in here.  The test to determine
		// if this is our first time in this block is to see if the m_uiDataLength
		// is equal to the m_uiDataRemaining.  They would only be the same on the
		// first time for each DO block.
		
		if( getPrevInChain( m_pucBlock) == 0)
		{
			FLMUINT16	ui16KeyLen = FB2UW( pucDataPtr);

			// Key lengths should be the same
			
			f_assert( uiKeyLen == (FLMUINT)ui16KeyLen);

			pucDataPtr += (ui16KeyLen + 2);
		}
	}

	pucDataPtr += (m_uiDataLength - m_uiDataRemaining);
	
	if( RC_BAD( rc = extractEntryData( pucKey, uiKeyLen, pucData,
		uiDataBufSize, puiDataLen, &pucDataPtr)))
	{
		goto Exit;
	}
	
	// Mark that we have completed our first read operation.
	// No more read synchronization allowed.
	
	m_bFirstRead = TRUE;

Exit:

	if( m_pBlock)
	{
		m_pBlock->Release();
		m_pBlock = NULL;
		m_pucBlock = NULL;
	}

	releaseBlocks( FALSE);
	return( rc);
}

/***************************************************************************
Desc: Function to locate the next entry in the Btree.  The key buffer and
		actual size is passed in.
****************************************************************************/
RCODE F_BTree::btNextEntry(
	FLMBYTE *				pucKey,
	FLMUINT					uiKeyBufSize,
	FLMUINT *				puiKeyLen,
	FLMUINT *				puiDataLength,
	FLMUINT32 *				pui32BlockAddr,
	FLMUINT *				puiOffsetIndex)
{
	RCODE						rc = NE_FLM_OK;
	FLMBYTE *				pucEntry = NULL;
	FLMBOOL					bAdvanced = FALSE;

	if( !m_bOpened || !m_bSetupForRead)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_BAD_STATE);
		goto Exit;
	}

	// Make sure we are looking at btree  block.  If the m_bDataOnlyBlock
	// flag is set, then the block address in m_ui32CurBlockAddr is a
	// data only block.  We must reset it to the primary block address.
	
	if( m_bDataOnlyBlock)
	{
		m_ui32CurBlockAddr = m_ui32PrimaryBlockAddr;
	}
	else
	{
		// If the entry did not reference a DO block, then we need to
		// reset the primary block and offset with where we currently
		// are incase the current block is further ahead.  This saves time
		// so that we don't have to scan past old blocks we are not intereseted
		// in.
		
		m_ui32PrimaryBlockAddr = m_ui32CurBlockAddr;
		m_uiPrimaryOffset = m_uiCurOffset;
	}

	// Get the current block if we need it.
	
	if( !m_pBlock)
	{
		if( RC_BAD( rc = m_pBlockMgr->getBlock( m_ui32CurBlockAddr, 
			&m_pBlock, &m_pucBlock)))
		{
			goto Exit;
		}
	}

	// If we have already advanced due to a resynch, then we don't need to call
	// the advanceToNextElement function, however, we do need to get the
	// current entry.
	
	if( bAdvanced)
	{
		pucEntry = BtEntry( m_pucBlock, m_uiCurOffset);
	}
	else
	{
		for (;;)
		{
			// Advance to the next entry in the block.  We don't have a stack so
			// don't advance it.
			
			if( RC_BAD( rc = advanceToNextElement( FALSE)))
			{
				goto Exit;
			}

			pucEntry = BtEntry( m_pucBlock, m_uiCurOffset);

			if( m_bTreeHoldsData)
			{
				if( bteFirstElementFlag( pucEntry))
				{
					break;
				}
			}
			else
			{
				break;
			}
		}
	}

	// Return the optional data length - get the overall data length only.
	
	if( puiDataLength)
	{
		fbtGetEntryDataLength( pucEntry, NULL, puiDataLength, NULL);
	}
	
	if( RC_BAD( rc = setupReadState( m_pucBlock, pucEntry)))
	{
		goto Exit;
	}

	// Incase the returning key is not what was originally requested, such as in
	// the case of FLM_FIRST, FLM_LAST, FLM_EXCL and possibly FLM_INCL,
	// we will pass back the key we actually found.
	
	if( RC_BAD( rc = setReturnKey( pucEntry, getBlockType( m_pucBlock),
		pucKey, puiKeyLen, uiKeyBufSize)))
	{
		goto Exit;
	}

	if( pui32BlockAddr)
	{
		*pui32BlockAddr = getBlockAddr( m_pucBlock);
	}

	if( puiOffsetIndex)
	{
		*puiOffsetIndex = m_uiCurOffset;
	}

	m_bFirstRead = FALSE;

Exit:

	if( m_pBlock)
	{
		m_pBlock->Release();
		m_pBlock = NULL;
		m_pucBlock = NULL;
	}

	releaseBlocks( FALSE);
	return( rc);
}

/***************************************************************************
Desc: Function to get the previous entry in the Btree.
****************************************************************************/
RCODE F_BTree::btPrevEntry(
	FLMBYTE *				pucKey,
	FLMUINT					uiKeyBufSize,
	FLMUINT *				puiKeyLen,
	FLMUINT *				puiDataLength,
	FLMUINT32 *				pui32BlockAddr,
	FLMUINT *				puiOffsetIndex)
{
	RCODE						rc = NE_FLM_OK;
	FLMBYTE *				pucEntry = NULL;

	if( !m_bOpened || !m_bSetupForRead)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_BAD_STATE);
		goto Exit;
	}

	// Make sure we are looking at the first block of the
	// current entry.  Reading of the entry could have moved us
	// to another block, or if it was in a DO block, we would be
	// looking at the wrong block altogether.
	
	m_ui32CurBlockAddr = m_ui32PrimaryBlockAddr;
	m_uiCurOffset = m_uiPrimaryOffset;

	if( !m_pBlock)
	{
		// Fetch the current block, then backup from there.
		
		if( RC_BAD( rc = m_pBlockMgr->getBlock( m_ui32CurBlockAddr, 
			&m_pBlock, &m_pucBlock)))
		{
			goto Exit;
		}
	}

	for (;;)
	{
		// Backup to the previous entry in the block.
		
		if( RC_BAD( rc = backupToPrevElement( FALSE)))
		{
			goto Exit;
		}

		// Get the entry, size etc.
		
		pucEntry = BtEntry( m_pucBlock, m_uiCurOffset);

		if( m_bTreeHoldsData)
		{
			if( bteFirstElementFlag( pucEntry))
			{
				break;
			}
		}
		else
		{
			break;
		}
	}

	// Return the optional data length - get the overall data length only.
	
	if( puiDataLength)
	{
		fbtGetEntryDataLength( pucEntry, NULL, puiDataLength, NULL);
	}
	
	if( RC_BAD( rc = setupReadState( m_pucBlock, pucEntry)))
	{
		goto Exit;
	}

	// In case the returning key is not what was originally requested, such as in
	// the case of FLM_FIRST, FLM_LAST, FLM_EXCL and possibly FLM_INCL,
	// we will pass back the key we actually found.

	if( RC_BAD( rc = setReturnKey( pucEntry, getBlockType( m_pucBlock),
		pucKey, puiKeyLen, uiKeyBufSize)))
	{
		goto Exit;
	}

	if( pui32BlockAddr)
	{
		*pui32BlockAddr = getBlockAddr( m_pucBlock);
	}

	if( puiOffsetIndex)
	{
		*puiOffsetIndex = m_uiCurOffset;
	}

	m_bFirstRead = FALSE;

Exit:

	if( m_pucBlock)
	{
		m_pBlock->Release();
		m_pBlock = NULL;
		m_pucBlock = NULL;
	}

	releaseBlocks( FALSE);
	return( rc);
}

/***************************************************************************
Desc:	Locate the first entry in the Btree and return the key.
****************************************************************************/
RCODE F_BTree::btFirstEntry(
	FLMBYTE *			pucKey,
	FLMUINT				uiKeyBufSize,
	FLMUINT *			puiKeyLen,
	FLMUINT *			puiDataLength,
	FLMUINT32 *			pui32BlockAddr,
	FLMUINT *			puiOffsetIndex)
{
	RCODE					rc = NE_FLM_OK;

	m_Stack[ 0].pucKeyBuf = pucKey;
	m_Stack[ 0].uiKeyBufSize = uiKeyBufSize;

	if( RC_BAD( rc = btLocateEntry( pucKey, uiKeyBufSize, puiKeyLen,
		FLM_FIRST, NULL, puiDataLength, pui32BlockAddr, puiOffsetIndex)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Locate the last entry in the Btree and return the key.
****************************************************************************/
RCODE F_BTree::btLastEntry(
	FLMBYTE *				pucKey,
	FLMUINT					uiKeyBufSize,
	FLMUINT *				puiKeyLen,
	FLMUINT *				puiDataLength,
	FLMUINT32 *				pui32BlockAddr,
	FLMUINT *				puiOffsetIndex)
{
	RCODE				rc = NE_FLM_OK;

	m_Stack[ 0].pucKeyBuf = pucKey;
	m_Stack[ 0].uiKeyBufSize = uiKeyBufSize;

	if( RC_BAD( rc = btLocateEntry( pucKey, uiKeyBufSize, puiKeyLen,
		FLM_LAST, NULL, puiDataLength, pui32BlockAddr, puiOffsetIndex)))
	{
		if( rc == NE_FLM_BOF_HIT)
		{
			rc = RC_SET( NE_FLM_EOF_HIT);
		}
		
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc: Function to search the Btree for a specific key.
****************************************************************************/
RCODE F_BTree::btPositionTo(
	FLMUINT					uiPosition,
	FLMBYTE *				pucKey,
	FLMUINT					uiKeyBufSize,
	FLMUINT *				puiKeyLen)
{
	RCODE				rc = NE_FLM_OK;
	FLMBYTE *		pucEntry = NULL;

	f_assert( pucKey && uiKeyBufSize && puiKeyLen);

	m_bSetupForRead = FALSE;

	if( !m_bOpened || !m_bCounts)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_BAD_STATE);
		goto Exit;
	}

	// Find the entry we are interested in.
	
	if( RC_BAD(rc = positionToEntry( uiPosition)))
	{
		goto Exit;
	}

	m_ui32PrimaryBlockAddr = m_pStack->ui32BlockAddr;
	m_uiPrimaryOffset = m_pStack->uiCurOffset;
	m_ui32CurBlockAddr = m_ui32PrimaryBlockAddr;
	m_uiCurOffset = m_uiPrimaryOffset;

	// Point to the entry ...
	
	pucEntry = BtEntry( m_pStack->pucBlock, m_pStack->uiCurOffset);
	
	if( RC_BAD( rc = setupReadState( m_pStack->pucBlock, pucEntry)))
	{
		goto Exit;
	}

	// In case the returning key is not what was originally requested, such
	// as in the case of FLM_FIRST, FLM_LAST, FLM_EXCL and 
	// possibly FLM_INCL, we will pass back the key we actually found.
	
	if( RC_BAD( rc = setReturnKey( pucEntry,
		getBlockType( m_pStack->pucBlock), pucKey, puiKeyLen, uiKeyBufSize)))
	{
		goto Exit;
	}

	m_bFirstRead = FALSE;
	m_bSetupForRead = TRUE;

Exit:

	f_assert( !m_pBlock);
	releaseBlocks( FALSE);
	return( rc);
}

/***************************************************************************
Desc:	Method to get the actual poisition of the entry.  Note: Must be
		maintaining counts in the b-tree AND also have located to an entry
		first.  The key that is passed in is used only if we have to
		resynchronize due to a transaction change.
****************************************************************************/
RCODE F_BTree::btGetPosition(
	FLMUINT *		puiPosition)
{
	RCODE				rc = NE_FLM_OK;

	if( !m_bOpened || !m_bSetupForRead || !m_bCounts)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_BAD_STATE);
		goto Exit;
	}

	*puiPosition = 0;

	m_ui32CurBlockAddr = m_ui32PrimaryBlockAddr;
	m_uiCurOffset = m_uiPrimaryOffset;

	// To calculate the position, we will have to reconstruct the stack.
	
	m_pStack = &m_Stack[ m_uiStackLevels - 1];
	for (;;)
	{
		// Get the block at this level.
		
		f_assert( m_pStack->ui32BlockAddr);
		f_assert( m_pStack->pBlock == NULL);

		if( RC_BAD( rc = m_pBlockMgr->getBlock( m_pStack->ui32BlockAddr, 
			&m_pStack->pBlock, &m_pStack->pucBlock)))
		{
			goto Exit;
		}

		*puiPosition += countRangeOfKeys( m_pStack, 0, m_pStack->uiCurOffset);

		if( getBlockType( m_pStack->pucBlock) == F_BLK_TYPE_BT_LEAF ||
			 getBlockType( m_pStack->pucBlock) == F_BLK_TYPE_BT_LEAF_DATA)
		{
			break;
		}
		else
		{
			// Next level down. (stack is inverted).
			
			m_pStack--;
		}
	}

Exit:

	releaseBlocks( FALSE);
	return( rc);
}

/***************************************************************************
Desc:	Method to rewind back to the beginning of the current entry.
****************************************************************************/
RCODE F_BTree::btRewind( void)
{
	RCODE					rc = NE_FLM_OK;
	IF_Block *			pBlock = NULL;
	FLMBYTE *			pucBlock = NULL;

	if( !m_bSetupForRead)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_BAD_STATE);
		goto Exit;
	}

	m_ui32CurBlockAddr = m_ui32PrimaryBlockAddr;
	m_uiCurOffset = m_uiPrimaryOffset;

	m_uiOADataRemaining = m_uiOADataLength;
	m_uiDataLength = m_uiPrimaryDataLen;
	m_uiDataRemaining = m_uiDataLength;

	if( m_bDataOnlyBlock)
	{
		m_ui32CurBlockAddr = m_ui32DOBlockAddr;

		if( RC_BAD( rc = m_pBlockMgr->getBlock( m_ui32DOBlockAddr, 
			&pBlock, &pucBlock)))
		{
			goto Exit;
		}
		
		// Local amount of data in this block
		
		m_uiDataRemaining = m_uiBlockSize - sizeofDOBlockHdr( pucBlock) - 
										getBytesAvail( pucBlock);

		// Keep the actual local data size for later.
		
		m_uiDataLength = m_uiDataRemaining;

		// Now release the DO Block.  We will get it again when we need it.
		
		pBlock->Release();
		pBlock = NULL;
		pucBlock = NULL;
	}

	m_bFirstRead = FALSE;
	m_bSetupForRead = TRUE;

Exit:

	if( pBlock)
	{
		pBlock->Release();
		pBlock = NULL;
	}

	releaseBlocks( FALSE);
	return( rc);
}

/***************************************************************************
Desc:	Method for computing the number of keys and blocks between two points 
		in the Btree.  The key count is inclusive of the two end points and
		the block count is exclusive of the two end points.
****************************************************************************/
#if 0
RCODE F_BTree::btComputeCounts(
	IF_BTree *		pUntilBtree,
	FLMUINT *		puiBlockCount,
	FLMUINT *		puiKeyCount,
	FLMBOOL *		pbTotalsEstimated,
	FLMUINT			uiAvgBlockFullness)
{
	RCODE			rc = NE_FLM_OK;

	if( !m_bSetupForRead)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_BAD_STATE);
		goto Exit;
	}

	// Ensure that both Btrees are from the same container.
	
	if( m_ui32RootBlockAddr != pUntilBtree->getRootBlockAddr())
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_FAILURE);
		goto Exit;
	}

	rc = computeCounts( m_pStack, pUntilBtree->m_pStack, puiBlockCount,
			  puiKeyCount, pbTotalsEstimated, uiAvgBlockFullness);

Exit:

	releaseBlocks( FALSE);
	pUntilBtree->releaseBlocks( FALSE);
	return( rc);
}
#endif

/***************************************************************************
Desc:	Function to release the blocks in the stack, and optionally, reset
		the stack
****************************************************************************/
void F_BTree::releaseBlocks(
	FLMBOOL			bResetStack)
{
	FLMUINT		uiLevel;

	for( uiLevel = 0; uiLevel <= m_uiRootLevel; uiLevel++)
	{
		if( m_Stack[ uiLevel].pBlock)
		{
			m_Stack[ uiLevel].pBlock->Release();
			m_Stack[ uiLevel].pBlock = NULL;
			m_Stack[ uiLevel].pucBlock = NULL;
		}
		
		if( bResetStack)
		{
			m_Stack[ uiLevel].ui32BlockAddr = 0;
			m_Stack[ uiLevel].uiKeyLen = 0;
			m_Stack[ uiLevel].uiCurOffset = 0;
			m_Stack[ uiLevel].uiLevel = 0;
		}
	}
	
	if( bResetStack)
	{
		m_uiStackLevels = 0;
		m_uiRootLevel = 0;
		m_bStackSetup = FALSE;
		m_pStack = NULL;
	}
}

/***************************************************************************
Desc: Function to create a new block at the current level.  The new block 
		will always be inserted previous to the current block.  All entries
		that sort ahead of the current insertion point will be moved into
		the new block.  If there is room, the new entry will be inserted
		into the current block.  Otherwise, if there is room, the new entry
		will be inserted into the new block.  If there is still not enough
		room, then if possible, it try to store a partial entry in the new
		block. If we still cannot store anything, we will see if we can 
		store a partial entry in the current block.  If that does not work,
		then it will set the remaining amount and return.  Another block
		split will be needed before we store this entry.
****************************************************************************/
RCODE F_BTree::splitBlock(
	const FLMBYTE *		pucKey,
	FLMUINT					uiKeyLen,
	const FLMBYTE *		pucValue,
	FLMUINT					uiLen,
	FLMUINT					uiFlags,
	FLMUINT					uiOADataLen,
	FLMUINT					uiChildBlockAddr,
	FLMUINT 					uiCounts,
	const FLMBYTE **		ppucRemainingValue,
	FLMUINT *				puiRemainingLen,
	FLMBOOL *				pbBlockSplit)
{
	RCODE						rc = NE_FLM_OK;
	IF_Block *				pNewBlock = NULL;
	IF_Block *				pPrevBlock = NULL;
	FLMBYTE *				pucNewBlock = NULL;
	FLMBYTE *				pucPrevBlock = NULL;
	FLMUINT					uiBlockAddr;
	FLMUINT					uiEntrySize;
	FLMBOOL					bHaveRoom;
	FLMBOOL					bMovedToPrev = FALSE;
	FLMBOOL					bLastEntry;
	FLMUINT					uiMinEntrySize;
	FLMBOOL					bDefragBlock = FALSE;
	FLMBOOL					bSavedReplaceInfo = FALSE;
	FLMUINT32				ui32NewBlockAddr;

	// If the current block is a root block, then we will have to introduce
	// a new level into the B-Tree.
	
	if( isRootBlock( m_pStack->pucBlock))
	{
		if( RC_BAD( rc = createNewLevel()))
		{
			goto Exit;
		}
	}

	// If the current block is empty we must insert what we can here.
	// This scenario only occurs when we are engaged in a ReplaceByInsert
	// operation. Normal inserts would never result in an empty block.
	// Since we know we are part of a replace operation, we know that the
	// parent of this block only needs the counts updated, not the key.
	
	if( m_pStack->uiLevel == 0 && getNumKeys( m_pStack->pucBlock) == 0)
	{
		if( RC_BAD( rc = storePartialEntry( pucKey, uiKeyLen, pucValue,
				uiLen, uiFlags, uiChildBlockAddr, uiCounts, ppucRemainingValue,
				puiRemainingLen, FALSE)))
		{
			goto Exit;
		}
		
		*pbBlockSplit = FALSE;
		goto MoveToPrev;
	}

	// Create a new block and insert it as previous to this block.
	
	if( RC_BAD( rc = m_pBlockMgr->createBlock( &pNewBlock, &pucNewBlock,
		&ui32NewBlockAddr)))
	{
		goto Exit;
	}

	*pbBlockSplit = TRUE;

	// Setup the header ...
	
	setBlockAddr( pucNewBlock, ui32NewBlockAddr);
	unsetRootBlock( pucNewBlock);
	setNumKeys( pucNewBlock, 0);
	setBlockLevel( pucNewBlock, (FLMUINT8)m_pStack->uiLevel);
	setBlockType( pucNewBlock, getBlockType( m_pStack->pucBlock));
	setBTreeId( pucNewBlock, getBTreeId( m_pStack->pucBlock));
	
	// Check for encrypted block.
	
	if( isEncryptedBlock( m_pStack->pucBlock))
	{
		setBlockEncrypted( pucNewBlock);
	}

	setBytesAvail( pucNewBlock,
		(FLMUINT16)(m_uiBlockSize - sizeofBTreeBlockHdr( pucNewBlock)));
	setHeapSize( pucNewBlock, 
		(FLMUINT16)(m_uiBlockSize - sizeofBTreeBlockHdr( pucNewBlock)));

	// We are going to make changes to the current block.  The pBlock could
	// have changed since making this call, so we need to update the block
	// header
	
	if( RC_BAD( rc = m_pBlockMgr->prepareForUpdate( &m_pStack->pBlock,
		&m_pStack->pucBlock)))
	{
		goto Exit;
	}

	m_pStack->pui16OffsetArray = BtOffsetArray( m_pStack->pucBlock, 0); 

	// Get the current previous block if there is one.
	
	uiBlockAddr = getPrevInChain( m_pStack->pucBlock);

	if( uiBlockAddr)
	{
		if( RC_BAD( rc = m_pBlockMgr->getBlock( (FLMUINT32)uiBlockAddr, &pPrevBlock, 
			&pucPrevBlock)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = m_pBlockMgr->prepareForUpdate( &pPrevBlock, 
			&pucPrevBlock)))
		{
			goto Exit;
		}
	}

	// Link the new block between the current and it's previous
	
	setNextInChain( pucNewBlock, m_pStack->ui32BlockAddr);
	setPrevInChain( pucNewBlock, (FLMUINT32)uiBlockAddr);
	setPrevInChain( m_pStack->pucBlock, getBlockAddr( pucNewBlock));

	// There may not be a previous block.
	
	if( pPrevBlock)
	{
		setNextInChain( pucPrevBlock, getBlockAddr( pucNewBlock));
		pPrevBlock->Release();
		pPrevBlock = NULL;
		pucPrevBlock = NULL;
	}

	// We will move all entries in the current block up to but NOT including
	// the entry pointed to by uiCurOffset to the new block.
	
	if( m_pStack->uiCurOffset > 0)
	{
		if( RC_BAD( rc = moveToPrev( 0, m_pStack->uiCurOffset - 1, &pNewBlock,
			&pucNewBlock)))
		{
			goto Exit;
		}
		
		// All entries prior to the old insertion point were moved.
		// Therefore, the new insertion point must be at the beginning.
		
		m_pStack->uiCurOffset = 0;

		// If we emptied the block.  This will require us to update the parent.

		if( getNumKeys( m_pStack->pucBlock) == 0)
		{
			if (RC_BAD( rc = saveReplaceInfo( pucKey, uiKeyLen)))
			{
				goto Exit;
			}
			
			bSavedReplaceInfo = TRUE;
		}
	}

	// If the block is now empty, we will store a partial entry in it here.
	// This scenario only occurs when we are engaged in a ReplaceByInsert
	// operation. Normal inserts would never result in an empty block.
	// Since we know we are part of a replace operation, we know that the
	// parent of this block only needs the counts updated, not the key.
	
	if( m_pStack->uiLevel == 0 && getNumKeys( m_pStack->pucBlock) == 0)
	{
		if( RC_BAD( rc = storePartialEntry( pucKey, uiKeyLen, pucValue, uiLen,
			uiFlags, uiChildBlockAddr, uiCounts, ppucRemainingValue, 
			puiRemainingLen, FALSE)))
		{
			goto Exit;
		}

		goto MoveToPrev;
	}

	// Is there room for the new entry now in the current block?
	
	if( RC_BAD( rc = calcNewEntrySize( uiKeyLen, uiLen, &uiEntrySize,
		&bHaveRoom, &bDefragBlock)))
	{
		goto Exit;
	}

	if( bHaveRoom)
	{
		if( bDefragBlock)
		{
			if( RC_BAD( rc = defragmentBlock( &m_pStack->pBlock, 
				&m_pStack->pucBlock)))
			{
				goto Exit;
			}
		}

		if( RC_BAD( rc = storeEntry( pucKey, uiKeyLen, pucValue, uiLen,
				uiFlags, uiOADataLen, uiChildBlockAddr, uiCounts, uiEntrySize,
				&bLastEntry)))
		{
			goto Exit;
		}

		if( bLastEntry && !bSavedReplaceInfo)
		{
			// Since we just added/replaced an entry to the last position of the
			// current block. we will need to preserve the current stack so that
			// we can finish updating the parentage later. Should only happen as
			// a result of a replace operation where the new entry is larger than
			// the existing one while in the upper levels.
			
			if( RC_BAD( rc = saveReplaceInfo( pucKey, uiKeyLen)))
			{
				goto Exit;
			}
		}

		// If we are keeping counts, we must update those too.
		
		if( m_bCounts && !isRootBlock( m_pStack->pucBlock))
		{
			if( RC_BAD( rc = updateCounts()))
			{
				goto Exit;
			}
		}

		if( m_pStack->uiLevel == 0)
		{
			*ppucRemainingValue = NULL;
			*puiRemainingLen = 0;
		}

		goto MoveToPrev;
	}

	// Can we store the whole thing in the new block?
	
	if( uiEntrySize <= getBytesAvail( pucNewBlock))
	{
		// If this block has a parent block, and the btree is maintaining counts
		// we will want to update the counts on the parent block.
		
		if( m_bCounts && !isRootBlock( m_pStack->pucBlock))
		{
			if (RC_BAD( rc = updateCounts()))
			{
				goto Exit;
			}
		}

		// We can release the current block since it is no longer needed.
		
		m_pStack->pBlock->Release();
		m_pStack->pBlock = pNewBlock;
		m_pStack->pucBlock = pucNewBlock;
		
		pNewBlock = NULL;
		pucNewBlock = NULL;
		
		m_pStack->ui32BlockAddr = getBlockAddr( m_pStack->pucBlock);
		m_pStack->pui16OffsetArray = BtOffsetArray( m_pStack->pucBlock, 0);

		// Setting the uiCurOffset to the actual number of keys will cause the
		// new entry to go in as the last element.
		
		m_pStack->uiCurOffset = getNumKeys( m_pStack->pucBlock);

		// We don't need to check to see if we need to defragment this block
		// because it is "new".  Anything that just got written to it will
		// be contiguous already.

		if( RC_BAD( rc = storeEntry( pucKey, uiKeyLen, pucValue, uiLen,
			uiFlags, uiOADataLen, uiChildBlockAddr, uiCounts, uiEntrySize,
			&bLastEntry)))
		{
			goto Exit;
		}

		f_assert( bLastEntry);

		if( m_pStack->uiLevel == 0)
		{
			*ppucRemainingValue = NULL;
			*puiRemainingLen = 0;
		}

		bMovedToPrev = TRUE;
		goto MoveToPrev;
	}

	// Can we store part of the new entry into the new block?
	// Calculate the minimum entry size to store.
	
	if( RC_BAD( rc = calcNewEntrySize( uiKeyLen, 1, &uiMinEntrySize,
		&bHaveRoom, &bDefragBlock)))
	{
		goto Exit;
	}

	// bHaveRoom refers to the current block, and we want to put this into
	// the previous  block.
	
	if( uiMinEntrySize <= getBytesAvail( pucNewBlock))
	{
		// If this block has a parent block, and the btree is maintaining counts
		// we will want to update the counts on the parent block.
		
		if( !isRootBlock( m_pStack->pucBlock))
		{
			if( m_bCounts)
			{
				if( RC_BAD( rc = updateCounts()))
				{
					goto Exit;
				}
			}
		}

		// We can release the current block since it is no longer needed.
		
		m_pStack->pBlock->Release();
		m_pStack->pBlock = pNewBlock;
		m_pStack->pucBlock = pucNewBlock;
		
		pNewBlock = NULL;
		pucNewBlock = NULL;
		
		m_pStack->ui32BlockAddr = getBlockAddr( m_pStack->pucBlock);
		m_pStack->pui16OffsetArray = BtOffsetArray( m_pStack->pucBlock, 0);

		// Setting the uiCurOffset to the actual number of keys will cause the
		// new entry to go in as the last element.
		
		m_pStack->uiCurOffset = getNumKeys( m_pStack->pucBlock);

		if( RC_BAD( rc = storePartialEntry( pucKey, uiKeyLen, pucValue,
			uiLen, uiFlags, uiChildBlockAddr, uiCounts, ppucRemainingValue,
			puiRemainingLen, TRUE)))
		{
			goto Exit;
		}

		bMovedToPrev = TRUE;
	}
	else if( uiMinEntrySize <= getBytesAvail( m_pStack->pucBlock))
	{
		// We will store part of the entry in the current block
		
		if( RC_BAD( rc = storePartialEntry(
			pucKey, uiKeyLen, pucValue, uiLen, uiFlags, uiChildBlockAddr, uiCounts,
			ppucRemainingValue, puiRemainingLen, FALSE)))
		{
			goto Exit;
		}
	}
	else
	{
		// Couldn't store anything, so try again after updating the parents.
		
		*ppucRemainingValue = pucValue;
		*puiRemainingLen = uiLen;
	}

MoveToPrev:

	if( *pbBlockSplit)
	{
		// Release the current entry if it hasn't already been released.
		
		if( !bMovedToPrev && RC_OK( rc))
		{

			// If this block has a parent block, and the btree is maintaining counts
			// we will want to update the counts on the parent block.
			
			if( !isRootBlock( m_pStack->pucBlock) && m_bCounts)
			{
				if( RC_BAD( rc = updateCounts()))
				{
					goto Exit;
				}
			}

			f_assert( pNewBlock);
			f_assert( pucNewBlock);

			m_pStack->pBlock->Release();
			m_pStack->pBlock = pNewBlock;
			m_pStack->pucBlock = pucNewBlock;
			
			pNewBlock = NULL;
			pucNewBlock = NULL;
			
			m_pStack->ui32BlockAddr = getBlockAddr( m_pStack->pucBlock);
			m_pStack->uiCurOffset = getNumKeys( m_pStack->pucBlock) - 1;
			m_pStack->pui16OffsetArray = BtOffsetArray( m_pStack->pucBlock, 0);
		}
	}

Exit:

	if( pPrevBlock)
	{
		pPrevBlock->Release();
	}

	if( pNewBlock)
	{
		pNewBlock->Release();
	}

	return( rc);
}

/***************************************************************************
Desc: Function to create a new level in the Btree.
		This function will ensure that the F_BTSK stack is consistent with
		the way it was configured before the function was called.

		This function will create a new block and copy the current contents
		of the root block into it.  It will then insert a single entry into
		the root block to point to the new child.

		Note that there is a maximum of F_BTREE_MAX_LEVELS levels to the Btree.
		Any effort to exceed that level will result in an error.
****************************************************************************/
RCODE F_BTree::createNewLevel( void)
{
	RCODE					rc = NE_FLM_OK;
	IF_Block *			pNewBlock = NULL;
	FLMBYTE *			pucSrcBlock;
	FLMBYTE *			pucDstBlock;
	FLMBYTE *			pucNewBlock;
	FLMUINT				uiCounts = 0;
	FLMBYTE *			pucEntry;
	FLMBYTE *			pucNull = NULL;
	FLMBYTE				ucBuffer[ FLM_MAX_KEY_SIZE + BTE_NLC_KEY_START];
	FLMUINT				uiMaxNLKey = FLM_MAX_KEY_SIZE + BTE_NLC_KEY_START;
	FLMUINT				uiEntrySize;
	F_BTSK *				pRootStack;
	FLMUINT				uiFlags;
	FLMUINT32			ui32NewBlockAddr;

	// Assert that we are looking at the root block!
	
	f_assert( isRootBlock( m_pStack->pucBlock));

	// Check the root level
	
	if( m_pStack->uiLevel >= F_BTREE_MAX_LEVELS - 1)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_FULL);
		goto Exit;
	}

	// Create a new block to copy the contents of the root block into
	
	if( RC_BAD( rc = m_pBlockMgr->createBlock( &pNewBlock, 
		&pucNewBlock, &ui32NewBlockAddr)))
	{
		RC_UNEXPECTED_ASSERT( rc);
		goto Exit;
	}
	
	setBlockAddr( pucNewBlock, ui32NewBlockAddr);

	// Log that we are about to change the root block
	
	if( RC_BAD( rc = m_pBlockMgr->prepareForUpdate( &m_pStack->pBlock,
		&m_pStack->pucBlock)))
	{
		goto Exit;
	}

	// Update the stack since the pBlock could have changed
	
	m_pStack->pui16OffsetArray = BtOffsetArray( m_pStack->pucBlock, 0);

	// Copy the data from the root block to the new block
	
	pucSrcBlock = (FLMBYTE *)m_pStack->pui16OffsetArray;

	// Check for encryption

	if( isEncryptedBlock( m_pStack->pucBlock))
	{
		setBlockEncrypted( pucNewBlock);
	}

	pucDstBlock = (FLMBYTE *)BtOffsetArray( pucNewBlock, 0);

	unsetRootBlock( pucNewBlock);
	setBTreeId( pucNewBlock, getBTreeId( m_pStack->pucBlock));
	setNumKeys( pucNewBlock, getNumKeys( m_pStack->pucBlock));
	setBlockLevel( pucNewBlock, getBlockLevel( m_pStack->pucBlock));
	setHeapSize( pucNewBlock, getHeapSize( m_pStack->pucBlock));
	setBlockType( pucNewBlock, getBlockType( m_pStack->pucBlock));
	setBytesAvail( pucNewBlock, getBytesAvail( m_pStack->pucBlock));
	setPrevInChain( pucNewBlock, 0);
	setNextInChain( pucNewBlock, 0);

	// Copy the data from the root block to the new block.
	
	f_memcpy( pucDstBlock, pucSrcBlock, 
			m_uiBlockSize - sizeofBTreeBlockHdr( pucNewBlock));

	// Empty out the root block data.

#ifdef FLM_DEBUG
	f_memset( BtOffsetArray( m_pStack->pucBlock, 0),
				 0, m_uiBlockSize - sizeofBTreeBlockHdr( m_pStack->pucBlock));
#endif

	setNumKeys( m_pStack->pucBlock, 0);
	setBytesAvail( m_pStack->pucBlock,
		(FLMUINT16)(m_uiBlockSize - sizeofBTreeBlockHdr( m_pStack->pucBlock))); 
	setHeapSize( m_pStack->pucBlock, getBytesAvail( m_pStack->pucBlock)); 

	// Check the root block type to see if we need to change it.  The root
	// block may have been a leaf node.
	
	if( getBlockType( m_pStack->pucBlock) == F_BLK_TYPE_BT_LEAF ||
		 getBlockType( m_pStack->pucBlock) == F_BLK_TYPE_BT_LEAF_DATA)
	{
		// Need to set the block type to either 
		// F_BLK_TYPE_BT_NON_LEAF or F_BLK_TYPE_BT_NON_LEAF_COUNTS
		
		if( m_bCounts)
		{
			setBlockType( m_pStack->pucBlock, F_BLK_TYPE_BT_NON_LEAF_COUNTS);
		}
		else
		{
			setBlockType( m_pStack->pucBlock, F_BLK_TYPE_BT_NON_LEAF);
		}
	}

	// Now add a new entry to the stack.
	
	pRootStack = m_pStack;
	pRootStack++;

	f_memcpy( pRootStack, m_pStack, sizeof( F_BTSK));

	// Now fix the entries in the stack.
	
	pRootStack->uiLevel++;
	incBlockLevel( pRootStack->pucBlock);
	pRootStack->uiCurOffset = 0;
	pRootStack->pui16OffsetArray = BtOffsetArray( pRootStack->pucBlock, 0);

	m_pStack->pBlock = pNewBlock;
	m_pStack->pucBlock = pucNewBlock;
	
	pNewBlock = NULL;
	pucNewBlock = NULL;

	m_pStack->ui32BlockAddr = getBlockAddr( m_pStack->pucBlock);
	m_pStack->pui16OffsetArray = BtOffsetArray( m_pStack->pucBlock, 0);

	// Build a new entry for the root block that will point to the newly created
	// child block. If the root block type is F_BLK_TYPE_BT_NON_LEAF_COUNTS, then we
	// need to sum the counts from the child block
	
	if( m_bCounts)
	{
		uiCounts = countKeys( m_pStack->pucBlock);
	}

	// Create and insert a LEM entry to mark the last position in the block.
	
	uiFlags = BTE_FLAG_LAST_ELEMENT | BTE_FLAG_FIRST_ELEMENT;

	if( RC_BAD( rc = buildAndStoreEntry(
		getBlockType( pRootStack->pucBlock),
		uiFlags, pucNull, 0, pucNull, 0, 0, m_pStack->ui32BlockAddr,
		uiCounts, &ucBuffer[ 0], uiMaxNLKey, &uiEntrySize)))
	{
		goto Exit;
	}

	// Copy the entry into the root block.
	
	pucEntry = (FLMBYTE *)pRootStack->pucBlock + m_uiBlockSize - uiEntrySize;
	f_memcpy( pucEntry, &ucBuffer[ 0], uiEntrySize);
	bteSetEntryOffset( pRootStack->pui16OffsetArray, 0,
							 (FLMUINT16)(pucEntry - pRootStack->pucBlock));

	incNumKeys( pRootStack->pucBlock);
	decBytesAvail( pRootStack->pucBlock, uiEntrySize + 2);
	decHeapSize( pRootStack->pucBlock, uiEntrySize + 2);

	m_uiStackLevels++;
	m_uiRootLevel++;

Exit:

	if( pNewBlock)
	{
		pNewBlock->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:	Method to calculate the optimal data length size to store.  This 
		method is called when storing a partial entry, and we need to know
		what the largest data size we c an store is.
****************************************************************************/
RCODE F_BTree::calcOptimalDataLength(
	FLMUINT			uiKeyLen,
	FLMUINT			uiDataLen,
	FLMUINT			uiBytesAvail,
	FLMUINT *		puiNewDataLen)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiFixedAmounts;
	FLMUINT			uiRemainder;

	switch( getBlockType( m_pStack->pucBlock))
	{
		case F_BLK_TYPE_BT_LEAF:
		case F_BLK_TYPE_BT_NON_LEAF:
		case F_BLK_TYPE_BT_NON_LEAF_COUNTS:
		{
			// These blocks do not have any data.
			
			*puiNewDataLen = 0;
			break;
		}

		case F_BLK_TYPE_BT_LEAF_DATA:
		{
			// These amounts don't change. Note that the overhead includes the
			// Overall Data Length Field, even though it may not be there in
			// the end.
			
			uiFixedAmounts = BTE_LEAF_DATA_OVHD +
								  (uiKeyLen > ONE_BYTE_SIZE ? 2 : 1) +
								  uiKeyLen;

			uiRemainder = uiBytesAvail - uiFixedAmounts;

			if (uiRemainder >= (ONE_BYTE_SIZE + 2))
			{
				*puiNewDataLen = uiRemainder - 2;
			}
			else
			{
				*puiNewDataLen = uiRemainder - 1;
			}
			break;
		}

		default:
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
			goto Exit;
		}
	}

	if( uiDataLen < *puiNewDataLen)
	{
		*puiNewDataLen = uiDataLen;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	This function will count the total number of keys in the block. 
		Typically the value ui16NumKeys will yield this number, however, if
		the block type is F_BLK_TYPE_BT_NON_LEAF_COUNTS, we also want to include the 
		counts in each entry.
****************************************************************************/
RCODE F_BTree::updateParentCounts(
	FLMBYTE *			pucChildBlock,
	IF_Block **			ppParentBlock,
	FLMBYTE **			ppucParentBlock,
	FLMUINT				uiParentElm)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT				uiCounts;
	FLMBYTE *			pucCounts;

	f_assert( getBlockType( *ppucParentBlock) == F_BLK_TYPE_BT_NON_LEAF_COUNTS);
	uiCounts = countKeys( pucChildBlock);

	if( RC_BAD( rc = m_pBlockMgr->prepareForUpdate( ppParentBlock, 
		ppucParentBlock)))
	{
		goto Exit;
	}

	pucCounts = BtEntry( *ppucParentBlock, uiParentElm);
	UD2FBA( (FLMUINT32)uiCounts, &pucCounts[ BTE_NLC_COUNTS]);

Exit:

	return( rc);
}

/***************************************************************************
Desc:	This function will count the total number of keys in the block.
		Typically the value ui16NumKeys will yield this number, however, if 
		the block type is F_BLK_TYPE_BT_NON_LEAF_COUNTS, we also want to include the 
		counts in each entry.
****************************************************************************/
FLMUINT F_BTree::countKeys(
	FLMBYTE *			pucBlock)
{
	FLMUINT				uiTotal = 0;
	FLMUINT				uiIndex;
	FLMBYTE *			pucEntry;
	FLMUINT16 *			puiOffsetArray;

	puiOffsetArray = BtOffsetArray( pucBlock, 0);

	if( getBlockType( pucBlock) != F_BLK_TYPE_BT_NON_LEAF_COUNTS)
	{
		uiTotal = getNumKeys( pucBlock);
	}
	else
	{
		for (uiIndex = 0; uiIndex < getNumKeys( pucBlock); uiIndex++)
		{
			pucEntry = BtEntry( pucBlock, uiIndex);
			uiTotal += FB2UD( &pucEntry[ BTE_NLC_COUNTS]);
		}
	}

	return( uiTotal);
}

/***************************************************************************
Desc: Function to store an entry in a Data-only block.
****************************************************************************/
RCODE F_BTree::storeDataOnlyBlocks(
	const FLMBYTE *	pucKey,
	FLMUINT				uiKeyLen,
	FLMBOOL				bSaveKey,
	const FLMBYTE *	pucData,
	FLMUINT				uiDataLen)
{
	RCODE					rc = NE_FLM_OK;
	IF_Block *			pPrevBlock = NULL;
	FLMBYTE *			pucPrevBlock = NULL;
	const FLMBYTE *	pucLocalData = pucData;
	FLMUINT				uiDataToWrite = uiDataLen;
	FLMBYTE *			pDestPtr = NULL;
	FLMUINT				uiAmtToCopy;
	FLMUINT32			ui32TmpBlockAddr;

	if( bSaveKey)
	{
		if( !m_pBlock)
		{
			if( RC_BAD( rc = m_pBlockMgr->getBlock( m_ui32CurBlockAddr, 
				&m_pBlock, &m_pucBlock)))
			{
				goto Exit;
			}
		}

		if( RC_BAD( rc = m_pBlockMgr->prepareForUpdate( &m_pBlock, &m_pucBlock)))
		{
			goto Exit;
		}

		// Assert that the current block is empty and has no previous link.
		
		f_assert( getBytesAvail( m_pucBlock) == 
					 m_uiBlockSize - sizeofDOBlockHdr( m_pucBlock));
		f_assert( getPrevInChain( m_pucBlock) == 0);

		pDestPtr = m_pucBlock + sizeofDOBlockHdr( m_pucBlock);

		UW2FBA( (FLMUINT16)uiKeyLen, pDestPtr);
		pDestPtr += sizeof( FLMUINT16);

		f_memcpy( pDestPtr, pucKey, uiKeyLen);
		pDestPtr += uiKeyLen;
		
		m_uiDataRemaining -= (uiKeyLen + sizeof( FLMUINT16));
		setBytesAvail( m_pucBlock, (FLMUINT16)m_uiDataRemaining);
	}

	while( uiDataToWrite > 0)
	{
		if( !m_pBlock)
		{
			if( RC_BAD( rc = m_pBlockMgr->getBlock( m_ui32CurBlockAddr, 
				&m_pBlock, &m_pucBlock)))
			{
				goto Exit;
			}
		}

		if( RC_BAD( rc = m_pBlockMgr->prepareForUpdate( &m_pBlock, &m_pucBlock)))
		{
			goto Exit;
		}

		if( !bSaveKey)
		{
			// Now copy as much of the remaining data as we  can into the new block.
			
			pDestPtr = m_pucBlock + sizeofDOBlockHdr( m_pucBlock);
			pDestPtr += (m_uiBlockSize - sizeofDOBlockHdr( m_pucBlock) - m_uiDataRemaining);
		}
		else
		{
			bSaveKey = FALSE;
		}

		uiAmtToCopy = (uiDataToWrite <= m_uiDataRemaining
								? uiDataToWrite
								: m_uiDataRemaining);

		f_memcpy( pDestPtr, pucLocalData, uiAmtToCopy);

		m_uiDataRemaining -= uiAmtToCopy;
		m_uiOADataLength += uiAmtToCopy;
		uiDataToWrite -= uiAmtToCopy;
		pucLocalData += uiAmtToCopy;
		setBytesAvail( m_pucBlock, (FLMUINT16)m_uiDataRemaining);

		// Now get the next block (if needed)
		
		if( uiDataToWrite)
		{
			pPrevBlock = m_pBlock;
			pucPrevBlock = m_pucBlock;
			
			m_pBlock = NULL;
			m_pucBlock = NULL;

			// Now create a new block
			
			if( RC_BAD( rc = m_pBlockMgr->createBlock( 
				&m_pBlock, &m_pucBlock, &ui32TmpBlockAddr)))
			{
				goto Exit;
			}
			
			setBlockAddr( m_pucBlock, ui32TmpBlockAddr);
			setBlockType( m_pucBlock, F_BLK_TYPE_BT_DATA_ONLY);
			setPrevInChain( m_pucBlock, getBlockAddr( pucPrevBlock));
			setNextInChain( m_pucBlock, 0);

//			if( m_pLFile->uiEncId)
//			{
//				((F_ENC_DO_BLK_HDR *)pBlockHdr)->ui32EncId = (FLMUINT32)m_pLFile->uiEncId;
//				setBlockEncrypted( pBlockHdr);
//			}

			setBytesAvail( m_pucBlock, 
				(FLMUINT16)(m_uiBlockSize - sizeofDOBlockHdr( m_pucBlock)));
			setNextInChain( pucPrevBlock, getBlockAddr( m_pucBlock));

			m_ui32CurBlockAddr = getBlockAddr( m_pucBlock);
			m_uiDataRemaining = m_uiBlockSize - sizeofDOBlockHdr( m_pucBlock);

			if( pPrevBlock)
			{
				pPrevBlock->Release();
				pPrevBlock = NULL;
				pucPrevBlock = NULL;
			}
		}
	}

Exit:

	if( m_pBlock)
	{
		m_pBlock->Release();
		m_pBlock = NULL;
		m_pucBlock = NULL;
	}
	
	if( pPrevBlock)
	{
		pPrevBlock->Release();
	}
	
	return( rc);
}

/***************************************************************************
Desc: Function to Replace data in data only blocks.
****************************************************************************/
RCODE F_BTree::replaceDataOnlyBlocks(
	const FLMBYTE *	pucKey,
	FLMUINT				uiKeyLen,
	FLMBOOL				bSaveKey,
	const FLMBYTE *	pucData,
	FLMUINT				uiDataLen,
	FLMBOOL				bLast,
	FLMBOOL				bTruncate)
{
	RCODE						rc = NE_FLM_OK;
	IF_Block *				pPrevBlock = NULL;
	const FLMBYTE *		pucLocalData = pucData;
	FLMUINT					uiDataToWrite = uiDataLen;
	FLMBYTE *				pDestPtr = NULL;
	FLMBYTE *				pucPrevBlock = NULL;
	FLMUINT					uiAmtToCopy;
	FLMUINT32				ui32NextBlockAddr;
	FLMUINT32				ui32TmpBlockAddr;

	// Do we need to store the key too?
	
	if( bSaveKey)
	{
		if( !m_pBlock)
		{
			if( RC_BAD( rc = m_pBlockMgr->getBlock( m_ui32CurBlockAddr, 
				&m_pBlock, &m_pucBlock)))
			{
				goto Exit;
			}
		}

		if( RC_BAD( rc = m_pBlockMgr->prepareForUpdate( &m_pBlock, &m_pucBlock)))
		{
			goto Exit;
		}

		// Assert that the current block is empty and has no previous link.
		
		f_assert( getBytesAvail( m_pucBlock) == 
					 m_uiBlockSize - sizeofDOBlockHdr( m_pucBlock));
		f_assert( getPrevInChain( m_pucBlock) == 0);

		pDestPtr = m_pucBlock + sizeofDOBlockHdr( m_pucBlock);

		UW2FBA( (FLMUINT16)uiKeyLen, pDestPtr);
		pDestPtr += sizeof( FLMUINT16);

		f_memcpy( pDestPtr, pucKey, uiKeyLen);
		pDestPtr += uiKeyLen;
		
		m_uiDataRemaining -= (uiKeyLen + sizeof( FLMUINT16));
		setBytesAvail( m_pucBlock, (FLMUINT16)m_uiDataRemaining);
	}

	while( uiDataToWrite > 0)
	{
		if( !m_pBlock)
		{
			if( RC_BAD( rc = m_pBlockMgr->getBlock( m_ui32CurBlockAddr, 
				&m_pBlock, &m_pucBlock)))
			{
				goto Exit;
			}
		}

		if( RC_BAD( rc = m_pBlockMgr->prepareForUpdate( &m_pBlock, &m_pucBlock)))
		{
			goto Exit;
		}

		if( !bSaveKey)
		{
			// Now copy as much of the remaining data as we  can into the new block.
			
			pDestPtr = m_pucBlock + sizeofDOBlockHdr( m_pucBlock);
			pDestPtr += (m_uiBlockSize - sizeofDOBlockHdr( m_pucBlock) -
							m_uiDataRemaining);
		}
		else
		{
			bSaveKey = FALSE;
		}

		uiAmtToCopy = (uiDataToWrite <= m_uiDataRemaining
								? uiDataToWrite
								: m_uiDataRemaining);

		f_memcpy( pDestPtr, pucLocalData, uiAmtToCopy);

		m_uiDataRemaining -= uiAmtToCopy;
		m_uiOADataLength += uiAmtToCopy;
		uiDataToWrite -= uiAmtToCopy;
		pucLocalData += uiAmtToCopy;

		if( bTruncate || (m_uiDataRemaining < getBytesAvail( m_pucBlock)))
		{
			setBytesAvail( m_pucBlock, (FLMUINT16)m_uiDataRemaining);
		}

		// Now get the next block (if needed)
		
		if( uiDataToWrite)
		{
			pPrevBlock = m_pBlock;
			pucPrevBlock = m_pucBlock;
			
			m_pBlock = NULL;
			m_pucBlock = NULL;
			
			ui32NextBlockAddr = getNextInChain( pucPrevBlock);
			
			if( ui32NextBlockAddr)
			{
				if( RC_BAD( rc = m_pBlockMgr->getBlock( ui32NextBlockAddr, 
					&m_pBlock, &m_pucBlock)))
				{
					goto Exit;
				}

				if( RC_BAD( rc = m_pBlockMgr->prepareForUpdate( 
					&m_pBlock, &m_pucBlock)))
				{
					goto Exit;
				}
			}
			else
			{
				// Now create a new block
				
				if( RC_BAD( rc = m_pBlockMgr->createBlock( &m_pBlock, 
					&m_pucBlock, &ui32TmpBlockAddr)))
				{
					goto Exit;
				}

				setBlockAddr( m_pucBlock, ui32TmpBlockAddr);
				setBlockType( m_pucBlock, F_BLK_TYPE_BT_DATA_ONLY);
				setPrevInChain( m_pucBlock, getBlockAddr( pucPrevBlock));
				setNextInChain( m_pucBlock, 0);

//				if( m_pLFile->uiEncId)
//				{
//					setBlockEncrypted( pBlockHdr);
//					((F_ENC_DO_BLK_HDR *)pBlockHdr)->ui32EncId = (FLMUINT32)m_pLFile->uiEncId;
//				}

				setBytesAvail( m_pucBlock, 
					(FLMUINT16)(m_uiBlockSize - sizeofDOBlockHdr( m_pucBlock)));
			}

			setNextInChain( pucPrevBlock, getBlockAddr( m_pucBlock));
			m_ui32CurBlockAddr = getBlockAddr( m_pucBlock);
			m_uiDataRemaining = m_uiBlockSize - sizeofDOBlockHdr( m_pucBlock);

			if( pPrevBlock)
			{
				pPrevBlock->Release();
				pPrevBlock = NULL;
				pucPrevBlock = NULL;
			}
		}
	}

	// If this was the last pass to store the data, then see if we need to
	// remove any left over blocks.  We will not truncate the data if
	// the bTruncate parameter is not set.
	
	if( bLast && bTruncate)
	{
		f_assert( m_pBlock && m_pucBlock);

		ui32NextBlockAddr = getNextInChain( m_pucBlock);
		setNextInChain( m_pucBlock, 0);
		
		m_pBlock->Release();
		m_pBlock = NULL;
		m_pucBlock = NULL;

		// If there are any blocks left over, they must be freed.
		
		while( ui32NextBlockAddr)
		{
			if( RC_BAD( rc = m_pBlockMgr->getBlock( ui32NextBlockAddr, 
				&m_pBlock, &m_pucBlock)))
			{
				goto Exit;
			}
			
			ui32NextBlockAddr = getNextInChain( m_pucBlock);

			if( RC_BAD( rc = m_pBlockMgr->freeBlock( &m_pBlock, &m_pucBlock)))
			{
				goto Exit;
			}
		}
	}

Exit:

	if( m_pBlock)
	{
		m_pBlock->Release();
		m_pBlock = NULL;
		m_pucBlock = NULL;
	}
	
	if( pPrevBlock)
	{
		pPrevBlock->Release();
	}
	
	return( rc);
}

/***************************************************************************
Desc:	Method to construct a new leaf entry using the key and value
		information passed in.
****************************************************************************/
RCODE F_BTree::buildAndStoreEntry(
	FLMUINT				uiBlockType,
	FLMUINT				uiFlags,
	const FLMBYTE *	pucKey,
	FLMUINT				uiKeyLen,
	const FLMBYTE *	pucData,
	FLMUINT				uiDataLen,
	FLMUINT				uiOADataLen,		// If zero, it will not be used.
	FLMUINT				uiChildBlockAddr,
	FLMUINT				uiCounts,
	FLMBYTE *			pucBuffer,
	FLMUINT				uiBufferSize,
	FLMUINT *			puiEntrySize)
{
	RCODE				rc = NE_FLM_OK;
	FLMBYTE *		pucTemp = pucBuffer;

	if( puiEntrySize)
	{
		*puiEntrySize = calcEntrySize( uiBlockType, uiFlags, 
									uiKeyLen, uiDataLen, uiOADataLen);

		if( !(*puiEntrySize) || *puiEntrySize > uiBufferSize)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_CONV_DEST_OVERFLOW);
			goto Exit;
		}
	}

	switch( uiBlockType)
	{
		case F_BLK_TYPE_BT_LEAF:
		{
			// No Data in this entry, so it is easy to make.

			UW2FBA( (FLMUINT16)uiKeyLen, pucTemp);
			pucTemp += 2;
			
			f_memcpy( pucTemp, pucKey, uiKeyLen);
			break;
		}

		case F_BLK_TYPE_BT_LEAF_DATA:
		{
			// Make sure the correct flags are set...

			if( uiKeyLen > ONE_BYTE_SIZE)
			{
				uiFlags |= BTE_FLAG_KEY_LEN;
			}
			else
			{
				uiFlags &= ~BTE_FLAG_KEY_LEN;
			}

			if( uiDataLen > ONE_BYTE_SIZE)
			{
				uiFlags |= BTE_FLAG_DATA_LEN;
			}
			else
			{
				uiFlags &= ~BTE_FLAG_DATA_LEN;
			}

			// Only the first element of an entry that spans elements
			// will hold an OADataLen field.

			if( uiOADataLen && (uiFlags & BTE_FLAG_FIRST_ELEMENT))
			{
				uiFlags |= BTE_FLAG_OA_DATA_LEN;
			}
			else
			{
				uiFlags &= ~BTE_FLAG_OA_DATA_LEN;
			}

			// Now start setting the elements of the entry.
			// Flags first.

			*pucTemp = (FLMBYTE)uiFlags;
			pucTemp++;

			// KeyLen

			if( uiFlags & BTE_FLAG_KEY_LEN)
			{
				UW2FBA( (FLMUINT16)uiKeyLen, pucTemp);
				pucTemp += 2;
			}
			else
			{
				*pucTemp = (FLMBYTE)uiKeyLen;
				pucTemp++;
			}

			if( uiFlags & BTE_FLAG_DATA_LEN)
			{
				UW2FBA( (FLMUINT16)uiDataLen, pucTemp);
				pucTemp += 2;
			}
			else
			{
				*pucTemp = (FLMBYTE)uiDataLen;
				pucTemp++;
			}

			if( uiFlags & BTE_FLAG_OA_DATA_LEN)
			{
				UD2FBA( (FLMUINT32)uiOADataLen, pucTemp);
				pucTemp += 4;
			}

			// Key

			f_memcpy( pucTemp, pucKey, uiKeyLen);
			pucTemp += uiKeyLen;

			// Data

			f_memcpy( pucTemp, pucData, uiDataLen);
			break;
		}

		case F_BLK_TYPE_BT_NON_LEAF:
		case F_BLK_TYPE_BT_NON_LEAF_COUNTS:
		{
			// Child block address - 4 bytes

			pucTemp = pucBuffer;

			f_assert( uiChildBlockAddr);
			UD2FBA( (FLMUINT32)uiChildBlockAddr, pucTemp);
			pucTemp += 4;

			// Counts - 4 bytes

			if( uiBlockType == F_BLK_TYPE_BT_NON_LEAF_COUNTS)
			{
				UD2FBA( (FLMUINT32)uiCounts, pucTemp);
				pucTemp += 4;
			}

			// KeyLen field - 2 bytes

			UW2FBA( (FLMUINT16)uiKeyLen, pucTemp);
			pucTemp += 2;

			// Key - variable length (uiKeyLen)

			f_memcpy( pucTemp, pucKey, uiKeyLen);
			break;
		}

		default:
		{
			// Invalid block type

			rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method to remove an entry from a block. This method will delete the
		entry pointed to by the current Stack. This method does NOT defragment
		the block.  If the entry points to any data only blocks, they will
		also be removed from circulation if the parameter bDeleteDOBlocks is
		set to true.  Otherwise, they will not be freed.  This is so we can
		call this method when we are moving entries between blocks or
		replacing entries etc.
****************************************************************************/
RCODE F_BTree::remove(
	FLMBOOL			bDeleteDOBlocks)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT16 *			pui16OffsetArray;
	FLMUINT				uiNumKeys;
	FLMUINT				uiEntrySize;
	FLMUINT				uiTmp;
	FLMBYTE *			pucEntry;
	FLMBOOL				bDOBlock;
	IF_Block *			pBlock = NULL;
	FLMBYTE *			pucBlock = NULL;
	FLMUINT				uiBlockAddr;
	FLMBYTE *			pucEndOfHeap;

	if( RC_BAD( rc = m_pBlockMgr->prepareForUpdate( &m_pStack->pBlock, 
		&m_pStack->pucBlock)))
	{
		goto Exit;
	}

	m_pStack->pui16OffsetArray = BtOffsetArray( m_pStack->pucBlock, 0);
	uiNumKeys = getNumKeys( m_pStack->pucBlock);

	if( !uiNumKeys)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
		goto Exit;
	}

	// Point to the entry...
	
	pucEntry = BtEntry( m_pStack->pucBlock, m_pStack->uiCurOffset);
	uiEntrySize = getEntrySize( m_pStack->pucBlock, m_pStack->uiCurOffset);

	pucEndOfHeap = m_pStack->pucBlock + sizeofBTreeBlockHdr( m_pStack->pucBlock) +
						(uiNumKeys * 2) + getHeapSize( m_pStack->pucBlock);

	// We are only going to have data only blocks if we are storing data
	// in the btree.
	
	if( m_bTreeHoldsData && blkIsLeaf( m_pStack->pucBlock))
	{
		bDOBlock = bteDataBlockFlag( pucEntry);

		// If the data for this entry is in one or more data-only blocks,
		// we must delete those blocks first.
		
		if( bDOBlock && bDeleteDOBlocks)
		{
			FLMBYTE	ucDOBlockAddr[ 4];

			// Get the block address of the DO Block.
			
			if( RC_BAD( rc = fbtGetEntryData( pucEntry, ucDOBlockAddr,
				sizeof( FLMUINT), NULL)))
			{
				goto Exit;
			}

			uiBlockAddr = bteGetBlockAddr( (FLMBYTE *)&ucDOBlockAddr[ 0]);
			while( uiBlockAddr)
			{
				// We need to delete the data only blocks first.
				
				if( RC_BAD( rc = m_pBlockMgr->getBlock( (FLMUINT32)uiBlockAddr, 
					&pBlock, &pucBlock)))
				{
					goto Exit;
				}

				// Get the next block address (if any)
				
				uiBlockAddr = getNextInChain( pucBlock);

				// Now put the block into the Avail list.
				
				if( RC_BAD( rc = m_pBlockMgr->freeBlock( &pBlock, &pucBlock)))
				{
					goto Exit;
				}
			}
		}
	}

	pui16OffsetArray = m_pStack->pui16OffsetArray;

	// Move the offsets around to effectively remove the entry.
	
	for( uiTmp = m_pStack->uiCurOffset; (uiTmp + 1) < uiNumKeys; uiTmp++)
	{
		bteSetEntryOffset( pui16OffsetArray, uiTmp,
								 bteGetEntryOffset( pui16OffsetArray, (uiTmp + 1)));
	}

#ifdef FLM_DEBUG
	// Erase the last offset entry.
	
	bteSetEntryOffset( pui16OffsetArray, uiTmp, 0);
#endif

	decNumKeys( m_pStack->pucBlock);
	incBytesAvail( m_pStack->pucBlock, (FLMUINT16)uiEntrySize);
	incHeapSize( m_pStack->pucBlock, 2);

	// Was this entry we just removed adjacent to the heap space?  If
	// so then we can increase the heap space.
	
	if( pucEndOfHeap == pucEntry)
	{
		incHeapSize( m_pStack->pucBlock, 
			(FLMUINT16)actualEntrySize( uiEntrySize));
	}

#ifdef FLM_DEBUG
	// Let's erase whatever was in the entry space.
	
	f_memset( pucEntry, 0, actualEntrySize( uiEntrySize));
#endif

Exit:

	if( pBlock)
	{
		pBlock->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:	Method to remove multiple entries from a block. The entries must be
		contiguous.  If any entries store data in data-only blocks, they will
		be freed and put into the avail list.
****************************************************************************/
RCODE F_BTree::removeRange(
	FLMUINT			uiStartElm,
	FLMUINT			uiEndElm,
	FLMBOOL			bDeleteDOBlocks)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT16 *			pui16OffsetArray;
	FLMUINT				uiNumKeys;
	FLMUINT				uiEntrySize;
	FLMBYTE *			pucEntry;
	FLMBOOL				bDOBlock;
	IF_Block *			pBlock = NULL;
	FLMBYTE *			pucBlock = NULL;
	FLMUINT				uiBlockAddr;
	FLMUINT				uiCurOffset;
	FLMUINT				uiCounter;
	FLMBYTE *			pucEndOfHeap;
	FLMBYTE *			pucStartOfHeap;

	if( RC_BAD( rc = m_pBlockMgr->prepareForUpdate( &m_pStack->pBlock,
		&m_pStack->pucBlock)))
	{
		goto Exit;
	}

	m_pStack->pui16OffsetArray = BtOffsetArray( m_pStack->pucBlock, 0);
	uiNumKeys = getNumKeys( m_pStack->pucBlock);
	
	if( !uiNumKeys)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
		goto Exit;
	}

	f_assert( uiEndElm < uiNumKeys);

	// Point to the entry ...
	
	for( uiCurOffset = uiStartElm; uiCurOffset <= uiEndElm; uiCurOffset++)
	{
		pucEntry = BtEntry( m_pStack->pucBlock, uiCurOffset);
		uiEntrySize = getEntrySize( m_pStack->pucBlock, uiCurOffset);
		incBytesAvail( m_pStack->pucBlock, (FLMUINT16)uiEntrySize);
		decNumKeys( m_pStack->pucBlock);

		bDOBlock = bteDataBlockFlag( pucEntry);

		// If the data for this entry is in a Data Only block, then we must delete
		// those blocks first.
		
		if( bDOBlock && bDeleteDOBlocks)
		{
			FLMBYTE	ucDOBlockAddr[ 4];

			// Get the block address of the DO Block.
			
			if( RC_BAD( rc = fbtGetEntryData( pucEntry, ucDOBlockAddr, 4, NULL)))
			{
				goto Exit;
			}

			uiBlockAddr = bteGetBlockAddr( &ucDOBlockAddr[ 0]);
			while( uiBlockAddr)
			{
				// We need to delete the data only blocks first.
				
				if( RC_BAD( rc = m_pBlockMgr->getBlock( (FLMUINT32)uiBlockAddr, 
					&pBlock, &pucBlock)))
				{
					goto Exit;
				}

				// Get the next block address (if any)
				
				uiBlockAddr = getNextInChain( pucBlock);

				// Now put the block into the Avail list.
				
				if( RC_BAD( rc = m_pBlockMgr->freeBlock( &pBlock, &pucBlock)))
				{
					goto Exit;
				}
			}
		}

		// Now erase the old entry
		
#ifdef FLM_DEBUG
		f_memset( pucEntry, 0, actualEntrySize( uiEntrySize));
#endif
	}

	// Move the offsets around to effectively remove the entries.
	
	pui16OffsetArray = m_pStack->pui16OffsetArray;
	if( uiEndElm < (uiNumKeys - 1))
	{
		// We will need to move the remaining offsets forward
		// to delete the desired range.

		for (uiCurOffset = uiStartElm, uiCounter = 0;
			  uiCounter < (uiNumKeys - (uiEndElm + 1));
			  uiCounter++, uiCurOffset++)
		{
			bteSetEntryOffset( pui16OffsetArray, uiCurOffset,
									 bteGetEntryOffset( pui16OffsetArray,
															  (uiEndElm + uiCounter + 1)));
		}
	}

#ifdef FLM_DEBUG
	// Erase the remaining offsets
	
	while (uiCurOffset < (uiNumKeys - 1))
	{
		bteSetEntryOffset( pui16OffsetArray, uiCurOffset++, 0);
	}
	
#endif

	// We need to determine if we have gained any more heap space.  We start
	// by pointing to the end of the block, them moving forward until we reach
	// the closest entry.
	
	pucEndOfHeap = m_pStack->pucBlock + m_uiBlockSize;
	
	for( uiCurOffset = 0; 
		 uiCurOffset < getNumKeys( m_pStack->pucBlock);
		 uiCurOffset++)
	{
		pucEntry = BtEntry( m_pStack->pucBlock, uiCurOffset);
		
		if (pucEntry < pucEndOfHeap)
		{
			pucEndOfHeap = pucEntry;
		}
	}
	
	// Now clean up the heap space.
	
	pucStartOfHeap = m_pStack->pucBlock + 
						  sizeofBTreeBlockHdr( m_pStack->pucBlock) +
						  (getNumKeys( m_pStack->pucBlock) * 2);

	setHeapSize( m_pStack->pucBlock, 
		(FLMUINT16)(pucEndOfHeap - pucStartOfHeap));

#ifdef FLM_DEBUG
	f_memset( pucStartOfHeap, 0, getHeapSize( m_pStack->pucBlock));
#endif

Exit:

	if( pBlock)
	{
		pBlock->Release();
	}
	
	return( rc);
}

/***************************************************************************
Desc:	Method to try to move entries (whole) from the target block to the
		previous block.  The entries may be moved, up to but not including
		the current entry position.  We do not want to change the parentage
		of this block.  We need to use the stack to fix up the parentage of
		the previous block. Entries are moved from the lowest to highest.
****************************************************************************/
RCODE F_BTree::moveEntriesToPrevBlock(
	FLMUINT				uiNewEntrySize,
	IF_Block **			ppPrevBlock,
	FLMBYTE **			ppucPrevBlock,
	FLMBOOL *			pbEntriesWereMoved)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT				uiLocalAvailSpace;
	FLMUINT				uiAvailSpace;
	FLMUINT				uiHeapSize;
	IF_Block *			pPrevBlock = NULL;
	FLMBYTE *			pucPrevBlock = NULL;
	FLMUINT				uiPrevBlockAddr;
	FLMUINT				uiOAEntrySize = 0;
	FLMUINT				uiStart;
	FLMUINT				uiFinish;
	FLMUINT				uiCount;
	FLMUINT				uiOffset;

	f_assert( !(*ppPrevBlock));
	f_assert( !(*ppucPrevBlock));

	// Assume nothing to move.
	
	*pbEntriesWereMoved = FALSE;

	// If we are already at the first entry in the block, there
	// is nothing that we can move since we will always insert ahead of
	// the current position.
	
	if( !m_pStack->uiCurOffset)
	{
		goto Exit;
	}

	// Get the previous block.
	
	if( (uiPrevBlockAddr = getPrevInChain( m_pStack->pucBlock)) == 0)
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pBlockMgr->getBlock( (FLMUINT32)uiPrevBlockAddr, &pPrevBlock,
		&pucPrevBlock)))
	{
		goto Exit;
	}

	uiLocalAvailSpace = getBytesAvail( m_pStack->pucBlock);
	uiAvailSpace = getBytesAvail( pucPrevBlock);
	uiHeapSize = getHeapSize( pucPrevBlock);

	// If we add the available space in this block and the previous block, would
	// it be enough to make room for the new entry?  If so, then we will
	// see if we can make that room by moving ( whole) entries.
	
	if( (uiAvailSpace + uiLocalAvailSpace) < uiNewEntrySize)
	{
		goto Exit;
	}

	uiStart = 0;
	uiFinish = m_pStack->uiCurOffset;

	// Get the size of each entry until we are over the available size limit
	
	for( uiOffset = 0, uiCount = 0 ; uiOffset < uiFinish; uiOffset++)
	{
		FLMUINT		uiLocalEntrySize;

		uiLocalEntrySize = getEntrySize( m_pStack->pucBlock, uiOffset);

		if( (uiLocalEntrySize + uiOAEntrySize) < uiAvailSpace)
		{
			uiOAEntrySize += uiLocalEntrySize;
			uiLocalAvailSpace += uiLocalEntrySize;
			uiCount++;
		}
		else
		{
			break;
		}
	}

	if( !uiCount)
	{
		goto Exit;
	}

	// It looks like we can move at least one entry.
	// Will this give use enough room to store the new entry?
	
	if( uiLocalAvailSpace < uiNewEntrySize)
	{
		// Moving these entries will not benefit us, so don't bother
		
		goto Exit;
	}

	// Do we need to defragment the block first?
	
	if( uiHeapSize < uiOAEntrySize)
	{
		f_assert( uiHeapSize != uiAvailSpace);
		if( RC_BAD( rc = defragmentBlock( &pPrevBlock, &pucPrevBlock)))
		{
			goto Exit;
		}
	}

	// We are going to get some benefit from moving, so let's do it...
	
	if (RC_BAD( rc = moveToPrev( uiStart, uiStart + uiCount - 1, 
		&pPrevBlock, &pucPrevBlock)))
	{
		goto Exit;
	}

	// We will need to return this block.
	
	*ppPrevBlock = pPrevBlock;
	*ppucPrevBlock = pucPrevBlock;
	
	pPrevBlock = NULL;
	pucPrevBlock = NULL;

	// Adjust the current offset in the stack so we are still pointing to the
	// same entry.
	
	m_pStack->uiCurOffset -= uiCount;

	// If this block has a parent block, and the btree is maintaining counts
	// we will want to update the counts on the parent block.
	
	if( !isRootBlock( m_pStack->pucBlock))
	{
		if( m_bCounts)
		{
			if( RC_BAD( rc = updateCounts()))
			{
				goto Exit;
			}
		}
	}

	*pbEntriesWereMoved = TRUE;

Exit:

	if (pPrevBlock)
	{
		pPrevBlock->Release();
	}

	return( rc);
}

/***************************************************************************
Desc: This method will move entries beginning at uiStart, up to and
		including uiFinish from the current block (m_pStack) to pPrevBlock.
		As a part of this operation, both the target block and the source 
		block will be changed.  A call to logPhysBlock will be made before
		each block is changed.  Never move the highest key in the block 
		because we don't want to have to update the parentage of the 
		current block...
****************************************************************************/
RCODE F_BTree::moveToPrev(
	FLMUINT				uiStart,
	FLMUINT				uiFinish,
	IF_Block **			ppPrevBlock,
	FLMBYTE **			ppucPrevBlock)
{
	RCODE							rc = NE_FLM_OK;
	FLMUINT16 *					pui16DstOffsetA = NULL;
	FLMBYTE *					pucSrcEntry;
	FLMBYTE *					pucDstEntry;
	FLMBYTE *					pucTempBlock;
	FLMUINT						uiEntrySize;
	FLMUINT						uiIndex;
	FLMBOOL						bEntriesCombined = FALSE;
	void *						pvPoolMark = m_pool.poolMark();

	// Make sure we have logged the block we are changing.
	// Note that the source block will be logged in the removeRange method.

	if( RC_BAD( rc = m_pBlockMgr->prepareForUpdate( 
		ppPrevBlock, ppucPrevBlock)))
	{
		goto Exit;
	}

	pui16DstOffsetA = BtOffsetArray( *ppucPrevBlock, 0);
	pucDstEntry = getBlockEnd( *ppucPrevBlock);
	
	if( RC_BAD( rc = m_pool.poolAlloc( m_uiBlockSize, (void **)&pucTempBlock)))
	{
		goto Exit;
	}

	// Beginning at the start, copy each entry over from the source
	// to the destination block.
	
	for( uiIndex = uiStart; uiIndex <= uiFinish; uiIndex++)
	{
		if( RC_BAD( rc = combineEntries( m_pStack->pucBlock,
			uiIndex, *ppucPrevBlock, getNumKeys( *ppucPrevBlock)
														? getNumKeys( *ppucPrevBlock) - 1
														: 0,
			&bEntriesCombined, &uiEntrySize, pucTempBlock)))
		{
			goto Exit;
		}

		if( bEntriesCombined)
		{
			F_BTSK		tmpStack;
			F_BTSK *		pTmpStack;

			tmpStack.pBlock = *ppPrevBlock;
			tmpStack.pucBlock = *ppucPrevBlock;
			tmpStack.uiCurOffset = getNumKeys( *ppucPrevBlock) - 1;

			pTmpStack = m_pStack;
			m_pStack = &tmpStack;

			rc = remove( FALSE);
			m_pStack = pTmpStack;
			
			if( RC_BAD( rc))
			{
				goto Exit;
			}

			if( getHeapSize( *ppucPrevBlock) != getBytesAvail( *ppucPrevBlock))
			{
				if( RC_BAD( rc = defragmentBlock( ppPrevBlock, ppucPrevBlock)))
				{
					goto Exit;
				}
			}

			pucDstEntry = getBlockEnd( *ppucPrevBlock) - uiEntrySize;
			f_memcpy( pucDstEntry, pucTempBlock, uiEntrySize);

			bteSetEntryOffset( pui16DstOffsetA,
									 getNumKeys( *ppucPrevBlock),
									 (FLMUINT16)(pucDstEntry - *ppucPrevBlock));
			incNumKeys( *ppucPrevBlock);

			decBytesAvail( *ppucPrevBlock, ((FLMUINT16)uiEntrySize + 2));
			decHeapSize( *ppucPrevBlock, ((FLMUINT16)uiEntrySize + 2));
			bEntriesCombined = FALSE;
		}
		else
		{
			pucSrcEntry = BtEntry( m_pStack->pucBlock, uiIndex);
			uiEntrySize = getEntrySize( m_pStack->pucBlock, uiIndex);
			pucDstEntry -= actualEntrySize( uiEntrySize);

			f_memcpy( pucDstEntry, pucSrcEntry, actualEntrySize( uiEntrySize));

			bteSetEntryOffset( pui16DstOffsetA,
									 getNumKeys( *ppucPrevBlock),
									 (FLMUINT16)(pucDstEntry - *ppucPrevBlock));
									 
			incNumKeys( *ppucPrevBlock);
			decBytesAvail( *ppucPrevBlock, (FLMUINT16)uiEntrySize);
			decHeapSize( *ppucPrevBlock, (FLMUINT16)uiEntrySize);
		}
	}

	// Now remove the entries from the Src block.
	
	if( RC_BAD( rc = removeRange( uiStart, uiFinish, FALSE)))
	{
		goto Exit;
	}

Exit:

	m_pool.poolReset( pvPoolMark);
	return( rc);
}

/***************************************************************************
Desc:	Method to try to move entries (whole) from the target block to the
		next block.  The entries may be moved up to but not including
		the current entry position depending on how much room is available if
		any.  Entries are moved from the highest to lowest.
****************************************************************************/
RCODE F_BTree::moveEntriesToNextBlock(
	FLMUINT		uiNewEntrySize,
	FLMBOOL *	pbEntriesWereMoved)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT				uiLocalAvailSpace;
	FLMUINT				uiAvailSpace;
	FLMUINT				uiHeapSize;
	IF_Block *			pNextBlock = NULL;
	FLMBYTE *			pucNextBlock = NULL;
	FLMUINT				uiNextBlockAddr;
	FLMUINT				uiOAEntrySize = 0;
	FLMUINT				uiStart;
	FLMUINT				uiFinish;
	FLMUINT				uiCount;
	FLMUINT				uiOffset;
	IF_Block *			pChildBlock = NULL;
	IF_Block *			pParentBlock = NULL;
	FLMBYTE *			pucChildBlock = NULL;
	FLMBYTE *			pucParentBlock = NULL;
	F_BTSK *				pParentStack;
	FLMUINT				uiLevel;
	FLMBOOL				bCommonParent = FALSE;

	// Assume nothing to move.
	
	*pbEntriesWereMoved = FALSE;

	// Get the next block.
	
	if( (uiNextBlockAddr = getNextInChain( m_pStack->pucBlock)) == 0)
	{
		goto Exit;
	}

	if( (FLMUINT16)m_pStack->uiCurOffset >= getNumKeys(m_pStack->pucBlock) - 1)
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pBlockMgr->getBlock( (FLMUINT32)uiNextBlockAddr, 
		&pNextBlock, &pucNextBlock)))
	{
		goto Exit;
	}

	// Our first task is to determine if we can move anything at all.
	// How much free space is there in the next block?
	
	uiLocalAvailSpace = getBytesAvail( m_pStack->pucBlock);
	uiAvailSpace = getBytesAvail( pucNextBlock);
	uiHeapSize = getHeapSize( pucNextBlock);

	// If we add the available space in this block and the next block, would
	// it be enough to make room for the new entry?  If so, then we will
	// see if we can make that room by moving ( whole) entries.
	
	if( (uiAvailSpace + uiLocalAvailSpace) < uiNewEntrySize)
	{
		goto Exit;
	}

	// Begin at the last entry and work backward.
	
	uiStart = getNumKeys( m_pStack->pucBlock) - 1;
	uiFinish = m_pStack->uiCurOffset;

	// Get the size of each entry (plus 2 for the offset entry) until we are
	// over the available size limit.
	
	for( uiOffset = uiStart, uiCount = 0 ; uiOffset > uiFinish; uiOffset--)
	{
		FLMUINT		uiLocalEntrySize;

		uiLocalEntrySize = getEntrySize( m_pStack->pucBlock, uiOffset);

		if( (uiLocalEntrySize + uiOAEntrySize) < uiAvailSpace)
		{
			uiOAEntrySize += uiLocalEntrySize;
			uiLocalAvailSpace += uiLocalEntrySize;
			uiCount++;
		}
		else
		{
			break;
		}
	}

	if( uiCount == 0)
	{
		goto Exit;
	}

	// It looks like we can move at least one entry.
	// Will this give use enough room to store the new entry?
	
	if( uiLocalAvailSpace < uiNewEntrySize)
	{
		goto Exit;
	}

	f_assert( uiStart > uiFinish);

	// Do we need to defragment the block first before we do the move?
	
	if( uiHeapSize < uiOAEntrySize)
	{
		f_assert( uiHeapSize != uiAvailSpace);
		if( RC_BAD( rc = defragmentBlock( &pNextBlock, &pucNextBlock)))
		{
			goto Exit;
		}
	}

	// We are going to get some benefit from moving, so let's do it...
	
	if( RC_BAD( rc = moveToNext( uiStart, uiStart - uiCount + 1, 
		&pNextBlock, &pucNextBlock)))
	{
		goto Exit;
	}

	// If this block has a parent block, and the btree is maintaining counts
	// we will need to update the counts on the parent blocks.
	
	if( m_bCounts)
	{
		for( uiLevel = m_pStack->uiLevel;
			  uiLevel < m_uiStackLevels - 1;
			  uiLevel++)
		{
			pParentStack = &m_Stack[ uiLevel + 1];

			// If we are at "current" level, then we want to use the pNextBlock
			// block as the child.  Otherwise, we want to use the previous parent
			// block as the child.
			
			if( uiLevel == m_pStack->uiLevel)
			{
				pChildBlock = pNextBlock;
				pucChildBlock = pucNextBlock;
				
				pNextBlock = NULL;
				pucNextBlock = NULL;
			}
			else
			{
				if( pParentBlock)
				{
					pChildBlock = pParentBlock;
					pucChildBlock = pucParentBlock;
					pChildBlock->AddRef();
				}
			}

			// Check to see if the parent entry is the last entry in the
			// block.   If it is, then we will need to get the next block.
			// If the parent block is the same for both blocks, then we
			// only need to reference the next entry.  We don't want to release
			// the parent as it is referenced in the stack.
			
			if( bCommonParent || (pParentStack->uiCurOffset <
					(FLMUINT)(getNumKeys( pParentStack->pucBlock) - 1)))
			{
				if (pParentBlock)
				{
					pParentBlock->Release();
				}
				pParentBlock = pParentStack->pBlock;
				pucParentBlock = pParentStack->pucBlock;
				pParentBlock->AddRef();

				if (RC_BAD( rc = updateParentCounts( pucChildBlock, 
					&pParentBlock, &pucParentBlock,
					(bCommonParent
						? pParentStack->uiCurOffset
						: pParentStack->uiCurOffset + 1))))
				{
					goto Exit;
				}
				
				if( pParentStack->pBlock)
				{
					pParentStack->pBlock->Release();
					pParentStack->pucBlock = NULL;
				}
				
				pParentStack->pBlock = pParentBlock;
				pParentStack->pBlock->AddRef();
				pParentStack->pucBlock = pucParentBlock;

				bCommonParent = TRUE;
			}
			else
			{
				// We need to get the next block at the parent level first.  We
				// release the previous parent if there was one.
				
				uiNextBlockAddr = getNextInChain( pParentStack->pucBlock);

				f_assert( uiNextBlockAddr);

				if( RC_BAD( rc = m_pBlockMgr->getBlock( 
					(FLMUINT32)uiNextBlockAddr, &pParentBlock, &pucParentBlock)))
				{
					goto Exit;
				}

				if( RC_BAD( rc = updateParentCounts( pucChildBlock,
					&pParentBlock, &pucParentBlock, 0)))
				{
					goto Exit;
				}
			}

			if( pChildBlock)
			{
				pChildBlock->Release();
				pChildBlock = NULL;
				pucChildBlock = NULL;
			}
		}
	}

	*pbEntriesWereMoved = TRUE;

Exit:

	if( pChildBlock)
	{
		pChildBlock->Release();
	}

	if( pParentBlock)
	{
		pParentBlock->Release();
	}

	if( pNextBlock)
	{
		pNextBlock->Release();
	}

	return( rc);
}

/***************************************************************************
Desc: This method will move entries beginning at uiStart, down to and
		including uiFinish from the current block (m_pStack) to pNextBlock.
		As a part of this operation, both the target block and the source
		block will be changed.
****************************************************************************/
RCODE F_BTree::moveToNext(
	FLMUINT				uiStart,
	FLMUINT				uiFinish,
	IF_Block **			ppNextBlock,
	FLMBYTE **			ppucNextBlock)
{
	RCODE							rc = NE_FLM_OK;
	FLMUINT16 *					pui16DstOffsetA = NULL;
	FLMBYTE *					pucSrcEntry;
	FLMBYTE *					pucDstEntry;
	FLMUINT						uiEntrySize;
	FLMINT						iIndex;
	FLMUINT						uiBytesToCopy;
	FLMUINT						uiNumKeysToAdd;
	FLMBOOL						bEntriesCombined;
	FLMBYTE *					pucOffsetArray;
	FLMBYTE *					pucBuffer = NULL;
	FLMBYTE *					pucTmpBlock = NULL;
	FLMUINT						uiBufferSize = 0;
	void *						pvPoolMark = m_pool.poolMark();
	
	uiBufferSize = m_uiBlockSize * 2;
	
	if( RC_BAD( rc = m_pool.poolAlloc( uiBufferSize, (void **)&pucBuffer)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pool.poolAlloc( m_uiBlockSize, (void **)&pucTmpBlock)))
	{
		goto Exit;
	}
	
	// Make sure we have logged the block we are changing.
	// Note that the source block will be logged in the removeRange method.
	
	if( RC_BAD( rc = m_pBlockMgr->prepareForUpdate( 
		ppNextBlock, ppucNextBlock)))
	{
		goto Exit;
	}
	
	// We will need to save off the current offset array.  We will do this
	// by copying it into our temporary block.
	
	uiBytesToCopy = getNumKeys( *ppucNextBlock) * 2;
	if( uiBytesToCopy > uiBufferSize)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
		goto Exit;
	}

	pui16DstOffsetA = BtOffsetArray( *ppucNextBlock, 0);
	pucOffsetArray = &pucBuffer[ uiBufferSize] - uiBytesToCopy;

	f_memcpy( pucOffsetArray, (FLMBYTE *)pui16DstOffsetA, uiBytesToCopy);

	// Point to the last entry in the block.
	
	pucDstEntry = getBlockEnd( *ppucNextBlock);

	// Beginning at the start, copy each entry over from the Src to the Dst
	// block.  Note that the uiStart parameter represents a higher position
	// in the block.  In otherwords, we are actually copying from the end or
	// highest position to a lower position in the block.  Therefore we want
	// to make sure the offset array is copied in the same way, otherwise it
	// would reverse the order of the entries.
	
	uiNumKeysToAdd = uiStart - uiFinish + 1;
	pui16DstOffsetA = (FLMUINT16 *)pucOffsetArray;

	for( iIndex = uiStart; iIndex >= (FLMINT)uiFinish; iIndex--)
	{
		if( RC_BAD( rc = combineEntries( m_pStack->pucBlock,
			iIndex, *ppucNextBlock, 0, &bEntriesCombined,
			&uiEntrySize, pucTmpBlock)))
		{
			goto Exit;
		}

		if( bEntriesCombined)
		{
			F_BTSK		tmpStack;
			F_BTSK *		pTmpStack;

			tmpStack.pBlock = *ppNextBlock;
			tmpStack.pucBlock = *ppucNextBlock;
			tmpStack.uiCurOffset = 0;

			pTmpStack = m_pStack;
			m_pStack = &tmpStack;

			rc = remove( FALSE);
			m_pStack = pTmpStack;
			
			if (RC_BAD( rc))
			{
				goto Exit;
			}

			if( getHeapSize( *ppucNextBlock) != getBytesAvail( *ppucNextBlock))
			{
				if( RC_BAD( rc = defragmentBlock( ppNextBlock, ppucNextBlock)))
				{
					goto Exit;
				}

				// Refresh the saved offset array.
				
				uiBytesToCopy -= 2;
				pucOffsetArray = &pucBuffer[ uiBufferSize] - uiBytesToCopy;

				f_memcpy( pucOffsetArray, 
					BtOffsetArray( *ppucNextBlock, 0), uiBytesToCopy);
			}

			pucDstEntry = getBlockEnd( *ppucNextBlock) - uiEntrySize;
			f_memcpy( pucDstEntry, pucTmpBlock, uiEntrySize);

			bteSetEntryOffset( pui16DstOffsetA, 0, pucDstEntry - *ppucNextBlock);

			incNumKeys( *ppucNextBlock);
			decBytesAvail( *ppucNextBlock, ((FLMUINT16)uiEntrySize + 2));
			decHeapSize( *ppucNextBlock, ((FLMUINT16)uiEntrySize + 2));

			bEntriesCombined = FALSE;
		}
		else
		{
			pucSrcEntry = BtEntry( m_pStack->pucBlock, iIndex);
			uiEntrySize = getEntrySize( m_pStack->pucBlock, iIndex);

			pucDstEntry -= actualEntrySize( uiEntrySize);

			f_memcpy( pucDstEntry, pucSrcEntry,
						 actualEntrySize( uiEntrySize));

			pui16DstOffsetA--;

			bteSetEntryOffset( pui16DstOffsetA, 0, pucDstEntry - *ppucNextBlock);

			incNumKeys( *ppucNextBlock);
			decBytesAvail( *ppucNextBlock, (FLMUINT16)uiEntrySize);
			decHeapSize( *ppucNextBlock, (FLMUINT16)uiEntrySize);
		}
	}

	// Now put the new offset array into the block.
	
	f_memcpy( BtOffsetArray( *ppucNextBlock, 0), pui16DstOffsetA,
				 &pucBuffer[ uiBufferSize] - (FLMBYTE *)pui16DstOffsetA);

	// Now remove the entries from the Src block.
	
	if( RC_BAD( rc = removeRange( uiFinish, uiStart, FALSE)))
	{
		goto Exit;
	}

Exit:

	m_pool.poolReset( pvPoolMark);
	return( rc);
}

/***************************************************************************
Desc:	Method to advance to the next entry.  If there are no more entries
		in the block, it will release the current block and get the next in
		the chain. If there are no more entries, i.e. no more blocks in 
		the chain, NE_FLM_EOF_HIT will be returned.
****************************************************************************/
RCODE F_BTree::advanceToNextElement(
	FLMBOOL				bAdvanceStack)
{
	RCODE					rc = NE_FLM_OK;

	f_assert( m_pBlock && m_pucBlock);

	if( m_uiCurOffset + 1 >= getNumKeys( m_pucBlock))
	{
		// We are out of entries in this block, so we will release it
		// and get the next block in the chain (if any).
		
		if( RC_BAD( rc = getNextBlock( &m_pBlock, &m_pucBlock)))
		{
			goto Exit;
		}

		m_ui32PrimaryBlockAddr = getBlockAddr( m_pucBlock);
		m_uiPrimaryOffset = 0;
		m_ui32CurBlockAddr = m_ui32PrimaryBlockAddr;
		m_uiCurOffset = 0;

		if( bAdvanceStack)
		{
			if( RC_BAD( rc = moveStackToNext( m_pBlock, m_pucBlock)))
			{
				goto Exit;
			}
		}
	}
	else
	{
		m_uiPrimaryOffset++;
		m_uiCurOffset++;
		m_pStack->uiCurOffset++;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method to backup the stack to the previous entry.  If there are no
		more entries in the block, it will release the current block and get
		the previous in the chain. If there are no more entries, i.e. no
		more blocks in the chain, NE_FLM_BOF_HIT will be returned.
****************************************************************************/
RCODE F_BTree::backupToPrevElement(
	FLMBOOL				bBackupStack)
{
	RCODE					rc = NE_FLM_OK;

	f_assert( m_pBlock && m_pucBlock);

	if( !m_uiCurOffset)
	{
		// We are out of entries in this block, so we will release it
		// and get the previous block in the chain (if any).
		
		if( RC_BAD( rc = getPrevBlock( &m_pBlock, &m_pucBlock)))
		{
			goto Exit;
		}
		
		m_ui32PrimaryBlockAddr = getBlockAddr( m_pucBlock);
		m_uiPrimaryOffset = getNumKeys( m_pucBlock) - 1;
		m_ui32CurBlockAddr = m_ui32PrimaryBlockAddr;
		m_uiCurOffset = m_uiPrimaryOffset;

		if( bBackupStack)
		{
			if( RC_BAD( rc = moveStackToPrev( m_pBlock, m_pucBlock)))
			{
				goto Exit;
			}
		}
	}
	else
	{
		m_uiPrimaryOffset--;
		m_uiCurOffset--;
		m_pStack->uiCurOffset--;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method to extract the key length from a given entry.  The optional
		pucKeyRV is a buffer where we can return the address of the start of
		the actual key.
****************************************************************************/
FLMUINT F_BTree::getEntryKeyLength(
	FLMBYTE *			pucEntry,
	FLMUINT				uiBlockType,
	const FLMBYTE **	ppucKeyRV)
{
	FLMUINT				uiKeyLength;
	FLMBYTE *			pucTmp;

	// The way we get the key length depends on the type of block we have.

	switch( uiBlockType)
	{
		case F_BLK_TYPE_BT_LEAF_DATA:
		{
			pucTmp = &pucEntry[ 1];  // skip past the flags
			
			if( bteKeyLenFlag( pucEntry))
			{
				uiKeyLength = FB2UW( pucTmp);
				pucTmp += 2;
			}
			else
			{
				uiKeyLength = *pucTmp;
				pucTmp += 1;
			}

			if( bteDataLenFlag(pucEntry))
			{
				pucTmp += 2;
			}
			else
			{
				pucTmp += 1;
			}

			// Check for the presence of the OverallDataLength field (4 bytes).
			
			if( bteOADataLenFlag( pucEntry))
			{
				pucTmp += 4;
			}

			break;
		}
		
		case F_BLK_TYPE_BT_LEAF:
		{
			uiKeyLength = FB2UW( pucEntry);

			if( ppucKeyRV)
			{
				pucTmp = &pucEntry[ BTE_KEY_START];
			}

			break;
		}
		
		case F_BLK_TYPE_BT_NON_LEAF:
		{
			uiKeyLength = FB2UW( &pucEntry[ BTE_NL_KEY_LEN]);

			if( ppucKeyRV)
			{
				pucTmp = &pucEntry[ BTE_NL_KEY_START];
			}

			break;
		}
		
		case F_BLK_TYPE_BT_NON_LEAF_COUNTS:
		{
			uiKeyLength = FB2UW( &pucEntry[ BTE_NLC_KEY_LEN]);

			if( ppucKeyRV)
			{
				pucTmp = &pucEntry[ BTE_NLC_KEY_START];
			}

			break;
		}
		
		default:
		{
			f_assert( 0);
			uiKeyLength = 0;
			pucTmp = NULL;
			break;
		}
	}

	// Do we need to return the key pointer?
	
	if( ppucKeyRV)
	{
		*ppucKeyRV = pucTmp;
	}

	return( uiKeyLength);
}

/***************************************************************************
Desc:	Method to extract the data length from a given entry. The parameter
		pucDataRV is an optional return value that will hold the address
		of the beginning of the data in the entry.  This method 
		** assumes ** the entry is from a F_BLK_TYPE_BT_LEAF_DATA block.  No other block
		type has any data.
****************************************************************************/
FSTATIC FLMUINT fbtGetEntryDataLength(
	FLMBYTE *			pucEntry,
	const FLMBYTE **	ppucDataRV,				// Optional
	FLMUINT *			puiOADataLengthRV,	// Optional
	FLMBOOL *			pbDOBlockRV)			// Optional
{
	const FLMBYTE *	pucTmp;
	FLMUINT				uiDataLength;
	FLMUINT				uiKeyLength;

	pucTmp = &pucEntry[ 1];  // skip past the flags
	
	if( bteKeyLenFlag( pucEntry))
	{
		uiKeyLength = FB2UW( pucTmp);
		pucTmp += 2; 
	}
	else
	{
		uiKeyLength = *pucTmp;
		pucTmp += 1;
	}

	if( bteDataLenFlag( pucEntry))
	{
		uiDataLength = FB2UW( pucTmp);
		pucTmp += 2;
	}
	else
	{
		uiDataLength = *pucTmp;
		pucTmp += 1;
	}

	// Check for the presence of the OverallDataLength field (4 bytes).
	
	if( bteOADataLenFlag(pucEntry))
	{
		if( puiOADataLengthRV)
		{
			*puiOADataLengthRV = FB2UD( pucTmp);
		}
		pucTmp += 4;
	}
	else if (puiOADataLengthRV)
	{
		*puiOADataLengthRV = uiDataLength;
	}

	// Are we to return a pointer to the data?
	
	if( ppucDataRV)
	{
		// Advance to the Data since we are currently pointing to the Key.
		
		*ppucDataRV = (FLMBYTE *)(pucTmp + uiKeyLength);
	}

	if( pbDOBlockRV)
	{
		*pbDOBlockRV = bteDataBlockFlag( pucEntry);
	}

	return( uiDataLength);
}

/***************************************************************************
Desc:	Method to extract the data value from a given block. This method
		expects to receive a buffer to copy the data into.  This method does
		not read data across blocks.  The puiLenDataRV is an optional 
		parameter that will hold the actual data size returned.
****************************************************************************/
FSTATIC RCODE fbtGetEntryData(
	FLMBYTE *		pucEntry,	// Pointer to the entry containing the data
	FLMBYTE *		pucBufferRV,
	FLMUINT			uiBufferSize,
	FLMUINT *		puiLenDataRV)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT				uiDataLength;
	const FLMBYTE *	pucData;

	// Get the data length
	
	uiDataLength = fbtGetEntryDataLength( pucEntry, &pucData, NULL, NULL);

	if( uiDataLength > uiBufferSize)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}

#ifdef FLM_DEBUG
	f_memset( pucBufferRV, 0, uiBufferSize);
#endif
	f_memcpy( pucBufferRV, pucData, uiDataLength);

	// Do we need to return the data length?

	if( puiLenDataRV)
	{
		*puiLenDataRV = uiDataLength;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	This method will return the overall size of the entry at uiOffset in
		pBlock.  The size returned includes a two byte allowance for the offset
		entry used by this entry.
****************************************************************************/
FLMUINT F_BTree::getEntrySize(
	FLMBYTE *		pucBlock,
	FLMUINT			uiOffset,
	FLMBYTE **		ppucEntry)
{
	FLMBYTE *		pucEntry;
	FLMUINT			uiEntrySize;

	// Point to the entry ...
	
	pucEntry = BtEntry( pucBlock, uiOffset);

	if( ppucEntry)
	{
		*ppucEntry = pucEntry;
	}

	// Different block types have different entry formats.
	
	switch( getBlockType( pucBlock))
	{
		case F_BLK_TYPE_BT_LEAF:
		{
			uiEntrySize =  4 + FB2UW( pucEntry);
			break;
		}
		case F_BLK_TYPE_BT_LEAF_DATA:
		{
			FLMBYTE * pucTmp = &pucEntry[ 1];

			// Stuff we know
			
			uiEntrySize = 3;

			// Get the key length
			
			if( *pucEntry & BTE_FLAG_KEY_LEN)
			{
				uiEntrySize += FB2UW( pucTmp) + 2;
				pucTmp += 2;
			}
			else
			{
				uiEntrySize += (*pucTmp + 1);
				pucTmp++;
			}

			// Get the data length
			
			if( *pucEntry & BTE_FLAG_DATA_LEN)
			{
				// 2 byte data length field
				
				uiEntrySize += (FB2UW( pucTmp) + 2);
			}
			else
			{
				// 1 byte data length field
				
				uiEntrySize += (FLMUINT)*pucTmp + 1;
			}

			// Get the Overall Data length (if present)
			
			if( *pucEntry & BTE_FLAG_OA_DATA_LEN)
			{
				uiEntrySize += 4;
			}
			
			break;
		}
		
		case F_BLK_TYPE_BT_NON_LEAF:
		{
			uiEntrySize = 8 + FB2UW( &pucEntry[ BTE_NL_KEY_LEN]);
			break;
		}
		
		case F_BLK_TYPE_BT_NON_LEAF_COUNTS:
		{
			uiEntrySize = 12 + FB2UW( &pucEntry[ BTE_NLC_KEY_LEN]);
			break;
		}
		
		default:
		{
			f_assert( 0);
			uiEntrySize = 0;
			break;
		}
	}

	return( uiEntrySize);
}

/***************************************************************************
Desc:	Method to search the BTree for a specific entry. Upon a successful
		return from this method, the local stack will be setup and pointing
		to either the desired entry, or if the entry does not exist, it will
		be pointing to the entry that would be immediately after the desired
		entry.  This method therefore can be used both for reads and updates
		where we want to insert a new entry into the BTree.
****************************************************************************/
RCODE F_BTree::findEntry(
	const FLMBYTE *	pucKey,				// In
	FLMUINT 				uiKeyLen,			// In
	FLMUINT				uiMatch,				// In
	FLMUINT *			puiPosition,		// Out
	FLMUINT32 *			pui32BlockAddr,			// In/Out
	FLMUINT *			puiOffsetIndex)	// In/Out
{
	RCODE					rc = NE_FLM_OK;
	F_BTSK *				pStack = NULL;
	FLMUINT32			ui32BlockAddr;
	IF_Block *			pBlock = NULL;
	FLMBYTE *			pucBlock = NULL;
	FLMBYTE *			pucEntry;
	FLMUINT				uiPrevCounts = 0;
	FLMUINT				uiLevel;

	// Make sure the stack is clean before we start.
	
	btRelease();

	// No input key is needed to get the first or last key.
	
	if( uiMatch == FLM_FIRST || uiMatch == FLM_LAST)
	{
		uiKeyLen = 0;
	}

	if( uiKeyLen > FLM_MAX_KEY_SIZE)
	{
		rc = RC_SET( NE_FLM_BTREE_KEY_SIZE);
		goto Exit;
	}

	// Have we been passed a block address to look in?
	
	if( pui32BlockAddr && *pui32BlockAddr)
	{
		if( RC_OK( rc = findInBlock( pucKey, uiKeyLen, uiMatch, puiPosition,
			pui32BlockAddr, puiOffsetIndex)))
		{
			goto Exit;
		}
	}

	// Beginning at the root node, we will scan until we find the first key
	// that is greater than or equal to our target key.  If we don't find any
	// key that is larger than our target key, we will use the last block found.
	
	ui32BlockAddr = m_ui32RootBlockAddr;

	for( ;;)
	{
		// Get the block - Note that this will place a use on the block.
		// It must be properly released when done.

		if( RC_BAD( rc = m_pBlockMgr->getBlock( ui32BlockAddr, 
			&pBlock, &pucBlock)))
		{
			goto Exit;
		}

		f_assert( pucBlock);

		// We are building the stack inverted to make traversing it a bit easier.
		
		uiLevel = getBlockLevel( pucBlock);
		pStack = &m_Stack[ uiLevel];

		m_uiStackLevels++;

		pStack->ui32BlockAddr = ui32BlockAddr;
		pStack->pBlock = pBlock;
		pStack->pucBlock = pucBlock;
		
		pBlock = NULL;
		pucBlock = NULL;
		
		pStack->uiLevel = uiLevel;
		pStack->uiKeyLen = uiKeyLen;
		pStack->pucKeyBuf = pucKey;
		pStack->uiKeyBufSize = m_Stack[0].uiKeyBufSize;
		pStack->pui16OffsetArray = BtOffsetArray( pStack->pucBlock, 0);

		if( isRootBlock( pStack->pucBlock))
		{
			m_uiRootLevel = uiLevel;
		}

		// Search the block for the key.  When we return from this method
		// the pStack will be pointing to the last entry we looked at.
		
		if( RC_BAD( rc = scanBlock( pStack, uiMatch)))
		{
			// It is okay if we couldn't find the key.  Especially if
			// we are still in the upper levels of the B-tree.
			
			if( (rc != NE_FLM_NOT_FOUND) && (rc != NE_FLM_EOF_HIT))
			{
				goto Exit;
			}
		}

		// Are we at the end of our search?
		
		if( getBlockType( pStack->pucBlock) == F_BLK_TYPE_BT_LEAF_DATA ||
			 (getBlockType( pStack->pucBlock) == F_BLK_TYPE_BT_LEAF) ||
			 (m_uiStackLevels - 1 >= m_uiSearchLevel))
		{
			if( m_bCounts && puiPosition)
			{
				f_assert( m_uiSearchLevel >= F_BTREE_MAX_LEVELS);
				*puiPosition = uiPrevCounts + pStack->uiCurOffset;
			}

			// If this is a search for the last entry, then we should adjust the
			// uiCurOffset so that it points to a valid entry.
			
			if( uiMatch == FLM_LAST)
			{
				m_pStack = pStack;

				for (;;)
				{
					if( RC_BAD( rc = moveStackToPrev( NULL, NULL)))
					{
						goto Exit;
					}
					
					// If we are on the leaf level, we need to make sure we are
					// looking at a first occurrence of an entry.
					
					if( getBlockType( pStack->pucBlock) == F_BLK_TYPE_BT_LEAF_DATA)
					{
						pucEntry = BtEntry( m_pStack->pucBlock, m_pStack->uiCurOffset);

						if( bteFirstElementFlag( pucEntry))
						{
							break;
						}
					}
					else
					{
						break;
					}
				}
			}

			break;
		}
		else
		{
			if( m_bCounts && puiPosition)
			{
				uiPrevCounts += countRangeOfKeys( pStack, 0, pStack->uiCurOffset);
			}

			// Get the Child Block Address
			
			pucEntry = BtEntry( pStack->pucBlock, pStack->uiCurOffset);
			ui32BlockAddr = bteGetBlockAddr( pucEntry);
		}
	}

	// Return the block and offset if needed.
	
	if( pui32BlockAddr)
	{
		*pui32BlockAddr = pStack->ui32BlockAddr;
	}

	if( puiOffsetIndex)
	{
		*puiOffsetIndex = pStack->uiCurOffset;
	}

	m_bStackSetup = TRUE;

Exit:

	if( RC_OK( rc) || (rc == NE_FLM_NOT_FOUND) || (rc == NE_FLM_EOF_HIT))
	{
		if( pStack)
		{
			m_pStack = pStack;
		}
	}

	if( pBlock)
	{
		pBlock->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:	Private method to search for a particular key in a pre-designted
		block offset.  If we don't find it at the given offset, we will do a
		binary search for it.  Note that a uiMatch of FLM_FIRST & FLM_LAST
		will be ignored if we locate the entry by the puiOffsetIndex parameter.  
		Also, this method does not setup the full stack.  Only the level where
		the block address passed in resides.
****************************************************************************/
RCODE F_BTree::findInBlock(
	const FLMBYTE *	pucKey,
	FLMUINT				uiKeyLen,
	FLMUINT				uiMatch,
	FLMUINT *			puiPosition,
	FLMUINT32 *			pui32BlockAddr,
	FLMUINT *			puiOffsetIndex)
{
	RCODE					rc = NE_FLM_OK;
	F_BTSK *				pStack;
	IF_Block *			pBlock = NULL;
	FLMBYTE *			pucBlock = NULL;
	FLMBYTE *			pucEntry;
	const FLMBYTE *	pucBlockKey;
	FLMUINT				uiBlockKeyLen;

	// Get the block - Note that this will place a use on the block.
	// It must be properly released when done.

	if( RC_BAD( rc = m_pBlockMgr->getBlock( *pui32BlockAddr, &pBlock, &pucBlock)))
	{
		goto Exit;
	}
	
	if( !blkIsBTree( pucBlock))
	{
		rc = RC_SET( NE_FLM_NOT_FOUND);
		goto Exit;
	}
	
	// If the block is not a leaf block, the caller will
	// need to do a full search down the B-Tree

	if( getBlockLevel( pucBlock) != 0)
	{
		rc = RC_SET( NE_FLM_NOT_FOUND);
		goto Exit;
	}

	pStack = &m_Stack[ 0];
	m_uiStackLevels++;

	pStack->ui32BlockAddr = *pui32BlockAddr;
	pStack->pBlock = pBlock;
	pStack->pucBlock = pucBlock;
	
	pBlock = NULL;
	pucBlock = NULL;
	
	pStack->uiLevel = 0;
	pStack->uiKeyLen = uiKeyLen;
	pStack->pucKeyBuf = pucKey;
	pStack->uiKeyBufSize = m_Stack[0].uiKeyBufSize;
	pStack->pui16OffsetArray = BtOffsetArray( pStack->pucBlock, 0);
	pStack->uiCurOffset = puiOffsetIndex ? *puiOffsetIndex : 0;

	if( isRootBlock( pStack->pucBlock))
	{
		m_uiRootLevel = 0;
	}

	// See if the entry we are looking for is at the passed offset
	
	if( puiOffsetIndex)
	{
		if( *puiOffsetIndex < getNumKeys( pStack->pucBlock))
		{
			pucEntry = BtEntry( pStack->pucBlock, *puiOffsetIndex);

			uiBlockKeyLen = getEntryKeyLength( pucEntry,
								getBlockType( pStack->pucBlock), &pucBlockKey);

			if( uiKeyLen == uiBlockKeyLen)
			{
				if( f_memcmp( pucKey, pucBlockKey, uiKeyLen) == 0)
				{
					goto GotEntry;
				}
			}
		}
	}

	// Search the block for the key.  When we return from this method
	// the pStack will be pointing to the last entry we looked at.

	if( RC_BAD( rc = scanBlock( pStack, uiMatch)))
	{
		goto Exit;
	}

GotEntry:

	if( m_bCounts && puiPosition)
	{
		f_assert( m_uiSearchLevel >= F_BTREE_MAX_LEVELS);
		*puiPosition = pStack->uiCurOffset;
	}

	// Verify that we are looking at an entry with the firstElement flag set.
	
	m_pStack = pStack;

	for (;;)
	{
		// If we are on the leaf level, we need to make sure we are
		// looking at a first occurrence of an entry.
		
		if( getBlockType( m_pStack->pucBlock) == F_BLK_TYPE_BT_LEAF_DATA)
		{
			pucEntry = BtEntry( m_pStack->pucBlock, m_pStack->uiCurOffset);

			if( bteFirstElementFlag( pucEntry))
			{
				break;
			}
		}
		else
		{
			break;
		}

		if( RC_BAD( rc = moveStackToPrev( NULL, NULL)))
		{
			goto Exit;
		}
	}

	*pui32BlockAddr = m_pStack->ui32BlockAddr;

	if( puiOffsetIndex)
	{
		*puiOffsetIndex = m_pStack->uiCurOffset;
	}

Exit:

	if( pBlock)
	{
		pBlock->Release();
	}

	if( RC_BAD( rc))
	{
		btRelease();
	}

	return( rc);
}

/***************************************************************************
Desc:	Method to search through a BTree block to find a specific key.  If 
		that key cannot be found, then the pStack will be positioned right 
		after the last entry in the block.  The search is a binary search that
		is looking for the first key that is >= the target key.  The uiMatch
		parameter further qualifies the search.  The FLM_FIRST & FLM_LAST
		values will ignore the key altogether and just return the first or last
		key respectively.  The FLM_INCL value will return the key if found or the 
		first key following if not found.  The FLM_EXACT will return an 
		NE_FLM_NOT_FOUND if the key cannot be found.  FLM_EXCL will return
		the first key following the target key.
****************************************************************************/
RCODE F_BTree::scanBlock(
	F_BTSK *			pStack,
	FLMUINT			uiMatch)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT				uiTop;
	FLMUINT				uiMid;
	FLMUINT				uiBottom;
	FLMINT				iResult;
	IF_Block *			pBlock = NULL;
	const FLMBYTE *	pucBlockKey;
	FLMBYTE *			pucEntry;
	FLMUINT				uiBlockKeyLen;

	if( getNumKeys( pStack->pucBlock) == 0)
	{
		rc = RC_SET( NE_FLM_BOF_HIT);
		goto Exit;
	}

	uiTop = 0;
	uiBottom = (FLMUINT)(getNumKeys( pStack->pucBlock) - 1);

	if( uiMatch == FLM_FIRST)
	{
		pStack->uiCurOffset = uiTop;
		goto Exit;
	}

	if( uiMatch == FLM_LAST || pStack->uiKeyLen == 0)
	{
		pStack->uiCurOffset = uiBottom;
		goto Exit;
	}

	f_assert( uiMatch == FLM_INCL || uiMatch == FLM_EXCL || 
				  uiMatch == FLM_EXACT);

	// Test the first entry
	
	pucEntry = (FLMBYTE *)pStack->pucBlock +
										bteGetEntryOffset( pStack->pui16OffsetArray,
										uiTop);
											 
	uiBlockKeyLen = getEntryKeyLength( pucEntry, getBlockType( pStack->pucBlock),
										 			&pucBlockKey);

	// Compare the entries ...
	
	if( !uiBlockKeyLen)
	{
		// The LEM entry will always sort last!!

		iResult = 1;
		goto ResultGreater1;
	}
	else
	{
		if( RC_BAD( rc = compareBlockKeys( pucBlockKey, uiBlockKeyLen,
				pStack->pucKeyBuf, pStack->uiKeyLen, &iResult)))
		{
			goto Exit;
		}
	}
	
	if( iResult >= 0)
	{
ResultGreater1:

		if( iResult && uiMatch == FLM_EXACT)
		{
			rc = RC_SET( NE_FLM_NOT_FOUND);
		}
		
		uiMid = uiTop;
		goto VerifyPosition;
	}

	// If there is more than one entry in the block, we can skip the first
	// one since we have already seen it.
	
	if( uiTop < uiBottom)
	{
		uiTop++;
	}

	// Test the last
	
	pucEntry = (FLMBYTE *)pStack->pucBlock +
					bteGetEntryOffset( pStack->pui16OffsetArray,
											 uiBottom);
											 
	uiBlockKeyLen = getEntryKeyLength( pucEntry, getBlockType( pStack->pucBlock),
													&pucBlockKey);

	if( !uiBlockKeyLen)
	{
		// The LEM entry will always sort last!!

		iResult = 1;
		goto ResultGreater2;
	}
	else
	{
		if( RC_BAD( rc = compareBlockKeys( pucBlockKey, uiBlockKeyLen,
				pStack->pucKeyBuf, pStack->uiKeyLen, &iResult)))
		{
			goto Exit;
		}
	}
	
	if( iResult <= 0)
	{
		if( iResult < 0 && uiMatch != FLM_INCL)
		{
			rc = RC_SET( NE_FLM_NOT_FOUND);
		}
		
		uiMid = uiBottom;
		goto VerifyPosition;
	}

ResultGreater2:

	for( ;;)
	{

		if( uiTop == uiBottom)
		{
			// We're done - didn't find it.
			
			if( uiMatch == FLM_EXACT)
			{
				rc = RC_SET( NE_FLM_NOT_FOUND);
			}
			
			uiMid = uiTop;
			break;
		}

		// Get the midpoint
		
		uiMid = (uiTop + uiBottom) / 2;

		pucEntry = (FLMBYTE *)pStack->pucBlock +
						bteGetEntryOffset( pStack->pui16OffsetArray,
												 uiMid);
												 
		uiBlockKeyLen = getEntryKeyLength( pucEntry, 
					getBlockType( pStack->pucBlock), &pucBlockKey);

		// Compare the entries

		if( !uiBlockKeyLen)
		{
			// The LEM entry will always sort last!!

			iResult = 1;
			goto ResultGreater;
		}
		else
		{
			if( RC_BAD( rc = compareBlockKeys( pucBlockKey, uiBlockKeyLen,
				pStack->pucKeyBuf, pStack->uiKeyLen, &iResult)))
			{
				goto Exit;
			}
		}

		if( iResult > 0)
		{
ResultGreater:

			// Midpoint (block key) is > Target key
			
			uiBottom = uiMid;
			continue;
		}

		if( iResult < 0)
		{
			// Midpoint (block key) is < Target key
			// Since we want to find the first key that is >= to the target key,
			// and we have aleady visited the key at uiMid and know that it is <
			// our target key, we can skip it and advance to the key that is one
			// beyond it.
			
			f_assert( uiMid < uiBottom);
			uiTop = uiMid + 1;
			continue;
		}
		
		break;
	}

VerifyPosition:

	if( uiMatch != FLM_EXCL)
	{
		// Verify that we are looking at the first occurrence of this key.
		
		while( iResult == 0)
		{
			if( uiMid > 0)
			{
				pucEntry = (FLMBYTE *)pStack->pucBlock +
								bteGetEntryOffset( pStack->pui16OffsetArray,
														 (uiMid - 1));

				uiBlockKeyLen = getEntryKeyLength( 
										pucEntry, getBlockType( pStack->pucBlock),
										&pucBlockKey);

				if( !uiBlockKeyLen)
				{
					// The LEM entry will always sort last!!

					iResult = 1;
				}
				else
				{
					if( RC_BAD( rc = compareBlockKeys( pucBlockKey, uiBlockKeyLen,
						pStack->pucKeyBuf, pStack->uiKeyLen, &iResult)))
					{
						goto Exit;
					}

					if( iResult == 0)
					{
						uiMid--;
					}
				}
			}
			else
			{
				break;
			}
		}
		
		pStack->uiCurOffset = uiMid;
	}
	else if( uiMatch == FLM_EXCL)
	{
		// If we are at the leaf level, then we want to see if
		// this is the last entry in the last block.
		// If it is, then we cannot satisfy the request, otherwise
		// we will position to the next key and return ok.
		
		if( getBlockLevel( pStack->pucBlock) == 0 &&
			 getNextInChain( pStack->pucBlock) == 0 &&
			 uiMid == (FLMUINT)getNumKeys( pStack->pucBlock) - 1 &&
			 iResult == 0)
		{
			rc = RC_SET( NE_FLM_EOF_HIT);
		}
		else if( getBlockLevel( pStack->pucBlock) == 0)
		{
			// Check for the next entry at leaf level			
			
			while( iResult == 0)
			{
				// Are we on the last key?
				
				if( uiMid == (FLMUINT)(getNumKeys( pStack->pucBlock) - 1))
				{
					if( getNextInChain( pStack->pucBlock) == 0)
					{
						rc = RC_SET( NE_FLM_NOT_FOUND);
					}
					else
					{
						pStack->uiCurOffset = uiMid;
						m_pStack = pStack;

						if( RC_BAD( rc = moveStackToNext( NULL, NULL)))
						{
							goto Exit;
						}
						
						uiMid = 0;
					}
				}
				else
				{
					uiMid++;
				}

				pucEntry = (FLMBYTE *)pStack->pucBlock +
								bteGetEntryOffset( pStack->pui16OffsetArray,
														 uiMid);

				uiBlockKeyLen = getEntryKeyLength( pucEntry,
											getBlockType( pStack->pucBlock), &pucBlockKey);

				if( !uiBlockKeyLen)
				{
					// The LEM entry will always sort last!!

					iResult = 1;
				}
				else
				{
					if( RC_BAD( rc = compareBlockKeys( pucBlockKey, uiBlockKeyLen,
						pStack->pucKeyBuf, pStack->uiKeyLen, &iResult)))
					{
						goto Exit;
					}
				}
			}
			
			pStack->uiCurOffset = uiMid;
			if( uiMid == (FLMUINT)(getNumKeys( pStack->pucBlock) - 1) &&
							getNextInChain( pStack->pucBlock) == 0)
			{
				rc = RC_SET( NE_FLM_EOF_HIT);
			}
		}
		else
		{
			pStack->uiCurOffset = uiMid;
		}
	}

Exit:

	if( pBlock)
	{
		pBlock->Release();
	}

	return( rc);
}

/***************************************************************************
Desc: This method will compare two key fields.
		Returned values:  0 - Keys are equal
								1 - Key in Block is > Target key
							  -1 - Key in Block is < Target key
****************************************************************************/
RCODE F_BTree::compareKeys(
	const FLMBYTE *	pucKey1,
	FLMUINT				uiKeyLen1,
	const FLMBYTE *	pucKey2,
	FLMUINT				uiKeyLen2,
	FLMINT *				piCompare)
{
	RCODE		rc = NE_FLM_OK;
	
	if( !m_pCompare)
	{
		if( (*piCompare = f_memcmp( pucKey1, pucKey2, 
				f_min( uiKeyLen1, uiKeyLen2))) == 0)
		{
			*piCompare = uiKeyLen1 == uiKeyLen2
								? 0
								: uiKeyLen1 < uiKeyLen2 
									? -1
									: 1;
		}
	}
	else
	{
		if( RC_BAD( rc = m_pCompare->compare( pucKey1, uiKeyLen1,
										pucKey2, uiKeyLen2, piCompare)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method for positioning to a specific entry.
****************************************************************************/
RCODE F_BTree::positionToEntry(
	FLMUINT			uiPosition)
{
	RCODE				rc = NE_FLM_OK;
	F_BTSK *			pStack = NULL;
	FLMUINT32		ui32BlockAddr;
	IF_Block *		pBlock = NULL;
	FLMBYTE *		pucBlock = NULL;
	FLMUINT			uiLevel;
	FLMBYTE *		pucEntry;
	FLMUINT			uiPrevCounts = 0;

	// Make sure the stack is clean before we start.
	
	btRelease();

	// Beginning at the root node.
	
	ui32BlockAddr = m_ui32RootBlockAddr;

	// Get the block - Note that this will place a use on the block.
	// It must be properly released when done.
	
	while( ui32BlockAddr)
	{
		if( RC_BAD( rc = m_pBlockMgr->getBlock( ui32BlockAddr, 
			&pBlock, &pucBlock)))
		{
			goto Exit;
		}

		uiLevel = getBlockLevel( pucBlock);
		pStack = &m_Stack[ uiLevel];

		pStack->ui32BlockAddr = ui32BlockAddr;
		pStack->pBlock = pBlock;
		pStack->pucBlock = pucBlock;
		
		pBlock = NULL;
		pucBlock = NULL;
		
		pStack->uiLevel = uiLevel;
		pStack->pui16OffsetArray = BtOffsetArray( pStack->pucBlock, 0);

		m_uiStackLevels++;

		if( RC_BAD( rc = searchBlock( pStack->pucBlock, &uiPrevCounts,
			uiPosition, &pStack->uiCurOffset)))
		{
			goto Exit;
		}

		if( getBlockType( pStack->pucBlock) == F_BLK_TYPE_BT_LEAF_DATA ||
			 getBlockType( pStack->pucBlock) == F_BLK_TYPE_BT_LEAF)
		{
			ui32BlockAddr = 0;
		}
		else
		{
			// Get the next child block address
			
			pucEntry = BtEntry( pStack->pucBlock, pStack->uiCurOffset);
			ui32BlockAddr = bteGetBlockAddr( pucEntry);
		}
	}

	m_uiRootLevel = m_uiStackLevels - 1;

Exit:

	if( RC_OK( rc) || (rc == NE_FLM_NOT_FOUND) || (rc == NE_FLM_EOF_HIT))
	{
		m_pStack = pStack;
	}

	if( pBlock)
	{
		pBlock->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE F_BTree::searchBlock(
	FLMBYTE *		pucBlock,
	FLMUINT *		puiPrevCounts,
	FLMUINT			uiPosition,
	FLMUINT *		puiOffset)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiOffset;
	FLMUINT			uiNumKeys;
	FLMUINT			uiCounts;
	FLMBYTE *		pucEntry;

	uiNumKeys = getNumKeys( pucBlock);

	if( getBlockType( pucBlock) != F_BLK_TYPE_BT_NON_LEAF_COUNTS)
	{
		f_assert( uiPosition >= *puiPrevCounts);
		
		uiOffset = uiPosition - *puiPrevCounts;
		*puiPrevCounts = uiPosition;
	}
	else
	{
		for( uiOffset = 0; uiOffset < uiNumKeys; uiOffset++)
		{
			pucEntry = BtEntry( pucBlock, uiOffset);
			pucEntry += 4;
			
			uiCounts = FB2UD( pucEntry);

			if( *puiPrevCounts + uiCounts >= (uiPosition + 1))
			{
				break;
			}
			else
			{
				*puiPrevCounts += uiCounts;
			}
		}
	}

	if( uiOffset >= uiNumKeys)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
	}

	*puiOffset = uiOffset;
	return( rc);
}

/***************************************************************************
Desc:	Method to move all the data in the block into a contiguous space.
****************************************************************************/
RCODE F_BTree::defragmentBlock(
	IF_Block **				ppBlock,
	FLMBYTE **				ppucBlock)
{
	RCODE						rc = NE_FLM_OK;
	FLMUINT					uiNumKeys;
	FLMBOOL					bSorted;
	FLMBYTE *				pucCurEntry;
	FLMBYTE *				pucPrevEntry;
	FLMBYTE *				pucTempEntry;
	FLMUINT					uiTempToMove;
	FLMUINT					uiIndex;
	FLMUINT					uiAmtToMove;
	FLMUINT					uiFirstHole;
	FLMUINT16				ui16BlockBytesAvail;
	FLMUINT16 *				pui16OffsetArray;
	FLMBYTE *				pucHeap;
	FLMBYTE *				pucBlockEnd;
	IF_Block *				pOldBlock = NULL;
	FLMBYTE *				pucOldBlock = NULL;
	void *					pvPoolMark = m_pool.poolMark();

	f_assert( getBytesAvail( *ppucBlock) != getHeapSize( *ppucBlock));

	pOldBlock = *ppBlock;
	pucOldBlock = *ppucBlock;
	pOldBlock->AddRef();
	
	if( RC_BAD( rc = m_pBlockMgr->prepareForUpdate( ppBlock, ppucBlock)))
	{
		goto Exit;
	}

	uiNumKeys = getNumKeys( *ppucBlock);
	
	// Determine if the entries are sorted

	pucPrevEntry = *ppucBlock + m_uiBlockSize;
	bSorted = TRUE;
	uiFirstHole = 0;
	pucHeap = *ppucBlock + m_uiBlockSize;

	for( uiIndex = 0; uiIndex < uiNumKeys; uiIndex++)
	{
		pucCurEntry = BtEntry( *ppucBlock, uiIndex);

		if( pucPrevEntry < pucCurEntry)
		{
			bSorted = FALSE;
			break;
		}
		else
		{
			uiAmtToMove = actualEntrySize( getEntrySize( *ppucBlock, uiIndex));
			pucHeap -= uiAmtToMove;

			if( !uiFirstHole && pucHeap != pucCurEntry)
			{
				uiFirstHole = uiIndex + 1;
			}
		}

		pucPrevEntry = pucCurEntry;
	}
	
	ui16BlockBytesAvail = (FLMUINT16)(m_uiBlockSize - 
							  sizeofBTreeBlockHdr( *ppucBlock)) -
							  (FLMUINT16)(uiNumKeys * 2);
	pui16OffsetArray = BtOffsetArray( *ppucBlock, 0);
	pucBlockEnd = *ppucBlock + m_uiBlockSize;

	if( uiFirstHole > 1)
	{
		uiFirstHole--;
		pucHeap = BtEntry( *ppucBlock, uiFirstHole - 1);
		ui16BlockBytesAvail -= (FLMUINT16)(pucBlockEnd - pucHeap);
	}
	else
	{
		uiFirstHole = 0;
		pucHeap = pucBlockEnd;
	}

	if( !bSorted)
	{
		FLMBYTE *			pucTempDefragBlock;
		FLMUINT16 *			pui16OldOffsetArray;

		// If old and new blocks are the same (because of a 
		// prior call to logBlock), we need to save a copy of the block
		// before making changes.

		if( pOldBlock == *ppBlock)
		{
			if( RC_BAD( rc = m_pool.poolAlloc( m_uiBlockSize, 
				(void **)&pucTempDefragBlock)))
			{
				goto Exit;
			}
			
			f_memcpy( pucTempDefragBlock, *ppucBlock, m_uiBlockSize);
			pucOldBlock = pucTempDefragBlock;
		}

		pui16OldOffsetArray = BtOffsetArray( pucOldBlock, 0);

		// Rebuild the block so that all of the entries are in order

		for( uiIndex = uiFirstHole; uiIndex < uiNumKeys; uiIndex++)
		{
			pucCurEntry = BtEntry( pucOldBlock, uiIndex);
			uiAmtToMove = actualEntrySize( getEntrySize( pucOldBlock, uiIndex));
			pucHeap -= uiAmtToMove;
			bteSetEntryOffset( pui16OffsetArray, uiIndex, pucHeap - *ppucBlock);
			uiIndex++;

			while( uiIndex < uiNumKeys)
			{
				pucTempEntry = BtEntry( pucOldBlock, uiIndex);
				uiTempToMove = actualEntrySize( 
										getEntrySize( pucOldBlock, uiIndex));

				if ((pucCurEntry - uiTempToMove) != pucTempEntry)
				{
					uiIndex--;
					break;
				}
				else
				{
					pucCurEntry -= uiTempToMove;
					pucHeap -= uiTempToMove;
					uiAmtToMove += uiTempToMove;
					bteSetEntryOffset( pui16OffsetArray, uiIndex, pucHeap - *ppucBlock);
					uiIndex++;
				}
			}

			f_memcpy( pucHeap, pucCurEntry, uiAmtToMove);
			ui16BlockBytesAvail -= (FLMUINT16)uiAmtToMove;
		}
	}
	else
	{
		// Work back from the first hole.  Move entries to fill all of the
		// holes in the block.

		for( uiIndex = uiFirstHole; uiIndex < uiNumKeys; uiIndex++)
		{
			pucCurEntry = BtEntry( *ppucBlock, uiIndex);
			uiAmtToMove = actualEntrySize( getEntrySize( *ppucBlock, uiIndex));
			pucHeap -= uiAmtToMove;

			if( pucHeap != pucCurEntry)
			{
				// We have a hole.  We don't want to move just one entry
				// if we can avoid it.  We would like to continue searching
				// until we find either the end, or another hole.  Then we
				// can move a larger block of data instead of one entry.

				bteSetEntryOffset( pui16OffsetArray, uiIndex, pucHeap - *ppucBlock);
				uiIndex++;

				while( uiIndex < uiNumKeys)
				{
					pucTempEntry = BtEntry( *ppucBlock, uiIndex);
					uiTempToMove = actualEntrySize( getEntrySize( *ppucBlock, uiIndex));

					if( (pucCurEntry - uiTempToMove) != pucTempEntry)
					{
						uiIndex--;
						break;
					}
					else
					{
						pucCurEntry -= uiTempToMove;
						pucHeap -= uiTempToMove;
						uiAmtToMove += uiTempToMove;
						bteSetEntryOffset( pui16OffsetArray, uiIndex, pucHeap - *ppucBlock);
						uiIndex++;
					}
				}
			}

			// Now move the range we have determined.

			f_memmove( pucHeap, pucCurEntry, uiAmtToMove);
			ui16BlockBytesAvail -= (FLMUINT16)(uiAmtToMove);
		}
	}

	// Set the available space.  If there are no keys in this block, we should
	// set the it to the calculated available space

	if( !uiNumKeys)
	{
		setBytesAvail( *ppucBlock, ui16BlockBytesAvail);
	}

	f_assert( getBytesAvail( *ppucBlock) == ui16BlockBytesAvail);
	setHeapSize( *ppucBlock, ui16BlockBytesAvail);

	// Clean up the heap space.

#ifdef FLM_DEBUG
	f_memset( getBlockEnd( *ppucBlock) - ui16BlockBytesAvail, 0, ui16BlockBytesAvail);
#endif

Exit:

	if( pOldBlock)
	{
		pOldBlock->Release();
	}
	
	m_pool.poolReset( pvPoolMark);
	return( rc);
}

/***************************************************************************
Desc:	Method to handle the insertion, deletion and replacment of a single
		entry in a block.
		Assumption:  The find method has already been called to locate the
		insertion point, so the stack has already been setup.
****************************************************************************/
RCODE F_BTree::updateEntry(
	const FLMBYTE *		pucKey,		// In
	FLMUINT					uiKeyLen,	// In
	const FLMBYTE *		pucValue,	// In
	FLMUINT					uiLen,		// In
	F_ELM_UPD_ACTION		eAction,
	FLMBOOL					bTruncate)
{
	RCODE						rc = NE_FLM_OK;
	const FLMBYTE *		pucRemainingValue = NULL;
	FLMUINT					uiRemainingLen = 0;
	const FLMBYTE *		pucSavKey = pucKey;
	FLMUINT					uiSavKeyLen = uiKeyLen;
	FLMUINT					uiChildBlockAddr = 0;
	FLMUINT					uiCounts = 0;
	FLMUINT					uiFlags = BTE_FLAG_FIRST_ELEMENT | BTE_FLAG_LAST_ELEMENT;
	FLMBOOL					bMoreToRemove = FALSE;
	FLMBOOL					bDone = FALSE;
	FLMUINT					uiOrigDataLen = uiLen;
	FLMBOOL					bOrigTruncate = bTruncate;

	f_assert( m_pReplaceInfo == NULL);

	// For each level that needs modifying...
	
	while( !bDone)
	{

		switch( eAction)
		{
			case ELM_INSERT_DO:
			{
				// In this case, the uiLen parameter represents the OADataLength.
				
				uiFlags = BTE_FLAG_DATA_BLOCK |
							 BTE_FLAG_FIRST_ELEMENT |
							 BTE_FLAG_LAST_ELEMENT |
							 BTE_FLAG_OA_DATA_LEN;

				if( RC_BAD( rc = insertEntry( &pucKey, &uiKeyLen, pucValue,
					uiLen, uiFlags, &uiChildBlockAddr, &uiCounts, &pucRemainingValue,
					&uiRemainingLen, &eAction)))
				{
					goto Exit;
				}
				
				// Not needed for upper levels of the Btree.
				
				pucValue = NULL;
				uiLen = 0;
				break;
			}

			case ELM_INSERT:
			{
				// This function will return all info needed to handle the next
				// level up in the Btree (if anything), including setting up
				// the stack.  pucKey & uiKeyLen will be pointing to the key that
				// the upper level needs to insert, replace or delete.
				//
				// It will be pointing to an entry in a lower level block, so that
				// block must not be released until after we are all done.

				if( RC_BAD( rc = insertEntry( &pucKey, &uiKeyLen, pucValue,
					uiLen, uiFlags, &uiChildBlockAddr, &uiCounts, &pucRemainingValue,
					&uiRemainingLen, &eAction)))
				{
					goto Exit;
				}
				
				// Not needed for upper levels of the Btree.
				
				pucValue = NULL;
				uiLen = 0;
				break;
			}
			
			case ELM_REPLACE_DO:
			{
				// In this case, the uiLen parameter represents the OADataLength.
				
				uiFlags = BTE_FLAG_DATA_BLOCK |
							 BTE_FLAG_FIRST_ELEMENT |
							 BTE_FLAG_LAST_ELEMENT |
							 BTE_FLAG_OA_DATA_LEN;

				// Should only get here if we are able to truncate the data.
				
				f_assert( bTruncate);

				if( RC_BAD( rc = replaceEntry( &pucKey, &uiKeyLen, pucValue,
					uiLen, uiFlags, &uiChildBlockAddr, &uiCounts,
					&pucRemainingValue, &uiRemainingLen, &eAction)))
				{
					goto Exit;
				}

				// Not needed for upper levels of the Btree.
				
				pucValue = NULL;
				uiLen = 0;
				bTruncate = TRUE;
				break;
			}
			
			case ELM_REPLACE:
			{
				if( RC_BAD( rc = replaceEntry( &pucKey, &uiKeyLen, pucValue,
					uiLen, uiFlags, &uiChildBlockAddr, &uiCounts, &pucRemainingValue,
					&uiRemainingLen, &eAction, bTruncate)))
				{
					goto Exit;
				}

				// Not needed for upper levels of the Btree.
				
				pucValue = NULL;
				uiLen = 0;
				bTruncate = TRUE;
				break;
			}
			
			case ELM_REMOVE:
			{
				if (RC_BAD( rc = removeEntry( &pucKey, &uiKeyLen, &uiChildBlockAddr,
					&uiCounts, &bMoreToRemove, &eAction)))
				{
					goto Exit;
				}

				// Not needed for upper levels of the B-Tree.
				
				pucValue = NULL;
				uiLen = 0;

				break;
			}
			
			case ELM_DONE:
			{
				if( m_pReplaceInfo)
				{
					// This info structure gets generated when the replaced entry in
					// the upper levels is the last entry in the block and we had to
					// move entries to a previous block to accommodate it.
					// We will therefore need to update the parent block with this
					// new information. We need to take care of this before we check
					// for any additional data to store.
					
					if( RC_BAD( rc = restoreReplaceInfo( &pucKey, &uiKeyLen,
						&uiChildBlockAddr, &uiCounts)))
					{
					  goto Exit;
					}
					
					bTruncate = bOrigTruncate;
					eAction = ELM_REPLACE;
				}
				else if( bMoreToRemove)
				{
					eAction = ELM_REMOVE;
					
					// We need to locate where we should remove the entry.
					
					if( RC_BAD( rc = findEntry( pucSavKey, uiSavKeyLen, FLM_EXACT)))
					{
						goto Exit;
					}

				}
				else if( pucRemainingValue && uiRemainingLen)
				{
					eAction = ELM_INSERT;
					
					// We need to locate where we should insert the new entry.
					
					rc = findEntry( pucSavKey, uiSavKeyLen, FLM_EXCL);

					// We could find this entry.  If we get back anything other than
					// an NE_FLM_EOF_HIT or NE_FLM_OK, then there is a problem.
					
					if( rc != NE_FLM_OK && rc != NE_FLM_EOF_HIT &&
						 rc != NE_FLM_NOT_FOUND)
					{
						goto Exit;
					}

					pucValue = pucRemainingValue;
					uiLen = uiRemainingLen;
					pucKey = pucSavKey;
					uiKeyLen = uiSavKeyLen;

					// Make certain that the  BTE_FIRST_ELEMENT flag is NOT set if
					// the first part of the data was stored.
					
					if( uiOrigDataLen != uiLen)
					{
						uiFlags = BTE_FLAG_LAST_ELEMENT;
					}
					else
					{
						uiFlags = BTE_FLAG_FIRST_ELEMENT | BTE_FLAG_LAST_ELEMENT;
					}
				}
				else
				{
					bDone = TRUE;
				}

				break;
			}
			
			// Should never get this!
			
			case ELM_BLK_MERGE:
			{
				rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
				goto Exit;
			}
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	This method will coordinate inserting an entry into a block.  If it
		cannot fit it all in, then it may have to break the entry up so that
		it spans more than one block.  It will also setup for the next level
		before returning.
****************************************************************************/
RCODE F_BTree::insertEntry(
	const FLMBYTE **		ppucKey,
	FLMUINT *				puiKeyLen,
	const FLMBYTE *		pucValue,
	FLMUINT					uiLen,
	FLMUINT					uiFlags,
	FLMUINT *				puiChildBlockAddr,
	FLMUINT *				puiCounts,
	const FLMBYTE **		ppucRemainingValue,
	FLMUINT *				puiRemainingLen,
	F_ELM_UPD_ACTION *	peAction)
{
	RCODE						rc = NE_FLM_OK;
	const FLMBYTE *		pucDataValue = pucValue;
	FLMUINT					uiDataLen = uiLen;
	FLMUINT					uiOADataLen = 0;
	FLMUINT					uiEntrySize = 0;
	FLMBOOL					bEntriesWereMoved = FALSE;
	FLMBOOL					bHaveRoom;
	FLMBOOL					bLastEntry;
	const FLMBYTE *		pucKey = *ppucKey;
	FLMUINT					uiKeyLen = *puiKeyLen;
	FLMUINT					uiChildBlockAddr = *puiChildBlockAddr;
	FLMUINT					uiCounts = *puiCounts;
	IF_Block *				pPrevBlock = NULL;
	FLMBYTE *				pucPrevBlock = NULL;
	FLMBYTE *				pucEntry;
	FLMBOOL					bDefragBlock = FALSE;
	FLMBOOL					bBlockSplit;

	if( m_pStack->uiLevel == 0)
	{
		// We are only safe to do this when we are working on level 0
		// (leaf level) of the Btree.
		
		*ppucRemainingValue = NULL;
		*puiRemainingLen = 0;
	}

	if( *peAction == ELM_INSERT_DO)
	{
		// Adjust the data entry sizes as the data passed in is the 
		// OA Data Length.
		
		uiOADataLen = uiLen;
		uiDataLen = 4;
	}

	// Process until we are done

StartOver:

	if( RC_BAD( rc = calcNewEntrySize( uiKeyLen, uiDataLen, &uiEntrySize,
			&bHaveRoom, &bDefragBlock)))
	{
		goto Exit;
	}

	// Does the entry fit into the block?
	
	if( bHaveRoom)
	{
		if( bDefragBlock)
		{
			// We will have to defragment the block before we can store the data
			
			if( RC_BAD( rc = defragmentBlock( &m_pStack->pBlock, 
				&m_pStack->pucBlock)))
			{
				goto Exit;
			}
		}
		
		if( RC_BAD( rc = storeEntry( pucKey, uiKeyLen, pucDataValue,
				uiDataLen, uiFlags, uiOADataLen, uiChildBlockAddr, uiCounts,
				uiEntrySize, &bLastEntry)))
		{
			goto Exit;
		}

		if( (bLastEntry || m_bCounts) && !isRootBlock( m_pStack->pucBlock))
		{
			// Are we in here because of the counts only?  If so, then we
			// can update the counts right here, no need to continue.
			
			if( !bLastEntry)
			{
				if( RC_BAD( rc = updateCounts()))
				{
					goto Exit;
				}
				
				*peAction = ELM_DONE;
			}
			else
			{
				// Ensure we are updating with the correct key.
				
				pucEntry = BtLastEntry( m_pStack->pucBlock);

				*puiKeyLen = getEntryKeyLength( pucEntry, 
									getBlockType( m_pStack->pucBlock), ppucKey);

				*puiChildBlockAddr = m_pStack->ui32BlockAddr;

				// Do we need counts for the next level?
				
				if( m_bCounts)
				{
					*puiCounts = countKeys( m_pStack->pucBlock);
				}
				
				m_pStack++;
				*peAction = ELM_REPLACE;
			}
		}
		else
		{
			*peAction = ELM_DONE;
		}
		
		goto Exit;
	}

	// Can we move entries around at all to make some room?
	
	if( RC_BAD( rc = moveEntriesToPrevBlock( uiEntrySize, &pPrevBlock, 
		&pucPrevBlock, &bEntriesWereMoved)))
	{
		goto Exit;
	}

	if( bEntriesWereMoved)
	{
		// Only defragment the block if the heap size is not big enough.
		
		if( uiEntrySize > getHeapSize( m_pStack->pucBlock))
		{
			if( RC_BAD( rc = defragmentBlock( &m_pStack->pBlock, 
				&m_pStack->pucBlock)))
			{
				goto Exit;
			}
		}
		
		// Store the entry now because we know there is enough room
		
		if( RC_BAD( rc = storeEntry( pucKey, uiKeyLen, pucDataValue,
				uiDataLen, uiFlags, uiOADataLen, uiChildBlockAddr, uiCounts,
				uiEntrySize, &bLastEntry)))
		{
			goto Exit;
		}

		// Ordinarily, this would NEVER be the last element in the
		// block because we need to adjust the stack to take care of the
		// elements we just moved!  There is only one condition where we would
		// insert as the last entry in the block, and that is when this
		// insert is actually a part of a replace operation where the data
		// is too large to fit in the block.  We had to remove the entry, then
		// insert the new one and we are in the upper levels of the
		// btree. (i.e. not at the leaf).
		
		if( bLastEntry)
		{
			// Since we just added an entry to the last position of the
			// current block.  We will need to preserve the current stack so
			// that we can finish updating the parentage later. Should only
			// happen as a result of a replace operation where the new entry
			// is larger than the existing one while in the upper levels.
			
			if( RC_BAD( rc = saveReplaceInfo( pucKey, uiKeyLen)))
			{
				goto Exit;
			}
		}

		// Need to update the counts of the parents if we are maintining
		// counts before we abandon
		
		if( m_bCounts)
		{
			if( RC_BAD( rc = updateCounts()))
			{
				goto Exit;
			}
		}

		// This method will release any blocks no longer referenced
		// in the stack.  Then pull in the previous block information into
		// the stack.
		
		if( RC_BAD( rc = moveStackToPrev( pPrevBlock, pucPrevBlock)))
		{
			goto Exit;
		}

		// If we are maintaining counts, then lets return a count of the
		// current number of keys referenced below this point.
		
		if( m_bCounts)
		{
			*puiCounts = countKeys( m_pStack->pucBlock);
		}

		f_assert( !isRootBlock( m_pStack->pucBlock));

		// Return the key to the last entry in the prevous block.
		// Recall that we have changed that stack now so that it
		// is referencing the changed block (pPrevBlock).
		
		pucEntry = BtEntry( m_pStack->pucBlock, m_pStack->uiCurOffset);
								  
		*puiKeyLen = getEntryKeyLength( pucEntry,
								getBlockType( pucPrevBlock), ppucKey);

		// Return the new child block address
		
		*puiChildBlockAddr = m_pStack->ui32BlockAddr;

		// Set up to fixup the parentage of the previous block on return...
		
		m_pStack++;

		// Return the new action for the parent block.
		
		*peAction = ELM_REPLACE;
		goto Exit;
	}

	// Try moving to the next block...
	
	if( RC_BAD( rc = moveEntriesToNextBlock( uiEntrySize, &bEntriesWereMoved)))
	{
		goto Exit;
	}

	if( bEntriesWereMoved)
	{
		// Only defragment the block if the heap size is not big enough.
		
		if( uiEntrySize > getHeapSize( m_pStack->pucBlock))
		{
			if( RC_BAD( rc = defragmentBlock( &m_pStack->pBlock,
				&m_pStack->pucBlock)))
			{
				goto Exit;
			}
		}

		// Store the entry now because we know there is enough room
		
		if( RC_BAD( rc = storeEntry( pucKey, uiKeyLen, pucDataValue,
				uiDataLen, uiFlags, uiOADataLen, uiChildBlockAddr, uiCounts,
				uiEntrySize, &bLastEntry)))
		{
			goto Exit;
		}

		// Return the key to the last entry in the current block.
		// Note: If bLastEntry is TRUE, we already know what the key is.
		
		if( !bLastEntry)
		{
			// Get the last key from the block.
			
			pucEntry = BtLastEntry( m_pStack->pucBlock);

			*puiKeyLen = getEntryKeyLength( pucEntry,
									getBlockType( m_pStack->pucBlock), ppucKey);

		}

		f_assert( !isRootBlock( m_pStack->pucBlock));

		// if we are maintaining counts, then lets return a count of the
		// current number of keys referenced below this point.
		
		if( m_bCounts)
		{
			*puiCounts = countKeys( m_pStack->pucBlock);
		}

		// Return the new child block address
		
		*puiChildBlockAddr = m_pStack->ui32BlockAddr;

		// Set up to fixup the parentage of the this block on return...
		
		m_pStack++;
		*peAction = ELM_REPLACE;

		goto Exit;
	}

	// Before we incur the expense of a block split, see if we can store this
	// entry in the previous block.  If we can, we will save some space.  This
	// will only happen if we are trying to insert at the first position in
	// this block.  We would only ever get into this block of code once for
	// each level of the btree.

	if( m_pStack->uiCurOffset == 0 && getPrevInChain( m_pStack->pucBlock))
	{
		if( pPrevBlock)
		{
			pPrevBlock->Release();
			pPrevBlock = NULL;
			pucPrevBlock = NULL;
		}
		
		if( RC_BAD( rc = m_pBlockMgr->getBlock(
			getPrevInChain( m_pStack->pucBlock), &pPrevBlock, &pucPrevBlock)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = moveStackToPrev( pPrevBlock, pucPrevBlock)))
		{
			goto Exit;
		}

		pPrevBlock->Release();
		pPrevBlock = NULL;
		pucPrevBlock = NULL;

		// Increment so we point to one past the last entry.

		m_pStack->uiCurOffset++;
		goto StartOver;
	}

	// We will have to split the block to make room for this entry.
	
	if( RC_BAD( rc = splitBlock( *ppucKey, *puiKeyLen, pucDataValue,
			uiDataLen, uiFlags, uiOADataLen, uiChildBlockAddr, uiCounts,
			ppucRemainingValue, puiRemainingLen, &bBlockSplit)))
	{
		goto Exit;
	}

	// Return the new key value.
	
	pucEntry = BtEntry( m_pStack->pucBlock, m_pStack->uiCurOffset);

	*puiKeyLen = getEntryKeyLength( pucEntry,
						getBlockType( m_pStack->pucBlock), ppucKey);

	// Return the child block address and the counts (if needed).
	
	*puiChildBlockAddr = m_pStack->ui32BlockAddr;

	// Return the counts if we are maintaining them
	
	if( m_bCounts)
	{
		*puiCounts = countKeys( m_pStack->pucBlock);
	}

	// The bBlockSplit boolean will only be FALSE if we were involved in a
	// ReplaceByInsert operation and the call to split resulted in an empty
	// block.  Thus we were able to store the new entry.  In such cases,
	// only the count (if any) need to be updated, not the keys.
	
	if( bBlockSplit)
	{
		*peAction = ELM_INSERT;
		m_pStack++;
	}
	else
	{
		*peAction = ELM_DONE;
	}

Exit:

	if( pPrevBlock)
	{
		pPrevBlock->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:	Method to handle the insertion of a single entry into a block.
		Assumption:  The find method has already been called to locate the 
		insertion point, so the stack has already been setup.
****************************************************************************/
RCODE F_BTree::storeEntry(
	const FLMBYTE *		pucKey,
	FLMUINT					uiKeyLen,
	const FLMBYTE *		pucValue,
	FLMUINT					uiLen,
	FLMUINT					uiFlags,
	FLMUINT					uiOADataLen,
	FLMUINT					uiChildBlockAddr,
	FLMUINT					uiCounts,
	FLMUINT					uiEntrySize,
	FLMBOOL *				pbLastEntry)
{
	RCODE						rc = NE_FLM_OK;
	FLMUINT					uiBlockType = getBlockType( m_pStack->pucBlock);
	FLMBYTE *				pucInsertAt;
	FLMUINT16 *				pui16OffsetArray;
	FLMUINT					uiNumKeys;
	FLMUINT					uiTmp;

	// Assume this is not the last entry for now.
	// We will change it later if needed.

	*pbLastEntry = FALSE;

	// We can go ahead and insert this entry as it is.  All checking has been
	// made before getting to this point.

	uiEntrySize = calcEntrySize( uiBlockType, uiFlags,
										  uiKeyLen, uiLen, uiOADataLen);

	// Log this block before making any changes to it.  Since the
	// pBlock could change, we must update the block header after the call.

	if( RC_BAD( rc = m_pBlockMgr->prepareForUpdate( &m_pStack->pBlock, 
		&m_pStack->pucBlock)))
	{
		goto Exit;
	}

	m_pStack->pui16OffsetArray = BtOffsetArray( m_pStack->pucBlock, 0);
	uiNumKeys = getNumKeys( m_pStack->pucBlock);
	pucInsertAt = getBlockEnd( m_pStack->pucBlock) - uiEntrySize;
	pui16OffsetArray = m_pStack->pui16OffsetArray;

	if( RC_BAD( rc = buildAndStoreEntry( uiBlockType, uiFlags, pucKey, uiKeyLen,
		pucValue, uiLen, uiOADataLen, uiChildBlockAddr, uiCounts,
		pucInsertAt, uiEntrySize, NULL)))
	{
		goto Exit;
	}

	// Now to update the offset in the offset array.  This will move all
	// entries that sort after the new entry down by one position.

	for( uiTmp = uiNumKeys; uiTmp > m_pStack->uiCurOffset; uiTmp--)
	{
		bteSetEntryOffset( pui16OffsetArray, uiTmp,
								 bteGetEntryOffset( pui16OffsetArray, uiTmp - 1));
	}

	bteSetEntryOffset( pui16OffsetArray, m_pStack->uiCurOffset,
							 (FLMUINT16)(pucInsertAt - m_pStack->pucBlock));

	// Update the available space and the number of keys.
	// Account for the new offset entry too.

	decBytesAvail( m_pStack->pucBlock, uiEntrySize + 2);
	decHeapSize( m_pStack->pucBlock, uiEntrySize + 2);
	incNumKeys( m_pStack->pucBlock);

	// Check to see if this was the last entry

	if( m_pStack->uiCurOffset == (FLMUINT)(getNumKeys( m_pStack->pucBlock) - 1))
	{
		*pbLastEntry = TRUE;
	}

	if( !m_pStack->uiLevel && (uiFlags & BTE_FLAG_FIRST_ELEMENT))
	{
		m_ui32PrimaryBlockAddr = m_pStack->ui32BlockAddr;
		m_uiCurOffset = m_pStack->uiCurOffset;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	This method will coordinate removing an entry from a block. If the
		entry spans more than one block, it will set the flag pbMoreToRemove.
		It will also setup for the next level before returning.
****************************************************************************/
RCODE F_BTree::removeEntry(
	const FLMBYTE **		ppucKey,
	FLMUINT *				puiKeyLen,
	FLMUINT *				puiChildBlockAddr,
	FLMUINT *				puiCounts,
	FLMBOOL *				pbMoreToRemove,
	F_ELM_UPD_ACTION *	peAction)
{
	RCODE				rc = NE_FLM_OK;
	FLMBOOL			bLastEntry = FALSE;
	FLMBYTE *		pucEntry;
	FLMBOOL			bMergedWithPrev = FALSE;
	FLMBOOL			bMergedWithNext = FALSE;

	if( m_pStack->uiLevel == 0)
	{
		// We are only safe to do this when we are working on level 0
		// (leaf level) of the Btree.
		
		*pbMoreToRemove = FALSE;
	}

	// Check the current entry to see if it spans more than a single block.

	pucEntry = BtEntry( m_pStack->pucBlock, m_pStack->uiCurOffset);

	// We only need to worry about data spanning more than one block if it is
	// at level zero (i.e. leaf block) and the lastElement flag is not set.

	if( (m_pStack->uiLevel == 0) && m_bTreeHoldsData && 
		 !bteLastElementFlag( pucEntry))
	{
		*pbMoreToRemove = TRUE;
	}

	// Find out if we are looking at the last entry in the block.

	if( m_pStack->uiCurOffset == (FLMUINT)(getNumKeys( m_pStack->pucBlock) - 1))
	{
		bLastEntry = TRUE;
	}

	// Now we remove the entry... Will also remove any chained Data Only blocks

	if( RC_BAD( rc = remove( TRUE)))
	{
		goto Exit;
	}

	// If the block is now empty, we will free the block.

	if( !getNumKeys( m_pStack->pucBlock))
	{
		FLMBOOL			bIsRoot;

		// Test for root block.

		bIsRoot = isRootBlock( m_pStack->pucBlock);

		if( RC_BAD( rc = deleteEmptyBlock()))
		{
			goto Exit;
		}

		// Need to remove the parent entry referencing the deleted block.

		if( !bIsRoot)
		{
			*peAction = ELM_REMOVE;
			m_pStack++;
		}
		else
		{
			// If we ever get here, it means we have just deleted the root block.
			// I have put in the possibility, but typically, deleting the Btree
			// is done by calling btDeleteTree.

			*peAction = ELM_DONE;
		}
	}
	else
	{
		if( ((getBytesAvail( m_pStack->pucBlock) * 100) / m_uiBlockSize) >= 
				BT_LOW_WATER_MARK)
		{
			// We will need to check to see if we can merge two blocks into one to
			// conserve space.

			if( RC_BAD( rc = mergeBlocks( bLastEntry, &bMergedWithPrev,
				&bMergedWithNext, peAction)))
			{
				goto Exit;
			}
		}

		// If the entry that we just removed was the last entry in the block and
		// we did not merge any blocks, we will need to prep for an update to the
		// parent with a new key.

		if( bLastEntry && !bMergedWithPrev && !bMergedWithNext)
		{
			if( m_bCounts)
			{
				*puiCounts = countKeys( m_pStack->pucBlock);
			}

			// Backup to the new "last" entry (remove() does not adjust the offset
			// in the stack).

			f_assert( m_pStack->uiCurOffset > 0);

			m_pStack->uiCurOffset--;
			pucEntry = BtEntry( m_pStack->pucBlock, m_pStack->uiCurOffset);

			*puiKeyLen = getEntryKeyLength( pucEntry,
								getBlockType( m_pStack->pucBlock), ppucKey);

			*puiChildBlockAddr = m_pStack->ui32BlockAddr;
			*peAction = ELM_REPLACE;
			m_pStack++;
		}
		else
		{
			// Are we tracking counts?

			if( !bMergedWithPrev && !bMergedWithNext)
			{
				if( m_bCounts)
				{
					if( RC_BAD( rc = updateCounts()))
					{
						goto Exit;
					}
				}
				
				*peAction = ELM_DONE;
			}
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method to replace an existing entry with a new one.
****************************************************************************/
RCODE F_BTree::replaceEntry(
	const FLMBYTE **		ppucKey,
	FLMUINT *				puiKeyLen,
	const FLMBYTE *		pucValue,
	FLMUINT					uiLen,
	FLMUINT					uiFlags,
	FLMUINT *				puiChildBlockAddr,
	FLMUINT *				puiCounts,
	const FLMBYTE **		ppucRemainingValue,
	FLMUINT *				puiRemainingLen,
	F_ELM_UPD_ACTION *	peAction,
	FLMBOOL					bTruncate)
{
	RCODE					rc = NE_FLM_OK;
	const FLMBYTE *	pucDataValue = pucValue;
	FLMUINT				uiDataLen = uiLen;
	FLMUINT				uiOADataLen = 0;
	FLMBYTE *			pucEntry = NULL;
	FLMUINT32			ui32OrigDOAddr = 0;
	const FLMBYTE *	pucData = NULL;

	if( m_pStack->uiLevel == 0)
	{
		*ppucRemainingValue = NULL;
		*puiRemainingLen = 0;
	}

	if( *peAction == ELM_REPLACE_DO)
	{
		// Adjust the data entry sizes as the data passed in
		// is the OA Data Length.
		
		uiOADataLen = uiLen;
		uiDataLen = 4;
	}

	if( m_pStack->uiLevel == 0 && m_bTreeHoldsData)
	{
		if( m_bOrigInDOBlocks)
		{
			f_assert( bTruncate);

			pucEntry = BtEntry( m_pStack->pucBlock, m_pStack->uiCurOffset);

			fbtGetEntryDataLength( pucEntry, &pucData, NULL, NULL);
			ui32OrigDOAddr = bteGetBlockAddr( pucData);
		}
	}

	// We only have to worry about updating the upper levels of the Btree
	// when we are doing a replacement at a non-leaf level or we are maintaining
	// counts.  Replacements at the leaf level do not require a change in the
	// parent block.  The only exception is when the old entry spanned to
	// another block, but the new one did not.  This results in removing the
	// excess part of the old entry unless we are not truncating the element.
	// Even then, we only update the parent if the excess entry was the only key
	// in the block, i.e. the block became empty as a result of the removal.
	// All of this would have been handled already by the time we return from
	// this call.

	// When bTruncate is FALSE we do not trim back the entry so we don't worry
	// about updating the parentage.
	
	if( RC_BAD( rc = replaceOldEntry( ppucKey, puiKeyLen, pucDataValue,
		uiDataLen, uiFlags, uiOADataLen, puiChildBlockAddr, puiCounts,
		ppucRemainingValue, puiRemainingLen, peAction, bTruncate)))
	{
		goto Exit;
	}

	// Do we need to free the original DO blocks since they are not
	// used in the new entry?

	if( m_bOrigInDOBlocks && !m_bDataOnlyBlock && m_pStack->uiLevel == 0)
	{
		if( RC_BAD( rc = removeDOBlocks( ui32OrigDOAddr)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method to handle replacing a single entry in a block.
		ASSUMPTION:  The find method has already been called to locate the
		insertion point, so the stack has already been setup.
****************************************************************************/
RCODE F_BTree::replaceOldEntry(
	const FLMBYTE **		ppucKey,
	FLMUINT *				puiKeyLen,
	const FLMBYTE *		pucValue,
	FLMUINT					uiLen,
	FLMUINT					uiFlags,
	FLMUINT					uiOADataLen,
	FLMUINT *				puiChildBlockAddr,
	FLMUINT *				puiCounts,
	const FLMBYTE **		ppucRemainingValue,
	FLMUINT *				puiRemainingLen,
	F_ELM_UPD_ACTION *	peAction,
	FLMBOOL					bTruncate)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT				uiOldEntrySize;
	FLMBYTE *			pucEntry = NULL;
	FLMBYTE *			pucData = NULL;
	FLMUINT				uiEntrySize;
	FLMBOOL				bLastEntry = FALSE;
	FLMBOOL				bLastElement = TRUE;
	FLMBOOL				bHaveRoom;
	FLMBOOL				bDefragBlock;
	FLMUINT				uiDataLen = 0;
	FLMUINT				uiOldOADataLen = 0;
	FLMBOOL				bRemoveOADataAllowance = FALSE;
	FLMBYTE *			pucTmpBlock = NULL;
	void *				pvPoolMark = m_pool.poolMark();

	uiOldEntrySize = actualEntrySize( getEntrySize( m_pStack->pucBlock,
												  m_pStack->uiCurOffset, &pucEntry));

	if( m_pStack->uiLevel == 0 && m_bTreeHoldsData)
	{
		bLastElement = bteLastElementFlag( pucEntry);

		uiDataLen = fbtGetEntryDataLength( pucEntry, (const FLMBYTE **)&pucData, 
			&uiOldOADataLen, NULL);

		// Test to see if we need to worry about the bTruncate flag.

		if( uiDataLen == uiOldOADataLen)
		{
			if( uiLen > uiDataLen)
			{
				bTruncate = TRUE;
			}
			else if( uiLen <= uiDataLen && uiOADataLen == 0)
			{
				bRemoveOADataAllowance = TRUE;
			}
		}
		else
		{
			if( uiLen > uiOldOADataLen)
			{
				bTruncate = TRUE;
			}
		}
	}

	// bTruncate has no meaning if we have no data or we are not at the
	// leaf level.

	if( m_pStack->uiLevel != 0 || !m_bTreeHoldsData)
	{
		bTruncate = TRUE;
	}

	// The calcNewEntrySize function will tack on 2 bytes for the offset.
	// It also adds an extra 4 bytes for the OADataLen, even though it may
	// not be needed.  We will need to be aware of this here as it may affect
	// our decision as to how we will replace the entry.

	if( RC_BAD( rc = calcNewEntrySize( *puiKeyLen, uiLen, &uiEntrySize,
			&bHaveRoom, &bDefragBlock)))
	{
		goto Exit;
	}

	if( bRemoveOADataAllowance)
	{
		uiEntrySize -= 4;
	}

	// Since this is a replace operation, we don't need to know about the offset
	// as that won't be a factor in what we are doing. 'actualEntrySize' will
	// remove those two bytyes from the size.

	uiEntrySize = actualEntrySize( uiEntrySize);
	if( uiEntrySize <= uiOldEntrySize)
	{
		if( !bTruncate)
		{
			f_assert( uiLen <= uiDataLen);
			f_memcpy( pucData, pucValue, uiLen);

			if( m_pStack->uiCurOffset == 
					(FLMUINT)(getNumKeys( m_pStack->pucBlock) - 1))
			{
				bLastEntry = TRUE;
			}
		}
		else
		{
			if( !pucTmpBlock)
			{
				if( RC_BAD( rc = m_pool.poolAlloc( m_uiBlockSize, 
					(void **)&pucTmpBlock)))
				{
					goto Exit;
				}
			}
			
			// We can go ahead and replace this entry as it is.  All checking
			// has been made before getting to this point.

			if( RC_BAD( rc = buildAndStoreEntry( 
				getBlockType( m_pStack->pucBlock),
				uiFlags, *ppucKey, *puiKeyLen, pucValue, uiLen, uiOADataLen,
				*puiChildBlockAddr, *puiCounts, pucTmpBlock, m_uiBlockSize,
				&uiEntrySize)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = replace( pucTmpBlock, uiEntrySize, &bLastEntry)))
			{
				goto Exit;
			}
		}

		if( !bLastElement && bTruncate)
		{
			// The element that we replaced actually spans more than one entry.
			// We will have to remove the remaining entries.

			if( RC_BAD( rc = removeRemainingEntries( *ppucKey, *puiKeyLen)))
			{
				goto Exit;
			}
		}

		if( (bLastEntry || m_bCounts) && !isRootBlock( m_pStack->pucBlock) &&
			  (m_pStack->uiLevel != 0))
		{
			// Are we in here because of the counts only?  If so, then make
			// sure we don't change the key in the parent.

			if( !bLastEntry)
			{
				if( RC_BAD( rc = updateCounts()))
				{
					goto Exit;
				}
				
				*peAction = ELM_DONE;
			}
			else
			{
				// Return the key to the last entry in the block.

				pucEntry = BtLastEntry( m_pStack->pucBlock);

				*puiKeyLen = getEntryKeyLength( pucEntry,
									getBlockType( m_pStack->pucBlock), ppucKey);
				*puiChildBlockAddr = m_pStack->ui32BlockAddr;

				// Do we need counts for the next level?

				if( m_bCounts)
				{
					*puiCounts = countKeys( m_pStack->pucBlock);
				}
				
				m_pStack++;
				*peAction = ELM_REPLACE;
			}
		}
		else
		{
			*peAction = ELM_DONE;
		}
		
		goto Exit;
	}

	// If we do not have a stack setup yet (which can happen if the replace
	// is trying to shortcut to the previously known block address and offset),
	// then at this point, we must build the stack, since it may be required
	// to adjust the upper levels of the btree.

	if( !m_bStackSetup)
	{
		if( RC_BAD( rc = findEntry( *ppucKey, *puiKeyLen, FLM_EXACT)))
		{
			goto Exit;
		}
	}

	// The new entry will not fit into the original entry's space.
	// If we remove the entry in the block, will there be enough room
	// to put it in?

	if( bTruncate && getBytesAvail( m_pStack->pucBlock) +
		 uiOldEntrySize >= uiEntrySize)
	{
		// First remove the current entry.  Do not delete any DO blocks chained
		// to this entry.

		if( RC_BAD( rc = remove( FALSE)))
		{
			goto Exit;
		}

		if( (getBytesAvail( m_pStack->pucBlock) != 
			  getHeapSize( m_pStack->pucBlock)) &&
			 ((uiEntrySize + 2) > getHeapSize( m_pStack->pucBlock)))
		{
			if( RC_BAD( rc = defragmentBlock( &m_pStack->pBlock, 
				&m_pStack->pucBlock)))
			{
				goto Exit;
			}
		}

		// Now insert the new entry.

		if( RC_BAD( rc = storeEntry( *ppucKey, *puiKeyLen, pucValue, uiLen,
				uiFlags, uiOADataLen, *puiChildBlockAddr, *puiCounts, uiEntrySize,
				&bLastEntry)))
		{
			goto Exit;
		}

		// Check if the original element spanned more than one entry

		if( !bLastElement)
		{
			// The element that we replaced actually spans more than one entry.
			// We will have to remove the remaining entries.

			if( RC_BAD( rc = removeRemainingEntries( *ppucKey, *puiKeyLen)))
			{
				goto Exit;
			}
		}

		if( (bLastEntry || m_bCounts) && !isRootBlock( m_pStack->pucBlock) &&
			 (m_pStack->uiLevel != 0))
		{
			// Are we in here because of the counts only?

			if( !bLastEntry)
			{
				if( RC_BAD( rc = updateCounts()))
				{
					goto Exit;
				}
				
				*peAction = ELM_DONE;
			}
			else
			{
				// Set the key to the last entry in the block.

				pucEntry = BtLastEntry( m_pStack->pucBlock);

				*puiKeyLen = getEntryKeyLength( pucEntry,
									getBlockType( m_pStack->pucBlock), ppucKey);
				*puiChildBlockAddr = m_pStack->ui32BlockAddr;

				// Do we need counts for the next level?

				if( getBlockType( m_pStack->pucBlock) == F_BLK_TYPE_BT_NON_LEAF_COUNTS)
				{
					*puiCounts = countKeys( m_pStack->pucBlock);
				}
				
				m_pStack++;
				*peAction = ELM_REPLACE;
			}
		}
		else
		{
			*peAction = ELM_DONE;
		}
		
		goto Exit;
	}

	// If the original element does not span multiple entries and we still don't
	// have room for the replacement, then we will remove this entry and insert
	// the replacement.  When the insert happens, it will take care of moving
	// things around or splitting the block as needed to get it in.  If bTruncate
	// is FALSE, and the new entry is larger than the original, we can ignore it.

	if( bLastElement)
	{
		if( RC_BAD( rc = replaceByInsert( ppucKey, puiKeyLen,
			pucValue, uiLen, uiOADataLen, uiFlags, puiChildBlockAddr,
			puiCounts, ppucRemainingValue, puiRemainingLen,
			peAction)))
		{
			goto Exit;
		}

		goto Exit;
	}

	if( bTruncate)
	{
		if( RC_BAD( rc = replaceMultiples( ppucKey, puiKeyLen, pucValue,
			uiLen, ppucRemainingValue, puiRemainingLen, peAction)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = replaceMultiNoTruncate( ppucKey, puiKeyLen,
			pucValue, uiLen, ppucRemainingValue, puiRemainingLen, peAction)))
		{
			goto Exit;
		}
	}

Exit:

	m_pool.poolReset( pvPoolMark);
	return( rc);
}

/***************************************************************************
Desc:	This method is called whenever a replacement entry will not fit in
		the block, even if we remove the existing entry.  It ASSUMES that the
		original element does not continue to another entry, either in the
		same block or in another block.
****************************************************************************/
RCODE F_BTree::replaceByInsert(
	const FLMBYTE **		ppucKey,
	FLMUINT *				puiKeyLen,
	const FLMBYTE *		pucDataValue,
	FLMUINT					uiDataLen,
	FLMUINT					uiOADataLen,
	FLMUINT					uiFlags,
	FLMUINT *				puiChildBlockAddr,
	FLMUINT *				puiCounts,
	const FLMBYTE **		ppucRemainingValue,
	FLMUINT *				puiRemainingLen,
	F_ELM_UPD_ACTION *	peAction)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiLen = uiDataLen;

	if( *peAction == ELM_REPLACE_DO)
	{
		uiLen = uiOADataLen;
		*peAction = ELM_INSERT_DO;
	}
	else
	{
		*peAction = ELM_INSERT;
	}

	// At this point, it is clear that this new entry is larger than the
	// old entry.  We will remove the old entry first.  Then we can treat
	// this whole operation as an insert rather than as a replace.

	if( RC_BAD( rc = remove( FALSE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = insertEntry( ppucKey, puiKeyLen, pucDataValue, uiLen,
		uiFlags, puiChildBlockAddr, puiCounts, ppucRemainingValue, puiRemainingLen,
		peAction)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method to replace an entry in a block and update the available
		space.  This method expects to receive a buffer with an entry already
		prepared to be written to the block.
****************************************************************************/
RCODE F_BTree::replace(
	FLMBYTE *		pucEntry,
	FLMUINT			uiEntrySize,
	FLMBOOL *		pbLastEntry)
{
	RCODE						rc = NE_FLM_OK;
	FLMBYTE *				pucReplaceAt;
	FLMUINT					uiNumKeys;
	FLMUINT					uiOldEntrySize;

	*pbLastEntry = FALSE;

	// Log this block before making any changes to it.  Since the
	// pBlock could change, we must update the block header after the call.

	if( RC_BAD( rc = m_pBlockMgr->prepareForUpdate( 
		&m_pStack->pBlock, &m_pStack->pucBlock)))
	{
		goto Exit;
	}
	
	m_pStack->pui16OffsetArray = BtOffsetArray( m_pStack->pucBlock, 0);
	uiNumKeys = getNumKeys( m_pStack->pucBlock);
	uiOldEntrySize = actualEntrySize( 
							getEntrySize( m_pStack->pucBlock, m_pStack->uiCurOffset));

	f_assert( uiOldEntrySize >= uiEntrySize);

	pucReplaceAt = BtEntry( m_pStack->pucBlock, m_pStack->uiCurOffset);

	// Let's go ahead and copy the entry into the block now.

	f_memcpy( pucReplaceAt, pucEntry, uiEntrySize);

#ifdef FLM_DEBUG
	// Clean up the empty space (if any)

	if( uiOldEntrySize > uiEntrySize)
	{
		pucReplaceAt += uiEntrySize;
		f_memset( pucReplaceAt, 0, uiOldEntrySize - uiEntrySize);
	}
#endif

	// Update the available space.  It may not have changed at all if the
	// two entries are the same size. The Heap size will not have changed.
	// This is because we write the entry into the same location as the
	// original.  Even though the new entry may be smaller, we start at
	// the same location, possibly leaving a hole in the block.

	incBytesAvail( m_pStack->pucBlock, uiOldEntrySize - uiEntrySize);

	if( m_pStack->uiCurOffset == (FLMUINT)(getNumKeys( m_pStack->pucBlock) - 1))
	{
		*pbLastEntry = TRUE;
	}

	// Preserve the block and offset index in case it is wanted on the way out.

	if( !m_pStack->uiLevel && bteFirstElementFlag( pucEntry))
	{
		m_ui32PrimaryBlockAddr = m_pStack->ui32BlockAddr;
		m_uiCurOffset = m_pStack->uiCurOffset;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method to rebuild the stack so that it references the parentage of
		the parameter pBlock.  The assumption is that we will begin at
		whatever level m_pStack is currently sitting at. Therefore, this
		method can be called for any level in the Btree.
****************************************************************************/
RCODE F_BTree::moveStackToPrev(
	IF_Block *			pBlock,
	FLMBYTE *			pucBlock)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT				uiBlockAddr;
	F_BTSK *				pStack = m_pStack;
	IF_Block *			pPrevBlock = NULL;
	FLMBYTE *			pucPrevBlock = NULL;

	if( pBlock)
	{
		if( pStack->pBlock)
		{
			// Make sure the block we passed in really is the previous
			// block in the chain.

			if( getBlockAddr( pucBlock) != getPrevInChain( pStack->pucBlock))
			{
				rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
				goto Exit;
			}

			// Cannot be the same block.

			if( pBlock == pStack->pBlock)
			{
				rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
				goto Exit;
			}

			// Release the current block.  We don't need to fetch
			// the new block because it was passed in to us.  If
			// we encounter this situation further up the tree,
			// we will have to fetch the block as well.

			pStack->pBlock->Release();
			pStack->pBlock = NULL;
			pStack->pucBlock = NULL;
		}

		pStack->pBlock = pBlock;
		pStack->pucBlock = pucBlock;
		pStack->pBlock->AddRef();
		
		pStack->ui32BlockAddr = getBlockAddr( pucBlock);
		pStack->uiCurOffset = getNumKeys( pucBlock) - 1;
		pStack->uiLevel =	 getBlockLevel( pucBlock);
		pStack->pui16OffsetArray = BtOffsetArray( pucBlock, 0);

		// Now walk up the stack until done.

		pStack++;
	}

	for (;;)
	{
		// If we don't have this block in the stack, we must first get it.

		if( !pStack->pBlock)
		{
			// Don't continue if we don't have this level in the stack.

			if( !pStack->ui32BlockAddr)
			{
				break;
			}

			if( RC_BAD( rc = m_pBlockMgr->getBlock( pStack->ui32BlockAddr, 
				&pStack->pBlock, &pStack->pucBlock)))
			{
				goto Exit;
			}
		}

		// See if we need to go to the previous block.

		if( !pStack->uiCurOffset)
		{
			// If this is the root block and we are looking at the first
			// entry in the block, then we have a problem.

			if( !isRootBlock( pStack->pucBlock))
			{
				// When the stack is pointing to the first entry, this
				// means that we want the target stack to point to the previous
				// block in the chain.

				uiBlockAddr = getPrevInChain( pStack->pucBlock);
				f_assert( uiBlockAddr);

				// Fetch the new block

				if( RC_BAD( rc = m_pBlockMgr->getBlock( (FLMUINT32)uiBlockAddr, 
					&pPrevBlock, &pucPrevBlock)))
				{
					goto Exit;
				}

				// Release the old block

				pStack->pBlock->Release();
				pStack->pBlock = pPrevBlock;
				pStack->pucBlock = pucPrevBlock;
				
				pPrevBlock = NULL;
				pucPrevBlock = NULL;
				
				pStack->ui32BlockAddr = getBlockAddr( pStack->pucBlock);
				pStack->uiCurOffset = getNumKeys( pStack->pucBlock) - 1;
				pStack->uiLevel = getBlockLevel( pStack->pucBlock);
				pStack->pui16OffsetArray = BtOffsetArray( pStack->pucBlock, 0);
			}
			else
			{
				// We have no previous.

				rc = RC_SET( NE_FLM_BOF_HIT);
				goto Exit;
			}
		}
		else
		{
			// Move to the previous entry
			
			pStack->uiCurOffset--;
			break;
		}
		
		pStack++;
	}

Exit:

	if( pPrevBlock)
	{
		pPrevBlock->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:	Method to rebuild the stack so that it references the parentage of
		the parameter pBlock. The assumption is that we will begin at
		whatever level m_pStack is currently sitting at.  Therefore, this 
		method can be called for any level in the Btree.
****************************************************************************/
RCODE F_BTree::moveStackToNext(
	IF_Block *			pBlock,
	FLMBYTE *			pucBlock)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT				uiBlockAddr;
	F_BTSK *				pStack = m_pStack;

	if( pBlock)
	{
		if( pStack->pBlock)
		{
			// Make sure the block we passed in really is the next in chain.

			if( getBlockAddr( pucBlock) != getNextInChain( pStack->pucBlock))
			{
				rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
				goto Exit;
			}

			// Cannot be the same block.

			if( pBlock == pStack->pBlock)
			{
				rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
				goto Exit;
			}

			// Release the current block.  We don't need to fetch
			// the new block because it was passed in to us.  If
			// we encounter this situation further up the tree,
			// we will have to fetch the block as well.

			pStack->pBlock->Release();
			pStack->pBlock = NULL;
			pStack->pucBlock = NULL;
		}
		
		f_assert( !pStack->pBlock && !pStack->pucBlock);
		
		pStack->pBlock = pBlock;
		pStack->pucBlock = pucBlock;
		pStack->pBlock->AddRef();
		
		pStack->ui32BlockAddr = getBlockAddr( pucBlock);
		pStack->uiCurOffset = 0;
		pStack->uiLevel = getBlockLevel( pucBlock);
		pStack->pui16OffsetArray = BtOffsetArray( pucBlock, 0);

		// Now walk up the stack until done.

		pStack++;
	}

	for (;;)
	{
		// If we don't currently have the block, let's get it.

		if( !pStack->pBlock)
		{
			if( RC_BAD( rc = m_pBlockMgr->getBlock( pStack->ui32BlockAddr,
				&pStack->pBlock, &pStack->pucBlock)))
			{
				goto Exit;
			}
		}

		// See if we need to go to the next block.

		if( pStack->uiCurOffset == (FLMUINT)(getNumKeys( pStack->pucBlock) - 1))
		{
			// If this is the root block and we are looking at the last entry in the
			// block, then we have a problem.

			if( !isRootBlock( pStack->pucBlock))
			{
				// When the stack is pointing to the last entry, this
				// means that we want the target stack to point the next block in
				// the chain.

				uiBlockAddr = getNextInChain( pStack->pucBlock);
				f_assert( uiBlockAddr);

				// Get the next block
				
				if( RC_BAD( rc = getNextBlock( &pStack->pBlock, 
					&pStack->pucBlock)))
				{
					goto Exit;
				}

				pStack->ui32BlockAddr = getBlockAddr( pStack->pucBlock);
				pStack->uiCurOffset = 0;
				pStack->uiLevel = getBlockLevel( pStack->pucBlock);
				pStack->pui16OffsetArray = BtOffsetArray( pStack->pucBlock, 0);
			}
			else
			{
				// We should never have to attempt to get a previous block
				// on the root.

				rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
				goto Exit;
			}
		}
		else
		{
			// Move to the next entry
			
			pStack->uiCurOffset++;
			break;
		}
		
		pStack++;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method to calculate the actual entry size of a new entry
****************************************************************************/
RCODE F_BTree::calcNewEntrySize(
	FLMUINT			uiKeyLen,
	FLMUINT			uiDataLen,
	FLMUINT *		puiEntrySize,
	FLMBOOL *		pbHaveRoom,
	FLMBOOL *		pbDefragBlock)
{
	RCODE				rc = NE_FLM_OK;

	// Calculate the entry size.

	switch( getBlockType( m_pStack->pucBlock))
	{
		case F_BLK_TYPE_BT_LEAF:
		{
			// This block type is a leaf block, No Data
			
			*puiEntrySize = BTE_LEAF_OVHD + uiKeyLen;
			break;
		}

		case F_BLK_TYPE_BT_LEAF_DATA:
		{
			// Leaf block with data
			
			*puiEntrySize = BTE_LEAF_DATA_OVHD +
								 (uiKeyLen > ONE_BYTE_SIZE ? 2 : 1) +
								 (uiDataLen > ONE_BYTE_SIZE ? 2 : 1) +
								 uiKeyLen + uiDataLen;
			break;
		}

		case F_BLK_TYPE_BT_NON_LEAF:
		{
			*puiEntrySize = BTE_NON_LEAF_OVHD + uiKeyLen;
			break;
		}

		case F_BLK_TYPE_BT_NON_LEAF_COUNTS:
		{
			*puiEntrySize = BTE_NON_LEAF_COUNTS_OVHD + uiKeyLen;
			break;
		}

		default:
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
			*puiEntrySize = 0;
			goto Exit;
		}
	}

	// See if we have room in the heap first.  If not, maybe we can make
	// room by defraging the block.

	if( *puiEntrySize <= getHeapSize( m_pStack->pucBlock))
	{
		*pbDefragBlock = FALSE;
		*pbHaveRoom = TRUE;
	}
	else if( *puiEntrySize <= getBytesAvail( m_pStack->pucBlock))
	{
		// A defrag of the block is required to make room.  We will only defrag
		// if we can recover a minimum of 5% of the total block size.

		if( getBytesAvail( m_pStack->pucBlock) >= m_uiDefragThreshold)
		{
			*pbHaveRoom = TRUE;
			*pbDefragBlock = TRUE;
		}
		else
		{
			*pbHaveRoom = FALSE;
			*pbDefragBlock = FALSE;
		}
	}
	else
	{
		*pbHaveRoom = FALSE;
		*pbDefragBlock = FALSE;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Function to save the replacement information that we could not store
		on the current go round.  The replace function will check for the
		presence of this structure and deal with it later.
****************************************************************************/
RCODE F_BTree::saveReplaceInfo(
	const FLMBYTE *	pucNewKey,
	FLMUINT				uiNewKeyLen)
{
	RCODE								rc = NE_FLM_OK;
	BTREE_REPLACE_STRUCT *		pPrev;
	F_BTSK *							pStack = m_pStack;
	const FLMBYTE *				pucParentKey;
	FLMBYTE *						pucEntry;

	if( m_uiReplaceLevels + 1 >= F_BTREE_MAX_LEVELS)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
		goto Exit;
	}

	pPrev = m_pReplaceInfo;
	m_pReplaceInfo = &m_pReplaceStruct[ m_uiReplaceLevels++];
	m_pReplaceInfo->pPrev = (void *)pPrev;

	// We should not be at the root level already!

	f_assert( pStack->uiLevel != m_uiStackLevels - 1);

	m_pReplaceInfo->uiParentLevel = pStack->uiLevel+1;
	m_pReplaceInfo->uiNewKeyLen = uiNewKeyLen;
	m_pReplaceInfo->uiChildBlockAddr = pStack->ui32BlockAddr;
	
	if( m_bCounts)
	{
		m_pReplaceInfo->uiCounts = countKeys( pStack->pucBlock);
	}
	else
	{
		m_pReplaceInfo->uiCounts = 0;
	}

	f_memcpy( &m_pReplaceInfo->pucNewKey[0], pucNewKey, uiNewKeyLen);

	pStack++;
	pucEntry = BtEntry( pStack->pucBlock, pStack->uiCurOffset);

	m_pReplaceInfo->uiParentKeyLen = getEntryKeyLength( pucEntry,
						getBlockType( pStack->pucBlock), &pucParentKey);

	f_memcpy( &m_pReplaceInfo->pucParentKey[0], pucParentKey, 
				 m_pReplaceInfo->uiParentKeyLen);

	m_pReplaceInfo->uiParentChildBlockAddr = bteGetBlockAddr( pucEntry);

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method to restore the stack to a state where we  can finish updating
		the parent with the new key information.
****************************************************************************/
RCODE F_BTree::restoreReplaceInfo(
	const FLMBYTE **	ppucKey,
	FLMUINT *			puiKeyLen,
	FLMUINT *			puiChildBlockAddr,
	FLMUINT *			puiCounts)
{
	RCODE					rc = NE_FLM_OK;
	RCODE					rcTmp = NE_FLM_OK;
	FLMUINT				uiLoop;
	FLMBYTE *			pucEntry;
	FLMUINT				uiKeyLen;
	const FLMBYTE *	pucKey;
	FLMUINT				uiSearchLevel = m_uiSearchLevel;
	FLMUINT				uiStackLevels = m_uiStackLevels;

	// We will need to redo our stack from the top down to
	// make sure we are looking at the correct blocks.

	m_uiSearchLevel = m_uiStackLevels - m_pReplaceInfo->uiParentLevel - 1;
	rcTmp = findEntry( m_pReplaceInfo->pucParentKey,
							 m_pReplaceInfo->uiParentKeyLen, FLM_EXACT);

	m_uiSearchLevel = uiSearchLevel;

	if ((rcTmp != NE_FLM_OK) &&
		 (rcTmp != NE_FLM_NOT_FOUND) &&
		 (rcTmp != NE_FLM_EOF_HIT))
	{
		rc = RC_SET( rcTmp);
		goto Exit;
	}

	// Set the stack pointer to the parent level that we want to replace.

	m_pStack = &m_Stack[ m_pReplaceInfo->uiParentLevel];

	// There is always the possibility that the key we are searching for
	// has a duplicate key ahead of it, as a result of a continuation element.
	// We really must replace the entry we were looking at when the information
	// was stored, therefore, we will verify that we have the right entry.

	for( ;;)
	{
		pucEntry = BtEntry( m_pStack->pucBlock, m_pStack->uiCurOffset);

		uiKeyLen = getEntryKeyLength( pucEntry, 
							getBlockType( m_pStack->pucBlock), &pucKey);

		if( uiKeyLen != m_pReplaceInfo->uiParentKeyLen)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
			goto Exit;
		}

		if( f_memcmp( &m_pReplaceInfo->pucParentKey[0], pucKey, uiKeyLen) == 0)
		{
			if( bteGetBlockAddr( pucEntry) != m_pReplaceInfo->uiParentChildBlockAddr)
			{
				// Try moving forward to the next entry ...
				
				if( RC_BAD( rc = moveStackToNext( NULL, NULL)))
				{
					rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
					goto Exit;
				}
			}
			else
			{
				break;
			}
		}
		else
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
			goto Exit;
		}
	}

	// Now return the other important stuff

	*puiChildBlockAddr = m_pReplaceInfo->uiChildBlockAddr;
	*puiKeyLen = m_pReplaceInfo->uiNewKeyLen;
	*puiCounts = m_pReplaceInfo->uiCounts;

	for( uiLoop = 0; uiLoop < m_uiStackLevels; uiLoop++)
	{
		m_Stack[ uiLoop].uiKeyLen = m_pReplaceInfo->uiNewKeyLen;
	}

	m_uiStackLevels = uiStackLevels;

	// Point to the key

	*ppucKey = &m_pReplaceInfo->pucNewKey[ 0];

	// Free the current ReplaceInfo Buffer

	m_pReplaceInfo = (BTREE_REPLACE_STRUCT *)m_pReplaceInfo->pPrev;
	m_uiReplaceLevels--;

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method to set the key to be returned to the caller.
****************************************************************************/
FINLINE RCODE F_BTree::setReturnKey(
	FLMBYTE *		pucEntry,
	FLMUINT			uiBlockType,
	FLMBYTE *		pucKey,
	FLMUINT *		puiKeyLen,
	FLMUINT			uiKeyBufSize)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT				uiKeyLen;
	const FLMBYTE *	pucKeyRV;

	uiKeyLen = getEntryKeyLength( pucEntry, uiBlockType, &pucKeyRV);
	
	if( uiKeyLen == 0)
	{
		// We hit the LEM, hence the EOF error
		
		rc = RC_SET( NE_FLM_EOF_HIT);
		goto Exit;
	}

	if( uiKeyLen <= uiKeyBufSize)
	{
		f_memcpy( pucKey, pucKeyRV, uiKeyLen);
		*puiKeyLen = uiKeyLen;
	}
	else
	{
		rc = RC_SET( NE_FLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc: Method to return the data from either the BTREE block or
		the DO block.  It will update the tracking variables too.
		This method assumes that the m_pBlock has already been setup for 
		the 1st go-round.
****************************************************************************/
RCODE F_BTree::extractEntryData(
	FLMBYTE *			pucKey,
	FLMUINT				uiKeyLen,
	FLMBYTE *			pucBuffer,
	FLMUINT				uiBufSiz,
	FLMUINT *			puiDataLen,
	FLMBYTE **			ppucDataPtr)
{
	RCODE					rc = NE_FLM_OK;
	FLMBYTE *			pucDestPtr = pucBuffer;
	FLMUINT32			ui32BlockAddr = 0;
	FLMBOOL				bNewBlock;
	FLMUINT				uiDataLen = 0;

	f_assert( m_pBlock && m_pucBlock);

	if( puiDataLen)
	{
		*puiDataLen = 0;
	}

#ifdef FLM_DEBUG
	if( pucBuffer)
	{
		f_memset( pucBuffer, 0, uiBufSiz);
	}
#endif

	// Is there anything to read?
	
	if( m_uiOADataRemaining == 0)
	{
		rc = RC_SET( NE_FLM_EOF_HIT);
		goto Exit;
	}

	while( m_uiOADataRemaining && (uiDataLen < uiBufSiz))
	{
		if( m_uiDataRemaining <= (uiBufSiz - uiDataLen))
		{
			// Let's take what we have left in this block first.

			if( pucDestPtr)
			{
				f_memcpy( pucDestPtr, *ppucDataPtr, m_uiDataRemaining);
				pucDestPtr += m_uiDataRemaining;
			}

			uiDataLen += m_uiDataRemaining;
			m_uiOADataRemaining -= m_uiDataRemaining;
			m_uiDataRemaining = 0;
		}
		else
		{
			// Buffer is too small to hold everything in this block.
			
			if( pucDestPtr)
			{
				f_memcpy( pucDestPtr, *ppucDataPtr, uiBufSiz - uiDataLen);
				pucDestPtr += (uiBufSiz - uiDataLen);
			}

			(*ppucDataPtr) += (uiBufSiz - uiDataLen);
			m_uiOADataRemaining -= (uiBufSiz - uiDataLen);
			m_uiDataRemaining -= (uiBufSiz - uiDataLen);
			uiDataLen += (uiBufSiz - uiDataLen);
		}

		// If there is still more overall data remaining, we need to get the
		// next DO block or standard block and setup to read it too.
		// i.e. More to come, but nothing left in this block.
		
		if( (m_uiOADataRemaining > 0) && (m_uiDataRemaining == 0))
		{
			if (!m_bDataOnlyBlock &&
				 (m_uiCurOffset < (FLMUINT)(getNumKeys( m_pucBlock) - 1)))
			{
				m_uiCurOffset++;
				bNewBlock = FALSE;
			}
			else
			{
				// Get the next block address
				
				ui32BlockAddr = getNextInChain( m_pucBlock);

				// Release the current block before we get the next one.
				
				m_pBlock->Release();
				m_pBlock = NULL;
				m_pucBlock = NULL;

				if( ui32BlockAddr == 0)
				{
					rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
					goto Exit;
				}

				if( RC_BAD( rc = m_pBlockMgr->getBlock( ui32BlockAddr, 
					&m_pBlock, &m_pucBlock)))
				{
					goto Exit;
				}

				bNewBlock = TRUE;
			}

			// If this is a data only block, then we can get the local data size
			// from the header.

			if( m_bDataOnlyBlock)
			{
				f_assert( getBlockType( m_pucBlock) == F_BLK_TYPE_BT_DATA_ONLY);
				
				*ppucDataPtr = m_pucBlock + sizeofDOBlockHdr( m_pucBlock);
										  
				m_uiDataRemaining = m_uiBlockSize -
										  sizeofDOBlockHdr( m_pucBlock) -
										  getBytesAvail( m_pucBlock);
										  
				m_uiDataLength = m_uiDataRemaining;
				m_ui32CurBlockAddr = ui32BlockAddr;
			}
			else
			{
				FLMBYTE *				pucEntry;

				// In a BTREE block, we MUST ensure that the first entry is a
				// continuation of the previous entry in the previous block.
				
				if( getNumKeys( m_pucBlock) == 0)
				{
					rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
					goto Exit;
				}

				if( bNewBlock)
				{
					m_uiCurOffset = 0;
				}

				// Point to the first entry ...
				
				pucEntry = BtEntry( m_pucBlock, m_uiCurOffset);

				if( !checkContinuedEntry( pucKey, uiKeyLen, NULL, pucEntry,
					getBlockType( m_pucBlock)))
				{
					rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
					goto Exit;
				}

				m_uiDataRemaining = fbtGetEntryDataLength( pucEntry,
					(const FLMBYTE **)ppucDataPtr, NULL, NULL);
					
				m_uiDataLength = m_uiDataRemaining;

				if( bNewBlock)
				{
					m_ui32CurBlockAddr = ui32BlockAddr;
				}
			}
			
			// Update the offset at the begining of the current entry.
			
			m_uiOffsetAtStart = m_uiOADataLength - m_uiOADataRemaining;
		}
	}

Exit:

	if( puiDataLen)
	{
		*puiDataLen = uiDataLen;
	}

	if( m_pBlock)
	{
		m_pBlock->Release();
		m_pBlock = NULL;
		m_pucBlock = NULL;
	}

	return( rc);
}

/***************************************************************************
Desc:	Method to prepare the Btree state for reading.  Since several APIs do
		the same thing, this has been put into a private method.
****************************************************************************/
RCODE F_BTree::setupReadState(
	FLMBYTE *			pucBlock,
	FLMBYTE *			pucEntry)
{
	RCODE					rc = NE_FLM_OK;
	IF_Block *			pDataBlock = NULL;
	FLMBYTE *			pucDataBlock = NULL;
	const FLMBYTE *	pucData;

	// Is there any data?  Check the block type.
	
	if( getBlockType( pucBlock) == F_BLK_TYPE_BT_LEAF_DATA)
	{
		// How large is the value for this entry?
		
		m_uiDataLength = fbtGetEntryDataLength( pucEntry, &pucData,
									&m_uiOADataLength, &m_bDataOnlyBlock);

		m_uiPrimaryDataLen = m_uiDataLength;
	}
	else
	{
		m_uiDataLength = 0;
		m_uiOADataLength = 0;
		m_bDataOnlyBlock = FALSE;
	}

  // Represents the offset at the beginning entry in the first block.  This
  // will change as we move through the blocks.
									
	m_uiOffsetAtStart = 0;
	
	// Track the overall length progress
  
	m_uiOADataRemaining = m_uiOADataLength;
	
	// Track the local entry progress
	
	m_uiDataRemaining = m_uiDataLength;

	if( m_bDataOnlyBlock)
	{
		m_ui32DOBlockAddr = bteGetBlockAddr( pucData);
		m_ui32CurBlockAddr = m_ui32DOBlockAddr;

		if( RC_BAD( rc = m_pBlockMgr->getBlock( m_ui32DOBlockAddr, 
			&pDataBlock, &pucDataBlock)))
		{
			goto Exit;
		}
		
		// Local amount of data in this block
		
		m_uiDataRemaining = m_uiBlockSize -
										sizeofDOBlockHdr( pucDataBlock) -
										getBytesAvail( pucDataBlock);

		// Keep the actual local data size for later.
		
		m_uiDataLength = m_uiDataRemaining;

		// Adjust for the key at the beginning of the first block.
		
		if( !getPrevInChain( pucDataBlock))
		{
			FLMBYTE *	pucPtr = pucDataBlock + sizeofDOBlockHdr( pucDataBlock);
			FLMUINT16	ui16KeyLen = FB2UW( pucPtr);

			m_uiDataLength -= (ui16KeyLen + 2);
			m_uiDataRemaining -= (ui16KeyLen + 2);
		}

		// Now release the DO Block.  We will get it again when we need it.
		
		pDataBlock->Release();
		pDataBlock = NULL;
		pucDataBlock = NULL;
	}

Exit:

	if( pDataBlock)
	{
		pDataBlock->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:	Method to remove extra entries after a replace operation.
****************************************************************************/
RCODE F_BTree::removeRemainingEntries(
	const FLMBYTE *		pucKey,
	FLMUINT					uiKeyLen)
{
	RCODE						rc = NE_FLM_OK;
	FLMBOOL					bLastElement = FALSE;
	FLMBYTE *				pucEntry;
	FLMBOOL					bFirst = TRUE;

	// We should never get to this function when in the upper levels.

	f_assert( m_pStack->uiLevel == 0);

	// If we do not have a stack setup yet (which can happen if the replace
	// is trying to shortcut to the previously known block address and offset),
	// then at this point, we must build the stack, since it may be required
	// to adjust the upper levels of the btree.

	if( !m_bStackSetup)
	{
		if( RC_BAD( rc = findEntry( pucKey, uiKeyLen, FLM_EXACT)))
		{
			goto Exit;
		}
	}

	while( !bLastElement)
	{
		// Begin each iteration at the leaf level.
		
		m_pStack = &m_Stack[ 0];

		// Advance the stack to the next entry.
		
		if (bFirst ||
				m_pStack->uiCurOffset >= getNumKeys( m_pStack->pucBlock))
		{
			if( RC_BAD( rc = moveStackToNext( NULL, NULL)))
			{
				goto Exit;
			}
		}

		bFirst = FALSE;
		pucEntry = BtEntry( m_pStack->pucBlock, m_pStack->uiCurOffset);

		if( !checkContinuedEntry( pucKey, uiKeyLen, &bLastElement,
					pucEntry, getBlockType( m_pStack->pucBlock)))
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
			goto Exit;
		}

		// Remove the entry from this block.
		
		if( RC_BAD( rc = remove( FALSE)))
		{
			goto Exit;
		}
		
		// Is the block empty now?  If it is, then we will want to remove this
		// block and remove the entry in the parent that points to this block.
		
		if( getNumKeys( m_pStack->pucBlock) == 0)
		{
			for (;;)
			{
				f_assert( !isRootBlock( m_pStack->pucBlock));

				// Remove this block, then update the parent.
				
				if( RC_BAD( rc = deleteEmptyBlock()))
				{
					goto Exit;
				}

				// Now update the parent blocks
				
				m_pStack++;

				if( RC_BAD( rc = remove( FALSE)))
				{
					goto Exit;
				}

				// Update the counts if keeping counts.
				
				if( m_bCounts && !isRootBlock( m_pStack->pucBlock))
				{
					if( RC_BAD( rc = updateCounts()))
					{
						goto Exit;
					}
				}

				if( getNumKeys( m_pStack->pucBlock) > 0)
				{
					break;
				}
			}

			// Rebuild the stack to the beginning after a delete block operation.
			
			if( RC_BAD( findEntry( pucKey, uiKeyLen, FLM_EXACT)))
			{
				goto Exit;
			}
			
			bFirst = TRUE;
		}
		else
		{
			// Update the counts if keeping counts.
			
			if( m_bCounts)
			{
				if( RC_BAD( rc = updateCounts()))
				{
					goto Exit;
				}
			}
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method to delete an empty block.  The block that will be deleted is
		the current block pointed to by m_pStack.
****************************************************************************/
RCODE F_BTree::deleteEmptyBlock( void)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT32			ui32PrevBlockAddr;
	FLMUINT32			ui32NextBlockAddr;
	IF_Block *			pBlock = NULL;
	FLMBYTE *			pucBlock = NULL;

	// Get the previous block address so we can back everything up in the stack

	ui32PrevBlockAddr = getPrevInChain( m_pStack->pucBlock);
	ui32NextBlockAddr = getNextInChain( m_pStack->pucBlock);

	// Free the block

	if( RC_BAD( rc = m_pBlockMgr->freeBlock( &m_pStack->pBlock, 
		&m_pStack->pucBlock)))
	{
		goto Exit;
	}
	
	// Update the previous block.

	if( ui32PrevBlockAddr)
	{
		if( RC_BAD( rc = m_pBlockMgr->getBlock( ui32PrevBlockAddr, 
			&pBlock, &pucBlock)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = m_pBlockMgr->prepareForUpdate( &pBlock, &pucBlock)))
		{
			goto Exit;
		}

		setNextInChain( pucBlock, ui32NextBlockAddr);
		
		pBlock->Release();
		pBlock = NULL;
		pucBlock = NULL;
	}

	// Update the next block

	if( ui32NextBlockAddr)
	{
		if( RC_BAD( rc = m_pBlockMgr->getBlock( ui32NextBlockAddr, 
			&pBlock, &pucBlock)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = m_pBlockMgr->prepareForUpdate( &pBlock, &pucBlock)))
		{
			goto Exit;
		}

		setPrevInChain( pucBlock, ui32PrevBlockAddr);

		pBlock->Release();
		pBlock = NULL;
		pucBlock = NULL;
	}

Exit:

	if( pBlock)
	{
		pBlock->Release();
	}
	
	return( rc);
}

/***************************************************************************
Desc:	Method to remove (free) all data only blocks that are linked to the
		data only block whose address is passed in (inclusive).
****************************************************************************/
RCODE F_BTree::removeDOBlocks(
	FLMUINT32			ui32BlockAddr)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT32			ui32NextBlockAddr;
	IF_Block *			pBlock = NULL;
	FLMBYTE *			pucBlock = NULL;

	ui32NextBlockAddr = ui32BlockAddr;

	while( ui32NextBlockAddr)
	{
		if( RC_BAD( rc = m_pBlockMgr->getBlock( ui32NextBlockAddr, 
			&pBlock, &pucBlock)))
		{
			goto Exit;
		}
		
		f_assert( getBlockType( pucBlock) == F_BLK_TYPE_BT_DATA_ONLY);
		ui32NextBlockAddr = getNextInChain( pucBlock);

		if( RC_BAD( rc = m_pBlockMgr->freeBlock( &pBlock, &pucBlock)))
		{
			goto Exit;
		}
	}

Exit:

	if( pBlock)
	{
		pBlock->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:	Method used to replace entries where the original spans multiple
		elements and we are NOT to truncate it.  To do this, we will attempt
		to fill each block until we have stored everything.
****************************************************************************/
RCODE F_BTree::replaceMultiples(
	const FLMBYTE **		ppucKey,
	FLMUINT *				puiKeyLen,
	const FLMBYTE *		pucDataValue,
	FLMUINT					uiLen,
	const FLMBYTE **		ppucRemainingValue,
	FLMUINT *				puiRemainingLen,
	F_ELM_UPD_ACTION *	peAction)
{
	RCODE						rc = NE_FLM_OK;
	FLMBOOL					bLastElement = FALSE;
	FLMUINT					uiRemainingData = uiLen;
	const FLMBYTE *		pucRemainingValue = pucDataValue;
	FLMBYTE *				pucEntry = NULL;
	FLMBYTE *				pucData;
	FLMUINT					uiDataLength;
	FLMUINT					uiOADataLength = uiLen;
	FLMUINT					uiOldOADataLength;
	FLMUINT					uiAmtCopied;

	// Must be at the leaf level!

	f_assert( m_pStack->uiLevel == 0);

	while( uiRemainingData)
	{
		if( RC_BAD( rc = m_pBlockMgr->prepareForUpdate( &m_pStack->pBlock,
			&m_pStack->pucBlock)))
		{
			goto Exit;
		}

		m_pStack->pui16OffsetArray = BtOffsetArray( m_pStack->pucBlock, 0);

		// Get a pointer to the current entry
		
		pucEntry = BtEntry( m_pStack->pucBlock, m_pStack->uiCurOffset);

		// Determine the data size for this entry
		
		uiDataLength = fbtGetEntryDataLength( pucEntry, (const FLMBYTE **)&pucData,
								&uiOldOADataLength, NULL);

		// Now over-write as much of the data as we can
		
		if( uiRemainingData >= uiDataLength)
		{
			f_memcpy( pucData, pucRemainingValue, uiDataLength);

			uiAmtCopied = uiDataLength;
			pucRemainingValue += uiDataLength;
			uiRemainingData -= uiDataLength;
		}
		else
		{
			f_memcpy( pucData, pucRemainingValue, uiRemainingData);
			uiAmtCopied = uiRemainingData;
			pucRemainingValue += uiRemainingData;
			uiRemainingData = 0;
		}

		// Do we need to adjust the data length?
		
		if( uiDataLength > uiAmtCopied)
		{
			FLMBYTE *	pucTmp = pucEntry;
			
			// Skip the flag
			
			pucTmp++;
			
			if( bteKeyLenFlag( pucEntry))
			{
				pucTmp += 2;
			}
			else
			{
				pucTmp++;
			}

			if( bteDataLenFlag( pucEntry))
			{
				UW2FBA( (FLMUINT16)uiAmtCopied, pucTmp);
				pucTmp += 2;
			}
			else
			{
				*pucTmp = (FLMBYTE)uiAmtCopied;
				pucTmp++;
			}

			// We need to adjust the free space in the block too.

			incBytesAvail( m_pStack->pucBlock, uiDataLength - uiAmtCopied);


#ifdef FLM_DEBUG
			// Clear the unused portion of the block now.
	
			pucTmp = pucData + uiAmtCopied;
			f_memset( pucTmp, 0, (uiDataLength - uiAmtCopied));
#endif
		}

		// Adjust the OA Data length if needed.  We only need to worry about this
		// on the first element.  No others have it.

		if( bteFirstElementFlag( pucEntry) && uiOADataLength != uiOldOADataLength)
		{
			FLMBYTE *		pucTmp = pucEntry;

			f_assert( bteOADataLenFlag( pucEntry));

			pucTmp++;
			
			if( bteKeyLenFlag( pucEntry))
			{
				pucTmp += 2;
			}
			else
			{
				pucTmp++;
			}

			if( bteDataLenFlag( pucEntry))
			{
				pucTmp += 2;
			}
			else
			{
				pucTmp++;
			}

			UD2FBA( (FLMUINT32)uiOADataLength, pucTmp);
		}

		// If we just updated the last member of this entry so break out.
		
		if( uiRemainingData == 0)
		{
			break;
		}

		// Was this the last element for this entry?
		
		if( bteLastElementFlag(pucEntry))
		{
			FLMBYTE *	pucTmp = pucEntry;

			// Turn off the lastElement flag on this entry.
			
			*pucTmp &= ~BTE_FLAG_LAST_ELEMENT;

			// No more to replace, the rest is going to be new data.
			
			*ppucRemainingValue = pucRemainingValue;
			*puiRemainingLen = uiRemainingData;
			break;
		}

		// Advance to the next entry, this block or the next...
		// The function expects to find the block in m_pBlock, so
		// let's put it there for now.

		if( RC_BAD( rc = moveStackToNext( NULL, NULL)))
		{
			goto Exit;
		}

		pucEntry = BtEntry( m_pStack->pucBlock, m_pStack->uiCurOffset);

		// Make sure we are still looking at the same key etc.
		
		if( !checkContinuedEntry( *ppucKey, *puiKeyLen, &bLastElement,
			pucEntry, getBlockType( m_pStack->pucBlock)))
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
			goto Exit;
		}
	}

	// Are there any more entries to remove?
	
	if( !bteLastElementFlag( pucEntry) && !uiRemainingData)
	{
		*pucEntry |= BTE_FLAG_LAST_ELEMENT;
		
		if( RC_BAD( rc = removeRemainingEntries( *ppucKey, *puiKeyLen)))
		{
			goto Exit;
		}
	}

	*peAction = ELM_DONE;

Exit:

	if( m_pBlock)
	{
		m_pBlock->Release();
		m_pBlock = NULL;
		m_pucBlock = NULL;
	}
	
	return( rc);
}

/***************************************************************************
Desc:	Method used to replace entries where the original spans multiple
		elements and we are not to truncate it.  To do this, we will attempt
		to fill each block until we have stored everything.
****************************************************************************/
RCODE F_BTree::replaceMultiNoTruncate(
	const FLMBYTE **		ppucKey,
	FLMUINT *				puiKeyLen,
	const FLMBYTE *		pucDataValue,
	FLMUINT					uiLen,
	const FLMBYTE **		ppucRemainingValue,
	FLMUINT *				puiRemainingLen,
	F_ELM_UPD_ACTION *	peAction)
{
	RCODE						rc = NE_FLM_OK;
	FLMBOOL					bLastElement = FALSE;
	FLMUINT					uiRemainingData = uiLen;
	const FLMBYTE *		pucRemainingValue = pucDataValue;
	FLMBYTE *				pucEntry;
	FLMBYTE *				pucData;
	FLMUINT					uiDataLength;

	// Must be at the leaf level
	
	f_assert( m_pStack->uiLevel == 0);

	while( uiRemainingData)
	{
		if( RC_BAD( rc = m_pBlockMgr->prepareForUpdate( &m_pStack->pBlock,
			&m_pStack->pucBlock)))
		{
			goto Exit;
		}

		m_pStack->pui16OffsetArray = BtOffsetArray( m_pStack->pucBlock, 0);

		// Get a pointer to the current entry
		
		pucEntry = BtEntry( m_pStack->pucBlock, m_pStack->uiCurOffset);

		// Determine the data size for this entry
		
		uiDataLength = fbtGetEntryDataLength( pucEntry, 
							(const FLMBYTE **)&pucData, NULL, NULL);

		// Now over-write as much of the data as we can.
		
		if( uiRemainingData > uiDataLength)
		{
			f_memcpy( pucData, pucRemainingValue, uiDataLength);
			pucRemainingValue += uiDataLength;
			uiRemainingData -= uiDataLength;
		}
		else
		{
			f_memcpy( pucData, pucRemainingValue, uiRemainingData);
			pucRemainingValue += uiRemainingData;
			uiRemainingData = 0;
		}

		// We just updated the last member of this entry so break out.
		
		if( uiRemainingData == 0)
		{
			break;
		}

		// Was this the last element for this entry?
		
		if( bteLastElementFlag( pucEntry))
		{
			// No more to replace, the rest is going to be new data.
			
			*ppucRemainingValue = pucRemainingValue;
			*puiRemainingLen = uiRemainingData;
			break;
		}

		// Advance to the next entry, this block or the next...
		// The function expects to find the block in m_pBlock, so
		// let's put it there for now.

		if( RC_BAD( rc = moveStackToNext( NULL, NULL)))
		{
			goto Exit;
		}

		pucEntry = BtEntry( m_pStack->pucBlock, m_pStack->uiCurOffset);

		// Make sure we are still looking at the same key etc.
		
		if( !checkContinuedEntry( *ppucKey, *puiKeyLen, &bLastElement,
			pucEntry, getBlockType( m_pStack->pucBlock)))
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
			goto Exit;
		}
	}

	*peAction = ELM_DONE;

Exit:

	if( m_pBlock)
	{
		m_pBlock->Release();
		m_pBlock = NULL;
		m_pucBlock = NULL;
	}
	
	return( rc);
}

/***************************************************************************
Desc:	Private method to retrieve the next block in the chain relative to
		the block that is passed in.  The block that is passed in is always
		released prior to getting the next block.
****************************************************************************/
RCODE F_BTree::getNextBlock(
	IF_Block **		ppBlock,
	FLMBYTE **		ppucBlock)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT32		ui32BlockAddr;

	ui32BlockAddr = getNextInChain( *ppucBlock);

	(*ppBlock)->Release();
	*ppBlock = NULL;
	*ppucBlock = NULL;
	
	if( !ui32BlockAddr)
	{
		rc = RC_SET( NE_FLM_EOF_HIT);
		goto Exit;
	}

	if( RC_BAD( rc = m_pBlockMgr->getBlock( ui32BlockAddr, ppBlock, ppucBlock)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Private method to retrieve the previous block in the chain relative to
		the  block that is passed in.  The block that is passed in is always
		released prior to getting the previous block.
****************************************************************************/
RCODE F_BTree::getPrevBlock(
	IF_Block **		ppBlock,
	FLMBYTE **		ppucBlock)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT32		ui32BlockAddr;

	ui32BlockAddr = getPrevInChain( *ppucBlock);

	(*ppBlock)->Release();
	*ppBlock = NULL;
	*ppucBlock = NULL;
	
	if( !ui32BlockAddr)
	{
		rc = RC_SET( NE_FLM_BOF_HIT);
		goto Exit;
	}

	if( RC_BAD( rc = m_pBlockMgr->getBlock( ui32BlockAddr, ppBlock, ppucBlock)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Private method to verify that the entry we are looking at in the stack
		is a continuation entry.  The key must match the key we pass in and
		the entry must be marked as a continuation, i.e. not the first
		element.
****************************************************************************/
FLMBOOL F_BTree::checkContinuedEntry(
	const FLMBYTE *	pucKey,
	FLMUINT				uiKeyLen,
	FLMBOOL *			pbLastElement,
	FLMBYTE *			pucEntry,
	FLMUINT				uiBlockType)
{
	FLMBOOL				bOk = TRUE;
	FLMUINT				uiBlockKeyLen;
	const FLMBYTE *	pucBlockKey;

	if( pbLastElement)
	{
		*pbLastElement = bteLastElementFlag( pucEntry);
	}

	uiBlockKeyLen = getEntryKeyLength( pucEntry, uiBlockType, &pucBlockKey);

	// Must be the same size key!
	
	if( uiKeyLen != uiBlockKeyLen)
	{
		bOk = FALSE;
		goto Exit;
	}

	// Must be identical!
	
	if( f_memcmp( pucKey, pucBlockKey, uiKeyLen) != 0)
	{
		bOk = FALSE;
		goto Exit;
	}
		
	// Must not be the first element!
	
	if( bteFirstElementFlag( pucEntry))
	{
		bOk = FALSE;
		goto Exit;
	}

Exit:

	return( bOk);
}

/***************************************************************************
Desc:	Private method to assend the tree, updating the counts for a
		particular block.  This method allows us to update the counts quickly
		without the need to continually loop, replacing existing keys with 
		new counts.
****************************************************************************/
RCODE F_BTree::updateCounts( void)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiLevel;

	for( uiLevel = m_pStack->uiLevel;
		  uiLevel < m_uiStackLevels - 1;
		  uiLevel++)
	{
		if( RC_BAD( rc = updateParentCounts(
			m_Stack[ uiLevel].pucBlock,
			&m_Stack[ uiLevel + 1].pBlock, &m_Stack[ uiLevel + 1].pucBlock,
			m_Stack[ uiLevel + 1].uiCurOffset)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Private method to store part of an entry in a block.  This method will
		determine how much of the data can be stored in the block.  The amount
		that does not get stored will be returned in ppucRemainingValue and
		puiRemainingLen.
****************************************************************************/
RCODE F_BTree::storePartialEntry(
	const FLMBYTE *		pucKey,
	FLMUINT					uiKeyLen,
	const FLMBYTE *		pucValue,
	FLMUINT					uiLen,
	FLMUINT					uiFlags,
	FLMUINT					uiChildBlockAddr,
	FLMUINT 					uiCounts,
	const FLMBYTE **		ppucRemainingValue,
	FLMUINT *				puiRemainingLen,
	FLMBOOL					bNewBlock)
{
	RCODE						rc = NE_FLM_OK;
	FLMUINT					uiNewDataLen;
	FLMUINT					uiOADataLen = 0;
	FLMUINT					uiEntrySize;
	FLMBOOL					bHaveRoom;
	FLMBOOL					bDefragBlock;
	FLMBOOL					bLastEntry;

	if( RC_BAD( rc = calcOptimalDataLength( uiKeyLen,
				uiLen, getBytesAvail( m_pStack->pucBlock), &uiNewDataLen)))
	{
		goto Exit;
	}

	if( uiNewDataLen < uiLen)
	{
		// Turn off the last element flag.
		
		uiFlags &= ~BTE_FLAG_LAST_ELEMENT;
		
		if( uiFlags & BTE_FLAG_FIRST_ELEMENT)
		{
			// Store the overall data length from this point forward.
			
			uiOADataLen = uiLen;
		}
	}

	if( RC_BAD( rc = calcNewEntrySize( uiKeyLen, uiNewDataLen, &uiEntrySize,
		&bHaveRoom, &bDefragBlock)))
	{
		goto Exit;
	}

	// We will defragment the block first if the avail and heap
	// are not the same size.

	if( getHeapSize( m_pStack->pucBlock) != getBytesAvail( m_pStack->pucBlock))
	{
		if( RC_BAD( rc = defragmentBlock( &m_pStack->pBlock, 
			&m_pStack->pucBlock)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = storeEntry( pucKey, uiKeyLen, pucValue, uiNewDataLen,
		uiFlags, uiOADataLen, uiChildBlockAddr, uiCounts, 
		uiEntrySize, &bLastEntry)))
	{
		goto Exit;
	}

	// If this block has a parent block, and the btree is maintaining counts
	// we will want to update the counts on the parent block.

	if( !isRootBlock( m_pStack->pucBlock) && m_bCounts && !bNewBlock)
	{
		if( RC_BAD( rc = updateCounts()))
		{
			goto Exit;
		}
	}

	if( uiNewDataLen < uiLen)
	{
		// Save the portion of the data that was not written.
		// It will be written later.
		
		*ppucRemainingValue = pucValue + uiNewDataLen;
		*puiRemainingLen = uiLen - uiNewDataLen;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Private meethod for checking the down links in the btree to make sure
		they are not corrupt.
****************************************************************************/
RCODE F_BTree::checkDownLinks( void)
{
	RCODE					rc = NE_FLM_OK;
	IF_Block *			pParentBlock = NULL;
	FLMBYTE *			pucParentBlock = NULL;

	if( RC_BAD( rc = m_pBlockMgr->getBlock( m_ui32RootBlockAddr, 
		&pParentBlock, &pucParentBlock)))
	{
		goto Exit;
	}
	
	if( getBlockType( pucParentBlock) == F_BLK_TYPE_BT_NON_LEAF ||
		 (getBlockType( pucParentBlock) == F_BLK_TYPE_BT_NON_LEAF_COUNTS))
	{
		if( RC_BAD( rc = verifyChildLinks( pucParentBlock)))
		{
			goto Exit;
		}
	}

Exit:

	if( pParentBlock)
	{
		pParentBlock->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:	Private method (recursive) that checks the child links in the given
		blocks to ensure they are correct.
****************************************************************************/
RCODE F_BTree::verifyChildLinks(
	FLMBYTE *				pucParentBlock)
{
	RCODE						rc = NE_FLM_OK;
	FLMUINT					uiNumKeys;
	IF_Block *				pChildBlock = NULL;
	FLMBYTE *				pucChildBlock = NULL;
	FLMUINT					uiCurOffset;
	FLMBYTE *				pucEntry;
	FLMUINT32				ui32BlockAddr;
	const FLMBYTE *		pucParentKey;
	FLMBYTE *				pucChildEntry;
	const FLMBYTE *		pucChildKey;
	FLMUINT					uiParentKeyLen;
	FLMUINT					uiChildKeyLen;

	uiNumKeys = getNumKeys( pucParentBlock);

	for( uiCurOffset = 0; uiCurOffset < uiNumKeys; uiCurOffset++)
	{
		pucEntry = BtEntry( pucParentBlock, uiCurOffset);
		
		// Non-leaf nodes have children.
		
		ui32BlockAddr = bteGetBlockAddr( pucEntry);
		f_assert( ui32BlockAddr);

		if( RC_BAD( rc = m_pBlockMgr->getBlock( ui32BlockAddr, 
			&pChildBlock, &pucChildBlock)))
		{
			goto Exit;
		}
		
		// Get key from the parent entry and compare it to the
		// last key in the child block.
		
		uiParentKeyLen = getEntryKeyLength(
			pucEntry, getBlockType( pucParentBlock), &pucParentKey);

		// Get the last entry in the child block.
		
		pucChildEntry = BtLastEntry( pucChildBlock);

		uiChildKeyLen = getEntryKeyLength(
			pucChildEntry, getBlockType( pucChildBlock), &pucChildKey);

		if( uiParentKeyLen != uiChildKeyLen)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
			goto Exit;
		}

		if( f_memcmp( pucParentKey, pucChildKey, uiParentKeyLen) != 0)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
			goto Exit;
		}

		if( getBlockType( pucChildBlock) == F_BLK_TYPE_BT_NON_LEAF ||
			 getBlockType( pucChildBlock) == F_BLK_TYPE_BT_NON_LEAF_COUNTS)
		{
			if( RC_BAD( rc = verifyChildLinks( pucChildBlock)))
			{
				goto Exit;
			}
		}
		
		pChildBlock->Release();
		pChildBlock = NULL;
		pucChildBlock = NULL;
	}

Exit:

	if( pChildBlock)
	{
		pChildBlock->Release();
	}
	
	return( rc);
}

/***************************************************************************
Desc:	This is a private method that computes the number of entries (keys)
		and the number of blocks between two points in the Btree.
****************************************************************************/
RCODE F_BTree::computeCounts(
	F_BTSK *			pFromStack,
	F_BTSK *			pUntilStack,
	FLMUINT *		puiBlockCount,
	FLMUINT *		puiKeyCount,
	FLMBOOL *		pbTotalsEstimated,
	FLMUINT			uiAvgBlockFullness)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiTotalKeys = 0;
	FLMUINT			uiTempKeyCount = 0;
	FLMUINT			uiEstKeyCount = 0;
	FLMUINT			uiTotalBlocksBetween = 0;
	FLMUINT			uiEstBlocksBetween = 0;

	uiTotalBlocksBetween = 0;
	*pbTotalsEstimated = FALSE;

	// The stack that we are looking at does not hold the blocks
	// we need. We first need to restore the blocks as needed.

	if( RC_BAD( rc = getBlocks( pFromStack, pUntilStack)))
	{
		goto Exit;
	}

	// Are the from and until positions in the same block?

	if( pFromStack->ui32BlockAddr == pUntilStack->ui32BlockAddr)
	{
		rc = blockCounts( pFromStack, pFromStack->uiCurOffset,
			pUntilStack->uiCurOffset, &uiTotalKeys, NULL);
		goto Exit;
	}

	// Are we maintaining counts on this Btree?  If so, we can just
	// use the counts we have...  The blocks count may still be estimated.

	if( m_bCounts)
	{
		return( getStoredCounts( pFromStack, pUntilStack, puiBlockCount,
			puiKeyCount, pbTotalsEstimated, uiAvgBlockFullness));
	}

	// Since we are not keeping counts on this Btree, we will need to
	// count them and possibly estimate them.

	// Gather the counts in the from and until leaf blocks.

	if( RC_BAD( rc = blockCounts( pFromStack, pFromStack->uiCurOffset,
			getNumKeys( pFromStack->pucBlock) - 1, &uiTotalKeys, NULL)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = blockCounts( pUntilStack, 0,
			pUntilStack->uiCurOffset, &uiTempKeyCount, NULL)))
	{
		goto Exit;
	}

	uiTotalKeys += uiTempKeyCount;

	// Do the obvious check to see if the blocks are neighbors.  If they
	// are, we are done.

	if( getNextInChain( pFromStack->pucBlock) == pUntilStack->ui32BlockAddr)
	{
		goto Exit;
	}

	// Estimate the number of elements in the parent block.

	*pbTotalsEstimated = TRUE;

	uiEstKeyCount = getAvgKeyCount( pFromStack, pUntilStack, uiAvgBlockFullness);
	uiEstBlocksBetween = 1;

	for (;;)
	{
		FLMUINT	uiElementCount;
		FLMUINT	uiTempElementCount;
		FLMUINT	uiEstElementCount;

		// Go up a b-tree level and check out how far apart the elements are.

		pFromStack++;
		pUntilStack++;

		if( RC_BAD( rc = getBlocks( pFromStack, pUntilStack)))
		{
			goto Exit;
		}

		// Share the same block?

		if( pFromStack->ui32BlockAddr == pUntilStack->ui32BlockAddr)
		{
			if( RC_BAD( rc = blockCounts( pFromStack, pFromStack->uiCurOffset,
					pUntilStack->uiCurOffset, NULL, &uiElementCount)))
			{
				goto Exit;
			}

			// Don't count the pFromStack or the pUntilStack current elements.

			uiElementCount -= 2;

			uiTotalBlocksBetween += uiEstBlocksBetween *
											(uiElementCount > 0 ? uiElementCount : 1);
			uiTotalKeys += uiEstKeyCount *
								(uiElementCount > 0 ? uiElementCount : 1);
			goto Exit;
		}

		// Gather the counts in the from and until non-leaf blocks.

		if( RC_BAD( rc = blockCounts( pFromStack, pFromStack->uiCurOffset,
				getNumKeys( pFromStack->pucBlock) - 1, NULL, &uiElementCount)))
		{
			goto Exit;
		}

		// Don't count the first element.

		uiElementCount--;

		if( RC_BAD( rc = blockCounts( pUntilStack, 0,
				pUntilStack->uiCurOffset, NULL, &uiTempElementCount)))
		{
			goto Exit;
		}

		uiElementCount += (uiTempElementCount - 1);

		uiTotalBlocksBetween += uiEstBlocksBetween * uiElementCount;
		uiTotalKeys += uiEstKeyCount * uiElementCount;

		// Do the obvious check to see if the blocks are neighbors.

		if( getNextInChain( pFromStack->pucBlock) == pUntilStack->ui32BlockAddr)
		{
			goto Exit;
		}

		// Recompute the estimated element count on every b-tree level
		// because the compression is better the lower in the b-tree we go.

		uiEstElementCount = getAvgKeyCount( pFromStack, pUntilStack, uiAvgBlockFullness);

		// Adjust the estimated key/ref count to be the counts from a complete
		// (not partial) block starting at this level going to the leaf.

		uiEstKeyCount *= uiEstElementCount;
		uiEstBlocksBetween *= uiEstElementCount;
	}

Exit:

	if( puiKeyCount)
	{
		*puiKeyCount = uiTotalKeys;
	}

	if( puiBlockCount)
	{
		*puiBlockCount = uiTotalBlocksBetween;
	}

	return( rc);
}

/***************************************************************************
Desc:	Private method to count the number of unique keys between two points.
		The count returned is inclusive of the first and last offsets.
****************************************************************************/
RCODE F_BTree::blockCounts(
	F_BTSK *			pStack,
	FLMUINT			uiFirstOffset,
	FLMUINT			uiLastOffset,
	FLMUINT *		puiKeyCount,
	FLMUINT *		puiElementCount)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiKeyCount;
	FLMUINT			uiElementCount;
	FLMBYTE *		pucEntry;

	// Debug checks.

	f_assert( uiFirstOffset <= uiLastOffset);
	f_assert( uiLastOffset <= (FLMUINT)(getNumKeys( pStack->pucBlock) - 1));

	uiKeyCount = uiElementCount = 0;

	// Loop gathering the statistics.

	while( uiFirstOffset <= uiLastOffset)
	{
		uiElementCount++;

		if( puiKeyCount)
		{
			pucEntry = BtEntry( pStack->pucBlock, uiFirstOffset);

			// We only have to worry about first key elements when we are at the
			// leaf level and we are keeping data at that level.

			if( pStack->uiLevel == 0 && m_bTreeHoldsData)
			{
				if( bteFirstElementFlag( pucEntry))
				{
					uiKeyCount++;
				}
			}
			else
			{
				uiKeyCount++;
			}
		}

		// Next element.

		if( uiFirstOffset == (FLMUINT)(getNumKeys( pStack->pucBlock) - 1))
		{
			break;
		}
		else
		{
			uiFirstOffset++;
		}
	}

	if( puiKeyCount)
	{
		*puiKeyCount = uiKeyCount;
	}

	if( puiElementCount)
	{
		*puiElementCount = uiElementCount;
	}

	return( rc);
}

/***************************************************************************
Desc:	Similar to computeCounts, except we use the stored counts.
****************************************************************************/
RCODE F_BTree::getStoredCounts(
	F_BTSK *			pFromStack,
	F_BTSK *			pUntilStack,
	FLMUINT *		puiBlockCount,
	FLMUINT *		puiKeyCount,
	FLMBOOL *		pbTotalsEstimated,
	FLMUINT			uiAvgBlockFullness)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiOmittedKeys;
	FLMUINT			uiTotalKeys;
	FLMUINT			uiEstBlocksBetween;
	FLMUINT			uiTotalBlocksBetween;

	*pbTotalsEstimated = FALSE;
	*puiBlockCount = 0;
	uiTotalBlocksBetween = 0;

	// Are these blocks adjacent?

	if( getNextInChain( pFromStack->pucBlock) == pUntilStack->ui32BlockAddr)
	{
		*puiKeyCount = (getNumKeys( pFromStack->pucBlock) - 
								pFromStack->uiCurOffset) + pUntilStack->uiCurOffset + 1;
		goto Exit;
	}

	*pbTotalsEstimated = TRUE;

	// How many keys are excluded in the From and Until blocks?

	uiOmittedKeys = countRangeOfKeys(
					pFromStack, 0, pFromStack->uiCurOffset) - 1;

	uiOmittedKeys += countRangeOfKeys(
					pUntilStack, pUntilStack->uiCurOffset,
					getNumKeys( pUntilStack->pucBlock) - 1) - 1;

	uiTotalKeys = 0;
	uiEstBlocksBetween = 1;

	for( ;;)
	{
		FLMUINT	uiElementCount;
		FLMUINT	uiTempElementCount;
		FLMUINT	uiEstElementCount;

		// Go up a b-tree level and check out how far apart the elements are.

		pFromStack++;
		pUntilStack++;

		if( RC_BAD( rc = getBlocks( pFromStack, pUntilStack)))
		{
			goto Exit;
		}

		// Share the same block?  We can get the actual key count now.

		if( pFromStack->ui32BlockAddr == pUntilStack->ui32BlockAddr)
		{

			if( RC_BAD( rc = blockCounts( pFromStack, pFromStack->uiCurOffset,
					pUntilStack->uiCurOffset, NULL, &uiElementCount)))
			{
				goto Exit;
			}

			// Don't count the pFromStack current element.

			uiElementCount -= 2;
			uiTotalBlocksBetween += uiEstBlocksBetween *
											(uiElementCount > 0 ? uiElementCount : 1);

			// Add one to the lasty offset to include the last entry in the count.
			
			uiTotalKeys = countRangeOfKeys(
				pFromStack, pFromStack->uiCurOffset, pUntilStack->uiCurOffset);

			*puiKeyCount = uiTotalKeys - uiOmittedKeys;
			*puiBlockCount = uiTotalBlocksBetween;
			goto Exit;
		}

		// How many to exclude from the From & Until blocks.

		if( pFromStack->uiCurOffset)
		{
			uiOmittedKeys += countRangeOfKeys(
				pFromStack, 0, pFromStack->uiCurOffset - 1);
		}

		uiOmittedKeys += countRangeOfKeys(
				pUntilStack,  pUntilStack->uiCurOffset + 1,
				getNumKeys( pUntilStack->pucBlock) - 1);

		// Gather the counts in the from and until non-leaf blocks.

		if( RC_BAD( rc = blockCounts( pFromStack, pFromStack->uiCurOffset,
				getNumKeys( pFromStack->pucBlock) - 1, NULL, &uiElementCount)))
		{
			goto Exit;
		}

		// Don't count the first element.

		uiElementCount--;

		if( RC_BAD( rc = blockCounts( pUntilStack, 0,
				pUntilStack->uiCurOffset, NULL, &uiTempElementCount)))
		{
			goto Exit;
		}

		uiElementCount += (uiTempElementCount - 1);
		uiTotalBlocksBetween += uiEstBlocksBetween * uiElementCount;

		// We are not going to check if these blocks are neighbors here because
		// we want to find the common parent.  That will tell us what the actual
		// counts are at the leaf level.

		// Recompute the estimated element count on every b-tree level
		// because the compression is better the lower in the b-tree we go.

		uiEstElementCount = getAvgKeyCount( 
										pFromStack, pUntilStack, uiAvgBlockFullness);
		uiEstBlocksBetween *= uiEstElementCount;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Retrieve the blocks identified in the two stack entries.  Used in
		computing counts (btComputeCounts etc.)
****************************************************************************/
RCODE F_BTree::getBlocks(
	F_BTSK *			pStack1,
	F_BTSK *			pStack2)
{
	RCODE				rc = NE_FLM_OK;

	// If these blocks are at the root level, we must ensure that we retrieve
	// the root block.  The root block can potentially change address, so
	// we wil reset it here to be sure.

	if( pStack1->uiLevel == m_uiRootLevel)
	{
		pStack1->ui32BlockAddr = m_ui32RootBlockAddr;
	}

	if( pStack2->uiLevel == m_uiRootLevel)
	{
		pStack2->ui32BlockAddr = m_ui32RootBlockAddr;
	}

	if( RC_BAD( m_pBlockMgr->getBlock( pStack1->ui32BlockAddr, 
		&pStack1->pBlock, &pStack1->pucBlock)))
	{
		goto Exit;
	}
	
	if( RC_BAD( m_pBlockMgr->getBlock( pStack2->ui32BlockAddr,
		&pStack2->pBlock, &pStack2->pucBlock)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method to tally the counts in a block between (inclusive) the
		uiFromOffset & uiUntilOffset parameters.
****************************************************************************/
FLMUINT F_BTree::countRangeOfKeys(
	F_BTSK *			pFromStack,
	FLMUINT			uiFromOffset,
	FLMUINT			uiUntilOffset)
{
	FLMUINT			uiCount = 0;
	FLMUINT			uiLoop = uiFromOffset;
	FLMBYTE *		pucEntry;
	FLMUINT			uiBlockType;

	uiBlockType = getBlockType( pFromStack->pucBlock);

	if( uiBlockType == F_BLK_TYPE_BT_NON_LEAF_COUNTS)
	{
		while( uiLoop < uiUntilOffset)
		{
			pucEntry = BtEntry( pFromStack->pucBlock, uiLoop);
			pucEntry += 4;
			uiCount += FB2UD( pucEntry);
			uiLoop++;
		}
	}
	else
	{
		uiCount = uiUntilOffset;
	}

	return( uiCount);
}

/***************************************************************************
Desc:	Method to estimate the averge number of keys, based on the anticipated
		average block usage (passed in) and the actual block usage.
****************************************************************************/
FLMUINT F_BTree::getAvgKeyCount(
	F_BTSK *			pFromStack,
	F_BTSK *			pUntilStack,
	FLMUINT			uiAvgBlockFullness)
{
	FLMUINT			uiFromUsed;
	FLMUINT			uiUntilUsed;
	FLMUINT			uiTotalUsed;
	FLMUINT			uiFromKeys;
	FLMUINT			uiUntilKeys;
	FLMUINT			uiTotalKeys;

	uiFromUsed = m_uiBlockSize - getBytesAvail( pFromStack->pucBlock);
	uiUntilUsed = m_uiBlockSize - getBytesAvail( pUntilStack->pucBlock);

	uiTotalUsed = uiFromUsed + uiUntilUsed;

	uiFromKeys = getNumKeys( pFromStack->pucBlock);
	uiUntilKeys = getNumKeys( pUntilStack->pucBlock);
	uiTotalKeys = uiFromKeys + uiUntilKeys;

	return( (uiAvgBlockFullness * uiTotalKeys) / uiTotalUsed);
}

/***************************************************************************
Desc:	Method to test if two blocks can be merged together to make a single
		block.  This is done only after a remove operation and is intended to
		try to consolidate space as much as possible.  If we can consolidate
		two blocks, we will do it, then update the tree.
****************************************************************************/
RCODE F_BTree::mergeBlocks(
	FLMBOOL						bLastEntry,
	FLMBOOL *					pbMergedWithPrev,
	FLMBOOL *					pbMergedWithNext,
	F_ELM_UPD_ACTION *		peAction)
{
	RCODE							rc = NE_FLM_OK;
	IF_Block *					pPrevBlock = NULL;
	IF_Block *					pNextBlock = NULL;
	FLMBYTE *					pucPrevBlock = NULL;
	FLMBYTE *					pucNextBlock = NULL;
	FLMUINT32					ui32PrevBlockAddr;
	FLMUINT32					ui32NextBlockAddr;

	*pbMergedWithPrev = FALSE;
	*pbMergedWithNext = FALSE;

	// Our first check is to see if we can merge the current block with its
	// previous block.

	ui32PrevBlockAddr = getPrevInChain( m_pStack->pucBlock);
	if( ui32PrevBlockAddr)
	{
		// Get the block.

		if( RC_BAD( rc = m_pBlockMgr->getBlock( ui32PrevBlockAddr,
			&pPrevBlock, &pucPrevBlock)))
		{
			goto Exit;
		}

		// Is there room to merge?

		if( (FLMUINT)(getBytesAvail( pucPrevBlock) + 
							getBytesAvail( m_pStack->pucBlock)) >=
			 (FLMUINT)(m_uiBlockSize - sizeofBTreeBlockHdr( m_pStack->pucBlock)))
		{
			// Looks like we can merge these two.  We will move the content
			// of the previous block into this one.

			if( RC_BAD( rc = merge( &pPrevBlock, &pucPrevBlock, 
				&m_pStack->pBlock, &m_pStack->pucBlock)))
			{
				goto Exit;
			}

			// Update the counts for the current block before releasing it.

			if( m_bCounts)
			{
				if( RC_BAD( rc = updateCounts()))
				{
					goto Exit;
				}
			}

			if( bLastEntry)
			{
				// Need to save the replace information for the last entry in
				// the block before we move to the previous block.  This will
				// allow us to do the replace later.

				FLMBYTE *			pucEntry;
				const FLMBYTE *	pucKey;
				FLMUINT				uiKeyLen;

				pucEntry = BtEntry( m_pStack->pucBlock, 
									getNumKeys( m_pStack->pucBlock) - 1);
					
				uiKeyLen = getEntryKeyLength(
					pucEntry, getBlockType( m_pStack->pucBlock), &pucKey);

				if( RC_BAD( rc = saveReplaceInfo( pucKey, uiKeyLen)))
				{
					goto Exit;
				}
			}

			// Move the stack to the previous entry

			if( RC_BAD( rc = moveStackToPrev( pPrevBlock, pucPrevBlock)))
			{
				goto Exit;
			}
			
			pPrevBlock->Release();
			pPrevBlock = NULL;
			pucPrevBlock = NULL;

			f_assert( getNumKeys( m_pStack->pucBlock) == 0);

			// Free the empty block.

			if( RC_BAD( rc = deleteEmptyBlock()))
			{
				goto Exit;
			}

			// Now we want to remove the parent entry for the block that was
			// freed.

			m_pStack++;
			*peAction = ELM_REMOVE;
			*pbMergedWithPrev = TRUE;
			
			goto Exit;
		}
		else
		{
			// No room here so release the block.

			pPrevBlock->Release();
			pPrevBlock = NULL;
			pucPrevBlock = NULL;
		}
	}

	// Can we merge with the next block?

	ui32NextBlockAddr = getNextInChain( m_pStack->pucBlock);
	if( ui32NextBlockAddr)
	{
		// Get the block.

		if( RC_BAD( rc = m_pBlockMgr->getBlock( ui32NextBlockAddr, 
			&pNextBlock, &pucNextBlock)))
		{
			goto Exit;
		}
		
		// Is there room to merge?

		if( (FLMUINT)(getBytesAvail( pucNextBlock) + 
						  getBytesAvail( m_pStack->pucBlock)) >=
			 (FLMUINT)(m_uiBlockSize - sizeofBTreeBlockHdr( m_pStack->pucBlock)))
		{
			// Looks like we can merge these two.

			if( RC_BAD( rc = merge( &m_pStack->pBlock, &m_pStack->pucBlock,
				&pNextBlock, &pucNextBlock)))
			{
				goto Exit;
			}

			// Update the counts for the current block and the next block.

			if( m_bCounts)
			{
				pPrevBlock = m_pStack->pBlock;
				pucPrevBlock = m_pStack->pucBlock;
				pPrevBlock->AddRef();

				// Need to move the stack to the next entry.  Don't let the current
				// block get released because we still need it.

				if( RC_BAD( rc = moveStackToNext( pNextBlock, pucNextBlock)))
				{
					goto Exit;
				}
				
				pNextBlock->Release();
				pNextBlock = NULL;
				pucNextBlock = NULL;

				if( RC_BAD( rc = updateCounts()))
				{
					goto Exit;
				}

				// Move back to the original stack again.  It's okay to release the
				// now current block.

				if( RC_BAD( rc = moveStackToPrev( pPrevBlock, pucPrevBlock)))
				{
					goto Exit;
				}
				
				pPrevBlock->Release();
				pPrevBlock = NULL;
				pucPrevBlock = NULL;
			}

			f_assert( getNumKeys( m_pStack->pucBlock) == 0);

			// Free the empty block.

			if( RC_BAD( rc = deleteEmptyBlock()))
			{
				goto Exit;
			}

			// Now we want to remove the parent entry for the block that was freed.

			m_pStack++;
			*peAction = ELM_REMOVE;
			*pbMergedWithNext = TRUE;
			goto Exit;
		}
		else
		{
			// No room here so release the block.

			pNextBlock->Release();
			pNextBlock = NULL;
			pucNextBlock = NULL;
		}
	}

Exit:

	if( pPrevBlock)
	{
		pPrevBlock->Release();
	}

	if( pNextBlock)
	{
		pNextBlock->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:	Method to move the contents of ppFromBlock into ppToBlock.  
		Note that all merges are a move to next operation.
****************************************************************************/
RCODE F_BTree::merge(
	IF_Block **		ppFromBlock,
	FLMBYTE **		ppucFromBlock,
	IF_Block **		ppToBlock,
	FLMBYTE **		ppucToBlock)
{
	RCODE					rc = NE_FLM_OK;
	F_BTSK				tempStack;
	F_BTSK *				pStack = NULL;
	IF_Block *			pBlock = NULL;
	FLMBYTE *			pucBlock = NULL;

	// May need to defragment the blocks first.

	if( getBytesAvail( *ppucToBlock) != getHeapSize( *ppucToBlock))
	{
		if( RC_BAD( rc = defragmentBlock( ppToBlock, ppucToBlock)))
		{
			goto Exit;
		}
	}

	// Make a temporary stack entry so we can "fool" the moveToNext
	// function into moving the entries for us.

	pBlock = *ppFromBlock;
	pucBlock = *ppucFromBlock;
	
	*ppFromBlock = NULL;
	*ppucFromBlock = NULL;
	
	tempStack.ui32BlockAddr = getBlockAddr( pucBlock);
	tempStack.pBlock = pBlock;
	tempStack.pucBlock = pucBlock;
	tempStack.uiCurOffset = 0;
	tempStack.uiLevel = m_pStack->uiLevel;
	tempStack.pui16OffsetArray = BtOffsetArray( pucBlock, 0);

	// Save the current m_pStack.

	pStack = m_pStack;
	m_pStack = &tempStack;

	// Now do the move

	if( RC_BAD( rc = moveToNext( getNumKeys( tempStack.pucBlock) - 1,
		0, ppToBlock, ppucToBlock)))
	{
		goto Exit;
	}

	*ppFromBlock = tempStack.pBlock;
	*ppucFromBlock = tempStack.pucBlock;
	
	tempStack.pBlock = NULL;
	tempStack.pucBlock = NULL;

Exit:

	m_pStack = pStack;
	return( rc);
}

/***************************************************************************
Desc:	Method to test if the src and dst entries can be combined into one
		entry.
****************************************************************************/
RCODE F_BTree::combineEntries(
	FLMBYTE *			pucSrcBlock,
	FLMUINT				uiSrcOffset,
	FLMBYTE *			pucDstBlock,
	FLMUINT				uiDstOffset,
	FLMBOOL *			pbEntriesCombined,
	FLMUINT *			puiEntrySize,
	FLMBYTE *			pucTmpBlock)
{
	RCODE					rc = NE_FLM_OK;
	FLMBYTE *			pucSrcEntry;
	FLMBYTE *			pucDstEntry;
	FLMUINT				uiSrcKeyLen;
	FLMUINT				uiDstKeyLen;
	const FLMBYTE *	pucSrcKey;
	const FLMBYTE *	pucDstKey;
	FLMUINT				uiFlags = 0;
	FLMBYTE *			pucTmp;
 	FLMUINT				uiSrcOADataLen;
 	FLMUINT				uiDstOADataLen;
	const FLMBYTE *	pucSrcData;
	const FLMBYTE *	pucDstData;
	FLMUINT				uiSrcDataLen;
	FLMUINT				uiDstDataLen;
	FLMUINT				uiEntrySize;

	*pbEntriesCombined = FALSE;
	*puiEntrySize = 0;

	if( getNumKeys( pucDstBlock) == 0)
	{
		goto Exit;
	}

	if( getNumKeys( pucSrcBlock) == 0)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
		goto Exit;
	}

	if( getBlockType( pucSrcBlock) != F_BLK_TYPE_BT_LEAF_DATA)
	{
		goto Exit;
	}

	pucSrcEntry = BtEntry( pucSrcBlock, uiSrcOffset);
	pucDstEntry = BtEntry( pucDstBlock, uiDstOffset);

	// Do we have the same key?

	uiSrcKeyLen = getEntryKeyLength( pucSrcEntry, 
							F_BLK_TYPE_BT_LEAF_DATA, &pucSrcKey);
							
	uiDstKeyLen = getEntryKeyLength( pucDstEntry, 
							F_BLK_TYPE_BT_LEAF_DATA, &pucDstKey);

	if( uiSrcKeyLen != uiDstKeyLen)
	{
		// Not the same key.
		
		goto Exit;
	}

	if( f_memcmp( pucSrcKey, pucDstKey, uiSrcKeyLen) != 0)
	{
		// Not the same key.
		
		goto Exit;
	}

	// They match, so we can combine them.

	pucTmp = &pucTmpBlock[ 1];		// Key length position
	uiFlags = (pucDstEntry[0] & (BTE_FLAG_FIRST_ELEMENT | BTE_FLAG_LAST_ELEMENT)) |
				 (pucSrcEntry[0] & (BTE_FLAG_FIRST_ELEMENT | BTE_FLAG_LAST_ELEMENT));
	uiEntrySize = 1;

	if( uiSrcKeyLen > ONE_BYTE_SIZE)
	{
		uiFlags |= BTE_FLAG_KEY_LEN;
		UW2FBA( (FLMUINT16)uiSrcKeyLen, pucTmp);
		pucTmp += 2;
		uiEntrySize += 2;
	}
	else
	{
		*pucTmp = (FLMBYTE)uiSrcKeyLen;
		pucTmp++;
		uiEntrySize++;
	}

	uiSrcDataLen = fbtGetEntryDataLength(
		pucSrcEntry, &pucSrcData, &uiSrcOADataLen, NULL);

	uiDstDataLen = fbtGetEntryDataLength(
		pucDstEntry, &pucDstData, &uiDstOADataLen, NULL);

	if( (uiSrcDataLen + uiDstDataLen) > ONE_BYTE_SIZE)
	{
		uiFlags |= BTE_FLAG_DATA_LEN;
		UW2FBA( (FLMUINT16)(uiSrcDataLen + uiDstDataLen), pucTmp);
		pucTmp += 2;
		uiEntrySize += 2;
	}
	else
	{
		*pucTmp = (FLMBYTE)(uiSrcDataLen + uiDstDataLen);
		pucTmp++;
		uiEntrySize++;
	}

	// Verify the OA Data length

	if( (*pucSrcEntry & BTE_FLAG_OA_DATA_LEN) &&
			(uiSrcOADataLen > (uiSrcDataLen + uiDstDataLen)))
	{
		uiFlags |= BTE_FLAG_OA_DATA_LEN;
		UD2FBA( (FLMUINT32)uiSrcOADataLen, pucTmp);
		pucTmp += 4;
		uiEntrySize += 4;
	}
	else if( (*pucDstEntry & BTE_FLAG_OA_DATA_LEN) &&
			(uiDstOADataLen > (uiSrcDataLen + uiDstDataLen)))
	{
		uiFlags |= BTE_FLAG_OA_DATA_LEN;
		UD2FBA( (FLMUINT32)uiDstOADataLen, pucTmp);
		pucTmp += 4;
		uiEntrySize += 4;
	}

	f_memcpy( pucTmp, pucSrcKey, uiSrcKeyLen);
	pucTmp += uiSrcKeyLen;
	uiEntrySize += uiSrcKeyLen;

	// Need to put the entry together in the right order.  If the Src block is
	// before the Dst block, then we will put down the Src data first.

	if( getNextInChain( pucSrcBlock) == getBlockAddr( pucDstBlock))
	{
		f_memcpy( pucTmp, pucSrcData, uiSrcDataLen);
		pucTmp += uiSrcDataLen;
		uiEntrySize += uiSrcDataLen;

		f_memcpy( pucTmp, pucDstData, uiDstDataLen);
		uiEntrySize += uiDstDataLen;
	}
	else
	{
		f_memcpy( pucTmp, pucDstData, uiDstDataLen);
		uiEntrySize += uiDstDataLen;
		pucTmp += uiDstDataLen;
		
		f_memcpy( pucTmp, pucSrcData, uiSrcDataLen);
		uiEntrySize += uiSrcDataLen;
	}

	pucTmpBlock[ 0] = (FLMBYTE)uiFlags;
	*puiEntrySize = uiEntrySize;
	*pbEntriesCombined = TRUE;

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method to move a block from one location to another.
****************************************************************************/
RCODE F_BTree::btMoveBlock(
	FLMUINT32			ui32FromBlockAddr,
	FLMUINT32			ui32ToBlockAddr)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT				uiType;

	if( !m_bOpened || m_bSetupForRead || m_bSetupForReplace ||
		  (m_bSetupForWrite))
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_BAD_STATE);
		goto Exit;
	}

	f_assert( m_uiSearchLevel >= F_BTREE_MAX_LEVELS);

	// Get the From block and retrieve the last key in the block.  Make note
	// of the level of the block.  We will need this to make sure we get the
	// right block.

	if( RC_BAD( rc = m_pBlockMgr->getBlock( ui32FromBlockAddr, 
		&m_pBlock, &m_pucBlock)))
	{
		goto Exit;
	}

	// Find out if this is a Btree block or a DO block.

	uiType = getBlockType( m_pucBlock);

	if( uiType == F_BLK_TYPE_FREE)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
		goto Exit;
	}

	if( uiType == F_BLK_TYPE_BT_DATA_ONLY)
	{
		if( RC_BAD( rc = moveDOBlock( ui32FromBlockAddr, ui32ToBlockAddr)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = moveBtreeBlock( ui32FromBlockAddr, ui32ToBlockAddr)))
		{
			goto Exit;
		}
	}

Exit:

	if( m_pBlock)
	{
		m_pBlock->Release();
		m_pBlock = NULL;
		m_pucBlock = NULL;
	}

	return( rc);
}

/***************************************************************************
Desc:	Move a Btree block from one address to another, updating its parent.
****************************************************************************/
RCODE F_BTree::moveBtreeBlock(
	FLMUINT32			ui32FromBlockAddr,
	FLMUINT32			ui32ToBlockAddr)
{
	RCODE						rc = NE_FLM_OK;
	FLMBYTE *				pucEntry;
	const FLMBYTE *		pucKeyRV = NULL;
	FLMBYTE *				pucKey = NULL;
	FLMUINT					uiBlockLevel;
	FLMBYTE *				pucSrc;
	FLMBYTE *				pucDest;
	IF_Block *				pTmpBlock = NULL;
	FLMBYTE *				pucTmpBlock = NULL;
	FLMUINT					uiKeyLen;

	f_assert( m_pBlock && m_pucBlock);

	uiBlockLevel = getBlockLevel( m_pucBlock);
	pucEntry = BtLastEntry( m_pucBlock);
	uiKeyLen = getEntryKeyLength( pucEntry, getBlockType( m_pucBlock), &pucKeyRV);

	if( RC_BAD( rc = f_calloc( uiKeyLen, &pucKey)))
	{
		goto Exit;
	}
	
	f_memcpy( pucKey, pucKeyRV, uiKeyLen);

	// Release the block and search for the key.
	
	m_pBlock->Release();
	m_pBlock = NULL;
	m_pucBlock = NULL;

	if( RC_BAD( rc = findEntry( pucKey, uiKeyLen, FLM_EXACT)))
	{
		// We must find it!
		
		RC_UNEXPECTED_ASSERT( rc);
		goto Exit;
	}

	// Verify that we found the right block.
	
	m_pStack = &m_Stack[ uiBlockLevel];

	if( ui32FromBlockAddr != m_pStack->ui32BlockAddr)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
		goto Exit;
	}

	if( RC_BAD( rc = m_pBlockMgr->prepareForUpdate( &m_pStack->pBlock,
		&m_pStack->pucBlock)))
	{
		goto Exit;
	}

	m_pStack->pui16OffsetArray = BtOffsetArray( m_pStack->pucBlock, 0);

	// Get the new block and verify that it is a free block.
	
	if( RC_BAD( rc = m_pBlockMgr->getBlock( ui32ToBlockAddr, 
		&m_pBlock, &m_pucBlock)))
	{
		goto Exit;
	}

	if( getBlockType( m_pucBlock) != F_BLK_TYPE_FREE)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
		goto Exit;
	}

	// Update the header of the new block to point to the prev and next
	// blocks etc ...
	
	if( RC_BAD( rc = m_pBlockMgr->prepareForUpdate( &m_pBlock, &m_pucBlock)))
	{
		goto Exit;
	}

	setPrevInChain( m_pucBlock, getPrevInChain( m_pStack->pucBlock));
	setNextInChain( m_pucBlock, getNextInChain( m_pStack->pucBlock));
	setBytesAvail( m_pucBlock, getBytesAvail( m_pStack->pucBlock));
	setBlockType( m_pucBlock, getBlockType( m_pStack->pucBlock));
	setBlockFlags( m_pucBlock, getBlockFlags( m_pStack->pucBlock));
	setBTreeId( m_pucBlock, getBTreeId( m_pStack->pucBlock));
	setNumKeys( m_pucBlock, getNumKeys( m_pStack->pucBlock));
	setBlockLevel( m_pucBlock, getBlockLevel( m_pStack->pucBlock));
	setBTreeFlags( m_pucBlock, getBTreeFlags( m_pStack->pucBlock));
	setHeapSize( m_pucBlock, getHeapSize( m_pStack->pucBlock));

	// Get the previous and next blocks and set their next and prev addresses.
	
	if( getPrevInChain( m_pStack->pucBlock))
	{
		if( RC_BAD( rc = m_pBlockMgr->getBlock(
			getPrevInChain( m_pStack->pucBlock), &pTmpBlock, &pucTmpBlock)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = m_pBlockMgr->prepareForUpdate( 
			&pTmpBlock, &pucTmpBlock)))
		{
			goto Exit;
		}
		
		f_assert( getNextInChain( pucTmpBlock) == ui32FromBlockAddr);
		setNextInChain( pucTmpBlock, ui32ToBlockAddr);

		pTmpBlock->Release();
		pTmpBlock = NULL;
		pucTmpBlock = NULL;
	}

	if( getNextInChain( m_pStack->pucBlock))
	{
		if( RC_BAD( rc = m_pBlockMgr->getBlock(
			getNextInChain( m_pStack->pucBlock), &pTmpBlock, &pucTmpBlock)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = m_pBlockMgr->prepareForUpdate( &pTmpBlock, 
			&pucTmpBlock)))
		{
			goto Exit;
		}
		
		f_assert( getPrevInChain( pucTmpBlock) == ui32FromBlockAddr);
		setPrevInChain( pucTmpBlock, ui32ToBlockAddr);

		pTmpBlock->Release();
		pTmpBlock = NULL;
		pucTmpBlock = NULL;
	}

	// Copy the content of the old block into the new block.
	
	pucSrc = m_pStack->pucBlock + sizeofBTreeBlockHdr( m_pStack->pucBlock);
	pucDest = m_pucBlock + sizeofBTreeBlockHdr( m_pucBlock);

	f_memcpy( pucDest, pucSrc, 
			m_uiBlockSize - sizeofBTreeBlockHdr( m_pStack->pucBlock));

	if( isRootBlock( m_pStack->pucBlock))
	{
		m_ui32RootBlockAddr = ui32ToBlockAddr;
		goto Exit;
	}

	// Move up one level to the parent entry.
	
	m_pStack++;
	f_assert( m_pStack->pBlock);

	// Log that we are making a change to the block.
	
	if( RC_BAD( rc = m_pBlockMgr->prepareForUpdate( 
		&m_pStack->pBlock, &m_pStack->pucBlock)))
	{
		goto Exit;
	}
	
	// Update the parent block with a new address for the new block.
	
	pucEntry = BtEntry( m_pStack->pucBlock, m_pStack->uiCurOffset);
	UD2FBA( ui32ToBlockAddr, pucEntry);

Exit:

	if( m_pBlock)
	{
		m_pBlock->Release();
		m_pBlock = NULL;
		m_pucBlock = NULL;
	}

	if( pTmpBlock)
	{
		pTmpBlock->Release();
	}

	f_free( &pucKey);
	releaseBlocks( TRUE);
	return( rc);
}

/***************************************************************************
Desc:	Move a DO block from one address to another, updating its reference
		btree entry.
****************************************************************************/
RCODE F_BTree::moveDOBlock(
	FLMUINT32				ui32FromBlockAddr,
	FLMUINT32				ui32ToBlockAddr)
{
	RCODE						rc = NE_FLM_OK;
	FLMBYTE *				pucEntry;
	FLMBYTE *				pucKey = NULL;
	FLMBYTE *				pucSrc;
	FLMBYTE *				pucDest;
	IF_Block *				pBlock = NULL;
	IF_Block *				pPrevBlock = NULL;
	IF_Block *				pNextBlock = NULL;
	FLMBYTE *				pucBlock = NULL;
	FLMBYTE *				pucPrevBlock = NULL;
	FLMBYTE *				pucNextBlock = NULL;
	FLMUINT					uiKeyLen;
	FLMUINT					uiOADataLen;
	const FLMBYTE *		pucData;
	FLMUINT32				ui32DOBlockAddr;
	FLMUINT					uiDataLen;
	FLMBYTE					ucDataBuffer[ sizeof(FLMUINT32)];
	FLMUINT					uiBlockHdrSize;
	
	f_assert( m_pBlock && m_pucBlock);

	// Log that we are changing this block.

	if( RC_BAD( m_pBlockMgr->prepareForUpdate( &m_pBlock, &m_pucBlock)))
	{
		goto Exit;
	}
	
	// Get the new block and verify that it is a free block.

	if( RC_BAD( m_pBlockMgr->getBlock( ui32ToBlockAddr, &pBlock, &pucBlock)))
	{
		goto Exit;
	}

	if( getBlockType( pucBlock) != F_BLK_TYPE_FREE)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
		goto Exit;
	}

	// Update the header of the new block to point to the prev and next
	// blocks etc..

	if( RC_BAD( m_pBlockMgr->prepareForUpdate( &pBlock, &pucBlock)))
	{
		goto Exit;
	}

	setPrevInChain( pucBlock, getPrevInChain( m_pucBlock));
	setNextInChain( pucBlock, getNextInChain( m_pucBlock));
	setBytesAvail( pucBlock, getBytesAvail( m_pucBlock));
	setBlockType( pucBlock, getBlockType( m_pucBlock));
	setBlockFlags( pucBlock, getBlockFlags( m_pucBlock));

	// Get the previous and next blocks and set their next and prev addresses.

	if( getPrevInChain( m_pucBlock))
	{
		if( RC_BAD( rc = m_pBlockMgr->getBlock(
			getPrevInChain( m_pucBlock), &pPrevBlock, &pucPrevBlock)))
		{
			goto Exit;
		}
		
		if( RC_BAD( m_pBlockMgr->prepareForUpdate( &pPrevBlock, &pucPrevBlock)))
		{
			goto Exit;
		}
		
		f_assert( getNextInChain( pucPrevBlock) == ui32FromBlockAddr);
		setNextInChain( pucPrevBlock, ui32ToBlockAddr);
		
		pPrevBlock->Release();
		pPrevBlock = NULL;
		pucPrevBlock = NULL;
	}

	if( getNextInChain( m_pucBlock))
	{
		if( RC_BAD( rc = m_pBlockMgr->getBlock(
			getNextInChain( m_pucBlock), &pNextBlock, &pucNextBlock)))
		{
			goto Exit;
		}
		
		if( RC_BAD( m_pBlockMgr->prepareForUpdate( &pNextBlock, &pucNextBlock)))
		{
			goto Exit;
		}
		
		f_assert( getPrevInChain( pucNextBlock) == ui32FromBlockAddr);
		setPrevInChain( pucNextBlock, ui32ToBlockAddr);
		
		pNextBlock->Release();
		pNextBlock = NULL;
		pucNextBlock = NULL;
	}

	// Copy the content of the old block into the new block.

	uiBlockHdrSize = sizeofDOBlockHdr( m_pucBlock);
	pucSrc = m_pucBlock + uiBlockHdrSize;
	pucDest = pucBlock + uiBlockHdrSize;
	f_memcpy( pucDest, pucSrc, m_uiBlockSize - uiBlockHdrSize);

	// Do we need to update the reference btree entry.

	if( getPrevInChain( m_pucBlock) == 0)
	{
		// Get the key from the beginning of the block.

		uiKeyLen = FB2UW( pucDest);
		pucKey = pucDest + sizeof( FLMUINT16);

		if( RC_BAD( rc = findEntry( pucKey, uiKeyLen, FLM_EXACT)))
		{
			// We must find it!
			
			RC_UNEXPECTED_ASSERT( rc);
			goto Exit;
		}

		// Verify that we found the right block.

		pucEntry = BtEntry( m_pStack->pucBlock, m_pStack->uiCurOffset);

		if( !bteDataBlockFlag( pucEntry))
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
			goto Exit;
		}

		uiDataLen = fbtGetEntryDataLength( pucEntry, &pucData,
							&uiOADataLen, NULL);

		ui32DOBlockAddr = bteGetBlockAddr( pucData);

		if( ui32DOBlockAddr != ui32FromBlockAddr)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
			goto Exit;
		}

		if( uiDataLen != sizeof( ucDataBuffer))
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
			goto Exit;
		}

		// Make the data entry with the new block address

		UD2FBA( ui32ToBlockAddr, ucDataBuffer);

		if( RC_BAD( rc = updateEntry(
			pucKey, uiKeyLen, ucDataBuffer, uiOADataLen, ELM_REPLACE_DO)))
		{
			goto Exit;
		}
	}

Exit:

	if( m_pBlock)
	{
		m_pBlock->Release();
		m_pBlock = NULL;
		m_pucBlock = NULL;
	}

	if( pBlock)
	{
		pBlock->Release();
	}

	if( pPrevBlock)
	{
		pPrevBlock->Release();
	}

	if( pNextBlock)
	{
		pNextBlock->Release();
	}

	releaseBlocks( TRUE);
	return( rc);
}

/***************************************************************************
Desc: Method to move the read point in an entry to a particular position
		within the entry.  This method will move to a previous or a later
		position.
****************************************************************************/
RCODE F_BTree::btSetReadPosition(
	FLMBYTE *			pucKey,
	FLMUINT				uiKeyLen,
	FLMUINT				uiPosition)
{
	RCODE					rc = NE_FLM_OK;
	FLMBYTE *			pucEntry;
	FLMUINT32			ui32BlockAddr;
	FLMBOOL				bLastElement = FALSE;

	if( !m_bOpened || !m_bSetupForRead)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_BAD_STATE);
		goto Exit;
	}

	// We cannot position to a point beyond the end of the current entry.
	
	if( uiPosition >= m_uiOADataLength)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
		goto Exit;
	}

	// The easiest case to handle is when we want to position within the
	// current entry.  We should not have to worry about the data only blocks
	// because the m_uiDataLength and m_uiDataRemaining are being set correctly
	// in setupReadState (via btLocateEntry, btNextEntry, btPrevEntry,
	// btFirstEntry and btLastEntry) which is always called before this method is
	// called.

	if( (uiPosition < (m_uiOffsetAtStart + m_uiDataLength)) &&
			(uiPosition >= m_uiOffsetAtStart))
	{
		m_uiDataRemaining = m_uiDataLength - (uiPosition - m_uiOffsetAtStart);
		m_uiOADataRemaining = m_uiOADataLength - uiPosition;
		goto Exit;
	}

	// Get the current block.  It is either a DO or a Btree block.
	
	if( !m_pBlock)
	{
		if( RC_BAD( rc = m_pBlockMgr->getBlock( m_ui32CurBlockAddr, 
			&m_pBlock, &m_pucBlock)))
		{
			goto Exit;
		}
	}

	// The next case is when the new position is in a *previous* entry, possibly
	// a previous block.
	
	while( uiPosition < m_uiOffsetAtStart)
	{
		// Are we dealing with DataOnly blocks?
		
		if( m_bDataOnlyBlock)
		{
			ui32BlockAddr = getPrevInChain( m_pucBlock);
			f_assert( ui32BlockAddr);

			m_pBlock->Release();
			m_pBlock = NULL;
			m_pucBlock = NULL;

			if( RC_BAD( rc = m_pBlockMgr->getBlock( ui32BlockAddr, 
				&m_pBlock, &m_pucBlock)))
			{
				goto Exit;
			}
			
			m_ui32CurBlockAddr = ui32BlockAddr;
			m_uiDataLength = m_uiBlockSize - getBytesAvail( m_pucBlock) -
										sizeofDOBlockHdr( m_pucBlock);
										
			if( !getPrevInChain( m_pucBlock))
			{
				FLMBYTE *		pucPtr = m_pucBlock + sizeofDOBlockHdr( m_pucBlock);
				FLMUINT16		ui16KeyLen = FB2UW( pucPtr);

				// We need to adjust for the key in the first block.
				
				m_uiDataLength -= ui16KeyLen;
			}
			
			// Decrement by the size of the current data
			
			m_uiOffsetAtStart -= m_uiDataLength;
		}
		else
		{
			// Backup to the previous element. This may or may not get
			// another block
			
			if( RC_BAD( rc = backupToPrevElement( FALSE)))
			{
				goto Exit;
			}

			pucEntry = BtEntry( m_pucBlock, m_uiCurOffset);

			// Make sure we are still looking at the same key etc.
			
			if( !checkContinuedEntry(
				pucKey, uiKeyLen, &bLastElement, pucEntry, getBlockType( m_pucBlock)))
			{
				// Should always match at this point!
				
				rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
				goto Exit;
			}

			m_uiDataLength = fbtGetEntryDataLength( pucEntry, NULL, NULL, NULL);
			m_uiOffsetAtStart -= m_uiDataLength;
		}
	}

	// Did we find the block?
	
	if( (uiPosition < (m_uiOffsetAtStart + m_uiDataLength)) &&
			(uiPosition >= m_uiOffsetAtStart))
	{
		m_uiDataRemaining = m_uiDataLength - (uiPosition - m_uiOffsetAtStart);
		m_uiOADataRemaining = m_uiOADataLength - uiPosition;
		goto Exit;
	}

	// Finally, we realize that the new position is beyond the current entry.
	
	while( uiPosition >= (m_uiOffsetAtStart + m_uiDataLength))
	{
		f_assert( m_uiDataLength + m_uiOffsetAtStart <= m_uiOADataLength);

		// Are we dealing with DataOnly blocks?
		
		if( m_bDataOnlyBlock)
		{
			ui32BlockAddr = getNextInChain( m_pucBlock);
			f_assert( ui32BlockAddr);

			m_pBlock->Release();
			m_pBlock = NULL;
			m_pucBlock = NULL;

			if( RC_BAD( rc = m_pBlockMgr->getBlock( ui32BlockAddr,
				&m_pBlock, &m_pucBlock)))
			{
				goto Exit;
			}
			
			m_ui32CurBlockAddr = ui32BlockAddr;

			// Increment by the size of the previous data.  Note that in this
			// case, we do not have to be concerned about the key in the first
			// DO block since we will never move forward to it.
			
			m_uiOffsetAtStart += m_uiDataLength;
			m_uiDataLength = m_uiBlockSize - getBytesAvail( m_pucBlock) - 
									sizeofDOBlockHdr( m_pucBlock);
		}
		else
		{
			// Advance to the next element. This may or may not get another block.
			// Be sure we do not advance the stack since we do not have one.
			
			if( RC_BAD( rc = advanceToNextElement( FALSE)))
			{
				goto Exit;
			}

			pucEntry = BtEntry( m_pucBlock, m_uiCurOffset);

			// Make sure we are still looking at the same key etc.
			
			if( !checkContinuedEntry(
				pucKey, uiKeyLen, &bLastElement, pucEntry, 
				getBlockType( m_pucBlock)))
			{
				// Should always match at this point!
				
				rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
				goto Exit;
			}

			// Get the data length of the current entry.
			
			m_uiOffsetAtStart += m_uiDataLength;
			m_uiDataLength = fbtGetEntryDataLength( pucEntry, NULL, NULL, NULL);
		}
	}

	// Did we find the block?  If we still don't find it, then we
	// have a big problem.
	
	if( (uiPosition >= (m_uiOffsetAtStart + m_uiDataLength)) ||
			(uiPosition < m_uiOffsetAtStart))
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
		goto Exit;
	}

	m_uiDataRemaining = m_uiDataLength - (uiPosition - m_uiOffsetAtStart);
	m_uiOADataRemaining = m_uiOADataLength - uiPosition;

Exit:

	if( m_pBlock)
	{
		m_pBlock->Release();
		m_pBlock = NULL;
		m_pucBlock = NULL;
	}

	releaseBlocks( FALSE);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE F_BTree::btGetReadPosition(
	FLMUINT *			puiPosition)
{
	RCODE					rc = NE_FLM_OK;

	if( !m_bOpened || !m_bSetupForRead)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_BAD_STATE);
		goto Exit;
	}

	f_assert( puiPosition);
	*puiPosition = m_uiOffsetAtStart + (m_uiDataLength - m_uiDataRemaining);

Exit:

	if( m_pBlock)
	{
		m_pBlock->Release();
		m_pBlock = NULL;
		m_pucBlock = NULL;
	}

	releaseBlocks( FALSE);
	return( rc);
}

/***************************************************************************
Desc: Performs a consistancy check on the BTree
		NOTE: Must be performed inside of a read transaction!
****************************************************************************/
RCODE F_BTree::btCheck(
	BTREE_ERR_INFO *		pErrInfo)
{
	RCODE						rc = NE_FLM_OK;
	FLMUINT32				ui32NextBlockAddr = 0;
	FLMUINT32				ui32NextLevelBlockAddr = 0;
	FLMUINT32				ui32ChildBlockAddr = 0;
	FLMUINT32				ui32DOBlockAddr = 0;
	FLMUINT					uiNumKeys;
	const FLMBYTE *		pucPrevKey;
	FLMUINT					uiPrevKeySize;
	const FLMBYTE *		pucCurKey;
	FLMUINT					uiCurKeySize;
	IF_Block *				pCurrentBlock = NULL;
	IF_Block *				pPrevBlock = NULL;
	FLMBYTE *				pucCurrentBlock = NULL;
	FLMBYTE *				pucPrevBlock = NULL;
	FLMBYTE *				pucEntry = NULL;
	FLMBYTE *				pucPrevEntry = NULL;
	IF_Block *				pChildBlock = NULL;
	FLMBYTE *				pucChildBlock = NULL;
	FLMUINT16 *				puiOffsetArray;
	BTREE_ERR_INFO			localErrInfo;
	FLMINT					iCmpResult;
	FLMUINT					uiOADataLength = 0;

	// Initial setup...
	
	ui32NextLevelBlockAddr = m_ui32RootBlockAddr;
	f_memset( &localErrInfo, 0, sizeof( localErrInfo));
	localErrInfo.uiBlockSize = m_uiBlockSize;

	// While there's a next level....
	
	while( ui32NextLevelBlockAddr)
	{
		localErrInfo.uiLevels++;
		ui32NextBlockAddr = ui32NextLevelBlockAddr;

		// Update uiNextLevelBlockAddr
		
		if( RC_BAD( rc = m_pBlockMgr->getBlock( ui32NextBlockAddr, 
			&pCurrentBlock, &pucCurrentBlock)))
		{
			localErrInfo.type = GET_BLOCK_FAILED;
			f_sprintf( localErrInfo.szMsg, 
				"Failed to get block at %X", ui32NextBlockAddr);
			goto Exit;
		}
		
		puiOffsetArray = BtOffsetArray( pucCurrentBlock, 0);
		
		if( getBlockType( pucCurrentBlock) == F_BLK_TYPE_BT_LEAF ||
			 getBlockType( pucCurrentBlock) == F_BLK_TYPE_BT_LEAF_DATA)
		{
			ui32NextLevelBlockAddr = 0;
		}
		else
		{
			pucEntry = BtEntry( pucCurrentBlock, 0);

			// The child block address is the first part of the entry
			
			ui32NextLevelBlockAddr = bteGetBlockAddr( pucEntry);
		}

		if( pPrevBlock)
		{
			pPrevBlock->Release();
			pPrevBlock = NULL;
			pucPrevBlock = NULL;
		}

		// While there's another block on this level...
		
		while( ui32NextBlockAddr) 
		{
			// This loop assumes that pCurrentBlock is already initialized.
			
			localErrInfo.uiBlocksChecked++;
			localErrInfo.uiAvgFreeSpace =
				(localErrInfo.uiAvgFreeSpace * 
					(localErrInfo.uiBlocksChecked - 1) / 
						localErrInfo.uiBlocksChecked) +
				(getBytesAvail( pucCurrentBlock) / localErrInfo.uiBlocksChecked);
			localErrInfo.ui64FreeSpace += getBytesAvail( pucCurrentBlock);

			localErrInfo.LevelStats[ localErrInfo.uiLevels - 1].uiBlockCnt++;
			localErrInfo.LevelStats[ localErrInfo.uiLevels - 1].uiBytesUsed +=
										(m_uiBlockSize - getBytesAvail( pucCurrentBlock));

			uiNumKeys = getNumKeys( pucCurrentBlock);

			// Verify that the keys are in order...
			// Make sure that we check the keys between blocks as well.
			
			if( pPrevBlock)
			{
				pucEntry = BtLastEntry( pucPrevBlock);
				uiPrevKeySize = getEntryKeyLength( pucEntry, 
										getBlockType( pucPrevBlock), &pucPrevKey);
			}
			else
			{
				pucEntry = BtEntry( pucCurrentBlock, 0);
				uiPrevKeySize = getEntryKeyLength( pucEntry, 
						getBlockType( pucCurrentBlock), &pucPrevKey);

				if( getBlockType( pucCurrentBlock) == F_BLK_TYPE_BT_LEAF_DATA)
				{
					if( bteFirstElementFlag( pucEntry))
					{
						localErrInfo.LevelStats[ 
								localErrInfo.uiLevels - 1].uiFirstKeyCnt++;
					}
				}
				else
				{
					// Everything else is a first key.
					
					localErrInfo.LevelStats[
						localErrInfo.uiLevels - 1].uiFirstKeyCnt++;
				}
			}
			
			for( FLMUINT uiLoop = (pPrevBlock ? 0: 1); 
				  uiLoop < uiNumKeys; uiLoop++)
			{
				pucPrevEntry = pucEntry;
				pucEntry = BtEntry( pucCurrentBlock, uiLoop);

				if( getBlockType( pucCurrentBlock) == F_BLK_TYPE_BT_LEAF_DATA)
				{
					if( bteFirstElementFlag( pucEntry))
					{
						localErrInfo.LevelStats[ 
							localErrInfo.uiLevels - 1].uiFirstKeyCnt++;
					}
				}
				else
				{
					// Everything else is a first key.
					
					localErrInfo.LevelStats[
						localErrInfo.uiLevels - 1].uiFirstKeyCnt++;
				}

				uiCurKeySize = getEntryKeyLength( pucEntry,
										getBlockType( pucCurrentBlock), &pucCurKey);

				// The last key in the last block of each level is an infinity marker
				// It must have a 0 keylength and if it's a leaf node, a 0 datalength.
				
				if( (uiLoop == uiNumKeys - 1) && 
					 getNextInChain( pucCurrentBlock) == 0)
				{
					// If the key size is not 0, or we're a leaf block, and the
					// data size is not 0 ...
					
					if( (uiCurKeySize != 0) ||
						 (((getBlockType( pucCurrentBlock) == F_BLK_TYPE_BT_LEAF_DATA)) &&
							(fbtGetEntryDataLength( pucEntry, NULL, NULL, NULL) > 0)))
					{
						localErrInfo.type = INFINITY_MARKER;
						localErrInfo.uiBlockAddr = getBlockAddr( pucCurrentBlock);
						f_sprintf( localErrInfo.szMsg, "Invalid Infinity Marker %ul", uiLoop);
						rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
						goto Exit;
					}
				}
				else
				{
					// Do a comparison of the previous and current keys ...
					
					if( RC_BAD( rc = compareKeys( pucPrevKey, uiPrevKeySize,
						pucCurKey, uiCurKeySize, &iCmpResult)))
					{
						goto Exit;
					}
					
					if( iCmpResult > 0)
					{
						localErrInfo.type = KEY_ORDER;
						localErrInfo.uiBlockAddr = getBlockAddr( pucCurrentBlock);
						f_sprintf( localErrInfo.szMsg, "Key Number %ul", uiLoop);
						rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
						goto Exit;
					}

					if( getBlockType( pucCurrentBlock) == F_BLK_TYPE_BT_LEAF_DATA)
					{
						if( iCmpResult < 0)
						{
							f_assert( *pucEntry & BTE_FLAG_FIRST_ELEMENT);
						}
						else if( iCmpResult == 0)
						{
							f_assert( (*pucEntry & BTE_FLAG_FIRST_ELEMENT) == 0);
							f_assert( (*pucPrevEntry & BTE_FLAG_LAST_ELEMENT) == 0);
						}
					}
				}

				pucPrevKey = pucCurKey;
				uiPrevKeySize = uiCurKeySize;
			}

			localErrInfo.uiNumKeys += uiNumKeys;
			localErrInfo.LevelStats[ 
							localErrInfo.uiLevels - 1].uiKeyCnt += uiNumKeys;

			// If this is a leaf block, check for any pointers to data-only
			// blocks.  Verify the blocks...
			
			if( getBlockType( pucCurrentBlock) == F_BLK_TYPE_BT_LEAF || 
				 getBlockType( pucCurrentBlock) == F_BLK_TYPE_BT_LEAF_DATA)
			{
				if( getBlockType( pucCurrentBlock) == F_BLK_TYPE_BT_LEAF_DATA)
				{
					for( FLMUINT uiLoop = 0; uiLoop < uiNumKeys; uiLoop++)
					{
						pucEntry = BtEntry( pucCurrentBlock, uiLoop);
						
						if( bteDataBlockFlag( pucEntry))
						{
							FLMBYTE	ucDOBlockAddr[ 4];

							if( RC_BAD( rc = fbtGetEntryData( pucEntry, 
								&ucDOBlockAddr[ 0], 4, NULL)))
							{
								RC_UNEXPECTED_ASSERT( rc);
								localErrInfo.type = CATASTROPHIC_FAILURE;
								localErrInfo.uiBlockAddr = getBlockAddr( pucCurrentBlock);
								f_sprintf( localErrInfo.szMsg, 
										"getEntryData couldn't get the DO blk addr.");
								goto Exit;
							}

							ui32DOBlockAddr = bteGetBlockAddr( &ucDOBlockAddr[ 0]);

							// Verify that there is an OverallDataLength field

							if( bteOADataLenFlag( pucEntry) == 0)
							{
								localErrInfo.type = MISSING_OVERALL_DATA_LENGTH;
								localErrInfo.uiBlockAddr = getBlockAddr( pucCurrentBlock);
								f_sprintf( localErrInfo.szMsg, 
									"OverallDataLength field is missing");
							}
							else
							{
								if( bteKeyLenFlag( pucEntry))
								{
									uiOADataLength = FB2UD( pucEntry + 4);
								}
								else
								{
									uiOADataLength = FB2UD( pucEntry + 3);
								}
							}

							if( RC_BAD( rc = verifyDOBlockChain( ui32DOBlockAddr,
								uiOADataLength , &localErrInfo)))
							{
								goto Exit;
							}
						}
					}
				}
			}
			else
			{
				// This is a non-leaf block, verify that blocks exist for all
				// the child block addresses

				// NOTE: Also need to somehow verify that no two elements have the
				// same child block address...				
					
				for( FLMUINT uiLoop = 0; uiLoop < uiNumKeys; uiLoop++)
				{
					pucEntry = BtEntry( pucCurrentBlock, uiLoop);
					ui32ChildBlockAddr = bteGetBlockAddr( pucEntry);
					
					if( RC_BAD( rc = m_pBlockMgr->getBlock( 
						ui32ChildBlockAddr, &pChildBlock, &pucChildBlock)))
					{
						localErrInfo.type = GET_BLOCK_FAILED;
						f_sprintf( localErrInfo.szMsg, "Failed to get block at %X", 
							ui32ChildBlockAddr);
						goto Exit;
					}

					pChildBlock->Release();
					pChildBlock = NULL;
					pucChildBlock = NULL;
				}
			}

			// Release the current block and get the next one
			
			ui32NextBlockAddr = getNextInChain( pucCurrentBlock);
			
			if( pPrevBlock)
			{
				pPrevBlock->Release();
				pPrevBlock = NULL;
				pucPrevBlock = NULL;
			}
			
			pPrevBlock = pCurrentBlock;
			pucPrevBlock = pucCurrentBlock;
			
			pCurrentBlock = NULL;
			pucCurrentBlock = NULL;
			
			if( ui32NextBlockAddr)
			{
				if( RC_BAD( rc = m_pBlockMgr->getBlock(
					ui32NextBlockAddr, &pCurrentBlock, &pucCurrentBlock)))
				{
					localErrInfo.type = GET_BLOCK_FAILED;
					f_sprintf( localErrInfo.szMsg, 
						"Failed to get block at %X", ui32ChildBlockAddr);
					goto Exit;
				}
			}
		}
	}

	if( m_bCounts)
	{
		if( RC_BAD( rc = verifyCounts( &localErrInfo)))
		{
			goto Exit;
		}
	}

Exit:

	if( pPrevBlock)
	{
		pPrevBlock->Release();
	}

	if( pCurrentBlock)
	{
		pCurrentBlock->Release();
	}
	
	if( pErrInfo)
	{
		f_memcpy( pErrInfo, &localErrInfo, sizeof( localErrInfo));
	}
	
	return( rc);
}

/***************************************************************************
Desc: Performs an integrity check on a chain of data-only blocks.  Should
		only be called from btCheck().  Note that unlike btCheck(),
		pErrInfo CANNOT be NULL here.
****************************************************************************/
RCODE F_BTree::verifyDOBlockChain(
	FLMUINT					uiDOAddr,
	FLMUINT					uiDataLength,
	BTREE_ERR_INFO *		pErrInfo)
{
	RCODE						rc = NE_FLM_OK;
	FLMUINT					uiRunningLength = 0;
	IF_Block *				pCurrentBlock = NULL;
	FLMBYTE *				pucCurrentBlock = NULL;
	FLMUINT32				ui32NextAddr = (FLMUINT32)uiDOAddr;
	FLMUINT					uiDataSize;

	while( ui32NextAddr)
	{
		pErrInfo->LevelStats[ pErrInfo->uiLevels - 1].uiDOBlockCnt++;
		
		// Get the next block
		
		if( RC_BAD( m_pBlockMgr->getBlock( ui32NextAddr, 
			&pCurrentBlock, &pucCurrentBlock)))
		{
			pErrInfo->type = GET_BLOCK_FAILED;
			f_sprintf( pErrInfo->szMsg, "Failed to get block at %X", uiDOAddr);
			goto Exit;
		}
		
		// Verify that it's really a DO Block
		
		if( getBlockType( pucCurrentBlock) != F_BLK_TYPE_BT_DATA_ONLY)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
			pErrInfo->type = NOT_DATA_ONLY_BLOCK;
			goto Exit;
		}

		// Update counts info in pErrInfo
		
		pErrInfo->LevelStats[ pErrInfo->uiLevels - 1].uiDOBytesUsed +=
						m_uiBlockSize - getBytesAvail( pucCurrentBlock);
						
		// Update the data length running total
		
		uiDataSize = m_uiBlockSize - sizeofDOBlockHdr( pucCurrentBlock) - 
						 getBytesAvail( pucCurrentBlock);
										
		if( getPrevInChain( pucCurrentBlock) == 0)
		{
			FLMBYTE *		pucPtr = pucCurrentBlock + sizeofDOBlockHdr( pucCurrentBlock);
			FLMUINT16		ui16KeyLen = FB2UW( pucPtr);
			
			uiDataSize -= (ui16KeyLen + 2);
		}
		
		uiRunningLength += uiDataSize;

		// Update ui32nextAddr
		
		ui32NextAddr = getNextInChain( pucCurrentBlock);

		// Release it when we no longer need it.
		
		pCurrentBlock->Release();
		pCurrentBlock = NULL;
		pucCurrentBlock = NULL;
	}

	// Check the calculated overall length vs. uiDataLength
	
	if( uiRunningLength != uiDataLength)
	{
		pErrInfo->type = BAD_DO_BLOCK_LENGTHS;
		rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
		goto Exit;
	}

Exit:

	if( pCurrentBlock)
	{
		pCurrentBlock->Release();
	}

	if( rc == NE_FLM_BTREE_ERROR)
	{
		f_sprintf( pErrInfo->szMsg, "Corrupt DO chain starting at %X", uiDOAddr);
	}

	return( NE_FLM_OK);
}

/***************************************************************************
Desc:	Method to check the counts in a database with counts.
****************************************************************************/
RCODE F_BTree::verifyCounts(
	BTREE_ERR_INFO *	pErrInfo)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT				uiNextLevelBlockAddr;
	FLMUINT				uiNextBlockAddr;
	FLMUINT				uiChildBlockAddr;
	IF_Block *			pCurrentBlock = NULL;
	IF_Block *			pChildBlock = NULL;
	FLMBYTE *			pucCurrentBlock = NULL;
	FLMBYTE *			pucChildBlock = NULL;
	FLMBYTE *			pucEntry;
	FLMUINT				uiNumKeys;
	FLMUINT				uiEntryNum;
	FLMUINT				uiParentCounts;
	FLMUINT				uiChildCounts;
	FLMBOOL				bDone = FALSE;

	f_assert( m_bCounts);

	// Repeat at each level, starting at the root.
	
	uiNextLevelBlockAddr = m_ui32RootBlockAddr;

	while( uiNextLevelBlockAddr)
	{
		if( RC_BAD( rc = m_pBlockMgr->getBlock( 
			(FLMUINT32)uiNextLevelBlockAddr, &pCurrentBlock, &pucCurrentBlock)))
		{
			goto Exit;
		}

		if( getBlockType( pucCurrentBlock) != F_BLK_TYPE_BT_NON_LEAF_COUNTS)
		{
			pCurrentBlock->Release();
			pCurrentBlock = NULL;
			pucCurrentBlock = NULL;
			break;
		}

		pucEntry = BtEntry( pucCurrentBlock, 0);
		uiNextLevelBlockAddr = bteGetBlockAddr( pucEntry);

		// For every entry in the block, and for every block on this level,
		// check that the counts match the actual counts in the corresponding
		// child block.
		
		bDone = FALSE;
		while( !bDone)
		{
			uiNumKeys = getNumKeys( pucCurrentBlock);

			// Now check every entry in this block.
			
			for( uiEntryNum = 0; uiEntryNum < uiNumKeys; uiEntryNum++)
			{
				pucEntry = BtEntry( pucCurrentBlock, uiEntryNum);
				uiChildBlockAddr = bteGetBlockAddr( pucEntry);

				pucEntry += 4;
				uiParentCounts = FB2UD( pucEntry);

				if( RC_BAD( rc = m_pBlockMgr->getBlock(
					(FLMUINT32)uiChildBlockAddr, &pChildBlock, &pucChildBlock)))
				{
					goto Exit;
				}
				
				uiChildCounts = countKeys( pucChildBlock);

				if( uiChildCounts != uiParentCounts)
				{
					pErrInfo->type = BAD_COUNTS;
					pErrInfo->uiBlockAddr = getBlockAddr( pucChildBlock);
					
					f_sprintf(
						pErrInfo->szMsg,
						"Counts do not match.  Expected %d, got %d",
						uiParentCounts, uiChildCounts);
					rc = RC_SET_AND_ASSERT( NE_FLM_BTREE_ERROR);
					goto Exit;
				}

				pChildBlock->Release();
				pChildBlock = NULL;
				pucChildBlock = NULL;
			}

			// Now get the next block at this level.
			
			uiNextBlockAddr = getNextInChain( pucCurrentBlock);
			
			pCurrentBlock->Release();
			pCurrentBlock = NULL;
			pucCurrentBlock = NULL;

			if( uiNextBlockAddr == 0)
			{
				bDone = TRUE;
			}
			else
			{
				if( RC_BAD( rc = m_pBlockMgr->getBlock(
					(FLMUINT32)uiNextBlockAddr, &pCurrentBlock, &pucCurrentBlock)))
				{
					goto Exit;
				}
			}
		}
	}

Exit:

	if( pCurrentBlock)
	{
		pCurrentBlock->Release();
	}

	if( pChildBlock)
	{
		pChildBlock->Release();
	}
	
	return( rc);
}

/***************************************************************************
Desc:	Verify that the distance (in bytes) between pvStart and pvEnd is
		what was specified in uiOffset.
****************************************************************************/
void f_verifyOffset(
	FLMUINT			uiCompilerOffset,
	FLMUINT			uiOffset,
	RCODE *			pRc)
{
	if( RC_OK( *pRc))
	{
		if ( uiCompilerOffset != uiOffset)
		{
			*pRc = RC_SET_AND_ASSERT( NE_FLM_BAD_PLATFORM_FORMAT);
		}
	}
}

/***************************************************************************
Desc:	Verify the offsets of each member of every on-disk structure.  This
		is a safety check to ensure that things work correctly on every
		platform.
****************************************************************************/
RCODE f_verifyDiskStructOffsets( void)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiSizeOf;

	// Verify the offsets in the F_STD_BLK_HDR structure.

	f_verifyOffset( (FLMUINT)f_offsetof(F_LARGEST_BLK_HDR, all.stdBlockHdr.ui32BlockAddr),
						  F_STD_BLK_HDR_ui32BlockAddr_OFFSET, &rc);
	f_verifyOffset( (FLMUINT)f_offsetof(F_LARGEST_BLK_HDR, all.stdBlockHdr.ui32PrevBlockInChain),
						  F_STD_BLK_HDR_ui32PrevBlockInChain_OFFSET, &rc);
	f_verifyOffset( (FLMUINT)f_offsetof(F_LARGEST_BLK_HDR, all.stdBlockHdr.ui32NextBlockInChain),
						  F_STD_BLK_HDR_ui32NextBlockInChain_OFFSET, &rc);
	f_verifyOffset( (FLMUINT)f_offsetof(F_LARGEST_BLK_HDR, all.stdBlockHdr.ui32PriorBlockImgAddr),
						  F_STD_BLK_HDR_ui32PriorBlockImgAddr_OFFSET, &rc);
	f_verifyOffset( (FLMUINT)f_offsetof(F_LARGEST_BLK_HDR, all.stdBlockHdr.ui64TransId),
						  F_STD_BLK_HDR_ui64TransId_OFFSET, &rc);
	f_verifyOffset( (FLMUINT)f_offsetof(F_LARGEST_BLK_HDR, all.stdBlockHdr.ui32BlockChecksum),
						  F_STD_BLK_HDR_ui32BlockChecksum_OFFSET, &rc);
	f_verifyOffset( (FLMUINT)f_offsetof(F_LARGEST_BLK_HDR, all.stdBlockHdr.ui16BlockBytesAvail),
						  F_STD_BLK_HDR_ui16BlockBytesAvail_OFFSET, &rc);
	f_verifyOffset( (FLMUINT)f_offsetof(F_LARGEST_BLK_HDR, all.stdBlockHdr.ui8BlockFlags),
						  F_STD_BLK_HDR_ui8BlockFlags_OFFSET, &rc);
	f_verifyOffset( (FLMUINT)f_offsetof(F_LARGEST_BLK_HDR, all.stdBlockHdr.ui8BlockType),
						  F_STD_BLK_HDR_ui8BlockType_OFFSET, &rc);

	// Have to use a variable for sizeof.  If we don't, compiler barfs
	// because we are comparing two constants.

	uiSizeOf = SIZEOF_STD_BLK_HDR;
	if (sizeof( F_STD_BLK_HDR) != uiSizeOf)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BAD_PLATFORM_FORMAT);
	}

	// Verify the offsets in the F_BTREE_BLK_HDR structure

	f_verifyOffset( (FLMUINT)f_offsetof(F_LARGEST_BLK_HDR, all.BTreeBlockHdr.ui16BtreeId),
						  F_BTREE_BLK_HDR_ui16BtreeId_OFFSET, &rc);
	f_verifyOffset( (FLMUINT)f_offsetof(F_LARGEST_BLK_HDR, all.BTreeBlockHdr.ui16NumKeys),
						  F_BTREE_BLK_HDR_ui16NumKeys_OFFSET, &rc);
	f_verifyOffset( (FLMUINT)f_offsetof(F_LARGEST_BLK_HDR, all.BTreeBlockHdr.ui8BlockLevel),
						  F_BTREE_BLK_HDR_ui8BlockLevel_OFFSET, &rc);
	f_verifyOffset( (FLMUINT)f_offsetof(F_LARGEST_BLK_HDR, all.BTreeBlockHdr.ui8BTreeFlags),
						  F_BTREE_BLK_HDR_ui8BTreeFlags_OFFSET, &rc);
	f_verifyOffset( (FLMUINT)f_offsetof(F_LARGEST_BLK_HDR, all.BTreeBlockHdr.ui16HeapSize),
						  F_BTREE_BLK_HDR_ui16HeapSize_OFFSET, &rc);

	// Have to use a variable for sizeof.  If we don't, compiler barfs
	// because we are comparing two constants.

	uiSizeOf = 40;
	if (sizeof( F_BTREE_BLK_HDR) != uiSizeOf)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BAD_PLATFORM_FORMAT);
	}

	// Have to use a variable for sizeof.  If we don't, compiler barfs
	// because we are comparing two constants.

	uiSizeOf = 40;
	if (sizeof( F_LARGEST_BLK_HDR) != uiSizeOf)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_BAD_PLATFORM_FORMAT);
	}

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
F_BlockMgr::~F_BlockMgr()
{
	if( m_pHashTbl)
	{
		freeAllBlocks();
		f_free( &m_pHashTbl);
	}
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE F_BlockMgr::setup(
	FLMUINT					uiBlockSize)
{
	RCODE			rc = NE_FLM_OK;
	
	m_uiBlockSize = uiBlockSize;
	m_uiBuckets = 1024;
	
	if( RC_BAD( rc = f_alloc( m_uiBuckets * sizeof( F_Block *), &m_pHashTbl)))
	{
		goto Exit;
	}

	f_memset( m_pHashTbl, 0, sizeof( F_Block *) * m_uiBuckets);
	
Exit:

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
FLMUINT FTKAPI F_BlockMgr::getBlockSize( void)
{
	return( m_uiBlockSize);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_BlockMgr::getBlock(
	FLMUINT32				ui32BlockAddr,
	IF_Block **				ppBlock,
	FLMBYTE **				ppucBlock)
{
	RCODE						rc = NE_FLM_OK;
	F_Block *				pBlock =  m_pHashTbl[ ui32BlockAddr % m_uiBuckets];
	
	f_assert( *ppBlock == NULL && *ppucBlock == NULL);
	
	while( pBlock && pBlock->m_ui32BlockAddr != ui32BlockAddr)
	{
		pBlock = pBlock->m_pNextInBucket;
	}
	
	if( !pBlock)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_NOT_FOUND);
		goto Exit;
	}
	
	*ppBlock = pBlock;
	(*ppBlock)->AddRef();
	*ppucBlock = pBlock->m_pucBlock;
	
Exit:

	return( rc);
}
	
/***************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_BlockMgr::createBlock(
	IF_Block **				ppBlock,
	FLMBYTE **				ppucBlock,
	FLMUINT32 *				pui32BlockAddr)
{
	RCODE						rc = NE_FLM_OK;
	F_Block *				pBlock = NULL;
	F_Block **				ppBucket = NULL;
	
	if( (pBlock = f_new F_Block) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = f_alloc( m_uiBlockSize, &pBlock->m_pucBlock)))
	{
		goto Exit;
	}
	
	pBlock->m_ui32BlockAddr = m_ui32NextBlockAddr++;
	ppBucket = &m_pHashTbl[ pBlock->m_ui32BlockAddr % m_uiBuckets]; 	
	
	if( (pBlock->m_pNextInBucket = *ppBucket) != NULL)
	{
		pBlock->m_pNextInBucket->m_pPrevInBucket = pBlock;
	}

	*ppBucket = pBlock;
	*ppBlock = pBlock;
	pBlock->AddRef();
	
	*ppucBlock = pBlock->m_pucBlock;
	*pui32BlockAddr = pBlock->m_ui32BlockAddr;
	
	pBlock = NULL;
	
Exit:

	if( pBlock)
	{
		pBlock->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_BlockMgr::freeBlock(
	IF_Block **,//		ppBlock,
	FLMBYTE **)	//	ppucBlock)
{
	
//	// Block should be referenced only by the manager and caller
//	
//	f_assert( (*ppBlock)->getRefCount() == 2);
	
	return( NE_FLM_OK);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_BlockMgr::prepareForUpdate(
	IF_Block **,		// ppBlock,
	FLMBYTE **)			// ppucBlock)
{
	return( NE_FLM_OK);
}

/***************************************************************************
Desc:
****************************************************************************/
void F_BlockMgr::freeAllBlocks( void)
{
	FLMUINT			uiLoop;
	F_Block *		pBlock;
	F_Block *		pNextBlock;
	
	for( uiLoop = 0; uiLoop < m_uiBuckets; uiLoop++)
	{
		pBlock = m_pHashTbl[ uiLoop];
		while( pBlock)
		{
			f_assert( pBlock->getRefCount() == 1);
			pNextBlock = pBlock->m_pNextInBucket;
			pBlock->m_pPrevInBucket = NULL;
			pBlock->m_pNextInBucket = NULL;
			pBlock->Release();
			pBlock = pNextBlock;
		}
	}
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI FlmAllocBlockMgr(
	FLMUINT				uiBlockSize,
	IF_BlockMgr **		ppBlockMgr)
{
	RCODE					rc = NE_FLM_OK;
	F_BlockMgr *		pBlockMgr = NULL;
	
	if( (pBlockMgr = f_new F_BlockMgr) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pBlockMgr->setup( uiBlockSize)))
	{
		goto Exit;
	}
	
	*ppBlockMgr = pBlockMgr;
	pBlockMgr = NULL;
	
Exit:

	if( pBlockMgr)
	{
		pBlockMgr->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI FlmAllocBTree(
	IF_BlockMgr *		pBlockMgr,
	IF_BTree **			ppBTree)
{
	RCODE					rc = NE_FLM_OK;
	F_BTree *			pBTree = NULL;
	IF_BlockMgr *		pTmpBlockMgr = NULL;

	if( !pBlockMgr)
	{
		if( RC_BAD( rc = FlmAllocBlockMgr( 4096, &pTmpBlockMgr)))
		{
			goto Exit;
		}

		pBlockMgr = pTmpBlockMgr;
	}
	
	if( (pBTree = f_new F_BTree( pBlockMgr)) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	*ppBTree = pBTree;
	pBTree = NULL;
	
Exit:

	if( pBTree)
	{
		pBTree->Release();
	}

	if( pTmpBlockMgr)
	{
		pTmpBlockMgr->Release();
	}

	return( rc);
}
