//------------------------------------------------------------------------------
// Desc:	Verify data in a database.
// Tabs:	3
//
// Copyright (c) 1991-2007 Novell, Inc. All Rights Reserved.
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

#include "flaimsys.h"

FSTATIC FLMBYTE * getEntryEnd(
	FLMBYTE *			pucEntry,
	FLMUINT				uiBlkType);

FLMUINT64 getLinkVal(
	FLMUINT				uiBMId,
	NODE_RS_ENTRY *	pRSEntry);

FSTATIC RCODE flmVerifyKeyOrder(
	STATE_INFO *	pStateInfo,
	LFILE *			pLFile,
	IXD *				pIxd,
	F_BLK_HDR *		pBlkHdr,
	FLMBYTE *		pucElmKey,
	FLMUINT			uiElmKeyLen,
	FLMUINT			uiElmOffset);
	
FSTATIC FLMBOOL flmVerifyElementChain(
	STATE_INFO *			pStateInfo,
	LFILE *					pLFile);
	
FSTATIC RCODE verifyRootLink(
	NODE_RS_ENTRY *		pRSEntry,
	FLMUINT					uiRSEntrySize,
	NODE_RS_ENTRY *		pTmpRSEntry,
	F_BtResultSet *		pResult,
	FLMINT32 *				pi32ErrCode);

FSTATIC RCODE verifyParentLink(
	NODE_RS_ENTRY *		pRSEntry,
	NODE_RS_ENTRY *		pTmpRSEntry,
	F_BtResultSet *		pResult,
	FLMINT32 *				pi32ErrCode);

FSTATIC RCODE verifyFirstChildLink(
	NODE_RS_ENTRY *		pRSEntry,
	NODE_RS_ENTRY *		pTmpRSEntry,
	F_BtResultSet *		pResult,
	FLMINT32 *				pi32ErrCode);

FSTATIC RCODE verifyLastChildLink(
	NODE_RS_ENTRY *		pRSEntry,
	NODE_RS_ENTRY *		pTmpRSEntry,
	F_BtResultSet *		pResult,
	FLMINT32 *				pi32ErrCode);

FSTATIC RCODE verifyPrevSiblingLink(
	NODE_RS_ENTRY *		pRSEntry,
	NODE_RS_ENTRY *		pTmpRSEntry,
	F_BtResultSet *		pResult,
	FLMINT32 *				pi32ErrCode);

FSTATIC RCODE verifyNextSiblingLink(
	NODE_RS_ENTRY *		pRSEntry,
	NODE_RS_ENTRY *		pTmpRSEntry,
	F_BtResultSet *		pResult,
	FLMINT32 *				pi32ErrCode);

FSTATIC RCODE verifyAnnotationLink(
	NODE_RS_ENTRY *		pRSEntry,
	NODE_RS_ENTRY *		pTmpRSEntry,
	F_BtResultSet *		pResult,
	FLMINT32 *				pi32ErrCode);

/********************************************************************
Desc: Verifies a block's header and sets up the STATE_INFO structure
		to verify the rest of the block.
*********************************************************************/
FLMINT32 flmVerifyBlockHeader(
	STATE_INFO *	pStateInfo,
	BLOCK_INFO *	pBlockInfo,
	FLMUINT			uiBlockSize,
	FLMUINT			uiExpNextBlkAddr,
	FLMUINT			uiExpPrevBlkAddr,
	FLMBOOL			bCheckEOF)
{
	F_BLK_HDR *				pBlkHdr = pStateInfo->pBlkHdr;
	
	if (pBlockInfo)
	{
		pBlockInfo->uiBlockCount++;
	}
	
	pStateInfo->ui32NextBlkAddr = pBlkHdr->ui32NextBlkInChain;
	
	if( (FLMUINT)pBlkHdr->ui16BlkBytesAvail >	uiBlockSize - blkHdrSize( pBlkHdr))
	{
		return( FLM_BAD_BLK_HDR_BLK_END);
	}
	else
	{
		if( pBlockInfo)
		{
			pBlockInfo->ui64BytesUsed +=
				(FLMUINT64)(uiBlockSize - pBlkHdr->ui16BlkBytesAvail - 
					blkHdrSize( pBlkHdr));
		}
	}

	// Verify the block address.

	if ((FLMUINT)pBlkHdr->ui32BlkAddr != pStateInfo->ui32BlkAddress)
	{
		return( FLM_BAD_BLK_HDR_ADDR);
	}

	// Verify that block address is below the logical EOF
	// Rebuild passes in FALSE for bCheckEOF

	if( bCheckEOF && pStateInfo->pDb)
	{
		if( !FSAddrIsBelow( pStateInfo->ui32BlkAddress, 
								  pStateInfo->pDb->getLogicalEOF()))
		{
			return( FLM_BAD_FILE_SIZE);
		}
	}

	// Verify the block type.

	if( pStateInfo->uiBlkType != 0xFF &&
		 pStateInfo->uiBlkType != (FLMUINT)pBlkHdr->ui8BlkType)
	{
		return( FLM_BAD_BLK_HDR_TYPE);
	}

	// Verify the block level - if there is one

	if( pStateInfo->uiBlkType != BT_DATA_ONLY &&
		 pStateInfo->uiLevel != 0xFF &&
		 blkIsBTree( pBlkHdr) &&
		 pStateInfo->uiLevel !=
			(FLMUINT)(((F_BTREE_BLK_HDR *)pBlkHdr)->ui8BlkLevel))
	{
		return( FLM_BAD_BLK_HDR_LEVEL);
	}

	// Verify the previous block address.  If uiExpPrevBlkAddr is 0xFFFFFFFF,
	// we do not know what the previous address should be, so we don't verify.

	if (uiExpPrevBlkAddr != 0xFFFFFFFF &&
		 uiExpPrevBlkAddr != (FLMUINT)pBlkHdr->ui32PrevBlkInChain)
	{
		return( FLM_BAD_BLK_HDR_PREV);
	}

	// Verify the next block address.  If uiExpNextBlkAddr is 0xFFFFFFFF,
	// we do not know what the next address should be, se we don't verify.

	if( uiExpNextBlkAddr != 0xFFFFFFFF &&
		 uiExpNextBlkAddr != pStateInfo->ui32NextBlkAddr)
	{
		return( FLM_BAD_BLK_HDR_NEXT);
	}

	// Verify that if it is a root block, the root bit flags is set,
	// or if it is NOT a root block, that the root bit flag is NOT set.
	// Note that Data Only blocks don't have the extended header, so we
	// can't do this check...
	
	if( pStateInfo->pCollection &&
		 (pStateInfo->uiBlkType != BT_DATA_ONLY) )
	{
		if( pStateInfo->uiLevel != 0xFF)
		{
			F_BTREE_BLK_HDR *	pBTreeBlkHdr = (F_BTREE_BLK_HDR *)pBlkHdr;
			FLMBOOL				bShouldBeRootBlk =
										(pStateInfo->uiLevel ==
										 pStateInfo->uiRootLevel)
										? TRUE
										: FALSE;

			if ((bShouldBeRootBlk && !isRootBlk( pBTreeBlkHdr)) ||
			 	 (!bShouldBeRootBlk && isRootBlk( pBTreeBlkHdr)))
			{
				return( FLM_BAD_BLK_HDR_ROOT_BIT);
			}
		}

		// Verify the logical file number - if any.
		
		if( (pBlkHdr->ui8BlkType != BT_DATA_ONLY) &&
			  (pStateInfo->pCollection->lfInfo.uiLfNum !=
		 		(FLMUINT)(((F_BTREE_BLK_HDR *)pBlkHdr)->ui16LogicalFile)) )
		{
			return( FLM_BAD_BLK_HDR_LF_NUM);
		}
	}

	return( 0);
}

/********************************************************************
Desc:	Verify an index or data element in a b-tree and set pStateInfo.
*********************************************************************/
RCODE flmVerifyElement(
	STATE_INFO *	pStateInfo,
	LFILE *			pLFile,
	IXD *				pIxd,
	FLMINT32 *		pi32ErrCode)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBYTE *		pucEntry;
	FLMUINT32		ui32DOAddr = 0;
	FLMUINT32		ui32ChildAddr = 0;
	FLMUINT			uiCounts = 0;

	*pi32ErrCode = 0;

	// Get the pointer to the element.
	
	pucEntry = BtEntry( (FLMBYTE *)pStateInfo->pBlkHdr, pStateInfo->uiElmOffset);
	pStateInfo->pucElm = pucEntry; 

	// Verify that the address of the entry is within the block.
	
	if ((FLMUINT)pucEntry > (FLMUINT)pStateInfo->pBlkHdr + 
				pStateInfo->pDb->getDatabase()->getBlockSize())
	{
		*pi32ErrCode = FLM_BAD_ELM_OFFSET;
		goto Exit;

	}

	switch( pStateInfo->pBlkHdr->ui8BlkType)
	{
		case BT_LEAF:
		{
			pStateInfo->uiElmKeyLen = FB2UW( pucEntry);
			f_memcpy( pStateInfo->pucElmKey, pucEntry + 2, pStateInfo->uiElmKeyLen);
			pStateInfo->uiElmLen = pStateInfo->uiElmKeyLen + 2;
			break;
		}
		
		case BT_LEAF_DATA:
		{
			FLMBYTE		ucFlags = *pucEntry;
			FLMBYTE *	pucPtr = pucEntry + 1;
			FLMUINT		uiElmLen = 1;

			if( ucFlags & BTE_FLAG_KEY_LEN)
			{
				// 2 byte key length
				
				pStateInfo->uiElmKeyLen = FB2UW( pucPtr);
				pucPtr += 2;
				uiElmLen += 2;
			}
			else
			{
				// 1 byte key length
				
				pStateInfo->uiElmKeyLen = (FLMUINT)*pucPtr;
				pucPtr++;
				uiElmLen++;

			}

			if( ucFlags & BTE_FLAG_DATA_LEN)
			{
				// 2 byte data length
				
				pStateInfo->uiElmDataLen = FB2UW( pucPtr);
				pucPtr += 2;
				uiElmLen += 2;
			}
			else
			{
				// 1 byte data length
				
				pStateInfo->uiElmDataLen = (FLMUINT)*pucPtr;
				pucPtr++;
				uiElmLen++;
			}

			if( ucFlags & BTE_FLAG_OA_DATA_LEN)
			{
				pStateInfo->uiElmOADataLen = FB2UD( pucPtr);
				pucPtr += 4;
				uiElmLen += 4;
			}

			f_memcpy( pStateInfo->pucElmKey, pucPtr, pStateInfo->uiElmKeyLen);
			pucPtr += pStateInfo->uiElmKeyLen;
			uiElmLen += pStateInfo->uiElmKeyLen;

			pStateInfo->pucElmData = pucPtr;
			uiElmLen += pStateInfo->uiElmDataLen;

			if( ucFlags & BTE_FLAG_DATA_BLOCK)
			{
				ui32DOAddr = FB2UD( pucPtr);
			}

			pStateInfo->uiElmLen = uiElmLen;
			break;
		}
		
		case BT_NON_LEAF:
		{
			FLMBYTE *	pucPtr = pucEntry;

			ui32ChildAddr = FB2UD( pucPtr);
			pucPtr += 4;

			pStateInfo->uiElmKeyLen = FB2UW( pucPtr);
			pucPtr += 2;

			f_memcpy( pStateInfo->pucElmKey, pucPtr, pStateInfo->uiElmKeyLen);
			pStateInfo->uiElmLen = pStateInfo->uiElmKeyLen + 6;
			break;
		}
		
		case BT_NON_LEAF_COUNTS:
		{
			FLMBYTE *	pucPtr = pucEntry;

			ui32ChildAddr = FB2UD( pucPtr);
			pucPtr += 4;

			uiCounts = FB2UD( pucPtr);
			pucPtr += 4;

			pStateInfo->uiElmKeyLen = FB2UW( pucPtr);
			pucPtr += 2;

			f_memcpy( pStateInfo->pucElmKey, pucPtr, pStateInfo->uiElmKeyLen);
			pStateInfo->uiElmLen = pStateInfo->uiElmKeyLen + 10;
			break;
		}
		
		default:
		{
			*pi32ErrCode = FLM_BAD_BLK_TYPE;
			goto Exit;
		}
	}

	// Verify that the end of the element is not past the end of the block.
	
	if( pStateInfo->pucElm + pStateInfo->uiElmLen > 
				(FLMBYTE *)pStateInfo->pBlkHdr +
				pStateInfo->pDb->getDatabase()->getBlockSize())
	{
		*pi32ErrCode = FLM_BAD_ELM_LEN;
		goto Exit;
	}

	// Verify the key length is not too big
	
	if( pStateInfo->uiElmKeyLen > XFLM_MAX_KEY_SIZE)
	{
		*pi32ErrCode = FLM_BAD_ELM_KEY_SIZE;
		goto Exit;
	}

	// Verify the key order.  First the previous key, then the next key.
	
	if( RC_BAD( rc = flmVerifyKeyOrder( pStateInfo, pLFile, pIxd,
		pStateInfo->pBlkHdr, pStateInfo->pucElmKey, pStateInfo->uiElmKeyLen,
		pStateInfo->uiElmOffset)))
	{
		*pi32ErrCode = FLM_BAD_ELM_KEY_ORDER;
		goto Exit;
	}

	pStateInfo->bValidKey = TRUE;

	if( !isIndexBlk( (F_BTREE_BLK_HDR *)pStateInfo->pBlkHdr))
	{
		switch( pStateInfo->pBlkHdr->ui8BlkType)
		{
			case BT_LEAF:
			case BT_LEAF_DATA:
			{
				FLMBOOL		bNeg;
				FLMUINT		uiBytesProcessed;

				if( pStateInfo->pucElm[0] & BTE_FLAG_FIRST_ELEMENT)
				{
					if( !flmVerifyElementChain( pStateInfo, pLFile))
					{
						*pi32ErrCode = FLM_BAD_ELEMENT_CHAIN;
						goto Exit;
					}
				}

				// The key length may be zero on a LEM
				
				if( pStateInfo->uiElmKeyLen)
				{
					if( RC_BAD( rc = flmCollation2Number(
								pStateInfo->uiElmKeyLen, pStateInfo->pucElmKey,
								&pStateInfo->ui64ElmNodeId, &bNeg, &uiBytesProcessed)))
					{
						*pi32ErrCode = FLM_BAD_ELM_KEY;
						goto Exit;
					}

					if( bNeg || uiBytesProcessed != pStateInfo->uiElmKeyLen)
					{
						*pi32ErrCode = FLM_BAD_ELM_KEY;
						goto Exit;
					}

					if( !pStateInfo->ui64ElmNodeId)
					{
						flmAssert( 0);
						*pi32ErrCode = FLM_BAD_ELM_KEY;
						goto Exit;
					}
				}
				else
				{
					// If the key length is zero, then this MUST be the last block!
					
					if( pStateInfo->pBlkHdr->ui32NextBlkInChain)
					{
						*pi32ErrCode = FLM_BAD_ELM_KEY;
						goto Exit;
					}
				}
				
				break;
			}
			
			case BT_NON_LEAF:
			{
				break;
			}
			
			case BT_NON_LEAF_COUNTS:
			{
				break;
			}
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Verify that the key order is correct.
*****************************************************************************/
FSTATIC RCODE flmVerifyKeyOrder(
	STATE_INFO *	pStateInfo,
	LFILE *			pLFile,
	IXD *				pIxd,
	F_BLK_HDR *		pBlkHdr,
	FLMBYTE *		pucElmKey,
	FLMUINT			uiElmKeyLen,
	FLMUINT			uiElmOffset)
{
	RCODE						rc = NE_XFLM_OK;
	F_CachedBlock *		pPrevSCache = NULL;
	F_CachedBlock *		pNextSCache = NULL;
	F_BTREE_BLK_HDR *		pPrevBlk;
	F_BTREE_BLK_HDR *		pNextBlk;
	FLMBYTE *				pucEntry = NULL;
	FLMBYTE *				pucKey = NULL;
	FLMUINT					uiKeyLen = 0;
	FLMBOOL					bCheckPrev = FALSE;
	FLMBOOL					bCheckNext = FALSE;
	FLMUINT					uiBlockSize;

	uiBlockSize = pStateInfo->pDb->getDatabase()->getBlockSize();

	// Get the previous key
	
	if( uiElmOffset)
	{
		flmAssert( uiElmOffset < ((F_BTREE_BLK_HDR *)pBlkHdr)->ui16NumKeys);
		pucEntry = BtEntry( (FLMBYTE *)pBlkHdr, uiElmOffset - 1);

		if( (FLMUINT)pucEntry > (FLMUINT)pBlkHdr + uiBlockSize)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_FAILURE);
			goto Exit;
		}
		
		bCheckPrev = TRUE;
	}
	else if( uiElmOffset == 0 && pBlkHdr->ui32PrevBlkInChain)
	{
		if( pLFile)
		{
			// Need to get the previous block.
			
			if( RC_BAD( rc = pStateInfo->pDb->getDatabase()->getBlock( 
				pStateInfo->pDb, pLFile, pBlkHdr->ui32PrevBlkInChain, NULL,
				&pPrevSCache)))
			{
				goto Exit;
			}

			pPrevBlk = (F_BTREE_BLK_HDR *)pPrevSCache->getBlockPtr();
			pucEntry = BtEntry( (FLMBYTE *)pPrevBlk, pPrevBlk->ui16NumKeys - 1);
									
			if( (FLMUINT)pucEntry > (FLMUINT)pPrevBlk + uiBlockSize)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_FAILURE);
				goto Exit;
			}

			if( pBlkHdr->ui8BlkType != pPrevBlk->stdBlkHdr.ui8BlkType)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_FAILURE);
				goto Exit;
			}
			
			bCheckPrev = TRUE;
		}
	}

	if( bCheckPrev)
	{
		// Get the key
		
		switch (pBlkHdr->ui8BlkType)
		{
			case BT_LEAF:
			{
				uiKeyLen = FB2UW( pucEntry);
				pucKey = pucEntry + 2;
				break;
			}
			
			case BT_LEAF_DATA:
			{
				FLMBYTE			ucFlags = *pucEntry;
				FLMBYTE *		pucPtr = pucEntry + 1;

				flmAssert( (*pucEntry & 0x03) == 0);

				if (ucFlags & BTE_FLAG_KEY_LEN)
				{
					uiKeyLen = FB2UW( pucPtr);
					pucPtr += 2;
				}
				else
				{
					uiKeyLen = *pucPtr;
					pucPtr++;
				}

				if( ucFlags & BTE_FLAG_DATA_LEN)
				{
					pucPtr += 2;
				}
				else
				{
					pucPtr++;
				}

				if (ucFlags & BTE_FLAG_OA_DATA_LEN)
				{
					pucPtr += 4;
				}
				
				pucKey = pucPtr;
				break;
			}
			
			case BT_NON_LEAF:
			{
				FLMBYTE *		pucPtr = pucEntry + 4;

				uiKeyLen = FB2UW( pucPtr);
				pucKey = pucPtr + 2;
				break;
			}
			
			case BT_NON_LEAF_COUNTS:
			{
				FLMBYTE *		pucPtr = pucEntry + 8;

				uiKeyLen = FB2UW( pucPtr);
				pucKey = pucPtr + 2;
				break;
			}
		}

		if( uiKeyLen)
		{
			if( !pLFile || pLFile->eLfType == XFLM_LF_COLLECTION)
			{
				if( f_memcmp( pucElmKey, pucKey, uiElmKeyLen < uiKeyLen
													? uiElmKeyLen
													: uiKeyLen) < 0)
				{
					rc = RC_SET_AND_ASSERT( NE_XFLM_FAILURE);
					goto Exit;
				}
			}
			else
			{
				// If uiElmKeyLen == 0, it is the LEM, in which case we won't
				// do the comparison, because pucElmKey is greater than pucKey
				// by definition.
				
				if( uiElmKeyLen)
				{
					FLMINT	iCompare;

					if( RC_BAD( rc = ixKeyCompare( pStateInfo->pDb, pIxd, NULL,
						NULL, NULL, TRUE, TRUE, pucElmKey, uiElmKeyLen,
						pucKey, uiKeyLen, &iCompare)))
					{
						goto Exit;
					}
					
					if( iCompare < 0)
					{
						rc = RC_SET_AND_ASSERT( NE_XFLM_FAILURE);
						goto Exit;
					}
				}
			}
		}
		else
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_FAILURE);
			goto Exit;
		}
	}

	// Get the next key
	
	if( uiElmOffset < (FLMUINT)((F_BTREE_BLK_HDR *)pBlkHdr)->ui16NumKeys - 1)
	{
		pucEntry = BtEntry( (FLMBYTE *)pBlkHdr, uiElmOffset + 1);
		
		if( pucEntry > (FLMBYTE *)pBlkHdr + uiBlockSize)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_FAILURE);
			goto Exit;
		}
		bCheckNext = TRUE;
	}
	else if( uiElmOffset == 
			  (FLMUINT)((F_BTREE_BLK_HDR *)pBlkHdr)->ui16NumKeys - 1 &&
			pBlkHdr->ui32NextBlkInChain)
	{
		if( pLFile)
		{
			if( RC_BAD( rc = pStateInfo->pDb->getDatabase()->getBlock( 
				pStateInfo->pDb, pLFile, pBlkHdr->ui32NextBlkInChain,
				NULL, &pNextSCache)))
			{
				goto Exit;
			}

			pNextBlk = (F_BTREE_BLK_HDR *)pNextSCache->getBlockPtr();
			pucEntry = BtEntry( (FLMBYTE *)pNextBlk, 0);
			
			if( pucEntry > (FLMBYTE *)pNextBlk + uiBlockSize)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_FAILURE);
				goto Exit;
			}

			if( pBlkHdr->ui8BlkType != pNextBlk->stdBlkHdr.ui8BlkType)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_FAILURE);
				goto Exit;
			}
			
			bCheckNext = TRUE;
		}
	}

	if( bCheckNext)
	{
		switch (pBlkHdr->ui8BlkType)
		{
			case BT_LEAF:
			{
				uiKeyLen = FB2UW( pucEntry);
				pucKey = pucEntry + 2;
				break;
			}
			
			case BT_LEAF_DATA:
			{
				FLMBYTE			ucFlags = *pucEntry;
				FLMBYTE *		pucPtr = pucEntry + 1;

				if( (*pucEntry & 0x03) != 0)
				{
					rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
					goto Exit;
				}

				if( ucFlags & BTE_FLAG_KEY_LEN)
				{
					uiKeyLen = FB2UW( pucPtr);
					pucPtr += 2;
				}
				else
				{
					uiKeyLen = *pucPtr;
					pucPtr++;
				}

				if( ucFlags & BTE_FLAG_DATA_LEN)
				{
					pucPtr += 2;
				}
				else
				{
					pucPtr++;
				}

				if( ucFlags & BTE_FLAG_OA_DATA_LEN)
				{
					pucPtr += 4;
				}
				
				pucKey = pucPtr;
				break;
			}
			
			case BT_NON_LEAF:
			{
				FLMBYTE *		pucPtr = pucEntry + 4;

				uiKeyLen = FB2UW( pucPtr);
				pucKey = pucPtr + 2;
				break;
			}
			
			case BT_NON_LEAF_COUNTS:
			{
				FLMBYTE *		pucPtr = pucEntry + 8;

				uiKeyLen = FB2UW( pucPtr);
				pucKey = pucPtr + 2;
				break;
			}
		}

		if (uiKeyLen)
		{
			if (!pLFile || pLFile->eLfType == XFLM_LF_COLLECTION)
			{
				if (f_memcmp(  pucKey, pucElmKey, uiElmKeyLen < uiKeyLen
													? uiElmKeyLen
													: uiKeyLen) < 0)
				{
					rc = RC_SET_AND_ASSERT( NE_XFLM_FAILURE);
					goto Exit;
				}
			}
			else
			{
				// If uiElmKeyLen == 0, it is the LEM, in which case we won't
				// do the comparison, because pucKey will always be less than
				// the LEM, but it is not an error in that case.
				
				if( uiElmKeyLen)
				{
					FLMINT	iCompare;
					
					if (RC_BAD( rc = ixKeyCompare( pStateInfo->pDb, pIxd, NULL,
						NULL, NULL, TRUE, TRUE, pucKey, uiKeyLen, pucElmKey,
						uiElmKeyLen, &iCompare)))
					{
						goto Exit;
					}
					
					if( iCompare < 0)
					{
						rc = RC_SET_AND_ASSERT( NE_XFLM_FAILURE);
						goto Exit;
					}
				}
			}
		}
		else
		{
			// A zero length key should only occur on the last block in the chain.
			
			if( pBlkHdr->ui32NextBlkInChain)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_FAILURE);
				goto Exit;
			}
		}
	}

Exit:

	if( pPrevSCache)
	{
		ScaReleaseCache( pPrevSCache, FALSE);
	}

	if( pNextSCache)
	{
		ScaReleaseCache( pNextSCache, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc:	
*****************************************************************************/
FSTATIC FLMBOOL flmVerifyElementChain(
	STATE_INFO *			pStateInfo,
	LFILE *					pLFile)
{
	RCODE						rc = NE_XFLM_OK;
	FLMBOOL					bResult = TRUE;
	F_BTREE_BLK_HDR *		pBlkHdr = (F_BTREE_BLK_HDR *)pStateInfo->pBlkHdr;
	FLMBYTE *				pucElmKey = pStateInfo->pucElmKey;
	FLMUINT					uiElmKeyLen = pStateInfo->uiElmKeyLen;
	FLMUINT					uiElmOffset = pStateInfo->uiElmOffset;
	FLMBYTE *				pucNextKey = NULL;
	FLMUINT					uiNextKeySize;
	FLMBYTE					ucFlags = pStateInfo->pucElm[ 0];
	F_CachedBlock *		pSCache = NULL;

	if (pLFile)
	{
		while (!(ucFlags & BTE_FLAG_LAST_ELEMENT))
		{
			// Get the next element.
			if (uiElmOffset >= (FLMUINT)(pBlkHdr->ui16NumKeys - 1))
			{
				if (pSCache)
				{
					ScaReleaseCache( pSCache, FALSE);
					pSCache = NULL;
				}

				// Will have to go the the next block to find the key we want.
				if (RC_BAD( rc = pStateInfo->pDb->getDatabase()->getBlock(
					pStateInfo->pDb, pLFile, pBlkHdr->stdBlkHdr.ui32NextBlkInChain,
					NULL, &pSCache)))
				{
					RC_UNEXPECTED_ASSERT( rc);
					bResult = FALSE;
					goto Exit;
				}

				flmAssert( pBlkHdr->stdBlkHdr.ui8BlkType ==
												pSCache->getBlockPtr()->ui8BlkType);

				pBlkHdr = (F_BTREE_BLK_HDR *)pSCache->getBlockPtr();
				uiElmOffset = 0;  // Get the first element...
			}
			else
			{
				uiElmOffset++;
			}

			pucNextKey = BtEntry( (FLMBYTE *)pBlkHdr, uiElmOffset);

			// Update the flag for the next iteration
			ucFlags = pucNextKey[ 0];

			if (ucFlags & BTE_FLAG_FIRST_ELEMENT)
			{
				flmAssert( 0);
				bResult = FALSE;
				goto Exit;
			}

			// Find the key.
			pucNextKey++;

			if (ucFlags & BTE_FLAG_KEY_LEN)
			{
				uiNextKeySize = FB2UW( pucNextKey);
				pucNextKey += 2;
			}
			else
			{
				uiNextKeySize = *pucNextKey;
				pucNextKey++;
			}

			// Key size is now set. They should match.
			if (uiElmKeyLen != uiNextKeySize)
			{
				flmAssert( 0);
				bResult = FALSE;
				goto Exit;
			}

			// Scoot past the info we don't need
			if (ucFlags & BTE_FLAG_DATA_LEN)
			{
				pucNextKey += 2;
			}
			else
			{
				pucNextKey++;
			}

			if (ucFlags & BTE_FLAG_OA_DATA_LEN)
			{
				pucNextKey += 4;
			}

			// Make sure the keys match.
			if ( f_memcmp( pucElmKey, pucNextKey, uiElmKeyLen) != 0)
			{
				flmAssert( 0);
				bResult = FALSE;
				goto Exit;
			}

		}
	}

Exit:

	if (pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}
	return TRUE;
}

/********************************************************************
Desc:	Build the result set of all keys in the database.
*********************************************************************/
RCODE F_DbCheck::buildIndexKeyList(
	FLMUINT64 *			pui64TotalKeys)
{
	RCODE						rc = NE_XFLM_OK;
	F_KeyCollector *		pKeyColl = NULL;
	DOC_IXD_XREF			XRef;
	IXD *						pIxd;
	LFILE *					pLFile;
	F_DOMNode *				pDocNode = NULL;
	F_Dict *					pDict;
	FLMBOOL					bUpdTranStarted = FALSE;
	RCODE						tmpRc = NE_XFLM_OK;
	FLMUINT					uiSizeRV;
	F_Btree *				pBTree = NULL;
	
	// Set information for the result set sort phase.
	
	m_Progress.i32CheckPhase = XFLM_CHECK_RS_SORT;
	m_Progress.bStartFlag = TRUE;

	if ((pKeyColl = f_new F_KeyCollector( this)) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	// Save the key collector in the m_pDb to redirect the
	// keys generated into the key result set.
	
	m_pDb->setKeyCollector( pKeyColl);

	// Start an update transaction, end the read trans.
	
	if (m_pDb->getTransType() == XFLM_READ_TRANS)
	{
		if (RC_OK( rc = m_pDb->transCommit()))
		{
			if (RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
			{
				goto Exit;
			}
			bUpdTranStarted = TRUE;
		}
		else
		{
			goto Exit;
		}
	}

	if (RC_BAD( rc = m_pDb->getDictionary( &pDict)))
	{
		goto Exit;
	}
	
	// For each entry in the Xref result set, call indexDocument to generate the
	// keys.

	if (RC_BAD( rc = m_pXRefRS->getBTree( NULL, NULL, &pBTree)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pXRefRS->getFirst( NULL, NULL, pBTree, (FLMBYTE *)&XRef,
		sizeof( XRef), &uiSizeRV, NULL, 0, NULL)))
	{
		goto Exit;
	}

	for( ;;)
	{
		if( RC_BAD( rc = m_pDb->getNode( XRef.uiCollection, XRef.ui64DocId,
			&pDocNode)))
		{
			goto Exit;
		}

		if( pDocNode->getDocumentId() == XRef.ui64DocId)
		{
			if( RC_BAD( rc = pDict->getIndex( XRef.uiIndexNum, &pLFile, &pIxd)))
			{
				// If the index is offline, skip it and move to the next, if any

				if( rc == NE_XFLM_INDEX_OFFLINE)
				{
					if( RC_BAD( rc = m_pXRefRS->getNext( NULL, NULL, pBTree,
						(FLMBYTE *)&XRef, sizeof( XRef), &uiSizeRV, NULL,
						0, NULL)))
					{
						if( rc == NE_XFLM_NOT_FOUND || rc == NE_XFLM_EOF_HIT)
						{
							rc = NE_XFLM_OK;
						}
						
						break;
					}
					
					continue;
				}
				
				goto Exit;
			}

			if (RC_BAD( rc = m_pDb->indexDocument( pIxd, pDocNode)))
			{
				goto Exit;
			}

			m_Progress.ui64NumKeys++; // += ui64KeysProcessed;
		}

		// Get the next document...
		
		if (RC_BAD( rc = m_pXRefRS->getNext( NULL, NULL, pBTree,
			(FLMBYTE *)&XRef, sizeof( XRef), &uiSizeRV, NULL, 0, NULL)))
		{
			if (rc == NE_XFLM_NOT_FOUND || rc == NE_XFLM_EOF_HIT)
			{
				rc = NE_XFLM_OK;
				break;
			}
			else
			{
				goto Exit;
			}
		}

		if( RC_BAD( rc = chkCallProgFunc()))
		{
			goto Exit;
		}
	}

	m_pXRefRS->freeBTree( &pBTree);

	if( bUpdTranStarted)
	{
		if (RC_BAD( rc = m_pDb->transCommit()))
		{
			goto Exit;
		}

		bUpdTranStarted = FALSE;

		if( RC_BAD( rc = m_pDb->transBegin( XFLM_READ_TRANS)))
		{
			goto Exit;
		}
	}

	*pui64TotalKeys = pKeyColl->getTotalKeys();

Exit:

	if( pBTree)
	{
		m_pXRefRS->freeBTree( &pBTree);
	}

	if( bUpdTranStarted)
	{
		// If we fail here, the whole thing should abort.
		
		if( RC_OK( rc))
		{
			if( RC_OK ( rc = m_pDb->transCommit()))
			{
				rc = m_pDb->transBegin( XFLM_READ_TRANS);
			}
			else
			{
				m_pDb->transAbort();
			}
		}
		else
		{
			m_pDb->transAbort();
			if( RC_BAD( tmpRc  = m_pDb->transBegin( XFLM_READ_TRANS)))
			{
				rc = tmpRc;
			}
		}
	}

	if( pDocNode)
	{
		pDocNode->Release();
	}

	// Be sure we don't leave it this way.
	
	m_pDb->setKeyCollector( NULL);

	if( pKeyColl)
	{
		pKeyColl->Release();
	}

	m_Progress.bStartFlag = TRUE;
	return( rc);
}

/***************************************************************************
Desc:	This routine checks all of the B-TREES in the database -- all
		indexes and containers.
*****************************************************************************/
RCODE F_DbCheck::verifyBTrees(
	FLMBOOL *	pbStartOverRV)
{
	RCODE		  						rc = NE_XFLM_OK;
	FLMUINT							uiCurrLf;
	FLMUINT							uiCurrLevel;
	FLMBYTE *						pucKeyBuffer = NULL;
	FLMUINT							uiKeysAllocated = 0;
	STATE_INFO						State [BH_MAX_LEVELS];
	FLMBOOL							bStateInitialized [BH_MAX_LEVELS];
	FLMBYTE *						pucResetKey = NULL;
	FLMUINT							uiResetKeyLen = ~(FLMUINT)0;
	FLMUINT64						ui64ResetNodeId = 0;
	LF_HDR *							pLogicalFile;
	FLMUINT							uiSaveDictSeq;
	FLMBOOL							bRSFinalized = FALSE;
	char								szTmpIoPath [F_PATH_MAX_SIZE];
	char								szBaseName [F_FILENAME_SIZE];
	F_NodeVerifier *				pNodeVerifier = NULL;
	F_BLK_HDR *						pBlkHdr = NULL;
	F_CachedBlock *				pSCache = NULL;
	
	f_memset( State, 0, sizeof( State));

	// The StateInfo structs may have pointer to this object,
	// but we only need one instance, so do the new here, rather
	// than inside the loop where we initialize the StateInfo structs.
	
	if( (pNodeVerifier = f_new F_NodeVerifier) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	// szTmpIoPath is the directory where the result sets containing
	// node data will be stored.
	
	if (RC_BAD( rc = gv_pXFlmDbSystem->getTempDir( szTmpIoPath)))
	{
		if (rc == NE_FLM_IO_PATH_NOT_FOUND ||
			 rc == NE_FLM_IO_INVALID_FILENAME)
		{
			if (RC_BAD( rc = gv_XFlmSysData.pFileSystem->pathReduce( 
				m_pDb->m_pDatabase->m_pszDbPath, szTmpIoPath, szBaseName)))
			{
				goto Exit;
			}
		}
		else
		{
			goto Exit;
		}
	}

	for (uiCurrLevel = 0; uiCurrLevel < BH_MAX_LEVELS; uiCurrLevel++)
	{
		bStateInitialized [uiCurrLevel] = FALSE;
	}

	if (*pbStartOverRV)
	{
		goto Exit;
	}

	uiSaveDictSeq = m_pDb->m_pDict->getDictSeq();
	if (RC_BAD( rc = setupLfTable()))
	{
		goto Exit;
	}

	if (m_uiFlags & XFLM_DO_LOGICAL_CHECK)
	{
		if (RC_BAD( rc = setupIxInfo()))
		{
			goto Exit;
		}
	}

	// Loop through all of the logical files in the database
	// and perform a structural and logical check.
	
	m_pIxd = NULL;
	uiCurrLf = 0;
	while (uiCurrLf < m_pDbInfo->m_uiNumLogicalFiles)
	{
		m_Progress.ui32CurrLF = (FLMUINT32)(uiCurrLf + 1);
		pLogicalFile = &m_pDbInfo->m_pLogicalFiles[uiCurrLf];
		
		if (pLogicalFile->eLfType == XFLM_LF_COLLECTION)
		{
			if (RC_BAD( rc = m_pDb->m_pDict->getCollection( pLogicalFile->uiLfNum,
																			&m_pCollection, TRUE)))
			{
				
				goto Exit;
			}
			m_pLFile = &m_pCollection->lfInfo;
			m_pIxd = NULL;
		}
		else
		{
			
			// If this is our first index, and we are doing a logical check,
			// create the index key result set from all of the documents we
			// have created.
	
			if (!m_bPhysicalCorrupt &&
				 (m_uiFlags & XFLM_DO_LOGICAL_CHECK) &&
				 !bRSFinalized)
			{
				FLMUINT64	ui64NumRSKeys = 0;
	
				if (RC_BAD( rc = buildIndexKeyList( &ui64NumRSKeys)))
				{
					if (rc == NE_XFLM_EOF_HIT && ui64NumRSKeys == 0)
					{
						rc = NE_XFLM_OK;
					}
					else
					{
						goto Exit;
					}
				}
	
				// Reset uiNumKeys to reflect the number of keys
				// in the result set now that all duplicates have
				// been eliminated.
	
				if (m_Progress.ui64NumKeys > ui64NumRSKeys)
				{
					m_Progress.ui64NumDuplicateKeys =
						m_Progress.ui64NumKeys - ui64NumRSKeys;
				}
				m_Progress.ui64NumKeys = ui64NumRSKeys;
	
				// Set bRSFinalized to TRUE so that subsequent passes will not
				// attempt to finalize the result set again.
	
				bRSFinalized = TRUE;
			}

			if (RC_BAD( rc = m_pDb->m_pDict->getIndex( pLogicalFile->uiLfNum,
															&m_pLFile, &m_pIxd, TRUE)))
			{
				goto Exit;
			}
			m_pCollection = NULL;
		}
		pLogicalFile->uiRootBlk = m_pLFile->uiRootBlk;
		flmAssert( m_pLFile->uiRootBlk);

		// Allocate space to hold the keys, if not already allocated.

		if (uiKeysAllocated < pLogicalFile->uiNumLevels)
		{
			if (RC_BAD( rc = f_realloc( pLogicalFile->uiNumLevels * XFLM_MAX_KEY_SIZE,
										&pucKeyBuffer)))
			{
				goto Exit;
			}
			uiKeysAllocated = pLogicalFile->uiNumLevels;
		}

		// Setup XFLM_PROGRESS_CHECK_INFO structure

		m_Progress.i32CheckPhase = XFLM_CHECK_B_TREE;
		m_Progress.bStartFlag = TRUE;
		m_Progress.ui32LfNumber = (FLMUINT32)m_pLFile->uiLfNum;
		m_Progress.ui32LfType = (FLMUINT32)m_pLFile->eLfType;

		if (RC_BAD( rc = chkCallProgFunc()))
		{
			break;
		}

		m_Progress.bStartFlag = FALSE;

		f_yieldCPU();

		// Initialize the state information for each level of the B-TREE.

		for (uiCurrLevel = 0;
			  uiCurrLevel < pLogicalFile->uiNumLevels;
			  uiCurrLevel++)
		{
			FLMUINT	uiExpectedBlkType;

			// If we are resetting to a particular key, save the statistics
			// which were gathered so far.

			if (uiResetKeyLen != ~(FLMUINT)0)
			{

				// Save the statistics which were gathered.

				pLogicalFile->pLevelInfo [uiCurrLevel].ui64KeyCount =
														State [uiCurrLevel].ui64KeyCount;

				f_memcpy( &pLogicalFile->pLevelInfo [uiCurrLevel].BlockInfo,
							 &State [uiCurrLevel].BlkInfo,
							 sizeof( BLOCK_INFO));
			}

			if (m_pLFile->eLfType == XFLM_LF_INDEX)
			{
				if (uiCurrLevel == 0)
				{
					if (m_pIxd->uiNumDataComponents)
					{
						uiExpectedBlkType = BT_LEAF_DATA;
					}
					else
					{
						uiExpectedBlkType = BT_LEAF;
					}
				}
				else if (m_pIxd->uiFlags & IXD_ABS_POS)
				{
					uiExpectedBlkType = BT_NON_LEAF_COUNTS;
				}
				else
				{
					uiExpectedBlkType = BT_NON_LEAF;
				}
			}
			else // collection BTree...
			{
				if (uiCurrLevel == 0)
				{
					uiExpectedBlkType = BT_LEAF_DATA;
				}
				else
				{
					uiExpectedBlkType = BT_NON_LEAF;
				}

			}

			flmInitReadState( &State [uiCurrLevel],
									&bStateInitialized [uiCurrLevel],
									(FLMUINT)m_pDb->
										m_pDatabase->m_lastCommittedDbHdr.ui32DbVersion,
									m_pDb,
									pLogicalFile,
									uiCurrLevel,
									uiExpectedBlkType,
									&pucKeyBuffer [uiCurrLevel * XFLM_MAX_KEY_SIZE]);

			State[ uiCurrLevel].pCollection = m_pCollection;
			State[ uiCurrLevel].uiRootLevel = pLogicalFile->uiNumLevels - 1;
			State[ uiCurrLevel].uiCurrLf = uiCurrLf;

			if (uiResetKeyLen == ~(FLMUINT)0)
			{
				State [uiCurrLevel].ui32LastChildAddr = 0;
				State [uiCurrLevel].uiElmLastFlag = TRUE;
			}
			else
			{

				// Restore the statistics which were gathered so far.

				State [uiCurrLevel].ui64KeyCount =
					pLogicalFile->pLevelInfo [uiCurrLevel].ui64KeyCount;
				f_memcpy( &State [uiCurrLevel].BlkInfo,
							 &pLogicalFile->pLevelInfo [uiCurrLevel].BlockInfo,
							 sizeof( BLOCK_INFO));
			}
		}


		if (m_pLFile->eLfType == XFLM_LF_COLLECTION)
		{
			
			// Only leaf blocks of collections need a NodeVerifier object
			
			State[0].pNodeVerifier = pNodeVerifier;

			// If this is a collection BTree, create a result set to hold the pointer
			// information from all the nodes in this btree
			
			if (RC_BAD( rc = getBtResultSet( &State[ 0].pNodeRS)))
			{
				goto Exit;
			}

			// The nodeVerifier will setup the Node Result Set etc..
			
			if (pNodeVerifier)
			{
				pNodeVerifier->setupNodeRS( State[ 0].pNodeRS);
			}
		}

		if ((m_uiFlags & XFLM_DO_LOGICAL_CHECK) &&
			 State[ 0].pXRefRS == NULL)
		{
			if (m_pXRefRS == NULL)
			{
				if (RC_BAD( rc = getBtResultSet( &m_pXRefRS)))
				{
					goto Exit;
				}
		
			}

			State[ 0].pXRefRS = m_pXRefRS;

			// The nodeVerifier will setup the Node Result Set etc..
			
			if (pNodeVerifier)
			{
				pNodeVerifier->setupXRefRS( State[ 0].pXRefRS);
			}
		}

		// Call verifySubTree to check the B-TREE starting at the
		// root block.
Reset:

		rc = verifySubTree( NULL,
								  &State [pLogicalFile->uiNumLevels - 1],
								  m_pLFile->uiRootBlk,
								  &pucResetKey,
								  uiResetKeyLen,
								  ui64ResetNodeId);

		if (rc == NE_XFLM_RESET_NEEDED || rc == NE_XFLM_OLD_VIEW)
		{
			FLMUINT		uiNumLevels;

			if (rc == NE_XFLM_RESET_NEEDED)
			{
				m_LastStatusRc = NE_XFLM_OK;
			}

			// If it is a read transaction, reset.
			
			if (m_pDb->getTransType() == XFLM_READ_TRANS)
			{

				// Free the KrefCntrl

				m_pDb->krefCntrlFree();

				// Abort the read transaction

				if (RC_BAD( rc = m_pDb->transAbort()))
				{
					goto Exit;
				}

				// Try to start a new read transaction
			
				if (RC_BAD( rc = m_pDb->transBegin( XFLM_READ_TRANS,
																FLM_NO_TIMEOUT,
																XFLM_DONT_POISON_CACHE)))
				{
					goto Exit;
				}
			}
			
			// If we already have a reset key buffer, we need to free it.
			// We will start by repositioning to the root level key we were
			// last on.

			if (pucResetKey)
			{
				f_free( &pucResetKey);
			}

			uiResetKeyLen = State[ pLogicalFile->uiNumLevels - 1].uiElmKeyLen;
			ui64ResetNodeId = State[ pLogicalFile->uiNumLevels - 1].ui64ElmNodeId;

			if (RC_BAD( rc = f_calloc( uiResetKeyLen + 1, &pucResetKey)))
			{
				goto Exit;
			}

			f_memcpy( pucResetKey,
						 State[ pLogicalFile->uiNumLevels - 1].pucElmKey,
						 uiResetKeyLen);

			// On Reset, we may need to reget the LFILE and IXD.
			
			uiNumLevels = pLogicalFile->uiNumLevels;

			if (pLogicalFile->eLfType == XFLM_LF_COLLECTION)
			{
				if (RC_BAD( rc = m_pDb->m_pDict->getCollection(
						pLogicalFile->uiLfNum, &m_pCollection, TRUE)))
				{
					goto Exit;
				}
				m_pIxd = NULL;
				m_pLFile = &m_pCollection->lfInfo;
			}
			else
			{
				if (RC_BAD( rc = m_pDb->m_pDict->getIndex(
															pLogicalFile->uiLfNum,
															&m_pLFile, &m_pIxd, TRUE)))
				{
					goto Exit;
				}
				m_pCollection = NULL;
			}
			if (RC_BAD( rc = getLfInfo( pLogicalFile, m_pLFile)))
			{
				goto Exit;
			}
			
			if (uiNumLevels != pLogicalFile->uiNumLevels)
			{
				// Since the block structure of the BTree has been changed, we have
				// to begin our scan again from the top.  We need to gather stats too.
				uiResetKeyLen = ~(FLMUINT)0;
				if (pucResetKey)
				{
					f_free( &pucResetKey);
				}
				ui64ResetNodeId = 0;

				// Initialize the state information for each level of the B-TREE.
				for (uiCurrLevel = 0;
					uiCurrLevel < pLogicalFile->uiNumLevels;
					uiCurrLevel++)
				{
					FLMUINT	uiExpectedBlkType;

					// If we are resetting to a particular key, save the statistics
					// which were gathered so far.

					if (uiResetKeyLen != ~(FLMUINT)0)
					{

						// Save the statistics which were gathered.

						pLogicalFile->pLevelInfo [uiCurrLevel].ui64KeyCount =
																State [uiCurrLevel].ui64KeyCount;

						f_memcpy( &pLogicalFile->pLevelInfo [uiCurrLevel].BlockInfo,
									&State [uiCurrLevel].BlkInfo,
									sizeof( BLOCK_INFO));
					}

					if (m_pLFile->eLfType == XFLM_LF_INDEX)
					{
						if (uiCurrLevel == 0)
						{
							if (m_pIxd->uiNumDataComponents)
							{
								uiExpectedBlkType = BT_LEAF_DATA;
							}
							else
							{
								uiExpectedBlkType = BT_LEAF;
							}
						}
						else if (m_pIxd->uiFlags & IXD_ABS_POS)
						{
							uiExpectedBlkType = BT_NON_LEAF_COUNTS;
						}
						else
						{
							uiExpectedBlkType = BT_NON_LEAF;
						}
					}
					else // collection BTree...
					{
						if (uiCurrLevel == 0)
						{
							uiExpectedBlkType = BT_LEAF_DATA;
						}
						else
						{
							uiExpectedBlkType = BT_NON_LEAF;
						}

					}

					flmInitReadState( &State [uiCurrLevel],
											&bStateInitialized [uiCurrLevel],
											(FLMUINT)m_pDb->
												m_pDatabase->m_lastCommittedDbHdr.ui32DbVersion,
											m_pDb,
											pLogicalFile,
											uiCurrLevel,
											uiExpectedBlkType,
											&pucKeyBuffer [uiCurrLevel * XFLM_MAX_KEY_SIZE]);

					State[ uiCurrLevel].pCollection = m_pCollection;
					State[ uiCurrLevel].uiRootLevel = pLogicalFile->uiNumLevels - 1;
					State[ uiCurrLevel].uiCurrLf = uiCurrLf;

					if (uiResetKeyLen == ~(FLMUINT)0)
					{
						State [uiCurrLevel].ui32LastChildAddr = 0;
						State [uiCurrLevel].uiElmLastFlag = TRUE;
					}
					else
					{

						// Restore the statistics which were gathered so far.

						State [uiCurrLevel].ui64KeyCount =
							pLogicalFile->pLevelInfo [uiCurrLevel].ui64KeyCount;
						f_memcpy( &State [uiCurrLevel].BlkInfo,
									&pLogicalFile->pLevelInfo [uiCurrLevel].BlockInfo,
									sizeof( BLOCK_INFO));
					}
				}
			}

			goto Reset;
		}

		if (RC_BAD( rc))
		{
			goto Exit;
		}

		// If this was a collection, then go through the result set and verify
		// all of the node pointers...
		
		if (State[ 0].pNodeRS)
		{
			FLMINT32			i32ErrCode;


			// Setup the current progress phase
			
			m_Progress.i32CheckPhase = XFLM_CHECK_DOM_LINKS;
			m_Progress.bStartFlag = TRUE;
			m_Progress.ui32LfNumber = (FLMUINT32)m_pLFile->uiLfNum;
			m_Progress.ui32LfType = (FLMUINT32)m_pLFile->eLfType;

			if (RC_BAD( rc = chkCallProgFunc()))
			{
				break;
			}

			m_Progress.bStartFlag = FALSE;

			f_yieldCPU();

			m_LastStatusRc = verifyNodePointers( &State[ 0], &i32ErrCode);

			if (i32ErrCode)
			{
				chkReportError( i32ErrCode,
									 XFLM_LOCALE_B_TREE,
									 (FLMUINT32)m_Progress.ui32LfNumber,
									 (FLMUINT32)m_Progress.ui32LfType,
									 (FLMUINT32)State[ 0].uiLevel,
									 (FLMUINT32)m_pLFile->uiBlkAddress,
									 0,
									 0,
									 0);
			}

			State[ 0].pNodeRS->Release();
			State[ 0].pNodeRS = NULL;
			if (pNodeVerifier)
			{
				
				// Resets the Node verifier RS to NULL.
				
				pNodeVerifier->setupNodeRS( State[ 0].pNodeRS);
			}
		}

		// Verify that all of the levels' next block address's are 0.

		if (RC_OK( m_LastStatusRc))
		{
			for (uiCurrLevel = 0;
				  uiCurrLevel < pLogicalFile->uiNumLevels;
				  uiCurrLevel++)
			{

				// Save the statistics which were gathered.

				pLogicalFile->pLevelInfo [uiCurrLevel].ui64KeyCount =
					State [uiCurrLevel].ui64KeyCount;
				f_memcpy( &pLogicalFile->pLevelInfo [uiCurrLevel].BlockInfo,
							 &State [uiCurrLevel].BlkInfo, sizeof( BLOCK_INFO));

				// Make sure the last block had a NEXT block address of 0.

				if (State [uiCurrLevel].ui32NextBlkAddr != 0xFFFFFFFF &&
					 State [uiCurrLevel].ui32NextBlkAddr != 0)
				{
					FLMINT32			i32BlkErrCode;

					// Verify our finding.  Get the block in question and see
					// if it realy has a problem.

					if (RC_BAD( rc = 	blkRead( State[ uiCurrLevel].ui32BlkAddress,
							&pBlkHdr, &pSCache, &i32BlkErrCode)))
					{
						// Log the error.

						if (i32BlkErrCode)
						{
							chkReportError( i32BlkErrCode, XFLM_LOCALE_LFH_LIST,
								0, 0, 0xFF, State[ uiCurrLevel].ui32BlkAddress,
								0, 0, 0);
						}
						goto Exit;
					}

					if (pBlkHdr->ui32NextBlkInChain != 0)
					{
						chkReportError( FLM_BAD_LAST_BLK_NEXT, XFLM_LOCALE_B_TREE,
							(FLMUINT32)m_Progress.ui32LfNumber,
							(FLMUINT32)m_Progress.ui32LfType,
							(FLMUINT32)uiCurrLevel,
							0, 0, 0, 0);
					}
					
					ScaReleaseCache( pSCache, FALSE);
					pSCache = NULL;
					pBlkHdr = NULL;

				}
			}
		}

		if (RC_BAD( m_LastStatusRc))
		{
			break;
		}
		
		// If we are doing a logical index check, need to see if we used up
		// all of the keys in the result set for this index.
		
		if (pLogicalFile->eLfType == XFLM_LF_INDEX &&
			 !m_bPhysicalCorrupt &&
			 (m_uiFlags & XFLM_DO_LOGICAL_CHECK) &&
			 bRSFinalized)
		{
			for (;;)
			{
				if (RC_BAD( rc = chkGetNextRSKey()))
				{
					if (rc == NE_XFLM_EOF_HIT || rc == NE_XFLM_NOT_FOUND)
					{
						rc = NE_XFLM_OK;
						break;
					}
					goto Exit;
				}
				else
				{
					
					// Updated statistics
					
					m_Progress.ui64NumKeysExamined++;
	
					if (RC_BAD( rc = resolveIXMissingKey( &(State[ 0]))))
					{
						goto Exit;
					}
				}
			}
		}

		uiCurrLf++;
		if (pucResetKey)
		{
			f_free( &pucResetKey);
		}
		pucResetKey = NULL;
		uiResetKeyLen = ~(FLMUINT)0;
		ui64ResetNodeId = 0;
	}

Exit:

	if (pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}
	else if (pBlkHdr)
	{
		f_free( &pBlkHdr);
	}
	
	if (pucKeyBuffer)
	{
		f_free( &pucKeyBuffer);
	}

	if (pucResetKey)
	{
		f_free( &pucResetKey);
	}

	if (State[ 0].pNodeRS)
	{
		State[ 0].pNodeRS->Release();
		State[ 0].pNodeRS = NULL;
		if (pNodeVerifier)
		{
			pNodeVerifier->setupNodeRS( NULL);
		}
	}

	// Clean up the NodeVerifier instance...

	if (pNodeVerifier)
	{
		pNodeVerifier->Release();
	}
	
	// Cleanup any temporary index check files

	if (m_pIxRSet)
	{
		m_pIxRSet->Release();
		m_pIxRSet = NULL;
	}
	if (m_puiIxArray)
	{
		f_free( &m_puiIxArray);
	}

	if (RC_OK( rc) && RC_BAD( m_LastStatusRc))
	{
		rc = m_LastStatusRc;
	}

	return( rc);
}



/***************************************************************************
Desc:	This routine allocates and initializes the LF table (array of
		LF_HDR structures).
*****************************************************************************/
RCODE F_DbCheck::setupLfTable()
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiLfHdrOffset;
	IXD *					pIxd;
	F_COLLECTION *		pCollection;
	F_Dict *				pDict = m_pDb->m_pDict;
	FLMUINT				uiIndexNum;
	FLMUINT				uiCollectionNum;

	// Set up the table such that the dictionary is checked first,
	// followed by data containers, and then indexes.  This is
	// necessary for the logical (index) check to work.  The
	// data records must be extracted before the indexes are
	// checked so that the temporary result set, used during
	// the logical check, can be built.
	
	m_pDbInfo->freeLogicalFiles();
	m_Progress.ui32NumLFs = 0;

	if (pDict)
	{

		// Count the indexes.

		uiIndexNum = 0;
		for (;;)
		{
			if ((pIxd = pDict->getNextIndex( uiIndexNum, TRUE)) == NULL)
			{
				break;
			}
			uiIndexNum = pIxd->uiIndexNum;
			m_pDbInfo->m_uiNumIndexes++;
		}

		// Count the collections

		uiCollectionNum = 0;
		for (;;)
		{
			if ((pCollection = pDict->getNextCollection( uiCollectionNum,
										TRUE)) == NULL)
			{
				break;
			}
			uiCollectionNum = pCollection->lfInfo.uiLfNum;
			m_pDbInfo->m_uiNumCollections++;
		}

		m_pDbInfo->m_uiNumLogicalFiles = m_pDbInfo->m_uiNumIndexes +
											  m_pDbInfo->m_uiNumCollections;
		m_Progress.ui32NumLFs = (FLMUINT32)m_pDbInfo->m_uiNumLogicalFiles;

		// Allocate memory for each collection and index, then set up each
		// collection and index

		if (RC_BAD( rc = f_calloc( (FLMUINT)(sizeof( LF_HDR) *
										m_pDbInfo->m_uiNumLogicalFiles),
										(void **)&m_pDbInfo->m_pLogicalFiles)))
		{
			goto Exit;
		}

		uiLfHdrOffset = 0;

		// Add in our dictionary collection first.  Do field
		// definitions first, then collection definitions, then index
		// definitions.

		if (RC_BAD( rc = pDict->getCollection( XFLM_DICT_COLLECTION, &pCollection)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = getLfInfo( &m_pDbInfo->m_pLogicalFiles [uiLfHdrOffset],
											 &pCollection->lfInfo)))
		{
			goto Exit;
		}
		uiLfHdrOffset++;

		// Add in default data collection

		if (RC_BAD( rc = pDict->getCollection( XFLM_DATA_COLLECTION, &pCollection)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = getLfInfo( &m_pDbInfo->m_pLogicalFiles [uiLfHdrOffset],
											 &pCollection->lfInfo)))
		{
			goto Exit;
		}
		uiLfHdrOffset++;

		// Add in the maintenance collection

		if (RC_BAD( rc = pDict->getCollection( XFLM_MAINT_COLLECTION, &pCollection)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = getLfInfo( &m_pDbInfo->m_pLogicalFiles [uiLfHdrOffset],
											 &pCollection->lfInfo)))
		{
			goto Exit;
		}
		uiLfHdrOffset++;

		// Add in user defined collections

		uiCollectionNum = 0;
		for (;;)
		{
			if ((pCollection = pDict->getNextCollection( uiCollectionNum,
									FALSE)) == NULL)
			{
				break;
			}
			uiCollectionNum = pCollection->lfInfo.uiLfNum;
			if (RC_BAD( rc = getLfInfo( &m_pDbInfo->m_pLogicalFiles [uiLfHdrOffset],
												 &pCollection->lfInfo)))
			{
				goto Exit;
			}
			uiLfHdrOffset++;
		}

		// Indexes need to be in order from lowest to highest
		// because the result set is ordered that way.

		uiIndexNum = 0;
		for (;;)
		{
			if ((pIxd = pDict->getNextIndex( uiIndexNum, TRUE)) == NULL)
			{
				break;
			}
			uiIndexNum = pIxd->uiIndexNum;
			if (RC_BAD( rc = getLfInfo( &m_pDbInfo->m_pLogicalFiles [uiLfHdrOffset],
												 &pIxd->lfInfo)))
			{
				goto Exit;
			}
			uiLfHdrOffset++;
		}
	}

Exit:

	return( rc);
}


/***************************************************************************
Desc:	Initializes index checking information.
*****************************************************************************/
RCODE F_DbCheck::setupIxInfo( void)
{
	RCODE			rc = NE_XFLM_OK;
	LF_HDR *		pLogicalFile;
	FLMUINT		uiLoop;
	FLMUINT		uiIxCount;

	// Initialize the result set.  The result set will be used
	// to build an ordered list of keys for comparison to
	// the database's indexes.

	if (RC_BAD( rc = getBtResultSet( &m_pIxRSet)))
	{
		goto Exit;
	}

	// Build list of all indexes

	if (m_pDbInfo->m_uiNumIndexes)
	{

		// Allocate memory to save each index number

		if (RC_BAD( rc = f_alloc( 
					(FLMUINT)(sizeof( FLMUINT) * m_pDbInfo->m_uiNumIndexes),
					&m_puiIxArray)))
		{
			goto Exit;
		}

		// Save the index numbers into the array.

		uiIxCount = 0;
		pLogicalFile = m_pDbInfo->m_pLogicalFiles;
		for( uiLoop = 0;
			  uiLoop < m_pDbInfo->m_uiNumLogicalFiles;
			  uiLoop++, pLogicalFile++)
		{
			if (pLogicalFile->eLfType == XFLM_LF_INDEX)
			{
				m_puiIxArray[ uiIxCount] = pLogicalFile->uiLfNum;
				uiIxCount++;
			}
		}
	}

	m_bGetNextRSKey = TRUE;

Exit:

	// Clean up any memory on error exit.

	if (RC_BAD( rc))
	{
		if (m_pIxRSet)
		{
			m_pIxRSet->Release();
			m_pIxRSet = NULL;
		}
		if (m_puiIxArray)
		{
			f_free( &m_puiIxArray);
		}
	}

	return( rc);
}

/***************************************************************************
Desc:	This routine reads the LFH areas from disk to make sure they are up
		to date in memory.
*****************************************************************************/
RCODE F_DbCheck::getLfInfo(
	LF_HDR *	pLogicalFile,
	LFILE *	pLFile
	)
{
	RCODE					rc = NE_XFLM_OK;
	F_CachedBlock *	pSCache = NULL;
	F_BLK_HDR *			pBlkHdr = NULL;
	FLMUINT				uiSaveLevel;
	FLMINT32				i32BlkErrCode;

	pLogicalFile->eLfType = pLFile->eLfType;
	pLogicalFile->uiLfNum = pLFile->uiLfNum;
	pLogicalFile->uiRootBlk = pLFile->uiRootBlk;
	
	// Read in the block containing the logical file header.

	if (RC_BAD( rc = blkRead( pLFile->uiBlkAddress,
							&pBlkHdr, &pSCache, &i32BlkErrCode)))
	{

		// Log the error.

		if (i32BlkErrCode)
		{
			chkReportError( i32BlkErrCode,
								 XFLM_LOCALE_LFH_LIST,
								 0,
								 0,
								 0xFF,
								 (FLMUINT32)pLFile->uiBlkAddress,
								 0,
								 0,
								 0);
		}
		goto Exit;
	}
	uiSaveLevel = pLogicalFile->uiNumLevels;
	
	// Read root block to get the number of levels in the B-TREE

	flmAssert( pLFile->uiRootBlk);
	if (RC_BAD( rc = blkRead( pLFile->uiRootBlk,
									  &pBlkHdr,
									  &pSCache,
									  &i32BlkErrCode)))
	{
		if (i32BlkErrCode)
		{
			chkReportError( i32BlkErrCode,
								 XFLM_LOCALE_B_TREE,
								 (FLMUINT32)pLFile->uiLfNum,
								 (FLMUINT32)pLFile->eLfType,
								 0xFF,
								 (FLMUINT32)pLFile->uiRootBlk,
								 0,
								 0,
								 0);
		}
		goto Exit;
	}
	pLogicalFile->uiNumLevels =
		(FLMUINT)(((F_BTREE_BLK_HDR *)pBlkHdr)->ui8BlkLevel) + 1;

	// Need to make sure that the level extracted from
	// the block is valid.

	if (((F_BTREE_BLK_HDR *)pBlkHdr)->ui8BlkLevel >= BH_MAX_LEVELS)
	{
		chkReportError( FLM_BAD_BLK_HDR_LEVEL,
							 XFLM_LOCALE_B_TREE,
							 (FLMUINT32)pLFile->uiLfNum,
							 (FLMUINT32)pLFile->eLfType,
							 (FLMUINT32)(((F_BTREE_BLK_HDR *)pBlkHdr)->ui8BlkLevel),
							 (FLMUINT32)pLFile->uiRootBlk,
							 0,
							 0,
							 0);

		// Force pLogicalFile->uiNumLevels to 1 so that we don't crash

		pLogicalFile->uiNumLevels = 1;
	}

	// If the number of levels changed, reset the level information
	// on this logical file.

	if (uiSaveLevel != pLogicalFile->uiNumLevels &&
		 pLogicalFile->uiNumLevels)
	{
		if (pLogicalFile->uiNumLevels > uiSaveLevel)
		{
			if ( pLogicalFile->pLevelInfo)
			{
				f_free( &pLogicalFile->pLevelInfo);
			}
			if (RC_BAD( rc = f_calloc( 
						(sizeof( LEVEL_INFO) * pLogicalFile->uiNumLevels),
						(void **)&pLogicalFile->pLevelInfo)))
			{
				goto Exit;
			}
		}
	}

Exit:

	if (pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}
	else if (pBlkHdr)
	{
		f_free( &pBlkHdr);
	}

	return( rc);
}


/***************************************************************************
Desc:	Goes throught the (finalized) result set and validates that all the
		node pointers point to the right nodes
*****************************************************************************/
RCODE F_DbCheck::verifyNodePointers(
	STATE_INFO *					pStateInfo,
	FLMINT32 *						pi32ErrCode
	)
{
	RCODE								rc = NE_XFLM_OK;
	NODE_RS_ENTRY *				pRSEntry = NULL;
	NODE_RS_ENTRY *				pTmpRSEntry = NULL;
	F_BtResultSet *				pResult = pStateInfo->pNodeRS;
	FLMUINT							uiRSEntrySize = sizeof( NODE_RS_ENTRY);
	FLMBOOL							bFirst = TRUE;
	FLMBYTE							pucKey[ XFLM_MAX_KEY_SIZE];
	FLMUINT							uiKeyLength = XFLM_MAX_KEY_SIZE;
	F_Btree *						pBTree = NULL;
	FLMINT32							i32ErrCode = 0;

	*pi32ErrCode = 0;

	if (RC_BAD( rc = f_calloc( sizeof( NODE_RS_ENTRY), &pRSEntry)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = f_calloc( sizeof( NODE_RS_ENTRY), &pTmpRSEntry)))
	{
		goto Exit;
	}

	for (;;)
	{
		m_Progress.ui64NumDomNodes++;;

		if (bFirst)
		{

			if (RC_BAD( rc = pResult->getBTree( NULL, NULL, &pBTree)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = pResult->getFirst( NULL, NULL, pBTree, 
															pucKey,
															XFLM_MAX_KEY_SIZE,
															&uiKeyLength,
															(FLMBYTE *)pRSEntry,
															sizeof( NODE_RS_ENTRY),
															&uiRSEntrySize)))
			{
				if (rc == NE_XFLM_EOF_HIT || rc == NE_XFLM_BOF_HIT)
				{
					rc = NE_XFLM_OK;
					break;
				}

				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = pResult->getNext( NULL, NULL, pBTree, 
														  pucKey,
														  XFLM_MAX_KEY_SIZE,
														  &uiKeyLength,
														  (FLMBYTE *)pRSEntry,
														  sizeof( NODE_RS_ENTRY),
														  &uiRSEntrySize)))
			{
				if (rc == NE_XFLM_EOF_HIT || rc == NE_XFLM_BOF_HIT)
				{
					rc = NE_XFLM_OK;
					break;
				}

				goto Exit;
			}
		}
		bFirst = FALSE;

		if (RC_BAD( rc = verifyRootLink(
						pRSEntry, uiRSEntrySize, pTmpRSEntry, pResult, &i32ErrCode)))
		{
			goto Exit;
		}

		if (i32ErrCode)
		{
			chkReportError( i32ErrCode, XFLM_LOCALE_B_TREE,
						(FLMUINT32)m_Progress.ui32LfNumber,
						(FLMUINT32)m_Progress.ui32LfType, 0, 0, 0, FLM_MAX_UINT32,
						pRSEntry->hdr.ui64NodeId);
			i32ErrCode = 0;
			m_Progress.ui64NumBrokenDomLinks++;
		}
		else
		{
			m_Progress.ui64NumDomLinksVerified++;
		}

		if (RC_BAD( rc = verifyParentLink(
						pRSEntry, pTmpRSEntry, pResult, &i32ErrCode)))
		{
			goto Exit;
		}

		if (i32ErrCode)
		{
			chkReportError( i32ErrCode, XFLM_LOCALE_B_TREE,
						(FLMUINT32)m_Progress.ui32LfNumber,
						(FLMUINT32)m_Progress.ui32LfType, 0, 0, 0, FLM_MAX_UINT32,
						pRSEntry->hdr.ui64NodeId);
			i32ErrCode = 0;
			m_Progress.ui64NumBrokenDomLinks++;
		}
		else
		{
			m_Progress.ui64NumDomLinksVerified++;
		}

		if( RC_BAD( rc = verifyFirstChildLink( pRSEntry, pTmpRSEntry, 
			pResult, &i32ErrCode)))
		{
			goto Exit;
		}

		if( i32ErrCode)
		{
			chkReportError( i32ErrCode, XFLM_LOCALE_B_TREE,
						(FLMUINT32)m_Progress.ui32LfNumber,
						(FLMUINT32)m_Progress.ui32LfType, 0, 0, 0, FLM_MAX_UINT32,
						pRSEntry->hdr.ui64NodeId);
			i32ErrCode = 0;
			m_Progress.ui64NumBrokenDomLinks++;
		}
		else
		{
			m_Progress.ui64NumDomLinksVerified++;
		}

		if (RC_BAD( rc = verifyLastChildLink(
						pRSEntry, pTmpRSEntry, pResult, &i32ErrCode)))
		{
			goto Exit;
		}

		if (i32ErrCode)
		{
			chkReportError( i32ErrCode, XFLM_LOCALE_B_TREE,
						(FLMUINT32)m_Progress.ui32LfNumber,
						(FLMUINT32)m_Progress.ui32LfType, 0, 0, 0, FLM_MAX_UINT32,
						pRSEntry->hdr.ui64NodeId);
			i32ErrCode = 0;
			m_Progress.ui64NumBrokenDomLinks++;
		}
		else
		{
			m_Progress.ui64NumDomLinksVerified++;
		}

		if (RC_BAD( rc = verifyPrevSiblingLink(
						pRSEntry, pTmpRSEntry, pResult, &i32ErrCode)))
		{
			goto Exit;
		}

		if (i32ErrCode)
		{
			chkReportError( i32ErrCode, XFLM_LOCALE_B_TREE,
						(FLMUINT32)m_Progress.ui32LfNumber,
						(FLMUINT32)m_Progress.ui32LfType, 0, 0, 0, FLM_MAX_UINT32,
						pRSEntry->hdr.ui64NodeId);
			i32ErrCode = 0;
			m_Progress.ui64NumBrokenDomLinks++;
		}
		else
		{
			m_Progress.ui64NumDomLinksVerified++;
		}

		if (RC_BAD( rc = verifyNextSiblingLink(
						pRSEntry, pTmpRSEntry, pResult, &i32ErrCode)))
		{
			goto Exit;
		}

		if (i32ErrCode)
		{
			chkReportError( i32ErrCode, XFLM_LOCALE_B_TREE,
						(FLMUINT32)m_Progress.ui32LfNumber,
						(FLMUINT32)m_Progress.ui32LfType, 0, 0, 0, FLM_MAX_UINT32,
						pRSEntry->hdr.ui64NodeId);
			i32ErrCode = 0;
			m_Progress.ui64NumBrokenDomLinks++;
		}
		else
		{
			m_Progress.ui64NumDomLinksVerified++;
		}

		if (RC_BAD( rc = verifyAnnotationLink(
						pRSEntry, pTmpRSEntry, pResult, &i32ErrCode)))
		{
			goto Exit;
		}

		if (i32ErrCode)
		{
			chkReportError( i32ErrCode, XFLM_LOCALE_B_TREE,
						(FLMUINT32)m_Progress.ui32LfNumber,
						(FLMUINT32)m_Progress.ui32LfType, 0, 0, 0, FLM_MAX_UINT32,
						pRSEntry->hdr.ui64NodeId);
			i32ErrCode = 0;
			m_Progress.ui64NumBrokenDomLinks++;
		}
		else
		{
			m_Progress.ui64NumDomLinksVerified++;
		}

		if (RC_BAD( rc = chkCallProgFunc()))
		{
			break;
		}

		f_yieldCPU();

	}

Exit:

	if (pBTree)
	{
		pResult->freeBTree( &pBTree);
	}

	if (pRSEntry)
	{
		f_free( &pRSEntry);
	}

	if (pTmpRSEntry)
	{
		f_free( &pTmpRSEntry);
	}

	return rc;

}

/********************************************************************
Desc:	Verify that the Root Id field points to a valid node.
********************************************************************/
FSTATIC RCODE verifyRootLink(
	NODE_RS_ENTRY *		pRSEntry,
	FLMUINT					uiRSEntrySize,
	NODE_RS_ENTRY *		pTmpRSEntry,
	F_BtResultSet *		pResult,
	FLMINT32 *				pi32ErrCode
	)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT64		ui64RootId = getLinkVal( CHK_BM_ROOT_ID, pRSEntry);
	FLMUINT			uiTmpRSEntrySize;
	FLMUINT			uiKeySize = sizeof( FLMUINT64);

	f_memset( pTmpRSEntry, 0, sizeof(NODE_RS_ENTRY));

	if (!ui64RootId)
	{
		// Returns NE_XFLM_OK
		goto Exit;
	}

	if( ui64RootId != pRSEntry->hdr.ui64NodeId)
	{
		pTmpRSEntry->hdr.ui64NodeId = ui64RootId;

		if (RC_BAD( rc = pResult->findEntry( NULL, NULL,
			(FLMBYTE *)&(pTmpRSEntry->hdr.ui64NodeId), sizeof( FLMUINT64),
			&uiKeySize, (FLMBYTE *)pTmpRSEntry, sizeof( NODE_RS_ENTRY),
			&uiTmpRSEntrySize)))
		{
			*pi32ErrCode = FLM_BAD_ROOT_LINK;
			goto Exit;
		}

		// Found it!
		// Make sure this Root is a "True" root node.

		// Cannot have a parent node
		if (pTmpRSEntry->hdr.ui16BitMap & CHK_BM_PARENT_ID)
		{
			*pi32ErrCode = FLM_BAD_ROOT_PARENT;
			goto Exit;
		}

		// Cannot be another node child, or attribute or annotation node.
		if (pTmpRSEntry->hdr.ui16Flags & CHK_FIRST_CHILD_VERIFIED ||
			 pTmpRSEntry->hdr.ui16Flags & CHK_LAST_CHILD_VERIFIED ||
			 pTmpRSEntry->hdr.ui16Flags & CHK_ANNOTATION_VERIFIED)
		{
			*pi32ErrCode = FLM_BAD_ROOT_LINK;
			goto Exit;
		}

		pTmpRSEntry->hdr.ui16Flags |= CHK_ROOT_VERIFIED;

		if (RC_BAD( rc = pResult->modifyEntry( NULL, NULL, 
			(FLMBYTE *)&(pTmpRSEntry->hdr.ui64NodeId), sizeof( FLMUINT64),
			(FLMBYTE *)pTmpRSEntry, uiTmpRSEntrySize)))
		{
			goto Exit;
		}
	}
	else
	{
		// If the node we were passed IS the root node...
		// Cannot have a parent node
		if (pRSEntry->hdr.ui16BitMap & CHK_BM_PARENT_ID)
		{
			*pi32ErrCode = FLM_BAD_ROOT_LINK;
			goto Exit;
		}

		// Cannot be another node child, or attribute or annotation node.
		if (pRSEntry->hdr.ui16Flags & CHK_FIRST_CHILD_VERIFIED ||
			 pRSEntry->hdr.ui16Flags & CHK_LAST_CHILD_VERIFIED ||
			 pRSEntry->hdr.ui16Flags & CHK_ANNOTATION_VERIFIED)
		{
			*pi32ErrCode = FLM_BAD_ROOT_LINK;
			goto Exit;
		}

		pRSEntry->hdr.ui16Flags |= CHK_ROOT_VERIFIED;

		if (RC_BAD( rc = pResult->modifyEntry( NULL, NULL, 
			(FLMBYTE *)&(pRSEntry->hdr.ui64NodeId), sizeof( FLMUINT64),
			(FLMBYTE *)pRSEntry, uiRSEntrySize)))
		{
			goto Exit;
		}

	}

Exit:

	return rc;
}

/********************************************************************
Desc:	Verify that the parent Id field points to a valid node.
********************************************************************/
FSTATIC RCODE verifyParentLink(
	NODE_RS_ENTRY *		pRSEntry,
	NODE_RS_ENTRY *		pTmpRSEntry,
	F_BtResultSet *		pResult,
	FLMINT32 *				pi32ErrCode
	)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT64		ui64ParentId = getLinkVal( CHK_BM_PARENT_ID, pRSEntry);
	FLMUINT64		ui64RootId = getLinkVal( CHK_BM_ROOT_ID, pRSEntry);
	FLMUINT			uiTmpRSEntrySize;
	FLMUINT64		ui64TmpRootId;
	FLMUINT			uiKeySize = sizeof( FLMUINT64);

	f_memset( pTmpRSEntry, 0, sizeof(NODE_RS_ENTRY));

	if (!ui64ParentId)
	{
		// With no parent node, this node cannot have any child
		// or attribute or annotation nodes.
		if (pRSEntry->hdr.ui16Flags & CHK_FIRST_CHILD_VERIFIED ||
			 pRSEntry->hdr.ui16Flags & CHK_LAST_CHILD_VERIFIED ||
			 pRSEntry->hdr.ui16Flags & CHK_ANNOTATION_VERIFIED)
		{
			*pi32ErrCode = FLM_BAD_PARENT_LINK;
		}

		goto Exit;
	}

	if (ui64ParentId == pRSEntry->hdr.ui64NodeId)
	{
		*pi32ErrCode = FLM_BAD_PARENT_LINK;
		goto Exit;
	}

	pTmpRSEntry->hdr.ui64NodeId = ui64ParentId;

	if (RC_BAD( rc = pResult->findEntry( NULL, NULL,
		(FLMBYTE *)&(pTmpRSEntry->hdr.ui64NodeId), sizeof( FLMUINT64),
		&uiKeySize, (FLMBYTE *)pTmpRSEntry, sizeof( NODE_RS_ENTRY),
		&uiTmpRSEntrySize)))
	{
		*pi32ErrCode = FLM_BAD_PARENT_LINK;
		goto Exit;
	}

	// Verify that they are in the same document...
	ui64TmpRootId = getLinkVal( CHK_BM_ROOT_ID, pTmpRSEntry);
	if (ui64TmpRootId)
	{
		if (ui64RootId != ui64TmpRootId)
		{
			*pi32ErrCode = FLM_BAD_PARENT_LINK;
			goto Exit;
		}
	}
	else
	{
		if (ui64ParentId != ui64RootId)
		{
			*pi32ErrCode = FLM_BAD_PARENT_LINK;
			goto Exit;
		}
	}

	pTmpRSEntry->hdr.ui16Flags |= CHK_PARENT_VERIFIED;

	if( RC_BAD( rc = pResult->modifyEntry( NULL, NULL, 
		(FLMBYTE *)&(pTmpRSEntry->hdr.ui64NodeId), sizeof( FLMUINT64),
		(FLMBYTE *)pTmpRSEntry, uiTmpRSEntrySize)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/********************************************************************
Desc:	Verify that the First Child link points to a vaild node.
********************************************************************/
FSTATIC RCODE verifyFirstChildLink(
	NODE_RS_ENTRY *		pRSEntry,
	NODE_RS_ENTRY *		pTmpRSEntry,
	F_BtResultSet *		pResult,
	FLMINT32 *				pi32ErrCode)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT64		ui64FirstChildId = getLinkVal( CHK_BM_FIRST_CHILD, pRSEntry);
	FLMUINT64		ui64RootId = getLinkVal( CHK_BM_ROOT_ID, pRSEntry);
	FLMUINT			uiTmpRSEntrySize;
	FLMUINT64		ui64TmpRootId;
	FLMUINT			uiKeySize = sizeof( FLMUINT64);

	f_memset( pTmpRSEntry, 0, sizeof(NODE_RS_ENTRY));

	if (!ui64FirstChildId)
	{
		// Better not have a last child either.
		
		if (getLinkVal( CHK_BM_LAST_CHILD, pRSEntry))
		{
			*pi32ErrCode = FLM_BAD_FIRST_CHILD_LINK;
		}

		if (pRSEntry->hdr.ui16Flags & CHK_PARENT_VERIFIED)
		{
			if( !getLinkVal( CHK_BM_ANNOTATION, pRSEntry))
			{
				*pi32ErrCode = FLM_BAD_FIRST_CHILD_LINK;
			}
		}
		goto Exit;

	}

	if (ui64FirstChildId == pRSEntry->hdr.ui64NodeId)
	{
		*pi32ErrCode = FLM_BAD_FIRST_CHILD_LINK;
		goto Exit;
	}

	pTmpRSEntry->hdr.ui64NodeId = ui64FirstChildId;

	if (RC_BAD( rc = pResult->findEntry( NULL, NULL,
		(FLMBYTE *)&(pTmpRSEntry->hdr.ui64NodeId), sizeof( FLMUINT64),
		&uiKeySize, (FLMBYTE *)pTmpRSEntry, sizeof( NODE_RS_ENTRY),
		&uiTmpRSEntrySize)))
	{
		*pi32ErrCode = FLM_BAD_FIRST_CHILD_LINK;
		goto Exit;
	}

	// Must belong to the same document / root
	
	ui64TmpRootId = getLinkVal( CHK_BM_ROOT_ID, pTmpRSEntry);
	if (ui64RootId)
	{
		if (ui64RootId != ui64TmpRootId)
		{
			*pi32ErrCode = FLM_BAD_FIRST_CHILD_LINK;
			goto Exit;
		}
	}
	else
	{
		if (ui64TmpRootId != pRSEntry->hdr.ui64NodeId)
		{
			*pi32ErrCode = FLM_BAD_FIRST_CHILD_LINK;
			goto Exit;
		}
	}

	// Make sure this child has not been visited as a first child already.
	if (pTmpRSEntry->hdr.ui16Flags & CHK_FIRST_CHILD_VERIFIED)
	{
		*pi32ErrCode = FLM_BAD_FIRST_CHILD_LINK;
		goto Exit;
	}

	// Does this child reference the correct parent?
	if (getLinkVal( CHK_BM_PARENT_ID, pTmpRSEntry) != pRSEntry->hdr.ui64NodeId)
	{
		*pi32ErrCode = FLM_BAD_FIRST_CHILD_LINK;
		goto Exit;
	}

	// Mark the child as visited as a first child.
	pTmpRSEntry->hdr.ui16Flags |= CHK_FIRST_CHILD_VERIFIED;

	if (RC_BAD( rc = pResult->modifyEntry( NULL, NULL, 
														(FLMBYTE *)&(pTmpRSEntry->hdr.ui64NodeId),
														sizeof( FLMUINT64),
														(FLMBYTE *)pTmpRSEntry,
														uiTmpRSEntrySize)))
	{
		goto Exit;
	}

Exit:

	return rc;
}

/********************************************************************
Desc:	Verify that the Last Child link points to a vaild node.
********************************************************************/
FSTATIC RCODE verifyLastChildLink(
	NODE_RS_ENTRY *		pRSEntry,
	NODE_RS_ENTRY *		pTmpRSEntry,
	F_BtResultSet *		pResult,
	FLMINT32 *				pi32ErrCode
	)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT64		ui64LastChildId = getLinkVal( CHK_BM_LAST_CHILD, pRSEntry);
	FLMUINT64		ui64RootId = getLinkVal( CHK_BM_ROOT_ID, pRSEntry);
	FLMUINT			uiTmpRSEntrySize;
	FLMUINT64		ui64TmpRootId;
	FLMUINT			uiKeySize = sizeof( FLMUINT64);

	f_memset( pTmpRSEntry, 0, sizeof(NODE_RS_ENTRY));

	if (!ui64LastChildId)
	{
		// Better not have a first child either.
		
		if (getLinkVal( CHK_BM_FIRST_CHILD, pRSEntry))
		{
			*pi32ErrCode = FLM_BAD_LAST_CHILD_LINK;
		}

		if (pRSEntry->hdr.ui16Flags & CHK_PARENT_VERIFIED)
		{
			if( !getLinkVal( CHK_BM_ANNOTATION, pRSEntry))
			{
				*pi32ErrCode = FLM_BAD_FIRST_CHILD_LINK;
			}
		}
		goto Exit;

	}

	if (ui64LastChildId == pRSEntry->hdr.ui64NodeId)
	{
		*pi32ErrCode = FLM_BAD_LAST_CHILD_LINK;
		goto Exit;
	}

	pTmpRSEntry->hdr.ui64NodeId = ui64LastChildId;

	if( RC_BAD( rc = pResult->findEntry( NULL, NULL,
		(FLMBYTE *)&(pTmpRSEntry->hdr.ui64NodeId), sizeof( FLMUINT64),
		&uiKeySize, (FLMBYTE *)pTmpRSEntry, sizeof( NODE_RS_ENTRY),
		&uiTmpRSEntrySize)))
	{
		*pi32ErrCode = FLM_BAD_LAST_CHILD_LINK;
		goto Exit;
	}

	// Must belong to the same document / root
	ui64TmpRootId = getLinkVal( CHK_BM_ROOT_ID, pTmpRSEntry);
	if (ui64RootId)
	{
		if (ui64RootId != ui64TmpRootId)
		{
			*pi32ErrCode = FLM_BAD_LAST_CHILD_LINK;
			goto Exit;
		}
	}
	else
	{
		if (ui64TmpRootId != pRSEntry->hdr.ui64NodeId)
		{
			*pi32ErrCode = FLM_BAD_LAST_CHILD_LINK;
			goto Exit;
		}
	}

	// Make sure this child has not been visited as a last child already.
	if (pTmpRSEntry->hdr.ui16Flags & CHK_LAST_CHILD_VERIFIED)
	{
		*pi32ErrCode = FLM_BAD_LAST_CHILD_LINK;
		goto Exit;
	}

	// Does this child reference the correct parent?
	if (getLinkVal( CHK_BM_PARENT_ID, pTmpRSEntry) != pRSEntry->hdr.ui64NodeId)
	{
		*pi32ErrCode = FLM_BAD_LAST_CHILD_LINK;
		goto Exit;
	}

	// Mark the child as visited as a last child.
	pTmpRSEntry->hdr.ui16Flags |= CHK_LAST_CHILD_VERIFIED;

	if (RC_BAD( rc = pResult->modifyEntry( NULL, NULL, 
														(FLMBYTE *)&(pTmpRSEntry->hdr.ui64NodeId),
														sizeof( FLMUINT64),
														(FLMBYTE *)pTmpRSEntry,
														uiTmpRSEntrySize)))
	{
		goto Exit;
	}


Exit:

	return rc;
}

/********************************************************************
Desc:	Verify that the Prev Sibling link points to a valid node.
********************************************************************/
FSTATIC RCODE verifyPrevSiblingLink(
	NODE_RS_ENTRY *		pRSEntry,
	NODE_RS_ENTRY *		pTmpRSEntry,
	F_BtResultSet *		pResult,
	FLMINT32 *				pi32ErrCode
	)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT64		ui64PrevSibling = getLinkVal( CHK_BM_PREV_SIBLING, pRSEntry);
	FLMUINT64		ui64RootId = getLinkVal( CHK_BM_ROOT_ID, pRSEntry);
	FLMUINT64		ui64ParentId = getLinkVal( CHK_BM_PARENT_ID, pRSEntry);
	FLMUINT			uiTmpRSEntrySize;
	FLMUINT			uiKeySize = sizeof( FLMUINT64);

	f_memset( pTmpRSEntry, 0, sizeof(NODE_RS_ENTRY));

	if (!ui64PrevSibling)
	{
		// Should not be a Next Sibling to anyone.
		if (pRSEntry->hdr.ui16Flags & CHK_NEXT_SIBLING_VERIFIED)
		{
			*pi32ErrCode = FLM_BAD_PREV_SIBLING_LINK;
		}

		// Must also verify that this node is the first child of the parent node
		// - if there is a parent.
		if (ui64ParentId)
		{
			FLMUINT64		ui64FirstChild;
	
			pTmpRSEntry->hdr.ui64NodeId = ui64ParentId;
			if( RC_BAD( rc = pResult->findEntry( NULL, NULL,
				(FLMBYTE *)&(pTmpRSEntry->hdr.ui64NodeId), sizeof( FLMUINT64),
				&uiKeySize, (FLMBYTE *)pTmpRSEntry, sizeof( NODE_RS_ENTRY),
				&uiTmpRSEntrySize)))
			{
				*pi32ErrCode = FLM_BAD_PARENT_LINK;
				goto Exit;
			}
			ui64FirstChild = getLinkVal( CHK_BM_FIRST_CHILD, pTmpRSEntry);
			if (ui64FirstChild != pRSEntry->hdr.ui64NodeId)
			{
				FLMUINT64		ui64Annot;
		
				// It may be an annotation Node.
				
				ui64Annot = getLinkVal( CHK_BM_ANNOTATION, pTmpRSEntry);
				if (ui64Annot != pRSEntry->hdr.ui64NodeId)
				{
					*pi32ErrCode = FLM_BAD_PREV_SIBLING_LINK;
					goto Exit;
				}
			}
		}
		goto Exit;
	}

	pTmpRSEntry->hdr.ui64NodeId = ui64PrevSibling;

	if( RC_BAD( rc = pResult->findEntry( NULL, NULL,
		(FLMBYTE *)&(pTmpRSEntry->hdr.ui64NodeId), sizeof( FLMUINT64),
		&uiKeySize, (FLMBYTE *)pTmpRSEntry, sizeof( NODE_RS_ENTRY),
		&uiTmpRSEntrySize)))
	{
		*pi32ErrCode = FLM_BAD_PREV_SIBLING_LINK;
		goto Exit;
	}

	// Must belong to the same document unless both nodes are
	// document roots

	if( ui64RootId != getLinkVal( CHK_BM_ROOT_ID, pTmpRSEntry))
	{
		if( ui64ParentId || getLinkVal( CHK_BM_PARENT_ID, pTmpRSEntry))
		{
			*pi32ErrCode = FLM_BAD_PREV_SIBLING_LINK;
			goto Exit;
		}
	}

	// The previous sibling should not have been visited as a previous
	// sibling before now.

	if (pTmpRSEntry->hdr.ui16Flags & CHK_PREV_SIBLING_VERIFIED)
	{
		*pi32ErrCode = FLM_BAD_PREV_SIBLING_LINK;
		goto Exit;
	}

	// Should point to "this" node.
	if (getLinkVal( CHK_BM_NEXT_SIBLING, pTmpRSEntry) != pRSEntry->hdr.ui64NodeId)
	{
		*pi32ErrCode = FLM_BAD_PREV_SIBLING_LINK;
		goto Exit;
	}

	// Mark the previous sibling as being visited as such.
	pTmpRSEntry->hdr.ui16Flags |= CHK_PREV_SIBLING_VERIFIED;

	if (RC_BAD( rc = pResult->modifyEntry( NULL, NULL, 
														(FLMBYTE *)&(pTmpRSEntry->hdr.ui64NodeId),
														sizeof( FLMUINT64),
														(FLMBYTE *)pTmpRSEntry,
														uiTmpRSEntrySize)))
	{
		goto Exit;
	}


Exit:

	return rc;
}

/********************************************************************
Desc:	Verify that the Next Sibling link points to a valid node.
********************************************************************/
FSTATIC RCODE verifyNextSiblingLink(
	NODE_RS_ENTRY *		pRSEntry,
	NODE_RS_ENTRY *		pTmpRSEntry,
	F_BtResultSet *		pResult,
	FLMINT32 *				pi32ErrCode
	)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT64		ui64NextSibling = getLinkVal( CHK_BM_NEXT_SIBLING, pRSEntry);
	FLMUINT64		ui64RootId = getLinkVal( CHK_BM_ROOT_ID, pRSEntry);
	FLMUINT64		ui64ParentId = getLinkVal( CHK_BM_PARENT_ID, pRSEntry);
	FLMUINT			uiTmpRSEntrySize;
	FLMUINT			uiKeySize = sizeof( FLMUINT64);

	f_memset( pTmpRSEntry, 0, sizeof(NODE_RS_ENTRY));

	if (!ui64NextSibling)
	{
		// Should not be a Prev Sibling to anyone.
		if (pRSEntry->hdr.ui16Flags & CHK_PREV_SIBLING_VERIFIED)
		{
			*pi32ErrCode = FLM_BAD_NEXT_SIBLING_LINK;
		}
		// Must also verify that this node is the last child of the parent node
		// - if there is a parent.
		if (ui64ParentId)
		{
			FLMUINT64		ui64LastChild;
	
			pTmpRSEntry->hdr.ui64NodeId = ui64ParentId;
			if( RC_BAD( rc = pResult->findEntry( NULL, NULL,
				(FLMBYTE *)&(pTmpRSEntry->hdr.ui64NodeId), sizeof( FLMUINT64),
				&uiKeySize, (FLMBYTE *)pTmpRSEntry, sizeof( NODE_RS_ENTRY),
				&uiTmpRSEntrySize)))
			{
				*pi32ErrCode = FLM_BAD_PARENT_LINK;
				goto Exit;
			}
			ui64LastChild = getLinkVal( CHK_BM_LAST_CHILD, pTmpRSEntry);
			if (ui64LastChild != pRSEntry->hdr.ui64NodeId)
			{
				FLMUINT64		ui64Annot;
		
				// It may be an annotation Node.
				
				ui64Annot = getLinkVal( CHK_BM_ANNOTATION, pTmpRSEntry);
				if (ui64Annot != pRSEntry->hdr.ui64NodeId)
				{
					*pi32ErrCode = FLM_BAD_NEXT_SIBLING_LINK;
					goto Exit;
				}
			}

		}
		goto Exit;
	}


	pTmpRSEntry->hdr.ui64NodeId = ui64NextSibling;

	if( RC_BAD( rc = pResult->findEntry( NULL, NULL,
		(FLMBYTE *)&(pTmpRSEntry->hdr.ui64NodeId), sizeof( FLMUINT64),
		&uiKeySize, (FLMBYTE *)pTmpRSEntry, sizeof( NODE_RS_ENTRY),
		&uiTmpRSEntrySize)))
	{
		*pi32ErrCode = FLM_BAD_NEXT_SIBLING_LINK;
		goto Exit;
	}

	// Must belong to the same document unless both nodes are
	// document roots

	if( ui64RootId != getLinkVal( CHK_BM_ROOT_ID, pTmpRSEntry))
	{
		if( ui64ParentId || getLinkVal( CHK_BM_PARENT_ID, pTmpRSEntry))
		{
			*pi32ErrCode = FLM_BAD_NEXT_SIBLING_LINK;
			goto Exit;
		}
	}

	// The next sibling should not have been visited as a next
	// sibling before now.

	if( pTmpRSEntry->hdr.ui16Flags & CHK_NEXT_SIBLING_VERIFIED)
	{
		*pi32ErrCode = FLM_BAD_NEXT_SIBLING_LINK;
		goto Exit;
	}

	// Should point to "this" node.

	if( getLinkVal( CHK_BM_PREV_SIBLING, pTmpRSEntry) != pRSEntry->hdr.ui64NodeId)
	{
		*pi32ErrCode = FLM_BAD_NEXT_SIBLING_LINK;
		goto Exit;
	}

	// Mark the previous sibling as being visited as such.

	pTmpRSEntry->hdr.ui16Flags |= CHK_NEXT_SIBLING_VERIFIED;

	if( RC_BAD( rc = pResult->modifyEntry( NULL, NULL,
		(FLMBYTE *)&(pTmpRSEntry->hdr.ui64NodeId),
		sizeof( FLMUINT64), (FLMBYTE *)pTmpRSEntry, uiTmpRSEntrySize)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/********************************************************************
Desc:	
********************************************************************/
FSTATIC RCODE verifyAnnotationLink(
	NODE_RS_ENTRY *		pRSEntry,
	NODE_RS_ENTRY *		pTmpRSEntry,
	F_BtResultSet *		pResult,
	FLMINT32 *				pi32ErrCode
	)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT64		ui64Annotation = getLinkVal( CHK_BM_ANNOTATION, pRSEntry);
	FLMUINT64		ui64RootId = getLinkVal( CHK_BM_ROOT_ID, pRSEntry);
	FLMUINT			uiTmpRSEntrySize;
	FLMUINT64		ui64TmpRootId;
	FLMUINT			uiKeySize = sizeof( FLMUINT64);

	f_memset( pTmpRSEntry, 0, sizeof(NODE_RS_ENTRY));

	if (!ui64Annotation)
	{
		goto Exit;
	}

	pTmpRSEntry->hdr.ui64NodeId = ui64Annotation;

	if( RC_BAD( rc = pResult->findEntry( NULL, NULL,
		(FLMBYTE *)&(pTmpRSEntry->hdr.ui64NodeId), sizeof( FLMUINT64),
		&uiKeySize, (FLMBYTE *)pTmpRSEntry, sizeof( NODE_RS_ENTRY),
		&uiTmpRSEntrySize)))
	{
		*pi32ErrCode = FLM_BAD_ANNOTATION_LINK;
		goto Exit;
	}

	// Must belong to the same document / root
	// Annotations may belong to a node that does not have a Root Id.  If
	// that is the case, the parent and root of the annotaion should match and
	// the parent should point to the source node.
	
	ui64TmpRootId = getLinkVal( CHK_BM_ROOT_ID, pTmpRSEntry);
	
	if( ui64RootId)
	{
		if (ui64RootId != ui64TmpRootId)
		{
			*pi32ErrCode = FLM_BAD_ANNOTATION_LINK;
			goto Exit;
		}
	}
	else
	{
		// With no root, the temporary root must point to "this" node.
		
		if (ui64TmpRootId != pRSEntry->hdr.ui64NodeId)
		{
			*pi32ErrCode = FLM_BAD_ANNOTATION_LINK;
			goto Exit;
		}
	}

	// The annotation should not have been visited as such before now.
	if (pTmpRSEntry->hdr.ui16Flags & CHK_ANNOTATION_VERIFIED)
	{
		*pi32ErrCode = FLM_BAD_ANNOTATION_LINK;
		goto Exit;
	}

	// Parent should point to "this" node.
	if (getLinkVal( CHK_BM_PARENT_ID, pTmpRSEntry) != pRSEntry->hdr.ui64NodeId)
	{
		*pi32ErrCode = FLM_BAD_ANNOTATION_LINK;
		goto Exit;
	}

	// Mark the last attr as being visited as such.
	pTmpRSEntry->hdr.ui16Flags |= CHK_ANNOTATION_VERIFIED;

	if (RC_BAD( rc = pResult->modifyEntry( NULL, NULL, 
														(FLMBYTE *)&(pTmpRSEntry->hdr.ui64NodeId),
														sizeof( FLMUINT64),
														(FLMBYTE *)pTmpRSEntry,
														uiTmpRSEntrySize)))
	{
		goto Exit;
	}


Exit:

	return rc;
}

/***************************************************************************
Desc:	This routine does for chains of data-only blocks what verifySubTree
		does for the other blocks.  Basically, it's going to read in each
		block in the chain, perform some basic verification on the header
		and add the data to the NodeVerifier object.
*****************************************************************************/
RCODE F_DbCheck::verifyDOChain(
	STATE_INFO *	pParentState,
	FLMUINT			uiBlkAddr,
	FLMINT32 *		pi32ElmErrCode)
{
	RCODE					rc = NE_XFLM_OK;
	F_NodeVerifier *	pNodeVerifier = pParentState->pNodeVerifier;
	F_CachedBlock *	pSCache = NULL;
	F_BLK_HDR *			pBlkHdr = NULL;
	FLMUINT				uiParentBlkAddr = pParentState->pBlkHdr->ui32BlkAddr;
	FLMUINT				uiPrevNextBlkAddr;  // The ui32NextBlkInChain field from the previous block
	FLMUINT				uiNumErrors = 0;
	FLMUINT				uiNumBlksRead = 0;
	FLMUINT				uiBlockSize = m_pDb->m_pDatabase->getBlockSize();
	FLMBYTE *			pucData = NULL;
	FLMUINT				uiDataLen = 0;
	BLOCK_INFO *		pBlkInfo = &pParentState->BlkInfo;
	STATE_INFO			StateInfo;

	
	// All leaf nodes are level 0, and only leaf nodes can point
	// to data only blocks...
	if (pParentState->uiLevel != 0)
	{
		*pi32ElmErrCode = FLM_BAD_ELM_INVALID_LEVEL;
		rc = RC_SET_AND_ASSERT( NE_XFLM_FAILURE);
		goto Exit;
	}


	//Initialize the StateInfo struct
	f_memset( &StateInfo, 0, sizeof( STATE_INFO));
	StateInfo.pCollection = pParentState->pCollection;
	StateInfo.pDb = pParentState->pDb;
	StateInfo.ui64ElmNodeId = pParentState->ui64ElmNodeId;

	// Initialize the NodeVerifier object
	if (pNodeVerifier)
	{
		pNodeVerifier->Reset( pParentState);
	}

	// Now, loop through every block in the chain...
	uiPrevNextBlkAddr = 0;
	while (uiBlkAddr)
	{
		// Read in the next block in the chain
		f_yieldCPU();

		if (RC_BAD( rc = blkRead( uiBlkAddr,
										  &pBlkHdr,
										  &pSCache,
										  pi32ElmErrCode)))
		{
			if (*pi32ElmErrCode)
			{
				uiNumErrors++;
				chkReportError( *pi32ElmErrCode,
									 XFLM_LOCALE_B_TREE,
									 (FLMUINT32)m_Progress.ui32LfNumber,
									 (FLMUINT32)m_Progress.ui32LfType,
									 (FLMUINT32)StateInfo.uiLevel,
									 (FLMUINT32)uiBlkAddr,
									 (FLMUINT32)uiParentBlkAddr,
									 0,
									 0);
			}
			else if (rc == NE_XFLM_OLD_VIEW)
			{
				// We're going to have to reset.  Unfortunately,
				// we don't know how to position ourselves into the middle
				// of a record, so we'll have to reset back to the beginning
				// of the record.  We should still be able to finish processing.
				// We only need to see enough of the DOM node to build the header.
				// It's the header that gives us the DOM link information.
				if (pNodeVerifier)
				{
					pNodeVerifier->Reset( pParentState);
				}
				m_Progress.ui64BytesExamined -=
												(FLMUINT64)(uiBlockSize*uiNumBlksRead);
				chkCallProgFunc();
			}

			goto Exit;
		}

		// Chains of data only blocks should always have at least 2 blocks...
		if ((uiNumBlksRead == 0) && (pBlkHdr->ui32NextBlkInChain == 0))
		{
			*pi32ElmErrCode = FLM_BAD_DATA_BLOCK_COUNT;
			rc = RC_SET_AND_ASSERT( NE_XFLM_FAILURE);
			goto Exit;
		}
		
		// Record the progress that we're making
		uiNumBlksRead++;
		m_Progress.ui64BytesExamined += (FLMUINT64)uiBlockSize;
		chkCallProgFunc();

		// Check the block header.
		StateInfo.pBlkHdr = pBlkHdr;
		StateInfo.uiBlkType = BT_DATA_ONLY;
		StateInfo.ui32BlkAddress = (FLMUINT32)uiBlkAddr;
		*pi32ElmErrCode = flmVerifyBlockHeader( &StateInfo, pBlkInfo, uiBlockSize,
														  0xFFFFFFFF, (uiNumBlksRead > 1) ?
														  uiPrevNextBlkAddr : 0,	TRUE);
		if (*pi32ElmErrCode != 0)
		{
			uiNumErrors++;
			chkReportError( *pi32ElmErrCode, XFLM_LOCALE_B_TREE,
								 (FLMUINT32)m_Progress.ui32LfNumber,
								 (FLMUINT32)m_Progress.ui32LfType,
								 (FLMUINT32)StateInfo.uiLevel, (FLMUINT32)uiBlkAddr,
								 (FLMUINT32)uiParentBlkAddr, 0, 0);
		}

		// Verify that the ui16BlkBytesAvail is a reasonable size...
		
		if( (pBlkHdr->ui32NextBlkInChain != 0) &&
			 (pBlkHdr->ui16BlkBytesAvail != 0))
		{
			*pi32ElmErrCode = FLM_BAD_AVAIL_SIZE;
			rc = RC_SET_AND_ASSERT( NE_XFLM_FAILURE);
			goto Exit;
		}

		// Add the current data to the verifier....
		// We really only should need to do this if this is the first block
		// in the chain.
		
		if( !pBlkHdr->ui32PrevBlkInChain)
		{
			FLMBYTE *		pucPtr = (FLMBYTE *)pBlkHdr + sizeof( F_BLK_HDR);
			FLMUINT			uiKeySize = (FLMUINT)FB2UW( pucPtr);
			
			pucData = pucPtr + uiKeySize + 2;

			uiDataLen = uiBlockSize - sizeof( F_BLK_HDR) - 
							pBlkHdr->ui16BlkBytesAvail - uiKeySize - 2;
			
			if (pNodeVerifier)
			{
				if (RC_BAD( rc = pNodeVerifier->AddData(
								StateInfo.ui64ElmNodeId, pucData, uiDataLen)))
				{
					goto Exit;
				}
			}
		}

		uiPrevNextBlkAddr = uiBlkAddr;
		uiBlkAddr = pBlkHdr->ui32NextBlkInChain;		
	} // end of while (uiNextBlkAddress)


Exit:
	if (pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}
	else if (pBlkHdr)
	{
		// The reason for the else is that pBlkHdr will normally be
		// pointing into pSCache.  Only if the call to getBlock()
		// inside of blkRead() fails will memory be allocated
		// explicitly for pBlkHdr
		f_free( &pBlkHdr);
	}
	
	return rc;
}


/***************************************************************************
Desc:	This routine checks all of the blocks/links in a sub-tree of a
		B-TREE.  It calls itself recursively whenever it descends a level
		in the tree.
*****************************************************************************/
RCODE F_DbCheck::verifySubTree(
	STATE_INFO *	pParentState,
	STATE_INFO *	pStateInfo,
	FLMUINT			uiBlkAddress,
	FLMBYTE **		ppucResetKey,
	FLMUINT			uiResetKeyLen,
	FLMUINT64		ui64ResetNodeId)
{
	RCODE			  		rc = NE_XFLM_OK;
	F_CachedBlock *	pSCache = NULL;
	F_BLK_HDR *			pBlkHdr = NULL;
	FLMUINT				uiLevel = pStateInfo->uiLevel;
	FLMUINT				uiBlkType = pStateInfo->uiBlkType;
	FLMUINT				uiLfType = m_pLFile->eLfType;
	FLMUINT				uiBlockSize = m_pDb->m_pDatabase->m_uiBlockSize;
	FLMUINT				uiParentBlkAddress;
	FLMUINT				uiChildBlkAddress;
	FLMUINT				uiPrevNextBlkAddress;
	FLMINT32			  	i32ElmErrCode;
	FLMINT32			  	i32BlkErrCode = 0;
	FLMINT32			  	i32LastErrCode = 0;
	FLMUINT				uiNumErrors = 0;
	FLMUINT64		  	ui64SaveKeyCount = 0;
	FLMUINT64			ui64SaveKeyRefs = 0;
	BLOCK_INFO			SaveBlkInfo;
	BLOCK_INFO *		pBlkInfo;
	FLMBOOL			  	bProcessElm;
	FLMBOOL			  	bCountElm;
	FLMBOOL				bDescendToChildBlocks;
	FLMINT			  	iCompareStatus;
	FLMINT32				i32HdrErrCode;
	F_NodeVerifier *	pNodeVerifier = pStateInfo->pNodeVerifier;
	STATE_INFO *		pChildStateInfo = NULL;
	F_CachedBlock *	pTmpSCache = NULL;
	F_BLK_HDR *			pTmpBlkHdr = NULL;

	// Setup the state information.

	pStateInfo->pBlkHdr = NULL;
	pStateInfo->ui32BlkAddress = (FLMUINT32)uiBlkAddress;
	uiPrevNextBlkAddress = pStateInfo->ui32NextBlkAddr;
	uiParentBlkAddress = (pParentState)
										? pParentState->ui32BlkAddress
										: 0xFFFFFFFF;

	// Read the sub-tree block into memory
	
	bDescendToChildBlocks = TRUE;

	if (RC_BAD( rc = blkRead( uiBlkAddress, &pBlkHdr, &pSCache, &i32BlkErrCode)))
	{
		if (i32BlkErrCode)
		{
			uiNumErrors++;
			i32LastErrCode = i32BlkErrCode;

			chkReportError( i32BlkErrCode, XFLM_LOCALE_B_TREE, 
				(FLMUINT32)m_Progress.ui32LfNumber, (FLMUINT32)m_Progress.ui32LfType,
				(FLMUINT32)uiLevel, (FLMUINT32)uiBlkAddress,
				(FLMUINT32)uiParentBlkAddress, 0, 0);
				
			if( i32BlkErrCode == FLM_BAD_BLK_CHECKSUM)
			{
				bDescendToChildBlocks = FALSE;

				// Allow to continue the check, but if this is a non-leaf block
				// a non-zero i32BlkErrCode will prevent us from descending to
				// child blocks.  Set rc to SUCCESS so we won't goto Exit below.

				rc = NE_XFLM_OK;
			}
			else if (i32BlkErrCode == FLM_COULD_NOT_SYNC_BLK)
			{
				i32LastErrCode = i32BlkErrCode;

				// Need the goto here, because rc is changed to SUCCESS,
				// and the goto below would get skipped.

				rc = NE_XFLM_OK;
				goto fix_state;
			}
		}

		// If rc was not changed to SUCCESS above, goto Exit.

		if (RC_BAD( rc))
		{
			goto Exit;
		}
	}
	pStateInfo->pBlkHdr = pBlkHdr;

	// Verify the block header
	// Don't recount the block if we are resetting.

	if (uiResetKeyLen == ~(FLMUINT)0)
	{
		m_Progress.ui64BytesExamined += (FLMUINT64)uiBlockSize;
		pBlkInfo = &pStateInfo->BlkInfo;
	}
	else
	{
		pBlkInfo = NULL;
	}
	
	chkCallProgFunc();

	// Check the block header.

	if ((i32HdrErrCode =
				flmVerifyBlockHeader( pStateInfo, pBlkInfo, uiBlockSize,
											 (pParentState == NULL)
													 ? 0
													 : 0xFFFFFFFF,
											 (pParentState == NULL)
													 ? 0
													 : pParentState->ui32LastChildAddr,
											 TRUE)) == 0)
	{

		// Verify the previous block's next block address -- it should
		// equal the current block's address.

		if (uiPrevNextBlkAddress != 0xFFFFFFFF &&
			 uiPrevNextBlkAddress != uiBlkAddress &&
			 (uiResetKeyLen == ~(FLMUINT)0))
		{
			i32HdrErrCode = FLM_BAD_PREV_BLK_NEXT;
		}
	}
	
	if (i32HdrErrCode != 0)
	{
		// Check to see if the previous block is still valid.
		// It may be that the block has gone away, and so is no longer valid.
		
		if (i32HdrErrCode == FLM_BAD_BLK_HDR_PREV)
		{

			flmAssert( pParentState);

			if (RC_BAD( rc = blkRead( pParentState->ui32LastChildAddr,
				&pTmpBlkHdr, &pTmpSCache, &i32BlkErrCode)))
			{
				i32LastErrCode = i32BlkErrCode;
				uiNumErrors++;
				
				chkReportError( i32BlkErrCode, XFLM_LOCALE_B_TREE,
					(FLMUINT32)m_Progress.ui32LfNumber, (FLMUINT32)m_Progress.ui32LfType,
					(FLMUINT32)uiLevel,
					(FLMUINT32)uiBlkAddress, (FLMUINT32)uiParentBlkAddress, 0, 0);
			}
			else
			{
				// If the block is free, it means that somewhere in our check, we
				// deleted entries that resulted in this block being emptied,
				// thus freed.
				
				if (pTmpBlkHdr->ui8BlkType == BT_FREE)
				{
					i32HdrErrCode = 0;
				}
				else
				{
					i32LastErrCode = i32HdrErrCode;
					uiNumErrors++;
					chkReportError( i32HdrErrCode, XFLM_LOCALE_B_TREE,
						(FLMUINT32)m_Progress.ui32LfNumber,
						(FLMUINT32)m_Progress.ui32LfType, (FLMUINT32)uiLevel,
						(FLMUINT32)uiBlkAddress, (FLMUINT32)uiParentBlkAddress, 0, 0);
				}
			}
		}
		else
		{
			i32LastErrCode = i32HdrErrCode;
			uiNumErrors++;
			chkReportError( i32HdrErrCode, XFLM_LOCALE_B_TREE,
				(FLMUINT32)m_Progress.ui32LfNumber,
				(FLMUINT32)m_Progress.ui32LfType, (FLMUINT32)uiLevel,
				(FLMUINT32)uiBlkAddress, (FLMUINT32)uiParentBlkAddress,
				0, 0);
		}

		if (pTmpSCache)
		{
			ScaReleaseCache( pTmpSCache, FALSE);
			pTmpSCache = NULL;
			pTmpBlkHdr = NULL;
		}
	}

	// Verify the structure of the block
	
	if( RC_BAD( rc = verifyBlockStructure( uiBlockSize, 
		(F_BTREE_BLK_HDR *)pBlkHdr)))
	{
		if (rc == NE_XFLM_BTREE_ERROR)
		{
			i32BlkErrCode = FLM_BAD_BLOCK_STRUCTURE;
			rc = NE_XFLM_OK;
			goto fix_state;
		}
		else
		{
			goto Exit;
		}
	}

	// Go through the elements in the block.
	
	for (	pStateInfo->uiElmOffset=0;
			(pStateInfo->uiElmOffset <
				((F_BTREE_BLK_HDR *)pBlkHdr)->ui16NumKeys &&
					RC_OK( m_LastStatusRc));
			pStateInfo->uiElmOffset++)
	{
		// If we are resetting, save any statistical information so we
		// can back it out if we need to.
		
		if (uiResetKeyLen != ~(FLMUINT)0)
		{
			ui64SaveKeyCount = pStateInfo->ui64KeyCount;
			ui64SaveKeyRefs = pStateInfo->ui64KeyRefs;
			f_memcpy( &SaveBlkInfo, &pStateInfo->BlkInfo, sizeof( BLOCK_INFO));
			bCountElm = FALSE;
			bProcessElm = FALSE;
		}
		else
		{
			bCountElm = TRUE;
			bProcessElm = TRUE;
		}

		// Verify the element first, then check if we are restting...

		m_LastStatusRc = flmVerifyElement( pStateInfo, m_pLFile, m_pIxd,
																    &i32ElmErrCode);
		if (i32ElmErrCode)
		{
			// Report any errors in the element.
			
			i32LastErrCode = i32ElmErrCode;
			uiNumErrors++;
			
			if (RC_BAD( rc = chkReportError( i32ElmErrCode, XFLM_LOCALE_B_TREE,
				(FLMUINT32)m_Progress.ui32LfNumber, (FLMUINT32)m_Progress.ui32LfType,
				(FLMUINT32)uiLevel, (FLMUINT32)uiBlkAddress,
				(FLMUINT32)uiParentBlkAddress, (FLMUINT32)pStateInfo->uiElmOffset,
				pStateInfo->ui64ElmNodeId)))
			{
				break;
			}
		}

		// See if we are resetting

		iCompareStatus = 0;

		if ((uiResetKeyLen != ~(FLMUINT)0) &&
				pStateInfo->bValidKey &&
				(!pStateInfo->uiElmKeyLen ||
				((iCompareStatus = f_memcmp(
							pStateInfo->pucElmKey, *ppucResetKey,
							pStateInfo->uiElmKeyLen < uiResetKeyLen 
								? pStateInfo->uiElmKeyLen
								: uiResetKeyLen)) >= 0)))
		{

			// A key length of 0 is valid.  It refers to the LEM.  All keys are
			// less than the LEM if their length is > 0.

			if ((uiResetKeyLen && pStateInfo->uiElmKeyLen ||
					!pStateInfo->uiElmKeyLen))
			{
				// If we passed the target key, or we are on the last element
				// then count it.

				bProcessElm = TRUE;
				if ((iCompareStatus > 0) || !(pStateInfo->uiElmKeyLen))
				{
					if ( (uiBlkType == BT_LEAF_DATA) ||
							(uiBlkType == BT_LEAF) )
					{
						bCountElm = TRUE;

						uiResetKeyLen = ~(FLMUINT)0;
						
						if (*ppucResetKey)
						{
							f_free( ppucResetKey);
						}
						
						*ppucResetKey = NULL;
						ui64ResetNodeId = 0;
					}
				}
				else if ( uiLfType == XFLM_LF_INDEX)
				{
					bCountElm = TRUE;
				}
			}
		}

		if (bCountElm)
		{
			pStateInfo->BlkInfo.ui64ElementCount++;
		}

		// Check for index keys that can be verified on leaf level blocks.

		if (uiResetKeyLen == ~(FLMUINT)0)
		{
			if ( isIndexBlk( (F_BTREE_BLK_HDR *)pStateInfo->pBlkHdr) &&
				  (pStateInfo->pBlkHdr->ui8BlkType == BT_LEAF ||
								pStateInfo->pBlkHdr->ui8BlkType == BT_LEAF_DATA) &&
				  pStateInfo->uiElmKeyLen)
			{
				FLMUINT64	ui64CurrTransId = pStateInfo->pDb->m_ui64CurrTransID;

				if (RC_BAD( rc = verifyIXRefs( pStateInfo, ui64ResetNodeId)))
				{
					goto Exit;
				}

				// Make sure we resynchronize when we make changes to
				// blocks we are looking at.
				
				if (pStateInfo->pDb->m_ui64CurrTransID != ui64CurrTransId)
				{
					if (m_bPhysicalCorrupt)
					{
						m_bPhysicalCorrupt = FALSE;
						m_uiFlags |= XFLM_DO_LOGICAL_CHECK;
					}

					rc = m_LastStatusRc = RC_SET( NE_XFLM_RESET_NEEDED);
					goto Exit;
				}
			}

			if (RC_BAD( m_LastStatusRc))
			{
				break;
			}

			// Keep track of the number of continuation elements.
			
			if( (uiBlkType == BT_LEAF_DATA) &&
				((*pStateInfo->pucElm & BTE_FLAG_FIRST_ELEMENT) == 0))
			{
				pStateInfo->BlkInfo.ui64ContElementCount++;
				pStateInfo->BlkInfo.ui64ContElmBytes += pStateInfo->uiElmLen;
			}
		}

		// Do some further checking.

		if (i32ElmErrCode == 0)
		{
			if (bProcessElm &&
				 (uiBlkType == BT_LEAF_DATA ||
				  uiBlkType == BT_LEAF) &&
				  uiLfType == XFLM_LF_COLLECTION) 
			{
				// No need to process LEM element
				
				if ((pStateInfo->uiElmKeyLen != 0) && (pStateInfo->bValidKey))
				{
					// Is this record stored in a chain of DO blocks...?
					
					if ((*pStateInfo->pucElm & BTE_FLAG_DATA_BLOCK) != 0)	
					{
						flmAssert( pStateInfo->uiElmDataLen == 4);

						if( RC_BAD( rc = verifyDOChain( pStateInfo,
							FB2UD( pStateInfo->pucElmData), &i32ElmErrCode)))
						{
							goto Exit;
						}
					}
					else
					{
						// Since DOM nodes may be spread across multiple entries
						// in the Btree block, it may be impractical to read in
						// the entire node all at once, we need a way of doing the
						// verification a little at a time.  The NodeVerifier
						// object handles this. We pass in the data as we get it,
						// it appends the data to any data it had left over from
						// a previous call, and then verifies as much as it can.  
						// Any "left over" data is saved for the next call.
						
						// If the "first" flag is set on this element, twe need
						// to reset the verifier

						if( *pStateInfo->pucElm & BTE_FLAG_FIRST_ELEMENT && 
							pNodeVerifier)
						{
							pNodeVerifier->Reset( pStateInfo);
						}

						// Add the current data to the verifier....

						flmAssert( pStateInfo->ui64ElmNodeId);
						if (pNodeVerifier)
						{
							if (RC_BAD( rc = pNodeVerifier->AddData( 
								pStateInfo->ui64ElmNodeId, pStateInfo->pucElmData,
								pStateInfo->uiElmDataLen)))
							{
								goto Exit;
							}
						}
					}

					// If this is the last entry for this element, then we can 
					// finalize the node verifier.  This entails assembling
					// the DOM node, and possibly adding required information
					// to a result set for later DOM link verification.  We also
					// add the node Id to the index result set if we are
					// checking/verifying indexes.

					if( *pStateInfo->pucElm & BTE_FLAG_LAST_ELEMENT)
					{
						if( pNodeVerifier)
						{
							if( RC_BAD( rc = pNodeVerifier->finalize( 
								m_pDb, m_pDb->m_pDict,
								pStateInfo->pCollection->lfInfo.uiLfNum,
								pStateInfo->ui64ElmNodeId, m_bSkipDOMLinkCheck,
								&i32ElmErrCode)))
							{
								goto Exit;
							}
						}
					}
				}

				if (bProcessElm)
				{
					if (i32ElmErrCode != 0)
					{
						// Report any errors in the element.

						i32LastErrCode = i32ElmErrCode;
						uiNumErrors++;
						
						chkReportError( i32ElmErrCode, XFLM_LOCALE_B_TREE,
							(FLMUINT32)m_Progress.ui32LfNumber, (FLMUINT32)m_Progress.ui32LfType,
							(FLMUINT32)uiLevel,
							(FLMUINT32)uiBlkAddress, (FLMUINT32)uiParentBlkAddress,
							(FLMUINT32)pStateInfo->uiElmOffset,
							pStateInfo->ui64ElmNodeId);
				
						if (RC_BAD( m_LastStatusRc))
						{
							break;
						}
					}
				}
			}
			else if (uiBlkType != BT_LEAF_DATA && uiBlkType != BT_LEAF)
			{
				flmAssert( uiBlkType != BT_DATA_ONLY);
				uiChildBlkAddress = (FLMUINT)FB2UD(pStateInfo->pucElm);

				// Check the child sub-tree -- NOTE, this is a recursive call.

				if (bProcessElm)
				{
					if (!bDescendToChildBlocks)
					{
						rc = NE_XFLM_OK;
					}
					else
					{
						pChildStateInfo = (pStateInfo - 1);
						if (pChildStateInfo->uiElmKeyLen && 
							(uiResetKeyLen != ~(FLMUINT)0))
						{
							uiResetKeyLen = pChildStateInfo->uiElmKeyLen;
							ui64ResetNodeId = pChildStateInfo->ui64ElmNodeId;
							
							if (*ppucResetKey)
							{
								f_free( ppucResetKey);
							}
							
							if (RC_BAD( rc = f_calloc( uiResetKeyLen + 1, 
								ppucResetKey)))
							{
								goto Exit;
							}
							
							f_memcpy( *ppucResetKey, pChildStateInfo->pucElmKey,
								uiResetKeyLen);
						}

						rc = verifySubTree( pStateInfo, (pStateInfo - 1),
									uiChildBlkAddress, ppucResetKey, uiResetKeyLen,
									ui64ResetNodeId);
					}

					if (RC_BAD( rc) || RC_BAD( m_LastStatusRc))
					{
						goto Exit;
					}

					// Once we reach the key, set it to an empty to key so that
					// we will always descend to the child block after this point.

					uiResetKeyLen = ~(FLMUINT)0;
					if (*ppucResetKey)
					{
						f_free( ppucResetKey);
					}
					
					*ppucResetKey = NULL;
					ui64ResetNodeId = 0;
				}

				// Save the child block address in the level information
				
				pStateInfo->ui32LastChildAddr = (FLMUINT32)uiChildBlkAddress;
			}
		}

		// If we were resetting on this element, restore the statistics to what
		// they were before.

		if( !bCountElm)
		{
			pStateInfo->ui64KeyCount = ui64SaveKeyCount;
			pStateInfo->ui64KeyRefs = ui64SaveKeyRefs;
			f_memcpy( &pStateInfo->BlkInfo, &SaveBlkInfo, sizeof( BLOCK_INFO));
		}

		if (RC_BAD( rc = chkCallProgFunc()))
		{
			break;
		}
	}

	// Verify that the last key in the block matches the parent's key.

	if (i32LastErrCode == 0 && pParentState && RC_OK( m_LastStatusRc))
	{
		if (pStateInfo->bValidKey && pParentState->bValidKey &&
			 f_memcmp( pStateInfo->pucElmKey,
						  pParentState->pucElmKey,
						  pStateInfo->uiElmKeyLen < pParentState->uiElmKeyLen
								? pStateInfo->uiElmKeyLen
								: pParentState->uiElmKeyLen) != 0)
		{
			i32LastErrCode = FLM_BAD_PARENT_KEY;
			uiNumErrors++;
			
			chkReportError( i32LastErrCode, XFLM_LOCALE_B_TREE, 
				(FLMUINT32)m_Progress.ui32LfNumber, (FLMUINT32)m_Progress.ui32LfType,
				(FLMUINT32)uiLevel, (FLMUINT32)uiBlkAddress,
				(FLMUINT32)uiParentBlkAddress, 0, 0);
		}
	}

fix_state:

	// If the block could not be verified, set the level's next block
	// address and last child address to zero to indicate that we really
	// aren't sure we're at the right place in this level in the B-TREE.

	if (i32LastErrCode != 0)
	{
		pStateInfo->BlkInfo.i32ErrCode = i32LastErrCode;
		pStateInfo->BlkInfo.uiNumErrors += uiNumErrors;

		// Reset all child block states.

		for (;;)
		{
			pStateInfo->ui32NextBlkAddr = 0xFFFFFFFF;
			pStateInfo->ui32LastChildAddr = 0xFFFFFFFF;
			pStateInfo->bValidKey = FALSE;
			pStateInfo->uiElmLastFlag = 0xFF;

			// Quit when the leaf level has been reached.
			
			if (pStateInfo->uiLevel == 0)
			{
				break;
			}
			
			pStateInfo--;
		}
	}

Exit:

	if (pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}
	else if (pBlkHdr)
	{
		f_free( &pBlkHdr);
	}

	pStateInfo->pBlkHdr = NULL;
	return( rc);
}

/********************************************************************
Desc:	
*********************************************************************/
FSTATIC FLMBYTE * getEntryEnd(
	FLMBYTE *			pucEntry,
	FLMUINT				uiBlkType)
{
	FLMBYTE *			pucEnd = pucEntry;

	switch (uiBlkType)
	{
		case BT_LEAF:
		{
			FLMUINT		uiKL = (FLMUINT)FB2UW( pucEnd);
			pucEnd += (uiKL + 2);
			break;
		}
		
		case BT_LEAF_DATA:
		{
			FLMUINT		uiKL;
			FLMUINT		uiDL;
			FLMUINT		ucFlags = *pucEnd;

			pucEnd++;

			if (ucFlags & BTE_FLAG_KEY_LEN)
			{
				uiKL = (FLMUINT)FB2UW( pucEnd);
				pucEnd += 2;
			}
			else
			{
				uiKL = *pucEnd;
				pucEnd++;
			}

			if (ucFlags & BTE_FLAG_DATA_LEN)
			{
				uiDL = FB2UW( pucEnd);
				pucEnd += 2;
			}
			else
			{
				uiDL = *pucEnd;
				pucEnd++;
			}

			if (ucFlags & BTE_FLAG_OA_DATA_LEN)
			{
				pucEnd += 4;
			}

			pucEnd += (uiKL + uiDL);
			break;
		}
		
		case BT_NON_LEAF:
		{
			FLMUINT		uiKL;
			
			pucEnd += 4;
			uiKL = (FLMUINT)FB2UW( pucEnd);
			pucEnd += (uiKL + 2);
			break;
		}
		case BT_NON_LEAF_COUNTS:
		{
			FLMUINT		uiKL;

			pucEnd += 8;
			uiKL = (FLMUINT)FB2UW( pucEnd);
			pucEnd += (uiKL + 2);
			break;
		}
	}

	return( pucEnd);
}

/***************************************************************************
Desc:	Compare two cache blocks during a sort to determine which 
		one has lower address.
*****************************************************************************/
FINLINE FLMINT XFLAPI blkSortCompare(
	void *		pvBuffer,
	FLMUINT		uiPos1,
	FLMUINT		uiPos2)
{
	BlkStruct *	pBlk1 = ((BlkStruct *)pvBuffer) + uiPos1;
	BlkStruct *	pBlk2 = ((BlkStruct *)pvBuffer) + uiPos2;

	if (pBlk1->uiStartOfEntry < pBlk2->uiStartOfEntry)
	{
		return( -1);
	}
	else if (pBlk1->uiStartOfEntry > pBlk2->uiStartOfEntry)
	{
		return( 1);
	}
	else
	{
		return( 0);
	}
}

/***************************************************************************
Desc:	Swap two entries in cache table during sort.
*****************************************************************************/
FINLINE void XFLAPI blkSortSwap(
	void *		pvBuffer,
	FLMUINT		uiPos1,
	FLMUINT		uiPos2)
{
	BlkStruct *	pBlkEntryTable = (BlkStruct *)pvBuffer;
	BlkStruct	TmpEntry = pBlkEntryTable [uiPos1];
	
	pBlkEntryTable[ uiPos1] = pBlkEntryTable[ uiPos2];
	pBlkEntryTable[ uiPos2] = TmpEntry;
}

/********************************************************************
Desc:	
*********************************************************************/
RCODE F_DbCheck::verifyBlockStructure(
	FLMUINT						uiBlockSize,
	F_BTREE_BLK_HDR *			pBlkHdr)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiIndex;
	FLMBYTE *		pucEntry;
	FLMBYTE *		pucEnd;
	BlkStruct *		pCurBlk;
	FLMUINT			uiNumEntries;
	
	// No need to check anything in the block if there are less than 2 entries.
	
	if ((uiNumEntries = pBlkHdr->ui16NumKeys) < 2)
	{
		goto Exit;
	}

	if (uiNumEntries > m_uiBlkEntryArraySize)
	{
		FLMUINT	uiNewEntryArraySize = uiNumEntries + 200;
		
		if (RC_BAD( rc = f_realloc( sizeof( BlkStruct) * uiNewEntryArraySize,
									&m_pBlkEntries)))
		{
			goto Exit;
		}
		f_memset( &m_pBlkEntries [m_uiBlkEntryArraySize], 0,
				sizeof( BlkStruct) * (uiNewEntryArraySize - m_uiBlkEntryArraySize));
		m_uiBlkEntryArraySize = uiNewEntryArraySize;
	}
		
	for (uiIndex = 0, pCurBlk = m_pBlkEntries;
		  uiIndex < uiNumEntries;
		  uiIndex++, pCurBlk++)
	{
		pucEntry = BtEntry( (FLMBYTE *)pBlkHdr, uiIndex);
		
		if( pucEntry > (FLMBYTE *)pBlkHdr + uiBlockSize)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_FAILURE);
			goto Exit;
		}

		pucEnd = getEntryEnd( pucEntry, pBlkHdr->stdBlkHdr.ui8BlkType);
		pCurBlk->uiStartOfEntry = (FLMUINT)pucEntry;
		pCurBlk->uiEndOfEntry = (FLMUINT)pucEnd;
	}
	
	// Now sort the entries and check the offsets.
	
	f_qsort( m_pBlkEntries, 0, uiNumEntries - 1, blkSortCompare, blkSortSwap);
	
	for (uiIndex = 1; uiIndex < uiNumEntries; uiIndex++)
	{
		if (m_pBlkEntries [uiIndex - 1].uiEndOfEntry >
			 m_pBlkEntries [uiIndex].uiStartOfEntry)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_BTREE_ERROR);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/********************************************************************
Desc:	Verify the reference set for a key
*********************************************************************/
RCODE F_DbCheck::verifyIXRefs(
	STATE_INFO *	pStateInfo,
	FLMUINT64		ui64ResetNodeId
	)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT64	ui64NodeId;

	// Get the NodeId for the element.
	ui64NodeId = pStateInfo->ui64ElmNodeId;
	
	if (ui64NodeId <= ui64ResetNodeId)
	{
		ui64ResetNodeId = 0;
	}

	if (!ui64ResetNodeId && !m_bPhysicalCorrupt)
	{
		if (RC_BAD( rc = verifyIXRSet( pStateInfo)))
		{
			goto Exit;
		}
	}

	pStateInfo->ui64KeyRefs++;

Exit:

	return( rc);
}


/********************************************************************
Desc:	Initialize the STATE_INFO in preparation to do checking.
*********************************************************************/
void flmInitReadState(
	STATE_INFO *	pStateInfo,
	FLMBOOL *		pbStateInitialized,
	FLMUINT,			//uiVersionNum,
	F_Db *			pDb,					// May be NULL.
	LF_HDR *,		//pLogicalFile,
	FLMUINT			uiLevel,
	FLMUINT			uiExpectedBlkType,
	FLMBYTE *		pucKeyBuffer)
{

	f_memset( pStateInfo, 0, sizeof( STATE_INFO));
	*pbStateInitialized = TRUE;
	pStateInfo->pDb = pDb;
	pStateInfo->uiLevel = uiLevel;

	pStateInfo->uiBlkType = uiExpectedBlkType;
	pStateInfo->pucElmKey = pucKeyBuffer;
	pStateInfo->uiElmLastFlag = 0xFF;
	pStateInfo->ui32NextBlkAddr = 0xFFFFFFFF;
	pStateInfo->ui32LastChildAddr = 0xFFFFFFFF;
}

/******************************************************************************
Desc:	Constructor
******************************************************************************/
F_NodeVerifier::F_NodeVerifier()
{
	m_pucBuf = &m_ucBuf [0];
	m_uiBufSize = sizeof( m_ucBuf);
	m_uiBytesInBuf = 0;
	m_pXRefRS = NULL;
	m_bFinalizeCalled = FALSE;
	m_pRS = NULL;
	
	f_memset( &m_nodeInfo, 0, sizeof( F_NODE_INFO));
}


/******************************************************************************
Desc:	Destructor
******************************************************************************/
F_NodeVerifier::~F_NodeVerifier()
{
	if (m_pRS)
	{
		m_pRS->Release();
	}
	if (m_pucBuf != &m_ucBuf [0])
	{
		f_free( &m_pucBuf);
	}
}

/******************************************************************************
Desc:	Reset
******************************************************************************/
void F_NodeVerifier::Reset(
	STATE_INFO *	pStateInfo)
{
	m_uiBytesInBuf = 0;
	m_bFinalizeCalled = FALSE;
	f_memset( m_pucBuf, 0, m_uiBufSize);
	f_memset( &m_nodeInfo, 0, sizeof( F_NODE_INFO));
	
	if( *pStateInfo->pucElm & BTE_FLAG_OA_DATA_LEN)
	{
		m_uiOverallLength = pStateInfo->uiElmOADataLen;
	}
	else
	{
		m_uiOverallLength = pStateInfo->uiElmDataLen;
	}
}

/******************************************************************************
Desc:
******************************************************************************/
RCODE F_NodeVerifier::AddData(
	FLMUINT64		ui64NodeId,
	void *			pucData,
	FLMUINT			uiDataLen)
{
	RCODE				rc = NE_XFLM_OK;

	// Verify the node Id or Save it (first time)
	if( m_nodeInfo.ui64NodeId)
	{
		if (m_nodeInfo.ui64NodeId != ui64NodeId)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_FAILURE);
			goto Exit;
		}
	}
	else
	{
		m_nodeInfo.ui64NodeId = ui64NodeId;
	}

	// Copy the entry data into the local buffer.  At most,
	// we only need MAX_DOM_HEADER_SIZE bytes to make a 
	// complete DOM node header - unless we have an
	// element node, in which case we want to get all of
	// the attribute nodes.

	if (uiDataLen + m_uiBytesInBuf > m_uiBufSize)
	{
		eDomNodeType	eNodeType;

		// Node type should be in the first byte of data we have for
		// the node.

		if (!m_uiBytesInBuf)
		{
			eNodeType = (eDomNodeType)((*((FLMBYTE *)pucData)) & 0x0F);
		}
		else
		{
			eNodeType = (eDomNodeType)((*m_pucBuf) & 0x0F);
		}
		if (eNodeType != ELEMENT_NODE)
		{
			flmAssert( m_uiBufSize >= MAX_DOM_HEADER_SIZE);

			// Only copy enough to fill up the current buffer - don't
			// really need any more than the header for non-element nodes.

			uiDataLen = m_uiBufSize - m_uiBytesInBuf;
		}
		else
		{
			// Must get everything for element nodes

			if (m_pucBuf != &m_ucBuf [0])
			{
				if (RC_BAD( rc = f_realloc( uiDataLen + m_uiBytesInBuf, &m_pucBuf)))
				{
					goto Exit;
				}
			}
			else
			{
				FLMBYTE *	pucNew;

				if (RC_BAD( rc = f_alloc( uiDataLen + m_uiBytesInBuf, &pucNew)))
				{
					goto Exit;
				}
				if (m_uiBytesInBuf)
				{
					f_memcpy( pucNew, m_pucBuf, m_uiBytesInBuf);
				}
				m_pucBuf = pucNew;
			}
			m_uiBufSize = uiDataLen + m_uiBytesInBuf;
		}
	}

	f_memcpy( &m_pucBuf[m_uiBytesInBuf], pucData, uiDataLen);
	m_uiBytesInBuf += uiDataLen;

Exit:

	return rc;
}

/********************************************************************
Desc: Do the final setup to store the node information in the Node 
		Result Set.
*********************************************************************/
RCODE F_NodeVerifier::finalize(
	F_Db *				pDb,
	F_Dict *				pDict,
	FLMUINT				uiCollection,
	FLMUINT64			ui64NodeId,
	FLMBOOL				bSkipDOMLinkCheck,
	FLMINT32 *			pi32ElmErrCodeRV)
{
	RCODE						rc	= NE_XFLM_OK;
	NODE_RS_ENTRY *		pRSEntry = NULL;
	FLMUINT					uiRSBufIndex;
	FLMUINT					uiStorageFlags;
	F_NameTable *			pNameTable = pDict->getNameTable();
	IF_BufferIStream *	pBufferStream = NULL;

	*pi32ElmErrCodeRV = 0;
	
	if( m_bFinalizeCalled)
	{
		flmAssert( 0);
		goto Exit;
	}
	
	f_memset( &m_nodeInfo, 0, sizeof( F_NODE_INFO));
	
	if( RC_BAD( rc = FlmAllocBufferIStream( &pBufferStream)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pBufferStream->openStream( 
		(const char *)m_pucBuf, m_uiBytesInBuf)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = flmReadNodeInfo( uiCollection, ui64NodeId, 
		pBufferStream, m_uiOverallLength, FALSE, &m_nodeInfo, &uiStorageFlags)))
	{
		goto Exit;
	}

	// See if child element count is set.

	if( uiStorageFlags & NSF_HAVE_CELM_LIST_BIT)
	{
		// This bit should only be set for elements.
		
		if( m_nodeInfo.eNodeType != ELEMENT_NODE)
		{
			*pi32ElmErrCodeRV = FLM_BAD_NODE_TYPE;
			goto Exit;
		}
		
		// If the count > 0, the NSF_HAVE_CHILDREN_BIT better also be set.
		
		if( m_nodeInfo.uiChildElmCount)
		{
			if( !(uiStorageFlags & NSF_HAVE_CHILDREN_BIT))
			{
				*pi32ElmErrCodeRV = FLM_BAD_CHILD_ELM_COUNT;
				goto Exit;
			}
		}
	}
	
	// Verify the Name and Prefix Ids.
	
	if( RC_BAD( rc = verifyNameId( pDb, m_nodeInfo.eNodeType,
			m_nodeInfo.uiNameId, pNameTable, pi32ElmErrCodeRV)))
	{
		goto Exit;
	}

	if( *pi32ElmErrCodeRV)
	{
		goto Exit;
	}

	if( RC_BAD( rc = verifyPrefixId( pDb, 
		m_nodeInfo.uiPrefixId, pNameTable, pi32ElmErrCodeRV)))
	{
		goto Exit;
	}

	if( *pi32ElmErrCodeRV)
	{
		goto Exit;
	}

	if( !bSkipDOMLinkCheck)
	{
		FLMUINT16			ui16BitMap = 0;
		
		// Build a buffer to set into the Result Set for later verification...

		if( RC_BAD( rc = f_calloc( sizeof( NODE_RS_ENTRY), &pRSEntry)))
		{
			goto Exit;
		}

		uiRSBufIndex = 0;
		if( m_nodeInfo.ui64DocumentId)
		{
			pRSEntry->ui64FieldArray[ uiRSBufIndex++] = m_nodeInfo.ui64DocumentId;
			ui16BitMap |= CHK_BM_ROOT_ID;
		}

		if( m_nodeInfo.ui64ParentId)
		{
			pRSEntry->ui64FieldArray[ uiRSBufIndex++] = m_nodeInfo.ui64ParentId;
			ui16BitMap |= CHK_BM_PARENT_ID;
		}

		if( m_nodeInfo.ui64PrevSibId)
		{
			pRSEntry->ui64FieldArray[ uiRSBufIndex++] = m_nodeInfo.ui64PrevSibId;
			ui16BitMap |= CHK_BM_PREV_SIBLING;
		}

		if( m_nodeInfo.ui64NextSibId)
		{
			pRSEntry->ui64FieldArray[ uiRSBufIndex++] = m_nodeInfo.ui64NextSibId;
			ui16BitMap |= CHK_BM_NEXT_SIBLING;
		}

		if( m_nodeInfo.ui64FirstChildId)
		{
			pRSEntry->ui64FieldArray[ uiRSBufIndex++] = m_nodeInfo.ui64FirstChildId;
			ui16BitMap |= CHK_BM_FIRST_CHILD;
		}

		if( m_nodeInfo.ui64LastChildId)
		{
			pRSEntry->ui64FieldArray[ uiRSBufIndex++] = m_nodeInfo.ui64LastChildId;
			ui16BitMap |= CHK_BM_LAST_CHILD;
		}

		if( m_nodeInfo.ui64AnnotationId)
		{
			pRSEntry->ui64FieldArray[ uiRSBufIndex++] = m_nodeInfo.ui64AnnotationId;
			ui16BitMap |= CHK_BM_ANNOTATION;
		}

		pRSEntry->hdr.ui64NodeId = m_nodeInfo.ui64NodeId;
		pRSEntry->hdr.ui16BitMap = ui16BitMap;
		m_bFinalizeCalled = TRUE;

		// Finally add the result to the result set for later evaluation.

		if( RC_BAD( rc = m_pRS->addEntry( NULL, NULL,
			(FLMBYTE *)&(pRSEntry->hdr.ui64NodeId),
			sizeof( FLMUINT64), (FLMBYTE *)pRSEntry, 
			sizeof( NODE_RS_HDR) + (uiRSBufIndex * sizeof( FLMUINT64)))))
		{
			*pi32ElmErrCodeRV = -1;
			goto Exit;
		}
	}

	if( m_pXRefRS)
	{
		if (RC_BAD( rc = checkForIndexes( pDb, pDict, uiCollection)))
		{
			goto Exit;
		}
	}

Exit:

	if( pBufferStream)
	{
		pBufferStream->Release();
	}

	if( pRSEntry)
	{
		f_free( &pRSEntry);
	}

	return( rc);
}

/******************************************************************************
Desc:	Method to add any indexes to the document list that qualify.
******************************************************************************/
RCODE F_NodeVerifier::checkForIndexes( 
	F_Db *			pDb,
	F_Dict *			pDict,
	FLMUINT			uiCollection)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT64			ui64DocumentId;
	DOC_IXD_XREF		DocXRef;
	ICD *					pIcd;
	F_AttrElmInfo		defInfo;

	// Determine if there are any indexs that qualify to be included in the
	// index list.
	//
	// If there is no document id, then this node is the the root node?
	
	ui64DocumentId = m_nodeInfo.ui64DocumentId;
	
	if( !ui64DocumentId)
	{
		if( !m_nodeInfo.ui64ParentId)
		{
			ui64DocumentId = m_nodeInfo.ui64NodeId;
		}
		else
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_FAILURE);
		}
	}

	switch( m_nodeInfo.eNodeType)
	{
		case ELEMENT_NODE:
		case DATA_NODE:
		{
			if( RC_BAD( rc = pDict->getElement( pDb, 
				m_nodeInfo.uiNameId, &defInfo)))
			{
				goto Exit;
			}
			break;
		}
		
		case ATTRIBUTE_NODE:
		{
			if (RC_BAD( rc = pDict->getAttribute( pDb, m_nodeInfo.uiNameId,
				&defInfo)))
			{
				goto Exit;
			}
			break;
		}
		
		default:
		{
			goto Exit;
		}
	}

	pIcd = defInfo.m_pFirstIcd;

	// Do we have any qualifying indexes?

	while (pIcd)
	{
		// Only process if the target collection matches.
		
		if( pIcd->pIxd->uiCollectionNum == uiCollection)
		{
			if( pIcd->uiFlags & (ICD_REQUIRED_PIECE | ICD_REQUIRED_IN_SET))
			{
				FLMBYTE		ucDummy = 0;

				// Get the document list entry for this node.
				// Build a buffer to set into the Result Set for later verification...

				DocXRef.uiIndexNum = pIcd->pIxd->uiIndexNum;
				DocXRef.ui64DocId = m_nodeInfo.ui64DocumentId;
				DocXRef.uiCollection = uiCollection;

				if (RC_BAD( rc = m_pXRefRS->addEntry( NULL, NULL,
																(FLMBYTE *)&DocXRef,
																sizeof( DOC_IXD_XREF),
																&ucDummy, 1)))
				{
					// It's okay if we get a duplicate entry, only one will be saved in
					// the result set.

					if (rc == NE_XFLM_NOT_UNIQUE)
					{
						rc = NE_XFLM_OK;
					}
					else
					{
						goto Exit;
					}
				}
			}
		}
		pIcd = pIcd->pNextInChain;
	}

Exit:

	return rc;
}

/******************************************************************************
Desc:
******************************************************************************/
FLMUINT64 getLinkVal(
	FLMUINT				uiBMId,
	NODE_RS_ENTRY *	pRSEntry
	)
{
	FLMUINT64 *		pui64Ptr = pRSEntry->ui64FieldArray;
	FLMUINT16		ui16BitMap = pRSEntry->hdr.ui16BitMap;
	FLMUINT64		ui64Link = 0;

	if (ui16BitMap & CHK_BM_ROOT_ID)
	{
		if (uiBMId == CHK_BM_ROOT_ID)
		{
			ui64Link = *pui64Ptr;
			goto Exit;
		}
		pui64Ptr++;
	}

	if (ui16BitMap & CHK_BM_PARENT_ID)
	{
		if (uiBMId == CHK_BM_PARENT_ID)
		{
			ui64Link = *pui64Ptr;
			goto Exit;
		}
		pui64Ptr++;
	}

	if (ui16BitMap & CHK_BM_PREV_SIBLING)
	{
		if (uiBMId == CHK_BM_PREV_SIBLING)
		{
			ui64Link = *pui64Ptr;
			goto Exit;
		}
		pui64Ptr++;
	}

	if (ui16BitMap & CHK_BM_NEXT_SIBLING)
	{
		if (uiBMId == CHK_BM_NEXT_SIBLING)
		{
			ui64Link = *pui64Ptr;
			goto Exit;
		}
		pui64Ptr++;
	}

	if (ui16BitMap & CHK_BM_FIRST_CHILD)
	{
		if (uiBMId == CHK_BM_FIRST_CHILD)
		{
			ui64Link = *pui64Ptr;
			goto Exit;
		}
		pui64Ptr++;
	}

	if (ui16BitMap & CHK_BM_LAST_CHILD)
	{
		if (uiBMId == CHK_BM_LAST_CHILD)
		{
			ui64Link = *pui64Ptr;
			goto Exit;
		}
		pui64Ptr++;
	}

	if (ui16BitMap & CHK_BM_ANNOTATION)
	{
		if (uiBMId == CHK_BM_ANNOTATION)
		{
			ui64Link = *pui64Ptr;
			goto Exit;
		}
	}


Exit:

	return ui64Link;
}


/******************************************************************************
Desc:	Verify that the nameId is in the dictionary.
******************************************************************************/
RCODE F_NodeVerifier::verifyNameId(
	F_Db *				pDb,
	eDomNodeType		eNodeType,
	FLMUINT				uiNameId,
	F_NameTable *		pNameTable,
	FLMINT32 *			pi32ErrCode)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiType;
	FLMUINT				uiLen;
	
	if( !uiNameId)
	{
		goto Exit;
	}

	switch (eNodeType)
	{
		case DOCUMENT_NODE:
		case ELEMENT_NODE:
		case DATA_NODE:
		case COMMENT_NODE:
		case CDATA_SECTION_NODE:
		case ANNOTATION_NODE:
		{
			uiType = ELM_ELEMENT_TAG;
			break;
		}
		case ATTRIBUTE_NODE:
		{
			uiType = ELM_ATTRIBUTE_TAG;
			break;
		}
		default:
		{
			flmAssert( 0);
			*pi32ErrCode = FLM_UNSUPPORTED_NODE_TYPE;
			goto Exit;
		}
	}

	if (RC_BAD( rc = pNameTable->getFromTagTypeAndNum( pDb, uiType,
		uiNameId, NULL, NULL, &uiLen, NULL, NULL, NULL, NULL, TRUE)))
	{
		*pi32ErrCode = FLM_BAD_INVALID_NAME_ID;
		goto Exit;
	}


Exit:

	return rc;
}

/******************************************************************************
Desc:	Verify that the prefixId is in the dictionary.
******************************************************************************/
RCODE F_NodeVerifier::verifyPrefixId(
	F_Db *				pDb,
	FLMUINT				uiPrefixId,
	F_NameTable *		pNameTable,
	FLMINT32 *			pi32ErrCode
	)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiLen;

	if (!uiPrefixId)
	{
		goto Exit;  // Ok.
	}

	if (RC_BAD( rc = pNameTable->getFromTagTypeAndNum(
								pDb, ELM_PREFIX_TAG, uiPrefixId, NULL, NULL, &uiLen)))
	{
		*pi32ErrCode = FLM_BAD_INVALID_PREFIX_ID;
		goto Exit;
	}

Exit:

	return rc;
}

/******************************************************************************
Desc:	Add a key to the result set.
******************************************************************************/
RCODE F_KeyCollector::addKey(
	F_Db *			pDb,
	IXD *				pIxd,
	KREF_ENTRY *	pKref
	)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBYTE *	pucKey = (FLMBYTE *)&pKref[1];
	FLMBYTE *	pucData = pucKey + pKref->ui16KeyLen;
	FLMUINT		uiDataLen;

	flmAssert( pKref->ui16KeyLen <= XFLM_MAX_KEY_SIZE);
	
	// Can't store an entry with zero length data.
	
	if ((uiDataLen = pKref->uiDataLen) == 0)
	{
		*pucData = 0;
		uiDataLen = 1;
	}

	// Save the key in the result set.
	
	if (RC_BAD( rc = m_pDbCheck->m_pIxRSet->addEntry( pDb, pIxd, pucKey,
								(FLMUINT)pKref->ui16KeyLen,
								pucData, uiDataLen)))
	{
		if (rc == NE_XFLM_NOT_UNIQUE)
		{
			rc = NE_XFLM_OK;
		}
		goto Exit;
	}

	m_ui64TotalKeys++;

Exit:

	return( rc);
}

