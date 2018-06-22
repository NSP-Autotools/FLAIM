//-------------------------------------------------------------------------
// Desc:	B-tree searching.
// Tabs:	3
//
// Copyright (c) 1990-2007 Novell, Inc. All Rights Reserved.
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

FSTATIC FLMUINT FSKeyCmp(
	BTSK *			pStack,
	FLMBYTE *		key,
	FLMUINT			uiKeyLen,
	FLMUINT			drnDomain);

/****************************************************************************
Desc:	Search the b-tree for a matching key.
****************************************************************************/
RCODE FSBtSearch(
	FDB *			pDb,
	LFILE *		pLFile,
	BTSK **		pStackRV,
	FLMBYTE *	key,
	FLMUINT		keyLen,
	FLMUINT		dinDomain)
{
	RCODE			rc = FERR_OK;
	BTSK *		pStack = *pStackRV;
	FLMBYTE *	pKeyBuf = pStack->pKeyBuf;
	FLMUINT		uiBlkAddr;
	FLMUINT		uiKeyBufSize;
	LFILE			TmpLFile;

	uiKeyBufSize = (pLFile->uiLfType == LF_INDEX) ? MAX_KEY_SIZ : DIN_KEY_SIZ;

	// Get the correct root block specified in the LFILE.

	if (RC_BAD( rc = FSGetRootBlock( pDb, &pLFile, &TmpLFile, pStack)))
	{
		if (rc == FERR_NO_ROOT_BLOCK)
		{
			flmAssert( pLFile->uiRootBlk == BT_END);
			rc = FERR_OK;
		}

		goto Exit;
	}

	// MAIN LOOP Read each block going down the b-tree. Save state
	// information in the pStack[].

	for (;;)
	{
		pStack->uiFlags = FULL_STACK;
		pStack->uiKeyBufSize = uiKeyBufSize;
		
		if (pStack->uiBlkType != BHT_NON_LEAF_DATA)
		{
			rc = FSBtScan( pStack, key, keyLen, dinDomain);
		}
		else
		{
			rc = FSBtScanNonLeafData( pStack, 
						keyLen == 1 
							? (FLMUINT) *key 
							: (FLMUINT) f_bigEndianToUINT32( key));
		}

		if (RC_BAD( rc))
		{
			goto Exit;
		}

		if (!pStack->uiLevel)
		{
			// Leaf level - we are done.
			
			break;
		}

		uiBlkAddr = FSChildBlkAddr( pStack);

		pStack++;
		pStack->pKeyBuf = pKeyBuf;

		if (RC_BAD( rc = FSGetBlock( pDb, pLFile, uiBlkAddr, pStack)))
		{
			goto Exit;
		}
	}

	*pStackRV = pStack;

Exit:

	return (rc);
}

/****************************************************************************
Desc:	Search the right-most end of the b-tree.
****************************************************************************/
RCODE FSBtSearchEnd(
	FDB *			pDb,
	LFILE *		pLFile,		// Logical file definition
	BTSK **		pStackRV,	// Stack of variables for each level
	FLMUINT		uiDrn)		// Used to position and setup for update
{
	RCODE			rc = FERR_OK;
	BTSK *		pStack = *pStackRV;
	FLMBYTE *	pKeyBuf = pStack->pKeyBuf;
	FLMBYTE		key[ DIN_KEY_SIZ + 4];
	FLMUINT		uiBlkAddr;
	LFILE			TmpLFile;

	// Get the correct root block specified in the LFILE.

	if (RC_BAD( rc = FSGetRootBlock( pDb, &pLFile, &TmpLFile, pStack)))
	{
		if (rc == FERR_NO_ROOT_BLOCK)
		{
			flmAssert( pLFile->uiRootBlk == BT_END);
			rc = FERR_OK;
		}

		goto Exit;
	}

	f_UINT32ToBigEndian( (FLMUINT32)uiDrn, key);
	
	for (;;)
	{
		pStack->uiFlags = FULL_STACK;
		pStack->uiKeyBufSize = DIN_KEY_SIZ;

		// Remove all scanning from non-leaf data blocks (both formats).

		if (pStack->uiLevel)
		{
			pStack->uiCurElm = pStack->uiBlkEnd;	// Position past last element
			FSBtPrevElm( pDb, pLFile, pStack);		// Build full key in pKeyBuf[]
		}
		else
		{
			if (pStack->uiBlkType != BHT_NON_LEAF_DATA)
			{
				rc = FSBtScan( pStack, key, DIN_KEY_SIZ, 0);
			}
			else
			{
				rc = FSBtScanNonLeafData( pStack, uiDrn);
			}

			if (RC_BAD( rc))
			{
				goto Exit;
			}
		}

		if (!pStack->uiLevel)
		{
			// Leaf level - we are done.
			
			break;
		}

		uiBlkAddr = FSChildBlkAddr( pStack);
		pStack++;
		pStack->pKeyBuf = pKeyBuf;

		if (RC_BAD( rc = FSGetBlock( pDb, pLFile, uiBlkAddr, pStack)))
		{
			goto Exit;
		}
	}

	*pStackRV = pStack;

Exit:

	return (rc);
}

/****************************************************************************
Desc: Returns the root block of a passed-in LFILE
****************************************************************************/
RCODE FSGetRootBlock(
	FDB *			pDb,
	LFILE **		ppLFile,
	LFILE *		pTmpLFile,
	BTSK *		pStack)
{
	RCODE			rc = FERR_OK;
	LFILE *		pLFile = *ppLFile;
	FLMUINT		uiBlkAddr;
	FLMBOOL		bRereadLFH = FALSE;

	// Make Sure this is the correct root block in the LFILE area. If not
	// then read in the LFH structure and try again. It would be nice to
	// have a routine that reads only root blocks. DSS: Added check for
	// uiBlkAddr >= pDb->Loghdr.uiLogicalEOF because the pLFile could have
	// a root block address of an aborted update transaction where the root
	// block has not yet been fixed up by the aborting transaction (in a
	// shared environment).

	if (((uiBlkAddr = pLFile->uiRootBlk) == BT_END) ||
		 (uiBlkAddr >= pDb->LogHdr.uiLogicalEOF))
	{
		bRereadLFH = TRUE;
	}
	else if (RC_BAD( rc = FSGetBlock( pDb, pLFile, uiBlkAddr, pStack)))
	{
		if (rc == FERR_DATA_ERROR || (rc == FERR_OLD_VIEW && !pDb->uiKilledTime))
		{
			bRereadLFH = TRUE;
			pStack->uiBlkAddr = BT_END;
		}
		else
		{
			goto Exit;
		}
	}
	else
	{
		// Check for valid root block - Root Flag and Logical file number		
		
		FLMBYTE *		pBlk = pStack->pBlk;

		if (!(BH_IS_ROOT_BLK( pBlk)) ||
			 (pLFile->uiLfNum != FB2UW( &pBlk[BH_LOG_FILE_NUM])))
		{
			bRereadLFH = TRUE;
			FSReleaseBlock( pStack, FALSE);
			pStack->uiBlkAddr = BT_END;
		}
	}

	// Reread the LFH from disk if we do not have the root block

	if (bRereadLFH)
	{

		// If we are in a read transaction, copy the LFILE structure so
		// that we don't mess up a thread that may be doing an update.  

		if (flmGetDbTransType( pDb) == FLM_READ_TRANS)
		{
			f_memcpy( pTmpLFile, pLFile, sizeof(LFILE));
			pLFile = pTmpLFile;
		}

		if (RC_BAD( rc = flmLFileRead( pDb, pLFile)))
		{
			goto Exit;
		}

		// If there is no root block, return right away

		if ((uiBlkAddr = pLFile->uiRootBlk) == BT_END)
		{

			// The caller of FSGetRootBlock is expected to check for and
			// handle FERR_NO_ROOT_BLOCK. It should NEVER be returned to the
			// application. NOTE: Checking for BT_END_OF_DATA will not work
			// in every case to check for no root block because it is not
			// always initialized before calling FSGetRootBlock, so it could
			// have garbage in it if we don't end up going through this code
			// path.

			rc = RC_SET( FERR_NO_ROOT_BLOCK);
			pStack->uiCmpStatus = BT_END_OF_DATA;
			pStack->uiBlkAddr = BT_END;
			goto Exit;
		}

		if (RC_BAD( rc = FSGetBlock( pDb, pLFile, uiBlkAddr, pStack)))
		{
			goto Exit;
		}
	}

Exit:

	*ppLFile = pLFile;
	return (rc);
}

/****************************************************************************
Desc:	Scan a b-tree block for a matching key at any b-tree block level.
****************************************************************************/
RCODE FSBtScan(
	BTSK *		pStack,				// [in/out] Stack of variables for each level
	FLMBYTE *	pSearchKey, 		// The input key to search for
	FLMUINT		uiSearchKeyLen,	// Length of the key (not null terminated)
	FLMUINT		dinDomain)			// INDEXES ONLY - lower bounds of din
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pCurElm;				// Points to the current element.
	FLMBYTE *	pBlk;					// Points to the cache block.
	FLMBYTE *	pKeyBuf;				// Points to pStack->pKeyBuf (optimization).
	FLMBYTE *	pElmKey;				// Points to the key within the element.
	FLMUINT		uiRecLen = 0;		// Length of the record portion.
	FLMUINT		uiPrevKeyCnt;		// Number left end bytes compressed
	FLMUINT		uiElmKeyLen;		// Length of the current element's key portion
	FLMUINT		uiBlkType;			// B-tree block type - Leaf or non-leaf.
	FLMUINT		uiElmOvhd;			// Number bytes overhead for element.
	FLMUINT		uiBytesMatched;	// Number of bytes matched with pSearchKey

	uiBlkType = pStack->uiBlkType;
	flmAssert( uiBlkType != BHT_NON_LEAF_DATA);

	// Initialize stack variables for possibly better performance.

	pKeyBuf = pStack->pKeyBuf;
	pBlk = pStack->pBlk;
	uiElmOvhd = pStack->uiElmOvhd;
	pStack->uiCurElm = BH_OVHD;
	pStack->uiKeyLen = pStack->uiPKC = pStack->uiPrevElmPKC = 0;
	uiBytesMatched = 0;

	for (;;)
	{
		pCurElm = &pBlk[pStack->uiCurElm];
		uiElmKeyLen = BBE_GETR_KL( pCurElm);

		// Read in RAW mode - doesn't do all bit checking

		if ((uiPrevKeyCnt = (BBE_GETR_PKC( pCurElm))) > BBE_PKC_MAX)
		{
			uiElmKeyLen += (uiPrevKeyCnt & BBE_KL_HBITS) << BBE_KL_SHIFT_BITS;
			uiPrevKeyCnt &= BBE_PKC_MAX;
		}

		// Should not have a non-zero PKC if we are on the first element of
		// a block

		if (uiPrevKeyCnt && pStack->uiCurElm == BH_OVHD)
		{
			rc = RC_SET( FERR_DATA_ERROR);
			goto Exit;
		}

		// Get the record portion length when on the leaf blocks.

		if (uiBlkType == BHT_LEAF)
		{
			uiRecLen = BBE_GET_RL( pCurElm);
		}

		pStack->uiPrevElmPKC = pStack->uiPKC;

		// The zero length key is the terminating element in a right-most
		// block.

		if ((pStack->uiKeyLen = uiPrevKeyCnt + uiElmKeyLen) == 0)
		{
			pStack->uiPrevElmPKC = f_min( uiBytesMatched, BBE_PKC_MAX);
			pStack->uiPKC = 0;
			pStack->uiCmpStatus = BT_END_OF_DATA;
			goto Exit;
		}

		// Handle special case of left-end compression maxing out.

		if (uiPrevKeyCnt == BBE_PKC_MAX && BBE_PKC_MAX < uiBytesMatched)
		{
			uiBytesMatched = BBE_PKC_MAX;
		}

		// Check out this element to see if the key matches.

		if (uiPrevKeyCnt == uiBytesMatched)
		{
			pElmKey = &pCurElm[uiElmOvhd];
			for (;;)
			{

				// All bytes of the search key are matched?

				if (uiBytesMatched == uiSearchKeyLen)
				{
					pStack->uiPKC = f_min( uiBytesMatched, BBE_PKC_MAX);

					// Build pKeyBuf with the search key because it matches.
					// Current key is either equal or greater than search key.

					if (uiSearchKeyLen < pStack->uiKeyLen)
					{
						f_memcpy( &pKeyBuf[uiSearchKeyLen], pElmKey,
									pStack->uiKeyLen - uiSearchKeyLen);
						pStack->uiCmpStatus = BT_GT_KEY;
					}
					else
					{
						if (dinDomain)
						{
							FLMBYTE*		pCurRef = pCurElm;
							if ((dinDomain - 1) < 
									FSGetDomain( &pCurRef, (FLMBYTE) uiElmOvhd))
							{

								// Keep going...

								goto Next_Element;
							}
						}

						pStack->uiCmpStatus = BT_EQ_KEY;
					}

					f_memcpy( pKeyBuf, pSearchKey, uiSearchKeyLen);
					goto Exit;
				}

				// .. else matches all the bytes in the element key.

				if (uiBytesMatched == pStack->uiKeyLen)
				{
					pStack->uiPKC = f_min( uiBytesMatched, BBE_PKC_MAX);

					// Need an outer break call here - forced to do a goto.

					goto Next_Element;
				}

				// Compare the next byte in the search key and element

				if (pSearchKey[uiBytesMatched] != *pElmKey)
				{
					break;
				}

				uiBytesMatched++;
				pElmKey++;
			}

			pStack->uiPKC = f_min( uiBytesMatched, BBE_PKC_MAX);

			// Check if we are done comparing, if so build pKeyBuf[].

			if (pSearchKey[uiBytesMatched] < *pElmKey)
			{
				if (uiBytesMatched)
				{
					f_memcpy( pKeyBuf, pSearchKey, uiBytesMatched);
				}

				f_memcpy( &pKeyBuf[uiBytesMatched], pElmKey,
							pStack->uiKeyLen - uiBytesMatched);
				pStack->uiCmpStatus = BT_GT_KEY;
				goto Exit;
			}
		}
		else if (uiPrevKeyCnt < uiBytesMatched)
		{

			// Current key > search key. Set pKeyBuf and break out.

			pStack->uiPKC = uiPrevKeyCnt;
			if (uiPrevKeyCnt)
			{
				f_memcpy( pKeyBuf, pSearchKey, uiPrevKeyCnt);
			}

			f_memcpy( &pKeyBuf[uiPrevKeyCnt], &pCurElm[uiElmOvhd], uiElmKeyLen);
			pStack->uiCmpStatus = BT_GT_KEY;
			goto Exit;
		}

		// else the key is less than the search key (uiPrevKeyCnt >
		// uiBytesMatched).

Next_Element:

		// Position to the next element

		pStack->uiCurElm += uiElmKeyLen + 
									((uiBlkType == BHT_LEAF) 
											? (BBE_KEY + uiRecLen)
											: (BNE_IS_DOMAIN( pCurElm) 
												? (BNE_DOMAIN_LEN + uiElmOvhd) 
												: uiElmOvhd));

		// Most common check first.

		if (pStack->uiCurElm < pStack->uiBlkEnd)
		{
			continue;
		}

		if (pStack->uiCurElm == pStack->uiBlkEnd)
		{

			// On the equals conditition it may be OK in some very special
			// cases.

			pStack->uiCmpStatus = BT_END_OF_DATA;
			goto Exit;
		}

		// Marched off the end of the block - something is corrupt.

		rc = RC_SET( FERR_CACHE_ERROR);
		goto Exit;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc:	Binary search into a non-leaf data record block.
****************************************************************************/
RCODE FSBtScanNonLeafData(
	BTSK *		pStack,
	FLMUINT		uiDrn)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pBlk = pStack->pBlk;
	FLMUINT		uiLow = 0;
	FLMUINT		uiMid;
	FLMUINT		uiHigh = ((pStack->uiBlkEnd - BH_OVHD) >> 3) - 1;
	FLMUINT		uiTblSize = uiHigh;
	FLMUINT		uiCurDrn;

	pStack->uiCmpStatus = BT_GT_KEY;
	for (;;)
	{
		uiMid = (uiLow + uiHigh) >> 1;

		uiCurDrn = f_bigEndianToUINT32( &pBlk[ BH_OVHD + (uiMid << 3)]);
		if (uiCurDrn == 0)
		{

			// Special case - at the end of a rightmost block.

			pStack->uiCmpStatus = BT_EQ_KEY;
			break;
		}

		if (uiDrn == uiCurDrn)
		{

			// Remember a data record can span multiple blocks (same DRN).

			while (uiMid)
			{
				uiCurDrn = f_bigEndianToUINT32( 
					&pBlk[ BH_OVHD + ((uiMid - 1) << 3)]);
					
				if (uiDrn != uiCurDrn)
				{
					break;
				}

				uiMid--;
			}

			pStack->uiCmpStatus = BT_EQ_KEY;
			break;
		}

		// Down to one item if too high then position to next item.

		if (uiLow >= uiHigh)
		{
			if ((uiDrn > uiCurDrn) && uiMid < uiTblSize)
			{
				uiMid++;
			}
			break;
		}

		// If too high then try lower section

		if (uiDrn < uiCurDrn)
		{

			// First item too high?

			if (uiMid == 0)
			{
				break;
			}

			uiHigh = uiMid - 1;
		}
		else
		{
			// Try upper section because mid value is too low.			
			
			if (uiMid == uiTblSize)
			{
				uiMid++;
				pStack->uiCmpStatus = BT_END_OF_DATA;
				break;
			}

			uiLow = uiMid + 1;
		}
	}

	// Set curElm and the key buffer.

	pStack->uiCurElm = BH_OVHD + (uiMid << 3);
	f_UINT32ToBigEndian( (FLMUINT32)uiCurDrn, pStack->pKeyBuf);
	return (rc);
}

/****************************************************************************
Desc: Read the block information and initialize all needed pStack
		elements.
****************************************************************************/
void FSBlkToStack(
	BTSK *		pStack)
{
	FLMBYTE *	pBlk = pStack->pBlk;
	FLMUINT		uiBlkType;

	pStack->uiBlkType = uiBlkType = (FLMUINT) (BH_GET_TYPE( pBlk));

	// The standard overhead is used in the pStack Compares are made to
	// determine if the element is extended.

	if (uiBlkType == BHT_LEAF)
	{
		pStack->uiElmOvhd = BBE_KEY;
	}
	else if (uiBlkType == BHT_NON_LEAF_DATA)
	{
		pStack->uiElmOvhd = BNE_DATA_OVHD;
	}
	else if (uiBlkType == BHT_NON_LEAF)
	{
		pStack->uiElmOvhd = BNE_KEY_START;
	}
	else if (uiBlkType == BHT_NON_LEAF_COUNTS)
	{
		pStack->uiElmOvhd = BNE_KEY_COUNTS_START;
	}
	else
	{
		flmAssert( 0);
		pStack->uiElmOvhd = BNE_KEY_START;
	}

	pStack->uiKeyLen = pStack->uiPKC = pStack->uiPrevElmPKC = 0;
	pStack->uiCurElm = BH_OVHD;
	pStack->uiBlkEnd = (FLMUINT) FB2UW( &pBlk[BH_ELM_END]);
	pStack->uiLevel = (FLMUINT) pBlk[BH_LEVEL];
}

/****************************************************************************
Desc:	Scan to a specific element (pStack->uiCurElm) in a b-tree block.
****************************************************************************/
RCODE FSBtScanTo(
	BTSK *		pStack,
	FLMBYTE *	pSearchKey,
	FLMUINT		uiSearchKeyLen,
	FLMUINT		dinDomain)
{
	RCODE				rc = FERR_OK;
	FLMBYTE *		pCurElm;
	FLMBYTE *		pBlk;
	FLMBYTE *		pKeyBuf = pStack->pKeyBuf;
	FLMBYTE *		pPrevElm;
	FLMUINT			uiPrevKeyCnt = 0;
	FLMUINT			uiElmKeyLen = 0;
	FLMUINT			uiTargetCurElm = pStack->uiCurElm;
	FLMUINT			uiElmOvhd;
	FLMUINT			uiKeyBufLen;

	FSBlkToStack( pStack);
	pBlk = pStack->pBlk;
	uiElmOvhd = pStack->uiElmOvhd;

	if (uiTargetCurElm > pStack->uiBlkEnd)
	{
		uiTargetCurElm = pStack->uiBlkEnd;
	}

	// The code is easy for non-leaf data blocks.

	if (pStack->uiBlkType == BHT_NON_LEAF_DATA)
	{

		// Target may be any byte offset in the block.

		while (pStack->uiCurElm < uiTargetCurElm)
		{
			pStack->uiCurElm += BNE_DATA_OVHD;
		}

		if (uiTargetCurElm < pStack->uiBlkEnd)
		{
			flmCopyDrnKey( pKeyBuf, &pBlk[pStack->uiCurElm]);
			pStack->uiCmpStatus = BT_EQ_KEY;
		}
		else
		{
			pStack->uiCmpStatus = BT_END_OF_DATA;
		}

		goto Exit;
	}

	// Note: There is no way pPrevElm can be accessed and point to NULL
	// unless the block is corrupt and starts with a PKC value.

	pCurElm = NULL;
	uiKeyBufLen = 0;
	while (pStack->uiCurElm < uiTargetCurElm)
	{
		pPrevElm = pCurElm;
		pCurElm = &pBlk[pStack->uiCurElm];
		uiPrevKeyCnt = BBE_GET_PKC( pCurElm);
		uiElmKeyLen = BBE_GET_KL( pCurElm);
		if ((pStack->uiKeyLen = uiPrevKeyCnt + uiElmKeyLen) > pStack->uiKeyBufSize)
		{
			rc = RC_SET( FERR_CACHE_ERROR);
			goto Exit;
		}

		// Copy the minimum number of bytes from the previous element.

		if (uiPrevKeyCnt > uiKeyBufLen)
		{
			FLMUINT		uiCopyLength = uiPrevKeyCnt - uiKeyBufLen;
			FLMBYTE *	pSrcPtr = &pPrevElm[uiElmOvhd];

			flmAssert( pCurElm != NULL);

			while (uiCopyLength--)
			{
				pKeyBuf[uiKeyBufLen++] = *pSrcPtr++;
			}
		}
		else
		{
			uiKeyBufLen = uiPrevKeyCnt;
		}

		// Position to the next element

		if (pStack->uiBlkType == BHT_LEAF)
		{
			pStack->uiCurElm += (FLMUINT) (BBE_LEN( pCurElm));
			if (pStack->uiCurElm + BBE_LEM_LEN >= pStack->uiBlkEnd)
			{
				f_memcpy( &pKeyBuf[uiKeyBufLen], &pCurElm[uiElmOvhd], uiElmKeyLen);

				if (uiSearchKeyLen && (pStack->uiCurElm < pStack->uiBlkEnd))
				{

					// This is a rare and unsure case where caller needs to
					// have pStack->uiPrevElmPKC set correctly.

					FSKeyCmp( pStack, pSearchKey, uiSearchKeyLen, dinDomain);
				}

				goto Hit_End;
			}
		}
		else
		{
			pStack->uiCurElm += (FLMUINT) (BNE_LEN( pStack, pCurElm));
			
			if (pStack->uiCurElm >= pStack->uiBlkEnd)
			{

				// Make sure that pKeyBuf has the last element's key.

				f_memcpy( &pKeyBuf[uiKeyBufLen], &pCurElm[uiElmOvhd], uiElmKeyLen);

Hit_End:

				pStack->uiKeyLen = 0;
				pStack->uiPrevElmPKC = pStack->uiPKC;
				pStack->uiPKC = 0;
				pStack->uiCmpStatus = BT_END_OF_DATA;
				goto Exit;
			}
		}
	}

	// Check to see if the scan hit where you wanted, if so setup stack &
	// pKeyBuf.

	if (pStack->uiCurElm == uiTargetCurElm)
	{

		// BE CAREFUL. Names with "target" point to this element. All other
		// references include pCurElm point to the previous element.

		FLMBYTE *	pTargetCurElm = CURRENT_ELM( pStack);
		FLMUINT		uiTargetPrevKeyCnt = BBE_GET_PKC( pTargetCurElm);
		FLMUINT		uiTargetElmKeyLen = BBE_GET_KL( pTargetCurElm);

		// Compare the current key so that prevPKC and PKC are set.

		pStack->uiCmpStatus = BT_EQ_KEY;
		if (pCurElm)
		{
			if (uiSearchKeyLen)
			{

				// Copy the entire key into keyBuf to compare

				f_memcpy( &pKeyBuf[uiPrevKeyCnt], &pCurElm[uiElmOvhd], uiElmKeyLen);
				pStack->uiCmpStatus = FSKeyCmp( pStack, pSearchKey, uiSearchKeyLen,
														 dinDomain);
			}
			else if (uiTargetPrevKeyCnt > uiKeyBufLen)
			{

				// Copy what is necessary. uiPrevKeyCnt is equal to
				// uiKeyBufLen.

				FLMUINT		uiCopyLength = uiTargetPrevKeyCnt - uiKeyBufLen;
				FLMBYTE *	pSrcPtr = &pCurElm[uiElmOvhd];

				while (uiCopyLength--)
				{
					pKeyBuf[uiKeyBufLen++] = *pSrcPtr++;
				}
			}
		}

		if ((pStack->uiKeyLen = uiTargetPrevKeyCnt + uiTargetElmKeyLen) > 
				pStack->uiKeyBufSize)
		{
			rc = RC_SET( FERR_CACHE_ERROR);
			goto Exit;
		}

		if (uiTargetElmKeyLen)
		{
			f_memcpy( &pKeyBuf[uiTargetPrevKeyCnt], &pTargetCurElm[uiElmOvhd],
						uiTargetElmKeyLen);

			if (uiSearchKeyLen)
			{
				pStack->uiCmpStatus = FSKeyCmp( pStack, pSearchKey, uiSearchKeyLen,
														 dinDomain);
			}
		}
		else
		{

			// This will be hit on a condition where we want to insert "ABCD
			// (10)" into ABCD (15) ABCD (5) between the two keys. (10) is
			// the DIN value. Because the keys are equal we don't have to
			// call compare again. The uiPKC is the uiPrevKeyCnt value.

			pStack->uiPrevElmPKC = pStack->uiPKC;
			pStack->uiPKC = uiTargetPrevKeyCnt;
		}
	}
	else
	{

		// Copy the remaining bytes of the current key into the buffer.

		if (pCurElm)
		{
			f_memcpy( &pKeyBuf[uiPrevKeyCnt], &pCurElm[uiElmOvhd], uiElmKeyLen);
		}

		pStack->uiCmpStatus = BT_GT_KEY;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc: Standard key compare routine for a key and a b-tree element
****************************************************************************/
FSTATIC FLMUINT FSKeyCmp(
	BTSK *			pStack,
	FLMBYTE *		key,
	FLMUINT			uiKeyLen,
	FLMUINT			dinDomain)
{
	FLMBYTE *		pCurElm;
	FLMBYTE *		pKeyBuf;							// Current element's key
	FLMUINT			uiCmp;							// Return value
	FLMUINT			uiCompareLen;					// Length to compare
	FLMUINT			uiOrigCompareLen;				// Original compare length
	FLMUINT			uiCurElmKeyLen;				// Current element's length
	FLMUINT			uiPKCTemp;

	// Get again the current element's key length & compute compare length

	uiCurElmKeyLen = pStack->uiKeyLen;
	uiOrigCompareLen = uiCompareLen = f_min( uiKeyLen, uiCurElmKeyLen);
	pKeyBuf = pStack->pKeyBuf;					// Point to the local key buffer
	pStack->uiPrevElmPKC = pStack->uiPKC;	// Change previous element
	pStack->uiPKC = 0;

	while (uiCompareLen--)
	{
		if (*key++ == *pKeyBuf++)
		{
			continue;
		}

		uiPKCTemp = uiOrigCompareLen - (uiCompareLen + 1);
		pStack->uiPKC = (uiPKCTemp > BBE_PKC_MAX) ? BBE_PKC_MAX : uiPKCTemp;

		// Not equal so return

		return ((*(--key) < *(--pKeyBuf)) ? BT_GT_KEY : BT_LT_KEY);
	}

	// Set the prev key count value

	pStack->uiPKC = (uiOrigCompareLen <= BBE_PKC_MAX) 
									? uiOrigCompareLen 
									: BBE_PKC_MAX;

	// Set return status, If equal then compare the dinDomain if needed.

	uiCmp = uiKeyLen > uiCurElmKeyLen 
					? BT_LT_KEY 
					: (uiKeyLen < uiCurElmKeyLen 
							? BT_GT_KEY 
							: BT_EQ_KEY);

	if ((uiCmp == BT_EQ_KEY) && dinDomain)
	{
		pCurElm = CURRENT_ELM( pStack);
		if ((dinDomain - 1) < FSGetDomain( &pCurElm, (FLMBYTE) pStack->uiElmOvhd))
		{
			uiCmp = BT_LT_KEY;
		}
	}

	return (uiCmp);
}

/****************************************************************************
Desc:	Go to the next element within the block
****************************************************************************/
RCODE FSBlkNextElm(
	BTSK *		pStack)
{
	RCODE			rc = FERR_BT_END_OF_DATA;
	FLMBYTE *	elmPtr;
	FLMUINT		uiElmSize;

	elmPtr = &pStack->pBlk[pStack->uiCurElm];

	if (pStack->uiBlkType == BHT_LEAF)
	{
		uiElmSize = BBE_LEN( elmPtr);
		if (pStack->uiCurElm + BBE_LEM_LEN < pStack->uiBlkEnd)
		{
			if ((pStack->uiCurElm += uiElmSize) + BBE_LEM_LEN < pStack->uiBlkEnd)
			{
				rc = FERR_OK;
			}
		}
	}
	else
	{
		if (pStack->uiBlkType == BHT_NON_LEAF_DATA)
		{
			uiElmSize = BNE_DATA_OVHD;
		}
		else
		{
			uiElmSize = (FLMUINT) BNE_LEN( pStack, elmPtr);
		}

		if (pStack->uiCurElm < pStack->uiBlkEnd)
		{

			// Check if this is not the last element within the block

			if ((pStack->uiCurElm += uiElmSize) < pStack->uiBlkEnd)
			{
				rc = FERR_OK;
			}
		}
	}

	return (rc);
}

/****************************************************************************
Desc: Go to the next element in the logical b-tree while building the key
****************************************************************************/
RCODE FSBtNextElm(
	FDB *		pDb,
	LFILE *	pLFile,
	BTSK *	pStack)
{
	RCODE 	rc = FERR_OK;

	if (pStack->uiCurElm < BH_OVHD)
	{
		pStack->uiCurElm = BH_OVHD;
	}
	else if ((rc = FSBlkNextElm( pStack)) == FERR_BT_END_OF_DATA)
	{
		FLMBYTE *	pBlk = BLK_ELM_ADDR( pStack, BH_NEXT_BLK);
		FLMUINT		blkNum = FB2UD( pBlk);
		
		if (blkNum != BT_END)
		{

			// Current element was last element in the block - go to next
			// block

			if (RC_OK( rc = FSGetBlock( pDb, pLFile, blkNum, pStack)))
			{

				// Set blk end and adjust parent block to next element

				pBlk = pStack->pBlk;
				pStack->uiBlkEnd = (FLMUINT) FB2UW( &pBlk[BH_ELM_END]);
				pStack->uiCurElm = BH_OVHD;
				pStack->uiPKC = 0;
				pStack->uiPrevElmPKC = 0;

				if (pStack->uiFlags & FULL_STACK)
				{
					rc = FSAdjustStack( pDb, pLFile, pStack, TRUE);
				}
			}
		}
	}

	if (RC_OK( rc))
	{
		FLMBYTE *	pCurElm = CURRENT_ELM( pStack);
		FLMUINT		uiKeyLen;

		if (pStack->uiBlkType == BHT_NON_LEAF_DATA)
		{
			flmCopyDrnKey( pStack->pKeyBuf, pCurElm);
			goto Exit;
		}

		// Copy key to the stack->pKeyBuf & check for end key

		if ((uiKeyLen = BBE_GET_KL( pCurElm)) != 0)
		{
			FLMUINT	uiPKC = (FLMUINT) (BBE_GET_PKC( pCurElm));

			if (uiKeyLen + uiPKC <= pStack->uiKeyBufSize)
			{
				pStack->uiKeyLen = (uiKeyLen + uiPKC);
				f_memcpy( &pStack->pKeyBuf[ uiPKC], &pCurElm[ pStack->uiElmOvhd],
							uiKeyLen);
			}
			else
			{
				rc = RC_SET( FERR_CACHE_ERROR);
				goto Exit;
			}
		}
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc: Adjust a full stack
****************************************************************************/
RCODE FSAdjustStack(
	FDB *		pDb,
	LFILE *	pLFile,
	BTSK *	pStack,
	FLMBOOL	bMovedNext)
{
	RCODE 	rc = FERR_OK;

	pStack->uiFlags = FULL_STACK;

	// Pop the stack and go to the next element This is a recursive call
	// back to FSBtNextElm() or FSBtPrevElm() Watch out, this will not work
	// if the concurrency model changes to a b-tree locking method like
	// other products use.
	//
	// Pop the pStack going to the parents block
	
	pStack--;

	// It is very rare that block will need to be read.  Maybe some sort of
	// split case.  The block should have already have been read.

	if (RC_OK( rc = FSGetBlock( pDb, pLFile, pStack->uiBlkAddr, pStack)))
	{
		rc = bMovedNext 
					? FSBtNextElm( pDb, pLFile, pStack)
					: FSBtPrevElm( pDb, pLFile, pStack);
	}

	// Push the pStack and unpin the current block

	pStack++;
	return (rc);
}

/****************************************************************************
Desc: Read in a block from the cache and set most stack elements.
****************************************************************************/
RCODE FSGetBlock(
	FDB *		pDb,
	LFILE *	pLFile,
	FLMUINT	uiBlkAddr,
	BTSK *	pStack)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pBlk;

	// Release whatever block might be there first. If no block is there
	// (pStack->pSCache == NULL), FSReleaseBlock does nothing. Stacks are
	// ALWAYS initialized to set pSCache to NULL, so this is OK to call
	// even if stack has never been used to read a block yet.

	if (pStack->pSCache)
	{

		// If we already have the block we want, keep it!

		if (pStack->pSCache->uiBlkAddress != uiBlkAddr)
		{
			FSReleaseBlock( pStack, FALSE);
		}
	}

	if (!pStack->pSCache)
	{
		flmAssert( !pStack->pBlk);

		if (RC_BAD( rc = ScaGetBlock( pDb, pLFile, BHT_LEAF, uiBlkAddr, NULL,
					  &pStack->pSCache)))
		{
			goto Exit;
		}
	}

	pStack->pBlk = pBlk = pStack->pSCache->pucBlk;
	if (pStack->uiBlkAddr != uiBlkAddr)
	{
		FLMUINT	uiBlkType;

		pStack->uiBlkAddr = uiBlkAddr;

		// Set other pStack elements.

		pStack->uiBlkType = uiBlkType = (FLMUINT) (BH_GET_TYPE( pBlk));

		// The standard overhead is used in the stack Compares are made to
		// determine if the element is extended.

		if (uiBlkType == BHT_LEAF)
		{
			pStack->uiElmOvhd = BBE_KEY;
		}
		else if (uiBlkType == BHT_NON_LEAF_DATA)
		{
			pStack->uiElmOvhd = BNE_DATA_OVHD;
		}
		else if (uiBlkType == BHT_NON_LEAF)
		{
			pStack->uiElmOvhd = BNE_KEY_START;
		}
		else if (uiBlkType == BHT_NON_LEAF_COUNTS)
		{
			pStack->uiElmOvhd = BNE_KEY_COUNTS_START;
		}
		else
		{
			rc = RC_SET( FERR_DATA_ERROR);
			FSReleaseBlock( pStack, FALSE);
			goto Exit;
		}

		pStack->uiKeyLen = pStack->uiPKC = pStack->uiPrevElmPKC = 0;
		pStack->uiLevel = (FLMUINT) pBlk[BH_LEVEL];
		pStack->uiCurElm = BH_OVHD;
	}

	pStack->uiBlkEnd = (FLMUINT) FB2UW( &pBlk[BH_ELM_END]);

Exit:

	return (rc);
}

/****************************************************************************
Desc: Release all of the cache associated with a stack.
****************************************************************************/
void FSReleaseStackCache(
	BTSK *		pStack,
	FLMUINT		uiNumLevels,
	FLMBOOL		bMutexAlreadyLocked)
{
	FLMBOOL		bMutexLocked = FALSE;

	while (uiNumLevels)
	{
		if (pStack->pSCache)
		{
			if (!bMutexLocked && !bMutexAlreadyLocked)
			{
				f_mutexLock( gv_FlmSysData.hShareMutex);
				bMutexLocked = TRUE;
			}

			ScaReleaseCache( pStack->pSCache, TRUE);
			pStack->pSCache = NULL;
			pStack->pBlk = NULL;
		}

		uiNumLevels--;
		pStack++;
	}

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}
}
