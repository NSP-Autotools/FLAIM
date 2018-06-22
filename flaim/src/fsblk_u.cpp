//-------------------------------------------------------------------------
// Desc:	Free blocks, avail list handling
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

/****************************************************************************
Desc: Need to use the current avail block - free up and point to next
****************************************************************************/
RCODE FSBlockUseNextAvail(
	FDB *			pDb,
	LFILE *		pLFile,
	SCACHE **	ppSCacheRV)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiPbcAddr;
	SCACHE *		pSCache;
	FLMBYTE *	pucBlkBuf;
	FLMBOOL		bGotBlock = FALSE;
	FLMBYTE *	pucLogHdr;

	pucLogHdr = &pDb->pFile->ucUncommittedLogHdr[0];

	if (RC_BAD( rc = ScaGetBlock( pDb, NULL, BHT_FREE,
				  pDb->LogHdr.uiFirstAvailBlkAddr, NULL, &pSCache)))
	{
		goto Exit;
	}

	bGotBlock = TRUE;

	// A corruption we have seen a couple of times is where a free block
	// points to itself in the free list.  This will hang the machine so 
	// this check has been added to verify that the block is a free block.

	if (BH_GET_TYPE( pSCache->pucBlk) != BHT_FREE)
	{
		rc = RC_SET( FERR_DATA_ERROR);
		goto Exit;
	}

	// Log the block because we are changing it!

	if (RC_BAD( rc = ScaLogPhysBlk( pDb, &pSCache)))
	{
		goto Exit;
	}

	*ppSCacheRV = pSCache;
	pucBlkBuf = pSCache->pucBlk;

	pDb->LogHdr.uiFirstAvailBlkAddr = (FLMUINT) FB2UD( &pucBlkBuf[BH_NEXT_BLK]);
	UD2FBA( (FLMUINT32)pDb->LogHdr.uiFirstAvailBlkAddr, &pucLogHdr[LOG_PF_AVAIL_BLKS]);
	UD2FBA( 0, &pucBlkBuf[BH_NEXT_BLK]);

	// One less block in the avail list.

	flmDecrUint( &pucLogHdr[LOG_PF_NUM_AVAIL_BLKS], 1);

	// Decrement so chains are consistent

	pucLogHdr[LOG_PF_FIRST_BC_CNT]--;

	if (ALGetNBC( pucBlkBuf) == BT_END)
	{

		// This is a chain block - so take care of the back chains

		uiPbcAddr = ALGetPBC( pucBlkBuf);
		ALResetAvailBlk( pucBlkBuf);

		if (uiPbcAddr == BT_END)
		{
			UD2FBA( (FLMUINT32)BT_END, &pucLogHdr[LOG_PF_FIRST_BACKCHAIN]);
			pucLogHdr[LOG_PF_FIRST_BC_CNT] = 0;
		}
		else
		{
			SCACHE *		pPbcSCache;

			// Hit a backchain block Setup backchain links and adjust
			// LOG_PF_FIRST_BC_CNT to BACKCHAIN_CNT This is not perfect
			// because there may be less blocks in that chain than expected.
			//
			// Ensure next block is chained.
			
			pucLogHdr[LOG_PF_FIRST_BC_CNT] = BACKCHAIN_CNT;
			UD2FBA( (FLMUINT32) uiPbcAddr, &pucLogHdr[LOG_PF_FIRST_BACKCHAIN]);

			// Read the previous backchain and modify its nextBackchain
			// pointer.
			//
			// Read the old first backchain block and change the NBC.
			
			if (RC_BAD( rc = ScaGetBlock( pDb, NULL, BHT_FREE, uiPbcAddr, NULL,
						  &pPbcSCache)))
			{
				goto Exit;
			}

			// Log the block because we are changing it!

			if (RC_OK( rc = ScaLogPhysBlk( pDb, &pPbcSCache)))
			{
				ALPutNBC( pPbcSCache->pucBlk, (FLMUINT32)BT_END);
			}

			ScaReleaseCache( pPbcSCache, FALSE);
			if (RC_BAD( rc))
			{
				goto Exit;
			}
		}
	}

	// If this is an index block, check to see if it is encrypted.

	if (pLFile && pLFile->pIxd && pLFile->pIxd->uiEncId)
	{
		pucBlkBuf[BH_ENCRYPTED] = 1;
	}

Exit:

	if ((RC_BAD( rc)) && (bGotBlock))
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	return (rc);
}

/****************************************************************************
Desc:	This routine puts a block back in a physical file's avail list.
****************************************************************************/
RCODE FSBlockFree(
	FDB *			pDb,
	SCACHE *		pSCache)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiFirstAvailAddress;
	FLMBYTE *	pucBlkBuf;
	FLMUINT		uiBlkAddress;
	SCACHE *		pPbcSCache;
	FLMUINT		uiPbcAddr;
	FLMBYTE *	pucLogHdr;

	pucLogHdr = &pDb->pFile->ucUncommittedLogHdr[0];

	// Log the block before modifying it.

	if (RC_BAD( rc = ScaLogPhysBlk( pDb, &pSCache)))
	{
		goto Exit;
	}

	pucBlkBuf = pSCache->pucBlk;

	// Set all elements except block address and checkpoint info to zeros.
	// If you add any new block elements please make sure they are taken
	// care of here.
	//
	// Leave block address alone.
	
	uiBlkAddress = GET_BH_ADDR( pucBlkBuf);

	ALResetAvailBlk( pucBlkBuf);

	uiFirstAvailAddress = pDb->LogHdr.uiFirstAvailBlkAddr;
	UD2FBA( (FLMUINT32) uiFirstAvailAddress, &pucBlkBuf[BH_NEXT_BLK]);
	pucBlkBuf[BH_TYPE] = BHT_FREE;
	pucBlkBuf[BH_LEVEL] = 0;
	UW2FBA( (FLMUINT16)BH_OVHD, &pucBlkBuf[BH_ELM_END]);

	// Wipe the contents of encrypted blocks...

	if (pucBlkBuf[BH_ENCRYPTED])
	{
		f_memset( &pucBlkBuf[BH_OVHD], 0, 
				pDb->pFile->FileHdr.uiBlockSize - BH_OVHD);
		pucBlkBuf[BH_ENCRYPTED] = 0;
	}

	// Leave CHECKPOINT, PREV_CP and PREV_BLK_ADDR alone.
	// Update the physical file log information.
	
	pDb->LogHdr.uiFirstAvailBlkAddr = uiBlkAddress;
	UD2FBA( (FLMUINT32) uiBlkAddress, &pucLogHdr[LOG_PF_AVAIL_BLKS]);

	// Is it time to add a new backchain?

	if (pucLogHdr[LOG_PF_FIRST_BC_CNT] >= BACKCHAIN_CNT ||
		 FB2UD( &pucLogHdr[LOG_PF_NUM_AVAIL_BLKS]) == 0)
	{

		// Start over - increments to 1 below.

		pucLogHdr[LOG_PF_FIRST_BC_CNT] = 0;
		ALPutNBC( pucBlkBuf, (FLMUINT32)BT_END);

		// Increment and check if no avail blocks

		if (FB2UD( &pucLogHdr[LOG_PF_NUM_AVAIL_BLKS]) == 0)
		{
			ALPutPBC( pucBlkBuf, BT_END);
		}
		else
		{
			uiPbcAddr = (FLMUINT) FB2UD( &pucLogHdr[LOG_PF_FIRST_BACKCHAIN]);
			ALPutPBC( pucBlkBuf, uiPbcAddr);

			// Read the old first backchain block and change the NBC.

			if (RC_BAD( rc = ScaGetBlock( pDb, NULL, BHT_FREE, uiPbcAddr, NULL,
						  &pPbcSCache)))
			{
				goto Exit;
			}

			// Log the block because we are changing it!

			if (RC_OK( rc = ScaLogPhysBlk( pDb, &pPbcSCache)))
			{
				ALPutNBC( pPbcSCache->pucBlk, (FLMUINT32)uiBlkAddress);
			}

			ScaReleaseCache( pPbcSCache, FALSE);
			if (RC_BAD( rc))
			{
				goto Exit;
			}
		}

		UD2FBA( (FLMUINT32) uiBlkAddress, &pucLogHdr[LOG_PF_FIRST_BACKCHAIN]);
	}

	// Be sure to increment these.

	flmIncrUint( &pucLogHdr[LOG_PF_NUM_AVAIL_BLKS], 1);
	pucLogHdr[LOG_PF_FIRST_BC_CNT]++;

Exit:

	ScaReleaseCache( pSCache, FALSE);
	return (rc);
}

/****************************************************************************
Desc: Fix up the previous/next links
****************************************************************************/
RCODE FSBlockFixLinks(
	FDB *			pDb,
	LFILE *		pLFile,
	SCACHE *		pSCache)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiPrevBlkAddr;
	FLMUINT		uiNextBlkAddr;
	FLMBYTE *	pucBlkBuf;

	pucBlkBuf = pSCache->pucBlk;
	uiPrevBlkAddr = (FLMUINT) FB2UD( &pucBlkBuf[BH_PREV_BLK]);
	uiNextBlkAddr = (FLMUINT) FB2UD( &pucBlkBuf[BH_NEXT_BLK]);

	// First free block. NOTE: Do NOT access pSCache after this call

	if (RC_BAD( rc = FSBlockFree( pDb, pSCache)))
	{
		goto Exit;
	}

	// Read the previous block if current is not the left end.

	if (uiPrevBlkAddr != BT_END)
	{
		if (RC_BAD( rc = ScaGetBlock( pDb, pLFile, BHT_LEAF, uiPrevBlkAddr, NULL,
					  &pSCache)))
		{
			goto Exit;
		}

		// Log the block before modifying it.

		if (RC_OK( rc = ScaLogPhysBlk( pDb, &pSCache)))
		{
			UD2FBA( (FLMUINT32)uiNextBlkAddr, &pSCache->pucBlk[BH_NEXT_BLK]);
		}

		ScaReleaseCache( pSCache, FALSE);
		if (RC_BAD( rc))
		{
			goto Exit;
		}
	}

	// Read the next block if current is not the left end.

	if (uiNextBlkAddr != BT_END)
	{
		if (RC_BAD( rc = ScaGetBlock( pDb, pLFile, BHT_LEAF, uiNextBlkAddr, 
			NULL, &pSCache)))
		{
			goto Exit;
		}

		// Log the block before modifying it.

		if (RC_OK( rc = ScaLogPhysBlk( pDb, &pSCache)))
		{
			UD2FBA( (FLMUINT32)uiPrevBlkAddr, &pSCache->pucBlk[BH_PREV_BLK]);
		}

		ScaReleaseCache( pSCache, FALSE);
		if (RC_BAD( rc))
		{
			goto Exit;
		}
	}

Exit:

	return (rc);
}
