//-------------------------------------------------------------------------
// Desc:	Reduce database size.
// Tabs:	3
//
// Copyright (c) 1992-2007 Novell, Inc. All Rights Reserved.
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

FSTATIC RCODE FLRReadBlkHdr(
	FDB *			pDb,
	FLMUINT		uiBlkAddress,
	FLMBYTE *	pucBlockHeader,
	FLMINT * 	iTypeRV);
	
FSTATIC RCODE FLRMoveBtreeBlk(
	FDB *			pDb,
	FLMUINT		uiBlkAddr,
	FLMUINT		uiLfNumber,
	FLMBOOL *	pbDone);

FSTATIC RCODE FLRMovePcodeLFHBlk(
	FDB *			pDb,
	FLMUINT		uiBlkAddr,
	FLMUINT		uiBlkType);

FSTATIC RCODE FLRFreeAvailBlk(
	FDB *			pDb,
	FLMBYTE *	pucBlkHeader,
	FLMUINT		uiBlkAddr);

FSTATIC RCODE FLRFindPrevAvailBlk(
	FDB *			pDb,
	FLMBYTE *	pucBlkHeader,
	FLMUINT *	puiBlkAddrRV,
	FLMBOOL *	pbFirstChainFlagRV);


/****************************************************************************
Desc : Reduces the size of a FLAIM database file.
Notes: The size of the database file is reduced by freeing a specified
		 number of blocks from the available (unused) block list.  The blocks
		 are moved to the end of the file and the file is truncated.  If the
		 available block list is empty, FLAIM will attemp to add blocks to
		 the list by freeing log extent blocks.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmDbReduceSize(
	HFDB				hDb,
	FLMUINT			uiCount,
	FLMUINT *		puiCountRV)
{
	RCODE				rc = FERR_OK;
	FDB *				pDb = (FDB *) hDb;
	F_Rfl *			pRfl = NULL;
	FLMUINT			uiLogicalEOF;
	FLMUINT			uiBlkAddr;
	FLMUINT			uiNumBlksMoved = 0;
	FLMUINT			uiBlkSize;
	FLMUINT			uiLogicalFileNum;
	FLMBYTE *		pucBlkHeader = NULL;
	FLMINT			iType;
	FLMBOOL			bIgnore;
	FLMBOOL			bLoggingWasOff = FALSE;
	FLMBOOL			bRestoreLoggingOffFlag = FALSE;
	FLMBOOL			bLockedDatabase = FALSE;
	FLMBOOL			bDone = FALSE;

	if( RC_BAD( rc = f_allocAlignedBuffer( MAX_BLOCK_SIZE, &pucBlkHeader)))
	{
		goto Exit;
	}

	// Lock the database if not already locked.
	// Cannot lose exclusive access between the checkpoint and
	// the update transaction that does the truncation.

	if( (pDb->uiFlags & FDB_HAS_FILE_LOCK) == 0)
	{
		if( RC_BAD( rc = FlmDbLock( hDb, FLM_LOCK_EXCLUSIVE, 0, FLM_NO_TIMEOUT)))
		{
			goto Exit;
		}
		bLockedDatabase = TRUE;
	}

	if( IsInCSMode( pDb))
	{
		fdbInitCS( pDb);

		CS_CONTEXT *	pCSSession = pDb->pCSContext;
		FCL_WIRE			Wire( pCSSession, pDb);

		// Send a request to reduce the file

		if (RC_BAD( rc = Wire.sendOp(
			FCS_OPCLASS_DATABASE, FCS_OP_DB_REDUCE_SIZE)))
		{
			goto Exit;
		}

		if (uiCount)
		{
			if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_COUNT, uiCount)))
			{
				goto Transmission_Error;
			}
		}
	
		if (RC_BAD( rc = Wire.sendTerminate()))
		{
			goto Transmission_Error;
		}

		// Read the response
	
		if (RC_BAD( rc = Wire.read()))
		{
			goto Transmission_Error;
		}
		uiNumBlksMoved = (FLMUINT)Wire.getCount();

		rc = Wire.getRCode();
		goto Exit;

Transmission_Error:

		pCSSession->bConnectionGood = FALSE;
		goto Exit;
	}

	if (RC_BAD( rc = fdbInit( pDb, FLM_NO_TRANS,
										FDB_TRANS_GOING_OK, 0, &bIgnore)))
	{
		goto Exit;
	}

	// Make sure we are NOT in a database transaction

	if (pDb->uiTransType != FLM_NO_TRANS)
	{
		rc = RC_SET( FERR_TRANS_ACTIVE);
		goto Exit;
	}

	// Save the state of RFL logging flag.

	pRfl = pDb->pFile->pRfl;
	bLoggingWasOff = pRfl->loggingIsOff();

	// Change state of logging OFF to TRUE - don't want anything
	// logged during reduce except for the reduce packet.

	pRfl->setLoggingOffState( TRUE);
	bRestoreLoggingOffFlag = TRUE;

	// Start a database transaction

	if (RC_BAD(rc = flmBeginDbTrans( pDb, FLM_UPDATE_TRANS, 
		FLM_NO_TIMEOUT, FLM_DONT_POISON_CACHE)))
	{
		goto Exit;
	}

	// Make sure that commit does something.

	pDb->bHadUpdOper = TRUE;
	uiBlkSize = pDb->pFile->FileHdr.uiBlockSize;

	// Get the logical end of file and use internally.
	// Loop until there are not any more free blocks left or the
	// input count is matched.  Switch on each block type found

	uiLogicalEOF = pDb->LogHdr.uiLogicalEOF;

	while( (pDb->LogHdr.uiFirstAvailBlkAddr != BT_END) &&
			 ((!uiCount) || (uiNumBlksMoved < uiCount)))
	{

		// Read the last block and determine block type

		if( FSGetFileOffset( uiLogicalEOF) == 0)
		{
			FLMUINT				uiFileNumber = FSGetFileNumber( uiLogicalEOF) - 1;
			FLMUINT64			ui64FileSize;
			FLMUINT				uiTemp;
			
			if( RC_BAD( rc = pDb->pSFileHdl->getFileSize( uiFileNumber, 
				&ui64FileSize))) 
			{
				goto Reduce_Size_Error;
			}
			
			// Adjust to a block bounds
			
			uiTemp = (FLMUINT)((ui64FileSize / uiBlkSize) * uiBlkSize);
			if( uiTemp < ui64FileSize)
			{
				ui64FileSize = uiTemp + uiBlkSize;
			}
			uiLogicalEOF = FSBlkAddress( uiFileNumber, (FLMUINT)ui64FileSize);
		}
			
		uiBlkAddr = uiLogicalEOF - uiBlkSize;

		if( RC_BAD( rc = FLRReadBlkHdr( pDb, uiBlkAddr, pucBlkHeader, &iType)))
		{
			goto Reduce_Size_Error;
		}

		uiLogicalFileNum = FB2UW( &pucBlkHeader[ BH_LOG_FILE_NUM]);

		switch( iType )
		{
			case	BHT_FREE:
			{
				rc = FLRFreeAvailBlk( pDb, pucBlkHeader, uiBlkAddr);
				break;
			}

			case	BHT_LEAF:
			case	BHT_NON_LEAF:
			case	BHT_NON_LEAF_DATA:
			{
				rc = FLRMoveBtreeBlk( pDb, uiBlkAddr, uiLogicalFileNum, &bDone);
				break;
			}

			case	BHT_LFH_BLK:
			case	BHT_PCODE_BLK:
			{
				rc = FLRMovePcodeLFHBlk( pDb, uiBlkAddr, iType);
				break;
			}
			
			default:
			{
				rc = RC_SET( FERR_BTREE_ERROR);
				break;
			}
		}
		
		if (RC_BAD(rc))
		{
			goto Reduce_Size_Error;
		}
		
		if (bDone)
		{
			break;
		}

		uiNumBlksMoved++;

		// Adjust the logical EOF to the new value.
		// This is complex when dealing with block files.
		
		if( FSGetFileOffset( uiLogicalEOF) == 0)
		{
			FLMUINT				uiFileNumber = FSGetFileNumber( uiLogicalEOF);
			FLMUINT64			ui64FileOffset;

			if( uiFileNumber <= 1)
			{
				break;
			}

			// Leave the current file at zero bytes and move to the 
			// previous store file.
			
			uiFileNumber--;
			
			// Compute the end of the previous block file.
			
			if( RC_BAD( rc = pDb->pSFileHdl->getFileSize( uiFileNumber, 
				&ui64FileOffset)))
			{
				goto Exit;
			}
			
			uiLogicalEOF = FSBlkAddress( uiFileNumber, (FLMUINT)ui64FileOffset);
		}
		
		uiLogicalEOF -= uiBlkSize;
	}

	// Log the reduce packet to the RFL if we are not in the middle of a
	// restore or recovery.  Will need to re-enable logging temporarily
	// and then turn it back off after logging the packet.

	if (!(pDb->uiFlags & FDB_REPLAYING_RFL) &&
		  pDb->pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_3)
	{

		// We would have turned logging OFF above, so we need to
		// turn it back on here.

		pRfl->setLoggingOffState( FALSE);

		// Log the reduce.

		rc = pRfl->logReduce( pDb->LogHdr.uiCurrTransID, uiCount);

		// Turn logging back off.

		pRfl->setLoggingOffState( TRUE);
		if (RC_BAD( rc))
		{
			goto Reduce_Size_Error;
		}
	}

	// Commit the transaction

	{
		FLMBOOL		bFlagSet;
		FLMUINT		uiSaveRflFootprintSize = pDb->pFile->uiRflFootprintSize;
		FLMUINT		uiSaveRblFootprintSize = pDb->pFile->uiRblFootprintSize;
		
		if (pDb->uiFlags & FDB_DO_TRUNCATE)
		{
			bFlagSet = TRUE;
		}
		else
		{
			bFlagSet = FALSE;
			pDb->uiFlags |= FDB_DO_TRUNCATE;
		}
		
		// Set the roll-forward and roll-back log thresholds to
		// their minimum sizes
		
		pDb->pFile->uiRflFootprintSize = 512;
		pDb->pFile->uiRblFootprintSize = pDb->pFile->FileHdr.uiBlockSize;

		rc = flmCommitDbTrans( pDb, uiLogicalEOF, TRUE);
		
		// Restore the RFL and RBL footprint sizes
		
		pDb->pFile->uiRflFootprintSize = uiSaveRflFootprintSize;
		pDb->pFile->uiRblFootprintSize = uiSaveRblFootprintSize;
		
		if (!bFlagSet)
		{
			pDb->uiFlags &= (~(FDB_DO_TRUNCATE));
		}
		
		if (RC_BAD( rc))
		{
			goto Exit;
		}
	}

Exit:

	if( puiCountRV)
	{
		*puiCountRV = uiNumBlksMoved;
	}

	if( bRestoreLoggingOffFlag)
	{
		pRfl->setLoggingOffState( bLoggingWasOff);
	}

	if( bLockedDatabase)
	{
		FlmDbUnlock( hDb);
	}

	if( pucBlkHeader)
	{
		f_freeAlignedBuffer( &pucBlkHeader);
	}
	
	flmExit( FLM_DB_REDUCE_SIZE, pDb, rc);
	return( rc);

Reduce_Size_Error:

	(void)flmAbortDbTrans( pDb, FALSE);
	uiNumBlksMoved = 0;
	goto Exit;
}

/****************************************************************************
Desc:	Read the block header and return the type of block it is
****************************************************************************/
FSTATIC RCODE FLRReadBlkHdr(
	FDB *				pDb,
	FLMUINT			uiBlkAddress,
	FLMBYTE *		pucBlockHeader,
	FLMINT *			piTypeRV)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiBytesRead;
	FLMUINT			uiNumLooks;
	SCACHE *			pBlkSCache;
	DB_STATS *		pDbStats = pDb->pDbStats;
	LFILE_STATS *	pLFileStats;
	F_TMSTAMP		StartTime;
	FLMUINT64		ui64ElapTime;

	// See if first the block is in cache.  Previous writes may not have been
	// forced out to cache.

	if( RC_BAD( rc = ScaGetBlock( pDb, NULL, BHT_LEAF,
		uiBlkAddress, &uiNumLooks, &pBlkSCache)))
	{
		goto Exit;
	}

	if( pBlkSCache)
	{
		f_memcpy( pucBlockHeader, pBlkSCache->pucBlk, BH_OVHD);
		ScaReleaseCache( pBlkSCache, FALSE);
	}
	else
	{
		if( pDbStats)
		{
			ui64ElapTime = 0;
			f_timeGetTimeStamp( &StartTime);
		}
		
		rc = pDb->pSFileHdl->readBlock( uiBlkAddress, 
			pDb->pFile->FileHdr.uiBlockSize, 
			pucBlockHeader, &uiBytesRead);

		if( pDbStats)
		{
			flmAddElapTime( &StartTime, &ui64ElapTime);

			if( RC_BAD( rc))
			{
				pDbStats->bHaveStats = TRUE;
				pDbStats->uiReadErrors++;
			}
			else
			{
				FLMUINT				uiLFileNum;
				BLOCKIO_STATS *	pBlockIOStats;
				FLMUINT				uiBlkType;

				uiLFileNum = FB2UW( &pucBlockHeader [BH_LOG_FILE_NUM]);
				if( !uiLFileNum)
				{
					pLFileStats = NULL;
					uiBlkType = (FLMUINT)BH_GET_TYPE( pucBlockHeader);
				}
				else
				{
					FLMBYTE	ucLfType = 0xFF;
					LFILE *	pTmpLFile;

					if( uiLFileNum == FLM_DICT_INDEX ||
						RC_OK( fdictGetIndex(
							pDb->pDict, pDb->pFile->bInLimitedMode,
							uiLFileNum, NULL, NULL, TRUE)))
					{
						ucLfType = LF_INDEX;
					}
					else if( RC_OK( fdictGetContainer( pDb->pDict, uiLFileNum, &pTmpLFile)))
					{
						ucLfType = LF_CONTAINER;
					}

					if( RC_BAD( flmStatGetLFile( pDbStats,
													uiLFileNum, ucLfType, 0,
													&pLFileStats, NULL, NULL)))
					{
						pLFileStats = NULL;
					}
					if (pLFileStats)
						uiBlkType = BHT_LEAF;
					else
						uiBlkType = (FLMUINT)BH_GET_TYPE( pucBlockHeader);
				}
				if ((pBlockIOStats = flmGetBlockIOStatPtr( pDbStats,
												pLFileStats, pucBlockHeader,
												uiBlkType)) != NULL)
				{
					pDbStats->bHaveStats = TRUE;
					if (pLFileStats)
					{
						pLFileStats->bHaveStats = TRUE;
					}
					pBlockIOStats->BlockReads.ui64ElapMilli += ui64ElapTime;
					pBlockIOStats->BlockReads.ui64Count++;
					pBlockIOStats->BlockReads.ui64TotalBytes += BH_OVHD;
				}
			}
		}

		if( RC_BAD( rc))
		{
			goto Exit;
		}
		
		pucBlockHeader [BH_CHECKSUM_LOW] = (FLMBYTE) uiBlkAddress;
	}

	// If the block address does not agree with what is expected then the
	// block is a log extent.  Otherwise the block contains the type it is.

	if (piTypeRV)
	{
		*piTypeRV = BH_GET_TYPE( pucBlockHeader );
	}
				
Exit:

	return( rc);
}

/****************************************************************************
Desc:		Find where in the b-tree a matching block is located.  Move to
			a free block and change all pointers to the block.
Notes:	Some of this code could be called in movePcodeLFHBlk but we have
			to worry about if the block is a root or right most leaf block.
****************************************************************************/
FSTATIC RCODE FLRMoveBtreeBlk(
	FDB *			pDb,
	FLMUINT		uiBlkAddr,
	FLMUINT		uiLfNumber,
	FLMBOOL *	pbDone)
{
	RCODE			rc;
	FFILE *		pFile = pDb->pFile;
	LFILE *		pLFile;
	SCACHE *		pSCache;
	FLMBOOL		bReleaseCache = FALSE;
	FLMBYTE *	pucBlk;
	FLMBYTE *	pucFreeBlk;
	FLMUINT		uiFreeBlkAddr;
	FLMUINT		uiLeftBlkAddr;
	FLMUINT		uiRightBlkAddr;
	SCACHE *		pFreeSCache;
	FLMBOOL		bReleaseCache2 = FALSE;
	BTSK			StackArea;
	BTSK *		pStack = &StackArea;
	FLMUINT		uiElmOvhd;
	FLMUINT		uiSearchKeyLen;
	FLMBYTE		ucKeyBuf [MAX_KEY_SIZ];
	FLMBYTE		ucSearchKey [MAX_KEY_SIZ];
	FLMUINT		uiTargetLevel;
	FLMUINT		uiLevel;
	FLMUINT		uiRootBlkFlag;
	FLMUINT		uiSavePrevTransID;
	FLMUINT		uiSavePrevBlkAddr;
	FLMUINT		uiBlockType;

	FSInitStackCache( &StackArea, 1);
	pStack->pKeyBuf = ucKeyBuf;
	pStack->uiKeyBufSize = MAX_KEY_SIZ;

	if ((RC_BAD( rc = fdictGetContainer( pDb->pDict, uiLfNumber, &pLFile))) &&
		 (RC_BAD( rc = fdictGetIndex( pDb->pDict, pDb->pFile->bInLimitedMode,
		 		uiLfNumber, &pLFile, NULL, TRUE))))
	{
		// It may be that the index or container is being deleted by a background
		// thread.  In that case, we need to bail out, as there is nothing
		// more we can do until the background thread finishes its delete.

		if (rc == FERR_BAD_IX)
		{
			*pbDone = TRUE;
			rc = FERR_OK;
		}
		goto Exit;
	}

	if (RC_BAD( rc = ScaGetBlock( pDb, pLFile, BHT_LEAF,
									uiBlkAddr, NULL, &pSCache)))
	{
		goto Exit;
	}

	bReleaseCache = TRUE;
	if (RC_BAD( rc = ScaLogPhysBlk( pDb, &pSCache)))
	{
		goto Exit;
	}
	pucBlk = pSCache->pucBlk;

	// Need to make sure that LFILE is up to date.

	if (RC_BAD( rc = flmLFileRead( pDb, pLFile)))
	{
		goto Exit;
	}

	// Get all of the information from the block like linkages and a search key.

	uiLeftBlkAddr  = FB2UD( &pucBlk [BH_PREV_BLK ]);
	uiRightBlkAddr = FB2UD( &pucBlk [BH_NEXT_BLK ]);
	uiTargetLevel = (FLMUINT)(pucBlk [BH_LEVEL ] + 1);
	uiBlockType = BH_GET_TYPE( pucBlk);

	switch( uiBlockType)
	{
	case BHT_LEAF:
		uiElmOvhd = BBE_KEY;
		break;
	case BHT_NON_LEAF:
		uiElmOvhd = BNE_KEY_START;
		break;
	case BHT_NON_LEAF_DATA:
		uiElmOvhd = 0;
		break;
	case BHT_NON_LEAF_COUNTS:
		uiElmOvhd = BNE_KEY_COUNTS_START;
		break;
	default:
		rc = RC_SET( FERR_BTREE_ERROR);
		goto Exit;
	}

	uiSearchKeyLen = (uiBlockType == BHT_NON_LEAF_DATA 
								? 4 
								: BBE_GET_KL( &pucBlk [BH_OVHD ]));

	f_memcpy( ucSearchKey, &pucBlk [BH_OVHD + uiElmOvhd ], uiSearchKeyLen );
	
	// Get the next free block.  Copy block to free block.

	if (RC_BAD( rc = FSBlockUseNextAvail( pDb, pLFile, &pFreeSCache)))
	{
		goto Exit;
	}
	bReleaseCache2 = TRUE;
	pucFreeBlk = pFreeSCache->pucBlk;	
	uiFreeBlkAddr = GET_BH_ADDR( pucFreeBlk);

	// The free block has been logged and set to dirty
	// in FSBlockUseNextAvail().
	// BUT, need to preserve previous transaction ID and previous
	// block address - those should NOT be copied over from the block
	// we are switching with.

	uiSavePrevTransID = (FLMUINT)FB2UD(
									&pFreeSCache->pucBlk [BH_PREV_TRANS_ID]);
	uiSavePrevBlkAddr = (FLMUINT)FB2UD(
									&pFreeSCache->pucBlk [BH_PREV_BLK_ADDR]);

	f_memcpy( pFreeSCache->pucBlk, pucBlk, pFile->FileHdr.uiBlockSize);
	SET_BH_ADDR( pucFreeBlk, (FLMUINT32)uiFreeBlkAddr );

	// Restore the saved previous transaction ID and block address.

	UD2FBA( (FLMUINT32)uiSavePrevTransID,
								&pFreeSCache->pucBlk [BH_PREV_TRANS_ID]);
	UD2FBA( (FLMUINT32)uiSavePrevBlkAddr,
								&pFreeSCache->pucBlk [BH_PREV_BLK_ADDR]);

	ScaReleaseCache( pFreeSCache, FALSE);	// Done with new block
	bReleaseCache2 = FALSE;
	uiRootBlkFlag = (FLMUINT)(BH_IS_ROOT_BLK( pucBlk ));

	// Done with block.

	ScaReleaseCache( pSCache, FALSE);
	bReleaseCache = FALSE;

	// If this is a root block this is easy!  Otherwise you must find the 
	// blocks parent and next/prev blocks and change linkages.

	if( uiRootBlkFlag)
	{
		// Create a new dictionary

		if( !(pDb->uiFlags & FDB_UPDATED_DICTIONARY))
		{
			if( RC_BAD( rc = fdictCloneDict( pDb)))
			{
				goto Exit;
			}

			// Re-get the LFile

			if ((RC_BAD( rc = fdictGetContainer( pDb->pDict, uiLfNumber, &pLFile))) &&
				 (RC_BAD( rc = fdictGetIndex( pDb->pDict, pDb->pFile->bInLimitedMode,
				 		uiLfNumber, &pLFile, NULL, TRUE))))
			{
				goto Exit;
			}
		}

		pLFile->uiRootBlk = uiFreeBlkAddr;
		rc = flmLFileWrite( pDb, pLFile);
		goto Exit;
	}

	// Read left and right blocks and adjust their pointers to point to
	// the new block.  This doesn't matter what level of the b-tree
	// you are on.

	if( uiLeftBlkAddr != BT_END)
	{
		if (RC_BAD( rc = ScaGetBlock( pDb, pLFile, BHT_LEAF,
									uiLeftBlkAddr, NULL, &pSCache)))
		{
			goto Exit;
		}
		
		bReleaseCache = TRUE;
		
		if (RC_BAD( rc = ScaLogPhysBlk( pDb, &pSCache)))
		{
			goto Exit;
		}
		
		pucBlk = pSCache->pucBlk;
		UD2FBA( (FLMUINT32)uiFreeBlkAddr, &pucBlk [BH_NEXT_BLK ]);
		ScaReleaseCache( pSCache, FALSE);
		bReleaseCache = FALSE;
	}

	if( uiRightBlkAddr != BT_END)
	{
		if (RC_BAD( rc = ScaGetBlock( pDb, pLFile, BHT_LEAF,
									uiRightBlkAddr, NULL, &pSCache)))
		{
			goto Exit;
		}
		
		bReleaseCache = TRUE;
		
		if (RC_BAD( rc = ScaLogPhysBlk( pDb, &pSCache)))
		{
			goto Exit;
		}

		pucBlk = pSCache->pucBlk;
		UD2FBA( (FLMUINT32)uiFreeBlkAddr, &pucBlk [BH_PREV_BLK ]);
		ScaReleaseCache( pSCache, FALSE);
		bReleaseCache = FALSE;
	}

	// Now for the hard part!  Build a search key.
	// Scan down the tree one level above blk.
	// Scan right until find element that has child blk
	// that matches uiBlkAddr and adjust child blk addr.

	if (RC_BAD(rc = FSGetBlock( pDb, pLFile,
										  pLFile->uiRootBlk, pStack)))
	{
		goto Exit;
	}

	pStack->uiBlkAddr = pLFile->uiRootBlk;
	for( ;;)
	{
		uiLevel = (FLMUINT)CABLK_ELM( pStack, BH_LEVEL );
		pStack->uiLevel = uiLevel;

		// Scan the block for the matching key

		if( pStack->uiBlkType != BHT_NON_LEAF_DATA)
		{
			rc = FSBtScan( pStack, ucSearchKey, uiSearchKeyLen, 0);
		}
		else
		{
			rc = FSBtScanNonLeafData( pStack, f_bigEndianToUINT32( ucSearchKey));
		}
		if( RC_BAD( rc))
		{
			goto Exit;
		}

		if (uiLevel == uiTargetLevel)
			break;

		if (RC_BAD( rc = FSGetBlock( pDb, pLFile, 
									FSChildBlkAddr( pStack), pStack)))
		{
			goto Exit;
		}
	}

	// The block MUST be a non-leaf block so our job is easier.
	// Scan the elements going right to find the element that
	// has the matching block address.
	// Set NO_STACK flag so get next element doesn't pop stack!

	pStack->uiFlags = NO_STACK;
	
	for( ;;)
	{	
		if (FSChildBlkAddr( pStack) == uiBlkAddr )
		{

			// Found the block
			
			if (RC_BAD( rc = FSLogPhysBlk( pDb, pStack)))
			{
				goto Exit;
			}

			pucBlk = BLK_PTR( pStack );
			FSSetChildBlkAddr( &pucBlk [pStack->uiCurElm],
				uiFreeBlkAddr, (FLMBYTE) pStack->uiElmOvhd);
			break;
		}
		if (RC_BAD( rc = FSBtNextElm( pDb, pLFile, pStack)))
		{
			if (rc == FERR_BT_END_OF_DATA)
			{
				rc = RC_SET( FERR_BTREE_ERROR);
			}
			goto Exit;
		}
	}
	
Exit:

	FSReleaseBlock( &StackArea, FALSE);
	
	if (bReleaseCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}
	
	if (bReleaseCache2)
	{
		ScaReleaseCache( pFreeSCache, FALSE);
	}

	return( rc);
}

/****************************************************************************
Desc:	Find where a pcode list the input block is located.  Move to
		a free block and change all pointers to the block.
****************************************************************************/
FSTATIC RCODE FLRMovePcodeLFHBlk(
	FDB *			pDb,
	FLMUINT		uiBlkAddr,
	FLMUINT		uiBlkType)
{
	RCODE			rc;
	SCACHE *		pSCache;
	FLMBOOL		bReleaseCache = FALSE;
	SCACHE *		pFreeSCache;
	FLMBOOL		bReleaseCache2 = FALSE;
	FLMBYTE *	pucBlk;
	FLMBYTE *	pucFreeBlk;
	FLMUINT		uiLeftBlkAddr;
	FLMUINT		uiRightBlkAddr;
	FLMUINT		uiFreeBlkAddr;
	FFILE *		pFile = pDb->pFile;
	FLMUINT		uiSavePrevTransID;
	FLMUINT		uiSavePrevBlkAddr;

	if (RC_BAD( rc = ScaGetBlock( pDb, NULL, uiBlkType,
										uiBlkAddr, NULL, &pSCache)))
	{
		goto Exit;
	}
	bReleaseCache = TRUE;

	if (RC_BAD( rc = ScaLogPhysBlk( pDb, &pSCache)))
	{
		goto Exit;
	}
	pucBlk = pSCache->pucBlk;

	// Get left and rigth block addresses.
	// Get next avail block and move data over

	uiLeftBlkAddr  = FB2UD( &pucBlk [BH_PREV_BLK ]);
	uiRightBlkAddr = FB2UD( &pucBlk [BH_NEXT_BLK ]);

	if (RC_BAD( rc = FSBlockUseNextAvail( pDb, NULL, &pFreeSCache)))
	{
		goto Exit;
	}
	bReleaseCache2 = TRUE;

	pucFreeBlk = pFreeSCache->pucBlk;	
	uiFreeBlkAddr = GET_BH_ADDR( pucFreeBlk );

	// The free block has been logged and set to dirty in
	// FSBlockUseNextAvail().
	// BUT, need to preserve previous transaction ID and previous
	// block address - those should NOT be copied over from the block
	// we are switching with.

	uiSavePrevTransID = (FLMUINT)FB2UD(
									&pFreeSCache->pucBlk [BH_PREV_TRANS_ID]);
	uiSavePrevBlkAddr = (FLMUINT)FB2UD(
									&pFreeSCache->pucBlk [BH_PREV_BLK_ADDR]);

	f_memcpy( pFreeSCache->pucBlk, pucBlk, pFile->FileHdr.uiBlockSize);
	SET_BH_ADDR( pucFreeBlk, (FLMUINT32)uiFreeBlkAddr);

	// Restore the saved previous transaction ID and block address.

	UD2FBA( (FLMUINT32)uiSavePrevTransID,
								&pFreeSCache->pucBlk [BH_PREV_TRANS_ID]);
	UD2FBA( (FLMUINT32)uiSavePrevBlkAddr,
								&pFreeSCache->pucBlk [BH_PREV_BLK_ADDR]);

	// Fix up any LFile entries that were pointing to the
	// original LFH block

	if( uiBlkType == BHT_LFH_BLK)
	{
		FLMUINT		uiPos = BH_OVHD;
		FLMUINT		uiEndPos = (FLMUINT)FB2UW( &pucFreeBlk[ BH_ELM_END ]);
		LFILE *		pTmpLFile;

		// Create a new dictionary

		if( !(pDb->uiFlags & FDB_UPDATED_DICTIONARY))
		{
			if( RC_BAD( rc = fdictCloneDict( pDb)))
			{
				goto Exit;
			}
		}

		// Iterate over the set of LFiles in the block and
		// update their LFH block addresses

		while( uiPos < uiEndPos)
		{
			if( pucFreeBlk[ uiPos + LFH_TYPE_OFFSET] != LF_INVALID)
			{
				FLMUINT uiTmpLfNum = (FLMUINT)FB2UW( 
												&pucFreeBlk[ uiPos + LFH_LF_NUMBER_OFFSET]);

				if( RC_BAD( fdictGetContainer( pDb->pDict, 
					uiTmpLfNum, &pTmpLFile)))
				{
					if( RC_BAD( rc = fdictGetIndex(
						pDb->pDict, pDb->pFile->bInLimitedMode,
						uiTmpLfNum, &pTmpLFile, NULL, TRUE)))
					{
						flmAssert( 0);
						rc = RC_SET( FERR_DATA_ERROR);
						goto Exit;
					}
				}

				pTmpLFile->uiBlkAddress = uiFreeBlkAddr;
			}
			uiPos += LFH_SIZE;
		}
	}

	// Done with both blocks.

	ScaReleaseCache( pFreeSCache, FALSE);
	ScaReleaseCache( pSCache, FALSE);
	bReleaseCache2 = bReleaseCache = FALSE;

	// Read left and right blocks and adjust their
	// pointers to point to the new block.
	// This doesn't matter what level of the b-tree
	// you are on.

	if (uiLeftBlkAddr != BT_END)
	{
		if (RC_BAD( rc = ScaGetBlock( pDb, NULL, uiBlkType,
									uiLeftBlkAddr, NULL, &pSCache)))
		{
			goto Exit;
		}
		bReleaseCache = TRUE;
		if (RC_BAD( rc = ScaLogPhysBlk( pDb, &pSCache)))
		{
			goto Exit;
		}

		pucBlk = pSCache->pucBlk;
		UD2FBA( (FLMUINT32)uiFreeBlkAddr, &pucBlk [BH_NEXT_BLK ]);
		ScaReleaseCache( pSCache, FALSE);
		bReleaseCache = FALSE;
	}

	if (uiRightBlkAddr != BT_END)
	{
		if (RC_BAD( rc = ScaGetBlock( pDb, NULL, uiBlkType,
									uiRightBlkAddr, NULL, &pSCache)))
		{
			goto Exit;
		}
		bReleaseCache = TRUE;
		if (RC_BAD( rc = ScaLogPhysBlk( pDb, &pSCache)))
		{
			goto Exit;
		}

		pucBlk = pSCache->pucBlk;
		UD2FBA( (FLMUINT32)uiFreeBlkAddr, &pucBlk [BH_PREV_BLK ]);
		ScaReleaseCache( pSCache, FALSE);
		bReleaseCache = FALSE;
	}

Exit:

	if (bReleaseCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}
	if (bReleaseCache2)
	{
		ScaReleaseCache( pFreeSCache, FALSE);
	}

	return( rc);
}


/****************************************************************************
Desc:	Free the input avail block.  Link the block out of the free list
****************************************************************************/
FSTATIC RCODE FLRFreeAvailBlk(
	FDB *			pDb,
	FLMBYTE *	pucBlkHeader,
	FLMUINT		uiBlkAddr)
{
	RCODE			rc = FERR_OK;
	FFILE *		pFile = pDb->pFile;
	FLMBYTE *	pucBlk;
	FLMUINT		uiPrevBlkAddr;
	FLMUINT		uiNextBlkAddr;
	FLMUINT		uiPbcAddr;
	FLMUINT		uiNbcAddr;
	SCACHE *		pSCache;
	FLMBOOL		bReleaseCache = FALSE;
	FLMBOOL		bFirstChainFlag;
	FLMBYTE *	pucLogHdr = &pFile->ucUncommittedLogHdr [0];

	// Check for first avail block condition.

	if (uiBlkAddr == pDb->LogHdr.uiFirstAvailBlkAddr)
	{
		if (RC_OK( rc = FSBlockUseNextAvail( pDb, NULL, &pSCache)))
		{
			ScaReleaseCache( pSCache, FALSE);
		}
		goto Exit;
	}

	// Read the block header and get pointers

	if (RC_BAD( rc = FLRReadBlkHdr( pDb, uiBlkAddr,
									pucBlkHeader, (FLMINT *)0 )))
	{
		goto Exit;
	}
	
	uiPrevBlkAddr = uiBlkAddr;
	uiNextBlkAddr = FB2UD( &pucBlkHeader[ BH_NEXT_BLK]);
	
	if( pFile->FileHdr.uiVersionNum >= 111)
	{
		uiPbcAddr = ALGetPBC( pucBlkHeader);
		uiNbcAddr = ALGetNBC( pucBlkHeader);
		flmDecrUint( &pucLogHdr[ LOG_PF_NUM_AVAIL_BLKS], 1);
	}
	else
	{
		uiPbcAddr = uiNbcAddr = 0;
	}

	if( RC_BAD( rc = FLRFindPrevAvailBlk( pDb, pucBlkHeader,
		&uiPrevBlkAddr, &bFirstChainFlag)))
	{
		goto Exit;
	}

	// Check for unexpected error conditions

	if( (uiPrevBlkAddr == uiBlkAddr)  ||
			((!uiNbcAddr) && ( uiPbcAddr)) || 
			(( uiNbcAddr) && (!uiPbcAddr)))
	{
		rc = RC_SET( FERR_DATA_ERROR);
		goto Exit;
	}

	// NON-CHAIN BLOCK
	// This is also minor verion 0

	if (!uiNbcAddr)
	{
		if (RC_BAD( rc = ScaGetBlock( pDb, NULL, BHT_FREE,
										uiPrevBlkAddr, NULL, &pSCache)))
		{
			goto Exit;
		}
		bReleaseCache = TRUE;

		if (RC_BAD( rc = ScaLogPhysBlk( pDb, &pSCache)))
		{
			goto Exit;
		}
		pucBlk = pSCache->pucBlk;

		if (bFirstChainFlag)
		{
			pucLogHdr [LOG_PF_FIRST_BC_CNT]--;
		}

		UD2FBA( (FLMUINT32)uiNextBlkAddr, &pucBlk [BH_NEXT_BLK]);
		ScaReleaseCache( pSCache, FALSE);
		bReleaseCache = FALSE;
		goto Exit;
	}

	if (uiPrevBlkAddr == uiNbcAddr)
	{
		if ( RC_BAD( rc = ScaGetBlock( pDb, NULL, BHT_FREE,
									uiPrevBlkAddr, NULL, &pSCache)))
		{
			goto Exit;
		}
		bReleaseCache = TRUE;

		if (RC_BAD( rc = ScaLogPhysBlk( pDb, &pSCache)))
		{
			goto Exit;
		}

		pucBlk = pSCache->pucBlk;
		ALPutPBC( pucBlk, uiPbcAddr);
		UD2FBA( (FLMUINT32)uiNextBlkAddr, &pucBlk [BH_NEXT_BLK]);
		ScaReleaseCache( pSCache, FALSE);
		bReleaseCache = FALSE;

		if (uiPbcAddr != BT_END)
		{
			if (RC_BAD( rc = ScaGetBlock( pDb, NULL, BHT_FREE,
										uiPbcAddr, NULL, &pSCache)))
			{
				goto Exit;
			}
			bReleaseCache = TRUE;

			if (RC_BAD( rc = ScaLogPhysBlk( pDb, &pSCache)))
			{
				goto Exit;
			}

			pucBlk = pSCache->pucBlk;
			ALPutNBC( pucBlk, (FLMUINT32)uiNbcAddr);
			ScaReleaseCache( pSCache, FALSE);
			bReleaseCache = FALSE;
		}
	}
	else if ((uiNextBlkAddr == uiPbcAddr) && (uiNextBlkAddr != BT_END))
	{
		if (RC_BAD( rc = ScaGetBlock( pDb, NULL, BHT_FREE,
									uiPrevBlkAddr, NULL, &pSCache)))
		{
			goto Exit;
		}
		bReleaseCache = TRUE;

		if (RC_BAD( rc = ScaLogPhysBlk( pDb, &pSCache)))
		{
			goto Exit;
		}

		pucBlk = pSCache->pucBlk;
		UD2FBA( (FLMUINT32)uiNextBlkAddr, &pucBlk [BH_NEXT_BLK]);
		ScaReleaseCache( pSCache, FALSE);
		bReleaseCache = FALSE;

		if (uiPbcAddr != BT_END)
		{
			if ( RC_BAD( rc = ScaGetBlock( pDb, NULL, BHT_FREE,
											uiNextBlkAddr, NULL, &pSCache)))
			{
				goto Exit;
			}
			bReleaseCache = TRUE;

			if (RC_BAD( rc = ScaLogPhysBlk( pDb, &pSCache)))
			{
				goto Exit;
			}

			pucBlk = pSCache->pucBlk;
			ALPutNBC( pucBlk, (FLMUINT32)uiNbcAddr);
			ScaReleaseCache( pSCache, FALSE);
			bReleaseCache = FALSE;
		}

		if (uiNbcAddr == BT_END)
		{
			UD2FBA( (FLMUINT32)uiPbcAddr, &pucLogHdr [LOG_PF_FIRST_BACKCHAIN]);
		}
		else
		{
			if (RC_BAD( rc = ScaGetBlock( pDb, NULL, BHT_FREE,
									uiNbcAddr, NULL, &pSCache)))
			{
				goto Exit;
			}
			bReleaseCache = TRUE;

			if (RC_BAD( rc = ScaLogPhysBlk( pDb, &pSCache)))
			{
				goto Exit;
			}

			pucBlk = pSCache->pucBlk;
			ALPutPBC( pucBlk, uiPbcAddr);
			ScaReleaseCache( pSCache, FALSE);
			bReleaseCache = FALSE;
		}
	}
	else
	{
		if (RC_BAD( rc = ScaGetBlock( pDb, NULL, BHT_FREE,
									uiPrevBlkAddr, NULL, &pSCache)))
		{
			goto Exit;
		}
		bReleaseCache = TRUE;

		if (RC_BAD( rc = ScaLogPhysBlk( pDb, &pSCache)))
		{
			goto Exit;
		}

		pucBlk = pSCache->pucBlk;
		ALPutNBC( pucBlk, (FLMUINT32)uiNbcAddr);
		ALPutPBC( pucBlk, (FLMUINT32)uiPbcAddr);
		UD2FBA( (FLMUINT32)uiNextBlkAddr, &pucBlk [BH_NEXT_BLK]);
		ScaReleaseCache( pSCache, FALSE);
		bReleaseCache = FALSE;

		if (uiPbcAddr != BT_END)
		{
			if (RC_BAD( rc = ScaGetBlock( pDb, NULL, BHT_FREE,
										uiPbcAddr, NULL, &pSCache)))
			{
				goto Exit;
			}
			bReleaseCache = TRUE;

			if (RC_BAD( rc = ScaLogPhysBlk( pDb, &pSCache)))
			{
				goto Exit;
			}

			pucBlk = pSCache->pucBlk;
			ALPutNBC( pucBlk, (FLMUINT32)uiPrevBlkAddr);
			ScaReleaseCache( pSCache, FALSE);
			bReleaseCache = FALSE;
		}

		if (uiNbcAddr == BT_END)
		{
			UD2FBA( (FLMUINT32)uiPrevBlkAddr,
						&pucLogHdr [LOG_PF_FIRST_BACKCHAIN]);
			if (bFirstChainFlag)
			{
				pucLogHdr [LOG_PF_FIRST_BC_CNT]--;
			}
		}
		else
		{
			if (RC_BAD( rc = ScaGetBlock( pDb, NULL, BHT_FREE,
										uiNbcAddr, NULL, &pSCache)))
			{
				goto Exit;
			}
			bReleaseCache = TRUE;

			if (RC_BAD( rc = ScaLogPhysBlk( pDb, &pSCache)))
			{
				goto Exit;
			}

			pucBlk = pSCache->pucBlk;
			ALPutPBC( pucBlk, uiPrevBlkAddr);
			ScaReleaseCache( pSCache, FALSE);
			bReleaseCache = FALSE;
		}
	}

Exit:

	if (bReleaseCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	return( rc);
}

/****************************************************************************
Desc:	Move an avail block out of the avail block list.
		Worry about version 1.11+ back chaining
****************************************************************************/
FSTATIC RCODE  FLRFindPrevAvailBlk(
	FDB *			pDb,
	FLMBYTE *	pucBlkHeader,
	FLMUINT *	puiBlkAddrRV,
	FLMBOOL *	pbFirstChainFlagRV)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiTargetBlkAddr = *puiBlkAddrRV;
	FLMUINT		uiNextBlkAddr;
	FLMUINT		uiNbcAddr;
	FLMUINT		uiTempBlkAddr = 0;

	*pbFirstChainFlagRV = FALSE;

	if( RC_BAD( rc = FLRReadBlkHdr( pDb, uiTargetBlkAddr,
		pucBlkHeader, (FLMINT *)0)))
	{
		goto Exit;
	}

	uiNextBlkAddr = FB2UD( &pucBlkHeader[ BH_NEXT_BLK]);
	if( uiNextBlkAddr == pDb->LogHdr.uiFirstAvailBlkAddr)
	{
		goto Exit;
	}

	// Find next chain block

	uiNbcAddr = ALGetNBC( pucBlkHeader);
	
	while( !uiNbcAddr && (uiNextBlkAddr != BT_END))
	{
		if( RC_BAD( rc = FLRReadBlkHdr( pDb, uiNextBlkAddr,
												  pucBlkHeader, (FLMINT *)0)))
		{
			goto Exit;
		}

		if( (uiTempBlkAddr = GET_BH_ADDR( pucBlkHeader)) != uiNextBlkAddr)
		{
			rc = RC_SET( FERR_DATA_ERROR);
			goto Exit;
		}
		
		uiNbcAddr = ALGetNBC( pucBlkHeader);
		uiNextBlkAddr = FB2UD( &pucBlkHeader [BH_NEXT_BLK]);
	}

	// Now find the previous avail block

	if( uiNbcAddr == BT_END)
	{
		uiNextBlkAddr = pDb->LogHdr.uiFirstAvailBlkAddr;
		*pbFirstChainFlagRV = TRUE;
	}
	else
	{
		uiNextBlkAddr = uiNbcAddr;
	}

	while( (uiNextBlkAddr != uiTargetBlkAddr) && (uiNextBlkAddr != BT_END))
	{
		if (RC_BAD( rc = FLRReadBlkHdr( pDb, uiNextBlkAddr,
												  pucBlkHeader, (FLMINT *)0 )))
		{
			goto Exit;
		}

		if( (uiTempBlkAddr = GET_BH_ADDR( pucBlkHeader)) != uiNextBlkAddr)
		{
			rc = RC_SET( FERR_DATA_ERROR);
			goto Exit;
		}
		
		uiNextBlkAddr = FB2UD( &pucBlkHeader [BH_NEXT_BLK]);
	}

	*puiBlkAddrRV = uiTempBlkAddr;

Exit:

	return( rc);
}
