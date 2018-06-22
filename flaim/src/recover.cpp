//-------------------------------------------------------------------------
// Desc:	Recover database after a failure.
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

FSTATIC RCODE flmReadLog(
	FDB *					pDb,
	FLMUINT				uiLogEOF,
	FLMUINT *			puiCurrAddrRV,
	FLMBYTE *			pBuf,
	FLMBOOL *			pbIsBeforeImageBlkRV);

FSTATIC RCODE flmProcessBeforeImage(
	FDB *					pDb,
	FLMUINT				uiLogEOF,
	FLMUINT *			puiCurrAddrRV,
	FLMBYTE *			pBuf,
	FLMBOOL				bDoingRecovery,
	FLMUINT				uiMaxTransID);

/****************************************************************************
Desc:	This routine reads the next before-image block from the file.
****************************************************************************/
FSTATIC RCODE flmReadLog(
	FDB *					pDb,
	FLMUINT				uiLogEOF,				// Address of end of rollback log
	FLMUINT *			puiCurrAddrRV,			// This is the current address we are
														// reading in the log file.  It
														// will be updated after reading the
														// data
	FLMBYTE *			pBlk,						// This is the buffer that is to hold
														// the data that is read from the
														// log file
	FLMBOOL *			pbIsBeforeImageBlkRV	// Is block a before-image block?
	)
{
	RCODE			rc = FERR_OK;
	FFILE *		pFile = pDb->pFile;
	FLMUINT		uiFilePos;
	FLMUINT		uiBlkSize = pFile->FileHdr.uiBlockSize;
	FLMUINT		uiBytesRead;
	F_TMSTAMP	StartTime;
	DB_STATS *	pDbStats = pDb->pDbStats;

	uiFilePos = *puiCurrAddrRV;

	// Verify that we are not going to read beyond the log EOF

	if (!FSAddrIsAtOrBelow( uiFilePos + uiBlkSize, uiLogEOF))
	{
		rc = RC_SET( FERR_INCOMPLETE_LOG);
		goto Exit;
	}

	// Position to the appropriate place and read the data

	if (pDbStats)
	{
		pDbStats->bHaveStats = TRUE;
		pDbStats->LogBlockReads.ui64Count++;
		pDbStats->LogBlockReads.ui64TotalBytes += uiBlkSize;
		f_timeGetTimeStamp( &StartTime);
	}

	if( RC_BAD( rc = pDb->pSFileHdl->readBlock( uiFilePos,
			uiBlkSize, pBlk, &uiBytesRead)))
	{
		if (rc == FERR_IO_END_OF_FILE)
		{
			rc = RC_SET( FERR_INCOMPLETE_LOG);
		}

		if (pDbStats)
		{
			pDbStats->uiReadErrors++;
		}
		goto Exit;
	}

	if (pDbStats)
	{
		flmAddElapTime( &StartTime, &pDbStats->LogBlockReads.ui64ElapMilli);
	}

	if (uiBytesRead != uiBlkSize)
	{
		if (pDbStats)
		{
			pDbStats->uiLogBlockChkErrs++;
		}

		rc = RC_SET( FERR_DATA_ERROR);
		goto Exit;
	}

	// Verify the checksum on the block

	if( RC_BAD( rc = BlkCheckSum( pBlk, CHECKSUM_CHECK, BT_END, uiBlkSize)))
	{
		if (pDbStats)
		{
			pDbStats->uiLogBlockChkErrs++;
		}
		goto Exit;
	}

	*pbIsBeforeImageBlkRV = (FLMBOOL)((BH_IS_BI( pBlk))
											  ? (FLMBOOL)TRUE
											  : (FLMBOOL)FALSE);
	BH_UNSET_BI( pBlk);

	// Adjust the current address for the next read

	uiFilePos += uiBlkSize;
	if (FSGetFileOffset( uiFilePos) >= pFile->uiMaxFileSize)
	{
		FLMUINT	uiFileNumber = FSGetFileNumber( uiFilePos);

		if (!uiFileNumber)
		{
			uiFileNumber =
				FIRST_LOG_BLOCK_FILE_NUMBER(
					pFile->FileHdr.uiVersionNum);
		}
		else
		{
			uiFileNumber++;
		}

		if (uiFileNumber > 
			 MAX_LOG_BLOCK_FILE_NUMBER( pFile->FileHdr.uiVersionNum))
		{
			rc = RC_SET( FERR_DB_FULL);
			goto Exit;
		}
		uiFilePos = FSBlkAddress( uiFileNumber, 0 );
	}

	*puiCurrAddrRV = uiFilePos;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine reads and processes a before-image block record
		in the log file.  The reapply flag indicates whether the
		block should be written back to the database file.
****************************************************************************/
FSTATIC RCODE flmProcessBeforeImage(
	FDB *					pDb,
	FLMUINT				uiLogEOF,			// Address of the end of the rollback
													// log
	FLMUINT *			puiCurrAddrRV,		// This is the current offset we are
													// reading in the log file.
													// It will be updated after reading the
													// data
	FLMBYTE *			pBlk,					// This is a pointer to a buffer that
													// will be used to hold the block that
													// is read
	FLMBOOL				bDoingRecovery,	// Are we doing a recovery as opposed to
													// rolling back a transaction?
	FLMUINT				uiMaxTransID)		// Maximum transaction ID to recover to when
													// bDoingRecovery is TRUE.  This parameter
													// is ignored when bDoingRecover is FALSE.
{
	RCODE			rc = FERR_OK;
	FFILE *		pFile = pDb->pFile;
	FLMUINT		uiBlkAddress;
	FLMUINT		uiBlkLength;
	FLMUINT		uiBytesWritten;
	FLMBOOL		bIsBeforeImageBlk = FALSE;
	F_TMSTAMP	StartTime;
	DB_STATS *	pDbStats = pDb->pDbStats;

	// Read the block from the log

	if (RC_BAD( rc = flmReadLog( pDb, uiLogEOF, puiCurrAddrRV, pBlk,
										  &bIsBeforeImageBlk)))
		goto Exit;

	// Determine if we want to restore the block.
	// If we are doing a recovery, restore the block only if
	// its checkpoint is <= uiMaxTransID.  If we are
	// rolling back a transaction, restore the block only if
	// it is marked as a before-image block.

	// For the recovery process, multiple versions
	// of the same block may be restored if there are
	// multiple versions in the log.  However, because
	// the versions will be in ascending order in the
	// file, ultimately, the one with the highest
	// checkpoint that does not exceed uiMaxTransID
	// will be restored - which is precisely the one
	// we want to be restored for a recovery.

	// For a transaction rollback, it is impossible for us
	// to see more than one version of a block that is
	// marked as the before-image version, because we
	// started from a point in the log where the last
	// update transaction logged its first block.  All
	// blocks after that point that have the BI bits
	// set should be restored.  Any that do not have
	// the BI bit set should NOT be restored.

	if (bDoingRecovery)
	{
		if ((FLMUINT)FB2UD( &pBlk [BH_TRANS_ID]) > uiMaxTransID)
		{
			goto Exit;
		}
	}
	else if (!bIsBeforeImageBlk)
	{
		goto Exit;
	}

	// Determine the block address before setting the checksum

	uiBlkAddress = (FLMUINT)GET_BH_ADDR( pBlk);
	uiBlkLength = pFile->FileHdr.uiBlockSize;

	// Set the block checksum AFTER encrypting

	BlkCheckSum( pBlk, CHECKSUM_SET, 
					 uiBlkAddress, pFile->FileHdr.uiBlockSize);


	if (pDbStats)
	{
		pDbStats->bHaveStats = TRUE;
		pDbStats->LogBlockRestores.ui64Count++;
		pDbStats->LogBlockRestores.ui64TotalBytes += uiBlkLength;
		f_timeGetTimeStamp( &StartTime);
	}

	pDb->pSFileHdl->setMaxAutoExtendSize( pFile->uiMaxFileSize);
	pDb->pSFileHdl->setExtendSize( pFile->uiFileExtendSize);
	rc = pDb->pSFileHdl->writeBlock( uiBlkAddress, uiBlkLength, pBlk,
						 &uiBytesWritten);
						 
#ifdef FLM_DBG_LOG
	flmDbgLogWrite( pFile->uiFFileId, uiBlkAddress, 0,
							FB2UD( &pBlk [BH_TRANS_ID]),
								"ROLLBACK");
#endif

	if (pDbStats)
	{
		flmAddElapTime( &StartTime, &pDbStats->LogBlockRestores.ui64ElapMilli);
		if (RC_BAD( rc))
		{
			pDbStats->uiWriteErrors++;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Writes the log header to disk.  The checksum is calculated before
		writing the log header to disk.
*****************************************************************************/
RCODE flmWriteLogHdr(
	DB_STATS *			pDbStats,
	F_SuperFileHdl *	pSFileHdl,
	FFILE *				pFile,
	FLMBYTE *			pucLogHdr,
	FLMBYTE *			pucCPLogHdr,
	FLMBOOL				bIsCheckpoint)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiBytesToWrite;
	FLMUINT			uiNewCheckSum;
	FLMBYTE *		pucTmpLogHdr;
	F_TMSTAMP		StartTime;

	// Force any recent writes to disk before modifying the log file
	// header.  This routine is generally called after having added
	// things to the log file.  It is critical that any previous writes
	// be flushed before the header is updated because the header will
	// generally have been modified to point to the new things that were
	// added.

	if (RC_BAD( rc = pSFileHdl->flush()))
	{
		goto Exit;
	}

	pucTmpLogHdr = &pFile->pucLogHdrIOBuf[ 16];
	uiBytesToWrite = LOG_HEADER_SIZE + 16;

	// Very Important Note:  FlmDbConfig relies on the fact that we will
	// write out the prefix area of the database header.  Do not remove
	// this code.

	flmSetFilePrefix( pFile->pucLogHdrIOBuf, pFile->FileHdr.uiAppMajorVer,
							pFile->FileHdr.uiAppMinorVer);

	// Only copy the part of the header that is relevant for this
	// database version.

	if( pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_3)
	{
		f_memcpy( pucTmpLogHdr, pucLogHdr, LOG_HEADER_SIZE_VER40);
	}
	else
	{
		f_memcpy( pucTmpLogHdr, pucLogHdr, LOG_HEADER_SIZE);
	}

	// If we are not doing a checkpoint, we don't really want
	// to write out certain items, so we restore them from
	// the save info. buffer, which is the buffer that contains
	// the log header data as it was at the time of the
	// checkpoint.

	if (!bIsCheckpoint && pucCPLogHdr)
	{
		f_memcpy( &pucTmpLogHdr [LOG_RFL_LAST_CP_FILE_NUM],
					 &pucCPLogHdr [LOG_RFL_LAST_CP_FILE_NUM], 4);
		f_memcpy( &pucTmpLogHdr [LOG_RFL_LAST_CP_OFFSET],
					 &pucCPLogHdr [LOG_RFL_LAST_CP_OFFSET], 4);
		f_memcpy( &pucTmpLogHdr [LOG_CURR_TRANS_ID],
					 &pucCPLogHdr [LOG_CURR_TRANS_ID], 4);
		f_memcpy( &pucTmpLogHdr [LOG_COMMIT_COUNT],
					 &pucCPLogHdr [LOG_COMMIT_COUNT], 4);
		f_memcpy( &pucTmpLogHdr [LOG_PF_FIRST_BACKCHAIN],
					 &pucCPLogHdr [LOG_PF_FIRST_BACKCHAIN], 4);
		f_memcpy( &pucTmpLogHdr [LOG_PF_AVAIL_BLKS],
					 &pucCPLogHdr [LOG_PF_AVAIL_BLKS], 4);
		f_memcpy( &pucTmpLogHdr [LOG_LOGICAL_EOF],
					 &pucCPLogHdr [LOG_LOGICAL_EOF], 4);
		pucTmpLogHdr [LOG_PF_FIRST_BC_CNT] =
				pucCPLogHdr [LOG_PF_FIRST_BC_CNT];
		f_memcpy( &pucTmpLogHdr [LOG_PF_NUM_AVAIL_BLKS],
					 &pucCPLogHdr [LOG_PF_NUM_AVAIL_BLKS], 4);
		
		if( pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_3)
		{
			f_memcpy( &pucTmpLogHdr [LOG_BLK_CHG_SINCE_BACKUP],
						 &pucCPLogHdr [LOG_BLK_CHG_SINCE_BACKUP], 4);
		}
		
		if( pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_31)
		{
			f_memcpy( &pucTmpLogHdr [LOG_LAST_RFL_COMMIT_ID],
						 &pucCPLogHdr [LOG_LAST_RFL_COMMIT_ID], 4);
		}
	}

	// If this is not a 4.3 database, make sure the old values
	// in the log header slots are preserved.

	if( pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_3)
	{
		// Compatibility for parts that were unused.

		UD2FBA( 0, &pucTmpLogHdr [20]);
		UD2FBA( 0, &pucTmpLogHdr [48]);
		UD2FBA( 0, &pucTmpLogHdr [52]);
		UD2FBA( 0, &pucTmpLogHdr [84]);

		// Compatibility for trans active and maint in progress.

		pucTmpLogHdr [76] = 0;
		pucTmpLogHdr [79] = 0;
	}
	
	uiNewCheckSum = lgHdrCheckSum( pucTmpLogHdr, FALSE);
	UW2FBA( (FLMUINT16)uiNewCheckSum, &pucTmpLogHdr [LOG_HDR_CHECKSUM]);

	// Now update the log header record on disk

	if( pDbStats)
	{
		pDbStats->bHaveStats = TRUE;
		pDbStats->LogHdrWrites.ui64Count++;
		pDbStats->LogHdrWrites.ui64TotalBytes += uiBytesToWrite;
		f_timeGetTimeStamp( &StartTime);
	}
	
	if( RC_BAD( rc = pSFileHdl->writeBlock( 0, 
		(FLMUINT)f_roundUp( uiBytesToWrite, 512),
		pFile->pucLogHdrIOBuf, NULL)))
	{
		if (pDbStats)
		{
			pDbStats->uiWriteErrors++;
		}

		goto Exit;
	}

	if( pDbStats)
	{
		flmAddElapTime( &StartTime, &pDbStats->LogHdrWrites.ui64ElapMilli);
	}

	if( RC_BAD( rc = pSFileHdl->flush()))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine recovers the database to a physically consistent
		state.
Ret:	FERR_OK	-	Indicates the database has been recovered.
		other		-	other FLAIM error codes
****************************************************************************/
RCODE flmPhysRollback(
	FDB *					pDb,
	FLMUINT				uiLogEOF,
	FLMUINT				uiFirstLogBlkAddr,	// Address of first log block
	FLMBOOL				bDoingRecovery,		// Doing recovery?  If so, we will
														// ignore blocks whose transaction
														// ID is higher than uiMaxTransID.
														// Also, we will not check the BI
														// bits in the logged blocks, because
														// we are not rolling back a
														// transaction.
	FLMUINT				uiMaxTransID			// Ignored when bDoingRecovery is
														// FALSE
	)
{
	RCODE			rc = FERR_OK;
	FFILE *		pFile = pDb->pFile;
	FLMUINT		uiCurrAddr;
	FLMBYTE *	pucBlk = NULL;

	// If the log is empty, no need to do anything.
	// A uiFirstLogBlkAddr of zero indicates that there
	// is nothing in the log to rollback.  This will be true
	// if we are rolling back a transaction, and the transaction
	// has not logged anything or if we are doing a recovery and
	// nothing was logged since the last checkpoint.

	if (uiLogEOF == pFile->FileHdr.uiBlockSize ||
		 !uiFirstLogBlkAddr)
	{
		goto Exit;
	}

	// Allocate a buffer to be used for reading.
	
	if( RC_BAD( rc = f_allocAlignedBuffer( pFile->FileHdr.uiBlockSize, 
		(void **)&pucBlk)))
	{
		goto Exit;
	}

	// Start from beginning of log and read to EOF restoring before-image
	// blocks along the way.

	uiCurrAddr = uiFirstLogBlkAddr;
	while (FSAddrIsBelow( uiCurrAddr, uiLogEOF))
	{
		if (RC_BAD( rc = flmProcessBeforeImage( pDb,
									uiLogEOF, &uiCurrAddr, pucBlk, bDoingRecovery,
									uiMaxTransID)))
		{
			goto Exit;
		}
	}

	// Force the writes to the file.

	if (RC_BAD( rc = pDb->pSFileHdl->flush()))
	{
		goto Exit;
	}

Exit:

	// Free the memory handle, if one was allocated.

	if (pucBlk)
	{
		f_freeAlignedBuffer( (void **)&pucBlk);
	}
	
	return( rc);
}
