//------------------------------------------------------------------------------
// Desc:	Contains routines for recovering a database after
//			a failure.
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
Desc:	This routine reads the next before-image block from the database.
****************************************************************************/
RCODE F_Db::readRollbackLog(
	FLMUINT		uiLogEOF,			// Address of end of rollback log.
	FLMUINT *	puiCurrAddr,		// This is the current address we are
											//	reading in the log file.  It
											//	will be updated after reading the
											//	data.
	F_BLK_HDR *	pBlkHdr,				// This is the buffer that is to hold
											//	the data that is read from the
											//	log file.
	FLMBOOL *	pbIsBeforeImageBlk// Is block a before-image block?
	)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiFilePos;
	FLMUINT		uiBlkSize = m_pDatabase->m_uiBlockSize;
	FLMUINT		uiBytesRead;
	F_TMSTAMP	StartTime;

	uiFilePos = *puiCurrAddr;

	// Verify that we are not going to read beyond the log EOF

	if (!FSAddrIsAtOrBelow( uiFilePos + uiBlkSize, uiLogEOF))
	{
		rc = RC_SET( NE_XFLM_INCOMPLETE_LOG);
		goto Exit;
	}

	// Position to the appropriate place and read the data

	if (m_pDbStats)
	{
		m_pDbStats->bHaveStats = TRUE;
		m_pDbStats->LogBlockReads.ui64Count++;
		m_pDbStats->LogBlockReads.ui64TotalBytes += uiBlkSize;
		f_timeGetTimeStamp( &StartTime);
	}

	if (RC_BAD( rc = m_pSFileHdl->readBlock( uiFilePos,
			uiBlkSize, (FLMBYTE *)pBlkHdr, &uiBytesRead)))
	{
		if (rc == NE_FLM_IO_END_OF_FILE)
		{
			rc = RC_SET( NE_XFLM_INCOMPLETE_LOG);
		}

		if (m_pDbStats)
		{
			m_pDbStats->uiReadErrors++;
		}
		goto Exit;
	}

	if (m_pDbStats)
	{
		flmAddElapTime( &StartTime, &m_pDbStats->LogBlockReads.ui64ElapMilli);
	}

	if (uiBytesRead != uiBlkSize)
	{
		if (m_pDbStats)
		{
			m_pDbStats->uiLogBlockChkErrs++;
		}

		rc = RC_SET( NE_XFLM_DATA_ERROR);
		goto Exit;
	}

	if (RC_BAD( rc = flmPrepareBlockForUse( uiBlkSize, pBlkHdr)))
	{
		if (m_pDbStats && rc == NE_XFLM_BLOCK_CRC)
		{
			m_pDbStats->uiLogBlockChkErrs++;
		}
		goto Exit;
	}

	// See if before image bit is set.  Need to unset it if it is.

	*pbIsBeforeImageBlk = (FLMBOOL)((pBlkHdr->ui8BlkFlags &
													BLK_IS_BEFORE_IMAGE)
												 ? (FLMBOOL)TRUE
												 : (FLMBOOL)FALSE);
	pBlkHdr->ui8BlkFlags &= ~(BLK_IS_BEFORE_IMAGE);

	// Adjust the current address for the next read

	uiFilePos += uiBlkSize;
	if (FSGetFileOffset( uiFilePos) >= m_pDatabase->m_uiMaxFileSize)
	{
		FLMUINT	uiFileNumber = FSGetFileNumber( uiFilePos);

		if (!uiFileNumber)
		{
			uiFileNumber = FIRST_LOG_BLOCK_FILE_NUMBER;
		}
		else
		{
			uiFileNumber++;
		}

		if (uiFileNumber > MAX_LOG_BLOCK_FILE_NUMBER)
		{
			rc = RC_SET( NE_XFLM_DB_FULL);
			goto Exit;
		}
		uiFilePos = FSBlkAddress( uiFileNumber, 0 );
	}

	*puiCurrAddr = uiFilePos;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine reads and processes a before-image block record
		in the log file.  The reapply flag indicates whether the
		block should be written back to the database file.
****************************************************************************/
RCODE F_Db::processBeforeImage(
	FLMUINT		uiLogEOF,			// Address of the end of the rollback
											//	log.
	FLMUINT *	puiCurrAddrRV,		// This is the current offset we are
											//	reading in the log file.
											//	It will be updated after reading the
											//	data.
	F_BLK_HDR *	pBlkHdr,				// This is a pointer to a buffer that
											//	will be used to hold the block that
											//	is read.
	FLMBOOL		bDoingRecovery,	// Are we doing a recovery as opposed to
											// rolling back a transaction?
	FLMUINT64	ui64MaxTransID		// Maximum transaction ID to recover to when
											// bDoingRecovery is TRUE.  This parameter
											// is ignored when bDoingRecover is FALSE.

	)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiBlkAddress;
	FLMUINT		uiBlkLength;
#ifdef FLM_DBG_LOG
	FLMUINT64	ui64TransID;
#endif
	FLMUINT		uiBytesWritten;
	FLMBOOL		bIsBeforeImageBlk;
	F_TMSTAMP	StartTime;

	// Read the block from the log

	if (RC_BAD( rc = readRollbackLog( uiLogEOF, puiCurrAddrRV, pBlkHdr,
										  &bIsBeforeImageBlk)))
	{
		goto Exit;
	}

	// Determine if we want to restore the block.
	// If we are doing a recovery, restore the block only if
	// its checkpoint is <= ui64MaxTransID.  If we are
	// rolling back a transaction, restore the block only if
	// it is marked as a before-image block.

	// For the recovery process, multiple versions
	// of the same block may be restored if there are
	// multiple versions in the log.  However, because
	// the versions will be in ascending order in the
	// file, ultimately, the one with the highest
	// checkpoint that does not exceed ui64MaxTransID
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
		if (pBlkHdr->ui64TransID > ui64MaxTransID)
		{
			goto Exit;
		}
	}
	else if (!bIsBeforeImageBlk)
	{
		goto Exit;
	}

	// Determine the block address before setting the checksum.

	uiBlkAddress = (FLMUINT)pBlkHdr->ui32BlkAddr;
	uiBlkLength = blkGetEnd( m_pDatabase->m_uiBlockSize, blkHdrSize( pBlkHdr),
								pBlkHdr);
#ifdef FLM_DBG_LOG
	ui64TransID = pBlkHdr->ui64TransID;
#endif

	if (RC_BAD( rc = flmPrepareBlockToWrite( m_pDatabase->m_uiBlockSize,
									pBlkHdr)))
	{
		goto Exit;
	}

	if (m_pDbStats)
	{
		m_pDbStats->bHaveStats = TRUE;
		m_pDbStats->LogBlockRestores.ui64Count++;
		m_pDbStats->LogBlockRestores.ui64TotalBytes += uiBlkLength;
		f_timeGetTimeStamp( &StartTime);
	}

	m_pSFileHdl->setMaxAutoExtendSize( m_pDatabase->m_uiMaxFileSize);
	m_pSFileHdl->setExtendSize( m_pDatabase->m_uiFileExtendSize);
	rc = m_pSFileHdl->writeBlock( uiBlkAddress, uiBlkLength, pBlkHdr,
						 &uiBytesWritten);
#ifdef FLM_DBG_LOG
	flmDbgLogWrite( m_pDatabase, uiBlkAddress, 0, ui64TransID,
								"ROLLBACK");
#endif

	if (m_pDbStats)
	{
		flmAddElapTime( &StartTime, &m_pDbStats->LogBlockRestores.ui64ElapMilli);
		if (RC_BAD( rc))
		{
			m_pDbStats->uiWriteErrors++;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Writes the log header to disk.  The checksum is calculated before
		writing the log header to disk.
*****************************************************************************/
RCODE F_Database::writeDbHdr(
	XFLM_DB_STATS *	pDbStats,
	F_SuperFileHdl *	pSFileHdl,
	XFLM_DB_HDR *		pDbHdr,				// DB header to be written out.
	XFLM_DB_HDR *		pCPDbHdr,			// DB header as it was at the time
													// of the checkpoint.
	FLMBOOL				bIsCheckpoint		// Are we writing a checkpoint?  If we
													// we are, we may write the DB header
													// as is.  Otherwise, we need to make
													// sure we don't write out certain
													// parts of the DB header - they must
													// not be updated on disk until a
													// checkpoint actually occurs.
	)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiBytesWritten;
	XFLM_DB_HDR *	pTmpDbHdr;
	F_TMSTAMP		StartTime;

	// Force any recent writes to disk before modifying the DB
	// header.  This routine is generally called after having
	// written out data blocks or rollback blocks.  It is
	// critial that any previous writes be flushed before the
	// header is updated because the header will generally have
	// been modified to point to the new things that were added.

	if (RC_BAD( rc = pSFileHdl->flush()))
	{
		goto Exit;
	}

	// No need to ever actually write the header to disk if this is
	// a temporary database

	if (m_bTempDb)
	{
		goto Exit;
	}

	pTmpDbHdr = m_pDbHdrWriteBuf;

	uiBytesWritten = sizeof( XFLM_DB_HDR);
	f_memcpy( pTmpDbHdr, pDbHdr, sizeof( XFLM_DB_HDR));

	// If we are not doing a checkpoint, we don't really want
	// to write out certain items, so we restore them from
	// the database header as it was at the time of the last
	// checkpoint.

	if (!bIsCheckpoint && pCPDbHdr)
	{
		pTmpDbHdr->ui32RflLastCPFileNum = pCPDbHdr->ui32RflLastCPFileNum;
		pTmpDbHdr->ui32RflLastCPOffset = pCPDbHdr->ui32RflLastCPOffset;
		pTmpDbHdr->ui64CurrTransID = pCPDbHdr->ui64CurrTransID;
		pTmpDbHdr->ui64TransCommitCnt = pCPDbHdr->ui64TransCommitCnt;
		pTmpDbHdr->ui32FirstAvailBlkAddr = pCPDbHdr->ui32FirstAvailBlkAddr;
		pTmpDbHdr->ui32LogicalEOF = pCPDbHdr->ui32LogicalEOF;
		pTmpDbHdr->ui32BlksChangedSinceBackup =
			pCPDbHdr->ui32BlksChangedSinceBackup;
		pTmpDbHdr->ui64LastRflCommitID = pCPDbHdr->ui64LastRflCommitID;
	}

	// Header is always written out in native format.  Set the CRC

	flmAssert( !hdrIsNonNativeFormat( pTmpDbHdr));

	pTmpDbHdr->ui32HdrCRC = calcDbHdrCRC( pTmpDbHdr);

	// Now update the log header record on disk.

	if (pDbStats)
	{
		pDbStats->bHaveStats = TRUE;
		pDbStats->DbHdrWrites.ui64Count++;
		pDbStats->DbHdrWrites.ui64TotalBytes +=
				uiBytesWritten;
		f_timeGetTimeStamp( &StartTime);
	}

	if( RC_BAD( rc = pSFileHdl->writeBlock( 0,
		uiBytesWritten, pTmpDbHdr, &uiBytesWritten)))
	{
		if (pDbStats)
		{
			pDbStats->uiWriteErrors++;
		}

		goto Exit;
	}

	if (pDbStats)
	{
		flmAddElapTime( &StartTime, &pDbStats->DbHdrWrites.ui64ElapMilli);
	}

	// Finally, force the header to disk.

	if (RC_BAD( rc = pSFileHdl->flush()))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine recovers the database to a physically consistent
		state.
Ret:	NE_XFLM_OK	-	Indicates the database has been recovered.
		other		-	other FLAIM error codes
****************************************************************************/
RCODE F_Db::physRollback(
	FLMUINT		uiLogEOF,
	FLMUINT		uiFirstLogBlkAddr,	// Address of first log block
	FLMBOOL		bDoingRecovery,		// Doing recovery?  If so, we will
												// ignore blocks whose transaction
												// ID is higher than ui64MaxTransID.
												// Also, we will not check the BI
												// bits in the logged blocks, because
												// we are not rolling back a
												// transaction.
	FLMUINT64	ui64MaxTransID			// Ignored when bDoingRecovery is
												// FALSE
	)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiCurrAddr;
	FLMBYTE *	pucBlk = NULL;

	// If the log is empty, no need to do anything.
	// A uiFirstLogBlkAddr of zero indicates that there
	// is nothing in the log to rollback.  This will be true
	// if we are rolling back a transaction, and the transaction
	// has not logged anything or if we are doing a recovery and
	// nothing was logged since the last checkpoint.

	if (uiLogEOF == m_pDatabase->m_uiBlockSize || !uiFirstLogBlkAddr)
	{
		goto Exit;		// Will return NE_XFLM_OK
	}

	// Allocate a buffer to be used for reading.
	
	if( RC_BAD( rc = f_allocAlignedBuffer( m_pDatabase->m_uiBlockSize,
		(void **)&pucBlk)))
	{
		goto Exit;
	}

	// Start from beginning of log and read to EOF restoring before-image
	// blocks along the way.

	uiCurrAddr = uiFirstLogBlkAddr;
	while (FSAddrIsBelow( uiCurrAddr, uiLogEOF))
	{
		if (RC_BAD( rc = processBeforeImage( uiLogEOF, &uiCurrAddr,
								(F_BLK_HDR *)pucBlk, bDoingRecovery,
								ui64MaxTransID)))
		{
			goto Exit;
		}
	}

	// Force the writes to the file.

	if (RC_BAD( rc = m_pSFileHdl->flush()))
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
