//------------------------------------------------------------------------------
// Desc:	Routines to handle transaction logging
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

FSTATIC void XFLAPI lgWriteComplete(
	IF_IOBuffer *	pIOBuffer,
	void *			pvData);

#ifdef FLM_DBG_LOG
/****************************************************************************
Desc:	This routine is used to write out information about logged blocks to
		the log file.
****************************************************************************/
void scaLogWrite(
	F_Database *	pDatabase,
	FLMUINT			uiWriteAddress,
	FLMBYTE *		pucBlkBuf,
	FLMUINT			uiBufferLen,
	FLMUINT			uiBlockSize,
	char *			pszEvent
	)
{
	FLMUINT		uiOffset = 0;
	FLMUINT32	ui32BlkAddr;
	FLMUINT64	ui64TransID;

	while (uiOffset < uiBufferLen)
	{
		ui32BlkAddr = ((F_BLK_HDR *)pucBlkBuf)->ui32BlkAddr;
		ui64TransID = ((F_BLK_HDR *)pucBlkBuf)->ui64TransID;

		// A uiWriteAddress of zero means we are writing exactly at the
		// block address - i.e., it is the data block, not the log block.

		flmDbgLogWrite( pDatabase, (FLMUINT)ui32BlkAddr,
							(FLMUINT)((uiWriteAddress)
										 ? uiWriteAddress + uiOffset
										 : (FLMUINT)ui32BlkAddr),
							ui64TransID, pszEvent);
		uiOffset += uiBlockSize;
		pucBlkBuf += uiBlockSize;
	}
}
#endif

/****************************************************************************
Desc:	This is the callback routine that is called when a disk write is
		completed.
****************************************************************************/
FSTATIC void XFLAPI lgWriteComplete(
	IF_IOBuffer *	pIOBuffer,
	void *			pvData)
{
	F_Database *		pDatabase =
								(F_Database *)pIOBuffer->getCallbackData( 0);
#ifdef FLM_DBG_LOG
	FLMUINT				uiBlockSize = pDatabase->getBlockSize();
	FLMUINT				uiLength = pIOBuffer->getBufferSize();
	char *				pszEvent;
#endif
	XFLM_DB_STATS *	pDbStats = (XFLM_DB_STATS *)pvData;

#ifdef FLM_DBG_LOG
	pszEvent = (char *)(RC_OK( pIOBuffer->getCompletionCode())
							  ? (char *)"LGWRT"
							  : (char *)"LGWRT-FAIL");
	scaLogWrite( pDatabase, 0, pIOBuffer->getBuffer(), uiLength,
							 uiBlockSize, pszEvent);
#endif

	if (pDbStats)
	{

		// Must lock mutex, because this may be called from async write
		// completion at any time.

		pDatabase->lockMutex();
		pDbStats->LogBlockWrites.ui64ElapMilli += pIOBuffer->getElapsedTime();
		pDatabase->unlockMutex();
	}
}

/****************************************************************************
Desc:	This routine flushes a log buffer to the log file.
****************************************************************************/
RCODE F_Database::lgFlushLogBuffer(
	XFLM_DB_STATS *	pDbStats,
	F_SuperFileHdl *	pSFileHdl)
{
	RCODE				rc = NE_XFLM_OK;

	if (pDbStats)
	{
		pDbStats->bHaveStats = TRUE;
		pDbStats->LogBlockWrites.ui64Count++;
		pDbStats->LogBlockWrites.ui64TotalBytes += m_uiCurrLogWriteOffset;
	}

	m_pCurrLogBuffer->setCompletionCallback( lgWriteComplete, pDbStats);
	m_pCurrLogBuffer->addCallbackData( (void *)this);
	pSFileHdl->setMaxAutoExtendSize( m_uiMaxFileSize);
	pSFileHdl->setExtendSize( m_uiFileExtendSize);

	// NOTE: No guarantee that m_pCurrLogBuffer will still be around
	// after the call to writeBlock, unless we are doing
	// non-asynchronous write.

	if( RC_BAD( rc = pSFileHdl->writeBlock( m_uiCurrLogBlkAddr,
				m_uiCurrLogWriteOffset, m_pCurrLogBuffer)))
	{
		if (pDbStats)
		{
			pDbStats->uiWriteErrors++;
		}
		
		goto Exit;
	}

Exit:

	m_uiCurrLogWriteOffset = 0;
	m_pCurrLogBuffer->Release();
	m_pCurrLogBuffer = NULL;
	
	return( rc);
}

/****************************************************************************
Desc:	This routine writes a block to the log file.
****************************************************************************/
RCODE F_Database::lgOutputBlock(
	XFLM_DB_STATS	*	pDbStats,
	F_SuperFileHdl *	pSFileHdl,
	F_CachedBlock *	pLogBlock,		// Cached log block.
	F_BLK_HDR *			pBlkHdr,			// Pointer to the corresponding modified
												// block in cache.  This block will be
												// modified to the logged version of
												// the block
	FLMUINT *			puiLogEofRV)	// Returns log EOF
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiFilePos = *puiLogEofRV;
	FLMBYTE *	pucLogBlk;
	F_BLK_HDR *	pLogBlkHdr;
	FLMUINT		uiBlkAddress;
	FLMUINT		uiLogBufferSize;

	// Time for a new block file?

	if (FSGetFileOffset( uiFilePos) >= m_uiMaxFileSize)
	{
		FLMUINT	uiFileNumber;

		// Write out the current buffer, if it has anything in it.

		if (m_uiCurrLogWriteOffset)
		{
			if (RC_BAD( rc = lgFlushLogBuffer( pDbStats, pSFileHdl)))
			{
				goto Exit;
			}
		}

		uiFileNumber = FSGetFileNumber( uiFilePos);

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

		if (RC_BAD( rc = pSFileHdl->createFile( uiFileNumber )))
		{
			goto Exit;
		}
		uiFilePos = FSBlkAddress( uiFileNumber, 0 );
	}

	// Copy the log block to the log buffer.

	if (!m_uiCurrLogWriteOffset)
	{
		m_uiCurrLogBlkAddr = uiFilePos;

		// Get a buffer for logging.
		// NOTE: Buffers are not kept by the F_Database's buffer manager,
		// so once we are done with this buffer, it will be freed.

		uiLogBufferSize = MAX_LOG_BUFFER_SIZE;

		for( ;;)
		{
			if (RC_BAD( rc = m_pBufferMgr->getBuffer(
				uiLogBufferSize, &m_pCurrLogBuffer)))
			{
				// If we failed to get a buffer of the requested size,
				// reduce the buffer size by half and try again.

				if( rc == NE_XFLM_MEM)
				{
					uiLogBufferSize /= 2;
					if( uiLogBufferSize < m_uiBlockSize)
					{
						goto Exit;
					}
					rc = NE_XFLM_OK;
					continue;
				}
				goto Exit;
			}
			break;
		}
	}

	// Copy data from log block to the log buffer

	pucLogBlk = m_pCurrLogBuffer->getBufferPtr() + m_uiCurrLogWriteOffset;
	pLogBlkHdr = (F_BLK_HDR *)pucLogBlk;
	f_memcpy( pLogBlkHdr, pLogBlock->m_pBlkHdr, m_uiBlockSize);

	// If we are logging this block for the current update
	// transaction, set the BEFORE IMAGE (BI) flag in the block header
	// so we will know that this block is a before image block that
	// needs to be restored when aborting the current update
	// transaction.

	if (pLogBlock->m_ui16Flags & CA_WRITE_TO_LOG)
	{
		pLogBlkHdr->ui8BlkFlags |= BLK_IS_BEFORE_IMAGE;
	}
	uiBlkAddress = (FLMUINT)pLogBlkHdr->ui32BlkAddr;

	// Encrypt the block if needed

	if (RC_BAD( rc = encryptBlock( m_pDictList,
											 (FLMBYTE *)pLogBlkHdr)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = flmPrepareBlockToWrite( m_uiBlockSize, pLogBlkHdr)))
	{
		goto Exit;
	}

	// Set up for next log block write

	m_uiCurrLogWriteOffset += m_uiBlockSize;

	// If this log buffer is full, write it out.

	if (m_uiCurrLogWriteOffset == m_pCurrLogBuffer->getBufferSize())
	{
		if (RC_BAD( rc = lgFlushLogBuffer( pDbStats, pSFileHdl)))
		{
			goto Exit;
		}
	}

	// Save the previous block address into the modified block's
	// block header area.  Also save the transaction id.

	pBlkHdr->ui32PriorBlkImgAddr = (FLMUINT32)uiFilePos;

	*puiLogEofRV = uiFilePos + m_uiBlockSize;

Exit:

	return( rc);
}
