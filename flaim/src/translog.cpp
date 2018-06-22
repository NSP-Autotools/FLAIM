//-------------------------------------------------------------------------
// Desc:	Rollback logging.
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

FSTATIC void FLMAPI lgWriteComplete(
	IF_IOBuffer *		pIOBuffer,
	void *				pvData);

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_DBG_LOG
void scaLogWrite(
	FLMUINT		uiFFileId,
	FLMUINT		uiWriteAddress,
	FLMBYTE *	pucBlkBuf,
	FLMUINT		uiBufferLen,
	FLMUINT		uiBlockSize,
	char *		pszEvent)
{
	FLMUINT	uiOffset = 0;
	FLMUINT	uiBlkAddress;

	while (uiOffset < uiBufferLen)
	{
		uiBlkAddress = (FLMUINT)(GET_BH_ADDR( pucBlkBuf));

		// A uiWriteAddress of zero means we are writing exactly at the
		// block address - i.e., it is the data block, not the log block.

		flmDbgLogWrite( uiFFileId, uiBlkAddress,
							(FLMUINT)((uiWriteAddress)
										 ? uiWriteAddress + uiOffset
										 : uiBlkAddress),
			(FLMUINT)(FB2UD( &pucBlkBuf [BH_TRANS_ID])), pszEvent);
		uiOffset += uiBlockSize;
		pucBlkBuf += uiBlockSize;
	}
}
#endif

/****************************************************************************
Desc:	This is the callback routine that is called when a disk write is
		completed.
****************************************************************************/
FSTATIC void FLMAPI lgWriteComplete(
	IF_IOBuffer * 		pIOBuffer,
	void *				pvData)
{
#ifdef FLM_DBG_LOG
	FFILE *		pFile = (FFILE *)pIOBuffer->getCallbackData( 0);
	FLMUINT		uiBlockSize = pFile->FileHdr.uiBlockSize;
	FLMUINT		uiLength = pIOBuffer->getBufferSize();
	char *		pszEvent;
#endif
	DB_STATS *	pDbStats = (DB_STATS *)pvData;

#ifdef FLM_DBG_LOG
	pszEvent = (char *)(RC_OK( pIOBuffer->getCompletionCode())
							  ? (char *)"LGWRT"
							  : (char *)"LGWRT-FAIL");
	scaLogWrite( pFile->uiFFileId, 0, pIOBuffer->getBuffer(), uiLength,
							 uiBlockSize, pszEvent);
#endif

	if (pDbStats)
	{
		// Must lock mutex, because this may be called from async write
		// completion at any time.

		f_mutexLock( gv_FlmSysData.hShareMutex);
		pDbStats->LogBlockWrites.ui64ElapMilli += pIOBuffer->getElapsedTime();
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}
}

/****************************************************************************
Desc:	This routine flushes a log buffer to the log file.
****************************************************************************/
RCODE lgFlushLogBuffer(
	DB_STATS *			pDbStats,
	F_SuperFileHdl *	pSFileHdl,
	FFILE *				pFile)
{
	RCODE					rc = FERR_OK;

	if( pDbStats)
	{
		pDbStats->bHaveStats = TRUE;
		pDbStats->LogBlockWrites.ui64Count++;
		pDbStats->LogBlockWrites.ui64TotalBytes += pFile->uiCurrLogWriteOffset;
	}

	pFile->pCurrLogBuffer->setCompletionCallback( lgWriteComplete, pDbStats);
	pFile->pCurrLogBuffer->addCallbackData( (void *)pFile);
	
	pSFileHdl->setMaxAutoExtendSize( pFile->uiMaxFileSize);
	pSFileHdl->setExtendSize( pFile->uiFileExtendSize);

	if( RC_BAD( rc = pSFileHdl->writeBlock( pFile->uiCurrLogBlkAddr,
				pFile->uiCurrLogWriteOffset, pFile->pCurrLogBuffer)))
	{
		if (pDbStats)
		{
			pDbStats->uiWriteErrors++;
		}
		
		goto Exit;
	}
	
Exit:

	pFile->uiCurrLogWriteOffset = 0;
	pFile->pCurrLogBuffer->Release();
	pFile->pCurrLogBuffer = NULL;

	return( rc);
}

/****************************************************************************
Desc:	This routine writes a block to the log file.
****************************************************************************/
RCODE lgOutputBlock(
	DB_STATS	*			pDbStats,
	F_SuperFileHdl *	pSFileHdl,
	FFILE *				pFile,
	SCACHE *				pLogBlock,
	FLMBYTE *			pucBlk,
	FLMUINT *			puiLogEofRV)
{
	RCODE					rc = FERR_OK;
	FLMUINT				uiFilePos = *puiLogEofRV;
	FLMUINT				uiBlkSize = pFile->FileHdr.uiBlockSize;
	FLMBYTE *			pucLogBlk;
	FLMUINT				uiBlkAddress;
	FLMUINT				uiLogBufferSize;

	// Time for a new block file?
	
	if (FSGetFileOffset( uiFilePos) >= pFile->uiMaxFileSize)
	{
		FLMUINT	uiFileNumber;

		// Write out the current buffer, if it has anything in it.

		if (pFile->uiCurrLogWriteOffset)
		{
			if (RC_BAD( rc = lgFlushLogBuffer( pDbStats, pSFileHdl, pFile)))
			{
				goto Exit;
			}
		}

		uiFileNumber = FSGetFileNumber( uiFilePos);

		if (!uiFileNumber)
		{
			uiFileNumber = FIRST_LOG_BLOCK_FILE_NUMBER(
									pFile->FileHdr.uiVersionNum);
		}
		else
		{
			uiFileNumber++;
		}

		if (uiFileNumber > MAX_LOG_BLOCK_FILE_NUMBER(
									pFile->FileHdr.uiVersionNum))
		{
			rc = RC_SET( FERR_DB_FULL);
			goto Exit;
		}

		if (RC_BAD( rc = pSFileHdl->createFile( uiFileNumber )))
		{
			goto Exit;
		}
		uiFilePos = FSBlkAddress( uiFileNumber, 0 );
	}

	// Copy the log block to the log buffer.

	if (!pFile->uiCurrLogWriteOffset)
	{
		pFile->uiCurrLogBlkAddr = uiFilePos;

		// Get a buffer for logging.
		//
		// NOTE: Buffers are not kept by the FFILE's buffer manager,
		// so once we are done with this buffer, it will be freed

		uiLogBufferSize = MAX_LOG_BUFFER_SIZE;

		for( ;;)
		{
			if (RC_BAD( rc = pFile->pBufferMgr->getBuffer( 
				uiLogBufferSize, &pFile->pCurrLogBuffer)))
			{
				// If we failed to get a buffer of the requested size,
				// reduce the buffer size by half and try again

				if( rc == FERR_MEM)
				{
					uiLogBufferSize /= 2;
					if( uiLogBufferSize < uiBlkSize)
					{
						goto Exit;
					}
					
					rc = FERR_OK;
					continue;
				}
				goto Exit;
			}
			break;
		}
	}

	// Copy data from log block to the log buffer

	pucLogBlk = pFile->pCurrLogBuffer->getBufferPtr() +
						pFile->uiCurrLogWriteOffset;
	f_memcpy( pucLogBlk, pLogBlock->pucBlk, uiBlkSize);

	// If we are logging this block for the current update
	// transaction, set the BEFORE IMAGE (BI) flag in the block header
	// so we will know that this block is a before image block that
	// needs to be restored when aborting the current update
	// transaction

	if (pLogBlock->ui16Flags & CA_WRITE_TO_LOG)
	{
		BH_SET_BI( pucLogBlk);
	}

	// If this is an index block, and it is encrypted, we need to encrypt
	// it before we calculate the checksum
	
	if (BH_GET_TYPE( pucLogBlk) != BHT_FREE && pucLogBlk[ BH_ENCRYPTED])
	{
		FLMUINT		uiBufLen = getEncryptSize( pucLogBlk);

		flmAssert( uiBufLen <= uiBlkSize);

		if (RC_BAD( rc = ScaEncryptBlock( pLogBlock->pFile,
													 pucLogBlk,
													 uiBufLen,
													 uiBlkSize)))
		{
			goto Exit;
		}
	}
	
	// Calculate the block checksum

	uiBlkAddress = GET_BH_ADDR( pucLogBlk);
	BlkCheckSum( pucLogBlk, CHECKSUM_SET, uiBlkAddress, uiBlkSize);

	// Set up for next log block write

	pFile->uiCurrLogWriteOffset += uiBlkSize;

	// If this log buffer is full, write it out

	if( pFile->uiCurrLogWriteOffset == 
		pFile->pCurrLogBuffer->getBufferSize())
	{
		if( RC_BAD( rc = lgFlushLogBuffer( pDbStats, pSFileHdl, pFile)))
		{
			goto Exit;
		}
	}

	// Save the previous block address into the modified block's
	// block header area.  Also save the transaction id

	UD2FBA( (FLMUINT32)uiFilePos, &pucBlk [BH_PREV_BLK_ADDR]);
	f_memcpy( &pucBlk [BH_PREV_TRANS_ID], &pLogBlock->pucBlk [BH_TRANS_ID], 4);

	*puiLogEofRV = uiFilePos + uiBlkSize;

Exit:

	return( rc);
}
