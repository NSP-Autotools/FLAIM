//------------------------------------------------------------------------------
// Desc:	Various macros, prototypes, structures.
// Tabs:	3
//
// Copyright (c) 1990-1993, 1995-2007 Novell, Inc. All Rights Reserved.
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

#ifndef FILESYS_H
#define FILESYS_H

/***************************************************************
**
**		Defined Constants that the File system cares about
**
****************************************************************/

#define	MAX_DATA_BLOCK_FILE_NUMBER		0x7FF
#define	FIRST_LOG_BLOCK_FILE_NUMBER	(MAX_DATA_BLOCK_FILE_NUMBER + 1)
#define	MAX_LOG_BLOCK_FILE_NUMBER		0xFFF

#define	FSGetFileNumber( uiBlkAddr)	 ((uiBlkAddr) & MAX_LOG_BLOCK_FILE_NUMBER)
#define	FSGetFileOffset( udBlkAddr)	 ((udBlkAddr) & 0xFFFFF000)
#define	FSBlkAddress( iFileNum, udFileOfs) ((udFileOfs) + (iFileNum))

// Max file size and log threshold.

#define	LOG_THRESHOLD_SIZE			((FLMUINT) 0x40000)

// very large threshhold is the size we will allow the physical
// log to grow to before we force a truncation.  At the low end,
// it is about 10 megabytes.  At the high end it is about
// 1 gigabyte.

#define	LOW_VERY_LARGE_LOG_THRESHOLD_SIZE	((FLMUINT)0xA00000)
#define	HIGH_VERY_LARGE_LOG_THRESHOLD_SIZE ((FLMUINT) 0x40000000)

// RFL_TRUNCATE_SIZE is the size we will let an RFL file grow to
// before we truncate it back.  RFL files are only truncated if
// we are configured to delete old RFL files.

#define	RFL_TRUNCATE_SIZE	((FLMUINT)1024 * (FLMUINT)1024 * (FLMUINT)10)

/*============================================================================
									Shared Cache Routines
============================================================================*/

RCODE flmPrepareBlockForUse(
	FLMUINT				uiBlockSize,
	F_BLK_HDR *			pBlkHdr);

RCODE flmPrepareBlockToWrite(
	FLMUINT				uiBlockSize,
	F_BLK_HDR *			pBlkHdr);

void ScaUseCache(
	F_CachedBlock *	pSCache,
	FLMBOOL				bMutexAlreadyLocked);
	
void ScaReleaseCache(
	F_CachedBlock *	pSCache,
	FLMBOOL				bMutexAlreadyLocked);

/*============================================================================
							File system Btree Cache Routines
============================================================================*/

FLMUINT SENNextVal(
	FLMBYTE **			senPtrRV	);

FLMUINT FSGetDomain(
	FLMBYTE **			curElmRV,
	FLMUINT				uiElmOvhd);

RCODE	dbLock(
	F_Db *				pDb,
	FLMUINT				uiMaxLockWait	);

RCODE	dbUnlock(
	F_Db *				pDb);

FINLINE void FSLFileIn(
	FLMBYTE *			pucBuf,
	LFILE *				pLFile,
	F_COLLECTION *		pCollection,
	FLMUINT				uiBlkAddress,
	FLMUINT				uiOffsetInBlk)
{
	F_LF_HDR *	pLfHdr = (F_LF_HDR *)pucBuf;

	pLFile->uiBlkAddress	= uiBlkAddress;
	pLFile->uiOffsetInBlk= uiOffsetInBlk;

	if ((pLFile->eLfType = (eLFileType)pLfHdr->ui32LfType) != XFLM_LF_INVALID)
	{
		pLFile->uiLfNum = (FLMUINT)pLfHdr->ui32LfNumber;
		pLFile->uiRootBlk = (FLMUINT)pLfHdr->ui32RootBlkAddr;
		pLFile->uiEncId = (FLMUINT)pLfHdr->ui32EncId;
		
		if (pCollection)
		{
			flmAssert( pLFile == &pCollection->lfInfo);
			flmAssert( pLFile->eLfType == XFLM_LF_COLLECTION);
			pCollection->ui64NextNodeId = pLfHdr->ui64NextNodeId;
			pCollection->ui64FirstDocId = pLfHdr->ui64FirstDocId;
			pCollection->ui64LastDocId = pLfHdr->ui64LastDocId;
			pCollection->bNeedToUpdateNodes = FALSE;
		}
		else
		{
			flmAssert( pLFile->eLfType == XFLM_LF_INDEX);
		}
	}
}

void lgSetSyncCheckpoint(
	F_Database *			pDatabase,
	FLMUINT					uiCheckpoint,
	FLMUINT					uiBlkAddress);

FLMUINT32 calcBlkCRC(
	F_BLK_HDR *				pBlkHdr,
	FLMUINT					uiBlkEnd);

FLMUINT flmAdjustBlkSize(
	FLMUINT					uiBlkSize);

void flmInitDbHdr(
	XFLM_CREATE_OPTS *	pCreateOpts,
	FLMBOOL					bCreatingDatabase,
	FLMBOOL					bTempDb,
	XFLM_DB_HDR *			pDbHdr);

RCODE flmCreateLckFile(
	const char *			pszFilePath,
	IF_FileHdl **			ppLockFileHdl);

RCODE flmReadAndVerifyHdrInfo(
	XFLM_DB_STATS *		pDbStats,
	IF_FileHdl *			pFileHdl,
	XFLM_DB_HDR *			pDbHdr,
	FLMUINT32 *				pui32CalcCRC = NULL);

void flmDoEventCallback(
	eEventCategory			eCategory,
	eEventType				eEvent,
	IF_Db *					pDb,
	FLMUINT					uiThreadId,
	FLMUINT64				ui64TransID,
	FLMUINT					uiIndexOrCollection,
	FLMUINT64				ui64NodeId,
	RCODE						rc);

void flmLogError(
	RCODE						rc,
	const char *			pszDoing,
	const char *			pszFileName = NULL,
	FLMINT					iLineNumber = 0);

RCODE flmCollation2Number(
	FLMUINT					uiBufLen,
	const FLMBYTE *		pucBuf,
	FLMUINT64 *				pui64Num,
	FLMBOOL *				pbNeg,
	FLMUINT *				puiBytesProcessed);

RCODE flmStorageNum2CollationNum(
	const FLMBYTE *		pucStorageBuf,
	FLMUINT					uiStorageLen,
	FLMBYTE *				pucCollBuf,
	FLMUINT *				puiCollLen);

RCODE flmCollationNum2StorageNum(
	const FLMBYTE *		pucCollBuf,
	FLMUINT					uiCollLen,
	FLMBYTE *				pucStorageBuf,
	FLMUINT *				puiStorageLen);

RCODE flmStorageNum2StorageText(
	const FLMBYTE *		pucNum,
	FLMUINT					uiNumLen,
	FLMBYTE *				pucBuffer,
	FLMUINT *				puiBufLen);

RCODE flmStorageNumberToNumber(
	const FLMBYTE *		pucNumBuf,
	FLMUINT					uiNumBufLen,
	FLMUINT64 *				pui64Number,
	FLMBOOL *				pbNeg);
	
void kyReleaseCdls(
	IXD *						pIxd,
	CDL_HDR *				pCdlTbl);

RCODE KYCollateValue(
	FLMBYTE *				pucDest,
	FLMUINT *				puiDestLen,
	IF_PosIStream *		pIStream,
	FLMUINT					uiDataType,
	FLMUINT					uiFlags,
	FLMUINT					uiCompareRules,
	FLMUINT					uiLimit,
	FLMUINT *				puiCollationLen,
	FLMUINT *				puiLuLen,
	FLMUINT					uiLanguage,
	FLMBOOL					bFirstSubstring,
	FLMBOOL					bDataTruncated,
	FLMBOOL *				pbDataTruncated,
	FLMBOOL *				pbOriginalCharsLost);

#define UNDF_CHR			0x0000		// Undefined char - ignore for now
#define IGNR_CHR			0x0001		// Ignore this char
#define SDWD_CHR			0x0002		// Space delimited word chr
#define DELI_CHR			0x0040		// Delimiter
#define WDJN_CHR			0x0080		// Word Joining chr ".,/-_"

// Implement later

#define KATA_CHR			0x0004		// Katakana word chr
#define HANG_CHR			0x0008		// Hangul word chr
#define CJK_CHR			0x0010		// CJK word chr

RCODE KYSubstringParse(
	IF_PosIStream *		pIStream,
	FLMUINT *				puiCompareRules,
	FLMUINT					uiLimit,
	FLMBYTE *				pucSubstrBuf,
	FLMUINT *				puiSubstrBytes,
	FLMUINT *				puiSubstrChars);

RCODE KYEachWordParse(
	IF_PosIStream *		pIStream,
	FLMUINT *				puiCompareRules,
	FLMUINT					uiLimit,
	FLMBYTE *				pucWordBuf,
	FLMUINT *				puiWordLen);

RCODE fdictGetState(
	const char *			pszState,
	FLMUINT *				puiState);

RCODE fdictGetIndexState(
	const char *			pszState,
	FLMUINT *				puiState);

#define FLM_BACKGROUND_LOCK_PRIORITY			-100

void flmLogIndexingProgress(
	FLMUINT					uiIndexNum,
	FLMUINT64				ui64LastDocumentId);

F_BKGND_IX * flmBackgroundIndexGet(
	F_Database *			pDatabase,
	FLMUINT					uiValue,
	FLMBOOL					bMutexLocked,
	FLMUINT *				puiThreadId = NULL);

RCODE flmGetHdrInfo(
	F_SuperFileHdl *		pSFileHdl,
	XFLM_DB_HDR *			pDbHdr,
	FLMUINT32 *				pui32CalcCRC = NULL);

void convert64(
	FLMUINT64 *				pui64Num);

void convert32(
	FLMUINT32 *				pui32Num);

void convert16(
	FLMUINT16 *				pui16Num);

void convertDbHdr(
	XFLM_DB_HDR *			pDbHdr);

void convertBlkHdr(
	F_BLK_HDR *				pBlkHdr);

void convertBlk(
	FLMUINT					uiBlockSize,
	F_BLK_HDR *				pBlkHdr);

void convertLfHdr(
	F_LF_HDR *				pLfHdr);

/*--------------------------------------------------------
** 	Inline Functions
**-------------------------------------------------------*/

/**************************************************************************
Desc:	Returns TRUE if a block address is less than another block address.
**************************************************************************/
FINLINE FLMBOOL FSAddrIsBelow(
	FLMUINT				uiAddress1,
	FLMUINT				uiAddress2)
{
	if( FSGetFileNumber( uiAddress1) == FSGetFileNumber( uiAddress2))
	{
		if( FSGetFileOffset( uiAddress1) >= FSGetFileOffset( uiAddress2))
		{
			return FALSE;
		}
	}
	else if( FSGetFileNumber( uiAddress1) > FSGetFileNumber( uiAddress2))
	{
		return FALSE;
	}
	
	return TRUE;
}

/**************************************************************************
Desc:	Returns TRUE if a block address is less than or equal another
		block address.
**************************************************************************/
FINLINE FLMBOOL FSAddrIsAtOrBelow(
	FLMUINT			uiAddress1,
	FLMUINT			uiAddress2)
{
	if( FSGetFileNumber( uiAddress1) == FSGetFileNumber( uiAddress2))
	{
		if( FSGetFileOffset( uiAddress1) > FSGetFileOffset( uiAddress2))
		{
			return FALSE;
		}
	}
	else if( FSGetFileNumber( uiAddress1) > FSGetFileNumber( uiAddress2))
	{
		return FALSE;
	}
	return TRUE;
}

/****************************************************************************
Desc:	Get the total bytes represented by a particular block address.
****************************************************************************/
FINLINE FLMUINT64 FSGetSizeInBytes(
	FLMUINT	uiMaxFileSize,
	FLMUINT	uiBlkAddress)
{
	FLMUINT	uiFileNum;
	FLMUINT	uiFileOffset;
	FLMUINT64	ui64Size;

	uiFileNum = FSGetFileNumber( uiBlkAddress);
	uiFileOffset = FSGetFileOffset( uiBlkAddress);
	if( uiFileNum > 1)
	{
		ui64Size = (FLMUINT64)(((FLMUINT64)uiFileNum - (FLMUINT64)1) *
											(FLMUINT64)uiMaxFileSize +
											(FLMUINT64)uiFileOffset);
	}
	else
	{
		ui64Size = (FLMUINT64)uiFileOffset;
	}
	return( ui64Size);
}

/**************************************************************************
Desc:	Calculate the significant bits in a block size.
**************************************************************************/
FINLINE FLMUINT calcSigBits(
	FLMUINT	uiBlockSize
	)
{
	FLMUINT	uiSigBitsInBlkSize = 0;
	while (!(uiBlockSize & 1))
	{
		uiSigBitsInBlkSize++;
		uiBlockSize >>= 1;
	}
	return( uiSigBitsInBlkSize);
}

/**************************************************************************
Desc:	Outputs an update event callback.
**************************************************************************/
FINLINE void flmTransEventCallback(
	eEventType	eEventType,
	F_Db *		pDb,
	RCODE			rc,
	FLMUINT64	ui64TransId)
{
	flmDoEventCallback( XFLM_EVENT_UPDATES, eEventType, (IF_Db *)pDb,
					f_threadId(), ui64TransId, 0, 0, rc);
}

/********************************************************************
Desc:	Calculate the CRC for the database header.
*********************************************************************/
FINLINE FLMUINT32 calcDbHdrCRC(
	XFLM_DB_HDR *			pDbHdr)
{
	FLMUINT32	ui32SaveCRC;
	FLMUINT32	ui32Checksum;

	// Checksum everything except for the ui32HdrCRC value.

	ui32SaveCRC = pDbHdr->ui32HdrCRC;
	pDbHdr->ui32HdrCRC = 0;
	
	// Calculate the checksum

	ui32Checksum = f_calcFastChecksum( pDbHdr, sizeof( XFLM_DB_HDR));
	
	// Restore the checksum that was in the header

	pDbHdr->ui32HdrCRC = ui32SaveCRC;
	return( ui32Checksum);
}

/********************************************************************
Desc:	Calculate the checksum for a block.  NOTE: This is ALWAYS done
		on the raw image that will be written to disk.  This means
		that if the block needs to be converted before writing it out,
		it should be done before calculating the checksum.
*********************************************************************/
FINLINE FLMUINT32 calcBlkCRC(
	F_BLK_HDR *		pBlkHdr,
	FLMUINT			uiBlkEnd)
{
	FLMUINT32		ui32SaveCRC;
	FLMUINT32		ui32Checksum;

	// Calculate CRC on everything except for the ui32BlkCRC value.
	// To do this, we temporarily change it to zero.  The saved
	// value will be restored after calculating the CRC.

	ui32SaveCRC = pBlkHdr->ui32BlkCRC;
	pBlkHdr->ui32BlkCRC = 0;

	// Calculate the checksum

	ui32Checksum = f_calcFastChecksum( pBlkHdr, uiBlkEnd);
	
	// Restore the CRC that was in the block.

	pBlkHdr->ui32BlkCRC = ui32SaveCRC;
	return( ui32Checksum);
}

#endif	// FILESYS_H

