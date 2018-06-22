//-------------------------------------------------------------------------
// Desc:	Database header routines.
// Tabs:	3
//
// Copyright (c) 1995-2007 Novell, Inc. All Rights Reserved.
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

/********************************************************************
Desc: 	Initializes the file prefix for the .db database file.
*********************************************************************/
void flmSetFilePrefix(
	FLMBYTE *		pPrefix,
	FLMUINT			uiMajorVer,
	FLMUINT			uiMinorVer)
{
	f_memset( pPrefix, 0, 16);
	pPrefix [0] = 0xFF;
	pPrefix [1] = f_toascii('W');
	pPrefix [2] = f_toascii('P');
	pPrefix [3] = f_toascii('C');
		
	UD2FBA( (FLMUINT32)16, &pPrefix [4]);

	pPrefix [8] = 0xF3;		// old product type
	pPrefix [9] = 0x01;		// old file type
	pPrefix [10] = (FLMBYTE)uiMajorVer;
	pPrefix [11] = (FLMBYTE)uiMinorVer;

	// Bytes 12 and 13 are the encryption key (not used)

	pPrefix [12] = 0; 
	pPrefix [13] = 0;

	// Bytes 14 and 15 point are the offset to file specific packets

	pPrefix [14] = 0; 
	pPrefix [15] = 0;
}

/********************************************************************
Desc:	This routine adjusts the block size to the nearest valid
		block size.
*********************************************************************/
FLMUINT flmAdjustBlkSize(
	FLMUINT	uiBlkSize)
{
	FLMUINT	uiTmpBlkSize;
	
	uiTmpBlkSize = MIN_BLOCK_SIZE;
	while( (uiBlkSize > uiTmpBlkSize) && (uiTmpBlkSize < MAX_BLOCK_SIZE))
	{
		uiTmpBlkSize <<= 1;
	}

	return( uiTmpBlkSize);
}

/***************************************************************************
Desc:	This routine extracts and verifies the information within
		the file header.
*****************************************************************************/
RCODE flmGetFileHdrInfo(
	FLMBYTE *		pPrefixBuf,
	FLMBYTE *		pFileHdrBuf,
	FILE_HDR *		pFileHdrRV)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiVersionNum;
	FLMUINT			uiTmpBlkSize;

	// Get the create options

	pFileHdrRV->uiBlockSize = (FLMUINT)FB2UW( &pFileHdrBuf [DB_BLOCK_SIZE]);
	pFileHdrRV->uiAppMajorVer = pPrefixBuf [10];
	pFileHdrRV->uiAppMinorVer = pPrefixBuf [11];
	pFileHdrRV->uiDefaultLanguage = pFileHdrBuf [DB_DEFAULT_LANGUAGE];
	pFileHdrRV->uiVersionNum = uiVersionNum =
		((FLMUINT16)(pFileHdrBuf [FLM_FILE_FORMAT_VER_POS] - ASCII_ZERO) * 100 +
		 (FLMUINT16)(pFileHdrBuf [FLM_MINOR_VER_POS] - ASCII_ZERO) * 10 +
		 (FLMUINT16)(pFileHdrBuf [FLM_SMINOR_VER_POS] - ASCII_ZERO));

	uiTmpBlkSize = pFileHdrRV->uiBlockSize;
	if( !VALID_BLOCK_SIZE( uiTmpBlkSize))
	{
		uiTmpBlkSize = flmAdjustBlkSize( pFileHdrRV->uiBlockSize);
	}

	// Get other log header elements.

	pFileHdrRV->uiFirstLFHBlkAddr =
		(FLMUINT)FB2UD( &pFileHdrBuf [DB_1ST_LFH_ADDR]);

	// See if this looks like a valid database

	if( (pPrefixBuf [1] != f_toascii('W')) ||
		 (pPrefixBuf [2] != f_toascii('P')) ||
		 (pPrefixBuf [3] != f_toascii('C')) || 
		 (!VALID_BLOCK_SIZE( pFileHdrRV->uiBlockSize)))
	{
		rc = RC_SET( FERR_NOT_FLAIM);
		goto Exit;
	}

	if( pFileHdrBuf [FLAIM_NAME_POS     ] != f_toascii( FLAIM_NAME[0]) || 
		 pFileHdrBuf [FLAIM_NAME_POS + 1 ] != f_toascii( FLAIM_NAME[1]) || 
		 pFileHdrBuf [FLAIM_NAME_POS + 2 ] != f_toascii( FLAIM_NAME[2]) || 
		 pFileHdrBuf [FLAIM_NAME_POS + 3 ] != f_toascii( FLAIM_NAME[3]) || 
		 pFileHdrBuf [FLAIM_NAME_POS + 4 ] != f_toascii( FLAIM_NAME[4]))
	{
		rc = RC_SET( FERR_NOT_FLAIM);
		goto Exit;
	}

	pFileHdrRV->uiSigBitsInBlkSize = flmGetSigBits( pFileHdrRV->uiBlockSize);

	// Check the FLAIM version number

	if( RC_BAD( rc = flmCheckVersionNum( uiVersionNum)))
	{
		goto Exit;
	}

	f_memcpy( pFileHdrRV->ucFileHdr, pFileHdrBuf, FLM_FILE_HEADER_SIZE);
	
Exit:

	return( rc);
}

/********************************************************************
Desc: This routine initializes a FILE_HDR structure from the
		create options that are passed in.  It also initializes the
		file header buffer (pFileHdrBuf) that will be written to disk.
*********************************************************************/
void flmInitFileHdrInfo(
	CREATE_OPTS *	pCreateOpts,
	FILE_HDR *		pFileHdr,
	FLMBYTE *		pFileHdrBuf)
{
	f_memset( pFileHdrBuf, 0, FLM_FILE_HEADER_SIZE);

	// If pCreateOpts is non-NULL, copy it into the file header.

	if (pCreateOpts)
	{
		pFileHdr->uiBlockSize = pCreateOpts->uiBlockSize;
		pFileHdr->uiDefaultLanguage = pCreateOpts->uiDefaultLanguage;
		pFileHdr->uiAppMajorVer = pCreateOpts->uiAppMajorVer;
		pFileHdr->uiAppMinorVer = pCreateOpts->uiAppMinorVer;
	}
	else
	{

		// If pCreateOpts is NULL, initialize some default values.

		pFileHdr->uiBlockSize = DEFAULT_BLKSIZ;
		pFileHdr->uiDefaultLanguage = DEFAULT_LANG;
		pFileHdr->uiAppMajorVer =
		pFileHdr->uiAppMinorVer = 0;
	}

	// Only allow database to be created with current version number

	pFileHdr->uiVersionNum = FLM_CUR_FILE_FORMAT_VER_NUM;
	f_memcpy( &pFileHdrBuf [FLM_FILE_FORMAT_VER_POS], 
					(FLMBYTE *)FLM_CUR_FILE_FORMAT_VER_STR,
					FLM_FILE_FORMAT_VER_LEN);

	// Round block size up to nearest legal block size.

	pFileHdr->uiBlockSize =
		flmAdjustBlkSize( pFileHdr->uiBlockSize);

	pFileHdr->uiSigBitsInBlkSize = flmGetSigBits( pFileHdr->uiBlockSize);
	f_memcpy( &pFileHdrBuf [FLAIM_NAME_POS], (FLMBYTE *)FLAIM_NAME,
				 FLAIM_NAME_LEN);

	pFileHdrBuf [DB_DEFAULT_LANGUAGE] =
		(FLMBYTE)pFileHdr->uiDefaultLanguage;
	UW2FBA( (FLMUINT16)pFileHdr->uiBlockSize,
		&pFileHdrBuf [DB_BLOCK_SIZE]);
	pFileHdr->uiFirstLFHBlkAddr = FSBlkAddress(1, 0);
	UD2FBA( (FLMUINT32)pFileHdr->uiFirstLFHBlkAddr, &pFileHdrBuf [DB_1ST_LFH_ADDR]);

	if (pFileHdr->uiVersionNum < FLM_FILE_FORMAT_VER_4_3)
	{

		// Things to maintain for backward compatibility - pre 4.3.

		FLMUINT	uiFirstPcodeAddr = pFileHdr->uiFirstLFHBlkAddr +
											 pFileHdr->uiBlockSize;

		UD2FBA( (FLMUINT32)pFileHdr->uiBlockSize, &pFileHdrBuf [DB_INIT_LOG_SEG_ADDR]);
		UD2FBA( DB_LOG_HEADER_START, &pFileHdrBuf [DB_LOG_HEADER_ADDR]);
		UD2FBA( (FLMUINT32)uiFirstPcodeAddr, &pFileHdrBuf [DB_1ST_PCODE_ADDR]);
	}
	
	f_memcpy( pFileHdr->ucFileHdr, pFileHdrBuf, FLM_FILE_HEADER_SIZE);
}

/***************************************************************************
Desc:	This routine reads and verifies the information contained in the
		file header and log header of a FLAIM database.  This routine
		is called by both FlmDbOpen and flmGetHdrInfo.
*****************************************************************************/
RCODE flmReadAndVerifyHdrInfo(
	DB_STATS *		pDbStats,
	IF_FileHdl *	pFileHdl,
	FLMBYTE *		pucReadBuf,
	FILE_HDR *		pFileHdrRV,
	LOG_HDR *		pLogHdrRV,
	FLMBYTE *		pLogHdr)
{
	RCODE				rc = FERR_OK;
	RCODE				rc0;
	RCODE				rc1;
	FLMBYTE *		pucBuf;
	FLMBYTE *		pucLogHdr;
	FLMUINT			uiBytesRead;
	FLMUINT			uiVersionNum;

	// Read the fixed information area

	rc0 = pFileHdl->read( 0, 2048, pucReadBuf, &uiBytesRead);
	
	// Increment bytes read - to account for byte zero, which
	// was not really read in.

	pucBuf = pucReadBuf;
	*pucBuf = 0xFF;

	// Before doing any checking, get whatever we can from the
	// first 2048 bytes.  For the flmGetHdrInfo routine, we want
	// to get whatever we can from the headers, even if it is
	// invalid.

	rc1 = flmGetFileHdrInfo( pucBuf, &pucBuf[ FLAIM_HEADER_START], pFileHdrRV);

	// Get the log header information

	pucLogHdr = &pucBuf[ DB_LOG_HEADER_START];

	if( pLogHdr)
	{
		f_memcpy( pLogHdr, pucLogHdr, LOG_HEADER_SIZE);
	}
	
	if( pLogHdrRV)
	{
		flmGetLogHdrInfo( pucLogHdr, pLogHdrRV);
	}

	// Take the version from the log header if non-zero.
	// Storing the version in the log header is new to 40 code base.

	uiVersionNum = FB2UW( &pucLogHdr[ LOG_FLAIM_VERSION]);
	if( uiVersionNum)
	{
		pFileHdrRV->uiVersionNum = uiVersionNum;
	}

	// If there is not enough data to satisfy the read, this
	// is probably not a FLAIM file.

	if( RC_BAD( rc0))
	{
		if( rc0 != FERR_IO_END_OF_FILE)
		{
			if( pDbStats)
			{
				pDbStats->uiReadErrors++;
			}

			rc = rc0;
			goto Exit;
		}

		if( uiBytesRead < 2048)
		{
			rc = RC_SET( FERR_NOT_FLAIM);
			goto Exit;
		}
	}

	// See if we got any other errors where we might want to retry
	// the read.

	if( RC_BAD( rc1))
	{
		rc = rc1;
		goto Exit;
	}

	// Verify the checksums in the log header

	if( lgHdrCheckSum( pucLogHdr, TRUE) != 0)
	{
		rc = RC_SET( FERR_BLOCK_CHECKSUM);
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Write the version number to disk and flush the write to disk.
*****************************************************************************/
RCODE flmWriteVersionNum(
	F_SuperFileHdl *		pSFileHdl,
	FLMUINT					uiVersionNum)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiWriteBytes;
	FLMBYTE	szVersionStr[ 8];

	if( RC_BAD( rc = flmCheckVersionNum( uiVersionNum)))
	{
		flmAssert( 0);
		goto Exit;
	}

	szVersionStr[ 0] = (FLMBYTE)(uiVersionNum / 100) + '0';
	szVersionStr[ 1] = '.';
	szVersionStr[ 2] = (FLMBYTE)((uiVersionNum % 100) / 10) + '0';
	szVersionStr[ 3] = (FLMBYTE)(uiVersionNum % 10) + '0';
	szVersionStr[ 4] = 0;

	if (RC_OK( rc = pSFileHdl->writeBlock(
					FSBlkAddress( 0, FLAIM_HEADER_START + FLM_FILE_FORMAT_VER_POS), 
					FLM_FILE_FORMAT_VER_LEN,
					szVersionStr, &uiWriteBytes)))
	{
		if (RC_BAD( rc = pSFileHdl->flush()))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	This routine reads the header information in a FLAIM database,
		verifies the password, and returns the file header and log
		header information.
*****************************************************************************/
RCODE flmGetHdrInfo(
	IF_FileHdl *		pFileHdl,
	FILE_HDR *			pFileHdrRV,
	LOG_HDR *			pLogHdrRV,
	FLMBYTE *			pLogHdr)
{
	RCODE					rc = FERR_OK;
	FLMBYTE *			pucBuf = NULL;

	if( RC_BAD( rc = f_allocAlignedBuffer( 2048, &pucBuf)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = flmReadAndVerifyHdrInfo( NULL, pFileHdl, 
		pucBuf, pFileHdrRV, pLogHdrRV, pLogHdr)))
	{
		goto Exit;
	}

Exit:

	if( pucBuf)
	{
		f_freeAlignedBuffer( &pucBuf);
	}

	return( rc);
}
