//------------------------------------------------------------------------------
// Desc:	Routines for accessing information in the database header.
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

FSTATIC RCODE verifyDbHdr(
	SFLM_DB_HDR *	pDbHdr);

/********************************************************************
Desc:	This routine adjusts the block size that is passe in (wBlkSize)
		to the nearest valid block size.
*********************************************************************/
FLMUINT flmAdjustBlkSize(
	FLMUINT	uiBlkSize)
{
	FLMUINT	uiTmpBlkSize;

	uiTmpBlkSize = SFLM_MIN_BLOCK_SIZE;
	while (uiBlkSize > uiTmpBlkSize && uiTmpBlkSize < SFLM_MAX_BLOCK_SIZE)
	{
		uiTmpBlkSize <<= 1;
	}

	return( uiTmpBlkSize);
}

/********************************************************************
Desc: This routine initializes a SFLM_DB_HDR structure.
*********************************************************************/
void flmInitDbHdr(
	SFLM_CREATE_OPTS *	pCreateOpts,
	FLMBOOL					bCreatingDatabase,
	FLMBOOL					bTempDb,
	SFLM_DB_HDR *			pDbHdr)
{
	FLMUINT	uiMinRflFileSize;
	FLMUINT	uiMaxRflFileSize;

	if (bCreatingDatabase)
	{
		f_memset( pDbHdr, 0, sizeof( SFLM_DB_HDR));
	}

	// If pCreateOpts is non-NULL, copy it into the file header.

	f_strcpy( (char *)pDbHdr->szSignature, SFLM_DB_SIGNATURE);
	pDbHdr->ui8IsLittleEndian = SFLM_NATIVE_IS_LITTLE_ENDIAN;

	if (pCreateOpts)
	{
		pDbHdr->ui16BlockSize = (FLMUINT16)pCreateOpts->uiBlockSize;
		pDbHdr->ui8DefaultLanguage = (FLMUINT8)pCreateOpts->uiDefaultLanguage;
		if (pCreateOpts->bKeepRflFiles)
		{
			pDbHdr->ui8RflKeepFiles = 1;
		}
		if (pCreateOpts->bLogAbortedTransToRfl)
		{
			pDbHdr->ui8RflKeepAbortedTrans = 1;
		}

		if( (uiMinRflFileSize = pCreateOpts->uiMinRflFileSize) == 0)
		{
			uiMinRflFileSize = SFLM_DEFAULT_MIN_RFL_FILE_SIZE;
		}

		if( (uiMaxRflFileSize = pCreateOpts->uiMaxRflFileSize) == 0)
		{
			uiMaxRflFileSize = SFLM_DEFAULT_MAX_RFL_FILE_SIZE;
		}
	}
	else
	{

		// If pCreateOpts is NULL, initialize some default values.

		pDbHdr->ui16BlockSize = SFLM_DEFAULT_BLKSIZ;
		pDbHdr->ui8DefaultLanguage = SFLM_DEFAULT_LANG;
		uiMinRflFileSize = SFLM_DEFAULT_MIN_RFL_FILE_SIZE;
		uiMaxRflFileSize = SFLM_DEFAULT_MAX_RFL_FILE_SIZE;
	}

	// Make sure the RFL size limits are valid.
	// Maximum must be enough to hold at least one packet plus
	// the RFL header.  Minimum must not be greater than the
	// maximum.  NOTE: Minimum and maximum are allowed to be
	// equal, but in all cases, maximum takes precedence over
	// minimum.  We will first NOT exceed the maximum.  Then,
	// if possible, we will go above the minimum.

	if (uiMaxRflFileSize < RFL_MAX_PACKET_SIZE + 512)
	{
		uiMaxRflFileSize = RFL_MAX_PACKET_SIZE + 512;
	}
	if (uiMaxRflFileSize > gv_SFlmSysData.uiMaxFileSize)
	{
		uiMaxRflFileSize = gv_SFlmSysData.uiMaxFileSize;
	}
	if (uiMinRflFileSize > uiMaxRflFileSize)
	{
		uiMinRflFileSize = uiMaxRflFileSize;
	}
	pDbHdr->ui32RflMinFileSize = (FLMUINT32)uiMinRflFileSize;
	pDbHdr->ui32RflMaxFileSize = (FLMUINT32)uiMaxRflFileSize;

	// Only allow database to be created with current version number

	pDbHdr->ui32DbVersion = SFLM_CURRENT_VERSION_NUM;
	pDbHdr->ui8BlkChkSummingEnabled = 1;

	// Round block size up to nearest legal block size.

	pDbHdr->ui16BlockSize =
		(FLMUINT16)flmAdjustBlkSize( (FLMUINT)pDbHdr->ui16BlockSize);

	if (!bTempDb)
	{
		pDbHdr->ui32FirstLFBlkAddr = (FLMUINT32)FSBlkAddress(1, 0);
	}

	// If creating a database, initialize some more items.

	if (bCreatingDatabase)
	{

		// Set the logical EOF.

		if (!bTempDb)
		{
			pDbHdr->ui32LogicalEOF = pDbHdr->ui32FirstLFBlkAddr +
											(FLMUINT32)pDbHdr->ui16BlockSize;
		}
		else
		{
			pDbHdr->ui32LogicalEOF = (FLMUINT32)FSBlkAddress(1, 0);
		}
		pDbHdr->ui64CurrTransID = (FLMUINT64)0;
		pDbHdr->ui32RflCurrFileNum = 1;

		// Putting a zero in this value tells the RFL code that the
		// RFL file should be created - overwriting it if it already
		// exists.

		pDbHdr->ui32RflLastCPFileNum = 1;
		pDbHdr->ui32RflLastCPOffset = 512;
		pDbHdr->ui32RblEOF = pDbHdr->ui16BlockSize;

		// Set the database serial number

		f_createSerialNumber( pDbHdr->ucDbSerialNum);

		// Set the "current" RFL serial number - will be stamped into the RFL
		// file when it is first created.

		f_createSerialNumber( pDbHdr->ucLastTransRflSerialNum);

		// Set the "next" RFL serial number

		f_createSerialNumber( pDbHdr->ucNextRflSerialNum);

		// Set the incremental backup serial number and sequence number

		f_createSerialNumber( pDbHdr->ucIncBackupSerialNum);
		pDbHdr->ui32IncBackupSeqNum = 1;

		// Set the file size limits

		pDbHdr->ui32MaxFileSize = (FLMUINT32)gv_SFlmSysData.uiMaxFileSize;

		// Need at least two blocks to create a database!

		flmAssert( (FLMUINT)pDbHdr->ui32MaxFileSize >=
					  (FLMUINT)pDbHdr->ui16BlockSize * 2);
	}
}

/***************************************************************************
Desc:	This routine changes the endian-ness of the passed in 64 bit number.
		If it was big-endian, it will be converted to little-endian and
		vice versa.
*****************************************************************************/
void convert64(
	FLMUINT64 *	pui64Num
	)
{
	FLMBYTE *	pucTmp = (FLMBYTE *)pui64Num;
	FLMBYTE		ucTmp;

	// Swap bytes 0 and 7

	ucTmp = pucTmp [0];
	pucTmp [0] = pucTmp [7];
	pucTmp [7] = ucTmp;

	// Swap bytes 1 and 6

	ucTmp = pucTmp [1];
	pucTmp [1] = pucTmp [6];
	pucTmp [6] = ucTmp;

	// Swap bytes 2 and 5

	ucTmp = pucTmp [2];
	pucTmp [2] = pucTmp [5];
	pucTmp [5] = ucTmp;

	// Swap bytes 3 and 4

	ucTmp = pucTmp [3];
	pucTmp [3] = pucTmp [4];
	pucTmp [4] = ucTmp;
}

/***************************************************************************
Desc:	This routine changes the endian-ness of the passed in 32 bit number.
		If it was big-endian, it will be converted to little-endian and
		vice versa.
*****************************************************************************/
void convert32(
	FLMUINT32 *	pui32Num
	)
{
	FLMBYTE *	pucTmp = (FLMBYTE *)pui32Num;
	FLMBYTE		ucTmp;

	// Swap bytes 0 and 3

	ucTmp = pucTmp [0];
	pucTmp [0] = pucTmp [3];
	pucTmp [3] = ucTmp;

	// Swap bytes 1 and 2

	ucTmp = pucTmp [1];
	pucTmp [1] = pucTmp [2];
	pucTmp [2] = ucTmp;
}

/***************************************************************************
Desc:	This routine changes the endian-ness of the passed in 16 bit number.
		If it was big-endian, it will be converted to little-endian and
		vice versa.
*****************************************************************************/
void convert16(
	FLMUINT16 *	pui16Num
	)
{
	FLMBYTE *	pucTmp = (FLMBYTE *)pui16Num;
	FLMBYTE		ucTmp;

	// Swap bytes 0 and 1

	ucTmp = pucTmp [0];
	pucTmp [0] = pucTmp [1];
	pucTmp [1] = ucTmp;
}

/***************************************************************************
Desc:	This routine changed a database header to native platform format.
*****************************************************************************/
void convertDbHdr(
	SFLM_DB_HDR *	pDbHdr
	)
{

	// This routine should only be called to convert a header to native
	// format.

	flmAssert( hdrIsNonNativeFormat( pDbHdr));
	convert16( &pDbHdr->ui16BlockSize);
	convert32( &pDbHdr->ui32DbVersion);
	convert64( &pDbHdr->ui64LastRflCommitID);
	convert32( &pDbHdr->ui32RflLastFileNumDeleted);
	convert32( &pDbHdr->ui32RflCurrFileNum);
	convert32( &pDbHdr->ui32RflLastTransOffset);
	convert32( &pDbHdr->ui32RflLastCPFileNum);
	convert32( &pDbHdr->ui32RflLastCPOffset);
	convert64( &pDbHdr->ui64RflLastCPTransID);
	convert32( &pDbHdr->ui32RflMinFileSize);
	convert32( &pDbHdr->ui32RflMaxFileSize);
	convert64( &pDbHdr->ui64CurrTransID);
	convert64( &pDbHdr->ui64TransCommitCnt);
	convert32( &pDbHdr->ui32RblEOF);
	convert32( &pDbHdr->ui32RblFirstCPBlkAddr);
	convert32( &pDbHdr->ui32FirstAvailBlkAddr);
	convert32( &pDbHdr->ui32FirstLFBlkAddr);
	convert32( &pDbHdr->ui32LogicalEOF);
	convert32( &pDbHdr->ui32MaxFileSize);
	convert64( &pDbHdr->ui64LastBackupTransID);
	convert32( &pDbHdr->ui32IncBackupSeqNum);
	convert32( &pDbHdr->ui32BlksChangedSinceBackup);
	convert32( &pDbHdr->ui32HdrCRC);
	pDbHdr->ui8IsLittleEndian = SFLM_NATIVE_IS_LITTLE_ENDIAN;
}

/***************************************************************************
Desc:	This routine verifies that the header is a real FLAIM header.
		It will also change the endian-ness if need be.  This should always
		be called immediately after reading a header from disk.
*****************************************************************************/
FSTATIC RCODE verifyDbHdr(
	SFLM_DB_HDR *	pDbHdr
	)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiLen;
	FLMUINT32	ui32CRC;

	// Calculate the checksum before doing any conversions.

	ui32CRC = calcDbHdrCRC( pDbHdr);

	// Convert the header to native platform format if necessary.

	if (hdrIsNonNativeFormat( pDbHdr))
	{
		convertDbHdr( pDbHdr);
	}

	// Check the signature.

	uiLen = f_strlen( SFLM_DB_SIGNATURE);
	if (f_memcmp( pDbHdr->szSignature, SFLM_DB_SIGNATURE, uiLen) != 0)
	{
		rc = RC_SET( NE_SFLM_NOT_FLAIM);
		goto Exit;
	}

	// See if the database version is OK.

	switch (pDbHdr->ui32DbVersion)
	{
		case SFLM_CURRENT_VERSION_NUM:
		{
			break;
		}
		
		default:
		{
			if (pDbHdr->ui32DbVersion > SFLM_CURRENT_VERSION_NUM)
			{
				rc = RC_SET( NE_SFLM_NEWER_FLAIM);
			}
			else
			{
				rc = RC_SET( NE_SFLM_UNSUPPORTED_VERSION);
			}
			
			goto Exit;
		}
	}

	// Validate the checksum

	if (ui32CRC != pDbHdr->ui32HdrCRC)
	{
		rc = RC_SET( NE_SFLM_HDR_CRC);
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	This routine reads and verifies the information contained in the
		file header and log header of a FLAIM database.
*****************************************************************************/
RCODE flmReadAndVerifyHdrInfo(
	SFLM_DB_STATS *	pDbStats,
	IF_FileHdl *		pFileHdl,
	SFLM_DB_HDR *		pDbHdr,
	FLMUINT32 *			pui32CalcCRC)
{
	RCODE		rc = NE_SFLM_OK;
	FLMUINT	uiBytesRead;

	// Read the database header.

	f_memset( pDbHdr, 0, sizeof( SFLM_DB_HDR));
	if (RC_BAD( rc = pFileHdl->read( (FLMUINT)0, sizeof( SFLM_DB_HDR),
									pDbHdr, &uiBytesRead)))
	{
		if (rc != NE_FLM_IO_END_OF_FILE)
		{
			if (pDbStats)
			{
				pDbStats->uiReadErrors++;
			}
		}
		else
		{
			if (pui32CalcCRC)
			{
				*pui32CalcCRC = calcDbHdrCRC( pDbHdr);
			}

			// Get what we can out of the header.

			if (hdrIsNonNativeFormat( pDbHdr))
			{
				convertDbHdr( pDbHdr);
			}
		}
		goto Exit;
	}

	if (pui32CalcCRC)
	{
		*pui32CalcCRC = calcDbHdrCRC( pDbHdr);
	}
	if (uiBytesRead < sizeof( SFLM_DB_HDR))
	{

		// Still get what we can out of the header.

		if (hdrIsNonNativeFormat( pDbHdr))
		{
			convertDbHdr( pDbHdr);
		}
		rc = RC_SET( NE_SFLM_NOT_FLAIM);
		goto Exit;
	}
	else
	{

		// This routine will convert to native format if it is not
		// in native format.

		if (RC_BAD( rc = verifyDbHdr( pDbHdr)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}
