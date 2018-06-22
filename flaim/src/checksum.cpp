//-------------------------------------------------------------------------
// Desc:	Calculate block checksum
// Tabs:	3
//
// Copyright (c) 1999-2007 Novell, Inc. All Rights Reserved.
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
Desc: Compares or sets the checksum value in a block.  
		Operates to compare the block checksum with the actual checksum.
Ret:	if (Compare) returns FERR_BLOCK_CHECKSUM block checksum does
		not agree with checksum header values.  
*********************************************************************/
RCODE BlkCheckSum(
	FLMBYTE *	pucBlkPtr,
	FLMINT		iCompare,
	FLMUINT		uiBlkAddress,
	FLMUINT		uiBlkSize)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiCurrChecksum = 0;
	FLMUINT			uiNewChecksum;
	FLMUINT			uiEncryptSize;
	FLMUINT			uiAdds = 0;
	FLMUINT			uiXORs = 0;
	FLMBYTE *		pucSaveBlkPtr = pucBlkPtr;

	// Check the block length against the maximum block size

	uiEncryptSize = (FLMUINT)getEncryptSize( pucBlkPtr);
	if( uiEncryptSize > uiBlkSize || uiEncryptSize < BH_OVHD)
	{
		rc = RC_SET( FERR_BLOCK_CHECKSUM);
		goto Exit;
	}	

	// If we are comparing, but there is no current checksum just return.
	// The next time the checksum is modified, the comparison will be performed. 
	// Version 3.x will store the full block address or if
	// a checksum is used, the lost low byte of block address is checksummed.

	if( iCompare == CHECKSUM_CHECK)
	{
		uiCurrChecksum = (FLMUINT)(((FLMUINT)pucBlkPtr[ BH_CHECKSUM_HIGH] << 8) + 
											 (FLMUINT)pucBlkPtr[ BH_CHECKSUM_LOW]);
	}

	// We need to checksum the data that is encrypted. 
	// This is done by the getEncryptSize() call.

	// Check all of block, except for embedded checksum bytes.
	// For speed, the initial values of uiAdds and uiXORs effectively ignore/skip
	// the checksum values already embedded in the source: (a - a) == 0 and 
	// (a ^ a) == 0 so the initial values, net of the 2nd operations, equal zero
	// too.  

	uiAdds = 0 - (pucBlkPtr[ BH_CHECKSUM_LOW] + pucBlkPtr[ BH_CHECKSUM_HIGH]);
	uiXORs = pucBlkPtr[ BH_CHECKSUM_LOW] ^ pucBlkPtr[ BH_CHECKSUM_HIGH];

	// The 3.x version checksums the low byte of the address.

	if( uiBlkAddress != BT_END)
	{
		uiAdds += (FLMBYTE)uiBlkAddress;
		uiXORs ^= (FLMBYTE)uiBlkAddress;
	}
	
	f_calcFastChecksum( pucBlkPtr, uiEncryptSize, &uiAdds, &uiXORs); 
	uiNewChecksum = (((uiAdds << 8) + uiXORs) & 0xFFFF);
	
	// Set the checksum
	
	if (iCompare == CHECKSUM_SET)
	{
		pucSaveBlkPtr[ BH_CHECKSUM_HIGH] = (FLMBYTE)(uiNewChecksum >> 8);
		pucSaveBlkPtr[ BH_CHECKSUM_LOW] = (FLMBYTE)uiNewChecksum;
		goto Exit;
	}

	// The checksum is different from the stored checksum.
	// For version 3.x database we don't store the low byte of the
	// address.  Thus, it will have to be computed from the checksum.

	if( uiBlkAddress == BT_END)
	{
		FLMBYTE		byXor;
		FLMBYTE		byAdd;
		FLMBYTE		byDelta;
		
		// If there is a one byte value that will satisfy both
		// sides of the checksum, the checksum is OK and that value
		// is the first byte value.
		
		byXor = (FLMBYTE) uiNewChecksum;
		byAdd = (FLMBYTE) (uiNewChecksum >> 8);
		byDelta = byXor ^ pucSaveBlkPtr [BH_CHECKSUM_LOW];
		
		// Here is the big check, if byDelta is also what is
		// off with the add portion of the checksum, we have
		// a good value.
		
		if( ((FLMBYTE) (byAdd + byDelta)) == pucSaveBlkPtr[ BH_CHECKSUM_HIGH] )
		{
			// Set the low checksum value with the computed value.
			
			pucSaveBlkPtr[ BH_CHECKSUM_LOW] = byDelta;
			goto Exit;
		}
	}
	else
	{
		// This has the side effect of setting the low block address byte
		// in the block thus getting rid of the low checksum byte.

		// NOTE: We are allowing the case where the calculated checksum is
		// zero and the stored checksum is one because we used to change
		// a calculated zero to a one in old databases and store the one.
		// This is probably a somewhat rare case (1 out of 65536 checksums
		// will be zero), so forgiving it will be OK most of the time.
		// So that those don't cause us to report block checksum errors,
		// we just allow it - checksumming isn't a perfect check anyway.
		
		if (uiNewChecksum == uiCurrChecksum ||
			 ((!uiNewChecksum) && (uiCurrChecksum == 1)))
		{
			pucSaveBlkPtr [BH_CHECKSUM_LOW] = (FLMBYTE) uiBlkAddress;
			goto Exit;
		}
	}
	
	// Otherwise, we have a checksum error.
	
	rc = RC_SET( FERR_BLOCK_CHECKSUM);

Exit:

	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
FLMUINT lgHdrCheckSum(
	FLMBYTE *	pucLogHdr,
	FLMBOOL		bCompare)
{
	FLMUINT	uiCnt;
	FLMUINT	uiTempSum;
	FLMUINT	uiCurrChecksum;
	FLMUINT	uiTempSum2;
	FLMUINT	uiBytesToChecksum;

	uiBytesToChecksum = (FB2UW( &pucLogHdr [LOG_FLAIM_VERSION]) < 
									FLM_FILE_FORMAT_VER_4_3)
										? LOG_HEADER_SIZE_VER40
										: LOG_HEADER_SIZE;

	// If we are comparing, but there is no current checksum, return
	// zero to indicate success.  The next time the checksum is
	// modified, the comparison will be performed.
	//
	// Unconverted databases may have a 0xFFFF or a zero in the checksum
	// If 0xFFFF, change to a zero so we only have to deal with one value.

	if( (uiCurrChecksum = (FLMUINT)FB2UW( 
				&pucLogHdr[ LOG_HDR_CHECKSUM])) == 0xFFFF)
	{
		uiCurrChecksum = 0;
	}

	if( bCompare && !uiCurrChecksum)
	{
		return( 0);
	}

	// Check all of log header except for the bytes which contain the
	// checksum.
	//
	// For speed, uiTempSum is initialized to effectively ignore or skip
	// the checksum embedded in the source:  (a - a) == 0 so we store a negative
	// that the later addition clears out.  Also, the loop counter, i,
	// is 1 larger than the number of FLMUINT16's so that we can
	// pre-decrement by "for(;--i != 0;)" -- basically "loop-non-zero".
	
	for( uiTempSum = 0 - (FLMUINT)FB2UW( &pucLogHdr[ LOG_HDR_CHECKSUM]),
		  uiCnt = 1 + uiBytesToChecksum / sizeof( FLMUINT16);	--uiCnt != 0; )
	{
		uiTempSum += (FLMUINT)FB2UW( pucLogHdr);
		pucLogHdr += sizeof( FLMUINT16);
	}

	// Don't want a zero or 0xFFFF checksum - change to 1

	if( (0 == (uiTempSum2 = (uiTempSum & 0xFFFF))) || (uiTempSum2 == 0xFFFF))
	{
		uiTempSum2 = 1;
	}

	return( (FLMUINT)(((bCompare) && (uiTempSum2 == uiCurrChecksum))
							? (FLMUINT)0
							: uiTempSum2) );
}
