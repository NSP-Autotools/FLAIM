//------------------------------------------------------------------------------
// Desc:	Routines that do conversions between internal number and numeric
//			key format to platform number types.
// Tabs:	3
//
// Copyright (c) 2002-2007 Novell, Inc. All Rights Reserved.
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

#define DEFINE_NUMBER_MAXIMUMS
#include "flaimsys.h"

/****************************************************************************
Desc:		Converts a UINT to its storage value
*****************************************************************************/
RCODE FlmUINT2Storage(
	FLMUINT			uiNum,
	FLMUINT *		puiBufLen,	// In (buffer size) / Out (bytes used)
	FLMBYTE *		pucBuf)
{
	RCODE			rc = NE_XFLM_OK;

	if( RC_BAD( rc = flmNumber64ToStorage( uiNum, puiBufLen,
		pucBuf, FALSE, FALSE)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Converts an INT to its storage value
*****************************************************************************/
RCODE FlmINT2Storage(
	FLMINT			iNum,
	FLMUINT *		puiBufLen,	// In (buffer size) / Out (bytes used)
	FLMBYTE *		pucBuf)
{
	FLMBOOL		bNeg = FALSE;
	RCODE			rc = NE_XFLM_OK;

	if( iNum < 0)
	{
		iNum = -iNum;
		bNeg = TRUE;
	}

	if( RC_BAD( rc = flmNumber64ToStorage( (FLMUINT64)iNum,
		puiBufLen, pucBuf, bNeg, FALSE)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Converts a number to its storage or collation format
Notes:	Changes to this code must also be made to flmUINT32ToStorage
*****************************************************************************/
RCODE flmNumber64ToStorage(
	FLMUINT64		ui64Num,
	FLMUINT *		puiBufLen,
	FLMBYTE *		pucBuf,
	FLMBOOL			bNegative,
	FLMBOOL			bCollation)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiByteCount = 0;
	FLMUINT32		ui32Low = (FLMUINT32)ui64Num;
	FLMUINT32		ui32High = (FLMUINT32)(ui64Num >> 32);

	if( *puiBufLen < FLM_MAX_NUM_BUF_SIZE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}

	// Negative flag should not be set if the number is 0

	if( !ui64Num && bNegative)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

	// Build the storage format, either for an index key (if bCollation
	// is TRUE) or for a data element.

	if( !bCollation)
	{
		// Build the data storage representation of the number
		// Numbers are stored in a little-endian format

		do
		{
			*pucBuf++ = (FLMBYTE)ui32Low;
			ui32Low >>= 8;
			uiByteCount++;
		} while (ui32Low);

		if( ui32High)
		{
			for( ; uiByteCount < 4; uiByteCount++)
			{
				*pucBuf++ = 0;
			}

			for( ; ui32High; uiByteCount++)
			{
				*pucBuf++ = (FLMBYTE)ui32High;
				ui32High >>= 8;
			}
		}

		// If a number is negative, the high bit of the right-most
		// byte needs to be set.  If the value of the number is such
		// that its positive representation already has the high-bit
		// set, we need to store an additional sign byte.

		if( !bNegative)
		{
			if( *(pucBuf - 1) & 0x80)
			{
				*pucBuf++ = 0;
				uiByteCount++;
			}
		}
		else
		{
			if( (*(pucBuf - 1) & 0x80) == 0)
			{
				*(pucBuf - 1) |= 0x80;
			}
			else
			{
				*pucBuf++ = 0x80;
				uiByteCount++;
			}
		}
	}
	else
	{
		FLMBYTE *	pucStart = pucBuf++;

		if( ui32High)
		{
			if( ui32High & 0xFF000000)
			{
				*pucBuf++ = (FLMBYTE)(ui32High >> 24);
				*pucBuf++ = (FLMBYTE)(ui32High >> 16);
				*pucBuf++ = (FLMBYTE)(ui32High >> 8);
				*pucBuf++ = (FLMBYTE)ui32High;
				uiByteCount += 4;
			}
			else if( ui32High & 0x00FF0000)
			{
				*pucBuf++ = (FLMBYTE)(ui32High >> 16);
				*pucBuf++ = (FLMBYTE)(ui32High >> 8);
				*pucBuf++ = (FLMBYTE)ui32High;
				uiByteCount += 3;
			}
			else if( ui32High & 0x0000FF00)
			{
				*pucBuf++ = (FLMBYTE)(ui32High >> 8);
				*pucBuf++ = (FLMBYTE)ui32High;
				uiByteCount += 2;
			}
			else if( ui32High)
			{
				*pucBuf++ = (FLMBYTE)ui32High;
				uiByteCount++;
			}
		}

		if( ui32Low)
		{
			if( ui32Low & 0xFF000000)
			{
				*pucBuf++ = (FLMBYTE)(ui32Low >> 24);
				*pucBuf++ = (FLMBYTE)(ui32Low >> 16);
				*pucBuf++ = (FLMBYTE)(ui32Low >> 8);
				*pucBuf++ = (FLMBYTE)ui32Low;
				uiByteCount += 4;
			}
			else if( ui32Low & 0x00FF0000)
			{
				*pucBuf++ = (FLMBYTE)(ui32Low >> 16);
				*pucBuf++ = (FLMBYTE)(ui32Low >> 8);
				*pucBuf++ = (FLMBYTE)ui32Low;
				uiByteCount += 3;
			}
			else if( ui32Low & 0x0000FF00)
			{
				*pucBuf++ = (FLMBYTE)(ui32Low >> 8);
				*pucBuf++ = (FLMBYTE)ui32Low;
				uiByteCount += 2;
			}
			else if( ui32Low)
			{
				*pucBuf++ = (FLMBYTE)ui32Low;
				uiByteCount++;
			}
		}
		else if( !ui32High)
		{
			*pucBuf++ = 0;
			uiByteCount++;
		}

		if( !bNegative)
		{
			// Positive numbers must collate after negative numbers,
			// so all positive numbers will start with a byte
			// in the range of 0xC8 - 0xCF.

			*pucStart = (FLMBYTE)(0xC8 + (uiByteCount - 1));
			uiByteCount++;
		}
		else
		{
			FLMBYTE *	pucTmp = pucStart + 1;

			while( pucTmp < pucBuf)
			{
				*pucTmp = ~(*pucTmp);
				pucTmp++;
			}

			// Negative numbers must collate before positive numbers,
			// so all negative numbers will start with a byte
			// in the range of 0xC0 - 0xC7.

			*pucStart = (FLMBYTE)(0xC8 - uiByteCount);
			uiByteCount++;
		}
	}

	// Set the number of bytes in the buffer before returning.

	*puiBufLen = uiByteCount;

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Converts a storage value back into a number
Notes:	Changes to this code must also be made to flmStorage2Number64
*****************************************************************************/
RCODE flmStorage2Number(
	FLMUINT				uiType,
	FLMUINT				uiBufLen,
	const FLMBYTE *	pucBuf,
	FLMUINT *			puiNum,
	FLMINT *				piNum)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiLoop;
	FLMUINT		uiNum = 0;
	FLMBOOL		bNeg = FALSE;

	if( !uiBufLen)
	{
		if( puiNum)
		{
			*puiNum = 0;
		}
		else
		{
			*piNum = 0;
		}
		goto Exit;
	}

	if( !pucBuf)
	{
		rc = RC_SET( NE_XFLM_CONV_NULL_SRC);
		goto Exit;
	}

	switch( uiType)
	{
		case XFLM_NUMBER_TYPE :
		{
			// Make sure the number buffer does not exceed the
			// max length.  If there is an extra byte for the
			// sign (byte 9) make sure it has a value of either
			// 0x80 or 0 (by masking of the high bit).

			if( uiBufLen > FLM_MAX_NUM_BUF_SIZE ||
				(uiBufLen == FLM_MAX_NUM_BUF_SIZE &&
					(pucBuf[ uiBufLen - 1] & 0x7F) != 0))
			{
				rc = RC_SET( NE_XFLM_CONV_NUM_OVERFLOW);
				goto Exit;
			}

			// Look at the high bit of the most-significant byte
			// to determine if the number is signed

			if( pucBuf[ uiBufLen - 1] & 0x80)
			{
				if( puiNum)
				{
					rc = RC_SET( NE_XFLM_CONV_NUM_UNDERFLOW);
					goto Exit;
				}
				bNeg = TRUE;
			}

			uiNum = pucBuf[ uiBufLen - 1] & 0x7F;
			uiBufLen--;

			for( uiLoop = 1; uiLoop <= uiBufLen; uiLoop++)
			{
				if( gv_b32BitPlatform && (uiNum & 0xFF000000))
				{
					rc = RC_SET( NE_XFLM_CONV_NUM_OVERFLOW);
					goto Exit;
				}
				uiNum = (uiNum << 8) + pucBuf[ uiBufLen - uiLoop];
			}

			break;
		}

		case XFLM_TEXT_TYPE:
		{
			FLMBYTE		ucNumBuf[ 64];
			FLMUINT		uiNumBufLen = sizeof( ucNumBuf);
			FLMBYTE *	pucTmp;

			if( RC_BAD( rc = flmStorage2UTF8( XFLM_TEXT_TYPE, uiBufLen, pucBuf,
				&uiNumBufLen, ucNumBuf)))
			{
				goto Exit;
			}

			pucTmp = &ucNumBuf[ 0];
			if( *pucTmp == ASCII_DASH)
			{
				if( puiNum)
				{
					rc = RC_SET( NE_XFLM_CONV_NUM_UNDERFLOW);
					goto Exit;
				}
				bNeg = TRUE;
				pucTmp++;
			}

			while( *pucTmp)
			{
				if( *pucTmp < ASCII_ZERO || *pucTmp > ASCII_NINE)
				{
					break;
				}

				if( uiNum > (~(FLMUINT)0) / 10)
				{
					rc = RC_SET( NE_XFLM_CONV_NUM_OVERFLOW);
					goto Exit;
				}

				uiNum *= (FLMUINT)10;

				if( uiNum > (~(FLMUINT)0) - (FLMUINT)(*pucTmp - ASCII_ZERO))
				{
					rc = RC_SET( NE_XFLM_CONV_NUM_OVERFLOW);
					goto Exit;
				}

				uiNum += (FLMUINT)(*pucTmp - ASCII_ZERO);
				pucTmp++;
			}

			break;
		}

		default :
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_ILLEGAL);
			goto Exit;
		}
	}

	if( puiNum)
	{
		if( bNeg)
		{
			rc = RC_SET( NE_XFLM_CONV_NUM_UNDERFLOW);
			goto Exit;
		}

		*puiNum = uiNum;
	}
	else
	{
		flmAssert( piNum);

		if( bNeg)
		{
			if( uiNum > gv_uiMaxSignedIntVal + 1)
			{
				rc = RC_SET( NE_XFLM_CONV_NUM_UNDERFLOW);
				goto Exit;
			}

			*piNum = -(FLMINT)uiNum;
		}
		else
		{
			if( uiNum > gv_uiMaxSignedIntVal)
			{
				rc = RC_SET( NE_XFLM_CONV_NUM_OVERFLOW);
				goto Exit;
			}

			*piNum = (FLMINT)uiNum;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Converts a storage value back into a number
Notes:	Changes to this code must also be made to flmStorage2Number
*****************************************************************************/
RCODE flmStorage2Number64(
	FLMUINT				uiType,
	FLMUINT				uiBufLen,
	const FLMBYTE *	pucBuf,
	FLMUINT64 *			pui64Num,
	FLMINT64 *			pi64Num)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiLoop;
	FLMUINT64	ui64Num = 0;
	FLMBOOL		bNeg = FALSE;

	if( !uiBufLen)
	{
		if( pui64Num)
		{
			*pui64Num = 0;
		}
		else
		{
			*pi64Num = 0;
		}
		goto Exit;
	}

	if( !pucBuf)
	{
		rc = RC_SET( NE_XFLM_CONV_NULL_SRC);
		goto Exit;
	}

	switch( uiType)
	{
		case XFLM_NUMBER_TYPE :
		{
			// Make sure the number buffer does not exceed the
			// max length.  If there is an extra byte for the
			// sign (byte 9) make sure it has a value of either
			// 0x80 or 0 (by masking of the high bit).

			if( uiBufLen > FLM_MAX_NUM_BUF_SIZE ||
				(uiBufLen == FLM_MAX_NUM_BUF_SIZE &&
					(pucBuf[ uiBufLen - 1] & 0x7F) != 0))
			{
				rc = RC_SET( NE_XFLM_CONV_NUM_OVERFLOW);
				goto Exit;
			}

			// Look at the high bit of the most-significant byte
			// to determine if the number is signed

			if( pucBuf[ uiBufLen - 1] & 0x80)
			{
				if( pui64Num)
				{
					rc = RC_SET( NE_XFLM_CONV_NUM_UNDERFLOW);
					goto Exit;
				}
				bNeg = TRUE;
			}

			ui64Num = pucBuf[ uiBufLen - 1] & 0x7F;
			uiBufLen--;

			for( uiLoop = 1; uiLoop <= uiBufLen; uiLoop++)
			{
				ui64Num = (ui64Num << 8) + pucBuf[ uiBufLen - uiLoop];
			}

			break;
		}

		case XFLM_TEXT_TYPE :
		{
			FLMBYTE		ucNumBuf[ 64];
			FLMUINT		uiNumBufLen = sizeof( ucNumBuf);
			FLMBYTE *	pucTmp;

			if( RC_BAD( rc = flmStorage2UTF8( XFLM_TEXT_TYPE, uiBufLen, pucBuf,
				&uiNumBufLen, ucNumBuf)))
			{
				goto Exit;
			}

			pucTmp = &ucNumBuf[ 0];

			if( *pucTmp == ASCII_DASH)
			{
				if( pui64Num)
				{
					rc = RC_SET( NE_XFLM_CONV_NUM_UNDERFLOW);
					goto Exit;
				}
				bNeg = TRUE;
				pucTmp++;
			}

			while( *pucTmp)
			{
				if( *pucTmp < ASCII_ZERO || *pucTmp > ASCII_NINE)
				{
					break;
				}

				if( ui64Num > (~(FLMUINT64)0) / 10)
				{
					rc = RC_SET( NE_XFLM_CONV_NUM_OVERFLOW);
					goto Exit;
				}

				ui64Num *= (FLMUINT64)10;

				if( ui64Num > (~(FLMUINT64)0) - (FLMUINT64)(*pucTmp - ASCII_ZERO))
				{
					rc = RC_SET( NE_XFLM_CONV_NUM_OVERFLOW);
					goto Exit;
				}

				ui64Num += (FLMUINT64)(*pucTmp - ASCII_ZERO);
				pucTmp++;
			}

			break;
		}

		default :
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_ILLEGAL);
			goto Exit;
		}
	}

	if( pui64Num)
	{
		*pui64Num = ui64Num;
	}
	else
	{
		flmAssert( pi64Num);

		if( bNeg)
		{
			if( ui64Num > gv_ui64MaxSignedIntVal + 1)
			{
				rc = RC_SET( NE_XFLM_CONV_NUM_UNDERFLOW);
				goto Exit;
			}

			*pi64Num = -(FLMINT64)ui64Num;
		}
		else
		{
			if( ui64Num > gv_ui64MaxSignedIntVal)
			{
				rc = RC_SET( NE_XFLM_CONV_NUM_OVERFLOW);
				goto Exit;
			}

			*pi64Num = (FLMINT64)ui64Num;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Converts a numeric storage value into a collation value
*****************************************************************************/
RCODE flmStorageNum2CollationNum(
	const FLMBYTE *	pucStorageBuf,
	FLMUINT				uiStorageLen,
	FLMBYTE *			pucCollBuf,
	FLMUINT *			puiCollLen)
{
	FLMUINT		uiLoop;
	FLMUINT		uiOffset;
	FLMUINT		uiMaxLen = *puiCollLen;
	FLMBYTE		ucVal;
	FLMBOOL		bNegative = FALSE;
	RCODE			rc = NE_XFLM_OK;

	if( !pucStorageBuf || !uiStorageLen)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_INVALID_PARM);
		goto Exit;
	}

	// Look at the high bit of the most-significant byte
	// to determine if the number is signed

	if( pucStorageBuf[ uiStorageLen - 1] & 0x80)
	{
		bNegative = TRUE;
	}

	uiOffset = 1;
	if( (ucVal = pucStorageBuf[ uiStorageLen - 1] & 0x7F) != 0 ||
		uiStorageLen == 1) // Handle the special case of zero
	{
		if( bNegative)
		{
			ucVal = ~ucVal;
		}

		if( uiOffset >= uiMaxLen)
		{
			rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
			goto Exit;
		}

		pucCollBuf[ uiOffset++] = ucVal;
	}
	uiStorageLen--;

	// Check for overflow

	if( uiOffset + uiStorageLen >= uiMaxLen)
	{
		rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}

	// Map the little-endian storage format to the big-endian
	// collation format

	if( !bNegative)
	{
		for( uiLoop = 1; uiLoop <= uiStorageLen; uiLoop++)
		{
			pucCollBuf[ uiOffset++] = pucStorageBuf[ uiStorageLen - uiLoop];
		}
	}
	else
	{
		for( uiLoop = 1; uiLoop <= uiStorageLen; uiLoop++)
		{
			pucCollBuf[ uiOffset++] = ~pucStorageBuf[ uiStorageLen - uiLoop];
		}
	}

	flmAssert( uiOffset >= 2);

	// Store the numeric collation marker and byte count

	if( !bNegative)
	{
		// Positive numbers must collate after negative numbers,
		// so all positive numbers will start with a byte
		// in the range of 0xC8 - 0xCF.

		pucCollBuf[ 0] = (FLMBYTE)(0xC8 + (uiOffset - 2));
	}
	else
	{
		// Negative numbers must collate before positive numbers,
		// so all negative numbers will start with a byte
		// in the range of 0xC0 - 0xC7.

		pucCollBuf[ 0] = (FLMBYTE)(0xC8 - (uiOffset - 1));
	}

	// Set the key length

	*puiCollLen = uiOffset;

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Converts a numeric collation value into a storage value
*****************************************************************************/
RCODE flmCollationNum2StorageNum(
	const FLMBYTE *	pucCollBuf,
	FLMUINT				uiCollLen,
	FLMBYTE *			pucStorageBuf,
	FLMUINT *			puiStorageLen)
{
	FLMUINT		uiLoop;
	FLMUINT		uiOffset;
	FLMUINT		uiMaxOffset = *puiStorageLen;
	FLMUINT		uiNumKeyBytes;
	FLMBOOL		bNegative = FALSE;
	RCODE			rc = NE_XFLM_OK;

	if( !pucCollBuf || !uiCollLen)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_INVALID_PARM);
		goto Exit;
	}

	// Make sure this looks like a valid numeric key piece

	if( (pucCollBuf[ 0] & 0xC0) != 0xC0)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
		goto Exit;
	}

	// Get the byte count

	if( (uiNumKeyBytes = (FLMUINT)(pucCollBuf[ 0] & 0x0F)) >= 8)
	{
		uiNumKeyBytes -= 7;
	}
	else
	{
		uiNumKeyBytes = 8 - uiNumKeyBytes;
		bNegative = TRUE;
	}
	pucCollBuf++;
	uiCollLen--;

	// Make sure the buffer has at least the number of bytes
	// we need

	if( uiCollLen != uiNumKeyBytes)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
		goto Exit;
	}

	if( uiNumKeyBytes >= uiMaxOffset)
	{
		rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}

	// Translate the collation value into a numeric value

	if( !bNegative)
	{
		for( uiLoop = 0; uiLoop < uiNumKeyBytes; uiLoop++)
		{
			pucStorageBuf[ uiNumKeyBytes - uiLoop - 1] = pucCollBuf[ uiLoop];
		}
	}
	else
	{
		for( uiLoop = 0; uiLoop < uiNumKeyBytes; uiLoop++)
		{
			pucStorageBuf[ uiNumKeyBytes - uiLoop - 1] = ~pucCollBuf[ uiLoop];
		}
	}

	uiOffset = uiNumKeyBytes;

	// If a number is negative, the high bit of the right-most
	// byte needs to be set.  If the value of the number is such
	// that its positive representation already has the high-bit
	// set, we need to store an additional sign byte.

	if( !bNegative)
	{
		if( pucStorageBuf[ uiOffset - 1] & 0x80)
		{
			if( uiOffset >= uiMaxOffset)
			{
				rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
				goto Exit;
			}

			pucStorageBuf[ uiOffset++] = 0;
		}
	}
	else
	{
		if( (pucStorageBuf[ uiOffset - 1] & 0x80) == 0)
		{
			pucStorageBuf[ uiOffset - 1] |= 0x80;
		}
		else
		{
			if( uiOffset >= uiMaxOffset)
			{
				rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
				goto Exit;
			}

			pucStorageBuf[ uiOffset++] = 0x80;
		}
	}

	// Set the storage length

	*puiStorageLen = uiOffset;

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Converts a collation value (numeric only) to a number
*****************************************************************************/
RCODE flmCollation2Number(
	FLMUINT					uiBufLen,
	const FLMBYTE *		pucBuf,
	FLMUINT64 *				pui64Num,
	FLMBOOL *				pbNeg,
	FLMUINT *				puiBytesProcessed)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiLoop;
	FLMUINT		uiNumBytes;
	FLMUINT64	ui64Num;
	FLMBOOL		bNeg = FALSE;

	*pui64Num = 0;

	if( !uiBufLen)
	{
		goto Exit;
	}

	if( !pucBuf)
	{
		rc = RC_SET( NE_XFLM_CONV_NULL_SRC);
		goto Exit;
	}

	// Make sure this looks like a valid numeric key piece

	if( (pucBuf[ 0] & 0xC0) != 0xC0)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
		goto Exit;
	}

	// Get the byte count

	if( (uiNumBytes = (FLMUINT)(pucBuf[ 0] & 0x0F)) >= 8)
	{
		uiNumBytes -= 7;
	}
	else
	{
		uiNumBytes = 8 - uiNumBytes;
		bNeg = TRUE;
	}
	pucBuf++;
	uiBufLen--;

	// Make sure the buffer has at least the number of bytes
	// we need

	if( uiBufLen < uiNumBytes)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
		goto Exit;
	}

	// Reconstruct the number

	ui64Num = 0;
	if( !bNeg)
	{
		for( uiLoop = 0; uiLoop < uiNumBytes; uiLoop++)
		{
			ui64Num += (((FLMUINT64)pucBuf[ uiLoop]) <<
				(8 * ((uiNumBytes - uiLoop) - 1)));
		}
	}
	else
	{
		for( uiLoop = 0; uiLoop < uiNumBytes; uiLoop++)
		{
			ui64Num += (((FLMUINT64)((FLMBYTE)~pucBuf[ uiLoop])) <<
				(8 * ((uiNumBytes - uiLoop) - 1)));
		}
	}

	*pui64Num = ui64Num;

	if( puiBytesProcessed)
	{
		*puiBytesProcessed = uiNumBytes + 1;
	}

	if( pbNeg)
	{
		*pbNeg = bNeg;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE flmStorageNumberToNumber(
	const FLMBYTE *	pucNumBuf,
	FLMUINT				uiNumBufLen,
	FLMUINT64 *			pui64Number,
	FLMBOOL *			pbNeg)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiLoop;
	FLMUINT64	ui64Num = 0;
	FLMBOOL		bNeg = FALSE;

	if( !uiNumBufLen)
	{
		goto Exit;
	}

	// Make sure the number buffer does not exceed the
	// max length.  If there is an extra byte for the
	// sign (byte 9) make sure it has a value of either
	// 0x80 or 0 (by masking of the high bit).

	if( uiNumBufLen > FLM_MAX_NUM_BUF_SIZE ||
		(uiNumBufLen == FLM_MAX_NUM_BUF_SIZE &&
			(pucNumBuf[ uiNumBufLen - 1] & 0x7F) != 0))
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_NUM_OVERFLOW);
		goto Exit;
	}

	// Look at the high bit of the most-significant byte
	// to determine if the number is signed

	if( pucNumBuf[ uiNumBufLen - 1] & 0x80)
	{
		bNeg = TRUE;
	}

	ui64Num = pucNumBuf[ uiNumBufLen - 1] & 0x7F;
	uiNumBufLen--;

	for( uiLoop = 1; uiLoop <= uiNumBufLen; uiLoop++)
	{
		ui64Num = (ui64Num << 8) + pucNumBuf[ uiNumBufLen - uiLoop];
	}

Exit:

	*pui64Number = ui64Num;
	*pbNeg = bNeg;

	return( rc);
}
