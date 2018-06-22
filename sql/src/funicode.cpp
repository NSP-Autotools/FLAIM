//------------------------------------------------------------------------------
// Desc:	This file contains the Unicode conversion routines
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

//#define DEF_FLM_UNI_GLOBALS
#include "flaimsys.h"

/****************************************************************************
Desc:		Encode a string into FLAIM's internal text format
			(SEN-prefixed, null-terminated UTF8).  The string is prefixed with
			a SEN that indicates the number of characters, not including the
			terminating NULL.

			This routine can be called with a NULL input buffer.  If it is
			called in this way, the length of the encoded string will be
			returned in *puiBufLength, but no encoding will actually be
			performed.
****************************************************************************/
RCODE	flmUnicode2Storage(
	const FLMUNICODE *	puzStr,			// UNICODE string to encode
	FLMUINT					uiStrLen,		// 0 = Unknown
	FLMBYTE *				pucBuf,			// Destination buffer
	FLMUINT *				puiBufLength,	// [IN ] Size of pucBuf,
													// [OUT] Amount of pucBuf used
	FLMUINT *				puiCharCount)
{
	RCODE						rc = NE_SFLM_OK;
	FLMBYTE *				pucEncPtr;
	const FLMUNICODE *	puzPtr = NULL;
	FLMUINT					uiMaxLen;
	FLMUINT					uiCharsEncoded = 0;
	FLMUINT					uiEncodedLen = 0;
	FLMUINT					uiTmp;
	FLMUNICODE				uChar;
	FLMBYTE					ucTmpSen[ 5];
	FLMBYTE *				pucTmpSen = &ucTmpSen[ 0];

	if( !pucBuf)
	{
		uiMaxLen = (~(FLMUINT)0);
	}
	else
	{
		uiMaxLen = *puiBufLength;
	}

	// If uiStrLen is 0, determine the number of characters.

	if( !uiStrLen)
	{
		uiStrLen = f_unilen( puzStr);
	}
	else if( puzStr[ uiStrLen] != 0)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_PARM);
		goto Exit;
	}

	if( puiCharCount)
	{
		*puiCharCount = uiStrLen;
	}

	if( !uiStrLen)
	{
		// Nothing to encode

		*puiBufLength = 0;
		goto Exit;
	}

	pucEncPtr = pucBuf;

	// Encode the number of characters as a SEN.  If pucEncPtr is
	// NULL, the caller is only interested in the length of the encoded
	// string, so a temporary buffer is used to call f_encodeSEN.

	uiTmp = f_encodeSEN( uiStrLen, &pucTmpSen);

	if( pucEncPtr)
	{
		if( (uiEncodedLen + uiTmp) >= uiMaxLen)
		{
			rc = RC_SET( NE_SFLM_CONV_DEST_OVERFLOW);
			goto Exit;
		}
		
		if( uiTmp == 1)
		{
			*pucEncPtr++ = ucTmpSen[ 0];
		}
		else
		{
			f_memcpy( pucEncPtr, &ucTmpSen[ 0], uiTmp);
			pucEncPtr += uiTmp;
		}
	}
	uiEncodedLen += uiTmp;

	// Encode the string using UTF-8

	puzPtr = puzStr;
	if( uiStrLen)
	{
		while( (uChar = *puzPtr) != 0)
		{
			if( (uiTmp = uiMaxLen - uiEncodedLen) == 0)
			{
				rc = RC_SET( NE_SFLM_CONV_DEST_OVERFLOW);
				goto Exit;
			}

			if( uChar <= 127)
			{
				if( pucEncPtr)
				{
					*pucEncPtr++ = (FLMBYTE)uChar;
				}

				uiEncodedLen++;
			}
			else
			{
				if( RC_BAD( rc = f_uni2UTF8( uChar, pucEncPtr, &uiTmp)))
				{
					goto Exit;
				}

				if( pucEncPtr)
				{
					pucEncPtr += uiTmp;
				}

				uiEncodedLen += uiTmp;
			}

			puzPtr++;
			uiCharsEncoded++;
		}

		// Make sure the string length (which may have been provided by
		// the caller) was correct.

		if( uiCharsEncoded != uiStrLen)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_PARM);
			goto Exit;
		}
	}

	// Terminate the string with a 0 byte

	if( (uiMaxLen - uiEncodedLen) < 1)
	{
		rc = RC_SET( NE_SFLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}

	if( pucEncPtr)
	{
		*pucEncPtr++ = 0;
	}
	uiEncodedLen++;

	// Return the length of the encoded string

	*puiBufLength = uiEncodedLen;

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Encode a string into FLAIM's internal text format
			(SEN-prefixed, null-terminated UTF8).  The string is prefixed with
			a SEN that indicates the number of characters, not including the
			terminating NULL.

			This routine can be called with a NULL input buffer.  If it is
			called in this way, the length of the encoded string will be
			returned in *puiBufLength, but no encoding will actually be
			performed.
****************************************************************************/
RCODE	flmNative2Storage(
	char *			pszStr,			// Native string to encode
	FLMUINT			uiStrLen,		// 0 = Unknown
	FLMBYTE *		pucBuf,			// Destination buffer
	FLMUINT *		puiBufLength,	// [IN ] Size of pucBuf,
											// [OUT] Amount of pucBuf used
	FLMUINT *		puiCharCount)
{
	FLMBYTE *		pucEncPtr;
	char *			pszPtr = NULL;
	FLMUINT			uiMaxLen;
	FLMUINT			uiCharsEncoded = 0;
	FLMUINT			uiEncodedLen = 0;
	FLMUINT			uiTmp;
	FLMBYTE			ucTmpSen[ 5];
	FLMBYTE *		pucTmpSen = &ucTmpSen[ 0];
	FLMBOOL			bDirectCopy = FALSE;
	RCODE				rc = NE_SFLM_OK;

	if( !pucBuf)
	{
		uiMaxLen = (~(FLMUINT)0);
	}
	else
	{
		uiMaxLen = *puiBufLength;
	}

	// If uiStrLen is 0, determine the number of characters.

	if( !uiStrLen)
	{
#ifdef FLM_ASCII_PLATFORM
		bDirectCopy = TRUE;
		for( pszPtr = pszStr; *pszPtr; pszPtr++, uiStrLen++)
		{
			if( (FLMBYTE)*pszPtr > 0x7F)
			{
				bDirectCopy = FALSE;
			}
		}
#else
		uiStrLen = f_strlen( pszStr);
#endif
	}
	else if( pszStr[ uiStrLen] != 0)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_PARM);
		goto Exit;
	}

	if( puiCharCount)
	{
		*puiCharCount = uiStrLen;
	}

	if( !uiStrLen)
	{
		// Nothing to encode

		*puiBufLength = 0;
		goto Exit;
	}

	pucEncPtr = pucBuf;

	// Encode the number of characters as a SEN.  If pucEncPtr is
	// NULL, the caller is only interested in the length of the encoded
	// string, so a temporary buffer is used to call f_encodeSEN.

	uiTmp = f_encodeSEN( uiStrLen, &pucTmpSen);
	if( pucEncPtr)
	{
		if( (uiEncodedLen + uiTmp) >= uiMaxLen)
		{
			rc = RC_SET( NE_SFLM_CONV_DEST_OVERFLOW);
			goto Exit;
		}
		f_memcpy( pucEncPtr, &ucTmpSen[ 0], uiTmp);
		pucEncPtr += uiTmp;
	}
	uiEncodedLen += uiTmp;

	// Encode the string using UTF-8

	if( uiStrLen)
	{
		if( bDirectCopy)
		{
			// Since all of the characters are ASCII and have
			// values <= 0x7F, the string can be copied directly
			// into the destination buffer buffer.

			if( uiEncodedLen + uiStrLen >= uiMaxLen)
			{
				rc = RC_SET( NE_SFLM_CONV_DEST_OVERFLOW);
				goto Exit;
			}

			if( pucEncPtr)
			{
				f_memcpy( pucEncPtr, pszStr, uiStrLen);
				pucEncPtr += uiStrLen;
			}
			uiEncodedLen += uiStrLen;
		}
		else
		{
			pszPtr = pszStr;
			while( *pszPtr)
			{
				if( (uiTmp = uiMaxLen - uiEncodedLen) == 0)
				{
					rc = RC_SET( NE_SFLM_CONV_DEST_OVERFLOW);
					goto Exit;
				}

				// Convert the native character to ASCII before
				// mapping it to Unicode.

				if( RC_BAD( rc = f_uni2UTF8( (FLMUNICODE)f_toascii( *pszPtr),
					pucEncPtr, &uiTmp)))
				{
					goto Exit;
				}

				uiEncodedLen += uiTmp;

				if( pucEncPtr)
				{
					pucEncPtr += uiTmp;
				}

				pszPtr++;
				uiCharsEncoded++;
			}

			// Make sure the string length (which may have been provided by
			// the caller) was correct.

			if( uiCharsEncoded != uiStrLen)
			{
				rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_PARM);
				goto Exit;
			}
		}
	}

	// Terminate the string with a 0 byte

	if( (uiMaxLen - uiEncodedLen) == 0)
	{
		rc = RC_SET( NE_SFLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}

	if( pucEncPtr)
	{
		*pucEncPtr++ = 0;
	}
	uiEncodedLen++;

	// Return the length of the encoded string

	*puiBufLength = uiEncodedLen;

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Encode a string into FLAIM's internal text format
			(SEN-prefixed, null-terminated UTF8).  The string is prefixed with
			a SEN that indicates the number of characters, not including the
			terminating NULL.

			This routine can be called with a NULL input buffer.  If it is
			called in this way, the length of the encoded string will be
			returned in *puiBufLength, but no encoding will actually be
			performed.
****************************************************************************/
RCODE	flmUTF8ToStorage(
	const FLMBYTE *	pucUTF8,					// UTF8 string to encode
	FLMUINT				uiBytesInBuffer,		// Maximum bytes to process from source
	FLMBYTE *			pucBuf,					// Destination buffer
	FLMUINT *			puiBufLength)			// [IN ] Size of pucBuf,
														// [OUT] Amount of pucBuf used
{
	RCODE					rc = NE_SFLM_OK;
	FLMBYTE *			pucEncPtr;
	const FLMBYTE *	pucPtr = NULL;
	FLMUINT				uiMaxLen;
	FLMUINT				uiEncodedLen = 0;
	FLMUINT				uiCharCount;
	FLMUINT				uiByteCount;
	FLMUINT				uiTmp;
	FLMUNICODE			uChar;
	FLMBYTE				ucTmpSen[ 5];
	FLMBYTE *			pucTmpSen = &ucTmpSen[ 0];
	const FLMBYTE *	pucEnd = NULL;

	if( !pucBuf)
	{
		uiMaxLen = (~(FLMUINT)0);
	}
	else
	{
		uiMaxLen = *puiBufLength;
	}
	
	if( uiBytesInBuffer)
	{
		pucEnd = &pucUTF8[ uiBytesInBuffer];
	}

	// Determine the number of bytes and characters in the
	// string

	uiCharCount = 0;
	pucPtr = pucUTF8;
	for( ;;)
	{
		if( RC_BAD( rc = f_getCharFromUTF8Buf( &pucPtr, pucEnd, &uChar)))
		{
			goto Exit;
		}

		if( !uChar)
		{
			break;
		}
		
		uiCharCount++;
	}

	if( !uiCharCount)
	{
		*puiBufLength = 0;
		goto Exit;
	}

	pucEncPtr = pucBuf;

	// Encode the number of characters as a SEN.  If pucEncPtr is
	// NULL, the caller is only interested in the length of the encoded
	// string, so a temporary buffer is used to call f_encodeSEN.

	uiTmp = f_encodeSEN( uiCharCount, &pucTmpSen);
	if( pucEncPtr)
	{
		if( (uiEncodedLen + uiTmp) >= uiMaxLen)
		{
			rc = RC_SET( NE_SFLM_CONV_DEST_OVERFLOW);
			goto Exit;
		}
		f_memcpy( pucEncPtr, &ucTmpSen[ 0], uiTmp);
		pucEncPtr += uiTmp;
	}
	uiEncodedLen += uiTmp;

	// Copy the UTF8 characters into the destination buffer

	uiByteCount = (FLMUINT)(pucPtr - pucUTF8);
	if( pucEncPtr)
	{
		if( (uiMaxLen - uiEncodedLen) < uiByteCount)
		{
			rc = RC_SET( NE_SFLM_CONV_DEST_OVERFLOW);
			goto Exit;
		}
		f_memcpy( pucEncPtr, pucUTF8, uiByteCount);
		pucEncPtr += uiByteCount;
	}
	uiEncodedLen += uiByteCount;

	// Terminate the string with a 0 byte

	if( uiEncodedLen == uiMaxLen)
	{
		rc = RC_SET( NE_SFLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}

	if( pucEncPtr)
	{
		*pucEncPtr++ = 0;
	}
	uiEncodedLen++;

	// Return the length of the encoded string

	*puiBufLength = uiEncodedLen;

Exit:

	return( rc);
}

/****************************************************************************
Desc: 	Convert text storage string to Unicode.
Notes:	If puzOutBuf is NULL, only a count is returned
			in puiOutBufLen to indicate the number of bytes needed to
			contain the data.  Two (unicode) bytes must be
			added to this value to account for null termination.
****************************************************************************/
RCODE flmStorage2Unicode(
	FLMUINT				uiType,
	FLMUINT				uiBufLength,
	const FLMBYTE *	pucBuffer,
	FLMUINT *			puiOutBufLen,
			// [IN] Number of bytes available in buffer
			// [OUT] Returns the number of bytes that are needed to
			// represent the data.  The null termination byte(s) are
			// not included in this value.
	void *				pOutBuf)
			// [IN/OUT] Buffer to hold the data.
{
	RCODE					rc = NE_SFLM_OK;
	FLMUNICODE			uChar;
	FLMUINT				uiOffset = 0;
	FLMUINT				uiSenLen;
	FLMUINT				uiNumChars;
	FLMUINT				uiMaxOutChars;
	FLMBYTE				ucTempBuf[ 64];
	FLMUNICODE *		puzOutBuf = NULL;
	FLMBYTE *			pszOutBuf = NULL;
	const FLMBYTE *	pucEnd;
	FLMUINT				uiDecodeCount;

	if( !pucBuffer || !uiBufLength)
	{
		ucTempBuf[ 0] = 0;	// SEN encoding of 0
		ucTempBuf[ 1] = 0;	// String terminator
		pucBuffer = &ucTempBuf[ 0];
		uiBufLength = 2;
	}
	else if( uiType != SFLM_STRING_TYPE)
	{
		// If the value is a number, convert to text

		if( uiType == SFLM_NUMBER_TYPE)
		{
			FLMUINT	uiTmp;

			uiTmp = sizeof( ucTempBuf);
			if( RC_BAD( rc = flmStorageNum2StorageText( pucBuffer, uiBufLength,
				ucTempBuf, &uiTmp)))
			{
				goto Exit;
			}
			pucBuffer = &ucTempBuf[ 0];
			uiBufLength = uiTmp;
		}
		else
		{
			rc = RC_SET( NE_SFLM_CONV_ILLEGAL);
			goto Exit;
		}
	}

	if( !uiBufLength)
	{
		if( puiOutBufLen)
		{
			if( *puiOutBufLen >= 2)
			{
				*((FLMUNICODE *)pOutBuf) = 0;
			}
			
			*puiOutBufLen = 0;
		}
		goto Exit;
	}

	pucEnd = &pucBuffer[ uiBufLength];

	uiSenLen = f_getSENLength( *pucBuffer);
	if( pucBuffer + uiSenLen >= pucEnd)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_DATA_ERROR);
		goto Exit;
	}

	if( RC_BAD( rc = f_decodeSEN( &pucBuffer, pucEnd, &uiNumChars)))
	{
		goto Exit;
	}

	// If only a length is needed (number of bytes), we can
	// return that without parsing the string

	if( !pOutBuf)
	{
		uiOffset = uiNumChars;
		goto Exit;
	}

	flmAssert( puiOutBufLen);
	uiMaxOutChars = (*puiOutBufLen) / sizeof( FLMUNICODE);
	puzOutBuf = (FLMUNICODE *)pOutBuf;

	// If we have a zero-length string, jump to exit.

	if( !uiNumChars)
	{
		if( (pucBuffer + 1) != pucEnd || *pucBuffer != 0)
		{
			rc = RC_SET( NE_SFLM_DATA_ERROR);
			goto Exit;
		}

		if( *pucBuffer != 0)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_DATA_ERROR);
		}
		else if (!uiMaxOutChars)
		{
			goto Overflow_Error;
		}

		if( puzOutBuf)
		{
			*puzOutBuf = 0;
		}
		else
		{
			*pszOutBuf = 0;
		}
		
		goto Exit;
	}

	// Parse through the string, outputting data to the buffer as we go.

	uChar = 0;
	uiDecodeCount = 0;
	for( ;;)
	{
		// Decode the bytes.

		if( RC_BAD( rc = f_getCharFromUTF8Buf( &pucBuffer, pucEnd, &uChar)))
		{
			goto Exit;
		}

		if( !uChar)
		{
			break;
		}

		if( uiOffset == uiMaxOutChars)
		{
			goto Overflow_Error;
		}

		if( puzOutBuf)
		{
			puzOutBuf[ uiOffset++] = uChar;
		}
		else
		{
			if ( uChar <= 0xFF)
			{
				uChar = (FLMUNICODE)f_tonative( (FLMBYTE)uChar);
				pszOutBuf[ uiOffset++] = (FLMBYTE)uChar;
			}
			else
			{
				rc = RC_SET( NE_SFLM_CONV_ILLEGAL);
				goto Exit;
			}
		}
		
		uiDecodeCount++;
	}

	if( uChar || uiDecodeCount != uiNumChars)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_DATA_ERROR);
		goto Exit;
	}

	// There is room for the 0 terminating character, but we
	// will not increment return length.

	if( uiOffset < uiMaxOutChars)
	{
		if( puzOutBuf)
		{
			puzOutBuf[ uiOffset] = 0;
		}
		else
		{
			pszOutBuf[ uiOffset] = 0;
		}
	}
	else
	{
Overflow_Error:
		flmAssert( uiOffset == uiMaxOutChars);

		// If uiOffset is zero, so is uiMaxOutChars, which means
		// that we can't even put out the zero terminator.

		if (uiOffset)
		{
			uiOffset--;
			if( puzOutBuf)
			{
				puzOutBuf[ uiOffset] = 0;
			}
			else
			{
				pszOutBuf[ uiOffset] = 0;
			}
		}
		rc = RC_SET( NE_SFLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}

Exit:

	if( puiOutBufLen)
	{
		*puiOutBufLen = uiOffset + uiOffset;
	}

	return( rc);
}

/****************************************************************************
Desc:	Converts storage formats to UNICODE
****************************************************************************/
RCODE flmStorage2Unicode(
	FLMUINT				uiType,
	FLMUINT				uiStorageLength,
	const FLMBYTE *	pucStorageBuffer,
	F_DynaBuf *			pBuffer)
{
	RCODE					rc = NE_SFLM_OK;
	FLMBYTE				ucTempBuf[ 80];
	const FLMBYTE *	pucEnd;
	FLMUINT				uiSenLen;
	FLMUINT				uiNumChars;
	FLMUNICODE *		puzDestBuffer;

	pBuffer->truncateData( 0);
	
	if( uiType != SFLM_STRING_TYPE)
	{
		// If the value is a number, convert to text

		if( uiType == SFLM_NUMBER_TYPE)
		{
			FLMUINT	uiTmp;

			uiStorageLength = sizeof( ucTempBuf);
			if( RC_BAD( rc = flmStorageNum2StorageText( pucStorageBuffer, 
				uiStorageLength, ucTempBuf, &uiTmp)))
			{
				goto Exit;
			}
			pucStorageBuffer = &ucTempBuf[ 0];
			uiStorageLength = uiTmp;
		}
		else
		{
			rc = RC_SET( NE_SFLM_CONV_ILLEGAL);
			goto Exit;
		}
	}

	pucEnd = &pucStorageBuffer[ uiStorageLength];
	uiSenLen = f_getSENLength( *pucStorageBuffer);
	
	if( pucStorageBuffer + uiSenLen >= pucEnd)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_DATA_ERROR);
		goto Exit;
	}
	
	if( RC_BAD( rc = f_decodeSEN( &pucStorageBuffer, pucEnd, &uiNumChars)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pBuffer->allocSpace( 
		(uiNumChars + 1) * sizeof( FLMUNICODE), (void **)&puzDestBuffer)))
	{
		goto Exit;
	}

	// Parse through the string outputting data to the buffer as we go

	for( ;;)
	{
		if( RC_BAD( rc = f_getCharFromUTF8Buf( 
			&pucStorageBuffer, pucEnd, puzDestBuffer)))
		{
			goto Exit;
		}
		
		if( !(*puzDestBuffer))
		{
			break;
		}

		puzDestBuffer++;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc: 	Converts a storage buffer to UTF-8 (null-terminated) text
****************************************************************************/
RCODE flmStorage2UTF8(
	FLMUINT				uiType,
	FLMUINT				uiBufLength,
	const FLMBYTE *	pucBuffer,
	FLMUINT *			puiOutBufLen,
	FLMBYTE *			pucOutBuf)
{
	RCODE					rc = NE_SFLM_OK;
	const FLMBYTE *	pucEnd;
	FLMBYTE				ucTempBuf[ 64];
	FLMUINT				uiSenLen;

	if( !pucBuffer)
	{
		ucTempBuf[ 0] = 0;	// SEN encoding of 0
		ucTempBuf[ 1] = 0;	// String terminator
		pucBuffer = &ucTempBuf[ 0];
		uiBufLength = 2;
	}
	else if( uiType != SFLM_STRING_TYPE)
	{
		// If the value is a number, convert to text

		if( uiType == SFLM_NUMBER_TYPE)
		{
			FLMUINT	uiTmp;

			uiTmp = sizeof( ucTempBuf);
			if( RC_BAD( rc = flmStorageNum2StorageText( pucBuffer, uiBufLength,
				ucTempBuf, &uiTmp)))
			{
				goto Exit;
			}
			pucBuffer = &ucTempBuf[ 0];
			uiBufLength = uiTmp;
		}
		else
		{
			rc = RC_SET( NE_SFLM_CONV_ILLEGAL);
			goto Exit;
		}
	}

	if( !uiBufLength)
	{
		if( *puiOutBufLen && pucOutBuf)
		{
			*pucOutBuf = 0;
		}
		*puiOutBufLen = 0;
		goto Exit;
	}

	pucEnd = &pucBuffer[ uiBufLength];

	uiSenLen = f_getSENLength( *pucBuffer);
	if( pucBuffer + uiSenLen >= pucEnd)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_DATA_ERROR);
		goto Exit;
	}

	if( RC_BAD( rc = f_decodeSEN( &pucBuffer, pucEnd, NULL)))
	{
		goto Exit;
	}

	if( pucOutBuf)
	{
		if( *puiOutBufLen >= uiBufLength - uiSenLen)
		{
			f_memcpy( pucOutBuf, pucBuffer, uiBufLength - uiSenLen);
		}
	}

	*puiOutBufLen = (uiBufLength - uiSenLen) - 1;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Converts storage formats to UTF8
****************************************************************************/
RCODE flmStorage2UTF8(
	FLMUINT				uiType,
	FLMUINT				uiStorageLength,
	const FLMBYTE *	pucStorageBuffer,
	F_DynaBuf *			pBuffer)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBYTE		ucTempBuf[ 80];
	FLMUINT		uiSenLen;
	FLMBYTE *	pucDestBuffer;

	pBuffer->truncateData( 0);
	
	if( uiType != SFLM_STRING_TYPE)
	{
		// If the value is a number, convert to text

		if( uiType == SFLM_NUMBER_TYPE)
		{
			FLMUINT	uiTmp;

			uiStorageLength = sizeof( ucTempBuf);
			if( RC_BAD( rc = flmStorageNum2StorageText( pucStorageBuffer, 
				uiStorageLength, ucTempBuf, &uiTmp)))
			{
				goto Exit;
			}
			pucStorageBuffer = &ucTempBuf[ 0];
			uiStorageLength = uiTmp;
		}
		else
		{
			rc = RC_SET( NE_SFLM_CONV_ILLEGAL);
			goto Exit;
		}
	}

	uiSenLen = f_getSENLength( *pucStorageBuffer);
	if (uiSenLen >= uiStorageLength)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_DATA_ERROR);
		goto Exit;
	}
	
	if( RC_BAD( rc = pBuffer->allocSpace( uiStorageLength - uiSenLen, 
		(void **)&pucDestBuffer)))
	{
		goto Exit;
	}
	f_memcpy( pucDestBuffer, pucStorageBuffer + uiSenLen,
					uiStorageLength - uiSenLen);

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Reads and returns the value of the SEN indicating the number of
		characters encoded into the storage (UTF-8) string
****************************************************************************/
RCODE flmGetCharCountFromStorageBuf(
	const FLMBYTE **	ppucBuf,
	FLMUINT				uiBufSize,
	FLMUINT *			puiNumChars,
	FLMUINT *			puiSenLen)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiSenLen;
	FLMUINT		uiNumChars;

	if( !uiBufSize)
	{
		if( puiNumChars)
		{
			*puiNumChars = 0;
		}

		if( puiSenLen)
		{
			*puiSenLen = 0;
		}
		goto Exit;
	}

	if( (uiSenLen = f_getSENLength( (*ppucBuf)[ 0])) >= uiBufSize)
	{
		rc = RC_SET( NE_SFLM_DATA_ERROR);
		goto Exit;
	}

	if( RC_BAD( rc = f_decodeSEN( ppucBuf, *ppucBuf + uiSenLen, &uiNumChars)))
	{
		goto Exit;
	}

	if( puiNumChars)
	{
		*puiNumChars = uiNumChars;
	}

	if( puiSenLen)
	{
		*puiSenLen = uiSenLen;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		This routine converts an internal number to internal (ASCII) text.
Notes:	If the buffer pointer is NULL, the routine just determines how
			much buffer space is needed to store the number in a text string.
****************************************************************************/
RCODE flmStorageNum2StorageText(
	const FLMBYTE *	pucNum,
	FLMUINT				uiNumLen,
	FLMBYTE *			pucBuffer,
	FLMUINT *			puiBufLen)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT64	ui64Num;
	FLMINT64		i64Num;
	FLMUINT		uiOffset = 0;
	FLMBOOL		bNeg = FALSE;
	char			szTmpBuf[ 64];

	if( RC_BAD( rc = flmStorage2Number64( SFLM_NUMBER_TYPE, uiNumLen,
		pucNum, &ui64Num, NULL)))
	{
		if( rc == NE_SFLM_CONV_NUM_UNDERFLOW)
		{
			if( RC_BAD( rc = flmStorage2Number64( SFLM_NUMBER_TYPE, uiNumLen,
				pucNum, NULL, &i64Num)))
			{
				goto Exit;
			}

			ui64Num = (FLMUINT64)-i64Num;
			bNeg = TRUE;
		}
		else
		{
			goto Exit;
		}
	}

	if( bNeg)
	{
		szTmpBuf[ uiOffset++] = '-';
	}

	uiOffset += f_sprintf( &szTmpBuf[ uiOffset], "%I64u", ui64Num);

	if( RC_BAD( rc = flmNative2Storage( szTmpBuf, uiOffset,
									pucBuffer, puiBufLen, NULL)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}
