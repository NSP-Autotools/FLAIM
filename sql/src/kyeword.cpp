//------------------------------------------------------------------------------
// Desc:	This file contains the code to parse out individual words and
//			substrings in a text string.
// Tabs:	3
//
// Copyright (c) 1990-2000, 2002-2007 Novell, Inc. All Rights Reserved.
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

FSTATIC RCODE flmGetCharacter(
	IF_PosIStream *	pIStream,
	FLMUINT *			puiCompareRules,
	FLMUINT16 *			pui16WPValue,
	FLMUNICODE *		puUniValue);

FSTATIC RCODE flmTextGetCharType(
	IF_PosIStream *	pIStream,
	FLMUINT *			puiCompareRules,
	FLMUNICODE *		puUniValue,
	FLMUINT *			puiType);

/*****************************************************************************
Desc:
*****************************************************************************/
FINLINE FLMUINT flmCharTypeAnsi7(
	FLMUINT16	ui16Char)
{
	if( (ui16Char >= ASCII_LOWER_A && ui16Char <= ASCII_LOWER_Z) ||
		 (ui16Char >= ASCII_UPPER_A && ui16Char <= ASCII_UPPER_Z) ||
		 (ui16Char >= ASCII_ZERO && ui16Char <= ASCII_NINE))
	{
		return SDWD_CHR;
	}

	if( ui16Char == 0x27)
	{
		return WDJN_CHR;
	}

	if( ui16Char <= 0x2B)
	{
		return DELI_CHR;
	}

	if( ui16Char == ASCII_COMMA ||
		 ui16Char == ASCII_DASH ||
		 ui16Char == ASCII_DOT ||
		 ui16Char == ASCII_SLASH ||
		 ui16Char == ASCII_COLON ||
		 ui16Char == ASCII_AT ||
		 ui16Char == ASCII_BACKSLASH ||
		 ui16Char == ASCII_UNDERSCORE)
	{
		return WDJN_CHR;
	}

	return DELI_CHR;
}

/*****************************************************************************
Desc:  	Return the next WP or unicode character value.
Return:	Number of bytes formatted to return the character value.
*****************************************************************************/
FSTATIC RCODE flmGetCharacter(
	IF_PosIStream *	pIStream,
	FLMUINT *			puiCompareRules,
	FLMUINT16 *			pui16WPValue,
	FLMUNICODE *		puUniValue)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUNICODE	uChar = 0;
	FLMUINT64	ui64AfterLastSpacePos = 0;
	FLMBOOL		bLastCharWasSpace = FALSE;
	FLMUINT		uiCompareRules = *puiCompareRules;

	for( ;;)
	{
		if (RC_BAD( rc = f_readUTF8CharAsUnicode( pIStream, &uChar)))
		{
			if (rc != NE_SFLM_EOF_HIT)
			{
				goto Exit;
			}
			rc = NE_SFLM_OK;
			if (bLastCharWasSpace &&
				 !(uiCompareRules & FLM_COMP_IGNORE_TRAILING_SPACE))
			{
				// bLastCharWasSpace flag can only be TRUE if either
				// FLM_COMP_IGNORE_TRAILING_SPACE is set or
				// FLM_COMP_COMPRESS_WHITESPACE is set.
				
				flmAssert( uiCompareRules & FLM_COMP_COMPRESS_WHITESPACE);
				uChar = ASCII_SPACE;
			}
			else
			{
				uChar = 0;
			}
			break;
		}

		if ((uChar = f_convertChar( uChar, uiCompareRules)) == 0)
		{
			continue;
		}

		if (uChar == ASCII_SPACE)
		{
			if (uiCompareRules & FLM_COMP_COMPRESS_WHITESPACE)
			{
				bLastCharWasSpace = TRUE;
				ui64AfterLastSpacePos = pIStream->getCurrPosition();
			}
			else if (uiCompareRules & FLM_COMP_IGNORE_TRAILING_SPACE)
			{
				
				// If the ignore trailing space flag is set, but the compress
				// space flag is not set, remember the position of the
				// first space character.  If we hit a non-space character,
				// we will reposition to after this space character.
				
				if (!bLastCharWasSpace)
				{
					bLastCharWasSpace = TRUE;
					ui64AfterLastSpacePos = pIStream->getCurrPosition();
				}
			}
			else
			{
				break;
			}
		}
		else
		{
			
			// Disable the ignore leading space flag, because we are now
			// past all leading space, and we don't want spaces ignored
			// now on account of that flag.
			
			uiCompareRules &= (~(FLM_COMP_IGNORE_LEADING_SPACE));
			if (bLastCharWasSpace)
			{
					
				// Position to after the last space
				
				if (RC_BAD( rc = pIStream->positionTo( ui64AfterLastSpacePos)))
				{
					goto Exit;
				}
				uChar = ASCII_SPACE;
				bLastCharWasSpace = FALSE;
			}
			break;
		}
	}

	if (pui16WPValue)
	{
		if (!f_unicodeToWP( uChar, pui16WPValue))
		{
			*pui16WPValue = 0;
		}
	}

	if (puUniValue)
	{
		*puUniValue = uChar;
	}

Exit:

	*puiCompareRules = uiCompareRules;

	return( rc);
}

/****************************************************************************
Desc:	Substring-ize the string in a node.  Normalize spaces and hyphens if
		told to.  Example: ABC  DEF
			ABC DEF
			BC DEF
			C DEF
			DEF
****************************************************************************/
RCODE KYSubstringParse(
	IF_PosIStream *	pIStream,
	FLMUINT *			puiCompareRules,	// [in/out] comparison rules
	FLMUINT				uiLimitParm,		// [in] Max characters
	FLMBYTE *			pucSubstrBuf,		// [out] buffer to fill
	FLMUINT *			puiSubstrBytes,	// [out] returns length
	FLMUINT *			puiSubstrChars)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiDestOffset = 0;
	FLMUINT		uiDestSize = *puiSubstrBytes;
	FLMUINT		uiLimit = uiLimitParm ? uiLimitParm : ICD_DEFAULT_SUBSTRING_LIMIT;
	FLMUINT		uiCharCnt = 0;
	FLMUINT		uiSize;
	FLMBOOL		bFirstCharacter = TRUE;
	FLMUINT64	ui64SavePosition = pIStream->getCurrPosition();

	// The limit must return one more than requested in order
	// for the text to collation routine to set the truncated flag.

	uiLimit++;

	while (uiLimit--)
	{
		FLMUNICODE	uChar;

		if( RC_BAD( rc = flmGetCharacter( pIStream, puiCompareRules, NULL, &uChar)))
		{
			goto Exit;
		}

		if (!uChar)
		{
			break;
		}

		uiCharCnt++;

		uiSize = uiDestSize - uiDestOffset;
		if (RC_BAD( rc = f_uni2UTF8( uChar, &pucSubstrBuf[ uiDestOffset], &uiSize)))
		{
			goto Exit;
		}
		uiDestOffset += uiSize;

		// If on the first word, position to start on next character
		// for the next call.

		if (bFirstCharacter)
		{
			bFirstCharacter = FALSE;

			// First character - save position so we can restore it
			// upon leaving the routine.

			ui64SavePosition = pIStream->getCurrPosition();
		}
	}

	if (uiDestOffset)
	{
		pucSubstrBuf[ uiDestOffset++] = 0;
	}

	*puiSubstrBytes = (FLMUINT)uiDestOffset;
	*puiSubstrChars = uiCharCnt;

	// Restore position of stream to first character after the first
	// character we found - to ready for next call.

	if (RC_BAD( rc = pIStream->positionTo( ui64SavePosition)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE KYEachWordParse(
	IF_PosIStream *	pIStream,
	FLMUINT *			puiCompareRules,
	FLMUINT				uiLimit,				// [in] Max characters
	FLMBYTE *			pucWordBuf,			// [out] Buffer of at least SFLM_MAX_KEY_SIZE
  	FLMUINT *			puiWordLen)
{
	RCODE				rc = NE_SFLM_OK;
	FLMBOOL			bSkippingDelim = TRUE;
	FLMUINT			uiWordLen = 0;
	FLMUINT			uiWordBufSize = *puiWordLen;
	FLMUNICODE		uChar;
	FLMUINT			uiType = 0;
	FLMUINT			uiSize;
	
	if (!uiLimit)
	{
		uiLimit = ICD_DEFAULT_SUBSTRING_LIMIT;
	}

	while (uiLimit)
	{
		if (RC_BAD( rc = flmTextGetCharType( pIStream, puiCompareRules, &uChar, &uiType)))
		{
			goto Exit;
		}
		if (!uChar)
		{
			break;
		}
		
		// Determine how to handle what we got.

		if (bSkippingDelim)
		{
			// If we were skipping delimiters, and we run into a non-delimiter
			// character, set the bSkippingDelim flag to FALSE to indicate the
			// beginning of a word.

			if (uiType & SDWD_CHR)
			{
				bSkippingDelim = FALSE;
				uiLimit--;
				uiSize = uiWordBufSize - uiWordLen;
				if (RC_BAD( rc = f_uni2UTF8( uChar, &pucWordBuf [uiWordLen],
												&uiSize)))
				{
					goto Exit;
				}
				uiWordLen += uiSize;
			}
		}
		else
		{

			// If we were NOT skipping delimiters, and we run into a delimiter
			// output the word.

			if (uiType & (DELI_CHR | WDJN_CHR))
			{
				break;
			}
			uiSize = uiWordBufSize - uiWordLen;
			if (RC_BAD( rc = f_uni2UTF8( uChar, &pucWordBuf [uiWordLen],
											&uiSize)))
			{
				goto Exit;
			}
			uiWordLen += uiSize;
		}
	}

	// Return the word, if any

	if (uiWordLen)
	{
		pucWordBuf [uiWordLen++] = 0;
	}
	*puiWordLen = uiWordLen;

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Return the next WP or unicode character value and parsing type.
*****************************************************************************/
FSTATIC RCODE flmTextGetCharType(
	IF_PosIStream *	pIStream,
	FLMUINT *			puiCompareRules,
	FLMUNICODE *		puUniValue,		// [out] Unicode value
	FLMUINT *			puiType			// Char attribute type.
	)
{
	RCODE				rc = NE_SFLM_OK;
	FLMUINT16		ui16WPValue;
	FLMUINT			uiCharSet;

	// We add on compress white space flag because we really want to ignore
	// spaces anyway - we are trying to get the "words" from this stream.
	
	if( RC_BAD( rc = flmGetCharacter( pIStream, puiCompareRules,
								&ui16WPValue, puUniValue)))
	{
		goto Exit;
	}

	if (ui16WPValue)
	{
		if (ui16WPValue < 0x080)
		{
			*puiType = flmCharTypeAnsi7( ui16WPValue);
			goto Exit;
		}
		uiCharSet = (FLMUINT)(ui16WPValue >> 8);

		if (uiCharSet == 1 || uiCharSet == 2 ||
			 (uiCharSet >= 8 && uiCharSet <= 11))
		{
			*puiType = SDWD_CHR;
			goto Exit;
		}

		*puiType = DELI_CHR;
	}
	else
	{

		// For now all unmapped unicode characters are treated
		// as delimeters

		*puiType = DELI_CHR;
	}

Exit:

	return( rc);
}
