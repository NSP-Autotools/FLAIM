//------------------------------------------------------------------------------
// Desc:	Routines for building collation keys
// Tabs:	3
//
// Copyright (c) 1993-2007 Novell, Inc. All Rights Reserved.
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

/**************************************************************************
Desc:		Get the collation string and convert back to a text string
Ret:		Length of new wpStr
Notes:	Allocates the area for the word string buffer if will be over 256.
***************************************************************************/
RCODE flmColText2StorageText(
	const FLMBYTE *	pucColStr,				// Points to the collated string
	FLMUINT				uiColStrLen,			// Length of the collated string
	FLMBYTE *			pucStorageBuf,			// Output string to build - TEXT string
	FLMUINT *			puiStorageLen,			// In: Size of buffer, Out: Bytes used
	FLMUINT	   		uiLang,
	FLMBOOL *			pbDataTruncated,		// Sets to TRUE if data had been truncated
	FLMBOOL *			pbFirstSubstring)		// Sets to TRUE if first substring
{
#define LOCAL_CHARS		150
	FLMBYTE		ucWPStr[ LOCAL_CHARS * 2 + LOCAL_CHARS / 5 ];	// Sample + 20%
	FLMBYTE *  	pucWPPtr = NULL;
	FLMBYTE *	pucAllocatedWSPtr = NULL;
	FLMUINT		uiWPStrLen;
	FLMBYTE *	pucStoragePtr;
	FLMUINT		uiUnconvChars;
	FLMUINT		uiTmp;
	FLMUINT		uiMaxStorageBytes = *puiStorageLen;
	FLMUINT		uiMaxWPBytes;
	FLMUINT		uiStorageOffset;
	FLMBYTE		ucTmpSen[ 5];
	FLMBYTE *	pucTmpSen = &ucTmpSen[ 0];
	RCODE			rc = NE_FLM_OK;

	if( uiColStrLen > LOCAL_CHARS)
	{
		// If it won't fit, allocate a new buffer

		if( RC_BAD( rc = f_alloc( XFLM_MAX_KEY_SIZE * 2, &pucWPPtr)))
		{
			goto Exit;
		}

		pucAllocatedWSPtr = pucWPPtr;
		uiMaxWPBytes = uiWPStrLen = XFLM_MAX_KEY_SIZE * 2;
	}
	else
	{
		pucWPPtr = &ucWPStr[ 0];
		uiMaxWPBytes = uiWPStrLen = sizeof( ucWPStr);
	}

 	if( (uiLang >= FLM_FIRST_DBCS_LANG) &&
 		 (uiLang <= FLM_LAST_DBCS_LANG))
 	{
		if( RC_BAD( rc = f_asiaColStr2WPStr( pucColStr, uiColStrLen,
			pucWPPtr, &uiWPStrLen, &uiUnconvChars,
			pbDataTruncated, pbFirstSubstring)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = f_colStr2WPStr( pucColStr, uiColStrLen,
			pucWPPtr, &uiWPStrLen, uiLang, &uiUnconvChars,
			pbDataTruncated, pbFirstSubstring)))
		{
			goto Exit;
		}
	}

	// Copy word string to the storage string area

	uiWPStrLen >>= 1;	// Convert # of bytes to # of words
	pucStoragePtr = pucStorageBuf;
	uiStorageOffset = 0;

	// Encode the number of characters as a SEN.  If pucEncPtr is
	// NULL, the caller is only interested in the length of the encoded
	// string, so a temporary buffer is used to call f_encodeSEN.

	uiTmp = f_encodeSEN( uiWPStrLen - uiUnconvChars, &pucTmpSen);
	if( (uiStorageOffset + uiTmp) >= uiMaxStorageBytes)
	{
		rc = RC_SET( NE_FLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}
	f_memcpy( pucStoragePtr, &ucTmpSen[ 0], uiTmp);
	uiStorageOffset += uiTmp;

	// Encode each of the WP characters into UTF-8

	while( uiWPStrLen--)
	{
		FLMBYTE			ucChar;
		FLMBYTE			ucCharSet;
		FLMUNICODE		uChar;

		// Put the character in a local variable for speed

		ucChar = *pucWPPtr++;
		ucCharSet = *pucWPPtr++;

		if( ucCharSet == 0xFF && ucChar == 0xFF)
		{
			uChar = (((FLMUNICODE)*(pucWPPtr + 1)) << 8) | *pucWPPtr;
			pucWPPtr += 2;
			uiWPStrLen--; // Skip past 4 bytes for UNICODE
		}
		else
		{
			if( RC_BAD( rc = f_wpToUnicode(
				(((FLMUINT16)ucCharSet) << 8) + ucChar, &uChar)))
			{
				goto Exit;
			}
		}

		uiTmp = uiMaxStorageBytes - uiStorageOffset;
		if( RC_BAD( rc = f_uni2UTF8( uChar,
			&pucStorageBuf[ uiStorageOffset], &uiTmp)))
		{
			goto Exit;
		}
		uiStorageOffset += uiTmp;
	}

	if( uiStorageOffset >= uiMaxStorageBytes)
	{
		rc = RC_SET( NE_FLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}

	// Tack on a trailing NULL byte

	pucStorageBuf[ uiStorageOffset++] = 0;

	// Return the length of the storage buffer

	*puiStorageLen = uiStorageOffset;

Exit:

	if( pucAllocatedWSPtr)
	{
		f_free( &pucAllocatedWSPtr);
	}

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE F_DbSystem::compareUTF8Strings(
	const FLMBYTE *		pucLString,
	FLMUINT					uiLStrBytes,
	FLMBOOL					bLeftWild,
	const FLMBYTE *		pucRString,
	FLMUINT					uiRStrBytes,
	FLMBOOL					bRightWild,
	FLMUINT					uiCompareRules,
	FLMUINT					uiLanguage,
	FLMINT *					piResult)
{
	return( f_compareUTF8Strings( pucLString, uiLStrBytes, bLeftWild,
		pucRString, uiRStrBytes, bRightWild, uiCompareRules, uiLanguage,
		piResult));
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE F_DbSystem::compareUnicodeStrings(
	const FLMUNICODE *	puzLString,
	FLMUINT					uiLStrBytes,
	FLMBOOL					bLeftWild,
	const FLMUNICODE *	puzRString,
	FLMUINT					uiRStrBytes,
	FLMBOOL					bRightWild,
	FLMUINT					uiCompareRules,
	FLMUINT					uiLanguage,
	FLMINT *					piResult)
{
	return( f_compareUnicodeStrings( puzLString, uiLStrBytes, bLeftWild,
		puzRString, uiRStrBytes, bRightWild, uiCompareRules,
		uiLanguage, piResult));
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI F_DbSystem::utf8IsSubStr(
	const FLMBYTE *	pszString,
	const FLMBYTE *	pszSubString,
	FLMUINT				uiCompareRules,
	FLMUINT				uiLanguage,
	FLMBOOL *			pbExists)
{
	return( f_utf8IsSubStr( pszString, pszSubString, uiCompareRules,
		uiLanguage, pbExists));
}
