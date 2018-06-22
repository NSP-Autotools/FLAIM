//-------------------------------------------------------------------------
// Desc:	Verify database structure.
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

FSTATIC eCorruptionType flmVerifyBlobField(
	FLMBYTE *		pBlobHdr,
	FLMUINT			uiBlobHdrLen);

FSTATIC eCorruptionType flmVerifyTextKey(
	IFD *				pIfd,
	FLMBOOL			bIxAsia,
	FLMBOOL			bIxArab,
	FLMBYTE *		pKey,
	FLMUINT			uiKeyLen,
	FLMUINT *		puiOffsetRV,
	FLMUINT *		puiCollateCountRV);

FSTATIC eCorruptionType flmVerifyNumberKey(
	FLMBYTE *		pKey,
	FLMUINT			uiKeyLen,
	FLMUINT *		puiOffset);

FSTATIC FLMBOOL flmGetSEN(
	FLMBYTE *		pTmpElmRec,
	FLMUINT *		puiDrnRV,
	FLMUINT *		puiNumBytesRV);
	
extern FLMBYTE 	gnDaysInMonth[];
extern FLMBYTE 	SENLenArray[];

#define ASC_N								95
#define ML1_N								242
#define ML2_N								145
#define BOX_N								88
#define TYP_N								103
#define ICN_N								255
#define MTH_N								238
#define MTX_N								229
#define GRK_N								219
#define HEB_N								123
#define CYR_N								250
#define KAN_N								63
#define USR_N								255
#define ARB_N								196
#define ARS_N								220

/****************************************************************************
Desc:	Number of characters in each character set
****************************************************************************/
FLMBYTE flm_c60_max[] = 
{
	ASC_N,									// ascii
	ML1_N,									// multinational 1
	ML2_N,									// multinational 2
	BOX_N,									// line draw
	TYP_N,									// typographic
	ICN_N,									// icons
	MTH_N,									// math
	MTX_N,									// math extension
	GRK_N,									// Greek
	HEB_N,									// Hebrew
	CYR_N,									// Cyrillic - Russian
	KAN_N,									// Kana
	USR_N,									// user
	ARB_N,									// Arabic
	ARS_N,									// Arabic Script
};

/****************************************************************************
Desc: Verifies that a WordPerfect character is a legal character.
****************************************************************************/
eCorruptionType flmVerifyWPChar(
	FLMUINT		uiCharSet,
	FLMUINT		uiChar)
{
	if (uiCharSet < F_NCHSETS)
	{
		if (uiChar >= (FLMUINT) flm_c60_max[uiCharSet])
		{
			return (FLM_BAD_CHAR);
		}
	}
	else if ((uiCharSet >= F_ACHSMIN) && (uiCharSet < F_ACHSETS))
	{
		if (uiChar > F_ACHCMAX)
		{
			return (FLM_BAD_ASIAN_CHAR);
		}
	}
	else
	{
		return (FLM_BAD_CHAR_SET);
	}

	return (FLM_NO_CORRUPTION);
}

/****************************************************************************
Desc:	This routine verifies a text field. It makes sure that all of the
		characters, formatting codes, etc. are valid.
****************************************************************************/
eCorruptionType flmVerifyTextField(
	FLMBYTE *			pText,
	FLMUINT				uiTextLen)
{
	FLMUINT				uiChar1;
	FLMUINT				uiBytesProcessed;
	FLMUINT				uiObjType;
	FLMUINT				uiObjLen;
	FLMUINT				uiLength;
	FLMUINT				uiCharSet;
	FLMUINT				uiChar;
	eCorruptionType	eCorruptionCode;

	if (!uiTextLen)
	{
		return (FLM_NO_CORRUPTION);
	}

	// Parse through the data, verifying that it is consistent with the
	// internal TEXT format.

	uiBytesProcessed = 0;
	while (uiBytesProcessed < uiTextLen)
	{

		// Determine what we are pointing at

		uiChar1 = (FLMUINT) * pText;
		uiObjType = (FLMUINT) (flmTextObjType( uiChar1));
		
		switch (uiObjType)
		{
			case ASCII_CHAR_CODE:
			{
				uiObjLen = 1;

				// Before testing anything else, make sure that we are not going
				// to access something beyond the end of the buffer.

				if (uiBytesProcessed + uiObjLen > uiTextLen)
				{
					return (FLM_BAD_TEXT_FIELD);
				}

				// There should NEVER be a character less than 32.

				if (uiChar1 < 32)
				{
					return (FLM_BAD_CHAR);
				}
				
				break;
			}
			
			case CHAR_SET_CODE:
			{
				uiObjLen = 2;

				// Before testing anything else, make sure that we are not going
				// to access something beyond the end of the buffer.

				if (uiBytesProcessed + uiObjLen > uiTextLen)
				{
					return (FLM_BAD_TEXT_FIELD);
				}

				// If the character set is zero, it had better be a valid
				// character.

				uiChar = *(pText + 1);
				if ((uiCharSet = (FLMUINT) (uiChar1 & (~uiObjType))) == 0)
				{
					if ((uiChar < 32) || (uiChar > 127))
					{
						return (FLM_BAD_CHAR);
					}
				}
				else
				{

					// Make sure we have a valid WordPerfect character.

					if ((eCorruptionCode = 
							flmVerifyWPChar( uiCharSet, uiChar)) != FLM_NO_CORRUPTION)
					{
						return (eCorruptionCode);
					}
				}
				
				break;
			}
			
			case WHITE_SPACE_CODE:
			{
				uiObjLen = 1;

				// Before testing anything else, make sure that we are not going
				// to access something beyond the end of the buffer.

				if (uiBytesProcessed + uiObjLen > uiTextLen)
				{
					return (FLM_BAD_TEXT_FIELD);
				}
				
				break;
			}
			
			case UNICODE_CODE:
			{
				uiObjLen = 3;

				if (uiBytesProcessed + uiObjLen > uiTextLen)
				{
					return (FLM_BAD_TEXT_FIELD);
				}

				// Anything is valid except 0xFFFF or 0xFFFE.

				break;
			}
			
			case UNK_GT_255_CODE:
			case UNK_LE_255_CODE:
			{
				if (uiObjType == UNK_GT_255_CODE)
				{
					uiObjLen = 1 + sizeof(FLMUINT16);

					// Before testing anything else, make sure that we are not
					// going to access something beyond the end of the buffer.

					if (uiBytesProcessed + uiObjLen > uiTextLen)
					{
						return (FLM_BAD_TEXT_FIELD);
					}

					uiLength = (FLMUINT) FB2UW( pText + 1);
				}
				else
				{
					uiObjLen = 2;

					// Before testing anything else, make sure that we are not
					// going to access something beyond the end of the buffer.

					if (uiBytesProcessed + uiObjLen > uiTextLen)
					{
						return (FLM_BAD_TEXT_FIELD);
					}

					uiLength = (FLMUINT) * (pText + 1);
				}

				if (uiLength > 65535 - uiObjLen)
				{
					return (FLM_BAD_TEXT_FIELD);
				}

				uiObjLen += uiLength;

				// Before testing anything else, make sure that we are not going
				// to access something beyond the end of the buffer.

				if (uiBytesProcessed + uiObjLen > uiTextLen)
				{
					return (FLM_BAD_TEXT_FIELD);
				}

				// Make sure it is one of our valid types.

				if (((uiChar1 & (~uiObjType)) != WP60_TYPE) &&
					 ((uiChar1 & (~uiObjType)) != NATIVE_TYPE))
				{
					return (FLM_BAD_TEXT_FIELD);
				}
				break;
			}
			
			case UNK_EQ_1_CODE:
			{
				uiObjLen = 2;

				// Before testing anything else, make sure that we are not going
				// to access something beyond the end of the buffer.

				if (uiBytesProcessed + uiObjLen > uiTextLen)
				{
					return (FLM_BAD_TEXT_FIELD);
				}

				// Make sure it is one of our valid types.

				if (((uiChar1 & (~uiObjType)) != WP60_TYPE) &&
					 ((uiChar1 & (~uiObjType)) != NATIVE_TYPE))
				{
					return (FLM_BAD_TEXT_FIELD);
				}
				
				break;
			}
			
			case EXT_CHAR_CODE:
			{
				uiObjLen = 3;

				// Before testing anything else, make sure that we are not going
				// to access something beyond the end of the buffer.

				if (uiBytesProcessed + uiObjLen > uiTextLen)
				{
					return (FLM_BAD_TEXT_FIELD);
				}

				// If the character set is zero, the character had better be
				// between 32 and 127.

				uiChar = (FLMUINT) (*(pText + 2));
				if ((uiCharSet = (FLMUINT) (*(pText + 1))) == 0)
				{
					if ((uiChar < 32) || (uiChar > 127))
					{
						return (FLM_BAD_CHAR);
					}
				}
				else
				{

					// Make sure we have a valid WordPerfect character.

					if ((eCorruptionCode = 
							flmVerifyWPChar( uiCharSet, uiChar)) != FLM_NO_CORRUPTION)
					{
						return (eCorruptionCode);
					}
				}
				
				break;
			}
			
			case OEM_CODE:
			{
				uiObjLen = 2;

				// Before testing anything else, make sure that we are not going
				// to access something beyond the end of the buffer.

				if (uiBytesProcessed + uiObjLen > uiTextLen)
				{
					return (FLM_BAD_TEXT_FIELD);
				}

				// OEM characters must be > 127.

				if (*(pText + 1) <= 127)
				{
					return (FLM_BAD_CHAR);
				}
				break;
			}
			
			default:
			{

				// These codes should NEVER HAPPEN.

				return (FLM_BAD_TEXT_FIELD);
			}
		}

		pText += uiObjLen;
		uiBytesProcessed += uiObjLen;
	}

	return (FLM_NO_CORRUPTION);
}

/****************************************************************************
Desc:	This routine verifies a number field.
****************************************************************************/
eCorruptionType flmVerifyNumberField(
	STATE_INFO *	pStateInfo,
	FLMBYTE *		pNumber,
	FLMUINT			uiNumberLen)
{
	FLMUINT			uiChar;
	FLMBOOL			bFirstNibble;
	FLMUINT			uiNibbleCount;
	FLMBOOL			bHitExponent = FALSE;
	FLMBOOL			bRealNumberFlag = FALSE;

	if (!uiNumberLen)
	{
		return (FLM_NO_CORRUPTION);
	}

	bFirstNibble = TRUE;
	uiNibbleCount = 0;
	
	for (;;)
	{

		// Determine what we are pointing at

		uiChar = (FLMUINT) (*pNumber);
		if (bFirstNibble)
		{
			uiChar >>= 4;
			uiChar &= 0x0F;
			bFirstNibble = FALSE;
		}
		else
		{
			uiChar &= 0x0F;
			pNumber++;
			bFirstNibble = TRUE;
		}

		uiNibbleCount++;
		switch (uiChar)
		{
			case 0x0A:
			{
				
				// Periods are currently NOT supported.

				return (FLM_BAD_NUMBER_FIELD);
			}
			
			case 0x0B:
			{

				// The minus had better be the very FIRST character except E-xF.

				if (uiNibbleCount > 1 && !bHitExponent)
				{
					return (FLM_BAD_NUMBER_FIELD);
				}
				break;
			}
			
			case 0x0C:
			case 0x0D:
			{
				return (FLM_BAD_NUMBER_FIELD);
			}
			
			case 0x0E:
			{
				
				// 'E' should be in the first or second position, but the format
				// has the ability to change at a later time.

				if (bRealNumberFlag)
				{
					return (FLM_BAD_NUMBER_FIELD);
				}

				bRealNumberFlag = TRUE;
				bHitExponent = TRUE;
				break;
			}
			
			case 0x0F:
			{
				if (bHitExponent)
				{
					bHitExponent = FALSE;
					break;
				}

				// If we didn't end right on the last byte, we have a problem.

				if ((uiNibbleCount + 1) / 2 < uiNumberLen)
				{
					return (FLM_BAD_NUMBER_FIELD);
				}
				else
				{
					return (FLM_NO_CORRUPTION);
				}
			}

			default:
			{
				break;
			}
		}

		// If we are at the last byte, but have not encountered a 0x0F we
		// have a corrupted number.

		if (uiNibbleCount / 2 == uiNumberLen)
		{
			return (FLM_BAD_NUMBER_FIELD);
		}

		if (pStateInfo->uiVersionNum >= FLM_FILE_FORMAT_VER_4_62)
		{
			// Numbers greater than 21 digits not yet supported.

			if (!bRealNumberFlag && (uiNibbleCount > 21))
			{
				return (FLM_BAD_NUMBER_FIELD);
			}
		}
		else
		{
			// Numbers greater than 11 digits not yet supported.

			if (!bRealNumberFlag && (uiNibbleCount > 11))
			{
				return (FLM_BAD_NUMBER_FIELD);
			}
		}
	}
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC eCorruptionType flmVerifyBlobField(
	FLMBYTE *	pBlobHdr,
	FLMUINT		uiBlobHdrLen)
{
	FLMUINT		uiSubType;
	FLMUINT		uiPathLen;
	FLMUINT		uiCharLen = 1;
	FLMBYTE *	pPath;

#define BLOB_ALIGNMENT_FUDGE	8

	if (!uiBlobHdrLen)
	{
		return (FLM_NO_CORRUPTION);
	}

	// Definitions taken from fblob.cpp

	#define BLOB_H_VERSION_LEN_POS	0
	#define BLOB_CODE_VERSION			28
	#define BLOB_H_STORAGE_TYPE_POS	1
	#define BLOB_H_FLAGS_POS			2
	#define BLOB_H_TYPE_POS				4

	// Type of DATA or 0 if unknown

	#define BLOB_H_FUTURE2				6
	#define BLOB_H_RAW_SIZE_POS		8
	#define BLOB_H_STORAGE_SIZE_POS	12
	#define BLOB_H_MATCH_STAMP_POS	16
	#define BLOB_MATCH_STAMP_SIZE		8
	#define BLOB_H_RIGHT_KEY_POS		24

	// Non-portable Reference BLOB Field Layout

	#define BLOB_R_CHARSET_POS			28
	#define BLOB_R_STRLENGTH_POS		29
	#define BLOB_R_PATH_POS				30

	// Must be at least as big as the smallest header.

	if (pBlobHdr[BLOB_H_VERSION_LEN_POS] != BLOB_CODE_VERSION)
	{
		return (FLM_BAD_BLOB_FIELD);
	}

	uiSubType = (FLMUINT) (pBlobHdr[BLOB_H_STORAGE_TYPE_POS] & 0x0F);

	if (uiSubType == BLOB_REFERENCE_TYPE)
	{
		if (uiBlobHdrLen < BLOB_R_PATH_POS)
		{
			return (FLM_BAD_BLOB_FIELD);
		}

		if (pBlobHdr[BLOB_R_CHARSET_POS] == 1)			// ANSI
		{
			uiPathLen = pBlobHdr[BLOB_R_STRLENGTH_POS];
		}
		else if (pBlobHdr[BLOB_R_CHARSET_POS] == 2)	// UNICODE
		{
			uiPathLen = (FLMUINT) (pBlobHdr[BLOB_R_STRLENGTH_POS] * 2);
			uiCharLen = 2;
		}
		else
		{
			return (FLM_BAD_BLOB_FIELD);
		}

		// uiPathLen will include the NULL byte(s).

		if (uiBlobHdrLen < BLOB_R_PATH_POS + uiPathLen)
		{
			return (FLM_BAD_BLOB_FIELD);
		}

		pPath = pBlobHdr + BLOB_R_PATH_POS;
	}
	else
	{
		return (FLM_BAD_BLOB_FIELD);
	}

	// Verify FILENAME that no characters are less than 0x20 and zero
	// terminates. uiPathLen includes the NULL byte/word so pre-decrement.
	// Zero or one path length is invalid. EXTERNAL and REFERENCE BLOBS
	// ONLY!

	if (uiPathLen <= 1)
	{
		return (FLM_BAD_BLOB_FIELD);
	}

	for (; --uiPathLen;)
	{
		if (uiCharLen == 1)
		{
			if (*pPath++ < 0x20)
			{
				return (FLM_BAD_BLOB_FIELD);
			}
		}
		else
		{
			if (FB2UW( pPath) < 0x20)
			{
				return (FLM_BAD_BLOB_FIELD);
			}

			pPath += 2;
		}
	}

	if (uiCharLen == 1)
	{
		if (*pPath++ != 0)
		{
			return (FLM_BAD_BLOB_FIELD);
		}
	}
	else
	{
		if (FB2UW( pPath) != 0)
		{
			return (FLM_BAD_BLOB_FIELD);
		}

		pPath += 2;
	}

	return (FLM_NO_CORRUPTION);
}

/****************************************************************************
Desc:
****************************************************************************/
eCorruptionType flmVerifyField(
	STATE_INFO *	pStateInfo,
	FLMBYTE*			pField,
	FLMUINT			uiFieldLen,
	FLMUINT			uiFieldType)
{
	if (((uiFieldLen) && (!pField)) || ((!uiFieldLen) && (pField)))
	{
		return (FLM_BAD_FIELD_PTR);
	}

	switch (uiFieldType)
	{
		case FLM_TEXT_TYPE:
		{
			return (flmVerifyTextField( pField, uiFieldLen));
		}
		
		case FLM_NUMBER_TYPE:
		{
			return (flmVerifyNumberField( pStateInfo, pField, uiFieldLen));
		}
		
		case FLM_BINARY_TYPE:
		{
			break;
		}
		
		case FLM_BLOB_TYPE:
		{
			return (flmVerifyBlobField( pField, uiFieldLen));
		}
		
		case FLM_CONTEXT_TYPE:
		{

			// Length must be zero or four in context fields.

			if ((uiFieldLen != 0) && (uiFieldLen != 4))
			{
				return (FLM_BAD_CONTEXT_FIELD);
			}
			break;
		}
		
		default:
		{

			// Unknown type.

			return (FLM_BAD_FIELD_TYPE);
		}
	}

	return (FLM_NO_CORRUPTION);
}

/****************************************************************************
Desc: Verifies a text field within a key - compound or single.
****************************************************************************/
FSTATIC eCorruptionType flmVerifyTextKey(
	IFD *			pIfd,						// Pointer to index field definition structure.
	FLMBOOL		bIxAsia,					// Is this one of the ASIAN indexes?
	FLMBOOL		bIxArab,					// Is this one of the ARAB indexes?
	FLMBYTE *	pKey,						// Pointer to beginning of key.
	FLMUINT		uiKeyLen,				// Byte length of entire key.
	FLMUINT *	puiOffsetRV,			// Offset in key where text key begins.
												// Returns offset where text key ends.
	FLMUINT*		puiCollateCountRV)	// Returns the number of collation characters.
{
	FLMUINT	uiCntJ = *puiOffsetRV;
	FLMUINT	uiTextCollateCount = 0;
	FLMUINT	uiArabCollateCount = 0;
	FLMUINT	uiCntK;
	FLMUINT	uiBitsToSkip;
	FLMUINT	uiBit;
	FLMUINT	uiChar;
	FLMUINT	uiNextChar = 0;
	FLMUINT	uiCaseBits;

	// See how many collated values there are - go until we hit 0x07, 0x02,
	// 0x04, 0x05, 0x06, 0x01, or end of key.

	while (uiCntJ < uiKeyLen)
	{
		uiNextChar = (FLMUINT) pKey[uiCntJ];
		if (bIxAsia)
		{
			if (uiCntJ + 1 >= uiKeyLen)
			{
				return (FLM_BAD_KEY_LEN);
			}

			uiNextChar <<= 8;
			uiNextChar += pKey[uiCntJ + 1];
		}

		if ((uiNextChar == END_COMPOUND_MARKER) ||
			 (uiNextChar == COMPOUND_MARKER) ||
			 (uiNextChar == COLL_FIRST_SUBSTRING) ||
			 (uiNextChar == (COLL_MARKER | SC_SUB_COL)) ||
			 (uiNextChar == (COLL_MARKER | SC_MIXED)) ||
			 ((uiNextChar == (COLL_MARKER | SC_LOWER)) && (!bIxAsia)) ||
			 ((uiNextChar == (COLL_MARKER | SC_UPPER)) && (!bIxAsia)) ||
			 (uiNextChar == COLL_TRUNCATED))
		{

			// These checks must be in order

			if (uiCntJ > *puiOffsetRV && uiNextChar == COLL_FIRST_SUBSTRING)
			{
				uiCntJ++;
				if (bIxAsia)
				{
					uiCntJ++;
				}

				continue;
			}

			if (uiNextChar == COLL_TRUNCATED)
			{
				uiCntJ++;
				if (bIxAsia)
				{
					uiCntJ++;
				}

				// Get character after COLL_TRUNCATED, if any

				if (uiCntJ < uiKeyLen)
				{
					uiNextChar = (FLMUINT) pKey[uiCntJ];
					if (bIxAsia)
					{
						if (uiCntJ + 1 >= uiKeyLen)
						{
							return (FLM_BAD_KEY_LEN);
						}

						uiNextChar <<= 8;
						uiNextChar += pKey[uiCntJ + 1];
					}
				}
			}
			break;
		}

		// It is impossible to have a collated value that is less than 0x20
		// (space).

		if (uiNextChar < 0x20)
		{
			return (FLM_BAD_TEXT_KEY_COLL_CHAR);
		}

		uiTextCollateCount++;
		uiCntJ++;
		if (bIxAsia)
		{
			uiCntJ++;
		}
	}

	// See if we got sub-collation values.

	if ((uiCntJ < uiKeyLen) && (uiNextChar == 0x07))
	{
		uiCntJ++;
		if (bIxAsia)
		{
			uiCntJ++;
		}

		if (uiCntJ >= uiKeyLen)
		{
			return (FLM_BAD_KEY_LEN);
		}

		uiCntK = 0;
		uiBit = 0x80;
		uiChar = pKey[uiCntJ];
		for (;;)
		{
			FLMUINT	uiBitCode;

			// Determine what the code is.

			uiBitCode = 0;
			while (uiChar & uiBit)
			{
				uiBitCode |= 0x01;
				uiBitCode <<= 1;

				// Get the next bit.

				uiBit >>= 1;

				// See if we need to get the next byte.

				if (!uiBit)
				{
					uiCntJ++;
					if (uiCntJ >= uiKeyLen)
					{
						return (FLM_BAD_KEY_LEN);
					}

					uiChar = (FLMUINT) pKey[uiCntJ];
					uiBit = 0x80;
				}
			}

			// The uiBitCode value tells whether or not there is a collating
			// value and what type it is.

			switch (uiBitCode)
			{

				// Code of zero means there is no collating value.

				case 0:
				{
					if ((!uiTextCollateCount) && (bIxArab))
					{
						uiBitsToSkip = 0;
					}
					else
					{
						uiBitsToSkip = 1;
					}

					uiCntK++;
					break;
				}

				// Code of 0x02 means that the sub-collation value is the next
				// five bits.

				case 0x02:
				{
					uiBitsToSkip = 6;
					uiCntK++;
					break;
				}

				// Code of 0x06 means that next two bytes contains sub-collation.
				// Code of 0x0E should only happen in Arabic, means that next two
				// bytes contains sub-collation, but should not be counted.

				case 0x0E:
				{
					if (!bIxArab)
					{
						return (FLM_BAD_KEY_LEN);
					}

					uiArabCollateCount++;

					// Fall through to 0x06 case.
				}

				case 0x06:
				{
					uiBitsToSkip = 0;
					uiCntJ += 3;
					
					if (uiCntJ > uiKeyLen)
					{
						return (FLM_BAD_KEY_LEN);
					}

					uiBit = 0x80;
					uiChar = (FLMUINT) pKey[uiCntJ];

					// If Arabic, the sub-collation should not be counted.

					if (uiBitCode != 0x0E)
					{
						uiCntK++;
					}
					
					break;
				}

				// Unicode character that did not convert to a WP char. The
				// actual unicode character follows the 1E.

				case 0x1E:
				{
					uiBitsToSkip = 0;
					uiCntJ += 3;
					if (uiCntJ > uiKeyLen)
					{
						return (FLM_BAD_KEY_LEN);
					}

					uiBit = 0x80;
					uiChar = (FLMUINT) pKey[uiCntJ];
					uiCntK++;

					// The spec for Unicode has an additional case bit. Added Oct
					// 98. Note, the extra case bit is only set if we are not an
					// asian index.

					if (!bIxAsia)
					{
						uiArabCollateCount++;
					}
					break;
				}
				
				default:
				{
					return (FLM_BAD_KEY_LEN);
				}
			}

			// Skip the required number of bits.

			while (uiBitsToSkip)
			{
				uiBit >>= 1;
				uiBitsToSkip--;
				if ((!uiBit) && ((uiBitsToSkip) || (uiCntK < uiTextCollateCount)))
				{
					uiCntJ++;
					if (uiCntJ == uiKeyLen)
					{
						return (FLM_BAD_KEY_LEN);
					}

					uiChar = (FLMUINT) pKey[uiCntJ];
					uiBit = 0x80;
				}
			}

			if (uiCntK >= uiTextCollateCount)
			{
				if (!bIxArab)
				{
					break;
				}

				// Arab languages have one more terminating bit. See if we
				// need to go to the next byte.

				if (!uiBit)
				{

					// Terminating bit is in next byte.

					uiCntJ++;
					if (uiCntJ >= uiKeyLen)
					{
						return (FLM_BAD_KEY_LEN);
					}

					uiChar = (FLMUINT) pKey[uiCntJ];
					uiBit = 0x80;
				}

				// If the next bit isn't set, we are done. However, we must
				// skip the current character

				if (!(uiChar & uiBit))
				{
					uiCntJ++;
					if (uiCntJ >= uiKeyLen)
					{
						return (FLM_BAD_KEY_LEN);
					}

					uiChar = (FLMUINT) pKey[uiCntJ];
					uiBit = 0x80;
					break;
				}
			}
		}

		// If NOT at beginning of byte, increment uiCntJ to skip rest of
		// byte.

		if (uiBit != 0x80)
		{
			uiCntJ++;
		}
	}

	// Do lower/upper case bits -- unless a POST index.

	if (pIfd->uiFlags & IFD_POST)
	{
		*puiCollateCountRV = uiTextCollateCount + uiArabCollateCount;
	}
	else
	{

		// If there are no UPPER/LOWER indicators, we have a problem.

		if (uiCntJ >= uiKeyLen)
		{
			return (FLM_BAD_KEY_LEN);
		}

		uiNextChar = (FLMUINT) pKey[uiCntJ];
		uiCntJ++;
		if (bIxAsia)
		{
			if (uiCntJ >= uiKeyLen)
			{
				return (FLM_BAD_KEY_LEN);
			}

			uiNextChar <<= 8;
			uiNextChar += (FLMUINT) pKey[uiCntJ];
			uiCntJ++;
		}

		switch (uiNextChar)
		{
			case COLL_FIRST_SUBSTRING:
			{
				break;
			}
			
			case COLL_TRUNCATED:
			{
				break;
			}
			
			case (COLL_MARKER | SC_LOWER):
			case (COLL_MARKER | SC_UPPER):
			{
				if (bIxAsia)
				{
					return (FLM_BAD_TEXT_KEY_CASE_MARKER);
				}
				break;
			}
			
			case (COLL_MARKER | SC_MIXED):
			{
				uiCaseBits = uiTextCollateCount + uiArabCollateCount;
				if (bIxAsia)
				{
					uiCaseBits <<= 1;
				}

				uiCntJ += ((uiCaseBits + 7) >> 3);
				break;
			}
			
			default:
			{
				return (FLM_BAD_TEXT_KEY_CASE_MARKER);
			}
		}
	}

	if (uiCntJ > uiKeyLen)
	{
		return (FLM_BAD_KEY_LEN);
	}

	*puiOffsetRV = uiCntJ;
	return (FLM_NO_CORRUPTION);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC eCorruptionType flmVerifyNumberKey(
	FLMBYTE *	pKey,
	FLMUINT		uiKeyLen,
	FLMUINT *	puiOffsetRV)
{
	FLMUINT	uiCntJ = *puiOffsetRV;
	FLMUINT	uiChar;
	FLMUINT	uiNumDigits;
	FLMUINT	uiNibble;

	// Determine the number of digits.

	uiChar = pKey[uiCntJ++];
	uiNumDigits = (FLMUINT) (uiChar & 0x7F);

	// If negative, the number of digits must be NOTed.

	if (!(uiChar & 0x80))
	{
		uiNumDigits = (FLMUINT) ((~uiNumDigits) & 0x7F);
	}

	// Adjust the number of digits by -64 + 1.

	uiNumDigits = (FLMUINT) (uiNumDigits - COLLATED_NUM_EXP_BIAS + 1);

	// Process until we run out of digits or key or until we hit the
	// compound marker or post marker.

	while( uiNumDigits && (uiCntJ < uiKeyLen) && (pKey[uiCntJ] != 0x02) &&
		(pKey[uiCntJ] != 0x01))
	{

		// Check the first nibble.

		uiNibble = (FLMUINT) ((pKey[uiCntJ] >> 4) & 0x0F);
		if ((uiNibble < 0x05) || (uiNibble > 0x0E))
		{
			return (FLM_BAD_NUMBER_KEY);
		}

		uiNumDigits--;

		// Check the 2nd nibble. If we are out of digits it had better be
		// 0x0F.

		uiNibble = (FLMUINT) (pKey[uiCntJ] & 0x0F);
		if (!uiNumDigits)
		{
			if (uiNibble != 0x0F)
			{
				return (FLM_BAD_NUMBER_KEY);
			}
		}
		else
		{
			if ((uiNibble < 0x05) || (uiNibble > 0x0E))
			{
				return (FLM_BAD_NUMBER_KEY);
			}

			uiNumDigits--;
		}

		uiCntJ++;
	}

	// If we ran out of key before we processed all of the digits, we have
	// an error.

	if ((uiNumDigits) && (uiCntJ > uiKeyLen))
	{
		return (FLM_BAD_KEY_LEN);
	}

	*puiOffsetRV = uiCntJ;
	return (FLM_NO_CORRUPTION);
}

/****************************************************************************
Desc:	This routine verifies that a collated key conforms to the index it
		belongs to.
****************************************************************************/
eCorruptionType flmVerifyKey(
	FLMBYTE*		pKey,				// Key which is to be verified.
	FLMUINT		uiKeyLen,		// Byte length of pKey.
	FLMUINT		uiIxLang,		// Language for index.
	IFD*			pIfdArray,		// List of fields in index.
	FLMUINT		uiNumIxFields)	// Number of fields in pIfdArray.
{
	IFD*					pIfd = pIfdArray;
	FLMUINT				uiI;
	FLMUINT				uiJ;
#define MAX_IX_FIELDS	100
	FLMUINT				uiCollateCount[MAX_IX_FIELDS];
	FLMUINT				uiPostByteCount;
	eCorruptionType	eCorruptionCode;
	FLMUINT				uiTotalTextChars = 0;
	FLMBOOL				bIxAsia;
	FLMBOOL				bIxArab;
	FLMBOOL				bIxIsPost = FALSE;
	FLMUINT				uiTrueNumIxFields = uiNumIxFields;
	FLMUINT				uiMarkerChar;
	FLMUINT				uiCaseBits;
	FLMUINT				uiFieldType;

	bIxAsia = (uiIxLang >= FLM_FIRST_DBCS_LANG && uiIxLang <= FLM_LAST_DBCS_LANG) 
							? TRUE 
							: FALSE;
							
	bIxArab = (uiIxLang == FLM_AR_LANG || uiIxLang == FLM_FA_LANG ||
				  uiIxLang == FLM_HE_LANG || uiIxLang == FLM_UR_LANG) 
				  			? TRUE 
							: FALSE;
		
	// If we weren't able to get the IX information from the dictionary
	// just return FLM_NO_CORRUPTION.

	if ((!pIfdArray) || (!uiNumIxFields))
	{
		return (FLM_NO_CORRUPTION);
	}

	// See if we have a POST index.

	for (uiI = 0; uiI < uiNumIxFields; uiI++)
	{
		if (pIfdArray[uiI].uiFlags & IFD_POST)
		{
			bIxIsPost = TRUE;
			break;
		}
	}

	// If it is not a compound index, set the uiNumIxFields to one - we
	// only need to examine one field, because they are all the same type.

	if (!(pIfdArray->uiFlags & IFD_COMPOUND))
	{
		if (bIxIsPost)
		{
			return (FLM_BAD_IX_DEF);
		}

		uiNumIxFields = 1;
	}

	uiJ = 0;
	uiI = 0;
	
	if (uiNumIxFields > MAX_IX_FIELDS)
	{
		return (FLM_BAD_IX_DEF);
	}

	while (uiJ < uiKeyLen)
	{
		uiCollateCount[uiI] = 0;

		// First see if the component has anything in it. If we hit the
		// compound marker right away, the component piece is empty.

		uiMarkerChar = (FLMUINT) pKey[uiJ];
		if ((bIxAsia) &&
			 (IFD_GET_FIELD_TYPE( pIfd) == FLM_TEXT_TYPE) &&
			 (!(pIfd->uiFlags & IFD_CONTEXT)))
		{
			if (uiJ + 1 == uiKeyLen)
			{
				return (FLM_BAD_KEY_LEN);
			}

			uiMarkerChar <<= 8;
			uiMarkerChar += (FLMUINT) pKey[uiJ + 1];
		}

		if (uiMarkerChar == 2)
		{
			;
		}
		else if (uiMarkerChar == 1)
		{

			// If we hit a 1, it should be the beginning of upper/lower case
			// bits for a POST index.

			break;
		}
		else if (uiMarkerChar == NULL_KEY_MARKER)
		{
			uiJ++;
			if ((bIxAsia) &&
				 (IFD_GET_FIELD_TYPE( pIfd) == FLM_TEXT_TYPE) &&
				 (!(pIfd->uiFlags & IFD_CONTEXT)))
			{
				uiJ++;
			}
		}

		// See if indexing context. If not, use the field's type to
		// determine how the key is formatted.

		else if (pIfd->uiFlags & IFD_CONTEXT)
		{

			// If indexing context, the first byte of the key is a 0x1E
			// followed by the two byte tag number in high/low order.

			if (pKey[uiJ] != 0x1E)
			{
				return (FLM_BAD_CONTEXT_KEY);
			}

			// Verify that the tag portion of the key matches the field
			// number.

			if (pIfd->uiFlags & IFD_COMPOUND)
			{
				if (f_bigEndianToUINT16( &pKey[uiJ + 1]) != pIfd->uiFldNum)
				{
					return (FLM_BAD_CONTEXT_KEY);
				}
			}
			else
			{
				FLMUINT	uiH;
				IFD *		pTmpIfd;

				// If it is NOT a compound index, be sure to check each field
				// in the index to see if it matches that field number - it
				// could be a multi-field index.

				for (uiH = 0, pTmpIfd = pIfdArray;
					  uiH < uiTrueNumIxFields;
					  uiH++, pTmpIfd++)
				{
					if (f_bigEndianToUINT16( &pKey[uiJ + 1]) == pTmpIfd->uiFldNum)
					{
						break;
					}
				}

				// If it did not match any of the fields, return an error.

				if (uiH == uiTrueNumIxFields)
				{
					return (FLM_BAD_CONTEXT_KEY);
				}
			}

			uiJ += 3;
		}
		else
		{
			switch (uiFieldType = (FLMUINT) (IFD_GET_FIELD_TYPE( pIfd)))
			{
				case FLM_TEXT_TYPE:
				{
					if ((eCorruptionCode = flmVerifyTextKey( pIfd, bIxAsia, bIxArab,
							pKey, uiKeyLen, &uiJ, 
							&uiCollateCount[uiI])) != FLM_NO_CORRUPTION)
					{
						return (eCorruptionCode);
					}

					uiTotalTextChars += uiCollateCount[uiI];
					break;
				}
				
				case FLM_NUMBER_TYPE:
				{
					if ((eCorruptionCode = flmVerifyNumberKey( pKey, 
						uiKeyLen, &uiJ)) != FLM_NO_CORRUPTION)
					{
						return (eCorruptionCode);
					}
					break;
				}
				
				case FLM_BINARY_TYPE:
				{
					while( (uiJ < uiKeyLen) && (pKey[uiJ] != 0x02) &&
						(pKey[uiJ] != 0x01) && (pKey[uiJ] != COLL_TRUNCATED))
					{
						if ((pKey[uiJ] < 0x20) || (pKey[uiJ] > 0x2F))
						{
							return (FLM_BAD_BINARY_KEY);
						}

						uiJ++;
					}

					if (uiJ < uiKeyLen && pKey[uiJ] == COLL_TRUNCATED)
					{
						uiJ++;
					}
					break;
				}
				
				case FLM_CONTEXT_TYPE:
				{
					if (pKey[uiJ] != 0x1F)
					{
						return (FLM_BAD_DRN_KEY);
					}

					uiJ += 5;
					break;
				}
				
				default:
				{
					return (FLM_BAD_KEY_FIELD_TYPE);
				}
			}
		}

		// See if there is another field.

		while( ((pIfd->uiFlags & IFD_LAST) == 0) &&
			(pIfd->uiCompoundPos == (pIfd + 1)->uiCompoundPos))
		{
			pIfd++;
		}

		// pIfd will point to the LAST ifd with the same compound position.
		// pIfd increments below AFTER the compound marker is added.

		uiI++;
		if (uiI == uiNumIxFields)
		{
			break;
		}

		// If this is not the last field, make sure we are pointing to a
		// compound marker.

		if (uiJ >= uiKeyLen)
		{
			return (FLM_BAD_KEY_COMPOUND_MARKER);
		}

		uiMarkerChar = (FLMUINT) pKey[uiJ];
		uiJ++;
		
		if (bIxAsia &&
			 IFD_GET_FIELD_TYPE( pIfd) == FLM_TEXT_TYPE &&
			 !(pIfd->uiFlags & IFD_CONTEXT))
		{
			if (uiJ >= uiKeyLen)
			{
				return (FLM_BAD_KEY_COMPOUND_MARKER);
			}

			uiMarkerChar <<= 8;
			uiMarkerChar += (FLMUINT) pKey[uiJ];
			uiJ++;
		}

		if (uiMarkerChar != 2)
		{
			return (FLM_BAD_KEY_COMPOUND_MARKER);
		}

		pIfd++;
	}

	// If we didn't get through all of the fields in the loop above make
	// sure the remaining fields are all optional. No need to check for
	// alternate keys.

	while (uiI < uiNumIxFields)
	{
		uiCollateCount[uiI] = 0;
		uiI++;
		pIfd++;
	}

	// If we have a POST index, get the lower/upper case bits for each text
	// field.

	if ((bIxIsPost) && (uiTotalTextChars))
	{

		// If it is not a compound key, we have an error.

		if (uiNumIxFields == 1)
		{
			return (FLM_BAD_IX_DEF);
		}

		// If we did not hit a 0x01, we have an error.

		if (uiJ >= uiKeyLen)
		{
			return (FLM_BAD_KEY_POST_MARKER);
		}

		uiMarkerChar = (FLMUINT) pKey[uiJ];
		uiJ++;

		// If the last field is text, and we are in an Asian index we need
		// two bytes for the post marker.

		if (bIxAsia)
		{
			pIfd = &pIfdArray[uiNumIxFields - 1];
			if ((IFD_GET_FIELD_TYPE( pIfd) == FLM_TEXT_TYPE) &&
				 (!(pIfd->uiFlags & IFD_CONTEXT)))
			{
				if (uiJ >= uiKeyLen)
				{
					return (FLM_BAD_KEY_POST_MARKER);
				}

				uiMarkerChar <<= 8;
				uiMarkerChar += (FLMUINT) pKey[uiJ];
				uiJ++;
			}
		}

		if (uiMarkerChar != 1)
		{
			return (FLM_BAD_KEY_POST_MARKER);
		}

		// Go through all of the fields looking for TEXT fields.

		uiPostByteCount = 0;
		uiI = 0;
		pIfd = pIfdArray;
		while ((uiI < uiNumIxFields) && (uiJ < uiKeyLen))
		{
			if (uiCollateCount[uiI])
			{
				FLMINT	uiTempCnt;

				uiMarkerChar = pKey[uiJ];
				uiJ++;
				uiPostByteCount++;
				if (bIxAsia)
				{
					if (uiJ >= uiKeyLen)
					{
						return (FLM_BAD_KEY_POST_BYTE_COUNT);
					}

					uiMarkerChar <<= 8;
					uiMarkerChar += (FLMUINT) pKey[uiJ];
					uiJ++;
					uiPostByteCount++;
				}

				switch (uiMarkerChar)
				{
					case 4:
					case 6:
					{
						if (bIxAsia)
						{
							return (FLM_BAD_TEXT_KEY_CASE_MARKER);
						}
						break;
					}
					
					case 5:
					{
						uiCaseBits = uiCollateCount[uiI];
						if (bIxAsia)
						{
							uiCaseBits <<= 1;
						}

						uiTempCnt = (FLMUINT) ((uiCaseBits + 7) >> 3);
						uiJ += uiTempCnt;
						uiPostByteCount += uiTempCnt;
						break;
					}
					
					default:
					{
						return (FLM_BAD_TEXT_KEY_CASE_MARKER);
					}
				}
			}

			while( ((pIfd->uiFlags & IFD_LAST) == 0) &&
					 (pIfd->uiCompoundPos == (pIfd + 1)->uiCompoundPos))
			{
				uiI++;	// Compares against uiNumIxFields.
				pIfd++;
			}

			uiI++;
			pIfd++;
		}

		// Account for the post byte count

		if ((uiJ >= uiKeyLen) || (uiPostByteCount != (FLMUINT) pKey[uiJ]))
		{
			return (FLM_BAD_KEY_POST_BYTE_COUNT);
		}

		uiJ++;
	}

	// We must end exactly right on the end of the key.

	return (((uiJ != uiKeyLen) || (uiI != uiNumIxFields)) 
						? FLM_BAD_KEY_LEN 
						: FLM_NO_CORRUPTION);
}

/****************************************************************************
Desc:	Verifies a block's header and sets up the STATE_INFO structure to
		verify the rest of the block.
****************************************************************************/
eCorruptionType flmVerifyBlockHeader(
	STATE_INFO *	pStateInfo,
	BLOCK_INFO *	pBlockInfoRV,
	FLMUINT			uiBlockSize,
	FLMUINT			uiExpNextBlkAddr,
	FLMUINT			uiExpPrevBlkAddr,
	FLMBOOL			bCheckEOF,
	FLMBOOL			bCheckFullBlkAddr)
{
	FLMBYTE *		pBlk = pStateInfo->pBlk;

	if (pBlockInfoRV)
	{
		pBlockInfoRV->uiBlockCount++;
	}

	pStateInfo->uiNextBlkAddr = (FLMUINT) FB2UD( &pBlk[BH_NEXT_BLK]);
	if ((pStateInfo->uiEndOfBlock = 
			(FLMUINT) FB2UW( &pBlk[BH_ELM_END])) < BH_OVHD)
	{
		pStateInfo->uiEndOfBlock = BH_OVHD;
		return (FLM_BAD_BLK_HDR_BLK_END);
	}
	else if (pStateInfo->uiEndOfBlock > uiBlockSize)
	{
		pStateInfo->uiEndOfBlock = uiBlockSize;
		return (FLM_BAD_BLK_HDR_BLK_END);
	}
	else if (pBlockInfoRV)
	{
		pBlockInfoRV->ui64BytesUsed += 
			(FLMUINT64) (pStateInfo->uiEndOfBlock - BH_OVHD);
	}

	pStateInfo->uiElmOffset = BH_OVHD;

	// Verify the block address.

	if (bCheckFullBlkAddr)
	{
		if (GET_BH_ADDR( pBlk) != pStateInfo->uiBlkAddress)
		{
			return (FLM_BAD_BLK_HDR_ADDR);
		}
	}
	else
	{
		if ((GET_BH_ADDR( pBlk) & 0xFFFFFF00) !=
				 (pStateInfo->uiBlkAddress & 0xFFFFFF00))
		{
			return (FLM_BAD_BLK_HDR_ADDR);
		}
	}

	// Verify that block address is below the logical EOF

	if (bCheckEOF && pStateInfo->pDb)
	{
		if (!FSAddrIsBelow( pStateInfo->uiBlkAddress,
								 pStateInfo->pDb->LogHdr.uiLogicalEOF))
		{
			return (FLM_BAD_FILE_SIZE);
		}
	}

	// Verify the block type.

	if ((pStateInfo->uiBlkType != 0xFF) &&
		 (pStateInfo->uiBlkType != (FLMUINT) BH_GET_TYPE( pBlk)))
	{
		return (FLM_BAD_BLK_HDR_TYPE);
	}

	// Verify the block level.

	if ((pStateInfo->uiLevel != 0xFF) && (pStateInfo->uiLevel != pBlk[BH_LEVEL]))
	{
		return (FLM_BAD_BLK_HDR_LEVEL);
	}

	// Verify the previous block address. If uiExpPrevBlkAddr is zero, we
	// do not know what the previous address should be, so we don't verify.

	if ((uiExpPrevBlkAddr) &&
		 (uiExpPrevBlkAddr != (FLMUINT) FB2UD( &pBlk[BH_PREV_BLK])))
	{
		return (FLM_BAD_BLK_HDR_PREV);
	}

	// Verify the next block address. If uiExpNextBlkAddr is zero, we do
	// not know what the next address should be, se we don't verify.

	if ((uiExpNextBlkAddr) && (uiExpNextBlkAddr != pStateInfo->uiNextBlkAddr))
	{
		return (FLM_BAD_BLK_HDR_NEXT);
	}

	// Verify that if it is a root block, the root bit flags is set, or if
	// it is NOT a root block, that the root bit flag is NOT set.

	if (pStateInfo->pLogicalFile)
	{
		if (pStateInfo->uiLevel != 0xFF)
		{
			FLMBOOL	bShouldBeRootBlk; 
			
			bShouldBeRootBlk = (pStateInfo->uiLevel == 
					pStateInfo->pLogicalFile->pLfStats->uiNumLevels - 1) 
							? TRUE 
							: FALSE;

			if (((bShouldBeRootBlk) && (!BH_IS_ROOT_BLK( pBlk))) ||
				 ((!bShouldBeRootBlk) && (BH_IS_ROOT_BLK( pBlk))))
			{
				return (FLM_BAD_BLK_HDR_ROOT_BIT);
			}
		}

		// Verify the logical file number - if any.

		if (pStateInfo->pLogicalFile->pLFile->uiLfNum != 
				(FLMUINT) FB2UW( &pBlk[BH_LOG_FILE_NUM]))
		{
			return (FLM_BAD_BLK_HDR_LF_NUM);
		}
	}

	return (FLM_NO_CORRUPTION);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMINT flmCompareKeys(
	FLMBYTE *	pBuf1,
	FLMUINT		uiBuf1Len,
	FLMBYTE *	pBuf2,
	FLMUINT		uiBuf2Len)
{
	FLMINT		iStatus;

	if (!uiBuf1Len)
	{
		iStatus = (!uiBuf2Len) ? 0 : -1;
	}
	else if (!uiBuf2Len)
	{
		iStatus = 1;
	}
	else if (uiBuf1Len < uiBuf2Len)
	{
		if ((iStatus = f_memcmp( pBuf1, pBuf2, uiBuf1Len)) == 0)
		{
			iStatus = -1;
		}
	}
	else if (uiBuf1Len > uiBuf2Len)
	{
		if ((iStatus = f_memcmp( pBuf1, pBuf2, uiBuf2Len)) == 0)
		{
			iStatus = 1;
		}
	}
	else
	{
		iStatus = f_memcmp( pBuf1, pBuf2, uiBuf1Len);
	}

	return (iStatus);
}

/****************************************************************************
Desc: Verify a index or data element in a b-tree and set pStateInfo.
****************************************************************************/
eCorruptionType flmVerifyElement(
	STATE_INFO *	pStateInfo,
	FLMUINT			uiFlags)
{
	FLMUINT		uiEndOfBlock = pStateInfo->uiEndOfBlock;
	FLMUINT		uiOffset = pStateInfo->uiElmOffset;
	FLMUINT		uiBlkType = pStateInfo->uiBlkType;
	FLMBYTE *	pElm;
	FLMUINT		uiElmLen;
	FLMBYTE *	pElmKey;
	FLMUINT		uiElmKeyLen;
	FLMUINT		uiElmPKCLen;
	FLMINT		iCmpStatus = 0;
	FLMBYTE *	pCurKey = pStateInfo->pCurKey;
	FLMUINT		uiCurKeyLen = pStateInfo->uiCurKeyLen;
	FLMBOOL		bLfIsContainer = FALSE;
	FLMBOOL		bLfIsIndex = FALSE;
	FLMUINT		uiLfType = LF_INVALID;
	IXD *			pIxd = NULL;
	IFD *			pIfd = NULL;

	if (pStateInfo->pLogicalFile)
	{
		uiLfType = pStateInfo->pLogicalFile->pLFile->uiLfType;
		bLfIsContainer = (uiLfType == LF_CONTAINER) ? TRUE : FALSE;
		bLfIsIndex = (uiLfType == LF_INDEX) ? TRUE : FALSE;
		pIxd = pStateInfo->pLogicalFile->pIxd;
		pIfd = pStateInfo->pLogicalFile->pIfd;
	}

	// Get a pointer to the element to work with.

	pElm = pStateInfo->pElm = &pStateInfo->pBlk[uiOffset];

	// Get the element length.

	if (uiBlkType == BHT_LEAF)
	{
		if (uiOffset + BBE_KEY > uiEndOfBlock)
		{
			return (FLM_BAD_ELM_LEN);
		}

		uiElmLen = pStateInfo->uiElmLen = (FLMUINT) (BBE_LEN( pElm));
		pElmKey = pStateInfo->pElmKey = &pElm[BBE_KEY];
		pStateInfo->pElmRec = BBE_REC_PTR( pElm);
		pStateInfo->uiElmRecLen = BBE_GET_RL( pElm);
		pStateInfo->uiElmRecOffset = 0;

		// Get the element key length and previous key count (PKC).

		uiElmKeyLen = pStateInfo->uiElmKeyLen = (FLMUINT) (BBE_GET_KL( pElm));
		uiElmPKCLen = pStateInfo->uiElmPKCLen = (FLMUINT) (BBE_GET_PKC( pElm));
	}
	else if (uiBlkType == BHT_NON_LEAF_DATA)
	{
		if (uiOffset + pStateInfo->uiElmOvhd > uiEndOfBlock)
		{
			return (FLM_BAD_ELM_LEN);
		}

		uiElmLen = pStateInfo->uiElmLen = BNE_DATA_OVHD;
		pElmKey = pStateInfo->pElmKey = pElm;
		uiElmKeyLen = 4;
		uiElmPKCLen = 0;
	}
	else
	{
		if (uiBlkType == BHT_NON_LEAF_COUNTS)
		{
			pStateInfo->uiChildCount = FB2UD( pElm + BNE_CHILD_COUNT);
		}

		if (uiOffset + pStateInfo->uiElmOvhd > uiEndOfBlock)
		{
			return (FLM_BAD_ELM_LEN);
		}

		uiElmLen = pStateInfo->uiElmLen = (FLMUINT) BBE_GET_KL( pElm) +
			pStateInfo->uiElmOvhd +
			(BNE_IS_DOMAIN( pElm) ? BNE_DOMAIN_LEN : 0);
		pElmKey = pStateInfo->pElmKey = &pElm[pStateInfo->uiElmOvhd];

		// Get the element key length and previous key count (PKC).

		uiElmKeyLen = pStateInfo->uiElmKeyLen = (FLMUINT) (BBE_GET_KL( pElm));
		uiElmPKCLen = pStateInfo->uiElmPKCLen = (FLMUINT) (BBE_GET_PKC( pElm));
	}

	// Make sure the element length is within the block boundary.

	if (uiOffset + uiElmLen > uiEndOfBlock)
	{
		return (FLM_BAD_ELM_LEN);
	}

	// Get the record number from the element, if any.

	pStateInfo->uiElmDrn = 0;
	if ((bLfIsContainer) && (uiElmKeyLen + uiElmPKCLen == 4))
	{
		FLMBYTE	ucRecBuff[4];

		if (uiElmPKCLen)
		{
			if ((uiCurKeyLen >= uiElmPKCLen) && (pStateInfo->bValidKey))
			{
				f_memcpy( ucRecBuff, pCurKey, uiElmPKCLen);
			}
			else
			{
				f_memset( ucRecBuff, 0, uiElmPKCLen);
			}
		}

		if (uiElmKeyLen)
		{
			f_memcpy( &ucRecBuff[uiElmPKCLen], pElmKey, uiElmKeyLen);
		}

		pStateInfo->uiElmDrn = f_bigEndianToUINT32( ucRecBuff);
		if (pStateInfo->uiElmDrn == DRN_LAST_MARKER && uiBlkType == BHT_LEAF)
		{
			FLMUINT	uiTempDrn;

			// Verify that the marker value is > the last DRN value.

			uiTempDrn = (FLMUINT) FB2UD( &pElmKey[uiElmKeyLen]);
			if (uiTempDrn <= pStateInfo->uiLastElmDrn)
			{
				return (FLM_BAD_LAST_DRN);
			}

			pStateInfo->uiLastElmDrn = uiTempDrn;
		}
		else
		{
			pStateInfo->uiLastElmDrn = pStateInfo->uiElmDrn;
		}
	}

	// Verify the first/last flags if it is a leaf element.

	if (uiBlkType == BHT_LEAF)
	{
		FLMUINT	uiFirstFlag = (FLMUINT) (BBE_IS_FIRST( pElm));
		FLMUINT	uiPrevLastFlag = pStateInfo->uiElmLastFlag;

		// Verify the first element flag

		pStateInfo->uiElmLastFlag = (FLMUINT) (BBE_IS_LAST( pElm));
		if (uiPrevLastFlag != 0xFF)
		{
			if ((uiPrevLastFlag) && (!uiFirstFlag))
			{
				return (FLM_BAD_FIRST_ELM_FLAG);
			}
			else if ((!uiPrevLastFlag) && (uiFirstFlag))
			{
				return (FLM_BAD_LAST_ELM_FLAG);
			}
		}
	}

	// If we are on the last element, verify that we are indeed. If we are,
	// set the key length to zero.

	if ((uiElmLen == pStateInfo->uiElmOvhd) &&
		 (uiElmLen + uiOffset == uiEndOfBlock) &&
		 (pStateInfo->uiNextBlkAddr == BT_END))
	{
		pStateInfo->bValidKey = TRUE;
		pStateInfo->uiCurKeyLen = uiCurKeyLen = 0;
	}

	// If the length in a leaf element is BBE_LEM_LEN and it is not the
	// last element, we have an error.

	else if ((uiBlkType == BHT_LEAF) && (uiElmLen == BBE_LEM_LEN))
	{
		return (FLM_BAD_LEM);
	}

	// If this is the last element in the block, and this is the last block
	// in the chain, this had better be the LEM.

	else if ((uiOffset + uiElmLen == uiEndOfBlock) &&
				(pStateInfo->uiNextBlkAddr == BT_END))
	{
		return (FLM_BAD_LEM);
	}

	// Verify that the key is OK. The key must pass three tests in order
	// for it to be OK. First, the total key length must not exceed the
	// maximum key size. Second, if there is a previous key count, it must
	// not exceed the size of the previous key. Third, the new key must be
	// greater than or equal to the previous key -- all keys in the block
	// must be in ascending order. The third part is tested by comparing
	// only the part of the key which is going to change - the part pointed
	// to by pElmKey. We already know that the part of the key represented
	// in the previous key count is equal - by definition, so there is no
	// need to test it.

	else if ((uiElmKeyLen + uiElmPKCLen > MAX_KEY_SIZ) || 
				(bLfIsContainer && (uiElmKeyLen + uiElmPKCLen != 4) && 
				(uiBlkType != BHT_NON_LEAF_DATA)))
	{
		pStateInfo->bValidKey = FALSE;
		return (FLM_BAD_ELM_KEY_SIZE);
	}
	else if ((uiOffset == BH_OVHD && uiElmPKCLen) || 
				(uiElmPKCLen > uiCurKeyLen && uiCurKeyLen && pStateInfo->bValidKey))
	{
		pStateInfo->bValidKey = FALSE;
		return (FLM_BAD_ELM_PKC_LEN);
	}
	else
	{
		eCorruptionType	eKeyCorruptionCode = FLM_NO_CORRUPTION;

		// NOTE: The reason we are saving the error code into
		// eKeyCorruptionCode instead of returning when we detect it is
		// because we want to save the new key into pStateInfo->pCurKey
		// before we return. However, there are some checks which we must
		// make before saving the new key.

		if ((pStateInfo->bValidKey) || (uiElmPKCLen == 0))
		{
			if (pStateInfo->bValidKey)
			{
				if (uiBlkType == BHT_NON_LEAF_DATA)
				{
					iCmpStatus = flmCompareKeys( pElmKey, uiElmKeyLen, pCurKey,
														 DIN_KEY_SIZ);
				}
				else
				{
					iCmpStatus = flmCompareKeys( pElmKey, uiElmKeyLen,
														 &pCurKey[uiElmPKCLen],
														 uiCurKeyLen - uiElmPKCLen);
				}

				if (iCmpStatus < 0)
				{
					eKeyCorruptionCode = FLM_BAD_ELM_KEY_ORDER;
				}

				// Check the key compression to see if it is good.

				else if ((uiBlkType != BHT_NON_LEAF_DATA) &&
							(uiOffset > BH_OVHD) &&
							(uiElmKeyLen > 0) &&
							(uiElmPKCLen < BBE_PKC_MAX) &&
							(uiElmPKCLen < uiCurKeyLen) &&
							(pElmKey[0] == pCurKey[uiElmPKCLen]))
				{
					eKeyCorruptionCode = FLM_BAD_ELM_KEY_COMPRESS;
				}
			}

			// The keys had better be equal if it is a continuation element.
			// Otherwise, they had better be different.

			if (!pStateInfo->bValidKey)
			{
				pStateInfo->ui64KeyCount++;
				pStateInfo->ui64KeyRefs = 0;
			}
			else if (iCmpStatus != 0)
			{
				pStateInfo->ui64KeyCount++;
				pStateInfo->ui64KeyRefs = 0;

				// If this is a continuation element in a leaf block, the key
				// should be the same as the last key.

				if ((uiBlkType == BHT_LEAF) && (!BBE_IS_FIRST( pElm)))
				{
					eKeyCorruptionCode = FLM_BAD_CONT_ELM_KEY;
				}
			}
			else
			{
				if (pIxd)
				{
					if (((uiBlkType == BHT_LEAF) && (BBE_IS_FIRST( pElm))) ||
						 ((uiBlkType != BHT_LEAF) && (pIxd->uiFlags & IXD_UNIQUE)))
					{
						eKeyCorruptionCode = FLM_NON_UNIQUE_FIRST_ELM_KEY;
					}
				}
			}

			// Save the new key.

			if (uiBlkType != BHT_NON_LEAF_DATA)
			{
				pStateInfo->uiCurKeyLen = uiCurKeyLen = uiElmPKCLen + uiElmKeyLen;
				f_memcpy( &pCurKey[uiElmPKCLen], pElmKey, uiElmKeyLen);
			}
			else
			{
				pStateInfo->uiCurKeyLen = uiCurKeyLen = DIN_KEY_SIZ;
				*(FLMUINT32*) pCurKey = *(FLMUINT32*) pElmKey;
			}

			pStateInfo->bValidKey = TRUE;

			// Perform some additional checks on the key if an index key.

			if (eKeyCorruptionCode == FLM_NO_CORRUPTION &&
				 bLfIsIndex &&
				 pIxd &&
				 (uiFlags & FLM_CHK_FIELDS))
			{
				if (!pIxd->uiContainerNum)
				{
					FLMUINT	uiContainerPartLen = getIxContainerPartLen( pIxd);
					RCODE		tmpRc;
					LFILE *	pTmpLFile;

					if (uiCurKeyLen <= uiContainerPartLen)
					{
						eKeyCorruptionCode = FLM_BAD_KEY_LEN;
						goto Bad_Key;
					}

					if (pStateInfo->pDb)
					{
						if (RC_BAD( tmpRc = fdictGetContainer( pStateInfo->pDb->pDict,
									  getContainerFromKey( pCurKey, uiCurKeyLen),
									  &pTmpLFile)))
						{
							eKeyCorruptionCode = FLM_BAD_CONTAINER_IN_KEY;
							goto Bad_Key;
						}
					}

					uiCurKeyLen -= uiContainerPartLen;
				}

				eKeyCorruptionCode = flmVerifyKey( pCurKey, uiCurKeyLen,
															 pIxd->uiLanguage, pIfd,
															 pIxd->uiNumFlds);
			}
		}

		if (eKeyCorruptionCode != FLM_NO_CORRUPTION)
		{
Bad_Key:

			pStateInfo->bValidKey = FALSE;
			return (eKeyCorruptionCode);
		}
	}

	return (FLM_NO_CORRUPTION);
}

/****************************************************************************
Desc:
****************************************************************************/
eCorruptionType flmVerifyElmFOP(
	STATE_INFO *		pStateInfo)
{
	eCorruptionType	eCorruptionCode = FLM_NO_CORRUPTION;
	FLMBYTE *			pElmRec = pStateInfo->pElmRec;
	FLMUINT				uiElmRecOffset = pStateInfo->uiElmRecOffset;
	FLMUINT				uiElmRecLen = pStateInfo->uiElmRecLen;
	FLMBYTE *			pField;
	FLMBYTE *			pTmpFld;
	FLMBOOL				bDictField;
	FLMBOOL				bFOPIsField;
	FLMBYTE *			pElm = pStateInfo->pElm;
	FLMUINT				uiFOPDataLen;
	FLMUINT				uiFldOverhead;
	FLMUINT				uiBaseFldFlags;
	FLMUINT				uiLfNumber;
	FLMUINT				uiMaxDictFieldNum = FLM_LAST_DICT_FIELD_NUM;

	uiLfNumber = (pStateInfo->pLogicalFile) 
						? pStateInfo->pLogicalFile->pLFile->uiLfNum 
						: (FLMUINT) 0;

	if ((BBE_IS_FIRST( pElm)) && (uiElmRecOffset == 0))
	{
		pStateInfo->uiFOPType = 0xFF;
		pStateInfo->uiFieldLen = 0;
		pStateInfo->uiFieldProcessedLen = 0;
		pStateInfo->uiFieldType = 0xFF;
		pStateInfo->uiFieldNum = 0;
		pStateInfo->uiFieldLevel = 0;
		pStateInfo->uiJumpLevel = 0;
		pStateInfo->pFOPData = NULL;
		pStateInfo->uiFOPDataLen = 0;
		pStateInfo->pValue = NULL;
		pStateInfo->uiEncId = 0;
		pStateInfo->uiEncFieldLen = 0;
		pStateInfo->pData = NULL;
		pStateInfo->pvField = NULL;
		if (pStateInfo->pRecord)
		{
			pStateInfo->pRecord->clear();
		}

		pStateInfo->bElmRecOK = TRUE;
	}

	// If the state is goofed up, just give back the reset of the element.
	// Don't try to parse it.

	if (!pStateInfo->bElmRecOK)
	{
		pStateInfo->uiFOPType = FLM_FOP_CONT_DATA;
		pStateInfo->uiFieldLen = uiElmRecLen - uiElmRecOffset;
		pStateInfo->uiFieldProcessedLen = 0;
	}

	// If we are at the first of an element's record and we have a half
	// processed field, return that first.

	else if (uiElmRecOffset == 0 && 
				(pStateInfo->uiFieldProcessedLen < (pStateInfo->uiEncId 
						? pStateInfo->uiEncFieldLen 
						: pStateInfo->uiFieldLen)))
	{

		// If this is a FIRST element, we have a problem.

		if (BBE_IS_FIRST( pElm))
		{
			pStateInfo->bElmRecOK = FALSE;
			eCorruptionCode = FLM_BAD_FIRST_ELM_FLAG;
			goto Exit;
		}
		else
		{
			pStateInfo->uiFOPType = FLM_FOP_CONT_DATA;
		}
	}
	else if (pStateInfo->uiElmDrn == DRN_LAST_MARKER)
	{
		pStateInfo->uiFOPType = FLM_FOP_NEXT_DRN;
		if (uiElmRecLen != 4)
		{
			eCorruptionCode = FLM_BAD_LAST_DRN;
			goto Exit;
		}
	}
	else
	{
		bDictField = FALSE;
		bFOPIsField = TRUE;
		pStateInfo->uiFieldType = 0xFF;
		pStateInfo->uiFieldNum = 0;
		pStateInfo->uiFieldLen = 0;
		pStateInfo->uiEncId = 0;
		pStateInfo->uiEncFieldLen = 0;
		pField = &pElmRec[uiElmRecOffset];

		// Test for STANDARD field -- must be defined in dictionary.

		if (FOP_IS_STANDARD( pField))
		{
			pStateInfo->uiFOPType = FLM_FOP_STANDARD;
			bDictField = TRUE;
			uiFldOverhead = 2;
			if (uiElmRecOffset + uiFldOverhead > uiElmRecLen)
			{
				pStateInfo->bElmRecOK = FALSE;
				eCorruptionCode = FLM_BAD_ELM_FLD_OVERHEAD;
				goto Exit;
			}

			pStateInfo->uiFieldLen = (FLMUINT) (FSTA_FLD_LEN( pField));
			pStateInfo->uiFieldNum = FSTA_FLD_NUM( pField);

			// See if field is a child or sibling of previous field.

			if (FSTA_LEVEL( pField))
			{
				pStateInfo->uiFieldLevel++;
			}
		}

		// Test for OPEN type -- must also be defined in dictionary.

		else if (FOP_IS_OPEN( pField))
		{
			pStateInfo->uiFOPType = FLM_FOP_OPEN;
			bDictField = TRUE;

			// See if the field is a child or sibling of the previous field.

			if (FOPE_LEVEL( pField))
			{
				pStateInfo->uiFieldLevel++;
			}

			// See if field number is one or two bytes.

			pTmpFld = pField;
			uiFldOverhead = 0;
			uiBaseFldFlags = (FLMUINT) (FOP_GET_FLD_FLAGS( pTmpFld));
			pTmpFld++;

			if (FOP_2BYTE_FLDNUM( uiBaseFldFlags))
			{
				uiFldOverhead += 3;
				if (uiElmRecOffset + uiFldOverhead > uiElmRecLen)
				{
					pStateInfo->bElmRecOK = FALSE;
					eCorruptionCode = FLM_BAD_ELM_FLD_OVERHEAD;
					goto Exit;
				}

				pStateInfo->uiFieldNum = (FLMUINT) FB2UW( pTmpFld);
				pTmpFld += 2;
			}
			else
			{
				uiFldOverhead += 2;
				if (uiElmRecOffset + uiFldOverhead > uiElmRecLen)
				{
					pStateInfo->bElmRecOK = FALSE;
					eCorruptionCode = FLM_BAD_ELM_FLD_OVERHEAD;
					goto Exit;
				}

				pStateInfo->uiFieldNum = (FLMUINT) (*pTmpFld);
				pTmpFld++;
			}

			// Determine the field length

			if (FOP_2BYTE_FLDLEN( uiBaseFldFlags))
			{
				uiFldOverhead += 2;
				if (uiElmRecOffset + uiFldOverhead > uiElmRecLen)
				{
					pStateInfo->bElmRecOK = FALSE;
					eCorruptionCode = FLM_BAD_ELM_FLD_OVERHEAD;
					goto Exit;
				}

				pStateInfo->uiFieldLen = (FLMUINT) FB2UW( pTmpFld);
			}
			else
			{
				uiFldOverhead++;
				if (uiElmRecOffset + uiFldOverhead > uiElmRecLen)
				{
					pStateInfo->bElmRecOK = FALSE;
					eCorruptionCode = FLM_BAD_ELM_FLD_OVERHEAD;
					goto Exit;
				}

				pStateInfo->uiFieldLen = (FLMUINT) (*pTmpFld);
			}
		}

		// Test for UNREGISTERED fields -- not in dictionary.

		else if (FOP_IS_TAGGED( pField))
		{
			pStateInfo->uiFOPType = FLM_FOP_TAGGED;

			// See if the field is a child or sibling of the previous field.

			if (FTAG_LEVEL( pField))
			{
				pStateInfo->uiFieldLevel++;
			}

			pTmpFld = pField;
			uiBaseFldFlags = (FLMUINT) (FOP_GET_FLD_FLAGS( pTmpFld));
			pTmpFld++;
			uiFldOverhead = 1;

			// Get the field type.

			pStateInfo->uiFieldType = (FLMUINT) FTAG_GET_FLD_TYPE( *pTmpFld);
			pTmpFld++;
			uiFldOverhead++;

			// See if field number is one or two bytes.

			if (FOP_2BYTE_FLDNUM( uiBaseFldFlags))
			{
				uiFldOverhead += 2;
				if (uiElmRecOffset + uiFldOverhead > uiElmRecLen)
				{
					pStateInfo->bElmRecOK = FALSE;
					eCorruptionCode = FLM_BAD_ELM_FLD_OVERHEAD;
					goto Exit;
				}

				pStateInfo->uiFieldNum = (FLMUINT) FB2UW( pTmpFld);
				pTmpFld += 2;
			}
			else
			{
				uiFldOverhead++;
				if (uiElmRecOffset + uiFldOverhead > uiElmRecLen)
				{
					pStateInfo->bElmRecOK = FALSE;
					eCorruptionCode = FLM_BAD_ELM_FLD_OVERHEAD;
					goto Exit;
				}

				pStateInfo->uiFieldNum = (FLMUINT) (*pTmpFld);
				pTmpFld++;
			}

			// Toggle high bit to get true field number. FOP_TAGGED is now
			// used for more than just UNREGISTERED fields.

			if (pStateInfo->uiFieldNum & 0x8000)
			{
				pStateInfo->uiFieldNum &= (FLMUINT) 0x7FFF;
			}
			else
			{
				pStateInfo->uiFieldNum |= (FLMUINT) 0x8000;
			}

			// See if the field length is one or two bytes

			if (FOP_2BYTE_FLDLEN( uiBaseFldFlags))
			{
				uiFldOverhead += 2;
				if (uiElmRecOffset + uiFldOverhead > uiElmRecLen)
				{
					pStateInfo->bElmRecOK = FALSE;
					eCorruptionCode = FLM_BAD_ELM_FLD_OVERHEAD;
					goto Exit;
				}

				pStateInfo->uiFieldLen = (FLMUINT) FB2UW( pTmpFld);
			}
			else
			{
				uiFldOverhead++;
				if (uiElmRecOffset + uiFldOverhead > uiElmRecLen)
				{
					pStateInfo->bElmRecOK = FALSE;
					eCorruptionCode = FLM_BAD_ELM_FLD_OVERHEAD;
					goto Exit;
				}

				pStateInfo->uiFieldLen = (FLMUINT) (*pTmpFld);
			}
		}

		// Test for a field with NO value -- must be in dictionary

		else if (FOP_IS_NO_VALUE( pField))
		{
			pStateInfo->uiFOPType = FLM_FOP_NO_VALUE;
			bDictField = TRUE;
			pStateInfo->uiFieldLen = 0;

			// See if the field is a child or sibling of previous field

			if (FNOV_LEVEL( pField))
			{
				pStateInfo->uiFieldLevel++;
			}

			// See if field number is one or two bytes

			pTmpFld = pField + 1;
			uiBaseFldFlags = (FLMUINT) (FOP_GET_FLD_FLAGS( pField));
			if (FOP_2BYTE_FLDNUM( uiBaseFldFlags))
			{
				uiFldOverhead = 3;
				if (uiElmRecOffset + uiFldOverhead > uiElmRecLen)
				{
					pStateInfo->bElmRecOK = FALSE;
					eCorruptionCode = FLM_BAD_ELM_FLD_OVERHEAD;
					goto Exit;
				}

				pStateInfo->uiFieldNum = (FLMUINT) FB2UW( pTmpFld);
				pTmpFld += 2;
			}
			else
			{
				uiFldOverhead = 2;
				if (uiElmRecOffset + uiFldOverhead > uiElmRecLen)
				{
					pStateInfo->bElmRecOK = FALSE;
					eCorruptionCode = FLM_BAD_ELM_FLD_OVERHEAD;
					goto Exit;
				}

				pStateInfo->uiFieldNum = (FLMUINT) (*pTmpFld);
				pTmpFld++;
			}
		}

		// Test for the code which just resets the field level

		else if (FOP_IS_SET_LEVEL( pField))
		{
			FLMUINT	uiTempLevel;

			pStateInfo->uiFOPType = FLM_FOP_JUMP_LEVEL;
			uiFldOverhead = 1;
			if (uiElmRecOffset + uiFldOverhead > uiElmRecLen)
			{
				pStateInfo->bElmRecOK = FALSE;
				eCorruptionCode = FLM_BAD_ELM_FLD_OVERHEAD;
				goto Exit;
			}

			bFOPIsField = FALSE;
			pStateInfo->uiFieldNum = 0;
			pStateInfo->uiFieldLen = 0;
			pStateInfo->uiFieldType = 0xFF;

			// Jumping back better not cause us to go below level one

			uiTempLevel = (FLMUINT) (FSLEV_GET( pField));
			pStateInfo->uiJumpLevel = uiTempLevel;

			if (pStateInfo->uiFieldLevel <= uiTempLevel)
			{
				pStateInfo->bElmRecOK = FALSE;
				eCorruptionCode = FLM_BAD_ELM_FLD_LEVEL_JUMP;
				goto Exit;
			}

			pStateInfo->uiFieldLevel -= uiTempLevel;
		}
		else if (FOP_IS_RECORD_INFO( pField))
		{
			bFOPIsField = FALSE;
			pStateInfo->uiFOPType = FLM_FOP_REC_INFO;
			uiFldOverhead = 1;
			pStateInfo->uiFieldNum = 0;
			pStateInfo->uiFieldType = 0xFF;

			pTmpFld = pField + 1;
			uiBaseFldFlags = (FLMUINT) (FOP_GET_FLD_FLAGS( pField));

			if (FOP_2BYTE_FLDLEN( uiBaseFldFlags))
			{
				uiFldOverhead += 2;
				if (uiElmRecOffset + uiFldOverhead > uiElmRecLen)
				{
					pStateInfo->bElmRecOK = FALSE;
					eCorruptionCode = FLM_BAD_ELM_FLD_OVERHEAD;
					goto Exit;
				}

				pStateInfo->uiFieldLen = *pTmpFld++;
				pStateInfo->uiFieldLen += ((FLMUINT) * pTmpFld++) << 8;
			}
			else
			{
				uiFldOverhead++;
				if (uiElmRecOffset + uiFldOverhead > uiElmRecLen)
				{
					pStateInfo->bElmRecOK = FALSE;
					eCorruptionCode = FLM_BAD_ELM_FLD_OVERHEAD;
					goto Exit;
				}

				pStateInfo->uiFieldLen = *pTmpFld++;
			}
		}
		else if (FOP_IS_ENCRYPTED( pField))
		{
			FLMBOOL	bTagSz;
			FLMBOOL	bLenSz;
			FLMBOOL	bENumSz;
			FLMBOOL	bELenSz;

			pStateInfo->uiFOPType = FLM_FOP_ENCRYPTED;
			bFOPIsField = TRUE;

			uiFldOverhead = 2;

			if (uiElmRecOffset + uiFldOverhead > uiElmRecLen)
			{
				pStateInfo->bElmRecOK = FALSE;
				eCorruptionCode = FLM_BAD_ELM_FLD_OVERHEAD;
				goto Exit;
			}

			if (FENC_LEVEL( pField))
			{
				pStateInfo->uiFieldLevel++;
			}

			pStateInfo->uiFieldType = (FLMUINT) (FENC_FLD_TYPE( pField));
			bTagSz = FENC_TAG_SZ( pField);
			if (bTagSz)
			{
				uiFldOverhead += 2;
			}
			else
			{
				uiFldOverhead++;
			}

			bLenSz = FENC_LEN_SZ( pField);
			if (bLenSz)
			{
				uiFldOverhead += 2;
			}
			else
			{
				uiFldOverhead++;
			}

			bENumSz = FENC_ETAG_SZ( pField);
			if (bENumSz)
			{
				uiFldOverhead += 2;
			}
			else
			{
				uiFldOverhead++;
			}

			bELenSz = FENC_ELEN_SZ( pField);
			if (bELenSz)
			{
				uiFldOverhead += 2;
			}
			else
			{
				uiFldOverhead++;
			}

			if (uiElmRecOffset + uiFldOverhead > uiElmRecLen)
			{
				pStateInfo->bElmRecOK = FALSE;
				eCorruptionCode = FLM_BAD_ELM_FLD_OVERHEAD;
				goto Exit;
			}

			pTmpFld = pField + 2;

			pStateInfo->uiFieldNum = (FLMUINT) * pTmpFld++;
			if (bTagSz)
			{
				pStateInfo->uiFieldNum += ((FLMUINT) * pTmpFld++) << 8;
			}

			pStateInfo->uiFieldLen = (FLMUINT) * pTmpFld++;
			if (bLenSz)
			{
				pStateInfo->uiFieldLen += ((FLMUINT) * pTmpFld++) << 8;
			}

			pStateInfo->uiEncId = (FLMUINT) * pTmpFld++;
			if (bENumSz)
			{
				pStateInfo->uiEncId += ((FLMUINT) * pTmpFld++) << 8;
			}

			pStateInfo->uiEncFieldLen = (FLMUINT) * pTmpFld++;
			if (bELenSz)
			{
				pStateInfo->uiEncFieldLen += ((FLMUINT) * pTmpFld++) << 8;
			}
		}
		else if (FOP_IS_LARGE( pField))
		{
			if( pStateInfo->uiVersionNum < FLM_FILE_FORMAT_VER_4_61)
			{
				pStateInfo->uiFOPType = FLM_FOP_BAD;
				pStateInfo->bElmRecOK = FALSE;
				eCorruptionCode = FLM_BAD_ELM_FLD_OVERHEAD;
				goto Exit;
			}
			
			pStateInfo->uiFOPType = FLM_FOP_LARGE;
			bFOPIsField = TRUE;

			uiFldOverhead = 2;

			if (uiElmRecOffset + uiFldOverhead > uiElmRecLen)
			{
				pStateInfo->bElmRecOK = FALSE;
				eCorruptionCode = FLM_BAD_ELM_FLD_OVERHEAD;
				goto Exit;
			}

			if (FLARGE_LEVEL( pField))
			{
				pStateInfo->uiFieldLevel++;
			}

			pStateInfo->uiFieldType = FLARGE_FLD_TYPE( pField);

			pStateInfo->uiFieldNum = FLARGE_TAG_NUM( pField);
			uiFldOverhead += 2;

			pStateInfo->uiFieldLen = FLARGE_DATA_LEN( pField);
			uiFldOverhead += 4;

			if (FLARGE_ENCRYPTED( pField))
			{
				pStateInfo->uiEncId = FLARGE_ETAG_NUM( pField);
				uiFldOverhead += 2;

				pStateInfo->uiEncFieldLen = FLARGE_EDATA_LEN( pField);
				uiFldOverhead += 4;
			}

			if (uiElmRecOffset + uiFldOverhead > uiElmRecLen)
			{
				pStateInfo->bElmRecOK = FALSE;
				eCorruptionCode = FLM_BAD_ELM_FLD_OVERHEAD;
				goto Exit;
			}
		}
		else
		{

			// Anything else is a code we don't understand

			pStateInfo->uiFOPType = FLM_FOP_BAD;
			pStateInfo->bElmRecOK = FALSE;
			eCorruptionCode = FLM_BAD_ELM_FLD_OVERHEAD;
			goto Exit;
		}

		// If it is a field, get its type and make further checks

		if (bFOPIsField)
		{
			FLMUINT	uiFldNum = pStateInfo->uiFieldNum;

			if (pStateInfo->pLogicalFile)
			{
				pStateInfo->pLogicalFile->pLfStats->ui64FldRefCount++;
			}

			// If it is a dictionary field, verify that it is indeed and get
			// the field type from the dictionary.

			if (bDictField)
			{

				// If the field number is a reserved dictionary tag, it must
				// be a TEXT field. These are always stored in the FOP_OPEN
				// format - hence, the bDictField flag will be TRUE.

				if ((uiFldNum >= FLM_DICT_FIELD_NUMS) &&
					 (uiFldNum <= uiMaxDictFieldNum))
				{
					pStateInfo->uiFieldType = FLM_TEXT_TYPE;
				}

				// If we have no dictionary, set the field type to binary so
				// that the field will pass every test.

				else if (!pStateInfo->pDb)
				{
					pStateInfo->uiFieldType = FLM_BINARY_TYPE;
				}
				else
				{

					// If we can't find the field in the dictionary we have a
					// corruption.

					if (RC_BAD( fdictGetField( pStateInfo->pDb->pDict, uiFldNum,
								  &pStateInfo->uiFieldType, NULL, NULL)))
					{
						pStateInfo->bElmRecOK = FALSE;
						pStateInfo->uiFieldType = FLM_BINARY_TYPE;
						eCorruptionCode = FLM_BAD_ELM_FLD_NUM;

						// Keep processing and fill out the rest of the state
						// information. Even though we have a bad field number
						// here, the caller may want to simply skip the field
						// instead of aborting the entire element.

						goto Keep_Processing_Field;
					}
				}
			}

			// Check the field type

			switch (pStateInfo->uiFieldType)
			{
				case FLM_TEXT_TYPE:
				case FLM_NUMBER_TYPE:
				case FLM_BINARY_TYPE:
				case FLM_BLOB_TYPE:
				{
					break;
				}
				
				case FLM_CONTEXT_TYPE:
				{
					if (pStateInfo->uiFieldLen != 0 && pStateInfo->uiFieldLen != 4)
					{
						pStateInfo->bElmRecOK = FALSE;
						eCorruptionCode = FLM_BAD_ELM_FLD_LEN;
						goto Exit;
					}
					
					break;
				}
				
				default:
				{
					pStateInfo->bElmRecOK = FALSE;
					eCorruptionCode = FLM_BAD_ELM_FLD_TYPE;
					goto Exit;
				}
			}
		}

Keep_Processing_Field:

		// At this point, it is possible for us to have a bad field number
		// error, but we still want to set up pStateInfo so the caller can
		// simply skip the field if desired.

		uiElmRecOffset += uiFldOverhead;
		pStateInfo->uiFieldProcessedLen = 0;
	}

	// Setup the state structure to point to the data

	if (pStateInfo->uiEncId)
	{
		uiFOPDataLen = pStateInfo->uiEncFieldLen - pStateInfo->uiFieldProcessedLen;
	}
	else
	{
		uiFOPDataLen = pStateInfo->uiFieldLen - pStateInfo->uiFieldProcessedLen;
	}

	if (uiFOPDataLen > uiElmRecLen - uiElmRecOffset)
	{
		uiFOPDataLen = uiElmRecLen - uiElmRecOffset;
		if (BBE_IS_LAST( pElm))
		{
			eCorruptionCode = FLM_BAD_ELM_FLD_LEN;
			goto Exit;
		}
	}

	pStateInfo->uiFieldProcessedLen += uiFOPDataLen;
	pStateInfo->uiFOPDataLen = uiFOPDataLen;
	pStateInfo->pFOPData = &pElmRec[uiElmRecOffset];
	uiElmRecOffset += uiFOPDataLen;
	pStateInfo->uiElmRecOffset = uiElmRecOffset;

Exit:

	return (eCorruptionCode);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE flmVerifyIXRefs(
	STATE_INFO *		pStateInfo,
	IX_CHK_INFO *		pIxChkInfo,
	FLMUINT				uiResetDrn,
	eCorruptionType *	peElmCorruptionCode)
{
	FLMUINT		uiElmRecOffset = pStateInfo->uiElmRecOffset;
	FLMBYTE *	pElm = pStateInfo->pElm;
	FLMBYTE *	pElmRec = pStateInfo->pElmRec;
	FLMUINT		uiElmRecLen = pStateInfo->uiElmRecLen;
	FLMUINT		uiLowestDrn;
	FLMUINT		uiDrn;
	FLMUINT		uiTmpNum;
	FLMBYTE *	pTmpElmRec;
	FLMUINT		uiNumBytes;
	FLMUINT		uiElmRefs = 0;
	FLMBOOL		bOneRun;
	RCODE			rc = FERR_OK;

	*peElmCorruptionCode = FLM_NO_CORRUPTION;

	if ((BBE_IS_FIRST( pElm)) && (uiElmRecOffset == 0))
	{
		pStateInfo->uiCurrIxRefDrn = 0;
		pStateInfo->bElmRecOK = TRUE;
	}

	// Determine the element domain

	uiLowestDrn = 0;
	if (*pElmRec == 0xFC)
	{
		uiElmRecOffset++;
		if (!flmGetSEN( &pElmRec[uiElmRecOffset], &uiLowestDrn, &uiNumBytes))
		{
			pStateInfo->bElmRecOK = FALSE;
			*peElmCorruptionCode = FLM_BAD_ELM_DOMAIN_SEN;
			goto Exit;
		}

		uiElmRecOffset += uiNumBytes;
		uiLowestDrn <<= 8;
	}

	// Get the base DRN for the element

	if (!flmGetSEN( &pElmRec[uiElmRecOffset], &uiDrn, &uiNumBytes))
	{
		pStateInfo->bElmRecOK = FALSE;
		*peElmCorruptionCode = FLM_BAD_ELM_BASE_SEN;
		goto Exit;
	}

	uiElmRecOffset += uiNumBytes;

	// If this is the first element or the state info has not yet been
	// set, set the pStateInfo.

	if ((BBE_IS_FIRST( pElm)) || (!pStateInfo->uiCurrIxRefDrn))
	{
		pStateInfo->uiCurrIxRefDrn = uiDrn;
	}
	else if (uiDrn >= pStateInfo->uiCurrIxRefDrn)
	{

		// If the DRN's are not descending, we have a problem

		pStateInfo->bElmRecOK = FALSE;
		*peElmCorruptionCode = FLM_BAD_ELM_IX_REF;
		goto Exit;
	}

	uiElmRefs++;

	if (uiDrn <= uiResetDrn)
	{
		uiResetDrn = 0;
	}

	if (pIxChkInfo != NULL && !uiResetDrn)
	{
		pStateInfo->uiCurrIxRefDrn = uiDrn;
		if ((RC_BAD( rc = chkVerifyIXRSet( pStateInfo, pIxChkInfo, uiDrn))) ||
			 (pIxChkInfo->pDbInfo->bReposition))
		{
			goto Exit;
		}
	}

	while (uiElmRecOffset < uiElmRecLen)
	{
		pTmpElmRec = &pElmRec[uiElmRecOffset];

		// See if we have a one run

		bOneRun = FALSE;
		if (*pTmpElmRec >= 0xF0 && *pTmpElmRec <= 0xF7)
		{
			uiTmpNum = (FLMUINT) ((*pTmpElmRec & 0x0F) + 2);
			uiElmRefs += uiTmpNum;
			uiNumBytes = 1;
			bOneRun = TRUE;
		}
		else if (*pTmpElmRec == 0xF8)
		{
			if (!flmGetSEN( pTmpElmRec + 1, &uiTmpNum, &uiNumBytes))
			{
				pStateInfo->bElmRecOK = FALSE;
				*peElmCorruptionCode = FLM_BAD_ELM_ONE_RUN_SEN;
				goto Exit;
			}

			uiNumBytes++;
			uiElmRefs += uiTmpNum;
			bOneRun = TRUE;
		}
		else
		{

			// We have a delta

			if (!flmGetSEN( pTmpElmRec, &uiTmpNum, &uiNumBytes))
			{
				pStateInfo->bElmRecOK = FALSE;
				*peElmCorruptionCode = FLM_BAD_ELM_DELTA_SEN;
				goto Exit;
			}

			uiElmRefs++;
		}

		// The new drn must not take us zero or negative

		if (uiDrn <= uiTmpNum)
		{
			pStateInfo->bElmRecOK = FALSE;
			*peElmCorruptionCode = FLM_BAD_ELM_IX_REF;
			goto Exit;
		}

		if (bOneRun)
		{
			while (uiTmpNum > 0)
			{
				uiTmpNum--;
				uiDrn--;

				if (uiDrn <= uiResetDrn)
				{
					uiResetDrn = 0;
				}

				pStateInfo->uiCurrIxRefDrn = uiDrn;
				if (pIxChkInfo != NULL && !uiResetDrn)
				{

					// Verify that the key+ref is in the result set

					if ((RC_BAD( rc = chkVerifyIXRSet( pStateInfo, pIxChkInfo,
									  uiDrn))) || pIxChkInfo->pDbInfo->bReposition)
					{
						goto Exit;
					}
				}
			}
		}
		else
		{
			uiDrn -= uiTmpNum;

			if (uiDrn <= uiResetDrn)
			{
				uiResetDrn = 0;
			}

			pStateInfo->uiCurrIxRefDrn = uiDrn;
			if (pIxChkInfo != NULL && !uiResetDrn)
			{

				// Verify that the key+ref is in the result set

				if ((RC_BAD( rc = chkVerifyIXRSet( pStateInfo, pIxChkInfo, 
					uiDrn))) || pIxChkInfo->pDbInfo->bReposition)
				{
					goto Exit;
				}
			}
		}

		uiElmRecOffset += uiNumBytes;
	}

	// If we didn't end up right at the end of the element, we have
	// corruption.

	if (uiElmRecOffset != uiElmRecLen)
	{
		pStateInfo->bElmRecOK = FALSE;
		*peElmCorruptionCode = FLM_BAD_ELM_END;
		goto Exit;
	}

	// The last drn must not be lower than the lowest Drn for the element

	if (uiDrn < uiLowestDrn)
	{
		pStateInfo->bElmRecOK = FALSE;
		*peElmCorruptionCode = FLM_BAD_ELM_DOMAIN;
		goto Exit;
	}

	pStateInfo->uiCurrIxRefDrn = uiDrn;
	pStateInfo->uiElmRecOffset = uiElmRecOffset;

	if (pStateInfo->pLogicalFile)
	{
		pStateInfo->pLogicalFile->pLfStats->ui64FldRefCount += (FLMUINT64) uiElmRefs;
	}

	pStateInfo->ui64KeyRefs += (FLMUINT64) uiElmRefs;

	// Check for a non-unique reference in a unique key

	if ((pStateInfo->pLogicalFile) &&
		 (pStateInfo->pLogicalFile->pIxd->uiFlags & IXD_UNIQUE) &&
		 (pStateInfo->ui64KeyRefs > (FLMUINT64) 1))
	{
		pStateInfo->bElmRecOK = FALSE;

		if (pIxChkInfo != NULL)
		{

			// Give the application the option of deleting the record to
			// resolve the corruption.

			if (RC_BAD( rc = chkResolveNonUniqueKey( pStateInfo, pIxChkInfo,
						  pStateInfo->pLogicalFile->pLFile->uiLfNum,
						  pStateInfo->pCurKey, pStateInfo->uiCurKeyLen,
						  pStateInfo->uiCurrIxRefDrn)))
			{
				if (rc == FERR_OLD_VIEW)
				{

					// Set uiCurrIxRefDrn to zero so that the check will
					// re-position to the first reference of the current key.

					pStateInfo->uiCurrIxRefDrn = 0;
				}

				goto Exit;
			}
		}

		*peElmCorruptionCode = FLM_NON_UNIQUE_ELM_KEY_REF;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMBOOL flmGetSEN(
	FLMBYTE *		pTmpElmRec,
	FLMUINT *		puiDrnRV,
	FLMUINT *		puiNumBytesRV)
{
	FLMUINT			uiNumBytes;
	FLMUINT			uiDrn;
	FLMUINT			uiChar = (FLMUINT) *pTmpElmRec;
	FLMUINT			uiTempDrn;

	pTmpElmRec++;
	if (!(uiChar & 0x80))
	{
		uiNumBytes = 0;
		uiDrn = (FLMUINT) uiChar;
	}
	else if ((uiChar & 0xC0) == 0x80)
	{
		uiNumBytes = 1;
		uiDrn = (FLMUINT) (uiChar & 0x3F);
	}
	else if ((uiChar & 0xF0) == 0xC0)
	{
		uiNumBytes = 2;
		uiDrn = (FLMUINT) (uiChar & 0x0F);
	}
	else if ((uiChar & 0xF0) == 0xD0)
	{
		uiNumBytes = 3;
		uiDrn = (FLMUINT) (uiChar & 0x0F);
	}
	else if ((uiChar & 0xF0) == 0xE0)
	{
		uiNumBytes = 4;
		uiDrn = (FLMUINT) (uiChar & 0x0F);
	}
	else
	{
		return (FALSE);
	}

	*puiNumBytesRV = uiNumBytes + 1;

	while (uiNumBytes)
	{

		// Check for overflow

		uiTempDrn = (FLMUINT) (*pTmpElmRec);
		if (0xFFFFFFFF - uiDrn < (FLMUINT) 256 + uiTempDrn)
		{
			return (FALSE);
		}

		uiDrn <<= 8;
		uiDrn += uiTempDrn;
		pTmpElmRec++;
		uiNumBytes--;
	}

	*puiDrnRV = uiDrn;
	return (TRUE);
}

/****************************************************************************
Desc:
****************************************************************************/
void flmInitReadState(
	STATE_INFO *	pStateInfo,
	FLMBOOL *		pbStateInitialized,
	FLMUINT			uiVersionNum,
	FDB *				pDb,	// May be NULL.
	LF_HDR *			pLogicalFile,
	FLMUINT			uiLevel,
	FLMUINT			uiBlkType,
	FLMBYTE *		pKeyBuffer)
{
	if ((*pbStateInitialized) && (pStateInfo->pRecord))
	{
		pStateInfo->pRecord->Release();
		pStateInfo->pRecord = NULL;
	}

	f_memset( pStateInfo, 0, sizeof(STATE_INFO));
	*pbStateInitialized = TRUE;
	pStateInfo->uiVersionNum = uiVersionNum;
	pStateInfo->pDb = pDb;
	pStateInfo->pLogicalFile = pLogicalFile;
	pStateInfo->uiLevel = uiLevel;

	// Special cases for leaf and non-leaf blocks

	if (uiBlkType == BHT_LEAF)
	{
		pStateInfo->uiElmOvhd = BBE_KEY;
	}
	else if (uiBlkType == BHT_NON_LEAF)
	{
		if (pLogicalFile)
		{
			if (pLogicalFile->pLFile->uiLfType == LF_INDEX)
			{
				if (pLogicalFile->pIxd &&
					 (pLogicalFile->pIxd->uiFlags & IXD_POSITIONING))
				{
					uiBlkType = BHT_NON_LEAF_COUNTS;
				}
				else
				{
					uiBlkType = BHT_NON_LEAF;
				}
			}
			else
			{
				uiBlkType = BHT_NON_LEAF_DATA;
			}
		}
	}

	if (uiBlkType == BHT_NON_LEAF_DATA)
	{
		pStateInfo->uiElmOvhd = BNE_DATA_OVHD;
	}
	else if (uiBlkType == BHT_NON_LEAF)
	{
		pStateInfo->uiElmOvhd = BNE_KEY_START;
	}
	else if (uiBlkType == BHT_NON_LEAF_COUNTS)
	{
		pStateInfo->uiElmOvhd = BNE_KEY_COUNTS_START;
	}

	pStateInfo->uiBlkType = uiBlkType;
	pStateInfo->pCurKey = pKeyBuffer;
	pStateInfo->uiElmLastFlag = 0xFF;
	pStateInfo->uiFieldType = 0xFF;
}
