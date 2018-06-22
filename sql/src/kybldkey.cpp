//------------------------------------------------------------------------------
// Desc:	Build from and until keys from a predicate
// Tabs:	3
//
// Copyright (c) 1996-2007 Novell, Inc. All Rights Reserved.
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

FSTATIC RCODE flmAddNonTextKeyPiece(
	SQL_PRED *			pPred,
	F_INDEX *			pIndex,
	FLMUINT				uiKeyComponent,
	ICD *					pIcd,
	F_COLUMN *			pColumn,
	F_DataVector *		pFromSearchKey,
	FLMBYTE *			pucFromKey,
	FLMUINT *			puiFromKeyLen,
	F_DataVector *		pUntilSearchKey,
	FLMBYTE *			pucUntilKey,
	FLMUINT *			puiUntilKeyLen,
	FLMBOOL *			pbCanCompareOnKey,
	FLMBOOL *			pbFromIncl,
	FLMBOOL *			pbUntilIncl);

FSTATIC RCODE flmUTF8FindWildcard(
	const FLMBYTE *	pucValue,
	FLMUINT *			puiCharPos,
	FLMUINT *			puiCompareRules);

FSTATIC RCODE flmCountCharacters(
	const FLMBYTE * 	pucValue,
	FLMUINT				uiValueLen,
	FLMUINT				uiMaxToCount,			
	FLMUINT *			puiCompareRules,
	FLMUINT *			puiCount);

FSTATIC RCODE flmSelectBestSubstr(
	const FLMBYTE **	ppucValue,
	FLMUINT *			puiValueLen,
	FLMUINT *			puiCompareRules,
	FLMBOOL *			pbTrailingWildcard,
	FLMBOOL *			pbNotUsingFirstOfString);

FSTATIC void setFromCaseByte(
	FLMBYTE *			pucFromKey,
	FLMUINT *			puiFromComponentLen,
	FLMUINT				uiCaseLen,
	FLMBOOL				bIsDBCS,
	FLMBOOL				bAscending,
	FLMBOOL				bExcl);
	
FSTATIC void setUntilCaseByte(
	FLMBYTE *			pucUntilKey,
	FLMUINT *			puiUntilComponentLen,
	FLMUINT				uiCaseLen,
	FLMBOOL				bIsDBCS,
	FLMBOOL				bAscending,
	FLMBOOL				bExcl);
	
FSTATIC RCODE flmAddTextKeyPiece(
	SQL_PRED *			pPred,
	F_INDEX *			pIndex,
	FLMUINT				uiKeyComponent,
	ICD *					pIcd,
	F_DataVector *		pFromSearchKey,
	FLMBYTE *			pucFromKey,
	FLMUINT *			puiFromKeyLen,
	F_DataVector *		pUntilSearchKey,
	FLMBYTE *			pucUntilKey,
	FLMUINT *			puiUntilKeyLen,
	FLMBOOL *			pbCanCompareOnKey,
	FLMBOOL *			pbFromIncl,
	FLMBOOL *			pbUntilIncl);

/****************************************************************************
Desc:		Add a key piece to the from and until key.  Text fields are not 
			handled in this routine because of their complexity.
Notes:	The goal of this code is to build a the collated compound piece
			for the 'from' and 'until' key only once instead of twice.
****************************************************************************/
FSTATIC RCODE flmAddNonTextKeyPiece(
	SQL_PRED *		pPred,
	F_INDEX *		pIndex,
	FLMUINT			uiKeyComponent,
	ICD *				pIcd,
	F_COLUMN *		pColumn,
	F_DataVector *	pFromSearchKey,
	FLMBYTE *		pucFromKey,
	FLMUINT *		puiFromKeyLen,
	F_DataVector *	pUntilSearchKey,
	FLMBYTE *		pucUntilKey,
	FLMUINT *		puiUntilKeyLen,
	FLMBOOL *		pbCanCompareOnKey,
	FLMBOOL *		pbFromIncl,
	FLMBOOL *		pbUntilIncl)
{
	RCODE						rc = NE_SFLM_OK;
	FLMBYTE *				pucFromKeyLenPos;
	FLMBYTE *				pucUntilKeyLenPos;
	FLMBOOL					bDataTruncated;
	FLMBYTE *				pucFromBuf;
	FLMUINT					uiFromBufLen;
	FLMBYTE *				pucUntilBuf;
	FLMUINT					uiUntilBufLen;
	FLMBYTE					ucFromNumberBuf [FLM_MAX_NUM_BUF_SIZE];
	FLMBYTE					ucUntilNumberBuf [FLM_MAX_NUM_BUF_SIZE];
	FLMUINT					uiValue;
	FLMINT					iValue;
	FLMBOOL					bNeg;
	FLMUINT64				ui64Value;
	FLMINT64					i64Value;
	FLMUINT					uiFromFlags = 0;
	FLMUINT					uiUntilFlags = 0;
	SQL_VALUE *				pFromValue;
	SQL_VALUE *				pUntilValue;
	FLMBOOL					bInclFrom;
	FLMBOOL					bInclUntil;
	FLMBOOL					bAscending = (pIcd->uiFlags & ICD_DESCENDING) ? FALSE: TRUE;
	F_BufferIStream		bufferIStream;
	FLMUINT					uiFromKeyLen = *puiFromKeyLen;
	FLMUINT					uiUntilKeyLen = *puiUntilKeyLen;
	FLMUINT					uiFromComponentLen;
	FLMUINT					uiUntilComponentLen;
	FLMUINT					uiFromSpaceLeft;
	FLMUINT					uiUntilSpaceLeft;
	
	// Leave room for the component length

	pucFromKeyLenPos = pucFromKey + uiFromKeyLen;
	pucUntilKeyLenPos = pucUntilKey + uiUntilKeyLen;
	pucFromKey = pucFromKeyLenPos + 2;
	pucUntilKey = pucUntilKeyLenPos + 2;
	uiFromSpaceLeft = SFLM_MAX_KEY_SIZE - uiFromKeyLen - 2;
	uiUntilSpaceLeft = SFLM_MAX_KEY_SIZE - uiUntilKeyLen - 2;

	// Handle the presence case here - this is not done in kyCollate.

	if (pIcd->uiFlags & ICD_PRESENCE)
	{
		
		// If we don't have enough space for the component in the from
		// key, just get all values.
		
		if (uiFromSpaceLeft < 4)
		{
			uiFromComponentLen = KEY_LOW_VALUE;
		}
		else
		{
			f_UINT32ToBigEndian( (FLMUINT32)pIcd->uiColumnNum, pucFromKey);
			uiFromComponentLen = 4;
		}
		
		// If we don't have enough space for the component in the until
		// key, just get all values.
		
		if (uiUntilSpaceLeft < 4)
		{
			uiUntilComponentLen = KEY_HIGH_VALUE;
		}
		else
		{
			f_UINT32ToBigEndian( (FLMUINT32)pIcd->uiColumnNum, pucUntilKey);
			uiUntilComponentLen = 4;
		}
	}
	else if (pIcd->uiFlags & ICD_METAPHONE)
	{
		if (pPred->eOperator != SQL_APPROX_EQ_OP ||
			 pPred->pFromValue->eValType != SQL_UTF8_VAL)
		{
			uiFromComponentLen = KEY_LOW_VALUE;
			uiUntilComponentLen = KEY_HIGH_VALUE;
			if (pPred->eOperator != SQL_EXISTS_OP)
			{
				*pbCanCompareOnKey = FALSE;
			}
		}
		else
		{
			*pbCanCompareOnKey = FALSE;
	
			// The value type in pPred->pFromValue is SQL_UTF8_VAL, but the
			// calling routine should have put the metaphone value into
			// pPred->pFromValue->val.uiVal.  Sort of weird, but was the
			// only way we could evaluate the cost of multiple words in
			// the string.
	
			uiFromBufLen = sizeof( ucFromNumberBuf);
			if( RC_BAD( rc = FlmUINT2Storage( pPred->pFromValue->val.uiVal,
									&uiFromBufLen, ucFromNumberBuf)))
			{
				goto Exit;
			}
			pucFromBuf = &ucFromNumberBuf [0];
			
			if (RC_BAD( rc = bufferIStream.openStream( 
				(const char *)pucFromBuf, uiFromBufLen)))
			{
				goto Exit;
			}
	
			uiFromComponentLen = uiFromSpaceLeft;
			bDataTruncated = FALSE;
			
			// Pass 0 for compare rules because it is non-text
			
			if (RC_BAD( rc = KYCollateValue( pucFromKey, &uiFromComponentLen,
									&bufferIStream, SFLM_NUMBER_TYPE,
									pIcd->uiFlags, 0,
									pIcd->uiLimit, NULL, NULL, 
									pIndex->uiLanguage, FALSE, FALSE,
									&bDataTruncated, NULL)))
			{
				goto Exit;
			}
			
			bufferIStream.closeStream();
			
			// Key component needs to fit in both the from and until
			// buffers.  If it will not, we simply set it up to search
			// from first to last - all values.
			
			if (bDataTruncated || uiUntilSpaceLeft < uiFromComponentLen)
			{
				uiFromComponentLen = KEY_LOW_VALUE;
				uiUntilComponentLen = KEY_HIGH_VALUE;
				*pbCanCompareOnKey = FALSE;
			}
			else
			{
				if ((uiUntilComponentLen = uiFromComponentLen) != 0)
				{
					f_memcpy( pucUntilKey, pucFromKey, uiFromComponentLen);
				}
			}
		}
	}
	else if (pPred->eOperator == SQL_EXISTS_OP ||
				pPred->eOperator == SQL_NE_OP ||
				pPred->eOperator == SQL_APPROX_EQ_OP)
	{
		
		// Setup a first-to-last key

		uiFromComponentLen = KEY_LOW_VALUE;
		uiUntilComponentLen = KEY_HIGH_VALUE;
	}
	else
	{

		// Only other operator possible is the range operator

		flmAssert( pPred->eOperator == SQL_RANGE_OP);
		
		if (bAscending)
		{
			pFromValue = pPred->pFromValue;
			bInclFrom = pPred->bInclFrom;
			pUntilValue = pPred->pUntilValue;
			bInclUntil = pPred->bInclUntil;
		}
		else
		{
			pFromValue = pPred->pUntilValue;
			bInclFrom = pPred->bInclUntil;
			pUntilValue = pPred->pFromValue;
			bInclUntil = pPred->bInclFrom;
		}
		
		// Set up from buffer

		if (!pFromValue)
		{
			pucFromBuf = NULL;
			uiFromBufLen = 0;
		}
		else
		{
			switch (pFromValue->eValType)
			{
				case SQL_UINT_VAL:
					uiValue = pFromValue->val.uiVal;
					if (!bInclFrom)
					{
						if (bAscending)
						{
							uiValue++;
						}
						else
						{
							uiValue--;
						}
					}
					uiFromBufLen = sizeof( ucFromNumberBuf);
					if( RC_BAD( rc = FlmUINT2Storage( uiValue, &uiFromBufLen,
											ucFromNumberBuf)))
					{
						goto Exit;
					}
					pucFromBuf = &ucFromNumberBuf [0];
					break;

				case SQL_INT_VAL:
					iValue = pFromValue->val.iVal;
					if (!bInclFrom)
					{
						if (bAscending)
						{
							iValue++;
						}
						else
						{
							iValue--;
						}
					}
					uiFromBufLen = sizeof( ucFromNumberBuf);
					if (RC_BAD( rc = FlmINT2Storage( iValue, &uiFromBufLen,
											ucFromNumberBuf)))
					{
						goto Exit;
					}
					pucFromBuf = &ucFromNumberBuf [0];
					break;

				case SQL_UINT64_VAL:
					ui64Value = pFromValue->val.ui64Val;
					if (!bInclFrom)
					{
						if (bAscending)
						{
							ui64Value++;
						}
						else
						{
							ui64Value--;
						}
					}
					uiFromBufLen = sizeof( ucFromNumberBuf);
					if (RC_BAD( rc = flmNumber64ToStorage( ui64Value, &uiFromBufLen,
						ucFromNumberBuf, FALSE, FALSE)))
					{
						goto Exit;
					}
					pucFromBuf = &ucFromNumberBuf [0];
					break;

				case SQL_INT64_VAL:
					i64Value = pFromValue->val.i64Val;
					if (!bInclFrom)
					{
						if (bAscending)
						{
							i64Value++;
						}
						else
						{
							i64Value--;
						}
					}
					if (i64Value < 0)
					{
						bNeg = TRUE;
						ui64Value = (FLMUINT64)-i64Value;
					}
					else
					{
						bNeg = FALSE;
						ui64Value = (FLMUINT64)i64Value;
					}

					uiFromBufLen = sizeof( ucFromNumberBuf);
					if (RC_BAD( rc = flmNumber64ToStorage( ui64Value, &uiFromBufLen,
						ucFromNumberBuf, bNeg, FALSE)))
					{
						goto Exit;
					}
					pucFromBuf = &ucFromNumberBuf [0];
					break;

				case SQL_BINARY_VAL:
					pucFromBuf = pFromValue->val.bin.pucValue;
					uiFromBufLen = pFromValue->val.bin.uiByteLen;
					if (!bInclFrom)
					{
						
						// Should use EXCLUSIVE_GT_FLAG even if in descending
						// order, because the comparison routines will take
						// that into account.
						
						uiFromFlags |= EXCLUSIVE_GT_FLAG;
					}
					break;

				default:

					// Text type should have been taken care of elsewhere.

					rc = RC_SET_AND_ASSERT( NE_SFLM_QUERY_SYNTAX);
					goto Exit;
			}
		}

		// Set up until buffer.

		if (!pUntilValue)
		{
			pucUntilBuf = NULL;
			uiUntilBufLen = 0;
		}
		else if (pUntilValue == pFromValue)
		{
			pucUntilBuf = pucFromBuf;
			uiUntilBufLen = uiFromBufLen;
		}
		else
		{
			switch (pUntilValue->eValType)
			{
				case SQL_UINT_VAL:
					uiValue = pUntilValue->val.uiVal;
					if (!bInclUntil)
					{
						if (bAscending)
						{
							uiValue--;
						}
						else
						{
							uiValue++;
						}
					}
					uiUntilBufLen = sizeof( ucUntilNumberBuf);
					if( RC_BAD( rc = FlmUINT2Storage( uiValue, &uiUntilBufLen,
											ucUntilNumberBuf)))
					{
						goto Exit;
					}
					pucUntilBuf = &ucUntilNumberBuf [0];
					break;

				case SQL_INT_VAL:
					iValue = pUntilValue->val.iVal;
					if (!bInclUntil)
					{
						if (bAscending)
						{
							iValue--;
						}
						else
						{
							iValue++;
						}
					}
					uiUntilBufLen = sizeof( ucUntilNumberBuf);
					if (RC_BAD( rc = FlmINT2Storage( iValue, &uiUntilBufLen,
											ucUntilNumberBuf)))
					{
						goto Exit;
					}
					pucUntilBuf = &ucUntilNumberBuf [0];
					break;

				case SQL_UINT64_VAL:
					ui64Value = pUntilValue->val.ui64Val;
					if (!bInclUntil)
					{
						if (bAscending)
						{
							ui64Value--;
						}
						else
						{
							ui64Value++;
						}
					}
					uiUntilBufLen = sizeof( ucUntilNumberBuf);
					if (RC_BAD( rc = flmNumber64ToStorage( ui64Value, &uiUntilBufLen,
						ucUntilNumberBuf, FALSE, FALSE)))
					{
						goto Exit;
					}
					pucUntilBuf = &ucUntilNumberBuf [0];
					break;

				case SQL_INT64_VAL:
					i64Value = pUntilValue->val.i64Val;
					if (!bInclUntil)
					{
						if (bAscending)
						{
							i64Value--;
						}
						else
						{
							i64Value++;
						}
					}
					if (i64Value < 0)
					{
						bNeg = TRUE;
						ui64Value = (FLMUINT64)-i64Value;
					}
					else
					{
						bNeg = FALSE;
						ui64Value = (FLMUINT64)i64Value;
					}

					uiUntilBufLen = sizeof( ucUntilNumberBuf);
					if (RC_BAD( rc = flmNumber64ToStorage( ui64Value, &uiUntilBufLen,
						ucUntilNumberBuf, bNeg, FALSE)))
					{
						goto Exit;
					}
					pucUntilBuf = &ucUntilNumberBuf [0];
					break;

				case SQL_BINARY_VAL:
					pucUntilBuf = pUntilValue->val.bin.pucValue;
					uiUntilBufLen = pUntilValue->val.bin.uiByteLen;
					if (!bInclUntil)
					{
						
						// Should use EXCLUSIVE_LT_FLAG even if in descending
						// order, because the comparison routines will take
						// that into account.
						
						uiUntilFlags |= EXCLUSIVE_LT_FLAG;
					}
					break;

				default:

					// Text type should have been taken care of elsewhere.

					rc = RC_SET_AND_ASSERT( NE_SFLM_QUERY_SYNTAX);
					goto Exit;
			}
		}

		// Generate the keys using the from and until buffers that
		// have been set up.

		// Set up the from key
		
		if (!pucFromBuf)
		{
			uiFromComponentLen = KEY_LOW_VALUE;
		}
		else
		{
			if (RC_BAD( rc = bufferIStream.openStream( 
				(const char *)pucFromBuf, uiFromBufLen)))
			{
				goto Exit;
			}

			uiFromComponentLen = uiFromSpaceLeft;
			bDataTruncated = FALSE;
			
			// Pass 0 for compare rules on non-text component.
			
			if (RC_BAD( rc = KYCollateValue( pucFromKey, &uiFromComponentLen,
									&bufferIStream, pColumn->eDataTyp,
									pIcd->uiFlags, 0,
									pIcd->uiLimit, NULL, NULL, 
									pIndex->uiLanguage, FALSE, FALSE,
									&bDataTruncated, NULL)))
			{
				goto Exit;
			}
			
			bufferIStream.closeStream();

			if (bDataTruncated)
			{
				*pbCanCompareOnKey = FALSE;
				if (pFromValue->eValType != SQL_BINARY_VAL)
				{
					
					// Couldn't fit the key into the remaining buffer space, so
					// we will just ask for all values.
					
					uiFromComponentLen = KEY_LOW_VALUE;
				}
				else
				{
				
					// Save the original data into pFromSearchKey so the comparison
					// routines can do a comparison on the full value if
					// necessary.
				
					if (RC_BAD( rc = pFromSearchKey->setBinary( uiKeyComponent,
												pucFromBuf, uiFromBufLen)))
					{
						goto Exit;
					}
					uiFromFlags |= (SEARCH_KEY_FLAG | TRUNCATED_FLAG);
				}
			}
		}

		// Set up the until key

		if (!pucUntilBuf)
		{
			uiUntilComponentLen = KEY_HIGH_VALUE;
		}
		
		// Little optimization here if both the from and until are pointing
		// to the same data - it is an EQ operator, and we don't need
		// to do the collation again if we have room in the until buffer
		// and we didn't truncate the from key.
		
		else if (pucUntilBuf == pucFromBuf &&
					uiFromComponentLen != KEY_LOW_VALUE &&
					uiUntilSpaceLeft >= uiFromComponentLen &&
					!(uiFromFlags & TRUNCATED_FLAG))
		{
			
			// If the truncated flag is not set in the from key, neither
			// should the search key flag be set.
			
			flmAssert( !(uiFromFlags & SEARCH_KEY_FLAG));
			if ((uiUntilComponentLen = uiFromComponentLen) != 0)
			{
				f_memcpy( pucUntilKey, pucFromKey, uiFromComponentLen);
			}
			
			// The "exclusive" flags better not have been set in this
			// case - because this should only be possible if the operator
			// was an EQ.
			
			flmAssert( !(uiFromFlags & EXCLUSIVE_GT_FLAG) &&
						  !(uiUntilFlags & EXCLUSIVE_LT_FLAG));
		}
		else
		{
			if (RC_BAD( rc = bufferIStream.openStream( 
				(const char *)pucUntilBuf, uiUntilBufLen)))
			{
				goto Exit;
			}
			uiUntilComponentLen = uiUntilSpaceLeft;
			bDataTruncated = FALSE;
			
			// Pass 0 for compare rule because it is a non-text piece.
			
			if (RC_BAD( rc = KYCollateValue( pucUntilKey, &uiUntilComponentLen,
									&bufferIStream, pColumn->eDataTyp,
									pIcd->uiFlags, 0,
									pIcd->uiLimit, NULL, NULL, 
									pIndex->uiLanguage, FALSE, FALSE,
									&bDataTruncated, NULL)))
			{
				goto Exit;
			}
			
			bufferIStream.closeStream();

			if (bDataTruncated)
			{
				*pbCanCompareOnKey = FALSE;
				if (pUntilValue->eValType != SQL_BINARY_VAL)
				{
					
					// Couldn't fit the key into the remaining buffer space, so
					// we will just ask for all values.
					
					uiUntilComponentLen = KEY_HIGH_VALUE;
				}
				else
				{
				
					// Save the original data into pUntilSearchKey so the comparison
					// routines can do a comparison on the full value if
					// necessary.
	
					if (RC_BAD( rc = pUntilSearchKey->setBinary( uiKeyComponent,
												pucUntilBuf, uiUntilBufLen)))
					{
						goto Exit;
					}
				}
				uiUntilFlags |= (SEARCH_KEY_FLAG | TRUNCATED_FLAG);
			}
		}
	}

	UW2FBA( (FLMUINT16)(uiFromComponentLen | uiFromFlags), pucFromKeyLenPos);
	UW2FBA( (FLMUINT16)(uiUntilComponentLen | uiUntilFlags), pucUntilKeyLenPos);
	
	// Set the FROM and UNTIL key length return values.

	uiFromKeyLen += 2;
	if (uiFromComponentLen != KEY_LOW_VALUE)
	{
		uiFromKeyLen += uiFromComponentLen;
		if (!(uiFromFlags & EXCLUSIVE_GT_FLAG))
		{
			*pbFromIncl = TRUE;
		}
	}
	uiUntilKeyLen += 2;
	if (uiUntilComponentLen != KEY_HIGH_VALUE)
	{
		uiUntilKeyLen += uiUntilComponentLen;
		if (!(uiUntilFlags & EXCLUSIVE_LT_FLAG))
		{
			*pbUntilIncl = TRUE;
		}
	}
	
Exit:

	*puiFromKeyLen = uiFromKeyLen;
	*puiUntilKeyLen = uiUntilKeyLen;

	return( rc);
}

/****************************************************************************
Desc:	Finds the location of a wildcard in the internal text string, if any.
****************************************************************************/
FSTATIC RCODE flmUTF8FindWildcard(
	const FLMBYTE *	pucValue,
	FLMUINT *			puiCharPos,
	FLMUINT *			puiCompareRules)
{
	RCODE					rc = NE_SFLM_OK;
	const FLMBYTE *	pucSaveVal;
	const FLMBYTE *	pucStart = pucValue;
	FLMUNICODE			uzChar;
	FLMUINT				uiCompareRules = *puiCompareRules;

	flmAssert( pucValue);
	*puiCharPos = FLM_MAX_UINT;
	
	for( ;;)
	{
		pucSaveVal = pucValue;
		if (RC_BAD( rc = f_getCharFromUTF8Buf( &pucValue, NULL, &uzChar)))
		{
			goto Exit;
		}

		if (!uzChar)
		{
			break;
		}
		
		if ((uzChar = f_convertChar( uzChar, uiCompareRules)) == 0)
		{
			continue;
		}
		if (uzChar == ASCII_WILDCARD)
		{
			*puiCharPos = (FLMUINT)(pucSaveVal - pucStart);
			goto Exit;
		}
		if (uzChar != ASCII_SPACE)
		{
			
			// Once we hit a non-space character - except for the wildcard,
			// we can remove the ignore leading space rule.
			
			uiCompareRules &= (~(FLM_COMP_IGNORE_LEADING_SPACE));
			if (uzChar == ASCII_BACKSLASH)
			{
	
				// Skip the escaped character
	
				if (RC_BAD( rc = f_getCharFromUTF8Buf( &pucValue, NULL, &uzChar)))
				{
					goto Exit;
				}
	
				if (!uzChar)
				{
					rc = RC_SET( NE_SFLM_Q_BAD_SEARCH_STRING);
					goto Exit;
				}
			}
		}
	}

Exit:

	*puiCompareRules = uiCompareRules;

	return( rc);
}

/****************************************************************************
Desc:	Count the number of characters that would be returned.
****************************************************************************/
FSTATIC RCODE flmCountCharacters(
	const FLMBYTE * 	pucValue,
	FLMUINT				uiValueLen,
	FLMUINT				uiMaxToCount,			
	FLMUINT *			puiCompareRules,
	FLMUINT *			puiCharCount)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiNumChars = 0;
	FLMUINT				uiCompareRules = *puiCompareRules;
	const FLMBYTE *	pucEnd = &pucValue [uiValueLen];
	FLMUNICODE			uzChar;
	FLMBOOL				bLastCharWasSpace = FALSE;
	FLMUINT				uiNumSpaces = 0;

	while (uiNumChars < uiMaxToCount)
	{
		if (RC_BAD( rc = f_getCharFromUTF8Buf( &pucValue, pucEnd, &uzChar)))
		{
			goto Exit;
		}

		if (!uzChar)
		{
			if (bLastCharWasSpace)
			{
				// The spaces are trailing spaces, so if the ignore trailing
				// space flag is set, we do nothing.  If the compress space
				// flag is set, we will increment by one.  Otherwise, we will
				// add in a count for all of the spaces.
				
				if (!(uiCompareRules & FLM_COMP_IGNORE_TRAILING_SPACE))
				{
					if (uiCompareRules & FLM_COMP_COMPRESS_WHITESPACE)
					{
						uiNumChars++;
					}
					else
					{
						uiNumChars += uiNumSpaces;
					}
				}
			}
			break;
		}
		
		if ((uzChar = f_convertChar( uzChar, uiCompareRules)) == 0)
		{
			continue;
		}

		if (uzChar == ASCII_SPACE)
		{
			if (!bLastCharWasSpace)
			{
				bLastCharWasSpace = TRUE;
				uiNumSpaces = 0;
			}
			uiNumSpaces++;
		}
		else
		{
			
			// Once we hit a non-space character, disable the ignore
			// leading space flag.
			
			uiCompareRules &= (~(FLM_COMP_IGNORE_LEADING_SPACE));
			if (bLastCharWasSpace)
			{
				bLastCharWasSpace = FALSE;
				if (uiCompareRules & FLM_COMP_COMPRESS_WHITESPACE)
				{
					
					// Consecutive spaces are compressed to a single space.
					
					uiNumChars++;
				}
				else
				{
					
					// The spaces were not trailing spaces and were not compressed
					// so we need to count all of them.
					
					uiNumChars += uiNumSpaces;
				}
			}
			if (uzChar == ASCII_BACKSLASH)
			{
	
				// Skip the next character, no matter what it is - only want
				// to count one character here.  A backslash followed by any
				// character is only a single character.
	
				if (RC_BAD( rc = f_getCharFromUTF8Buf( &pucValue, pucEnd, &uzChar)))
				{
					goto Exit;
				}
			}
			uiNumChars++;
		}
	}

Exit:

	*puiCharCount = uiNumChars;
	*puiCompareRules = uiCompareRules;
	return( rc);
}

/****************************************************************************
Desc:	Select the best substring for a CONTAINS or MATCH_END search.
****************************************************************************/
FSTATIC RCODE flmSelectBestSubstr(
	const FLMBYTE **	ppucValue,				// [in/out]
	FLMUINT *			puiValueLen,			// [in/out]
	FLMUINT *			puiCompareRules,
	FLMBOOL *			pbTrailingWildcard,	// [in] change if found a wildcard
	FLMBOOL *			pbNotUsingFirstOfString)
{
	RCODE					rc = NE_SFLM_OK;
	const FLMBYTE *	pucValue = *ppucValue;
	const FLMBYTE *	pucCurValue;
	const FLMBYTE *	pucBest;
	const FLMBYTE *	pucEnd;
	const FLMBYTE *	pucTmp;
	FLMBOOL				bBestTerminatesWithWildCard = *pbTrailingWildcard;
	FLMUINT				uiCurLen;
	FLMUINT				uiBestNumChars;
	FLMUINT				uiBestValueLen;
	FLMUINT				uiWildcardPos;
	FLMUINT				uiTargetNumChars;
	FLMUINT				uiNumChars;
	FLMBOOL				bNotUsingFirstOfString = FALSE;
	FLMUNICODE			uzChar;
	FLMUNICODE			uzDummy;

#define	GOOD_ENOUGH_CHARS			16

	// There may not be any wildcards at all.  Find the first one.
	
	if (RC_BAD( rc = flmUTF8FindWildcard( pucValue,
								&uiWildcardPos, puiCompareRules)))
	{
		goto Exit;
	}
	
	// FLM_MAX_UINT is returned if no wildcard was found.
	
	if (uiWildcardPos == FLM_MAX_UINT)
	{
		goto Exit;
	}

	pucEnd = &pucValue [*puiValueLen];
	bBestTerminatesWithWildCard = TRUE;
	pucBest = pucValue;

	// Skip past the wildcard

	pucTmp = &pucValue [uiWildcardPos];
	if (RC_BAD( rc = f_getCharFromUTF8Buf( &pucTmp, pucEnd, &uzDummy)))
	{
		goto Exit;
	}

	uiCurLen = *puiValueLen - (FLMUINT)(pucTmp - pucValue);
	pucCurValue = pucTmp;

	uiBestValueLen = uiWildcardPos;
	if (RC_BAD( rc = flmCountCharacters( pucValue, uiWildcardPos,
							GOOD_ENOUGH_CHARS, puiCompareRules, &uiBestNumChars)))
	{
		goto Exit;
	}
	uiTargetNumChars = uiBestNumChars + uiBestNumChars;

	// Here is the great FindADoubleLengthThatIsBetter algorithm.
	// Below are the values to pick a next better contains key.
	// 	First Key Size			Next Key Size that will be used
	// 		1				* 2		2		// Single char searches are REALLY BAD
	// 		2				* 2		4
	// 		3				* 2		6
	// 		4				* 2		8
	// 		...						...
	// At each new key piece, increment the target length by 2 so that it
	// will be even harder to find a better key.

	while (uiBestNumChars < GOOD_ENOUGH_CHARS)
	{
		pucTmp = pucCurValue;
		if (RC_BAD( rc = f_getCharFromUTF8Buf( &pucTmp, pucEnd, &uzChar)))
		{
			goto Exit;
		}

		if (!uzChar)
		{
			break;
		}

		if (RC_BAD( rc = flmUTF8FindWildcard( pucCurValue, &uiWildcardPos,
										puiCompareRules)))
		{
			goto Exit;
		}
		
		// FLM_MAX_UINT is returned when no wildcard is found.
		
		if (uiWildcardPos == FLM_MAX_UINT)
		{
			
			// No wildcard found
			// Check the last section that may or may not have trailing *.
				
			if (RC_BAD( rc = flmCountCharacters( pucCurValue, uiCurLen,
								GOOD_ENOUGH_CHARS, puiCompareRules, &uiNumChars)))
			{
				goto Exit;
			}

			if (uiNumChars >= uiTargetNumChars)
			{
				pucBest = pucCurValue;
				uiBestValueLen = uiCurLen;
				bBestTerminatesWithWildCard = *pbTrailingWildcard;
			}
			break;
		}
		else
		{
			if (RC_BAD( rc = flmCountCharacters( pucCurValue, uiWildcardPos,
										GOOD_ENOUGH_CHARS, puiCompareRules, &uiNumChars)))
			{
				goto Exit;
			}

			if (uiNumChars >= uiTargetNumChars)
			{
				pucBest = pucCurValue;
				uiBestValueLen = uiWildcardPos;
				uiBestNumChars = uiNumChars;
				uiTargetNumChars = uiNumChars + uiNumChars;
			}
			else
			{
				uiTargetNumChars += 2;
			}

			// Skip past the wildcard

			pucTmp = &pucCurValue[ uiWildcardPos];
			if (RC_BAD( rc = f_getCharFromUTF8Buf( &pucTmp, pucEnd, &uzDummy)))
			{
				goto Exit;
			}

			uiCurLen -= (FLMUINT)(pucTmp - pucCurValue);
			pucCurValue = pucTmp;
		}
	}

	if (pucBest != *ppucValue)
	{
		bNotUsingFirstOfString = TRUE;
	}

	*ppucValue = pucBest;
	*puiValueLen = uiBestValueLen;
	*pbTrailingWildcard = bBestTerminatesWithWildCard;

Exit:

	*pbNotUsingFirstOfString = bNotUsingFirstOfString;
	return( rc);
}

/****************************************************************************
Desc:	Set the case byte on the from key for a case-insensitive search
		that is using a case sensitive index.
****************************************************************************/
FSTATIC void setFromCaseByte(
	FLMBYTE *	pucFromKey,
	FLMUINT *	puiFromComponentLen,
	FLMUINT		uiCaseLen,
	FLMBOOL		bIsDBCS,
	FLMBOOL		bAscending,
	FLMBOOL		bExcl)
{
			
	// Subtract off all but the case marker.
	// Remember that for DBCS (Asian) the case marker is two bytes.

	*puiFromComponentLen -= (uiCaseLen -
									 ((FLMUINT)(bIsDBCS
												  ? (FLMUINT)2
												  : (FLMUINT)1)));
	if (bExcl)
	{
		if (bAscending)
		{
			// Keys are in ascending order:
			// "abc" key == abc+4 (4 is COLL_MARKER | SC_LOWER) 
			// "ABC" key == abc+6 (6 is COLL_MARKER | SC_UPPER)
			// Thus, to exclude all "abc"s on "from" side we need the
			// following key:
			// key == abc+6 (COLL_MARKER | SC_UPPER) + 1
			
			pucFromKey[ *puiFromComponentLen - 1] = (COLL_MARKER | SC_UPPER);
		}
		else
		{
			
			// Keys are in descending order:
			// "ABC" key == abc+6 (6 is COLL_MARKER | SC_UPPER)
			// "abc" key == abc+4 (4 is COLL_MARKER | SC_LOWER) 
			// Thus, to exclude "abc"s on "from" side we need the
			// following key:
			// key == abc+4 (COLL_MARKER | SC_LOWER)
			
			pucFromKey[ *puiFromComponentLen - 1] = (COLL_MARKER | SC_LOWER);
		}
	}
	else	// Inclusive
	{
		if (bAscending)
		{
			// Keys are in ascending order:
			// "abc" key == abc+4 (4 is COLL_MARKER | SC_LOWER) 
			// "ABC" key == abc+6 (6 is COLL_MARKER | SC_UPPER)
			// Thus, to include all "abc"s on "from" side,
			// we need the following key:
			// key == abc+4 (COLL_MARKER | SC_LOWER)
			
			pucFromKey [*puiFromComponentLen - 1] = COLL_MARKER | SC_LOWER;
		}
		else
		{

			// Keys are in descending order:
			// "ABC" key == abc+6 (6 is COLL_MARKER | SC_UPPER)
			// "abc" key == abc+4 (4 is COLL_MARKER | SC_LOWER) 
			// Thus, to include all "abc"s on "from" side we need the
			// following key:
			// key == abc+6 (COLL_MARKER | SC_UPPER)
			
			pucFromKey [*puiFromComponentLen - 1] = COLL_MARKER | SC_UPPER;
		}
	}
}

/****************************************************************************
Desc:	Set the case byte on the until key for a case-insensitive search
		that is using a case sensitive index.
****************************************************************************/
FSTATIC void setUntilCaseByte(
	FLMBYTE *	pucUntilKey,
	FLMUINT *	puiUntilComponentLen,
	FLMUINT		uiCaseLen,
	FLMBOOL		bIsDBCS,
	FLMBOOL		bAscending,
	FLMBOOL		bExcl)
{
			
	// Subtract off all but the case marker.
	// Remember that for DBCS (Asian) the case marker is two bytes.

	*puiUntilComponentLen -= (uiCaseLen -
									  ((FLMUINT)(bIsDBCS
													 ? (FLMUINT)2
													 : (FLMUINT)1)));
	if (bExcl)
	{
		if (bAscending)
		{
			// Keys are in ascending order:
			// "abc" key == abc+4 (4 is COLL_MARKER | SC_LOWER) 
			// "ABC" key == abc+6 (6 is COLL_MARKER | SC_UPPER)
			// Thus, to exclude all "abc"s on the "until" side we need
			// the following key:
			// key == abc+4 (COLL_MARKER | SC_LOWER)
			
			pucUntilKey[ *puiUntilComponentLen - 1] = (COLL_MARKER | SC_LOWER);
		}
		else
		{
			
			// Keys are in descending order:
			// "ABC" key == abc+6 (6 is COLL_MARKER | SC_UPPER)
			// "abc" key == abc+4 (4 is COLL_MARKER | SC_LOWER) 
			// Thus, to exclude all "abc"s on the "until" side we need
			// the following key:
			// key == abc+6 (COLL_MARKER | SC_UPPER) + 1
			
			pucUntilKey[ *puiUntilComponentLen - 1] = (COLL_MARKER | SC_UPPER);
		}
	}
	else
	{
		if (bAscending)
		{
			// Keys are in ascending order:
			// "abc" key == abc+4 (4 is COLL_MARKER | SC_LOWER) 
			// "ABC" key == abc+6 (6 is COLL_MARKER | SC_UPPER)
			// Thus, to get include all "abc"s on the "until" side we need
			// the following key:
			// key == abc+6 (COLL_MARKER | SC_UPPER)
			
			pucUntilKey [*puiUntilComponentLen - 1] = (COLL_MARKER | SC_UPPER);
		}
		else
		{

			// Keys are in descending order:
			// "ABC" key == abc+6 (6 is COLL_MARKER | SC_UPPER)
			// "abc" key == abc+4 (4 is COLL_MARKER | SC_LOWER) 
			// Thus, to include all "abc"s on the "until side we need
			// the following key:
			// key == abc+4 (COLL_MARKER | SC_LOWER)
			
			pucUntilKey [*puiUntilComponentLen - 1] = (COLL_MARKER | SC_LOWER);
		}
	}
}

/****************************************************************************
Desc:	Build a text key.
****************************************************************************/
FSTATIC RCODE flmAddTextKeyPiece(
	SQL_PRED *		pPred,
	F_INDEX *		pIndex,
	FLMUINT			uiKeyComponent,
	ICD *				pIcd,
	F_DataVector *	pFromSearchKey,
	FLMBYTE *		pucFromKey,
	FLMUINT *		puiFromKeyLen,
	F_DataVector *	pUntilSearchKey,
	FLMBYTE *		pucUntilKey,
	FLMUINT *		puiUntilKeyLen,
	FLMBOOL *		pbCanCompareOnKey,
	FLMBOOL *		pbFromIncl,
	FLMBOOL *		pbUntilIncl)
{
	RCODE       			rc = NE_SFLM_OK;
	FLMBYTE *				pucFromKeyLenPos;
	FLMBYTE *				pucUntilKeyLenPos;
	FLMUINT					uiLanguage = pIndex->uiLanguage;
	FLMUINT					uiFromCollationLen = 0;
	FLMUINT					uiUntilCollationLen = 0;
	FLMUINT					uiCharCount;
	FLMUINT					uiFromCaseLen;
	FLMUINT					uiUntilCaseLen;
	FLMBOOL					bFromOriginalCharsLost = FALSE;
	FLMBOOL					bUntilOriginalCharsLost = FALSE;
	FLMBOOL					bIsDBCS = (uiLanguage >= FLM_FIRST_DBCS_LANG &&
								  uiLanguage <= FLM_LAST_DBCS_LANG)
								  ? TRUE
								  : FALSE;

	FLMBOOL					bCaseInsensitive = (FLMBOOL)((pPred->uiCompareRules &
															FLM_COMP_CASE_INSENSITIVE)
															? TRUE
															: FALSE);
	FLMBOOL					bDoFirstSubstring = (FLMBOOL)((pIcd->uiFlags & ICD_SUBSTRING)
															 ? TRUE
															 : FALSE);
	FLMBOOL					bDoMatchBegin = FALSE;
	FLMBOOL					bTrailingWildcard = FALSE;
	const FLMBYTE *		pucFromUTF8Buf = NULL;
	FLMUINT					uiFromBufLen = 0;
	const FLMBYTE *		pucUntilUTF8Buf = NULL;
	FLMUINT					uiUntilBufLen = 0;
	FLMUINT					uiWildcardPos;
	FLMBOOL					bFromDataTruncated;
	FLMBOOL					bUntilDataTruncated;
	FLMUINT					uiFromFlags = 0;
	FLMUINT					uiUntilFlags = 0;
	FLMUINT					uiCompareRules;
	FLMBOOL					bAscending = (pIcd->uiFlags & ICD_DESCENDING) ? FALSE: TRUE;
	F_BufferIStream		bufferIStream;
	FLMUINT					uiFromKeyLen = *puiFromKeyLen;
	FLMUINT					uiUntilKeyLen = *puiUntilKeyLen;
	FLMUINT					uiFromComponentLen;
	FLMUINT					uiUntilComponentLen;
	FLMUINT					uiFromSpaceLeft;
	FLMUINT					uiUntilSpaceLeft;
	
	// Leave room for the component length

	pucFromKeyLenPos = pucFromKey + uiFromKeyLen;
	pucUntilKeyLenPos = pucUntilKey + uiUntilKeyLen;
	pucFromKey = pucFromKeyLenPos + 2;
	pucUntilKey = pucUntilKeyLenPos + 2;
	uiFromSpaceLeft = SFLM_MAX_KEY_SIZE - uiFromKeyLen - 2;
	uiUntilSpaceLeft = SFLM_MAX_KEY_SIZE - uiUntilKeyLen - 2;

	switch (pPred->eOperator)
	{

		// The difference between MATCH and EQ_OP is that EQ does
		// not support wildcards embedded in the search key.

		case SQL_MATCH_OP:
			flmAssert( pPred->pFromValue->eValType == SQL_UTF8_VAL);
			pucFromUTF8Buf = pPred->pFromValue->val.str.pszStr;
			uiFromBufLen = pPred->pFromValue->val.str.uiByteLen;
			uiCompareRules = pIcd->uiCompareRules;

			if (RC_BAD( rc = flmUTF8FindWildcard( pucFromUTF8Buf, &uiWildcardPos,
										&uiCompareRules)))
			{
				goto Exit;
			}
			
			// If there is no wildcard, uiWildcardPos will be FLM_MAX_UINT
			
			if (uiWildcardPos != FLM_MAX_UINT)
			{

				// If wildcard is in position 0, it is NOT
				// a match begin.

				if (uiWildcardPos)
				{
					bDoMatchBegin = TRUE;
					uiFromBufLen = uiWildcardPos;
				}
			}
			if (!(pIcd->uiFlags & ICD_SUBSTRING))
			{

				// Index is NOT a substring index

				if (!bDoMatchBegin)
				{

					// Wildcard was at the beginning, will have
					// to search the index from first to last

					pucFromUTF8Buf = NULL;
				}
				else
				{
					bTrailingWildcard = TRUE;
				}
			}
			else
			{
				FLMBOOL	bNotUsingFirstOfString;

				// If this is a substring index look for a 
				// better 'contains' string to search for. 
				// We don't like "A*BCDEFG" searches.
				
				bTrailingWildcard = bDoMatchBegin;

				uiCompareRules = pIcd->uiCompareRules;
				if (RC_BAD( rc = flmSelectBestSubstr( &pucFromUTF8Buf,
					&uiFromBufLen, 
					&uiCompareRules, &bTrailingWildcard,
					&bNotUsingFirstOfString)))
				{
					goto Exit;
				}

				if (bNotUsingFirstOfString)
				{
					bDoMatchBegin = bTrailingWildcard;
					*pbCanCompareOnKey = FALSE;
					bDoFirstSubstring = FALSE;
				}
				else if (bTrailingWildcard)
				{
					bDoMatchBegin = TRUE;
				}

				if (RC_BAD( rc = flmCountCharacters( pucFromUTF8Buf,
									uiFromBufLen, 2,
									&uiCompareRules, &uiCharCount)))
				{
					goto Exit;
				}
				
				// Special case: Single character contains/MEnd in a substr ix.

				if (!bIsDBCS && uiCharCount < 2)
				{
					pucFromUTF8Buf = NULL;
				}
			}
			
			pucUntilUTF8Buf = pucFromUTF8Buf;
			uiUntilBufLen = uiFromBufLen;
			break;

		case SQL_RANGE_OP:
			if (bAscending)
			{
				if (pPred->pFromValue)
				{
					flmAssert( pPred->pFromValue->eValType == SQL_UTF8_VAL);
					pucFromUTF8Buf = pPred->pFromValue->val.str.pszStr;
					uiFromBufLen = pPred->pFromValue->val.str.uiByteLen;
				}
				else
				{
					// Should have been done up above
					
					// pucFromUTF8Buf = NULL;
					// uiFromBufLen = 0;
				}
				if (pPred->pUntilValue)
				{
					flmAssert( pPred->pUntilValue->eValType == SQL_UTF8_VAL);
					pucUntilUTF8Buf = pPred->pUntilValue->val.str.pszStr;
					uiUntilBufLen = pPred->pUntilValue->val.str.uiByteLen;
				}
				else
				{
					// Should have been done up above.
					
					// pucUntilUTF8Buf = NULL;
					// uiUntilBufLen = 0;
				}
				if (!pPred->bInclFrom)
				{
					uiFromFlags |= EXCLUSIVE_GT_FLAG;
				}
				if (!pPred->bInclUntil)
				{
					uiUntilFlags |= EXCLUSIVE_LT_FLAG;
				}
			}
			else
			{
				if (pPred->pUntilValue)
				{
					flmAssert( pPred->pUntilValue->eValType == SQL_UTF8_VAL);
					pucFromUTF8Buf = pPred->pUntilValue->val.str.pszStr;
					uiFromBufLen = pPred->pUntilValue->val.str.uiByteLen;
				}
				else
				{
					// Should have been done up above
					
					// pucFromUTF8Buf = NULL;
					// uiFromBufLen = 0;
				}
					
				if (pPred->pFromValue)
				{
					flmAssert( pPred->pFromValue->eValType == SQL_UTF8_VAL);
					pucUntilUTF8Buf = pPred->pFromValue->val.str.pszStr;
					uiUntilBufLen = pPred->pFromValue->val.str.uiByteLen;
				}
				else
				{
					// Should have been done up above.
					
					// pucUntilUTF8Buf = NULL;
					// uiUntilBufLen = 0;
				}
				if (!pPred->bInclUntil)
				{
					uiFromFlags |= EXCLUSIVE_GT_FLAG;
				}
				if (!pPred->bInclFrom)
				{
					uiUntilFlags |= EXCLUSIVE_LT_FLAG;
				}
			}
			break;

		case SQL_NE_OP:

			// Set up to do full index scan.
			
			// Buffers should already be NULL.

			// pucFromUTF8Buf = NULL;
			// pucUntilUTF8Buf = NULL;
			break;

		case SQL_APPROX_EQ_OP:

			// Set up to do full index scan.
			
			// Buffers should already be NULL

			// pucFromUTF8Buf = NULL;
			// pucUntilUTF8Buf = NULL;

			// Cannot compare on the key if index is upper case,
			// even if the bCaseInsensitive flag is set.

			if (pIcd->uiCompareRules & FLM_COMP_CASE_INSENSITIVE)
			{
				*pbCanCompareOnKey = FALSE;
			}
			break;

		default:

			// Every predicate should have been converted to one of the above
			// cases, or should be handled by another routine.

			rc = RC_SET_AND_ASSERT( NE_SFLM_QUERY_SYNTAX);
			goto Exit;
	}

	// If index is case insensitive, but search is case sensitive
	// we must NOT do a key match - we would fail things we should
	// not be failing.
	
	if ((pIcd->uiCompareRules & FLM_COMP_CASE_INSENSITIVE) && !bCaseInsensitive)
	{
		*pbCanCompareOnKey = FALSE;
	}
	
	if (!pucFromUTF8Buf)
	{
		uiFromComponentLen = KEY_LOW_VALUE;
	}
	else
	{
		if (RC_BAD( rc = bufferIStream.openStream( 
			(const char *)pucFromUTF8Buf, uiFromBufLen)))
		{
			goto Exit;
		}

		// Add ICD_ESC_CHAR to the icd flags because
		// the search string must have BACKSLASHES and '*' escaped.

		uiFromComponentLen = uiFromSpaceLeft;
		bFromDataTruncated = FALSE;
		if (RC_BAD( rc = KYCollateValue( pucFromKey, &uiFromComponentLen,
								&bufferIStream, SFLM_STRING_TYPE,
							pIcd->uiFlags | ICD_ESC_CHAR, pIcd->uiCompareRules,
							pIcd->uiLimit,
							&uiFromCollationLen, &uiFromCaseLen,
							uiLanguage, bDoFirstSubstring,
							FALSE, &bFromDataTruncated,
							&bFromOriginalCharsLost)))
		{
			goto Exit;
		}
		
		bufferIStream.closeStream();
		
		if (bFromDataTruncated)
		{
			*pbCanCompareOnKey = FALSE;
			
			// Save the original data into pFromSearchKey so the comparison
			// routines can do a comparison on the full value if
			// necessary.

			if (RC_BAD( rc = pFromSearchKey->setUTF8( uiKeyComponent,
										pucFromUTF8Buf, uiFromBufLen)))
			{
				goto Exit;
			}
			uiFromFlags |= (SEARCH_KEY_FLAG | TRUNCATED_FLAG);
		}
		else if (bFromOriginalCharsLost)
		{
			*pbCanCompareOnKey = FALSE;
		}
		
		if (pucFromUTF8Buf != pucUntilUTF8Buf)
		{
			
			// Handle scenario of a case-sensitive index, but search is
			// case-insensitive.

			if (uiFromComponentLen &&
				 (bIsDBCS ||
				   (!(pIcd->uiCompareRules & FLM_COMP_CASE_INSENSITIVE) &&
					 bCaseInsensitive)))
			{
				setFromCaseByte( pucFromKey, &uiFromComponentLen, uiFromCaseLen,
										bIsDBCS, bAscending,
										(uiFromFlags & EXCLUSIVE_GT_FLAG)
										? TRUE
										: FALSE);
			}
		}
	}

	// Do the until key now

	if (!pucUntilUTF8Buf)
	{
		uiUntilComponentLen = KEY_HIGH_VALUE;
	}
	else if (pucFromUTF8Buf == pucUntilUTF8Buf)
	{
		
		// Handle case where from and until buffers are the same.
		// This should only be possible in the equality case or match begin
		// case, in which cases neither the EXCLUSIVE_LT_FLAG or the
		// EXCLUSIVE_GT_FLAG should be set.
		
		flmAssert( uiFromBufLen == uiUntilBufLen);
		flmAssert( !(uiFromFlags & (EXCLUSIVE_GT_FLAG | EXCLUSIVE_LT_FLAG)));
		flmAssert( !(uiUntilFlags & (EXCLUSIVE_GT_FLAG | EXCLUSIVE_LT_FLAG)));
		
		// Need to collate the until key from the original data if
		// the from key was truncated or there is not enough room in
		// the until key.  Otherwise, we can simply copy
		// the from key into the until key - a little optimization.
		
		if (uiUntilSpaceLeft >= uiFromComponentLen && !bFromDataTruncated)
		{
			if ((uiUntilComponentLen = uiFromComponentLen) != 0)
			{
				f_memcpy( pucUntilKey, pucFromKey, uiFromComponentLen);
			}
			uiUntilCaseLen = uiFromCaseLen;
			uiUntilCollationLen = uiFromCollationLen;
			bUntilOriginalCharsLost = bFromOriginalCharsLost;
			bUntilDataTruncated = FALSE;
		}
		else
		{
			if (RC_BAD( rc = bufferIStream.openStream( 
				(const char *)pucUntilUTF8Buf, uiUntilBufLen)))
			{
				goto Exit;
			}
	
			// Add ICD_ESC_CHAR to the icd flags because
			// the search string must have BACKSLASHES and '*' escaped.
	
			uiUntilComponentLen = uiUntilSpaceLeft;
			bUntilDataTruncated = FALSE;
			if (RC_BAD( rc = KYCollateValue( pucUntilKey, &uiUntilComponentLen,
								&bufferIStream, SFLM_STRING_TYPE,
								pIcd->uiFlags | ICD_ESC_CHAR, pIcd->uiCompareRules,
								pIcd->uiLimit,
								&uiUntilCollationLen, &uiUntilCaseLen,
								uiLanguage, bDoFirstSubstring, 
								FALSE, &bUntilDataTruncated,
								&bUntilOriginalCharsLost)))
			{
				goto Exit;
			}
			
			bufferIStream.closeStream();
			
			if (bUntilDataTruncated)
			{
				
				// Save the original data into pUntilSearchKey so the comparison
				// routines can do a comparison on the full value if
				// necessary.
	
				if (RC_BAD( rc = pUntilSearchKey->setUTF8( uiKeyComponent,
											pucUntilUTF8Buf, uiUntilBufLen)))
				{
					goto Exit;
				}
				*pbCanCompareOnKey = FALSE;
				uiUntilFlags |= (SEARCH_KEY_FLAG | TRUNCATED_FLAG);
			}
			else if (bUntilOriginalCharsLost)
			{
				*pbCanCompareOnKey = FALSE;
			}
		}
		
		if (bDoMatchBegin)
		{
			if (bAscending)
			{
				
				// Handle scenario of a case-sensitive index, but search is
				// case-insensitive.
		
				if (uiFromComponentLen &&
					 (bIsDBCS ||
					  (!(pIcd->uiCompareRules & FLM_COMP_CASE_INSENSITIVE) &&
						bCaseInsensitive)))
				{
					setFromCaseByte( pucFromKey, &uiFromComponentLen, uiFromCaseLen,
											bIsDBCS, bAscending, FALSE);
				}
	
				// Fill everything after the collation values in the until
				// key with high values (0xFF)
	
				f_memset( &pucUntilKey[ uiUntilCollationLen], 0xFF,
								uiUntilSpaceLeft - uiUntilCollationLen);
				uiUntilComponentLen = uiUntilSpaceLeft;
			}
			else
			{
				
				if (uiUntilComponentLen &&
					 (bIsDBCS ||
					  (!(pIcd->uiCompareRules & FLM_COMP_CASE_INSENSITIVE) &&
						bCaseInsensitive)))
				{
					// NOTE: Always inclusive because this is a matchbegin.
				
					setUntilCaseByte( pucUntilKey, &uiUntilComponentLen, uiUntilCaseLen,
										bIsDBCS, bAscending, FALSE);
				}
									
				// Fill rest of from key with high values after collation values.
				
				f_memset( &pucFromKey[ uiFromCollationLen], 0xFF,
							uiFromSpaceLeft - uiFromCollationLen);
				uiFromComponentLen = uiFromSpaceLeft;
			}
		}
		else
		{
			if (bDoFirstSubstring)
			{
				FLMUINT	uiBytesToRemove = (bIsDBCS) ? 2 : 1;
				
				if (bAscending)
				{
					
					// Get rid of the first substring byte in the until
					// key.
					
					f_memmove( &pucUntilKey [uiUntilCollationLen],
								  &pucUntilKey [uiUntilCollationLen + uiBytesToRemove],
								  uiUntilComponentLen - uiUntilCollationLen - uiBytesToRemove);
					uiUntilComponentLen -= uiBytesToRemove;
				}
				else
				{
					
					// Descending order - put the string without the
					// first-substring-marker into the from key instead of
					// the until key.
					
					f_memmove( &pucFromKey [uiFromCollationLen],
								  &pucFromKey [uiFromCollationLen + uiBytesToRemove],
								  uiFromComponentLen - uiFromCollationLen - uiBytesToRemove);
					uiFromComponentLen -= uiBytesToRemove;
				}
			
				// Handle scenario of a case-sensitive index, but search is
				// case-insensitive.
		
				if (bIsDBCS ||
					 (!(pIcd->uiCompareRules & FLM_COMP_CASE_INSENSITIVE) &&
					  bCaseInsensitive))
				{
					setFromCaseByte( pucFromKey, &uiFromComponentLen, uiFromCaseLen,
												bIsDBCS, bAscending, FALSE);
					setUntilCaseByte( pucUntilKey, &uiUntilComponentLen, uiUntilCaseLen,
												bIsDBCS, bAscending, FALSE);
				}
			}
		}
	}
	else // pucFromUTF8Buf != pucUntilUTF8Buf
	{
		if (RC_BAD( rc = bufferIStream.openStream( 
			(const char *)pucUntilUTF8Buf, uiUntilBufLen)))
		{
			goto Exit;
		}

		// Add ICD_ESC_CHAR to the icd flags because
		// the search string must have BACKSLASHES and '*' escaped.

		uiUntilComponentLen = uiUntilSpaceLeft;
		bUntilDataTruncated = FALSE;
		if (RC_BAD( rc = KYCollateValue( pucUntilKey, &uiUntilComponentLen,
							&bufferIStream, SFLM_STRING_TYPE,
							pIcd->uiFlags | ICD_ESC_CHAR, pIcd->uiCompareRules,
							pIcd->uiLimit,
							&uiUntilCollationLen, &uiUntilCaseLen,
							uiLanguage, bDoFirstSubstring, 
							FALSE, &bUntilDataTruncated,
							&bUntilOriginalCharsLost)))
		{
			goto Exit;
		}
		
		bufferIStream.closeStream();
		
		if (bUntilDataTruncated)
		{
			
			// Save the original data into pUntilSearchKey so the comparison
			// routines can do a comparison on the full value if
			// necessary.

			if (RC_BAD( rc = pUntilSearchKey->setUTF8( uiKeyComponent,
										pucUntilUTF8Buf, uiUntilBufLen)))
			{
				goto Exit;
			}
			*pbCanCompareOnKey = FALSE;
			uiUntilFlags |= (SEARCH_KEY_FLAG | TRUNCATED_FLAG);
		}
		else if (bUntilOriginalCharsLost)
		{
			*pbCanCompareOnKey = FALSE;
		}

		if (uiUntilComponentLen &&
			 (bIsDBCS ||
			  (!(pIcd->uiCompareRules & FLM_COMP_CASE_INSENSITIVE) &&
			   bCaseInsensitive)))
		{
			setUntilCaseByte( pucUntilKey, &uiUntilComponentLen, uiUntilCaseLen,
									bIsDBCS, bAscending,
									(uiUntilFlags & EXCLUSIVE_LT_FLAG)
									? TRUE
									: FALSE);
		}
	}
	
	UW2FBA( (FLMUINT16)(uiFromKeyLen | uiFromFlags), pucFromKeyLenPos);
	UW2FBA( (FLMUINT16)(uiUntilKeyLen | uiUntilFlags), pucUntilKeyLenPos);
	
	// Set the FROM and UNTIL key length return values.

	uiFromKeyLen += 2;
	if (uiFromComponentLen != KEY_LOW_VALUE)
	{
		uiFromKeyLen += uiFromComponentLen;
		if (!(uiFromFlags & EXCLUSIVE_GT_FLAG))
		{
			*pbFromIncl = TRUE;
		}
	}
	uiUntilKeyLen += 2;
	if (uiUntilComponentLen != KEY_HIGH_VALUE)
	{
		uiUntilKeyLen += uiUntilComponentLen;
		if (!(uiUntilFlags & EXCLUSIVE_LT_FLAG))
		{
			*pbUntilIncl = TRUE;
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:		Build the from and until keys given a field list with operators and
			values and an index.
Notes:	The knowledge of query definitions is limited in these routines.
****************************************************************************/
RCODE flmBuildFromAndUntilKeys(
	F_Dict *			pDict,
	F_INDEX *		pIndex,
	F_TABLE *		pTable,
	SQL_PRED **		ppKeyComponents,
	F_DataVector *	pFromSearchKey,
	FLMBYTE *		pucFromKey,
	FLMUINT *		puiFromKeyLen,
	F_DataVector *	pUntilSearchKey,
	FLMBYTE *		pucUntilKey,
	FLMUINT *		puiUntilKeyLen,
	FLMBOOL *		pbDoRowMatch,
	FLMBOOL *		pbCanCompareOnKey)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiKeyComponent;
	SQL_PRED *	pPred;
	F_COLUMN *	pColumn;
	ICD *			pIcd;
	FLMBOOL		bFromIncl;
	FLMBOOL		bUntilIncl;
	FLMUINT		uiFromKeyLen = 0;
	FLMUINT		uiUntilKeyLen = 0;
								
	*pbDoRowMatch = FALSE;
	*pbCanCompareOnKey = TRUE;

	if (!ppKeyComponents)
	{
		
		// Setup a first-to-last key
		
		UW2FBA( KEY_LOW_VALUE, pucFromKey);
		UW2FBA( KEY_HIGH_VALUE, pucUntilKey);
		uiFromKeyLen += 2;
		uiUntilKeyLen += 2;
		*pbDoRowMatch = TRUE;
		*pbCanCompareOnKey = FALSE;
	}
	else
	{
		for (uiKeyComponent = 0, pIcd = pIndex->pKeyIcds;
			  uiKeyComponent < pIndex->uiNumKeyComponents;
			  uiKeyComponent++, pIcd++)
		{
			if ((pPred = ppKeyComponents [uiKeyComponent]) == NULL)
			{
				break;
			}
			
			// At this point, we always better have room to put at least
			// two bytes in the key.
			
			flmAssert( SFLM_MAX_KEY_SIZE - uiFromKeyLen >= 2 &&
				 		  SFLM_MAX_KEY_SIZE - uiUntilKeyLen >= 2);

			// Predicates we are looking at should NEVER be notted.  They
			// will have been weeded out earlier.
	
			flmAssert( !pPred->bNotted);
			
			bFromIncl = FALSE;
			bUntilIncl = FALSE;
		
			// Handle special cases for indexing presence and/or exists predicate.
	
			pColumn = pDict->getColumn( pTable, pIcd->uiColumnNum);
			if (pColumn->eDataTyp == SFLM_STRING_TYPE &&
						!(pIcd->uiFlags & (ICD_PRESENCE | ICD_METAPHONE)) &&
						pPred->eOperator != SQL_EXISTS_OP)
			{
				if (RC_BAD( rc = flmAddTextKeyPiece( pPred, pIndex,
							uiKeyComponent, pIcd,
							pFromSearchKey, pucFromKey, &uiFromKeyLen,
							pUntilSearchKey, pucUntilKey, &uiUntilKeyLen,
							pbCanCompareOnKey, &bFromIncl, &bUntilIncl)))
				{
					goto Exit;
				}
			}
			else
			{
				if (RC_BAD( rc = flmAddNonTextKeyPiece( pPred, pIndex,
											uiKeyComponent, pIcd, pColumn,
											pFromSearchKey, pucFromKey, &uiFromKeyLen,
											pUntilSearchKey, pucUntilKey, &uiUntilKeyLen,
											pbCanCompareOnKey, &bFromIncl, &bUntilIncl)))
				{
					goto Exit;
				}
			}
			
			// If this is our last component, see if the from or until
			// component is inclusive, in which case we need to add one
			// more component to the key so it will be "inclusive".
			
			if (uiKeyComponent + 1 == pIndex->uiNumKeyComponents ||
				 !ppKeyComponents [uiKeyComponent + 1] ||
				 SFLM_MAX_KEY_SIZE - uiFromKeyLen < 2 ||
				 SFLM_MAX_KEY_SIZE - uiUntilKeyLen < 2)
			{
				
				// bFromIncl means we had a >= or == for the component.
				// If there is a next key component that would be expected, set it to
				// the lowest possible value.  There must also be room for a
				// two byte value.
				
				if (bFromIncl &&
					 uiKeyComponent + 1 < pIndex->uiNumKeyComponents &&
					 SFLM_MAX_KEY_SIZE - uiFromKeyLen >= 2)
				{
					UW2FBA( (FLMUINT16)KEY_LOW_VALUE, &pucFromKey [uiFromKeyLen]);
					uiFromKeyLen += 2;
				}
				
				// bUntilIncl means we had a <= or == for the component.
				// If there is a next key component that would be expected, set it to
				// the highest possible value.
					
				if (bUntilIncl)
				{
					if (uiKeyComponent + 1 < pIndex->uiNumKeyComponents)
					{
						if (SFLM_MAX_KEY_SIZE - uiUntilKeyLen >= 2)
						{
							UW2FBA( (FLMUINT16)KEY_LOW_VALUE, &pucUntilKey [uiUntilKeyLen]);
							uiUntilKeyLen += 2;
						}
					}
					else if (SFLM_MAX_KEY_SIZE - uiUntilKeyLen >= 1)
					{
						
						// There are no more key components.
						// Output one byte of 0xFF - which should be higher than any
						// possible SEN that could be output for row ID.
						// Only do this for until keys.  For from keys, no need to add
						// anything, because an empty list of IDs will be be inclusive
						// on the from side - it will sort lower, and therefore be inclusive.
						
						pucUntilKey [uiUntilKeyLen] = 0xFF;
						uiUntilKeyLen++;
					}
				}
				break;
			}
		}
	}

Exit:

	*puiFromKeyLen = uiFromKeyLen;
	*puiUntilKeyLen = uiUntilKeyLen;

	if (!(*pbCanCompareOnKey))
	{
		*pbDoRowMatch = TRUE;
	}

	return( rc);
}

