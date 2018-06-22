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

FSTATIC FLMUINT kyAddInclComponent(
	ICD *					pIcd,
	FLMBYTE *			pucKeyEnd,
	FLMBOOL				bFromKey,
	FLMUINT				uiMaxSpaceLeft);
	
FINLINE void flmSetupFirstToLastKey(
	FLMBYTE *			pucFromKey,
	FLMUINT *			puiFromKeyLen,
	FLMBYTE *			pucUntilKey,
	FLMUINT *			puiUntilKeyLen);
	
FSTATIC RCODE flmAddNonTextKeyPiece(
	PATH_PRED *			pPred,
	IXD *					pIxd,
	ICD *					pIcd,
	F_DataVector *		pFromSearchKey,
	FLMBYTE *			pucFromKey,
	FLMUINT *			puiFromKeyLen,
	F_DataVector *		pUntilSearchKey,
	FLMBYTE *			pucUntilKey,
	FLMUINT *			puiUntilKeyLen,
	FLMBOOL *			pbCanCompareOnKey);

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
	FLMUINT *			puiFromKeyLen,
	FLMUINT				uiCaseLen,
	FLMBOOL				bIsDBCS,
	FLMBOOL				bAscending,
	FLMBOOL				bExcl);
	
FSTATIC void setUntilCaseByte(
	FLMBYTE *			pucUntilKey,
	FLMUINT *			puiUntilKeyLen,
	FLMUINT				uiCaseLen,
	FLMBOOL				bIsDBCS,
	FLMBOOL				bAscending,
	FLMBOOL				bExcl);
	
FSTATIC RCODE flmAddTextKeyPiece(
	PATH_PRED *			pPred,
	IXD *					pIxd,
	ICD *					pIcd,
	F_DataVector *		pFromSearchKey,
	FLMBYTE *			pucFromKey,
	FLMUINT *			puiFromKeyLen,
	F_DataVector *		pUntilSearchKey,
	FLMBYTE *			pucUntilKey,
	FLMUINT *			puiUntilKeyLen,
	FLMBOOL *			pbCanCompareOnKey);

/****************************************************************************
Desc:	Add what is needed to an until key so that it is greater than or equal
		for all possible components that come after the primary component.
		In other words, this key should be less than any keys whose primary
		component is greater than it, but greater than all keys whose primary
		component is less than or equal to it.
****************************************************************************/
FSTATIC FLMUINT kyAddInclComponent(
	ICD *			pIcd,
	FLMBYTE *	pucKeyEnd,
	FLMBOOL		bFromKey,
	FLMUINT		uiMaxSpaceLeft)
{
	FLMUINT	uiBytesAdded = 0;
	
	// If there is a next key component that would be expected, set it to
	// the highest possible value.
	
	if (pIcd->pNextKeyComponent)
	{
		
		// Must at least be room for a 2 byte length.
		
		if (uiMaxSpaceLeft >= 2)
		{
			// Need 2nd key component to sort lower if it is the from key
			// higher if it is the until key.  Note that KEY_LOW_VALUE and
			// KEY_HIGH_VALUE always sort lower/higher no matter whether the
			// component is ascending or descending.
			
			if (bFromKey)
			{
				UW2FBA( (FLMUINT16)KEY_LOW_VALUE, pucKeyEnd);
			}
			else
			{
				UW2FBA( (FLMUINT16)KEY_HIGH_VALUE, pucKeyEnd);
			}
			uiBytesAdded = 2;
		}
	}
	else
	{
		
		// There are no more key components.
		// Output one byte of 0xFF - which should be higher than any
		// possible SEN that could be output for document ID and node IDs.
		// Only do this for until keys.  For from keys, no need to add
		// anything, because an empty list of IDs will be be inclusive
		// on the from side - it will sort lower, and therefore be inclusive.
		
		if (uiMaxSpaceLeft && !bFromKey)
		{
			*pucKeyEnd = 0xFF;
			uiBytesAdded = 1;
		}
	}
	
	return( uiBytesAdded);
}

/****************************************************************************
Desc:	Setup a first-to-last key for the index.
****************************************************************************/
FINLINE void flmSetupFirstToLastKey(
	FLMBYTE *		pucFromKey,
	FLMUINT *		puiFromKeyLen,
	FLMBYTE *		pucUntilKey,
	FLMUINT *		puiUntilKeyLen
	)
{
	UW2FBA( KEY_LOW_VALUE, pucFromKey);
	UW2FBA( KEY_HIGH_VALUE, pucUntilKey);
	*puiFromKeyLen = 2;
	*puiUntilKeyLen = 2;
}

/****************************************************************************
Desc:		Add a key piece to the from and until key.  Text fields are not 
			handled in this routine because of their complexity.
Notes:	The goal of this code is to build a the collated compound piece
			for the 'from' and 'until' key only once instead of twice.
****************************************************************************/
FSTATIC RCODE flmAddNonTextKeyPiece(
	PATH_PRED *		pPred,
	IXD *				pIxd,
	ICD *				pIcd,
	F_DataVector *	pFromSearchKey,
	FLMBYTE *		pucFromKey,
	FLMUINT *		puiFromKeyLen,
	F_DataVector *	pUntilSearchKey,
	FLMBYTE *		pucUntilKey,
	FLMUINT *		puiUntilKeyLen,
	FLMBOOL *		pbCanCompareOnKey)
{
	RCODE						rc = NE_XFLM_OK;
	FLMUINT					uiFromKeyLen = 0;
	FLMUINT					uiUntilKeyLen = 0;
	FLMBYTE *				pucFromKeyLenPos = pucFromKey;
	FLMBYTE *				pucUntilKeyLenPos = pucUntilKey;
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
	FQVALUE *				pFromValue;
	FQVALUE *				pUntilValue;
	FLMBOOL					bInclFrom;
	FLMBOOL					bInclUntil;
	FLMBOOL					bAscending = (pIcd->uiFlags & ICD_DESCENDING) ? FALSE: TRUE;
	IF_BufferIStream *	pBufferIStream = NULL;
	
	// Leave room for the component length

	pucFromKey += 2;
	pucUntilKey += 2;
	
	// Handle the presence case here - this is not done in kyCollate.

	if (pIcd->uiFlags & ICD_PRESENCE)
	{
		f_UINT32ToBigEndian( (FLMUINT32)pIcd->uiDictNum, pucFromKey);
		uiFromKeyLen = uiUntilKeyLen = 4;
		f_memcpy( pucUntilKey, pucFromKey, uiUntilKeyLen);
	}
	else if (pIcd->uiFlags & ICD_METAPHONE)
	{
		if (pPred->eOperator != XFLM_APPROX_EQ_OP ||
			 pPred->pFromValue->eValType != XFLM_UTF8_VAL)
		{
			flmSetupFirstToLastKey( pucFromKeyLenPos, puiFromKeyLen,
											pucUntilKeyLenPos, puiUntilKeyLen);
			if (pPred->eOperator != XFLM_EXISTS_OP)
			{
				*pbCanCompareOnKey = FALSE;
			}
			goto Exit;
		}
		*pbCanCompareOnKey = FALSE;

		// The value type in pPred->pFromValue is XFLM_UTF8_VAL, but the
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
		
		if( !pBufferIStream)
		{
			if( RC_BAD( rc = FlmAllocBufferIStream( &pBufferIStream)))
			{
				goto Exit;
			}
		}

		if (RC_BAD( rc = pBufferIStream->openStream( 
			(const char *)pucFromBuf, uiFromBufLen)))
		{
			goto Exit;
		}

		uiFromKeyLen = XFLM_MAX_KEY_SIZE - 2;
		bDataTruncated = FALSE;
		
		// Pass 0 for compare rules because it is non-text
		
		if (RC_BAD( rc = KYCollateValue( pucFromKey, &uiFromKeyLen,
								pBufferIStream, XFLM_NUMBER_TYPE,
								pIcd->uiFlags, 0,
								pIcd->uiLimit, NULL, NULL, 
								pIxd->uiLanguage, FALSE, FALSE,
								&bDataTruncated, NULL)))
		{
			goto Exit;
		}
		
		pBufferIStream->closeStream();
		
		if (bDataTruncated)
		{
			// This should never happen on numeric data.

			flmAssert( 0);
			*pbCanCompareOnKey = FALSE;
		}

		if (uiFromKeyLen)
		{
			f_memcpy( pucUntilKey, pucFromKey, uiFromKeyLen);
		}

		uiUntilKeyLen = uiFromKeyLen;
	}
	else
	{
		if (pPred->eOperator == XFLM_EXISTS_OP ||
			 pPred->eOperator == XFLM_NE_OP ||
			 pPred->eOperator == XFLM_APPROX_EQ_OP)
		{

			// Setup a first-to-last key

			flmSetupFirstToLastKey( pucFromKeyLenPos, puiFromKeyLen,
											pucUntilKeyLenPos, puiUntilKeyLen);
			goto Exit;
		}

		// Only other operator possible is the range operator

		flmAssert( pPred->eOperator == XFLM_RANGE_OP);
		
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
				case XFLM_UINT_VAL:
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

				case XFLM_INT_VAL:
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

				case XFLM_UINT64_VAL:
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

				case XFLM_INT64_VAL:
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

				case XFLM_BINARY_VAL:
					pucFromBuf = pFromValue->val.pucBuf;
					uiFromBufLen = pFromValue->uiDataLen;
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

					rc = RC_SET_AND_ASSERT( NE_XFLM_QUERY_SYNTAX);
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
				case XFLM_UINT_VAL:
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

				case XFLM_INT_VAL:
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

				case XFLM_UINT64_VAL:
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

				case XFLM_INT64_VAL:
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

				case XFLM_BINARY_VAL:
					pucUntilBuf = pUntilValue->val.pucBuf;
					uiUntilBufLen = pUntilValue->uiDataLen;
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

					rc = RC_SET_AND_ASSERT( NE_XFLM_QUERY_SYNTAX);
					goto Exit;
			}
		}

		// Generate the keys using the from and until buffers that
		// have been set up.

		if (!pucFromBuf && !pucUntilBuf)
		{
			
			// setup a first-to-last key

			flmSetupFirstToLastKey( pucFromKeyLenPos, puiFromKeyLen,
											pucUntilKeyLenPos, puiUntilKeyLen);
			goto Exit;
		}

		// Set up the from key
		
		if (!pucFromBuf)
		{
			uiFromKeyLen = KEY_LOW_VALUE;
		}
		else
		{
			if( !pBufferIStream)
			{
				if( RC_BAD( rc = FlmAllocBufferIStream( &pBufferIStream)))
				{
					goto Exit;
				}
			}
			
			if (RC_BAD( rc = pBufferIStream->openStream( 
				(const char *)pucFromBuf, uiFromBufLen)))
			{
				goto Exit;
			}

			uiFromKeyLen = XFLM_MAX_KEY_SIZE - 2;
			bDataTruncated = FALSE;
			
			// Pass 0 for compare rules on non-text component.
			
			if (RC_BAD( rc = KYCollateValue( pucFromKey, &uiFromKeyLen,
									pBufferIStream, icdGetDataType( pIcd),
									pIcd->uiFlags, 0,
									pIcd->uiLimit, NULL, NULL, 
									pIxd->uiLanguage, FALSE, FALSE,
									&bDataTruncated, NULL)))
			{
				goto Exit;
			}
			
			pBufferIStream->closeStream();

			if (bDataTruncated)
			{
				*pbCanCompareOnKey = FALSE;
				
				// Save the original data into pFromSearchKey so the comparison
				// routines can do a comparison on the full value if
				// necessary.

				// Better only be a binary data type at this point.

				flmAssert( pFromValue->eValType == XFLM_BINARY_VAL);
				if (RC_BAD( rc = pFromSearchKey->setBinary( pIcd->uiKeyComponent - 1,
											pucFromBuf, uiFromBufLen)))
				{
					goto Exit;
				}
				uiFromFlags |= (SEARCH_KEY_FLAG | TRUNCATED_FLAG);
			}
		}

		// Set up the until key

		if (!pucUntilBuf)
		{
			uiUntilKeyLen = KEY_HIGH_VALUE;
		}
		else if (pucUntilBuf == pucFromBuf)
		{
			if (uiFromKeyLen)
			{
				f_memcpy( pucUntilKey, pucFromKey, uiFromKeyLen);
			}
			uiUntilKeyLen = uiFromKeyLen;
			
			// The "exclusive" flags better not have been set in this
			// case - because this should only be possible if the operator
			// was an EQ.
			
			flmAssert( !(uiFromFlags & EXCLUSIVE_GT_FLAG) &&
						  !(uiUntilFlags & EXCLUSIVE_LT_FLAG));
			
			if (uiFromFlags & SEARCH_KEY_FLAG)
			{
				
				// Save the original data into pUntilSearchKey so the comparison
				// routines can do a comparison on the full value if
				// necessary.

				// Better only be a binary data type at this point.

				flmAssert( pUntilValue->eValType == XFLM_BINARY_VAL);
				if (RC_BAD( rc = pUntilSearchKey->setBinary( pIcd->uiKeyComponent - 1,
											pucUntilBuf, uiUntilBufLen)))
				{
					goto Exit;
				}
				uiUntilFlags |= (SEARCH_KEY_FLAG | TRUNCATED_FLAG);
			}
		}
		else
		{
			if( !pBufferIStream)
			{
				if( RC_BAD( rc = FlmAllocBufferIStream( &pBufferIStream)))
				{
					goto Exit;
				}
			}
			
			if (RC_BAD( rc = pBufferIStream->openStream( 
				(const char *)pucUntilBuf, uiUntilBufLen)))
			{
				goto Exit;
			}
			uiUntilKeyLen = XFLM_MAX_KEY_SIZE - 2;
			bDataTruncated = FALSE;
			
			// Pass 0 for compare rule because it is a non-text piece.
			
			if (RC_BAD( rc = KYCollateValue( pucUntilKey, &uiUntilKeyLen,
									pBufferIStream, icdGetDataType( pIcd),
									pIcd->uiFlags, 0,
									pIcd->uiLimit, NULL, NULL, 
									pIxd->uiLanguage, FALSE, FALSE,
									&bDataTruncated, NULL)))
			{
				goto Exit;
			}
			
			pBufferIStream->closeStream();

			if (bDataTruncated)
			{
				*pbCanCompareOnKey = FALSE;
				
				// Save the original data into pUntilSearchKey so the comparison
				// routines can do a comparison on the full value if
				// necessary.

				// Better only be a binary data type at this point.

				flmAssert( pUntilValue->eValType == XFLM_BINARY_VAL);
				if (RC_BAD( rc = pUntilSearchKey->setBinary( pIcd->uiKeyComponent - 1,
											pucUntilBuf, uiUntilBufLen)))
				{
					goto Exit;
				}
				uiUntilFlags |= (SEARCH_KEY_FLAG | TRUNCATED_FLAG);
			}
		}
	}

	UW2FBA( (FLMUINT16)(uiFromKeyLen | uiFromFlags), pucFromKeyLenPos);
	UW2FBA( (FLMUINT16)(uiUntilKeyLen | uiUntilFlags), pucUntilKeyLenPos);
	
	if (!(uiFromFlags & EXCLUSIVE_GT_FLAG) && uiFromKeyLen < XFLM_MAX_KEY_SIZE - 2)
	{
		uiFromKeyLen += kyAddInclComponent( pIcd, &pucFromKey [uiFromKeyLen],
										TRUE, XFLM_MAX_KEY_SIZE - 2 - uiFromKeyLen);
	}
	if (!(uiUntilFlags & EXCLUSIVE_LT_FLAG) && uiUntilKeyLen < XFLM_MAX_KEY_SIZE - 2)
	{
		uiUntilKeyLen += kyAddInclComponent( pIcd, &pucUntilKey [uiUntilKeyLen],
										FALSE, XFLM_MAX_KEY_SIZE - 2 - uiUntilKeyLen);
	}
		
	// Set the FROM and UNTIL key length return values.

	if (uiFromKeyLen != KEY_HIGH_VALUE && uiFromKeyLen != KEY_LOW_VALUE)
	{
		*puiFromKeyLen = uiFromKeyLen + 2;
	}
	else
	{
		*puiFromKeyLen = 2;
	}
	if (uiUntilKeyLen != KEY_HIGH_VALUE && uiUntilKeyLen != KEY_LOW_VALUE)
	{
		*puiUntilKeyLen = uiUntilKeyLen + 2;
	}
	else
	{
		*puiUntilKeyLen = 2;
	}
	
Exit:

	if( pBufferIStream)
	{
		pBufferIStream->Release();
	}

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
	RCODE					rc = NE_XFLM_OK;
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
			
			uiCompareRules &= (~(XFLM_COMP_IGNORE_LEADING_SPACE));
			if (uzChar == ASCII_BACKSLASH)
			{
	
				// Skip the escaped character
	
				if (RC_BAD( rc = f_getCharFromUTF8Buf( &pucValue, NULL, &uzChar)))
				{
					goto Exit;
				}
	
				if (!uzChar)
				{
					rc = RC_SET( NE_XFLM_SYNTAX);
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
	RCODE					rc = NE_XFLM_OK;
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
				
				if (!(uiCompareRules & XFLM_COMP_IGNORE_TRAILING_SPACE))
				{
					if (uiCompareRules & XFLM_COMP_COMPRESS_WHITESPACE)
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
			
			uiCompareRules &= (~(XFLM_COMP_IGNORE_LEADING_SPACE));
			if (bLastCharWasSpace)
			{
				bLastCharWasSpace = FALSE;
				if (uiCompareRules & XFLM_COMP_COMPRESS_WHITESPACE)
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
	RCODE					rc = NE_XFLM_OK;
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
	FLMUINT *	puiFromKeyLen,
	FLMUINT		uiCaseLen,
	FLMBOOL		bIsDBCS,
	FLMBOOL		bAscending,
	FLMBOOL		bExcl)
{
			
	// Subtract off all but the case marker.
	// Remember that for DBCS (Asian) the case marker is two bytes.

	*puiFromKeyLen -= (uiCaseLen -
							 ((FLMUINT)(bIsDBCS
										  ? (FLMUINT)2
										  : (FLMUINT)1)));
	if (bExcl)
	{
		if (bAscending)
		{
			// Keys are in ascending order:
			// "abc" key == abc+4 (4 is F_COLL_MARKER | F_SC_LOWER) 
			// "ABC" key == abc+6 (6 is F_COLL_MARKER | F_SC_UPPER)
			// Thus, to exclude all "abc"s on "from" side we need the
			// following key:
			// key == abc+6 (F_COLL_MARKER | F_SC_UPPER) + 1
			
			pucFromKey[ *puiFromKeyLen - 1] = (F_COLL_MARKER | F_SC_UPPER);
		}
		else
		{
			
			// Keys are in descending order:
			// "ABC" key == abc+6 (6 is F_COLL_MARKER | F_SC_UPPER)
			// "abc" key == abc+4 (4 is F_COLL_MARKER | F_SC_LOWER) 
			// Thus, to exclude "abc"s on "from" side we need the
			// following key:
			// key == abc+4 (F_COLL_MARKER | F_SC_LOWER)
			
			pucFromKey[ *puiFromKeyLen - 1] = (F_COLL_MARKER | F_SC_LOWER);
		}
	}
	else	// Inclusive
	{
		if (bAscending)
		{
			// Keys are in ascending order:
			// "abc" key == abc+4 (4 is F_COLL_MARKER | F_SC_LOWER) 
			// "ABC" key == abc+6 (6 is F_COLL_MARKER | F_SC_UPPER)
			// Thus, to include all "abc"s on "from" side,
			// we need the following key:
			// key == abc+4 (F_COLL_MARKER | F_SC_LOWER)
			
			pucFromKey [*puiFromKeyLen - 1] = F_COLL_MARKER | F_SC_LOWER;
		}
		else
		{

			// Keys are in descending order:
			// "ABC" key == abc+6 (6 is F_COLL_MARKER | F_SC_UPPER)
			// "abc" key == abc+4 (4 is F_COLL_MARKER | F_SC_LOWER) 
			// Thus, to include all "abc"s on "from" side we need the
			// following key:
			// key == abc+6 (F_COLL_MARKER | F_SC_UPPER)
			
			pucFromKey [*puiFromKeyLen - 1] = F_COLL_MARKER | F_SC_UPPER;
		}
	}
}

/****************************************************************************
Desc:	Set the case byte on the until key for a case-insensitive search
		that is using a case sensitive index.
****************************************************************************/
FSTATIC void setUntilCaseByte(
	FLMBYTE *	pucUntilKey,
	FLMUINT *	puiUntilKeyLen,
	FLMUINT		uiCaseLen,
	FLMBOOL		bIsDBCS,
	FLMBOOL		bAscending,
	FLMBOOL		bExcl)
{
			
	// Subtract off all but the case marker.
	// Remember that for DBCS (Asian) the case marker is two bytes.

	*puiUntilKeyLen -= (uiCaseLen -
							  ((FLMUINT)(bIsDBCS
										    ? (FLMUINT)2
										    : (FLMUINT)1)));
	if (bExcl)
	{
		if (bAscending)
		{
			// Keys are in ascending order:
			// "abc" key == abc+4 (4 is F_COLL_MARKER | F_SC_LOWER) 
			// "ABC" key == abc+6 (6 is F_COLL_MARKER | F_SC_UPPER)
			// Thus, to exclude all "abc"s on the "until" side we need
			// the following key:
			// key == abc+4 (F_COLL_MARKER | F_SC_LOWER)
			
			pucUntilKey[ *puiUntilKeyLen - 1] = (F_COLL_MARKER | F_SC_LOWER);
		}
		else
		{
			
			// Keys are in descending order:
			// "ABC" key == abc+6 (6 is F_COLL_MARKER | F_SC_UPPER)
			// "abc" key == abc+4 (4 is F_COLL_MARKER | F_SC_LOWER) 
			// Thus, to exclude all "abc"s on the "until" side we need
			// the following key:
			// key == abc+6 (F_COLL_MARKER | F_SC_UPPER) + 1
			
			pucUntilKey[ *puiUntilKeyLen - 1] = (F_COLL_MARKER | F_SC_UPPER);
		}
	}
	else
	{
		if (bAscending)
		{
			// Keys are in ascending order:
			// "abc" key == abc+4 (4 is F_COLL_MARKER | F_SC_LOWER) 
			// "ABC" key == abc+6 (6 is F_COLL_MARKER | F_SC_UPPER)
			// Thus, to get include all "abc"s on the "until" side we need
			// the following key:
			// key == abc+6 (F_COLL_MARKER | F_SC_UPPER)
			
			pucUntilKey [*puiUntilKeyLen - 1] = (F_COLL_MARKER | F_SC_UPPER);
		}
		else
		{

			// Keys are in descending order:
			// "ABC" key == abc+6 (6 is F_COLL_MARKER | F_SC_UPPER)
			// "abc" key == abc+4 (4 is F_COLL_MARKER | F_SC_LOWER) 
			// Thus, to include all "abc"s on the "until side we need
			// the following key:
			// key == abc+4 (F_COLL_MARKER | F_SC_LOWER)
			
			pucUntilKey [*puiUntilKeyLen - 1] = (F_COLL_MARKER | F_SC_LOWER);
		}
	}
}

/****************************************************************************
Desc:	Build a text key.
****************************************************************************/
FSTATIC RCODE flmAddTextKeyPiece(
	PATH_PRED *		pPred,
	IXD *				pIxd,
	ICD *				pIcd,
	F_DataVector *	pFromSearchKey,
	FLMBYTE *		pucFromKey,
	FLMUINT *		puiFromKeyLen,
	F_DataVector *	pUntilSearchKey,
	FLMBYTE *		pucUntilKey,
	FLMUINT *		puiUntilKeyLen,
	FLMBOOL *		pbCanCompareOnKey)
{
	RCODE       			rc = NE_XFLM_OK;
	FLMUINT					uiFromKeyLen = 0;
	FLMUINT					uiUntilKeyLen = 0;
	FLMBYTE *				pucFromKeyLenPos = pucFromKey;
	FLMBYTE *				pucUntilKeyLenPos = pucUntilKey;
	FLMUINT					uiLanguage = pIxd->uiLanguage;
	FLMUINT					uiCollationLen = 0;
	FLMUINT					uiCharCount;
	FLMUINT					uiCaseLen;
	FLMBOOL					bOriginalCharsLost = FALSE;
	FLMBOOL					bIsDBCS = (uiLanguage >= FLM_FIRST_DBCS_LANG &&
								  uiLanguage <= FLM_LAST_DBCS_LANG)
								  ? TRUE
								  : FALSE;

	FLMBOOL					bCaseInsensitive = (FLMBOOL)((pPred->uiCompareRules &
															XFLM_COMP_CASE_INSENSITIVE)
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
	FLMBOOL					bDataTruncated;
	FLMUINT					uiFromFlags = 0;
	FLMUINT					uiUntilFlags = 0;
	FLMUINT					uiCompareRules;
	FLMBOOL					bAscending = (pIcd->uiFlags & ICD_DESCENDING) ? FALSE: TRUE;
	IF_BufferIStream *	pBufferIStream = NULL;

	switch (pPred->eOperator)
	{

		// The difference between MATCH and EQ_OP is that EQ does
		// not support wildcards embedded in the search key.

		case XFLM_MATCH_OP:
			flmAssert( pPred->pFromValue->eValType == XFLM_UTF8_VAL);
			pucFromUTF8Buf = pPred->pFromValue->val.pucBuf;
			uiFromBufLen = pPred->pFromValue->uiDataLen;
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

		case XFLM_RANGE_OP:
			if (bAscending)
			{
				if (pPred->pFromValue)
				{
					flmAssert( pPred->pFromValue->eValType == XFLM_UTF8_VAL);
					pucFromUTF8Buf = pPred->pFromValue->val.pucBuf;
					uiFromBufLen = pPred->pFromValue->uiDataLen;
				}
				else
				{
					// Should have been done up above
					
					// pucFromUTF8Buf = NULL;
					// uiFromBufLen = 0;
				}
				if (pPred->pUntilValue)
				{
					flmAssert( pPred->pUntilValue->eValType == XFLM_UTF8_VAL);
					pucUntilUTF8Buf = pPred->pUntilValue->val.pucBuf;
					uiUntilBufLen = pPred->pUntilValue->uiDataLen;
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
					flmAssert( pPred->pUntilValue->eValType == XFLM_UTF8_VAL);
					pucFromUTF8Buf = pPred->pUntilValue->val.pucBuf;
					uiFromBufLen = pPred->pUntilValue->uiDataLen;
				}
				else
				{
					// Should have been done up above
					
					// pucFromUTF8Buf = NULL;
					// uiFromBufLen = 0;
				}
					
				if (pPred->pFromValue)
				{
					flmAssert( pPred->pFromValue->eValType == XFLM_UTF8_VAL);
					pucUntilUTF8Buf = pPred->pFromValue->val.pucBuf;
					uiUntilBufLen = pPred->pFromValue->uiDataLen;
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

		case XFLM_NE_OP:

			// Set up to do full index scan.
			
			// Buffers should already be NULL.

			// pucFromUTF8Buf = NULL;
			// pucUntilUTF8Buf = NULL;
			break;

		case XFLM_APPROX_EQ_OP:

			// Set up to do full index scan.
			
			// Buffers should already be NULL

			// pucFromUTF8Buf = NULL;
			// pucUntilUTF8Buf = NULL;

			// Cannot compare on the key if index is upper case,
			// even if the bCaseInsensitive flag is set.

			if (pIcd->uiCompareRules & XFLM_COMP_CASE_INSENSITIVE)
			{
				*pbCanCompareOnKey = FALSE;
			}
			break;

		default:

			// Every predicate should have been converted to one of the above
			// cases, or should be handled by another routine.

			rc = RC_SET_AND_ASSERT( NE_XFLM_QUERY_SYNTAX);
			goto Exit;
	}

	// If index is case insensitive, but search is case sensitive
	// we must NOT do a key match - we would fail things we should
	// not be failing.
	
	if ((pIcd->uiCompareRules & XFLM_COMP_CASE_INSENSITIVE) && !bCaseInsensitive)
	{
		*pbCanCompareOnKey = FALSE;
	}
	
	if (!pucFromUTF8Buf && !pucUntilUTF8Buf)
	{
		
		// setup a first-to-last key
		
		flmSetupFirstToLastKey( pucFromKeyLenPos, puiFromKeyLen,
										pucUntilKeyLenPos, puiUntilKeyLen);
		goto Exit;
	}

	pucFromKey += 2;
	pucUntilKey += 2;
	if (!pucFromUTF8Buf)
	{
		uiFromKeyLen = KEY_LOW_VALUE;
	}
	else
	{
		if( !pBufferIStream)
		{
			if( RC_BAD( rc = FlmAllocBufferIStream( &pBufferIStream)))
			{
				goto Exit;
			}
		}
		
		if (RC_BAD( rc = pBufferIStream->openStream( 
			(const char *)pucFromUTF8Buf, uiFromBufLen)))
		{
			goto Exit;
		}

		// Add ICD_ESC_CHAR to the icd flags because
		// the search string must have BACKSLASHES and '*' escaped.

		uiFromKeyLen = XFLM_MAX_KEY_SIZE - 2;
		bDataTruncated = FALSE;
		if (RC_BAD( rc = KYCollateValue( pucFromKey, &uiFromKeyLen,
								pBufferIStream, XFLM_TEXT_TYPE,
							pIcd->uiFlags | ICD_ESC_CHAR, pIcd->uiCompareRules,
							pIcd->uiLimit,
							&uiCollationLen, &uiCaseLen,
							uiLanguage, bDoFirstSubstring,
							FALSE, &bDataTruncated,
							&bOriginalCharsLost)))
		{
			goto Exit;
		}
		
		pBufferIStream->closeStream();
		
		if (bDataTruncated)
		{
			*pbCanCompareOnKey = FALSE;
			
			// Save the original data into pFromSearchKey so the comparison
			// routines can do a comparison on the full value if
			// necessary.

			if (RC_BAD( rc = pFromSearchKey->setUTF8( pIcd->uiKeyComponent - 1,
										pucFromUTF8Buf, uiFromBufLen)))
			{
				goto Exit;
			}
			uiFromFlags |= (SEARCH_KEY_FLAG | TRUNCATED_FLAG);
		}
		else if (bOriginalCharsLost)
		{
			*pbCanCompareOnKey = FALSE;
		}
		
		if (pucFromUTF8Buf != pucUntilUTF8Buf)
		{
			
			// Handle scenario of a case-sensitive index, but search is
			// case-insensitive.

			if (uiFromKeyLen &&
				 (bIsDBCS ||
				   (!(pIcd->uiCompareRules & XFLM_COMP_CASE_INSENSITIVE) &&
					 bCaseInsensitive)))
			{
				setFromCaseByte( pucFromKey, &uiFromKeyLen, uiCaseLen,
										bIsDBCS, bAscending,
										(uiFromFlags & EXCLUSIVE_GT_FLAG)
										? TRUE
										: FALSE);
			}
		}
		else
		{
			// Handle case where from and until buffers are the same.
			// This should only be possible in the equality case or match begin
			// case, in which cases neither the EXCLUSIVE_LT_FLAG or the
			// EXCLUSIVE_GT_FLAG should be set.
			
			flmAssert( uiFromBufLen == uiUntilBufLen);
			flmAssert( !(uiFromFlags & (EXCLUSIVE_GT_FLAG | EXCLUSIVE_LT_FLAG)));
			flmAssert( !(uiUntilFlags & (EXCLUSIVE_GT_FLAG | EXCLUSIVE_LT_FLAG)));
			
			if (uiFromFlags & SEARCH_KEY_FLAG)
			{
				
				// Save the original data into pUntilSearchKey so the comparison
				// routines can do a comparison on the full value if
				// necessary.
	
				if (RC_BAD( rc = pUntilSearchKey->setUTF8( pIcd->uiKeyComponent - 1,
											pucUntilUTF8Buf, uiUntilBufLen)))
				{
					goto Exit;
				}
				uiUntilFlags |= (SEARCH_KEY_FLAG | TRUNCATED_FLAG);
			}
			
			if (bDoMatchBegin)
			{
				if (bAscending)
				{
					
					// Handle scenario of a case-sensitive index, but search is
					// case-insensitive.
			
					if (uiFromKeyLen &&
						 (bIsDBCS ||
						  (!(pIcd->uiCompareRules & XFLM_COMP_CASE_INSENSITIVE) &&
						   bCaseInsensitive)))
					{
						setFromCaseByte( pucFromKey, &uiFromKeyLen, uiCaseLen,
												bIsDBCS, bAscending, FALSE);
					}
		
					// From key is set up properly, setup until key.
					
					if (uiCollationLen)
					{
						f_memcpy( pucUntilKey, pucFromKey, uiCollationLen);
					}
					
					// Fill the rest of the until key with high values.
		
					f_memset( &pucUntilKey[ uiCollationLen], 0xFF,
									XFLM_MAX_KEY_SIZE - uiCollationLen - 2);
					uiUntilKeyLen = XFLM_MAX_KEY_SIZE - 2;
				}
				else
				{
					
					// Copy from key into until key.

					if (uiFromKeyLen)
					{
						f_memcpy( pucUntilKey, pucFromKey, uiFromKeyLen);
					}
					uiUntilKeyLen = uiFromKeyLen;
					
					if (uiUntilKeyLen &&
						 (bIsDBCS ||
						  (!(pIcd->uiCompareRules & XFLM_COMP_CASE_INSENSITIVE) &&
						   bCaseInsensitive)))
					{
						// NOTE: Always inclusive because this is a matchbegin.
					
						setUntilCaseByte( pucUntilKey, &uiUntilKeyLen, uiCaseLen,
											bIsDBCS, bAscending, FALSE);
					}
										
					// Fill rest of from key with high values after collation values.
					
					f_memset( &pucFromKey[ uiCollationLen], 0xFF,
								XFLM_MAX_KEY_SIZE - uiCollationLen - 2);
					uiFromKeyLen = XFLM_MAX_KEY_SIZE - 2;
				}
			}
			else
			{
				
				// Copy from key into until key.

				if (!uiFromKeyLen)
				{
					uiUntilKeyLen = 0;
				}
				else
				{
					if (!bDoFirstSubstring)
					{
						f_memcpy( pucUntilKey, pucFromKey, uiFromKeyLen);
						uiUntilKeyLen = uiFromKeyLen;
					}
					else if (bAscending)
					{
						
						// Do two copies so that the first substring byte is gone
						// in the until key.
						
						f_memcpy( pucUntilKey, pucFromKey, uiCollationLen);
						uiUntilKeyLen = uiCollationLen;
						if (bIsDBCS)
						{
							uiCollationLen++;
						}
						uiCollationLen++;
						f_memcpy( &pucUntilKey [uiUntilKeyLen],
										pucFromKey + uiCollationLen,
										uiFromKeyLen - uiCollationLen);
						uiUntilKeyLen += (uiFromKeyLen - uiCollationLen);
					}
					else
					{
						
						// Descending order - put the string without the
						// first-substring-marker into the from key instead of
						// the until key.
						
						f_memcpy( pucUntilKey, pucFromKey, uiFromKeyLen);
						uiUntilKeyLen = uiFromKeyLen;
						
						// Modify from key to NOT have first-substring-marker.
						
						f_memcpy( pucUntilKey, pucFromKey, uiCollationLen);
						uiFromKeyLen = uiCollationLen;
						if (bIsDBCS)
						{
							uiCollationLen++;
						}
						uiCollationLen++;
						f_memcpy( &pucFromKey [uiFromKeyLen],
										pucUntilKey + uiCollationLen,
										uiUntilKeyLen - uiCollationLen);
						uiFromKeyLen += (uiUntilKeyLen - uiCollationLen);
					}
				
					// Handle scenario of a case-sensitive index, but search is
					// case-insensitive.
			
					if (bIsDBCS ||
						 (!(pIcd->uiCompareRules & XFLM_COMP_CASE_INSENSITIVE) &&
						  bCaseInsensitive))
					{
						setFromCaseByte( pucFromKey, &uiFromKeyLen, uiCaseLen,
													bIsDBCS, bAscending, FALSE);
						setUntilCaseByte( pucUntilKey, &uiUntilKeyLen, uiCaseLen,
													bIsDBCS, bAscending, FALSE);
					}
				}
			}
		}
	}

	// Do the until key now

	if (!pucUntilUTF8Buf)
	{
		uiUntilKeyLen = KEY_HIGH_VALUE;
	}
	else if (pucFromUTF8Buf != pucUntilUTF8Buf)
	{
		if( !pBufferIStream)
		{
			if( RC_BAD( rc = FlmAllocBufferIStream( &pBufferIStream)))
			{
				goto Exit;
			}
		}
		
		if (RC_BAD( rc = pBufferIStream->openStream( 
			(const char *)pucUntilUTF8Buf, uiUntilBufLen)))
		{
			goto Exit;
		}

		// Add ICD_ESC_CHAR to the icd flags because
		// the search string must have BACKSLASHES and '*' escaped.

		uiUntilKeyLen = XFLM_MAX_KEY_SIZE - 2;
		bDataTruncated = FALSE;
		if (RC_BAD( rc = KYCollateValue( pucUntilKey, &uiUntilKeyLen,
							pBufferIStream, XFLM_TEXT_TYPE,
							pIcd->uiFlags | ICD_ESC_CHAR, pIcd->uiCompareRules,
							pIcd->uiLimit,
							&uiCollationLen, &uiCaseLen,
							uiLanguage, bDoFirstSubstring, 
							FALSE, &bDataTruncated,
							&bOriginalCharsLost)))
		{
			goto Exit;
		}
		
		pBufferIStream->closeStream();
		
		if (bDataTruncated)
		{
			
			// Save the original data into pUntilSearchKey so the comparison
			// routines can do a comparison on the full value if
			// necessary.

			if (RC_BAD( rc = pUntilSearchKey->setUTF8( pIcd->uiKeyComponent - 1,
										pucUntilUTF8Buf, uiUntilBufLen)))
			{
				goto Exit;
			}
			*pbCanCompareOnKey = FALSE;
			uiUntilFlags |= (SEARCH_KEY_FLAG | TRUNCATED_FLAG);
		}
		else if (bOriginalCharsLost)
		{
			*pbCanCompareOnKey = FALSE;
		}

		if (uiUntilKeyLen &&
			 (bIsDBCS ||
			  (!(pIcd->uiCompareRules & XFLM_COMP_CASE_INSENSITIVE) &&
			   bCaseInsensitive)))
		{
			setUntilCaseByte( pucUntilKey, &uiUntilKeyLen, uiCaseLen,
									bIsDBCS, bAscending,
									(uiUntilFlags & EXCLUSIVE_LT_FLAG)
									? TRUE
									: FALSE);
		}
	}
	
	UW2FBA( (FLMUINT16)(uiFromKeyLen | uiFromFlags), pucFromKeyLenPos);
	UW2FBA( (FLMUINT16)(uiUntilKeyLen | uiUntilFlags), pucUntilKeyLenPos);
	
	if (!(uiFromFlags & EXCLUSIVE_GT_FLAG) && uiFromKeyLen < XFLM_MAX_KEY_SIZE - 2)
	{
		uiFromKeyLen += kyAddInclComponent( pIcd, &pucFromKey [uiFromKeyLen],
										TRUE, XFLM_MAX_KEY_SIZE - 2 - uiFromKeyLen);
	}
	if (!(uiUntilFlags & EXCLUSIVE_LT_FLAG) && uiUntilKeyLen < XFLM_MAX_KEY_SIZE - 2)
	{
		uiUntilKeyLen += kyAddInclComponent( pIcd, &pucUntilKey [uiUntilKeyLen],
										FALSE, XFLM_MAX_KEY_SIZE - 2 - uiUntilKeyLen);
	}
		
	// Set the FROM and UNTIL key lengths

	if (uiFromKeyLen != KEY_HIGH_VALUE && uiFromKeyLen != KEY_LOW_VALUE)
	{
		*puiFromKeyLen = uiFromKeyLen + 2;
	}
	else
	{
		*puiFromKeyLen = 2;
	}
	if (uiUntilKeyLen != KEY_HIGH_VALUE && uiUntilKeyLen != KEY_LOW_VALUE)
	{
		*puiUntilKeyLen = uiUntilKeyLen + 2;
	}
	else
	{
		*puiUntilKeyLen = 2;
	}
	
Exit:

	if( pBufferIStream)
	{
		pBufferIStream->Release();
	}
	
	return( rc);
}

/****************************************************************************
Desc:		Build the from and until keys given a field list with operators and
			values and an index.
Notes:	The knowledge of query definitions is limited in these routines.
****************************************************************************/
RCODE flmBuildFromAndUntilKeys(
	IXD *				pIxd,
	PATH_PRED *		pPred,
	F_DataVector *	pFromSearchKey,
	FLMBYTE *		pucFromKey,
	FLMUINT *		puiFromKeyLen,
	F_DataVector *	pUntilSearchKey,
	FLMBYTE *		pucUntilKey,
	FLMUINT *		puiUntilKeyLen,
	FLMBOOL *		pbDoNodeMatch,
	FLMBOOL *		pbCanCompareOnKey)
{
	RCODE	rc = NE_XFLM_OK;
	ICD *	pIcd = pIxd->pFirstKey;
								
	*puiFromKeyLen = *puiUntilKeyLen = 0;
	*pbDoNodeMatch = FALSE;
	*pbCanCompareOnKey = TRUE;

	if (!pPred)
	{
		
		// Setup a first-to-last key
		
		flmSetupFirstToLastKey( pucFromKey, puiFromKeyLen,
										pucUntilKey, puiUntilKeyLen);
		*pbDoNodeMatch = TRUE;
		*pbCanCompareOnKey = FALSE;
	}
	else
	{

		// Predicates we are looking at should NEVER be notted.  They
		// will have been weeded out earlier.

		flmAssert( !pPred->bNotted);
	
		// Handle special cases for indexing presence and/or exists predicate.


		if (icdGetDataType( pIcd) == XFLM_TEXT_TYPE &&
					!(pIcd->uiFlags & (ICD_PRESENCE | ICD_METAPHONE)) &&
					pPred->eOperator != XFLM_EXISTS_OP)
		{
			if (RC_BAD( rc = flmAddTextKeyPiece( pPred, pIxd,
						pIcd, pFromSearchKey, pucFromKey, puiFromKeyLen,
						pUntilSearchKey, pucUntilKey, puiUntilKeyLen,
						pbCanCompareOnKey)))
			{
				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = flmAddNonTextKeyPiece( pPred, pIxd, pIcd,
										pFromSearchKey, pucFromKey, puiFromKeyLen,
										pUntilSearchKey, pucUntilKey, puiUntilKeyLen,
										pbCanCompareOnKey)))
			{
				goto Exit;
			}
		}
	}

Exit:

	if (!(*pbCanCompareOnKey))
	{
		*pbDoNodeMatch = TRUE;
	}

	return( rc);
}

