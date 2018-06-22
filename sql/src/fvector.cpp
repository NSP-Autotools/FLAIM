//------------------------------------------------------------------------------
// Desc:	Contains the code for the F_DataVector class.
// Tabs:	3
//
// Copyright (c) 2003-2007 Novell, Inc. All Rights Reserved.
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

#if defined( FLM_WATCOM_NLM)
	// Disable "Warning! W549: col(XX) 'sizeof' operand contains
	// compiler generated information"
	#pragma warning 549 9
#endif

/****************************************************************************
Desc:
****************************************************************************/
F_DataVector::F_DataVector()
{
	m_pVectorElements = &m_VectorArray [0];
	m_uiVectorArraySize = MIN_VECTOR_ELEMENTS;
	m_pucDataBuf = m_ucIntDataBuf;
	m_uiDataBufLength = sizeof( m_ucIntDataBuf);
	reset();
}

/****************************************************************************
Desc:
****************************************************************************/
F_DataVector::~F_DataVector()
{
	if (m_pVectorElements != &m_VectorArray [0])
	{
		f_free( &m_pVectorElements);
	}
	if (m_pucDataBuf && m_pucDataBuf != m_ucIntDataBuf)
	{
		f_free( &m_pucDataBuf);
	}
	reset();
}


/****************************************************************************
Desc:	Clear the data vector, but don't free any buffers that have been
		allocated.  That will only happen in the destructor.  The reset()
		method is so that we can get efficient re-use of the vector.  So, if
		it has allocated buffers, etc. we don't want to free them.
****************************************************************************/
void F_DataVector::reset( void)
{
	m_ui64RowId = 0;
	m_uiNumElements = 0;
	m_uiDataBufOffset = 0;
}

/****************************************************************************
Desc:	Make sure the vector array is allocated at least up to the element
		number that is passed in.
****************************************************************************/
RCODE F_DataVector::allocVectorArray(
	FLMUINT	uiElementNumber)
{
	RCODE	rc = NE_SFLM_OK;

	if (uiElementNumber >= m_uiNumElements)
	{

		// May need to allocate a new vector array

		if (uiElementNumber >= m_uiVectorArraySize)
		{
			FLMUINT					uiNewArraySize = uiElementNumber + 32;
			F_VECTOR_ELEMENT *	pNewVector;

			if (m_pVectorElements == &m_VectorArray [0])
			{
				if (RC_BAD( rc = f_alloc( uiNewArraySize * sizeof( F_VECTOR_ELEMENT),
											&pNewVector)))
				{
					goto Exit;
				}
				if (m_uiNumElements)
				{
					f_memcpy( pNewVector, m_pVectorElements,
						m_uiNumElements * sizeof( F_VECTOR_ELEMENT));
				}
			}
			else
			{
				pNewVector = m_pVectorElements;

				if (RC_BAD( rc = f_realloc( uiNewArraySize * sizeof( F_VECTOR_ELEMENT),
											&pNewVector)))
				{
					goto Exit;
				}

			}
			m_pVectorElements = pNewVector;
			m_uiVectorArraySize = uiNewArraySize;
		}

		// Initialized everything between the old last element and
		// the new element, including the new element, to zeroes.

		f_memset( &m_pVectorElements [m_uiNumElements], 0,
			sizeof( F_VECTOR_ELEMENT) *
			(uiElementNumber - m_uiNumElements + 1));

		m_uiNumElements = uiElementNumber + 1;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Store a data value into its vector element.
****************************************************************************/
RCODE F_DataVector::storeValue(
	FLMINT				uiElementNumber,
	eDataType			eDataTyp,
	const FLMBYTE *	pucData,
	FLMUINT				uiDataLen,
	FLMBYTE **			ppucDataPtr)
{
	RCODE						rc = NE_SFLM_OK;
	F_VECTOR_ELEMENT *	pVector;
	FLMBYTE *				pucDataPtr;
	FLMUINT					uiTemp;

	// Find or allocate space for the vector

	if (RC_BAD( rc = allocVectorArray( uiElementNumber)))
	{
		goto Exit;
	}

	pVector = &m_pVectorElements [uiElementNumber];

	// Will the data fit inside uiDataOffset?

	if (uiDataLen <= sizeof( FLMUINT))
	{
		pucDataPtr = (FLMBYTE *)&pVector->uiDataOffset;
	}
	else if (uiDataLen <= pVector->uiDataLength)
	{

		// New data will fit in original space.  Simply reuse it.

		pucDataPtr = m_pucDataBuf + pVector->uiDataOffset;
	}
	else
	{

		// New data will not fit in originally allocated space.
		// Must allocate new space.

		// Always align the new allocation so that if it gets
		// reused later for binary data it will be properly aligned.

		if ((m_uiDataBufOffset & FLM_ALLOC_ALIGN) != 0)
		{
			uiTemp = (FLM_ALLOC_ALIGN + 1) - (m_uiDataBufOffset & FLM_ALLOC_ALIGN);
			m_uiDataBufOffset += uiTemp;
		}

		if (uiDataLen + m_uiDataBufOffset > m_uiDataBufLength)
		{
			// Re-allocate the data buffer.

			if( m_pucDataBuf == m_ucIntDataBuf)
			{
				if (RC_BAD( rc = f_alloc(
								m_uiDataBufOffset + uiDataLen + 512,
								&m_pucDataBuf)))
				{
					goto Exit;
				}

				f_memcpy( m_pucDataBuf, m_ucIntDataBuf, m_uiDataBufOffset);
			}
			else
			{
				if (RC_BAD( rc = f_realloc(
								m_uiDataBufOffset + uiDataLen + 512,
								&m_pucDataBuf)))
				{
					goto Exit;
				}
			}

			m_uiDataBufLength = m_uiDataBufOffset + uiDataLen + 512;
		}
		pucDataPtr = m_pucDataBuf + m_uiDataBufOffset;
		pVector->uiDataOffset = m_uiDataBufOffset;
		m_uiDataBufOffset += uiDataLen;
	}

	// Store the data - may be zero length.

	if( pucData)
	{
		if( uiDataLen > 1)
		{
			f_memcpy( pucDataPtr, pucData, uiDataLen);
		}
		else if( uiDataLen)
		{
			*pucDataPtr = *pucData;
		}
	}

	pVector->uiFlags |= VECT_SLOT_HAS_DATA;
	pVector->uiDataLength = uiDataLen;
	pVector->eDataTyp = eDataTyp;

	if( ppucDataPtr)
	{
		*ppucDataPtr = pucDataPtr;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Set the column number for a vector element.
****************************************************************************/
RCODE F_DataVector::setColumnNum(
	FLMUINT	uiElementNumber,
	FLMUINT	uiColumnNum,
	FLMBOOL	bIsData)
{
	RCODE						rc = NE_SFLM_OK;
	F_VECTOR_ELEMENT *	pVector;

	// Find or allocate space for the vector element

	if (RC_BAD( rc = allocVectorArray( uiElementNumber)))
	{
		goto Exit;
	}
	pVector = &m_pVectorElements [uiElementNumber];

	pVector->uiFlags |= VECT_SLOT_HAS_COLUMN_NUM;
	if (bIsData)
	{
		pVector->uiFlags |= VECT_SLOT_IS_DATA;
	}
	else
	{
		pVector->uiFlags &= (~(VECT_SLOT_IS_DATA));
	}
	pVector->uiColumnNum = uiColumnNum;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Set a FLMINT value for a vector element.
****************************************************************************/
RCODE F_DataVector::setINT(
	FLMUINT	uiElementNumber,
	FLMINT	iNum)
{
	RCODE		rc = NE_SFLM_OK;
	FLMBYTE	ucStorageBuf [FLM_MAX_NUM_BUF_SIZE];
	FLMUINT	uiStorageLen;
	FLMBOOL	bNeg = FALSE;

	if (iNum < 0)
	{
		bNeg = TRUE;
		iNum = -iNum;
	}

	uiStorageLen = sizeof( ucStorageBuf);
	if( ((FLMUINT)iNum) <= gv_uiMaxUInt32Val)
	{
		if( RC_BAD( rc = flmNumber64ToStorage( (FLMUINT64)iNum,
			&uiStorageLen, ucStorageBuf, bNeg, FALSE)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = flmNumber64ToStorage( (FLMUINT64)iNum,
			&uiStorageLen, ucStorageBuf, bNeg, FALSE)))
		{
			goto Exit;
		}
	}

	if (RC_BAD( rc = storeValue( uiElementNumber,
		SFLM_NUMBER_TYPE, ucStorageBuf, uiStorageLen)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Set a FLMINT64 value for a vector element.
****************************************************************************/
RCODE F_DataVector::setINT64(
	FLMUINT		uiElementNumber,
	FLMINT64		i64Num)
{
	RCODE		rc = NE_SFLM_OK;
	FLMBYTE	ucStorageBuf [FLM_MAX_NUM_BUF_SIZE];
	FLMUINT	uiStorageLen;
	FLMBOOL	bNeg = FALSE;

	if (i64Num < 0)
	{
		bNeg = TRUE;
		i64Num = -i64Num;
	}

	uiStorageLen = sizeof( ucStorageBuf);
	if (RC_BAD( rc = flmNumber64ToStorage( i64Num, &uiStorageLen,
		ucStorageBuf, bNeg, FALSE)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = storeValue( uiElementNumber,
		SFLM_NUMBER_TYPE, ucStorageBuf, uiStorageLen)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Set a FLMUINT value for a vector element.
****************************************************************************/
RCODE F_DataVector::setUINT(
	FLMUINT	uiElementNumber,
	FLMUINT	uiNum)
{
	RCODE		rc = NE_SFLM_OK;
	FLMBYTE	ucStorageBuf [FLM_MAX_NUM_BUF_SIZE];
	FLMUINT	uiStorageLen;

	uiStorageLen = sizeof( ucStorageBuf);
	if (uiNum <= gv_uiMaxUInt32Val)
	{
		if (RC_BAD( rc = flmNumber64ToStorage( uiNum,
									&uiStorageLen, ucStorageBuf, FALSE, FALSE)))
		{
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = flmNumber64ToStorage( uiNum,
									&uiStorageLen, ucStorageBuf, FALSE, FALSE)))
		{
			goto Exit;
		}
	}

	if (RC_BAD( rc = storeValue( uiElementNumber,
		SFLM_NUMBER_TYPE, ucStorageBuf, uiStorageLen)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Set a FLMUINT64 value for a vector element.
****************************************************************************/
RCODE F_DataVector::setUINT64(
	FLMUINT		uiElementNumber,
	FLMUINT64	ui64Num)
{
	RCODE		rc = NE_SFLM_OK;
	FLMBYTE	ucStorageBuf [FLM_MAX_NUM_BUF_SIZE];
	FLMUINT	uiStorageLen;

	uiStorageLen = sizeof( ucStorageBuf);
	if (RC_BAD( rc = flmNumber64ToStorage( ui64Num,
							&uiStorageLen, ucStorageBuf, FALSE, FALSE)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = storeValue( uiElementNumber,
		SFLM_NUMBER_TYPE, ucStorageBuf, uiStorageLen)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Set a FLMUNICODE value for a vector element.
****************************************************************************/
RCODE F_DataVector::setUnicode(
	FLMUINT					uiElementNumber,
	const FLMUNICODE *	puzUnicode)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBYTE *	pucDataPtr;
	FLMUINT		uiLen;
	FLMUINT		uiCharCount;
	FLMBYTE		ucTmpBuf [64];

	// A NULL or empty puzUnicode string is allowed - on those cases
	// just set the data type.

	if (puzUnicode == NULL || *puzUnicode == 0)
	{
		rc = storeValue( uiElementNumber, SFLM_STRING_TYPE, NULL, 0);
		goto Exit;
	}

	// See if it will fit in our temporary buffer on the stack.

	uiLen = sizeof( ucTmpBuf);
	if (RC_OK( rc = flmUnicode2Storage( puzUnicode, 0, ucTmpBuf,
							&uiLen, &uiCharCount)))
	{
		if (RC_BAD( rc = storeValue( uiElementNumber,
								SFLM_STRING_TYPE, ucTmpBuf, uiLen)))
		{
			goto Exit;
		}
	}
	else if (rc != NE_SFLM_CONV_DEST_OVERFLOW)
	{
		goto Exit;
	}
	else
	{

		// Determine the length needed.

		if (RC_BAD( rc = flmUnicode2Storage( puzUnicode, 0, NULL,
									&uiLen, &uiCharCount)))
		{
			goto Exit;
		}

		// Allocate space for it in the vector and get a pointer
		// back so we can then store it.

		if (RC_BAD( rc = storeValue( uiElementNumber,
								SFLM_STRING_TYPE, NULL, uiLen, &pucDataPtr)))
		{
			goto Exit;
		}

		// Store it out to the space we just allocated.

		if (RC_BAD( rc = flmUnicode2Storage( puzUnicode, uiCharCount,
									pucDataPtr, &uiLen, NULL)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Set a UTF8 value for a vector element.
****************************************************************************/
RCODE F_DataVector::setUTF8(
	FLMUINT				uiElementNumber,
	const FLMBYTE *	pszUTF8,
	FLMUINT				uiBytesInBuffer)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBYTE *	pucDataPtr;
	FLMUINT		uiLen;
	FLMBYTE		ucTmpBuf [64];

	// A NULL or empty pszNative string is allowed - on those cases
	// just set the data type.

	if (pszUTF8 == NULL || *pszUTF8 == 0)
	{
		rc = storeValue( uiElementNumber, SFLM_STRING_TYPE, NULL, 0);
		goto Exit;
	}

	// See if it will fit in our temporary buffer on the stack.

	uiLen = sizeof( ucTmpBuf);
	if (RC_OK( rc = flmUTF8ToStorage( 
		pszUTF8, uiBytesInBuffer, ucTmpBuf, &uiLen)))
	{
		if (RC_BAD( rc = storeValue( uiElementNumber,
								SFLM_STRING_TYPE, ucTmpBuf, uiLen)))
		{
			goto Exit;
		}
	}
	else if (rc != NE_SFLM_CONV_DEST_OVERFLOW)
	{
		goto Exit;
	}
	else
	{
		// Determine the length needed.

		if (RC_BAD( rc = flmUTF8ToStorage( 
			pszUTF8, uiBytesInBuffer, NULL, &uiLen)))
		{
			goto Exit;
		}

		// Allocate space for it in the vector and get a pointer
		// back so we can then store it.

		if (RC_BAD( rc = storeValue( uiElementNumber,
								SFLM_STRING_TYPE, NULL, uiLen, &pucDataPtr)))
		{
			goto Exit;
		}

		// Store it out to the space we just allocated.

		if (RC_BAD( rc = flmUTF8ToStorage( 
			pszUTF8, uiBytesInBuffer, pucDataPtr, &uiLen)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Get a pointer to the UTF8 - no conversions are done.
****************************************************************************/
RCODE F_DataVector::getUTF8Ptr(
	FLMUINT				uiElementNumber,
	const FLMBYTE **	ppszUTF8,
	FLMUINT *			puiBufLen)
{
	RCODE						rc = NE_SFLM_OK;
	F_VECTOR_ELEMENT *	pVector = getVector( uiElementNumber,
												VECT_SLOT_HAS_DATA);
	void *					pvValue;
	FLMUINT					uiStorageLen;
	FLMUINT					uiSenLen;

	if (!pVector)
	{
		*ppszUTF8 = NULL;
		if (puiBufLen)
		{
			*puiBufLen = 0;
		}
		goto Exit;
	}
	if (pVector->eDataTyp != SFLM_STRING_TYPE)
	{
		rc = RC_SET( NE_SFLM_BAD_DATA_TYPE);
		goto Exit;
	}

	if ((pvValue = getDataPtr( pVector)) != NULL)
	{
		*ppszUTF8 = (FLMBYTE *)pvValue;
		uiStorageLen = pVector->uiDataLength;
		if( RC_BAD( rc = flmGetCharCountFromStorageBuf( ppszUTF8,
			uiStorageLen, NULL, &uiSenLen)))
		{
			goto Exit;
		}

		flmAssert( uiStorageLen > uiSenLen);
		uiStorageLen -= uiSenLen;
	}
	else
	{
		*ppszUTF8 = NULL;
		uiStorageLen = 0;
	}

	if (puiBufLen)
	{
		*puiBufLen = uiStorageLen;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Allocate data for a unicode element and retrieve it.
****************************************************************************/
RCODE F_DataVector::getUnicode(
	FLMUINT			uiElementNumber,
	FLMUNICODE **	ppuzUnicode)
{
	RCODE		rc = NE_SFLM_OK;
	FLMUINT	uiLen;

	// Get the unicode length (does not include NULL terminator)

	if (RC_BAD( rc = getUnicode( uiElementNumber, NULL, &uiLen)))
	{
		goto Exit;
	}

	if (uiLen)
	{

		// Account for NULL character.

		uiLen += sizeof( FLMUNICODE);

		if( RC_BAD( rc = f_alloc( uiLen, ppuzUnicode)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = getUnicode( uiElementNumber, *ppuzUnicode,
									&uiLen)))
		{
			goto Exit;
		}
	}
	else
	{
		*ppuzUnicode = NULL;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Compose a key buffer from the vector's components.
****************************************************************************/
RCODE F_DataVector::outputKey(
	F_Db *			pDb,
	FLMUINT			uiIndexNum,
	FLMUINT			uiMatchFlags,
	FLMBYTE *		pucKeyBuf,
	FLMUINT			uiKeyBufSize,
	FLMUINT *		puiKeyLen,
	FLMUINT			uiSearchKeyFlag)
{
	RCODE						rc = NE_SFLM_OK;
	ICD *						pIcd;
	F_Dict *					pDict = pDb->getDict();
	F_INDEX *				pIndex = pDict->getIndex( uiIndexNum);
	F_TABLE *				pTable = pDict->getTable( pIndex->uiTableNum);
	F_COLUMN *				pColumn;
	FLMBYTE *				pucToKey;
	FLMBYTE *				pucKeyLenPos;
	FLMUINT					uiToKeyLen;
	FLMUINT					uiKeyLen;
	FLMUINT					uiKeyComponent;
	FLMBOOL					bDataTruncated;
	eDataType				eDataTyp;
	FLMUINT					uiLanguage;
	F_VECTOR_ELEMENT *	pVector = NULL;
	FLMBYTE					ucIDBuf [10];
	FLMUINT					uiIDLen = 0;
	FLMUINT					uiMaxKeySize;
	FLMUINT					uiDataLen;
	const FLMBYTE *		pucDataPtr;
	FLMUINT					uiIDMatchFlags = uiMatchFlags & FLM_MATCH_ROW_ID;
	F_BufferIStream		bufferIStream;

	if (uiIDMatchFlags)
	{
		pucToKey = &ucIDBuf [0];
		uiIDLen = f_encodeSEN( m_ui64RowId, &pucToKey);
		if (uiIDLen >= uiKeyBufSize)
		{
			rc = RC_SET( NE_SFLM_CONV_DEST_OVERFLOW);
			goto Exit;
		}
	}

	// Output the key components

	uiMaxKeySize = uiKeyBufSize - uiIDLen;
	uiLanguage = pIndex->uiLanguage;
	uiKeyLen = 0;
	pucToKey = pucKeyBuf;
	for (uiKeyComponent = 0, pIcd = pIndex->pKeyIcds;
		  uiKeyComponent < pIndex->uiNumKeyComponents;
		  uiKeyComponent++, pIcd++)
	{
		pucKeyLenPos = pucToKey;
		pucToKey += 2;
		uiKeyLen += 2;
		pColumn = pDict->getColumn( pTable, pIcd->uiColumnNum);
		
		eDataTyp = pColumn->eDataTyp;

		// Find matching node in the tree - if not found skip and continue.

		if ((pVector = getVector( uiKeyComponent, VECT_SLOT_HAS_DATA)) == NULL)
		{
			UW2FBA( 0, pucKeyLenPos);
		}
		else
		{
			uiToKeyLen = 0;
			bDataTruncated = FALSE;

			// Take the dictionary number and make it the key

			if (pIcd->uiFlags & ICD_PRESENCE)
			{
        		// Output the column number

				if (uiKeyLen + 4 > uiMaxKeySize)
				{
					rc = RC_SET( NE_SFLM_CONV_DEST_OVERFLOW);
					goto Exit;
				}
				
				f_UINT32ToBigEndian( (FLMUINT32)pIcd->uiColumnNum, pucToKey);
				uiToKeyLen = 4;
			}
			else if (pIcd->uiFlags & ICD_METAPHONE)
			{
				FLMUINT	uiMeta;
				FLMBYTE	ucStorageBuf[ FLM_MAX_NUM_BUF_SIZE];
				FLMUINT	uiStorageLen;

				if (eDataTyp != SFLM_STRING_TYPE)
				{
					rc = RC_SET_AND_ASSERT( NE_SFLM_BAD_DATA_TYPE);
					goto Exit;
				}
				
				if (pVector->eDataTyp == SFLM_STRING_TYPE)
				{
					if (RC_BAD( rc = getUTF8Ptr( uiKeyComponent,
											&pucDataPtr, &uiDataLen)))
					{
						goto Exit;
					}
					
					if (RC_BAD( rc = bufferIStream.openStream( 
						(const char *)pucDataPtr, uiDataLen)))
					{
						goto Exit;
					}
	
					if (RC_BAD( rc = f_getNextMetaphone( &bufferIStream, &uiMeta)))
					{
						if( rc == NE_SFLM_EOF_HIT)
						{
							rc = RC_SET( NE_SFLM_BAD_DATA_TYPE);
						}
						goto Exit;
					}
					
					bufferIStream.closeStream();					
				}
				else if (pVector->eDataTyp == SFLM_NUMBER_TYPE)
				{
					if( RC_BAD( rc = getUINT( uiKeyComponent, &uiMeta)))
					{
						goto Exit;
					}
				}
				else
				{
					rc = RC_SET_AND_ASSERT( NE_SFLM_BAD_DATA_TYPE);
					goto Exit;
				}

				if (uiMeta)
				{
					uiStorageLen = FLM_MAX_NUM_BUF_SIZE;
					if( RC_BAD( rc = flmNumber64ToStorage( uiMeta,
						&uiStorageLen, ucStorageBuf, FALSE, FALSE)))
					{
						goto Exit;
					}

					if (RC_BAD( rc = bufferIStream.openStream( 
						(const char *)ucStorageBuf, uiStorageLen)))
					{
						goto Exit;
					}

        			// Output the metaphone key piece

					uiToKeyLen = uiMaxKeySize - uiKeyLen;
					if( RC_BAD( rc = KYCollateValue( pucToKey, &uiToKeyLen,
						&bufferIStream, SFLM_NUMBER_TYPE,
						pIcd->uiFlags, pIcd->uiCompareRules, pIcd->uiLimit,
						NULL, NULL, uiLanguage,
						FALSE, FALSE, &bDataTruncated, NULL)))
					{
						goto Exit;
					}
					
					bufferIStream.closeStream();					
				}
			}
			else
			{
				if (eDataTyp == SFLM_STRING_TYPE)
				{
					if (RC_BAD( rc = getUTF8Ptr( uiKeyComponent,
											&pucDataPtr, &uiDataLen)))
					{
						goto Exit;
					}
				}
				else
				{
					pucDataPtr = (FLMBYTE *)getDataPtr( pVector);
					uiDataLen = pVector->uiDataLength;
				}
				
				if (uiDataLen)
				{
					
					if (RC_BAD( rc = bufferIStream.openStream( 
						(const char *)pucDataPtr, uiDataLen)))
					{
						goto Exit;
					}

					uiToKeyLen = uiMaxKeySize - uiKeyLen;
					if( RC_BAD( rc = KYCollateValue( pucToKey, &uiToKeyLen,
						&bufferIStream, eDataTyp,
						pIcd->uiFlags, pIcd->uiCompareRules, pIcd->uiLimit,
						NULL, NULL, uiLanguage,
						(FLMBOOL) ((pIcd->uiFlags & ICD_SUBSTRING)
								? (isLeftTruncated( pVector)
									? FALSE : TRUE)
								: FALSE),
						isRightTruncated( pVector),
						&bDataTruncated, NULL)))
					{
						goto Exit;
					}
					
					bufferIStream.closeStream();					
				}
			}

			if (uiToKeyLen)
			{

				//	Increment total key length

				pucToKey += uiToKeyLen;
				uiKeyLen += uiToKeyLen;
			}
			if (!bDataTruncated)
			{
				UW2FBA( (FLMUINT16)(uiToKeyLen | uiSearchKeyFlag),
									pucKeyLenPos);
			}
			else
			{
				UW2FBA( (FLMUINT16)(uiToKeyLen | TRUNCATED_FLAG |
									uiSearchKeyFlag), pucKeyLenPos); 
			}
		}
	}
	
	// Output the row ID, if requested.

	if (uiIDMatchFlags)
	{

		// There will always be room at this point for the
		// row ID - because it was subtracted out above.

		f_memcpy( pucToKey, ucIDBuf, uiIDLen);
	}
	*puiKeyLen = uiKeyLen + uiIDLen;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Populate a vector's components from the key part of an index key.
****************************************************************************/
RCODE F_DataVector::inputKey(
	F_Db *				pDb,
	FLMUINT				uiIndexNum,
	const FLMBYTE *	pucKey,
	FLMUINT				uiKeyLen)
{
	RCODE					rc = NE_SFLM_OK;
	F_Dict *				pDict = pDb->getDict();
	F_INDEX *			pIndex = pDict->getIndex( uiIndexNum);
	F_TABLE *			pTable = pDict->getTable( pIndex->uiTableNum);
	F_COLUMN *			pColumn;
	const FLMBYTE *	pucKeyEnd = pucKey + uiKeyLen;
	FLMBYTE				ucDataBuf [SFLM_MAX_KEY_SIZE];
	FLMUINT				uiDataLen;
	ICD *					pIcd;
	FLMUINT				uiLanguage = pIndex->uiLanguage;
	FLMUINT				uiComponentLen;
	eDataType			eDataTyp;
	FLMBOOL				bDataRightTruncated;
	FLMBOOL				bFirstSubstring;
	FLMBOOL				bIsText;
	FLMUINT				uiComponent;
	FLMUINT				uiColumnNum;

	flmAssert( uiKeyLen);

	// Loop for each compound piece of key

	for (uiComponent = 0, pIcd = pIndex->pKeyIcds;
		  uiComponent < pIndex->uiNumKeyComponents;
		  pIcd++, uiComponent++)
	{
		if (uiKeyLen < 2)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
			goto Exit;
		}
		uiColumnNum = pIcd->uiColumnNum;
		pColumn = pDict->getColumn( pTable, uiColumnNum);
		
		uiComponentLen = getKeyComponentLength( pucKey);
		bDataRightTruncated = isKeyComponentTruncated( pucKey);
		uiKeyLen -= 2;
		pucKey += 2;
		
		if (uiComponentLen > uiKeyLen)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
			goto Exit;
		}
		
		bFirstSubstring = FALSE;
		eDataTyp = pColumn->eDataTyp;
		bIsText = (eDataTyp == SFLM_STRING_TYPE &&
					  !(pIcd->uiFlags & (ICD_PRESENCE | ICD_METAPHONE)))
					 ? TRUE
					 : FALSE;

		if (uiComponentLen)
		{
			if (pIcd->uiFlags & ICD_PRESENCE)
			{
				FLMUINT	uiNum;
				
				if (uiComponentLen != 4 || bDataRightTruncated)
				{
					rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
					goto Exit;
				}
				uiNum = (FLMUINT)f_bigEndianToUINT32( pucKey);
	
				// What is stored in the key better match the column
				// number of the ICD.
	
				if (uiNum != uiColumnNum)
				{
					rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
					goto Exit;
				}
				if (RC_BAD( rc = setUINT( uiComponent, uiNum)))
				{
					goto Exit;
				}
			}
			else if (pIcd->uiFlags & ICD_METAPHONE)
			{
				uiDataLen = sizeof( ucDataBuf);
	
				if (uiComponentLen)
				{
					if( RC_BAD( rc = flmCollationNum2StorageNum( pucKey,
						uiComponentLen, ucDataBuf, &uiDataLen)))
					{
						goto Exit;
					}
				}
				else
				{
					uiDataLen = 0;
				}
	
				if (bDataRightTruncated)
				{
					rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
					goto Exit;
				}
	
				// Allocate and copy value into the component.  NOTE:
				// storeValue handles zero length data.
	
				if (RC_BAD( rc = storeValue( uiComponent, SFLM_NUMBER_TYPE, 
														ucDataBuf, uiDataLen)))
				{
					goto Exit;
				}
			}
			else
			{
	
				// Grab only the Nth section of key if compound key
	
				switch (eDataTyp)
				{
					case SFLM_STRING_TYPE:
					{
						FLMBOOL bTmpTruncated = FALSE;
	
						if (uiComponentLen)
						{
							uiDataLen = sizeof( ucDataBuf);
							if (RC_BAD( rc = flmColText2StorageText( pucKey,
								uiComponentLen,
								ucDataBuf, &uiDataLen, uiLanguage,
								&bTmpTruncated, &bFirstSubstring)))
							{
								goto Exit;
							}
							
							if (bTmpTruncated != bDataRightTruncated)
							{
								rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
								goto Exit;
							}
						}
						else
						{
							uiDataLen = 0;
						}
						break;
					}
	
					case SFLM_NUMBER_TYPE:
					{
						if (uiComponentLen)
						{
							uiDataLen = sizeof( ucDataBuf);
							if( RC_BAD( rc = flmCollationNum2StorageNum( pucKey,
								uiComponentLen, ucDataBuf, &uiDataLen)))
							{
								goto Exit;
							}
						}
						else
						{
							uiDataLen = 0;
						}
						
						if (bDataRightTruncated)
						{
							rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
							goto Exit;
						}
						break;
					}
	
					case SFLM_BINARY_TYPE:
					{
						uiDataLen = uiComponentLen;
						if (uiComponentLen > sizeof( ucDataBuf))
						{
							rc = RC_SET( NE_SFLM_CONV_DEST_OVERFLOW);
							goto Exit;
						}
						if (uiComponentLen)
						{
							f_memcpy( ucDataBuf, pucKey, uiComponentLen);
						}
						break;
					}
	
					default:
						rc = RC_SET( NE_SFLM_DATA_ERROR);
						goto Exit;
				}
	
				// Allocate and copy value into the component.  NOTE:
				// storeValue handles zero length data.
	
				if (RC_BAD( rc = storeValue( uiComponent, eDataTyp, ucDataBuf,
											uiDataLen)))
				{
					goto Exit;
				}
	
				// Set first sub-string and truncated flags.
	
				if ((pIcd->uiFlags & ICD_SUBSTRING) && !bFirstSubstring)
				{
					setLeftTruncated( uiComponent);
				}
				if (bDataRightTruncated)
				{
					setRightTruncated( uiComponent);
				}
			}
		}

		// Store the column number

		if (RC_BAD( rc = setColumnNum( uiComponent, uiColumnNum, FALSE)))
		{
			goto Exit;
		}

		// Position to the end of this component

		flmAssert( uiKeyLen >= uiComponentLen);
		pucKey += uiComponentLen;
		uiKeyLen -= uiComponentLen;
	}
	
	// See if we have a row ID.
	
	if (RC_BAD( rc = f_decodeSEN64( &pucKey, pucKeyEnd, &m_ui64RowId)))
	{
		goto Exit;
	}

	// Store the column numbers for the data components, if any.

	for (uiComponent = 0, pIcd = pIndex->pDataIcds;
		  uiComponent < pIndex->uiNumDataComponents;
		  pIcd++, uiComponent++)
	{

		// Store the column number

		if (RC_BAD( rc = setColumnNum( uiComponent + pIndex->uiNumKeyComponents,
												pIcd->uiColumnNum, TRUE)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Compose a data buffer from the vector's components.
****************************************************************************/
RCODE F_DataVector::outputData(
	F_Db *		pDb,
	FLMUINT		uiIndexNum,
	FLMBYTE *	pucDataBuf,
	FLMUINT		uiDataBufSize,
	FLMUINT *	puiDataLen)
{
	RCODE						rc = NE_SFLM_OK;
	F_Dict *					pDict = pDb->getDict();
	F_INDEX *				pIndex = pDict->getIndex( uiIndexNum);
	F_TABLE *				pTable = pDict->getTable( pIndex->uiTableNum);
	F_COLUMN *				pColumn;
	ICD *						pIcd;
	FLMUINT					uiDataComponent;
	F_VECTOR_ELEMENT *	pVector;
	FLMBYTE *				pucData;
	FLMUINT					uiDataLength;
	FLMUINT					uiTotalLength = 0;
	FLMBYTE					ucTmpSen [32];
	FLMBYTE *				pucTmpSen = &ucTmpSen [0];
	FLMUINT					uiSENLen;
	FLMUINT					uiLastDataLen = 0;

	// Loop for each data component of key

	for (uiDataComponent = 0, pIcd = pIndex->pDataIcds;
		  uiDataComponent < pIndex->uiNumDataComponents;
		  pIcd++, uiDataComponent++)
	{
		pColumn = pDict->getColumn( pTable, pIcd->uiColumnNum);
		if ((pVector = getVector( uiDataComponent + pIndex->uiNumKeyComponents,
											VECT_SLOT_HAS_DATA)) != NULL)
		{

			// Cannot do data conversions right now.

			flmAssert( pVector->eDataTyp == pColumn->eDataTyp);
			uiDataLength = pVector->uiDataLength;
			pucData = (FLMBYTE *)getDataPtr( pVector);
		}
		else
		{
			uiDataLength = 0;
			pucData = NULL;
		}

		// Output the length of the data as a SEN value

		uiSENLen = f_encodeSEN( uiDataLength, &pucTmpSen);
		if (uiTotalLength + uiSENLen > uiDataBufSize)
		{
			rc = RC_SET( NE_SFLM_CONV_DEST_OVERFLOW);
			goto Exit;
		}
		f_memcpy( pucDataBuf, ucTmpSen, uiSENLen);
		pucDataBuf += uiSENLen;
		uiTotalLength += uiSENLen;

		// Output the data

		if (uiDataLength)
		{
			if (uiTotalLength + uiDataLength > uiDataBufSize)
			{
				rc = RC_SET( NE_SFLM_CONV_DEST_OVERFLOW);
				goto Exit;
			}
			f_memcpy( pucDataBuf, pucData, uiDataLength);
			pucDataBuf += uiDataLength;
			uiTotalLength += uiDataLength;
			uiLastDataLen = uiTotalLength;
		}
	}

Exit:

	// Even if rc == NE_SFLM_CONV_DEST_OVERFLOW, return a length

	*puiDataLen = uiLastDataLen;

	return( rc);
}

/****************************************************************************
Desc:	Populate a vector's data components from the data part of a key.
****************************************************************************/
RCODE F_DataVector::inputData(
	F_Db *				pDb,
	FLMUINT				uiIndexNum,
	const FLMBYTE *	pucData,
	FLMUINT				uiInputLen)
{
	RCODE			rc = NE_SFLM_OK;
	F_Dict *		pDict = pDb->getDict();
	F_INDEX *	pIndex = pDict->getIndex( uiIndexNum);
	F_TABLE *	pTable = pDict->getTable( pIndex->uiTableNum);
	F_COLUMN *	pColumn;
	ICD *			pIcd;
	FLMUINT		uiDataComponent = 0;
	FLMUINT		uiDataLength;
	FLMUINT		uiSENLen;

	// Loop for each data component of key

	for (uiDataComponent = 0, pIcd = pIndex->pDataIcds;
		  uiDataComponent < pIndex->uiNumDataComponents;
		  pIcd++, uiDataComponent++)
	{
		pColumn = pDict->getColumn( pTable, pIcd->uiColumnNum);
		if (RC_BAD( rc = setColumnNum( uiDataComponent + pIndex->uiNumKeyComponents,
									pIcd->uiColumnNum,  TRUE)))
		{
			goto Exit;
		}
		if (!uiInputLen)
		{
			continue;
		}

		// Get the data length - it is stored as a SEN

		uiSENLen = f_getSENLength( *pucData);
		if (uiSENLen > uiInputLen)
		{
			rc = RC_SET( NE_SFLM_DATA_ERROR);
			goto Exit;
		}

		if( RC_BAD( rc = f_decodeSEN( &pucData,
			&pucData[ uiSENLen], &uiDataLength)))
		{
			goto Exit;
		}

		uiInputLen -= uiSENLen;
		if (uiDataLength > uiInputLen)
		{
			rc = RC_SET( NE_SFLM_DATA_ERROR);
			goto Exit;
		}

		// Store the data into the vector.

		if (RC_BAD( rc = storeValue( uiDataComponent + pIndex->uiNumKeyComponents,
									pColumn->eDataTyp,
									pucData, uiDataLength, NULL)))
		{
			goto Exit;
		}
		pucData += uiDataLength;
		uiInputLen -= uiDataLength;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Copy one data vector into another.
****************************************************************************/
RCODE F_DataVector::copyVector(
	F_DataVector *	pSrcVector)
{
	RCODE	rc = NE_SFLM_OK;
	
	m_ui64RowId = pSrcVector->m_ui64RowId;
	
	// If none of the elements are populated, there is nothing that
	// needs to be done.
	
	if ((m_uiNumElements = pSrcVector->m_uiNumElements) > 0)
	{
		if (RC_BAD( rc = allocVectorArray( pSrcVector->m_uiNumElements - 1)))
		{
			goto Exit;
		}
		f_memcpy( m_pVectorElements, pSrcVector->m_pVectorElements,
				sizeof( F_VECTOR_ELEMENT) * m_uiNumElements);
				
		// Make sure we have enough room in our data buffer.
		
		if (pSrcVector->m_uiDataBufOffset > m_uiDataBufLength)
		{
			if (m_pucDataBuf == &m_ucIntDataBuf [0])
			{
				if (RC_BAD( rc = f_alloc( pSrcVector->m_uiDataBufOffset,
								&m_pucDataBuf)))
				{
					goto Exit;
				}

			}
			else
			{
				if (RC_BAD( rc = f_realloc( pSrcVector->m_uiDataBufOffset,
								&m_pucDataBuf)))
				{
					goto Exit;
				}
			}
			m_uiDataBufLength = pSrcVector->m_uiDataBufOffset;
		}
		
		// Copy the source data buffer into ours.
		
		m_uiDataBufOffset = pSrcVector->m_uiDataBufOffset;
		f_memcpy( m_pucDataBuf, pSrcVector->m_pucDataBuf,
					 pSrcVector->m_uiDataBufOffset);
	}
	else
	{
		m_uiDataBufOffset = 0;
	}
	
Exit:

	return( rc);
}

