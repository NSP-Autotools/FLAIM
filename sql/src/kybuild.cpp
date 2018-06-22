//------------------------------------------------------------------------------
// Desc:	This file contains the main routines for building of index keys,
//			and adding them to the database.
// Tabs:	3
//
// Copyright (c) 1990-1992, 1994-2007 Novell, Inc. All Rights Reserved.
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

#define STACK_DATA_BUF_SIZE		64

// Local function prototypes

FSTATIC RCODE kyAddRowIdToKey(
	FLMUINT64		ui64RowId,
	FLMBYTE *		pucKeyBuf,
	FLMUINT			uiIDBufSize,
	FLMUINT *		puiIDLen);

FSTATIC RCODE getColumnIStream(
	F_Db *				pDb,
	F_Row *				pRow,
	F_COLUMN_VALUE *	pColumnValues,
	FLMUINT				uiColumnNum,
	FLMBOOL *			pbIsNull,
	F_BufferIStream *	pBufferIStream,
	eDataType *			peDataType,
	FLMUINT *			puiDataLength);
	
FSTATIC RCODE getColumnTextIStream(
	F_Db *				pDb,
	F_Row *				pRow,
	F_COLUMN_VALUE *	pColumnValues,
	FLMUINT				uiColumnNum,
	FLMBOOL *			pbIsNull,
	F_BufferIStream *	pBufferIStream,
	FLMUINT *			puiNumChars);
	
/****************************************************************************
Desc:	Append row ID to the key buffer.
****************************************************************************/
FSTATIC RCODE kyAddRowIdToKey(
	FLMUINT64		ui64RowId,
	FLMBYTE *		pucKeyBuf,
	FLMUINT			uiIDBufSize,
	FLMUINT *		puiIDLen)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBYTE		ucTmpSen [FLM_MAX_NUM_BUF_SIZE];
	FLMBYTE *	pucTmpSen;

	// Put row ID into buffer.  If there is room for at least nine
	// bytes, we can encode the ID right into the buffer safely.  Otherwise,
	// we have to use a temporary buffer and see if there is room.
	
	if (uiIDBufSize >= 9)
	{
		*puiIDLen = f_encodeSEN( ui64RowId, &pucKeyBuf);
	}
	else
	{
		pucTmpSen = &ucTmpSen [0];
		*puiIDLen = f_encodeSEN( ui64RowId, &pucTmpSen);
		if (*puiIDLen > uiIDBufSize)
		{
			rc = RC_SET( NE_SFLM_KEY_OVERFLOW);
			goto Exit;
		}
		f_memcpy( pucKeyBuf, ucTmpSen, *puiIDLen);
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Add an index key to the buffers
****************************************************************************/
RCODE F_Db::addToKrefTbl(
	FLMUINT	uiKeyLen,
	FLMUINT	uiDataLen)
{
	RCODE				rc = NE_SFLM_OK;
	KREF_ENTRY *	pKref;
	FLMUINT			uiSizeNeeded;
	FLMBYTE *		pucDest;

	// If the table is FULL, expand the table

	if (m_uiKrefCount == m_uiKrefTblSize)
	{
		FLMUINT		uiAllocSize;
		FLMUINT		uiOrigKrefTblSize = m_uiKrefTblSize;

		if (m_uiKrefTblSize > 0x8000 / sizeof( KREF_ENTRY *))
		{
			m_uiKrefTblSize += 4096;
		}
		else
		{
			m_uiKrefTblSize *= 2;
		}

		uiAllocSize = m_uiKrefTblSize * sizeof( KREF_ENTRY *);

		rc = f_realloc( uiAllocSize, &m_pKrefTbl);
		if (RC_BAD(rc))
		{
			m_uiKrefTblSize = uiOrigKrefTblSize;
			rc = RC_SET( NE_SFLM_MEM);
			goto Exit;
		}
	}

	// Allocate memory for the key's KREF and the key itself.
	// We allocate one extra byte so we can zero terminate the key
	// below.  The extra zero character is to ensure that the compare
	// in the qsort routine will work.

	uiSizeNeeded = sizeof( KREF_ENTRY) + uiKeyLen + 1 + uiDataLen;

	if (RC_BAD( rc = m_pKrefPool->poolAlloc( uiSizeNeeded,
										(void **)&pKref)))
	{
		goto Exit;
	}
	
	m_pKrefTbl [ m_uiKrefCount++] = pKref;
	m_uiTotalKrefBytes += uiSizeNeeded;

	// Fill in all of the fields in the KREF structure.

	pKref->ui16IxNum = (FLMUINT16)m_keyGenInfo.pIndex->uiIndexNum;
	pKref->bDelete = m_keyGenInfo.bAddKeys ? FALSE : TRUE;
	pKref->ui16KeyLen = (FLMUINT16)uiKeyLen;
	pKref->uiSequence = m_uiKrefCount;
	pKref->uiDataLen = uiDataLen;
	pKref->pRow = m_keyGenInfo.pRow;

	// Copy the key to just after the KREF structure.

	pucDest = (FLMBYTE *)(&pKref [1]);
 	f_memcpy( pucDest, m_keyGenInfo.pucKeyBuf, uiKeyLen);

	// Null terminate the key so compare in qsort will work.

	pucDest [uiKeyLen] = 0;
	if (uiDataLen)
	{
		f_memcpy( pucDest + uiKeyLen + 1, m_keyGenInfo.pucData, uiDataLen);
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Build the data part for a key.
Notes:	This routine is recursive in nature.  Will recurse the number of
			data components defined in the index.
****************************************************************************/
RCODE F_Db::buildData(
	ICD *			pIcd,
	FLMUINT		uiDataComponent,
	FLMUINT		uiKeyLen,
	FLMUINT		uiDataLen)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiDataComponentLen;
	FLMBYTE				ucTmpSen [FLM_MAX_NUM_BUF_SIZE];
	FLMBYTE *			pucTmpSen;
	FLMUINT				uiSENLen;
	FLMUINT				uiIDLen;
	F_COLUMN_ITEM *	pColumnItem;

	if ((pColumnItem = m_keyGenInfo.pRow->getColumn( pIcd->uiColumnNum)) != NULL)
	{
		uiDataComponentLen = pColumnItem->uiDataLen;
	}
	else
	{
		uiDataComponentLen = 0;
	}

	// Output the length of the data as a SEN value

	pucTmpSen = &ucTmpSen [0];
	uiSENLen = f_encodeSEN( uiDataComponentLen, &pucTmpSen);
	if (uiDataComponentLen + uiSENLen + uiDataLen > m_keyGenInfo.uiDataBufSize)
	{
		FLMUINT	uiNewSize = uiDataComponentLen + uiSENLen + uiDataLen + 512;

		// Allocate the data buffer if it has not been allocated.  Otherwise,
		// realloc it.

		if (!m_keyGenInfo.bDataBufAllocated)
		{
			FLMBYTE *	pucNewData;

			if (RC_BAD( rc = f_alloc( uiNewSize, &pucNewData)))
			{
				goto Exit;
			}

			if( uiDataLen)
			{
				f_memcpy( pucNewData, m_keyGenInfo.pucData, uiDataLen);
			}
			m_keyGenInfo.pucData = pucNewData;
			m_keyGenInfo.bDataBufAllocated = TRUE;
		}
		else
		{
			
			// Reallocate the buffer.

			if (RC_BAD( rc = f_realloc( uiNewSize, &m_keyGenInfo.pucData)))
			{
				goto Exit;
			}
		}

		m_keyGenInfo.uiDataBufSize = uiNewSize;
	}
	f_memcpy( m_keyGenInfo.pucData + uiDataLen, ucTmpSen, uiSENLen);
	if (uiDataComponentLen)
	{
		f_memcpy( m_keyGenInfo.pucData + uiDataLen + uiSENLen,
					 m_keyGenInfo.pRow->getColumnDataPtr( pIcd->uiColumnNum),
					 uiDataComponentLen);
	}

	// If this is the last data CDL, append IDs to the
	// key and output the key and data to the KREF.
	// Otherwise, recurse down.

	if (uiDataComponent < m_keyGenInfo.pIndex->uiNumDataComponents)
	{
		if (RC_BAD( rc = buildData( pIcd + 1, uiDataComponent + 1, uiKeyLen,
									uiDataLen + uiDataComponentLen + uiSENLen)))
		{
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = kyAddRowIdToKey( m_keyGenInfo.pRow->m_ui64RowId,
									&m_keyGenInfo.pucKeyBuf [uiKeyLen],
									SFLM_MAX_KEY_SIZE - uiKeyLen, &uiIDLen)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = addToKrefTbl( uiKeyLen + uiIDLen,
								uiDataLen + uiDataComponentLen + uiSENLen)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Finish the current key component.  If there is a next one call
		build keys.  Otherwise, go on to doing data and context pieces.
****************************************************************************/
RCODE F_Db::finishKeyComponent(
	ICD *		pIcd,
	FLMUINT	uiKeyComponent,
	FLMUINT	uiKeyLen)
{
	RCODE		rc = NE_SFLM_OK;
	FLMUINT	uiIDLen;
	
	if (uiKeyComponent < m_keyGenInfo.pIndex->uiNumKeyComponents)
	{
		flmAssert( m_keyGenInfo.bIsCompound);
		if (RC_BAD( rc = buildKeys( pIcd + 1, uiKeyComponent + 1, uiKeyLen)))
		{
			goto Exit;
		}
	}
	else
	{
		if (m_keyGenInfo.pIndex->pDataIcds)
		{
			if (RC_BAD( rc = buildData( m_keyGenInfo.pIndex->pDataIcds, 1, uiKeyLen, 0)))
			{
				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = kyAddRowIdToKey( m_keyGenInfo.pRow->m_ui64RowId,
										&m_keyGenInfo.pucKeyBuf [uiKeyLen],
										SFLM_MAX_KEY_SIZE - uiKeyLen, &uiIDLen)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = addToKrefTbl( uiKeyLen + uiIDLen, 0)))
			{
				goto Exit;
			}
		}
	}
	
Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
FSTATIC RCODE getColumnIStream(
	F_Db *				pDb,
	F_Row *				pRow,
	F_COLUMN_VALUE *	pColumnValues,
	FLMUINT				uiColumnNum,
	FLMBOOL *			pbIsNull,
	F_BufferIStream *	pBufferIStream,
	eDataType *			peDataType,
	FLMUINT *			puiDataLength)
{
	RCODE					rc = NE_SFLM_OK;
	F_TABLE *			pTable;
	F_COLUMN *			pColumn;
	F_COLUMN_VALUE *	pColumnValue;
	
	// See if there is a column data item
	
	pColumnValue = pColumnValues;
	while (pColumnValue && pColumnValue->uiColumnNum != uiColumnNum)
	{
		pColumnValue = pColumnValue->pNext;
	}
	
	if (!pColumnValue)
	{
		rc = pRow->getIStream( pDb, uiColumnNum, pbIsNull, pBufferIStream,
										peDataType, puiDataLength);
		goto Exit;
	}
	
	*pbIsNull = FALSE;
	if (!pColumnValue->uiValueLen)
	{
		*pbIsNull = TRUE;
		goto Exit;
	}
	if (puiDataLength)
	{
		*puiDataLength = pColumnValue->uiValueLen;
	}
	if (peDataType)
	{
		pTable = pDb->getDict()->getTable( pRow->getTableNum());
		pColumn = pDb->getDict()->getColumn( pTable, uiColumnNum);
		*peDataType = pColumn->eDataTyp;
	}
	if (RC_BAD( rc = pBufferIStream->openStream( 
		(const char *)pColumnValue->pucColumnValue,
		pColumnValue->uiValueLen, NULL)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}
	
/*****************************************************************************
Desc:
******************************************************************************/
FSTATIC RCODE getColumnTextIStream(
	F_Db *				pDb,
	F_Row *				pRow,
	F_COLUMN_VALUE *	pColumnValues,
	FLMUINT				uiColumnNum,
	FLMBOOL *			pbIsNull,
	F_BufferIStream *	pBufferIStream,
	FLMUINT *			puiNumChars)
{
	RCODE			rc = NE_SFLM_OK;
	eDataType	eDataTyp;
	FLMUINT		uiDataLength;

	*puiNumChars = 0;

	if( RC_BAD( rc = getColumnIStream( pDb, pRow, pColumnValues,
								uiColumnNum, pbIsNull,
								pBufferIStream, &eDataTyp, &uiDataLength)))
	{
		goto Exit;
	}

	if (eDataTyp != SFLM_STRING_TYPE)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BAD_DATA_TYPE);
		goto Exit;
	}
	
	if (*pbIsNull)
	{
		goto Exit;
	}

	// Skip the leading SEN so that the stream is positioned to
	// read raw utf8.

	if (pBufferIStream->remainingSize())
	{
		if (RC_BAD( rc = f_readSEN( pBufferIStream, puiNumChars)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Generate the keys for a text component.
****************************************************************************/
RCODE F_Db::genTextKeyComponents(
	F_COLUMN *,	// pColumn,
	ICD *			pIcd,
	FLMUINT		uiKeyComponent,
	FLMUINT		uiKeyLen,
	FLMBYTE **	ppucTmpBuf,
	FLMUINT *	puiTmpBufSize,
	void **		ppvMark)
{
	RCODE						rc = NE_SFLM_OK;
	FLMUINT					uiNumChars;
	FLMUINT					uiStrBytes;
	FLMUINT					uiSubstrChars;
	FLMUINT					uiMeta;
	FLMBOOL					bEachWord = FALSE;
	FLMBOOL					bMetaphone = FALSE;
	FLMBOOL					bSubstring = FALSE;
	FLMBOOL					bWholeString = FALSE;
	FLMBOOL					bHadAtLeastOneString = FALSE;
	FLMBOOL					bDataTruncated;
	FLMUINT					uiSaveKeyLen;
	FLMUINT					uiElmLen;
	FLMUINT					uiKeyLenPos = uiKeyLen;
	FLMUINT					uiCompareRules = pIcd->uiCompareRules;
	F_BufferIStream		columnBufferIStream;
	F_BufferIStream		bufferIStream;
	FLMBOOL					bIsNull;
	
	uiKeyLen += 2;
	uiSaveKeyLen = uiKeyLen;

	if (RC_BAD( rc = getColumnTextIStream( this, m_keyGenInfo.pRow,
									m_keyGenInfo.pColumnValues,
									pIcd->uiColumnNum, &bIsNull,
									&columnBufferIStream, &uiNumChars)))
	{
		goto Exit;
	}
	
	if (bIsNull || !uiNumChars)
	{
No_Strings:

		// Save the key component length
		
		UW2FBA( 0, &m_keyGenInfo.pucKeyBuf [uiKeyLenPos]);

		rc = finishKeyComponent( pIcd, uiKeyComponent, uiKeyLen);
		goto Exit;
	}

	if (pIcd->uiFlags & ICD_EACHWORD)
	{
		bEachWord = TRUE;
		
		// OR in the compressing of spaces, because we only want to treat
		// spaces as delimiters between words.
	
		uiCompareRules |= FLM_COMP_COMPRESS_WHITESPACE; 
	}
	else if (pIcd->uiFlags & ICD_METAPHONE)
	{
		bMetaphone = TRUE;
	}
	else if (pIcd->uiFlags & ICD_SUBSTRING)
	{
		bSubstring = TRUE;
	}
	else
	{
		bWholeString = TRUE;
	}

	// Loop on each word or substring in the value

	for (;;)
	{
		uiKeyLen = uiSaveKeyLen;
		bDataTruncated = FALSE;

		if (bWholeString)
		{
			uiElmLen = SFLM_MAX_KEY_SIZE - uiKeyLen;
			if( RC_BAD( rc = KYCollateValue( &m_keyGenInfo.pucKeyBuf [uiKeyLen],
							&uiElmLen,
							&columnBufferIStream, SFLM_STRING_TYPE,
							pIcd->uiFlags, pIcd->uiCompareRules,
							pIcd->uiLimit, NULL, NULL,
							m_keyGenInfo.pIndex->uiLanguage,
							FALSE, FALSE,
							&bDataTruncated, NULL)))
			{
				goto Exit;
			}
		}
		else if (bEachWord)
		{
			if (*ppucTmpBuf == NULL)
			{
				*ppvMark = m_tempPool.poolMark();
				*puiTmpBufSize = (FLMUINT)SFLM_MAX_KEY_SIZE + 8;
				if (RC_BAD( rc = m_tempPool.poolAlloc( *puiTmpBufSize,
												(void **)ppucTmpBuf)))
				{
					goto Exit;
				}
			}
	
			uiStrBytes = *puiTmpBufSize;
			if( RC_BAD( rc = KYEachWordParse( &columnBufferIStream, &uiCompareRules,
				pIcd->uiLimit, *ppucTmpBuf, &uiStrBytes)))
			{
				goto Exit;
			}

			if (!uiStrBytes)
			{
				if (!bHadAtLeastOneString)
				{
					goto No_Strings;
				}
				break;
			}

			if (RC_BAD( rc = bufferIStream.openStream( 
				(const char *)*ppucTmpBuf, uiStrBytes)))
			{
				goto Exit;
			}
			
			// Pass 0 for compare rules because KYEachWordParse will already
			// have taken care of them - except for FLM_COMP_CASE_INSENSITIVE.

			uiElmLen = SFLM_MAX_KEY_SIZE - uiKeyLen;
			rc = KYCollateValue( &m_keyGenInfo.pucKeyBuf [uiKeyLen],
										&uiElmLen,
										&bufferIStream, SFLM_STRING_TYPE,
										pIcd->uiFlags,
										pIcd->uiCompareRules & FLM_COMP_CASE_INSENSITIVE,
										pIcd->uiLimit,
										NULL, NULL,
										m_keyGenInfo.pIndex->uiLanguage,
										FALSE, FALSE, &bDataTruncated, NULL);
			bufferIStream.closeStream();

			if( RC_BAD( rc))
			{
				RC_UNEXPECTED_ASSERT( rc);
				goto Exit;
			}
			bHadAtLeastOneString = TRUE;
		}
		else if (bMetaphone)
		{
			FLMBYTE	ucStorageBuf[ FLM_MAX_NUM_BUF_SIZE];
			FLMUINT	uiStorageLen;

			if( RC_BAD( rc = f_getNextMetaphone( &columnBufferIStream, &uiMeta)))
			{
				if( rc != NE_SFLM_EOF_HIT)
				{
					goto Exit;
				}
				rc = NE_SFLM_OK;
				if (!bHadAtLeastOneString)
				{
					goto No_Strings;
				}
				break;
			}

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

			// Pass 0 for compare rules - only applies to strings.
			
			uiElmLen = SFLM_MAX_KEY_SIZE - uiKeyLen;
			rc = KYCollateValue( &m_keyGenInfo.pucKeyBuf [uiKeyLen],
										&uiElmLen,
										&bufferIStream, SFLM_NUMBER_TYPE,
										pIcd->uiFlags, 0,
										pIcd->uiLimit,
										NULL, NULL,
										m_keyGenInfo.pIndex->uiLanguage,
										FALSE, FALSE, NULL, NULL);
			bufferIStream.closeStream();

			if( RC_BAD( rc))
			{
				RC_UNEXPECTED_ASSERT( rc);
				goto Exit;
			}
			bHadAtLeastOneString = TRUE;
		}
		else
		{
			flmAssert( bSubstring);
			if (*ppucTmpBuf == NULL)
			{
				*ppvMark = m_tempPool.poolMark();
				*puiTmpBufSize = (FLMUINT)SFLM_MAX_KEY_SIZE + 8;
				if (RC_BAD( rc = m_tempPool.poolAlloc( *puiTmpBufSize,
												(void **)ppucTmpBuf)))
				{
					goto Exit;
				}
			}
			uiStrBytes = *puiTmpBufSize;
			
			if( RC_BAD( rc = KYSubstringParse( &columnBufferIStream, &uiCompareRules,
				pIcd->uiLimit, *ppucTmpBuf, &uiStrBytes, &uiSubstrChars)))
			{
				goto Exit;
			}

			if (!uiStrBytes)
			{
				if (!bHadAtLeastOneString)
				{
					goto No_Strings;
				}
				break;
			}

			if (bHadAtLeastOneString && uiSubstrChars == 1 && !m_keyGenInfo.bIsAsia)
			{
				break;
			}

			if (RC_BAD( rc = bufferIStream.openStream( 
				(const char *)*ppucTmpBuf, uiStrBytes)))
			{
				goto Exit;
			}
			
			// Pass 0 for compare rules, because KYSubstringParse has already
			// taken care of them, except for FLM_COMP_CASE_INSENSITIVE

			uiElmLen = SFLM_MAX_KEY_SIZE - uiKeyLen;
			rc = KYCollateValue( &m_keyGenInfo.pucKeyBuf [uiKeyLen],
										&uiElmLen,
										&bufferIStream, SFLM_STRING_TYPE,
										pIcd->uiFlags,
										pIcd->uiCompareRules & FLM_COMP_CASE_INSENSITIVE,
										pIcd->uiLimit,
										NULL, NULL,
										m_keyGenInfo.pIndex->uiLanguage,
										bHadAtLeastOneString ? FALSE : TRUE, FALSE,
										&bDataTruncated, NULL);

			bufferIStream.closeStream();

			if( RC_BAD( rc))
			{
				RC_UNEXPECTED_ASSERT( rc);
				goto Exit;
			}
			bHadAtLeastOneString = TRUE;
		}

		uiKeyLen += uiElmLen;
		
		// Save the key component length
		
		if (!bDataTruncated)
		{
			UW2FBA( (FLMUINT16)(uiElmLen),
							&m_keyGenInfo.pucKeyBuf [uiKeyLenPos]);
		}
		else
		{
			UW2FBA( (FLMUINT16)(uiElmLen | TRUNCATED_FLAG),
								&m_keyGenInfo.pucKeyBuf [uiKeyLenPos]);
		}

		if (RC_BAD( rc = finishKeyComponent( pIcd, uiKeyComponent, uiKeyLen)))
		{
			goto Exit;
		}

		if (bWholeString)
		{
			break;
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Generate the keys for other data types besides text.
****************************************************************************/
RCODE F_Db::genOtherKeyComponent(
	F_COLUMN *,	// pColumn,
	ICD *				pIcd,
	FLMUINT			uiKeyComponent,
	FLMUINT			uiKeyLen)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiElmLen;
	FLMUINT				uiKeyLenPos = uiKeyLen;
	FLMBOOL				bDataTruncated;
	FLMBOOL				bIsNull;
	F_BufferIStream	columnBufferIStream;
	eDataType			eDataTyp;
	FLMUINT				uiDataLength;
	
	uiKeyLen += 2;
	
	if (pIcd->uiFlags & ICD_PRESENCE)
	{
		f_UINT32ToBigEndian( (FLMUINT32)pIcd->uiColumnNum, &m_keyGenInfo.pucKeyBuf [uiKeyLen]);
		uiKeyLen += 4;
		
		// Save the key component length.
		
		UW2FBA( 4, &m_keyGenInfo.pucKeyBuf [uiKeyLenPos]);
	}
	else
	{
		if (RC_BAD( rc = getColumnIStream( this,
											m_keyGenInfo.pRow,
											m_keyGenInfo.pColumnValues,
											pIcd->uiColumnNum,
											&bIsNull, &columnBufferIStream,
											&eDataTyp, &uiDataLength)))
		{
			goto Exit;
		}

		if (bIsNull || !columnBufferIStream.remainingSize())
		{
			// Save the key component length
			
			UW2FBA( 0, &m_keyGenInfo.pucKeyBuf [uiKeyLenPos]);
			
			rc = finishKeyComponent( pIcd, uiKeyComponent, uiKeyLen);
			goto Exit;
		}
			
		// Compute number of bytes left

		uiElmLen = SFLM_MAX_KEY_SIZE - uiKeyLen;
		bDataTruncated = FALSE;
		
		// Pass zero for compare rules - these are not strings.
		
		if( RC_BAD( rc = KYCollateValue( &m_keyGenInfo.pucKeyBuf [uiKeyLen],
						&uiElmLen, &columnBufferIStream,
						eDataTyp, pIcd->uiFlags,
						0, pIcd->uiLimit, NULL, NULL,
						m_keyGenInfo.pIndex->uiLanguage,
						FALSE, FALSE,
						&bDataTruncated, NULL)))
		{
			goto Exit;
		}
		uiKeyLen += uiElmLen;
		
		// Save the key component length.
		
		if (!bDataTruncated)
		{
			UW2FBA( (FLMUINT16)(uiElmLen),
						&m_keyGenInfo.pucKeyBuf [uiKeyLenPos]);
		}
		else
		{
			UW2FBA( (FLMUINT16)(uiElmLen | TRUNCATED_FLAG),
						&m_keyGenInfo.pucKeyBuf [uiKeyLenPos]);
		}
	}
	
	if (RC_BAD( rc = finishKeyComponent( pIcd, uiKeyComponent, uiKeyLen)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:		Build all compound keys from the CDL table.
Notes:	This routine is recursive in nature.  Will recurse the number of
			key components defined in the index.
****************************************************************************/
RCODE F_Db::buildKeys(
	ICD *		pIcd,
	FLMUINT	uiKeyComponent,
	FLMUINT	uiKeyLen)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBYTE *	pucTmpBuf = NULL;
	void *		pvMark = NULL;
	FLMUINT		uiTmpBufSize = 0;
	F_COLUMN *	pColumn = m_pDict->getColumn( m_keyGenInfo.pTable,
												pIcd->uiColumnNum);

	flmAssert( m_keyGenInfo.bIsCompound || uiKeyComponent == 1);

	// Generate the key component

	if (pColumn->eDataTyp == SFLM_STRING_TYPE && !(pIcd->uiFlags & ICD_PRESENCE))
	{
		if (RC_BAD( rc = genTextKeyComponents( pColumn, pIcd, uiKeyComponent, uiKeyLen,
									&pucTmpBuf, &uiTmpBufSize, &pvMark)))
		{
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = genOtherKeyComponent( pColumn, pIcd, uiKeyComponent, uiKeyLen)))
		{
			goto Exit;
		}
	}

Exit:

	if (pvMark)
	{
		m_tempPool.poolReset( pvMark);
	}
	return( rc);
}

/****************************************************************************
Desc:	Build all keys from combinations of CDLs.  Add keys to KREF table.
****************************************************************************/
RCODE F_Db::buildKeys(
	F_INDEX *			pIndex,
	F_TABLE *			pTable,
	F_Row *				pRow,
	FLMBOOL				bAddKeys,
	F_COLUMN_VALUE *	pColumnValues)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBYTE		ucDataBuf [STACK_DATA_BUF_SIZE];

	if (RC_BAD( rc = krefCntrlCheck()))
	{
		goto Exit;
	}
	
	// Build all of the keys
	
	m_keyGenInfo.pTable = pTable;
	m_keyGenInfo.pIndex = pIndex;
	m_keyGenInfo.bIsAsia = (FLMBOOL)(pIndex->uiLanguage >= FLM_FIRST_DBCS_LANG &&
							  					pIndex->uiLanguage <= FLM_LAST_DBCS_LANG)
											  ? TRUE
											  : FALSE;
	m_keyGenInfo.bIsCompound = pIndex->uiNumKeyComponents > 1 ? TRUE : FALSE;
	m_keyGenInfo.pucKeyBuf = m_pucKrefKeyBuf;
	m_keyGenInfo.pucData = &ucDataBuf [0];
	m_keyGenInfo.uiDataBufSize = sizeof( ucDataBuf);
	m_keyGenInfo.bDataBufAllocated = FALSE;
	
	// Build the keys for the row.
	
	m_keyGenInfo.pRow = pRow;
	m_keyGenInfo.pColumnValues = NULL;
	m_keyGenInfo.bAddKeys = bAddKeys;
	if (RC_BAD( rc = buildKeys( pIndex->pKeyIcds, 1, 0)))
	{
		goto Exit;
	}
		
	// Add the new keys, if this is an update operation.
	// pColumnValues will be non-NULL for update operations.
	
	if (pColumnValues)
	{
		// The bAddKeys that was passed in better have been FALSE - to delete
		// the old keys, and pRow should have been pointing to the old
		// row before it was modified.
		
		flmAssert( !bAddKeys);
		m_keyGenInfo.pColumnValues = pColumnValues;
		m_keyGenInfo.bAddKeys = TRUE;
		if (RC_BAD( rc = buildKeys( pIndex->pKeyIcds, 1, 0)))
		{
			goto Exit;
		}
	}
	
	// Commit keys if we are over the limit.
	
	if( isKrefOverThreshold())
	{
		processDupKeys( pIndex);
		if (RC_BAD( rc = keysCommit( FALSE, FALSE)))
		{
			goto Exit;
		}
	}
	
Exit:

	if (m_keyGenInfo.bDataBufAllocated)
	{
		f_free( &m_keyGenInfo.pucData);
		m_keyGenInfo.bDataBufAllocated = FALSE;
	}

	return( rc);
}

/****************************************************************************
Desc:	Routine that is called when inserting, modifying, or removing a row
		from a table.
****************************************************************************/
RCODE F_Db::updateIndexKeys(
	FLMUINT				uiTableNum,
	F_Row *				pRow,
	FLMBOOL				bAddKeys,
	F_COLUMN_VALUE *	pColumnValues)
{
	RCODE					rc = NE_SFLM_OK;
	F_TABLE *			pTable = m_pDict->getTable( uiTableNum);
	F_INDEX *			pIndex;
	FLMUINT				uiIndexNum;
	FLMUINT				uiColumnNum;
	ICD *					pIcd;
	FLMUINT				uiLoop;
	FLMUINT64			ui64RowId;
	F_COLUMN_VALUE *	pColumnValue;
	
	ui64RowId = pRow->getRowId();
	
	// Go through each index on this table.
	
	uiIndexNum = pTable->uiFirstIndexNum;
	while (uiIndexNum)
	{
		pIndex = m_pDict->getIndex( uiIndexNum);
		
		if (!(pIndex->uiFlags & (IXD_OFFLINE | IXD_SUSPENDED)) ||
			 ui64RowId <= pIndex->ui64LastRowIndexed)
		{
			FLMBOOL	bBuildKeys = TRUE;
			
			// If we are doing a modify operation, see if any of the columns
			// in this index are actually going to be modified.
			
			if (pColumnValues)
			{
				bBuildKeys = FALSE;
				pColumnValue = pColumnValues;
				while (pColumnValue && !bBuildKeys)
				{
					uiColumnNum = pColumnValue->uiColumnNum;
					
					// See if it is one of the key columns
					
					for (pIcd = pIndex->pKeyIcds, uiLoop = 0;
						  !bBuildKeys && uiLoop < pIndex->uiNumKeyComponents;
						  uiLoop++, pIcd++)
					{
						if (pIcd->uiColumnNum == uiColumnNum)
						{
							bBuildKeys = TRUE;
							break;
						}
					}
					
					// See if it is one of the data columns
					
					for (pIcd = pIndex->pDataIcds, uiLoop = 0;
						  !bBuildKeys && uiLoop < pIndex->uiNumDataComponents;
						  uiLoop++, pIcd++)
					{
						if (pIcd->uiColumnNum == uiColumnNum)
						{
							bBuildKeys = TRUE;
							break;
						}
					}
					pColumnValue = pColumnValue->pNext;
				}
			}
			
			if (bBuildKeys)
			{
				if (RC_BAD( rc = buildKeys( pIndex, pTable, pRow, bAddKeys,
												pColumnValues)))
				{
					goto Exit;
				}
			}
		}
		
		uiIndexNum = pIndex->uiNextIndexNum;
	}
	
Exit:

	return( rc);
}

