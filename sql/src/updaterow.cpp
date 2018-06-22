//------------------------------------------------------------------------------
// Desc:	This module contains the routines for inserting a row into a table.
// Tabs:	3
//
// Copyright (c) 2006-2007 Novell, Inc. All Rights Reserved.
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

static const char * gv_setExprTerminators [3] =
{
	"where",
	",",
	NULL
};

FSTATIC RCODE convertValueToStringFormat(
	SQL_VALUE *			pSqlValue,
	F_COLUMN *			pColumn,
	F_COLUMN_VALUE *	pColValue,
	F_Pool *				pPool);
	
FSTATIC RCODE convertToNumber(
	const char *	pszStr,
	FLMUINT64 *		pui64Num,
	FLMBOOL *		pbNeg);
	
FSTATIC RCODE convertValueToNumberFormat(
	SQL_VALUE *			pSqlValue,
	F_COLUMN_VALUE *	pColValue,
	F_Pool *				pPool);
	
FSTATIC RCODE convertValueToBinaryFormat(
	SQL_VALUE *			pSqlValue,
	F_COLUMN *			pColumn,
	F_COLUMN_VALUE *	pColValue,
	F_Pool *				pPool);
	
FSTATIC RCODE convertValueToStorageFormat(
	SQL_VALUE *			pSqlValue,
	F_COLUMN *			pColumn,
	F_COLUMN_VALUE *	pColValue,
	F_Pool *				pPool);
	
//------------------------------------------------------------------------------
// Desc:	Update a row in the database.
//------------------------------------------------------------------------------
RCODE F_Db::updateRow(
	FLMUINT				uiTableNum,
	F_Row **				ppRow,
	F_COLUMN_VALUE *	pColumnValues)
{
	RCODE					rc = NE_SFLM_OK;
	F_COLUMN_VALUE *	pColumnValue;
	F_TABLE *			pTable;
	F_COLUMN *			pColumn;
	FLMBOOL				bStartedTrans = FALSE;
	F_Row *				pRow;
	
	// Make sure we are in an update transaction.
	
	if (RC_BAD( rc = checkTransaction( SFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	pTable = m_pDict->getTable( uiTableNum);
	if (pTable->bSystemTable)
	{
		rc = RC_SET( NE_SFLM_CANNOT_UPDATE_IN_SYSTEM_TABLE);
		goto Exit;
	}
	
	// Make sure we have a write-copy of the row before we start
	// modifying it.
	
	if (RC_BAD( rc = gv_SFlmSysData.pRowCacheMgr->makeWriteCopy( this, ppRow)))
	{
		goto Exit;
	}
	pRow = *ppRow;
	
	// Do whatever indexing needs to be done - BEFORE changing the values.
	
	if (RC_BAD( rc = updateIndexKeys( uiTableNum, pRow, FALSE, pColumnValues)))
	{
		goto Exit;
	}
	
	// Set the column values into the row.
	
	for (pColumnValue = pColumnValues;
		  pColumnValue;
		  pColumnValue = pColumnValue->pNext)
	{
		pColumn = m_pDict->getColumn( pTable, pColumnValue->uiColumnNum);
		if (pColumn->uiMaxLen)
		{
			if (pColumn->eDataTyp == SFLM_STRING_TYPE)
			{
				FLMUINT				uiNumChars;
				const FLMBYTE *	pucData = (const FLMBYTE *)pColumnValue->pucColumnValue;
				const FLMBYTE *	pucEnd = pucData + pColumnValue->uiValueLen;
				
				// Number of characters is the first part of the value
				
				if (RC_BAD( rc = f_decodeSEN( &pucData, pucEnd, &uiNumChars)))
				{
					goto Exit;
				}
				if (pColumnValue->uiValueLen > uiNumChars)
				{
					rc = RC_SET( NE_SFLM_STRING_TOO_LONG);
					goto Exit;
				}
			}
			else if (pColumn->eDataTyp == SFLM_BINARY_TYPE)
			{
				if (pColumnValue->uiValueLen > pColumn->uiMaxLen)
				{
					rc = RC_SET( NE_SFLM_BINARY_TOO_LONG);
					goto Exit;
				}
			}
		}
		if (RC_BAD( rc = pRow->setValue( this, pColumn,
													pColumnValue->pucColumnValue,
													pColumnValue->uiValueLen)))
		{
			goto Exit;
		}
	}
	
	// Log the insert row.
	
	if (RC_BAD( rc = m_pDatabase->m_pRfl->logUpdateRow( this, pRow->m_ui64RowId,
							uiTableNum, pColumnValues)))
	{
		goto Exit;
	}

	// Commit the transaction if we started it
	
	if (bStartedTrans)
	{
		bStartedTrans = FALSE;
		if (RC_BAD( rc = transCommit()))
		{
			goto Exit;
		}
	}

Exit:

	if (bStartedTrans)
	{
		transAbort();
	}

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Convert an SQL_VALUE to string format.
//------------------------------------------------------------------------------
FSTATIC RCODE convertValueToStringFormat(
	SQL_VALUE *			pSqlValue,
	F_COLUMN *			pColumn,
	F_COLUMN_VALUE *	pColValue,
	F_Pool *				pPool)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiSenLen;
	FLMBYTE *	pucValue;
	FLMUINT		uiLoop;
	FLMBYTE *	pucTmp;
	char			szTmp [100];
	
	switch (pSqlValue->eValType)
	{
		case SQL_BOOL_VAL:
			switch (pSqlValue->val.eBool)
			{
				case SQL_FALSE:
					f_strcpy( szTmp, "FALSE");
					pSqlValue->val.str.pszStr = (FLMBYTE *)&szTmp [0];
					pSqlValue->val.str.uiByteLen = 6;
					pSqlValue->val.str.uiNumChars = 5;
					break;
				case SQL_TRUE:
					f_strcpy( szTmp, "TRUE");
					pSqlValue->val.str.pszStr = (FLMBYTE *)&szTmp [0];
					pSqlValue->val.str.uiByteLen = 5;
					pSqlValue->val.str.uiNumChars = 4;
					break;
				case SQL_UNKNOWN:
				default:
					f_strcpy( szTmp, "UNKNOWN");
					pSqlValue->val.str.pszStr = (FLMBYTE *)&szTmp [0];
					pSqlValue->val.str.uiByteLen = 8;
					pSqlValue->val.str.uiNumChars = 7;
					break;
			}
			goto Output_Str;
		case SQL_UINT_VAL:
			f_sprintf( szTmp, "%u", (unsigned)pSqlValue->val.uiVal);
			pSqlValue->val.str.pszStr = (FLMBYTE *)&szTmp [0];
			pSqlValue->val.str.uiNumChars = f_strlen( szTmp);
			pSqlValue->val.str.uiByteLen = pSqlValue->val.str.uiNumChars + 1;
			goto Output_Str;
		case SQL_UINT64_VAL:
			f_sprintf( szTmp, "%I64u", pSqlValue->val.ui64Val);
			pSqlValue->val.str.pszStr = (FLMBYTE *)&szTmp [0];
			pSqlValue->val.str.uiNumChars = f_strlen( szTmp);
			pSqlValue->val.str.uiByteLen = pSqlValue->val.str.uiNumChars + 1;
			goto Output_Str;
		case SQL_INT_VAL:
			f_sprintf( szTmp, "%d", pSqlValue->val.iVal);
			pSqlValue->val.str.pszStr = (FLMBYTE *)&szTmp [0];
			pSqlValue->val.str.uiNumChars = f_strlen( szTmp);
			pSqlValue->val.str.uiByteLen = pSqlValue->val.str.uiNumChars + 1;
			goto Output_Str;
		case SQL_INT64_VAL:
			f_sprintf( szTmp, "%I64d", pSqlValue->val.i64Val);
			pSqlValue->val.str.pszStr = (FLMBYTE *)&szTmp [0];
			pSqlValue->val.str.uiNumChars = f_strlen( szTmp);
			pSqlValue->val.str.uiByteLen = pSqlValue->val.str.uiNumChars + 1;
			goto Output_Str;
		case SQL_BINARY_VAL:
		
			// Output two HEX bytes for every one byte of binary data.
			
			// See if the string would be too long.
			
			if (pColumn->uiMaxLen &&
				 pSqlValue->val.bin.uiByteLen * 2 > pColumn->uiMaxLen)
			{
				rc = RC_SET( NE_SFLM_STRING_TOO_LONG);
				goto Exit;
			}

			uiSenLen = f_getSENByteCount( pSqlValue->val.bin.uiByteLen * 2);
			pColValue->uiValueLen = pSqlValue->val.bin.uiByteLen * 2 + 1 + uiSenLen;
			if (RC_BAD( rc = pPool->poolAlloc( pColValue->uiValueLen,
													(void **)&pucValue)))
			{
				goto Exit;
			}
			pColValue->pucColumnValue = pucValue;
			f_encodeSEN( pSqlValue->val.bin.uiByteLen * 2, &pucValue);
			
			// Output the binary as hex - two bytes for every one byte.
			
			for (pucTmp = pSqlValue->val.bin.pucValue, uiLoop = 0;
				  uiLoop < pSqlValue->val.bin.uiByteLen;
				  uiLoop++, pucTmp++)
			{
				FLMBYTE	ucChar = (*pucTmp) >> 4;
				
				if (ucChar <= 9)
				{
					*pucValue++ = '0' + ucChar;
				}
				else
				{
					*pucValue++ = 'A' + ucChar - 10;
				}
				ucChar = (*pucTmp) & 0x0F;
				if (ucChar <= 9)
				{
					*pucValue++ = '0' + ucChar;
				}
				else
				{
					*pucValue++ = 'A' + ucChar - 10;
				}
			}
			*pucValue = 0;
			break;
		case SQL_UTF8_VAL:
Output_Str:
		
			// See if the string is too long.
			
			if (pColumn->uiMaxLen &&
				 pSqlValue->val.str.uiNumChars > pColumn->uiMaxLen)
			{
				rc = RC_SET( NE_SFLM_STRING_TOO_LONG);
				goto Exit;
			}

			uiSenLen = f_getSENByteCount( pSqlValue->val.str.uiNumChars);
			pColValue->uiValueLen = pSqlValue->val.str.uiByteLen + uiSenLen;
			if (RC_BAD( rc = pPool->poolAlloc( pColValue->uiValueLen,
													(void **)&pucValue)))
			{
				goto Exit;
			}
			pColValue->pucColumnValue = pucValue;
			f_encodeSEN( pSqlValue->val.str.uiNumChars, &pucValue);
			
			// Copy the string from the dynaBuf to the column.
			
			f_memcpy( pucValue, pSqlValue->val.str.pszStr,
									  pSqlValue->val.str.uiByteLen);
			break;
		default:
			flmAssert( 0);
			rc = RC_SET( NE_SFLM_FAILURE);
			goto Exit;
	}
	
Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Convert a string to a number.
//------------------------------------------------------------------------------
FSTATIC RCODE convertToNumber(
	const char *	pszStr,
	FLMUINT64 *		pui64Num,
	FLMBOOL *		pbNeg)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBYTE		ucChar;
	FLMUINT64	ui64Value = 0;
	FLMBOOL		bHex = FALSE;
	FLMUINT		uiDigitValue = 0;
	
	// See if it is a negative number.
	
	if (*pszStr == '-')
	{
		*pbNeg = TRUE;
		pszStr++;
	}
	else
	{
		*pbNeg = FALSE;
	}
	
	// See if it is a hex number.
	
	if (*pszStr == 'x' || *pszStr == 'X')
	{
		pszStr++;
		bHex = TRUE;
	}
	else if (*pszStr == '0' && (*(pszStr + 1) == 'x' || *(pszStr + 1) == 'X'))
	{
		pszStr++;
		bHex = TRUE;
	}
	
	// Go until we hit a character that is not a number.
	
	while (pszStr)
	{
		ucChar = (FLMBYTE)(*pszStr);
		pszStr++;
		
		if (ucChar >= '0' && ucChar <= '9')
		{
			uiDigitValue = (FLMUINT)(ucChar - '0');
		}
		else if (ucChar >= 'a' && ucChar <= 'f')
		{
			if (!bHex)
			{
				rc = RC_SET( NE_SFLM_CONV_BAD_DIGIT);
				goto Exit;
			}
			uiDigitValue = (FLMUINT)(ucChar - 'a' + 10);
		}
		else if (ucChar >= 'A' && ucChar <= 'F')
		{
			if (!bHex)
			{
				rc = RC_SET( NE_SFLM_CONV_BAD_DIGIT);
				goto Exit;
			}
			uiDigitValue = (FLMUINT)(ucChar - 'A' + 10);
		}
		else
		{
			rc = RC_SET( NE_SFLM_CONV_BAD_DIGIT);
			goto Exit;
		}
		
		if (bHex)
		{
			if (ui64Value > (FLM_MAX_UINT64 >> 4))
			{
				rc = RC_SET( NE_SFLM_CONV_NUM_OVERFLOW);
				goto Exit;
			}
			ui64Value <<= 4;
			ui64Value += (FLMUINT64)uiDigitValue;
		}
		else
		{
			if (ui64Value > (FLM_MAX_UINT64 / 10))
			{
				rc = RC_SET( NE_SFLM_CONV_NUM_OVERFLOW);
				goto Exit;
			}
			ui64Value *= 10;
			ui64Value += (FLMUINT64)uiDigitValue;
		}
		
		// If it is a negative number, make sure we have not
		// exceeded the maximum negative value.
		
		if (*pbNeg && ui64Value > ((FLMUINT64)1 << 63))
		{
			rc = RC_SET( NE_SFLM_CONV_NUM_OVERFLOW);
			goto Exit;
		}
	}
	
	if (!ui64Value)
	{
		*pbNeg = FALSE;
	}
	*pui64Num = ui64Value;
	
Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Convert an SQL_VALUE to number format.
//------------------------------------------------------------------------------
FSTATIC RCODE convertValueToNumberFormat(
	SQL_VALUE *			pSqlValue,
	F_COLUMN_VALUE *	pColValue,
	F_Pool *				pPool)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBYTE		ucNumBuf [40];
	FLMUINT		uiValLen;
	FLMBYTE *	pucValue;
	FLMBOOL		bNeg;
	FLMUINT64	ui64Value;
	
	switch (pSqlValue->eValType)
	{
		case SQL_BOOL_VAL:
			bNeg = FALSE;
			switch (pSqlValue->val.eBool)
			{
				case SQL_FALSE:
					ui64Value = 0;
					goto Output_Num;
				case SQL_TRUE:
					ui64Value = 1;
					goto Output_Num;
				case SQL_UNKNOWN:
				default:
					ui64Value = 2;
					goto Output_Num;
			}
		case SQL_UINT_VAL:
			ui64Value = (FLMUINT64)pSqlValue->val.uiVal;
			bNeg = FALSE;
Output_Num:
			uiValLen = sizeof( ucNumBuf);
			if (RC_BAD( rc = flmNumber64ToStorage( ui64Value, &uiValLen, ucNumBuf,
										bNeg, FALSE)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = pPool->poolAlloc( uiValLen, (void **)&pucValue)))
			{
				goto Exit;
			}
			pColValue->uiValueLen = uiValLen;
			pColValue->pucColumnValue = pucValue;
			f_memcpy( pucValue, ucNumBuf, uiValLen);
			break;
			
		case SQL_UINT64_VAL:
			ui64Value = (FLMUINT64)pSqlValue->val.ui64Val;
			bNeg = FALSE;
			goto Output_Num;
			
		case SQL_INT_VAL:
			if (pSqlValue->val.iVal < 0)
			{
				bNeg = TRUE;
				ui64Value = (FLMUINT64)(-pSqlValue->val.iVal);
			}
			else
			{
				bNeg = FALSE;
				ui64Value = (FLMUINT64)(pSqlValue->val.iVal);
			}
			goto Output_Num;
			
		case SQL_INT64_VAL:
			if (pSqlValue->val.i64Val < 0)
			{
				bNeg = TRUE;
				ui64Value = (FLMUINT64)(-pSqlValue->val.i64Val);
			}
			else
			{
				bNeg = FALSE;
				ui64Value = (FLMUINT64)(pSqlValue->val.i64Val);
			}
			goto Output_Num;
			
		case SQL_UTF8_VAL:
			if (RC_BAD( rc = convertToNumber(
									(const char *)pSqlValue->val.str.pszStr,
									&ui64Value, &bNeg)))
			{
				goto Exit;
			}
			goto Output_Num;
		case SQL_BINARY_VAL:
			if (pSqlValue->val.bin.uiByteLen > sizeof( FLMUINT64))
			{
				rc = RC_SET( NE_SFLM_CONV_NUM_OVERFLOW);
				goto Exit;
			}
			bNeg = FALSE;
			if (pSqlValue->val.bin.uiByteLen)
			{
				f_memcpy( &ui64Value, pSqlValue->val.bin.pucValue,
					pSqlValue->val.bin.uiByteLen);
				if (pSqlValue->val.bin.uiByteLen < sizeof( FLMUINT64))
				{
					ui64Value >>= (sizeof( FLMUINT64) - pSqlValue->val.bin.uiByteLen);
				}
			}
			else
			{
				ui64Value = 0;
			}
			goto Output_Num;
		default:
			flmAssert( 0);
			rc = RC_SET( NE_SFLM_FAILURE);
			goto Exit;
	}
	
Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Convert an SQL_VALUE to binary format.
//------------------------------------------------------------------------------
FSTATIC RCODE convertValueToBinaryFormat(
	SQL_VALUE *			pSqlValue,
	F_COLUMN *			pColumn,
	F_COLUMN_VALUE *	pColValue,
	F_Pool *				pPool)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBYTE *	pucValue;
	FLMUINT64	ui64Val;
	FLMUINT		uiVal;
	FLMINT		iVal;
	FLMINT64		i64Val;
	FLMBYTE		ucTmp;
	
	switch (pSqlValue->eValType)
	{
		case SQL_BOOL_VAL:
			switch (pSqlValue->val.eBool)
			{
				case SQL_FALSE:
					ucTmp = 0;
					break;
				case SQL_TRUE:
					ucTmp = 1;
					break;
				case SQL_UNKNOWN:
				default:
					ucTmp = 2;
					break;
			}
			pSqlValue->val.bin.pucValue = &ucTmp;
			pSqlValue->val.bin.uiByteLen = 1;
			goto Output_Binary;
		case SQL_UINT_VAL:
			uiVal = pSqlValue->val.uiVal;
			pSqlValue->val.bin.pucValue = (FLMBYTE *)(&uiVal);
			pSqlValue->val.bin.uiByteLen = sizeof( FLMUINT);
			goto Output_Binary;
		case SQL_UINT64_VAL:
			ui64Val = pSqlValue->val.ui64Val;
			pSqlValue->val.bin.pucValue = (FLMBYTE *)(&ui64Val);
			pSqlValue->val.bin.uiByteLen = sizeof( FLMUINT64);
			goto Output_Binary;
		case SQL_INT_VAL:
			iVal = pSqlValue->val.iVal;
			pSqlValue->val.bin.pucValue = (FLMBYTE *)(&iVal);
			pSqlValue->val.bin.uiByteLen = sizeof( FLMINT);
			goto Output_Binary;
		case SQL_INT64_VAL:
			i64Val = pSqlValue->val.i64Val;
			pSqlValue->val.bin.pucValue = (FLMBYTE *)(&i64Val);
			pSqlValue->val.bin.uiByteLen = sizeof( FLMINT64);
			goto Output_Binary;
		case SQL_UTF8_VAL:
		
			// Try to convert the string to binary - assume it is a hex
			// string, so the largest binary value will be half the string
			// size - because we convert two bytes to a single binary byte.
			// Ignore white space in the string.
			
			if (!pSqlValue->val.str.uiByteLen)
			{
				pColValue->pucColumnValue = NULL;
			}
			else
			{
				FLMBOOL		bHaveHighNibble;
				FLMBYTE		ucBinChar;
				FLMBYTE *	pucTmp;
				FLMBYTE		ucStrChar;
				
				if (RC_BAD( rc = pPool->poolAlloc( pSqlValue->val.str.uiByteLen / 2 + 1,
														(void **)&pucValue)))
				{
					goto Exit;
				}
				pColValue->pucColumnValue = pucValue;
				pucTmp = (FLMBYTE *)pSqlValue->val.str.pszStr;
				
				bHaveHighNibble = FALSE;
				ucBinChar = 0;
				while (*pucTmp)
				{
					ucStrChar = *pucTmp;
					pucTmp++;
					if (ucStrChar >= '0' && ucStrChar <= '9')
					{
						if (!bHaveHighNibble)
						{
							ucBinChar = (FLMBYTE)(ucStrChar - '0') << 4;
							bHaveHighNibble = TRUE;
						}
						else
						{
							*pucValue++ = ucBinChar | (FLMBYTE)(ucStrChar - '0');
							bHaveHighNibble = FALSE;
						}
					}
					else if (ucStrChar >= 'a' && ucStrChar <= 'f')
					{
						if (!bHaveHighNibble)
						{
							ucBinChar = (FLMBYTE)(ucStrChar - 'a' + 10) << 4;
							bHaveHighNibble = TRUE;
						}
						else
						{
							*pucValue++ = ucBinChar | (FLMBYTE)(ucStrChar - 'a' + 10);
							bHaveHighNibble = FALSE;
						}
					}
					else if (ucStrChar >= 'A' && ucStrChar <= 'F')
					{
						if (!bHaveHighNibble)
						{
							ucBinChar = (FLMBYTE)(ucStrChar - 'A' + 10) << 4;
							bHaveHighNibble = TRUE;
						}
						else
						{
							*pucValue++ = ucBinChar | (FLMBYTE)(ucStrChar - 'A' + 10);
							bHaveHighNibble = FALSE;
						}
					}
					else if (ucStrChar == ' ' || ucStrChar == '\t' ||
								ucStrChar == '\n' || ucStrChar == '\r')
					{
						// Skip over white space.
					}
					else
					{
						rc = RC_SET( NE_SFLM_CONV_BAD_DIGIT);
						goto Exit;
					}
				}
				if (bHaveHighNibble)
				{
					*pucValue++ = ucBinChar;
				}
				
				// See if we have too many bytes.
				
				pColValue->uiValueLen = (FLMUINT)(pucValue - pColValue->pucColumnValue);
				if (pColumn->uiMaxLen && pColValue->uiValueLen > pColumn->uiMaxLen)
				{
					rc = RC_SET( NE_SFLM_BINARY_TOO_LONG);
					goto Exit;
				}
			}
			break;
		case SQL_BINARY_VAL:
Output_Binary:		
			// See if the binary data is too long for the
			// column.
			
			if (pColumn->uiMaxLen &&
				 pSqlValue->val.bin.uiByteLen > pColumn->uiMaxLen)
			{
				rc = RC_SET( NE_SFLM_BINARY_TOO_LONG);
				goto Exit;
			}
			if ((pColValue->uiValueLen = pSqlValue->val.bin.uiByteLen) > 0)
			{
				if (RC_BAD( rc = pPool->poolAlloc( pColValue->uiValueLen,
														(void **)&pucValue)))
				{
					goto Exit;
				}
				pColValue->pucColumnValue = pucValue;
				
				// Copy the string from the dynaBuf to the column.
				
				f_memcpy( pucValue, pSqlValue->val.bin.pucValue,
										  pSqlValue->val.bin.uiByteLen);
			}
			else
			{
				pColValue->pucColumnValue = NULL;
			}
			break;
		default:
			flmAssert( 0);
			rc = RC_SET( NE_SFLM_FAILURE);
			goto Exit;
	}
	
Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Convert an SQL_VALUE to storage format.
//------------------------------------------------------------------------------
FSTATIC RCODE convertValueToStorageFormat(
	SQL_VALUE *			pSqlValue,
	F_COLUMN *			pColumn,
	F_COLUMN_VALUE *	pColValue,
	F_Pool *				pPool)
{
	RCODE			rc = NE_SFLM_OK;
	
	// Check for a missing value - return a NULL.
	
	if (pSqlValue->eValType == SQL_MISSING_VAL)
	{
		pColValue->uiValueLen = 0;
		pColValue->pucColumnValue = NULL;
		goto Exit;
	}
	
	switch (pColumn->eDataTyp)
	{
		case SFLM_STRING_TYPE:
			if (RC_BAD( rc = convertValueToStringFormat( pSqlValue,
										pColumn, pColValue, pPool)))
			{
				goto Exit;
			}
			break;
		case SFLM_NUMBER_TYPE:
			if (RC_BAD( rc = convertValueToNumberFormat( pSqlValue,
										pColValue, pPool)))
			{
				goto Exit;
			}
			break;
		case SFLM_BINARY_TYPE:
			if (RC_BAD( rc = convertValueToBinaryFormat( pSqlValue,
										pColumn, pColValue, pPool)))
			{
				goto Exit;
			}
			break;

		default:
			flmAssert( 0);
			rc = RC_SET( NE_SFLM_FAILURE);
			goto Exit;
	}
	
Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Update selected rows in the database.
//------------------------------------------------------------------------------
RCODE F_Db::updateSelectedRows(
	FLMUINT			uiTableNum,
	SQLQuery *		pSqlQuery,
	COLUMN_SET *	pFirstColumnSet,
	FLMUINT)			// uiNumColumnsToSet)
{
	RCODE					rc = NE_SFLM_OK;
	FLMBOOL				bStartedTrans = FALSE;
	F_TABLE *			pTable;
	F_COLUMN *			pColumn;
	F_Row *				pRow = NULL;
	F_COLUMN_VALUE *	pFirstColValue;
	F_COLUMN_VALUE *	pLastColValue;
	F_COLUMN_VALUE *	pColValue;
	F_COLUMN_ITEM *	pColItem;
	COLUMN_SET *		pColSet;
	SQL_VALUE *			pSqlValue;
	F_Pool				tmpPool;
	FLMBOOL				bValueChanged;
	
	tmpPool.poolInit( 2048);
	
	// Make sure we are in an update transaction.
	
	if (RC_BAD( rc = checkTransaction( SFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}
	
	// Cannot update in internal system tables.
	
	pTable = m_pDict->getTable( uiTableNum);
	if (pTable->bSystemTable)
	{
		rc = RC_SET( NE_SFLM_CANNOT_UPDATE_IN_SYSTEM_TABLE);
		goto Exit;
	}
	
	// Execute the query
	
	for (;;)
	{
		if (RC_BAD( rc = pSqlQuery->getNext( &pRow)))
		{
			if (rc == NE_SFLM_EOF_HIT)
			{
				rc = NE_SFLM_OK;
				break;
			}
			else
			{
				goto Exit;
			}
		}
		pFirstColValue = NULL;
		pLastColValue = NULL;
		
		pColSet = pFirstColumnSet;
		while (pColSet)
		{
			
			// Allocate a column value and link it into the list of
			// column values.
			
			if (RC_BAD( rc = tmpPool.poolAlloc( sizeof( F_COLUMN_VALUE),
												(void **)&pColValue)))
			{
				goto Exit;
			}
			pColValue->uiColumnNum = pColSet->uiColumnNum;
			bValueChanged = TRUE;
			
			// Evaluate the column value.
			
			if (!pColSet->pSqlQuery)
			{
Set_Null_Value:
				
				// Set value to NULL
				
				pColValue->pucColumnValue = NULL;
				pColValue->uiValueLen = 0;
				
				// If the value is already NULL, no need to set it again.
				
				if (pRow->getColumn( pColSet->uiColumnNum) == NULL)
				{
					bValueChanged = FALSE;
				}
			}
			else
			{
				if (RC_BAD( rc = sqlEvalCriteria(  this,
													pSqlQuery->m_pQuery,
													&pSqlValue,
													&tmpPool, pRow,
													m_pDatabase->m_uiDefaultLanguage)))
				{
					goto Exit;
				}
				pColumn = m_pDict->getColumn( pTable, pColSet->uiColumnNum);
				
				if (RC_BAD( rc = convertValueToStorageFormat( pSqlValue,
											pColumn, pColValue, &tmpPool)))
				{
					goto Exit;
				}
				
				if (!pColValue->uiValueLen)
				{
					goto Set_Null_Value;
				}
				
				// See if the value changed from what it was.  If not, unlink
				// it from the list.
				
				if ((pColItem = pRow->getColumn( pColSet->uiColumnNum)) != NULL)
				{
					FLMBYTE *	pucColDataPtr = pRow->getColumnDataPtr( pColSet->uiColumnNum);
					
					if (pColItem->uiDataLen == pColValue->uiValueLen &&
						 f_memcmp( pColValue->pucColumnValue, pucColDataPtr,
									  pColItem->uiDataLen) == 0)
					{
						bValueChanged = FALSE;
					}
				}
			}
			
			// Only link the value into the list if it actually changed from
			// what it was before.

			if (bValueChanged)
			{
				pColValue->pNext = NULL;
				if (pLastColValue)
				{
					pLastColValue->pNext = pColValue;
				}
				else
				{
					pFirstColValue = pColValue;
				}
				pLastColValue = pColValue;
			}
			pColSet = pColSet->pNext;
		}
		
		// No need to actually do an update if no columns changed on the row.
		
		if (pFirstColValue)
		{
			if (RC_BAD( rc = updateRow( uiTableNum, &pRow, pFirstColValue)))
			{
				goto Exit;
			}
		}
		tmpPool.poolReset( NULL);
	}
	
	// Commit the transaction if we started it
	
	if (bStartedTrans)
	{
		bStartedTrans = FALSE;
		if (RC_BAD( rc = transCommit()))
		{
			goto Exit;
		}
	}

Exit:

	if (pRow)
	{
		pRow->ReleaseRow();
	}

	if (bStartedTrans)
	{
		transAbort();
	}
	
	tmpPool.poolFree();

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Parse the columns that are to be set in the UPDATE statement.
//------------------------------------------------------------------------------
RCODE SQLStatement::parseSetColumns(
	TABLE_ITEM *	pTableList,
	COLUMN_SET **	ppFirstColumnSet,
	COLUMN_SET **	ppLastColumnSet,
	FLMUINT *		puiNumColumnsToSet,
	FLMBOOL *		pbHadWhere)
{
	RCODE				rc = NE_SFLM_OK;
	char				szToken [MAX_SQL_TOKEN_SIZE + 1];
	char				szColumnName [MAX_SQL_NAME_LEN + 1];
	FLMUINT			uiColumnNameLen;
	F_COLUMN *		pColumn;
	F_TABLE *		pTable = m_pDb->m_pDict->getTable( pTableList->uiTableNum);
	FLMUINT			uiTokenLineOffset;
	COLUMN_SET *	pColumnSet;
	SQLQuery *		pSqlQuery = NULL;
	
	*pbHadWhere = FALSE;
	
	// Must have the keyword "SET"
	
	if (RC_BAD( rc = haveToken( "set", FALSE, SQL_ERR_EXPECTING_SET)))
	{
		goto Exit;
	}
	
	for (;;)
	{
		
		// Get a column name.
		
		if (RC_BAD( rc = getName( szColumnName, sizeof( szColumnName),
										&uiColumnNameLen, &uiTokenLineOffset)))
		{
			goto Exit;
		}
			
		// See if the column is defined in the table.
		
		if ((pColumn = m_pDb->m_pDict->findColumn( pTable, szColumnName)) == NULL)
		{
			
			// See if it is the table name.  If so, the next token must be
			// a period, followed by the column name.
			
			if (f_stricmp( szColumnName, pTable->pszTableName) == 0)
			{
				if (RC_BAD( rc = haveToken( ".", FALSE, SQL_ERR_EXPECTING_PERIOD)))
				{
					goto Exit;
				}
				if (RC_BAD( rc = getName( szColumnName, sizeof( szColumnName),
												&uiColumnNameLen, &uiTokenLineOffset)))
				{
					goto Exit;
				}
				pColumn = m_pDb->m_pDict->findColumn( pTable, szColumnName);
			}
		}
		if (!pColumn)
		{
			setErrInfo( m_uiCurrLineNum,
					uiTokenLineOffset,
					SQL_ERR_UNDEFINED_COLUMN,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
			goto Exit;
		}
		
		// Equal must follow the colum name
		
		if (RC_BAD( rc = haveToken( "=", FALSE, SQL_ERR_EXPECTING_EQUAL)))
		{
			goto Exit;
		}
		
		// Allocate a column set structure
		
		if (RC_BAD( rc = m_tmpPool.poolAlloc( sizeof( COLUMN_SET),
												(void **)&pColumnSet)))
		{
			goto Exit;
		}
		
		// See if the next token is NULL.  If so, there is no expression
		// for this column - it is to be set to NULL.
		
		if (RC_OK( rc = haveToken( "null", FALSE)))
		{
			pSqlQuery = NULL;
		}
		else if (rc != NE_SFLM_NOT_FOUND)
		{
			goto Exit;
		}
		else
		{
			
			// Allocate an SQLQuery object, have the pColumnSet structure
			// point to it, and link the pColumnSet structure into the linked
			// list.
			
			if ((pSqlQuery = f_new SQLQuery) == NULL)
			{
				rc = RC_SET( NE_SFLM_MEM);
				goto Exit;
			}
		}
		pColumnSet->pSqlQuery = pSqlQuery;
		pColumnSet->uiColumnNum = pColumn->uiColumnNum;
		pColumnSet->pNext = NULL;
		if (*ppLastColumnSet)
		{
			(*ppLastColumnSet)->pNext = pColumnSet;
		}
		else
		{
			*ppFirstColumnSet = pColumnSet;
		}
		*ppLastColumnSet = pColumnSet;
		(*puiNumColumnsToSet)++;
		
		// Now parse the criteria for the SET command, unless NULL has already
		// been detected.
		
		if (!pSqlQuery)
		{
			if (RC_BAD( rc = getToken( szToken, sizeof( szToken), TRUE,
												&uiTokenLineOffset, NULL)))
			{
				if (rc == NE_SFLM_EOF_HIT)
				{
					rc = NE_SFLM_OK;
					break;
				}
				goto Exit;
			}
			else if (f_stricmp( szToken, "where") == 0)
			{
				*pbHadWhere = TRUE;
				break;
			}
			else if (f_stricmp( szToken, ",") != 0)
			{
				setErrInfo( m_uiCurrLineNum,
						uiTokenLineOffset,
						SQL_ERR_EXPECTING_COMMA,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
		}
		else
		{
			const char *	pszTerminator;
			
			if (RC_BAD( rc = parseCriteria( pTableList,
										&gv_setExprTerminators [0], TRUE,
										&pszTerminator, pSqlQuery)))
			{
				goto Exit;
			}
			
			// Strip out NOT operators, resolve constant arithmetic expressions,
			// and weed out boolean constants, but do not flatten the AND
			// and OR operators in the query tree.
	
			if (RC_BAD( rc = pSqlQuery->reduceTree( FALSE)))
			{
				goto Exit;
			}
		
			// Next token should either be a comma or the WHERE keyword, or we
			// should have hit EOF.  pszTerminator will return NULL if EOF was
			// hit.
			
			if (!pszTerminator)
			{
				break;
			}
			
			if (f_stricmp( pszTerminator, "where") == 0)
			{
				*pbHadWhere = TRUE;
				break;
			}
		}
	}
	
Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Process the UPDATE statement.  The "UPDATE" keyword has already been
//			parsed.
//------------------------------------------------------------------------------
RCODE SQLStatement::processUpdateRows( void)
{
	RCODE					rc = NE_SFLM_OK;
	FLMBOOL				bStartedTrans = FALSE;
	char					szTableName [MAX_SQL_NAME_LEN + 1];
	FLMUINT				uiTableNameLen;
	F_TABLE *			pTable;
	TABLE_ITEM			tableList [2];
	COLUMN_SET *		pFirstColumnSet = NULL;
	COLUMN_SET *		pLastColumnSet = NULL;
	FLMUINT				uiNumColumnsToSet = 0;
	SQLQuery				sqlQuery;
	FLMBOOL				bHadWhere;

	// If we are in a read transaction, we cannot do this operation
	
	if (RC_BAD( rc = m_pDb->checkTransaction( SFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// SYNTAX: UPDATE table_name
	//	SET column = {expression | NULL}
	// 	[, column = {expression | NULL}]...
	// [WHERE <search criteria>]

	// Get the table name.

	if (RC_BAD( rc = getTableName( TRUE, szTableName, sizeof( szTableName),
								&uiTableNameLen, &pTable)))
	{
		goto Exit;
	}
	
	if (pTable->bSystemTable)
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset,
				SQL_ERR_CANNOT_UPDATE_SYSTEM_TABLE,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_SFLM_CANNOT_UPDATE_IN_SYSTEM_TABLE);
		goto Exit;
	}
	
	// Must SET at least one column - get list of columns to be set and
	// the expressions for each column.
	
	tableList [0].uiTableNum = pTable->uiTableNum;
	tableList [0].pszTableAlias = pTable->pszTableName;
		
	// Null terminate the list.
	
	tableList [1].uiTableNum = 0;
	
	if (RC_BAD( rc = parseSetColumns( &tableList [0], &pFirstColumnSet,
								&pLastColumnSet, &uiNumColumnsToSet,
								&bHadWhere)))
	{
		goto Exit;
	}
	
	// See if we have a WHERE clause
	
	if (!bHadWhere)
	{
		if (RC_BAD( rc = sqlQuery.addTable( pTable->uiTableNum, NULL)))
		{
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = parseCriteria( &tableList [0], NULL, TRUE, NULL, &sqlQuery)))
		{
			goto Exit;
		}
	}
	
	if (RC_BAD( rc = m_pDb->updateSelectedRows( pTable->uiTableNum, &sqlQuery,
								pFirstColumnSet, uiNumColumnsToSet)))
	{
		goto Exit;
	}
	
	// Commit the transaction if we started it
	
	if (bStartedTrans)
	{
		bStartedTrans = FALSE;
		if (RC_BAD( rc = m_pDb->transCommit()))
		{
			goto Exit;
		}
	}

Exit:

	while (pFirstColumnSet)
	{
		if (pFirstColumnSet->pSqlQuery)
		{
			pFirstColumnSet->pSqlQuery->Release();
			pFirstColumnSet->pSqlQuery = NULL;
		}
		pFirstColumnSet = pFirstColumnSet->pNext;
	}

	if (bStartedTrans)
	{
		m_pDb->transAbort();
	}

	return( rc);
}

