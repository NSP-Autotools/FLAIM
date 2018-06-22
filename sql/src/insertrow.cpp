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

//------------------------------------------------------------------------------
// Desc:	Insert a row into the database.
//------------------------------------------------------------------------------
RCODE F_Db::insertRow(
	FLMUINT				uiTableNum,
	F_COLUMN_VALUE *	pColumnValues)
{
	RCODE					rc = NE_SFLM_OK;
	F_Row *				pRow = NULL;
	F_COLUMN_VALUE *	pColumnValue;
	F_TABLE *			pTable;
	F_COLUMN *			pColumn;
	FLMBOOL				bStartedTrans = FALSE;
	FLMUINT				uiNumReqColumns;
	
	// Make sure we are in an update transaction.
	
	if (RC_BAD( rc = checkTransaction( SFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	pTable = m_pDict->getTable( uiTableNum);
	if (pTable->bSystemTable)
	{
		rc = RC_SET( NE_SFLM_CANNOT_INSERT_IN_SYSTEM_TABLE);
		goto Exit;
	}
	
	// Create a row object.
	
	if (RC_BAD( rc = gv_SFlmSysData.pRowCacheMgr->createRow( this,
								uiTableNum, &pRow)))
	{
		goto Exit;
	}
	
	// Set the column values into the row.
	
	for (pColumnValue = pColumnValues, uiNumReqColumns = 0;
		  pColumnValue;
		  pColumnValue = pColumnValue->pNext)
	{
		if (!pColumnValue->uiValueLen)
		{
			continue;
		}
		
		// Make sure the data does not exceed the maximum if it is string
		// or binary.
		
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
		if (!(pColumn->uiFlags & COL_NULL_ALLOWED))
		{
			uiNumReqColumns++;
		}
		
		if (RC_BAD( rc = pRow->setValue( this, pColumn,
													pColumnValue->pucColumnValue,
													pColumnValue->uiValueLen)))
		{
			goto Exit;
		}
	}
	
	// See if we got all of the required columns
	
	if (uiNumReqColumns != pTable->uiNumReqColumns)
	{
		rc = RC_SET( NE_SFLM_MISSING_REQUIRED_COLUMN);
		goto Exit;
	}
	
	// Do whatever indexing needs to be done.
	
	if (RC_BAD( rc = updateIndexKeys( uiTableNum, pRow, TRUE, NULL)))
	{
		goto Exit;
	}
	
	// Log the insert row.
	
	if (RC_BAD( rc = m_pDatabase->m_pRfl->logInsertRow( this, uiTableNum,
							pColumnValues)))
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

	if (pRow)
	{
		pRow->ReleaseRow();
	}

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Process the insert statement.  The "INSERT" keyword has already been
//			parsed.
//------------------------------------------------------------------------------
RCODE SQLStatement::processInsertRow( void)
{
	RCODE					rc = NE_SFLM_OK;
	FLMBOOL				bStartedTrans = FALSE;
	F_COLUMN_VALUE *	pFirstColValue;
	F_COLUMN_VALUE *	pLastColValue;
	F_COLUMN_VALUE *	pColumnValue;
	F_COLUMN *			pColumn;
	FLMUINT				uiLoop;
	char					szColumnName [MAX_SQL_NAME_LEN + 1];
	FLMUINT				uiColumnNameLen;
	char					szTableName [MAX_SQL_NAME_LEN + 1];
	FLMUINT				uiTableNameLen;
	F_TABLE *			pTable;
	char					szToken [MAX_SQL_TOKEN_SIZE + 1];
	FLMUINT				uiTokenLineOffset;

	// If we are in a read transaction, we cannot do this operation
	
	if (RC_BAD( rc = m_pDb->checkTransaction( SFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// SYNTAX: INSERT INTO table_name (column1,column2,...) VALUES (value1,value2,...)
	// OR:     INSERT INTO table_name VALUES (value1,value2,...)

	// INTO must follow the INSERT.

	if (RC_BAD( rc = haveToken( "into", FALSE, SQL_ERR_EXPECTING_INTO)))
	{
		goto Exit;
	}

	// Whitespace must follow the "INTO"

	if (RC_BAD( rc = skipWhitespace( TRUE)))
	{
		goto Exit;
	}

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
		rc = RC_SET( NE_SFLM_CANNOT_INSERT_IN_SYSTEM_TABLE);
		goto Exit;
	}
	
	// If left paren follows table name, then columns are being listed.

	pFirstColValue = NULL;
	pLastColValue = NULL;
	if (RC_BAD( rc = haveToken( "(", FALSE)))
	{
		if (rc != NE_SFLM_NOT_FOUND)
		{
			rc = NE_SFLM_OK;
			for (uiLoop = 0, pColumn = pTable->pColumns;
				  uiLoop < pTable->uiNumColumns;
				  uiLoop++, pColumn++)
			{
				if (pColumn->uiColumnNum)
				{
					// Allocate a column value.
					
					if (RC_BAD( rc = m_tmpPool.poolAlloc( sizeof( F_COLUMN_VALUE),
												(void **)&pColumnValue)))
					{
						goto Exit;
					}
					pColumnValue->uiColumnNum = pColumn->uiColumnNum;
					pColumnValue->uiValueLen = 0;
					pColumnValue->pNext = NULL;
					if (pLastColValue)
					{
						pLastColValue->pNext = pColumnValue;
					}
					else
					{
						pFirstColValue = pColumnValue;
					}
					pLastColValue = pColumnValue;
				}
			}
		}
		else
		{
			goto Exit;
		}
	}
	else
	{

		// Get the list of columns for which there will be values.
		
		for (;;)
		{
			
			// Get the column name
			
			if (RC_BAD( rc = getName( szColumnName, sizeof( szColumnName),
											&uiColumnNameLen, &uiTokenLineOffset)))
			{
				goto Exit;
			}
			
			// See if the column is defined in the table.
			
			if (uiColumnNameLen)
			{
				if ((pColumn = m_pDb->m_pDict->findColumn( pTable, szColumnName)) == NULL)
				{
					setErrInfo( m_uiCurrLineNum,
							uiTokenLineOffset,
							SQL_ERR_UNDEFINED_COLUMN,
							m_uiCurrLineFilePos,
							m_uiCurrLineBytes);
					rc = RC_SET( NE_SFLM_INVALID_SQL);
					goto Exit;
				}
				
				// Allocate a column value.
				
				if (RC_BAD( rc = m_tmpPool.poolAlloc( sizeof( F_COLUMN_VALUE),
											(void **)&pColumnValue)))
				{
					goto Exit;
				}
				
				pColumnValue->uiColumnNum = pColumn->uiColumnNum;
				pColumnValue->uiValueLen = 0;
				pColumnValue->pNext = NULL;
				if (pLastColValue)
				{
					pLastColValue->pNext = pColumnValue;
				}
				else
				{
					pFirstColValue = pColumnValue;
				}
				pLastColValue = pColumnValue;
			}

			if (RC_BAD( rc = getToken( szToken, sizeof( szToken), FALSE,
										&uiTokenLineOffset, NULL)))
			{
				goto Exit;
			}
			
			// See if we are at the end of the list of columns
			
			if (f_stricmp( szToken, ")") == 0)
			{
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
	}

	// Allow for no values to be specified if no columns were.
	
	if (pFirstColValue)
	{
		if (RC_BAD( rc = haveToken( "values", FALSE, SQL_ERR_EXPECTING_VALUES)))
		{
			goto Exit;
		}
		
		// Should be a left paren
		
		if (RC_BAD( rc = haveToken( "(", FALSE, SQL_ERR_EXPECTING_LPAREN)))
		{
			goto Exit;
		}
		
		pColumnValue = pFirstColValue;
		for (;;)
		{
			pColumn = m_pDb->m_pDict->getColumn( pTable, pColumnValue->uiColumnNum);
			if (RC_BAD( rc = skipWhitespace( FALSE)))
			{
				goto Exit;
			}
			
			// Get the column value
			
			if (RC_BAD( rc = getValue( pColumn, pColumnValue)))
			{
				goto Exit;
			}
			
			if (RC_BAD( rc = skipWhitespace( FALSE)))
			{
				goto Exit;
			}
			if ((pColumnValue = pColumnValue->pNext) == NULL)
			{
				if (RC_BAD( rc = haveToken( ")", FALSE, SQL_ERR_EXPECTING_RPAREN)))
				{
					goto Exit;
				}
				else
				{
					break;
				}
			}
			else if (RC_BAD( rc = haveToken( ",", FALSE, SQL_ERR_EXPECTING_COMMA)))
			{
				goto Exit;
			}
		}
	}
	
	// Insert the row.
	
	if (RC_BAD( rc = m_pDb->insertRow( pTable->uiTableNum, pFirstColValue)))
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

	if (bStartedTrans)
	{
		m_pDb->transAbort();
	}

	return( rc);
}

