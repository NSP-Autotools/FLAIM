//------------------------------------------------------------------------------
// Desc:	This module contains routines that will drop an index.
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
// Desc:	Drop a table.
//------------------------------------------------------------------------------
RCODE F_Db::dropTable(
	FLMUINT				uiTableNum)
{
	RCODE					rc = NE_SFLM_OK;
	F_TABLE *			pTable;
	F_COLUMN *			pColumn;
	FLMUINT				uiLoop;
	FLMBOOL				bStartedTrans = FALSE;
	
	// Make sure we are in an update transaction.
	
	if (RC_BAD( rc = checkTransaction( SFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}
	
	// Create a new dictionary, if we don't already have one.
	
	if (!(m_uiFlags & FDB_UPDATED_DICTIONARY))
	{
		if (RC_BAD( rc = dictClone()))
		{
			goto Exit;
		}
	}
	
	pTable = m_pDict->getTable( uiTableNum);
	flmAssert( pTable);
	
	// Cannot drop internal system tables.
	
	if (pTable->bSystemTable)
	{
		rc = RC_SET( NE_SFLM_CANNOT_DROP_SYSTEM_TABLE);
		goto Exit;
	}
	
	// Delete the row for the table in the table table.
	
	if (RC_BAD( rc = deleteRow( SFLM_TBLNUM_TABLES,
						pTable->ui64DefRowId, FALSE)))
	{
		goto Exit;
	}
	
	// Delete all of the rows that define the table's columns
	
	for (pColumn = pTable->pColumns, uiLoop = 0;
		  uiLoop < pTable->uiNumColumns;
		  uiLoop++, pColumn++)
	{
		if (RC_BAD( rc = deleteRow( SFLM_TBLNUM_COLUMNS,
							pColumn->ui64DefRowId, FALSE)))
		{
			goto Exit;
		}
	}
	
	// Initiate the background process to delete the b-tree.

	if (RC_BAD( rc = m_pDatabase->lFileDelete( this, &pTable->lfInfo,
												FALSE, TRUE)))
	{
		goto Exit;
	}

	// Remove the column name table.
	
	pTable->pColumnNames->Release();
	pTable->pColumnNames = NULL;
	
	// Remove the table name from the table name table.
	
	m_pDict->m_pTableNames->removeName( pTable->pszTableName);

	// Remove the table from the in-memory dictionary.  This is done
	// simply by zeroing out the entire structure.
	
	f_memset( pTable, sizeof( F_TABLE), 0);
	
	// Log the operation
	
	if (RC_BAD( rc = m_pDatabase->m_pRfl->logDropTable( this, uiTableNum)))
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
// Desc:	Process the drop index statement.  The "DROP INDEX" keywords
//			have already been parsed.
//------------------------------------------------------------------------------
RCODE SQLStatement::processDropTable( void)
{
	RCODE						rc = NE_SFLM_OK;
	FLMBOOL					bStartedTrans = FALSE;
	char						szTableName [MAX_SQL_NAME_LEN + 1];
	FLMUINT					uiTableNameLen;
	F_TABLE *				pTable;
	
	// If we are in a read transaction, we cannot do this operation
	
	if (RC_BAD( rc = m_pDb->checkTransaction( SFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// SYNTAX: DROP INDEX <indexname> [ON <tablename>]
	
	// Whitespace must follow the "DROP INDEX"

	if (RC_BAD( rc = skipWhitespace( TRUE)))
	{
		goto Exit;
	}

	// Get the table name - table name must exist
	
	if (RC_BAD( rc = getTableName( TRUE, szTableName, sizeof( szTableName),
							&uiTableNameLen, &pTable)))
	{
		goto Exit;
	}
	
	// Cannot drop system tables
	
	if (pTable->bSystemTable)
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset - 1,
				SQL_ERR_CANNOT_DROP_SYSTEM_TABLE,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_SFLM_INVALID_SQL);
		goto Exit;
	}
	
	// Drop the table.

	if (RC_BAD( rc = m_pDb->dropTable( pTable->uiTableNum)))
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

