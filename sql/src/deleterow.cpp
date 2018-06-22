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
// Desc:	Delete a row from the database.
//------------------------------------------------------------------------------
RCODE F_Db::deleteRow(
	FLMUINT				uiTableNum,
	FLMUINT64			ui64RowId,
	FLMBOOL				bLogDelete)
{
	RCODE					rc = NE_SFLM_OK;
	FLMBYTE				ucKeyBuf[ FLM_MAX_NUM_BUF_SIZE];
	FLMUINT				uiKeyLen;
	F_Row *				pRow = NULL;
	F_Btree *			pBTree = NULL;
	FLMBOOL				bStartedTrans = FALSE;
	
	// Make sure we are in an update transaction.
	
	if (RC_BAD( rc = checkTransaction( SFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}
	
	// Cannot delete from internal system tables, unless it is an
	// internal delete.
	
	if (bLogDelete)
	{
		F_TABLE *	pTable = m_pDict->getTable( uiTableNum);
		if (pTable->bSystemTable)
		{
			rc = RC_SET( NE_SFLM_CANNOT_DELETE_IN_SYSTEM_TABLE);
			goto Exit;
		}
	}
	
	// First we need to retrieve the row and update any index keys.
	
	if (RC_BAD( rc = gv_SFlmSysData.pRowCacheMgr->retrieveRow( this,
								uiTableNum, ui64RowId, &pRow)))
	{
		goto Exit;
	}
	
	// Delete any index keys for the row.
	
	if (RC_BAD( rc = updateIndexKeys( uiTableNum, pRow, FALSE, NULL)))
	{
		goto Exit;
	}
	
	// Get a B-Tree object to delete the row from the b-tree.

	if( RC_BAD( rc = getCachedBTree( uiTableNum, &pBTree)))
	{
		goto Exit;
	}
	uiKeyLen = sizeof( ucKeyBuf);
	if( RC_BAD( rc = flmNumber64ToStorage( ui64RowId, &uiKeyLen, ucKeyBuf,
									FALSE, TRUE)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pBTree->btRemoveEntry( ucKeyBuf, uiKeyLen)))
	{
		goto Exit;
	}

	// Remove the row from row cache if it is there.
	
	gv_SFlmSysData.pRowCacheMgr->removeRow( this, pRow, TRUE, FALSE);
	pRow = NULL;
	
	if (bLogDelete)
	{
		if (RC_BAD( rc = m_pDatabase->m_pRfl->logDeleteRow( this, uiTableNum, ui64RowId)))
		{
			goto Exit;
		}
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

	if (pBTree)
	{
		pBTree->Release();
	}

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Delete selected rows from the database.
//------------------------------------------------------------------------------
RCODE F_Db::deleteSelectedRows(
	FLMUINT		uiTableNum,
	SQLQuery *	pSqlQuery)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBOOL		bStartedTrans = FALSE;
	F_TABLE *	pTable;
	F_Row *		pRow = NULL;
	
	// Make sure we are in an update transaction.
	
	if (RC_BAD( rc = checkTransaction( SFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}
	
	// Cannot delete from internal system tables.
	
	pTable = m_pDict->getTable( uiTableNum);
	if (pTable->bSystemTable)
	{
		rc = RC_SET( NE_SFLM_CANNOT_DELETE_IN_SYSTEM_TABLE);
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
		
		if (RC_BAD( rc = deleteRow( uiTableNum, pRow->m_ui64RowId, TRUE)))
		{
			goto Exit;
		}
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

	return( rc);
}
	
//------------------------------------------------------------------------------
// Desc:	Process the DELETE statement.  The "DELETE" keyword has already been
//			parsed.
//------------------------------------------------------------------------------
RCODE SQLStatement::processDeleteRows( void)
{
	RCODE					rc = NE_SFLM_OK;
	FLMBOOL				bStartedTrans = FALSE;
	char					szTableName [MAX_SQL_NAME_LEN + 1];
	FLMUINT				uiTableNameLen;
	F_TABLE *			pTable;
	TABLE_ITEM			tableList [2];
	SQLQuery				sqlQuery;

	// If we are in a read transaction, we cannot do this operation
	
	if (RC_BAD( rc = m_pDb->checkTransaction( SFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// SYNTAX: DELETE FROM table_name WHERE select-criteria

	// FROM must follow the DELETE.

	if (RC_BAD( rc = haveToken( "from", FALSE, SQL_ERR_EXPECTING_FROM)))
	{
		goto Exit;
	}

	// Whitespace must follow the "FROM"

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
		rc = RC_SET( NE_SFLM_CANNOT_DELETE_IN_SYSTEM_TABLE);
		goto Exit;
	}
	
	// See if we have a WHERE clause
	
	if (RC_BAD( rc = haveToken( "where", TRUE)))
	{
		if (rc == NE_SFLM_NOT_FOUND || rc == NE_SFLM_EOF_HIT)
		{
			if (RC_BAD( rc = sqlQuery.addTable( pTable->uiTableNum, NULL)))
			{
				goto Exit;
			}
		}
		else
		{
			goto Exit;
		}
	}
	else
	{
	
		tableList [0].uiTableNum = pTable->uiTableNum;
		tableList [0].pszTableAlias = pTable->pszTableName;
		
		// Null terminate the list.
		
		tableList [1].uiTableNum = 0;
		if (RC_BAD( rc = parseCriteria( &tableList [0], NULL, TRUE, NULL, &sqlQuery)))
		{
			goto Exit;
		}
	}
	
	if (RC_BAD( rc = m_pDb->deleteSelectedRows( pTable->uiTableNum, &sqlQuery)))
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

