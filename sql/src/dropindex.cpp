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
// Desc:	Drop an index.
//------------------------------------------------------------------------------
RCODE F_Db::dropIndex(
	FLMUINT				uiIndexNum)
{
	RCODE					rc = NE_SFLM_OK;
	F_INDEX *			pIndex;
	ICD *					pIcd;
	FLMUINT				uiLoop;
	FLMBOOL				bStartedTrans = FALSE;
	IXD_FIXUP *			pIxdFixup;
	IXD_FIXUP *			pPrevIxdFixup;
	
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
	
	pIndex = m_pDict->getIndex( uiIndexNum);
	flmAssert( pIndex);
	
	// Cannot drop internal system indexes.
	
	if (pIndex->uiFlags & IXD_SYSTEM)
	{
		rc = RC_SET( NE_SFLM_CANNOT_DROP_SYSTEM_INDEX);
		goto Exit;
	}
	
	// Delete the row for the index in the index table.
	
	if (RC_BAD( rc = deleteRow( SFLM_TBLNUM_INDEXES,
						pIndex->ui64DefRowId, FALSE)))
	{
		goto Exit;
	}
	
	// Delete all of the rows that define the key components.
	
	for (pIcd = pIndex->pKeyIcds, uiLoop = 0;
		  uiLoop < pIndex->uiNumKeyComponents;
		  uiLoop++, pIcd++)
	{
		if (RC_BAD( rc = deleteRow( SFLM_TBLNUM_INDEX_COMPONENTS,
							pIcd->ui64DefRowId, FALSE)))
		{
			goto Exit;
		}
	}
	
	// Delete all of the rows that define the data components.
	
	for (pIcd = pIndex->pDataIcds, uiLoop = 0;
		  uiLoop < pIndex->uiNumDataComponents;
		  uiLoop++, pIcd++)
	{
		if (RC_BAD( rc = deleteRow( SFLM_TBLNUM_INDEX_COMPONENTS,
							pIcd->ui64DefRowId, FALSE)))
		{
			
			// The row may have been deleted in the loop above where key
			// components are defined.  It is possible to define both a
			// key component and a data component in a single row in the
			// database.
			
			if (rc == NE_SFLM_ROW_NOT_FOUND)
			{
				rc = NE_SFLM_OK;
			}
			else
			{
				goto Exit;
			}
		}
	}
	
	// Initiate the background process to delete the b-tree.

	if (RC_BAD( rc = m_pDatabase->lFileDelete( this, &pIndex->lfInfo,
												pIndex->uiFlags & IXD_ABS_POS
												? TRUE
												: FALSE,
												pIndex->uiNumDataComponents
												? TRUE
												: FALSE)))
	{
		goto Exit;
	}
	
	// If the index was being built in the background, stop the indexing
	// thread.  NOTE: No indexing thread will be going if we are replaying
	// the RFL.
	
	if ((pIndex->uiFlags & IXD_OFFLINE) && !(pIndex->uiFlags & IXD_SUSPENDED) &&
		 !(m_uiFlags & FDB_REPLAYING_RFL))
	{
		if (RC_BAD( rc = addToStopList( uiIndexNum)))
		{
			goto Exit;
		}
	}
	
	// Remove any fixups for this index.
	
	pIxdFixup = m_pIxdFixups;
	pPrevIxdFixup = NULL;
	while (pIxdFixup)
	{
		if (pIndex->uiIndexNum == pIxdFixup->uiIndexNum)
		{
			IXD_FIXUP *	pDeleteIxdFixup = pIxdFixup;
			
			pIxdFixup = pIxdFixup->pNext;
			f_free( &pDeleteIxdFixup);
			if (pPrevIxdFixup)
			{
				pPrevIxdFixup->pNext = pIxdFixup;
			}
			else
			{
				m_pIxdFixups = pIxdFixup;
			}
		}
		else
		{
			pPrevIxdFixup = pIxdFixup;
			pIxdFixup = pIxdFixup->pNext;
		}
	}
	
	// Remove the index name from the index name table.
	
	m_pDict->m_pIndexNames->removeName( pIndex->pszIndexName);

	// Remove the index from the in-memory dictionary.  This is done
	// simply by zeroing out the entire structure.
	
	f_memset( pIndex, sizeof( F_INDEX), 0);
	
	// Log the operation
	
	if (RC_BAD( rc = m_pDatabase->m_pRfl->logDropIndex( this, uiIndexNum)))
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
RCODE SQLStatement::processDropIndex( void)
{
	RCODE						rc = NE_SFLM_OK;
	FLMBOOL					bStartedTrans = FALSE;
	char						szIndexName [MAX_SQL_NAME_LEN + 1];
	FLMUINT					uiIndexNameLen;
	char						szTableName [MAX_SQL_NAME_LEN + 1];
	FLMUINT					uiTableNameLen;
	F_TABLE *				pTable;
	F_INDEX *				pIndex;
	
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

	// Get the index name - index name must exist
	
	if (RC_BAD( rc = getIndexName( TRUE, NULL, szIndexName, sizeof( szIndexName),
							&uiIndexNameLen, &pIndex)))
	{
		goto Exit;
	}
	
	// Cannot drop system indexes
	
	if (pIndex->uiFlags & IXD_SYSTEM)
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset - 1,
				SQL_ERR_CANNOT_DROP_SYSTEM_INDEX,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_SFLM_INVALID_SQL);
		goto Exit;
	}
	
	// See if the keyword "ON" is present
	
	if (RC_BAD( rc = haveToken( "on", TRUE)))
	{
		if (rc == NE_SFLM_NOT_FOUND || rc == NE_SFLM_EOF_HIT)
		{
			rc = NE_SFLM_OK;
		}
		else
		{
			goto Exit;
		}
	}
	else
	{

		// Get the table name - must exist, and must be the same as
		// the table the index is associated with.
		
		if (RC_BAD( rc = getTableName( TRUE, szTableName, sizeof( szTableName),
									&uiTableNameLen, &pTable)))
		{
			goto Exit;
		}
		
		if (pTable->uiTableNum != pIndex->uiTableNum)
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset - 1,
					SQL_ERR_TABLE_NOT_FOR_INDEX,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
			goto Exit;
		}
	}
	
	// Drop the index.

	if (RC_BAD( rc = m_pDb->dropIndex( pIndex->uiIndexNum)))
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

