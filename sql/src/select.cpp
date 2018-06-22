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

static const char * gv_selectExprTerminators [3] =
{
	"from",
	",",
	NULL
};

static const char * gv_selectWhereTerminators [3] =
{
	"order",
	NULL
};

//------------------------------------------------------------------------------
// Desc:	Process the expressions that are going to be retrieved in a SELECT
//			statement.
//------------------------------------------------------------------------------
RCODE SQLStatement::parseSelectExpressions(
	SELECT_EXPR **	ppFirstSelectExpr,
	SELECT_EXPR **	ppLastSelectExpr)
{
	RCODE				rc = NE_SFLM_OK;
	SELECT_EXPR *	pSelectExpr;
	SQLQuery *		pSqlQuery = NULL;
	const char *	pszTerminator;
	
	// See if they specified "*"
	
	if (RC_OK( rc = haveToken( "*", FALSE)))
	{
		
		// Better be followed by "from" - need to consume the "from" token
		// because the caller expects that it will have been consumed.
		
		rc = haveToken( "from", FALSE, SQL_ERR_EXPECTING_FROM);
		goto Exit;
	}
	else if (rc == NE_SFLM_NOT_FOUND)
	{
		rc = NE_SFLM_OK;
	}
	else
	{
		goto Exit;
	}
	
	for (;;)
	{
		
		// Allocate a SELECT_EXPR structure
		
		if (RC_BAD( rc = m_tmpPool.poolAlloc( sizeof( SELECT_EXPR),
												(void **)&pSelectExpr)))
		{
			goto Exit;
		}
		
			
		// Allocate an SQLQuery object, have the pColumnSet structure
		// point to it, and link the pColumnSet structure into the linked
		// list.
		
		if ((pSqlQuery = f_new SQLQuery) == NULL)
		{
			rc = RC_SET( NE_SFLM_MEM);
			goto Exit;
		}
		pSelectExpr->pSqlQuery = pSqlQuery;
		pSelectExpr->pNext = NULL;
		if (*ppLastSelectExpr)
		{
			(*ppLastSelectExpr)->pNext = pSelectExpr;
		}
		else
		{
			*ppFirstSelectExpr = pSelectExpr;
		}
		*ppLastSelectExpr = pSelectExpr;
		
		// Now parse the criteria
		
		if (RC_BAD( rc = parseCriteria( NULL,
									&gv_selectExprTerminators [0], FALSE,
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
		
		// Terminator should never be NULL, because we passed a FALSE into
		// parseCriteria for the bEofOK flag.
		
		flmAssert( pszTerminator);
		
		// Terminator will have been either be a comma or the FROM keyword.
		
		if (f_stricmp( pszTerminator, "from") == 0)
		{
			break;
		}
	}
	
Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Process the SELECT statement.  The "SELECT" keyword has already been
//			parsed.
//------------------------------------------------------------------------------
RCODE SQLStatement::processSelect( void)
{
	RCODE					rc = NE_SFLM_OK;
	FLMBOOL				bStartedTrans = FALSE;
	char					szToken [MAX_SQL_TOKEN_SIZE + 1];
	FLMUINT				uiTokenLineOffset;
	FLMUINT				uiTokenLen;
	FLMBOOL				bDistinct;
	F_TABLE *			pTable;
	F_INDEX *			pIndex;
	SELECT_EXPR *		pFirstSelectExpr = NULL;
	SELECT_EXPR *		pLastSelectExpr = NULL;
	SELECT_EXPR *		pSelectExpr;
	SQLQuery				sqlQuery;
	TABLE_ITEM			tableList [MAX_SELECT_TABLES + 1];
	FLMUINT				uiNumTables = 0;
	FLMUINT				uiLoop;
	const char *		pszTerminator;
	char					szName [MAX_SQL_NAME_LEN + 1];
	FLMUINT				uiNameLen;
	char					szColumnName [MAX_SQL_NAME_LEN + 1];
	FLMUINT				uiColumnNameLen;
	FLMUINT				uiTableNum;
	FLMUINT				uiColumnNum;
	FLMBOOL				bDescending;
	SQLParseError		eParseError;
	FLMBOOL				bHaveWhere = FALSE;
	FLMBOOL				bHaveOrderBy = TRUE;
	FLMBOOL				bDoneParsingOrderBy;

	// Make sure we have at least a read transaction going.
	
	if (RC_BAD( rc = m_pDb->checkTransaction( SFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// SYNTAX: SELECT [ALL | DISTINCT] {* | expression [,expression]...}
	// FROM table_name [table_alias] [, table_name [table_alias]]...
	// [WHERE <search_criteria>]
	// [ORDER BY column [ASC | DESC] [, column [ASC | DESC]]...]

	// See if "ALL" or "DISTINCT" were specified
	
	bDistinct = FALSE;

	if (RC_BAD( rc = getToken( szToken, sizeof( szToken), FALSE,
									&uiTokenLineOffset, NULL)))
	{
		goto Exit;
	}
	if (f_stricmp( szToken, "all") == 0)
	{
	}
	else if (f_stricmp( szToken, "distinct") == 0)
	{
		bDistinct = TRUE;
	}
	else
	{
		
		// Push the token back into the stream so it will be read again.
		
		m_uiCurrLineOffset = uiTokenLineOffset;
	}
	
	// Parse the expressions that are to be selected.
	
	if (RC_BAD( rc = parseSelectExpressions( &pFirstSelectExpr, &pLastSelectExpr)))
	{
		goto Exit;
	}
	
	// Get the table names and their aliases.
	
	for (;;)
	{
		if (RC_BAD( rc = getTableName( TRUE, szName, sizeof( szName),
									&uiNameLen, &pTable)))
		{
			goto Exit;
		}
		
		// See if we have already defined this table - cannot define it twice.
		
		for (uiLoop = 0; uiLoop < uiNumTables; uiLoop++)
		{
			if (tableList [uiLoop].uiTableNum == pTable->uiTableNum)
			{
				setErrInfo( m_uiCurrLineNum,
						uiTokenLineOffset,
						SQL_ERR_DUPLICATE_TABLE_NAME,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
		}
		
		// Must not overflow the name table.
		
		if (uiNumTables == MAX_SELECT_TABLES)
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset - 1,
					SQL_ERR_TOO_MANY_TABLES,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
			goto Exit;
		}
		
		// Add the table name to the list
		
		tableList [uiNumTables].bScan = FALSE;
		tableList [uiNumTables].uiIndexNum = 0;
		tableList [uiNumTables].uiTableNum = pTable->uiTableNum;
		tableList [uiNumTables].pszTableAlias = NULL;
		uiNumTables++;
		
		// See if a table alias was specified, or if we are at a comma or
		// an "ORDER BY" or EOF.
		
		if (RC_BAD( rc = getToken( szToken, sizeof( szToken), TRUE,
										&uiTokenLineOffset, &uiTokenLen)))
		{
			if (rc == NE_SFLM_EOF_HIT)
			{
				rc = NE_SFLM_OK;
				break;
			}
			goto Exit;
		}
		if (f_stricmp( szToken, "where") == 0)
		{
			bHaveWhere = TRUE;
			break;
		}
		else if (f_stricmp( szToken, "order") == 0)
		{
			bHaveOrderBy = TRUE;
			break;
		}
		else if (f_stricmp( szToken, "using") == 0)
		{
Get_Index:
			if (RC_BAD( rc = getToken( szToken, sizeof( szToken), FALSE,
											&uiTokenLineOffset, &uiTokenLen)))
			{
				goto Exit;
			}
			if (f_stricmp( szToken, "noindex") == 0)
			{
				tableList [uiNumTables - 1].bScan = TRUE;
				tableList [uiNumTables - 1].uiIndexNum = 0;
			}
			else if (f_stricmp( szToken, "index") == 0)
			{
				
				// Get the index name.
				
				if (RC_BAD( rc = getIndexName( TRUE, pTable, szToken, sizeof( szToken),
											&uiTokenLen, &pIndex)))
				{
					goto Exit;
				}
				tableList [uiNumTables - 1].bScan = FALSE;
				tableList [uiNumTables - 1].uiIndexNum = pIndex->uiIndexNum;
			}
			else
			{
				setErrInfo( m_uiCurrLineNum,
						uiTokenLineOffset,
						SQL_ERR_EXPECTING_INDEX_OR_NOINDEX,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
		}
		else if ((szToken [0] >= 'a' && szToken [0] <= 'z') ||
					(szToken [0] >= 'A' && szToken [0] <= 'Z'))
		{
			
			// If token is the keyword "as", it must be followed by the
			// alias name.
			
			if (f_stricmp( szToken, "as") == 0)
			{
				if (RC_BAD( rc = getName( szToken, sizeof( szToken),
												&uiTokenLen, &uiTokenLineOffset)))
				{
					goto Exit;
				}
			}
			
			// See if this alias name has been used in the table list already.
			
			for (uiLoop = 0; uiLoop < uiNumTables; uiLoop++)
			{
				if (f_stricmp( tableList [uiLoop].pszTableAlias, szToken) == 0)
				{
					setErrInfo( m_uiCurrLineNum,
							uiTokenLineOffset,
							SQL_ERR_DUPLICATE_ALIAS_NAME,
							m_uiCurrLineFilePos,
							m_uiCurrLineBytes);
					rc = RC_SET( NE_SFLM_INVALID_SQL);
					goto Exit;
				}
			}
			
			// Set the alias name for our current table.  NOTE: Our current
			// table is at element uiNumTables-1, because we have already
			// incremented uiNumTables.
			
			if (RC_BAD( rc = m_tmpPool.poolAlloc( uiTokenLen + 1,
								(void **)&tableList [uiNumTables - 1].pszTableAlias)))
			{
				goto Exit;
			}
			f_memcpy( &tableList [uiNumTables - 1].pszTableAlias, szToken,
							uiTokenLen + 1);
			if (RC_BAD( rc = getToken( szToken, sizeof( szToken), TRUE,
											&uiTokenLineOffset, &uiTokenLen)))
			{
				if (rc == NE_SFLM_EOF_HIT)
				{
					rc = NE_SFLM_OK;
					break;
				}
				goto Exit;
			}
			if (f_stricmp( szToken, "where") == 0)
			{
				bHaveWhere = TRUE;
				break;
			}
			else if (f_stricmp( szToken, "order") == 0)
			{
				bHaveOrderBy = TRUE;
				break;
			}
			else if (f_stricmp( szToken, "using") == 0)
			{
				goto Get_Index;
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
	
	// parseSelectExpressions will have already parsed the "FROM" keyword.
	// We must now parse table names until we hit a "WHERE" or "ORDER BY" or
	// EOF.
	
	// Resolve the column names that were parsed in the select expressions.

	tableList [uiNumTables].uiTableNum = 0;
	tableList [uiNumTables].pszTableAlias = NULL;
	for (pSelectExpr = pFirstSelectExpr; pSelectExpr; pSelectExpr = pSelectExpr->pNext)
	{
		if (RC_BAD( rc = pSelectExpr->pSqlQuery->resolveColumnNames( &tableList [0])))
		{
			goto Exit;
		}
	}
	
	// Add each of the tables in the table list.
	
	for (uiLoop = 0; uiLoop < uiNumTables; uiLoop++)
	{
		if (RC_BAD( rc = sqlQuery.addTable( tableList [uiLoop].uiTableNum, NULL)))
		{
			goto Exit;
		}
		if (tableList [uiLoop].bScan)
		{
			if (RC_BAD( rc = sqlQuery.setIndex( tableList [uiLoop].uiTableNum, 0)))
			{
				goto Exit;
			}
		}
		else if (tableList [uiLoop].uiIndexNum)
		{
			if (RC_BAD( rc = sqlQuery.setIndex( tableList [uiLoop].uiTableNum,
										tableList [uiLoop].uiIndexNum)))
			{
				goto Exit;
			}
		}
	}
	
	// If there is a WHERE clause, parse it.
	
	if (bHaveWhere)
	{
		if (RC_BAD( rc = parseCriteria( &tableList [0],
									&gv_selectWhereTerminators [0], TRUE,
									&pszTerminator, &sqlQuery)))
		{
			goto Exit;
		}
		if (pszTerminator && f_stricmp( pszTerminator, "order") == 0)
		{
			bHaveOrderBy = TRUE;
		}
	}
	
	// If there is an ORDER BY clause, parse it.
	
	if (bHaveOrderBy)
	{
		
		// Make sure we have the "BY" keyword
		
		
		if (RC_BAD( rc = haveToken( "by", FALSE, SQL_ERR_EXPECTING_BY)))
		{
			goto Exit;
		}
		
		for (;;)
		{
			char *	pszTableAlias;
			char *	pszColumnName;
			
			// Get either a table name or column name.
			
			if (RC_BAD( rc = getName( szName, sizeof( szName),
									&uiNameLen, &uiTokenLineOffset)))
			{
				goto Exit;
			}
//			JMC - FIXME: commented out due to missing functionality in flaimtk.h
//			if (uiNumOrderByColumns == MAX_ORDER_BY_COLUMNS)
//			{
//				setErrInfo( m_uiCurrLineNum,
//						uiTokenLineOffset,
//						SQL_ERR_TOO_MANY_ORDER_BY_COLUMNS,
//						m_uiCurrLineFilePos,
//						m_uiCurrLineBytes);
//				rc = RC_SET( NE_SFLM_INVALID_SQL);
//				goto Exit;
//			}
			
			// See if we have a period after the name.
			
			if (RC_BAD( rc = haveToken( ".", TRUE)))
			{
				if (rc != NE_SFLM_NOT_FOUND && rc != NE_SFLM_EOF_HIT)
				{
					goto Exit;
				}
				pszTableAlias = NULL;
				pszColumnName = &szName [0];
			}
			else
			{
				if (RC_BAD( rc = getName( szColumnName, sizeof( szColumnName),
										&uiColumnNameLen, &uiTokenLineOffset)))
				{
					goto Exit;
				}
				pszTableAlias = &szName [0];
				pszColumnName = &szColumnName [0];
			}
			if (RC_BAD( rc = resolveColumnName( m_pDb, &tableList [0], pszTableAlias,
											pszColumnName, &uiTableNum, &uiColumnNum,
											&eParseError)))
			{
				if (eParseError != SQL_NO_ERROR)
				{
					setErrInfo( m_uiCurrLineNum,
							uiTokenLineOffset,
							eParseError,
							m_uiCurrLineFilePos,
							m_uiCurrLineBytes);
				}
				goto Exit;
			}
			
			// See if the next token is the keyword "ASC" or "DESC" or comma.
			
			bDoneParsingOrderBy = FALSE;
			bDescending = FALSE;
			if (RC_BAD( rc = getToken( szToken, sizeof( szToken), TRUE,
										&uiTokenLineOffset, &uiTokenLen)))
			{
				if (rc == NE_SFLM_EOF_HIT)
				{
					rc = NE_SFLM_OK;
					bDoneParsingOrderBy = TRUE;
				}
				else
				{
					goto Exit;
				}
			}
			else if (f_stricmp( szToken, "asc") == 0 ||
						f_stricmp( szToken, "desc") == 0)
			{
				if (szToken [0] == 'D' || szToken [0] == 'd')
				{
					bDescending = TRUE;
				}
				if (RC_BAD( rc = getToken( szToken, sizeof( szToken), TRUE,
											&uiTokenLineOffset, &uiTokenLen)))
				{
					if (rc == NE_SFLM_EOF_HIT)
					{
						rc = NE_SFLM_OK;
						bDoneParsingOrderBy = TRUE;
					}
					else
					{
						goto Exit;
					}
				}
			}
			
			// Add an ORDER BY to the query
			
			if (RC_BAD( rc = sqlQuery.orderBy( uiTableNum, uiColumnNum, bDescending)))
			{
				goto Exit;
			}
			if (bDoneParsingOrderBy)
			{
				break;
			}
			
			// Token better be a comma
			
			if (f_stricmp( szToken, ",") != 0)
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

	while (pFirstSelectExpr)
	{
		if (pFirstSelectExpr->pSqlQuery)
		{
			pFirstSelectExpr->pSqlQuery->Release();
			pFirstSelectExpr->pSqlQuery = NULL;
		}
		pFirstSelectExpr = pFirstSelectExpr->pNext;
	}

	if (bStartedTrans)
	{
		m_pDb->transAbort();
	}

	return( rc);
}

