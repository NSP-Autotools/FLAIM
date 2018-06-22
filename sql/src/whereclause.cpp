//-------------------------------------------------------------------------
// Desc:	Parse SQL
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

// Local function prototypes

FSTATIC void sqlUnlinkFromParent(
	SQL_NODE *	pSQLNode);

FSTATIC void sqlLinkLastChild(
	SQL_NODE *	pParent,
	SQL_NODE *	pChild);
	
FSTATIC FLMBOOL isTerminatingToken(
	const char *	pszToken,
	const char **	ppszTerminatingTokens,
	const char **	ppszTerminator);
	
static FLMUINT uiSQLOpPrecedenceTable[ SQL_NEG_OP - SQL_AND_OP + 1] =
{
	2,		// SQL_AND_OP
	1,		// SQL_OR_OP
	10,	// SQL_NOT_OP
	6,		// SQL_EQ_OP
	6,		// SQL_NE_OP
	6,		// SQL_APPROX_EQ_OP
	7,		// SQL_LT_OP
	7,		// SQL_LE_OP
	7,		// SQL_GT_OP
	7,		// SQL_GE_OP
	5,		// SQL_BITAND_OP
	3,		// SQL_BITOR_OP
	4,		// SQL_BITXOR_OP
	9,		// SQL_MULT_OP
	9,		// SQL_DIV_OP
	9,		// SQL_MOD_OP
	8,		// SQL_PLUS_OP
	8,		// SQL_MINUS_OP
	10		// SQL_NEG_OP
};

FINLINE FLMUINT getSQLOpPrecedence(
	eSQLQueryOperators	eOperator)
{
	return( uiSQLOpPrecedenceTable [eOperator - SQL_AND_OP]);
}

//-------------------------------------------------------------------------
// Desc:	Constructor
//-------------------------------------------------------------------------
SQLQuery::SQLQuery()
{
	m_uiLanguage = FLM_US_LANG;
	m_pool.poolInit( 1024);
	m_pFirstSubQuery = NULL;
	m_pLastSubQuery = NULL;
	m_pFirstSQLTable = NULL;
	m_pLastSQLTable = NULL;
	m_pFirstOrderBy = NULL;
	m_pLastOrderBy = NULL;
	m_bResolveNames = FALSE;
	m_bOptimized = FALSE;
	m_bEmpty = FALSE;
	m_pQuery = NULL;
	m_pDatabase = NULL;
	m_pDb = NULL;
	m_pNext = NULL;		
	m_pPrev = NULL;		
}

//-------------------------------------------------------------------------
// Desc:	freel all of the SQL_KEY structures associated with an SQL_INDEX.
//-------------------------------------------------------------------------
void freeIndexKeys(
	SQL_INDEX *	pSQLIndex)
{
	SQL_KEY *	pSQLKey;
	
	for (pSQLKey = pSQLIndex->pFirstSQLKey; pSQLKey; pSQLKey = pSQLKey->pNext)
	{
		if (pSQLKey->pFSIndexCursor)
		{
			pSQLKey->pFSIndexCursor->Release();
			pSQLKey->pFSIndexCursor = NULL;
		}
	}
	pSQLIndex->pFirstSQLKey = NULL;
	pSQLIndex->pLastSQLKey = NULL;
}

//-------------------------------------------------------------------------
// Desc:	Free all of the SQL_INDEX structures associated with an SQL_TABLE.
//-------------------------------------------------------------------------
void freeTableIndexes(
	SQL_TABLE *	pSQLTable)
{
	SQL_INDEX *	pSQLIndex;
	
	for (pSQLIndex = pSQLTable->pFirstSQLIndex; pSQLIndex; pSQLIndex = pSQLIndex->pNext)
	{
		freeIndexKeys( pSQLIndex);
	}
	pSQLTable->pFirstSQLIndex = NULL;
	pSQLTable->pLastSQLIndex = NULL;
}

//-------------------------------------------------------------------------
// Desc:	Destructor
//-------------------------------------------------------------------------
SQLQuery::~SQLQuery()
{
	SQL_TABLE *	pSQLTable;
	
	// Free all of the table and index cursors.
	
	for (pSQLTable = m_pFirstSQLTable; pSQLTable; pSQLTable = pSQLTable->pNext)
	{
		if (pSQLTable->pFSTableCursor)
		{
			pSQLTable->pFSTableCursor->Release();
			pSQLTable->pFSTableCursor = NULL;
		}
		freeTableIndexes( pSQLTable);
	}
	
	// Free all of the memory allocated from the memory pool.
	
	m_pool.poolFree();
	
	// Unlink the query from the database it is associated with.
	
	if (m_pDatabase)
	{
		m_pDatabase->lockMutex();

		// Unlink the query from the list off of the F_Database object.

		if (m_pPrev)
		{
			m_pPrev->m_pNext = m_pNext;
		}
		else
		{
			m_pDatabase->m_pFirstSQLQuery = m_pNext;
		}
		if (m_pNext)
		{
			m_pNext->m_pPrev = m_pPrev;
		}
		else
		{
			m_pDatabase->m_pLastSQLQuery = m_pPrev;
		}
		m_pDatabase->unlockMutex();
	}
}

//-------------------------------------------------------------------------
// Desc:	Unlinks a node from its parent and siblings.  This routine assumes
//			that the test has already been made that the node has a parent.
//-------------------------------------------------------------------------
FSTATIC void sqlUnlinkFromParent(
	SQL_NODE *	pSQLNode)
{
	flmAssert( pSQLNode->pParent);
	if (pSQLNode->pPrevSib)
	{
		pSQLNode->pPrevSib->pNextSib = pSQLNode->pNextSib;
	}
	else
	{
		pSQLNode->pParent->pFirstChild = pSQLNode->pNextSib;
	}
	if (pSQLNode->pNextSib)
	{
		pSQLNode->pNextSib->pPrevSib = pSQLNode->pPrevSib;
	}
	else
	{
		pSQLNode->pParent->pLastChild = pSQLNode->pPrevSib;
	}

	pSQLNode->pParent = NULL;
	pSQLNode->pPrevSib = NULL;
	pSQLNode->pNextSib = NULL;
}

//-------------------------------------------------------------------------
// Desc:	Links one SQL_NODE as the last child of another.  Will unlink the
//			child node from any parent it may be linked to.
//-------------------------------------------------------------------------
FSTATIC void sqlLinkLastChild(
	SQL_NODE *	pParent,
	SQL_NODE *	pChild
	)
{

	// If necessary, unlink the child from parent and siblings

	if (pChild->pParent)
	{
		sqlUnlinkFromParent( pChild);
	}

	// Link child as the last child to parent

	pChild->pParent = pParent;
	pChild->pNextSib = NULL;
	if ((pChild->pPrevSib = pParent->pLastChild) != NULL)
	{
		pChild->pPrevSib->pNextSib = pChild;
	}
	else
	{
		pParent->pFirstChild = pChild;
	}
	pParent->pLastChild = pChild;
}

//-------------------------------------------------------------------------
// Desc:	Allocate a value node.
//-------------------------------------------------------------------------
RCODE SQLQuery::allocValueNode(
	FLMUINT			uiValLen,
	eSQLValTypes	eValType,
	SQL_NODE **		ppSQLNode
	)
{
	RCODE			rc = NE_SFLM_OK;
	SQL_NODE *	pSQLNode;

	if (!expectingOperand())
	{
		rc = RC_SET( NE_SFLM_Q_UNEXPECTED_VALUE);
		goto Exit;
	}

	if (RC_BAD( rc = m_pool.poolCalloc( sizeof( SQL_NODE),
									(void **)ppSQLNode)))
	{
		goto Exit;
	}
	pSQLNode = *ppSQLNode;
	pSQLNode->eNodeType = SQL_VALUE_NODE;
	pSQLNode->currVal.eValType = eValType;
	pSQLNode->currVal.uiFlags = SQL_VAL_IS_CONSTANT;

	// For string and binary data, allocate a buffer.

	if (uiValLen)
	{
		if (eValType == SQL_UTF8_VAL)
		{
			if (RC_BAD( rc = m_pool.poolAlloc( uiValLen,
													(void **)&pSQLNode->currVal.val.str.pszStr)))
			{
				goto Exit;
			}
			pSQLNode->currVal.val.str.uiByteLen = uiValLen;
		}
		else if (eValType == SQL_BINARY_VAL)
		{
			if (RC_BAD( rc = m_pool.poolAlloc( uiValLen,
													(void **)&pSQLNode->currVal.val.bin.pucValue)))
			{
				goto Exit;
			}
			pSQLNode->currVal.val.bin.uiByteLen = uiValLen;
		}
	}
	
	if (m_pQuery)
	{
		sqlLinkLastChild( m_pCurOperatorNode, pSQLNode);
	}
	else
	{
		m_pQuery = pSQLNode;
	}
	m_bExpectingOperator = TRUE;
	m_pLastNode = pSQLNode;

Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Add an operator to the query expression
//-------------------------------------------------------------------------
RCODE SQLQuery::addOperator(
	eSQLQueryOperators	eOperator,
	FLMUINT					uiCompareRules)
{
	RCODE				rc = NE_SFLM_OK;
	SQL_NODE *		pSQLNode;
	SQL_NODE *		pParentNode;

	if (eOperator == SQL_MINUS_OP && expectingOperand())
	{
		eOperator = SQL_NEG_OP;
	}

	switch (eOperator)
	{
		case SQL_LPAREN_OP:

			// If the operator is a left paren, increment the nesting level

			if (expectingOperator())
			{
				rc = RC_SET( NE_SFLM_Q_UNEXPECTED_LPAREN);
				goto Exit;
			}
			m_uiNestLevel++;
			goto Exit;

		case SQL_RPAREN_OP:
			if (expectingOperand())
			{
				rc = RC_SET( NE_SFLM_Q_UNEXPECTED_RPAREN);
				goto Exit;
			}
			if (!m_uiNestLevel)
			{
				rc = RC_SET( NE_SFLM_Q_UNMATCHED_RPAREN);
				goto Exit;
			}
			m_uiNestLevel--;

			goto Exit;

		case SQL_NEG_OP:
		case SQL_NOT_OP:
			if (expectingOperator())
			{
				rc = RC_SET( NE_SFLM_Q_EXPECTING_OPERATOR);
				goto Exit;
			}
			break;

		default:

			if (expectingOperand())
			{
				rc = RC_SET( NE_SFLM_Q_EXPECTING_OPERAND);
				goto Exit;
			}
			if (!isLegalSQLOperator( eOperator))
			{
				rc = RC_SET( NE_SFLM_Q_ILLEGAL_OPERATOR);
				goto Exit;
			}

			break;
	}

	// Cannot set both FLM_COMP_COMPRESS_WHITESPACE and FLM_COMP_NO_WHITESPACE
	// in comparison rules.  Also, cannot set FLM_COMP_IGNORE_LEADING_SPACE or
	// FLM_COMP_IGNORE_TRAILING_SPACE with FLM_COMP_NO_WHITESPACE.

	if ((uiCompareRules & FLM_COMP_NO_WHITESPACE) &&
		 (uiCompareRules & (FLM_COMP_COMPRESS_WHITESPACE |
								  FLM_COMP_IGNORE_LEADING_SPACE |
								  FLM_COMP_IGNORE_TRAILING_SPACE)))
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_Q_ILLEGAL_COMPARE_RULES);
		goto Exit;
	}

	// Make a QNODE and find a place for it in the query tree

	if (RC_BAD( rc = m_pool.poolCalloc( sizeof( SQL_NODE),
											(void **)&pSQLNode)))
	{
		goto Exit;
	}
	pSQLNode->eNodeType = SQL_OPERATOR_NODE;
	pSQLNode->nd.op.eOperator = eOperator;
	pSQLNode->nd.op.uiCompareRules = uiCompareRules;
	pSQLNode->uiNestLevel = m_uiNestLevel;

	// Go up the stack until an operator whose nest level or precedence is <
	// this one's is encountered, then link this one in as the last child

	pParentNode = m_pCurOperatorNode;
	while (pParentNode &&
			 (pParentNode->uiNestLevel > pSQLNode->uiNestLevel ||
			  (pParentNode->uiNestLevel == pSQLNode->uiNestLevel &&
			   getSQLOpPrecedence( pParentNode->nd.op.eOperator) >=
				getSQLOpPrecedence( eOperator))))
	{
		pParentNode = pParentNode->pParent;
	}
	if (!pParentNode)
	{
		if (m_pQuery)
		{
			sqlLinkLastChild( pSQLNode, m_pQuery);
		}
		m_pQuery = pSQLNode;
	}
	else if (eOperator == SQL_NOT_OP || eOperator == SQL_NEG_OP)
	{

		// Need to treat NOT and NEG as if they were operands.

		// Parent better be an operator.

		flmAssert( pParentNode->eNodeType == SQL_OPERATOR_NODE);

#ifdef FLM_DEBUG
		if (pParentNode->nd.op.eOperator == SQL_NEG_OP ||
			 pParentNode->nd.op.eOperator == SQL_NOT_OP)
		{

			// Must have no children.

			flmAssert( pParentNode->pFirstChild == NULL);
		}
		else
		{

			// Must only have one or zero children.

			flmAssert( pParentNode->pFirstChild == pParentNode->pLastChild);
		}
#endif

		sqlLinkLastChild( pParentNode, pSQLNode);
		flmAssert( !m_bExpectingOperator);
	}
	else
	{

		// Parent better be an operator.

		flmAssert( pParentNode->eNodeType == SQL_OPERATOR_NODE);

		// Unlink last child of parent node and replace with this
		// new node.  The parent node better already have the correct
		// number of children, or we are not parsing correctly.

		flmAssert( pParentNode->pFirstChild);
		if (pParentNode->nd.op.eOperator == SQL_NEG_OP ||
			 pParentNode->nd.op.eOperator == SQL_NOT_OP)
		{

			// Better only be one child.

			flmAssert( !pParentNode->pFirstChild->pNextSib);

			sqlLinkLastChild( pSQLNode, pParentNode->pFirstChild);
		}
		else
		{

			// Better only be two child nodes

			flmAssert( pParentNode->pFirstChild->pNextSib ==
						  pParentNode->pLastChild);
			sqlLinkLastChild( pSQLNode, pParentNode->pLastChild);
		}
		sqlLinkLastChild( pParentNode, pSQLNode);
	}

	m_pCurOperatorNode = pSQLNode;
	m_bExpectingOperator = FALSE;
	m_pLastNode = pSQLNode;

Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Allocate a node for an operand.
//-------------------------------------------------------------------------
RCODE SQLQuery::allocOperandNode(
	eSQLNodeTypes	eNodeType,
	SQL_NODE **		ppSQLNode)
{
	RCODE			rc = NE_SFLM_OK;
	SQL_NODE *	pSQLNode;
	
	if (RC_BAD( rc = m_pool.poolCalloc( sizeof( SQL_NODE),
									(void **)&pSQLNode)))
	{
		goto Exit;
	}
	pSQLNode->eNodeType = eNodeType;

	if (m_pQuery)
	{
		sqlLinkLastChild( m_pCurOperatorNode, pSQLNode);
	}
	else
	{
		m_pQuery = pSQLNode;
	}
	m_bExpectingOperator = TRUE;
	m_pLastNode = *ppSQLNode = pSQLNode;
	
Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Add a table to the query.
//-------------------------------------------------------------------------
RCODE SQLQuery::addTable(
	FLMUINT			uiTableNum,
	SQL_TABLE **	ppSQLTable)
{
	RCODE				rc = NE_SFLM_OK;
	SQL_TABLE *		pSQLTable;

	// Add or find the table structure for the node.
	
	pSQLTable = m_pFirstSQLTable;
	while (pSQLTable && pSQLTable->uiTableNum != uiTableNum)
	{
		pSQLTable = pSQLTable->pNext;
	}
	if (!pSQLTable)
	{
		if (RC_BAD( rc = m_pool.poolCalloc( sizeof( SQL_TABLE),
										(void **)&pSQLTable)))
		{
			goto Exit;
		}
		pSQLTable->uiTableNum = uiTableNum;
		if ((pSQLTable->pPrev = m_pLastSQLTable) != NULL)
		{
			m_pLastSQLTable->pNext = pSQLTable;
		}
		else
		{
			m_pFirstSQLTable = pSQLTable;
		}
		m_pLastSQLTable = pSQLTable;
	}
	
	// These should have been set by the poolCalloc.
	
	// pSQLTable->bScan = FALSE;
	// pSQLTable->uiIndexNum = 0;
	
	if (ppSQLTable)
	{
		*ppSQLTable = pSQLTable;
	}

Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Add a column name.
//-------------------------------------------------------------------------
RCODE SQLQuery::addColumn(
	FLMUINT	uiTableNum,
	FLMUINT	uiColumnNum)
{
	RCODE				rc = NE_SFLM_OK;
	SQL_NODE *		pSQLNode;
	SQL_TABLE *		pSQLTable;

	// Must be expecting an operand

	if (!expectingOperand())
	{
		rc = RC_SET( NE_SFLM_Q_UNEXPECTED_COLUMN);
		goto Exit;
	}

	if (RC_BAD( rc = addTable( uiTableNum, &pSQLTable)))
	{
		goto Exit;
	}

	// Allocate a column node

	if (RC_BAD( rc = allocOperandNode( SQL_COLUMN_NODE, &pSQLNode)))
	{
		goto Exit;
	}
	pSQLNode->nd.column.pSQLTable = pSQLTable;
	pSQLNode->nd.column.uiColumnNum = uiColumnNum;

Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Determine which table/column a table alias name and column name
//			refer to.
//-------------------------------------------------------------------------
RCODE resolveColumnName(
	F_Db *				pDb,
	TABLE_ITEM *		pTableList,
	const char *		pszTableAlias,
	const char *		pszColumnName,
	FLMUINT *			puiTableNum,
	FLMUINT *			puiColumnNum,
	SQLParseError *	peParseError)
{
	RCODE				rc = NE_SFLM_OK;
	F_COLUMN *		pColumn;
	F_TABLE *		pTable;
	F_COLUMN *		pFoundColumn;
	F_TABLE *		pFoundTable;
	TABLE_ITEM *	pTableItem;
	
	if (pszTableAlias)
	{
		for (pTableItem = pTableList; pTableItem->uiTableNum; pTableItem++)
		{
			if (f_stricmp( pszTableAlias, pTableItem->pszTableAlias) == 0)
			{
				pTable = pDb->getDict()->getTable( pTableItem->uiTableNum);
				if ((pColumn = pDb->getDict()->findColumn( pTable,
											pszColumnName)) == NULL)
				{
					if (peParseError)
					{
						*peParseError = SQL_ERR_INVALID_COLUMN_NAME;
					}
					rc = RC_SET( NE_SFLM_Q_INVALID_COLUMN_NAME);
					goto Exit;
				}
				*puiTableNum = pTable->uiTableNum;
				*puiColumnNum = pColumn->uiColumnNum;
				goto Exit;
			}
		}
		
		// If we get this far, we didn't find the table.
		
		if (peParseError)
		{
			*peParseError = SQL_ERR_UNDEFINED_TABLE_NAME;
		}
		rc = RC_SET( NE_SFLM_Q_UNDEFINED_TABLE_FOR_COLUMN);
		goto Exit;
	}
	else
	{
		pFoundColumn = NULL;
		pFoundTable = NULL;
		for (pTableItem = pTableList; pTableItem->uiTableNum; pTableItem++)
		{
			pTable = pDb->getDict()->getTable( pTableItem->uiTableNum);
			if ((pColumn = pDb->getDict()->findColumn( pTable,
										pszColumnName)) != NULL)
			{
				// Column name is ambiguous - belongs to more than one of
				// the tables specified.
				
				if (pFoundColumn)
				{
					if (peParseError)
					{
						*peParseError = SQL_ERR_AMBIGUOUS_COLUMN_NAME;
					}
					rc = RC_SET( NE_SFLM_Q_AMBIGUOUS_COLUMN_NAME);
					goto Exit;
				}
				pFoundColumn = pColumn;
				pFoundTable = pTable;
			}
		}
		
		if (!pFoundColumn)
		{
			if (peParseError)
			{
				*peParseError = SQL_ERR_INVALID_COLUMN_NAME;
			}
			rc = RC_SET( NE_SFLM_Q_INVALID_COLUMN_NAME);
			goto Exit;
		}
		*puiTableNum = pFoundTable->uiTableNum;
		*puiColumnNum = pFoundColumn->uiColumnNum;
	}
	if (peParseError)
	{
		*peParseError = SQL_NO_ERROR;
	}
	
Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Resolve all of the column names that have been specified to one of
//			the tables specified in the table list.  Each column must belong
//			to one of the tables in the list.
//-------------------------------------------------------------------------
RCODE SQLQuery::resolveColumnNames(
	TABLE_ITEM *	pTableList)
{
	RCODE				rc = NE_SFLM_OK;
	SQL_NODE *		pSQLNode;
	SQL_TABLE *		pSQLTable;
	FLMUINT			uiTableNum;
	FLMUINT			uiColumnNum;

	// May not be any names to resolve if the select statement specifed "*"
	// for the list of columns.
	
	if (!m_bResolveNames)
	{
		goto Exit;
	}
	
	pSQLNode = m_pQuery;
	
	for (;;)
	{
		if (pSQLNode->pFirstChild)
		{
			pSQLNode = pSQLNode->pFirstChild;
			continue;
		}
		
		if (pSQLNode->eNodeType == SQL_COLUMN_NODE)
		{
			if (RC_BAD( rc = resolveColumnName( m_pDb, pTableList,
									pSQLNode->nd.column.pszTableAlias,
									pSQLNode->nd.column.pszColumnName,
									&uiTableNum, &uiColumnNum, NULL)))
			{
				goto Exit;
			}
			
			if (RC_BAD( rc = addTable( uiTableNum, &pSQLTable)))
			{
				goto Exit;
			}
			pSQLNode->nd.column.pSQLTable = pSQLTable;
			pSQLNode->nd.column.uiColumnNum = uiColumnNum;
		}
		
		// Continue to the next sibling - or the next sibling of the first
		// node in parent chain that has a sibling.
		
		for (;;)
		{
			if (pSQLNode->pNextSib)
			{
				pSQLNode = pSQLNode->pNextSib;
				break;
			}
			if ((pSQLNode = pSQLNode->pParent) == NULL)
			{
				break;
			}
		}
		if (!pSQLNode)
		{
			break;
		}
	}
	
Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Add a column name.
//-------------------------------------------------------------------------
RCODE SQLQuery::addColumn(
	const char *	pszTableAlias,
	FLMUINT			uiTableAliasLen,
	const char *	pszColumnName,
	FLMUINT			uiColumnNameLen)
{
	RCODE				rc = NE_SFLM_OK;
	SQL_NODE *		pSQLNode;
	char *			pszTmp;

	// Must be expecting an operand

	if (!expectingOperand())
	{
		rc = RC_SET( NE_SFLM_Q_UNEXPECTED_COLUMN);
		goto Exit;
	}

	// Allocate a column node

	if (RC_BAD( rc = allocOperandNode( SQL_COLUMN_NODE, &pSQLNode)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = m_pool.poolAlloc( uiTableAliasLen + uiColumnNameLen + 2,
									(void **)&pszTmp)))
	{
		goto Exit;
	}
	
	// Save the names - will resolve to numbers later.
	
	if (uiTableAliasLen)
	{
		pSQLNode->nd.column.pszTableAlias = pszTmp;
		f_memcpy( pszTmp, pszTableAlias, uiTableAliasLen + 1);
		pszTmp += (uiTableAliasLen + 1);
	}
	else
	{
		pSQLNode->nd.column.pszTableAlias = NULL;
	}
	pSQLNode->nd.column.pszColumnName = pszTmp;
	f_memcpy( pszTmp, pszColumnName, uiColumnNameLen + 1);
	pSQLNode->nd.column.pSQLTable = NULL;
	pSQLNode->nd.column.uiColumnNum = 0;
	
	m_bResolveNames = TRUE;

Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Add a FLMUINT64 number constant.
//-------------------------------------------------------------------------
RCODE SQLQuery::addUINT64(
	FLMUINT64	ui64Num)
{
	RCODE				rc = NE_SFLM_OK;
	SQL_NODE *		pSQLNode;

	// Must be expecting an operand

	if (!expectingOperand())
	{
		rc = RC_SET( NE_SFLM_Q_UNEXPECTED_CONSTANT);
		goto Exit;
	}
	
	// Allocate a value node

	if (RC_BAD( rc = allocOperandNode( SQL_VALUE_NODE, &pSQLNode)))
	{
		goto Exit;
	}
	pSQLNode->currVal.eValType = SQL_UINT64_VAL;
	pSQLNode->currVal.uiFlags |= SQL_VAL_IS_CONSTANT;	
	pSQLNode->currVal.val.ui64Val = ui64Num;

Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Add a FLMUINT number constant.
//-------------------------------------------------------------------------
RCODE SQLQuery::addUINT(
	FLMUINT	uiNum)
{
	RCODE				rc = NE_SFLM_OK;
	SQL_NODE *		pSQLNode;

	// Must be expecting an operand

	if (!expectingOperand())
	{
		rc = RC_SET( NE_SFLM_Q_UNEXPECTED_CONSTANT);
		goto Exit;
	}
	
	// Allocate a value node

	if (RC_BAD( rc = allocOperandNode( SQL_VALUE_NODE, &pSQLNode)))
	{
		goto Exit;
	}
	pSQLNode->currVal.eValType = SQL_UINT_VAL;
	pSQLNode->currVal.uiFlags |= SQL_VAL_IS_CONSTANT;	
	pSQLNode->currVal.val.uiVal = uiNum;

Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Add a FLMINT64 number constant.
//-------------------------------------------------------------------------
RCODE SQLQuery::addINT64(
	FLMINT64	i64Num)
{
	RCODE				rc = NE_SFLM_OK;
	SQL_NODE *		pSQLNode;

	// Must be expecting an operand

	if (!expectingOperand())
	{
		rc = RC_SET( NE_SFLM_Q_UNEXPECTED_CONSTANT);
		goto Exit;
	}
	
	// Allocate a value node

	if (RC_BAD( rc = allocOperandNode( SQL_VALUE_NODE, &pSQLNode)))
	{
		goto Exit;
	}
	pSQLNode->currVal.eValType = SQL_INT64_VAL;
	pSQLNode->currVal.uiFlags |= SQL_VAL_IS_CONSTANT;	
	pSQLNode->currVal.val.i64Val = i64Num;

Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Add a FLMINT number constant.
//-------------------------------------------------------------------------
RCODE SQLQuery::addINT(
	FLMINT	iNum)
{
	RCODE				rc = NE_SFLM_OK;
	SQL_NODE *		pSQLNode;

	// Must be expecting an operand

	if (!expectingOperand())
	{
		rc = RC_SET( NE_SFLM_Q_UNEXPECTED_CONSTANT);
		goto Exit;
	}
	
	// Allocate a value node

	if (RC_BAD( rc = allocOperandNode( SQL_VALUE_NODE, &pSQLNode)))
	{
		goto Exit;
	}
	pSQLNode->currVal.eValType = SQL_INT_VAL;
	pSQLNode->currVal.uiFlags |= SQL_VAL_IS_CONSTANT;	
	pSQLNode->currVal.val.iVal = iNum;

Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Add a boolean constant.
//-------------------------------------------------------------------------
RCODE SQLQuery::addBoolean(
	FLMBOOL	bValue,
	FLMBOOL	bUnknown)
{
	RCODE				rc = NE_SFLM_OK;
	SQL_NODE *		pSQLNode;

	// Must be expecting an operand

	if (!expectingOperand())
	{
		rc = RC_SET( NE_SFLM_Q_UNEXPECTED_BOOLEAN);
		goto Exit;
	}
	
	// Allocate a value node

	if (RC_BAD( rc = allocOperandNode( SQL_VALUE_NODE, &pSQLNode)))
	{
		goto Exit;
	}
	pSQLNode->currVal.eValType = SQL_BOOL_VAL;
	pSQLNode->currVal.uiFlags |= SQL_VAL_IS_CONSTANT;
	if (bUnknown)
	{
		pSQLNode->currVal.val.eBool = SQL_UNKNOWN;
	}
	else if (bValue)
	{
		pSQLNode->currVal.val.eBool = SQL_TRUE;
	}
	else
	{
		pSQLNode->currVal.val.eBool = SQL_FALSE;
	}

Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Add a UTF8 string constant.
//-------------------------------------------------------------------------
RCODE SQLQuery::addUTF8String(
	const FLMBYTE *	pszUTF8Str,
	FLMUINT				uiStrLen,
	FLMUINT				uiNumChars)
{
	RCODE					rc = NE_SFLM_OK;
	SQL_NODE *			pSQLNode;
	FLMBYTE *			pszTmp;
	
	// Must be expecting an operand

	if (!expectingOperand())
	{
		rc = RC_SET( NE_SFLM_Q_UNEXPECTED_CONSTANT);
		goto Exit;
	}
	
	if (!uiStrLen)
	{
		uiStrLen = f_strlen( (const char *)pszUTF8Str);
	}
	
	// Allocate a value node

	if (RC_BAD( rc = m_pool.poolCalloc( uiStrLen + 1, (void **)&pszTmp)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = allocOperandNode( SQL_VALUE_NODE, &pSQLNode)))
	{
		goto Exit;
	}
	pSQLNode->currVal.eValType = SQL_UTF8_VAL;
	pSQLNode->currVal.uiFlags |= SQL_VAL_IS_CONSTANT;
	pSQLNode->currVal.val.str.pszStr = pszTmp;	
	pSQLNode->currVal.val.str.uiByteLen = uiStrLen + 1;
	pSQLNode->currVal.val.str.uiNumChars = uiNumChars;
	f_memcpy( pszTmp, pszUTF8Str, uiStrLen);
	pszTmp [uiStrLen] = 0;
	
	// See if there are any wildcards in the string.
	
	while (*pszTmp)
	{
		if (*pszTmp == '*')
		{
			pSQLNode->currVal.uiFlags |= SQL_VAL_HAS_WILDCARDS;
			break;
		}
		else if (*pszTmp == '\\')
		{
			
			// Skip over whatever comes after a backslash - it is an
			// escaped character, so it won't count as a wildcard.
			
			pszTmp++;
			if (!(*pszTmp))
			{
				break;
			}
		}
		pszTmp++;
	}

Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Add a binary constant.
//-------------------------------------------------------------------------
RCODE SQLQuery::addBinary(
	const FLMBYTE *	pucValue,
	FLMUINT				uiValueLen)
{
	RCODE			rc = NE_SFLM_OK;
	SQL_NODE *	pSQLNode;
	FLMBYTE *	pucTmp;
	
	// Must be expecting an operand

	if (!expectingOperand())
	{
		rc = RC_SET( NE_SFLM_Q_UNEXPECTED_CONSTANT);
		goto Exit;
	}
	
	// Allocate a value node

	if (RC_BAD( rc = m_pool.poolCalloc( uiValueLen, (void **)&pucTmp)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = allocOperandNode( SQL_VALUE_NODE, &pSQLNode)))
	{
		goto Exit;
	}
	pSQLNode->currVal.eValType = SQL_BINARY_VAL;
	pSQLNode->currVal.uiFlags |= SQL_VAL_IS_CONSTANT;
	pSQLNode->currVal.val.bin.uiByteLen = uiValueLen;
	pSQLNode->currVal.val.bin.pucValue = pucTmp;	
	f_memcpy( pucTmp, pucValue, uiValueLen);
	
Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Add an ORDER BY component
//-------------------------------------------------------------------------
RCODE SQLQuery::orderBy(
	FLMUINT			uiTableNum,
	FLMUINT			uiColumnNum,
	FLMBOOL			bDescending)
{
	RCODE				rc = NE_SFLM_OK;
	SQL_TABLE *		pSQLTable;
	SQL_ORDER_BY *	pOrderBy;

	// Find the table structure for the table - should already exist.
	
	pSQLTable = m_pFirstSQLTable;
	while (pSQLTable && pSQLTable->uiTableNum != uiTableNum)
	{
		pSQLTable = pSQLTable->pNext;
	}
	if (!pSQLTable)
	{
		rc = RC_BAD( NE_SFLM_Q_BAD_ORDER_BY_TABLE);
		goto Exit;
	}
	
	// Make sure that the order by component has not already been
	// specified.
	
	pOrderBy = m_pFirstOrderBy;
	while (pOrderBy)
	{
		if (pOrderBy->pSQLTable == pSQLTable &&
			 pOrderBy->uiColumnNum == uiColumnNum)
		{
			rc = RC_SET( NE_SFLM_Q_DUP_COLUMN_IN_ORDER_BY);
			goto Exit;
		}
		pOrderBy = pOrderBy->pNext;
	}
	
	// Allocate an SQL_ORDER_BY structure and link it into the list of
	// SQL_ORDER_BY structures.  It is assumed that the order in which this
	// method is called specifies 1st, 2nd, 3rd, etc. order of the sort
	// components.
	
	if (RC_BAD( rc = m_pool.poolAlloc( sizeof( SQL_ORDER_BY),
								(void **)&pOrderBy)))
	{
		goto Exit;
	}
	pOrderBy->pSQLTable = pSQLTable;
	pOrderBy->uiColumnNum = uiColumnNum;
	pOrderBy->bDescending = bDescending;
	pOrderBy->pNext = NULL;
	if (m_pLastOrderBy)
	{
		m_pLastOrderBy->pNext = pOrderBy;
	}
	else
	{
		m_pFirstOrderBy = pOrderBy;
	}
	m_pLastOrderBy = pOrderBy;
	
Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Set an index on a table.
//-------------------------------------------------------------------------
RCODE SQLQuery::setIndex(
	FLMUINT	uiTableNum,
	FLMUINT	uiIndexNum)
{
	RCODE				rc = NE_SFLM_OK;
	SQL_TABLE *		pSQLTable;
	
	flmAssert( !m_bOptimized);

	// Find the table structure for the table - should already exist.
	
	pSQLTable = m_pFirstSQLTable;
	while (pSQLTable && pSQLTable->uiTableNum != uiTableNum)
	{
		pSQLTable = pSQLTable->pNext;
	}
	if (!pSQLTable)
	{
		rc = RC_BAD( NE_SFLM_Q_INVALID_TABLE_FOR_INDEX);
		goto Exit;
	}
	
	if ((pSQLTable->uiIndexNum = uiIndexNum) == 0)
	{
		pSQLTable->bScan = TRUE;
	}
	else
	{
		pSQLTable->bScan = FALSE;
	}
	
Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Determine if the current token is one that would terminate the
//			expression.
//------------------------------------------------------------------------------
FSTATIC FLMBOOL isTerminatingToken(
	const char *	pszToken,
	const char **	ppszTerminatingTokens,
	const char **	ppszTerminator)
{
	FLMUINT	uiLoop;
	
	if (ppszTerminatingTokens)
	{
		for (uiLoop = 0; ppszTerminatingTokens [uiLoop]; uiLoop++)
		{
			if (f_stricmp( ppszTerminatingTokens [uiLoop], pszToken) == 0)
			{
				*ppszTerminator = ppszTerminatingTokens [uiLoop];
				return( TRUE);
			}
		}
	}
	return( FALSE);	
}

//------------------------------------------------------------------------------
// Desc:	Process a token in the expression that begins with an alphabetic
// 		character.
//------------------------------------------------------------------------------
RCODE SQLStatement::processAlphaToken(
	TABLE_ITEM *	pTableList,
	const char **	ppszTerminatingTokens,
	const char **	ppszTerminator,
	SQLQuery *		pSqlQuery,
	FLMBOOL *		pbDone)
{
	RCODE				rc = NE_SFLM_OK;
	FLMUINT			uiSaveLineNum;
	FLMUINT			uiSaveLineOffset;
	FLMUINT			uiSaveLineFilePos;
	FLMUINT			uiSaveLineBytes;
	char				szToken [MAX_SQL_NAME_LEN + 1];
	FLMUINT			uiTokenLen;
	char				szColumnName [MAX_SQL_NAME_LEN + 1];
	FLMUINT			uiColumnNameLen;
	FLMUINT			uiTokenLineOffset;
	FLMUINT			uiTableNum;
	FLMUINT			uiColumnNum;
	SQLParseError	eParseError;
	FLMBYTE			ucBuffer [200];
	F_DynaBuf		dynaBuf( ucBuffer, sizeof( ucBuffer));
	
	*pbDone = FALSE;
	if (RC_BAD( rc = getName( szToken, sizeof( szToken),
										&uiTokenLen, &uiTokenLineOffset)))
	{
		goto Exit;
	}
	uiSaveLineNum = m_uiCurrLineNum;
	uiSaveLineOffset = uiTokenLineOffset;
	uiSaveLineFilePos = m_uiCurrLineFilePos;
	uiSaveLineBytes = m_uiCurrLineBytes;
	
	if (isTerminatingToken( szToken, ppszTerminatingTokens,
									ppszTerminator))
	{
		if (pSqlQuery->criteriaIsComplete())
		{
			*pbDone = TRUE;
			goto Exit;
		}
	}
	else if (f_stricmp( szToken, "and") == 0 ||
				f_stricmp( szToken, "or") == 0)
	{
		if (pSqlQuery->expectingOperator())
		{

			// Treat as AND operator
			
			rc = pSqlQuery->addOperator(
							(eSQLQueryOperators)(f_toupper( szToken[0]) == 'A'
														? SQL_AND_OP
														: SQL_OR_OP), 0);
			goto Exit;
		}
	}
	else if (f_stricmp( szToken, "not") == 0)
	{
		
		// Interestingly, NOT operators should only appear when
		// we are expecting an operand.
		
		if (pSqlQuery->expectingOperator())
		{
			setErrInfo( uiSaveLineNum,
					uiSaveLineOffset,
					SQL_ERR_UNEXPECTED_NOT_OPERATOR,
					uiSaveLineFilePos,
					uiSaveLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
			goto Exit;
		}
		else
		{
			rc = pSqlQuery->addOperator( SQL_NOT_OP, 0);
			goto Exit;
		}
	}
	else if (f_stricmp( szToken, "true") == 0 ||
				f_stricmp( szToken, "false") == 0 ||
				f_stricmp( szToken, "unknown") == 0)
	{
		if (pSqlQuery->expectingOperand())
		{
			
			// Treat as AND operator
			
			rc = pSqlQuery->addBoolean(
							(FLMBOOL)(f_toupper( szToken[0]) == 'T'
														? TRUE
														: FALSE),
							(FLMBOOL)(f_toupper( szToken[0]) == 'U'
														? TRUE
														: FALSE));
			goto Exit;
		}
	}
	else if (f_stricmp( szToken, "binary") == 0)
	{
		if (pSqlQuery->expectingOperand())
		{
			if (RC_BAD( rc = haveToken( "(", TRUE)))
			{
				if (rc != NE_SFLM_NOT_FOUND && rc != NE_SFLM_EOF_HIT)
				{
					goto Exit;
				}
			}
			else
			{
				dynaBuf.truncateData( 0);
				if (RC_BAD( rc = getBinaryValue( &dynaBuf)))
				{
					goto Exit;
				}
				rc = pSqlQuery->addBinary( dynaBuf.getBufferPtr(),
												dynaBuf.getDataLength());
				goto Exit;
			}
		}
	}
	
	// At this point, the only thing left is for it to be a column name
	// or a tablename.columnname.
	
	// If we are expecting an operator, it is an error for
	// a column name to be specified.
	
	if (pSqlQuery->expectingOperator())
	{
		setErrInfo( uiSaveLineNum,
				uiSaveLineOffset,
				SQL_ERR_EXPECTING_OPERATOR,
				uiSaveLineFilePos,
				uiSaveLineBytes);
		rc = RC_SET( NE_SFLM_INVALID_SQL);
		goto Exit;
	}
	
	// If we fall through to here, treat the token as a column
	// name or a table alias.  See if there is a period for the
	// next token.
	
	if (RC_BAD( rc = haveToken( ".", TRUE)))
	{
		if (rc != NE_SFLM_NOT_FOUND && rc != NE_SFLM_EOF_HIT)
		{
			goto Exit;
		}
		rc = NE_SFLM_OK;
		
		if (!pTableList)
		{
			
			// If there is no table list, the list of tables has not yet been
			// specified, so we will have to save this column in a slightly
			// different way.
		
			if (RC_BAD( rc = pSqlQuery->addColumn( NULL, 0, szToken, uiTokenLen)))
			{
				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = resolveColumnName( m_pDb, pTableList, NULL,
											szToken, &uiTableNum, &uiColumnNum,
											&eParseError)))
			{
				if (eParseError != SQL_NO_ERROR)
				{
					setErrInfo( uiSaveLineNum,
							uiSaveLineOffset,
							eParseError,
							uiSaveLineFilePos,
							uiSaveLineBytes);
				}
				goto Exit;
			}
			if (RC_BAD( rc = pSqlQuery->addColumn( uiTableNum, uiColumnNum)))
			{
				goto Exit;
			}
		}
	}
	else
	{
		
		if (RC_BAD( rc = getName( szColumnName, sizeof( szColumnName),
								&uiColumnNameLen, &uiTokenLineOffset)))
		{
			goto Exit;
		}

		if (!pTableList)
		{
			
			// No table list yet - save using the table alias and column name.
		
			if (RC_BAD( rc = pSqlQuery->addColumn( szToken, uiTokenLen,
											szColumnName, uiColumnNameLen)))
			{
				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = resolveColumnName( m_pDb, pTableList, szToken,
											szColumnName, &uiTableNum, &uiColumnNum,
											&eParseError)))
			{
				if (eParseError != SQL_NO_ERROR)
				{
					setErrInfo( uiSaveLineNum,
							uiSaveLineOffset,
							eParseError,
							uiSaveLineFilePos,
							uiSaveLineBytes);
				}
				goto Exit;
			}
		}
		if (RC_BAD( rc = pSqlQuery->addColumn( uiTableNum, uiColumnNum)))
		{
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Process criteria.
//------------------------------------------------------------------------------
RCODE SQLStatement::parseCriteria(
	TABLE_ITEM *	pTableList,
	const char **	ppszTerminatingTokens,
	FLMBOOL			bEofOK,
	const char **	ppszTerminator,
	SQLQuery *		pSqlQuery)
{
	RCODE						rc = NE_SFLM_OK;
	char						cChar;
	FLMUINT64				ui64Num;
	FLMBOOL					bNeg;
	FLMBYTE					ucBuffer [200];
	F_DynaBuf				dynaBuf( ucBuffer, sizeof( ucBuffer));
	FLMUINT					uiNumChars;
	eSQLQueryOperators	eOperator = SQL_UNKNOWN_OP;
	FLMUINT					uiNestedParens = 0;
	FLMUINT					uiTokenLineOffset;
	
	// Process tokens
	
	for (;;)
	{
		if (RC_BAD( rc = skipWhitespace( FALSE)))
		{
			if (rc == NE_SFLM_EOF_HIT)
			{
				*ppszTerminator = NULL;
				if (bEofOK)
				{
					if (!pSqlQuery->criteriaIsComplete())
					{
						goto Incomplete_Query;
					}
					else
					{
						rc = NE_SFLM_OK;
					}
				}
				else
				{
					setErrInfo( m_uiCurrLineNum,
							m_uiCurrLineOffset,
							SQL_ERR_UNEXPECTED_EOF,
							m_uiCurrLineFilePos,
							m_uiCurrLineBytes);
					rc = RC_SET( NE_SFLM_INVALID_SQL);
				}
			}
			goto Exit;
		}
		
		// See if we can figure out what kind of token it is.
		
		cChar = (char)m_pucCurrLineBuf [m_uiCurrLineOffset];
		uiTokenLineOffset = m_uiCurrLineOffset;

		switch (cChar)
		{
			case '"':
			case '\'':
			
				// If we are expecting an operator, it is an error for
				// a string constant to be specified.
				
				if (pSqlQuery->expectingOperator())
				{
					setErrInfo( m_uiCurrLineNum,
							uiTokenLineOffset,
							SQL_ERR_EXPECTING_OPERATOR,
							m_uiCurrLineFilePos,
							m_uiCurrLineBytes);
					rc = RC_SET( NE_SFLM_INVALID_SQL);
					goto Exit;
				}
				
				// Better be a quoted string.
				
				dynaBuf.truncateData( 0);
				if (RC_BAD( rc = getUTF8String( FALSE, FALSE, NULL, 0, NULL,
											&uiNumChars, &dynaBuf)))
				{
					goto Exit;
				}
				
				if (RC_BAD( rc = pSqlQuery->addUTF8String( dynaBuf.getBufferPtr(),
												dynaBuf.getDataLength() - 1,
												uiNumChars)))
				{
					goto Exit;
				}
				
				break;
			case '+':
			
				// A plus in front of an operand may be ignored.
				
				m_uiCurrLineOffset++;
				if (pSqlQuery->expectingOperand())
				{
					continue;
				}
				eOperator = SQL_PLUS_OP;
				goto Add_Operator;
			case '-':
				eOperator = (pSqlQuery->expectingOperand())
							  ? SQL_NEG_OP
							  : SQL_MINUS_OP;
				m_uiCurrLineOffset++;
				goto Add_Operator;
			case '*':
				eOperator = SQL_MULT_OP;
				m_uiCurrLineOffset++;
				goto Test_If_Expecting_Operand;
			case '/':
				eOperator = SQL_DIV_OP;
				m_uiCurrLineOffset++;
				goto Test_If_Expecting_Operand;
			case '%':
				eOperator = SQL_MOD_OP;
				m_uiCurrLineOffset++;
				goto Test_If_Expecting_Operand;
			case '(':
				if (pSqlQuery->expectingOperator())
				{
					setErrInfo( m_uiCurrLineNum,
							uiTokenLineOffset,
							SQL_ERR_UNEXPECTED_LPAREN,
							m_uiCurrLineFilePos,
							m_uiCurrLineBytes);
					rc = RC_SET( NE_SFLM_INVALID_SQL);
					goto Exit;
				}
				eOperator = SQL_LPAREN_OP;
				m_uiCurrLineOffset++;
				uiNestedParens++;
				goto Add_Operator;
			case ')':
				if (pSqlQuery->expectingOperand())
				{
					setErrInfo( m_uiCurrLineNum,
							uiTokenLineOffset,
							SQL_ERR_UNEXPECTED_RPAREN,
							m_uiCurrLineFilePos,
							m_uiCurrLineBytes);
					rc = RC_SET( NE_SFLM_INVALID_SQL);
					goto Exit;
				}
				else if (!uiNestedParens)
				{
					setErrInfo( m_uiCurrLineNum,
							uiTokenLineOffset,
							SQL_ERR_UNMATCHED_RPAREN,
							m_uiCurrLineFilePos,
							m_uiCurrLineBytes);
					rc = RC_SET( NE_SFLM_INVALID_SQL);
					goto Exit;
				}
				eOperator = SQL_RPAREN_OP;
				m_uiCurrLineOffset++;
				uiNestedParens--;
				break;
			case '!':
				if (pSqlQuery->expectingOperator())
				{
					setErrInfo( m_uiCurrLineNum,
							uiTokenLineOffset,
							SQL_ERR_UNEXPECTED_NOT_OPERATOR,
							m_uiCurrLineFilePos,
							m_uiCurrLineBytes);
					rc = RC_SET( NE_SFLM_INVALID_SQL);
					goto Exit;
				}
				m_uiCurrLineOffset++;
				goto Add_Operator;
			case '=':
				eOperator = SQL_EQ_OP;
				if (m_uiCurrLineOffset + 1 < m_uiCurrLineBytes &&
					 m_pucCurrLineBuf [m_uiCurrLineOffset + 1] == '&')
				{
					m_uiCurrLineOffset += 2;
				}
				else
				{
					m_uiCurrLineOffset++;
				}
				goto Test_If_Expecting_Operand;
			case '&':
				if (m_uiCurrLineOffset + 1 < m_uiCurrLineBytes &&
					 m_pucCurrLineBuf [m_uiCurrLineOffset + 1] == '&')
				{
					eOperator = SQL_AND_OP;
					m_uiCurrLineOffset += 2;
				}
				else
				{
					eOperator = SQL_BITAND_OP;
					m_uiCurrLineOffset++;
				}
				goto Test_If_Expecting_Operand;
			case '|':
				if (m_uiCurrLineOffset + 1 < m_uiCurrLineBytes &&
					 m_pucCurrLineBuf [m_uiCurrLineOffset + 1] == '|')
				{
					eOperator = SQL_OR_OP;
					m_uiCurrLineOffset += 2;
				}
				else
				{
					eOperator = SQL_BITOR_OP;
					m_uiCurrLineOffset++;
				}
				goto Test_If_Expecting_Operand;
			case '^':
				eOperator = SQL_BITXOR_OP;
				m_uiCurrLineOffset++;
				goto Test_If_Expecting_Operand;

Test_If_Expecting_Operand:
				if (pSqlQuery->expectingOperand())
				{
					setErrInfo( m_uiCurrLineNum,
							uiTokenLineOffset,
							SQL_ERR_INVALID_OPERAND,
							m_uiCurrLineFilePos,
							m_uiCurrLineBytes);
					rc = RC_SET( NE_SFLM_INVALID_SQL);
					goto Exit;
				}
Add_Operator:			
				if (RC_BAD( rc = pSqlQuery->addOperator( eOperator, 0)))
				{
					goto Exit;
				}
				break;
			
			default:
				if ((cChar >= 'a' && cChar <= 'z') || (cChar >= 'A' && cChar <= 'Z'))
				{
					FLMBOOL	bDone;
					
					if (RC_BAD( rc = processAlphaToken( pTableList,
												ppszTerminatingTokens,
												ppszTerminator,
												pSqlQuery, &bDone)))
					{
						goto Exit;
					}
					if (bDone)
					{
						goto Exit;
					}
				}
				else if (cChar >= '0' && cChar <= '9')
				{
					
					// If we are expecting an operator, it is an error for
					// a number to be specified.
					
					if (pSqlQuery->expectingOperator())
					{
						setErrInfo( m_uiCurrLineNum,
								m_uiCurrLineOffset,
								SQL_ERR_EXPECTING_OPERATOR,
								m_uiCurrLineFilePos,
								m_uiCurrLineBytes);
						rc = RC_SET( NE_SFLM_INVALID_SQL);
						goto Exit;
					}
					
					// Better be a number.
					
					if (RC_BAD( rc = getNumber( FALSE, &ui64Num, &bNeg, FALSE)))
					{
						goto Exit;
					}
					if (RC_BAD( rc = pSqlQuery->addNumber( ui64Num, bNeg)))
					{
						goto Exit;
					}
				}
				
				// At this point, it has to be an invalid token for selection
				// criteria.
				
				else
				{
					
Incomplete_Query:
					if (pSqlQuery->expectingOperator())
					{
						setErrInfo( m_uiCurrLineNum,
								m_uiCurrLineOffset,
								SQL_ERR_EXPECTING_OPERATOR,
								m_uiCurrLineFilePos,
								m_uiCurrLineBytes);
						rc = RC_SET( NE_SFLM_INVALID_SQL);
						goto Exit;
					}
					else
					{
						flmAssert( pSqlQuery->expectingOperand());
						setErrInfo( m_uiCurrLineNum,
								m_uiCurrLineOffset,
								SQL_ERR_INVALID_OPERAND,
								m_uiCurrLineFilePos,
								m_uiCurrLineBytes);
						rc = RC_SET( NE_SFLM_INVALID_SQL);
						goto Exit;
					}
				}
				break;
		}
	}
	
Exit:

	return( rc);
}

