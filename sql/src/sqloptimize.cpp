//-------------------------------------------------------------------------
// Desc:	Optimize an SQL query
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

#define MINIMUM_COST_ESTIMATE	8

// Local function prototypes

FSTATIC RCODE sqlGetRowIdValue(
	SQL_VALUE *	pValue);
	
FSTATIC RCODE setupPredicate(
	SQL_PRED *					pPred,
	SQL_TABLE *					pSQLTable,
	FLMUINT						uiColumnNum,
	eSQLQueryOperators		eOperator,
	FLMUINT						uiCompareRules,
	FLMBOOL						bNotted,
	SQL_VALUE *					pValue);
	
FSTATIC RCODE sqlCompareValues(
	SQL_VALUE *			pValue1,
	FLMBOOL				bInclusive1,
	FLMBOOL				bNullIsLow1,
	SQL_VALUE *			pValue2,
	FLMBOOL				bInclusive2,
	FLMBOOL				bNullIsLow2,
	FLMUINT				uiCompareRules,
	FLMUINT				uiLanguage,
	FLMINT *				piCmp);
	
FSTATIC SQL_NODE * sqlEvalLogicalOperands(
	SQL_NODE *		pSQLNode);
	
FSTATIC SQL_NODE * sqlClipNotNode(
	SQL_NODE *	pNotNode,
	SQL_NODE **	ppExpr);
	
FSTATIC RCODE createDNFNode(
	F_Pool *				pPool,
	SQL_DNF_NODE *		pParentDNFNode,
	SQL_DNF_NODE **	ppDNFNode,
	SQL_NODE *			pNode);
	
FSTATIC RCODE copyAndLinkSubTree(
	F_Pool *			pPool,
	SQL_DNF_NODE *	pSrcSubTree,
	SQL_DNF_NODE *	pParentNode);
	
FSTATIC RCODE distributeAndOverOr(
	F_Pool *				pPool,
	SQL_DNF_NODE *		pOldOrNode,
	SQL_DNF_NODE **	ppDNFTree);
	
FSTATIC FLMBOOL predIsForOnlyThisTable(
	SQL_NODE *	pPredRootNode,
	SQL_TABLE *	pSQLTable);
	
FSTATIC void rankIndexes(
	F_Db *			pDb,
	SQL_INDEX **	ppFirstSQLIndex,
	SQL_INDEX **	ppLastSQLIndex);
	
//-------------------------------------------------------------------------
// Desc:	Get the row ID constant from an SQL_VALUE node.
//-------------------------------------------------------------------------
FSTATIC RCODE sqlGetRowIdValue(
	SQL_VALUE *	pValue)
{
	RCODE	rc = NE_SFLM_OK;

	switch (pValue->eValType)
	{
		case SQL_UINT_VAL:
			pValue->eValType = SQL_UINT64_VAL;
			pValue->val.ui64Val = (FLMUINT64)pValue->val.uiVal;
			break;
		case SQL_MISSING_VAL:
		case SQL_UINT64_VAL:
			break;
		case SQL_INT_VAL:
			pValue->eValType = SQL_UINT64_VAL;
			pValue->val.ui64Val = (FLMUINT64)((FLMINT64)(pValue->val.iVal));
			break;
		case SQL_INT64_VAL:
			pValue->eValType = SQL_UINT64_VAL;
			pValue->val.ui64Val = (FLMUINT64)(pValue->val.i64Val);
			break;
		default:
			rc = RC_SET_AND_ASSERT( NE_SFLM_Q_INVALID_ROW_ID_VALUE);
			goto Exit;
	}

Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Setup a predicate using the passed in parameters.
//-------------------------------------------------------------------------
FSTATIC RCODE setupPredicate(
	SQL_PRED *					pPred,
	SQL_TABLE *					pSQLTable,
	FLMUINT						uiColumnNum,
	eSQLQueryOperators		eOperator,
	FLMUINT						uiCompareRules,
	FLMBOOL						bNotted,
	SQL_VALUE *					pValue)
{
	RCODE		rc = NE_SFLM_OK;

	pPred->pSQLTable = pSQLTable;
	pPred->uiColumnNum = uiColumnNum;
	if (!pValue || pValue->eValType != SQL_UTF8_VAL)
	{

		// Comparison rules don't matter for anything that is
		// not text, so we normalize them to zero, so the test
		// below to see if the comparison rule is the same as
		// the comparison rule of the operator will work.

		pPred->uiCompareRules = 0;
	}
	else
	{
		pPred->uiCompareRules = uiCompareRules;
	}
	pPred->bNotted = bNotted;
	switch (eOperator)
	{
		case SQL_EXISTS_OP:
		case SQL_NE_OP:
			pPred->eOperator = eOperator;
			pPred->pFromValue = pValue;
			break;
		case SQL_APPROX_EQ_OP:
			pPred->eOperator = eOperator;
			pPred->pFromValue = pValue;
			pPred->bInclFrom = TRUE;
			pPred->bInclUntil = TRUE;
			break;
		case SQL_EQ_OP:
			if ((pValue->uiFlags & SQL_VAL_IS_CONSTANT) &&
				 (pValue->uiFlags & SQL_VAL_HAS_WILDCARDS))
			{
				pPred->eOperator = SQL_MATCH_OP;
				pPred->pFromValue = pValue;
			}
			else
			{
				pPred->eOperator = SQL_RANGE_OP;
				pPred->pFromValue = pValue;
				pPred->pUntilValue = pValue;
				pPred->bInclFrom = TRUE;
				pPred->bInclUntil = TRUE;
			}
			break;
		case SQL_LE_OP:
			pPred->eOperator = SQL_RANGE_OP;
			pPred->pFromValue = NULL;
			pPred->pUntilValue = pValue;
			pPred->bInclUntil = TRUE;
			break;
		case SQL_LT_OP:
			pPred->eOperator = SQL_RANGE_OP;
			pPred->pFromValue = NULL;
			pPred->pUntilValue = pValue;
			pPred->bInclUntil = FALSE;
			break;
		case SQL_GE_OP:
			pPred->eOperator = SQL_RANGE_OP;
			pPred->pFromValue = pValue;
			pPred->pUntilValue = NULL;
			pPred->bInclFrom = TRUE;
			break;
		case SQL_GT_OP:
			pPred->eOperator = SQL_RANGE_OP;
			pPred->pFromValue = pValue;
			pPred->pUntilValue = NULL;
			pPred->bInclFrom = FALSE;
			break;
		default:
			rc = RC_SET_AND_ASSERT( NE_SFLM_NOT_IMPLEMENTED);
			goto Exit;
	}
	
Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Compare two values
//-------------------------------------------------------------------------
FSTATIC RCODE sqlCompareValues(
	SQL_VALUE *			pValue1,
	FLMBOOL				bInclusive1,
	FLMBOOL				bNullIsLow1,
	SQL_VALUE *			pValue2,
	FLMBOOL				bInclusive2,
	FLMBOOL				bNullIsLow2,
	FLMUINT				uiCompareRules,
	FLMUINT				uiLanguage,
	FLMINT *				piCmp)
{
	RCODE	rc = NE_SFLM_OK;

	// We have already called sqlCanCompare, so no need to do it here

	if (!pValue1)
	{
		if (!pValue2)
		{
			if (bNullIsLow2)
			{
				*piCmp = (FLMINT)(bNullIsLow1 ? 0 : 1);
			}
			else
			{
				*piCmp = (FLMINT)(bNullIsLow1 ? -1 : 0);
			}
		}
		else
		{
			*piCmp = (FLMINT)(bNullIsLow1 ? -1 : 1);
		}
		goto Exit;
	}
	else if (!pValue2)
	{
		*piCmp = (FLMINT)(bNullIsLow2 ? 1 : -1);
		goto Exit;
	}

	if (RC_BAD( rc = sqlCompare( pValue1, pValue2, 
		uiCompareRules, uiLanguage, piCmp)))
	{
		goto Exit;
	}

	// If everything else is equal, the last distinguisher
	// is the inclusive flags and which side of the
	// value we are on if we are exclusive which is indicated
	// by the bNullIsLow flags

	if (*piCmp == 0)
	{
		if (bInclusive1 != bInclusive2)
		{
			if (bNullIsLow1)
			{
				if (bNullIsLow2)
				{
					//			*--> v1
					//			o--> v2		v1 < v2

					//			o--> v1
					//			*--> v2		v1 > v2

					*piCmp = bInclusive1 ? -1 : 1;
				}
				else
				{
					//			*--> v1
					// v2 <--o				v1 > v2

					//			o--> v1
					// v2	<--*				v1 > v2

					*piCmp = 1;
				}
			}
			else
			{
				if (bNullIsLow2)
				{
					// v1 <--*
					//			o--> v2		v1 < v2

					// v1 <--o
					//			*--> v2		v1 < v2

					*piCmp = -1;
				}
				else
				{
					// v1	<--*
					//	v2	<--o				v1 > v2

					// v1	<--o
					// v2	<--*				v1 < v2

					*piCmp = bInclusive1 ? 1 : -1;
				}
			}
		}
		else if (!bInclusive1)
		{

			// bInclusive2 is also FALSE

			if (bNullIsLow1)
			{
				if (!bNullIsLow2)
				{
					//			o--> v1
					// v2	<--o				v1 > v2
					*piCmp = 1;
				}
//				else
//				{
					// 		o--> v1
					// 		o--> v2		v1 == v2
					// *piCmp = 0;
//				}
			}
			else
			{
				if (bNullIsLow2)
				{

					// v1	<--o
					//			o--> v2		v1 < v2

					*piCmp = -1;
				}
//				else
//				{
					// v1	<--o
					// v2	<--o				v1 == v2
					// *piCmp = 0;
//				}
			}
		}
//		else
//		{
			// bInclusive1 == TRUE && bInclusive2 == TRUE
			// else case covers the cases where
			// both are inclusive, in which case it
			// doesn't matter which is low and which
			// is high

					// v1	<--*
					//			*--> v2		v1 == v2

					// v1	<--*
					// v2	<--*				v1 == v2

					//			*--> v1
					// v2	<--*				v1 == v2

					//			*--> v1
					//			*--> v2		v1 == v2

//		}
	}

Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Intersect a predicate into an existing predicate.
//-------------------------------------------------------------------------
RCODE SQLQuery::intersectPredicates(
	SQL_PRED *				pPred,
	eSQLQueryOperators	eOperator,
	FLMUINT					uiCompareRules,
	FLMBOOL					bNotted,
	SQL_VALUE *				pValue,
	FLMBOOL *				pbAlwaysFalse,
	FLMBOOL *				pbIntersected)
{
	RCODE		rc = NE_SFLM_OK;
	FLMINT	iCmp;
	FLMBOOL	bDoMatch;

	*pbIntersected = FALSE;
	if (!pValue || pValue->eValType != SQL_UTF8_VAL)
	{
		bDoMatch = FALSE;

		// Comparison rules don't matter for anything that is
		// not text, so we normalize them to zero, so the test
		// below to see if the comparison rule is the same as
		// the comparison rule of the operator will work.

		uiCompareRules = 0;
	}
	else
	{
		bDoMatch = (eOperator == SQL_EQ_OP &&
						(pValue->uiFlags & SQL_VAL_IS_CONSTANT) &&
						(pValue->uiFlags & SQL_VAL_HAS_WILDCARDS))
						? TRUE
						: FALSE;
	}

	if (eOperator == SQL_EXISTS_OP)
	{
		*pbIntersected = TRUE;

		// An exists operator will either merge with an existing predicate or
		// cancel the whole thing out as an empty result.

		// If this predicate is not-exists, another predicate ANDed with this
		// one can never return a result that will match, unless that predicate
		// is also a not-exists, in which case, we simply combine this one
		// with that one.

		if (bNotted)
		{
			if (pPred->eOperator != SQL_EXISTS_OP || !pPred->bNotted)
			{
				*pbAlwaysFalse = TRUE;
			}
		}
	}
	else if (pPred->eOperator == SQL_EXISTS_OP)
	{

		*pbIntersected = TRUE;
		
		// If the first predicate is an exists operator
		// it will be the only one, because otherwise
		// it will have been merged with another operator
		// in the code just above.

		flmAssert( !pPred->pNext);

		// If the predicate is notted, another predicate
		// ANDed with this one can never return a result.

		if (pPred->bNotted)
		{
			*pbAlwaysFalse = TRUE;
		}
		else
		{

			// Change the predicate to the current
			// operator.
			
			if (RC_BAD( rc = setupPredicate( pPred, pPred->pSQLTable,
								pPred->uiColumnNum, eOperator, uiCompareRules,
								bNotted, pValue)))
			{
				goto Exit;
			}
		}
	}
	
	// See if the operator intersects a range operator

	else if (pPred->eOperator == SQL_RANGE_OP &&
				pPred->uiCompareRules == uiCompareRules &&
				!bDoMatch &&
				(eOperator == SQL_EQ_OP ||
				 eOperator == SQL_LE_OP ||
				 eOperator == SQL_LT_OP ||
				 eOperator == SQL_GE_OP ||
				 eOperator == SQL_GT_OP))
	{
		SQL_VALUE *	pFromValue;
		SQL_VALUE *	pUntilValue;
		FLMBOOL		bInclFrom;
		FLMBOOL		bInclUntil;
		
		*pbIntersected = TRUE;

		pFromValue = (eOperator == SQL_EQ_OP ||
						  eOperator == SQL_GE_OP ||
						  eOperator == SQL_GT_OP)
						  ? pValue
						  : NULL;
		pUntilValue = (eOperator == SQL_EQ_OP ||
							eOperator == SQL_LE_OP ||
							eOperator == SQL_LT_OP)
							? pValue
							: NULL;
		bInclFrom = (FLMBOOL)(eOperator == SQL_EQ_OP ||
									 eOperator == SQL_GE_OP
									 ? TRUE
									 : FALSE);
		bInclUntil = (FLMBOOL)(eOperator == SQL_EQ_OP ||
									  eOperator == SQL_LE_OP
									  ? TRUE
									  : FALSE);

		// If the value type is not compatible with the predicate's
		// value type, we cannot do the comparison, and there is
		// no intersection.

		if (!sqlCanCompare( pValue, pPred->pFromValue) ||
			 !sqlCanCompare( pValue, pPred->pUntilValue))
		{
			*pbAlwaysFalse = TRUE;
		}
		else if (RC_BAD( rc = sqlCompareValues( pFromValue,
							bInclFrom, TRUE,
							pPred->pFromValue, pPred->bInclFrom, TRUE,
							uiCompareRules, m_uiLanguage, &iCmp)))
		{
			goto Exit;
		}
		else if (iCmp > 0)
		{

			// From value is greater than predicate's from value.
			// If the from value is also greater than the predicate's
			// until value, we have no intersection.

			if (RC_BAD( rc = sqlCompareValues( pFromValue,
						bInclFrom, TRUE,
						pPred->pUntilValue, pPred->bInclUntil, FALSE,
						uiCompareRules, m_uiLanguage, &iCmp)))
			{
				goto Exit;
			}
			if (iCmp > 0)
			{
				*pbAlwaysFalse = TRUE;
			}
			else
			{
				pPred->pFromValue = pFromValue;
				pPred->bInclFrom = bInclFrom;
			}
		}
		else if (RC_BAD( rc = sqlCompareValues( pUntilValue,
							bInclUntil, FALSE,
							pPred->pUntilValue, pPred->bInclUntil, FALSE,
							uiCompareRules, m_uiLanguage, &iCmp)))
		{
			goto Exit;
		}
		else if (iCmp < 0)
		{

			// Until value is less than predicate's until value.  If the
			// until value is also less than predicate's from value, we
			// have no intersection.

			if (RC_BAD( rc = sqlCompareValues( pUntilValue,
							bInclUntil, FALSE,
							pPred->pFromValue, pPred->bInclFrom, TRUE,
							uiCompareRules, m_uiLanguage, &iCmp)))
			{
				goto Exit;
			}
			if (iCmp < 0)
			{
				*pbAlwaysFalse = TRUE;
			}
			else
			{
				pPred->pUntilValue = pUntilValue;
				pPred->bInclUntil = bInclUntil;
			}
		}
	}
	
Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Convert an operand to a predicate.  If it is merged with another
//			predicate, remove it and return the next node in the list of
//			operands.  If it is not merged, still return the next node in
//			the list of operands.
//-------------------------------------------------------------------------
RCODE SQLQuery::addPredicate(
	SQL_SUBQUERY *			pSubQuery,
	FLMUINT *				puiOperand,
	SQL_TABLE *				pSQLTable,
	FLMUINT					uiColumnNum,
	eSQLQueryOperators	eOperator,
	FLMUINT					uiCompareRules,
	FLMBOOL					bNotted,
	SQL_VALUE *				pValue)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiOperand = *puiOperand;
	FLMUINT		uiLoop;
	SQL_NODE *	pCurrNode;
	SQL_NODE *	pOperandNode;
	FLMBOOL		bAlwaysFalse;
	SQL_PRED *	pPred;
	
	// Convert the constant value in a row id predicate to
	// a 64 bit unsigned value.

	if (eOperator != SQL_EXISTS_OP && !uiColumnNum)
	{
		if (RC_BAD( rc = sqlGetRowIdValue( pValue)))
		{
			goto Exit;
		}
	}
	
	// Look at all of the operands up to the one we are processing to
	// see if this operand should be merged with a previous one.

	for (uiLoop = 0; uiLoop < uiOperand; uiLoop++)
	{
		pCurrNode = pSubQuery->ppOperands [uiLoop];
		if (pCurrNode->eNodeType != SQL_PRED_NODE)
		{
			pCurrNode = pCurrNode->pNextSib;
			continue;
		}
		pPred = &pCurrNode->nd.pred;
		if (pPred->pSQLTable == pSQLTable && pPred->uiColumnNum == uiColumnNum)
		{
			FLMBOOL bIntersected;
			
			if (RC_BAD( rc = intersectPredicates( pPred, eOperator,
										uiCompareRules, bNotted, pValue,
										&bAlwaysFalse, &bIntersected)))
			{
				goto Exit;
			}
			if (!bIntersected && !bAlwaysFalse)
			{
				continue;
			}
			
			// If we get a false result, then we know that the
			// intersection of predicates is creating a situation where
			// it can never be true, so this sub-query can never return
			// anything.  Therefore, we remove the sub-query.
	
			if (bAlwaysFalse)
			{
				
				// Remove the sub-query - it will never return anything.
				
				if (pSubQuery->pPrev)
				{
					pSubQuery->pPrev->pNext = pSubQuery->pNext;
				}
				else
				{
					m_pFirstSubQuery = pSubQuery->pNext;
				}
				if (pSubQuery->pNext)
				{
					pSubQuery->pNext->pPrev = pSubQuery->pPrev;
				}
				else
				{
					m_pLastSubQuery = pSubQuery->pPrev;
				}
				
				// Setup so that we will quit processing this sub-query's
				// operands - it is now unlinked.
				
				uiOperand = pSubQuery->uiOperandCount;
			}
			else
			{
				
				flmAssert( bIntersected);
				
				// We intersected, so we want to remove the current
				// operand node out of the list and set up so that
				// we will increment to the next one in the list.

				pSubQuery->uiOperandCount--;
				if (uiOperand < pSubQuery->uiOperandCount)
				{
					f_memmove( &pSubQuery->ppOperands [uiOperand],
						&pSubQuery->ppOperands [uiOperand + 1],
						sizeof( SQL_NODE *) * (pSubQuery->uiOperandCount - uiOperand));
				}
			}
			goto Exit;
		}
	}

	// If we didn't find one to intersect with, we need to
	// create a new operand node of type SQL_PRED_NODE.  Can't just modify
	// this node, because other sub-queries may be pointing to it also, and
	// they would modify it in a different way.  Unlike other nodes, predicate
	// nodes are ALWAYS tied to one and only one sub-query.
	
	if (RC_BAD( rc = m_pool.poolCalloc( sizeof( SQL_NODE),
										(void **)&pOperandNode)))
	{
		goto Exit;
	}

	// Set the stuff that needs to be set for this predicate.
	
	pOperandNode->eNodeType = SQL_PRED_NODE;
	if (RC_BAD( rc = setupPredicate( &pOperandNode->nd.pred,
								pSQLTable, uiColumnNum,
								eOperator, uiCompareRules, bNotted, pValue)))
	{
		goto Exit;
	}
	pSubQuery->ppOperands [uiOperand] = pOperandNode;
	
	// Go to the next operand
	
	uiOperand++;

Exit:

	*puiOperand = uiOperand;

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Convert all of the operands underneath an AND or OR operator to
//			predicates where possible, except for the operands which are AND
//			or OR nodes.
//-------------------------------------------------------------------------
RCODE SQLQuery::convertOperandsToPredicates( void)
{
	RCODE						rc = NE_SFLM_OK;
	SQL_NODE *				pSQLNode;
	SQL_VALUE *				pValue;
	SQL_TABLE *				pSQLTable;
	FLMUINT					uiColumnNum;
	FLMUINT					uiOperand;
	SQL_SUBQUERY *			pSubQuery;
	SQL_SUBQUERY *			pNextSubQuery;
	
	pSubQuery = m_pFirstSubQuery;
	while (pSubQuery)
	{
		pNextSubQuery = pSubQuery->pNext;
		uiOperand = 0;
		while (uiOperand < pSubQuery->uiOperandCount)
		{
			pSQLNode = pSubQuery->ppOperands [uiOperand];
			if (pSQLNode->eNodeType == SQL_COLUMN_NODE)
			{
				if (RC_BAD( rc = addPredicate( pSubQuery, &uiOperand,
									pSQLNode->nd.column.pSQLTable,
									pSQLNode->nd.column.uiColumnNum,
									SQL_EXISTS_OP, 0, pSQLNode->bNotted, NULL)))
				{
					goto Exit;
				}
			}
			else if (pSQLNode->eNodeType == SQL_OPERATOR_NODE &&
						isSQLCompareOp( pSQLNode->nd.op.eOperator) &&
						((pSQLNode->pFirstChild->eNodeType == SQL_COLUMN_NODE &&
						  pSQLNode->pLastChild->eNodeType == SQL_VALUE_NODE) ||
						 (pSQLNode->pFirstChild->eNodeType == SQL_VALUE_NODE &&
						  pSQLNode->pLastChild->eNodeType == SQL_COLUMN_NODE)))
			{
				eSQLQueryOperators	eOperator = pSQLNode->nd.op.eOperator;
				
				// Have a Column,Op,Value or Value,Op,Column.  Convert to a
				// predicate node and merge with other predicate nodes that
				// have already been created, if possible.
				
				if (pSQLNode->pFirstChild->eNodeType == SQL_COLUMN_NODE)
				{
					pSQLTable = pSQLNode->pFirstChild->nd.column.pSQLTable;
					uiColumnNum = pSQLNode->pFirstChild->nd.column.uiColumnNum;
					pValue = &pSQLNode->pLastChild->currVal;
				}
				else
				{
					pSQLTable = pSQLNode->pLastChild->nd.column.pSQLTable;
					uiColumnNum = pSQLNode->pLastChild->nd.column.uiColumnNum;
					pValue = &pSQLNode->pFirstChild->currVal;
					
					// Need to invert the operator in this case.
					
					switch (pSQLNode->nd.op.eOperator)
					{
						case SQL_EQ_OP:
						case SQL_NE_OP:
							// No change
							break;
						case SQL_LT_OP:
							eOperator = SQL_GE_OP;
							break;
						case SQL_LE_OP:
							eOperator = SQL_GT_OP;
							break;
						case SQL_GT_OP:
							eOperator = SQL_LE_OP;
							break;
						case SQL_GE_OP:
							eOperator = SQL_LT_OP;
							break;
						default:
							// Should never get here!
							flmAssert( 0);
							break;
					}
				}
				
				if (RC_BAD( rc = addPredicate( pSubQuery, &uiOperand,
									pSQLTable, uiColumnNum,
									eOperator, pSQLNode->nd.op.uiCompareRules,
									pSQLNode->bNotted, pValue)))
				{
					goto Exit;
				}
			}
			else
			{
				
				// Can't do anything with this operand, leave it and go to the
				// next one.
				
				uiOperand++;
			}
		}
		pSubQuery = pNextSubQuery;
	}
	
Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Evaluate operands of an AND or OR operator to see if we can
//			replace one.
//			TRUE && P1 will be replaced with P1
//			FALSE && P1 will be replaced with FALSE
//			UNKNOWN && P1 will be replaced with UNKNOWN
//			TRUE || P1 will be replaced with TRUE
//			UNKNOWN || P1 will be replaced with UNKNOWN
//			FALSE || P1 will be replaced with P1
//-------------------------------------------------------------------------
FSTATIC SQL_NODE * sqlEvalLogicalOperands(
	SQL_NODE *		pSQLNode)
{
	eSQLQueryOperators	eOperator = pSQLNode->nd.op.eOperator;
	SQL_NODE *				pChildNode;
	SQLBoolType				eChildBoolVal;
	SQLBoolType				eClipValue = (eOperator == SQL_AND_OP)
												 ? SQL_TRUE
												 : SQL_FALSE;
	SQL_NODE *				pReplacementNode = NULL;
	
	pChildNode = pSQLNode->pFirstChild;
	while (pChildNode)
	{
		if (isSQLNodeBool( pChildNode))
		{
			eChildBoolVal = pChildNode->currVal.val.eBool;
		}
		else
		{
			pChildNode = pChildNode->pNextSib;
			continue;
		}
	
		// For AND operators eClipValue will be SQL_TRUE.  For OR
		// operators, it will be SQL_FALSE.  Those nodes should all be
		// clipped out.  If, after clipping the value, there is only
		// one node left, whatever it is should be moved up to replace
		// the AND or the OR  node.
		
		if (eChildBoolVal == eClipValue)
		{
			if (pChildNode->pPrevSib)
			{
				pChildNode->pPrevSib->pNextSib = pChildNode->pNextSib;
			}
			else
			{
				pSQLNode->pFirstChild = pChildNode->pNextSib;
			}
			if (pChildNode->pNextSib)
			{
				pChildNode->pNextSib->pPrevSib = pChildNode->pPrevSib;
			}
			else
			{
				pSQLNode->pLastChild = pChildNode->pPrevSib;
			}
			if (pSQLNode->pFirstChild != pSQLNode->pLastChild)
			{
				pChildNode = pChildNode->pNextSib;
				continue;
			}
			else
			{
				pReplacementNode = pSQLNode->pFirstChild;
				break;
			}
		}
		else
		{
			
			// The child node is a a boolean value that should simply replace
			// the AND or OR operator node.  This handles the following cases:
			//	1. Value is SQL_UNKNOWN and operator is SQL_OR or SQL_AND
			// 2. Value is SQL_FALSE and operator is SQL_AND
			// 3. Value is SQL_TRUE and operator is SQL_OR.
			
			pReplacementNode = pChildNode;
			break;
		}
	}

	// If we got a replacement node, link it in where the AND or OR
	// node was.
	
	if (pReplacementNode)
	{
		SQL_NODE *	pParentNode;
		
		if ((pParentNode = pSQLNode->pParent) == NULL)
		{
			pReplacementNode->pParent = NULL;
			pReplacementNode->pPrevSib = NULL;
			pReplacementNode->pNextSib = NULL;
		}
		else
		{
			pReplacementNode->pParent = pParentNode;
			if ((pReplacementNode->pPrevSib = pSQLNode->pPrevSib) != NULL)
			{
				pReplacementNode->pPrevSib->pNextSib = pReplacementNode;
			}
			else
			{
				pParentNode->pFirstChild = pReplacementNode;
			}
			
			if ((pReplacementNode->pNextSib = pSQLNode->pNextSib) != NULL)
			{
				pReplacementNode->pNextSib->pPrevSib = pReplacementNode;
			}
			else
			{
				pParentNode->pLastChild = pReplacementNode;
			}
		}
		pSQLNode = pReplacementNode;
	}

	return( pSQLNode);
}

//-------------------------------------------------------------------------
// Desc:	Clip a NOT node out of the tree.
//-------------------------------------------------------------------------
FSTATIC SQL_NODE * sqlClipNotNode(
	SQL_NODE *	pNotNode,
	SQL_NODE **	ppExpr)
{
	SQL_NODE *	pKeepNode;

	// If this NOT node has no parent, the root
	// of the tree needs to be set to its child.

	pKeepNode = pNotNode->pFirstChild;

	// Child better not have any siblings - NOT nodes only have
	// one operand.

	flmAssert( !pKeepNode->pNextSib && !pKeepNode->pPrevSib);

	// Set child to point to the NOT node's parent.

	if ((pKeepNode->pParent = pNotNode->pParent) == NULL)
	{
		*ppExpr = pKeepNode;
	}
	else
	{

		// Link child in where the NOT node used to be.

		if ((pKeepNode->pPrevSib = pNotNode->pPrevSib) != NULL)
		{
			pKeepNode->pPrevSib->pNextSib = pKeepNode;
		}
		else
		{
			pKeepNode->pParent->pFirstChild = pKeepNode;
		}
		if ((pKeepNode->pNextSib = pNotNode->pNextSib) != NULL)
		{
			pKeepNode->pNextSib->pPrevSib = pKeepNode;
		}
		else
		{
			pKeepNode->pParent->pLastChild = pKeepNode;
		}
	}
	return( pKeepNode);
}

//-------------------------------------------------------------------------
// Desc:	
//-------------------------------------------------------------------------
RCODE SQLQuery::getNext(
	F_Row **	ppRow)
{
	RCODE			rc = NE_SFLM_OK;
   // JMC - FIXME: Implement this!
   return ( rc);
}

//-------------------------------------------------------------------------
// Desc:	
//-------------------------------------------------------------------------
RCODE SQLQuery::getPrev(
	F_Row **	ppRow)
{
	RCODE			rc = NE_SFLM_OK;
   // JMC - FIXME: Implement this!
   return ( rc);
}
	
//-------------------------------------------------------------------------
// Desc:	
//-------------------------------------------------------------------------
RCODE SQLQuery::getFirst(
	F_Row **	ppRow)
{
	RCODE			rc = NE_SFLM_OK;
   // JMC - FIXME: Implement this!
   return ( rc);
}
	
//-------------------------------------------------------------------------
// Desc:	
//-------------------------------------------------------------------------
RCODE SQLQuery::getLast(
	F_Row **	ppRow)
{
	RCODE			rc = NE_SFLM_OK;
   // JMC - FIXME: Implement this!
   return ( rc);
}

//-------------------------------------------------------------------------
// Desc:	Reduce the query tree.  This will strip out NOT nodes and
//			resolve constant expressions to a single node.  It also weeds
//			out all boolean constants that are operands of AND or OR operators.
//			Finally, if the bFlattenTree parameter is TRUE, it will coalesce
//			AND and OR nodes so that they can have multiple operands.
//-------------------------------------------------------------------------
RCODE SQLQuery::reduceTree(
	FLMBOOL	bFlattenTree)
{
	RCODE						rc = NE_SFLM_OK;
	SQL_NODE *				pSQLNode = m_pQuery;
	SQL_NODE *				pTmpNode;
	SQL_NODE *				pParentNode = NULL;
	eSQLNodeTypes			eNodeType;
	eSQLQueryOperators	eOperator;
	FLMBOOL					bNotted = FALSE;

	for (;;)
	{
		eNodeType = pSQLNode->eNodeType;

		// Need to save bNotted on each node so that when we traverse
		// back up the tree it can be reset properly.  If bNotted is
		// TRUE and pSQLNode is an operator, we may change the operator in
		// some cases.  Even if we change the operator, we still want to
		// set the bNotted flag because it also implies "for every" when set
		// to TRUE, and we need to remember that as well.

		pSQLNode->bNotted = bNotted;
		if (eNodeType == SQL_OPERATOR_NODE)
		{
			eOperator = pSQLNode->nd.op.eOperator;
			if (eOperator == SQL_AND_OP || eOperator == SQL_OR_OP)
			{
				// AND and OR nodes better have child nodes
				
				if (!pSQLNode->pFirstChild || !pSQLNode->pLastChild)
				{
					flmAssert( 0);
					rc = RC_SET( NE_SFLM_Q_ILLEGAL_OPERAND);
					goto Exit;
				}
				if (bNotted)
				{
					eOperator = (eOperator == SQL_AND_OP
									 ? SQL_OR_OP
									 : SQL_AND_OP);
					pSQLNode->nd.op.eOperator = eOperator;
				}
				if (pParentNode)
				{
					
					// Logical sub-expressions can only be operands of
					// AND, OR, or NOT operators.

					if (!isSQLLogicalOp( pParentNode->nd.op.eOperator))
					{
						rc = RC_SET( NE_SFLM_Q_ILLEGAL_OPERAND);
						goto Exit;
					}
					if (bFlattenTree && pParentNode->nd.op.eOperator == eOperator)
					{
						
						// Move all of pSQLNode's children become the immediate
						// children of pParentNode.
						
						pTmpNode = pSQLNode->pFirstChild;
						while (pTmpNode)
						{
							pTmpNode->pParent = pParentNode;
							pTmpNode = pTmpNode->pNextSib;
						}
						
						if (pSQLNode->pPrevSib)
						{
							pSQLNode->pPrevSib->pNextSib = pSQLNode->pFirstChild;
							pSQLNode->pFirstChild->pPrevSib = pSQLNode->pPrevSib;
						}
						if (pSQLNode->pNextSib)
						{
							pSQLNode->pNextSib->pPrevSib = pSQLNode->pLastChild;
							pSQLNode->pLastChild->pNextSib = pSQLNode->pNextSib;
						}
						
						// Continue processing from pSQLNode's first child, which
						// is the beginning of the list of nodes we just replaced
						// pSQLNode with.
						
						pSQLNode = pSQLNode->pFirstChild;
						continue;
					}
				}
			}
			else if (eOperator == SQL_NOT_OP)
			{

				// Logical sub-expressions can only be operands of
				// AND, OR, or NOT operators.

				if (pParentNode)
				{
					if (!isSQLLogicalOp( pParentNode->nd.op.eOperator))
					{
						rc = RC_SET( NE_SFLM_Q_ILLEGAL_OPERAND);
						goto Exit;
					}
				}
				bNotted = !bNotted;

				// Clip NOT nodes out of the tree.

				pSQLNode = sqlClipNotNode( pSQLNode, &m_pQuery);
				pParentNode = pSQLNode->pParent;
				continue;
			}
			else if (isSQLCompareOp( eOperator))
			{

				// Comparison sub-expressions can only be operands of
				// AND, OR, or NOT operators.

				if (pParentNode)
				{
					if (!isSQLLogicalOp( pParentNode->nd.op.eOperator))
					{
						rc = RC_SET( NE_SFLM_Q_ILLEGAL_OPERAND);
						goto Exit;
					}
				}
				if (bNotted)
				{
					switch (eOperator)
					{
						case SQL_EQ_OP:
							eOperator = SQL_NE_OP;
							break;
						case SQL_NE_OP:
							eOperator = SQL_EQ_OP;
							break;
						case SQL_LT_OP:
							eOperator = SQL_GE_OP;
							break;
						case SQL_LE_OP:
							eOperator = SQL_GT_OP;
							break;
						case SQL_GT_OP:
							eOperator = SQL_LE_OP;
							break;
						case SQL_GE_OP:
							eOperator = SQL_LT_OP;
							break;
						default:

							// Don't change the other operators.
							// Will just use the bNotted flag when
							// evaluating.

							break;
					}
					pSQLNode->nd.op.eOperator = eOperator;
				}
			}
			else
			{

				// Better be an arithmetic operator we are dealing with
				// at this point.

				flmAssert( isSQLArithOp( eOperator));

				// Arithmetic sub-expressions can only be operands
				// of arithmetic or comparison operators

				if (pParentNode)
				{
					if (!isSQLCompareOp( pParentNode->nd.op.eOperator) &&
						 !isSQLArithOp( pParentNode->nd.op.eOperator))
					{
						rc = RC_SET( NE_SFLM_Q_ILLEGAL_OPERAND);
						goto Exit;
					}
				}
			}
		}
		else if (eNodeType == SQL_COLUMN_NODE)
		{
			flmAssert( !pSQLNode->pFirstChild);
		}
		else
		{
			flmAssert( eNodeType == SQL_VALUE_NODE);

			// If bNotted is TRUE and we have a boolean value, change
			// the value: FALSE ==> TRUE, TRUE ==> FALSE.

			if (bNotted && pSQLNode->currVal.eValType == SQL_BOOL_VAL)
			{
				if (pSQLNode->currVal.val.eBool == SQL_TRUE)
				{
					pSQLNode->currVal.val.eBool = SQL_FALSE;
				}
				else if (pSQLNode->currVal.val.eBool == SQL_FALSE)
				{
					pSQLNode->currVal.val.eBool = SQL_TRUE;
				}
			}

			// Values can only be operands of arithmetic or comparison operators,
			// unless they are boolean values, in which case they can only be
			// operands of logical operators.

			if (pParentNode)
			{
				if (pSQLNode->currVal.eValType == SQL_BOOL_VAL)
				{
					if (!isSQLLogicalOp( pParentNode->nd.op.eOperator))
					{
						rc = RC_SET( NE_SFLM_Q_ILLEGAL_OPERAND);
						goto Exit;
					}
				}
				else
				{
					if (!isSQLCompareOp( pParentNode->nd.op.eOperator) &&
						 !isSQLArithOp( pParentNode->nd.op.eOperator))
					{
						rc = RC_SET( NE_SFLM_Q_ILLEGAL_OPERAND);
						goto Exit;
					}
				}
			}

			// A value node should not have any children

			flmAssert( !pSQLNode->pFirstChild);
		}

		// Do traversal to child node, if any

		if (pSQLNode->pFirstChild)
		{
			pParentNode = pSQLNode;
			pSQLNode = pSQLNode->pFirstChild;
			continue;
		}

		// Go back up the tree until we hit something that has
		// a sibling.

		while (!pSQLNode->pNextSib)
		{

			// If there are no more parents, we are done.

			if ((pSQLNode = pSQLNode->pParent) == NULL)
			{
				goto Exit;
			}

			flmAssert( pSQLNode->eNodeType == SQL_OPERATOR_NODE);

			// Evaluate arithmetic expressions if both operands are
			// constants.

			if (isSQLArithOp( pSQLNode->nd.op.eOperator) &&
				 pSQLNode->pFirstChild->eNodeType == SQL_VALUE_NODE &&
				 pSQLNode->pLastChild->eNodeType == SQL_VALUE_NODE)
			{
				if (RC_BAD( rc = sqlEvalArithOperator(
											&pSQLNode->pFirstChild->currVal,
											&pSQLNode->pLastChild->currVal,
											pSQLNode->nd.op.eOperator,
											&pSQLNode->currVal)))
				{
					goto Exit;
				}
				pSQLNode->eNodeType = SQL_VALUE_NODE;
				pSQLNode->currVal.uiFlags = SQL_VAL_IS_CONSTANT;
				pSQLNode->pFirstChild = NULL;
				pSQLNode->pLastChild = NULL;
			}
			else
			{

				// For the AND and OR operators, check the operands to
				// see if they are boolean values.  Boolean values can
				// be weeded out of the criteria as we go back up the
				// tree.

				if (pSQLNode->nd.op.eOperator == SQL_OR_OP ||
					 pSQLNode->nd.op.eOperator == SQL_AND_OP)
				{
					pSQLNode = sqlEvalLogicalOperands( pSQLNode);
					if (!pSQLNode->pParent)
					{
						m_pQuery = pSQLNode;
					}
				}
			}

			pParentNode = pSQLNode->pParent;
		}

		// pSQLNode will NEVER be NULL if we get here, because we
		// will jump to Exit in those cases.

		pSQLNode = pSQLNode->pNextSib;

		// Need to reset the bNotted flag to what it would have
		// been as we traverse back up the tree.

		bNotted = pParentNode->bNotted;
	}

Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Allocate and set up a DNF node.
//-------------------------------------------------------------------------
FSTATIC RCODE createDNFNode(
	F_Pool *				pPool,
	SQL_DNF_NODE *		pParentDNFNode,
	SQL_DNF_NODE **	ppDNFNode,
	SQL_NODE *			pNode)
{
	RCODE				rc = NE_SFLM_OK;
	SQL_DNF_NODE *	pDNFNode;
	
	if (RC_BAD( rc = pPool->poolCalloc( sizeof( SQL_DNF_NODE),
												(void **)&pDNFNode)))
	{
		goto Exit;
	}
	
	// pDNFNode->pNode will be NULL if it is an AND or OR operator.
	
	if (pNode->eNodeType == SQL_OPERATOR_NODE)
	{
		if (pNode->nd.op.eOperator == SQL_AND_OP)
		{
			pDNFNode->bAndOp = TRUE;
		}
		else if (pNode->nd.op.eOperator == SQL_OR_OP)
		{
			// No need to really set as it is already 0 from poolCalloc.
			// pDNFNode->bAndOp = FALSE;
		}
		else
		{
			pDNFNode->pNode = pNode;
		}
	}
	else
	{
		pDNFNode->pNode = pNode;
	}
	if ((pDNFNode->pParent = pParentDNFNode) != NULL)
	{
		if ((pDNFNode->pPrevSib = pParentDNFNode->pLastChild) != NULL)
		{
			pDNFNode->pPrevSib->pNextSib = pDNFNode;
		}
		else
		{
			pParentDNFNode->pFirstChild = pDNFNode;
		}
		pParentDNFNode->pLastChild = pDNFNode;
	}
	*ppDNFNode = pDNFNode;
	
Exit:

	return( rc);
}
	
//-------------------------------------------------------------------------
// Desc:	Copy the sub-tree pointed to by pSrcSubTree and then link the
//			new sub-tree as the last child of pParentNode.
//-------------------------------------------------------------------------
FSTATIC RCODE copyAndLinkSubTree(
	F_Pool *				pPool,
	SQL_DNF_NODE *		pSrcSubTree,
	SQL_DNF_NODE *		pParentNode)
{
	RCODE					rc = NE_SFLM_OK;
	SQL_DNF_NODE *		pNewSubTree = NULL;
	SQL_DNF_NODE *		pCurrDestParentNode = NULL;
	SQL_DNF_NODE *		pCurrSrcNode = pSrcSubTree;
	SQL_DNF_NODE *		pNewDestNode = NULL;
	
	for (;;)
	{
		if (RC_BAD( rc = pPool->poolCalloc( sizeof( SQL_DNF_NODE),
													(void **)&pNewDestNode)))
		{
			goto Exit;
		}
		pNewDestNode->pNode = pCurrSrcNode->pNode;
		pNewDestNode->bAndOp = pCurrSrcNode->bAndOp;
		if (!pNewSubTree)
		{
			pNewSubTree = pNewDestNode;
		}
		else
		{
			pNewDestNode->pParent = pCurrDestParentNode;
			if ((pNewDestNode->pPrevSib = pCurrDestParentNode->pLastChild) != NULL)
			{
				pNewDestNode->pPrevSib->pNextSib = pNewDestNode;
			}
			else
			{
				pCurrDestParentNode->pFirstChild = pNewDestNode;
			}
			pCurrDestParentNode->pLastChild = pNewDestNode;
		}
		
		// Try to go down to a child node
		
		if (pCurrSrcNode->pFirstChild)
		{
			pCurrSrcNode = pCurrSrcNode->pFirstChild;
			pCurrDestParentNode = pNewDestNode;
			continue;
		}
		
		// No child nodes, go back up parent chain until we find one that
		// has a sibling.
		
		for (;;)
		{
			if (pCurrSrcNode == pSrcSubTree)
			{
				break;
			}
			if (pCurrSrcNode->pNextSib)
			{
				break;
			}
			pCurrSrcNode = pCurrSrcNode->pParent;
			pCurrDestParentNode = pCurrDestParentNode->pParent;
		}
		if (pCurrSrcNode == pSrcSubTree)
		{
			break;
		}
		pCurrSrcNode = pCurrSrcNode->pNextSib;
	}
	
	// Link the newly created sub-tree to the passed in parent node as its
	// last child.
	
	flmAssert( pNewSubTree);
	pNewSubTree->pParent = pParentNode;
	pNewSubTree->pNextSib = NULL;
	if ((pNewSubTree->pPrevSib = pParentNode->pLastChild) != NULL)
	{
		pNewSubTree->pPrevSib->pNextSib = pNewSubTree;
	}
	else
	{
		pParentNode->pFirstChild = pNewSubTree;
	}
	pParentNode->pLastChild = pNewSubTree;
	
Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Distribute an AND operator over an OR operator.  The AND operator
//			is the parent node of the passed in pOldOrNode.  A new list of
//			AND nodes is created which will replace the original AND node in
//			the tree.
//-------------------------------------------------------------------------
FSTATIC RCODE distributeAndOverOr(
	F_Pool *				pPool,
	SQL_DNF_NODE *		pOldOrNode,
	SQL_DNF_NODE **	ppDNFTree)
{
	RCODE				rc = NE_SFLM_OK;
	SQL_DNF_NODE *	pOldAndNode;
	SQL_DNF_NODE *	pOldAndParentNode;
	SQL_DNF_NODE *	pNewAndNode;
	SQL_DNF_NODE *	pFirstNewAndNode;
	SQL_DNF_NODE *	pLastNewAndNode;
	SQL_DNF_NODE *	pOrChildNode;
	SQL_DNF_NODE *	pAndChildNode;

	// Parent node to pOldOrNode better be an AND node.
	
	pOldAndNode = pOldOrNode->pParent;
	flmAssert( !pOldAndNode->pNode && pOldAndNode->bAndOp);
				
	// Distribute ALL of the AND node's children (except this OR node)
	// across ALL of the OR node's children
	
	pFirstNewAndNode = NULL;
	pLastNewAndNode = NULL;
	pOrChildNode = pOldOrNode->pFirstChild;
	while (pOrChildNode)
	{
		if (RC_BAD( rc = pPool->poolCalloc( sizeof( SQL_DNF_NODE),
													(void **)&pNewAndNode)))
		{
			goto Exit;
		}
		pNewAndNode->bAndOp = TRUE;
		if ((pNewAndNode->pPrevSib = pLastNewAndNode) != NULL)
		{
			pLastNewAndNode->pNextSib = pNewAndNode;
		}
		else
		{
			pFirstNewAndNode = pNewAndNode;
		}
		pLastNewAndNode = pNewAndNode;
		
		// Copy all of the old AND node's children, except for this
		// OR node as children of the new AND node.
		
		pAndChildNode = pOldAndNode->pFirstChild;
		while (pAndChildNode)
		{
			if (pAndChildNode != pOldOrNode)
			{
				
				if (RC_BAD( rc = copyAndLinkSubTree( pPool, pAndChildNode, pNewAndNode)))
				{
					goto Exit;
				}
			}
			pAndChildNode = pAndChildNode->pNextSib;
		}
		
		// Copy the entire sub-tree of pOrChildNode and link it as the last
		// child of the new AND node.
		
		if (RC_BAD( rc = copyAndLinkSubTree( pPool, pOrChildNode, pNewAndNode)))
		{
			goto Exit;
		}
		pOrChildNode = pOrChildNode->pNextSib;
	}
	
	// Link the newly created AND list in where the old
	// AND node was (pOldAndNode).  If it was at the root
	// of the tree, we will need to create a new OR root.
	
	if ((pOldAndParentNode = pOldAndNode->pParent) == NULL)
	{
		if (RC_BAD( rc = pPool->poolCalloc( sizeof( SQL_DNF_NODE),
													(void **)&pOldAndParentNode)))
		{
			goto Exit;
		}
		
		// NOTE: No need to set anything in this new node, we want it to be
		// an OR node, which means that bAndOp is FALSE and pNode is NULL - both
		// of which are set by the poolCalloc.
		
		*ppDNFTree = pOldAndParentNode;
	}
	
	// Point all of the new AND nodes to the parent of the old AND node.
	
	pAndChildNode = pFirstNewAndNode;
	while (pAndChildNode)
	{
		pAndChildNode->pParent = pOldAndParentNode;
		pAndChildNode = pAndChildNode->pNextSib;
	}
	
	// Link the new list of AND nodes where the old AND node was.
	// Although the old AND node is still allocated, it is no longer
	// pointed to from the tree.
	
	if ((pFirstNewAndNode->pPrevSib = pOldAndNode->pPrevSib) != NULL)
	{
		pFirstNewAndNode->pPrevSib->pNextSib = pFirstNewAndNode;
	}
	else
	{
		pOldAndParentNode->pFirstChild = pFirstNewAndNode;
	}
	if ((pLastNewAndNode->pNextSib = pOldAndNode->pNextSib) != NULL)
	{
		pLastNewAndNode->pNextSib->pPrevSib = pLastNewAndNode;
	}
	else
	{
		pOldAndParentNode->pLastChild = pLastNewAndNode;
	}
	
Exit:

	return( rc);
}
				
//-------------------------------------------------------------------------
// Desc:	Convert query tree to disjunctive normal form (DNF).  Result is
//			a list of sub-queries that are ORed together.
//-------------------------------------------------------------------------
RCODE SQLQuery::convertToDNF( void)
{
	RCODE				rc = NE_SFLM_OK;
	SQL_NODE *		pCurrNode;
	SQL_DNF_NODE *	pParentDNFNode;
	SQL_DNF_NODE *	pCurrDNFNode;
	SQL_DNF_NODE *	pDNFTree;
	SQL_DNF_NODE *	pAndList;
	SQL_DNF_NODE *	pExprList;
	F_Pool			pool;
	SQL_SUBQUERY *	pSubQuery;
	FLMUINT			uiLoop;
	
	pool.poolInit( 1024); 
	
	// If the top node in the tree is not an AND or OR operator,
	// create a single subquery that has a single operand.
	
	if (m_pQuery->eNodeType != SQL_OPERATOR_NODE ||
		 (m_pQuery->nd.op.eOperator != SQL_AND_OP &&
		  m_pQuery->nd.op.eOperator != SQL_OR_OP))
	{
		if (RC_BAD( rc = m_pool.poolCalloc( sizeof( SQL_SUBQUERY),
													(void **)&m_pFirstSubQuery)))
		{
			goto Exit;
		}
		m_pLastSubQuery = m_pFirstSubQuery;
		if (RC_BAD( rc = m_pool.poolCalloc( sizeof( SQL_NODE *),
												(void **)&m_pFirstSubQuery->ppOperands)))
		{
			goto Exit;
		}
		m_pFirstSubQuery->uiOperandCount = 1;
		m_pFirstSubQuery->ppOperands [0] = m_pQuery;
		goto Exit;
	}
	
	// Create the tree of DNF nodes to point to all of the AND and OR nodes
	// in the tree and their immediate child nodes.

	pCurrNode = m_pQuery;
	pParentDNFNode = NULL;
	pDNFTree = NULL;
	for (;;)
	{
		if (RC_BAD( rc = createDNFNode( &pool, pParentDNFNode,
									&pCurrDNFNode, pCurrNode)))
		{
			goto Exit;
		}
		if (!pDNFTree)
		{
			pDNFTree = pCurrDNFNode;
		}
		
		// Don't traverse down to child nodes if it is not an AND or OR node.
		
		if (pCurrNode->eNodeType == SQL_OPERATOR_NODE &&
			 (pCurrNode->nd.op.eOperator == SQL_AND_OP ||
			  pCurrNode->nd.op.eOperator == SQL_OR_OP))
		{
			if (pCurrNode->pFirstChild)
			{
				pCurrNode = pCurrNode->pFirstChild;
				pParentDNFNode = pCurrDNFNode;
				continue;
			}
		}
		
		// Go back up to parent until we find one that has a sibling.
		
		while (!pCurrNode->pNextSib)
		{
			if ((pCurrNode = pCurrNode->pParent) == NULL)
			{
				break;
			}
			pParentDNFNode = pParentDNFNode->pParent;
		}
		if (!pCurrNode)
		{
			break;
		}
		pCurrNode = pCurrNode->pNextSib;
	}
	
	// Now traverse the DNF tree and move all OR operators to the top.
	// When we are done we should have a DNF tree with either a single AND
	// node and a list of subordinate expressions, or a single OR node with
	// a mix of AND child nodes or non-AND expressions.

	pCurrDNFNode = pDNFTree;	
	for (;;)
	{
		
		// If we hit an OR node that is not the root node, it's parent should be
		// an AND node.  Distribute the AND node's operands over all of the
		// OR node's operands.
			
		if (pCurrDNFNode->pNode->eNodeType == SQL_OPERATOR_NODE &&
			 pCurrDNFNode->pNode->nd.op.eOperator == SQL_OR_OP &&
			 pCurrDNFNode->pParent)
		{
			if (RC_BAD( rc = distributeAndOverOr( &pool, pCurrDNFNode,
											&pDNFTree)))
			{
				goto Exit;
			}
			
			// Start over at the top of the tree.
			
			pCurrDNFNode = pDNFTree;
			continue;
		}
		
		// Go to first child, if there is one.
		
		if (pCurrDNFNode->pFirstChild)
		{
			pCurrDNFNode = pCurrDNFNode->pFirstChild;
			continue;
		}
		
		// No child nodes, go to sibling nodes.  If no sibling nodes,
		// traverse back up parent chain until we find one.
		
		while (!pCurrDNFNode->pNextSib)
		{
			if ((pCurrDNFNode = pCurrDNFNode->pParent) == NULL)
			{
				break;
			}
		}
		if (!pCurrDNFNode)
		{
			break;
		}
		pCurrDNFNode = pCurrDNFNode->pNextSib;
	}
	
	// If we get to this point, we have created a DNF tree that either
	// as an OR at the top or an AND at the top.  If it is an OR at the
	// top, we have multiple sub-queries.  If it is an AND at the top, we
	// have a single sub-query.
	
	if (pDNFTree->bAndOp)
	{
		pAndList = pDNFTree;
	}
	else
	{
		pAndList = pDNFTree->pFirstChild;
		flmAssert( pAndList);
	}
	
	while (pAndList)
	{
		if (RC_BAD( rc = m_pool.poolCalloc( sizeof( SQL_SUBQUERY),
													(void **)&pSubQuery)))
		{
			goto Exit;
		}
		
		// Link the subquery as the last sub-query in our sub-query list
		
		if ((pSubQuery->pPrev = m_pLastSubQuery) != NULL)
		{
			pSubQuery->pPrev->pNext = pSubQuery;
		}
		else
		{
			m_pFirstSubQuery = pSubQuery;
		}
		m_pLastSubQuery = pSubQuery;

		// The child may be a simple expression, in which case it is its
		// own sub-query.

		if (pAndList->pNode)
		{
			pSubQuery->uiOperandCount = 1;
			
			// The expression should not have any child nodes.
			
			flmAssert( !pExprList->pFirstChild && !pExprList->pLastChild);
			if (RC_BAD( rc = m_pool.poolCalloc( sizeof( SQL_NODE *),
													(void **)&pSubQuery->ppOperands)))
			{
				goto Exit;
			}
			pSubQuery->ppOperands [0] = pAndList->pNode;
			
			// NULL out the node's parent pointer and sibling pointers - just
			// to keep things tidy.
			
			pAndList->pNode->pParent = NULL;
			pAndList->pNode->pNextSib = NULL;
			pAndList->pNode->pPrevSib = NULL;
		}
		else
		{
			
			// Count the expressions in the list - should be at least one.
			
			pExprList = pAndList->pFirstChild;
			flmAssert( pExprList);
			while (pExprList)
			{
				
				// All of the expressions should point to nodes in the query
				// tree, and should  not be AND or OR nodes.  Furthermore,
				// they should not have child nodes
				
				flmAssert( pExprList->pNode && !pExprList->pFirstChild &&
								!pExprList->pLastChild);
				pSubQuery->uiOperandCount++;
				pExprList = pExprList->pNextSib;
			}
			if (RC_BAD( rc = m_pool.poolCalloc( sizeof( SQL_NODE *) * pSubQuery->uiOperandCount,
													(void **)&pSubQuery->ppOperands)))
			{
				goto Exit;
			}
			
			// Set the pointers in the operand list for the sub-query.
			
			for (uiLoop = 0, pExprList = pAndList->pFirstChild;
				  pExprList;
				  uiLoop++, pExprList = pExprList->pNextSib)
			{
				pSubQuery->ppOperands [uiLoop] = pExprList->pNode;
			}
		}
		flmAssert( uiLoop == pSubQuery->uiOperandCount);
		pAndList = pAndList->pNextSib;
	}
	
Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc: Determine if a particular predicate is associated with the
//			specified table.  Only return TRUE if the predicate is associated
//			with this table and only with this table.
//-------------------------------------------------------------------------
FSTATIC FLMBOOL predIsForOnlyThisTable(
	SQL_NODE *	pPredRootNode,
	SQL_TABLE *	pSQLTable)
{
	FLMBOOL		bIsAssociated = FALSE;
	SQL_NODE *	pCurrNode = pPredRootNode;
	
	for (;;)
	{
		if (pCurrNode->eNodeType == SQL_COLUMN_NODE)
		{
			 if (pCurrNode->nd.column.pSQLTable == pSQLTable)
			 {
				 bIsAssociated = TRUE;
			 }
			 else
			 {
				 
				 // Predicate is associated with more than the table that
				 // was passed in.
				 
				 bIsAssociated = FALSE;
				 break;
			 }
		}
		
		if (pCurrNode->pFirstChild)
		{
			pCurrNode = pCurrNode->pFirstChild;
			continue;
		}
		
		// No child nodes, traverse to sibling - or sibling of first node
		// in the parent chain that has a next sibling.
		
		for (;;)
		{
			if (pCurrNode == pPredRootNode)
			{
				break;
			}
			if (pCurrNode->pNextSib)
			{
				break;
			}
			pCurrNode = pCurrNode->pParent;
		}
		if (pCurrNode == pPredRootNode)
		{
			break;
		}
		
		// If we get to here, there should be a next sibling.
		
		pCurrNode = pCurrNode->pNextSib;
		flmAssert( pCurrNode);
	}
	
	return( bIsAssociated);
}

//-------------------------------------------------------------------------
// Desc:	Associate a predicate with all of the indexes it pertains to
// 		with respect to a particular table.
//-------------------------------------------------------------------------
RCODE SQLQuery::getPredKeys(
	F_TABLE *		pTable,
	FLMUINT			uiForceIndexNum,
	SQL_PRED *		pPred,
	SQL_INDEX **	ppFirstSQLIndex,
	SQL_INDEX **	ppLastSQLIndex)
{
	RCODE				rc = NE_SFLM_OK;
	ICD *				pIcd;
	SQL_INDEX *		pSQLIndex;
	SQL_KEY *		pKey;
	F_COLUMN *		pColumn = m_pDb->m_pDict->getColumn( pTable, pPred->uiColumnNum);
	F_INDEX *		pIndex;
	FLMUINT			uiKeyComponent;

	// This ICD chain will only contain ICDs for this particular column on
	// the table the column belongs to.

	for (pIcd = pColumn->pFirstIcd; pIcd; pIcd = pIcd->pNextInChain)
	{
		
		// If the table has an index specified for it, skip this ICD if
		// it is not that index.
		
		if (uiForceIndexNum && uiForceIndexNum != pIcd->uiIndexNum)
		{
			continue;
		}
		pIndex = m_pDb->m_pDict->getIndex( pIcd->uiIndexNum);

		// Cannot use the index if it is off-line.

		if (pIndex->uiFlags & (IXD_OFFLINE | IXD_SUSPENDED))
		{
			continue;
		}
		
		// Find the index off of the table.  If not there, add it.
		
		pSQLIndex = *ppFirstSQLIndex;
		while (pSQLIndex->uiIndexNum != pIcd->uiIndexNum)
		{
			pSQLIndex = pSQLIndex->pNext;
		}
		if (!pSQLIndex)
		{
			if (RC_BAD( rc = m_pool.poolCalloc( sizeof( SQL_INDEX),
												(void **)&pSQLIndex)))
			{
				goto Exit;
			}
			pSQLIndex->uiIndexNum = pIcd->uiIndexNum;
			if ((pSQLIndex->pPrev = *ppLastSQLIndex) != NULL)
			{
				pSQLIndex->pPrev->pNext = pSQLIndex;
			}
			else
			{
				*ppFirstSQLIndex = pSQLIndex;
			}
			*ppLastSQLIndex = pSQLIndex;
			
			// Allocate a single key for the index.
			
			if (RC_BAD( rc = m_pool.poolCalloc( sizeof( SQL_KEY),
												(void **)&pKey)))
			{
				goto Exit;
			}
			pSQLIndex->pLastSQLKey = pSQLIndex->pFirstSQLKey = pKey;
			
			// Allocate an array of key components for the key.
			
			if (RC_BAD( rc = m_pool.poolCalloc(
									sizeof( SQL_PRED *) * pIndex->uiNumKeyComponents,
									(void **)&pKey->ppPredicates)))
			{
				goto Exit;
			}
		}
		else
		{
			pKey = pSQLIndex->pFirstSQLKey;
		}
		
		// There should not be multiple predicates in a sub-query that
		// have the same column, so this key component should NOT already
		// be populated.
		
		uiKeyComponent = (FLMUINT)(pIcd - pIndex->pKeyIcds); 
		flmAssert( !pKey->ppPredicates [uiKeyComponent]);
		pKey->ppPredicates [uiKeyComponent] = pPred;
		
		// NOTE: Costs will be calculated later.
		
	}
	
Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Determine the order in which to evaluate indexes.  Those that have
//			a primary key will be given preference over those that don't.
//-------------------------------------------------------------------------
FSTATIC void rankIndexes(
	F_Db *			pDb,
	SQL_INDEX **	ppFirstSQLIndex,
	SQL_INDEX **	ppLastSQLIndex)
{
	SQL_INDEX *	pSQLIndex;
	SQL_INDEX *	pPrevSQLIndex;
	SQL_INDEX *	pNextSQLIndex;
	SQL_KEY *	pKey;
	F_INDEX *	pIndex;
	
	pSQLIndex = *ppFirstSQLIndex;
	while (pSQLIndex)
	{
		pNextSQLIndex = pSQLIndex->pNext;
		pPrevSQLIndex = pSQLIndex->pPrev;
		
		// There should only be one key off of the index right now.
		
		pKey = pSQLIndex->pFirstSQLKey;
		flmAssert( !pKey->pNext);
		
		// Determine how many of the key's components point to a
		// predicate.  This stops at the first NULL pointer.  There may
		// be pointers after that one, but we really don't care, because
		// we won't use those components to generate a key.
		
		pIndex = pDb->getDict()->getIndex( pSQLIndex->uiIndexNum);
		pKey->uiComponentsUsed = 0;
		while (pKey->uiComponentsUsed < pIndex->uiNumKeyComponents &&
				 pKey->ppPredicates [pKey->uiComponentsUsed])
		{
			pKey->uiComponentsUsed++;
		}
		
		// See if this key is using more components that the key for
		// prior indexes.
		
		while (pPrevSQLIndex)
		{
			if (pKey->uiComponentsUsed > pPrevSQLIndex->pFirstSQLKey->uiComponentsUsed)
			{
				// Move our current key up in front of the previous key - meaning
				// it will be evaluated ahead of that key.
				
				// First, unlink the index from its current spot.  pSQLIndex->pPrev
				// must be non-NULL - otherwise, we wouldn't have a pPrevSQLIndex.
				
				flmAssert( pSQLIndex->pPrev);
				pSQLIndex->pPrev->pNext = pSQLIndex->pNext;
				if (pSQLIndex->pNext)
				{
					pSQLIndex->pNext->pPrev = pSQLIndex->pPrev;
				}
				else
				{
					*ppLastSQLIndex = pSQLIndex->pPrev;
				}
				
				// Now, link it in front of pPrevSQLIndex
				
				pSQLIndex->pNext = pPrevSQLIndex;
				if ((pSQLIndex->pPrev = pPrevSQLIndex->pPrev) != NULL)
				{
					pSQLIndex->pPrev->pNext = pSQLIndex;
				}
				else
				{
					*ppFirstSQLIndex = pSQLIndex;
				}
				pPrevSQLIndex->pPrev = pSQLIndex;
				pPrevSQLIndex = pSQLIndex->pPrev;
			}
			else
			{
				pPrevSQLIndex = pPrevSQLIndex->pPrev;
			}
		}
		
		pSQLIndex = pNextSQLIndex;
	}
}

//-------------------------------------------------------------------------
// Desc:	Choose the best index for a table of the indexes for which we have
//			generated predicate keys.  All other indexes for the table will
//			be freed
//-------------------------------------------------------------------------
RCODE SQLQuery::chooseBestIndex(
	F_TABLE *		pTable,
	SQL_INDEX *		pFirstSQLIndex,
	SQL_INDEX **	ppBestSQLIndex
	)
{
	RCODE					rc = NE_SFLM_OK;
	SQL_INDEX *			pBestSQLIndex = NULL;
	SQL_KEY *			pSQLKey;
	FSIndexCursor *	pFSIndexCursor = NULL;
	F_INDEX *			pIndex;
	
	while (pFirstSQLIndex)
	{
		
		// Should only be one key on each index at this point.
		
		flmAssert( pFirstSQLIndex->pFirstSQLKey &&
						pFirstSQLIndex->pFirstSQLKey == pFirstSQLIndex->pLastSQLKey);
		
		pSQLKey = pFirstSQLIndex->pFirstSQLKey;
		
		pIndex = m_pDb->m_pDict->getIndex( pFirstSQLIndex->uiIndexNum);
		flmAssert( pIndex);
		
		// Allocate an index cursor, if necessary.
		
		if (!pFSIndexCursor)
		{
			if ((pFSIndexCursor = f_new FSIndexCursor) == NULL)
			{
				rc = RC_SET( NE_SFLM_MEM);
				goto Exit;
			}
		}
		else
		{
			pFSIndexCursor->resetCursor();
		}
		
		// Setup from and until keys and calculate the cost.
	
		if (RC_BAD( rc = pFSIndexCursor->setupKeys( m_pDb, pIndex, pTable,
									pSQLKey->ppPredicates)))
		{
			goto Exit;
		}
		
		// See if this index has a lower cost than our current best index.
		
		if (!pBestSQLIndex ||
			 pFSIndexCursor->getCost() < pBestSQLIndex->pFirstSQLKey->pFSIndexCursor->getCost())
		{
			pFirstSQLIndex->pFirstSQLKey->pFSIndexCursor = pFSIndexCursor;
			
			// If we have a best index, keep its index cursor so we can just
			// reset it in the loop above rather than having to free it and
			// allocate another one - a little optimization.
			
			if (!pBestSQLIndex)
			{
				pFSIndexCursor = NULL;
			}
			else
			{
				pFSIndexCursor = pBestSQLIndex->pFirstSQLKey->pFSIndexCursor;
			}
			pBestSQLIndex = pFirstSQLIndex;
			
			// If our best index's cost is low enough, no need to check any other
			// indexes.
			
			if (pFSIndexCursor->getCost() < MINIMUM_COST_ESTIMATE)
			{
				break;
			}
		}
		
		pFirstSQLIndex = pFirstSQLIndex->pNext;
	}

Exit:

	if (pFSIndexCursor)
	{
		pFSIndexCursor->Release();
	}
	*ppBestSQLIndex = pBestSQLIndex;

	return( rc);
}
		
//-------------------------------------------------------------------------
// Desc:	Merge keys from pSQLIndex into the list of indexes for pSQLTable.
//-------------------------------------------------------------------------
RCODE SQLQuery::mergeKeys(
	SQL_TABLE *	pSQLTable,
	SQL_INDEX *	pSQLIndex)
{
	RCODE			rc = NE_SFLM_OK;
	SQL_INDEX *	pFoundSQLIndex;
	
	// Should only be one key for this index.
	
	flmAssert( pSQLIndex->pFirstSQLKey == pSQLIndex->pLastSQLKey);
	
	// See if there is an index for the table that matches this one.  There
	// should only be one.
	
	pFoundSQLIndex = pSQLTable->pFirstSQLIndex;
	while (pFoundSQLIndex && pFoundSQLIndex->uiIndexNum != pSQLIndex->uiIndexNum)
	{
		pFoundSQLIndex = pFoundSQLIndex->pNext;
	}
	
	if (!pFoundSQLIndex)
	{
		
		// Did not find a matching index.
		// Just link the index at the end of the table's list of indexes.
		// NOTE: It really doesn't matter if it is linked in at the beginning
		// or the end or the middle somewhere as long as it gets into the
		// list.
		
		if ((pSQLIndex->pPrev = pSQLTable->pLastSQLIndex) != NULL)
		{
			pSQLIndex->pPrev->pNext = pSQLIndex;
		}
		else
		{
			pSQLTable->pFirstSQLIndex = pSQLIndex;
		}
		pSQLTable->pLastSQLIndex = pSQLIndex;
		
		// Increment the total cost for the table
		
		pSQLTable->ui64TotalCost += pSQLIndex->pFirstSQLKey->pFSIndexCursor->getCost();
	}
	else
	{
		FLMUINT				ui64SaveCost;
		FLMBOOL				bUnioned = FALSE;
		SQL_KEY *			pSQLKey;
		SQL_KEY *			pIncomingSQLKey = pSQLIndex->pFirstSQLKey;
		FSIndexCursor *	pMergeFromFSIndexCursor = pIncomingSQLKey->pFSIndexCursor;
		FLMINT				iCompare;
		
		// Found a matching index, see if we can union the incoming index's
		// key with one of the existing keys for the found index.
		// If not, we will simply link the new key into the list.
		
		pSQLKey = pFoundSQLIndex->pFirstSQLKey;
		while (pSQLKey)
		{
			ui64SaveCost = pSQLKey->pFSIndexCursor->getCost();
			
			if (RC_BAD( rc = pSQLKey->pFSIndexCursor->unionKeys( m_pDb,
									pMergeFromFSIndexCursor, &bUnioned, &iCompare)))
			{
				goto Exit;
			}
			if (bUnioned)
			{
				pSQLTable->ui64TotalCost -= ui64SaveCost;
				pSQLTable->ui64TotalCost += pSQLKey->pFSIndexCursor->getCost();
				break;
			}
			else if (iCompare < 0)
			{
				
				// The incoming key does not overlap the current key, and is
				// less than it, so we will link it in before the current key.
				
				pIncomingSQLKey->pNext = pSQLKey;
				if ((pIncomingSQLKey->pPrev = pSQLKey->pPrev) != NULL)
				{
					pSQLKey->pPrev->pNext = pIncomingSQLKey;
				}
				else
				{
					pFoundSQLIndex->pFirstSQLKey = pIncomingSQLKey;
				}
				pSQLKey->pPrev = pIncomingSQLKey;
				
				// We didn't really union, but we did link it into the list,
				// so we set this flag to keep us from linking it at the end
				// of the list outside this loop.
				
				bUnioned = TRUE;
				break;
			}
			pSQLKey = pSQLKey->pNext;
		}
		
		// If we did not union with any of the existing keys, this incoming
		// key is greater than all of the keys in the index (otherwise it
		// would have been linked into the list in the above loop).  Hence,
		// we link it in at the end of the list of keys for the found index.
		// The keys in the list are deliberately kept in ascending order
		// so that we can traverse through the index in order if need be.
		
		if (!bUnioned)
		{
			
			// Link the incoming key as the last key off of the found index.
			
			pSQLKey = pSQLIndex->pFirstSQLKey;
			if ((pSQLKey->pPrev = pFoundSQLIndex->pLastSQLKey) != NULL)
			{
				pSQLKey->pPrev->pNext = pSQLKey;
			}
			else
			{
				pFoundSQLIndex->pFirstSQLKey = pSQLKey;
			}
			pFoundSQLIndex->pLastSQLKey = pSQLKey;
		}
			
		// Probably don't need to do this, because we should be getting
		// rid of the SQL_INDEX structure pointed to by pSQLIndex (the caller
		// should not use it anymore), but this it keeps us from having
		// two SQL_INDEX structures point to the same key - just in case
		// something on the outside changes.
		
		pSQLIndex->pFirstSQLKey = NULL;
		pSQLIndex->pLastSQLKey = NULL;
	}
	
Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Optimize a particular table for a particular sub-query.
//-------------------------------------------------------------------------
RCODE SQLQuery::optimizeTable(
	SQL_SUBQUERY *	pSubQuery,
	SQL_TABLE *		pSQLTable)
{
	RCODE						rc = NE_SFLM_OK;
	F_TABLE *				pTable = m_pDb->m_pDict->getTable( pSQLTable->uiTableNum);
	SQL_INDEX *				pFirstSQLIndex = NULL;
	SQL_INDEX *				pLastSQLIndex = NULL;
	SQL_INDEX *				pSQLIndex;
	FLMUINT					uiLoop;
	SQL_NODE *				pOperand;
	void *					pvMark = m_pool.poolMark();
	
	// This routine should not be called if the table has already been
	// marked to do a table scan.
	
	flmAssert( !pSQLTable->bScan);
	
	// Traverse the predicates of the sub-query.  If any are found
	// that are not predicates, the table must be scanned.
	
	for (uiLoop = 0, pOperand = pSubQuery->ppOperands [0];
		  uiLoop < pSubQuery->uiOperandCount;
		  uiLoop++, pOperand = pSubQuery->ppOperands [uiLoop])
	{
		
		// If we hit a predicate that has not been turned into
		// an SQL_PRED_NODE, it is not optimizable.
		
		if (pOperand->eNodeType != SQL_PRED_NODE)
		{
			
			// See if the current table is involved in this predicate.  If so,
			// and it is the only table involved, the table should be scanned.
			// Setting pFirstSQLIndex and pLastSQLIndex to NULL will cause this to
			// happen below.
			
			if (predIsForOnlyThisTable( pOperand, pSQLTable))
			{
				if (RC_BAD( rc = setupTableScan( pSQLTable)))
				{
					goto Exit;
				}
				freeTableIndexes( pSQLTable);
				m_pool.poolReset( pvMark);
				break;
			}
		}
		else if (pOperand->nd.pred.pSQLTable == pSQLTable)
		{
			SQL_PRED *	pPred = &pOperand->nd.pred;
			
			// We cannot use from and until keys for not/negative operators.
			// We set pFirstSQLIndex and pLastSQLIndex to NULL to indicate that a
			// table scan must occur.
			
			if ((pPred->bNotted && pPred->eOperator == SQL_MATCH_OP) ||
				  pPred->eOperator == SQL_NE_OP)
			{
				if (RC_BAD( rc = setupTableScan( pSQLTable)))
				{
					goto Exit;
				}
				freeTableIndexes( pSQLTable);
				m_pool.poolReset( pvMark);
				break;
			}
			
			// See if there are any indexes for this predicate's column.
			// For now we are just collecting them.  We will calculate
			// the best one later.
			
			if (RC_BAD( rc = getPredKeys( pTable, pSQLTable->uiIndexNum,
											pPred, &pFirstSQLIndex, &pLastSQLIndex)))
			{
				goto Exit;
			}
		}
	}
	
	// If we found indexes for the table, choose the best one.

	if (pFirstSQLIndex)
	{
	
		// Rank the indexes to determine which ones to estimate cost for
		// first.
		
		rankIndexes( m_pDb, &pFirstSQLIndex, &pLastSQLIndex);
		
		// Find the index with the lowest cost.
		
		if (RC_BAD( rc = chooseBestIndex( pTable, pFirstSQLIndex, &pSQLIndex)))
		{
			goto Exit;
		}
		
		// Merge the selected key for selected index into the keys for the
		// master table.
		
		if (RC_BAD( rc = mergeKeys( pSQLTable, pSQLIndex)))
		{
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Optimize the sub-queries of an SQL query.
//-------------------------------------------------------------------------
RCODE SQLQuery::optimizeSubQueries( void)
{
	RCODE					rc = NE_SFLM_OK;
	SQL_SUBQUERY *		pSubQuery;
	SQL_TABLE *			pSQLTable;
	
	// For each table in our expression, attempt to pick an index for each
	// subquery.
	
	for (pSQLTable = m_pFirstSQLTable; pSQLTable; pSQLTable = pSQLTable->pNext)
	{
		if (pSQLTable->bScan)
		{
			if (RC_BAD( rc = setupTableScan( pSQLTable)))
			{
				goto Exit;
			}
		}
		else
		{
			pSubQuery = m_pFirstSubQuery;
			while (pSubQuery)
			{
				if (RC_BAD( rc = optimizeTable( pSubQuery, pSQLTable)))
				{
					goto Exit;
				}
				
				// If the optimization decided we should scan the table, there
				// is no need to look at any more sub-queries for this table.
				
				if (pSQLTable->bScan)
				{
					break;
				}
				pSubQuery = pSubQuery->pNext;
			}
			
			// If the table's cost is still zero, that means it was not optimized for
			// any of the sub-queries, so we need to have it do a table scan or
			// an index scan.
			
			if (!pSQLTable->ui64TotalCost)
			{
				if (pSQLTable->uiIndexNum)
				{
					if (RC_BAD( rc = setupIndexScan( pSQLTable)))
					{
						goto Exit;
					}
				}
				else
				{
					if (RC_BAD( rc = setupTableScan( pSQLTable)))
					{
						goto Exit;
					}
				}
			}
			else if (!pSQLTable->bScan &&
						pSQLTable->ui64TotalCost > MINIMUM_COST_ESTIMATE)
			{
				FLMUINT64	ui64SaveCost = pSQLTable->ui64TotalCost;
				
				if (RC_BAD( rc = setupTableScan( pSQLTable)))
				{
					goto Exit;
				}
				if (pSQLTable->pFSTableCursor->getCost() < ui64SaveCost)
				{
					pSQLTable->uiIndexNum = 0;
					freeTableIndexes( pSQLTable);
				}
				else
				{
					
					// Get rid of the table cursor that was set up and stay
					// with the index cursors.
					
					pSQLTable->bScan = FALSE;
					pSQLTable->pFSTableCursor->Release();
					pSQLTable->pFSTableCursor = NULL;
					pSQLTable->ui64TotalCost = ui64SaveCost;
				}
			}
		}
	}
	
Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Add an SQL_INDEX structure to the list of indexes for an
//			SQL_TABLE structure.
//-------------------------------------------------------------------------
RCODE SQLQuery::addIndexToTable(
	SQL_TABLE *		pSQLTable,
	FLMUINT			uiIndexNum,
	SQL_INDEX **	ppSQLIndex)
{
	RCODE			rc = NE_SFLM_OK;
	SQL_INDEX *	pSQLIndex;
	
	// Allocate the SQL_INDEX structure
	
	if (RC_BAD( rc = m_pool.poolCalloc( sizeof( SQL_INDEX),
									(void **)&pSQLIndex)))
	{
		goto Exit;
	}
	
	// Link index to end of the list of indexes for the table.
	
	pSQLIndex->uiIndexNum = uiIndexNum;
	if ((pSQLIndex->pPrev = pSQLTable->pLastSQLIndex) != NULL)
	{
		pSQLTable->pLastSQLIndex->pNext = pSQLIndex;
	}
	else
	{
		pSQLTable->pFirstSQLIndex = pSQLIndex;
	}
	pSQLTable->pLastSQLIndex = pSQLIndex;
	
	if (ppSQLIndex)
	{
		*ppSQLIndex = pSQLIndex;
	}
	
Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Add an SQL_KEY structure to the list of keys for an
//			SQL_INDEX structure.
//-------------------------------------------------------------------------
RCODE SQLQuery::addKeyToIndex(
	SQL_INDEX *	pSQLIndex,
	SQL_KEY **	ppSQLKey)
{
	RCODE			rc = NE_SFLM_OK;
	SQL_KEY *	pSQLKey;
	
	// Allocate the SQL_KEY structure
	
	if (RC_BAD( rc = m_pool.poolCalloc( sizeof( SQL_KEY),
									(void **)&pSQLKey)))
	{
		goto Exit;
	}
	
	// Link key to end of the list of keys for the index.
	
	if ((pSQLKey->pPrev = pSQLIndex->pLastSQLKey) != NULL)
	{
		pSQLIndex->pLastSQLKey->pNext = pSQLKey;
	}
	else
	{
		pSQLIndex->pFirstSQLKey = pSQLKey;
	}
	pSQLIndex->pLastSQLKey = pSQLKey;
	
	*ppSQLKey = pSQLKey;
	
Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Setup to scan an index for a table.
//-------------------------------------------------------------------------
RCODE SQLQuery::setupIndexScan(
	SQL_TABLE *	pSQLTable)
{
	RCODE					rc = NE_SFLM_OK;
	F_INDEX *			pIndex;
	F_TABLE *			pTable;
	FSIndexCursor *	pFSIndexCursor = NULL;
	SQL_INDEX *			pSQLIndex;
	SQL_KEY *			pSQLKey;
			
	// At this point, there should not be any SQL_INDEX structures
	// associated with the table as yet.
	
	flmAssert( !pSQLTable->pFirstSQLIndex);
	
	if ((pIndex = m_pDb->m_pDict->getIndex( pSQLTable->uiIndexNum)) == NULL)
	{
		rc = RC_SET( NE_SFLM_Q_INVALID_INDEX);
		goto Exit;
	}
	if (pSQLTable->uiTableNum != pIndex->uiTableNum)
	{
		rc = RC_SET( NE_SFLM_Q_INVALID_TABLE_FOR_INDEX);
		goto Exit;
	}
	pTable = m_pDb->m_pDict->getTable( pSQLTable->uiTableNum);
	
	// Make sure the index is not offline.
	
	if (pIndex->uiFlags & (IXD_OFFLINE | IXD_SUSPENDED))
	{
		rc = RC_SET( NE_SFLM_INDEX_OFFLINE);
		goto Exit;
	}

	// Allocate an index structure and associate it with the table.
	
	if (RC_BAD( rc = addIndexToTable( pSQLTable, pSQLTable->uiIndexNum,
								&pSQLIndex)))
	{
		goto Exit;
	}
	
	// Allocate a key structure and associate it with the index.
	
	if (RC_BAD( rc = addKeyToIndex( pSQLIndex, &pSQLKey)))
	{
		goto Exit;
	}
	
	// Allocate an index cursor and set it up to scan the index from
	// first to last.
	
	if ((pFSIndexCursor = f_new FSIndexCursor) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}

	// Setup to scan from beginning of index to end of index.

	if (RC_BAD( rc = pFSIndexCursor->setupKeys( m_pDb, pIndex, pTable, NULL)))
	{
		goto Exit;
	}
	
	// Setup a single key for the index.
	
	pSQLKey->pFSIndexCursor = pFSIndexCursor;
	pSQLTable->ui64TotalCost = pFSIndexCursor->getCost();
	
	// Set pFSIndexCursor to NULL so it will not be released
	// below.
	
	pFSIndexCursor = NULL;
	
Exit:

	if (pFSIndexCursor)
	{
		pFSIndexCursor->Release();
	}
	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Setup to scan a table.
//-------------------------------------------------------------------------
RCODE SQLQuery::setupTableScan(
	SQL_TABLE *	pSQLTable)
{
	RCODE					rc = NE_SFLM_OK;
	FSTableCursor *	pFSTableCursor = NULL;
		
	pSQLTable->bScan = TRUE;
	
	// No index set, or the index that was set was zero, so do
	// a full table scan.
	
	if ((pFSTableCursor = f_new FSTableCursor) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}
	if (RC_BAD( rc = pFSTableCursor->setupRange( m_pDb,
								pSQLTable->uiTableNum, 1, FLM_MAX_UINT64,
								TRUE)))
	{
		goto Exit;
	}
	pSQLTable->pFSTableCursor = pFSTableCursor;
	pSQLTable->ui64TotalCost = pFSTableCursor->getCost();
	
	// Set to NULL so it won't be released at Exit.
	
	pFSTableCursor = NULL;
	
Exit:

	if (pFSTableCursor)
	{
		pFSTableCursor->Release();
	}
	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Setup to scan each table that has been specified.  It may be that
//			indexes were specified for the tables - in which case we will
//			setup to scan the indexes.
//-------------------------------------------------------------------------
RCODE SQLQuery::setupScans( void)
{
	RCODE			rc = NE_SFLM_OK;
	SQL_TABLE *	pSQLTable;
	
	// If no tables were specified, the query will return an
	// empty result.
	
	if (!m_pFirstSQLTable)
	{
		m_bEmpty = TRUE;
		goto Exit;
	}
	
	for (pSQLTable = m_pFirstSQLTable; pSQLTable; pSQLTable = pSQLTable->pNext)
	{
		if (pSQLTable->uiIndexNum)
		{
			if (RC_BAD( rc = setupIndexScan( pSQLTable)))
			{
				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = setupTableScan( pSQLTable)))
			{
				goto Exit;
			}
		}
	}

Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Optimize an SQL query.
//-------------------------------------------------------------------------
RCODE SQLQuery::optimize( void)
{
	RCODE	rc = NE_SFLM_OK;
	
	if (m_bOptimized)
	{
		goto Exit;
	}
	
	// We save the F_Database object so that we can always check and make
	// sure we are associated with this database on any query operations
	// that occur after optimization. -- Link it into the list of queries
	// off of the F_Database object.  NOTE: We may not always use the
	// same F_Db object, but it must always be the same F_Database object.

	m_pDatabase = m_pDb->m_pDatabase;
	m_pNext = NULL;
	m_pDatabase->lockMutex();
	if ((m_pPrev = m_pDatabase->m_pLastSQLQuery) != NULL)
	{
		m_pPrev->m_pNext = this;
	}
	else
	{
		m_pDatabase->m_pFirstSQLQuery = this;
	}
	m_pDatabase->m_pLastSQLQuery = this;
	m_pDatabase->unlockMutex();
	
	// Make sure we have a completed expression

	if (!criteriaIsComplete())
	{
		rc = RC_SET( NE_SFLM_Q_INCOMPLETE_QUERY_EXPR);
		goto Exit;
	}

	m_uiLanguage = m_pDb->getDefaultLanguage();

	// An empty expression should scan the tables listed - using either
	// the index that was specified for the table, or just scanning
	// the rows of the table.

	if (!m_pQuery)
	{
		rc = setupScans();
		goto Exit;
	}

	// Handle the case of a value node or arithmetic expression at the root
	// These types of expressions do not return results from the database.

	if (m_pQuery->eNodeType == SQL_VALUE_NODE)
	{
		if (m_pQuery->currVal.eValType == SQL_BOOL_VAL &&
			 m_pQuery->currVal.val.eBool == SQL_TRUE)
		{
			rc = setupScans();
			goto Exit;
		}
		else
		{
			m_bEmpty = TRUE;
		}
	}
	else if (m_pQuery->eNodeType == SQL_OPERATOR_NODE &&
		  		isSQLArithOp( m_pQuery->nd.op.eOperator))
	{
		m_bEmpty = TRUE;
		goto Exit;
	}

	// Flatten the AND and OR operators in the query tree.  Strip out
	// NOT operators, resolve constant arithmetic expressions, and
	// weed out boolean constants.
	
	if (RC_BAD( rc = reduceTree( TRUE)))
	{
		goto Exit;
	}
	
	// Convert to DNF
	
	if (RC_BAD( rc = convertToDNF()))
	{
		goto Exit;
	}
	
	// Convert all operands of each sub-query to predicates where
	// possible.
	
	if (RC_BAD( rc = convertOperandsToPredicates()))
	{
		goto Exit;
	}
	
	// Optimize each sub-query.
	
	if (RC_BAD( rc = optimizeSubQueries()))
	{
		goto Exit;
	}
	
	m_bOptimized = TRUE;
	
Exit:

	return( rc);
}

