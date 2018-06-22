//-------------------------------------------------------------------------
// Desc:	Query optimization
// Tabs:	3
//
// Copyright (c) 1994-2007 Novell, Inc. All Rights Reserved.
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

FLMBOOL gv_DoValAndDictTypesMatch[ LAST_VALUE][ FLM_CONTEXT_TYPE + 1] = 
{
	// Dictionary Types (node types - FLM_XXXX_TYPE) 
	//			TEXT		NUMBER	BINARY	CONTEXT
//qTypes (NO_TYPE=0)
/*BOOL=1*/	{FALSE,	TRUE,		FALSE,	FALSE},
/*UINT32*/	{FALSE,	TRUE,		FALSE,	TRUE},
/*INT32*/	{FALSE,	TRUE,		FALSE,	TRUE},
/*REAL*/		{FALSE,	FALSE,	FALSE,	FALSE},
/*REC_PTR*/	{FALSE,	TRUE,		FALSE,	TRUE},
/*UINT64*/	{FALSE,	TRUE,		FALSE,	TRUE},
/*INT64*/	{FALSE,	TRUE,		FALSE,	TRUE},
/*NOTUSED*/	{FALSE,	FALSE,	FALSE,	FALSE},
/*BINARY*/	{FALSE,	FALSE,	TRUE,		FALSE},
/*STRING*/	{FALSE,	FALSE,	FALSE,	FALSE},
/*UNICODE*/	{FALSE,	FALSE,	FALSE,	FALSE},
/*TEXT*/		{TRUE,	FALSE,	FALSE,	FALSE},
};

#define RANK_EQ					0
#define RANK_NE					1
#define RANK_MATCH				2
#define RANK_NOT_MATCH			3
#define RANK_MATCH_BEGIN		4
#define RANK_NOT_MATCH_BEGIN	5
#define RANK_MATCH_END			6
#define RANK_NOT_MATCH_END		7
#define RANK_CONTAINS			8
#define RANK_NOT_CONTAINS		9
#define RANK_COMPARE				10
#define RANK_EXISTS				11
#define RANK_NOT_EXISTS			12
#define RANK_OTHER				13
#define NUM_RANK_OPS				14

#define RANK_IFD_SUBSTRING		0
#define RANK_IFD_VALUE			1
#define RANK_IFD_CONTEXT		2
#define NUM_IFD_RANKS			3

FLMUINT gv_uiRanks [NUM_IFD_RANKS][ NUM_RANK_OPS] = 
{
//				EQ		NE		MTCH	!MTCH	MTCHB	!MTCHB	MTCHE	!MTCHE	CONT	!CONT	CMP	EXIST	!EXIST	OTHER
/*SS*/	{	3,		16,	4,		16,	6,		16,		7,		16,		8,		16,	10,	15,	16,		100},
/*VAL*/	{	1,		16,	2,		16,	5,		16,		14,	16,		13,	16,	9,		12,	16,		100},
/*CTX*/	{	17,	17,	17,	17,	17,	17,		17,	17,		17,	17,	17,	11,	17,		100},
};


/*
Desc: Checks the from (value) and until (dict) values in a field info structure to verify
		that they match the field type.  
*/
FINLINE FLMBOOL DoValAndDictTypesMatch(
	QTYPES	eValType,
	FLMUINT	uiDictType)
{
	if (uiDictType > FLM_CONTEXT_TYPE)
	{
		return( FALSE);
	}
	else
	{
		// subtract 1 from QTYPES - array doesn't have space for the NO_TYPE enum
		return gv_DoValAndDictTypesMatch[ ((int)eValType) - FIRST_VALUE][ uiDictType];
	}
}

/*
Desc: Clips a SUBQUERY from a list, and frees memory associated with it.
*/
FINLINE void flmClipSubQuery(
	CURSOR *		pCursor,
	SUBQUERY *	pSubQuery
	)
{
	if( pSubQuery == pCursor->pSubQueryList)
	{
		pCursor->pSubQueryList = pSubQuery->pNext;
	}
	if( pSubQuery->pPrev)
	{
		pSubQuery->pPrev->pNext = pSubQuery->pNext;
	}
	if( pSubQuery->pNext)
	{
		pSubQuery->pNext->pPrev = pSubQuery->pPrev;
	}
	flmSQFree( pSubQuery, TRUE);
}

FSTATIC RCODE flmAllocIndexInfo(
	F_Pool *			pPool,
	QINDEX **		ppIndex,
	QINDEX **		ppIndexList,
	IXD *				pIxd);

FSTATIC RCODE flmSQGetDrnRanges(
	SUBQUERY *		pSubQuery,
	QTYPES			eOperator,
	FLMUINT			uiVal);

FSTATIC RCODE flmSQGenPredicateList(
	FDB *				pDb,
	SUBQUERY *		pSubQuery,
	F_Pool *			pPool,
	QPREDICATE * *	ppPredicateList,
	FLMUINT *		puiTotalPredicates,
	FLMBOOL *		pbHaveUserPredicates);

FSTATIC FLMUINT flmCurCalcPredicateRank(
	QPREDICATE *	pPredicate,
	IFD *				pIfd);

FSTATIC void flmCurLinkPredicate(
	QINDEX *					pIndex,
	QFIELD_PREDICATE *	pPredToLink,
	QFIELD_PREDICATE **	ppPredicateList);

FSTATIC FLMBOOL flmIxFldPathSuitable(
	FLMUINT *	puiIxFldPath,
	FLMUINT *	puiQueryFldPath,
	FLMBOOL *	pbMustVerifyQueryPath);

FSTATIC RCODE flmSQGetSuitableIndexes(
	FDB *					pDb,
	FLMUINT				uiForceIndex,
	CURSOR *				pCursor,
	SUBQUERY *			pSubQuery,
	FLMUINT				uiContainer,
	F_Pool *				pPool,
	QPREDICATE *		pPredicateList,
	FLMUINT				uiTotalPredicates,
	FLMBOOL				bHaveUserPredicates,
	QINDEX * *			ppIndexList,
	FLMUINT *			puiMaxIfds);

FSTATIC RCODE flmSQEvaluateCurrIndexKey(
	FDB *						pDb,
	SUBQUERY *				pSubQuery,
	FSIndexCursor **		ppTmpFSIndexCursor,
	QINDEX *					pIndex,
	QFIELD_PREDICATE **	ppFieldCurrPredicate,
	QPREDICATE **			ppPredicateList);

FSTATIC RCODE flmCheckUserPredicateCosts(
	FDB *			pDb,
	SUBQUERY *	pSubQuery,
	FLMBOOL		bOkToOptimizeWithPredicate);

FSTATIC RCODE flmMergeSubQueries(
	CURSOR *			pCursor,
	SUBQUERY * *	ppFromSubQuery,
	SUBQUERY *		pIntoSubQuery,
	FLMBOOL			bFromSubQuerySubsumed);

FSTATIC RCODE flmSQSetupFullContainerScan(
	CURSOR *		pCursor,
	SUBQUERY *	pSubQuery);

FSTATIC RCODE flmSQChooseBestIndex(
	CURSOR *			pCursor,
	FDB *				pDb,
	FLMUINT			uiForceIndex,
	FLMUINT			bForceFirstToLastKey,
	SUBQUERY *		pSubQuery,
	F_Pool *			pTempPool,
	QPREDICATE *	pPredicateList,
	FLMUINT			uiTotalPredicates,
	FLMBOOL			bHaveUserPredicates);

/****************************************************************************
Desc:	Keep track of DRN ranges for a subquery.
****************************************************************************/
FSTATIC RCODE flmSQGetDrnRanges(
	SUBQUERY *	pSubQuery,
	QTYPES		eOperator,
	FLMUINT		uiVal
	)
{
	RCODE	rc = FERR_OK;

	switch (eOperator)
	{
		case FLM_EQ_OP:
			if ((!uiVal) ||
				 (pSubQuery->uiLowDrn > uiVal) ||
				 (pSubQuery->uiHighDrn < uiVal) ||
				 (pSubQuery->uiNotEqualDrn == uiVal))
			{
				rc = RC_SET( FERR_EMPTY_QUERY);
				goto Exit;
			}
			else
			{
				pSubQuery->uiLowDrn = pSubQuery->uiHighDrn = uiVal;
			}
			break;
		case FLM_GT_OP:
			if (pSubQuery->uiHighDrn <= uiVal || uiVal == 0xFFFFFFFF)
			{
				rc = RC_SET( FERR_EMPTY_QUERY);
				goto Exit;
			}
			if (pSubQuery->uiLowDrn < uiVal + 1)
			{
				pSubQuery->uiLowDrn = uiVal + 1;
				if ((pSubQuery->uiLowDrn == pSubQuery->uiHighDrn) &&
					 (pSubQuery->uiNotEqualDrn == pSubQuery->uiLowDrn))
				{
					rc = RC_SET( FERR_EMPTY_QUERY);
					goto Exit;
				}
			}
			break;
		case FLM_GE_OP:
			if (pSubQuery->uiHighDrn < uiVal)
			{
				rc = RC_SET( FERR_EMPTY_QUERY);
				goto Exit;
			}
			if (pSubQuery->uiLowDrn < uiVal)
			{
				pSubQuery->uiLowDrn = uiVal;
				if ((pSubQuery->uiLowDrn == pSubQuery->uiHighDrn) &&
					 (pSubQuery->uiNotEqualDrn == pSubQuery->uiLowDrn))
				{
					rc = RC_SET( FERR_EMPTY_QUERY);
					goto Exit;
				}
			}
			break;
		case FLM_LT_OP:
			if (pSubQuery->uiLowDrn >= uiVal || !uiVal)
			{
				rc = RC_SET( FERR_EMPTY_QUERY);
				goto Exit;
			}
			if (pSubQuery->uiHighDrn > uiVal - 1)
			{
				pSubQuery->uiHighDrn = uiVal - 1;
				if ((pSubQuery->uiLowDrn == pSubQuery->uiHighDrn) &&
					 (pSubQuery->uiNotEqualDrn == pSubQuery->uiLowDrn))
				{
					rc = RC_SET( FERR_EMPTY_QUERY);
					goto Exit;
				}
			}
			break;
		case FLM_LE_OP:
			if (pSubQuery->uiLowDrn > uiVal)
			{
				rc = RC_SET( FERR_EMPTY_QUERY);
				goto Exit;
			}
			if (pSubQuery->uiHighDrn > uiVal)
			{
				pSubQuery->uiHighDrn = uiVal;
				if ((pSubQuery->uiLowDrn == pSubQuery->uiHighDrn) &&
					 (pSubQuery->uiNotEqualDrn == pSubQuery->uiLowDrn))
				{
					rc = RC_SET( FERR_EMPTY_QUERY);
					goto Exit;
				}
			}
			break;
		case FLM_NE_OP:
			if (pSubQuery->uiLowDrn == uiVal &&
				 pSubQuery->uiHighDrn == uiVal)
			{
				rc = RC_SET( FERR_EMPTY_QUERY);
				goto Exit;
			}
			pSubQuery->uiNotEqualDrn = uiVal;
			break;
		default:

			// Other operators are not allowed for DRN queries.

			flmAssert( 0);
			rc = RC_SET( FERR_CURSOR_SYNTAX);
			goto Exit;
	}
Exit:
	return( rc);
}

/****************************************************************************
Desc:	Generate the predicate list for a sub-query.
****************************************************************************/
FSTATIC RCODE flmSQGenPredicateList(
	FDB *				pDb,
	SUBQUERY *		pSubQuery,
	F_Pool *			pPool,
	QPREDICATE * *	ppPredicateList,
	FLMUINT *		puiTotalPredicates,
	FLMBOOL *		pbHaveUserPredicates)
{
	RCODE				rc = FERR_OK;
	FQNODE *			pQNode;
	QTYPES			eOp;
	QPREDICATE *	pPredicate;
	QPREDICATE *	pLastPredicate;
	QTYPES			eParentOp;
	FQNODE *			pSibling;
	QTYPES			eSibOp;
	FLMUINT			uiFldId;
	FLMUINT			uiDictType;

	*ppPredicateList = NULL;
	*puiTotalPredicates = 0;
	*pbHaveUserPredicates = FALSE;

	// Nothing to do in an empty tree.

	if ((pQNode = pSubQuery->pTree) == NULL)
	{
		goto Exit;
	}

	// Traverse through all of the nodes of the tree - non-recursively.

	pLastPredicate = NULL;
	for (;;)
	{
		eOp = GET_QNODE_TYPE( pQNode);
		if (IS_FIELD( eOp))
		{

			// Create a predicate for this field.

			// Better be a parent node that is the operator for
			// this field.

			if( RC_BAD( rc = pPool->poolCalloc( sizeof( QPREDICATE), 
				(void **)&pPredicate)))
			{
				goto Exit;
			}
			
			(*puiTotalPredicates)++;
			pPredicate->puiFldPath = pQNode->pQAtom->val.QueryFld.puiFldPath;
			if (pQNode->pQAtom->uiFlags & FLM_SINGLE_VALUED)
			{
				pPredicate->bFldSingleValued = TRUE;
			}
			uiFldId = *pPredicate->puiFldPath;

			// Make sure the field is in the dictionary if it is below the
			// reserved range.

			if (uiFldId <= FLM_RESERVED_TAG_NUMS)
			{
				if (RC_BAD( rc = fdictGetField( pDb->pDict, uiFldId,
												&uiDictType, NULL, NULL)))
				{
					rc = RC_SET( FERR_BAD_FIELD_NUM);
					pDb->Diag.uiInfoFlags |= FLM_DIAG_FIELD_NUM;
					pDb->Diag.uiFieldNum = uiFldId;
					goto Exit;
				}
			}

			// If there is no parent node, this is an existence operator.

			if (!pQNode->pParent)
			{
				flmAssert( uiFldId != FLM_RECID_FIELD);
				pPredicate->pPredNode = pQNode;
				pPredicate->eOperator = FLM_EXISTS_OP;
				if (pQNode->uiStatus & FLM_NOTTED)
				{
					pPredicate->bNotted = TRUE;
				}
			}
			else
			{

				// If parent is a logical operator, it is an existence test, so
				// we leave pVal to NULL.

				eParentOp = GET_QNODE_TYPE( pQNode->pParent);
				if (IS_LOG_OP( eParentOp))
				{
					flmAssert( uiFldId != FLM_RECID_FIELD);
					pPredicate->pPredNode = pQNode;
					pPredicate->eOperator = FLM_EXISTS_OP;
					if (pQNode->uiStatus & FLM_NOTTED)
					{
						pPredicate->bNotted = TRUE;
					}
				}
				else
				{

					pPredicate->pPredNode = pQNode->pParent;
					pPredicate->eOperator = eParentOp;
					if (pQNode->pParent->uiStatus & FLM_NOTTED)
					{
						pPredicate->bNotted = TRUE;
					}

					// Better be a previous or next sibling

					if ((pSibling = pQNode->pNextSib) == NULL)
					{
						pSibling = pQNode->pPrevSib;
					}
					flmAssert( pSibling != NULL);
					eSibOp = GET_QNODE_TYPE( pSibling);

					// Better be a value or field

					if (IS_VAL( eSibOp))
					{
						pPredicate->pVal = pSibling->pQAtom;
						if (uiFldId == FLM_RECID_FIELD)
						{
							pSubQuery->bHaveDrnFlds = TRUE;
							flmAssert( eSibOp == FLM_UINT32_VAL);
							if (RC_BAD( rc = flmSQGetDrnRanges( pSubQuery,
													pPredicate->eOperator,
													(FLMUINT)pPredicate->pVal->val.ui32Val)))
							{
								goto Exit;
							}
						}
						else if (uiFldId <= FLM_RESERVED_TAG_NUMS)
						{

							// Make sure that the value type we are comparing to
							// is compatible.

							if (!DoValAndDictTypesMatch( pPredicate->pVal->eType,
										uiDictType))
							{
								flmAssert( 0);
								rc = RC_SET( FERR_CONV_BAD_DEST_TYPE);
								goto Exit;
							}
						}
					}
					else
					{
						// Can't generate a key with this because it is a field
						// to field comparison, or it is comparing to an
						// arithmetic expression.  Must set operator to NO_TYPE to
						// indicate this.

						if (uiFldId == FLM_RECID_FIELD)
						{
							pSubQuery->bHaveDrnFlds = TRUE;
						}
						pPredicate->eOperator = NO_TYPE;
					}
				}
			}

			// Link in order the predicates are found in the tree.
			// The order doesn't matter from a pure evaluation standpoint,
			// but we do it this way so that the predicates are in somewhat
			// the same order that the user expressed them.

			if (pLastPredicate)
			{
				pLastPredicate->pNext = pPredicate;
			}
			else
			{
				*ppPredicateList = pPredicate;
			}
			pLastPredicate = pPredicate;

			// Go to parent and then to parent's sibling, unless the
			// predicate is an exists operator, in which case we need
			// to go to this node's sibling, if any.

			if (pPredicate->eOperator != FLM_EXISTS_OP)
			{
				pQNode = pQNode->pParent;
			}

Goto_Sibling:

			// If no sibling, go back up the tree.

			while (pQNode && !pQNode->pNextSib)
			{
				pQNode = pQNode->pParent;
			}

			// If got to top of tree, we are done.

			if (!pQNode)
			{
				break;
			}

			// Goto the sibling node - has to be non-NULL at this point.

			pQNode = pQNode->pNextSib;
			flmAssert( pQNode != NULL);
		}
		else if (eOp == FLM_USER_PREDICATE)
		{
			*pbHaveUserPredicates = TRUE;
			goto Goto_Sibling;
		}
		else if (IS_VAL( eOp))
		{

			// Go to sibling in case sibling is a field - don't want to miss
			// a field that is on the right-hand side of the predicate.

			goto Goto_Sibling;
		}
		else
		{

			// At this point, we know we are on a logical,
			// relational (comparison), or arithmetic
			// operator.  There must always be a child
			// node at this point.  We simply descend to it.

			pQNode = pQNode->pChild;
			flmAssert( pQNode != NULL);

			// Since things have been De-Morganized, we should not encounter
			// any NOT operators.

			flmAssert( eOp != FLM_NOT_OP);
		}
	}

Exit:
	return( rc);
}

/****************************************************************************
Desc:	Allocate a QINDEX structure for an index.
****************************************************************************/
FSTATIC RCODE flmAllocIndexInfo(
	F_Pool *			pPool,
	QINDEX **		ppIndex,
	QINDEX **		ppIndexList,
	IXD *				pIxd)
{
	RCODE			rc = FERR_OK;
	QINDEX *		pIndex;

	if( RC_BAD( rc = pPool->poolCalloc( sizeof( QINDEX), (void **)&pIndex)))
	{
		goto Exit;
	}
	
	*ppIndex = pIndex;

	// The following items are initialized because of the calloc:
	//		pIndex->bDoRecMatch = FALSE;
	//		pIndex->bPredicatesRequireMatch = FALSE;
	//		pIndex->uiNumPredicatesCovered = 0;

	pIndex->uiIndexNum = pIxd->uiIndexNum;
	pIndex->uiNumFields = pIxd->uiNumFlds;
	pIndex->pIxd = pIxd;

	// Allocate space for a list of predicate pointers
	// for each IFD.
	
	if( RC_BAD( rc = pPool->poolCalloc( 
		sizeof( QFIELD_PREDICATE *) * pIndex->uiNumFields,
		(void **)&pIndex->ppFieldPredicateList)))
	{
		goto Exit;
	}

	// Link the index into the list of indexes.

	pIndex->pPrev = NULL;
	if ((pIndex->pNext = *ppIndexList) != NULL)
	{
		(*ppIndexList)->pPrev = pIndex;
	}
	
	*ppIndexList = pIndex;
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Calculate a predicate's "rank" with respect to a particular IFD.
****************************************************************************/
FSTATIC FLMUINT flmCurCalcPredicateRank(
	QPREDICATE *	pPredicate,
	IFD *				pIfd
	)
{
	FLMUINT	uiIfdType;
	FLMUINT	uiOpType;

	if (pIfd->uiFlags & IFD_SUBSTRING)
	{
		uiIfdType = RANK_IFD_SUBSTRING;
	}
	else if (pIfd->uiFlags & IFD_CONTEXT)
	{
		uiIfdType = RANK_IFD_CONTEXT;
	}
	else
	{
		uiIfdType = RANK_IFD_VALUE;
	}
	switch (pPredicate->eOperator)
	{
		case FLM_EQ_OP:
			uiOpType = RANK_EQ;
			break;
		case FLM_MATCH_OP:
			uiOpType = (FLMUINT)((pPredicate->bNotted)
										? RANK_NOT_MATCH
										: RANK_MATCH);
			break;
		case FLM_MATCH_BEGIN_OP:
			uiOpType = (FLMUINT)((pPredicate->bNotted)
										? RANK_NOT_MATCH_BEGIN
										: RANK_MATCH_BEGIN);
			break;
		case FLM_MATCH_END_OP:
			uiOpType = (FLMUINT)((pPredicate->bNotted)
										? RANK_NOT_MATCH_END
										: RANK_MATCH_END);
			break;
		case FLM_CONTAINS_OP:
			uiOpType = (FLMUINT)((pPredicate->bNotted)
										? RANK_NOT_CONTAINS
										: RANK_CONTAINS);
			break;
		case FLM_NE_OP:
			uiOpType = RANK_NE;
			break;
		case FLM_LT_OP:
		case FLM_LE_OP:
		case FLM_GT_OP:
		case FLM_GE_OP:
			uiOpType = RANK_COMPARE;
			break;
		case FLM_EXISTS_OP:
			uiOpType = (FLMUINT)((pPredicate->bNotted)
										? RANK_NOT_EXISTS
										: RANK_EXISTS);
			break;
		default:
			uiOpType = RANK_OTHER;
			break;
	}
	return( gv_uiRanks [uiIfdType][uiOpType]);
}

/****************************************************************************
Desc:	Link a predicate into the predicate list for a particular index's IFD.
		The predicate is linked according to its ranking - lower rankings are
		better (1st, 2nd, 3rd, etc.) than higher rankings, so the order is from
		low to high rankings.
****************************************************************************/
FSTATIC void flmCurLinkPredicate(
	QINDEX *					pIndex,
	QFIELD_PREDICATE *	pPredToLink,
	QFIELD_PREDICATE **	ppPredicateList
	)
{
	QFIELD_PREDICATE *	pPriorPred = NULL;
	QFIELD_PREDICATE *	pAfterPred = *ppPredicateList;

	// Position this predicate according to how good it looks in
	// comparison to others in the list.

	while (pAfterPred && pAfterPred->uiRank < pPredToLink->uiRank)
	{
		pPriorPred = pAfterPred;
		pAfterPred = pAfterPred->pNext;
	}

	// Link between the after and before predicates.

	if (!pPriorPred)
	{
		*ppPredicateList = pPredToLink;
	}
	else
	{
		pPriorPred->pNext = pPredToLink;
		if (!pPredToLink->pPredicate->bFldSingleValued ||
			 !pPriorPred->pPredicate->bFldSingleValued)
		{
			pIndex->bMultiplePredsOnIfd = TRUE;
		}
	}
	if ((pPredToLink->pNext = pAfterPred) != NULL)
	{
		if (!pPredToLink->pPredicate->bFldSingleValued ||
			 !pAfterPred->pPredicate->bFldSingleValued)
		{
			pIndex->bMultiplePredsOnIfd = TRUE;
		}
	}
}

/****************************************************************************
Desc:	Determine if an index field path is suitable for the query field path.
		It must either match or be less specific than the query field path.
****************************************************************************/
FSTATIC FLMBOOL flmIxFldPathSuitable(
	FLMUINT *	puiIxFldPath,
	FLMUINT *	puiQueryFldPath,
	FLMBOOL *	pbMustVerifyQueryPath
	)
{
	FLMBOOL	bIxPathHasWildcard = FALSE;
	FLMBOOL	bSuitable = FALSE;

	while (*puiIxFldPath)
	{
		if (*puiIxFldPath == FLM_ANY_FIELD)
		{

			// Look at next field in IFD path to see if it matches
			// the current field.  If it does, continue from there.

			if (*(puiIxFldPath + 1))
			{
				bIxPathHasWildcard = TRUE;
				if (*puiQueryFldPath == *(puiIxFldPath + 1))
				{

					// Skip wild card and field that matched.

					puiIxFldPath += 2;
				}

				if (*puiQueryFldPath)
				{

					// Go to next field in path being evaluated no matter
					// what.  If it didn't match, we continue looking at
					// the wild card.  If it did match, we go to the next
					// field in the path.

					puiQueryFldPath++;
				}
				else
				{

					// Index path not suitable - more specific than
					// query path.

					goto Exit;	// Will return FALSE
				}
			}
			else
			{

				// Rest of path is an automatic match - had wildcard
				// at top of IFD path.

				break;
			}
		}
		else if (*puiQueryFldPath != *puiIxFldPath)
		{

			// If we did not go through all of the index's field path,
			// the index field path either doesn't match or is more
			// specific than the query's field path.

			goto Exit;	// will return FALSE.
		}
		else
		{
			puiIxFldPath++;
			puiQueryFldPath++;
		}
	}

	bSuitable = TRUE;

	// If the query path is more specific, or the index path has
	// a wild card, the query path must be verified by fetching
	// the record.

	*pbMustVerifyQueryPath = (*puiQueryFldPath || bIxPathHasWildcard)
									 ? TRUE
									 : FALSE;
Exit:
	return( bSuitable);
}

/****************************************************************************
Desc:	Generate the list of suitable indexes for a sub-query.  Rank each
		index to determine which of them we should do cost estimation on
		first.
****************************************************************************/
FSTATIC RCODE flmSQGetSuitableIndexes(
	FDB *					pDb,
	FLMUINT				uiForceIndex,
	CURSOR *				pCursor,
	SUBQUERY *			pSubQuery,
	FLMUINT				uiContainer,
	F_Pool *				pPool,
	QPREDICATE *		pPredicateList,
	FLMUINT				uiTotalPredicates,
	FLMBOOL				bHaveUserPredicates,
	QINDEX * *			ppIndexList,
	FLMUINT *			puiMaxIfds)
{
	RCODE						rc = FERR_OK;
	QPREDICATE *			pPredicate;
	IFD *						pIfd;
	QINDEX *					pIndexList;
	QINDEX *					pIndex;
	QINDEX *					pNextIndex;
	FLMUINT					uiLoop;
	QFIELD_PREDICATE *	pFieldPredicate;
	FLMUINT					uiMaxIfds = 0;
	FLMBOOL					bMustVerifyQueryPath;

	// Cycle through all of the predicates and traverse each field's
	// list of indexes.

	pIndexList = NULL;
	if (uiForceIndex)
	{
		IXD *	pIxd;

		if (RC_BAD( rc = fdictGetIndex(
					pDb->pDict, pDb->pFile->bInLimitedMode,
					uiForceIndex, NULL, &pIxd)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = flmAllocIndexInfo( pPool, &pIndex, &pIndexList, pIxd)))
		{
			goto Exit;
		}
		if (pCursor->uiRecType)
		{
			pIndex->bDoRecMatch = TRUE;
		}
		if (uiMaxIfds < pIndex->uiNumFields)
		{
			uiMaxIfds = pIndex->uiNumFields;
		}
	}
	else
	{
		for (pPredicate = pPredicateList;
			  pPredicate;
			  pPredicate = pPredicate->pNext)
		{

			// No need to check for IFDs if this is the DRN field.
			// It will never have an IFD.

			if (*pPredicate->puiFldPath == FLM_RECID_FIELD)
			{
				continue;
			}

			// Get the field's IFD list.

			if (RC_BAD( rc = fdictGetField( pDb->pDict,
										*pPredicate->puiFldPath, NULL,
										&pIfd, NULL)))
			{
				goto Exit;
			}

			// Cycle through the IFDs where the field is required.

			for (; pIfd; pIfd = pIfd->pNextInChain)
			{

				// Stop at the first non-required field.

				if (!(pIfd->uiFlags &
						(IFD_REQUIRED_PIECE | IFD_REQUIRED_IN_SET)))
				{
					break;
				}

				// Check the following conditions of suitability:
				// 1) Index must be on the container we are searching.
				// 2) Index must be on-line
				// 3) If index is encrypted and we must not be in limited mode.
				// NOTE: A container number of zero means we are indexing
				// all containers.

				if ((pIfd->pIxd->uiContainerNum &&
					  pIfd->pIxd->uiContainerNum != uiContainer) ||
					 ((pIfd->pIxd->uiFlags & IXD_OFFLINE) != 0) ||
					 (pIfd->pIxd->uiEncId && pDb->pFile->bInLimitedMode))
				{
					continue;
				}

				// Make sure the IFD's path is of less or equal specificity
				// to the field's path.  We can skip the zeroeth element,
				// because we already know it matches.

				if (!flmIxFldPathSuitable( &pIfd->pFieldPathCToP [1],
													&pPredicate->puiFldPath [1],
													&bMustVerifyQueryPath))
				{
					continue;
				}

				// Verify that the predicate does not return TRUE when
				// tested against a NULL record.  If we have already
				// evaluated the predicate, no need to do it again, just
				// take the results from last time.

				if (!pPredicate->bEvaluatedNullRec)
				{
					if (IS_COMPARE_OP( pPredicate->eOperator))
					{
						FQATOM		Result;

						// f_memset( &Result, 0, sizeof( FQATOM));
						// flmCurEvalCompareOp inits the following values:
						// eType, pNext, val.Bool;

						Result.uiFlags = Result.uiBufLen = 0;

						if (RC_BAD( rc = flmCurEvalCompareOp( pDb, pSubQuery,
												NULL, pPredicate->pPredNode,
												pPredicate->eOperator,
												FALSE, &Result)))
						{
							goto Exit;
						}
						if (Result.eType == FLM_BOOL_VAL && Result.val.uiBool == FLM_TRUE)
						{
							pPredicate->bReturnsTrueOnNullRec = TRUE;
						}
					}
					else if (pPredicate->eOperator == FLM_EXISTS_OP &&
								pPredicate->bNotted)
					{
						pPredicate->bReturnsTrueOnNullRec = TRUE;
					}
					pPredicate->bEvaluatedNullRec = TRUE;
				}

				// Put this index in the list - if not already there.

				pIndex = pIndexList;
				while (pIndex && pIndex->uiIndexNum != pIfd->uiIndexNum)
				{
					pIndex = pIndex->pNext;
				}

				// If we did not find the index, allocate it and link into
				// the list.

				if (!pIndex)
				{
					if (RC_BAD( rc = flmAllocIndexInfo( pPool, &pIndex, &pIndexList,
											pIfd->pIxd)))
					{
						goto Exit;
					}
					if (uiMaxIfds < pIndex->uiNumFields)
					{
						uiMaxIfds = pIndex->uiNumFields;
					}
				}

				// If we are testing the record type, must do a record match.

				// Also if we are forcing a particular index and we did not
				// go through all of the index's field path,
				// the index field path either does not match the query's
				// field path or it is more specific that the query's
				// field path.  In either case, we must force a record match.

				if (pCursor->uiRecType)
				{
					pIndex->bDoRecMatch = TRUE;
				}

				// If the field's query path is more specific than the index's query
				// path, we need to set bDoRecMatch to TRUE - because that can only
				// be verified by fetching the record.
				// If the IFD is on field's tag and the operator is not the exists
				// operator, we must also fetch the record to evaluate the predicate.

				if (bMustVerifyQueryPath ||
					 ((pIfd->uiFlags & IFD_CONTEXT) &&
						pPredicate->eOperator != FLM_EXISTS_OP))
				{
					pIndex->bDoRecMatch = TRUE;
				}

				// Add this predicate to the list of predicates for this IFD.

				if( RC_BAD( rc = pPool->poolAlloc( sizeof( QFIELD_PREDICATE),
					(void **)&pFieldPredicate)))
				{
					goto Exit;
				}
				
				pFieldPredicate->pIfd = pIfd;
				pFieldPredicate->pPredicate = pPredicate;
				pFieldPredicate->uiRank = flmCurCalcPredicateRank( pPredicate,
														pIfd);
				flmCurLinkPredicate( pIndex, pFieldPredicate,
					&pIndex->ppFieldPredicateList [pIfd->uiCompoundPos]);
				pIndex->uiNumPredicatesCovered++;
			}
		}
	}

	// Get all predicates that match each IFD of each index.  Must eliminate
	// any indexes that do not have criteria on each required field.  This pass
	// will also set the bDoRecMatch flag if necessary.

	pIndex = pIndexList;
	while (pIndex)
	{
		FLMBOOL	bHavePredInRequiredSet;
		FLMBOOL	bHaveRequiredSet;

		// Need to get the next index in case we remove pIndex from the list.

		pNextIndex = pIndex->pNext;

		// Loop through all of the IFDs for the index - make sure that every
		// required IFD has a predicate.

		bHavePredInRequiredSet = FALSE;
		bHaveRequiredSet = FALSE;
		for (uiLoop = 0, pIfd = pIndex->pIxd->pFirstIfd;
			  uiLoop < pIndex->uiNumFields;
			  uiLoop++, pIfd++)
		{

			// If we are forcing an index, we MUST process the IFD because it
			// will not have been done above - even if it is required.

			if (uiForceIndex)
			{
				goto Process_IFD;
			}

			// See if there is a predicate for this IFD.  If non-NULL, we have
			// already collected them above - when we traversed the required IFDs.
			// We are now mainly traversing to collect the non-required IFDs.

			if (pIndex->ppFieldPredicateList [uiLoop])
			{

				// At this point, we know that the IFD is required, otherwise
				// it would not have a predicate linked off of it.

				flmAssert( pIfd->uiFlags &
								(IFD_REQUIRED_IN_SET | IFD_REQUIRED_PIECE));

				// Since this IFD is required, we must have at least one
				// predicate where the bReturnsTrueOnNullRec flag is NOT TRUE.
				// If it is TRUE for all of the predicates linked off of this
				// IFD, the index is not suitable.
				// NOTE: We could have skipped these predicates in the loop
				// above, but we actually need to link them into the list
				// because if there is at least one good predicate and one or
				// more bad predicates, we need to set the bMultiplePredsOnIfd
				// flag for the index.  If we skipped over the bad predicates
				// without linking them off of the IFD, we would not know that
				// an IFD had multiple predicates.

				pFieldPredicate = pIndex->ppFieldPredicateList [uiLoop];
				while (pFieldPredicate &&
						 pFieldPredicate->pPredicate->bReturnsTrueOnNullRec)
				{
					pFieldPredicate = pFieldPredicate->pNext;
				}
				if (!pFieldPredicate)
				{
					goto Remove_Index;
				}

				if (pIfd->uiFlags & IFD_REQUIRED_IN_SET)
				{
					// Only one of the fields in a required set has to
					// be in the query.  Here we set the flag indicating that
					// we have a required set and we have a predicate on at
					// least one of the fields in the required set.  At the
					// end of the looping through the IFDs we make our final
					// test of this condition.

					bHaveRequiredSet = TRUE;
					bHavePredInRequiredSet = TRUE;
				}
			}
			else
			{

				// If this IFD is required, but there is no predicate, we have
				// an unsuitable index.

				if (pIfd->uiFlags & IFD_REQUIRED_PIECE)
				{
					goto Remove_Index;
				}
				else if (pIfd->uiFlags & IFD_REQUIRED_IN_SET)
				{

					// Only one of the fields in a required set has to
					// be in the query.  Although this field has no predicate,
					// there may be another that does.  We can't really tell
					// until we come to the end of the IFDs for this index.
					// For now, we just set a flag to indicate that this index
					// has a required set of IFDs.

					bHaveRequiredSet = TRUE;
				}
				else
				{
Process_IFD:

					// Field is optional or we are forcing the index and
					// have not yet checked for predicates on the IFDs.
					// See if there are any predicates in the query for this
					// IFD.

					for (pPredicate = pPredicateList;
						  pPredicate;
						  pPredicate = pPredicate->pNext)
					{

						if (*pPredicate->puiFldPath == FLM_RECID_FIELD)
						{
							pIndex->bPredicatesRequireMatch = TRUE;

							// Count this as a covered predicate - because we
							// can evaluate it on just the key if necessary.

							pIndex->uiNumPredicatesCovered++;
							continue;
						}

						// See if the predicate matches this IFD.

						if (!flmIxFldPathSuitable( pIfd->pFieldPathCToP,
															pPredicate->puiFldPath,
															&bMustVerifyQueryPath))
						{
							continue;
						}

						// If the query field path is more specific, we need to
						// fetch the record to evaluate this predicate.
						// If the IFD is on field's tag, the operator better be
						// an exists operator.  If not, we must fetch the
						// record to evaluate this predicate.

						if (bMustVerifyQueryPath ||
							 ((pIfd->uiFlags & IFD_CONTEXT) &&
							   pPredicate->eOperator != FLM_EXISTS_OP))
						{
							pIndex->bDoRecMatch = TRUE;
						}

						// Add this predicate to the list of predicates for this IFD.
						
						if( RC_BAD( rc = pPool->poolAlloc( sizeof( QFIELD_PREDICATE),
							(void **)&pFieldPredicate)))
						{
							goto Exit;
						}
						
						pFieldPredicate->pIfd = pIfd;
						pFieldPredicate->pPredicate = pPredicate;
						pFieldPredicate->uiRank = flmCurCalcPredicateRank(
															pPredicate, pIfd);
						flmCurLinkPredicate( pIndex, pFieldPredicate,
							&pIndex->ppFieldPredicateList [uiLoop]);
						pIndex->uiNumPredicatesCovered++;
					}
				}
			}
		}

		// See if we had a required required set of fields in the IFD list.
		// If so, we better have had a predicate for at least one of the
		// IFDs in the set.  If we didn't, the index is not suitable for
		// the query.
		// NOTE: This doesn't matter if we are forcing an index.

		if (bHaveRequiredSet && !bHavePredInRequiredSet && !uiForceIndex)
		{
Remove_Index:

			// Remove the index from the list.

			if (pIndex->pNext)
			{
				pIndex->pNext->pPrev = pIndex->pPrev;
			}
			if (pIndex->pPrev)
			{
				pIndex->pPrev->pNext = pIndex->pNext;
			}
			else
			{
				pIndexList = pIndex->pNext;
			}

			// Set pIndex to NULL so we won't process below.

			pIndex = NULL;
		}

		// pIndex will be NULL if we removed it from the list above.

		if (pIndex)
		{

			// If we did not cover all of the predicates with this index
			// we need to fetch the records.  Also, if there are user
			// predicates, we need to fetch the record for evaluation.

			if (pIndex->uiNumPredicatesCovered < uiTotalPredicates ||
				 bHaveUserPredicates)
			{
				pIndex->bDoRecMatch = TRUE;
			}

			// Set the index's ranking.  Use the rank that was given
			// to the first IFD's first predicate.  If there is no
			// first predicate, set the index's rank to be very
			// high so that it will be evaluated last.

			pIndex->uiRank = (pIndex->ppFieldPredicateList &&
									pIndex->ppFieldPredicateList [0])
									? pIndex->ppFieldPredicateList [0]->uiRank
									: 0xFFFFFFFF;
		}
		pIndex = pNextIndex;
	}

	// Order the indexes according to their rank that was
	// assigned above.

	pIndex = pIndexList;
	while (pIndex &&
			 ((pNextIndex = pIndex->pNext) != NULL))
	{
		if (pIndex->uiRank <= pNextIndex->uiRank)
		{
			pIndex = pNextIndex;
		}
		else
		{
			QINDEX *	pPrevIndex = NULL;

			// Unlink pNextIndex from chain.

			pIndex->pNext = pNextIndex->pNext;

			// Insert pNextIndex into the list according to its rank.

			pIndex = pIndexList;
			for (;;)
			{
				if (pNextIndex->uiRank <= pIndex->uiRank)
				{

					// Insert pNextIndex between pPrevIndex and pIndex.

					if (pPrevIndex)
					{
						pPrevIndex->pNext = pNextIndex;
					}
					else
					{
						pIndexList = pNextIndex;
					}
					pNextIndex->pNext = pIndex;
					break;
				}
				else
				{
					pPrevIndex = pIndex;
					pIndex = pIndex->pNext;

					// Should be impossible for pIndex to go NULL here!
					// This is because we know that there is at least
					// one index that has a higher rank number than
					// pNextIndex has.

					flmAssert( pIndex != NULL);
				}
			}
		}
	}

	*ppIndexList = pIndexList;

Exit:
	*puiMaxIfds = uiMaxIfds;
	return( rc);
}

/****************************************************************************
Desc:	Generate a key for the current set of predicates being pointed to
		and evaluate their cost.
****************************************************************************/
FSTATIC RCODE flmSQEvaluateCurrIndexKey(
	FDB *						pDb,
	SUBQUERY *				pSubQuery,
	FSIndexCursor **		ppTmpFSIndexCursor,
	QINDEX *					pIndex,
	QFIELD_PREDICATE **	ppFieldCurrPredicate,
	QPREDICATE **			ppPredicateList
	)
{
	RCODE					rc = FERR_OK;
	FSIndexCursor *	pTmpFSIndexCursor;
	FLMUINT				uiLoop;
	QPREDICATE *		pPredicate;
	FLMUINT				uiLeafBlocksBetween;
	FLMUINT				uiTotalKeys;
	FLMUINT				uiTotalRefs;
	FLMBOOL				bDoRecMatch;
	FLMBOOL				bDoKeyMatch;
	FLMBOOL				bTotalsEstimated;
	FLMUINT				uiCost;

	for (uiLoop = 0; uiLoop < pIndex->uiNumFields; uiLoop++)
	{

		if ((pPredicate =
				(QPREDICATE *)((ppFieldCurrPredicate [uiLoop])
									? ppFieldCurrPredicate [uiLoop]->pPredicate
									: (QPREDICATE *)NULL)) == NULL)
		{
			ppPredicateList [uiLoop] = NULL;
		}
		else
		{
			switch (pPredicate->eOperator)
			{
				case FLM_EQ_OP:
				case FLM_MATCH_OP:
				case FLM_MATCH_BEGIN_OP:
				case FLM_MATCH_END_OP:
				case FLM_CONTAINS_OP:
				case FLM_NE_OP:
				case FLM_LT_OP:
				case FLM_LE_OP:
				case FLM_GT_OP:
				case FLM_GE_OP:
					if (pPredicate->pVal)
					{
						ppPredicateList [uiLoop] = pPredicate;
					}
					else
					{
						ppPredicateList [uiLoop] = NULL;
						pIndex->bPredicatesRequireMatch = TRUE;
					}
					break;
				case FLM_EXISTS_OP:
					ppPredicateList [uiLoop] = pPredicate;
					break;
				default:
					ppPredicateList [uiLoop] = NULL;
					pIndex->bPredicatesRequireMatch = TRUE;
					break;
			}
		}
	}

	// Use the temporary file system cursor to evaluate the cost.
	// Allocate a temporary file system cursor if we have not
	// already allocated one.

	if ((pTmpFSIndexCursor = *ppTmpFSIndexCursor) == NULL)
	{
		if ((pTmpFSIndexCursor = *ppTmpFSIndexCursor =
					f_new FSIndexCursor) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
	}
	else
	{
		pTmpFSIndexCursor->reset();
	}
	bDoRecMatch = pIndex->bDoRecMatch;
	bDoKeyMatch = TRUE;

	if (RC_BAD( rc = pTmpFSIndexCursor->setupKeys( pDb, pIndex->pIxd,
											ppPredicateList,
											&bDoRecMatch, &bDoKeyMatch,
											&uiLeafBlocksBetween, &uiTotalKeys,
											&uiTotalRefs, &bTotalsEstimated)))
	{
		goto Exit;
	}

	// If we have multiple predicates on an IFD, we MUST NOT
	// do a key match, because any key read from the index
	// will NOT have multiple values, and hence would possibly
	// fail one of the predicates. Instead, we must do a
	// record match.

	if (pIndex->bMultiplePredsOnIfd)
	{
		bDoKeyMatch = FALSE;
		bDoRecMatch = TRUE;
	}


	// If we must do some kind of predicate matching, either
	// bDoKeyMatch or bDoRecMatch must be set to TRUE,
	// preferrably bDoKeyMatch.
	// If bDoKeyMatch was FALSE and the intent was that it be
	// forced to FALSE, bDoRecMatch would have been set to TRUE.
	// Thus, Since bDoRecMatch is FALSE, we can safely set
	// bDoKeyMatch to TRUE, regardless of what it was before,
	// because there was no intent that it be forced to FALSE.

	if (pIndex->bPredicatesRequireMatch)
	{
		if (!bDoRecMatch)
		{
			bDoKeyMatch = TRUE;
		}
		// else bDoRecMatch is TRUE, and that is sufficient.
	}

	uiCost = (FLMUINT)((bDoRecMatch)
							 ? uiLeafBlocksBetween + uiTotalRefs
							 : uiLeafBlocksBetween);

	// Could be that there are zero leaf blocks between and
	// bDoRecMatch is FALSE.  But we do not want a cost of
	// zero.

	if (!uiCost)
	{
		uiCost = 1;
	}

	if (!pSubQuery->OptInfo.uiCost || uiCost < pSubQuery->OptInfo.uiCost)
	{


		// Exchange the temporary file system cursor and the
		// file system cursor inside the sub-query.  Want to
		// keep the temporary one and reuse the one that was
		// inside the sub-query.

		if (pSubQuery->pFSIndexCursor)
		{
			pSubQuery->pFSIndexCursor->reset();
		}
		*ppTmpFSIndexCursor = pSubQuery->pFSIndexCursor;

		pSubQuery->OptInfo.eOptType = QOPT_USING_INDEX;
		pSubQuery->OptInfo.uiIxNum = pIndex->uiIndexNum;
		pSubQuery->pFSIndexCursor = pTmpFSIndexCursor;
		pSubQuery->OptInfo.uiCost = uiCost;
		pSubQuery->OptInfo.uiDrnCost = uiTotalRefs;
		pSubQuery->OptInfo.bDoRecMatch = bDoRecMatch;
		pSubQuery->OptInfo.bDoKeyMatch = bDoKeyMatch;

		// Not really necessary to set these, but it is
		// cleaner.

		pSubQuery->OptInfo.uiDrn = 0;
		pSubQuery->pPredicate = NULL;

		// The following better already be set.

		flmAssert( pSubQuery->pFSDataCursor == NULL);
	}
Exit:
	return( rc);
}

/****************************************************************************
Desc:	Set up a sub-query to do a full container scan.
****************************************************************************/
FSTATIC RCODE flmSQSetupFullContainerScan(
	CURSOR *		pCursor,
	SUBQUERY *	pSubQuery
	)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bTotalsEstimated;
	FLMUINT	uiLeafBlocksBetween;
	FLMUINT	uiEstimatedDrns;

	// Set up a file system data cursor

	if ((pSubQuery->pFSDataCursor = f_new FSDataCursor) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Set up the range to calculate the cost.

	if (RC_BAD( rc = pSubQuery->pFSDataCursor->setupRange( pCursor->pDb,
								pCursor->uiContainer, 1, 0xFFFFFFFF,
								&uiLeafBlocksBetween, &uiEstimatedDrns,
								&bTotalsEstimated)))
	{
		goto Exit;
	}

	// Set up everything to be a full container scan.

	pSubQuery->OptInfo.eOptType = QOPT_FULL_CONTAINER_SCAN;
	pSubQuery->OptInfo.bDoRecMatch = TRUE;
	pSubQuery->OptInfo.bDoKeyMatch = FALSE;

	// Must not ever have a cost of zero.  If the container is
	// empty, we will get zero leaf blocks between, in which case
	// we want to return at least a cost of one.

	if ((pSubQuery->OptInfo.uiCost = uiLeafBlocksBetween) == 0)
	{
		pSubQuery->OptInfo.uiCost = 1;
	}
	pSubQuery->OptInfo.uiDrnCost = uiEstimatedDrns;

	// Not really necessary to set these things, but it is cleaner.

	pSubQuery->OptInfo.uiIxNum = 0;
	pSubQuery->OptInfo.uiDrn = 0;
	pSubQuery->pPredicate = NULL;

	// Kill the file system index cursor, if any.

	if (pSubQuery->pFSIndexCursor)
	{
		pSubQuery->pFSIndexCursor->Release();
		pSubQuery->pFSIndexCursor = NULL;
	}
Exit:
	return( rc);
}

/****************************************************************************
Desc:	Chooses a best index to use for a sub-query.
****************************************************************************/
FSTATIC RCODE flmSQChooseBestIndex(
	CURSOR *			pCursor,
	FDB *				pDb,
	FLMUINT			uiForceIndex,
	FLMUINT			bForceFirstToLastKey,
	SUBQUERY *		pSubQuery,
	F_Pool *			pTempPool,
	QPREDICATE *	pPredicateList,
	FLMUINT			uiTotalPredicates,
	FLMBOOL			bHaveUserPredicates)
{
	RCODE						rc = FERR_OK;
	QINDEX *					pIndexList;
	QINDEX *					pIndex;
	FLMUINT					uiCurrIfd;
	FLMUINT					uiMaxIfds;
	QFIELD_PREDICATE **	ppFieldCurrPredicate = NULL;
	QPREDICATE **			ppPredicateList = NULL;
	FSIndexCursor *		pTmpFSIndexCursor = NULL;
	FSDataCursor *			pTmpFSDataCursor = NULL;

	// If this is a DRN==x query, don't bother looking at indexes.

	if (!uiForceIndex &&
		 pSubQuery->uiLowDrn == pSubQuery->uiHighDrn)
	{

		// This is the lowest cost possible - the cost of reading
		// one record.

		pSubQuery->OptInfo.eOptType = QOPT_SINGLE_RECORD_READ;
		pSubQuery->OptInfo.uiCost = 1;
		pSubQuery->OptInfo.uiDrnCost = 1;
		pSubQuery->OptInfo.uiDrn = pSubQuery->uiLowDrn;
		pSubQuery->OptInfo.bDoRecMatch = TRUE;
		pSubQuery->OptInfo.bDoKeyMatch = FALSE;

		// Not really necessary to set these things, but makes
		// it cleaner.

		pSubQuery->OptInfo.uiIxNum =0;
		pSubQuery->pPredicate = NULL;

		// The following things should already be set correctly.

		flmAssert( pSubQuery->pFSIndexCursor == NULL &&
					  pSubQuery->pFSDataCursor == NULL);

		goto Exit;		// Nothing more to do - should return SUCCESS.
	}

	// Generate the list of suitable indexes.

	if (RC_BAD( rc = flmSQGetSuitableIndexes( pDb, uiForceIndex,
								pCursor, pSubQuery,
								pCursor->uiContainer, pTempPool,
								pPredicateList, uiTotalPredicates, 
								bHaveUserPredicates, &pIndexList,
								&uiMaxIfds)))
	{
		goto Exit;
	}

	// Allocate temporary predicate pointer arrays.

	if (uiMaxIfds)
	{

		// Allocate space for a second list of predicate pointers
		// for each IFD.  This one is used to keep track of which
		// predicate we are on when we are generating keys.
		
		if( RC_BAD( rc = pTempPool->poolCalloc(
			sizeof( QFIELD_PREDICATE *) * uiMaxIfds,
			(void **)&ppFieldCurrPredicate)))
		{
			goto Exit;
		}

		// Allocate space for the array that will be passed into the
		// key generation routine.
		
		if( RC_BAD( rc = pTempPool->poolCalloc( 
			sizeof( QPREDICATE *) * uiMaxIfds,
			(void **)&ppPredicateList)))
		{
			goto Exit;
		}
	}

	// Calculate the cost of each of the indexes.

	pSubQuery->OptInfo.eOptType = QOPT_NONE;
	pSubQuery->OptInfo.uiCost = 0;
	pSubQuery->OptInfo.uiDrnCost = 0;
	for (pIndex = pIndexList; pIndex; pIndex = pIndex->pNext)
	{

		// Generate all search keys, keep one with lowest cost.

		if (bForceFirstToLastKey)
		{

			// If we are forcing a first-to-last key, we set up an
			// array of NULL predicates.  Forcing of a first-to-last
			// key is only done when we are forcing a particular index
			// and we were unable to stratify the query into a
			// disjunction of conjunct sub-queries.

			f_memset( ppFieldCurrPredicate, 0,
						sizeof( QFIELD_PREDICATE *) * pIndex->uiNumFields);
		}
		else
		{
			f_memcpy( ppFieldCurrPredicate, pIndex->ppFieldPredicateList,
						sizeof( QFIELD_PREDICATE *) * pIndex->uiNumFields);
		}
		uiCurrIfd = 0;
		for (;;)
		{

			if (RC_BAD( rc = flmSQEvaluateCurrIndexKey( pDb, pSubQuery,
							&pTmpFSIndexCursor, pIndex,
							ppFieldCurrPredicate,
							ppPredicateList)))
			{
				goto Exit;
			}

			// See if it is worth going on.  If cost is lower than 8, it is not.
			// Also, if we are forcing a first to last key, we are done.

			if (pSubQuery->OptInfo.uiCost && pSubQuery->OptInfo.uiCost < 8)
			{
				goto Done_Evaluating_Indexes;
			}
			else if (bForceFirstToLastKey)
			{
				goto Next_Index;
			}

			// Go to next set of predicates

			for (;;)
			{

				// See if the current IFD has another predicate to process.

				if ((ppFieldCurrPredicate [uiCurrIfd]) &&
					 (ppFieldCurrPredicate [uiCurrIfd]->pNext))
				{
Next_Ifd_Predicate:
					ppFieldCurrPredicate [uiCurrIfd] =
						ppFieldCurrPredicate [uiCurrIfd]->pNext;

					// If this is not the last IFD in the index, change
					// uiCurrIfd so that we will be looking at the predicate
					// list for the next IFD in the index the next time we
					// come in to get another predicate.

					if (uiCurrIfd < pIndex->uiNumFields - 1)
					{
						uiCurrIfd++;
					}
					break;
				}
				else if (pIndex->uiNumFields == 1)
				{

					// Only one IFD in the index, and we have traversed all of
					// its predicates - go to next index.

					goto Next_Index;
				}
				else if (++uiCurrIfd == pIndex->uiNumFields)
				{

					// If we have gone through all of the IFDs, traverse back
					// up the list of IFDs until we hit one that has more
					// predicates to process.  As we go back up the list, if an
					// IFD's predicate list has been completely processed,
					// reset its current predicate to start at the first of the
					// list again.

					uiCurrIfd--;
					for (;;)
					{

						// Reset this IFD's current predicate to the first of its
						// predicate list.

						ppFieldCurrPredicate [uiCurrIfd] =
							pIndex->ppFieldPredicateList [uiCurrIfd];

						// See if prior IFD's predicate list was completely
						// processed.

						uiCurrIfd--;
						if ((ppFieldCurrPredicate [uiCurrIfd]) &&
							 (ppFieldCurrPredicate [uiCurrIfd]->pNext))
						{
							goto Next_Ifd_Predicate;
						}

						// If we are at the first IFD in the index, we have
						// processed all combinations of predicates for the
						// index - time to go to the next index.

						if (!uiCurrIfd)
						{
							goto Next_Index;
						}
					}
				}
			}
		}
Next_Index:
		;

	}

Done_Evaluating_Indexes:

	// If the sub-query has DRN fields, analyze it to see if it would be
	// better to use the DRN keys than any index at all.  Also need to
	// do this if there were no suitable indexes.

	if (!uiForceIndex &&
		 (pSubQuery->bHaveDrnFlds || !pSubQuery->pFSIndexCursor))
	{

		// NOTE: Will have already taken care of the case where
		// low drn == high drn - see above.

		// If it is first to last, set up a full container scan.

		if (pSubQuery->uiLowDrn == 1 && pSubQuery->uiHighDrn == 0xFFFFFFFF)
		{

			// Only use full container scan if there is no suitable index.

			if (!pSubQuery->pFSIndexCursor)
			{
				if (RC_BAD( rc = flmSQSetupFullContainerScan( pCursor,
												pSubQuery)))
				{
					goto Exit;
				}
				pSubQuery->pFSDataCursor->setContainer( pCursor->uiContainer);
			}
		}
		else
		{
			FLMUINT	uiLeafBlocksBetween;
			FLMUINT	uiEstimatedDrns;
			FLMBOOL	bTotalsEstimated;

			// Set up a partial container scan.

			if ((pTmpFSDataCursor = f_new FSDataCursor) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}

			if (RC_BAD( rc = pTmpFSDataCursor->setupRange( pDb,
										pCursor->uiContainer, pSubQuery->uiLowDrn,
										pSubQuery->uiHighDrn,
										&uiLeafBlocksBetween,
										&uiEstimatedDrns, &bTotalsEstimated)))
			{
				goto Exit;
			}

			// If this is lower cost than the index, use it and
			// free up the index cursor.

			if (uiLeafBlocksBetween < pSubQuery->OptInfo.uiCost ||
				 !pSubQuery->OptInfo.uiCost)
			{
				pSubQuery->OptInfo.eOptType = QOPT_PARTIAL_CONTAINER_SCAN;

				// Must not ever have a cost of zero.  If the range is
				// empty, we will get zero leaf blocks between, in which case
				// we want to return at least a cost of one.

				if ((pSubQuery->OptInfo.uiCost = uiLeafBlocksBetween) == 0)
				{
					pSubQuery->OptInfo.uiCost = 1;
				}
				pSubQuery->OptInfo.uiDrnCost = uiEstimatedDrns;
				pSubQuery->OptInfo.bDoRecMatch = TRUE;
				pSubQuery->OptInfo.bDoKeyMatch = FALSE;
				pSubQuery->pFSDataCursor = pTmpFSDataCursor;

				// Must set pTmpFSDataCursor to NULL so that it will
				// not be freed below.

				pTmpFSDataCursor = NULL;

				// Not really necessary to set these, but makes things
				// cleaner.

				pSubQuery->OptInfo.uiIxNum = 0;
				pSubQuery->OptInfo.uiDrn = 0;
				pSubQuery->pPredicate = NULL;

				// Free up the file system index cursor, if any.

				if (pSubQuery->pFSIndexCursor)
				{
					pSubQuery->pFSIndexCursor->Release();
					pSubQuery->pFSIndexCursor = NULL;
				}
			}
		}
	}

Exit:
	if (pTmpFSIndexCursor)
	{
		pTmpFSIndexCursor->Release();
	}
	if (pTmpFSDataCursor)
	{
		pTmpFSDataCursor->Release();
	}
	return( rc);
}

/****************************************************************************
Desc:	Gets scores for any embedded user predicate.
****************************************************************************/
FSTATIC RCODE flmCheckUserPredicateCosts(
	FDB *			pDb,
	SUBQUERY *	pSubQuery,
	FLMBOOL		bOkToOptimizeWithPredicate
	)
{
	RCODE						rc = FERR_OK;
	FQNODE *					pQNode = pSubQuery->pTree;
	FLMUINT					uiCost;
	FLMUINT					uiDrnCost;
	FlmUserPredicate *	pPredicate;
	FlmUserPredicate *	pLowestCostPredicate = NULL;
	FLMUINT					uiLowestCost = 0;
	FLMUINT					uiLowestDrnCost = 0;
	FLMUINT					uiSumTestRecordCost = 0;
	FLMUINT					uiTestRecordCost;
	FLMUINT					uiLowestTestRecordCost = 0;
	FLMUINT					uiSumTestAllRecordCost = 0;
	FLMUINT					uiTestAllRecordCost;
	FLMBOOL					bPassesEmptyRec;

	while (pQNode)
	{

		// If we have a user predicate, get its score.

		if (GET_QNODE_TYPE( pQNode) == FLM_USER_PREDICATE)
		{
			FLMBOOL	bSavedInvisTrans;

			pPredicate = pQNode->pQAtom->val.pPredicate;
			CB_ENTER( pDb, &bSavedInvisTrans);
			rc = pPredicate->searchCost( (HFDB)pDb,
							(FLMBOOL)((pQNode->uiStatus & FLM_NOTTED)
										 ? (FLMBOOL)TRUE
										 : (FLMBOOL)FALSE),
							(FLMBOOL)((pQNode->uiStatus & FLM_FOR_EVERY)
										 ? (FLMBOOL)FALSE
										 : (FLMBOOL)TRUE), &uiCost, &uiDrnCost,
										 &uiTestRecordCost, &bPassesEmptyRec);
			CB_EXIT( pDb, bSavedInvisTrans);
			if (RC_BAD( rc))
			{
				goto Exit;
			}

			uiSumTestRecordCost += uiTestRecordCost;
			uiTestAllRecordCost = 0;
			if (pSubQuery->OptInfo.eOptType == QOPT_FULL_CONTAINER_SCAN)
			{
				CB_ENTER( pDb, &bSavedInvisTrans);
				rc = pPredicate->testAllRecordCost( (HFDB)pDb,
								&uiTestAllRecordCost);
				CB_EXIT( pDb, bSavedInvisTrans);
				if (RC_BAD( rc))
				{
					goto Exit;
				}
				uiSumTestAllRecordCost += uiTestAllRecordCost;
			}

			// Cannot use a predicate that would pass an empty record - 
			// because the predicate might not return all records that
			// would pass the criteria.

			if (!bPassesEmptyRec &&
				 (!pLowestCostPredicate || uiCost < uiLowestCost))
			{
				pLowestCostPredicate = pPredicate;
				uiLowestCost = uiCost;
				uiLowestDrnCost = uiDrnCost;
				uiLowestTestRecordCost = uiTestRecordCost;
			}
		}
		if (pQNode->pChild)
		{
			pQNode = pQNode->pChild;
		}
		else
		{

			// Traverse back up the tree until we hit the top
			// or we hit a sibling that has not been processed.

			for (;;)
			{
				if (pQNode->pNextSib)
				{
					pQNode = pQNode->pNextSib;
					break;
				}
				else if ((pQNode = pQNode->pParent) == NULL)
				{
					break;
				}
			}
		}
	}

	// Adjust the current predicate with the additional test record
	// costs.
	// For a full container scan, it is less likely that we
	// will be calling the testRecord() method for every record,
	// because not as many will pass.  In this case, the additional
	// cost is the result of the call we make the
	// additional cost is calculated by getting the cost from
	// the predicates of what it would be to test every record
	// the predicate is likely going to cover.
	// For a sub-query that is NOT a full container scan, the assumption
	// is that the records retrieved in the key (or DRN) ranges will mostly
	// pass the criteria - thus, we will likely call testRecord()
	// for each record fetched.  So the additional cost is
	// the testRecordCost() times the number of records we
	// are probably going to have to fetch.

	if (pSubQuery->OptInfo.eOptType == QOPT_FULL_CONTAINER_SCAN)
	{
		pSubQuery->OptInfo.uiCost += uiSumTestAllRecordCost;
	}
	else
	{
		pSubQuery->OptInfo.uiCost +=
			(uiSumTestRecordCost * pSubQuery->OptInfo.uiDrnCost);
	}

	if (pLowestCostPredicate)
	{

		// Lowest predicate cost is the lowest predicate's actual cost,
		// plus the test record cost of all of the other predicates - note
		// that we subtract out the test record cost of this
		// predicate.

		uiLowestCost += (uiSumTestRecordCost - uiLowestTestRecordCost) *
								uiLowestDrnCost;
	}

	// If the predicate with the lowest cost is lower than
	// the current sub-query estimated cost, use it to
	// optimize the query.

	if (bOkToOptimizeWithPredicate &&
		 pLowestCostPredicate &&
		 uiLowestCost < pSubQuery->OptInfo.uiCost)
	{
		pSubQuery->OptInfo.eOptType = QOPT_USING_PREDICATE;
		pSubQuery->pPredicate = pLowestCostPredicate;

		pSubQuery->OptInfo.uiCost = uiLowestCost;

		// Release index cursor if any.

		pSubQuery->OptInfo.uiIxNum = 0;
		if (pSubQuery->pFSIndexCursor)
		{
			pSubQuery->pFSIndexCursor->Release();
			pSubQuery->pFSIndexCursor = NULL;
		}

		// Release data cursor if any

		pSubQuery->OptInfo.uiDrn = 0;
		if (pSubQuery->pFSDataCursor)
		{
			pSubQuery->pFSDataCursor->Release();
			pSubQuery->pFSDataCursor = NULL;
		}
		pSubQuery->OptInfo.bDoRecMatch = TRUE;
		pSubQuery->OptInfo.bDoKeyMatch = FALSE;
	}
Exit:
	return( rc);
}

/****************************************************************************
Desc: Merges two SUBQUERY structures.
****************************************************************************/
FSTATIC RCODE flmMergeSubQueries(
	CURSOR *			pCursor,
	SUBQUERY * *	ppFromSubQuery,
	SUBQUERY *		pIntoSubQuery,
	FLMBOOL			bFromSubQuerySubsumed
	)
{
	RCODE				rc = FERR_OK;
	SUBQUERY *		pFromSubQuery = *ppFromSubQuery;
	FSDataCursor *	pTmpFSCursor = NULL;
	OPT_INFO			TmpOptInfo;

	if( RC_BAD( rc = flmCurGraftNode( &pCursor->QueryPool,
									pFromSubQuery->pTree,
									FLM_OR_OP, &pIntoSubQuery->pTree)))
	{
		goto Exit;
	}
	pFromSubQuery->pTree = NULL;

	switch (pIntoSubQuery->OptInfo.eOptType)
	{
		case QOPT_USING_INDEX:

			// This kind of a merge should only occur if the
			// destination cursor and source cursor both had
			// the same index.

			flmAssert( pFromSubQuery->OptInfo.eOptType == QOPT_USING_INDEX &&
						  pFromSubQuery->pFSIndexCursor != NULL &&
						  pFromSubQuery->OptInfo.uiIxNum ==
						  pIntoSubQuery->OptInfo.uiIxNum);
			if (RC_BAD( rc = pIntoSubQuery->pFSIndexCursor->unionKeys(
								pFromSubQuery->pFSIndexCursor)))
			{
				goto Exit;
			}

			// Only change flags if pIntoSubQuery->bDoRecMatch is FALSE.
			// If pIntoSubQuery->bDoRecMatch is TRUE, we will not
			// change it or bDoKeyMatch.  Remember, if
			// pIntoSubQuery->bDoKeyMatch is FALSE and
			// pIntoSubQuery->bDoRecMatch is TRUE, it means that
			// we MUST NOT do a key match - we are forcing a record match
			// INSTEAD of a key match.

			if (!pIntoSubQuery->OptInfo.bDoRecMatch)
			{
				if (pFromSubQuery->OptInfo.bDoRecMatch)
				{
					pIntoSubQuery->OptInfo.bDoRecMatch = TRUE;
				}

				if (pFromSubQuery->OptInfo.bDoKeyMatch)
				{
					pIntoSubQuery->OptInfo.bDoKeyMatch = TRUE;
				}
			}
			break;
		case QOPT_USING_PREDICATE:

			// Can only merge into a predicate sub-query if the
			// from sub-query is also a predicate.

			flmAssert( pFromSubQuery->OptInfo.eOptType ==
								QOPT_USING_PREDICATE);
			break;
		case QOPT_SINGLE_RECORD_READ:
			if (pFromSubQuery->OptInfo.eOptType == QOPT_SINGLE_RECORD_READ)
			{

				// Can only merge into a single record read if the
				// from sub-query is also a single record read AND
				// it is the exact same DRN.

				flmAssert( pFromSubQuery->OptInfo.eOptType ==
							QOPT_SINGLE_RECORD_READ &&
						  pFromSubQuery->OptInfo.uiDrn ==
						  pIntoSubQuery->OptInfo.uiDrn);
			}
			else if (pFromSubQuery->OptInfo.eOptType ==
							QOPT_PARTIAL_CONTAINER_SCAN)
			{

				// Set up a file system data cursor

				if ((pIntoSubQuery->pFSDataCursor = f_new FSDataCursor) == NULL)
				{
					rc = RC_SET( FERR_MEM);
					goto Exit;
				}

				if (RC_BAD( rc = pIntoSubQuery->pFSDataCursor->setupRange(
												pCursor->pDb, pCursor->uiContainer,
												pIntoSubQuery->OptInfo.uiDrn,
												pIntoSubQuery->OptInfo.uiDrn,
												NULL, NULL, NULL)))
				{
					goto Exit;
				}
				if (RC_BAD( rc = pIntoSubQuery->pFSDataCursor->unionRange(
												pFromSubQuery->pFSDataCursor)))
				{
					goto Exit;
				}
				pIntoSubQuery->OptInfo.eOptType = QOPT_PARTIAL_CONTAINER_SCAN;
			}
			else if (pFromSubQuery->OptInfo.eOptType ==
							QOPT_FULL_CONTAINER_SCAN)
			{

				// Swap the types and data cursors of each sub-query.  Could do
				// a merge, but we would have to set up a temporary data cursor
				// to do it.  It is actually faster this way.
				// NOTE: We do the swapping simply so that the other fields in
				// a sub-query are consistent with the eOptType.  Although we
				// are simply going to free up the pFromSubQuery down below, it
				// is important that the sub-query always be consistently set up
				// because it may be that the routine which frees a subquery
				// will assume that it is.

Swap_Data_Opt_Info:
				pTmpFSCursor = pIntoSubQuery->pFSDataCursor;
				f_memcpy( &TmpOptInfo, &pIntoSubQuery->OptInfo,
					sizeof( OPT_INFO));

				pIntoSubQuery->pFSDataCursor =
					pFromSubQuery->pFSDataCursor;
				f_memcpy( &pIntoSubQuery->OptInfo, &pFromSubQuery->OptInfo,
					sizeof( OPT_INFO));

				pFromSubQuery->pFSDataCursor = pTmpFSCursor;
				f_memcpy( &pFromSubQuery->OptInfo, &TmpOptInfo,
					sizeof( OPT_INFO));

				// Must set pTmpFSCursor back to NULL or it will be deleted
				// below.

				pTmpFSCursor = NULL;
			}
			else
			{
				flmAssert( 0);
			}

			break;
		case QOPT_PARTIAL_CONTAINER_SCAN:

			// The from sub-query better be another partial
			// container scan or a single record retrieve.

			if (pFromSubQuery->OptInfo.eOptType == QOPT_SINGLE_RECORD_READ)
			{
				// Set up a file system data cursor

				if ((pTmpFSCursor = f_new FSDataCursor) == NULL)
				{
					rc = RC_SET( FERR_MEM);
					goto Exit;
				}

				if (RC_BAD( rc = pTmpFSCursor->setupRange(
												pCursor->pDb, pCursor->uiContainer,
												pFromSubQuery->OptInfo.uiDrn,
												pFromSubQuery->OptInfo.uiDrn,
												NULL, NULL, NULL)))
				{
					goto Exit;
				}
				if (RC_BAD( rc = pIntoSubQuery->pFSDataCursor->unionRange(
												pTmpFSCursor)))
				{
					goto Exit;
				}
			}
			else if (pFromSubQuery->OptInfo.eOptType ==
							QOPT_PARTIAL_CONTAINER_SCAN)
			{
				flmAssert( pFromSubQuery->pFSDataCursor != NULL);
				if (RC_BAD( rc = pIntoSubQuery->pFSDataCursor->unionRange(
												pFromSubQuery->pFSDataCursor)))
				{
					goto Exit;
				}
			}
			else if (pFromSubQuery->OptInfo.eOptType == QOPT_FULL_CONTAINER_SCAN)
			{

				// Swap the types and data cursors of each sub-query.  Could do
				// a merge, but a swap is going to be faster.

				goto Swap_Data_Opt_Info;
			}
			else
			{
				flmAssert( 0);
			}

			break;
		case QOPT_FULL_CONTAINER_SCAN:

			// Don't need to do anything with pFromSubQuery
			// simply discard it below.

			break;
		default:
			flmAssert( 0);
			break;
	}

	// Add the costs, unless the from query was subsumed by the into query.

	if (!bFromSubQuerySubsumed)
	{
		pIntoSubQuery->OptInfo.uiCost += pFromSubQuery->OptInfo.uiCost;
		pIntoSubQuery->OptInfo.uiDrnCost += pFromSubQuery->OptInfo.uiDrnCost;
	}

	// Set *ppFromSubQuery to the next sub-query in the list and
	// clip out pFromSubQuery.

	*ppFromSubQuery = pFromSubQuery->pNext;
	flmClipSubQuery( pCursor, pFromSubQuery);
Exit:
	if (pTmpFSCursor)
	{
		pTmpFSCursor->Release();
	}
	return( rc);
}

/****************************************************************************
Desc: Optimizes the passed-in query.
****************************************************************************/
RCODE flmCurOptimize(
	CURSOR *			pCursor,
	FLMBOOL			bStratified)
{
	RCODE				rc = FERR_OK;
	FDB *				pDb = NULL;
	SUBQUERY *		pSubQuery;
	SUBQUERY *		pTmpSubQuery;
	SUBQUERY *		pContainerScanSubQuery = NULL;
	FLMBOOL			bChoosingIndex;
	DB_STATS *		pDbStats;
	QPREDICATE *	pPredicateList = NULL;
	FLMUINT			uiTotalPredicates = 0;
	F_Pool *			pTempPool;
	void *			pvMark;
	qOptTypes		eOptType;
	FLMBOOL			bFromSubQuerySubsumed = FALSE;
	FLMBOOL			bHaveUserPredicates = FALSE;

	// Set up the operation control structure.

	pDb = pCursor->pDb;
	if (RC_BAD( rc = flmCurDbInit( pCursor)))
	{
		goto Exit;
	}

	// Verify that we have a valid container.

	if(( pCursor->uiContainer != FLM_DATA_CONTAINER) &&
		( pCursor->uiContainer != FLM_DICT_CONTAINER))
	{
		if (RC_BAD( rc = fdictGetContainer( pDb->pDict, pCursor->uiContainer, NULL)))
		{
			goto Exit;
		}
	}

	if ((pDbStats = pDb->pDbStats) != NULL)
	{
		pDbStats->bHaveStats = TRUE;
		pDbStats->ui64NumCursors++;
	}

	pSubQuery = pCursor->pSubQueryList;
	pTempPool = &pDb->TempPool;
	pvMark = pTempPool->poolMark();
	bChoosingIndex = (FLMBOOL)((pCursor->uiIndexNum == FLM_SELECT_INDEX)
										? (FLMBOOL)TRUE
										: (FLMBOOL)FALSE);

	// Process all of the sub-queries.

	for(;;)
	{
		pTempPool->poolReset( pvMark);

		// First create the predicate list for the sub-query - so
		// that it only has to be done once.  This call also verifies
		// all fields in the subquery.  Only do if we successfully
		// stratified the query.

		if (bStratified)
		{
			// pSubQuery->pFSIndexCursor = NULL;	Should already be initialized.
			// pSubQuery->pFSDataCursor = NULL;		Should already be initialized.
			pSubQuery->uiLowDrn = 1;
			pSubQuery->uiHighDrn = 0xFFFFFFFF;
			// pSubQuery->uiNotEqualDrn = 0;			Should already be initialized.

			if (RC_BAD( rc = flmSQGenPredicateList( pDb, pSubQuery, pTempPool,
										&pPredicateList, &uiTotalPredicates,
										&bHaveUserPredicates)))
			{
				goto Do_Bad_Rc;
			}
		}

		// If one of the sub-queries has to do a container scan, there is no
		// point in optimizing any of the sub-queries on indexes.  We will just
		// merge all sub-queries into a single one so that we only do the
		// container scan once.

		if (pContainerScanSubQuery)
		{

			// Need to check the predicates, even though it is a container
			// scan so that we will call the searchCost() method.

			if (RC_BAD( rc = flmCheckUserPredicateCosts( pDb,
										pSubQuery, FALSE)))
			{
				goto Do_Bad_Rc;
			}
			if( RC_BAD( rc = flmMergeSubQueries( pCursor, &pSubQuery,
																pContainerScanSubQuery,
																TRUE)))
			{
				goto Do_Bad_Rc;
			}
			if( pSubQuery)
			{
				continue;
			}
			else
			{
				break;
			}
		}

		// If no index number has been set, find a best index to satisfy the
		// query. NOTE: as the best index is chosen, a field group will be
		// built for it in the subquery.

		if (bChoosingIndex)
		{
			if (bStratified)
			{
				if( RC_BAD( rc = flmSQChooseBestIndex( pCursor, pDb,
											0, FALSE, pSubQuery,
											pTempPool, pPredicateList,
											uiTotalPredicates, bHaveUserPredicates)))
				{
					goto Do_Bad_Rc;
				}

				// See if there is a better embedded user predicate

				if (RC_BAD( rc = flmCheckUserPredicateCosts( pDb,
												pSubQuery, TRUE)))
				{
					goto Do_Bad_Rc;
				}
			}
			else
			{

				if (RC_BAD( rc= flmSQSetupFullContainerScan( pCursor,
												pSubQuery)))
				{
					goto Do_Bad_Rc;
				}

				// Must make this call so that each user predicate
				// is traversed.  Must do AFTER setting up the full
				// container scan because we want to make sure we
				// add in the cost of doing any user predicates.

				if (RC_BAD( rc = flmCheckUserPredicateCosts( pDb,
												pSubQuery, FALSE)))
				{
					goto Do_Bad_Rc;
				}

			}
		}
		else
		{

			// If no index was specified, set up sub-query to do a container
			// scan.  Otherwise, call flmSQChooseBestIndex to set up keys
			// for the specified index.

			if (!pCursor->uiIndexNum)
			{
				if (RC_BAD( rc= flmSQSetupFullContainerScan( pCursor,
												pSubQuery)))
				{
					goto Do_Bad_Rc;
				}
			}
			else
			{
				if( RC_BAD( rc = flmSQChooseBestIndex( pCursor, pDb,
											pCursor->uiIndexNum, !bStratified,
											pSubQuery, pTempPool, pPredicateList,
											uiTotalPredicates, bHaveUserPredicates)))
				{
					goto Do_Bad_Rc;
				}
			}

			// Call flmCheckUserPredicateCosts so that searchCost gets
			// called for every user predicate.  Do only AFTER the setting
			// up of the sub-query so that the cost of processing the
			// predicates gets added in to the cost for the sub-query.

			if (RC_BAD( rc = flmCheckUserPredicateCosts( pDb,
									pSubQuery, FALSE)))
			{
				goto Do_Bad_Rc;
			}

		}

		// See if this sub-query should be merged with another one.

		eOptType = pSubQuery->OptInfo.eOptType;
		pTmpSubQuery = pCursor->pSubQueryList;
		switch (eOptType)
		{

			// If an index has been chosen or set, see if we need to merge
			// this sub-query with another subquery that has the same index.

			case QOPT_USING_INDEX:

				while (pTmpSubQuery != pSubQuery)
				{
					if (pTmpSubQuery->OptInfo.eOptType == QOPT_USING_INDEX &&
						 pTmpSubQuery->OptInfo.uiIxNum ==
						 pSubQuery->OptInfo.uiIxNum)
					{
						bFromSubQuerySubsumed = FALSE;
						goto Merge_SubQueries;
					}
					pTmpSubQuery = pTmpSubQuery->pNext;
				}

				// Didn't merge with any other sub-query, need to
				// get the index's language.

				if (pSubQuery->OptInfo.uiIxNum == FLM_DICT_INDEX)
				{
					pSubQuery->uiLanguage =
						pDb->pFile->FileHdr.uiDefaultLanguage;
				}
				else
				{
					IXD *	pIxd;

					// Get the index language.

					if (RC_BAD( rc = fdictGetIndex(
								pDb->pDict, pDb->pFile->bInLimitedMode,
								pSubQuery->OptInfo.uiIxNum, NULL, &pIxd)))
					{
						goto Exit;
					}

					pSubQuery->uiLanguage = pIxd->uiLanguage;
				}
				break;

			// If we optimized to a user predicate, merge it with any
			// sub-query that has the same user predicate.

			case QOPT_USING_PREDICATE:
				while (pTmpSubQuery != pSubQuery)
				{
					if (pTmpSubQuery->OptInfo.eOptType == QOPT_USING_PREDICATE &&
						 pTmpSubQuery->pPredicate == pSubQuery->pPredicate)
					{
						bFromSubQuerySubsumed = TRUE;
						goto Merge_SubQueries;
					}
					pTmpSubQuery = pTmpSubQuery->pNext;
				}
				pSubQuery->uiLanguage =
					pDb->pFile->FileHdr.uiDefaultLanguage;
				break;

			// Merge single record retrieve sub-query with other
			// sub-query doing a single record retrieve of the same
			// record or a partial container scan.

			case QOPT_SINGLE_RECORD_READ:
				while (pTmpSubQuery != pSubQuery)
				{
					if ((pTmpSubQuery->OptInfo.eOptType ==
								QOPT_SINGLE_RECORD_READ &&
						  pTmpSubQuery->OptInfo.uiDrn ==
								pSubQuery->OptInfo.uiDrn) ||
						 (pTmpSubQuery->OptInfo.eOptType ==
								QOPT_PARTIAL_CONTAINER_SCAN))
					{
						bFromSubQuerySubsumed =
							(pTmpSubQuery->OptInfo.eOptType ==
							 QOPT_SINGLE_RECORD_READ)
							 ? TRUE
							 : FALSE;
						goto Merge_SubQueries;
					}
					pTmpSubQuery = pTmpSubQuery->pNext;
				}
				pSubQuery->uiLanguage =
					pDb->pFile->FileHdr.uiDefaultLanguage;
				break;

			// Merge partial container scan sub-queries with other sub-query
			// doing a single record retrieve or a partial container scan.

			case QOPT_PARTIAL_CONTAINER_SCAN:
				while (pTmpSubQuery != pSubQuery)
				{
					if (pTmpSubQuery->OptInfo.eOptType ==
														QOPT_SINGLE_RECORD_READ ||
						 pTmpSubQuery->OptInfo.eOptType ==
									QOPT_PARTIAL_CONTAINER_SCAN)
					{
						bFromSubQuerySubsumed = FALSE;
						goto Merge_SubQueries;
					}
					pTmpSubQuery = pTmpSubQuery->pNext;
				}
				pSubQuery->uiLanguage =
					pDb->pFile->FileHdr.uiDefaultLanguage;
				break;

			// Merge full container scan sub-query with ALL sub-queries
			// that have been processed so far.  All subsequent
			// sub-queries will also be merged with this sub-query
			// (see above).

			case QOPT_FULL_CONTAINER_SCAN:
				pContainerScanSubQuery = pSubQuery;
				while (pTmpSubQuery != pSubQuery)
				{
					if (RC_BAD( rc = flmMergeSubQueries( pCursor,
											&pTmpSubQuery, pSubQuery, TRUE)))
					{
						goto Do_Bad_Rc;
					}
				}
				pSubQuery->uiLanguage =
					pDb->pFile->FileHdr.uiDefaultLanguage;
				break;

			default:

				// Should never hit this case!

				flmAssert( 0);
		}

		// Go to the next sub-query.

		if (pTmpSubQuery != pSubQuery)
		{
Merge_SubQueries:
			if (RC_BAD( rc = flmMergeSubQueries( pCursor,
										&pSubQuery, pTmpSubQuery,
										bFromSubQuerySubsumed)))
			{
				goto Do_Bad_Rc;
			}
		}
		else
		{
			pSubQuery = pSubQuery->pNext;
		}
		if (!pSubQuery)
		{
			break;
		}
		continue;

Do_Bad_Rc:
		if (rc == FERR_EMPTY_QUERY)
		{
			if (pSubQuery->pNext)
			{
				rc = FERR_OK;
				pSubQuery = pSubQuery->pNext;
				flmClipSubQuery( pCursor, pSubQuery->pPrev);
				continue;
			}
			else if (pSubQuery->pPrev)
			{
				rc = FERR_OK;
				flmClipSubQuery( pCursor, pSubQuery);
				pSubQuery->pPrev->pNext = NULL;
				break;
			}
		}
		goto Exit;
	}

	// Set cursor up so it always has a current sub-query.
	// Cannot do this until the end, because pSubQueryList
	// might change due to merges, etc.

	pCursor->pCurrSubQuery = pCursor->pSubQueryList;

Exit:
	if (pDb)
	{
		(void)fdbExit( pDb);
	}
	return( rc);
}
