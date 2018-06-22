//-------------------------------------------------------------------------
// Desc:	Query stack
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

static FLMUINT PrecedenceTable [FLM_USER_PREDICATE - FLM_AND_OP + 1] =
{
	2,		// FLM_AND_OP
	1,		// FLM_OR_OP
	10,	// FLM_NOT_OP
	6,		// FLM_EQ_OP
	6,		// FLM_MATCH_OP
	6,		// FLM_MATCH_BEGIN_OP
	6,		// FLM_MATCH_END_OP
	6,		// FLM_CONTAINS_OP
	6,		// FLM_NE_OP
	7,		// FLM_LT_OP
	7,		// FLM_LE_OP
	7,		// FLM_GT_OP
	7,		// FLM_GE_OP
	5,		// FLM_BITAND_OP
	3,		// FLM_BITOR_OP
	4,		// FLM_BITXOR_OP
	9,		// FLM_MULT_OP
	9,		// FLM_DIV_OP
	9,		// FLM_MOD_OP
	8,		// FLM_PLUS_OP
	8,		// FLM_MINUS_OP
	10,	// FLM_NEG_OP
	0,		// FLM_LPAREN_OP
	0,		// FLM_RPAREN_OP
	0,		// FLM_UNKNOWN
	6		// FLM_USER_PREDICATE
};

#define PRECEDENCE( e) (((e) >= FLM_AND_OP && (e) <= FLM_USER_PREDICATE) \
								? PrecedenceTable [(e) - FLM_AND_OP] \
								: (FLMUINT)0)

/****************************************************************************
Desc : Adds an operator to the selection criteria of a given cursor.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmCursorAddOp(
	HFCURSOR		hCursor,
	QTYPES		eOperator,
	FLMBOOL		bResolveUnknown
	)
{
	RCODE			rc = FERR_OK;
	CURSOR *		pCursor = (CURSOR *)hCursor;
	FQNODE *		pTmpQNode;
	FQNODE *		pTmpGraftNode;
	FQNODE *		pTmpChildNode;
	FLMBOOL		bDecrementNestLvl = FALSE;
	FLMUINT		uiFlags = bResolveUnknown ? FLM_RESOLVE_UNK : 0;

	if (!pCursor)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_INVALID_PARM);
		goto Exit;
	}
	if (RC_BAD( rc = pCursor->rc))
	{
		goto Exit;
	}

	// If a read operation has already been performed on this query, no
	// selection criteria may be added.
	
	if (pCursor->bOptimized)
	{
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}
	
	// If the operator is a left paren, link it as the last sibling in the
	// argument list of the current operator.

	if (eOperator == FLM_LPAREN_OP)
	{
		(pCursor->QTInfo.uiNestLvl)++;
		goto Exit;
	}

	// If it is a right paren, find the left paren and close it out

	if (eOperator == FLM_RPAREN_OP)
	{
		if (!pCursor->QTInfo.uiNestLvl)
		{
			rc = RC_SET( FERR_CURSOR_SYNTAX);
			goto Exit;
		}
		(pCursor->QTInfo.uiNestLvl)--;
		goto Exit;
	}
		
	// If it is not an operator, return an error

	if (!IS_OP( eOperator))
	{
		rc = RC_SET( FERR_CURSOR_SYNTAX);
		goto Exit;
	}

	// If an operator is not expected, bail out

	if (!(pCursor->QTInfo.uiExpecting & FLM_Q_OPERATOR) &&
		 eOperator != FLM_NEG_OP &&
		 eOperator != FLM_NOT_OP)
	{
		rc = RC_SET( FERR_CURSOR_SYNTAX);
		goto Exit;
	}

	// Make a QNODE and find a place for it in the query tree

	if (RC_BAD( rc = flmCurMakeQNode( &pCursor->QueryPool, eOperator,
								NULL, 0, uiFlags, &pTmpQNode)))
	{
		goto Exit;
	}
	pTmpQNode->uiNestLvl = pCursor->QTInfo.uiNestLvl;

	// If this is the first operator in the query, set the current operator
	// to it and graft in the current operand as its child.  NOTE:  there
	// should always be a current operand at this point.

	if (!pCursor->QTInfo.pTopNode)
	{
		pCursor->QTInfo.pTopNode = pTmpQNode;
		pCursor->QTInfo.pCurOpNode = pTmpQNode;
		if (pCursor->QTInfo.pCurAtomNode)
		{

			// If the current operand node is a user predicate, the only
			// thing that can become its parent is a logical operator.

			if (GET_QNODE_TYPE( pCursor->QTInfo.pCurAtomNode) ==
				 FLM_USER_PREDICATE && !IS_LOG_OP( eOperator))
			{
				rc = RC_SET( FERR_CURSOR_SYNTAX);
				goto Exit;
			}
			flmCurLinkLastChild( pTmpQNode, pCursor->QTInfo.pCurAtomNode);
		}
		pCursor->QTInfo.uiExpecting = FLM_Q_OPERAND;
		goto Exit;
	}

	// Go up the stack until an operator whose nest level or precedence is < 
	// this one's is encountered, then link this one in as the last child

	for (pTmpChildNode = NULL, pTmpGraftNode = pCursor->QTInfo.pCurOpNode;
				;
		  pTmpChildNode = pTmpGraftNode, pTmpGraftNode = pTmpGraftNode->pParent)
	{
		if (pTmpGraftNode->uiNestLvl < pTmpQNode->uiNestLvl ||
			 (pTmpGraftNode->uiNestLvl == pTmpQNode->uiNestLvl &&
			  PRECEDENCE( pTmpGraftNode->eOpType) < PRECEDENCE( eOperator)))
		{

			// If the node under which this operator is to be grafted already
			//	has two children, or if its child is at a greater nesting level,
			//	link the child as the last child of this operator.  Example:
			//	((A - B) == C) && (((D + E) * F) == G).
			//	When the '*' operator in this expression is added, it will be
			//	grafted as the last child of the '&&' operator.  But the '+'
			//	must first be unlinked from the '&&' and then linked as the child
			//	of the '*'.  Otherwise, they will be siblings, and the expression
			//	will be evaluated incorrectly.

			if (pTmpChildNode &&
				 (pTmpChildNode->uiNestLvl > pTmpQNode->uiNestLvl ||
				  pTmpChildNode->pPrevSib != NULL ||
				  pTmpGraftNode->eOpType == FLM_NEG_OP ||
				  pTmpGraftNode->eOpType == FLM_NOT_OP))
			{
				flmCurLinkLastChild( pTmpQNode, pTmpChildNode);
			}

			// If this operator is to be grafted into the query tree at the leaf
			//	level, link the current operand as its last child.  Examples:
			//	in A * (B + C), we want B to be linked to +;
			//	in A + B * C, we want B linked to *.

			if (pTmpGraftNode == pCursor->QTInfo.pCurOpNode &&
				 eOperator != FLM_NEG_OP &&
				 eOperator != FLM_NOT_OP)
			{

				// If the current operand node is a user predicate, the only
				// thing that can become its parent is a logical operator.

				if (pCursor->QTInfo.pCurAtomNode &&
					 GET_QNODE_TYPE( pCursor->QTInfo.pCurAtomNode) ==
						FLM_USER_PREDICATE && !IS_LOG_OP( eOperator))
				{
					rc = RC_SET( FERR_CURSOR_SYNTAX);
					goto Exit;
				}
				flmCurLinkLastChild( pTmpQNode, pCursor->QTInfo.pCurAtomNode);
			}
			flmCurLinkLastChild( pTmpGraftNode, pTmpQNode);
			break;
		}
		if (!pTmpGraftNode->pParent)
		{
			pCursor->QTInfo.pTopNode = pTmpQNode;
			flmCurLinkLastChild( pTmpQNode, pTmpGraftNode);
			break;
		}
	}

	pCursor->QTInfo.pCurOpNode = pTmpQNode;
	pCursor->QTInfo.uiExpecting = FLM_Q_OPERAND;

Exit:
	if (bDecrementNestLvl)
	{
		(pCursor->QTInfo.uiNestLvl)--;
	}
	if (pCursor)
	{
		pCursor->rc = rc;
	}
	return( rc);
}

/****************************************************************************
Desc: Add a reference to an embedded user predicate.
****************************************************************************/
RCODE flmCurAddRefPredicate(
	QTINFO *					pQTInfo,
	FlmUserPredicate *	pPredicate
	)
{
	RCODE		rc = FERR_OK;

	if (pQTInfo->uiNumPredicates == pQTInfo->uiMaxPredicates)
	{

		// Are we still in the embedded array? or have we
		// done an allocation?

		if (pQTInfo->uiMaxPredicates == MAX_USER_PREDICATES)
		{
			if (RC_BAD( rc = f_calloc(
										sizeof( FlmUserPredicate *) *
										(MAX_USER_PREDICATES * 2),
										&pQTInfo->ppPredicates)))
			{
				goto Exit;
			}

			// Copy all old pointers from embedded array.

			f_memcpy( pQTInfo->ppPredicates,
						 &pQTInfo->Predicates [0],
						 MAX_USER_PREDICATES * sizeof( FlmUserPredicate *));
		}
		else
		{

			// Reallocate the structure.

			if (RC_BAD( rc = f_recalloc(
										sizeof( FlmUserPredicate *) *
										(pQTInfo->uiNumPredicates * 2),
										&pQTInfo->ppPredicates)))
			{
				goto Exit;
			}
		}
		pQTInfo->uiMaxPredicates *= 2;
	}

	pQTInfo->ppPredicates [pQTInfo->uiNumPredicates] = pPredicate;
	pPredicate->AddRef();
	pQTInfo->uiNumPredicates++;
Exit:
	return( rc);
}

/****************************************************************************
Desc: Adds an embedded user predicate.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmCursorAddUserPredicate(
	HFCURSOR					hCursor,
	FlmUserPredicate *	pPredicate
	)
{
	RCODE			rc = FERR_OK;
	CURSOR *		pCursor = (CURSOR *)hCursor;
	FQNODE *		pQNode;
	QTYPES		eOperator;

	if (!pCursor || !pPredicate)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_INVALID_PARM);
		goto Exit;
	}

	if (RC_BAD( rc = pCursor->rc))
	{
		goto Exit;
	}

	// If a read operation has already been performed on this query, no
	// selection criteria may be added.
	
	if (pCursor->bOptimized)
	{
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	// Better be expecting an operand.
	
	if (!(pCursor->QTInfo.uiExpecting & FLM_Q_OPERAND))
	{
		rc = RC_SET( FERR_CURSOR_SYNTAX);
		goto Exit;
	}

	// Make the user predicate node.

	if (RC_OK( rc = flmCurMakeQNode( &pCursor->QueryPool,
									FLM_USER_PREDICATE, NULL,
									0, pCursor->QTInfo.uiFlags,
									&(pCursor->QTInfo.pCurAtomNode))))
	{
		if (pCursor->QTInfo.pCurOpNode)
		{
			eOperator = GET_QNODE_TYPE( pCursor->QTInfo.pCurOpNode);

			// Operator above a user predicate must be a logical operator.

			if (!IS_LOG_OP( eOperator))
			{
				rc = RC_SET( FERR_CURSOR_SYNTAX);
				goto Exit;
			}
			flmCurLinkLastChild( pCursor->QTInfo.pCurOpNode,
					pCursor->QTInfo.pCurAtomNode);
		}
		pCursor->QTInfo.uiExpecting &= ~FLM_Q_OPERAND;
		pCursor->QTInfo.uiExpecting |= FLM_Q_OPERATOR;
		pQNode = pCursor->QTInfo.pCurAtomNode;
		pQNode->pQAtom->eType = FLM_USER_PREDICATE;
		pQNode->pQAtom->val.pPredicate = pPredicate;

		if (RC_BAD( rc = flmCurAddRefPredicate( &pCursor->QTInfo, pPredicate)))
		{
			goto Exit;
		}

		// Don't do an addRef on pPredicate because there is really no place
		// to call a corresponding release.
	}

Exit:
	if (pCursor)
	{
		pCursor->rc = rc;
	}
	return( rc);
}

/****************************************************************************
Desc:	Links one FQNODE as the first child of another.
****************************************************************************/
void flmCurLinkFirstChild(
	FQNODE * pParent,
	FQNODE * pChild
	)
{
	FQNODE *	pTmpQNode;

	// If necessary, unlink the child from a sibling list and link it back in
	// as the first child.

	if (pChild->pPrevSib)
	{
		pChild->pPrevSib->pNextSib = pChild->pNextSib;
		if (pChild->pNextSib)
		{
			pChild->pNextSib->pPrevSib = pChild->pPrevSib;
		}
		for (pTmpQNode = pChild;
				pTmpQNode->pPrevSib;
				pTmpQNode = pTmpQNode->pPrevSib)
		{
			;
		}
		pChild->pNextSib = pTmpQNode;
		pTmpQNode->pPrevSib = pChild;
	}

	// If there is already a child node, link this one (and any siblings)
	// before it.

	if (pParent->pChild)
	{
		for (pTmpQNode = pChild;
				pTmpQNode->pNextSib;
				pTmpQNode = pTmpQNode->pNextSib)
		{
			pTmpQNode->pParent = pParent;
		}
		pParent->pChild->pPrevSib = pTmpQNode;
		pTmpQNode->pNextSib = pParent->pChild;
	}
	pParent->pChild = pChild;
	pChild->pParent = pParent;
	pChild->pPrevSib = NULL;
}

/****************************************************************************
Desc:	Links one FQNODE as the last child of another.
****************************************************************************/
void flmCurLinkLastChild(
	FQNODE *	pParent,
	FQNODE *	pChild
	)
{
	FQNODE *	pTmpQNode;

	// If necessary, unlink the child from any parent or siblings

	if (pChild->pParent)
	{
		if (!pChild->pPrevSib)
		{
			pChild->pParent->pChild = pChild->pNextSib;
			if (pChild->pNextSib)
			{
				pChild->pNextSib->pPrevSib = NULL;
			}
		}
		else
		{
			pChild->pPrevSib->pNextSib = pChild->pNextSib;
			if (pChild->pNextSib)
			{
				pChild->pNextSib->pPrevSib = pChild->pPrevSib;
			}
		}
	}

	// Link pChild as the next sibling to the last node in the sibling list
	// of pParent->pChild

	if (pParent->pChild)
	{
		for (pTmpQNode = pParent->pChild;
				pTmpQNode->pNextSib;
				pTmpQNode = pTmpQNode->pNextSib)
		{
			;
		}
		pTmpQNode->pNextSib = pChild;
		pChild->pPrevSib = pTmpQNode;
	}
	else
	{
		pParent->pChild = pChild;
		pChild->pPrevSib = NULL;
	}
	pChild->pParent = pParent;
	pChild->pNextSib = NULL;
}

/****************************************************************************
Desc:	Put a value in an FQATOM node - so we can call it from SMI.  Can
		only be used for values, not field paths.  Cannot be used for
		unknown values.
****************************************************************************/
RCODE flmPutValInAtom(
	void *	pAtom,
	QTYPES	eValType,
	void *	pvVal,
	FLMUINT	uiValLen,
	FLMUINT	uiFlags
	)
{
	RCODE		rc = FERR_OK;
	FQATOM *	pQAtom = (FQATOM *)pAtom;

	pQAtom->uiFlags = uiFlags;
	pQAtom->eType = eValType;
	switch (eValType)
	{
		case FLM_BOOL_VAL:
			pQAtom->val.uiBool = *((FLMUINT *)pvVal);
			break;
		case FLM_UINT32_VAL:
		case FLM_REC_PTR_VAL:
			pQAtom->val.ui32Val = *((FLMUINT32 *)pvVal);
			break;
		case FLM_UINT64_VAL:
			pQAtom->val.ui64Val = *((FLMUINT64 *)pvVal);
			break;
		case FLM_INT32_VAL:
			pQAtom->val.i32Val = *((FLMINT32 *)pvVal);
			break;
		case FLM_INT64_VAL:
			pQAtom->val.i64Val = *((FLMINT64 *)pvVal);
			break;
		case FLM_BINARY_VAL:
		case FLM_TEXT_VAL:
			pQAtom->val.pucBuf = (FLMBYTE *)pvVal;
			pQAtom->uiBufLen = uiValLen;
			break;
		case FLM_UNKNOWN:
			break;
		default:
			flmAssert( 0);
			rc = RC_SET( FERR_CURSOR_SYNTAX);
			goto Exit;
	}
Exit:
	return( rc);
}

/****************************************************************************
Desc:	Makes an FQNODE of a given type, and puts a value in it if necessary.
****************************************************************************/
RCODE flmCurMakeQNode(
	F_Pool *		pPool,
	QTYPES		eType,
	void *		pVal,
	FLMUINT		uiValLen,
	FLMUINT		uiFlags,
	FQNODE * *	ppQNode)
{
	FLMUINT *	puiTmpPath;
	FLMUINT *	puiPToCPath;
	FLMUINT *	puiFldPath;
	FLMUINT		uiTmpLen = uiValLen;
	FLMBYTE *	pTmpBuf;
	FLMUINT		uiPathCnt;
	FLMUINT		uiCnt;
	RCODE			rc = FERR_OK;
	FQNODE *		pQNode;
	FQATOM *		pQAtom;

	if( RC_BAD( rc = pPool->poolCalloc( sizeof( FQNODE), (void **)&pQNode)))
	{
		goto Exit;
	}
	
	*ppQNode = pQNode;

	// Always set eOpType to the eType

	pQNode->eOpType = eType;
	if (IS_OP( eType))
	{
		pQNode->uiStatus = uiFlags;
		goto Exit;
	}
	
	if( RC_BAD( rc = pPool->poolCalloc( sizeof( FQATOM), (void **)&pQAtom)))
	{
		goto Exit;
	}
	
	pQNode->pQAtom = pQAtom;
	pQAtom->uiFlags = uiFlags;

	switch (eType)
	{
		case FLM_TEXT_VAL:
		{
			if( RC_BAD( rc = pPool->poolAlloc( uiTmpLen + 1, (void **)&pTmpBuf)))
			{
				goto Exit;
			}
			
			f_memcpy( pTmpBuf, pVal, uiTmpLen);
			pTmpBuf[ uiTmpLen ] = '\0';
			pQAtom->val.pucBuf = pTmpBuf;
			pQAtom->uiBufLen = uiTmpLen;
			break;
		}
		
		case FLM_BOOL_VAL:
		{
			pQAtom->val.uiBool = *(FLMUINT *)pVal;
			break;
		}
		
		case FLM_INT32_VAL:
		{
			pQAtom->val.i32Val = *(FLMINT32 *)pVal;
			break;
		}
		
		case FLM_INT64_VAL:
		{
			pQAtom->val.i64Val = *(FLMINT64 *)pVal;
			break;
		}
		
		case FLM_REC_PTR_VAL:
		case FLM_UINT32_VAL:
		{
			pQAtom->val.ui32Val = *(FLMUINT32 *)pVal;
			break;
		}
		
		case FLM_UINT64_VAL:
		{
			pQAtom->val.ui64Val = *(FLMUINT64 *)pVal;
			break;
		}
		
		case FLM_FLD_PATH:
		{
			for (uiPathCnt = 0;
					((FLMUINT *)pVal)[ uiPathCnt];
					uiPathCnt++)
			{
				if (uiPathCnt > GED_MAXLVLNUM + 1)
				{
					rc = RC_SET( FERR_CURSOR_SYNTAX);
					goto Exit;
				}
			}
			
			if( RC_BAD( rc = pPool->poolCalloc(
				((FLMUINT)(uiPathCnt + 1) * 2 * sizeof( FLMUINT)),
				(void **)&puiTmpPath)))
			{
				goto Exit;
			}
				
			puiPToCPath = &puiTmpPath [uiPathCnt + 1];

			puiFldPath = (FLMUINT *)pVal;
			for (uiCnt = 0; uiCnt < uiPathCnt; uiCnt++)
			{
				puiTmpPath[ uiPathCnt - uiCnt - 1] = puiFldPath[ uiCnt];
				puiPToCPath [uiCnt] = puiFldPath [uiCnt];
			}
			pQAtom->val.QueryFld.puiFldPath = puiTmpPath;
			pQAtom->val.QueryFld.puiPToCPath = puiPToCPath;
			break;
		}
		
		case FLM_BINARY_VAL:
		{
			if( RC_BAD( rc = pPool->poolAlloc( uiTmpLen, (void **)&pTmpBuf)))
			{
				goto Exit;
			}
			
			f_memcpy( pTmpBuf, pVal, uiTmpLen);
			pQAtom->val.pucBuf = pTmpBuf;
			pQAtom->uiBufLen = uiTmpLen;
			break;
		}
		
		case FLM_USER_PREDICATE:
		{
			break;
		}
		
		default:
		{
			rc = RC_SET( FERR_CURSOR_SYNTAX);
			goto Exit;
		}
	}
	
	pQAtom->eType = eType;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Grafts FQNODE onto a passed-in query tree as the right branch of a new
		root node which contains the passed-in operator.
Ret:
****************************************************************************/
RCODE flmCurGraftNode(
	F_Pool *		pPool,
	FQNODE * 	pQNode,
	QTYPES		eGraftOp,
	FQNODE *  * ppQTree)
{
	FQNODE *		pTmpQNode;
	RCODE			rc = FERR_OK;

	if (*ppQTree == NULL)
	{
		*ppQTree = pQNode;
		goto Exit;
	}

	if (RC_BAD( rc = flmCurMakeQNode( pPool, eGraftOp, NULL,
															0, 0, &pTmpQNode)))
	{
		goto Exit;
	}

	flmCurLinkLastChild( pTmpQNode, *ppQTree);
	flmCurLinkLastChild( pTmpQNode, pQNode);
	*ppQTree = pTmpQNode;

Exit:
	return( rc);
}

/****************************************************************************
Desc : Adds a value to the selection criteria of a given cursor.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmCursorAddValue(
	HFCURSOR		hCursor,
	QTYPES		eValType,
	void *		pVal,
	FLMUINT		uiValLen
	)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiVal;
	void *		pTmpVal = pVal;
	CURSOR *		pCursor = (CURSOR *)hCursor;
	F_Pool		pool;
	FLMBOOL		bPoolInitialized = FALSE;

	if (!pCursor)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_INVALID_PARM);
		goto Exit;
	}
	
	if (RC_BAD( rc = pCursor->rc))
	{
		goto Exit;
	}

	// If a read operation has already been performed on this query, no
	// selection criteria may be added.
	
	if (pCursor->bOptimized)
	{
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}
	
	if (!( pCursor->QTInfo.uiExpecting & FLM_Q_OPERAND))
	{
		rc = RC_SET( FERR_CURSOR_SYNTAX);
		goto Exit;
	}

	switch (eValType)
	{

		// Convert all string types to FLM_TEXT_VALUE 
		//	in order to handle pure unicode coming in.

		case FLM_UNICODE_VAL:
		case FLM_STRING_VAL:
		{
			NODE	node;

			f_memset( &node, 0, sizeof(NODE));

			pool.poolInit( 512);
			bPoolInitialized = TRUE;

			rc = (eValType == FLM_UNICODE_VAL) 
					? GedPutUNICODE( &pool, &node, (FLMUNICODE *) pVal)
					: GedPutNATIVE( &pool, &node, (const char *)pVal);
			if (RC_BAD( rc))
			{
				goto Exit;
			}

			pTmpVal = GedValPtr( &node);
			uiValLen = GedValLen( &node);
			eValType = FLM_TEXT_VAL;
			break;
		}

		case FLM_BOOL_VAL:
			if (!pVal)
			{
				uiVal = FLM_UNK;
			}
			else
			{
				FLMBOOL bTrueFalse = (FLMBOOL)*(FLMBOOL *)pVal;
				uiVal = (bTrueFalse) ? FLM_TRUE : FLM_FALSE;
			}
			pTmpVal = &uiVal;
			eValType = FLM_BOOL_VAL;
			break;

		case FLM_INT32_VAL:
		case FLM_UINT32_VAL:
		case FLM_REC_PTR_VAL:
		case FLM_UINT64_VAL:
		case FLM_INT64_VAL:
		case FLM_TEXT_VAL:
		case FLM_BINARY_VAL:
		
			// pTmpVal is already pointing to pVal, and
			// eValType does not need to be changed.
			
			break;

		default:
			flmAssert( 0);
			rc = RC_SET( FERR_CURSOR_SYNTAX);
			break;
	}

	if (RC_OK( rc = flmCurMakeQNode( &pCursor->QueryPool, eValType, pTmpVal,
									uiValLen, pCursor->QTInfo.uiFlags,
									&(pCursor->QTInfo.pCurAtomNode))))
	{
		if (pCursor->QTInfo.pCurOpNode)
		{
			flmCurLinkLastChild( pCursor->QTInfo.pCurOpNode,
					pCursor->QTInfo.pCurAtomNode);
		}
		pCursor->QTInfo.uiExpecting &= ~FLM_Q_OPERAND;
		pCursor->QTInfo.uiExpecting |= FLM_Q_OPERATOR;
	}

Exit:

	if (pCursor)
	{
		pCursor->rc = rc;
	}

	if (bPoolInitialized)
	{
		pool.poolFree();
	}
	
	return( rc);
}

/****************************************************************************
Desc:	Adds a field ID to the selection criteria of a given cursor.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmCursorAddField(
	HFCURSOR		hCursor,
	FLMUINT		uiFldId,
	FLMUINT		uiFlags)
{
	RCODE			rc = FERR_OK;
	CURSOR *		pCursor = (CURSOR *)hCursor;
	FQNODE *		pTmpQNode;
	FLMUINT		uiPath [2];

	if (!pCursor)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_INVALID_PARM);
		goto Exit;
	}

	if (RC_BAD( rc = pCursor->rc))
	{
		goto Exit;
	}

	// If a read operation has already been performed on this query, no
	// selection criteria may be added.
	
	if (pCursor->bOptimized)
	{
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}
	
	if (!( pCursor->QTInfo.uiExpecting & FLM_Q_OPERAND))
	{
		rc = RC_SET( FERR_CURSOR_SYNTAX);
		goto Exit;
	}

	if (uiFldId == FLM_RECID_FIELD)
	{
		uiFlags |= FLM_SINGLE_VALUED;
	}

	// Set up a field path with one element in it.

	uiPath [0] = uiFldId;
	uiPath [1] = 0;
	if (RC_OK( rc = flmCurMakeQNode( &pCursor->QueryPool, FLM_FLD_PATH,
							&uiPath [0], 0, pCursor->QTInfo.uiFlags, &pTmpQNode)))
	{
		pTmpQNode->pQAtom->uiFlags |= uiFlags;
		pCursor->QTInfo.pCurAtomNode = pTmpQNode;
		if (pCursor->QTInfo.pCurOpNode)
		{
			flmCurLinkLastChild( pCursor->QTInfo.pCurOpNode,
					pCursor->QTInfo.pCurAtomNode);
		}
		pCursor->QTInfo.uiExpecting &= ~FLM_Q_OPERAND;
		pCursor->QTInfo.uiExpecting |= FLM_Q_OPERATOR;
	}

Exit:
	if (pCursor)
	{
		pCursor->rc = rc;
	}
	return( rc);
}

/****************************************************************************
Desc:	Adds a field path to the selection criteria of a given cursor.  A
		field path is the fully qualified context of a field within a record.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmCursorAddFieldPath(
	HFCURSOR		hCursor,
	FLMUINT *	puiFldPath,
	FLMUINT		uiFlags)
{
	RCODE			rc = FERR_OK;
	FQNODE *		pTmpQNode;
	CURSOR *		pCursor = (CURSOR *)hCursor;

	if (!pCursor)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_INVALID_PARM);
		goto Exit;
	}
	if (RC_BAD( rc = pCursor->rc))
	{
		goto Exit;
	}

	// If a read operation has already been performed on this query, no
	// selection criteria may be added.
	
	if (pCursor->bOptimized)
	{
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}
	
	if (!( pCursor->QTInfo.uiExpecting & FLM_Q_OPERAND))
	{
		rc = RC_SET( FERR_CURSOR_SYNTAX);
		goto Exit;
	}

	if (RC_OK( rc = flmCurMakeQNode( &pCursor->QueryPool, FLM_FLD_PATH,
									puiFldPath, 0, pCursor->QTInfo.uiFlags,
									&pTmpQNode)))
	{
		pTmpQNode->pQAtom->uiFlags |= uiFlags;
		pCursor->QTInfo.pCurAtomNode = pTmpQNode;
		if (pCursor->QTInfo.pCurOpNode)
		{
			flmCurLinkLastChild( pCursor->QTInfo.pCurOpNode,
										pCursor->QTInfo.pCurAtomNode);
		}
		pCursor->QTInfo.uiExpecting &= ~FLM_Q_OPERAND;
		pCursor->QTInfo.uiExpecting |= FLM_Q_OPERATOR;
	}

Exit:
	if (pCursor)
	{
		pCursor->rc = rc;
	}
	return( rc);
}

/****************************************************************************
Desc:	Adds a field path to the selection criteria of a given cursor - with
		a callback to retrieve the field.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmCursorAddFieldCB(
	HFCURSOR					hCursor,
	FLMUINT *				puiFldPath,
	FLMUINT					uiFlags,
	FLMBOOL					bValidateOnly,
	CURSOR_GET_FIELD_CB	fnGetField,
	void *					pvUserData,
	FLMUINT					uiUserDataLen)
{
	RCODE			rc = FERR_OK;
	FQNODE *		pTmpQNode;
	CURSOR *		pCursor = (CURSOR *)hCursor;

	if (!pCursor)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_INVALID_PARM);
		goto Exit;
	}
	if (RC_BAD( rc = pCursor->rc))
	{
		goto Exit;
	}

	// If a read operation has already been performed on this query, no
	// selection criteria may be added.
	
	if (pCursor->bOptimized)
	{
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}
	
	if (!( pCursor->QTInfo.uiExpecting & FLM_Q_OPERAND))
	{
		rc = RC_SET( FERR_CURSOR_SYNTAX);
		goto Exit;
	}

	if (RC_OK( rc = flmCurMakeQNode( &pCursor->QueryPool, FLM_FLD_PATH,
									puiFldPath, 0, pCursor->QTInfo.uiFlags,
									&pTmpQNode)))
	{
		FQATOM *	pQAtom = pTmpQNode->pQAtom;

		pQAtom->val.QueryFld.fnGetField = fnGetField;
		pQAtom->val.QueryFld.bValidateOnly = bValidateOnly;
		
		if (pvUserData && uiUserDataLen)
		{
			if( RC_BAD( rc = pCursor->QueryPool.poolAlloc( uiUserDataLen,
				(void **)&pQAtom->val.QueryFld.pvUserData)))
			{
				goto Exit;
			}
			
			f_memcpy( pQAtom->val.QueryFld.pvUserData, pvUserData,
							uiUserDataLen);
			pQAtom->val.QueryFld.uiUserDataLen = uiUserDataLen;
		}
		else
		{
			pQAtom->val.QueryFld.pvUserData = NULL;
			pQAtom->val.QueryFld.uiUserDataLen = 0;
		}
		pQAtom->uiFlags |= uiFlags;
		pCursor->QTInfo.pCurAtomNode = pTmpQNode;
		if (pCursor->QTInfo.pCurOpNode)
		{
			flmCurLinkLastChild( pCursor->QTInfo.pCurOpNode,
										pCursor->QTInfo.pCurAtomNode);
		}
		pCursor->QTInfo.uiExpecting &= ~FLM_Q_OPERAND;
		pCursor->QTInfo.uiExpecting |= FLM_Q_OPERATOR;
	}

Exit:
	if (pCursor)
	{
		pCursor->rc = rc;
	}
	return( rc);
}
