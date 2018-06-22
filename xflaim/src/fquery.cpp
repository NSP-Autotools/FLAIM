//------------------------------------------------------------------------------
// Desc:	Contains the methods for F_Query class.
// Tabs:	3
//
// Copyright (c) 2003-2007 Novell, Inc. All Rights Reserved.
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
#include "fquery.h"
#include "fscursor.h"

#define MIN_OPT_COST	8

static FLMUINT uiPrecedenceTable[ XFLM_RPAREN_OP - XFLM_AND_OP + 1] =
{
	2,		// XFLM_AND_OP
	1,		// XFLM_OR_OP
	10,	// XFLM_NOT_OP
	6,		// XFLM_EQ_OP
	6,		// XFLM_NE_OP
	6,		// XFLM_APPROX_EQ_OP
	7,		// XFLM_LT_OP
	7,		// XFLM_LE_OP
	7,		// XFLM_GT_OP
	7,		// XFLM_GE_OP
	5,		// XFLM_BITAND_OP
	3,		// XFLM_BITOR_OP
	4,		// XFLM_BITXOR_OP
	9,		// XFLM_MULT_OP
	9,		// XFLM_DIV_OP
	9,		// XFLM_MOD_OP
	8,		// XFLM_PLUS_OP
	8,		// XFLM_MINUS_OP
	10,	// XFLM_NEG_OP
	0,		// XFLM_LPAREN_OP
	0		// XFLM_RPAREN_OP
};

FINLINE FLMUINT getPrecedence(
	eQueryOperators	eOperator)
{
	return( uiPrecedenceTable [eOperator - XFLM_AND_OP]);
}

FSTATIC void fqUnlinkFromParent(
	FQNODE *	pQNode);

FSTATIC void fqLinkFirstChild(
	FQNODE *	pParent,
	FQNODE *	pChild);

FSTATIC void fqLinkLastChild(
	FQNODE *	pParent,
	FQNODE *	pChild);

FSTATIC void fqReplaceNode(
	FQNODE *	pNodeToReplace,
	FQNODE *	pReplacementNode);

FSTATIC RCODE fqGetPosition(
	FQVALUE *	pQValue,
	FLMUINT *	puiPos);

FSTATIC RCODE fqCompareValues(
	FQVALUE *			pValue1,
	FLMBOOL				bInclusive1,
	FLMBOOL				bNullIsLow1,
	FQVALUE *			pValue2,
	FLMBOOL				bInclusive2,
	FLMBOOL				bNullIsLow2,
	FLMUINT				uiCompareRules,
	FLMUINT				uiLanguage,
	FLMINT *				piCmp);

FSTATIC RCODE fqCheckUnionPredicates(
	CONTEXT_PATH *	pContextPath,
	FLMUINT			uiLanguage,
	PATH_PRED *		pPred);

FSTATIC void fqClipContext(
	OP_CONTEXT *	pContext);

FSTATIC void fqImportChildContexts(
	OP_CONTEXT *	pDestContext,
	OP_CONTEXT *	pSrcContext);
	
FSTATIC void fqImportContextPaths(
	OP_CONTEXT *	pDestContext,
	OP_CONTEXT *	pSrcContext);
	
FSTATIC void fqImportContext(
	OP_CONTEXT *	pDestContext,
	OP_CONTEXT *	pSrcContext);
	
FSTATIC void fqMergeContexts(
	FQNODE *			pQNode,
	OP_CONTEXT *	pDestContext);

FSTATIC void fqCheckPathMatch(
	XPATH_COMPONENT *	pXPathContextComponent,
	XPATH_COMPONENT *	pXPathComponent);

FSTATIC FQNODE * fqEvalLogicalOperands(
	FQNODE *		pQNode);

FSTATIC FQNODE * fqClipNotNode(
	FQNODE *		pQNode,
	FQNODE **	ppExpr);

FSTATIC RCODE fqGetNodeIdValue(
	FQVALUE *	pQValue);

FSTATIC FLMBOOL haveChildKeyComponents(
	ICD *	pParentIcd);
	
FSTATIC RCODE fqEvalOperator(
	FLMUINT	uiLanguage,
	FQNODE *	pQNode);

FSTATIC void fqResetIterator(
	FQNODE *		pQNode,
	FLMBOOL		bFullRelease,
	FLMBOOL		bUseKeyNodes);

FSTATIC RCODE fqGetValueFromNode(
	F_Db *			pDb,
	IF_DOMNode *	pNode,
	FQVALUE *		pQValue,
	FLMUINT			uiMetaDataType);

FSTATIC void fqResetQueryTree(
	FQNODE *	pQueryTree,
	FLMBOOL	bUseKeyNodes,
	FLMBOOL	bResetAllXPaths);
	
FSTATIC RCODE fqTryEvalOperator(
	FLMUINT		uiLanguage,
	FQNODE **	ppCurrNode);
	
FSTATIC FQNODE * fqBackupTree(
	FQNODE *		pCurrNode,
	FLMBOOL *	pbGetNodeValue);
	
FSTATIC void fqReleaseQueryExpr(
	FQNODE *	pQNode);

FSTATIC FLMBOOL fqTestValue(
	FQNODE *	pQueryExpr);
	
FSTATIC RCODE fqGetValueFromKey(
	FLMUINT			uiDataType,
	F_DataVector *	pKey,
	FQVALUE *		pQValue,
	FLMBYTE **		ppucValue,
	FLMUINT			uiValueBufSize);

FSTATIC RCODE fqPredCompare(
	FLMUINT		uiLanguage,
	PATH_PRED *	pPred,
	FQVALUE *	pQValue,
	FLMBOOL *	pbPasses);

FSTATIC void fqMarkXPathNodeListPassed(
	PATH_PRED *			pPred);

FSTATIC int nodeIdCompareFunc(
	void *	pvData1,
	void *	pvData2,
	void *	pvUserData);

/***************************************************************************
Desc:	Constructor
***************************************************************************/
F_Query::F_Query()
{
	m_pool.poolInit( 1024);
	m_uiLanguage = FLM_US_LANG;
	m_uiCollection = XFLM_DATA_COLLECTION;
	initVars();
}

/***************************************************************************
Desc:	Destructor
***************************************************************************/
F_Query::~F_Query()
{
	clearQuery();
	m_pool.poolFree();
}

/***************************************************************************
Desc:	Reset function
***************************************************************************/
void F_Query::clearQuery( void)
{
	stopBuildingResultSet();
	resetQuery();

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
			m_pDatabase->m_pFirstQuery = m_pNext;
		}
		if (m_pNext)
		{
			m_pNext->m_pPrev = m_pPrev;
		}
		else
		{
			m_pDatabase->m_pLastQuery = m_pPrev;
		}
		m_pDatabase->unlockMutex();
	}
	
	if (m_pCurrDoc)
	{
		m_pCurrDoc->Release();
		m_pCurrDoc = NULL;
	}
	
	if (m_pCurrNode)
	{
		m_pCurrNode->Release();
		m_pCurrNode = NULL;
	}
	
	if (m_ppObjectList)
	{
		while (m_uiObjectCount)
		{
			m_uiObjectCount--;
			m_ppObjectList [m_uiObjectCount]->Release();
			m_ppObjectList [m_uiObjectCount] = NULL;
		}
		f_free( &m_ppObjectList);
	}

	if (m_pDocIdSet)
	{
		m_pDocIdSet->Release();
		m_pDocIdSet = NULL;
	}

	if (m_pFSIndexCursor)
	{
		m_pFSIndexCursor->Release();
		m_pFSIndexCursor = NULL;
	}

	// Clear all of the predicate cursors that may still be
	// laying around.

	if (m_pQuery && m_pQuery->pContext)
	{
		OP_CONTEXT *	pContext = m_pQuery->pContext;
		CONTEXT_PATH *	pContextPath;
		PATH_PRED *		pPred;

		for (;;)
		{

			// Clear predicates of the context we are in.

			pContextPath = pContext->pFirstPath;
			while (pContextPath)
			{
				pPred = pContextPath->pFirstPred;
				while (pPred)
				{

					// Predicate should only have either an index cursor
					// or a collection cursor or an app. predicate.

					if (pPred->pFSIndexCursor)
					{
						flmAssert( !pPred->pFSCollectionCursor);
						flmAssert( !pPred->pNodeSource);
						pPred->pFSIndexCursor->Release();
					}
					else if (pPred->pFSCollectionCursor)
					{
						flmAssert( !pPred->pFSIndexCursor);
						flmAssert( !pPred->pNodeSource);
						pPred->pFSCollectionCursor->Release();
					}
					else if (pPred->pNodeSource)
					{
						flmAssert( !pPred->pFSIndexCursor);
						flmAssert( !pPred->pFSCollectionCursor);
						pPred->pNodeSource->releaseResources();
					}
					pPred = pPred->pNext;
				}
				pContextPath = pContextPath->pNext;
			}

			if (pContext->pFirstChild)
			{
				pContext = pContext->pFirstChild;
				continue;
			}

			// Go to sibling context, if any

			while (!pContext->pNextSib)
			{
				if ((pContext = pContext->pParent) == NULL)
				{
					break;
				}
			}

			// If pContext is NULL at this point, there are no
			// more contexts.

			if (!pContext)
			{
				break;
			}
			pContext = pContext->pNextSib;

			// There has to have been a sibling context at this point.

			flmAssert( pContext);
		}
	}
	if (m_pSortResultSet)
	{
		m_pSortResultSet->Release();
		m_pSortResultSet = NULL;
	}
	if (m_pQueryStatus)
	{
		m_pQueryStatus->Release();
		m_pQueryStatus = NULL;
	}
	if (m_pQueryValidator)
	{
		m_pQueryValidator->Release();
		m_pQueryValidator = NULL;
	}
	initVars();
}

/***************************************************************************
Desc:	Initialize all the member variables.
***************************************************************************/
void F_Query::initVars( void)
{
	m_rc = NE_XFLM_OK;
	m_bScan = FALSE;
	m_bScanIndex = FALSE;
	m_bResetAllXPaths = FALSE;
	m_pFSIndexCursor = NULL;
	m_bEmpty = FALSE;
	m_pSortIxd = NULL;
	m_pSortResultSet = NULL;
	m_pFirstWaiter = NULL;
	m_bStopBuildingResultSet = FALSE;
	m_uiBuildThreadId = 0;
	m_bPositioningEnabled = FALSE;
	m_bResultSetPopulated = FALSE;
	m_bEntriesAlreadyInOrder = FALSE;
	m_bEncryptResultSet = FALSE;
	m_eState = XFLM_QUERY_NOT_POSITIONED;
	m_pQueryStatus = NULL;
	m_pQueryValidator = NULL;
	m_pDatabase = NULL;
	m_pPrev = NULL;
	m_pNext = NULL;
	m_bOptimized = FALSE;
	m_pCurrContext = NULL;
	m_pCurrContextPath = NULL;
	m_pCurrPred = NULL;
	m_pExprXPathSource = NULL;
	m_pQuery = NULL;
	m_pCurrOpt = NULL;
	m_pDb = NULL;
	m_pCurExprState = NULL;
	m_pCurrDoc = NULL;
	m_pCurrNode = NULL;
	m_ppObjectList = NULL;
	m_uiObjectListSize = 0;
	m_uiObjectCount = 0;
	m_bRemoveDups = FALSE;
	m_pDocIdSet = NULL;
	m_uiIndex = 0;
	m_bIndexSet = FALSE;
	m_uiTimeLimit = 0;
	m_uiStartTime = 0;
	m_pool.poolReset( NULL);
}

/***************************************************************************
Desc:	Allocate a new expression evaluation state structure
***************************************************************************/
RCODE F_Query::allocExprState( void)
{
	RCODE				rc = NE_XFLM_OK;
	EXPR_STATE *	pExprState;

	if (!m_pCurExprState || !m_pCurExprState->pNext)
	{
		if (RC_BAD( rc = m_pool.poolCalloc( sizeof( EXPR_STATE),
												(void **)&pExprState)))
		{
			goto Exit;
		}
		if ((pExprState->pPrev = m_pCurExprState) != NULL)
		{
			m_pCurExprState->pNext = pExprState;
		}
		m_pCurExprState = pExprState;
	}
	else
	{
		EXPR_STATE *	pSaveNext;
		EXPR_STATE *	pSavePrev;

		m_pCurExprState = m_pCurExprState->pNext;

		// Zero out everything except for the prev and next pointers

		pSaveNext = m_pCurExprState->pNext;
		pSavePrev = m_pCurExprState->pPrev;
		f_memset( m_pCurExprState, 0, sizeof( EXPR_STATE));
		m_pCurExprState->pNext = pSaveNext;
		m_pCurExprState->pPrev = pSavePrev;
	}
	m_pCurExprState->uiNumExprNeeded = 1;

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Unlinks a node from its parent and siblings.  This routine assumes
		that the test has already been made that the node has a parent.
***************************************************************************/
FSTATIC void fqUnlinkFromParent(
	FQNODE *		pQNode)
{
	flmAssert( pQNode->pParent);
	if (pQNode->pPrevSib)
	{
		pQNode->pPrevSib->pNextSib = pQNode->pNextSib;
	}
	else
	{
		pQNode->pParent->pFirstChild = pQNode->pNextSib;
	}
	if (pQNode->pNextSib)
	{
		pQNode->pNextSib->pPrevSib = pQNode->pPrevSib;
	}
	else
	{
		pQNode->pParent->pLastChild = pQNode->pPrevSib;
	}

	pQNode->pParent = NULL;
	pQNode->pPrevSib = NULL;
	pQNode->pNextSib = NULL;
}

/***************************************************************************
Desc:	Links one FQNODE as the first child of another.  Will unlink the
		child node from any parent it may be linked to.
***************************************************************************/
FSTATIC void fqLinkFirstChild(
	FQNODE *	pParent,
	FQNODE *	pChild
	)
{

	// If necessary, unlink the child from parent and siblings

	if (pChild->pParent)
	{
		fqUnlinkFromParent( pChild);
	}

	// Link child as the first child to parent

	pChild->pParent = pParent;
	pChild->pPrevSib = NULL;
	if ((pChild->pNextSib = pParent->pFirstChild) != NULL)
	{
		pChild->pNextSib->pPrevSib = pChild;
	}
	else
	{
		pParent->pLastChild = pChild;
	}
	pParent->pFirstChild = pChild;
}

/***************************************************************************
Desc:	Links one FQNODE as the last child of another.  Will unlink the
		child node from any parent it may be linked to.
***************************************************************************/
FSTATIC void fqLinkLastChild(
	FQNODE *	pParent,
	FQNODE *	pChild
	)
{

	// If necessary, unlink the child from parent and siblings

	if (pChild->pParent)
	{
		fqUnlinkFromParent( pChild);
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

/****************************************************************************
Desc:	Replace one node with another node in the tree.
****************************************************************************/
FSTATIC void fqReplaceNode(
	FQNODE *	pNodeToReplace,
	FQNODE *	pReplacementNode
	)
{
	FQNODE_p	pParentNode;
	FLMBOOL	bLinkAsFirst = (pNodeToReplace->pNextSib) ? TRUE : FALSE;

	if (pReplacementNode->pParent)
	{
		fqUnlinkFromParent( pReplacementNode);
	}
	if ((pParentNode = pNodeToReplace->pParent) != NULL)
	{
		fqUnlinkFromParent( pNodeToReplace);
		if (bLinkAsFirst)
		{
			fqLinkFirstChild( pParentNode, pReplacementNode);
		}
		else
		{
			fqLinkLastChild( pParentNode, pReplacementNode);
		}
	}
}

/***************************************************************************
Desc:	Allocate a value node
***************************************************************************/
RCODE F_Query::allocValueNode(
	FLMUINT		uiValLen,
	eValTypes	eValType,
	FQNODE **	ppQNode
	)
{
	RCODE			rc = NE_XFLM_OK;
	FQNODE *		pQNode;

	// If an error has already occurred, cannot add more to query.

	if (RC_BAD( rc = m_rc))
	{
		goto Exit;
	}

	if (!m_pCurExprState)
	{
		if (RC_BAD( rc = allocExprState()))
		{
			goto Exit;
		}
	}

	if (m_pCurExprState->bExpectingLParen)
	{
		rc = RC_SET( NE_XFLM_Q_EXPECTING_LPAREN);
		goto Exit;
	}

	if (!expectingOperand())
	{
		rc = RC_SET( NE_XFLM_Q_UNEXPECTED_VALUE);
		goto Exit;
	}

	if (RC_BAD( rc = m_pool.poolCalloc( sizeof( FQNODE),
									(void **)ppQNode)))
	{
		goto Exit;
	}
	pQNode = *ppQNode;
	pQNode->eNodeType = FLM_VALUE_NODE;
	pQNode->currVal.eValType = eValType;
	pQNode->currVal.uiDataLen = uiValLen;
	pQNode->currVal.uiFlags = VAL_IS_CONSTANT;

	// For string and binary data, allocate a buffer.

	if (uiValLen &&
		 (eValType == XFLM_UTF8_VAL || eValType == XFLM_BINARY_VAL))
	{
		if (RC_BAD( rc = m_pool.poolAlloc( uiValLen,
												(void **)&pQNode->currVal.val.pucBuf)))
		{
			goto Exit;
		}
	}

	if (m_pCurExprState->pCurOperatorNode)
	{
		fqLinkLastChild( m_pCurExprState->pCurOperatorNode, pQNode);
	}
	else
	{
		flmAssert( !m_pCurExprState->pExpr);
		m_pCurExprState->pExpr = pQNode;
	}
	m_pCurExprState->bExpectingOperator = TRUE;
	m_pCurExprState->pLastNode = pQNode;

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Adds a unicode value to the query criteria.
***************************************************************************/
RCODE XFLAPI F_Query::addUnicodeValue(
	const FLMUNICODE *	puzVal)
{
	RCODE			rc = NE_XFLM_OK;
	FQNODE *		pQNode;
	FLMUINT		uiValLen;
	FLMUINT		uiCharCount;
	FLMUINT		uiSenLen;

	// If an error has already occurred, cannot add more to query.

	if (RC_BAD( rc = m_rc))
	{
		goto Exit;
	}

	if (RC_BAD( rc = flmUnicode2Storage( puzVal, 0, NULL,
								&uiValLen, &uiCharCount)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = allocValueNode( uiValLen, XFLM_UTF8_VAL, &pQNode)))
	{
		goto Exit;
	}

	if (uiValLen)
	{
		FLMBOOL					bHaveWildCards;
		const FLMUNICODE *	puzTmp;

		// See if there are wildcards

		puzTmp = puzVal;
		bHaveWildCards = FALSE;
		while (*puzTmp)
		{
			if (*puzTmp == ASCII_BACKSLASH)
			{

				// Skip over the next character no matter what
				// because it is escaped.

				puzTmp++;
				if (*puzTmp == 0)
				{
					break;
				}
			}
			else if (*puzTmp == ASCII_WILDCARD)
			{
				bHaveWildCards = TRUE;
				break;
			}
			puzTmp++;
		}

		if (RC_BAD( rc = flmUnicode2Storage( puzVal, uiCharCount,
			pQNode->currVal.val.pucBuf,
			&pQNode->currVal.uiDataLen, &uiCharCount)))
		{
			goto Exit;
		}

		// Skip past the SEN

		if (RC_BAD( rc = flmGetCharCountFromStorageBuf(
									(const FLMBYTE **)&pQNode->currVal.val.pucBuf,
									pQNode->currVal.uiDataLen, NULL, &uiSenLen)))
		{
			goto Exit;
		}

		pQNode->currVal.uiDataLen -= uiSenLen;
		if (bHaveWildCards)
		{
			pQNode->currVal.uiFlags |= VAL_HAS_WILDCARDS;
		}
	}

Exit:

	m_rc = rc;

	return( rc);
}

/***************************************************************************
Desc:	Adds a UTF8 value to the query criteria.
***************************************************************************/
RCODE XFLAPI F_Query::addUTF8Value(
	const char *	pszVal,
	FLMUINT			uiUTF8Len)
{
	RCODE			rc = NE_XFLM_OK;
	FQNODE *		pQNode;
	FLMUINT		uiValLen;
	FLMUINT		uiSenLen;
	FLMBYTE *	pucEnd = NULL;

	// If an error has already occurred, cannot add more to query.

	if (RC_BAD( rc = m_rc))
	{
		goto Exit;
	}

	if (RC_BAD( rc = flmUTF8ToStorage( 
		(FLMBYTE *)pszVal, uiUTF8Len, NULL, &uiValLen)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = allocValueNode( uiValLen, XFLM_UTF8_VAL, &pQNode)))
	{
		goto Exit;
	}

	if (uiValLen)
	{
		FLMBOOL				bHaveWildCards;
		const FLMBYTE *	pszTmp;
		FLMUNICODE			uzChar;

		// See if there are wildcards

		pszTmp = (FLMBYTE *)pszVal;
		if (uiUTF8Len)
		{
			pucEnd = (FLMBYTE *)pszVal + uiUTF8Len;
		}
		bHaveWildCards = FALSE;
		for (;;)
		{
			if (RC_BAD( rc = f_getCharFromUTF8Buf( &pszTmp, pucEnd, &uzChar)))
			{
				goto Exit;
			}
			
			if (uzChar == ASCII_BACKSLASH)
			{

				// Skip over the next character no matter what
				// because it is escaped.

				if (RC_BAD( rc = f_getCharFromUTF8Buf( &pszTmp, pucEnd, &uzChar)))
				{
					goto Exit;
				}
				if (!uzChar)
				{
					break;
				}
			}
			else if (uzChar == ASCII_WILDCARD)
			{
				bHaveWildCards = TRUE;
				break;
			}

			if (!uzChar)
			{
				break;
			}
		}

		if (RC_BAD( rc = flmUTF8ToStorage( (FLMBYTE *)pszVal,
			uiUTF8Len, (FLMBYTE *)pQNode->currVal.val.pucBuf,
			&pQNode->currVal.uiDataLen)))
		{
			goto Exit;
		}

		// Skip past the SEN

		if (RC_BAD( rc = flmGetCharCountFromStorageBuf(
									(const FLMBYTE **)&pQNode->currVal.val.pucBuf,
									pQNode->currVal.uiDataLen, NULL, &uiSenLen)))
		{
			goto Exit;
		}

		pQNode->currVal.uiDataLen -= uiSenLen;
		if (bHaveWildCards)
		{
			pQNode->currVal.uiFlags |= VAL_HAS_WILDCARDS;
		}
	}

Exit:

	m_rc = rc;

	return( rc);
}

/***************************************************************************
Desc:	Adds a binary value to the query criteria.
***************************************************************************/
RCODE XFLAPI F_Query::addBinaryValue(
	const void *		pvVal,
	FLMUINT				uiValLen)
{
	RCODE			rc = NE_XFLM_OK;
	FQNODE *		pQNode;

	if (RC_BAD( rc = allocValueNode( uiValLen, XFLM_BINARY_VAL, &pQNode)))
	{
		goto Exit;
	}

	if (uiValLen)
	{
		f_memcpy( pQNode->currVal.val.pucBuf, pvVal, uiValLen);
	}

Exit:

	m_rc = rc;

	return( rc);
}

/***************************************************************************
Desc:	Adds a UINT value to the query criteria.
***************************************************************************/
RCODE XFLAPI F_Query::addUINTValue(
	FLMUINT	uiVal
	)
{
	RCODE			rc = NE_XFLM_OK;
	FQNODE *		pQNode;

	if (RC_BAD( rc = allocValueNode( 0, XFLM_UINT_VAL, &pQNode)))
	{
		goto Exit;
	}
	pQNode->currVal.val.uiVal = uiVal;

Exit:

	m_rc = rc;

	return( rc);
}

/***************************************************************************
Desc:	Adds an INT value to the query criteria.
***************************************************************************/
RCODE XFLAPI F_Query::addINTValue(
	FLMINT	iVal
	)
{
	RCODE			rc = NE_XFLM_OK;
	FQNODE *		pQNode;

	if (RC_BAD( rc = allocValueNode( 0, XFLM_INT_VAL, &pQNode)))
	{
		goto Exit;
	}
	pQNode->currVal.val.iVal = iVal;

Exit:

	m_rc = rc;

	return( rc);
}

/***************************************************************************
Desc:	Adds a UINT64 value to the query criteria.
***************************************************************************/
RCODE XFLAPI F_Query::addUINT64Value(
	FLMUINT64	ui64Val
	)
{
	RCODE			rc = NE_XFLM_OK;
	FQNODE *		pQNode;

	if (RC_BAD( rc = allocValueNode( 0, XFLM_UINT64_VAL, &pQNode)))
	{
		goto Exit;
	}
	pQNode->currVal.val.ui64Val = ui64Val;

Exit:

	m_rc = rc;

	return( rc);
}

/***************************************************************************
Desc:	Adds an INT64 value to the query criteria.
***************************************************************************/
RCODE XFLAPI F_Query::addINT64Value(
	FLMINT64	i64Val
	)
{
	RCODE			rc = NE_XFLM_OK;
	FQNODE *		pQNode;

	if (RC_BAD( rc = allocValueNode( 0, XFLM_INT64_VAL, &pQNode)))
	{
		goto Exit;
	}
	pQNode->currVal.val.i64Val = i64Val;

Exit:

	m_rc = rc;

	return( rc);
}

/***************************************************************************
Desc:	Adds a BOOL value to the query criteria.
***************************************************************************/
RCODE XFLAPI F_Query::addBoolean(
	FLMBOOL	bVal,
	FLMBOOL	bUnknown
	)
{
	RCODE			rc = NE_XFLM_OK;
	FQNODE *		pQNode;

	if (RC_BAD( rc = allocValueNode( 0, XFLM_BOOL_VAL, &pQNode)))
	{
		goto Exit;
	}
	pQNode->currVal.val.eBool = (XFlmBoolType)(bUnknown
												 ? XFLM_UNKNOWN
												 : (XFlmBoolType)(bVal
																 ? XFLM_TRUE
																 : XFLM_FALSE));

Exit:

	m_rc = rc;

	return( rc);
}

/***************************************************************************
Desc:	Add an XPATH component
***************************************************************************/
RCODE XFLAPI F_Query::addXPathComponent(
	eXPathAxisTypes		eXPathAxis,
	eDomNodeType			eNodeType,
	FLMUINT					uiDictNum,
	IF_QueryNodeSource *	pNodeSource
	)
{
	RCODE					rc = NE_XFLM_OK;
	XPATH_COMPONENT *	pXPathComponent;
	FXPATH *				pXPath;
	FQNODE *				pQNode;

	// If an error has already occurred, cannot add more to query.

	if (RC_BAD( rc = m_rc))
	{
		goto Exit;
	}

	if (!m_pCurExprState)
	{
		if (RC_BAD( rc = allocExprState()))
		{
			goto Exit;
		}
	}

	// Must be expecting an operand, or the last node
	// must be an XPATH

	if (!expectingOperand() &&
		 m_pCurExprState->pLastNode->eNodeType != FLM_XPATH_NODE)
	{
		rc = RC_SET( NE_XFLM_Q_UNEXPECTED_XPATH_COMPONENT);
		goto Exit;
	}

	// If axis is META_AXIS, verify that the specific type of meta data
	// requested is valid.

	if (eXPathAxis == META_AXIS)
	{
		switch (uiDictNum)
		{
			case XFLM_META_NODE_ID:
			case XFLM_META_DOCUMENT_ID:
			case XFLM_META_PARENT_ID:
			case XFLM_META_FIRST_CHILD_ID:
			case XFLM_META_LAST_CHILD_ID:
			case XFLM_META_NEXT_SIBLING_ID:
			case XFLM_META_PREV_SIBLING_ID:
			case XFLM_META_VALUE:
				break;
			default:
				rc = RC_SET( NE_XFLM_Q_INVALID_META_DATA_TYPE);
				goto Exit;
		}
	}

	// Allocate an XPATH component

	if (RC_BAD( rc = m_pool.poolCalloc( sizeof( XPATH_COMPONENT),
										(void **)&pXPathComponent)))
	{
		goto Exit;
	}
	pXPathComponent->eNodeType = eNodeType;
	pXPathComponent->eXPathAxis = eXPathAxis;
	pXPathComponent->uiDictNum = uiDictNum;
	pXPathComponent->pNodeSource = pNodeSource;
	if (m_pCurExprState->pPrev &&
		 m_pCurExprState->pXPathComponent)
	{

		// pXPathContext is the XPATH component context for this
		// XPATH component.  This may be used in optimization.

		pXPathComponent->pXPathContext = m_pCurExprState->pXPathComponent;
	}

	// If we are not expecting an operand, then the last component
	// has to be an XPATH node.

	if (!expectingOperand())
	{
		pXPath = m_pCurExprState->pLastNode->nd.pXPath;
	}
	else
	{

		// Need to allocate a node and an XPATH

		if (RC_BAD( rc = m_pool.poolCalloc( sizeof( FQNODE),
											(void **)&pQNode)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = m_pool.poolCalloc( sizeof( FXPATH),
											(void **)&pXPath)))
		{
			goto Exit;
		}
		pQNode->eNodeType = FLM_XPATH_NODE;
		pQNode->nd.pXPath = pXPath;

		// Link this node into the expression

		if (m_pCurExprState->pCurOperatorNode)
		{
			fqLinkLastChild( m_pCurExprState->pCurOperatorNode, pQNode);
		}
		else
		{
			flmAssert( !m_pCurExprState->pExpr);
			m_pCurExprState->pExpr = pQNode;
		}
		m_pCurExprState->bExpectingOperator = TRUE;
		m_pCurExprState->pLastNode = pQNode;
	}
	pXPathComponent->pXPathNode = m_pCurExprState->pLastNode;

	// Link the new XPATH component into the XPATH as the last
	// component.

	if ((pXPathComponent->pPrev = pXPath->pLastComponent) != NULL)
	{
		pXPath->pLastComponent->pNext = pXPathComponent;
	}
	else
	{
		pXPath->pFirstComponent = pXPathComponent;
	}
	pXPath->pLastComponent = pXPathComponent;
	
	if (pNodeSource)
	{
		if (RC_BAD( rc = objectAddRef( pNodeSource)))
		{
			goto Exit;
		}
	}

Exit:

	m_rc = rc;

	return( rc);
}

/***************************************************************************
Desc:	Keep track of objects supplied by the application that we use
		for callbacks, etc.
***************************************************************************/
RCODE F_Query::objectAddRef(
	F_Object *		pObject)
{
	RCODE				rc = NE_XFLM_OK;

	// If object list is full, make room for 20 more

	if (m_uiObjectCount == m_uiObjectListSize)
	{
		if (RC_BAD( rc = f_realloc( sizeof( F_Object *) *
											(m_uiObjectListSize + 20),
											&m_ppObjectList)))
		{
			goto Exit;
		}
		m_uiObjectListSize += 20;
	}
	
	m_ppObjectList [m_uiObjectCount++] = pObject;
	pObject->AddRef();

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get a context position from a value.  It must be a positive
		integer.  Anything else will cause an error to be returned.
***************************************************************************/
FSTATIC RCODE fqGetPosition(
	FQVALUE *	pQValue,
	FLMUINT *	puiPos
	)
{
	RCODE	rc = NE_XFLM_OK;

	switch (pQValue->eValType)
	{
		case XFLM_UINT_VAL:
			if (!pQValue->val.uiVal)
			{
				rc = RC_SET( NE_XFLM_Q_INVALID_CONTEXT_POS);
				goto Exit;
			}
			*puiPos = pQValue->val.uiVal;
			break;
		case XFLM_UINT64_VAL:
			if (!pQValue->val.ui64Val ||
				 pQValue->val.ui64Val > (FLMUINT64)(~((FLMUINT)0)))
			{
				rc = RC_SET( NE_XFLM_Q_INVALID_CONTEXT_POS);
				goto Exit;
			}
			*puiPos = (FLMUINT)pQValue->val.ui64Val;
			break;
		case XFLM_INT_VAL:
			if (pQValue->val.iVal <= 0)
			{
				rc = RC_SET( NE_XFLM_Q_INVALID_CONTEXT_POS);
				goto Exit;
			}
			*puiPos = (FLMUINT)pQValue->val.iVal;
			break;
		case XFLM_INT64_VAL:
			if (pQValue->val.i64Val <= 0 ||
				 pQValue->val.i64Val > (FLMINT64)(~((FLMUINT)0)))
			{
				rc = RC_SET( NE_XFLM_Q_INVALID_CONTEXT_POS);
				goto Exit;
			}
			*puiPos = (FLMUINT)pQValue->val.i64Val;
			break;
		default:
			rc = RC_SET( NE_XFLM_Q_INVALID_CONTEXT_POS);
			goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Determine if an XPATH component has a test for context position.
***************************************************************************/
FINLINE FLMBOOL hasContextPosTest(
	XPATH_COMPONENT *	pXPathComponent
	)
{
	return( pXPathComponent->pContextPosExpr ||
			  pXPathComponent->uiContextPosNeeded
			  ? TRUE
			  : FALSE);
}

/***************************************************************************
Desc:	Add an operator to the query expression
***************************************************************************/
RCODE XFLAPI F_Query::addOperator(
	eQueryOperators		eOperator,
	FLMUINT					uiCompareRules,
	IF_OperandComparer *	pOpComparer)
{
	RCODE				rc = NE_XFLM_OK;
	FQNODE *			pQNode;
	FQEXPR *			pQExpr;
	FQNODE *			pParentNode;

	// If an error has already occurred, cannot add more to query.

	if (RC_BAD( rc = m_rc))
	{
		goto Exit;
	}

	if (!m_pCurExprState)
	{
		if (RC_BAD( rc = allocExprState()))
		{
			goto Exit;
		}
	}

	// If we are expecting a left paren (for a function), that is
	// the only thing that is acceptable at this point.

	if (m_pCurExprState->bExpectingLParen && eOperator != XFLM_LPAREN_OP)
	{
		rc = RC_SET( NE_XFLM_Q_EXPECTING_LPAREN);
		goto Exit;
	}

	switch (eOperator)
	{
		case XFLM_LPAREN_OP:

			// If the operator is a left paren, increment the nesting level

			if (expectingOperator())
			{
				rc = RC_SET( NE_XFLM_Q_UNEXPECTED_LPAREN);
				goto Exit;
			}
			m_pCurExprState->uiNestLevel++;
			m_pCurExprState->bExpectingLParen = FALSE;
			goto Exit;

		case XFLM_RPAREN_OP:
			if (expectingOperand())
			{
				rc = RC_SET( NE_XFLM_Q_UNEXPECTED_RPAREN);
				goto Exit;
			}
			if (!m_pCurExprState->uiNestLevel)
			{
				rc = RC_SET( NE_XFLM_Q_UNMATCHED_RPAREN);
				goto Exit;
			}
			m_pCurExprState->uiNestLevel--;

			// See if this is the right paren to a function call

			if (!m_pCurExprState->uiNestLevel && parsingFunction())
			{

				// If we have a valid expression, link it to the
				// function node as its last parameter.

				if (m_pCurExprState->pExpr)
				{
					m_pCurExprState->uiNumExpressions++;
					
					// uiNumExprNeeded might be zero.
					
					if (m_pCurExprState->uiNumExpressions >
						 m_pCurExprState->uiNumExprNeeded)
					{
						rc = RC_SET( NE_XFLM_Q_INVALID_NUM_FUNC_ARGS);
						goto Exit;
					}

					// Allocate an expression node and link it to the
					// function.

					if (RC_BAD( rc = m_pool.poolCalloc( sizeof( FQEXPR),
												(void **)&pQExpr)))
					{
						goto Exit;
					}
					if ((pQExpr->pPrev =
							m_pCurExprState->pQFunction->pLastArg) != NULL)
					{
						pQExpr->pPrev->pNext = pQExpr;
					}
					else
					{
						m_pCurExprState->pQFunction->pFirstArg = pQExpr;
					}
					m_pCurExprState->pQFunction->pLastArg = pQExpr;
					pQExpr->pExpr = m_pCurExprState->pExpr;
					if (RC_BAD( rc = getPredicates( &pQExpr->pExpr, NULL,
													m_pCurExprState->pXPathComponent)))
					{
						goto Exit;
					}
					
					// If this is a user-defined function, make sure the
					// expression parameter (there should only be one) is
					// an XPATH expression, and that the getPredicates call
					// didn't eliminate the expression.
					
					if (m_pCurExprState->pQFunction->pFuncObj)
					{
						if (!pQExpr->pExpr ||
							 pQExpr->pExpr->eNodeType != FLM_XPATH_NODE)
						{
							rc = RC_SET( NE_XFLM_Q_INVALID_FUNC_ARG);
							goto Exit;
						}
					}
				}

				// See if we got the required number of arguments for
				// the function

				if (m_pCurExprState->uiNumExpressions <
					 m_pCurExprState->uiNumExprNeeded)
				{
					rc = RC_SET( NE_XFLM_Q_INVALID_NUM_FUNC_ARGS);
					goto Exit;
				}
				
				// Return to the former context.

				m_pCurExprState = m_pCurExprState->pPrev;
			}
			goto Exit;

		case XFLM_NEG_OP:
		case XFLM_NOT_OP:
			if (expectingOperator())
			{
				rc = RC_SET( NE_XFLM_Q_EXPECTING_OPERATOR);
				goto Exit;
			}
			break;

		case XFLM_COMMA_OP:

			// In order for a comma to be legal, the following conditions
			// must be met:
			// 1) Must be inside a function
			// 2) Must need at least two arguments for the function
			// 3) Must not already have enough arguments
			// 4) Must be at nesting level 1
			// 5) Must be expecting an operator
			// 6) Must have a non-empty expression we can link to
			//		function node.

			if (!parsingFunction() ||
				 m_pCurExprState->uiNumExprNeeded < 2 ||
				 m_pCurExprState->uiNumExpressions <
					m_pCurExprState->uiNumExprNeeded - 1 ||
				 m_pCurExprState->uiNestLevel > 1 ||
				 expectingOperand() ||
				 !m_pCurExprState->pExpr)
			{
				rc = RC_SET( NE_XFLM_Q_UNEXPECTED_COMMA);
				goto Exit;
			}
			m_pCurExprState->uiNumExpressions++;

			// Allocate an expression node and link it to the

			if (RC_BAD( rc = m_pool.poolCalloc( sizeof( FQEXPR),
										(void **)&pQExpr)))
			{
				goto Exit;
			}
			if ((pQExpr->pPrev =
					m_pCurExprState->pQFunction->pLastArg) != NULL)
			{
				pQExpr->pPrev->pNext = pQExpr;
			}
			else
			{
				m_pCurExprState->pQFunction->pFirstArg = pQExpr;
			}
			m_pCurExprState->pQFunction->pLastArg = pQExpr;
			if (RC_BAD( rc = getPredicates( &pQExpr->pExpr, NULL,
											m_pCurExprState->pXPathComponent)))
			{
				goto Exit;
			}

			// Reset the expression

			m_pCurExprState->pExpr = NULL;
			m_pCurExprState->pCurOperatorNode = NULL;

			// The following conditions better already be set

			flmAssert( expectingOperand());
			flmAssert( !m_pCurExprState->bExpectingLParen);
			flmAssert( m_pCurExprState->uiNestLevel == 1);
			goto Exit;

		case XFLM_LBRACKET_OP:

			// Last node has to be an XPATH node

			if (m_pCurExprState->pLastNode->eNodeType != FLM_XPATH_NODE)
			{
				rc = RC_SET( NE_XFLM_Q_ILLEGAL_LBRACKET);
				goto Exit;
			}

			// Save the last node into pQNode, because when we call allocExprState
			// we will no longer be able to get at it.

			pQNode = m_pCurExprState->pLastNode;

			// Cannot add expressions if the last component already has
			// an expression to test context position.

			if (hasContextPosTest( pQNode->nd.pXPath->pLastComponent))
			{
				rc = RC_SET( NE_XFLM_Q_NEW_EXPR_NOT_ALLOWED);
				goto Exit;
			}

			// Always allocate a new expression state for an xpath expression

			if (RC_BAD( rc = allocExprState()))
			{
				goto Exit;
			}

			m_pCurExprState->pXPathComponent = pQNode->nd.pXPath->pLastComponent;

			goto Exit;

		case XFLM_RBRACKET_OP:

			// Right bracket is only allowed if we are parsing an
			// xpath expression and we are at nesting level zero and
			// we are not expecting an operand

			if (!parsingXPathExpr() ||
				 m_pCurExprState->uiNestLevel ||
				 (expectingOperand() && m_pCurExprState->pExpr))
			{
				rc = RC_SET( NE_XFLM_Q_ILLEGAL_RBRACKET);
				goto Exit;
			}

			// If we have a non-empty expression, link it to the
			// list of xpath component expressions off of the xpath
			// component.

			if (m_pCurExprState->pExpr)
			{

				// If the XPATH component does not have any expression yet,
				// make this expression its expression.  Otherwise, AND
				// this expression to the existing expression.

				if (m_pCurExprState->pExpr->eNodeType == FLM_VALUE_NODE)
				{
					if (RC_BAD( rc = fqGetPosition( &m_pCurExprState->pExpr->currVal,
							&m_pCurExprState->pXPathComponent->uiContextPosNeeded)))
					{
						goto Exit;
					}
				}
				else if (m_pCurExprState->pExpr->eNodeType == FLM_OPERATOR_NODE &&
					isArithOp( m_pCurExprState->pExpr->nd.op.eOperator))
				{
					m_pCurExprState->pXPathComponent->pContextPosExpr =
						m_pCurExprState->pExpr;
					if (RC_BAD( rc = getPredicates(
									&m_pCurExprState->pXPathComponent->pContextPosExpr,
									NULL,
									m_pCurExprState->pXPathComponent)))
					{
						goto Exit;
					}

					// If, after the optimization, we are left with a constant,
					// NULL out pContextPosExpr and put it into uiContextPosNeeded.

					if (m_pCurExprState->pXPathComponent->pContextPosExpr->eNodeType ==
										FLM_VALUE_NODE)
					{
						if (RC_BAD( rc = fqGetPosition(
								&m_pCurExprState->pXPathComponent->pContextPosExpr->currVal,
								&m_pCurExprState->pXPathComponent->uiContextPosNeeded)))
						{
							goto Exit;
						}
						m_pCurExprState->pXPathComponent->pContextPosExpr = NULL;
					}
				}
				else if (!m_pCurExprState->pXPathComponent->pExpr)
				{
					m_pCurExprState->pXPathComponent->pExpr =
						m_pCurExprState->pExpr;
					if (RC_BAD( rc = getPredicates(
									&m_pCurExprState->pXPathComponent->pExpr,
									NULL,
									m_pCurExprState->pXPathComponent)))
					{
						goto Exit;
					}
				}
				else
				{

					// Create an AND node and link the existing expression with
					// this new expression as children of this new AND node.

					if (RC_BAD( rc = m_pool.poolCalloc( sizeof( FQNODE),
															(void **)&pQNode)))
					{
						goto Exit;
					}
					pQNode->eNodeType = FLM_OPERATOR_NODE;
					pQNode->nd.op.eOperator = XFLM_AND_OP;

					fqLinkFirstChild( pQNode,
							m_pCurExprState->pXPathComponent->pExpr);
					fqLinkLastChild( pQNode, m_pCurExprState->pExpr);
					m_pCurExprState->pXPathComponent->pExpr = pQNode;

					// Set up a context node for the new AND node.
					// If the left operand's context (which would have been
					// set up previously by a call to getPredicates) is
					// an intersect context, we can point this node
					// right at it, and make the context's root node this
					// new node.  Otherwise, we have to create a new context
					// and link the left operand's context to it.

					if (pQNode->pFirstChild->pContext->bIntersect)
					{
						pQNode->pContext = pQNode->pFirstChild->pContext;
						pQNode->pContext->pQRootNode = pQNode;
					}
					else
					{
						if (RC_BAD( rc = createOpContext( NULL, TRUE, pQNode)))
						{
							goto Exit;
						}

						// Put first child's context as child of this node's
						// context.

						pQNode->pContext->pFirstChild = pQNode->pFirstChild->pContext;
						pQNode->pContext->pLastChild = pQNode->pFirstChild->pContext;
						pQNode->pFirstChild->pContext->pParent = pQNode->pContext;
					}

					// Get the predicates of ONLY the right-hand side of the
					// tree - because we haven't gotten its predicates yet, but
					// the left hand side has already been done.

					if (RC_BAD( rc = getPredicates(
													&m_pCurExprState->pXPathComponent->pExpr,
													pQNode->pLastChild,
													m_pCurExprState->pXPathComponent)))
					{
						goto Exit;
					}
				}
			}

			// Return to the former context.

			m_pCurExprState = m_pCurExprState->pPrev;

			goto Exit;

		default:

			if (expectingOperand())
			{
				rc = RC_SET( NE_XFLM_Q_EXPECTING_OPERAND);
				goto Exit;
			}
			if (!isLegalOperator( eOperator))
			{
				rc = RC_SET( NE_XFLM_Q_ILLEGAL_OPERATOR);
				goto Exit;
			}

			break;
	}

	// Cannot set both XFLM_COMP_COMPRESS_WHITESPACE and XFLM_COMP_NO_WHITESPACE
	// in comparison rules.  Also, cannot set XFLM_COMP_IGNORE_LEADING_SPACE or
	// XFLM_COMP_IGNORE_TRAILING_SPACE with XFLM_COMP_NO_WHITESPACE.

	if ((uiCompareRules & XFLM_COMP_NO_WHITESPACE) &&
		 (uiCompareRules & (XFLM_COMP_COMPRESS_WHITESPACE |
								  XFLM_COMP_IGNORE_LEADING_SPACE |
								  XFLM_COMP_IGNORE_TRAILING_SPACE)))
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_Q_ILLEGAL_COMPARE_RULES);
		goto Exit;
	}

	// Make a QNODE and find a place for it in the query tree

	if (RC_BAD( rc = m_pool.poolCalloc( sizeof( FQNODE),
											(void **)&pQNode)))
	{
		goto Exit;
	}
	pQNode->eNodeType = FLM_OPERATOR_NODE;
	pQNode->nd.op.eOperator = eOperator;
	pQNode->nd.op.uiCompareRules = uiCompareRules;
	pQNode->nd.op.pOpComparer = pOpComparer;
	pQNode->uiNestLevel = m_pCurExprState->uiNestLevel;

	// If this is the first operator in the query, set the current operator
	// to it and graft in the current operand as its child.

	if (!m_pCurExprState->pExpr)
	{
		m_pCurExprState->pExpr = pQNode;
		m_pCurExprState->pCurOperatorNode = pQNode;
		goto Exit;
	}

	// Go up the stack until an operator whose nest level or precedence is <
	// this one's is encountered, then link this one in as the last child

	pParentNode = m_pCurExprState->pCurOperatorNode;
	while (pParentNode &&
			 (pParentNode->uiNestLevel > pQNode->uiNestLevel ||
			  (pParentNode->uiNestLevel == pQNode->uiNestLevel &&
			   getPrecedence( pParentNode->nd.op.eOperator) >=
				getPrecedence( eOperator))))
	{
		pParentNode = pParentNode->pParent;
	}
	if (!pParentNode)
	{
		if (m_pCurExprState->pExpr)
		{
			fqLinkLastChild( pQNode, m_pCurExprState->pExpr);
		}
		m_pCurExprState->pExpr = pQNode;
	}
	else if (eOperator == XFLM_NOT_OP || eOperator == XFLM_NEG_OP)
	{

		// Need to treat NOT and NEG as if they were operands.

		// Parent better be an operator.

		flmAssert( pParentNode->eNodeType == FLM_OPERATOR_NODE);

#ifdef FLM_DEBUG
		if (pParentNode->nd.op.eOperator == XFLM_NEG_OP ||
			 pParentNode->nd.op.eOperator == XFLM_NOT_OP)
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

		fqLinkLastChild( pParentNode, pQNode);
		flmAssert( !m_pCurExprState->bExpectingOperator);
	}
	else
	{

		// Parent better be an operator.

		flmAssert( pParentNode->eNodeType == FLM_OPERATOR_NODE);

		// Unlink last child of parent node and replace with this
		// new node.  The parent node better already have the correct
		// number of children, or we are not parsing correctly.

		flmAssert( pParentNode->pFirstChild);
		if (pParentNode->nd.op.eOperator == XFLM_NEG_OP ||
			 pParentNode->nd.op.eOperator == XFLM_NOT_OP)
		{

			// Better only be one child.

			flmAssert( !pParentNode->pFirstChild->pNextSib);

			fqLinkLastChild( pQNode, pParentNode->pFirstChild);
		}
		else
		{

			// Better only be two child nodes

			flmAssert( pParentNode->pFirstChild->pNextSib ==
						  pParentNode->pLastChild);
			fqLinkLastChild( pQNode, pParentNode->pLastChild);
		}
		fqLinkLastChild( pParentNode, pQNode);
	}

	m_pCurExprState->pCurOperatorNode = pQNode;
	m_pCurExprState->bExpectingOperator = FALSE;
	m_pCurExprState->pLastNode = pQNode;

	if (pOpComparer)
	{
		if (RC_BAD( rc = objectAddRef( pOpComparer)))
		{
			goto Exit;
		}
	}

Exit:

	m_rc = rc;

	return( rc);
}

/***************************************************************************
Desc:	Add a function to the query expression
***************************************************************************/
RCODE XFLAPI F_Query::addFunction(
	eQueryFunctions	eFunction,
	IF_QueryValFunc *	pFuncObj,
	FLMBOOL				bHaveXPathExpr)
{
	RCODE					rc = NE_XFLM_OK;
	FQNODE *				pQNode;
	FQFUNCTION *		pQFunction;
	FQNODE *				pParentNode;
	XPATH_COMPONENT *	pSaveXPathComponent;

	// If an error has already occurred, cannot add more to query.

	if (RC_BAD( rc = m_rc))
	{
		goto Exit;
	}

	if (!m_pCurExprState)
	{
		if (RC_BAD( rc = allocExprState()))
		{
			goto Exit;
		}
	}

	// Must be expecting an operand

	if (expectingOperator())
	{
		rc = RC_SET( NE_XFLM_Q_EXPECTING_OPERATOR);
		goto Exit;
	}

	// Allocate a function node

	if (RC_BAD( rc = m_pool.poolCalloc( sizeof( FQNODE),
											(void **)&pQNode)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = m_pool.poolCalloc( sizeof( FQFUNCTION),
											(void **)&pQFunction)))
	{
		goto Exit;
	}
	pQNode->nd.pQFunction = pQFunction;
	pQNode->eNodeType = FLM_FUNCTION_NODE;
	pQFunction->eFunction = eFunction;
	pQFunction->pFuncObj = pFuncObj;
				  
	// See if this is the first node in the expression.

	if (!m_pCurExprState->pExpr)
	{
		m_pCurExprState->pExpr = pQNode;
	}
	else
	{
		pParentNode = m_pCurExprState->pCurOperatorNode;
		flmAssert( pParentNode);

		// Parent better be an operator, and better have room for
		// function to be linked as one of its operands.

		flmAssert( pParentNode->eNodeType == FLM_OPERATOR_NODE);

		if (pParentNode->nd.op.eOperator == XFLM_NEG_OP ||
			 pParentNode->nd.op.eOperator == XFLM_NOT_OP)
		{

			// Better not have any children yet.

			flmAssert( !pParentNode->pFirstChild);
		}
		else
		{

			// Better only have one child node.

			flmAssert( pParentNode->pFirstChild == pParentNode->pLastChild);
		}
		fqLinkLastChild( pParentNode, pQNode);
	}
	m_pCurExprState->pLastNode = pQNode;
	pSaveXPathComponent = m_pCurExprState->pXPathComponent;

	// Always allocate a new expression state for a function

	if (RC_BAD( rc = allocExprState()))
	{
		goto Exit;
	}

	// First thing we expect in this state is a left paren

	m_pCurExprState->bExpectingLParen = TRUE;
	m_pCurExprState->pQFunction = pQFunction;
	m_pCurExprState->pXPathComponent = pSaveXPathComponent;
	if (pFuncObj)
	{
		if (RC_BAD( rc = objectAddRef( pFuncObj)))
		{
			goto Exit;
		}
		
		// In this case, the expression must return a node.
		
		m_pCurExprState->uiNumExprNeeded = bHaveXPathExpr ? (FLMUINT)1 : (FLMUINT)0;
	}
	else
	{

//visit
//	m_pCurExprState->uiNumExprNeeded = ??? - number specified by the function.
	}
	
	// Parent state needs to be expecting an operator when we come out from
	// parsing the function.
	
	m_pCurExprState->pPrev->bExpectingOperator = TRUE;

Exit:

	m_rc = rc;

	return( rc);
}

/***************************************************************************
Desc:	Compare two values
***************************************************************************/
FSTATIC RCODE fqCompareValues(
	FQVALUE *			pValue1,
	FLMBOOL				bInclusive1,
	FLMBOOL				bNullIsLow1,
	FQVALUE *			pValue2,
	FLMBOOL				bInclusive2,
	FLMBOOL				bNullIsLow2,
	FLMUINT				uiCompareRules,
	FLMUINT				uiLanguage,
	FLMINT *				piCmp)
{
	RCODE	rc = NE_XFLM_OK;

	// We have already called fqCanCompare, so no need to do it here

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

	if (RC_BAD( rc = fqCompare( pValue1, pValue2, 
		uiCompareRules, NULL, uiLanguage, piCmp)))
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

/***************************************************************************
Desc:	Intersect a predicate into a context path.
***************************************************************************/
RCODE F_Query::intersectPredicates(
	CONTEXT_PATH *			pContextPath,
	FQNODE *					pXPathNode,
	eQueryOperators		eOperator,
	FLMUINT					uiCompareRules,
	IF_OperandComparer *	pOpComparer,
	FQNODE *					pContextNode,
	FLMBOOL					bNotted,
	FQVALUE *				pQValue,
	FLMBOOL *				pbClipContext
	)
{
	RCODE				rc = NE_XFLM_OK;
	PATH_PRED *		pPred;
	FLMINT			iCmp;
	FLMBOOL			bDoMatch;

	if (!pQValue || pQValue->eValType != XFLM_UTF8_VAL)
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
		bDoMatch = (eOperator == XFLM_EQ_OP &&
						(pQValue->uiFlags & VAL_IS_CONSTANT) &&
						(pQValue->uiFlags & VAL_HAS_WILDCARDS))
						? TRUE
						: FALSE;
	}

	if ((pPred = pContextPath->pFirstPred) != NULL)
	{
		if (eOperator == XFLM_EXISTS_OP)
		{

			// An exists operator will either
			// merge with an existing predicate or
			// cancel the whole thing out as an
			// empty result.

			// If this predicate is not-exists, another
			// predicate ANDed with this one can never
			// return a result that will match, unless
			// that predicate is also a not-exists, in
			// which case, we simply combine this one
			// with that one.

			if (bNotted)
			{
				if (pPred->eOperator != XFLM_EXISTS_OP ||
					 !pPred->bNotted)
				{
					*pbClipContext = TRUE;
				}
			}
			goto Exit;
		}
		else if (pPred->eOperator == XFLM_EXISTS_OP)
		{

			// If the first predicate is an exists operator
			// it will be the only one, because otherwise
			// it will have been merged with another operator
			// in the code just above.

			flmAssert( !pPred->pNext);

			// If the predicate is notted, another predicate
			// ANDed with this one can never return a result.

			if (pPred->bNotted)
			{
				*pbClipContext = TRUE;
			}
			else
			{

				// Change the predicate to the current
				// operator.

				pPred->eOperator = eOperator;
				pPred->pFromValue = pQValue;
				pPred->bNotted = bNotted;
			}
			goto Exit;
		}
		else if ((eOperator == XFLM_EQ_OP && !bDoMatch) ||
					eOperator == XFLM_LE_OP ||
					eOperator == XFLM_LT_OP ||
					eOperator == XFLM_GE_OP ||
					eOperator == XFLM_GT_OP)
		{

			// If there is range operator, there
			// should only be one of them with the
			// same compare rules, because they
			// will all always be merged with these operators
			// or they will cancel to yield an empty result
			// when doing intersections.

			while (pPred)
			{
				if (pPred->eOperator == XFLM_RANGE_OP &&
					 pPred->uiCompareRules == uiCompareRules)
				{
					FQVALUE *	pFromValue;
					FQVALUE *	pUntilValue;
					FLMBOOL		bInclFrom;
					FLMBOOL		bInclUntil;

					pFromValue = (eOperator == XFLM_EQ_OP ||
									  eOperator == XFLM_GE_OP ||
									  eOperator == XFLM_GT_OP)
									  ? pQValue
									  : NULL;
					pUntilValue = (eOperator == XFLM_EQ_OP ||
									   eOperator == XFLM_LE_OP ||
									   eOperator == XFLM_LT_OP)
									   ? pQValue
									   : NULL;
					bInclFrom = (FLMBOOL)(eOperator == XFLM_EQ_OP ||
												 eOperator == XFLM_GE_OP
												 ? TRUE
												 : FALSE);
					bInclUntil = (FLMBOOL)(eOperator == XFLM_EQ_OP ||
												  eOperator == XFLM_LE_OP
												  ? TRUE
												  : FALSE);

					// If the value type is not compatible with the predicate's
					// value type, we cannot do the comparison, and there is
					// no intersection.

					if (!fqCanCompare( pQValue, pPred->pFromValue) ||
						 !fqCanCompare( pQValue, pPred->pUntilValue))
					{
						*pbClipContext = TRUE;
					}
					else if (RC_BAD( rc = fqCompareValues( pFromValue,
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

						if (RC_BAD( rc = fqCompareValues( pFromValue,
									bInclFrom, TRUE,
									pPred->pUntilValue, pPred->bInclUntil, FALSE,
									uiCompareRules, m_uiLanguage, &iCmp)))
						{
							goto Exit;
						}
						if (iCmp > 0)
						{
							*pbClipContext = TRUE;
						}
						else
						{
							pPred->pFromValue = pFromValue;
							pPred->bInclFrom = bInclFrom;
						}
					}
					else if (RC_BAD( rc = fqCompareValues( pUntilValue,
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

						if (RC_BAD( rc = fqCompareValues( pUntilValue,
										bInclUntil, FALSE,
										pPred->pFromValue, pPred->bInclFrom, TRUE,
										uiCompareRules, m_uiLanguage, &iCmp)))
						{
							goto Exit;
						}
						if (iCmp < 0)
						{
							*pbClipContext = TRUE;
						}
						else
						{
							pPred->pUntilValue = pUntilValue;
							pPred->bInclUntil = bInclUntil;
						}
					}
					goto Exit;
				}
				pPred = pPred->pNext;
			}
		}
	}

	// Add a new predicate to the context path

	if (RC_BAD( rc = m_pool.poolCalloc( sizeof( PATH_PRED),
												(void **)&pPred)))
	{
		goto Exit;
	}
	pPred->uiCompareRules = uiCompareRules;
	pPred->pOpComparer = pOpComparer;

	// Link the predicate as the last predicate for the path

	if ((pPred->pPrev = pContextPath->pLastPred) != NULL)
	{
		pPred->pPrev->pNext = pPred;
	}
	else
	{
		pContextPath->pFirstPred = pPred;
	}
	pContextPath->pLastPred = pPred;

	// Set other items in the predicate.

	pPred->pContextNode = pContextNode;
	pPred->bNotted = bNotted;
	switch (eOperator)
	{
		case XFLM_EXISTS_OP:
		case XFLM_NE_OP:
			pPred->eOperator = eOperator;
			pPred->pFromValue = pQValue;
			break;
		case XFLM_APPROX_EQ_OP:
			pPred->eOperator = eOperator;
			pPred->pFromValue = pQValue;
			pPred->bInclFrom = TRUE;
			pPred->bInclUntil = TRUE;
			break;
		case XFLM_EQ_OP:
			if (bDoMatch)
			{
				pPred->eOperator = XFLM_MATCH_OP;
				pPred->pFromValue = pQValue;
			}
			else
			{
				pPred->eOperator = XFLM_RANGE_OP;
				pPred->pFromValue = pQValue;
				pPred->pUntilValue = pQValue;
				pPred->bInclFrom = TRUE;
				pPred->bInclUntil = TRUE;
			}
			break;
		case XFLM_LE_OP:
			pPred->eOperator = XFLM_RANGE_OP;
			pPred->pFromValue = NULL;
			pPred->pUntilValue = pQValue;
			pPred->bInclUntil = TRUE;
			break;
		case XFLM_LT_OP:
			pPred->eOperator = XFLM_RANGE_OP;
			pPred->pFromValue = NULL;
			pPred->pUntilValue = pQValue;
			pPred->bInclUntil = FALSE;
			break;
		case XFLM_GE_OP:
			pPred->eOperator = XFLM_RANGE_OP;
			pPred->pFromValue = pQValue;
			pPred->pUntilValue = NULL;
			pPred->bInclFrom = TRUE;
			break;
		case XFLM_GT_OP:
			pPred->eOperator = XFLM_RANGE_OP;
			pPred->pFromValue = pQValue;
			pPred->pUntilValue = NULL;
			pPred->bInclFrom = FALSE;
			break;
		default:
			rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
			goto Exit;
	}

Exit:

	if (RC_OK( rc) && !(*pbClipContext))
	{
		PATH_PRED_NODE *	pPathPredNode;

		if (RC_OK( rc = m_pool.poolCalloc( sizeof( PATH_PRED_NODE),
										(void **)&pPathPredNode)))
		{
			pPathPredNode->pXPathNode = pXPathNode;
			pPathPredNode->pNext = pPred->pXPathNodeList;
			pPred->pXPathNodeList = pPathPredNode;
		}
	}

	return( rc);
}

/***************************************************************************
Desc:	Check to see if any predicates need to be unioned with the passed
		in predicate.  If so, perform the union.
***************************************************************************/
FSTATIC RCODE fqCheckUnionPredicates(
	CONTEXT_PATH *	pContextPath,
	FLMUINT			uiLanguage,
	PATH_PRED *		pPred
	)
{
	RCODE			rc = NE_XFLM_OK;
	PATH_PRED *	pMergePred;
	FLMINT		iCmp;
	FLMBOOL		bDidOverlap;

	pMergePred = pContextPath->pFirstPred;

	// This should only be done on predicates that have a range
	// operator.

	flmAssert( pPred->eOperator == XFLM_RANGE_OP);
	while (pMergePred)
	{
		bDidOverlap = FALSE;
		if (pMergePred != pPred &&
			 pMergePred->eOperator == XFLM_RANGE_OP &&
			 pMergePred->uiCompareRules == pPred->uiCompareRules)
		{

			// If the value type is not compatible with the predicate's
			// value type, we cannot do the comparison, and there is
			// no overlap.

			if (!fqCanCompare( pMergePred->pFromValue, pPred->pFromValue) ||
				 !fqCanCompare( pMergePred->pFromValue, pPred->pUntilValue) ||
				 !fqCanCompare( pMergePred->pUntilValue, pPred->pFromValue) ||
				 !fqCanCompare( pMergePred->pUntilValue, pPred->pUntilValue))
			{
				// Nothing to do here
			}
			else if (RC_BAD( rc = fqCompareValues( pMergePred->pFromValue,
								pMergePred->bInclFrom, TRUE,
								pPred->pFromValue, pPred->bInclFrom, TRUE,
								pPred->uiCompareRules, uiLanguage, &iCmp)))
			{
				goto Exit;
			}
			else if (iCmp >= 0)
			{

				// From value is greater than or equal to the predicate's
				// from value.
				// If the from value is also less than or equal to the
				// predicate's until value, we have an overlap.

				if (RC_BAD( rc = fqCompareValues( pMergePred->pFromValue,
							pMergePred->bInclFrom, TRUE,
							pPred->pUntilValue, pPred->bInclUntil, FALSE,
							pPred->uiCompareRules, uiLanguage, &iCmp)))
				{
					goto Exit;
				}
				if (iCmp <= 0)
				{

					// If the until value is greater than the predicate's
					// until value, change the predicate's until value.

					if (RC_BAD( rc = fqCompareValues( pMergePred->pUntilValue,
								pMergePred->bInclUntil, FALSE,
								pPred->pUntilValue, pPred->bInclUntil, FALSE,
								pPred->uiCompareRules, uiLanguage, &iCmp)))
					{
						goto Exit;
					}
					if (iCmp > 0)
					{
						pPred->pUntilValue = pMergePred->pUntilValue;
						pPred->bInclUntil = pMergePred->bInclUntil;
					}
					bDidOverlap = TRUE;
				}
			}

			// At this point we already know that the from value is
			// less than the predicate's from value.
			// See if the until value is greater than or equal
			// to the from value.  If it is we have an overlap.

			else if (RC_BAD( rc = fqCompareValues( pMergePred->pUntilValue,
								pMergePred->bInclUntil, FALSE,
								pPred->pFromValue, pPred->bInclFrom, TRUE,
								pPred->uiCompareRules, uiLanguage, &iCmp)))
			{
				goto Exit;
			}
			else if (iCmp >= 0)
			{

				// Until value is greater than or equal to the predicate's
				// from value, so we definitely have an overlap.  We
				// already know that the from value is less than the
				// predicate's from value, so we will change that for sure.

				pPred->pFromValue = pMergePred->pFromValue;
				pPred->bInclFrom = pMergePred->bInclFrom;

				// See if the until value is greater than the
				// predicate's until value, in which case we need to
				// change the predicate's until value.

				if (RC_BAD( rc = fqCompareValues( pMergePred->pUntilValue,
								pMergePred->bInclUntil, FALSE,
								pPred->pUntilValue, pPred->bInclUntil, FALSE,
								pPred->uiCompareRules, uiLanguage, &iCmp)))
				{
					goto Exit;
				}
				if (iCmp > 0)
				{
					pPred->pUntilValue = pMergePred->pUntilValue;
					pPred->bInclUntil = pMergePred->bInclUntil;
				}
				bDidOverlap = TRUE;
			}
		}

		// If the predicates overlapped, remove pMergePred from the list
		// of predicates.  But move its list of predicate nodes into the
		// list off of pPred.

		if (bDidOverlap)
		{
			PATH_PRED_NODE *	pPathPredNode;

			// Merge the predicate node lists, if any - into pPred's list.

			if ((pPathPredNode = pPred->pXPathNodeList) != NULL)
			{
				while (pPathPredNode->pNext)
				{
					pPathPredNode = pPathPredNode->pNext;
				}
				pPathPredNode->pNext = pMergePred->pXPathNodeList;
			}
			else
			{
				pPred->pXPathNodeList = pMergePred->pXPathNodeList;
			}

			// Remove pMergePred from the list

			if (pMergePred->pPrev)
			{
				pMergePred->pPrev->pNext = pMergePred->pNext;
			}
			else
			{
				pContextPath->pFirstPred = pMergePred->pNext;
			}
			if (pMergePred->pNext)
			{
				pMergePred->pNext->pPrev = pMergePred->pPrev;

				// Set up so we are on the next node

				pMergePred = pMergePred->pNext;
			}
			else
			{
				pContextPath->pLastPred = pMergePred->pPrev;
			}
		}
		else
		{
			pMergePred = pMergePred->pNext;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Union a predicate into a context path.
***************************************************************************/
RCODE F_Query::unionPredicates(
	CONTEXT_PATH *			pContextPath,
	FQNODE *					pXPathNode,
	eQueryOperators		eOperator,
	FLMUINT					uiCompareRules,
	IF_OperandComparer *	pOpComparer,
	FQNODE *					pContextNode,
	FLMBOOL					bNotted,
	FQVALUE *				pQValue
	)
{
	RCODE				rc = NE_XFLM_OK;
	PATH_PRED *		pPred;
	FLMINT			iCmp;
	FLMBOOL			bDoMatch;
	FLMBOOL			bDidOverlap = FALSE;

	if (!pQValue || pQValue->eValType != XFLM_UTF8_VAL)
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
		bDoMatch = (eOperator == XFLM_EQ_OP &&
						(pQValue->uiFlags & VAL_IS_CONSTANT) &&
						(pQValue->uiFlags & VAL_HAS_WILDCARDS))
						? TRUE
						: FALSE;
	}

	if ((pPred = pContextPath->pFirstPred) != NULL)
	{
		if (eOperator == XFLM_EXISTS_OP || eOperator == XFLM_NE_OP)
		{

			// See if there is another operator that is an exact
			// match of this one.

			while (pPred)
			{
				if (pPred->eOperator == eOperator)
				{
					if ((bNotted && pPred->bNotted) ||
						 (!bNotted && !pPred->bNotted))
					{

						// Perfect match - no need to do any more.

						goto Exit;
					}
				}
				pPred = pPred->pNext;
			}
		}
		else if ((eOperator == XFLM_EQ_OP && !bDoMatch) ||
					eOperator == XFLM_LE_OP ||
					eOperator == XFLM_LT_OP ||
					eOperator == XFLM_GE_OP ||
					eOperator == XFLM_GT_OP)
		{

			// See if the operator overlaps with another range operator

			while (pPred)
			{
				if (pPred->eOperator == XFLM_RANGE_OP &&
					 pPred->uiCompareRules == uiCompareRules)
				{
					FQVALUE *	pFromValue;
					FQVALUE *	pUntilValue;
					FLMBOOL		bInclFrom;
					FLMBOOL		bInclUntil;

					pFromValue = (eOperator == XFLM_EQ_OP ||
									  eOperator == XFLM_GE_OP ||
									  eOperator == XFLM_GT_OP)
									  ? pQValue
									  : NULL;
					pUntilValue = (eOperator == XFLM_EQ_OP ||
									   eOperator == XFLM_LE_OP ||
									   eOperator == XFLM_LT_OP)
									   ? pQValue
									   : NULL;
					bInclFrom = (FLMBOOL)(eOperator == XFLM_EQ_OP ||
												 eOperator == XFLM_GE_OP
												 ? TRUE
												 : FALSE);
					bInclUntil = (FLMBOOL)(eOperator == XFLM_EQ_OP ||
												  eOperator == XFLM_LE_OP
												  ? TRUE
												  : FALSE);

					// If the value type is not compatible with the predicate's
					// value type, we cannot do the comparison, and there is
					// no overlap.

					if (!fqCanCompare( pQValue, pPred->pFromValue) ||
						 !fqCanCompare( pQValue, pPred->pUntilValue))
					{
						// Nothing to do here
					}
					else if (RC_BAD( rc = fqCompareValues( pFromValue,
										bInclFrom, TRUE,
										pPred->pFromValue, pPred->bInclFrom, TRUE,
										uiCompareRules, m_uiLanguage, &iCmp)))
					{
						goto Exit;
					}
					else if (iCmp >= 0)
					{

						// From value is greater than or equal to the predicate's
						// from value.
						// If the from value is also less than or equal to the
						// predicate's until value, we have an overlap.

						if (RC_BAD( rc = fqCompareValues( pFromValue,
									bInclFrom, TRUE,
									pPred->pUntilValue, pPred->bInclUntil, FALSE,
									uiCompareRules, m_uiLanguage, &iCmp)))
						{
							goto Exit;
						}
						if (iCmp <= 0)
						{

							// If the until value is greater than the predicate's
							// until value, change the predicate's until value.

							if (RC_BAD( rc = fqCompareValues( pUntilValue,
										bInclUntil, FALSE,
										pPred->pUntilValue, pPred->bInclUntil, FALSE,
										uiCompareRules, m_uiLanguage, &iCmp)))
							{
								goto Exit;
							}
							if (iCmp > 0)
							{
								pPred->pUntilValue = pUntilValue;
								pPred->bInclUntil = bInclUntil;
							}
							bDidOverlap = TRUE;
							goto Exit;
						}
					}

					// At this point we already know that the from value is
					// less than the predicate's from value.
					// See if the until value is greater than or equal
					// to the from value.  If it is we have an overlap.

					else if (RC_BAD( rc = fqCompareValues( pUntilValue,
										bInclUntil, FALSE,
										pPred->pFromValue, pPred->bInclFrom, TRUE,
										uiCompareRules, m_uiLanguage, &iCmp)))
					{
						goto Exit;
					}
					else if (iCmp >= 0)
					{

						// Until value is greater than or equal to the predicate's
						// from value, so we definitely have an overlap.  We
						// already know that the from value is less than the
						// predicate's from value, so we will change that for sure.

						pPred->pFromValue = pFromValue;
						pPred->bInclFrom = bInclFrom;

						// See if the until value is greater than the
						// predicate's until value, in which case we need to
						// change the predicate's until value.

						if (RC_BAD( rc = fqCompareValues( pUntilValue,
										bInclUntil, FALSE,
										pPred->pUntilValue, pPred->bInclUntil, FALSE,
										uiCompareRules, m_uiLanguage, &iCmp)))
						{
							goto Exit;
						}
						if (iCmp > 0)
						{
							pPred->pUntilValue = pUntilValue;
							pPred->bInclUntil = bInclUntil;
						}
						bDidOverlap = TRUE;
						goto Exit;
					}
				}
				pPred = pPred->pNext;
			}
		}
	}

	// Add a new predicate to the context path

	if (RC_BAD( rc = m_pool.poolCalloc( sizeof( PATH_PRED),
											(void **)&pPred)))
	{
		goto Exit;
	}
	pPred->uiCompareRules = uiCompareRules;
	pPred->pOpComparer = pOpComparer;

	// Link the predicate as the last predicate for the path

	if ((pPred->pPrev = pContextPath->pLastPred) != NULL)
	{
		pPred->pPrev->pNext = pPred;
	}
	else
	{
		pContextPath->pFirstPred = pPred;
	}
	pContextPath->pLastPred = pPred;

	// Set other items in the predicate.

	pPred->pContextNode = pContextNode;
	pPred->bNotted = bNotted;
	switch (eOperator)
	{
		case XFLM_EXISTS_OP:
		case XFLM_NE_OP:
			pPred->eOperator = eOperator;
			pPred->pFromValue = pQValue;
			break;
		case XFLM_APPROX_EQ_OP:
			pPred->eOperator = eOperator;
			pPred->pFromValue = pQValue;
			pPred->bInclFrom = TRUE;
			pPred->bInclUntil = TRUE;
			break;
		case XFLM_EQ_OP:
			if (bDoMatch)
			{
				pPred->eOperator = XFLM_MATCH_OP;
				pPred->pFromValue = pQValue;
			}
			else
			{
				pPred->eOperator = XFLM_RANGE_OP;
				pPred->pFromValue = pQValue;
				pPred->pUntilValue = pQValue;
				pPred->bInclFrom = TRUE;
				pPred->bInclUntil = TRUE;
			}
			break;
		case XFLM_LE_OP:
			pPred->eOperator = XFLM_RANGE_OP;
			pPred->pFromValue = NULL;
			pPred->pUntilValue = pQValue;
			pPred->bInclUntil = TRUE;
			break;
		case XFLM_LT_OP:
			pPred->eOperator = XFLM_RANGE_OP;
			pPred->pFromValue = NULL;
			pPred->pUntilValue = pQValue;
			pPred->bInclUntil = FALSE;
			break;
		case XFLM_GE_OP:
			pPred->eOperator = XFLM_RANGE_OP;
			pPred->pFromValue = pQValue;
			pPred->pUntilValue = NULL;
			pPred->bInclFrom = TRUE;
			break;
		case XFLM_GT_OP:
			pPred->eOperator = XFLM_RANGE_OP;
			pPred->pFromValue = pQValue;
			pPred->pUntilValue = NULL;
			pPred->bInclFrom = FALSE;
			break;
		default:
			rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
			goto Exit;
	}

Exit:

	if (RC_OK( rc))
	{
		PATH_PRED_NODE *	pPathPredNode;

		if (RC_OK( rc = m_pool.poolCalloc( sizeof( PATH_PRED_NODE),
										(void **)&pPathPredNode)))
		{
			pPathPredNode->pXPathNode = pXPathNode;
			pPathPredNode->pNext = pPred->pXPathNodeList;
			pPred->pXPathNodeList = pPathPredNode;

			if (bDidOverlap)
			{
				rc = fqCheckUnionPredicates( pContextPath, m_uiLanguage, pPred);
			}
		}
	}

	return( rc);
}

/***************************************************************************
Desc:	Prune a context out of the context tree.
***************************************************************************/
FSTATIC void fqClipContext(
	OP_CONTEXT *	pContext
	)
{

	// If this context had a parent, we can unlink it from
	// its parent.

	if (pContext->pParent)
	{
		if (pContext->pPrevSib)
		{
			pContext->pPrevSib->pNextSib = pContext->pNextSib;
		}
		else
		{
			pContext->pParent->pFirstChild = pContext->pNextSib;
		}
		if (pContext->pNextSib)
		{
			pContext->pNextSib->pPrevSib = pContext->pPrevSib;
		}
		else
		{
			pContext->pParent->pLastChild = pContext->pPrevSib;
		}
	}
}

/***************************************************************************
Desc:	Add a predicate to a context.
***************************************************************************/
RCODE F_Query::addPredicateToContext(
	OP_CONTEXT *			pContext,
	XPATH_COMPONENT *		pXPathContext,
	XPATH_COMPONENT *		pXPathComp,
	eQueryOperators		eOperator,
	FLMUINT					uiCompareRules,
	IF_OperandComparer *	pOpComparer,
	FQNODE *					pContextNode,
	FLMBOOL					bNotted,
	FQVALUE *				pQValue,
	FLMBOOL *				pbClipContext,
	FQNODE **				ppQNode
	)
{
	RCODE				rc = NE_XFLM_OK;
	CONTEXT_PATH *	pContextPath = NULL;

	*pbClipContext = FALSE;

	// Better be the leaf component of the XPATH

	flmAssert( !pXPathComp->pNext);

	// Convert the constant value in a node id predicate to
	// a 64 bit unsigned value.

	if (eOperator != XFLM_EXISTS_OP && pXPathComp->eXPathAxis == META_AXIS)
	{
		if (RC_BAD( rc = fqGetNodeIdValue( pQValue)))
		{
			goto Exit;
		}
	}

	// See if we can find a matching XPATH component.  If not,
	// create a new one.  NOTE: We can only match if the axis
	// is the SELF_AXIS, or the XPATH component is an attribute.
	// If the XPATH component is an attribute, it is guaranteed
	// there there will only be one instance of the attribute
	// Note also that the matching is only done if we are in the
	// context of another XPATH component.

	if (pXPathContext && !pXPathComp->pPrev && !pXPathComp->pNodeSource)
	{
		if (pXPathComp->eXPathAxis == META_AXIS)
		{
			pContextPath = pContext->pFirstPath;
			while (pContextPath)
			{
				if (!pContextPath->pXPathComponent->pPrev &&
					 pContextPath->pXPathComponent->uiDictNum ==
						pXPathComp->uiDictNum &&
					 pContextPath->pXPathComponent->eXPathAxis == META_AXIS)
				{
					break;
				}
				pContextPath = pContextPath->pNext;
			}
		}
		else if (pXPathComp->eNodeType == ELEMENT_NODE ||
					pXPathComp->eNodeType == DATA_NODE)
		{
			if (pXPathComp->eXPathAxis == SELF_AXIS)
			{
				pContextPath = pContext->pFirstPath;
				while (pContextPath)
				{
					if (!pContextPath->pXPathComponent->pPrev &&
						 pContextPath->pXPathComponent->eXPathAxis == SELF_AXIS &&
						 pContextPath->pXPathComponent->uiDictNum ==
							pXPathComp->uiDictNum &&
						 pContextPath->pXPathComponent->eNodeType ==
							pXPathComp->eNodeType)
					{
						break;
					}
					pContextPath = pContextPath->pNext;
				}
			}
		}
		else if (pXPathComp->eNodeType == ATTRIBUTE_NODE)
		{
			pContextPath = pContext->pFirstPath;
			while (pContextPath)
			{
				if (!pContextPath->pXPathComponent->pPrev &&
					 pContextPath->pXPathComponent->uiDictNum ==
						pXPathComp->uiDictNum &&
					 pContextPath->pXPathComponent->eNodeType == ATTRIBUTE_NODE)
				{
					break;
				}
				pContextPath = pContextPath->pNext;
			}
		}
	}

	// If we did not find one, allocate it.

	if (!pContextPath)
	{
		if (RC_BAD( rc = m_pool.poolCalloc( sizeof( CONTEXT_PATH),
									(void **)&pContextPath)))
		{
			goto Exit;
		}

		// Link the new context path into the context as its last path

		if ((pContextPath->pPrev = pContext->pLastPath) != NULL)
		{
			pContextPath->pPrev->pNext = pContextPath;
		}
		else
		{
			pContext->pFirstPath = pContextPath;
		}
		pContext->pLastPath = pContextPath;
		pContextPath->pXPathComponent = pXPathComp;
	}

	// See if this operator can be merged with another one.
	
	if (pContext->bIntersect)
	{
		if (RC_BAD( rc = intersectPredicates( pContextPath, *ppQNode,
									eOperator,
									uiCompareRules, pOpComparer,
									pContextNode, bNotted, pQValue, pbClipContext)))
		{
			goto Exit;
		}

		// If we get a false result, then we know that the
		// intersection of predicates is creating a situation where
		// it can never be true, so we will turn the root node of
		// the context into a XFLM_BOOL_VAL node with a FALSE value.
		// The branch of the query tree represented underneath this
		// node will have been cut off.  The caller must account for this.

		if (*pbClipContext)
		{
			FQNODE *			pRootNode = pContext->pQRootNode;

			pRootNode->eNodeType = FLM_VALUE_NODE;
			pRootNode->pFirstChild = NULL;
			pRootNode->pLastChild = NULL;
			pRootNode->pContext = pContext->pParent;
			pRootNode->currVal.eValType = XFLM_BOOL_VAL;
			pRootNode->currVal.uiFlags = VAL_IS_CONSTANT;
			pRootNode->currVal.val.eBool = XFLM_FALSE;
			*ppQNode = pRootNode;

			// Clip this context out of the context tree.
			// Don't want to travel down this path when optimizing.

			fqClipContext( pContext);
		}
	}
	else
	{
		if (RC_BAD( rc = unionPredicates( pContextPath, *ppQNode,
									eOperator,
									uiCompareRules, pOpComparer,
									pContextNode, bNotted, pQValue)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Import child context from pSrcContext into pDestContext.
***************************************************************************/
FSTATIC void fqImportChildContexts(
	OP_CONTEXT *	pDestContext,
	OP_CONTEXT *	pSrcContext)
{
	OP_CONTEXT *	pTmpContext;
	
	if ((pTmpContext = pSrcContext->pFirstChild) != NULL)
	{

		// Change all of the parent pointers of the child contexts to
		// point to pDestContext;

		while (pTmpContext)
		{
			flmAssert( (!pTmpContext->bIntersect && pDestContext->bIntersect) ||
						  (pTmpContext->bIntersect && !pDestContext->bIntersect));
			pTmpContext->pParent = pDestContext;
			pTmpContext = pTmpContext->pNextSib;
		}

		// Link all of pSrcContext's children as children of pDestContext.

		if ((pSrcContext->pFirstChild->pPrevSib =
					pDestContext->pLastChild) != NULL)
		{
			pDestContext->pLastChild->pNextSib = pSrcContext->pFirstChild;
		}
		else
		{
			pDestContext->pFirstChild = pSrcContext->pFirstChild;
		}
		pDestContext->pLastChild = pSrcContext->pLastChild;
		pSrcContext->pFirstChild = NULL;
		pSrcContext->pLastChild = NULL;
	}
}

/***************************************************************************
Desc:	Import context paths from pSrcContext into pDestContext.
***************************************************************************/
FSTATIC void fqImportContextPaths(
	OP_CONTEXT *	pDestContext,
	OP_CONTEXT *	pSrcContext)
{
	if (pSrcContext->pFirstPath)
	{
		if ((pSrcContext->pFirstPath->pPrev =
					pDestContext->pLastPath) != NULL)
		{
			pDestContext->pLastPath->pNext = pSrcContext->pFirstPath;
		}
		else
		{
			pDestContext->pFirstPath = pSrcContext->pFirstPath;
		}
		pDestContext->pLastPath = pSrcContext->pLastPath;
		pSrcContext->pFirstPath = NULL;
		pSrcContext->pLastPath = NULL;
	}
}

/***************************************************************************
Desc:	Import pSrcContext into pDestContext.
***************************************************************************/
FSTATIC void fqImportContext(
	OP_CONTEXT *	pDestContext,
	OP_CONTEXT *	pSrcContext)
{
	
	// Merge all of the child contexts from pSrcContext into pDestContext
	
	fqImportChildContexts( pDestContext, pSrcContext);
	
	// Merge all of the paths from pSrcContext into pDestContext

	fqImportContextPaths( pDestContext, pSrcContext);
}

/***************************************************************************
Desc:	Merge the context of pQNode, if any, into pDestContext
***************************************************************************/
FSTATIC void fqMergeContexts(
	FQNODE *			pQNode,
	OP_CONTEXT *	pDestContext
	)
{
	OP_CONTEXT *	pSrcContext = pQNode->pContext;

	// At this point, pQNode MUST have a context to merge in, and
	// pDestContext MUST be an intersect context.

	flmAssert( pSrcContext && pDestContext->bIntersect);

	// The context we are merging should not have any parent context.
	// It should be a root context.

	flmAssert( !pSrcContext->pNextSib && !pSrcContext->pPrevSib &&
				  !pSrcContext->pParent);

	// Root node of context should be same as pQNode.

	flmAssert( pSrcContext->pQRootNode == pQNode);

	if (pSrcContext->bIntersect)
	{
		
		// If the context to be merged is an intersect context (AND),
		// we take its child OP_CONTEXTs (which should all be non-intersect)
		// and make them children of the destination context.  We also
		// move its predicates into the predicate list of the destination
		// context.

		fqImportContext( pDestContext, pSrcContext);
	}
	else
	{

		// If the context to be merged is a non-intersect context (OR)
		// we  just put the whole thing in as a child of the destination
		// context.

		if ((pSrcContext->pPrevSib = pDestContext->pLastChild) != NULL)
		{
			pSrcContext->pPrevSib->pNextSib = pSrcContext;
		}
		else
		{
			pDestContext->pFirstChild = pSrcContext;
		}
		pDestContext->pLastChild = pSrcContext;
		pSrcContext->pParent = pDestContext;
	}
	pQNode->pContext = pDestContext;
}

/***************************************************************************
Desc:	Create a new context for predicates
***************************************************************************/
RCODE F_Query::createOpContext(
	OP_CONTEXT *	pParentContext,
	FLMBOOL			bIntersect,
	FQNODE *			pQRootNode
	)
{
	RCODE				rc = NE_XFLM_OK;
	OP_CONTEXT *	pContext;

	// Allocate a new context and link it in as a child
	// to the current context.

	if (RC_BAD( rc = m_pool.poolCalloc( sizeof( OP_CONTEXT),
									(void **)&pContext)))
	{
		goto Exit;
	}
	pQRootNode->pContext = pContext;
	pContext->pQRootNode = pQRootNode;
	pContext->bIntersect = bIntersect;
	pContext->bMustScan = FALSE;
	pContext->uiCost = (FLMUINT)(bIntersect
											? ~((FLMUINT)0)
											: 0);
	if ((pContext->pParent = pParentContext) != NULL)
	{
		if ((pContext->pPrevSib =
					pParentContext->pLastChild) != NULL)
		{
			pParentContext->pLastChild->pNextSib = pContext;
		}
		else
		{
			pParentContext->pFirstChild = pContext;
		}
		pParentContext->pLastChild = pContext;

	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Check to see if an XPATH component matches the XPATH context
		component we are inside of.
***************************************************************************/
FSTATIC void fqCheckPathMatch(
	XPATH_COMPONENT *	pXPathContextComponent,
	XPATH_COMPONENT *	pXPathComponent
	)
{

	// If this is a self:: axis, and we are in the context of another
	// xpath component, see if we match the context xpath component

	if (pXPathContextComponent &&
		 pXPathContextComponent->uiDictNum &&
		 (pXPathContextComponent->eNodeType == ELEMENT_NODE ||
		  pXPathContextComponent->eNodeType == DATA_NODE ||
		  pXPathContextComponent->eNodeType == ATTRIBUTE_NODE ||
		  pXPathContextComponent->eXPathAxis == META_AXIS) &&
		 pXPathComponent->eXPathAxis == SELF_AXIS &&
		 !pXPathComponent->pNext &&
		 ((pXPathComponent->eNodeType == pXPathContextComponent->eNodeType &&
			!pXPathComponent->uiDictNum) ||
			pXPathComponent->eNodeType == ANY_NODE_TYPE))
	{
		pXPathComponent->uiDictNum = pXPathContextComponent->uiDictNum;
		pXPathComponent->eNodeType = pXPathContextComponent->eNodeType;
		if( pXPathContextComponent->eXPathAxis == META_AXIS)
		{
			pXPathComponent->eXPathAxis = META_AXIS;
		}
	}
}

/***************************************************************************
Desc:	Get predicates for the XPATH node (pQNode) that was passed in.
***************************************************************************/
RCODE F_Query::getPathPredicates(
	FQNODE *				pParentNode,
	FQNODE **			ppQNode,
	XPATH_COMPONENT *	pXPathContext
	)
{
	RCODE					rc = NE_XFLM_OK;
	FQNODE *				pQNode = *ppQNode;
	FXPATH *				pXPath = pQNode->nd.pXPath;
	XPATH_COMPONENT *	pXPathComp = pXPath->pFirstComponent;
	OP_CONTEXT *		pContext;
	FQNODE *				pContextNode;
	FLMBOOL				bHadComponentExpressions = FALSE;
	FLMBOOL				bClippedContext;

	// pXPathContext is the XPATH component context for this
	// XPATH component.  This may be used in optimization.
	// It should have already been set up.

	flmAssert( pXPathComp->pXPathContext == pXPathContext);

	fqCheckPathMatch( pXPathContext, pXPathComp);

	// See if any of the XPATH components have expressions ([]).  If they
	// do, we want to AND this XPATH expression with them into
	// an intersect context.  If we are not currently in an intersect
	// context, we need to create one.

	pContext = NULL;
	pContextNode = pXPathContext ? pXPathContext->pXPathNode : NULL;
	while (pXPathComp)
	{
		if (pXPathComp->pExpr && !pContext)
		{
			bHadComponentExpressions = TRUE;

			// If we have not yet determined the context for the
			// XPATH components to be merged into, do it now.

			if (pParentNode)
			{
				pContext = pQNode->pContext = pParentNode->pContext;

				// If the context is not an intersect context, we need to
				// create a new one that is, and link it as the last child
				// of the current context.

				if (!pContext->bIntersect)
				{
					if (RC_BAD( rc = createOpContext( pParentNode->pContext,
												TRUE, pQNode)))
					{
						goto Exit;
					}
					pContext = pQNode->pContext;
				}
			}
			else
			{

				// We set the bIntersect flag to TRUE for this context,
				// even though it is not an AND operator.  This is done
				// because we know there will only ever be one predicate
				// in this context, and therefore, if it is ever merged
				// into another "intersect" context, we want the predicate
				// merged in by itself instead of the entire context.
				// See fqMergeContexts.

				if (RC_BAD( rc = createOpContext( NULL, TRUE, pQNode)))
				{
					goto Exit;
				}
				pContext = pQNode->pContext;
			}
			break;
		}
		pXPathComp = pXPathComp->pNext;
	}

	// Create a context if one was not created above.

	if (!pContext)
	{
		if (pParentNode)
		{
			pContext = pQNode->pContext = pParentNode->pContext;
		}
		else
		{

			// We set the bIntersect flag to TRUE for this context,
			// even though it is not an AND operator.  This is done
			// because we know there will only ever be one predicate
			// in this context, and therefore, if it is ever merged
			// into another "intersect" context, we want the predicate
			// merged in by itself instead of the entire context.
			// See fqMergeContexts.

			if (RC_BAD( rc = createOpContext( NULL, TRUE, pQNode)))
			{
				goto Exit;
			}
			pContext = pQNode->pContext;
		}
	}
	else
	{
		flmAssert( pQNode->pContext == pContext);
	}

	// Create predicate, if possible

	// Find the terminating XPATH component.

	pXPathComp = pXPath->pLastComponent;

	// See if we can create a predicate from this component.

	if (pXPathComp->pNodeSource)
	{

		// Create an exists predicate

		if (RC_BAD( rc = addPredicateToContext( pContext, pXPathContext,
									pXPathComp,
									XFLM_EXISTS_OP, 0, NULL,
									pContextNode, pQNode->bNotted, NULL,
									&bClippedContext, &pQNode)))
		{
			goto Exit;
		}
		
		// Context should never be clipped because it shouldn't ever
		// merge with another predicate.
		
		flmAssert( !bClippedContext);
	}
	else if (pXPathComp->uiDictNum &&
				 (pXPathComp->eXPathAxis == META_AXIS ||
				  pXPathComp->eNodeType == ELEMENT_NODE ||
				  pXPathComp->eNodeType == DATA_NODE ||
				  pXPathComp->eNodeType == ATTRIBUTE_NODE))
	{
		if (pParentNode)
		{
			if (isLogicalOp( pParentNode->nd.op.eOperator))
			{

				// Create an exists predicate

				if (RC_BAD( rc = addPredicateToContext( pContext, pXPathContext,
											pXPathComp,
											XFLM_EXISTS_OP, 0, NULL,
											pContextNode, pQNode->bNotted, NULL,
											&bClippedContext, &pQNode)))
				{
					goto Exit;
				}

				// If adding this predicate would cause a false result,
				// the root node of the context will have been turned into
				// a boolean value with a FALSE value.  There is no need to
				// process any more of this branch, because this branch will
				// have been cut off.  pQNode will have been altered.

				if (bClippedContext)
				{
					goto Exit;
				}
			}
			else if (isCompareOp( pParentNode->nd.op.eOperator))
			{

				// Sibling must be a value node

				if (pQNode->pNextSib)
				{
					flmAssert( !pQNode->pPrevSib);
					if (pQNode->pNextSib->eNodeType == FLM_VALUE_NODE)
					{

						// Create a compare predicate.

						if (RC_BAD( rc = addPredicateToContext( pContext,
													pXPathContext,
													pXPathComp,
													pParentNode->nd.op.eOperator,
													pParentNode->nd.op.uiCompareRules,
													pParentNode->nd.op.pOpComparer,
													pContextNode, pQNode->bNotted,
													&pQNode->pNextSib->currVal,
													&bClippedContext, &pQNode)))
						{
							goto Exit;
						}

						// If adding this predicate would cause a false
						// result, the root node of the context will
						// have been turned into a boolean value with a
						// FALSE value.  There is no need to process any
						// more of this branch, because this branch will
						// have been cut off.  pQNode will have been altered.

						if (bClippedContext)
						{
							goto Exit;
						}
					}
					else
					{

						// We have a comparison operation that cannot be
						// optimized.  If this is a non-intersect context,
						// there is no point in doing optimizations.
						// If it is an intersect context, other predicates
						// may come along that can be used to optimize.
						// We won't know until we have processed all of
						// them.

						if (!pContext->bIntersect)
						{
							pContext->bForceOptToScan = TRUE;
						}
					}
				}
				else
				{
					flmAssert( pQNode->pPrevSib);
					if (pQNode->pPrevSib->eNodeType == FLM_VALUE_NODE)
					{
						FQNODE *	pSaveSib;

						// Switch the two operators around

						switch (pParentNode->nd.op.eOperator)
						{
							case XFLM_EQ_OP:
							case XFLM_NE_OP:
								pSaveSib = pQNode->pPrevSib;
								break;
							case XFLM_LT_OP:
								pSaveSib = pQNode->pPrevSib;
								pParentNode->nd.op.eOperator = XFLM_GE_OP;
								break;
							case XFLM_LE_OP:
								pSaveSib = pQNode->pPrevSib;
								pParentNode->nd.op.eOperator = XFLM_GT_OP;
								break;
							case XFLM_GT_OP:
								pSaveSib = pQNode->pPrevSib;
								pParentNode->nd.op.eOperator = XFLM_LE_OP;
								break;
							case XFLM_GE_OP:
								pSaveSib = pQNode->pPrevSib;
								pParentNode->nd.op.eOperator = XFLM_LT_OP;
								break;
							default:
								pSaveSib = NULL;
								break;
						}

						// If we can switch the operator, we can create a
						// predicate for optimization.  Otherwise, we must
						// leave it alone - or assert?

						if (pSaveSib)
						{
							pQNode->pNextSib = pSaveSib;
							pQNode->pPrevSib = NULL;
							pSaveSib->pPrevSib = pQNode;
							pSaveSib->pNextSib = NULL;
							pParentNode->pFirstChild = pQNode;
							pParentNode->pLastChild = pSaveSib;

							// Create a compare predicate, but need to switch
							// the operators around

							if (RC_BAD( rc = addPredicateToContext( pContext,
														pXPathContext,
														pXPathComp,
														pParentNode->nd.op.eOperator,
														pParentNode->nd.op.uiCompareRules,
														pParentNode->nd.op.pOpComparer,
														pContextNode, pQNode->bNotted,
														&pQNode->pNextSib->currVal,
														&bClippedContext, &pQNode)))
							{
								goto Exit;
							}

							// If adding this predicate would cause a false
							// result, the root node of the context will
							// have been turned into a boolean value with a
							// FALSE value.  There is no need to process any
							// more of this branch, because this branch will
							// have been cut off.  pQNode will have been
							// altered.

							if (bClippedContext)
							{
								goto Exit;
							}
						}
						else
						{

							// We have a comparison operation that cannot be
							// optimized.

							if (!pContext->bIntersect)
							{
								pContext->bForceOptToScan = TRUE;
							}
						}
					}
					else
					{

						// We have a comparison operation that cannot be
						// optimized.

						if (!pContext->bIntersect)
						{
							pContext->bForceOptToScan = TRUE;
						}
					}
				}
			}
		}
		else
		{

			// Create an exists predicate

			if (RC_BAD( rc = addPredicateToContext( pContext, pXPathContext,
										pXPathComp, XFLM_EXISTS_OP, 0, NULL,
										pContextNode, pQNode->bNotted, NULL, &bClippedContext,
										&pQNode)))
			{
				goto Exit;
			}

			// If adding this predicate would cause a false
			// result, the root node of the context will
			// have been turned into a boolean value with a
			// FALSE value.  There is no need to process any
			// more of this branch, because this branch will
			// have been cut off.  pQNode will have been
			// altered.

			if (bClippedContext)
			{
				goto Exit;
			}
		}
	}
	else
	{

		// We have something in the expression that cannot be
		// optimized.

		if (!pContext->bIntersect)
		{
			pContext->bForceOptToScan = TRUE;
		}
	}

	if (bHadComponentExpressions)
	{

		// Now, merge in the expressions for each component of the
		// XPATH.  We wait to do this until AFTER the predicates
		// have been added above, because this "merge" does not
		// actually merge RANGE operators, etc., but just adds
		// the predicates to the list - which is all we want to
		// have happen at this point.

		pXPathComp = pXPath->pFirstComponent;
		while (pXPathComp)
		{
			if (pXPathComp->pExpr)
			{

				// Merge each expression's context into pContext, which
				// should be an intersection context - should have
				// been set up above.

				flmAssert( pContext && pContext->bIntersect);

				fqMergeContexts( pXPathComp->pExpr, pContext);
			}
			pXPathComp = pXPathComp->pNext;
		}
	}

Exit:

	*ppQNode = pQNode;

	return( rc);
}

/***************************************************************************
Desc:	Evaluate operands of an AND or OR operator to see if we can
		replace one.
		TRUE && P1 will be replaced with P1
		FALSE && P1 will be replaced with FALSE
		TRUE || P1 will be replaced with TRUE
		FALSE || P1 will be replaced with P1
***************************************************************************/
FSTATIC FQNODE * fqEvalLogicalOperands(
	FQNODE *		pQNode
	)
{
	FLMBOOL			bLeftIsBool = FALSE;
	FLMBOOL			bRightIsBool = FALSE;
	XFlmBoolType	eLeftBoolVal = XFLM_UNKNOWN;
	XFlmBoolType	eRightBoolVal = XFLM_UNKNOWN;
	FQNODE *			pLeftNode = pQNode->pFirstChild;
	FQNODE *			pRightNode = pLeftNode->pNextSib;
	FQNODE *			pReplacementNode = NULL;
	OP_CONTEXT *	pContext;
	OP_CONTEXT *	pParentContext;

	if (isBoolNode( pLeftNode))
	{
		bLeftIsBool = TRUE;
		eLeftBoolVal = pLeftNode->currVal.val.eBool;
	}
	if (isBoolNode( pRightNode))
	{
		bRightIsBool = TRUE;
		eRightBoolVal = pRightNode->currVal.val.eBool;
	}

	// If neither operand is a boolean value, there is no replacement
	// that can be done, but we still need to go up one level.

	if (!bLeftIsBool && !bRightIsBool)
	{
		goto Exit;
	}

	// Handle the case where both operands are boolean values.

	if (bLeftIsBool && bRightIsBool)
	{
		XFlmBoolType	eNewBoolVal;

		if (pQNode->nd.op.eOperator == XFLM_AND_OP)
		{
			if (eLeftBoolVal == XFLM_FALSE ||
				 eRightBoolVal == XFLM_FALSE)
			{
				eNewBoolVal = XFLM_FALSE;
			}
			else if (eLeftBoolVal == XFLM_TRUE &&
						eRightBoolVal == XFLM_TRUE)
			{
				eNewBoolVal = XFLM_TRUE;
			}
			else
			{
				eNewBoolVal = XFLM_UNKNOWN;
			}
		}
		else // XFLM_OR_OP
		{
			if (eLeftBoolVal == XFLM_TRUE ||
				 eRightBoolVal == XFLM_TRUE)
			{
				eNewBoolVal = XFLM_TRUE;
			}
			else if (eLeftBoolVal == XFLM_FALSE &&
						eRightBoolVal == XFLM_FALSE)
			{
				eNewBoolVal = XFLM_FALSE;
			}
			else
			{
				eNewBoolVal = XFLM_UNKNOWN;
			}
		}

		// Doesn't really matter which one we use to
		// replace the AND or OR node - we will use
		// the left one.

		pLeftNode->currVal.val.eBool = eNewBoolVal;
		pReplacementNode = pLeftNode;
	}
	else if (pQNode->nd.op.eOperator == XFLM_OR_OP)
	{

		// Operator is an OR, and only one of the operands
		// is a boolean value.

		if (bLeftIsBool)
		{
			pReplacementNode = (eLeftBoolVal == XFLM_TRUE)
									 ? pLeftNode
									 : pRightNode;
		}
		else
		{
			pReplacementNode = (eRightBoolVal == XFLM_TRUE)
									 ? pRightNode
									 : pLeftNode;
		}
	}
	else
	{

		// Operator is an AND, and only one of the operands is
		// a boolean value.

		if (bLeftIsBool)
		{
			pReplacementNode = (eLeftBoolVal != XFLM_TRUE)
									 ? pLeftNode
									 : pRightNode;
		}
		else
		{
			pReplacementNode = (eRightBoolVal != XFLM_TRUE)
									 ? pRightNode
									 : pLeftNode;
		}
	}

	// If the node we are going to replace (pQNode) is the root
	// of the context, change the context to point to the replacement
	// node as the new root of the context.  Also, if the replacement
	// node is a boolean, or a non-logical operator, set the context
	// to be an intersect context, so that if it is merged into another
	// context, it will merge correctly.

	pContext = pQNode->pContext;
	if (pContext->pQRootNode == pQNode)
	{
		pParentContext = pContext->pParent;
		if (pReplacementNode->eNodeType == FLM_VALUE_NODE)
		{

			// If this context node is a child to another
			// context, we can excise it completely

			// At this point, if pQNode was an OR, the
			// replacement node is guaranteed to be TRUE.
			// If the replacement node was an AND, the
			// replacement node is guaranteed to be FALSE.
			// We are at the top of a context, so the entire
			// context can be removed.

			// Strip off any child contexts and predicates of this context -
			// Don't want to travel down this path when optimizing.

			pReplacementNode->pContext = pParentContext;
			fqClipContext( pContext);
		}
		else if (pReplacementNode->eNodeType == FLM_FUNCTION_NODE)
		{
			
			// Better not be anything in the context that needs to
			// be imported into a parent context

			flmAssert( !pContext->pFirstPath);
			flmAssert( !pContext->pFirstChild);
			if (pParentContext)
			{
				pReplacementNode->pContext = pParentContext;
				fqClipContext( pContext);
			}
			else
			{
				pReplacementNode->pContext = pContext;
				pContext->bIntersect = TRUE;
				pContext->pQRootNode = pReplacementNode;
			}
		}
		else if (pReplacementNode->eNodeType != FLM_OPERATOR_NODE ||
			!isLogicalOp( pReplacementNode->nd.op.eOperator))
		{
			if (pParentContext)
			{
				// Context has a parent context.
				// If there is only one path and no child contexts,
				// import that path into the parent context.
				// Otherwise, change the root node of the context
				// to point to the replacement node.

				if( pContext->pFirstPath == pContext->pLastPath &&
					!pContext->pFirstChild)
				{
					fqImportContextPaths( pParentContext, pContext);
					pReplacementNode->pContext = pParentContext;
					fqClipContext( pContext);
				}
				else
				{
					pContext->pQRootNode = pReplacementNode;
				}
			}
			else
			{

				// No parent context, so set the bIntersect flag to
				// TRUE so that if it ever is merged to a higher
				// context, it will merge properly - because we
				// know that there is only one predicate in this
				// context.

				pContext->bIntersect = TRUE;
				pReplacementNode->pContext = pContext;
				pContext->pQRootNode = pReplacementNode;
			}
		}
		else
		{
			
			// pReplacement node is either an AND or OR operator.
			
			// At this point, we know that we are moving an AND or OR up the tree
			// to replace an AND or an OR.  If they are the same operator, they
			// would have already been in the same context.  If not, they would have
			// been in different contexts, and the replacement node will be the
			// same operator as the parent of the node being replaced.  Hence,
			// the replacement node's context needs to be merged with the parent
			// node's context.
			
			if (pReplacementNode->pContext != pContext)
			{
				fqClipContext( pContext);
				
				// Replace node's operator is different than the node it is
				// replacing. AND --> OR, or OR --> AND.  This means that
				// the parent context should be the same as the replacement
				// node's context, and hence, they should be merged.
				
				if (pParentContext)
				{
					flmAssert( (pParentContext->bIntersect &&
								   pReplacementNode->pContext->bIntersect) ||
								  (!pParentContext->bIntersect &&
								   !pReplacementNode->pContext->bIntersect));
					fqImportContext( pParentContext, pReplacementNode->pContext);
					fqClipContext( pReplacementNode->pContext);
					pReplacementNode->pContext = pParentContext;
				}
			}
		}
	}
	fqReplaceNode( pQNode, pReplacementNode);
	pQNode = pReplacementNode;

Exit:

	return( pQNode);
}

/***************************************************************************
Desc:	Clip a NOT node out of the tree.
***************************************************************************/
FSTATIC FQNODE * fqClipNotNode(
	FQNODE *		pQNode,
	FQNODE **	ppExpr
	)
{
	FQNODE *	pKeepNode;

	// If this NOT node has no parent, the root
	// of the tree needs to be set to its child.

	pKeepNode = pQNode->pFirstChild;

	// Child better not have any siblings - NOT nodes only have
	// one operand.

	flmAssert( !pKeepNode->pNextSib && !pKeepNode->pPrevSib);

	// Set child to point to the NOT node's parent.

	if ((pKeepNode->pParent = pQNode->pParent) == NULL)
	{
		*ppExpr = pKeepNode;
	}
	else
	{

		// Link child in where the NOT node used to be.

		if ((pKeepNode->pPrevSib = pQNode->pPrevSib) != NULL)
		{
			pKeepNode->pPrevSib->pNextSib = pKeepNode;
		}
		else
		{
			pKeepNode->pParent->pFirstChild = pKeepNode;
		}
		if ((pKeepNode->pNextSib = pQNode->pNextSib) != NULL)
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

/***************************************************************************
Desc:	Get predicates for a query expression.  Also strips out NOT nodes.
***************************************************************************/
RCODE F_Query::getPredicates(
	FQNODE **				ppExpr,
	FQNODE *					pStartNode,
	XPATH_COMPONENT *		pXPathContextComponent
	)
{
	RCODE						rc = NE_XFLM_OK;
	FQNODE *					pQNode = pStartNode ? pStartNode : *ppExpr;
	FQNODE *					pParentNode = NULL;
	eNodeTypes				eNodeType;
	eQueryOperators		eOperator;
	FLMBOOL					bNotted = FALSE;
	FLMBOOL					bGetPredicates;

	// Don't get predicates if this is a constant or arithmetic expression.
	// But if it is an arithmetic expression, still need to reduce it to
	// a single constant if possible.

	if (pQNode->eNodeType == FLM_VALUE_NODE ||
		 pQNode->eNodeType == FLM_OPERATOR_NODE &&
		 isArithOp( pQNode->nd.op.eOperator))
	{
		bGetPredicates = FALSE;
	}
	else
	{
		bGetPredicates = TRUE;
	}

	for (;;)
	{
		eNodeType = pQNode->eNodeType;

		// Need to save bNotted on each node so that when we traverse
		// back up the tree it can be reset properly.  If bNotted is
		// TRUE and pQNode is an operator, we may change the operator in
		// some cases.  Even if we change the operator, we still want to
		// set the bNotted flag because it also implies "for every" when set
		// to TRUE, and we need to remember that as well.

		pQNode->bNotted = bNotted;
		if (eNodeType == FLM_OPERATOR_NODE)
		{
			eOperator = pQNode->nd.op.eOperator;
			if (eOperator == XFLM_AND_OP || eOperator == XFLM_OR_OP)
			{

				// Logical sub-expressions can only be operands of
				// AND, OR, or NOT operators.

				if (pParentNode)
				{
					if (!isLogicalOp( pParentNode->nd.op.eOperator))
					{
						rc = RC_SET( NE_XFLM_Q_ILLEGAL_OPERAND);
						goto Exit;
					}
				}
				if (bNotted)
				{
					eOperator = (eOperator == XFLM_AND_OP
									 ? XFLM_OR_OP
									 : XFLM_AND_OP);
					pQNode->nd.op.eOperator = eOperator;
				}
				if (pParentNode)
				{
					flmAssert( pParentNode->pContext);
					if (pParentNode->nd.op.eOperator == eOperator)
					{
						pQNode->pContext = pParentNode->pContext;
					}
					else
					{
						if (RC_BAD( rc = createOpContext( pParentNode->pContext,
													(FLMBOOL)(eOperator == XFLM_AND_OP
																	? TRUE
																	: FALSE), pQNode)))
						{
							goto Exit;
						}
					}
				}
				else
				{
					if (RC_BAD( rc = createOpContext( NULL,
												(FLMBOOL)(eOperator == XFLM_AND_OP
																? TRUE
																: FALSE), pQNode)))
					{
						goto Exit;
					}
				}
			}
			else if (eOperator == XFLM_NOT_OP)
			{

				// Logical sub-expressions can only be operands of
				// AND, OR, or NOT operators.

				if (pParentNode)
				{
					if (!isLogicalOp( pParentNode->nd.op.eOperator))
					{
						rc = RC_SET( NE_XFLM_Q_ILLEGAL_OPERAND);
						goto Exit;
					}
				}
				bNotted = !bNotted;

				// Clip NOT nodes out of the tree.

				pQNode = fqClipNotNode( pQNode, ppExpr);
				pParentNode = pQNode->pParent;
				continue;
			}
			else if (isCompareOp( eOperator))
			{

				// Comparison sub-expressions can only be operands of
				// AND, OR, or NOT operators.

				if (pParentNode)
				{
					if (!isLogicalOp( pParentNode->nd.op.eOperator))
					{
						rc = RC_SET( NE_XFLM_Q_ILLEGAL_OPERAND);
						goto Exit;
					}

					// Associate context with parent node

					pQNode->pContext = pParentNode->pContext;
				}
				else
				{

					// We set the bIntersect flag to TRUE for this context,
					// even though it is not an AND operator.  This is done
					// because we know there will only ever be one predicate
					// in this context, and therefore, if it is ever merged
					// into another "intersect" context, we want the predicate
					// merged in by itself instead of the entire context.
					// See fqMergeContexts.

					if (RC_BAD( rc = createOpContext( NULL, TRUE, pQNode)))
					{
						goto Exit;
					}
				}
				if (bNotted)
				{
					switch (eOperator)
					{
						case XFLM_EQ_OP:
							eOperator = XFLM_NE_OP;
							break;
						case XFLM_NE_OP:
							eOperator = XFLM_EQ_OP;
							break;
						case XFLM_LT_OP:
							eOperator = XFLM_GE_OP;
							break;
						case XFLM_LE_OP:
							eOperator = XFLM_GT_OP;
							break;
						case XFLM_GT_OP:
							eOperator = XFLM_LE_OP;
							break;
						case XFLM_GE_OP:
							eOperator = XFLM_LT_OP;
							break;
						default:

							// Don't change the other operators.
							// Will just use the bNotted flag when
							// evaluating.

							break;
					}
					pQNode->nd.op.eOperator = eOperator;
				}
			}
			else
			{

				// Better be an arithmetic operator we are dealing with
				// at this point.

				flmAssert( isArithOp( eOperator));

				// Arithmetic sub-expressions can only be operands
				// of arithmetic or comparison operators

				if (pParentNode)
				{
					if (!isCompareOp( pParentNode->nd.op.eOperator) &&
						 !isArithOp( pParentNode->nd.op.eOperator))
					{
						rc = RC_SET( NE_XFLM_Q_ILLEGAL_OPERAND);
						goto Exit;
					}
				}
			}
		}
		else if (eNodeType == FLM_XPATH_NODE)
		{
			if (bGetPredicates)
			{
				if (RC_BAD( rc = getPathPredicates( pParentNode,
												&pQNode, pXPathContextComponent)))
				{
					goto Exit;
				}
			}

			// NOTE: Upon return pQNode may no longer be
			// an XPATH node.  This branch of the tree may
			// have been clipped as we evaluated predicates
			// that were ANDed together, in which case
			// pQNode will be pointing at a value node
			// that has a value of FALSE.  Either way,
			// there should be no children at this point.

			flmAssert( !pQNode->pFirstChild);
		}
		else if (eNodeType == FLM_FUNCTION_NODE)
		{

			// a function node should not have any children

			flmAssert( !pQNode->pFirstChild);
			
			// If this function is ORed into the context, set the
			// bForceOptToScan flag on the context.  This will
			// force the context to scan when we optimize.  Note that
			// we cannot just set the bMustScan flag, as that flag is
			// initialized by the optimization code to FALSE in the case
			// of a non-intersect context.
		
			if (bGetPredicates)
			{
				if (pParentNode)
				{
					if (pParentNode->nd.op.eOperator == XFLM_OR_OP)
					{
						pParentNode->pContext->bForceOptToScan = TRUE;
					}
				}
				else
				{

					// Need to have a context for later merging.

					if (RC_BAD( rc = createOpContext( NULL, TRUE, pQNode)))
					{
						goto Exit;
					}
					pQNode->pContext->bForceOptToScan = TRUE;
				}
			}
		}
		else
		{
			flmAssert( eNodeType == FLM_VALUE_NODE);

			// If bNotted is TRUE and we have a boolean value, change
			// the value: FALSE ==> TRUE, TRUE ==> FALSE.

			if (bNotted && pQNode->currVal.eValType == XFLM_BOOL_VAL)
			{
				if (pQNode->currVal.val.eBool == XFLM_TRUE)
				{
					pQNode->currVal.val.eBool = XFLM_FALSE;
				}
				else if (pQNode->currVal.val.eBool == XFLM_FALSE)
				{
					pQNode->currVal.val.eBool = XFLM_TRUE;
				}
			}

			// Values can only be operands of arithmetic or comparison operators,
			// unless they are boolean values, in which case they can only be
			// operands of logical operators.

			if (pParentNode)
			{
				if (pQNode->currVal.eValType == XFLM_BOOL_VAL)
				{
					if (!isLogicalOp( pParentNode->nd.op.eOperator))
					{
						rc = RC_SET( NE_XFLM_Q_ILLEGAL_OPERAND);
						goto Exit;
					}
				}
				else
				{
					if (!isCompareOp( pParentNode->nd.op.eOperator) &&
						 !isArithOp( pParentNode->nd.op.eOperator))
					{
						rc = RC_SET( NE_XFLM_Q_ILLEGAL_OPERAND);
						goto Exit;
					}
				}
			}

			// A value node should not have any children

			flmAssert( !pQNode->pFirstChild);
		}

		// Do traversal to child node, if any

		if (pQNode->pFirstChild)
		{
			pParentNode = pQNode;
			pQNode = pQNode->pFirstChild;
			continue;
		}

		// Go back up the tree until we hit something that has
		// a sibling.

		while (!pQNode->pNextSib)
		{

			// If there are no more parents, we are done.

			if ((pQNode = pQNode->pParent) == NULL)
			{
				goto Exit;
			}

			flmAssert( pQNode->eNodeType == FLM_OPERATOR_NODE);

			// Evaluate arithmetic expressions if both operands are
			// constants.

			if (isArithOp( pQNode->nd.op.eOperator) &&
				 pQNode->pFirstChild->eNodeType == FLM_VALUE_NODE &&
				 pQNode->pLastChild->eNodeType == FLM_VALUE_NODE)
			{
				if (RC_BAD( rc = fqArithmeticOperator(
											&pQNode->pFirstChild->currVal,
											&pQNode->pLastChild->currVal,
											pQNode->nd.op.eOperator,
											&pQNode->currVal)))
				{
					goto Exit;
				}
				pQNode->eNodeType = FLM_VALUE_NODE;
				pQNode->currVal.uiFlags = VAL_IS_CONSTANT;
				pQNode->pFirstChild = NULL;
				pQNode->pLastChild = NULL;
			}
			else
			{

				// For the AND and OR operators, check the operands to
				// see if they are boolean values.  Boolean values can
				// be weeded out of the criteria as we go back up the
				// tree.

				if (pQNode->nd.op.eOperator == XFLM_OR_OP ||
					 pQNode->nd.op.eOperator == XFLM_AND_OP)
				{
					pQNode = fqEvalLogicalOperands( pQNode);
					if (!pQNode->pParent)
					{
						*ppExpr = pQNode;
					}
				}
			}

			pParentNode = pQNode->pParent;
		}

		// pQNode will NEVER be NULL if we get here, because we
		// will jump to Exit in those cases.

		pQNode = pQNode->pNextSib;

		// Need to reset the bNotted flag to what it would have
		// been as we traverse back up the tree.

		bNotted = pParentNode->bNotted;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Determine if an XPATH component is getting the node's node id or
		document id.
***************************************************************************/
FINLINE FLMBOOL isNodeOrDocIdComponent(
	XPATH_COMPONENT *	pXPathComponent
	)
{
	return( pXPathComponent->eXPathAxis == META_AXIS &&
			  (pXPathComponent->uiDictNum == XFLM_META_NODE_ID ||
			   pXPathComponent->uiDictNum == XFLM_META_DOCUMENT_ID)
			  ? TRUE
			  : FALSE);
}

/***************************************************************************
Desc:	Get the meta data type for an xpath component, if any.
***************************************************************************/
FINLINE FLMUINT getMetaDataType(
	XPATH_COMPONENT *	pXPathComponent
	)
{
	return( pXPathComponent->eXPathAxis != META_AXIS
			  ? 0
			  : pXPathComponent->uiDictNum);
}

/***************************************************************************
Desc:	Get the node ID constant from an FQVALUE node.
***************************************************************************/
FSTATIC RCODE fqGetNodeIdValue(
	FQVALUE *	pQValue
	)
{
	RCODE	rc = NE_XFLM_OK;

	switch (pQValue->eValType)
	{
		case XFLM_UINT_VAL:
			pQValue->eValType = XFLM_UINT64_VAL;
			pQValue->val.ui64Val = (FLMUINT64)pQValue->val.uiVal;
			break;
		case XFLM_MISSING_VAL:
		case XFLM_UINT64_VAL:
			break;
		case XFLM_INT_VAL:
			pQValue->eValType = XFLM_UINT64_VAL;
			pQValue->val.ui64Val = (FLMUINT64)((FLMINT64)(pQValue->val.iVal));
			break;
		case XFLM_INT64_VAL:
			pQValue->eValType = XFLM_UINT64_VAL;
			pQValue->val.ui64Val = (FLMUINT64)(pQValue->val.i64Val);
			break;
		default:
			rc = RC_SET_AND_ASSERT( NE_XFLM_Q_INVALID_NODE_ID_VALUE);
			goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Optimize a predicate.
***************************************************************************/
RCODE F_Query::optimizePredicate(
	XPATH_COMPONENT *	pXPathComponent,
	PATH_PRED *			pPred
	)
{
	RCODE						rc = NE_XFLM_OK;
	ICD *						pIcd;
	ICD *						pTmpIcd;
	XPATH_COMPONENT *		pTmpXPathComponent;
	FLMBOOL					bCanCompareOnKey;
	FLMBOOL					bDoNodeMatch;
	FLMBOOL					bTmpCanCompareOnKey;
	FLMBOOL					bMustVerifyPath;
	FSIndexCursor *		pFSIndexCursor = NULL;
	FLMUINT					uiLeafBlocksBetween;
	FLMUINT					uiTotalRefs;
	FLMBOOL					bTotalsEstimated;
	FLMUINT					uiCost;
	IF_BufferIStream *	pBufferIStream = NULL;
	F_AttrElmInfo			defInfo;

	// Special handling for app. defined source nodes
	
	if (pXPathComponent->pNodeSource)
	{
		FLMBOOL	bMustScan;
		
		if (RC_OK( rc = pXPathComponent->pNodeSource->searchCost(
								(IF_Db *)m_pDb, pPred->bNotted,
								&pPred->OptInfo.uiCost,
								&bMustScan)))
		{
			if (bMustScan)
			{
				pPred->OptInfo.eOptType = XFLM_QOPT_FULL_COLLECTION_SCAN;
			}
			pPred->pNodeSource = pXPathComponent->pNodeSource;
			pPred->pNodeSource->AddRef();
		}
		goto Exit;
	}

	// A predicate of the form "A operator Value" will always
	// return FALSE if A is missing, regardless of the operator.
	// So will a predicate of "exists(A)"
	// This is because we do not use default data for missing
	// values. The only time it will return TRUE
	// when A is missing is if the predicate is notted.
	// Hence, a predicate that is notted cannot be optimized.

	if (pPred->bNotted || (pPred->pContextNode && pPred->pContextNode->bNotted))
	{
		pPred->OptInfo.uiCost = ~((FLMUINT)0);
		pPred->OptInfo.eOptType = XFLM_QOPT_FULL_COLLECTION_SCAN;
		goto Exit;
	}
	
	// Special handling for node id and document id attributes.
	// These will never be indexed, but we use a collection
	// cursor for them, if necessary.

	if (pXPathComponent->eXPathAxis == META_AXIS)
	{

		// Default is to do a full collection scan - may be changed below.

		pPred->OptInfo.uiCost = ~((FLMUINT)0);
		pPred->OptInfo.eOptType = XFLM_QOPT_FULL_COLLECTION_SCAN;

		if (pXPathComponent->uiDictNum == XFLM_META_NODE_ID ||
			 pXPathComponent->uiDictNum == XFLM_META_DOCUMENT_ID)
		{
			FLMBOOL	bDocumentIds =
					pXPathComponent->uiDictNum == XFLM_META_DOCUMENT_ID
					? TRUE
					: FALSE;

			// If the user has specified an index, we must set it up to
			// do an index scan.

			if (m_bIndexSet)
			{
				pPred->OptInfo.uiCost = ~((FLMUINT)0);
				pPred->OptInfo.eOptType = XFLM_QOPT_FULL_COLLECTION_SCAN;
			}
			else if (pPred->eOperator == XFLM_APPROX_EQ_OP)
			{
				pPred->OptInfo.uiCost = 1;

				// Value should have already been converted to a 64 bit
				// unsigned value.

				flmAssert( pPred->pFromValue->eValType == XFLM_UINT64_VAL);
				pPred->OptInfo.ui64NodeId = pPred->pFromValue->val.ui64Val;
				pPred->OptInfo.eOptType = XFLM_QOPT_SINGLE_NODE_ID;
				pPred->OptInfo.bMustVerifyPath = TRUE;
			}
			else if (pPred->eOperator == XFLM_RANGE_OP)
			{

				// Value should have already been converted to a 64 bit
				// unsigned value.

				flmAssert( pPred->pFromValue->eValType == XFLM_UINT64_VAL);
				pPred->OptInfo.ui64NodeId = pPred->pFromValue->val.ui64Val;
				if (!pPred->bInclFrom)
				{
					pPred->OptInfo.ui64NodeId++;
				}

				// Value should have already been converted to a 64 bit
				// unsigned value.

				flmAssert( pPred->pUntilValue->eValType == XFLM_UINT64_VAL);
				pPred->OptInfo.ui64EndNodeId = pPred->pUntilValue->val.ui64Val;
				if (!pPred->bInclUntil)
				{
					pPred->OptInfo.ui64EndNodeId--;
				}
				if (pPred->OptInfo.ui64NodeId == pPred->OptInfo.ui64EndNodeId)
				{
					pPred->OptInfo.uiCost = 1;
					pPred->OptInfo.eOptType = XFLM_QOPT_SINGLE_NODE_ID;
				}
				else
				{
					pPred->OptInfo.eOptType = XFLM_QOPT_NODE_ID_RANGE;
					if ((pPred->pFSCollectionCursor = f_new FSCollectionCursor) == NULL)
					{
						rc = RC_SET( NE_XFLM_MEM);
						goto Exit;
					}
					if (RC_BAD( rc = pPred->pFSCollectionCursor->setupRange( m_pDb,
												m_uiCollection, bDocumentIds,
												pPred->OptInfo.ui64NodeId,
												pPred->OptInfo.ui64EndNodeId,
												&uiLeafBlocksBetween, &uiTotalRefs,
												&bTotalsEstimated)))
					{
						pPred->pFSCollectionCursor->Release();
						pPred->pFSCollectionCursor = NULL;
						goto Exit;
					}
					pPred->OptInfo.uiCost = uiLeafBlocksBetween;
					if (!pPred->OptInfo.uiCost)
					{
						pPred->OptInfo.uiCost = 1;
					}
				}
				pPred->OptInfo.bMustVerifyPath = TRUE;
			}
			else
			{

				// Only other operators allowed would be the NE
				// operator.  EXISTS would have been converted to
				// a TRUE constant.

				flmAssert( pPred->eOperator == XFLM_NE_OP);
			}
		}
		goto Exit;
	}

	// Get the ICD chain

	if (pXPathComponent->eNodeType == ELEMENT_NODE)
	{
		if (RC_BAD( rc = m_pDb->m_pDict->getElement( m_pDb,
									pXPathComponent->uiDictNum,
									&defInfo)))
		{
			goto Exit;
		}
	}
	else if (pXPathComponent->eNodeType == ATTRIBUTE_NODE)
	{
		if (RC_BAD( rc = m_pDb->m_pDict->getAttribute( m_pDb,
											pXPathComponent->uiDictNum, &defInfo)))
		{
			goto Exit;
		}
	}

	pIcd = defInfo.m_pFirstIcd;

	// Get the indexes in the ICD chain that are suitable for this
	// predicate.

	for (; pIcd; pIcd = pIcd->pNextInChain)
	{

		// Stop at the first non-required ICD.

		if (!(pIcd->uiFlags &
				(ICD_REQUIRED_PIECE | ICD_REQUIRED_IN_SET)))
		{
			break;
		}
		
		// If this ICD is not a key component, we cannot use it.
		// Also, it must be the FIRST key component, and none of the
		// other components can be required.

		if (pIcd->uiKeyComponent != 1)
		{
			continue;
		}
		pTmpIcd = pIcd->pNextKeyComponent;
		while (pTmpIcd &&
					!(pTmpIcd->uiFlags & ICD_REQUIRED_PIECE))
		{
			pTmpIcd = pTmpIcd->pNextKeyComponent;
		}
		if (pTmpIcd)
		{
			continue;
		}
		
		// Check the following conditions of suitability:
		// 1) Index must be on the collection we are searching.
		// 2) Index must be on-line

		if (pIcd->pIxd->uiCollectionNum != m_uiCollection ||
			 (pIcd->pIxd->uiFlags & (IXD_OFFLINE | IXD_SUSPENDED)))
		{
			continue;
		}

		// If the user has specified an index, and this is not
		// that index, we will ignore it.

		if (m_bIndexSet && pIcd->uiIndexNum != m_uiIndex)
		{
			continue;
		}

		// Make sure the ICD's path is of less or equal specificity
		// to the path for this predicate.

		pTmpIcd = pIcd->pParent;
		pTmpXPathComponent = pXPathComponent;

		bMustVerifyPath = FALSE;

		while (pTmpIcd)
		{

			// If this is the self axis, we stay on the ICD we are currently on.
			// The XPATH component previous to/above this one is the same as
			// this one.

			if (pTmpXPathComponent->eXPathAxis == SELF_AXIS)
			{
				FLMUINT				uiTmpDictNum = pTmpXPathComponent->uiDictNum;
				eDomNodeType	eTmpNodeType = pTmpXPathComponent->eNodeType;

				if (pTmpXPathComponent->pPrev)
				{
					pTmpXPathComponent = pTmpXPathComponent->pPrev;
					if (pTmpXPathComponent->pExpr)
					{
						bMustVerifyPath = TRUE;
					}
				}
				else if ((pTmpXPathComponent =
								pTmpXPathComponent->pXPathContext) == NULL)
				{
					break;
				}
				if (eTmpNodeType == ANY_NODE_TYPE ||
					 (eTmpNodeType == pTmpXPathComponent->eNodeType &&
					  (uiTmpDictNum == pTmpXPathComponent->uiDictNum ||
					   !uiTmpDictNum)))
				{
					continue;
				}
				else
				{
					break;
				}
			}

			// If the ICD is the root tag, then the path needs to be
			// the root axis in order for it to be a match.

			if (pTmpIcd->uiDictNum == ELM_ROOT_TAG)
			{

				// Root tag should not have a parent.

				flmAssert( pTmpIcd->pParent == NULL);

				// If the path is not off of the root, then the
				// index's path is too specific to have indexed

				if (pTmpXPathComponent->eXPathAxis == ROOT_AXIS)
				{

					// Should not be anything previous to a root axis.

					flmAssert( pTmpXPathComponent->pPrev == NULL);
					pTmpXPathComponent = NULL;
					pTmpIcd = NULL;
				}
				break;
			}

			// Cannot match on anything except parent/child relationships

			if (pTmpXPathComponent->eXPathAxis != CHILD_AXIS &&
				pTmpXPathComponent->eXPathAxis != ATTRIBUTE_AXIS)
			{
				bMustVerifyPath = TRUE;
				break;
			}
			if (pTmpXPathComponent->pPrev)
			{
				pTmpXPathComponent = pTmpXPathComponent->pPrev;
				if (pTmpXPathComponent->pExpr)
				{
					bMustVerifyPath = TRUE;
				}
			}
			else if ((pTmpXPathComponent =
							pTmpXPathComponent->pXPathContext) == NULL)
			{
				break;
			}

			// See if the element is the same as the ICD.
			// NOTE: At this point, the ICD MUST be an element, because attributes
			// are not allowed as parent ICDs of another ICD.

			if (pTmpXPathComponent->eNodeType != ELEMENT_NODE ||
				 pTmpXPathComponent->uiDictNum != pTmpIcd->uiDictNum)
			{
				break;
			}

			// Go to ICD's parent

			pTmpIcd = pTmpIcd->pParent;
		}

		// If we get here and we did not get all the way up to the
		// parent ICD, the index path is more specific than the XPATH.

		if (pTmpIcd)
		{
			continue;
		}

		// If there are more components in the XPATH, it is more specific
		// than the ICD path, so we must verify the path.

		if (!bMustVerifyPath && pTmpXPathComponent &&
			 (pTmpXPathComponent->eXPathAxis == ROOT_AXIS ||
			  pTmpXPathComponent->pPrev ||
			  pTmpXPathComponent->pXPathContext))
		{
			bMustVerifyPath = TRUE;
		}

		bCanCompareOnKey = TRUE;

		// If the ICD is on element or attributes and the operator is not
		// the exists operator, we must also fetch the node to evaluate
		// the predicate.

		if ((pIcd->uiFlags & ICD_PRESENCE) && pPred->eOperator != XFLM_EXISTS_OP)
		{
			bCanCompareOnKey = FALSE;
		}

		// If the comparison rules aren't the same as those specified
		// on the ICD, we will need to read the node to do the comparison.

		if (bCanCompareOnKey && defInfo.m_uiDataType == XFLM_TEXT_TYPE)
		{
			if (!(pPred->uiCompareRules & XFLM_COMP_CASE_INSENSITIVE))
			{
				if (pIcd->uiCompareRules & XFLM_COMP_CASE_INSENSITIVE)
				{
					bCanCompareOnKey = FALSE;
				}
			}

			// Check comparison flags that must match exactly

			if ((pPred->uiCompareRules &
					(XFLM_COMP_COMPRESS_WHITESPACE |
					 XFLM_COMP_NO_WHITESPACE |
					 XFLM_COMP_NO_UNDERSCORES |
					 XFLM_COMP_NO_DASHES |
					 XFLM_COMP_WHITESPACE_AS_SPACE |
					 XFLM_COMP_IGNORE_LEADING_SPACE |
					 XFLM_COMP_IGNORE_TRAILING_SPACE)) !=
				 (pIcd->uiCompareRules &
					(XFLM_COMP_COMPRESS_WHITESPACE |
					 XFLM_COMP_NO_WHITESPACE |
					 XFLM_COMP_NO_UNDERSCORES |
					 XFLM_COMP_NO_DASHES |
					 XFLM_COMP_WHITESPACE_AS_SPACE |
					 XFLM_COMP_IGNORE_LEADING_SPACE |
					 XFLM_COMP_IGNORE_TRAILING_SPACE)))
			{
				bCanCompareOnKey = FALSE;
			}
		}

		// Need to select best metaphone for approximate equals
		// on text value.

		if (pPred->eOperator == XFLM_APPROX_EQ_OP &&
			 pIcd->uiFlags & ICD_METAPHONE &&
			 pPred->pFromValue->eValType == XFLM_UTF8_VAL)
		{
			FLMUINT				uiMeta;
			FLMUINT				uiAltMeta;
			FQVALUE				metaValue;
			FQVALUE *			pSaveValue = pPred->pFromValue;
			FLMUINT				uiMetaCost;
			
			if( RC_BAD( rc = FlmAllocBufferIStream( &pBufferIStream)))
			{
				goto Exit;
			}

			uiCost = 0;
			if (RC_BAD( rc = pBufferIStream->openStream( 
				(const char *)pPred->pFromValue->val.pucBuf,
				pPred->pFromValue->uiDataLen)))
			{
				goto Exit;
			}

			// This is a little bit trickiness here.  We need to set up the
			// metaphone value as a XFLM_UTF8_VAL, but then set
			// metaValue.val.uiVal.  That is because the code that
			// generates the key is expecting this - see kybldkey.cpp,
			// flmAddNonTextKeyPiece.

			metaValue.eValType = XFLM_UTF8_VAL;
			for (;;)
			{
				if( RC_BAD( rc = f_getNextMetaphone( 
					pBufferIStream, &uiMeta, &uiAltMeta)))
				{
					if (rc != NE_XFLM_EOF_HIT)
					{
						goto Exit;
					}
					rc = NE_XFLM_OK;
					break;
				}

				if (!uiMeta)
				{
					if (!uiAltMeta)
					{
						continue;
					}

					uiMeta = uiAltMeta;
				}

				// Generate the from and until keys for this index.  If the estimated
				// cost is low enough, we will look no further for a better index.

				if (pFSIndexCursor)
				{
					pFSIndexCursor->resetCursor();
				}
				else
				{
					if ((pFSIndexCursor = f_new FSIndexCursor) == NULL)
					{
						rc = RC_SET( NE_XFLM_MEM);
						goto Exit;
					}
				}

				// Temporarily change the from value in the predicate
				// to point to the metaphone value.  That is what the
				// key will be generated on.

				metaValue.val.uiVal = uiMeta;
				pPred->pFromValue = &metaValue;
				rc = pFSIndexCursor->setupKeys( m_pDb, pIcd->pIxd, pPred,
											&bDoNodeMatch, &bTmpCanCompareOnKey,
											&uiLeafBlocksBetween, &uiTotalRefs,
											&bTotalsEstimated);

				// Restore the from value in the predicate before
				// going any further.

				pPred->pFromValue = pSaveValue;
				if (RC_BAD( rc))
				{
					goto Exit;
				}

				// Will always have to fetch the node, so cost should
				// always include uiTotalRefs in this case.

				uiMetaCost = uiLeafBlocksBetween + uiTotalRefs;
				if (!uiMetaCost)
				{
					uiMetaCost = 1;
				}
				if (!uiCost || uiMetaCost < uiCost)
				{
					uiCost = uiMetaCost;
					if (uiCost < MIN_OPT_COST)
					{
						break;
					}
				}
			}
			
			if( pBufferIStream)
			{
				pBufferIStream->Release();
				pBufferIStream = NULL;
			}
			
			bTmpCanCompareOnKey = FALSE;
			bDoNodeMatch = TRUE;
		}
		else
		{

			// Generate the from and until keys for this index.  If the estimated
			// cost is low enough, we will look no further for a better index.

			if (pFSIndexCursor)
			{
				pFSIndexCursor->resetCursor();
			}
			else
			{
				if ((pFSIndexCursor = f_new FSIndexCursor) == NULL)
				{
					rc = RC_SET( NE_XFLM_MEM);
					goto Exit;
				}
			}
			if (RC_BAD( rc = pFSIndexCursor->setupKeys( m_pDb, pIcd->pIxd, pPred,
										&bDoNodeMatch, &bTmpCanCompareOnKey,
										&uiLeafBlocksBetween, &uiTotalRefs,
										&bTotalsEstimated)))
			{
				goto Exit;
			}
			if (!bTmpCanCompareOnKey)
			{
				bDoNodeMatch = TRUE;
			}
			uiCost = uiLeafBlocksBetween + uiTotalRefs;
		}

		// Could be that there are zero leaf blocks between and
		// bDoRecMatch is FALSE.  But we do not want a cost of
		// zero.

		if (!uiCost)
		{
			uiCost = 1;
		}

		if (!pPred->OptInfo.uiCost || uiCost < pPred->OptInfo.uiCost)
		{
			FSIndexCursor *	pTmpFSIndexCursor;
			F_NameTable *		pNameTable = NULL;

			// Exchange the temporary file system cursor and the
			// file system cursor inside the predicate.  Want to
			// keep the temporary one and reuse the one that was
			// inside the sub-query.

			pTmpFSIndexCursor = pPred->pFSIndexCursor;
			pPred->pFSIndexCursor = pFSIndexCursor;
			pFSIndexCursor = pTmpFSIndexCursor;
			pPred->OptInfo.eOptType = XFLM_QOPT_USING_INDEX;
			pPred->OptInfo.uiIxNum = pIcd->pIxd->uiIndexNum;
			pPred->OptInfo.szIxName [0] = 0;
			if (RC_OK( m_pDb->getNameTable( &pNameTable)))
			{
				FLMUINT	uiIxNameLen = sizeof( pPred->OptInfo.szIxName);
				if (RC_BAD( pNameTable->getFromTagTypeAndNum(
									m_pDb, ELM_INDEX_TAG,
									pPred->OptInfo.uiIxNum, NULL,
									(char *)pPred->OptInfo.szIxName,
									&uiIxNameLen, NULL, NULL,
									NULL, NULL, TRUE)))
				{
					pPred->OptInfo.szIxName [0] = 0;
				}
			}
			if (pNameTable)
			{
				pNameTable->Release();
			}
			pPred->OptInfo.uiCost = uiCost;
			pPred->OptInfo.bDoNodeMatch = bDoNodeMatch;
			pPred->OptInfo.bMustVerifyPath = bMustVerifyPath;
			pPred->OptInfo.bCanCompareOnKey =
				(bTmpCanCompareOnKey && bCanCompareOnKey) ? TRUE : FALSE;

			// If the cost is now low enough, we will quit looking

			if (uiCost < MIN_OPT_COST)
			{
				break;
			}
		}
	}

	// If no index was found, the predicate will force a full
	// collection scan.

	if (!pPred->OptInfo.uiCost)
	{
		pPred->OptInfo.uiCost = ~((FLMUINT)0);
		pPred->OptInfo.eOptType = XFLM_QOPT_FULL_COLLECTION_SCAN;
	}

Exit:

	if( pBufferIStream)
	{
		pBufferIStream->Release();
		pBufferIStream = NULL;
	}
	
	if (pFSIndexCursor)
	{
		pFSIndexCursor->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:	Optimize a context path.
***************************************************************************/
RCODE F_Query::optimizePath(
	CONTEXT_PATH *	pContextPath,
	PATH_PRED *		pSingleNodeIdPred,
	FLMBOOL			bIntersect
	)
{
	RCODE					rc = NE_XFLM_OK;
	PATH_PRED *			pPred;
	XPATH_COMPONENT *	pXPathComponent = pContextPath->pXPathComponent;

	pPred = pContextPath->pFirstPred;
	if (bIntersect)
	{
		pContextPath->uiCost = ~((FLMUINT)0);
		pContextPath->pSelectedPred = NULL;
		pContextPath->bMustScan = TRUE;

		// If we already know we have a single node id predicate, there
		// is no need to optimize any of the other predicates, because
		// this one is guaranteed to have the lowest cost: 1.

		if (pSingleNodeIdPred)
		{
			if (RC_BAD( rc = optimizePredicate( pXPathComponent,
											pSingleNodeIdPred)))
			{
				goto Exit;
			}

			// Cost better have returned as 1! and the optimization
			// better have been XFLM_QOPT_SINGLE_NODE_ID.

			flmAssert( pSingleNodeIdPred->OptInfo.uiCost == 1);
			flmAssert( pSingleNodeIdPred->OptInfo.eOptType ==
							XFLM_QOPT_SINGLE_NODE_ID);
			pContextPath->uiCost = pSingleNodeIdPred->OptInfo.uiCost;
			pContextPath->pSelectedPred = pSingleNodeIdPred;
			pContextPath->bMustScan = FALSE;
		}
		else
		{
			while (pPred)
			{
				if (RC_BAD( rc = optimizePredicate( pXPathComponent, pPred)))
				{
					goto Exit;
				}
				if (pPred->OptInfo.eOptType != XFLM_QOPT_FULL_COLLECTION_SCAN &&
					 (pPred->OptInfo.uiCost < pContextPath->uiCost ||
					  pContextPath->bMustScan))
				{
					pContextPath->uiCost = pPred->OptInfo.uiCost;
					if (pContextPath->pSelectedPred)
					{
						if (pContextPath->pSelectedPred->pFSIndexCursor)
						{
							pContextPath->pSelectedPred->pFSIndexCursor->Release();
							pContextPath->pSelectedPred->pFSIndexCursor = NULL;
						}
						else if (pContextPath->pSelectedPred->pFSCollectionCursor)
						{
							pContextPath->pSelectedPred->pFSCollectionCursor->Release();
							pContextPath->pSelectedPred->pFSCollectionCursor = NULL;
						}
						else if (pContextPath->pSelectedPred->pNodeSource)
						{
							pContextPath->pSelectedPred->pNodeSource->Release();
							pContextPath->pSelectedPred->pNodeSource = NULL;
						}
					}
					pContextPath->pSelectedPred = pPred;
					pContextPath->bMustScan = FALSE;

					// No need to evaluate more predicates if the cost is
					// MIN_OPT_COST or below.

					if (pPred->OptInfo.uiCost < MIN_OPT_COST)
					{
						break;
					}
				}
				else
				{
					if (pPred->pFSIndexCursor)
					{
						pPred->pFSIndexCursor->Release();
						pPred->pFSIndexCursor = NULL;
					}
					else if (pPred->pFSCollectionCursor)
					{
						pPred->pFSCollectionCursor->Release();
						pPred->pFSCollectionCursor = NULL;
					}
					else if (pPred->pNodeSource)
					{
						pContextPath->pSelectedPred->pNodeSource->Release();
						pContextPath->pSelectedPred->pNodeSource = NULL;
					}
				}
				pPred = pPred->pNext;
			}
		}
	}
	else
	{
		pContextPath->uiCost = 0;
		while (pPred)
		{
			if (RC_BAD( rc = optimizePredicate( pXPathComponent, pPred)))
			{
				goto Exit;
			}
			if (~((FLMUINT)0) - pContextPath->uiCost > pPred->OptInfo.uiCost &&
				 pPred->OptInfo.eOptType != XFLM_QOPT_FULL_COLLECTION_SCAN)
			{
				pContextPath->uiCost += pPred->OptInfo.uiCost;
			}
			else
			{
				pContextPath->uiCost = ~((FLMUINT)0);
				pContextPath->bMustScan = TRUE;
				break;
			}
			pPred = pPred->pNext;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Optimize a context.  It's child contexts have already been
		optimized.
***************************************************************************/
RCODE F_Query::optimizeContext(
	OP_CONTEXT *	pContext,
	CONTEXT_PATH *	pSingleNodeIdPath,
	PATH_PRED *		pSingleNodeIdPred
	)
{
	RCODE				rc = NE_XFLM_OK;
	OP_CONTEXT *	pChildContext;
	CONTEXT_PATH *	pContextPath;

	// If this context has no CONTEXT_PATHs and no child contexts,
	// it had to have had only non-optimizable predicates, or
	// no context would have been created.  In that case, this context
	// must be forced to scan.

	if (pContext->bForceOptToScan ||
		 (!pContext->pFirstPath && !pContext->pFirstChild))
	{
		pContext->uiCost = ~((FLMUINT)0);
		pContext->bMustScan = TRUE;
		goto Exit;
	}

	// Optimize each of the context paths, if any.
	// If this is an intersect context, select the context path
	// one a cost that is lower than the context's current cost.
	// The context's current cost may have already been set when
	// its child contexts were optimized.

	if (pContext->bIntersect)
	{
		pContext->bMustScan = TRUE;
		pContext->uiCost = ~((FLMUINT)0);

		// Determine what the lowest cost child context is.
		// If we have a single node id path, we already know
		// it is guaranteed to have the lowest cost: 1.

		if (pSingleNodeIdPath)
		{
			if (RC_BAD( rc = optimizePath( pSingleNodeIdPath,
										pSingleNodeIdPred, TRUE)))
			{
				goto Exit;
			}

			// Cost better have returned as one!
			// Its bMustScan flag had also better be set to FALSE!

			flmAssert( pSingleNodeIdPath->uiCost == 1);
			flmAssert( !pSingleNodeIdPath->bMustScan);
			pContext->uiCost = pSingleNodeIdPath->uiCost;
			pContext->pSelectedPath = pSingleNodeIdPath;
			pContext->pSelectedChild = NULL;
			pContext->bMustScan = FALSE;
		}
		else
		{
			pChildContext = pContext->pFirstChild;
			while (pChildContext)
			{
				if (!pChildContext->bMustScan &&
					 (pChildContext->uiCost < pContext->uiCost ||
					  pContext->bMustScan))
				{
					pContext->uiCost = pChildContext->uiCost;
					pContext->pSelectedChild = pChildContext;
					pContext->pSelectedPath = NULL;
					pContext->bMustScan = FALSE;
				}
				pChildContext = pChildContext->pNextSib;
			}
			
			// Find the most optimal context path

			pContextPath = pContext->pFirstPath;
			while (pContextPath && pContext->uiCost >= MIN_OPT_COST)
			{
				if (RC_BAD( rc = optimizePath( pContextPath, NULL, TRUE)))
				{
					goto Exit;
				}
				if (!pContextPath->bMustScan &&
					 (pContextPath->uiCost < pContext->uiCost ||
					  pContext->bMustScan))
				{
					pContext->uiCost = pContextPath->uiCost;
					pContext->pSelectedPath = pContextPath;
					pContext->pSelectedChild = NULL;
					pContext->bMustScan = FALSE;

					// If the cost is below MIN_OPT_COST, no need to do any more
					// cost analysis.

					if (pContextPath->uiCost < MIN_OPT_COST)
					{
						break;
					}
				}
				pContextPath = pContextPath->pNext;
			}
		}
	}
	
	// Have a UNION (OR) context

	else
	{
		
		// In the case of union, there is no point in optimizing any
		// of the context paths paths once we discover that a context
		// must scan the database.

		pContext->bMustScan = FALSE;
		pContext->uiCost = 0;

		// Add in the cost of each child context.

		pChildContext = pContext->pFirstChild;
		while (pChildContext)
		{
			if (pChildContext->bMustScan)
			{
				pContext->bMustScan = TRUE;
				pContext->uiCost = ~((FLMUINT)0);

				// Once we set the bMustScan flag, there is no point
				// in going any further.

				break;
			}
			else if (~((FLMUINT)0) - pContext->uiCost > pChildContext->uiCost)
			{
				pContext->uiCost += pChildContext->uiCost;
			}
			else
			{
				pContext->uiCost = ~((FLMUINT)0);
				pContext->bMustScan = TRUE;

				// Once we set the bMustScan flag, there is no point
				// in going any further.

				break;
			}
			pChildContext = pChildContext->pNextSib;
		}

		// In the case of a non-intersecting context, sum the costs of
		// each context path into the cost for the context until we
		// hit ~0, at which time we will mark the context as
		// "must scan" and quit attempting to optimize it.

		pContextPath = pContext->pFirstPath;
		while (pContextPath && !pContext->bMustScan)
		{
			if (RC_BAD( rc = optimizePath( pContextPath, NULL, FALSE)))
			{
				goto Exit;
			}
			if (pContextPath->bMustScan)
			{
				pContext->uiCost = ~((FLMUINT)0);
				pContext->bMustScan = TRUE;

				// No need to do any more optimization once we have
				// determined to do a collection scan.

				break;
			}
			else if (~((FLMUINT)0) - pContext->uiCost > pContextPath->uiCost)
			{
				pContext->uiCost += pContextPath->uiCost;
			}
			else
			{
				pContext->uiCost = ~((FLMUINT)0);
				pContext->bMustScan = TRUE;

				// No need to do any more optimization once we have
				// determined to do a collection scan.

				break;
			}
			pContextPath = pContextPath->pNext;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Setup to scan an index - the index was specified by the user.
***************************************************************************/
RCODE F_Query::setupIndexScan( void)
{
	RCODE		rc = NE_XFLM_OK;
	IXD *		pIxd;
	FLMBOOL	bDoNodeMatch;
	FLMBOOL	bCanCompareOnKey;

	flmAssert( m_uiIndex);

	// If the index is not on-line or it is not associated
	// with the collection, we have a problem.

	if (RC_BAD( rc = m_pDb->m_pDict->getIndex( m_uiIndex, NULL,
								&pIxd, FALSE)))
	{
		goto Exit;
	}
	if (pIxd->uiCollectionNum != m_uiCollection)
	{
		rc = RC_SET( NE_XFLM_BAD_IX);
		goto Exit;
	}

	if ((m_pFSIndexCursor = f_new FSIndexCursor) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
	}

	// Setup to scan from beginning of key to end of key.

	if (RC_BAD( rc = m_pFSIndexCursor->setupKeys( m_pDb, pIxd, NULL,
								&bDoNodeMatch, &bCanCompareOnKey,
								NULL, NULL, NULL)))
	{
		goto Exit;
	}
	m_bScanIndex = TRUE;
	m_bScan = FALSE;

Exit:

	return( rc);
}

/***************************************************************************
Desc:	This routine determines if an ICD has an descendents that are key
		components.
***************************************************************************/
FSTATIC FLMBOOL haveChildKeyComponents(
	ICD *	pParentIcd)
{
	ICD *	pTmpIcd;
	
	if ((pTmpIcd = pParentIcd->pFirstChild) == NULL)
	{
		return( FALSE);
	}
	for (;;)
	{
		if (pTmpIcd->uiKeyComponent)
		{
			return( TRUE);
		}
		if (pTmpIcd->pFirstChild)
		{
			pTmpIcd = pTmpIcd->pFirstChild;
		}
		else
		{
			while (!pTmpIcd->pNextSibling)
			{
				if ((pTmpIcd = pTmpIcd->pParent) == pParentIcd)
				{
					return( FALSE);
				}
			}
			if (!pTmpIcd)
			{
				break;
			}
			pTmpIcd = pTmpIcd->pNextSibling;
		}
	}
	return( FALSE);
}

/***************************************************************************
Desc:	Determines if the optimization index already has the order we need
		for sorting.
***************************************************************************/
RCODE F_Query::checkSortIndex(
	FLMUINT	uiOptIndex)
{
	RCODE		rc = NE_XFLM_OK;
	IXD *		pOptIxd;
	ICD *		pSortIcd;
	ICD *		pOptIcd;

	// Should not be called unless we have a sort key specified.
	
	flmAssert( m_pSortIxd);
	
	// Get the index definition.
	
	if (RC_BAD( rc = m_pDb->m_pDict->getIndex( uiOptIndex, NULL, &pOptIxd, TRUE)))
	{
		RC_UNEXPECTED_ASSERT( rc);
		goto Exit;
	}
	
	// Verify that the language is the same.
	
	if (m_pSortIxd->uiLanguage != pOptIxd->uiLanguage)
	{
		goto Exit;
	}
	
	// Make sure the sort IXD is completely represented in the optimization IXD
	// If we find all of the keys in the same context, the optimization
	// index will work as the sort index.  Even if the optimization index
	// has more key components it will work, because it will provide the
	// keys in the same order.
	
	pSortIcd = m_pSortIxd->pIcdTree;
	pOptIcd = pOptIxd->pIcdTree;

	for (;;)
	{
		
		// See if there is a matching opt ICD in the list of siblings to
		// opt ICD.
		
		pOptIcd = (pOptIcd->pParent)
					 ? pOptIcd->pParent->pFirstChild
					 : pOptIxd->pIcdTree;
		while (pOptIcd)
		{
			if (pOptIcd->uiDictNum == pSortIcd->uiDictNum &&
				 (pOptIcd->uiFlags & ICD_IS_ATTRIBUTE) ==
				 (pSortIcd->uiFlags & ICD_IS_ATTRIBUTE))
			{
				
				// If the match ICD is a key component, we want a
				// search ICD that is the same key component and compare rules.
				
				if (pSortIcd->uiKeyComponent)
				{
					if (pOptIcd->uiKeyComponent == pSortIcd->uiKeyComponent &&
						 pOptIcd->uiFlags == pSortIcd->uiFlags)
					{
						break;
					}
				}
				else
				{
					break;
				}
			}
			pOptIcd = pOptIcd->pNextSibling;
		}
		
		// Did we find a matching opt ICD?
		
		if (!pOptIcd)
		{

			// If the sort ICD is a key component, or there are key ICDs
			// subordinate to the sort ICD, then the indexes cannot match,
			// because there is no matching context for the child key ICDs.			
			
			if (pSortIcd->uiKeyComponent || haveChildKeyComponents( pSortIcd))
			{
				goto Exit;
			}
			
Check_Siblings:

			while (!pSortIcd->pNextSibling)
			{
				if ((pSortIcd = pSortIcd->pParent) == NULL)
				{
					break;
				}
				pOptIcd = pOptIcd->pParent;
				
				// If pSortIcd != NULL, pOptIcd better be also!
				
				flmAssert( pOptIcd);
			}
			if (!pSortIcd)
			{
				
				// Done - index key components match.
				
				break;
			}
			
			// pNextSibling better be non-NULL at this point.
			// NOTE: Do not set pOptIcd to its sibling - it may not have one.
			// That doesn't matter, because we will search for a matching
			// sibling up above.
			
			pSortIcd = pSortIcd->pNextSibling;
			
			// No need to check this sort ICD if it is not a key
			// component and it has no child ICDs.
			
			if (!pSortIcd->uiKeyComponent && !pSortIcd->pFirstChild)
			{
				goto Check_Siblings;
			}
		}
		else
		{
			
			// Go to the child nodes, if any
			
			if (pSortIcd->pFirstChild)
			{
				if (!pOptIcd->pFirstChild)
				{
					
					// Sort ICD has a child, opt ICD doesn't.  See if there are
					// any index components subordinate to sort ICD.  If so, the indexes
					// are different.  Otherwise, it really doesn't matter - we can
					// simply proceed with sort ICD's siblings.

					if (haveChildKeyComponents( pSortIcd))
					{
						goto Exit;
					}
					goto Check_Siblings;
				}
				else
				{
					pSortIcd = pSortIcd->pFirstChild;
					pOptIcd = pOptIcd->pFirstChild;
				}
			}
			else
			{
				goto Check_Siblings;
			}
		}
	}

	m_bEntriesAlreadyInOrder = TRUE;
	
Exit:

	return( rc);
}

/***************************************************************************
Desc:	Optimize a query expression.
***************************************************************************/
RCODE F_Query::optimize( void)
{
	RCODE				rc = NE_XFLM_OK;
	FQNODE *			pQNode = m_pQuery;
	OP_CONTEXT *	pContext;

	if (m_bOptimized)
	{
		rc = RC_SET( NE_XFLM_Q_ALREADY_OPTIMIZED);
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
	if ((m_pPrev = m_pDatabase->m_pLastQuery) != NULL)
	{
		m_pPrev->m_pNext = this;
	}
	else
	{
		m_pDatabase->m_pFirstQuery = this;
	}
	m_pDatabase->m_pLastQuery = this;
	m_pDatabase->unlockMutex();
	
	// Verify sort keys, if any
	
	if (m_pSortIxd)
	{
		if (RC_BAD( rc = verifySortKeys()))
		{
			goto Exit;
		}
	}
	
	// Make sure we have a completed expression

	if( m_pCurExprState)
	{
		if ( m_pCurExprState->pPrev ||
			  m_pCurExprState->uiNestLevel ||
			  (m_pCurExprState->pLastNode &&
				m_pCurExprState->pLastNode->eNodeType == FLM_OPERATOR_NODE))
		{
			rc = RC_SET( NE_XFLM_Q_INCOMPLETE_QUERY_EXPR);
			goto Exit;
		}
		if (RC_BAD( rc = getPredicates( &m_pCurExprState->pExpr, NULL, NULL)))
		{
			goto Exit;
		}

		m_pQuery = m_pCurExprState->pExpr;
	}

	m_uiLanguage = m_pDb->getDefaultLanguage();

	// An empty expression should scan the database and return everything.

	if ((pQNode = m_pQuery) == NULL)
	{
		if (m_bIndexSet && m_uiIndex)
		{
			rc = setupIndexScan();
		}
		else
		{
			m_bScan = TRUE;
		}
		goto Exit;
	}

	// Handle the case of a value node or arithmetic expression at the root
	// These types of expressions do not return results from the database.

	if (pQNode->eNodeType == FLM_VALUE_NODE)
	{
		if (pQNode->currVal.eValType == XFLM_BOOL_VAL &&
			 pQNode->currVal.val.eBool == XFLM_TRUE)
		{
			m_bScan = TRUE;
		}
		else
		{
			m_bEmpty = TRUE;
		}
	}
	else if (pQNode->eNodeType == FLM_OPERATOR_NODE &&
		  isArithOp( pQNode->nd.op.eOperator))
	{
		m_bEmpty = TRUE;
		goto Exit;
	}

	// If the user explicitly said to NOT use an index, we will not

	if (m_bIndexSet && !m_uiIndex)
	{
		m_bScan = TRUE;
		goto Exit;
	}

	// Start with the context in the root node.

	if ((pContext = pQNode->pContext) == NULL)
	{
		m_bScan = TRUE;
		goto Exit;
	}

	for (;;)
	{
		CONTEXT_PATH *	pSingleNodeIdPath = NULL;
		PATH_PRED *		pSingleNodeIdPred = NULL;

		// See if any of the context paths in this context have a
		// single node ID predicate.  If so, we can optimize it right
		// away and ignore all of the other predicates.

		if (pContext->bIntersect)
		{
			pSingleNodeIdPath = pContext->pFirstPath;
			while (pSingleNodeIdPath)
			{
				if (isNodeOrDocIdComponent( pSingleNodeIdPath->pXPathComponent))
				{

					// See if any of the predicates are single-node id.

					pSingleNodeIdPred = pSingleNodeIdPath->pFirstPred;
					while (pSingleNodeIdPred)
					{
						if (pSingleNodeIdPred->pFromValue &&
							 pSingleNodeIdPred->pUntilValue)
						{

							// From and until values should have already been
							// converted to appropriate node ids - 64 bit values.

							flmAssert( pSingleNodeIdPred->pFromValue->eValType ==
											XFLM_UINT64_VAL);
							flmAssert( pSingleNodeIdPred->pUntilValue->eValType ==
											XFLM_UINT64_VAL);
							if (!pSingleNodeIdPred->bInclFrom)
							{
								pSingleNodeIdPred->pFromValue->val.ui64Val++;
								pSingleNodeIdPred->bInclFrom = TRUE;
							}
							if (!pSingleNodeIdPred->bInclUntil)
							{
								pSingleNodeIdPred->pUntilValue->val.ui64Val--;
								pSingleNodeIdPred->bInclUntil = TRUE;
							}
							if (pSingleNodeIdPred->pFromValue->val.ui64Val ==
								 pSingleNodeIdPred->pUntilValue->val.ui64Val)
							{
								break;
							}
						}
						pSingleNodeIdPred = pSingleNodeIdPred->pNext;
					}
				}
				if (pSingleNodeIdPred)
				{
					break;
				}

				pSingleNodeIdPath = pSingleNodeIdPath->pNext;
			}
		}
		if (pContext->pFirstChild && !pSingleNodeIdPath)
		{
			pContext = pContext->pFirstChild;
			continue;
		}

		// Optimize the context paths of the context we are on

		if (RC_BAD( rc = optimizeContext( pContext,
									pSingleNodeIdPath, pSingleNodeIdPred)))
		{
			goto Exit;
		}

		// Go to sibling context, if any

		while (!pContext->pNextSib)
		{
			if ((pContext = pContext->pParent) == NULL)
			{
				break;
			}

			// Child contexts have all been optimized and the cost
			// reflected up to this context.  But there may be
			// CONTEXT_PATHs on this context that will change that
			// cost.

			if (RC_BAD( rc = optimizeContext( pContext, NULL, NULL)))
			{
				goto Exit;
			}
		}

		// If pContext is NULL at this point, there are no
		// other contexts to optimize.

		if (!pContext)
		{
			break;
		}
		pContext = pContext->pNextSib;

		// There has to have been a sibling context at this point.

		flmAssert( pContext);
	}

	if (m_pQuery->pContext->bMustScan)
	{

		// If we were told to use an index, we will use that index
		// and scan the entire index.

		if (m_bIndexSet)
		{
			if (RC_BAD( rc = setupIndexScan()))
			{
				goto Exit;
			}
		}
		else
		{
			m_bScan = TRUE;
		}
	}

Exit:

	if (RC_OK( rc) && !m_bEmpty)
	{
		FLMUINT	uiOptIndex = 0;
		
		if (m_bScan || m_bScanIndex)
		{
			if (m_bScan)
			{
				m_scanOptInfo.eOptType = XFLM_QOPT_FULL_COLLECTION_SCAN;
			}
			else
			{
				F_NameTable *	pNameTable = NULL;
				
				m_scanOptInfo.eOptType = XFLM_QOPT_USING_INDEX;
				m_scanOptInfo.uiCost = ~((FLMUINT)0);
				m_scanOptInfo.uiIxNum = m_uiIndex;
				uiOptIndex = m_uiIndex;
				m_scanOptInfo.bMustVerifyPath = TRUE;
				m_scanOptInfo.bDoNodeMatch = TRUE;
				m_scanOptInfo.bCanCompareOnKey = FALSE;
				m_scanOptInfo.szIxName [0] = 0;
				if (RC_OK( m_pDb->getNameTable( &pNameTable)))
				{
					FLMUINT	uiIxNameLen = sizeof( m_scanOptInfo.szIxName);

					if (RC_BAD( pNameTable->getFromTagTypeAndNum(
										m_pDb, ELM_INDEX_TAG,
										m_uiIndex, NULL,
										(char *)m_scanOptInfo.szIxName,
										&uiIxNameLen, NULL, NULL,
										NULL, NULL, TRUE)))
					{
						m_scanOptInfo.szIxName [0] = 0;
					}
				}
				if (pNameTable)
				{
					pNameTable->Release();
				}
			}
			m_scanOptInfo.ui64KeysRead = 0;
			m_scanOptInfo.ui64KeyHadDupDoc = 0;
			m_scanOptInfo.ui64KeysPassed = 0;
			m_scanOptInfo.ui64NodesRead = 0;
			m_scanOptInfo.ui64NodesTested = 0;
			m_scanOptInfo.ui64NodesPassed = 0;
			m_scanOptInfo.ui64DocsRead = 0;
			m_scanOptInfo.ui64DupDocsEliminated = 0;
			m_scanOptInfo.ui64NodesFailedValidation = 0;
			m_scanOptInfo.ui64DocsFailedValidation = 0;
			m_scanOptInfo.ui64DocsPassed = 0;
			m_pCurrOpt = &m_scanOptInfo;
			rc = newSource();
		}
		else
		{
			FLMBOOL	bHaveMultipleIndexes = FALSE;

			// Need to initialize all of the optimization information.

			m_pCurrContext = m_pQuery->pContext;
			useLeafContext( TRUE);
			do
			{
				m_pCurrOpt->ui64KeysRead = 0;
				m_pCurrOpt->ui64KeyHadDupDoc = 0;
				m_pCurrOpt->ui64KeysPassed = 0;
				m_pCurrOpt->ui64NodesRead = 0;
				m_pCurrOpt->ui64NodesTested = 0;
				m_pCurrOpt->ui64NodesPassed = 0;
				m_pCurrOpt->ui64DocsRead = 0;
				m_pCurrOpt->ui64DupDocsEliminated = 0;
				m_pCurrOpt->ui64NodesFailedValidation = 0;
				m_pCurrOpt->ui64DocsFailedValidation = 0;
				m_pCurrOpt->ui64DocsPassed = 0;
				
				// See if we are using an index.  Later, if it turns out
				// we are only using one index, we will see if it matches
				// the sort index.
				
				if (m_pSortIxd && !bHaveMultipleIndexes)
				{
					if (m_pCurrPred->pNodeSource)
					{
						FLMUINT	uiIndex;
						
						if (RC_BAD( rc = m_pCurrPred->pNodeSource->getIndex(
													(IF_Db *)m_pDb, &uiIndex,
													&bHaveMultipleIndexes)))
						{
							break;
						}
						if (uiIndex)
						{
							if (uiOptIndex == 0)
							{
								uiOptIndex = uiIndex;
							}
							else if (uiIndex != uiOptIndex)
							{
								bHaveMultipleIndexes = TRUE;
							}
						}
					}
					else if (m_pCurrOpt->uiIxNum)
					{
						if (uiOptIndex == 0)
						{
							uiOptIndex = m_pCurrOpt->uiIxNum;
						}
						else if (m_pCurrOpt->uiIxNum != uiOptIndex)
						{
							bHaveMultipleIndexes = TRUE;
						}
					}
				}
			}
			while (useNextPredicate());
			if (bHaveMultipleIndexes || RC_BAD( rc))
			{
				uiOptIndex = 0;
			}
		}
		
		// If we are sorting the results, see if the optimization
		// index already has the keys in the order we need.
		
		if (RC_OK( rc) && m_pSortIxd && uiOptIndex)
		{
			rc = checkSortIndex( uiOptIndex);
		}
	}

	if (m_pSortIxd || m_bPositioningEnabled)
	{
		// Need to eliminate dups in case there are multiple sort key
		// values in a document.
		// NOTE: When a sort key is specified, if a document has multiple
		// keys, we want them to appear sorted according to whatever is the
		// "lowest" sort key in the document.  If we create a result set,
		// this will always work.  However, if the entries are already "in order"
		// we have a small problem, because the order in which the document
		// is retrieved will then depend on the direction of traversal used
		// by the application.  By eliminating duplicate documents, the application
		// can get the expected behavior of having the document sort according
		// to its "lowest" key if it reads forward.  However, if the application
		// reads "backward" the document will be retrieved as if it were sorted
		// by its "higest" key.

		setDupHandling( TRUE);

		// If we have sort keys and the optimization index is not in
		// the same order as the sort index, we must create a result set.
		// The same is true if the application asked to enable positioning.

		if (m_bPositioningEnabled || !m_bEntriesAlreadyInOrder)
		{
			if (RC_OK( rc))
			{
				rc = createResultSet();
			}
		}
	}

	if ( RC_OK( rc))
	{
		m_bOptimized = TRUE;
	}

	return( rc);
}

/***************************************************************************
Desc:	Set node's current value to missing.  Also releases the stream
		associated with the value, if any.
***************************************************************************/
FINLINE void fqReleaseNodeValue(
	FQNODE *	pQNode
	)
{
	if ((pQNode->currVal.eValType == XFLM_BINARY_VAL ||
		  pQNode->currVal.eValType == XFLM_UTF8_VAL) &&
		 (pQNode->currVal.uiFlags & VAL_IS_STREAM) &&
		 pQNode->currVal.val.pIStream)
	{
		pQNode->currVal.uiFlags &= (~(VAL_IS_STREAM));
		pQNode->currVal.val.pIStream->Release();
		pQNode->currVal.val.pIStream = NULL;
	}
	if (pQNode->eNodeType != FLM_VALUE_NODE)
	{
		pQNode->currVal.eValType = XFLM_MISSING_VAL;
	}
}

/***************************************************************************
Desc:	Evaluate a simple operator.
***************************************************************************/
FSTATIC RCODE fqEvalOperator(
	FLMUINT	uiLanguage,
	FQNODE *	pQNode
	)
{
	RCODE				rc = NE_XFLM_OK;
	FQNODE *			pLeftOperand;
	FQNODE *			pRightOperand;
	XFlmBoolType	eLeftBool;
	XFlmBoolType	eRightBool;

	// Right now we are only able to do operator nodes.

	flmAssert( pQNode->eNodeType == FLM_OPERATOR_NODE);

	pLeftOperand = pQNode->pFirstChild;
	pRightOperand = pQNode->pLastChild;

	// See if either of this operator's operands have already
	// passed.  If so, we can skip the evaluation.

	if (pLeftOperand->currVal.eValType == XFLM_PASSING_VAL ||
		 pRightOperand->currVal.eValType == XFLM_PASSING_VAL)
	{
		pQNode->currVal.eValType = XFLM_BOOL_VAL;
		pQNode->currVal.val.eBool = XFLM_TRUE;
		goto Exit;
	}

	if (pQNode->nd.op.eOperator != XFLM_AND_OP &&
		 pQNode->nd.op.eOperator != XFLM_OR_OP)
	{
		// If the left operand is an XPATH node that is the node id or
		// document id, be sure to convert the right operand value to
		// a node id value.

		if (pLeftOperand->eNodeType == FLM_XPATH_NODE &&
			pLeftOperand->nd.pXPath->pLastComponent->eXPathAxis == META_AXIS)
		{
			if (RC_BAD( rc = fqGetNodeIdValue( &pRightOperand->currVal)))
			{
				goto Exit;
			}
		}

		// If the right operand is an XPATH node that is the node id or
		// document id, be sure to convert the left operand value to
		// a node id value.

		if (pRightOperand->eNodeType == FLM_XPATH_NODE &&
			pRightOperand->nd.pXPath->pLastComponent->eXPathAxis == META_AXIS)
		{
			if (RC_BAD( rc = fqGetNodeIdValue( &pLeftOperand->currVal)))
			{
				goto Exit;
			}
		}
	}

	pQNode->currVal.eValType = XFLM_MISSING_VAL;

	switch (pQNode->nd.op.eOperator)
	{
		case XFLM_AND_OP:
		case XFLM_OR_OP:
			eLeftBool = XFLM_UNKNOWN;
			eRightBool = XFLM_UNKNOWN;

			// Get the left operand

			if (pLeftOperand->eNodeType == FLM_OPERATOR_NODE)
			{

				// This operator may not have been evaluated because of missing
				// XPATH values in one or both operands, in which case
				// its state will be XFLM_MISSING_VALUE.  If it was evaluated,
				// its state should show a boolean value.

				if (pLeftOperand->currVal.eValType == XFLM_MISSING_VAL)
				{
					eLeftBool = (pLeftOperand->bNotted ? XFLM_TRUE : XFLM_FALSE);
				}
				else
				{
					flmAssert( pLeftOperand->currVal.eValType == XFLM_BOOL_VAL);
					eLeftBool = pLeftOperand->currVal.val.eBool;
				}
			}
			else if (pLeftOperand->eNodeType == FLM_XPATH_NODE)
			{
				if (!pLeftOperand->bNotted)
				{
					eLeftBool = (pLeftOperand->currVal.eValType != XFLM_MISSING_VAL)
									  ? XFLM_TRUE
									  : XFLM_FALSE;
				}
				else
				{
					eLeftBool = (pLeftOperand->currVal.eValType != XFLM_MISSING_VAL)
									  ? XFLM_FALSE
									  : XFLM_TRUE;
				}
			}
			else if (pLeftOperand->eNodeType == FLM_VALUE_NODE)
			{
				flmAssert( pLeftOperand->currVal.eValType == XFLM_BOOL_VAL);
				eLeftBool = pLeftOperand->currVal.val.eBool;
			}
			else
			{
				flmAssert( pLeftOperand->eNodeType == FLM_FUNCTION_NODE);
				if (!pLeftOperand->bNotted)
				{
					eLeftBool = fqTestValue( pLeftOperand) ? XFLM_TRUE : XFLM_FALSE;
				}
				else
				{
					eLeftBool = fqTestValue( pLeftOperand) ? XFLM_FALSE : XFLM_TRUE;
				}
			}

			// Get the right operand

			if ( pRightOperand->eNodeType == FLM_OPERATOR_NODE)
			{

				// This operator may not have been evaluated because of missing
				// XPATH values in one or both operands, in which case
				// its state will be XFLM_MISSING_VALUE.  If it was evaluated,
				// its state should show a boolean value.

				if (pRightOperand->currVal.eValType == XFLM_MISSING_VAL)
				{
					eRightBool = (pRightOperand->bNotted ? XFLM_TRUE : XFLM_FALSE);
				}
				else
				{
					flmAssert( pRightOperand->currVal.eValType == XFLM_BOOL_VAL);
					eRightBool = pRightOperand->currVal.val.eBool;
				}
			}
			else if (pRightOperand->eNodeType == FLM_XPATH_NODE)
			{
				if (!pRightOperand->bNotted)
				{
					eRightBool = (pRightOperand->currVal.eValType != XFLM_MISSING_VAL)
									  ? XFLM_TRUE
									  : XFLM_FALSE;
				}
				else
				{
					eRightBool = (pRightOperand->currVal.eValType != XFLM_MISSING_VAL)
									  ? XFLM_FALSE
									  : XFLM_TRUE;
				}
			}
			else if (pRightOperand->eNodeType == FLM_VALUE_NODE)
			{
				flmAssert( pRightOperand->currVal.eValType == XFLM_BOOL_VAL);
				eRightBool = pRightOperand->currVal.val.eBool;
			}
			else
			{
				flmAssert( pRightOperand->eNodeType == FLM_FUNCTION_NODE);
				if (!pRightOperand->bNotted)
				{
					eRightBool = fqTestValue( pRightOperand) ? XFLM_TRUE : XFLM_FALSE;
				}
				else
				{
					eRightBool = fqTestValue( pRightOperand) ? XFLM_FALSE : XFLM_TRUE;
				}
			}

			// Calculate the answer

			pQNode->currVal.eValType = XFLM_BOOL_VAL;
			if (pQNode->nd.op.eOperator == XFLM_AND_OP)
			{
				if (eLeftBool == XFLM_FALSE || eRightBool == XFLM_FALSE)
				{
					pQNode->currVal.val.eBool = XFLM_FALSE;
				}
				else if (eLeftBool == XFLM_UNKNOWN || eRightBool == XFLM_UNKNOWN)
				{
					pQNode->currVal.val.eBool = XFLM_UNKNOWN;
				}
				else
				{

					// Both have to be XFLM_TRUE at this point

					pQNode->currVal.val.eBool = XFLM_TRUE;
				}
			}
			else // pQNode->nd.op.eOperator == XFLM_OR_OP
			{
				if (eLeftBool == XFLM_TRUE || eRightBool == XFLM_TRUE)
				{
					pQNode->currVal.val.eBool = XFLM_TRUE;
				}
				else if (eLeftBool == XFLM_UNKNOWN || eRightBool == XFLM_UNKNOWN)
				{
					pQNode->currVal.val.eBool = XFLM_UNKNOWN;
				}
				else
				{

					// Both have to be XFLM_FALSE at this point

					pQNode->currVal.val.eBool = XFLM_FALSE;
				}
			}
			break;

		case XFLM_EQ_OP:
		case XFLM_APPROX_EQ_OP:
		case XFLM_NE_OP:
		case XFLM_LT_OP:
		case XFLM_LE_OP:
		case XFLM_GT_OP:
		case XFLM_GE_OP:
			pQNode->currVal.eValType = XFLM_BOOL_VAL;
			if (RC_BAD( rc = fqCompareOperands( uiLanguage,
										&pLeftOperand->currVal,
										&pRightOperand->currVal,
										pQNode->nd.op.eOperator,
										pQNode->nd.op.uiCompareRules,
										pQNode->nd.op.pOpComparer,
										pQNode->bNotted,
										&pQNode->currVal.val.eBool)))
			{
				goto Exit;
			}
			break;

		case XFLM_BITAND_OP:
		case XFLM_BITOR_OP:
		case XFLM_BITXOR_OP:
		case XFLM_MULT_OP:
		case XFLM_DIV_OP:
		case XFLM_MOD_OP:
		case XFLM_PLUS_OP:
		case XFLM_MINUS_OP:
		case XFLM_NEG_OP:
			if (RC_BAD( rc = fqArithmeticOperator( &pLeftOperand->currVal,
										&pRightOperand->currVal,
										pQNode->nd.op.eOperator,
										&pQNode->currVal)))
			{
				goto Exit;
			}
			break;
		
		default:
			break;
	}

Exit:

	// Need to release node values in case we are holding on
	// to an IStream.

	if (pLeftOperand)
	{
		fqReleaseNodeValue( pLeftOperand);
	}
	if (pRightOperand)
	{
		fqReleaseNodeValue( pRightOperand);
	}

	return( rc);
}

/***************************************************************************
Desc:	Get the next value from an XPATH node.
***************************************************************************/
FSTATIC void fqResetIterator(
	FQNODE *		pQNode,
	FLMBOOL		bFullRelease,
	FLMBOOL		bUseKeyNodes
	)
{
	FXPATH *				pXPath;
	XPATH_COMPONENT *	pXPathComponent;

	// Node better be an XPATH node.

	flmAssert( pQNode->eNodeType == FLM_XPATH_NODE);
	pXPath = pQNode->nd.pXPath;
	if (bFullRelease)
	{
		pXPath->bIsSource = FALSE;
		pXPath->pSourceComponent = NULL;
		pXPath->bHavePassingNode = FALSE;
	}
	pXPathComponent = pXPath->pLastComponent;
	while (pXPathComponent)
	{
		if (bFullRelease)
		{
			pXPathComponent->bIsSource = FALSE;
			pXPathComponent->pExprXPathSource = NULL;
			pXPathComponent->pOptPred = NULL;
			if (pXPathComponent->pKeyNode)
			{
				pXPathComponent->pKeyNode->Release();
				pXPathComponent->pKeyNode = NULL;
			}
		}
		else if (pXPathComponent->bIsSource && bUseKeyNodes)
		{
			break;
		}
		if (pXPathComponent->pCurrNode)
		{
			pXPathComponent->pCurrNode->Release();
			pXPathComponent->pCurrNode = NULL;
		}

		if (bFullRelease && pXPathComponent->pExpr)
		{
			fqReleaseQueryExpr( pXPathComponent->pExpr);
		}

		pXPathComponent = pXPathComponent->pPrev;
	}
	pXPath->bGettingNodes = FALSE;
}

/***************************************************************************
Desc:	Get the first, last, next or previous node from a node source object.
***************************************************************************/
RCODE F_Query::getNodeSourceNode(
	FLMBOOL					bForward,
	IF_QueryNodeSource *	pNodeSource,
	IF_DOMNode *			pContextNode,
	IF_DOMNode **			ppCurrNode
	)
{
	RCODE		rc = NE_XFLM_OK;
	FLMUINT	uiTimeLimit = m_uiTimeLimit;
	FLMUINT	uiCurrTime;
	FLMUINT	uiElapsedTime;
	
	// Determine the timeout limit that is left.

	if (uiTimeLimit)
	{
		uiCurrTime = FLM_GET_TIMER();
		uiElapsedTime = FLM_ELAPSED_TIME( uiCurrTime, m_uiStartTime);
		if (uiElapsedTime >= m_uiTimeLimit)
		{
			rc = RC_SET( NE_XFLM_TIMEOUT);
			goto Exit;
		}
		else
		{
			uiTimeLimit = FLM_TIMER_UNITS_TO_MILLI( (m_uiTimeLimit - uiElapsedTime));
				
			// Always give at least one milli-second.
				
			if (!uiTimeLimit)
			{
				uiTimeLimit = 1;
			}
		}
	}
	
	if (*ppCurrNode)
	{

		// Get next or previous node.

		rc = (RCODE)(bForward
						 ? pNodeSource->getNext( (IF_Db *)m_pDb, pContextNode, ppCurrNode,
						 						uiTimeLimit, m_pQueryStatus)
						 : pNodeSource->getPrev( (IF_Db *)m_pDb, pContextNode, ppCurrNode,
						 						uiTimeLimit, m_pQueryStatus));
	}
	else
	{
		
		// Get first or last node.
		
		rc = (RCODE)(bForward
						 ? pNodeSource->getFirst( (IF_Db *)m_pDb, pContextNode, ppCurrNode,
						 						uiTimeLimit, m_pQueryStatus)
						 : pNodeSource->getLast( (IF_Db *)m_pDb, pContextNode, ppCurrNode,
						 						uiTimeLimit, m_pQueryStatus));
	}
	if (RC_BAD( rc))
	{
		if (rc == NE_XFLM_EOF_HIT ||
		 	 rc == NE_XFLM_BOF_HIT)
		{
			rc = NE_XFLM_OK;
			if (*ppCurrNode)
			{
				(*ppCurrNode)->Release();
				*ppCurrNode = NULL;
			}
		}
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get the next node from a ROOT_AXIS.
***************************************************************************/
RCODE F_Query::getRootAxisNode(
	IF_DOMNode **	ppCurrNode
	)
{
	RCODE	rc = NE_XFLM_OK;

	// Better be an element we are searching for

	if (m_pCurrDoc->getNodeType() == DOCUMENT_NODE)
	{
		if (RC_BAD( rc = m_pCurrDoc->getFirstChild( m_pDb, ppCurrNode)))
		{
			if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				rc = NE_XFLM_OK;
			}
			goto Exit;
		}
		if (RC_BAD( rc = incrNodesRead()))
		{
			goto Exit;
		}

		// Search for an element node - it will be the root
		// of the document

		while ((*ppCurrNode)->getNodeType() != ELEMENT_NODE)
		{
			if (RC_BAD( rc = (*ppCurrNode)->getNextSibling( m_pDb, ppCurrNode)))
			{
				if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					(*ppCurrNode)->Release();
					*ppCurrNode = NULL;
					rc = NE_XFLM_OK;
				}
				goto Exit;
			}
			if (RC_BAD( rc = incrNodesRead()))
			{
				goto Exit;
			}
		}
	}
	else
	{
		*ppCurrNode = m_pCurrDoc;
		(*ppCurrNode)->AddRef();
		if ((*ppCurrNode)->getNodeType() != ELEMENT_NODE)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get the next/previous node in a document
***************************************************************************/
RCODE F_Query::walkDocument(
	FLMBOOL			bForward,
	FLMBOOL			bWalkAttributes,
	FLMUINT			uiAttrNameId,
	IF_DOMNode **	ppCurrNode
	)
{
	RCODE	rc = NE_XFLM_OK;

	if (!(*ppCurrNode))
	{
		*ppCurrNode = m_pCurrDoc;
		(*ppCurrNode)->AddRef();
		goto Exit;
	}

	if ((*ppCurrNode)->getNodeType() == ATTRIBUTE_NODE)
	{
		if (uiAttrNameId)
		{
			
			// This attribute node should be the node that we already processed.
			// We simply want to go back to its parent node.  No need to
			// walk through all of the sibling attributes.
			
			rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
		}
		else
		{
			rc = bForward
							  ? (*ppCurrNode)->getNextSibling( m_pDb,
														ppCurrNode)
							  : (*ppCurrNode)->getPreviousSibling( m_pDb,
														ppCurrNode);
		}
		if (RC_OK( rc))
		{
			if (RC_BAD( rc = incrNodesRead()))
			{
				goto Exit;
			}
		}
		else if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			goto Exit;
		}

		// Need to get the node's encompassing element node and
		// process it now.

		else if (RC_BAD( rc = (*ppCurrNode)->getParentNode( m_pDb, ppCurrNode)))
		{
			if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			}
			goto Exit;
		}

		if (RC_BAD( rc = incrNodesRead()))
		{
			goto Exit;
		}
	}

	// See if the node has a child

	else if (RC_OK( rc = bForward
							? (*ppCurrNode)->getFirstChild( m_pDb, ppCurrNode)
							: (*ppCurrNode)->getLastChild( m_pDb, ppCurrNode)))
	{
		if (RC_BAD( rc = incrNodesRead()))
		{
			goto Exit;
		}

Walk_To_Attr_Nodes:

		if (bWalkAttributes && (*ppCurrNode)->getNodeType() == ELEMENT_NODE)
		{
			if( uiAttrNameId)
			{
				rc = (*ppCurrNode)->getAttribute( m_pDb, uiAttrNameId, ppCurrNode);
			}
			else
			{
				rc = bForward
						 ? (*ppCurrNode)->getFirstAttribute( m_pDb, ppCurrNode)
						 : (*ppCurrNode)->getLastAttribute( m_pDb, ppCurrNode);
			}
			if (RC_OK( rc))
			{
				if (RC_BAD( rc = incrNodesRead()))
				{
					goto Exit;
				}
			}
			else if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				goto Exit;
			}
			else
			{
				rc = NE_XFLM_OK;
			}
		}
	}
	else if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
	{
		goto Exit;
	}
	else
	{

		// Go up tree until we find a sibling.

		for (;;)
		{
			if (RC_OK( rc = bForward
								 ? (*ppCurrNode)->getNextSibling( m_pDb,
														ppCurrNode)
								 : (*ppCurrNode)->getPreviousSibling( m_pDb,
														ppCurrNode)))
			{
				if (RC_BAD( rc = incrNodesRead()))
				{
					goto Exit;
				}
				goto Walk_To_Attr_Nodes;
			}
			else if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				goto Exit;
			}
			else if (RC_BAD( rc = (*ppCurrNode)->getParentNode( m_pDb,
												ppCurrNode)))
			{
				if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					rc = NE_XFLM_OK;
					(*ppCurrNode)->Release();
					*ppCurrNode = NULL;
					break;
				}
				goto Exit;
			}
			else
			{
				if (RC_BAD( rc = incrNodesRead()))
				{
					goto Exit;
				}
			}
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get the next or previous child node.
***************************************************************************/
RCODE F_Query::getChildAxisNode(
	FLMBOOL			bForward,
	IF_DOMNode *	pContextNode,
	FLMUINT			uiChildNameId,
	IF_DOMNode **	ppCurrNode
	)
{
	RCODE	rc = NE_XFLM_OK;

	if (!pContextNode)
	{
		if (RC_BAD( rc = walkDocument( bForward, FALSE, 0, ppCurrNode)))
		{
			goto Exit;
		}
	}
	else
	{
		if (*ppCurrNode)
		{
			// If name id is non-zero, the caller should have prevented us
			// from coming back in here.
			
			flmAssert( !uiChildNameId);

			// Get next or previous sibling - should still be a child
			// of our context node, whatever it was.

			rc = (RCODE)(bForward
							 ? (*ppCurrNode)->getNextSibling( m_pDb, ppCurrNode)
							 : (*ppCurrNode)->getPreviousSibling( m_pDb, ppCurrNode));
		}
		else if (uiChildNameId)
		{
			*ppCurrNode = pContextNode;
			(*ppCurrNode)->AddRef();
			rc = (*ppCurrNode)->getChildElement( m_pDb, uiChildNameId, ppCurrNode);
		}
		else
		{
			*ppCurrNode = pContextNode;
			(*ppCurrNode)->AddRef();
			rc = (RCODE)(bForward
							 ? (*ppCurrNode)->getFirstChild( m_pDb, ppCurrNode)
							 : (*ppCurrNode)->getLastChild( m_pDb, ppCurrNode));
		}
		if (RC_BAD( rc))
		{
			if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				rc = NE_XFLM_OK;
				(*ppCurrNode)->Release();
				*ppCurrNode = NULL;
			}
			goto Exit;
		}
		else
		{
			if (RC_BAD( rc = incrNodesRead()))
			{
				goto Exit;
			}
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get the next node from a PARENT_AXIS.
***************************************************************************/
RCODE F_Query::getParentAxisNode(
	FLMBOOL			bForward,
	IF_DOMNode *	pContextNode,
	IF_DOMNode **	ppCurrNode
	)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT64	ui64NodeId;

	if (!pContextNode)
	{
		for (;;)
		{
			if (RC_BAD( rc = walkDocument( bForward, FALSE, 0, ppCurrNode)))
			{
				goto Exit;
			}
			if (!(*ppCurrNode))
			{
				break;
			}

			// This node must be a parent of another node,
			// which means it must have at least one child node.

			if (RC_BAD( rc = (*ppCurrNode)->getFirstChildId( m_pDb,
												&ui64NodeId)))
			{
				goto Exit;
			}
			if (ui64NodeId)
			{
				break;
			}
		}
	}
	else
	{

		// PARENT_AXIS always starts from the context node - it
		// really doesn't matter what the last node was.

		if (RC_BAD( rc = pContextNode->getParentNode( m_pDb, ppCurrNode)))
		{
			if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				if (*ppCurrNode)
				{
					(*ppCurrNode)->Release();
					*ppCurrNode = NULL;
				}
				rc = NE_XFLM_OK;
			}
			goto Exit;
		}
		else
		{
			if (RC_BAD( rc = incrNodesRead()))
			{
				goto Exit;
			}
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get the next ancestor node.
***************************************************************************/
RCODE F_Query::getAncestorAxisNode(
	FLMBOOL				bForward,
	FLMBOOL				bIncludeSelf,
	IF_DOMNode *		pContextNode,
	IF_DOMNode **		ppCurrNode)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT64			ui64ContextNodeId;
	FLMUINT64			ui64NodeId;
	FLMUINT64			ui64ParentId;
	FLMUINT64			ui64Tmp;
	FLMUINT				uiContextAttrNameId;
	FLMUINT				uiNameId;
	FLMUINT				uiTmp;

	if (!pContextNode)
	{
		for (;;)
		{
			if (RC_BAD( rc = walkDocument( bForward, FALSE, 0, ppCurrNode)))
			{
				goto Exit;
			}
			
			if (!(*ppCurrNode))
			{
				break;
			}

			// If we are including self, whatever node we get to matches
			// the axis.

			if (bIncludeSelf)
			{
				break;
			}

			// This node must be an ancestor of another node,
			// which means it must have at least one child node.

			if (RC_BAD( rc = (*ppCurrNode)->getFirstChildId( m_pDb,
												&ui64NodeId)))
			{
				goto Exit;
			}
			if (ui64NodeId)
			{
				break;
			}
		}
	}
	else
	{
		if( RC_BAD( rc = ((F_DOMNode *)pContextNode)->getNodeId( m_pDb, 
			&ui64ContextNodeId, &uiContextAttrNameId)))
		{
			goto Exit;
		}
		
		if (*ppCurrNode)
		{
			if (bForward)
			{
				if (RC_BAD( rc = (*ppCurrNode)->getParentNode( m_pDb, ppCurrNode)))
				{
					if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
					{
						rc = NE_XFLM_OK;
						(*ppCurrNode)->Release();
						*ppCurrNode = NULL;
					}
					goto Exit;
				}
				else
				{
					if (RC_BAD( rc = incrNodesRead()))
					{
						goto Exit;
					}
				}
			}
			else
			{
				if( RC_BAD( rc = ((F_DOMNode *)(*ppCurrNode))->getNodeId( m_pDb, 
					&ui64NodeId, &uiNameId)))
				{
					goto Exit;
				}
				
				// If the current node is the context node, we
				// have no further to go.

				if( ui64NodeId == ui64ContextNodeId && 
					 uiContextAttrNameId == uiNameId)
				{
					(*ppCurrNode)->Release();
					*ppCurrNode = NULL;
					goto Exit;
				}

				// Start from the context node, and go to just before
				// the node we are on.  If we don't hit it, this node
				// is not an ancestor of our context node.

				(*ppCurrNode)->Release();
				*ppCurrNode = pContextNode;
				(*ppCurrNode)->AddRef();

				for (;;)
				{
					if (RC_BAD( rc = (*ppCurrNode)->getParentId( 
						m_pDb, &ui64ParentId)))
					{
						goto Exit;
					}
					if (ui64ParentId == ui64NodeId)
					{
						if (*ppCurrNode == pContextNode && !bIncludeSelf)
						{
							(*ppCurrNode)->Release();
							*ppCurrNode = NULL;
						}
						break;
					}

					if (RC_BAD( rc = (*ppCurrNode)->getParentNode( 
						m_pDb, ppCurrNode)))
					{
						if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
						{
							rc = NE_XFLM_OK;
							(*ppCurrNode)->Release();
							*ppCurrNode = NULL;
						}
						goto Exit;
					}
					else
					{
						if (RC_BAD( rc = incrNodesRead()))
						{
							goto Exit;
						}
					}
				}
			}
		}
		else
		{
			if (bForward)
			{
				*ppCurrNode = pContextNode;
				(*ppCurrNode)->AddRef();

				// If including context node, we have what we want.
				// Otherwise, we have to go to the parent.

				if (bIncludeSelf)
				{
					goto Exit;
				}
				if (RC_BAD( rc = (*ppCurrNode)->getParentNode( m_pDb, ppCurrNode)))
				{
					if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
					{
						rc = NE_XFLM_OK;
						(*ppCurrNode)->Release();
						*ppCurrNode = NULL;
					}
					goto Exit;
				}
				else
				{
					if (RC_BAD( rc = incrNodesRead()))
					{
						goto Exit;
					}
				}
			}
			else
			{
				// Start at document root.  Return it if it is not the
				// context node, or if we are including self.
				
				if( RC_BAD( rc = ((F_DOMNode *)m_pCurrDoc)->getNodeId( m_pDb, 
					&ui64Tmp, &uiTmp)))
				{
					goto Exit;
				}

				if( ui64Tmp != ui64ContextNodeId || uiTmp != uiContextAttrNameId ||
					 bIncludeSelf)
				{
					*ppCurrNode = m_pCurrDoc;
					(*ppCurrNode)->AddRef();
				}
			}
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get the next node from a DESCENDANT_AXIS or a DESCENDANT_OR_SELF_AXIS.
***************************************************************************/
RCODE F_Query::getDescendantAxisNode(
	FLMBOOL				bForward,
	FLMBOOL				bIncludeSelf,
	IF_DOMNode *		pContextNode,
	IF_DOMNode **		ppCurrNode)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT64			ui64NodeId;
	FLMUINT64			ui64ContextNodeId;
	FLMUINT64			ui64Tmp;
	FLMUINT				uiContextAttrNameId;
	FLMUINT				uiTmp;

	if (!pContextNode)
	{
		for (;;)
		{
			if (RC_BAD( rc = walkDocument( bForward, FALSE, 0, ppCurrNode)))
			{
				goto Exit;
			}
			if (!(*ppCurrNode))
			{
				break;
			}

			// If we are including self, whatever node we get to matches
			// the axis.

			if (bIncludeSelf)
			{
				break;
			}

			// This node must be a descendant of some
			// other node, so it must have a parent.

			if (RC_BAD( rc = (*ppCurrNode)->getParentId( m_pDb, &ui64NodeId)))
			{
				goto Exit;
			}
			if (ui64NodeId)
			{
				break;
			}
		}
	}
	else
	{
		if( RC_BAD( rc = ((F_DOMNode *)pContextNode)->getNodeId( m_pDb, 
			&ui64ContextNodeId, &uiContextAttrNameId)))
		{
			goto Exit;
		}
		
		if (bForward)
		{
			if (!(*ppCurrNode))
			{
				*ppCurrNode = pContextNode;
				(*ppCurrNode)->AddRef();
				if (bIncludeSelf)
				{
					goto Exit;
				}
			}

			// See if the node has a child

			if (RC_OK( rc = (*ppCurrNode)->getFirstChild( m_pDb, ppCurrNode)))
			{
				if (RC_BAD( rc = incrNodesRead()))
				{
					goto Exit;
				}
			}
			else if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				goto Exit;
			}
			else
			{
				rc = NE_XFLM_OK;

				// Go up tree until we find a sibling. - Don't
				// go up past the context node.

				for (;;)
				{

					// If we are on the context node, we are done.
					
					if( RC_BAD( rc = ((F_DOMNode *)(*ppCurrNode))->getNodeId( m_pDb, 
						&ui64Tmp, &uiTmp)))
					{
						goto Exit;
					}

					if( ui64Tmp == ui64ContextNodeId && uiTmp == uiContextAttrNameId)
					{
						(*ppCurrNode)->Release();
						*ppCurrNode = NULL;
						goto Exit;
					}

					if (RC_OK( rc = (*ppCurrNode)->getNextSibling( m_pDb,
													ppCurrNode)))
					{
						if (RC_BAD( rc = incrNodesRead()))
						{
							goto Exit;
						}
						break;
					}
					else if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
					{
						goto Exit;
					}
					else if (RC_BAD( rc = (*ppCurrNode)->getParentNode( m_pDb,
														ppCurrNode)))
					{
						if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
						{
							rc = NE_XFLM_OK;
							(*ppCurrNode)->Release();
							*ppCurrNode = NULL;
						}
						goto Exit;
					}
					else
					{
						if (RC_BAD( rc = incrNodesRead()))
						{
							goto Exit;
						}
					}
				}
			}
		}
		else
		{
			if (*ppCurrNode)
			{

				// If we are going backwards and we are on the context node
				// there are no more nodes to get.
				
				if( RC_BAD( rc = ((F_DOMNode *)(*ppCurrNode))->getNodeId( m_pDb,
					&ui64Tmp, &uiTmp)))
				{
					goto Exit;
				}

				if( ui64Tmp == ui64ContextNodeId && uiTmp == uiContextAttrNameId)
				{
					(*ppCurrNode)->Release();
					*ppCurrNode = NULL;
					goto Exit;
				}

				// See if the node has a previous sibling

				if (RC_BAD( rc = (*ppCurrNode)->getPreviousSibling( m_pDb, ppCurrNode)))
				{
					if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
					{
						goto Exit;
					}
					if (RC_BAD( rc = (*ppCurrNode)->getParentNode( m_pDb, ppCurrNode)))
					{
						if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
						{
							rc = NE_XFLM_OK;
							(*ppCurrNode)->Release();
							*ppCurrNode = NULL;
						}
						goto Exit;
					}
					else
					{
						if (RC_BAD( rc = incrNodesRead()))
						{
							goto Exit;
						}
					}
				}
				else
				{
					if (RC_BAD( rc = incrNodesRead()))
					{
						goto Exit;
					}

					// Go down to the last child of the previous sibling
					// that was found.

Get_Last_Child:

					for (;;)
					{
						if (RC_BAD( rc = (*ppCurrNode)->getLastChild( m_pDb, ppCurrNode)))
						{
							if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
							{
								goto Exit;
							}

							// We are positioned on the rightmost child now.

							rc = NE_XFLM_OK;
							break;
						}
						else
						{
							if (RC_BAD( rc = incrNodesRead()))
							{
								goto Exit;
							}
						}
					}
				}
			}
			else
			{
				*ppCurrNode = pContextNode;
				(*ppCurrNode)->AddRef();

				// Go down to the last child of the context node.

				goto Get_Last_Child;
			}

			// If we arrive at the context node, we are done, unless
			// we are including self.
			
			if( RC_BAD( rc = ((F_DOMNode *)(*ppCurrNode))->getNodeId( m_pDb, 
				&ui64Tmp, &uiTmp)))
			{
				goto Exit;
			}

			if( ui64Tmp == ui64ContextNodeId && uiTmp == uiContextAttrNameId &&
				 !bIncludeSelf)
			{
				(*ppCurrNode)->Release();
				*ppCurrNode = NULL;
			}
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get the next node from a PRECEDING_SIBLING_AXIS or
		FOLLOWING_SIBLING_AXIS.
***************************************************************************/
RCODE F_Query::getSibAxisNode(
	FLMBOOL				bForward,
	FLMBOOL				bPrevSibAxis,
	IF_DOMNode *		pContextNode,
	IF_DOMNode **		ppCurrNode)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT64			ui64NodeId;
	FLMUINT64			ui64CurrNodeId;
	FLMUINT64			ui64ContextNodeId;
	FLMUINT				uiCurrNameId;
	FLMUINT				uiContextAttrNameId;

	if (!pContextNode)
	{
		for (;;)
		{
			if (RC_BAD( rc = walkDocument( bForward, FALSE, 0, ppCurrNode)))
			{
				goto Exit;
			}
			
			if (!(*ppCurrNode))
			{
				break;
			}

			// This node must be the next or previous sibling to some
			// node - which means it must have a previous or next sibling.

			rc = (RCODE)(bPrevSibAxis
							 ? (*ppCurrNode)->getNextSibId( m_pDb,
														&ui64NodeId)
							 : (*ppCurrNode)->getPrevSibId( m_pDb,
														&ui64NodeId));
			if (RC_BAD( rc))
			{
				goto Exit;
			}
			if (ui64NodeId)
			{
				break;
			}
		}
	}
	else
	{
		if (bForward)
		{
			if (!(*ppCurrNode))
			{
				*ppCurrNode = pContextNode;
				(*ppCurrNode)->AddRef();
			}
			rc = (RCODE)(bPrevSibAxis
							 ? (*ppCurrNode)->getPreviousSibling( m_pDb, ppCurrNode)
							 : (*ppCurrNode)->getNextSibling( m_pDb, ppCurrNode));
			if (RC_BAD( rc))
			{
				if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					rc = NE_XFLM_OK;
					(*ppCurrNode)->Release();
					*ppCurrNode = NULL;
				}
				goto Exit;
			}
			if (RC_BAD( rc = incrNodesRead()))
			{
				goto Exit;
			}
		}
		else
		{
			if (*ppCurrNode)
			{
				rc = (RCODE)(bPrevSibAxis
								 ? (*ppCurrNode)->getNextSibling( m_pDb, ppCurrNode)
								 : (*ppCurrNode)->getPreviousSibling( m_pDb, ppCurrNode));
				if (RC_BAD( rc))
				{
					if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
					{
						rc = NE_XFLM_OK;
						(*ppCurrNode)->Release();
						*ppCurrNode = NULL;
					}
					goto Exit;
				}
				if (RC_BAD( rc = incrNodesRead()))
				{
					goto Exit;
				}
			}
			else
			{
				FLMBOOL	bAttr;

				*ppCurrNode = pContextNode;
				(*ppCurrNode)->AddRef();

				bAttr = (*ppCurrNode)->getNodeType() == ATTRIBUTE_NODE
						  ? TRUE
						  : FALSE;

				// Go to the parent and get either the first or
				// last child, depending on the axis.  That is
				// where we need to start from.

				if (RC_BAD( rc = (*ppCurrNode)->getParentNode( m_pDb, ppCurrNode)))
				{
					if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
					{

						// A node without a parent should not have any siblings!

						rc = NE_XFLM_OK;
						(*ppCurrNode)->Release();
						*ppCurrNode = NULL;
					}
					goto Exit;
				}
				if (RC_BAD( rc = incrNodesRead()))
				{
					goto Exit;
				}

				// Get the node's first or last child.

				if (!bAttr)
				{
					rc = (RCODE)(bPrevSibAxis
									 ? (*ppCurrNode)->getFirstChild( m_pDb, ppCurrNode)
									 : (*ppCurrNode)->getLastChild( m_pDb, ppCurrNode));
				}
				else
				{
					rc = (RCODE)(bPrevSibAxis
									 ? (*ppCurrNode)->getFirstAttribute( m_pDb, ppCurrNode)
									 : (*ppCurrNode)->getLastAttribute( m_pDb, ppCurrNode));
				}
				if (RC_BAD( rc))
				{
					if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
					{
						rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
					}
					goto Exit;
				}
				
				if (RC_BAD( rc = incrNodesRead()))
				{
					goto Exit;
				}
			}

			// If we landed on the context node, we are done - there will be
			// no more siblings to get.
			
			if( RC_BAD( rc = ((F_DOMNode *)(*ppCurrNode))->getNodeId( m_pDb,
				&ui64CurrNodeId, &uiCurrNameId)))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = ((F_DOMNode *)pContextNode)->getNodeId( m_pDb, 
				&ui64ContextNodeId, &uiContextAttrNameId)))
			{
				goto Exit;
			}

			if( ui64CurrNodeId == ui64ContextNodeId && 
				uiCurrNameId == uiContextAttrNameId)
			{
				(*ppCurrNode)->Release();
				*ppCurrNode = NULL;
			}
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get the next node from a PRECEDING_AXIS or
		FOLLOWING_AXIS.
***************************************************************************/
RCODE F_Query::getPrevOrAfterAxisNode(
	FLMBOOL				bForward,
	FLMBOOL				bPrevAxis,
	IF_DOMNode *		pContextNode,
	IF_DOMNode **		ppCurrNode)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiChildCnt;
	IF_DOMNode *		pParentNode = NULL;
	IF_DOMNode *		pSibNode = NULL;
	IF_DOMNode *		pHighestAncestorWithSib = NULL;
	FLMUINT64			ui64NodeId;
	FLMUINT64			ui64ContextId;
	FLMUINT64			ui64Tmp;

	if (!pContextNode)
	{
		for (;;)
		{
			if (RC_BAD( rc = walkDocument( bForward, FALSE, 0, ppCurrNode)))
			{
				goto Exit;
			}
			if (!(*ppCurrNode))
			{
				break;
			}

			if (pParentNode)
			{
				pParentNode->Release();
			}
			pParentNode = *ppCurrNode;
			pParentNode->AddRef();

			// This node must be previous to some other node if the
			// axis is PRECEDING_AXIS (bPrevAxis == TRUE) or after some
			// other node if the axis is FOLLOWING_AXIS
			// (bPrevAxis == FALSE).  This does not count descendants
			// or ancestors or attribute nodes.

			for (;;)
			{
				rc = (RCODE)(bPrevAxis
								 ? pParentNode->getNextSibId( m_pDb, &ui64NodeId)
								 : pParentNode->getPrevSibId( m_pDb, &ui64NodeId));
				if (RC_BAD( rc))
				{
					goto Exit;
				}
				if (ui64NodeId)
				{
					goto Exit;
				}
				if (RC_BAD( rc = pParentNode->getParentNode( m_pDb,
												&pParentNode)))
				{
					if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
					{
						rc = NE_XFLM_OK;
						break;
					}
					goto Exit;
				}
				else
				{
					if (RC_BAD( rc = incrNodesRead()))
					{
						goto Exit;
					}
				}
			}
		}
	}
	else
	{
		if( RC_BAD( rc = pContextNode->getNodeId( m_pDb, &ui64ContextId)))
		{
			goto Exit;
		}
		
		// Context node better not be an attribute node.  That is illegal.
		// To be nice here, we will change the context node to be the
		// attribute's encompassing element node.

		if (pContextNode->getNodeType() == ATTRIBUTE_NODE)
		{
			if (RC_BAD( rc = pContextNode->getParentNode( m_pDb, &pContextNode)))
			{
				if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
				}
				goto Exit;
			}
		}

		if (bForward)
		{
			if (*ppCurrNode)
			{

				// Go to child of current node

				rc = (RCODE)(bPrevAxis
								 ? (*ppCurrNode)->getLastChild( m_pDb, ppCurrNode)
								 : (*ppCurrNode)->getFirstChild( m_pDb, ppCurrNode));
				if (RC_OK( rc))
				{
					rc = incrNodesRead();
					goto Exit;
				}
				else if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					goto Exit;
				}
			}
			else
			{
				*ppCurrNode = pContextNode;
				(*ppCurrNode)->AddRef();
			}

			for (;;)
			{

				rc = (RCODE)(bPrevAxis
								 ? (*ppCurrNode)->getPreviousSibling( m_pDb, ppCurrNode)
								 : (*ppCurrNode)->getNextSibling( m_pDb, ppCurrNode));
				if (RC_OK( rc))
				{
					rc = incrNodesRead();
					goto Exit;
				}
				else if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					goto Exit;
				}

				// Go to the parent node

				if (RC_BAD( rc = (*ppCurrNode)->getParentNode( m_pDb, ppCurrNode)))
				{
					if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
					{
						(*ppCurrNode)->Release();
						*ppCurrNode = NULL;
						rc = NE_XFLM_OK;
					}
					goto Exit;
				}
				else
				{
					if (RC_BAD( rc = incrNodesRead()))
					{
						goto Exit;
					}
				}
			}
		}
		else
		{

			// Searching backwards

			if (!(*ppCurrNode))
			{
				*ppCurrNode = pContextNode;
				(*ppCurrNode)->AddRef();
				pHighestAncestorWithSib = NULL;

				// Find highest parent that has a sibling

				for (;;)
				{
					rc = (RCODE)(bPrevAxis
									 ? (*ppCurrNode)->getPreviousSibling( m_pDb, &pSibNode)
									 : (*ppCurrNode)->getNextSibling( m_pDb, &pSibNode));
					if (RC_BAD( rc))
					{
						if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
						{
							rc = NE_XFLM_OK;
						}
						else
						{
							goto Exit;
						}
					}
					else
					{
						if (RC_BAD( rc = incrNodesRead()))
						{
							goto Exit;
						}
						if (pHighestAncestorWithSib)
						{
							pHighestAncestorWithSib->Release();
						}
						pHighestAncestorWithSib = *ppCurrNode;
						pHighestAncestorWithSib->AddRef();
					}

					// Go to node's parent

					if (RC_BAD( rc = (*ppCurrNode)->getParentNode( m_pDb, ppCurrNode)))
					{
						if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
						{
							rc = NE_XFLM_OK;
							break;
						}
						else
						{
							goto Exit;
						}
					}
					else
					{
						if (RC_BAD( rc = incrNodesRead()))
						{
							goto Exit;
						}
					}
				}
				(*ppCurrNode)->Release();
				*ppCurrNode = NULL;

				// If none of the ancestry had a prev/next sibling
				// we are done because there will be no nodes to
				// process.

				if (!pHighestAncestorWithSib)
				{
					goto Exit;
				}

				// Find the leftmost/rightmost sibling of the ancestor node,
				// and go down to its leftmost/rightmost child.
				// That is equivalent to going up to the parent node (which
				// there must be one!) and then going down to the
				// leftmost/rightmost child of that parent.

				*ppCurrNode = pHighestAncestorWithSib;
				(*ppCurrNode)->AddRef();
				if (RC_OK( rc = (*ppCurrNode)->getParentNode( m_pDb, ppCurrNode)))
				{
					if (RC_BAD( rc = incrNodesRead()))
					{
						goto Exit;
					}
				}
				else if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					goto Exit;
				}
				else
				{
					// Should not be possible, because we already know that
					// pHighestAncestorWithSib has a sibling! - which
					// necessitates it having a parent!

					rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
					goto Exit;
				}

				// Now go down to the leftmost/rightmost child

				uiChildCnt = 0;
				for (;;)
				{
					rc = (RCODE)(bPrevAxis
									 ? (*ppCurrNode)->getFirstChild( m_pDb, ppCurrNode)
									 : (*ppCurrNode)->getLastChild( m_pDb, ppCurrNode));
					if (RC_BAD( rc))
					{
						if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
						{
							// If we didn't go down to any child, this is
							// a problem, because we got to the parent node
							// from a child node!

							if (!uiChildCnt)
							{
								rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
							}
							else
							{
								rc = NE_XFLM_OK;
							}
						}
						goto Exit;
					}
					else
					{
						uiChildCnt++;
						if (RC_BAD( rc = incrNodesRead()))
						{
							goto Exit;
						}
					}
				}
			}

			// The following code is only executed if *ppCurrNode is non-NULL
			// when we enter this method.

			rc = (RCODE)(bPrevAxis
							 ? (*ppCurrNode)->getNextSibling( m_pDb, ppCurrNode)
							 : (*ppCurrNode)->getPreviousSibling( m_pDb, ppCurrNode));
			if (RC_OK( rc))
			{
				if (RC_BAD( rc = incrNodesRead()))
				{
					goto Exit;
				}

				// If we have hit the context node, we are done
				
				if( RC_BAD( rc = (*ppCurrNode)->getNodeId( m_pDb, &ui64Tmp)))
				{
					goto Exit;
				}

				if( ui64Tmp == ui64ContextId)
				{
					(*ppCurrNode)->Release();
					*ppCurrNode = NULL;
					goto Exit;
				}

				// Go to rightmost/leftmost child - If we hit the
				// context node anywhere in here, we are done.

				for (;;)
				{
					rc = (RCODE)(bPrevAxis
									 ? (*ppCurrNode)->getFirstChild( m_pDb, ppCurrNode)
									 : (*ppCurrNode)->getLastChild( m_pDb, ppCurrNode));
					if (RC_BAD( rc))
					{
						if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
						{
							rc = NE_XFLM_OK;
							break;
						}
						else
						{
							goto Exit;
						}
					}
					else
					{
						if (RC_BAD( rc = incrNodesRead()))
						{
							goto Exit;
						}
					}

					// See if we hit the context node.
					
					if( RC_BAD( rc = (*ppCurrNode)->getNodeId( m_pDb, &ui64Tmp)))
					{
						goto Exit;
					}

					if( ui64Tmp == ui64ContextId)
					{
						(*ppCurrNode)->Release();
						*ppCurrNode = NULL;
						goto Exit;
					}
				}
			}
			else if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				goto Exit;
			}
			else
			{
				if (RC_BAD( rc = (*ppCurrNode)->getParentNode( m_pDb, ppCurrNode)))
				{
					if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
					{
						(*ppCurrNode)->Release();
						*ppCurrNode = NULL;
						rc = NE_XFLM_OK;
					}
					goto Exit;
				}
				else
				{
					if (RC_BAD( rc = incrNodesRead()))
					{
						goto Exit;
					}

					// Should never hit the context node here
					
#ifdef FLM_DEBUG
					if( RC_BAD( rc = (*ppCurrNode)->getNodeId( m_pDb, &ui64Tmp)))
					{
						goto Exit;
					}

					flmAssert( ui64Tmp != ui64ContextId);
#endif
				}
			}
		}
	}

Exit:

	if (pParentNode)
	{
		pParentNode->Release();
	}
	if (pSibNode)
	{
		pSibNode->Release();
	}
	if (pHighestAncestorWithSib)
	{
		pHighestAncestorWithSib->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:	Get the next node from a ATTRIBUTE_AXIS or NAMESPACE_AXIS.
***************************************************************************/
RCODE F_Query::getAttrAxisNode(
	FLMBOOL				bForward,
	FLMBOOL				bAttrAxis,
	FLMUINT				uiAttrNameId,
	IF_DOMNode *		pContextNode,
	IF_DOMNode **		ppCurrNode
	)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bIsNamespaceDecl;

	if (!pContextNode)
	{
		for (;;)
		{
			if (RC_BAD( rc = walkDocument( bForward, TRUE, uiAttrNameId, ppCurrNode)))
			{
				goto Exit;
			}
			if (!(*ppCurrNode))
			{
				break;
			}

			// Must be an attribute node.

			if ((*ppCurrNode)->getNodeType() != ATTRIBUTE_NODE)
			{
				continue;
			}
			if (bAttrAxis)
			{
				break;
			}

			// Must be a namespace attribute.

			if (RC_BAD( rc = (*ppCurrNode)->isNamespaceDecl( m_pDb,
										&bIsNamespaceDecl)))
			{
				goto Exit;
			}
			if (bIsNamespaceDecl)
			{
				break;
			}
		}
	}
	else
	{
		for (;;)
		{
			if( *ppCurrNode)
			{
				flmAssert( (*ppCurrNode)->getNodeType() == ATTRIBUTE_NODE);
				
				// Better not be testing for a specific name.  This should
				// not have been called if we already processed that
				// attribute.
				
				flmAssert( !uiAttrNameId);

				// Get the next/previous sibling, if any

				rc = (RCODE)(bForward
								 ? (*ppCurrNode)->getNextSibling( m_pDb, ppCurrNode)
								 : (*ppCurrNode)->getPreviousSibling( m_pDb, ppCurrNode));
								 
				if (RC_BAD( rc))
				{
					if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
					{
						(*ppCurrNode)->Release();
						*ppCurrNode = NULL;
						rc = NE_XFLM_OK;
					}
					goto Exit;
				}
				else
				{
					if (RC_BAD( rc = incrNodesRead()))
					{
						goto Exit;
					}
				}
			}
			else
			{
				(*ppCurrNode) = pContextNode;
				(*ppCurrNode)->AddRef();

				// Context node better be an element node

				flmAssert( (*ppCurrNode)->getNodeType() == ELEMENT_NODE);

				// Get the first/last attribute of the node.  If a specific
				// attribute ID is specified, get it - no need to cycle through
				// all of the attributes.
				
				if (uiAttrNameId)
				{
					rc = (*ppCurrNode)->getAttribute( m_pDb, uiAttrNameId, ppCurrNode);
				}
				else
				{
					rc = (RCODE)(bForward
									 ? (*ppCurrNode)->getFirstAttribute( m_pDb, ppCurrNode)
									 : (*ppCurrNode)->getLastAttribute( m_pDb, ppCurrNode));
				}
				if (RC_BAD( rc))
				{
					if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
					{
						(*ppCurrNode)->Release();
						*ppCurrNode = NULL;
						rc = NE_XFLM_OK;
					}
				}
				else
				{
					if (RC_BAD( rc = incrNodesRead()))
					{
						goto Exit;
					}
				}
			}

			if (bAttrAxis)
			{
				break;
			}

			// Must be a namespace attribute.

			if (RC_BAD( rc = (*ppCurrNode)->isNamespaceDecl( m_pDb,
										&bIsNamespaceDecl)))
			{
				goto Exit;
			}
			if (bIsNamespaceDecl)
			{
				break;
			}
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Determine the inverted axis for a given axis type.
***************************************************************************/
FINLINE eXPathAxisTypes invertedAxis(
	eXPathAxisTypes	eAxis)
{
	eXPathAxisTypes	eInvertedAxis = ROOT_AXIS;

	switch (eAxis)
	{
		case ROOT_AXIS:

			// A root axis cannot be inverted.

			flmAssert( 0);
			eInvertedAxis = ROOT_AXIS;
			break;
		case CHILD_AXIS:
		case ATTRIBUTE_AXIS:
		case NAMESPACE_AXIS:
			eInvertedAxis = PARENT_AXIS;
			break;
		case PARENT_AXIS:
			eInvertedAxis = CHILD_AXIS;
			break;
		case ANCESTOR_AXIS:
			eInvertedAxis = DESCENDANT_AXIS;
			break;
		case DESCENDANT_AXIS:
			eInvertedAxis = ANCESTOR_AXIS;
			break;
		case FOLLOWING_SIBLING_AXIS:
			eInvertedAxis = PRECEDING_SIBLING_AXIS;
			break;
		case PRECEDING_SIBLING_AXIS:
			eInvertedAxis = FOLLOWING_SIBLING_AXIS;
			break;
		case FOLLOWING_AXIS:
			eInvertedAxis = PRECEDING_AXIS;
			break;
		case PRECEDING_AXIS:
			eInvertedAxis = FOLLOWING_AXIS;
			break;
		case SELF_AXIS:
			eInvertedAxis = SELF_AXIS;
			break;
		case DESCENDANT_OR_SELF_AXIS:
			eInvertedAxis = ANCESTOR_OR_SELF_AXIS;
			break;
		case ANCESTOR_OR_SELF_AXIS:
			eInvertedAxis = DESCENDANT_OR_SELF_AXIS;
			break;
		case META_AXIS:
			eInvertedAxis = SELF_AXIS;
			break;
	}
	return( eInvertedAxis);
}

/***************************************************************************
Desc:	See if the passed in node ID satisfies the occurrence criteria
		for the xpath component.
***************************************************************************/
RCODE F_Query::verifyOccurrence(
	FLMBOOL				bUseKeyNodes,
	XPATH_COMPONENT *	pXPathComponent,
	IF_DOMNode *		pCurrNode,
	FLMBOOL *			pbPassed)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DOMNode *		pContextNode = NULL;
	IF_DOMNode *		pTmpCurrNode = NULL;
	FLMUINT64			ui64NodeId;
	FLMUINT64			ui64Tmp;
	FLMBOOL				bGetContextNodes = FALSE;
	XPATH_COMPONENT *	pXPathContextComponent = pXPathComponent->pPrev
															 ? pXPathComponent->pPrev
															 : pXPathComponent->pXPathContext;
															 
	if( RC_BAD( rc = pCurrNode->getNodeId( m_pDb, &ui64NodeId)))
	{
		goto Exit;
	}

	if (!pXPathContextComponent)
	{
		pContextNode = NULL;
	}
	else if (bUseKeyNodes && pXPathContextComponent->pKeyNode)
	{
		pContextNode = pXPathContextComponent->pKeyNode;
		pContextNode->AddRef();
	}
	else if (pXPathContextComponent->pCurrNode)
	{
		pContextNode = pXPathContextComponent->pCurrNode;
		pContextNode->AddRef();
	}
	else
	{

		// Need to get context nodes.  May have to try multiple ones.

		bGetContextNodes = TRUE;
		if (RC_BAD( rc = getXPathComponentFromAxis( pCurrNode, TRUE, FALSE,
									pXPathContextComponent, &pContextNode,
									invertedAxis( pXPathComponent->eXPathAxis),
									TRUE, FALSE)))
		{
			goto Exit;
		}
		if (!pContextNode)
		{
			*pbPassed = FALSE;
			goto Exit;
		}
	}

	// *pbPassed better have been passed in as TRUE
	// It will get set to FALSE if we don't find the node
	// we are looking for.

	flmAssert( *pbPassed);

	for (;;)
	{
		if (RC_BAD( rc = getXPathComponentFromAxis( pContextNode, TRUE, FALSE,
								pXPathComponent, &pTmpCurrNode,
								pXPathComponent->eXPathAxis,
								FALSE, FALSE)))
		{
			goto Exit;
		}
		if (!pTmpCurrNode)
		{

No_More_Nodes:

			if (!bGetContextNodes)
			{
				*pbPassed = FALSE;
				goto Exit;
			}

			// If we didn't have a context node to begin with,
			// attempt another one.

			if (RC_BAD( rc = getXPathComponentFromAxis( pCurrNode, TRUE, FALSE,
										pXPathContextComponent,
										&pContextNode,
										invertedAxis( pXPathComponent->eXPathAxis),
										TRUE, FALSE)))
			{
				goto Exit;
			}

			// No more context nodes to try.  We are done.

			if (!pContextNode)
			{
				*pbPassed = FALSE;
				goto Exit;
			}
			continue;
		}
		
		if( RC_BAD( rc = pTmpCurrNode->getNodeId( m_pDb, &ui64Tmp)))
		{
			goto Exit;
		}
		
		if( ui64Tmp == ui64NodeId)
		{
			break;
		}

		// If we are dealing with a constant position, and we didn't
		// find the node we were looking for, we are done.  The
		// above call would have found it.  If we are dealing
		// with an expression, there is no guarantee that the above
		// code would have found just one occurrence of a matching
		// node, and it may not have been the one we were looking
		// for.

		if (pXPathComponent->uiContextPosNeeded)
		{
			goto No_More_Nodes;
		}
	}

Exit:

	if (pTmpCurrNode)
	{
		pTmpCurrNode->Release();
	}
	if (pContextNode)
	{
		pContextNode->Release();
	}
	return( rc);
}

/***************************************************************************
Desc:	Get the node for an xpath component that is related to the
		given xpath context component by the specified axis.
***************************************************************************/
RCODE F_Query::getXPathComponentFromAxis(
	IF_DOMNode *		pContextNode,
	FLMBOOL				bForward,
	FLMBOOL				bUseKeyNodes,
	XPATH_COMPONENT *	pXPathComponent,
	IF_DOMNode **		ppCurrNode,
	eXPathAxisTypes	eAxis,
	FLMBOOL				bAxisInverted,
	FLMBOOL				bCountNodes)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pCurrNode = NULL;
	FLMUINT			uiOccurrence = 0;
	FLMBOOL			bCanCountOccurrence;
	FLMBOOL			bMatches;
	FLMBOOL			bHasContextPosTest = hasContextPosTest( pXPathComponent);
	FLMUINT			uiNameId;

	// Remember the current node, and then set to NULL.

	if (*ppCurrNode)
	{
		pCurrNode = *ppCurrNode;
		pCurrNode->AddRef();
		(*ppCurrNode)->Release();
		*ppCurrNode = NULL;
	}

	// Determine if we can count occurrences.

	if (bForward && !bAxisInverted)
	{
		uiOccurrence = 0;
		bCanCountOccurrence = TRUE;
	}
	else
	{

		// Cannot keep a running count if we are going backwards or if
		// we are operating on an inverted axis.

		bCanCountOccurrence = FALSE;
	}

	// Go until we get a node that passes, or until we have no more
	// nodes to check.

	for (;;)
	{
		if (pXPathComponent->pNodeSource)
		{
			
			// Axis is ignored for xpath components that have a node source.
			
			if (RC_BAD( rc = getNodeSourceNode( bForward,
											pXPathComponent->pNodeSource,
											pContextNode, &pCurrNode)))
			{
				goto Exit;
			}
			goto Test_Node;
		}

		// Type of search we do will depend on axis

		switch (eAxis)
		{
			case ROOT_AXIS:

				// There is only one root node to get, so if we already
				// got it, we are done.

				if (pCurrNode)
				{
					goto Exit;
				}
				flmAssert( pXPathComponent->eNodeType == ELEMENT_NODE);
				if (RC_BAD( rc = getRootAxisNode( &pCurrNode)))
				{
					goto Exit;
				}

				// Will only ever be one occurrence of a root node.

				if (!bAxisInverted)
				{
					uiOccurrence = 0;
					bCanCountOccurrence = TRUE;
				}
				break;

			case CHILD_AXIS:
			
				// For a context that requires all child elements to be unique,
				// there will only be one child with that name ID.
				// If we already got it, we are done.

				if (pContextNode &&
					 (((F_DOMNode *)pContextNode)->getModeFlags() & 
							FDOM_HAVE_CELM_LIST) &&
					 pXPathComponent->uiDictNum)
				{
					if( pCurrNode)
					{
						if( RC_BAD( rc = pCurrNode->getNameId( m_pDb, &uiNameId)))
						{
							goto Exit;
						}
						
						if( uiNameId == pXPathComponent->uiDictNum)
						{
							goto Exit;
						}
					}
					
					if (RC_BAD( rc = getChildAxisNode( bForward, pContextNode,
												pXPathComponent->uiDictNum,
												&pCurrNode)))
					{
						goto Exit;
					}
				}
				else
				{
					if (RC_BAD( rc = getChildAxisNode( bForward, pContextNode, 0,
												&pCurrNode)))
					{
						goto Exit;
					}
				}
				break;

			case PARENT_AXIS:

				// There is only one parent node to get if we are in a
				// specific context, so if we already got it, we are
				// done.

				if (pCurrNode && pContextNode)
				{
					goto Exit;
				}
				if (RC_BAD( rc = getParentAxisNode( bForward, pContextNode,
												&pCurrNode)))
				{
					goto Exit;
				}

				// Will only ever be one occurrence of a parent node.

				if (!bAxisInverted)
				{
					uiOccurrence = 0;
					bCanCountOccurrence = TRUE;
				}
				break;

			case ANCESTOR_AXIS:
				if (RC_BAD( rc = getAncestorAxisNode( bForward, FALSE,
											pContextNode, &pCurrNode)))
				{
					goto Exit;
				}
				break;

			case DESCENDANT_AXIS:
				if (RC_BAD( rc = getDescendantAxisNode( bForward, FALSE,
											pContextNode, &pCurrNode)))
				{
					goto Exit;
				}
				break;

			case FOLLOWING_SIBLING_AXIS:
				if (RC_BAD( rc = getSibAxisNode( bForward, FALSE,
											pContextNode, &pCurrNode)))
				{
					goto Exit;
				}
				break;

			case PRECEDING_SIBLING_AXIS:
				if (RC_BAD( rc = getSibAxisNode( bForward, TRUE,
											pContextNode, &pCurrNode)))
				{
					goto Exit;
				}
				break;

			case FOLLOWING_AXIS:
				if (RC_BAD( rc = getPrevOrAfterAxisNode( bForward, FALSE,
											pContextNode, &pCurrNode)))
				{
					goto Exit;
				}
				break;

			case PRECEDING_AXIS:
				if (RC_BAD( rc = getPrevOrAfterAxisNode( bForward, TRUE,
											pContextNode, &pCurrNode)))
				{
					goto Exit;
				}
				break;

			case ATTRIBUTE_AXIS:

				// For a specific context, there is only one attribute with
				// a given name ID.  If we already got it, we are done.

				if( pCurrNode && pContextNode && pXPathComponent->uiDictNum)
				{
					if( RC_BAD( rc = pCurrNode->getNameId( m_pDb, &uiNameId)))
					{
						goto Exit;
					}
					
					if( uiNameId == pXPathComponent->uiDictNum)
					{
						goto Exit;
					}
				}
				
				if (RC_BAD( rc = getAttrAxisNode( bForward, TRUE,
											pXPathComponent->uiDictNum,
											pContextNode,
											&pCurrNode)))
				{
					goto Exit;
				}
				break;

			case NAMESPACE_AXIS:
				if (RC_BAD( rc = getAttrAxisNode( bForward, FALSE, 0, pContextNode,
											&pCurrNode)))
				{
					goto Exit;
				}
				break;

			case SELF_AXIS:
			case META_AXIS:

				// Within a given context, there is only one occurrence of the
				// META or SELF axis.  If we already got it, we are done.

				if (pCurrNode && pContextNode)
				{
					goto Exit;
				}
				if (!pContextNode)
				{

					// If the node is the SELF_AXIS, there better be
					// a context node!

					flmAssert( eAxis != SELF_AXIS);
					if (RC_BAD( rc = walkDocument( bForward, TRUE, 0, &pCurrNode)))
					{
						goto Exit;
					}
				}
				else
				{
					pCurrNode = pContextNode;
					pCurrNode->AddRef();
				}

				// Will only ever be one occurrence of a meta-axis node or
				// a self-axis node.

				if (!bAxisInverted)
				{
					uiOccurrence = 0;
					bCanCountOccurrence = TRUE;
				}
				break;

			case DESCENDANT_OR_SELF_AXIS:
				if (RC_BAD( rc = getDescendantAxisNode( bForward, TRUE,
											pContextNode, &pCurrNode)))
				{
					goto Exit;
				}
				break;

			case ANCESTOR_OR_SELF_AXIS:
				if (RC_BAD( rc = getAncestorAxisNode( bForward, TRUE,
											pContextNode, &pCurrNode)))
				{
					goto Exit;
				}
				break;
		}

Test_Node:

		if (!pCurrNode)
		{
			break;
		}

		if (bCountNodes)
		{
			m_pCurrOpt->ui64NodesTested++;
			if (RC_BAD( rc = queryStatus()))
			{
				goto Exit;
			}
		}

		bMatches = TRUE;
		if (pXPathComponent->eNodeType != ANY_NODE_TYPE && eAxis != META_AXIS)
		{
			if (pCurrNode->getNodeType() != pXPathComponent->eNodeType)
			{
				bMatches = FALSE;
			}

			// Verify that the dictionary number matches.

			else if (pXPathComponent->eNodeType == ELEMENT_NODE ||
					   pXPathComponent->eNodeType == ATTRIBUTE_NODE ||
					   pXPathComponent->eNodeType == DATA_NODE)
			{
				if (pXPathComponent->uiDictNum)
				{
					if( RC_BAD( rc = pCurrNode->getNameId( m_pDb, &uiNameId)))
					{
						goto Exit;
					}
					
					if( uiNameId != pXPathComponent->uiDictNum)
					{
						bMatches = FALSE;
					}
				}
			}
		}

		// If there is an expression evaluate it.
		// The expression must pass in order to
		// keep this particular node.

		if (bMatches && pXPathComponent->pExpr)
		{
			if (RC_BAD( rc = evalExpr( pCurrNode, TRUE, bUseKeyNodes,
									pXPathComponent->pExpr, &bMatches, NULL)))
			{
				goto Exit;
			}
		}

		// See if there is an occurrence test.

		if (bMatches && bHasContextPosTest)
		{
			if (bCanCountOccurrence)
			{
				uiOccurrence++;
				if (pXPathComponent->uiContextPosNeeded)
				{
					if (uiOccurrence != pXPathComponent->uiContextPosNeeded)
					{
						bMatches = FALSE;
					}
				}
				else
				{
					FLMUINT	uiPosNeeded;

					if (RC_BAD( rc = evalExpr( pCurrNode, TRUE, bUseKeyNodes,
											pXPathComponent->pContextPosExpr,
											NULL, NULL)))
					{
						goto Exit;
					}
					if (RC_BAD( fqGetPosition( &pXPathComponent->pContextPosExpr->currVal,
												&uiPosNeeded)) ||
						 uiOccurrence != uiPosNeeded)
					{
						bMatches = FALSE;
					}
				}
			}
			else
			{

				// Need to verify the context position because we couldn't
				// calculate it as we went.

				if (RC_BAD( rc = verifyOccurrence( bUseKeyNodes, pXPathComponent,
								pCurrNode, &bMatches)))
				{
					goto Exit;
				}
			}
		}

		if (bMatches)
		{
			*ppCurrNode = pCurrNode;

			// No need to AddRef on *ppCurrNode, just steal it from
			// pCurrNode.  Setting pCurrNode to NULL keeps it from
			// being Release()'d below.

			pCurrNode = NULL;
			break;
		}
	}

Exit:

	if (pCurrNode)
	{
		pCurrNode->Release();
	}
	
	return( rc);
}

/***************************************************************************
Desc:	Get an FQVALUE from a DOM node.
***************************************************************************/
FSTATIC RCODE fqGetValueFromNode(
	F_Db *			pDb,
	IF_DOMNode *	pNode,
	FQVALUE *		pQValue,
	FLMUINT			uiMetaDataType
	)
{
	RCODE		rc = NE_XFLM_OK;
	FLMUINT	uiDataType;
	FLMUINT	uiNumChars;

	pQValue->uiFlags = 0;
	if (uiMetaDataType)
	{
		switch (uiMetaDataType)
		{
			case XFLM_META_NODE_ID:
				pQValue->eValType = XFLM_UINT64_VAL;
				rc = pNode->getNodeId( pDb, &pQValue->val.ui64Val);
				goto Exit;
			case XFLM_META_DOCUMENT_ID:
				pQValue->eValType = XFLM_UINT64_VAL;
				rc = pNode->getDocumentId( pDb, &pQValue->val.ui64Val);
				goto Exit;
			case XFLM_META_PARENT_ID:
				pQValue->eValType = XFLM_UINT64_VAL;
				rc = pNode->getParentId( pDb, &pQValue->val.ui64Val);
				goto Exit;
			case XFLM_META_FIRST_CHILD_ID:
				pQValue->eValType = XFLM_UINT64_VAL;
				rc = pNode->getFirstChildId( pDb, &pQValue->val.ui64Val);
				goto Exit;
			case XFLM_META_LAST_CHILD_ID:
				pQValue->eValType = XFLM_UINT64_VAL;
				rc = pNode->getLastChildId( pDb, &pQValue->val.ui64Val);
				goto Exit;
			case XFLM_META_NEXT_SIBLING_ID:
				pQValue->eValType = XFLM_UINT64_VAL;
				rc = pNode->getNextSibId( pDb, &pQValue->val.ui64Val);
				goto Exit;
			case XFLM_META_PREV_SIBLING_ID:
				pQValue->eValType = XFLM_UINT64_VAL;
				rc = pNode->getPrevSibId( pDb, &pQValue->val.ui64Val);
				goto Exit;
			case XFLM_META_VALUE:
				pQValue->eValType = XFLM_UINT64_VAL;
				rc = pNode->getMetaValue( pDb, &pQValue->val.ui64Val);
				goto Exit;
			default:
				break;
		}
	}

	if (RC_BAD( rc = pNode->getDataType( pDb, &uiDataType)))
	{
		goto Exit;
	}
	switch (uiDataType)
	{
		case XFLM_NODATA_TYPE:

			// Should have been set to missing on the outside

			flmAssert( pQValue->eValType == XFLM_MISSING_VAL);
			pQValue->eValType = XFLM_BOOL_VAL;
			pQValue->val.eBool = XFLM_TRUE;
			break;
		case XFLM_TEXT_TYPE:
			if (RC_BAD( rc = pNode->getTextIStream( pDb,
												&pQValue->val.pIStream,
												&uiNumChars)))
			{
				goto Exit;
			}
			pQValue->eValType = XFLM_UTF8_VAL;
			pQValue->uiFlags |= VAL_IS_STREAM;
			break;
		case XFLM_NUMBER_TYPE:

			// First, see if we can get it as a UINT - the most common
			// type.

			if (RC_OK( rc = pNode->getUINT( pDb, &pQValue->val.uiVal)))
			{
				pQValue->eValType = XFLM_UINT_VAL;
			}
			else if (rc == NE_XFLM_CONV_NUM_OVERFLOW)
			{
				if (RC_BAD( rc = pNode->getUINT64( pDb,
												&pQValue->val.ui64Val)))
				{
					goto Exit;
				}
				pQValue->eValType = XFLM_UINT64_VAL;
			}
			else if (rc == NE_XFLM_CONV_NUM_UNDERFLOW)
			{
				if (RC_OK( rc = pNode->getINT( pDb, &pQValue->val.iVal)))
				{
					pQValue->eValType = XFLM_INT_VAL;
				}
				else if (rc == NE_XFLM_CONV_NUM_UNDERFLOW)
				{
					if (RC_BAD( rc = pNode->getINT64( pDb,
														&pQValue->val.i64Val)))
					{
						goto Exit;
					}
					pQValue->eValType = XFLM_INT64_VAL;
				}
			}
			else
			{
				goto Exit;
			}
			break;
		case XFLM_BINARY_TYPE:
			if (RC_BAD( rc = pNode->getIStream( pDb, &pQValue->val.pIStream,
												&uiDataType)))
			{
				goto Exit;
			}
			flmAssert( uiDataType == XFLM_BINARY_TYPE);
			pQValue->eValType = XFLM_BINARY_VAL;
			pQValue->uiFlags |= VAL_IS_STREAM;
			break;
		default:
			rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
			goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get the next value from an XPATH node.
***************************************************************************/
RCODE F_Query::getNextXPathValue(
	IF_DOMNode *	pContextNode,
	FLMBOOL			bForward,
	FLMBOOL			bUseKeyNodes,
	FLMBOOL			bXPathIsEntireExpr,
	FQNODE *			pQNode
	)
{
	RCODE					rc = NE_XFLM_OK;
	FXPATH *				pXPath;
	XPATH_COMPONENT *	pXPathComponent;

	// Node better be an XPATH node.

	flmAssert( pQNode->eNodeType == FLM_XPATH_NODE);
	pXPath = pQNode->nd.pXPath;

	// If the old value type was a stream, get rid of it.

	fqReleaseNodeValue( pQNode);
	if (pXPath->bGettingNodes)
	{
		pXPathComponent = pXPath->pLastComponent;
	}
	else if (pXPath->bIsSource && bUseKeyNodes)
	{
		pXPathComponent = pXPath->pSourceComponent->pNext;

		// This routine should never be called if the source component
		// is the rightmost component.  That is taken care of on
		// the outside.

		flmAssert( pXPathComponent);
	}
	else
	{
		pXPathComponent = pXPath->pFirstComponent;
	}

	// This loop will go until we get to the last xpath component and get
	// a node.  If it cannot get a node at one context, it will retreat
	// to the prior context, until it gets back to the first xpath
	// component, and there are no more to get

	for (;;)
	{
		IF_DOMNode *	pTmpContextNode;

		if (!pXPathComponent->pPrev)
		{
			pTmpContextNode = pContextNode;
		}
		else if (bUseKeyNodes && pXPathComponent->pPrev->pKeyNode)
		{
			pTmpContextNode = pXPathComponent->pPrev->pKeyNode;
		}
		else
		{
			pTmpContextNode = pXPathComponent->pPrev->pCurrNode;
		}
		if (RC_BAD( rc = getXPathComponentFromAxis( pTmpContextNode,
								bForward, bUseKeyNodes, pXPathComponent,
								&pXPathComponent->pCurrNode,
								pXPathComponent->eXPathAxis, FALSE,
								bXPathIsEntireExpr && !pXPathComponent->pNext
								? TRUE
								: FALSE)))
		{
			goto Exit;
		}

		// If we didn't get any nodes that passed here, we have nothing
		// to give back.

		if (pXPathComponent->pCurrNode)
		{
			if ((pXPathComponent = pXPathComponent->pNext) == NULL)
			{
				break;
			}
		}
		else if (pXPathComponent->pPrev &&
					(!pXPathComponent->pPrev->bIsSource || !bUseKeyNodes))
		{

			// Was no node, go to prior context to get its next node. 

			pXPathComponent = pXPathComponent->pPrev;
		}
		else
		{

			// There was no prior context, so we do not have a value
			// to return for this xpath.  Note that
			// pQNode->currVal.eValType should already have been set to
			// XFLM_MISSING_VAL up above, so no need to set it here.

			flmAssert( pQNode->currVal.eValType == XFLM_MISSING_VAL);
			fqResetIterator( pQNode, FALSE, bUseKeyNodes);
			goto Exit;
		}
	}

	// Extract the value from the DOM node pointed to by the
	// last component in the xpath.

	pXPath->bGettingNodes = TRUE;

	if (pQNode->pParent && isLogicalOp( pQNode->pParent->nd.op.eOperator))
	{
		pQNode->currVal.eValType = XFLM_BOOL_VAL;
		pQNode->currVal.val.eBool = (pQNode->bNotted ? XFLM_FALSE : XFLM_TRUE);
	}
	else
	{
		if (RC_BAD( rc = fqGetValueFromNode( m_pDb,
						pXPath->pLastComponent->pCurrNode,
						&pQNode->currVal,
						getMetaDataType( pXPath->pLastComponent))))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Test a value and return a TRUE/FALSE boolean.
***************************************************************************/
FSTATIC FLMBOOL fqTestValue(
	FQNODE *	pQueryExpr
	)
{
	FLMBOOL	bPassed;
	
	switch (pQueryExpr->currVal.eValType)
	{
		case XFLM_BOOL_VAL:
			bPassed = (pQueryExpr->currVal.val.eBool == XFLM_TRUE) ? TRUE : FALSE;
			break;
		case XFLM_UINT_VAL:
			bPassed = (pQueryExpr->currVal.val.uiVal) ? TRUE : FALSE;
			break;
		case XFLM_UINT64_VAL:
			bPassed = (pQueryExpr->currVal.val.ui64Val) ? TRUE : FALSE;
			break;
		case XFLM_INT_VAL:
			bPassed = (pQueryExpr->currVal.val.iVal) ? TRUE : FALSE;
			break;
		case XFLM_INT64_VAL:
			bPassed = (pQueryExpr->currVal.val.i64Val) ? TRUE : FALSE;
			break;
		case XFLM_BINARY_VAL:
		case XFLM_UTF8_VAL:
			bPassed = (pQueryExpr->currVal.uiDataLen) ? TRUE : FALSE;
			break;
		default:
			bPassed = FALSE;
			break;
	}
	return( bPassed);
}

/***************************************************************************
Desc:	Get the next value for a function.
***************************************************************************/
RCODE F_Query::getNextFunctionValue(
	IF_DOMNode *		pContextNode,
	FLMBOOL				bForward,
	FQNODE *				pCurrNode,
	F_DynaBuf *			pDynaBuf)
{
	RCODE				rc = NE_XFLM_OK;
	ValIterator		eIterator;
	FLMBYTE			ucValBuf [sizeof( FLMUINT64)];
	FLMBYTE *		pucVal;
	IF_DOMNode *	pNode = NULL;
	FLMBOOL			bPassed;
	
	// Evaluate the parameter expression, if any.
	
	if (pCurrNode->nd.pQFunction->pFirstArg)
	{
		if (RC_BAD( rc = evalExpr( pContextNode, TRUE, FALSE,
								 pCurrNode->nd.pQFunction->pFirstArg->pExpr,
								 &bPassed, &pNode)))
		{
			if (rc == NE_XFLM_EOF_HIT || rc == NE_XFLM_BOF_HIT)
			{
				flmAssert( pNode == NULL);
				rc = NE_XFLM_OK;
			}
			else
			{
				goto Exit;
			}
		}
	}
	
	if (bForward)
	{
		eIterator = (!pCurrNode->bUsedValue) ? GET_FIRST_VAL : GET_NEXT_VAL; 
	}
	else
	{
		eIterator = (!pCurrNode->bUsedValue) ? GET_LAST_VAL : GET_PREV_VAL; 
	}
	
	fqReleaseNodeValue( pCurrNode);
	pCurrNode->currVal.uiDataLen = 0;
	pCurrNode->currVal.uiFlags = 0;
	pDynaBuf->truncateData( 0);
	if (RC_BAD( rc = pCurrNode->nd.pQFunction->pFuncObj->getValue( (IF_Db *)m_pDb,
								pNode, eIterator, &pCurrNode->currVal.eValType,
								&pCurrNode->bLastValue, ucValBuf, pDynaBuf)))
	{
		goto Exit;
	}
	
	pucVal = &ucValBuf [0];
	switch (pCurrNode->currVal.eValType)
	{
		case XFLM_BOOL_VAL:
			pCurrNode->currVal.val.eBool = *((XFlmBoolType *)pucVal);
			break;
		case XFLM_UINT_VAL:
			pCurrNode->currVal.val.uiVal = *((FLMUINT *)pucVal);
			break;
		case XFLM_UINT64_VAL:
			pCurrNode->currVal.val.ui64Val = *((FLMUINT64 *)pucVal);
			break;
		case XFLM_INT_VAL:
			pCurrNode->currVal.val.iVal = *((FLMINT *)pucVal);
			break;
		case XFLM_INT64_VAL:
			pCurrNode->currVal.val.i64Val = *((FLMINT64 *)pucVal);
			break;
		case XFLM_BINARY_VAL:
		case XFLM_UTF8_VAL:
			pCurrNode->currVal.uiDataLen = pDynaBuf->getDataLength();
			pCurrNode->currVal.val.pucBuf = (FLMBYTE *)pDynaBuf->getBufferPtr();
			break;
		case XFLM_MISSING_VAL:
		default:
			break;
	}
	
Exit:

	if (pNode)
	{
		pNode->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:	Reset the entire query tree.  It could have been left in a partial
		evaluation state from the last document evaluated.
***************************************************************************/
FSTATIC void fqResetQueryTree(
	FQNODE *	pQueryTree,
	FLMBOOL	bUseKeyNodes,
	FLMBOOL	bResetAllXPaths)
{
	FQNODE *	pCurrNode = pQueryTree;
	
	for (;;)
	{
		fqReleaseNodeValue( pCurrNode);
		pCurrNode->bUsedValue = FALSE;
		pCurrNode->bLastValue = FALSE;
		if (pCurrNode->pFirstChild)
		{
			pCurrNode = pCurrNode->pFirstChild;
		}
		else
		{
			if (pCurrNode->eNodeType == FLM_XPATH_NODE)
			{

				// Don't reset the iterator if this xpath node is
				// the root node of the outermost query.

				if (bResetAllXPaths ||
					 pCurrNode->pParent ||
					 pCurrNode->nd.pXPath->pFirstComponent->pXPathContext)
				{
					fqResetIterator( pCurrNode, FALSE, bUseKeyNodes);
				}
			}
			while (!pCurrNode->pNextSib)
			{
				if ((pCurrNode = pCurrNode->pParent) == NULL)
				{
					break;
				}
			}
			if (!pCurrNode)
			{
				break;
			}
			pCurrNode = pCurrNode->pNextSib;
		}
	}
}

/***************************************************************************
Desc:	Evaluate an operand in a query expression.  If it is an operand of a
		boolean operator (AND/OR), we may be able to short-circuit the
		evaluation of the other operand.  This will be propagated up the
		query tree as far as possible.
		If it is an operand of some other operator, we check to see if we
		have both operands.  If not, we move *ppCurrNode to get the next
		sibling so we can evalute the other operand.
		VISIT: We could short-circuit arithmetic operations if an operand
		is missing.
***************************************************************************/
FSTATIC RCODE fqTryEvalOperator(
	FLMUINT		uiLanguage,
	FQNODE **	ppCurrNode)
{
	RCODE				rc = NE_XFLM_OK;
	FQNODE *			pCurrNode = *ppCurrNode;
	XFlmBoolType	eBoolVal;
	XFlmBoolType	eBoolPartialEval;

	for (;;)
	{
		if (!pCurrNode->pParent)
		{
			pCurrNode = NULL;
			break;
		}

		// If the current node's parent is an AND or OR
		// operator, see if we even need to go to the next
		// sibling.

		flmAssert( pCurrNode->pParent->eNodeType == FLM_OPERATOR_NODE);
		if (pCurrNode->pParent->nd.op.eOperator == XFLM_AND_OP ||
			 pCurrNode->pParent->nd.op.eOperator == XFLM_OR_OP)
		{
			eBoolVal = XFLM_UNKNOWN;
			eBoolPartialEval = pCurrNode->pParent->nd.op.eOperator == XFLM_AND_OP
									  ? XFLM_FALSE
									  : XFLM_TRUE;
			if (pCurrNode->eNodeType == FLM_OPERATOR_NODE)
			{

				// It may not have been evaluated because of missing
				// XPATH values in one or both operands, in which case
				// its state will be XFLM_MISSING_VALUE.  If it was
				// evaluated, its state should show a boolean value.

				if (pCurrNode->currVal.eValType == XFLM_MISSING_VAL)
				{
					eBoolVal = (pCurrNode->bNotted ? XFLM_TRUE : XFLM_FALSE);
				}
				else
				{
					flmAssert( pCurrNode->currVal.eValType == XFLM_BOOL_VAL);
					eBoolVal = pCurrNode->currVal.val.eBool;
				}
			}
			else if (pCurrNode->eNodeType == FLM_XPATH_NODE)
			{
				if (!pCurrNode->bNotted)
				{
					eBoolVal = (pCurrNode->currVal.eValType == XFLM_MISSING_VAL)
									? XFLM_FALSE
									: XFLM_TRUE;
				}
				else
				{
					eBoolVal = (pCurrNode->currVal.eValType == XFLM_MISSING_VAL)
									? XFLM_TRUE
									: XFLM_FALSE;
				}
			}
			else if (pCurrNode->eNodeType == FLM_VALUE_NODE)
			{

				// Only allowed value node underneath a logical operator is
				// a boolean value that has a value of XFLM_UNKNOWN.
				// XFLM_FALSE and XFLM_TRUE will already have been weeded out.

				flmAssert( pCurrNode->currVal.eValType == XFLM_BOOL_VAL &&
							  pCurrNode->currVal.val.eBool == XFLM_UNKNOWN);

				// No need to set eBoolVal to XFLM_UNKNOWN, because it will never
				// match eBoolPartialEval in the test below.  eBoolPartialEval
				// is always either XFLM_FALSE or XFLM_TRUE.

				// eBoolVal = XFLM_UNKNOWN;

			}
			else
			{
				flmAssert( pCurrNode->eNodeType == FLM_FUNCTION_NODE);
				if (!pCurrNode->bNotted)
				{
					eBoolVal = fqTestValue( pCurrNode) ? XFLM_TRUE : XFLM_FALSE;
				}
				else
				{
					eBoolVal = fqTestValue( pCurrNode) ? XFLM_FALSE : XFLM_TRUE;
				}
			}
			if (eBoolVal == eBoolPartialEval)
			{
				pCurrNode = pCurrNode->pParent;
				pCurrNode->currVal.eValType = XFLM_BOOL_VAL;
				pCurrNode->currVal.val.eBool = eBoolVal;
			}
			else if (pCurrNode->pNextSib)
			{
				pCurrNode = pCurrNode->pNextSib;
				break;
			}
			else
			{
				pCurrNode = pCurrNode->pParent;
				if (RC_BAD( rc = fqEvalOperator( uiLanguage, pCurrNode)))
				{
					goto Exit;
				}
			}
		}
		else
		{

			// Visit: can shortcircuit comparison operators if the
			// value is missing.

			if (pCurrNode->pNextSib)
			{
				pCurrNode = pCurrNode->pNextSib;
				break;
			}
			pCurrNode = pCurrNode->pParent;

			// All operands are now present - do evaluation

			if (RC_BAD( rc = fqEvalOperator( uiLanguage, pCurrNode)))
			{
				goto Exit;
			}

			// If this is a comparison operator, see if we need to
			// do existential or universal comparison.

			if (isCompareOp( pCurrNode->nd.op.eOperator))
			{

				// Evaluation should have forced current value to be a
				// boolean TRUE or FALSE value.

				flmAssert( pCurrNode->currVal.eValType == XFLM_BOOL_VAL);
				if ((isUniversal( pCurrNode) &&
					  pCurrNode->currVal.val.eBool == XFLM_TRUE) ||
					 (isExistential( pCurrNode) &&
					  pCurrNode->currVal.val.eBool == XFLM_FALSE))
				{

					// Go back down right hand side to get next rightmost
					// operand

					while (pCurrNode->pLastChild)
					{
						pCurrNode = pCurrNode->pLastChild;
					}
					break;
				}
			}
		}
	}
		
Exit:

	*ppCurrNode = pCurrNode;
	return( rc);
}

/***************************************************************************
Desc:	Backup the query tree
***************************************************************************/
FSTATIC FQNODE * fqBackupTree(
	FQNODE *		pCurrNode,
	FLMBOOL *	pbGetNodeValue
	)
{
	flmAssert( !(*pbGetNodeValue));
	
	// This was the last value, backup
				
	pCurrNode->bUsedValue = FALSE;
	pCurrNode->bLastValue = FALSE;
				
	// If parent node is a logical operator, there is
	// nothing more we can do on this branch, so we fall
	// through and process the thing.
				
	if (pCurrNode->pParent &&
		 !isLogicalOp( pCurrNode->pParent->nd.op.eOperator))
	{
		while (!pCurrNode->pPrevSib)
		{
			if ((pCurrNode = pCurrNode->pParent) == NULL)
			{
				goto Exit;
			}
				
			// Don't backup any higher than comparison operators
			// We are only backing up so we can do existential
			// and universal on them with multiple left and right
			// operands.
				
			flmAssert( pCurrNode->eNodeType == FLM_OPERATOR_NODE);
			if (isCompareOp( pCurrNode->nd.op.eOperator))
			{
				goto Exit;
			}
		}
				
		// If we came up to a parent that is a comparison operator
		// we are done - there is noplace else to backup to, so
		// we will fall through.
				
		if (!isCompareOp( pCurrNode->nd.op.eOperator))
		{
				
			// has to be a prev sib at this point.
				
			pCurrNode = pCurrNode->pPrevSib;
				
			// Go down to rightmost child
				
			while (pCurrNode->pLastChild)
			{
				pCurrNode = pCurrNode->pLastChild;
			}
			*pbGetNodeValue = TRUE;
		}
	}
	
Exit:

	return( pCurrNode);
}

/***************************************************************************
Desc:	Get the value of a function node, or move to another node.
***************************************************************************/
RCODE F_Query::getFuncValue(
	IF_DOMNode *		pContextNode,
	FLMBOOL				bForward,
	FQNODE **			ppCurrNode,
	FLMBOOL *			pbGetNodeValue,
	F_DynaBuf *			pDynaBuf)
{
	RCODE					rc = NE_XFLM_OK;
	FQNODE *				pCurrNode = *ppCurrNode;
	
	// We currently only support user-defined functions.
				
	flmAssert( pCurrNode->nd.pQFunction->pFuncObj);
	flmAssert( !(*pbGetNodeValue));
	
	if (pCurrNode->bLastValue)
	{
		pCurrNode = fqBackupTree( pCurrNode, pbGetNodeValue);
	}
	else
	{
		if (RC_BAD( rc = getNextFunctionValue( pContextNode,
									bForward, pCurrNode,
									pDynaBuf)))
		{
			goto Exit;
		}
				
		// Handle case where the function call is the only thing in the
		// query.
				
		if (!pCurrNode->pParent)
		{
			FLMBOOL	bTmpPassed;
			FLMBOOL	bFinal;
			
			if (pCurrNode->currVal.eValType == XFLM_MISSING_VAL)
			{
				bFinal = TRUE;
				
				// No more values.  If it is universal, and we got to this
				// point, it means that all prior values passed, so we set
				// bTmpPassed to TRUE in that case.  If it is existential and
				// we got to this point, it means that all prior values
				// failed, so we set bTmpPassed to FALSE in that case.
				
				bTmpPassed = isUniversal( pCurrNode);
			}
			else
			{
				bTmpPassed = fqTestValue( pCurrNode);
				bFinal = (pCurrNode->bLastValue ||
							 (isUniversal( pCurrNode) && !bTmpPassed) ||
							 (isExistential( pCurrNode) && bTmpPassed))
							? TRUE
							: FALSE;
			}
			fqReleaseNodeValue( pCurrNode);
			if (!bFinal)
			{
				*pbGetNodeValue = TRUE;
				goto Exit;
			}
			pCurrNode->currVal.eValType = XFLM_BOOL_VAL;
			pCurrNode->currVal.val.eBool = bTmpPassed ? XFLM_TRUE : XFLM_FALSE;
			
			// Must set to NULL so that evalExpr will break out of loop.
			
			pCurrNode = NULL;
			goto Exit;
		}
				
		// Handle case where it is not the top node of the query.
				
		if (pCurrNode->currVal.eValType != XFLM_MISSING_VAL)
		{
			pCurrNode->bUsedValue = TRUE;
		}
		else
		{
			if (!pCurrNode->bUsedValue)
			{
						
				// Need to handle missing value if we have not done so yet.
						
				pCurrNode->bUsedValue = TRUE;
				pCurrNode->bLastValue = TRUE;
			}
			else
			{
				// This was the last value, backup
						
				pCurrNode = fqBackupTree( pCurrNode, pbGetNodeValue);
			}
		}
	}
	
Exit:

	*ppCurrNode = pCurrNode;
	return( rc);
}

/***************************************************************************
Desc:	Get the value of an XPATH node or move to another node.
***************************************************************************/
RCODE F_Query::getXPathValue(
	IF_DOMNode *	pContextNode,
	FLMBOOL			bForward,
	FQNODE **		ppCurrNode,
	FLMBOOL *		pbGetNodeValue,
	FLMBOOL			bUseKeyNodes,
	FLMBOOL			bXPathIsEntireExpr
	)
{
	RCODE		rc = NE_XFLM_OK;
	FQNODE *	pCurrNode = *ppCurrNode;

	flmAssert( !(*pbGetNodeValue));
	flmAssert( !(*pbGetNodeValue));

	// See if value is already set from an index.

	if (pCurrNode->nd.pXPath->bHavePassingNode && bUseKeyNodes)
	{
		if (pCurrNode->bUsedValue)
		{
			pCurrNode->currVal.eValType = XFLM_MISSING_VAL;
		}
		else
		{
			pCurrNode->currVal.eValType = XFLM_PASSING_VAL;
			if (bXPathIsEntireExpr)
			{
				m_pCurrOpt->ui64NodesTested++;
				if (RC_BAD( rc = queryStatus()))
				{
					goto Exit;
				}
			}
		}
	}
	else if (bUseKeyNodes && pCurrNode->nd.pXPath->bIsSource &&
				 pCurrNode->nd.pXPath->pLastComponent->bIsSource)
	{
		fqReleaseNodeValue( pCurrNode);
		if (!pCurrNode->bUsedValue)
		{
			XPATH_COMPONENT *	pLastComponent =
								pCurrNode->nd.pXPath->pLastComponent;

			if (RC_BAD( rc = fqGetValueFromNode( m_pDb,
						pLastComponent->pKeyNode,
						&pCurrNode->currVal,
						getMetaDataType( pLastComponent))))
			{
				goto Exit;
			}
			pCurrNode->bUsedValue = TRUE;
			if (bXPathIsEntireExpr)
			{
				m_pCurrOpt->ui64NodesTested++;
				if (RC_BAD( rc = queryStatus()))
				{
					goto Exit;
				}
			}
		}
	}
	else
	{
		if (RC_BAD( rc = getNextXPathValue( pContextNode,
									bForward, bUseKeyNodes,
									bXPathIsEntireExpr, pCurrNode)))
		{
			goto Exit;
		}
	}
			
	// If this expression is an XPATH, set pCurrNode to NULL so that
	// it will cause evalExpr to exit from its loop and return
	// the node we have found.  No need to attempt to evaluate
	// an operators, there are none.
			
	if (!pCurrNode->pParent)
	{
		pCurrNode = NULL;
		goto Exit;
	}
			
	if (pCurrNode->currVal.eValType != XFLM_MISSING_VAL)
	{
		pCurrNode->bUsedValue = TRUE;
	}
	else
	{
		fqResetIterator( pCurrNode, FALSE, bUseKeyNodes);
				
		// See if we used the "missing" value in evaluating
		// the operator.  Need to if we have not yet.
		// Otherwise, we backup in the tree because we will
		// have used at least one value.

		if (!pCurrNode->bUsedValue)
		{
			pCurrNode->bUsedValue = TRUE;
		}
		else
		{
			pCurrNode = fqBackupTree( pCurrNode, pbGetNodeValue);
		}
	}
	
Exit:

	*ppCurrNode = pCurrNode;
	return( rc);
}

/***************************************************************************
Desc:	Set the return value for an expression.
***************************************************************************/
RCODE F_Query::setExprReturnValue(
	FLMBOOL				bUseKeyNodes,
	FQNODE *				pQueryExpr,
	FLMBOOL *			pbPassed,
	IF_DOMNode **		ppNode
	)
{
	RCODE	rc = NE_XFLM_OK;
	
	if (pQueryExpr->eNodeType == FLM_XPATH_NODE)
	{
		if (pQueryExpr->currVal.eValType != XFLM_MISSING_VAL)
		{
	
			// Need to clear the current node's value.
	
			fqReleaseNodeValue( pQueryExpr);
			if (ppNode)
			{
	
				// *ppNode should have been initialized to NULL at the
				// beginning of the evalExpr routine.
	
				flmAssert( *ppNode == NULL);
				if (bUseKeyNodes &&
					 pQueryExpr->nd.pXPath->pLastComponent->pKeyNode)
				{
					*ppNode = pQueryExpr->nd.pXPath->pLastComponent->pKeyNode;
				}
				else
				{
					*ppNode = pQueryExpr->nd.pXPath->pLastComponent->pCurrNode;
				}
				(*ppNode)->AddRef();
				m_pCurrOpt->ui64NodesPassed++;
				if (RC_BAD( rc = queryStatus()))
				{
					goto Exit;
				}
			}
			if (pbPassed)
			{
				*pbPassed = TRUE;
			}
		}
		else
		{
			
			// *pbPassed should have been initialized to FALSE in evalExpr,
			// if pbPassed is non-NULL.
	
			flmAssert( !pbPassed || !(*pbPassed));
	
			// *ppNode will already have been set to NULL - at beginning
			// of evalExpr - so no need to do it here.  Just assert
			// that either ppNode is NULL or *ppNode is NULL.
			
			flmAssert( !ppNode || *ppNode == NULL);
	
			// If pCurrNode or pKeyNode is non-NULL it means we found one, but
			// it doesn't have any data.  That's ok, because this is
			// only an exists test, and it should pass.
			
			if (bUseKeyNodes &&
				 pQueryExpr->nd.pXPath->pLastComponent->pKeyNode)
			{
				if (pbPassed)
				{
					*pbPassed = TRUE;
				}
				if (ppNode)
				{
					*ppNode = pQueryExpr->nd.pXPath->pLastComponent->pKeyNode;
					(*ppNode)->AddRef();
				}
				m_pCurrOpt->ui64NodesPassed++;
				if (RC_BAD( rc = queryStatus()))
				{
					goto Exit;
				}
			}
			else if (pQueryExpr->nd.pXPath->pLastComponent->pCurrNode)
			{
				if (pbPassed)
				{
					*pbPassed = TRUE;
				}
				if (ppNode)
				{
					*ppNode = pQueryExpr->nd.pXPath->pLastComponent->pCurrNode;
					(*ppNode)->AddRef();
				}
				m_pCurrOpt->ui64NodesPassed++;
				if (RC_BAD( rc = queryStatus()))
				{
					goto Exit;
				}
			}
		}
		
		// This routine is only called for expressions that are just an
		// XPATH.  If it is the outermost XPATH, we don't want to
		// reset the iterator, as that is what we are using to
		// iterate with.
		
		if (pQueryExpr->nd.pXPath->pFirstComponent->pXPathContext)
		{
			fqResetIterator( pQueryExpr, FALSE, bUseKeyNodes);
		}
	}
	else if (pbPassed)
	{
		*pbPassed = fqTestValue( pQueryExpr);
		if (ppNode)
		{
			if (*pbPassed)
			{
				*ppNode = m_pCurrDoc;
				(*ppNode)->AddRef();
			}
		}
	}
	
Exit:

	return( rc);
}
	
/***************************************************************************
Desc:	Evaluate a query expression.
***************************************************************************/
RCODE F_Query::evalExpr(
	IF_DOMNode *		pContextNode,
	FLMBOOL				bForward,
	FLMBOOL				bUseKeyNodes,
	FQNODE *				pQueryExpr,
	FLMBOOL *			pbPassed,
	IF_DOMNode **		ppNode
	)
{
	RCODE			rc = NE_XFLM_OK;
	FQNODE *		pCurrNode = pQueryExpr;
	FLMBOOL		bXPathIsEntireExpr = FALSE;
	FLMBYTE		ucDynaBuf[ 64];
	F_DynaBuf	dynaBuf( ucDynaBuf, sizeof( ucDynaBuf));
	FLMBOOL		bGetNodeValue;

	// IMPORTANT NOTE: A non-NULL ppNode should only be passed in for the
	// very outermost evalExpr, not for nested ones.  We use this to determine
	// whether to count documents read and documents passed.

	if (ppNode && *ppNode)
	{
		(*ppNode)->Release();
		*ppNode = NULL;
	}
	if (pbPassed)
	{
		*pbPassed = FALSE;
	}

	// If the query is empty, it passes

	if (!pQueryExpr)
	{
		if (pbPassed)
		{
			*pbPassed = TRUE;
		}
		if (ppNode)
		{
			*ppNode = m_pCurrDoc;
			(*ppNode)->AddRef();
		}
		goto Exit;
	}

	if (pQueryExpr->eNodeType == FLM_XPATH_NODE &&
		 !pQueryExpr->nd.pXPath->pFirstComponent->pXPathContext)
	{
		bXPathIsEntireExpr = TRUE;
	}
	fqResetQueryTree( pQueryExpr, bUseKeyNodes, m_bResetAllXPaths);
	m_bResetAllXPaths = FALSE;

	// Perform the evaluation

	pCurrNode = pQueryExpr;
	for (;;)
	{
		while (pCurrNode->pFirstChild)
		{
			pCurrNode = pCurrNode->pFirstChild;
		}

		// We should be positioned on a leaf node that is either a
		// value, a function, or an xpath.

		bGetNodeValue = FALSE;
		if (pCurrNode->eNodeType == FLM_VALUE_NODE)
		{
			if (!pCurrNode->pParent)
			{
				pCurrNode = NULL;
				break;
			}
			if (pCurrNode->bUsedValue)
			{
				pCurrNode = fqBackupTree( pCurrNode, &bGetNodeValue);
			}
			else
			{
				pCurrNode->bUsedValue = TRUE;
			}
		}
		else if (pCurrNode->eNodeType == FLM_FUNCTION_NODE)
		{
			if (RC_BAD( rc = getFuncValue( pContextNode, bForward,
											&pCurrNode, &bGetNodeValue,
											&dynaBuf)))
			{
				goto Exit;
			}
		}
		else
		{
			
			// Better be an xpath at this point

			flmAssert( pCurrNode->eNodeType == FLM_XPATH_NODE);
			if (RC_BAD( rc = getXPathValue( pContextNode, bForward, &pCurrNode,
										&bGetNodeValue, bUseKeyNodes,
										bXPathIsEntireExpr)))
			{
				goto Exit;
			}
		}
		if (!pCurrNode)
		{
			break;
		}
		if (bGetNodeValue)
		{
			continue;
		}

		// When we get to this point, we have at least one leaf
		// level operand in hand - pCurrNode.
		// See if we can evaluate the operator of pCurrNode.
		// This will take care of any short-circuiting evaluation
		// that can be done.
		
		if (RC_BAD( rc = fqTryEvalOperator( m_uiLanguage, &pCurrNode)))
		{
			goto Exit;
		}
		if (!pCurrNode)
		{
			break;
		}
	}
	
	if (RC_BAD( rc = setExprReturnValue( bUseKeyNodes, pQueryExpr,
								pbPassed, ppNode)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Release the resources of a query expression
***************************************************************************/
FSTATIC void fqReleaseQueryExpr(
	FQNODE *	pQNode
	)
{
	// Reset the entire query tree.  It could have been left in a partial
	// evaluation state from the last document evaluated.

	for (;;)
	{
		fqReleaseNodeValue( pQNode);
		pQNode->bUsedValue = FALSE;
		pQNode->bLastValue = FALSE;
		if (pQNode->pFirstChild)
		{
			pQNode = pQNode->pFirstChild;
		}
		else
		{
			if (pQNode->eNodeType == FLM_XPATH_NODE)
			{
				fqResetIterator( pQNode, TRUE, FALSE);
			}
			while (!pQNode->pNextSib)
			{
				if ((pQNode = pQNode->pParent) == NULL)
				{
					break;
				}
			}
			if (!pQNode)
			{
				break;
			}
			pQNode = pQNode->pNextSib;
		}
	}
}

/***************************************************************************
Desc:	Release the resources of a query
***************************************************************************/
void XFLAPI F_Query::resetQuery( void)
{
	if (m_pQuery)
	{
		fqReleaseQueryExpr( m_pQuery);
	}
	
	m_eState = XFLM_QUERY_NOT_POSITIONED;
	
	if (m_pCurrDoc)
	{
		m_pCurrDoc->Release();
		m_pCurrDoc = NULL;
	}
	
	if (m_pCurrNode)
	{
		m_pCurrNode->Release();
		m_pCurrNode = NULL;
	}
}

/***************************************************************************
Desc:	Get leaf predicate for the current context, changing current
		context, context path, and predicate as needed.
***************************************************************************/
void F_Query::useLeafContext(
	FLMBOOL	bGetFirst)
{
	for (;;)
	{
		if (m_pCurrContext->bIntersect)
		{
			if (m_pCurrContext->pSelectedChild)
			{
				flmAssert( !m_pCurrContext->pSelectedPath);
				m_pCurrContext = m_pCurrContext->pSelectedChild;
			}
			else
			{
				flmAssert( m_pCurrContext->pSelectedPath);
				m_pCurrContextPath = m_pCurrContext->pSelectedPath;
				m_pCurrPred = m_pCurrContextPath->pSelectedPred;
				flmAssert( m_pCurrContextPath && m_pCurrPred);
				m_pCurrOpt = &m_pCurrPred->OptInfo;
				break;
			}
		}
		else
		{

			// In a non-intersect context, pSelectedChild should NOT have
			// been set.  Nor should pSelectedPath.

			flmAssert( !m_pCurrContext->pSelectedChild);
			flmAssert( !m_pCurrContext->pSelectedPath);

			if (bGetFirst)
			{
				if (m_pCurrContext->pFirstChild)
				{
					m_pCurrContext = m_pCurrContext->pFirstChild;
				}
				else
				{
					m_pCurrContextPath = m_pCurrContext->pFirstPath;
					flmAssert( m_pCurrContextPath);

					// In a non-intersect context, pSelectedPred should NOT have
					// been set.

					flmAssert( !m_pCurrContextPath->pSelectedPred);
					m_pCurrPred = m_pCurrContextPath->pFirstPred;
					flmAssert( m_pCurrPred);
					m_pCurrOpt = &m_pCurrPred->OptInfo;
					break;
				}
			}
			else
			{
				if (m_pCurrContext->pLastChild)
				{
					m_pCurrContext = m_pCurrContext->pLastChild;
				}
				else
				{
					m_pCurrContextPath = m_pCurrContext->pLastPath;
					flmAssert( m_pCurrContextPath);

					// In a non-intersect context, pSelectedPred should NOT have
					// been set.

					flmAssert( !m_pCurrContextPath->pSelectedPred);
					m_pCurrPred = m_pCurrContextPath->pLastPred;
					flmAssert( m_pCurrPred);
					m_pCurrOpt = &m_pCurrPred->OptInfo;
					break;
				}
			}
		}
	}
}

/***************************************************************************
Desc:	Get next predicate to evaluate for a query.
***************************************************************************/
FLMBOOL F_Query::useNextPredicate( void)
{
	FLMBOOL	bGotNext = FALSE;

	// If we are in a non-intersecting context, exhaust its
	// context paths and predicates before going up a level.

	if (!m_pCurrContext->bIntersect)
	{
		if (m_pCurrPred)
		{
			if (m_pCurrPred->pNext)
			{
				m_pCurrPred = m_pCurrPred->pNext;
				m_pCurrOpt = &m_pCurrPred->OptInfo;
				bGotNext = TRUE;
				goto Exit;
			}
			if (m_pCurrContextPath->pNext)
			{
				m_pCurrContextPath = m_pCurrContextPath->pNext;
	
				// In a non-intersect context, pSelectedPred should NOT have
				// been set.
	
				flmAssert( !m_pCurrContextPath->pSelectedPred);
				m_pCurrPred = m_pCurrContextPath->pFirstPred;
				m_pCurrOpt = &m_pCurrPred->OptInfo;
				bGotNext = TRUE;
				goto Exit;
			}
		}

		// Go up one context level, if there is one.

		if (!m_pCurrContext->pParent)
		{
			goto Exit;
		}

		// Parent better be an intersecting context

		m_pCurrContext = m_pCurrContext->pParent;
		flmAssert( m_pCurrContext->bIntersect);
	}

	// Go back up the context tree to get next context.

	for (;;)
	{
		OP_CONTEXT *	pParent = m_pCurrContext->pParent;

		// If there is no parent context, we are done.

		if (!pParent)
		{
			break;
		}
		if (m_pCurrContext->bIntersect)
		{

			// Parent context should be non-intersecting, so we should
			// go to the sibling context, if there is one.

			flmAssert( !pParent->bIntersect);
			if (m_pCurrContext->pNextSib)
			{
				m_pCurrContext = m_pCurrContext->pNextSib;

				// Travel down this part of the context tree to get the
				// "leaf-most" context, context path, and predicate.

				useLeafContext( TRUE);
				bGotNext = TRUE;
				break;
			}
		}
		else
		{
			if (m_pCurrContext->pFirstPath)
			{
				m_pCurrContextPath = m_pCurrContext->pFirstPath;

				// In a non-intersect context, pSelectedPred should NOT have
				// been set.

				flmAssert( !m_pCurrContextPath->pSelectedPred);
				m_pCurrPred = m_pCurrContextPath->pFirstPred;
				m_pCurrOpt = &m_pCurrPred->OptInfo;
				bGotNext = TRUE;
				goto Exit;
			}

			// Parent context better be pointing to this context as its
			// "selected" context.

			flmAssert( pParent->pSelectedChild == m_pCurrContext);

			// Parent also better be an intersecting context.

			flmAssert( pParent->bIntersect);
		}
		m_pCurrContext = pParent;
	}

Exit:

	return( bGotNext);
}

/***************************************************************************
Desc:	Get previous predicate to evaluate for a query.
***************************************************************************/
FLMBOOL F_Query::usePrevPredicate( void)
{
	FLMBOOL	bGotPrev = FALSE;

	// If we are in a non-intersecting context, exhaust its
	// context paths and predicates before going up a level.

	if (!m_pCurrContext->bIntersect)
	{
		if (m_pCurrPred)
		{
			if (m_pCurrPred->pPrev)
			{
				m_pCurrPred = m_pCurrPred->pPrev;
				m_pCurrOpt = &m_pCurrPred->OptInfo;
				bGotPrev = TRUE;
				goto Exit;
			}
			if (m_pCurrContextPath->pPrev)
			{
				m_pCurrContextPath = m_pCurrContextPath->pPrev;
	
				// In a non-intersect context, pSelectedPred should NOT have
				// been set.
	
				flmAssert( !m_pCurrContextPath->pSelectedPred);
				m_pCurrPred = m_pCurrContextPath->pLastPred;
				m_pCurrOpt = &m_pCurrPred->OptInfo;
				bGotPrev = TRUE;
				goto Exit;
			}
		}

		// Go up one context level, if there is one.

		if (!m_pCurrContext->pParent)
		{
			goto Exit;
		}

		// Parent better be an intersecting context

		m_pCurrContext = m_pCurrContext->pParent;
		flmAssert( m_pCurrContext->bIntersect);
	}

	// Go back up the context tree to get previous context.

	for (;;)
	{
		OP_CONTEXT *	pParent = m_pCurrContext->pParent;

		// If there is no parent context, we are done.

		if (!pParent)
		{
			break;
		}
		if (m_pCurrContext->bIntersect)
		{

			// Parent context should be non-intersecting, so we should
			// go to the sibling context, if there is one.

			flmAssert( !pParent->bIntersect);
			if (m_pCurrContext->pPrevSib)
			{
				m_pCurrContext = m_pCurrContext->pPrevSib;

				// Travel down this part of the context tree to get the
				// "leaf-most" context, context path, and predicate.

				useLeafContext( FALSE);
				bGotPrev = TRUE;
				break;
			}
		}
		else
		{
			if (m_pCurrContext->pLastPath)
			{
				m_pCurrContextPath = m_pCurrContext->pLastPath;

				// In a non-intersect context, pSelectedPred should NOT have
				// been set.

				flmAssert( !m_pCurrContextPath->pSelectedPred);
				m_pCurrPred = m_pCurrContextPath->pLastPred;
				m_pCurrOpt = &m_pCurrPred->OptInfo;
				bGotPrev = TRUE;
				goto Exit;
			}

			// Parent context better be pointing to this context as its
			// "selected" context.

			flmAssert( pParent->pSelectedChild == m_pCurrContext);

			// Parent also better be an intersecting context.

			flmAssert( pParent->bIntersect);
		}
		m_pCurrContext = pParent;
	}

Exit:

	return( bGotPrev);
}

/***************************************************************************
Desc:	Get the next/previous node for an application predicate.
***************************************************************************/
RCODE F_Query::getAppNode(
	FLMBOOL *			pbFirstLast,
	FLMBOOL				bForward,
	XPATH_COMPONENT *	pXPathComponent
	)
{
	RCODE						rc = NE_XFLM_OK;
	IF_QueryNodeSource *	pNodeSource = m_pCurrPred->pNodeSource;
	FLMUINT					uiCurrTime;
	FLMUINT					uiElapsedTime;
	FLMUINT					uiTimeLimit = m_uiTimeLimit;
	FLMUINT64				ui64DocId;

	for (;;)
	{
		
		// Reset the timeout value everytime through the loop, if it
		// is non-zero.
		
		if (uiTimeLimit)
		{
			uiCurrTime = FLM_GET_TIMER();
			uiElapsedTime = FLM_ELAPSED_TIME( uiCurrTime, m_uiStartTime);
			if (uiElapsedTime >= m_uiTimeLimit)
			{
				rc = RC_SET( NE_XFLM_TIMEOUT);
				goto Exit;
			}
			else
			{
				uiTimeLimit = FLM_TIMER_UNITS_TO_MILLI( (m_uiTimeLimit - uiElapsedTime));
				
				// Always give at least one milli-second.
				
				if (!uiTimeLimit)
				{
					uiTimeLimit = 1;
				}
			}
		}
	
		if (pXPathComponent->pKeyNode)
		{
			pXPathComponent->pKeyNode->Release();
			pXPathComponent->pKeyNode = NULL;
		}

		// Get the next or previous key from the index.

		if (bForward)
		{
			rc = (RCODE)(*pbFirstLast
							 ? pNodeSource->getFirst( (IF_Db *)m_pDb, NULL,
							 						&pXPathComponent->pKeyNode,
							 						uiTimeLimit, m_pQueryStatus)
							 : pNodeSource->getNext( (IF_Db *)m_pDb, NULL,
							 						&pXPathComponent->pKeyNode,
							 						uiTimeLimit, m_pQueryStatus));

			if (RC_BAD( rc))
			{
				if (rc == NE_XFLM_EOF_HIT)
				{
					rc = NE_XFLM_OK;
				}
				goto Exit;
			}
		}
		else
		{
			rc = (RCODE)(*pbFirstLast
							 ? pNodeSource->getLast( (IF_Db *)m_pDb, NULL,
							 						&pXPathComponent->pKeyNode,
							 						uiTimeLimit, m_pQueryStatus)
							 : pNodeSource->getPrev( (IF_Db *)m_pDb, NULL,
							 						&pXPathComponent->pKeyNode,
							 						uiTimeLimit, m_pQueryStatus));
			if (RC_BAD( rc))
			{
				if (rc == NE_XFLM_BOF_HIT)
				{
					rc = NE_XFLM_OK;
				}
				goto Exit;
			}
		}
		*pbFirstLast = FALSE;

		// If we are eliminating duplicates, see if the document
		// has already been processed.  If so, skip the key.
		
		if (RC_BAD( rc = pXPathComponent->pKeyNode->getDocumentId( (IF_Db *)m_pDb,
								&ui64DocId)))
		{
			goto Exit;
		}
		if (m_pDocIdSet)
		{
			if (RC_BAD( rc = m_pDocIdSet->findMatch( &ui64DocId, NULL)))
			{
				if (rc != NE_FLM_NOT_FOUND)
				{
					goto Exit;
				}
				rc = NE_XFLM_OK;
			}
			else
			{

				// Document has already been passed, go to next/prev key.

				m_pCurrOpt->ui64KeyHadDupDoc++;
				if (RC_BAD( rc = queryStatus()))
				{
					goto Exit;
				}
				continue;
			}
		}
		
		// At this point, the key has passed at least the predicate function.
		// Get the document node.
	
		if (RC_BAD( rc = m_pDb->getNode( m_uiCollection, ui64DocId, &m_pCurrDoc)))
		{
			goto Exit;
		}
		
		// Found one!
		
		break;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get an FQVALUE from a key.
***************************************************************************/
FSTATIC RCODE fqGetValueFromKey(
	FLMUINT			uiDataType,
	F_DataVector *	pKey,
	FQVALUE *		pQValue,
	FLMBYTE **		ppucValue,
	FLMUINT			uiValueBufSize
	)
{
	RCODE	rc = NE_XFLM_OK;

	pQValue->uiFlags = 0;
	switch (uiDataType)
	{
		case XFLM_NODATA_TYPE:

			// Should have been set to missing on the outside

			flmAssert( pQValue->eValType == XFLM_MISSING_VAL);
			pQValue->eValType = XFLM_BOOL_VAL;
			pQValue->val.eBool = XFLM_TRUE;
			break;
		case XFLM_TEXT_TYPE:
			pQValue->uiDataLen = pKey->getDataLength( 0) + 1;
			if (pQValue->uiDataLen > uiValueBufSize)
			{
				if (RC_BAD( rc = f_alloc( pQValue->uiDataLen, ppucValue)))
				{
					goto Exit;
				}
			}
			pQValue->val.pucBuf = *ppucValue;
			if (RC_BAD( rc = pKey->getUTF8( 0, pQValue->val.pucBuf,
										&pQValue->uiDataLen)))
			{
				goto Exit;
			}
			pQValue->eValType = XFLM_UTF8_VAL;
			break;
		case XFLM_NUMBER_TYPE:

			// First, see if we can get it as a UINT - the most common
			// type.

			if (RC_OK( rc = pKey->getUINT( 0, &pQValue->val.uiVal)))
			{
				pQValue->eValType = XFLM_UINT_VAL;
			}
			else if (rc == NE_XFLM_CONV_NUM_OVERFLOW)
			{
				if (RC_BAD( rc = pKey->getUINT64( 0,
												&pQValue->val.ui64Val)))
				{
					goto Exit;
				}
				pQValue->eValType = XFLM_UINT64_VAL;
			}
			else if (rc == NE_XFLM_CONV_NUM_UNDERFLOW)
			{
				if (RC_OK( rc = pKey->getINT( 0, &pQValue->val.iVal)))
				{
					pQValue->eValType = XFLM_INT_VAL;
				}
				else if (rc == NE_XFLM_CONV_NUM_UNDERFLOW)
				{
					if (RC_BAD( rc = pKey->getINT64( 0,
														&pQValue->val.i64Val)))
					{
						goto Exit;
					}
					pQValue->eValType = XFLM_INT64_VAL;
				}
			}
			else
			{
				goto Exit;
			}
			break;
		case XFLM_BINARY_TYPE:
			pQValue->uiDataLen = pKey->getDataLength( 0) + 1;
			if (pQValue->uiDataLen > uiValueBufSize)
			{
				if (RC_BAD( rc = f_alloc( pQValue->uiDataLen, ppucValue)))
				{
					goto Exit;
				}
			}
			pQValue->val.pucBuf = *ppucValue;
			if (RC_BAD( rc = pKey->getBinary( 0, pQValue->val.pucBuf,
										&pQValue->uiDataLen)))
			{
				goto Exit;
			}
			pQValue->eValType = XFLM_BINARY_VAL;
			break;
		default:
			rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
			goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Test a value against the specified predicate.
***************************************************************************/
FSTATIC RCODE fqPredCompare(
	FLMUINT		uiLanguage,
	PATH_PRED *	pPred,
	FQVALUE *	pQValue,
	FLMBOOL *	pbPasses
	)
{
	RCODE				rc = NE_XFLM_OK;
	XFlmBoolType	eBool;

	switch (pPred->eOperator)
	{
		case XFLM_EXISTS_OP:

			// We already know this one passes.

			*pbPasses = TRUE;
			goto Exit;
		case XFLM_NE_OP:
		case XFLM_APPROX_EQ_OP:
			if (RC_BAD( rc = fqCompareOperands( uiLanguage,
										pQValue,
										pPred->pFromValue,
										pPred->eOperator,
										pPred->uiCompareRules,
										pPred->pOpComparer,
										pPred->bNotted,
										&eBool)))
			{
				goto Exit;
			}
			break;
		case XFLM_MATCH_OP:

			// From value of predicate better be a constant
			// with wildcards set.

			flmAssert( pPred->pFromValue->uiFlags & VAL_IS_CONSTANT);
			flmAssert( pPred->pFromValue->uiFlags & VAL_HAS_WILDCARDS);
			if (RC_BAD( rc = fqCompareOperands( uiLanguage,
										pQValue,
										pPred->pFromValue,
										XFLM_EQ_OP,
										pPred->uiCompareRules,
										pPred->pOpComparer,
										pPred->bNotted,
										&eBool)))
			{
				goto Exit;
			}
			break;
		case XFLM_RANGE_OP:
			eBool = XFLM_TRUE;

			// Take care of ==, > and >=

			if (pPred->pFromValue)
			{
				eQueryOperators	eOperator;

				if (pPred->pUntilValue == pPred->pFromValue)
				{
					eOperator = XFLM_EQ_OP;
				}
				else
				{
					eOperator = pPred->bInclFrom ? XFLM_GE_OP : XFLM_GT_OP;
				}
				if (RC_BAD( rc = fqCompareOperands( uiLanguage,
											pQValue,
											pPred->pFromValue,
											eOperator,
											pPred->uiCompareRules,
											pPred->pOpComparer,
											pPred->bNotted,
											&eBool)))
				{
					goto Exit;
				}
			}

			// Take care of < and <= if we are still TRUE

			if (eBool == XFLM_TRUE && pPred->pUntilValue &&
				 pPred->pUntilValue != pPred->pFromValue)
			{
				if (RC_BAD( rc = fqCompareOperands( uiLanguage,
											pQValue,
											pPred->pUntilValue,
											pPred->bInclUntil ? XFLM_LE_OP : XFLM_LT_OP,
											pPred->uiCompareRules,
											pPred->pOpComparer,
											pPred->bNotted,
											&eBool)))
				{
					goto Exit;
				}
			}
			break;
		default:
			*pbPasses = FALSE;
			rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
			goto Exit;
	}

	// If the value didn't pass the predicate, we should return
	// a FALSE in *pbPasses.

	*pbPasses = (eBool == XFLM_FALSE) ? FALSE : TRUE;

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Test the key against the specified predicate.
***************************************************************************/
RCODE F_Query::testKey(
	F_DataVector *	pKey,
	PATH_PRED *		pPred,
	FLMBOOL *		pbPasses,
	IF_DOMNode **	ppPassedNode)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT64	ui64NodeId;
	FQVALUE		currVal;
	FLMUINT		uiDataType;
	FLMBYTE		ucKeyValue [128];
	FLMBYTE *	pucKeyValue = &ucKeyValue [0];

	currVal.eValType = XFLM_MISSING_VAL;

	// At this point, all use of notted predicates should have been
	// weeded out.

	flmAssert( !pPred->bNotted);
	*pbPasses = TRUE;

	// First see if the key passes the criteria of the predicate.
	// If we can use the key, use it.  Otherwise, fetch the node
	// and do the comparison.

	if (pPred->OptInfo.bCanCompareOnKey)
	{
		// No need to retrieve the data if it is an exists operator

		if (pPred->eOperator != XFLM_EXISTS_OP)
		{
			if ((uiDataType = pKey->getDataType( 0)) == XFLM_UNKNOWN_TYPE)
			{

				// No data - component was missing.

				*pbPasses = FALSE;
				goto Exit;
			}

			if (RC_BAD( rc = fqGetValueFromKey( uiDataType, pKey, &currVal,
										&pucKeyValue, sizeof( ucKeyValue))))
			{
				goto Exit;
			}

			// Compare on the key

			if (RC_BAD( rc = fqPredCompare( m_uiLanguage, pPred,
												&currVal, pbPasses)))
			{
				goto Exit;
			}
			if (!(*pbPasses))
			{
				goto Exit;
			}
		}
	}

	// Have to get the node whether we compare on it or not

	if ((ui64NodeId = pKey->getID( 0)) == 0)
	{

		// No data - component was missing.

		*pbPasses = FALSE;
		goto Exit;
	}
	
	if( pKey->isAttr( 0))
	{
		rc = m_pDb->getAttribute( m_uiCollection, 
			ui64NodeId, pKey->getNameId( 0), ppPassedNode);
	}
	else
	{
		rc = m_pDb->getNode( m_uiCollection, ui64NodeId, ppPassedNode);
	}
	
	if( RC_BAD( rc))
	{
		if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
		}
		
		goto Exit;
	}
	
	if (RC_BAD( rc = incrNodesRead()))
	{
		goto Exit;
	}

	// See if we need to do the comparison on the node.

	if (pPred->OptInfo.bDoNodeMatch && pPred->eOperator != XFLM_EXISTS_OP)
	{
		if (RC_BAD( rc = fqGetValueFromNode( m_pDb, *ppPassedNode, &currVal, 0)))
		{
			goto Exit;
		}

		// Do the comparison

		if (RC_BAD( rc = fqPredCompare( m_uiLanguage, pPred,
									&currVal, pbPasses)))
		{
			goto Exit;
		}

		if (!(*pbPasses))
		{
			goto Exit;
		}
	}

	// At this point, the key has passed at least the predicate comparison.
	// Get the document node
	
	if (RC_BAD( rc = m_pDb->getNode( m_uiCollection,
							pKey->getDocumentID(), &m_pCurrDoc)))
	{
		if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
		{

			// If we cannot retrieve the node, we have a corruption
			// in the database.

			rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
		}
		goto Exit;
	}

	if (RC_BAD( rc = incrNodesRead()))
	{
		goto Exit;
	}

Exit:

	if (pucKeyValue != &ucKeyValue [0])
	{
		f_free( &pucKeyValue);
	}

	if ((currVal.eValType == XFLM_BINARY_VAL ||
		  currVal.eValType == XFLM_UTF8_VAL) &&
		 (currVal.uiFlags & VAL_IS_STREAM) &&
		 currVal.val.pIStream)
	{
		currVal.val.pIStream->Release();
	}
	return( rc);
}

/***************************************************************************
Desc:	Mark all of the XPATH nodes that are the same as the one in
		pXPathComponent as having a passing value.
***************************************************************************/
FSTATIC void fqMarkXPathNodeListPassed(
	PATH_PRED *	pPred
	)
{
	PATH_PRED_NODE *		pXPathNodeList;

	pXPathNodeList = pPred->pXPathNodeList;
	while (pXPathNodeList)
	{
		pXPathNodeList->pXPathNode->nd.pXPath->bHavePassingNode = TRUE;
		pXPathNodeList = pXPathNodeList->pNext;
	}
}

/***************************************************************************
Desc:	Get the next/previous key for an xpath component.
***************************************************************************/
RCODE F_Query::getKey(
	FLMBOOL *			pbFirstLast,
	FLMBOOL				bForward,
	XPATH_COMPONENT *	pXPathComponent
	)
{
	RCODE					rc = NE_XFLM_OK;
	PATH_PRED *			pPred = pXPathComponent->pOptPred;
	FSIndexCursor *	pFSIndexCursor = pPred->pFSIndexCursor;
	F_DataVector		key;
	FLMBOOL				bPassed;
	FLMBOOL				bSkipCurrKey = FALSE;
	FLMBOOL				bHasContextPosTest = hasContextPosTest( pXPathComponent);

	// Better be rightmost xpath component.

	flmAssert( !pXPathComponent->pNext);

	// Component could have an expression, but it should not be one
	// we are optimizing on this time around.

	flmAssert( !pXPathComponent->pExprXPathSource);

	for (;;)
	{

		// Release the current key node, if any

		if (pXPathComponent->pKeyNode)
		{
			pXPathComponent->pKeyNode->Release();
			pXPathComponent->pKeyNode = NULL;
		}

		// Get the next or previous key from the index.

		if (bForward)
		{
			rc = (RCODE)(*pbFirstLast
							 ? pFSIndexCursor->firstKey( m_pDb, &key)
							 : pFSIndexCursor->nextKey( m_pDb, &key, bSkipCurrKey));

			if (RC_BAD( rc))
			{
				if (rc == NE_XFLM_EOF_HIT)
				{
					rc = NE_XFLM_OK;
				}
				goto Exit;
			}
		}
		else
		{
			rc = (RCODE)(*pbFirstLast
							 ? pFSIndexCursor->lastKey( m_pDb, &key)
							 : pFSIndexCursor->prevKey( m_pDb, &key, bSkipCurrKey));
			if (RC_BAD( rc))
			{
				if (rc == NE_XFLM_BOF_HIT)
				{
					rc = NE_XFLM_OK;
				}
				goto Exit;
			}
		}
		*pbFirstLast = FALSE;
		m_pCurrOpt->ui64KeysRead++;
		if (RC_BAD( rc = queryStatus()))
		{
			goto Exit;
		}

		// If we are eliminating duplicates, see if the document
		// has already been processed.  If so, skip the key.

		if (m_pDocIdSet)
		{
			FLMUINT64	ui64DocId = key.getDocumentID();

			if (RC_BAD( rc = m_pDocIdSet->findMatch( &ui64DocId, NULL)))
			{
				if (rc != NE_FLM_NOT_FOUND)
				{
					goto Exit;
				}
				rc = NE_XFLM_OK;
			}
			else
			{

				// Document has already been passed, go to next/prev key.

				m_pCurrOpt->ui64KeyHadDupDoc++;
				if (RC_BAD( rc = queryStatus()))
				{
					goto Exit;
				}
				continue;
			}
		}

		// Evaluate the key.

		if (RC_BAD( rc = testKey( &key, pPred, &bPassed,
								&pXPathComponent->pKeyNode)))
		{
			goto Exit;
		}
		if (!bPassed)
		{
			// If this is a case-sensitive index or a case-insensitive compare,
			// we can skip the key.  Otherwise, we cannot skip any keys.

			if (!(pFSIndexCursor->m_pIxd->pFirstKey->uiCompareRules &
						XFLM_COMP_CASE_INSENSITIVE) ||
				  (pPred->uiCompareRules & XFLM_COMP_CASE_INSENSITIVE))
			{
				bSkipCurrKey = TRUE;
			}
			continue;
		}
		bSkipCurrKey = FALSE;
		m_pCurrOpt->ui64KeysPassed++;
		if (RC_BAD( rc = queryStatus()))
		{
			goto Exit;
		}

		// If the xpath component has an expression, evaluate it

		if (pXPathComponent->pExpr)
		{
			if (RC_BAD( rc = evalExpr( pXPathComponent->pKeyNode,
										bForward, TRUE, pXPathComponent->pExpr,
										&bPassed, NULL)))
			{
				goto Exit;
			}
			if (!bPassed)
			{
				continue;
			}
		}

		if (bHasContextPosTest)
		{

			// Need to verify the context position of the found node.

			if (RC_BAD( rc = verifyOccurrence( FALSE, pXPathComponent,
							pXPathComponent->pKeyNode, &bPassed)))
			{
				goto Exit;
			}
			if (!bPassed)
			{
				continue;
			}
		}

		// There may be more than one XPATH that this predicate
		// covers.  Set them up so they don't have to be evaluated
		// if at all possible.

		fqMarkXPathNodeListPassed( pPred);
		break;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Test a node's metadata against the specified predicate.
***************************************************************************/
RCODE F_Query::testMetaData(
	IF_DOMNode *	pNode,
	FLMUINT			uiMetaDataType,
	PATH_PRED *		pPred,
	FLMBOOL *		pbPasses
	)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT64	ui64DocId;
	FQVALUE		currVal;

	currVal.eValType = XFLM_MISSING_VAL;

	// At this point, all use of notted predicates should have been
	// weeded out.

	flmAssert( !pPred->bNotted);
	*pbPasses = TRUE;

	// See if we need to do the comparison on the node.

	if (pPred->eOperator != XFLM_EXISTS_OP)
	{
		if (RC_BAD( rc = fqGetValueFromNode( m_pDb, pNode, &currVal,
									uiMetaDataType)))
		{
			goto Exit;
		}

		// Do the comparison

		if (RC_BAD( rc = fqPredCompare( m_uiLanguage, pPred,
									&currVal, pbPasses)))
		{
			goto Exit;
		}

		if (!(*pbPasses))
		{
			goto Exit;
		}
	}

	// At this point, the node has passed at least the predicate comparison.
	// Get the document node, if we don't already have it.

	if( !(((F_DOMNode *)pNode)->isRootNode()))
	{
		if (RC_BAD( rc = pNode->getDocumentId( m_pDb, &ui64DocId)))
		{
			goto Exit;
		}
		
		if (RC_BAD( rc = m_pDb->getNode( m_uiCollection,
								ui64DocId, &m_pCurrDoc)))
		{
			if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{

				// If we cannot retrieve the node, we have a corruption
				// in the database.

				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			}
			goto Exit;
		}
	}
	else
	{
		m_pCurrDoc = pNode;
		m_pCurrDoc->AddRef();
	}

	if (RC_BAD( rc = incrNodesRead()))
	{
		goto Exit;
	}

Exit:

	if ((currVal.eValType == XFLM_BINARY_VAL ||
		  currVal.eValType == XFLM_UTF8_VAL) &&
		 (currVal.uiFlags & VAL_IS_STREAM) &&
		 currVal.val.pIStream)
	{
		currVal.val.pIStream->Release();
	}
	return( rc);
}

/***************************************************************************
Desc:	Get the next/previous node for an xpath component.
***************************************************************************/
RCODE F_Query::getANode(
	FLMBOOL *			pbFirstLast,
	FLMBOOL				bForward,
	XPATH_COMPONENT *	pXPathComponent
	)
{
	RCODE						rc = NE_XFLM_OK;
	PATH_PRED *				pPred = pXPathComponent->pOptPred;
	FSCollectionCursor *	pFSCollectionCursor = pPred->pFSCollectionCursor;
	FLMBOOL					bPassed;
	IF_DOMNode *			pNode = NULL;
	FLMUINT64				ui64NodeId;
	FLMBOOL					bHasContextPosTest = hasContextPosTest( pXPathComponent);

	// Better be rightmost xpath component.

	flmAssert( !pXPathComponent->pNext);

	// Component could have an expression, but it should not be one
	// we are optimizing on this time around.

	flmAssert( !pXPathComponent->pExprXPathSource);

	for (;;)
	{
		if (pNode)
		{
			pNode->Release();
			pNode = NULL;
		}

		// See if we need to continue from the current node in the current
		// document, or release it and get another node.

		if ((pNode = pXPathComponent->pKeyNode) != NULL)
		{
			pXPathComponent->pKeyNode = NULL;

			// No need to do pNode->AddRef(), because we will just
			// steal the AddRef that was on pCurrNode.

			if (getMetaDataType( pXPathComponent) != XFLM_META_DOCUMENT_ID)
			{
				pNode->Release();
				pNode = NULL;
			}
			else
			{

				// If we are doing document nodes, see if we can continue in the
				// document - because all of the nodes in the document should
				// be processed.

				if (RC_BAD( rc = walkDocument( bForward, TRUE, 0, &pNode)))
				{
					goto Exit;
				}
			}

			// If we didn't get a node back, and
			// If pFSCollectionCursor is NULL, it means we
			// are doing a single node, and we have already
			// done that single node (or document), so we can
			// quit.

			if (!pNode && !pFSCollectionCursor)
			{
				goto Exit;
			}
		}

		// Get the next or previous node from the collection, if we
		// didn't get a node from the loop above.

		if (!pNode)
		{
			if (pFSCollectionCursor)
			{
				if (bForward)
				{
					rc = (RCODE)(*pbFirstLast
									 ? pFSCollectionCursor->firstNode( m_pDb, &pNode,
																		&ui64NodeId)
									 : pFSCollectionCursor->nextNode( m_pDb, &pNode,
																		&ui64NodeId));
					if (RC_BAD( rc))
					{
						if (rc == NE_XFLM_EOF_HIT)
						{
							rc = NE_XFLM_OK;
						}
						goto Exit;
					}
				}
				else
				{
					rc = (RCODE)(*pbFirstLast
									 ? pFSCollectionCursor->lastNode( m_pDb, &pNode,
																		&ui64NodeId)
									 : pFSCollectionCursor->prevNode( m_pDb, &pNode,
																		&ui64NodeId));
					if (RC_BAD( rc))
					{
						if (rc == NE_XFLM_BOF_HIT)
						{
							rc = NE_XFLM_OK;
						}
						goto Exit;
					}
				}
			}
			else if (*pbFirstLast)
			{

				// Getting a single node.

				if (getMetaDataType( pXPathComponent) == XFLM_META_DOCUMENT_ID)
				{
					if (RC_BAD( rc = m_pDb->getDocument( m_uiCollection,
													XFLM_EXACT, pPred->OptInfo.ui64NodeId,
													&pNode)))
					{
						if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
						{
							rc = NE_XFLM_OK;
						}
						goto Exit;
					}
				}
				else
				{
					if (RC_BAD( rc = m_pDb->getNode( m_uiCollection,
													pPred->OptInfo.ui64NodeId,
													&pNode)))
					{
						if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
						{
							rc = NE_XFLM_OK;
						}
						goto Exit;
					}
				}
			}
			else
			{
				goto Exit;
			}
			*pbFirstLast = FALSE;
			m_pCurrOpt->ui64NodesRead++;
			if (RC_BAD( rc = queryStatus()))
			{
				goto Exit;
			}
		}

		// If we are eliminating duplicates, see if the document
		// has already been processed.  If so, skip the key.

		if (m_pDocIdSet)
		{
			FLMUINT64	ui64DocId;

			if (RC_BAD( rc = pNode->getDocumentId( m_pDb, &ui64DocId)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = m_pDocIdSet->findMatch( &ui64DocId, NULL)))
			{
				if (rc != NE_FLM_NOT_FOUND)
				{
					goto Exit;
				}
				rc = NE_XFLM_OK;
			}
			else
			{

				// Document has already been passed, go to next/prev node.

				m_pCurrOpt->ui64KeyHadDupDoc++;
				if (RC_BAD( rc = queryStatus()))
				{
					goto Exit;
				}
				continue;
			}
		}

		// Evaluate the node.

		if (RC_BAD( rc = testMetaData( pNode, getMetaDataType( pXPathComponent),
								pPred, &bPassed)))
		{
			goto Exit;
		}
		if (bPassed)
		{
			pXPathComponent->pKeyNode = pNode;
			pXPathComponent->pKeyNode->AddRef();
		}
		else
		{
			continue;
		}

		// If the xpath component has an expression, evaluate it

		if (pXPathComponent->pExpr)
		{
			if (RC_BAD( rc = evalExpr( pXPathComponent->pKeyNode,
										bForward, TRUE, pXPathComponent->pExpr,
										&bPassed, NULL)))
			{
				goto Exit;
			}
			if (!bPassed)
			{
				continue;
			}
		}

		if (bHasContextPosTest)
		{

			// Need to verify the context position of the found node.

			if (RC_BAD( rc = verifyOccurrence( FALSE, pXPathComponent,
							pXPathComponent->pKeyNode, &bPassed)))
			{
				goto Exit;
			}
			if (!bPassed)
			{
				continue;
			}
		}

		// There may be more than one XPATH that this predicate
		// covers.  Set them up so they don't have to be evaluated
		// if at all possible.

		fqMarkXPathNodeListPassed( pPred);
		break;
	}

Exit:

	if (pNode)
	{
		pNode->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:	Get a context node for the passed in xpath component.
***************************************************************************/
RCODE F_Query::getContextNode(
	FLMBOOL				bForward,
	XPATH_COMPONENT *	pXPathComponent
	)
{
	RCODE	rc = NE_XFLM_OK;

	// Component better be the left-most xpath component.

	flmAssert( !pXPathComponent->pPrev);

	// See if we can now get a node for the XPATH context
	// component.

	if (RC_BAD( rc = getXPathComponentFromAxis( pXPathComponent->pKeyNode,
								bForward, TRUE, pXPathComponent->pXPathContext,
								&pXPathComponent->pXPathContext->pKeyNode,
								invertedAxis( pXPathComponent->eXPathAxis),
								TRUE, FALSE)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get the next value for an XPATH component that has an expression
		that is the source for the query.
***************************************************************************/
RCODE F_Query::getNextIndexNode(
	FLMBOOL *			pbFirstLast,
	FLMBOOL				bForward,
	FQNODE *				pExprXPathSource,
	FLMBOOL				bSkipCurrKey)
{
	RCODE					rc = NE_XFLM_OK;
	FXPATH *				pSourceXPath;
	XPATH_COMPONENT *	pXPathComp;
	FLMUINT64			ui64NodeId;
	FLMUINT64			ui64Tmp;
	
	// This routine should only be called for components that have an
	// expression that is serving as the source for the query.
	// Both pExpr and pXPathSource must be non-NULL.

	flmAssert( pExprXPathSource->eNodeType == FLM_XPATH_NODE);

	pSourceXPath = pExprXPathSource->nd.pXPath;
	flmAssert( pSourceXPath->bIsSource);
	
	// Release all pCurrNodes that come AFTER the source component.
	// Start from last component and go backwards.
	
	pXPathComp = pSourceXPath->pLastComponent;
	while (pXPathComp)
	{
		if (pXPathComp->bIsSource)
		{
			break;
		}
		if (pXPathComp->pCurrNode)
		{
			pXPathComp->pCurrNode->Release();
			pXPathComp->pCurrNode = NULL;
		}
		pXPathComp = pXPathComp->pPrev;
	}
	
	// If this is the first time in, we need to go right to the
	// component in the XPath that is the key source.

	if (bSkipCurrKey)
	{

		// Release all of the DOM nodes up to and including the one
		// that is in the source component.  pXPathComp should be
		// pointing to the source component when we are done.

		pXPathComp = pSourceXPath->pFirstComponent;
		for (;;)
		{
			if (pXPathComp == pSourceXPath->pSourceComponent)
			{

				// Only release pKeyNode if we are not on the
				// optimization predicate.  Otherwise, we will
				// allow the call to getKey or getANode to do it.
				// This is mainly for the benefit of getANode, so it
				// can properly handle the document ID case, where it
				// needs to know the last node it was on inside a
				// document so it can continue walking from there.

				if (!pXPathComp->pOptPred && pXPathComp->pKeyNode)
				{
					pXPathComp->pKeyNode->Release();
					pXPathComp->pKeyNode = NULL;
				}
				break;
			}
			if (pXPathComp->pKeyNode)
			{
				pXPathComp->pKeyNode->Release();
				pXPathComp->pKeyNode = NULL;
			}
			pXPathComp = pXPathComp->pNext;
		}
		flmAssert( pXPathComp->bIsSource);
	}
	else if (!pSourceXPath->pFirstComponent->pKeyNode)
	{
		pXPathComp = pSourceXPath->pSourceComponent;
		flmAssert( pXPathComp->bIsSource);
	}
	else
	{
		pXPathComp = pSourceXPath->pFirstComponent;
	}
	for (;;)
	{
		if (pXPathComp->bIsSource)
		{
			if (pXPathComp->pOptPred)
			{
				if (pXPathComp->pOptPred->pFSIndexCursor)
				{
					if (RC_BAD( rc = getKey( pbFirstLast, bForward, pXPathComp)))
					{
						goto Exit;
					}
				}
				else if (pXPathComp->pOptPred->pNodeSource)
				{
					if (RC_BAD( rc = getAppNode( pbFirstLast, bForward, pXPathComp)))
					{
						goto Exit;
					}
				}
				else
				{
					
					// NOTE: pXPathComp->pOptPred->pFSCollectionCursor may be NULL
					// if getting a single node (eOptType == XFLM_QOPT_SINGLE_NODE_ID)
					
					if (RC_BAD( rc = getANode( pbFirstLast, bForward, pXPathComp)))
					{
						goto Exit;
					}
				}
				
				if (!pXPathComp->pKeyNode)
				{
					// Didn't get a node.
					// Cannot go any further than this - means we are
					// out of keys for this predicate.

					if (m_pCurrDoc)
					{
						m_pCurrDoc->Release();
						m_pCurrDoc = NULL;
					}
					
					goto Exit;
				}
			}
			else
			{
				flmAssert( pXPathComp->pExprXPathSource && pXPathComp->pExpr);

				// First see if we can get another context node without going
				// down to get another key.

				if (pXPathComp->pKeyNode)
				{

					// Got a node from expression, get context node for that node.

					if (RC_BAD( rc = getContextNode( bForward,
						pXPathComp->pExprXPathSource->nd.pXPath->pFirstComponent)))
					{
						goto Exit;
					}
				}
				while (!pXPathComp->pKeyNode)
				{
					if (RC_BAD( rc = getNextIndexNode( pbFirstLast, bForward,
												pXPathComp->pExprXPathSource,
												bSkipCurrKey)))
					{
						goto Exit;
					}

					// See if we got a node from below.  If not, there is nothing
					// more we can do at this level.

					if (!pXPathComp->pExprXPathSource->nd.pXPath->pFirstComponent->pKeyNode)
					{
						if (pXPathComp->pKeyNode)
						{
							pXPathComp->pKeyNode->Release();
							pXPathComp->pKeyNode = NULL;
						}
						goto Exit;
					}

					// Got a node from expression, get context node for that node.

					if (RC_BAD( rc = getContextNode( bForward,
							pXPathComp->pExprXPathSource->nd.pXPath->pFirstComponent)))
					{
						goto Exit;
					}
				}
			}
		}
		else
		{

			// There has to be a next to this node - because the source
			// component has to be somewhere after this component.

			flmAssert( pXPathComp->pNext);
			if (RC_BAD( rc = getXPathComponentFromAxis(
										pXPathComp->pNext->pKeyNode, bForward, TRUE,
										pXPathComp, &pXPathComp->pKeyNode,
										invertedAxis( pXPathComp->pNext->eXPathAxis),
										TRUE, FALSE)))
			{
				goto Exit;
			}

			// If we didn't get a node, go to next component and try to
			// get its next node.

			if (!pXPathComp->pKeyNode)
			{
				pXPathComp = pXPathComp->pNext;
				continue;
			}
		}

		// At this point we better have gotten a node.

		flmAssert( pXPathComp->pKeyNode);

		// If this is the left-most xpath component and we got a node
		// and the component's axis is the root axis, verify that the
		// node is, in fact the root node.

		if (!pXPathComp->pPrev && pXPathComp->eXPathAxis == ROOT_AXIS)
		{
			FLMUINT64		ui64DocId;
			
			if( RC_BAD( rc = m_pCurrDoc->getNodeId( m_pDb, &ui64DocId)))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = pXPathComp->pKeyNode->getNodeId( m_pDb, &ui64Tmp)))
			{
				goto Exit;
			}
			
			if( ui64DocId != ui64Tmp	)
			{

				if (RC_BAD( rc = pXPathComp->pKeyNode->getParentId(
																	m_pDb, &ui64NodeId)))
				{
					goto Exit;
				}
				
				if ( ui64NodeId != ui64DocId || m_pCurrDoc->getNodeType() != DOCUMENT_NODE)
				{
					continue;
				}
			}
		}

		// See if we can go to a previous component.  If not, we are done.

		if ((pXPathComp = pXPathComp->pPrev) == NULL)
		{
			break;
		}
	}
	
Exit:

	return( rc);
}

/***************************************************************************
Desc:	Set up to use the current predicate as the source for the query.
***************************************************************************/
RCODE F_Query::setupCurrPredicate(
	FLMBOOL	bForward
	)
{
	RCODE					rc = NE_XFLM_OK;
	FXPATH *				pXPath;
	XPATH_COMPONENT *	pXPathContextComponent;
	FLMBOOL				bFirstLast;

	// Clear all state information.

	resetQuery();

	if (RC_BAD( rc = newSource()))
	{
		goto Exit;
	}
	
	// m_pCurrPred has already been set.  We just need to set up each
	// XPATH appropriately.
	
	// Could be multiple XPATHs associated with the predicate, but it
	// is only necessary to set up one of them to point to the predicate.
	// We could pick any of them, but we take the first one in the list.
	
	pXPath = m_pCurrPred->pXPathNodeList->pXPathNode->nd.pXPath;
	pXPath->bIsSource = TRUE;
	
	// The source component for this XPATH will always be the
	// last component, because predicates are always optimized on
	// the last component of an XPATH.
	
	pXPath->pSourceComponent = pXPath->pLastComponent;
	pXPath->pSourceComponent->bIsSource = TRUE;
	pXPath->pSourceComponent->pOptPred = m_pCurrPred;

	// See if this XPATH is nested inside of another XPATH
	// component.  If so, that XPATH component must be marked
	// as the source, and its XPATH must also be marked as the
	// source.  That XPATH component must also point to this
	// particular XPATH node as the expression XPATH source for
	// the context component.
	
	m_pExprXPathSource = m_pCurrPred->pXPathNodeList->pXPathNode;
	pXPathContextComponent = pXPath->pSourceComponent->pXPathContext;
	while (pXPathContextComponent)
	{

		// Get the context component's XPATH

		pXPath = pXPathContextComponent->pXPathNode->nd.pXPath;
		pXPath->bIsSource = TRUE;
		pXPath->pSourceComponent = pXPathContextComponent;
		pXPathContextComponent->bIsSource = TRUE;
		pXPathContextComponent->pExprXPathSource = m_pExprXPathSource;

		// Setup for the next higher context, if any

		m_pExprXPathSource = pXPathContextComponent->pXPathNode;
		pXPathContextComponent = pXPathContextComponent->pXPathContext;
	}

	// Now get the first or last index node

	bFirstLast = TRUE;
	if (RC_BAD( rc = getNextIndexNode( &bFirstLast, bForward,
									m_pExprXPathSource, FALSE)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	See if a node/document passes all of the conditions for a query.
		Also increment the necessary counters.
		If it passes and we are building a result set, add it to the
		result set.
***************************************************************************/
RCODE F_Query::testPassed(
	IF_DOMNode **	ppNode,
	FLMBOOL *		pbPassed,
	FLMBOOL *		pbEliminatedDup)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bReportRSStatus = FALSE;
	FLMBOOL	bReportQueryStatus = FALSE;
	
	*pbEliminatedDup = FALSE;
	if (!m_pQuery || m_pQuery->eNodeType != FLM_XPATH_NODE || m_bRemoveDups)
	{
		
		// NOTE: If the document passed, but was eliminated by
		// duplicate checking, we will NOT increment documents read,
		// because we know that we have read the document before.
		// If the document failed, we also need to increment documents read,
		// even though we don't really know if we read it before.  In this
		// case, we may end up getting a larger document read count than
		// we should because we could count the same document as being
		// read more than once.
		
		if (*pbPassed)
		{
			if (RC_BAD( rc = checkIfDup( ppNode, pbPassed)))
			{
				goto Exit;
			}
			if (*pbPassed)
			{
				m_pCurrOpt->ui64DocsRead++;
				bReportQueryStatus = TRUE;
				if (m_pSortResultSet)
				{
					m_ui64RSDocsRead++;
					bReportRSStatus = TRUE;
				}
			}
			else
			{
				*pbEliminatedDup = TRUE;
			}
		}
		else
		{
			m_pCurrOpt->ui64DocsRead++;
			bReportQueryStatus = TRUE;
			if (m_pSortResultSet)
			{
				m_ui64RSDocsRead++;
				bReportRSStatus = TRUE;
			}
		}
	}
	
	if (RC_BAD( rc = validateNode( *ppNode, pbPassed)))
	{
		goto Exit;
	}
	if (*pbPassed)
	{
		m_pCurrOpt->ui64DocsPassed++;
		bReportQueryStatus = TRUE;
		
		// If we have a sort key and the sort order is different than
		// the optimization index, or the application requested that
		// we build a result set so it could do positioning, add the
		// document to the result set.
		
		if (m_pSortResultSet)
		{
			if (RC_BAD( rc = addToResultSet()))
			{
				goto Exit;
			}
			bReportRSStatus = TRUE;
		}
	}

	if (bReportQueryStatus)
	{
		if (RC_BAD( rc = queryStatus()))
		{
			goto Exit;
		}
	}
	if (bReportRSStatus && m_pQueryStatus)
	{
		if (RC_BAD( rc = m_pQueryStatus->resultSetStatus(
										m_ui64RSDocsRead, m_ui64RSDocsPassed,
										m_bEntriesAlreadyInOrder)))
		{
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get the next node from the index.
***************************************************************************/
RCODE F_Query::nextFromIndex(
	FLMBOOL			bEvalCurrDoc,
	FLMUINT			uiNumToSkip,
	FLMUINT *		puiNumSkipped,
	IF_DOMNode **	ppNode
	)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bPassed;
	FLMBOOL	bFirst;

	if (!bEvalCurrDoc)
	{
		bFirst = FALSE;
		if (RC_BAD( rc = getNextIndexNode( &bFirst, TRUE,
									m_pExprXPathSource, TRUE)))
		{
			goto Exit;
		}
	}

	for (;;)
	{

		// If m_pCurrDoc is non-NULL, it means we are set up
		// to call evalExpr.

		while (m_pCurrDoc)
		{
			FLMBOOL	bPassedEval;
			FLMBOOL	bEliminatedDup;
			
			if (RC_BAD( rc = evalExpr( NULL, TRUE, TRUE,
										m_pQuery, &bPassed, ppNode)))
			{
				if( rc == NE_XFLM_DOM_NODE_DELETED)
				{
					m_bResetAllXPaths = TRUE;
					rc = NE_XFLM_OK;
					goto Next_Index_Node;
				}
				goto Exit;
			}
			
			bPassedEval = bPassed;
			
			if (RC_BAD( rc = testPassed( ppNode, &bPassed, &bEliminatedDup)))
			{
				goto Exit;
			}
			
			if (bPassed)
			{
				m_eState = (m_eState == XFLM_QUERY_AT_BOF ||
								m_eState == XFLM_QUERY_NOT_POSITIONED)
							  ? XFLM_QUERY_AT_FIRST
							  : XFLM_QUERY_POSITIONED;
				if (puiNumSkipped)
				{
					(*puiNumSkipped)++;
				}
				if (uiNumToSkip <= 1)
				{
					goto Exit;
				}
				else
				{

					// puiNumSkipped will always be non-NULL in the case
					// where uiNumToSkip > 1

					flmAssert( puiNumSkipped);
					if (*puiNumSkipped >= uiNumToSkip)
					{
						goto Exit;
					}
					else
					{
						bPassed = FALSE;
					}
				}
			}
			
			// At this point we know that we failed to pass.  If we
			// passed evalExpr, though, we need to call it again so it
			// can iterate through all possible values.  No need to call
			// it again if it was eliminated because it was a duplicate howerver.
			
			if (bPassedEval && !bEliminatedDup)
			{
				continue;
			}
			
Next_Index_Node:

			// Get the next node from the index.
			// NOTE: m_pCurrDoc will be set to NULL if there are no
			// more nodes to get from this particular predicate.

			bFirst = FALSE;
			if (RC_BAD( rc = getNextIndexNode( &bFirst, TRUE,
											m_pExprXPathSource, FALSE)))
			{
				goto Exit;
			}
		}

		// If we get here, the loop above failed to get
		// anything.
		
		flmAssert( !m_pCurrDoc);

		// Try the next predicate.

		if (!useNextPredicate())
		{
			rc = RC_SET( NE_XFLM_EOF_HIT);
			m_eState = XFLM_QUERY_AT_EOF;
			goto Exit;
		}
		if (RC_BAD( rc = setupCurrPredicate( TRUE)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get the previous node from the index.
***************************************************************************/
RCODE F_Query::prevFromIndex(
	FLMBOOL			bEvalCurrDoc,
	FLMUINT			uiNumToSkip,
	FLMUINT *		puiNumSkipped,
	IF_DOMNode **	ppNode
	)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bPassed;
	FLMBOOL	bLast;

	if (!bEvalCurrDoc)
	{
		bLast = FALSE;
		if (RC_BAD( rc = getNextIndexNode( &bLast, FALSE,
								m_pExprXPathSource, TRUE)))
		{
			goto Exit;
		}
	}

	for (;;)
	{

		// If m_pCurrDoc is non-NULL, it means we are set up
		// to call evalExpr.

		while (m_pCurrDoc)
		{
			FLMBOOL	bPassedEval;
			FLMBOOL	bEliminatedDup;
			
			if (RC_BAD( rc = evalExpr( NULL, FALSE, TRUE,
									m_pQuery, &bPassed, ppNode)))
			{
				if( rc == NE_XFLM_DOM_NODE_DELETED)
				{
					m_bResetAllXPaths = TRUE;
					rc = NE_XFLM_OK;
					goto Prev_Index_Node;
				}
				goto Exit;
			}
			bPassedEval = bPassed;
			
			if (RC_BAD( rc = testPassed( ppNode, &bPassed, &bEliminatedDup)))
			{
				goto Exit;
			}

			if (bPassed)
			{
				m_eState = (m_eState == XFLM_QUERY_AT_EOF ||
								m_eState == XFLM_QUERY_NOT_POSITIONED)
							  ? XFLM_QUERY_AT_LAST
							  : XFLM_QUERY_POSITIONED;
				if (puiNumSkipped)
				{
					(*puiNumSkipped)++;
				}
				if (uiNumToSkip <= 1)
				{
					goto Exit;
				}
				else
				{

					// puiNumSkipped will always be non-NULL in the case
					// where uiNumToSkip > 1

					flmAssert( puiNumSkipped);
					if (*puiNumSkipped >= uiNumToSkip)
					{
						goto Exit;
					}
					else
					{
						bPassed = FALSE;
					}
				}
			}

			// At this point we know that we failed to pass.  If we
			// passed evalExpr, though, we need to call it again so it
			// can iterate through all possible values.  No need to call
			// it again if it was eliminated because it was a duplicate howerver.
			
			if (bPassedEval && !bEliminatedDup)
			{
				continue;
			}
			
Prev_Index_Node:

			// Get the previous node from the index.
			// NOTE: m_pCurrDoc will be set to NULL if there are no
			// more nodes to get from this particular predicate.

			bLast = FALSE;
			if (RC_BAD( rc = getNextIndexNode( &bLast, FALSE,
											m_pExprXPathSource, FALSE)))
			{
				goto Exit;
			}
		}

		// If we get here, the loop above failed to get
		// anything.
		
		flmAssert( !m_pCurrDoc);

		// Try the previous predicate.

		if (!usePrevPredicate())
		{
			rc = RC_SET( NE_XFLM_BOF_HIT);
			m_eState = XFLM_QUERY_AT_BOF;
			goto Exit;
		}
		if (RC_BAD( rc = setupCurrPredicate( FALSE)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get the next/previous document from the index we are scanning.
***************************************************************************/
RCODE F_Query::getDocFromIndexScan(
	FLMBOOL	bFirstLast,
	FLMBOOL	bForward
	)
{
	RCODE				rc = NE_XFLM_OK;
	F_DataVector	key;
	FLMUINT64		ui64DocId;

	for (;;)
	{
		if (bForward)
		{
			rc = (RCODE)(bFirstLast
							 ? m_pFSIndexCursor->firstKey( m_pDb, &key)
							 : m_pFSIndexCursor->nextKey( m_pDb, &key, FALSE));
			if (RC_BAD( rc))
			{
				if (rc == NE_XFLM_EOF_HIT)
				{
					m_eState = XFLM_QUERY_AT_EOF;
				}
				goto Exit;
			}
		}
		else
		{
			rc = (RCODE)(bFirstLast
							 ? m_pFSIndexCursor->lastKey( m_pDb, &key)
							 : m_pFSIndexCursor->prevKey( m_pDb, &key, FALSE));
			if (RC_BAD( rc))
			{
				if (rc == NE_XFLM_BOF_HIT)
				{
					m_eState = XFLM_QUERY_AT_BOF;
				}
				goto Exit;
			}
		}
		m_pCurrOpt->ui64KeysRead++;
		if (RC_BAD( rc = queryStatus()))
		{
			goto Exit;
		}

		// If we do not have a duplicate checking set, we can
		// break out of the loop and get the document associated
		// with this key.  Otherwise, we need to see if the
		// document associated with this key has already been
		// processed.  If so, we will go to the next key.

		if (!m_pDocIdSet)
		{
			break;
		}

		// If we are eliminating duplicates, see if the document
		// has already been processed.  If so, skip the key.

		ui64DocId = key.getDocumentID();
		if (RC_BAD( rc = m_pDocIdSet->findMatch( &ui64DocId, NULL)))
		{
			if (rc != NE_FLM_NOT_FOUND)
			{
				goto Exit;
			}

			// Document has not been processed.

			rc = NE_XFLM_OK;
			break;
		}

		// Document has already been passed, go to next/prev key.

		m_pCurrOpt->ui64KeyHadDupDoc++;
		if (RC_BAD( rc = queryStatus()))
		{
			goto Exit;
		}
		bFirstLast = FALSE;
	}

	// Retrieve the document node.

	if (RC_BAD( rc = m_pDb->getNode( m_uiCollection,
								key.getDocumentID(),
								&m_pCurrDoc)))
	{
		if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
		{

			// If we cannot retrieve the node, we have a corruption
			// in the database.

			rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
		}
		
		goto Exit;
	}
	else
	{
		if (RC_BAD( rc = incrNodesRead()))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get the next node from scanning sequentially through documents.
***************************************************************************/
RCODE F_Query::nextFromScan(
	FLMBOOL			bFirstDoc,
	FLMUINT			uiNumToSkip,
	FLMUINT *		puiNumSkipped,
	IF_DOMNode **	ppNode
	)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bPassed;
	FLMBOOL	bEliminatedDup;

	if (bFirstDoc ||
		 (m_pQuery && !m_bRemoveDups &&
		  m_pQuery->eNodeType == FLM_XPATH_NODE))
	{
		goto Eval_Doc;
	}

	// Read until we get a document/node that passes.

	for (;;)
	{
		if (m_bScan)
		{
			if (RC_BAD( rc = m_pCurrDoc->getNextDocument( m_pDb, &m_pCurrDoc)))
			{
				if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					m_eState = XFLM_QUERY_AT_EOF;
					rc = RC_SET( NE_XFLM_EOF_HIT);
				}
				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = getDocFromIndexScan( FALSE, TRUE)))
			{
				goto Exit;
			}
		}
		m_bResetAllXPaths = TRUE;

Eval_Doc:

		// See if the document passes.

		if (RC_BAD( rc = evalExpr( NULL, TRUE, TRUE,
									m_pQuery, &bPassed, ppNode)))
		{
			if( rc == NE_XFLM_DOM_NODE_DELETED)
			{
				m_bResetAllXPaths = TRUE;
				rc = NE_XFLM_OK;
				continue;
			}
			
			goto Exit;
		}
		
		if (bPassed && m_bScanIndex)
		{
			m_pCurrOpt->ui64KeysPassed++;
			if (RC_BAD( rc = queryStatus()))
			{
				goto Exit;
			}
		}
		
		if (RC_BAD( rc = testPassed( ppNode, &bPassed, &bEliminatedDup)))
		{
			goto Exit;
		}

		if (bPassed)
		{
			m_eState = (m_eState == XFLM_QUERY_AT_BOF ||
							m_eState == XFLM_QUERY_NOT_POSITIONED)
						  ? XFLM_QUERY_AT_FIRST
						  : XFLM_QUERY_POSITIONED;
			if (puiNumSkipped)
			{
				(*puiNumSkipped)++;
			}
			if (uiNumToSkip <= 1)
			{
				goto Exit;
			}
			else
			{

				// puiNumSkipped will always be non-NULL in the case
				// where uiNumToSkip > 1

				flmAssert( puiNumSkipped);
				if (*puiNumSkipped >= uiNumToSkip)
				{
					goto Exit;
				}
				else
				{
					bPassed = FALSE;
				}
			}
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get the previous node from scanning sequentially through documents.
***************************************************************************/
RCODE F_Query::prevFromScan(
	FLMBOOL			bLastDoc,
	FLMUINT			uiNumToSkip,
	FLMUINT *		puiNumSkipped,
	IF_DOMNode **	ppNode
	)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bPassed;
	FLMBOOL	bEliminatedDup;

	if (bLastDoc ||
		 (m_pQuery && !m_bRemoveDups &&
		  m_pQuery->eNodeType == FLM_XPATH_NODE))
	{
		goto Eval_Doc;
	}

	// Read until we get a document/node that passes.

	for (;;)
	{

		// Go to the previous document

		if (m_bScan)
		{
			if (RC_BAD( rc = m_pCurrDoc->getPreviousDocument( m_pDb, &m_pCurrDoc)))
			{
				if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					m_eState = XFLM_QUERY_AT_BOF;
					rc = RC_SET( NE_XFLM_BOF_HIT);
				}
				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = getDocFromIndexScan( FALSE, FALSE)))
			{
				goto Exit;
			}
		}

		m_bResetAllXPaths = TRUE;

Eval_Doc:

		// See if the document passes.

		if (RC_BAD( rc = evalExpr( NULL, FALSE, TRUE,
									m_pQuery, &bPassed, ppNode)))
		{
			if( rc == NE_XFLM_DOM_NODE_DELETED)
			{
				m_bResetAllXPaths = TRUE;
				rc = NE_XFLM_OK;
				continue;
			}
			goto Exit;
		}
		
		if (bPassed && m_bScanIndex)
		{
			m_pCurrOpt->ui64KeysPassed++;
			if (RC_BAD( rc = queryStatus()))
			{
				goto Exit;
			}
		}
		
		if (RC_BAD( rc = testPassed( ppNode, &bPassed, &bEliminatedDup)))
		{
			goto Exit;
		}

		if (bPassed)
		{
			m_eState = (m_eState == XFLM_QUERY_AT_EOF ||
							m_eState == XFLM_QUERY_NOT_POSITIONED)
						  ? XFLM_QUERY_AT_LAST
						  : XFLM_QUERY_POSITIONED;
			if (puiNumSkipped)
			{
				(*puiNumSkipped)++;
			}
			if (uiNumToSkip <= 1)
			{
				goto Exit;
			}
			else
			{

				// puiNumSkipped will always be non-NULL in the case
				// where uiNumToSkip > 1

				flmAssert( puiNumSkipped);
				if (*puiNumSkipped >= uiNumToSkip)
				{
					goto Exit;
				}
				else
				{
					bPassed = FALSE;
				}
			}
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get first node/document that passes query expression.
***************************************************************************/
RCODE XFLAPI F_Query::getFirst(
	IF_Db *			ifpDb,
	IF_DOMNode **	ppNode,
	FLMUINT			uiTimeLimit)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bStartedTrans = FALSE;
	
	// If we are building on a background thread and this is not the
	// background thread, we need to get the results from the result
	// set that is being built.
	
	if ((m_pSortResultSet && m_uiBuildThreadId != f_threadId()) ||
		 m_bResultSetPopulated)
	{
		rc = getFirstFromResultSet( ifpDb, ppNode, uiTimeLimit);
		goto Exit;
	}
		
	m_pDb = (F_Db *)ifpDb;
	if (ppNode && *ppNode)
	{
		(*ppNode)->Release();
		*ppNode = NULL;
	}

	if (m_pDatabase && m_pDb->m_pDatabase != m_pDatabase)
	{

		// Make sure the passed in F_Db matches the one associated with
		// the query.

		rc = RC_SET( NE_XFLM_Q_MISMATCHED_DB);
		goto Exit;
	}
	
	// See if the database is being forced to close

	if (RC_BAD( rc = m_pDb->checkState( __FILE__, __LINE__)))
	{
		goto Exit;
	}

	// If we are not in a transaction, we cannot read.

	if (m_pDb->m_eTransType == XFLM_NO_TRANS)
	{
		if( RC_BAD( rc = m_pDb->checkTransaction(
			XFLM_READ_TRANS, &bStartedTrans)))
		{
			goto Exit;
		}
	}

	// See if we have a transaction going which should be aborted.

	if (RC_BAD( m_pDb->m_AbortRc))
	{
		rc = RC_SET( NE_XFLM_ABORT_TRANS);
		goto Exit;
	}

	if (!m_bOptimized)
	{
		if (RC_BAD( rc = optimize()))
		{
			goto Exit;
		}
	}
	
	// If the query can never evaluate to TRUE, return EOF without
	// doing anything.

	if (m_bEmpty)
	{
		m_eState = XFLM_QUERY_AT_EOF;
		rc = RC_SET( NE_XFLM_EOF_HIT);
		goto Exit;
	}

	// See if we need to build a result set.

	if ((m_pSortResultSet && m_uiBuildThreadId != f_threadId()) ||
		 m_bResultSetPopulated)
	{
		rc = getFirstFromResultSet( ifpDb, ppNode, uiTimeLimit);
		goto Exit;
	}

	// Anytime we go back to the first, we must free whatever list
	// of document IDs we have collected so far.

	if (m_bRemoveDups && m_pDocIdSet)
	{
		m_pDocIdSet->Release();
		m_pDocIdSet = NULL;
	}

	if ((m_uiTimeLimit = uiTimeLimit) != 0)
	{
		m_uiTimeLimit = FLM_MILLI_TO_TIMER_UNITS( uiTimeLimit);
		m_uiStartTime = FLM_GET_TIMER();
	}
	
	if (m_bScan)
	{

		// Start at the beginning of the collection.

		if (RC_BAD( rc = m_pDb->getFirstDocument( m_uiCollection, &m_pCurrDoc)))
		{
			if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				m_eState = XFLM_QUERY_AT_EOF;
				rc = RC_SET( NE_XFLM_EOF_HIT);
			}
			goto Exit;
		}
		if (RC_BAD( rc = nextFromScan( TRUE, 0, NULL, ppNode)))
		{
			goto Exit;
		}
	}
	else if (m_bScanIndex)
	{
		if (RC_BAD( rc = getDocFromIndexScan( TRUE, TRUE)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = nextFromScan( TRUE, 0, NULL, ppNode)))
		{
			goto Exit;
		}
	}
	else
	{
		m_pCurrContext = m_pQuery->pContext;

		// Position the context to lowest selected context,
		// context path, and predicate.

		// NOTE: Because the m_bScan flag is not set, we are guaranteed
		// that the following traversal scheme will not encounter any
		// contexts, context paths, or predicates that require scanning, even
		// though there may have been some during optimization.  They cannot
		// have been the selected contexts, context paths, or predicates
		// without ultimately causing the m_bScan flag to have been set.

		useLeafContext( TRUE);

		// Setup predicate and get the first node.

		if (RC_BAD( rc = setupCurrPredicate( TRUE)))
		{
			goto Exit;
		}
	
		if (RC_BAD( rc = nextFromIndex( TRUE, 0, NULL, ppNode)))
		{
			goto Exit;
		}
	}

Exit:

	if (m_pCurrNode)
	{
		m_pCurrNode->Release();
		m_pCurrNode = NULL;
	}
	
	if (RC_BAD( rc))
	{
		if (m_pCurrDoc)
		{
			m_pCurrDoc->Release();
			m_pCurrDoc = NULL;
		}
	}
	else
	{
		m_pCurrNode = *ppNode;
		m_pCurrNode->AddRef();
	}
	
	if( bStartedTrans)
	{
		m_pDb->transAbort();
	}

	m_uiTimeLimit = 0;
	return( rc);
}

/***************************************************************************
Desc:	Get last node/document that passes query expression.
***************************************************************************/
RCODE XFLAPI F_Query::getLast(
	IF_Db *			ifpDb,
	IF_DOMNode **	ppNode,
	FLMUINT			uiTimeLimit)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bStartedTrans = FALSE;

	// If we are building on a background thread and this is not the
	// background thread, we need to get the results from the result
	// set that is being built.
	
	if ((m_pSortResultSet && m_uiBuildThreadId != f_threadId()) ||
		 m_bResultSetPopulated)
	{
		rc = getLastFromResultSet( ifpDb, ppNode, uiTimeLimit);
		goto Exit;
	}
		
	m_pDb = (F_Db *)ifpDb;
	if (ppNode && *ppNode)
	{
		(*ppNode)->Release();
		*ppNode = NULL;
	}

	if (m_pDatabase && m_pDb->m_pDatabase != m_pDatabase)
	{

		// Make sure the passed in F_Db matches the one associated with
		// the query.

		rc = RC_SET( NE_XFLM_Q_MISMATCHED_DB);
		goto Exit;
	}
	
	// See if the database is being forced to close

	if (RC_BAD( rc = m_pDb->checkState( __FILE__, __LINE__)))
	{
		goto Exit;
	}

	// If we are not in a transaction, we cannot read.

	if (m_pDb->m_eTransType == XFLM_NO_TRANS)
	{
		if( RC_BAD( rc = m_pDb->checkTransaction(
			XFLM_READ_TRANS, &bStartedTrans)))
		{
			goto Exit;
		}
	}

	// See if we have a transaction going which should be aborted.

	if (RC_BAD( m_pDb->m_AbortRc))
	{
		rc = RC_SET( NE_XFLM_ABORT_TRANS);
		goto Exit;
	}

	if (!m_bOptimized)
	{
		if (RC_BAD( rc = optimize()))
		{
			goto Exit;
		}
	}
	
	// If the query can never evaluate to TRUE, return EOF without
	// doing anything.

	if (m_bEmpty)
	{
		m_eState = XFLM_QUERY_AT_BOF;
		rc = RC_SET( NE_XFLM_BOF_HIT);
		goto Exit;
	}

	if ((m_pSortResultSet && m_uiBuildThreadId != f_threadId()) ||
		 m_bResultSetPopulated)
	{
		rc = getLastFromResultSet( ifpDb, ppNode, uiTimeLimit);
		goto Exit;
	}
		
	// Anytime we go to the last, we must free whatever list
	// of document IDs we have collected so far.

	if (m_bRemoveDups && m_pDocIdSet)
	{
		m_pDocIdSet->Release();
		m_pDocIdSet = NULL;
	}

	if ((m_uiTimeLimit = uiTimeLimit) != 0)
	{
		m_uiTimeLimit = FLM_MILLI_TO_TIMER_UNITS( uiTimeLimit);
		m_uiStartTime = FLM_GET_TIMER();
	}
	if (m_bScan)
	{

		// Start at the end of the collection.

		if (RC_BAD( rc = m_pDb->getLastDocument( m_uiCollection, &m_pCurrDoc)))
		{
			if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				m_eState = XFLM_QUERY_AT_BOF;
				rc = RC_SET( NE_XFLM_BOF_HIT);
			}
			goto Exit;
		}
		if (RC_BAD( rc = prevFromScan( TRUE, 0, NULL, ppNode)))
		{
			goto Exit;
		}
	}
	else if (m_bScanIndex)
	{
		if (RC_BAD( rc = getDocFromIndexScan( TRUE, FALSE)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = prevFromScan( TRUE, 0, NULL, ppNode)))
		{
			goto Exit;
		}
	}
	else
	{
		m_pCurrContext = m_pQuery->pContext;

		// Position the context to lowest selected context,
		// context path, and predicate.

		// NOTE: Because the m_bScan flag is not set, we are guaranteed
		// that the following traversal scheme will not encounter any
		// contexts, context paths, or predicates that require scanning, even
		// though there may have been some during optimization.  They cannot
		// have been the selected contexts, context paths, or predicates
		// without ultimately causing the m_bScan flag to have been set.

		useLeafContext( FALSE);

		// Setup predicate and get the last node.

		if (RC_BAD( rc = setupCurrPredicate( FALSE)))
		{
			goto Exit;
		}
	
		if (RC_BAD( rc = prevFromIndex( TRUE, 0, NULL, ppNode)))
		{
			goto Exit;
		}
	}

Exit:

	if (m_pCurrNode)
	{
		m_pCurrNode->Release();
		m_pCurrNode = NULL;
	}
	
	if (RC_BAD( rc))
	{
		if (m_pCurrDoc)
		{
			m_pCurrDoc->Release();
			m_pCurrDoc = NULL;
		}
	}
	else
	{
		m_pCurrNode = *ppNode;
		m_pCurrNode->AddRef();
	}
	
	if( bStartedTrans)
	{
		m_pDb->transAbort();
	}
	
	m_uiTimeLimit = 0;
	return( rc);
}

/***************************************************************************
Desc:	Get next node/document that passes query expression.
***************************************************************************/
RCODE XFLAPI F_Query::getNext(
	IF_Db *			ifpDb,
	IF_DOMNode **	ppNode,
	FLMUINT			uiTimeLimit,
	FLMUINT			uiNumToSkip,
	FLMUINT *		puiNumSkipped)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiNumSkipped;
	FLMBOOL			bStartedTrans = FALSE;
	
	// If we are building on a background thread and this is not the
	// background thread, we need to get the results from the result
	// set that is being built.
	
	if ((m_pSortResultSet && m_uiBuildThreadId != f_threadId()) ||
		 m_bResultSetPopulated)
	{
		rc = getNextFromResultSet( ifpDb, ppNode, uiTimeLimit, uiNumToSkip,
						puiNumSkipped);
		goto Exit;
	}
		
	m_pDb = (F_Db *)ifpDb;
	if (ppNode && *ppNode)
	{
		(*ppNode)->Release();
		*ppNode = NULL;
	}

	// See if the database is being forced to close

	if (RC_BAD( rc = m_pDb->checkState( __FILE__, __LINE__)))
	{
		goto Exit;
	}

	// If we are not in a transaction, we cannot read.

	if (m_pDb->m_eTransType == XFLM_NO_TRANS)
	{
		if( RC_BAD( rc = m_pDb->checkTransaction(
			XFLM_READ_TRANS, &bStartedTrans)))
		{
			goto Exit;
		}
	}

	// See if we have a transaction going which should be aborted.

	if (RC_BAD( m_pDb->m_AbortRc))
	{
		rc = RC_SET( NE_XFLM_ABORT_TRANS);
		goto Exit;
	}

	if (!puiNumSkipped)
	{

		// puiNumSkipped has to be non-NULL so it can be incremented only
		// if uiNumToSkip > 1

		if (uiNumToSkip > 1)
		{
			uiNumSkipped = 0;
			puiNumSkipped = &uiNumSkipped;
		}
	}
	else
	{
		*puiNumSkipped = 0;
	}
	switch (m_eState)
	{
		case XFLM_QUERY_AT_EOF:
			rc = RC_SET( NE_XFLM_EOF_HIT);
			goto Exit;
		case XFLM_QUERY_AT_BOF:
		case XFLM_QUERY_NOT_POSITIONED:
			if (RC_OK( rc = getFirst( ifpDb, ppNode, uiTimeLimit)))
			{
				if (puiNumSkipped)
				{
					*puiNumSkipped = 1;
				}
				if (uiNumToSkip <= 1)
				{
					goto Exit;
				}
			}
			else
			{
				goto Exit;
			}
			break;
		default:
			if( !m_pCurrNode)
			{
				rc = RC_SET( NE_XFLM_Q_NOT_POSITIONED);
				goto Exit;
			}
			break;
	}

	// Optimization has to already have occurred.

	flmAssert( m_bOptimized);

	// Make sure the passed in F_Db matches the one associated with
	// the query.

	if (m_pDb->m_pDatabase != m_pDatabase)
	{
		rc = RC_SET( NE_XFLM_Q_MISMATCHED_DB);
		goto Exit;
	}

	// If we have been positioned, we better have a current document and node

	flmAssert( m_pCurrDoc && m_pCurrNode);

	if ((m_uiTimeLimit = uiTimeLimit) != 0)
	{
		m_uiTimeLimit = FLM_MILLI_TO_TIMER_UNITS( uiTimeLimit);
		m_uiStartTime = FLM_GET_TIMER();
	}
	if (m_bScan || m_bScanIndex)
	{
		if (RC_BAD( rc = nextFromScan( FALSE, uiNumToSkip,
									puiNumSkipped, ppNode)))
		{
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = nextFromIndex(
									(m_pQuery && !m_bRemoveDups &&
									 m_pQuery->eNodeType == FLM_XPATH_NODE &&
									 !m_pQuery->nd.pXPath->pLastComponent->bIsSource)
									 ? TRUE
									 : FALSE, uiNumToSkip, puiNumSkipped, ppNode)))
		{
			goto Exit;
		}
	}

Exit:

	if (m_pCurrNode)
	{
		m_pCurrNode->Release();
		m_pCurrNode = NULL;
	}
	
	if (RC_BAD( rc))
	{
		if (m_pCurrDoc)
		{
			m_pCurrDoc->Release();
			m_pCurrDoc = NULL;
		}
	}
	else
	{
		m_pCurrNode = *ppNode;
		m_pCurrNode->AddRef();
	}
	
	if( bStartedTrans)
	{
		m_pDb->transAbort();
	}
	
	m_uiTimeLimit = 0;
	return( rc);
}

/***************************************************************************
Desc:	Get previous node/document that passes query expression.
***************************************************************************/
RCODE XFLAPI F_Query::getPrev(
	IF_Db *			ifpDb,
	IF_DOMNode **	ppNode,
	FLMUINT			uiTimeLimit,
	FLMUINT			uiNumToSkip,
	FLMUINT *		puiNumSkipped)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiNumSkipped;
	FLMBOOL			bStartedTrans = FALSE;
		
	// If we are building on a background thread and this is not the
	// background thread, we need to get the results from the result
	// set that is being built.
	
	if ((m_pSortResultSet && m_uiBuildThreadId != f_threadId()) ||
		 m_bResultSetPopulated)
	{
		rc = getPrevFromResultSet( ifpDb, ppNode, uiTimeLimit, uiNumToSkip,
						puiNumSkipped);
		goto Exit;
	}
		
	m_pDb = (F_Db *)ifpDb;
	if (ppNode && *ppNode)
	{
		(*ppNode)->Release();
		*ppNode = NULL;
	}

	// See if the database is being forced to close

	if (RC_BAD( rc = m_pDb->checkState( __FILE__, __LINE__)))
	{
		goto Exit;
	}

	// If we are not in a transaction, we cannot read.

	if (m_pDb->m_eTransType == XFLM_NO_TRANS)
	{
		if( RC_BAD( rc = m_pDb->checkTransaction(
			XFLM_READ_TRANS, &bStartedTrans)))
		{
			goto Exit;
		}
	}

	// See if we have a transaction going which should be aborted.

	if (RC_BAD( m_pDb->m_AbortRc))
	{
		rc = RC_SET( NE_XFLM_ABORT_TRANS);
		goto Exit;
	}

	if (!puiNumSkipped)
	{

		// puiNumSkipped has to be non-NULL so it can be incremented only
		// if uiNumToSkip > 1

		if (uiNumToSkip > 1)
		{
			uiNumSkipped = 0;
			puiNumSkipped = &uiNumSkipped;
		}
	}
	else
	{
		*puiNumSkipped = 0;
	}
	switch (m_eState)
	{
		case XFLM_QUERY_AT_BOF:
			rc = RC_SET( NE_XFLM_BOF_HIT);
			goto Exit;
		case XFLM_QUERY_AT_EOF:
		case XFLM_QUERY_NOT_POSITIONED:
			if (RC_OK( rc = getLast( ifpDb, ppNode, uiTimeLimit)))
			{
				if (puiNumSkipped)
				{
					*puiNumSkipped = 1;
				}
				if (uiNumToSkip <= 1)
				{
					goto Exit;
				}
			}
			else
			{
				goto Exit;
			}
			break;
		default:
			if( !m_pCurrNode)
			{
				rc = RC_SET( NE_XFLM_Q_NOT_POSITIONED);
				goto Exit;
			}
			break;
	}

	// Optimization has to already have occurred.

	flmAssert( m_bOptimized);

	// Make sure the passed in F_Db matches the one associated with
	// the query.

	if (m_pDb->m_pDatabase != m_pDatabase)
	{
		rc = RC_SET( NE_XFLM_Q_MISMATCHED_DB);
		goto Exit;
	}

	// If we have been positioned, we better have a current document and node

	flmAssert( m_pCurrDoc && m_pCurrNode);

	if ((m_uiTimeLimit = uiTimeLimit) != 0)
	{
		m_uiTimeLimit = FLM_MILLI_TO_TIMER_UNITS( uiTimeLimit);
		m_uiStartTime = FLM_GET_TIMER();
	}
	if (m_bScan || m_bScanIndex)
	{
		if (RC_BAD( rc = prevFromScan( FALSE, uiNumToSkip,
										puiNumSkipped, ppNode)))
		{
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = prevFromIndex(
									(m_pQuery && !m_bRemoveDups &&
									 m_pQuery->eNodeType == FLM_XPATH_NODE &&
									 !m_pQuery->nd.pXPath->pLastComponent->bIsSource)
									 ? TRUE
									 : FALSE, uiNumToSkip, puiNumSkipped, ppNode)))
		{
			goto Exit;
		}
	}

Exit:

	if (m_pCurrNode)
	{
		m_pCurrNode->Release();
		m_pCurrNode = NULL;
	}
	
	if (RC_BAD( rc))
	{
		if (m_pCurrDoc)
		{
			m_pCurrDoc->Release();
			m_pCurrDoc = NULL;
		}
	}
	else
	{
		m_pCurrNode = *ppNode;
		m_pCurrNode->AddRef();
	}
	
	if( bStartedTrans)
	{
		m_pDb->transAbort();
	}
	
	m_uiTimeLimit = 0;
	return( rc);
}

/***************************************************************************
Desc:	Get current document that passes query expression.
***************************************************************************/
RCODE XFLAPI F_Query::getCurrent(
	IF_Db *				ifpDb,
	IF_DOMNode **		ppNode)
{
	RCODE					rc = NE_XFLM_OK;
	FLMBOOL				bStartedTrans = FALSE;
	
	// If we are building on a background thread and this is not the
	// background thread, we need to get the results from the result
	// set that is being built.
	
	if ((m_pSortResultSet && m_uiBuildThreadId != f_threadId()) ||
		 m_bResultSetPopulated)
	{
		rc = getCurrentFromResultSet( ifpDb, ppNode);
		goto Exit;
	}
		
	m_pDb = (F_Db *)ifpDb;
	if (ppNode && *ppNode)
	{
		(*ppNode)->Release();
		*ppNode = NULL;
	}

	// See if the database is being forced to close

	if (RC_BAD( rc = m_pDb->checkState( __FILE__, __LINE__)))
	{
		goto Exit;
	}

	// If we are not in a transaction, we cannot read.

	if (m_pDb->m_eTransType == XFLM_NO_TRANS)
	{
		if( RC_BAD( rc = m_pDb->checkTransaction(
			XFLM_READ_TRANS, &bStartedTrans)))
		{
			goto Exit;
		}
	}

	// See if we have a transaction going which should be aborted.

	if (RC_BAD( m_pDb->m_AbortRc))
	{
		rc = RC_SET( NE_XFLM_ABORT_TRANS);
		goto Exit;
	}

	switch (m_eState)
	{
		case XFLM_QUERY_AT_BOF:
		case XFLM_QUERY_NOT_POSITIONED:
			rc = RC_SET( NE_XFLM_BOF_HIT);
			goto Exit;
		case XFLM_QUERY_AT_EOF:
			rc = RC_SET( NE_XFLM_EOF_HIT);
			goto Exit;
		default:
			if( !m_pCurrNode)
			{
				rc = RC_SET( NE_XFLM_Q_NOT_POSITIONED);
				goto Exit;
			}
			break;
	}

	// Optimization has to already have occurred.

	flmAssert( m_bOptimized);

	// Make sure the passed in F_Db matches the one associated with
	// the query.

	if (m_pDb->m_pDatabase != m_pDatabase)
	{
		rc = RC_SET( NE_XFLM_Q_MISMATCHED_DB);
		goto Exit;
	}

	// If we have been positioned, we better have a current document and
	// a current node id.

	flmAssert( m_pCurrDoc && m_pCurrNode);
	
	if( *ppNode)
	{
		(*ppNode)->Release();
	}
	
	*ppNode = m_pCurrNode;
	(*ppNode)->AddRef();

Exit:

	if (RC_BAD( rc))
	{
		if (m_pCurrDoc)
		{
			m_pCurrDoc->Release();
			m_pCurrDoc = NULL;
		}
		
		if (m_pCurrNode)
		{
			m_pCurrNode->Release();
			m_pCurrNode = NULL;
		}
	}

	if( bStartedTrans)
	{
		m_pDb->transAbort();
	}
	
	m_uiTimeLimit = 0;
	return( rc);
}

/***************************************************************************
Desc:	Get statistics and optimization information.
***************************************************************************/
RCODE XFLAPI F_Query::getStatsAndOptInfo(
	FLMUINT *			puiNumOptInfos,
	XFLM_OPT_INFO **	ppOptInfo)
{
	RCODE					rc = NE_XFLM_OK;
	XFLM_OPT_INFO *	pOptInfo;
	FLMUINT				uiOptInfoCount;

	if (!m_bOptimized)
	{
		*puiNumOptInfos = 0;
		*ppOptInfo = NULL;
		goto Exit;
	}

	if (m_bScan || m_bEmpty)
	{
		if (RC_BAD( rc = f_alloc( sizeof( XFLM_OPT_INFO), ppOptInfo)))
		{
			goto Exit;
		}
		f_memcpy( *ppOptInfo, &m_scanOptInfo, sizeof( XFLM_OPT_INFO));
		*puiNumOptInfos = 1;
	}
	else
	{
		OP_CONTEXT *		pSaveCurrContext = m_pCurrContext;
		CONTEXT_PATH *		pSaveCurrContextPath = m_pCurrContextPath;
		PATH_PRED *			pSaveCurrPred = m_pCurrPred;
		FQNODE *				pSaveExprXPathSource = m_pExprXPathSource;

		// Count the number of contexts.

		m_pCurrContext = m_pQuery->pContext;
		*puiNumOptInfos = 0;
		useLeafContext( TRUE);
		do
		{
			if (m_pCurrPred->pNodeSource)
			{
				if (RC_BAD( rc = m_pCurrPred->pNodeSource->getOptInfoCount(
															(IF_Db *)m_pDb,
															&uiOptInfoCount)))
				{
					goto Exit;
				}
				(*puiNumOptInfos) += uiOptInfoCount;
			}
			else
			{
				(*puiNumOptInfos)++;
			}
		} while (useNextPredicate());

		// Allocate the opt info array.

		if (RC_OK( rc = f_alloc( sizeof( XFLM_OPT_INFO) * (*puiNumOptInfos),
									ppOptInfo)))
		{
			pOptInfo = *ppOptInfo;
			m_pCurrContext = m_pQuery->pContext;
			useLeafContext( TRUE);
			do
			{
				if (m_pCurrPred->pNodeSource)
				{
					if (RC_BAD( rc = m_pCurrPred->pNodeSource->getOptInfoCount(
																(IF_Db *)m_pDb,
																&uiOptInfoCount)))
					{
						goto Exit;
					}
					if (RC_BAD( rc = m_pCurrPred->pNodeSource->getOptInfo(
															(IF_Db *)m_pDb,
															pOptInfo, uiOptInfoCount)))
					{
						goto Exit;
					}
					pOptInfo += uiOptInfoCount;
				}
				else
				{
					f_memcpy( pOptInfo, m_pCurrOpt, sizeof( XFLM_OPT_INFO));
					pOptInfo++;
				}
			} while (useNextPredicate());
		}

		// Restore the current predicate.

		m_pCurrContext = pSaveCurrContext;
		m_pCurrContextPath = pSaveCurrContextPath;
		m_pCurrPred = pSaveCurrPred;
		m_pExprXPathSource = pSaveExprXPathSource;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Free the optimization info structure.
***************************************************************************/
void XFLAPI F_Query::freeStatsAndOptInfo(
	XFLM_OPT_INFO **	ppOptInfo)
{
	if (*ppOptInfo)
	{
		f_free( ppOptInfo);
	}
}


/****************************************************************************
Desc:		Create an empty query object and return it's interface...
****************************************************************************/
RCODE XFLAPI F_DbSystem::createIFQuery(
	IF_Query **			ppQuery)
{
	RCODE					rc = NE_XFLM_OK;
	F_Query *			pQuery = NULL;

	if ((pQuery = f_new F_Query) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	
	*ppQuery = pQuery;
	pQuery = NULL;
	
Exit:

	if( pQuery)
	{
		pQuery->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Parse a query using the passed in query string.  The uiQueryType
		parameter is intended to identify the type of query syntax used.
		The XPATH syntax is currently the only sypported type.
****************************************************************************/
RCODE F_Query::setupQueryExpr(
	FLMBOOL				bUnicode,
	IF_Db *				ifpDb,
	const void *		pvQuery)
{
	RCODE					rc = NE_XFLM_OK;
	F_XPath				XPath;
	F_Db *				pDb = (F_Db *)ifpDb;

	// Reset the query object totally.

	clearQuery();

	if (!bUnicode)
	{
		if (RC_BAD( rc = XPath.parseQuery( pDb, (char *)pvQuery, this)))
		{
			goto Exit;
		}
	}
	else
	{
		flmAssert( 0);
	}

	// We need to make sure that from this point forward, we are using the same
	// database object.
	
	m_pDatabase = pDb->m_pDatabase;

Exit:

	return rc;
}

/****************************************************************************
Desc:	Comparison function for comparing node ids.
****************************************************************************/
FSTATIC int nodeIdCompareFunc(
	void *	pvData1,
	void *	pvData2,
	void *	// pvUserData
	)
{
	if (*((FLMUINT64 *)pvData1) < *((FLMUINT64 *)pvData2))
	{
		return( -1);
	}
	else if (*((FLMUINT64 *)pvData1) > *((FLMUINT64 *)pvData2))
	{
		return( 1);
	}
	else
	{
		return( 0);
	}
}

/****************************************************************************
Desc:	Allocate a result set for duplicate checking.
****************************************************************************/
RCODE F_Query::allocDupCheckSet( void)
{
	RCODE			rc = NE_XFLM_OK;
	char			szTmpDir [F_PATH_MAX_SIZE];

	if (m_pDocIdSet)
	{
		m_pDocIdSet->Release();
		m_pDocIdSet = NULL;
	}

	if ((m_pDocIdSet = f_new F_DynSearchSet) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	if (RC_BAD( rc = gv_pXFlmDbSystem->getTempDir( szTmpDir)))
	{
		if (rc == NE_FLM_IO_PATH_NOT_FOUND)
		{
			rc = NE_XFLM_OK;
		}
		else
		{
			goto Exit;
		}
	}

	if (!szTmpDir [0])
	{
		if (RC_BAD( rc = gv_XFlmSysData.pFileSystem->pathReduce( 
					m_pDb->m_pDatabase->m_pszDbPath, szTmpDir, NULL)))
		{
			goto Exit;
		}
	}

	if (RC_BAD( rc = m_pDocIdSet->setup( szTmpDir, sizeof( FLMUINT64))))
	{
		goto Exit;
	}

	m_pDocIdSet->setCompareFunc( nodeIdCompareFunc, NULL);

Exit:

	if (RC_BAD( rc))
	{
		if (m_pDocIdSet)
		{
			m_pDocIdSet->Release();
			m_pDocIdSet = NULL;
		}
	}

	return( rc);
}

/****************************************************************************
Desc:	Check to see if we have already passed the document that *ppNode
		is in.  If so, return FALSE in *pbPassed.
NOTE:	This routine will return the root node of the document if it still
		passes.
****************************************************************************/
RCODE F_Query::checkIfDup(
	IF_DOMNode **	ppNode,
	FLMBOOL *		pbPassed)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT64	ui64DocId;

	// If we have not yet allocated the result set, do it now.

	if (!m_pDocIdSet)
	{
		if (RC_BAD( rc = allocDupCheckSet()))
		{
			goto Exit;
		}
	}

	// See if we can add the document id to the result set
	
	if( RC_BAD( rc = m_pCurrDoc->getNodeId( m_pDb, &ui64DocId)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = m_pDocIdSet->addEntry( &ui64DocId)))
	{
		if (rc == NE_FLM_EXISTS)
		{
			*pbPassed = FALSE;
			rc = NE_XFLM_OK;
			m_pCurrOpt->ui64DupDocsEliminated++;
		}
		goto Exit;
	}

	// When eliminating duplicates, we always return the root node
	// of the document.

	(*ppNode)->Release();
	*ppNode = m_pCurrDoc;
	(*ppNode)->AddRef();

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Setup duplicate handling for a query.
****************************************************************************/
void XFLAPI F_Query::setDupHandling(
	FLMBOOL	bRemoveDups
	)
{
	// Should not be able to change this after optimization has occurred.
	
	flmAssert( !m_bOptimized);
	m_bRemoveDups = bRemoveDups;
	if (!bRemoveDups && m_pDocIdSet)
	{
		m_pDocIdSet->Release();
		m_pDocIdSet = NULL;
	}
}

/****************************************************************************
Desc:	Set an index for the query.
****************************************************************************/
RCODE XFLAPI F_Query::setIndex(
	FLMUINT	uiIndex
	)
{
	RCODE	rc = NE_XFLM_OK;

	// Cannot set the index if we have already optimized the query

	if (m_bOptimized)
	{
		rc = RC_SET( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}
	m_uiIndex = uiIndex;
	m_bIndexSet = TRUE;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Set an index for the query.
****************************************************************************/
RCODE XFLAPI F_Query::getIndex(
	IF_Db *			ifpDb,
	FLMUINT *		puiIndex,
	FLMBOOL *		pbHaveMultiple)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bStartedTrans = FALSE;

	if (m_bIndexSet)
	{
		*puiIndex = m_uiIndex;
		*pbHaveMultiple = FALSE;
		goto Exit;
	}

	// Optimize the query to determine the index.

	m_pDb = (F_Db *)ifpDb;
	if (!m_bOptimized)
	{

		if (m_pDatabase && m_pDb->m_pDatabase != m_pDatabase)
		{

			// Make sure the passed in F_Db matches the one associated with
			// the query.

			rc = RC_SET( NE_XFLM_Q_MISMATCHED_DB);
			goto Exit;
		}
		
		// See if the database is being forced to close

		if (RC_BAD( rc = m_pDb->checkState( __FILE__, __LINE__)))
		{
			goto Exit;
		}

		// See if we have a transaction going which should be aborted.

		if (RC_BAD( m_pDb->m_AbortRc))
		{
			rc = RC_SET( NE_XFLM_ABORT_TRANS);
			goto Exit;
		}

		// If we are not in a transaction, we cannot read.

		if (m_pDb->m_eTransType == XFLM_NO_TRANS)
		{
			if( RC_BAD( rc = m_pDb->checkTransaction(
				XFLM_READ_TRANS, &bStartedTrans)))
			{
				goto Exit;
			}
		}

		if (RC_BAD( rc = optimize()))
		{
			goto Exit;
		}
	}

	*pbHaveMultiple = FALSE;
	if (m_bScan || m_bEmpty)
	{
		*puiIndex = 0;
	}
	else
	{
		OP_CONTEXT *		pSaveCurrContext = m_pCurrContext;
		CONTEXT_PATH *		pSaveCurrContextPath = m_pCurrContextPath;
		PATH_PRED *			pSaveCurrPred = m_pCurrPred;
		FQNODE *				pSaveExprXPathSource = m_pExprXPathSource;

		// See if more than one index is being used.

		m_pCurrContext = m_pQuery->pContext;
		useLeafContext( TRUE);
		*puiIndex = 0;
		do
		{
			if (m_pCurrPred->pNodeSource)
			{
				FLMUINT	uiIndex;
				
				if (RC_BAD( rc = m_pCurrPred->pNodeSource->getIndex( ifpDb, &uiIndex,
															pbHaveMultiple)))
				{
					goto Exit;
				}
				if (uiIndex)
				{
					if (*puiIndex == 0)
					{
						*puiIndex = uiIndex;
					}
					if (*pbHaveMultiple)
					{
						break;
					}
					if (uiIndex != *puiIndex)
					{
						*pbHaveMultiple = TRUE;
						break;
					}
				}
			}
			else if (m_pCurrOpt->uiIxNum)
			{
				if (*puiIndex == 0)
				{
					*puiIndex = m_pCurrOpt->uiIxNum;
				}
				else if (m_pCurrOpt->uiIxNum != *puiIndex)
				{
					*pbHaveMultiple = TRUE;
					break;
				}
			}
		} while (useNextPredicate());

		// Restore the current predicate.

		m_pCurrContext = pSaveCurrContext;
		m_pCurrContextPath = pSaveCurrContextPath;
		m_pCurrPred = pSaveCurrPred;
		m_pExprXPathSource = pSaveExprXPathSource;
	}

Exit:

	if( bStartedTrans)
	{
		m_pDb->transAbort();
	}
	
	return( rc);
}

/****************************************************************************
Desc:	Make a copy of a value.
****************************************************************************/
RCODE F_Query::copyValue(
	FQVALUE *	pDestValue,
	FQVALUE *	pSrcValue
	)
{
	RCODE			rc = NE_XFLM_OK;

	pDestValue->eValType = pSrcValue->eValType;
	pDestValue->uiFlags = pSrcValue->uiFlags;

	// Cannot copy stream values.

	flmAssert( !(pDestValue->uiFlags & VAL_IS_STREAM));
	pDestValue->uiDataLen = pSrcValue->uiDataLen;
	switch (pDestValue->eValType)
	{
		case XFLM_BOOL_VAL:
			pDestValue->val.eBool = pSrcValue->val.eBool;
			break;
		case XFLM_UINT_VAL:
			pDestValue->val.uiVal = pSrcValue->val.uiVal;
			break;
		case XFLM_UINT64_VAL:
			pDestValue->val.ui64Val = pSrcValue->val.ui64Val;
			break;
		case XFLM_INT_VAL:
			pDestValue->val.iVal = pSrcValue->val.iVal;
			break;
		case XFLM_INT64_VAL:
			pDestValue->val.i64Val = pSrcValue->val.i64Val;
			break;
		case XFLM_BINARY_VAL:
		case XFLM_UTF8_VAL:
			if (pDestValue->uiDataLen)
			{
				if (RC_BAD( rc = m_pool.poolAlloc( pDestValue->uiDataLen,
													(void **)&pDestValue->val.pucBuf)))
				{
					goto Exit;
				}
				f_memcpy( pDestValue->val.pucBuf, pSrcValue->val.pucBuf,
								pDestValue->uiDataLen);
			}
			break;
		default:
			break;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Make a copy of a value.
****************************************************************************/
RCODE F_Query::copyXPath(
	XPATH_COMPONENT *	pXPathContext,
	FQNODE *				pDestNode,
	FXPATH **			ppDestXPath,
	FXPATH *				pSrcXPath
	)
{
	RCODE					rc = NE_XFLM_OK;
	FXPATH *				pDestXPath;
	XPATH_COMPONENT *	pXPathComponent;
	XPATH_COMPONENT *	pTmpXPathComponent;

	if (RC_BAD( rc = m_pool.poolCalloc( sizeof( FXPATH),
								(void **)&pDestXPath)))
	{
		goto Exit;
	}
	*ppDestXPath = pDestXPath;
	pXPathComponent = pSrcXPath->pFirstComponent;
	while (pXPathComponent)
	{
		if (RC_BAD( rc = m_pool.poolCalloc( sizeof( XPATH_COMPONENT),
									(void **)&pTmpXPathComponent)))
		{
			goto Exit;
		}
		if ((pTmpXPathComponent->pPrev = pDestXPath->pLastComponent) != NULL)
		{
			pTmpXPathComponent->pPrev->pNext = pTmpXPathComponent;
		}
		else
		{
			pDestXPath->pFirstComponent = pTmpXPathComponent;
		}
		pDestXPath->pLastComponent = pTmpXPathComponent;

		pTmpXPathComponent->pXPathContext = pXPathContext;
		pTmpXPathComponent->pXPathNode = pDestNode;
		pTmpXPathComponent->eXPathAxis = pXPathComponent->eXPathAxis;
		pTmpXPathComponent->eNodeType = pXPathComponent->eNodeType;
		pTmpXPathComponent->uiDictNum = pXPathComponent->uiDictNum;
		pTmpXPathComponent->uiContextPosNeeded = pXPathComponent->uiContextPosNeeded;
		if (pXPathComponent->pNodeSource)
		{
			if (RC_BAD( rc = pXPathComponent->pNodeSource->copy(
										&pTmpXPathComponent->pNodeSource)))
			{
				goto Exit;
			}
			
			// Call objectAddRef to add the node source to the list of objects
			// we have an AddRef on.  Then call Release, because the copy()
			// call above would have also done an AddRef()
			
			if (RC_BAD( rc = objectAddRef( pTmpXPathComponent->pNodeSource)))
			{
				goto Exit;
			}
			pTmpXPathComponent->pNodeSource->Release();
		}
		if (pXPathComponent->pContextPosExpr)
		{
			if (RC_BAD( rc = copyExpr( pTmpXPathComponent,
									&pTmpXPathComponent->pContextPosExpr,
									pXPathComponent->pContextPosExpr)))
			{
				goto Exit;
			}
		}
		if (pXPathComponent->pExpr)
		{
			if (RC_BAD( rc = copyExpr( pTmpXPathComponent,
									&pTmpXPathComponent->pExpr,
									pXPathComponent->pExpr)))
			{
				goto Exit;
			}
		}
		pXPathComponent = pXPathComponent->pNext;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Make a copy of a function.
****************************************************************************/
RCODE F_Query::copyFunction(
	XPATH_COMPONENT *	pXPathContext,
	FQFUNCTION **		ppDestFunc,
	FQFUNCTION *		pSrcFunc
	)
{
	RCODE					rc = NE_XFLM_OK;
	FQFUNCTION *		pDestFunc;
	FQEXPR *				pExpr;
	FQEXPR *				pTmpExpr;

	if (RC_BAD( rc = m_pool.poolCalloc( sizeof( FQFUNCTION),
								(void **)&pDestFunc)))
	{
		goto Exit;
	}
	*ppDestFunc = pDestFunc;
	pDestFunc->eFunction = pSrcFunc->eFunction;
	if (pSrcFunc->pFuncObj)
	{
		
		// Need to clone the object, because it may have state info.
		// on where it is at in iterating through values.
		
		if (RC_BAD( pSrcFunc->pFuncObj->cloneSelf( &pDestFunc->pFuncObj)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = objectAddRef( pDestFunc->pFuncObj)))
		{
			goto Exit;
		}
		
		// Need to release once because cloneSelf will have done
		// an AddRef() - only need the AddRef done by objectAddRef()
		
		pDestFunc->pFuncObj->Release();
	}
			
	pExpr = pSrcFunc->pFirstArg;
	while (pExpr)
	{
		if (RC_BAD( rc = m_pool.poolCalloc( sizeof( FQEXPR),
										(void **)&pTmpExpr)))
		{
			goto Exit;
		}
		if ((pTmpExpr->pPrev = pDestFunc->pLastArg) != NULL)
		{
			pTmpExpr->pPrev->pNext = pTmpExpr;
		}
		else
		{
			pDestFunc->pFirstArg = pTmpExpr;
		}
		pDestFunc->pLastArg = pTmpExpr;
		if (RC_BAD( rc = copyExpr( pXPathContext, &pTmpExpr->pExpr,
									pExpr->pExpr)))
		{
			goto Exit;
		}
		pExpr = pExpr->pNext;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Make a copy of a node.
****************************************************************************/
RCODE F_Query::copyNode(
	XPATH_COMPONENT *	pXPathContext,
	FQNODE **			ppDestNode,
	FQNODE *				pSrcNode
	)
{
	RCODE		rc = NE_XFLM_OK;
	FQNODE *	pDestNode;

	if (RC_BAD( rc = m_pool.poolCalloc( sizeof( FQNODE), (void **)&pDestNode)))
	{
		goto Exit;
	}
	*ppDestNode = pDestNode;
	pDestNode->eNodeType = pSrcNode->eNodeType;
	pDestNode->bNotted = pSrcNode->bNotted;
	switch (pSrcNode->eNodeType)
	{
		case FLM_OPERATOR_NODE:
			pDestNode->nd.op.eOperator = pSrcNode->nd.op.eOperator;
			pDestNode->nd.op.uiCompareRules = pSrcNode->nd.op.uiCompareRules;
			pDestNode->nd.op.pOpComparer = pSrcNode->nd.op.pOpComparer;
			if (pDestNode->nd.op.pOpComparer)
			{
				if (RC_BAD( rc = objectAddRef( pDestNode->nd.op.pOpComparer)))
				{
					goto Exit;
				}
			}
			break;
		case FLM_VALUE_NODE:
			if (RC_BAD( rc = copyValue( &pDestNode->currVal,
										&pSrcNode->currVal)))
			{
				goto Exit;
			}
			break;
		case FLM_XPATH_NODE:
			if (RC_BAD( rc = copyXPath( pXPathContext, pDestNode,
										&pDestNode->nd.pXPath,
										pSrcNode->nd.pXPath)))
			{
				goto Exit;
			}
			break;
		case FLM_FUNCTION_NODE:
			if (RC_BAD( rc = copyFunction( pXPathContext,
										&pDestNode->nd.pQFunction,
										pSrcNode->nd.pQFunction)))
			{
				goto Exit;
			}
			break;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Copy an expression
****************************************************************************/
RCODE F_Query::copyExpr(
	XPATH_COMPONENT *	pXPathContext,
	FQNODE **			ppDestExpr,
	FQNODE *				pSrcExpr
	)
{
	RCODE			rc = NE_XFLM_OK;
	FQNODE *		pTmpQNode;
	FQNODE *		pQNode = pSrcExpr;
	FQNODE *		pParent = NULL;
	FQNODE *		pPrevSib = NULL;

	if (!pQNode)
	{
		*ppDestExpr = NULL;
		goto Exit;  // rc = NE_XFLM_OK;
	}

	for (;;)
	{
		if (RC_BAD( rc = copyNode( pXPathContext, &pTmpQNode, pQNode)))
		{
			goto Exit;
		}
		if (!(*ppDestExpr))
		{
			*ppDestExpr = pTmpQNode;
		}
		pTmpQNode->pParent = pParent;
		if (pParent)
		{
			if ((pTmpQNode->pPrevSib = pPrevSib) == NULL)
			{
				pParent->pFirstChild = pTmpQNode;
			}
			else
			{
				pParent->pLastChild = pTmpQNode;
			}
		}
		if (pQNode->pFirstChild)
		{
			pParent = pTmpQNode;
			pPrevSib = NULL;
			pQNode = pQNode->pFirstChild;
		}
		else
		{
			while (!pQNode->pNextSib)
			{
				pParent = pTmpQNode->pParent;
				if ((pQNode = pQNode->pParent) == NULL)
				{

					flmAssert( !pParent);
					break;
				}
			}
			if (!pQNode)
			{
				break;
			}
			pQNode = pQNode->pNextSib;
			pPrevSib = pParent;
			pParent = pPrevSib->pParent;
		}
	}

	if (RC_BAD( rc = getPredicates( ppDestExpr, NULL, pXPathContext)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Copy criteria from another query object.
****************************************************************************/
RCODE XFLAPI F_Query::copyCriteria(
	IF_Query *	pSrcQuery
	)
{
	RCODE				rc = NE_XFLM_OK;
	EXPR_STATE *	pExprState = ((F_Query *)pSrcQuery)->m_pCurExprState;

	// Verify that the source query is in a "copyable" state

	if (pExprState)
	{
		if (pExprState->pPrev || pExprState->uiNestLevel ||
			  (pExprState->pLastNode &&
				pExprState->pLastNode->eNodeType == FLM_OPERATOR_NODE))
		{
			rc = RC_SET( NE_XFLM_Q_INCOMPLETE_QUERY_EXPR);
			goto Exit;
		}
	}

	// Clear out the existing query, if any.

	clearQuery();

	if (RC_BAD( rc = copyExpr( NULL, &m_pQuery,
								((F_Query *)pSrcQuery)->m_pQuery)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}
