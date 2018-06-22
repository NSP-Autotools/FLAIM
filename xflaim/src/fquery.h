//------------------------------------------------------------------------------
// Desc:	Contains defines, structures and prototypes for FLAIM cursors.
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

#ifndef FQUERY_H
#define FQUERY_H

#define FLM_MAX_POS_KEYS			1000
#define FLM_MIN_POS_KEYS			250
#define FLM_ADDR_GROW_SIZE			100
#define FLM_KEYS_GROW_SIZE			100

/****************************************************************************
Macro:
Desc:	 	Macros used for determining the nature and precedence of OP codes.
****************************************************************************/

class F_DataVector;
class FSIndexCursor;
class FSDataCursor;
class F_CollIStream;

typedef struct QueryValue *			FQVALUE_p;
typedef struct QueryXPathPath *		FXPATH_p;
typedef struct XPathComponent *		XPATH_COMPONENT_p;
typedef struct QueryExpr *				FQEXPR_p;
typedef struct QueryNode *				FQNODE_p;

typedef enum QueryNodeTypes
{
	FLM_OPERATOR_NODE = 0,
	FLM_VALUE_NODE,
	FLM_XPATH_NODE,
	FLM_FUNCTION_NODE
} eNodeTypes;

/****************************************************************************
Structures used for the query tree and other stuff
****************************************************************************/

typedef struct QueryValue
{
	eValTypes	eValType;
	FLMUINT		uiFlags;
#define VAL_IS_STREAM		0x0001
#define VAL_IS_CONSTANT		0x0002	// During query evaluation, this indicates
												// that this value is a constant.  If it
												// is a FLM_UTF8_VAL, then asterisks will
												// be treated as a wildcard, unless
												// escaped (\*).  If the value is NOT
												// a constant, the asterisk is NEVER
												// treated as a wildcard, and the
												// backslash is NEVER treated as an
												// escape character.
#define VAL_HAS_WILDCARDS	0x0004	// This is only set if the value is a
												// constant, FLM_UTF8_VAL, that has
												// wildcards.
	FLMUINT		uiDataLen;				// Length in bytes if the type is text
												// or binary
	union
	{
		XFlmBoolType			eBool;
		FLMUINT					uiVal;
		FLMUINT64				ui64Val;
		FLMINT					iVal;
		FLMINT64					i64Val;
		FLMBYTE *				pucBuf;
		IF_PosIStream *		pIStream;
	} val;									// Holds or points to the atom value.
} FQVALUE;

/***************************************************************************
Desc:	Can two values be compared?
***************************************************************************/
FINLINE FLMBOOL fqCanCompare(
	FQVALUE *	pValue1,
	FQVALUE *	pValue2
	)
{
	if (!pValue1 || !pValue2 ||
		 pValue1->eValType == pValue2->eValType)
	{
		return( TRUE);
	}
	else
	{
		switch (pValue1->eValType)
		{
			case XFLM_UINT_VAL:
			case XFLM_UINT64_VAL:
			case XFLM_INT_VAL:
			case XFLM_INT64_VAL:
				return( (FLMBOOL)(pValue2->eValType == XFLM_UINT_VAL ||
										pValue2->eValType == XFLM_UINT64_VAL ||
										pValue2->eValType == XFLM_INT_VAL ||
										pValue2->eValType == XFLM_INT64_VAL
										? TRUE
										: FALSE));
			default:
				return( FALSE);
		}
	}
}

typedef struct QueryExpr
{
	FQNODE_p	pExpr;
	FQEXPR_p	pNext;
	FQEXPR_p	pPrev;
} FQEXPR;

typedef struct PathPred *		PATH_PRED_p;
typedef struct PathPredNode *	PATH_PRED_NODE_p;

typedef struct XPathComponent
{
	FLMBOOL					bIsSource;			// Indicates component is query source
	PATH_PRED_p				pOptPred;			// Optimization predicate
	IF_DOMNode *			pCurrNode;			// Used when evaluating expressions
	IF_DOMNode *			pKeyNode;
	XPATH_COMPONENT_p		pXPathContext;
	XPATH_COMPONENT_p		pNext;
	XPATH_COMPONENT_p		pPrev;
	FQNODE_p					pXPathNode;
	eXPathAxisTypes		eXPathAxis;
	eDomNodeType			eNodeType;
	IF_QueryNodeSource *	pNodeSource;
	FLMUINT					uiDictNum;
	FLMUINT					uiContextPosNeeded;
	FQNODE_p					pContextPosExpr;
	FQNODE_p					pExpr;
	FQNODE_p					pExprXPathSource;	// XPATH node that is expression's source
} XPATH_COMPONENT;

typedef struct QueryXPath
{
	FLMBOOL				bGettingNodes;		// Used when evaluating expressions
	FLMBOOL				bIsSource;			// Indicates XPATH is query source
	XPATH_COMPONENT_p	pSourceComponent;	// Used when XPATH is query source
	FLMBOOL				bHavePassingNode;
	XPATH_COMPONENT_p	pFirstComponent;
	XPATH_COMPONENT_p	pLastComponent;
} FXPATH;

typedef struct PathPredNode
{
	FQNODE_p				pXPathNode;
	PATH_PRED_NODE_p	pNext;
} PATH_PRED_NODE;

typedef struct PathPred
{
	PATH_PRED_NODE *		pXPathNodeList;// List of XPATHs sharing this predicate.
	eQueryOperators		eOperator;		// Operator of the predicate
	FLMUINT					uiCompareRules;// Comparison rules
	IF_OperandComparer *	pOpComparer;	// Function to perform comparison
	FQNODE_p					pContextNode;	// Context node for this predicate, if
													// any
	FLMBOOL					bNotted;			// Has operator been notted?
	FQVALUE *				pFromValue;		// Points to FQVALUE that has the FROM value for
													// this predicate.  Will be NULL for unary
													// operators such as exists
	FLMBOOL					bInclFrom;		// Flag indicating if the from value is
													// inclusive.
	FQVALUE *				pUntilValue;	// Points to FQValue that has the UNTIL value
													// for this predicate.
	FLMBOOL					bInclUntil;		// Flag indicating if until value is
													// inclusive.
	XFLM_OPT_INFO			OptInfo;			// Optimization information.
	FSIndexCursor *		pFSIndexCursor;// Used if OptInfo.eOptType is
													// QOPT_USING_INDEX
	FSCollectionCursor *	pFSCollectionCursor;// Used if OptInfo.eOptType is
													// QOPT_USING_NODE_ID
	IF_QueryNodeSource *	pNodeSource;	// Used if OptInfo.eOptType is
													// QOPT_USING_APP_SOURCE
	PATH_PRED_p				pNext;
	PATH_PRED_p				pPrev;
} PATH_PRED;

typedef struct CONTEXT_PATH
{
	XPATH_COMPONENT *	pXPathComponent;
	FLMUINT				uiCost;
	FLMBOOL				bMustScan;
	PATH_PRED *			pSelectedPred;		// Only used for intersect contexts
	PATH_PRED *			pFirstPred;
	PATH_PRED *			pLastPred;
	CONTEXT_PATH *		pNext;
	CONTEXT_PATH *		pPrev;
} CONTEXT_PATH;

typedef struct OP_CONTEXT
{
	FLMBOOL				bIntersect;
	FLMBOOL				bMustScan;
	FLMBOOL				bForceOptToScan;
	FQNODE_p				pQRootNode;			// Root node of this context.
	FLMUINT				uiCost;
	OP_CONTEXT *		pSelectedChild;	// Only used for intersect contexts
	CONTEXT_PATH *		pSelectedPath;		// Only used for intersect contexts
	OP_CONTEXT *		pParent;
	OP_CONTEXT *		pFirstChild;
	OP_CONTEXT *		pLastChild;
	OP_CONTEXT *		pNextSib;
	OP_CONTEXT *		pPrevSib;
	CONTEXT_PATH *		pFirstPath;
	CONTEXT_PATH *		pLastPath;
} OP_CONTEXT;

typedef struct QueryFunction
{
	eQueryFunctions	eFunction;
	IF_QueryValFunc *	pFuncObj;
	FQEXPR_p				pFirstArg;
	FQEXPR_p				pLastArg;
} FQFUNCTION;

typedef struct QueryOperator
{
	eQueryOperators		eOperator;
	FLMUINT					uiCompareRules;
	IF_OperandComparer *	pOpComparer;
} FQOPERATOR;

typedef struct QueryNode
{
	eNodeTypes		eNodeType;			// Type of node this is
	FLMUINT			uiNestLevel;		// Nesting level of node - only used when
												// setting up the query
	OP_CONTEXT *	pContext;
	FQVALUE			currVal;				// Current value - used during evaluation
												// and for value types
	FLMBOOL			bUsedValue;			// Used during evaluation
	FLMBOOL			bLastValue;			// Used during evaluation
	FLMBOOL			bNotted;
	FQNODE_p			pParent;				// Parent of this query node
	FQNODE_p			pPrevSib;			// Previous sibling of this query node
	FQNODE_p			pNextSib;			// Next sibling of this query node
	FQNODE_p			pFirstChild;		// First child of this query node
	FQNODE_p			pLastChild;			// Last child of this query node
	union
	{
		FQOPERATOR				op;
		FQFUNCTION *			pQFunction;
		FXPATH *					pXPath;
	} nd;
} FQNODE;

RCODE fqCompare(							// fqeval.cpp
	FQVALUE *				pValue1,
	FQVALUE *				pValue2,
	FLMUINT					uiCompareRules,
	IF_OperandComparer *	pOpComparer,
	FLMUINT					uiLanguage,
	FLMINT *					piCmp);

RCODE fqCompareOperands(				// fqeval.cpp
	FLMUINT					uiLanguage,
	FQVALUE *				pLValue,
	FQVALUE *				pRValue,
	eQueryOperators		eOperator,
	FLMUINT					uiCompareRules,
	IF_OperandComparer *	pOpComparer,
	FLMBOOL					bNotted,
	XFlmBoolType *			peBool);

RCODE fqArithmeticOperator(			// fqeval.cpp
	FQVALUE *			pLValue,
	FQVALUE *			pRValue,
	eQueryOperators	eOperator,
	FQVALUE *			pResult);

RCODE fqCompareCollStreams(			// fqeval.cpp
	F_CollIStream *	pLStream,
	F_CollIStream *	pRStream,
	FLMBOOL				bOpIsMatch,
	FLMUINT				uiLanguage,
	FLMINT *				piResult);

RCODE flmBuildFromAndUntilKeys(		// kybldkey.cpp
	IXD *				pIxd,
	PATH_PRED *		pPred,
	F_DataVector *	pFromSearchKey,
	FLMBYTE *		pucFromKey,	
	FLMUINT *		puiFromKeyLen,
	F_DataVector *	pUntilSearchKey,
	FLMBYTE *		pucUntilKey,	
	FLMUINT *		puiUntilKeyLen,
	FLMBOOL *		pbDoNodeMatch,
	FLMBOOL *		pbCanCompareOnKey);

#endif   // FQUERY_H

