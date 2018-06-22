//------------------------------------------------------------------------------
// Desc:	This file contains the FLAIM XML import and export utility classes
// Tabs:	3
//
// Copyright (c) 1999-2007 Novell, Inc. All Rights Reserved.
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

#ifndef FXPATH_H
#define FXPATH_H

typedef enum
{
	UNKNOWN_TOKEN = 0,									// 0
	OP_AND_TOKEN = XFLM_AND_OP,						// 1
	OP_OR_TOKEN = XFLM_OR_OP,							// 2
	OP_NOT_TOKEN = XFLM_NOT_OP,						//	3
	OP_EQ_TOKEN = XFLM_EQ_OP,							// 4
	OP_NE_TOKEN = XFLM_NE_OP,							// 5
	OP_APPROX_EQ_TOKEN = XFLM_APPROX_EQ_OP,		// 6
	OP_LT_TOKEN = XFLM_LT_OP,							// 7
	OP_LE_TOKEN = XFLM_LE_OP,							// 8
	OP_GT_TOKEN = XFLM_GT_OP,							// 9
	OP_GE_TOKEN = XFLM_GE_OP,							// 10
	OP_BITAND_TOKEN = XFLM_BITAND_OP,				// 11
	OP_BITOR_TOKEN = XFLM_BITOR_OP,					// 12
	OP_BITXOR_TOKEN = XFLM_BITXOR_OP,				// 13
	OP_MULT_TOKEN = XFLM_MULT_OP,						// 14
	OP_DIV_TOKEN = XFLM_DIV_OP,						// 15
	OP_MOD_TOKEN = XFLM_MOD_OP,						// 16
	OP_PLUS_TOKEN = XFLM_PLUS_OP,						// 17
	OP_MINUS_TOKEN = XFLM_MINUS_OP,					// 18
	OP_NEG_TOKEN = XFLM_NEG_OP,						// 19
	OP_LPAREN_TOKEN = XFLM_LPAREN_OP,				// 20
	OP_RPAREN_TOKEN = XFLM_RPAREN_OP,				// 21
	OP_COMMA_TOKEN = XFLM_COMMA_OP,					// 22
	OP_LBRACKET_TOKEN = XFLM_LBRACKET_OP,			// 23
	OP_RBRACKET_TOKEN = XFLM_RBRACKET_OP,			// 24
	OP_FSLASH_TOKEN,										// 25
	OP_DOUBLE_FSLASH_TOKEN,								// 26
	OP_UNION_TOKEN,										// 27
	PERIOD_TOKEN,											// 28
	DOUBLE_PERIOD_TOKEN,									// 29
	COMMA_TOKEN,											// 30
	DOUBLE_COLON_TOKEN,									// 31
	NAME_TEST_WILD_TOKEN,								// 32
	NAME_TEST_NCWILD_TOKEN,								// 33
	NAME_TEST_QNAME_TOKEN,								// 34
	NODE_TYPE_COMMENT_TOKEN,							// 35
	NODE_TYPE_TEXT_TOKEN,								// 36
	NODE_TYPE_PI_TOKEN,									// 37
	NODE_TYPE_NODE_TOKEN,								// 38
	AXIS_ANCESTOR_TOKEN,									// 39
	AXIS_ANCESTOR_OR_SELF_TOKEN,						// 40
	AXIS_ATTRIB_TOKEN,									// 41
	AXIS_CHILD_TOKEN,										// 42
	AXIS_DESCENDANT_TOKEN,								// 43
	AXIS_DESCENDANT_OR_SELF_TOKEN,					// 44
	AXIS_FOLLOWING_TOKEN,								// 45
	AXIS_FOLLOWING_SIB_TOKEN,							// 46
	AXIS_NAMESPACE_TOKEN,								// 47
	AXIS_PARENT_TOKEN,									// 48
	AXIS_PRECEDING_TOKEN,								// 49
	AXIS_PRECEDING_SIB_TOKEN,							// 50
	AXIS_SELF_TOKEN,										// 51
	AXIS_ATSIGN_TOKEN,									// 52
	AXIS_META_TOKEN,										// 53
	LITERAL_TOKEN,											// 54
	NUMBER_TOKEN,											// 55
	VAR_REF_TOKEN,											// 56
	LBRACE_TOKEN,											// 57
	RBRACE_TOKEN,											// 58
	FUNC_LAST_TOKEN,										// 59
	FUNC_POSITION_TOKEN,									// 60
	FUNC_COUNT_TOKEN,										// 61
	FUNC_ID_TOKEN,											// 62
	FUNC_LOCAL_NAME_TOKEN,								// 63
	FUNC_NAMESPACE_URI_TOKEN,							// 64
	FUNC_NAME_TOKEN,										// 65
	FUNC_STRING_TOKEN,									// 66
	FUNC_CONCAT_TOKEN,									// 67
	FUNC_STARTS_WITH_TOKEN,								// 68
	FUNC_CONTAINS_TOKEN,									// 69
	FUNC_SUBSTR_BEFORE_TOKEN,							// 70
	FUNC_SUBSTR_AFTER_TOKEN,							// 71
	FUNC_SUBSTR_TOKEN,									// 72
	FUNC_STR_LEN_TOKEN,									// 73
	FUNC_NORM_SPACE_TOKEN,								// 74
	FUNC_TRANSLATE_TOKEN,								// 75
	FUNC_NOT_TOKEN,										// 76
	FUNC_TRUE_TOKEN,										// 77
	FUNC_FALSE_TOKEN,										// 78
	FUNC_UNKNOWN_TOKEN,									// 79
	FUNC_LANG_TOKEN,										// 80
	FUNC_NUMBER_TOKEN,									// 81
	FUNC_SUM_TOKEN,										// 82
	FUNC_FLOOR_TOKEN,										// 83
	FUNC_CEILING_TOKEN,									// 84
	FUNC_ROUND_TOKEN,										// 85
	BINARY_TOKEN,											// 86
	FUNC_CB_TOKEN,											// 87
	END_TOKEN												// 88
} eXPathTokenType;

class F_XPathBase : public F_Object
{
public:

	FINLINE FLMBOOL isOperator(
		eXPathTokenType	eType)
	{
		switch( eType)
		{
			case OP_AND_TOKEN:
			case OP_OR_TOKEN:
			case OP_MOD_TOKEN:
			case OP_DIV_TOKEN:
			case OP_MULT_TOKEN:
			case OP_FSLASH_TOKEN:
			case OP_DOUBLE_FSLASH_TOKEN:
			case OP_UNION_TOKEN:
			case OP_PLUS_TOKEN:
			case OP_MINUS_TOKEN:
			case OP_EQ_TOKEN:
			case OP_NE_TOKEN:
			case OP_LT_TOKEN:
			case OP_LE_TOKEN:
			case OP_GT_TOKEN:
			case OP_GE_TOKEN:
				return( TRUE);
			default:
				break;
		}

		return( FALSE);
	}

	FINLINE FLMBOOL tokenCanHaveFlags(
		eXPathTokenType	eType)
	{
		switch( eType)
		{
			case OP_EQ_TOKEN:
			case OP_NE_TOKEN:
			case OP_LT_TOKEN:
			case OP_LE_TOKEN:
			case OP_GT_TOKEN:
			case OP_GE_TOKEN:
				return( TRUE);
			default:
				break;
		}

		return( FALSE);
	}

	FINLINE FLMBOOL isAxisSpecifier(
		eXPathTokenType		eType)
	{
		switch( eType)
		{
			case AXIS_ANCESTOR_TOKEN:
			case AXIS_ANCESTOR_OR_SELF_TOKEN:
			case AXIS_ATTRIB_TOKEN:
			case AXIS_CHILD_TOKEN:
			case AXIS_DESCENDANT_TOKEN:
			case AXIS_DESCENDANT_OR_SELF_TOKEN:
			case AXIS_FOLLOWING_TOKEN:
			case AXIS_FOLLOWING_SIB_TOKEN:
			case AXIS_NAMESPACE_TOKEN:
			case AXIS_PARENT_TOKEN:
			case AXIS_PRECEDING_TOKEN:
			case AXIS_PRECEDING_SIB_TOKEN:
			case AXIS_SELF_TOKEN:
			case AXIS_ATSIGN_TOKEN:
				return( TRUE);
			default:
				break;
		}

		return( FALSE);
	}
};

/*****************************************************************************
Desc:
******************************************************************************/
class F_XPathToken : public F_XPathBase
{
public:

	F_XPathToken()
	{
		m_pValBuf = NULL;
		m_uiValBufSize = 0;
		reset();
	}

	~F_XPathToken()
	{
		if( m_pValBuf)
		{
			f_free( &m_pValBuf);
		}
	}

	FINLINE void reset( void)
	{
		m_eTokenType = UNKNOWN_TOKEN;
		m_uiTokenFlags = 0;
		m_ui64Val = 0;
		m_puzPrefix = NULL;
		m_puzLocal = NULL;;
	}

	FINLINE RCODE resizeBuffer(
		FLMUINT			uiNewSize)
	{
		RCODE				rc = NE_XFLM_OK;
		void *			pOrigBuf = m_pValBuf;

		if( !m_pValBuf)
		{
			if( RC_BAD( rc = f_alloc( uiNewSize, &m_pValBuf)))
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = f_realloc( uiNewSize, &m_pValBuf)))
			{
				goto Exit;
			}

			if( m_puzPrefix)
			{
				m_puzPrefix = (FLMUNICODE *)(((FLMBYTE *)m_puzPrefix -
					(FLMBYTE *)pOrigBuf) + (FLMBYTE *)m_pValBuf);
			}

			if( m_puzLocal)
			{
				m_puzLocal = (FLMUNICODE *)(((FLMBYTE *)m_puzLocal -
					(FLMBYTE *)pOrigBuf) + (FLMBYTE *)m_pValBuf);
			}
		}

		m_uiValBufSize = uiNewSize;

	Exit:

		return( rc);
	}

	FINLINE eXPathTokenType getType( void)
	{
		return( m_eTokenType);
	}

	FINLINE FLMUNICODE * getPrefixPtr( void)
	{
		return( m_puzPrefix);
	}

	FINLINE FLMUNICODE * getLocalPtr( void)
	{
		return( m_puzLocal);
	}

	FINLINE FLMUINT64 getNumber( void)
	{
		return( m_ui64Val);
	}

	FINLINE FLMUINT getTokenFlags( void)
	{
		return( m_uiTokenFlags);
	}

private:

	eXPathTokenType 		m_eTokenType;
	FLMUINT					m_uiTokenFlags;
	void *					m_pValBuf;
	FLMUINT					m_uiValBufSize;
	FLMUINT					m_uiValBufLen;
	FLMUINT64				m_ui64Val;
	FLMUNICODE *			m_puzPrefix;
	FLMUNICODE *			m_puzLocal;

friend class F_XPathTokenizer;
friend class F_XPath;
};

/*****************************************************************************
Desc:
******************************************************************************/
class F_XPathTokenizer : public F_XPathBase, public F_XMLNamespaceMgr
{
public:

	F_XPathTokenizer()
	{
		m_pIStream = NULL;
		m_uiUngetCount = 0;
		m_eLastTokenType = UNKNOWN_TOKEN;
	}

	~F_XPathTokenizer()
	{
		if( m_pIStream)
		{
			m_pIStream->Release();
		}
	}

	RCODE setup(
		IF_IStream *		pIStream);

	RCODE getNextToken(
		F_XPathToken *		pToken);

private:

	RCODE skipWhitespace( void);

	RCODE getChar(
		FLMUNICODE *		puChar);

	RCODE peekChar(
		FLMUNICODE *		puChar);

	RCODE ungetChar(
		FLMUNICODE			uChar);

	RCODE getNumber(
		F_XPathToken *		pToken);

	RCODE getName(
		F_XPathToken *		pToken);

	RCODE getBinary(
		F_XPathToken *			pToken);

	RCODE getLiteral(
		F_XPathToken *		pToken);

	IF_IStream *			m_pIStream;
	eXPathTokenType	m_eLastTokenType;
	FLMUINT				m_uiUngetCount;
#define XPATH_MAX_UNGET_CHARS		4
	FLMUNICODE			m_uUngetBuf[ XPATH_MAX_UNGET_CHARS];
};

class F_XPathExpr;
class F_XPathPredicate;
class F_XPathAxisProducer;
class F_XPathStep;

/*****************************************************************************
Desc:
******************************************************************************/
class F_XPath : public F_XPathBase, public F_XMLNamespaceMgr
{
public:

	F_XPath()
	{
	}

	~F_XPath()
	{
	}

	RCODE parseQuery(
		F_Db *				pDb,
		IF_IStream *		pIStream,
		IF_Query *			pQuery);

	RCODE parseQuery(
		F_Db *				pDb,
		char *				pszQuery,
		IF_Query *			pQuery);

private:

	RCODE processFilterExpr(
		F_XPathExpr **		ppExpr);

	RCODE processPathExpr(
		F_XPathExpr **		ppExpr);

	RCODE processUnionExpr(
		F_XPathExpr **		ppExpr);

	RCODE processNodeTest(
		FLMBOOL				bAttr,
		F_XPathExpr **		ppExpr);

	RCODE processStep(
		F_XPathExpr **		ppExpr);

	RCODE processRelativeLocationPath(
		F_XPathExpr **		ppExpr);

	RCODE processUnaryExpr(
		F_XPathExpr **		ppExpr);

	RCODE processOrExpr(
		F_XPathExpr **		ppExpr);

	RCODE processAndExpr(
		F_XPathExpr **		ppExpr);

	RCODE processEqualityExpr(
		F_XPathExpr **		ppExpr);

	RCODE processRelationalExpr(
		F_XPathExpr **		ppExpr);

	RCODE processAdditiveExpr(
		F_XPathExpr **		ppExpr);

	RCODE processMultiplicativeExpr(
		F_XPathExpr **		ppExpr);

	RCODE processPrimaryExpr(
		F_XPathExpr **		ppExpr);

	RCODE getNextToken( void);

	F_XPathTokenizer		m_tokenizer;
	F_XPathToken			m_curToken;
};

#endif // FXPATH_H
