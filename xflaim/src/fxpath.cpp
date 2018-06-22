//------------------------------------------------------------------------------
//	Desc:	Methods for parsing and evaluating XPATH queries
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

FSTATIC RCODE addCallbackFunc(
	IF_Query *	pQuery);

/****************************************************************************
Desc:		Setup the tokenizer
****************************************************************************/
RCODE F_XPathTokenizer::setup(
	IF_IStream *	pIStream)
{
	if( m_pIStream)
	{
		m_pIStream->Release();
		m_pIStream = NULL;
	}

	if( pIStream)
	{
		m_pIStream = pIStream;
		m_pIStream->AddRef();
	}

	return( NE_XFLM_OK);
}

/****************************************************************************
Desc:		Skips any whitespace characters starting at the current location
****************************************************************************/
RCODE F_XPathTokenizer::skipWhitespace( void)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUNICODE		uChar;

	for( ;;)
	{
		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		if( !gv_XFlmSysData.pXml->isWhitespace( uChar))
		{
			if( RC_BAD( rc = ungetChar( uChar)))
			{
				goto Exit;
			}

			break;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Reads the next character from the stream.
			If no more characters are available, a NULL (0) character will be
			returned
****************************************************************************/
RCODE F_XPathTokenizer::getChar(
	FLMUNICODE *		puChar)
{
	RCODE		rc = NE_XFLM_OK;

	if( m_uiUngetCount)
	{
		*puChar = m_uUngetBuf[ --m_uiUngetCount];
	}
	else
	{
		if( RC_BAD( rc = f_readUTF8CharAsUnicode( m_pIStream, puChar)))
		{
			if( rc != NE_XFLM_EOF_HIT)
			{
				goto Exit;
			}

			rc = NE_XFLM_OK;
			*puChar = 0;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Returns the next character that will be read from the stream.
			If no more characters are available, a NULL (0) character will be
			returned
****************************************************************************/
RCODE F_XPathTokenizer::peekChar(
	FLMUNICODE *		puChar)
{
	RCODE				rc = NE_XFLM_OK;

	if( m_uiUngetCount)
	{
		*puChar = m_uUngetBuf[ m_uiUngetCount - 1];
	}
	else
	{
		if( RC_BAD( rc = f_readUTF8CharAsUnicode( m_pIStream, puChar)))
		{
			if( rc == NE_XFLM_EOF_HIT)
			{
				*puChar = 0;
				rc = NE_XFLM_OK;
			}
			goto Exit;
		}

		if( RC_BAD( rc = ungetChar( *puChar)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Pushes the passed-in character onto the unget stack
****************************************************************************/
RCODE F_XPathTokenizer::ungetChar(
	FLMUNICODE 			uChar)
{
	RCODE		rc = NE_XFLM_OK;

	if( !uChar)
	{
		goto Exit;
	}

	if( m_uiUngetCount == XPATH_MAX_UNGET_CHARS)
	{
		rc = RC_SET( NE_XFLM_BUFFER_OVERFLOW);
		goto Exit;
	}

	m_uUngetBuf[ m_uiUngetCount++] = uChar;

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Gets the next XPATH token from the query stream
****************************************************************************/
RCODE F_XPathTokenizer::getNextToken(
	F_XPathToken *			pToken)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUNICODE		uChar;
	FLMBOOL			bConsumeParens = FALSE;

	pToken->reset();

	if( RC_BAD( rc = skipWhitespace()))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	switch( uChar)
	{
		case 0:
		{
			pToken->m_eTokenType = END_TOKEN;
			break;
		}

		case FLM_UNICODE_LPAREN:
		{
			pToken->m_eTokenType = OP_LPAREN_TOKEN;
			break;
		}

		case FLM_UNICODE_RPAREN:
		{
			pToken->m_eTokenType = OP_RPAREN_TOKEN;
			break;
		}

		case FLM_UNICODE_LBRACKET:
		{
			pToken->m_eTokenType =  OP_LBRACKET_TOKEN;
			break;
		}

		case FLM_UNICODE_RBRACKET:
		{
			pToken->m_eTokenType = OP_RBRACKET_TOKEN;
			break;
		}

		case FLM_UNICODE_PERIOD:
		{
			if( RC_BAD( rc = peekChar( &uChar)))
			{
				goto Exit;
			}

			if( uChar == FLM_UNICODE_PERIOD)
			{
				if( RC_BAD( rc = getChar( &uChar)))
				{
					goto Exit;
				}
				pToken->m_eTokenType = DOUBLE_PERIOD_TOKEN;
			}
			else
			{
				pToken->m_eTokenType = PERIOD_TOKEN;
			}
			break;
		}

		case FLM_UNICODE_ATSIGN:
		{
			pToken->m_eTokenType = AXIS_ATSIGN_TOKEN;
			pToken->m_ui64Val = (FLMUINT64)ATTRIBUTE_AXIS;
			break;
		}

		case FLM_UNICODE_COMMA:
		{
			pToken->m_eTokenType = COMMA_TOKEN;
			break;
		}

		case FLM_UNICODE_COLON:
		{
			if( RC_BAD( rc = getChar( &uChar)))
			{
				goto Exit;
			}

			if( uChar != FLM_UNICODE_COLON)
			{
				rc = RC_SET( NE_XFLM_SYNTAX);
				goto Exit;
			}

			pToken->m_eTokenType = DOUBLE_COLON_TOKEN;
			break;
		}

		case FLM_UNICODE_FSLASH:
		{
			if( RC_BAD( rc = peekChar( &uChar)))
			{
				goto Exit;
			}

			if( uChar == FLM_UNICODE_FSLASH)
			{
				if( RC_BAD( rc = getChar( &uChar)))
				{
					goto Exit;
				}
				pToken->m_eTokenType = OP_DOUBLE_FSLASH_TOKEN;
			}
			else
			{
				pToken->m_eTokenType = OP_FSLASH_TOKEN;
			}
			break;
		}

		case FLM_UNICODE_PIPE:
		{
			if( RC_BAD( rc = peekChar( &uChar)))
			{
				goto Exit;
			}

			if( uChar == FLM_UNICODE_PIPE)
			{
				if( RC_BAD( rc = getChar( &uChar)))
				{
					goto Exit;
				}
				pToken->m_eTokenType = OP_OR_TOKEN;
			}
			else
			{
				pToken->m_eTokenType = OP_UNION_TOKEN;
			}
			break;
		}

		case FLM_UNICODE_AMP:
		{
			if( RC_BAD( rc = peekChar( &uChar)))
			{
				goto Exit;
			}

			if( uChar == FLM_UNICODE_AMP)
			{
				if( RC_BAD( rc = getChar( &uChar)))
				{
					goto Exit;
				}
				pToken->m_eTokenType = OP_AND_TOKEN;
			}
			else
			{
				pToken->m_eTokenType = OP_BITAND_TOKEN;
			}
			break;
		}

		case FLM_UNICODE_PLUS:
		{
			pToken->m_eTokenType = OP_PLUS_TOKEN;
			break;
		}

		case FLM_UNICODE_HYPHEN:
		{
			pToken->m_eTokenType = OP_MINUS_TOKEN;
			break;
		}

		case FLM_UNICODE_TILDE:
		{
			if( RC_BAD( rc = getChar( &uChar)))
			{
				goto Exit;
			}

			if( uChar != FLM_UNICODE_EQ)
			{
				rc = RC_SET( NE_XFLM_SYNTAX);
				goto Exit;
			}

			pToken->m_eTokenType = OP_APPROX_EQ_TOKEN;
			break;
		}

		case FLM_UNICODE_EQ:
		{
			if( RC_BAD( rc = peekChar( &uChar)))
			{
				goto Exit;
			}

			// Check for double equal and allow it
			// in expressions

			if( uChar == FLM_UNICODE_EQ)
			{
				if( RC_BAD( rc = getChar( &uChar)))
				{
					goto Exit;
				}
			}

			pToken->m_eTokenType = OP_EQ_TOKEN;
			break;
		}

		case FLM_UNICODE_BANG:
		{
			if( RC_BAD( rc = peekChar( &uChar)))
			{
				goto Exit;
			}

			if( uChar == FLM_UNICODE_EQ)
			{
				if( RC_BAD( rc = getChar( &uChar)))
				{
					goto Exit;
				}
				pToken->m_eTokenType = OP_NE_TOKEN;
			}
			else
			{
				pToken->m_eTokenType = OP_NOT_TOKEN;
			}

			break;
		}

		case FLM_UNICODE_LT:
		{
			if( RC_BAD( rc = peekChar( &uChar)))
			{
				goto Exit;
			}

			if( uChar == FLM_UNICODE_EQ)
			{
				if( RC_BAD( rc = getChar( &uChar)))
				{
					goto Exit;
				}

				pToken->m_eTokenType = OP_LE_TOKEN;
			}
			else
			{
				pToken->m_eTokenType = OP_LT_TOKEN;
			}
			break;
		}

		case FLM_UNICODE_GT:
		{
			if( RC_BAD( rc = peekChar( &uChar)))
			{
				goto Exit;
			}

			if( uChar == FLM_UNICODE_EQ)
			{
				if( RC_BAD( rc = getChar( &uChar)))
				{
					goto Exit;
				}

				pToken->m_eTokenType = OP_GE_TOKEN;
			}
			else
			{
				pToken->m_eTokenType = OP_GT_TOKEN;
			}
			break;
		}

		case FLM_UNICODE_ASTERISK:
		{
			if( m_eLastTokenType != UNKNOWN_TOKEN &&
				m_eLastTokenType != AXIS_ATSIGN_TOKEN &&
				m_eLastTokenType != DOUBLE_COLON_TOKEN &&
				m_eLastTokenType != OP_LPAREN_TOKEN &&
				m_eLastTokenType != OP_LBRACKET_TOKEN &&
				m_eLastTokenType != COMMA_TOKEN &&
				!isOperator( m_eLastTokenType))
			{
				pToken->m_eTokenType = OP_MULT_TOKEN;
			}
			else
			{
				pToken->m_eTokenType = NAME_TEST_WILD_TOKEN;
			}
			break;
		}

		case FLM_UNICODE_DOLLAR:
		{
			if( RC_BAD( rc = getName( pToken)))
			{
				goto Exit;
			}

			pToken->m_eTokenType = VAR_REF_TOKEN;
			break;
		}

		case FLM_UNICODE_0:
		case FLM_UNICODE_1:
		case FLM_UNICODE_2:
		case FLM_UNICODE_3:
		case FLM_UNICODE_4:
		case FLM_UNICODE_5:
		case FLM_UNICODE_6:
		case FLM_UNICODE_7:
		case FLM_UNICODE_8:
		case FLM_UNICODE_9:
		{
			if( RC_BAD( rc = ungetChar( uChar)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = getNumber( pToken)))
			{
				goto Exit;
			}

			pToken->m_eTokenType = NUMBER_TOKEN;
			break;
		}

		case FLM_UNICODE_APOS:
		case FLM_UNICODE_QUOTE:
		{
			if( RC_BAD( rc = ungetChar( uChar)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = getLiteral( pToken)))
			{
				goto Exit;
			}

			pToken->m_eTokenType = LITERAL_TOKEN;
			break;
		}

		case FLM_UNICODE_LBRACE:
		{
			pToken->m_eTokenType = LBRACE_TOKEN;
			break;
		}

		case FLM_UNICODE_RBRACE:
		{
			pToken->m_eTokenType = RBRACE_TOKEN;
			break;
		}

		default:
		{
			if( RC_BAD( rc = ungetChar( uChar)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = getName( pToken)))
			{
				goto Exit;
			}

			if( pToken->m_puzPrefix)
			{
				pToken->m_eTokenType = NAME_TEST_QNAME_TOKEN;
				goto Exit;
			}

			if( RC_BAD( rc = peekChar( &uChar)))
			{
				goto Exit;
			}

			// Node type or function name?

			if( uChar == FLM_UNICODE_LPAREN)
			{
				switch( pToken->m_puzLocal[ 0])
				{
				case FLM_UNICODE_b:
						if( f_uninativecmp( pToken->m_puzLocal, "binary") == 0)
						{
							pToken->m_eTokenType = BINARY_TOKEN;
							if (RC_BAD( rc = getChar( &uChar)))
							{
								goto Exit;
							}
							if( RC_BAD( rc = skipWhitespace()))
							{
								goto Exit;
							}
							if (RC_BAD( rc = peekChar( &uChar)))
							{
								goto Exit;
							}
							if (uChar != FLM_UNICODE_APOS && uChar != FLM_UNICODE_QUOTE)
							{
								rc = RC_SET( NE_XFLM_SYNTAX);
								goto Exit;
							}
							if( RC_BAD( rc = getBinary( pToken)))
							{
								goto Exit;
							}
							if( RC_BAD( rc = skipWhitespace()))
							{
								goto Exit;
							}
							if (RC_BAD( rc = getChar( &uChar)))
							{
								goto Exit;
							}
							if (uChar != FLM_UNICODE_RPAREN)
							{
								rc = RC_SET( NE_XFLM_SYNTAX);
								goto Exit;
							}
						}
						else
						{
							rc = RC_SET( NE_XFLM_SYNTAX);
							goto Exit;
						}
						break;

					case FLM_UNICODE_c:
						if( f_uninativecmp( pToken->m_puzLocal, "ceiling") == 0)
						{
							pToken->m_eTokenType = FUNC_CEILING_TOKEN;
						}
						if( f_uninativecmp( pToken->m_puzLocal, "comment") == 0)
						{
							pToken->m_eTokenType = NODE_TYPE_COMMENT_TOKEN;
						}
						else if( f_uninativecmp( pToken->m_puzLocal, "concat") == 0)
						{
							pToken->m_eTokenType = FUNC_CONCAT_TOKEN;
						}
						else if( f_uninativecmp( 
							pToken->m_puzLocal, "contains") == 0)
						{
							pToken->m_eTokenType = FUNC_CONTAINS_TOKEN;
						}
						else if( f_uninativecmp( pToken->m_puzLocal, "count") == 0)
						{
							pToken->m_eTokenType = FUNC_COUNT_TOKEN;
						}
						else
						{
							rc = RC_SET( NE_XFLM_SYNTAX);
							goto Exit;
						}
						break;

					case FLM_UNICODE_f:
						if( f_uninativecmp( pToken->m_puzLocal, "false") == 0)
						{
							pToken->m_eTokenType = FUNC_FALSE_TOKEN;
						}
						else if( f_uninativecmp( pToken->m_puzLocal,
							"floor") == 0)
						{
							pToken->m_eTokenType = FUNC_FLOOR_TOKEN;
						}
						else if (f_uninativecmp( pToken->m_puzLocal,
							"funcCB") == 0)
						{
							pToken->m_eTokenType = FUNC_CB_TOKEN;
						}
						else
						{
							rc = RC_SET( NE_XFLM_SYNTAX);
							goto Exit;
						}
						break;

					case FLM_UNICODE_i:
						if( f_uninativecmp( pToken->m_puzLocal, "id") == 0)
						{
							pToken->m_eTokenType = FUNC_ID_TOKEN;
						}
						else
						{
							rc = RC_SET( NE_XFLM_SYNTAX);
							goto Exit;
						}
						break;

					case FLM_UNICODE_l:
						if( f_uninativecmp( pToken->m_puzLocal, "lang") == 0)
						{
							pToken->m_eTokenType = FUNC_LANG_TOKEN;
						}
						else if( f_uninativecmp( pToken->m_puzLocal, "last") == 0)
						{
							pToken->m_eTokenType = FUNC_LAST_TOKEN;
						}
						else if( f_uninativecmp( 
							pToken->m_puzLocal, "local-name") == 0)
						{
							pToken->m_eTokenType = FUNC_LOCAL_NAME_TOKEN;
						}
						else
						{
							rc = RC_SET( NE_XFLM_SYNTAX);
							goto Exit;
						}
						break;

					case FLM_UNICODE_n:
						if( f_uninativecmp( pToken->m_puzLocal, "name") == 0)
						{
							pToken->m_eTokenType = FUNC_NAME_TOKEN;
						}
						else if( f_uninativecmp( pToken->m_puzLocal, 
							"namespace-uri") == 0)
						{
							pToken->m_eTokenType = FUNC_NAMESPACE_URI_TOKEN;
						}
						else if( f_uninativecmp( pToken->m_puzLocal, "node") == 0)
						{
							pToken->m_eTokenType = NODE_TYPE_NODE_TOKEN;
							bConsumeParens = TRUE;
						}
						else if( f_uninativecmp( pToken->m_puzLocal,
							"normalize-space") == 0)
						{
							pToken->m_eTokenType = FUNC_NORM_SPACE_TOKEN;
						}
						else if( f_uninativecmp( pToken->m_puzLocal, "not") == 0)
						{
							pToken->m_eTokenType = FUNC_NOT_TOKEN;
						}
						else if( f_uninativecmp( pToken->m_puzLocal, "number") == 0)
						{
							pToken->m_eTokenType = FUNC_NUMBER_TOKEN;
						}
						else
						{
							rc = RC_SET( NE_XFLM_SYNTAX);
							goto Exit;
						}
						break;

					case FLM_UNICODE_p:
						if( f_uninativecmp( pToken->m_puzLocal, "position") == 0)
						{
							pToken->m_eTokenType = FUNC_POSITION_TOKEN;
						}
						else if( f_uninativecmp( pToken->m_puzLocal,
							"processing-instruction") == 0)
						{
							pToken->m_eTokenType = NODE_TYPE_PI_TOKEN;
						}
						else
						{
							rc = RC_SET( NE_XFLM_SYNTAX);
							goto Exit;
						}
						break;

					case FLM_UNICODE_r:
						if( f_uninativecmp( pToken->m_puzLocal,
							"round") == 0)
						{
							pToken->m_eTokenType = FUNC_ROUND_TOKEN;
						}
						else
						{
							rc = RC_SET( NE_XFLM_SYNTAX);
							goto Exit;
						}
						break;

					case FLM_UNICODE_s:
						if( f_uninativecmp( pToken->m_puzLocal,
							"starts-with") == 0)
						{
							pToken->m_eTokenType = FUNC_STARTS_WITH_TOKEN;
						}
						else if( f_uninativecmp( pToken->m_puzLocal, 
							"string") == 0)
						{
							pToken->m_eTokenType = FUNC_STRING_TOKEN;
						}
						else if( f_uninativecmp( pToken->m_puzLocal,
							"string-length") == 0)
						{
							pToken->m_eTokenType = FUNC_STR_LEN_TOKEN;
						}
						else if( f_uninativecmp( pToken->m_puzLocal,
							"substring") == 0)
						{
							pToken->m_eTokenType = FUNC_SUBSTR_TOKEN;
						}
						else if( f_uninativecmp( pToken->m_puzLocal,
							"substring-after") == 0)
						{
							pToken->m_eTokenType = FUNC_SUBSTR_AFTER_TOKEN;
						}
						else if( f_uninativecmp( pToken->m_puzLocal,
							"substring-before") == 0)
						{
							pToken->m_eTokenType = FUNC_SUBSTR_BEFORE_TOKEN;
						}
						else if( f_uninativecmp( pToken->m_puzLocal, "sum") == 0)
						{
							pToken->m_eTokenType = FUNC_SUM_TOKEN;
						}
						else
						{
							rc = RC_SET( NE_XFLM_SYNTAX);
							goto Exit;
						}
						break;

					case FLM_UNICODE_t:
						if( f_uninativecmp( pToken->m_puzLocal,
							"text") == 0)
						{
							pToken->m_eTokenType = NODE_TYPE_TEXT_TOKEN;
							bConsumeParens = TRUE;
						}
						else if( f_uninativecmp( pToken->m_puzLocal,
							"translate") == 0)
						{
							pToken->m_eTokenType = FUNC_TRANSLATE_TOKEN;
						}
						else if( f_uninativecmp( pToken->m_puzLocal,
							"true") == 0)
						{
							pToken->m_eTokenType = FUNC_TRUE_TOKEN;
						}
						else
						{
							rc = RC_SET( NE_XFLM_SYNTAX);
							goto Exit;
						}
						break;
					case FLM_UNICODE_u:
						if( f_uninativecmp( pToken->m_puzLocal,
							"unknown") == 0)
						{
							pToken->m_eTokenType = FUNC_UNKNOWN_TOKEN;
						}
						else
						{
							rc = RC_SET( NE_XFLM_SYNTAX);
							goto Exit;
						}
						break;

					default:
						rc = RC_SET( NE_XFLM_SYNTAX);
						goto Exit;
				}

				goto Exit;
			}

 			// Axis specifier or node name?

			if( uChar == FLM_UNICODE_COLON)
			{
				if( RC_BAD( rc = getChar( &uChar)))
				{
					goto Exit;
				}

				if( RC_BAD( rc = peekChar( &uChar)))
				{
					goto Exit;
				}

				if( uChar == FLM_UNICODE_COLON)
				{
					if( RC_BAD( rc = ungetChar( FLM_UNICODE_COLON)))
					{
						goto Exit;
					}

					// If we have reached this point, we have a name followed
					// by '::'.  Verify that the name is a valid axis specifier.

					switch( pToken->m_puzLocal[ 0])
					{
						case FLM_UNICODE_a:
							if( f_uninativecmp( pToken->m_puzLocal,
								"ancestor") == 0)
							{
								pToken->m_eTokenType = AXIS_ANCESTOR_TOKEN;
								pToken->m_ui64Val = (FLMUINT64)ANCESTOR_AXIS;
							}
							else if( f_uninativecmp( pToken->m_puzLocal,
								"ancestor-or-self") == 0)
							{
								pToken->m_eTokenType = AXIS_ANCESTOR_OR_SELF_TOKEN;
								pToken->m_ui64Val = (FLMUINT64)ANCESTOR_OR_SELF_AXIS;
							}
							else if( f_uninativecmp( pToken->m_puzLocal,
								"attribute") == 0)
							{
								pToken->m_eTokenType = AXIS_ATTRIB_TOKEN;
								pToken->m_ui64Val = (FLMUINT64)ATTRIBUTE_AXIS;
							}
							else
							{
								rc = RC_SET( NE_XFLM_SYNTAX);
								goto Exit;
							}
							break;

						case FLM_UNICODE_c:
							if( f_uninativecmp( pToken->m_puzLocal,
								"child") == 0)
							{
								pToken->m_eTokenType = AXIS_CHILD_TOKEN;
								pToken->m_ui64Val = (FLMUINT64)CHILD_AXIS;
							}
							else
							{
								rc = RC_SET( NE_XFLM_SYNTAX);
								goto Exit;
							}
							break;

						case FLM_UNICODE_d:
							if( f_uninativecmp( pToken->m_puzLocal,
								"descendant") == 0)
							{
								pToken->m_eTokenType = AXIS_DESCENDANT_TOKEN;
								pToken->m_ui64Val = (FLMUINT64)DESCENDANT_AXIS;
							}
							else if( f_uninativecmp( pToken->m_puzLocal,
								"descendant-or-self") == 0)
							{
								pToken->m_eTokenType = AXIS_DESCENDANT_OR_SELF_TOKEN;
								pToken->m_ui64Val = (FLMUINT64)DESCENDANT_OR_SELF_AXIS;
							}
							else
							{
								rc = RC_SET( NE_XFLM_SYNTAX);
								goto Exit;
							}
							break;

						case FLM_UNICODE_f:
							if( f_uninativecmp( pToken->m_puzLocal,
								"following") == 0)
							{
								pToken->m_eTokenType = AXIS_FOLLOWING_TOKEN;
								pToken->m_ui64Val = (FLMUINT64)FOLLOWING_AXIS;
							}
							else if( f_uninativecmp( pToken->m_puzLocal,
								"following-sibling") == 0)
							{
								pToken->m_eTokenType = AXIS_FOLLOWING_SIB_TOKEN;
								pToken->m_ui64Val = (FLMUINT64)FOLLOWING_SIBLING_AXIS;
							}
							else
							{
								rc = RC_SET( NE_XFLM_SYNTAX);
								goto Exit;
							}
							break;

						case FLM_UNICODE_m:
							if( f_uninativecmp( pToken->m_puzLocal,
								"meta") == 0)
							{
								pToken->m_eTokenType = AXIS_META_TOKEN;
								pToken->m_ui64Val = (FLMUINT64)META_AXIS;
							}
							else
							{
								rc = RC_SET( NE_XFLM_SYNTAX);
								goto Exit;
							}
							break;

						case FLM_UNICODE_n:
							if( f_uninativecmp( pToken->m_puzLocal,
								"namespace") == 0)
							{
								pToken->m_eTokenType = AXIS_NAMESPACE_TOKEN;
								pToken->m_ui64Val = (FLMUINT64)NAMESPACE_AXIS;
							}
							else
							{
								rc = RC_SET( NE_XFLM_SYNTAX);
								goto Exit;
							}
							break;

						case FLM_UNICODE_p:
							if( f_uninativecmp( pToken->m_puzLocal,
								"parent") == 0)
							{
								pToken->m_eTokenType = AXIS_PARENT_TOKEN;
								pToken->m_ui64Val = (FLMUINT64)PARENT_AXIS;
							}
							else if( f_uninativecmp( pToken->m_puzLocal,
								"preceding") == 0)
							{
								pToken->m_eTokenType = AXIS_PRECEDING_TOKEN;
								pToken->m_ui64Val = (FLMUINT64)PRECEDING_AXIS;
							}
							else if( f_uninativecmp( pToken->m_puzLocal,
								"preceding-sibling") == 0)
							{
								pToken->m_eTokenType = AXIS_PRECEDING_SIB_TOKEN;
								pToken->m_ui64Val = (FLMUINT64)PRECEDING_SIBLING_AXIS;
							}
							else
							{
								rc = RC_SET( NE_XFLM_SYNTAX);
								goto Exit;
							}
							break;

						case FLM_UNICODE_s:
							if( f_uninativecmp( pToken->m_puzLocal,
								"self") == 0)
							{
								pToken->m_eTokenType = AXIS_SELF_TOKEN;
								pToken->m_ui64Val = (FLMUINT64)SELF_AXIS;
							}
							else
							{
								rc = RC_SET( NE_XFLM_SYNTAX);
								goto Exit;
							}
							break;

						default:
							rc = RC_SET( NE_XFLM_SYNTAX);
							goto Exit;
					}
					goto Exit;
				}
				else if( uChar == FLM_UNICODE_ASTERISK)
				{
					if( RC_BAD( rc = getChar( &uChar)))
					{
						goto Exit;
					}

					pToken->m_eTokenType = NAME_TEST_NCWILD_TOKEN;
					goto Exit;
				}
				else
				{
					if( RC_BAD( rc = ungetChar( FLM_UNICODE_COLON)))
					{
						goto Exit;
					}
				}
			}

			// Operator?

			if( m_eLastTokenType != UNKNOWN_TOKEN &&
				m_eLastTokenType != LBRACE_TOKEN &&
				m_eLastTokenType != RBRACE_TOKEN &&
				m_eLastTokenType != DOUBLE_COLON_TOKEN &&
				m_eLastTokenType != OP_LPAREN_TOKEN &&
				m_eLastTokenType != OP_LBRACKET_TOKEN &&
				m_eLastTokenType != COMMA_TOKEN &&
				!isOperator( m_eLastTokenType) &&
				!isAxisSpecifier( m_eLastTokenType))
			{
				if( f_uninativecmp( pToken->m_puzLocal,
					"and") == 0)
				{
					pToken->m_eTokenType = OP_AND_TOKEN;
				}
				else if( f_uninativecmp( pToken->m_puzLocal,
					"or") == 0)
				{
					pToken->m_eTokenType = OP_OR_TOKEN;
				}
				else if( f_uninativecmp( pToken->m_puzLocal,
					"bitor") == 0)
				{
					pToken->m_eTokenType = OP_BITOR_TOKEN;
				}
				else if( f_uninativecmp( pToken->m_puzLocal,
					"bitand") == 0)
				{
					pToken->m_eTokenType = OP_BITAND_TOKEN;
				}
				else if( f_uninativecmp( pToken->m_puzLocal,
					"bitxor") == 0)
				{
					pToken->m_eTokenType = OP_BITXOR_TOKEN;
				}
				else if( f_uninativecmp( pToken->m_puzLocal,
					"mod") == 0)
				{
					pToken->m_eTokenType = OP_MOD_TOKEN;
				}
				else if( f_uninativecmp( pToken->m_puzLocal,
					"div") == 0)
				{
					pToken->m_eTokenType = OP_DIV_TOKEN;
				}
				else
				{
					rc = RC_SET( NE_XFLM_SYNTAX);
				}

				goto Exit;
			}

			// None of the above conditions matched, so we have a QName

			pToken->m_eTokenType = NAME_TEST_QNAME_TOKEN;
			break;
		}
	}

	// Get the operator flags (if any)

	if( tokenCanHaveFlags( pToken->m_eTokenType))
	{
		FLMUNICODE		uzTmpBuf[ 64];
		FLMUNICODE		uTmpChar;
		FLMUINT			uiOffset;

		if( RC_BAD( rc = skipWhitespace()))
		{
			goto Exit;
		}

		if( RC_BAD( rc = peekChar( &uTmpChar)))
		{
			goto Exit;
		}

		if( uTmpChar == FLM_UNICODE_LBRACE)
		{
			if( RC_BAD( rc = getChar( &uTmpChar)))
			{
				goto Exit;
			}

			for( ;;)
			{
				if( RC_BAD( rc = skipWhitespace()))
				{
					goto Exit;
				}

				uiOffset = 0;

				for( ;;)
				{
					if( RC_BAD( rc = getChar( &uTmpChar)))
					{
						goto Exit;
					}

					if( !gv_XFlmSysData.pXml->isLetter( uTmpChar) &&
						uTmpChar != FLM_UNICODE_UNDERSCORE)
					{
						uzTmpBuf[ uiOffset] = 0;
						if( f_uninativecmp( uzTmpBuf, "ci") == 0 ||
							f_uninativecmp( uzTmpBuf, "caseinsensitive") == 0)
						{
							pToken->m_uiTokenFlags |= XFLM_COMP_CASE_INSENSITIVE;
						}
						else if( f_uninativecmp( uzTmpBuf, "cs") == 0 ||
							f_uninativecmp( uzTmpBuf, "compressspace") == 0)
						{
							pToken->m_uiTokenFlags |= XFLM_COMP_COMPRESS_WHITESPACE;
						}
						else if( f_uninativecmp( uzTmpBuf, "ignls") == 0 ||
							f_uninativecmp( uzTmpBuf, "ignoreleadingspace") == 0)
						{
							pToken->m_uiTokenFlags |= XFLM_COMP_IGNORE_LEADING_SPACE;
						}
						else if( f_uninativecmp( uzTmpBuf, "ignts") == 0 ||
							f_uninativecmp( uzTmpBuf, "ignoretrailingspace") == 0)
						{
							pToken->m_uiTokenFlags |= XFLM_COMP_IGNORE_TRAILING_SPACE;
						}
						else if( f_uninativecmp( uzTmpBuf, "wstosp") == 0 ||
							f_uninativecmp( uzTmpBuf, "whitespaceasspace") == 0)
						{
							pToken->m_uiTokenFlags |= XFLM_COMP_WHITESPACE_AS_SPACE;
						}
						else if( f_uninativecmp( uzTmpBuf, "ns") == 0 ||
							f_uninativecmp( uzTmpBuf, "nospace") == 0)
						{
							pToken->m_uiTokenFlags |= XFLM_COMP_NO_WHITESPACE;
						}
						else if( f_uninativecmp( uzTmpBuf, "nu") == 0 ||
							f_uninativecmp( uzTmpBuf, "nounderscores") == 0)
						{
							pToken->m_uiTokenFlags |= XFLM_COMP_NO_UNDERSCORES;
						}
						else if( f_uninativecmp( uzTmpBuf, "nd") == 0 ||
							f_uninativecmp( uzTmpBuf, "nodashes") == 0)
						{
							pToken->m_uiTokenFlags |= XFLM_COMP_NO_DASHES;
						}
						else
						{
							rc = RC_SET( NE_XFLM_SYNTAX);
							goto Exit;
						}

						break;
					}

					uzTmpBuf[ uiOffset++] = uTmpChar;
				}

				if( !uTmpChar)
				{
					rc = RC_SET( NE_XFLM_SYNTAX);
					goto Exit;
				}
				else if( uTmpChar == FLM_UNICODE_RBRACE)
				{
					break;
				}
				else if( uTmpChar != FLM_UNICODE_COMMA)
				{
					rc = RC_SET( NE_XFLM_SYNTAX);
					goto Exit;
				}
			}
		}
	}

Exit:

	m_eLastTokenType = pToken->m_eTokenType;

	// Are we expecting a () sequence after the token that we need to
	// just consume?

	if (RC_OK( rc) && bConsumeParens)
	{
		if( RC_OK( rc = skipWhitespace()))
		{
			if( RC_OK( rc = getChar( &uChar)))
			{
				if (uChar != FLM_UNICODE_LPAREN)
				{
					rc = RC_SET( NE_XFLM_SYNTAX);
				}
				else if (RC_OK( rc = skipWhitespace()))
				{
					if (RC_OK( rc = getChar( &uChar)))
					{
						if (uChar != FLM_UNICODE_RPAREN)
						{
							rc = RC_SET( NE_XFLM_SYNTAX);
						}
					}
				}
			}
		}
	}
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_XPathTokenizer::getName(
	F_XPathToken *			pToken)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiOffset;
	FLMBOOL			bFoundColon = FALSE;
	FLMUINT			uiMaxChars;
	FLMUNICODE		uChar;
	FLMUNICODE		uTmpChar;

	uiOffset = 0;
	if( (uiMaxChars = pToken->m_uiValBufSize / sizeof( FLMUNICODE)) < 32)
	{
		if( RC_BAD( rc = pToken->resizeBuffer( 32 * sizeof( FLMUNICODE))))
		{
			goto Exit;
		}
		uiMaxChars = pToken->m_uiValBufSize / sizeof( FLMUNICODE);
	}

	pToken->m_puzLocal = (FLMUNICODE *)pToken->m_pValBuf;
	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( !gv_XFlmSysData.pXml->isLetter( uChar) &&
		uChar != FLM_UNICODE_UNDERSCORE)
	{
		rc = RC_SET( NE_XFLM_SYNTAX);
		goto Exit;
	}

	((FLMUNICODE *)pToken->m_pValBuf)[ uiOffset++] = uChar;
	for( ;;)
	{
		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		if( uiOffset == uiMaxChars)
		{
			if( RC_BAD( rc = pToken->resizeBuffer(
				pToken->m_uiValBufSize * sizeof( FLMUNICODE) * 2)))
			{
				goto Exit;
			}
			uiMaxChars *= 2;
		}

		if( uChar == FLM_UNICODE_COLON)
		{
			if( bFoundColon)
			{
				rc = RC_SET( NE_XFLM_SYNTAX);
				goto Exit;
			}

			if( RC_BAD( rc = peekChar( &uTmpChar)))
			{
				goto Exit;
			}

			if( !gv_XFlmSysData.pXml->isNCNameChar( uTmpChar))
			{
				break;
			}

			uChar = 0;
			pToken->m_puzPrefix = (FLMUNICODE *)pToken->m_pValBuf;
			pToken->m_puzLocal =
				&(((FLMUNICODE *)pToken->m_pValBuf)[ uiOffset + 1]);
			bFoundColon = TRUE;
		}
		else if( !gv_XFlmSysData.pXml->isNCNameChar( uChar))
		{
			break;
		}

		((FLMUNICODE *)pToken->m_pValBuf)[ uiOffset++] = uChar;
	}

	((FLMUNICODE *)pToken->m_pValBuf)[ uiOffset] = 0;

	if( bFoundColon &&
		(*pToken->m_puzPrefix == 0 || *pToken->m_puzLocal == 0))
	{
		rc = RC_SET( NE_XFLM_SYNTAX);
		goto Exit;
	}

	if( RC_BAD( rc = ungetChar( uChar)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_XPathTokenizer::getNumber(
	F_XPathToken *			pToken)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT64	ui64Num = 0;
	FLMUNICODE	uChar;

	for( ;;)
	{
		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		if( uChar < FLM_UNICODE_0 || uChar > FLM_UNICODE_9)
		{
			if( RC_BAD( rc = ungetChar( uChar)))
			{
				goto Exit;
			}

			break;
		}

#if defined ( FLM_LINUX) || defined ( FLM_NLM) || defined( FLM_OSX)
		if( ui64Num > ((0xFFFFFFFFFFFFFFFFULL / 10) + (uChar - FLM_UNICODE_0)))
#else
		if( ui64Num > ((0xFFFFFFFFFFFFFFFF / 10) + (uChar - FLM_UNICODE_0)))
#endif
		{
			rc = RC_SET( NE_XFLM_CONV_NUM_OVERFLOW);
			goto Exit;
		}

		ui64Num = (FLMUINT64)((ui64Num * 10) + (uChar - FLM_UNICODE_0));
	}

	pToken->m_ui64Val = ui64Num;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_XPathTokenizer::getBinary(
	F_XPathToken *			pToken)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUNICODE	uChar;
	FLMBOOL		bDoubleQuote = FALSE;
	FLMUINT		uiOffset = 0;
	FLMUINT		uiMaxBytes;
	FLMBYTE *	pucBuf;
	FLMBOOL		bHaveHighNibble = FALSE;
	FLMBYTE		ucHighNibble = 0;
	FLMBYTE		ucNibble;

	if( (uiMaxBytes = pToken->m_uiValBufSize) < 64)
	{
		if( RC_BAD( rc = pToken->resizeBuffer( 64)))
		{
			goto Exit;
		}
		uiMaxBytes = pToken->m_uiValBufSize;
	}

	pucBuf = (FLMBYTE *)pToken->m_pValBuf;

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar == FLM_UNICODE_QUOTE)
	{
		bDoubleQuote = TRUE;
	}
	else if( uChar != FLM_UNICODE_APOS)
	{
		rc = RC_SET( NE_XFLM_SYNTAX);
		goto Exit;
	}

	for( ;;)
	{
		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		if ( uChar == 0)
		{
			rc = RC_SET( NE_XFLM_UNEXPECTED_END_OF_INPUT);
			goto Exit;
		}

		if( bDoubleQuote && uChar == FLM_UNICODE_QUOTE)
		{
			break;
		}
		else if( !bDoubleQuote && uChar == FLM_UNICODE_APOS)
		{
			break;
		}

		if (uChar >= FLM_UNICODE_0 && uChar <= FLM_UNICODE_9)
		{
			ucNibble = (FLMBYTE)(uChar - FLM_UNICODE_0);
		}
		else if (uChar >= FLM_UNICODE_A && uChar <= FLM_UNICODE_F)
		{
			ucNibble = (FLMBYTE)(uChar - FLM_UNICODE_A + 10);
		}
		else if (uChar >= FLM_UNICODE_a && uChar <= FLM_UNICODE_f)
		{
			ucNibble = (FLMBYTE)(uChar - FLM_UNICODE_a + 10);
		}
		else if (gv_XFlmSysData.pXml->isWhitespace( uChar))
		{
			// Just skip whitespace, anything else we will treat as an
			// error.
			continue;
		}
		else
		{
			rc = RC_SET( NE_XFLM_SYNTAX);
			goto Exit;
		}
		if (!bHaveHighNibble)
		{
			ucHighNibble = (FLMBYTE)(ucNibble << 4);
			bHaveHighNibble = TRUE;
		}
		else
		{
			pucBuf [uiOffset++] = ucHighNibble | ucNibble;
			if( uiOffset == uiMaxBytes)
			{
				if( RC_BAD( rc = pToken->resizeBuffer(
					pToken->m_uiValBufSize * 2)))
				{
					goto Exit;
				}
				uiMaxBytes *= 2;
			}
			bHaveHighNibble = FALSE;
		}
	}

	if (bHaveHighNibble)
	{
		pucBuf [uiOffset++] = ucHighNibble;
	}
	pToken->m_uiValBufLen = uiOffset;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_XPathTokenizer::getLiteral(
	F_XPathToken *			pToken)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUNICODE	uChar;
	FLMBOOL		bDoubleQuote = FALSE;
	FLMUINT		uiOffset;
	FLMUINT		uiMaxChars;

	uiOffset = 0;
	if( (uiMaxChars = pToken->m_uiValBufSize / sizeof( FLMUNICODE)) < 32)
	{
		if( RC_BAD( rc = pToken->resizeBuffer( 32 * sizeof( FLMUNICODE))))
		{
			goto Exit;
		}
		uiMaxChars = pToken->m_uiValBufSize / sizeof( FLMUNICODE);
	}

	pToken->m_puzLocal = (FLMUNICODE *)pToken->m_pValBuf;

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar == FLM_UNICODE_QUOTE)
	{
		bDoubleQuote = TRUE;
	}
	else if( uChar != FLM_UNICODE_APOS)
	{
		rc = RC_SET( NE_XFLM_SYNTAX);
		goto Exit;
	}

	for( ;;)
	{
		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		if ( uChar == 0)
		{
			rc = RC_SET( NE_XFLM_UNEXPECTED_END_OF_INPUT);
			goto Exit;
		}

		if( bDoubleQuote && uChar == FLM_UNICODE_QUOTE)
		{
			break;
		}
		else if( !bDoubleQuote && uChar == FLM_UNICODE_APOS)
		{
			break;
		}

		pToken->m_puzLocal[ uiOffset++] = uChar;

		if( uiOffset == uiMaxChars)
		{
			if( RC_BAD( rc = pToken->resizeBuffer(
				pToken->m_uiValBufSize * sizeof( FLMUNICODE) * 2)))
			{
				goto Exit;
			}
			uiMaxChars *= 2;
		}
	}

	pToken->m_puzLocal[ uiOffset++] = 0;

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Class for handling query validator calls
***************************************************************************/
class XFLAIM_QueryValFunc : public IF_QueryValFunc
{
public:

	XFLAIM_QueryValFunc()
	{
	}
	
	virtual ~XFLAIM_QueryValFunc()
	{
	}
	
	RCODE XFLAPI getValue(
		IF_Db *					pDb,
		IF_DOMNode *			pContextNode,
		ValIterator				eValueToGet,
		eValTypes *				peValType,
		FLMBOOL *				pbLastValue,
		void *					pvVal,
		F_DynaBuf *				pDynaBuf = NULL);
			
	RCODE XFLAPI cloneSelf(
		IF_QueryValFunc **	ppNewObj);
};

/****************************************************************************
Desc:	Get the next value for a query function.  Since this is really just
		code to test the callback, it always returns a value whose type
		is boolean and whose value is true.
****************************************************************************/
RCODE XFLAPI XFLAIM_QueryValFunc::getValue(
	IF_Db *,			// pDb,
	IF_DOMNode *,	// pContextNode,
	ValIterator		eValueToGet,
	eValTypes *		peValType,
	FLMBOOL *		pbLastValue,
	void *			pvVal,
	F_DynaBuf *		pDynaBuf)
{
	RCODE	rc = NE_XFLM_OK;
	
	if (pDynaBuf)
	{
		(void)pDynaBuf->truncateData( 0);
	}
	
	if (eValueToGet != GET_FIRST_VAL && eValueToGet != GET_LAST_VAL)
	{
		rc = (eValueToGet == GET_NEXT_VAL)
			  ? NE_XFLM_EOF_HIT
			  : NE_XFLM_BOF_HIT;
		goto Exit;
	}
	
	*pbLastValue = TRUE;
	*peValType = XFLM_BOOL_VAL;
	*((XFlmBoolType *)pvVal) = XFLM_TRUE;
	
Exit:

	return( rc);
}
			
/****************************************************************************
Desc:	Copy self to create a new object.
****************************************************************************/
RCODE XFLAPI XFLAIM_QueryValFunc::cloneSelf(
	IF_QueryValFunc **	ppNewObj)
{
	
	// No need to copy - simply reference, because it has no private
	// state data that needs to have its own copy.
	
	*ppNewObj = (IF_QueryValFunc *)this;
	(*ppNewObj)->AddRef();
	return( NE_XFLM_OK);
}

/***************************************************************************
Desc:	Add a callback function predicate.
***************************************************************************/
FSTATIC RCODE addCallbackFunc(
	IF_Query *	pQuery)
{
	RCODE							rc = NE_XFLM_OK;
	XFLAIM_QueryValFunc *	pFuncObj = NULL;
			
	// Create a callback function object

	if ((pFuncObj = f_new XFLAIM_QueryValFunc) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	
	if (RC_BAD( rc = pQuery->addFunction( pFuncObj, TRUE)))
	{
		goto Exit;
	}

Exit:

	if (pFuncObj)
	{
		pFuncObj->Release();
	}
	return( rc);
}

/****************************************************************************
Desc:		Parse an XPATH query
****************************************************************************/
RCODE F_XPath::parseQuery(
	F_Db *			pDb,
	IF_IStream *	pIStream,
	IF_Query *		pQuery)
{
	RCODE					rc = NE_XFLM_OK;
	F_NameTable *		pNameTable = NULL;
	F_XMLNamespace *	pNamespace = NULL;
	eXPathAxisTypes	eCurrentAxis = CHILD_AXIS;
	eDomNodeType		eNodeType = ELEMENT_NODE;
	FLMUINT				uiDictNum;

	if( RC_BAD( rc = pDb->getNameTable( &pNameTable)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_tokenizer.setup( pIStream)))
	{
		goto Exit;
	}

	// Process any namespace declarations

	popNamespaces( getNamespaceCount());

	if( RC_BAD( rc = pushNamespace( NULL, NULL)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getNextToken()))
	{
		goto Exit;
	}

	if( m_curToken.getType() == LBRACE_TOKEN)
	{
		for( ;;)
		{
			if( (pNamespace = f_new F_XMLNamespace) == NULL)
			{
				rc = RC_SET( NE_XFLM_MEM);
				goto Exit;
			}

			if( RC_BAD( rc = getNextToken()))
			{
				goto Exit;
			}

			if( m_curToken.getType() != NAME_TEST_QNAME_TOKEN)
			{
				rc = RC_SET( NE_XFLM_SYNTAX);
				goto Exit;
			}

			if( m_curToken.getPrefixPtr())
			{
				if( f_uninativecmp( m_curToken.getPrefixPtr(),
					"xmlns") != 0)
				{
					rc = RC_SET( NE_XFLM_SYNTAX);
					goto Exit;
				}

				if( RC_BAD( rc = pNamespace->setPrefix(
					m_curToken.getLocalPtr())))
				{
					goto Exit;
				}
			}
			else
			{
				if( f_uninativecmp( m_curToken.getLocalPtr(),
					"xmlns") != 0)
				{
					rc = RC_SET( NE_XFLM_SYNTAX);
					goto Exit;
				}
			}

			if( RC_BAD( rc = getNextToken()))
			{
				goto Exit;
			}

			if( m_curToken.getType() != OP_EQ_TOKEN)
			{
				rc = RC_SET( NE_XFLM_SYNTAX);
				goto Exit;
			}

			if( RC_BAD( rc = getNextToken()))
			{
				goto Exit;
			}

			if( m_curToken.getType() != LITERAL_TOKEN)
			{
				rc = RC_SET( NE_XFLM_SYNTAX);
				goto Exit;
			}

			if( RC_BAD( rc = pNamespace->setURI(
				m_curToken.getLocalPtr())))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pushNamespace( pNamespace)))
			{
				goto Exit;
			}

			pNamespace->Release();
			pNamespace = NULL;

			if( RC_BAD( rc = getNextToken()))
			{
				goto Exit;
			}

			if( m_curToken.getType() == RBRACE_TOKEN)
			{
				if( RC_BAD( rc = getNextToken()))
				{
					goto Exit;
				}
				break;
			}
			else if( m_curToken.getType() != COMMA_TOKEN)
			{
				rc = RC_SET( NE_XFLM_SYNTAX);
				goto Exit;
			}
		}
	}

	for( ;;)
	{

		if( m_curToken.m_eTokenType == END_TOKEN)
		{
			break;
		}

		switch( m_curToken.m_eTokenType)
		{
			case OP_AND_TOKEN:
			case OP_OR_TOKEN:
			case OP_EQ_TOKEN:
			case OP_APPROX_EQ_TOKEN:
			case OP_NOT_TOKEN:
			case OP_NE_TOKEN:
			case OP_LT_TOKEN:
			case OP_LE_TOKEN:
			case OP_GT_TOKEN:
			case OP_GE_TOKEN:
			case OP_BITAND_TOKEN:
			case OP_BITOR_TOKEN:
			case OP_BITXOR_TOKEN:
			case OP_MULT_TOKEN:
			case OP_DIV_TOKEN:
			case OP_MOD_TOKEN:
			case OP_PLUS_TOKEN:
			case OP_MINUS_TOKEN:
			case OP_LPAREN_TOKEN:
			case OP_RPAREN_TOKEN:
			case OP_COMMA_TOKEN:
			case OP_LBRACKET_TOKEN:
			case OP_RBRACKET_TOKEN:
				if( RC_BAD( rc = pQuery->addOperator( 
					(eQueryOperators)m_curToken.m_eTokenType,
					m_curToken.m_uiTokenFlags)))
				{
					goto Exit;
				}
				break;

			case LITERAL_TOKEN:
				if( RC_BAD( rc = pQuery->addUnicodeValue(
					(FLMUNICODE *)m_curToken.m_pValBuf)))
				{
					goto Exit;
				}
				break;

			case BINARY_TOKEN:
				if( RC_BAD( rc = pQuery->addBinaryValue(
					m_curToken.m_pValBuf,
					m_curToken.m_uiValBufLen)))
				{
					goto Exit;
				}
				break;

			case NUMBER_TOKEN:
				if( RC_BAD( rc = pQuery->addUINT64Value( m_curToken.m_ui64Val)))
				{
					goto Exit;
				}
				break;

			case FUNC_TRUE_TOKEN:
			case FUNC_FALSE_TOKEN:
			case FUNC_UNKNOWN_TOKEN:
				{
					FLMBOOL	bVal = (m_curToken.m_eTokenType == FUNC_TRUE_TOKEN)
						? TRUE
						: FALSE;

					FLMBOOL	bUnknown = (m_curToken.m_eTokenType == FUNC_UNKNOWN_TOKEN)
						? TRUE
						: FALSE;

					if ( RC_BAD( rc = pQuery->addBoolean( bVal, bUnknown)))
					{
						goto Exit;
					}

					// consume the opening and closing paren

					if ( RC_BAD( rc = getNextToken()))
					{
						goto Exit;
					}

					if ( m_curToken.m_eTokenType != OP_LPAREN_TOKEN)
					{
						rc = RC_SET( NE_XFLM_Q_EXPECTING_LPAREN);
						goto Exit;
					}

					if ( RC_BAD( rc = getNextToken()))
					{
						goto Exit;
					}

					if ( m_curToken.m_eTokenType != OP_RPAREN_TOKEN)
					{
						rc = RC_SET( NE_XFLM_Q_EXPECTING_RPAREN);
						goto Exit;
					}
				}
				break;

			case FUNC_LAST_TOKEN:
			case FUNC_POSITION_TOKEN:
			case FUNC_COUNT_TOKEN:
			case FUNC_ID_TOKEN:
			case FUNC_LOCAL_NAME_TOKEN:
			case FUNC_NAMESPACE_URI_TOKEN:
			case FUNC_NAME_TOKEN:
			case FUNC_STRING_TOKEN:
			case FUNC_CONCAT_TOKEN:
			case FUNC_STARTS_WITH_TOKEN:
			case FUNC_CONTAINS_TOKEN:
			case FUNC_SUBSTR_BEFORE_TOKEN:
			case FUNC_SUBSTR_AFTER_TOKEN:
			case FUNC_SUBSTR_TOKEN:
			case FUNC_STR_LEN_TOKEN:
			case FUNC_NORM_SPACE_TOKEN:
			case FUNC_TRANSLATE_TOKEN:
			case FUNC_NOT_TOKEN:
			case FUNC_LANG_TOKEN:
			case FUNC_NUMBER_TOKEN:
			case FUNC_SUM_TOKEN:
			case FUNC_FLOOR_TOKEN:
			case FUNC_CEILING_TOKEN:
			case FUNC_ROUND_TOKEN:
				if( RC_BAD( rc = pQuery->addFunction( XFLM_FUNC_xxx)))
				{
					goto Exit;
				}
				break;

			case FUNC_CB_TOKEN:
				if (RC_BAD( rc = addCallbackFunc( pQuery)))
				{
					goto Exit;
				}
				break;

			case AXIS_ANCESTOR_TOKEN:
			case AXIS_ANCESTOR_OR_SELF_TOKEN:
			case AXIS_CHILD_TOKEN:
			case AXIS_DESCENDANT_TOKEN:
			case AXIS_DESCENDANT_OR_SELF_TOKEN:
			case AXIS_FOLLOWING_TOKEN:
			case AXIS_FOLLOWING_SIB_TOKEN:
			case AXIS_PARENT_TOKEN:
			case AXIS_PRECEDING_TOKEN:
			case AXIS_PRECEDING_SIB_TOKEN:
			case AXIS_SELF_TOKEN:
			case AXIS_ATTRIB_TOKEN:
			case AXIS_NAMESPACE_TOKEN:
			case AXIS_ATSIGN_TOKEN:
			case AXIS_META_TOKEN:
			{
				eCurrentAxis = (eXPathAxisTypes)m_curToken.m_ui64Val;
				eNodeType = ELEMENT_NODE;

				if (m_curToken.m_eTokenType == AXIS_ATSIGN_TOKEN)
				{
					eNodeType = ATTRIBUTE_NODE;
				}
				else
				{
					if (m_curToken.m_eTokenType == AXIS_META_TOKEN)
					{
						eNodeType = ANY_NODE_TYPE;
					}
					else if (m_curToken.m_eTokenType == AXIS_ATTRIB_TOKEN ||
								m_curToken.m_eTokenType == AXIS_NAMESPACE_TOKEN)
					{
						eNodeType = ATTRIBUTE_NODE;
					}
					else if (m_curToken.m_eTokenType == AXIS_SELF_TOKEN)
					{
						eNodeType = ANY_NODE_TYPE;
					}
					if( RC_BAD( rc = getNextToken()))
					{
						goto Exit;
					}

					if( m_curToken.m_eTokenType != DOUBLE_COLON_TOKEN)
					{
						rc = RC_SET( NE_XFLM_SYNTAX);
						goto Exit;
					}
				}

				if( RC_BAD( rc = getNextToken()))
				{
					goto Exit;
				}

				switch( m_curToken.m_eTokenType)
				{
					case NAME_TEST_WILD_TOKEN:
					{
						if( RC_BAD( rc = pQuery->addXPathComponent( eCurrentAxis,
							eNodeType, 0)))
						{
							goto Exit;
						}

						eCurrentAxis = CHILD_AXIS;
						eNodeType = ELEMENT_NODE;
						break;
					}

					case NAME_TEST_QNAME_TOKEN:
					{
						if (eCurrentAxis == META_AXIS)
						{
							if (f_uninativecmp( m_curToken.getLocalPtr(),
								 "nodeid") == 0)
							{
								uiDictNum = XFLM_META_NODE_ID;
							}
							else if (f_uninativecmp( m_curToken.getLocalPtr(),
										"documentid") == 0)
							{
								uiDictNum = XFLM_META_DOCUMENT_ID;
							}
							else if (f_uninativecmp( m_curToken.getLocalPtr(),
										"parentid") == 0)
							{
								uiDictNum = XFLM_META_PARENT_ID;
							}
							else if (f_uninativecmp( m_curToken.getLocalPtr(),
										"firstchildid") == 0)
							{
								uiDictNum = XFLM_META_FIRST_CHILD_ID;
							}
							else if (f_uninativecmp( m_curToken.getLocalPtr(),
										"lastchildid") == 0)
							{
								uiDictNum = XFLM_META_LAST_CHILD_ID;
							}
							else if (f_uninativecmp( m_curToken.getLocalPtr(),
										"nextsiblingid") == 0)
							{
								uiDictNum = XFLM_META_NEXT_SIBLING_ID;
							}
							else if (f_uninativecmp( m_curToken.getLocalPtr(),
										"prevsiblingid") == 0)
							{
								uiDictNum = XFLM_META_PREV_SIBLING_ID;
							}
							else if (f_uninativecmp( m_curToken.getLocalPtr(),
										"value") == 0)
							{
								uiDictNum = XFLM_META_VALUE;
							}
							else
							{
								rc = RC_SET( NE_XFLM_Q_INVALID_META_DATA_TYPE);
								goto Exit;
							}
						}
						else
						{
							if( m_curToken.getPrefixPtr())
							{
								if( RC_BAD( rc = findNamespace(
									m_curToken.getPrefixPtr(), &pNamespace)))
								{
									rc = RC_SET( NE_XFLM_SYNTAX);
									goto Exit;
								}
							}

							if( RC_BAD( rc = pNameTable->getFromTagTypeAndName( 
								pDb, eNodeType == ELEMENT_NODE
									? ELM_ELEMENT_TAG
									: ELM_ATTRIBUTE_TAG, 
								m_curToken.getLocalPtr(),
								NULL, pNamespace ? TRUE : FALSE, 
								pNamespace ? pNamespace->getURIPtr() : NULL,
								&uiDictNum, NULL)))
							{
								if( rc != NE_XFLM_NOT_FOUND && rc != NE_XFLM_MULTIPLE_MATCHES)
								{
									goto Exit;
								}
			
								if( !m_curToken.getPrefixPtr())
								{
									if( RC_BAD( rc = pNameTable->getFromTagTypeAndName( 
										pDb, eNodeType == ELEMENT_NODE
											? ELM_ELEMENT_TAG
											: ELM_ATTRIBUTE_TAG, 
										m_curToken.getLocalPtr(),
										NULL, TRUE, NULL, &uiDictNum, NULL)))
									{
										rc = RC_SET( NE_XFLM_SYNTAX);
										goto Exit;
									}
								}
								else
								{
									rc = RC_SET( NE_XFLM_SYNTAX);
									goto Exit;
								}
							}
							
							if( pNamespace)
							{
								pNamespace->Release();
								pNamespace = NULL;
							}
						}
		
						if( RC_BAD( rc = pQuery->addXPathComponent( eCurrentAxis,
							eNodeType, uiDictNum)))
						{
							goto Exit;
						}

						eCurrentAxis = CHILD_AXIS;
						eNodeType = ELEMENT_NODE;
						break;
					}
					case NODE_TYPE_NODE_TOKEN:
					{
						if( RC_BAD( rc = pQuery->addXPathComponent( eCurrentAxis,
							ANY_NODE_TYPE, 0)))
						{
							goto Exit;
						}

						eCurrentAxis = CHILD_AXIS;
						eNodeType = ELEMENT_NODE;
						break;
					}
					case NODE_TYPE_TEXT_TOKEN:
					{
						if( RC_BAD( rc = pQuery->addXPathComponent( eCurrentAxis,
							DATA_NODE, 0)))
						{
							goto Exit;
						}

						eCurrentAxis = CHILD_AXIS;
						eNodeType = ELEMENT_NODE;
						break;
					}
					case NAME_TEST_NCWILD_TOKEN:
					{
						rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
						goto Exit;
					}

					case NODE_TYPE_COMMENT_TOKEN:
					{
						if( RC_BAD( rc = pQuery->addXPathComponent( eCurrentAxis,
							COMMENT_NODE, 0)))
						{
							goto Exit;
						}

						eCurrentAxis = CHILD_AXIS;
						eNodeType = ELEMENT_NODE;
						break;
					}

					case NODE_TYPE_PI_TOKEN:
					{
						rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
						goto Exit;
					}

					default:
					{
						rc = RC_SET( NE_XFLM_SYNTAX);
						goto Exit;
					}
				}
				break;
			}

			case NAME_TEST_QNAME_TOKEN:
				
				eNodeType = ELEMENT_NODE;

				if( RC_BAD( rc = findNamespace(
					m_curToken.getPrefixPtr(), &pNamespace)))
				{
					rc = RC_SET( NE_XFLM_SYNTAX);
					goto Exit;
				}

				if( RC_BAD( rc = pNameTable->getFromTagTypeAndName( 
					pDb, ELM_ELEMENT_TAG, m_curToken.getLocalPtr(),
					NULL, pNamespace ? TRUE : FALSE, pNamespace ? pNamespace->getURIPtr() : NULL,
					&uiDictNum, NULL)))
				{
					if( rc != NE_XFLM_NOT_FOUND)
					{
						goto Exit;
					}

					if( !m_curToken.getPrefixPtr())
					{
						if( RC_BAD( rc = pNameTable->getFromTagTypeAndName( 
							pDb, ELM_ELEMENT_TAG, m_curToken.getLocalPtr(),
							NULL, FALSE, NULL, &uiDictNum, NULL)))
						{
							rc = RC_SET( NE_XFLM_SYNTAX);
							goto Exit;
						}
					}
					else
					{
						rc = RC_SET( NE_XFLM_SYNTAX);
						goto Exit;
					}
				}

				if( pNamespace)
				{
					pNamespace->Release();
					pNamespace = NULL;
				}

				if( RC_BAD( rc = pQuery->addXPathComponent( eCurrentAxis,
					eNodeType, uiDictNum)))
				{
					goto Exit;
				}

				eCurrentAxis = CHILD_AXIS;
				eNodeType = ELEMENT_NODE;
				break;

			case PERIOD_TOKEN:
				if( RC_BAD( rc = pQuery->addXPathComponent( SELF_AXIS,
					ANY_NODE_TYPE, 0)))
				{
					goto Exit;
				}

				eCurrentAxis = CHILD_AXIS;
				eNodeType = ELEMENT_NODE;
				break;

			case DOUBLE_PERIOD_TOKEN:
				if ( RC_BAD( rc = pQuery->addXPathComponent( PARENT_AXIS,
					ANY_NODE_TYPE, 0)))
				{
					goto Exit;
				}

				eCurrentAxis = PARENT_AXIS;
				eNodeType = ELEMENT_NODE;
				break;

			case OP_FSLASH_TOKEN:
				eCurrentAxis = CHILD_AXIS;
				break;

			case OP_DOUBLE_FSLASH_TOKEN:
				eCurrentAxis = DESCENDANT_AXIS;
				break;

			case NODE_TYPE_NODE_TOKEN:
			{
				if( RC_BAD( rc = pQuery->addXPathComponent( eCurrentAxis,
					ANY_NODE_TYPE, 0)))
				{
					goto Exit;
				}
				eCurrentAxis = CHILD_AXIS;
				break;
			}

			case NODE_TYPE_COMMENT_TOKEN:
			{
				if( RC_BAD( rc = pQuery->addXPathComponent( eCurrentAxis,
					COMMENT_NODE, 0)))
				{
					goto Exit;
				}

				eCurrentAxis = CHILD_AXIS;
				eNodeType = ELEMENT_NODE;

				break;
			}

			case NODE_TYPE_TEXT_TOKEN:
			{
				if( RC_BAD( rc = pQuery->addXPathComponent( eCurrentAxis,
					DATA_NODE, 0)))
				{
					goto Exit;
				}

				eCurrentAxis = CHILD_AXIS;
				eNodeType = ELEMENT_NODE;

				break;
			}

			case NAME_TEST_WILD_TOKEN:
			{
				if( RC_BAD( rc = pQuery->addXPathComponent( eCurrentAxis,
					ELEMENT_NODE, 0)))
				{
					goto Exit;
				}

				eCurrentAxis = CHILD_AXIS;
				eNodeType = ELEMENT_NODE;

				break;
			}

			case DOUBLE_COLON_TOKEN:
			case OP_UNION_TOKEN:
			case NODE_TYPE_PI_TOKEN:
			case NAME_TEST_NCWILD_TOKEN:
			case VAR_REF_TOKEN:
			case END_TOKEN:
			default:
				rc = RC_SET( NE_XFLM_SYNTAX);
				goto Exit;
		}

		if( RC_BAD( rc = getNextToken()))
		{
			goto Exit;
		}

	}

Exit:

	m_tokenizer.setup( NULL);

	if( pNamespace)
	{
		pNamespace->Release();
	}

	if( pNameTable)
	{
		pNameTable->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:		Parse an XPATH query
****************************************************************************/
RCODE F_XPath::parseQuery(
	F_Db *			pDb,
	char *			pszQuery,
	IF_Query *		pQuery)
{
	RCODE						rc = NE_XFLM_OK;
	IF_BufferIStream *	pBufferStream;
	
	if( RC_BAD( rc = FlmAllocBufferIStream( &pBufferStream)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pBufferStream->openStream( 
		(const char *)pszQuery, f_strlen( pszQuery))))
	{
		goto Exit;
	}

	if( RC_BAD( rc = parseQuery( pDb, pBufferStream, pQuery)))
	{
		goto Exit;
	}

Exit:

	if( pBufferStream)
	{
		pBufferStream->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_XPath::getNextToken( void)
{
	RCODE		rc = NE_XFLM_OK;

	if( RC_BAD( rc = m_tokenizer.getNextToken( &m_curToken)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}
