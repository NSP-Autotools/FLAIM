//-------------------------------------------------------------------------
// Desc:	Query parsing
// Tabs:	3
//
// Copyright (c) 1998-2000, 2002-2007 Novell, Inc. All Rights Reserved.
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

FSTATIC RCODE tokenAllocSpace(
	char **			ppszTokenStart,
	char **			ppszToken,
	FLMUINT *		puiTokenBufSize,
	FLMUINT			uiNewTokenBufSize);

FSTATIC RCODE tokenGet(
	const char **	ppszString,
	char **			ppszToken,
	FLMBOOL *		pbQuoted,
	QTYPES *			peType,
	FLMUINT *		puiTokenBufSize);

FSTATIC FLMBOOL tokenIsNum(
	const char *	pszToken,
	FLMUINT64		ui64Max,
	FLMUINT64 *		pui64Num);
	
FSTATIC FLMBOOL tokenIsOperator(
	const char *	pszToken,
	QTYPES *			peOperator);

FSTATIC FLMBOOL tokenIsField(
	const char *	pszToken,
	F_NameTable *	pNameTable,
	FLMUINT *		puiFieldPath,
	QTYPES *			peValueType,
	FLMBOOL			bAllowNamesOnly,
	FLMBOOL			bMustBeField);

FSTATIC RCODE allocValueSpace(
	void **			ppvVal,
	FLMUINT *		puiValBufSize,
	FLMUINT			uiNewSize);

FSTATIC RCODE tokenGetBinary(
	const char *	pszToken,
	void **			ppvVal,
	FLMUINT *		puiValLen,
	FLMUINT *		puiValBufSize);

FSTATIC RCODE tokenGetValue(
	QTYPES			eValueType,
	const char *	pszToken,
	FLMBOOL			bQuoted,
	void **			ppvVal,
	QTYPES *			peValType,
	FLMUINT *		puiValLen,
	FLMUINT *		puiValBufSize);

/****************************************************************************
Desc:	Allocate space for the token.
****************************************************************************/
FSTATIC RCODE tokenAllocSpace(
	char **		ppszTokenStart,
	char **		ppszToken,
	FLMUINT *	puiTokenBufSize,
	FLMUINT		uiNewTokenBufSize)
{
	RCODE			rc = FERR_OK;
	char *		pszTmp;

	if (RC_BAD( rc = f_alloc( uiNewTokenBufSize, &pszTmp)))
	{
		goto Exit;
	}

	// Copy the current token into the new buffer.

	f_memcpy( pszTmp, *ppszTokenStart, *puiTokenBufSize);

	// Free the old buffer, if it was allocated

	f_free( ppszTokenStart);
	*ppszTokenStart = pszTmp;
	*ppszToken = *ppszTokenStart + *puiTokenBufSize;
	*puiTokenBufSize = uiNewTokenBufSize;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Determine if a character is a delimiter character.
****************************************************************************/
FINLINE FLMBOOL tokenIsDelimiter(
	FLMBYTE	ucChar)
{
	if (ucChar <= ' ')
	{
		return( TRUE);
	}
	switch (ucChar)
	{
		case '!':
		case '+':
		case '-':
		case '*':
		case '%':
		case '/':
		case '(':
		case ')':
		case '=':
		case '>':
		case '<':
		case '&':
		case '|':
		case '\'':
		case '"':
			return( TRUE);
		default:
			return( FALSE);
	}
}

/****************************************************************************
Desc:	Get the next token from a query string.
****************************************************************************/
FSTATIC RCODE tokenGet(
	const char **		ppszString,
	char **				ppszToken,
	FLMBOOL *			pbQuoted,
	QTYPES *				peType,
	FLMUINT *			puiTokenBufSize)
{
	RCODE					rc = FERR_OK;
	const char *		pszStr = *ppszString;
	char *				pszToken = *ppszToken;
	char *				pszTokenStart = pszToken;
	FLMBYTE				ucQuote;
	FLMUINT				uiLen;

	// Skip leading white space

	*peType = NO_TYPE;
	*pbQuoted = FALSE;
	
	while (*pszStr && *pszStr <= ' ')
	{
		pszStr++;
	}
	
	switch (*pszStr)
	{
		case 0:
			break;
		case '!':
		case '+':
		case '-':
		case '*':
		case '%':
		case '/':
		case '(':
		case ')':
		{
			*pszToken++ = *pszStr++;
			break;
		}
		
		case '=':
		case '>':
		case '<':
		{
			*pszToken++ = *pszStr++;
			if (*pszStr == '=')
			{
				*pszToken++ = *pszStr++;
			}
			break;
		}
		
		case '&':
		{
			*pszToken++ = *pszStr++;
			if (*pszStr == '&')
			{
				*pszToken++ = *pszStr++;
			}
			break;
		}
		
		case '|':
		{
			*pszToken++ = *pszStr++;
			if (*pszStr == '|')
			{
				*pszToken++ = *pszStr++;
			}
			break;
		}
		
		case '\'':
		case '"':
		{
			*pbQuoted = TRUE;
			ucQuote = *pszStr++;
			uiLen = 0;
			
			while (*pszStr && *pszStr != ucQuote)
			{
				*pszToken++ = *pszStr++;
				uiLen++;

				// If we don't have room for a null terminating byte,
				// better reallocate.  Allocate enough so that we can
				// get the entire token.

				if (uiLen == *puiTokenBufSize)
				{
					const char *	pszTmp = pszStr;
					FLMUINT			uiExtraCharsNeeded = 1;	// For NULL

					// See how many more characters we need.

					while (*pszTmp && *pszTmp != ucQuote)
					{
						uiExtraCharsNeeded++;
						pszTmp++;
					}

					if (RC_BAD( rc = tokenAllocSpace( &pszTokenStart, &pszToken,
							puiTokenBufSize, *puiTokenBufSize + uiExtraCharsNeeded)))
					{
						goto Exit;
					}
				}
			}

			// If we ended on the quote, skip it.

			if (*pszStr == ucQuote)
			{
				pszStr++;
			}
			else
			{
				rc = RC_SET( FERR_CURSOR_SYNTAX);
				goto Exit;
			}
			break;
		}

		default:
		{
			uiLen = 0;
			while (!tokenIsDelimiter( *pszStr))
			{
				*pszToken++ = *pszStr++;
				uiLen++;

				// If we don't have room for a null terminating byte,
				// better reallocate.  Allocate enough so that we can
				// get the entire token.

				if (uiLen == *puiTokenBufSize)
				{
					const char *	pszTmp = pszStr;
					FLMUINT			uiExtraCharsNeeded = 1;	// For NULL

					// See how many more characters we need.

					while (!tokenIsDelimiter( *pszTmp))
					{
						uiExtraCharsNeeded++;
						pszTmp++;
					}

					if (RC_BAD( rc = tokenAllocSpace( &pszTokenStart, &pszToken,
												puiTokenBufSize,
												*puiTokenBufSize + uiExtraCharsNeeded)))
					{
						goto Exit;
					}
				}
			}
			
			break;
		}
	}

	// Always null terminate the token

	*pszToken = 0;

	// Test to see if we have any function keywords we want to parse out here.

	if( !(*pbQuoted))
	{
		if (f_stricmp( pszTokenStart, "field") == 0 ||
			 f_stricmp( pszTokenStart, "path") == 0)
		{
			*peType = FLM_FLD_PATH;
		}
		else if (f_stricmp( pszTokenStart, "unicode") == 0 ||
					f_stricmp( pszTokenStart, "text") == 0)
		{
			*peType = FLM_UNICODE_VAL;
		}
		else if (f_stricmp( pszTokenStart, "unsigned") == 0)
		{
			*peType = FLM_UINT32_VAL;
		}
		else if (f_stricmp( pszTokenStart, "unsigned64") == 0)
		{
			*peType = FLM_UINT64_VAL;
		}
		else if (f_stricmp( pszTokenStart, "boolean") == 0)
		{
			*peType = FLM_BOOL_VAL;
		}
		else if (f_stricmp( pszTokenStart, "signed") == 0)
		{
			*peType = FLM_INT32_VAL;
		}
		else if (f_stricmp( pszTokenStart, "signed64") == 0)
		{
			*peType = FLM_INT64_VAL;
		}
		else if (f_stricmp( pszTokenStart, "context") == 0)
		{
			*peType = FLM_REC_PTR_VAL;
		}
		else if (f_stricmp( pszTokenStart, "binary") == 0)
		{
			*peType = FLM_BINARY_VAL;
		}
		
		if (*peType != NO_TYPE)
		{
			FLMBYTE	ucEndChar;

			// Skip white space - should hit a left paren.

			while (*pszStr && *pszStr <= ' ')
			{
				pszStr++;
			}

			// Better have a left paren here, or it is a syntax
			// error.

			if (*pszStr != '(')
			{
				rc = RC_SET( FERR_CURSOR_SYNTAX);
				goto Exit;
			}

			// Skip past left paren

			pszStr++;

			// Skip white space again

			while (*pszStr && *pszStr <= ' ')
			{
				pszStr++;
			}

			// Better not be at end of string or have an empty value

			if (!(*pszStr) || *pszStr == ')')
			{
				rc = RC_SET( FERR_CURSOR_SYNTAX);
				goto Exit;
			}

			// Token can be quoted inside the parens.  However, it doesn't
			// change the type to string.  The type specified takes precedence.

			if (*pszStr == '"' || *pszStr == '\'')
			{
				ucEndChar = *pszStr;
				pszStr++;
			}
			else
			{
				ucEndChar = ')';
			}

			uiLen = 0;
			pszToken = pszTokenStart;
			
			while (*pszStr && *pszStr != ucEndChar)
			{
				*pszToken++ = *pszStr++;
				uiLen++;

				// If we don't have room for a null terminating byte,
				// better reallocate.  Allocate enough so that we can
				// get the entire token.

				if (uiLen == *puiTokenBufSize)
				{
					const char *	pszTmp = pszStr;
					FLMUINT			uiExtraCharsNeeded = 1;	// For NULL

					// See how many more characters we need.

					while (*pszStr && *pszStr != ucEndChar)
					{
						uiExtraCharsNeeded++;
						pszTmp++;
					}

					if (RC_BAD( rc = tokenAllocSpace( &pszTokenStart, &pszToken,
												puiTokenBufSize,
												*puiTokenBufSize + uiExtraCharsNeeded)))
					{
						goto Exit;
					}
				}
			}
			
			*pszToken = 0;

			// If the string is not quoted, trim off trailing white space

			if (ucEndChar == ')')
			{
				while (pszToken > pszTokenStart)
				{
					pszToken--;
					if (*pszToken <= ' ')
					{
						*pszToken = 0;
					}
					else
					{
						break;
					}
				}
			}

			// If we did not hit the end character, we have a syntax error.

			if (*pszStr != ucEndChar)
			{
				rc = RC_SET( FERR_CURSOR_SYNTAX);
				goto Exit;
			}

			// Skip past end character

			pszStr++;

			// If the end character is not a right paren, skip any white space
			// between it and the right paren.

			if (ucEndChar != ')')
			{
				while (*pszStr && *pszStr <= ' ')
				{
					pszStr++;
				}

				// If we did not hit a right paren, it is a syntax error.

				if (*pszStr != ')')
				{
					rc = RC_SET( FERR_CURSOR_SYNTAX);
					goto Exit;
				}

				// Skip the right paren

				pszStr++;
			}
		}
	}

Exit:

	// Return string should be positioned past the last character of the
	// returned token.

	*ppszString = pszStr;
	*ppszToken = pszTokenStart;
	return( rc);
}

/****************************************************************************
Desc:	Determine if a token is an operator.
****************************************************************************/
FSTATIC FLMBOOL tokenIsOperator(
	const char *	pszToken,
	QTYPES *			peOperator)
{
	FLMBOOL	bIsOperator = TRUE;

	if (f_stricmp( pszToken, "(") == 0)
	{
		*peOperator = FLM_LPAREN_OP;
	}
	else if (f_stricmp( pszToken, ")") == 0)
	{
		*peOperator = FLM_RPAREN_OP;
	}
	else if (f_stricmp( pszToken, "&&") == 0 ||
				f_stricmp( pszToken, "AND") == 0)
	{
		*peOperator = FLM_AND_OP;
	}
	else if (f_stricmp( pszToken, "||") == 0 ||
				f_stricmp( pszToken, "OR") == 0)
	{
		*peOperator = FLM_OR_OP;
	}
	else if (f_stricmp( pszToken, "!") == 0)
	{
		*peOperator = FLM_NOT_OP;
	}
	else if (f_stricmp( pszToken, "==") == 0 ||
				f_stricmp( pszToken, "EQ") == 0 ||
				f_stricmp( pszToken, "=") == 0)
	{
		*peOperator = FLM_EQ_OP;
	}
	else if (f_stricmp( pszToken, "!=") == 0 ||
				f_stricmp( pszToken, "NE") == 0)
	{
		*peOperator = FLM_NE_OP;
	}
	else if (f_stricmp( pszToken, "<=") == 0 ||
				f_stricmp( pszToken, "LE") == 0)
	{
		*peOperator = FLM_LE_OP;
	}
	else if (f_stricmp( pszToken, "<") == 0 ||
				f_stricmp( pszToken, "LT") == 0)
	{
		*peOperator = FLM_LT_OP;
	}
	else if (f_stricmp( pszToken, ">=") == 0 ||
				f_stricmp( pszToken, "GE") == 0)
	{
		*peOperator = FLM_GE_OP;
	}
	else if (f_stricmp( pszToken, ">") == 0 ||
				f_stricmp( pszToken, "GT") == 0)
	{
		*peOperator = FLM_GT_OP;
	}
	else if (f_stricmp( pszToken, "MATCH") == 0 ||
				f_stricmp( pszToken, "MATCHES") == 0)
	{
		*peOperator = FLM_MATCH_OP;
	}
	else if (f_stricmp( pszToken, "MATCHBEGIN") == 0 ||
				f_stricmp( pszToken, "MATCH_BEGIN") == 0)
	{
		*peOperator = FLM_MATCH_BEGIN_OP;
	}
	else if (f_stricmp( pszToken, "MATCHEND") == 0 ||
				f_stricmp( pszToken, "MATCH_END") == 0)
	{
		*peOperator = FLM_MATCH_END_OP;
	}
	else if (f_stricmp( pszToken, "CONTAINS") == 0 ||
				f_stricmp( pszToken, "CONTAIN") == 0)
	{
		*peOperator = FLM_CONTAINS_OP;
	}
	else if (f_stricmp( pszToken, "&") == 0 ||
				f_stricmp( pszToken, "BITAND") == 0)
	{
		*peOperator = FLM_BITAND_OP;
	}
	else if (f_stricmp( pszToken, "|") == 0 ||
				f_stricmp( pszToken, "BITOR") == 0)
	{
		*peOperator = FLM_AND_OP;
	}
	else if (f_stricmp( pszToken, "^") == 0 ||
				f_stricmp( pszToken, "BITXOR") == 0)
	{
		*peOperator = FLM_AND_OP;
	}
	else if (f_stricmp( pszToken, "*") == 0)
	{
		*peOperator = FLM_MULT_OP;
	}
	else if (f_stricmp( pszToken, "/") == 0)
	{
		*peOperator = FLM_DIV_OP;
	}
	else if (f_stricmp( pszToken, "%") == 0 ||
				f_stricmp( pszToken, "MOD") == 0)
	{
		*peOperator = FLM_MOD_OP;
	}
	else if (f_stricmp( pszToken, "+") == 0)
	{
		*peOperator = FLM_PLUS_OP;
	}
	else if (f_stricmp( pszToken, "-") == 0)
	{
		*peOperator = FLM_MINUS_OP;
	}
	else
	{
		bIsOperator = FALSE;
	}
	return( bIsOperator);
}

/****************************************************************************
Desc:	Map a field type to a query value type.
****************************************************************************/
FINLINE QTYPES mapFieldTypeToValType(
	FLMUINT	uiFieldType)
{
	switch (uiFieldType)
	{
		case FLM_TEXT_TYPE:
			return( FLM_UNICODE_VAL);
		case FLM_NUMBER_TYPE:
			return( FLM_UINT64_VAL);
		case FLM_CONTEXT_TYPE:
			return( FLM_REC_PTR_VAL);
		case FLM_BINARY_TYPE:
			return( FLM_BINARY_VAL);
	}

	// Should never reach here - but some compilers don't like it when
	// all of the cases are not covered.

	flmAssert( 0);
	return( NO_TYPE);
}

/****************************************************************************
Desc:	Determine if a token is a field.
****************************************************************************/
FSTATIC FLMBOOL tokenIsField(
	const char *		pszToken,
	F_NameTable *		pNameTable,
	FLMUINT *			puiFieldPath,
	QTYPES *				peValueType,
	FLMBOOL				bAllowNamesOnly,
	FLMBOOL				bMustBeField)
{
	FLMBOOL			bIsField = TRUE;
	FLMUINT			uiFieldCount = 0;
	FLMUINT			uiFieldNum;
	FLMUINT			uiDictType;
	FLMBOOL			bIsTagName;
	FLMBOOL			bIsNum;
	FLMBOOL			bIsDrnNum;
	FLMBOOL			bIsTagNum;
	FLMUINT64		ui64Num;
	FLMUINT			uiNum;
	FLMUINT64		ui64TagNum;
	FLMUINT			uiFieldType;
	char				szNameBuf[ 128];
	char *			pszNameEnd;

	// Parse out each part of the name.

	while( *pszToken)
	{
		// Path components are separated by periods or spaces.  They may be names or
		// tag numbers.
		
		pszNameEnd = &szNameBuf[ 0];

		while( *pszToken && *pszToken != '.')
		{
			*pszNameEnd++ = *pszToken;
			pszToken++;
		}

		*pszNameEnd = 0;

		// If we detect a period, the bAllowNamesOnly restriction is lifted.

		if( *pszToken == '.')
		{
			bAllowNamesOnly = FALSE;
		}
		
		if( pNameTable)
		{
			bIsTagName = pNameTable->getFromTagTypeAndName( NULL, szNameBuf,
										FLM_FIELD_TAG, &uiFieldNum, &uiFieldType);
		}
		else
		{
			bIsTagName = FALSE;
		}
		
		bIsNum = tokenIsNum( szNameBuf, (FLMUINT64)(FLM_MAX_UINT16), &ui64Num);
		uiNum = (FLMUINT)ui64Num;
		bIsDrnNum = FALSE;
		
		if( f_stricmp( szNameBuf, "DRN") == 0)
		{
			bIsDrnNum = TRUE;
			uiNum = FLM_RECID_FIELD;
		}
		
		bIsTagNum = (f_strnicmp( szNameBuf, "TAG_", 4) == 0 &&
						 tokenIsNum( &szNameBuf[ 4], (FLMUINT64)65535, &ui64TagNum))
						 ? TRUE
						 : FALSE;
						 
		// See if the token is a field name.
		//
		// NOTE: If it is a number and also comes out as a legitimate tag name,
		// we have a bit of an ambiguity.  In that case, we will ignore the tag
		// name if the bMustBeField flag is TRUE, because that means it was
		// specified inside a field() construct: e.g., field( 60).  In these
		// cases, we would take the literal number, even though the string "60"
		// also happens to map to a tag name.

		if( !bIsTagName || (bIsNum && bMustBeField))
		{
			if( bAllowNamesOnly)
			{
				bIsField = FALSE;
				goto Exit;
			}

			// Not a field name, see if it is a number in the proper
			// range for field numbers.

			if( bIsNum || bIsDrnNum)
			{
				uiFieldNum = uiNum;
			}
			else if( bIsTagNum)
			{
				uiFieldNum = (FLMUINT)ui64TagNum;
			}
			else
			{
				bIsField = FALSE;
				goto Exit;
			}
			
			if( !uiFieldNum || uiFieldNum > 0xFFFF)
			{
				bIsField = FALSE;
				goto Exit;
			}
			
			if( uiFieldNum >= FLM_UNREGISTERED_TAGS)
			{
				// This is only valid if the field name began with "TAG_"
				// Otherwise, we will treat it as a value, not a field.

				if( !bIsTagNum)
				{
					bIsField = FALSE;
					goto Exit;
				}
				
				*peValueType = NO_TYPE;
			}
			else if( uiFieldNum == FLM_RECID_FIELD)
			{
				*peValueType = FLM_REC_PTR_VAL;
			}
			else if( pNameTable &&
						pNameTable->getFromTagNum( uiFieldNum, NULL,
											NULL, 0, &uiDictType, &uiFieldType) &&
						uiDictType == FLM_FIELD_TAG)
			{
				*peValueType = mapFieldTypeToValType( uiFieldType);
			}
			else if( bMustBeField)
			{
				// Couldn't find in name table, or name table was NULL,
				// so we don't know its type, but we know that this is
				// inside a field() construct - so we have to take it as is.

				*peValueType = NO_TYPE;
			}
			else
			{
				bIsField = FALSE;
				goto Exit;
			}
		}
		else
		{
			*peValueType = mapFieldTypeToValType( uiFieldType);

			// If the name happens to be a number and we are only allowing
			// names, don't use this one.

			if( bIsNum && bAllowNamesOnly)
			{
				bIsField = FALSE;
				goto Exit;
			}
		}
		
		puiFieldPath [uiFieldCount++] = uiFieldNum;

		// If *pszToken is non-null (a period), skip it.

		if( *pszToken)
		{
			pszToken++;
		}
	}

	// A field count of zero means we didn't get anything that
	// looked like a legitimate field.

	if( !uiFieldCount)
	{
		bIsField = FALSE;
		goto Exit;
	}

	// Null terminate the field path.

	puiFieldPath[ uiFieldCount] = 0;

Exit:

	return( bIsField);
}

/****************************************************************************
Desc:	Allocate buffer space for a query value.
****************************************************************************/
FSTATIC RCODE allocValueSpace(
	void **		ppvVal,
	FLMUINT *	puiValBufSize,
	FLMUINT		uiNewSize)
{
	RCODE			rc = FERR_OK;
	void *		pvVal;

	if( RC_BAD( rc = f_alloc( uiNewSize, &pvVal)))
	{
		goto Exit;
	}
	
	f_free( ppvVal);
	
	*ppvVal = pvVal;
	*puiValBufSize = uiNewSize;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Get a unicode value from a token.
****************************************************************************/
RCODE tokenGetUnicode(
	const char *	pszToken,
	void **			ppvVal,
	FLMUINT *		puiValLen,
	FLMUINT *		puiValBufSize)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiLen = f_strlen( pszToken) + 1;
	FLMUNICODE *	puzTmp;

	// Just make sure we have enough to cover the maximum number
	// of unicode characters that could be created.

	if( *puiValBufSize < uiLen * sizeof( FLMUNICODE))
	{
		if( RC_BAD( rc = allocValueSpace( ppvVal, puiValBufSize,
			uiLen * sizeof( FLMUNICODE))))
		{
			goto Exit;
		}
	}

	// Convert the ASCII to unicode.

	puzTmp = (FLMUNICODE *)(*ppvVal);
	while (*pszToken)
	{
		if (*pszToken != '~' || *(pszToken + 1) != '[')
		{
			*puzTmp++ = (FLMUNICODE)(*pszToken);
			pszToken++;
		}
		else
		{
			const char *	pszSave = pszToken;
			FLMUNICODE *	puzSave = puzTmp;

			// Skip the ~[

			pszToken += 2;

			// Unicode characters may be specified as numbers
			// separated by commas and/or spaces.  The list
			// ends when we hit a ']' character or a non-numeric
			// value.

			while (*pszToken && *pszToken != ']')
			{
				char 			szNumBuf[ 32];
				char *		pszNumEnd;
				FLMBOOL		bIsNum;
				FLMUINT64	ui64Num;

				// Skip white space and commas

				while (*pszToken && (*pszToken <= ' ' || *pszToken == ','))
				{
					pszToken++;
				}

				// Number should start here.

				pszNumEnd = &szNumBuf[ 0];
				while( *pszToken > ' ' && *pszToken != ',' && *pszToken != ']')
				{
					*pszNumEnd++ = *pszToken;
					pszToken++;
				}
				
				*pszNumEnd = 0;

				// If we ended on a non-null character, see if it is a number.
				// If not, we will quit processing.

				if( !(*pszToken))
				{
					pszToken = pszSave;
					break;
				}

				bIsNum = tokenIsNum( szNumBuf, (FLMUINT64)(FLM_MAX_UINT16), &ui64Num);

				if( bIsNum && ui64Num && ui64Num <= 0xFFFE)
				{
					*puzTmp++ = (FLMUNICODE)ui64Num;
				}
				else
				{

					// Resetting pszToken to pszSave so that we will know that
					// we did not successfully process everything in the ~[].

					pszToken = pszSave;
					break;
				}
			}

			// If we hit a ']', we successfully processed whatever was between
			// the ~[], and we can skip the ']'.  If we did not hit the ']'
			// we were not able to interpret everything between the ~[] as unicode
			// characters, so we should just go back and redo everything between
			// the ~[] as ascii - including the ~[].

			if (*pszToken == ']')
			{
				pszToken++;
			}
			else
			{
				// UNICODE-ize the rest of the string, including the ~[ and
				// up to a trailing ']', if any.

				pszToken = pszSave;
				puzTmp = puzSave;
				
				while (*pszToken && *pszToken != ']')
				{
					*puzTmp++ = (FLMUNICODE)(*pszToken);
					pszToken++;
				}

				// Get trailing ']', if any

				if (*pszToken == ']')
				{
					*puzTmp++ = (FLMUNICODE)(*pszToken);
					pszToken++;
				}
			}
		}
	}

	// Null terminate

	*puzTmp = 0;

Exit:

	*puiValLen = 0;
	return( rc);
}

/****************************************************************************
Desc:	Get a binary value from a token.
****************************************************************************/
FSTATIC RCODE tokenGetBinary(
	const char *	pszToken,
	void **			ppvVal,
	FLMUINT *		puiValLen,
	FLMUINT *		puiValBufSize)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiLen = f_strlen( pszToken);
	FLMBYTE *		pucTmp;
	FLMUINT			uiBinaryChars;
	FLMUINT			uiNibble;
	FLMBOOL			bFirstNibble;

	// Just make sure we have enough to cover the maximum number
	// of binary characters that could be created.

	if (*puiValBufSize < uiLen / 2 + 1)
	{
		if (RC_BAD( rc = allocValueSpace( ppvVal, puiValBufSize,
									uiLen / 2 + 1)))
		{
			goto Exit;
		}
	}

	// Convert the ASCII to unicode.

	pucTmp = (FLMBYTE *)(*ppvVal);
	uiBinaryChars = 0;
	bFirstNibble = TRUE;
	while (*pszToken)
	{
		if (*pszToken >= '0' && *pszToken <= '9')
		{
			uiNibble = (FLMUINT)(*pszToken - '0');
		}
		else if (*pszToken >= 'A' && *pszToken <= 'F')
		{
			uiNibble = (FLMUINT)(*pszToken - 'A' + 10);
		}
		else if (*pszToken >= 'a' && *pszToken <= 'f')
		{
			uiNibble = (FLMUINT)(*pszToken - 'a' + 10);
		}
		else
		{
			rc = RC_SET( FERR_BAD_FIELD_TYPE);
			goto Exit;
		}
		pszToken++;

		if (bFirstNibble)
		{
			uiBinaryChars++;
			*pucTmp = (FLMBYTE)(uiNibble << 4);
			bFirstNibble = FALSE;
		}
		else
		{
			(*pucTmp) |= (FLMBYTE)(uiNibble);
			bFirstNibble = TRUE;
			pucTmp++;
		}
	}

	// If we ended on an odd number of characters, shift our last character
	// down by four bits.

	if (!bFirstNibble)
	{
		(*pucTmp) >>= 4;
	}

	*puiValLen = uiBinaryChars;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Parse a value and return it.
****************************************************************************/
FSTATIC RCODE tokenGetValue(
	QTYPES			eValueType,
	const char *	pszToken,
	FLMBOOL			bQuoted,
	void **			ppvVal,
	QTYPES *			peValType,
	FLMUINT *		puiValLen,
	FLMUINT *		puiValBufSize)
{
	RCODE				rc = FERR_OK;
	FLMINT32			i32Num;
	FLMUINT32		ui32Num;
	FLMUINT64		ui64Num;
	FLMINT64			i64Num;

	if (bQuoted || eValueType == FLM_UNICODE_VAL)
	{
		*peValType = FLM_UNICODE_VAL;
		if (RC_BAD( rc = tokenGetUnicode( pszToken, ppvVal, puiValLen,
									puiValBufSize)))
		{
			goto Exit;
		}
	}
	else if (eValueType == FLM_UINT32_VAL ||
				eValueType == FLM_UINT64_VAL ||
				eValueType == FLM_INT32_VAL ||
				eValueType == FLM_INT64_VAL)
	{
		FLMUINT64	ui64Max;
		FLMBOOL		bNeg;
		
		if (*pszToken == '-')
		{
			bNeg = TRUE;
			pszToken++;
		}
		else
		{
			bNeg = FALSE;
		}
		
		if (eValueType == FLM_UINT32_VAL)
		{
			if (bNeg)
			{
				ui64Max = (FLMUINT64)(FLM_MAX_INT32) + 1;
			}
			else
			{
				ui64Max = (FLMUINT64)(FLM_MAX_UINT32);
			}
		}
		else if (eValueType == FLM_INT32_VAL)
		{
			if (bNeg)
			{
				ui64Max = (FLMUINT64)(FLM_MAX_INT32) + 1;
			}
			else
			{
				ui64Max = (FLMUINT64)(FLM_MAX_INT32);
			}
		}
		else if (eValueType == FLM_UINT64_VAL)
		{
			if (bNeg)
			{
				ui64Max = (FLMUINT64)(FLM_MAX_INT64) + 1;
			}
			else
			{
				ui64Max = FLM_MAX_UINT64;
			}
		}
		else
		{
			if (bNeg)
			{
				ui64Max = (FLMUINT64)(FLM_MAX_INT64) + 1;
			}
			else
			{
				ui64Max = (FLMUINT64)(FLM_MAX_INT64);
			}
		}
		if (tokenIsNum( pszToken, ui64Max, &ui64Num))
		{
			if (*puiValBufSize < sizeof( FLMUINT64))
			{
				if (RC_BAD( rc = allocValueSpace( ppvVal, puiValBufSize,
											sizeof( FLMUINT64))))
				{
					goto Exit;
				}
			}
			if (bNeg || eValueType == FLM_INT32_VAL || eValueType == FLM_INT64_VAL)
			{
				if (eValueType == FLM_INT32_VAL || eValueType == FLM_UINT32_VAL)
				{
					if (bNeg)
					{
						if (ui64Num == ui64Max)
						{
							
							// If the number is negative, the maximum will have been
							// set up to be the minimum negative 32 bit integer.
							
							i32Num = FLM_MIN_INT32;
						}
						else
						{
							i32Num = -((FLMINT32)ui64Num);
						}
					}
					else
					{
						i32Num = (FLMINT32)ui64Num;
					}
					*peValType = FLM_INT32_VAL;
					*((FLMINT32 *)(*ppvVal)) = i32Num;
					*puiValLen = sizeof( FLMINT32);
				}
				else
				{
					if (bNeg)
					{
						if (ui64Num == ui64Max)
						{
							
							// If the number is negative, the maximum will have been
							// set up to be the minimum negative 64 bit integer.
							
							i64Num = FLM_MIN_INT64;
						}
						else
						{
							i64Num = -((FLMINT64)ui64Num);
						}
					}
					else
					{
						i64Num = (FLMINT64)ui64Num;
					}
					*peValType = FLM_INT64_VAL;
					*((FLMINT64 *)(*ppvVal)) = i64Num;
					*puiValLen = sizeof( FLMINT64);
				}
			}
			else
			{
				if (eValueType == FLM_UINT32_VAL)
				{
					ui32Num = (FLMUINT32)ui64Num;
					*peValType = FLM_UINT32_VAL;
					*((FLMUINT32 *)(*ppvVal)) = ui32Num;
					*puiValLen = sizeof( FLMUINT32);
				}
				else
				{
					*peValType = FLM_UINT64_VAL;
					*((FLMUINT64 *)(*ppvVal)) = ui64Num;
					*puiValLen = sizeof( FLMUINT64);
				}
			}
		}
		else
		{
			rc = RC_SET( FERR_BAD_FIELD_TYPE);
			goto Exit;
		}
	}
	else if (eValueType == FLM_REC_PTR_VAL)
	{
		if (*puiValBufSize < sizeof( FLMUINT32))
		{
			if (RC_BAD( rc = allocValueSpace( ppvVal, puiValBufSize,
										sizeof( FLMUINT32))))
			{
				goto Exit;
			}
		}
		if (tokenIsNum( pszToken, (FLMUINT64)(FLM_MAX_UINT32), &ui64Num))
		{
			ui32Num = (FLMUINT32)ui64Num;
			*peValType = FLM_REC_PTR_VAL;
			*((FLMUINT32 *)(*ppvVal)) = ui32Num;
			*puiValLen = sizeof( FLMUINT32);
		}
		else
		{
			rc = RC_SET( FERR_BAD_FIELD_TYPE);
			goto Exit;
		}
	}
	else if (eValueType == FLM_BOOL_VAL)
	{
		if (*puiValBufSize < sizeof( FLMBOOL))
		{
			if (RC_BAD( rc = allocValueSpace( ppvVal, puiValBufSize,
										sizeof( FLMBOOL))))
			{
				goto Exit;
			}
		}
		if (tokenIsNum( pszToken, FLM_MAX_UINT64, &ui64Num))
		{
			*peValType = FLM_BOOL_VAL;
			*((FLMBOOL *)(*ppvVal)) = ui64Num ? TRUE : FALSE;
			*puiValLen = sizeof( FLMBOOL);
		}
		else if (f_stricmp( pszToken, "false") == 0)
		{
			*peValType = FLM_BOOL_VAL;
			*((FLMBOOL *)(*ppvVal)) = FALSE;
			*puiValLen = sizeof( FLMBOOL);
		}
		else if (f_stricmp( pszToken, "true") == 0)
		{
			*peValType = FLM_BOOL_VAL;
			*((FLMBOOL *)(*ppvVal)) = TRUE;
			*puiValLen = sizeof( FLMBOOL);
		}
		else if (f_stricmp( pszToken, "unknown") == 0)
		{
			*peValType = FLM_BOOL_VAL;
			if (*ppvVal)
			{
				f_free( ppvVal);
			}
			*puiValBufSize = 0;
			*puiValLen = 0;
		}
		else
		{
			rc = RC_SET( FERR_BAD_FIELD_TYPE);
			goto Exit;
		}
	}
	else if (eValueType == FLM_BINARY_VAL)
	{
		*peValType = FLM_BINARY_VAL;
		if (RC_BAD( rc = tokenGetBinary( pszToken, ppvVal, puiValLen,
									puiValBufSize)))
		{
			goto Exit;
		}
	}
	else
	{

		// Field type is unknown.  Try converting to
		// a number.  If that doesn't work, simply
		// use text.

		if (tokenIsNum( pszToken, FLM_MAX_UINT64, &ui64Num))
		{
			if (ui64Num <= (FLMUINT64)(FLM_MAX_UINT32))
			{
				ui32Num = (FLMUINT32)ui64Num;
				*peValType = FLM_UINT32_VAL;
				*((FLMUINT32 *)(*ppvVal)) = ui32Num;
				*puiValLen = sizeof( FLMUINT32);
			}
			else
			{
				*peValType = FLM_UINT64_VAL;
				*((FLMUINT64 *)(*ppvVal)) = ui64Num;
				*puiValLen = sizeof( FLMUINT64);
			}
		}
		else
		{
			*peValType = FLM_UNICODE_VAL;
			if (RC_BAD( rc = tokenGetUnicode( pszToken, ppvVal, puiValLen,
										puiValBufSize)))
			{
				goto Exit;
			}
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Parse a query criteria string and populate an HFCURSOR from it.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmParseQuery(
	HFCURSOR				hCursor,
	F_NameTable *		pNameTable,
	const char *		pszQueryCriteria)
{
	RCODE					rc = FERR_OK;
	const char *		pszTmp = pszQueryCriteria;
	QTYPES				eTokenType;
	QTYPES				ePriorTokenType;
	void *				pvVal = NULL;
	FLMUINT				uiValBufSize;
	FLMUINT				uiValLen;
	QTYPES				eLastFldType = NO_TYPE;
	FLMUINT				uiFieldPath [20];
	char *				pszToken = NULL;
	FLMUINT				uiTokenBufSize;
	FLMBOOL				bQuoted;
	FLMUINT64			ui64Num;

	// Allocate space for tokens and values.  These will be
	// reallocated as needed.

	uiValBufSize = 512;
	if (RC_BAD( rc = f_alloc( uiValBufSize, &pvVal)))
	{
		goto Exit;
	}
	
	uiTokenBufSize = 512;
	if (RC_BAD( rc = f_alloc( uiTokenBufSize, &pszToken)))
	{
		goto Exit;
	}

	ePriorTokenType = NO_TYPE;
	for (;;)
	{
		// Get the next token.

		if( RC_BAD( rc = tokenGet( &pszTmp, &pszToken, &bQuoted, 
			&eTokenType, &uiTokenBufSize)))
		{
			goto Exit;
		}

		// If the token is empty, we are done

		if (!(*pszToken))
		{
			break;
		}

		// If eTokenType is not explicit, attempt to figure it out

		if (eTokenType == NO_TYPE)
		{
			if (bQuoted)
			{
				eLastFldType = FLM_UNICODE_VAL;
Get_Value:
				if (RC_BAD( rc = tokenGetValue( eLastFldType,
									pszToken, bQuoted, &pvVal, &eTokenType,
									&uiValLen, &uiValBufSize)))
				{
					goto Exit;
				}
				
				if (RC_BAD( rc = FlmCursorAddValue( hCursor, eTokenType,
											pvVal, uiValLen)))
				{
					goto Exit;
				}
				
				eLastFldType = NO_TYPE;
			}
			else if (tokenIsOperator( pszToken, &eTokenType))
			{
				// If this token is a minus operator, and the prior token was
				// a comparison operator, arithmetic operator, or
				// left paren, change this token to a negative sign.

				if (eTokenType == FLM_MINUS_OP &&
					 (ePriorTokenType == FLM_LPAREN_OP ||
						IS_COMPARE_OP( ePriorTokenType) ||
						IS_ARITH_OP( ePriorTokenType)))
				{
					eTokenType = FLM_NEG_OP;
				}
				
				if (RC_BAD( rc = FlmCursorAddOp( hCursor, eTokenType, TRUE)))
				{
					goto Exit;
				}

				// If the operator is not a comparison operator
				// and not an arithmetic operator and not a negative
				// operator, then the last field type is now unknown.

				if (!IS_COMPARE_OP( eTokenType) &&
					 !IS_ARITH_OP( eTokenType) &&
					 eTokenType != FLM_NEG_OP)
				{
					eLastFldType = NO_TYPE;
				}
			}
			else if (tokenIsField( pszToken, pNameTable,
								uiFieldPath, &eLastFldType, TRUE, FALSE))
			{
Add_Field:
				eTokenType = FLM_FLD_PATH;
				if (RC_BAD( rc = FlmCursorAddFieldPath( hCursor, uiFieldPath, 0)))
				{
					goto Exit;
				}
			}
			else if (IS_COMPARE_OP( ePriorTokenType) ||
						IS_ARITH_OP( ePriorTokenType) ||
						ePriorTokenType == FLM_NEG_OP)
			{
				if (ePriorTokenType == FLM_NEG_OP)
				{
					eLastFldType = FLM_INT32_VAL;
				}
				
				goto Get_Value;
			}
			else if (tokenIsField( pszToken, pNameTable,
									uiFieldPath, &eLastFldType, FALSE, FALSE))
			{
				goto Add_Field;
			}
			else if (tokenIsNum( pszToken, FLM_MAX_UINT64, &ui64Num))
			{
				goto Get_Value;
			}
			else
			{
				rc = RC_SET( FERR_CURSOR_SYNTAX);
				goto Exit;
			}
		}
		else
		{
			switch (eTokenType)
			{
				case FLM_FLD_PATH:
				{
					if (!tokenIsField( pszToken, pNameTable,
								uiFieldPath, &eLastFldType, FALSE, TRUE))
					{
						rc = RC_SET( FERR_CURSOR_SYNTAX);
						goto Exit;
					}
					
					goto Add_Field;
				}
				
				case FLM_UNICODE_VAL:
				case FLM_INT32_VAL:
				case FLM_UINT32_VAL:
				case FLM_UINT64_VAL:
				case FLM_INT64_VAL:
				case FLM_REC_PTR_VAL:
				case FLM_BINARY_VAL:
				case FLM_BOOL_VAL:
				{
					eLastFldType = eTokenType;
					goto Get_Value;
				}
				
				default:
				{
					rc = RC_SET( FERR_CURSOR_SYNTAX);
					goto Exit;
				}
			}
		}
		
		ePriorTokenType = eTokenType;
	}

Exit:

	if( pszToken)
	{
		f_free( &pszToken);
	}

	if( pvVal)
	{
		f_free( &pvVal);
	}

	return( rc);
}

/****************************************************************************
Desc:	Determine if a token is a number.
****************************************************************************/
FSTATIC FLMBOOL tokenIsNum(
	const char *	pszToken,
	FLMUINT64		ui64Max,
	FLMUINT64 *		pui64Num)
{
	FLMBOOL		bIsNum = TRUE;
	FLMUINT64	ui64Num;
	FLMBOOL		bAllowHex = FALSE;

	if (*pszToken == 0)
	{
		bIsNum = FALSE;
		goto Exit;
	}

	if (*pszToken == '0' && 
		(*(pszToken + 1) == 'x' || *(pszToken + 1) == 'X'))
	{
		pszToken += 2;
		bAllowHex = TRUE;
	}

	ui64Num = 0;
	while (*pszToken)
	{
		if (*pszToken >= '0' && *pszToken <= '9')
		{
			if (!bAllowHex)
			{
				if (ui64Num > ui64Max / 10)
				{

					// Number would overflow.

					bIsNum = FALSE;
					goto Exit;
				}
				else
				{
					ui64Num *= 10;
				}
			}
			else
			{
				if (ui64Num > ui64Max >> 4)
				{

					// Number would overflow.

					bIsNum = FALSE;
					goto Exit;
				}
				ui64Num <<= 4;
			}
			if (ui64Num > ui64Max - (FLMUINT64)(*pszToken - '0'))
			{
				
				// Number would overflow.
				
				bIsNum = FALSE;
				goto Exit;
			}
			ui64Num += (FLMUINT64)(*pszToken - '0');
		}
		else if (bAllowHex)
		{
			if (ui64Num > ui64Max >> 4)
			{

				// Number would overflow.

				bIsNum = FALSE;
				goto Exit;
			}
			if (*pszToken >= 'A' && *pszToken <= 'F')
			{
				ui64Num <<= 4;
				if (ui64Num > ui64Max - (FLMUINT64)(*pszToken - 'A' + 10))
				{
					
					// Number would overflow.
					
					bIsNum = FALSE;
					goto Exit;
				}
				ui64Num += (FLMUINT)(*pszToken - 'A') + 10;
			}
			else if (*pszToken >= 'a' && *pszToken <= 'f')
			{
				ui64Num <<= 4;
				if (ui64Num > ui64Max - (FLMUINT64)(*pszToken - 'a' + 10))
				{
					
					// Number would overflow.
					
					bIsNum = FALSE;
					goto Exit;
				}
				ui64Num += (FLMUINT)(*pszToken - 'a') + 10;
			}
			else
			{
				bIsNum = FALSE;
				goto Exit;
			}
		}
		else
		{
			bIsNum = FALSE;
			goto Exit;
		}
		pszToken++;
	}

	*pui64Num = ui64Num;

Exit:

	return( bIsNum);
}

