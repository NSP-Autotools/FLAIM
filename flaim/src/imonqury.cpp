//-------------------------------------------------------------------------
// Desc:	Class for displaying the query page for monitoring via HTTP.
// Tabs:	3
//
// Copyright (c) 2002-2003, 2005-2007 Novell, Inc. All Rights Reserved.
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

#define Q_FIELD_COLOR		FLM_MAGENTA
#define Q_LABEL_COLOR		FLM_BLACK
#define Q_VALUE_COLOR		FLM_GREEN
#define Q_PARAM_COLOR		FLM_BLUE
#define Q_OPERATOR_COLOR	FLM_BLUE
#define Q_YES_COLOR			FLM_GREEN
#define Q_NO_COLOR			FLM_RED
#define Q_DOT_COLOR			FLM_BLACK

FSTATIC FLMBOOL findSubQuery(
	CURSOR *		pCursor,
	SUBQUERY *	pSubQuery);

/****************************************************************************
Desc: Class for formatting a query into HTML
*****************************************************************************/
class F_QueryFormatter
{
public:
	F_QueryFormatter();
	~F_QueryFormatter();

	void formatQuery(
		HRequest *	pHRequest,
		F_WebPage *	pWebPage,
		CURSOR *		pCursor,
		FLMBOOL		bSingleLineOnly,
		FLMUINT		uiMaxChars);

	void outputSubqueryStats(
		HRequest *	pHRequest,
		F_WebPage *	pWebPage,
		SUBQUERY *	pSubQuery);

private:

	void outputStr(
		const char *	pszStr);

	void changeColor(
		eColorType		eColor,
		FLMBOOL			bForceColor = FALSE);

	void appendString(
		const char *	pszStr,
		eColorType		eColor = FLM_CURRENT_COLOR,
		FLMBOOL			bForceColor = FALSE);

	void newline( void);

	void outputIndent(
		FLMUINT	uiIndent);

	void outputOperator(
		QTYPES			eOperator,
		FLMBOOL			bEndLine);

	FINLINE void formatHex(
		char *	pszBuf,
		FLMBYTE	ucChar)
	{
		FLMUINT	uiNibble = ((FLMUINT)ucChar >> 4) & 0xF;

		*pszBuf++ = (char)(uiNibble <= 9
								 ? (char)(uiNibble + '0')
								 : (char)(uiNibble - 10 + 'A'));
		uiNibble = (FLMUINT)ucChar & 0xF;
		*pszBuf = (char)(uiNibble <= 9
							  ? (char)(uiNibble + '0')
							  : (char)(uiNibble - 10 + 'A'));
	}

	void outputBinary(
		FLMBYTE *		pucBuf,
		FLMUINT			uiBufLen,
		eColorType		eValueColor = Q_VALUE_COLOR);

	void outputText(
		FLMBYTE *		pucBuf,
		FLMUINT			uiBufLen,
		eColorType		eColor = Q_VALUE_COLOR);

	void outputPredicate(
		FLMUINT			uiIndent,
		FQNODE *			pQNode);

	void outputSubQuery(
		FLMUINT			uiIndent,
		QTYPES			eParentOp,
		CURSOR *			pReferenceCursor,
		SUBQUERY *		pSubQuery);

	void outputQuery(
		FLMUINT			uiIndent,
		CURSOR *			pReferenceCursor,
		CURSOR *			pCursor);

	void outputLabel(
		const char *	pszLabel,
		eColorType		eLabelColor = Q_LABEL_COLOR);

	void outputStringRow(
		const char *	pszLabel,
		const char *	pszValue,
		eColorType		eLabelColor = Q_LABEL_COLOR,
		eColorType		eValueColor = Q_VALUE_COLOR);

	void outputYesNoRow(
		const char *	pszLabel,
		FLMBOOL			bYesNo,
		eColorType		eLabelColor = Q_LABEL_COLOR,
		eColorType		eYesColor = Q_YES_COLOR,
		eColorType		eNoColor = Q_NO_COLOR);

	void outputUINTRow(
		const char *	pszLabel,
		FLMUINT			uiValue,
		eColorType		eLabelColor = Q_LABEL_COLOR,
		eColorType		eValueColor = Q_VALUE_COLOR);

	void outputBinaryRow(
		const char *	pszLabel,
		FLMBYTE *		pucValue,
		FLMUINT			uiValueLen,
		eColorType		eLabelColor,
		eColorType		eValueColor);

	HRequest *		m_pHRequest;
	F_WebPage *		m_pWebPage;
	eColorType		m_eCurrColor;
	FLMBOOL			m_bSingleLineOnly;
	FLMUINT			m_uiMaxChars;
	FLMUINT			m_uiVisibleChars;
	FLMUINT			m_uiRowCount;
};


/****************************************************************************
Desc:	Prints the web page for the list of queries.
****************************************************************************/
RCODE F_QueriesPage::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE					rc = FERR_OK;
	HFCURSOR				hCursor;
	const char *		pszTitle = "Queries";
	char					szTemp [100];
	QUERY_HDR *			pQueryHdr;
	F_QueryFormatter	qf;
	FLMBOOL				bMutexLocked = FALSE;
	FLMUINT				uiRows;
	
	F_UNREFERENCED_PARM( uiNumParams);
	F_UNREFERENCED_PARM( ppszParams);

	// Start the document

	printDocStart( pszTitle, FALSE);

	// Begin the table

	printTableStart( pszTitle, 3);

	// Write out the table headings

	printTableRowStart();
	printColumnHeading( "Query Criteria");
	printColumnHeading( "Terminate Status");
	printColumnHeading( "Record Count");
	printTableRowEnd();

	// Lock the mutex on the queries

	f_mutexLock( gv_FlmSysData.hQueryMutex);
	bMutexLocked = TRUE;

	// Position to the most recent query and display in order from
	// newest to oldest.

	uiRows = 0;
	pQueryHdr = gv_FlmSysData.pNewestQuery;
	while (pQueryHdr)
	{
		char			szAddress [20];
		CURSOR *		pCursor;
		SUBQUERY *	pSubQuery;
		FLMUINT		uiRecCount;

		hCursor = pQueryHdr->hCursor;
		pCursor = (CURSOR *)hCursor;

		// Setup a hyperlink for the query.

		printAddress( (void *)hCursor, szAddress);
		f_sprintf((char *)szTemp, 
						"%s/Query?QueryHandle=%s",
						m_pszURLString,
						szAddress);

		// Output a row for the query

		printTableRowStart( (++uiRows) & 0x00000001 ? TRUE : FALSE);

		// Start a cell for the data

		printTableDataStart();

		// Hyperlink and query criteria

		fnPrintf( m_pHRequest, "<a href=\"%s\">", szTemp);

		// Format query - up to 80 viewable characters.

		qf.formatQuery( m_pHRequest, this, pCursor, TRUE, 80);

		// Close the hyperlink

		fnPrintf( m_pHRequest, "</a>");

		// Close the cell

		printTableDataEnd();

		// Query status

		if (pCursor->rc == FERR_EOF_HIT)
		{
			f_strcpy( szTemp, "EOF");
		}
		else if (pCursor->rc == FERR_BOF_HIT)
		{
			f_strcpy( szTemp, "BOF");
		}
		else if (pCursor->rc != FERR_OK)
		{
			f_sprintf( (char *)szTemp, "Error: %04X", (unsigned)pCursor->rc);
		}
		else
		{
			f_strcpy( szTemp, "App Ended"); 
		}

		printTableDataStart();
		fnPrintf( m_pHRequest, "%s", szTemp);
		printTableDataEnd();

		// Count of records returned

		uiRecCount = 0;
		pSubQuery = pCursor->pSubQueryList;
		while (pSubQuery)
		{
			uiRecCount += pSubQuery->SQStatus.uiMatchedCnt;
			pSubQuery = pSubQuery->pNext;
		}
		fnPrintf( m_pHRequest, TD_ui, (unsigned)uiRecCount);

		printTableRowEnd();
		pQueryHdr = pQueryHdr->pNext;
	}

	// Unlock the mutex on the queries

	f_mutexUnlock( gv_FlmSysData.hQueryMutex);
	bMutexLocked = FALSE;

	printTableEnd();
	printDocEnd();

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hQueryMutex);
	}

	fnEmit();
	return( rc);
}

/****************************************************************************
Desc: Constructor
*****************************************************************************/
F_QueryFormatter::F_QueryFormatter()
{
	m_eCurrColor = FLM_CURRENT_COLOR;
	m_uiRowCount = 0;
	m_bSingleLineOnly = 0;
	m_uiMaxChars = 0;
	m_uiVisibleChars = 0;
	m_pWebPage = NULL;
	m_pHRequest = NULL;
}

/****************************************************************************
Desc: Destuctor
*****************************************************************************/
F_QueryFormatter::~F_QueryFormatter()
{
}

/****************************************************************************
Desc: Output a null-terminated string to the buffer.
****************************************************************************/
void F_QueryFormatter::outputStr(
	const char *	pszStr)
{
	gv_FlmSysData.HttpConfigParms.fnPrintf( m_pHRequest, pszStr);
}

/****************************************************************************
Desc: Output a new color code, but only if the color has changed.
****************************************************************************/
void F_QueryFormatter::changeColor(
	eColorType		eColor,
	FLMBOOL			bForceColor)
{
	const char *	pszColor;

	// If we are doing a single line, and we have filled our buffer
	// with the maximum visible characters, we are not going to add
	// anything else to the buffer.

	if (m_bSingleLineOnly && m_uiVisibleChars == m_uiMaxChars)
	{
		goto Exit;
	}

	// If no color has yet been set, and none was passed in,
	// set to a default of light gray.

	if (eColor == FLM_CURRENT_COLOR && m_eCurrColor == FLM_CURRENT_COLOR)
	{
		eColor = FLM_LIGHTGRAY;
	}
	
	// Change colors if necessary

	if (bForceColor || (eColor != FLM_CURRENT_COLOR && eColor != m_eCurrColor))
	{

		m_eCurrColor = eColor;

		switch (eColor)
		{
			case FLM_BLACK:
				pszColor = "<font color=\"Black\">";
				break;
			case FLM_BLUE:
				pszColor = "<font color=\"Blue\">";
				break;
			case FLM_GREEN:
				pszColor = "<font color=\"Green\">";
				break;
			case FLM_CYAN:
				//VISIT: Redo this one
				pszColor = "<font color=\"Teal\">";
				break;
			case FLM_RED:
				pszColor = "<font color=\"Red\">";
				break;
			case FLM_MAGENTA:
				pszColor = "<font color=\"Purple\">";
				break;
			case FLM_BROWN:
				//VISIT: Redo this one
				pszColor = "<font color=\"Maroon\">";
				break;
			case FLM_LIGHTGRAY:
				pszColor = "<font color=\"Gray\">";
				break;
			case FLM_DARKGRAY:
				//VISIT: Redo this one
				pszColor = "<font color=\"Gray\">";
				break;
			case FLM_LIGHTBLUE:
				//VISIT: Redo this one
				pszColor = "<font color=\"Navy\">";
				break;
			case FLM_LIGHTGREEN:
				//VISIT: Redo this one
				pszColor = "<font color=\"Lime\">";
				break;
			case FLM_LIGHTCYAN:
				//VISIT: Redo this one
				pszColor = "<font color=\"Teal\">";
				break;
			case FLM_LIGHTRED:
				//VISIT: Redo this one
				pszColor = "<font color=\"Red\">";
				break;
			case FLM_LIGHTMAGENTA:
				//VISIT: Redo this one
				pszColor = "<font color=\"Fuchsia\">";
				break;
			case FLM_YELLOW:
				pszColor = "<font color=\"Yellow\">";
				break;
			case FLM_WHITE:
				pszColor = "<font color=\"White\">";
				break;
			default:
				flmAssert( 0);
				goto Exit;
		}

		// Output the color buffer

		outputStr( pszColor);
	}

Exit:

	return;
}

/****************************************************************************
Desc: Append a string to the output buffer.
****************************************************************************/
void F_QueryFormatter::appendString(
	const char *	pszStr,
	eColorType		eColor,
	FLMBOOL			bForceColor)
{
	char		szTmpBuf [80];
	FLMUINT	uiLen;

	// If we are doing a single line, and we have filled our buffer
	// with the maximum visible characters, we are done.

	if (m_bSingleLineOnly && m_uiVisibleChars == m_uiMaxChars)
	{
		goto Exit;
	}

	// Output a color change, if any

	changeColor( eColor, bForceColor);

	// Check for a NULL pointer - treat like empty string.

	if (pszStr)
	{

		// Output each character to the buffer.  Use the special encoding
		// format for reserved HTML characters.

		uiLen = 0;
		while (*pszStr)
		{

			// Can format any character in six characters in HTML - "&#xxx;"
			// xxx is the decimal digit for the character.
			// Therefore, if we have room for less than six characters in
			// the temporary buffer, output what we currently have in the
			// buffer.

			if (uiLen + 6 >= sizeof( szTmpBuf))
			{
				szTmpBuf [uiLen] = 0;
				outputStr( szTmpBuf);

				// Start filling the temporary buffer again.

				uiLen = 0;
			}

			// Check for reserved characters that need to be encoded.

			if ((int)(*pszStr) > 126 ||
				 *pszStr == '\"' ||
				 *pszStr == '&' ||
				 *pszStr == '<' ||
				 *pszStr == '>')
			{
				f_sprintf( &szTmpBuf [uiLen], "&#%d;", (int)(*pszStr));
				while (szTmpBuf [uiLen])
				{
					uiLen++;
				}
			}
			else
			{
				szTmpBuf [uiLen++] = *pszStr;
			}

			if (m_bSingleLineOnly)
			{

				// If we have reached the maximum visible characters,
				// quit outputting.

				if (++m_uiVisibleChars == m_uiMaxChars)
				{
					break;
				}
			}

			// Go to the next character

			pszStr++;
		}

		// If there is something in the temporary buffer, output it.

		if (uiLen)
		{
			szTmpBuf [uiLen] = 0;
			outputStr( szTmpBuf);
		}
	}

Exit:

	return;
}

/****************************************************************************
Desc: Output a newline.
****************************************************************************/
void F_QueryFormatter::newline( void)
{
	if (!m_bSingleLineOnly)
	{
		outputStr( "\n");

		// Reset so we start again.

		m_uiVisibleChars = 0;
	}
}

/****************************************************************************
Desc:	This routine indents the output.
****************************************************************************/
void F_QueryFormatter::outputIndent(
	FLMUINT	uiIndent
	)
{
	if (!m_bSingleLineOnly)
	{

		// Note: just use current color here.  It doesn't matter because
		// we are outputting spaces, and we cannot change the background
		// color.

		while (uiIndent)
		{
			outputStr( " ");
			uiIndent--;
		}
	}
}

/****************************************************************************
Desc:	This routine outputs an operator
****************************************************************************/
void F_QueryFormatter::outputOperator(
	QTYPES			eOperator,
	FLMBOOL			bEndLine
	)
{
	const char *	pszOperator;

	switch (eOperator)
	{
		case FLM_AND_OP:
			pszOperator = "AND";
			break;
		case FLM_OR_OP:
			pszOperator = "OR";
			break;
		case FLM_NOT_OP:
			pszOperator = "!";
			break;
		case FLM_EQ_OP:
			pszOperator = "==";
			break;
		case FLM_MATCH_OP:
			pszOperator = "MATCH";
			break;
		case FLM_MATCH_BEGIN_OP:
			pszOperator = "MATCHBEGIN";
			break;
		case FLM_MATCH_END_OP:
			pszOperator = "MATCHEND";
			break;
		case FLM_CONTAINS_OP:
			pszOperator = "CONTAINS";
			break;
		case FLM_NE_OP:
			pszOperator = "!=";
			break;
		case FLM_LT_OP:
			pszOperator = "<";
			break;
		case FLM_LE_OP:
			pszOperator = "<=";
			break;
		case FLM_GT_OP:
			pszOperator = ">";
			break;
		case FLM_GE_OP:
			pszOperator = ">=";
			break;
		case FLM_BITAND_OP:
			pszOperator = "&";
			break;
		case FLM_BITOR_OP:
			pszOperator = "|";
			break;
		case FLM_BITXOR_OP:
			pszOperator = "^";
			break;
		case FLM_MULT_OP:
			pszOperator = "*";
			break;
		case FLM_DIV_OP:
			pszOperator = "/";
			break;
		case FLM_MOD_OP:
			pszOperator = "%";
			break;
		case FLM_PLUS_OP:
			pszOperator = "+";
			break;
		case FLM_MINUS_OP:
		case FLM_NEG_OP:
			pszOperator = "-";
			break;
		case FLM_LPAREN_OP:
			pszOperator = "(";
			break;
		case FLM_RPAREN_OP:
			pszOperator = ")";
			break;
		default:
			pszOperator = "UNKNOWN";
			break;
	}

	appendString( pszOperator, Q_OPERATOR_COLOR);

	if (bEndLine && !m_bSingleLineOnly)
	{
		newline();
	}
}

/****************************************************************************
Desc:	This routine outputs a buffer of binary bytes as ASCII hex.
****************************************************************************/
void F_QueryFormatter::outputBinary(
	FLMBYTE *		pucBuf,
	FLMUINT			uiBufLen,
	eColorType		eValueColor)
{
	FLMUINT			uiLoop;
	FLMUINT			uiOffset;
	char				szBuf [128];
	FLMBYTE			ucChar;
	FLMUINT			uiRepeatCnt;
	FLMUINT			uiCharsNeeded;
	FLMBOOL			bFirstChar = TRUE;

	// Opening paren

	appendString( "(", eValueColor);

	// Output the word "<empty>" if the binary value is empty.
	// Otherwise output as hex values.

	if (!uiBufLen)
	{
		appendString( "<empty>", eValueColor);
	}
	else
	{

		uiLoop = 0;
		uiOffset = 0;
		while (uiLoop < uiBufLen)
		{

			// See how many there are of this byte.

			ucChar = *pucBuf;
			uiRepeatCnt = 1;
			uiLoop++;
			pucBuf++;
			while (uiLoop < uiBufLen && *pucBuf == ucChar)
			{
				uiLoop++;
				pucBuf++;
				uiRepeatCnt++;
			}

			// If this is the first character, we need 2 characters
			// for the HEX digits.  If not the first character,
			// we need 3 character - one leading space and 2 for hex digits.

			uiCharsNeeded = (FLMUINT)(bFirstChar
											  ? (FLMUINT)2
											  : (FLMUINT)3);

			// Calculate number of characters needed for
			// the ":cnt" after the hex digits if we have
			// the character repeated more than once.

			if (uiRepeatCnt > 1)
			{
				FLMUINT	uiTmp = uiRepeatCnt;
				while (uiTmp)
				{
					uiCharsNeeded++;
					uiTmp /= 10;
				}
				uiCharsNeeded++;	// Add one for the colon.
			}

			// Don't overflow the buffer.

			if (uiOffset >= sizeof( szBuf) - uiCharsNeeded)
			{
				szBuf [uiOffset] = 0;
				appendString( szBuf, eValueColor);
				uiOffset = 0;
			}

			// Put a space between bytes.

			if (!bFirstChar)
			{
				szBuf [uiOffset++] = ' ';
			}
			else
			{
				bFirstChar = FALSE;
			}
			formatHex( &szBuf [uiOffset], ucChar);
			uiOffset += 2;
			if (uiRepeatCnt > 1)
			{
				szBuf [uiOffset++] = ':';
				f_sprintf( &szBuf [uiOffset], "%u", (unsigned)uiRepeatCnt);
				while (szBuf [uiOffset])
				{
					uiOffset++;
				}
			}
		}

		// Output whatever has not been output so far.

		if (uiOffset)
		{
			szBuf [uiOffset] = 0;
			appendString( szBuf, eValueColor);
		}
	}

	// Closing paren

	appendString( ")", eValueColor);
}

/****************************************************************************
Desc:	This routine outputs text data.
****************************************************************************/
void F_QueryFormatter::outputText(
	FLMBYTE *		pucBuf,
	FLMUINT			uiBufLen,
	eColorType		eColor)
{
	FLMUINT			uiLoop;
	FLMUINT			uiOffset;
	FLMBYTE			ucChar;
	FLMUINT			uiObjType;
	FLMUINT			uiObjLength = 0;
	char				szBuf [128];

	uiOffset = 0;
	for (uiLoop = 0, uiOffset = 0;
		  uiLoop < uiBufLen;
		  pucBuf += uiObjLength, uiLoop += uiObjLength)
	{

		// Don't overflow the buffer - must have room for maximum bytes needed
		// to represent a character.

		if (uiOffset >= sizeof( szBuf) - 14)
		{
			szBuf [uiOffset] = 0;
			appendString( szBuf, eColor);
			uiOffset = 0;
		}

		ucChar = *pucBuf;
		uiObjType = flmTextObjType( ucChar);
		switch (uiObjType)
		{
			case ASCII_CHAR_CODE:  			// 0nnnnnnn
				uiObjLength = 1;

				// Character set zero is assumed.

				szBuf [uiOffset++] = (char)ucChar;
				break;
			case CHAR_SET_CODE:	  			// 10nnnnnn
				uiObjLength = 2;

				// Character set followed by character

				f_strcpy( &szBuf [uiOffset], "~[UC-0x");
				uiOffset += 7;
				formatHex( &szBuf [uiOffset], ucChar & (~CHAR_SET_MASK));
				formatHex( &szBuf [uiOffset + 2], *(pucBuf + 1));
				szBuf [uiOffset + 4] = ']';
				uiOffset += 5;
				break;
			case WHITE_SPACE_CODE:			// 110nnnnn
				uiObjLength = 1;
				szBuf [uiOffset++] = ' ';
				break;
			case EXT_CHAR_CODE:
				uiObjLength = 3;

				// Character set followed by character

				f_strcpy( &szBuf [uiOffset], "~[WP-0x");
				uiOffset += 7;
				formatHex( &szBuf [uiOffset], *(pucBuf + 1));
				formatHex( &szBuf [uiOffset + 2], *(pucBuf + 2));
				szBuf [uiOffset + 4] = ']';
				uiOffset += 5;
				break;
			case OEM_CODE:

				// OEM characters are always >= 128
				// Use character set zero to process them.

				uiObjLength = 2;
				szBuf [uiOffset++] = (char)(*(pucBuf + 1));
				break;
			case UNICODE_CODE:			// Unconvertable UNICODE code
				uiObjLength = 3;

				// Unicode character followed by unicode character set

				f_strcpy( &szBuf [uiOffset], "~[UC-0x");
				uiOffset += 7;
				formatHex( &szBuf [uiOffset], *(pucBuf + 1));
				formatHex( &szBuf [uiOffset + 2], *(pucBuf + 2));
				szBuf [uiOffset + 4] = ']';
				uiOffset += 5;
				break;
			default:

				// Should not happen

				flmAssert( 0);
				break;
		}
	}

	// Log whatever has not yet been logged.

	if (uiOffset)
	{
		szBuf [uiOffset] = 0;
		appendString( szBuf, eColor);
		uiOffset = 0;
	}
}

/****************************************************************************
Desc:	This routine logs the query criteria for a cursor.
****************************************************************************/
void F_QueryFormatter::outputPredicate(
	FLMUINT			uiIndent,
	FQNODE *			pQNode
	)
{
	FLMUINT	uiNestLevel = 0;
	QTYPES	eCurrentOp;
	char		szBuf [80];

	if (!m_bSingleLineOnly)
	{
		outputIndent( uiIndent);
	}
	else
	{
		appendString( " ");
	}

	// Traverse the tree.

	for (;;)
	{
		eColorType	eSaveCurrColor;

		eCurrentOp = GET_QNODE_TYPE( pQNode);
		if (IS_OP( eCurrentOp))
		{
			if (uiNestLevel)
			{
				outputOperator( FLM_LPAREN_OP, FALSE);
			}
			pQNode = pQNode->pChild;
			eCurrentOp = GET_QNODE_TYPE( pQNode);
			uiNestLevel++;
			continue;
		}

		eSaveCurrColor = m_eCurrColor;
		if (IS_VAL( eCurrentOp))
		{
			switch (eCurrentOp)
			{
				case FLM_BOOL_VAL:
					f_sprintf( szBuf, "%u", (unsigned)pQNode->pQAtom->val.uiBool);
					appendString( szBuf, Q_VALUE_COLOR);
					break;
				case FLM_REC_PTR_VAL:
				case FLM_UINT32_VAL:
					f_sprintf( szBuf, "%u", (unsigned)pQNode->pQAtom->val.ui32Val);
					appendString( szBuf, Q_VALUE_COLOR);
					break;
				case FLM_UINT64_VAL:
					f_sprintf( szBuf, "%I64u", pQNode->pQAtom->val.ui64Val);
					appendString( szBuf, Q_VALUE_COLOR);
					break;
				case FLM_INT32_VAL:
					f_sprintf( szBuf, "%d", (int)pQNode->pQAtom->val.i32Val);
					appendString( szBuf, Q_VALUE_COLOR);
					break;
				case FLM_INT64_VAL:
					f_sprintf( szBuf, "%I64d", pQNode->pQAtom->val.i64Val);
					appendString( szBuf, Q_VALUE_COLOR);
					break;
				case FLM_BINARY_VAL:
					appendString( "BINARY", Q_LABEL_COLOR);
					outputBinary( pQNode->pQAtom->val.pucBuf,
									pQNode->pQAtom->uiBufLen,
									Q_VALUE_COLOR);
					break;
				case FLM_TEXT_VAL:
					appendString( "\"", Q_VALUE_COLOR);
					outputText( pQNode->pQAtom->val.pucBuf,
											pQNode->pQAtom->uiBufLen, Q_VALUE_COLOR);
					appendString( "\"", Q_VALUE_COLOR);
					break;
				default:
					flmAssert( 0);
					break;
			}
		}
		else
		{
			FLMUINT *	puiFldPath = pQNode->pQAtom->val.QueryFld.puiFldPath;
			FLMUINT		uiCnt;

			flmAssert( IS_FIELD( eCurrentOp));
			appendString( "FLD:", Q_FIELD_COLOR);

			// Fields are from child to parent order - must count fields
			// and print out in parent to child order.

			uiCnt = 0;
			while (puiFldPath [uiCnt])
			{
				uiCnt++;
			}
			while (uiCnt)
			{
				uiCnt--;
				if (uiCnt)
				{
					f_sprintf( szBuf, "%u.", (unsigned)puiFldPath [uiCnt]);
				}
				else
				{
					f_sprintf( szBuf, "%u", (unsigned)puiFldPath [uiCnt]);
				}
				appendString( szBuf, Q_FIELD_COLOR);
			}
		}

		if (!uiNestLevel)
		{
			goto Exit;
		}

		// Go to the sibling, if any.

		while (!pQNode->pNextSib)
		{
			pQNode = pQNode->pParent;
			uiNestLevel--;
			if (!uiNestLevel)
			{
				goto Exit;	// Done with this predicate.
			}
			outputOperator( FLM_RPAREN_OP, FALSE);
		}

		// Have a sibling, log the operator.

		eCurrentOp = GET_QNODE_TYPE( pQNode->pParent);
		appendString( " ");
		outputOperator( eCurrentOp, FALSE);
		appendString( " ");
		pQNode = pQNode->pNextSib;
	}

Exit:

	if( !m_bSingleLineOnly)
	{
		newline();
	}
}

/****************************************************************************
Desc:	This routine outputs a subquery.
****************************************************************************/
void F_QueryFormatter::outputSubQuery(
	FLMUINT			uiIndent,
	QTYPES			eParentOp,
	CURSOR *			pReferenceCursor,
	SUBQUERY *		pSubQuery
	)
{
	char *		pszURL = NULL;
	FQNODE *		pQNode;
	QTYPES		eCurrentOp;
	QTYPES		eTmpParentOp;
	FLMBOOL		bIndentOptInfo = TRUE;

	if ((pQNode = pSubQuery->pTree) == NULL)
	{
		if (!m_bSingleLineOnly)
		{
			outputIndent( uiIndent);
		}

		outputOperator( FLM_LPAREN_OP, FALSE);
		appendString( "<empty>", Q_VALUE_COLOR);
		outputOperator( FLM_RPAREN_OP, TRUE);
		goto Output_Opt_Info;
	}

	// Traverse the tree.

	for (;;)
	{
		eCurrentOp = GET_QNODE_TYPE( pQNode);
		eTmpParentOp = (QTYPES)(pQNode->pParent
										? GET_QNODE_TYPE( pQNode->pParent)
										: eParentOp);
		if (eCurrentOp == FLM_AND_OP)
		{
			if (eTmpParentOp == FLM_OR_OP)
			{
				if (!m_bSingleLineOnly)
				{
					outputIndent( uiIndent);
				}

				outputOperator( FLM_LPAREN_OP, TRUE);
				uiIndent += 2;
				bIndentOptInfo = FALSE;
			}
			pQNode = pQNode->pChild;
		}
		else if (eCurrentOp == FLM_OR_OP)
		{
			if (eTmpParentOp == FLM_AND_OP)
			{
				if (!m_bSingleLineOnly)
				{
					outputIndent( uiIndent);
				}

				outputOperator( FLM_LPAREN_OP, TRUE);
				uiIndent += 2;
			}
			pQNode = pQNode->pChild;
		}
		else if (eCurrentOp == FLM_USER_PREDICATE)
		{
			HFCURSOR	hCursor = pQNode->pQAtom->val.pPredicate->getCursor();

			if (!m_bSingleLineOnly)
			{
				outputIndent( uiIndent);
			}

			outputOperator( FLM_LPAREN_OP, FALSE);

			if (hCursor == HFCURSOR_NULL)
			{
				appendString( " [EmbeddedPredicate] ", Q_LABEL_COLOR);
				outputOperator( FLM_RPAREN_OP, TRUE);
			}
			else
			{
				appendString( " [BeginEmbedded", Q_LABEL_COLOR);
				if (pSubQuery->OptInfo.eOptType == QOPT_USING_PREDICATE &&
					 pSubQuery->pPredicate == pQNode->pQAtom->val.pPredicate)
				{
					appendString( ", Optimized]", Q_LABEL_COLOR);
				}
				else
				{
					appendString( "]", Q_LABEL_COLOR);
				}
				bIndentOptInfo = FALSE;
				if (!m_bSingleLineOnly)
				{
					newline();
				}
				uiIndent += 2;
				outputQuery( uiIndent, pReferenceCursor, (CURSOR *)hCursor);
				uiIndent -= 2;
				if (!m_bSingleLineOnly)
				{
					outputIndent( uiIndent);
				}
				outputOperator( FLM_RPAREN_OP, FALSE);
				appendString( " [EndEmbedded]", Q_LABEL_COLOR);
				if (!m_bSingleLineOnly)
				{
					newline();
				}
			}
			goto Traverse_Up;
		}
		else
		{
			flmAssert( eCurrentOp != FLM_NOT_OP);
			if (!pQNode->pNextSib && !pQNode->pParent)
			{
				outputPredicate( uiIndent, pQNode);
			}
			else
			{
				outputPredicate( uiIndent + 2, pQNode);
				bIndentOptInfo = FALSE;
			}
Traverse_Up:
			while (!pQNode->pNextSib)
			{
				if ((pQNode = pQNode->pParent) == NULL)
				{
					goto Output_Opt_Info;
				}
				eCurrentOp = GET_QNODE_TYPE( pQNode);
				eTmpParentOp = (QTYPES)(pQNode->pParent
												? GET_QNODE_TYPE( pQNode->pParent)
												: eParentOp);
				if ((eCurrentOp == FLM_AND_OP && eTmpParentOp == FLM_OR_OP) ||
					 (eCurrentOp == FLM_OR_OP && eTmpParentOp == FLM_AND_OP))
				{
					flmAssert( uiIndent >= 2);
					uiIndent -= 2;
					if (!m_bSingleLineOnly)
					{
						outputIndent( uiIndent);
					}
					outputOperator( FLM_RPAREN_OP, TRUE);
				}
			}

			// Have a sibling.

			if (!m_bSingleLineOnly)
			{
				outputIndent( uiIndent);
			}
			outputOperator( eTmpParentOp, TRUE);
			pQNode = pQNode->pNextSib;
		}
	}

Output_Opt_Info:
	if (m_bSingleLineOnly)
	{
		goto Exit;
	}

	if (bIndentOptInfo)
	{
		uiIndent += 2;
	}

	outputIndent( uiIndent);

	// Create a URL for the sub-query statistics.

	if( RC_BAD( f_alloc( 340, &pszURL)))
	{
		goto Exit;
	}

	printAddress( (void *)pReferenceCursor, pszURL);
	printAddress( (void *)pSubQuery, &pszURL [20]);
	f_sprintf( &pszURL [40], "<A href=\"javascript:openPopup('%s/QueryStats?"
						"QueryHandle=%s&SubQuery=%s')\">",
						gv_FlmSysData.HttpConfigParms.pszURLString,
						pszURL, &pszURL [20]);
	
	outputStr( &pszURL [40]);
	appendString( "{OptInfo & Stats}", FLM_RED, TRUE);
	outputStr( "</A>");
	newline();

	if (bIndentOptInfo)
	{
		flmAssert( uiIndent >= 2);
		uiIndent -= 2;
	}

Exit:

	if (pszURL)
	{
		f_free( &pszURL);
	}
}

/****************************************************************************
Desc:	This routine formats the query criteria for a cursor and optionally
		outputs it.
****************************************************************************/
void F_QueryFormatter::formatQuery(
	HRequest *	pHRequest,
	F_WebPage *	pWebPage,
	CURSOR *		pCursor,
	FLMBOOL		bSingleLineOnly,
	FLMUINT		uiMaxChars
	)
{
	m_pHRequest = pHRequest;
	m_pWebPage = pWebPage;
	m_bSingleLineOnly = bSingleLineOnly;
	m_uiMaxChars = uiMaxChars;
	m_uiVisibleChars = 0;
	m_eCurrColor = FLM_CURRENT_COLOR;

	outputQuery( 0, pCursor, pCursor);
}

/****************************************************************************
Desc:	This routine formats the query criteria for a cursor and optionally
		outputs it.  This routine may be called recursively.
****************************************************************************/
void F_QueryFormatter::outputQuery(
	FLMUINT			uiIndent,
	CURSOR *			pReferenceCursor,
	CURSOR *			pCursor
	)
{
	SUBQUERY *		pSubQuery;
	QTYPES			eParentOp = (pCursor->pSubQueryList &&
										 pCursor->pSubQueryList->pNext)
										 ? FLM_OR_OP
										 : NO_TYPE;

	if (!uiIndent)
	{
		outputStr( "<PRE>");

		if (!m_bSingleLineOnly)
		{
			appendString( "Query Criteria: ", Q_LABEL_COLOR);
		}
		if (!pCursor->pSubQueryList)
		{
			appendString( "<Empty>", Q_VALUE_COLOR);
		}

		if (!m_bSingleLineOnly)
		{
			newline();
		}
		uiIndent += 2;
	}

	// Output each sub-query.

	pSubQuery = pCursor->pSubQueryList;
	while (pSubQuery)
	{
		outputSubQuery( uiIndent, eParentOp, pReferenceCursor, pSubQuery);
		if ((pSubQuery = pSubQuery->pNext) != NULL)
		{
			if (!m_bSingleLineOnly)
			{
				outputIndent( uiIndent);
			}
			else
			{
				appendString( " ");
			}

			outputOperator( FLM_OR_OP, TRUE);

			if (m_bSingleLineOnly)
			{
				appendString( " ");
			}
		}
	}

	// Output the last line of the query.

	if (!uiIndent)
	{
		if (!m_bSingleLineOnly)
		{
			newline();
		}

		outputStr( "</PRE>");
	}
}

/****************************************************************************
Desc:	Prints the web page for a single query.
****************************************************************************/
RCODE F_QueryPage::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE					rc = FERR_OK;
	HFCURSOR				hCursor;
	QUERY_HDR *			pQueryHdr;
	char					szHandle [100];
	F_QueryFormatter	qf;
	FLMBOOL				bMutexLocked = FALSE;

	printDocStart( "Query");
	popupFrame();

	if (RC_BAD(rc = ExtractParameter( uiNumParams, ppszParams, 
									"QueryHandle", sizeof( szHandle),
									szHandle)))
	{
		goto Exit;
	}
	hCursor = (HFCURSOR)f_atoud( szHandle);

	// Lock the mutex on the queries

	f_mutexLock( gv_FlmSysData.hQueryMutex);
	bMutexLocked = TRUE;

	// See if the hCursor is in the list.

	pQueryHdr = gv_FlmSysData.pNewestQuery;
	while (pQueryHdr && pQueryHdr->hCursor != hCursor)
	{
		pQueryHdr = pQueryHdr->pNext;
	}

	if (pQueryHdr)
	{
		// Output query

		qf.formatQuery( m_pHRequest, this, (CURSOR *)hCursor, FALSE, 0);
	}
	else
	{
		fnPrintf( m_pHRequest,
			"<center>Query is no longer in the table</center>\n");
	}

	// Unlock the mutex on the queries

	f_mutexUnlock( gv_FlmSysData.hQueryMutex);
	bMutexLocked = FALSE;

	printDocEnd();

Exit:

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hQueryMutex);
	}

	fnEmit();
	return( rc);
}

/****************************************************************************
Desc:	Find a sub-query within a cursor.
****************************************************************************/
FSTATIC FLMBOOL findSubQuery(
	CURSOR *		pCursor,
	SUBQUERY *	pSubQuery
	)
{
	SUBQUERY *	pTmpSubQuery;
	FLMBOOL		bFound = FALSE;
	HFCURSOR		hTmpCursor;
	FLMUINT		uiLoop;

	// First check the sub-query list.

	pTmpSubQuery = pCursor->pSubQueryList;
	while (pTmpSubQuery)
	{
		if (pTmpSubQuery == pSubQuery)
		{
			bFound = TRUE;
			goto Exit;
		}
		pTmpSubQuery = pTmpSubQuery->pNext;
	}

	// Search through the query's embedded predicates

	for (uiLoop = 0; uiLoop < pCursor->QTInfo.uiNumPredicates; uiLoop++)
	{
		if ((hTmpCursor = pCursor->QTInfo.ppPredicates [uiLoop]->getCursor()) !=
						HFCURSOR_NULL)
		{
			if (findSubQuery( (CURSOR *)hTmpCursor, pSubQuery))
			{
				bFound = TRUE;
				goto Exit;
			}
		}
	}

Exit:

	return( bFound);

}

/****************************************************************************
Desc:	Prints the web page for statistics of a query/subquery
****************************************************************************/
RCODE F_QueryStatsPage::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE					rc = FERR_OK;
	QUERY_HDR *			pQueryHdr;
	HFCURSOR				hCursor;
	SUBQUERY *			pSubQuery;
	char					szPtr [100];
	F_QueryFormatter	qf;
	FLMBOOL				bMutexLocked = FALSE;

	printDocStart( "Query Statistics", FALSE);

	// Get the query handle

	if (RC_BAD(rc = ExtractParameter( uiNumParams, ppszParams, 
									"QueryHandle", sizeof( szPtr), szPtr)))
	{
		goto Exit;
	}
	hCursor = (HFCURSOR)f_atoud( szPtr);

	// Get the sub-query pointer

	if (RC_BAD(rc = ExtractParameter( uiNumParams, ppszParams, 
									"SubQuery", sizeof( szPtr), szPtr)))
	{
		goto Exit;
	}
	pSubQuery = (SUBQUERY *)f_atoud( szPtr);

	// Lock the mutex on the queries

	f_mutexLock( gv_FlmSysData.hQueryMutex);
	bMutexLocked = TRUE;

	// See if the hCursor is in the list.

	pQueryHdr = gv_FlmSysData.pNewestQuery;
	while (pQueryHdr && pQueryHdr->hCursor != hCursor)
	{
		pQueryHdr = pQueryHdr->pNext;
	}

	if (pQueryHdr)
	{

		// Make sure we can find the sub-query.

		if (!findSubQuery( (CURSOR *)hCursor, pSubQuery))
		{
			fnPrintf( m_pHRequest,
				"<center>SubQuery is no longer in the query!</center>\n");
		}
		else
		{

			// Output subquery statistics

			qf.outputSubqueryStats( m_pHRequest, this, pSubQuery);
		}
	}
	else
	{
		fnPrintf( m_pHRequest,
			"<center>Query is no longer in the table</center>\n");
	}

	// Unlock the mutex on the queries

	f_mutexUnlock( gv_FlmSysData.hQueryMutex);
	bMutexLocked = FALSE;

	printDocEnd();

Exit:

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hQueryMutex);
	}

	fnEmit();
	return( rc);
}

/****************************************************************************
Desc:	This routine outputs a label in its own column.
****************************************************************************/
void F_QueryFormatter::outputLabel(
	const char *	pszLabel,
	eColorType		eLabelColor)
{
	// Label goes in a column by itself

	m_pWebPage->printTableDataStart();

	// Output the label

	appendString( pszLabel, eLabelColor, TRUE);

	// End the data column

	m_pWebPage->printTableDataEnd();
}

/****************************************************************************
Desc:	This routine outputs a row that has a label in column one and a
		string value in column two.
****************************************************************************/
void F_QueryFormatter::outputStringRow(
	const char *	pszLabel,
	const char *	pszValue,
	eColorType		eLabelColor,
	eColorType		eValueColor)
{
	// Start a new row in the table

	m_pWebPage->printTableRowStart( (++m_uiRowCount) & 0x00000001 ? TRUE : FALSE);

	// Label the row

	outputLabel( pszLabel, eLabelColor);

	// Start a new column for the string value

	m_pWebPage->printTableDataStart();

	// Output the value

	if (pszValue)
	{
		appendString( pszValue, eValueColor, TRUE);
	}

	// End the data column

	m_pWebPage->printTableDataEnd();

	// End the row

	m_pWebPage->printTableRowEnd();
}

/****************************************************************************
Desc:	This routine outputs a row that has a label in column one and a
		YES/NO value in column two.
****************************************************************************/
void F_QueryFormatter::outputYesNoRow(
	const char *	pszLabel,
	FLMBOOL			bYesNo,
	eColorType		eLabelColor,
	eColorType		eYesColor,
	eColorType		eNoColor)
{
	// Start a new row in the table

	m_pWebPage->printTableRowStart( (++m_uiRowCount) & 0x00000001 ? TRUE : FALSE);

	outputLabel( pszLabel, eLabelColor);

	// Start a new column for the YES/NO value

	m_pWebPage->printTableDataStart();

	if (bYesNo)
	{
		appendString( "YES", eYesColor, TRUE);
	}
	else
	{
		appendString( "NO", eNoColor, TRUE);
	}

	// End the data column

	m_pWebPage->printTableDataEnd();

	// End the row

	m_pWebPage->printTableRowEnd();
}

/****************************************************************************
Desc:	This routine outputs a row that has a label in column one and a
		UINT value in column two.
****************************************************************************/
void F_QueryFormatter::outputUINTRow(
	const char *	pszLabel,
	FLMUINT			uiValue,
	eColorType		eLabelColor,
	eColorType		eValueColor)
{
	char	szTmp [20];

	// Start a new row in the table

	m_pWebPage->printTableRowStart( (++m_uiRowCount) & 0x00000001 ? TRUE : FALSE);

	outputLabel( pszLabel, eLabelColor);

	// Start a new column for the YES/NO value

	m_pWebPage->printTableDataStart();

	f_sprintf( szTmp, "%u", (unsigned)uiValue);
	appendString( szTmp, eValueColor, TRUE);

	// End the data column

	m_pWebPage->printTableDataEnd();

	// End the row in the table.

	m_pWebPage->printTableRowEnd();
}

/****************************************************************************
Desc:	This routine outputs a row that has a label in column one and a
		binary value in column two.
****************************************************************************/
void F_QueryFormatter::outputBinaryRow(
	const char *			pszLabel,
	FLMBYTE *		pucValue,
	FLMUINT			uiValueLen,
	eColorType		eLabelColor,
	eColorType		eValueColor)
{
	// Start a new row in the table

	m_pWebPage->printTableRowStart();

	outputLabel( pszLabel, eLabelColor);

	// Start a new column for the binary value

	m_pWebPage->printTableDataStart();

	// Force the color to be output

	changeColor( eValueColor, TRUE);

	outputBinary( pucValue, uiValueLen, eValueColor);

	// End the data column

	m_pWebPage->printTableDataEnd();

	// End the row in the table.

	m_pWebPage->printTableRowEnd();
}

/****************************************************************************
Desc:	This routine formats the sub-query statistics.
****************************************************************************/
void F_QueryFormatter::outputSubqueryStats(
	HRequest *		pHRequest,
	F_WebPage *		pWebPage,
	SUBQUERY *		pSubQuery
	)
{
	FLMBYTE *	pucFromKey = NULL;
	FLMUINT		uiFromKeyLen;
	FLMBYTE *	pucUntilKey = NULL;
	FLMUINT		uiUntilKeyLen;
	FLMBOOL		bUntilKeyExclusive;

	m_pWebPage = pWebPage;
	m_pHRequest = pHRequest;
	m_bSingleLineOnly = FALSE;
	m_eCurrColor = FLM_CURRENT_COLOR;

	// Begin the table

	m_pWebPage->printTableStart( "Subquery Statistics", 2);

	// Output the statistics

	switch (pSubQuery->OptInfo.eOptType)
	{
		case QOPT_USING_INDEX:
			outputStringRow( "OPTIMIZATION",
									"Using Index",
									Q_LABEL_COLOR, Q_PARAM_COLOR);

			outputUINTRow( "Index",
									pSubQuery->OptInfo.uiIxNum,
									Q_LABEL_COLOR, Q_PARAM_COLOR);

			outputYesNoRow( "Key Match",
									pSubQuery->OptInfo.bDoKeyMatch);

			outputYesNoRow( "Record Match",
									pSubQuery->OptInfo.bDoRecMatch);

			pucFromKey = NULL;
			pucUntilKey = NULL;
			if (RC_OK( pSubQuery->pFSIndexCursor->getFirstLastKeys(
								&pucFromKey, &uiFromKeyLen,
								&pucUntilKey, &uiUntilKeyLen,
								&bUntilKeyExclusive)))
			{
				// Show the from key

				outputUINTRow( "From Key Length", uiFromKeyLen,
										Q_LABEL_COLOR, Q_PARAM_COLOR);

				outputBinaryRow( "From Key", pucFromKey,
										uiFromKeyLen, Q_LABEL_COLOR, Q_PARAM_COLOR);

				// Show the until key.

				outputUINTRow( "Until Key Length",
										uiUntilKeyLen,
										Q_LABEL_COLOR, Q_PARAM_COLOR);

				outputYesNoRow( "Until Key Exclusive",
										bUntilKeyExclusive);

				outputBinaryRow( "Until Key", pucUntilKey,
											uiUntilKeyLen, Q_LABEL_COLOR, Q_PARAM_COLOR);

				f_free( &pucFromKey);
				f_free( &pucUntilKey);
			}
			break;

		case QOPT_USING_PREDICATE:
			outputStringRow( "OPTIMIZATION",
									"Using Embedded Predicate",
									Q_LABEL_COLOR, Q_PARAM_COLOR);
			break;

		case QOPT_SINGLE_RECORD_READ:
			outputStringRow( "OPTIMIZATION",
									"Using Single Record Read",
									Q_LABEL_COLOR, Q_PARAM_COLOR);

			outputUINTRow( "DRN To Read",
								pSubQuery->OptInfo.uiDrn,
								Q_LABEL_COLOR, Q_PARAM_COLOR);
			break;

		case QOPT_PARTIAL_CONTAINER_SCAN:
			outputStringRow( "OPTIMIZATION",
									"Using Partial Container Scan",
									Q_LABEL_COLOR, Q_PARAM_COLOR);

//VISIT: Output from and until DRNs - need a method from
//pSubQuery->pFSDataCursor to return them.
			break;

		case QOPT_FULL_CONTAINER_SCAN:
			outputStringRow( "OPTIMIZATION",
									"Using Full Container Scan",
									Q_LABEL_COLOR, Q_PARAM_COLOR);
			break;

		default:
			outputStringRow( "OPTIMIZATION",
									"Using Unknown",
									Q_LABEL_COLOR, Q_PARAM_COLOR);
			break;
	}

	outputStringRow( "STATISTICS", "",
								Q_LABEL_COLOR, Q_PARAM_COLOR);

	outputUINTRow( "Container",
						pSubQuery->SQStatus.uiContainerNum,
						Q_LABEL_COLOR, Q_PARAM_COLOR);

	outputUINTRow( "Records Matched",
						pSubQuery->SQStatus.uiMatchedCnt,
						Q_LABEL_COLOR, Q_PARAM_COLOR);

	if (pSubQuery->SQStatus.uiNumRejectedByCallback)
	{
		outputUINTRow( "Rejected By Callback",
							pSubQuery->SQStatus.uiNumRejectedByCallback,
							Q_LABEL_COLOR, Q_PARAM_COLOR);
	}

	if (pSubQuery->SQStatus.uiDupsEliminated)
	{
		outputUINTRow( "Duplicates Eliminated",
							pSubQuery->SQStatus.uiDupsEliminated,
							Q_LABEL_COLOR, Q_PARAM_COLOR);
	}

	if (pSubQuery->SQStatus.uiKeysTraversed ||
		 pSubQuery->SQStatus.uiKeysRejected)
	{
		outputUINTRow( "Keys Traversed",
							pSubQuery->SQStatus.uiKeysTraversed,
							Q_LABEL_COLOR, Q_PARAM_COLOR);

		outputUINTRow( "Keys Rejected",
							pSubQuery->SQStatus.uiKeysRejected,
							Q_LABEL_COLOR, Q_PARAM_COLOR);
	}

	if (pSubQuery->SQStatus.uiRefsTraversed ||
		 pSubQuery->SQStatus.uiRefsRejected)
	{
		outputUINTRow( "References Traversed",
							pSubQuery->SQStatus.uiRefsTraversed,
							Q_LABEL_COLOR, Q_PARAM_COLOR);

		outputUINTRow( "References Rejected",
							pSubQuery->SQStatus.uiRefsRejected,
							Q_LABEL_COLOR, Q_PARAM_COLOR);
	}

	if (pSubQuery->SQStatus.uiRecsFetchedForEval ||
		 pSubQuery->SQStatus.uiRecsRejected ||
		 pSubQuery->SQStatus.uiRecsNotFound)
	{
		outputUINTRow( "Records Fetched",
							pSubQuery->SQStatus.uiRecsFetchedForEval,
							Q_LABEL_COLOR, Q_PARAM_COLOR);

		outputUINTRow( "Records Rejected",
							pSubQuery->SQStatus.uiRecsRejected,
							Q_LABEL_COLOR, Q_PARAM_COLOR);

		outputUINTRow( "Records Not Found",
							pSubQuery->SQStatus.uiRecsNotFound,
							Q_LABEL_COLOR, Q_PARAM_COLOR);
	}

	// End the table

	m_uiRowCount = 0;
	m_pWebPage->printTableEnd();

	// Free allocated memory

	if( pucFromKey)
	{
		f_free( &pucFromKey);
	}

	if( pucUntilKey)
	{
		f_free( &pucUntilKey);
	}
}
