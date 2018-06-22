//-------------------------------------------------------------------------
// Desc:	Query logging
// Tabs:	3
//
// Copyright (c) 2001, 2003, 2005-2007 Novell, Inc. All Rights Reserved.
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

FSTATIC void flmLogIndent(
	IF_LogMessageClient *	pLogMsg,
	FLMUINT						uiIndent);

FSTATIC void flmLogOperator(
	IF_LogMessageClient *	pLogMsg,
	QTYPES						eOperator,
	FLMBOOL						bEndLine);

FSTATIC void flmLogBinary(
	IF_LogMessageClient *	pLogMsg,
	FLMBYTE *					pucBuf,
	FLMUINT						uiBufLen);

FSTATIC void flmLogText(
	IF_LogMessageClient *	pLogMsg,
	const char *				pucBuf,
	FLMUINT						uiBufLen);

FSTATIC void flmLogPredicate(
	IF_LogMessageClient *	pLogMsg,
	FLMUINT						uiIndent,
	FQNODE *						pQNode);

FSTATIC void flmLogSubQuery(
	IF_LogMessageClient *	pLogMsg,
	FLMUINT						uiIndent,
	QTYPES						eParentOp,
	SUBQUERY *					pSubQuery);

/****************************************************************************
Desc:	This routine indents in the log.
****************************************************************************/
FSTATIC void flmLogIndent(
	IF_LogMessageClient *	pLogMsg,
	FLMUINT						uiIndent)
{
	char		szIndent [100];

	if (uiIndent)
	{
		f_memset( szIndent, ' ', uiIndent);
		szIndent [uiIndent] = 0;
		pLogMsg->changeColor( FLM_LIGHTGRAY, FLM_BLACK);
		pLogMsg->appendString( szIndent);
	}
}

/****************************************************************************
Desc:	This routine logs an operator
****************************************************************************/
FSTATIC void flmLogOperator(
	IF_LogMessageClient *	pLogMsg,
	QTYPES						eOperator,
	FLMBOOL						bEndLine)
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
	pLogMsg->pushForegroundColor();
	pLogMsg->pushBackgroundColor();
	if (eOperator == FLM_LPAREN_OP || eOperator == FLM_RPAREN_OP)
	{
		pLogMsg->changeColor( FLM_CYAN, FLM_BLACK);
	}
	else
	{
		pLogMsg->changeColor( FLM_BLUE, FLM_LIGHTGRAY);
	}
	
	pLogMsg->appendString( pszOperator);
	pLogMsg->popForegroundColor();
	pLogMsg->popBackgroundColor();
	
	if( bEndLine)
	{
		pLogMsg->newline();
	}
}

/****************************************************************************
Desc:	Format a byte into two HEX characters.
****************************************************************************/
FINLINE void flmFormatByteToHex(
	char *		pszBuf,
	FLMBYTE		ucChar)
{
	FLMUINT	uiNibble = ((FLMUINT)ucChar >> 4) & 0xF;

	*pszBuf++ = (FLMBYTE)(uiNibble <= 9
								? (FLMBYTE)(uiNibble + '0')
								: (FLMBYTE)(uiNibble - 10 + 'A'));
	uiNibble = (FLMUINT)ucChar & 0xF;
	*pszBuf = (FLMBYTE)(uiNibble <= 9
								? (FLMBYTE)(uiNibble + '0')
								: (FLMBYTE)(uiNibble - 10 + 'A'));
}

/****************************************************************************
Desc:	This routine logs a buffer of binary bytes as ASCII hex.
****************************************************************************/
FSTATIC void flmLogBinary(
	IF_LogMessageClient *	pLogMsg,
	FLMBYTE *					pucBuf,
	FLMUINT						uiBufLen)
{
	FLMUINT		uiLoop;
	FLMUINT		uiOffset;
	char			szBuf [128];
	FLMBYTE		ucChar;
	FLMUINT		uiRepeatCnt;
	FLMUINT		uiCharsNeeded;
	FLMBOOL		bFirstChar = TRUE;

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
			pLogMsg->appendString( szBuf);
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
		
		flmFormatByteToHex( &szBuf [uiOffset], ucChar);
		uiOffset += 2;
		
		if (uiRepeatCnt > 1)
		{
			szBuf [uiOffset++] = ':';
			f_sprintf( (char *)&szBuf [uiOffset], "%u", (unsigned)uiRepeatCnt);
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
		pLogMsg->appendString( szBuf);
	}
}

/****************************************************************************
Desc:	This routine logs text data.
****************************************************************************/
FSTATIC void flmLogText(
	IF_LogMessageClient *	pLogMsg,
	const char *				pucBuf,
	FLMUINT						uiBufLen)
{
	FLMUINT			uiLoop;
	FLMUINT			uiOffset;
	FLMBYTE			ucChar;
	FLMUINT			uiObjType;
	FLMUINT			uiObjLength = 0;
	char				szBuf[ 128];

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
			pLogMsg->appendString( szBuf);
			uiOffset = 0;
		}

		ucChar = *pucBuf;
		uiObjType = flmTextObjType( ucChar);
		switch (uiObjType)
		{
			case ASCII_CHAR_CODE:  			// 0nnnnnnn
				uiObjLength = 1;

				// Character set zero is assumed.

				szBuf [uiOffset++] = ucChar;
				break;
			case CHAR_SET_CODE:	  			// 10nnnnnn
				uiObjLength = 2;

				// Character set followed by character

				f_strcpy( &szBuf [uiOffset], "~[UC-0x");
				uiOffset += 7;
				flmFormatByteToHex( &szBuf [uiOffset], ucChar & (~CHAR_SET_MASK));
				flmFormatByteToHex( &szBuf [uiOffset + 2], *(pucBuf + 1));
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
				flmFormatByteToHex( &szBuf [uiOffset], *(pucBuf + 1));
				flmFormatByteToHex( &szBuf [uiOffset + 2], *(pucBuf + 2));
				szBuf [uiOffset + 4] = ']';
				uiOffset += 5;
				break;
			case OEM_CODE:

				// OEM characters are always >= 128
				// Use character set zero to process them.

				uiObjLength = 2;
				szBuf [uiOffset++] = *(pucBuf + 1);
				break;
			case UNICODE_CODE:			// Unconvertable UNICODE code
				uiObjLength = 3;

				// Unicode character followed by unicode character set

				f_strcpy( &szBuf [uiOffset], "~[UC-0x");
				uiOffset += 7;
				flmFormatByteToHex( &szBuf [uiOffset], *(pucBuf + 1));
				flmFormatByteToHex( &szBuf [uiOffset + 2], *(pucBuf + 2));
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
		pLogMsg->appendString( szBuf);
		uiOffset = 0;
	}
}

/****************************************************************************
Desc:	This routine logs the query criteria for a cursor.
****************************************************************************/
FSTATIC void flmLogPredicate(
	IF_LogMessageClient *	pLogMsg,
	FLMUINT						uiIndent,
	FQNODE *						pQNode)
{
	FLMUINT	uiNestLevel = 0;
	QTYPES	eCurrentOp;

	flmLogIndent( pLogMsg, uiIndent);

	// Traverse the tree.

	for (;;)
	{
		eCurrentOp = GET_QNODE_TYPE( pQNode);
		if (IS_OP( eCurrentOp))
		{
			if (uiNestLevel)
			{
				flmLogOperator( pLogMsg, FLM_LPAREN_OP, FALSE);
			}
			pQNode = pQNode->pChild;
			eCurrentOp = GET_QNODE_TYPE( pQNode);
			uiNestLevel++;
			continue;
		}

		if (IS_VAL( eCurrentOp))
		{
			pLogMsg->changeColor( FLM_WHITE, FLM_BLACK);
			switch (eCurrentOp)
			{
				case FLM_BOOL_VAL:
					f_logPrintf( pLogMsg, "%u",
						(unsigned)pQNode->pQAtom->val.uiBool);
					break;
				case FLM_REC_PTR_VAL:
				case FLM_UINT32_VAL:
					f_logPrintf( pLogMsg, "%u",
						(unsigned)pQNode->pQAtom->val.ui32Val);
					break;
				case FLM_UINT64_VAL:
					f_logPrintf( pLogMsg, "%I64u",
						pQNode->pQAtom->val.ui64Val);
					break;
				case FLM_INT32_VAL:
					f_logPrintf( pLogMsg, "%d",
						(int)pQNode->pQAtom->val.i32Val);
					break;
				case FLM_INT64_VAL:
					f_logPrintf( pLogMsg, "%I64d",
						pQNode->pQAtom->val.i64Val);
					break;
				case FLM_BINARY_VAL:
					pLogMsg->appendString( "BINARY(");
					flmLogBinary( pLogMsg, pQNode->pQAtom->val.pucBuf,
									 pQNode->pQAtom->uiBufLen);
					pLogMsg->appendString( ")");
					break;
				case FLM_TEXT_VAL:
					pLogMsg->appendString( "\"");
					flmLogText( pLogMsg, (const char *)pQNode->pQAtom->val.pucBuf,
												pQNode->pQAtom->uiBufLen);
					pLogMsg->appendString( "\"");
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
			pLogMsg->changeColor( FLM_YELLOW, FLM_BLACK);
			pLogMsg->appendString( "FLD:");

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
					f_logPrintf( pLogMsg, "%u.", (unsigned)puiFldPath [uiCnt]);
				}
				else
				{
					f_logPrintf( pLogMsg, "%u", (unsigned)puiFldPath [uiCnt]);
				}
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
			flmLogOperator( pLogMsg, FLM_RPAREN_OP, FALSE);
		}

		// Have a sibling, log the operator.

		eCurrentOp = GET_QNODE_TYPE( pQNode->pParent);
		pLogMsg->appendString( " ");
		flmLogOperator( pLogMsg, eCurrentOp, FALSE);
		pLogMsg->appendString( " ");
		pQNode = pQNode->pNextSib;
	}
Exit:
	pLogMsg->newline();
	return;
}

/****************************************************************************
Desc:	This routine logs the query criteria for a cursor.
****************************************************************************/
FSTATIC void flmLogSubQuery(
	IF_LogMessageClient *	pLogMsg,
	FLMUINT						uiIndent,
	QTYPES						eParentOp,
	SUBQUERY *					pSubQuery)
{
	FQNODE *		pQNode;
	QTYPES		eCurrentOp;
	QTYPES		eTmpParentOp;
	FLMBYTE *	pucFromKey;
	FLMUINT		uiFromKeyLen;
	FLMBYTE *	pucUntilKey;
	FLMUINT		uiUntilKeyLen;
	FLMBOOL		bUntilKeyExclusive;
	FLMBOOL		bIndentOptInfo = TRUE;

	if ((pQNode = pSubQuery->pTree) == NULL)
	{
		flmLogIndent( pLogMsg, uiIndent);
		flmLogOperator( pLogMsg, FLM_LPAREN_OP, FALSE);
		pLogMsg->changeColor( FLM_WHITE, FLM_BLACK);
		pLogMsg->appendString( "<empty>");
		flmLogOperator( pLogMsg, FLM_RPAREN_OP, TRUE);
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
				flmLogIndent( pLogMsg, uiIndent);
				flmLogOperator( pLogMsg, FLM_LPAREN_OP, TRUE);
				uiIndent += 2;
				bIndentOptInfo = FALSE;
			}
			pQNode = pQNode->pChild;
		}
		else if (eCurrentOp == FLM_OR_OP)
		{
			if (eTmpParentOp == FLM_AND_OP)
			{
				flmLogIndent( pLogMsg, uiIndent);
				flmLogOperator( pLogMsg, FLM_LPAREN_OP, TRUE);
				uiIndent += 2;
			}
			pQNode = pQNode->pChild;
		}
		else if (eCurrentOp == FLM_USER_PREDICATE)
		{
			HFCURSOR	hCursor = pQNode->pQAtom->val.pPredicate->getCursor();

			flmLogIndent( pLogMsg, uiIndent);
			flmLogOperator( pLogMsg, FLM_LPAREN_OP, FALSE);
			if (hCursor == HFCURSOR_NULL)
			{
				pLogMsg->changeColor( FLM_WHITE, FLM_BLACK);
				pLogMsg->appendString( " [EmbeddedPredicate] ");
				flmLogOperator( pLogMsg, FLM_RPAREN_OP, TRUE);
			}
			else
			{
				pLogMsg->changeColor( FLM_LIGHTGRAY, FLM_BLACK);
				pLogMsg->appendString( " [BeginEmbedded");
				if (pSubQuery->OptInfo.eOptType == QOPT_USING_PREDICATE &&
					 pSubQuery->pPredicate == pQNode->pQAtom->val.pPredicate)
				{
					pLogMsg->appendString( ", Optimized]");
				}
				else
				{
					pLogMsg->appendString( "]");
				}
				bIndentOptInfo = FALSE;
				pLogMsg->newline();
				uiIndent += 2;
				flmLogQuery( pLogMsg, uiIndent, (CURSOR *)hCursor);
				uiIndent -= 2;
				flmLogIndent( pLogMsg, uiIndent);
				flmLogOperator( pLogMsg, FLM_RPAREN_OP, FALSE);
				pLogMsg->changeColor( FLM_LIGHTGRAY, FLM_BLACK);
				pLogMsg->appendString( " [EndEmbedded]");
				pLogMsg->newline();
			}
			goto Traverse_Up;
		}
		else
		{
			flmAssert( eCurrentOp != FLM_NOT_OP);
			if (!pQNode->pNextSib && !pQNode->pParent)
			{
				flmLogPredicate( pLogMsg, uiIndent, pQNode);
			}
			else
			{
				flmLogPredicate( pLogMsg, uiIndent + 2, pQNode);
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
					flmLogIndent( pLogMsg, uiIndent);
					flmLogOperator( pLogMsg, FLM_RPAREN_OP, TRUE);
				}
			}

			// Have a sibling.

			flmLogIndent( pLogMsg, uiIndent);
			flmLogOperator( pLogMsg, eTmpParentOp, TRUE);
			pQNode = pQNode->pNextSib;
		}
	}
Output_Opt_Info:
	if (bIndentOptInfo)
	{
		uiIndent += 2;
	}
	flmLogIndent( pLogMsg, uiIndent);
	pLogMsg->appendString( "{OptInfo: ");

	switch (pSubQuery->OptInfo.eOptType)
	{
		case QOPT_USING_INDEX:
			f_logPrintf( pLogMsg, F_FOREWHITE "UsingIX=" F_FOREYELLOW "%u",
				pSubQuery->OptInfo.uiIxNum);
			f_logPrintf( pLogMsg, F_FORELIGHTGRAY ", KeyMatch=");
			if (pSubQuery->OptInfo.bDoKeyMatch)
			{
				f_logPrintf( pLogMsg, F_FOREGREEN "YES");
			}
			else
			{
				f_logPrintf( pLogMsg, F_FORERED "NO");
			}

			f_logPrintf( pLogMsg, F_FORELIGHTGRAY ", RecMatch=");
			if (pSubQuery->OptInfo.bDoRecMatch)
			{
				f_logPrintf( pLogMsg, F_FOREGREEN "YES");
			}
			else
			{
				f_logPrintf( pLogMsg, F_FORERED "NO");
			}

			pucFromKey = NULL;
			pucUntilKey = NULL;
			if (RC_OK( pSubQuery->pFSIndexCursor->getFirstLastKeys(
								&pucFromKey, &uiFromKeyLen,
								&pucUntilKey, &uiUntilKeyLen,
								&bUntilKeyExclusive)))
			{

				// Show the from key

				f_logPrintf( pLogMsg,
					F_FORELIGHTGRAY ", FromKeyLen=" F_FOREYELLOW "%u"
					F_FORELIGHTGRAY ", FromKey=(",
					uiFromKeyLen);
				if (uiFromKeyLen)
				{
					pLogMsg->changeColor( FLM_YELLOW, FLM_BLACK);
					flmLogBinary( pLogMsg, pucFromKey, uiFromKeyLen);
				}
				else
				{
					f_logPrintf( pLogMsg, F_FOREYELLOW "<empty>");
				}
				f_logPrintf( pLogMsg, F_FORELIGHTGRAY ")");

				// Show the until key.

				f_logPrintf( pLogMsg,
					F_FORELIGHTGRAY ", UntilKeyLen=" F_FOREYELLOW "%u"
					F_FORELIGHTGRAY ", UntilExcl=" F_FOREYELLOW "%s"
					F_FORELIGHTGRAY ", UntilKey=(",
					uiUntilKeyLen,
					(bUntilKeyExclusive
								 ? "Yes"
								 : "No"));
				if (uiUntilKeyLen)
				{
					pLogMsg->changeColor( FLM_YELLOW, FLM_BLACK);
					flmLogBinary( pLogMsg, pucUntilKey, uiUntilKeyLen);
				}
				else
				{
					f_logPrintf( pLogMsg, F_FOREYELLOW "<empty>");
				}
				f_logPrintf( pLogMsg, F_FORELIGHTGRAY ")");
				f_free( &pucFromKey);
				f_free( &pucUntilKey);
			}
			break;

		case QOPT_USING_PREDICATE:
			f_logPrintf( pLogMsg, F_FOREWHITE "Using Embedded Predicate");
			break;

		case QOPT_SINGLE_RECORD_READ:
			f_logPrintf( pLogMsg, F_FOREWHITE "Single Record Read, DRN: "
				F_FOREYELLOW "%u", pSubQuery->OptInfo.uiDrn);
			break;

		case QOPT_PARTIAL_CONTAINER_SCAN:
			f_logPrintf( pLogMsg, F_FOREWHITE "Partial Container Scan");
//VISIT: Output from and until DRNs - need a method from
//pSubQuery->pFSDataCursor to return them.
			break;

		case QOPT_FULL_CONTAINER_SCAN:
			f_logPrintf( pLogMsg, F_FOREWHITE "Full Container Scan");
			break;

		default:
			f_logPrintf( pLogMsg, F_FOREWHITE "Unknown optimization");
			break;
	}
	f_logPrintf( pLogMsg, F_FORELIGHTGRAY "}\n");

	flmLogIndent( pLogMsg, uiIndent);
	pLogMsg->appendString( "{Stats: ");
	f_logPrintf( pLogMsg, F_FORELIGHTGRAY "Container=" F_FOREWHITE "%u",
		(unsigned)pSubQuery->SQStatus.uiContainerNum);

	f_logPrintf( pLogMsg, F_FORELIGHTGRAY ", Matched=" F_FOREWHITE "%u",
		(unsigned)pSubQuery->SQStatus.uiMatchedCnt);

	if (pSubQuery->SQStatus.uiNumRejectedByCallback)
	{
		f_logPrintf( pLogMsg, F_FORELIGHTGRAY ", CallbackRejected=" F_FOREWHITE "%u",
			(unsigned)pSubQuery->SQStatus.uiNumRejectedByCallback);
	}

	if (pSubQuery->SQStatus.uiDupsEliminated)
	{
		f_logPrintf( pLogMsg, F_FORELIGHTGRAY ", DupsElim=" F_FOREWHITE "%u",
			(unsigned)pSubQuery->SQStatus.uiDupsEliminated);
	}

	if (pSubQuery->SQStatus.uiKeysTraversed ||
		 pSubQuery->SQStatus.uiKeysRejected)
	{
		f_logPrintf( pLogMsg, F_FORELIGHTGRAY ", KeysFailed=" F_FOREWHITE "%u of %u",
			(unsigned)pSubQuery->SQStatus.uiKeysRejected,
			(unsigned)pSubQuery->SQStatus.uiKeysTraversed);
	}

	if (pSubQuery->SQStatus.uiRefsTraversed ||
		 pSubQuery->SQStatus.uiRefsRejected)
	{
		f_logPrintf( pLogMsg, F_FORELIGHTGRAY ", RefsFailed=" F_FOREWHITE "%u of %u",
			(unsigned)pSubQuery->SQStatus.uiRefsRejected,
			(unsigned)pSubQuery->SQStatus.uiRefsTraversed);
	}

	if (pSubQuery->SQStatus.uiRecsFetchedForEval ||
		 pSubQuery->SQStatus.uiRecsRejected ||
		 pSubQuery->SQStatus.uiRecsNotFound)
	{
		f_logPrintf( pLogMsg, F_FORELIGHTGRAY ", RecsFetched=" F_FOREWHITE "%u"
									  F_FORELIGHTGRAY ", RecsRejected=" F_FOREWHITE "%u"
									  F_FORELIGHTGRAY ", RecsNotFound=" F_FOREWHITE "%u",
			(unsigned)pSubQuery->SQStatus.uiRecsFetchedForEval,
			(unsigned)pSubQuery->SQStatus.uiRecsRejected,
			(unsigned)pSubQuery->SQStatus.uiRecsNotFound);
	}

	f_logPrintf( pLogMsg, F_FORELIGHTGRAY "}\n");

	if (bIndentOptInfo)
	{
		flmAssert( uiIndent >= 2);
		uiIndent -= 2;
	}
}

/****************************************************************************
Desc:	This routine logs the query criteria for a cursor.
****************************************************************************/
void flmLogQuery(
	IF_LogMessageClient *	pLogMsg,
	FLMUINT						uiIndent,
	CURSOR *						pCursor)
{
	SUBQUERY *	pSubQuery;
	QTYPES		eParentOp = (pCursor->pSubQueryList &&
									 pCursor->pSubQueryList->pNext)
									 ? FLM_OR_OP
									 : NO_TYPE;

	if (!uiIndent)
	{
		pLogMsg->changeColor( FLM_LIGHTGRAY, FLM_BLACK);
		pLogMsg->appendString( "QUERY CRITERIA:");
		if (!pCursor->pSubQueryList)
		{
			pLogMsg->appendString( " <NO CRITERIA>");
		}
		pLogMsg->newline();
		uiIndent += 2;
	}

	// Output each sub-query.

	pSubQuery = pCursor->pSubQueryList;
	while (pSubQuery)
	{
		flmLogSubQuery( pLogMsg, uiIndent, eParentOp, pSubQuery);
		if ((pSubQuery = pSubQuery->pNext) != NULL)
		{
			flmLogIndent( pLogMsg, uiIndent);
			flmLogOperator( pLogMsg, FLM_OR_OP, TRUE);
		}
	}
}
