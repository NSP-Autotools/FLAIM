//------------------------------------------------------------------------------
// Desc:	This module contains routines for parsing and executing SQL statements.
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

/****************************************************************************
Desc:		Constructor
****************************************************************************/
SQLStatement::SQLStatement()
{
	m_fnStatus = NULL;
	m_pvCallbackData = NULL;
	m_tmpPool.poolInit( 4096);
	m_pucCurrLineBuf = NULL;
	m_uiCurrLineBufMaxBytes = 0;
	m_pXml = NULL;
	m_pConnection = NULL;
	m_pNextInConnection = NULL;
	m_pPrevInConnection = NULL;
	resetStatement();
}

/****************************************************************************
Desc:		Destructor
****************************************************************************/
SQLStatement::~SQLStatement()
{
	resetStatement();

	if( m_pucCurrLineBuf)
	{
		f_free( &m_pucCurrLineBuf);
	}

	// Better not be associated with a connection at this point.
	
	flmAssert( !m_pConnection);

	m_tmpPool.poolFree();
}

/****************************************************************************
Desc:		Resets member variables so the object can be reused
****************************************************************************/
void SQLStatement::resetStatement( void)
{
	m_uiCurrLineNum = 0;
	m_uiCurrLineOffset = 0;
	m_ucUngetByte = 0;
	m_uiCurrLineFilePos = 0;
	m_uiCurrLineBytes = 0;
	m_pStream = NULL;
	m_uiFlags = 0;
	m_pDb = NULL;
	if (m_pXml)
	{
		m_pXml->Release();
		m_pXml = NULL;
	}
	f_memset( &m_sqlStats, 0, sizeof( SQL_STATS));

	m_tmpPool.poolReset( NULL);
}

/****************************************************************************
Desc:	Initializes the SQL statement object (allocates buffers, etc.)
****************************************************************************/
RCODE SQLStatement::setupStatement( void)
{
	RCODE			rc = NE_SFLM_OK;

	resetStatement();

	if (RC_BAD( rc = FlmGetXMLObject( &m_pXml)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	See if the next token in the stream is the specified keyword.
****************************************************************************/
RCODE SQLStatement::getToken(
	char *		pszToken,
	FLMUINT		uiTokenBufSize,
	FLMBOOL		bEofOK,
	FLMUINT *	puiTokenLineOffset,
	FLMUINT *	puiTokenLen)
{
	RCODE		rc = NE_SFLM_OK;
	FLMUINT	uiOffset;
	char		cChar;
	char *	pszTokenStart = pszToken;
	
	// Always leave room for a null terminating character
	
	uiTokenBufSize--;
	
	// Token buffer must hold at least one character and a null terminating
	// character!
	
	if (!uiTokenBufSize)
	{
		rc = RC_SET( NE_SFLM_BUFFER_OVERFLOW);
		goto Exit;
	}
	
	// Skip any whitespace preceding the token
	
	if (RC_BAD( rc = skipWhitespace( FALSE)))
	{
		if (rc == NE_SFLM_EOF_HIT)
		{
			if (!bEofOK)
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset,
						SQL_ERR_UNEXPECTED_EOF,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
			}
		}
		goto Exit;
	}
	
	// If the first character is alphanumeric, we need to parse until
	// we hit a non-alphanumeric character.

	*puiTokenLineOffset = uiOffset = m_uiCurrLineOffset;	
	cChar = (char)m_pucCurrLineBuf [uiOffset];
	
	if ((cChar >= 'a' && cChar <= 'z') ||
		 (cChar >= 'A' && cChar <= 'Z') ||
		 (cChar >= '0' && cChar <= '9') ||
		 cChar == '_')
	{
		*pszToken++ = cChar;
		uiTokenBufSize--;
		uiOffset++;
		while (uiOffset < m_uiCurrLineBytes)
		{
			cChar = (char)m_pucCurrLineBuf [uiOffset];
			if ((cChar >= 'a' && cChar <= 'z') ||
				 (cChar >= 'A' && cChar <= 'Z') ||
				 (cChar >= '0' && cChar <= '9') ||
				 cChar == '_')
			{
				if (!uiTokenBufSize)
				{
					rc = RC_SET( NE_SFLM_BUFFER_OVERFLOW);
					goto Exit;
				}
				*pszToken++ = cChar;
				uiTokenBufSize--;
				uiOffset++;
			}
			else
			{
				break;
			}
		}
		*pszToken = 0;
		m_uiCurrLineOffset = uiOffset;
	}
	else
	{
		switch (cChar)
		{
			case ',':
			case '(':
			case ')':
			case '[':
			case ']':
			case '{':
			case '}':
			case '$':
			case '@':
			case '.':
			case ':':
			case ';':
			case '"':
			case '\'':
			case '?':
			case '#':
				*pszToken++ = cChar;
				*pszToken = 0;
				m_uiCurrLineOffset = uiOffset + 1;
				break;
			case '>':
			case '<':
			case '+':
			case '-':
			case '*':
			case '/':
			case '=':
			case '!':
			case '%':
			case '^':
				*pszToken++ = cChar;
				if (uiOffset + 1 < m_uiCurrLineBytes &&
					 m_pucCurrLineBuf [uiOffset + 1] == '=')
				{
					if (uiTokenBufSize < 2)
					{
						rc = RC_SET( NE_SFLM_BUFFER_OVERFLOW);
						goto Exit;
					}
					*pszToken++ = '=';
					m_uiCurrLineOffset = uiOffset + 2;
				}
				else
				{
					m_uiCurrLineOffset = uiOffset + 1;
				}
				*pszToken = 0;
				break;
			case '|':
				*pszToken++ = cChar;
				if (uiOffset + 1 < m_uiCurrLineBytes &&
					 (m_pucCurrLineBuf [uiOffset + 1] == '=' ||
					  m_pucCurrLineBuf [uiOffset + 1] == '|'))
				{
					if (uiTokenBufSize < 2)
					{
						rc = RC_SET( NE_SFLM_BUFFER_OVERFLOW);
						goto Exit;
					}
					*pszToken++ = (char)m_pucCurrLineBuf [uiOffset + 1];
					m_uiCurrLineOffset = uiOffset + 2;
				}
				else
				{
					m_uiCurrLineOffset = uiOffset + 1;
				}
				*pszToken = 0;
				break;
			case '&':
				*pszToken++ = cChar;
				if (uiOffset + 1 < m_uiCurrLineBytes &&
					 (m_pucCurrLineBuf [uiOffset + 1] == '=' ||
					  m_pucCurrLineBuf [uiOffset + 1] == '&'))
				{
					if (uiTokenBufSize < 2)
					{
						rc = RC_SET( NE_SFLM_BUFFER_OVERFLOW);
						goto Exit;
					}
					*pszToken++ = (char)m_pucCurrLineBuf [uiOffset + 1];
					m_uiCurrLineOffset = uiOffset + 2;
				}
				else
				{
					m_uiCurrLineOffset = uiOffset + 1;
				}
				*pszToken = 0;
				break;
			default:
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset,
						SQL_ERR_INVALID_CHARACTER,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
		}
	}
	
	if (puiTokenLen)
	{
		*puiTokenLen = (FLMUINT)(pszToken - pszTokenStart);
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	See if the next token in the stream is the specified keyword.
****************************************************************************/
RCODE SQLStatement::haveToken(
	const char *	pszToken,
	FLMBOOL			bEofOK,
	SQLParseError	eNotHaveErr)
{
	RCODE		rc = NE_SFLM_OK;
	char		szToken [MAX_SQL_TOKEN_SIZE + 1];
	FLMUINT	uiTokenLineOffset;
	
	if (RC_BAD( rc = getToken( szToken, sizeof( szToken), bEofOK,
								&uiTokenLineOffset, NULL)))
	{
		if (rc == NE_SFLM_EOF_HIT)
		{
			flmAssert( bEofOK && eNotHaveErr == SQL_NO_ERROR);
			rc = RC_SET( NE_SFLM_NOT_FOUND);
		}
		goto Exit;
	}
	if (f_stricmp( pszToken, szToken) != 0)
	{
		
		// Restore the position where the token started - so it can
		// still be parsed.
		
		m_uiCurrLineOffset = uiTokenLineOffset;

		// At this point, we know we don't have the token
		
		if (eNotHaveErr != SQL_NO_ERROR)
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset,
					eNotHaveErr,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
		}
		else
		{
			rc = RC_SET( NE_SFLM_NOT_FOUND);
		}
		goto Exit;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Get next byte from input stream.
****************************************************************************/
RCODE SQLStatement::getByte(
	FLMBYTE *	pucByte)
{
	RCODE	rc = NE_SFLM_OK;
	
	if (m_ucUngetByte)
	{
		*pucByte = m_ucUngetByte;
		m_ucUngetByte = 0;
	}
	else
	{
		if( RC_BAD( rc = m_pStream->read( (char *)pucByte, 1, NULL)))
		{
			goto Exit;
		}
	}
	m_sqlStats.uiChars++;
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Reads next line from the input stream.
****************************************************************************/
RCODE SQLStatement::getLine( void)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBYTE		ucBytes [4];
	FLMUINT		uiNumBytes;
	FLMUINT		uiLoop;
	
	m_uiCurrLineBytes = 0;
	m_uiCurrLineOffset = 0;
	m_uiCurrLineFilePos = m_sqlStats.uiChars;	

	for (;;)
	{
		if( RC_BAD( rc = getByte( &ucBytes [0])))
		{
			if (rc == NE_SFLM_EOF_HIT)
			{
				if (m_uiCurrLineBytes)
				{
					rc = NE_SFLM_OK;
				}
			}
			goto Exit;
		}
		
		// Keep count of the characters.
		
		if( m_fnStatus && (m_sqlStats.uiChars % 1024) == 0)
		{
			m_fnStatus( SQL_PARSE_STATS,
				(void *)&m_sqlStats, NULL, NULL, m_pvCallbackData);
		}
		
		// Convert CRLF->CR

		if( ucBytes [0] == ASCII_CR)
		{
			if( RC_BAD( rc = getByte( &ucBytes [0])))
			{
				if (rc == NE_SFLM_EOF_HIT)
				{
					rc = NE_SFLM_OK;
					break;
				}
				else
				{
					goto Exit;
				}
			}

			if( ucBytes [0] != ASCII_NEWLINE)
			{
				ungetByte( ucBytes [0]);
			}
			
			// End of the line
			
			break;
		}
		else if (ucBytes [0] == ASCII_NEWLINE)
		{
			
			// End of the line
			
			break;
		}

		if( ucBytes [0] <= 0x7F)
		{
			uiNumBytes = 1;
		}
		else
		{

			if( RC_BAD( rc = getByte( &ucBytes [1])))
			{
				if (rc == NE_SFLM_EOF_HIT)
				{
					rc = RC_SET( NE_SFLM_BAD_UTF8);
				}
				goto Exit;
			}
	
			if( (ucBytes [1] >> 6) != 0x02)
			{
				rc = RC_SET( NE_SFLM_BAD_UTF8);
				goto Exit;
			}
	
			if( (ucBytes [0] >> 5) == 0x06)
			{
				uiNumBytes = 2;
			}
			else
			{
				if( RC_BAD( rc = getByte( &ucBytes [2])))
				{
					if (rc == NE_SFLM_EOF_HIT)
					{
						rc = RC_SET( NE_SFLM_BAD_UTF8);
					}
					goto Exit;
				}
		
				if( (ucBytes [2] >> 6) != 0x02 || (ucBytes [0] >> 4) != 0x0E)
				{
					rc = RC_SET( NE_SFLM_BAD_UTF8);
					goto Exit;
				}
				uiNumBytes = 3;
			}
		}

		// We have a character, add it to the current line.
		
		if (m_uiCurrLineBytes + uiNumBytes > m_uiCurrLineBufMaxBytes)
		{
			// Allocate more space for the line buffer
			
			if (RC_BAD( rc = f_realloc( m_uiCurrLineBufMaxBytes + 512,
						&m_pucCurrLineBuf)))
			{
				goto Exit;
			}
			m_uiCurrLineBufMaxBytes += 512;
		}
		for (uiLoop = 0; uiLoop < uiNumBytes; uiLoop++)
		{
			m_pucCurrLineBuf [m_uiCurrLineBytes++] = ucBytes [uiLoop];
		}
	}

	// Increment the line count

	m_uiCurrLineNum++;			
	m_sqlStats.uiLines++;
	if( m_fnStatus && (m_sqlStats.uiLines % 100) == 0)
	{
		m_fnStatus( SQL_PARSE_STATS,
			(void *)&m_sqlStats, NULL, NULL, m_pvCallbackData);
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:		Skips any whitespace characters in the input stream
****************************************************************************/
RCODE SQLStatement::skipWhitespace(
	FLMBOOL 			bRequired)
{
	FLMBYTE		ucChar;
	FLMUINT		uiCount = 0;
	RCODE			rc = NE_SFLM_OK;

	for( ;;)
	{
		if ((ucChar = getChar()) == 0)
		{
			uiCount++;
			if (RC_BAD( rc = getLine()))
			{
				goto Exit;
			}
			continue;
		}

		if (ucChar != ASCII_SPACE && ucChar != ASCII_TAB)
		{
			ungetChar();
			break;
		}
		uiCount++;
	}

	if( !uiCount && bRequired)
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset,
				SQL_ERR_EXPECTING_WHITESPACE,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_SFLM_INVALID_SQL);
		goto Exit;
	}

Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Parse a binary value.
//------------------------------------------------------------------------------
RCODE SQLStatement::getBinaryValue(
	F_DynaBuf *	pDynaBuf)
{
	RCODE		rc = NE_SFLM_OK;
	FLMBYTE	ucChar;
	FLMBOOL	bHaveHighNibble = FALSE;
	FLMBYTE	ucByte = 0;
	
	for (;;)
	{
		
		// If we hit end of line, just get the next line.
		
		if ((ucChar = getChar()) == 0)
		{
			if (RC_BAD( rc = getLine()))
			{
				goto Exit;
			}
			continue;
		}
		
		// If we hit the right paren, we are done
		
		if (ucChar == ')')
		{
			break;
		}
		
		if (ucChar >= '0' && ucChar <= '9')
		{
			if (bHaveHighNibble)
			{
				if (RC_BAD( rc = pDynaBuf->appendByte( ucByte | (FLMBYTE)(ucChar - '0'))))
				{
					goto Exit;
				}
				bHaveHighNibble = FALSE;
			}
			else
			{
				ucByte = (FLMBYTE)((ucChar - '0') << 4);
				bHaveHighNibble = TRUE;
			}
		}
		else if (ucChar >= 'a' && ucChar <= 'f')
		{
			if (bHaveHighNibble)
			{
				if (RC_BAD( rc = pDynaBuf->appendByte( ucByte | (FLMBYTE)(ucChar - 'a' + 10))))
				{
					goto Exit;
				}
				bHaveHighNibble = FALSE;
			}
			else
			{
				ucByte = (FLMBYTE)((ucChar - 'a' + 10) << 4);
				bHaveHighNibble = TRUE;
			}
		}
		else if (ucChar >= 'A' && ucChar <= 'F')
		{
			if (bHaveHighNibble)
			{
				if (RC_BAD( rc = pDynaBuf->appendByte( ucByte | (FLMBYTE)(ucChar - 'A' + 10))))
				{
					goto Exit;
				}
				bHaveHighNibble = FALSE;
			}
			else
			{
				ucByte = (FLMBYTE)((ucChar - 'A' + 10) << 4);
				bHaveHighNibble = TRUE;
			}
		}
		else if (ucChar == ' ' || ucChar == '\t' || ucChar == ',' ||
					ucChar == ';' || ucChar == ':')
		{
			
			// Some characters we will just ignore - who really cares - lets'
			// be forgiving.
			
		}
		else
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset - 1,
					SQL_ERR_NON_HEX_CHAR_IN_BINARY,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
			goto Exit;
			
		}
	}
	
	// If we have an unprocessed high nibble, add it to the dynamic buffer.
	
	if (bHaveHighNibble)
	{
		if (RC_BAD( rc = pDynaBuf->appendByte( ucByte)))
		{
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Parse a UTF8 string from the input stream.
//------------------------------------------------------------------------------
RCODE SQLStatement::getUTF8String(
	FLMBOOL		bMustHaveEqual,
	FLMBOOL		bStripWildcardEscapes,
	FLMBYTE *	pszStr,
	FLMUINT		uiStrBufSize,
	FLMUINT *	puiStrLen,
	FLMUINT *	puiNumChars,
	F_DynaBuf *	pDynaBuf)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBYTE		ucChar;
	FLMBYTE		ucQuoteChar = 0;
	FLMBOOL		bEscaped = FALSE;
	FLMUINT		uiStrLen = 0;
	FLMUINT		uiNumChars = 0;
	
	if (bMustHaveEqual)
	{
		if (RC_BAD( rc = haveToken( "=", FALSE, SQL_ERR_EXPECTING_EQUAL)))
		{
			goto Exit;
		}
	}
	if (RC_BAD( rc = skipWhitespace( FALSE)))
	{
		goto Exit;
	}

	if (!pDynaBuf)
	{
		
		// Always leave room for a null terminating character
		
		uiStrBufSize--;
	}
	
	// See if we have a quote character.
	
	ucChar = getChar();
	if (ucChar == '"' || ucChar == '\'')
	{
		ucQuoteChar = ucChar;
	}
	
	for (;;)
	{
		
		// If we hit end of line, it is invalid without hitting quote,
		// it is an error.
		
		if ((ucChar = getChar()) == 0)
		{
			if (ucQuoteChar)
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset,
						SQL_ERR_MISSING_QUOTE,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
			else
			{
				break;
			}
		}
		if (bEscaped)
		{
			// Can only escape backslash (the escape character), quotes, and
			// a few other characters.
			
			if (ucChar == 'n')
			{
				ucChar = ASCII_NEWLINE;
			}
			else if (ucChar == 't')
			{
				ucChar = ASCII_TAB;
			}
			else if (ucChar == 'r')
			{
				ucChar = ASCII_CR;
			}
			else if (ucChar == '\'' || ucChar == '"' ||
						ucChar == ASCII_SPACE || ucChar == ASCII_TAB ||
						ucChar == ',' || ucChar == ')')
			{
			}
			else if (ucChar == '*')
			{
				if (!bStripWildcardEscapes)
				{
					if (!pDynaBuf)
					{
						if (uiStrLen == uiStrBufSize)
						{
							setErrInfo( m_uiCurrLineNum,
									m_uiCurrLineOffset - 1,
									SQL_ERR_UTF8_STRING_TOO_LARGE,
									m_uiCurrLineFilePos,
									m_uiCurrLineBytes);
							rc = RC_SET( NE_SFLM_INVALID_SQL);
							goto Exit;
						}
						
						*pszStr++ = '\\';
					}
					else
					{
						if (RC_BAD( rc = pDynaBuf->appendByte( '\\')))
						{
							goto Exit;
						}
					}
					uiStrLen++;
					uiNumChars++;
				}
			}
			else
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset - 1,
						SQL_ERR_INVALID_ESCAPED_CHARACTER,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
	
			if (!pDynaBuf)
			{
				if (uiStrLen == uiStrBufSize)
				{
					setErrInfo( m_uiCurrLineNum,
							m_uiCurrLineOffset - 1,
							SQL_ERR_UTF8_STRING_TOO_LARGE,
							m_uiCurrLineFilePos,
							m_uiCurrLineBytes);
					rc = RC_SET( NE_SFLM_INVALID_SQL);
					goto Exit;
				}
				
				*pszStr++ = ucChar;
			}
			else
			{
				if (RC_BAD( rc = pDynaBuf->appendByte( ucChar)))
				{
					goto Exit;
				}
			}
			uiStrLen++;
			uiNumChars++;
		}
		else if (ucChar == '\\')
		{
			bEscaped = TRUE;
		}
		else if (ucChar == ucQuoteChar)
		{
			
			// If nothing follows the quote character, or the thing
			// that follows is not a quote character, we are at the
			// end of the string.
			
			if ((ucChar = getChar()) == 0 ||
				 ucChar != ucQuoteChar)
			{
				if (ucChar)
				{
					ungetChar();
				}
				break;
			}
			if (!pDynaBuf)
			{
				if (uiStrLen == uiStrBufSize)
				{
					setErrInfo( m_uiCurrLineNum,
							m_uiCurrLineOffset - 1,
							SQL_ERR_UTF8_STRING_TOO_LARGE,
							m_uiCurrLineFilePos,
							m_uiCurrLineBytes);
					rc = RC_SET( NE_SFLM_INVALID_SQL);
					goto Exit;
				}
	
				*pszStr++ = ucChar;
			}
			else
			{
				if (RC_BAD( rc = pDynaBuf->appendByte( ucChar)))
				{
					goto Exit;
				}
			}
			uiStrLen++;
			uiNumChars++;
		}
		
		// Non-quoted strings will end when we hit whitespace or
		// a comma or right paren.
		
		else if (ucChar == ASCII_SPACE || ucChar == ASCII_TAB ||
					ucChar == ',' || ucChar == ')')
		{
			ungetChar();
			break;
		}
		else
		{
			if (!pDynaBuf)
			{
				if (uiStrLen == uiStrBufSize)
				{
					setErrInfo( m_uiCurrLineNum,
							m_uiCurrLineOffset - 1,
							SQL_ERR_UTF8_STRING_TOO_LARGE,
							m_uiCurrLineFilePos,
							m_uiCurrLineBytes);
					rc = RC_SET( NE_SFLM_INVALID_SQL);
					goto Exit;
				}
				
				// Save the character to our buffer.
				
				*pszStr++ = ucChar;
			}
			else
			{
				if (RC_BAD( rc = pDynaBuf->appendByte( ucChar)))
				{
					goto Exit;
				}
			}
			uiStrLen++;
			uiNumChars++;
			
			// Handle multi-byte UTF8 characters.  The getLine() method has
			// already checked for valid UTF8, so that is all we should be
			// seeing here - thus the asserts.
			
			if (ucChar > 0x7F)
			{
				
				// It is at least two bytes.

				ucChar = getChar();
				flmAssert( (ucChar >> 6) == 0x02);
				if (!pDynaBuf)
				{
					if (uiStrLen == uiStrBufSize)
					{
						setErrInfo( m_uiCurrLineNum,
								m_uiCurrLineOffset - 1,
								SQL_ERR_UTF8_STRING_TOO_LARGE,
								m_uiCurrLineFilePos,
								m_uiCurrLineBytes);
						rc = RC_SET( NE_SFLM_INVALID_SQL);
						goto Exit;
					}
					*pszStr++ = ucChar;
				}
				else
				{
					if (RC_BAD( rc = pDynaBuf->appendByte( ucChar)))
					{
						goto Exit;
					}
				}
				
				uiStrLen++;
				
				// See if it is three bytes.
				
				if ((ucChar >> 5) != 0x06)
				{
					ucChar = getChar();
					flmAssert( (ucChar >> 6) == 0x02);
					if (!pDynaBuf)
					{
						if (uiStrLen == uiStrBufSize)
						{
							setErrInfo( m_uiCurrLineNum,
									m_uiCurrLineOffset - 1,
									SQL_ERR_UTF8_STRING_TOO_LARGE,
									m_uiCurrLineFilePos,
									m_uiCurrLineBytes);
							rc = RC_SET( NE_SFLM_INVALID_SQL);
							goto Exit;
						}
						*pszStr++ = ucChar;
					}
					else
					{
						if (RC_BAD( rc = pDynaBuf->appendByte( ucChar)))
						{
							goto Exit;
						}
					}
					uiStrLen++;
				}
			}
		}
	}
	
	// There will always be room for a null terminating byte if we
	// get to this point.

	if (!pDynaBuf)
	{
		*pszStr = 0;
	}
	else
	{
		if (RC_BAD( rc = pDynaBuf->appendByte( 0)))
		{
			goto Exit;
		}
	}
	if (puiStrLen)
	{
		*puiStrLen = uiStrLen;
	}
	if (puiNumChars)
	{
		*puiNumChars = uiNumChars;
	}
	
Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Parse a numeric value from the input stream.
//------------------------------------------------------------------------------
RCODE SQLStatement::getNumber(
	FLMBOOL		bMustHaveEqual,
	FLMUINT64 *	pui64Num,
	FLMBOOL *	pbNeg,
	FLMBOOL		bNegAllowed)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBYTE		ucChar;
	FLMUINT64	ui64Value = 0;
	FLMBOOL		bHex = FALSE;
	FLMUINT		uiDigitCount = 0;
	FLMUINT		uiDigitValue = 0;
	FLMUINT		uiSavedLineNum;
	FLMUINT		uiSavedOffset;
	FLMUINT		uiSavedFilePos;
	FLMUINT		uiSavedLineBytes;
	
	*pbNeg = FALSE;
	
	if (bMustHaveEqual)
	{
		if (RC_BAD( rc = haveToken( "=", FALSE, SQL_ERR_EXPECTING_EQUAL)))
		{
			goto Exit;
		}
	}
	if (RC_BAD( rc = skipWhitespace( FALSE)))
	{
		goto Exit;
	}
	
	uiSavedLineNum = m_uiCurrLineNum;
	uiSavedOffset = m_uiCurrLineOffset;
	uiSavedFilePos = m_uiCurrLineFilePos;
	uiSavedLineBytes = m_uiCurrLineBytes;
	
	// Go until we hit a character that is not a number.
	
	for (;;)
	{
		
		// If we hit the end of the line, we are done.
		
		if ((ucChar = getChar()) == 0)
		{
			break;
		}
		
		// Ignore white space
		
		if( f_isWhitespace( ucChar))
		{
			continue;
		}
		
		if (ucChar >= '0' && ucChar <= '9')
		{
			uiDigitValue = (FLMUINT)(ucChar - '0');
			uiDigitCount++;
		}
		else if (ucChar >= 'a' && ucChar <= 'f')
		{
			if (!bHex)
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset,
						SQL_ERR_ILLEGAL_HEX_DIGIT,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
			uiDigitValue = (FLMUINT)(ucChar - 'a' + 10);
			uiDigitCount++;
		}
		else if (ucChar >= 'A' && ucChar <= 'F')
		{
			if (!bHex)
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset,
						SQL_ERR_ILLEGAL_HEX_DIGIT,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
			uiDigitValue = (FLMUINT)(ucChar - 'A' + 10);
			uiDigitCount++;
		}
		else if (ucChar == ',' || ucChar == ')' ||
					ucChar == ASCII_SPACE || ucChar == ASCII_TAB)
		{
			
			// terminate when we hit a comma or right paren or white
			// space.  Need to unget the character so the caller can handle it.
			
			ungetChar();
			break;
		}
		else if (ucChar == 'X' || ucChar == 'x')
		{
			if (bHex || uiDigitCount != 1 || ui64Value)
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset,
						SQL_ERR_NON_NUMERIC_CHARACTER,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
			else
			{
				bHex = TRUE;
				uiDigitCount = 0;
				continue;
			}
		}
		else if (ucChar == '-')
		{
			if (bHex || uiDigitCount)
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset,
						SQL_ERR_NON_NUMERIC_CHARACTER,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
			else if (!bNegAllowed)
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset,
						SQL_ERR_INVALID_NEGATIVE_NUM,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
			else
			{
				*pbNeg = TRUE;
				continue;
			}
		}
		else
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset,
					SQL_ERR_NON_NUMERIC_CHARACTER,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
			goto Exit;
		}
		
		if (bHex)
		{
			if (ui64Value > (FLM_MAX_UINT64 >> 4))
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset,
						SQL_ERR_NUMBER_OVERFLOW,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
			ui64Value <<= 4;
			ui64Value += (FLMUINT64)uiDigitValue;
		}
		else
		{
			if (ui64Value > (FLM_MAX_UINT64 / 10))
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset,
						SQL_ERR_NUMBER_OVERFLOW,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
			ui64Value *= 10;
			ui64Value += (FLMUINT64)uiDigitValue;
		}
		
		// If it is a negative number, make sure we have not
		// exceeded the maximum negative value.
		
		if (*pbNeg && ui64Value > ((FLMUINT64)1 << 63))
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset,
					SQL_ERR_NUMBER_OVERFLOW,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
			goto Exit;
		}
	}
	
	// If we didn't hit any digits, we have an invalid number.
	
	if (!uiDigitCount)
	{
		setErrInfo( uiSavedLineNum,
				uiSavedOffset,
				SQL_ERR_NUMBER_VALUE_EMPTY,
				uiSavedFilePos,
				uiSavedLineBytes);
		rc = RC_SET( NE_SFLM_INVALID_SQL);
		goto Exit;
	}
	
	*pui64Num = ui64Value;
	
Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Parse a boolean value from the input stream.
//------------------------------------------------------------------------------
RCODE SQLStatement::getBool(
	FLMBOOL		bMustHaveEqual,
	FLMBOOL *	pbBool)
{
	RCODE			rc = NE_SFLM_OK;
	char			szBool [20];
	FLMUINT		uiBoolLen;
	
	if (RC_BAD( rc = getUTF8String( bMustHaveEqual, TRUE, (FLMBYTE *)szBool,
								sizeof( szBool), &uiBoolLen, NULL, NULL)))
	{
		goto Exit;
	}
	if (f_stricmp( szBool, "true") == 0)
	{
		*pbBool = TRUE;
	}
	else if (f_stricmp( szBool, "false") == 0)
	{
		*pbBool = FALSE;
	}
	else
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset,
				SQL_ERR_EXPECTING_BOOLEAN,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_SFLM_INVALID_SQL);
		goto Exit;
	}
	
Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Parse a FLMUINT numeric value from the input stream.
//------------------------------------------------------------------------------
RCODE SQLStatement::getUINT(
	FLMBOOL		bMustHaveEqual,
	FLMUINT *	puiNum)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBOOL		bNeg;
	FLMUINT64	ui64Value;
	
	if (RC_BAD( rc = getNumber( bMustHaveEqual, &ui64Value, &bNeg, FALSE)))
	{
		goto Exit;
	}
	
	// Number must be less than FLM_MAX_UINT
	
	if (ui64Value > (FLMUINT64)FLM_MAX_UINT)
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset,
				SQL_ERR_NUMBER_OVERFLOW,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_SFLM_INVALID_SQL);
		goto Exit;
	}
	*puiNum = (FLMUINT)ui64Value;
	
Exit:
	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Parse a table, column, or index name.
//------------------------------------------------------------------------------
RCODE SQLStatement::getName(
	char *		pszName,
	FLMUINT		uiNameBufSize,
	FLMUINT *	puiNameLen,
	FLMUINT *	puiTokenLineOffset)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiCharCount = 0;
	FLMBYTE		ucChar;
	FLMBYTE		ucQuoteChar;
	
	if (RC_BAD( rc = skipWhitespace( FALSE)))
	{
		goto Exit;
	}
	
	// Always leave room for a null terminating character.
	
	uiNameBufSize--;
	*puiTokenLineOffset = m_uiCurrLineOffset;
	
	// If the first character is a quote, it means we are going to
	// preserve case.
	
	ucChar = getChar();
	if (ucChar == '"' || ucChar == '\'')
	{
		ucQuoteChar = ucChar;
	}
	else
	{
		ucQuoteChar = 0;
	}

	// Get the first character - must be between A and Z

	ucChar = getChar();
	if ((ucChar >= 'a' && ucChar <= 'z') ||
		 (ucChar >= 'A' && ucChar <= 'Z'))
	{
		*pszName = (char)ucChar;
		uiCharCount++;
	}
	else
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset - 1,
				SQL_ERR_ILLEGAL_NAME_CHAR,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_SFLM_INVALID_SQL);
		goto Exit;
	}

	// Cannot go off of the current line
	
	for (;;)
	{
		if ((ucChar = getChar()) == 0)
		{
			break;
		}
		if (ucQuoteChar && ucChar == ucQuoteChar)
		{
			break;
		}
		if ((ucChar >= 'a' && ucChar <= 'z') ||
			 (ucChar >= 'A' && ucChar <= 'Z') ||
			 (ucChar >= '0' && ucChar <= '9') ||
			 (ucChar == '_'))
		{
			if (uiCharCount >= uiNameBufSize)
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset - 1,
						SQL_ERR_NAME_TOO_LONG,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
			
			// Convert name to upper case unless the name is quoted.
			
			if (ucChar >= 'a' && ucChar <= 'z' && !ucQuoteChar)
			{
				pszName [uiCharCount++] = (char)(ucChar - 'a' + 'A');
			}
			else
			{
				pszName [uiCharCount++] = (char)ucChar;
			}
		}
		else if (ucQuoteChar)
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset - 1,
					SQL_ERR_INVALID_CHAR_IN_NAME,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
			goto Exit;
		}
		else
		{
			ungetChar();
			break;
		}
	}

	pszName [uiCharCount] = 0;

Exit:

	if (puiNameLen)
	{
		*puiNameLen = uiCharCount;
	}
	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Parse the encryption definition name for the current statement.
//			Make sure it is valid.
//------------------------------------------------------------------------------
RCODE SQLStatement::getEncDefName(
	FLMBOOL		bMustExist,
	char *		pszEncDefName,
	FLMUINT		uiEncDefNameBufSize,
	FLMUINT *	puiEncDefNameLen,
	F_ENCDEF **	ppEncDef)
{
	RCODE		rc = NE_SFLM_OK;
	FLMUINT	uiTokenLineOffset;

	if (RC_BAD( rc = getName( pszEncDefName, uiEncDefNameBufSize,
								puiEncDefNameLen, &uiTokenLineOffset)))
	{
		goto Exit;
	}
	
	// See if the encryption definition name is defined
	
	if ((*ppEncDef = m_pDb->m_pDict->findEncDef( pszEncDefName)) == NULL)
	{
		if (bMustExist)
		{
			setErrInfo( m_uiCurrLineNum,
					uiTokenLineOffset,
					SQL_ERR_UNDEFINED_ENCDEF,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
			goto Exit;
		}
		else
		{
			rc = NE_SFLM_OK;
		}
	}
	else
	{
		if (!bMustExist)
		{
			setErrInfo( m_uiCurrLineNum,
					uiTokenLineOffset,
					SQL_ERR_ENCDEF_ALREADY_DEFINED,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Parse the table name for the current statement.  Make sure it is valid.
//------------------------------------------------------------------------------
RCODE SQLStatement::getTableName(
	FLMBOOL		bMustExist,
	char *		pszTableName,
	FLMUINT		uiTableNameBufSize,
	FLMUINT *	puiTableNameLen,
	F_TABLE **	ppTable)
{
	RCODE		rc = NE_SFLM_OK;
	FLMUINT	uiTokenLineOffset;

	if (RC_BAD( rc = getName( pszTableName, uiTableNameBufSize,
								puiTableNameLen, &uiTokenLineOffset)))
	{
		goto Exit;
	}
	
	// See if the table name is defined
	
	if (RC_BAD( rc = m_pDb->m_pDict->getTable( pszTableName, ppTable, TRUE)))
	{
		if (rc != NE_SFLM_BAD_TABLE)
		{
			goto Exit;
		}
		if (bMustExist)
		{
			setErrInfo( m_uiCurrLineNum,
					uiTokenLineOffset,
					SQL_ERR_UNDEFINED_TABLE,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
			goto Exit;
		}
		else
		{
			rc = NE_SFLM_OK;
		}
	}
	else
	{
		if (!bMustExist)
		{
			setErrInfo( m_uiCurrLineNum,
					uiTokenLineOffset,
					SQL_ERR_TABLE_ALREADY_DEFINED,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Parse the index name for the current statement.  Make sure it is valid.
//------------------------------------------------------------------------------
RCODE SQLStatement::getIndexName(
	FLMBOOL		bMustExist,
	F_TABLE *	pTable,
	char *		pszIndexName,
	FLMUINT		uiIndexNameBufSize,
	FLMUINT *	puiIndexNameLen,
	F_INDEX **	ppIndex)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiTokenLineOffset;

	if (RC_BAD( rc = getName( pszIndexName, uiIndexNameBufSize,
							puiIndexNameLen, &uiTokenLineOffset)))
	{
		goto Exit;
	}
	
	// See if the index name is defined
	
	if (RC_BAD( rc = m_pDb->m_pDict->getIndex( pszIndexName, ppIndex, TRUE)))
	{
		if (rc != NE_SFLM_BAD_IX)
		{
			goto Exit;
		}
		if (bMustExist)
		{
			setErrInfo( m_uiCurrLineNum,
					uiTokenLineOffset,
					SQL_ERR_UNDEFINED_INDEX,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
			goto Exit;
		}
		else
		{
			rc = NE_SFLM_OK;
		}
	}
	else
	{
		if (!bMustExist)
		{
			setErrInfo( m_uiCurrLineNum,
					uiTokenLineOffset,
					SQL_ERR_INDEX_ALREADY_DEFINED,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
			goto Exit;
		}
		else if (pTable)
		{
			if (pTable->uiTableNum != (*ppIndex)->uiTableNum)
			{
				setErrInfo( m_uiCurrLineNum,
						uiTokenLineOffset,
						SQL_ERR_INDEX_NOT_DEFINED_FOR_TABLE,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
		}
	}

Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Parse a string value from the input stream.
//------------------------------------------------------------------------------
RCODE SQLStatement::getStringValue(
	F_COLUMN *			pColumn,
	F_COLUMN_VALUE *	pColumnValue)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBYTE		ucChar;
	FLMBYTE		ucQuoteChar = 0;
	FLMBOOL		bEscaped = FALSE;
	FLMBYTE		szTmpBuf [300];
	F_DynaBuf	dynaBuf( szTmpBuf, sizeof( szTmpBuf));
	FLMUINT		uiNumChars = 0;
	FLMBYTE *	pucValue;
	FLMUINT		uiSenLen;
	
	// Leading white space has already been skipped.
	
	// See if we have a quote character.
	
	ucChar = getChar();
	if (ucChar == '"' || ucChar == '\'')
	{
		ucQuoteChar = ucChar;
	}
	else
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset,
				SQL_ERR_EXPECTING_QUOTE_CHAR,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_SFLM_INVALID_SQL);
		goto Exit;
	}
	
	for (;;)
	{
		// Should not hit the end of the line if quoted.
		
		if ((ucChar = getChar()) == 0)
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset,
					SQL_ERR_MISSING_QUOTE,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
			goto Exit;
		}
		if (bEscaped)
		{
			// Can only escape backslash (the escape character), quotes, and
			// a few other characters.
			
			if (ucChar == 'n')
			{
				ucChar = ASCII_NEWLINE;
			}
			else if (ucChar == 't')
			{
				ucChar = ASCII_TAB;
			}
			else if (ucChar == 'r')
			{
				ucChar = ASCII_CR;
			}
			else if (ucChar == '\'' || ucChar == '"')
			{
			}
			else
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset,
						SQL_ERR_INVALID_ESCAPED_CHARACTER,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
			
			// Save the escaped character to the buffer.

			if (RC_BAD( rc = dynaBuf.appendByte( ucChar)))
			{
				goto Exit;
			}
			uiNumChars++;
		}
		else if (ucChar == '\\')
		{
			bEscaped = TRUE;
		}
		else if (ucChar == ucQuoteChar)
		{
			break;
		}
		else
		{
			
			// Save the character to our buffer.
			
			if (RC_BAD( rc = dynaBuf.appendByte( ucChar)))
			{
				goto Exit;
			}
			
			// Handle multi-byte UTF8 characters.  The getLine() method has
			// already checked for valid UTF8, so that is all we should be
			// seeing here - thus the asserts.
			
			if (ucChar > 0x7F)
			{
				
				// It is at least two bytes.
				
				ucChar = getChar();
				flmAssert( (ucChar >> 6) == 0x02);
				if (RC_BAD( rc = dynaBuf.appendByte( ucChar)))
				{
					goto Exit;
				}
				
				// See if it is three bytes.
				
				if ((ucChar >> 5) != 0x06)
				{
					ucChar = getChar();
					flmAssert( (ucChar >> 6) == 0x02);
					if (RC_BAD( rc = dynaBuf.appendByte( ucChar)))
					{
						goto Exit;
					}
				}
			}
			uiNumChars++;
		}
	}
	
	// Add a null terminating byte
	
	if (RC_BAD( rc = dynaBuf.appendByte( 0)))
	{
		goto Exit;
	}
	
	// See if the string is too long.
	
	if (pColumn->uiMaxLen && uiNumChars > pColumn->uiMaxLen)
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset,
				SQL_ERR_STRING_TOO_LONG,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_SFLM_STRING_TOO_LONG);
		goto Exit;
	}
	
	// Allocate space for the UTF8 string.
	
	uiSenLen = f_getSENByteCount( uiNumChars);
	pColumnValue->uiValueLen = dynaBuf.getDataLength() + uiSenLen;
	if (RC_BAD( rc = m_tmpPool.poolAlloc( pColumnValue->uiValueLen,
											(void **)&pucValue)))
	{
		goto Exit;
	}
	pColumnValue->pucColumnValue = pucValue;
	f_encodeSEN( uiNumChars, &pucValue);
	
	// Copy the string from the dynaBuf to the column - NOTE: includes
	// null terminating byte.
	
	f_memcpy( pucValue, dynaBuf.getBufferPtr(), dynaBuf.getDataLength());
	
Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Parse a numeric value from the input stream.
//------------------------------------------------------------------------------
RCODE SQLStatement::getNumberValue(
	F_COLUMN_VALUE *	pColumnValue)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT64	ui64Value = 0;
	FLMBOOL		bNeg = FALSE;
	FLMBYTE *	pucValue;
	FLMBYTE		ucNumBuf [40];
	FLMUINT		uiValLen;

	if (RC_BAD( rc = getNumber( FALSE, &ui64Value, &bNeg, TRUE)))
	{
		goto Exit;
	}
	
	// Convert to storage format.
	
	uiValLen = sizeof( ucNumBuf);
	if (RC_BAD( rc = flmNumber64ToStorage( ui64Value, &uiValLen, ucNumBuf,
								bNeg, FALSE)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = m_tmpPool.poolAlloc( uiValLen, (void **)&pucValue)))
	{
		goto Exit;
	}
	pColumnValue->uiValueLen = uiValLen;
	pColumnValue->pucColumnValue = pucValue;
	f_memcpy( pucValue, ucNumBuf, uiValLen);
	
Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Parse a binary value from the input stream.
//------------------------------------------------------------------------------
RCODE SQLStatement::getBinaryValue(
	F_COLUMN *			pColumn,
	F_COLUMN_VALUE *	pColumnValue)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBYTE		ucChar;
	FLMBYTE		szTmpBuf [300];
	F_DynaBuf	dynaBuf( szTmpBuf, sizeof( szTmpBuf));
	FLMBYTE		ucCurrByte;
	FLMBOOL		bGetHighNibble;
	FLMUINT		uiSavedLineNum = m_uiCurrLineNum;
	FLMUINT		uiSavedOffset = m_uiCurrLineOffset;
	FLMUINT		uiSavedFilePos = m_uiCurrLineFilePos;
	FLMUINT		uiSavedLineBytes = m_uiCurrLineBytes;
	
	// Leading white space has already been skipped.
	
	// Go until we hit a character that is not a hex digit.
	
	ucCurrByte = 0;
	bGetHighNibble = TRUE;
	for (;;)
	{
		
		// It is OK for white space to be in the middle of a binary
		// piece of data.  It is also allowed to span multiple lines.
		
		if ((ucChar = getChar()) == 0)
		{
			if (RC_BAD( rc = getLine()))
			{
				goto Exit;
			}
			continue;
		}
		
		// Ignore white space
		
		if (ucChar == ASCII_SPACE || ucChar == ASCII_TAB)
		{
			continue;
		}
		if (ucChar >= '0' && ucChar <= '9')
		{
			if (bGetHighNibble)
			{
				ucCurrByte = (ucChar - '0') << 4;
			}
			else
			{
				ucCurrByte |= (ucChar - '0');
			}
		}
		else if (ucChar >= 'a' && ucChar <= 'f')
		{
			if (bGetHighNibble)
			{
				ucCurrByte = (ucChar - 'a' + 10) << 4;
			}
			else
			{
				ucCurrByte |= (ucChar - 'a' + 10);
			}
		}
		else if (ucChar >= 'A' && ucChar <= 'F')
		{
			if (bGetHighNibble)
			{
				ucCurrByte = (ucChar - 'A' + 10) << 4;
			}
			else
			{
				ucCurrByte |= (ucChar - 'A' + 10);
			}
		}
		else if (ucChar == ',' || ucChar == ')')
		{
			
			// terminate when we hit a comma or right paren.  Need to
			// unget the character because the caller will be expecting
			// to see it.
			
			ungetChar();
			break;
		}
		else
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset,
					SQL_ERR_NON_HEX_CHARACTER,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
			goto Exit;
		}
		if (bGetHighNibble)
		{
			bGetHighNibble = FALSE;
		}
		else
		{
			if (RC_BAD( rc = dynaBuf.appendByte( ucCurrByte)))
			{
				goto Exit;
			}
			bGetHighNibble = TRUE;
			ucCurrByte = 0;
		}
	}
	
	// Add last byte if bGetHighNibble is FALSE - means we got the high nibble
	// into the high four bits of ucCurrByte
	
	if (!bGetHighNibble)
	{
		if (RC_BAD( rc = dynaBuf.appendByte( ucCurrByte)))
		{
			goto Exit;
		}
	}
	
	// See if the binary data is too long.
	
	if (pColumn->uiMaxLen && dynaBuf.getDataLength() > pColumn->uiMaxLen)
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset,
				SQL_ERR_BINARY_TOO_LONG,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_SFLM_BINARY_TOO_LONG);
		goto Exit;
	}
	
	// An empty binary value is invalid.
	
	if ((pColumnValue->uiValueLen = dynaBuf.getDataLength()) == 0)
	{
		setErrInfo( uiSavedLineNum,
				uiSavedOffset,
				SQL_ERR_BINARY_VALUE_EMPTY,
				uiSavedFilePos,
				uiSavedLineBytes);
		rc = RC_SET( NE_SFLM_INVALID_SQL);
		goto Exit;
	}
	
	// Allocate space for the binary data.
	
	if (RC_BAD( rc = m_tmpPool.poolAlloc( pColumnValue->uiValueLen,
											(void **)&pColumnValue->pucColumnValue)))
	{
		goto Exit;
	}
	
	// Copy the binary data from the dynaBuf to the column.
	
	f_memcpy( pColumnValue->pucColumnValue, dynaBuf.getBufferPtr(),
				 pColumnValue->uiValueLen);
	
Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Parse a value from the input stream.  Value must be of the specified
//			type.
//------------------------------------------------------------------------------
RCODE SQLStatement::getValue(
	F_COLUMN *			pColumn,
	F_COLUMN_VALUE *	pColumnValue)
{
	RCODE	rc = NE_SFLM_OK;
	
	switch (pColumn->eDataTyp)
	{
		case SFLM_STRING_TYPE:
			if (RC_BAD( rc = getStringValue( pColumn, pColumnValue)))
			{
				goto Exit;
			}
			break;
		case SFLM_NUMBER_TYPE:
			if (RC_BAD( rc = getNumberValue( pColumnValue)))
			{
				goto Exit;
			}
			break;
		case SFLM_BINARY_TYPE:
			if (RC_BAD( rc = getBinaryValue( pColumn, pColumnValue)))
			{
				goto Exit;
			}
			break;
		default:
			flmAssert( 0);
			break;
	}
	
Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Parse and execute an SQL statement.
//------------------------------------------------------------------------------
RCODE SQLStatement::executeSQL(
	IF_IStream *	pStream,
	F_Db *			pDb,
	SQL_STATS *		pSQLStats)
{
	RCODE		rc = NE_SFLM_OK;
	char		szToken [MAX_SQL_TOKEN_SIZE + 1];
	FLMUINT	uiTokenLineOffset;

	// Reset the state of the parser

	if (RC_BAD( rc = setupStatement()))
	{
		goto Exit;
	}

	m_pStream = pStream;
	m_pDb = pDb;

	// Process all of the statements.

	for (;;)
	{
		if (RC_BAD( rc = getToken( szToken, sizeof( szToken), TRUE,
											&uiTokenLineOffset, NULL)))
		{
			if( rc != NE_SFLM_EOF_HIT)
			{
				goto Exit;
			}
			
			rc = NE_SFLM_OK;
			break;
		}
		
		if (f_stricmp( szToken, "select") == 0)
		{
			if (RC_BAD( rc = processSelect()))
			{
				goto Exit;
			}
		}
		else if (f_stricmp( szToken, "insert") == 0)
		{
			if (RC_BAD( rc = processInsertRow()))
			{
				goto Exit;
			}
		}
		else if (f_stricmp( szToken, "update") == 0)
		{
			if (RC_BAD( rc = processUpdateRows()))
			{
				goto Exit;
			}
		}
		else if (f_stricmp( szToken, "delete") == 0)
		{
			if (RC_BAD( rc = processDeleteRows()))
			{
				goto Exit;
			}
		}
		else if (f_stricmp( szToken, "open") == 0)
		{
			if (RC_BAD( rc = haveToken( "database", FALSE, 
				SQL_ERR_INVALID_OPEN_OPTION)))
			{
				goto Exit;
			}
			
			if (RC_BAD( rc = processOpenDatabase()))
			{
				goto Exit;
			}
		}
		else if (f_stricmp( szToken, "create") == 0)
		{
			if (RC_BAD( rc = getToken( szToken, sizeof( szToken), FALSE,
												&uiTokenLineOffset, NULL)))
			{
				goto Exit;
			}
			if (f_stricmp( szToken, "database") == 0)
			{
				if (RC_BAD( rc = processCreateDatabase()))
				{
					goto Exit;
				}
			}
			else if (f_stricmp( szToken, "table") == 0)
			{
				if (RC_BAD( rc = processCreateTable()))
				{
					goto Exit;
				}
			}
			else if (f_stricmp( szToken, "index") == 0)
			{
				if (RC_BAD( rc = processCreateIndex( FALSE)))
				{
					goto Exit;
				}
			}
			else if (f_stricmp( szToken, "unique") == 0)
			{
				if (RC_BAD( rc = haveToken( "index", FALSE,
												SQL_ERR_EXPECTING_INDEX)))
				{
					goto Exit;
				}
				if (RC_BAD( rc = processCreateIndex( TRUE)))
				{
					goto Exit;
				}
			}
			else
			{
				setErrInfo( m_uiCurrLineNum,
						uiTokenLineOffset,
						SQL_ERR_INVALID_CREATE_OPTION,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
		}
		else if (f_stricmp( szToken, "drop") == 0)
		{
			if (RC_BAD( rc = getToken( szToken, sizeof( szToken), FALSE,
												&uiTokenLineOffset, NULL)))
			{
				goto Exit;
			}
			if (f_stricmp( szToken, "database") == 0)
			{
				if (RC_BAD( rc = processDropDatabase()))
				{
					goto Exit;
				}
			}
			else if (f_stricmp( szToken, "table") == 0)
			{
				if (RC_BAD( rc = processDropTable()))
				{
					goto Exit;
				}
			}
			else if (f_stricmp( szToken, "index") == 0)
			{
				if (RC_BAD( rc = processDropIndex()))
				{
					goto Exit;
				}
			}
			else
			{
				setErrInfo( m_uiCurrLineNum,
						uiTokenLineOffset,
						SQL_ERR_INVALID_DROP_OPTION,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
		}
		else
		{
			setErrInfo( m_uiCurrLineNum,
					uiTokenLineOffset,
					SQL_ERR_INVALID_SQL_STATEMENT,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
			goto Exit;
		}
	}

	// Call the status hook one last time

	if (m_fnStatus)
	{
		m_fnStatus( SQL_PARSE_STATS,
			(void *)&m_sqlStats, NULL, NULL, m_pvCallbackData);
	}

	// Tally and return the SQL statistics

	if( pSQLStats)
	{
		pSQLStats->uiLines += m_sqlStats.uiLines;
		pSQLStats->uiChars += m_sqlStats.uiChars;
	}

Exit:

	if (rc == NE_SFLM_EOF_HIT)
	{
		rc = NE_SFLM_OK;
	}

	if( RC_BAD( rc) && pSQLStats)
	{
		pSQLStats->uiErrLineNum = m_sqlStats.uiErrLineNum
			? m_sqlStats.uiErrLineNum
			: m_uiCurrLineNum;

		pSQLStats->uiErrLineOffset = m_sqlStats.uiErrLineOffset
			? m_sqlStats.uiErrLineOffset
			: m_uiCurrLineOffset;

		pSQLStats->eErrorType = m_sqlStats.eErrorType;
		
		pSQLStats->uiErrLineFilePos = m_sqlStats.uiErrLineFilePos;
		pSQLStats->uiErrLineBytes = m_sqlStats.uiErrLineBytes;
	}

	m_pDb = NULL;

	return( rc);
}

