//------------------------------------------------------------------------------
// Desc:	Class to support reading/writing/parsing .ini files
// Tabs:	3
//
// Copyright (c) 2002-2007 Novell, Inc. All Rights Reserved.
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

#include "ftksys.h"

/****************************************************************************
Desc:
****************************************************************************/
typedef struct INI_LINE
{
	char *				pszParamName;
	char *				pszParamValue;	
	char *				pszComment;
	struct INI_LINE *	pPrev;
	struct INI_LINE *	pNext;
} INI_LINE;

/****************************************************************************
Desc:
****************************************************************************/
class F_IniFile : public IF_IniFile
{
public:

	F_IniFile();
	
	virtual ~F_IniFile();
	
	void init( void);
	
	RCODE FTKAPI read(
		const char *		pszFileName);
		
	RCODE FTKAPI write( void);

	FLMBOOL FTKAPI getParam(
		const char *	pszParamName,
		FLMUINT *		puiParamVal);
	
	FLMBOOL FTKAPI getParam(
		const char *	pszParamName,
		FLMBOOL *		pbParamVal);
	
	FLMBOOL FTKAPI getParam(
		const char *	pszParamName,
		char **			ppszParamVal);
	
	RCODE FTKAPI setParam(
		const char *	pszParamName,
		FLMUINT 			uiParamVal);

	RCODE FTKAPI setParam(
		const char *	pszParamName,
		FLMBOOL			bParamVal);

	RCODE FTKAPI setParam(
		const char *	pszParamName,
		const char *	pszParamVal);

	FINLINE FLMBOOL FTKAPI testParam(
		const char *	pszParamName)
	{
		if( findParam( pszParamName))
		{
			return( TRUE);
		}
		
		return( FALSE);
	}

private:

	RCODE readLine(
		char *			pucBuf,
		FLMUINT *		puiBytes,
		FLMBOOL *		pbMore);

	RCODE parseBuffer(
		char *			pucBuf,
		FLMUINT			uiNumButes);

	INI_LINE * findParam(
		const char *	pszParamName);

	RCODE setParamCommon( 
		INI_LINE **		ppLine,
		const char *	pszParamName);

	void fromAscii( 
		FLMUINT * 		puiVal,
		const char *	pszParamValue);
		
	void fromAscii(
		FLMBOOL *		pbVal,
		const char *	pszParamValue);

	RCODE toAscii( 
		char **			ppszParamValue,
		FLMUINT			puiVal);
		
	RCODE toAscii( 
		char **			ppszParamValue,
		FLMBOOL 			pbVal);
		
	RCODE toAscii(
		char **			ppszParamValue,
		const char * 	pszVal);

	FINLINE FLMBOOL isWhiteSpace(
		FLMBYTE			ucChar)
	{
		return( ucChar == 32 || ucChar == 9 ? TRUE : FALSE);
	}
	
	F_Pool				m_pool;
	IF_FileHdl * 		m_pFileHdl;
	char *				m_pszFileName;
	INI_LINE *			m_pFirstLine;	
	INI_LINE *			m_pLastLine;
	FLMBOOL				m_bReady;
	FLMBOOL				m_bModified;
	FLMUINT				m_uiFileOffset;
};

/****************************************************************************
Desc:
****************************************************************************/
F_IniFile::F_IniFile()
{
	m_pFirstLine = NULL;
	m_pLastLine = NULL;
	m_bReady = FALSE;
	m_bModified = FALSE;
	m_pszFileName = NULL;
	m_pFileHdl = NULL;
	m_pool.poolInit( 512);
}

/****************************************************************************
Desc:
****************************************************************************/
F_IniFile::~F_IniFile()
{
	if( m_pszFileName)
	{
		f_free( &m_pszFileName);
	}
	
	m_pool.poolFree();
	
	if( m_pFileHdl)
	{
		m_pFileHdl->Release();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI FlmAllocIniFile(
	IF_IniFile **				ppIniFile)
{
	RCODE				rc = NE_FLM_OK;
	F_IniFile *		pIniFile = NULL;
	
	if( (pIniFile = f_new F_IniFile) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	pIniFile->init();
	
	*ppIniFile = pIniFile;
	pIniFile = NULL;
	
Exit:

	if( pIniFile)
	{
		pIniFile->Release();
	}

	return( rc);
}
		
/****************************************************************************
Desc:
****************************************************************************/
void F_IniFile::init( void)
{
	m_pool.poolFree();	
	m_pool.poolInit( 512);
	m_pFirstLine = NULL;
	m_pLastLine = NULL;
	m_bReady = TRUE;
}

/****************************************************************************
Desc:	Read the ini file and parse its contents
****************************************************************************/
RCODE FTKAPI F_IniFile::read(
	const char *		pszFileName)
{
	RCODE					rc = NE_FLM_OK;
	FLMBOOL				bMore = FALSE;
	FLMBOOL				bEOF = FALSE;
#define INITIAL_READ_BUF_SIZE	100
	FLMUINT				uiReadBufSize = 0;
	FLMUINT				uiBytesAvail = 0;
	FLMUINT				uiBytesInLine = 0;
	char *				pszReadBuf = NULL;
	FLMUINT				uiLineNum = 0;
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();
	
	f_assert( m_bReady);
	f_assert( !m_pFileHdl);

	// Open the file

	if (RC_BAD( rc = f_alloc( f_strlen( pszFileName) + 1, &m_pszFileName)))
	{
		goto Exit;
	}

	f_strcpy( m_pszFileName, pszFileName);

	// It's not an error if the file doesn't exist.  If it does exist,
	// we'll read in its data.
	
	if( RC_BAD( pFileSystem->openFile( pszFileName, 
		FLM_IO_RDONLY, &m_pFileHdl)))
	{		
		goto Exit;
	}

	m_uiFileOffset = 0;

	// Read in and parse the file
	
	uiReadBufSize = INITIAL_READ_BUF_SIZE;
	if (RC_BAD( rc = f_alloc( uiReadBufSize, &pszReadBuf)))
	{
		goto Exit;
	}

	// Read in and parse each line in the file...
	while (!bEOF)
	{
		uiLineNum++;

		uiBytesAvail = uiReadBufSize;
		if( RC_BAD( rc = readLine( pszReadBuf, &uiBytesAvail, &bMore)) &&
			 rc != NE_FLM_IO_END_OF_FILE)
		{
			goto Exit;
		}
		
		if (rc == NE_FLM_IO_END_OF_FILE)
		{
			bEOF = TRUE;
		}
		
		// While there are more bytes in the line, re-alloc the buffer, and do
		// another read.
		
		uiBytesInLine = uiBytesAvail;
		while( bMore)
		{
			uiBytesAvail = uiReadBufSize;
			uiReadBufSize *= 2;

			if (RC_BAD( rc = f_realloc( uiReadBufSize, &pszReadBuf)))
			{
				goto Exit;
			}
			
			if (RC_BAD( rc = readLine( pszReadBuf+uiBytesAvail,
												&uiBytesAvail,	&bMore))	&&
				 (rc != NE_FLM_IO_END_OF_FILE) )
			{
				goto Exit;
			}
			
			if( rc == NE_FLM_IO_END_OF_FILE)
			{
				bEOF = TRUE;
			}
			uiBytesInLine += uiBytesAvail;
		}
		
		if ( (RC_OK( rc) || (rc == NE_FLM_IO_END_OF_FILE)) &&
				(uiBytesInLine > 0) )
		{
			// NumBytes will be 0 if the line was blank.  No need
			// to call parseBuffer in this case
			
			if (RC_BAD( rc = parseBuffer( pszReadBuf, uiBytesInLine)))
			{
				if (rc == NE_FLM_SYNTAX)
				{
					rc = NE_FLM_OK;
				}
				else
				{
					goto Exit;
				}
			}
		}
	}

Exit:

	// Close the file
	
	if( m_pFileHdl)
	{
		m_pFileHdl->closeFile();
		m_pFileHdl->Release();
		m_pFileHdl = NULL;
	}

	// Free the buffer
	
	if (pszReadBuf)
	{
		f_free( &pszReadBuf);
	}

	if (rc == NE_FLM_IO_END_OF_FILE)
	{
		rc = NE_FLM_OK;
	}

	return rc;
}

/****************************************************************************
Desc:	Copies the data stored in the INI_LINE structs to the ini file
****************************************************************************/
RCODE FTKAPI F_IniFile::write( void)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT				uiBytesWritten;
	INI_LINE *			pCurLine = NULL;
	FLMUINT				uiFileOffset = 0;		
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();

	f_assert( m_bReady);
	
	if (!m_bModified)
	{
		// Nothing needs to be written
		
		goto Exit;
	}

	// Open the file
	
	f_assert( !m_pFileHdl);
	
	if (RC_BAD( rc = pFileSystem->createFile( m_pszFileName,
								FLM_IO_RDWR, &m_pFileHdl)))
	{
		goto Exit;
	}

	pCurLine = m_pFirstLine;
	while (pCurLine)
	{
		if (pCurLine->pszParamName)
		{
			// Output the param name
			
			if (RC_BAD (rc = m_pFileHdl->write( uiFileOffset,
				f_strlen( pCurLine->pszParamName), pCurLine->pszParamName,
				&uiBytesWritten)))
			{
				goto Exit;
			}
			uiFileOffset += uiBytesWritten;
			
			if (pCurLine->pszParamValue)
			{
				// Output the "=" and the value
				
				if (RC_BAD (rc = m_pFileHdl->write( uiFileOffset, 1,
					(void *)"=", &uiBytesWritten)))
				{
					goto Exit;
				}

				uiFileOffset += uiBytesWritten;

				if (RC_BAD (rc = m_pFileHdl->write( uiFileOffset,
					f_strlen( pCurLine->pszParamValue), pCurLine->pszParamValue,
					&uiBytesWritten)))
				{
					goto Exit;
				}
				uiFileOffset += uiBytesWritten;
			}
		}
	

		if (pCurLine->pszComment)
		{
			// Output the comment
			
			if (pCurLine->pszParamName)
			{
				if (RC_BAD (rc = m_pFileHdl->write( uiFileOffset, 2,
					(void *)" #", &uiBytesWritten)))
				{
					goto Exit;
				}
			}
			else
			{
				if (RC_BAD (rc = m_pFileHdl->write( uiFileOffset, 1,
					(void *)"#", &uiBytesWritten)))
				{
					goto Exit;
				}
			}

			uiFileOffset += uiBytesWritten;

			if (RC_BAD (rc = m_pFileHdl->write( uiFileOffset,
				f_strlen( pCurLine->pszComment), pCurLine->pszComment,
				&uiBytesWritten)))
			{
				goto Exit;
			}

			uiFileOffset += uiBytesWritten;

		}

		// Write out a newline...
		
		if (RC_BAD (rc = m_pFileHdl->write( uiFileOffset, f_strlen( "\n"),
			(void *)"\n", &uiBytesWritten)))
		{
			goto Exit;
		}

		uiFileOffset += uiBytesWritten;
		pCurLine = pCurLine->pNext;
	}

	m_bModified = FALSE;

Exit:

	if (m_pFileHdl)
	{
		m_pFileHdl->closeFile();
		m_pFileHdl->Release();
		m_pFileHdl = NULL;
	}

	return( rc);
}

/****************************************************************************
Desc:	Retrieves the value associated with the specified name from the list
	   of INI_STRUCTs
****************************************************************************/
FLMBOOL FTKAPI F_IniFile::getParam(
	const char *	pszParamName,
	FLMUINT *		puiParamVal)
{
	FLMBOOL		bFound = FALSE;
	INI_LINE *	pLine = NULL;

	f_assert( m_bReady);
	
	pLine = findParam( pszParamName);
	if( !pLine)
	{
		goto Exit;
	}

	if( !pLine->pszParamValue)
	{
		goto Exit;
	}
	
	fromAscii( puiParamVal, pLine->pszParamValue);
	bFound = TRUE;
	
Exit:

	return( bFound);
}

/****************************************************************************
Desc:	Stores a new value for the specified name (or creates a new name/value
	   pair) in the list of INI_STRUCTs
****************************************************************************/
RCODE FTKAPI F_IniFile::setParam(
	const char *	pszParamName,
	FLMUINT 			uiParamVal)
{
	RCODE			rc = NE_FLM_OK;
	INI_LINE *	pLine;

	f_assert( m_bReady);

	// If the parameter exists in the list, just store the new value.
	// Othewise, create a new INI_LINE and add it to the list
	
	pLine = findParam( pszParamName);
	if( !pLine)
	{
		if (RC_BAD( rc = setParamCommon( &pLine, pszParamName)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = toAscii( &pLine->pszParamValue, uiParamVal)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Retrieves the value associated with the specified name from the list
	   of INI_STRUCTs
****************************************************************************/
FLMBOOL FTKAPI F_IniFile::getParam(
	const char *	pszParamName,
	FLMBOOL *		pbParamVal)		// Out: The value associated with name
{
	FLMBOOL			bFound = FALSE;
	INI_LINE *		pLine = NULL;
	
	f_assert( m_bReady);

	pLine = findParam( pszParamName);

	if( !pLine)
	{
		goto Exit;
	}

	if( !pLine->pszParamValue)
	{
		goto Exit;
	}
	
	fromAscii( pbParamVal, pLine->pszParamValue);
	bFound = TRUE;
	
Exit:

	return( bFound);
}

/****************************************************************************
Desc:	Stores a new value for the specified name (or creates a new name/value
	   pair) in the list of INI_STRUCTs
****************************************************************************/
RCODE FTKAPI F_IniFile::setParam(
	const char *	pszParamName,
	FLMBOOL			bParamVal)
{
	RCODE			rc = NE_FLM_OK;
	INI_LINE *	pLine;

	f_assert( m_bReady);

	// If the parameter exists in the list, just store the new value.
	// Othewise, create a new INI_LINE and add it to the list
	
	pLine = findParam( pszParamName);
	if( !pLine)
	{
		if (RC_BAD( rc = setParamCommon( &pLine, pszParamName)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = toAscii( &pLine->pszParamValue, bParamVal)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Retrieves the value associated with the specified name from the list
	   of INI_STRUCTs
****************************************************************************/
FLMBOOL FTKAPI F_IniFile::getParam(
	const char *	pszParamName,
	char **			ppszParamVal)
{
	FLMBOOL		bFound = FALSE;
	INI_LINE *	pLine = NULL;

	f_assert( m_bReady);
	*ppszParamVal = NULL;

	pLine = findParam( pszParamName);

	if( !pLine)
	{
		goto Exit;
	}

	if( pLine->pszParamValue == NULL)
	{
		goto Exit;
	}

	*ppszParamVal = pLine->pszParamValue;
	bFound = TRUE;

Exit:

	return( bFound);
}

/****************************************************************************
Desc:	Stores a new value for the specified name (or creates a new name/value
	   pair) in the list of INI_STRUCTs
****************************************************************************/
RCODE FTKAPI F_IniFile::setParam(
	const char *	pszParamName,
	const char *	pszParamVal)
{
	RCODE				rc = NE_FLM_OK;
	INI_LINE *		pLine;

	f_assert( m_bReady);

	// If the parameter exists in the list, just store the new value.
	// Othewise, create a new INI_LINE and add it to the list
	
	pLine = findParam( pszParamName);
	if( !pLine)
	{
		if( RC_BAD( rc = setParamCommon( &pLine, pszParamName)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = toAscii( &pLine->pszParamValue, pszParamVal)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Read a line from the ini file and store it in pszBuf
****************************************************************************/
RCODE F_IniFile::readLine(
	char *		pszBuf,
	FLMUINT *	puiBytes,
	FLMBOOL *	pbMore)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiBytesRead = 0;
	FLMUINT		uiBytesInLine = 0;
	FLMUINT		uiEOLBytes = 0;
	FLMBOOL		bEOL = FALSE;
		
	f_assert( m_pFileHdl);
	
	rc = m_pFileHdl->read( m_uiFileOffset, *puiBytes,
		(FLMBYTE *)pszBuf, &uiBytesRead);
	
	if ( RC_OK( rc) || rc == NE_FLM_IO_END_OF_FILE)
	{
		// Check to see if we got more than one line...
		
		while( !bEOL && (uiBytesInLine < uiBytesRead) )
		{
			if( pszBuf[ uiBytesInLine] == 13 || pszBuf[ uiBytesInLine] == 10)
			{
				// NOTE: If we end up reading the first byte of a CR/LF pair, but
				// but the second byte is read on the next call, then it will get
				// counted as a new (but empty) line.  We're not going to worry
				// about it though, because  empty lines end up getting ignored
				// and this isn't likely to happen often enough to effect
				// performance.
				
				bEOL = TRUE;
				*puiBytes = uiBytesInLine;
				uiEOLBytes=1;
				
				// Check for a CR/LF pair (or a LF/CR pair...)
				
				if( (uiBytesInLine + 1 < uiBytesRead) &&
					  (pszBuf[ uiBytesInLine + 1] == 13 ||
						 pszBuf[ uiBytesInLine + 1] == 10))
				{
					uiEOLBytes++;
				}
			}
			else
			{
				uiBytesInLine++;
			}
		}

		// Set the file position variable forward appropriately...
		
		m_uiFileOffset += uiBytesInLine + uiEOLBytes;
	}

	// If we read in more than one line, then don't want to return
	// NE_FLM_IO_END_OF_FILE...
	
	if( rc == NE_FLM_IO_END_OF_FILE && 
		(uiBytesInLine + uiEOLBytes) < uiBytesRead)
	{
		rc = NE_FLM_OK;
	}

	// Last step - update pbMore
	
	*pbMore = (bEOL || (uiBytesRead == 0))
					? FALSE
					: TRUE;
		
	return( rc);
}

/****************************************************************************
Desc:	Parse a single line from the ini file into its name, value and comment
	   parts.
****************************************************************************/
RCODE F_IniFile::parseBuffer(
	char *		pszBuf,
	FLMUINT		uiNumBytes)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiCurrentChar = 0;
	char *		pszNameStart = NULL;
	char *		pszNameEnd = NULL;
	char *		pszValStart = NULL;
	char *		pszValEnd = NULL;
	char *		pszCommentStart = NULL;
	INI_LINE *	pLine = NULL;
	FLMUINT		uiStrLen = 0;

	f_assert( pszBuf);
	f_assert( uiNumBytes);

	// Start looking for the parameter name...
	
	while (uiCurrentChar < uiNumBytes)
	{
		if( !isWhiteSpace( pszBuf[uiCurrentChar]))
		{
			if (pszBuf[uiCurrentChar] == '#') 
			{
				goto Comment;
			}
			else
			{
				pszNameStart = &pszBuf[uiCurrentChar];
				break;
			}
		}
		uiCurrentChar++;
	}

	// We've found a param name, now mark the end of it
	// We determine the end by looking for whitespace or '='
	// or '#'
	
	while (uiCurrentChar < uiNumBytes)
	{
		if( isWhiteSpace( pszBuf[uiCurrentChar]) ||
			  (pszBuf[uiCurrentChar] == '=') || 
			  (pszBuf[uiCurrentChar] == '#'))
		{
			pszNameEnd = &pszBuf[uiCurrentChar-1];
			break;
		}

		uiCurrentChar++;
	}

	if( (uiCurrentChar == uiNumBytes) && 
		  (pszNameEnd == NULL) )
	{
		pszNameEnd = &pszBuf[uiCurrentChar - 1];
	}

	// Now, there may be a value part or a comment part next.  If there's a
	// value, it had better be preceeded by an '='
	
	while( (uiCurrentChar < uiNumBytes) && 
			  isWhiteSpace( pszBuf[uiCurrentChar]) )
	{
		uiCurrentChar++;
	}
	
	if( uiCurrentChar < uiNumBytes && pszBuf[ uiCurrentChar] == '#')
	{
		goto Comment;
	}

	if( uiCurrentChar < uiNumBytes && pszBuf[uiCurrentChar] != '=' )
	{
		rc = RC_SET( NE_FLM_SYNTAX);
		goto Exit;
	}

	// Ok - at this point pszBuf[uiCurrentChar] contains an =.  Skip over
	// the = and any whitespace that follows.
	
	while( uiCurrentChar < uiNumBytes)
	{
		uiCurrentChar++;
		if( !isWhiteSpace( pszBuf[uiCurrentChar]))
		{
			pszValStart = &pszBuf[uiCurrentChar];
			break;
		}
	}

	// Now mark the end of the value.
	// We determine the end by looking for whitespace or '#'
	
	while( uiCurrentChar < uiNumBytes) 
	{
		if( isWhiteSpace( pszBuf[uiCurrentChar]) || 
			  (pszBuf[uiCurrentChar] == '#'))
		{
			pszValEnd = &pszBuf[uiCurrentChar-1];
			break;
		}		
		uiCurrentChar++;
	}

	if( uiCurrentChar == uiNumBytes && !pszValEnd)
	{
		pszValEnd = &pszBuf[uiCurrentChar-1];
	}

Comment:

	// Check out the rest of the line to see if there's a comment
	
	while( uiCurrentChar < uiNumBytes)
	{
		if( !isWhiteSpace( pszBuf[ uiCurrentChar]) &&
			 pszBuf[ uiCurrentChar] != '#')
		{
			rc = RC_SET( NE_FLM_SYNTAX);
			goto Exit;
		}
		else if( pszBuf[ uiCurrentChar] == '#')
		{
			// Comment found.  Set pszCommentStart to the next char
			
			pszCommentStart = &pszBuf[uiCurrentChar+1];
			break;
		}
		uiCurrentChar++;
	}

	// Done parsing.  Now, assuming the line had any info in it,
	// store all the strings...
	
	if( pszNameStart || pszCommentStart)
	{
		if( RC_BAD( rc = m_pool.poolCalloc( sizeof( INI_LINE),
										(void **)&pLine)))
		{
			goto Exit;
		}
		
		if( pszNameStart)
		{
			uiStrLen = pszNameEnd - pszNameStart + 1;
			if( RC_BAD( rc = m_pool.poolAlloc( uiStrLen + 1,
								(void **)&pLine->pszParamName)))
			{
				goto Exit;
			}
			
			f_memcpy( pLine->pszParamName, pszNameStart, uiStrLen);
			pLine->pszParamName[uiStrLen] = '\0';
		}

		if( pszValStart)
		{
			uiStrLen = pszValEnd - pszValStart + 1;
			if( RC_BAD( rc = m_pool.poolAlloc( uiStrLen + 1,
						(void **)&pLine->pszParamValue)))
			{
				goto Exit;
			}
			
			f_memcpy(pLine->pszParamValue, pszValStart, uiStrLen);
			pLine->pszParamValue[uiStrLen] = '\0';
		}
		
		if (pszCommentStart)
		{
			uiStrLen = uiNumBytes-(pszCommentStart-pszBuf);
			if (RC_BAD( rc = m_pool.poolAlloc( uiStrLen + 1,
										(void **)&pLine->pszComment)))
			{
				goto Exit;
			}
			
			f_memcpy(pLine->pszComment, pszCommentStart, uiStrLen);
			pLine->pszComment[uiStrLen] = '\0';
		}
		
		// Insert this struct into the linked list
		
		if( m_pLastLine)
		{
			m_pLastLine->pNext = pLine;
		}
		
		pLine->pPrev = m_pLastLine;
		pLine->pNext = NULL;
		m_pLastLine = pLine;
		
		if( !m_pFirstLine)
		{
			m_pFirstLine = pLine;
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Search through the list of INI_STRUCTs for a particular name
****************************************************************************/
INI_LINE * F_IniFile::findParam(
	const char *	pszParamName)
{
	INI_LINE *	pCurLine = m_pFirstLine;
	
	while( pCurLine)
	{
		if (pCurLine->pszParamName)
		{
			if (f_strcmp( pszParamName, pCurLine->pszParamName) == 0)
			{
				return pCurLine;
			}
		}

		pCurLine = pCurLine->pNext;
	}

	return( NULL);
}

/****************************************************************************
Desc:	This code is common to all of the SetParam() functions
****************************************************************************/
RCODE F_IniFile::setParamCommon( 
	INI_LINE **		ppLine,
	const char *	pszParamName)
{
	RCODE				rc = NE_FLM_OK;
	INI_LINE *		pLine;

	if( RC_BAD( rc = m_pool.poolCalloc( 
		sizeof( INI_LINE), (void **)&pLine)))
	{
		goto Exit;
	}
	
	if( m_pLastLine)
	{
		m_pLastLine->pNext = pLine;
	}
	
	pLine->pPrev = m_pLastLine;
	m_pLastLine = pLine;
	
	if( !m_pFirstLine)
	{
		m_pFirstLine = pLine;
	}

	if( RC_BAD( rc = m_pool.poolAlloc( f_strlen(pszParamName)+1,
								(void **)&pLine->pszParamName)))
	{
		goto Exit;
	}

	f_strcpy( pLine->pszParamName, pszParamName);

Exit:

	if( RC_OK( rc))
	{
		*ppLine = pLine;
	}

	return( rc);
}

/****************************************************************************
Desc:	All of the fromAscii() functions convert values stored in strings to
      various native formats
****************************************************************************/
void F_IniFile::fromAscii( 
	FLMUINT *		puiVal,
	const char *	pszBuf)
{
	FLMUINT		uiValue;
	FLMBOOL		bAllowHex = FALSE;

	if( *pszBuf == '0' &&
		(*(pszBuf + 1) == 'x' || *(pszBuf + 1) == 'X'))
	{
		pszBuf += 2;
		bAllowHex = TRUE;
	}

	uiValue = 0;
	while( *pszBuf)
	{
		if( *pszBuf >= '0' && *pszBuf <= '9')
		{
			if( !bAllowHex)
			{
				uiValue *= 10;
			}
			else
			{
				uiValue <<= 4;
			}

			uiValue += (FLMUINT)(*pszBuf - '0');
		}
		else if( bAllowHex)
		{
			if( *pszBuf >= 'A' && *pszBuf <= 'F')
			{
				uiValue <<= 4;
				uiValue += (FLMUINT)(*pszBuf - 'A') + 10;
			}
			else if( *pszBuf >= 'a' && *pszBuf <= 'f')
			{
				uiValue <<= 4;
				uiValue += (FLMUINT)(*pszBuf - 'a') + 10;
			}
			else
			{
				break;
			}
		}
		else
		{
			break;
		}
		pszBuf++;
	}
	
	*puiVal = uiValue;
}

/****************************************************************************
Desc:	All of the fromAscii() functions convert values stored in strings to
      various native formats
****************************************************************************/
void F_IniFile::fromAscii( 
	FLMBOOL *		pbVal, 
	const char *	pszParamValue)
{
	if( f_stricmp( pszParamValue, "true") == 0 ||
		 f_stricmp( pszParamValue, "enabled") == 0 ||
		 f_stricmp( pszParamValue, "on") == 0 ||
		 f_stricmp( pszParamValue, "1") == 0)
	{
		*pbVal = TRUE;
	}
	else
	{
		*pbVal = FALSE;
	}
}

/****************************************************************************
Desc:	All of the toAscii() functions convert values from their native
		formats to a string representation
****************************************************************************/
RCODE F_IniFile::toAscii( 
	char **		ppszParamValue,
	FLMUINT 		puiVal)
{
	RCODE			rc = NE_FLM_OK;
	char 			szTemp[ 50];

	f_sprintf( szTemp, "%*.*lu", sizeof(szTemp), sizeof(szTemp), puiVal);

	if( RC_BAD( rc = m_pool.poolAlloc( f_strlen( szTemp),
		(void **)ppszParamValue)))
	{
		goto Exit;
	}
	
	f_strcpy( *ppszParamValue, szTemp);
	m_bModified = TRUE;
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	All of the toAscii() functions convert values from their native
		formats to a string representation
****************************************************************************/
RCODE F_IniFile::toAscii( 
	char **		ppszParamValue,
	FLMBOOL 		bVal)
{
	RCODE		rc = NE_FLM_OK;
	
	if( RC_BAD( rc = m_pool.poolAlloc( 6, (void **)ppszParamValue)))
	{
		goto Exit;
	}
	
	if( bVal)
	{
		f_memcpy( *ppszParamValue, "TRUE ", 6);
	}
	else
	{
		f_memcpy( *ppszParamValue, "FALSE", 6);
	}
	
	m_bModified = TRUE;
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	All of the toAscii() functions convert values from their native
		formats to a string representation
****************************************************************************/
RCODE F_IniFile::toAscii( 
	char **			ppszParamValue,
	const char * 	pszVal)
{
	RCODE		rc = NE_FLM_OK;
	
	if( RC_BAD( rc = m_pool.poolAlloc( f_strlen( pszVal),
							(void **)ppszParamValue)))
	{
		goto Exit;
	}
	
	f_strcpy( *ppszParamValue, pszVal);
	m_bModified = TRUE;
	
Exit:

	return( rc);
}
