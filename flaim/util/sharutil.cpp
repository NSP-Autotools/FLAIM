//-------------------------------------------------------------------------
// Desc:	Utility routines shared among various utilities.
// Tabs:	3
//
// Copyright (c) 1997-2003, 2005-2007 Novell, Inc. All Rights Reserved.
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

#include "sharutil.h"

FSTATIC RCODE propertyExists(
	const char *		pszProperty,
	const char *		pszBuffer,
	char **				ppszValue);

/********************************************************************
Desc: Parses command-line parameters
*********************************************************************/
void flmUtilParseParams(
	char *			pszCommandBuffer,
	FLMINT			iMaxArgs,
	FLMINT *			iArgcRV,
	const char **	ppArgvRV)
{
	FLMINT			iArgC = 0;

	for (;;)
	{
		while( (*pszCommandBuffer == ' ') || (*pszCommandBuffer == '\t'))
		{
			pszCommandBuffer++;
		}
		
		if( *pszCommandBuffer == 0)
		{
			break;
		}

		if( *pszCommandBuffer == '"' || *pszCommandBuffer == '\'')
		{
			char	ucQuoteChar = *pszCommandBuffer;

			pszCommandBuffer++;
			ppArgvRV[ iArgC] = pszCommandBuffer;
			iArgC++;
			
			while( *pszCommandBuffer && *pszCommandBuffer != ucQuoteChar)
			{
				pszCommandBuffer++;
			}
			
			if( *pszCommandBuffer)
			{
				*pszCommandBuffer++ = 0;
			}
		}
		else
		{
			ppArgvRV[ iArgC] = pszCommandBuffer;
			iArgC++;
			
			while( *pszCommandBuffer && *pszCommandBuffer != ' ' &&
					 *pszCommandBuffer != '\t')
			{
				pszCommandBuffer++;
			}
			
			if( *pszCommandBuffer)
			{
				*pszCommandBuffer++ = 0;
			}
		}

		// Quit if we have reached the maximum allowable number of arguments

		if( iArgC == iMaxArgs)
		{
			break;
		}
	}
	
	*iArgcRV = iArgC;
}

/****************************************************************************
Desc:	a vector set item operation.  
****************************************************************************/
#define FLMVECTOR_START_AMOUNT 16
#define FLMVECTOR_GROW_AMOUNT 2
RCODE FlmVector::setElementAt( void * pData, FLMUINT uiIndex)
{
	RCODE rc = FERR_OK;
	if ( !m_pElementArray)
	{		
		TEST_RC( rc = f_calloc( sizeof( void*) * FLMVECTOR_START_AMOUNT,
			&m_pElementArray));
		m_uiArraySize = FLMVECTOR_START_AMOUNT;
	}

	if ( uiIndex >= m_uiArraySize)
	{		
		TEST_RC( rc = f_recalloc(
			sizeof( void*) * m_uiArraySize * FLMVECTOR_GROW_AMOUNT,
			&m_pElementArray));
		m_uiArraySize *= FLMVECTOR_GROW_AMOUNT;
	}

	m_pElementArray[ uiIndex] = pData;
Exit:
	return rc;
}

/****************************************************************************
Desc:	a vector get item operation
****************************************************************************/
void * FlmVector::getElementAt( FLMUINT uiIndex)
{
	//if you hit this you are indexing into the vector out of bounds.
	//unlike a real array, we can catch this here!  oh joy!
	flmAssert ( uiIndex < m_uiArraySize);	
	return m_pElementArray[ uiIndex];
}

/****************************************************************************
Desc:	append a char (or the same char many times) to the string
****************************************************************************/
RCODE FlmStringAcc::appendCHAR( char ucChar, FLMUINT uiHowMany)
{
	RCODE rc = FERR_OK;
	if ( uiHowMany == 1)
	{
		char 	szStr[ 2];
		
		szStr[ 0] = ucChar;
		szStr[ 1] = 0;
		
		rc = this->appendTEXT( szStr);
	}
	else
	{
		char * pszStr;

		if( RC_BAD( rc = f_alloc( uiHowMany + 1, &pszStr)))
		{
			goto Exit;
		}
		
		f_memset( pszStr, ucChar, uiHowMany);
		pszStr[ uiHowMany] = 0;
		rc = this->appendTEXT( pszStr);
		f_free( &pszStr);
	}
Exit:
	return rc;
}

/****************************************************************************
Desc:	appending text to the accumulator safely.  all other methods in
		the class funnel through this one, as this one contains the logic
		for making sure storage requirements are met.
****************************************************************************/
RCODE FlmStringAcc::appendTEXT( 
	const char * 	pszVal)
{	
	RCODE 			rc = FERR_OK;
	FLMUINT 			uiIncomingStrLen;
	FLMUINT 			uiStrLen;

	//be forgiving if they pass in a NULL
	if ( !pszVal)
	{
		goto Exit;
	}
	//also be forgiving if they pass a 0-length string
	else if ( (uiIncomingStrLen = f_strlen( pszVal)) == 0)
	{
		goto Exit;
	}
	
	//compute total size we need to store the new total
	if ( m_bQuickBufActive || m_pszVal)
	{
		uiStrLen = uiIncomingStrLen + m_uiValStrLen;
	}
	else
	{
		uiStrLen = uiIncomingStrLen;
	}

	//just use small buffer if it's small enough
	if ( uiStrLen < FSA_QUICKBUF_BUFFER_SIZE)
	{
		f_strcat( m_szQuickBuf, pszVal);
		m_bQuickBufActive = TRUE;
	}
	//we are exceeding the quickbuf size, so get the bytes from the heap
	else
	{
		//ensure storage requirements are met (and then some)
		if ( m_pszVal == NULL)
		{
			FLMUINT uiNewBytes = (uiStrLen+1) * 4;
			if ( RC_OK ( rc = f_alloc(
				(FLMUINT)(sizeof( FLMBYTE) * uiNewBytes),
				&m_pszVal)))
			{
				m_uiBytesAllocatedForPszVal = uiNewBytes;
				m_pszVal[ 0] = 0;
			}
			else
			{
				goto Exit;
			}
		}
		else if ( (m_uiBytesAllocatedForPszVal-1) < uiStrLen)
		{
			FLMUINT uiNewBytes = (uiStrLen+1) * 4;
			if ( RC_OK( rc = f_realloc(
				(FLMUINT)(sizeof( FLMBYTE) * uiNewBytes),
				&m_pszVal)))
			{
				m_uiBytesAllocatedForPszVal = uiNewBytes;
			}
			else
			{
				goto Exit;
			}
		}

		//if transitioning from quick buf to heap buf, we need to
		//transfer over the quick buf contents and unset the flag
		if ( m_bQuickBufActive)
		{
			m_bQuickBufActive = FALSE;
			f_strcpy( m_pszVal, m_szQuickBuf);
			//no need to zero out m_szQuickBuf because it will never
			//be used again, unless a clear() is issued, in which
			//case it will be zeroed out then.
		}		

		//copy over the string
		f_strcat( m_pszVal, pszVal);
	}
	m_uiValStrLen = uiStrLen;
Exit:
	return rc;
}

/****************************************************************************
Desc:	printf into the FlmStringAcc
****************************************************************************/
RCODE FlmStringAcc::printf(
	const char * pszFormatString,
	...)
{
	f_va_list		args;
	char *			pDestStr = NULL;
	RCODE				rc = FERR_OK;

	if( RC_BAD( rc = f_alloc( 4096, &pDestStr)))
	{
		goto Exit;
	}

	f_va_start( args, pszFormatString);
	f_vsprintf( pDestStr, pszFormatString, &args);
	f_va_end( args);

	this->clear();
	TEST_RC( rc = this->appendTEXT( pDestStr));

Exit:
	if ( pDestStr)
	{
		f_free( &pDestStr);
	}
	return rc;
}

/****************************************************************************
Desc:	formatted appender like sprintf
****************************************************************************/
RCODE FlmStringAcc::appendf(
	const char * pszFormatString,
	...)
{
	f_va_list		args;
	char *			pDestStr = NULL;
	RCODE				rc = FERR_OK;

	if( RC_BAD( rc = f_alloc( 8192, &pDestStr)))
	{
		goto Exit;
	}

	f_va_start( args, pszFormatString);
	f_vsprintf( pDestStr, pszFormatString, &args);
	f_va_end( args);

	TEST_RC( rc = this->appendTEXT( pDestStr));

Exit:
	if ( pDestStr)
	{
		f_free( &pDestStr);
	}
	return rc;
}
	

/****************************************************************************
Desc:	callback to use to output a line
****************************************************************************/
void utilOutputLine(
	const char * 		pszData, 
	void * 				pvUserData)
{
	FTX_WINDOW * 		pMainWindow = (FTX_WINDOW*)pvUserData;
	eColorType	 		uiBack;
	eColorType			uiFore;
	
	FTXWinGetBackFore( pMainWindow, &uiBack, &uiFore);
	FTXWinCPrintf( pMainWindow, uiBack, uiFore, "%s\n", pszData);
}

/****************************************************************************
Desc:	callback to serve as a 'pager' function when the Usage: help
		is too long to fit on one screen.
****************************************************************************/ 
void utilPressAnyKey(
	const char * 		pszMessage,
	void * 				pvUserData)
{
	FTX_WINDOW * 		pMainWindow = (FTX_WINDOW*)pvUserData;
	FLMUINT 				uiChar;
	eColorType			uiBack;
	eColorType			uiFore;
	
	FTXWinGetBackFore( pMainWindow, &uiBack, &uiFore);
	FTXWinCPrintf( pMainWindow, uiBack, uiFore, (char*)pszMessage);
	
	while( RC_BAD( FTXWinTestKB( pMainWindow)))
	{
		f_sleep( 100); //don't hog the cpu
	}
	FTXWinCPrintf( pMainWindow, uiBack, uiFore,
		"\r                                                                  ");
	FTXWinCPrintf( pMainWindow, uiBack, uiFore, "\r");
	FTXWinInputChar( pMainWindow, &uiChar);
}

/****************************************************************************
Desc:	routine to startup the TUI
****************************************************************************/
RCODE utilInitWindow(
	const char *	pszTitle,
	FLMUINT *		puiScreenRows,
	FTX_WINDOW **	ppMainWindow,
	FLMBOOL *		pbShutdown)
{
	FTX_SCREEN *	pScreen = NULL;
	FTX_WINDOW *	pTitleWin = NULL;
	FLMUINT			uiCols;
	int				iResCode = 0;

	if( RC_BAD( FTXInit( pszTitle, 80, 50, FLM_BLUE, FLM_WHITE, NULL, NULL)))
	{
		iResCode = 1;
		goto Exit;
	}

	FTXSetShutdownFlag( pbShutdown);

	if( RC_BAD( FTXScreenInit( pszTitle, &pScreen)))
	{
		iResCode = 1;
		goto Exit;
	}
	
	FTXScreenGetSize( pScreen, &uiCols, puiScreenRows);

	if( RC_BAD( FTXScreenInitStandardWindows( pScreen, FLM_RED, FLM_WHITE,
		FLM_BLUE, FLM_WHITE, FALSE, FALSE, pszTitle,
		&pTitleWin, ppMainWindow)))
	{
		iResCode = 1;
		goto Exit;
	}
	
Exit:
	return (RCODE)iResCode;
}

/****************************************************************************
Desc:	routine to shutdown the TUI
****************************************************************************/
void utilShutdownWindow( void)
{
	FTXExit();
}
	
/****************************************************************************
Desc:	fill a buffer with the current (or given) time
****************************************************************************/
FLMUINT utilGetTimeString(
	char *		pszOutString,
	FLMUINT		uiBufferSize,
	FLMUINT		uiInSeconds)
{
	F_TMSTAMP	timeStamp;
	FLMUINT		uiSeconds;

	if( uiInSeconds != 0)
	{
		f_timeSecondsToDate( uiInSeconds, &timeStamp);
	}
	else
	{
		f_timeGetTimeStamp( &timeStamp);
	}
	f_timeDateToSeconds( &timeStamp, &uiSeconds);
	char pszTemp[ 256];
	f_sprintf( pszTemp,
		"%4u-%02u-%02u %02u:%02u:%02u",
		(unsigned)timeStamp.year,
		(unsigned)(timeStamp.month + 1),
		(unsigned)timeStamp.day,
		(unsigned)timeStamp.hour,
		(unsigned)timeStamp.minute,
		(unsigned)timeStamp.second);
	f_strncpy( pszOutString, pszTemp, uiBufferSize - 1);
	pszOutString[ uiBufferSize-1] = 0;
	return uiSeconds;
}

#define UTIL_PROP_DELIMITER '!'
FSTATIC RCODE propertyExists(
	const char *	pszProperty,
	const char *	pszBuffer,
	char **			ppszValue)
{
	flmAssert( pszProperty);
	FlmStringAcc acc;
	RCODE rc = FERR_OK; //returns only memory errors

	*ppszValue = NULL;

	if ( !pszBuffer)
	{
		goto Exit;
	}
	else
	{
		char * 		pszValue;
		
		acc.appendf( "%s%c", pszProperty, UTIL_PROP_DELIMITER);
		
		if ( (pszValue = f_strstr( pszBuffer, pszProperty)) != NULL)
		{
			pszValue = (char *)(1 + f_strchr( pszValue, UTIL_PROP_DELIMITER));
			
			 if( RC_BAD( rc = f_strdup( pszValue, ppszValue)))
			 {
				 goto Exit;
			 }
			
			(f_strchr( *ppszValue, '\n'))[ 0] = 0; 
		}
		else
		{
			goto Exit;
		}
	}
Exit:
	return rc;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE utilWriteProperty(
	const char *		pszFile,
	const char *		pszProp,
	const char *		pszValue)
{
	RCODE					rc = FERR_OK;
	char *				pszContents = NULL;
	char *				pszExistingProperty;
	FlmStringAcc		newContents;
	IF_FileSystem *	pFileSystem = NULL;

	//can't have newlines in the props or values
	
	flmAssert( !f_strchr( pszProp, '\n'));
	flmAssert( !f_strchr( pszValue, '\n'));
	
	if( RC_BAD( rc = FlmGetFileSystem( &pFileSystem)))
	{
		goto Exit;
	}

	if( RC_BAD( pFileSystem->doesFileExist( pszFile)))
	{
		//add trailing newline
		TEST_RC( rc = f_filecpy( pszFile, "")); 
	}
	
	if ( RC_BAD( f_filetobuf( pszFile, &pszContents)))
	{
		goto Exit;
	}
	
	//propertyExists returns out a new
	TEST_RC( rc = propertyExists( pszProp, pszContents, &pszExistingProperty));
	
	if ( !pszExistingProperty)
	{
		newContents.appendf( "%s%c%s\n", pszProp, UTIL_PROP_DELIMITER, pszValue);
		newContents.appendTEXT( pszContents);
	}
	else
	{
		f_free( &pszExistingProperty);
		FLMUINT uiProps = 0;
		
		//write out nulls in place of the "\n"'s throughout the contents
		
		char *		pszNuller = pszContents;
		
		for( ;;)
		{
			pszNuller = f_strchr( pszNuller, '\n');
			
			if( pszNuller)
			{
				pszNuller[ 0] = 0;
				pszNuller++;
				uiProps++;
			}
			else
			{
				break;
			}
		}
		
		char * 		pszNextLine = pszContents;
		char * 		pszNextProp;
		char * 		pszNextVal;
		char * 		pszBang;
		
		while ( uiProps--)
		{
			pszBang = f_strchr( pszNextLine, UTIL_PROP_DELIMITER);
			flmAssert( pszBang);
			
			pszBang[ 0] = 0;
			pszNextProp = pszNextLine;
			pszNextVal = pszBang + 1;
			
			if( f_strcmp( pszNextProp, pszProp) != 0)
			{
				pszBang[ 0] = UTIL_PROP_DELIMITER;
				newContents.appendTEXT( pszNextLine);
				newContents.appendCHAR( '\n');
			}
			else
			{				
				newContents.appendf( "%s%c%s\n",
					pszNextProp, UTIL_PROP_DELIMITER, pszValue);
				pszBang[ 0] = UTIL_PROP_DELIMITER;
			}
			
			pszNextLine = pszNextLine + f_strlen( pszNextLine) + 1;
		}
	}
	
	rc = f_filecpy( pszFile, newContents.getTEXT());
	
Exit:

	if( pszContents)
	{
		f_free( &pszContents);
	}
	
	if( pFileSystem)
	{
		pFileSystem->Release();
	}
	
	return( rc); 
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE utilReadProperty(
	const char *		pszFile,
	const char *		pszProp,
	FlmStringAcc *		pAcc)
{
	RCODE					rc = FERR_OK;
	char *				pszContents = NULL;
	char *				pszValue = NULL;
	IF_FileSystem *	pFileSystem = NULL;
	
	if( RC_BAD( rc = FlmGetFileSystem( &pFileSystem)))
	{
		goto Exit;
	}
	
	if( RC_BAD( pFileSystem->doesFileExist( pszFile)))
	{
		goto Exit;
	}
	
	if ( RC_BAD( f_filetobuf( pszFile, &pszContents)))
	{
		goto Exit;
	}
	
	TEST_RC( rc = propertyExists( pszProp, pszContents, &pszValue));
	TEST_RC( rc = pAcc->appendTEXT( pszValue));

Exit:

	if( pszValue)
	{
		f_free( &pszValue);
	}
	
	if( pszContents)
	{
		f_free( &pszContents);
	}
	
	if( pFileSystem)
	{
		pFileSystem->Release();
	}
	
	return( rc); 
}
