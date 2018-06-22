//------------------------------------------------------------------------------
//	Desc:	Shared utility routines
// Tabs:	3
//
// Copyright (c) 1997-2007 Novell, Inc. All Rights Reserved.
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
#include "sharutil.h"

FSTATIC RCODE FTKAPI _flmWrapperFunc(
	IF_Thread *		pThread);

/********************************************************************
Desc: Parses command-line parameters
*********************************************************************/
void flmUtilParseParams(
	char *		pszCommandBuffer,
	FLMINT		iMaxArgs,
	FLMINT *		iArgcRV,
	char **		ppArgvRV)
{
	FLMINT	iArgC = 0;

	for (;;)
	{
		/* Strip off leading white space. */

		while ((*pszCommandBuffer == ' ') || (*pszCommandBuffer == '\t'))
			pszCommandBuffer++;
		if (!(*pszCommandBuffer))
			break;

		if ((*pszCommandBuffer == '"') || (*pszCommandBuffer == '\''))
		{
			char cQuoteChar = *pszCommandBuffer;

			pszCommandBuffer++;
			ppArgvRV [iArgC] = pszCommandBuffer;
			iArgC++;
			while ((*pszCommandBuffer) && (*pszCommandBuffer != cQuoteChar))
				pszCommandBuffer++;
			if (*pszCommandBuffer)
				*pszCommandBuffer++ = 0;
		}
		else
		{
			ppArgvRV [iArgC] = pszCommandBuffer;
			iArgC++;
			while ((*pszCommandBuffer) &&
					 (*pszCommandBuffer != ' ') &&
					 (*pszCommandBuffer != '\t'))
				pszCommandBuffer++;
			if (*pszCommandBuffer)
				*pszCommandBuffer++ = 0;
		}

		/* Quit if we have reached the maximum allowable number of arguments. */

		if (iArgC == iMaxArgs)
			break;
	}
	*iArgcRV = iArgC;
}

/****************************************************************************
Name:	FlmVector::setElementAt
Desc:	a vector set item operation.  
****************************************************************************/
#define FLMVECTOR_START_AMOUNT 16
#define FLMVECTOR_GROW_AMOUNT 2
RCODE FlmVector::setElementAt( void * pData, FLMUINT uiIndex)
{
	RCODE rc = NE_XFLM_OK;
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
Name:	FlmVector::getElementAt
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
Name:	FlmStringAcc::appendCHAR
Desc:	append a char (or the same char many times) to the string
****************************************************************************/
RCODE FlmStringAcc::appendCHAR( char ucChar, FLMUINT uiHowMany)
{
	RCODE rc = NE_XFLM_OK;
	if ( uiHowMany == 1)
	{
		FLMBYTE szStr[ 2];
		szStr[ 0] = ucChar;
		szStr[ 1] = 0;
		rc = this->appendTEXT( (const FLMBYTE*)szStr);
	}
	else
	{
		FLMBYTE * pszStr;
		
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
Name:	FlmStringAcc::appendTEXT
Desc:	appending text to the accumulator safely.  all other methods in
		the class funnel through this one, as this one contains the logic
		for making sure storage requirements are met.
****************************************************************************/
RCODE FlmStringAcc::appendTEXT( const FLMBYTE * pszVal)
{	
	RCODE 			rc = NE_XFLM_OK;
	FLMUINT 			uiIncomingStrLen;
	FLMUINT 			uiStrLen;

	//be forgiving if they pass in a NULL
	if ( !pszVal)
	{
		goto Exit;
	}
	//also be forgiving if they pass a 0-length string
	else if( (uiIncomingStrLen = f_strlen( (const char *)pszVal)) == 0)
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
		f_strcat( m_szQuickBuf, (const char *)pszVal);
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
		f_strcat( m_pszVal, (const char *)pszVal);
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
	FLMSIZET		iSize = 4096;
	RCODE				rc = NE_XFLM_OK;

	if( RC_BAD( rc = f_alloc( iSize, &pDestStr)))
	{
		goto Exit;
	}

	f_va_start( args, pszFormatString);
	f_vsprintf( pDestStr, pszFormatString, &args);
	f_va_end( args);

	this->clear();
	TEST_RC( rc = this->appendTEXT( (FLMBYTE *)pDestStr));

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
	FLMSIZET			iSize = 4096;
	RCODE				rc = NE_XFLM_OK;

	if( RC_BAD( rc = f_alloc( iSize, &pDestStr)))
	{
		goto Exit;
	}

	f_va_start( args, pszFormatString);
	f_vsprintf( pDestStr, pszFormatString, &args);
	f_va_end( args);

	TEST_RC( rc = this->appendTEXT( (FLMBYTE *)pDestStr));

Exit:

	if ( pDestStr)
	{
		f_free( &pDestStr);
	}
	return rc;
}
	
/****************************************************************************
Desc:	Constructor
*****************************************************************************/
FlmContext::FlmContext()
{
	m_szCurrDir[ 0] = '\0';
	m_hMutex = F_MUTEX_NULL;
	m_bIsSetup = FALSE;
}

/****************************************************************************
Desc:	Destructor
*****************************************************************************/
FlmContext::~FlmContext( void)
{
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
		m_hMutex = F_MUTEX_NULL;
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmContext::setup(
	FLMBOOL		bShared)
{
	RCODE		rc = NE_XFLM_OK;

	if( bShared)
	{
		if( RC_BAD( rc = f_mutexCreate( &m_hMutex)))
		{
			goto Exit;
		}
	}

	m_bIsSetup = TRUE;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmContext::setCurrDir(
	FLMBYTE *	pszCurrDir)
{
	RCODE		rc = NE_XFLM_OK;

	flmAssert( m_bIsSetup);

	lock();
	f_strcpy( (char *)m_szCurrDir, (const char *)pszCurrDir);
	unlock();

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmContext::getCurrDir(
	FLMBYTE *	pszCurrDir)
{
	RCODE		rc = NE_XFLM_OK;

	flmAssert( m_bIsSetup);

	lock();
	f_strcpy( (char *)pszCurrDir, (const char *)m_szCurrDir);
	unlock();

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmContext::lock( void)
{
	flmAssert( m_bIsSetup);

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexLock( m_hMutex);
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmContext::unlock( void)
{
	flmAssert( m_bIsSetup);

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( m_hMutex);
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FlmThreadContext::FlmThreadContext( void)
{
	m_pScreen = NULL;
	m_pWindow = NULL;
	m_bShutdown = FALSE;
	m_pLocalContext = NULL;
	m_pSharedContext = NULL;
	m_pNext = NULL;
	m_pPrev = NULL;
	m_uiID = 0;
	m_hMutex = F_MUTEX_NULL;
	m_pThrdFunc = NULL;
	m_pvAppData = NULL;
	m_pThread = NULL;
	m_bFuncExited = FALSE;
}

/****************************************************************************
Desc:
*****************************************************************************/
FlmThreadContext::~FlmThreadContext( void)
{
	// Free the local context
	if( m_pLocalContext)
	{
		m_pLocalContext->Release();
	}

	// Destroy the semaphore
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmThreadContext::lock( void)
{
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexLock( m_hMutex);
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmThreadContext::unlock( void)
{
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( m_hMutex);
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmThreadContext::setup(
	FlmSharedContext *	pSharedContext,
	const char *			pszThreadName,
	THREAD_FUNC_p			pFunc,
	void *					pvAppData)
{
	RCODE		rc = NE_XFLM_OK;

	flmAssert( pSharedContext != NULL);

	m_pSharedContext = pSharedContext;
	m_pThrdFunc = pFunc;
	m_pvAppData = pvAppData;

	if( (m_pLocalContext = f_new FlmContext) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = f_mutexCreate( &m_hMutex)))
	{
		goto Exit;
	}

	if( pszThreadName &&
		f_strlen( pszThreadName) <= MAX_THREAD_NAME_LEN)
	{
		f_strcpy( m_szName, pszThreadName);
	}
	else
	{
		f_sprintf( m_szName, "flmGenericThread");
	}

	if( RC_BAD( rc = m_pLocalContext->setup( FALSE)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmThreadContext::getName(
	char *		pszName,
	FLMBOOL		bLocked)
{
	if( !bLocked)
	{
		lock();
	}

	f_strcpy( pszName, m_szName);

	if( !bLocked)
	{
		unlock();
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmThreadContext::execute( void)
{
	flmAssert( m_pThrdFunc != NULL);
	m_FuncRC = (RCODE)m_pThrdFunc( this, m_pvAppData);
	return m_FuncRC;
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmThreadContext::shutdown()
{
	m_bShutdown = TRUE;
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmThreadContext::exec( void)
{
	flmAssert( m_pThrdFunc != NULL);
	return( (RCODE)(m_pThrdFunc( this, m_pvAppData)));
}

/****************************************************************************
Desc:
*****************************************************************************/
FlmSharedContext::FlmSharedContext( void)
{
	m_pParentContext = NULL;
	m_pThreadList = NULL;
	m_bLocalShutdownFlag = FALSE;
	m_pbShutdownFlag = &m_bLocalShutdownFlag;
	m_hSem = F_SEM_NULL;
	m_uiNextProcID = 1;
	m_bPrivateShare = FALSE;
}

/****************************************************************************
Desc:
*****************************************************************************/
FlmSharedContext::~FlmSharedContext( void)
{
	// Clean up the thread list
	shutdown();

	// Free the ESem
	if( m_hSem != F_SEM_NULL)
	{
		f_semDestroy( &m_hSem);
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmSharedContext::init(
	FlmSharedContext *	pSharedContext)
{
	RCODE		rc = NE_XFLM_OK;

	// Initialize the base class
	if( RC_BAD( rc = FlmContext::setup( TRUE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_semCreate( &m_hSem)))
	{
		goto Exit;
	}

	m_pParentContext = pSharedContext;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmSharedContext::shutdown( void)
{
	FLMBOOL	bLocked = FALSE;

	*m_pbShutdownFlag = TRUE;

	for( ;;)
	{
		lock();
		bLocked = TRUE;
		if( m_pThreadList)
		{
			m_pThreadList->shutdown();
		}
		else
		{
			break;
		}
		unlock();
		bLocked = FALSE;
		(void)f_semWait( m_hSem, 1000);
	}

	if( bLocked)
	{
		unlock();
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmSharedContext::wait( void)
{
	FLMBOOL	bLocked = FALSE;

	for( ;;)
	{
		lock();
		bLocked = TRUE;
		if( !m_pThreadList)
		{
			break;
		}
		unlock();
		bLocked = FALSE;
		(void)f_semWait( m_hSem, 1000);
	}

	if( bLocked)
	{
		unlock();
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmSharedContext::spawn(
	FlmThreadContext *	pThread,
	FLMUINT *				puiThreadID)
{
	RCODE						rc = NE_XFLM_OK;
	char						szName[ MAX_THREAD_NAME_LEN + 1];
	IF_ThreadMgr *			pThreadMgr = NULL;

	registerThread( pThread);
	pThread->getName( szName);
	
	if( RC_BAD( rc = FlmGetThreadMgr( &pThreadMgr)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pThreadMgr->createThread( NULL,
		_flmWrapperFunc, szName, 0, 0, pThread)))
	{
		goto Exit;
	}

	if( puiThreadID)
	{
		*puiThreadID = pThread->getID();
	}

Exit:

	if( pThreadMgr)
	{
		pThreadMgr->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmSharedContext::spawn(
	char *					pszThreadName,
	THREAD_FUNC_p			pFunc,
	void *					pvUserData,
	FLMUINT *				puiThreadID)
{
	FlmThreadContext *	pThread;
	RCODE						rc = NE_XFLM_OK;

	if( (pThread = f_new FlmThreadContext) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pThread->setup( this, pszThreadName, pFunc, pvUserData)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = spawn( pThread, puiThreadID)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmSharedContext::registerThread(
	FlmThreadContext *		pThread)
{
	RCODE		rc = NE_XFLM_OK;

	lock();
	pThread->setNext( m_pThreadList);
	if( m_pThreadList)
	{
		m_pThreadList->setPrev( pThread);
	}
	m_pThreadList = pThread;
	pThread->setID( m_uiNextProcID++);
	unlock();

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmSharedContext::deregisterThread(
	FlmThreadContext *		pThread)
{
	FlmThreadContext *	pTmpThrd;
	RCODE						rc = NE_XFLM_OK;

	lock();
	pTmpThrd = m_pThreadList;
	while( pTmpThrd)
	{
		if( pTmpThrd == pThread)
		{
			if( pTmpThrd->getPrev())
			{
				pTmpThrd->getPrev()->setNext( pTmpThrd->getNext());
			}

			if( pTmpThrd->getNext())
			{
				pTmpThrd->getNext()->setPrev( pTmpThrd->getPrev());
			}

			if( pTmpThrd == m_pThreadList)
			{
				m_pThreadList = pTmpThrd->getNext();
			}

			pTmpThrd->Release();
			break;
		}

		pTmpThrd = pTmpThrd->getNext();
	}

	f_semSignal( m_hSem);
	unlock();
	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmSharedContext::killThread(
	FLMUINT		uiThreadID,
	FLMUINT		uiMaxWait)
{
	FlmThreadContext *	pThread;
	FLMUINT					uiStartTime;
	RCODE						rc = NE_XFLM_OK;

	lock();
	pThread = m_pThreadList;
	while( pThread)
	{
		if( pThread->getID() == uiThreadID)
		{
			pThread->shutdown();
			break;
		}
		pThread = pThread->getNext();
	}
	unlock();

	// Wait for the thread to exit
	uiStartTime = FLM_GET_TIMER();
	uiMaxWait = FLM_SECS_TO_TIMER_UNITS( uiMaxWait);
	for( ;;)
	{
		(void)f_semWait( m_hSem, 200);
		lock();
		pThread = m_pThreadList;
		while( pThread)
		{
			if( pThread->getID() == uiThreadID)
			{
				break;
			}
			pThread = pThread->getNext();
		}
		unlock();

		if( !pThread)
		{
			break;
		}

		if( uiMaxWait)
		{
			if( FLM_GET_TIMER() - uiStartTime >= uiMaxWait)
			{
				rc = RC_SET( NE_XFLM_FAILURE);
				goto Exit;
			}
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmSharedContext::setFocus(
	FLMUINT		uiThreadID)
{
	FlmThreadContext *	pThread;

	lock();
	pThread = m_pThreadList;
	while( pThread)
	{
		if( pThread->getID() == uiThreadID)
		{
			if( pThread->getScreen())
			{
				FTXScreenDisplay( pThread->getScreen());
			}
			break;
		}
		pThread = pThread->getNext();
	}
	unlock();

	return( NE_XFLM_OK);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmSharedContext::isThreadTerminating(
	FLMUINT		uiThreadID)
{
	FLMBOOL					bTerminating = FALSE;
	FlmThreadContext *	pThread;

	lock();
	pThread = m_pThreadList;
	while( pThread)
	{
		if( pThread->getID() == uiThreadID)
		{
			if( pThread->getShutdownFlag())
			{
				bTerminating = TRUE;
			}
			break;
		}
		pThread = pThread->getNext();
	}
	unlock();

	return( bTerminating);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmSharedContext::getThread(
	FLMUINT					uiThreadID,
	FlmThreadContext **	ppThread)
{
	FlmThreadContext *	pThread;

	lock();
	pThread = m_pThreadList;
	while( pThread)
	{
		if( pThread->getID() == uiThreadID)
		{
			if( ppThread)
			{
				*ppThread = pThread;
			}
			break;
		}
		pThread = pThread->getNext();
	}
	unlock();

	return( ((pThread != NULL)
			? NE_XFLM_OK
			: RC_SET( NE_XFLM_NOT_FOUND)));
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FTKAPI _flmWrapperFunc(
	IF_Thread *		pFlmThread)
{
	FlmThreadContext *	pThread = (FlmThreadContext *)pFlmThread->getParm1();
	FlmSharedContext *	pSharedContext = pThread->getSharedContext();

	pThread->setFlmThread( pFlmThread);
	if( RC_BAD( pThread->execute()))
	{
		goto Exit;
	}

Exit:

	pThread->setFuncExited();
	pThread->setFlmThread( NULL);

	// Unlink the thread from the shared context
	pSharedContext->deregisterThread( pThread);
	return( NE_XFLM_OK);
}

/****************************************************************************
Desc:	callback to use to output a line
****************************************************************************/
void utilOutputLine( 
	char *				pszData, 
	void * 				pvUserData)
{
	FTX_WINDOW * 		pMainWindow = (FTX_WINDOW*)pvUserData;
	eColorType			uiBack, uiFore;
		
	FTXWinGetBackFore( pMainWindow, &uiBack, &uiFore);
	FTXWinCPrintf( pMainWindow, uiBack, uiFore, "%s\n", pszData);
}

/****************************************************************************
Name:	utilPressAnyKey
Desc:	callback to serve as a 'pager' function when the Usage: help
		is too long to fit on one screen.
****************************************************************************/ 
void utilPressAnyKey( char * pszMessage, void * pvUserData)
{
	FTX_WINDOW *		pMainWindow = (FTX_WINDOW*)pvUserData;
	FLMUINT 				uiChar;
	eColorType			uiBack, uiFore;
	
	FTXWinGetBackFore( pMainWindow, &uiBack, &uiFore);
	FTXWinCPrintf( pMainWindow, uiBack, uiFore, pszMessage);
	while( RC_BAD( FTXWinTestKB( pMainWindow)))
	{
		f_sleep( 100);
	}
	FTXWinCPrintf( pMainWindow, uiBack, uiFore,
		"\r                                                                  ");
	FTXWinCPrintf( pMainWindow, uiBack, uiFore, "\r");
	FTXWinInputChar( pMainWindow, &uiChar);
}

/****************************************************************************
Name:	utilInitWindow
Desc:	routine to startup the TUI
****************************************************************************/
RCODE utilInitWindow(
	char *			pszTitle,
	FLMUINT *		puiScreenRows,
	FTX_WINDOW **	ppMainWindow,
	FLMBOOL *		pbShutdown)
{
	FTX_SCREEN *	pScreen = NULL;
	FTX_WINDOW *	pTitleWin = NULL;
	FLMUINT			uiCols;
	int				iResCode = 0;

	if( RC_BAD( FTXInit( pszTitle, (FLMBYTE)80, (FLMBYTE)50,
		FLM_BLUE, FLM_WHITE, NULL, NULL)))
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
	
	if( RC_BAD( FTXScreenGetSize( pScreen, &uiCols, puiScreenRows)))
	{
		iResCode = 1;
		goto Exit;
	}

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
Name:	utilShutdownWindow
Desc:	routine to shutdown the TUI
****************************************************************************/
void utilShutdownWindow()
{
	FTXExit();
}
