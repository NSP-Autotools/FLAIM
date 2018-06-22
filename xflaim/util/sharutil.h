//------------------------------------------------------------------------------
// Desc:	Shared utility routines
// Tabs:	3
//
// Copyright (c) 1997, 1999-2001, 2003-2006 Novell, Inc.
// All Rights Reserved.
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

#ifndef SHARUTIL_H
#define SHARUTIL_H

/* Current utility version */

#define UTIL_VER			((FLMUINT)300)

/* Current FLAIM code version */

#define SRC_VER_STR		"Ver32 Alpha"

/* Prototypes */

void flmUtilParseParams(
	char *		pszCommandBuffer,
	FLMINT		iMaxArgs,
	FLMINT *		piArgc,
	char **		ppszArgv);

RCODE  flmUtilStatusHook(				// source: sharutl2.cpp
	FLMUINT			uiStatusType,
	void *			Parm1,
	void *			Parm2,
	void *			UserData);

#ifdef FLM_NLM
	#define flmUtilGiveUpCPU()     f_yieldCPU()
#else
	#define flmUtilGiveUpCPU()     f_sleep( 0)
#endif

//convenience macros
#define STREQ(s1,s2) ((f_strcmp( (s1), (s2))) == 0)
#define STREQI(s1,s2) ((f_stricmp( (s1), (s2))) == 0)
#define TEST_RC(rc)										\
if (RC_BAD( (rc)))										\
{																\
	goto Exit;												\
}
#define TEST_RC_LOCAL(rc)								\
if (RC_BAD( (rc)))										\
{																\
	goto Exit_local;										\
}
#define MAKE_BAD_RC_JUMP()								\
{																\
	rc = RC_SET( NE_XFLM_FAILURE);					\
	goto Exit;												\
}

#ifndef ELEMCOUNT
	#define ELEMCOUNT( elem)		sizeof(elem) / sizeof(elem[0])
#endif

/****************************************************************************
Name:	FlmVector
Desc:	treat this vector class like an array, except that you will never
		write to an item out-of-bounds.  This is because the vector
		dynamically allocates enough space to cover at least up through the
		index you are setting.  If you try to read out-of-bounds you will
		hit an assert rather than an access violation.  You will need to
		keep track of your own length, as there is no concept of "length"
		internal to this class.  You can exploit the fact that if you
		leave holes in the elements, the intermediate elements will
		be filled with 0's.
****************************************************************************/
class FlmVector : public F_Object
{
public:
	FlmVector()
	{
		m_pElementArray = NULL;
		m_uiArraySize = 0;
	}
	~FlmVector()
	{
		if ( m_pElementArray)
		{
			f_free( &m_pElementArray);
		}
	}
	RCODE setElementAt( void * pData, FLMUINT uiIndex);
	void * getElementAt( FLMUINT uiIndex);
private:
	void **	m_pElementArray;
	FLMUINT	m_uiArraySize;
};

/****************************************************************************
Name:	FlmStringAcc
Desc:	a class to safely build up a string accumulation, without worrying
		about buffer overflows.
****************************************************************************/
#define FSA_QUICKBUF_BUFFER_SIZE 128
class FlmStringAcc : public F_Object
{
public:
	FlmStringAcc()
	{
		commonInit();
	}
	FlmStringAcc( char * pszStr)
	{
		commonInit();
		this->appendTEXT( pszStr);
	}
	FlmStringAcc( FLMBYTE * pszStr)
	{
		commonInit();
		this->appendTEXT( pszStr);
	}
	~FlmStringAcc()
	{
		if ( m_pszVal)
		{
			f_free( &m_pszVal);
		}
	}
	void clear()
	{
		if ( m_pszVal)
		{
			m_pszVal[ 0] = 0;
		}
		m_szQuickBuf[ 0] = 0;
		m_uiValStrLen = 0;
	}

	FLMUINT	getLength()
	{
		return m_uiValStrLen;
	}

	RCODE printf( const char * pszFormatString, ...);		
	RCODE appendCHAR( char ucChar, FLMUINT uiHowMany = 1);
	RCODE appendTEXT( const FLMBYTE * pszVal);
	RCODE appendTEXT( const char * pszVal)
	{
		return appendTEXT( (FLMBYTE*)pszVal);
	}
	RCODE appendf( const char * pszFormatString, ...);
	
	const char * getTEXT()
	{
		//use quick buffer if applicable
		if ( m_bQuickBufActive)
		{
			return m_szQuickBuf;
		}
		else if ( m_pszVal)
		{
			return m_pszVal;
		}
		else
		{
			return( "");
		}
	}
private:
	void			commonInit() //called by all constructors
	{
		m_pszVal = NULL;
		m_uiValStrLen = 0;
		m_szQuickBuf[ 0] = 0;
		m_bQuickBufActive = FALSE;
	}
	RCODE			formatNumber( FLMUINT uiNum, FLMUINT uiBase);
	//use a small buffer for small strings to avoid heap allocations
	char			m_szQuickBuf[ FSA_QUICKBUF_BUFFER_SIZE];
	FLMBOOL		m_bQuickBufActive;
	char *		m_pszVal;
	FLMUINT		m_uiBytesAllocatedForPszVal;
	FLMUINT		m_uiValStrLen; //save the strlen stored to avoid recomputing it
};

/*===========================================================================
Class:	FlmContext
Desc:		This class manages a context or environment of variables.
===========================================================================*/
class FlmContext : public F_Object
{
public:
	FlmContext( void);

	~FlmContext( void);

	RCODE setup(
		FLMBOOL		bShared);

	RCODE setCurrDir( FLMBYTE * pszCurrDir);

	RCODE getCurrDir( FLMBYTE * pszCurrDir);

	void lock( void);
	void unlock( void);

private:

	// Data

	FLMBYTE					m_szCurrDir[ F_PATH_MAX_SIZE];
	FLMBOOL					m_bIsSetup;
	F_MUTEX					m_hMutex;		// Semaphore for controlling multi-thread
												// access.
};


class FlmSharedContext;
class FlmThreadContext;

typedef FLMUINT (* THREAD_FUNC_p)(
				FlmThreadContext *		pThread,
				void *						pvAppData);

/*===========================================================================
Class:	FlmThreadContext
Desc:		This class manages a thread.
===========================================================================*/
class FlmThreadContext : public F_Object
{
public:
	FlmThreadContext( void);
	virtual ~FlmThreadContext( void);

	RCODE setup(
		FlmSharedContext *	pSharedContext,
		const char *			pszThreadName,
		THREAD_FUNC_p			pFunc,
		void *					pvAppData);

	virtual RCODE execute( void);

	void shutdown();				// Needs to be thread-safe.

	FINLINE FlmContext * getLocalContext( void){ return m_pLocalContext;}
	FINLINE FlmSharedContext * getSharedContext( void){ return m_pSharedContext;}
	FINLINE FLMBOOL * getShutdownFlagAddr( void) { return( &m_bShutdown); }
	FINLINE void setShutdownFlag( void) { m_bShutdown = TRUE; }
	FINLINE FLMBOOL getShutdownFlag( void)
	{
		if( m_pThread && m_pThread->getShutdownFlag())
		{
			m_bShutdown = TRUE;
		}

		return( m_bShutdown);
	}

	FINLINE void setNext( FlmThreadContext * pNext) { m_pNext = pNext; }
	FINLINE void setPrev( FlmThreadContext * pPrev) { m_pPrev = pPrev; }

	FINLINE FlmThreadContext * getNext( void) { return( m_pNext); }
	FINLINE FlmThreadContext * getPrev( void) { return( m_pPrev); }

	FINLINE void setID( FLMUINT uiID) { m_uiID = uiID; }
	FINLINE FLMUINT getID( void) { return( m_uiID); }

	FINLINE void setScreen( FTX_SCREEN * pScreen) { m_pScreen = pScreen; }
	FINLINE FTX_SCREEN * getScreen( void) { return( m_pScreen); }

	FINLINE void setWindow( FTX_WINDOW * pWindow) { m_pWindow = pWindow; }
	FINLINE FTX_WINDOW * getWindow( void) { return( m_pWindow); }

	FINLINE void setFlmThread( IF_Thread * pThread)
	{
		m_pThread = pThread;
	}

	FINLINE IF_Thread * getFlmThread( void)
	{
		return( m_pThread);
	}

	void getName(
		char *	pszName,
		FLMBOOL	bLocked = FALSE);

	RCODE exec( void);
	void lock( void);
	void unlock( void);

	FLMBOOL funcExited() { return m_bFuncExited; }
	FINLINE void setFuncExited() { m_bFuncExited = TRUE; }
	RCODE getFuncErrorCode()
	{
		flmAssert( this->funcExited());
		return m_FuncRC;
	}


protected:

	FTX_SCREEN *			m_pScreen;
	FTX_WINDOW *			m_pWindow;

private:

	FLMBOOL					m_bShutdown;
	FLMUINT					m_uiID;
	FlmContext *			m_pLocalContext;
	FlmSharedContext *	m_pSharedContext;
	FlmThreadContext *	m_pNext;
	FlmThreadContext *	m_pPrev;
	F_MUTEX					m_hMutex;
	IF_Thread *				m_pThread;
	THREAD_FUNC_p			m_pThrdFunc;
	void *					m_pvAppData;
#define MAX_THREAD_NAME_LEN	64
	char						m_szName[ MAX_THREAD_NAME_LEN + 1];
	FLMBOOL					m_bFuncExited;
	RCODE						m_FuncRC;
};

/*===========================================================================
Class:	FlmSharedContext
Desc:		This class manages the shared context for a group of threads.
===========================================================================*/
class FlmSharedContext : public FlmContext
{
public:
	FlmSharedContext( void);
	~FlmSharedContext( void);

	RCODE init(								// Initialized the share object.
		FlmSharedContext *	pSharedContext);

	FINLINE void setShutdownFlag( FLMBOOL * pbShutdownFlag)
	{
		m_pbShutdownFlag = pbShutdownFlag;
	}

	// Threads

	RCODE spawn(
		FlmThreadContext *	pThread,
		FLMUINT *				puiThreadID = NULL);	// ID of spawned thread

	RCODE spawn(
		char *					pszThreadName,
		THREAD_FUNC_p			pFunc,
		void *					pvUserData,
		FLMUINT *				puiThreadID = NULL);	// ID of spawned thread

	void wait( void);

	void shutdown();		// Shutdown all threads in this shared context.

	RCODE killThread(
		FLMUINT					uiThreadID,
		FLMUINT					uiMaxWait = 0);

	RCODE setFocus( FLMUINT uiThreadID);

	FLMBOOL isThreadTerminating(
		FLMUINT					uiThreadID);

	RCODE getThread(
		FLMUINT					uiThreadID,
		FlmThreadContext **	ppThread);

	RCODE registerThread(
		FlmThreadContext *	pThread);

	RCODE deregisterThread(
		FlmThreadContext *	pThread);

private:

	FlmSharedContext *	m_pParentContext;
	FLMBOOL					m_bPrivateShare;
	F_MUTEX					m_hMutex;
	F_SEM						m_hSem;
	FlmThreadContext *	m_pThreadList;
	FLMBOOL					m_bLocalShutdownFlag;
	FLMBOOL *				m_pbShutdownFlag;
	FLMUINT					m_uiNextProcID;
};

void utilOutputLine( char * pszData, void * pvUserData);

void utilPressAnyKey( char * pszPressAnyKeyMessage, void * pvUserData);

RCODE utilInitWindow(
	char *			pszTitle,
	FLMUINT *		puiScreenRows,
	FTX_WINDOW **	ppMainWindow,
	FLMBOOL *		pbShutdown);
	
void utilShutdownWindow();

#endif // SHARUTIL_H

