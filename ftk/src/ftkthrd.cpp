//------------------------------------------------------------------------------
// Desc:	Functions for creating, starting, stopping, controlling threads.
// Tabs:	3
//
// Copyright (c) 2000-2007 Novell, Inc. All Rights Reserved.
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

#ifdef FLM_UNIX
	pid_t getpid( void);
#endif

#ifdef FLM_LIBC_NLM
	void * threadStub(
		void *	pvThread);
#elif defined( FLM_RING_ZERO_NLM)
	void * threadStub(
		void *	pvUnused,
		void *	pvThread);
#elif defined( FLM_WIN)
	unsigned __stdcall threadStub(
		void *	pvThread);
#elif defined( FLM_UNIX)
	extern "C" void * threadStub(
		void *	pvThread);
#endif

/****************************************************************************
Desc:
****************************************************************************/
class F_ThreadMgr : public IF_ThreadMgr
{
public:

	F_ThreadMgr()
	{
		m_hMutex = F_MUTEX_NULL;
		m_pThreadList = NULL;
		m_uiNumThreads = 0;
		m_groupCounter = 0;
	}

	virtual ~F_ThreadMgr();

	RCODE FTKAPI setupThreadMgr( void);

	FLMINT FTKAPI AddRef( void)
	{
		return( f_atomicInc( &m_refCnt));
	}

	FLMINT FTKAPI Release( void)
	{
		FLMINT		iRefCnt = f_atomicDec( &m_refCnt);
		
		if( !iRefCnt)
		{
			delete this;
		}
		
		return( iRefCnt);
	}
	
	RCODE FTKAPI createThread(
		IF_Thread **		ppThread,
		F_THREAD_FUNC		fnThread,
		const char *		pszThreadName,
		FLMUINT				uiThreadGroup,
		FLMUINT				uiAppId,
		void *				pvParm1,
		void *				pvParm2,
		FLMUINT				uiStackSize);
		
	void FTKAPI shutdownThreadGroup(
		FLMUINT				uiThreadGroup);

	void FTKAPI setThreadShutdownFlag(
		FLMUINT				uiThreadId);

	RCODE FTKAPI findThread(
		IF_Thread **		ppThread,
		FLMUINT				uiThreadGroup,
		FLMUINT				uiAppId = 0,
		FLMBOOL				bOkToFindMe = TRUE);

	RCODE FTKAPI getNextGroupThread(
		IF_Thread **		ppThread,
		FLMUINT				uiThreadGroup,
		FLMUINT *			puiThreadId);

	RCODE FTKAPI getThreadInfo(
		F_Pool *				pPool,
		F_THREAD_INFO **	ppThreadInfo,
		FLMUINT *			puiNumThreads);

	RCODE FTKAPI getThreadName(
		FLMUINT				uiThreadId,
		char *				pszThreadName,
		FLMUINT *			puiLength);
	
	FLMUINT FTKAPI getThreadGroupCount(
		FLMUINT				uiThreadGroup);
		
	FLMUINT FTKAPI allocGroupId( void);
		
	inline void lockMutex( void)
	{
		f_mutexLock( m_hMutex);
	}
	
	inline void unlockMutex( void)
	{
		f_mutexUnlock( m_hMutex);
	}

	void unlinkThread(
		IF_Thread *		pThread,
		FLMBOOL			bMutexLocked);

	RCODE getThread(
		FLMUINT				uiThreadId,
		F_Thread **			ppThread);
	
private:

	F_MUTEX			m_hMutex;
	F_Thread *		m_pThreadList;
	FLMUINT			m_uiNumThreads;
	FLMATOMIC		m_groupCounter;

friend class F_Thread;
};

/****************************************************************************
Desc:
****************************************************************************/
class F_Thread : public IF_Thread
{
public:

	F_Thread()
	{
		m_hMutex = F_MUTEX_NULL;
		m_pszThreadName = NULL;
		m_pszThreadStatus = NULL;
		m_uiStatusBufLen = 0;
		m_uiThreadGroup = F_INVALID_THREAD_GROUP;
		m_pPrev = NULL;
		m_pNext = NULL;
		cleanupThread();
	}

	virtual ~F_Thread()
	{
		stopThread();
		cleanupThread();
	}

	FLMINT FTKAPI AddRef( void);

	FLMINT FTKAPI Release( void);

	RCODE FTKAPI startThread(
		F_THREAD_FUNC	fnThread,
		const char *	pszThreadName,
		FLMUINT			uiThreadGroup,
		FLMUINT			uiAppId,
		void *			pvParm1,
		void *			pvParm2,
		FLMUINT        uiStackSize);

	void FTKAPI stopThread( void);

	FINLINE FLMUINT FTKAPI getThreadId( void)
	{
		return( m_uiThreadId);
	}

	FINLINE FLMBOOL FTKAPI getShutdownFlag( void)
	{
		return( m_bShutdown);
	}

	FINLINE RCODE FTKAPI getExitCode( void)
	{
		return( m_exitRc);
	}

	FINLINE void * FTKAPI getParm1( void)
	{
		return( m_pvParm1);
	}

	FINLINE void FTKAPI setParm1(
		void *		pvParm)
	{
		m_pvParm1 = pvParm;
	}

	FINLINE void * FTKAPI getParm2( void)
	{
		return( m_pvParm2);
	}

	FINLINE void FTKAPI setParm2(
		void *		pvParm)
	{
		m_pvParm2 = pvParm;
	}

	FINLINE void FTKAPI setShutdownFlag( void)
	{
		m_bShutdown = TRUE;
	}

	FINLINE FLMBOOL FTKAPI isThreadRunning( void)
	{
		return( m_bRunning);
	}

	void FTKAPI setThreadStatusStr(
		const char *	pszStatus);

	void FTKAPI setThreadStatus(
		const char *	pszBuffer, ...);

	void FTKAPI setThreadStatus(
		eThreadStatus	genericStatus);

	FINLINE void FTKAPI setThreadAppId(
		FLMUINT		uiAppId)
	{
		f_mutexLock( m_hMutex);
		m_uiAppId = uiAppId;
		f_mutexUnlock( m_hMutex);
	}

	FINLINE FLMUINT FTKAPI getThreadAppId( void)
	{
		return( m_uiAppId);
	}

	FINLINE FLMUINT FTKAPI getThreadGroup( void)
	{
		return( m_uiThreadGroup);
	}

	void FTKAPI cleanupThread( void);

	void FTKAPI sleep(
			FLMUINT		uiMilliseconds);
			
	void FTKAPI waitToComplete( void);

	F_MUTEX				m_hMutex;
	F_Thread *			m_pPrev;
	F_Thread *			m_pNext;
	char *				m_pszThreadName;
	char *				m_pszThreadStatus;
	FLMUINT				m_uiStatusBufLen;
	FLMBOOL				m_bShutdown;
	F_THREAD_FUNC		m_fnThread;
	FLMBOOL				m_bRunning;
	FLMUINT				m_uiStackSize;
	void *				m_pvParm1;
	void *				m_pvParm2;
	FLMUINT				m_uiThreadId;
	FLMUINT				m_uiThreadGroup;
	FLMUINT				m_uiAppId;
	FLMUINT				m_uiStartTime;
	RCODE					m_exitRc;

friend class F_ThreadMgr;
};

/****************************************************************************
Desc:
****************************************************************************/
class F_ThreadInfo : public IF_ThreadInfo
{
public:

	F_ThreadInfo()
	{
		m_pool.poolInit( 512);
		m_uiNumThreads = 0;
		m_pThreadInfoArray = NULL;
	}

	virtual ~F_ThreadInfo()
	{
		m_pool.poolFree();
	}

	FLMUINT FTKAPI getNumThreads( void)
	{
		return( m_uiNumThreads);
	}

	FINLINE void FTKAPI getThreadInfo(
		FLMUINT				uiThreadNum,
		FLMUINT *			puiThreadId,
		FLMUINT *			puiThreadGroup,
		FLMUINT *			puiAppId,
		FLMUINT *			puiStartTime,
		const char **		ppszThreadName,
		const char **		ppszThreadStatus)
	{
		if (uiThreadNum < m_uiNumThreads)
		{
			F_THREAD_INFO *	pThrdInfo = &m_pThreadInfoArray [uiThreadNum];

			*puiThreadId = pThrdInfo->uiThreadId;
			*puiThreadGroup = pThrdInfo->uiThreadGroup;
			*puiAppId = pThrdInfo->uiAppId;
			*puiStartTime = pThrdInfo->uiStartTime;
			*ppszThreadName = pThrdInfo->pszThreadName;
			*ppszThreadStatus = pThrdInfo->pszThreadStatus;
		}
		else
		{
			*puiThreadId = 0;
			*puiThreadGroup = 0;
			*puiAppId = 0;
			*puiStartTime = 0;
			*ppszThreadName = NULL;
			*ppszThreadStatus = NULL;
		}
	}

	F_Pool				m_pool;
	F_THREAD_INFO *	m_pThreadInfoArray;
	FLMUINT				m_uiNumThreads;
};

/****************************************************************************
Desc:
****************************************************************************/
RCODE f_allocThreadMgr(
	IF_ThreadMgr **		ppThreadMgr)
{
	RCODE				rc = NE_FLM_OK;
	F_ThreadMgr *	pThreadMgr = NULL;

	if( (pThreadMgr = f_new F_ThreadMgr) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
	}

	if( RC_BAD( rc = pThreadMgr->setupThreadMgr()))
	{
		goto Exit;
	}

	*ppThreadMgr = pThreadMgr;
	pThreadMgr = NULL;

Exit:

	if( pThreadMgr)
	{
		pThreadMgr->Release();
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI FlmGetThreadMgr(
	IF_ThreadMgr **		ppThreadMgr)
{
	*ppThreadMgr = f_getThreadMgrPtr();
	(*ppThreadMgr)->AddRef();
	return( NE_FLM_OK);
}
		
/****************************************************************************
Desc:	Add a Reference to this object.
****************************************************************************/
FLMINT FTKAPI F_Thread::AddRef( void)
{
	return( f_atomicInc( &m_refCnt));
}

/****************************************************************************
Desc: Removes a reference to this object.
****************************************************************************/
FLMINT FTKAPI F_Thread::Release( void)
{
	FLMINT		iRefCnt = f_atomicDec( &m_refCnt);
	
	if( !iRefCnt)
	{
		delete this;
	}

	return( iRefCnt);
}

/****************************************************************************
Desc:    Performs various setup work and starts a new thread
****************************************************************************/
RCODE FTKAPI F_Thread::startThread(
	F_THREAD_FUNC	fnThread,
	const char *	pszThreadName,
	FLMUINT			uiThreadGroup,
	FLMUINT			uiAppId,
	void *			pvParm1,
	void *			pvParm2,
	FLMUINT        uiStackSize)
{
	RCODE						rc = NE_FLM_OK;
	F_ThreadMgr *			pThreadMgr = (F_ThreadMgr *)f_getThreadMgrPtr();
	FLMBOOL					bManagerMutexLocked = FALSE;
#ifdef FLM_LIBC_NLM
	pthread_attr_t			thread_attr;
	pthread_t				uiThreadId;
#endif
#ifdef FLM_RING_ZERO_NLM
	void *					hThread = NULL;
#endif
#ifdef FLM_WIN
	unsigned					uiThreadId;
#endif
#if defined( FLM_UNIX)
	pthread_attr_t			thread_attr;
	pthread_t				uiThreadId;
#endif

	f_assert( fnThread != NULL && m_fnThread == NULL);
	f_assert( uiThreadGroup != F_INVALID_THREAD_GROUP);

	m_fnThread = fnThread;
	m_pvParm1 = pvParm1;
	m_pvParm2 = pvParm2;

	// Initialize the thread's mutex

	if( RC_BAD( rc = f_mutexCreate( &m_hMutex)))
	{
		goto Exit;
	}

	// Set the stack size

	m_uiStackSize = (uiStackSize < F_THREAD_MIN_STACK_SIZE)
							? F_THREAD_MIN_STACK_SIZE
							: uiStackSize;

	// Set the thread name

	if( pszThreadName && *pszThreadName)
	{
		FLMUINT		uiNameLen = f_strlen( pszThreadName) + 1;

		if( RC_BAD( rc = f_alloc( uiNameLen, &m_pszThreadName)))
		{
			goto Exit;
		}

		f_memcpy( (void *)m_pszThreadName, pszThreadName, uiNameLen);
	}

	// Set the thread group ID and the application-specified thread ID

	m_uiThreadGroup = uiThreadGroup;
	m_uiAppId = uiAppId;

	// Set the thread's state to "running" -- if we fail to
	// start the thread, this will be set back to false when
	// the cleanupThread() method is called below.  We set this
	// to TRUE here so that the stopThread() method won't get
	// stuck in an infinite loop if the thread was never started.

	m_bRunning = TRUE;

	// Lock the thread manager's mutex.

	f_mutexLock( pThreadMgr->m_hMutex);
	bManagerMutexLocked = TRUE;

	// Increment the active thread count

	pThreadMgr->m_uiNumThreads++;

	// Link the thread into the manager's list.  We can't link threads in order
	// by thread ID at this point, because we don't know what the new thread's
	// ID will be.

	if( pThreadMgr->m_pThreadList)
	{
		pThreadMgr->m_pThreadList->m_pPrev = this;
	}

	m_pNext = pThreadMgr->m_pThreadList;
	pThreadMgr->m_pThreadList = this;

	// Increment the reference count of the thread object now
	// that it is linked into the thread manager's list.

	m_refCnt++;

	// Start the thread

#ifdef FLM_WIN
	if( _beginthreadex(
		NULL, (unsigned int)m_uiStackSize, threadStub,
		(void *)this, 0, &uiThreadId) == 0)
	{
		rc = RC_SET( NE_FLM_COULD_NOT_START_THREAD);
		goto Exit;
	}
	m_uiThreadId = (FLMUINT)uiThreadId;
#elif defined( FLM_LIBC_NLM)
		pthread_attr_init( &thread_attr);
		pthread_attr_setdetachstate( &thread_attr, PTHREAD_CREATE_DETACHED);

		if (pthread_create( &uiThreadId, &thread_attr,
				threadStub, this) != 0)
		{
			rc = RC_SET( NE_FLM_COULD_NOT_START_THREAD);
			goto Exit;
		}

		m_uiThreadId = (FLMUINT)uiThreadId;
		pthread_attr_destroy( &thread_attr);
#elif defined( FLM_RING_ZERO_NLM)
	if( (hThread = kCreateThread( 
		(BYTE *)((m_pszThreadName)
			? (BYTE *)m_pszThreadName 
			: (BYTE *)"FTK"),
		threadStub, NULL, (LONG)m_uiStackSize,
		(void *)this)) == NULL)
	{
		rc = RC_SET( NE_FLM_COULD_NOT_START_THREAD);
		goto Exit;
	}
	m_uiThreadId = (FLMUINT)hThread;

	if( kSetThreadLoadHandle( hThread, (LONG)f_getNLMHandle()) != 0)
	{
		(void)kDestroyThread( hThread);
		rc = RC_SET( NE_FLM_COULD_NOT_START_THREAD);
		goto Exit;
	}
			
   if( kScheduleThread( hThread) != 0)
	{
		(void)kDestroyThread( hThread);
		rc = RC_SET( NE_FLM_COULD_NOT_START_THREAD);
		goto Exit;
	}
#elif defined( FLM_UNIX)
	pthread_attr_init( &thread_attr);
	pthread_attr_setdetachstate( &thread_attr, PTHREAD_CREATE_DETACHED);

	if (pthread_create( &uiThreadId, &thread_attr,
			threadStub, this) != 0)
	{
		rc = RC_SET( NE_FLM_COULD_NOT_START_THREAD);
		goto Exit;
	}

	m_uiThreadId = (FLMUINT)uiThreadId;
	pthread_attr_destroy( &thread_attr);
#endif

	// Code is not designed to handle a thread ID of 0

	f_assert( m_uiThreadId != 0);

	// Unlock the thread manager's mutex.

	f_mutexUnlock( pThreadMgr->m_hMutex);
	bManagerMutexLocked = FALSE;

Exit:

	if( RC_BAD( rc))
	{
		// Unlink the thread from the manager's list.  This call
		// won't do anything if the thread was not linked above.

		pThreadMgr->unlinkThread( this, bManagerMutexLocked);

		// Reset the thread object back to its initial state

		cleanupThread();
	}

	if( bManagerMutexLocked)
	{
		f_mutexUnlock( pThreadMgr->m_hMutex);
	}

	return( rc);
}

/****************************************************************************
Desc: Stop a running thread
****************************************************************************/
void FTKAPI F_Thread::stopThread( void)
{
	// Set the shutdown flag and wait for the thread's
	// status to be something other than "running"

	m_bShutdown = TRUE;
	while( m_bRunning)
	{
		f_sleep( 10);
	}

	// Reset the shutdown flag in case this object is re-used.

	m_bShutdown = FALSE;
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI F_Thread::waitToComplete( void)
{
	while( m_bRunning)
	{
		f_sleep( 10);
	}
}

/****************************************************************************
Desc:    Begins a new thread of execution and calls the passed function.
			Performs generic thread init and cleanup functions.
****************************************************************************/
#ifdef FLM_LIBC_NLM
void * threadStub(
	void *	pvThread)
#elif defined( FLM_RING_ZERO_NLM)
void * threadStub(
	void *	pvUnused,
	void *	pvThread)
#elif defined( FLM_WIN)
unsigned __stdcall threadStub(
	void *	pvThread)
#elif defined( FLM_UNIX)
void * threadStub(
	void *	pvThread)
#endif
{
	F_Thread *			pThread = (F_Thread *)pvThread;
	F_ThreadMgr *		pThreadMgr = (F_ThreadMgr *)f_getThreadMgrPtr();

#ifdef FLM_RING_ZERO_NLM
	F_UNREFERENCED_PARM( pvUnused);
#endif

#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
	// Block all signals (main thread will handle all signals)

	sigset_t mask;
	sigfillset(&mask);
	pthread_sigmask(SIG_SETMASK, &mask, 0);
#endif

	// Lock the manager's mutex

	pThreadMgr->lockMutex();

	// At this point, the thread ID must match.

	f_assert( pThread->m_uiThreadId == f_threadId());

	// Set the start time

	f_timeGetSeconds( &pThread->m_uiStartTime);

	// Unlock the manager's mutex

	pThreadMgr->unlockMutex();

	// Call the thread's function

	pThread->m_exitRc = pThread->m_fnThread( pThread);

	// Add a temporary reference to the thread object so
	// it doesn't go away when we unlink it from the
	// manager

	pThread->AddRef();

	// Unlink the thread from the thread manager.

	pThreadMgr->unlinkThread( pThread, FALSE);

	// Set the running flag to FALSE

	pThread->m_bRunning = FALSE;

	// Release the temporary reference to the thread.  Once the
	// reference is release, pThread must not be accessed because
	// the object may have gone away.

	pThread->Release();
	pThread = NULL;

	// Terminate the thread

#if defined( FLM_WIN)
	_endthreadex( 0);
	return( 0);
#elif defined( FLM_RING_ZERO_NLM)
	kExitThread( NULL);
#endif

#if defined( FLM_NLM) || defined( FLM_UNIX)
	return( NULL);
#endif
}

/****************************************************************************
Desc:    Frees any resources allocated to the thread and resets member
			variables to their initial state
****************************************************************************/
void FTKAPI F_Thread::cleanupThread( void)
{
	f_assert( !m_pPrev && !m_pNext);
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}

	if( m_pszThreadName)
	{
		f_free( &m_pszThreadName);
	}

	if( m_pszThreadStatus)
	{
		f_free( &m_pszThreadStatus);
	}

	m_uiStatusBufLen = 0;
	m_bShutdown = FALSE;
	m_fnThread = NULL;
	m_bRunning = FALSE;
	m_uiStackSize = 0;
	m_pvParm1 = NULL;
	m_pvParm2 = NULL;
	m_uiThreadId = 0;
	m_uiThreadGroup = F_INVALID_THREAD_GROUP;
	m_uiAppId = 0;
	m_uiStartTime = 0;
	m_exitRc = NE_FLM_OK;
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI F_Thread::sleep(
	FLMUINT		uiMilliseconds)
{
	FLMUINT		uiTimeToSleep;
	
	if( !uiMilliseconds)
	{
		f_yieldCPU();
		return;
	}
	
	while( uiMilliseconds && !m_bShutdown)
	{
		uiTimeToSleep = f_min( uiMilliseconds, 50);
		f_sleep( uiTimeToSleep);
		uiMilliseconds -= uiTimeToSleep;
	}
}
			
/****************************************************************************
Desc:    Set the thread's status
****************************************************************************/
void FTKAPI F_Thread::setThreadStatusStr(
	const char *		pszStatus)
{
	FLMUINT		uiStatusLen = f_strlen( pszStatus) + 1;

	if( m_uiStatusBufLen < uiStatusLen)
	{
		FLMUINT		uiAllocSize = uiStatusLen < 128 ? 128 : uiStatusLen;

		if( m_pszThreadStatus != NULL)
		{
			f_free( &m_pszThreadStatus);
		}
		m_uiStatusBufLen = 0;

		if( RC_BAD( f_alloc( uiAllocSize, &m_pszThreadStatus)))
		{
			goto Exit;
		}
		m_uiStatusBufLen = uiAllocSize;
	}

	f_mutexLock( m_hMutex);
	f_memcpy( m_pszThreadStatus, pszStatus, uiStatusLen);
	f_mutexUnlock( m_hMutex);

Exit:

	return;
}

/****************************************************************************
Desc:    Set the thread's status
****************************************************************************/
void FTKAPI F_Thread::setThreadStatus(
	const char *	pszFormat, ...)
{
	char				pucBuffer[ 128];
	f_va_list		args;

	f_va_start( args, pszFormat);
	f_vsprintf( pucBuffer, pszFormat, &args);
	f_va_end( args);

	setThreadStatusStr( pucBuffer);
}

/****************************************************************************
Desc:    Set the thread's status to a generic string
****************************************************************************/
void FTKAPI F_Thread::setThreadStatus(
	eThreadStatus	genericStatus)
{
	const char *	pszStatus = NULL;

	switch( genericStatus)
	{
		case FLM_THREAD_STATUS_INITIALIZING:
			pszStatus = "Initializing";
			break;

		case FLM_THREAD_STATUS_RUNNING:
			pszStatus = "Running";
			break;

		case FLM_THREAD_STATUS_SLEEPING:
			pszStatus = "Sleeping";
			break;

		case FLM_THREAD_STATUS_TERMINATING:
			pszStatus = "Terminating";
			break;

		case FLM_THREAD_STATUS_UNKNOWN:
		default:
			pszStatus = "Unknown";
			break;
	}

	if( pszStatus)
	{
		setThreadStatusStr( pszStatus);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
F_ThreadMgr::~F_ThreadMgr()
{
	F_Thread *		pTmpThread;

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexLock( m_hMutex);
		pTmpThread = m_pThreadList;
		while( pTmpThread)
		{
			pTmpThread->setShutdownFlag();
			pTmpThread = pTmpThread->m_pNext;
		}

		while( m_pThreadList)
		{
			f_mutexUnlock( m_hMutex);
			f_sleep( 50);
			f_mutexLock( m_hMutex);
		}

		f_mutexUnlock( m_hMutex);
		f_mutexDestroy( &m_hMutex);
	}
}
/****************************************************************************
Desc:		Allocates resources needed by the thread manager
****************************************************************************/
RCODE FTKAPI F_ThreadMgr::setupThreadMgr( void)
{
	RCODE		rc = NE_FLM_OK;

	f_assert( m_hMutex == F_MUTEX_NULL);

	if( RC_BAD( rc = f_mutexCreate( &m_hMutex)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Removes a thread from the thread manager's list
Notes:	This routine assumes that the manager's mutex is already locked.
****************************************************************************/
void F_ThreadMgr::unlinkThread(
	IF_Thread *		ifpThread,
	FLMBOOL			bMutexIsLocked)
{
	F_Thread *		pThread = (F_Thread *)ifpThread;
	
	// Lock the thread manager's mutex

	if( !bMutexIsLocked)
	{
		f_mutexLock( m_hMutex);
	}

	// If the thread isn't linked into the list,
	// don't do anything

	if( !pThread->m_pPrev && !pThread->m_pNext &&
		m_pThreadList != pThread)
	{
		goto Exit;
	}

	// Decrement the active thread count

	f_assert( m_uiNumThreads);
	m_uiNumThreads--;

	if( pThread->m_pPrev)
	{
		pThread->m_pPrev->m_pNext = pThread->m_pNext;
	}
	else
	{
		m_pThreadList = pThread->m_pNext;
	}

	if( pThread->m_pNext)
	{
		pThread->m_pNext->m_pPrev = pThread->m_pPrev;
	}

	pThread->m_pNext = NULL;
	pThread->m_pPrev = NULL;

	// Release the thread object

	pThread->Release();

Exit:

	if( !bMutexIsLocked)
	{
		f_mutexUnlock( m_hMutex);
	}
}

/****************************************************************************
Desc:		Signals all threads in a thread group to shut down and waits
			for them to terminate.
****************************************************************************/
void FTKAPI F_ThreadMgr::shutdownThreadGroup(
	FLMUINT		uiThreadGroup)
{
	F_Thread *		pThread;
	FLMUINT			uiCount;
	
	f_assert( uiThreadGroup != F_INVALID_THREAD_GROUP);

	for( ;;)
	{
		f_mutexLock( m_hMutex);

		uiCount = 0;
		pThread = m_pThreadList;
		while( pThread)
		{
			if( pThread->m_uiThreadGroup == uiThreadGroup)
			{
				pThread->setShutdownFlag();
				uiCount++;
			}
			pThread = pThread->m_pNext;
		}

		f_mutexUnlock( m_hMutex);

		if( !uiCount)
		{
			break;
		}

		// The threads will automatically unlink themselves from
		// the manager before they terminate.  Just sleep for
		// a few milliseconds and look through the list again to
		// verify that there are no more threads in the group.

		f_sleep( 200);
	}
}

/****************************************************************************
Desc:		Signals a thread to shut down.
****************************************************************************/
void FTKAPI F_ThreadMgr::setThreadShutdownFlag(
	FLMUINT			uiThreadId)
{
	F_Thread *		pThread;

	f_assert( uiThreadId != 0);

	f_mutexLock( m_hMutex);
	pThread = m_pThreadList;
	while( pThread)
	{
		if( pThread->m_uiThreadId == uiThreadId)
		{
			pThread->setShutdownFlag();
			break;
		}
		pThread = pThread->m_pNext;
	}

	f_mutexUnlock( m_hMutex);
}

/****************************************************************************
Desc:		Allocates an array of F_THREAD_INFO structures and populates them
			with information about the threads being managed by this object.
****************************************************************************/
RCODE FTKAPI F_ThreadMgr::getThreadInfo(
	F_Pool *				pPool,
	F_THREAD_INFO **	ppThreadInfo,
	FLMUINT *			puiNumThreads)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT				uiOffset;
	FLMUINT				uiLoop;
	FLMUINT				uiSubLoop;
	FLMUINT				uiLen;
	FLMBOOL				bMutexLocked = FALSE;
	F_THREAD_INFO *	pThreadInfo = NULL;
	F_THREAD_INFO		tmpThreadInfo;
	F_Thread *			pCurThread;
	void *				pvMark = pPool->poolMark();

	*ppThreadInfo = NULL;
	*puiNumThreads = 0;

	f_mutexLock( m_hMutex);
	bMutexLocked = TRUE;

	if( m_uiNumThreads == 0)
	{
		goto Exit;
	}

	if (RC_BAD( rc = pPool->poolCalloc( sizeof( F_THREAD_INFO) * m_uiNumThreads,
		(void **)&pThreadInfo)))
	{
		goto Exit;
	}

	uiOffset = 0;
	pCurThread = m_pThreadList;
	while( pCurThread)
	{
		f_assert( uiOffset < m_uiNumThreads);
		f_mutexLock( pCurThread->m_hMutex);

		pThreadInfo[ uiOffset].uiThreadId = pCurThread->m_uiThreadId;
		pThreadInfo[ uiOffset].uiThreadGroup = pCurThread->m_uiThreadGroup;
		pThreadInfo[ uiOffset].uiAppId = pCurThread->m_uiAppId;
		pThreadInfo[ uiOffset].uiStartTime = pCurThread->m_uiStartTime;

		if( pCurThread->m_pszThreadName)
		{
			uiLen = f_strlen( pCurThread->m_pszThreadName) + 1;

			if (RC_OK( pPool->poolCalloc( uiLen,
										(void **)&pThreadInfo[ uiOffset].pszThreadName)))
			{
				f_memcpy( (void *)pThreadInfo[ uiOffset].pszThreadName,
					pCurThread->m_pszThreadName, uiLen);
			}
		}

		if( pCurThread->m_pszThreadStatus)
		{
			uiLen = f_strlen( pCurThread->m_pszThreadStatus) + 1;

			if (RC_OK( pPool->poolCalloc( uiLen,
								(void **)&pThreadInfo[ uiOffset].pszThreadStatus)))
			{
				f_memcpy( (void *)(pThreadInfo[ uiOffset].pszThreadStatus),
					pCurThread->m_pszThreadStatus, uiLen);
			}
		}

		f_mutexUnlock( pCurThread->m_hMutex);
		uiOffset++;
		pCurThread = pCurThread->m_pNext;
	}

	f_assert( uiOffset == m_uiNumThreads);
	*puiNumThreads = m_uiNumThreads;

	f_mutexUnlock( m_hMutex);
	bMutexLocked = FALSE;

	// Sort the list by thread ID

	for( uiLoop = 0; uiLoop < *puiNumThreads; uiLoop++)
	{
		for( uiSubLoop = uiLoop + 1; uiSubLoop < *puiNumThreads; uiSubLoop++)
		{
			if( pThreadInfo[ uiLoop].uiThreadId >
				pThreadInfo[ uiSubLoop].uiThreadId)
			{
				f_memcpy( &tmpThreadInfo,
					&pThreadInfo[ uiLoop], sizeof( F_THREAD_INFO));
				f_memcpy( &pThreadInfo[ uiLoop],
					&pThreadInfo[ uiSubLoop], sizeof( F_THREAD_INFO));
				f_memcpy( &pThreadInfo[ uiSubLoop],
					&tmpThreadInfo, sizeof( F_THREAD_INFO));
			}
		}
	}

	*ppThreadInfo = pThreadInfo;

Exit:

	if( RC_BAD( rc))
	{
		pPool->poolReset( pvMark);
	}

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_ThreadMgr::getThreadName(
	FLMUINT			uiThreadId,
	char *			pszThreadName,
	FLMUINT *		puiLength)
{
	RCODE				rc = NE_FLM_OK;
	F_Thread *		pThread = NULL;
	FLMUINT			uiCopyLen;
	
	if( RC_BAD( rc = getThread( uiThreadId, &pThread)))
	{
		goto Exit;
	}
	
	f_mutexLock( pThread->m_hMutex);

	if( pThread->m_pszThreadName)
	{
		uiCopyLen = f_min( *puiLength - 1, 
							f_strlen( pThread->m_pszThreadName));
		f_strncpy( pszThreadName, pThread->m_pszThreadName, uiCopyLen);
		*puiLength = uiCopyLen;
	}
	else
	{
		*pszThreadName = 0;
	}
			
	f_mutexUnlock( pThread->m_hMutex);
	
	
Exit:

	if( pThread)
	{
		pThread->Release();
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_ThreadMgr::getThread(
	FLMUINT			uiThreadId,
	F_Thread **		ppThread)
{
	RCODE				rc = NE_FLM_OK;
	FLMBOOL			bMutexLocked = FALSE;
	F_Thread *		pCurThread;

	f_assert( *ppThread == NULL);
	
	f_mutexLock( m_hMutex);
	bMutexLocked = TRUE;

	pCurThread = m_pThreadList;
	while( pCurThread)
	{
		if( pCurThread->m_uiThreadId == uiThreadId)
		{
			*ppThread = pCurThread;
			pCurThread->AddRef();
			break;
		}

		pCurThread = pCurThread->m_pNext;
	}

	f_mutexUnlock( m_hMutex);
	bMutexLocked = FALSE;
	
	if( !pCurThread)
	{
		rc = RC_SET( NE_FLM_NOT_FOUND);
		goto Exit;
	}
	
Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}
	
	return( rc);
}
	
/****************************************************************************
Desc:		Finds a thread based on user-specified identifiers
****************************************************************************/
RCODE FTKAPI F_ThreadMgr::findThread(
	IF_Thread **	ppThread,
	FLMUINT			uiThreadGroup,
	FLMUINT			uiAppId,
	FLMBOOL			bOkToFindMe)
{
	RCODE					rc = NE_FLM_OK;
	FLMBOOL				bMutexLocked = FALSE;
	F_Thread *			pCurThread;

	*ppThread = NULL;

	f_mutexLock( m_hMutex);
	bMutexLocked = TRUE;

	if( m_uiNumThreads == 0)
	{
		rc = RC_SET( NE_FLM_NOT_FOUND);
		goto Exit;
	}

	pCurThread = m_pThreadList;
	while( pCurThread)
	{
		f_mutexLock( pCurThread->m_hMutex);

		if( pCurThread->m_uiThreadGroup == uiThreadGroup &&
			pCurThread->m_uiAppId == uiAppId)
		{
			if( bOkToFindMe ||
				(!bOkToFindMe && pCurThread->m_uiThreadId != f_threadId()))
			{
				// Found a match.

				pCurThread->AddRef();
				*ppThread = pCurThread;
				f_mutexUnlock( pCurThread->m_hMutex);
				goto Exit;
			}
		}

		f_mutexUnlock( pCurThread->m_hMutex);
		pCurThread = pCurThread->m_pNext;
	}

	rc = RC_SET( NE_FLM_NOT_FOUND);

Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:		Finds a thread based on user-specified identifiers
****************************************************************************/
RCODE FTKAPI F_ThreadMgr::getNextGroupThread(
	IF_Thread **		ppThread,
	FLMUINT			uiThreadGroup,
	FLMUINT *		puiThreadId)
{
	RCODE					rc = NE_FLM_OK;
	FLMBOOL				bMutexLocked = FALSE;
	F_Thread *			pCurThread;
	F_Thread *			pFoundThread = NULL;

	f_mutexLock( m_hMutex);
	bMutexLocked = TRUE;

	if( m_uiNumThreads == 0)
	{
		rc = RC_SET( NE_FLM_NOT_FOUND);
		goto Exit;
	}

	pCurThread = m_pThreadList;
	while( pCurThread)
	{
		if( pCurThread->m_uiThreadGroup == uiThreadGroup &&
			pCurThread->m_uiThreadId > *puiThreadId)
		{
			// The threads are not kept in order by thread ID in the
			// manager's list.  So, we need to make sure we get the
			// thread with the next ID beyond the ID passed into the
			// routine.

			if( !pFoundThread ||
				pCurThread->m_uiThreadId < pFoundThread->m_uiThreadId)
			{
				pFoundThread = pCurThread;
			}
		}

		pCurThread = pCurThread->m_pNext;
	}

	if( !pFoundThread)
	{
		rc = RC_SET( NE_FLM_NOT_FOUND);
		goto Exit;
	}

	pFoundThread->AddRef();
	*ppThread = pFoundThread;
	*puiThreadId = pFoundThread->m_uiThreadId;

Exit:

	if( RC_BAD( rc))
	{
		*ppThread = NULL;
		*puiThreadId = 0xFFFFFFFF;
	}

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:		Returns a count of the number of threads in a specified group
****************************************************************************/
FLMUINT FTKAPI F_ThreadMgr::getThreadGroupCount(
	FLMUINT			uiThreadGroup)
{
	F_Thread *		pThread;
	FLMUINT			uiCount;

	f_mutexLock( m_hMutex);

	uiCount = 0;
	pThread = m_pThreadList;
	while( pThread)
	{
		if( pThread->m_uiThreadGroup == uiThreadGroup)
		{
			uiCount++;
		}
		pThread = pThread->m_pNext;
	}

	f_mutexUnlock( m_hMutex);
	return( uiCount);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT FTKAPI F_ThreadMgr::allocGroupId( void)
{
	return( f_atomicInc( &m_groupCounter));
}

/****************************************************************************
Desc:    Allocate a thread object and start the thread
****************************************************************************/
RCODE FTKAPI F_ThreadMgr::createThread(
	IF_Thread **		ppThread,
	F_THREAD_FUNC		fnThread,
	const char *		pszThreadName,
	FLMUINT				uiThreadGroup,
	FLMUINT				uiAppId,
	void *				pvParm1,
	void *				pvParm2,
	FLMUINT				uiStackSize)
{
	RCODE					rc = NE_FLM_OK;
	F_Thread *			pThread = NULL;

	if( ppThread)
	{
		*ppThread = NULL;
	}

	if( (pThread = f_new F_Thread) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pThread->startThread(
		fnThread, pszThreadName, uiThreadGroup, uiAppId,
		pvParm1, pvParm2, uiStackSize)))
	{
		goto Exit;
	}

	if( ppThread)
	{
		*ppThread = pThread;

		// Set pThread to NULL so that the object won't be released
		// below.  The application has indicated (by passing in a
		// non-NULL ppThread) that it wants to keep a reference to
		// the thread.

		pThread = NULL;
	}

Exit:

	if( pThread)
	{
		pThread->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI f_threadCreate(
	IF_Thread **			ppThread,
	F_THREAD_FUNC			fnThread,
	const char *			pszThreadName,
	FLMUINT					uiThreadGroup,
	FLMUINT					uiAppId,
	void *					pvParm1,
	void *					pvParm2,
	FLMUINT					uiStackSize)
{
	return( f_getThreadMgrPtr()->createThread( ppThread, fnThread,
		pszThreadName, uiThreadGroup, uiAppId, pvParm1, pvParm2, uiStackSize));
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI f_threadDestroy(
	IF_Thread **		ppThread)
{
	if( *ppThread != NULL)
	{
		(*ppThread)->stopThread();
		(*ppThread)->Release();
		*ppThread = NULL;
	}
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT f_getpid( void)
{
#if defined( FLM_WIN)
	return _getpid();
#elif defined( FLM_UNIX)
	return getpid();
#elif defined( FLM_NLM)
	return( (FLMUINT)f_getNLMHandle());
#else
	#error "Unsupported Platform"
#endif
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT FTKAPI f_threadId( void)
{
#ifdef FLM_WIN
	return( (FLMUINT)_threadid);
#elif defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
	return( (FLMUINT)pthread_self());
#elif defined( FLM_RING_ZERO_NLM)
	return( (FLMUINT)kCurrentThread());
#else
	#error Platform not supprted
#endif
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI FlmGetThreadInfo(
	IF_ThreadInfo **	ppThreadInfo)
{
	RCODE					rc = NE_FLM_OK;
	F_ThreadInfo *		pThreadInfo = NULL;

	if( (pThreadInfo = f_new F_ThreadInfo) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = f_getThreadMgrPtr()->getThreadInfo(
								&pThreadInfo->m_pool,
								&pThreadInfo->m_pThreadInfoArray,
								&pThreadInfo->m_uiNumThreads)))
	{
		goto Exit;
	}
	
	*ppThreadInfo = pThreadInfo;
	pThreadInfo = NULL;

Exit:

	if( pThreadInfo)
	{
		pThreadInfo->Release();
	}

	return( rc);
}
