//------------------------------------------------------------------------------
// Desc:	Contains the methods for the lock manager and lock object classes
// Tabs:	3
//
// Copyright (c) 1998-2007 Novell, Inc. All Rights Reserved.
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

/**************************************************************************
Desc: This structure is used to keep track of threads waiting for a lock
**************************************************************************/
typedef struct F_LOCK_WAITER
{
	F_SEM						hWaitSem;
	FLMUINT					uiThreadId;
	RCODE *					pRc;
	FLMUINT					uiWaitStartTime;
	FLMUINT					uiWaitTime;
	FLMBOOL					bExclReq;
	FLMINT					iPriority;
	F_TMSTAMP				StartTime;
	F_LOCK_STATS *			pLockStats;
	F_LOCK_WAITER *		pNextInList;
	F_LOCK_WAITER *		pPrevInList;
	F_LOCK_WAITER *		pNextByTime;
	F_LOCK_WAITER *		pPrevByTime;
} F_LOCK_WAITER;

/****************************************************************************
Desc:
****************************************************************************/
class F_LockObject : public IF_LockObject
{
public:

	F_LockObject();

	virtual ~F_LockObject();
	
	FLMINT FTKAPI AddRef( void);
	
	FLMINT FTKAPI Release( void);

	RCODE setupLockObject( void);

	RCODE FTKAPI lock(
		F_SEM						hWaitSem,
		FLMBOOL					bExclLock,
		FLMUINT					uiMaxWaitSecs,
		FLMINT					iPriority,
		F_LOCK_STATS *			pLockStats = NULL);

	RCODE FTKAPI unlock(
		F_LOCK_STATS *			pLockStats = NULL);
		
	FLMUINT FTKAPI getLockCount( void)
	{
		return( m_uiLockCount);
	}

	FLMUINT FTKAPI getWaiterCount( void)
	{
		return( m_uiNumWaiters);
	}
	
	RCODE FTKAPI getLockInfo(
		FLMINT					iPriority,
		eLockType *				peCurrLockType,
		FLMUINT *				puiThreadId,
		FLMUINT *				puiLockHeldTime,
		FLMUINT *				puiNumExclQueued,
		FLMUINT *				puiNumSharedQueued,
		FLMUINT *				puiPriorityCount);
		
	RCODE FTKAPI getLockInfo(
		IF_LockInfoClient *	pLockInfo);

	RCODE FTKAPI getLockQueue(
		F_LOCK_USER **			ppLockUsers);
	
	FLMBOOL FTKAPI haveHigherPriorityWaiter(
		FLMINT					iPriority);

	void FTKAPI timeoutLockWaiter(
		FLMUINT					uiThreadId);

	void FTKAPI timeoutAllWaiters( void);

private:

	void cleanupLockObject( void);

	static RCODE FTKAPI timeoutThread(
		IF_Thread *				pThread);

	void insertWaiter(
		F_LOCK_WAITER *		pLockWaiter);

	void removeWaiter(
		F_LOCK_WAITER *		pLockWaiter);

	IF_Thread *					m_pTimeoutThread;
	F_MUTEX						m_hMutex;
	FLMUINT						m_uiLockThreadId;
	FLMUINT						m_uiLockTime;
	FLMUINT						m_uiLockCount;
	F_LOCK_WAITER *			m_pFirstInList;
	F_LOCK_WAITER *			m_pLastInList;
	F_LOCK_WAITER *			m_pFirstToTimeout;
	F_LOCK_WAITER *			m_pLastToTimeout;
	FLMUINT						m_uiNumWaiters;
	FLMUINT						m_uiSharedLockCnt;
	FLMBOOL						m_bExclLock;
	F_TMSTAMP					m_StartTime;
	FLMBOOL						m_bStartTimeSet;
};

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI FlmAllocLockObject(
	IF_LockObject **	ppLockObject)
{
	RCODE					rc = NE_FLM_OK;
	F_LockObject *		pLockObject = NULL;

	if ((pLockObject = f_new F_LockObject) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pLockObject->setupLockObject()))
	{
		goto Exit;
	}
	
	*ppLockObject = pLockObject;
	pLockObject = NULL;
	
Exit:

	if( pLockObject)
	{
		pLockObject->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
F_LockObject::F_LockObject()
{
	m_pTimeoutThread = NULL;
	m_hMutex = F_MUTEX_NULL;
	m_uiLockThreadId = 0;
	m_uiLockTime = 0;
	m_uiLockCount = 0;
	m_pFirstInList = NULL;
	m_pLastInList = NULL;
	m_pFirstToTimeout = NULL;
	m_pLastToTimeout = NULL;
	m_uiNumWaiters = 0;
	m_uiSharedLockCnt = 0;
	m_bExclLock = FALSE;
	m_bStartTimeSet = FALSE;
}

/****************************************************************************
Desc:
****************************************************************************/
F_LockObject::~F_LockObject()
{
	timeoutAllWaiters();
	cleanupLockObject();
	
}
/****************************************************************************
Desc:
****************************************************************************/
void F_LockObject::cleanupLockObject( void)
{
	if( m_pTimeoutThread)
	{
		f_threadDestroy( &m_pTimeoutThread);
	}
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}
}
	
/****************************************************************************
Desc:
****************************************************************************/
FLMINT FTKAPI F_LockObject::AddRef( void)
{
	return( f_atomicInc( &m_refCnt));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMINT FTKAPI F_LockObject::Release( void)
{
	FLMINT	iRefCnt = f_atomicDec( &m_refCnt);

	if( !iRefCnt)
	{
		delete this;
	}

	return( iRefCnt);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_LockObject::setupLockObject( void)
{
	RCODE			rc = NE_FLM_OK;
	
	if( RC_BAD( rc = f_mutexCreate( &m_hMutex)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = f_threadCreate( &m_pTimeoutThread,
		F_LockObject::timeoutThread, "FTK lock timeout thread",
		0, 0, (void *)this)))
	{
		goto Exit;
	}
	
Exit:

	if( RC_BAD( rc))
	{
		cleanupLockObject();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_LockObject::timeoutThread(
	IF_Thread *			pThread)
{
	RCODE					rc = NE_FLM_OK;
	F_LockObject *		pThis = (F_LockObject *)pThread->getParm1();
	FLMUINT				uiLoop;
	FLMUINT				uiCurrTime;
	F_LOCK_WAITER *	pLockWaiter;
	
	for( ;;)
	{
		if( pThis->m_pFirstInList && pThis->m_pFirstInList->uiWaitTime)
		{
			f_mutexLock( pThis->m_hMutex);
			uiCurrTime = FLM_GET_TIMER();
		
			while( pThis->m_pFirstToTimeout && 
					 pThis->m_pFirstToTimeout->uiWaitTime &&
					 FLM_ELAPSED_TIME( uiCurrTime, 
											 pThis->m_pFirstToTimeout->uiWaitStartTime) >=
								pThis->m_pFirstToTimeout->uiWaitTime)
			{
				f_assert( !pThis->m_pFirstToTimeout->pPrevByTime);
		
				// Lock waiter has timed out.
		
				pLockWaiter = pThis->m_pFirstToTimeout;
		
				// Remove the waiter from the list
		
				pThis->removeWaiter( pLockWaiter);
		
				// Tell the waiter that the lock request timed out.
		
				*(pLockWaiter->pRc) = RC_SET( NE_FLM_LOCK_REQ_TIMEOUT);
				f_semSignal( pLockWaiter->hWaitSem);
			}
		
			f_mutexUnlock( pThis->m_hMutex);
		}

		for( uiLoop = 0; uiLoop < 20; uiLoop++)
		{
			if( pThread->getShutdownFlag())
			{
				goto Exit;
			}
		
			f_sleep( 50);
		}
	}

Exit:
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI F_LockObject::timeoutLockWaiter(
	FLMUINT					uiThreadId)
{
	FLMUINT					uiCurrTime;
	F_LOCK_WAITER *		pLockWaiter;
	F_LOCK_WAITER *		pNextWaiter;

	f_mutexLock( m_hMutex);
	uiCurrTime = FLM_GET_TIMER();

	for( pLockWaiter = m_pFirstToTimeout;
		pLockWaiter;
		pLockWaiter = pNextWaiter)
	{
		pNextWaiter = pLockWaiter->pNextByTime;

		if( pLockWaiter->uiThreadId == uiThreadId)
		{
			// Remove the lock waiter from the list

			removeWaiter( pLockWaiter);

			// Tell the waiter that the lock request timed out.

			*(pLockWaiter->pRc) = RC_SET( NE_FLM_LOCK_REQ_TIMEOUT);
			f_semSignal( pLockWaiter->hWaitSem);
			break;
		}
	}

	f_mutexUnlock( m_hMutex);
}

/****************************************************************************
Desc:	Inserts a waiter into the global list of waiters, sorted by
		its end wait time.
****************************************************************************/
void FTKAPI F_LockObject::timeoutAllWaiters( void)
{
	F_LOCK_WAITER *	pLockWaiter;
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexLock( m_hMutex);
	}
	
	while( m_pFirstInList) 
	{
		pLockWaiter = m_pFirstInList;
		removeWaiter( pLockWaiter);
		*(pLockWaiter->pRc) = RC_SET( NE_FLM_LOCK_REQ_TIMEOUT);
		f_semSignal( pLockWaiter->hWaitSem);
	}
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( m_hMutex);
	}
}

/****************************************************************************
Desc:	Inserts a waiter into the global list of waiters, sorted by
		its end wait time.
****************************************************************************/
void F_LockObject::insertWaiter(
	F_LOCK_WAITER *	pLockWaiter)
{
	F_LOCK_WAITER *	pPrevLockWaiter;

	// Link into list of waiters on this object.

	if( (pLockWaiter->pPrevInList = m_pLastInList) != NULL)
	{
		pLockWaiter->pPrevInList->pNextInList = pLockWaiter;
	}
	else
	{
		m_pFirstInList = pLockWaiter;
	}
	
	m_pLastInList = pLockWaiter;

	// Determine where in the list this lock waiter should go.

	if ((pPrevLockWaiter = m_pFirstToTimeout) != NULL)
	{
		FLMUINT	uiCurrTime = FLM_GET_TIMER();
		FLMUINT	uiElapTime;
		FLMUINT	uiTimeLeft;

		while( pPrevLockWaiter)
		{
			// Waiters with zero wait time go to end of list.
			// They never time out.

			if( !pPrevLockWaiter->uiWaitTime)
			{

				// Should go BEFORE the first zero waiter.

				pPrevLockWaiter = pPrevLockWaiter->pPrevByTime;
				break;
			}
			else if( !pLockWaiter->uiWaitTime)
			{
				if( !pPrevLockWaiter->pNextByTime)
				{
					break;
				}
				
				pPrevLockWaiter = pPrevLockWaiter->pNextByTime;
			}
			else
			{
				// Determine how much time is left on the previous
				// lock waiter's timer.  If it is less than the
				// new lock waiter's wait time, the new lock waiter
				// should be inserted AFTER it.  Otherwise, the
				// new lock waiter should be inserted BEFORE it.

				uiElapTime = FLM_ELAPSED_TIME( uiCurrTime,
										pPrevLockWaiter->uiWaitStartTime);
										
				if( uiElapTime >= pPrevLockWaiter->uiWaitTime)
				{
					uiTimeLeft = 0;
				}
				else
				{
					uiTimeLeft = pPrevLockWaiter->uiWaitTime - uiElapTime;
				}

				// New lock waiter will time out before previous lock
				// waiter - insert it BEFORE the previous lock waiter.

				if( pLockWaiter->uiWaitTime < uiTimeLeft)
				{
					pPrevLockWaiter = pPrevLockWaiter->pPrevByTime;
					break;
				}
				else
				{
					if( !pPrevLockWaiter->pNextByTime)
					{
						break;
					}

					pPrevLockWaiter = pPrevLockWaiter->pNextByTime;
				}
			}
		}
	}

	// Insert into list AFTER pPrevLockWaiter.

	if( (pLockWaiter->pPrevByTime = pPrevLockWaiter) != NULL)
	{
		if ((pLockWaiter->pNextByTime = pPrevLockWaiter->pNextByTime) != NULL)
		{
			pLockWaiter->pNextByTime->pPrevByTime = pLockWaiter;
		}
		
		pPrevLockWaiter->pNextByTime = pLockWaiter;
	}
	else
	{
		if( (pLockWaiter->pNextByTime = m_pFirstToTimeout) != NULL)
		{
			m_pFirstToTimeout->pPrevByTime = pLockWaiter;
		}
		
		m_pFirstToTimeout = pLockWaiter;
	}

	m_uiNumWaiters++;
}

/****************************************************************************
Desc:
****************************************************************************/
void F_LockObject::removeWaiter(
	F_LOCK_WAITER *		pLockWaiter)
{
	if (pLockWaiter->pNextByTime)
	{
		pLockWaiter->pNextByTime->pPrevByTime = pLockWaiter->pPrevByTime;
	}

	if (pLockWaiter->pPrevByTime)
	{
		pLockWaiter->pPrevByTime->pNextByTime = pLockWaiter->pNextByTime;
	}
	else
	{
		m_pFirstToTimeout = pLockWaiter->pNextByTime;
	}
	
	if (pLockWaiter->pNextInList)
	{
		pLockWaiter->pNextInList->pPrevInList = pLockWaiter->pPrevInList;
	}
	else
	{
		m_pLastInList = pLockWaiter->pPrevInList;
	}

	if (pLockWaiter->pPrevInList)
	{
		pLockWaiter->pPrevInList->pNextInList = pLockWaiter->pNextInList;
	}
	else
	{
		m_pFirstInList = pLockWaiter->pNextInList;
	}
	
	f_assert( m_uiNumWaiters > 0);
	m_uiNumWaiters--;
}

/****************************************************************************
Desc:	Lock this object.  If object is locked, wait the specified
		number of seconds.
****************************************************************************/
RCODE FTKAPI F_LockObject::lock(
	F_SEM					hWaitSem,
	FLMBOOL				bExclReq,
	FLMUINT				uiMaxWaitSecs,
	FLMINT				iPriority,
	F_LOCK_STATS *		pLockStats)
{
	RCODE					rc = NE_FLM_OK;
	RCODE					TempRc;
	F_LOCK_WAITER		LockWait;
	FLMBOOL				bMutexLocked = FALSE;

	f_assert( hWaitSem != F_SEM_NULL);

	f_mutexLock( m_hMutex);
	bMutexLocked = TRUE;

	if (m_pFirstInList || m_bExclLock || (bExclReq && m_uiSharedLockCnt))
	{
		// Object is locked by another thread, wait to get lock.

		if( !uiMaxWaitSecs)
		{
			rc = RC_SET( NE_FLM_LOCK_REQ_TIMEOUT);
			goto Exit;
		}

		// Set up to wait for the lock.

		f_memset( &LockWait, 0, sizeof( LockWait));
		LockWait.hWaitSem = hWaitSem;

		LockWait.uiThreadId = f_threadId();
		LockWait.pRc = &rc;
		
		rc = RC_SET( NE_FLM_FAILURE);
		
		LockWait.bExclReq = bExclReq;
		LockWait.iPriority = iPriority;
		LockWait.uiWaitStartTime = (FLMUINT)FLM_GET_TIMER();
		
		if( bExclReq && pLockStats)
		{
			f_timeGetTimeStamp( &LockWait.StartTime);
			LockWait.pLockStats = pLockStats;
		}
		
		if( uiMaxWaitSecs >= 0xFF)
		{
			LockWait.uiWaitTime = 0;
		}
		else
		{
			LockWait.uiWaitTime = FLM_SECS_TO_TIMER_UNITS( uiMaxWaitSecs);
		}

		// Link to list of global waiters - ordered by end time.

		insertWaiter( &LockWait);
		
		f_mutexUnlock( m_hMutex);
		bMutexLocked = FALSE;

		// Now just wait to be signaled.

		if( RC_BAD( TempRc = f_semWait( hWaitSem, F_WAITFOREVER)))
		{
			RC_UNEXPECTED_ASSERT( TempRc);
			rc = TempRc;
		}
		else
		{
			// Process that signaled us better set the rc to something
			// besides NE_FLM_FAILURE.

			if (rc == NE_FLM_FAILURE)
			{
				RC_UNEXPECTED_ASSERT( rc);
			}
		}
	}
	else
	{
		// Object is NOT locked in a conflicting mode.  Grant the
		// lock immediately.

		m_uiLockThreadId = f_threadId();
		m_bExclLock = bExclReq;
		
		if (!bExclReq)
		{
			m_uiSharedLockCnt++;
		}
		else
		{
			m_uiLockTime = FLM_GET_TIMER();
			f_assert( m_uiSharedLockCnt == 0);

			// Take care of statistics gathering.

			if (pLockStats)
			{
				// If m_bStartTimeSet is TRUE, we started the
				// clock the last time nobody had the exclusive
				// lock, so we need to sum up idle time now.

				if (m_bStartTimeSet)
				{
					f_addElapsedTime( &m_StartTime, 
						&pLockStats->NoLocks.ui64ElapMilli);
					pLockStats->NoLocks.ui64Count++;
				}

				// Restart the clock for this locker.

				f_timeGetTimeStamp( &m_StartTime);
				m_bStartTimeSet = TRUE;
			}
			else
			{
				m_bStartTimeSet = FALSE;
			}
		}
	}
	
Exit:

	if (RC_OK( rc))
	{
		m_uiLockCount++;
	}
	
	if (bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}
	
	return( rc);
}

/****************************************************************************
Desc:	Unlock this object.  If there is a pending lock request, give
		the lock to the next waiter.
****************************************************************************/
RCODE FTKAPI F_LockObject::unlock(
	F_LOCK_STATS *		pLockStats)
{
	RCODE					rc = NE_FLM_OK;
	F_SEM					hWaitSem;
	F_LOCK_WAITER *	pLockWaiter;

	f_mutexLock( m_hMutex);

	if( m_bExclLock)
	{
		f_assert( !m_uiSharedLockCnt);
		m_bExclLock = FALSE;

		// Record how long the lock was held, if we were tracking it.

		if( pLockStats && m_bStartTimeSet)
		{
			f_addElapsedTime( &m_StartTime, &pLockStats->HeldLock.ui64ElapMilli);
			pLockStats->HeldLock.ui64Count++;
		}
		
		m_bStartTimeSet = FALSE;
	}
	else
	{
		f_assert( m_uiSharedLockCnt > 0);
		m_uiSharedLockCnt--;
	}

	m_uiLockThreadId = 0;

	// See if we need to signal the next set of waiters

	if( m_pFirstInList && !m_uiSharedLockCnt)
	{
		m_bExclLock = m_pFirstInList->bExclReq;
		
		while( m_pFirstInList)
		{
			if (!m_bExclLock)
			{
				m_uiSharedLockCnt++;
			}

			pLockWaiter = m_pFirstInList;
			hWaitSem = pLockWaiter->hWaitSem;

			// Unlink the waiter from the list of waiters on this lock object.
			//
			// IMPORTANT NOTE: Do NOT signal the semaphore until AFTER
			// doing this unlinking.  This is because F_LOCK_WAITER
			// structures exist only on the stack of the thread
			// being signaled.  If we tried to assign m_pFirstInList after
			// signaling the semaphore, the F_LOCK_WAITER structure could
			// disappear and m_pFirstInList would get garbage.

			removeWaiter( pLockWaiter);

			// Update statistics for the waiter.

			if (pLockWaiter->pLockStats)
			{
				f_addElapsedTime( &pLockWaiter->StartTime,
								&pLockWaiter->pLockStats->WaitingForLock.ui64ElapMilli);
				pLockWaiter->pLockStats->WaitingForLock.ui64Count++;
			}

			// Grant the lock to this waiter and signal the thread
			// to wake it up.

			m_uiLockThreadId = pLockWaiter->uiThreadId;
			if (m_bExclLock)
			{
				m_uiLockTime = FLM_GET_TIMER();

				// Restart the stats timer

				if (pLockStats)
				{
					m_bStartTimeSet = TRUE;
					f_timeGetTimeStamp( &m_StartTime);
				}
			}

			*(pLockWaiter->pRc) = NE_FLM_OK;
			f_semSignal( hWaitSem);

			// If the next waiter is not a shared lock request or
			// the lock that was granted was exclusive, we stop
			// here.

			if (m_bExclLock ||
				 (m_pFirstInList && m_pFirstInList->bExclReq))
			{
				break;
			}
		}
	}

	// Start timer, if not already running.  If the timer is not set at
	// this point, it will be because nobody has been granted the exclusive
	// lock.  If someone was granted the exclusive lock, the timer would
	// have been started above.  We start it here so we can track idle
	// time.

	if (pLockStats && !m_bStartTimeSet)
	{
		f_assert( !m_bExclLock);
		m_bStartTimeSet = TRUE;
		f_timeGetTimeStamp( &m_StartTime);
	}

	f_mutexUnlock( m_hMutex);
	return( rc);
}

/****************************************************************************
Desc: Returns information about the pending lock requests.
****************************************************************************/
RCODE FTKAPI F_LockObject::getLockInfo(
	FLMINT				iPriority,
	eLockType *			peCurrLockType,
	FLMUINT *			puiThreadId,
	FLMUINT *			puiLockHeldTime,
	FLMUINT *			puiNumExclQueued,
	FLMUINT *			puiNumSharedQueued,
	FLMUINT *			puiPriorityCount)
{
	F_LOCK_WAITER *	pLockWaiter;

	if( puiNumExclQueued)
	{
		*puiNumExclQueued = 0;
	}

	if( puiNumSharedQueued)
	{
		*puiNumSharedQueued = 0;
	}

	if( puiPriorityCount)
	{
		*puiPriorityCount = 0;
	}
	
	if( puiThreadId)
	{
		*puiThreadId = 0;
	}
	
	if( puiLockHeldTime)
	{
		*puiLockHeldTime = 0;
	}
	
	f_mutexLock( m_hMutex);

	// Get the type of lock, if any.

	if (m_bExclLock)
	{
		if( peCurrLockType)
		{
			*peCurrLockType = FLM_LOCK_EXCLUSIVE;
		}
		
		if( puiThreadId)
		{
			*puiThreadId = m_uiLockThreadId;
		}

		if( puiLockHeldTime)
		{
			*puiLockHeldTime = FLM_TIMER_UNITS_TO_MILLI( 
						FLM_ELAPSED_TIME( FLM_GET_TIMER(), m_uiLockTime));
		}
	}
	else if (m_uiSharedLockCnt)
	{
		if( peCurrLockType)
		{
			*peCurrLockType = FLM_LOCK_SHARED;
		}
	}
	else
	{
		if( peCurrLockType)
		{
			*peCurrLockType = FLM_LOCK_NONE;
		}
	}

	// Get information on pending lock requests.

	if( puiNumExclQueued || puiNumSharedQueued || puiPriorityCount)
	{
		pLockWaiter = m_pFirstInList;
		for( ; pLockWaiter; pLockWaiter = pLockWaiter->pNextInList)
		{
			// Count the number of exclusive and shared waiters.

			if (pLockWaiter->bExclReq)
			{
				if( puiNumExclQueued)
				{
					(*puiNumExclQueued)++;
				}
			}
			else
			{
				if( puiNumSharedQueued)
				{
					(*puiNumSharedQueued)++;
				}
			}

			// Count the number of waiters at or above input priority.

			if (pLockWaiter->iPriority >= iPriority)
			{
				if( puiPriorityCount)
				{
					(*puiPriorityCount)++;
				}
			}
		}
	}

	f_mutexUnlock( m_hMutex);
	return( NE_FLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_LockObject::getLockInfo(
	IF_LockInfoClient *	pLockInfo)
{
	RCODE						rc = NE_FLM_OK;
	F_LOCK_WAITER *		pLockWaiter;
	FLMUINT					uiCnt;
	FLMUINT					uiElapTime;
	FLMUINT					uiCurrTime;
	FLMUINT					uiMilli;

	f_mutexLock( m_hMutex);
	uiCurrTime = FLM_GET_TIMER();

	if( !m_uiNumWaiters && !m_uiLockThreadId)
	{
		pLockInfo->setLockCount( 0);
		goto Exit;
	}
	
	uiCnt = m_uiNumWaiters + 1;
	
	if( pLockInfo->setLockCount( uiCnt) == FALSE)
	{
		goto Exit;
	}

	// Output the lock holder first.

	uiElapTime = FLM_ELAPSED_TIME( uiCurrTime, m_uiLockTime);
	uiMilli = FLM_TIMER_UNITS_TO_MILLI( uiElapTime);
	
	if( pLockInfo->addLockInfo( 0, m_uiLockThreadId, uiMilli) == FALSE)
	{
		goto Exit;
	}
	uiCnt--;

	// Output the lock waiters.

	pLockWaiter = m_pFirstInList;
	while( pLockWaiter && uiCnt)
	{
		uiElapTime = FLM_ELAPSED_TIME( uiCurrTime, pLockWaiter->uiWaitStartTime);
		uiMilli = FLM_TIMER_UNITS_TO_MILLI( uiElapTime);
		
		if( pLockInfo->addLockInfo( (m_uiNumWaiters - uiCnt) + 1,
			pLockWaiter->uiThreadId, uiMilli) == FALSE)
		{
			goto Exit;
		}
		
		pLockWaiter = pLockWaiter->pNextInList;
		uiCnt--;
	}
	
	f_assert( !pLockWaiter && !uiCnt);

Exit:

	f_mutexUnlock( m_hMutex);
	return( rc);
}

/****************************************************************************
Desc:	Return a list that includes the current lock holder as well as
		the lock waiters.
****************************************************************************/
RCODE FTKAPI F_LockObject::getLockQueue(
	F_LOCK_USER **		ppLockUsers)
{
	RCODE					rc = NE_FLM_OK;
	F_LOCK_USER *		pLockUser;
	F_LOCK_WAITER *	pLockWaiter;
	FLMUINT				uiCnt;
	FLMUINT				uiElapTime;
	FLMUINT				uiCurrTime;

	f_mutexLock( m_hMutex);
	uiCurrTime = (FLMUINT)FLM_GET_TIMER();
	
	if( !m_uiNumWaiters && !m_uiLockThreadId)
	{
		*ppLockUsers = NULL;
		goto Exit;
	}
	
	uiCnt = m_uiNumWaiters + 1;

	if( RC_BAD( rc = f_alloc( 
		sizeof( F_LOCK_USER) * (uiCnt + 1), &pLockUser)))
	{
		goto Exit;
	}

	*ppLockUsers = pLockUser;

	// Output the lock holder first.

	pLockUser->uiThreadId = m_uiLockThreadId;
	uiElapTime = FLM_ELAPSED_TIME( uiCurrTime, m_uiLockTime);
	pLockUser->uiTime = FLM_TIMER_UNITS_TO_MILLI( uiElapTime);
	pLockUser++;
	uiCnt--;

	// Output the lock waiters.

	pLockWaiter = m_pFirstInList;
	while( pLockWaiter && uiCnt)
	{
		pLockUser->uiThreadId = pLockWaiter->uiThreadId;
		uiElapTime = FLM_ELAPSED_TIME( uiCurrTime,
									pLockWaiter->uiWaitStartTime);
		pLockUser->uiTime = FLM_TIMER_UNITS_TO_MILLI( uiElapTime);
		pLockWaiter = pLockWaiter->pNextInList;
		pLockUser++;
		uiCnt--;
	}
	flmAssert( pLockWaiter == NULL && uiCnt == 0);

	// Zero out the last one.

	f_memset( pLockUser, 0, sizeof( F_LOCK_USER));
	
Exit:

	f_mutexUnlock( m_hMutex);
	return( rc);
}

/****************************************************************************
Desc: Returns TRUE if there are lock waiters with a priority > iPriority
****************************************************************************/
FLMBOOL FTKAPI F_LockObject::haveHigherPriorityWaiter(
	FLMINT				iPriority)
{
	F_LOCK_WAITER *	pLockWaiter;
	FLMBOOL				bWaiters = FALSE;

	f_mutexLock( m_hMutex);

	pLockWaiter = m_pFirstInList;
	for( ; pLockWaiter; pLockWaiter = pLockWaiter->pNextInList)
	{
		// If we find a waiter with a priority > the specified
		// priority, we're done.

		if( pLockWaiter->iPriority > iPriority)
		{
			bWaiters = TRUE;
			break;
		}
	}

	f_mutexUnlock( m_hMutex);
	return( bWaiters);
}
