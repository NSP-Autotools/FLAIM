//------------------------------------------------------------------------------
// Desc:	This file contains mutex and semaphore functions
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

/****************************************************************************
Desc:
****************************************************************************/
typedef struct
{
	F_MUTEX						hMutex;
	F_NOTIFY_LIST_ITEM *		pNotifyList;
	FLMUINT						uiWriteThread;
	FLMINT						iRefCnt;
} F_RWLOCK_IMP;

/****************************************************************************
Desc:
****************************************************************************/
#if (defined( FLM_UNIX) || defined( FLM_LIBC_NLM)) && !defined( FLM_SOLARIS)
typedef struct
{
	pthread_mutex_t		lock;
	pthread_cond_t			cond;
	int						count;
} sema_t;
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_WIN)
typedef struct
{
	HANDLE					hWinSem;
	FLMATOMIC				uiSignalCount;
} sema_t;
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_WIN)
RCODE FTKAPI f_mutexCreate(
	F_MUTEX *	phMutex)
{
	f_assert( phMutex != NULL);

	if( (*phMutex = (F_MUTEX)malloc( sizeof( F_INTERLOCK))) == F_MUTEX_NULL)
	{
		return( RC_SET( NE_FLM_MEM));
	}

	((F_INTERLOCK *)(*phMutex))->locked = 0;
#ifdef FLM_DEBUG
	((F_INTERLOCK *)(*phMutex))->uiThreadId = 0;
	((F_INTERLOCK *)(*phMutex))->lockedCount = 0;
	((F_INTERLOCK *)(*phMutex))->waitCount = 0;
#endif

	return( NE_FLM_OK);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_WIN)
void FTKAPI f_mutexDestroy(
	F_MUTEX *	phMutex)
{
	f_assert( phMutex != NULL);

	if (*phMutex != F_MUTEX_NULL)
	{
		free( *phMutex);
		*phMutex = F_MUTEX_NULL;
	}
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if (defined( FLM_UNIX) || defined( FLM_LIBC_NLM)) && !defined( FLM_SOLARIS)
RCODE FTKAPI f_mutexCreate(
	F_MUTEX *			phMutex)
{
	RCODE								rc = NE_FLM_OK;
	pthread_mutexattr_t *		pMutexAttr = NULL;

	f_assert( phMutex != NULL);

	// NOTE: Cannot call f_alloc because the memory initialization needs
	// to be able to set up mutexes.

	if ((*phMutex = (F_MUTEX)malloc( 
		sizeof( pthread_mutex_t))) == F_MUTEX_NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}

#if defined( FLM_DEBUG) && defined( FLM_LINUX)
	{
		pthread_mutexattr_t			mutexAttr;
	
		if( !pthread_mutexattr_init( &mutexAttr))
		{
			pMutexAttr = &mutexAttr;
			pthread_mutexattr_settype( pMutexAttr, PTHREAD_MUTEX_ERRORCHECK_NP);
		}
	}
#endif

	if( pthread_mutex_init( (pthread_mutex_t *)*phMutex, pMutexAttr) != 0)
	{
		// NOTE: Cannot call f_free because we had to use malloc up above due
		// to the fact that the memory subsystem uses a mutex before itis
		// completely ready to go.

		free( *phMutex);
		*phMutex = F_MUTEX_NULL;
		rc = RC_SET( NE_FLM_COULD_NOT_CREATE_MUTEX);
		goto Exit;
	}

Exit:

	if( pMutexAttr)
	{
		pthread_mutexattr_destroy( pMutexAttr);
	}

	return( rc);
}
#endif
			  
/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_SOLARIS)
RCODE FTKAPI f_mutexCreate(
	F_MUTEX *			phMutex)
{
	RCODE					rc = NE_FLM_OK;
	lwp_mutex_t			defaultMutex = DEFAULTMUTEX;

	f_assert( phMutex != NULL);

	// NOTE: Cannot call f_alloc because the memory initialization needs
	// to be able to set up mutexes.

	if ((*phMutex = (F_MUTEX)malloc( sizeof( lwp_mutex_t))) == F_MUTEX_NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	f_memcpy( *phMutex, &defaultMutex, sizeof( lwp_mutex_t));

Exit:

	return( rc);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if (defined( FLM_UNIX) || defined( FLM_LIBC_NLM)) && !defined( FLM_SOLARIS)
void FTKAPI f_mutexDestroy(
	F_MUTEX *	phMutex)
{
	f_assert( phMutex != NULL);

	if (*phMutex != F_MUTEX_NULL)
	{
		pthread_mutex_destroy( (pthread_mutex_t *)*phMutex);

		// NOTE: Cannot call f_free because we had to use malloc up above due
		// to the fact that the memory subsystem uses a mutex before it is
		// completely ready to go.

		free( *phMutex);
		*phMutex = F_MUTEX_NULL;
	}
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_SOLARIS) 
void FTKAPI f_mutexDestroy(
	F_MUTEX *	phMutex)
{
	f_assert( phMutex != NULL);

	if (*phMutex != F_MUTEX_NULL)
	{
		free( *phMutex);
		*phMutex = F_MUTEX_NULL;
	}
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if (defined( FLM_UNIX) || defined( FLM_LIBC_NLM)) && !defined( FLM_SOLARIS)
void FTKAPI f_mutexLock(
	F_MUTEX		hMutex)
{
	(void)pthread_mutex_lock( (pthread_mutex_t *)hMutex);
}
#endif
	
/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_SOLARIS)
void FTKAPI f_mutexLock(
	F_MUTEX		hMutex)
{
	for( ;;)
	{
		if( _lwp_mutex_lock( (lwp_mutex_t *)hMutex) == 0)
		{
			break;
		}
	}
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if (defined( FLM_UNIX) || defined( FLM_LIBC_NLM)) && !defined( FLM_SOLARIS)
void FTKAPI f_mutexUnlock(
	F_MUTEX		hMutex)
{
	(void)pthread_mutex_unlock( (pthread_mutex_t *)hMutex);
}
#endif
	
/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_SOLARIS)
void FTKAPI f_mutexUnlock(
	F_MUTEX		hMutex)
{
	_lwp_mutex_unlock( (lwp_mutex_t *)hMutex);
}
#endif

#undef f_assertMutexLocked
#undef f_assertMutexNotLocked

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_NLM)
void FTKAPI f_assertMutexLocked(
	F_MUTEX)
{
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_NLM)
void FTKAPI f_assertMutexNotLocked(
	F_MUTEX)
{
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_WIN
void FTKAPI f_assertMutexLocked(
	F_MUTEX		hMutex)
{
#ifdef FLM_DEBUG
	f_assert( ((F_INTERLOCK *)hMutex)->locked == 1);
	f_assert( ((F_INTERLOCK *)hMutex)->uiThreadId == _threadid);
#else
	F_UNREFERENCED_PARM( hMutex);
#endif
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_WIN
void FTKAPI f_assertMutexNotLocked(
	F_MUTEX		hMutex)
{
#ifdef FLM_DEBUG
	f_assert( ((F_INTERLOCK *)hMutex)->uiThreadId != _threadid);
#else
	F_UNREFERENCED_PARM( hMutex);
#endif
}
#endif

/*************************************************************************
Desc:
*************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE FTKAPI f_mutexCreate(
	F_MUTEX *	phMutex)
{
	if( (*phMutex = (F_MUTEX)kMutexAlloc( (BYTE *)"FTK_MUTEX")) == F_MUTEX_NULL)
	{
		return( RC_SET( NE_FLM_MEM));
	}

	return( NE_FLM_OK);
}
#endif

/*************************************************************************
Desc:
*************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
void FTKAPI f_mutexDestroy(
	F_MUTEX *	phMutex)
{
	if (*phMutex != F_MUTEX_NULL)
	{
		(void)kMutexFree( (MUTEX)(*phMutex));
		*phMutex = F_MUTEX_NULL;
	}
}
#endif

/*************************************************************************
Desc:
*************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
void FTKAPI f_mutexLock( 
	F_MUTEX		hMutex)
{
	(void)kMutexLock( (MUTEX)hMutex);
}
#endif

/*************************************************************************
Desc:
*************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
void FTKAPI f_mutexUnlock(
	F_MUTEX		hMutex)
{
	(void)kMutexUnlock( (MUTEX)hMutex);
}
#endif
	
/****************************************************************************
Desc:	Initializes a semaphore handle on UNIX
****************************************************************************/
#if (defined( FLM_UNIX) || defined( FLM_LIBC_NLM)) && !defined( FLM_SOLARIS)
FINLINE int sema_init(
	sema_t *			pSem)
{
	int iErr = 0;

	if( (iErr = pthread_mutex_init( &pSem->lock, NULL)) < 0)
	{
		goto Exit;
	}

	if( (iErr = pthread_cond_init( &pSem->cond, NULL)) < 0)
	{
		pthread_mutex_destroy( &pSem->lock);
		goto Exit;
	}

	pSem->count = 0;

Exit:

	return( iErr);
}
#endif

/****************************************************************************
Desc:	Frees a semaphore handle on UNIX
****************************************************************************/
#if (defined( FLM_UNIX) || defined( FLM_LIBC_NLM)) && !defined( FLM_SOLARIS)
FINLINE void sema_destroy(
	sema_t *			pSem)
{
	pthread_mutex_destroy( &pSem->lock);
	pthread_cond_destroy( &pSem->cond);
}
#endif

/****************************************************************************
Desc:	Waits for a semaphore to be signaled on UNIX
****************************************************************************/
#if (defined( FLM_UNIX) || defined( FLM_LIBC_NLM)) && !defined( FLM_SOLARIS)
FINLINE int _sema_wait(
	sema_t *			pSem)
{
	int	iErr = 0;

	pthread_mutex_lock( &pSem->lock);
	while( !pSem->count)
	{
		if( (iErr = pthread_cond_wait( &pSem->cond, &pSem->lock)) != 0)
		{
			if( iErr == EINTR)
			{
				iErr = 0;
			}
			else
			{
				f_assert( 0);
				goto Exit;
			}
		}
	}

	pSem->count--;
	f_assert( pSem->count >= 0);

Exit:

	pthread_mutex_unlock( &pSem->lock);
	return( iErr);
}
#endif

/****************************************************************************
Desc:	Waits for a semaphore to be signaled on Solaris
****************************************************************************/
#if defined( FLM_SOLARIS)
FINLINE int _sema_wait(
	sema_t *			pSem)
{
	int	iErr = 0;

	for( ;;)
	{
		if( (iErr = sema_wait( pSem)) != 0)
		{
			if( iErr == EINTR)
			{
				iErr = 0;
				continue;
			}
			else
			{
				f_assert( 0);
				goto Exit;
			}
		}

		break;
	}

Exit:

	return( iErr);
}
#endif

/****************************************************************************
Desc:	Waits a specified number of milliseconds for a semaphore
		to be signaled on UNIX
****************************************************************************/
#if (defined( FLM_UNIX) || defined( FLM_LIBC_NLM)) && !defined( FLM_SOLARIS)
FINLINE int _sema_timedwait(
	sema_t *			pSem,
	unsigned int	msecs)
{
	int					iErr = 0;
   struct timeval		now;
	struct timespec	abstime;

   // If timeout is F_WAITFOREVER, do sem_wait.

   if( msecs == F_WAITFOREVER)
   {
      iErr = _sema_wait( pSem);
      return( iErr);
   }

   gettimeofday( &now, NULL);
	abstime.tv_sec = now.tv_sec + ((msecs) ? (msecs / 1000) : 0);
	abstime.tv_nsec = ( now.tv_usec + ((msecs % 1000) *	1000)) * 1000;

	pthread_mutex_lock( &pSem->lock);

Restart:

	while( !pSem->count)
	{
		if( (iErr = pthread_cond_timedwait( &pSem->cond,
			&pSem->lock, &abstime)) != 0)
		{
			if( iErr == EINTR)
			{
				iErr = 0;
				goto Restart;
			}
			goto Exit;
		}
	}

	pSem->count--;
	f_assert( pSem->count >= 0);

Exit:

	pthread_mutex_unlock( &pSem->lock);
	return( iErr);
}
#endif

/****************************************************************************
Desc:	Waits a specified number of milliseconds for a semaphore
		to be signaled on UNIX
****************************************************************************/
#if defined( FLM_SOLARIS)
FINLINE int _sema_timedwait(
	sema_t *			pSem,
	unsigned int	msecs)
{
	int					iErr = 0;

   // If timeout is F_WAITFOREVER, do sem_wait.

   if( msecs == F_WAITFOREVER)
   {
      iErr = _sema_wait( pSem);
      return( iErr);
   }

	for( ;;)
	{
		if( (iErr = sema_trywait( pSem)) != 0)
		{
			if( iErr == EINTR)
			{
				iErr = 0;
			}

			f_sleep( f_min( msecs, 10));
			msecs -= f_min( msecs, 10);

			if( !msecs)
			{
				iErr = -1;
				goto Exit;
			}

			continue;
		}
	}

Exit:

	return( iErr);
}
#endif

/****************************************************************************
Desc:	Signals a semaphore on UNIX
****************************************************************************/
#if (defined( FLM_UNIX) || defined( FLM_LIBC_NLM)) && !defined( FLM_SOLARIS)
int sema_signal(
	sema_t *			pSem)
{
	pthread_mutex_lock( &pSem->lock);
	pSem->count++;
	f_assert( pSem->count > 0);
	pthread_cond_signal( &pSem->cond);
	pthread_mutex_unlock( &pSem->lock);

	return( 0);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
RCODE f_semCreate(
	F_SEM *		phSem)
{
	RCODE			rc = NE_FLM_OK;

	f_assert( phSem != NULL);

	if( RC_BAD( rc = f_alloc( sizeof( sema_t), phSem)))
	{
		goto Exit;
	}

#if defined( FLM_SOLARIS)
	if( sema_init( (sema_t *)*phSem, 0, USYNC_THREAD, NULL) < 0) 
#else
	if( sema_init( (sema_t *)*phSem) < 0)
#endif
	{
		f_free( phSem);
		*phSem = F_SEM_NULL;
		rc = RC_SET( NE_FLM_COULD_NOT_CREATE_SEMAPHORE);
		goto Exit;
	}

Exit:

	return( rc);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
void f_semDestroy(
	F_SEM  *		phSem)
{
	f_assert( phSem != NULL);

	if (*phSem != F_SEM_NULL)
	{
		sema_destroy( (sema_t *)*phSem);
		f_free( phSem);
		*phSem = F_SEM_NULL;
	}
}
#endif

/****************************************************************************
Desc:   Get the lock on a semaphore - p operation
****************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
RCODE f_semWait(
	F_SEM			hSem,
	FLMUINT		uiTimeout)
{
	RCODE			rc	= NE_FLM_OK;

	f_assert( hSem != F_SEM_NULL);

	// Catch the F_WAITFOREVER flag so we can directly call _sema_wait
	// instead of passing F_WAITFOREVER through to _sema_timedwait.
	// Note that on AIX the datatype of the uiTimeout (in the timespec
	// struct) is surprisingly a signed int, which makes this catch
	// essential.

	if( uiTimeout == F_WAITFOREVER)
	{
		if( _sema_wait( (sema_t *)hSem))
		{
			rc = RC_SET( NE_FLM_ERROR_WAITING_ON_SEMAPHORE);
		}
	}
	else
	{
		if( _sema_timedwait( (sema_t *)hSem, (unsigned int)uiTimeout))
		{
			rc = RC_SET( NE_FLM_WAIT_TIMEOUT);
		}
	}

	return( rc);
}
#endif

/****************************************************************************
Desc:   Get the lock on a semaphore - p operation
****************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
void FTKAPI f_semSignal(
	F_SEM			hSem)
{
#if defined( FLM_SOLARIS)
	sema_post( (sema_t *)hSem);
#else
	sema_signal( (sema_t *)hSem);
#endif
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
FLMUINT FTKAPI f_semGetSignalCount(
	F_SEM							hSem)
{
	return( (FLMUINT)((sema_t *)hSem)->count);
}
#endif

/*************************************************************************
Desc:
*************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE FTKAPI f_semCreate(
	F_SEM *		phSem)
{
	if( (*phSem = (F_SEM)kSemaphoreAlloc( (BYTE *)"FTK_SEM", 0)) == F_SEM_NULL)
	{
		return( RC_SET( NE_FLM_MEM));
	}
	
	return( NE_FLM_OK);
}
#endif

/*************************************************************************
Desc:
*************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
void FTKAPI f_semDestroy(
	F_SEM *		phSem)
{
	if (*phSem != F_SEM_NULL)
	{
		(void)kSemaphoreFree( (SEMAPHORE)(*phSem));
		*phSem = F_SEM_NULL;
	}
}
#endif

/*************************************************************************
Desc:
*************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE FTKAPI f_semWait(
	F_SEM			hSem,
	FLMUINT		uiTimeout)
{
	RCODE			rc = NE_FLM_OK;
	
	if( uiTimeout == F_WAITFOREVER)
	{
		if( kSemaphoreWait( (SEMAPHORE)hSem) != 0)
		{
			rc = RC_SET( NE_FLM_ERROR_WAITING_ON_SEMAPHORE);
		}
	}
	else
	{
		if( kSemaphoreTimedWait( (SEMAPHORE)hSem, (UINT)uiTimeout) != 0)
		{
			rc = RC_SET( NE_FLM_WAIT_TIMEOUT);
		}
	}
	
	return( rc);
}
#endif

/*************************************************************************
Desc:
*************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
void FTKAPI f_semSignal(
	F_SEM			hSem)
{
	(void)kSemaphoreSignal( (SEMAPHORE)hSem);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
FLMUINT FTKAPI f_semGetSignalCount(
	F_SEM							hSem)
{
	return( (FLMUINT)kSemaphoreExamineCount( (SEMAPHORE)hSem));
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_WIN
void FTKAPI f_mutexLock(
	F_MUTEX			hMutex)
{
	F_INTERLOCK *		pInterlock = (F_INTERLOCK *)hMutex;

#ifdef FLM_DEBUG
	if( pInterlock->locked)
	{
		f_assert( pInterlock->uiThreadId != _threadid);
	}
#endif

	while( f_atomicExchange( &pInterlock->locked, 1) != 0)
	{
#ifdef FLM_DEBUG
		f_atomicInc( &pInterlock->waitCount);
#endif
		Sleep( 0);
	}

#ifdef FLM_DEBUG
	f_assert( pInterlock->uiThreadId == 0);
	pInterlock->uiThreadId = _threadid;
	f_atomicInc( &pInterlock->lockedCount);
#endif
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_WIN
void FTKAPI f_mutexUnlock(
	F_MUTEX		hMutex)
{
	F_INTERLOCK *		pInterlock = (F_INTERLOCK *)hMutex;

	f_assert( pInterlock->locked == 1);
#ifdef FLM_DEBUG
	f_assert( pInterlock->uiThreadId == _threadid);
	pInterlock->uiThreadId = 0;
#endif
	f_atomicExchange( &pInterlock->locked, 0);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_WIN
RCODE FTKAPI f_semCreate(
	F_SEM *		phSem)
{
	RCODE			rc = NE_FLM_OK;
	sema_t *		pSem = NULL;

	f_assert( phSem != NULL);
	f_assert( *phSem == F_SEM_NULL);

	if( RC_BAD( rc = f_calloc( sizeof( sema_t), &pSem)))
	{
		goto Exit;
	}

	if( (pSem->hWinSem = CreateSemaphore( (LPSECURITY_ATTRIBUTES)NULL,
		0, 10000, NULL )) == NULL)
	{
		rc = RC_SET( NE_FLM_COULD_NOT_CREATE_SEMAPHORE);
	}
	
	*phSem = pSem;
	pSem = NULL;

Exit:

	if( pSem)
	{
		if( pSem->hWinSem)
		{
			CloseHandle( pSem->hWinSem);
		}
		
		f_free( &pSem);
	}

	return( rc);
}
#endif
	
/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_WIN
void FTKAPI f_semDestroy(
	F_SEM *		phSem)
{
	sema_t *		pSem = (sema_t *)(*phSem); 
	
	if( pSem)
	{
		if( pSem->hWinSem)
		{
			CloseHandle( pSem->hWinSem);
		}
		
		f_free( &pSem);
	}
		
	*phSem = F_SEM_NULL;
}
#endif
	
/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_WIN
RCODE FTKAPI f_semWait(
	F_SEM			hSem,
	FLMUINT		uiTimeout)
{
	DWORD			dwStatus;

	for( ;;)
	{
		dwStatus = WaitForSingleObjectEx( ((sema_t *)hSem)->hWinSem,
														uiTimeout, true);

		if( dwStatus == WAIT_OBJECT_0)
		{
			f_atomicDec( &((sema_t *)hSem)->uiSignalCount);
			return( NE_FLM_OK);
		}

		if( dwStatus == WAIT_IO_COMPLETION)
		{
			continue;
		}

		break;
	}

	return( RC_SET( NE_FLM_WAIT_TIMEOUT));
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_WIN
void FTKAPI f_semSignal(
	F_SEM			hSem)
{
	f_atomicInc( &((sema_t *)hSem)->uiSignalCount);
	(void)ReleaseSemaphore( ((sema_t *)hSem)->hWinSem, 1, NULL);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_WIN
FLMUINT FTKAPI f_semGetSignalCount(
	F_SEM							hSem)
{
	return( ((sema_t *)hSem)->uiSignalCount);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC void f_rwlockNotify(
	F_RWLOCK_IMP *				pReadWriteLock)
{
	F_NOTIFY_LIST_ITEM *		pNotify = pReadWriteLock->pNotifyList;
	FLMBOOL						bFoundWriter = FALSE;
	
	f_assertMutexLocked( pReadWriteLock->hMutex);
	
	while( pNotify && !bFoundWriter)
	{
		F_SEM			hSem;

		*(pNotify->pRc) = NE_FLM_OK;
		hSem = pNotify->hSem;
		bFoundWriter = (FLMBOOL)((FLMINT)pNotify->pvData);
		pNotify = pNotify->pNext;
		f_semSignal( hSem);
	}
	
	pReadWriteLock->pNotifyList = pNotify;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI f_rwlockCreate(
	F_RWLOCK *			phReadWriteLock)
{
	RCODE					rc = NE_FLM_OK;
	F_RWLOCK_IMP *		pReadWriteLock = NULL;
	
	if( RC_BAD( rc = f_calloc( sizeof( F_RWLOCK_IMP), &pReadWriteLock)))
	{
		goto Exit;
	}
	
	pReadWriteLock->hMutex = F_MUTEX_NULL;
	
	if( RC_BAD( rc = f_mutexCreate( &pReadWriteLock->hMutex)))
	{
		goto Exit;
	}
	
	*phReadWriteLock = (F_RWLOCK)pReadWriteLock;
	pReadWriteLock = NULL;
	
Exit:

	if( pReadWriteLock)
	{
		f_rwlockDestroy( (F_RWLOCK *)&pReadWriteLock);
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI f_rwlockDestroy(
	F_RWLOCK *			phReadWriteLock)
{
	F_RWLOCK_IMP *		pReadWriteLock = (F_RWLOCK_IMP *)*phReadWriteLock;
	
	if( pReadWriteLock)
	{
		f_assert( !pReadWriteLock->pNotifyList);
		
		if( pReadWriteLock->hMutex != F_MUTEX_NULL)
		{
			f_mutexDestroy( &pReadWriteLock->hMutex);
		}
		
		f_free( &pReadWriteLock);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI f_rwlockAcquire(
	F_RWLOCK				hReadWriteLock,
	F_SEM					hSem,
	FLMBOOL				bWriter)
{
	RCODE					rc = NE_FLM_OK;
	F_RWLOCK_IMP *		pReadWriteLock = (F_RWLOCK_IMP *)hReadWriteLock;
	FLMBOOL				bMutexLocked = FALSE;
	
	f_mutexLock( pReadWriteLock->hMutex);
	bMutexLocked = TRUE;
	
	if( bWriter)
	{
		if( pReadWriteLock->iRefCnt != 0)
		{
			rc = f_notifyWait( pReadWriteLock->hMutex, hSem, (void *)((FLMINT)bWriter),
				&pReadWriteLock->pNotifyList); 
		}
		
		if( RC_OK( rc))
		{
			f_assert( !pReadWriteLock->iRefCnt);
			pReadWriteLock->iRefCnt = -1;
			pReadWriteLock->uiWriteThread = f_threadId();
		}
	}
	else
	{	 
		if( pReadWriteLock->iRefCnt < 0 || pReadWriteLock->pNotifyList)
		{
			rc = f_notifyWait( pReadWriteLock->hMutex, hSem, (void *)((FLMINT)bWriter), 
				&pReadWriteLock->pNotifyList); 
		}
		
		if( RC_OK( rc))
		{
			pReadWriteLock->iRefCnt++;
		}
	}
	
	f_assert( RC_BAD( rc) || pReadWriteLock->iRefCnt);
	
	if( bMutexLocked)
	{
		f_mutexUnlock( pReadWriteLock->hMutex);
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI f_rwlockPromote(
	F_RWLOCK				hReadWriteLock,
	F_SEM					hSem)
{
	RCODE					rc = NE_FLM_OK;
	F_RWLOCK_IMP *		pReadWriteLock = (F_RWLOCK_IMP *)hReadWriteLock;
	FLMBOOL				bMutexLocked = FALSE;
	
	f_mutexLock( pReadWriteLock->hMutex);
	bMutexLocked = TRUE;
	
	if( pReadWriteLock->iRefCnt <= 0)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_ILLEGAL_OP);
		goto Exit;
	}
	
	pReadWriteLock->iRefCnt--;
		
	if( pReadWriteLock->iRefCnt != 0)
	{
		rc = f_notifyWait( pReadWriteLock->hMutex, hSem, (void *)TRUE, 
			&pReadWriteLock->pNotifyList); 
	}
	
	if( RC_OK( rc))
	{
		f_assert( !pReadWriteLock->iRefCnt);
		pReadWriteLock->iRefCnt = -1;
		pReadWriteLock->uiWriteThread = f_threadId();
	}

Exit:

	f_assert( RC_BAD( rc) || pReadWriteLock->iRefCnt);
	
	if( bMutexLocked)
	{
		f_mutexUnlock( pReadWriteLock->hMutex);
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI f_rwlockTryAcquire(
	F_RWLOCK				hReadWriteLock,
	FLMBOOL				bWriter)
{
	RCODE					rc = NE_FLM_OK;
	F_RWLOCK_IMP *		pReadWriteLock = (F_RWLOCK_IMP *)hReadWriteLock;
	
	f_mutexLock( pReadWriteLock->hMutex);
	
	if( bWriter)
	{
		if( pReadWriteLock->iRefCnt != 0)
		{
			rc = RC_SET( NE_FLM_WAIT_TIMEOUT);
		}
		else
		{
			pReadWriteLock->iRefCnt = -1;
			pReadWriteLock->uiWriteThread = f_threadId();
		}
	}
	else
	{
		if( pReadWriteLock->iRefCnt < 0 || pReadWriteLock->pNotifyList)
		{
			rc = RC_SET( NE_FLM_WAIT_TIMEOUT);
		}
		else
		{
			pReadWriteLock->iRefCnt++;
		}
	}
	
	f_assert( RC_BAD( rc) || pReadWriteLock->iRefCnt);
	f_mutexUnlock( pReadWriteLock->hMutex);
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI f_rwlockRelease(
	F_RWLOCK				hReadWriteLock)
{
	RCODE					rc = NE_FLM_OK;
	F_RWLOCK_IMP *		pReadWriteLock = (F_RWLOCK_IMP *)hReadWriteLock;
	FLMBOOL				bMutexLocked = FALSE;
	
	f_mutexLock( pReadWriteLock->hMutex);
	bMutexLocked = TRUE;
	
	if( pReadWriteLock->iRefCnt > 0)
	{
		pReadWriteLock->iRefCnt--;
	}
	else if( pReadWriteLock->iRefCnt == -1)
	{
		f_assert( pReadWriteLock->uiWriteThread == f_threadId());
		pReadWriteLock->iRefCnt = 0;
	}
	else
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_ILLEGAL_OP);
		goto Exit;
	}
	
	if( !pReadWriteLock->iRefCnt && pReadWriteLock->pNotifyList)
	{
		f_rwlockNotify( pReadWriteLock);
	}
	
Exit:
	
	f_assert( RC_BAD( rc) || pReadWriteLock->iRefCnt >= 0);
	
	if( bMutexLocked)
	{
		f_mutexUnlock( pReadWriteLock->hMutex);
	}

	return( rc);
}

/****************************************************************************
Desc: This routine links a request into a notification list and
		then waits to be notified that the event has occurred.  The mutex
		is assumed to protect the notify list.
****************************************************************************/
RCODE FTKAPI f_notifyWait(
	F_MUTEX						hMutex,
	F_SEM							hSem,
	void *						pvData,
	F_NOTIFY_LIST_ITEM **	ppNotifyList)
{
	RCODE							rc = NE_FLM_OK;
	RCODE							tmpRc;
	F_NOTIFY_LIST_ITEM		stackNotify;
	F_NOTIFY_LIST_ITEM *		pNotify = &stackNotify;
	
	f_assertMutexLocked( hMutex);
	f_assert( pNotify != *ppNotifyList);

	f_memset( &stackNotify, 0, sizeof( F_NOTIFY_LIST_ITEM));
	
	pNotify->uiThreadId = f_threadId();
	pNotify->hSem = F_SEM_NULL;
	
	if( hSem == F_SEM_NULL)
	{
		if( RC_BAD( rc = f_semCreate( &pNotify->hSem)))
		{
			goto Exit;
		}
	}
	else
	{
		pNotify->hSem = hSem;
	}
	
	pNotify->pRc = &rc;
	pNotify->pvData = pvData;
	
	pNotify->pNext = *ppNotifyList;
	*ppNotifyList = pNotify;

	// Unlock the mutex and wait on the semaphore

	f_mutexUnlock( hMutex);

	if( RC_BAD( tmpRc = f_semWait( pNotify->hSem, F_WAITFOREVER)))
	{
		rc = tmpRc;
	}

	// Free the semaphore
	
	if( hSem != pNotify->hSem)
	{
		f_semDestroy( &pNotify->hSem);
	}

	// Relock the mutex

	f_mutexLock( hMutex);

Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine notifies threads waiting for a pending read or write
		to complete.  This routine assumes that the notify list mutex
		is already locked.
****************************************************************************/
void FTKAPI f_notifySignal(
	F_NOTIFY_LIST_ITEM *	pNotifyList,
	RCODE						notifyRc)
{
	while( pNotifyList)
	{
		F_SEM			hSem;

		*(pNotifyList->pRc) = notifyRc;
		hSem = pNotifyList->hSem;
		pNotifyList = pNotifyList->pNext;
		
		f_semSignal( hSem);
	}
}
