//------------------------------------------------------------------------------
// Desc:	This file contains the F_IOBuffer and F_IOBufferMgr classes.
// Tabs:	3
//
// Copyright (c) 2001-2007 Novell, Inc. All Rights Reserved.
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
class F_IOBufferMgr : public IF_IOBufferMgr
{
public:

	F_IOBufferMgr();

	virtual ~F_IOBufferMgr();
	
	RCODE setupBufferMgr(
		FLMUINT				uiMaxBuffers,
		FLMUINT				uiMaxBytes,
		FLMBOOL				bReuseBuffers);

	RCODE FTKAPI getBuffer(
		FLMUINT				uiBufferSize,
		IF_IOBuffer **		ppIOBuffer);

	RCODE FTKAPI waitForAllPendingIO( void);

	FINLINE FLMBOOL FTKAPI isIOPending( void)
	{
		return( m_pFirstPending ? TRUE : FALSE);
	}

	void linkToList(
		F_IOBuffer **		ppListHead,
		F_IOBuffer *		pIOBuffer);

	void unlinkFromList(
		F_IOBuffer *		pIOBuffer);
		
private:

	F_MUTEX					m_hMutex;
#if !defined( FLM_UNIX) && !defined( FLM_NLM)
	F_SEM						m_hAvailSem;
#endif
	FLMUINT					m_uiMaxBuffers;
	FLMUINT					m_uiMaxBufferBytes;
	FLMUINT					m_uiTotalBuffers;
	FLMUINT					m_uiTotalBufferBytes;
	F_IOBuffer *			m_pFirstPending;
	F_IOBuffer *			m_pFirstAvail;
	F_IOBuffer *			m_pFirstUsed;
	FLMBOOL					m_bReuseBuffers;
	F_NOTIFY_LIST_ITEM *	m_pAvailNotify;
	RCODE						m_completionRc;

	friend class F_IOBuffer;
};

/****************************************************************************
Desc:	
****************************************************************************/
RCODE FTKAPI FlmAllocIOBufferMgr(
	FLMUINT					uiMaxBuffers,
	FLMUINT					uiMaxBytes,
	FLMBOOL					bReuseBuffers,
	IF_IOBufferMgr **		ppIOBufferMgr)
{
	RCODE						rc = NE_FLM_OK;
	F_IOBufferMgr *		pBufferMgr = NULL;
	
	if( (pBufferMgr = f_new F_IOBufferMgr) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( pBufferMgr->setupBufferMgr( uiMaxBuffers, 
		uiMaxBytes, bReuseBuffers)))
	{
		goto Exit;
	}
	
	*ppIOBufferMgr = pBufferMgr;
	pBufferMgr = NULL;
	
Exit:
	
	if( pBufferMgr)
	{
		pBufferMgr->Release();
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
F_IOBufferMgr::F_IOBufferMgr()
{
	m_hMutex = F_MUTEX_NULL;
#if !defined( FLM_UNIX) && !defined( FLM_NLM)
	m_hAvailSem = F_SEM_NULL;
#endif
	
	m_uiMaxBuffers = 0;
	m_uiMaxBufferBytes = 0;
	
	m_uiTotalBuffers = 0;
	m_uiTotalBufferBytes = 0;
	
	m_pFirstPending = NULL;
	m_pFirstAvail = NULL;
	m_pFirstUsed = NULL;
	
	m_pAvailNotify = NULL;
	m_bReuseBuffers = FALSE;
	m_completionRc = NE_FLM_OK;
}

/****************************************************************************
Desc:
****************************************************************************/
F_IOBufferMgr::~F_IOBufferMgr()
{
	f_assert( !m_pFirstPending);
	f_assert( !m_pFirstUsed);
	f_assert( !m_pAvailNotify);
	
	while( m_pFirstAvail)
	{
		m_pFirstAvail->Release();
	}
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}
	
#if !defined( FLM_UNIX) && !defined( FLM_NLM)
	if( m_hAvailSem != F_SEM_NULL)
	{
		f_semDestroy( &m_hAvailSem);
	}
#endif
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_IOBufferMgr::setupBufferMgr(
	FLMUINT			uiMaxBuffers,
	FLMUINT			uiMaxBytes,
	FLMBOOL			bReuseBuffers)
{
	RCODE				rc = NE_FLM_OK;
	
	f_assert( uiMaxBuffers);
	f_assert( uiMaxBytes);
	
	if( RC_BAD( rc = f_mutexCreate( &m_hMutex)))
	{
		goto Exit;
	}
	
#if !defined( FLM_UNIX) && !defined( FLM_NLM)
	if( RC_BAD( rc = f_semCreate( &m_hAvailSem)))
	{
		goto Exit;
	}
#endif
	
	m_uiMaxBuffers = uiMaxBuffers;
	m_uiMaxBufferBytes = uiMaxBytes;
	m_bReuseBuffers = bReuseBuffers;
	
Exit:

	return( rc);
}
		
/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_IOBufferMgr::getBuffer(
	FLMUINT				uiBufferSize,
	IF_IOBuffer **		ppIOBuffer)
{
	RCODE					rc = NE_FLM_OK;
	F_IOBuffer *		pIOBuffer = NULL;
	FLMBOOL				bMutexLocked = FALSE;
	
	f_assert( *ppIOBuffer == NULL);
	
	if( RC_BAD( m_completionRc))
	{
		rc = m_completionRc;
		goto Exit;
	}
	
	f_mutexLock( m_hMutex);
	bMutexLocked = TRUE;
	
Retry:

	if( m_pFirstAvail)
	{
		pIOBuffer = m_pFirstAvail;
		unlinkFromList( pIOBuffer);
		pIOBuffer->resetBuffer();
		f_assert( pIOBuffer->getBufferSize() == uiBufferSize);
	}
	else if( !m_uiTotalBuffers ||
		(m_uiTotalBufferBytes + uiBufferSize <= m_uiMaxBufferBytes &&
		m_uiTotalBuffers < m_uiMaxBuffers))
	{
		if( m_uiTotalBufferBytes + uiBufferSize > m_uiMaxBufferBytes)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_MEM);
			goto Exit;
		}

		if( (pIOBuffer = f_new F_IOBuffer) == NULL)
		{
			rc = RC_SET( NE_FLM_MEM);
			goto Exit;
		}
		
		if (RC_BAD( rc = pIOBuffer->setupBuffer( uiBufferSize, this)))
		{
			goto Exit;
		}
		
		m_uiTotalBufferBytes += uiBufferSize;
		m_uiTotalBuffers++;
	}
	else if( m_pFirstPending)
	{
	#if !defined( FLM_UNIX) && !defined( FLM_NLM)
		if( RC_BAD( rc = f_notifyWait( m_hMutex, m_hAvailSem, 
			NULL, &m_pAvailNotify)))
		{
			goto Exit;
		}
	#else
		F_IOBuffer *		pPending = m_pFirstPending;
		
		pPending->AddRef();
		f_mutexUnlock( m_hMutex);
		bMutexLocked = FALSE;
	
		rc = pPending->waitToComplete();
		
		f_mutexLock( m_hMutex);
		bMutexLocked = TRUE;
		
		pPending->Release( bMutexLocked);
		
		if( RC_BAD( rc))
		{
			goto Exit;
		}
	#endif
		
		goto Retry;
	}
	else
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_MEM);
		goto Exit;
	}
	
	pIOBuffer->AddRef();
	linkToList( &m_pFirstUsed, pIOBuffer);
	*ppIOBuffer = pIOBuffer;
	pIOBuffer = NULL;
	
Exit:

	if( pIOBuffer)
	{
		pIOBuffer->Release();
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
RCODE FTKAPI F_IOBufferMgr::waitForAllPendingIO( void)
{
	RCODE				rc = NE_FLM_OK;
	RCODE				tmpRc;
	F_IOBuffer *	pBuf;
	FLMBOOL			bMutexLocked = FALSE;

	f_mutexLock( m_hMutex);
	bMutexLocked = TRUE;
	
	while( (pBuf = m_pFirstPending) != NULL)
	{
		pBuf->AddRef();
		
		f_mutexUnlock( m_hMutex);
		bMutexLocked = FALSE;
		
		if( RC_BAD( tmpRc = pBuf->waitToComplete()))
		{
			if( RC_OK( m_completionRc))
			{
				f_mutexLock( m_hMutex);
				bMutexLocked = TRUE;
				m_completionRc = tmpRc;
			}
		}
		
		if( !bMutexLocked)
		{
			f_mutexLock( m_hMutex);
			bMutexLocked = TRUE;
		}

		pBuf->Release( TRUE);
		pBuf = NULL;
	}
	
	rc = m_completionRc;
	m_completionRc = NE_FLM_OK;
	
	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_IOBufferMgr::linkToList(
	F_IOBuffer **		ppListHead,
	F_IOBuffer *		pIOBuffer)
{
	f_assertMutexLocked( m_hMutex);
	f_assert( pIOBuffer->m_eList == MGR_LIST_NONE);
	
	pIOBuffer->m_pPrev = NULL;
	
	if( (pIOBuffer->m_pNext = *ppListHead) != NULL)
	{
		(*ppListHead)->m_pPrev = pIOBuffer;
	}
	
	*ppListHead = pIOBuffer;
	
	if( ppListHead == &m_pFirstPending)
	{
		f_assert( !pIOBuffer->m_bPending);
		pIOBuffer->m_eList = MGR_LIST_PENDING;
	}
	else if( ppListHead == &m_pFirstUsed)
	{
		pIOBuffer->m_eList = MGR_LIST_USED;
	}
	else
	{
		pIOBuffer->m_eList = MGR_LIST_AVAIL;
	}
}

/****************************************************************************
Desc:
****************************************************************************/
void F_IOBufferMgr::unlinkFromList(
	F_IOBuffer *	pIOBuffer)
{
	f_assertMutexLocked( m_hMutex);
	
	if( pIOBuffer->m_pNext)
	{
		pIOBuffer->m_pNext->m_pPrev = pIOBuffer->m_pPrev;
	}
	
	if( pIOBuffer->m_pPrev)
	{
		pIOBuffer->m_pPrev->m_pNext = pIOBuffer->m_pNext;
	}
	else if( pIOBuffer->m_eList == MGR_LIST_AVAIL)
	{
		m_pFirstAvail = pIOBuffer->m_pNext;
	}
	else if( pIOBuffer->m_eList == MGR_LIST_PENDING)
	{
		m_pFirstPending = pIOBuffer->m_pNext;
	}
	else if( pIOBuffer->m_eList == MGR_LIST_USED)
	{
		m_pFirstUsed = pIOBuffer->m_pNext;
	}
	
	pIOBuffer->m_eList = MGR_LIST_NONE;
}

/****************************************************************************
Desc:
****************************************************************************/
FLMINT F_IOBuffer::Release(
	FLMBOOL				bMutexAlreadyLocked)
{
	FLMINT				iRefCnt;
	F_MUTEX				hMutex = F_MUTEX_NULL;
	F_IOBufferMgr *	pBufferMgr = NULL;

	if( m_pBufferMgr && !bMutexAlreadyLocked)
	{
		hMutex = m_pBufferMgr->m_hMutex;
		f_assertMutexNotLocked( hMutex);
		f_mutexLock( hMutex);
	}
	
	if( m_refCnt <= 2)
	{
		if( m_pBufferMgr && m_eList != MGR_LIST_NONE)
		{
			f_assert( m_eList != MGR_LIST_PENDING);
			m_pBufferMgr->unlinkFromList( this);
		}
	}
	
	if( m_refCnt == 2)
	{
		if( m_pAsyncClient)
		{
			m_pAsyncClient->Release();
			m_pAsyncClient = NULL;
		}

		if( (pBufferMgr = m_pBufferMgr) != NULL)
		{
			if( m_pBufferMgr->m_bReuseBuffers)
			{
				m_pBufferMgr->linkToList( &m_pBufferMgr->m_pFirstAvail, this);
			}
			else
			{
				f_assert( m_pBufferMgr->m_uiTotalBuffers);
				f_assert( m_pBufferMgr->m_uiTotalBufferBytes >= m_uiBufferSize);

				f_atomicDec( &m_refCnt);
				m_pBufferMgr->m_uiTotalBuffers--;
				m_pBufferMgr->m_uiTotalBufferBytes -= m_uiBufferSize;
				m_pBufferMgr = NULL;
			}
			
			if( pBufferMgr->m_pAvailNotify)
			{
				f_notifySignal( pBufferMgr->m_pAvailNotify, NE_FLM_OK);
				pBufferMgr->m_pAvailNotify = NULL;
			}
		}
	}
	
	iRefCnt = f_atomicDec( &m_refCnt);

	if( hMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( hMutex);
	}
	
	if( !iRefCnt)
	{
		delete this;
	}
	
	return( iRefCnt);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_IOBuffer::setupBuffer(
	FLMUINT				uiBufferSize,
	F_IOBufferMgr *	pBufferMgr)
{
	RCODE					rc = NE_FLM_OK;

	if( RC_BAD( rc = f_allocAlignedBuffer( uiBufferSize, 
		(void **)&m_pucBuffer)))
	{
		goto Exit;
	}
	
	m_uiBufferSize = uiBufferSize;
	m_pBufferMgr = pBufferMgr;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI F_IOBuffer::setPending( void)
{
	f_assert( !m_bPending);
	
	if( m_pBufferMgr)
	{
		f_assert( m_eList == MGR_LIST_USED);
		
		f_mutexLock( m_pBufferMgr->m_hMutex);
		m_pBufferMgr->unlinkFromList( this);
		m_pBufferMgr->linkToList( &m_pBufferMgr->m_pFirstPending, this);
		f_mutexUnlock( m_pBufferMgr->m_hMutex);
	}

#ifndef FLM_UNIX
	f_assert( !m_pAsyncClient || 
				 f_semGetSignalCount( ((F_FileAsyncClient *)m_pAsyncClient)->m_hSem) == 0);
#endif

	m_bPending = TRUE;
	m_uiStartTime = FLM_GET_TIMER();
	m_uiEndTime = 0;
}
		
/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI F_IOBuffer::clearPending( void)
{
	f_assert( m_bPending);
	
	if( m_pBufferMgr)
	{
		f_assert( m_eList == MGR_LIST_PENDING);
		
		f_mutexLock( m_pBufferMgr->m_hMutex);
		m_pBufferMgr->unlinkFromList( this);
		m_pBufferMgr->linkToList( &m_pBufferMgr->m_pFirstUsed, this);
		f_mutexUnlock( m_pBufferMgr->m_hMutex);
	}

	m_bPending = FALSE;
	m_uiStartTime = 0;
}

/****************************************************************************
Desc:
****************************************************************************/
void F_IOBuffer::notifyComplete(
	RCODE					completionRc)
{
	f_assert( m_bPending);
	
	m_bPending = FALSE;
	m_bCompleted = TRUE;
	m_completionRc = completionRc;
	m_uiEndTime = FLM_GET_TIMER();
	m_uiElapsedTime = FLM_TIMER_UNITS_TO_MILLI( 
		FLM_ELAPSED_TIME( m_uiEndTime, m_uiStartTime));

	if( m_fnCompletion)
	{
		m_fnCompletion( this, m_pvData);
		m_fnCompletion = NULL;
		m_pvData = NULL;
	}

	if( m_pBufferMgr)
	{
		f_assert( m_eList == MGR_LIST_PENDING);
		f_mutexLock( m_pBufferMgr->m_hMutex);
		
		m_pBufferMgr->unlinkFromList( this);
		m_pBufferMgr->linkToList( &m_pBufferMgr->m_pFirstUsed, this);
		
		if( RC_OK( m_pBufferMgr->m_completionRc) && RC_BAD( completionRc))
		{
			m_pBufferMgr->m_completionRc = completionRc;
		}
		
		f_mutexUnlock( m_pBufferMgr->m_hMutex);		
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_IOBuffer::addCallbackData(
	void *							pvData)
{
	RCODE			rc = NE_FLM_OK;
	
	if( m_uiCallbackDataCount >= m_uiMaxCallbackData)
	{
		if( m_ppCallbackData == m_callbackData)
		{
			void **	pNewTable;
			
			if( RC_BAD( rc = f_alloc( 
				(m_uiCallbackDataCount + 1) * sizeof( void *), &pNewTable)))
			{
				goto Exit;
			}
			
			f_memcpy( pNewTable, m_ppCallbackData, 
				m_uiMaxCallbackData * sizeof( void *));
			m_ppCallbackData = pNewTable;
		}
		else
		{
			if( RC_BAD( rc = f_realloc( 
				(m_uiCallbackDataCount + 1) * sizeof( void *), &m_ppCallbackData)))
			{
				goto Exit;
			}
		}
		
		m_uiMaxCallbackData = m_uiCallbackDataCount + 1;
	}
	
	m_ppCallbackData[ m_uiCallbackDataCount] = pvData;
	m_uiCallbackDataCount++;
	
Exit:

	return( rc);
}
			
/****************************************************************************
Desc:
****************************************************************************/
void * FTKAPI F_IOBuffer::getCallbackData(
	FLMUINT							uiSlot)
{
	if( uiSlot < m_uiCallbackDataCount)
	{
		return( m_ppCallbackData[ uiSlot]);
	}
	
	return( NULL);
}
