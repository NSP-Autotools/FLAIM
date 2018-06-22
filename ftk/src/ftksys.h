//------------------------------------------------------------------------------
// Desc:	Cross-platform macros, defines, etc.  Must visit this file
//			to port XFLAIM to another platform.
// Tabs:	3
//
// Copyright (c) 1991-2007 Novell, Inc. All Rights Reserved.
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

#ifndef FTKSYS_H
#define FTKSYS_H

    #include "config.h"

	#include "flaimtk.h"

	#ifdef FLM_NLM
		#if !defined( FLM_RING_ZERO_NLM) && !defined( FLM_LIBC_NLM)
			#define FLM_LIBC_NLM
		#endif
	
		#if defined( FLM_RING_ZERO_NLM) && defined( FLM_LIBC_NLM)
			#error Cannot target both LIBC and RING 0
		#endif
	#endif

	class F_FileHdl;
	class F_Thread;
	class F_ThreadMgr;
	class F_FileSystem;
	class F_ThreadMgr;
	class F_ResultSet;
	class F_ResultSetBlk;
	class F_IOBufferMgr;

	/****************************************************************************
	Desc: Global data
	****************************************************************************/

	#define FLM_DEFAULT_OPEN_THRESHOLD						100
	#define FLM_DEFAULT_MAX_AVAIL_TIME						900
	#define FLM_MAX_KEY_SIZE									1024
	#define FLM_NLM_SECTOR_SIZE								512

	/****************************************************************************
	Desc:		NLM
	****************************************************************************/
	#if defined( FLM_NLM)
		#include "ftknlm.h"
	#endif

	/****************************************************************************
	Desc:	WIN
	****************************************************************************/
	#if defined( FLM_WIN)

		#ifndef WIN32_LEAN_AND_MEAN
			#define WIN32_LEAN_AND_MEAN
		#endif
	
		#ifndef WIN32_EXTRA_LEAN
			#define WIN32_EXTRA_LEAN
		#endif
	
		// Enable critical section and spin count API to be visible in header
		// file.
	
		#define _WIN32_WINNT	0x0403
	
		#pragma pack( push, enter_windows, 8)
			#include <windows.h>
			#include <time.h>
			#include <stdlib.h>
			#include <stddef.h>
			#include <rpc.h>
			#include <process.h>
			#include <winsock.h>
			#include <imagehlp.h>
			#include <malloc.h>
			#include <stdio.h>
			#include <direct.h>
		#pragma pack( pop, enter_windows)
		
		// Conversion from XXX to YYY, possible loss of data
		#pragma warning( disable : 4244) 
	
		// Local variable XXX may be used without having been initialized
		#pragma warning( disable : 4701)
	
		// Function XXX not inlined
		#pragma warning( disable : 4710) 

		// Flow in or out of inline asm code suppresses global optimization
		#pragma warning( disable : 4740) 
		
		#define ENDLINE			ENDLINE_CRLF
		
	#endif

	/****************************************************************************
	Desc:		UNIX
	****************************************************************************/
	#if defined( FLM_UNIX)

		#ifdef FLM_OSX
			#include <sys/resource.h>
			#include <sys/param.h>
			#include <sys/mount.h>
			#include <libkern/OSAtomic.h>
		#endif

		#ifdef FLM_SOLARIS
			#include <signal.h>
			#include <synch.h>
		#endif

		#ifdef FLM_AIX
			#ifndef _LARGE_FILES
				#define _LARGE_FILES
			#endif
			#include <dlfcn.h>
			#include <sys/atomic_op.h>
			#include <sys/vminfo.h>
			#include <sys/statfs.h>
		#endif

		#ifdef FLM_HPUX
			#include <sys/pstat.h>
			#include <sys/param.h>
			#include <sys/unistd.h>
			#include <sys/fs/vx_ioctl.h>
		#endif

		#include <stdio.h>
		#include <fcntl.h>
		#include <assert.h>
		#include <pthread.h>
		#include <errno.h>
		#include <glob.h>
		#include <limits.h>
		#include <netdb.h>
		#include <sys/types.h>
		#include <netinet/in.h>
		#include <arpa/nameser.h>
		#include <resolv.h>
		#include <stdarg.h>
		#include <stdlib.h>
		#include <string.h>
		#include <strings.h>
		#include <time.h>
		#include <unistd.h>
		#include <utime.h>
		#include <aio.h>
		#include <sched.h>
		#include <arpa/inet.h>
		#include <netinet/tcp.h>
		#include <sys/mman.h>
		#include <sys/resource.h>
		#include <sys/socket.h>
		#include <sys/stat.h>
		#include <sys/time.h>

		typedef int						SOCKET;
		#define INVALID_SOCKET		-1
	
	#endif
	
	#ifdef FLM_OPENSSL
		#include <openssl/ssl.h>
		#include <openssl/err.h>
		#include <openssl/bio.h>
	#endif

	#if defined( __va_copy)
		#define  f_va_copy(to, from) __va_copy(to, from)
	#else
		#define f_va_copy(to, from)  ((to) = (from))
	#endif

	/**********************************************************************
	Desc:
	**********************************************************************/
	#if defined( FLM_SPARC_PLUS)
	extern "C" FLMINT32 sparc_atomic_add_32(
		volatile FLMINT32 *		piTarget,
		FLMINT32						iDelta);
	#endif
	
	/**********************************************************************
	Desc:
	**********************************************************************/
	#if defined( FLM_SPARC_PLUS)
	extern "C" FLMINT32 sparc_atomic_xchg_32(
		volatile FLMINT32 *		piTarget,
		FLMINT32						iNewValue);
	#endif
	
	/**********************************************************************
	Desc:
	**********************************************************************/
	#if defined( FLM_PPC) && defined( FLM_GNUC) && defined( FLM_LINUX)
		extern "C"  FLMATOMIC ppc_atomic_add(
			FLMATOMIC *		piTarget,
			FLMINT32			iDelta);
	#endif
	
	/**********************************************************************
	Desc:
	**********************************************************************/
	#if defined( FLM_PPC) && defined( FLM_GNUC) && defined( FLM_LINUX)
		extern "C"  FLMATOMIC ppc_atomic_xchg(
			FLMATOMIC *		piTarget,
			FLMATOMIC		iNewValue);
	#endif
	
	/**********************************************************************
	Desc:
	**********************************************************************/
	#if defined( FLM_AIX)
	FINLINE int aix_atomic_add(
		volatile int *			piTarget,
		int 						iDelta)
	{
		return( fetch_and_add( (int *)piTarget, iDelta) + iDelta);
	}
	#endif
	
	/**********************************************************************
	Desc:
	**********************************************************************/
	#if defined( FLM_UNIX)
	FLMINT32 posix_atomic_add_32(
		volatile FLMINT32 *		piTarget,
		FLMINT32						iDelta);
	#endif
	
	/**********************************************************************
	Desc:
	**********************************************************************/
	#if defined( FLM_UNIX)
	FLMINT32 posix_atomic_xchg_32(
		volatile FLMINT32 *		piTarget,
		FLMINT32						iNewValue);
	#endif
	
	/****************************************************************************
	Desc: Mutex and semaphore routines
	****************************************************************************/
	#if defined( FLM_WIN)
		typedef struct
		{
			FLMATOMIC						locked;
	#ifdef FLM_DEBUG
			FLMUINT							uiThreadId;
			FLMATOMIC						lockedCount;
			FLMATOMIC						waitCount;
	#endif
		} F_INTERLOCK;
	#endif

	/****************************************************************************
											 f_sprintf
	****************************************************************************/

	// Percent formating prefixes
	
	#define FLM_PREFIX_NONE				0
	#define FLM_PREFIX_MINUS 			1
	#define FLM_PREFIX_PLUS				2
	#define FLM_PREFIX_POUND 			3
	
	// Width and Precision flags
	
	#define FLM_PRINTF_MINUS_FLAG		0x0001
	#define FLM_PRINTF_PLUS_FLAG		0x0002
	#define FLM_PRINTF_SPACE_FLAG		0x0004
	#define FLM_PRINTF_POUND_FLAG		0x0008
	#define FLM_PRINTF_ZERO_FLAG		0x0010
	#define FLM_PRINTF_SHORT_FLAG		0x0020
	#define FLM_PRINTF_LONG_FLAG		0x0040
	#define FLM_PRINTF_DOUBLE_FLAG	0x0080
	#define FLM_PRINTF_INT64_FLAG		0x0100
	#define FLM_PRINTF_COMMA_FLAG		0x0200

	/****************************************************************************
	Desc:
	****************************************************************************/
	typedef enum
	{
		MGR_LIST_NONE,
		MGR_LIST_AVAIL,
		MGR_LIST_PENDING,
		MGR_LIST_USED
	} eBufferMgrList;
		
	#define F_DEFAULT_CBDATA_SLOTS		16
	
	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_Printf : public F_Object
	{
	public:
	
		F_Printf( IF_PrintfClient * pClient)
		{
			m_pClient = pClient;
			m_pClient->AddRef();
			m_iBytesOutput = 0;
		}
		
		virtual ~F_Printf()
		{
			if( m_pClient)
			{
				m_pClient->Release();
				m_pClient = NULL;
			}
		}
	
		void processFieldInfo(
			const char **		ppszFormat,
			FLMUINT *			puiWidth,
			FLMUINT *			puiPrecision,
			FLMUINT *			puiFlags,
			f_va_list *			args);
		
		void stringFormatter(
			char					cFormatChar,
			FLMUINT				uiWidth,
			FLMUINT				uiPrecision,
			FLMUINT				uiFlags,
			f_va_list *			args);
		
		void charFormatter(
			char					cFormatChar,
			f_va_list *			args);
		
		void errorFormatter(
			f_va_list *			args);
		
		void notHandledFormatter( void);
		
		void numberFormatter(
			char					cFormatChar,
			FLMUINT				uiWidth,
			FLMUINT				uiPrecision,
			FLMUINT				uiFlags,
			f_va_list *			args);
		
		FLMINT parseArgs(
			const char *		pszFormat,
			f_va_list *			args);
		
		void processFormatString(
			FLMUINT				uiLen,
			...);
			
		FLMUINT printNumber(
			FLMUINT64			ui64Val,
			FLMUINT				uiBase,
			FLMBOOL				bUpperCase,
			FLMBOOL				bCommas,
			char *				pszBuf);
			
	private:
	
		IF_PrintfClient *		m_pClient;
		FLMINT					m_iBytesOutput;
	};
	
	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_IOBuffer : public IF_IOBuffer
	{
	public:
	
		F_IOBuffer()
		{
			m_pucBuffer = NULL;
			m_uiBufferSize = 0;
			m_pBufferMgr = NULL;
			m_pAsyncClient = NULL;
			m_fnCompletion = NULL;
			m_pvData = NULL;
			m_ppCallbackData = m_callbackData;
			m_uiMaxCallbackData = F_DEFAULT_CBDATA_SLOTS;
			m_pPrev = NULL;
			m_pNext = NULL;
			m_eList = MGR_LIST_NONE;
			resetBuffer();
		}

		virtual ~F_IOBuffer()
		{
			if( m_pucBuffer)
			{
				cleanupBuffer();
				f_freeAlignedBuffer( (void **)&m_pucBuffer);
			}
			
			if( m_pAsyncClient)
			{
				m_pAsyncClient->Release();
			}
		}

		FLMINT FTKAPI AddRef( void)
		{
			return( f_atomicInc( &m_refCnt));
		}

		FLMINT Release(
			FLMBOOL					bMutexAlreadyLocked);
	
		FLMINT FTKAPI Release( void)
		{
			return( Release( FALSE));
		}

		RCODE setupBuffer(
			FLMUINT					uiBufferSize,
			F_IOBufferMgr *		pBufferMgr);
			
		FINLINE void resetBuffer( void)
		{
			f_assert( !m_pAsyncClient);

			cleanupBuffer();						
			m_uiElapsedTime = 0;
			m_completionRc = NE_FLM_OK;
			m_bPending = FALSE;
			m_bCompleted = FALSE;
		}
	
		FINLINE FLMBYTE * FTKAPI getBufferPtr( void)
		{
			return( m_pucBuffer);
		}
	
		FINLINE FLMUINT FTKAPI getBufferSize( void)
		{
			return( m_uiBufferSize);
		}
	
		FINLINE void FTKAPI setCompletionCallback(
			F_BUFFER_COMPLETION_FUNC	fnCompletion,
			void *							pvData)
		{
			m_fnCompletion = fnCompletion;
			m_pvData = pvData;
		}
		
		RCODE FTKAPI addCallbackData(
			void *							pvData);
			
		void * FTKAPI getCallbackData(
			FLMUINT							uiSlot);
			
		FINLINE FLMUINT FTKAPI getCallbackDataCount( void)
		{
			return( m_uiCallbackDataCount);
		}
		
		FINLINE void FTKAPI cleanupBuffer( void)
		{
			if( m_fnCompletion)
			{
				m_fnCompletion( this, m_pvData);
			}
			
			m_fnCompletion = NULL;
			m_pvData = NULL;
			
			if( m_ppCallbackData && m_ppCallbackData != m_callbackData)
			{
				f_free( &m_ppCallbackData);
			}
			
			m_uiMaxCallbackData = F_DEFAULT_CBDATA_SLOTS;
			m_uiCallbackDataCount = 0;
			m_ppCallbackData = m_callbackData;
		}
			
		void FTKAPI setAsyncClient(
			IF_AsyncClient *				pAsyncClient)
		{
			if( m_pAsyncClient)
			{
				m_pAsyncClient->Release();
			}
			
			if( (m_pAsyncClient = pAsyncClient) != NULL)
			{
				m_pAsyncClient->AddRef();
			}
		}
			
		void FTKAPI getAsyncClient(
			IF_AsyncClient **				ppAsyncClient)
		{
			if( (*ppAsyncClient = m_pAsyncClient) != NULL)
			{
				(*ppAsyncClient)->AddRef();
			}
		}
		
		void FTKAPI setPending( void);
		
		void FTKAPI clearPending( void);
		
		void FTKAPI notifyComplete(
			RCODE							completionRc);
			
		FINLINE FLMBOOL FTKAPI isPending( void)
		{
			return( m_bPending);
		}
		
		FINLINE RCODE FTKAPI waitToComplete( void)
		{
			RCODE		rc = NE_FLM_OK;
			
			if( m_pAsyncClient)
			{
				rc = m_pAsyncClient->waitToComplete();
			}
			
			return( rc);
		}
		
		FINLINE FLMBOOL FTKAPI isComplete( void)
		{
			return( m_bCompleted);
		}
		
		FINLINE RCODE FTKAPI getCompletionCode( void)
		{
			f_assert( m_bCompleted);
			return( m_completionRc);
		}

		FINLINE FLMUINT FTKAPI getElapsedTime( void)
		{
			f_assert( m_bCompleted);
			return( m_uiElapsedTime);
		}
			
	private:
	
		FLMBYTE *						m_pucBuffer;
		FLMUINT							m_uiBufferSize;
		F_IOBufferMgr *				m_pBufferMgr;
		IF_AsyncClient *				m_pAsyncClient;
		F_BUFFER_COMPLETION_FUNC	m_fnCompletion;
		void *							m_pvData;
		FLMUINT							m_uiElapsedTime;
		RCODE								m_completionRc;
		FLMBOOL							m_bPending;
		FLMBOOL							m_bCompleted;
		FLMUINT							m_uiStartTime;
		FLMUINT							m_uiEndTime;
		void *							m_callbackData[ F_DEFAULT_CBDATA_SLOTS];
		void **							m_ppCallbackData;
		FLMUINT							m_uiCallbackDataCount;
		FLMUINT							m_uiMaxCallbackData;
		F_IOBuffer *					m_pPrev;
		F_IOBuffer *					m_pNext;
		eBufferMgrList					m_eList;
		
		friend class F_FileAsyncClient;
		friend class F_IOBufferMgr;
	};
	
	/****************************************************************************
	Desc:
	****************************************************************************/
	class FTKEXP F_FileAsyncClient : public IF_AsyncClient
	{
	public:
	
		F_FileAsyncClient()
		{
			m_pFileHdl = NULL;
			m_pIOBuffer = NULL;
			m_completionRc = NE_FLM_OK;
			m_uiBytesToDo = 0;
			m_uiBytesDone = 0;
			m_pNext = NULL;
		#ifndef FLM_UNIX
			m_hSem = F_SEM_NULL;
		#endif
		#ifdef FLM_WIN
			m_Overlapped.hEvent = 0;
		#endif
		}
		
		~F_FileAsyncClient();
		
		FLMINT FTKAPI AddRef( void)
		{
			return( f_atomicInc( &m_refCnt));
		}

		FLMINT FTKAPI Release( void)
		{
			return( Release( TRUE));
		}

		FLMINT FTKAPI Release(
			FLMBOOL						bOkToReuse);

		RCODE FTKAPI waitToComplete( void);

		RCODE FTKAPI getCompletionCode( void);
		
		FLMUINT FTKAPI getElapsedTime( void);
		
		F_FileAsyncClient *			m_pNext;
		
		FLMUINT getBytesToDo( void)
		{
			return( m_uiBytesToDo);
		}

		void FTKAPI notifyComplete(
			RCODE							completionRc,
			FLMUINT						uiBytesDone);

		RCODE prepareForAsync(
			IF_IOBuffer *				pIOBuffer);
		
		F_FileHdl *						m_pFileHdl;
		IF_IOBuffer *					m_pIOBuffer;
		RCODE								m_completionRc;
		FLMUINT							m_uiBytesToDo;
		FLMUINT							m_uiBytesDone;
		FLMUINT							m_uiStartTime;
		FLMUINT							m_uiEndTime;
	#ifndef FLM_UNIX
		F_SEM								m_hSem;
	#endif
	#if defined( FLM_WIN)
		OVERLAPPED						m_Overlapped;
	#endif
	#if defined( FLM_UNIX) && defined( FLM_HAS_ASYNC_IO)
		struct aiocb					m_aio;
	#endif
	};

	/***************************************************************************
	Desc:
	***************************************************************************/
	class F_FileHdl : public IF_FileHdl
	{
	public:
	
		F_FileHdl();
	
		virtual ~F_FileHdl();
	
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

		RCODE FTKAPI flush( void);
		
		RCODE FTKAPI read(
			FLMUINT64			ui64Offset,
			FLMUINT				uiLength,
			void *				pvBuffer,
			FLMUINT *			puiBytesRead = NULL);

		RCODE FTKAPI read(
			FLMUINT64			ui64ReadOffset,
			FLMUINT				uiBytesToRead,
			IF_IOBuffer *		pIOBuffer);
			
		RCODE FTKAPI write(
			FLMUINT64			ui64Offset,
			FLMUINT				uiLength,
			const void *		pvBuffer,
			FLMUINT *			puiBytesWritten = NULL);

		RCODE FTKAPI write(
			FLMUINT64			ui64WriteOffset,
			FLMUINT				uiBytesToWrite,
			IF_IOBuffer *		pIOBuffer);
			
		RCODE FTKAPI seek(
			FLMUINT64			ui64Offset,
			FLMINT				iWhence,
			FLMUINT64 *			pui64NewOffset = NULL);

		RCODE FTKAPI size(
			FLMUINT64 *			pui64Size);

		RCODE FTKAPI tell(
			FLMUINT64 *			pui64Offset);
			
		RCODE FTKAPI extendFile(
			FLMUINT64			ui64FileSize);

		RCODE FTKAPI truncateFile(
			FLMUINT64			ui64FileSize = 0);

		RCODE FTKAPI closeFile( void);
		
		FINLINE FLMBOOL FTKAPI canDoAsync( void)
		{
			return( m_bOpenedInAsyncMode);
		}
		
		FINLINE FLMBOOL FTKAPI canDoDirectIO( void)
		{
			return( m_bDoDirectIO);
		}
	
		FINLINE void FTKAPI setExtendSize(
			FLMUINT				uiExtendSize)
		{
			f_assert( uiExtendSize < FLM_MAX_UINT);
			m_uiExtendSize = uiExtendSize;
		}
	
		FINLINE void FTKAPI setMaxAutoExtendSize(
			FLMUINT				uiMaxAutoExtendSize)
		{
			m_uiMaxAutoExtendSize = uiMaxAutoExtendSize;
		}
	
		FINLINE FLMBOOL FTKAPI isReadOnly( void)
		{
			return( m_bOpenedReadOnly);
		}
		
		FINLINE FLMBOOL FTKAPI isOpen( void)
		{
			return( m_bFileOpened);
		}
		
		FINLINE FLMUINT FTKAPI getSectorSize( void)
		{
			return( m_uiBytesPerSector);
		}
		
		RCODE FTKAPI lock( void);
	
		RCODE FTKAPI unlock( void);
		
		static F_MUTEX 				m_hAsyncListMutex;
		static F_FileAsyncClient *	m_pFirstAvailAsync;
		static FLMUINT					m_uiAvailAsyncCount;

	private:
	
		FINLINE FLMUINT64 roundToNextSector(
			FLMUINT64				ui64Bytes)
		{
			f_assert( m_ui64GetSectorBoundMask);
			f_assert( m_ui64NotOnSectorBoundMask);
			
			return( (ui64Bytes + m_ui64NotOnSectorBoundMask) & 
							m_ui64GetSectorBoundMask);
		}
	
		FINLINE FLMUINT64 truncateToPrevSector(
			FLMUINT64				ui64Offset)
		{
			return( ui64Offset & m_ui64GetSectorBoundMask);
		}
	
		FINLINE const char * getFileName( void)
		{
			return( m_pszFileName);
		}
		
		RCODE allocFileAsyncClient(
			F_FileAsyncClient **		ppAsyncClient);
			
		void initCommonData( void);
	
		void freeCommonData( void);
		
		RCODE createFile(
			const char *			pszFileName,
			FLMUINT					uiIoFlags);
	
		RCODE createUniqueFile(
			char *					pszDirName,
			const char *			pszFileExtension,
			FLMUINT					uiIoFlags);
	
		RCODE openFile(
			const char *			pszFileName,
			FLMUINT					uiIoFlags);
	
		RCODE openOrCreate(
			const char *			pszFileName,
			FLMUINT					uiAccess,
			FLMBOOL					bCreateFlag);
	
		RCODE lowLevelRead(
			FLMUINT64				ui64Offset,
			FLMUINT					uiLength,
			void *					pvBuffer,
			IF_IOBuffer *			pIOBuffer,
			FLMUINT *				puiBytesRead);
	
		RCODE lowLevelWrite(
			FLMUINT64				ui64WriteOffset,
			FLMUINT					uiBytesToWrite,
			const void *			pvBuffer,
			IF_IOBuffer *			pIOBuffer,
			FLMUINT *				puiBytesWritten);
			
		RCODE directRead(
			FLMUINT64				ui64ReadOffset,
			FLMUINT					uiBytesToRead,
			void *					pvBuffer,
			IF_IOBuffer *			pIOBuffer,
			FLMUINT *				puiBytesRead);
			
		RCODE directWrite(
			FLMUINT64				ui64WriteOffset,
			FLMUINT					uiBytesToWrite,
			const void *			pvBuffer,
			IF_IOBuffer *			pIOBuffer,
			FLMUINT *				puiBytesWritten);
			
		RCODE getPreWriteExtendSize(
			FLMUINT64				ui64WriteOffset,
			FLMUINT					uiBytesToWrite,
			FLMUINT64 *				pui64CurrFileSize,
			FLMUINT *				puiTotalBytesToExtend);
	
	#if defined( FLM_RING_ZERO_NLM)
	
		RCODE setup( void);							
	
		RCODE expand(
			LONG						lStartSector,
			LONG						lSectorsToAlloc);
	
		RCODE internalBlockingRead(
			FLMUINT64				ui64ReadOffset,
			FLMUINT					uiBytesToRead,	
			void *					pvBuffer,
			FLMUINT *				puiBytesRead);

		RCODE internalBlockingWrite(
			FLMUINT64				ui64WriteOffset,
			FLMUINT					uiBytesToWrite,	
			const void *			pvBuffer,
			FLMUINT *				puiBytesWritten);

		RCODE writeSectors(
			const void *			pvBuffer,
			F_FileAsyncClient *	pAsyncClient,
			LONG						lStartSector,
			LONG						lSectorCount);
			
	#endif
			
		char *						m_pszFileName;
		FLMUINT						m_uiBytesPerSector;
		FLMUINT64					m_ui64NotOnSectorBoundMask;
		FLMUINT64					m_ui64GetSectorBoundMask;
		FLMUINT						m_uiExtendSize;
		FLMUINT						m_uiMaxAutoExtendSize;
		FLMBYTE *					m_pucAlignedBuff;
		FLMUINT						m_uiAlignedBuffSize;
		FLMUINT64					m_ui64CurrentPos;
		FLMBOOL						m_bFileOpened;
		FLMBOOL						m_bDeleteOnRelease;
		FLMBOOL						m_bOpenedReadOnly;
		FLMBOOL						m_bOpenedExclusive;
		FLMBOOL						m_bDoDirectIO;
		FLMBOOL						m_bOpenedInAsyncMode;
		FLMBOOL						m_bRequireAlignedIO;
		FLMATOMIC					m_numAsyncPending;
		
	#if defined( FLM_WIN)
		HANDLE						m_hFile;
		FLMBOOL						m_bFlushRequired;
	#elif defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
		int				   		m_fd;
		FLMBOOL						m_bFlushRequired;
	#elif defined( FLM_RING_ZERO_NLM)		
		LONG							m_lFileHandle;
		LONG							m_lOpenAttr;
		LONG							m_lVolumeID;
		LONG							m_lLNamePathCount;
		FLMBOOL						m_bDoSuballocation;
		LONG							m_lSectorsPerBlock;
		LONG							m_lMaxBlocks;
		FLMBOOL						m_bNSS;
		FLMINT64						m_NssKey;
		FLMBOOL						m_bNSSFileOpen;
	#endif

		friend class F_FileSystem;
		friend class F_MultiFileHdl;
		friend class F_FileHdlCache;
		friend class F_FileAsyncClient;
	};
	
	/****************************************************************************
	Desc:
	****************************************************************************/
	
	#if defined( FLM_WIN)
	
		typedef struct
		{
			 HANDLE					findHandle;
			 WIN32_FIND_DATA		findBuffer;
			 char 	   			szSearchPath[ F_PATH_MAX_SIZE];
			 FLMUINT					uiSearchAttrib;
		} F_IO_FIND_DATA;
	
	#elif defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
	
		typedef struct
		{
			mode_t		mode_flag;
			struct stat	FileStat;
			char 			name[ F_PATH_MAX_SIZE+1];
			char			search_path[ F_PATH_MAX_SIZE+1];
			char			full_path[ F_PATH_MAX_SIZE];
			char			pattern_str[ F_PATH_MAX_SIZE];
			char			dirpath[ F_PATH_MAX_SIZE];
			glob_t      globbuf;
		} F_IO_FIND_DATA;
		
	#elif defined( FLM_RING_ZERO_NLM)
	
		typedef struct
		{
			LONG									lVolumeNumber;
			LONG									lDirectoryNumber;
			LONG									lCurrentEntryNumber;
			struct DirectoryStructure *	pCurrentItem;
			char									ucTempBuffer[ F_FILENAME_SIZE];
		} F_IO_FIND_DATA;
		
	#else
	
		#error Platform not supported
	
	#endif

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_DirHdl : public IF_DirHdl
	{
	public:
	
		F_DirHdl();
	
		virtual ~F_DirHdl();
	
		RCODE FTKAPI next( void);
	
		const char * FTKAPI currentItemName( void);
	
		void FTKAPI currentItemPath(
			char *	pszPath);
	
			FLMUINT64 FTKAPI currentItemSize( void);
	
		FLMBOOL FTKAPI currentItemIsDir( void);
	
	private:
	
		RCODE FTKAPI openDir(
			const char *	pszDirName,
			const char *	pszPattern);
	
		RCODE FTKAPI createDir(
			const char *	pszDirName);
	
		RCODE FTKAPI removeDir(
			const char *	pszDirPath);
	
		char					m_szDirectoryPath[ F_PATH_MAX_SIZE];
		char					m_szPattern[ F_PATH_MAX_SIZE];
		RCODE					m_rc;
		FLMBOOL				m_bFirstTime;
		FLMBOOL				m_bFindOpen;
		FLMUINT				m_uiAttrib;
		F_IO_FIND_DATA		m_FindData;
	#ifndef FLM_RING_ZERO_NLM
		char					m_szFileName[ F_PATH_MAX_SIZE];
	#endif
		
		friend class F_FileSystem;
	};

	/****************************************************************************
	Desc: XML
	****************************************************************************/

	typedef struct xmlChar
	{
		FLMBYTE		ucFlags;
	} XMLCHAR;
	
	class F_XML : public IF_XML
	{
	public:
	
		F_XML();
	
		virtual ~F_XML();
		
		RCODE FTKAPI setup( void);
	
		FLMBOOL FTKAPI isPubidChar(
			FLMUNICODE		uChar);
	
		FLMBOOL FTKAPI isQuoteChar(
			FLMUNICODE		uChar);
	
		FLMBOOL FTKAPI isWhitespace(
			FLMUNICODE		uChar);
	
		FLMBOOL FTKAPI isExtender(
			FLMUNICODE		uChar);
	
		FLMBOOL FTKAPI isCombiningChar(
			FLMUNICODE		uChar);
	
		FLMBOOL FTKAPI isNameChar(
			FLMUNICODE		uChar);
	
		FLMBOOL FTKAPI isNCNameChar(
			FLMUNICODE		uChar);
	
		FLMBOOL FTKAPI isIdeographic(
			FLMUNICODE		uChar);
	
		FLMBOOL FTKAPI isBaseChar(
			FLMUNICODE		uChar);
	
		FLMBOOL FTKAPI isDigit(
			FLMUNICODE		uChar);
	
		FLMBOOL FTKAPI isLetter(
			FLMUNICODE		uChar);
	
		FLMBOOL FTKAPI isNameValid(
			FLMUNICODE *	puzName,
			FLMBYTE *		pszName);
	
	private:
	
		void setCharFlag(
			FLMUNICODE		uLowChar,
			FLMUNICODE		uHighChar,
			FLMUINT16		ui16Flag);
	
		XMLCHAR *			m_pCharTable;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_FileSystem : public IF_FileSystem
	{
	public:

		F_FileSystem()
		{
		}

		virtual ~F_FileSystem()
		{
		}
		
		RCODE setup( void);

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
		
		RCODE FTKAPI createFile(
			const char *			pszFileName,
			FLMUINT					uiIoFlags,
			IF_FileHdl **			ppFile);

		RCODE FTKAPI createUniqueFile(
			char *					pszPath,
			const char *			pszFileExtension,
			FLMUINT					uiIoFlags,
			IF_FileHdl **			ppFile);

		RCODE FTKAPI createLockFile(
			const char *			pszPath,
			IF_FileHdl **			ppLockFileHdl);
			
		RCODE FTKAPI openFile(
			const char *			pszFileName,
			FLMUINT					uiIoFlags,
			IF_FileHdl **			ppFile);

		RCODE FTKAPI openDir(
			const char *			pszDirName,
			const char *			pszPattern,
			IF_DirHdl **			ppDir);

		RCODE FTKAPI createDir(
			const char *			pszDirName);

		RCODE FTKAPI removeDir(
			const char *			pszDirName,
			FLMBOOL					bClear = FALSE);

		RCODE FTKAPI doesFileExist(
			const char *			pszFileName);

		FLMBOOL FTKAPI isDir(
			const char *			pszFileName);

		RCODE FTKAPI getFileTimeStamp(
			const char *			pszFileName,
			FLMUINT *				puiTimeStamp);

		RCODE FTKAPI getFileSize(
			const char *			pszFileName,
			FLMUINT64 *				pui64FileSize);
			
		RCODE FTKAPI deleteFile(
			const char *			pszFileName);

		RCODE FTKAPI deleteMultiFileStream(
			const char *			pszDirectory,
			const char *			pszBaseName);
			
		RCODE FTKAPI copyFile(
			const char *			pszSrcFileName,
			const char *			pszDestFileName,
			FLMBOOL					bOverwrite,
			FLMUINT64 *				pui64BytesCopied);

		RCODE FTKAPI copyPartialFile(
			IF_FileHdl *			pSrcFileHdl,
			FLMUINT64				ui64SrcOffset,
			FLMUINT64				ui64SrcSize,
			IF_FileHdl *			pDestFileHdl,
			FLMUINT64				ui64DestOffset,
			FLMUINT64 *				pui64BytesCopiedRV);
		
		RCODE FTKAPI renameFile(
			const char *			pszFileName,
			const char *			pszNewFileName);

		void FTKAPI pathParse(
			const char *			pszPath,
			char *					pszServer,
			char *					pszVolume,
			char *					pszDirPath,
			char *					pszFileName);

		RCODE FTKAPI pathReduce(
			const char *			pszSourcePath,
			char *					pszDestPath,
			char *					pszString);

		RCODE FTKAPI pathAppend(
			char *					pszPath,
			const char *			pszPathComponent);

		RCODE FTKAPI pathToStorageString(
			const char *			pszPath,
			char *					pszString);

		void FTKAPI pathCreateUniqueName(
			FLMUINT *				puiTime,
			char *					pszFileName,
			const char *			pszFileExt,
			FLMBYTE *				pHighChars,
			FLMBOOL					bModext);

		FLMBOOL FTKAPI doesFileMatch(
			const char *			pszFileName,
			const char *			pszTemplate);

		RCODE FTKAPI getSectorSize(
			const char *			pszFileName,
			FLMUINT *				puiSectorSize);

		RCODE FTKAPI setReadOnly(
			const char *			pszFileName,
			FLMBOOL					bReadOnly);

		FLMBOOL FTKAPI canDoAsync( void);
		
		FLMUINT FTKAPI getPendingAsyncCount( void);
			
		RCODE FTKAPI getFileId(
			const char *			pszFileName,
			FLMUINT64 *				pui64FileId);
			
		RCODE FTKAPI allocIOBuffer(
			FLMUINT					uiMinSize,
			IF_IOBuffer **			ppIOBuffer);
			
		RCODE FTKAPI allocFileHandleCache(
			FLMUINT					uiMaxCachedFiles,
			FLMUINT					uiIdleTimeoutSecs,
			IF_FileHdlCache **	ppFileHdlCache);
			
	private:

		RCODE removeEmptyDir(
			const char *			pszDirName);

		RCODE allocFileAsyncClient(
			F_FileHdl *				pFileHdl,
			F_FileAsyncClient **	ppAsyncClient);
	
	#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
		RCODE renameSafe(
			const char *			pszSrcFile,
			const char *			pszDestFile);

		RCODE targetIsDir(
			const char	*			pszPath,
			FLMBOOL *				pbIsDir);
	#endif

		FLMBOOL				m_bCanDoAsync;
	};
	
	/****************************************************************************
	Desc: Logging
	****************************************************************************/

	void flmDbgLogInit( void);
	void flmDbgLogExit( void);
	void flmDbgLogFlush( void);

	/****************************************************************************
	Desc: Logger client
	****************************************************************************/
	RCODE f_loggerInit( void);

	void f_loggerShutdown( void);

	/****************************************************************************
	Desc:	Misc.
	****************************************************************************/
	FLMUINT f_getFSBlockSize(
		FLMBYTE *			pszFileName);
		
	#if defined( FLM_LINUX)
		void f_setupLinuxKernelVersion( void);

		void f_getLinuxKernelVersion(
			FLMUINT *		puiMajor,
			FLMUINT *		puiMinor,
			FLMUINT *		puiRevision);
			
		FLMUINT f_getLinuxMaxFileSize( void);
		
		void f_getLinuxMemInfo(
			FLMUINT64 *		pui64TotalMem,
			FLMUINT64 *		pui64AvailMem);
	#endif

	#if defined( FLM_AIX)
		void f_getAIXMemInfo(
			FLMUINT64 *		pui64TotalMem,
			FLMUINT64 *		pui64AvailMem);
	#endif
			
	#if defined( FLM_HPUX)
		void f_getHPUXMemInfo(
			FLMUINT64 *		pui64TotalMem,
			FLMUINT64 *		pui64AvailMem);
	#endif
	
	void f_memoryInit( void);
	
	void f_memoryCleanup( void);
	
	RCODE f_netwareStartup( void);
	
	void f_netwareShutdown( void);
	
	void f_initFastCheckSum( void);
	
	RCODE f_initCRCTable( void);

	void f_freeCRCTable( void);
	
	RCODE f_initFileAsyncClientList( void);
	
	void f_freeFileAsyncClientList( void);
	
	RCODE f_allocFileSystem(
		IF_FileSystem **	ppFileSystem);
		
	RCODE f_allocThreadMgr(
		IF_ThreadMgr **	ppThreadMgr);
	
	RCODE f_allocFileHdl(
		F_FileHdl **		ppFileHdl);
	
	RCODE f_allocDirHdl(
		F_DirHdl **			ppDirHdl);
		
	IF_ThreadMgr * f_getThreadMgrPtr( void);
	
	RCODE f_verifyMetaphoneRoutines( void);
	
	RCODE f_verifyDiskStructOffsets( void);

	RCODE f_initCharMappingTables( void);
	
	void f_freeCharMappingTables( void);
	
	IF_XML * f_getXmlObjPtr( void);
		
	RCODE f_netwareRemoveDir( 
		const char *		pszDirName);
	
	RCODE f_netwareTestIfFileExists(
		const char *		pPath);
		
	RCODE f_netwareDeleteFile(
		const char *		pPath);
		
	RCODE f_netwareRenameFile(
		const char *		pOldFilePath,
		const char *		pNewFilePath);

#if defined(FLM_WIN)
	typedef WINBASEAPI BOOL (WINAPI * SET_FILE_VALID_DATA_FUNC) ( 
		HANDLE				hFile,
		LONGLONG				ValidDataLength);
#endif

#endif	// FTKSYS_H
