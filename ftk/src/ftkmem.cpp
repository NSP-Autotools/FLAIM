//------------------------------------------------------------------------------
// Desc:	This file contains the f_alloc, f_calloc, f_realloc, f_recalloc,
//			and f_free routines.
// Tabs:	3
//
// Copyright (c) 1991, 1993, 1995-2007 Novell, Inc. All Rights Reserved.
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

// Cell sizes for buffer allocator

#define CELL_SIZE_0			16
#define CELL_SIZE_1			32
#define CELL_SIZE_2			64
#define CELL_SIZE_3			128
#define CELL_SIZE_4			192
#define CELL_SIZE_5			320
#define CELL_SIZE_6			512
#define CELL_SIZE_7			672
#define CELL_SIZE_8			832
#define CELL_SIZE_9			1088
#define CELL_SIZE_10			1344
#define CELL_SIZE_11			1760
#define CELL_SIZE_12			2176
#define CELL_SIZE_13			2848
#define CELL_SIZE_14			3520
#define CELL_SIZE_15			4608
#define CELL_SIZE_16			5152
#define CELL_SIZE_17			5696
#define CELL_SIZE_18 		8164
#define CELL_SIZE_19 		13068
#define CELL_SIZE_20 		16340
#define CELL_SIZE_21 		21796
#define MAX_CELL_SIZE		CELL_SIZE_21

#define NUM_BUF_ALLOCATORS	22

#if defined( FLM_RING_ZERO_NLM)

	extern rtag_t gv_lAllocRTag;
	
	#define os_malloc(size) \
		Alloc( (size), gv_lAllocRTag)

	void * nlm_realloc(
		void	*	pMemory,
		size_t	newSize);

	#define os_realloc		nlm_realloc

	#define os_free			Free

#else

	#define os_malloc			malloc
	#define os_realloc		realloc
	#define os_free			free

#endif

#if defined( FLM_UNIX)
	#ifndef MAP_ANONYMOUS
		#define MAP_ANONYMOUS	MAP_ANON
	#endif
#endif

#if defined( FLM_UNIX) && !defined( FLM_OSX)
	#include <dlfcn.h>
#endif

#if defined( FLM_AIX)
	typedef int (* VMGETINFO_FUNC)(void *, int, int);
#endif

/************************************************************************
Desc:
*************************************************************************/
typedef struct F_MemHdrTag
{
	FLMUINT			uiDataSize;
#ifdef FLM_DEBUG
	const char *	pszFileName;
	int				iLineNumber;
	FLMBOOL			bAllocFromNewOp;
	FLMUINT			uiAllocationId;
	FLMUINT			uiAllocCnt;
	FLMUINT *		puiStack;
#endif
#if FLM_ALIGN_SIZE == 8
	FLMUINT			uiDummy;
#endif
} F_MEM_HDR;

/************************************************************************
Desc:
*************************************************************************/
#define F_GET_ALLOC_PTR( pDataPtr) \
	(FLMBYTE *)((FLMBYTE *)(pDataPtr) - sizeof( F_MEM_HDR))

/************************************************************************
Desc:
*************************************************************************/
#define F_GET_DATA_PTR( pAllocPtr) \
	(FLMBYTE *)((FLMBYTE *)(pAllocPtr) + sizeof( F_MEM_HDR))

/************************************************************************
Desc:
*************************************************************************/
#define F_GET_MEM_DATA_SIZE( pDataPtr) \
	(((F_MEM_HDR *)(F_GET_ALLOC_PTR( pDataPtr)))->uiDataSize)

/************************************************************************
Desc:
*************************************************************************/
#if defined( FLM_WIN) || defined( FLM_UNIX) || defined( FLM_NLM)
	#define PTR_IN_MBLK(p,bp,offs) \
		(((FLMBYTE *)(p) > (FLMBYTE *)(bp)) && \
				((FLMBYTE *)(p) <= (FLMBYTE *)(bp) + (offs)))
#else
	#error Platform not supported
#endif

#ifdef FLM_DEBUG
	static FLMBOOL		gv_bMemTrackingInitialized = FALSE;
	static FLMUINT		gv_uiInitThreadId = 0;
	static F_MUTEX		gv_hMemTrackingMutex = F_MUTEX_NULL;
	static FLMUINT		gv_uiMemTrackingPtrArraySize = 0;
	static FLMUINT		gv_uiNumMemPtrs = 0;
	static void **		gv_ppvMemTrackingPtrs = NULL;
	static FLMUINT		gv_uiNextMemPtrSlotToUse = 0;
	static FLMUINT		gv_uiAllocCnt = 0;
	static FLMBOOL		gv_bTrackLeaks = FALSE;
	static FLMBOOL		gv_bStackWalk = FALSE;
	static FLMBOOL		gv_bLogLeaks = FALSE;
#endif

#ifdef FLM_WIN
	static HANDLE	gv_hMemProcess;
#endif

#define MEM_PTR_INIT_ARRAY_SIZE		512
#define MEM_MAX_STACK_WALK_DEPTH		32

#define F_PICKET_FENCE					"FFFFFFFF"
#if defined( FLM_DEBUG)
	#define F_PICKET_FENCE_SIZE		8
#else
	#define F_PICKET_FENCE_SIZE		0
#endif

#ifdef FLM_DEBUG
	FSTATIC FLMUINT * memWalkStack( void);
	
	FSTATIC FLMBOOL initMemTracking( void);
	
	FSTATIC void saveMemTrackingInfo(
		F_MEM_HDR *		pHdr);
	
	FSTATIC void updateMemTrackingInfo(
		F_MEM_HDR *		pHdr);
	
	FSTATIC void freeMemTrackingInfo(
		FLMBOOL			bMutexAlreadyLocked,
		FLMUINT			uiId,
		FLMUINT *		puiStack);
#endif

/****************************************************************************
Desc:
****************************************************************************/
class F_SlabManager : public IF_SlabManager
{
public:

	F_SlabManager();

	virtual ~F_SlabManager();

	RCODE FTKAPI setup(
		FLMUINT 				uiPreallocSize);
		
	RCODE FTKAPI allocSlab(
		void **				ppSlab);
		
	void FTKAPI freeSlab(
		void **				ppSlab);
		
	RCODE FTKAPI resize(
		FLMUINT 				uiNumBytes,
		FLMBOOL				bPreallocate,
		FLMUINT *			puiActualSize = NULL);

	void FTKAPI incrementTotalBytesAllocated(
		FLMUINT				uiCount)
	{
		f_assertMutexNotLocked( m_hMutex);

		f_mutexLock( m_hMutex);
		m_uiTotalBytesAllocated += uiCount;
		f_mutexUnlock( m_hMutex);
	}

	void FTKAPI decrementTotalBytesAllocated(
		FLMUINT				uiCount)
	{
		f_assertMutexNotLocked( m_hMutex);

		f_mutexLock( m_hMutex);
		f_assert( m_uiTotalBytesAllocated >= uiCount);
		m_uiTotalBytesAllocated -= uiCount;
		f_mutexUnlock( m_hMutex);
	}

	FLMUINT FTKAPI getSlabSize( void)
	{
		return( m_uiSlabSize);
	}

	FLMUINT FTKAPI getTotalSlabs( void)
	{
		return( m_uiTotalSlabs);
	}
	
	FLMUINT FTKAPI totalBytesAllocated( void)
	{
		return( m_uiTotalBytesAllocated);
	}

	FLMUINT FTKAPI getTotalSlabBytesAllocated( void)
	{
		FLMUINT		uiTotalSlabBytes;

		f_mutexLock( m_hMutex);
		uiTotalSlabBytes = m_uiSlabSize * m_uiTotalSlabs;
		f_mutexUnlock( m_hMutex);

		return( uiTotalSlabBytes);
	}

	FLMUINT FTKAPI availSlabs( void)
	{
		return( m_uiAvailSlabs);
	}
	
private:

	void freeAllSlabs( void);
	
	void * allocSlabFromSystem( void);
	
	void releaseSlabToSystem(
		void *				pSlab);

	RCODE sortSlabList( void);

	typedef struct
	{
		void *				pPrev;
		void *				pNext;
	} SLABHEADER;

	static FLMINT FTKAPI slabAddrCompareFunc(
		void *				pvBuffer,
		FLMUINT				uiPos1,
		FLMUINT				uiPos2);

	static void FTKAPI slabAddrSwapFunc(
		void *				pvBuffer,
		FLMUINT				uiPos1,
		FLMUINT				uiPos2);
	
	F_MUTEX					m_hMutex;
	FLMUINT					m_uiTotalBytesAllocated;
	void *					m_pFirstInSlabList;
	void *					m_pLastInSlabList;
	FLMUINT					m_uiSlabSize;
	FLMUINT					m_uiTotalSlabs;
	FLMUINT					m_uiAvailSlabs;
	FLMUINT					m_uiInUseSlabs;
	FLMUINT					m_uiPreallocSlabs;

friend class F_FixedAlloc;
};

/****************************************************************************
Desc:
****************************************************************************/
typedef struct FIXEDSLAB
{
	void *		pvAllocator;
	FIXEDSLAB *	pNext;
	FIXEDSLAB *	pPrev;
	FIXEDSLAB *	pNextSlabWithAvailCells;
	FIXEDSLAB *	pPrevSlabWithAvailCells;
	FLMBYTE *	pLocalAvailCellListHead;
	FLMUINT16	ui16NextNeverUsedCell;
	FLMUINT16	ui16AvailCellCount;
	FLMUINT16	ui16AllocatedCells;
} FIXEDSLAB;

/****************************************************************************
Desc:
****************************************************************************/
typedef struct CELLHEADER
{
	FIXEDSLAB *		pContainingSlab;
#ifdef FLM_DEBUG
	FLMUINT *		puiStack;
#endif
} CELLHEADER;

/****************************************************************************
Desc:
****************************************************************************/
typedef struct CELLHEADER2
{
	CELLHEADER		cellHeader;
	IF_Relocator *	pRelocator;
} CELLHEADER2;

/****************************************************************************
Desc:
****************************************************************************/
typedef struct CELLAVAILNEXT
{
	FLMBYTE *	pNextInList;
#ifdef FLM_DEBUG
	FLMBYTE		szDebugPattern[ 8];
#endif
} CELLAVAILNEXT;

/****************************************************************************
Desc:	Class to provide an efficient means of providing many allocations
		of a fixed size.
****************************************************************************/
class F_FixedAlloc : public IF_FixedAlloc
{
public:

	F_FixedAlloc();

	virtual ~F_FixedAlloc();

	RCODE FTKAPI setup(
		FLMBOOL					bMultiThreaded,
		IF_SlabManager *		pSlabManager,
		IF_Relocator *			pDefaultRelocator,
		FLMUINT					uiCellSize,
		FLM_SLAB_USAGE *		pUsageStats,
		FLMUINT *				puiTotalBytesAllocated);

	void * FTKAPI allocCell(
		IF_Relocator *			pRelocator,
		void *					pvInitialData = NULL,
		FLMUINT					uiDataSize = 0);

	void * FTKAPI allocCell(
		IF_Relocator *			pRelocator,
		F_ALLOC_INIT_FUNC		fnAllocInit);
	
	void FTKAPI freeCell( 
		void *					ptr);

	void FTKAPI freeUnused( void);

	void FTKAPI freeAll( void);

	FLMUINT FTKAPI getCellSize( void)
	{
		return( m_uiCellSize);
	}
	
	void FTKAPI defragmentMemory( void);
	
private:

	void * getCell(
		IF_Relocator *			pRelocator);

	FIXEDSLAB * getAnotherSlab( void);

	static FLMUINT getAllocAlignedSize(
		FLMUINT					uiAskedForSize)
	{
		return( (uiAskedForSize + FLM_ALLOC_ALIGN) & (~FLM_ALLOC_ALIGN));
	}

	void freeSlab( 
		FIXEDSLAB *				pSlab);

	void freeCell(
		void *					pCell,
		FLMBOOL					bFreeIfEmpty,
		FLMBOOL *				pbFreedSlab);

#ifdef FLM_DEBUG
	void testForLeaks( void);
#endif

	static FLMINT FTKAPI slabAddrCompareFunc(
		void *					pvBuffer,
		FLMUINT					uiPos1,
		FLMUINT					uiPos2)
	{
		FIXEDSLAB *	pSlab1 = (((FIXEDSLAB **)pvBuffer)[ uiPos1]);
		FIXEDSLAB *	pSlab2 = (((FIXEDSLAB **)pvBuffer)[ uiPos2]);

		f_assert( pSlab1 != pSlab2);

		if( pSlab1 < pSlab2)
		{
			return( -1);
		}

		return( 1);
	}

	static void FTKAPI slabAddrSwapFunc(
		void *					pvBuffer,
		FLMUINT					uiPos1,
		FLMUINT					uiPos2)
	{
		FIXEDSLAB **			ppSlab1 = &(((FIXEDSLAB **)pvBuffer)[ uiPos1]);
		FIXEDSLAB **			ppSlab2 = &(((FIXEDSLAB **)pvBuffer)[ uiPos2]);
		FIXEDSLAB *				pTmp;

		pTmp = *ppSlab1;
		*ppSlab1 = *ppSlab2;
		*ppSlab2 = pTmp;
	}

	IF_SlabManager *			m_pSlabManager;
	FIXEDSLAB *					m_pFirstSlab;
	FIXEDSLAB *					m_pLastSlab;
	FIXEDSLAB *					m_pFirstSlabWithAvailCells;
	FIXEDSLAB *					m_pLastSlabWithAvailCells;
	IF_Relocator *				m_pDefaultRelocator;
	FLMBOOL						m_bAvailListSorted;
	FLMUINT						m_uiSlabsWithAvailCells;
	FLMUINT						m_uiSlabHeaderSize;
	FLMUINT						m_uiCellHeaderSize;
	FLMUINT						m_uiCellSize;
	FLMUINT						m_uiSizeOfCellAndHeader; 
	FLMUINT						m_uiTotalFreeCells;
	FLMUINT						m_uiCellsPerSlab;
	FLMUINT						m_uiSlabSize;
	FLM_SLAB_USAGE *			m_pUsageStats;
	FLMUINT *					m_puiTotalBytesAllocated;
	F_MUTEX						m_hMutex;
	
friend class F_BufferAlloc;
friend class F_MultiAlloc;
};

/****************************************************************************
Desc:
****************************************************************************/
class F_BufferAlloc : public IF_BufferAlloc
{
public:

	F_BufferAlloc()
	{
		f_memset( m_ppAllocators, 0, sizeof( m_ppAllocators));
		m_pSlabManager = NULL;
		m_hMutex = F_MUTEX_NULL;
	}

	virtual ~F_BufferAlloc();

	RCODE FTKAPI setup(
		FLMBOOL					bMultiThreaded,
		IF_SlabManager *		pSlabManager,
		IF_Relocator *			pDefaultRelocator,
		FLM_SLAB_USAGE *		pUsageStats,
		FLMUINT *				puiTotalBytesAllocated);

	RCODE FTKAPI allocBuf(
		IF_Relocator *			pRelocator,
		FLMUINT					uiSize,
		void *					pvInitialData,
		FLMUINT					uiDataSize,
		FLMBYTE **				ppucBuffer,
		FLMBOOL *				pbAllocatedOnHeap = NULL);

	RCODE FTKAPI allocBuf(
		IF_Relocator *			pRelocator,
		FLMUINT					uiSize,
		F_ALLOC_INIT_FUNC		fnAllocInit,
		FLMBYTE **				ppucBuffer,
		FLMBOOL *				pbAllocatedOnHeap = NULL);
			
	RCODE FTKAPI reallocBuf(
		IF_Relocator *			pRelocator,
		FLMUINT					uiOldSize,
		FLMUINT					uiNewSize,
		void *					pvInitialData,
		FLMUINT					uiDataSize,
		FLMBYTE **				ppucBuffer,
		FLMBOOL *				pbAllocatedOnHeap = NULL);

	void FTKAPI freeBuf(
		FLMUINT					uiSize,
		FLMBYTE **				ppucBuffer);

	FLMUINT FTKAPI getTrueSize(
		FLMUINT					uiSize,
		FLMBYTE *				pucBuffer);

	FLMUINT FTKAPI getMaxCellSize( void)
	{
		return( MAX_CELL_SIZE);
	}
		
	void FTKAPI defragmentMemory( void);
	
private:

	IF_FixedAlloc * getAllocator(
		FLMUINT					uiSize);

	IF_SlabManager *			m_pSlabManager;
	IF_FixedAlloc *			m_ppAllocators[ NUM_BUF_ALLOCATORS];
	F_MUTEX						m_hMutex;
};

/****************************************************************************
Desc:
****************************************************************************/
class F_MultiAlloc : public IF_MultiAlloc
{
public:

	F_MultiAlloc()
	{
		m_pSlabManager = NULL;
		m_puiCellSizes = NULL;
		m_ppAllocators = NULL;
		m_hMutex = F_MUTEX_NULL;
	}

	~F_MultiAlloc()
	{
		cleanup();
	}

	RCODE FTKAPI setup(
		FLMBOOL					bMultiThreaded,
		IF_SlabManager *		pSlabManager,
		IF_Relocator *			pDefaultRelocator,
		FLMUINT *				puiCellSizes,
		FLM_SLAB_USAGE *		pUsageStats,
		FLMUINT *				puiTotalBytesAllocated);

	RCODE FTKAPI allocBuf(
		IF_Relocator *			pRelocator,
		FLMUINT					uiSize,
		FLMBYTE **				ppucBuffer);

	RCODE FTKAPI allocBuf(
		IF_Relocator *			pRelocator,
		FLMUINT					uiSize,
		F_ALLOC_INIT_FUNC		fnAllocInit,
		FLMBYTE **				ppucBuffer);
		
	RCODE FTKAPI reallocBuf(
		IF_Relocator *			pRelocator,
		FLMUINT					uiNewSize,
		FLMBYTE **				ppucBuffer);

	void FTKAPI freeBuf(
		FLMBYTE **				ppucBuffer);

	void FTKAPI defragmentMemory( void);

	FLMUINT FTKAPI getTrueSize(
		FLMBYTE *				pucBuffer);

private:

	IF_FixedAlloc * getAllocator(
		FLMUINT					uiSize);

	IF_FixedAlloc * getAllocator(
		FLMBYTE *				pucBuffer);

	void cleanup( void);

	IF_SlabManager *			m_pSlabManager;
	FLMUINT *					m_puiCellSizes;
	IF_FixedAlloc **			m_ppAllocators;
	F_MUTEX						m_hMutex;
};

class F_ObjRefTracker;

/****************************************************************************
Desc:	
****************************************************************************/
typedef RCODE (* ADDR_FMT_HOOK)(					// Address formatter / translator
	F_ObjRefTracker *			pRefTracker,		// Reference tracker object
	void *						pAddress,			// Pointer to the address
	FLMBYTE *					pucBuffer,			// Buffer for formatted address
	FLMUINT						uiSize,				// Size of buffer
	void *						pvUserData);		// User-supplied callback data

/****************************************************************************
Desc:	
****************************************************************************/
class F_ObjRefTracker : public F_Object
{
public:

	F_ObjRefTracker( void);
	
	virtual ~F_ObjRefTracker( void);

	RCODE setup(
		const char *		pszObjName,
		FLMBOOL				bLogToFile = FALSE);

	void trackRef(
		void *				pReferenceID,
		void *				pSubrefID = NULL);
		
	void untrackRef(
		void *				referenceID,
		void *				subrefID = NULL);
		
	void checkForUnreleasedRefs(
		FLMUINT *			puiCount = NULL);

	void setAddressFormatter(
		ADDR_FMT_HOOK 		pFunc,
		void *				pvUserData);

	void setModuleHandle( 
		void *				pModHandle) 
	{ 
		m_pModHandle = pModHandle;
	}
	
private:

	F_MUTEX					m_hRefListMutex;
	F_ListManager *		m_pListManager;
	F_ListNode				m_lnode;
	FLMUINT					m_lOptions;
	FLMUINT					m_lCallStackDepth;

#define FORTRACK_MAX_OBJ_NAME_LEN		63
	char	 					m_pszObjName[ FORTRACK_MAX_OBJ_NAME_LEN + 1];
	IF_FileSystem *		m_pFileSystem;
	ADDR_FMT_HOOK			m_pAddrFmtHook;
	void *					m_pUserData;
	void *					m_pModHandle;
	char						m_pLogPath[ F_PATH_MAX_SIZE];

	void formatAddress(
		char *				pucBuffer,
		FLMUINT				uiSize,
		void *				pAddress);

	static void getCallStack(
		void *				stack[],
		FLMUINT				uiCount,
		FLMUINT				uiSkip);
		
	void logError(
		const char *		pucMessage);

	RCODE logMessage(
		const char *		pucMessage,
		IF_FileHdl *		pFileHdl,
		FLMUINT64 *			pui64FileCursor);
};

/****************************************************************************
Desc:	
****************************************************************************/
class F_TrackingRecord : public F_ListItem
{
public:

	F_TrackingRecord( void * pReferenceID, void * pSubrefID)
	{
		m_pReferenceID = pReferenceID;
		m_pSubrefID = pSubrefID;
		m_uiThreadID = f_threadId();
		f_memset( m_stack, 0, sizeof(m_stack));
	}
	
	virtual ~F_TrackingRecord()
	{
	}

	void * getReferenceID()
	{
		return m_pReferenceID;
	}
	void * getSubrefID()
	{
		return m_pSubrefID;
	}
	FLMUINT getThreadID()
	{
		return m_uiThreadID;
	}
	void * getStack()
	{
		return m_stack;
	}
	
private:

	void *					m_pReferenceID;
	void *					m_pSubrefID;
	FLMUINT					m_uiThreadID;
#define CTRC_STACK_SIZE 20
	void *					m_stack[ CTRC_STACK_SIZE + 1];
};

/****************************************************************************
Desc:	
****************************************************************************/
RCODE FTKAPI FlmAllocSlabManager(
	IF_SlabManager **			ppSlabManager)
{
	if( (*ppSlabManager = f_new F_SlabManager) == NULL)
	{
		return( RC_SET( NE_FLM_MEM));
	}
	
	return( NE_FLM_OK);
}
	
/****************************************************************************
Desc:	
****************************************************************************/
RCODE FTKAPI FlmAllocFixedAllocator(
	IF_FixedAlloc **			ppFixedAllocator)
{
	if( (*ppFixedAllocator = f_new F_FixedAlloc) == NULL)
	{
		return( RC_SET( NE_FLM_MEM));
	}
	
	return( NE_FLM_OK);
}

/****************************************************************************
Desc:	
****************************************************************************/
RCODE FTKAPI FlmAllocBufferAllocator(
	IF_BufferAlloc **			ppBufferAllocator)
{
	if( (*ppBufferAllocator = f_new F_BufferAlloc) == NULL)
	{
		return( RC_SET( NE_FLM_MEM));
	}
	
	return( NE_FLM_OK);
}

/****************************************************************************
Desc:	
****************************************************************************/
RCODE FTKAPI FlmAllocMultiAllocator(
	IF_MultiAlloc **			ppMultiAllocator)
{
	if( (*ppMultiAllocator = f_new F_MultiAlloc) == NULL)
	{
		return( RC_SET( NE_FLM_MEM));
	}
	
	return( NE_FLM_OK);
}
		
/************************************************************************
Desc:
*************************************************************************/
FLMUINT f_msize(
	void *			pvPtr)
{
#if defined( FLM_UNIX)
	return( pvPtr ? F_GET_MEM_DATA_SIZE( (pvPtr)) : 0);
#elif defined( FLM_LIBC_NLM)
		return( pvPtr ? msize( (F_GET_ALLOC_PTR( (pvPtr)))) : 0);
#elif defined( FLM_RING_ZERO_NLM)
	return( pvPtr ? (unsigned)SizeOfAllocBlock(
		(F_GET_ALLOC_PTR( (pvPtr)))) : 0);
#else
		return( pvPtr ? _msize( (F_GET_ALLOC_PTR( (pvPtr)))) : 0);
#endif
}

/************************************************************************
Desc:	Returns the current value of EBP--the value of the caller's stack 
		frame pointer.
*************************************************************************/
#ifdef FLM_NLM
	void * memGetEBP(void);
	
	#ifdef FLM_MWERKS_NLM
		void * memGetEBP(void)
		{
			__asm
			{
				mov	eax,[ebp]
			}
		}
	#else
		#pragma aux memGetEBP = "mov eax,ebp";
	#endif
#endif

/************************************************************************
Desc:	Returns the value at SS:[POS+OFFSET].
*************************************************************************/
#ifdef FLM_NLM

	void * memValueAtStackOffset(void *pos, int offset);

	#ifdef FLM_MWERKS_NLM
		void * memValueAtStackOffset(void *, int)
		{
			__asm
			{
				mov	eax,[ebp+0x8]
				mov	ebx,[ebp+0xC]
				mov	eax,ss:[eax+ebx]
			}
		}
	#else
		#pragma aux memValueAtStackOffset = "mov eax,ss:[eax+ebx]" parm [eax] [ebx];
	#endif
	
#endif

#ifdef FLM_DEBUG
/************************************************************************
Desc:
*************************************************************************/
#ifdef FLM_NLM
FSTATIC FLMUINT * memWalkStack( void)
{
	FLMUINT		uiLoop;
 	FLMUINT		uiRtnAddr;
	FLMUINT		uiEbp = (FLMUINT) memGetEBP();
	FLMUINT		uiAddresses [MEM_MAX_STACK_WALK_DEPTH + 1];
	FLMUINT *	puiAddresses;

	uiEbp = (FLMUINT) memValueAtStackOffset( (void *)uiEbp, 0);
	uiRtnAddr = (FLMUINT) memValueAtStackOffset( (void *)uiEbp, 4);

	for (uiLoop = 0; uiLoop < MEM_MAX_STACK_WALK_DEPTH; uiLoop++)
	{
		FLMUINT	uiOldEbp;
		
		uiAddresses [uiLoop] = uiRtnAddr;
		if (!uiEbp)
		{
			break;
		}
		
		uiOldEbp = uiEbp;
		uiEbp = (FLMUINT) memValueAtStackOffset( (void *)uiEbp, 0);

		if (!uiEbp || uiEbp <= uiOldEbp || uiEbp > uiOldEbp + 5000)
		{
			break;
		}

		uiRtnAddr = (FLMUINT) memValueAtStackOffset( (void *) uiEbp, 4);
	}
	uiAddresses [uiLoop] = 0;
	if ((puiAddresses = (FLMUINT *)os_malloc( 
		sizeof( FLMUINT) * (uiLoop+1))) != NULL)
	{
		f_memcpy( puiAddresses, &uiAddresses [0], sizeof( FLMUINT) * (uiLoop + 1));
	}
	return( puiAddresses);
}
#endif

/********************************************************************
Desc: Walk the call stack.
*********************************************************************/
#ifdef FLM_WIN
FSTATIC FLMUINT * memWalkStack( void)
{
	STACKFRAME64	stackFrame;
	CONTEXT			context;
	DWORD				machineType;
	FLMUINT			uiLoop;
	FLMUINT			uiAddresses [MEM_MAX_STACK_WALK_DEPTH + 1];
	FLMUINT *		puiAddresses;
	HANDLE			hThread;
	HANDLE			hProcess;
	FLMUINT			uiAddrCount;

	f_memset( &stackFrame, 0, sizeof( stackFrame));
	f_memset( &context, 0, sizeof( context));

	// While you can continue walking the stack...

#ifdef FLM_64BIT
	machineType = IMAGE_FILE_MACHINE_IA64;

	GetThreadContext(GetCurrentThread(), &context);

	stackFrame.AddrPC.Offset = context.Rip;
	stackFrame.AddrFrame.Offset = context.Rbp;
#else
	machineType = IMAGE_FILE_MACHINE_I386;

	unsigned vEBP, vEIP;
	__asm mov vEBP, ebp
	__asm call near nextinstr
nextinstr:
	__asm pop vEIP;

	context.Ebp = vEBP;
	context.Eip = vEIP;

	stackFrame.AddrPC.Offset = vEIP;
	stackFrame.AddrFrame.Offset = vEBP;
#endif

	stackFrame.AddrPC.Mode = AddrModeFlat;
	stackFrame.AddrFrame.Mode = AddrModeFlat;

	// Must lock the mutex because StackWalk is not thread safe.

	f_mutexLock( gv_hMemTrackingMutex);
	hProcess = OpenProcess( PROCESS_VM_READ, FALSE, GetCurrentProcessId());
	hThread = OpenThread( THREAD_GET_CONTEXT | THREAD_SUSPEND_RESUME, 
			FALSE, GetCurrentThreadId());

	// We have already processed the address inside memWalkStack

	uiAddrCount = 1;
	uiLoop = 0;
	for (;;)
	{
		if (!StackWalk64( machineType, hProcess, hThread, &stackFrame,
							&context, NULL,
							SymFunctionTableAccess64, SymGetModuleBase64, NULL))
		{
			break;
		}

		// Skip the first two addresses.  These represent the following:
		// 1) memWalkStack
		// 2) saveMemTrackingInfo or updateMemTrackingInfo
		// We don't need to see them in the stack trace.

		uiAddrCount++;
		if (uiAddrCount > 2)
		{
			uiAddresses [uiLoop] = (FLMUINT)stackFrame.AddrReturn.Offset;
			uiLoop++;
			if (uiLoop == MEM_MAX_STACK_WALK_DEPTH)
			{
				break;
			}
		}
	}

	f_mutexUnlock( gv_hMemTrackingMutex);

	uiAddresses [uiLoop] = 0;
	if ((puiAddresses = (FLMUINT *)os_malloc( 
		sizeof( FLMUINT) * (uiLoop+1))) != NULL)
	{
		f_memcpy( puiAddresses, &uiAddresses [0], sizeof( FLMUINT) * (uiLoop + 1));
	}
	return( puiAddresses);
}
#endif

/********************************************************************
Desc:
*********************************************************************/
#ifdef FLM_UNIX
FSTATIC FLMUINT * memWalkStack( void)
{
	return( NULL);
}
#endif
#endif

/********************************************************************
Desc: Initialize memory tracking
*********************************************************************/
#ifdef FLM_DEBUG
FSTATIC FLMBOOL initMemTracking( void)
{
	RCODE		rc;
	F_MUTEX	memMutex;

	if (!gv_bMemTrackingInitialized && !gv_uiInitThreadId)
	{
		gv_uiInitThreadId = f_threadId();
		rc = f_mutexCreate( &memMutex);
		f_sleep( 50);

		// Only set to initialized if we were the last thread
		// to set gv_uiInitThreadId

		if (f_threadId() == gv_uiInitThreadId)
		{
			if (RC_OK( rc))
			{
				gv_hMemTrackingMutex = memMutex;
			}
			else
			{
				gv_hMemTrackingMutex = F_MUTEX_NULL;
			}
#ifdef FLM_WIN
			SymSetOptions( SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
			gv_hMemProcess = GetCurrentProcess();
			SymInitialize( gv_hMemProcess, NULL, TRUE);
			
			gv_bTrackLeaks = TRUE;
			gv_bStackWalk = FALSE;
#endif
			gv_bMemTrackingInitialized = TRUE;
		}
		else
		{
			if (RC_OK( rc))
			{
				f_mutexDestroy( &memMutex);
			}
		}
	}

	// Go into a loop until we see initialized flag set to TRUE
	// Could be another thread that is doing it.

	while( !gv_bMemTrackingInitialized)
	{
		f_sleep( 10);
	}
	
	return( (gv_hMemTrackingMutex != F_MUTEX_NULL) ? TRUE : FALSE);
}
#endif

/********************************************************************
Desc: Save memory tracking information - called on alloc or realloc.
*********************************************************************/
#ifdef FLM_DEBUG
FSTATIC void saveMemTrackingInfo(
	F_MEM_HDR *	pHdr)
{
	FLMUINT	uiNewCnt;
	FLMUINT	uiId;
	void **	pNew;

	if (gv_bTrackLeaks && initMemTracking())
	{
		f_mutexLock( gv_hMemTrackingMutex);

		// See if there is enough room in the array

		if (gv_uiNumMemPtrs == gv_uiMemTrackingPtrArraySize)
		{

			// If array is not initialized, use initial count.  Otherwise
			// double the size.

			uiNewCnt = (FLMUINT)((!gv_uiMemTrackingPtrArraySize)
										? MEM_PTR_INIT_ARRAY_SIZE
										: gv_uiMemTrackingPtrArraySize * 2);
			if ((pNew = (void **)os_malloc( sizeof( void *) * uiNewCnt)) != NULL)
			{

				// Copy the pointers from the old array, if any,
				// into the newly allocated array.

				if (gv_uiMemTrackingPtrArraySize)
				{
					f_memcpy( pNew, gv_ppvMemTrackingPtrs,
							sizeof( void *) * gv_uiMemTrackingPtrArraySize);
					os_free( gv_ppvMemTrackingPtrs);
					gv_ppvMemTrackingPtrs = NULL;
				}
				f_memset( &pNew [gv_uiMemTrackingPtrArraySize], 0,
						sizeof( void *) * (uiNewCnt - gv_uiMemTrackingPtrArraySize));
				gv_ppvMemTrackingPtrs = pNew;
				gv_uiMemTrackingPtrArraySize = uiNewCnt;
			}
		}

		// If we are still full, we were not able to reallocate memory, so we
		// do nothing.

		if (gv_uiNumMemPtrs == gv_uiMemTrackingPtrArraySize)
		{
			pHdr->uiAllocationId = 0;
		}
		else
		{
			// Find an empty slot - there has to be one!

			uiId = gv_uiNextMemPtrSlotToUse;
			while (gv_ppvMemTrackingPtrs [uiId])
			{
				if (++uiId == gv_uiMemTrackingPtrArraySize)
				{
					uiId = 0;
				}
			}

			// Allocation ID in the header is offset by one to avoid
			// using a value of zero.

			pHdr->uiAllocationId = uiId + 1;
			gv_ppvMemTrackingPtrs [uiId] = pHdr;
			gv_uiNumMemPtrs++;
			if ((gv_uiNextMemPtrSlotToUse = uiId + 1) ==
					gv_uiMemTrackingPtrArraySize)
			{
				gv_uiNextMemPtrSlotToUse = 0;
			}
		}
		pHdr->uiAllocCnt = ++gv_uiAllocCnt;
		f_mutexUnlock( gv_hMemTrackingMutex);
	}
	else
	{
		pHdr->uiAllocationId = 0;
		pHdr->uiAllocCnt = 0;
	}

	// Follow the stack.

	if (gv_bTrackLeaks && gv_bStackWalk)
	{
		pHdr->puiStack = memWalkStack();
	}
	else
	{
		pHdr->puiStack = NULL;
	}
}
#endif

/********************************************************************
Desc: Update memory tracking information - called after realloc
*********************************************************************/
#ifdef FLM_DEBUG
FSTATIC void updateMemTrackingInfo(
	F_MEM_HDR *	pHdr)
{
	if (pHdr->puiStack)
	{
		os_free( pHdr->puiStack);
		pHdr->puiStack = NULL;
	}
	if (gv_bTrackLeaks && gv_bStackWalk)
	{
		pHdr->puiStack = memWalkStack();
	}
}
#endif

/********************************************************************
Desc: Free memory tracking information - called on free.
*********************************************************************/
#ifdef FLM_DEBUG
FSTATIC void freeMemTrackingInfo(
	FLMBOOL		bMutexAlreadyLocked,
	FLMUINT		uiId,
	FLMUINT *	puiStack)
{
	if( uiId)
	{
		// NOTE: If uiId is non-zero, it means we had to have
		// successfully initialized, so we are guaranteed to
		// have a mutex.

		if( !bMutexAlreadyLocked)
		{
			f_mutexLock( gv_hMemTrackingMutex);
		}

		// Allocation ID in the header is offset by one so that it
		// is never zero - a value of zero means that the allocation
		// does not have a slot for tracking it in the array.

		if( gv_ppvMemTrackingPtrs)
		{
			gv_ppvMemTrackingPtrs[ uiId - 1] = NULL;
			f_assert( gv_uiNumMemPtrs);
			gv_uiNumMemPtrs--;
		}

		if ( !bMutexAlreadyLocked)
		{
			f_mutexUnlock( gv_hMemTrackingMutex);
		}
	}

	// Free the stack information, if any.

	if( puiStack)
	{
		os_free( puiStack);
	}
}
#endif

/********************************************************************
Desc: Log memory leaks.
*********************************************************************/
#ifdef FLM_DEBUG
void logMemLeak(
	F_MEM_HDR *			pHdr)
{
	char					szTmpBuffer [1024];
	FLMUINT				uiMsgBufSize;
	char *				pszMessageBuffer;
	char *				pszTmp;
	IF_FileHdl *		pFileHdl = NULL;
	FLMBOOL				bSaveTrackLeaks = gv_bTrackLeaks;
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();

	gv_bTrackLeaks = FALSE;

	// Need a big buffer to show an entire stack.

	uiMsgBufSize = 20480;
	if ((pszMessageBuffer = (char *)os_malloc( uiMsgBufSize)) == NULL)
	{
		pszMessageBuffer = &szTmpBuffer [0];
		uiMsgBufSize = sizeof( szTmpBuffer);
	}
	pszTmp = pszMessageBuffer;

	// Format message to be logged.

	f_strcpy( pszTmp, "Abort=Debug, Retry=Continue, Ignore=Don't Show\r\n");
	
	while (*pszTmp)
	{
		pszTmp++;
	}
	
#if defined( FLM_WIN) && defined( FLM_64BIT)
	f_sprintf(	pszTmp, "Unfreed Pointer: 0x%016I64x\r\n", (FLMUINT)(&pHdr [1]));
#else
	f_sprintf(	pszTmp, "Unfreed Pointer: 0x%08x\r\n",
		(unsigned)((FLMUINT)(&pHdr [1])));
#endif

	while( *pszTmp)
	{
		pszTmp++;
	}

	if( pHdr->pszFileName)
	{
		f_sprintf( pszTmp, "Source: %s, Line#: %u\r\n", pHdr->pszFileName,
								(unsigned)pHdr->iLineNumber);
		while (*pszTmp)
		{
			pszTmp++;
		}
	}

	if( pHdr->uiAllocCnt)
	{
		f_sprintf( pszTmp, "Malloc #: %u\r\n", (unsigned)pHdr->uiAllocCnt);
		while (*pszTmp)
		{
			pszTmp++;
		}
  	}
	
	f_sprintf( (char *)pszTmp, "Size: %u bytes\r\n", (unsigned)pHdr->uiDataSize);
	
	while( *pszTmp)
	{
		pszTmp++;
	}

	if( pHdr->puiStack)
	{
		FLMUINT *			puiStack = pHdr->puiStack;
		FLMUINT				uiLen = pszTmp - pszMessageBuffer;
		char					szFuncName [200];
		char *				pszFuncName;
#ifdef FLM_WIN
		IMAGEHLP_SYMBOL *	pImgHlpSymbol;

		pImgHlpSymbol = (IMAGEHLP_SYMBOL *)os_malloc(
									sizeof( IMAGEHLP_SYMBOL) + 100);
#endif

		while (*puiStack)
		{
			szFuncName [0] = 0;
#ifdef FLM_WIN
			if (pImgHlpSymbol)
			{
	#ifdef FLM_64BIT
				DWORD64		udDisplacement;
	#else
				DWORD			udDisplacement;
	#endif

				pImgHlpSymbol->SizeOfStruct = sizeof(IMAGEHLP_SYMBOL);
				pImgHlpSymbol->Address = *puiStack;
				pImgHlpSymbol->MaxNameLength = 100;

				if (SymGetSymFromAddr( gv_hMemProcess, *puiStack,
												&udDisplacement, pImgHlpSymbol))
				{
					f_sprintf( szFuncName, "\t%s + %X\r\n",
									(char *)(&pImgHlpSymbol->Name [0]),
									udDisplacement);
				}
			}
#else

#ifdef HAVE_DLADDR
			{
				Dl_info	dlip;

				if (dladdr( (void *)(*puiStack), &dlip) != 0 && dlip.dli_sname)
				{
					const char *	pszFileName;
					if (dlip.dli_saddr != (void *)(*puiStack))
					{
						pszFileName = strrchr(dlip.dli_fname, '/');
						if (!pszFileName)
						{
							pszFileName = dlip.dli_fname;
						}
						else
						{
							pszFileName++;		// skip over slash
						}
						f_sprintf( szFuncName, "\t0x%08x (%s)\r\n",
									(unsigned)(*puiStack), pszFileName); 
					}
					else
					{
						f_sprintf( szFuncName, "\t%s\r\n", dlip.dli_sname);
					}
				}
			}
#endif

#endif

			// If szFuncName [0] is zero, we didn't find a name, so we
			// just output the address in HEX.

			if (!szFuncName [0])
			{
				f_sprintf( szFuncName, "\t0x%08X\r\n", (unsigned)*puiStack );
			}

			// Output whatever portion of the name will fit into the
			// message buffer.

			pszFuncName = &szFuncName [0];
			while (*pszFuncName && uiLen < uiMsgBufSize - 1)
			{
				*pszTmp++ = *pszFuncName++;
				uiLen++;
			}

			// Process next address in the stack.

			puiStack++;
		}
		*pszTmp = 0;
#ifdef FLM_WIN
		if (pImgHlpSymbol)
		{
			os_free( pImgHlpSymbol);
		}
#endif
	}

#ifdef FLM_WIN
	FLMINT				iRet;

	iRet =  MessageBox( NULL, (LPCTSTR)pszMessageBuffer, "WIN32 Memory Testing",
					MB_ABORTRETRYIGNORE | MB_ICONINFORMATION | MB_TASKMODAL 
					| MB_SETFOREGROUND | MB_DEFBUTTON2);
	if (iRet == IDIGNORE)
	{
		gv_bLogLeaks = TRUE;
	}
	else if (iRet == IDABORT)
	{
		f_assert( 0);
	}
#else
	gv_bLogLeaks = TRUE;
#endif

	if (gv_bLogLeaks && pFileSystem)
	{
		RCODE				rc;
		FLMUINT			uiDummy;
#ifdef FLM_NLM
		const char *	pszErrPath = "sys:\\memtest.ert";
#else
		const char *	pszErrPath = "memtest.ert";
#endif

	if (RC_BAD( rc = pFileSystem->openFile( pszErrPath, 
		FLM_IO_RDWR | FLM_IO_SH_DENYNONE, &pFileHdl)))
		{
			if (rc == NE_FLM_IO_PATH_NOT_FOUND)
			{
				rc = pFileSystem->createFile( pszErrPath, 
					FLM_IO_RDWR | FLM_IO_SH_DENYNONE, &pFileHdl);
			}
		}
		else
		{
			// Position to append to file.

			rc = pFileHdl->seek( 0, FLM_IO_SEEK_END, NULL);
		}

		// If we successfully opened the file, write to it.

		if (RC_OK( rc))
		{
			if (RC_OK( pFileHdl->write( FLM_IO_CURRENT_POS,
						(FLMUINT)(pszTmp - pszMessageBuffer),
						pszMessageBuffer, &uiDummy)))
			{
				(void)pFileHdl->flush();
			}
			
			pFileHdl->closeFile();
		}
	}
	
	if (pFileHdl)
	{
		pFileHdl->Release();
	}

	if (pszMessageBuffer != &szTmpBuffer [0])
	{
		os_free( pszMessageBuffer);
	}

	gv_bTrackLeaks = bSaveTrackLeaks;
}
#endif

/********************************************************************
Desc: Initialize memory - if not already done.
*********************************************************************/
void f_memoryInit( void)
{
#ifdef FLM_DEBUG
	(void)initMemTracking();
#endif

#if defined( FLM_UNIX) && defined( RLIMIT_VMEM)
{
	struct rlimit	rlim;

	// Bump the process soft virtual limit up to the hard limit
	
	if( getrlimit( RLIMIT_VMEM, &rlim) == 0)
	{
		if( rlim.rlim_cur < rlim.rlim_max)
		{
			rlim.rlim_cur = rlim.rlim_max;
			(void)setrlimit( RLIMIT_VMEM, &rlim);
		}
	}
}
#endif

#if defined( FLM_UNIX) && defined( RLIMIT_DATA)
{
	struct rlimit	rlim;

	// Bump the process soft heap limit up to the hard limit
	
	if( getrlimit( RLIMIT_DATA, &rlim) == 0)
	{
		if( rlim.rlim_cur < rlim.rlim_max)
		{
			rlim.rlim_cur = rlim.rlim_max;
			(void)setrlimit( RLIMIT_DATA, &rlim);
		}
	}
}
#elif defined( FLM_OSX)
	#error OS X should provide support for RLIMIT_DATA
#endif
}

/********************************************************************
Desc: Clean up memory and check for unfreed memory.
*********************************************************************/
void f_memoryCleanup( void)
{
#ifdef FLM_DEBUG
	if (initMemTracking())
	{
		FLMUINT		uiId;
		F_MEM_HDR *	pHdr;

		f_mutexLock( gv_hMemTrackingMutex);
		for (uiId = 0; uiId < gv_uiMemTrackingPtrArraySize; uiId++)
		{
			if ((pHdr = (F_MEM_HDR *)gv_ppvMemTrackingPtrs [uiId]) != NULL)
			{
				logMemLeak( pHdr);
				freeMemTrackingInfo( TRUE, uiId + 1, pHdr->puiStack);
			}
		}

		// Free the memory pointer array.

		os_free( gv_ppvMemTrackingPtrs);
		gv_ppvMemTrackingPtrs = NULL;
		gv_uiMemTrackingPtrArraySize = 0;
		gv_uiNumMemPtrs = 0;

		f_mutexUnlock( gv_hMemTrackingMutex);

		// Free up the mutex.

		f_mutexDestroy( &gv_hMemTrackingMutex);

		// Reset to unitialized state.

		gv_uiInitThreadId = 0;
		gv_hMemTrackingMutex = F_MUTEX_NULL;
		gv_bMemTrackingInitialized = FALSE;
#ifdef FLM_WIN
		SymCleanup( gv_hMemProcess);
#endif
	}
#endif
}

/********************************************************************
Desc: Allocate Memory.
*********************************************************************/
RCODE FTKAPI f_allocImp(
	FLMUINT			uiSize,
	void **			ppvPtr,
	FLMBOOL			bAllocFromNewOp,
	const char *	pszFileName,
	int				iLineNumber)
{
	RCODE			rc = NE_FLM_OK;
	F_MEM_HDR *	pHdr;

#ifndef FLM_DEBUG
	F_UNREFERENCED_PARM( bAllocFromNewOp);
	F_UNREFERENCED_PARM( pszFileName);
	F_UNREFERENCED_PARM( iLineNumber);
#endif

	if( (pHdr = (F_MEM_HDR *)os_malloc( uiSize + sizeof( F_MEM_HDR) +
												F_PICKET_FENCE_SIZE)) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	pHdr->uiDataSize = uiSize;
	*ppvPtr = (void *)(&pHdr [1]);
	
#ifdef FLM_DEBUG
	pHdr->bAllocFromNewOp = bAllocFromNewOp;
	pHdr->iLineNumber = iLineNumber;
	pHdr->pszFileName = pszFileName;
	saveMemTrackingInfo( pHdr);

	#if F_PICKET_FENCE_SIZE
		f_memcpy( ((FLMBYTE *)(*ppvPtr)) + uiSize,
					F_PICKET_FENCE, F_PICKET_FENCE_SIZE);
	#endif
#endif

Exit:

	return( rc);
}

/********************************************************************
Desc: Allocate and initialize memory.
*********************************************************************/
RCODE FTKAPI f_callocImp(
	FLMUINT			uiSize,
	void **			ppvPtr,
	const char *	pszFileName,
	int				iLineNumber)
{
	RCODE			rc = NE_FLM_OK;
	F_MEM_HDR *	pHdr;

#ifndef FLM_DEBUG
	F_UNREFERENCED_PARM( pszFileName);
	F_UNREFERENCED_PARM( iLineNumber);
#endif
	if ((pHdr = (F_MEM_HDR *)os_malloc( uiSize + sizeof( F_MEM_HDR) +
											F_PICKET_FENCE_SIZE)) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	pHdr->uiDataSize = uiSize;
	*ppvPtr = (void *)(&pHdr [1]);
	f_memset( *ppvPtr, 0, uiSize);
#ifdef FLM_DEBUG
	pHdr->bAllocFromNewOp = FALSE;
	pHdr->iLineNumber = iLineNumber;
	pHdr->pszFileName = pszFileName;
	saveMemTrackingInfo( pHdr);

	#if F_PICKET_FENCE_SIZE

	f_memcpy( ((FLMBYTE *)(*ppvPtr)) + uiSize,
				F_PICKET_FENCE, F_PICKET_FENCE_SIZE);

	#endif

#endif
Exit:
	return( rc);
}

/********************************************************************
Desc: Reallocate memory.
*********************************************************************/
RCODE FTKAPI f_reallocImp(
	FLMUINT			uiSize,
	void **			ppvPtr,
	const char *	pszFileName,
	int				iLineNumber)
{
	RCODE			rc = NE_FLM_OK;
	F_MEM_HDR *	pNewHdr;
#ifdef FLM_DEBUG
	F_MEM_HDR *	pOldHdr;
	FLMUINT		uiOldAllocationId;
	FLMUINT *	puiOldStack;
#endif

	if (!(*ppvPtr))
	{
		rc = f_allocImp( uiSize, ppvPtr, FALSE, pszFileName, iLineNumber);
		goto Exit;
	}

#ifdef FLM_DEBUG
	pOldHdr = (F_MEM_HDR *)F_GET_ALLOC_PTR( *ppvPtr);

	#if F_PICKET_FENCE_SIZE

		// Verify the old picket fence
	
		if (f_memcmp( ((FLMBYTE *)(*ppvPtr)) + pOldHdr->uiDataSize,
							F_PICKET_FENCE, F_PICKET_FENCE_SIZE) != 0)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_MEM);
			goto Exit;
		}

	#endif

	// Cannot realloc memory that was allocated via a new operator

	if (pOldHdr->bAllocFromNewOp)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_MEM);
		goto Exit;
	}

	uiOldAllocationId = pOldHdr->uiAllocationId;
	puiOldStack = pOldHdr->puiStack;
#endif

	if ((pNewHdr = (F_MEM_HDR *)os_realloc( F_GET_ALLOC_PTR( *ppvPtr),
											uiSize + sizeof( F_MEM_HDR) +
											F_PICKET_FENCE_SIZE)) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	pNewHdr->uiDataSize = uiSize;
	*ppvPtr = (void *)(&pNewHdr [1]);
#ifdef FLM_DEBUG
	pNewHdr->bAllocFromNewOp = FALSE;
	pNewHdr->iLineNumber = iLineNumber;
	pNewHdr->pszFileName = pszFileName;
	if (pNewHdr != pOldHdr)
	{
		freeMemTrackingInfo( FALSE, uiOldAllocationId, puiOldStack);
		saveMemTrackingInfo( pNewHdr);
	}
	else
	{
		updateMemTrackingInfo( pNewHdr);
	}

	#if F_PICKET_FENCE_SIZE

	f_memcpy( ((FLMBYTE *)(*ppvPtr)) + uiSize,
				F_PICKET_FENCE, F_PICKET_FENCE_SIZE);

	#endif

#endif

Exit:
	return( rc);
}

/********************************************************************
Desc: Reallocate memory, and initialize the new part.
*********************************************************************/
RCODE FTKAPI f_recallocImp(
	FLMUINT			uiSize,
	void **			ppvPtr,
	const char *	pszFileName,
	int				iLineNumber)
{
	RCODE			rc = NE_FLM_OK;
	F_MEM_HDR *	pNewHdr;
	FLMUINT		uiOldSize;
#ifdef FLM_DEBUG
	F_MEM_HDR *	pOldHdr;
	FLMUINT		uiOldAllocationId;
	FLMUINT *	puiOldStack;
#endif

	if (!(*ppvPtr))
	{
		rc = f_callocImp( uiSize, ppvPtr, pszFileName, iLineNumber);
		goto Exit;
	}

#ifdef FLM_DEBUG
	pOldHdr = (F_MEM_HDR *)F_GET_ALLOC_PTR( *ppvPtr);

	#if F_PICKET_FENCE_SIZE

		// Verify the old picket fence
	
		if (f_memcmp( ((FLMBYTE *)(*ppvPtr)) + pOldHdr->uiDataSize,
							F_PICKET_FENCE, F_PICKET_FENCE_SIZE) != 0)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_MEM);
			goto Exit;
		}

	#endif

	// Cannot realloc memory that was allocated via a new operator

	if (pOldHdr->bAllocFromNewOp)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_MEM);
		goto Exit;
	}

	uiOldAllocationId = pOldHdr->uiAllocationId;
	puiOldStack = pOldHdr->puiStack;

#endif

	uiOldSize = F_GET_MEM_DATA_SIZE( *ppvPtr);
	if ((pNewHdr = (F_MEM_HDR *)os_realloc( F_GET_ALLOC_PTR( *ppvPtr),
											uiSize + sizeof( F_MEM_HDR) +
											F_PICKET_FENCE_SIZE)) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	pNewHdr->uiDataSize = uiSize;
	*ppvPtr = (void *)(&pNewHdr [1]);
	if (uiOldSize < uiSize)
	{
		f_memset( ((FLMBYTE *)(*ppvPtr)) + uiOldSize, 0,
					 uiSize - uiOldSize);
	}
#ifdef FLM_DEBUG
	pNewHdr->bAllocFromNewOp = FALSE;
	pNewHdr->iLineNumber = iLineNumber;
	pNewHdr->pszFileName = pszFileName;
	if (pNewHdr != pOldHdr)
	{
		freeMemTrackingInfo( FALSE, uiOldAllocationId, puiOldStack);
		saveMemTrackingInfo( pNewHdr);
	}
	else
	{
		updateMemTrackingInfo( pNewHdr);
	}

	#if F_PICKET_FENCE_SIZE

	f_memcpy( ((FLMBYTE *)(*ppvPtr)) + uiSize,
				F_PICKET_FENCE, F_PICKET_FENCE_SIZE);

	#endif

#endif
Exit:
	return( rc);
}

/********************************************************************
Desc: Free previously allocated memory.
*********************************************************************/
void FTKAPI f_freeImp(
	void **	ppvPtr,
	FLMBOOL	bFreeFromDeleteOp)
{
#ifndef FLM_DEBUG
	F_UNREFERENCED_PARM( bFreeFromDeleteOp);
#endif

	if (*ppvPtr)
	{
#ifdef FLM_DEBUG

		F_MEM_HDR *	pHdr = (F_MEM_HDR *)F_GET_ALLOC_PTR( *ppvPtr);

		if (pHdr->bAllocFromNewOp && !bFreeFromDeleteOp ||
			 !pHdr->bAllocFromNewOp && bFreeFromDeleteOp)
		{

			// Either trying to free memory using f_free when
			// allocated from new, or trying to free memory
			// using delete when allocated from f_alloc,
			// f_calloc, f_realloc, or f_recalloc.

			RC_UNEXPECTED_ASSERT( NE_FLM_MEM);
			return;
		}

		#if F_PICKET_FENCE_SIZE

			// Check the picket fence
	
			if (f_memcmp( ((FLMBYTE *)(*ppvPtr)) + pHdr->uiDataSize,
								F_PICKET_FENCE, F_PICKET_FENCE_SIZE) != 0)
			{
				RC_UNEXPECTED_ASSERT( NE_FLM_MEM);
			}

		#endif

		freeMemTrackingInfo( FALSE, pHdr->uiAllocationId, pHdr->puiStack);
#endif

		os_free( F_GET_ALLOC_PTR( *ppvPtr));
		*ppvPtr = NULL;
	}
}

/********************************************************************
Desc: Reset the stack information for an allocation.
*********************************************************************/
void f_resetStackInfoImp(
	void *			pvPtr,
	const char *	pszFileName,
	int				iLineNumber)
{
#ifdef FLM_DEBUG
	if (pvPtr)
	{

		F_MEM_HDR *	pHdr = (F_MEM_HDR *)F_GET_ALLOC_PTR( pvPtr);

		pHdr->iLineNumber = iLineNumber;
		pHdr->pszFileName = pszFileName;
		f_mutexLock( gv_hMemTrackingMutex);
		pHdr->uiAllocCnt = ++gv_uiAllocCnt;
		f_mutexUnlock( gv_hMemTrackingMutex);
		updateMemTrackingInfo( pHdr);
	}
#else
	F_UNREFERENCED_PARM( pvPtr);
	F_UNREFERENCED_PARM( pszFileName);
	F_UNREFERENCED_PARM( iLineNumber);
#endif
}

/************************************************************************
Desc:	Destructor
*************************************************************************/
F_Pool::~F_Pool()
{
	poolFree();
}

/************************************************************************
Desc:	Initialize a smart pool memory structure. A smart pool is one that
		will adjust it's block allocation size based on statistics it
		gathers within the POOL_STATS structure. For each pool that user
		wants to use smart memory management a global POOL_STATS structure
		should be declared. The POOL_STATS structure is used to track the
		total bytes allocated and determine what the correct pool block
		size should be.
*************************************************************************/
void F_Pool::smartPoolInit(
	POOL_STATS *	pPoolStats)
{
	m_pPoolStats = pPoolStats;
	if (m_pPoolStats && m_pPoolStats->uiCount)
	{
		setInitialSmartPoolBlockSize();
	}
	else
	{
		m_uiBlockSize = 2048;
	}
}

/****************************************************************************
Desc:	Allocates a block of memory from a memory pool.
Note:	If the number of bytes is more than the what is left in the
		current block then a new block will be allocated and the lbkl element
		of the PMS will be updated.
****************************************************************************/
RCODE FTKAPI F_Pool::poolAlloc(
	FLMUINT		uiSize,
	void **		ppvPtr)
{
	RCODE						rc = NE_FLM_OK;
	PoolMemoryBlock *		pBlock = m_pLastBlock;
	PoolMemoryBlock *		pOldLastBlock = pBlock;
	FLMBYTE *				pucFreePtr;
	FLMUINT					uiBlockSize;

	// Adjust the size to a machine word boundary
	// NOTE: ORed and ANDed 0x800.. & 0x7FFF to prevent partial
	// stalls on Netware

	if (uiSize & (FLM_ALLOC_ALIGN | 0x80000000))
	{
		uiSize = ((uiSize + FLM_ALLOC_ALIGN) & (~(FLM_ALLOC_ALIGN) & 0x7FFFFFFF));
	}

	// Check if room in block

	if (!pBlock || uiSize > pBlock->uiFreeSize)
	{

		// Check if previous block has space for allocation

		if (pBlock &&
			 pBlock->pPrevBlock &&
			 uiSize <= pBlock->pPrevBlock->uiFreeSize)
		{
			pBlock = pBlock->pPrevBlock;
			goto Exit;
		}

		// Not enough memory in block - allocate new block

		// Determine the block size:
		// 1) start with max of last block size, initial pool size, or alloc size
		// 2) if this is an extra block alloc then increase the size by 1/2
		// 3) adjust size to include block header

		uiBlockSize = (pBlock) ? pBlock->uiBlockSize : m_uiBlockSize;
		uiBlockSize = f_max( uiSize, uiBlockSize);

		if (pBlock &&
			 uiBlockSize == pBlock->uiBlockSize &&
			 uiBlockSize <= 32769)
		{
			uiBlockSize += uiBlockSize / 2;
		}

		// Add in extra bytes for block overhead

		uiBlockSize += sizeof( PoolMemoryBlock);

		if (RC_BAD( rc = f_alloc( uiBlockSize, &pBlock)))
		{
			goto Exit;
		}

		// Initialize the block elements

		pBlock->uiBlockSize = uiBlockSize;
		pBlock->uiFreeOffset = sizeof( PoolMemoryBlock);
		pBlock->uiFreeSize = uiBlockSize - sizeof( PoolMemoryBlock);

		// Link in newly allocated block

		m_pLastBlock = pBlock;
		pBlock->pPrevBlock = pOldLastBlock;
	}

Exit:

	if (RC_OK( rc))
	{
		pucFreePtr = (FLMBYTE *)pBlock;
		pucFreePtr += pBlock->uiFreeOffset;
		pBlock->uiFreeOffset += uiSize;
		pBlock->uiFreeSize -= uiSize;

		m_uiBytesAllocated += uiSize;
		*ppvPtr = (void *)pucFreePtr;
	}
	else
	{
		*ppvPtr = NULL;
	}
	return( rc);
}

/****************************************************************************
Desc:	Allocates a block of memory from a memory pool.
****************************************************************************/
RCODE FTKAPI F_Pool::poolCalloc(
  	FLMUINT		uiSize,
	void **		ppvPtr)
{
	RCODE	rc;

	if (RC_OK( rc = poolAlloc( uiSize, ppvPtr)))
	{
		f_memset( *ppvPtr, 0, uiSize);
	}
	return( rc);
}

/****************************************************************************
Desc : Releases all memory allocated to a pool.
Note : All memory allocated to the pool is returned to the operating system.
*****************************************************************************/
void FTKAPI F_Pool::poolFree( void)
{
	PoolMemoryBlock *	pBlock = m_pLastBlock;
	PoolMemoryBlock *	pPrevBlock;

	// Free all blocks in chain

	while (pBlock)
	{
		pPrevBlock = pBlock->pPrevBlock;
		f_free( &pBlock);
		pBlock = pPrevBlock;
	}

	m_pLastBlock = NULL;

	// For Smart pools, update pool statictics

	if (m_pPoolStats)
	{
		updateSmartPoolStats();
	}
}

/****************************************************************************
Desc:		Resets memory blocks allocated to a pool.
Note:		Will reset the free space in the first memory block, and if
			any extra blocks exist they will be freed (destroyed).
*****************************************************************************/
void FTKAPI F_Pool::poolReset(
	void *		pvMark,
	FLMBOOL		bReduceFirstBlock)
{
	PoolMemoryBlock *		pBlock = m_pLastBlock;
	PoolMemoryBlock *		pPrevBlock;

	if (!pBlock)
	{
		return;
	}

	// For Smart Pools update pool statictics

	if (m_pPoolStats)
	{
		updateSmartPoolStats();
	}

	if (pvMark)
	{
		freeToMark( pvMark);
		return;
	}

	// Free all blocks except last one in chain -- which is really
	// the first block allocated.  This will help us keep memory from
	// getting fragmented.

	while (pBlock->pPrevBlock)
	{
		pPrevBlock = pBlock->pPrevBlock;
		f_free( &pBlock);
		pBlock = pPrevBlock;
	}

	if (pBlock->uiBlockSize - sizeof( PoolMemoryBlock) > 
		 m_uiBlockSize && bReduceFirstBlock)
	{
		// The first block was not the default size, so free it

		f_free( &pBlock);
		m_pLastBlock = NULL;
	}
	else
	{
		// Reset the allocation pointers in the first block

		pBlock->uiFreeOffset  = sizeof( PoolMemoryBlock);
		pBlock->uiFreeSize = pBlock->uiBlockSize - sizeof( PoolMemoryBlock);
		m_pLastBlock = pBlock;
	}

	// On smart pools adjust the initial block size on pool resets

	if (m_pPoolStats)
	{
		setInitialSmartPoolBlockSize();
	}
}

/****************************************************************************
Desc:	Frees memory until the pvMark is found.
****************************************************************************/
void F_Pool::freeToMark(
	void *		pvMark)
{
	PoolMemoryBlock *		pBlock = m_pLastBlock;
	PoolMemoryBlock *		pPrevBlock;

	// Initialize pool to no blocks

	m_pLastBlock = NULL;
	while (pBlock)
	{
		pPrevBlock = pBlock->pPrevBlock;

		// Check for mark point

		if (PTR_IN_MBLK( pvMark, pBlock, pBlock->uiBlockSize))
		{
			FLMUINT  uiOldFreeOffset = pBlock->uiFreeOffset;

			// Reset uiFreeOffset and uiFreeSize variables

			pBlock->uiFreeOffset = (FLMUINT)((FLMBYTE *)pvMark -
														(FLMBYTE *)pBlock);
			pBlock->uiFreeSize = pBlock->uiBlockSize - pBlock->uiFreeOffset;

			// For Smart Pools deduct the bytes allocated since pool mark

			if (m_pPoolStats)
			{
				f_assert( uiOldFreeOffset >= pBlock->uiFreeOffset);
				m_uiBytesAllocated -= (uiOldFreeOffset - pBlock->uiFreeOffset);
			}

			break;
		}

		if (m_pPoolStats)
		{
			m_uiBytesAllocated -= (pBlock->uiFreeOffset - sizeof( PoolMemoryBlock));
		}

		f_free( &pBlock);
		pBlock = pPrevBlock;
	}

	if (pBlock)
	{
		m_pLastBlock = pBlock;
	}
}

/****************************************************************************
Desc:
****************************************************************************/
F_SlabManager::F_SlabManager()
{
	m_hMutex = F_MUTEX_NULL;
	m_uiTotalBytesAllocated = 0;
	m_pFirstInSlabList = NULL;
	m_pLastInSlabList = NULL;
	m_uiTotalSlabs = 0;
	m_uiAvailSlabs = 0;
	m_uiInUseSlabs = 0;
	m_uiPreallocSlabs = 0;
}

/****************************************************************************
Desc:
****************************************************************************/
F_SlabManager::~F_SlabManager()
{
	
	f_assert( !m_uiInUseSlabs);
	f_assert( m_uiAvailSlabs == m_uiTotalSlabs);

	freeAllSlabs();
	
	f_assert( !m_uiTotalBytesAllocated);
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_SlabManager::setup(
	FLMUINT 				uiPreallocSize)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiSysSlabSize = 0;
	FLMUINT		uiSlabSize = 64 * 1024;
	
	if( RC_BAD( rc = f_mutexCreate( &m_hMutex)))
	{
		goto Exit;
	}
	
	// Determine the slab size

#ifdef FLM_WIN
	{
		SYSTEM_INFO		sysInfo;

		GetSystemInfo( &sysInfo);
		uiSysSlabSize = sysInfo.dwAllocationGranularity;
	}
#endif

	if( !uiSysSlabSize)
	{
		uiSysSlabSize = uiSlabSize;
	}

	// Round the given slab size up to the closest operating 
	// system slab size so we don't waste any memory.

	if( uiSlabSize % uiSysSlabSize)
	{
		m_uiSlabSize = ((uiSlabSize / uiSysSlabSize) + 1) * uiSysSlabSize;
	}
	else
	{
		m_uiSlabSize = uiSlabSize;
	}
	
	// Pre-allocate the requested amount of memory from the system
	
	if( uiPreallocSize)
	{
		if( RC_BAD( rc = resize( uiPreallocSize, TRUE, NULL)))
		{
			goto Exit;
		}
	}
		
Exit:

	return( rc);
}
		
/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_SlabManager::resize(
	FLMUINT 			uiNumBytes,
	FLMBOOL			bPreallocate,
	FLMUINT *		puiActualSize)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiSlabsNeeded;
	void *			pSlab;
	FLMBOOL			bMutexLocked = FALSE;

	f_assertMutexNotLocked( m_hMutex);

	f_mutexLock( m_hMutex);
	bMutexLocked = TRUE;
	
	if( puiActualSize)
	{
		*puiActualSize = 0;
	}
	
	uiSlabsNeeded = (uiNumBytes / m_uiSlabSize) + 
						 ((uiNumBytes % m_uiSlabSize) ? 1 : 0);
						 
	if( !uiSlabsNeeded && !m_uiInUseSlabs)
	{
		freeAllSlabs();
	}
	else if( m_uiTotalSlabs > uiSlabsNeeded)
	{
		// Do the best we can to free slabs.  We can only get rid of
		// slabs that aren't in use.
		
		if( RC_BAD( rc = sortSlabList()))
		{
			goto Exit;
		}
		
		while( m_pLastInSlabList && m_uiTotalSlabs > uiSlabsNeeded)
		{
			pSlab = m_pLastInSlabList;
			if( (m_pLastInSlabList = ((SLABHEADER *)pSlab)->pPrev) != NULL)
			{
				((SLABHEADER *)m_pLastInSlabList)->pNext = NULL;
			}
			else
			{
				m_pFirstInSlabList = NULL;
			}
			
			releaseSlabToSystem( pSlab);
			
			f_assert( m_uiTotalSlabs);
			f_assert( m_uiInUseSlabs);
			f_assert( m_uiTotalBytesAllocated);
			
			m_uiAvailSlabs--;
			m_uiTotalSlabs--;
			m_uiTotalBytesAllocated -= m_uiSlabSize;
		}
	}
	else if( bPreallocate)
	{
		// Allocate the required number of slabs
		
		while( m_uiTotalSlabs < uiSlabsNeeded)
		{
			if( (pSlab = allocSlabFromSystem()) == NULL)
			{
				rc = RC_SET( NE_FLM_MEM);
				goto Exit;
			}
			
			// Touch every byte in the slab so that the operating system is
			// forced to immediately assign physical memory.
	
			f_memset( pSlab, 0, m_uiSlabSize);
			
			// Link the slab into the avail list
			
			if( m_pFirstInSlabList)
			{
				((SLABHEADER *)m_pFirstInSlabList)->pPrev = pSlab;
			}
			
			((SLABHEADER *)pSlab)->pNext = m_pFirstInSlabList;
			m_pFirstInSlabList = pSlab;
			
			if( !m_pLastInSlabList)
			{
				m_pLastInSlabList = pSlab;
			}

			m_uiTotalSlabs++;
			m_uiAvailSlabs++;
			m_uiTotalBytesAllocated += m_uiSlabSize;
		}
	}
	
	if( puiActualSize)
	{
		*puiActualSize = m_uiTotalSlabs * m_uiSlabSize;
	}
	
	if( bPreallocate)
	{
		m_uiPreallocSlabs = m_uiTotalSlabs;
	}
	else
	{
		m_uiPreallocSlabs = 0;
	}
	
Exit:

	if( RC_BAD( rc))
	{
		freeAllSlabs();
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
RCODE FTKAPI F_SlabManager::allocSlab(
	void **				ppSlab)
{
	RCODE			rc = NE_FLM_OK;
	FLMBOOL		bMutexLocked = FALSE;

	f_assertMutexNotLocked( m_hMutex);

	f_mutexLock( m_hMutex);
	bMutexLocked = TRUE;
	
	if( m_pFirstInSlabList)
	{
		*ppSlab = m_pFirstInSlabList;
		if( (m_pFirstInSlabList = 
				((SLABHEADER *)m_pFirstInSlabList)->pNext) != NULL)
		{
			((SLABHEADER *)m_pFirstInSlabList)->pPrev = NULL;
		}
		else
		{
			m_pLastInSlabList = NULL;
		}
		
		((SLABHEADER *)*ppSlab)->pNext = NULL;
		
		f_assert( m_uiAvailSlabs);
		m_uiAvailSlabs--;
		m_uiInUseSlabs++;
	}
	else
	{
		f_assert( !m_uiAvailSlabs);
		
		f_mutexUnlock( m_hMutex);
		bMutexLocked = FALSE;
		
		if( (*ppSlab = allocSlabFromSystem()) == NULL)
		{
			rc = RC_SET( NE_FLM_MEM);
			goto Exit;
		}
		
		f_mutexLock( m_hMutex);
		bMutexLocked = TRUE;
		
		m_uiTotalSlabs++;
		m_uiInUseSlabs++;
		m_uiTotalBytesAllocated += m_uiSlabSize;
	}
	
Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}
	
	return( rc);
}
		
/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI F_SlabManager::freeSlab(
	void **				ppSlab)
{
	f_assert( ppSlab && *ppSlab);
	f_assertMutexNotLocked( m_hMutex);

	f_mutexLock( m_hMutex);
	
	if( m_uiTotalSlabs <= m_uiPreallocSlabs)
	{
		((SLABHEADER *)*ppSlab)->pPrev = NULL;
		if( (((SLABHEADER *)*ppSlab)->pNext = m_pFirstInSlabList) != NULL)
		{
			((SLABHEADER *)m_pFirstInSlabList)->pPrev = *ppSlab;
		}
		else
		{
			m_pLastInSlabList = *ppSlab;
		}
		
		m_pFirstInSlabList = *ppSlab;
		*ppSlab = NULL;

		f_assert( m_uiInUseSlabs);		
		m_uiInUseSlabs--;
		m_uiAvailSlabs++;
	}
	else
	{
		f_mutexUnlock( m_hMutex);
		
		releaseSlabToSystem( *ppSlab);
		*ppSlab = NULL;
		
		f_mutexLock( m_hMutex);
	
		f_assert( m_uiTotalSlabs);
		f_assert( m_uiInUseSlabs);
		f_assert( m_uiTotalBytesAllocated);
		
		m_uiTotalSlabs--;
		m_uiInUseSlabs--;
		m_uiTotalBytesAllocated -= m_uiSlabSize;
	}
	
	f_mutexUnlock( m_hMutex);
}

/****************************************************************************
Desc:	Assumes that the mutex is locked
****************************************************************************/
void F_SlabManager::freeAllSlabs( void)
{
	void *			pNextSlab;
	SLABHEADER *	pSlabHeader;

	while( m_pFirstInSlabList)
	{
		pSlabHeader = (SLABHEADER *)m_pFirstInSlabList;
		pNextSlab = pSlabHeader->pNext;
		releaseSlabToSystem( m_pFirstInSlabList);
		m_pFirstInSlabList = pNextSlab;
		m_uiTotalSlabs--;
		m_uiAvailSlabs--;
		m_uiTotalBytesAllocated -= m_uiSlabSize;
	}
	
	f_assert( !m_uiAvailSlabs);
	m_pLastInSlabList = NULL;
}
	
/****************************************************************************
Desc:	Assumes that the mutex is locked
****************************************************************************/
void * F_SlabManager::allocSlabFromSystem( void)
{
	void *		pSlab;
	
#if defined( FLM_WIN)

	pSlab = VirtualAlloc( NULL,
		(DWORD)m_uiSlabSize, MEM_COMMIT, PAGE_READWRITE);
		
#elif defined( FLM_RING_ZERO_NLM)

	pSlab = Alloc( m_uiSlabSize, gv_lAllocRTag);

#elif defined( FLM_SOLARIS)

	pSlab = memalign( sysconf( _SC_PAGESIZE), m_uiSlabSize);
	
#elif defined( FLM_LINUX)

	if( posix_memalign( &pSlab, sysconf( _SC_PAGESIZE), m_uiSlabSize) != 0)
	{
		pSlab = NULL;
	}
	
#elif defined( FLM_UNIX)

	pSlab = valloc( m_uiSlabSize);
	
#else

	if( RC_BAD( f_alloc( m_uiSlabSize, &pSlab)))
	{
		pSlab = NULL;
	}
	
#endif

	return( pSlab);
}
	
/****************************************************************************
Desc:	Assumes that the mutex is locked
****************************************************************************/
void F_SlabManager::releaseSlabToSystem(
	void *		pSlab)
{
	f_assert( pSlab);
	
#ifdef FLM_WIN
	VirtualFree( pSlab, 0, MEM_RELEASE);
#elif defined( FLM_RING_ZERO_NLM)
	Free( pSlab);
#elif defined( FLM_UNIX)
	free( pSlab);
#else
	f_free( &pSlab);
#endif
}

/****************************************************************************
Desc:
****************************************************************************/
FLMINT FTKAPI F_SlabManager::slabAddrCompareFunc(
	void *		pvBuffer,
	FLMUINT		uiPos1,
	FLMUINT		uiPos2)
{
	void *		pSlab1 = (((void **)pvBuffer)[ uiPos1]);
	void *		pSlab2 = (((void **)pvBuffer)[ uiPos2]);

	f_assert( pSlab1 != pSlab2);

	if( pSlab1 < pSlab2)
	{
		return( -1);
	}

	return( 1);
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI F_SlabManager::slabAddrSwapFunc(
	void *		pvBuffer,
	FLMUINT		uiPos1,
	FLMUINT		uiPos2)
{
	void **		ppSlab1 = &(((void **)pvBuffer)[ uiPos1]);
	void **		ppSlab2 = &(((void **)pvBuffer)[ uiPos2]);
	void *		pTmp;

	pTmp = *ppSlab1;
	*ppSlab1 = *ppSlab2;
	*ppSlab2 = pTmp;
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE F_SlabManager::sortSlabList( void)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiLoop;
	void **			pSortBuf = NULL;
	FLMUINT			uiMaxSortEntries;
	FLMUINT			uiSortEntries = 0;
#define SMALL_SORT_BUF_SIZE 256
	void *			smallSortBuf[ SMALL_SORT_BUF_SIZE];
	void *			pCurSlab;
	void *			pPrevSib;

	f_assertMutexLocked( m_hMutex);

	if( m_uiAvailSlabs <= 1)
	{
		goto Exit;
	}

	uiMaxSortEntries = m_uiAvailSlabs;

	// Sort the avail list according to the starting memory addresses of the
	// slabs

	if( uiMaxSortEntries <= SMALL_SORT_BUF_SIZE)
	{
		pSortBuf = smallSortBuf;
	}
	else
	{
		if( RC_BAD( rc = f_alloc( uiMaxSortEntries * sizeof( void *), &pSortBuf)))
		{
			goto Exit;
		}
	}
	
	pCurSlab = m_pFirstInSlabList;

	while( pCurSlab)
	{
		f_assert( uiSortEntries != uiMaxSortEntries);
		pSortBuf[ uiSortEntries++] = pCurSlab;
		pCurSlab = ((SLABHEADER *)pCurSlab)->pNext;
	}
	
	f_assert( uiSortEntries == uiMaxSortEntries);

	// Quick sort

	f_assert( uiSortEntries);

	f_qsort( (FLMBYTE *)pSortBuf, 0, uiSortEntries - 1, 
		F_SlabManager::slabAddrCompareFunc,
		F_SlabManager::slabAddrSwapFunc);

	// Re-link the items in the list according to the new 
	// sort order

	m_pFirstInSlabList = NULL;
	m_pLastInSlabList = NULL;
	
	pCurSlab = NULL;
	pPrevSib = NULL;

	for( uiLoop = 0; uiLoop < uiSortEntries; uiLoop++)
	{
		pCurSlab = pSortBuf[ uiLoop];
		((SLABHEADER *)pCurSlab)->pNext = NULL;
		((SLABHEADER *)pCurSlab)->pPrev = NULL;

		if( pPrevSib)
		{
			((SLABHEADER *)pCurSlab)->pPrev = pPrevSib;
			((SLABHEADER *)pPrevSib)->pNext = pCurSlab;
		}
		else
		{
			m_pFirstInSlabList = pCurSlab;
		}

		pPrevSib = pCurSlab;
	}
	
	m_pLastInSlabList = pCurSlab;

Exit:

	if( pSortBuf && pSortBuf != smallSortBuf)
	{
		f_free( &pSortBuf);
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
F_FixedAlloc::F_FixedAlloc()
{
	m_pSlabManager = NULL;
	m_pFirstSlab = NULL;
	m_pLastSlab = NULL;
	m_pDefaultRelocator = NULL;
	m_pFirstSlabWithAvailCells = NULL;
	m_pLastSlabWithAvailCells = NULL;
	m_uiSlabsWithAvailCells = 0;
	m_bAvailListSorted = TRUE;
	m_uiTotalFreeCells = 0;
	m_uiSlabSize = 0;
	m_pUsageStats = NULL;
	m_puiTotalBytesAllocated = NULL;
	m_hMutex = F_MUTEX_NULL;
}

/****************************************************************************
Desc:	Destructor for F_FixedAlloc.  checks for memory leaks, and
		frees all memory in use.
****************************************************************************/
F_FixedAlloc::~F_FixedAlloc()
{
#ifdef FLM_DEBUG
	testForLeaks();
#endif

	freeAll();
	
	if( m_pSlabManager)
	{
		m_pSlabManager->Release();
	}
	
	if( m_pDefaultRelocator)
	{
		m_pDefaultRelocator->Release();
	}
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}
}

/****************************************************************************
Desc:	Setup method for any setup that can fail 
****************************************************************************/
RCODE F_FixedAlloc::setup(
	FLMBOOL					bMultiThreaded,
	IF_SlabManager *		pSlabManager,
	IF_Relocator *			pDefaultRelocator,
	FLMUINT					uiCellSize,
	FLM_SLAB_USAGE *		pUsageStats,
	FLMUINT *				puiTotalBytesAllocated)
{
	RCODE			rc = NE_FLM_OK;

	f_assert( pSlabManager);
	f_assert( uiCellSize);
	f_assert( pUsageStats);
	
	if( bMultiThreaded)
	{
		if( RC_BAD( rc = f_mutexCreate( &m_hMutex)))
		{
			goto Exit;
		}
	}
	
	m_pUsageStats = pUsageStats;
	m_puiTotalBytesAllocated = puiTotalBytesAllocated;
	
	m_pSlabManager = pSlabManager;
	m_pSlabManager->AddRef();
	
	if( pDefaultRelocator)
	{
		m_pDefaultRelocator = pDefaultRelocator;
		m_pDefaultRelocator->AddRef();
	}
	
	m_uiCellSize = uiCellSize;
	m_uiSlabSize = m_pSlabManager->getSlabSize();

	// Get the alloc-aligned versions of all the sizes

	m_uiSlabHeaderSize = getAllocAlignedSize( sizeof( FIXEDSLAB));
	
	if (pDefaultRelocator)
	{
		m_uiCellHeaderSize = getAllocAlignedSize( sizeof( CELLHEADER));
	}
	else
	{
		m_uiCellHeaderSize = getAllocAlignedSize( sizeof( CELLHEADER2));
	}
	
	m_uiCellSize = getAllocAlignedSize( m_uiCellSize);

	// Ensure that there's enough space for our overhead

	f_assert( m_uiCellSize >= sizeof( CELLAVAILNEXT));

	m_uiSizeOfCellAndHeader = m_uiCellHeaderSize + m_uiCellSize;

	m_uiCellsPerSlab = 
		(m_uiSlabSize - m_uiSlabHeaderSize) / m_uiSizeOfCellAndHeader;

	f_assert( m_uiCellsPerSlab);
	f_assert( m_uiCellsPerSlab <= FLM_MAX_UINT16);
	f_assert( (m_uiCellsPerSlab * m_uiCellSize) < m_uiSlabSize);
	
Exit:
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void * FTKAPI F_FixedAlloc::allocCell(
	IF_Relocator *		pRelocator,
	void *				pvInitialData,
	FLMUINT				uiDataSize)
{
	void *				pvCell;
	FLMBOOL				bMutexLocked = FALSE;
	
	f_assert( pRelocator || m_pDefaultRelocator);

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexNotLocked( m_hMutex);
		f_mutexLock( m_hMutex);
		bMutexLocked = TRUE;
	}
	
	if( (pvCell = getCell( pRelocator)) == NULL)
	{
		goto Exit;
	}
	
	if( uiDataSize == sizeof( FLMUINT *))
	{
		*((FLMUINT *)pvCell) = *((FLMUINT *)pvInitialData); 
	}
	else if( uiDataSize)
	{
		f_memcpy( pvCell, pvInitialData, uiDataSize);
	}
	
Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}
	
	return( pvCell);
}

/****************************************************************************
Desc:	Public method to free a cell of memory back to the system. 
****************************************************************************/
void * FTKAPI F_FixedAlloc::allocCell(
	IF_Relocator *				pRelocator,
	F_ALLOC_INIT_FUNC			fnAllocInit)
{
	void *						pvCell;
	FLMBOOL						bMutexLocked = FALSE;

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexNotLocked( m_hMutex);
		f_mutexLock( m_hMutex);
		bMutexLocked = TRUE;
	}
	
	if( (pvCell = getCell( pRelocator)) == NULL)
	{
		goto Exit;
	}
	
	if( pvCell && fnAllocInit)
	{
		fnAllocInit( pvCell, m_uiCellSize);
	}

Exit:
	
	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}
	
	return( pvCell);
}
	
/****************************************************************************
Desc:	Private, internal method to fetch a cell
****************************************************************************/
void * F_FixedAlloc::getCell(
	IF_Relocator *		pRelocator)
{
	FIXEDSLAB *		pSlab = NULL;
	FLMBYTE *		pCell = NULL;
	CELLHEADER *	pHeader;

#ifdef FLM_DEBUG
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexLocked( m_hMutex);
	}
#endif

	// If there's a slab that has an avail cell, that one gets priority

	if( (pSlab = m_pFirstSlabWithAvailCells) != NULL)
	{
		f_assert( pSlab->ui16AvailCellCount <= m_uiTotalFreeCells);
		f_assert( m_uiTotalFreeCells);
		f_assert( pSlab->ui16AllocatedCells < m_uiCellsPerSlab);

		pCell = m_pFirstSlabWithAvailCells->pLocalAvailCellListHead;
		f_assert( pCell);

		pHeader = (CELLHEADER *)((FLMBYTE *)pCell - m_uiCellHeaderSize);
		pSlab->ui16AllocatedCells++;
		pSlab->ui16AvailCellCount--;
		m_uiTotalFreeCells--;
		
		// An avail cell holds as its contents the next pointer in the avail chain.
		// Avail chains do not span slabs.

		pSlab->pLocalAvailCellListHead = ((CELLAVAILNEXT *)pCell)->pNextInList;

		// If there are no other avail cells in this slab at this point,
		// then we need to unlink the slab from the
		// slabs-with-avail-cells list, headed by m_pFirstSlabWithAvailCells

		if( !pSlab->pLocalAvailCellListHead)
		{
			// Save a copy of the slab we're going to unlink

			FIXEDSLAB * pSlabToUnlink = pSlab;

			// Need to keep the NULLNESS of the content of the cell consistent
			// with the slab's ui16AvailCellCount being equal to 0

			f_assert( !pSlabToUnlink->ui16AvailCellCount);

			// There can't be a pPrevSlabWithAvailCells since
			// we're positioned to the first one

			f_assert( !pSlabToUnlink->pPrevSlabWithAvailCells);				

			// Update m_pFirstSlabWithAvailCells to point to the next one

			if( (m_pFirstSlabWithAvailCells =
				pSlabToUnlink->pNextSlabWithAvailCells) == NULL)
			{
				f_assert( m_pLastSlabWithAvailCells == pSlabToUnlink);
				m_pLastSlabWithAvailCells = NULL;
			}

			// Unlink from slabs-with-avail-cells list

			if( pSlabToUnlink->pNextSlabWithAvailCells)
			{
				pSlabToUnlink->pNextSlabWithAvailCells->pPrevSlabWithAvailCells =
					pSlabToUnlink->pPrevSlabWithAvailCells;
				pSlabToUnlink->pNextSlabWithAvailCells = NULL;
			}

			// Decrement the slab count

			f_assert( m_uiSlabsWithAvailCells);
			m_uiSlabsWithAvailCells--;
		}
	}
	else
	{
		// If our m_pFirstSlab is completely full, or there is no
		// m_pFirstSlab, it is time to allocate a new slab

		if( !m_pFirstSlab ||
			(m_pFirstSlab->ui16NextNeverUsedCell == m_uiCellsPerSlab))
		{
			FIXEDSLAB *		pNewSlab;

			if( (pNewSlab = getAnotherSlab()) == NULL)
			{
				goto Exit;
			}

			if( m_pFirstSlab)
			{
				pNewSlab->pNext = m_pFirstSlab;
				m_pFirstSlab->pPrev = pNewSlab;
			}
			else
			{
				m_pLastSlab = pNewSlab;
			}

			m_pFirstSlab = pNewSlab;
		}

		pSlab = m_pFirstSlab;
		pSlab->ui16AllocatedCells++;
		
		pHeader = (CELLHEADER *)
				((FLMBYTE *)pSlab + m_uiSlabHeaderSize +
					(m_uiSizeOfCellAndHeader * m_pFirstSlab->ui16NextNeverUsedCell));

		pCell = ((FLMBYTE *)pHeader + m_uiCellHeaderSize);		
		m_pFirstSlab->ui16NextNeverUsedCell++;
	}

	pHeader->pContainingSlab = pSlab;

#ifdef FLM_DEBUG
	if (gv_bTrackLeaks && gv_bStackWalk)
	{
		pHeader->puiStack = memWalkStack();
	}
	else
	{
		pHeader->puiStack = NULL;
	}
#endif
	if (!m_pDefaultRelocator)
	{
		((CELLHEADER2 *)((void *)pHeader))->pRelocator = pRelocator;
	}

	if( m_pUsageStats)
	{
		m_pUsageStats->ui64AllocatedCells++;
	}
	
Exit:

	return( pCell);
}

/****************************************************************************
Desc:	Public method to free a cell of memory back to the system. 
****************************************************************************/
void FTKAPI F_FixedAlloc::freeCell( 
	void *		ptr)
{
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexNotLocked( m_hMutex);
		f_mutexLock( m_hMutex);
	}
	
	freeCell( ptr, FALSE, NULL);
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( m_hMutex);
	}
}
	
/****************************************************************************
Desc:	Public method to free a cell of memory back to the system. 
****************************************************************************/
void F_FixedAlloc::freeCell(
	void *		pCell,
	FLMBOOL		bFreeIfEmpty,
	FLMBOOL *	pbFreedSlab)
{
	CELLAVAILNEXT *	pCellContents;
	CELLHEADER *		pHeader;
	FIXEDSLAB *			pSlab;

#ifdef FLM_DEBUG
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexLocked( m_hMutex);
	}
#endif

	if( pbFreedSlab)
	{
		*pbFreedSlab = FALSE;
	}

	if( !pCell)
	{
		return;
	}
	
	pCellContents = (CELLAVAILNEXT *)pCell;
	pHeader = (CELLHEADER *)(((FLMBYTE *)pCell) - m_uiCellHeaderSize);
	pSlab = pHeader->pContainingSlab;

	// Memory corruption detected!

	if( !pSlab || pSlab->pvAllocator != (void *)this)
	{
		f_assert( 0);
		goto Exit;
	}

	pHeader->pContainingSlab = NULL;
	
#ifdef FLM_DEBUG
	if( pHeader->puiStack)
	{
		os_free( pHeader->puiStack);
		pHeader->puiStack = NULL;
	}
#endif

	// Should always be set on a free
	
	f_assert( m_pFirstSlab);
	
	// Add the cell to the pSlab's free list

	pCellContents->pNextInList = pSlab->pLocalAvailCellListHead;

#ifdef FLM_DEBUG
	// Write out a string that's easy to see in memory when debugging

	f_strcpy( (char *)pCellContents->szDebugPattern, "FREECELL");
#endif

	f_assert( pCell);
	pSlab->pLocalAvailCellListHead = (FLMBYTE *)pCell;
	pSlab->ui16AvailCellCount++;

	f_assert( pSlab->ui16AllocatedCells);
	pSlab->ui16AllocatedCells--;

	// If there's no chain, make this one the first

	if( !m_pFirstSlabWithAvailCells)
	{
		m_pFirstSlabWithAvailCells = pSlab;
		m_pLastSlabWithAvailCells = pSlab;
		f_assert( !pSlab->pNextSlabWithAvailCells);
		f_assert( !pSlab->pPrevSlabWithAvailCells);
		m_uiSlabsWithAvailCells++;
		m_bAvailListSorted = TRUE;
	}
	else if( pSlab->ui16AvailCellCount == 1)
	{
		// This item is not linked in to the chain, so link it in

		if( m_bAvailListSorted && pSlab > m_pFirstSlabWithAvailCells)
		{
			m_bAvailListSorted = FALSE;
		}

		pSlab->pNextSlabWithAvailCells = m_pFirstSlabWithAvailCells;
		pSlab->pPrevSlabWithAvailCells = NULL;
		m_pFirstSlabWithAvailCells->pPrevSlabWithAvailCells = pSlab;
		m_pFirstSlabWithAvailCells = pSlab;
		m_uiSlabsWithAvailCells++;
	}

	// Adjust counter, because the cell is now considered free

	m_uiTotalFreeCells++;

	// If this slab is now totally avail

	if( pSlab->ui16AvailCellCount == m_uiCellsPerSlab)
	{
		f_assert( !pSlab->ui16AllocatedCells);

		// If we have met our threshold for being able to free a slab

		if( m_uiTotalFreeCells >= m_uiCellsPerSlab || bFreeIfEmpty)
		{
			freeSlab( pSlab);

			if( pbFreedSlab)
			{
				*pbFreedSlab = TRUE;
			}
		}
		else if( pSlab != m_pFirstSlabWithAvailCells)
		{
			// Link the slab to the front of the avail list so that
			// it can be freed quickly at some point in the future

			if( pSlab->pPrevSlabWithAvailCells)
			{
				pSlab->pPrevSlabWithAvailCells->pNextSlabWithAvailCells =
					pSlab->pNextSlabWithAvailCells;
			}

			if( pSlab->pNextSlabWithAvailCells)
			{
				pSlab->pNextSlabWithAvailCells->pPrevSlabWithAvailCells =
					pSlab->pPrevSlabWithAvailCells;
			}
			else
			{
				f_assert( m_pLastSlabWithAvailCells == pSlab);
				m_pLastSlabWithAvailCells = pSlab->pPrevSlabWithAvailCells;
			}

			if( m_pFirstSlabWithAvailCells)
			{
				m_pFirstSlabWithAvailCells->pPrevSlabWithAvailCells = pSlab;
			}

			pSlab->pPrevSlabWithAvailCells = NULL;
			pSlab->pNextSlabWithAvailCells = m_pFirstSlabWithAvailCells;
			m_pFirstSlabWithAvailCells = pSlab;
		}
	}
	
	if( m_pUsageStats)
	{
		m_pUsageStats->ui64AllocatedCells--;
	}

Exit:

	return;
}

/****************************************************************************
Desc:	Grabs another slab of memory from the operating system
****************************************************************************/
FIXEDSLAB * F_FixedAlloc::getAnotherSlab( void)
{
	FIXEDSLAB *	pSlab = NULL;
	
#ifdef FLM_DEBUG
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexLocked( m_hMutex);
	}
#endif

	if( RC_BAD( m_pSlabManager->allocSlab( (void **)&pSlab)))
	{
		goto Exit;
	}

	// Initialize the slab header fields

	f_memset( pSlab, 0, sizeof( FIXEDSLAB));
	pSlab->pvAllocator = (void *)this;
	
	if( m_pUsageStats)
	{
		m_pUsageStats->ui64Slabs++;
	}
	
	if( m_puiTotalBytesAllocated)
	{
		(*m_puiTotalBytesAllocated) += m_uiSlabSize;
	}

Exit:
	
	return( pSlab);
}

/****************************************************************************
Desc:	Private internal method to free an unused empty slab back to the OS.
****************************************************************************/
void F_FixedAlloc::freeSlab(
	FIXEDSLAB *		pSlab)
{
#ifdef FLM_DEBUG
	CELLAVAILNEXT *		pAvailNext = NULL;
	FLMUINT32				ui32AvailCount = 0;
#endif

	f_assert( pSlab);
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexLocked( m_hMutex);
	}

	// Memory corruption detected!

	if( pSlab->ui16AllocatedCells || pSlab->pvAllocator != this)
	{
		f_assert( 0);
		return;
	}

#ifdef FLM_DEBUG
	// Walk the avail chain as a sanity check

	pAvailNext = (CELLAVAILNEXT *)pSlab->pLocalAvailCellListHead;
	while( pAvailNext)
	{
		ui32AvailCount++;
		pAvailNext = (CELLAVAILNEXT *)pAvailNext->pNextInList;
	}

	f_assert( pSlab->ui16AvailCellCount == ui32AvailCount);
	f_assert( pSlab->ui16NextNeverUsedCell >= ui32AvailCount);
#endif
	
	// Unlink from all-slabs-list

	if( pSlab->pNext)
	{
		pSlab->pNext->pPrev = pSlab->pPrev;
	}
	else
	{
		m_pLastSlab = pSlab->pPrev;
	}

	if( pSlab->pPrev)
	{
		pSlab->pPrev->pNext = pSlab->pNext;
	}
	else
	{
		m_pFirstSlab = pSlab->pNext;
	}

	// Unlink from slabs-with-avail-cells list

	if( pSlab->pNextSlabWithAvailCells)
	{
		pSlab->pNextSlabWithAvailCells->pPrevSlabWithAvailCells =
			pSlab->pPrevSlabWithAvailCells;
	}
	else
	{
		m_pLastSlabWithAvailCells = pSlab->pPrevSlabWithAvailCells;
	}

	if( pSlab->pPrevSlabWithAvailCells)
	{
		pSlab->pPrevSlabWithAvailCells->pNextSlabWithAvailCells =
			pSlab->pNextSlabWithAvailCells;
	}
	else
	{
		m_pFirstSlabWithAvailCells = pSlab->pNextSlabWithAvailCells;
	}

	f_assert( m_uiSlabsWithAvailCells);
	m_uiSlabsWithAvailCells--;
	f_assert( m_uiTotalFreeCells >= pSlab->ui16AvailCellCount);
	m_uiTotalFreeCells -= pSlab->ui16AvailCellCount;
	m_pSlabManager->freeSlab( (void **)&pSlab);
	
	if( m_pUsageStats)
	{
		f_assert( m_pUsageStats->ui64Slabs);
		m_pUsageStats->ui64Slabs--;
	}
	
	if( m_puiTotalBytesAllocated)
	{
		f_assert( (*m_puiTotalBytesAllocated) >= m_uiSlabSize);
		(*m_puiTotalBytesAllocated) -= m_uiSlabSize;
	}
}

/****************************************************************************
Desc:	Public method to free all the memory in the system.  
****************************************************************************/
void FTKAPI F_FixedAlloc::freeAll( void)
{
	FIXEDSLAB *		pFreeMe;

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexNotLocked( m_hMutex);
		f_mutexLock( m_hMutex);
	}

	while( m_pFirstSlab)
	{
		pFreeMe = m_pFirstSlab;
		m_pFirstSlab = m_pFirstSlab->pNext;
		freeSlab( pFreeMe);
	}

	f_assert( !m_uiTotalFreeCells);

	m_pFirstSlab = NULL;
	m_pLastSlab = NULL;
	m_pFirstSlabWithAvailCells = NULL;
	m_pLastSlabWithAvailCells = NULL;
	m_uiSlabsWithAvailCells = 0;
	m_bAvailListSorted = TRUE;
	m_uiTotalFreeCells = 0;

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( m_hMutex);
	}
}

/****************************************************************************
Desc:		If a relocation callback function has been registered, and memory 
			can be compressed, the avail list will be compressed
****************************************************************************/ 
void F_FixedAlloc::defragmentMemory( void)
{
	FIXEDSLAB *		pCurSlab;
	FIXEDSLAB *		pPrevSib;
	CELLHEADER *	pCellHeader;
	FLMBOOL			bSlabFreed;
	FLMBYTE *		pucOriginal;
	FLMBYTE *		pucReloc = NULL;
	FLMUINT			uiLoop;
	FIXEDSLAB **	pSortBuf = NULL;
	FLMUINT			uiMaxSortEntries;
	FLMUINT			uiSortEntries = 0;
#define SMALL_SORT_BUF_SIZE 256
	FIXEDSLAB *		smallSortBuf[ SMALL_SORT_BUF_SIZE];
	FLMBOOL			bMutexLocked = FALSE;

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexNotLocked( m_hMutex);
		f_mutexLock( m_hMutex);
		bMutexLocked = TRUE;
	}
	
	if( m_uiTotalFreeCells < m_uiCellsPerSlab)
	{
		goto Exit;
	}

	uiMaxSortEntries = m_uiSlabsWithAvailCells;

	// Re-sort the slabs in the avail list according to
	// their memory addresses to help reduce logical fragmentation

	if( !m_bAvailListSorted && uiMaxSortEntries > 1)
	{
		if( uiMaxSortEntries <= SMALL_SORT_BUF_SIZE)
		{
			pSortBuf = smallSortBuf;
		}
		else
		{
			if( RC_BAD( f_alloc( uiMaxSortEntries * sizeof( FIXEDSLAB *),
				&pSortBuf)))
			{
				goto Exit;
			}
		}

		pCurSlab = m_pFirstSlabWithAvailCells;

		while( pCurSlab)
		{
			f_assert( uiSortEntries != uiMaxSortEntries);
			pSortBuf[ uiSortEntries++] = pCurSlab;
			pCurSlab = pCurSlab->pNextSlabWithAvailCells;
		}

		// Quick sort

		f_assert( uiSortEntries);

		f_qsort( (FLMBYTE *)pSortBuf, 0, uiSortEntries - 1, 
			F_FixedAlloc::slabAddrCompareFunc,
			F_FixedAlloc::slabAddrSwapFunc);

		// Re-link the items in the list according to the new 
		// sort order

		m_pFirstSlabWithAvailCells = NULL;
		m_pLastSlabWithAvailCells = NULL;

		pCurSlab = NULL;
		pPrevSib = NULL;

		for( uiLoop = 0; uiLoop < uiSortEntries; uiLoop++)
		{
			pCurSlab = pSortBuf[ uiLoop];
			pCurSlab->pNextSlabWithAvailCells = NULL;
			pCurSlab->pPrevSlabWithAvailCells = NULL;

			if( pPrevSib)
			{
				pCurSlab->pPrevSlabWithAvailCells = pPrevSib;
				pPrevSib->pNextSlabWithAvailCells = pCurSlab;
			}
			else
			{
				m_pFirstSlabWithAvailCells = pCurSlab;
			}

			pPrevSib = pCurSlab;
		}

		m_pLastSlabWithAvailCells = pCurSlab;
		m_bAvailListSorted = TRUE;
	}

	// Process the avail list (which should be sorted unless
	// we are too low on memory)

	pCurSlab = m_pLastSlabWithAvailCells;

	while( pCurSlab)
	{
		if( m_uiTotalFreeCells < m_uiCellsPerSlab)
		{
			// No need to continue ... we aren't above the
			// free cell threshold

			goto Exit;
		}

		pPrevSib = pCurSlab->pPrevSlabWithAvailCells;

		if( pCurSlab == m_pFirstSlabWithAvailCells ||
				!pCurSlab->ui16AvailCellCount)
		{
			// We've either hit the beginning of the avail list or
			// the slab that we are now positioned on has been
			// removed from the avail list.  In either case,
			// we are done.

			break;
		}

		if( pCurSlab->ui16AvailCellCount == m_uiCellsPerSlab ||
			pCurSlab->ui16NextNeverUsedCell == pCurSlab->ui16AvailCellCount)
		{
			freeSlab( pCurSlab);
			pCurSlab = pPrevSib;
			continue;
		}

		for( uiLoop = 0; uiLoop < pCurSlab->ui16NextNeverUsedCell &&
			pCurSlab != m_pFirstSlabWithAvailCells &&
			m_uiTotalFreeCells >= m_uiCellsPerSlab; uiLoop++)
		{
			IF_Relocator *	pRelocator;

			pCellHeader = (CELLHEADER *)
				((FLMBYTE *)pCurSlab + m_uiSlabHeaderSize +
					(uiLoop * m_uiSizeOfCellAndHeader));
			if ((pRelocator = m_pDefaultRelocator) == NULL)
			{
				pRelocator = ((CELLHEADER2 *)((void *)pCellHeader))->pRelocator;
			}

			if( pCellHeader->pContainingSlab)
			{

				// If pContainingSlab is non-NULL, the cell is currently allocated

				f_assert( pCellHeader->pContainingSlab == pCurSlab);

				pucOriginal = ((FLMBYTE *)pCellHeader + m_uiCellHeaderSize);

				if( pRelocator->canRelocate( pucOriginal))
				{
					if( (pucReloc = (FLMBYTE *)getCell( pRelocator)) == NULL)
					{
						goto Exit;
					}

					f_memcpy( pucReloc, pucOriginal, m_uiCellSize);
					pRelocator->relocate( pucOriginal, pucReloc);

					freeCell( pucOriginal, TRUE, &bSlabFreed);
					
					if( bSlabFreed)
					{
						break;
					}
				}
			}
		}

		pCurSlab = pPrevSib;
	}

Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	if( pSortBuf && pSortBuf != smallSortBuf)
	{
		f_free( &pSortBuf);
	}
}

/****************************************************************************
Desc:		
****************************************************************************/ 
void FTKAPI F_FixedAlloc::freeUnused( void)
{
	FIXEDSLAB *		pSlab;

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexNotLocked( m_hMutex);
		f_mutexLock( m_hMutex);
	}

	if( (pSlab = m_pFirstSlabWithAvailCells) != NULL &&
		!pSlab->ui16AllocatedCells)
	{
		freeSlab( pSlab);
	}

	if( (pSlab = m_pFirstSlab) != NULL &&
		!pSlab->ui16AllocatedCells)
	{
		freeSlab( pSlab);
	}

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( m_hMutex);
	}
}

/****************************************************************************
Desc:	Debug method to do mem leak testing.  Any cells allocated via
		allocCell but not freed via freeCell() will be triggered here.
****************************************************************************/ 
#ifdef FLM_DEBUG
void F_FixedAlloc::testForLeaks( void)
{
	FIXEDSLAB *		pSlabRover = m_pFirstSlab;
	CELLHEADER *	pHeader;
	FLMUINT			uiLoop;
	F_MEM_HDR		memHeader;

	// Test for leaks

	while( pSlabRover)
	{
		for( uiLoop = 0; uiLoop < pSlabRover->ui16NextNeverUsedCell; uiLoop++)
		{
			pHeader = (CELLHEADER *)
				((FLMBYTE *)pSlabRover + m_uiSlabHeaderSize +
					(uiLoop * m_uiSizeOfCellAndHeader));

			// Nonzero here means we have a leak

			if( pHeader->pContainingSlab)
			{
				// We have a leak, so let's call logMemLeak with the
				// appropriate header passed in

				f_memset( &memHeader, 0, sizeof( F_MEM_HDR));
				memHeader.uiDataSize = m_uiCellSize;
				memHeader.puiStack = pHeader->puiStack;
				logMemLeak( &memHeader);
			}
		}

		pSlabRover = pSlabRover->pNext;
	}
}
#endif

/****************************************************************************
Desc:
****************************************************************************/ 
F_BufferAlloc::~F_BufferAlloc()
{
	FLMUINT	uiLoop;

	for (uiLoop = 0; uiLoop < NUM_BUF_ALLOCATORS; uiLoop++)
	{
		if( m_ppAllocators[ uiLoop])
		{
			m_ppAllocators[ uiLoop]->Release();
			m_ppAllocators[ uiLoop] = NULL;
		}
	}
	
	if( m_pSlabManager)
	{
		m_pSlabManager->Release();
	}
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}
}
	
/****************************************************************************
Desc:
****************************************************************************/ 
RCODE F_BufferAlloc::setup(
	FLMBOOL					bMultiThreaded,
	IF_SlabManager *		pSlabManager,
	IF_Relocator *			pDefaultRelocator,
	FLM_SLAB_USAGE *		pUsageStats,
	FLMUINT *				puiTotalBytesAllocated)
{
	RCODE						rc = NE_FLM_OK;
	FLMUINT					uiLoop;
	FLMUINT					uiSize;
	
	f_assert( pSlabManager);
	
	if( bMultiThreaded)
	{
		if( RC_BAD( rc = f_mutexCreate( &m_hMutex)))
		{
			goto Exit;
		}
	}
	
	m_pSlabManager = pSlabManager;
	m_pSlabManager->AddRef();
	
	for( uiLoop = 0; uiLoop < NUM_BUF_ALLOCATORS; uiLoop++)
	{
		if( (m_ppAllocators[ uiLoop] = f_new F_FixedAlloc) == NULL)
		{
			rc = RC_SET( NE_FLM_MEM);
			goto Exit;
		}
		
		switch (uiLoop)
		{
			case 0: 
				uiSize = CELL_SIZE_0;
				break;
			case 1: 
				uiSize = CELL_SIZE_1;
				break;
			case 2: 
				uiSize = CELL_SIZE_2;
				break;
			case 3: 
				uiSize = CELL_SIZE_3;
				break;
			case 4: 
				uiSize = CELL_SIZE_4;
				break;
			case 5: 
				uiSize = CELL_SIZE_5;
				break;
			case 6: 
				uiSize = CELL_SIZE_6;
				break;
			case 7: 
				uiSize = CELL_SIZE_7;
				break;
			case 8:
				uiSize = CELL_SIZE_8;
				break;
			case 9: 
				uiSize = CELL_SIZE_9;
				break;
			case 10:
				uiSize = CELL_SIZE_10;
				break;
			case 11: 
				uiSize = CELL_SIZE_11;
				break;
			case 12: 
				uiSize = CELL_SIZE_12;
				break;
			case 13: 
				uiSize = CELL_SIZE_13;
				break;
			case 14: 
				uiSize = CELL_SIZE_14;
				break;
			case 15: 
				uiSize = CELL_SIZE_15;
				break;
			case 16: 
				uiSize = CELL_SIZE_16;
				break;
			case 17: 
				uiSize = CELL_SIZE_17;
				break;
			case 18: 
				uiSize = CELL_SIZE_18;
				break;
			case 19: 
				uiSize = CELL_SIZE_19;
				break;
			case 20: 
				uiSize = CELL_SIZE_20;
				break;
			case 21: 
				uiSize = CELL_SIZE_21;
				break;
			default:
				uiSize = 0;
				rc = RC_SET_AND_ASSERT( NE_FLM_NOT_IMPLEMENTED);
				goto Exit;
		}

		if (RC_BAD( rc = m_ppAllocators[ uiLoop]->setup( FALSE,
			pSlabManager, pDefaultRelocator, uiSize, 
			pUsageStats, puiTotalBytesAllocated)))
		{
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/ 
RCODE FTKAPI F_BufferAlloc::allocBuf(
	IF_Relocator *		pRelocator,
	FLMUINT				uiSize,
	void *				pvInitialData,
	FLMUINT				uiDataSize,
	FLMBYTE **			ppucBuffer,
	FLMBOOL *			pbAllocatedOnHeap)
{
	RCODE					rc = NE_FLM_OK;
	IF_FixedAlloc *	pAllocator = getAllocator( uiSize);
	FLMBOOL				bMutexLocked = FALSE;

	if( pbAllocatedOnHeap)
	{
		*pbAllocatedOnHeap = FALSE;
	}
	
	if( pAllocator)
	{
		f_assert( pAllocator->getCellSize() >= uiSize);

		if( m_hMutex != F_MUTEX_NULL)
		{
			f_assertMutexNotLocked( m_hMutex);
			f_mutexLock( m_hMutex);
			bMutexLocked = TRUE;
		}
			
		if( (*ppucBuffer = (FLMBYTE *)pAllocator->allocCell( pRelocator, 
			pvInitialData, uiDataSize)) == NULL)
		{
			rc = RC_SET( NE_FLM_MEM);
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = f_alloc( uiSize, ppucBuffer)))
		{
			goto Exit;
		}
		
		m_pSlabManager->incrementTotalBytesAllocated( f_msize( *ppucBuffer));
		
		if( pvInitialData)
		{
			f_memcpy( *ppucBuffer, pvInitialData, uiDataSize);
		}
		
		if( pbAllocatedOnHeap)
		{
			*pbAllocatedOnHeap = TRUE;
		}
	}
	
Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/ 
RCODE FTKAPI F_BufferAlloc::allocBuf(
	IF_Relocator *			pRelocator,
	FLMUINT					uiSize,
	F_ALLOC_INIT_FUNC		fnAllocInit,
	FLMBYTE **				ppucBuffer,
	FLMBOOL *				pbAllocatedOnHeap)
{
	RCODE						rc = NE_FLM_OK;
	IF_FixedAlloc *		pAllocator = getAllocator( uiSize);
	FLMBOOL					bMutexLocked = FALSE;

	if( pbAllocatedOnHeap)
	{
		*pbAllocatedOnHeap = FALSE;
	}
	
	if( pAllocator)
	{
		f_assert( pAllocator->getCellSize() >= uiSize);
		
		if( m_hMutex != F_MUTEX_NULL)
		{
			f_assertMutexNotLocked( m_hMutex);
			f_mutexLock( m_hMutex);
			bMutexLocked = TRUE;
		}
		
		if( (*ppucBuffer = (FLMBYTE *)pAllocator->allocCell( 
			pRelocator, fnAllocInit)) == NULL)
		{
			rc = RC_SET( NE_FLM_MEM);
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = f_alloc( uiSize, ppucBuffer)))
		{
			goto Exit;
		}
		
		m_pSlabManager->incrementTotalBytesAllocated( f_msize( *ppucBuffer));
		
		if( fnAllocInit)
		{
			fnAllocInit( *ppucBuffer, uiSize);
		}
		
		if( pbAllocatedOnHeap)
		{
			*pbAllocatedOnHeap = TRUE;
		}
	}
	
Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}
	
	return( rc);
}
	
/****************************************************************************
Desc:
****************************************************************************/ 
RCODE FTKAPI F_BufferAlloc::reallocBuf(
	IF_Relocator *			pRelocator,
	FLMUINT					uiOldSize,
	FLMUINT					uiNewSize,
	void *					pvInitialData,
	FLMUINT					uiDataSize,
	FLMBYTE **				ppucBuffer,
	FLMBOOL *				pbAllocatedOnHeap)
{
	RCODE						rc = NE_FLM_OK;
	FLMBYTE *				pucTmp;
	IF_FixedAlloc *		pOldAllocator;
	IF_FixedAlloc *		pNewAllocator;
	FLMBOOL					bMutexLocked = FALSE;

	f_assert( uiNewSize);
	
	if( !uiOldSize)
	{
		rc = allocBuf( pRelocator, uiNewSize, pvInitialData, uiDataSize, 
			ppucBuffer, pbAllocatedOnHeap);
		goto Exit;
	}

	pOldAllocator = getAllocator( uiOldSize);
	pNewAllocator = getAllocator( uiNewSize);

	if( pOldAllocator && pOldAllocator == pNewAllocator)
	{
		// The allocation will still fit in the same cell

		goto Exit;
	}
	
	if( pbAllocatedOnHeap)
	{
		*pbAllocatedOnHeap = FALSE;
	}
	
	if( pOldAllocator)
	{
		if( m_hMutex != F_MUTEX_NULL)
		{
			f_assertMutexNotLocked( m_hMutex);
			f_mutexLock( m_hMutex);
			bMutexLocked = TRUE;
		}
		
		if( pNewAllocator)
		{
			f_assert( pOldAllocator != pNewAllocator);

			if( (pucTmp = (FLMBYTE *)pNewAllocator->allocCell( pRelocator,
										NULL, 0)) == NULL)
			{
				rc = RC_SET( NE_FLM_MEM);
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = f_alloc( uiNewSize, &pucTmp)))
			{
				goto Exit;
			}
			
			m_pSlabManager->incrementTotalBytesAllocated( f_msize( pucTmp));
			
			if( pbAllocatedOnHeap)
			{
				*pbAllocatedOnHeap = TRUE;
			}
		}

		f_memcpy( pucTmp, *ppucBuffer, f_min( uiOldSize, uiNewSize));
		pOldAllocator->freeCell( *ppucBuffer);
		*ppucBuffer = pucTmp;
	}
	else
	{
		if( pNewAllocator)
		{
			if( m_hMutex != F_MUTEX_NULL)
			{
				f_assertMutexNotLocked( m_hMutex);
				f_mutexLock( m_hMutex);
				bMutexLocked = TRUE;
			}
			
			if( (pucTmp = (FLMBYTE *)pNewAllocator->allocCell( pRelocator, 
				*ppucBuffer, f_min( uiOldSize, uiNewSize))) == NULL)
			{
				rc = RC_SET( NE_FLM_MEM);
				goto Exit;
			}

			if( bMutexLocked)
			{
				f_mutexUnlock( m_hMutex);
				bMutexLocked = FALSE;
			}

			m_pSlabManager->decrementTotalBytesAllocated( f_msize( *ppucBuffer));			
			f_free( ppucBuffer);
			*ppucBuffer = pucTmp;
		}
		else
		{
			FLMUINT		uiOldAllocSize = f_msize( *ppucBuffer);
	
			f_assert( uiOldSize > m_ppAllocators[ NUM_BUF_ALLOCATORS - 1]->getCellSize());
			f_assert( uiNewSize > m_ppAllocators[ NUM_BUF_ALLOCATORS - 1]->getCellSize());
			
			if( RC_BAD( rc = f_realloc( uiNewSize, ppucBuffer)))
			{
				goto Exit;
			}
			
			m_pSlabManager->decrementTotalBytesAllocated( uiOldAllocSize);
			m_pSlabManager->incrementTotalBytesAllocated( f_msize( *ppucBuffer));
			
			if( pbAllocatedOnHeap)
			{
				*pbAllocatedOnHeap = TRUE;
			}
		}
	}

Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/ 
void F_BufferAlloc::freeBuf(
	FLMUINT				uiSize,
	FLMBYTE **			ppucBuffer)
{
	IF_FixedAlloc *	pAllocator = getAllocator( uiSize);
	FLMBOOL				bMutexLocked = FALSE;
	
	if( pAllocator)
	{
		if( m_hMutex != F_MUTEX_NULL)
		{
			f_assertMutexNotLocked( m_hMutex);
			f_mutexLock( m_hMutex);
			bMutexLocked = TRUE;
		}
		
		pAllocator->freeCell( *ppucBuffer);
		*ppucBuffer = NULL;
		
		if( bMutexLocked)
		{
			f_mutexUnlock( m_hMutex);
		}
	}
	else
	{
		m_pSlabManager->decrementTotalBytesAllocated( f_msize( *ppucBuffer));			
		f_free( ppucBuffer);
	}
}

/****************************************************************************
Desc:
****************************************************************************/ 
void F_BufferAlloc::defragmentMemory( void)
{
	FLMUINT		uiLoop;

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexNotLocked( m_hMutex);
		f_mutexLock( m_hMutex);
	}
	
	for( uiLoop = 0; uiLoop < NUM_BUF_ALLOCATORS; uiLoop++)
	{
		if( m_ppAllocators[ uiLoop])
		{
			m_ppAllocators[ uiLoop]->defragmentMemory();
			m_ppAllocators[ uiLoop]->freeUnused();
		}

		uiLoop++;
	}
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( m_hMutex);
	}
}

/****************************************************************************
Desc:
****************************************************************************/ 
FLMUINT F_BufferAlloc::getTrueSize(
	FLMUINT				uiSize,
	FLMBYTE *			pucBuffer)
{
	FLMUINT				uiTrueSize;
	IF_FixedAlloc *	pAllocator;

	if( !uiSize)
	{
		uiTrueSize = 0;
	}
	else if( (pAllocator = getAllocator( uiSize)) != NULL)
	{
		uiTrueSize = pAllocator->getCellSize();
	}
	else
	{
		uiTrueSize = f_msize( pucBuffer);
	}

	return( uiTrueSize);
}

/****************************************************************************
Desc:
****************************************************************************/ 
IF_FixedAlloc * F_BufferAlloc::getAllocator(
	FLMUINT				uiSize)
{
	IF_FixedAlloc *		pAllocator;

	f_assert( uiSize);
	
	if( uiSize <= CELL_SIZE_10)
	{
		if( uiSize <= CELL_SIZE_4)
		{
			if( uiSize <= CELL_SIZE_2)
			{
				if( uiSize <= CELL_SIZE_0)
				{
					pAllocator = m_ppAllocators [0];
				}
				else
				{
					pAllocator = (uiSize <= CELL_SIZE_1
															? m_ppAllocators [1]
															: m_ppAllocators [2]);
				}
			}
			else
			{
				pAllocator = (uiSize <= CELL_SIZE_3
														? m_ppAllocators [3]
														: m_ppAllocators [4]);
			}
		}
		else if( uiSize <= CELL_SIZE_7)
		{
			if( uiSize <= CELL_SIZE_5)
			{
				pAllocator = m_ppAllocators [5];
			}
			else
			{
				pAllocator = (uiSize <= CELL_SIZE_6
														? m_ppAllocators [6]
														: m_ppAllocators [7]);
			}
		}
		else
		{
			if( uiSize <= CELL_SIZE_8)
			{
				pAllocator = m_ppAllocators [8];
			}
			else
			{
				pAllocator = (uiSize <= CELL_SIZE_9
														? m_ppAllocators [9]
														: m_ppAllocators [10]);
			}
		}
	}
	else if( uiSize <= CELL_SIZE_16)
	{
		if( uiSize <= CELL_SIZE_13)
		{
			if( uiSize <= CELL_SIZE_11)
			{
				pAllocator = m_ppAllocators [11];
			}
			else
			{
				pAllocator = (uiSize <= CELL_SIZE_12
														? m_ppAllocators [12]
														: m_ppAllocators [13]);
			}
		}
		else
		{
			if( uiSize <= CELL_SIZE_14)
			{
				pAllocator = m_ppAllocators [14];
			}
			else
			{
				pAllocator = (uiSize <= CELL_SIZE_15
														? m_ppAllocators [15]
														: m_ppAllocators [16]);
			}
		}
	}
	else if( uiSize <= CELL_SIZE_19)
	{
		if( uiSize <= CELL_SIZE_17)
		{
			pAllocator = m_ppAllocators [17];
		}
		else
		{
			pAllocator = (uiSize <= CELL_SIZE_18
													? m_ppAllocators [18]
													: m_ppAllocators [19]);
		}
	}
	else if( uiSize <= CELL_SIZE_21)
	{
		pAllocator = (uiSize <= CELL_SIZE_20
												? m_ppAllocators [20]
												: m_ppAllocators [21]);
	}
	else
	{
		pAllocator = NULL;
	}

	return( pAllocator);
}
/****************************************************************************
Desc:
****************************************************************************/ 
RCODE F_MultiAlloc::setup(
	FLMBOOL					bMultiThreaded,
	IF_SlabManager *		pSlabManager,
	IF_Relocator *			pDefaultRelocator,
	FLMUINT *				puiCellSizes,
	FLM_SLAB_USAGE *		pUsageStats,
	FLMUINT *				puiTotalBytesAllocated)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiLoop;
	FLMUINT		uiCellCount;
	
	if( bMultiThreaded)
	{
		if( RC_BAD( rc = f_mutexCreate( &m_hMutex)))
		{
			goto Exit;
		}
	}

	m_pSlabManager = pSlabManager;
	m_pSlabManager->AddRef();
	
	uiCellCount = 0;
	while( puiCellSizes[ uiCellCount])
	{
		uiCellCount++;
	}

	if( !uiCellCount)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_INVALID_PARM);
		goto Exit;
	}

	f_qsort( puiCellSizes, 0, uiCellCount - 1, 
		f_qsortUINTCompare, f_qsortUINTSwap);

	if( RC_BAD( rc = f_alloc( 
		sizeof( FLMUINT *) * (uiCellCount + 1), &m_puiCellSizes)))
	{
		goto Exit;
	}
	
	m_pSlabManager->incrementTotalBytesAllocated( f_msize( m_puiCellSizes));
	
	f_memcpy( m_puiCellSizes, puiCellSizes, 
		(uiCellCount + 1) * sizeof( FLMUINT));

	// Set up the allocators

	if( RC_BAD( rc = f_calloc( 
		sizeof( F_FixedAlloc *) * (uiCellCount + 1), &m_ppAllocators)))
	{
		goto Exit;
	}
	
	m_pSlabManager->incrementTotalBytesAllocated( f_msize( m_ppAllocators));

	uiLoop = 0;
	while( m_puiCellSizes[ uiLoop])
	{
		if( (m_ppAllocators[ uiLoop] = f_new F_FixedAlloc) == NULL)
		{
			rc = RC_SET( NE_FLM_MEM);
			goto Exit;
		}

		if( RC_BAD( rc = m_ppAllocators[ uiLoop]->setup( FALSE, 
			pSlabManager, pDefaultRelocator, m_puiCellSizes[ uiLoop], 
			pUsageStats, puiTotalBytesAllocated)))
		{
			goto Exit;
		}

		uiLoop++;
	}

Exit:

	if( RC_BAD( rc))
	{
		cleanup();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/ 
void F_MultiAlloc::cleanup( void)
{
	FLMUINT		uiLoop = 0;

	if( m_puiCellSizes && m_ppAllocators)
	{
		while( m_puiCellSizes[ uiLoop])
		{
			if( m_ppAllocators[ uiLoop])
			{
				m_ppAllocators[ uiLoop]->Release();
				m_ppAllocators[ uiLoop] = NULL;
			}
	
			uiLoop++;
		}
	}

	if( m_puiCellSizes)
	{
		m_pSlabManager->decrementTotalBytesAllocated( f_msize( m_puiCellSizes));
		f_free( &m_puiCellSizes);
	}
	
	if( m_ppAllocators)
	{
		m_pSlabManager->decrementTotalBytesAllocated( f_msize( m_ppAllocators));
		f_free( &m_ppAllocators);
	}
	
	if( m_pSlabManager)
	{
		m_pSlabManager->Release();
		m_pSlabManager = NULL;
	}
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}
}

/****************************************************************************
Desc:
****************************************************************************/ 
RCODE FTKAPI F_MultiAlloc::allocBuf(
	IF_Relocator *		pRelocator,
	FLMUINT				uiSize,
	FLMBYTE **			ppucBuffer)
{
	RCODE					rc = NE_FLM_OK;
	IF_FixedAlloc *	pAllocator = getAllocator( uiSize);
	FLMBOOL				bMutexLocked = FALSE;
	
	f_assert( pAllocator);
	f_assert( pAllocator->getCellSize() >= uiSize);
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexNotLocked( m_hMutex);
		f_mutexLock( m_hMutex);
		bMutexLocked = TRUE;
	}
	
	if( (*ppucBuffer = (FLMBYTE *)pAllocator->allocCell( 
		pRelocator, NULL, 0)) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
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
Desc:
****************************************************************************/ 
RCODE FTKAPI F_MultiAlloc::allocBuf(
	IF_Relocator *			pRelocator,
	FLMUINT					uiSize,
	F_ALLOC_INIT_FUNC		fnAllocInit,
	FLMBYTE **				ppucBuffer)
{
	RCODE					rc = NE_FLM_OK;
	IF_FixedAlloc *	pAllocator = getAllocator( uiSize);
	FLMBOOL				bMutexLocked = FALSE;
	
	f_assert( pAllocator);
	f_assert( pAllocator->getCellSize() >= uiSize);
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexNotLocked( m_hMutex);
		f_mutexLock( m_hMutex);
		bMutexLocked = TRUE;
	}
	
	if( (*ppucBuffer = (FLMBYTE *)pAllocator->allocCell( 
		pRelocator, fnAllocInit)) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
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
Desc:
****************************************************************************/ 
RCODE FTKAPI F_MultiAlloc::reallocBuf(
	IF_Relocator *		pRelocator,
	FLMUINT				uiNewSize,
	FLMBYTE **			ppucBuffer)
{
	RCODE					rc = NE_FLM_OK;
	FLMBYTE *			pucTmp;
	IF_FixedAlloc *	pOldAllocator;
	IF_FixedAlloc *	pNewAllocator;
	FLMBOOL				bMutexLocked = FALSE;

	f_assert( uiNewSize);

	if( !(*ppucBuffer))
	{
		rc = allocBuf( pRelocator, uiNewSize, ppucBuffer);
		goto Exit;
	}

	pOldAllocator = getAllocator( *ppucBuffer);
	pNewAllocator = getAllocator( uiNewSize);

	if( pOldAllocator == pNewAllocator)
	{
		// The allocation will still fit in the same cell

		goto Exit;
	}
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexNotLocked( m_hMutex);
		f_mutexLock( m_hMutex);
		bMutexLocked = TRUE;
	}
	
	if( (pucTmp = (FLMBYTE *)pNewAllocator->allocCell( pRelocator, *ppucBuffer, 
		f_min( uiNewSize, pOldAllocator->getCellSize()))) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}

	pOldAllocator->freeCell( *ppucBuffer);
	*ppucBuffer = pucTmp;
	
Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/ 
void FTKAPI F_MultiAlloc::freeBuf(
	FLMBYTE **				ppucBuffer)
{
	if( ppucBuffer && *ppucBuffer)
	{
		if( m_hMutex != F_MUTEX_NULL)
		{
			f_assertMutexNotLocked( m_hMutex);
			f_mutexLock( m_hMutex);
		}
		
		getAllocator( *ppucBuffer)->freeCell( *ppucBuffer);
		*ppucBuffer = NULL;
		
		if( m_hMutex != F_MUTEX_NULL)
		{
			f_mutexUnlock( m_hMutex);
		}
	}
}
	
/****************************************************************************
Desc:
****************************************************************************/ 
FLMUINT FTKAPI F_MultiAlloc::getTrueSize(
	FLMBYTE *				pucBuffer)
{
	FLMUINT			uiSize;
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexNotLocked( m_hMutex);
		f_mutexLock( m_hMutex);
	}
	
	uiSize = getAllocator( pucBuffer)->getCellSize();
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( m_hMutex);
	}
	
	return( uiSize);
}
	
/****************************************************************************
Desc:
****************************************************************************/ 
void F_MultiAlloc::defragmentMemory( void)
{
	FLMUINT		uiLoop = 0;

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexNotLocked( m_hMutex);
		f_mutexLock( m_hMutex);
	}
	
	while( m_puiCellSizes[ uiLoop])
	{
		if( m_ppAllocators[ uiLoop])
		{
			m_ppAllocators[ uiLoop]->defragmentMemory();
			m_ppAllocators[ uiLoop]->freeUnused();
		}

		uiLoop++;
	}
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( m_hMutex);
	}
}

/****************************************************************************
Desc:
****************************************************************************/ 
IF_FixedAlloc * F_MultiAlloc::getAllocator(
	FLMUINT				uiSize)
{
	IF_FixedAlloc *	pAllocator = NULL;
	FLMUINT				uiLoop;

	f_assert( uiSize);

	for( uiLoop = 0; m_puiCellSizes[ uiLoop]; uiLoop++)
	{
		if( m_puiCellSizes[ uiLoop] >= uiSize)
		{
			pAllocator = m_ppAllocators[ uiLoop];
			break;
		}
	}

	return( pAllocator);
}

/****************************************************************************
Desc:
****************************************************************************/ 
IF_FixedAlloc * F_MultiAlloc::getAllocator(
	FLMBYTE *			pucBuffer)
{
	CELLHEADER *		pHeader;
	FIXEDSLAB *			pSlab;
	IF_FixedAlloc *	pAllocator = NULL;

#ifdef FLM_DEBUG
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexLocked( m_hMutex);
	}
#endif
	
	pHeader = (CELLHEADER *)(pucBuffer - 
			F_FixedAlloc::getAllocAlignedSize( sizeof( CELLHEADER2)));
	pSlab = pHeader->pContainingSlab;
	pAllocator = (IF_FixedAlloc *)pSlab->pvAllocator;

	return( pAllocator);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI f_allocAlignedBufferImp(
	FLMUINT			uiMinSize,
	void **			ppvAlloc)
{
	RCODE		rc = NE_FLM_OK;
	
#if defined( FLM_WIN)

	if ((*ppvAlloc = (void *)VirtualAlloc( NULL,
		uiMinSize, MEM_COMMIT, PAGE_READWRITE)) == NULL)
	{
		rc = f_mapPlatformError( GetLastError(), NE_FLM_MEM);
		goto Exit;
	}
	
#elif defined( FLM_SOLARIS)

	if( (*ppvAlloc = memalign( sysconf( _SC_PAGESIZE), uiMinSize)) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
#elif defined( FLM_LINUX)

	if( posix_memalign( ppvAlloc, sysconf( _SC_PAGESIZE), uiMinSize) != 0)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
#elif defined( FLM_UNIX)

	if( (*ppvAlloc = valloc( uiMinSize)) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
#else

	{
		FLMUINT		uiPageSize = 512;
		FLMUINT		uiAllocSize;
		FLMBYTE *	pucAlloc;
		FLMBYTE *	pucStartOfAlloc;
		
		uiAllocSize = 
			(FLMUINT)(f_roundUp( uiMinSize, uiPageSize) + (2 * uiPageSize));
		
	#if defined( FLM_RING_ZERO_NLM)
		if( (pucAlloc = (FLMBYTE *)Alloc( uiAllocSize, gv_lAllocRTag)) == NULL)
		{
			rc = RC_SET( NE_FLM_MEM);
			goto Exit;
		}
	#else
		if( RC_BAD( rc = f_alloc( uiAllocSize, &pucAlloc)))
		{
			goto Exit;
		}
	#endif
		
		pucStartOfAlloc = pucAlloc;
		pucAlloc += (uiPageSize - (((FLMUINT)pucAlloc) % uiPageSize)) + uiPageSize;
		
		f_assert( ((FLMUINT)(pucAlloc) % uiPageSize) == 0);

		U642FBA( (FLMUINT64)pucStartOfAlloc, pucAlloc - 8);
		*ppvAlloc = pucAlloc;
	}

#endif

	f_memset( *ppvAlloc, 0, uiMinSize);

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI f_freeAlignedBufferImp(
	void **			ppvAlloc)
{
	if( *ppvAlloc)
	{
#if defined( FLM_WIN)

		(void)VirtualFree( *ppvAlloc, 0, MEM_RELEASE);
		*ppvAlloc = NULL;
		
#elif defined( FLM_UNIX)

		free( *ppvAlloc);
		*ppvAlloc = NULL;
		
#else

		{
			FLMBYTE *		pucAlloc;
			
			pucAlloc = (FLMBYTE *)FB2U64( ((FLMBYTE *)(*ppvAlloc)) - 8);
		#if defined( FLM_RING_ZERO_NLM)
			Free( pucAlloc);
		#else
			f_free( &pucAlloc);
		#endif
			*ppvAlloc = NULL;
		}

#endif
	}
}

/****************************************************************************
Desc:
****************************************************************************/
FLMBOOL FTKAPI f_canGetMemoryInfo( void)
{
	if( RC_OK( f_getMemoryInfo( NULL, NULL)))
	{
		return( TRUE);
	}
	
	return( FALSE);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI f_getMemoryInfo(
	FLMUINT64 *		pui64TotalPhysMem,
	FLMUINT64 *		pui64AvailPhysMem)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT64		ui64TotalPhysMem = 0;
	FLMUINT64		ui64AvailPhysMem = 0;

#ifdef FLM_WIN
	{
		MEMORYSTATUS	MemStatus;
		
		GlobalMemoryStatus( &MemStatus);
		
		ui64TotalPhysMem = MemStatus.dwTotalPhys;
		ui64AvailPhysMem = MemStatus.dwAvailPhys;
		
		// There could be more physical memory in the system than we could
		// actually allocate in our virtual address space.  Thus, we need to
		// make sure that we never exceed our total virtual address space.
	
		if (ui64TotalPhysMem > (FLMUINT)MemStatus.dwTotalVirtual)
		{
			ui64TotalPhysMem = (FLMUINT)MemStatus.dwTotalVirtual;
		}
		
		if( ui64AvailPhysMem > ui64TotalPhysMem)
		{
			ui64AvailPhysMem = ui64TotalPhysMem;
		}
	}
#elif defined( FLM_UNIX)
	{
		FLMUINT			uiProcMemLimit = FLM_MAX_UINT;
		FLMUINT			uiProcVMemLimit = FLM_MAX_UINT;
		
		#if defined( FLM_LINUX)
		
				f_getLinuxMemInfo( &ui64TotalPhysMem, &ui64AvailPhysMem);
		
		#elif defined( FLM_AIX)
			
			f_getAIXMemInfo( &ui64TotalPhysMem, &ui64AvailPhysMem);
		
		#elif defined( FLM_HPUX)
		
			f_getHPUXMemInfo( &ui64TotalPhysMem, &ui64AvailPhysMem);
		
		#elif defined( _SC_PAGESIZE) && defined( _SC_AVPHYS_PAGES)

			long		iPageSize = sysconf( _SC_PAGESIZE);
	
			// Get the amount of memory available to the system
			
			ui64TotalPhysMem = sysconf( _SC_PHYS_PAGES) * iPageSize;
			ui64AvailPhysMem = sysconf( _SC_AVPHYS_PAGES) * iPageSize;
			
		#else
		
			return( RC_SET( NE_FLM_NOT_IMPLEMENTED));
	
		#endif
		
		// The process might be limited in the amount of memory it
		// can access.

		#if defined( RLIMIT_VMEM)
		{
			struct rlimit	rlim;

			if( getrlimit( RLIMIT_VMEM, &rlim) != 0)
			{
				rlim.rlim_cur = (rlim_t)RLIM_INFINITY;
				rlim.rlim_max = (rlim_t)RLIM_INFINITY;
			}

			if( rlim.rlim_cur != (rlim_t)RLIM_INFINITY)
			{
				uiProcVMemLimit = (FLMUINT)rlim.rlim_cur;
			}
		}
		#endif

		#if defined( RLIMIT_DATA)
		{
			struct rlimit	rlim;

			if( getrlimit( RLIMIT_DATA, &rlim) != 0)
			{
				rlim.rlim_cur = (rlim_t)RLIM_INFINITY;
				rlim.rlim_max = (rlim_t)RLIM_INFINITY;
			}

			if( rlim.rlim_cur != (rlim_t)RLIM_INFINITY)
			{
				uiProcMemLimit = (FLMUINT)rlim.rlim_cur;
			}
		}
		#endif
	
		if( ui64TotalPhysMem > uiProcMemLimit)
		{
			ui64TotalPhysMem = uiProcMemLimit;
		}
	
		if( ui64TotalPhysMem > uiProcVMemLimit)
		{
			ui64TotalPhysMem = uiProcVMemLimit;
		}
	}
#elif defined( FLM_LIBC_NLM)
	{
		#ifndef _SC_PHYS_PAGES
			#define _SC_PHYS_PAGES        56
		#endif
		#ifndef _SCAVPHYS_PAGES
			#define _SC_AVPHYS_PAGES		57
		#endif

		long		iPageSize = sysconf( _SC_PAGESIZE);

		// Get the amount of memory available to the system
		
		ui64TotalPhysMem = sysconf(_SC_PHYS_PAGES) * iPageSize;
		ui64AvailPhysMem = sysconf(_SC_AVPHYS_PAGES) * iPageSize;
	}
#elif defined( FLM_RING_ZERO_NLM)
	{
		FLMUINT	uiCacheBufferSize = GetCacheBufferSize();
		
		ui64TotalPhysMem = GetOriginalNumberOfCacheBuffers() * uiCacheBufferSize;
		ui64AvailPhysMem = GetCurrentNumberOfCacheBuffers() * uiCacheBufferSize;
		
		// Get available memory in local process pool

		{
			FLMUINT	uiFreeBytes;
			FLMUINT	uiFreeNodes;
			FLMUINT	uiAllocatedBytes;
			FLMUINT	uiAllocatedNodes;
			FLMUINT	uiTotalMemory;

			if (GetNLMAllocMemoryCounts( f_getNLMHandle(),
									&uiFreeBytes, &uiFreeNodes,
									&uiAllocatedBytes, &uiAllocatedNodes,
									&uiTotalMemory) == 0)
			{
				ui64AvailPhysMem += uiFreeBytes;
			}
		}
	}
#else
	rc = RC_SET( NE_FLM_NOT_IMPLEMENTED);
#endif

	if( ui64AvailPhysMem > ui64TotalPhysMem)
	{
		ui64AvailPhysMem = ui64TotalPhysMem;
	}
	
	if( pui64TotalPhysMem)
	{
		*pui64TotalPhysMem = ui64TotalPhysMem;
	}
	
	if( pui64AvailPhysMem)
	{
		*pui64AvailPhysMem = ui64AvailPhysMem;
	}

	return( rc);
}

/***************************************************************************
Desc:
***************************************************************************/
#ifdef FLM_LINUX
FLMUINT64 f_getLinuxMemInfoValue(
	char *			pszMemInfoBuffer,
	const char *	pszTag)
{
	char *			pszTmp;
	FLMUINT64		ui64Bytes = 0;

	if( (pszTmp = f_strstr( pszMemInfoBuffer, pszTag)) == NULL)
	{
		return( 0);
	}
	
	pszTmp += f_strlen( pszTag);
	
	while( *pszTmp == ASCII_SPACE)
	{
		pszTmp++;
	}

	while( *pszTmp >= '0' && *pszTmp <= '9')
	{
		ui64Bytes *= 10;
		ui64Bytes += (FLMUINT)(*pszTmp - '0');
		pszTmp++;
	}
	
	return( ui64Bytes * 1024);
}
#endif

/***************************************************************************
Desc:
***************************************************************************/
#ifdef FLM_LINUX
void f_getLinuxMemInfo(
	FLMUINT64 *		pui64TotalMem,
	FLMUINT64 *		pui64AvailMem)
{
	int				fd = -1;
	int				iBytesRead;
	int				iMemInfoBufSize = 4096;
	char *			pszMemInfoBuf = NULL;
	FLMUINT64		ui64TotalMem = 0;
	FLMUINT64		ui64AvailMem = 0;

	if( (pszMemInfoBuf = (char *)os_malloc( iMemInfoBufSize)) == NULL)
	{
		goto Exit;
	}
	
	if( (fd = open( "/proc/meminfo", O_RDONLY, 0600)) == -1)
	{
		goto Exit;
	}

	if( (iBytesRead = read( fd, pszMemInfoBuf, iMemInfoBufSize - 1)) == -1)
	{
		goto Exit;
	}
	
	pszMemInfoBuf[ iBytesRead] = 0;
	
	if( (ui64TotalMem = 
		f_getLinuxMemInfoValue( pszMemInfoBuf, "MemTotal:")) != 0)
	{
		ui64AvailMem = 
				f_getLinuxMemInfoValue( pszMemInfoBuf, "MemFree:") +
				f_getLinuxMemInfoValue( pszMemInfoBuf, "Buffers:") +
				f_getLinuxMemInfoValue( pszMemInfoBuf, "Cached:");
	}
	
Exit:

	if( pui64TotalMem)
	{
		*pui64TotalMem = ui64TotalMem;
	}
	
	if( pui64AvailMem)
	{
		*pui64AvailMem = ui64AvailMem;
	}

	if( pszMemInfoBuf)
	{
		os_free( pszMemInfoBuf);
	}
	
	if( fd != -1)
	{
		close( fd);
	}
}
#endif

/***************************************************************************
Desc:
***************************************************************************/
#ifdef FLM_AIX
void f_getAIXMemInfo(
	FLMUINT64 *		pui64TotalMem,
	FLMUINT64 *		pui64AvailMem)
{
	struct vminfo		tmpvminfo;
	FLMUINT64			ui64TotalPhysMem;
	FLMUINT64			ui64AvailPhysMem;
	void *				pvModule = NULL;
	VMGETINFO_FUNC		fnVMGetInfo;

#ifdef _SC_PAGESIZE
	long					iPageSize = sysconf(_SC_PAGESIZE);
#else
	long					iPageSize = 4096;
#endif

	if( iPageSize == -1)
	{
		// If sysconf returned an error, resort to using the default
		// page size for the Power architecture.

		iPageSize = 4096;
	}
	
	ui64TotalPhysMem = FLM_MAX_UINT;
	ui64AvailPhysMem = FLM_MAX_UINT;
	
	if( (pvModule = dlopen( "/unix", RTLD_NOW | RTLD_GLOBAL)) == NULL)
	{
		goto Exit;
	}
	
	if( (fnVMGetInfo = (VMGETINFO_FUNC)dlsym( pvModule, "vmgetinfo")) == NULL)
	{
		goto Exit;
	}
	
	if( fnVMGetInfo( &tmpvminfo, VMINFO, sizeof( tmpvminfo)) != -1)
	{
		ui64TotalPhysMem = tmpvminfo.memsizepgs * iPageSize;
		ui64AvailPhysMem = tmpvminfo.numfrb * iPageSize;
	}
	
Exit:

	if( pvModule)
	{
		dlclose( pvModule);
	}
	
	*pui64TotalMem = ui64TotalPhysMem;
	*pui64AvailMem = ui64AvailPhysMem;
}
#endif				

/***************************************************************************
Desc:
***************************************************************************/
#ifdef FLM_HPUX
void f_getHPUXMemInfo(
	FLMUINT64 *		pui64TotalMem,
	FLMUINT64 *		pui64AvailMem)
{
	FLMUINT					uiPageSize;
	struct pst_static		pst;
	struct pst_dynamic	dyn;

	if( pstat_getstatic( &pst, sizeof( pst), (size_t)1, 0) == -1)
	{
		uiPageSize = 4096;
	}
	else
	{
		uiPageSize = pst.page_size;
	}
	
	if( pstat_getdynamic( &dyn, sizeof( dyn), 1, 0) != -1)
	{
		*pui64AvailMem = dyn.psd_free * uiPageSize;
	}
	else
	{
		*pui64AvailMem = pst.physical_memory * uiPageSize;
	}
	
	*pui64TotalMem = pst.physical_memory * uiPageSize;
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_RING_ZERO_NLM
void * nlm_realloc(
	void	*	pMemory,
	size_t	newSize)
{
	void *		pNewMemory;
	LONG			lSize;

 	if( !pMemory)
 	{
 		pNewMemory = Alloc( newSize, gv_lAllocRTag);
		goto Exit;
	}

	lSize = SizeOfAllocBlock( pMemory);

	pNewMemory = os_malloc( newSize);
	if( !pNewMemory)
	{
		goto Exit;
	}

	if( lSize > newSize)
	{
		lSize = newSize;
	}
	
	f_memcpy( pNewMemory, pMemory, lSize);

	if( pMemory)
	{
		Free( pMemory);
	}

Exit:

	return( pNewMemory);
}
#endif


/****************************************************************************
The NetWare Internal Debugger encrypts all symbols by XORing each character 
in the symbol name with the character at the corresponding position in the 
following mask. To see how the crypt mask is defined in NetWare, see 
SYMDEB.386 in the NetWare source. We have three options:
	
	1. We can just emulate the internal debugger and decrypt the symbol, 
		print it, and reencrypt it each time we want to display it (less than 
		efficient). 
	2. We can make a decrypted copy of the symbols we are interested in on 
		module init, then use our own copy and free it on module exit.
	3. We can use the symbol list, but decrypt character by character into
		an internal string buffer, then print the buffer.
****************************************************************************/

/****************************************************************************
Desc:
****************************************************************************/
F_ObjRefTracker::F_ObjRefTracker(void)
{
	m_hRefListMutex = F_MUTEX_NULL;
	m_pListManager = NULL;;
	m_pFileSystem = NULL;
	m_pszObjName[ 0] = '\0';
	m_pAddrFmtHook = NULL;
	m_pUserData = NULL;
	m_pModHandle = NULL;
}

/****************************************************************************
Desc: Allocates required memory and initializes a local file system
****************************************************************************/
RCODE F_ObjRefTracker::setup(
	const char *	pszObjName,
	FLMBOOL			bLogToFile)
{
	RCODE				rc = NE_FLM_OK;
	char				pszTmpBuf[ FORTRACK_MAX_OBJ_NAME_LEN + 5];
	char *			pucTmp;

	// Allocate a mutex

	if( RC_BAD( rc = f_mutexCreate( &m_hRefListMutex)))
	{
		goto Exit;
	}

	// Allocate the list

	if( (m_pListManager = f_new F_ListManager( &m_lnode, 1)) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}

	// Create a local file system object

	if( bLogToFile)
	{
		if( RC_BAD( rc = FlmGetFileSystem( &m_pFileSystem)))
		{
			goto Exit;
		}
	}

	if( f_strlen( pszObjName) <= FORTRACK_MAX_OBJ_NAME_LEN)
	{
		f_strcpy( m_pszObjName, pszObjName);
	}
	else
	{
		f_sprintf( m_pszObjName, "OBJTRCK");
	}

	// Set the log path

	f_strcpy( pszTmpBuf, m_pszObjName);
	pucTmp = pszTmpBuf;
	while( *pucTmp)
	{
		if( *pucTmp >= 'a' && *pucTmp <= 'z')
		{
			*pucTmp = (*pucTmp - 'a') + 'A';
		}
		pucTmp++;
	}
	f_strcat( pszTmpBuf, ".OTL");

#ifdef FLM_NLM
	f_strcpy( m_pLogPath, "SYS:\\SYSTEM\\");
	f_strcat( m_pLogPath, pszTmpBuf);
#else
	f_strcpy( m_pLogPath, pszTmpBuf);
#endif

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
F_ObjRefTracker::~F_ObjRefTracker(void)
{
	if( m_pListManager)
	{
		m_pListManager->Release();
		m_pListManager = NULL;
	}

	if( m_hRefListMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hRefListMutex);
	}

	if( m_pFileSystem)
	{
		m_pFileSystem->Release();
	}
}

/****************************************************************************
Desc: Track the given reference.
****************************************************************************/
void F_ObjRefTracker::trackRef(
	void *					pReferenceID,
	void *					pSubrefID)
{
	F_TrackingRecord *	pTrackingRec = NULL;
	void **					pStack;
	
	if( m_hRefListMutex == F_MUTEX_NULL)
	{
		// Reference Tracking has not been initialized, just exit.
		
		goto Exit;
	}

	if( !pReferenceID)
	{
		// Do not track NULL references
		
		goto Exit;
	}

	// If there is insufficient memory to allocate a tracking record, 
	// then we will never know if this reference is properly released.
	
	if( (pTrackingRec = f_new F_TrackingRecord( 
		pReferenceID, pSubrefID)) == NULL)
	{
		char	pucMessage[ 100];
		
		logError( "trackRef: Insufficient memory to allocate tracking record");
		f_sprintf( pucMessage, "\treference %x.%x will not be tracked",
			(unsigned)((FLMUINT)pReferenceID), (unsigned)((FLMUINT)pSubrefID));
		logError( pucMessage);
		goto Exit;
	}

	pTrackingRec->setup( m_pListManager, &m_lnode, 1);

	// Add the tracking record to the list
	
	f_mutexLock( m_hRefListMutex);
	m_pListManager->insertLast( 0, pTrackingRec);
	f_mutexUnlock( m_hRefListMutex);

	pStack = (void **)pTrackingRec->getStack();
	getCallStack( pStack, CTRC_STACK_SIZE, 1);

Exit:

	return;
}

/****************************************************************************
Desc:	This reference has been released, don't track it any more.
****************************************************************************/
void F_ObjRefTracker::untrackRef(
	void *					pReferenceID,
	void *					pSubrefID)
{
	F_TrackingRecord *	pTrackingRec = NULL;
	FLMBOOL					bListLocked = FALSE;

	if( m_hRefListMutex == F_MUTEX_NULL)
	{
		goto Exit;
	}

	if( !pReferenceID)
	{
		goto Exit;
	}
	
	// Lock the list
	
	f_mutexLock( m_hRefListMutex);
	bListLocked	= TRUE;
	
	// Try to find the reference in the list
	
	pTrackingRec = (F_TrackingRecord *) m_pListManager->getItem( 0, 0);
	while( pTrackingRec)
	{
		if( pTrackingRec->getReferenceID() == pReferenceID
		 && pTrackingRec->getSubrefID() == pSubrefID)
		{
			// The reference has been found.
			
			pTrackingRec->removeFromList();
			pTrackingRec->Release();
			break;
		}
		
		pTrackingRec = (F_TrackingRecord *) pTrackingRec->getNextListItem();
	}
	
	if( !pTrackingRec)
	{
		// The reference was never tracked.  This isn't supposed to happen.
		
		char	pucMessage[100];
		
		f_sprintf( pucMessage, 
			"untrackRef: Reference %x.%x was not tracked", 
			(unsigned)((FLMUINT)pReferenceID), (unsigned)((FLMUINT)pSubrefID));
		logError( pucMessage);
		logError( "\tModify code to track this reference");
		goto Exit;
	}
	
Exit:

	if( bListLocked)
	{
		f_mutexUnlock( m_hRefListMutex);
	}
}

/****************************************************************************
Desc: Check the list for references that were never released.
****************************************************************************/
void F_ObjRefTracker::checkForUnreleasedRefs(
	FLMUINT *				puiCount)
{
	RCODE						rc = NE_FLM_OK;
	F_TrackingRecord *	pTrackingRec;
	FLMUINT64				ui64FileCursor;
	FLMUINT					uiLoop;
	char						pucSymbol[ 125];
	char						pucBuffer[ 150];
	FLMBOOL					bHeaderDisplayed;
	IF_FileHdl *			pFileHdl = NULL;
	FLMBOOL					bListLocked = FALSE;
	FLMUINT					uiCount = 0;

	if( m_hRefListMutex == F_MUTEX_NULL)
	{
		logError( "checkForUnreleasedReferences: Reference tracking "
					 "was not initialized");
		goto Exit;
	}

	if( m_pFileSystem)
	{
		if( RC_BAD( rc = m_pFileSystem->openFile( m_pLogPath,
			FLM_IO_RDWR | FLM_IO_EXCL | FLM_IO_SH_DENYNONE,
			&pFileHdl)))
		{
			if( RC_BAD( rc = m_pFileSystem->createFile( m_pLogPath,
				FLM_IO_RDWR | FLM_IO_EXCL | FLM_IO_SH_DENYNONE,
				&pFileHdl)))
			{
				goto Exit;
			}
		}
	}

	// Find EOF so text can be appended to the trace file.
	
	if( pFileHdl)
	{
		if( RC_BAD( rc = pFileHdl->size( &ui64FileCursor)))
		{
			goto Exit;
		}
	}

	// Lock the list
	
	f_mutexLock( m_hRefListMutex);
	bListLocked	= TRUE;
	bHeaderDisplayed = FALSE;
	
	// Process all unreleased references
	
	for( pTrackingRec = (F_TrackingRecord *)m_pListManager->getItem( 0, 0);
   		pTrackingRec;
			pTrackingRec = (F_TrackingRecord *)m_pListManager->getItem( 0, 0))
	{
		void **		pStack;

		uiCount++;
		if( !bHeaderDisplayed)
		{
			f_sprintf( pucBuffer, "Unreleased references of type [%s]\n",
				m_pszObjName);

			if( RC_BAD( rc = logMessage( pucBuffer, pFileHdl, &ui64FileCursor)))
			{
				goto Exit;
			}
			
			bHeaderDisplayed = TRUE;
		}
		
		if( RC_BAD( rc = logMessage( " ", pFileHdl, &ui64FileCursor)))
		{
			goto Exit;
		}

		f_sprintf( pucBuffer, "    Unreleased reference (%X.%X) from thread: %X\n",
			(unsigned)((FLMUINT)pTrackingRec->getReferenceID()),
			(unsigned)((FLMUINT)pTrackingRec->getSubrefID()),
			(unsigned) pTrackingRec->getThreadID());
		
		if( RC_BAD( rc = logMessage( pucBuffer, pFileHdl, &ui64FileCursor)))
		{
			goto Exit;
		}

		pStack = (void **) pTrackingRec->getStack();
		for( uiLoop = 0; pStack[ uiLoop]; uiLoop++ )
		{
			formatAddress( pucSymbol, sizeof( pucSymbol), pStack[ uiLoop]);
			f_sprintf( pucBuffer, "        %-45.45s [addr = %8.8x]\n", pucSymbol,
				(unsigned)((FLMUINT)pStack[ uiLoop]));

			if( RC_BAD( rc = logMessage( pucBuffer, pFileHdl, &ui64FileCursor)))
			{
				goto Exit;
			}
		}
		
		m_pListManager->removeItem( 0, pTrackingRec);
	}

Exit:

	if( bListLocked)
	{
		f_mutexUnlock( m_hRefListMutex);
	}

	if( pFileHdl)
	{
		pFileHdl->Release();
	}

	if( puiCount)
	{
		*puiCount = uiCount;
	}
}

/****************************************************************************
Desc:	Sets the address coonversion callback function
****************************************************************************/
void F_ObjRefTracker::setAddressFormatter(
	ADDR_FMT_HOOK 		pFunc,
	void *				pvUserData)
{
	m_pAddrFmtHook = pFunc;
	m_pUserData = pvUserData;
}

/****************************************************************************
Desc:	Converts a return address to displayable format
****************************************************************************/
void F_ObjRefTracker::formatAddress(
	char *				pucBuffer,
	FLMUINT				uiSize,
	void *				pAddress)
{
#ifdef FLM_WIN
	PIMAGEHLP_SYMBOL	pihs = NULL;
#ifdef FLM_64BIT
	DWORD64				displacement;
#else
	DWORD					displacement;
#endif
	RCODE					rc = NE_FLM_OK;
#endif

	if( m_pAddrFmtHook)
	{
		pucBuffer[ 0] = '\0';
		m_pAddrFmtHook( this, pAddress, (FLMBYTE *)pucBuffer, uiSize, m_pUserData);
		return;
	}

#if defined( FLM_RING_ZERO_NLM)

	if( uiSize == 0)
	{
		return;
	}

	GetClosestSymbol( (BYTE *)pucBuffer, (LONG)pAddress);
	return;

#elif defined( FLM_WIN)

	if( RC_OK( rc = f_alloc( sizeof( IMAGEHLP_SYMBOL) + 100, &pihs)))
	{
		pihs->SizeOfStruct = sizeof(IMAGEHLP_SYMBOL);
		pihs->Address = (FLMUINT)pAddress; //stackFrame.AddrPC.Offset;
		pihs->MaxNameLength = (FLMINT32)uiSize;

		if ( SymGetSymFromAddr( GetCurrentProcess(),	(FLMUINT)pAddress,
										&displacement, pihs ) )
		{
			wsprintf( pucBuffer, "%s + %X",
				(char *)pihs->Name, (unsigned)displacement);
		}
		else
		{
			wsprintf( pucBuffer, "0x%08X", (unsigned)((FLMUINT)pAddress));
		}
	}
	else
	{
		wsprintf( pucBuffer, "0x%08X", (unsigned)((FLMUINT)pAddress));
	}

	f_free( &pihs);

#else
	#ifdef HAVE_DLADDR
		Dl_info dlip;
		if (dladdr(pAddress, &dlip) != 0 && dlip.dli_sname)
		{
			const char *filename;
			if (dlip.dli_saddr != pAddress)
			{
				filename = strrchr(dlip.dli_fname, '/');
				if (!filename)
					filename = dlip.dli_fname;
				else
					filename++;		// skip over slash
				f_sprintf( pucBuffer, "0x%08x (%s)", (unsigned)((FLMUINT)pAddress),
							filename); 
			}
			else
				f_sprintf( pucBuffer, "%s", dlip.dli_sname);
			return;
		}
	#endif
		f_sprintf( pucBuffer, "0x%08x", (unsigned)((FLMUINT)pAddress));
#endif
}

/****************************************************************************
Desc:	Walk the BP chain down the call stack, gathering return addresses.
****************************************************************************/
#if defined( FLM_WIN) && !defined( FLM_64BIT)
void F_ObjRefTracker::getCallStack(
	void *						stack[],
	FLMUINT						uiCount,
	FLMUINT						uiSkip)
{
	STACKFRAME					stackFrame;
	FLMINT32						ui32LastBP;
	FLMUINT						uiLoop;

	F_UNREFERENCED_PARM( uiSkip);
	ZeroMemory( (PVOID)&stackFrame, sizeof(STACKFRAME) );

	// TDOMAN: do this in assembly since we aren't sure we can rely on the
	//			  GetThreadContext and StackWalk API's
	_asm
	{
		mov	ui32LastBP, ebp						// save off next bp
	}

	// while you can continue walking the stack...
	for ( uiLoop = 0; uiLoop < uiCount; uiLoop++ )
	{
		// TDOMAN: we have to walk the stack ourselves since the VC4x API's
		//			  don't appear to be consistently reliable.
		__try
		{
			// TDOMAN: don't crash if the last bp wasn't want we expected
			_asm
			{
				push	esi
				push	edi
				mov	edi, ui32LastBP
				mov	esi, [edi]
				cmp	esi, edi
				 jbe	Done
				mov	ui32LastBP, esi				// save off next bp
				mov	esi, [edi + 4]
				mov	stackFrame, esi			// setup AddrPC
				pop	edi
				pop	esi
			}
		}
		__except (EXCEPTION_EXECUTE_HANDLER)
		{
			// If you do want to get these exceptions, you can turn off stack
			//	walking by setting fStackWalk to FALSE.
			goto Done;
		}
		stack[ uiLoop] = (void *)(FLMUINT)stackFrame.AddrPC.Offset;
	}

Done:
	stack[ uiLoop] = (void *)0;
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_64BIT)
void F_ObjRefTracker::getCallStack(
	void *						stack[],
	FLMUINT,						//uiCount,
	FLMUINT)						//uiSkip)
{
	// Not supported on this platform
	stack[0] = (void *)0;
}
#endif

/****************************************************************************
****************************************************************************/
#if defined( FLM_NLM)

	void * DMGetEBP(void);

#if defined( __MWERKS__)


	void *DMGetEBP(void)
	{
		__asm
		{
			mov	eax,[ebp]
		}
	}  // end of assembly

#else

	#pragma aux DMGetEBP = "mov eax,ebp";

#endif

/****************************************************************************
Desc:
****************************************************************************/
void * DMValueAtStackOffset(void *pos, int offset);

#if defined( __MWERKS__)

	void *DMValueAtStackOffset(void *, int )
	{
		__asm
		{
			mov	eax,[ebp+0x8]
			mov	ebx,[ebp+0xC]
			mov	eax,ss:[eax+ebx]
		}
	}

#else

	#pragma aux DMValueAtStackOffset = "mov eax,ss:[eax+ebx]" parm [eax] [ebx];

#endif

/****************************************************************************
Desc: Traces back COUNT entries on the call stack, storing
		them in STACK. Note that this code requires that DS be build using the 
		Watcom /of+ option, (or equivalent) to generate traceable stack frames by
		emitting prelude code for every function that looks something like that
		of the SubRoutine below:
		
		Caller:
			push Parms						; caller pushes parameters
			call SubRoutine				; caller pushes his own return address
			add esp, parmSize				; caller clears parameters from stack
			...
		SubRoutine:		
			push ebp							; pushes caller's frame pointer
			mov ebp, esp					; creates SubRoutine's frame pointer
			...
			pop ebp							; restores caller's frame pointer
			ret								; returns to caller
		
		In this scheme, the MOV instruction in the prelude code always sets EBP 
		pointing to the PUSHed value of the previous (caller's) frame pointer 
		(see the first instruction in SubRoutine above). We discard the first
		'skipCount' + 1 return addresses because we aren't interested in the fact
		that the caller called DMAlloc, which called DMAllocFromTag,
		which called getCallStack. 
		The stack trace loop terminates when it detects a return address that is 
		outside of NDS code space (start and limit are stored in the load def 
		struct - module handle). Some inline assembly is used to access the stack 
		like data (see DMGetEBP and DMValueAtStackOffset above).
****************************************************************************/
void F_ObjRefTracker::getCallStack(
	void *		stack[],
	FLMUINT		uiCount,
	FLMUINT		uiSkipCount)
{
	FLMUINT		uiLoop;
 	void *		rtnAddr;
	void *		ebp = DMGetEBP();

	while( uiSkipCount--)
	{
		ebp = DMValueAtStackOffset( ebp, 0);
	}
	rtnAddr = DMValueAtStackOffset( ebp, 4);

	for( uiLoop = 0; --uiCount; )
	{
		void *oldebp;
		
		stack[ uiLoop++] = rtnAddr;
		if( !ebp)
		{
			break;
		}
		
		oldebp = ebp;
		ebp = DMValueAtStackOffset( ebp, 0);				// Caller's frame ptr

		if ( !ebp || ebp <= oldebp || ebp > (void *)((char *)oldebp+3000))
		{
			break;
		}

		rtnAddr = DMValueAtStackOffset( ebp, 4);			// Caller's return addr
	}
	stack[ uiLoop] = 0;
	return;
}
#endif // defined( FLM_NLM)

/****************************************************************************
Desc: Log an error message
****************************************************************************/
void F_ObjRefTracker::logError(
	const char *	pucMessage)
{
	char				pucBuffer[ 120];
	FLMUINT64		ui64Dummy = 0;

	f_sprintf( pucBuffer, "Error: %s", pucMessage);
	logMessage( pucBuffer, NULL, &ui64Dummy);
	flmAssert(0);
}

/****************************************************************************
Desc: Log a message to the trace file and to the DS trace screen
****************************************************************************/
RCODE F_ObjRefTracker::logMessage(
	const char *	message,
	IF_FileHdl *	pFileHdl,
	FLMUINT64 *		pui64FileCursor)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiBytesWritten;
	FLMBOOL			bFileOpened = FALSE;
	const char *	pCarriageReturn = "\n";
	
	if( !pFileHdl && m_pFileSystem)
	{
		if( RC_BAD( rc = m_pFileSystem->openFile(
			m_pLogPath, FLM_IO_RDWR | FLM_IO_SH_DENYNONE, &pFileHdl)))
		{
			if( RC_BAD( rc = m_pFileSystem->createFile(
				m_pLogPath, FLM_IO_RDWR | FLM_IO_EXCL | FLM_IO_SH_DENYNONE,
				&pFileHdl)))
			{
				goto Exit;
			}
		}

		bFileOpened = TRUE;
		flmAssert( pFileHdl);

		// Find EOF so text can be appended to the trace file.
		
		if( RC_BAD( rc = pFileHdl->size( pui64FileCursor)))
		{
			goto Exit;
		}
	}

	if( pFileHdl)
	{
		if( RC_BAD( rc = pFileHdl->write(
			*pui64FileCursor, f_strlen( message), (void *)message,
			&uiBytesWritten)))
		{
			goto Exit;
		}
		
		(*pui64FileCursor) += uiBytesWritten;

		if( RC_BAD( rc = pFileHdl->write(
			*pui64FileCursor, f_strlen( pCarriageReturn), 
			(void *)pCarriageReturn, &uiBytesWritten)))
		{
			(*pui64FileCursor) += uiBytesWritten;
		}
	}

Exit:

	if( bFileOpened)
	{
		pFileHdl->Release();
	}

	return( rc);
}

#undef	new
#undef	delete
/****************************************************************************
Desc:	
****************************************************************************/
void * F_Object::operator new(
	FLMSIZET			uiSize,
	const char *	pszFile,
	int				iLine)
#ifndef FLM_WATCOM_NLM
		throw()
#endif
{
	void *	pvReturnPtr = NULL;

	f_allocImp( uiSize, &pvReturnPtr, TRUE, pszFile, iLine);
	return( pvReturnPtr);
}

/****************************************************************************
Desc:	
****************************************************************************/
void * F_Object::operator new(
	FLMSIZET			uiSize)
#ifndef FLM_WATCOM_NLM
		throw()
#endif
{
	void *	pvReturnPtr = NULL;

	f_allocImp( uiSize, &pvReturnPtr, TRUE, NULL, 0);
	return( pvReturnPtr);
}

/****************************************************************************
Desc:	
****************************************************************************/
void * F_Object::operator new[](
	FLMSIZET			uiSize,
	const char *	pszFile,
	int				iLine)
#ifndef FLM_WATCOM_NLM
		throw()
#endif
{
	void *	pvReturnPtr = NULL;

	f_allocImp( uiSize, &pvReturnPtr, TRUE, pszFile, iLine);
	return( pvReturnPtr);
}

/****************************************************************************
Desc:	
****************************************************************************/
void * F_Object::operator new[](
	FLMSIZET			uiSize)
#ifndef FLM_WATCOM_NLM
		throw()
#endif
{
	void *	pvReturnPtr = NULL;

	f_allocImp( uiSize, &pvReturnPtr, TRUE, NULL, 0);
	return( pvReturnPtr);
}

/****************************************************************************
Desc:	
****************************************************************************/
void F_Object::operator delete(
	void *			ptr)
{
	if( !ptr)
	{
		return;
	}

	f_freeImp( &ptr, TRUE);
}

/****************************************************************************
Desc:	
****************************************************************************/
void F_Object::operator delete[](
	void *			ptr)
{
	if( !ptr)
	{
		return;
	}

	f_freeImp( &ptr, TRUE);
}

/****************************************************************************
Desc:	
****************************************************************************/
#if !defined( FLM_WATCOM_NLM)
void F_Object::operator delete(
	void *			ptr,
	const char *,	// file
	int)				// line
{
	if( !ptr)
	{
		return;
	}

	f_freeImp( &ptr, TRUE);
}
#endif

/****************************************************************************
Desc:	
****************************************************************************/
#if !defined( FLM_WATCOM_NLM)
void F_Object::operator delete[](
	void *			ptr,
	const char *,	// file
	int)				// line
{
	if( !ptr)
	{
		return;
	}

	f_freeImp( &ptr, TRUE);
}
#endif

/****************************************************************************
Desc:	
****************************************************************************/
void * F_OSBase::operator new(
	FLMSIZET			uiSize,
	const char *,	// pszFile,
	int)				// iLine)
#ifndef FLM_WATCOM_NLM
		throw()
#endif
{
	return( os_malloc( uiSize));
}

/************************************************************************
Desc:	
*************************************************************************/
void F_OSBase::operator delete(
	void *			ptr)
{
	os_free( ptr);
}

/************************************************************************
Desc:	
*************************************************************************/
void F_OSBase::operator delete[](
	void *			ptr)
{
	os_free( ptr);
}

/****************************************************************************
Desc:	
****************************************************************************/
#if !defined( FLM_WATCOM_NLM)
void F_OSBase::operator delete(
	void *			ptr,
	const char *,	// file
	int)				// line
{
	os_free( ptr);
}
#endif

/****************************************************************************
Desc:	
****************************************************************************/
#if !defined( FLM_WATCOM_NLM)
void F_OSBase::operator delete[](
	void *			ptr,
	const char *,	// file
	int)				// line
{
	os_free( ptr);
}
#endif
