//-------------------------------------------------------------------------
// Desc:	Various internal structure definitions.
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

#ifndef FSTRUCTS_H
#define FSTRUCTS_H

struct FFILE;
struct QUERY_HDR;
struct CDL;
struct FDICT;
struct CP_INFO;
struct IFD;

/**************************************************************************
	Typedefs for structure pointers.
**************************************************************************/

// Typedefs for the http callback function and for the functions that register
// and deregister the http callback function.

typedef void * FLM_MODULE_HANDLE;

typedef int (* URLHandler)(
	HRequest *		pHRequest,
	void *			pvUserData); 

typedef int (* REG_URL_HANDLER_FN)(
	FLM_MODULE_HANDLE	hModule, 
	const char *		pszURL,
	unsigned				uiFlags,
	URLHandler			fnHandler,
	const char *		pszTitle,
	void *				pvUserData);

typedef int (* DEREG_URL_HANDLER_FN)(
	const char *	pszURL, 
	URLHandler		fnHandler);

typedef const char * (* REQ_PATH_FN)(
	HRequest *		pHRequest);

typedef const char * (* REQ_QUERY_FN)(
	HRequest *		pHRequest);

typedef const char * (* REQ_HDR_VALUE_FN)(
	HRequest *		pHRequest,
	const char *	pszName);

typedef int (* SET_HDR_VAL_FN)(
	HRequest *		pHRequest,
	const char *	pszName,
	const char *	pszValue);

typedef int (* PRINTF_FN)(
	HRequest *		pHRequest,
	const char *	pszFormat,
	... );

typedef int (* EMIT_FN)(
	HRequest *		pHRequest);

typedef void (* SET_NO_CACHE_FN)(
	HRequest *		pHRequest,
	const char *	pszHeader);

typedef int (* SEND_HDR_FN)(
	HRequest *		pHRequest,
	int				iStatus);

typedef int (* SET_IO_MODE_FN)(
	HRequest *		pHRequest,
	int				bRaw,
	int				bOutput);

typedef int (* SEND_BUFF_FN)(
	HRequest *		hRequest,
	const void *	pvBuf,
	FLMSIZET			bufsz);

typedef void * (* ACQUIRE_SESSION_FN)(
	HRequest *		pHRequest);

typedef void (* RELEASE_SESSION_FN)(
	void *			pvHSession);

typedef void * (* ACQUIRE_USER_FN)(
	void *			pvHSession,
	HRequest *		pHRequest);

typedef void (* RELEASE_USER_FN)(
	void *			pvHUser);

typedef int (* SET_SESSION_VALUE_FN)(
	void *			pvHSession,
	const char *	pcTag,
	const void *	pvData,
	FLMSIZET			uiSize);

typedef int (* GET_SESSION_VALUE_FN)(
	void *			pvHSession,
	const char *	pcTag,
	void *			pvData,
	FLMSIZET *		puiSize);

typedef int (* GET_GBL_VALUE_FN)(
	const char *	pcTag,
	void *			pvData,
	FLMSIZET *		puiSize);

typedef int (* SET_GBL_VALUE_FN)(
	const char *	pcTag,
	const void *	pvData,
	FLMSIZET			uiSize);

typedef int (* RECV_BUFFER_FN)(
	HRequest *		pHRequest,
	void *			pvBuf,
	FLMSIZET *		puiBufSize);

// These are flags that are passed to the http server during registration.
// They're copied verbatum from John Calcote's code

#define HR_STK_BOTH		0x0000	/*	-- MUTUALLY EXCLUSIVE --			*/
#define HR_STK_NOTLS		0x0001	/* register ONLY with CLEAR stack	*/
#define HR_STK_NOCLEAR	0x0002	/* register ONLY with TLS stack		*/
#define HR_STK_MASK		0x000F

// Which level of authentication is required - the default is NONE

#define HR_AUTH_NONE		0x0000	/*	-- NOT MUTUALLY EXCLUSIVE --		*/
#define HR_AUTH_USER		0x0010	/* user authentication required		*/
#define HR_AUTH_SADMIN	0x0020	/* SADMIN authentication required	*/
#define HR_AUTH_USERSA	(HR_AUTH_USER|HR_AUTH_SADMIN)
#define HR_AUTH_MASK		0x00F0

// Which authentication realm should be used - the default is none

#define HR_REALM_NONE	0x0000	/* -- MUTUALLY EXCLUSIVE --			*/
#define HR_REALM_HFIO	0x0100	/* http file through .htaccess		*/
#define HR_REALM_NDS		0x0200	/* nds through dclient					*/
#define HR_REALM_LDAP	0x0300	/* directory through ldap				*/
#define HR_REALM_MASK	0xFF00


// These are also copied verbatum from John Calcote's code...
/* HTTP Response Status Codes */

/* 100-level codes, informational */
#define HTS_CONTINUE					100	/* 1.1 */
#define HTS_SWITCH_PROT				101	/* 1.1 */

/* 200-level codes, success */
#define HTS_OK							200	/* 1.0 */
#define HTS_CREATED					201	/* 1.0 */
#define HTS_ACCEPTED					202	/* 1.0 */
#define HTS_NON_AUTH_INFO			203	/* 1.1 */
#define HTS_NO_CONTENT				204	/* 1.0 */
#define HTS_RESET_CONTENT			205	/* 1.1 */
#define HTS_PARTIAL_CONTENT		206	/* 1.1 */

/* 300-level codes, redirection */
#define HTS_MULTIPLE_CHOICES		300	/* 1.1 */
#define HTS_MOVED_PERM				301	/* 1.0 */
#define HTS_MOVED_TEMP				302	/* 1.0 */
#define HTS_FOUND						302	/* 1.1 */
#define HTS_SEE_OTHER				303	/* 1.1 */
#define HTS_NOT_MODIFIED			304	/* 1.0 */
#define HTS_USE_PROXY				305	/* 1.1 */
#define HTS_TEMP_REDIRECT			307	/* 1.1 */

/* 400-level codes, client error */
#define HTS_BAD_REQUEST				400	/* 1.0 */
#define HTS_UNAUTHORIZED			401	/* 1.0 */
#define HTS_PAYMENT_REQ				402	/* 1.1 */
#define HTS_FORBIDDEN				403	/* 1.0 */
#define HTS_NOT_FOUND				404	/* 1.0 */
#define HTS_METH_NOT_ALLOWED		405	/* 1.1 */
#define HTS_NOT_ACCEPTABLE			406	/* 1.1 */
#define HTS_PROXY_AUTH_REQ			407	/* 1.1 */
#define HTS_REQUEST_TIMEOUT		408	/* 1.1 */
#define HTS_CONFLICT					409	/* 1.1 */
#define HTS_GONE						410	/* 1.1 */
#define HTS_LENGTH_REQ				411	/* 1.1 */
#define HTS_PRECOND_FAILED			412	/* 1.1 */
#define HTS_REQ_ENT_TOO_LARGE		413	/* 1.1 */
#define HTS_REQ_URI_TOO_LARGE		414	/* 1.1 */
#define HTS_BAD_MEDIA_TYPE			415	/* 1.1 */
#define HTS_BAD_REQ_RANGE			416	/* 1.1 */
#define HTS_EXPECTATION_FAILED	417	/* 1.1 */

/* 500-level codes, server error */
#define HTS_INTERNAL_ERROR			500	/* 1.0 */
#define HTS_NOT_IMPLEMENTED		501	/* 1.0 */
#define HTS_BAD_GATEWAY				502	/* 1.0 */
#define HTS_SERVICE_UNAVAIL		503	/* 1.0 */
#define HTS_GATEWAY_TIMEOUT		504	/* 1.1 */
#define HTS_BAD_HTTP_VERSION		505	/* 1.1 */

// Authentication Levels  - again, copied from John Calcote: httpdefs.h

#define HAL_NONE		0
#define HAL_USER		1
#define HAL_SADMIN	2

// Flags for the uiQsortFlags parameter - used in sorting/indexing.

#define KY_DUP_CHK_SRT			0x01 
							// Sort: LFD, KEY, Action
#define KY_FINAL_SRT 			0x02
							// Sort: Database, LFD, KEY, SeqNum
#define KY_DUPS_FOUND			0x04
							// Dups were found in DUP_CHK_SRT

// Flags for the uiKrAction parameter - used in sorting/indexing.

#define KREF_DEL_KEYS							0x01
#define KREF_ADD_KEYS							0x02
#define KREF_INDEXING_ONLY						0x04
#define KREF_IN_MODIFY							0x10
#define KREF_MISSING_KEYS_OK					0x20

#include "fpackon.h"

// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

#define 				LOG_HEADER_SIZE 			512
#define				LOG_HEADER_SIZE_VER40	88

// C/S Address types

#define FLM_CS_NO_ADDR							0x00
#define FLM_CS_IP_ADDR							0x01
#define FLM_CS_STREAM_ADDR						0x02

// Define the overhead space required to manage encrypted fields.

#define FLM_ENC_FLD_OVERHEAD					11

#define FLD_ENC_FLAGS_OFFSET					0
#define FLD_ENC_ENCID_OFFSET					1
#define FLD_ENC_DATA_LEN_OFFSET				3
#define FLD_ENC_ENCRYPTED_DATA_LEN_OFFSET	7

// C/S Address sizes

#define FLM_CS_MAX_ADDR_LEN					128

typedef void * (* ALLOC_PAGE_FUNC)(
	FLMUINT	uiSizeToAllocate);

typedef FLMUINT (* FREE_PAGE_FUNC)(
	void *	pvPage);

typedef FLMUINT (* PROTECT_PAGE_FUNC)(
	void *	pvPage);

typedef FLMUINT (* UNPROTECT_PAGE_FUNC)(
	void *	pvPage);

typedef FLMUINT (* SET_DBG_PAGE_WRITER_FUNC)(
	FLMUINT	uiThreadId);

/****************************************************************************
Desc:	 	This is a debug only structure that is used to keep track of the
			threads that are currently using a block.
****************************************************************************/
#ifdef FLM_DEBUG
typedef struct SCACHE_USE
{
	SCACHE_USE *	pNext;			// Pointer to next SCACHE_USE structure in
											// the list.
	FLMUINT			uiThreadId;		// Thread ID of thread using the block.
	FLMUINT			uiUseCount;		// Use count for this particular thread.
} SCACHE_USE;
#endif

/****************************************************************************
Desc:	 	This is the header structure for a cached data block.
****************************************************************************/
typedef struct SCACHE
{
	SCACHE *		pPrevInFile;		// This is a pointer to the previous block
											// in the linked list of blocks that are
											// in the same file.
	SCACHE *		pNextInFile;		// This is a pointer to the next block in
											// the linked list of blocks that are in
											// the same file.
	FLMBYTE *	pucBlk;				// Pointer to this block's data.  The
											// block's data is allocated with the
											// SCACHE structure and immediately follows
											// the structure.  We keep a pointer to it
											// so we are not always having to calculate
											// its address.
	FFILE *		pFile;				// Pointer to the file this data block
											// belongs to.
	FLMUINT		uiBlkAddress;		// Block address.
	SCACHE *		pPrevInGlobalList;
											// This is a pointer to the previous block
											// in the global linked list of cache
											// blocks.  The previous block is more
											// recently used than this block.
	SCACHE *		pNextInGlobalList;
											// This is a pointer to the next block in
											// the global linked list of cache blocks.
											// The next block is less recently used
											// than this block.
	SCACHE *		pPrevInReplaceList;
											// This is a pointer to the previous block
											// in the global linked list of cache
											// blocks that have a flags value of zero.
	SCACHE *		pNextInReplaceList;
											// This is a pointer to the next block in
											// the global linked list of cache blocks
											// that have a flags value of zero.
	SCACHE *		pPrevInHashBucket;
											// This is a pointer to the previous block
											// in the linked list of blocks that are
											// in the same hash bucket.
	SCACHE *		pNextInHashBucket;
											// This is a pointer to the next block in
											// the linked list of blocks that are in
											// the same hash bucket.
	SCACHE *		pPrevInVersionList;
											// This is a pointer to the previous block
											// in the linked list of blocks that are
											// all just different versions of the
											// same block.  The previous block is a
											// more recent version of the block.
	SCACHE *		pNextInVersionList;
											// This is a pointer to the next block in
											// the linked list of blocks that are all
											// just different versions of the same
											// block.  The next block is an older
											// version of the block.
	F_NOTIFY_LIST_ITEM *	pNotifyList;
											// This is a pointer to a list of threads
											// that want to be notified when a pending
											// I/O is complete.  This pointer is only
											// non-null if the block is currently being
											// read from disk and there are multiple
											// threads all waiting for the block to
											// be read in.
	FLMUINT		uiHighTransID;		// This indicates the highest known moment
											// in the file's update history when this
											// version of the block was the active
											// block.  NOTE: The block's low trans ID
											// is retrieved by calling the
											// SCACHE_LOW_CHECKPOINT macro - see below.
											// A block's low transaction ID and high
											// transaction ID indicate a span of
											// transactions where this version of the
											// block was the active version of the
											// block.
	FLMUINT		uiUseCount;			// Number of times this block has been
											// retrieved for use by threads.  A use
											// count of zero indicates that no thread
											// is currently using the block.  Note that
											// a block cannot be replaced when its use
											// count is non-zero.
	FLMUINT16	ui16Flags;			// This is a set of flags for the block
											// that indicate various things about the
											// block's current state.
		#define CA_DIRTY			0x0001
											// This bit indicates that the block is
											// dirty and needs to be flushed to disk.
											// NOTE: For 3.x files, this bit may remain
											// set on prior versions of blocks until the
											// current transaction commits.
		#define CA_WRITE_INHIBIT	0x0002
											// Must not write block until use count
											// goes to zero.  NOTE: Can ignore when
											// in the checkpoint thread.
		#define CA_READ_PENDING	0x0004
											// This bit indicates that the block is
											// currently being read in from disk.
		#define CA_WRITE_TO_LOG	0x0008
											// This bit indicates that this version of
											// the block should be written to the
											// rollback log before being replaced.
											// During an update transaction, the first
											// time a block is updated, FLAIM will
											// create a new version of the block and
											// insert it into cache.  The prior version
											// of the block is marked with this flag
											// to indicate that it needs to be written
											// to the log before it can be replaced.
		#define CA_LOG_FOR_CP	0x0010
											// This bit indicates that this version of
											// the block needs to be logged to the
											// physical rollback in order to restore
											// the last checkpoint.  This is only
											// applicable to 3.x files.
		#define CA_WAS_DIRTY		0x0020
											// This bit indicates that this version of
											// the block was dirty before the newer
											// version of the block was created.
											// Its dirty state should be restored if
											// the current transaction aborts.  This
											// flag is only used for 3.x files.
		#define CA_WRITE_PENDING	0x0040
											// This bit indicates that a block is in
											// the process of being written out to
											// disk.
		#define CA_IN_WRITE_PENDING_LIST	0x0080
											// This bit indicates that a block is in
											// the write pending list.
		#define CA_FREE				0x0100
											// The block has been linked to the free
											// list (and unlinked from all other lists)
		#define CA_IN_FILE_LOG_LIST		0x0200
											// Block is in the list of blocks that may
											// have one or more versions that need to
											// be logged
		#define CA_IN_NEW_LIST		0x0400
											// Dirty block that is beyond the last CP EOF
		#define CA_DUMMY_FLAG		0x0800
											// Used to prevent blocks from being linked
											// into the replace list in cases where
											// they will be removed immediately (because
											// a bit is going to being set)
	FLMUINT16	ui16BlkSize;		// Block size

// NOTE: Keep debug items at the END of the structure.

#ifdef FLM_DEBUG
	FLMUINT			uiChecksum;		// Checksum for the block and header.
	SCACHE_USE *	pUseList;		// This is a pointer to a list of threads
											// that are currently using the block.
#endif
} SCACHE;

/****************************************************************************
Desc:	 	This is the structure that will be embedded in the FLMSYSDATA 
			structure to manage cache.
****************************************************************************/
typedef struct SCACHE_MGR
{
	SCACHE *				pMRUCache;	// This is a pointer to the
											// most-recently used cache block.  It
											// is essentially a pointer to the head
											// of a linked list of cache blocks,
											// linked from most-recently-used to
											// least-recently-used.
	SCACHE *				pLRUCache;	// This is a pointer to the
											// least-recently used cache block.  It
											// is essentially a pointer to the tail
											// of a linked list of cache blocks,
											// linked from most-recently-used to
											// least-recently-used.
	SCACHE *				pMRUReplace;// Pointer to the MRU end of the list
											// of cache items with no flags set.
	SCACHE *				pLRUReplace;// Pointer to the LRU end of the list
											// of cache items with no flags set.
	SCACHE *				pFirstFree;
											// Pointer to a linked list of cache
											// blocks that need to be freed.
											// Cache blocks in this list are no
											// longer associated with a file and can
											// be freed or re-used as needed.  They
											// are linked using the pNextInFile and
											// pPrevInFile pointers.
	SCACHE *				pLastFree;
											// Pointer to a linked list of cache
											// blocks that need to be freed.
											// Cache blocks in this list are no
											// longer associated with a file and can
											// be freed or re-used as needed.  They
											// are linked using the pNextInFile and
											// pPrevInFile pointers.
	SCACHE **			ppHashTbl;	// This is a pointer to a hash table that
											// is used to find cache blocks.  Each
											// element in the table points to a
											// linked list of SCACHE structures that
											// all hash to the same hash bucket.
	FLM_CACHE_USAGE	Usage;		// Contains maximum, bytes used, etc.
	FLMUINT				uiFreeBytes;// Number of free bytes
	FLMUINT				uiFreeCount;// Number of free blocks
	FLMUINT				uiReplaceableCount;
											// Number of blocks whose flags are 0
	FLMUINT				uiReplaceableBytes;
											// Number of bytes belonging to blocks whose
											// flags are 0
	FLMBOOL				bAutoCalcMaxDirty;
											// Flag indicating we should automatically
											// calculate maximum dirty cache.
	FLMUINT				uiMaxDirtyCache;
											// Maximum cache that can be dirty.
	FLMUINT				uiLowDirtyCache;
											// When maximum dirty cache is exceeded,
											// threshhold it should be brought back
											// under
	FLMUINT				uiTotalUses;// Total number of uses currently held
											// on blocks in cache.
	FLMUINT				uiBlocksUsed;
											// Total number of blocks in cache that
											// are being used.
	FLMUINT				uiPendingReads;
											// Total reads currently pending.
	FLMUINT				uiIoWaits;	// Number of times multiple threads
											// were reading the same block from
											// disk at the same time.
	FLMUINT				uiHashTblSize;
											// This contains the number of buckets
											// in the hash table.
	FLMUINT				uiHashTblBits;
											// Number of bits that are significant
											// for the hash table size.
	IF_FixedAlloc *	pSCacheAllocator;
											// Fixed size allocator for SCACHE structures
	IF_BlockAlloc *	pBlockAllocators[ 2];	
											// Fixed size allocators for cache blocks
											// We only support 4K and 8K blocks

#ifdef FLM_DEBUG
	FLMBOOL				bDebug;		// Enables checksumming and cache use
											// monitoring.  Only available when
											// debug is compiled in.
#endif

} SCACHE_MGR;

/****************************************************************************
Desc:	 	This structure is used to manage a particular record in the
			FLAIM record cache.
****************************************************************************/
typedef struct RCACHE
{
	FlmRecord *			pRecord;				// Pointer to record object in cache.
	FFILE *				pFile;				// Pointer to the file this record
													// belongs to.
	FLMUINT				uiContainer;		// Container the record comes from
	FLMUINT				uiDrn;				// Data record number.
	FLMUINT				uiLowTransId;		// Low transaction ID for this record
	FLMUINT				uiHighTransId;		// High transaction ID for this record.
	RCACHE *				pNextInBucket;		// Next record cache object in linked
													// list off of record hash bucket.
	RCACHE *				pPrevInBucket;		// Prev record cache object in linked
													// list off of record hash bucket.
	RCACHE *				pNextInFile;		// Next record cache object in linked
													// list off of the FFILE structure.
	RCACHE *				pPrevInFile;		// Prev record cache object in linked
													// list off of the FFILE structure.
	RCACHE *				pNextInGlobal;		// Next record cache object in linked
													// list off of the FLMSYSDATA structure.
	RCACHE *				pPrevInGlobal;		// Prev record cache object in linked
													// list off of the FLMSYSDATA structure.
	RCACHE *				pOlderVersion;		// Older version of record in linked
													// list of versions of the record.
	RCACHE *				pNewerVersion;		// Newer version of record in linked
													// list of versions of the record.
	RCACHE *				pPrevInHeapList;	// Prev in the list of records whose
													// memory was allocated on the heap
	RCACHE *				pNextInHeapList;	// Next in the list of records whose
													// memory was allocated on the heap
	F_NOTIFY_LIST_ITEM *	pNotifyList;	// This is a pointer to a list of
													// threads that want to be notified
													// when a pending I/O is complete.
													// This pointer is only non-null if the
													// record is currently being read from
													// disk and there are multiple threads
													// all waiting for the record to be
													// read in.
	FLMUINT				uiFlags;				// Flags and use count.
#define RCA_READING_IN					0x80000000
#define RCA_UNCOMMITTED					0x40000000
#define RCA_HEAP_LIST					0x04000000
#define RCA_LATEST_VER					0x02000000
#define RCA_PURGED						0x01000000
#define RCA_LINKED_TO_FILE				0x00800000
#define RCA_COUNTER_BITS		(~(RCA_READING_IN | RCA_UNCOMMITTED | \
											RCA_LATEST_VER | RCA_PURGED | \
											RCA_HEAP_LIST | RCA_LINKED_TO_FILE))

#define RCA_IS_READING_IN(uiFlags) \
	(((uiFlags) & RCA_READING_IN) ? TRUE : FALSE)

#define RCA_SET_READING_IN(uiFlags) \
	((uiFlags) |= RCA_READING_IN)

#define RCA_UNSET_READING_IN(uiFlags) \
	((uiFlags) &= (~RCA_READING_IN))

#define RCA_IS_UNCOMMITTED(uiFlags) \
	(((uiFlags) & RCA_UNCOMMITTED) ? TRUE : FALSE)

#define RCA_SET_UNCOMMITTED(uiFlags) \
	((uiFlags) |= RCA_UNCOMMITTED)

#define RCA_UNSET_UNCOMMITTED(uiFlags) \
	((uiFlags) &= (~RCA_UNCOMMITTED))

#define RCA_IS_IN_HEAP_LIST(uiFlags) \
	(((uiFlags) & RCA_HEAP_LIST) ? TRUE : FALSE)

#define RCA_SET_IN_HEAP_LIST(uiFlags) \
	((uiFlags) |= RCA_HEAP_LIST)

#define RCA_UNSET_IN_HEAP_LIST(uiFlags) \
	((uiFlags) &= (~RCA_HEAP_LIST))
	
#define RCA_IS_LATEST_VER(uiFlags) \
	(((uiFlags) & RCA_LATEST_VER) ? TRUE : FALSE)

#define RCA_SET_LATEST_VER(uiFlags) \
	((uiFlags) |= RCA_LATEST_VER)

#define RCA_UNSET_LATEST_VER(uiFlags) \
	((uiFlags) &= (~RCA_LATEST_VER))

#define RCA_IS_PURGED(uiFlags) \
	(((uiFlags) & RCA_PURGED) ? TRUE : FALSE)

#define RCA_SET_PURGED(uiFlags) \
	((uiFlags) |= RCA_PURGED)

#define RCA_UNSET_PURGED(uiFlags) \
	((uiFlags) &= (~RCA_PURGED))

#define RCA_IS_LINKED_TO_FILE(uiFlags) \
	(((uiFlags) & RCA_LINKED_TO_FILE) ? TRUE : FALSE)

#define RCA_SET_LINKED_TO_FILE(uiFlags) \
	((uiFlags) |= RCA_LINKED_TO_FILE)

#define RCA_UNSET_LINKED_TO_FILE(uiFlags) \
	((uiFlags) &= (~RCA_LINKED_TO_FILE))

#define RCA_IS_IN_USE(uiFlags) \
	(((uiFlags) & RCA_COUNTER_BITS) ? TRUE : FALSE)

#define RCA_INCR_USE_COUNT(uiFlags) \
	((uiFlags) = ((uiFlags) & (~(RCA_COUNTER_BITS))) | \
				    ((((uiFlags) & RCA_COUNTER_BITS) + 1)))

#define RCA_DECR_USE_COUNT(uiFlags) \
	((uiFlags) = ((uiFlags) & (~(RCA_COUNTER_BITS))) | \
				    ((((uiFlags) & RCA_COUNTER_BITS) - 1)))
} RCACHE;

/****************************************************************************
Desc:	 	This structure defines the header information that is used to
			control the FLAIM record cache.  This structure will be embedded
			in the FLMSYSDATA structure.
****************************************************************************/
typedef struct RCACHE_MGR
{
	RCACHE *				pPurgeList;					// List of RCACHE structures that
															// need to be purged when use
															// count goes to zero.
	RCACHE *				pHeapList;					// List of cached records whose
															// memory was allocated on the
															// heap
	RCACHE *				pMRURecord;					// Most recently used record
	RCACHE *				pLRURecord;					// Least recently used record
	FLM_CACHE_USAGE	Usage;						// Contains maximum, bytes used,
															// etc.
	RCACHE **			ppHashBuckets;				// Array of hash buckets.
#define MIN_RCACHE_BUCKETS	0x10000				// 65536 buckets - multiple of 2.
#define MAX_RCACHE_BUCKETS 0x20000000			// roughly 500,000,000 buckets.
	FLMUINT				uiNumBuckets;				// Total number of hash buckets.
															// must be a multiple of 2.
	FLMUINT				uiHashMask;					// Hash mask mask for hashing a
															// DRN to a hash bucket.
	FLMUINT				uiPendingReads;			// Total reads currently pending.
	FLMUINT				uiIoWaits;					// Number of times multiple threads
															// were reading the same record from
															// disk at the same time.
	F_MUTEX				hMutex;						// Mutex for controlling record
															// cache.
	IF_FixedAlloc *	pRCacheAlloc;				// RCACHE structure allocator
	IF_FixedAlloc *	pRecAlloc;					// Fixed size allocator for record
															// objects
	IF_BufferAlloc *	pRecBufAlloc;				// Record buffer allocator

#ifdef FLM_DEBUG
	FLMBOOL				bDebug;						// Debug mode?
#endif
} RCACHE_MGR;

#define	FTHREAD_ACTION_IDLE					0
#define	FTHREAD_ACTION_INDEX_OFFLINE		1

/***************************************************************************
Desc:		Contains elements for passing parms into the background thread.
***************************************************************************/
typedef struct F_BKGND_IX
{
	FFILE *				pFile;
	FLMUINT				uiIndexingAction;
	FINDEX_STATUS		indexStatus;
	F_BKGND_IX *		pPrev;
	F_BKGND_IX *		pNext;
} F_BKGND_IX;

/***************************************************************************
Desc:		This is the FLAIM Event Structure.  It keeps track of a registered
			event callback function that has been registered for a particular
			event category.
***************************************************************************/
typedef struct FEVENT
{
	FEventCategory	eCategory;
	FEVENT_CB		fnEventCB;
	void *			pvAppData;
	FEVENT *			pNext;
	FEVENT *			pPrev;
} FEVENT;

/***************************************************************************
Desc:		This is the FLAIM Event Header Structure.  It is the header for
			the list of events that have been registered for a particular
			event category.
***************************************************************************/
typedef struct FEVENT_HDR
{
	FEVENT *			pEventCBList;		// List of registered event callbacks.
	F_MUTEX			hMutex;				// Mutex to control access to the
												// the event list.
} FEVENT_HDR;

/***************************************************************************
Desc:	Contains various parameters needed for displaying debugging data
		via http.  (The reason for all of the function pointers is so that
		we can compile and link without an http stack if one isn't available
		and we can dynamicly load one in after we've started.)
***************************************************************************/
typedef struct HTTPCONFIGPARAMS
{
	F_MUTEX					hMutex;		
	FLMUINT					uiUseCount;	// Webpage display functions increment this while
												// they're running.  The FlmConfig call to unimport
												// the symbols won't proceed until this is 0
	char	*					pszURLString;	// The URL that our http callback is
													// going to respond to.
	FLMUINT					uiURLStringLen;
	FLMBOOL					bRegistered;
	REG_URL_HANDLER_FN	fnReg;		// The registration function for our http callback
	DEREG_URL_HANDLER_FN	fnDereg;		// Function to deregister our http callback
	REQ_PATH_FN				fnReqPath;	// Function to retrieve the URI
	REQ_QUERY_FN			fnReqQuery;	//	Function to retrieve the query string
	REQ_HDR_VALUE_FN		fnReqHdrValue;	// Function to retrieve header values
	SET_HDR_VAL_FN			fnSetHdrValue;	// Function to set header values
	PRINTF_FN				fnPrintf;		// Function to send back formatted HTML
	EMIT_FN					fnEmit;			// Function to close and transmit
	SET_NO_CACHE_FN		fnSetNoCache;	// Function to send the header lines that
													// tell the client not to cache this page
	SEND_HDR_FN				fnSendHeader;	// Function to signal start of header info
	SET_IO_MODE_FN			fnSetIOMode;	// Function to switch between sending text and binary
	SEND_BUFF_FN			fnSendBuffer;	// Function to send data without formatting
	ACQUIRE_SESSION_FN	fnAcquireSession;	// Function to retrieve the Http Session
	RELEASE_SESSION_FN	fnReleaseSession;	// Function to release the Http Session
	ACQUIRE_USER_FN		fnAcquireUser;		// Function to retrieve the Http User
	RELEASE_USER_FN		fnReleaseUser;		// Function to release the Http User
	SET_SESSION_VALUE_FN	fnSetSessionValue;	// Function to set a persistant
															// value in the session
	GET_SESSION_VALUE_FN	fnGetSessionValue;	// Function to retrieve a persistant
															// parameter value from the session
	GET_GBL_VALUE_FN		fnGetGblValue;	// Function to retrieve parameters set on a global basis
	SET_GBL_VALUE_FN		fnSetGblValue;	// Function to set parametrers on a global basis
	RECV_BUFFER_FN			fnRecvBuffer;	// Function to retrieve a raw buffer of Post data.

} HTTPCONFIGPARAMS;

/***************************************************************************
Desc:		This is the FLAIM Shared System Data Structure.  It is the anchor
			for all of the other shared structures.
***************************************************************************/
typedef struct FLMSYSDATA
{
	FLMUINT				uiOpenFFiles;	// Number of open FFILEs
	FFILE *				pMrnuFile;		// Pointer to the most recently non-used
												// FFILE structure.
	FFILE *				pLrnuFile;		// Pointer to the least recently non-used
												// FFILE structure.
	F_BUCKET *			pFileHashTbl;	// File name hash table (array of F_BUCKET).
#define						FILE_HASH_ENTRIES		256

	FLMUINT				uiNextFFileId;	// ID that will be assigned to the next
												// FFILE created by flmAllocFile.
	F_MUTEX				hShareMutex;	// Mutex for controlling access to
												// FFILE structures, and shared cache.
	IF_FileSystem *	pFileSystem;	// File system used to configure options
												// for interacting with OS file system.
	FLMBOOL				bTempDirSet;	// TRUE if temporary directory has been set
	FLMBOOL				bCheckCache;	// Do extra checking of cache?
	FLMUINT				uiMaxCPInterval;
												// Maximum number of seconds to allow between
												// checkpoints
	FLMUINT				uiMaxTransTime;
												// Maximum number of seconds a transaction
												// can run before it can be forcibly killed. 
	FLMUINT				uiMaxTransInactiveTime;
												// Maximum number of seconds a transaction
												// can be inactive before it can be forcibly
												// killed.
	FLMBOOL				bCachePreallocated;
												// Cache memory was pre-allocated and is
												// a fixed size
	FLMBOOL				bDynamicCacheAdjust;
												// Is cache to be dynamically adjusted?
	FLMUINT				uiBlockCachePercentage;
												// Percentage of total cache to be used for
												// caching blocks
	FLMUINT				uiCacheAdjustPercent;
												// Percent of available memory to adjust to.
	FLMUINT				uiCacheAdjustMin;
												// Minimum limit to adjust cache to.
	FLMUINT				uiCacheAdjustMax;
												// Maximum limit to adjust cache to.
	FLMUINT				uiCacheAdjustMinToLeave;
												// Minimum bytes to leave when adjusting cache.
	FLMUINT				uiCacheAdjustInterval;
												// Interval for adjusting cache limit.
	FLMUINT				uiCacheCleanupInterval;
												// Interval for cleaning up old things out of
												// cache.
	FLMUINT				uiUnusedCleanupInterval;
												// Interval for cleaning up unused structures
	FLMUINT				uiMaxCache;		// Maximum amount of record and block cache (in bytes)
	SCACHE_MGR			SCacheMgr;		// Shared cache manager
	RCACHE_MGR			RCacheMgr;		// Record cache manager
	IF_Thread *			pMonitorThrd;	// Monitor thread
	FLM_STATS			Stats;			// Statistics structure
	F_MUTEX				hQueryMutex;	// Mutex for managing query list
	QUERY_HDR *			pNewestQuery;	// Head of query list (newest)
	QUERY_HDR *			pOldestQuery;	// Tail of query list (oldest)
	FLMUINT				uiQueryCnt;		// Number of queries in the list
	FLMUINT				uiMaxQueries;	// Maximum number of queries to keep around
	FLMBOOL				bNeedToUnsetMaxQueries;
												// When TRUE, indicates that a call to stop
												// statistics should also stop saving
												// queries.
	FLMBOOL				bStatsInitialized;
												// Has statistics structure been
												// initialized?
	char					szTempDir[ F_PATH_MAX_SIZE];
												// Temporary working directory for
												// ResultSets, RecordCache
												// and other sub-systems that need 
												// temporary files.  This is aligned
												// on a 4-byte boundary
	FLMUINT				uiMaxUnusedTime;
												// Maximum number of timer units to keep
												// unused structures in memory before
												// freeing them.
	FLMBYTE				ucBlobExt [64];// Blob Override extension
	FEVENT_HDR			LockEvents;		// Lock events
	FEVENT_HDR			UpdateEvents;	// Update events
	FEVENT_HDR			SizeEvents;		// Size threshold events
	F_Pool				KRefPool;		// Memory Pool that is only used by 
												// record updaters for key building
	HTTPCONFIGPARAMS	HttpConfigParms;
	FLMUINT				uiMaxFileSize;
	IF_LoggerClient *	pLogger;
	IF_SlabManager *	pSlabManager;
	IF_ThreadMgr *		pThreadMgr;
	IF_FileHdlCache *	pFileHdlCache;
	F_SessionMgr *		pSessionMgr;
	F_MUTEX				hHttpSessionMutex;
	FLMUINT				uiMaxStratifyIterations;
	FLMUINT				uiMaxStratifyTime;
	FLMUINT				uiFileOpenFlags;
	FLMUINT				uiFileCreateFlags;

} FLMSYSDATA;

#ifndef ALLOCATE_SYS_DATA
	extern FLMSYSDATA		gv_FlmSysData;
	extern FLMUINT			gv_uiBackIxThrdGroup;
	extern FLMUINT			gv_uiCPThrdGrp;
	extern FLMUINT			gv_uiDbThrdGrp;
#else
	#ifdef FLM_SOLARIS
		#pragma align 8 (gv_FlmSysData)
	#endif
	FLMSYSDATA				gv_FlmSysData;
	FLMUINT					gv_uiBackIxThrdGroup = F_INVALID_THREAD_GROUP;
	FLMUINT					gv_uiCPThrdGrp = F_INVALID_THREAD_GROUP;
	FLMUINT					gv_uiDbThrdGrp = F_INVALID_THREAD_GROUP;
#endif

/****************************************************************************
Desc:		This structure is used to sort keys before the keys are actually
			added to an index.
****************************************************************************/
typedef struct KREF_ENTRY
{
	FLMUINT		uiFlags;					// Flags for this KREF entry.
		#define KREF_UNIQUE_KEY		0x01
		#define KREF_DELETE_FLAG	0x02	// Must be defined for more than 1
		#define KREF_EQUAL_FLAG 	0x04	// Was equal with another reference
		#define KREF_IGNORE_FLAG	0x08	// Ignore this kref
		#define KREF_MISSING_OK		0x10	// OK For key to be missing from
													// index on delete operations?
		#define KREF_ENCRYPTED_KEY	0x20	// Key cannot be stored.  It is used
													// for testing purposes only.
	FLMUINT		uiDrn;		  			// DRN
	FLMUINT		uiTrnsSeq;	  			// Sequence of updates within trans.

	// Note: used uint16 below to reduce memory allocations.

	FLMUINT16	ui16IxNum;		  		// Index number
	FLMUINT16	ui16KeyLen;				// Key Length for this entry.  The key
												// comes immediately after this structure.
} KREF_ENTRY;

/****************************************************************************
Desc:		This structure is used in the key building process
****************************************************************************/
typedef struct KREF_CNTRL
{
	KREF_ENTRY **	pKrefTbl;			// Pointer to KREF Table which is an array of
												// KREF_ENTRY * pointers.
	CDL **			ppCdlTbl;			// Pointer to table of CDL pointers.
												// There is one CDL pointer per IFD.
	FLMBYTE *		pIxHasCmpKeys;		// Pointer to table of FLMBYTEs.  There
												// is one FLMBYTE for each index. The
												// FLMBYTE indicates whether or not the
												// index had any compound keys for the
												// current operation.
	FLMBYTE *		pKrefKeyBuf;		// Pointer to temporary key buffer.
	FLMUINT			uiKrefTblSize;		// KREF table size.
	FLMUINT			uiCount;				// Number of entries in KREF table that
												// are currently used.
	FLMUINT			uiTotalBytes;		// Total number of entries allocated
												// in the pool.
	FLMUINT			uiLastRecEnd;		// Entry in the KREF table of the last
												// key that was generated in the
												// previous update operation.
	FLMUINT			uiTrnsSeqCntr;		// Counts updates done within the current
												// transaction.  It is used when doing
					 							// duplicate checking.
	FLMBOOL			bKrefSetup;			// True if the KRefCntrl has been initialized.
	F_Pool *			pPool;				// GEDCOM pool to use
	FLMBOOL			bReusePool;			// Reuse pool instead of free it?
	FLMBOOL			bHaveCompoundKey;	// True if a compound key has been processed.
	void *			pReset;				// Used to reset pool for failed records.
} KREF_CNTRL;

/****************************************************************************
Desc:	 	This structure keeps track of diagnostic information (if any)
			associated with the last FLAIM operation.  It is reset at the
			beginning of every operation.
****************************************************************************/
typedef struct FDIAG
{
	FLMUINT		uiInfoFlags;			// Bit flags indicating what diagnostic
												// information, if any, was recorded by
												// the last FLAIM operation. (See flaim.h)

	// The remaining elements of FDIAG contain the diagnotic information
	// for the predefined Diagnotic codes that are defined in FLAIM.h
	// see FLM_DIAG_xxxx defines

	FLMUINT		uiDrn;					// contains data for FLM_DIAG_DRN
	FLMUINT		uiIndexNum;				// contains data for FLM_DIAG_INDEX_NUM
	FLMUINT		uiFieldNum;				// contains data for FLM_DIAG_FIELD_NUM
	FLMUINT		uiFieldType;			// contains data for FLM_DIAG_FIELD_TYPE
	FLMUINT		uiEncId;					// contains data for FLM_DIAG_ENC_ID

} FDIAG;

/****************************************************************************
Desc:	 	This structure keeps track of a connection that is in progress with
			a server.
****************************************************************************/
typedef struct CS_CONTEXT
{
	void *			pTcpClient;
	void *			pDDSLocalRequestFunc;
	FLMUINT			uiStreamHandlerId;
#define FSEV_HANDLER_UNKNOWN		0x00000000
#define FSEV_HANDLER_LOOPBACK		0x00000001
#define FSEV_HANDLER_DS				0x00000002

	FLMUINT			uiSessionId;		// Session ID sent from the server
	FLMUINT			uiSessionCookie;	// Server's cookie for this session

	FLMBOOL			bConnectionGood;	// Is the connection still good?
	FLMBOOL			bTransActive;		// Is a transaction active?
	FLMINT			iSubProtocol;		// Sub-Protocol for this connection.
	FCS_ISTM *		pIStream;			// Input stream for receiving data from
												// server.
	FCS_OSTM *		pOStream;			// Output stream for sending data to
												// server.
	FCS_DIS *		pIDataStream;		// Input data stream for receiving data
												// from server.
	FCS_DOS *		pODataStream;		// Output data stream for sending data
												// to server.
	FLMUINT			uiOpSeqNum;			// Operation Sequence Number -
												// inremented for every operation that is
												// performed.
	F_Pool			pool;					// Pool to pass into Wire objects.
	char				pucAddr[ FLM_CS_MAX_ADDR_LEN];
												// Stream address.  This is
												// aligned on a 4-byte boundary
	FLMUINT			uiAddressType;		// Type of address in address buf
	FLMBOOL			bGedcomSupport;	// TRUE indicates that the 
												// server understands GEDCOM
	FLMUINT			uiServerFlaimVer; // FLAIM code version of the server
	char				pucUrl[ FLM_CS_MAX_ADDR_LEN];
} CS_CONTEXT;

/**************************************************************************
Desc:		Information from the log header.  The fields in this structure
			are used temporarily during both read and update transactions.
			They are set at the beginning of the transaction when the
			log header is read.  Some are updated during update transactions
			and will be written back into the log header when the update
			transaction completes.
**************************************************************************/
typedef struct LOG_HDR
{
	FLMUINT  	uiCurrTransID;			// Current transaction ID.
#define	TRANS_ID_HIGH_VALUE		0xFFFFFFFF
#define	TRANS_ID_LOW_VALUE		0
	FLMUINT		uiFirstAvailBlkAddr;	// Address of first block in avail list
	FLMUINT		uiAvailBlkCount;		// Avail block count
	FLMUINT		uiLogicalEOF;			// Current logical end of file.  New
												// blocks are allocated at this address.
} LOG_HDR;

/**************************************************************************
Desc:
**************************************************************************/
typedef struct IX_STATS
{
	FLMUINT		uiIndexNum;
	FLMINT		iDeltaRefs;
	FLMINT		iDeltaKeys;
	IX_STATS *	pNext;
} IX_STATS;

/**************************************************************************
Desc:
**************************************************************************/
typedef struct IXD_FIXUP
{
	FLMUINT		uiIndexNum;
	FLMUINT		uiLastContainerIndexed;
	FLMUINT		uiLastDrnIndexed;
	IXD_FIXUP *	pNext;
} IXD_FIXUP;

/**************************************************************************
Desc: 	This structure is the current database context.  The database
			context structure points to a particular FLAIM file and contains
			the application context for accessing that file.
**************************************************************************/
typedef struct FDB
{
	FFILE *					pFile;				// Pointer to FFILE structure.
	FDICT *					pDict;				// Pointer to local dictionary
	FDB *						pNextForFile;		// Next FDB associated with FFILE
	FDB *						pPrevForFile;		// Prev FDB associated with FFILE
	void *					pvAppData;			// Application data that is used
														// to associate this FDB with
														// an object in the application
														// space.
	FLMUINT					uiThreadId;			// Thread that started the current
														// transaction, if any.  NOTE:
														// Only set on transaction begin.
														// Hence, if operations are performed
														// by multiple threads, within the
														// transaction, it will not necessarily
														// reflect the thread that is currently
														// using the FDB.
	FLMBOOL					bMustClose;			// An error has occurred that requires
														// the application to stop using (close)
														// this FDB
	FLMUINT					uiInitNestLevel;	// Number of times fdbInit has
														// been called recursively.
	FLMUINT 					uiInFlmFunc;		// This variable is incremented
														// prior to calling a user call-back.
														// Currently, a non-zero value
														// will prevent the early resetting
														// of the 'TempPool' by a call to
														// another FLAIM function within
														// within the users call-back
														// function.
	F_SuperFileHdl *		pSFileHdl;			// Pointer to the super file handle
	FLMUINT					uiFlags;				// Flags for this FDB.
		#define FDB_NU_FLAG					0x0001
														// This flag is unused
		#define FDB_UPDATED_DICTIONARY	0x0002
														// Flag indicating whether the file's
														// local dictionary was updated
														// during the transaction.
		#define FDB_DO_TRUNCATE				0x0004
														// Truncate log extents at the end
														// of a transaction.
		#define FDB_INVISIBLE_TRANS		0x0008
														// If a transaction is going,
														// indicates if it was implicitly
														// started by FLAIM and should be
														// invisible to an application.
														// Invisible transactions will be
														// aborted if the application wants
														// to start its own transaction.
		#define FDB_HAS_FILE_LOCK			0x0010
														//	FDB has a file lock.
		#define FDB_FILE_LOCK_SHARED		0x0020
														// File lock is shared.  Update
														// transactions are not allowed when
														// the lock is shared.
		#define FDB_FILE_LOCK_IMPLICIT	0x0040
														// File lock is implicit - means file
														// lock was obtained when the update
														// transaction began and cannot be
														// released by a call to FlmDbUnlock.
		#define FDB_DONT_KILL_TRANS		0x0080
														// Do not attempt to kill an active
														// read transaction on this database
														// handle.  This is used by FlmDbBackup.
		#define FDB_INTERNAL_OPEN			0x0100
														// FDB is an internal one used by a
														// background thread.
		#define FDB_DONT_POISON_CACHE		0x0200
														// If blocks are read from disk during
														// a transaction, release them at the LRU
														// end of the cache chain.
		#define FDB_UPGRADING				0x0400
														// Database is being upgraded.
		#define FDB_REPLAYING_RFL			0x0800
														// Database is being recovered
		#define FDB_REPLAYING_COMMIT		0x1000
														// During replay of the RFL, this
														// is an actual call to FlmDbTransCommit.
		#define FDB_BACKGROUND_INDEXING	0x2000
														// FDB is being used by a background indexing
														// thread
		#define FDB_HAS_WRITE_LOCK			0x4000
														// FDB has the write lock
		#define FDB_COMMITTING_TRANS		0x8000

	// TRANSACTION STATE STUFF

	FLMUINT					uiTransCount;		// Transaction counter for the FDB.
														// Incremented whenever a transaction
														// is started on this FDB.  Used so
														// that FLAIM can tell if an implicit
														// transaction it started is still in
														// effect.  This should NOT be
														// confused with update transaction
														// IDs.
	FLMUINT					uiTransType;		// Type of transaction - see FLAIM.H
														// FLM_NO_TRANS if no transaction.
	eFlmFuncs				eAbortFuncId;		// Set to the function ID of the func
														// that caused AbortRc (below) to be
														// set
	RCODE						AbortRc;				// If not FERR_OK, transaction must be
														// aborted.
	LOG_HDR					LogHdr;				// This contains information taken
														// from the log header at the
														// beginning of the transaction.  An
														// update transaction may update the
														// information and write it back to
														// the log header if the transaction
														// commits.
	FLMUINT					uiUpgradeCPFileNum;
	FLMUINT					uiUpgradeCPOffset;
														// RFL file number and offset to set
														// RFL to during an upgrade operation
														// that happens during a restore or
														// recovery.
	FLMUINT					uiTransEOF;			// Address of logical end of file
														// when the last transaction
														// committed. A block beyond this
														// point in the file is going to be
														// a new block and will not need to
														// be logged.
	KREF_CNTRL				KrefCntrl;			// This structure is used to manage
														// KREF data that is generated during
														// a transaction.  NOTE: Not used in
														// read transactions except when
														// doing an index check.  In that
														// case, it will be initialized by
														// the check code.
	IX_STATS *				pIxStats;

	F_TMSTAMP				TransStartTime;	// Transaction start time, for stats

	// UPDATE TRANSACTION STUFF

	FLMBOOL					bHadUpdOper;		// Did this transaction have any
														// updates?
	FLMUINT					uiBlkChangeCnt;	// Number of times ScaLogPhysBlk has
														// been called during this transaction.
														// This is used by the cursor code to
														// know when it is necessary to
														// re-position in the B-Tree.
	FlmBlobImp *			pBlobList;			// Linked list of BLOBs that have
														// been created or deleted during the
														// transaction - not used for read
														// transactions.
	IXD_FIXUP *				pIxdFixups;			// List of indexes whose IXD needs
														// to be restored to its prior
														// state if the transaction aborts

	// READ TRANSACTION STUFF

	FDB *						pNextReadTrans;	// Next active read transaction for
														// this file.
														// NOTE: If uiKilledTime (see below)
														// is non-zero, then transaction is
														// in killed list.
	FDB *						pPrevReadTrans;	// Previous active read transaction
														// for this file.
														// NOTE: If uiKilledTime (see below)
														// is non-zero, then transaction is
														// in killed list.
	FLMUINT					uiInactiveTime;	// If non-zero, this is the last time
														// the checkpoint thread marked this
														// transaction as inactive.  If zero,
														// it means that the transaction is
														// active, or it has not been marked
														// by the checkpoint thread as
														// inactive.  If it stays non-zero for
														// five or more minutes, it will be
														// killed.
	FLMUINT					uiKilledTime;		// Time transaction was killed, if
														// non-zero.
	F_Pool					tmpKrefPool;		// GEDCOM KREF pool to be used during
														// read transactions - only used when
														// checking indexes.
	// Misc. DB Info.

	FLMBOOL					bFldStateUpdOk;	//	This variable is used to ensure
														// that FlmDbSweep / recovery are the
														// only ways that:
														// 1) a fld's state can be changed
														//    to 'unused'
														// 2) a 'purge' fld can be deleted

	FDIAG						Diag;					// Diagnostic information from the
														// last FLAIM operation.
	F_Pool					TempPool;			// Temporary GEDCOM Memory Pool.  It
														// is only used for the duration of
														// a FLAIM operation and then reset.
														// The first block in the pool is
														// retained between operations to
														// help performance.

	// Callback functions.

	REC_VALIDATOR_HOOK	fnRecValidator;	// Record Validator Hook
	void *					RecValData;			// Record Validator User Data

	STATUS_HOOK				fnStatus;			// Returns various status within flaim
	void *					StatusData;			// User data for status function
	IX_CALLBACK				fnIxCallback;		// Indexing callback
	void *					IxCallbackData;	// User data for indexing callback.
	COMMIT_FUNC				fnCommit;			// Commit callback
	void *					pvCommitData;		// User data for commit callback.

	FLM_STATS *				pStats;
	DB_STATS *				pDbStats;			// DB statistics pointer.
	LFILE_STATS *			pLFileStats;		// LFILE statistics pointer.
	FLMUINT					uiLFileAllocSeq;	// Allocation sequence number for
														// LFILE statistics array so we
														// can tell if the array has been
														// reallocated and we need to reset
														// our pLFileStats pointer.
	FLM_STATS				Stats;				// Statistics kept here until end
														// of transaction.
	FLMBOOL					bStatsInitialized;// Has statistics structure been
														// initialized?
	CS_CONTEXT *			pCSContext;			// Pointer to client/server
														// connection this FDB is associated
														// with, NULL if none.
	F_BKGND_IX *			pIxStartList;		// Indexing threads to start at 
														// the conclusion of the transaction.
	F_BKGND_IX *			pIxStopList;		// Indexing threads to stop at 
														// the conclusion of the transaction.
	F_SEM						hWaitSem;			// Semaphore used when waiting to
														// acquire a file or write lock
														// on the database
	FLMBYTE *				pucAlignedReadBuf;// Aligned buffer for block reads
#ifdef FLM_DEBUG

	//NOTE: Always set - no need to be part of memset.

	F_MUTEX					hMutex;				// Mutex for controlling access to
														// FDB - don't want multiple threads
														// accessing the FDB.
	FLMUINT					uiUseCount;			// Number of times thread has
														// incremented the use count.
#endif
} FDB;

// The following positions are relative to FLAIM_HEADER_START.  The are
// no longer absolute file addresses.

#define FLAIM_NAME_POS						0
#define FLAIM_NAME		 					"FLAIM"
#define FLAIM_NAME_LEN						5

// FLAIM Version Number Defines

#define FLM_FILE_FORMAT_VER_POS			(FLAIM_NAME_LEN)
#define FLM_FILE_FORMAT_VER_LEN			4
#define FLM_MINOR_VER_POS 					(FLM_FILE_FORMAT_VER_POS + 2)
#define FLM_SMINOR_VER_POS   				(FLM_FILE_FORMAT_VER_POS + 3)

// Defines to access elements within the database version number

#define GetMajorVerNum( wVer)				((wVer) / 100)
#define GetMinorVerNum( wVer)				(((wVer) % 100) / 10)
#define GetSubMinorVerNum( wVer)			(((wVer) % 100) % 10)

//#define DB_NOT_USED						9-12	// Should be zero for pre 4.3
#define DB_DEFAULT_LANGUAGE				13
#define DB_BLOCK_SIZE		  				14
//#define DB_NOT_USED						16-23	// Should be zero for pre 4.3
#define DB_INIT_LOG_SEG_ADDR				24		// Not used in 4.3
#define DB_LOG_HEADER_ADDR					28		// Not used in 4.3
#define DB_1ST_LFH_ADDR						32
#define DB_1ST_PCODE_ADDR					36		// Not used in 4.3
//#define DB_NOT_USED						40		// was DB_ENCRYPT_VER
#define DB_RESERVED							42
//#define DB_NOT_USED						44		// was DB_ENCRYPT_BLOCK	

#define FLM_UNUSED_FILE_HDR_SPACE		128	// was ENC_BLOCK_SIZE

#define FLM_FILE_HEADER_SIZE				44

#define FLAIM_HEADER_START					(2048 - (FLM_FILE_HEADER_SIZE + \
														FLM_UNUSED_FILE_HDR_SPACE))

#define DB_LOG_HEADER_START				16

/**************************************************************************
Desc: 	This structure contains the file header information for a file as
			well as its create options.
**************************************************************************/
typedef struct FILE_HDR
{
	FLMUINT		uiFirstLFHBlkAddr;	// Address of first LFH block.
	FLMUINT		uiVersionNum;			// Database version		
	FLMUINT		uiBlockSize;			// Block size
	FLMUINT		uiDefaultLanguage;	// Default language
	FLMUINT		uiAppMajorVer;			// Application major version number
	FLMUINT		uiAppMinorVer;			// Application minor version number
	FLMUINT		uiSigBitsInBlkSize;	// Number of significant bits in block
												// size. 1K = 10, 2K = 11, 4K = 12 ...
	FLMBYTE		ucFileHdr[ FLM_FILE_HEADER_SIZE];
} FILE_HDR;

/**************************************************************************
Desc: 	This structure is the main shared structure for a FLAIM file.  It
			contains static information about the file.
**************************************************************************/
typedef struct FFILE
{
	FFILE *					pNext;					// Next FFILE structure in in name hash
															// bucket, dependent store hash
															// bucket, or avail list.
	FFILE *					pPrev;					// Previous FFILE structure in name hash
															// bucket or dependent store hash
															// bucket.
	FLMUINT					uiFFileId;				// Unique FFILE identifier
	FLMUINT					uiZeroUseCountTime;	// Time that use count went to zero.
	FLMUINT					uiUseCount;				// Number of FDBs currently using this	file.
	FLMUINT					uiInternalUseCount;	// Number of the uses that are internal
															// background threads.
	FDB *						pFirstDb;				// List of ALL FDB's associated with
															// this FFILE.
	char *					pszDbPath;				// Database file name.
	char *					pszDataDir;				// Path for data files.
	FFILE *					pNextNUFile;			// Next FFILE structure in list of
															// unused files.  When use count goes
															// to zero, the structure is linked
															// into a list of unused files off of
															// the FSYSDATA structure.
	FFILE *					pPrevNUFile;			// Previous FFILE structure in list of
															// unused files.
	SCACHE *					pSCacheList;			// This is a pointer to a linked list
															// of all shared cache blocks
															// belonging to this file.
	SCACHE *					pPendingWriteList;	// This is a pointer to a linked list
															// of all shared cache blocks
															// that are in the pending-write state.
	SCACHE *					pLastDirtyBlk;			// Pointer to last dirty block in the
															// list.
	SCACHE *					pFirstInLogList;		// First block that needs to be logged
	SCACHE *					pLastInLogList;		// Last block that needs to be logged
	FLMUINT					uiLogListCount;		// Number of items in the log list
	SCACHE *					pFirstInNewList;		// First new block that is dirty
	SCACHE *					pLastInNewList;		// Last new block that is dirty
	FLMUINT					uiNewCount;				// Number of items in new list
	FLMUINT					uiDirtyCacheCount;	// Number of dirty blocks
	FLMUINT					uiLogCacheCount;		// Log blocks needing to be written.
	RCACHE *					pFirstRecord;			// Head of list of records in record cache
															// that are associated with this file.
	RCACHE *					pLastRecord;			// End of list of records in record cache
															// that are associated with this file.
	SCACHE **				ppBlocksDone;			// List of blocks to be written to rollback
															// log or database.
	FLMUINT					uiBlocksDoneArraySize;
															// Size of ppBlocksDone array.
	FLMUINT					uiBlocksDone;			// Number of blocks currently in the
															// ppBlocksDone array.
	SCACHE *					pTransLogList;			// This is a pointer to a linked list
															// of all shared cache blocks
															// belonging to this file that need
															// to be logged to the rollback log
															// for the current transaction.
	F_NOTIFY_LIST_ITEM *	pOpenNotifies;			// Pointer to a list of notifies to
															// perform when this file is finally
															// opened (points to a linked list of
															// F_NOTIFY_LIST_ITEM structures).
	F_NOTIFY_LIST_ITEM *	pCloseNotifies;		// Pointer to a list of notifies to
															// perform when this file is finally
															// closed (points to a linked list of
															// F_NOTIFY_LIST_ITEM structures).
	FDICT *					pDictList;				// Pointer to linked list of 
															// dictionaries currently being used
															// for this file.  The linked list
															// is a list of versions of the 
															// dictionary.  When a version is no
															// longer used, it is removed from the
															// list.  Hence, the list is usually
															//	has only one member.
	FLMBOOL					bMustClose;				// The FFILE is being forced to close
															// because of a critical error.
	RCODE						rcMustClose;			// Return code that caused bMustClose to
															// be set.
	F_Pool					krefPool;				// GEDCOM Kref pool to be used during update
															// transactions.
	FILE_HDR					FileHdr;					// This structure contains the file
															// header information for the file.
	FLMUINT					uiMaxFileSize;			// Maximum file size.
	FLMUINT					uiFileExtendSize;		// Bytes to extend files by.
	FLMUINT					uiRflFootprintSize;	// When RFL files are not being kept,
															// this is the desired footprint of
															// the RFL on disk
	FLMUINT					uiRblFootprintSize;	// This is the desired footprint of
															// the RBL on disk
	FLMUINT					uiUpdateTransID;		// This is the transaction ID currently
															// being used by an active update
															// transaction on this file.  When
															// an update transaction begins it
															// sets this value to its
															// checkpoint.
	F_Rfl *					pRfl;						// Pointer RFL object.
	FLMBYTE					ucLastCommittedLogHdr[ LOG_HEADER_SIZE];
															// This is the last committed log header.
	FLMBYTE					ucCheckpointLogHdr[ LOG_HEADER_SIZE];
															// This is the log header as of the start
															// of the last checkpoint.
	FLMBYTE					ucUncommittedLogHdr[ LOG_HEADER_SIZE];
															// This is the uncommitted log header.
															// It is used by the current update
															// transaction.

#define LOG_RFL_FILE_NUM					0
#define LOG_RFL_LAST_TRANS_OFFSET		4		// NOTE: Could be zero
#define LOG_RFL_LAST_CP_FILE_NUM			8		// Only written on checkpoint
#define LOG_RFL_LAST_CP_OFFSET			12		// Only written on checkpoint.
															// Should NEVER be less than 512
#define LOG_ROLLBACK_EOF					16
#define LOG_INC_BACKUP_SEQ_NUM			20
#define LOG_CURR_TRANS_ID					24		// Only written on checkpoint
#define LOG_COMMIT_COUNT					28		// Only written on checkpoint
#define LOG_PL_FIRST_CP_BLOCK_ADDR		32
#define LOG_LAST_RFL_FILE_DELETED		36
#define LOG_RFL_MIN_FILE_SIZE				40
#define LOG_HDR_CHECKSUM		 			44
#define LOG_FLAIM_VERSION					46
#define LOG_LAST_BACKUP_TRANS_ID			48
#define LOG_BLK_CHG_SINCE_BACKUP			52		// Only written on checkpoint
#define LOG_LAST_CP_TRANS_ID				56
#define LOG_PF_FIRST_BACKCHAIN			60		// Only written on checkpoint
#define LOG_PF_AVAIL_BLKS		 			64		// Only written on checkpoint
#define LOG_LOGICAL_EOF    	 			68		// Only written on checkpoint
#define LOG_LAST_RFL_COMMIT_ID			72		// Only written on checkpoint
#define LOG_KEEP_ABORTED_TRANS_IN_RFL	76		// Was trans active flag in pre 4.3
#define LOG_PF_FIRST_BC_CNT	 			77		// Only written on checkpoint
#define LOG_KEEP_RFL_FILES					78
#define LOG_AUTO_TURN_OFF_KEEP_RFL		79		// Was maintenance in progress in pre 4.3
#define LOG_PF_NUM_AVAIL_BLKS 			80		// Only written on checkpoint
#define LOG_RFL_MAX_FILE_SIZE				84
#define LOG_DB_SERIAL_NUM					88
#define LOG_LAST_TRANS_RFL_SERIAL_NUM	104
#define LOG_RFL_NEXT_SERIAL_NUM			120
#define LOG_INC_BACKUP_SERIAL_NUM		136
#define LOG_NU_152_153						152	// Two bytes are unused
#define LOG_MAX_FILE_SIZE					154	// Multiply by 64K to get actual maximum
#define LOG_DATABASE_KEY_LEN				156 	// Current Length of the database key
#define LOG_DATABASE_KEY					158 	// Wrapped or shrouded copy of the database key (max 256 bytes)
#define LOG_RFL_DISK_SPACE_THRESHOLD	414	// RFL disk space threshold in K bytes (0 = unlimited)
#define LOG_RFL_LIMIT_TIME_FREQ			418	// How often (in seconds) to throw an over-the-limit event (0 = ignore)
#define LOG_RFL_LIMIT_SPACE_FREQ			422	// How often (based on the size increase in megabytes since the last event) 
															// to throw an over-the-limit-event (0 = ignore)

#define FLM_MAX_DB_ENC_KEY_LEN			256

	IF_IOBufferMgr *		pBufferMgr;

#define MAX_WRITE_BUFFER_BYTES			(4 * 1024 * 1024)
#define MAX_PENDING_WRITES					(MAX_WRITE_BUFFER_BYTES / 4096)
#define MAX_LOG_BUFFER_SIZE				(256 * 1024)

	IF_IOBuffer *			pCurrLogBuffer;
	FLMUINT					uiCurrLogWriteOffset;// Offset in current write buffer
	FLMUINT					uiCurrLogBlkAddr;		// Address of first block in the current
															// buffer.
	FLMBYTE *				pucLogHdrIOBuf;		// Aligned buffer for reading and writing
															// the log header.
	IF_LockObject *		pFileLockObj;			// Object for locking the file.
	IF_LockObject *		pWriteLockObj;			// Object for locking to do writing.
	IF_FileHdl *			pLockFileHdl;			// Lock file handle for 3.x databases.
	F_NOTIFY_LIST_ITEM *	pLockNotifies;			// Pointer to a list of notifies to
															// perform when this file is finally
															// locked (points to a linked list of
															// F_NOTIFY_LIST_ITEM structures).
	FLMBOOL					bBeingLocked;			// Flag indicating whether or not this
															// file is in the process of being
															// locked for exclusive access.
	FDB *						pFirstReadTrans;		// Pointer to first read transaction for
															// this file.
	FDB *						pLastReadTrans;		// Pointer to last read transaction for
															// this file.
	FDB *						pFirstKilledTrans;	// List of read transactions that have
															// been killed.
	FLMUINT					uiFirstLogBlkAddress;// Address of first block logged for the
															// current update transaction.

	FLMUINT					uiFirstLogCPBlkAddress;
															// Address of first block logged for the
															// current checkpoint.
	FLMUINT					uiLastCheckpointTime;
															// Last time we successfully completed a
															// checkpoint.
	IF_Thread *				pMonitorThrd;			// Database monitor thread
	IF_Thread *				pCPThrd;					// Checkpoint thread.
	CP_INFO *				pCPInfo;					// Pointer to checkpoint thread's
															// information buffer - used for
															// communicating information to the
															// checkpoint thread.
	RCODE						CheckpointRc;			// Return code from last checkpoint
															// that was attempted.
	FLMUINT					uiBucket;				// Hash bucket this file is in.
															// 0xFFFF means it is not currently
															// in a bucket.
	FLMUINT					uiFlags;					// Flags for this file.
#define DBF_BEING_OPENED	0x01					// Flag indicating whether this file is
															// in the process of being opened.
#define DBF_IN_NU_LIST		0x02					// Flag indicating whether this file was
															// opened in exclusive access mode.
#define DBF_BEING_CLOSED	0x04					// Database is being closed - cannot open.
	FLMBOOL					bBackupActive;			// Backup is currently being run against the
															// database.
	F_CCS *					pDbWrappingKey;		// Master Wrapping Key
	FLMBOOL					bInLimitedMode;		// Set to true if we're running
															// in limited mode.
	RCODE						rcLimitedCode;			// Reason we are in limited mode.
	FLMBOOL					bHaveEncKey;			// The database has an encryption key. There are times when we will
															// want to override the bAllowLimitedMode parameter on the open.  If 
															// the version of flaim supports encryption, but the database was not
															// built with encryption, we will set the database into limited mode.
	IF_Thread *				pMaintThrd;				// Processes background jobs queued in
															// the tracker (except indexing)
	F_SEM						hMaintSem;				// Used to signal the maintenance thread
															// that there may be some work to do
	FMAINT_STATUS			maintStatus;
	char *					pszDbPassword;			// A password that was used to open the database (may be NULL).
	FLMUINT64				ui64RflDiskUsage;		// Current RFL disk space usage estimate (used only if keeping RFL files)
} FFILE;

/****************************************************************************
Desc:	 	Structure used to pass information to the checkpoint thread for 3.x
			databases.
****************************************************************************/
typedef struct CP_INFO
{
	FFILE *				pFile;
	F_SuperFileHdl *	pSFileHdl;
	FLM_STATS			Stats;
	FLMBOOL				bStatsInitialized;
	FLMBOOL				bShuttingDown;
	FLMBOOL				bDoingCheckpoint;
	FLMUINT				uiStartTime;
	FLMBOOL				bForcingCheckpoint;
	FLMUINT				uiForceCheckpointStartTime;
	FLMINT				iForceCheckpointReason;
	FLMUINT				uiLogBlocksWritten;
	FLMBOOL				bWritingDataBlocks;
	FLMUINT				uiDataBlocksWritten;
	FLMUINT				uiStartWaitTruncateTime;
} CP_INFO;

/****************************************************************************
Desc:	 	State information for parsing forward/back in reference set lists.
****************************************************************************/
typedef struct DIN_STATE
{
	FLMUINT		uiOffset;
	FLMUINT		uiOnes;
} DIN_STATE;

#define RESET_DINSTATE( state) \
{ \
	(state).uiOffset = (state).uiOnes = 0; \
}

#define RESET_DINSTATE_p( pState) \
{ \
	(pState)->uiOffset = (pState)->uiOnes = 0; \
}

#define MAX_KEY_SIZ						640

/****************************************************************************
Desc:	 	State information for parsing forward/back in reference set lists.
****************************************************************************/
typedef struct BTSK
{
	FLMBYTE *	pBlk;						// Points to the cache block buffer
	FLMBYTE *	pKeyBuf; 				// Points to a key buffer - near ptr
	SCACHE *		pSCache;					// Points to current cache entry.
	FLMUINT		uiBlkAddr;				// Block address (number?)
	FLMUINT		uiCmpStatus;			// Status from compare
	FLMUINT		uiKeyLen; 				// Length of the key in bsKeyBuf
	FLMUINT		uiCurElm; 				// Offset of element in the block
	FLMUINT		uiBlkEnd; 				// End of the current block of data
	FLMUINT		uiPKC;					// # of bytes used from the previous elm
	FLMUINT		uiPrevElmPKC;			// Previous element's PKC
	FLMUINT		uiKeyBufSize;			// Maximum size of the key buffer.
	FLMUINT		uiFlags;					// Flags to set
#define				FULL_STACK		1 	// Stack setup for update
#define				NO_STACK			2 	// Stack NOT setup for update
	FLMUINT		uiElmOvhd;				// Element overhead in the block
	FLMUINT		uiBlkType;				// Block type - 0 = leaf, 1 = non-leaf
	FLMUINT		uiLevel;					// Level number of block
} BTSK;

/****************************************************************************
Desc:	 	State information for performing a backup
****************************************************************************/
typedef struct FBak
{
	HFDB				hDb;
	FLMUINT			uiTransType;
	FLMUINT			uiTransId;
	FLMUINT			uiLastBackupTransId;
	FLMUINT			uiDbVersion;
	FLMUINT			uiBlkChgSinceLastBackup;
	FLMBOOL			bTransStarted;
	FLMBOOL			bCSMode;
	FLMUINT			uiBlockSize;
	FLMUINT			uiLogicalEOF;
	FLMUINT			uiFirstReqRfl;
	FLMUINT			uiIncSeqNum;
	FLMBOOL			bCompletedBackup;
	FBackupType		eBackupType;
	RCODE				backupRc;
	FLMBYTE *		pucDbHeader;
	FLMBYTE			ucNextIncSerialNum[ F_SERIAL_NUM_SIZE];
	FLMBYTE			ucDbPath[ F_PATH_MAX_SIZE];
} FBak;

/****************************************************************************
Desc:
****************************************************************************/
typedef struct GED_STREAM
{
	IF_FileHdl *	pFileHdl;
	FLMUINT			uiBufSize;
	FLMUINT64		ui64FilePos;
	char *			pBuf;
	char *			pThis;
	char *			pLast;
	FLMINT			errorIO;
	FLMINT			thisC;
} GED_STREAM;

#define	MAX_COMPOUND_PIECES		32

/****************************************************************************
Desc:
****************************************************************************/
typedef struct FLD_CONTEXT
{
	void *		pParentAnchor;
	void *		rootContexts[ MAX_COMPOUND_PIECES];
	void *		leafFlds[ MAX_COMPOUND_PIECES];
} FLD_CONTEXT;

/****************************************************************************
Desc:
****************************************************************************/
typedef struct EXP_IMP_INFO
{
	IF_FileHdl *	pFileHdl;
	FLMBYTE *		pBuf;
	FLMUINT			uiBufSize;
	FLMUINT			uiBufUsed;
	FLMUINT			uiCurrBuffOffset;
	FLMUINT64		ui64FilePos;
	FLMBOOL			bDictRecords;
	FLMBOOL			bBufDirty;
} EXP_IMP_INFO;

/****************************************************************************
Desc:	The ND2BF structure is used to convert a NODE into a
		buffer and is used within GedNodeToBuf.
****************************************************************************/
typedef struct ND2BF
{
	FLMBYTE * 			buffer;
	FLMINT				iLimit;
	F_NameTable *		pNameTable;
} ND2BF;

/****************************************************************************
Desc:
****************************************************************************/
typedef struct QUERY_HDR
{
	HFCURSOR		hCursor;
	QUERY_HDR *	pNext;
	QUERY_HDR *	pPrev;
} QUERY_HDR;

/****************************************************************************
Desc:
****************************************************************************/
typedef enum
{
	HASH_SESSION_OBJ = 0,
	HASH_DB_OBJ
} eHashObjType;

/****************************************************************************
Desc: FLAIM session database object
****************************************************************************/
class F_SessionDb : public F_HashObject
{
public:

	F_SessionDb();

	virtual ~F_SessionDb();

	RCODE setupSessionDb(
		F_Session *	pSession,
		HFDB			hDb);

	const void * FLMAPI getKey( void);
	
	FLMUINT FLMAPI getKeyLength( void);

	FINLINE HFDB getDbHandle( void)
	{
		return( m_hDb);
	}

	FINLINE FLMUINT FLMAPI getObjectType( void)
	{
		return( HASH_DB_OBJ);
	}

private:

	F_Session *		m_pSession;
	HFDB				m_hDb;
#define F_SESSION_DB_KEY_LEN		((sizeof( FLMUINT) * 5))
	FLMBYTE			m_ucKey[ F_SESSION_DB_KEY_LEN];

friend class F_Session;
};

/****************************************************************************
Desc: FLAIM session object
****************************************************************************/
class F_Session : public F_HashObject
{
public:

	F_Session();

	virtual ~F_Session();

	RCODE setupSession(
		F_SessionMgr *		pSessionMgr);

	// Database handles

	RCODE addDbHandle(
		HFDB					hDb,
		char *				pucKey = NULL);

	void closeDb(
		const char *		pucKey);

	RCODE getDbHandle(
		const char *		pucKey,
		HFDB *				phDb);

	RCODE getNextDb(
		F_SessionDb **		ppSessionDb);

	void releaseFileResources(
		FFILE *				pFile);

	// Misc.

	RCODE getNameTable(
		HFDB					hDb,
		F_NameTable **		ppNameTable);

	RCODE getNameTable(
		FFILE *				pFile,
		F_NameTable **		ppNameTable);

	FLMUINT getNextToken( void);

	const void * FLMAPI getKey( void);
	
	FLMUINT FLMAPI getKeyLength( void);

	FLMINT FLMAPI AddRef();

	FLMINT FLMAPI Release();

	FINLINE FLMUINT FLMAPI getObjectType( void)
	{
		return( HASH_SESSION_OBJ);
	}

private:

	RCODE lockSession(
		FLMBOOL				bWait = TRUE);
	
	void unlockSession( void);

	void signalLockWaiters(
		RCODE					rc,
		FLMBOOL				bMutexLocked);

	F_SessionMgr *			m_pSessionMgr;
	F_Session *				m_pNext;
	F_Session *				m_pPrev;
	FLMUINT					m_uiLastUsed;
	FLMUINT					m_uiThreadId;
	FLMUINT					m_uiThreadLockCount;
	F_MUTEX					m_hMutex;
	F_NOTIFY_LIST_ITEM *	m_pNotifyList;
	F_NameTable *			m_pNameTable;
	FLMUINT					m_uiDictSeqNum;
	FLMUINT					m_uiNameTableFFileId;
	FLMUINT					m_uiNextToken;
	F_HashTable *			m_pDbTable;
#define F_SESSION_KEY_LEN			((sizeof( FLMUINT) * 5))
	FLMBYTE					m_ucKey[ F_SESSION_KEY_LEN];

friend class F_SessionMgr;
};

/****************************************************************************
Desc: FLAIM session manager object
****************************************************************************/
class F_SessionMgr : public F_Object
{
public:

	F_SessionMgr()
	{
		m_hMutex = F_MUTEX_NULL;
		m_pSessionTable = NULL;
		m_uiNextId = 1;
		f_timeGetSeconds( &m_uiNextToken);
	}

	virtual ~F_SessionMgr();

	RCODE setupSessionMgr( void);

	RCODE getSession(
		const char *	pszKey,
		F_Session **	ppSession);

	void releaseSession(
		F_Session **	ppSession);

	RCODE createSession(
		F_Session **	ppSession);

	void shutdownSessions();

	void releaseFileResources(
		FFILE *			pFile);

#define MAX_SESSION_INACTIVE_SECS		((FLMUINT)5 * 60)
	void timeoutInactiveSessions(
		FLMUINT			uiInactiveSecs,
		FLMBOOL			bWaitForLocks);

	FINLINE FLMUINT getNextToken( void)
	{
		FLMUINT		uiToken;

		f_mutexLock( m_hMutex);
		uiToken = m_uiNextToken++;
		f_mutexUnlock( m_hMutex);

		return( uiToken);
	}

private:

	F_MUTEX				m_hMutex;
	FLMUINT				m_uiNextId;
	F_HashTable *		m_pSessionTable;
	FLMUINT				m_uiNextToken;
};

#include "fpackoff.h"

#endif
