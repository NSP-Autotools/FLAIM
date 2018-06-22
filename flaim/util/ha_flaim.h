//-------------------------------------------------------------------------
// Desc:	FLAIM handler for MySQL
// Tabs:	3
//
// Copyright (c) 2006-2007 Novell, Inc. All Rights Reserved.
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

#ifndef HA_FLAIM_H
#define HA_FLAIM_H

#include "../flaim/flaim.h"

#ifdef USE_PRAGMA_INTERFACE
	#pragma interface
#endif

#define FLM_MAX_KEY_LENGTH		256

typedef struct
{
	FLMUINT		uiDictNum;
	FLMUINT		uiDataType;
} FIELD_INFO;

typedef struct
{
	FLMUINT		uiDictNum;
} INDEX_INFO;

typedef struct
{
  char *				pszTablePath;
  uint				uiTablePathLen;
  uint				uiUseCount;
  FLMUINT			uiTableContainer;
  FIELD_INFO *		pColumnFields;
  FIELD_INFO *		pKeyFields;
  INDEX_INFO *		pIndexes;
  pthread_mutex_t hMutex;
  THR_LOCK			lock;
} FLAIM_SHARE;

class FlmConnectionTable;
	
/****************************************************************************
Desc:
****************************************************************************/
class FlmConnection : public F_Base
{
public:

	FlmConnection( FlmConnectionTable * pConnTbl);

	virtual ~FlmConnection();

	inline FLMUINT getThreadId( void)
	{
		return( m_uiThreadId);
	}

	inline void setThreadId(
		FLMUINT		uiThreadId)
	{
		m_uiThreadId = uiThreadId;
	}

	inline FlmConnection * getNext( void)
	{
		return( m_pNext);
	}

	inline void setDb(
		HFDB		hDb)
	{
		assert( m_hDb == HFDB_NULL);
		m_hDb = hDb;
	}

	inline HFDB getDb( void)
	{
		return( m_hDb);
	}

	FLMUINT Release(
		FLMBOOL		bMutexLocked);

	inline FLMUINT Release( void)
	{
		return( Release( FALSE));
	}

	inline FLMUINT getTransType( void)
	{
		FLMUINT		uiTransType;

		if( m_hDb == HFDB_NULL ||
			RC_BAD( FlmDbGetTransType( m_hDb, &uiTransType)))
		{
			return( FLM_NO_TRANS);
		}

		return( uiTransType);
	}

	inline void setTransTypeNeeded(
		FLMUINT		uiTransTypeNeeded)
	{
		m_uiTransTypeNeeded = uiTransTypeNeeded;
	}

	inline FLMUINT getTransTypeNeeded( void)
	{
		return( m_uiTransTypeNeeded);
	}

	inline FLMUINT incLockCount( void)
	{
		return( ++m_uiLockCount);
	}

	inline FLMUINT decLockCount( void)
	{
		assert( m_uiLockCount);
		return( --m_uiLockCount);
	}

	inline FLMUINT getLockCount( void)
	{
		return( m_uiLockCount);
	}

	inline void setLockCount(
		FLMUINT		uiLockCount)
	{
		m_uiLockCount = uiLockCount;
	}

	RCODE storeRowCount(
		FLMUINT					uiTableId,
		FLMUINT					uiRowCount);

	RCODE retrieveRowCount(
		FLMUINT					uiTableId,
		FLMUINT *				puiRowCount);

	RCODE incrementRowCount(
		FLMUINT					uiTableId);

	RCODE decrementRowCount(
		FLMUINT					uiTableId);

	inline FLMBOOL getAbortFlag( void)
	{
		return( m_bAbortFlag);
	}

	inline void setAbortFlag( void)
	{
		m_bAbortFlag = TRUE;
	}

	inline void clearAbortFlag( void)
	{
		m_bAbortFlag = FALSE;
	}

private:

	FLMUINT					m_uiThreadId;
	HFDB						m_hDb;
	FlmConnection *		m_pPrev;
	FlmConnection *		m_pNext;
	FlmConnectionTable *	m_pConnTbl;
	FLMUINT					m_uiTransTypeNeeded;
	FLMUINT					m_uiLockCount;
	FLMBOOL					m_bAbortFlag;

friend class FlmConnectionTable;
};

/****************************************************************************
Desc:
****************************************************************************/
class FlmConnectionTable : public F_Base
{
public:

	FlmConnectionTable()
	{
		m_pConnList = NULL;
		m_bFreeMutex = FALSE;
	}

	virtual ~FlmConnectionTable()
	{
		FlmConnection *	pConn = m_pConnList;

		while( pConn)
		{
			pConn->Release( TRUE);
			pConn = m_pConnList;
		}

		if( m_bFreeMutex)
		{
			pthread_mutex_destroy( &m_hMutex);
		}
	}

	RCODE setup( void);

	RCODE getConnection(
		const char *			pszTablePath,
		FlmConnection **		ppConnection);

	RCODE closeConnection(
		FlmConnection *		pConn,
		FLMBOOL					bMutexLocked);

private:

	inline void lockMutex( void)
	{
		pthread_mutex_lock( &m_hMutex);
	}

	inline void unlockMutex( void)
	{
		pthread_mutex_unlock( &m_hMutex);
	}

	FlmConnection *	m_pConnList;
	pthread_mutex_t	m_hMutex;
	FLMBOOL				m_bFreeMutex;

friend class FlmConnection;
};

/****************************************************************************
Desc:
****************************************************************************/
class ha_flaim : public handler
{
public:
	
	ha_flaim( TABLE * table_arg);
	
	virtual ~ha_flaim()
	{
		if( m_pConn)
		{
			m_pConn->Release();
		}

		if( m_pCurrKey)
		{
			m_pCurrKey->Release();
		}
	}
	
	// The name that will be used for display purposes
	
	inline const char * table_type( void) const
	{ 
		return( "FLAIM");
	}
	
	// The name of the index type that will be used for display
	// don't implement this method unless you really have indexes
	
	inline const char * index_type( uint inx)
	{
		return( "BTREE");
	}
	
	const char ** bas_ext( void) const;

	// The storage engine supports transactions
	
	inline bool has_transactions( void)
	{
		return( true);
	}

	// This is a list of flags that says what the storage engine
	// implements. The current table flags are documented in
	// handler.h
	
	inline ulong table_flags( void) const
	{
		return( 0);
	}
	
	// This is a bitmap of flags that says how the storage engine
	// implements indexes. The current index flags are documented in
	// handler.h.
	//	
	// part is the key part to check.  First key part is 0
	// If all_parts it's set, MySQL want to know the flags for the combined
	// index up to and including 'part'.
	
	inline ulong index_flags(
		uint				uiIndex, 
		uint 				uiPart,
		bool 				bAllParts) const
	{
		return( HA_READ_NEXT | HA_READ_ORDER | HA_READ_RANGE);
	}
	
	// unireg.cpp will call the following to make sure that the storage engine
	// can handle the data it is about to send.
	//	
	// Return *real* limits of your storage engine here. MySQL will do
	// min( your_limits, MySQL_limits) automatically
	//	
	// There is no need to implement ..._key_... methods if you don't support
	// indexes.
	
	inline uint max_supported_record_length( void) const
	{ 
		return( HA_MAX_REC_LENGTH);
	}
	
	inline uint max_supported_keys( void) const
	{
		return( ~0);
	}
	
	inline uint max_supported_key_parts( void) const
	{ 
		return( 32);
	}
	
	inline uint max_supported_key_length( void) const
	{
		return( FLM_MAX_KEY_LENGTH);
	}
	
	// Called in test_quick_select to determine if indexes should be used.
	
	inline double scan_time( void)
	{
		return( 0);
	}
	
	// The next method will never be called if you do not implement indexes.
	
	inline double read_time( 
		ha_rows					rows)
	{
		return( 0);
	}
	
	// Everything below are methods that we implment in ha_flaim.cpp.
	//
	// Most of these methods are not obligatory, skip them and
	// MySQL will treat them as not implemented
	
	int open( 
		const char *			name,
		int 						mode,
		uint 						test_if_locked);
	
	int close( void);
	
	int write_row(
		byte * 					pucData);
	
	int update_row(
		const byte * 			pucOldData,
		byte * 					pucNewData);
	
	int delete_row(
		const byte * 			pucData);
	
	int index_init(
		uint						uiIndex);

	int index_end( void);

	int index_read(
		byte *					pucData, 
		const byte * 			pucKey,
		uint 						uiKeyLen,
		enum ha_rkey_function eFindFlag);
	
	int index_next(
		byte *					pucData);
	
	int index_first(
		byte *					pucData);
	
	int index_last(
		byte *					pucData);
	
	// unlike index_init(), rnd_init() can be called two times
	// without rnd_end() in between (it only makes sense if scan=1).
	// then the second call should prepare for the new table scan
	// (e.g if rnd_init allocates the cursor, second call should
	// position it to the start of the table, no need to deallocate
	// and allocate it again
	
	int rnd_init(
		bool						bScan);
	
	int rnd_end( void);
	
	int rnd_next(
		byte *					pucData);
	
	int rnd_pos(
		byte *					pucData,
		byte *					pucPos);
	
	void position(
		const byte *			pucRecord);
	
	void info( uint);
	
	int extra( 
		enum ha_extra_function eOperation);
	
	int reset( void);
	
	int external_lock(
		THD *						pThread,
		int 						iLockType);
	
	int delete_all_rows( void);
	
	ha_rows records_in_range(
		uint						uiIndex,
		key_range *				pMinKey,
		key_range *				pMaxKey);
	
	int delete_table(
		const char *			pszTablePath);
	
	int rename_table(
		const char *			pszOldName,
		const char *			pszNewName);
	
	int create(
		const char *			pszTableFormPath,
		TABLE *					pTable,
		HA_CREATE_INFO *		pCreateInfo);
	
	THR_LOCK_DATA ** store_lock(
		THD *						pThread,
		THR_LOCK_DATA **		pLockData,
		enum thr_lock_type	eLockType);

	RCODE exportRowToRec(
		FLMBYTE *				pucData,
		FlmRecord **			ppRec);

	RCODE importRowFromRec(
		FlmRecord *				pRec,
		FLMBYTE *				pucData);

	RCODE exportKeyToRec(
		uint						uiKeyNum,
		const byte *			pucRecord,
		FlmRecord *				pRec);

	RCODE exportKeyToTree(
		FLMUINT					uiIndexOffset,
		const byte *			pucKey,
		FLMUINT					uiKeyLen,
		FlmRecord **			ppKeyTree);

	inline FLMBOOL isFieldNull(
		TABLE *					pTable,
		Field *					pField,
		const char *			pucRow)
	{
		FLMUINT		uiNullOffset;

		if( !pField->null_ptr)
		{
			return( FALSE);
		}

		uiNullOffset = (FLMUINT)((char *)pField->null_ptr - 
										 (char *)pTable->record[ 0]);
		
		assert( uiNullOffset == 0);

		if( pucRow[ uiNullOffset] & pField->null_bit)
		{
			return( TRUE);
		}

		return( FALSE);
	}

	bool convertString(
		String *					pString, 
		CHARSET_INFO *			pFromCharSet,
		CHARSET_INFO *			pToCharSet);

	int check(
		THD *						pThread,
		HA_CHECK_OPT *			pCheckOpt);

	inline int backup(
		THD *						pThread,
		HA_CHECK_OPT *			pCheckOpt)
	{
		return( 0);
	}

private:

	FLAIM_SHARE *		m_pShare;
	THR_LOCK_DATA		m_lockData;
	FlmConnection *	m_pConn;
	FlmRecord *			m_pCurrKey;
	FLMUINT				m_uiCurrRowDrn;
	String				m_convertBuffer;
	char					m_szDbPath[ F_PATH_MAX_SIZE];
	char					m_pucKeyBuf[ FLM_MAX_KEY_LENGTH];
};
	
bool flaim_init( void);

bool flaim_end( void);

int flaim_start_trx_and_assign_read_view(
	THD *			pThread);
	
#endif
