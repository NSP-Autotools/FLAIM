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

#ifdef USE_PRAGMA_IMPLEMENTATION
	#pragma implementation
#endif

#include "mysql_priv.h"

//#ifdef HAVE_FLAIM

#include "ha_flaim.h"

#define FLAIM_DB_NAME						"flaim.db"
#define FLM_TABLE_INFO_FIELD_NAME		"table_info"
#define FLM_TABLE_INFO_FIELD_ID			1
#define FLM_ROW_CONTEXT_FIELD_NAME		"row"
#define FLM_ROW_CONTEXT_FIELD_ID			2
#define FLM_ROW_NULL_BITS_FIELD_NAME	"null_bits"
#define FLM_ROW_NULL_BITS_FIELD_ID		3
#define FLM_ROW_COUNT_FIELD_NAME			"row_count"
#define FLM_ROW_COUNT_FIELD_ID			4

const char * gv_pszBaseDict =
	"0 @1@ field " FLM_TABLE_INFO_FIELD_NAME "\n" \
	"   1 type context\n" \
	"0 @2@ field " FLM_ROW_CONTEXT_FIELD_NAME "\n" \
	"   1 type context\n" \
	"0 @3@ field " FLM_ROW_NULL_BITS_FIELD_NAME "\n" \
	"   1 type binary\n" \
	"0 @4@ field " FLM_ROW_COUNT_FIELD_NAME "\n" \
	"   1 type number";

static int flaim_commit(
	THD *			pThread, 
	bool			bAll);

static int flaim_rollback(
	THD *			pThread, 
	bool			bAll);

FlmConnectionTable *		gv_pConnTbl = NULL;
pthread_mutex_t			gv_hShareMutex;
static HASH					gv_openTables;

/****************************************************************************
Desc:
****************************************************************************/
static const char * ha_flaim_exts[] =
{
  NullS
};

/****************************************************************************
Desc:
****************************************************************************/
handlerton flaim_hton =
{
  "FLAIM",							// name
  SHOW_OPTION_YES,				// state
  "FLAIM storage engine",		// comment
  DB_TYPE_FLAIM,					// db_type
  flaim_init,						// init
  0,									// slot
  0,									// savepoint size
  NULL,								// close connection
  NULL,								// savepoint
  NULL,								// rollback to savepoint
  NULL,								// release savepoint
  flaim_commit,					// commit
  flaim_rollback,					// rollback
  NULL,								// prepare
  NULL,								// recover
  NULL,								// commit_by_xid
  NULL,								// rollback_by_xid
  NULL,								// create_cursor_read_view
  NULL,								// set_cursor_read_view
  NULL,								// close_cursor_read_view
  HTON_NO_FLAGS					// flags
};

/****************************************************************************
Desc:
****************************************************************************/
inline void buildDbPathFromTablePath(
	const char *		pszTablePath,
	char *				pszDbPath)
{
	f_pathReduce( pszTablePath, pszDbPath, NULL);
	f_pathAppend( pszDbPath, FLAIM_DB_NAME);
}

/****************************************************************************
Desc:
****************************************************************************/
static byte * flmShareGetKey(
	FLAIM_SHARE *		pShare,
	uint *				puiLength,
	my_bool				not_used __attribute__((unused)))
{
  *puiLength = pShare->uiTablePathLen;
  return( (byte *)pShare->pszTablePath);
}

/****************************************************************************
Desc:
****************************************************************************/
static RCODE flmGetShare(
	FlmConnection *	pConn,
	const char *		pszTablePath,
	TABLE *				pTable,
	FLAIM_SHARE **		ppShare)
{
	RCODE					rc = FERR_OK;
	FLAIM_SHARE *		pShare;
	FLMBOOL				bCreatedMutex = FALSE;
	FLMBOOL				bShareLocked = FALSE;
	F_NameTable			nameTable;
	Field **				ppMyField;
	FIELD_INFO *		pColumnFields;
	FIELD_INFO *		pKeyFields;
	INDEX_INFO *		pIndexes;
	FLMUINT				uiLoop;
	uint					uiLength;
	char *				pszTmpPath;
	char					szTmpBuf[ 512];

	*ppShare = NULL;

	pthread_mutex_lock( &gv_hShareMutex);
	bShareLocked = TRUE;

	uiLength = (uint)strlen( pszTablePath);

	if( (pShare = (FLAIM_SHARE *)hash_search( &gv_openTables,
		(byte *)pszTablePath, uiLength)) == NULL)
	{
		if( !(pShare = (FLAIM_SHARE *)my_multi_malloc(
			MYF( MY_WME | MY_ZEROFILL), &pShare, sizeof( *pShare),
			&pszTmpPath, uiLength + 1, 
			&pColumnFields, pTable->s->fields * sizeof( FIELD_INFO), 
			&pKeyFields, pTable->s->keys * sizeof( FIELD_INFO),
			&pIndexes, pTable->s->keys * sizeof( INDEX_INFO),
			NullS)))
		{
			rc = FERR_MEM;
			goto Exit;
		}

		pShare->uiUseCount = 0;
		pShare->uiTablePathLen = uiLength;
		pShare->pszTablePath = pszTmpPath;
		strmov( pShare->pszTablePath, pszTablePath);
		pShare->pColumnFields = pColumnFields;
		pShare->pKeyFields = pKeyFields;
		pShare->pIndexes = pIndexes;

		if( RC_BAD( rc = nameTable.setupFromDb( pConn->getDb())))
		{
			goto Exit;
		}

		sprintf( szTmpBuf, ":%s:", pTable->s->table_name);

		if( !nameTable.getFromTagTypeAndName( NULL, szTmpBuf, 
			FLM_CONTAINER_TAG, &pShare->uiTableContainer))
		{
			rc = FERR_DATA_ERROR;
			goto Exit;
		}

		for( ppMyField = pTable->field; *ppMyField; ppMyField++)
		{
			FIELD_INFO *	pFieldInfo = &pShare->pColumnFields[ 
														(*ppMyField)->field_index];

			sprintf( szTmpBuf, ":%s:col:%s:", 
						pTable->s->table_name, (*ppMyField)->field_name);

			if( !nameTable.getFromTagTypeAndName( NULL, szTmpBuf, FLM_FIELD_TAG, 
				&pFieldInfo->uiDictNum, &pFieldInfo->uiDataType))
			{
				rc = FERR_DATA_ERROR;
				goto Exit;
			}
		}

		for( uiLoop = 0; uiLoop < pTable->s->keys; uiLoop++)
		{
			sprintf( szTmpBuf, ":%s:key:%u:", pTable->s->table_name, uiLoop);

			if( !nameTable.getFromTagTypeAndName( NULL, szTmpBuf, FLM_FIELD_TAG, 
				&pShare->pKeyFields[ uiLoop].uiDictNum,
				&pShare->pKeyFields[ uiLoop].uiDataType))
			{
				rc = FERR_DATA_ERROR;
				goto Exit;
			}

			sprintf( szTmpBuf, ":%s:index:%u:", pTable->s->table_name, uiLoop);

			if( !nameTable.getFromTagTypeAndName( NULL, szTmpBuf, FLM_INDEX_TAG, 
				&pShare->pIndexes[ uiLoop].uiDictNum, NULL))
			{
				rc = FERR_DATA_ERROR;
				goto Exit;
			}
		}

		if( my_hash_insert( &gv_openTables, (byte *)pShare))
		{
			rc = FERR_FAILURE;
			goto Exit;
		}

		thr_lock_init( &pShare->lock);
		bCreatedMutex = TRUE;
		pthread_mutex_init( &pShare->hMutex, MY_MUTEX_INIT_FAST);
	}

	pShare->uiUseCount++;
	pthread_mutex_unlock( &gv_hShareMutex);
	bShareLocked = FALSE;

	*ppShare = pShare;

Exit:

	if( RC_BAD( rc))
	{
		if( pShare)
		{
			if( bCreatedMutex)
			{
				pthread_mutex_destroy( &pShare->hMutex);
			}

			my_free( (gptr)pShare, MYF(0));
		}
	}

	if( bShareLocked)
	{
		pthread_mutex_unlock( &gv_hShareMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
static int flmReleaseShare(
	FLAIM_SHARE *		pShare)
{
	pthread_mutex_lock( &gv_hShareMutex);
	if( --pShare->uiUseCount == 0)
	{
		hash_delete( &gv_openTables, (byte *)pShare);
		thr_lock_delete( &pShare->lock);
		pthread_mutex_destroy( &pShare->hMutex);
		my_free( (gptr)pShare, MYF(0));
	}

	pthread_mutex_unlock( &gv_hShareMutex);
	return( 0);
}
/****************************************************************************
Desc:
****************************************************************************/
bool flaim_init( void)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bCleanup = FALSE;
	
	DBUG_ENTER( "flmInit");

	if( have_flaim != SHOW_OPTION_YES)
	{
		goto Exit;
	}

	if( RC_BAD( rc = FlmStartup()))
	{
		goto Exit;
	}

	bCleanup = TRUE;

	if( (gv_pConnTbl = new FlmConnectionTable) == NULL)
	{
		rc = FERR_FAILURE;
		goto Exit;
	}

	if( RC_BAD( rc = gv_pConnTbl->setup()))
	{
		goto Exit;
	}

	if( (pthread_mutex_init( &gv_hShareMutex, MY_MUTEX_INIT_FAST)) != 0)
	{
		rc = FERR_FAILURE;
		goto Exit;
	}

	if( (hash_init( &gv_openTables, system_charset_info, 32, 0, 0,
		(hash_get_key)flmShareGetKey, 0, 0)) != 0)
	{
		rc = FERR_FAILURE;
		goto Exit;
	}

	DBUG_RETURN( FALSE);
  
Exit:

	if( bCleanup)
	{
		flaim_end();
	}

	have_flaim = SHOW_OPTION_DISABLED;
	DBUG_RETURN( TRUE);
}

/****************************************************************************
Desc:
****************************************************************************/
bool flaim_end( void)
{
	DBUG_ENTER( "flaim_end");

	if( gv_pConnTbl)
	{
		gv_pConnTbl->Release();
		gv_pConnTbl = NULL;
	}

	FlmShutdown();
	DBUG_RETURN( FALSE);
}

/****************************************************************************
Desc:
****************************************************************************/
ha_flaim::ha_flaim( TABLE * table_arg) : handler( &flaim_hton, table_arg)
{
	m_pConn = NULL;
	m_pShare = NULL;
	m_uiCurrRowDrn = 0;
	active_index = MAX_KEY;
	m_pCurrKey = NULL;
	m_szDbPath[ 0] = 0;
}

/****************************************************************************
Desc:
****************************************************************************/
const char ** ha_flaim::bas_ext( void) const
{
  return( ha_flaim_exts);
}

/****************************************************************************
Desc: Used for opening tables. The name will be the name of the file.
		A table is opened when it needs to be opened. For instance
		when a request comes in for a select on the table (tables are not
		open and closed for each request, they are cached).
****************************************************************************/
int ha_flaim::open(
	const char *		pszTablePath, 
	int 					iMode,
	uint 					bTestIfLocked)
{
	RCODE					rc = FERR_OK;

	DBUG_ENTER( "ha_flaim::open");

	buildDbPathFromTablePath( pszTablePath, m_szDbPath);

	if( RC_BAD( rc = gv_pConnTbl->getConnection( m_szDbPath, &m_pConn)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = flmGetShare( m_pConn, pszTablePath, table, &m_pShare)))
	{
		goto Exit;
	}

	thr_lock_data_init( &m_pShare->lock, &m_lockData, NULL);
	ref_length = sizeof( FLMUINT32);

Exit:

	DBUG_RETURN( rc);
}

/****************************************************************************
Desc:	Closes a table
****************************************************************************/
int ha_flaim::close(void)
{
	DBUG_ENTER( "ha_flaim::close");

	flmReleaseShare( m_pShare);
	m_pShare = NULL;
	m_szDbPath[ 0] = 0;
	m_uiCurrRowDrn = 0;
	active_index = MAX_KEY;

	if( m_pConn)
	{
		m_pConn->Release();
		m_pConn = NULL;
	}

	if( m_pCurrKey)
	{
		m_pCurrKey->Release();
		m_pCurrKey = NULL;
	}

	DBUG_RETURN( 0);
}

/****************************************************************************
Desc:	Inserts a row. The buffer is a byte array of data.  The field
		information can be used to extract the data from the native byte
		array.
****************************************************************************/
int ha_flaim::write_row(
	byte * 				pucData)
{
	RCODE					rc = FERR_OK;
	FlmRecord *			pRec = NULL;
	FLMBOOL				bMustAbortOnError = FALSE;

	DBUG_ENTER( "ha_flaim::write_row");

	if( RC_BAD( rc = gv_pConnTbl->getConnection( m_szDbPath, &m_pConn)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = exportRowToRec( pucData, &pRec)))
	{
		goto Exit;
	}

	bMustAbortOnError = TRUE;

	m_uiCurrRowDrn = 0;
	if( RC_BAD( rc = FlmRecordAdd( m_pConn->getDb(),
		m_pShare->uiTableContainer, &m_uiCurrRowDrn, pRec, 0)))
	{
		if( rc == FERR_NOT_UNIQUE)
		{
			rc = (RCODE)HA_ERR_FOUND_DUPP_KEY;
			bMustAbortOnError = FALSE;
		}

		goto Exit;
	}

	if( RC_BAD( rc = m_pConn->incrementRowCount( 
		m_pShare->uiTableContainer)))
	{
		goto Exit;
	}

	statistic_increment( table->in_use->status_var.ha_write_count, 
		&LOCK_status);

	// If we have a timestamp column, update it to the current time
  
	if( table->timestamp_field_type & TIMESTAMP_AUTO_SET_ON_INSERT)
	{
	  table->timestamp_field->set_time();
	}
 
  // If we have an auto_increment column and we are writing a changed row
  // or a new row, then update the auto_increment value in the record.
  
	if( table->next_number_field && pucData == table->record[ 0])
	{
	  update_auto_increment();
	}
	
Exit:

	if( pRec)
	{
		pRec->Release();
	}

	if( RC_BAD( rc) && bMustAbortOnError)
	{
		m_pConn->setAbortFlag();
	}

	DBUG_RETURN( rc);
}

/****************************************************************************
Desc:	Updates a row. old_data will have the previous row record in
		it, while new_data will have the newest data in it.
****************************************************************************/
int ha_flaim::update_row(
	const byte * 		pucOldData,
	byte * 				pucNewData)
{
	RCODE					rc = FERR_OK;
	FlmRecord *			pRec = NULL;
	FLMBOOL				bMustAbortOnError = FALSE;

	DBUG_ENTER( "ha_flaim::update_row");
	
	if( RC_BAD( rc = gv_pConnTbl->getConnection( m_szDbPath, &m_pConn)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = exportRowToRec( pucNewData, &pRec)))
	{
		goto Exit;
	}

	bMustAbortOnError = TRUE;

	if( RC_BAD( rc = FlmRecordModify( m_pConn->getDb(), 
		m_pShare->uiTableContainer, m_uiCurrRowDrn, pRec, 0)))
	{
		if( rc == FERR_NOT_UNIQUE)
		{
			rc = (RCODE)HA_ERR_FOUND_DUPP_KEY;
			bMustAbortOnError = FALSE;
		}

		goto Exit;
	}
	
	statistic_increment(table->in_use->status_var.ha_read_rnd_next_count,
 		      &LOCK_status);
 
	// If we have a timestamp column, update it to the current time
  
	if (table->timestamp_field_type & TIMESTAMP_AUTO_SET_ON_UPDATE)
	{
		table->timestamp_field->set_time();
	}
	
Exit:
 
	if( pRec)
	{
		pRec->Release();
	}

	if( RC_BAD( rc) && bMustAbortOnError)
	{
		m_pConn->setAbortFlag();
	}

	DBUG_RETURN( rc);
}

/****************************************************************************
Desc:	This will delete a row.  pucData will contain a copy of the row to
		be deleted.
****************************************************************************/
int ha_flaim::delete_row( 
	const byte *		pucData)
{
	RCODE					rc = FERR_OK;
	FLMBOOL				bMustAbortOnError = FALSE;

	DBUG_ENTER( "ha_flaim::delete_row");

	if( RC_BAD( rc = gv_pConnTbl->getConnection( m_szDbPath, &m_pConn)))
	{
		goto Exit;
	}

	bMustAbortOnError = TRUE;

	if( RC_BAD( rc = FlmRecordDelete( m_pConn->getDb(), 
		m_pShare->uiTableContainer, m_uiCurrRowDrn, 0)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pConn->decrementRowCount( 
		m_pShare->uiTableContainer)))
	{
		goto Exit;
	}

   statistic_increment( table->in_use->status_var.ha_delete_count,
                       &LOCK_status);

Exit:

	if( RC_BAD( rc) && bMustAbortOnError)
	{
		m_pConn->setAbortFlag();
	}
 
	DBUG_RETURN( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
int ha_flaim::index_init(
	uint						uiIndex)
{
	DBUG_ENTER( "ha_flaim::index_init");

	active_index = uiIndex;
	assert( m_pCurrKey == NULL);

	DBUG_RETURN( 0);
}

/****************************************************************************
Desc:
****************************************************************************/
int ha_flaim::index_end( void)
{
	DBUG_ENTER( "ha_flaim::index_end");

	active_index = MAX_KEY;

	if( m_pCurrKey)
	{
		m_pCurrKey->Release();
		m_pCurrKey = NULL;
	}

	DBUG_RETURN( 0);
}

/****************************************************************************
Desc:	Positions an index cursor to the index specified in the handle.
		Fetches the row if available.  If the key value is null, begin at
		the first key of the index.
****************************************************************************/
int ha_flaim::index_read(
	byte *						pucData,
	const byte *				pucKey,
	uint							uiKeyLen,
	enum ha_rkey_function	eFindFlag)
{
	RCODE					rc = FERR_OK;
	FlmRecord *			pKeyRec = NULL;
	FlmRecord *			pRec = NULL;
	FLMUINT				uiReference;
	FLMUINT				uiFindFlags = 0;
	void *				pvField;

	DBUG_ENTER( "ha_flaim::index_read");

	if( RC_BAD( rc = gv_pConnTbl->getConnection( m_szDbPath, &m_pConn)))
	{
		goto Exit;
	}

	if( pucKey)
	{
		if( RC_BAD( rc = exportKeyToTree( active_index, 
			pucKey, uiKeyLen, &pKeyRec)))
		{
			goto Exit;
		}

		switch( eFindFlag)
		{
			case HA_READ_KEY_EXACT:
			{
				uiFindFlags = FO_INCL;
				break;
			}

			case HA_READ_KEY_OR_NEXT:
			{
				uiFindFlags = FO_INCL;
				break;
			}

			case HA_READ_AFTER_KEY:
			{
				uiFindFlags = FO_EXCL;
				break;
			}

			default:
			{
				rc = (RCODE)HA_ERR_WRONG_COMMAND;
				goto Exit;
			}
		}
	}
	else
	{
		uiFindFlags = FO_FIRST;
	}

	if( RC_BAD( rc = FlmKeyRetrieve( m_pConn->getDb(), 
		m_pShare->pIndexes[ active_index].uiDictNum, 0, 
		pKeyRec, 0, uiFindFlags, &m_pCurrKey, &uiReference)))
	{
		if( rc == FERR_NOT_FOUND)
		{
			rc = (RCODE)HA_ERR_KEY_NOT_FOUND;
			table->status = STATUS_NOT_FOUND;
		}
		else if( rc == FERR_EOF_HIT)
		{
			rc = (RCODE)HA_ERR_END_OF_FILE;
			table->status = STATUS_NOT_FOUND;
		}

		goto Exit;
	}

	if( eFindFlag == HA_READ_KEY_EXACT)
	{
		FLMUINT				uiTmpKeyLen = uiKeyLen;
		char *				pucBuf = m_pucKeyBuf;
		char *				pucBufEnd = &m_pucKeyBuf[ FLM_MAX_KEY_LENGTH];
		KEY *					pKey = table->key_info + active_index;
		KEY_PART_INFO *	pCurKeyPart = pKey->key_part;
		KEY_PART_INFO *	pEndKeyPart = pCurKeyPart + pKey->key_parts;

		if( (pvField = m_pCurrKey->find( m_pCurrKey->root(), 
			m_pShare->pKeyFields[ active_index].uiDictNum)) == NULL)
		{
			rc = FERR_DATA_ERROR;
			goto Exit;
		}

		for( ; pCurKeyPart != pEndKeyPart && uiTmpKeyLen > 0; pCurKeyPart++)
		{
			FLMUINT		uiOffset = 0;

			if( pCurKeyPart->null_bit)
			{
				*pucBuf++ = *pucKey ? 1 : 0;
				uiTmpKeyLen -= pCurKeyPart->store_length;
				pucKey += pCurKeyPart->store_length;
				uiOffset = 1;
			}
			
			pucBuf = pCurKeyPart->field->pack_key_from_key_image( pucBuf, 
						(const char *)pucKey + uiOffset, pCurKeyPart->length);
			pucKey += pCurKeyPart->store_length;
			uiTmpKeyLen -= pCurKeyPart->store_length;

			assert( pucBuf <= pucBufEnd);
		}

		uiTmpKeyLen = pucBuf - m_pucKeyBuf;

		if( m_pCurrKey->getDataLength( pvField) < uiTmpKeyLen ||
			memcmp( m_pCurrKey->getDataPtr( pvField), m_pucKeyBuf, uiTmpKeyLen) != 0)
		{
			rc = (RCODE)HA_ERR_END_OF_FILE;
			table->status = STATUS_NOT_FOUND;
			goto Exit;
		}
	}

	if( RC_BAD( rc = FlmRecordRetrieve( m_pConn->getDb(), 
		m_pShare->uiTableContainer, uiReference, 
		FO_EXACT, &pRec, &m_uiCurrRowDrn)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = importRowFromRec( pRec, pucData)))
	{
		goto Exit;
	}

	table->status = 0;

Exit:

	if( pRec)
	{
		pRec->Release();
	}

	if( pKeyRec)
	{
		pKeyRec->Release();
	}

	if( RC_BAD( rc))
	{
		m_uiCurrRowDrn = 0;
	}
 
	DBUG_RETURN( rc);
}

/****************************************************************************
Desc:	Used to read forward through the index
****************************************************************************/
int ha_flaim::index_next(
	byte *			pucData)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiReference;
	FlmRecord *		pRec = NULL;

	DBUG_ENTER( "ha_flaim::index_read");

	if( RC_BAD( rc = gv_pConnTbl->getConnection( m_szDbPath, &m_pConn)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FlmKeyRetrieve( m_pConn->getDb(), 
		m_pShare->pIndexes[ active_index].uiDictNum, 0, 
		m_pCurrKey, 0, FO_EXCL, &m_pCurrKey, &uiReference)))
	{
		if( rc == FERR_NOT_FOUND)
		{
			rc = (RCODE)HA_ERR_KEY_NOT_FOUND;
			table->status = STATUS_NOT_FOUND;
		}
		else if( rc == FERR_EOF_HIT)
		{
			rc = (RCODE)HA_ERR_END_OF_FILE;
			table->status = STATUS_NOT_FOUND;
		}

		goto Exit;
	}

	if( RC_BAD( rc = FlmRecordRetrieve( m_pConn->getDb(), 
		m_pShare->uiTableContainer, uiReference, 
		FO_EXACT, &pRec, &m_uiCurrRowDrn)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = importRowFromRec( pRec, pucData)))
	{
		goto Exit;
	}

	table->status = 0;

Exit:

	if( pRec)
	{
		pRec->Release();
	}

	DBUG_RETURN( rc);
}

/****************************************************************************
Desc:	Returns the first key in the index
****************************************************************************/
int ha_flaim::index_first(
	byte *			pucData)
{
	RCODE					rc = FERR_OK;
	FlmRecord *			pRec = NULL;
	FLMUINT				uiReference;

	DBUG_ENTER( "ha_flaim::index_first");

	if( RC_BAD( rc = gv_pConnTbl->getConnection( m_szDbPath, &m_pConn)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FlmKeyRetrieve( m_pConn->getDb(), 
		m_pShare->pIndexes[ active_index].uiDictNum, 0, 
		NULL, 0, FO_FIRST, &m_pCurrKey, &uiReference)))
	{
		if( rc == FERR_NOT_FOUND)
		{
			rc = (RCODE)HA_ERR_KEY_NOT_FOUND;
			table->status = STATUS_NOT_FOUND;
		}
		else if( rc == FERR_EOF_HIT)
		{
			rc = (RCODE)HA_ERR_END_OF_FILE;
			table->status = STATUS_NOT_FOUND;
		}

		goto Exit;
	}

	if( RC_BAD( rc = FlmRecordRetrieve( m_pConn->getDb(), 
		m_pShare->uiTableContainer, uiReference, 
		FO_EXACT, &pRec, &m_uiCurrRowDrn)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = importRowFromRec( pRec, pucData)))
	{
		goto Exit;
	}

	table->status = 0;

Exit:

	if( pRec)
	{
		pRec->Release();
	}

	if( RC_BAD( rc))
	{
		m_uiCurrRowDrn = 0;
	}
 
	DBUG_RETURN( rc);
}

/****************************************************************************
Desc:	Returns the last key in the index
****************************************************************************/
int ha_flaim::index_last(
	byte *			pucData)
{
	RCODE					rc = FERR_OK;
	FlmRecord *			pRec = NULL;
	FLMUINT				uiReference;

	DBUG_ENTER( "ha_flaim::index_last");

	if( RC_BAD( rc = gv_pConnTbl->getConnection( m_szDbPath, &m_pConn)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FlmKeyRetrieve( m_pConn->getDb(), 
		m_pShare->pIndexes[ active_index].uiDictNum, 0, 
		NULL, 0, FO_LAST, &m_pCurrKey, &uiReference)))
	{
		if( rc == FERR_NOT_FOUND)
		{
			rc = (RCODE)HA_ERR_KEY_NOT_FOUND;
			table->status = STATUS_NOT_FOUND;
		}
		else if( rc == FERR_EOF_HIT)
		{
			rc = (RCODE)HA_ERR_END_OF_FILE;
			table->status = STATUS_NOT_FOUND;
		}

		goto Exit;
	}

	if( RC_BAD( rc = FlmRecordRetrieve( m_pConn->getDb(), 
		m_pShare->uiTableContainer, uiReference, 
		FO_EXACT, &pRec, &m_uiCurrRowDrn)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = importRowFromRec( pRec, pucData)))
	{
		goto Exit;
	}

	table->status = 0;

Exit:

	if( pRec)
	{
		pRec->Release();
	}

	if( RC_BAD( rc))
	{
		m_uiCurrRowDrn = 0;
	}
 
	DBUG_RETURN( rc);
}

/****************************************************************************
Desc:	Called when the system wants the storage engine to do a table scan.
****************************************************************************/
int ha_flaim::rnd_init(
	bool				scan)
{
	DBUG_ENTER( "ha_flaim::rnd_init");
	m_uiCurrRowDrn = 0;
	DBUG_RETURN( 0);
}

/****************************************************************************
Desc:
****************************************************************************/
int ha_flaim::rnd_end( void)
{
	DBUG_ENTER( "ha_flaim::rnd_end");
	m_uiCurrRowDrn = 0;
	DBUG_RETURN( 0);
}

/****************************************************************************
Desc: This is called for each row of the table scan.  When the end of the
		table is hit, HA_ERR_END_OF_FILE is returned.  The buffer is filled
		with the row information.
****************************************************************************/
int ha_flaim::rnd_next(
	byte *				pucData)
{
	RCODE					rc = FERR_OK;
	FlmRecord *			pRec = NULL;

	DBUG_ENTER( "ha_flaim::rnd_next");

	if( RC_BAD( rc = gv_pConnTbl->getConnection( 
		m_szDbPath, &m_pConn)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FlmRecordRetrieve( m_pConn->getDb(), 
		m_pShare->uiTableContainer, 
		m_uiCurrRowDrn, FO_EXCL, &pRec, &m_uiCurrRowDrn)))
	{
		if( rc == FERR_EOF_HIT)
		{
			m_uiCurrRowDrn = 0;
			rc = (RCODE)HA_ERR_END_OF_FILE;
			table->status = STATUS_NOT_FOUND;
		}

		goto Exit;
	}

	if( RC_BAD( rc = importRowFromRec( pRec, pucData)))
	{
		goto Exit;
	}

	table->status = 0;

Exit:

	if( pRec)
	{
		pRec->Release();
	}

	DBUG_RETURN( rc);
}

/****************************************************************************
Desc:	Called after each call to rnd_next() if the data needs
		to be ordered.
****************************************************************************/
void ha_flaim::position( const byte *record)
{
	DBUG_ENTER( "ha_flaim::position");
	my_store_ptr( ref, ref_length, m_uiCurrRowDrn);
	DBUG_VOID_RETURN;
}

/****************************************************************************
Desc:	This is like rnd_next, but a position is specified and is used to
		determine the row.  The position will be of the type that was stored
		in position().
****************************************************************************/
int ha_flaim::rnd_pos(
	byte *			pucData,
	byte *			pucPos)
{
	RCODE				rc = FERR_OK;
	FlmRecord *		pRec = NULL;
	FLMUINT			uiLength;

	DBUG_ENTER( "ha_flaim::rnd_pos");

	statistic_increment( table->in_use->status_var.ha_read_rnd_next_count, 
		&LOCK_status);

	if( RC_BAD( rc = gv_pConnTbl->getConnection( 
		m_szDbPath, &m_pConn)))
	{
		goto Exit;
	}

	m_uiCurrRowDrn = (FLMUINT)my_get_ptr( pucPos, ref_length);

	if( RC_BAD( rc = FlmRecordRetrieve( m_pConn->getDb(), 
		m_pShare->uiTableContainer, 
		m_uiCurrRowDrn, FO_EXACT, &pRec, NULL)))
	{
		goto Exit;
	}

	uiLength = table->s->reclength;
	if( RC_BAD( rc = pRec->getBinary( pRec->root(), 
		pucData, &uiLength)))
	{
		goto Exit;
	}

Exit:

	if( pRec)
	{
		pRec->Release();
	}

  DBUG_RETURN( rc);
}

/****************************************************************************
Desc:	Used to return information to the optimizer.
****************************************************************************/
void ha_flaim::info(
	uint				uiFlag)
{
	RCODE		rc = FERR_OK;

	DBUG_ENTER( "ha_flaim::info");

	if( RC_BAD( rc = gv_pConnTbl->getConnection( m_szDbPath, &m_pConn)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pConn->retrieveRowCount(
		m_pShare->uiTableContainer, &records)))
	{
		goto Exit;
	}

Exit:

	if( RC_BAD( rc))
	{
		records = 2;
	}

	DBUG_VOID_RETURN;
}

/****************************************************************************
Desc:	Called whenever the server wishes to send a hint to the storage
		engine.
****************************************************************************/
int ha_flaim::extra(
	enum ha_extra_function	eOperation)
{
  DBUG_ENTER( "ha_flaim::extra");
  DBUG_RETURN( 0);
}

/****************************************************************************
Desc:
****************************************************************************/
int ha_flaim::reset( void)
{
	DBUG_ENTER( "ha_flaim::reset");
	DBUG_RETURN( 0);
}

/****************************************************************************
Desc:	Used to delete all rows in a table
****************************************************************************/
int ha_flaim::delete_all_rows( void)
{
	RCODE					rc = FERR_OK;
	FlmRecord *			pRec = NULL;
	FLMBOOL				bMustAbortOnError = FALSE;

	DBUG_ENTER( "ha_flaim::delete_all_rows");

	if( RC_BAD( rc = gv_pConnTbl->getConnection( 
		m_szDbPath, &m_pConn)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FlmRecordRetrieve( m_pConn->getDb(),
		FLM_DICT_CONTAINER,
		m_pShare->uiTableContainer, FO_EXACT, &pRec, NULL)))
	{
		goto Exit;
	}

	bMustAbortOnError = TRUE;

	if( RC_BAD( rc = FlmRecordDelete( m_pConn->getDb(),
		FLM_DICT_CONTAINER,
		m_pShare->uiTableContainer, 0)))
	{
		goto Exit;
	}

	if( pRec->isReadOnly())
	{
		FlmRecord *		pTmpRec;

		if( (pTmpRec = pRec->copy()) == NULL)
		{
			rc = FERR_MEM;
			goto Exit;
		}

		pRec->Release();
		pRec = pTmpRec;
	}

	if( RC_BAD( rc = FlmRecordAdd( m_pConn->getDb(),
		FLM_DICT_CONTAINER, 
		&m_pShare->uiTableContainer, pRec, 0)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pConn->storeRowCount( 
		m_pShare->uiTableContainer, 0)))
	{
		goto Exit;
	}

	m_uiCurrRowDrn = 0;

Exit:

	if( pRec)
	{
		pRec->Release();
	}

	if( RC_BAD( rc) && bMustAbortOnError)
	{
		m_pConn->setAbortFlag();
	}

	DBUG_RETURN( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
int ha_flaim::external_lock(
	THD *			pThread, 
	int			iLockType)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bDecLockCountOnError = FALSE;

	DBUG_ENTER( "ha_flaim::external_lock");

	if( RC_BAD( rc = gv_pConnTbl->getConnection( 
 		m_szDbPath, &m_pConn)))
	{
		goto Exit;
	}
	
	if( iLockType != F_UNLCK)
	{
		if( iLockType == F_WRLCK &&
			(m_pConn->getTransType() == FLM_READ_TRANS ||
			m_pConn->getTransTypeNeeded() == FLM_READ_TRANS))
		{
			rc = (RCODE)HA_ERR_READ_ONLY_TRANSACTION;
			goto Exit;
		}			
		
		if( m_pConn->incLockCount() == 1 &&
			m_pConn->getTransType() == FLM_NO_TRANS)
		{
			bDecLockCountOnError = TRUE;

			if( RC_BAD( rc = FlmDbTransBegin( m_pConn->getDb(), 
				m_pConn->getTransTypeNeeded(), FLM_NO_TIMEOUT)))
			{
				goto Exit;
			}

			m_pConn->setTransTypeNeeded( FLM_NO_TRANS);

			if( pThread->options & (OPTION_BEGIN | OPTION_NOT_AUTOCOMMIT))
			{
				trans_register_ha( pThread, TRUE, &flaim_hton);

				// Since this transaction has been registered with
				// MySQL, flaim_commit or flaim_rollback will be called
				// when appropriate.  To prevent external_unlock from
				// releasing the transaction too soon, increment the
				// lock count again.  This allows us to handle the case
				// of a multi-statement transaction.

				m_pConn->incLockCount();
			}
		}
	}
	else
	{
		m_lockData.type = TL_UNLOCK;
		
		if( m_pConn->getLockCount() != 0 && 
			m_pConn->decLockCount() == 0 && 
			m_pConn->getTransType() != FLM_NO_TRANS)
		{
			if( m_pConn->getAbortFlag())
			{
				FlmDbTransAbort( m_pConn->getDb());
				m_pConn->clearAbortFlag();
			}
			else
			{
				if( RC_BAD( rc = FlmDbTransCommit( m_pConn->getDb())))
				{
					goto Exit;
				}
			}
		}
	}

Exit:

	if( RC_BAD( rc))
	{
		if( bDecLockCountOnError)
		{
			m_pConn->decLockCount();
		}
	}

	DBUG_RETURN( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
int flaim_start_trx_and_assign_read_view(
	THD *			pThread)
{
	RCODE					rc = FERR_OK;
	FlmConnection *	pConn = NULL;
	char					szDbPath[ F_PATH_MAX_SIZE];

  	DBUG_ENTER( "flaim_start_trx_and_assign_read_view");

	szDbPath[ 0] = 0;
	f_pathAppend( szDbPath, ".");
	f_pathAppend( szDbPath, pThread->db);
	f_pathAppend( szDbPath, FLAIM_DB_NAME);

	if( RC_BAD( rc = gv_pConnTbl->getConnection( szDbPath, &pConn)))
	{
		goto Exit;
	}

	if( pConn->getLockCount() == 0 &&
		pConn->getTransType() == FLM_NO_TRANS)
	{
		if( RC_BAD( rc = FlmDbTransBegin( pConn->getDb(), 
			FLM_READ_TRANS, FLM_NO_TIMEOUT)))
		{
			goto Exit;
		}

		pConn->setTransTypeNeeded( FLM_NO_TRANS);
		trans_register_ha( pThread, TRUE, &flaim_hton);
		pConn->incLockCount();
	}

Exit:

	if( pConn)
	{
		pConn->Release();
	}

	DBUG_RETURN( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
THR_LOCK_DATA ** ha_flaim::store_lock(
	THD *						pThread,
	THR_LOCK_DATA **		pLockData,
	enum thr_lock_type	eLockType)
{
	if( RC_BAD( gv_pConnTbl->getConnection( 
		m_szDbPath, &m_pConn)))
	{
		assert( 0);
	}

	if( eLockType != TL_IGNORE && m_lockData.type == TL_UNLOCK)
	{
		if( eLockType >= TL_WRITE_ALLOW_WRITE && 
				eLockType <= TL_WRITE_ONLY)
		{
			eLockType = TL_WRITE_ALLOW_READ;
		}
		else if( eLockType >= TL_READ && eLockType <= TL_READ_NO_INSERT)
		{
			eLockType = TL_READ_HIGH_PRIORITY;
		}

		if( eLockType == TL_WRITE_ALLOW_READ ||
			(pThread->options & (OPTION_BEGIN | OPTION_NOT_AUTOCOMMIT)))
		{
			eLockType = TL_WRITE_ALLOW_READ;
			if( m_pConn->getTransType() == FLM_NO_TRANS)
			{
				m_pConn->setTransTypeNeeded( FLM_UPDATE_TRANS);
			}
		}
		else if( m_pConn->getTransTypeNeeded() == FLM_NO_TRANS)
		{
			eLockType = TL_READ_HIGH_PRIORITY;
			if( m_pConn->getTransType() == FLM_NO_TRANS)
			{
				m_pConn->setTransTypeNeeded( FLM_READ_TRANS);
			}
		}

		m_lockData.type = eLockType;
	}

	*pLockData++ = &m_lockData;
	return( pLockData);
}

/****************************************************************************
Desc:	Renames a table
****************************************************************************/
int ha_flaim::rename_table(
	const char *		pszOldName,
	const char *		pszNewName)
{
	RCODE					rc = FERR_OK;
	FLMUINT				uiDictNum;
	F_NameTable			nameTable;
	FlmRecord *			pDictRec = NULL;
	char					szOldPrefix[ F_PATH_MAX_SIZE];
	char					szNewPrefix[ F_PATH_MAX_SIZE];
	char					szTmpBuf[ F_PATH_MAX_SIZE];
	char					szNewName[ F_PATH_MAX_SIZE];
	int					iCmp;
	FLMUINT				uiOldPrefixLen;
	FLMUINT				uiNewPrefixLen;
	FLMUINT				uiLoop;
	FLMBOOL				bStartedTrans = FALSE;
	FLMBOOL				bMustAbortOnError = FALSE;

	DBUG_ENTER( "ha_flaim::rename_table");

	buildDbPathFromTablePath( pszOldName, m_szDbPath);

	if( RC_BAD( rc = gv_pConnTbl->getConnection( 
		m_szDbPath, &m_pConn)))
	{
		goto Exit;
	}

	if( m_pConn->getTransType() != FLM_UPDATE_TRANS)
	{
		if( RC_BAD( rc = FlmDbTransBegin( m_pConn->getDb(), 
			FLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
		{
			goto Exit;
		}

		bStartedTrans = TRUE;
	}

	if( RC_BAD( rc = nameTable.setupFromDb( m_pConn->getDb())))
	{
		goto Exit;
	}

	f_pathReduce( pszOldName, NULL, szTmpBuf);
	sprintf( szOldPrefix, ":%s:", szTmpBuf);
	uiOldPrefixLen = strlen( szOldPrefix);

	f_pathReduce( pszNewName, NULL, szTmpBuf);
	sprintf( szNewPrefix, ":%s:", szTmpBuf);
	uiNewPrefixLen = strlen( szNewPrefix);

	// From this point forward, any errors must cause the
	// transaction to abort

	bMustAbortOnError = TRUE;

	// Rename indexes, fields, and the table container

	uiLoop = 0;
	for( ;;)
	{
		if( !nameTable.getNextTagNameOrder( &uiLoop, NULL, szTmpBuf, 
			sizeof( szTmpBuf), &uiDictNum, NULL, NULL))
		{
			break;
		}

		if( (iCmp = strnicmp( szTmpBuf, szOldPrefix, uiOldPrefixLen)) > 0)
		{
			break;
		}

		if( iCmp == 0)
		{
			if( RC_BAD( rc = FlmRecordRetrieve( m_pConn->getDb(), 
				FLM_DICT_CONTAINER, uiDictNum, FO_EXACT, &pDictRec, NULL)))
			{
				goto Exit;
			}

			if( pDictRec->isReadOnly())
			{
				FlmRecord *		pTmpRec;

				if( (pTmpRec = pDictRec->copy()) == NULL)
				{
					rc = FERR_MEM;
					goto Exit;
				}

				pDictRec->Release();
				pDictRec = pTmpRec;
			}

			sprintf( szNewName, "%s%s", szNewPrefix, 
						strchr( &szTmpBuf[ 1], ':') + 1);

			if( RC_BAD( rc = pDictRec->setNative( 
				pDictRec->root(), szNewName)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = FlmRecordModify( m_pConn->getDb(), 
				FLM_DICT_CONTAINER, uiDictNum, pDictRec, 0)))
			{
				goto Exit;
			}
		}
	}

	if( bStartedTrans)
	{
		bStartedTrans = FALSE;

		if( RC_BAD( rc = FlmDbTransCommit( m_pConn->getDb())))
		{
			goto Exit;
		}
	}

Exit:

	if( pDictRec)
	{
		pDictRec->Release();
	}

	if( RC_BAD( rc) && bMustAbortOnError)
	{
		m_pConn->setAbortFlag();
	}

	DBUG_RETURN( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
int ha_flaim::delete_table(
	const char *		pszTablePath)
{
	RCODE					rc = FERR_OK;
	FLMUINT				uiDictNum;
	F_NameTable			nameTable;
	FlmRecord *			pDictRec = NULL;
	char					szPrefix[ F_PATH_MAX_SIZE];
	char					szTmpBuf[ F_PATH_MAX_SIZE];
	int					iCmp;
	void *				pvField;
	FLMUINT				uiPrefixLen;
	FLMUINT				uiLoop;
	FLMUINT				uiDefType;
	FLMBOOL				bSweep = FALSE;
	FLMBOOL				bStartedTrans = FALSE;
	FLMBOOL				bMustAbortOnError = FALSE;

	DBUG_ENTER( "ha_flaim::delete_table");

	buildDbPathFromTablePath( pszTablePath, m_szDbPath);

	if( RC_BAD( rc = gv_pConnTbl->getConnection( 
		m_szDbPath, &m_pConn)))
	{
		goto Exit;
	}

	if( m_pConn->getTransType() != FLM_UPDATE_TRANS)
	{
		if( RC_BAD( rc = FlmDbTransBegin( m_pConn->getDb(), 
			FLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
		{
			goto Exit;
		}

		bStartedTrans = TRUE;
	}

	if( RC_BAD( rc = nameTable.setupFromDb( m_pConn->getDb())))
	{
		goto Exit;
	}

	f_pathReduce( pszTablePath, NULL, szTmpBuf);
	sprintf( szPrefix, ":%s:", szTmpBuf);
	uiPrefixLen = strlen( szPrefix);

	// From this point forward, any errors must cause the
	// transaction to abort

	bMustAbortOnError = TRUE;

	// Drop indexes and mark fields for deletion

	uiLoop = 0;
	for( ;;)
	{
		if( !nameTable.getNextTagNameOrder( &uiLoop, NULL, szTmpBuf, 
			sizeof( szTmpBuf), &uiDictNum, &uiDefType, NULL))
		{
			break;
		}

		if( (iCmp = strnicmp( szTmpBuf, szPrefix, uiPrefixLen)) > 0)
		{
			break;
		}

		if( iCmp == 0)
		{
			if( uiDefType == FLM_FIELD_TAG)
			{
				if( RC_BAD( rc = FlmRecordRetrieve( m_pConn->getDb(), 
					FLM_DICT_CONTAINER, uiDictNum, FO_EXACT, &pDictRec, NULL)))
				{
					goto Exit;
				}

				if( pDictRec->isReadOnly())
				{
					FlmRecord *		pTmpRec;

					if( (pTmpRec = pDictRec->copy()) == NULL)
					{
						rc = FERR_MEM;
						goto Exit;
					}

					pDictRec->Release();
					pDictRec = pTmpRec;
				}

				if( RC_BAD( rc = pDictRec->insertLast( 1, FLM_STATE_TAG, 
					FLM_TEXT_TYPE, &pvField)))
				{
					goto Exit;
				}

				if( RC_BAD( rc = pDictRec->setNative( pvField, "purge")))
				{
					goto Exit;
				}

				if( RC_BAD( rc = FlmRecordModify( m_pConn->getDb(), 
					FLM_DICT_CONTAINER, uiDictNum, pDictRec, 0)))
				{
					goto Exit;
				}

				bSweep = TRUE;
			}
			else if( uiDefType == FLM_INDEX_TAG)
			{
				if( RC_BAD( rc = FlmRecordDelete( m_pConn->getDb(), 
					FLM_DICT_CONTAINER, uiDictNum, 0)))
				{
					goto Exit;
				}
			}
		}
	}

	// Drop the table container

	if( !nameTable.getFromTagTypeAndName( NULL, szPrefix, 
		FLM_CONTAINER_TAG, &uiDictNum))
	{
		rc = FERR_DATA_ERROR;
		goto Exit;
	}

	if( RC_BAD( rc = FlmRecordDelete( m_pConn->getDb(),
		FLM_DICT_CONTAINER, uiDictNum, 0)))
	{
		goto Exit;
	}

	// Delete the table info record

	if( RC_BAD( rc = FlmRecordDelete( m_pConn->getDb(),
		FLM_DATA_CONTAINER, uiDictNum, 0)))
	{
		if( rc != FERR_NOT_FOUND)
		{
			goto Exit;
		}

		rc = FERR_OK;
	}

	if( bStartedTrans)
	{
		bStartedTrans = FALSE;

		if( RC_BAD( rc = FlmDbTransCommit( m_pConn->getDb())))
		{
			goto Exit;
		}
	}

	if( bSweep)
	{
		if( RC_BAD( rc = FlmDbSweep( m_pConn->getDb(), 
			SWEEP_PURGED_FLDS, 0, NULL, NULL)))
		{
			goto Exit;
		}
	}

Exit:

	if( pDictRec)
	{
		pDictRec->Release();
	}

	if( RC_BAD( rc) && bMustAbortOnError)
	{
		m_pConn->setAbortFlag();
	}

	DBUG_RETURN( rc);
}

/****************************************************************************
Desc:	Given a starting key and an ending key, estimate the number of rows
		that will exist between the two.  The end key may be empty.
****************************************************************************/
ha_rows ha_flaim::records_in_range(
	uint				uiIndex, 
	key_range *		pMinKey,
	key_range *		pMaxKey)
{
	// VISIT: Implement
	DBUG_ENTER( "ha_flaim::records_in_range");
	DBUG_RETURN( 2);
}

/****************************************************************************
Desc:	Called to create a database and/or table.
****************************************************************************/
int ha_flaim::create(
	const char *		pszFormFilePath,
	TABLE *				pTable,
	HA_CREATE_INFO *	pCreateInfo)
{
	RCODE					rc = FERR_OK;
	FlmRecord *			pRec = NULL;
	void *				pvField;
	char					szTmpBuf[ 512];
	FLMUINT				uiTableDefId;
	FLMUINT				uiLoop;
	FLMUINT				uiLastDrnUsed = 0;
	FLMBOOL				bStartedTrans = FALSE;
	FLMBOOL				bMustAbortOnError = FALSE;
	FLMBOOL				bCreatedDatabase = FALSE;

	DBUG_ENTER( "ha_flaim::create");

Retry:

	f_pathReduce( pszFormFilePath, m_szDbPath, NULL);
	f_pathAppend( m_szDbPath, FLAIM_DB_NAME);

	assert( m_pConn == NULL);

	if( RC_BAD( rc = gv_pConnTbl->getConnection( m_szDbPath, &m_pConn)))
	{
		if( rc == FERR_IO_PATH_NOT_FOUND)
		{
			if( RC_BAD( rc = FlmDbCreate( m_szDbPath, NULL, NULL, NULL,
				gv_pszBaseDict, NULL, NULL)))
			{
				goto Exit;
			}

			bCreatedDatabase = TRUE;
		}

		goto Retry;
	}

	if( pTable && pTable->s->table_name)
	{
		if( m_pConn->getTransType() != FLM_UPDATE_TRANS)
		{
			if( RC_BAD( rc = FlmDbTransBegin( m_pConn->getDb(), 
				FLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
			{
				goto Exit;
			}

			bStartedTrans = TRUE;
		}

		// Create the table container

		if( (pRec = new FlmRecord) == NULL)
		{
			rc = FERR_MEM;
			goto Exit;
		}

		if( RC_BAD( rc = pRec->insertLast( 0, FLM_CONTAINER_TAG, 
			FLM_TEXT_TYPE, &pvField)))
		{
			goto Exit;
		}

		sprintf( szTmpBuf, ":%s:", pTable->s->table_name);

		if( RC_BAD( rc = pRec->setNative( pvField, szTmpBuf)))
		{
			goto Exit;
		}

		bMustAbortOnError = TRUE;

		if( RC_BAD( rc = FlmFindUnusedDictDrn( m_pConn->getDb(),
			uiLastDrnUsed, FLM_RESERVED_TAG_NUMS - 1, &uiLastDrnUsed)))
		{
			goto Exit;
		}

		uiTableDefId = uiLastDrnUsed;

		if( RC_BAD( rc = FlmRecordAdd( m_pConn->getDb(), 
			FLM_DICT_CONTAINER, &uiTableDefId, pRec, 0)))
		{
			goto Exit;
		}

		pRec->Release();

		// Create the table column fields
		
		for( Field ** ppField = table->field; *ppField; ppField++)
		{
			if( (pRec = new FlmRecord) == NULL)
			{
				rc = FERR_MEM;
				goto Exit;
			}
	
			if( RC_BAD( rc = pRec->insertLast( 0, FLM_FIELD_TAG, 
				FLM_TEXT_TYPE, &pvField)))
			{
				goto Exit;
			}
	
			sprintf( szTmpBuf, ":%s:col:%s:", pTable->s->table_name, 
				(*ppField)->field_name);

			if( RC_BAD( rc = pRec->setNative( pvField, szTmpBuf)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pRec->insertLast( 1, FLM_TYPE_TAG, 
				FLM_TEXT_TYPE, &pvField)))
			{
				goto Exit;
			}

			switch( (*ppField)->type())
			{
				case MYSQL_TYPE_VARCHAR:
				case MYSQL_TYPE_VAR_STRING:
				case MYSQL_TYPE_STRING:
				case MYSQL_TYPE_DECIMAL:
				{
					if( RC_BAD( rc = pRec->setNative( pvField, "text")))
					{
						goto Exit;
					}

					break;
				}

				case MYSQL_TYPE_TINY:
				case MYSQL_TYPE_SHORT:
				case MYSQL_TYPE_LONG:
				case MYSQL_TYPE_INT24:
				case MYSQL_TYPE_DATE:
				case MYSQL_TYPE_TIME:
				case MYSQL_TYPE_TIMESTAMP:
				case MYSQL_TYPE_YEAR:
				case MYSQL_TYPE_NEWDATE:
				{
					if( RC_BAD( rc = pRec->setNative( pvField, "number")))
					{
						goto Exit;
					}

					break;
				}

				case MYSQL_TYPE_FLOAT:
				case MYSQL_TYPE_DOUBLE:
				case MYSQL_TYPE_NULL:
				case MYSQL_TYPE_LONGLONG:
				case MYSQL_TYPE_DATETIME:
				case MYSQL_TYPE_BIT:
				case MYSQL_TYPE_NEWDECIMAL:
				case MYSQL_TYPE_ENUM:
				case MYSQL_TYPE_SET:
				case MYSQL_TYPE_TINY_BLOB:
				case MYSQL_TYPE_MEDIUM_BLOB:
				case MYSQL_TYPE_LONG_BLOB:
				case MYSQL_TYPE_BLOB:
				case MYSQL_TYPE_GEOMETRY:
				{
					if( RC_BAD( rc = pRec->setNative( pvField, "binary")))
					{
						goto Exit;
					}

					break;
				}

				default:
				{
					assert( 0);
					rc = FERR_NOT_IMPLEMENTED;
					goto Exit;
				}
			}

			if( RC_BAD( rc = FlmFindUnusedDictDrn( m_pConn->getDb(),
				uiLastDrnUsed, FLM_RESERVED_TAG_NUMS - 1, &uiLastDrnUsed)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = FlmRecordAdd( m_pConn->getDb(), 
				FLM_DICT_CONTAINER, &uiLastDrnUsed, pRec, 0)))
			{
				if( rc != FERR_DUPLICATE_DICT_NAME)
				{
					goto Exit;
				}

				rc = FERR_OK;
			}
		}

		// Create the table key fields
		
		for( uiLoop = 0; uiLoop < pTable->s->keys; uiLoop++)
		{
			KEY *					pKey = table->key_info + uiLoop;
			FLMUINT				uiKeyField;

			if( (pRec = new FlmRecord) == NULL)
			{
				rc = FERR_MEM;
				goto Exit;
			}
	
			if( RC_BAD( rc = pRec->insertLast( 0, FLM_FIELD_TAG, 
				FLM_TEXT_TYPE, &pvField)))
			{
				goto Exit;
			}

			sprintf( szTmpBuf, ":%s:key:%u:", pTable->s->table_name, uiLoop);
	
			if( RC_BAD( rc = pRec->setNative( pvField, szTmpBuf)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pRec->insertLast( 1, FLM_TYPE_TAG, 
				FLM_TEXT_TYPE, &pvField)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pRec->setNative( pvField, "binary")))
			{
				goto Exit;
			}

			if( RC_BAD( rc = FlmFindUnusedDictDrn( m_pConn->getDb(),
				uiLastDrnUsed, FLM_RESERVED_TAG_NUMS - 1, &uiLastDrnUsed)))
			{
				goto Exit;
			}

			uiKeyField = uiLastDrnUsed;

			if( RC_BAD( rc = FlmRecordAdd( m_pConn->getDb(), 
				FLM_DICT_CONTAINER, &uiKeyField, pRec, 0)))
			{
				if( rc != FERR_DUPLICATE_DICT_NAME)
				{
					goto Exit;
				}

				rc = FERR_OK;
			}

			pRec->Release();
			pRec = NULL;

			// Add the index

			if( (pRec = new FlmRecord) == NULL)
			{
				rc = FERR_MEM;
				goto Exit;
			}
	
			if( RC_BAD( rc = pRec->insertLast( 0, FLM_INDEX_TAG, 
				FLM_TEXT_TYPE, &pvField)))
			{
				goto Exit;
			}

			sprintf( szTmpBuf, ":%s:index:%u:", pTable->s->table_name, uiLoop);
	
			if( RC_BAD( rc = pRec->setNative( pvField, szTmpBuf)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pRec->insertLast( 1, FLM_CONTAINER_TAG, 
				FLM_TEXT_TYPE, &pvField)))
			{
				goto Exit;
			}

			sprintf( szTmpBuf, "%u", uiTableDefId);

			if( RC_BAD( rc = pRec->setNative( pvField, szTmpBuf)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pRec->insertLast( 1, FLM_KEY_TAG, 
				FLM_TEXT_TYPE, &pvField)))
			{
				goto Exit;
			}

			if( (pKey->flags & (HA_NOSAME | HA_NULL_PART_KEY)) == HA_NOSAME)
			{
				if( RC_BAD( rc = pRec->insertLast( 2, FLM_UNIQUE_TAG, 
					FLM_TEXT_TYPE, &pvField)))
				{
					goto Exit;
				}
			}

			if( RC_BAD( rc = pRec->insertLast( 2, FLM_FIELD_TAG, 
				FLM_TEXT_TYPE, &pvField)))
			{
				goto Exit;
			}

			sprintf( szTmpBuf, "%u", uiKeyField);

			if( RC_BAD( rc = pRec->setNative( pvField, szTmpBuf)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = FlmFindUnusedDictDrn( m_pConn->getDb(),
				uiLastDrnUsed, FLM_RESERVED_TAG_NUMS - 1, &uiLastDrnUsed)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = FlmRecordAdd( m_pConn->getDb(), 
				FLM_DICT_CONTAINER, &uiLastDrnUsed, pRec, 0)))
			{
				goto Exit;
			}

			pRec->Release();
			pRec = NULL;
		}

		if( bStartedTrans)
		{
			bStartedTrans = FALSE;

			if( RC_BAD( rc = FlmDbTransCommit( m_pConn->getDb())))
			{
				goto Exit;
			}
		}
	}

Exit:

	if( pRec)
	{
		pRec->Release();
	}

	if( bStartedTrans)
	{
		FlmDbTransAbort( m_pConn->getDb());
		bMustAbortOnError = FALSE;
	}

	if( RC_BAD( rc))
	{
		if( bMustAbortOnError)
		{
			assert( !bCreatedDatabase);
			m_pConn->setAbortFlag();
		}
		
		if( bCreatedDatabase)
		{
			FlmDbRemove( m_szDbPath, NULL, NULL, TRUE);
		}
	}

	DBUG_RETURN( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE ha_flaim::importRowFromRec(
	FlmRecord *		pRec,
	FLMBYTE *		pucData)
{
	RCODE				rc = FERR_OK;
	Field **			ppMyField;
	void *			pvField;
	FLMUINT			uiLength;
	FIELD_INFO *	pColumnFields = m_pShare->pColumnFields;
	FLMBYTE *		pucAllocBuf = NULL;

	pvField = pRec->firstChild( pRec->root());

	// Import NULL bitmap

	if( (uiLength = table->s->null_bytes) != 0)
	{
		if( (pvField = pRec->find( pvField, 
			FLM_ROW_NULL_BITS_FIELD_ID)) == NULL)
		{
			rc = FERR_DATA_ERROR;
			goto Exit;
		}

		if( RC_BAD( rc = pRec->getBinary( pvField, pucData, &uiLength)))
		{
			goto Exit;
		}
	}

	// Import each column's data

	for( ppMyField = table->field; *ppMyField; ppMyField++)
	{
		if( !isFieldNull( table, *ppMyField, (const char *)pucData))
		{
			FIELD_INFO *	pFieldInfo = &pColumnFields[ (*ppMyField)->field_index];
			char *			pucFieldData = (char *)(pucData + (*ppMyField)->offset());

			if( (pvField = pRec->find( pvField, pFieldInfo->uiDictNum)) == NULL)
			{
				rc = FERR_DATA_ERROR;
				goto Exit;
			}

			switch( pFieldInfo->uiDataType)
			{
				case FLM_TEXT_TYPE:
				{
					FLMUNICODE		uzTmpBuf[ 256];
					FLMUNICODE *	puzStr = uzTmpBuf;
					FLMUINT			uiStrLen;

					uiStrLen = sizeof( uzTmpBuf);
					if( RC_BAD( rc = pRec->getUnicode( pvField, uzTmpBuf, &uiStrLen)))
					{
						if( rc != FERR_CONV_DEST_OVERFLOW)
						{
							goto Exit;
						}

						if( RC_BAD( rc = pRec->getUnicode( pvField, NULL, &uiStrLen)))
						{
							goto Exit;
						}

						if( !my_multi_malloc( MYF( MY_WME), &pucAllocBuf, uiStrLen + 2, NullS))
						{
							rc = FERR_MEM;
							goto Exit;
						}

						uiStrLen += 2;

						if( RC_BAD( rc = pRec->getUnicode( pvField, 
							(FLMUNICODE *)pucAllocBuf, &uiStrLen)))
						{
							goto Exit;
						}

						puzStr = (FLMUNICODE *)pucAllocBuf;
					}

					#ifndef WORDS_BIGENDIAN
					{
						FLMUNICODE *	puzTmp = puzStr;

						while( *puzTmp)
						{
							*puzTmp++ = (*puzTmp >> 8) | (*puzTmp << 8);
						}
					}
					#endif

					String	tmpStr( (const char *)puzStr, uiStrLen, &my_charset_ucs2_bin);

					if( convertString( &tmpStr, &my_charset_ucs2_bin, (*ppMyField)->charset()))
					{
						rc = FERR_MEM;
						goto Exit;
					}

					uiStrLen = tmpStr.length();

					switch( (*ppMyField)->type())
					{
						case MYSQL_TYPE_STRING:
						case MYSQL_TYPE_VAR_STRING:
						case MYSQL_TYPE_DECIMAL:
						{
							memcpy( pucFieldData, tmpStr.ptr(), uiStrLen);
							break;
						}

						case MYSQL_TYPE_VARCHAR:
						{
							if( ((Field_varstring *)(*ppMyField))->length_bytes == 1)
							{
								*pucFieldData++ = (uchar)uiStrLen;
							}
							else
							{
								int2store( pucFieldData, tmpStr.length());
								pucFieldData += 2;
							}

							memcpy( pucFieldData, tmpStr.ptr(), uiStrLen);
							break;
						}

						default:
						{
							assert( 0);
							rc = FERR_NOT_IMPLEMENTED;
							goto Exit;
						}
					}

					break;
				}

				case FLM_NUMBER_TYPE:
				{
					longlong		i64Val;

					if( (*ppMyField)->flags & UNSIGNED_FLAG)
					{
						FLMUINT		uiVal;

						if( RC_BAD( rc = pRec->getUINT( pvField, &uiVal)))
						{
							goto Exit;
						}

						i64Val = (longlong)uiVal;
					}
					else
					{
						FLMINT		iVal;

						if( RC_BAD( rc = pRec->getINT( pvField, &iVal)))
						{
							goto Exit;
						}

						i64Val = (longlong)iVal;
					}

					switch( (*ppMyField)->type())
					{
						case MYSQL_TYPE_TINY:
						case MYSQL_TYPE_YEAR:
						{
							*pucFieldData = (char)i64Val;
							break;
						}

						case MYSQL_TYPE_SHORT:
						{
							#ifdef WORDS_BIGENDIAN
							if( table->s->db_low_byte_first)
							{
								int2store( pucFieldData, i64Val);
							}
							else
							#endif
							{
								shortstore( pucFieldData, i64Val);
							}

							break;
						}

						case MYSQL_TYPE_INT24:
						case MYSQL_TYPE_TIME:
						case MYSQL_TYPE_NEWDATE:
						{
							int3store( pucFieldData, i64Val);
							break;
						}

						case MYSQL_TYPE_LONG:
						case MYSQL_TYPE_DATE:
						case MYSQL_TYPE_TIMESTAMP:
						{
							#ifdef WORDS_BIGENDIAN
							if( table->s->db_low_byte_first)
							{
								int4store( pucFieldData, i64Val);
							}
							else
							#endif
							{
								longstore( pucFieldData, i64Val);
							}

							break;
						}

						default:
						{
							assert( 0);
							rc = FERR_NOT_IMPLEMENTED;
							goto Exit;
						}
					}

					break;
				}

				case FLM_BINARY_TYPE:
				{
					(*ppMyField)->unpack( pucFieldData,
												(const char *)(pRec->getDataPtr( pvField)));
					break;
				}

				default:
				{
					assert( 0);
					rc = FERR_NOT_IMPLEMENTED;
					goto Exit;
				}
			}
		}
	}

Exit:

	if( pucAllocBuf)
	{
		my_free( (gptr)pucAllocBuf, MYF(0));
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE ha_flaim::exportRowToRec(
	FLMBYTE *		pucData,
	FlmRecord **	ppRec)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiLoop;
	FlmRecord *		pRec = NULL;
	Field **			ppMyField;
	void *			pvField;
	FIELD_INFO *	pColumnFields = m_pShare->pColumnFields;
	FIELD_INFO *	pKeyFields = m_pShare->pKeyFields;

	assert( *ppRec == NULL);

	if( (pRec = new FlmRecord) == NULL)
	{
		rc = FERR_MEM;
		goto Exit;
	}

	if( RC_BAD( rc = pRec->insertLast( 0, 
		FLM_ROW_CONTEXT_FIELD_ID, FLM_CONTEXT_TYPE, &pvField)))
	{
		goto Exit;
	}

	// Store NULL bitmap

	if( table->s->null_bytes)
	{
		if( RC_BAD( rc = pRec->insertLast( 1, 
			FLM_ROW_NULL_BITS_FIELD_ID, FLM_BINARY_TYPE, &pvField)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pRec->setBinary( pvField, 
			pucData, table->s->null_bytes)))
		{
			goto Exit;
		}
	}

	// Store each column's data

	for( ppMyField = table->field; *ppMyField; ppMyField++)
	{
		if( !isFieldNull( table, *ppMyField, (const char *)pucData))
		{
			FIELD_INFO *	pFieldInfo = &pColumnFields[ (*ppMyField)->field_index];

			if( RC_BAD( rc = pRec->insertLast( 1, 
				pFieldInfo->uiDictNum, pFieldInfo->uiDataType, &pvField)))
			{
				goto Exit;
			}

			switch( pFieldInfo->uiDataType)
			{
				case FLM_TEXT_TYPE:
				{
					String			tmpStr;

					(*ppMyField)->val_str( &tmpStr);

					if( convertString( &tmpStr, (*ppMyField)->charset(), 
						&my_charset_ucs2_bin))
					{
						rc = FERR_MEM;
						goto Exit;
					}

					if( tmpStr.append( "\0\0", 2, &my_charset_ucs2_bin))
					{
						rc = FERR_MEM;
						goto Exit;
					}

					// FLAIM expects the Unicode string to have the
					// same endian order as the host platform.  MySQL
					// always represents Unicode as big endian.  Thus,
					// we need to do some byte swapping if we are on
					// a little-endian platform.

					#ifndef WORDS_BIGENDIAN
					{
						FLMUNICODE *	puzTmp = (FLMUNICODE *)tmpStr.ptr();

						while( *puzTmp)
						{
							*puzTmp++ = (*puzTmp >> 8) | (*puzTmp << 8);
						}
					}
					#endif

					if( RC_BAD( rc = pRec->setUnicode( pvField, 
						(FLMUNICODE *)tmpStr.ptr())))
					{
						goto Exit;
					}

					break;
				}

				case FLM_NUMBER_TYPE:
				{
					longlong			iNum = (*ppMyField)->val_int();

					if( (*ppMyField)->flags & UNSIGNED_FLAG)
					{
						if( RC_BAD( rc = pRec->setUINT( pvField, (FLMUINT)iNum)))
						{
							goto Exit;
						}
					}
					else
					{
						if( RC_BAD( rc = pRec->setINT( pvField, (FLMINT)iNum)))
						{
							goto Exit;
						}
					}

					break;
				}

				case FLM_BINARY_TYPE:
				{
					if( RC_BAD( rc = pRec->setBinary( pvField, 
						&pucData[ (*ppMyField)->offset()], 
						(*ppMyField)->packed_col_length( 
							(const char *)&pucData[ (*ppMyField)->offset()],
						(*ppMyField)->pack_length()))))
					{
						goto Exit;
					}

					break;
				}

				default:
				{
					assert( 0);
					rc = FERR_NOT_IMPLEMENTED;
					goto Exit;
				}
			}
		}
	}

	// Store the row's keys

	for( uiLoop = 0; uiLoop < table->s->keys; uiLoop++)
	{
		if( RC_BAD( rc = exportKeyToRec( uiLoop, pucData, pRec)))
		{
			goto Exit;
		}
	}

	*ppRec = pRec;
	pRec = NULL;

Exit:

	if( pRec)
	{
		pRec->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE ha_flaim::exportKeyToTree(
	FLMUINT			uiIndexOffset,
	const byte *	pucKey,
	FLMUINT			uiKeyLen,
	FlmRecord **	ppKeyTree)
{
	RCODE					rc = FERR_OK;
	char *				pucBuf = m_pucKeyBuf;
	char *				pucBufEnd = &m_pucKeyBuf[ FLM_MAX_KEY_LENGTH];
	FlmRecord *			pKeyTree = NULL;
	void *				pvField;
	KEY *					pKey = table->key_info + uiIndexOffset;
	KEY_PART_INFO *	pCurKeyPart = pKey->key_part;
	KEY_PART_INFO *	pEndKeyPart = pCurKeyPart + pKey->key_parts;

	DBUG_ENTER( "ha_flaim::exportKeyToTree");

	assert( *ppKeyTree == NULL);

	if( (pKeyTree = new FlmRecord) == NULL)
	{
		rc = FERR_MEM;
		goto Exit;
	}

	if( RC_BAD( rc = pKeyTree->insertLast( 0, 
		FLM_KEY_TAG, FLM_CONTEXT_TYPE, &pvField)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pKeyTree->insertLast( 1, 
		m_pShare->pKeyFields[ uiIndexOffset].uiDictNum,
		FLM_BINARY_TYPE, &pvField)))
	{
		goto Exit;
	}

	for( ; pCurKeyPart != pEndKeyPart && uiKeyLen > 0; pCurKeyPart++)
	{
		FLMUINT		uiOffset = 0;

		if( pCurKeyPart->null_bit)
		{
			*pucBuf++ = *pucKey ? 1 : 0;
			uiKeyLen -= pCurKeyPart->store_length;
			pucKey += pCurKeyPart->store_length;
			uiOffset = 1;
		}
		
		pucBuf = pCurKeyPart->field->pack_key_from_key_image( pucBuf, 
					(const char *)pucKey + uiOffset, pCurKeyPart->length);
		pucKey += pCurKeyPart->store_length;
		uiKeyLen -= pCurKeyPart->store_length;

		assert( pucBuf <= pucBufEnd);
	}

	if( RC_BAD( rc = pKeyTree->setBinary( pvField, m_pucKeyBuf, 
		pucBuf - m_pucKeyBuf)))
	{
		goto Exit;
	}

	*ppKeyTree = pKeyTree;
	pKeyTree = NULL;

Exit:

	if( pKeyTree)
	{
		pKeyTree->Release();
	}

	DBUG_RETURN( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FlmConnectionTable::getConnection(
	const char *		pszDbPath,
	FlmConnection **	ppConnection)
{
	RCODE					rc = FERR_OK;
	HFDB					hDb = HFDB_NULL;
	FlmConnection *	pConn = NULL;
	FLMUINT				uiThreadId = my_thread_id();
	FLMBOOL				bMutexLocked = FALSE;

	if( *ppConnection)
	{
		if( (*ppConnection)->getThreadId() == uiThreadId)
		{
			goto Exit;
		}

		(*ppConnection)->Release();
		*ppConnection = NULL;
	}

	// Search the connection list for an existing connection

	pthread_mutex_lock( &m_hMutex);
	bMutexLocked = TRUE;

	pConn = m_pConnList;
	while( pConn)
	{
		if( pConn->getThreadId() == uiThreadId)
		{
			break;
		}

		pConn = pConn->getNext();
	}

	if( pConn)
	{
		// VISIT: Add debug code to make sure this thread
		// is still accessing the same database as the last
		// time this connection was used.

		pConn->AddRef();
		*ppConnection = pConn;
		pConn = NULL;
		goto Exit;
	}

	pthread_mutex_unlock( &m_hMutex);
	bMutexLocked = FALSE;

	if( !pszDbPath)
	{
		rc = FERR_ILLEGAL_OP;
		goto Exit;
	}

	if( RC_BAD( rc = FlmDbOpen( pszDbPath, NULL, NULL, 0, NULL, &hDb)))
	{
		goto Exit;
	}

	if( (pConn = new FlmConnection( gv_pConnTbl)) == NULL)
	{
		rc = FERR_MEM;
		goto Exit;
	}

	pConn->setThreadId( uiThreadId);
	pConn->setDb( hDb);
	hDb = HFDB_NULL;

	pthread_mutex_lock( &m_hMutex);
	bMutexLocked = TRUE;

	if( (pConn->m_pNext = m_pConnList) != NULL)
	{
		pConn->m_pNext->m_pPrev = pConn;
	}

	m_pConnList = pConn;
	m_pConnList->AddRef();
	*ppConnection = pConn;
	pConn = NULL;

Exit:

	if( pConn)
	{
		pConn->Release();
	}

	if( bMutexLocked)
	{
		pthread_mutex_unlock( &m_hMutex);
	}

	if( hDb != HFDB_NULL)
	{
		FlmDbClose( &hDb);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FlmConnectionTable::setup( void)
{
	RCODE		rc = FERR_OK;

	if( pthread_mutex_init( &m_hMutex, MY_MUTEX_INIT_FAST))
	{
		rc = FERR_MEM;
		goto Exit;
	}

	m_bFreeMutex = TRUE;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT FlmConnection::Release(
	FLMBOOL		bMutexLocked)
{
	FLMUINT	uiRefCount = --m_ui32RefCnt;

	if( uiRefCount == 1)
	{
		m_pConnTbl->closeConnection( this, bMutexLocked);
		delete this;
	}

	return( uiRefCount);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FlmConnectionTable::closeConnection(
	FlmConnection *	pConn,
	FLMBOOL				bMutexLocked)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bLastConnection = FALSE;

	if( !bMutexLocked)
	{
		lockMutex();
	}

	if( pConn->m_pPrev)
	{
		pConn->m_pPrev->m_pNext = pConn->m_pNext;
	}
	else
	{
		if( (m_pConnList = pConn->m_pNext) == NULL)
		{
			bLastConnection = TRUE;
		}
	}

	if( pConn->m_pNext)
	{
		pConn->m_pNext->m_pPrev = pConn->m_pPrev;
	}

	if( !bMutexLocked)
	{
		unlockMutex();
	}

	if( pConn->m_hDb != HFDB_NULL)
	{
		FLMBYTE		szDbPath[ F_PATH_MAX_SIZE];

		szDbPath[ 0] = 0;

		if( bLastConnection)
		{
			if( RC_BAD( FlmDbGetConfig( pConn->m_hDb, FDB_GET_PATH, szDbPath)))
			{
				szDbPath[ 0] = 0;
			}
		}

		FlmDbClose( &pConn->m_hDb);

		if( szDbPath[ 0])
		{
			FlmConfig( FLM_CLOSE_FILE, szDbPath, NULL);
		}
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FlmConnection::FlmConnection( FlmConnectionTable * pConnTbl)
{
	m_uiThreadId = 0;
	m_hDb = HFDB_NULL;
	m_pPrev = NULL;
	m_pNext = NULL;
	m_bAbortFlag = FALSE;

	m_uiLockCount = 0;
	m_uiTransTypeNeeded = FLM_NO_TRANS;

	m_pConnTbl = pConnTbl;
	m_pConnTbl->AddRef();
}

/****************************************************************************
Desc:
****************************************************************************/
FlmConnection::~FlmConnection()
{
	assert( !m_pPrev);
	assert( !m_pNext);

	if( m_hDb != HFDB_NULL)
	{
		FlmDbClose( &m_hDb);
	}

	if( m_pConnTbl)
	{
		m_pConnTbl->Release();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
static int flaim_commit(
	THD *			pThread, 
	bool			bAll)
{
	RCODE					rc = FERR_OK;
	FlmConnection *	pConn = NULL;

	DBUG_ENTER( "flaim_commit");

	if( bAll)
	{
		if( RC_BAD( rc = gv_pConnTbl->getConnection( 
			NULL, &pConn)))
		{
			goto Exit;
		}

		assert( pConn->getTransType() == FLM_UPDATE_TRANS);

		if( pConn->getTransType() != FLM_NO_TRANS)
		{
			assert( pConn->getLockCount() == 1);

			pConn->setLockCount( 0);
			pConn->setTransTypeNeeded( FLM_NO_TRANS);

			if( RC_BAD( rc = FlmDbTransCommit( pConn->getDb())))
			{
				goto Exit;
			}
		}
	}

Exit:

	if( pConn)
	{
		pConn->Release();
	}

	DBUG_RETURN( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
static int flaim_rollback(
	THD *			pThread, 
	bool			bAll)
{
	RCODE					rc = FERR_OK;
	FlmConnection *	pConn = NULL;

	DBUG_ENTER( "flaim_rollback");

	if( bAll)
	{
		if( RC_BAD( rc = gv_pConnTbl->getConnection( 
			NULL, &pConn)))
		{
			goto Exit;
		}

		assert( pConn->getTransType() == FLM_UPDATE_TRANS);

		if( pConn->getTransType() != FLM_NO_TRANS)
		{
			assert( pConn->getLockCount() == 1);

			pConn->setLockCount( 0);
			pConn->setTransTypeNeeded( FLM_NO_TRANS);

			if( RC_BAD( rc = FlmDbTransAbort( pConn->getDb())))
			{
				goto Exit;
			}
		}
	}

Exit:

	if( pConn)
	{
		pConn->Release();
	}

	DBUG_RETURN( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FlmConnection::storeRowCount(
	FLMUINT					uiTableId,
	FLMUINT					uiRowCount)
{
	RCODE				rc = FERR_OK;
	FlmRecord *		pRec = NULL;
	void *			pvField;
	FLMBOOL			bAddNewRec = FALSE;

	if( RC_BAD( rc = FlmRecordRetrieve( m_hDb, FLM_DATA_CONTAINER, 
		uiTableId, FO_EXACT, &pRec, NULL)))
	{
		if( rc != FERR_NOT_FOUND)
		{
			goto Exit;
		}

		if( (pRec = new FlmRecord) == NULL)
		{
			rc = FERR_MEM;
			goto Exit;
		}

		if( RC_BAD( rc = pRec->insertLast( 0, FLM_TABLE_INFO_FIELD_ID,
			FLM_CONTEXT_TYPE, &pvField)))
		{
			goto Exit;
		}

		bAddNewRec = TRUE;
	}

	if( pRec->isReadOnly())
	{
		FlmRecord *		pTmpRec;

		if( (pTmpRec = pRec->copy()) == NULL)
		{
			rc = FERR_MEM;
			goto Exit;
		}

		pRec->Release();
		pRec = pTmpRec;
	}

	if( (pvField = pRec->find( pRec->root(), FLM_ROW_COUNT_FIELD_ID)) == NULL)
	{
		if( RC_BAD( rc = pRec->insertLast( 1, FLM_ROW_COUNT_FIELD_ID, 
			FLM_NUMBER_TYPE, &pvField)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = pRec->setUINT( pvField, uiRowCount)))
	{
		goto Exit;
	}

	if( !bAddNewRec)
	{
		if( RC_BAD( rc = FlmRecordModify( m_hDb,
			FLM_DATA_CONTAINER, uiTableId, pRec, 0)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = FlmRecordAdd( m_hDb,
			FLM_DATA_CONTAINER, &uiTableId, pRec, 0)))
		{
			goto Exit;
		}
	}

Exit:

	if( pRec)
	{
		pRec->Release();
	}

	return( rc);
}
		
/****************************************************************************
Desc:
****************************************************************************/
RCODE FlmConnection::retrieveRowCount(
	FLMUINT				uiTableId,
	FLMUINT *			puiRowCount)
{
	RCODE					rc = FERR_OK;
	FlmRecord *			pRec = NULL;
	void *				pvField;

	*puiRowCount = 0;

	if( RC_BAD( rc = FlmRecordRetrieve( m_hDb, FLM_DATA_CONTAINER, 
		uiTableId, FO_EXACT, &pRec, NULL)))
	{
		if( rc == FERR_NOT_FOUND)
		{
			rc = FERR_OK;
		}

		goto Exit;
	}

	if( (pvField = pRec->find( pRec->root(), FLM_ROW_COUNT_FIELD_ID)) != NULL)
	{
		if( RC_BAD( rc = pRec->getUINT( pvField, puiRowCount)))
		{
			goto Exit;
		}
	}

Exit:

	if( pRec)
	{
		pRec->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FlmConnection::incrementRowCount(
	FLMUINT				uiTableId)
{
	RCODE					rc = FERR_OK;
	FLMUINT				uiRowCount;

	if( RC_BAD( rc = retrieveRowCount( uiTableId, &uiRowCount)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = storeRowCount( uiTableId, uiRowCount + 1)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FlmConnection::decrementRowCount(
	FLMUINT				uiTableId)
{
	RCODE					rc = FERR_OK;
	FLMUINT				uiRowCount;

	if( RC_BAD( rc = retrieveRowCount( uiTableId, &uiRowCount)))
	{
		goto Exit;
	}

	if( !uiRowCount)
	{
		rc = FERR_ILLEGAL_OP;
		goto Exit;
	}

	if( RC_BAD( rc = storeRowCount( uiTableId, uiRowCount - 1)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE ha_flaim::exportKeyToRec(
	uint				uiKeyNum,
	const byte *	pucRecord,
	FlmRecord *		pRec)
{
	RCODE					rc = FERR_OK;
	char *				pucBuf = m_pucKeyBuf;
	char *				pucBufEnd = &m_pucKeyBuf[ FLM_MAX_KEY_LENGTH];
	FIELD_INFO *		pKeyFields = m_pShare->pKeyFields;
	void *				pvField;
	KEY *					pKey = table->key_info + uiKeyNum;
	KEY_PART_INFO *	pCurKeyPart = pKey->key_part;
	KEY_PART_INFO *	pEndKeyPart = pCurKeyPart + pKey->key_parts;
  
	DBUG_ENTER( "ha_flaim::exportKeyToRec");

	for( ; pCurKeyPart != pEndKeyPart && pucBuf < pucBufEnd; pCurKeyPart++)
	{
		if( pCurKeyPart->null_bit)
		{
			// Store 0 if the key part is a NULL part
			
			if( pucRecord[ pCurKeyPart->null_offset] & pCurKeyPart->null_bit)
			{
				*pucBuf++ = 0;
				continue;
			}

			// Store NOT NULL marker

			*pucBuf++ = 1;
		}
		
		pucBuf = pCurKeyPart->field->pack_key( pucBuf, 
					(const char *)(pucRecord + pCurKeyPart->offset),
					pCurKeyPart->length);

		assert( pucBuf <= pucBufEnd);
	}

	if( RC_BAD( rc = pRec->insertLast( 1, 
		pKeyFields[ uiKeyNum].uiDictNum, FLM_BINARY_TYPE, &pvField)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pRec->setBinary( pvField, m_pucKeyBuf, pucBuf - m_pucKeyBuf)))
	{
		goto Exit;
	}
	
Exit:
	
	DBUG_RETURN( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
bool ha_flaim::convertString(
	String *				pString, 
	CHARSET_INFO *		pFromCharSet,
	CHARSET_INFO *		pToCharSet)
{
	uint dummy_errors;

	if( m_convertBuffer.copy( pString->ptr(), pString->length(), 
		pFromCharSet, pToCharSet, &dummy_errors))
	{
		return( true);
	}

	if( m_convertBuffer.alloced_length() >= m_convertBuffer.length() * 2 ||
		!pString->is_alloced())
	{
		return( pString->copy( m_convertBuffer));
	}

	pString->swap( m_convertBuffer);
	return( false);
}

/****************************************************************************
Desc:
****************************************************************************/
int ha_flaim::check(
	THD *						pThread,
	HA_CHECK_OPT *			pCheckOpt)
{
	RCODE					rc = FERR_OK;
	char					szDbPath[ F_PATH_MAX_SIZE];

	DBUG_ENTER( "ha_flaim::check");

	szDbPath[ 0] = 0;
	f_pathAppend( szDbPath, ".");
	f_pathAppend( szDbPath, pThread->db);
	f_pathAppend( szDbPath, FLAIM_DB_NAME);

	if( RC_BAD( rc = FlmDbCheck( NULL, szDbPath, NULL, 
		NULL, 0, NULL, NULL, NULL, NULL)))
	{
		goto Exit;
	}

Exit:

	if( rc == FERR_DATA_ERROR)
	{
		rc = (RCODE)HA_ADMIN_CORRUPT;
	}

	DBUG_RETURN( rc);
}

//#endif /* HAVE_FLAIM */
