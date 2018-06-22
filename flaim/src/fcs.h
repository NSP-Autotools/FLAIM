//-------------------------------------------------------------------------
// Desc:	Class definitions for client/server.
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

#ifndef FCS_H
#define FCS_H

#include "fpackon.h"
// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

// Record flags / masks

#define RECORD_ID_SIZE					((FLMUINT) 8)
#define RECORD_RESERVED_FLAG			((FLMUINT) 0x80)
#define RECORD_HAS_HTD_FLAG			((FLMUINT) 0x40)
#define RECORD_ID_SIZE_MASK			((FLMUINT) 0x3F)

// Session flags

#define FCS_SESSION_GEDCOM_SUPPORT	((FLMUINT) 0x0001)

// Misc. defines

#define FCS_INVALID_ID					0xFFFFFFFF
#define FCS_ITERATOR_MAX_PATH			((FLMUINT) 32)

// Supported versions

#define FCS_VERSION_1_1_0				((FLMUINT) 110)
#define FCS_VERSION_1_1_1				((FLMUINT) 111)

// Global operations

#define FCS_OPCLASS_GLOBAL				((FLMUINT) 0x01)
#define FCS_OP_GLOBAL_STATS_START		((FLMUINT) 0x02)
#define FCS_OP_GLOBAL_STATS_STOP			((FLMUINT) 0x03)
#define FCS_OP_GLOBAL_STATS_RESET		((FLMUINT) 0x04)
#define FCS_OP_GLOBAL_MEM_INFO_GET		((FLMUINT) 0x05)
#define FCS_OP_GLOBAL_GET_THREAD_INFO	((FLMUINT) 0x06)

// Session operations

#define FCS_OPCLASS_SESSION		((FLMUINT) 0x02)
#define FCS_OP_SESSION_OPEN			((FLMUINT) 0x01)
#define FCS_OP_SESSION_CLOSE			((FLMUINT) 0x02)
#define FCS_OP_SESSION_INTERRUPT		((FLMUINT) 0x03)

// Database operations

#define FCS_OPCLASS_DATABASE		((FLMUINT) 0x03)
#define FCS_OP_DATABASE_OPEN			((FLMUINT) 0x01)
#define FCS_OP_DATABASE_CREATE		((FLMUINT) 0x02)
#define FCS_OP_DATABASE_CLOSE			((FLMUINT) 0x03)
#define FCS_OP_DB_REDUCE_SIZE			((FLMUINT) 0x07)
#define FCS_OP_GET_ITEM_ID				((FLMUINT) 0x09)
#define FCS_OP_GET_ITEM_NAME			((FLMUINT) 0x0A)
#define FCS_OP_GET_NAME_TABLE			((FLMUINT) 0x0B)
#define FCS_OP_GET_COMMIT_CNT			((FLMUINT) 0x0C)
#define FCS_OP_GET_TRANS_ID			((FLMUINT) 0x0E)
#define FCS_OP_DATABASE_GET_CONFIG	((FLMUINT) 0x10)
#define FCS_OP_DATABASE_LOCK			((FLMUINT) 0x11)
#define FCS_OP_DATABASE_UNLOCK		((FLMUINT) 0x12)
#define FCS_OP_DATABASE_GET_BLOCK	((FLMUINT) 0x13)
#define FCS_OP_DATABASE_CHECKPOINT	((FLMUINT) 0x14)
#define FCS_OP_DB_SET_BACKUP_FLAG	((FLMUINT) 0x15) // Only used by FlmDbBackup routines
#define FCS_OP_DATABASE_CONFIG		((FLMUINT) 0x16)

// Transaction operations

#define FCS_OPCLASS_TRANS				((FLMUINT) 0x04)
#define FCS_OP_TRANSACTION_BEGIN			((FLMUINT) 0x01)
#define		FCS_TRANS_FLAG_GET_HEADER		((FLMUINT) 0x01)
#define		FCS_TRANS_FLAG_DONT_KILL		((FLMUINT) 0x02)
#define		FCS_TRANS_FORCE_CHECKPOINT		((FLMUINT) 0x04)
#define		FCS_TRANS_FLAG_DONT_POISON		((FLMUINT) 0x08)
#define FCS_OP_TRANSACTION_COMMIT		((FLMUINT) 0x02)
#define FCS_OP_TRANSACTION_ABORT			((FLMUINT) 0x03)
#define FCS_OP_TRANSACTION_GET_TYPE		((FLMUINT) 0x04)
#define FCS_OP_TRANSACTION_RESET			((FLMUINT) 0x05)
#define FCS_OP_TRANSACTION_COMMIT_EX	((FLMUINT) 0x06)

// Record / DRN operations

#define FCS_OPCLASS_RECORD			((FLMUINT) 0x05)
#define FCS_OP_RECORD_RETRIEVE		((FLMUINT) 0x01)
#define FCS_OP_RECORD_ADD				((FLMUINT) 0x02)
#define FCS_OP_RECORD_MODIFY			((FLMUINT) 0x03)
#define FCS_OP_RECORD_DELETE			((FLMUINT) 0x04)
#define FCS_OP_RESERVE_NEXT_DRN		((FLMUINT) 0x05)
#define FCS_OP_RECORD_PREP				((FLMUINT) 0x06)
#define FCS_OP_KEY_RETRIEVE			((FLMUINT) 0x07)

// Cursor / Query operations

#define FCS_OPCLASS_ITERATOR		((FLMUINT) 0x06)
#define FCS_OP_ITERATOR_INIT				((FLMUINT) 0x01)
#define FCS_OP_ITERATOR_FREE				((FLMUINT) 0x02)
#define FCS_OP_ITERATOR_FIRST				((FLMUINT) 0x04)
#define FCS_OP_ITERATOR_LAST				((FLMUINT) 0x05)
#define FCS_OP_ITERATOR_PREV				((FLMUINT) 0x06)
#define FCS_OP_ITERATOR_NEXT				((FLMUINT) 0x07)
#define FCS_OP_ITERATOR_COUNT				((FLMUINT) 0x08)
#define FCS_OP_ITERATOR_GETITEMS			((FLMUINT) 0x09)
#define FCS_OP_ITERATOR_NU_1				((FLMUINT) 0x0A)
#define FCS_OP_ITERATOR_TEST_REC			((FLMUINT) 0x0B)
#define FCS_OP_ITERATOR_NU_2				((FLMUINT) 0x0C)

// Callbacks

#define FCS_OPCLASS_CALLBACK		((FLMUINT) 0x07)
#define FCS_OP_CALLBACK_REGISTER		((FLMUINT) 0x01)
#define FCS_OP_CALLBACK_UNREGISTER	((FLMUINT) 0x02)

// BLOBs

#define FCS_OPCLASS_BLOB			((FLMUINT) 0x08)
#define FCS_OP_BLOB_OPEN				((FLMUINT) 0x01)
#define FCS_OP_BLOB_CREATE				((FLMUINT) 0x02)
#define FCS_OP_BLOB_CREATE_REF		((FLMUINT) 0x03)
#define FCS_OP_BLOB_CLONE				((FLMUINT) 0x04)
#define FCS_OP_BLOB_READ				((FLMUINT) 0x05)
#define FCS_OP_BLOB_SEEK				((FLMUINT) 0x06)
#define FCS_OP_BLOB_APPEND				((FLMUINT) 0x07)
#define FCS_OP_BLOB_ATTACH				((FLMUINT) 0x08)
#define FCS_OP_BLOB_CREATE_WLIST		((FLMUINT) 0x09)
#define FCS_OP_BLOB_EXPORT				((FLMUINT) 0x0A)
#define FCS_OP_BLOB_IMPORT				((FLMUINT) 0x0B)
#define FCS_OP_BLOB_PURGE				((FLMUINT) 0x0C)

// Maintenance

#define FCS_OPCLASS_MAINTENANCE	((FLMUINT) 0x0A)
#define FCS_OP_REBUILD					((FLMUINT) 0x01)
#define FCS_OP_CHECK						((FLMUINT) 0x02)
#define FCS_OP_PCODE_REBUILD			((FLMUINT) 0x03)
#define FCS_OP_INDEX_MAINTENANCE		((FLMUINT) 0x04)

// File System

#define FCS_OPCLASS_FILE			((FLMUINT) 0x0B)
#define FCS_OP_FILE_EXISTS				((FLMUINT) 0x01)
#define FCS_OP_FILE_DELETE				((FLMUINT) 0x02)
#define FCS_OP_FILE_COPY				((FLMUINT) 0x03)

// Indexing

#define FCS_OPCLASS_INDEX			((FLMUINT) 0x0C)
#define FCS_OP_INDEX_SUSPEND			((FLMUINT) 0x01)
#define FCS_OP_INDEX_RESUME			((FLMUINT) 0x02)
#define FCS_OP_INDEX_GET_STATUS		((FLMUINT) 0x03)
#define FCS_OP_INDEX_GET_NEXT			((FLMUINT) 0x04)

// Misc. operations

#define FCS_OPCLASS_MISC			((FLMUINT) 0x0D)
#define FCS_OP_CREATE_SERIAL_NUM		((FLMUINT) 0x01)

// Diagnostic operations

#define FCS_OPCLASS_DIAG			((FLMUINT) 0xF0)
#define FCS_OP_DIAG_HTD_ECHO			((FLMUINT) 0x01)

// Administration operations

#define FCS_OPCLASS_ADMIN			((FLMUINT) 0xF1)
#define FCS_OP_ABORT						((FLMUINT) 0x01)

// Create Opts Tags

#define FCS_COPT_CONTEXT				((FLMUINT) 0x0001)
#define FCS_COPT_BLOCK_SIZE			((FLMUINT) 0x0002)
#define FCS_COPT_MIN_RFL_FILE_SIZE	((FLMUINT) 0x0003)
#define FCS_COPT_DEFAULT_LANG			((FLMUINT) 0x0006)
#define FCS_COPT_VERSION				((FLMUINT) 0x0007)
#define FCS_COPT_RFL_STATE				((FLMUINT) 0x0008)
#define FCS_COPT_RESERVED				((FLMUINT) 0x0009)
#define FCS_COPT_APP_MAJOR_VER		((FLMUINT) 0x00A3)
#define FCS_COPT_APP_MINOR_VER		((FLMUINT) 0x00A4)
#define FCS_COPT_MAX_RFL_FILE_SIZE	((FLMUINT) 0x00A5)
#define FCS_COPT_KEEP_RFL_FILES		((FLMUINT) 0x00A6)
#define FCS_COPT_LOG_ABORTED_TRANS	((FLMUINT) 0x00A7)


// Name Table Tags

#define FCS_NAME_TABLE_CONTEXT		((FLMUINT) 0x0001)
#define FCS_NAME_TABLE_ITEM_ID		((FLMUINT) 0x0002)
#define FCS_NAME_TABLE_ITEM_NAME		((FLMUINT) 0x0003)
#define FCS_NAME_TABLE_ITEM_TYPE		((FLMUINT) 0x0004)
#define FCS_NAME_TABLE_ITEM_SUBTYPE	((FLMUINT) 0x0005)

// Iterator Tags

#define FCS_ITERATOR_SELECT					((FLMUINT) 1)
#define FCS_ITERATOR_FROM						((FLMUINT) 2)
#define FCS_ITERATOR_CANDIDATE_SET			((FLMUINT) 3)
#define FCS_ITERATOR_RECORD_TYPE				((FLMUINT) 4)
#define FCS_ITERATOR_FLAIM_INDEX				((FLMUINT) 5)
#define FCS_ITERATOR_QF_INDEX					((FLMUINT) 6)
#define FCS_ITERATOR_RECORD_SOURCE			((FLMUINT) 7)
#define FCS_ITERATOR_CONTAINER_ID			((FLMUINT) 10)
#define FCS_ITERATOR_WHERE						((FLMUINT) 11)
#define FCS_ITERATOR_OPERATOR					((FLMUINT) 12)
#define FCS_ITERATOR_ATTRIBUTE				((FLMUINT) 13)
#define FCS_ITERATOR_ATTRIBUTE_PATH			((FLMUINT) 14)
//#define FCS_ITERATOR_NOT_USED				((FLMUINT) 15)
#define FCS_ITERATOR_NUMBER_VALUE			((FLMUINT) 16)
#define FCS_ITERATOR_UNICODE_VALUE			((FLMUINT) 17)
#define FCS_ITERATOR_BINARY_VALUE			((FLMUINT) 18)
#define FCS_ITERATOR_REQUIRED					((FLMUINT) 19)
#define FCS_ITERATOR_CONFIG					((FLMUINT) 20)
#define FCS_ITERATOR_WP60_VALUE				((FLMUINT) 21)
#define FCS_ITERATOR_NATIVE_VALUE			((FLMUINT) 22)
#define FCS_ITERATOR_WDSTR_VALUE				((FLMUINT) 23)
#define FCS_ITERATOR_REAL_VALUE				((FLMUINT) 24)
#define FCS_ITERATOR_REC_PTR_VALUE			((FLMUINT) 25)
#define FCS_ITERATOR_DATE_VALUE				((FLMUINT) 26)
#define FCS_ITERATOR_TIME_VALUE				((FLMUINT) 27)
#define FCS_ITERATOR_TIMESTAMP_VALUE		((FLMUINT) 28)
#define FCS_ITERATOR_VIEW_TREE				((FLMUINT) 29)
#define FCS_ITERATOR_NULL_VIEW_NOT_REC		((FLMUINT) 30)
#define FCS_ITERATOR_QF_STRING				((FLMUINT) 31)
#define FCS_ITERATOR_NO_QF_SLOW_HITS		((FLMUINT) 32)
#define FCS_ITERATOR_MODE						((FLMUINT) 34)
#define FCS_ITERATOR_FLM_TEXT_VALUE			((FLMUINT) 35)
#define FCS_ITERATOR_OK_TO_RETURN_KEYS		((FLMUINT) 36)

#define FCS_ITERATOR_OP_START					((FLMUINT) 1)
#define FCS_ITERATOR_AND_OP					((FLMUINT) 1)
#define FCS_ITERATOR_OR_OP						((FLMUINT) 2)
#define FCS_ITERATOR_NOT_OP					((FLMUINT) 3)
#define FCS_ITERATOR_EQ_OP						((FLMUINT) 4)
#define FCS_ITERATOR_MATCH_OP					((FLMUINT) 5)
#define FCS_ITERATOR_MATCH_BEGIN_OP			((FLMUINT) 6)
#define FCS_ITERATOR_CONTAINS_OP				((FLMUINT) 7)
#define FCS_ITERATOR_NE_OP						((FLMUINT) 8)
#define FCS_ITERATOR_LT_OP						((FLMUINT) 9)
#define FCS_ITERATOR_LE_OP						((FLMUINT) 10)
#define FCS_ITERATOR_GT_OP						((FLMUINT) 11)
#define FCS_ITERATOR_GE_OP						((FLMUINT) 12)
#define FCS_ITERATOR_BITAND_OP				((FLMUINT) 13)
#define FCS_ITERATOR_BITOR_OP					((FLMUINT) 14)
#define FCS_ITERATOR_BITXOR_OP				((FLMUINT) 15)
#define FCS_ITERATOR_MULT_OP					((FLMUINT) 16)
#define FCS_ITERATOR_DIV_OP					((FLMUINT) 17)
#define FCS_ITERATOR_MOD_OP					((FLMUINT) 18)
#define FCS_ITERATOR_PLUS_OP					((FLMUINT) 19)
#define FCS_ITERATOR_MINUS_OP					((FLMUINT) 20)
#define FCS_ITERATOR_NEG_OP					((FLMUINT) 21)
#define FCS_ITERATOR_LPAREN_OP				((FLMUINT) 22)
#define FCS_ITERATOR_RPAREN_OP				((FLMUINT) 23)
#define FCS_ITERATOR_OP_END					((FLMUINT) 23)

// Iterator Flags

#define FCS_ITERATOR_DRN_FLAG			((FLMUINT) 0x0001)

// Checkpoint Info Tags

#define FCS_CPI_CONTEXT							((FLMUINT) 1)
#define FCS_CPI_RUNNING							((FLMUINT) 2)
#define FCS_CPI_START_TIME						((FLMUINT) 3)
#define FCS_CPI_FORCING_CP						((FLMUINT) 4)
#define FCS_CPI_FORCE_CP_START_TIME			((FLMUINT) 5)
#define FCS_CPI_FORCE_CP_REASON				((FLMUINT) 6)
#define FCS_CPI_WRITING_DATA_BLOCKS			((FLMUINT) 7)
#define FCS_CPI_LOG_BLOCKS_WRITTEN			((FLMUINT) 8)
#define FCS_CPI_DATA_BLOCKS_WRITTEN			((FLMUINT) 9)
#define FCS_CPI_DIRTY_CACHE_BYTES			((FLMUINT) 10)
#define FCS_CPI_BLOCK_SIZE						((FLMUINT) 11)
#define FCS_CPI_WAIT_TRUNC_TIME				((FLMUINT) 12)

// Lock User Tags

#define FCS_LUSR_CONTEXT						((FLMUINT) 1)
#define FCS_LUSR_THREAD_ID						((FLMUINT) 2)
#define FCS_LUSR_TIME							((FLMUINT) 3)

// Index Status Tags

#define FCS_IXSTAT_CONTEXT						((FLMUINT) 1)
#define FCS_IXSTAT_INDEX_NUM					((FLMUINT) 2)
#define FCS_IXSTAT_SUSPEND_TIME				((FLMUINT) 3)
#define FCS_IXSTAT_THREAD_ID					((FLMUINT) 4)
#define FCS_IXSTAT_START_TIME					((FLMUINT) 5)
#define FCS_IXSTAT_FIRST_REC_INDEXED		((FLMUINT) 6)
#define FCS_IXSTAT_LAST_REC_INDEXED			((FLMUINT) 7)
#define FCS_IXSTAT_KEYS_PROCESSED			((FLMUINT) 8)
#define FCS_IXSTAT_RECS_PROCESSED			((FLMUINT) 9)
#define FCS_IXSTAT_AUTO_ONLINE				((FLMUINT) 10)
#define FCS_IXSTAT_PRIORITY					((FLMUINT) 11)
#define FCS_IXSTAT_STATE						((FLMUINT) 12)

// Memory Info Tags

#define FCS_MEMINFO_CONTEXT					((FLMUINT) 1)
#define FCS_MEMINFO_DYNA_CACHE_ADJ			((FLMUINT) 2)
#define FCS_MEMINFO_CACHE_ADJ_PERCENT		((FLMUINT) 3)
#define FCS_MEMINFO_CACHE_ADJ_MIN			((FLMUINT) 4)
#define FCS_MEMINFO_CACHE_ADJ_MAX			((FLMUINT) 5)
#define FCS_MEMINFO_CACHE_ADJ_MIN_LEAVE	((FLMUINT) 6)
#define FCS_MEMINFO_RECORD_CACHE				((FLMUINT) 7)
#define FCS_MEMINFO_BLOCK_CACHE				((FLMUINT) 8)
#define FCS_MEMINFO_MAX_BYTES					((FLMUINT) 9)
#define FCS_MEMINFO_COUNT						((FLMUINT) 10)
#define FCS_MEMINFO_OLD_VER_COUNT			((FLMUINT) 11)
#define FCS_MEMINFO_TOTAL_BYTES_ALLOC		((FLMUINT) 12)
#define FCS_MEMINFO_OLD_VER_BYTES			((FLMUINT) 13)
#define FCS_MEMINFO_CACHE_HITS				((FLMUINT) 14)
#define FCS_MEMINFO_CACHE_HIT_LOOKS			((FLMUINT) 15)
#define FCS_MEMINFO_CACHE_FAULTS				((FLMUINT) 16)
#define FCS_MEMINFO_CACHE_FAULT_LOOKS		((FLMUINT) 17)

// Thread Info Tags

#define FCS_THREAD_INFO_ROOT					((FLMUINT) 1)
#define FCS_THREAD_INFO_CONTEXT				((FLMUINT) 2)
#define FCS_THREADINFO_THREAD_ID				((FLMUINT) 3)
#define FCS_THREADINFO_THREAD_GROUP			((FLMUINT) 4)
#define FCS_THREADINFO_APP_ID					((FLMUINT) 5)
#define FCS_THREADINFO_START_TIME			((FLMUINT) 6)
#define FCS_THREADINFO_THREAD_NAME			((FLMUINT) 7)
#define FCS_THREADINFO_THREAD_STATUS		((FLMUINT) 8)

// HTD types

#define HTD_TYPE_UINT					((FLMUINT) 0x01)
#define HTD_TYPE_INT						((FLMUINT) 0x02)
#define HTD_TYPE_REAL					((FLMUINT) 0x03)
#define HTD_TYPE_UNICODE				((FLMUINT) 0x04)
#define HTD_TYPE_BINARY					((FLMUINT) 0x05)
#define HTD_TYPE_CONTEXT				((FLMUINT) 0x06)
#define HTD_TYPE_DATE					((FLMUINT) 0x07)
#define HTD_TYPE_TIME					((FLMUINT) 0x08)
#define HTD_TYPE_TMSTAMP				((FLMUINT) 0x09)
#define HTD_TYPE_BLOB					((FLMUINT) 0x0A)
#define HTD_TYPE_GEDCOM					((FLMUINT) 0x0B)
#define HTD_TYPE_UINT64					((FLMUINT) 0x1C)
#define HTD_TYPE_INT64					((FLMUINT) 0x1D)
#define HTD_TYPE_RESERVED_3			((FLMUINT) 0x1E)
#define HTD_TYPE_RESERVED_4			((FLMUINT) 0x1F)

// HTD type masks

#define HTD_HAS_VALUE_FLAG				((FLMUINT) 0x80)
#define HTD_VALUE_TYPE_MASK			((FLMUINT) 0x1F)
#define HTD_LEVEL_MASK					((FLMUINT) 0x60)
#define HTD_LEVEL_POS					((FLMUINT) 5)

// HTD level tags

#define HTD_LEVEL_SIBLING				((FLMUINT) 0x00)
#define HTD_LEVEL_CHILD					((FLMUINT) 0x01)
#define HTD_LEVEL_BACK					((FLMUINT) 0x02)
#define HTD_LEVEL_BACK_X				((FLMUINT) 0x03)

// Paramter and return value size bits are embedded in the value
// tags (see below).  The size is extracted from the 4 high-order
// bits of the parameter / return value tag.  Bits 10 and 11 are
// reserved for future use.  The number of value tags is limited to
// a 10-bit representation (1024).  The reserved bits could be
//used to expand the number of available tags.

#define WIRE_VALUE_TYPE_GEN_0				((FLMUINT) 0x00)
#define WIRE_VALUE_TYPE_GEN_1				((FLMUINT) 0x01)
#define WIRE_VALUE_TYPE_GEN_2				((FLMUINT) 0x02)
#define WIRE_VALUE_TYPE_GEN_4				((FLMUINT) 0x03)
#define WIRE_VALUE_TYPE_GEN_8				((FLMUINT) 0x04)
#define WIRE_VALUE_TYPE_UTF				((FLMUINT) 0x05)
#define WIRE_VALUE_TYPE_BINARY			((FLMUINT) 0x06)
#define WIRE_VALUE_TYPE_HTD				((FLMUINT) 0x07)
#define WIRE_VALUE_TYPE_RECORD			((FLMUINT) 0x08)
#define WIRE_VALUE_TYPE_LARGE_BINARY	((FLMUINT) 0x09)
#define WIRE_VALUE_TYPE_RESERVED_2		((FLMUINT) 0x0A)
#define WIRE_VALUE_TYPE_RESERVED_3		((FLMUINT) 0x0B)
#define WIRE_VALUE_TYPE_RESERVED_4		((FLMUINT) 0x0C)
#define WIRE_VALUE_TYPE_RESERVED_5		((FLMUINT) 0x0D)
#define WIRE_VALUE_TYPE_RESERVED_6		((FLMUINT) 0x0E)
#define WIRE_VALUE_TYPE_RESERVED_7		((FLMUINT) 0x0F)

#define WIRE_VALUE_TAG_MASK				((FLMUINT) 0x03FF)
#define WIRE_VALUE_TYPE_MASK				((FLMUINT) 0xF000)
#define WIRE_VALUE_TYPE_START_BIT		((FLMUINT) 12)

// Parameters and return values

#define WIRE_VALUE_START				((FLMUINT) 0x0000)

#define WIRE_VALUE_TERMINATE			((FLMUINT) WIRE_VALUE_START)
#define WIRE_VALUE_SESSION_ID			((FLMUINT) WIRE_VALUE_START + 1     ) // Number
#define WIRE_VALUE_DB_ID				((FLMUINT) WIRE_VALUE_START + 2     ) // Number
#define WIRE_VALUE_FILE_PATH			((FLMUINT) WIRE_VALUE_START + 3     ) // UTF
#define WIRE_VALUE_DICT_FILE_PATH	((FLMUINT) WIRE_VALUE_START + 4     ) // UTF
#define WIRE_VALUE_PASSWORD			((FLMUINT) WIRE_VALUE_START + 5     ) // Binary
#define WIRE_VALUE_FLAGS				((FLMUINT) WIRE_VALUE_START + 6     ) // Number
#define WIRE_VALUE_CLIENT_VERSION	((FLMUINT) WIRE_VALUE_START + 7     ) // Number
#define WIRE_VALUE_MOUNT_POINT		((FLMUINT) WIRE_VALUE_START + 8     ) // Number
#define WIRE_VALUE_RCODE				((FLMUINT) WIRE_VALUE_START + 9     ) // Number
#define WIRE_VALUE_DRN					((FLMUINT) WIRE_VALUE_START + 10    ) // Number
#define WIRE_VALUE_CONTAINER_ID		((FLMUINT) WIRE_VALUE_START + 11    ) // Number
#define WIRE_VALUE_AUTOTRANS			((FLMUINT) WIRE_VALUE_START + 13    ) // Number
#define WIRE_VALUE_RECORD				((FLMUINT) WIRE_VALUE_START + 14    ) // Record
#define WIRE_VALUE_DICT_BUFFER		((FLMUINT) WIRE_VALUE_START + 15    ) // UTF
#define WIRE_VALUE_SHARED_DICT_ID	((FLMUINT) WIRE_VALUE_START + 16    ) // Number
#define WIRE_VALUE_PARENT_DICT_ID	((FLMUINT) WIRE_VALUE_START + 17    ) // Number
#define WIRE_VALUE_AREA_ID				((FLMUINT) WIRE_VALUE_START + 18    ) // Number
#define WIRE_VALUE_FILE_NAME			((FLMUINT) WIRE_VALUE_START + 19    ) // UTF
#define WIRE_VALUE_COUNT				((FLMUINT) WIRE_VALUE_START + 20    ) // Number
#define WIRE_VALUE_TRANSACTION_ID	((FLMUINT) WIRE_VALUE_START + 21    ) // Number
#define WIRE_VALUE_TRANSACTION_TYPE	((FLMUINT) WIRE_VALUE_START + 22    ) // Number
#define WIRE_VALUE_MAX_LOCK_WAIT		((FLMUINT) WIRE_VALUE_START + 23    ) // Number
#define WIRE_VALUE_HTD					((FLMUINT) WIRE_VALUE_START + 24    ) // HTD
#define WIRE_VALUE_ITERATOR_ID		((FLMUINT) WIRE_VALUE_START + 25    ) // Number
#define WIRE_VALUE_ITERATOR_SELECT	((FLMUINT) WIRE_VALUE_START + 26    ) // HTD
#define WIRE_VALUE_ITERATOR_FROM		((FLMUINT) WIRE_VALUE_START + 27    ) // HTD
#define WIRE_VALUE_ITERATOR_WHERE	((FLMUINT) WIRE_VALUE_START + 28    ) // HTD
#define WIRE_VALUE_ITERATOR_CONFIG	((FLMUINT) WIRE_VALUE_START + 29    ) // HTD
#define WIRE_VALUE_RECORD_COUNT		((FLMUINT) WIRE_VALUE_START + 30    ) // Number
#define WIRE_VALUE_CALLBACK_TYPE		((FLMUINT) WIRE_VALUE_START + 31    ) // Number
#define WIRE_VALUE_FUNCTION_ID		((FLMUINT) WIRE_VALUE_START + 32    ) // Number
#define WIRE_VALUE_NUMBER2				((FLMUINT) WIRE_VALUE_START + 33    ) // Number
#define WIRE_VALUE_NUMBER3				((FLMUINT) WIRE_VALUE_START + 34    ) // Number
#define WIRE_VALUE_USER_DATA			((FLMUINT) WIRE_VALUE_START + 35    ) // Number
#define WIRE_VALUE_ITEM_ID				((FLMUINT) WIRE_VALUE_START + 36    ) // Number
#define WIRE_VALUE_ITEM_NAME			((FLMUINT) WIRE_VALUE_START + 37    ) // UTF
#define WIRE_VALUE_CREATE_OPTS		((FLMUINT) WIRE_VALUE_START + 38    ) // HTD
#define WIRE_VALUE_NAME_TABLE			((FLMUINT) WIRE_VALUE_START + 39    ) // HTD
#define WIRE_VALUE_ROPS_ID				((FLMUINT) WIRE_VALUE_START + 40    ) // Number
#define WIRE_VALUE_ROPS					((FLMUINT) WIRE_VALUE_START + 41    ) // HTD
#define WIRE_VALUE_INDEX_ID			((FLMUINT) WIRE_VALUE_START + 42    ) // Number
#define WIRE_VALUE_DRN_LIST			((FLMUINT) WIRE_VALUE_START + 43    ) // Binary
#define WIRE_VALUE_OP_SEQ_NUM			((FLMUINT) WIRE_VALUE_START + 44    ) // Number
#define WIRE_VALUE_BOOLEAN				((FLMUINT) WIRE_VALUE_START + 45    ) // Number
#define WIRE_VALUE_MAINT_SEQ_NUM		((FLMUINT) WIRE_VALUE_START + 46    ) // Number
#define WIRE_VALUE_OFFSET				((FLMUINT) WIRE_VALUE_START + 47    ) // Number
#define WIRE_VALUE_WHENCE				((FLMUINT) WIRE_VALUE_START + 48    ) // Number
#define WIRE_VALUE_BLOB_TYPE			((FLMUINT) WIRE_VALUE_START + 49    ) // Number
#define WIRE_VALUE_BLOB_ID				((FLMUINT) WIRE_VALUE_START + 50    ) // Number
#define WIRE_VALUE_EXTENDED_PATH		((FLMUINT) WIRE_VALUE_START + 51    ) // UTF
#define WIRE_VALUE_BUFFER				((FLMUINT) WIRE_VALUE_START + 52    ) // Binary
#define WIRE_VALUE_DEST_PATH			((FLMUINT) WIRE_VALUE_START + 53    ) // UTF
#define WIRE_VALUE_SESSION_COOKIE	((FLMUINT) WIRE_VALUE_START + 54    ) // Number
#define WIRE_VALUE_TYPE					((FLMUINT) WIRE_VALUE_START + 55    ) // Number
#define WIRE_VALUE_NUMBER1				((FLMUINT) WIRE_VALUE_START + 56    ) // Number
#define WIRE_VALUE_SIGNED_NUMBER		((FLMUINT) WIRE_VALUE_START + 57    ) // Number
#define WIRE_VALUE_BLOCK				((FLMUINT) WIRE_VALUE_START + 58    ) // Binary
#define WIRE_VALUE_ADDRESS				((FLMUINT) WIRE_VALUE_START + 59    ) // Number
#define WIRE_VALUE_FROM_KEY			((FLMUINT) WIRE_VALUE_START + 60    ) // Record
#define WIRE_VALUE_UNTIL_KEY			((FLMUINT) WIRE_VALUE_START + 61    ) // Record
#define WIRE_VALUE_FILE_PATH_2		((FLMUINT) WIRE_VALUE_START + 62    ) // UTF
#define WIRE_VALUE_SERIAL_NUM			((FLMUINT) WIRE_VALUE_START + 63    ) // Binary
#define WIRE_VALUE_FLAIM_VERSION		((FLMUINT) WIRE_VALUE_START + 64		) // Number
#define WIRE_VALUE_FILE_PATH_3		((FLMUINT) WIRE_VALUE_START + 65    ) // UTF

#define WIRE_VALUE_START_ASYNC		((FLMUINT) WIRE_VALUE_START + 0x0300) // Tag

// Stream Protocol

#define FCS_STREAM_POST_FLAG					((FLMUINT) 0x00000001)
#define FCS_STREAM_GET_FLAG					((FLMUINT) 0x00000002)
#define FCS_STREAM_LAST_PKT_FLAG				((FLMUINT) 0x00000004)
#define FCS_STREAM_PENDING_FLAG				((FLMUINT) 0x00000008)

/****************************************************************************
Desc:
****************************************************************************/
class	FCS_OSTM : public virtual F_Object
{
public:

	virtual RCODE close( void) = 0;

	virtual RCODE flush( void) = 0;

	virtual RCODE write( 
		FLMBYTE *		pucData,
		FLMUINT			uiLength) = 0;

	virtual RCODE endMessage( void) = 0;
};

/****************************************************************************
Desc:
****************************************************************************/
class	FCS_ISTM : public virtual F_Object
{
public:

	virtual FLMBOOL isOpen( void) = 0;

	virtual RCODE close( void) = 0;

	virtual RCODE flush( void) = 0;

	virtual RCODE endMessage( void) = 0;

	virtual RCODE read(
		FLMBYTE *		pucData,
		FLMUINT			uiLength,
		FLMUINT *		puiBytesRead) = 0;
};

/****************************************************************************
Desc: 
****************************************************************************/
class	FCS_DIS : public FCS_ISTM
{
public:

	#define FCS_DIS_BUFFER_SIZE		1024

	FCS_DIS( void);

	virtual ~FCS_DIS( void);

	RCODE setup( 
		FCS_ISTM *		pIStream);

	RCODE readByte( 
		FLMBYTE *		pValue);

	RCODE readShort( 
		FLMINT16 *		pValue);

	RCODE readUShort( 
		FLMUINT16 *		pValue);

	RCODE readInt( 
		FLMINT32 *		pValue);

	RCODE readUInt( 
		FLMUINT32 *		pValue);

	RCODE readInt64( 
		FLMINT64 *		pValue);

	RCODE readUInt64( 
		FLMUINT64 *		pValue);

	RCODE readBinary( 
		F_Pool *			pPool,
		FLMBYTE **		ppValue,
		FLMUINT *		puiDataSize);

	RCODE readLargeBinary( 
		F_Pool *			pPool,
		FLMBYTE **		ppValue,
		FLMUINT *		puiDataSize);

	RCODE readUTF( 
		F_Pool *			pPool,
		FLMUNICODE **	ppValue);

	RCODE readHTD(
		F_Pool *			pPool,
		FLMUINT			uiContainer,
		FLMUINT			uiDrn,
		NODE **			ppNode,
		FlmRecord **	ppRecord);

	FLMBOOL isOpen( void);

	RCODE flush( void);
	
	RCODE close( void);
	
	RCODE endMessage( void);
	
	RCODE read( 
		FLMBYTE *		pucData,
		FLMUINT			uiLength,
		FLMUINT *		puiBytesRead);
	
	RCODE skip( 
		FLMUINT			uiBytesToSkip);

private:

	FCS_ISTM *		m_pIStream;
	FLMBYTE			m_pucBuffer[ FCS_DIS_BUFFER_SIZE];
	FLMUINT			m_uiBOffset;
	FLMUINT			m_uiBDataSize;
	FLMBOOL			m_bSetupCalled;

};

/****************************************************************************
Desc: 
****************************************************************************/
class	FCS_DOS : public FCS_OSTM
{
#define FCS_DOS_BUFFER_SIZE		1024

	FCS_OSTM *		m_pOStream;
	FLMBYTE			m_pucBuffer[ FCS_DOS_BUFFER_SIZE];
	FLMUINT			m_uiBOffset;
	FLMBOOL			m_bSetupCalled;
	F_Pool			m_tmpPool;

public:

	FCS_DOS( void);

	virtual ~FCS_DOS( void);
	
	FINLINE RCODE setup(
		FCS_OSTM *		pOStream)
	{
		m_pOStream = pOStream;
		m_bSetupCalled = TRUE;
		return( FERR_OK);
	}
	
	FINLINE RCODE writeByte(
		FLMBYTE	ucValue)
	{
		return( write( &ucValue, 1));
	}

	FINLINE RCODE writeShort(
		FLMINT16		i16Value)
	{
		FLMBYTE	tmpBuf[ 2];

		f_INT16ToBigEndian( i16Value, tmpBuf);
		return( write( tmpBuf, 2));
	}

	FINLINE RCODE writeUShort(
		FLMUINT16	ui16Value)
	{
		FLMBYTE	tmpBuf[ 2];

		f_UINT16ToBigEndian( ui16Value, tmpBuf);
		return( write( tmpBuf, 2));
	}

	FINLINE RCODE writeInt32(
		FLMINT32		i32Value)
	{
		FLMBYTE	tmpBuf[ 4];

		f_INT32ToBigEndian( i32Value, tmpBuf);
		return( write( tmpBuf, 4));
	}
	
	FINLINE RCODE writeUInt32(
		FLMUINT32	ui32Value)
	{
		FLMBYTE	tmpBuf[ 4];

		f_UINT32ToBigEndian( ui32Value, tmpBuf);
		return( write( tmpBuf, 4));
	}

	FINLINE RCODE writeInt64(
		FLMINT64		i64Value)
	{
		FLMBYTE	tmpBuf[ 8];

		f_INT64ToBigEndian( i64Value, tmpBuf);
		return( write( tmpBuf, 8));
	}

	FINLINE RCODE writeUInt64(
		FLMUINT64	ui64Value)
	{
		FLMBYTE	tmpBuf[ 8];

		f_UINT64ToBigEndian( ui64Value, tmpBuf);
		return( write( tmpBuf, 8));
	}
	
	RCODE writeBinary(
		FLMBYTE *		pucValue,
		FLMUINT			uiSize);
	
	RCODE writeLargeBinary(
		FLMBYTE *		pucValue,
		FLMUINT			uiSize);

	RCODE writeUTF(
		FLMUNICODE *	puzValue);

	RCODE writeHTD(
		NODE *			pHTD,
		FlmRecord *		pRecord,
		FLMBOOL			bSendForest,
		FLMBOOL			bSendAsGedcom);

	RCODE write( 
		FLMBYTE *		pucData,
		FLMUINT			uiLength);
	
	RCODE close( void);
	
	RCODE endMessage( void);
	
	RCODE flush( void);
};

/****************************************************************************
Desc: 
****************************************************************************/
class	FCS_WIRE : public F_Object
{
protected:

	FLMUINT				m_uiClass;
	FLMUINT				m_uiOp;
	FLMUINT				m_uiRCode;
	FLMUINT				m_uiDrn;
	FLMUINT64			m_ui64Count;
	FLMUINT				m_uiItemId;
	FLMUNICODE *		m_puzItemName;
	FLMUNICODE *		m_puzFilePath;
	FLMUNICODE *		m_puzFilePath2;
	FLMUNICODE *		m_puzFilePath3;
	FLMUINT				m_uiTransType;
	FLMUINT				m_uiBlockSize;
	FLMBYTE *			m_pucBlock;
	FLMBYTE *			m_pucSerialNum;
	FlmRecord *			m_pRecord;
	FlmRecord *			m_pFromKey;
	FlmRecord *			m_pUntilKey;
	NODE *				m_pHTD;
	CREATE_OPTS			m_CreateOpts;
	FLMUINT				m_uiSessionId;
	FLMUINT				m_uiSessionCookie;
	FLMUINT				m_uiContainer;
	FLMUINT				m_uiTransId;
	FLMUINT				m_uiIteratorId;
	FLMUINT64			m_ui64Number1;
	FLMUINT64			m_ui64Number2;
	FLMUINT64			m_ui64Number3;
	FLMUINT				m_uiAddress;
	FLMINT64				m_i64SignedValue;
	FLMUINT				m_uiIndexId;
	FLMBOOL				m_bIncludesAsync;
	FLMBOOL				m_bFlag;
	FLMUINT				m_uiFlags;
	FLMUINT				m_uiFlaimVersion;

	HFDB					m_hDb;
	F_Pool				m_pool;
	F_Pool *				m_pPool;
	FLMBOOL				m_bSendGedcom;
	FCS_DIS *			m_pDIStream;
	FCS_DOS *			m_pDOStream;
	CS_CONTEXT *		m_pCSContext;		// Used by FCL_WIRE
	FDB *					m_pDb;				// Used by FCL_WIRE

	void resetCommon( void);

	RCODE readOpcode( void);

	RCODE readCommon(
		FLMUINT *	puiTagRV,
		FLMBOOL *	pbEndRV);

	RCODE	receiveCreateOpts( void);

	RCODE	readNumber(
		FLMUINT			uiTag,
		FLMUINT *		puiNumber,
		FLMINT *			piNumber = NULL,
		FLMUINT64 *		pui64Number = NULL,
		FLMINT64 *		pi64Number = NULL);

	RCODE	writeUnsignedNumber(
		FLMUINT			uiTag,
		FLMUINT64		ui64Number);

	RCODE writeSignedNumber(
		FLMUINT			uiTag,
		FLMINT64			i64Number);

	RCODE	skipValue(
		FLMUINT			uiParm);

	RCODE receiveRecord(
		FlmRecord **	ppRecord);

	RCODE	receiveNameTable(
		F_NameTable **		ppNameTable);

public:

	FCS_WIRE(
		FCS_DIS *		pDIStream = NULL,
		FCS_DOS *		pDOStream = NULL);

	virtual ~FCS_WIRE( void);

	virtual void reset( void) = 0;

	virtual RCODE read( void) = 0;

	RCODE	sendOpcode(
		FLMUINT			uiClass,
		FLMUINT			uiOp);

	RCODE sendTerminate( void);

	FINLINE RCODE sendRc(
		RCODE			rcToSend)
	{
		// Send the return code if it is non-zero.

		if( RC_BAD( rcToSend))
		{
			return( writeUnsignedNumber( WIRE_VALUE_RCODE, (FLMUINT)rcToSend));
		}
		
		return( FERR_OK);
	}

	RCODE sendNumber(
		FLMUINT			uiTag,
		FLMUINT64		ui64Value,
		FLMINT64			i64Value = 0);

	RCODE sendBinary(
		FLMUINT			uiTag,
		FLMBYTE *		pData,
		FLMUINT			uiLength);

	RCODE sendRecord(
		FLMUINT			uiTag,
		FlmRecord *		pRecord);

	RCODE sendDrnList(
		FLMUINT			uiTag,
		FLMUINT *		puiList);

	RCODE sendString(
		FLMUINT			uiTag,
		FLMUNICODE *	puzString);

	RCODE sendHTD(
		FLMUINT			uiTag,
		NODE *			pHTD);

	RCODE sendHTD(
		FLMUINT			uiTag,
		FlmRecord *		pRecord);

	RCODE sendCreateOpts(
		FLMUINT			uiTag,
		CREATE_OPTS *	pCreateOpts);

	RCODE sendNameTable(
		FLMUINT			uiTag,
		F_NameTable *	pNameTable);

	FINLINE FlmRecord * getRecord( void)
	{ 
		return( m_pRecord);
	}

	FINLINE void setRecord( FlmRecord * pRecord)
	{ 
		// If records are equal, no need to do anything.
		// In fact the code would not work properly because the call
		// to Release might free the record, in which case m_pRec would
		// be pointing to freed space.  The AddRef would then be done on a
		// freed record.

		if( m_pRecord != pRecord)
		{
			if( m_pRecord)
			{
				m_pRecord->Release();
			}
	
			m_pRecord = pRecord;
			
			if( m_pRecord)
			{
				m_pRecord->AddRef();
			}
		}
	}

	FINLINE FlmRecord * getFromKey( void)
	{ 
		return( m_pFromKey);
	}

	FINLINE void setFromKey( FlmRecord * pFromKey)
	{ 
		// If records are equal, no need to do anything.
		// In fact the code would not work properly because the call
		// to Release might free the record, in which case m_pRec would
		// be pointing to freed space.  The AddRef would then be done on a
		// freed record.

		if( m_pFromKey != pFromKey)
		{
			if( m_pFromKey)
			{
				m_pFromKey->Release();
			}
	
			m_pFromKey = pFromKey;
			
			if( m_pFromKey)
			{
				m_pFromKey->AddRef();
			}
		}
	}

	FINLINE FlmRecord * getUntilKey( void)
	{ 
		return( m_pUntilKey);
	}

	FINLINE void setUntilKey( FlmRecord * pUntilKey)
	{ 
		// If records are equal, no need to do anything.
		// In fact the code would not work properly because the call
		// to Release might free the record, in which case m_pRec would
		// be pointing to freed space.  The AddRef would then be done on a
		// freed record.

		if( m_pUntilKey != pUntilKey)
		{
			if( m_pUntilKey)
			{
				m_pUntilKey->Release();
			}
	
			m_pUntilKey = pUntilKey;
			
			if( m_pUntilKey)
			{
				m_pUntilKey->AddRef();
			}
		}
	}

	FINLINE NODE * getHTD( void) 
	{ 
		return( m_pHTD);
	}
	
	RCODE getHTD(
		F_Pool *			pPool,
		NODE ** 			ppTreeRV);
	
	void copyCreateOpts( 
		CREATE_OPTS * 	pCreateOpts);
	
	FINLINE FLMUINT getClass( void)
	{
		return( m_uiClass);
	}
	
	FINLINE FLMUINT getOp( void) 
	{
		return( m_uiOp);
	}
	
	FINLINE RCODE getRCode( void)
	{
		return( (RCODE)m_uiRCode);
	}
	
	FINLINE FLMUINT getDrn( void)
	{
		return( m_uiDrn);
	}
	
	FINLINE FLMUINT64 getCount( void)
	{
		return( m_ui64Count);
	}
	
	FINLINE FLMUINT getTransType( void)
	{
		return( m_uiTransType);
	}
	
	FINLINE FLMUINT getSessionId( void)
	{
		return( m_uiSessionId);
	}
	
	FINLINE FLMUINT getSessionCookie( void)
	{
		return( m_uiSessionCookie);
	}
	
	FINLINE FLMUINT getContainerId( void)
	{
		return( m_uiContainer);
	}
	
	FINLINE FLMUINT getTransId( void)
	{
		return( m_uiTransId);
	}
	
	FINLINE FLMUINT getIteratorId( void) 
	{
		return( m_uiIteratorId);
	}
	
	FINLINE FLMUNICODE * getItemName( void)
	{
		return( m_puzItemName);
	}
	
	FINLINE FLMUNICODE * getFilePath( void)
	{
		return( m_puzFilePath);
	}
	
	FINLINE FLMUNICODE * getFilePath2( void)
	{
		return( m_puzFilePath2);
	}
	
	FINLINE FLMUNICODE * getFilePath3( void)
	{
		return( m_puzFilePath3);
	}
	
	FINLINE FLMUINT getItemId( void) 
	{
		return( m_uiItemId);
	}
	
	FINLINE FLMBOOL includesAsync( void)
	{
		return( m_bIncludesAsync);
	}
	
	FINLINE FLMBOOL getBoolean( void) 
	{
		return( m_bFlag);
	}
	
	FINLINE FLMUINT64 getNumber1( void)
	{
		return( m_ui64Number1);
	}
	
	FINLINE FLMUINT64 getNumber2( void)
	{
		return( m_ui64Number2);
	}
	
	FINLINE FLMUINT64 getNumber3( void)
	{
		return( m_ui64Number3);
	}
	
	FINLINE FLMUINT getAddress( void)
	{
		return( m_uiAddress);
	}
	
	FINLINE FLMINT64 getSignedValue( void)
	{
		return( m_i64SignedValue);
	}
	
	FINLINE FLMUINT getIndexId( void)
	{
		return( m_uiIndexId);
	}
	
	FINLINE FLMBYTE * getBlock( void)
	{
		return( m_pucBlock);
	}
	
	FINLINE FLMUINT getBlockSize( void)
	{
		return( m_uiBlockSize);
	}
	
	FINLINE FLMBYTE * getSerialNum( void)
	{
		return( m_pucSerialNum);
	}
	
	FINLINE FLMUINT getFlags( void) 
	{
		return( m_uiFlags);
	}
	
	FINLINE FLMUINT getFlaimVersion( void)
	{
		return( m_uiFlaimVersion);
	}

	FINLINE void setPool( 
		F_Pool *		pPool)
	{
		m_pPool = pPool;
	}
	
	FINLINE F_Pool * getPool( void)
	{
		return( m_pPool);
	}
	
	FINLINE void setFDB( 
		FDB *				pDb)
	{ 
		m_pDb = pDb;
	}
	
	FINLINE FDB * getFDB( void)
	{
		return( m_pDb);
	}
	
	FINLINE void setDIStream( 
		FCS_DIS * 		pDIStream)
	{ 
		m_pDIStream = pDIStream;
	}
	
	FINLINE void setDOStream( 
		FCS_DOS * 		pDOStream)
	{
		m_pDOStream = pDOStream;
	}
};

/****************************************************************************
Desc:
****************************************************************************/
class	FCL_WIRE : public FCS_WIRE
{
private:

	F_NameTable *		m_pNameTable;
	
public:

	FCL_WIRE( 
		CS_CONTEXT *		pCSContext = NULL,
		FDB * 				pDb = NULL);

	FINLINE virtual ~FCL_WIRE( void)
	{
	}

	FINLINE void reset( void)
	{
		resetCommon();
		m_pNameTable = NULL;
	}

	RCODE read( void);

	RCODE sendOp(
		FLMUINT			uiClass,
		FLMUINT			uiOp);

	RCODE doTransOp(
		FLMUINT			uiOp,
		FLMUINT			uiTransType,
		FLMUINT			uiFlags,
		FLMUINT			uiMaxLockWait,
		FLMBYTE *		pszHeader = NULL,
		FLMBOOL			bForceCheckpoint = FALSE);

	FINLINE void setNameTable( F_NameTable * pNameTable)
	{ 
		m_pNameTable = pNameTable;
	}

	FINLINE F_NameTable * getNameTable( void)
	{ 
		return( m_pNameTable);
	}

	void setContext( CS_CONTEXT *);
	
	FINLINE CS_CONTEXT * getContext( void)
	{ 
		return( m_pCSContext);
	}
};

#define FCS_BIOS_BLOCK_SIZE			8192
#define FCS_BIOS_EOM_EVENT				0x0001		// End-Of-Message event

class FCS_BIOS;

typedef struct FCSBIOSBLOCK
{
	FCSBIOSBLOCK *		pNextBlock;
	FLMUINT				uiCurrWriteOffset;
	FLMUINT				uiCurrReadOffset;
	FLMBYTE *			pucBlock;
}	FCSBIOSBLOCK;

typedef RCODE (* FCS_BIOS_EVENT_HOOK)(
	FCS_BIOS *	pStream,
	FLMUINT		uiEvent,
	void *		pvUserData);

/****************************************************************************
Desc:
****************************************************************************/
class FCS_FIS : public FCS_ISTM
{
private:

	IF_FileHdl *			m_pFileHdl;
	FLMBYTE *				m_pucBuffer;
	FLMBYTE *				m_pucBufPos;
	FLMUINT					m_uiFileOffset;
	FLMUINT					m_uiBlockSize;
	FLMUINT					m_uiBlockEnd;

	RCODE getNextPacket( void);

public:

	FCS_FIS( void);
	virtual ~FCS_FIS( void);

	RCODE setup(
		const char *	pszFilePath,
		FLMUINT			uiBlockSize);

	FLMBOOL isOpen( void);
	
	RCODE	close( void);
	
	RCODE flush( void);
	
	RCODE endMessage( void);

	RCODE read( 
		FLMBYTE *		pucData,
		FLMUINT 			uiLength,
		FLMUINT *		puiBytesRead);
};

/****************************************************************************
Desc:
****************************************************************************/
class	FCS_BIOS : public FCS_ISTM, public FCS_OSTM
{
public:

	FCS_BIOS( void);
	
	virtual ~FCS_BIOS();

	RCODE	reset( void);
	
	FLMUINT getAvailable( void);
	
	RCODE close( void);
	
	RCODE endMessage( void);
	
	RCODE write(
		FLMBYTE *		pucData,
		FLMUINT			uiLength);
		
	RCODE read( 
		FLMBYTE *		pucData,
		FLMUINT			uiLength,
		FLMUINT *		puiBytesRead);
		
	FLMBOOL isDataAvailable( void);

	FINLINE FLMBOOL isOpen( void)
	{
		return( TRUE);
	}

	FINLINE void setEventHook(
		FCS_BIOS_EVENT_HOOK	pEventHook,
		void *					pvUserData)
	{
		m_pEventHook = pEventHook;
		m_pvUserData = pvUserData;
	}

	FINLINE RCODE flush( void)
	{
		return( FERR_OK);
	}

private:

	FLMBOOL						m_bOpen;
	FLMBOOL						m_bMessageActive;
	FLMBOOL						m_bAcceptingData;
	FCSBIOSBLOCK *				m_pRootBlock;
	FCSBIOSBLOCK *				m_pCurrWriteBlock;
	FCSBIOSBLOCK *				m_pCurrReadBlock;
	FCS_BIOS_EVENT_HOOK 		m_pEventHook;
	void *						m_pvUserData;
	F_Pool						m_pool;
};

/****************************************************************************
Desc:
****************************************************************************/
class	FCS_BUFISTM : public FCS_ISTM
{
public:

	virtual ~FCS_BUFISTM()
	{
	}

	FINLINE FCS_BUFISTM(
		FLMBYTE *		pucBuf,
		FLMUINT			uiSize)
	{
		m_pucBuf = pucBuf;
		m_uiSize = uiSize;
		m_uiOffset = 0;
		m_bOpen = TRUE;
	}

	FINLINE FLMBOOL isOpen( void)
	{
		return( m_bOpen);
	}

	FINLINE RCODE close( void)
	{
		m_bOpen = FALSE;
		return( FERR_OK);
	}

	FINLINE RCODE flush( void)
	{
		m_uiOffset = m_uiSize;
		return( FERR_OK);
	}

	FINLINE RCODE endMessage( void)
	{
		m_uiOffset = m_uiSize;
		return( FERR_OK);
	}

	FINLINE RCODE read(
		FLMBYTE *	pucBuf,
		FLMUINT		uiLength,
		FLMUINT *	puiBytesRead)
	{
		FLMUINT		uiReadLen = f_min( uiLength, m_uiSize - m_uiOffset);
		RCODE			rc = FERR_OK;

		if( uiReadLen)
		{
			f_memcpy( pucBuf, &m_pucBuf[ m_uiOffset], uiReadLen);
			m_uiOffset += uiReadLen;
		}

		if( puiBytesRead)
		{
			*puiBytesRead = uiReadLen;
		}

		if( uiReadLen < uiLength)
		{
			rc = RC_SET( FERR_IO_END_OF_FILE);
			goto Exit;
		}

	Exit:

		return( rc);
	}

private:

	FLMUINT			m_uiSize;
	FLMUINT			m_uiOffset;
	FLMBYTE *		m_pucBuf;
	FLMBOOL			m_bOpen;
};

#include "fpackon.h"
// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

class FCS_TCP_SERVER;
class FCS_TCP_CLIENT;

RCODE fcsConvertUnicodeToNative(
	F_Pool *					pPool,
	const FLMUNICODE *	puzUnicode,
	char **					ppucNative);

RCODE	fcsConvertNativeToUnicode(
	F_Pool *					pPool,
	const char *			pucNative,
	FLMUNICODE **			ppuzUnicode);

RCODE	fcsBuildCheckpointInfo(
	CHECKPOINT_INFO *		pChkptInfo,
	F_Pool *					pPool,
	NODE **					ppTree);

RCODE	fcsExtractCheckpointInfo(
	NODE *					pTree,
	CHECKPOINT_INFO *		pChkptInfo);

RCODE	fcsBuildLockUser(
	F_LOCK_USER *		pLockUser,
	FLMBOOL				bList,
	F_Pool *				pPool,
	NODE **				ppTree);

RCODE	fcsExtractLockUser(
	NODE *				pTree,
	FLMBOOL				bExtractAsList,
	void *				pvLockUser);

void	fcsInitCreateOpts(
	CREATE_OPTS *		pCreateOptsRV);

RCODE fcsTranslateQFlmToQCSOp(
	QTYPES				eFlmOp,
	FLMUINT *			puiCSOp);

RCODE fcsTranslateQCSToQFlmOp(
	FLMUINT				uiCSOp,
	QTYPES *				peFlmOp);

RCODE fcsBuildIndexStatus(
	FINDEX_STATUS *	pIndexStatus,
	F_Pool *				pPool,
	NODE **				ppTree);

RCODE fcsExtractIndexStatus(
	NODE *				pTree,
	FINDEX_STATUS *	pIndexStatus);

RCODE fcsBuildMemInfo(
	FLM_MEM_INFO *		pMemInfo,
	F_Pool *				pPool,
	NODE **				ppTree);

RCODE fcsExtractMemInfo(
	NODE *				pTree,
	FLM_MEM_INFO *		pMemInfo);

RCODE fcsBuildThreadInfo(
	F_Pool *				pPool,
	NODE **				ppTree);

RCODE fcsExtractThreadInfo(
	NODE *				pTree,
	F_Pool *				pPool,
	F_THREAD_INFO **	ppThreadInfo,
	FLMUINT *			puiNumThreads);

RCODE fcsGetBlock(
	HFDB					hDb,
	FLMUINT				uiAddress,
	FLMUINT				uiMinTransId,
	FLMUINT *			puiCount,
	FLMUINT *			puiBlocksExamined,
	FLMUINT *			puiNextBlkAddr,
	FLMUINT				uiFlags,
	FLMBYTE *			pucBlock);

RCODE fcsCreateSerialNumber(
	void *				pCSContext,
	FLMBYTE *			pucSerialNum);

RCODE fcsSetBackupActiveFlag(
	HFDB					hDb,
	FLMBOOL				bBackupActive);

RCODE fcsDbTransCommitEx(
	HFDB					hDb,
	FLMBOOL				bForceCheckpoint,
	FLMBYTE *			pucLogHdr);

RCODE flmGenerateHexPacket(
	FLMBYTE *			pucData,
	FLMUINT				uiDataSize,
	FLMBYTE **			ppucPacket);

RCODE flmExtractHexPacketData(
	FLMBYTE *			pucPacket,
	FLMBYTE **			ppucData,
	FLMUINT *			puiDataSize);

void fcsDecodeHttpString(
	char *				pszSrc);

RCODE flmStreamEventDispatcher(
	FCS_BIOS *			pStream,
	FLMUINT				uiEvent,
	void *				pvUserData);

#include "fpackoff.h"

#endif	// #ifdef FCS_H
