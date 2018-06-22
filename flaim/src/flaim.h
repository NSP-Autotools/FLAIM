//-------------------------------------------------------------------------
// Desc:	Structures, classes, prototypes, and defines needed by an application
//			to use FLAIM functionality
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

/// \file

#ifndef FLAIM_H
#define FLAIM_H

	#include <flaimtk.h>

   // platform-specific API definitions for FLM* macros
   #if defined( FLM_WIN)
      #if defined( FLM_STATIC_LINK)
         #define FLMEXP
      #else
         #if defined( FLM_SOURCE)
            #define FLMEXP                __declspec(dllexport)
         #else
            #define FLMEXP                __declspec(dllimport)
         #endif
      #endif
		#define FLMAPI     						__stdcall
	#elif defined( FLM_NLM)
      #define FLMEXP
		#define FLMAPI     						__stdcall
	#elif defined( FLM_UNIX)
      #define FLMEXP
		#define FLMAPI
	#else
		#error Platform not supported
   #endif
	#define FLMXPC							      extern "C" FLMEXP

	#ifdef FLM_PACK_STRUCTS
		#pragma pack(push, 1)
	#endif

	/// \defgroup dbsystem FLAIM System Functions

		/// \defgroup startupshutdown FLAIM System Startup/Shutdown
		/// \ingroup dbsystem

		/// \defgroup systemconfiguration FLAIM System Configuration/Information
		/// \ingroup dbsystem

		/// \defgroup cacheconfiguration Cache Configuration Functions
		/// \ingroup dbsystem

		/// \defgroup stats Statistics Collection Functions
		/// \ingroup dbsystem

	/// \defgroup database Database Functions

		/// \defgroup dbcreateopen Database Create, Open, Close
		/// \ingroup database

		/// \defgroup trans Transaction Functions
		/// \ingroup database

		/// \defgroup update Database Update Functions
		/// \ingroup database

		/// \defgroup retrieval Record and Key Retrieval Functions
		/// \ingroup database

		/// \defgroup dbdict Database Dictionary Functions
		/// \ingroup database

		/// \defgroup indexing Index Management Functions
		/// \ingroup database

		/// \defgroup dbconfig Database Configuration Functions
		/// \ingroup database

		/// \defgroup dbbackup Database Backup/Restore
		/// \ingroup database

		/// \defgroup dbmaint Database Maintenance Functions
		/// \ingroup database
		
		/// \defgroup dbcopy Database Copy, Rename, Delete Functions
		/// \ingroup database

		/// \defgroup encryption Database Encryption Key Management Functions
		/// \ingroup database

	/// \defgroup query Query Functions

		/// \defgroup queryobj Query Object Creation/Initialization/Deletion
		/// \ingroup query

		/// \defgroup querydef Query Criteria Definition Functions
		/// \ingroup query

		/// \defgroup queryret Query Result Set Retrieval Functions
		/// \ingroup query

		/// \defgroup queryconfig Query Configuration
		/// \ingroup query

		/// \defgroup querycomp Comparison Functions
		/// \ingroup query

	/// \defgroup misc Miscellaneous Functions

		/// \defgroup errhandling Error Handling Functions
		/// \ingroup misc

		/// \defgroup storageconversion Storage Format Conversion Functions
		/// \ingroup misc

		/// \defgroup language Language To String Conversion Functions
		/// \ingroup misc

		/// \defgroup stringcompare String Comparison Functions
		/// \ingroup misc

		/// \defgroup event Event Registration/Deregistration Functions
		/// \ingroup misc

		/// \defgroup pool Pool Memory Functions
		/// \ingroup misc

		/// \defgroup memoryalloc Memory Functions
		/// \ingroup misc

	/****************************************************************************
	Desc: General errors
	****************************************************************************/
	/// \addtogroup retcodes
	/// @{
		
	#define FERR_OK								NE_FLM_OK								///< 0 - Operation succeeded
	
	#define FERR_BOF_HIT							NE_FLM_BOF_HIT							///< 0xC001 - Beginning of file or set hit.
	#define FERR_EOF_HIT							NE_FLM_EOF_HIT							///< 0xC002 - End of file or set hit.
	#define FERR_END								0xC003									///< 0xC003 - End of GEDCOM file - this is an internal error.
	#define FERR_EXISTS							NE_FLM_EXISTS							///< 0xC004 - Record already exists.
	#define FERR_FAILURE							NE_FLM_FAILURE							///< 0xC005 - Internal failure.
	#define FERR_NOT_FOUND						NE_FLM_NOT_FOUND						///< 0xC006 - A record, key, or key reference was not found.
	#define FERR_BAD_DICT_ID					0xC007									///< 0xC007 - Invalid dictionary record number -- outside unreserved range.
	#define FERR_BAD_CONTAINER					0xC008									///< 0xC008 - Invalid container number.
	#define FERR_NO_ROOT_BLOCK					0xC009									///< 0xC009 - LFILE does not have a root block - always handled internally - never returned to application.
	#define FERR_BAD_DRN							0xC00A									///< 0xC00A - Cannot pass a zero DRN into modify or delete or 0xFFFFFFFF into add.
	#define FERR_BAD_FIELD_NUM					0xC00B									///< 0xC00B - Bad field number in record being added.
	#define FERR_BAD_FIELD_TYPE				0xC00C									///< 0xC00C - Bad field type in record being added.
	#define FERR_BAD_HDL							0xC00D									///< 0xC00D - Request contained bad db handle.
	#define FERR_BAD_IX							0xC00E									///< 0xC00E - Invalid index number.
	#define FERR_BACKUP_ACTIVE					0xC00F									///< 0xC00F - Operation could not be completed - a backup is being performed.
	#define FERR_SERIAL_NUM_MISMATCH			0xC010									///< 0xC010 - Comparison of serial numbers failed.
	#define FERR_BAD_RFL_DB_SERIAL_NUM		0xC011									///< 0xC011 - Bad database serial number in RFL file header.
	#define FERR_BTREE_ERROR					NE_FLM_BTREE_ERROR					///< 0xC012 - A corruption was found in an index or container b-tree.
	#define FERR_BTREE_FULL						NE_FLM_BTREE_FULL						///< 0xC013 - An index or container b-tree is full.
	#define FERR_BAD_RFL_FILE_NUMBER			0xC014									///< 0xC014 - Bad RFL file number in RFL file header.
	#define FERR_CANNOT_DEL_ITEM				0xC015									///< 0xC015 - Cannot delete field definitions.
	#define FERR_CANNOT_MOD_FIELD_TYPE		0xC016									///< 0xC016 - Cannot modify a field's type.
	#define FERR_64BIT_NUMS_NOT_SUPPORTED	0xC017									///< 0xC017 - 64 bit numbers not supported for databases whose revision is less than 462
	#define FERR_CONV_BAD_DEST_TYPE			0xC018									///< 0xC018 - Bad destination type specified for conversion.
	#define FERR_CONV_BAD_DIGIT				0xC019									///< 0xC019 - Non-numeric digit found in text to numeric conversion.
	#define FERR_CONV_BAD_SRC_TYPE			0xC01A									///< 0xC01A - Bad source type specified for conversion.
	#define FERR_RFL_FILE_NOT_FOUND			0xC01B									///< 0xC01B - Could not open an RFL file.
	#define FERR_CONV_DEST_OVERFLOW			NE_FLM_CONV_DEST_OVERFLOW			///< 0xC01C - Destination buffer not large enough to hold converted data.
	#define FERR_CONV_ILLEGAL					NE_FLM_CONV_ILLEGAL					///< 0xC01D - Illegal conversion -- not supported.
	#define FERR_CONV_NULL_SRC					0xC01E									///< 0xC01E - Source cannot be a NULL pointer in conversion.
	#define FERR_CONV_NULL_DEST				0xC01F									///< 0xC01F - Destination cannot be a NULL pointer in conversion.
	#define FERR_CONV_NUM_OVERFLOW			NE_FLM_CONV_NUM_OVERFLOW			///< 0xC020 - Numeric overflow (GT upper bound) converting to numeric type.
	#define FERR_CONV_NUM_UNDERFLOW			0xC021									///< 0xC021 - Numeric underflow (LT lower bound) converting to numeric type.
	#define FERR_DATA_ERROR						NE_FLM_DATA_ERROR						///< 0xC022 - Database corruption found.
	#define FERR_NOT_USED_C023					0xC023									///< 0xC023 - Not used
	#define FERR_DD_ERROR						0xC024									///< 0xC024 - Corruption found in logical file block chain.
	#define FERR_INVALID_FILE_SEQUENCE		0xC025									///< 0xC025 - Incremental backup file number provided during a restore is invalid.
	#define FERR_ILLEGAL_OP						NE_FLM_ILLEGAL_OP						///< 0xC026 - Illegal operation for database.
	#define FERR_DUPLICATE_DICT_REC			0xC027									///< 0xC027 - Duplicate dictionary record found.
	#define FERR_CANNOT_CONVERT				0xC028									///< 0xC028 - Condition occurred which prevents database conversion.
	#define FERR_UNSUPPORTED_VERSION			0xC029									///< 0xC029 - Database version is not supported.
	#define FERR_FILE_ER							0xC02A									///< 0xC02A - File error in a GEDCOM routine.
	#define FERR_BAD_FIELD_LEVEL				0xC02B									///< 0xC02B - Invalid field level.
	#define FERR_GED_BAD_RECID					0xC02C									///< 0xC02C - Bad record ID syntax.
	#define FERR_GED_BAD_VALUE					0xC02D									///< 0xC02D - Bad or ambiguous/extra value in GEDCOM.
	#define FERR_GED_MAXLVLNUM					0xC02E									///< 0xC02E - Exceeded GED_MAXLVLNUM in gedcom routines.
	#define FERR_GED_SKIP_LEVEL				0xC02F									///< 0xC02F - Bad GEDCOM tree structure -- level skipped.
	#define FERR_ILLEGAL_TRANS					0xC030									///< 0xC030 - Attempt to start an illegal type of transaction.
	#define FERR_ILLEGAL_TRANS_OP				0xC031									///< 0xC031 - Illegal operation for transaction type.
	#define FERR_INCOMPLETE_LOG				0xC032									///< 0xC032 - Incomplete log record encountered during recovery.
	#define FERR_INVALID_BLOCK_LENGTH		0xC033									///< 0xC033 - Invalid block length.
	#define FERR_INVALID_TAG					0xC034									///< 0xC034 - Invalid tag name.
	#define FERR_KEY_NOT_FOUND					0xC035									///< 0xC035 - A key or reference is not found -- modify/delete error.
	#define FERR_VALUE_TOO_LARGE				0xC036									///< 0xC036 - Value too large.
	#define FERR_MEM								NE_FLM_MEM								///< 0xC037 - Memory allocation error.
	#define FERR_BAD_RFL_SERIAL_NUM			0xC038									///< 0xC038 - Bad serial number in RFL file header.
	#define FERR_NOT_USED_C039					0xC039									///< 0xC039 - Not used
	#define FERR_NEWER_FLAIM					0xC03A									///< 0xC03A - Database version newer than this code base will support, must use newer version of code.
	#define FERR_CANNOT_MOD_FIELD_STATE		0xC03B									///< 0xC03B - Attempted to change a field state illegally.
	#define FERR_NO_MORE_DRNS					0xC03C									///< 0xC03C - The highest DRN number has already been used in an add.
	#define FERR_NO_TRANS_ACTIVE				0xC03D									///< 0xC03D - Attempted to updated database outside transaction.
	#define FERR_NOT_UNIQUE						NE_FLM_NOT_UNIQUE						///< 0xC03E - Found duplicate key for unique index.
	#define FERR_NOT_FLAIM						0xC03F									///< 0xC03F - File is not a FLAIM database.
	#define FERR_NULL_RECORD					0xC040									///< 0xC040 - NULL record cannot be passed to add or modify.
	#define FERR_NO_HTTP_STACK					0xC041									///< 0xC041 - No http stack was loaded.
	#define FERR_OLD_VIEW						0xC042									///< 0xC042 - While reading was unable to get previous version of block or record.
	#define FERR_PCODE_ERROR					0xC043									///< 0xC043 - Corruption found in dictionary.
	#define FERR_PERMISSION						0xC044									///< 0xC044 - Invalid permission for file operation.
	#define FERR_SYNTAX							NE_FLM_SYNTAX							///< 0xC045 - Dictionary record has improper syntax, or syntax error in query criteria.
	#define FERR_CALLBACK_FAILURE				0xC046									///< 0xC046 - Callback failure.
	#define FERR_TRANS_ACTIVE					0xC047									///< 0xC047 - Attempted to close database while transaction was active.
	#define FERR_RFL_TRANS_GAP					0xC048									///< 0xC048 - A gap was found in the transaction sequence in the RFL.
	#define FERR_BAD_COLLATED_KEY				0xC049									///< 0xC049 - Something in collated key is bad.
	#define FERR_UNSUPPORTED_FEATURE			0xC04A									///< 0xC04A - Attempting a feature that is not supported for the database version.
	#define FERR_MUST_DELETE_INDEXES			0xC04B									///< 0xC04B - Attempting to delete a container that has indexes defined for it -- indexes must be deleted first.
	#define FERR_RFL_INCOMPLETE				0xC04C									///< 0xC04C - RFL file is incomplete.
	#define FERR_CANNOT_RESTORE_RFL_FILES	0xC04D									///< 0xC04D - Cannot restore RFL files - not using multiple RFL files.
	#define FERR_INCONSISTENT_BACKUP			0xC04E									///< 0xC04E - A problem (corruption, etc) was detected in a backup set.
	#define FERR_BLOCK_CHECKSUM				0xC04F									///< 0xC04F - Block checksum error.
	#define FERR_ABORT_TRANS					0xC050									///< 0xC050 - Attempted operation after a critical error - should abort transaction.
	#define FERR_NOT_RFL							0xC051									///< 0xC051 - Attempted to open file which was not an RFL file.
	#define FERR_BAD_RFL_PACKET				0xC052									///< 0xC052 - RFL packet was bad.
	#define FERR_DATA_PATH_MISMATCH			0xC053									///< 0xC053 - Bad data path specified to open database.
	#define FERR_HTTP_REGISTER_FAILURE		0xC054									///< 0xC054 - Call to FlmConfig() with FLM_HTTP_REGISTER_URL option failed.
	#define FERR_HTTP_DEREG_FAILURE			0xC055									///< 0xC055 - Call to FlmConfig() with FLM_HTTP_DEREGISTER_URL option failed.
	#define FERR_IX_FAILURE						0xC056									///< 0xC056 - Indexing process failed, non-unique data was found when a unique index was being created.
	#define FERR_HTTP_SYMS_EXIST				0xC057									///< 0xC057 - Tried to import new http related symbols before unimporting the old ones.
	#define FERR_NOT_USED_C058					0xC058									///< 0xC058 - Not used
	#define FERR_FILE_EXISTS					0xC059									///< 0xC059 - Attempt to create a database, but the database already exists.
	#define FERR_SYM_RESOLVE_FAIL				0xC05A									///< 0xC05A - Could not resolve a symbol needed to run.
	#define FERR_BAD_SERVER_CONNECTION		0xC05B									///< 0xC05B - Connection to FLAIM server is bad.
	#define FERR_CLOSING_DATABASE				0xC05C									///< 0xC05C - Database is being closed due to a critical error.
	#define FERR_INVALID_CRC					0xC05D									///< 0xC05D - CRC could not be verified.
	#define FERR_KEY_OVERFLOW					0xC05E									///< 0xC05E - Key generated by the record causes the maximum key size to be exceeded.
	#define FERR_NOT_IMPLEMENTED				NE_FLM_NOT_IMPLEMENTED				///< 0xC05F - Functionality not implemented.
	#define FERR_MUTEX_OPERATION_FAILED		0xC060									///< 0xC060 - Mutex operation failed.
	#define FERR_MUTEX_UNABLE_TO_LOCK		0xC061									///< 0xC061 - Unable to get the mutex lock.
	#define FERR_SEM_OPERATION_FAILED		0xC062									///< 0xC062 - Semaphore operation failed.
	#define FERR_SEM_UNABLE_TO_LOCK			0xC063									///< 0xC063 - Unable to get the semaphore lock.
	#define FERR_NOT_USED_C064					0xC064									///< 0xC064 - Not used
	#define FERR_NOT_USED_C065					0xC065									///< 0xC065 - Not used
	#define FERR_NOT_USED_C066					0xC066									///< 0xC066 - Not used
	#define FERR_NOT_USED_C067					0xC067									///< 0xC067 - Not used
	#define FERR_NOT_USED_C068					0xC068									///< 0xC068 - Not used
	#define FERR_BAD_REFERENCE					0xC069									///< 0xC069 - Bad reference in the dictionary.
	#define FERR_NOT_USED_C06A					0xC06A									///< 0xC06A - Not used
	#define FERR_NOT_USED_C06B					0xC06B									///< 0xC06B - Not used
	#define FERR_NOT_USED_C06C					0xC06C									///< 0xC06C - Not used
	#define FERR_NOT_USED_C06D					0xC06D									///< 0xC06D - Not used
	#define FERR_NOT_USED_C06E					0xC06E									///< 0xC06E - Not used
	#define FERR_NOT_USED_C06F					0xC06F									///< 0xC06F - Not used
	#define FERR_UNALLOWED_UPGRADE			0xC070									///< 0xC070 - FlmDbUpgrade cannot upgrade the database.
	#define FERR_NOT_USED_C071					0xC071									///< 0xC071 - Not used
	#define FERR_NOT_USED_C072					0xC072									///< 0xC072 - Not used
	#define FERR_NOT_USED_C073					0xC073									///< 0xC073 - Not used
	#define FERR_ID_RESERVED					0xC074									///< 0xC074 - Attempted to use a dictionary ID that has been reserved.
	#define FERR_CANNOT_RESERVE_ID			0xC075									///< 0xC075 - Attempted to reserve a dictionary ID that has been used.
	#define FERR_DUPLICATE_DICT_NAME			0xC076									///< 0xC076 - Dictionary record with duplicate name found.
	#define FERR_CANNOT_RESERVE_NAME			0xC077									///< 0xC077 - Attempted to reserve a dictionary name that is in use.
	#define FERR_BAD_DICT_DRN					0xC078									///< 0xC078 - Attempted to add, modify, or delete a dictionary DRN >= FLM_RESERVED_TAG_NUMS.
	#define FERR_CANNOT_MOD_DICT_REC_TYPE	0xC079									///< 0xC079 - Cannot modify a dictionary item into another type of item, must delete then add.
	#define FERR_PURGED_FLD_FOUND				0xC07A									///< 0xC07A - Record contained a field whose field definition has been marked as purged.
	#define FERR_DUPLICATE_INDEX				0xC07B									///< 0xC07B - Duplicate index.
	#define FERR_TOO_MANY_OPEN_DBS			0xC07C									///< 0xC07C - Too many open databases.
	#define FERR_ACCESS_DENIED					0xC07D									///< 0xC07D - Cannot access database.
	#define FERR_NOT_USED_C07E					0xC07E									///< 0xC07E - Not used
	#define FERR_CACHE_ERROR					0xC07F									///< 0xC07F - Cache block is corrupt.
	#define FERR_NOT_USED_C080					0xC080									///< 0xC080 - Not used
	#define FERR_BLOB_MISSING_FILE			0xC081									///< 0xC081 - Missing BLOB file on add/modify.
	#define FERR_NO_REC_FOR_KEY				0xC082									///< 0xC082 - Record pointed to by an index key is missing.
	#define FERR_DB_FULL							0xC083									///< 0xC083 - Database is full, cannot create more blocks.
	#define FERR_TIMEOUT							0xC084									///< 0xC084 - Operation timed out (usually a query operation).
	#define FERR_CURSOR_SYNTAX					0xC085									///< 0xC085 - Query criteria had improper syntax.
	#define FERR_THREAD_ERR						0xC086									///< 0xC086 - Thread error.
	#define FERR_UNIMPORT_SYMBOL				0xC087									///< 0xC087 - Failed to unimport a public symbol.
	#define FERR_EMPTY_QUERY					0xC088									///< 0xC088 - Warning: Query has no results.
	#define FERR_INDEX_OFFLINE					0xC089									///< 0xC089 - Warning: Index is offline and being rebuilt.
	#define FERR_TRUNCATED_KEY					0xC08A									///< 0xC08A - Warning: Can't evaluate truncated key against selection criteria.
	#define FERR_INVALID_PARM					NE_FLM_INVALID_PARM					///< 0xC08B - Invalid parameter.
	#define FERR_USER_ABORT						0xC08C									///< 0xC08C - User or application aborted the operation.
	#define FERR_RFL_DEVICE_FULL				0xC08D									///< 0xC08D - No space on RFL device for logging.
	#define FERR_MUST_WAIT_CHECKPOINT		0xC08E									///< 0xC08E - Must wait for a checkpoint before starting transaction - due to disk problems - usually in RFL volume.
	#define FERR_NAMED_SEMAPHORE_ERR			0xC08F									///< 0xC08F - Error occurred while accessing a named semaphore.
	#define FERR_LOAD_LIBRARY					0xC090									///< 0xC090 - Failed to load a shared library module.
	#define FERR_UNLOAD_LIBRARY				0xC091									///< 0xC091 - Failed to unload a shared library module.
	#define FERR_IMPORT_SYMBOL					0xC092									///< 0xC092 - Failed to import a symbol from a shared library module.
	#define FERR_BLOCK_FULL						0xC093									///< 0xC093 - Destination block for insert is full.
	#define FERR_BAD_BASE64_ENCODING			0xC094									///< 0xC094 - Could not perform base 64 encoding.
	#define FERR_MISSING_FIELD_TYPE			0xC095									///< 0xC095 - Field type not specified in field definition record.
	#define FERR_BAD_DATA_LENGTH				0xC096									///< 0xC096 - Invalid field data length.

		/****************************************************************************
								IO Errors
		****************************************************************************/

	#define FERR_IO_ACCESS_DENIED				NE_FLM_IO_ACCESS_DENIED				///< 0xC201 - Access denied. Caller is not allowed access to a file.
	#define FERR_IO_BAD_FILE_HANDLE			NE_FLM_IO_BAD_FILE_HANDLE			///< 0xC202 - Bad file handle.
	#define FERR_IO_COPY_ERR					NE_FLM_IO_COPY_ERR					///< 0xC203 - Copy error.
	#define FERR_IO_DISK_FULL					NE_FLM_IO_DISK_FULL					///< 0xC204 - Disk full.
	#define FERR_IO_END_OF_FILE				NE_FLM_IO_END_OF_FILE				///< 0xC205 - End of file.
	#define FERR_IO_OPEN_ERR					NE_FLM_IO_OPEN_ERR					///< 0xC206 - Error opening file.
	#define FERR_IO_SEEK_ERR					NE_FLM_IO_SEEK_ERR					///< 0xC207 - File seek error.
	#define FERR_IO_DIRECTORY_ERR				NE_FLM_IO_DIRECTORY_ERR				///< 0xC208 - Error occurred while accessing or deleting a directory.
	#define FERR_IO_PATH_NOT_FOUND			NE_FLM_IO_PATH_NOT_FOUND			///< 0xC209 - Path not found.
	#define FERR_IO_TOO_MANY_OPEN_FILES		NE_FLM_IO_TOO_MANY_OPEN_FILES		///< 0xC20A - Too many files open.
	#define FERR_IO_PATH_TOO_LONG				NE_FLM_IO_PATH_TOO_LONG				///< 0xC20B - Path too long.
	#define FERR_IO_NO_MORE_FILES				NE_FLM_IO_NO_MORE_FILES				///< 0xC20C - No more files in directory.
	#define FERR_DELETING_FILE					NE_FLM_IO_DELETING_FILE				///< 0xC20D - Had error deleting a file.
	#define FERR_IO_FILE_LOCK_ERR				NE_FLM_IO_FILE_LOCK_ERR				///< 0xC20E - File lock error.
	#define FERR_IO_FILE_UNLOCK_ERR			NE_FLM_IO_FILE_UNLOCK_ERR			///< 0xC20F - File unlock error.
	#define FERR_IO_PATH_CREATE_FAILURE		NE_FLM_IO_PATH_CREATE_FAILURE		///< 0xC210 - Path create failed.
	#define FERR_IO_RENAME_FAILURE			NE_FLM_IO_RENAME_FAILURE			///< 0xC211 - File rename failed.
	#define FERR_IO_INVALID_PASSWORD			NE_FLM_IO_INVALID_PASSWORD			///< 0xC212 - Invalid file password.
	#define FERR_SETTING_UP_FOR_READ			NE_FLM_SETTING_UP_FOR_READ			///< 0xC213 - Had error setting up to do a read.
	#define FERR_SETTING_UP_FOR_WRITE		NE_FLM_SETTING_UP_FOR_WRITE		///< 0xC214 - Had error setting up to do a write.
	#define FERR_IO_AT_PATH_ROOT				NE_FLM_IO_CANNOT_REDUCE_PATH		///< 0xC215 - Currently positioned at the path root level.
	#define FERR_INITIALIZING_IO_SYSTEM		NE_FLM_INITIALIZING_IO_SYSTEM		///< 0xC216 - Had error initializing the file system.
	#define FERR_FLUSHING_FILE					NE_FLM_FLUSHING_FILE					///< 0xC217 - Had error flushing a file.
	#define FERR_IO_INVALID_PATH				NE_FLM_IO_INVALID_FILENAME			///< 0xC218 - Invalid path.
	#define FERR_IO_CONNECT_ERROR				NE_FLM_IO_CONNECT_ERROR				///< 0xC219 - Failed to connect to a remote network resource.
	#define FERR_OPENING_FILE					NE_FLM_OPENING_FILE					///< 0xC21A - Had error opening a file.
	#define FERR_DIRECT_OPENING_FILE			NE_FLM_DIRECT_OPENING_FILE			///< 0xC21B - Had error opening a file for direct I/O.
	#define FERR_CREATING_FILE					NE_FLM_CREATING_FILE					///< 0xC21C - Had error creating a file.
	#define FERR_DIRECT_CREATING_FILE		NE_FLM_DIRECT_CREATING_FILE		///< 0xC21D - Had error creating a file for direct I/O.
	#define FERR_READING_FILE					NE_FLM_READING_FILE					///< 0xC21E - Had error reading a file.
	#define FERR_DIRECT_READING_FILE			NE_FLM_DIRECT_READING_FILE			///< 0xC21F - Had error reading a file using direct I/O.
	#define FERR_WRITING_FILE					NE_FLM_WRITING_FILE					///< 0xC220 - Had error writing to a file.
	#define FERR_DIRECT_WRITING_FILE			NE_FLM_DIRECT_WRITING_FILE			///< 0xC221 - Had error writing to a file using direct I/O.
	#define FERR_POSITIONING_IN_FILE			NE_FLM_POSITIONING_IN_FILE			///< 0xC222 - Had error positioning within a file.
	#define FERR_GETTING_FILE_SIZE			NE_FLM_GETTING_FILE_SIZE			///< 0xC223 - Had error getting file size.
	#define FERR_TRUNCATING_FILE				NE_FLM_TRUNCATING_FILE				///< 0xC224 - Had error truncating a file.
	#define FERR_PARSING_FILE_NAME			NE_FLM_PARSING_FILE_NAME			///< 0xC225 - Had error parsing a file name.
	#define FERR_CLOSING_FILE					NE_FLM_CLOSING_FILE					///< 0xC226 - Had error closing a file.
	#define FERR_GETTING_FILE_INFO			NE_FLM_GETTING_FILE_INFO			///< 0xC227 - Had error getting file information.
	#define FERR_EXPANDING_FILE				NE_FLM_EXPANDING_FILE				///< 0xC228 - Had error expanding a file (using direct I/O).
	#define FERR_GETTING_FREE_BLOCKS			NE_FLM_GETTING_FREE_BLOCKS			///< 0xC229 - Had error getting free blocks from file system.
	#define FERR_CHECKING_FILE_EXISTENCE	NE_FLM_CHECKING_FILE_EXISTENCE	///< 0xC22A - Had error checking if a file exists.
	#define FERR_RENAMING_FILE					NE_FLM_RENAMING_FILE					///< 0xC22B - Had error renaming a file.
	#define FERR_SETTING_FILE_INFO			NE_FLM_SETTING_FILE_INFO			///< 0xC22C - Had error setting file information.

		/****************************************************************************
								Encryption / Decryption Errors
		****************************************************************************/
	#define FERR_NICI_CONTEXT					0xC301									///< 0xC301 - Failed to obtain a NICI context.
	#define FERR_NICI_FIND_INIT				0xC302									///< 0xC302 - CCS_FindInit failed.
	#define FERR_NICI_FIND_OBJECT				0xC303									///< 0xC303 - CCS_FindObject failed.
	#define FERR_NICI_WRAPKEY_NOT_FOUND		0xC304									///< 0xC304 - Could not locate a wrapping key.
	#define FERR_NICI_ATTRIBUTE_VALUE		0xC305									///< 0xC305 - CCS_AttributeValue failed.
	#define FERR_NICI_BAD_ATTRIBUTE			0xC306									///< 0xC306 - Invalid attribute.
	#define FERR_NICI_BAD_RANDOM				0xC307									///< 0xC307 - CCS_GetRandom failed.
	#define FERR_NOT_USED_C308					0xC308									///< 0xC308 - Not used
	#define FERR_NICI_WRAPKEY_FAILED			0xC309									///< 0xC309 - CCS_WrapKey failed.
	#define FERR_NICI_GENKEY_FAILED			0xC30A									///< 0xC30A - CCS_GenerateKey failed.
	#define FERR_REQUIRE_PASSWD				0xC30B									///< 0xC30B - Password required to unwrap key.
	#define FERR_NICI_SHROUDKEY_FAILED		0xC30C									///< 0xC30C - CCS_pbeShroudPrivateKey failed.
	#define FERR_NICI_UNSHROUDKEY_FAILED	0xC30D									///< 0xC30D - CCS_pbdUnshroudPrivateKey failed.
	#define FERR_NICI_UNWRAPKEY_FAILED		0xC30E									///< 0xC30E - CCS_UnrapKey failed.
	#define FERR_NICI_ENC_INIT_FAILED		0xC30F									///< 0xC30F - CCS_DataEncryptInit failed.
	#define FERR_NICI_ENCRYPT_FAILED			0xC310									///< 0xC310 - CCS_DataEncrypt failed.
	#define FERR_NICI_DECRYPT_INIT_FAILED	0xC311									///< 0xC311 - CCS_DataDecryptInit failed.
	#define FERR_NICI_DECRYPT_FAILED			0xC312									///< 0xC312 - CCS_DataDecrypt failed.
	#define FERR_NICI_INIT_FAILED				0xC313									///< 0xC313 - CCS_Init failed.
	#define FERR_NICI_KEY_NOT_FOUND			0xC314									///< 0xC314 - Could not locate encryption/decryption key.
	#define FERR_NICI_INVALID_ALGORITHM		0xC315									///< 0xC315 - Unsupported NICI ecncryption algorithm.
	#define FERR_FLD_NOT_ENCRYPTED			0xC316									///< 0xC316 - Field is not encrypted.
	#define FERR_CANNOT_SET_KEY				0xC317									///< 0xC317 - Attempted to set an encryption key for new encryption definition record.
	#define FERR_MISSING_ENC_TYPE				0xC318									///< 0xC318 - Encryption type not specified in encryption definition record.
	#define FERR_CANNOT_MOD_ENC_TYPE			0xC319									///< 0xC319 - Attempting to change the encryption type in encryption definition record.
	#define FERR_MISSING_ENC_KEY				0xC31A									///< 0xC31A - Encryption key must be present in modified encryption definition record.
	#define FERR_CANNOT_CHANGE_KEY			0xC31B									///< 0xC31B - Attempt to modify the encryption key in an encryption definition record.
	#define FERR_BAD_ENC_KEY					0xC31C									///< 0xC31C - Bad encryption key.
	#define FERR_CANNOT_MOD_ENC_STATE		0xC31D									///< 0xC31D - Illegal state change for an encryption definition record.
	#define FERR_DATA_SIZE_MISMATCH			0xC31E									///< 0xC31E - Calculated encrypted data length does not match the length returned from encryption/decryption routines.
	#define FERR_ENCRYPTION_UNAVAILABLE		0xC31F									///< 0xC31F - Encryption capabilities are not available for encrypting/decrypting data in database.
	#define FERR_PURGED_ENCDEF_FOUND			0xC320									///< 0xC320 - Cannot use encryption ID for encryption of data - encryption definition record is marked as purged.
	#define FERR_FLD_NOT_DECRYPTED			0xC321									///< 0xC321 - Attempting to access data from a field that is encrypted, field could not be decrypted for some reason - probably because encryption/decryption capabilities are not available.
	#define FERR_BAD_ENCDEF_ID					0xC322									///< 0xC322 - Encryption ID is invalid - not defined in dictionary.
	#define FERR_PBE_ENCRYPT_FAILED			0xC323									///< 0xC323 - Call to NICI function CCS_pbeEncrypt failed.
	#define FERR_DIGEST_FAILED					0xC324									///< 0xC324 - Call to NICI function CCS_Digest failed.
	#define FERR_DIGEST_INIT_FAILED			0xC325									///< 0xC325 - Call to NICI function CCS_DigestInit failed.
	#define FERR_EXTRACT_KEY_FAILED			0xC326									///< 0xC326 - Call to NICI function CCS_ExtractKey failed.
	#define FERR_INJECT_KEY_FAILED			0xC327									///< 0xC327 - Call to NICI function CCS_InjectKey failed.
	#define FERR_PBE_DECRYPT_FAILED			0xC328									///< 0xC328 - Call to NICI function CCS_pbeDecrypt failed.
	#define FERR_PASSWD_INVALID				0xC329									///< 0xC329 - Invalid password passed, database could not be opened.

	#define FERR_BT_END_OF_DATA				0xFFFF									///< 0xFFFF	- Used internally

	/// @}
	
	/***************************************************************************
										Forward Declarations
	***************************************************************************/

	class FlmRecord;
	class FlmRecordSet;
	class F_Restore;

	/***************************************************************************
	                              FLAIM Types
	***************************************************************************/

	typedef void *					HFDB;			///< Database handle.
	typedef void *					HFCURSOR;	///< Query object handle.
	typedef void *					HFBLOB;		///< BLOB handle.
	typedef void *					HFBACKUP;	///< Backup object handle.
	
	#define HFDB_NULL				NULL
	#define HFCURSOR_NULL		NULL
	#define HFBLOB_NULL			NULL
	#define HFBACKUP_NULL		NULL

	/// Database create options.\ This structure is passed to FlmDbCreate()
	/// to specify create options for a new database.
	typedef struct
	{
		FLMUINT		uiBlockSize;					///< Block size for the database.
	#define DEFAULT_BLKSIZ								4096

		FLMUINT		uiVersionNum;					///< Database version number.
	#define FLM_FILE_FORMAT_VER_3_0					301
	#define FLM_FILE_FORMAT_VER_3_02					302
	#define FLM_FILE_FORMAT_VER_3_10					310
	#define FLM_FILE_FORMAT_VER_4_0					400
	#define FLM_FILE_FORMAT_VER_4_3					430
	#define FLM_FILE_FORMAT_VER_4_31					431	// Added last committed trans ID to the log header
	#define FLM_FILE_FORMAT_VER_4_50					450	// Added ability to create cross-container indexes.
	#define FLM_FILE_FORMAT_VER_4_51					451	// Added ability to permanently suspend indexes
	#define FLM_FILE_FORMAT_VER_4_52					452	// Added ability to delete indexes in the background
	#define FLM_FILE_FORMAT_VER_4_60					460	// Added support for encrypted attributes
	#define FLM_FILE_FORMAT_VER_4_61					461	// Added support for RFL disk usage limits, large field values, and async I/O on Linux and Solaris
	#define FLM_FILE_FORMAT_VER_4_62					462	// Added support for 64 bit numbers
	#define FLM_CUR_FILE_FORMAT_VER_NUM				FLM_FILE_FORMAT_VER_4_62
	#define FLM_CUR_FILE_FORMAT_VER_STR				"4.62"

		FLMUINT		uiMinRflFileSize;				///< Minimum bytes per RFL file.
	#define DEFAULT_MIN_RFL_FILE_SIZE	((FLMUINT)100 * (FLMUINT)1024 * (FLMUINT)1024)
		FLMUINT		uiMaxRflFileSize;				///< Maximum bytes per RFL file.
	#define DEFAULT_MAX_RFL_FILE_SIZE	FLM_MAXIMUM_FILE_SIZE
		FLMBOOL		bKeepRflFiles;					///< Keep RFL files?
	#define DEFAULT_KEEP_RFL_FILES_FLAG	FALSE
		FLMBOOL		bLogAbortedTransToRfl;		///< Log aborted transactions to RFL?
	#define DEFAULT_LOG_ABORTED_TRANS_FLAG	FALSE

		FLMUINT		uiDefaultLanguage;			///< Default language for the database.
		FLMUINT		uiAppMajorVer;					///< The application's major version number.
		FLMUINT		uiAppMinorVer;					///< The application's minor version number
	} CREATE_OPTS;

	/****************************************************************************
								Name Table Function Structures
	****************************************************************************/

	typedef struct
	{
		const FLMUNICODE *	puzTagName;
		FLMUINT					uiTagNum;
		FLMUINT					uiType;
		FLMUINT					uiSubType;
	} FLM_TAG_INFO;

	/// Class for mapping names to IDs and vice versa.\  Methods
	/// in this class allow an application to get the dictionary
	/// number for a field, index, or container using the field name,
	/// index name, or container name.\   It also allows an application to
	/// to get a field name, index name, or field name using the dictionary
	/// number.
	class FLMEXP F_NameTable : public F_Object
	{
	public:

		F_NameTable();

		virtual ~F_NameTable();

		/// Clear the name table.
		void clearTable( void);

		/// Populate a name table from the dictionary of the specified database.
		RCODE setupFromDb(
			HFDB				hDb						///< Database whose dictionary is to be used to populate the name table.
			);

		/// Get the next item from the the table in dictionary number order.
		/// This method allows an application to traverse through all of the items in a table
		/// in dictionary number order.
		FLMBOOL getNextTagNumOrder(
			FLMUINT *		puiNextPos,				///< Points to a variable that keeps the position of the last item returned.\  Returns
															///< the position of the next item.\   Initialize to zero to retrieve the first item.
			FLMUNICODE *	puzTagName,				///< If non-NULL, name is returned here as a Unicode string.
			char *			pszTagName,				///< If non-NULL, name is returned here as an ASCII string.\  NOTE: If both pszTagName
															///< and puzTagName are non-NULL, only puzTagName will be populated.
			FLMUINT			uiNameBufSize,			///< Size, in bytes, of the puzTagName or pszTagName buffer.\   Needs to be big enough
															///< include a null terminator character.
			FLMUINT *		puiTagNum = NULL,		///< If non-NULL, dictionary number for item is returned here.
			FLMUINT *		puiType = NULL,		///< If non-NULL, dictionary type (field, index, container) is returned here.
			FLMUINT *		puiSubType = NULL		///< If non-NULL, dictionary sub-type is returned here.\   NOTE: This only applies
															///< to field items, in which case the sub-type is the data type for the field.
			);

		/// Get the next item from the the table in name order.
		/// This method allows an application to traverse through all of the names in a table
		/// in name order.
		FLMBOOL getNextTagNameOrder(
			FLMUINT *		puiNextPos,				///< Points to a variable that keeps the position of the last item returned.\  Returns
															///< the position of the next item.\   Initialize to zero to retrieve the first item.
			FLMUNICODE *	puzTagName,				///< If non-NULL, name is returned here as a Unicode string.
			char *			pszTagName,				///< If non-NULL, name is returned here as an ASCII string.\  NOTE: If both pszTagName
															///< and puzTagName are non-NULL, only puzTagName will be populated.
			FLMUINT			uiNameBufSize,			///< Size, in bytes, of the puzTagName or pszTagName buffer.\   Needs to be big enough
															///< include a null terminator character.
			FLMUINT *		puiTagNum = NULL,		///< If non-NULL, dictionary number for item is returned here.
			FLMUINT *		puiType = NULL,		///< If non-NULL, dictionary type (field, index, container) is returned here.
			FLMUINT *		puiSubType = NULL		///< If non-NULL, dictionary sub-type is returned here.\   NOTE: This only applies
															///< to field items, in which case the sub-type is the data type for the field.
			);

		/// Get the next item from the the table of the specified type.
		/// This method allows an application to traverse through all of the items of a particular
		/// type (field, index, container).  Items will be returned in name order.
		FLMBOOL getFromTagType(
			FLMUINT			uiType,					///< Type of items to be returned.
			FLMUINT *		puiNextPos,				///< Points to a variable that keeps the position of the last item returned.\  Returns
															///< the position of the next item.\   Initialize to zero to retrieve the
															///< first item of the specified type.
			FLMUNICODE *	puzTagName,				///< If non-NULL, name is returned here as a Unicode string.
			char *			pszTagName,				///< If non-NULL, name is returned here as an ASCII string.\  NOTE: If both pszTagName
															///< and puzTagName are non-NULL, only puzTagName will be populated.
			FLMUINT			uiNameBufSize,			///< Size, in bytes, of the puzTagName or pszTagName buffer.\   Needs to be big enough
															///< include a null terminator character.
			FLMUINT *		puiTagNum = NULL,		///< If non-NULL, dictionary number for item is returned here.
			FLMUINT *		puiSubType = NULL		///< If non-NULL, dictionary sub-type is returned here.\   NOTE: This only applies
															///< to field items, in which case the sub-type is the data type for the field.
			);

		/// Get the item from the table with the specified dictionary number.
		FLMBOOL getFromTagNum(
			FLMUINT			uiTagNum,				///< Dictionary number of item to be retrieved.
			FLMUNICODE *	puzTagName,				///< If non-NULL, name is returned here as a Unicode string.
			char *			pszTagName,				///< If non-NULL, name is returned here as an ASCII string.\  NOTE: If both pszTagName
															///< and puzTagName are non-NULL, only puzTagName will be populated.
			FLMUINT			uiNameBufSize,			///< Size, in bytes, of the puzTagName or pszTagName buffer.\   Needs to be big enough
															///< include a null terminator character.
			FLMUINT *		puiType = NULL,		///< If non-NULL, dictionary type (field, index, container) is returned here.
			FLMUINT *		puiSubType = NULL		///< If non-NULL, dictionary sub-type is returned here.\   NOTE: This only applies
															///< to field items, in which case the sub-type is the data type for the field.
			);

		/// Get the item from the table with the specified name.
		FLMBOOL getFromTagName(
			const FLMUNICODE *	puzTagName,			///< If non-NULL, specifies name of item to find.
			const char *			pszTagName,			///< If non-NULL, specifies name of item to find.\   NOTE: If puzTagName is also
																///< non-NULL, the pszTagName parameter will be ignored.
			FLMUINT *				puiTagNum,			///< Dictionary number for item is returned here - must be non-NULL.
			FLMUINT *				puiType = NULL,	///< If non-NULL, dictionary type (field, index, container) is returned here.
			FLMUINT *				puiSubType = NULL	///< If non-NULL, dictionary sub-type is returned here.\   NOTE: This only applies
																///< to field items, in which case the sub-type is the data type for the field.
			);

		/// Get the item from the table with the specified type (field, index, or container) and name.
		FLMBOOL getFromTagTypeAndName(
			const FLMUNICODE *	puzTagName,			///< If non-NULL, specifies name of item to find.
			const char *			pszTagName,			///< If non-NULL, specifies name of item to find.\   NOTE: If puzTagName is also
																///< non-NULL, the pszTagName parameter will be ignored.
			FLMUINT					uiType,				///< Type of item to be found.
			FLMUINT *				puiTagNum,			///< Dictionary number for item is returned here - must be non-NULL.
			FLMUINT *				puiSubType = NULL	///< If non-NULL, dictionary sub-type is returned here.\   NOTE: This only applies
																///< to field items, in which case the sub-type is the data type for the field.
			);

		/// Insert an item into the table.
		RCODE addTag(
			const FLMUNICODE *	puzTagName,					///< If non-NULL, specifies name of item being added to the table.
			const char *			pszTagName,					///< If non-NULL, specifies name of item being added to the table.\   NOTE:
																		///< If puzTagName is also non-NULL, the pszTagName parameter
																		///< will be ignored.\   It is illegal for both puzTagName and pszTagName to be NULL.
			FLMUINT					uiTagNum,					///< Specifies the dictionary number of the item being added to the table.
			FLMUINT					uiType,						///< Specifies the dictionary type (field, index,or container) of the item being added to the table.
			FLMUINT					uiSubType,					///< Specifies the dictionary sub-type of the item being added to the table.\   NOTE:
																		///< This is only needed for field items, in which case the sub-type should be
																		///< the data type for the field.
			FLMBOOL					bCheckDuplicates = TRUE	///< Flag specifying whether to check for duplicates.\   Checks will be made
																		///< for duplicate name, duplicate type+name, and duplicate dictionary number.
																		///< If FALSE, duplicate checking is not done.\   If TRUE, this call becomes
																		///< more expensive because the table must be sorted in order to check for
																		///< duplicates.
			);

		/// Sort the items in the table.
		/// This method is typically called after adding a group of items to the table via the F_NameTable::addTag() method.
		void sortTags( void);

	private:

		RCODE allocTag(
			const FLMUNICODE *	puzTagName,
			const char *			pszTagName,
			FLMUINT					uiTagNum,
			FLMUINT					uiType,
			FLMUINT					uiSubType,
			FLM_TAG_INFO **		ppTagInfo);

		RCODE reallocSortTables(
			FLMUINT	uiNewTblSize);

		void copyTagName(
			FLMUNICODE *			puzDestTagName,
			char *					pszDestTagName,
			FLMUINT					uiDestBufSize,
			const FLMUNICODE *	puzSrcTagName);

		FLM_TAG_INFO * findTagByName(
			const FLMUNICODE *	puzTagName,
			const char *			pszTagName,
			FLMUINT *				puiInsertPos = NULL);

		FLM_TAG_INFO * findTagByNum(
			FLMUINT			uiTagNum,
			FLMUINT *		puiInsertPos = NULL);

		FLM_TAG_INFO * findTagByTypeAndName(
			const FLMUNICODE *	puzTagName,
			const char *			pszTagName,
			FLMUINT					uiType,
			FLMUINT *				puiInsertPos = NULL);

		RCODE insertTagInTables(
			FLM_TAG_INFO *			pTagInfo,
			FLMUINT					uiTagNameTblInsertPos,
			FLMUINT					uiTagTypeAndNameTblInsertPos,
			FLMUINT					uiTagNumTblInsertPos);

		F_Pool						m_pool;
		FLM_TAG_INFO **			m_ppSortedByTagName;
		FLM_TAG_INFO **			m_ppSortedByTagNum;
		FLM_TAG_INFO **			m_ppSortedByTagTypeAndName;
		FLMUINT						m_uiTblSize;
		FLMUINT						m_uiNumTags;
		FLMBOOL						m_bTablesSorted;
	};

	#define RECID_UNDEFINED		0xFFFFFFFF

	/// Structure returned from FlmIndexStatus() to report current status of an index.
	typedef struct
	{
		FLMUINT			uiIndexNum;					///< Index number.
		FLMBOOL			bSuspended;					///< If TRUE, index is suspended.
		FLMUINT			uiStartTime;				///< Time (GMT) that the background indexing thread started - if zero, no background thread is running.
		FLMUINT			uiLastRecordIdIndexed;	///< If RECID_UNDEFINED then index is online,
															///< otherwise this is the value of the last 
															///< DRN that was indexed.
		FLMUINT			uiKeysProcessed;			///< Keys processed by background indexing thread.
		FLMUINT			uiRecordsProcessed;		///< Records processed by background indexing thread.
		FLMUINT			uiTransactions;			///< Number of transactions started by the background indexing thread.
	} FINDEX_STATUS;

	/// Sub-query optimization types.
	typedef enum
	{
		QOPT_NONE = 0,
		QOPT_USING_INDEX,					///< Sub-query was optimized using an index.
		QOPT_USING_PREDICATE,			///< Sub-query was optimized using an application-defined predicate.
		QOPT_SINGLE_RECORD_READ,		///< Sub-query was optimized to retrieve a single record.
		QOPT_PARTIAL_CONTAINER_SCAN,	///< Sub-query was optimized to scan through a subset of records in a container.
		QOPT_FULL_CONTAINER_SCAN		///< Sub-query was optimized to scan through all of the records in a container.
	} qOptTypes;

	/// Structure returned when FCURSOR_GET_OPT_INFO_LIST option is passed to FlmCursorGetConfig().
	typedef struct
	{
		qOptTypes	eOptType;				///< Type of optimization done for sub-query.
		FLMUINT		uiCost;					///< Cost calculated for sub-query.
		FLMUINT		uiDrnCost;				///< DRN cost for sub-query.
		FLMUINT		uiIxNum;					///< Index used to execute query if eOptType is qOptTypes::QOPT_USING_INDEX.
		FLMBOOL		bDoRecMatch;			///< Record must be retrieved to test against query criteria.\   Only valid
													///< if OPT_INFO::eOptType is qOptTypes::QOPT_USING_INDEX.
		FLMBOOL		bDoKeyMatch;			///< Must match against index keys.  Only valid if OPT_INFO::eOptType is qOptTypes::QOPT_USING_INDEX.
		FLMUINT		uiDrn;					///< DRN to read if OPT_INFO::eOptType is qOptTypes::QOPT_SINGLE_RECORD_READ.
	} OPT_INFO;

	/// Structure that holds cache usage statistics.  The statistics will be for either block cache or record cache.
	typedef struct
	{
		FLMUINT				uiMaxBytes;					///< Maximum bytes allowed in cache.
		FLMUINT				uiTotalBytesAllocated;	///< Total bytes currently allocated in cache.
		FLMUINT				uiCount;						///< Number of items cached (blocks or records).
		FLMUINT				uiOldVerCount;				///< Number of items cached that are prior versions.
		FLMUINT				uiOldVerBytes;				///< Total bytes in prior versions.
		FLMUINT				uiCacheHits;				///< Total number of times an item was found in cache.
		FLMUINT				uiCacheHitLooks;			///< Total number of items traversed to find items in cache.
		FLMUINT				uiCacheFaults;				///< Total number of times an item was not found in cache.
		FLMUINT				uiCacheFaultLooks;		///< Total number of items traversed to determine that an item was not in cache.
		FLM_SLAB_USAGE		SlabUsage;					///< Slab usage statistics
	} FLM_CACHE_USAGE;

	/// Structure returned from FlmGetMemoryInfo().
	typedef struct
	{
		FLMBOOL				bDynamicCacheAdjust;			///< Flag indicating if FLAIM is using a dynamic cache limit or a hard cache limit.\   TRUE
																	///< if dynamic.
		FLMUINT				uiCacheAdjustPercent;		///< If using a dynamic cache limit, this will tell the percent of available memory to use for the limit.
		FLMUINT				uiCacheAdjustMin;				///< If using a dynamic cache limit, this will tell the minimum limit (in bytes) that can be set.
		FLMUINT				uiCacheAdjustMax;				///< If using a dynamic cache limit, this will tell the maximum limit (in bytes) that can be set.
		FLMUINT				uiCacheAdjustMinToLeave;	///< If using a dynamic cache limit, this tells the minimum amount of memory that must be left after
																	///< setting a limit.\   NOTE: This is only used if FLM_MEM_INFO::uiCacheAdjustMax is zero.
		FLMUINT				uiDirtyCount;					///< Number of blocks in block cache that are currently dirty.
		FLMUINT				uiDirtyBytes;					///< Total number of bytes in block cache that are currently dirty.
		FLMUINT				uiNewCount;						///< Number of blocks in block cache that are new blocks - blocks that were created new at
																	///< the end of the database.
		FLMUINT				uiNewBytes;						///< Total number of bytes for the new blocks.
		FLMUINT				uiLogCount;						///< Total number of blocks in the block cache that need to be logged to the rollback log.
		FLMUINT				uiLogBytes;						///< Total number of bytes in the log blocks.
		FLMUINT				uiFreeCount;					///< Total number of blocks in the block cache that are no longer associated with a particular
																	///< database.\   They can be reused.
		FLMUINT				uiFreeBytes;					///< Total number of bytes in the free blocks.
		FLMUINT				uiReplaceableCount;			///< Total number of blocks that can be replaced without having to write them to disk.
		FLMUINT				uiReplaceableBytes;			///< Total number of bytes in the replaceable blocks.
		FLM_CACHE_USAGE	RecordCache;					///< Record cache usage statistics.
		FLM_CACHE_USAGE	BlockCache;						///< Block cache usage statistics.
	} FLM_MEM_INFO;

	/// Structure returned to an event handler function whenever transaction events occur.\  Specifically, this structure is
	/// returned for transaction begin, commit, and abort events.
	typedef struct
	{
		FLMUINT		uiThreadId;		///< Thread id of thread that generated the event.
		HFDB			hDb;				///< Database handle of database that generated the event.
		FLMUINT		uiTransID;		///< Transaction id.
		RCODE			rc;				///< Return code of the event.
	} FLM_TRANS_EVENT;

	/// Structure returned to an event handler function whenever database update events occur.\  Specifically, this structure is
	/// returned for record add, modify, and delete events, as well as DRN reserve events.
	typedef struct
	{
		FLMUINT		uiThreadId;		///< Thread id of the thread that generated the event.
		HFDB			hDb;				///< Database handle of the database that generated the event.
		FLMUINT		uiTransID;		///< Transaction id the update occurred in.
		RCODE			rc;				///< Return code of the update event.
		FLMUINT		uiDrn;			///< DRN of the record that was added, modified, deleted, or reserved.
		FLMUINT		uiContainer;	///< Container number where the update occurred.
		FlmRecord *	pNewRecord;		///< New record (adds, modifies).
		FlmRecord *	pOldRecord;		///< Old record (modifies, deletes).
	} FLM_UPDATE_EVENT;
	
	/// Structure returned to an event handler function whenever RFL size events occur.\  Specifically, this structure is
	/// returned whenever the RFL exceeds the on-disk size threshold specified for a database.
	typedef struct
	{
		const char *	pszRflDir;				///< RFL directory path
		FLMUINT64		ui64RflDiskUsage;		///< Size of the RFL
	} FLM_RFL_SIZE_EVENT;

	/// Structure that gives the current state of the checkpoint thread.\  Returned from FlmDbGetConfig() when
	/// eDbGetConfigType::FDB_GET_CHECKPOINT_INFO is passed in as the option.
	typedef struct
	{
		FLMBOOL		bRunning;								///< Is checkpoint thread currently running?
		FLMUINT		uiRunningTime;							///< Time (milliseconds) the checkpoint thread has been running (if bRunning is TRUE).
		FLMBOOL		bForcingCheckpoint;					///< Is a checkpoint being forced?
		FLMUINT		uiForceCheckpointRunningTime;		///< Time (milliseconds) the checkpoint thread has been forcing a checkpoint
																	///< (only valid if both bRunning and bForcingCheckpoint are TRUE).
		FLMINT		iForceCheckpointReason;				///< Reason checkpoint is being forced (only valid if bForcingCheckpoint is TRUE).\   It may
																	///< be one of the following:\n
																	///< - CP_TIME_INTERVAL_REASON - Maximum time since last completed checkpoint has elapsed
																	///< - CP_SHUTTING_DOWN_REASON - Database is being closed
																	///< - CP_RFL_VOLUME_PROBLEM - Had problems writing to the roll-forward log
			#define			CP_TIME_INTERVAL_REASON				1
			#define			CP_SHUTTING_DOWN_REASON				3
			#define			CP_RFL_VOLUME_PROBLEM				4
		FLMBOOL		bWritingDataBlocks;					///< TRUE if checkpoint thread is currently writing out dirty data blocks.
		FLMUINT		uiLogBlocksWritten;					///< Total number of blocks written to the rollback log.
		FLMUINT		uiDataBlocksWritten;					///< Total number of dirty data blocks written.
		FLMUINT		uiDirtyCacheBytes;					///< Total bytes of dirty cache that still needs to be written.
		FLMUINT		uiBlockSize;							///< Block size for database (in bytes).
		FLMUINT		uiWaitTruncateTime;					///< Time (milliseconds) the checkpoint thread has been waiting to truncate the rollback log.
																	///< If zero, the checkpoint thread is not currently waiting to truncate the rollback log.
	} CHECKPOINT_INFO;

	/// Structure that reports information on the progress of FlmDbSweep().  The FlmDbSweep() status callback
	/// function is called and passed a pointer to this structure.
	typedef struct
	{
		HFDB			hDb;					///< Handle to database being traversed by FlmDbSweep().
		FLMUINT		uiRecId;				///< DRN of the current record being traversed by FlmDbSweep().
		FLMUINT		uiContainer;		///< Current container being traversed by FlmDbSweep().
		FlmRecord *	pRecord;				///< Pointer to current record being traversed by FlmDbSweep().
		void *		pvField;				///< Current field with the record being traversed by FlmDbSweep().
	} SWEEP_INFO;

	/// Structure that reports query processing progress.  This is returned to the callback function that is set by the
	/// FlmCursorConfig() function using the eCursorConfigType::FCURSOR_SET_STATUS_HOOK option.  The callback function
	/// is called from within FlmCursorFirst(), FlmCursorLast(), FlmCursorNext(), FlmCursorPrev(), and any other functions
	/// that retrieve records and evaluates them against the query criteria.  This structure is passed to the callback
	/// function when the eStatusType::FLM_SUBQUERY_STATUS status is reported.
	typedef struct
	{
		HFDB		hDb;							///< Database handle.
		FLMUINT	uiContainerNum;			///< Container number records are being retrieved from.
		FLMUINT	uiIndexNum;					///< Index being used, if any.\  Zero if none.
		FLMUINT	uiProcessedCnt;			///< Total records processed so far.
		FLMUINT	uiMatchedCnt;				///< Total records that matched the query criteria so far.
		FLMUINT	uiNumRejectedByCallback;///< Total records rejected by the record validator callback function.\  The record
													///< validator callback function is set by calling FlmCursorConfig() using the
													///< eCursorConfigType::FCURSOR_SET_REC_VALIDATOR option.
		FLMUINT	uiDupsEliminated;			///< Total duplicate records eliminated.
		FLMUINT	uiKeysTraversed;			///< Total index keys traversed.
		FLMUINT	uiKeysRejected;			///< Total index keys rejected.
		FLMUINT	uiRefsTraversed;			///< Total index key references traversed.
		FLMUINT	uiRefsRejected;			///< Total index key references rejected.
		FLMUINT	uiRecsFetchedForEval;	///< Total records read from the container to be evaluated.
		FLMUINT	uiRecsRejected;			///< Total records rejected.
		FLMUINT	uiRecsFetchedForView;	///< Total records retrieved after key passes criteria.
		FLMUINT	uiRecsNotFound;			///< Total records that were pointed to from an index key that could not be found.\  NOTE:
													///< If this number is non-zero, there may be a logical index corruption in the database.
	} FCURSOR_SUBQUERY_STATUS;

	/// FLAIM Functions
	typedef enum
	{
		FLM_UNKNOWN_FUNC = 0,

		FLM_CURSOR_COMPARE_DRNS,			///< FlmCursorCompareDRNs()
		FLM_CURSOR_CONFIG,					///< FlmCursorConfig()
		FLM_CURSOR_FIRST,						///< FlmCursorFirst()
		FLM_CURSOR_FIRST_DRN,				///< FlmCursorFirstDRN()
		FLM_CURSOR_GET_CONFIG,				///< FlmCursorGetConfig()
		FLM_CURSOR_LAST,						///< FlmCursorLast()
		FLM_CURSOR_LAST_DRN,					///< FlmCursorLastDRN()
		FLM_CURSOR_MOVE_RELATIVE,			///< FlmCursorMoveRelative()
		FLM_CURSOR_NEXT,						///< FlmCursorNext()
		FLM_CURSOR_NEXT_DRN,					///< FlmCursorNextDRN()
		FLM_CURSOR_PREV,						///< FlmCursorPrev()
		FLM_CURSOR_PREV_DRN,					///< FlmCursorPrevDRN()
		FLM_CURSOR_REC_COUNT,				///< FlmCursorRecCount()

		FLM_DB_CHECKPOINT,					///< FlmDbCheckpoint()
		FLM_DB_UPGRADE,						///< FlmDbUpgrade()
		FLM_DB_CREATE,							///< FlmDbCreate()
		FLM_DB_GET_COMMIT_CNT,				///< FlmDbGetCommitCnt()
		FLM_DB_GET_LOCK_INFO,				///< FlmDbGetLockInfo()
		FLM_DB_GET_LOCK_TYPE,				///< FlmDbGetLockType()
		FLM_DB_GET_TRANS_ID,					///< FlmDbGetTransId()
		FLM_DB_GET_TRANS_TYPE,				///< FlmDbGetTransType()
		FLM_DB_LOCK,							///< FlmDbLock()
		FLM_DB_OPEN,							///< FlmDbOpen()
		FLM_DB_REDUCE_SIZE,					///< FlmDbReduceSize()
		FLM_DB_SWEEP,							///< FlmDbSweep()
		FLM_DB_TRANS_ABORT,					///< FlmDbTransAbort()
		FLM_DB_TRANS_BEGIN,					///< FlmDbTransBegin()
		FLM_DB_TRANS_COMMIT,					///< FlmDbTransCommit()
		FLM_DB_UNLOCK,							///< FlmDbUnlock()

		FLM_INDEX_GET_NEXT,					///< FlmIndexGetNext()
		FLM_INDEX_STATUS,						///< FlmIndexStatus()
		FLM_INDEX_RESUME,						///< FlmIndexResume()
		FLM_INDEX_SUSPEND,					///< FlmIndexSuspend()

		FLM_KEY_RETRIEVE,						///< FlmKeyRetrieve()
		FLM_RESERVE_NEXT_DRN,				///< FlmReserveNextDrn()
		FLM_RECORD_ADD,						///< FlmRecordAdd()
		FLM_RECORD_MODIFY,					///< FlmRecordModify()
		FLM_RECORD_DELETE,					///< FlmRecordDelete()
		FLM_RECORD_RETRIEVE,					///< FlmRecordRetrieve()

		FLM_DB_REMOVE,
		FLM_DB_LOGHDR,

		// Always insert new funcs before LAST_FLM_FUNC 
		LAST_FLM_FUNC
	} eFlmFuncs;

	/// Structure that is returned to the callback function that is passed to FlmDbCopy().  This structure is passed
	/// to the callback function when the eStatusType::FLM_DB_COPY_STATUS status is reported.
	typedef struct
	{
		FLMUINT64		ui64BytesToCopy;							///< Total bytes to copy.
		FLMUINT64		ui64BytesCopied;							///< Total bytes copied so far.
		FLMBOOL			bNewSrcFile;								///< If TRUE, we are starting to copy a new file.\  szSrcFileName and
																			///< szDestFileName will be set.
		char				szSrcFileName[ F_PATH_MAX_SIZE];		///< Name of source file currently being copied.
		char				szDestFileName[ F_PATH_MAX_SIZE];	///< Name of destination file being copied to.
	} DB_COPY_INFO;

	/// Structure that is returned to the callback function that is passed to FlmDbRebuild().  This structure is passed
	/// to the callback function when the eStatusType::FLM_CHECK_RECORD_STATUS or eStatusType::FLM_EXAMINE_RECORD_STATUS
	/// status is reported.
	typedef struct
	{
		FlmRecord *		pRecord;			///< Record to examine to see if it contains dictionary information.
		FLMUINT			uiContainer;	///< Container the record came from.
		FLMUINT			uiDrn;			///< DRN of record.
		FlmRecordSet *	pDictRecSet;	///< Callback function may create a ::FlmRecordSet object that is returned to
												///< FlmDbRebuild().\  This object contains a set of dictionary records the application
												///< wants FlmDbRebuild() to inject into the dictionary it is rebuilding.\  FlmDbRebuild()
												///< will only use these records if it cannot recover the actual dictionary record
												///< from the dictionary container.\  FlmDbRebuild() will free the ::FlmRecordSet
												///< object pointed to by pDictRecSet when it is done using the records in the set.\  NOTE:
												///< FlmDbRebuild() does not do anything with records that are returned in this set
												///< if the status being reported is eStatusType::FLM_EXAMINE_RECORD_STATUS.
	} CHK_RECORD;

	/// Structure that is returned to the callback function that is passed to FlmDbUpgrade().  This structure is passed
	/// to the callback function when the eStatusType::FLM_DB_UPGRADE_STATUS status is reported.
	typedef struct
	{
		FLMUINT		uiDrn;			///< Current DRN being processed.
		FLMUINT		uiLastDrn;		///< Highest DRN in this container.
		FLMUINT		uiContainer;	///< Container being processed.
	} DB_UPGRADE_INFO;

	/// Structure that is returned to the callback function that is passed to FlmDbBackup().  This structure is passed
	/// to the callback function when the eStatusType::FLM_DB_BACKUP_STATUS status is reported.
	typedef struct
	{
		FLMUINT64		ui64BytesToDo;	///< Total bytes to be backed up.
		FLMUINT64		ui64BytesDone;	///< Total bytes backed up so far.
	} DB_BACKUP_INFO;

	/// Structure that is returned to the callback function that is passed to FlmDbRename().  This structure is passed
	/// to the callback function when the eStatusType::FLM_DB_RENAME_STATUS status is reported.
	typedef struct
	{
		char			szSrcFileName[ F_PATH_MAX_SIZE];	///< File being renamed.
		char			szDstFileName[ F_PATH_MAX_SIZE];	///< Name the file is to be renamed to.
	} DB_RENAME_INFO;

	/// Structure for nodes used in GEDCOM functions.\  Nodes are the basic components of GEDCOM trees.
	typedef struct NODE
	{
		NODE *		next;					///< Pointer to child, next sib, or uncle (compare levels).
		NODE *		prior;				///< Pointer to parent, prior sib, or nephew (compare levels).
		FLMBYTE *	value;				///< Value of node (if length <= 4), or pointer to value.
		FLMUINT32	ui32Length;			///< Length of value (in bytes).
		FLMUINT16	ui16TagNum;			///< Tag number.
		FLMUINT8		ui8Level;			///< Hierarchy level (0 = root)
		FLMUINT8		ui8Type;				///< Value's data type.\  This should be one of the
												///< following:\n
												///< - FLM_TEXT_TYPE (0)
												///< - FLM_NUMBER_TYPE (1)
												///< - FLM_BINARY_TYPE (2)
												///< - FLM_CONTEXT_TYPE (3)
												///< - FLM_BLOB_TYPE (8)

		// Define the allowable field types

		#define FLM_TEXT_TYPE			0
		#define FLM_NUMBER_TYPE			1
		#define FLM_BINARY_TYPE			2
		#define FLM_CONTEXT_TYPE		3
		#define FLM_BLOB_TYPE			8		// Blob type - internal or external

		#define FLM_DEFAULT_TYPE	FLM_CONTEXT_TYPE

		#define FLM_DATA_LEFT_TRUNCATED	0x10	// Data is left truncated
		#define FLM_DATA_RIGHT_TRUNCATED	0x20	// Data is right truncated.
		#define HAS_REC_SOURCE				0x40	// Node has HFDB, container, and
															// RecId immediatly following the node.
		#define HAS_REC_ID					0x80	// Node has Record Id (4 bytes) immediately
															// following the node.
		FLMUINT32	ui32EncFlags;					///< Encryption flags.
		FLMUINT32	ui32EncLength;					///< The length of the encrypted data.
		FLMUINT32	ui32EncId;						///< The DRN of the encryption definition record in the dictionary.
		FLMBYTE *	pucEncValue;					///< The encrypted value.
	} NODE;

	/// This class allows an application to keep a set of ::FlmRecord objects.
	class FLMEXP FlmRecordSet : public F_Object
	{
	public:

		FlmRecordSet()
		{
			m_iCurrRec = -1;
			m_ppRecArray = NULL;
			m_iRecArraySize = 0;
			m_iTotalRecs = 0;
		}

		virtual ~FlmRecordSet();

		/// Insert a ::FlmRecord into the set.
		RCODE insert(
			FlmRecord * pRecord	///< Pointer to ::FlmRecord that is to be inserted into the set.
			);

		/// Clear all records in the set.
		void clear( void);

		/// Position to and return a pointer to the first ::FlmRecord object in the set.
		FINLINE FlmRecord * first( void)
		{
			if (!m_iTotalRecs)
			{
				return( NULL);
			}
			m_iCurrRec = 0;
			return( m_ppRecArray [0]);
		}

		/// Position to and return a pointer to the last ::FlmRecord object in the set.
		FINLINE FlmRecord * last( void)
		{
			if (!m_iTotalRecs)
			{
				return( NULL);
			}
			m_iCurrRec = m_iTotalRecs - 1;
			return( m_ppRecArray [m_iCurrRec]);
		}

		/// Position to and return a pointer to the next ::FlmRecord object in the set.  NOTE: This method
		/// will return the first record in the set if no previous calls have been made to retrieve
		/// records from the set.
		FlmRecord * next( void);

		/// Position to and return a pointer to the next ::FlmRecord object in the set.  NOTE: This method
		/// will return NULL if no previous calls have been made to retrieve
		/// records from the set.
		FINLINE FlmRecord * prev( void)
		{
			if (m_iCurrRec - 1 < 0)
			{
				m_iCurrRec = -1;
				return( NULL);
			}
			m_iCurrRec--;
			return( m_ppRecArray [m_iCurrRec]);
		}

		/// Return the total number of records in the set.
		FINLINE FLMINT count( void)
		{	
			return m_iTotalRecs;
		}

	private:

		FLMINT			m_iCurrRec;
		FlmRecord **	m_ppRecArray;
		FLMINT			m_iRecArraySize;
		FLMINT			m_iTotalRecs;
	};

	/// This is an abstract base class which defines the interface that an application
	/// must implement to embed its own predicate in a query.
	class FLMEXP FlmUserPredicate : public F_Object
	{
	public:

		/// Method that returns the search cost of this object in providing
		/// records for a query.  FLAIM uses the information returned from this
		/// method to determine how to optimize the query.
		virtual RCODE searchCost(
			HFDB			hDb,					///< Database handle.  NOTE: Application should NOT save this
													///< handle.\   It may be changed from one call to the next.
			FLMBOOL		bNotted,				///< Flag indicating predicate is notted in the search criteria.
			FLMBOOL		bExistential,		///< Flag indicating predicate is "existential" (TRUE) or "universal" (FALSE).
			FLMUINT *	puiCost,				///< Estimated cost is returned here.
			FLMUINT *	puiDrnCost,			///< Estimated DRN cost is returned here.
			FLMUINT *	puiTestRecordCost,///< Test record cost is returned here.
			FLMBOOL *	pbPassesEmptyRec	///< Returns flag indicating whether the predicate would pass or fail an empty record.
			) = 0;

		/// Method that returns the cost of testing ALL record.
		virtual RCODE testAllRecordCost(
			HFDB			hDb,			///< Database handle.
			FLMUINT *	puiCost		///< Estimated cost is returned here.
			) = 0;

		/// Position to and return the first record that satisfies the predicate.
		virtual RCODE firstRecord(
			HFDB				hDb,		///< Database handle.\   NOTE: Application should NOT save this
											///< handle.\   It may be changed from one call to the next.
			FLMUINT *		puiDrn,	///< If non-NULL, record's DRN is returned here.
			FlmRecord **	ppRecord	///< If non-NULL, record is returned here.
			) = 0;

		/// Position to and return the last record that satisfies the predicate.
		virtual RCODE lastRecord(
			HFDB				hDb,		///< Database handle.\   NOTE: Application should NOT save this
											///< handle.\   It may be changed from one call to the next.
			FLMUINT *		puiDrn,	///< If non-NULL, record's DRN is returned here.
			FlmRecord **	ppRecord	///< If non-NULL, record is returned here.
			) = 0;

		/// Position to and return the next record that satisfies the predicate.
		/// If no prior positioning has been done, position to and return the first record.
		virtual RCODE nextRecord(
			HFDB				hDb,		///< Database handle.\   NOTE: Application should NOT save this
											///< handle.\   It may be changed from one call to the next.
			FLMUINT *		puiDrn,	///< If non-NULL, record's DRN is returned here.
			FlmRecord **	ppRecord	///< If non-NULL, record is returned here.
			) = 0;

		/// Position to and return the previous record that satisfies the predicate.
		/// If no prior positioning has been done, position to and return the last record.
		virtual RCODE prevRecord(
			HFDB				hDb,		///< Database handle.\   NOTE: Application should NOT save this
											///< handle.\   It may be changed from one call to the next.
			FLMUINT *		puiDrn,	///< If non-NULL, record's DRN is returned here.
			FlmRecord **	ppRecord	///< If non-NULL, record is returned here.
			) = 0;

		/// Test a record to see if it passes the criteria of the predicate.
		virtual RCODE testRecord(
			HFDB				hDb,		///< Database handle.\   NOTE: Application should NOT save this
											///< handle.\   It may be changed from one call to the next.
			FlmRecord *	pRecord,		///< Record to be tested.
			FLMUINT		uiDrn,		///< DRN of record to be tested.
			FLMUINT *	puiResult	///< Result is returned here.\  Should be one of the
											///< following:\n
											///< - FLM_FALSE - should be returned if predicate fails the record
											///< - FLM_TRUE - should be returned if the predicate passes the record
											///< - FLM_UNK - should be returned if it cannot be determined whether the record passes or fails
			) = 0;

		/// Return index being used for this predicate.
		virtual FLMUINT getIndex( 
			FLMUINT *	puiIndex		///< Index number is returned here.\   If no index is being used, a zero should be returned.
			) = 0;

		/// Copy the predicate.  In the new object, it should be as if the predicate had just
		/// been defined.  The current position, calculated score, etc. should not be preserved in the
		/// new predicate.  Only the predicate criteria - however that is represented - should be
		/// preserved.  This method should return NULL if a new predicate cannot be created.
		/// FLAIM will call this method whenever it needs to copy an application-defined
		/// predicate - such as when a query or some part of a query is cloned.
		virtual FlmUserPredicate * copy( void) = 0;

		/// Return predicate's FLAIM query handle, if any.  This is useful if an application-defined
		/// predicate is implemented as a FLAIM query.  It is often very useful to embed one FLAIM
		/// query inside another.
		virtual HFCURSOR getCursor( void) = 0;

		/// Position this predicate to the same position as
		/// another predicate.
		virtual RCODE positionTo(
			HFDB						hDb,						///< Database handle.
			FlmUserPredicate *	pPredicate				///< Predicate whose position this predicate is to be positioned to.
			) = 0;

		/// Save current position of predicate.
		virtual RCODE savePosition( void) = 0;

		/// Restore last saved position of predicate.
		virtual RCODE restorePosition( void) = 0;

		/// Determine if predicate is absolute positionable
		virtual RCODE isAbsPositionable(
			HFDB			hDb,									///< Database handle.
			FLMBOOL *	pbIsAbsPositionable				///< Returns TRUE/FALSE indicating if predicate is "absolute" positionable.
			) = 0;

		/// Get absolute record count.  NOTE: This method should only be called if the predicate is "absolute" positionable (see
		/// FlmUserPredicate::isAbsPositionable()).
		virtual RCODE getAbsCount(
			HFDB			hDb,					///< Database handle.
			FLMUINT *	puiCount				///< Returns total number of records this predicate would return.
			) = 0;

		/// Get absolute position.  NOTE: This method should only be called if the predicate is "absolute" positionable (see
		/// FlmUserPredicate::isAbsPositionable()).
		virtual RCODE getAbsPosition(
			HFDB			hDb,					///< Database handle.
			FLMUINT *	puiPosition			///< Returns the current "absolute" position of this predicate.
			) = 0;

		/// Set absolute position.  NOTE: This method should only be called if the predicate is "absolute" positionable (see
		/// FlmUserPredicate::isAbsPositionable()).
		virtual RCODE positionToAbs(
			HFDB			hDb,					///< Database handle.
			FLMUINT		uiPosition,			///< Absolute position to position this predicate to.
			FLMBOOL		bFallForward,		///< If the record at the position specified cannot be returned, this flag indicates whether
													///< the method should attempt to "fall forward" to the next record in the result set that
													///< can be returned.
			FLMUINT		uiTimeLimit,		///< Time limit (seconds) for this operation.
			FLMUINT *	puiPosition,		///< Returns the actual position that was positioned to.\   The only time this could be different
													///< than the position requested in the uiPosition parameter is if the record at the
													///< requested position cannot be returned and the bFallForward flag is TRUE.
			FLMUINT *	puiDrn				///< Returns the DRN of the record at the position we ended getting positioned to.
			) = 0;

		/// Release any resources held by this predicate.  This method should release any resources (memory, etc.) consumed by
		/// the predicate except those that would be needed to show the predicate's query criteria.  At the point in time where
		/// FLAIM calls this method, the predicate is no longer going to be used to retrieve or test records, but the query is
		/// being kept around to allow a user to see information about queries that have been run.  Thus, enough information
		/// should be preserved to allow the predicate to show its query criteria and any statistics it may have collected if it was
		/// called to get records (firstRecord, lastRecord, nextRecord, prevRecord).
		virtual void releaseResources( void) = 0;
	};

	/****************************************************************************
						Cursor Typedefs and Function Prototypes
	****************************************************************************/

	#define FLM_FALSE			(1)
	#define FLM_TRUE			(2)
	#define FLM_UNK			(4)

	/// Query value types and operators.
	typedef enum
	{
		NO_TYPE = 0,				// 0 (internal use only)

	// WARNING: Don't renumber below _VAL enums without 
	// redoing gv_DoValAndDictTypesMatch table

		FLM_BOOL_VAL = 1,			///< 1 - Boolean value.
		FLM_UINT32_VAL,			///< 2 - 32 bit unsigned integer value.
		FLM_INT32_VAL,				///< 3 - 32 bit signed integer value.
		FLM_REAL_VAL,				///< 4 - Real number value.\   NOTE: Not currently supported.
		FLM_REC_PTR_VAL,			///< 5 - DRN value.
		FLM_UINT64_VAL,			///< 6 - 64 bit unsigned integer value.
		FLM_INT64_VAL,				///< 7 - 64 bit signed integer value.
		FLM_BINARY_VAL = 9,		///< 9 - Binary value.
		FLM_STRING_VAL,			///< 10 - ASCII string value.
		FLM_UNICODE_VAL,			///< 11 - Unicode string value.
		FLM_TEXT_VAL,				///< 12 - Internal FLAIM string value.

			// Enums for internal use
			FIRST_VALUE = FLM_BOOL_VAL,	
			LAST_VALUE = FLM_TEXT_VAL,	

		FLM_FLD_PATH = 25,		// 25
		FLM_CB_FLD,					// 26

		// NOTE: These operators MUST stay in this order - this order is assumed
		// by the precedence table - see fqstack.cpp

		FLM_AND_OP = 100,			///< 100 - Logical AND operator.
		FLM_OR_OP,					///< 101 - Logical OR operator.
		FLM_NOT_OP,					///< 102 - Logical NOT operator.
		FLM_EQ_OP,					///< 103 - Equals comparison operator.
		FLM_MATCH_OP,				///< 104 - String match comparison operator.
		FLM_MATCH_BEGIN_OP,		///< 105 - Sring match-begin comparison operator.
		FLM_MATCH_END_OP,			///< 106 - String match-end comparison operator.
		FLM_CONTAINS_OP,			///< 107 - String contains comparison operator.
		FLM_NE_OP,					///< 108 - Not equal comparison operator.
		FLM_LT_OP,					///< 109 - Less than comparison operator.
		FLM_LE_OP,					///< 110 - Less than or equal comparison operator.
		FLM_GT_OP,					///< 111 - Greater than comparison operator.
		FLM_GE_OP,					///< 112 - Greater than or equal comparison operator.
		FLM_BITAND_OP,				///< 113 - Bitwise-AND arithmetic operator.
		FLM_BITOR_OP,				///< 114 - Bitwise-OR arithmetic operator.
		FLM_BITXOR_OP,				///< 115 - Bitwise-XOR arithmetic operator.
		FLM_MULT_OP,				///< 116 - Multiply arithmetic operator.
		FLM_DIV_OP,					///< 117 - Divide arithmetic operator.
		FLM_MOD_OP,					///< 118 - Modulo arithmetic operator.
		FLM_PLUS_OP,				///< 119 - Addition arithmetic operator.
		FLM_MINUS_OP,				///< 120 - Subtraction arithmetic operator.
		FLM_NEG_OP,					///< 121 - Unary minus (negative) arithmetic operator.
		FLM_LPAREN_OP,				///< 122 - Left parentheses.
		FLM_RPAREN_OP,				///< 123 - Right parentheses.
		FLM_UNKNOWN,	 			// 124
		FLM_USER_PREDICATE,		// 125
		FLM_EXISTS_OP,				// 126 For internal use only - key generation

			// Enums for internal use
			FIRST_OP				= FLM_AND_OP,
			FIRST_LOG_OP		= FIRST_OP,
			LAST_LOG_OP			= FLM_NOT_OP,
			FIRST_COMPARE_OP	= FLM_EQ_OP,
			LAST_COMPARE_OP	= FLM_GE_OP,
			FIRST_ARITH_OP		= FLM_BITAND_OP,
			LAST_ARITH_OP		= FLM_MINUS_OP,
			LAST_OP				= LAST_ARITH_OP

	} QTYPES;

	/// Initialize a query object.
	/// \ingroup queryobj
	FLMXPC RCODE FLMAPI FlmCursorInit(
		HFDB			hDb,					///< Database handle.
		FLMUINT		uiContainerNum,	///< Container to be searched.
		HFCURSOR *	phCursor				///< Query handle is returned here.
		);

	/// Free a query object.
	/// \ingroup queryobj
	FLMXPC RCODE FLMAPI FlmCursorFree(
		HFCURSOR *	phCursor				///< Pointer to query handle to be freed.\   Should be the handle returned from FlmCursorInit().
		);

	/// Release query object resources.  NOTE: This will free all of the resources for a query object except those needed to
	/// display the query's criteria and any statistics for the query that were collected while it was running.  After this
	/// method is called, the query object is no longer in a state where it can be used to retrieve records from the query
	/// result set.
	/// \ingroup queryobj
	FLMXPC void FLMAPI FlmCursorReleaseResources(
		HFCURSOR	hCursor					///< Handle to query object whose resources are to be released.
		);

	/// Clone a query object.  The new cloned query object should be set up with the same query criteria as the query
	/// object being cloned, but it should not be optimized yet.
	/// \ingroup queryobj
	FLMXPC RCODE FLMAPI FlmCursorClone(
		HFCURSOR		hSource,				///< Handle to query object that is to be cloned.
		HFCURSOR *	phCursor				///< Newly cloned query object handle is returned here.
		);

	/// Configuration options for FlmCursorConfig().
	typedef enum
	{
		FCURSOR_CLEAR_QUERY = 2,			///< Clear query criteria.
		FCURSOR_GEN_POS_KEYS,				///< Generate positioning keys.
		FCURSOR_SET_HDB,						///< Set the database handle for the query.\   pvValue1 is an HFDB - the database handle.
		FCURSOR_SET_FLM_IX,					///< Set the index for the query.\   pvValue1 is a FLMUINT - the index number.\   A value of zero may be
													///< specified to indicate that no index is to be used.\   A value of #FLM_SELECT_INDEX specifies that
													///< FLAIM is to select the index.\   This is the default.
		FCURSOR_SET_OP_TIME_LIMIT,			///< Set a time limit for the query.\   pvValue1 is a FLMUINT - timeout in seconds.
		FCURSOR_SET_PERCENT_POS,			///< Position to a percent position in the query result set.\   pvValue1 is a FLMUINT between 1 and 100.
		FCURSOR_SET_POS,						///< Position to the same position another query object is positioned to.\   pvValue1 is CURSOR * - a
													///< pointer to the query object whose position we are to position to.
		FCURSOR_SET_POS_FROM_DRN,			///< Position to a specific record in the query result set.\   pvValue1 is a FLMUINT - the DRN of the
													///< record to be positioned to.\   If the record is not in the result set, an error will be returned.
		FCURSOR_SET_REC_TYPE,				///< Set the type of record that this query should return.\   This basically adds a special criteria for
													///< the query that specifies that only records whose root field number matches a certain value should be
													///< be returned.\   pvValue1 is a FLMUINT - the root field number (or record type) to be matched.
		FCURSOR_RETURN_KEYS_OK,				///< Sets a flag that specifies whether it is ok for this query to return keys from the index instead
													///< of actual records from the container.\   pvValue1 is a FLMBOOL - TRUE means that index keys may be
													///< returned.\   FALSE means that only real records from the container may be returned.\   This flag is
													///< only used if it is possible to test the query criteria using only information retrieved from an
													///< index - provided, of course, that an index is used to perform the query.
		FCURSOR_DISCONNECT = 14,			///< Disconnect the cursor from any association with the current database handle, if any.\  If any internal
													///< read transaction is going, it will be aborted.
		FCURSOR_ALLOW_DUPS,					///< Allow duplicate records in query result set.
		FCURSOR_ELIMINATE_DUPS,				///< Disallow duplicate records in query result set.
		FCURSOR_SET_REC_VALIDATOR,			///< Set a record validator function that is to be called for each record that passes the query
													///< query criteria.\  pvValue1 is a ::REC_VALIDATOR_HOOK - the record validator function.\   The 
													///< record validator function may apply additional criteria to determine if the record should
													///< really be allowed to be returned or if it should be failed.
		FCURSOR_SET_STATUS_HOOK,			///< Set a status function that is called to report progress in evaluating the query.\   pvValue1 is
													///< a ::STATUS_HOOK - the status function.\   pvValue2 is application data that will be passed to the
													///< status function whenever it is called.
		FCURSOR_SAVE_POSITION,				///< Save the current position of the query within its query result set.\   This option is provided so
													///< that an application can temporarily position to some other place in its result set, but then
													///< returned to a saved position easily.
		FCURSOR_RESTORE_POSITION,			///< Restore the current position of the query within its query result set.
		FCURSOR_SET_ABS_POS					///< Set the absolute position in the query query result set.\   pvValue1 is a FLMUINT *.\   On input
													///< *(FLMUINT *)pvValue1 is the absolute position the query is to be set to.\   On output, it returns
													///< the position actually set to.\   pvValue2 is a FLMBOOL.\   If record at the position specified does
													///< not pass the query criteria, this flag specifies whether to position to the next or previous
													///< record in the result set that does match the criteria.
	} eCursorConfigType;

	/// Value passed to FlmCursorConfig() when ::FCURSOR_SET_FLM_IX is specified - allows FLAIM to select the index(es) for the query.
	#define FLM_SELECT_INDEX	32050

	/// Options for FlmCursorGetConfig().
	typedef enum
	{
		FCURSOR_GET_OPT_INFO_LIST = 3,	///< Get the optimization information for the query.\  pvValue1 is a pointer to an array of
													///< ::OPT_INFO structures.\  If a NULL is passed, no optimization information is returned,
													///< but a count of ::OPT_INFO structures needed will be returned in the pvValue2 parameter.\   pvValue2 
													///< is a FLMUINT *.\  It returns the number of elements needed in the ::OPT_INFO
													///< array.\   One ::OPT_INFO structure is returned for each sub-query
													///< that is optimized separately from other sub-queries.\   Typically, an application should
													///< call FlmCursorGetConfig() twice for this option - the first time with a NULL in pvValue1
													///< to get the size of the ::OPT_INFO array needed.\ Then, allocate memory for the array and
													///< call it again.
		FCURSOR_GET_FLM_IX,					///< Get the index being used for the query.\  pvValue1 is a FLMUINT *.\   The index number,
													///< if any, is returned here.\  A zero means that no index is being used.\   If multiple
													///< indexes are being used, pvValue1 will only return the first index.\   pvValue2 is a
													///< FLMUINT *.\   It returns flags which indicates if more than one index is being used.\   It
													///< will be one of the following: \n
													///< - HAVE_NO_INDEX - Query was not using an index (pvValue1 should return 0 in this case)
													///< - HAVE_ONE_INDEX - Query is using exactly one index (pvValue1 will return index number)
													///< - HAVE_ONE_INDEX_MULT_PARTS - Query is using exactly one index, but there are multiple parts of
													///< the index that are being searched
													///< - HAVE_MULTIPLE_INDEXES - Query is using multiple indexes (pvValue1 will return only the first index)
		FCURSOR_GET_OPT_INFO,				///< Get optimization information for the query.\  pvValue1 is a pointer to an ::OPT_INFO structure
													///< where the optimization information will be returned.\  This option only returns the first ::OPT_INFO
													///< structure.\  If there are multiple sub-queries, use eCursorGetConfigType::FCURSOR_GET_OPT_INFO_LIST.
		FCURSOR_GET_PERCENT_POS,			///< Get the current percent position of the query in the query result set.\  pvValue1 is a FLMUINT *.\   It
													///< returns the current percent position.\   NOTE: This option should only be called if the query is
													///< percent positionable - see eCursorGetConfigType::FCURSOR_GET_POSITIONABLE.
		FCURSOR_GET_REC_TYPE = 9,			///< Get the record type that has been set for the query, if any.\  If a record type was set, it would
													///< have been set using the eCursorConfigType::FCURSOR_SET_REC_TYPE option of the FlmCursorConfig()
													///< function.\   pvValue1 is a FLMUINT *.\  It returns the record type.
		FCURSOR_GET_FLAGS = 12,				///< Get the current mode flags for the query.\  These are the mode flags that would have been set using
													///< the FlmCursorSetMode() function.\   pvValue1 is a FLMUINT *.\  It returns the flags.
		FCURSOR_GET_STATE,					///< Get the current state of the query.\  pvValue1 is a FLMUINT *.\  It returns flags indicating what
													///< the current state of the query is.\  The flags are as follows:\n
													///< - FCURSOR_HAVE_CRITERIA - Indicates that some query criteria has been set.\  Query criteria may or
													///< may not be complete
													///< - FCURSOR_EXPECTING_OPERATOR - Indicates that the query criteria is expecting an operator to be
													///< submitted next
													///< - FCURSOR_QUERY_COMPLETE - Indicates that the query criteria is in a "complete" state - that is,
													///< it is syntatically complete and could be used to retrieve records
													///< - FCURSOR_QUERY_OPTIMIZED - Indicates that the query has already been optimized
													///< - FCURSOR_READ_PERFORMED - Indicates that the query is ready to have records retrieved
		FCURSOR_GET_POSITIONABLE,			///< Get whether or not the query is "percentage" positionable.\  pvValue1 is a FLMBOOL *.\   It returns
													///< a TRUE/FALSE flag indicating whether the query can be percentage positioned.
		FCURSOR_AT_BOF,						///< Get whether or not the query is positioned at BOF.\  pvValue1 is a FLMBOOL *.\  It returns a TRUE/FALSE
													///< flag indicating whether the query is at BOF.
		FCURSOR_AT_EOF,						///< Get whether or not the query is positioned at EOF.\  pvValue1 is a FLMBOOL *.\  It returns a TRUE/FALSE
													///< flag indicating whether the query is at EOF.
		FCURSOR_GET_ABS_POSITIONABLE,		///< Get whether or not the query is "absolute" positionable.\  pvValue1 is a FLMBOOL *.\  It returns a TRUE/FALSE
													///< flag indicating whether the query is absolute positionable.
		FCURSOR_GET_ABS_POS,					///< Get the current absolute position of the query in the query result set.\  pvValue1 is a FLMUINT *.\   It
													///< returns the current absolute position.\  NOTE: This option should only be used if the query is
													///< absolute positionable - see eCursorGetConfigType::FCURSOR_GET_ABS_POSITIONABLE.
		FCURSOR_GET_ABS_COUNT				///< Get the absolute count of the query result set.\  pvValue1 is a FLMBOOL *.\  It returns the
													///< absolute count.\  NOTE: This option should only be used if the query is absolute positionable - see
													///< eCursorGetConfigType::FCURSOR_GET_ABS_POSITIONABLE.
	} eCursorGetConfigType;

	// Values returned when FlmCursorGetConfig() is called with the FCURSOR_GET_FLM_IX option.

	#define HAVE_NO_INDEX					0
	#define HAVE_ONE_INDEX					1
	#define HAVE_ONE_INDEX_MULT_PARTS	2
	#define HAVE_MULTIPLE_INDEXES			3

	// Predefined values for query state (FCURSOR_GET_STATE)

	#define FCURSOR_HAVE_CRITERIA			0x01
	#define FCURSOR_EXPECTING_OPERATOR	0x02
	#define FCURSOR_QUERY_COMPLETE		0x04
	#define FCURSOR_QUERY_OPTIMIZED		0x08
	#define FCURSOR_READ_PERFORMED		0x10

	/// Configure a query object.
	/// \ingroup queryconfig
	FLMXPC RCODE FLMAPI FlmCursorConfig(
		HFCURSOR					hCursor,				///< Handle to query object that is to be configured.
		eCursorConfigType		eConfigType,		///< Specifies what is to be configured in the query object.
		void *					pvValue1,			///< Configuration parameter - depends on eConfigType - see documentation on ::eCursorConfigType.
		void *					pvValue2				///< Configuration parameter - depends on eConfigType - see documentation on ::eCursorConfigType.
		);

	/// Get query configuration.
	/// \ingroup queryconfig
	FLMXPC RCODE FLMAPI FlmCursorGetConfig(
		HFCURSOR						hCursor,				///< Handle to query object whose configuration information is to be retrieved.
		eCursorGetConfigType		eGetConfigType,	///< Specifies what configuration information is to be retrieved.
		void *						pvValue1,			///< Configuration parameter - depends on eGetConfigType - see documentation on ::eCursorGetConfigType.
		void *						pvValue2				///< Configuration parameter - depends on eGetConfigType - see documentation on ::eCursorGetConfigType.
		);

	/// Set order index for a query.
	/// \ingroup queryconfig
	FLMXPC RCODE FLMAPI FlmCursorSetOrderIndex(
		HFCURSOR		hCursor,							///< Handle to query object whose order index is to be set.
		FLMUINT *	puiFieldPaths,					///< List of field paths that specify the desired ordering.\ Each field path is
															///< terminated with a single zero, and the entire list is terminated
															///< by two zeroes.
		FLMUINT *	puiIndex							///< Index is returned here.\  If zero is returned, it means that no
															///< index could be found that matched the specified field order.
		);

	/// Set mode for string comparison operations in query criteria.
	/// \ingroup queryconfig
	FLMXPC RCODE FLMAPI FlmCursorSetMode(
		HFCURSOR		hCursor,							///< Handle to query object whose order index is to be set.
		FLMUINT		uiFlags							///< Mode flags to be set for the query.\  Multiple flags may be ORed together.\  Valid
															///< flags are as follows:
															///< - FLM_WILD - Treat '*' as a wildcard
															///< - FLM_NOCASE - Case-insensitive comparison
															///< - FLM_NO_SPACE - Ignore all white space
															///< - FLM_NO_DASH - Ignore all dash characters (-)
															///< - FLM_NO_UNDERSCORE - Treat underscores as white space
															///< - FLM_MIN_SPACES - Ignore leading and trailing white space, and compress consecutive white
															///< space into a single space character
		);

	// Predefined value for no time limit

	#define		FLM_NO_LIMIT		0xFFFF

	/// Parse a query string to set query criteria.
	/// \ingroup querydef
	FLMXPC RCODE FLMAPI FlmParseQuery(
		HFCURSOR				hCursor,					///< Handle to query object whose criteria is to be set.
		F_NameTable *		pNameTable,				///< Name table to use when looking up field names.
		const char *		pszQueryCriteria		///< Query criteria.
		);

	/// Add a field to the query criteria.
	/// \ingroup querydef
	FLMXPC RCODE FLMAPI FlmCursorAddField(
		HFCURSOR		hCursor,							///< Handle to query object.
		FLMUINT		uiFieldNum,						///< Field number that is to be added to query criteria.
		FLMUINT		uiFlags							///< Flags for field.\  Flags may be any of the following ORed
															///< together:\n
															///< - FLM_USE_DEFAULT_VALUE - This specifies that if the field is not found in
															///< a record that is being evaluated, FLAIM should use a default value.\  The
															///< default value is calculated based on the field's data type
															///< - FLM_SINGLE_VALUED - This tells FLAIM that the field is known to never have
															///< more than one value per record.\  This allows FLAIM to evaluate the predicate
															///< using information from an index key, without having to fetch the record to
															///< evaluate all possible values
		);
			
	// Predefined values for uiFlags (see FlmCursorAddField, etc.)

	#define FLM_USE_DEFAULT_VALUE	0x20
	#define FLM_SINGLE_VALUED		0x40
	#define FLM_ROOTED_PATH			0x80
		
	// Predefined values for special fields

	#define FLM_RECID_FIELD		FLM_DRN_TAG
	#define FLM_ANY_FIELD		FLM_WILD_TAG
	#define FLM_WILD				FLM_WILD_TAG

	/// Add a field path to the query criteria.
	/// \ingroup querydef
	FLMXPC RCODE FLMAPI FlmCursorAddFieldPath(
		HFCURSOR			hCursor,						///< Handle to query object.
		FLMUINT *		puiFldPath,					///< Field path that is to be added to query criteria.\  Field path is an array of
															///< zero-terminated field numbers.
		FLMUINT			uiFlags						///< Flags for field.\  See documentation on uiFlags parameter of FlmCursorAddField().
		);
			
	/// Add an application defined predicate to the query criteria.
	/// \ingroup querydef
	FLMXPC RCODE FLMAPI FlmCursorAddUserPredicate(
		HFCURSOR					hCursor,				///< Handle to query object.
		FlmUserPredicate *	pPredicate			///< User defined predicate object.
		);

	/// Typedef for query "get field" function that can be inserted into query criteria wherever a field or field
	/// path could be inserted.   This type of function is passed as a parameter to the FlmCursorAddFieldCB() function.
	typedef RCODE ( * CURSOR_GET_FIELD_CB)(
		void *			pvAppData,						///< Pointer to application data.\   This is the application data that
																///< was passed into FlmCursorAddFieldCB().
		FlmRecord *		pRecord,							///< Record that the field is to be retrieved from.
		HFDB				hDb,								///< Handle to database.
		FLMUINT *		puiFldPath,						///< Field path for the field being requested.
		FLMUINT			uiAction,						///< Specifies which field is to be retrieved.\  May be one of the
																///< following:\n
																///< - FLM_FLD_FIRST - Retrieve first occurrence of the field
																///< - FLM_FLD_NEXT - Retrieve next occurrence of the field
																///< - FLM_FLD_CLEANUP - Free any memory associated with keeping track of
																///< the last field that was returned.\  Basically, if the application data
																///< is being used to keep track of anything and has allocated memory for it,
																///< that memory should be freed
																///< - FLM_FLD_VALIDATE - Validate the puiFldPath that was passed in.\  NOTE:
																///< this option will only be used if the bValidateOnly parameter of the
																///< FlmCursorAddFieldCB() is TRUE
																///< - FLM_FLD_RESET - If the application data is being used to keep track of
																///< anything (like current position), it should set itself to an initialized
																///< state, as if no calls to FLM_FLD_FIRST or FLM_FLD_NEXT have been made
			#define FLM_FLD_FIRST			1
			#define FLM_FLD_NEXT				2
			#define FLM_FLD_CLEANUP			3
			#define FLM_FLD_VALIDATE		4
			#define FLM_FLD_RESET			5
		FlmRecord **	ppFieldRecRV,					///< Returns the pointer to the record that has the field.
		void **			ppFieldRV,						///< Returns the pointer to the field within the record.
		FLMUINT *		puiResult						///< Should only be returned if uiAction == FLM_FLD_VALIDATE.\  Should be one of
																///< the following:\n
																///< - FLM_TRUE - Field path is valid
																///< - FLM_FALSE - Field path is not valid
																///< - FLM_UNK - Cannot determine if field path is valid
		);

	/// Add a field callback function to the query criteria.  This function can be called anywhere an application might call
	/// FlmCursorAddField() or FlmCursorAddFieldPath().  Added a field callback function gives the application more flexibility
	/// in how fields are retrieved from records and validated.  Special processing rules may be applied to determine if
	/// a field is valid.  In addition, the callback function may be used to determine the field to be returned - even fetching
	/// it from other records or external sources.
	/// \ingroup querydef
	FLMXPC RCODE FLMAPI FlmCursorAddFieldCB(
		HFCURSOR					hCursor,					///< Handle to query object.
		FLMUINT *				puiFldPath,				///< Field path.\  This field path is passed into the field callback when it is called.
		FLMUINT					uiFlags,					///< Flags for field.\  See documentation on uiFlags parameter of FlmCursorAddField().
		FLMBOOL					bValidateOnly,			///< Field should only be validated by the callback.\  Instances of the field to be
																///< validated will be found by FLAIM.
		CURSOR_GET_FIELD_CB	fnGetField,				///< Field callback function.
		void *					pvAppData,				///< Pointer to application data.\  This will be passed into the callback function when
																///< it is called.
		FLMUINT					uiUserDataLen			///< Length of data pointed to by pvAppData.\  FLAIM will copy the application data as
																///< needed.\  It uses memcpy to do this, so it should always be a simple structure
																///< instead of a structure with pointers to other structures - because FLAIM will not
																///< know to follow those pointers when copying the data.
		);

	/// Add a query operator to the query criteria.
	/// \ingroup querydef
	FLMXPC RCODE FLMAPI FlmCursorAddOp(
		HFCURSOR			hCursor,							///< Handle to query object.
		QTYPES			eOperator,						///< Operator to be added to the query criteria.
		FLMBOOL			bResolveUnknown = FALSE		///< Resolve comparison operators to TRUE or FALSE even if one of the operands is
																///< unknown.
		);
		
	/// Add a value to the query criteria.  A value is generally added where an operand would appear - such as in a comparison expression.
	/// \ingroup querydef
	FLMXPC RCODE FLMAPI FlmCursorAddValue(
		HFCURSOR			hCursor,							///< Handle to query object.
		QTYPES			eValType,						///< Type of value being added to the query criteria.
		void *			pVal,								///< Pointer to the value being added.\  This should point to a value that corresponds to the type
																///< specified in eValType.
		FLMUINT			uiValLen							///< For binary values, this will contain the value length.\  It is not used for any other type of
																///< value.\  String values are expected to be null-terminated.
		);

	/// Finalize and validate query syntax.  After this function has been called, no more query criteria may be added.
	/// \ingroup querydef
	FLMXPC RCODE FLMAPI FlmCursorValidate(
		HFCURSOR			hCursor							///< Handle to query object.
		);

	/// Startup FLAIM database system.
	/// \ingroup startupshutdown
	FLMXPC RCODE FLMAPI FlmStartup( void);

	/// Shutdown FLAIM database system.
	/// \ingroup startupshutdown
	FLMXPC void FLMAPI FlmShutdown( void);

	/// Database system configuration options that are passed into FlmConfig() and FlmGetConfig().
	typedef enum
	{
		/// FlmConfig().\  Close all files that have not been used for the specified number of seconds.\ \n
		/// Input: pvValue1 is (FLMUINT), unused seconds.
		FLM_CLOSE_UNUSED_FILES,

		/// FlmConfig().\  Close all available file handles as well as all used files as as they are
		/// available.\   Files opened after this call will not be immediately closed after use.
		FLM_CLOSE_ALL_FILES,

		/// FlmConfig().\  Set maximum number of file handles.\ \n
		/// Input: pvValue1 is (FLMUINT), maximum file handles.\ \n
		/// FlmGetConfig().\   Returns maximum number of files handles.\ \n
		/// Output: pvValue is (FLMUINT *), returns maximum file handles.
		FLM_OPEN_THRESHOLD,
		
		/// FlmGetConfig().\   Returns number of open file handles.\ \n
		/// Output: pvValue is (FLMUINT *), returns number of open file handles.
		FLM_OPEN_FILES,
		
		/// FlmConfig().\   Set maximum cache size in bytes.\ \n
		/// Input: pvValue1 is (FLMUINT), maximum cache size in bytes.\ \n
		/// Input: pvValue2 is (FLMBOOL), pre-allocate cache?
		FLM_CACHE_LIMIT,

		/// FlmConfig().\  Enable/disable cache debugging.\ \n
		/// Input: pvValue1 is (FLMBOOL), TRUE=enable, FALSE=disable
		FLM_SCACHE_DEBUG,

		/// FlmConfig().\   Start gathering statistics.
		FLM_START_STATS,

		/// FlmConfig().\  Stop gathering statistics.
		FLM_STOP_STATS,

		/// FlmConfig().\   Reset statistics.
		FLM_RESET_STATS,

		/// FlmConfig().\   Set temporary directory.\ \n
		/// Input: pvValue1 is (char *), name of temporary directory.
		FLM_TMPDIR,

		/// FlmConfig().\   Set maximum seconds between checkpoints.\ \n
		/// Input: pvValue1 is (FLMUINT), maximum seconds between checkpoints.\ \n
		/// FlmGetConfig().\   Get maximum seconds between checkpoints.\ \n
		/// Output: pvValue is (FLMUINT *), maximum seconds between checkpoints.
		FLM_MAX_CP_INTERVAL,

		/// FlmConfig().\  Set BLOB override file extension.\ \n
		/// Input: pvValue1 is (char *), file extension.\ \n
		/// NULL or empty string disables override.\ \n
		/// FlmGetConfig().\   Get BLOB override fle extension.\ \n
		/// Output: pvValue is (char *), at least a 4 byte buffer for extension.
		FLM_BLOB_EXT,

		/// FlmConfig().\   Set maximum transaction time limit.\   Used to determine whether a transaction
		/// should be killed.\ \n
		/// Input: pvValue1 is (FLMUINT), maximum seconds.\ \n
		/// FlmGetConfig().\  Get maximum transaction time limit.\ \n
		/// Output: pvValue is (FLMUINT *), maximum seconds.
		FLM_MAX_TRANS_SECS,

		/// FlmConfig().\   Set maximum time a transaction can be inactive before it will be killed.\ \n
		/// Input: pvValue1 is (FLMUINT), maximum seconds.\ \n
		/// FlmGetConfig().\  Get maximum time a transaction can be inactive before it will be killed.\ \n
		/// Output: pvValue is (FLMUINT *), maximum seconds.
		FLM_MAX_TRANS_INACTIVE_SECS,

		/// FlmConfig().\  Set interval for dynamically adjusting cache limit.\ \n
		/// Input: pvValue1 is (FLMUINT), interval in seconds.\ \n
		/// FlmGetConfig().\   Get interval for dynamically adjusting cache limit.\ \n
		/// Output: pvValue is (FLMUINT *), interval in seconds.
		FLM_CACHE_ADJUST_INTERVAL,

		/// FlmConfig().\  Set interval for dynamically cleaning out old cache blocks and records.\ \n
		/// Input: pvValue1 is (FLMUINT), interval in seconds.\ \n
		/// FlmGetConfig().\   Get interval for dynamically cleaning out old cache blocks and records.\ \n
		/// Output: pvValue is (FLMUINT *), interval in seconds.
		FLM_CACHE_CLEANUP_INTERVAL,

		/// FlmConfig().\  Set interval for cleaning up unused structures.\ \n
		/// Input: pvValue1 is (FLMUINT), interval in seconds.\ \n
		/// FlmGetConfig().\   Get interval for cleaning up unused structures.\ \n
		/// Output: pvValue is (FLMUINT *), interval in seconds.
		FLM_UNUSED_CLEANUP_INTERVAL,

		/// FlmConfig().\  Set maximum time for an item to be unused.\ \n
		/// Input: pvValue1 is (FLMUINT), maximum time in seconds.\ \n
		/// FlmGetConfig().\   Get maximum time for an item to be unused.\ \n
		/// Output: pvValue is (FLMUINT *), maximum time in seconds.
		FLM_MAX_UNUSED_TIME,

		/// FlmConfig().\ Set percentage of cache to be allocated to block cache.\ \n
		/// Input: pvValue1 is (FLMUINT), percent (0 to 100).\ \n
		/// FlmGetConfig().\   Get percentage of cache allocated to block cache.\ \n
		/// Output: pvValue is (FLMUINT *), percent.
		FLM_BLOCK_CACHE_PERCENTAGE,

		/// FlmConfig().\  Enable/disable cache checking.\ \n
		/// Input: pvValue1 is (FLMUINT), 0=disable,other=enable.\ \n
		/// FlmGetConfig().\   Get cache checking state.\ \n
		/// Output: pvValue is (FLMBOOL *), FALSE=disabled,TRUE=enabled.
		FLM_CACHE_CHECK,

		/// FlmConfig().\  Force a database to close all of its files.\ \n
		/// Input: pvValue1 is (char *), name of database.
		FLM_CLOSE_FILE,

		/// FlmConfig().\  Set the logger object.\ \n
		/// Input: pvValue1 is (IF_LoggerClient *), pointer to logger object.\ \n
		/// NULL means disable logging.
		FLM_LOGGER,

		/// FlmConfig().\   Set function pointers for HTTP server.\ \n
		/// Input: pvValue1 is (HTTPCONFIGPARAMS *), pointer to structure
		/// containing HTTP handling functions.
		FLM_ASSIGN_HTTP_SYMS,

		/// FlmConfig().\   Unset function pointers for HTTP server functions to NULL.\  This 
		/// effectively disables the HTTP server.
		FLM_UNASSIGN_HTTP_SYMS,

		/// FlmConfig().\   Set the base URL the HTTP server is going to handle.\ \n
		/// Input: pvValue1 is (char *), base URL to start handling.
		FLM_REGISTER_HTTP_URL,

		/// FlmConfig().\   Unset the base URL the HTTP server was handling.\   Tells the HTTP server to no longer call
		/// our HTTP functions when a request for the specified URL comes in.\ \n
		/// Input: pvValue1 is (char *), base URL to stop handling.
		FLM_DEREGISTER_HTTP_URL,

		/// FlmConfig().\   Invalidate open database handles, forcing the database to (eventually) be closed.\ \n
		/// Input: pvValue1 is (char *), name of database.\ \n
		/// Input: pvValue2 is (char *), name of directory for database data files.\ \n
		/// NOTE: Passing NULL for pvValue1 and pvValue2 will cause all active database handles to be forced to close.
		FLM_KILL_DB_HANDLES,

		/// FlmConfig().\  Set maximum number of queries to save for statistics.\ \n
		/// Input: pvValue1 is (FLMUINT), maximum number of queries to save.\ \n
		/// FlmGetConfig().\   Get maximum number of queries to save for statistics.\ \n
		/// Output: pvValue is (FLMUINT *), maximum number of queries to save.
		FLM_QUERY_MAX,

		/// FlmConfig().\  Set maximum dirty cache.\ \n
		/// Input: pvValue1 is (FLMUINT), maximum dirty cache (bytes).\ \n
		/// Input: pvValue2 is (FLMUINT), low dirty cache (bytes).\ \n
		/// FlmGetConfig().\   Get maximum dirty cache.\ \n
		/// Output: pvValue is (FLMUINT *), maximum dirty cache (bytes).
		FLM_MAX_DIRTY_CACHE,

		/// FlmGetConfig().\   Get whether dynamic cache limits are supported.\ \n
		/// Output: pvValue is (FLMBOOL *), TRUE if dynamic cache limits are supported, FALSE if not.
		FLM_DYNA_CACHE_SUPPORTED,

		/// FlmConfig().\  Set maximum query stratify iterations and query stratify time limit.\ \n
		/// Input: pvValue1 is (FLMUINT), maximum query stratify iterations.\ \n
		/// Input: pvValue2 is (FLMUINT), query stratify time limit (seconds).\ \n
		/// FlmGetConfig().\   Get maximum query stratify iterations.\ \n
		/// Output: pvValue is (FLMUINT *), maximum query stratify iterations.
		FLM_QUERY_STRATIFY_LIMITS,

		/// FlmConfig().\  Enable or disable direct I/O.\ \n
		/// Input: pvValue1 is (FLMBOOL), TRUE = enable, FALSE = disable.\ \n
		/// FlmGetConfig().\   Get direct I/O state.\ \n
		/// Output: pvValue is (FLMBOOL *), TRUE = enabled, FALSE = disabled.
		FLM_DIRECT_IO_STATE
		
	} eFlmConfigTypes;

	// Defaults for certain settable items

	#define DEFAULT_MAX_CP_INTERVAL				180
	#define DEFAULT_MAX_TRANS_SECS				2400
	#define DEFAULT_MAX_TRANS_INACTIVE_SECS	30
	#define DEFAULT_CACHE_ADJUST_PERCENT		70
	#define DEFAULT_CACHE_ADJUST_MIN				(16 * 1024 * 1024)
	#define DEFAULT_CACHE_ADJUST_MAX				0xE0000000
	#define DEFAULT_CACHE_ADJUST_MIN_TO_LEAVE	0
	#define DEFAULT_CACHE_ADJUST_INTERVAL		15
	#define DEFAULT_CACHE_CLEANUP_INTERVAL		15
	#define DEFAULT_UNUSED_CLEANUP_INTERVAL	2
	#define DEFAULT_MAX_UNUSED_TIME				120
	#define DEFAULT_BLOCK_CACHE_PERCENTAGE		50
	#define DEFAULT_FILE_EXTEND_SIZE				(8192 * 1024)
	#define DEFAULT_RFL_FOOTPRINT_SIZE			(100 * 1024 * 1024)
	#define DEFAULT_RBL_FOOTPRINT_SIZE			(100 * 1024 * 1024)
	#define DEFAULT_MAX_STRATIFY_ITERATIONS	10000
	#define DEFAULT_MAX_STRATIFY_TIME			10

	// Levels for block sanity checks.

	#define FLM_NO_CHECK								1
	#define FLM_BASIC_CHECK							2
	#define FLM_INTERMEDIATE_CHECK				3
	#define FLM_EXTENSIVE_CHECK					4

	/// Configure the FLAIM database system.
	/// \ingroup systemconfiguration
	FLMXPC RCODE FLMAPI FlmConfig(
		eFlmConfigTypes	eConfigType,	///< Specified what is to be configured.
		void *				pvValue1,		///< Parameter for configuration - see documentation for ::eFlmConfigTypes for specifics.
		void *				pvValue2			///< Parameter for configuration - see documentation for ::eFlmConfigTypes for specifics.
		);

	/// Get configuration information about the FLAIM database system.
	/// \ingroup systemconfiguration
	FLMXPC RCODE FLMAPI FlmGetConfig(
		eFlmConfigTypes	eConfigType,	///< Configuration information to be retrieved.
		void *				pvValue			///< Configuration information is returned here - see documentation for ::eFlmConfigTypes for
													///< what will be returned for each configuration type.
		);

	/// Set dynamic cache limit.
	/// \ingroup cacheconfiguration
	FLMXPC RCODE FLMAPI FlmSetDynamicMemoryLimit(
		FLMUINT			uiCacheAdjustPercent,	///< Percent of available memory to set cache limit to.
		FLMUINT			uiCacheAdjustMin,			///< Minimum cache limit (bytes) to allow.
		FLMUINT			uiCacheAdjustMax,			///< Maximum cache limit (bytes) to allow.
		FLMUINT			uiCacheAdjustMinToLeave	///< Minumum memory that must be left after setting cache limit.
															///< NOTE: This is an alternative to setting maximum. This parameter
															///< is ignored if uiCacheAdjustMax is non-zero.
		);

	/// Set hard cache limit.
	/// \ingroup cacheconfiguration
	FLMXPC RCODE FLMAPI FlmSetHardMemoryLimit(
		FLMUINT			uiPercent,					///< If non-zero, the hard limit is calculated as a percentage of either available memory
															///< or total physical memory.\  If zero, the uiMax parameter is the hard limit.
		FLMBOOL			bPercentOfAvail,			///< Only used if uiPercent is non-zero.\  If TRUE, the limit is calculated as a percentage
															///< of currently available memory.\  If FALSE, the limit is calculated as a percentage of
															///< total physical memory.
		FLMUINT			uiMin,						///< Only used if uiPercent is non-zero.\  This is the minimum hard limit that can be set.\  If
															///< the calculated hard limit is less than this, it will be adjusted up to this minimum.
		FLMUINT			uiMax,						///< If uiPercent is zero, this is the hard limit to set.\  Otherwise, this is the maximum
															///< limit that should be set.\  If the limit is calculated as a percentage of either available
															///< memory or total physical memory, and it is over this maximum, it will be adjusted down
															///< to this maximum.\  NOTE: When a calculation is being done, FLAIM will first adjust down
															///< to the maximum, and then, if necessary, up to the minimum.
		FLMUINT			uiMinToLeave,				///< Only used if uiPercent is non-zero and uiMax is zero.\  In that scenario, the hard limit
															///< is being calculated, but no maximum limit was specified, so uiMinToLeave is used to
															///< calculate a maximum.\  The maximum will be calculated as the available memory (if
															///< bPercentOfAvail is TRUE) or total physical memory (if bPercentOfAvail is FALSE) minus
															///< uiMinToLeave.
		FLMBOOL			bPreallocate = FALSE		///< Preallocate all of the memory once the limit is calculated.
		);

	/// Get cache information.
	/// \ingroup cacheconfiguration
	FLMXPC void FLMAPI FlmGetMemoryInfo(
		FLM_MEM_INFO *	pMemInfo						///< Memory information is returned here.
		);

	/// Get information on background threads in the FLAIM database system.
	/// \ingroup systemconfiguration
   // JMC - FIXME: FTK provides a function of the same name - modify one of them
   //    so that we can export this one "C" rather than "C++"
	FLMEXP RCODE FLMAPI FlmGetThreadInfo(
		F_Pool *				pPool,					///< Memory pool for allocating memory.\  This pool is used to allocate the structures
															///< and other buffers that will contain the thread information.\  To free all of the
															///< information, the application only needs to call GedPoolFree().
		F_THREAD_INFO **	ppThreadInfo,			///< Pointer to array of thread information structures is returned here.\  The memory
															///< for these structures is allocated from the memory pool.
		FLMUINT *			puiNumThreads,			///< Number of structures in the array is returned here.
		const char *		pszUrl = NULL			///< URL to use to send thread information request to a remote system (via TCP).\  This
															///< allows information to be collected from a remote FLAIM database system, as opposed to
															///< the local FLAIM system running inside the process space.
		);

	/// Free memory that was allocated by various functions.
	/// \ingroup memoryalloc
	FLMXPC void FLMAPI FlmFreeMem(
		void *		pMem				///< Pointer to memory to be freed.
		);

	/****************************************************************************
											Statistics
	****************************************************************************/

	/// Structure used in gathering statistics to hold a operation count, a byte count, and an elapsed time.  This
	/// is typically used for I/O operations where it is useful to know the number of bytes that were read or
	/// written by the operation.
	typedef struct
	{
		FLMUINT64	ui64Count;							///< Number of times operation was performed.
		FLMUINT64	ui64TotalBytes;					///< Total number of bytes involved in the operations.\  This usually represents
																///< bytes read from or written to disk.
		FLMUINT64	ui64ElapMilli;						///< Total elapsed time (milliseconds) for the operations.
	} DISKIO_STAT;

	/// Statistics for read transactions.
	typedef struct
	{
		F_COUNT_TIME_STAT	CommittedTrans;			///< Statistics for read transactions committed.
		F_COUNT_TIME_STAT	AbortedTrans;				///< Statistics for read transactions aborted.
		F_COUNT_TIME_STAT	InvisibleTrans;			///< Statistics for invisible read transactions.
	} RTRANS_STATS;

	/// Statistics for update transactions.
	typedef struct
	{
		F_COUNT_TIME_STAT	CommittedTrans;			///< Statistics for update transactions committed.
		F_COUNT_TIME_STAT	GroupCompletes;			///< Statistics for number of times multiple transactions were committed together.
		FLMUINT64			ui64GroupFinished;		///< Total update transactions that were committed in a group.
		F_COUNT_TIME_STAT	AbortedTrans;				///< Statistics for update transactions aborted.
	} UTRANS_STATS;

	/// Statistics for block reads and writes.
	typedef struct
	{
		DISKIO_STAT		BlockReads;						///< Statistics on block reads.
		DISKIO_STAT		OldViewBlockReads;			///< Statistics on block reads that resulted in an old view error.
		FLMUINT			uiBlockChkErrs;				///< Number of times we had errors checking the block after it
																///< was read in - either checksum errors or other problems
																///< validating data in the block.
		FLMUINT			uiOldViewBlockChkErrs;		///< Number of times we had errors checking the block after it
																///< was read in - either checksum errors or other problems
																///< validating data in the block.\  This statistic is for
																///< older versions of a block as opposed to the current version.
		FLMUINT			uiOldViewErrors;				///< Number of times we had an old view error when reading blocks.
		DISKIO_STAT		BlockWrites;					///< Statistics on block writes.
	} BLOCKIO_STATS;

	/// Statistics gathered for a particular logical file (index or container).
	typedef struct
	{
		FLMBOOL			bHaveStats;						///< Flag indicating whether or not there are statistics
																///< for this logical file.
		FLMUINT			uiLFileNum;						///< Logical file number.
		FLMUINT			uiFlags;							///< Flags for logical file.\  These may be ORed together, and are
																///< as follows:\n
																///< - LFILE_IS_INDEX - If set, specifies that the logical file is an index.\ If not
																///< set, specifies that the logical file is a container or the type is unknown.\ If the
																///< logical file's type is unknown, then LFILE_TYPE_UNKNOWN bit will be set
																///< - LFILE_TYPE_UNKNOWN - Type of the logical file is not known
																///< - LFILE_LEVEL_MASK - The the number of levels in the logical file's b-tree is contained
																///< in the lower four bits of the flags.\  This mask (0xF) allows an application to mask
																///< out the other bits to retrieve the level
	#define					LFILE_IS_INDEX			0x80
	#define					LFILE_TYPE_UNKNOWN	0x40
	#define					LFILE_LEVEL_MASK		0x0F	
		BLOCKIO_STATS	RootBlockStats;				///< Block I/O statistics for the logical file's root blocks.
		BLOCKIO_STATS	MiddleBlockStats;				///< Block I/O statistics for for the blocks in the logical file that are not
																///< root blocks or leaf blocks.
		BLOCKIO_STATS	LeafBlockStats;				///< Block I/O statistics for the logical file's leaf blocks.
		FLMUINT64		ui64BlockSplits;				///< Number of block splits that have occurred in this logical file.
		FLMUINT64		ui64BlockCombines;			///< Number of block combines that have occurred in this logical file.
	} LFILE_STATS;

	/// Database statistics.
	typedef struct
	{
		const char *		pszDbName;					///< Name of database these statistics are for.
		FLMBOOL				bHaveStats;					///< Flag indicating whether or not there are statistics for this database.
		RTRANS_STATS		ReadTransStats;			///< Read transaction statistics for the database.
		UTRANS_STATS		UpdateTransStats;			///< Update transaction statistics for the database.
		FLMUINT64			ui64NumCursors;			///< Number of times a query object was created for this database.\  This is the
																///< number of times FlmCursorInit() was called.
		FLMUINT64			ui64NumCursorReads;		///< Number of query operations that have been performed on this database.\  This
																///< includes counts for FlmCursorFirst(), FlmCursorLast(), FlmCursorNext(),
																///< FlmCursorPrev(), and FlmCursorCurrent().
		F_COUNT_TIME_STAT	RecordAdds;					///< Number of record add operations (FlmRecordAdd()) that have been performed on
																///< this database.
		F_COUNT_TIME_STAT	RecordDeletes;				///< Number of record delete operations (FlmRecordDelete()) that have been performed
																///< on this database.
		F_COUNT_TIME_STAT	RecordModifies;			///< Number of record modify operations (FlmRecordModify()) that have been performed
																///< on this database.
		FLMUINT64			ui64NumRecordReads;		///< Number of record read operations (FlmRecordRetrieve()) that have been performed
																///< on this database.
		FLMUINT				uiLFileAllocSeq;			///< Allocation sequence number for pLFileStats array - used internally.
		LFILE_STATS *		pLFileStats;				///< Logical file statistics for this database.
		FLMUINT				uiLFileStatArraySize;	///< Number of logical files in the pLFileStats array - used internally.
		FLMUINT				uiNumLFileStats;			///< Number of elements in the pLFileStats array currently in use.
		BLOCKIO_STATS		LFHBlockStats;				// Block I/O statistics for LFH blocks.
		BLOCKIO_STATS		AvailBlockStats;			// Block I/O statistics for AVAIL blocks.

		// Write statistics

		DISKIO_STAT			LogHdrWrites;				///< Statistics for writes to the database's log header.
		DISKIO_STAT			LogBlockWrites;			///< Statistics for writes of blocks to the rollback log.
		DISKIO_STAT			LogBlockRestores;			///< Statistics for writing of blocks from the rollback log back into data files (done
																///< during database recovery or when aborting a transaction).

		// Read statistics

		DISKIO_STAT			LogBlockReads;				///< Statistics on reading blocks from the rollback log.
		FLMUINT				uiLogBlockChkErrs;		///< Number of times we had checksum errors reading blocks from the rollback log.
		FLMUINT				uiReadErrors;				///< Number of times we got read errors.
		FLMUINT				uiWriteErrors;				///< Number of times we got write errors.

		// Lock statistics
		
		F_LOCK_STATS		LockStats;					///< Database lock statistics

	} DB_STATS;

	/// FLAIM statistics returned from FlmGetStats().
	typedef struct
	{
		F_MUTEX			hMutex;							///< Mutex for controlling access to this structure - only used internally.
		DB_STATS *		pDbStats;						///< Pointer to array of database statistics.\  There will be a ::DB_STATS
																///< structure for every database currently open.
		FLMUINT			uiDBAllocSeq;					///< Allocation sequence number for the pDbStats array - only used internally.
		FLMUINT			uiDbStatArraySize;			///< Size of the pDbStats array.
		FLMUINT			uiNumDbStats;					///< Number of elements in the pDbStats array that are currently in use.
		FLMBOOL			bCollectingStats;				///< Flag indicating whether or not we are currently collecting statistics.
		FLMUINT			uiStartTime;					///< Time we started collecting statistics.\  This is GMT time - seconds since
																///< January 1, 1970 midnight.
		FLMUINT			uiStopTime;						///< Time we stopped collecting statistics.\  This is GMT time - seconds since
																///< January 1, 1970 midnight.
	} FLM_STATS;

	/// Get statistics.  This function will allocate memory to return statistics.  FlmFreeStats() should
	/// be called to free that memory once the application has processed the statistics.
	/// \ingroup stats
	FLMXPC RCODE FLMAPI FlmGetStats(
		FLM_STATS *	pFlmStats	///< Statistics are returned here.
		);

	/// Free statistics.  This function should be called to free whatever memory was allocated
	/// to retrieve statistics when FlmGetStats() was called.
	/// \ingroup stats
	FLMXPC void FLMAPI FlmFreeStats(
		FLM_STATS *	pFlmStats	///< Statistics to be freed.
		);

	/// Event categories that an application can register to catch - see FlmRegisterForEvent().
	typedef enum
	{
		F_EVENT_LOCKS,					///< Catch all database lock events.
		F_EVENT_UPDATES,				///< Catch all transaction and update event events.
		F_EVENT_SIZE					///< Catch all size threshold events
	} FEventCategory;

	/// Types of events returned to registered event handling functions.  See FlmRegisterForEvent()
	/// for information on how to register an event handling function.
	typedef enum
	{
		F_EVENT_LOCK_WAITING,		///< Thread waiting for lock to be granted, pvEventData1 == database handle, pvEventData2 == thread id.
		F_EVENT_LOCK_GRANTED,		///< Lock granted to thread, pvEventData1 == database handle, pvEventData2 == thread id.
		F_EVENT_LOCK_SUSPENDED,		///< Thread suspended waiting for lock to be granted, pvEventData1 == database handle, pvEventData2 == thread id.
		F_EVENT_LOCK_RESUMED,		///< Lock granted to suspended thread, thread resumed, pvEventData1 == database handle, pvEventData2 == thread id.
		F_EVENT_LOCK_RELEASED,		///< Lock released by thread, pvEventData1 == database handle, pvEventData2 == thread id.
		F_EVENT_LOCK_TIMEOUT,		///< Thread timed out waiting for lock, pvEventData1 == database handle, pvEventData2 == thread id.
		F_EVENT_BEGIN_TRANS,			///< Transaction started, pvEventData1 == FLM_TRANS_EVENT *.
		F_EVENT_COMMIT_TRANS,		///< Transaction committed, pvEventData1 == FLM_TRANS_EVENT *.
		F_EVENT_ABORT_TRANS,			///< Transaction aborted, pvEventData1 == FLM_TRANS_EVENT *.
		F_EVENT_ADD_RECORD,			///< Record added to database, pvEventData1 == FLM_UPDATE_EVENT *.
		F_EVENT_MODIFY_RECORD,		///< Record modified in database, pvEventData1 == FLM_UPDATE_EVENT *.
		F_EVENT_DELETE_RECORD,		///< Record deleted from database, pvEventData1 == FLM_UPDATE_EVENT *.
		F_EVENT_RESERVE_DRN,			///< DRN reserved in database, pvEventData1 == FLM_UPDATE_EVENT *.
		F_EVENT_INDEXING_COMPLETE, ///< Background indexing status, pvEventData1 (FLMUINT) == index number, 
											///< pvEventData2 (FLMUINT) == last drn indexed, 
											///< if zero indexing is complete and the index is now online.
		F_EVENT_RFL_SIZE				///< RFL size threshold has been exceeded, pvEventData1 == FLM_RFL_SIZE_EVENT *.
	} FEventType;

	typedef void *					HFEVENT;	///< Handle returned from FlmRegisterForEvent() - needed when calling FlmDeregisterForEvent().
	#define HFEVENT_NULL			NULL

	/// Typedef for function that is passed into FlmRegisterForEvent().
	typedef void (* FEVENT_CB)(
		FEventType	eEventType,			///< Type of event that occurred.
		void *		pvAppData,			///< Application data that was originally passed into FlmRegisterForEvent().
		void *		pvEventData1,		///< Event specific data.  See documentation for ::FEventType for information one what is returned in this parameter.
		void *		pvEventData2		///< Event specific data.  See documentation for ::FEventType for information one what is returned in this parameter.
		);

	/// Register to catch events from the database system.
	/// \ingroup event
	FLMXPC RCODE FLMAPI FlmRegisterForEvent(
		FEventCategory	eCategory,		///< Category of events to be caught.
		FEVENT_CB		fnEventCB,		///< Function to be called when events of the specified category happen.
		void *			pvAppData,		///< Application supplied data that is to be passed to the registered function whenever it is called.
		HFEVENT *		phEventRV		///< Event handle.  This handle should be passed to FlmDeregisterForEvent() to deregister event handling.
		);

	/// Deregister event handling function.
	/// \ingroup event
	FLMXPC void FLMAPI FlmDeregisterForEvent(
		HFEVENT *	phEventRV			///< Event handle that was returned by FlmRegisterForEvent().
		);

	/// Function prototype for the commit function that can be set by calling FlmDbConfig() using the
	/// eDbConfigType::FDB_SET_COMMIT_CALLBACK option.
	typedef void (* COMMIT_FUNC)(
		HFDB				hDb,					///< Database handle passed into function when it is called.
		void *			pvUserData			///< Application data passed into function when it is called.\  This pointer is the data pointer
													///< that was passed into FlmDbConfig() when it was called with the
													///< eDbConfigType::FDB_SET_COMMIT_CALLBACK option.
		);

	/// Record validator function.  A record validator function has several uses.
	/// It is used in a query to allow an application to apply tests to records
	/// that are not easily expressed with FLAIM's query criteria syntax.
	/// The REC_VALIDATOR_HOOK callback returns an RCODE.  In the case
	/// of an update operation, the RCODE returned by the validator function
	/// is ignored.  However, in the case of a read operation, the
	/// return code is evaluated.  If it is FERR_OK, the current
	/// record is returned to the application.  Otherwise, the record
	/// is not returned, but the read operation is allowed to continue.
	typedef FLMBOOL (* REC_VALIDATOR_HOOK)(
		eFlmFuncs		eFlmFuncId,			///< Function calling the record validator.
		HFDB				hDb,					///< Database handle.
		FLMUINT			uiContainerId,		///< Container number.
		FlmRecord *		pRecord,				///< Record being validated.
		FlmRecord *		pOldRecord,			///< Old record (passed in during modify operations).
		void *			pvAppData,			///< Application data that was specified when the
													///< validator function was registered.
		RCODE *			pRCode				///< Return code to be returned by the FLAIM
													///< which invoked the validator function.
													///< If *pRCode is FERR_OK, the routine which invoked the callback
													///< will continue the current operation.  Otherwise, the operation
													///< will be terminated and *pRCode will be returned to the application.

		);


	/// Indexing callback function.  This function is implemented by an application.  It allows an
	/// application to modify a record after it has been indexed.  This only happens when an index
	/// is first created or when its definition is modified.
	typedef RCODE (* IX_CALLBACK)(
		HFDB				hDb,						///< Database handle.
		FLMUINT			uiIndexNum,				///< Index that was updated.
		FLMUINT			uiContainerNum,		///< Container the record that was indexed belongs to.
		FLMUINT			uiDrn,					///< DRN of the record that was indexed.
		FlmRecord *		pInputRecord,			///< Record that was indexed.
		FlmRecord **	ppModifiedRecord,		///< Modified record is returned here.
		void *			pvAppData				///< Pointer to application data.\  This is the pointer that was passed into
														///< the FlmSetIndexingCallback() function.
		);

	/// Status types returned in the general purpose status callback function (::STATUS_HOOK).
	typedef enum
	{
		FLM_NO_STATUS = 0,
		FLM_INDEXING_STATUS = 2,				///< Reports indexing progress.\   pvParm1 is a FLMUINT that contains the last DRN that was indexed.
		FLM_DELETING_STATUS,						///< Reports progress of deleting an index or container.\  pvParm1 is a FLMUINT that contains the
														///< the number of blocks deleted so far.\  pvParm2 is a FLMUINT that contains the database block size.
		FLM_SWEEP_STATUS = 5,					///< Reports the progress of a call to FlmDbSweep().\  pvParm1 is a pointer to a ::SWEEP_INFO
														///< structure.\  pvParm2 is a FLMUINT that indicates why the callback is being called.\  It may
														///< be one of the following:\n
														///< - EACH_CONTAINER - callback happens once for each container that is traversed
														///< - EACH_RECORD - callback happens once for each record that is read
														///< - EACH_FIELD - callback happens for each field in the record
														///< - EACH_CHANGE - callback happens whenever a field definition that was marked as 
														///< check is changed to unused, or whenever a field that was marked as purged is
														///< deleted
		FLM_CHECK_STATUS = 7,					///< Reports status of a database check - called from within FlmDbCheck().\  This is returned to
														///< the callback function that is passed into FlmDbCheck().\   pvParm1 is a pointer to a
														///< ::DB_CHECK_PROGRESS structure.
		FLM_SUBQUERY_STATUS = 13,				///< Reports status of query processing.\  This is returned to the callback function that
														///< is set by calling FlmCursorConfig() with the eCursorConfigType::FCURSOR_SET_STATUS_HOOK
														///< option.\  pvParm1 is a pointer to a ::FCURSOR_SUBQUERY_STATUS structure.\   pvParm2 is a
														///< FLMBOOL that indicates whether a particular subquery is finished processing or not.\  TRUE
														///< is returned if it is finished, FALSE if not.
		FLM_DB_COPY_STATUS = 21,				///< Reports status of a database copy - called from within FlmDbCopy().\  This is returned to the
														///< callback function that is passed into FlmDbCopy().\  pvParm1 is a pointer to a ::DB_COPY_INFO
														///< structure.
		FLM_REBUILD_STATUS = 22,				///< Reports the status of a database rebuild operation - called from within FlmDbRebuild().\   This is
														///< returned to the callback function that is passed into FlmDbRebuild().\   pvParm1 is a pointer to
														///< a ::REBUILD_INFO structure.
		FLM_PROBLEM_STATUS,						///< Reports corruption found by FlmDbCheck().\  This is returned to the callback fnction that
														///< that is passed into FlmDbCheck().\   pvParm1 is a pointer to a ::CORRUPT_INFO structure.\   pvParm2
														///< is a FLMBOOL *.\  If non-NULL, it means that this particular corruption is a logical index
														///< corruption that FlmDbCheck() can probably repair.\  The callback function should return a TRUE
														///< if it wants FlmDbCheck() to repair the corruption, FALSE otherwise.
		FLM_CHECK_RECORD_STATUS,				///< Reports each non-dictionary record found by FlmDbRebuild().\  This is returned to the callback
														///< function that is passed into FlmDbRebuild().\  It is only reported during the phase when
														///< FlmDbRebuild() is collecting dictionary records.\  pvParm1 is a pointer to a ::CHK_RECORD
														///< structure.\  The purpose of this status callback is to allow an application to extract
														///< dictionary information from non-dictionary records and supply them to FLAIM to inject into
														///< the dictionary it is rebuilding.\  This allows an application a backup mechanism for creating
														///< needed dictionary records if those records cannot be recovered from the dictionary container.
		FLM_EXAMINE_RECORD_STATUS,				///< Reports each non-dictionary record that is recovered by FlmDbRebuild().\  This is returned to
														///< the callback function that is passed into FlmDbRebuild().\  It is only reported during the
														///< phase when FlmDbRebuild() is recovering non-dictionary records.\  pvParm1 is a pointer to
														///< a ::CHK_RECORD structure.\  This callback status gives an applicatoin an opportunity to
														///< examine each non-dictionary record that is recovered - usually so that the application can
														///< collect whatever information it might need to as records are recovered.\  In this way, the
														///< application is not required to make another pass through the recovered records in the rebuilt
														///< database if it wants to extract information from the records that were recovered.
		FLM_DB_UPGRADE_STATUS,					///< Reports the status of a database upgrade operation - called from within FlmDbUpgrade().\  This is
														///< returned to the callback funcgtion that is passed into FlmDbUpgrade().\  pvParm1 is a pointer to a
														///< ::DB_UPGRADE_INFO structure.
		FLM_DB_BACKUP_STATUS,					///< Reports status of a backup - called from within FlmDbBackup().\  This is returned to the
														///< callback function that is passed into FlmDbBackup().\   pvParm1 is a pointer to a
														///< ::DB_BACKUP_INFO structure.
		FLM_DB_RENAME_STATUS,					///< Reports status of a rename - called from within FlmDbRename().\  This is returned to the
														///< callback function that is passed into FlmDbRename().\  pvParm1 is a pointer to a
														///< ::DB_RENAME_INFO structure.
		FLM_DELETING_KEYS,						///< Reports status of removing keys from an index that spans multiple containers.\  This is called
														///< when a container is deleted that the index spans.\  When that happens, all keys in the index
														///< that point to records in the container must be deleted from the index.\  This callback status
														///< reports the progress of removing those keys.\   pvParm1 is a FLMUINT that contains the
														///< index number whose keys are being removed.\   pvParm2 is a FLMUINT that contains the total
														///< number of elements that have been traversed in the index so far.\  The callback function
														///< that is called is the callback function that is registered when FlmSetStatusHook() is called to
														///< set the database handle's ::STATUS_HOOK callback function.
		FLM_REBUILD_ADD_DICT_REC_STATUS		///< Called from within FlmDbRebuild() prior to adding a dictionary record to the dictionary
														///< container in the rebuilt database.\  This gives an application an opportunity to modify
														///< the dictionary record before it is added to the dictionary.\  pvParm1 is a pointer to the
														///< ::FlmRecord object that is to be added to the dictionary.
	} eStatusType;

	/// General purpose status callback function.  This function is implemented by an application.  It allows
	/// an application to receive all kinds of status information from FLAIM during various FLAIM operations.
	typedef RCODE (* STATUS_HOOK)(
		eStatusType		eStatus,					///< Type of status information being reported by FLAIM.
		void *			pvParm1,					///< Status information.\  Type of information returned in this
														///< parameter depends on the eStatus parameter.\   See documentation
														///< on ::eStatusType for details.
		void *			pvParm2,					///< Status information.\  Type of information returned in this
														///< parameter depends on the eStatus parameter.\   See documentation
														///< on ::eStatusType for details.
		void *			pvAppData				///< Pointer to application data.\  This is the pointer that was
														///< passed into whatever function was called to set the callback function.
		);

	/// Configuration options for FlmDbConfig().
	typedef enum
	{
		FDB_SET_APP_VERSION = 3,				///< Set the application version numbers into the database header.\  pvValue1 is a FLMUINT which
														///< holds the major version number.\   pvValue2 is a FLMUINT which holds the minor version number.
		FDB_RFL_KEEP_FILES,						///< Sets flag which specifies whether or not to keep roll-forward log files.\  pvValue1 is a FLMBOOL
														///< which is TRUE or FALSE.
		FDB_RFL_DIR,								///< Set the roll-forward log directory.\   pvValue1 is a char * which contains the name of the
														///< RFL directory.
		FDB_RFL_FILE_LIMITS,						///< Set the minimum and maximum sizes for RFL files.\   pvValue1 is a FLMUINT which contains the
														///< minimum RFL file size (in bytes).\   pvValue2 is a FLMUINT which contains the maximum RFL file
														///< size (in bytes).
		FDB_RFL_ROLL_TO_NEXT_FILE,				///< Force the database to create and start using the next RFL file in the sequence.
		FDB_KEEP_ABORTED_TRANS_IN_RFL,		///< Set flag which specifies whether or not to keep aborted transactions in roll-forward log
														///< files.\   pvValue1 is a FLMBOOL which is TRUE or FALSE.
		FDB_AUTO_TURN_OFF_KEEP_RFL,			///< Set flag which specifies whether or not to automatically turn off keeping of RFL files when
														///< the RFL volume gets full.\  pvValue1 is a FLMBOOL which is TRUE or FALSE.
		FDB_FILE_EXTEND_SIZE,					///< Set the extend size for data files in the database.\  pvValue1 is a FLMUINT which specifies
														///< the extend size (in bytes).\  Whenever data files need to be extended, they will be extended
														///< by this amount.
		FDB_SET_APP_DATA,							///< Allows an application to have the database object remember some data.\  pvValue1 contains a
														///< pointer to the data to be remembered.\  An application may retrieve this pointer at any time
														///< by calling FlmDbGetConfig() with the eDbGetConfigType::FDB_GET_APP_DATA option.
		FDB_SET_COMMIT_CALLBACK,				///< Set a callback function that is to be called whenever this database handle commits an
														///< update transaction.\  pvValue1 is a pointer to the callback
														///< function - a ::COMMIT_FUNC.\   pvValue2 is a pointer to application data that will be passed into the
														///< function whenever it is called.
		FDB_ENABLE_FIELD_ID_TABLE,				///< Enable the creating of a field ID table for level-one fields in cached records for a specific
														///< container.\  pvValue1 is a FLMUINT that holds the container number.\  pvValue2 is a FLMBOOL indicating
														///< whether the field id table is to be enabled or disabled.
		FDB_SET_RFL_SIZE_THRESHOLD,			///< Sets the RFL size threshold (in K bytes).  If registered to receive RFL size events, an event will
														///< be reported when the on-disk size of the RFL exceeds this value.\  pvValue1 is a FLMUINT which
														///< specifies the threshold value in K bytes.
		FDB_SET_RFL_SIZE_EVENT_INTERVALS,	///< Sets the criteria for determining how often to report RFL size events once the RFL exceeds the
														///< size threshold.\  pvValue1 is a FLMUINT which specifies the minimum number of seconds between
														///< events.\  pvValue2 is a FLMUINT  which specifies the minimum increase in K bytes of the RFL
														///< between events.
		FDB_RFL_FOOTPRINT_SIZE,					///< Set the footprint size of the RFL when files are not being kept.\  pvValue1 is a FLMUINT
														///< which specifies the footprint size (in bytes).
		FDB_RBL_FOOTPRINT_SIZE					///< Set the footprint size of the roll-back log.\  pvValue1 is a FLMUINT
														///< which specifies the footprint size (in bytes).
	} eDbConfigType;

	/// Options for FlmDbGetConfig().
	typedef enum
	{
		FDB_GET_VERSION = 1,						///< Get the FLAIM database version.\  pvValue1 is a FLMUINT * that returns database version.
		FDB_GET_BLKSIZ,							///< Get the database block size.\  pvValue1 is a FLMUINT * that returns database block size.
		FDB_GET_DEFAULT_LANG,					///< Get the default language for the database.\  pvValue1 is a FLMUINT * that returns the language.
		FDB_GET_PATH = 17,						///< Get the database file name.\  pvValue1 is a char * that points to a buffer where the
														///< file name is to be returned.\  Buffer should be large enough to hold the largest possible file name.
		FDB_GET_TRANS_ID,							///< Get the current transaction ID for the database.\  pvValue1 is a FLMUINT * that returns the
														///< transaction ID.\  NOTE: If no transaction is active, this option will return the last committed
														///< update transaction ID if this database handle has the database locked.\  Otherwise it will return zero.
		FDB_GET_CHECKPOINT_INFO,				///< Get the current state of the checkpoint thread.\  pvValue1 is a pointer to a ::CHECKPOINT_INFO
														///< structure where the checkpoint information is to be returned.
		FDB_GET_LOCK_HOLDER,						///< Get the current lock holder for the database.\  pvValue1 is a pointer to a ::F_LOCK_USER structure
														///< where the lock information is to be returned.
		FDB_GET_LOCK_WAITERS,					///< Get the entire list of threads that are either holding the lock on the database or are waiting
														///< to obtain the lock on the database.\   pvValue1 is a ::F_LOCK_USER **.\  This option will allocate
														///< an array of ::F_LOCK_USER structures and return a pointer to them.\   The zeroeth element of the
														///< array contains the lock holder.\   All other elements contain lock waiters.\  The last element
														///< in the array will be zeroed out.\  NOTE: The memory allocated by this function should be freed
														///< by calling FlmFreeMem().
		FDB_GET_LOCK_WAITERS_EX,				///< Get the lock holders and waiters using a ::IF_LockInfoClient object.\  pvValue1 is a pointer to the
														///< ::IF_LockInfoClient object that is to be used.
		FDB_GET_RFL_DIR,							///< Get the directory where RFL files are stored.\  pvValue1 is a char * that points to a buffer
														///< where the file name is to be returned.\  Buffer should be large enough to hold the largest
														///< possible directory name.
		FDB_GET_RFL_FILE_NUM,					///< Get the current RFL file number.\   pvValue1 is a FLMUINT * that returns the file number.
		FDB_GET_RFL_HIGHEST_NU,					///< Get the highest RFL file number that is not currently in use.\  pvValue1 is a FLMUINT * that
														///< returns the file number.
		FDB_GET_RFL_FILE_SIZE_LIMITS,			///< Get the minimum and maximum RFL file sizes.\  pvValue1 is a FLMUINT * that returns the
														///< minimum file size (in bytes).\   pvValue2 is a FLMUINT * that returns the maximum file
														///< file size (in bytes).\  NOTE: These sizes may be set by calling FlmDbConfig() using the
														///< eDbConfigType::FDB_RFL_FILE_LIMITS option.
		FDB_GET_RFL_KEEP_FLAG,					///< Get the flag which tells whether the database is configured to keep RFL files.\  pvValue1 is
														///< a FLMBOOL * which returns a TRUE or FALSE.
		FDB_GET_LAST_BACKUP_TRANS_ID,			///< Get the last backup transaction ID.\  pvValue1 is a FLMUINT * which returns the transaction ID.
		FDB_GET_BLOCKS_CHANGED_SINCE_BACKUP,///< Get the number of blocks in the database that have changes since the last backup.\  pvValue1 is
														///< a FLMUINT * that returns the number of blocks.
		FDB_GET_SERIAL_NUMBER,					///< Get the database serial number.\   pvValue1 is a FLMBYTE * that points to a buffer where the
														///< serial number will be returned.\  The buffer should be at least F_SERIAL_NUM_SIZE bytes.
		FDB_GET_AUTO_TURN_OFF_KEEP_RFL_FLAG,///< Get the flag which tells whether the database is configured to automatically turn off
														///< the keeping of RFL files when the RFL volume gets full.\  pvValue1 is
														///< a FLMBOOL * which returns a TRUE or FALSE.\   NOTE: This flag may be set by calling FlmDbConfig()
														///< using the eDbConfigType::FDB_AUTO_TURN_OFF_KEEP_RFL option.
		FDB_GET_KEEP_ABORTED_TRANS_IN_RFL_FLAG,///< Get the flag which tells whether the database is configured to keep aborted transactions
														///< in the roll-forward log.\  pvValue1 is a FLMBOOL * which returns a TRUE or FALSE.\  NOTE: This
														///< flag may be set by calling FlmDbConfig() using the eDbConfigType::FDB_KEEP_ABORTED_TRANS_IN_RFL option.
		FDB_GET_SIZES,								///< Get database file sizes.\   pvValue1 is a FLMUINT64 * which returns the total size of all
														///< data files.\  pvValue2 is a FLMUINT64 * which returns the total size of all rollback
														///< files.\  pvValue3 is a FLMUINT64 * which returns the total size of all roll-forward log files.
		FDB_GET_FILE_EXTEND_SIZE,				///< Get the database file extend size.\  pvValue1 is a FLMUINT * which returns the file extend size.\   NOTE:
														///< This size may be set by calling FlmDbConfig() using the eDbConfigType::FDB_FILE_EXTEND_SIZE option.
		FDB_GET_APP_DATA,							///< Get the application data pointer that was set using the eDbConfigType::FDB_SET_APP_DATA option
														///< in FlmDbConfig().\   pvValue1 is a void ** which returns the pointer.
		FDB_GET_NEXT_INC_BACKUP_SEQ_NUM,		///< Get the next incremental backup sequence number for the database.\  pvValue1 is a FLMUINT * that
														///< returns the sequence number.
		FDB_GET_DICT_SEQ_NUM,					///< Get the dictionary sequence number.\   pvValue1 is a FLMUINT * that returns the sequence number.
		FDB_GET_FFILE_ID,							///< Get the database's unique ID number.\   pvValue1 is a FLMUINT * that returns the number.
		FDB_GET_MUST_CLOSE_RC,					///< Get the ::RCODE that caused the "must close" flag to be set on the database.\   pvValue1 is
														///< an ::RCODE * that returns the ::RCODE.
		FDB_GET_RFL_FOOTPRINT_SIZE,			///< Get the RFL footprint size.\  pvValue1 is a FLMUINT * which returns the RFL footprint size.\   NOTE:
														///< This size may be set by calling FlmDbConfig() using the eDbConfigType::FDB_RFL_FOOTPRINT_SIZE option.
		FDB_GET_RBL_FOOTPRINT_SIZE				///< Get the roll-back log footprint size.\  pvValue1 is a FLMUINT * which returns the RBL footprint size.\   NOTE:
														///< This size may be set by calling FlmDbConfig() using the eDbConfigType::FDB_RBL_FOOTPRINT_SIZE option.
	} eDbGetConfigType;

	/// Create a new database.
	/// \ingroup dbcreateopen
	FLMXPC RCODE FLMAPI FlmDbCreate(
		const char *		pszDbFileName,			///< Name of database to be created.\  May be full path name or partial path name.
		const char *		pszDataDir,				///< Name of directory where data files are to be created.\  If NULL, data files will be
															///< in the same directory as the main database file - pszDbFileName.
		const char *		pszRflDir,				///< Name of the directory where RFL files are to be created.\  If NULL, RFL files will be
															///< in the same directory as the main database file - pszDbFileName.
		const char *		pszDictFileName,		///< Name of a file containing dictionary definitions that are to be read in and put into
															///< the database's dictionary.\  This is only used if the pszDictBuf parameter is NULL.\  If
															///< both pszDictFileName and pszDictBuf parameters are NULL, the database's dictionary
															///< will not be populated.
		const char *		pszDictBuf,				///< String buffer containing dictionary definitions that are to be put into the database's
															///< dictionary.\  If this parameter is NULL, then pszDictFileName is used.\   If
															///< both pszDictFileName and pszDictBuf parameters are NULL, the database's dictionary
															///< will not be populated.
		CREATE_OPTS *		pCreateOpts,			///< Create options for the database.
		HFDB *				phDb						///< If database is successfully created, a database handle is returned here.\  It is not
															///< necessary to call FlmDbOpen() to get a database handle.
		);

	/// Open a database.
	/// \ingroup dbcreateopen
	FLMXPC RCODE FLMAPI FlmDbOpen(
		const char *		pszDbFileName,			///< Name of database to be opened.\  May be full path name or partial path name.
		const char *		pszDataDir,				///< Name of directory where data files for the database are located.\  If NULL, data files are
															///< assumed to be in the same directory as the main database file - pszDbFileName.
		const char *		pszRflDir,				///< Name of the directory where RFL files are located.\  If NULL, RFL files are
															///< assumed to be in the same directory as the main database file - pszDbFileName.
		FLMUINT				uiOpenFlags,			///< Flags for opening the database.\  They are as
															///< follows:\n
															///< - FO_ALLOW_LIMITED - Allow limited access to database even if the database key cannot
															///< be accessed for some reason.\  It may be that NICI is not available, but the
															///< application would still like to be able to access non-encrypted data
															///< - FO_DONT_RESUME_BACKGROUND_THREADS - Tells FLAIM to NOT restart any indexing background
															///< threads.\  This should only be used when the application does not want modifications made
															///< to the database.\  This flag is only recognized on the first open of the
															///< database.\  If the database has already been opened elsewhere, this flag is ignored
															///< - FO_DONT_REDO_LOG - Don't replay the RFL log to recover transactions.\  This should only
															///< be performed if the application does not want the database to be changed in any way, including
															///< replaying of roll-forward logs.\  NOTE: The checkpoint thread will not be started for this
															///< database if this flag is set.\  This flag is only recognized on the first open of the
															///< database.\  If the database has already been opened elsewhere, this flag is ignored
		const char *		pszPassword,			///< Password for opening the database.\  This parameter is normally NULL.\  It should only
															///< be specified if the database's database key is currently wrapped in a password.
		HFDB *				phDb						///< If database is successfully opened, database handle is returned here.
		);

	// Open flags for FlmDbOpen

	#define FO_DONT_REDO_LOG						0x0040	// Used only in FlmDbOpen
	#define FO_DONT_RESUME_BACKGROUND_THREADS	0x0080	// Used only in FlmDbOpen
	#define FO_ALLOW_LIMITED						0x0400	// Used only in FlmDbOpen

	/// Close a database.
	/// \ingroup dbcreateopen
	FLMXPC RCODE FLMAPI FlmDbClose(
		HFDB *			phDb							///< Pointer to database handle that is to be closed.\   The database handle will be
															///< set back to HFDB_NULL.
		);

	/// Configure an open database.
	/// \ingroup dbconfig
	FLMXPC RCODE FLMAPI FlmDbConfig(
		HFDB				hDb,							///< Database handle of database that is to be configured.
		eDbConfigType	eConfigType,				///< Configuration option.
		void *			pvValue1,					///< Configuration parameter.\  Type of value here depends on the eConfigType parameter.\  See
															///< documentation on ::eDbConfigType for details.
		void *			pvValue2						///< Configuration parameter.\  Type of value here depends on the eConfigType parameter.\  See
															///< documentation on ::eDbConfigType for details.
		);

	/// Get configuration information on an open database.
	/// \ingroup dbconfig
	FLMXPC RCODE FLMAPI FlmDbGetConfig(
		HFDB					hDb,						///< Database handle of database whose configuration information is to be retrieved.
		eDbGetConfigType	eGetDbConfigType,		///< Specifies what information is to be retrieved.
		void *				pvValue1,				///< Information is returned via this parameter.\  Type of value required depends on the
															///< eGetDbConfigType parameter.\  See documentation on ::eDbGetConfigType for details.
		void *				pvValue2 = NULL,		///< Information is returned via this parameter.\  Type of value required depends on the
															///< eGetDbConfigType parameter.\  See documentation on ::eDbGetConfigType for details.
		void *				pvValue3 = NULL		///< Information is returned via this parameter.\  Type of value required depends on the
															///< eGetDbConfigType parameter.\  See documentation on ::eDbGetConfigType for details.
		);

	/// Set indexing callback function.
	/// \ingroup dbconfig
	FLMXPC void FLMAPI FlmSetIndexingCallback(
		HFDB			hDb,								///< Database handle whose indexing callback function is to be set.
		IX_CALLBACK	fnIxCallback,					///< Indexing callback function.
		void *		pvAppData						///< Pointer to application data that will be passed into the callback function when
															///< it is called by FLAIM.
		);

	/// Get indexing callback function.
	/// \ingroup dbconfig
	FLMXPC void FLMAPI FlmGetIndexingCallback(
		HFDB				hDb,							///< Database handle whose indexing callback function is to be retrieved.
		IX_CALLBACK *	pfnIxCallback,				///< Callback function is returned here.\  This is the function that was
															///< set using the FlmSetIndexingCallback() function.
		void **			ppvAppData					///< This returns the pointer to application data that was passed into the
															///< FlmSetIndexingCallback() function when the indexing callback function was set.
		);

	/// Set record validator callback function.
	/// \ingroup dbconfig
	FLMXPC void FLMAPI FlmSetRecValidatorHook(
		HFDB						hDb,						///< Database handle whose record validator function is to be set.
		REC_VALIDATOR_HOOK	fnRecValidatorHook,	///< Record validator callback function.\  If this is NULL, record
																///< validation is disabled.
		void *					pvAppData				///< Pointer to application data that will be passed into the record validator function
																///< when it is called by FLAIM.
		);

	/// Get the record validator callback function.
	/// \ingroup dbconfig
	FLMXPC void FLMAPI FlmGetRecValidatorHook(
		HFDB						hDb,						///< Database handle whose record validator function is to be returned.
		REC_VALIDATOR_HOOK * pfnRecValidatorHook, ///< Record validator function is returned here.\   This is the function that was
																///< set using the FlmSetRecValidatorHook() function.
		void **					ppvAppData				///< This returns the pointer to application data that was passed into the
																///< FlmSetRecValidatorHook() function when the record validator function was set.
		);

	/// Set the general purpose status callback function.
	/// \ingroup dbconfig
	FLMXPC void FLMAPI FlmSetStatusHook(
		HFDB				hDb,								///< Database handle whose general purpose status callback function is to be set.
		STATUS_HOOK		fnStatusHook,					///< General purpose status callback function.\  If this is NULL, the general
																///< purpose status callback is disabled.
		void *			pvAppData						///< Pointer to application data that will be passed into the status callback
																///< function when it is called by FLAIM.
		);

	/// Get the general purpose status callback function.
	/// \ingroup dbconfig
	FLMXPC void FLMAPI FlmGetStatusHook(
		HFDB				hDb,								///< Database handle whose general purpose status callback function is to be returned.
		STATUS_HOOK *	pfnStatusHook,					///< Status callback function is returned here.\  This is the function that was
																///< set using the FlmSetStatusHook() function.
		void **			ppvAppData						///< This returns the pointer to application data that was passed into the
																///< FlmSetStatusHook() function when the status callback function was set.
		);

	/// Retrieve status of an index.
	/// \ingroup indexing
	FLMXPC RCODE FLMAPI FlmIndexStatus(
		HFDB					hDb,					///< Database handle - see FlmDbOpen() or FlmDbCreate().
		FLMUINT				uiIndexNum,			///< Index number to return status on.
		FINDEX_STATUS *	pIndexStatus		///< Index status is returned in structure pointed to.
		);

	/// Retrieve next index.
	/// \ingroup indexing
	FLMXPC RCODE FLMAPI FlmIndexGetNext(
		HFDB			hDb,							///< Database handle - see FlmDbOpen() or FlmDbCreate().
		FLMUINT *	puiIndexNum					///< Index number is returned here.
		);

	/// Suspend an index.
	/// \ingroup indexing
	FLMXPC RCODE FLMAPI FlmIndexSuspend(
		HFDB			hDb,							///< Database handle - see FlmDbOpen() or FlmDbCreate().
		FLMUINT		uiIndexNum					///< Number of index to suspend.
		);

	/// Resume an index.
	/// \ingroup indexing
	FLMXPC RCODE FLMAPI FlmIndexResume(
		HFDB			hDb,							///< Database handle - see FlmDbOpen() or FlmDbCreate().
		FLMUINT		uiIndexNum					///< Number of index to resume.
		);
		
	/// Determine if a return code (RCODE) indicates a corruption.
	/// \ingroup errhandling
	FLMXPC FLMBOOL FLMAPI FlmErrorIsFileCorrupt(
		RCODE			rc								///< Error code to be tested.
		);

	/// Convert a return code (RCODE) into a string.
	/// \ingroup errhandling
	FLMXPC const char * FLMAPI FlmErrorString(
		RCODE	rc			///< Error code that is to be converted to a string.
		);

	/// Types of diagnostic information available from FlmGetDiagInfo().
	typedef enum
	{
		FLM_GET_DIAG_INDEX_NUM = 1,	///< Get the index number.\  pvDiagInfo is a FLMUINT * that returns index number.\  This
												///< diagnostic is available when the RCODE::FERR_NOT_UNIQUE error code is returned.
		FLM_GET_DIAG_DRN,					///< Get the DRN.\  pvDiagInfo is a FLMUINT * that returns DRN.\  This diagnostic is available
												///< after attempting to add or modify a dictionary definition record to the dictionary.\  It is
												///< available for the following error codes:\n
												///< - RCODE::FERR_SYNTAX - dictionary syntax error, returns dictionary definition number
												///< - RCODE::FERR_INVALID_TAG - returns DRN of last valid dictionary record processed
												///< - RCODE::FERR_DUPLICATE_DICT_REC - returns DRN of record with the duplicate ID
												///< - RCODE::FERR_DUPLICATE_DICT_NAME - returns DRN of record with the duplicate name
												///< - RCODE::FERR_ID_RESERVED - returns DRN of reserved ID
												///< - RCODE::FERR_CANNOT_RESERVE_ID - returns DRN of ID that cannot be reserved
												///< - RCODE::FERR_CANNOT_RESERVE_NAME - returns DRN of name that cannot be reserved
												///< - RCODE::FERR_BAD_DICT_DRN - returns DRN of dictionary record that was bad
		FLM_GET_DIAG_FIELD_NUM,			///< Get the field number.\  pvDiagInfo is a FLMUINT * that returns field number.\  This diagnostic is
												///< available after attempting to add or modify a record in the database.\  It is available for
												///< the following error codes:\n
												///< - RCODE::FERR_SYNTAX - dictionary syntax error, returns field number
												///< - RCODE::FERR_BAD_FIELD_NUM - returns bad field number in the record that was being added or modified
		FLM_GET_DIAG_FIELD_TYPE,		///< Get the field type.\  pvDiagInfo is a FLMUINT * that returns field type.\  This
												///< diagnostics is available when the RCODE::FERR_BAD_FIELD_NUM error code is returned from a
												///< record add (FlmRecordAdd()) or record modify (FlmRecordModify()) operation.
		FLM_GET_DIAG_ENC_ID				///< Get the encryption ID.\  pvDiagInfo is a FLMUINT * that returns encryption ID.\  This
												///< diagnostics is available when the RCODE::FERR_PURGED_ENCDEF_FOUND error code is returned.
	} eDiagInfoType;

	/// Get diagnostic information.
	/// \ingroup errhandling
	FLMXPC RCODE FLMAPI FlmGetDiagInfo(
		HFDB					hDb,				///< Database handle.
		eDiagInfoType		eDiagCode,		///< Diagnostic desired.
		void *				pvDiagInfo		///< Diagnostic information returned here.\  See documentation on ::eDiagInfoType for more
													///< detailed information.
		);

	// Defines used for 'uiTransType' parameter

	#define FLM_NO_TRANS				0
	#define FLM_UPDATE_TRANS		1
	#define FLM_READ_TRANS			2

	#define FLM_DONT_KILL_TRANS	0x10
	#define FLM_DONT_POISON_CACHE	0x20

	// Defines used for uiMaxLockWait parameter

	#define FLM_NO_TIMEOUT			0xFF

	/// Begin a transaction on the database.
	/// \ingroup trans
	FLMXPC RCODE FLMAPI FlmDbTransBegin(
		HFDB			hDb,							///< Database handle.
		FLMUINT		uiTransType,				///< Type of transaction to start.\  May be FLM_UPDATE_TRANS or FLM_READ_TRANS.\  The
														///< following flags may also be ORed into the transaction type to get special
														///< behaviors during the transaction:\n
														///< - FLM_DONT_KILL_TRANS - Marks a read transaction as one that cannot be killed by
														///< FLAIM.\  FLAIM will occasionally be forced to kill a long-running read transaction
														///< so that a checkpoint can be finished.\  It is NOT recommended that applications
														///< use this flag
														///< - FLM_DONT_POISON_CACHE - Marks the transaction so that any items the transaction
														///< brings into cache (records or blocks) will not poison cache.\  That is, they will
														///< not be allowed to replace other items.\  If an application has a transaction that
														///< is going to read through all of the records in the database, it would probably
														///< be wise to set this flag - so that it won't poison cache.\   However, generally
														///< it is not necessary to set this flag
		FLMUINT		uiMaxLockWait,				///< Only applicable for update transactions.\  Specifies the maximum number of
														///< seconds to wait to obtain the database lock.\  NOTE: A value of FLM_NO_TIMEOUT
														///< specifies that it should wait forever - until the lock becomes available.
		FLMBYTE *	pucHeader = NULL			///< If non-NULL, the entire log header is returned in this buffer.\  The buffer
														///< should be at least F_TRANS_HEADER_SIZE bytes.
		);

	#define F_TRANS_HEADER_SIZE		2048	// Size of buffer required for pszHeader parameter of FlmDbTransBegin

	/// Commit current transaction (if any) on a database.
	/// \ingroup trans
	FLMXPC RCODE FLMAPI FlmDbTransCommit(
		HFDB			hDb,							///< Database handle.
		FLMBOOL *	pbEmpty = NULL				///< If non-NULL, this returns a flag indicating whether or not the transaction was
														///< empty.\  This is only returned for update transactions.\   If TRUE, it means
														///< that no updates were performed inside the transaction, and hence, the transaction
														///< was not logged to the roll-forward log.\  Furthermore, the transaction ID and
														///< the count of committed transactions will not have been altered.\  In short, it
														///< will be as if the transaction never happened.
		);

	/// Abort current transaction (if any) on a database.
	/// \ingroup trans
	FLMXPC RCODE FLMAPI FlmDbTransAbort(
		HFDB			hDb							///< Database handle.
		);

	/// Get type of current transaction (if any) on a database.
	/// \ingroup trans
	FLMXPC RCODE FLMAPI FlmDbGetTransType(
		HFDB			hDb,							///< Database handle.
		FLMUINT *	puiTransType				///< Transaction type is returned here.\  It will be
														///< one of the following:\n
														///< - FLM_NO_TRANS - No transaction currently active on this database handle
														///< - FLM_UPDATE_TRANS - Update transaction is active on this database handle
														///< - FLM_READ_TRANS - Read transaction is active on this database handle
		);

	/// Get current transaction ID.
	/// \ingroup trans
	FLMXPC RCODE FLMAPI FlmDbGetTransId(
		HFDB			hDb,							///< Database handle.
		FLMUINT *	puiTransID					///< Current transaction ID is returned here.\  If no transaction is currently active,
														///< the function will return RCODE::FERR_NO_TRANS_ACTIVE.
		);

	/// Get number of committed transactions for a database.
	/// \ingroup trans
	FLMXPC RCODE FLMAPI FlmDbGetCommitCnt(
		HFDB			hDb,							///< Database handle.
		FLMUINT *	puiCommitCount				///< Number of transactions that have been committed is returned here.
		);

	/// Lock a database.
	/// \ingroup trans
	FLMXPC RCODE FLMAPI FlmDbLock(
		HFDB				hDb,						///< Database handle.
		eLockType		lockType,				///< Type of lock being requested.
		FLMINT			iPriority,				///< Priority of lock being requested.
		FLMUINT			uiTimeout				///< Specifies the maximum number of seconds to wait to obtain the lock.\  NOTE: A
														///< value of FLM_NO_TIMEOUT specifies that it should wait forever - until the
														///< lock becomes available.
		);

	/// Unlock a database.
	/// \ingroup trans
	FLMXPC RCODE FLMAPI FlmDbUnlock(
		HFDB				hDb						///< Database handle.
		);

	/// Get the type of lock currently in effect on a database (if any).
	/// \ingroup trans
	FLMXPC RCODE FLMAPI FlmDbGetLockType(
		HFDB				hDb,						///< Database handle.
		eLockType *		pLockType,				///< Type of lock currently held returned here.
		FLMBOOL *		pbImplicit				///< Flag indicating if the lock is an implicit lock.\  An implicit lock is one that
														///< FLAIM obtained automatically when it started an update transaction.\  An
														///< implicit lock will be released automatically when the transaction commits or
														///< aborts.\  An explicit lock is one which was obtained by calling FlmDbLock().\  An
														///< explicit lock is released when the application calls FlmDbUnlock().
		);

	/// Perform a checkpoint on the database.
	/// \ingroup trans
	FLMXPC RCODE FLMAPI FlmDbCheckpoint(
		HFDB				hDb,						///< Database handle.
		FLMUINT			uiTimeout				///< Specifies the maximum number of seconds to wait to obtain the database lock.\  An
														///< exclusive lock must be obtained to do a checkpoint.\  NOTE: A value of
														///< FLM_NO_TIMEOUT specifies that it should wait forever - until the
														///< lock becomes available.
		);

	#define FLM_AUTO_TRANS				0x0100	// Value ORed into uiAutoTrans parameter
	#define FLM_DO_IN_BACKGROUND		0x0400	// Value ORed into uiAutoTrans parameter
	#define FLM_DONT_INSERT_IN_CACHE	0x0800	// Value ORed into uiAutoTrans parameter
	#define FLM_SUSPENDED				0x1000	// Value ORed into uiAutoTrans parameter

	/// Add a record to the database.
	/// \ingroup update
	FLMXPC RCODE FLMAPI FlmRecordAdd(
		HFDB			hDb,					///< Database handle.
		FLMUINT		uiContainerNum,	///< Container record is to be added to.
		FLMUINT *	puiDrn,				///< On input, *puiDrn contains the DRN to be assigned to the record.\  If *puiDrn == 0
												///< FLAIM will assign the DRN - it will be one higher than the highest DRN that was
												///< ever assigned in this container.\  In this case, *puiDrn will return the DRN that
												///< was assigned.
		FlmRecord *	pRecord,				///< Record to be added to the database.\  NOTE: After this record has been added to
												///< the database, the object pointed to by pRecord will have been inserted into
												///< FLAIM's record cache (unless the FLM_DONT_INSERT_IN_CACHE flag is set in the
												///< uiAutoTrans parameter).\  Once the object is cached, it is marked as read-only.\   This
												///< prevents it from being altered by the application.\  If the application desires to
												///< use pRecord to make subsequent modifications to the record, it must call the copy()
												///< method (FlmRecord::copy()) to obtain a writeable copy of the object.
		FLMUINT		uiAutoTrans			///< This is a set of flags and a timeout that can be used to accomplish the following
												///< things:\n
												///< - FLM_AUTO_TRANS - Specifies that if there is not already an update transaction going, this
												///< operation should have one auto-started and auto-committed for it.\  If this flag is set,
												///< the lower 8 bits of the parameter is assumed to be the timeout for obtaining the lock.\  A
												///< value of FLM_NO_TIMEOUT may be ORed in to specify that the operation should wait forever
												///< to obtain the database lock
												///< - FLM_DO_IN_BACKGROUND - This flag is only applicable when adding or modifying an
												///< index definition record in the dictionary.\  It specifies that the operation is to be
												///< performed in a background thread after the transaction in which this operation is performed
												///< has been committed.\  If the transaction aborts, no background thread will be launched
												///< - FLM_DONT_INSERT_IN_CACHE - This flag specifies that the record is not to be inserted into
												///< FLAIM's record cache
												///< - FLM_SUSPENDED - This flag is only applicable when adding or modifying an
												///< index definition record in the dictionary.\  It specifies that the index is to be immediately
												///< suspended after it is added or modified.\  This means that the index will not be populated
												///< until it is explicitly resumed by the application (see FlmIndexResume()).
		);

	/// Modify a record in the database.
	/// \ingroup update
	FLMXPC RCODE FLMAPI FlmRecordModify(
		HFDB			hDb,					///< Database handle.
		FLMUINT		uiContainerNum,	///< Container number record is to be modified in.
		FLMUINT		uiDrn,				///< DRN of record to be modified.
		FlmRecord *	pRecord,				///< Record that is to replace the existing record.\  This is basically a "blind" update - this
												///< record will replace, in its entirety, the existing record.\  NOTE: After this record has been
												///< modified in the database, the object pointed to by pRecord will have been inserted into
												///< FLAIM's record cache (unless the FLM_DONT_INSERT_IN_CACHE flag is set in the
												///< uiAutoTrans parameter).\  Once the object is cached, it is marked as read-only.\   This
												///< prevents it from being altered by the application.\  If the application desires to
												///< use pRecord to make additional modifications to the record, it must call the copy()
												///< method (FlmRecord::copy()) to obtain a writeable copy of the object.
		FLMUINT		uiAutoTrans			///< See documentation for the uiAutoTrans parameter in the FlmRecordAdd() function.
		);

	/// Delete a record from the database.
	/// \ingroup update
	FLMXPC RCODE FLMAPI FlmRecordDelete(
		HFDB			hDb,					///< Database handle.
		FLMUINT		uiContainerNum,	///< Container number the record is to be deleted from.
		FLMUINT		uiDrn,				///< DRN of record to be deleted.
		FLMUINT		uiAutoTrans			///< See documentation for the uiAutoTrans parameter in the FlmRecordAdd() function.\  NOTE:
												///< The only flag that applies to FlmRecordDelete() is the FLM_AUTO_TRANS flag.
		);

	/// Reserve the next available DRN in a database container.  This allows an application to get a DRN before calling
	/// FlmRecordAdd().  It has the same effect as passing a zero into FlmRecordAdd(), except that no record is added
	/// to the database.  The DRN returned from this function may then be passed into FlmRecordAdd() to assign the DRN
	/// to the record being added.
	/// \ingroup update
	FLMXPC RCODE FLMAPI FlmReserveNextDrn(
		HFDB			hDb,					///< Database handle.
		FLMUINT		uiContainerNum,	///< Container number the DRN is to be reserved from.
		FLMUINT *	puiDrn				///< The reserved DRN is returned here.
		);

	/// Find an unused DRN in the dictionary.
	/// \ingroup update
	FLMXPC RCODE FLMAPI FlmFindUnusedDictDrn(
		HFDB			hDb,					///< Database handle.
		FLMUINT		uiStartDrn,			///< Beginning of range of DRNs to look for an non-used DRN.
		FLMUINT		uiEndDrn,			///< Ending of range of DRNs to look for a non-used DRN.
		FLMUINT *	puiDrn				///< Unused DRN, if any, is returned here.
		);


	/// Get the name of a dictionary item.
	/// \ingroup dbdict
	FLMXPC RCODE FLMAPI FlmGetItemName(
		HFDB				hDb,					///< Database handle.
		FLMUINT			uiItemId,			///< Dictionary ID whose name is to be returned.
		FLMUINT			uiNameBufSize,		///< Size of pszNameBuf in bytes.\  Buffer should be large enough to hold the
													///< name of the dictionary item plus a null terminating character.
		char *			pszNameBuf			///< Dictionary name is returned here.
		);

	// uiFlag or uiFlags values in FlmRecordRetrieve or FlmKeyRetrieve

	#define FO_INCL		0x10
	#define FO_EXCL		0x20
	#define FO_EXACT		0x40
	#define FO_KEY_EXACT	0x80
	#define FO_FIRST		0x100
	#define FO_LAST		0x200

	/// Find and retrieve a record in a container.
	/// \ingroup retrieval
	FLMXPC RCODE FLMAPI FlmRecordRetrieve(
		HFDB				hDb,					///< Database handle.
		FLMUINT			uiContainerNum,	///< Container the record is to be retrieved from.
		FLMUINT			uiDrn,				///< DRN of record to be retrieved.\  NOTE: The actual record retrieved depends on
													///< the uiFlag parameter as well.
		FLMUINT			uiFlag,				///< Flag that is used in conjunction with the uiDrn parameter to determine the
													///< record that should be retrieved.\  Flag may be one of the following:\n
													///< - FO_INCL - If the record specified by uiDrn is not found, find the record
													///< with the next highest DRN after it
													///< - FO_EXCL - Retrieve the record whose DRN is the next highest after the DRN
													///< specified in uiDrn
													///< - FO_EXACT - Retrieve the record whose DRN is specified by uiDrn.\  If there
													///< is no record with that DRN, the function should return RCODE::FERR_NOT_FOUND
													///< - FO_FIRST - Retrieve the first record in the container - the record with
													///< the lowest DRN.\  The uiDrn parameter is ignored if this flag is passed in
													///< - FO_LAST - Retrieve the last record in the container - the record with
													///< the highest DRN.\  The uiDrn parameter is ignored if this flag is passed in
		FlmRecord **	ppRecord,			///< If non-NULL, pointer to found record object is returned here.
		FLMUINT *		puiDrn				///< If non-NULL, DRN of found record is returned here.
		);

	/// Find and retrieve a key in an index.
	/// \ingroup retrieval
	FLMXPC RCODE FLMAPI FlmKeyRetrieve(
		HFDB				hDb,					///< Database handle.
		FLMUINT			uiIndex,				///< Index the key is to be retrieved from.
		FLMUINT			uiContainerNum,	///< If the index is a cross-container index, this may be used to specify a particular
													///< container the found key should be pointing to.\  A value of zero indicates that
													///< any container will do.
		FlmRecord *		pSearchKey,			///< Key to be searched for.\  NOTE: The actual key retrieved depends on the uiFlags
													///< parameter as well.
		FLMUINT			uiSearchDrn,		///< DRN in the key's reference set that is to be searched for.\  If a zero is passed in
													///< this parameter, only a key search is done, not a key+reference search.
		FLMUINT			uiFlags,				///< Flags used in conjunction with the pSearchKey and uiSearchDrn parameters to determine the
													///< key/DRN that should be retrieved.\  Flags may be ORed together and are as follows:\n
													///< - FO_INCL - Return either the exact key+reference specified by pSearchKey and uiSearchDrn or
													///< the next key+reference after.\  NOTE: This flag may be used in conjunction with the FO_KEY_EXACT
													///< flag - see documentation below
													///< - FO_EXCL - Retrieve the key+reference that comes after the key+reference specified
													///< by pSearchKey and uiSearchDrn.\  NOTE: This flag may be used in conjunction with the FO_KEY_EXACT
													///< flag - see documentation below
													///< - FO_KEY_EXACT - This flag is used in conjunction with the FO_INCL and FO_EXCL flags.\  It
													///< specifies that FlmKeyRetrieve() is NOT to go to the next key in the index, but confine itself
													///< to references for the key specified in pSearchKey.\  If there are no more references for the
													///< key specified in pSearchKey, FlmKeyRetrieve() will return RCODE::FERR_EOF_HIT, even if there are
													///< keys that come after pSearchKey
													///< - FO_EXACT - Retrieve the exact key+reference specified by pSearchKey and uiSearchDrn.\  If there
													///< is no such key+reference, the function should return RCODE::FERR_NOT_FOUND
													///< - FO_FIRST - Retrieve the first key+reference in the index.\  If this flag is passed in, all
													///< other flags will be ignored, as will pSearchKey and uiSearchDrn
													///< - FO_LAST - Retrieve the last key+reference in the index.\  If this flag is passed in, all
													///< other flags will be ignored except for FO_FIRST (which takes precedence over FO_LAST if they
													///< are both set), as will pSearchKey and uiSearchDrn
		FlmRecord **	ppFoundKey,			///< If non-NULL, found key is returned here.
		FLMUINT *		puiFoundDrn			///< If non-NULL, found reference (DRN) is returned here.
		);

	/// Types of backups supported by FLAIM.  This type is passed into FlmDbBackupBegin()
	typedef enum
	{
		// These values are stored in the header of the 
		// backup, so do not change their values.
		FLM_FULL_BACKUP = 0,			///< Full backup.
		FLM_INCREMENTAL_BACKUP		///< Incremental backup.
	} FBackupType;

	/// Begin a database backup.
	/// \ingroup dbbackup
	FLMXPC RCODE FLMAPI FlmDbBackupBegin(
		HFDB			hDb,					///< Database handle.
		FBackupType	eBackupType,		///< Type of backup being requested.
		FLMBOOL		bHotBackup,			///< Specifies whether backup should be "hot" or "warm".\  A hot backup is one where the database is
												///< not locked during the backup.\  A "warm" backup is one where the database is locked during
												///< during the backup to prevent any updates from happening.
		HFBACKUP *	phBackup				///< A handle to a database backup object is returned here.\  This object is basically used
												///< to maintain state during the backup.\  It is passed into FlmBackupGetConfig(),
												///< FlmDbBackup(), and FlmDbBackupEnd().
		);

	/// Backup configuration information that can be requested by FlmBackupGetConfig().
	typedef enum
	{
		FBAK_GET_BACKUP_TRANS_ID = 1,				///< Get backup transaction ID.\  pvValue1 is a FLMUINT * that returns the transaction ID.
		FBAK_GET_LAST_BACKUP_TRANS_ID				///< Get the last backup transactioN ID.\  pvValue1 is a FLMUINT * that returns the last
															///< transaction ID.
	} eBackupGetConfigType;

	/// Get backup configuration on a backup that was started by FlmDbBackupBegin.
	/// \ingroup dbbackup
	FLMXPC RCODE FLMAPI FlmBackupGetConfig(
		HFBACKUP					hBackup,				///< Backup handle that was returned from FlmDbBackupBegin().
		eBackupGetConfigType	eConfigType,		///< Type of configuration information being requested.
		void *					pvValue1,			///< Configuration information is returned here.\  See documentation on ::eBackupGetConfigType for
															///< details.
		void *					pvValue2 = NULL	///< Configuration information is returned here.\  See documentation on ::eBackupGetConfigType for
															///< details.
		);


	/// Typedef for callback function that is called from FlmDbBackup() to write out backed up data.  It is this function's
	/// responsibility to write the data to an appropriate backup medium - tape, disk, etc.
	typedef RCODE (* BACKER_WRITE_HOOK)( 
		void *		pvBuffer,			///< Buffer that is to be backed up.
		FLMUINT		uiBytesToWrite,	///< Number of bytes to write out.
		void *		pvAppData			///< Application data that was passed into FlmDbBackup().
		);

	/// Perform a backup that was started by FlmDbBackupBegin.
	/// \ingroup dbbackup
	FLMXPC RCODE FLMAPI FlmDbBackup(
		HFBACKUP					hBackup,				///< Backup handle that was returned from FlmDbBackupBegin().
		const char *			pszBackupPath,		///< This specifieds the directory where FlmDbBackup() is to create a backup file set
															///< for the backed up data.\  The files in the backup set will be named 00000001.64,
															///< 00000002.64, etc.\   This parameter is only used if the fnWrite parameter is NULL.\  If
															///< fnWrite is non-NULL, all backed up data will be passed to that function to be written to
															///< a backup medium.
		const char *			pszPassword,		///< Password used to shroud the database encryption key in the backup.\  If NULL, the database
															///< encryption key will remain wrapped in the NICI local storage key.\  A NULL password means
															///< that the backup can only be restored to the same server the backup was taken from, because
															///< the database key can only be unwrapped using the NICI local storage key of that server.
		BACKER_WRITE_HOOK		fnWrite,				///< This is the callback function that FlmDbBackup() will call to write data to the
															///< backup medium (tape, disk, etc.).\  If NULL, FlmDbBackup() will create a backup
															///< file set in the directory specified by the pszBackupPath parameter.\  If a callback
															///< function is specified, the application will also want to have a corresponding
															///< implementation for the ::F_Restore class so that it can read data back during a
															///< FlmDbRestore() operation.
		STATUS_HOOK				fnStatus,			///< This is a callback function that FlmDbBackup() calls to report backup progress.
		void *					pvAppData,			///< Pointer to application data.\  This pointer will be passed into the fnWrite callback
															///< function as well as the fnStatus callback function whenever they are called.
		FLMUINT *				puiIncSeqNum		///< If the backup is an incremental backup, this returns the incremental backup sequence
															///< number.
		);

	/// End a backup that was started by FlmDbBackupBegin().  This is necessary to free any resources (such as memory) that may have
	/// been allocated during the backup.  This should always be called if FlmDbBackupBegin() is successful, even if FlmDbBackup() is
	/// never called, or if it fails with an error code.
	/// \ingroup dbbackup
	FLMXPC RCODE FLMAPI FlmDbBackupEnd(
		HFBACKUP *				phBackup				/// Pointer to backup handle that is to be freed.
		);

	/// Restore a database from a backup.
	/// \ingroup dbbackup
	FLMXPC RCODE FLMAPI FlmDbRestore(
		const char *			pszDbPath,			///< Name of database FlmDbRestore() is to create from the backup.
		const char *			pszDataDir,			///< Directory where the restored database's data files are to be created.
		const char *			pszBackupPath,		///< Directory where backup file set is located.\   If NULL, the backup data is
															///< read by calling various methods on the ::F_Restore object specified in the
															///< pRestoreObj parameter.\  Otherwise, the backup data is read from files in
															///< this directory that are named 00000001.64, 00000002.64, etc.\  These are
															///< files that would have been created by FlmDbBackup() if it was passed a
															///< backup path instead of a writer callback function.
		const char *			pszRflDir,			///< This is only used if pRestoreObj is NULL and pszBackupPath is non-NULL.\  It
															///< specifies the directory where RFL files are located.\  If possible, FlmDbRestore()
															///< will attempt to replay any RFL files that were created after the backup was
															///< taken.\  NOTE: The RFL files are actually in a subdirectory to the directory
															///< specified in this parameter.\  The subdirectory name is \<dbname\>.rfl, where
															///< dbname is the base name found in pszDbPath.\  For example, the dbname for
															///< abc/xyz.db would be xyz.db.\  Thus, the subdirectory would be xyz.rfl.\  If
															///< pszRflDir is NULL, FlmDbRestore() assumes that the RFL directory is the same
															///< as the directory for pszDbPath.\  \<dbname\>.rfl is still assumed to be the
															///< subdirectory.
		const char *			pszPassword,		///< Password for unshrouding the database key.\  This should be the same password
															///< that was used to create the backup - i.e.\ the password that was passed into
															///< FlmDbBackupBegin().
		F_Restore *				pRestoreObj			///< Pointer to the object whose methods will be called to read data from the
															///< backup medium.\  This object should know how to read data from the backup medium.\  It
															///< should understand whatever formatting was used by the fnWrite callback function
															///< (see FlmDbBackup()) to write the data out to the backup medium.
		);

	/// This is an abstract base class that allows an application to read "unknown" data from the
	/// RFL or to write "unknown" data to the RFL.
	/// The application must implement this class.
	class FLMEXP F_UnknownStream : public F_Object
	{
	public:

		/// Read data for an "unknown" object from the RFL.
		virtual RCODE read(
			FLMUINT			uiLength,				///< Number of bytes to read.
			void *			pvBuffer,				///< Buffer to place read bytes into.
			FLMUINT *		puiBytesRead			///< Number of bytes actually read.
			) = 0;

		/// Write data to an "unknown" object in the RFL.
		virtual RCODE write(
			FLMUINT			uiLength,				///< Number of bytes to write.
			void *			pvBuffer					///< Data to be written out.
			) = 0;

		/// Close the stream.
		/// If this is an input stream (read-only), the object should read to
		/// the end of the stream, discarding any remaining data.
		virtual RCODE close( void) = 0;
	};

	FLMXPC RCODE FLMAPI FlmDbGetUnknownStreamObj(
		HFDB						hDb,
		F_UnknownStream **	ppUnknownStream);

	FLMXPC RCODE FLMAPI FlmDbGetRflFileName(
		HFDB 					hDb,
		FLMUINT				uiFileNum,
		char *				pszFileName);

	/// Structure used to report the progress of a restore operation.  This structure is returned to the
	/// F_Restore::status() method when it is called with the eRestoreStatusType::RESTORE_PROGRESS status code.
	typedef struct
	{
		FLMUINT64		ui64BytesToDo;
		FLMUINT64		ui64BytesDone;
	} BYTE_PROGRESS;

	/// Restore status types reported through the F_Restore::status() method.
	typedef enum
	{
		RESTORE_BEGIN_TRANS = 1,	///< Restoring a FlmDbTransBegin() operation.\  pvValue1 is a FLMUINT that
											///< contains the transaction start time.
		RESTORE_COMMIT_TRANS,		///< Restoring a FlmDbTransCommit() operation.
		RESTORE_ABORT_TRANS,			///< Restoring a FlmDbTransAbort() operation.
		RESTORE_ADD_REC,				///< Restoring a FlmRecordAdd() operation.\   pvValue1 is a FLMUINT that contains the
											///< container number.\   pvValue2 is a FLMUINT that contains the DRN.\  pvValue3
											///< is a FlmRecord * that points to the record object to be added.
		RESTORE_DEL_REC,				///< Restoring a FlmRecordDelete() operation.\   pvValue1 is a FLMUINT that contains the
											///< container number.\   pvValue2 is a FLMUINT that contains the DRN.
		RESTORE_MOD_REC,				///< Restoring a FlmRecordModify() operation.\   pvValue1 is a FLMUINT that contains the
											///< container number.\   pvValue2 is a FLMUINT that contains the DRN.\  pvValue3
											///< is a FlmRecord * that points to the modified record object.
		RESTORE_RESERVE_DRN,			///< Restoring a FlmReserveNextDrn() operation.\   pvValue1 is a FLMUINT that contains the
											///< container number.\   pvValue2 is a FLMUINT that contains the DRN that was reserved.
		RESTORE_INDEX_SET,			///< Restoring index set of records operation.\  pvValue1 is a FLMUINT that contains the index
											///< number.\  pvValue2 is a FLMUINT that contains the start DRN to be indexed.\  pvValue3 is a
											///< FLMUINT that contains the end DRN to be indexed.
		RESTORE_PROGRESS,				///< Report restore progress.\  pvValue1 is a ::BYTE_PROGRESS *.
		RESTORE_REDUCE,				///< Restoring a FlmDbReduceSize() operation.\  pvValue1 is a FLMUINT that contains the count
											///< of blocks reduced.
		RESTORE_UPGRADE,				///< Restoring a FlmDbUpgrade() operation.\  pvValue1 is a FLMUINT that contains the old version
											///< being upgraded from.\  pvValue2 is a FLMUINT that contains the new version being upgraded to.
		RESTORE_ERROR,					///< Report an error that occurred during the restore.\  pvValue1 is a ::RCODE that contains the error
											///< which occurred.
		RESTORE_INDEX_SUSPEND,		///< Restoring a FlmIndexSuspend() operation.\  pvValue1 is a FLMUINT that contains the
											///< index number.
		RESTORE_INDEX_RESUME,		///< Restoring a FlmIndexResume() operation.\  pvValue1 is a FLMUINT that contains the
											///< index number.
		RESTORE_BLK_CHAIN_DELETE,	///< Restoring a block chaine delete operation - deleting blocks from an index or container.\  pvValue1 is
											///< a FLMUINT that contains the DRN of the record in the tracker that is being used to track this
											///< operation.\  pvValue2 is a FLMUINT that contains the number of blocks to delete.\  pvValue3 is a 
											///< FLMUINT that contains the block address of the last block that was deleted.
		RESTORE_WRAP_KEY,				///< Restoring a FlmDbWrapKey() operation.\  pvValue1 is a FLMUINT that contains the length of the database key.
		RESTORE_ENABLE_ENCRYPTION,	///< Restoring a FlmEnableEncryption() operation.\  pvValue1 is a FLMUINT that contains the length of
											///< the database key.
		RESTORE_CONFIG_SIZE_EVENT	///< Restoring a FlmSetSizeEventThreshold() operation.\  pvValue1 is a FLMUINT ....
	} eRestoreStatusType;

	/// Actions that an application may want to tell FlmDbRestore() to take during a restore operation.
	/// These action codes are returned from the F_Restore::status() method.
	typedef enum
	{
		RESTORE_ACTION_CONTINUE = 0,	///< Continue restore.
		RESTORE_ACTION_STOP,				///< Abort the restore.
		RESTORE_ACTION_SKIP,				///< Skip the current operation.\  NOTE: FlmDbRestore does not currently
												///< do anything if this code is returned.
		RESTORE_ACTION_RETRY				///< Retry the operation.\  This should only be returned when the
												///< F_Restore::status() method passes an eRestoreStatusType::RESTORE_ERROR
												///< and the application wants FlmDbRestore() to retry whatever it was that
												///< caused the error.
	} eRestoreActionType;

	/// This is an abstract base class for reading backup data during a FlmDbRestore() operation.
	/// This class must be implemented by an application.  The FlmDbRestore() function calls methods
	/// of this class to read backup data from the backup medium.  This object and the write callback
	/// function of FlmDbBackup() (see its fnWrite parameter) allow an application to have complete
	/// control over writing and reading of backup data.  Backup data could be streamed directly to
	/// a tape device, or any other media the application chooses.
	class FLMEXP F_Restore : public F_Object
	{
	public:

		virtual ~F_Restore()
		{
		}

		/// FlmDbRestore() calls this method to give the application an opportunity to open the
		/// backup media, if needed.  Subsequent calls to the F_Restore::read() method should
		/// return data from the backup that is being restored.  When FlmDbRestore() is done
		/// reading from the backup set, it will call the F_Restore::close() method.
		virtual RCODE openBackupSet( void) = 0;

		/// FlmDbRestore() calls this method to tell the application to open an incremental backup.
		/// FlmDbRestore() attempts to restore incremental backups after it is finished restoring the
		/// regular (non-incremental backup).  It will restore incremental backups until this method
		/// returns RCODE::FERR_IO_PATH_NOT_FOUND.  This method should return RCODE::FERR_IO_PATH_NOT_FOUND
		/// if the requested incremental backup does not exist, or if it does not want FlmDbRestore()
		/// to restore any more incremental backups.  If this method returns FERR_OK, subsequent calls
		/// to the F_Restore::read() method are expected to return data from the incremental backup.
		/// When FlmDbRestore() is done reading from the incremental backup, it will call the
		/// F_Restore::close() method.
		virtual RCODE openIncFile(
			FLMUINT	uiIncBackupSeqNum	///< Sequence number of incremental backup that is to be opened.
			) = 0;

		/// FlmDbRestore() calls this method to tell the application to open an RFL file that it
		/// wants to replay to recover transactions.  FlmDbRestore() attempts to restore RFL files
		/// after it is finished restoring the regular backup and any incremental backups.  It will
		/// restore RFL files until this method returns RCODE::FERR_IO_PATH_NOT_FOUND.  This method
		/// should return RCODE_FERR_IO_PATH_NOT_FOUND if the requested RFL file file does not exist,
		/// or if it does not want FlmDbRestore() to restore any more RFL files.  If this method
		/// returns FERR_OK, subsequent calls to the F_Restore::read() method are expected to return
		/// data from the RFL file.  When FlmDbRestore() is done reading from the RFL file, it will
		/// call the F_Restore::close() method.
		virtual RCODE openRflFile(
			FLMUINT			uiRFLFileNum	///< Sequence number of RFL file that is to be opened.
			) = 0;

		/// Read data from the current file.  The current "file" will either be the regular backup
		/// file set, an incremental backup, or an RFL file.
		virtual RCODE read(
			FLMUINT			uiLength,				///< Number of bytes of data to read.
			void *			pvBuffer,				///< Buffer to place read data into.
			FLMUINT *		puiBytesRead			///< Returns the actual number of bytes read.
			) = 0;

		/// Close the current file.  The current "file" will either be the regular backup
		/// file set, an incremental backup, or an RFL file.
		virtual RCODE close( void) = 0;

		/// Abort the current file.  The current "file" will either be the regular backup
		/// file set, an incremental backup, or an RFL file.  The action to be taken
		/// by the application is similar to what it should do for a call to the
		/// F_Restore::close() method.  The subtle distinction lies in the fact that
		/// FlmDbRestore() encountered some kind of error and may want to retry.  For most
		/// implementations, it will be sufficient to simply call the F_Restore::close()
		/// method from within this method.
		virtual RCODE abortFile( void) = 0;

		/// Process an "unknown" object that was encountered while reading an RFL file.
		/// When FlmDbRestore calls this method, it passes an ::F_UnknownStream object
		/// to the application, which allows the application to read the unknown data
		/// from the RFL file and process it as needed.  Unknown data in the RFL
		/// file is data put there by the application as part of a transaction.  It
		/// is generally data that is in some way associated with data in the database,
		/// but is not actually stored in the database.  Even though it is not stored
		/// in the database, it is desireable to restore it during a FlmDbRestore()
		/// operation.
		virtual RCODE processUnknown(
			F_UnknownStream *		pUnkStrm		///< Object that allows the application to read the unknown data.
			) = 0;

		/// FlmDbRestore() calls this method to report restore progress.  An application may abort the
		/// database restore operation by returning RESTORE_ACTION_STOP in *puiAction.
		virtual RCODE status(
			eRestoreStatusType		eStatusType,		///< Restore status types.
			FLMUINT						uiTransId,			///< Transaction id of RFL operation being restored.
			void *						pvValue1,			///< Value for RFL operation being restored - details defined for each
																	///< ::eRestoreStatusType.
			void *						pvValue2,			///< Value for RFL operation being restored - details defined for each
																	///< ::eRestoreStatusType.
			void *						pvValue3,			///< Value for RFL operation being restored - details defined for each
																	///< ::eRestoreStatusType.
			eRestoreActionType *		peRestoreAction	///< Action the application wants FlmDbRestore() to take.
			) = 0;	
	};

	/****************************************************************************
						BLOB Class, Functions and Definitions
	****************************************************************************/

	// FlmBlob::create uiBlobType values

	#define BLOB_UNKNOWN_TYPE	0						// Unknown (binary) type

	// FlmBlob::create uiFlags values

	#define BLOB_OWNED_REFERENCE_FLAG	0x10		// Set when BLOB is owned
	#define BLOB_UNOWNED_REFERENCE_FLAG	0x1000	// Set for BLOB reference

	/// This class provides an interface for handling binary large objects.  Currently, FLAIM only
	/// supports referencing of external files.  BLOB data is not actually stored "in" the database.
	class FLMEXP FlmBlob : public F_Object
	{
	public:

		FlmBlob()
		{
		}

		virtual ~FlmBlob()
		{
		}

		/// Setup the blob object to reference an external file.
		virtual RCODE referenceFile( 
			HFDB					hDb,				///< Database handle.
			const char *		pszFilePath,	///< Name of file the blob is to reference.
			FLMBOOL				bOwned = FALSE	///< Is the external file "owned" by the database?  If TRUE,
														///< when the record referencing the BLOB is deleted, FLAIM
														///< will automatically delete the file.
			) = 0;

		/// Compare the file name refered to by the BLOB object with the passed in file name.  Will return
		/// zero if the file names are equal, non-zero otherwise.
		virtual FLMINT compareFileName(
			const char *	pszFileName			///< File name to compare to.
			) = 0;

		/// Return the file name referred to by the BLOB object.
		virtual RCODE buildFileName(
			char *			pszFileName			///< File name is returned here.
			) = 0;
	};

	/// Allocate a BLOB object that can then be used to create a new BLOB to store in a ::FlmRecord object.
	/// \ingroup update
	FLMXPC RCODE FLMAPI FlmAllocBlob(
		FlmBlob **		ppBlob				///< Pointer to newly allocated BLOB object is returned here.
		);

	/// Categories of messages that FLAIM can log and that an application can request to receive.
	typedef enum
	{
		FLM_QUERY_MESSAGE,			///< Query information is logged using this message type.
		FLM_TRANSACTION_MESSAGE,	///< Transactional messages, including updates.
		FLM_GENERAL_MESSAGE,			///< General category of messages - anything not belonging to the
											///< other message categories.
		FLM_NUM_MESSAGE_TYPES
	} FlmLogMessageType;

	#define F_MAX_NUM_BUF		12
	#define F_MAX_NUM64_BUF		24

	/// Convert a FLMUINT value to FLAIM's internal storage format for numbers.
	/// \ingroup storageconversion
	FLMXPC RCODE FLMAPI FlmUINT2Storage(
		FLMUINT			uiNum,			///< Number to convert.
		FLMUINT *		puiStorageLen,	///< On input, *puiStorageLen is the size of pucStorageBuf.\  It must be atleast F_MAX_NUM_BUF
												///< bytes.\  On output *puiStorageLen is set to the number of bytes used in pucStorageBuf.
		FLMBYTE *		pucStorageBuf	///< Number converted to FLAIM's internal storage format is returned here.
		);

	/// Convert a FLMUINT64 value to FLAIM's internal storage format for numbers.
	/// \ingroup storageconversion
	FLMXPC RCODE FLMAPI FlmUINT64ToStorage(
		FLMUINT64		ui64Num,			///< Number to convert.
		FLMUINT *		puiStorageLen,	///< On input, *puiStorageLen is the size of pucStorageBuf.\  It must be atleast F_MAX_NUM64_BUF
												///< bytes.\  On output *puiStorageLen is set to the number of bytes used in pucStorageBuf.
		FLMBYTE *		pucStorageBuf	///< Number converted to FLAIM's internal storage format is returned here.
		);

	/// Convert a FLMINT value to FLAIM's internal storage format for numbers.
	/// \ingroup storageconversion
	FLMXPC RCODE FLMAPI FlmINT2Storage(
		FLMINT			iNum,				///< Number to convert.
		FLMUINT *		puiStorageLen,	///< On input, *puiStorageLen is the size of pucStorageBuf.\  It must be atleast F_MAX_NUM_BUF
												///< bytes.\  On output *puiStorageLen is set to the number of bytes used in pucStorageBuf.
		FLMBYTE *		pucStorageBuf	///< Number converted to FLAIM's internal storage format is returned here.
		);

	/// Convert a FLMINT64 value to FLAIM's internal storage format for numbers.
	/// \ingroup storageconversion
	FLMXPC RCODE FLMAPI FlmINT64ToStorage(
		FLMINT64			i64Num,			///< Number to convert.
		FLMUINT *		puiStorageLen,	///< On input, *puiStorageLen is the size of pucStorageBuf.\  It must be atleast F_MAX_NUM64_BUF
												///< bytes.\  On output *puiStorageLen is set to the number of bytes used in pucStorageBuf.
		FLMBYTE *		pucStorageBuf	///< Number converted to FLAIM's internal storage format is returned here.
		);

	/// Convert a value from FLAIM's internal format to a FLMUINT.  Note that the value may be a FLM_NUMBER_TYPE,
	/// FLM_TEXT_TYPE, or FLM_CONTEXT_TYPE.
	/// \ingroup storageconversion
	FLMXPC RCODE FLMAPI FlmStorage2UINT(
		FLMUINT				uiValueType,	///< Data type of value being converted.\  May be FLM_NUMBER_TYPE, FLM_TEXT_TYPE, or
													///< FLM_CONTEXT_TYPE.
		FLMUINT 				uiValueLength,	///< Length of value to be converted (in bytes).
		const FLMBYTE *	pucValue,		///< Value to be converted.\  Data is expected to be in FLAIM's internal format.
		FLMUINT *			puiNum			///< Converted number is returned here.
		);

	/// Convert a value from FLAIM's internal format to a FLMUINT32.  Note that the value may be a FLM_NUMBER_TYPE,
	/// FLM_TEXT_TYPE, or FLM_CONTEXT_TYPE.
	/// \ingroup storageconversion
	FLMXPC RCODE FLMAPI FlmStorage2UINT32(
		FLMUINT				uiValueType,	///< Data type of value being converted.\  May be FLM_NUMBER_TYPE, FLM_TEXT_TYPE, or
													///< FLM_CONTEXT_TYPE.
		FLMUINT 				uiValueLength,	///< Length of value to be converted (in bytes).
		const FLMBYTE *	pucValue,		///< Value to be converted.\  Data is expected to be in FLAIM's internal format.
		FLMUINT32 *			pui32Num			///< Converted number is returned here.
		);

	/// Convert a value from FLAIM's internal format to a FLMUINT64.  Note that the value may be a FLM_NUMBER_TYPE,
	/// FLM_TEXT_TYPE, or FLM_CONTEXT_TYPE.
	/// \ingroup storageconversion
	FLMXPC RCODE FLMAPI FlmStorage2UINT64(
		FLMUINT				uiValueType,	///< Data type of value being converted.\  May be FLM_NUMBER_TYPE, FLM_TEXT_TYPE, or
													///< FLM_CONTEXT_TYPE.
		FLMUINT 				uiValueLength,	///< Length of value to be converted (in bytes).
		const FLMBYTE *	pucValue,		///< Value to be converted.\  Data is expected to be in FLAIM's internal format.
		FLMUINT64 *			pui64Num			///< Converted number is returned here.
		);

	/// Convert a value from FLAIM's internal format to a FLMINT.  Note that the value may be a FLM_NUMBER_TYPE,
	/// FLM_TEXT_TYPE, or FLM_CONTEXT_TYPE.
	/// \ingroup storageconversion
	FLMXPC RCODE FLMAPI FlmStorage2INT(
		FLMUINT				uiValueType,	///< Data type of value being converted.\  May be FLM_NUMBER_TYPE, FLM_TEXT_TYPE, or
													///< FLM_CONTEXT_TYPE.
		FLMUINT 				uiValueLength,	///< Length of value to be converted (in bytes).
		const FLMBYTE *	pucValue,		///< Value to be converted.\  Data is expected to be in FLAIM's internal format.
		FLMINT *				puiNum			///< Converted number is returned here.
		);

	/// Convert a value from FLAIM's internal format to a FLMINT32.  Note that the value may be a FLM_NUMBER_TYPE,
	/// FLM_TEXT_TYPE, or FLM_CONTEXT_TYPE.
	/// \ingroup storageconversion
	FLMXPC RCODE FLMAPI FlmStorage2INT32(
		FLMUINT				uiValueType,	///< Data type of value being converted.\  May be FLM_NUMBER_TYPE, FLM_TEXT_TYPE, or
													///< FLM_CONTEXT_TYPE.
		FLMUINT 				uiValueLength,	///< Length of value to be converted (in bytes).
		const FLMBYTE *	pucValue,		///< Value to be converted.\  Data is expected to be in FLAIM's internal format.
		FLMINT32 *			pui32Num			///< Converted number is returned here.
		);

	/// Convert a value from FLAIM's internal format to a FLMINT64.  Note that the value may be a FLM_NUMBER_TYPE,
	/// FLM_TEXT_TYPE, or FLM_CONTEXT_TYPE.
	/// \ingroup storageconversion
	FLMXPC RCODE FLMAPI FlmStorage2INT64(
		FLMUINT				uiValueType,	///< Data type of value being converted.\  May be FLM_NUMBER_TYPE, FLM_TEXT_TYPE, or
													///< FLM_CONTEXT_TYPE.
		FLMUINT 				uiValueLength,	///< Length of value to be converted (in bytes).
		const FLMBYTE *	pucValue,		///< Value to be converted.\  Data is expected to be in FLAIM's internal format.
		FLMINT64 *			pui64Num			///< Converted number is returned here.
		);

	/// Convert a unicode string to FLAIM's internal storage format.
	/// \ingroup storageconversion
	FLMXPC RCODE FLMAPI FlmUnicode2Storage(
		const FLMUNICODE *	puzStr,			///< Unicode string that is to be converted.\  FLAIM expects the string
														///< to be null-terminated.
		FLMUINT *				puiStorageLen,	///< On input, *puiStorageLen is length (in bytes) of pucStorageBuf.\  On output, *puiStorageLen
														///< contains number of bytes returned.
		FLMBYTE *				pucStorageBuf	///< Converted string, in FLAIM's internal storage format, is returned here.
		);

	/// Determine the number of bytes needed to store a unicode string in FLAIM's internal storage format.
	/// \ingroup storageconversion
	FLMXPC FLMUINT FLMAPI FlmGetUnicodeStorageLength(
		const FLMUNICODE *	puzStr		///< Unicode string whose internal storage length is to be determined.\  It is
													///< expected that the string will be null-terminated.
		);

	/// Convert a value from FLAIM's internal format to a unicode string.  Note that the value may be a FLM_NUMBER_TYPE,
	/// or FLM_TEXT_TYPE.
	/// \ingroup storageconversion
	FLMXPC RCODE FLMAPI FlmStorage2Unicode(
		FLMUINT				uiValueType,	///< Data type of data being converted.\  May be FLM_NUMBER_TYPE or FLM_TEXT_TYPE.
		FLMUINT 				uiValueLength,	///< Length of value to be converted (in bytes).
		const FLMBYTE *	pucValue,		///< Value to be converted.\  Data is expected to be in FLAIM's internal format.
		FLMUINT *			puiStrBufLen,	///< On input *puiStrBufLen should be the number of bytes available in puzStrBuf.\  puzStrBuf
													///< should be large enough to hold a unicode null terminating character (2 bytes).\  On output
													///< *puiStrBufLen returns the number of bytes needed to hold the converted Unicode
													///< string.\  NOTE: The two null termination bytes are NOT included in this value.
		FLMUNICODE *		puzStrBuf		///< Buffer to hold the Unicode string.\  NOTE: If this parameter is NULL then
													///< *puiStrBufLen will return the number of bytes needed to hold the string.\  However,
													///< that does NOT count the two bytes needed to null-terminate the string.\  Thus, if
													///< the application is calling this routine to find out how big of a buffer to allocate
													///< to hold the string, it should add 2 to the value returned in *puiStrBufLen.
		);

	/// Get the number of bytes needed to convert a value from FLAIM's internal format to a unicode string.  The value may be either
	/// a FLM_NUMBER_TYPE or a FLM_TEXT_TYPE.  The length returned does NOT account for null-termination, so if the application is
	/// calling this routine to determine how big of a buffer to allocate, it should add 2 to the size returned from this routine.
	/// \ingroup storageconversion
	FINLINE RCODE FlmGetUnicodeLength(
		FLMUINT				uiValueType,	///< Data type of value to be converted.\  May be either FLM_NUMBER_TYPE or FLM_TEXT_TYPE.
		FLMUINT				uiValueLength,	///< Length of value in bytes.
		const FLMBYTE *	pucValue,		///< Data value.
		FLMUINT *			puiUniLength	///< Unicode length is returned (in bytes).\  The length does not include what it would take
													///< for null-termination.
		)
	{
		return( FlmStorage2Unicode( uiValueType, uiValueLength, pucValue, puiUniLength, NULL));
	}

	/// Convert a native string to FLAIM's internal storage format.
	/// \ingroup storageconversion
	FLMXPC RCODE FLMAPI FlmNative2Storage(
		const char *		pszStr,			///< Native string that is to be converted.
		FLMUINT 				uiStrLen,		///< Length (in bytes) of the native string.\  If zero, the string is
													///< expected to be NULL-terminated.
													///< *puiStorageLen contains number of bytes returned.
		FLMUINT *			puiStorageLen,	///< On input, *puiStorageLen is length (in bytes) of pucStorageBuf.\  On output,
													///< *puiStorageLen contains number of bytes returned.
		FLMBYTE *			pucStorageBuf	///< Converted string, in FLAIM's internal storage format, is returned here.
		);

	/// Convert a value from FLAIM's internal format to a native string.  Note that the value may be a FLM_NUMBER_TYPE,
	/// or FLM_TEXT_TYPE.
	/// \ingroup storageconversion
	FLMXPC RCODE FLMAPI FlmStorage2Native(
		FLMUINT				uiValueType,	///< Data type of value being converted.\  May be FLM_NUMBER_TYPE or FLM_TEXT_TYPE.
		FLMUINT 				uiValueLength,	///< Length of value to be converted (in bytes).
		const FLMBYTE *	pucValue,		///< Value to be converted.\  Data is expected to be in FLAIM's internal format.
		FLMUINT *			puiStrBufLen,	///< On input *puiStrBufLen should be the number of bytes available in buffer.\  The
													///< buffer should be large enough to hold a terminating null byte.\  On output
													///< *puiStrBufLen returns the number of bytes needed to hold the converted native
													///< string.\  The null termination byte is NOT included in this value.
		char *				pszStrBuf		///< Buffer to hold the native string.\  NOTE: If this parameter is NULL then
													///< *puiStrBufLen will return the number of bytes needed to hold the string.\  However,
													///< that does NOT count the byte needed to null-terminate the string.\  Thus, if
													///< the application is calling this routine to find out how big of a buffer to allocate
													///< to hold the string, it should add 1 to the value returned in *puiStrBufLen.
		);

	/// Determine the number of bytes needed to store a native string in FLAIM's internal storage format.
	/// \ingroup storageconversion
	FLMXPC FLMUINT FLMAPI FlmGetNativeStorageLength(
		const char *		pszStr		///< Native string whose internal storage length is to be determined.\  It is
												///< expected that the string will be null-terminated.
		);

	/// Get the number of bytes needed to convert a value from FLAIM's internal format to a native string.  The value may be either
	/// a FLM_NUMBER_TYPE or a FLM_TEXT_TYPE.  The length returned does NOT account for null-termination, so if the application is
	/// calling this routine to determine how big of a buffer to allocate, it should add 1 to the size returned from this routine.
	/// \ingroup storageconversion
	FINLINE RCODE FlmGetNativeLength(
		FLMUINT				uiValueType,	///< Data type of value to be converted.\  May be either FLM_NUMBER_TYPE or FLM_TEXT_TYPE.
		FLMUINT				uiValueLength,	///< Length of value in bytes.
		const FLMBYTE *	pucValue,		///< Data value.
		FLMUINT *			puiStrLength	///< String length is returned (in bytes).\  The length does not include what it would take
													///< for null-termination.
		)
	{
		return( FlmStorage2Native( uiValueType, uiValueLength, pucValue, puiStrLength, NULL));
	}

	/**************************************************************************
	*                  FLAIM Dictionary Tag Numbers 
	*
	* FLAIM Database Dictionary and Internal Tags
	*
	* FLAIM TEAM NOTES:
	*  1) These numbers cannot be changed for backward compatibility reasons.                     *
	*  2) IF ANY NEW TAGS ARE INSERTED - Then you MUST change the database
	*   version number, because old databases will become invalid.....
	*
	***************************************************************************/

	#define FLM_RESERVED_TAG_NUMS							32000

	// Special purpose container and index numbers

	#define FLM_DICT_CONTAINER							32000
	#define FLM_LOCAL_DICT_CONTAINER					FLM_DICT_CONTAINER
	#define FLM_DATA_CONTAINER							32001
	#define FLM_TRACKER_CONTAINER						32002
	#define FLM_DICT_INDEX								32003

	#define FLM_MISSING_FIELD_TAG						32049		// Used on CursorAddField call
	#define FLM_WILD_TAG									32050		// Wild card - matches everything

	// Range where unregistered fields begin

	#define FLM_FREE_TAG_NUMS							32769
	#define FLM_UNREGISTERED_TAGS						32769

	/****************************************************************************
								Dictionary Record Field Numbers

	WARNINGS:
		1) These numbers cannot be changed for backward compatibility reasons.
		2) Any Changes Made to any '_TAG' defines must be reflected in
			FlmDictTags table found in fntable.cpp
	****************************************************************************/

	#define FLM_TAGS_START								32100
	#define FLM_DICT_FIELD_NUMS						FLM_TAGS_START
	#define TS												FLM_TAGS_START

	#define FLM_FIELD_TAG								(TS + 0)
	#define FLM_FIELD_TAG_NAME							"Field"
	#define FLM_INDEX_TAG								(TS + 1)
	#define FLM_INDEX_TAG_NAME							"Index"
	#define FLM_TYPE_TAG									(TS + 2)
	#define FLM_TYPE_TAG_NAME							"Type"
	#define FLM_COMMENT_TAG								(TS + 3)
	#define FLM_COMMENT_TAG_NAME						"Comment"
	#define FLM_CONTAINER_TAG							(TS + 4)
	#define FLM_CONTAINER_TAG_NAME					"Container"
	#define FLM_LANGUAGE_TAG							(TS + 5)
	#define FLM_LANGUAGE_TAG_NAME						"Language"
	#define FLM_OPTIONAL_TAG							(TS + 6)
	#define FLM_OPTIONAL_TAG_NAME						"Optional"
	#define FLM_UNIQUE_TAG								(TS + 7)
	#define FLM_UNIQUE_TAG_NAME						"Unique"
	#define FLM_KEY_TAG									(TS + 8)
	#define FLM_KEY_TAG_NAME							"Key"
	#define FLM_REFS_TAG									(TS + 9)
	#define FLM_REFS_TAG_NAME							"Refs"
	#define FLM_ENCDEF_TAG								(TS + 10)
	#define FLM_ENCDEF_TAG_NAME						"EncDef"
	#define FLM_DELETE_TAG								(TS + 11)
	#define FLM_DELETE_TAG_NAME						"Delete"
	#define FLM_BLOCK_CHAIN_TAG						(TS + 12)
	#define FLM_BLOCK_CHAIN_TAG_NAME					"BlockChain"
	//#define FLM_NU_13_TAG								(TS + 13)
	//#define FLM_NU_14_TAG								(TS + 14)
	//#define FLM_NU_15_TAG								(TS + 15)
	//#define FLM_NU_16_TAG								(TS + 16)
	#define FLM_AREA_TAG									(TS + 17)
	#define FLM_AREA_TAG_NAME							"Area"
	//#define FLM_NU_18_TAG								(TS + 18)
	//#define FLM_NU_19_TAG								(TS + 19)
	//#define FLM_NU_20_TAG								(TS + 20)
	//#define FLM_NU_21_TAG								(TS + 21)
	//#define FLM_NU_22_TAG								(TS + 22)
	//#define FLM_NU_23_TAG								(TS + 23)
	//#define FLM_NU_24_TAG								(TS + 24)
	#define FLM_STATE_TAG								(TS + 25)
	#define FLM_STATE_TAG_NAME							"State"
	#define FLM_BLOB_TAG									(TS + 26)
	#define FLM_BLOB_TAG_NAME							"Blob"
	#define FLM_THRESHOLD_TAG							(TS + 27)
	#define FLM_THRESHOLD_TAG_NAME					"Threshold"
	//#define FLM_NU_28_TAG								(TS + 28)
	#define FLM_SUFFIX_TAG								(TS + 29)
	#define FLM_SUFFIX_TAG_NAME						"Suffix"
	#define FLM_SUBDIRECTORY_TAG						(TS + 30)
	#define FLM_SUBDIRECTORY_TAG_NAME				"Subdirectory"
	#define FLM_RESERVED_TAG							(TS + 31)
	#define FLM_RESERVED_TAG_NAME						"Reserved"
	#define FLM_SUBNAME_TAG								(TS + 32)
	#define FLM_SUBNAME_TAG_NAME						"Subname"
	#define FLM_NAME_TAG									(TS + 33)
	#define FLM_NAME_TAG_NAME							"Name"
	//#define FLM_NU_34_TAG								(TS + 34)
	//#define FLM_NU_35_TAG								(TS + 35) 
	#define FLM_BASE_TAG									(TS + 36)
	#define FLM_BASE_TAG_NAME							"Base"
	//#define FLM_NU_37_TAG								(TS + 37)
	#define FLM_CASE_TAG									(TS + 38)
	#define FLM_CASE_TAG_NAME							"Case"
	//#define FLM_NU_39_TAG								(TS + 39)
	#define FLM_COMBINATIONS_TAG						(TS + 40)
	#define FLM_COMBINATIONS_TAG_NAME				"Combinations"
	#define FLM_COUNT_TAG								(TS + 41)
	#define FLM_COUNT_TAG_NAME							"Count"
	#define FLM_POSITIONING_TAG						(TS + 42)
	#define FLM_POSITIONING_TAG_NAME					"Positioning"
	//#define FLM_NU_43_TAG								(TS + 43)
	#define FLM_PAIRED_TAG								(TS + 44)
	#define FLM_PAIRED_TAG_NAME						"Paired"
	#define FLM_PARENT_TAG								(TS + 45)
	#define FLM_PARENT_TAG_NAME						"Parent"
	#define FLM_POST_TAG									(TS + 46)
	#define FLM_POST_TAG_NAME							"Post"
	#define FLM_REQUIRED_TAG							(TS + 47)
	#define FLM_REQUIRED_TAG_NAME						"Required"
	#define FLM_USE_TAG									(TS + 48)
	#define FLM_USE_TAG_NAME							"Use"
	#define FLM_FILTER_TAG								(TS + 49)
	#define FLM_FILTER_TAG_NAME						"Filter"
	#define FLM_LIMIT_TAG								(TS + 50)
	#define FLM_LIMIT_TAG_NAME							"Limit"
	//#define FLM_NU_51_TAG								(TS + 51)
	//#define FLM_NU_52_TAG								(TS + 52)
	//#define FLM_NU_53_TAG								(TS + 53)
	#define FLM_DICT_TAG									(TS + 54)
	#define FLM_DICT_TAG_NAME							"Dict"
	//#define FLM_NU_55_TAG								(TS + 55)
	//#define FLM_NU_56_TAG								(TS + 56)
	//#define FLM_NU_57_TAG								(TS + 57)
	//#define FLM_NU_58_TAG								(TS + 58)
	//#define FLM_NU_59_TAG								(TS + 59)
	//#define FLM_NU_60_TAG								(TS + 60) 
	//#define FLM_NU_61_TAG								(TS + 61)
	//#define FLM_NU_62_TAG								(TS + 62)
	//#define FLM_NU_63_TAG								(TS + 63)
	//#define FLM_NU_64_TAG								(TS + 64)
	//#define FLM_NU_65_TAG								(TS + 65)
	//#define FLM_NU_66_TAG								(TS + 66)
	//#define FLM_NU_67_TAG								(TS + 67)
	//#define FLM_NU_68_TAG								(TS + 68)
	//#define FLM_NU_69_TAG								(TS + 69)
	#define FLM_RECINFO_TAG								(TS + 70)
	#define FLM_RECINFO_TAG_NAME						"RecInfo"
	#define FLM_DRN_TAG									(TS + 71)
	#define FLM_DRN_TAG_NAME							"Drn"
	#define FLM_DICT_SEQ_TAG							(TS + 72)
	#define FLM_DICT_SEQ_TAG_NAME						"DictSeq"
	#define FLM_LAST_CONTAINER_INDEXED_TAG			(TS + 73)
	#define FLM_LAST_CONTAINER_INDEXED_TAG_NAME	"LastContainerIndexed"
	#define FLM_LAST_DRN_INDEXED_TAG					(TS + 74)
	#define FLM_LAST_DRN_INDEXED_TAG_NAME			"LastDrnIndexed"
	#define FLM_ONLINE_TRANS_ID_TAG					(TS + 75)
	#define FLM_ONLINE_TRANS_ID_TAG_NAME			"OnlineTransId"
	#define FLM_LAST_DICT_FIELD_NUM					(TS + 75)

	/****************************************************************************
	Dictionary Record Definitions - below are comments that document valid 
	dictionary objects and their structure.
	****************************************************************************/

	/*
	Field Definition
	Desc: The below syntax is used to define a field within a database dictionary
			container.

	0 [@<ID>@] field <name>          # FLM_FIELD_TAG
	| 1 type <below>                 # FLM_TYPE_TAG
			{context|number|text|binary|blob}]
	[ 1 state <below>                # FLM_STATE_TAG - what is the state of the field
			{ *active                  #  The field is active (being used).
			| checking                 #  User request to determine if field is used.
												#  This is done by calling FlmDbSweep.
			| unused                   #  Result of 'checking'. Field is not used and
												#  maybe deleted. Note: a field in this state
												#  may still have other dictionary item that
												#  are referencing it.
			| purge}]                  #  Remove all fld occurances, and delete def.
	*/
	
	/*
	 Encryption Definition
	 Desc: The below syntax is used to define an encryption definition record.
	 0 [@<ID>@] EncDef <name>		# FLM_ENCDEF_TAG
	 	1 type <below>						# FLM_TYPE_TAG
	 		{ des3 | aes }
	 */

	/*
	Container Definition
	Desc: The below syntax is used to define a container within a database dictionary
			container.

	0 [@<ID>@] container <name>      # FLM_CONTAINER_TAG
	*/

	/*
	Index Definition
	Desc: Below is the syntax that is used to define a FLAIM index.

	0 [@<ID>@] index <psName>        # FLM_INDEX_TAG
	[ 1 container* DEFAULT* | <ID> ] # FLM_CONTAINER_TAG - indexes span only one container
	[ 1 count KEYS &| REFS* ]        # FLM_COUNT_TAG - key count of keys and/or refs
	[ 1 language* US* | <language> ] # FLM_LANGUAGE_TAG - for full-text parsing and/or sorting

	  1 key [EACHWORD]					# FLM_KEY_TAG - 'use' defaults based on type
	  [ 2 post]                      # FLM_POST_TAG - case-flags post-pended to key
	  [ 2 required*]                 # FLM_REQUIRED_TAG - key value is required
	  [ 2 unique]                    # FLM_UNIQUE_TAG - key has only 1 reference
	  { 2 <field> }...               # FLM_FIELD_TAG - compound key if 2 or more
		 [ 3 case* mixed* | upper]    # FLM_CASE_TAG - text-only, define chars case
		 [ 3 <field>]...              # FLM_FIELD_TAG - alternate field(s)
		 [ 3 optional*                # FLM_OPTIONAL_TAG - component's value is optional
		  |3 required ]               # FLM_REQUIRED_TAG - component's value is required
		{[ 3 use* eachword**|value*|field|substring]} # FLM_USE_TAG
		{[ 3 filter minspace|nodash|nounderscore|nospace]} # FLM_FILTER_TAG
		 [ 3 limit {256 | limit}]

	<field> ==
	  n field <field path>           #  path identifies field -- maybe "based"
	  [ m type <data type>]          # FLM_TYPE_TAG - only for ixing unregistered fields
	*/


	/****************************************************************************
			
	NON-Dictionary Record Definitions - the following record definitions are for
	internal FLAIM usage.

	****************************************************************************/

	/*
	Deleted BLOB Tracker Record
	Desc: BLOB tracker records are found in the FLM_TRACKER_CONTAINER.
			It records blob files that are on ready to be deleted. 
			BDELETE records are single-field records.  The fields are BLOB fields.

	0 bdelete <BLOB>                 # FLM_BLOB_DELETE_TAG - single-field record;
												#  DRN > 65,536
	*/

	/*
	Record Info Record - for EXPORT/IMPORT
	Desc: The Record Info Record is currently only used in export/import files.
			It contains the record information for each exported record.

	0 recinfo                        # FLM_RECINFO_TAG
	  1 drn <#>                      # FLM_DRN_TAG - DRN for the record
	[ 1 dseq <#>]                    # FLM_DICT_SEQ_TAG - dictionary sequence ID for the record
	*/

	FLMXPC RCODE FLMAPI FlmKeyBuild(
		HFDB			hDb,
		FLMUINT		uiIxNum,
		FLMUINT		uiContainer,
		FlmRecord *	pKeyTree,
		FLMUINT		uiFlags,
		FLMBYTE *	pucKeyBuf,
		FLMUINT *	puiKeyLenRV);

	typedef FLMUINT32		FIELDLINK;

	/****************************************************************************
	Struct: 	FlmField
	****************************************************************************/
	typedef struct
	{
		FLMUINT32	ui32DataOffset;
		FLMUINT16	ui16FieldID;
		FLMUINT8		ui8DataLen;
		FLMUINT8		ui8TypeAndLevel;

			// Bits 0 - 2 used for type

			#define FLD_TEXT_TYPE				0x00
			#define FLD_NUMBER_TYPE				0x01
			#define FLD_BINARY_TYPE				0x02
			#define FLD_CONTEXT_TYPE			0x03
			#define FLD_BLOB_TYPE				0x04

			// Bits 3 - 4 used for flags

			#define FLD_DATA_LEFT_TRUNCATED	0x08
			#define FLD_DATA_RIGHT_TRUNCATED	0x10

			// Bits 5 - 7 for level (max level is 7)

		FIELDLINK	uiPrev;
		FIELDLINK	uiNext;

	} FlmField;
	
	/****************************************************************************
	Struct: 	FIELD_ID
	****************************************************************************/
	typedef struct FIELD_ID
	{
		FIELDLINK	ui32FieldOffset;
		FLMUINT16	ui16FieldId;
	} FIELD_ID;
	
	FINLINE FLMUINT calcFieldIdTableByteSize(
		FLMUINT	uiTableItems)
	{
		return( FLM_ALIGN_SIZE + FLM_ALIGN_SIZE + FLM_ALIGN_SIZE +
					sizeof( FIELD_ID) * uiTableItems);
	}
	
	FINLINE void setFieldIdTableItemCount(
		FLMBYTE *	pucFieldIdTable,
		FLMUINT		uiItemCount)
	{
		*((FLMUINT *)(pucFieldIdTable + FLM_ALIGN_SIZE)) = uiItemCount;
	}

	FINLINE void setFieldIdTableArraySize(
		FLMBYTE *	pucFieldIdTable,
		FLMUINT		uiTableArraySize)
	{
		*((FLMUINT *)(pucFieldIdTable + FLM_ALIGN_SIZE + FLM_ALIGN_SIZE)) = uiTableArraySize;
	}

	FINLINE FLMUINT getFieldIdTableItemCount(
		FLMBYTE *	pucFieldIdTable)
	{
		if (!pucFieldIdTable)
		{
			return( 0);
		}
		else
		{
			return( *((FLMUINT *)(pucFieldIdTable + FLM_ALIGN_SIZE)));
		}
	}

	FINLINE FLMUINT getFieldIdTableArraySize(
		FLMBYTE *	pucFieldIdTable)
	{
		if (!pucFieldIdTable)
		{
			return( 0);
		}
		else
		{
			return( *((FLMUINT *)(pucFieldIdTable + FLM_ALIGN_SIZE + FLM_ALIGN_SIZE)));
		}
	}

	FINLINE FIELD_ID * getFieldIdTable(
		FLMBYTE *	pucFieldIdTable)
	{
		return( (FIELD_ID *)(pucFieldIdTable +
					FLM_ALIGN_SIZE + FLM_ALIGN_SIZE + FLM_ALIGN_SIZE));
	}
		
	/****************************************************************************
	Desc: 	Class which provides the record interface that FLAIM uses to
				access and manipulate all records.
	****************************************************************************/
	/// Class for creating and modifying database records.
	class FLMEXP FlmRecord : public F_Object
	{
	public:

		#define RCA_READ_ONLY_FLAG						0x00000001
		#define RCA_CACHED								0x00000002
		#define RCA_OK_TO_DELETE						0x00000004
		#define RCA_OLD_VERSION							0x00000008
		#define RCA_HEAP_BUFFER							0x00000010
		#define RCA_ID_TABLE_HEAP_BUFFER				0x00000020
		#define RCA_FIELD_ID_TABLE_ENABLED			0x00000040
		#define RCA_NEED_TO_SORT_FIELD_ID_TABLE	0x00000080

		FlmRecord();

		virtual ~FlmRecord();

		/// Overloaded new operator for ::FlmRecord objects.
		void * operator new(
			FLMSIZET			uiSize	///< Number of bytes to allocate - should be sizeof( ::FlmRecord).
			)
	#ifndef FLM_WATCOM_NLM
			throw()
	#endif
			;

		/// Overloaded new operator (array) for ::FlmRecord objects.
		/// This method is called when an array of ::FlmRecord objects is allocated.
		void * operator new[](
			FLMSIZET			uiSize	///< Number of bytes to allocate - should be a multiple of sizeof( ::FlmRecord).
			)
	#ifndef FLM_WATCOM_NLM
			throw()
	#endif
			;

		/// Overloaded delete operator for ::FlmRecord objects.
		void operator delete(
			void *			ptr);		///< Pointer to ::FlmRecord object being freed.

		/// Overloaded delete operator (array) for ::FlmRecord objects.
		/// This method is called when an array of ::FlmRecord objects is freed.
		void operator delete[](
			void *			ptr);		///< Pointer to array of ::FlmRecord objects being freed.

		/// Overloaded new operator for ::FlmRecord objects (with source file and line number).
		/// This new operator passes in the current file and line number.  This information is
		/// useful in tracking memory allocations to determine where memory leaks are coming from.
		void * operator new(
			FLMSIZET			uiSize,	///< Number of bytes to allocate - should be sizeof( ::FlmRecord).
			const char *	pszFile,	///< Name of source file where this allocation is made.
			int				iLine		///< Line number in source file where this allocation request is made.
			)
	#ifndef FLM_WATCOM_NLM
			throw()
	#endif
			;

		/// Overloaded new operator (array) for ::FlmRecord objects (with source file and line number).
		/// This new operator is called when an array of ::FlmRecord objects is allocated.
		/// This new operator passes in the current file and line number.  This information is
		/// useful in tracking memory allocations to determine where memory leaks are coming from.
		void * operator new[](
			FLMSIZET			uiSize,	///< Number of bytes to allocate - should be a multiple of sizeof( ::FlmRecord).
			const char *	pszFile,	///< Name of source file where this allocation is made.
			int				iLine		///< Line number in source file where this allocation request is made.
			)
	#ifndef FLM_WATCOM_NLM
			throw()
	#endif
			;

		/// Overloaded delete operator for ::FlmRecord objects (with source file and line number).
		/// This delete operator passes in the current file and line number.  This information is
		/// useful in tracking memory allocations to determine where memory leaks are coming from.
		void operator delete(
			void *			ptr,		///< Pointer to ::FlmRecord object being freed.
			const char *	pszFile,	///< Name of source file where this delete occurs.
			int				iLine);	///< Line number in source file where this delete occurs.

		/// Overloaded delete operator (array) for ::FlmRecord objects (with source file and line number).
		/// This delete operator is called when an array of ::FlmRecord objects is freed.
		/// This delete operator passes in the current file and line number.  This information is
		/// useful in tracking memory allocations to determine where memory leaks are coming from.
		void operator delete[](
			void *			ptr,		///< Pointer to array of ::FlmRecord objects being freed.
			const char *	pszFile,	///< Name of source file where this delete occurs.
			int				iLine);	///< Line number in source file where this delete occurs.

		/// Increment the reference count for this ::FlmRecord object.
		/// The reference count is the number of pointers that are referencing this ::FlmRecord object.
		/// Return value is the incremented reference count.
		FLMINT FLMAPI AddRef( void);

		/// Decrement the reference count for this ::FlmRecord object.
		/// The reference count is the number of pointers that are referencing this ::FlmRecord object.
		/// Return value is the decremented reference count.  If the reference count goes to
		/// zero, the ::FlmRecord object will be deleted.
		FINLINE FLMINT FLMAPI Release( void)
		{
			return( Release( FALSE));
		}

		/// Make a writeable copy of this ::FlmRecord object.
		FlmRecord * copy( void);

		/// Clear all fields in this ::FlmRecord object.
		RCODE clear(
			FLMBOOL			bReleaseMemory = FALSE	///< If TRUE, free the memory buffer used to hold field data.\  If FALSE,
																///< the memory buffer will be cleared without freeing it.
			);

		/// Get a field's value as a FLMINT.  Data conversions will be done if the field is a FLM_CONTEXT_TYPE or FLM_TEXT_TYPE.
		RCODE getINT( 
			void *			pvField,		///< Field whose value is to be retrieved (and converted if necessary).
			FLMINT *			piNumber		///< Number is returned here.
			);

		/// Get a field's value as a FLMINT32.  Data conversions will be done if the field is a FLM_CONTEXT_TYPE or FLM_TEXT_TYPE.
		RCODE getINT32( 
			void *			pvField,		///< Field whose value is to be retrieved (and converted if necessary).
			FLMINT32 *		pi32Number	///< Number is returned here.
			);

		/// Get a field's value as a FLMINT64.  Data conversions will be done if the field is a FLM_CONTEXT_TYPE or FLM_TEXT_TYPE.
		RCODE getINT64( 
			void *			pvField,		///< Field whose value is to be retrieved (and converted if necessary).
			FLMINT64 *		pi64Number	///< Number is returned here.
			);

		/// Get a field's value as a FLMUINT.  Data conversions will be done if the field is a FLM_CONTEXT_TYPE or FLM_TEXT_TYPE.
		RCODE getUINT( 
			void *			pvField,		///< Field whose value is to be retrieved (and converted if necessary).
			FLMUINT *		puiNumber	///< Number is returned here.
			);

		/// Get a field's value as a FLMUINT32.  Data conversions will be done if the field is a FLM_CONTEXT_TYPE or FLM_TEXT_TYPE.
		RCODE getUINT32(
			void *			pvField,			///< Field whose value is to be retrieved (and converted if necessary).
			FLMUINT32 *		pui32Number		///< Number is returned here.
			);

		/// Get a field's value as a FLMUINT64.  Data conversions will be done if the field is a FLM_CONTEXT_TYPE or FLM_TEXT_TYPE.
		RCODE getUINT64(
			void *			pvField,			///< Field whose value is to be retrieved (and converted if necessary).
			FLMUINT64 *		pui64Number		///< Number is returned here.
			);

		/// Get the number of bytes needed to retrieve a field's value as a unicode string.  The value may be either
		/// a FLM_NUMBER_TYPE or a FLM_TEXT_TYPE.  The length returned does NOT account for null-termination, so if the application is
		/// calling this routine to determine how big of a buffer to allocate, it should add 2 to the size returned from this routine.
		FINLINE RCODE getUnicodeLength( 
			void *			pvField,			///< Field containing the value whose unicode string length is to be determined.
			FLMUINT *		puiLength		///< Unicode length is returned (in bytes).\  The length does not include what it would take
													///< for null-termination.
			)
		{
			if( pvField)
			{
				return( FlmGetUnicodeLength( 
					getDataType( pvField), getDataLength( pvField), 
					(const FLMBYTE *)getDataPtr( pvField), puiLength));
			}

			return( FERR_NOT_FOUND);
		}

		/// Get a field's value as a unicode string.  Note that the field may be a FLM_NUMBER_TYPE, or FLM_TEXT_TYPE.
		RCODE getUnicode(
			void *			pvField,			///< Field whose data is to be returned as a unicode string.
			FLMUNICODE *	puzStrBuf,		///< Buffer to hold the Unicode string.\  NOTE: If this parameter is NULL then
													///< *puiStrBufLen will return the number of bytes needed to hold the string.\  However,
													///< that does NOT count the two bytes needed to null-terminate the string.\  Thus, if
													///< the application is calling this routine to find out how big of a buffer to allocate
													///< to hold the string, it should add 2 to the value returned in *puiStrBufLen.
			FLMUINT *		puiStrBufLen	///< On input *puiStrBufLen should be the number of bytes available in puzStrBuf.\  puzStrBuf
													///< should be large enough to hold a unicode null terminating character (2 bytes).\  On output
													///< *puiStrBufLen returns the number of bytes needed to hold the converted Unicode
													///< string.\  NOTE: The two null termination bytes are NOT included in this value.
			);

		/// Get the number of bytes needed to retrieve a field's value as a native string.  The value may be either
		/// a FLM_NUMBER_TYPE or a FLM_TEXT_TYPE.  The length returned does NOT account for null-termination, so if the application is
		/// calling this routine to determine how big of a buffer to allocate, it should add 1 to the size returned from this routine.
		FINLINE RCODE getNativeLength( 
			void *			pvField,			///< Field containing the value whose native string length is to be determined.
			FLMUINT *		puiLength		///< Native length is returned (in bytes).\  The length does not include what it would take
													///< for null-termination.
			)
		{
			if( pvField)
			{
				return( FlmGetNativeLength( 
									getDataType( pvField), getDataLength( pvField),
									(const FLMBYTE *)getDataPtr( pvField), puiLength));
			}

			return( FERR_NOT_FOUND);
		}

		/// Get a field's value as a native string.  Note that the field may be a FLM_NUMBER_TYPE, or FLM_TEXT_TYPE.
		RCODE getNative( 
			void *			pvField,			///< Field whose data is to be returned as a native string.
			char *			pszStrBuf,		///< Buffer to hold the native string.\  NOTE: If this parameter is NULL then
													///< *puiStrBufLen will return the number of bytes needed to hold the string.\  However,
													///< that does NOT count the byte needed to null-terminate the string.\  Thus, if
													///< the application is calling this routine to find out how big of a buffer to allocate
													///< to hold the string, it should add 1 to the value returned in *puiStrBufLen.
			FLMUINT *		puiStrBufLen	///< On input *puiStrBufLen should be the number of bytes available in pszStrBuf.\  pszStrBuf
													///< should be large enough to hold a null terminating character.\  On output
													///< *puiStrBufLen returns the number of bytes needed to hold the converted native
													///< string.\  NOTE: The null termination byte is NOT included in this value.
			);

		/// Get the number of bytes needed to retrieve a field's value as binary data.  This method may be called for any
		/// type of field, but it really only makes sense for fields whose data type is FLM_BINARY_TYPE.
		FINLINE RCODE getBinaryLength( 
			void *			pvField,			///< Field whose length is to be returned.
			FLMUINT *		puiLength		///< Field's length is returned here.\  NOTE: For fields whose data type is not FLM_BINARY_TYPE, this
													///< returns the internal storage length.\  It really doesn't make much sense to call this
													///< method for fields whose data type is not FLM_BINARY_TYPE.
			)
		{
			if( pvField)
			{
				*puiLength = getDataLength( pvField);
				return FERR_OK;
			}

			return( FERR_NOT_FOUND);
		}

		/// Get the value for a FLM_CONTEXT_TYPE field as a FLMUINT.
		RCODE getRecPointer(
			void *			pvField,			///< Field whose value is to be returned.
			FLMUINT *		puiRecPointer	///< Value is returned here.
			);

		/// Get the value for a FLM_CONTEXT_TYPE field as a FLMUINT32.
		FINLINE RCODE getRecPointer32( 
			void *			pvField,				///< Field whose value is to be returned.
			FLMUINT32 *		pui32RecPointer	///< Value is returned here.
			)
		{
			FLMUINT	uiRecPointer;
			RCODE		rc;
			
			rc = getRecPointer( pvField, &uiRecPointer);
			*pui32RecPointer = (FLMUINT32)uiRecPointer;

			return( rc);
		}

		/// Get a field's value as binary data.  This method may be used for any type of field, but it really
		/// only makes sense for fields whose data type is FLM_BINARY_TYPE.  If it is used on a field whose
		/// data type is not FLM_BINARY_TYPE, the data returned is in FLAIM's internal storage format.
		RCODE getBinary(
			void *			pvField,		///< Field whose value is to be returned as binary data.
			void *			pvBuf,		///< Buffer to hold the binary data.
			FLMUINT *		puiBufLen	///< On input *puiBufLen is the size of pvBuf (in bytes).\  On output, it returns
												///< the number of bytes in the value.\  NOTE: If the value length is greater
												///< than the *puiBufLen that was passed in, the returned value will be truncated
												///< to fit in that buffer.\  However, no error will be returned.
			);

		/// Get a field's value as a ::FlmBlob object.  This method may only be used on field's whose data type is FLM_BLOB_TYPE.
		RCODE getBlob(
			void *			pvField,		///< Field whose value is to be returned as a ::FlmBlob object.
			FlmBlob **		ppBlob		///< A pointer to the ::FlmBlob object is returned here.
			);

		/// Set a field's value to a FLMINT value.  The resulting data type for the field will be FLM_NUMBER_TYPE.
		RCODE setINT(
			void *			pvField,		///< Field whose value is to be set.
			FLMINT			iNumber,		///< Value to set.
			FLMUINT			uiEncId = 0	///< Encryption ID.\  If zero, the value will not be encrypted.\  If non-zero, the number
												///< should be the ID of an encryption definition record in the data dictionary.\  The
												///< encryption key for that encryption definition will be used to encrypt this value.
			);

		/// Set a field's value to a FLMINT64 value.  The resulting data type for the field will be FLM_NUMBER_TYPE.
		RCODE setINT64(
			void *			pvField,		///< Field whose value is to be set.
			FLMINT64			i64Number,	///< Value to set.
			FLMUINT			uiEncId = 0	///< Encryption ID.\  If zero, the value will not be encrypted.\  If non-zero, the number
												///< should be the ID of an encryption definition record in the data dictionary.\  The
												///< encryption key for that encryption definition will be used to encrypt this value.
			);

		/// Set a field's value to a FLMUINT value.  The resulting data type for the field will be FLM_NUMBER_TYPE.
		RCODE setUINT( 
			void *			pvField,		///< Field whose value is to be set.
			FLMUINT			uiNumber,	///< Value to set.
			FLMUINT			uiEncId = 0	///< Encryption ID.\  If zero, the value will not be encrypted.\  If non-zero, the number
												///< should be the ID of an encryption definition record in the data dictionary.\  The
												///< encryption key for that encryption definition will be used to encrypt this value.
			);

		/// Set a field's value to a FLMUINT64 value.  The resulting data type for the field will be FLM_NUMBER_TYPE.
		RCODE setUINT64( 
			void *			pvField,		///< Field whose value is to be set.
			FLMUINT64		ui64Number,	///< Value to set.
			FLMUINT			uiEncId = 0	///< Encryption ID.\  If zero, the value will not be encrypted.\  If non-zero, the number
												///< should be the ID of an encryption definition record in the data dictionary.\  The
												///< encryption key for that encryption definition will be used to encrypt this value.
			);

		/// Set a field's value to a FLMUINT value.  The resulting data type for the field will be FLM_CONTEXT_TYPE.
		RCODE setRecPointer(
			void *			pvField,			///< Field whose value is to be set.
			FLMUINT			uiRecPointer,	///< Value to set.
			FLMUINT			uiEncId = 0		///< Encryption ID.\  If zero, the value will not be encrypted.\  If non-zero, the number
													///< should be the ID of an encryption definition record in the data dictionary.\  The
													///< encryption key for that encryption definition will be used to encrypt this value.
			);

		/// Set a field's value to a Unicode string value.  The resulting data type for the field will be FLM_TEXT_TYPE.
		RCODE setUnicode(
			void *					pvField,		///< Field whose value is to be set.
			const FLMUNICODE *	puzUnicode,	///< Value to set.\  This should be a null-terminated Unicode string.
			FLMUINT					uiEncId = 0	///< Encryption ID.\  If zero, the value will not be encrypted.\  If non-zero, the number
														///< should be the ID of an encryption definition record in the data dictionary.\  The
														///< encryption key for that encryption definition will be used to encrypt this value.
			);

		/// Set a field's value to a native string value.  The resulting data type for the field will be FLM_TEXT_TYPE.
		RCODE setNative(
			void *				pvField,		///< Field whose value is to be set.
			const char *		pszString,	///< Value to set.\  This should be a null-terminated native string.
			FLMUINT				uiEncId = 0	///< Encryption ID.\  If zero, the value will not be encrypted.\  If non-zero, the number
													///< should be the ID of an encryption definition record in the data dictionary.\  The
													///< encryption key for that encryption definition will be used to encrypt this value.
			);

		/// Set a field's value to a binary value.  The resulting data type for the field will be FLM_BINARY_TYPE.
		RCODE setBinary(
			void *			pvField,		///< Field whose value is to be set.
			const void *	pvBuf,		///< Binary value to set.
			FLMUINT			uiBufLen,	///< Length of binary data (in bytes).
			FLMUINT			uiEncId = 0	///< Encryption ID.\  If zero, the value will not be encrypted.\  If non-zero, the number
												///< should be the ID of an encryption definition record in the data dictionary.\  The
												///< encryption key for that encryption definition will be used to encrypt this value.
			);

		/// Set a field's value to a BLOB value.  The resulting data type for the field will be FLM_BLOB_TYPE.
		RCODE setBlob(
			void *			pvField,		///< Field whose value is to be set.
			FlmBlob *		pBlob,		///< BLOB value to set.
			FLMUINT			uiEncId = 0	///< Encryption ID.\  If zero, the value will not be encrypted.\  If non-zero, the number
												///< should be the ID of an encryption definition record in the data dictionary.\  The
												///< encryption key for that encryption definition will be used to encrypt this value.
			);

		#define INSERT_PREV_SIB			1
		#define INSERT_NEXT_SIB			2
		#define INSERT_FIRST_CHILD		3
		#define INSERT_LAST_CHILD		4

		/// Insert a new field into the ::FlmRecord object.  Insertion is relative to an already existing field.
		RCODE insert(
			void *			pvField,			///< Field that already exists in the record.\  New field is created relative to this
													///< field.
			FLMUINT			uiInsertAt,		///< Relative position with respect to pvField where new field is to be created.  It may
													///< be one of the following:\n
													///< - INSERT_PREV_SIB - insert new field as the previous sibling of pvField
													///< - INSERT_NEXT_SIB - insert new field as the next sibling of pvField
													///< - INSERT_FIRST_CHILD - insert new field as the first child of pvField
													///< - INSERT_LAST_CHILD - insert new field as the last child of pvField
			FLMUINT			uiFieldID,		///< Field number for new field.\  This should be a valid field number as defined in
													///< the data dictionary.
			FLMUINT			uiDataType,		///< Data type for new field.\  This may be one of the
													///< following:\n
													///< - FLM_TEXT_TYPE
													///< - FLM_NUMBER_TYPE
													///< - FLM_BINARY_TYPE
													///< - FLM_CONTEXT_TYPE
													///< - FLM_BLOB_TYPE
			void **			ppvField			///< Pointer to newly created field is returned here.
			);

		/// Insert a new field into the ::FlmRecord object.  New field is always inserted as the last field in the
		/// record.
		RCODE insertLast( 
			FLMUINT			uiLevel,			///< Nesting level for the new field.\  NOTE: If this is the first field created in the
													///< record, the nesting level must be zero.\  Only the first field added may have a
													///< nesting level of zero.\  If this is not the first field in the record, the nesting level
													/// must be between 1 and N+1, where N is the level of the current last field in the record.
			FLMUINT			uiFieldID,		///< Field number for new field.\  This should be a valid field number as defined in
													///< the data dictionary.
			FLMUINT			uiDataType,		///< Data type for new field.\  This may be one of the
													///< following:\n
													///< - FLM_TEXT_TYPE
													///< - FLM_NUMBER_TYPE
													///< - FLM_BINARY_TYPE
													///< - FLM_CONTEXT_TYPE
													///< - FLM_BLOB_TYPE
			void **			ppvField			///< Pointer to newly created field is returned here.
			);

		/// Remove a field from a record.  The field and all of its descendant fields will be removed from the record.
		FINLINE RCODE remove(
			void *			pvField			///< Field to be removed.\  NOTE: If the field has descendant fields, all of those fields
													///< will also be removed.
			)
		{
			return remove( getFieldPointer( pvField));
		}

		/// Return the root field of a record.  The root field is the field that is at level zero in the record.
		FINLINE void * root( void)
		{
			if( m_uiFldTblOffset)
			{
				return( (void *)1);
			}

			return( NULL);
		}

		/// Return the next sibling field of a field.  Returns NULL if there is no next sibling field.
		FINLINE void * nextSibling(
			void *			pvField		///< Field whose next sibling is to be returned.
			)
		{
			return( pvField 
						? getFieldVoid( nextSiblingField( 
								getFieldPointer( pvField)))
						: NULL);
		}

		/// Return the previous sibling field of a field.  Returns NULL if there is no previous sibling field.
		void * prevSibling( 
			void *			pvField		///< Field whose previous sibling is to be returned.
			);

		/// Return the first child field of a field.  Returns NULL if field has no child fields.
		FINLINE void * firstChild( 
			void *			pvField		///< Field whose first child is to be returned.
			)
		{
			return( pvField 
						? getFieldVoid( firstChildField( 
								getFieldPointer( pvField)))
						: NULL);
		}

		/// Return the last child field of a field.  Returns NULL if field has no child fields.
		FINLINE void * lastChild( 
			void *			pvField		///< Field whose last child is to be returned.
			)
		{
			return( getFieldVoid( 
				lastChildField( getFieldPointer( pvField))));
		}

		/// Return the parent field of a field.  Returns NULL if the field is the "root" field of the record.
		FINLINE void * parent( 
			void *			pvField		///< Field whose parent is to be returned.
			)
		{
			return parent( getFieldPointer( pvField));
		}

		/// Return the "next" field of a field.  If the field has child fields, then "next" means first child.  If the
		/// field has no child fields, but has siblings, then "next" means next sibling.  If the field has no next sibling
		/// field, then "next" means the next sibling of the field's parent field (if any), or the next sibling of the
		/// grandparent field (if any), etc.  If there is no next sibling to a parent, grandparent, etc., then NULL will be returned.
		FINLINE void * next(
			void *			pvField		/// Field whose "next" field is to be returned.
			)
		{
			return( pvField 
						? getFieldVoid( nextField( 
								getFieldPointer( pvField)))
						: NULL);
		}

		/// Return the "previous" field of a field.  If the field has a previous sibling field, and that previous sibling
		/// has children, grandchildren, etc., then "previous" means previous sibling's last child's, last child, last child (etc.).
		/// If the previous sibling field has no children, then "previous" means previous sibling.  If there is no previous sibling,
		/// then "previous" means parent field.  If the field is the "root" field, then NULL will be returned.
		FINLINE void * prev(
			void *			pvField		/// Field whose "previous" field is to be returned.
			)
		{
			return( pvField 
						? getFieldVoid( prevField( 
								getFieldPointer( pvField)))
						: NULL);
		}

		#define SEARCH_TREE		1	
		#define SEARCH_FOREST	2

		/// Find a field in a record that has a particular field number.  Search is conducted relative to some other field in the record.
		void * find(
			void *			pvStartField,	///< Field where search is to start.
			FLMUINT			uiFieldID,		///< Field number being searched for.
			FLMUINT			uiOccur = 1,	///< Occurrence being searched for.
			FLMUINT			uiFindOption = SEARCH_FOREST	///< Specifies how much of the record to search in (relative to pvStartField).  It
													///< may be one of the following:\n
													///< - SEARCH_FOREST - Search the sub-tree beginning at pvStartField, and all subtrees that are
													///< next siblings to pvStartField
													///< - SEARCH_TREE - Search only the sub-tree beginning at pvStartField
			);

		/// Find a field in a record that has a particular field path.  Search is conducted relative to some other field in the record.
		void * find( 
			void *			pvStartField,	///< field where search is to start.
			FLMUINT *		puiFieldPath,	///< Field path being searched for.
			FLMUINT			uiOccur = 1,	///< Occurrence being searched for.
			FLMUINT			uiFindOption = SEARCH_FOREST	///< Specifies how much of the record to search in (relative to pvStartField).  It
													///< may be one of the following:\n
													///< - SEARCH_FOREST - Search the sub-tree beginning at pvStartField, and all subtrees that are
													///< next siblings to pvStartField
													///< - SEARCH_TREE - Search only the sub-tree beginning at pvStartField
			);

		/// Get the nesting level of a field in the record.  The nesting level begins at 0 (root field), 1 (children of the root field),
		/// 2 (grandchildren of the root field), etc.
		FINLINE FLMUINT getLevel(
			void *			pvField	///< Field whose nesting level is to be returned.
			)
		{
			return( getFieldLevel( getFieldPointer( pvField)));
		}

		/// Get the field number for a field.
		FINLINE FLMUINT getFieldID(
			void *			pvField		///< Field whose field number is to be returned.
			)
		{
			return( getFieldPointer( pvField)->ui16FieldID);
		}

		/// Set the field number for a field.
		FINLINE void setFieldID(
			void *			pvField,		///< Field whose field number is to be set.
			FLMUINT			uiFieldID	///< Field number to be set.
			)
		{
			if( uiFieldID)
			{
				getFieldPointer( pvField)->ui16FieldID = (FLMUINT16)uiFieldID;
			}
		}

		/// Get a field's data type.
		FINLINE FLMUINT getDataType(
			void *			pvField		///< Field whose data type is to be returned.
			)
		{
			return( getFieldDataType( getFieldPointer( pvField)));
		}

		/// Get a field's data length.  NOTE: This is the field's internal storage length.
		FINLINE FLMUINT getDataLength(
			void *			pvField		///< Field whose data length is to be returned.
			)
		{
			return( getFieldDataLength( getFieldPointer( pvField)));
		}

		/// Determine if a field has any child fields.
		FINLINE FLMBOOL hasChild( 
			void *			pvField		///< Field that is to be checked for child fields.
			)
		{
			return( firstChildField( 
				getFieldPointer( pvField)) != NULL) ? TRUE : FALSE;
		}

		/// Determine if a field has a next sibling field.
		FINLINE FLMBOOL isLast( 
			void *			pvField	///< Field that is to be checked for next sibling fields.
			)
		{
			return( nextField( 
				getFieldPointer( pvField)) == NULL) ? TRUE : FALSE;
		}

		/// Set a flag for a field indicating if its value has been "right truncated."  Right truncation really only applies
		/// to binary data or strings.  It specifies that data at the end of the binary value or string value is missing.
		FINLINE void setRightTruncated( 
			void *			pvField,			///< Field that is to be marked as "right truncated" or "not right truncated."
			FLMBOOL			bTrueFalse		///< Flag indicating if field is to be marked as "right truncated" or "not right truncated."
			)
		{
			setRightTruncated( getFieldPointer( pvField), bTrueFalse);
		}

		/// Determine if field is marked as "right truncated."  This really only applies to binary data or strings.  It indicates
		/// that data at the end of the binary value or string value is missing.
		FINLINE FLMBOOL isRightTruncated(
			void *			pvField	///< Field that is to be checked to see if it is "right truncated."
			)
		{
			return( isRightTruncated( getFieldPointer( pvField)));
		}

		/// Set a flag for a field indicating if its value has been "left truncated."  Left truncation really only applies
		/// to binary data or strings.  It specifies that data at the beginning of the binary value or string value is missing.
		FINLINE void setLeftTruncated( 
			void *			pvField,			///< Field that is to be marked as "left truncated" or "not left truncated."
			FLMBOOL			bTrueFalse		///< Flag indicating if field is to be marked as "left truncated" or "not left truncated."
			)
		{
			setLeftTruncated( getFieldPointer( pvField), bTrueFalse);
		}

		/// Determine if field is marked as "left truncated."  This really only applies to binary data or strings.  It indicates
		/// that data at the beginning of the binary value or string value is missing.
		FINLINE FLMBOOL isLeftTruncated(
			void *			pvField	///< Field that is to be checked to see if it is "left truncated."
			)
		{
			return( isLeftTruncated( getFieldPointer( pvField)));
		}

		/// Get information about a field.  Information includes field number, level, data type, data length,
		/// encryption length, and encryption id.
		FINLINE RCODE getFieldInfo( 
			void *			pvField,			///< Field whose information is to be retrieved.
			FLMUINT *		puiFieldNum,	///< Field number is returned here.
			FLMUINT *		puiLevel,		///< Field's level is returned here.
			FLMUINT *		puiDataType,	///< Field's data type is returned here.
			FLMUINT *		puiLength,		///< Field's data length is returned here.
			FLMUINT *		puiEncLength,	///< Field's encryption length is returned here.\  NOTE: This parameter may be NULL.
			FLMUINT *		puiEncId			///< Field's encryption id is returned here.\  NOTE: This parameter may be NULL.
			)
		{
			FlmField *	pField = getFieldPointer( pvField);

			*puiFieldNum = pField->ui16FieldID;
			*puiLevel	= getLevel( pvField);
			*puiLength	= getDataLength( pvField);
			*puiDataType = getDataType( pvField);

			if (isEncryptedField( pField))
			{
				if (puiEncLength)
				{
					*puiEncLength = getEncryptedDataLength( pField);
				}

				if (puiEncId)
				{
					*puiEncId = getEncryptionID( pField);
				}
			}
			else
			{
				if (puiEncLength)
				{
					*puiEncLength = 0;
				}
				if (puiEncId)
				{
					*puiEncId = 0;
				}
			}

			return FERR_OK;
		}

		RCODE preallocSpace( 
			FLMUINT			uiFieldCount,
			FLMUINT			uiDataSize);

		RCODE allocStorageSpace(
			void *			pvField,
			FLMUINT			uiDataType,
			FLMUINT			uiLength,
			FLMUINT			uiEncLength,
			FLMUINT			uiEncId,
			FLMUINT			uiFlags,
			FLMBYTE **		ppDataPtr,
			FLMBYTE **		ppEncDataPtr);

		/// Get the total memory being consumed by this ::FlmRecord object.
		FLMUINT getTotalMemory( void);

		/// Get the amount of memory allocated to the ::FlmRecord object that is currently not used.
		FINLINE FLMUINT getFreeMemory( void)
		{
			return( ((m_uiFldTblSize - m_uiFldTblOffset) * sizeof( FlmField)) +
				(m_uiAvailFields * sizeof( FlmField)) +
				(getDataBufSize() - m_uiDataBufOffset));
		}

		/// Compress out unused memory within the ::FlmRecord object.  See also FlmRecord::getFreeMemory().
		RCODE compressMemory( void);

		/// Get the record ID (DRN) for this ::FlmRecord object.
		FINLINE FLMUINT getID( void)
		{
			return( m_uiRecordID);
		}

		/// Set the record ID (DRN) for this ::FlmRecord object.
		FINLINE void setID(
			FLMUINT			uiRecordID)
		{
			m_uiRecordID = uiRecordID;
		}

		/// Get the container this ::FlmRecord object belongs to.
		FINLINE FLMUINT getContainerID( void)
		{
			return( m_uiContainerID);
		}

		/// Set the container this ::FlmRecord object belongs to.
		FINLINE void setContainerID(
			FLMUINT			uiContainerID)
		{
			m_uiContainerID = uiContainerID;
		}

		/// Import a record from a file.  The record in the file should be formatted according
		/// to the specification for GEDCOM.
		RCODE importRecord(
			IF_FileHdl *		pFileHdl,	///< Open file handle where the data for the record is to be read from.
			F_NameTable *		pNameTable	///< Name table object that is to be used to translate field names to
													///< field numbers.
			);

		/// Import a record from a string buffer.  The record in the string should be formatted according
		/// to the specification for GEDCOM.
		RCODE importRecord(
			const char **		ppszBuffer,	///< Buffer containing the data that is to be imported.
			FLMUINT				uiBufSize,	///< Number of bytes in the buffer.
			F_NameTable *		pNameTable	///< Name table object that is to be used to translate field names to
													///< field numbers.
			);

		/// Import a record from a Gedcom ::NODE tree.
		RCODE importRecord(
			NODE *			pNode	///< Node that is the root of a Gedcom tree to be imported.
			);

		/// Export a record to a Gedcom ::NODE tree.
		RCODE exportRecord(
			HFDB			hDb,			///< Database handle.\  The root node of the Gedcom tree will be associated with this handle.
			F_Pool *		pPool,		///< Memory pool for allocating ::NODE structures and space for field data.
			NODE **		ppNode		///< Root of the Gedcom ::NODE tree will be returned here.
			);

		/// Determine if the ::FlmRecord object is read-only.
		FINLINE FLMBOOL isReadOnly( void)
		{
			return( (m_uiFlags & RCA_READ_ONLY_FLAG) ? TRUE : FALSE);
		}

		/// Determine if the ::FlmRecord object is cached in FLAIM's record cache.
		FINLINE FLMBOOL isCached( void)
		{
			return( (m_uiFlags & RCA_CACHED) ? TRUE : FALSE);
		}

		/// Determine if the ::FlmRecord object is a prior version of the record or if it is the current version.
		FINLINE FLMBOOL isOldVersion( void)
		{
			return( (m_uiFlags & RCA_OLD_VERSION) ? TRUE : FALSE);
		}

		/// Determine if the ::FlmRecord object can be changed.
		FINLINE FLMBOOL isMutable( void)
		{
			return( (m_uiFlags & (RCA_READ_ONLY_FLAG | RCA_CACHED)) ? FALSE : TRUE);
		}
		
		/// Get a pointer to a field's data.  The returned pointer will be pointing to the data in FLAIM's internal
		/// storage format.
		FINLINE const FLMBYTE * getDataPtr(
			void *			pvField		///< Field whose data pointer is to be returned.
			)
		{
			return( getDataPtr( getFieldPointer( pvField)));
		}

		/// Determine if a field is encrypted.
		FINLINE FLMBOOL isEncryptedField(
			void *			pvField		///< Field that is to be checked to see if it is encrypted.
			)
		{
			return isEncryptedField( getFieldPointer( pvField));
		}

		/// Get a pointer to a field's encrypted data.  NOTE: This method should NOT be called if
		/// the field is not encrypted.  Call FlmRecord::isEncryptedField() to determine if the field
		/// is encrypted.  If called for a non-encrypted field, it will return NULL.  In debug
		/// mode it will also assert.
		FINLINE const FLMBYTE * getEncryptionDataPtr(
			void *			pvField		///< Field whose encrypted data pointer is to be retrieved.
			)
		{
			return( getEncryptionDataPtr( getFieldPointer( pvField)));
		}

		/// Get a field's encrypted data length.  NOTE: This method should NOT be called if
		/// the field is not encrypted.  Call FlmRecord::isEncryptedField() to determine if the field
		/// is encrypted.  If called for a non-encrypted field, it will return 0.  In debug
		/// mode it will also assert.
		FINLINE FLMUINT getEncryptedDataLength(
			void *			pvField		///< Field whose encrypted data length is to be retrieved.
			)
		{
			return getEncryptedDataLength( getFieldPointer(pvField));
		}
		
		/// Get a field's encryption id.  NOTE: This method should NOT be called if
		/// the field is not encrypted.  Call FlmRecord::isEncryptedField() to determine if the field
		/// is encrypted.  If called for a non-encrypted field, it will return 0.  In debug
		/// mode it will also assert.
		FLMUINT getEncryptionID( 
			void *			pvField		///< Field whose encryption ID is to be retrieved.
			)
		{
			return getEncryptionID( getFieldPointer( pvField));
		}
			
		/// Get a field's encryption flags.  NOTE: This method should NOT be called if
		/// the field is not encrypted.  Call FlmRecord::isEncryptedField() to determine if the field
		/// is encrypted.  If called for a non-encrypted field, it will return 0.  In debug
		/// mode it will also assert.
		FLMUINT getEncFlags(
			void *			pvField		///< Field whose encryption flags are to be returned.
			)
		{
			return getEncFlags( getFieldPointer( pvField));
		}
		
		// Set a field's encryption flags.  This method should only be used internally by FLAIM.  It
		// is not intended for an application to use.
		void setEncFlags(
			void *			pvField,
			FLMUINT			uiFlags)
		{
			setEncFlags( getFieldPointer( pvField), uiFlags);
		}

		/// Determine if a record has a level one field ID table.
		FINLINE FLMBOOL fieldIdTableEnabled( void)
		{
			return( (m_uiFlags & RCA_FIELD_ID_TABLE_ENABLED)
					  ? TRUE
					  : FALSE);
		}
		
		/// Set a flag in the record that will cause it to generate a level
		/// one field ID table.
		FINLINE void enableFieldIdTable( void)
		{
			m_uiFlags |= RCA_FIELD_ID_TABLE_ENABLED;
		}
		
		/// Create a level one field ID table in a record.
		RCODE createFieldIdTable(
			FLMBOOL	bTruncateTable		///< Truncate the field id table after sorting?
			);
			
		FINLINE FLMBYTE * getFieldIdTbl( void)
		{
			return m_pucFieldIdTable;
		}
		
		RCODE truncateFieldIdTable( void);
		
		void sortFieldIdTable( void);
		
		/// Find a level one field ID in a record.
		void * findLevelOneField(
			FLMUINT		uiFieldID,			///< Field number of field to be found.
			FLMBOOL		bFindInclusive,	///< OK to find next field after uiFieldID?
			FLMUINT *	puiFieldPos			///< Field position in field ID table is returned here.
			);

		/// Determine if the level one field at the specified position in the field ID table
		/// matches the passed in field ID.\  If so, return that field pointer.
		void * getLevelOneField(
			FLMUINT	uiFieldId,				///< Field id to be matched.
			FLMUINT	uiLevelOnePosition	///< Level one field position to be checked.
			);

		/// Get the next level one field after the position specified.\  Make sure
		/// that the next field has the same field ID as the current field.
		void * nextLevelOneField(
			FLMUINT *	puiFieldPos,			///< Current level one field position.\  Returns
														///< the next level one field position.
			FLMBOOL		bFieldIdsMustMatch	///< Specifies whether the field ID of the next
														///< field in the field ID table must match the
														///< field ID of the current field position.
			);

		/// Get the field ID of the field that is in the specified position in the
		/// field ID table.
		FLMUINT getLevelOneFieldId(
			FLMUINT	uiLevelOnePosition		///< Level one field position whose field ID is to
														///< be returned.
			);

		void * locateFieldByPosition(
			FLMUINT			uiPosition);

		
		RCODE checkField(
			FlmField *			pFld);

#define FLD_HAVE_ENCRYPTED_DATA		0x01
#define FLD_HAVE_DECRYPTED_DATA		0x02
#define FLD_PICKET_FENCE_SIZE			8		// Only used in debug builds in encrypted fields
														// Represents the size of two picket fences.
#define FLD_RAW_FENCE					"RAWD"
#define FLD_ENC_FENCE					"ENCD"

	private:

		FLMINT Release( 
			FLMBOOL			bMutexLocked);

		static void FLMAPI objectAllocInit(
			void *			pvAlloc,
			FLMUINT			uiSize);
			
		void * parent( 
			FlmField *		pField);

		FINLINE FLMUINT getFieldLevel(
			FlmField *		pField)
		{
			return( (pField->ui8TypeAndLevel & 0xE0) >> 5);
		}

		FINLINE void setReadOnly( void)
		{
			m_uiFlags |= RCA_READ_ONLY_FLAG;
		}

		FINLINE void setCached( void)
		{
			m_uiFlags |= RCA_CACHED;
		}

		FINLINE void clearCached( void)
		{
			m_uiFlags &= ~RCA_CACHED;
		}

		FINLINE void setOldVersion( void)
		{
			m_uiFlags |= RCA_OLD_VERSION;
		}

		FINLINE void clearOldVersion( void)
		{
			m_uiFlags &= ~RCA_OLD_VERSION;
		}

		void * getFieldVoid(
			FlmField *		pField);

		FlmField * getFieldPointer(
			void *			pvField);

		FLMBYTE * getDataPtr(
			FlmField *		pField);

		FlmField * nextSiblingField(
			FlmField *		pField);

		FlmField * lastSubTreeField( 
			FlmField *		pField);

		RCODE createField( 
			FlmField *		pPrevField,
			FlmField **		ppNewField); 

		RCODE removeFields(
			FlmField *		pFirstField,
			FlmField *		pLastField = NULL);

		RCODE copyFields( 
			FlmField *		pSrcFields);

		RCODE getNewDataPtr(
			FlmField *		pField,
			FLMUINT			uiDataType,
			FLMUINT			uiNewLength,
			FLMUINT			uiNewEncLength,
			FLMUINT			uiEncId,
			FLMUINT			uiFlags,
			FLMBYTE **		ppDataPtr,
			FLMBYTE **		ppEncDataPtr);

		FINLINE FlmField * firstChildField( 
			FlmField *		pField)
		{
			FLMUINT	uiLevel = getLevel( getFieldVoid( pField));

			return ((pField = nextField( pField)) != NULL && 
						getLevel( getFieldVoid( pField)) > uiLevel)
							? pField
							: NULL;
		}

		FlmField * lastChildField(
			FlmField *		pField);

		FlmField * getFirstField( void);

		FlmField * getLastField( void);

		FlmField * prevField(
			FlmField *		pField);

		FlmField * nextField(
			FlmField *		pField);

		RCODE setFieldLevel(
			FlmField *		pField,
			FLMUINT			uiLevel);

		void setFieldDataType(
			FlmField *		pField,
			FLMUINT			uiDataType);

		FINLINE FLMUINT getFieldDataType(
			FlmField *			pField)
		{
			FLMUINT	uiFldType;

			if( (uiFldType = pField->ui8TypeAndLevel & 0x07) <= 3)
			{
				return( uiFldType);
			}

			return( FLM_BLOB_TYPE);
		}

		FLMUINT getFieldDataLength(
			FlmField *			pField);
	
		FINLINE FlmField * getFieldTable( void)
		{
			return( (FlmField *)(m_pucBuffer + FLM_ALIGN_SIZE));
		}

		FINLINE FLMUINT getDataBufSize( void)
		{
			return( (FLMUINT)((m_pucBuffer + m_uiBufferSize) - 
				(FLMBYTE *)(&(getFieldTable()[ m_uiFldTblSize]))));
		}

		FINLINE FLMBYTE * getDataBufPtr( void)
		{
			return( (FLMBYTE *)(&(getFieldTable()[ m_uiFldTblSize])));
		}

		RCODE compactMemory( void);

		FLMBOOL isEncryptedField(
			FlmField *			pField);

		FLMBYTE * getEncryptionDataPtr(
			FlmField *		pField);
			
		FLMUINT getEncryptedDataLength(
			FlmField *		pField);

		FLMUINT getEncryptionID( 
			FlmField *			pField);
			
		FLMUINT getEncFlags(
			FlmField *			pField);
		
		void setEncFlags(
			FlmField *		pField,
			FLMUINT			uiFlags);
			
		void setEncHeader(
			FLMBYTE *		pBuffer,
			FLMUINT			uiFlags,
			FLMUINT			uiEncId,
			FLMUINT			uiNewLength,
			FLMUINT			uiEncNewLength);

		void setRightTruncated( 
			FlmField *		pField,
			FLMBOOL			bTrueFalse);

		void setLeftTruncated( 
			FlmField *		pField,
			FLMBOOL			bTrueFalse);

		FINLINE FLMBOOL isRightTruncated(
			FlmField *		pField)
		{
			return( pField->ui8TypeAndLevel & 
							FLD_DATA_RIGHT_TRUNCATED
						? TRUE 
						: FALSE);
		}

		FINLINE FLMBOOL isLeftTruncated(
			FlmField *		pField)
		{
			return( pField->ui8TypeAndLevel & 
							FLD_DATA_LEFT_TRUNCATED
						? TRUE 
						: FALSE);
		}
		
		RCODE remove(
			FlmField *		pField);

		RCODE addToFieldIdTable(
			FLMUINT16		ui16FieldId,
			FIELDLINK		ui32FieldOffset);

		RCODE removeFromFieldIdTable(
			FLMUINT16		ui16FieldId,
			FIELDLINK		ui32FieldOffset);
			
		FIELD_ID * findFieldId(
			FLMUINT16	ui16FieldId,
			FIELDLINK	ui32FieldOffset,
			FLMUINT *	puiInsertPos);
			
		FINLINE FLMUINT fieldIdTableByteSize( void)
		{
			FLMUINT	uiTableArraySize = getFieldIdTableArraySize( m_pucFieldIdTable);
			return calcFieldIdTableByteSize( uiTableArraySize);
		}
			
		FLMUINT		m_uiContainerID;
		FLMUINT		m_uiRecordID;
		FLMUINT		m_uiFlags;
		FLMBYTE *	m_pucBuffer;
		FLMUINT		m_uiBufferSize;
		FLMUINT		m_uiFldTblSize;
		FLMUINT		m_uiFldTblOffset;
		FLMUINT		m_uiDataBufOffset;
		FLMBOOL		m_bHolesInData;
		FLMUINT		m_uiAvailFields;
		FIELDLINK	m_uiFirstAvail;
		FLMBYTE *	m_pucFieldIdTable;

		friend struct FlmRecordExt;
		friend class F_RecRelocator;
		friend class F_RecBufferRelocator;
		friend class F_Rfl;
	};

	FLMXPC RCODE FLMAPI flmCurPerformRead(
		eFlmFuncs		eFlmFuncId,
		HFCURSOR 		hCursor,
		FLMBOOL			bReadForward,
		FLMBOOL			bSetFirst,
		FLMUINT *		puiSkipCount,
		FlmRecord **	ppRecord,
		FLMUINT *		puiDrn );

	/// Positions to and retrieves the first record in the query result set.
	/// \ingroup queryret
	FINLINE RCODE FlmCursorFirst(
		HFCURSOR 			hCursor,			///< Handle to query object.
		FlmRecord **		ppRecord			///< Pointer to found record, if any, is returned here.\  NULL is returned if no record was found.
		)
	{
		return flmCurPerformRead( FLM_CURSOR_FIRST, 
			hCursor, TRUE, TRUE, NULL, ppRecord, NULL);
	}

	/// Positions to and retrieves the last record in the query result set.
	/// \ingroup queryret
	FINLINE RCODE FlmCursorLast(
		HFCURSOR 			hCursor,			///< Handle to query object.
		FlmRecord **		ppRecord			///< Pointer to found record, if any, is returned here.\  NULL is returned if no record was found.
		)
	{
		return flmCurPerformRead( FLM_CURSOR_LAST, hCursor,
			FALSE, TRUE, NULL, ppRecord, NULL);
	}

	/// Positions to and retrieves the next record in the query result set.
	/// \ingroup queryret
	FINLINE RCODE FlmCursorNext(
		HFCURSOR 			hCursor,			///< Handle to query object.
		FlmRecord **		ppRecord			///< Pointer to found record, if any, is returned here.\  NULL is returned if no record was found.
		)
	{
		return flmCurPerformRead( FLM_CURSOR_NEXT, hCursor, 
			TRUE, FALSE, NULL, ppRecord, NULL);
	}

	/// Positions to and retrieves the previous record in the query result set.
	/// \ingroup queryret
	FINLINE RCODE FlmCursorPrev(
		HFCURSOR 			hCursor,			///< Handle to query object.
		FlmRecord **		ppRecord			///< Pointer to found record, if any, is returned here.\  NULL is returned if no record was found.
		)
	{
		return flmCurPerformRead( FLM_CURSOR_PREV, hCursor,
			FALSE, FALSE, NULL, ppRecord, NULL);
	}

	/// Positions to the first record in the query result set and retrieves the record's DRN.
	/// \ingroup queryret
	FINLINE RCODE FlmCursorFirstDRN(
		HFCURSOR 			hCursor,			///< Handle to query object.
		FLMUINT *			puiDrn			///< DRN is returned here.
		)
	{
		return flmCurPerformRead( FLM_CURSOR_FIRST_DRN, hCursor, 
			TRUE, TRUE, NULL, NULL, puiDrn);
	}

	/// Positions to the last record in the query result set and retrieves the record's DRN.
	/// \ingroup queryret
	FINLINE RCODE FlmCursorLastDRN(
		HFCURSOR 			hCursor,			///< Handle to query object.
		FLMUINT *			puiDrn			///< DRN is returned here.
		)
	{
		return flmCurPerformRead( FLM_CURSOR_LAST_DRN, hCursor, 
			FALSE, TRUE, NULL, NULL, puiDrn);
	}

	/// Positions to the next record in the query result set and retrieves the record's DRN.
	/// \ingroup queryret
	FINLINE RCODE FlmCursorNextDRN(
		HFCURSOR 			hCursor,			///< Handle to query object.
		FLMUINT *			puiDrn			///< DRN is returned here.
		)
	{
		return flmCurPerformRead( FLM_CURSOR_NEXT_DRN, hCursor, 
			TRUE, FALSE, NULL, NULL, puiDrn);
	}

	/// Positions to the previous record in the query result set and retrieves the record's DRN.
	/// \ingroup queryret
	FINLINE RCODE FlmCursorPrevDRN(
		HFCURSOR 			hCursor,			///< Handle to query object.
		FLMUINT *			puiDrn			///< DRN is returned here.
		)
	{
		return flmCurPerformRead( FLM_CURSOR_PREV_DRN, hCursor,
			FALSE, FALSE, NULL, NULL, puiDrn);
	}

	/// Retrieve current record from query result set.
	/// \ingroup queryret
	FLMXPC RCODE FLMAPI FlmCursorCurrent(
		HFCURSOR 			hCursor,			///< Handle to query object.
		FlmRecord **		ppRecord			///< Pointer to found record, if any, is returned here.\  NULL is returned if no record was found.
		);

	/// Retrieve the DRN of the current recrord in query result set.
	/// \ingroup queryret
	FLMXPC RCODE FLMAPI FlmCursorCurrentDRN(
		HFCURSOR 			hCursor,			///< Handle to query object.
		FLMUINT *			puiDrn			///< DRN is returned here.
		);

	/// Position relative to the current record (forward or backward) in the query result set
	/// and retrieve the record positioned to.
	/// \ingroup queryret
	FLMXPC RCODE FLMAPI FlmCursorMoveRelative(
		HFCURSOR 			hCursor,			///< Handle to query object.
		FLMINT *				piPosition,		///< On input *piPosition indicates the relative position to move within the
													///< query result set.\  A negative value will move the position back
													///< *piPosition records and a positive value will move the position
													///< forward *piPosition records.\  On output, *piPosition will return
													///< the relative position after the move.\  This should always equal
													///< the input position unless an error is returned.
		FlmRecord **		ppRecord			///< Pointer to found record, if any, is returned here.\  NULL is returned if no record was found.
		);

	/// Get record count for a query result set.  NOTE: This function generates the query result set, counting the
	/// records as it goes.  Therefore, it may take a long time to compute, depending on the size of the result
	/// set and whether or not indexes can be used to optimize the query.
	/// \ingroup queryret
	FLMXPC RCODE FLMAPI FlmCursorRecCount(
		HFCURSOR 			hCursor,			///< Handle to query object.
		FLMUINT *			puiCount			///< Count of records in the query result set is returned here.
		);

	/// Determine the relative position of two records in a query's result set.  This only makes sense if the
	/// query is optimized using an index.  The function does the following:\n
	/// -# Reads the two records from the database
	/// -# Uses the query's index to get the index keys contained in the two records
	/// -# Compares the keys to determine which is greater
	/// -# Optionally gets an count of the keys between the two keys (count is inclusive).
	/// \ingroup querycomp
	FLMXPC RCODE FLMAPI FlmCursorCompareDRNs(
		HFCURSOR 			hCursor,			///< Handle to query object.
		FLMUINT				uiDRN1,			///< DRN of first record to be compared.
		FLMUINT				uiDRN2,			///< DRN of second record to be compated.
		FLMUINT				uiTimeLimit,	///< Timeout for this operation.\  Timeout is in seconds.
		FLMINT *				piCmpResult,	///< Comparison results is returned here.
		FLMBOOL *			pbTimedOut,		///< Did the function time out?
		FLMUINT *			puiKeyCount		///< Count of index keys betwen the two records (inclusive).
		);
			
	/// Test a record to see if it passes the query criteria.
	/// \ingroup querycomp
	FLMXPC RCODE FLMAPI FlmCursorTestRec(
		HFCURSOR 			hCursor,			///< Handle to query object.
		FlmRecord *			pRecord,			///< Record to be tested against the query criteria.
		FLMBOOL *			pbIsMatch		///< Flag is returned here indicating whether or not the record matches the criteria.
		);

	/// Retrieve and test a record (using a DRN) to see if it passes the query criteria.
	/// \ingroup querycomp
	FLMXPC RCODE FLMAPI FlmCursorTestDRN(
		HFCURSOR 			hCursor,			///< Handle to query object.
		FLMUINT				uiDRN,			///< DRN of record to be tested against the query criteria.\  FLAIM will retrieve the
													///< record and test it.\  The container is the container that was passed into
													///< FlmCursorInit().
		FLMBOOL *			pbIsMatch		///< Flag is returned here indicating whether or not the record matches the criteria.
		);

	/// Types of corruption that can be reported by FlmDbCheck().
	typedef enum
	{
		FLM_NO_CORRUPTION = 0,				///< 0
		FLM_BAD_CHAR,							///< 1
		FLM_BAD_ASIAN_CHAR,					///< 2
		FLM_BAD_CHAR_SET,						///< 3
		FLM_BAD_TEXT_FIELD,					///< 4
		FLM_BAD_NUMBER_FIELD,				///< 5
		FLM_BAD_CONTEXT_FIELD,				///< 6
		FLM_BAD_FIELD_TYPE,					///< 7
		FLM_BAD_IX_DEF,						///< 8
		FLM_MISSING_REQ_KEY_FIELD,			///< 9
		FLM_BAD_TEXT_KEY_COLL_CHAR,		///< 10
		FLM_BAD_TEXT_KEY_CASE_MARKER,		///< 11
		FLM_BAD_NUMBER_KEY,					///< 12
		FLM_BAD_CONTEXT_KEY,					///< 13
		FLM_BAD_BINARY_KEY,					///< 14
		FLM_BAD_DRN_KEY,						///< 15
		FLM_BAD_KEY_FIELD_TYPE,				///< 16
		FLM_BAD_KEY_COMPOUND_MARKER,		///< 17
		FLM_BAD_KEY_POST_MARKER,			///< 18
		FLM_BAD_KEY_POST_BYTE_COUNT,		///< 19
		FLM_BAD_KEY_LEN,						///< 20
		FLM_BAD_LFH_LIST_PTR,				///< 21
		FLM_BAD_LFH_LIST_END,				///< 22
		FLM_BAD_PCODE_LIST_END,				///< 23
		FLM_BAD_BLK_END,						///< 24
		FLM_KEY_COUNT_MISMATCH,				///< 25
		FLM_REF_COUNT_MISMATCH,				///< 26
		FLM_BAD_CONTAINER_IN_KEY,			///< 27
		FLM_BAD_BLK_HDR_ADDR,				///< 28
		FLM_BAD_BLK_HDR_LEVEL,				///< 29
		FLM_BAD_BLK_HDR_PREV,				///< 30
		FLM_BAD_BLK_HDR_NEXT,				///< 31
		FLM_BAD_BLK_HDR_TYPE,				///< 32
		FLM_BAD_BLK_HDR_ROOT_BIT,			///< 33
		FLM_BAD_BLK_HDR_BLK_END,			///< 34
		FLM_BAD_BLK_HDR_LF_NUM,				///< 35
		FLM_BAD_AVAIL_LIST_END,				///< 36
		FLM_BAD_PREV_BLK_NEXT,				///< 37
		FLM_BAD_FIRST_ELM_FLAG,				///< 38
		FLM_BAD_LAST_ELM_FLAG,				///< 39
		FLM_BAD_LEM,							///< 40
		FLM_BAD_ELM_LEN,						///< 41
		FLM_BAD_ELM_KEY_SIZE,				///< 42
		FLM_BAD_ELM_PKC_LEN,					///< 43
		FLM_BAD_ELM_KEY_ORDER,				///< 44
		FLM_BAD_ELM_KEY_COMPRESS,			///< 45
		FLM_BAD_CONT_ELM_KEY,				///< 46
		FLM_NON_UNIQUE_FIRST_ELM_KEY,		///< 47
		FLM_BAD_ELM_FLD_OVERHEAD,			///< 48
		FLM_BAD_ELM_FLD_LEVEL_JUMP,		///< 49
		FLM_BAD_ELM_FLD_NUM,					///< 50
		FLM_BAD_ELM_FLD_LEN,					///< 51
		FLM_BAD_ELM_FLD_TYPE,				///< 52
		FLM_BAD_ELM_END,						///< 53
		FLM_BAD_PARENT_KEY,					///< 54
		FLM_BAD_ELM_DOMAIN_SEN,				///< 55
		FLM_BAD_ELM_BASE_SEN,				///< 56
		FLM_BAD_ELM_IX_REF,					///< 57
		FLM_BAD_ELM_ONE_RUN_SEN,			///< 58
		FLM_BAD_ELM_DELTA_SEN,				///< 59
		FLM_BAD_ELM_DOMAIN,					///< 60
		FLM_BAD_LAST_BLK_NEXT,				///< 61
		FLM_BAD_FIELD_PTR,					///< 62
		FLM_REBUILD_REC_EXISTS,				///< 63
		FLM_REBUILD_KEY_NOT_UNIQUE,		///< 64
		FLM_NON_UNIQUE_ELM_KEY_REF,		///< 65
		FLM_OLD_VIEW,							///< 66
		FLM_COULD_NOT_SYNC_BLK,				///< 67
		FLM_IX_REF_REC_NOT_FOUND,			///< 68
		FLM_IX_KEY_NOT_FOUND_IN_REC,		///< 69
		FLM_DRN_NOT_IN_KEY_REFSET,			///< 70
		FLM_BAD_BLK_CHECKSUM,				///< 71
		FLM_BAD_LAST_DRN,						///< 72
		FLM_BAD_FILE_SIZE,					///< 73
		FLM_BAD_AVAIL_BLOCK_COUNT,			///< 74
		FLM_BAD_DATE_FIELD,					///< 75
		FLM_BAD_TIME_FIELD,					///< 76
		FLM_BAD_TMSTAMP_FIELD,				///< 77
		FLM_BAD_DATE_KEY,						///< 78
		FLM_BAD_TIME_KEY,						///< 79
		FLM_BAD_TMSTAMP_KEY,					///< 80
		FLM_BAD_BLOB_FIELD,					///< 81
		FLM_BAD_PCODE_IXD_TBL,				///< 82
		FLM_DICT_REC_ADD_ERR,				///< 83
		FLM_BAD_FIELD_FLAG,					///< 84
		FLM_BAD_FOP,							///< 85
		FLM_LAST_CORRUPT_ERROR
	} eCorruptionType;

	/// Structure containing statistics collected during FlmDbCheck() for a particular category of blocks in the database.
	typedef struct
	{
		FLMUINT				uiBlockCount;					///< Total blocks found in the database that were in the in the block category.
		FLMUINT64			ui64BytesUsed;					///< Total bytes used in the blocks.
		FLMUINT64			ui64ElementCount;				///< Total elements in the blocks.\  NOTE: This only applies to b-tree blocks.
		FLMUINT64 			ui64ContElementCount;		///< Total continuation elements in the blocks.\  NOTE: This only applies to b-tree blocks.
		FLMUINT64 			ui64ContElmBytes;				///< Total bytes in continuation elements.\  NOTE: This only applies to b-tree blocks.
		eCorruptionType	eCorruption;					///< First corruption error found in blocks in this block category.
		FLMUINT				uiNumErrors;					///< Total corruption errors found in blocks in this block category.
	} BLOCK_INFO;

	/// Locations of corruptions in the database.
	typedef enum
	{
		LOCALE_NONE = 0,
		LOCALE_LFH_LIST,										///< Corruption was found in the list of logical file blocks.
		LOCALE_AVAIL_LIST = 3,								///< Corruption was found in the list of available blocks.
		LOCALE_B_TREE,											///< Corruption was found in an index or container b-tree block.
		LOCALE_IXD_TBL,										///< Corruption was found in index table.
		LOCALE_INDEX											///< Corruption was logical index corruption.
	} eCorruptionLocale;

	/// Structure used to create a linked list of index keys from a record.
	typedef struct REC_KEY
	{
		FlmRecord *				pKey;							///< Pointer to index key that was generated from a record.
		REC_KEY *				pNextKey;					///< Pointer to next key in the record.
	} REC_KEY;

	/// Structure containing information about a specific corruption that is being reported by FlmDbCheck().
	/// This structure is passed to the callback function when eStatusType::FLM_PROBLEM_STATUS status is reported.
	typedef struct
	{
		eCorruptionType		eCorruption;				///< Type of corruption being reported.
		eCorruptionLocale		eErrLocale;					///< Location of the corruption in the database.
		FLMUINT					uiErrLfNumber;				///< If eErrLocale is eCorruptionLocale::LOCALE_B_TREE or
																	///< eCorruptionLocale::LOCALE_IXD_TBL or eCorruptionLocale::LOCALE_INDEX this
																	///< will contain the index or container number.
		FLMUINT					uiErrLfType;				///< If eErrLocale is eCorruptionLocale::LOCALE_B_TREE, this will contain either LF_INDEX or
																	///< LF_CONTAINER.
		FLMUINT					uiErrBTreeLevel;			///< If eErrLocale is eCorruptionLocale::LOCALE_B_TREE, this will contain the level in the
																	///< b-tree where the corruption was found.\  A value of 0xFF means that
																	///< the b-tree level is unknown.
		FLMUINT					uiErrBlkAddress;			///< If non-zero, this contains the address of the block where the corruption was found.
		FLMUINT					uiErrParentBlkAddress;	///< If non-zero, this contains the address of the parent block of the block where
																	///< the corruption was found.\  NOTE: This will only be set when eErrLocale is
																	///< eCorruptionLocale::LOCALE_B_TREE.
		FLMUINT					uiErrElmOffset;			///< If non-zero, this is the offset of the element within the block where the
																	///< corruption was found.\  NOTE: This will only be set when eErrLocale is
																	///< eCorruptionLocale::LOCALE_B_TREE.
		FLMUINT					uiErrDrn;					///< If non-zero, this is the DRN of the record where the corruption was found.\  NOTE:
																	///< It may also be set to indicate a reference from an index if the corruption is
																	///< in an index.
		FLMUINT					uiErrElmRecOffset;		///< If non-zero, this is the offset within the "record" part of the element where
																	///< the corruption was found.\  NOTE: This will only be set when eErrLocale is
																	///< eCorruptionLocale::LOCALE_B_TREE.
		FLMUINT					uiErrFieldNum;				///< If non-zero, this is the field number where the corruption was found.
		const FLMBYTE *		pBlk;							///< If non-NULL, this is a pointer to block where corruption was found.

		// Index corruption information

		FlmRecord *				pErrIxKey;					///< If non-NULL, this will contain a pointer to the key from an index for an index
																	///< logical corruption.\  NOTE: This will only be set when eErrLocale is
																	///< eCorruptionLocale::LOCALE_INDEX.
		FlmRecord *				pErrRecord;					///< If non-NULL, this will contain a pointer to the record involved in an index
																	///< logical corruption.\  NOTE: This will only be set when eErrLocale is
																	///< eCorruptionLocale::LOCALE_INDEX.
		REC_KEY *				pErrRecordKeyList;		///< If non-NULL, this will contain a pointer to a linked list of keys from the record that
																	///< was involved in an index logical corruption.\  NOTE: This will only be set when
																	///< eErrLocale is eCorruptionLocale::LOCALE_INDEX.

	} CORRUPT_INFO;

	// Logical file types

	#define				LF_CONTAINER	1
	#define				LF_INDEX			3
	#define				LF_INVALID		15

	/// Statistics for a particular level in an index or container b-tree.\  These statistics are gathered by FlmDbCheck().
	typedef struct
	{
		FLMUINT64		ui64KeyCount;			///< Total keys at this level of the b-tree.
		BLOCK_INFO		BlockInfo;				///< Statistics for blocks at this level of the b-tree.
	} LEVEL_INFO;

	/// Statistics gathered by FlmDbCheck() for a particular logical file (index or container).
	typedef struct LF_STATS
	{
		FLMUINT			uiLfType;				///< Logical file type.\  Will be LF_INDEX or LF_CONTAINER.
		FLMUINT			uiIndexNum;				///< Index number.  Only set if uiLfType == LF_INDEX.
		FLMUINT			uiContainerNum;		///< Container number.\  If uiLfType == LF_INDEX, this is the container number that is
														///< associated with the index.
		FLMUINT64		ui64FldRefCount;		///< If uiLfType == LF_INDEX, this is the number of records referenced from the index.\  If
														///< uiLfType == LF_CONTAINER, this is the number of fields in the records in the container.
		FLMUINT			uiNumLevels;			///< Number of levels in the b-tree for this index or container.
		LEVEL_INFO *	pLevelInfo;				///< Statistics for each level of the b-tree.
	} LF_STATS;

	/// Structure returned during status callback from FlmDbCheck().  This structure is passed to the callback function when the
	/// eStatusType::FLM_CHECK_STATUS status is reported.
	typedef struct
	{
		void *			AppArg;								///< Application data.\  This is the application data pointer that was passed into
																	///< FlmDbCheck().
		FLMINT			iCheckPhase;						///< Phase of the check currently being performed by FlmDbCheck().\  It may be one
																	///< of the following:\n
																	///< - CHECK_LFH_BLOCKS - Checking logical file header blocks
																	///< - CHECK_B_TREE - Checking container and index b-trees
																	///< - CHECK_AVAIL_BLOCKS - Checking available block list
																	///< - CHECK_RS_SORT - Sorting result set for index keys
																	///< - CHECK_FINISHED - Database check is finished
	#define					CHECK_LFH_BLOCKS		1
	#define					CHECK_B_TREE			2
	#define					CHECK_AVAIL_BLOCKS	3
	#define					CHECK_RS_SORT			4
	#define					CHECK_FINISHED			5
		FLMBOOL			bStartFlag;							///< Flag indicating if we are at the beginning of the check phase specified by iCheckPhase.
		FLMUINT			uiCurrLF;							///< Logical file currently being processed.
		FLMUINT			uiLfNumber;							///< Current logical file number.
		FLMUINT			uiLfType;							///< Logical file type.\  May be one of the
																	///< following:\n
																	///< - LF_INDEX - Index
																	///< - LF_CONTAINER - Container
		FLMUINT64		ui64DatabaseSize;					///< Total database size.
		FLMUINT64		ui64BytesExamined;				///< Total bytes examined so far in the database.
		FLMUINT			uiNumProblemsFixed;				///< Total problems fixed.\  NOTE: This count only refers to logical index corruptions.
		FLMBOOL			bPhysicalCorrupt;					///< Flag indicating if physical database corruptions were found.
		FLMBOOL			bLogicalIndexCorrupt;			///< Flag indicating if logical index corruptions were found.
		FLMUINT			uiLogicalIndexCorruptions;		///< Total number of logical index corruptions that were found.
		FLMUINT			uiLogicalIndexRepairs;			///< Total number of logical index corruptions that were repaired.
		FLMUINT			uiNumFields;						///< Total fields defined in the dictionary.
		FLMUINT			uiNumIndexes;						///< Total indexes in the database.
		FLMUINT			uiNumContainers;					///< Total containers in the database.
		FLMUINT			uiNumLogicalFiles;				///< Total logical files (indexes & containers) in the database.
		LF_STATS *		pLfStats;							///< Statistics collected for all logical files (indexes and containers) in the database.
		BLOCK_INFO		AvailBlocks;						///< Statistics collected on available blocks.
		BLOCK_INFO		LFHBlocks;							///< Statistics collected on logical file header blocks.

		// Index check progress

		FLMBOOL			bUniqueIndex;						///< Is index being checked a unique index?
		FLMUINT64		ui64NumKeys;						///< Total keys gathered from records.
		FLMUINT64		ui64NumDuplicateKeys;			///< Total duplicate keys found in records.
		FLMUINT64		ui64NumKeysExamined;				///< Total keys examined so far.
		FLMUINT64		ui64NumKeysNotFound;				///< Total keys found in index but not in record.
		FLMUINT64		ui64NumRecKeysNotFound;			///< Total keys found in record but not in index.
		FLMUINT64		ui64NumNonUniqueKeys;			///< Total non-unique keys found in records for unique indexes.
		FLMUINT64		ui64NumConflicts;					///< Number of key inconsistencies that turned out not to be inconsistent when
																	///< FLAIM attempted to repair them.
		FLMUINT64		ui64NumRSUnits;					///< Total number of "units" in the key result set that need to be sorted.\  NOTE:
																	///< This only applies when iCheckPhase == CHECK_RS_SORT.
		FLMUINT64		ui64NumRSUnitsDone;				///< Number of "units" in the key result set that have been sorted so far.\  NOTE:
																	///< This only applies when iCheckPhase == CHECK_RS_SORT.

		// Information about the database.

		FLMUINT			uiVersionNum;						///< Database version.
		FLMUINT			uiBlockSize;						///< Database block size.
		FLMUINT			uiDefaultLanguage;				///< Database default language.
	} DB_CHECK_PROGRESS;

	/// Structure returned during status callback from FlmDbRebuild().  This structure is passed to the callback function when the
	/// eStatusType::FLM_REBUILD_STATUS status is reported.
	typedef struct
	{
		FLMINT			iDoingFlag;							///< This indicates what the rebuild operation is currently doing.\  It may be one
																	///< of the following:\n
																	///< - REBUILD_GET_BLK_SIZ - FlmDbRebuild() is trying to determine the database's block size
																	///< - REBUILD_RECOVER_DICT - FlmDbRebuild() is recovering dictionary records
																	///< - REBUILD_RECOVER_DATA - FlmDbRebuild() is recovering non-dictionary records
																	///< - REBUILD_FINISHED - FlmDbRebuild() is done rebuilding the database
	#define					REBUILD_GET_BLK_SIZ		1
	#define					REBUILD_RECOVER_DICT		2
	#define					REBUILD_RECOVER_DATA		3
	#define					REBUILD_FINISHED			4
		FLMBOOL			bStartFlag;							///< This flag is TRUE when FlmDbRebuild() is just starting its current operation
																	///< (the one specified in iDoingFlag), FALSE otherwise.
		FLMUINT64		ui64DatabaseSize;					///< Total size of the database data files (in bytes).
		FLMUINT64		ui64BytesExamined;				///< Total bytes examined in the data files so far.
		FLMUINT			uiTotRecs;							///< Total records traversed.
		FLMUINT			uiRecsRecov;						///< Total records recovered so far.
	} REBUILD_INFO;

	/// Return an error string for a corruption code.
	FLMXPC const char * FLMAPI FlmVerifyErrToStr(
		eCorruptionType	eCorruption		///< Corruption code.
		);

	/// Check a database for corruptions.
	/// \ingroup dbmaint
	FLMXPC RCODE FLMAPI FlmDbCheck(
		HFDB						hDb,					///< Database handle of database to be checked.\  If HFDB_NULL, FlmDbCheck will call
															///< FlmDbOpen() using the pszDbFileName, pszDataDir, pszRflDir, and uiFlags parameters.
		const char *			pszDbFileName,		///< Name of database to be checked.\  This is only used if hDb is HFDB_NULL.
		const char *			pszDataDir,			///< Directory where database's data files are located.\  This is only used if hDb is
															///< HFDB_NULL.\   If this parameter is NULL, data files are assumed to be in the same
															///< directory as pszDbFileName.
		const char *			pszRflDir,			///< Directory where database's RFL files are located.\  This is only used if hDb is
															///< HFDB_NULL.\   If this parameter is NULL, RFL files are assumed to be in the same
															///< directory as pszDbFileName.
		FLMUINT					uiCheckFlags,		///< Checking options.\  May include one or more of the following flags ORed
															///< together:\n
															///< - FLM_CHK_INDEX_REFERENCING - Check logical integrity of indexes to make sure all keys
															///< in the index are in the referenced records and that all keys generated from records
															///< are in the index
															///< - FLM_CHK_FIELDS - Check fields in records
		F_Pool *					pPool,				///< Memory pool for allocating memory to hold various statistics in the pDbStats parameter.
		DB_CHECK_PROGRESS *	pDbStats,			///< Statistics collected about the database during the check.
		STATUS_HOOK				fnStatusHook,		///< Callback status function.
		void *					pvAppArg				///< Pointer to application data.\  This pointer is passed into fnStatusHook whenever it
															///< is called.
		);

	// Flags for FlmDbCheck

	#define FLM_CHK_INDEX_REFERENCING				0x01
	#define FLM_CHK_FIELDS								0x02

	/// Rebuild a database.  This function creates a new database from an existing database by extracting
	/// records from the existing database and inserting them into the new database.  Any corrupted parts
	/// of the existing database are ignored.  In this way, if a database is corrupt, a new
	/// database may be built from it that has no corruptions.  In addition, indexes in the new database
	/// will be rebuilt.
	/// \ingroup dbmaint
	FLMXPC RCODE FLMAPI FlmDbRebuild(
		const char *		pszSourceDbPath,	///< Name of database to be rebuilt.
		const char *		pszSourceDataDir,	///< Directory where database's data files are located.\  If NULL, it is assumed that the
														///< database's data files are located in the same directory as pszSourceDbPath.
		const char *		pszDestDbPath,		///< Name of new database that is to be created.
		const char *		pszDestDataDir,	///< Directory where new database's data files are to be created.\  If NULL, data files will be
														///< created in the same directory as pszDestDbPath.
		const char *		pszDestRflDir,		///< Directory where new database's RFL files are to be created.\  If NULL, RFL files will be
														///< created in the same directory as pszDestDbPath.
		const char *		pszDictPath,		///< If non-NULL, this is the name of a file that has dictionary definitions which are to be
														///< loaded into the new database's dictionary container.
		CREATE_OPTS *		pCreateOpts,		///< Create options for the new database.
		FLMUINT *			puiTotalRecords,	///< Returns the total number of records that were found in the source database.
		FLMUINT *			puiRecsRecovered,	///< Returns the total number of records that were recovered from the source database.
		STATUS_HOOK			fnStatusHook,		///< Callback function.\  FlmDbRebuild() calls this function to report rebuild progress.
		void *				pvAppData			///< Pointer to application data which will be passed into the callback function whenever it is called.
		);

	/// Reduce the database size - returning unused blocks back to the file system.
	/// \ingroup dbmaint
	FLMXPC RCODE FLMAPI FlmDbReduceSize(
		HFDB				hDb,				///< Database handle.
		FLMUINT			uiCount,			///< Maximum number of unused blocks to be returned to file system.
		FLMUINT *		puiCount			///< Number of blocks actually returned.\  This should be the same as the number
												///< of blocks requested, unless there are not that many unused blocks.
		);

	/// Traverse records in the database looking for unused fields.
	/// \ingroup dbmaint
	FLMXPC RCODE FLMAPI FlmDbSweep(
		HFDB				hDb,					///< Database handle.
		FLMUINT			uiSweepMode,		///< Flags indicating what actions FlmDbSweep() should do while it is traversing the database.\  It
													///< may be any of the following flags ORed together:\n
													///< - SWEEP_CHECKING_FLDS - Look for field definitions marked as "checking".\  These fields should
													///< be checked to see if they are still in use
													///< - SWEEP_PURGED_FLDS - Look for field definitions marke as "purged".\  These fields should be
													///< removed from records
													///< - SWEEP_STATS - Only calls the callback function.\  This provides a way for an application
													///< to collect database statistics
		FLMUINT			uiCallbackFlags,	///< Flags indicating what what events FlmDbSweep() should report through the callback function.\  It
													///< may be any of the following flags ORed together:\n
													///< - EACH_CONTAINER - Callback function is called whenever FlmDbSweep() begins traversing a new container
													///< - EACH_RECORD - Callback function is called for each record traversed
													///< - EACH_FIELD - Callback function is called for each field traversed
													///< - EACH_CHANGE - Callback function is called whenever a field that was marked as "checking" is found
													///< in the database and the state is changed back to "unused".\  It is also calld whenever a field that
													///< was marked as "purged" is removed from a record
		STATUS_HOOK		fnStatusHook,		///< Callback function.\  See eStatusType::FLM_SWEEP_STATUS for documentation on data that is passed
													///< to the callback function when it is called.
		void *			pvAppData			///< Pointer to application data that will be passed to the callback function whenever it is called.
		);

		// Options for wSweepMode

		#define SWEEP_CHECKING_FLDS	0x01	// look for 'checking' field/record states.
		#define SWEEP_PURGED_FLDS		0x02	// remove 'purged' items.
		#define SWEEP_STATS				0x04	// only calls the STATUS_HOOK

		// Options for wCallbackFreq

		#define EACH_CONTAINER			0x02	// calls fnStatusHook on each container
		#define EACH_RECORD				0x04	// calls fnStatusHook on each record
		#define EACH_FIELD				0x08	// calls fnStatusHook on each field
		#define EACH_CHANGE				0x10	// calls when a 'checking' or 'purged'
														// field is being changed 'check' -> 'unused'
														// and deleting 'purged' fields

	/// Upgrade a database.
	/// \ingroup dbmaint
	FLMXPC RCODE FLMAPI FlmDbUpgrade(
		HFDB			hDb,						///< Database handle.
		FLMUINT		uiNewVersion,			///< Version database is to be upgraded to.\  This must be greater than the current version of the database.
		STATUS_HOOK	fnStatusCallback,		///< Callback function that is called while the database is being upgraded.\  See
													///< documentaiton eStatusType::FLM_DB_UPGRADE_STATUS for documentation on data that is passed
													///< to the callback function when it is called.
		void *		pvAppData				///< Pointer to application data that will be passed to the callback function whenever it is called.
		);

	/// Types of actions the background maintenance thread may currently be doing.
	typedef enum
	{
		FLM_MAINT_UNKNOWN = 0,
		FLM_MAINT_IDLE,					///< Thread is idle.
		FLM_MAINT_LOOKING_FOR_WORK,	///< Thread is looking for work to do.
		FLM_MAINT_WAITING_FOR_LOCK,	///< Thread is waiting to get the database lock.
		FLM_MAINT_ENDING_TRANS,			///< Thread is committing an update transaction.
		FLM_MAINT_TERMINATED,			///< Thread is not currently running.
		FLM_MAINT_FREEING_BLOCKS		///< Thread is freeing blocks.
	} eMaintAction;

	/// This structure is returned from FlmMaintenanceStatus().\  It contains information about
	/// the background maintenance thread.
	typedef struct
	{
		eMaintAction	eDoing;				///< Current action of the maintenance thread.
		FLMUINT64		ui64BlocksFreed;	///< Total blocks freed.\  NOTE: Only valid if eDoing == eMaintAction::FLM_MAINT_FREEING_BLOCKS.
	} FMAINT_STATUS;

	/// Get the current status of the background maintenance thread for a database.
	/// \ingroup dbmaint
	FLMXPC RCODE FLMAPI FlmMaintenanceStatus(
		HFDB					hDb,				///< Database handle.
		FMAINT_STATUS *	pMaintStatus	///< Status is returned in this structure.
		);

	/// Copy a database.
	/// \ingroup dbcopy
	FLMXPC RCODE FLMAPI FlmDbCopy(
		const char *		pszSrcDbName,				///< Name of database to be copied.\  May be full path name or partial path name.
		const char *		pszSrcDataDir,				///< Name of directory where data files for the database are located.\  If NULL, data files are
																///< assumed to be in the same directory as the main database file - pszSrcDbName.
		const char *		pszSrcRflDir,				///< Name of the directory where RFL files are located.\  If NULL, RFL files are
																///< assumed to be in the same directory as the main database file - pszSrcDbName.
		const char *		pszDestDbName,				///< Name of destination database.\  May be full path name or partial path name.
		const char *		pszDestDataDir,			///< Name of desitnation directory where data files for the new database are to be 
																///< copied.\  If NULL, destination data files will be copied to the same
																///< directory as the main destination database file - pszDestDbName.
		const char *		pszDestRflDir,				///< Name of desitnation directory where RFL files for the new database are to be 
																///< copied.\  If NULL, destination RFL files will be copied to the same
																///< directory as the main destination database file - pszDestDbName.
		STATUS_HOOK			fnStatusCallback,			///< Callback function called by FlmDbCopy() to show copy progress.\  See
																///< documentation on eStatusType::FLM_DB_COPY_STATUS for information on the data
																///< that FlmDbCopy() will pass to the callback function.
		void *				pvAppData					///< Pointer to application data that will be passed to the callback function whenever it is called.
		);

	/// Rename a database.
	/// \ingroup dbcopy
	FLMXPC RCODE FLMAPI FlmDbRename(
		const char *		pszDbName,					///< Name of database to be renamed.\  May be full path name or partial path name.
		const char *		pszDataDir,					///< Name of directory where data files for the database are located.\  If NULL, data files are
																///< assumed to be in the same directory as the main database file - pszDbName.
		const char *		pszRflDir,					///< Name of the directory where RFL files are located.\  If NULL, RFL files are
																///< assumed to be in the same directory as the main database file - pszDbName.
		const char *		pszNewDbName,				///< New name to be given to the database.\  NOTE: All data files and RFL subdirectories will
																///< be renamed using this name as the template.
		FLMBOOL				bOverwriteDestOk,			///< Flag indicating whether or not it is ok to overwrite the destination database
																///< if a database already exists with the new name.
		STATUS_HOOK			fnStatusCallback,			///< Callback function called by FlmDbRename() to show copy progress.\  See
																///< documentation on eStatusType::FLM_DB_RENAME_STATUS for information on the data
																///< that FlmDbRename() will pass to the callback function.
		void *				pvAppData					///< Pointer to application data that will be passed to the callback function whenever it is called.
		);

	/// Delete a database.
	/// \ingroup dbcopy
	FLMXPC RCODE FLMAPI FlmDbRemove(
		const char *		pszDbName,					///< Name of database to be deleted.\  May be full path name or partial path name.
		const char *		pszDataDir,					///< Name of directory where data files for the database are located.\  If NULL, data files are
																///< assumed to be in the same directory as the main database file - pszDbName.
		const char *		pszRflDir,					///< Name of the directory where RFL files are located.\  If NULL, RFL files are
																///< assumed to be in the same directory as the main database file - pszDbName.
		FLMBOOL				bRemoveRflFiles			///< Flag indicating whether or not RFL files should be deleted.
		);

	/// Enable encryption for a database.
	/// \ingroup encryption
	FLMXPC RCODE FLMAPI FlmEnableEncryption(
		HFDB				hDb,						///< Database handle.
		FLMBYTE **		ppucWrappingKey,		///< This returns a pointer to a buffer containing the database key wrapped in the
														///< NICI local storage key.\  FlmEnableEncryption() allocates memory for this buffer.\  The
														///< memory must be freed by calling FlmFreeMem().
		FLMUINT32 *		pui32KeyLen				///< Length of data in *ppucWrappingKey.
		);

	/// Wrap a database's encryption key in a password.
	/// \ingroup encryption
	FLMXPC RCODE FLMAPI FlmDbWrapKey(
		HFDB					hDb,						///< Database handle.
		const char *		pszPassword				///< Password to wrap the database key in.\  May be NULL to wrap the key in the NICI
															///< local storage key.\  NOTE: Once the database key has been wrapped in a password,
															///< that password must be supplied to FlmDbOpen() when opening the database.
		);

	#ifdef FLM_PACK_STRUCTS
		#pragma pack(pop)
	#endif

#endif
