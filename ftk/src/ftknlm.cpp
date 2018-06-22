//-------------------------------------------------------------------------
// Desc:	I/O for Netware OS
// Tabs:	3
//
// Copyright (c) 1998-2003, 2005-2007 Novell, Inc. All Rights Reserved.
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
//-------------------------------------------------------------------------

#include "ftksys.h"
#include "ftknlm.h"

#if defined( FLM_NLM)

#define NSS_BLOCK_SIZE				65536
#define NSS_SECTORS_PER_BLOCK		(NSS_BLOCK_SIZE / FLM_NLM_SECTOR_SIZE)

#if defined( FLM_RING_ZERO_NLM)
	#define zMAX_COMPONENT_NAME						256
	#define zGET_INFO_VARIABLE_DATA_SIZE			(zMAX_COMPONENT_NAME * 2)
	
	#define zOK												0			// the operation succeeded
	#define zERR_NO_MEMORY								20000 	// insufficent memory to complete the request
	#define zERR_NOT_SUPPORTED							20011 	// the operation is not supported
	#define zERR_CONNECTION_NOT_LOGGED_IN			20007 	// the connection has not been logged in
	#define zERR_END_OF_FILE							20100 	// read past the end of file
	#define zERR_OUT_OF_SPACE							20103 	// no available disk space is left
	#define zERR_BAD_FILE_HANDLE						20401 	// the file handle is out of range, bad instance, or doesn't exist
	#define zERR_INVALID_NAME							20403		// path name is invalid -- bad syntax
	#define zERR_INVALID_CHAR_IN_NAME				20404 	// path name had an invalid character
	#define zERR_INVALID_PATH							20405 	// the path is syntactically incorrect
	#define zERR_NAME_NOT_FOUND_IN_DIRECTORY		20407 	// name does not exist in the direcory being searched
	#define zERR_NO_NAMES_IN_PATH						20409 	// a NULL file name was given
	#define zERR_NO_MORE_NAMES_IN_PATH 				20410 	// doing a wild search but ran out of names to search
	#define zERR_PATH_MUST_BE_FULLY_QUALIFIED		20411 	// path name must be fully qualified in this context
	#define zERR_FILE_ALREADY_EXISTS					20412 	// the given file already exists
	#define zERR_NAME_NO_LONGER_VALID				20413 	// the dir/file name is no longer valid
	#define zERR_DIRECTORY_NOT_EMPTY					20417 	// the directory still has files in it
	#define zERR_NO_FILES_FOUND						20424 	// no files matched the given wildcard pattern
	#define zERR_DIR_CANNOT_BE_OPENED				20435 	// the requested parent was not found
	#define zERR_NO_OPEN_PRIVILEGE					20438 	// no the right privileges to open the file
	#define zERR_NO_MORE_CONTEXT_HANDLE_IDS		20439 	// there are no more available context handle IDs
	#define zERR_INVALID_PATH_FORMAT					20441 	// the pathFormat is either invalid or unsupported
	#define zERR_ALL_FILES_IN_USE						20500 	// all files were in use
	#define zERR_SOME_FILES_IN_USE					20501 	// some of the files were in use
	#define zERR_ALL_FILES_READ_ONLY					20502 	// all files were READONLY
	#define zERR_SOME_FILES_READ_ONLY				20503 	// some of the files were READONLY
	#define zERR_ALL_NAMES_EXIST						20504 	// all of the names already existed
	#define zERR_SOME_NAMES_EXIST						20505 	// some of the names already existed
	#define zERR_NO_RENAME_PRIVILEGE					20506 	// you do not have privilege to rename the file
	#define zERR_RENAME_DIR_INVALID					20507 	// the selected directory may not be renamed
	#define zERR_RENAME_TO_OTHER_VOLUME				20508 	// a rename/move may not move the beast to a different volume
	#define zERR_CANT_RENAME_DATA_STREAMS			20509 	// not allowed to rename a data stream
	#define zERR_FILE_RENAME_IN_PROGRESS			20510 	// the file is already being renamed by a different process
	#define zERR_CANT_RENAME_TO_DELETED				20511 	// only deleted files may be renamed to a deleted state
	#define zERR_HOLE_IN_DIO_FILE  	            20651 	// DIO files cannot have holes
	#define zERR_BEYOND_EOF  	            		20652 	// DIO files cannot be read beyond EOF
	#define zERR_INVALID_PATH_SEPARATOR				20704 	// the name space does not support the requested path separator type
	#define zERR_VOLUME_SEPARATOR_NOT_SUPPORTED	20705 	// the name space does not support volume separators
	#define zERR_BAD_VOLUME_NAME   					20800 	// the given volume name is syntactically incorrect
	#define zERR_VOLUME_NOT_FOUND  					20801 	// the given volume name could not be found
	#define zERR_NO_SET_PRIVILEGE  					20850 	// does not have rights to modify metadata
	#define zERR_NO_CREATE_PRIVILEGE					20851		// does not have rights to create an object
	#define zERR_ACCESS_DENIED							20859 	// authorization/attributes denied access
	#define zERR_NO_WRITE_PRIVILEGE					20860 	// no granted write privileges
	#define zERR_NO_READ_PRIVILEGE					20861 	// no granted read privileges
	#define zERR_NO_DELETE_PRIVILEGE					20862 	// no delete privileges
	#define zERR_SOME_NO_DELETE_PRIVILEGE			20863 	// on wildcard some do not have delete privileges
	#define zERR_NO_SUCH_OBJECT						20867 	// no such object in the naming services
	#define zERR_CANT_DELETE_OPEN_FILE				20868 	// cant delete an open file without rights
	#define zERR_NO_CREATE_DELETE_PRIVILEGE		20869 	// no delete on create privileges
	#define zERR_NO_SALVAGE_PRIVILEGE				20870 	// no privileges to salvage this file
	#define zERR_FILE_READ_LOCKED						20905 	// cant grant read access to the file
	#define zERR_FILE_WRITE_LOCKED					20906 	// cant grant write access to the file
	
	#define zRR_READ_ACCESS								0x00000001
	#define zRR_WRITE_ACCESS							0x00000002
	#define zRR_DENY_READ								0x00000004
	#define zRR_DENY_WRITE								0x00000008
	#define zRR_SCAN_ACCESS								0x00000010
	#define zRR_ENABLE_IO_ON_COMPRESSED_DATA		0x00000100
	#define zRR_LEAVE_FILE_COMPRESSED	        	0x00000200
	#define zRR_DELETE_FILE_ON_CLOSE					0x00000400
	#define zRR_FLUSH_ON_CLOSE							0x00000800
	#define zRR_PURGE_IMMEDIATE_ON_CLOSE			0x00001000
	#define zRR_DIO_MODE									0x00002000
	#define zRR_ALLOW_SECURE_DIRECTORY_ACCESS		0x00020000
	#define zRR_TRANSACTION_ACTIVE					0x00100000
	#define zRR_READ_ACCESS_TO_SNAPSHOT				0x04000000
	#define zRR_DENY_RW_OPENER_CAN_REOPEN			0x08000000
	#define zRR_CREATE_WITHOUT_READ_ACCESS			0x10000000
	#define zRR_OPENER_CAN_DELETE_WHILE_OPEN		0x20000000
	#define zRR_CANT_DELETE_WHILE_OPEN				0x40000000
	#define zRR_DONT_UPDATE_ACCESS_TIME				0x80000000
	
	#define zFA_READ_ONLY		 						0x00000001
	#define zFA_HIDDEN 									0x00000002
	#define zFA_SYSTEM 									0x00000004
	#define zFA_EXECUTE									0x00000008
	#define zFA_SUBDIRECTORY	 						0x00000010
	#define zFA_ARCHIVE									0x00000020
	#define zFA_SHAREABLE		 						0x00000080
	#define zFA_SMODE_BITS		 						0x00000700
	#define zFA_NO_SUBALLOC								0x00000800
	#define zFA_TRANSACTION								0x00001000
	#define zFA_NOT_VIRTUAL_FILE						0x00002000
	#define zFA_IMMEDIATE_PURGE						0x00010000
	#define zFA_RENAME_INHIBIT	 						0x00020000
	#define zFA_DELETE_INHIBIT	 						0x00040000
	#define zFA_COPY_INHIBIT	 						0x00080000
	#define zFA_IS_ADMIN_LINK							0x00100000
	#define zFA_IS_LINK									0x00200000
	#define zFA_REMOTE_DATA_INHIBIT					0x00800000
	#define zFA_COMPRESS_FILE_IMMEDIATELY 			0x02000000
	#define zFA_DATA_STREAM_IS_COMPRESSED 			0x04000000
	#define zFA_DO_NOT_COMPRESS_FILE	  				0x08000000
	#define zFA_CANT_COMPRESS_DATA_STREAM 			0x20000000
	#define zFA_ATTR_ARCHIVE	 						0x40000000
	#define zFA_VOLATILE									0x80000000
	
	#define zNSPACE_DOS									0
	#define zNSPACE_MAC									1
	#define zNSPACE_UNIX									2
	#define zNSPACE_LONG									4
	#define zNSPACE_DATA_STREAM						6
	#define zNSPACE_EXTENDED_ATTRIBUTE				7
	#define zNSPACE_INVALID								(-1)
	#define zNSPACE_DOS_MASK							(1 << zNSPACE_DOS)
	#define zNSPACE_MAC_MASK							(1 << zNSPACE_MAC)
	#define zNSPACE_UNIX_MASK							(1 << zNSPACE_UNIX)
	#define zNSPACE_LONG_MASK							(1 << zNSPACE_LONG)
	#define zNSPACE_DATA_STREAM_MASK					(1 << zNSPACE_DATA_STREAM)
	#define zNSPACE_EXTENDED_ATTRIBUTE_MASK 		(1 << zNSPACE_EXTENDED_ATTRIBUTE)
	#define zNSPACE_ALL_MASK							(0xffffffff)
	
	#define zMODE_VOLUME_ID								0x80000000
	#define zMODE_UTF8									0x40000000
	#define zMODE_DELETED								0x20000000
	#define zMODE_LINK									0x10000000
	
	#define zCREATE_OPEN_IF_THERE						0x00000001
	#define zCREATE_TRUNCATE_IF_THERE				0x00000002
	#define zCREATE_DELETE_IF_THERE					0x00000004
	
	#define zMATCH_ALL_DERIVED_TYPES					0x00000001
	#define zMATCH_HIDDEN								0x1
	#define zMATCH_NON_HIDDEN							0x2
	#define zMATCH_DIRECTORY							0x4
	#define zMATCH_NON_DIRECTORY						0x8
	#define zMATCH_SYSTEM								0x10
	#define zMATCH_NON_SYSTEM							0x20
	#define zMATCH_ALL									(~0)
	
	#define zSETSIZE_NON_SPARSE_FILE					0x00000001
	#define zSETSIZE_NO_ZERO_FILL						0x00000002
	#define zSETSIZE_UNDO_ON_ERR	 					0x00000004
	#define zSETSIZE_PHYSICAL_ONLY	 				0x00000008
	#define zSETSIZE_LOGICAL_ONLY	 					0x00000010
	#define zSETSIZE_COMPRESSED      				0x00000020
	
	#define zMOD_FILE_ATTRIBUTES						0x00000001
	#define zMOD_CREATED_TIME							0x00000002
	#define zMOD_ARCHIVED_TIME							0x00000004
	#define zMOD_MODIFIED_TIME							0x00000008
	#define zMOD_ACCESSED_TIME							0x00000010
	#define zMOD_METADATA_MODIFIED_TIME				0x00000020
	#define zMOD_OWNER_ID								0x00000040
	#define zMOD_ARCHIVER_ID							0x00000080
	#define zMOD_MODIFIER_ID							0x00000100
	#define zMOD_METADATA_MODIFIER_ID				0x00000200
	#define zMOD_PRIMARY_NAMESPACE					0x00000400
	#define zMOD_DELETED_INFO							0x00000800
	#define zMOD_MAC_METADATA							0x00001000
	#define zMOD_UNIX_METADATA							0x00002000
	#define zMOD_EXTATTR_FLAGS							0x00004000
	#define zMOD_VOL_ATTRIBUTES						0x00008000
	#define zMOD_VOL_NDS_OBJECT_ID					0x00010000
	#define zMOD_VOL_MIN_KEEP_SECONDS				0x00020000
	#define zMOD_VOL_MAX_KEEP_SECONDS				0x00040000
	#define zMOD_VOL_LOW_WATER_MARK					0x00080000
	#define zMOD_VOL_HIGH_WATER_MARK					0x00100000
	#define zMOD_POOL_ATTRIBUTES						0x00200000
	#define zMOD_POOL_NDS_OBJECT_ID					0x00400000
	#define zMOD_VOL_DATA_SHREDDING_COUNT			0x00800000
	#define zMOD_VOL_QUOTA								0x01000000
	
	/***************************************************************************
	Desc:
	***************************************************************************/
	enum zGetInfoMask_t
	{
		zGET_STD_INFO										= 0x1,
		zGET_NAME											= 0x2,
		zGET_ALL_NAMES										= 0x4,
		zGET_PRIMARY_NAMESPACE							= 0x8,
		zGET_TIMES_IN_SECS								= 0x10,
		zGET_TIMES_IN_MICROS								= 0x20,
		zGET_IDS												= 0x40,
		zGET_STORAGE_USED									= 0x80,
		zGET_BLOCK_SIZE									= 0x100,
		zGET_COUNTS											= 0x200,
		zGET_EXTENDED_ATTRIBUTE_INFO					= 0x400,
		zGET_DATA_STREAM_INFO							= 0x800,
		zGET_DELETED_INFO									= 0x1000,
		zGET_MAC_METADATA									= 0x2000,
		zGET_UNIX_METADATA								= 0x4000,
		zGET_EXTATTR_FLAGS								= 0x8000,
		zGET_VOLUME_INFO									= 0x10000,
		zGET_VOL_SALVAGE_INFO							= 0x20000,
		zGET_POOL_INFO										= 0x40000
	};
	
	/***************************************************************************
	Desc:
	***************************************************************************/
	enum
	{
		zINFO_VERSION_A = 1
	};
	
	/***************************************************************************
	Desc:
	***************************************************************************/
	typedef enum FileType_t
	{
		zFILE_UNKNOWN,
		zFILE_REGULAR,
		zFILE_EXTENDED_ATTRIBUTE,
		zFILE_NAMED_DATA_STREAM,
		zFILE_PIPE,
		zFILE_VOLUME,
		zFILE_POOL,
		zFILE_MAX
	} FileType_t;
	
	/***************************************************************************
	Desc:
	***************************************************************************/
	typedef struct	GUID_t
	{
		LONG					timeLow;
		WORD					timeMid;
		WORD					timeHighAndVersion;
		BYTE					clockSeqHighAndReserved;
		BYTE					clockSeqLow;
		BYTE					node[ 6];
	} GUID_t;
	
	/***************************************************************************
	Desc:
	***************************************************************************/
	typedef struct zMacInfo_s
	{
		BYTE 					finderInfo[32];
		BYTE 					proDOSInfo[6];
		BYTE					filler[2];
		LONG					dirRightsMask;
	} zMacInfo_s;
	
	/***************************************************************************
	Desc:
	***************************************************************************/
	typedef struct zUnixInfo_s
	{
		LONG					fMode;
		LONG					rDev;
		LONG					myFlags;
		LONG					nfsUID;
		LONG 					nfsGID;
		LONG					nwUID;
		LONG					nwGID;
		LONG					nwEveryone;
		LONG					nwUIDRights;
		LONG					nwGIDRights;
		LONG					nwEveryoneRights;
		BYTE					acsFlags;
		BYTE					firstCreated;
		FLMINT16				variableSize;
	} zUnixInfo_s;
	
	typedef struct zVolumeInfo_s
	{
		GUID_t				volumeID;
		GUID_t				ndsObjectID;
		LONG					volumeState;
		LONG					nameSpaceMask;
		
		struct
		{
			FLMUINT64		enabled;
			FLMUINT64		enableModMask;
			FLMUINT64		supported;
		} features;
		
		FLMUINT64			maximumFileSize;	
		FLMUINT64			totalSpaceQuota;
		FLMUINT64			numUsedBytes;
		FLMUINT64			numObjects;
		FLMUINT64			numFiles;
		LONG					authModelID;
		LONG					dataShreddingCount;
		
		struct
		{
			FLMUINT64		purgeableBytes;
			FLMUINT64		nonPurgeableBytes;
			FLMUINT64		numDeletedFiles;
			FLMUINT64		oldestDeletedTime;
			LONG				minKeepSeconds;
			LONG				maxKeepSeconds;
			LONG				lowWaterMark;
			LONG				highWaterMark;
		} salvage;
		
		struct
		{
			FLMUINT64		numCompressedFiles;
			FLMUINT64		numCompDelFiles;
			FLMUINT64		numUncompressibleFiles;
			FLMUINT64		numPreCompressedBytes;
			FLMUINT64		numCompressedBytes;
		} comp;
		 
	} zVolumeInfo_s;
	
	/***************************************************************************
	Desc:
	***************************************************************************/
	typedef struct zPoolInfo_s
	{
		GUID_t				poolID;
		GUID_t				ndsObjectID;
		LONG					poolState;
		LONG					nameSpaceMask;
		
		struct 
		{
			FLMUINT64		enabled;
			FLMUINT64		enableModMask;
			FLMUINT64		supported;
		} features;
		
		FLMUINT64			totalSpace;
		FLMUINT64			numUsedBytes;
		FLMUINT64			purgeableBytes;
		FLMUINT64			nonPurgeableBytes;
	} zPoolInfo_s;
	
	/***************************************************************************
	Desc:
	***************************************************************************/
	typedef struct zInfo_s
	{
		LONG					infoVersion;
		FLMINT				totalBytes;
		FLMINT				nextByte;
		LONG					padding;
		FLMUINT64			retMask;
		
		struct 
		{
			FLMUINT64		zid;
			FLMUINT64		dataStreamZid;
			FLMUINT64		parentZid;
			FLMUINT64		logicalEOF;
			GUID_t			volumeID;
			LONG				fileType;
			LONG				fileAttributes;
			LONG				fileAttributesModMask;
			LONG				padding;
		} std;
	
		struct
		{
			FLMUINT64		physicalEOF;
			FLMUINT64		dataBytes;
			FLMUINT64		metaDataBytes;
		} storageUsed;
	
		LONG					primaryNameSpaceID;
		LONG 					nameStart;
	
		struct 
		{
			LONG 				numEntries;
			LONG 				fileNameArray;
		} names;
	
		struct
		{
			FLMUINT64		created;
			FLMUINT64		archived;
			FLMUINT64		modified;
			FLMUINT64		accessed;
			FLMUINT64		metaDataModified;
		} time;
	
		struct 
		{
			GUID_t 			owner;
			GUID_t 			archiver;
			GUID_t 			modifier;
			GUID_t 			metaDataModifier;
		} id;
	
		struct 
		{
			LONG	 			size;
			LONG	 			sizeShift;
		} blockSize;
	
		struct 
		{
			LONG	 			open;
			LONG	 			hardLink;
		} count;
	
		struct 
		{
			LONG	 			count;
			LONG	 			totalNameSize;
			FLMUINT64		totalDataSize;
		} dataStream;
	
		struct 
		{
			LONG	 			count;
			LONG	 			totalNameSize;
			FLMUINT64		totalDataSize;
		} extAttr;
	
		struct 
		{
			FLMUINT64		time;
			GUID_t 			id;
		} deleted;
	
		struct 
		{
			zMacInfo_s 		info;
		} macNS;
	
		struct 
		{
			zUnixInfo_s 	info;
			LONG				offsetToData;
		} unixNS;
	
		zVolumeInfo_s		vol;
		zPoolInfo_s			pool;
		LONG					extAttrUserFlags;
		BYTE					variableData[zGET_INFO_VARIABLE_DATA_SIZE];
	
	} zInfo_s;
	
	RCODE DfsMapError(
		LONG					lResult,
		RCODE					defaultRc);
	
	FLMUINT f_getNSSOpenFlags(
		FLMUINT				uiIoFlags,
		FLMBOOL				bDoDirectIo);
	
	typedef FLMINT (* zROOT_KEY_FUNC)(
		FLMUINT				connectionID,
		FLMINT64 *			retRootKey);
	
	typedef FLMINT (* zCLOSE_FUNC)(
		FLMINT64				key);
	
	typedef FLMINT (* zCREATE_FUNC)(
		FLMINT64				key,
		FLMUINT				taskID,	
		FLMUINT64			xid,
		FLMUINT				nameSpace,
		const void *		path,
		FLMUINT				fileType,
		FLMUINT64			fileAttributes,
		FLMUINT				createFlags,
		FLMUINT				requestedRights,
		FLMINT64 *			retKey);
	
	typedef FLMINT (* zOPEN_FUNC)(
		FLMINT64				key,
		FLMUINT				taskID,
		FLMUINT				nameSpace,
		const void *		path,
		FLMUINT				requestedRights,
		FLMINT64 *			retKey);
	
	typedef FLMINT (* zDELETE_FUNC)(
		FLMINT64				key,
		FLMUINT64			xid,
		FLMUINT				nameSapce,
		const void *		path,
		FLMUINT				match,
		FLMUINT				deleteFlags);
	
	typedef FLMINT (* zREAD_FUNC)(
		FLMINT64				key,
		FLMUINT64			xid,	
		FLMUINT64			startingOffset,
		FLMUINT				bytesToRead,
		void *				retBuffer,
		FLMUINT *			retBytesRead);
	
	typedef FLMINT (* zDIO_READ_FUNC)(
		FLMINT64				key,
		FLMUINT64			unitOffset,
		FLMUINT				unitsToRead,
		FLMUINT				callbackData,
		void					(*dioReadCallBack)(
									FLMUINT	reserved,
									FLMUINT	callbackData,
									FLMUINT 	retStatus),
		void *				retBuffer);
	
	typedef FLMINT (* zGET_INFO_FUNC)(
		FLMINT64				key,
		FLMUINT64			getInfoMask,
		FLMUINT				sizeRetGetInfo,
		FLMUINT				infoVersion,
		zInfo_s *			retGetInfo);
	
	typedef FLMINT (* zMODIFY_INFO_FUNC)(
		FLMINT64				key,
		FLMUINT64			xid,
		FLMUINT64			modifyInfoMask,
		FLMUINT				sizeModifyInfo,
		FLMUINT				infoVersion,
		const zInfo_s *	modifyInfo);
	
	typedef FLMINT (* zSET_EOF_FUNC)(
		FLMINT64				key,
		FLMUINT64			xid,	
		FLMUINT64			startingOffset,
		FLMUINT				flags);
	
	typedef FLMINT (* zWRITE_FUNC)(
		FLMINT64				key,
		FLMUINT64			xid,	
		FLMUINT64			startingOffset,
		FLMUINT				bytesToWrite,
		const void *		buffer,
		FLMUINT *			retBytesWritten);
	
	typedef FLMINT (* zDIO_WRITE_FUNC)(
		FLMINT64				key,
		FLMUINT64			unitOffset,
		FLMUINT				unitsToWrite,
		FLMUINT				callbackData,
		void					(*dioWriteCallBack)(
									FLMUINT	reserved,
									FLMUINT	callbackData,
									FLMUINT	retStatus),
		const void *		buffer);
	
	typedef FLMINT (* zRENAME_FUNC)(
		FLMINT64				key,
		FLMUINT64			xid,
		FLMUINT				srcNameSpace,
		const void *		srcPath,
		FLMUINT				srcMatchAttributes,
		FLMUINT				dstNameSpace,
		const void *		dstPath,
		FLMUINT				renameFlags);
	
	typedef BOOL (* zIS_NSS_VOLUME_FUNC)(
		const char *		path);
	
	FSTATIC zIS_NSS_VOLUME_FUNC		gv_zIsNSSVolumeFunc = NULL;
	FSTATIC zROOT_KEY_FUNC				gv_zRootKeyFunc = NULL;
	FSTATIC zCLOSE_FUNC					gv_zCloseFunc = NULL;
	FSTATIC zCREATE_FUNC					gv_zCreateFunc = NULL;
	FSTATIC zOPEN_FUNC					gv_zOpenFunc = NULL;
	FSTATIC zDELETE_FUNC					gv_zDeleteFunc = NULL;
	FSTATIC zREAD_FUNC					gv_zReadFunc = NULL;
	FSTATIC zDIO_READ_FUNC				gv_zDIOReadFunc = NULL;
	FSTATIC zGET_INFO_FUNC				gv_zGetInfoFunc = NULL;
	FSTATIC zMODIFY_INFO_FUNC			gv_zModifyInfoFunc = NULL;
	FSTATIC zSET_EOF_FUNC				gv_zSetEOFFunc = NULL;
	FSTATIC zWRITE_FUNC					gv_zWriteFunc = NULL;
	FSTATIC zDIO_WRITE_FUNC				gv_zDIOWriteFunc = NULL;
	FSTATIC zRENAME_FUNC					gv_zRenameFunc = NULL;
	
	void *									gv_MyModuleHandle = NULL;
	FLMATOMIC								gv_NetWareStartupCount = 0;
	rtag_t									gv_lAllocRTag = 0;
	
	FSTATIC FLMINT64						gv_NssRootKey;
	FSTATIC FLMBOOL						gv_bNSSKeyInitialized = FALSE;
	FSTATIC SEMAPHORE						gv_lFlmSyncSem = 0;
	FSTATIC FLMBOOL						gv_bUnloadCalled = FALSE;
	FSTATIC FLMBOOL						gv_bMainRunning = FALSE;
	FSTATIC F_EXIT_FUNC					gv_fnExit = NULL;
	extern FLMATOMIC						gv_openFiles;

	#if !defined( __MWERKS__)
		extern unsigned long ReadInternalClock(void);
	#else
		unsigned long ReadInternalClock(void);
	#endif
	
	FSTATIC void ConvertToQualifiedNWPath(
		const char *		pInputPath,
		char *				pQualifiedPath);
	
	FSTATIC RCODE nssTurnOffRenameInhibit(
		const char *		pszFileName);
	
	FSTATIC LONG ConvertPathToLNameFormat(
		const char *		pPath,
		LONG *				plVolumeID,
		FLMBOOL *			pbNssVolume,
		FLMBYTE *			pLNamePath,
		LONG *				plLNamePathCount);
	
	FSTATIC void DirectIONoWaitCallBack(
		LONG					unknownAlwaysZero,
		LONG					callbackData,
		LONG 					completionCode);
	
	FSTATIC void nssDioCallback(
		FLMUINT				reserved,
		FLMUINT				UserData,
		FLMUINT				retStatus);
	
	FSTATIC RCODE MapNSSError(
		FLMINT				lStatus,
		RCODE					defaultRc);
	
	RCODE f_nssInitialize( void);
	
	void f_nssUninitialize( void);
	
	extern "C" int nlm_main(
		int					iArgC,
		char **				ppszArgV);

#endif
	
/***************************************************************************
Desc:
***************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
#if !defined( __MWERKS__)
	#pragma aux ReadInternalClock = \
	0x0F 0x31 \
	modify exact [EAX EDX];
#else
	unsigned long ReadInternalClock(void)
	{
		__asm
		{
			rdtsc
			ret
		}
	}
#endif
#endif

/***************************************************************************
Desc:	Initialize the root NSS key.
***************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE f_nssInitialize( void)
{
	RCODE		rc = NE_FLM_OK;
	FLMINT	lStatus;

	if (!gv_bNSSKeyInitialized)
	{
		// Import the required NSS functions

		if( (gv_zIsNSSVolumeFunc = (zIS_NSS_VOLUME_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x0C" "zIsNSSVolume")) == NULL)
		{
			// NSS is not available on this server.  Jump to exit.
			goto Exit;
		}
		
		if( (gv_zRootKeyFunc = (zROOT_KEY_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x08" "zRootKey")) == NULL)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}

		if( (gv_zCloseFunc = (zCLOSE_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x06" "zClose")) == NULL)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}

		if( (gv_zCreateFunc = (zCREATE_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x07" "zCreate")) == NULL)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}

		if( (gv_zOpenFunc = (zOPEN_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x05" "zOpen")) == NULL)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}
		
		if( (gv_zDeleteFunc = (zDELETE_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x07" "zDelete")) == NULL)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}
		
		if( (gv_zReadFunc = (zREAD_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x05" "zRead")) == NULL)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}
		
		if( (gv_zDIOReadFunc = (zDIO_READ_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x08" "zDIORead")) == NULL)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}
		
		if( (gv_zGetInfoFunc = (zGET_INFO_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x08" "zGetInfo")) == NULL)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}
		
		if( (gv_zModifyInfoFunc = (zMODIFY_INFO_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x0B" "zModifyInfo")) == NULL)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}
		
		if( (gv_zSetEOFFunc = (zSET_EOF_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x07" "zSetEOF")) == NULL)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}
		
		if( (gv_zWriteFunc = (zWRITE_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x06" "zWrite")) == NULL)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}
		
		if( (gv_zDIOWriteFunc = (zDIO_WRITE_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x09" "zDIOWrite")) == NULL)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}

		if( (gv_zRenameFunc = (zRENAME_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x07" "zRename")) == NULL)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}

		// Get the NSS root key

		if ((lStatus = gv_zRootKeyFunc( 0, &gv_NssRootKey)) != zOK)
		{
			rc = MapNSSError( lStatus, NE_FLM_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}
		
		gv_bNSSKeyInitialized = TRUE;
	}

Exit:

	return( rc);
}
#endif

/***************************************************************************
Desc:	Close the root NSS key.
***************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
void f_nssUninitialize( void)
{
	if (gv_bNSSKeyInitialized)
	{
		(void)gv_zCloseFunc( gv_NssRootKey);
		gv_bNSSKeyInitialized = FALSE;
	}
}
#endif

/***************************************************************************
Desc:	Maps NSS errors to IO errors.
***************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
FSTATIC RCODE MapNSSError(
	FLMINT	lStatus,
	RCODE		defaultRc)
{
	RCODE		rc;

	switch (lStatus)
	{
		case zERR_FILE_ALREADY_EXISTS:
		case zERR_DIRECTORY_NOT_EMPTY:
		case zERR_DIR_CANNOT_BE_OPENED:
		case zERR_NO_SET_PRIVILEGE:
		case zERR_NO_CREATE_PRIVILEGE:
		case zERR_ACCESS_DENIED:
		case zERR_NO_WRITE_PRIVILEGE:
		case zERR_NO_READ_PRIVILEGE:
		case zERR_NO_DELETE_PRIVILEGE:
		case zERR_SOME_NO_DELETE_PRIVILEGE:
		case zERR_CANT_DELETE_OPEN_FILE:
		case zERR_NO_CREATE_DELETE_PRIVILEGE:
		case zERR_NO_SALVAGE_PRIVILEGE:
		case zERR_FILE_READ_LOCKED:
		case zERR_FILE_WRITE_LOCKED:
			rc = RC_SET( NE_FLM_IO_ACCESS_DENIED);
			break;

		case zERR_BAD_FILE_HANDLE:
			rc = RC_SET( NE_FLM_IO_BAD_FILE_HANDLE);
			break;

		case zERR_OUT_OF_SPACE:
			rc = RC_SET_AND_ASSERT( NE_FLM_IO_DISK_FULL);
			break;

		case zERR_NO_OPEN_PRIVILEGE:
			rc = RC_SET( NE_FLM_IO_OPEN_ERR);
			break;

		case zERR_NAME_NOT_FOUND_IN_DIRECTORY:
		case zERR_NO_FILES_FOUND:
		case zERR_VOLUME_NOT_FOUND:
		case zERR_NO_SUCH_OBJECT:
		case zERR_INVALID_NAME:
		case zERR_INVALID_CHAR_IN_NAME:
		case zERR_INVALID_PATH:
		case zERR_NO_NAMES_IN_PATH:
		case zERR_NO_MORE_NAMES_IN_PATH:
		case zERR_PATH_MUST_BE_FULLY_QUALIFIED:
		case zERR_NAME_NO_LONGER_VALID:
		case zERR_INVALID_PATH_FORMAT:
		case zERR_INVALID_PATH_SEPARATOR:
		case zERR_VOLUME_SEPARATOR_NOT_SUPPORTED:
		case zERR_BAD_VOLUME_NAME:
			rc = RC_SET( NE_FLM_IO_PATH_NOT_FOUND);
			break;

		case zERR_NO_MORE_CONTEXT_HANDLE_IDS:
			rc = RC_SET( NE_FLM_IO_TOO_MANY_OPEN_FILES);
			break;
		case zERR_ALL_FILES_IN_USE:
		case zERR_SOME_FILES_IN_USE:
		case zERR_ALL_FILES_READ_ONLY:
		case zERR_SOME_FILES_READ_ONLY:
		case zERR_ALL_NAMES_EXIST:
		case zERR_SOME_NAMES_EXIST:
		case zERR_NO_RENAME_PRIVILEGE:
		case zERR_RENAME_DIR_INVALID:
		case zERR_RENAME_TO_OTHER_VOLUME:
		case zERR_CANT_RENAME_DATA_STREAMS:
		case zERR_FILE_RENAME_IN_PROGRESS:
		case zERR_CANT_RENAME_TO_DELETED:
			rc = RC_SET( NE_FLM_IO_RENAME_FAILURE);
			break;

		case zERR_CONNECTION_NOT_LOGGED_IN:
			rc = RC_SET( NE_FLM_IO_CONNECT_ERROR);
			break;
		case zERR_NO_MEMORY:
			rc = RC_SET( NE_FLM_MEM);
			break;
		case zERR_NOT_SUPPORTED:
			rc = RC_SET_AND_ASSERT( NE_FLM_NOT_IMPLEMENTED);
			break;
		case zERR_END_OF_FILE:
		case zERR_BEYOND_EOF:
			rc = RC_SET( NE_FLM_IO_END_OF_FILE);
			break;

		default:
			rc = RC_SET( defaultRc);
			break;
	}
	
	return( rc );
}
#endif

/***************************************************************************
Desc:	Maps direct IO errors to IO errors.
fix: we shouldn't have 2 copies of this function.  this is just temporary.
      long term, we need to make the FDFS.CPP version public.
***************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE DfsMapError(
	LONG		lResult,
	RCODE		defaultRc)
{
	switch (lResult)
	{
		case DFSHoleInFileError:
		case DFSOperationBeyondEndOfFile:
			return( RC_SET( NE_FLM_IO_END_OF_FILE));
		case DFSHardIOError:
		case DFSInvalidFileHandle:
			return( RC_SET( NE_FLM_IO_BAD_FILE_HANDLE));
		case DFSNoReadPrivilege:
			return( RC_SET( NE_FLM_IO_ACCESS_DENIED));
		case DFSInsufficientMemory:
			return( RC_SET( NE_FLM_MEM));
		default:
			return( RC_SET( defaultRc));
	}
}
#endif

/****************************************************************************
Desc:		Map flaim I/O flags to NDS I/O flags for NSS volumes
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
FLMUINT f_getNSSOpenFlags(
   FLMUINT		uiIoFlags,
	FLMBOOL		bDoDirectIo)
{
	FLMUINT		lFlags = zRR_ALLOW_SECURE_DIRECTORY_ACCESS |
								zRR_CANT_DELETE_WHILE_OPEN;

	if (uiIoFlags & (FLM_IO_RDONLY | FLM_IO_RDWR))
	{
		lFlags |= zRR_READ_ACCESS;
	}
	if (uiIoFlags & FLM_IO_RDWR)
	{
		lFlags |= zRR_WRITE_ACCESS;
	}

	if (uiIoFlags & FLM_IO_SH_DENYRW)
	{
		lFlags |= zRR_DENY_READ;
	}
	if (uiIoFlags & (FLM_IO_SH_DENYWR | FLM_IO_SH_DENYRW))
	{
		lFlags |= zRR_DENY_WRITE;
	}
	if (bDoDirectIo)
	{
		lFlags |= zRR_DIO_MODE;
	}
	return( lFlags );
}
#endif

/****************************************************************************
Desc:		Map flaim I/O flags to NetWare I/O flags
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
LONG f_getNWOpenFlags(
   FLMUINT		uiIoFlags,
	FLMBOOL		bDoDirectIo)
{
	LONG	lFlags = 0;

	if (uiIoFlags & (FLM_IO_RDONLY | FLM_IO_RDWR))
	{
		lFlags |= READ_ACCESS_BIT;
	}
	
	if (uiIoFlags & FLM_IO_RDWR)
	{
		lFlags |= WRITE_ACCESS_BIT;
	}

	if (uiIoFlags & FLM_IO_SH_DENYRW )
	{
		lFlags |= DENY_READ_BIT;
	}
	
	if (uiIoFlags & (FLM_IO_SH_DENYWR | FLM_IO_SH_DENYRW))
	{
		lFlags |= DENY_WRITE_BIT;
	}
	
	if (bDoDirectIo)
	{
		lFlags |= NEVER_READ_AHEAD_BIT;
	}
	
	return( lFlags);
}
#endif

/****************************************************************************
Desc: Legacy async I/O completion callback
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
FSTATIC void DirectIONoWaitCallBack(
	LONG		unknownAlwaysZero,
	LONG		callbackData,
	LONG 		completionCode)
{
	RCODE						rc = NE_FLM_OK;
	F_FileAsyncClient *	pAsyncClient = (F_FileAsyncClient *)callbackData;

	F_UNREFERENCED_PARM( unknownAlwaysZero);
	
	if( completionCode != DFSNormalCompletion)
	{
		rc = DfsMapError( completionCode, NE_FLM_DIRECT_WRITING_FILE);
	}

	pAsyncClient->m_completionRc = completionCode;	
	if( RC_OK( completionCode))
	{
		pAsyncClient->m_uiBytesDone = pAsyncClient->m_uiBytesToDo;
	}
	else
	{
		pAsyncClient->m_uiBytesDone = 0;
	}
	
	f_assert( f_semGetSignalCount( pAsyncClient->m_hSem) == 0);
	f_semSignal( pAsyncClient->m_hSem);
}
#endif

/****************************************************************************
Desc: NSS async I/O completion callback
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
FSTATIC void nssDioCallback(
	FLMUINT	reserved,
	FLMUINT	callbackData,
	FLMUINT	completionCode)
{
	RCODE						rc = NE_FLM_OK;
	F_FileAsyncClient *	pAsyncClient = (F_FileAsyncClient *)callbackData;

	F_UNREFERENCED_PARM( reserved);

	if( completionCode != zOK)
	{
		rc = MapNSSError( completionCode, NE_FLM_DIRECT_WRITING_FILE);
	}

	pAsyncClient->m_completionRc = completionCode;	
	if( RC_OK( completionCode))
	{
		pAsyncClient->m_uiBytesDone = pAsyncClient->m_uiBytesToDo;
	}
	else
	{
		pAsyncClient->m_uiBytesDone = 0;
	}
	
	f_assert( f_semGetSignalCount( pAsyncClient->m_hSem) == 0);
	f_semSignal( pAsyncClient->m_hSem);
}
#endif

/****************************************************************************
Desc:		Default Constructor for F_FileHdl class
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
F_FileHdl::F_FileHdl()
{
	initCommonData();
	m_lFileHandle = -1;
	m_lOpenAttr = 0;
	m_lVolumeID = F_NW_DEFAULT_VOLUME_NUMBER;
	m_bDoSuballocation = FALSE;
	m_lLNamePathCount = 0;
	m_lSectorsPerBlock = 0;
	m_lMaxBlocks = 0;
	m_NssKey = 0;
	m_bNSS = FALSE;
	m_bNSSFileOpen = FALSE;
}
#endif

/***************************************************************************
Desc:
***************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
F_FileHdl::~F_FileHdl( void)
{
	if( m_bFileOpened)
	{
		(void)closeFile();
	}
	
	freeCommonData();
}
#endif

/***************************************************************************
Desc:		Open or create a file.
***************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE F_FileHdl::openOrCreate(
	const char *		pFileName,
   FLMUINT				uiIoFlags,
	FLMBOOL				bCreateFlag)
{
	RCODE					rc = NE_FLM_OK;
	LONG					unused;
	void *				unused2;
	char *				pszQualifiedPath = NULL;
	LONG					lErrorCode;
	FLMBYTE *			pLNamePath;
	LONG *				plLNamePathCount;
	struct VolumeInformationStructure *
							pVolumeInfo;
	char *				pszTemp;
	char *				pIoDirPath;
	FLMBOOL				bNssVolume = FALSE;
	FLMBOOL				bDoDirectIO;
	FLMBOOL				bUsingAsync = FALSE;
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();

	bDoDirectIO = (uiIoFlags & FLM_IO_DIRECT) ? TRUE : FALSE;
	
	if( bDoDirectIO)
	{
		bUsingAsync = TRUE;
	}
	
	// Save the file path
	
	if( RC_BAD( rc = f_alloc( F_PATH_MAX_SIZE, &m_pszFileName)))
	{
		goto Exit;
	}
	
	f_strcpy( m_pszFileName, pFileName);
	
	// Allocate other temporary buffers
	
	if( RC_BAD( rc = f_alloc( (FLMUINT)(F_PATH_MAX_SIZE + F_PATH_MAX_SIZE +
		F_PATH_MAX_SIZE + sizeof( struct VolumeInformationStructure) +
		F_PATH_MAX_SIZE), &pszQualifiedPath)))
	{
		goto Exit;
	}

	pIoDirPath = (((char *)pszQualifiedPath) + F_PATH_MAX_SIZE);
	pVolumeInfo = (struct VolumeInformationStructure *)
								(((char *)pIoDirPath) + F_PATH_MAX_SIZE);
	pszTemp = (char *)(((char *)(pVolumeInfo)) +
									sizeof( struct VolumeInformationStructure));

	pLNamePath = (FLMBYTE *)m_pszFileName;
	plLNamePathCount = &m_lLNamePathCount;
	
	ConvertToQualifiedNWPath( pFileName, pszQualifiedPath);

	if( (lErrorCode = ConvertPathToLNameFormat( pszQualifiedPath, &m_lVolumeID,
							&bNssVolume, pLNamePath, plLNamePathCount)) != 0)
	{
      rc = f_mapPlatformError( lErrorCode, NE_FLM_PARSING_FILE_NAME);
		goto Exit;
   }

	// Determine if the volume is NSS or not

	if( gv_bNSSKeyInitialized)
	{
		if( bNssVolume)
		{
			m_bNSS = TRUE;
		}
	}

	if( bDoDirectIO)
	{
		if( !m_bNSS)
		{
			if( (lErrorCode = ReturnVolumeMappingInformation( 
				m_lVolumeID, pVolumeInfo)) != 0)
			{
				rc = DfsMapError( lErrorCode, NE_FLM_INITIALIZING_IO_SYSTEM);
				goto Exit;
			}
			
			m_lSectorsPerBlock = 
					(LONG)(pVolumeInfo->VolumeAllocationUnitSizeInBytes /
							 FLM_NLM_SECTOR_SIZE);
			m_lMaxBlocks = (LONG)(f_getMaxFileSize() /
								(FLMUINT)pVolumeInfo->VolumeAllocationUnitSizeInBytes);
		}
		else
		{
			m_lMaxBlocks = (LONG)(f_getMaxFileSize() / (FLMUINT)65536);
		}
	}

	m_uiBytesPerSector = FLM_NLM_SECTOR_SIZE;
	m_ui64NotOnSectorBoundMask = m_uiBytesPerSector - 1;
	m_ui64GetSectorBoundMask = ~m_ui64NotOnSectorBoundMask;
	
	// Set up the file characteristics requested by caller.

	if( bCreateFlag)
	{
		// File is to be created

		if( f_netwareTestIfFileExists( pszQualifiedPath ) == NE_FLM_OK)
		{
			if( uiIoFlags & FLM_IO_EXCL)
			{
				rc = RC_SET( NE_FLM_IO_ACCESS_DENIED);
				goto Exit;
			}
			
			(void)f_netwareDeleteFile( pszQualifiedPath);
		}
	}

	// Try to create or open the file

	if( m_bNSS)
	{
		FLMINT	lStatus;

		if( bCreateFlag)
		{
			FLMUINT64	qFileAttr;

			qFileAttr = (FLMUINT64)(((m_bDoSuballocation)
									 ? (FLMUINT)(zFA_DO_NOT_COMPRESS_FILE)
									 : (FLMUINT)(zFA_NO_SUBALLOC |
												 zFA_DO_NOT_COMPRESS_FILE)) |
									zFA_IMMEDIATE_PURGE);

Retry_NSS_Create:

			m_lOpenAttr = f_getNSSOpenFlags( uiIoFlags, bDoDirectIO);
			if( (lStatus = gv_zCreateFunc( gv_NssRootKey, 1, 0,
				zNSPACE_LONG | zMODE_UTF8, pszQualifiedPath, zFILE_REGULAR,
				qFileAttr, zCREATE_DELETE_IF_THERE, (FLMUINT)m_lOpenAttr,
				&m_NssKey)) != zOK)
			{
				if( uiIoFlags & FLM_IO_CREATE_DIR)
				{
					uiIoFlags &= ~FLM_IO_CREATE_DIR;

					// Remove the file name for which we are creating the directory.

					if( pFileSystem->pathReduce( m_pszFileName, 
						pIoDirPath, pszTemp) == NE_FLM_OK)
					{
						if( RC_OK( pFileSystem->createDir( pIoDirPath)))
						{
							goto Retry_NSS_Create;
						}
					}
				}
				
				rc = MapNSSError( lStatus,
							(RCODE)(bDoDirectIO
									  ? (RCODE)NE_FLM_DIRECT_CREATING_FILE
									  : (RCODE)NE_FLM_CREATING_FILE));
				goto Exit;
			}
			
			m_bNSSFileOpen = TRUE;
		}
		else
		{
			m_lOpenAttr = f_getNSSOpenFlags( uiIoFlags, bDoDirectIO);
			if( (lStatus = gv_zOpenFunc( gv_NssRootKey, 1,
				zNSPACE_LONG | zMODE_UTF8, pszQualifiedPath, (FLMUINT)m_lOpenAttr,
				&m_NssKey)) != zOK)
			{
				rc = MapNSSError( lStatus,
							(RCODE)(bDoDirectIO
									  ? (RCODE)NE_FLM_DIRECT_OPENING_FILE
									  : (RCODE)NE_FLM_OPENING_FILE));
				goto Exit;
			}
			
			m_bNSSFileOpen = TRUE;
		}
	}
	else
	{
		if (bCreateFlag)
		{
			m_lOpenAttr = (LONG)(((m_bDoSuballocation)
											 ? (LONG)(DO_NOT_COMPRESS_FILE_BIT)
											 : (LONG)(NO_SUBALLOC_BIT |
														 DO_NOT_COMPRESS_FILE_BIT)) | 
										IMMEDIATE_PURGE_BIT);

Retry_Create:

			lErrorCode = CreateFile( 0, 1, m_lVolumeID, 0, (BYTE *)pLNamePath,
				*plLNamePathCount, LONGNameSpace, m_lOpenAttr, 0xff,
				PrimaryDataStream, &m_lFileHandle, &unused, &unused2);

			if( lErrorCode != 0 && (uiIoFlags & FLM_IO_CREATE_DIR))
			{
				uiIoFlags &= ~FLM_IO_CREATE_DIR;

				// Remove the file name for which we are creating the directory

				if( pFileSystem->pathReduce( m_pszFileName, 
					pIoDirPath, pszTemp) == NE_FLM_OK)
				{
					if( RC_OK( pFileSystem->createDir( pIoDirPath)))
					{
						goto Retry_Create;
					}
				}
			}

			// Too many error codes map to 255, so we put in a special
			// case check here.

			if( lErrorCode == 255)
			{
				rc = RC_SET( NE_FLM_IO_ACCESS_DENIED);
				goto Exit;
			}
		}
		else
		{
			m_lOpenAttr = f_getNWOpenFlags(uiIoFlags, bDoDirectIO);
			lErrorCode = OpenFile( 0, 1, m_lVolumeID, 0, (BYTE *)pLNamePath,
				*plLNamePathCount, LONGNameSpace, 0, m_lOpenAttr,
				PrimaryDataStream, &m_lFileHandle, &unused, &unused2);

			// Too many error codes map to 255, so we put in a special
			// case check here.

			if( lErrorCode == 255)
			{
				rc = RC_SET( NE_FLM_IO_PATH_NOT_FOUND);
				goto Exit;
			}
		}

		// Check if the file operation was successful

		if( lErrorCode != 0)
		{
			rc = f_mapPlatformError( lErrorCode,
						(RCODE)(bCreateFlag
								  ? (RCODE)(bDoDirectIO
												? (RCODE)NE_FLM_DIRECT_CREATING_FILE
												: (RCODE)NE_FLM_CREATING_FILE)
								  : (RCODE)(bDoDirectIO
												? (RCODE)NE_FLM_DIRECT_OPENING_FILE
												: (RCODE)NE_FLM_OPENING_FILE)));
			goto Exit;
		}

		if( bCreateFlag)
		{
			// Revoke the file handle rights and close the file
			// (signified by passing 2 for the QueryFlag parameter).
			// If this call fails and returns a 255 error, it may
			// indicate that the FILESYS.NLM being used on the server
			// does not implement option 2 for the QueryFlag parameter.
			// In this case, we will default to our old behavior
			// and simply call CloseFile.  This, potentially, will
			// not free all of the lock objects and could result in
			// a memory leak in filesys.nlm.  However, we want to
			// at least make sure that there is a corresponding
			// RevokeFileHandleRights or CloseFile call for every
			// file open / create call.

			if( (lErrorCode = RevokeFileHandleRights( 0, 1, 
					m_lFileHandle, 2, m_lOpenAttr & 0x0000000F, &unused)) == 0xFF)
			{
				lErrorCode = CloseFile( 0, 1, m_lFileHandle);
			}
			
			m_lOpenAttr = 0;

			if( lErrorCode != 0)
			{
				rc = f_mapPlatformError(lErrorCode, NE_FLM_CLOSING_FILE);
				goto Exit;
			}

			m_lOpenAttr = f_getNWOpenFlags(uiIoFlags, bDoDirectIO);
			lErrorCode = OpenFile( 0, 1, m_lVolumeID, 0, (BYTE *)pLNamePath,
				*plLNamePathCount, LONGNameSpace, 0, m_lOpenAttr,
				PrimaryDataStream, &m_lFileHandle, &unused, &unused2);

			if( lErrorCode != 0)
			{
				// Too many error codes map to 255, so we put in a special
				// case check here.

				if( lErrorCode == 255)
				{
					rc = RC_SET( NE_FLM_IO_PATH_NOT_FOUND);
				}
				else
				{
					rc = f_mapPlatformError( lErrorCode,
								(RCODE)(bDoDirectIO
										 ? (RCODE)NE_FLM_DIRECT_OPENING_FILE
										 : (RCODE)NE_FLM_OPENING_FILE));
				}
				goto Exit;
			}
		}
		
		if( bDoDirectIO)
		{
			lErrorCode = SwitchToDirectFileMode(0, m_lFileHandle);
			if( lErrorCode != 0)
			{
				if( RevokeFileHandleRights( 0, 1, 
					m_lFileHandle, 2, m_lOpenAttr & 0x0000000F, &unused) == 0xFF)
				{
					(void)CloseFile( 0, 1, m_lFileHandle);
				}
				rc = f_mapPlatformError( lErrorCode,
						(RCODE)(bCreateFlag
								  ? (RCODE)NE_FLM_DIRECT_CREATING_FILE
								  : (RCODE)NE_FLM_DIRECT_OPENING_FILE));
				goto Exit;
			}
		}
	}

	if( uiIoFlags & FLM_IO_DELETE_ON_RELEASE)
	{
		m_bDeleteOnRelease = TRUE;
	}
	else
	{
		m_bDeleteOnRelease = FALSE;
	}

	// Allocate at least 64K - this will handle most read and write
	// operations and will also be a multiple of the sector size most of
	// the time.  The calculation below rounds it up to the next sector
	// boundary if it is not already on one.

	m_uiAlignedBuffSize = 64 * 1024;
	if( bDoDirectIO)
	{
		m_uiAlignedBuffSize = (FLMUINT)roundToNextSector( m_uiAlignedBuffSize);
	}

	if( RC_BAD( rc = f_allocAlignedBuffer( m_uiAlignedBuffSize, 
		&m_pucAlignedBuff)))
	{
		goto Exit;
	}
	
	if( bDoDirectIO)
	{
		if( uiIoFlags & FLM_IO_NO_MISALIGNED)
		{
			m_bRequireAlignedIO = TRUE;
		}
	}
	
	m_bFileOpened = TRUE;
	m_bDoDirectIO = bDoDirectIO;
	m_bOpenedInAsyncMode = bUsingAsync;
	m_ui64CurrentPos = 0;
	m_bOpenedReadOnly = (uiIoFlags & FLM_IO_RDONLY) ? TRUE : FALSE;
	m_bOpenedExclusive = (uiIoFlags & FLM_IO_SH_DENYRW) ? TRUE : FALSE;
	f_atomicInc( &gv_openFiles);

Exit:

	if( RC_BAD( rc))
	{
		closeFile();
	}

	if( pszQualifiedPath)
	{
		f_free( &pszQualifiedPath);
	}

   return( rc);
}
#endif

/****************************************************************************
Desc:		Close a file
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE FTKAPI F_FileHdl::closeFile( void)
{
	if( m_bNSS)
	{
		if( m_bNSSFileOpen)
		{
			(void)gv_zCloseFunc( m_NssKey);
			m_bNSSFileOpen = FALSE;
		}
	}
	else if( m_lFileHandle != -1)
	{
		LONG			unused;

		// Revoke the file handle rights and close the file
		// (signified by passing 2 for the QueryFlag parameter).
		// If this call fails and returns a 255 error, it may
		// indicate that the FILESYS.NLM being used on the server
		// does not implement option 2 for the QueryFlag parameter.
		// In this case, we will default to our old behavior
		// and simply call CloseFile.  This, potentially, will
		// not free all of the lock objects and could result in
		// a memory leak in filesys.nlm.  However, we want to
		// at least make sure that there is a corresponding
		// RevokeFileHandleRights or CloseFile call for every
		// file open / create call.

		if( RevokeFileHandleRights( 0, 1, 
				m_lFileHandle, 2, m_lOpenAttr & 0x0000000F, &unused) == 0xFF)
		{
			(void)CloseFile( 0, 1, m_lFileHandle);
		}
		
		m_lFileHandle = -1;
	}

	if( m_bDeleteOnRelease)
	{
		if( m_bNSS)
		{
			(void)gv_zDeleteFunc( gv_NssRootKey, 0, zNSPACE_LONG | zMODE_UTF8,
								m_pszFileName, zMATCH_ALL, 0);
		}
		else
		{
			(void)EraseFile( 0, 1, m_lVolumeID, 0, (BYTE *)m_pszFileName,
				m_lLNamePathCount, LONGNameSpace, 0);
		}

		m_bDeleteOnRelease = FALSE;
		m_lLNamePathCount = 0;
	}
	
	if( m_bFileOpened)
	{
		f_atomicDec( &gv_openFiles);
	}
	
	freeCommonData();

	m_bFileOpened = FALSE;
	m_ui64CurrentPos = 0;
	m_bOpenedReadOnly = FALSE;
	m_bOpenedExclusive = FALSE;
	m_bDoDirectIO = FALSE;
	m_bOpenedInAsyncMode = FALSE;
	m_lOpenAttr = 0;
	m_lFileHandle = -1;

	return( NE_FLM_OK);
}
#endif

/****************************************************************************
Desc:		Return the size of the file
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE FTKAPI F_FileHdl::size(
	FLMUINT64 *		pui64Size)
{
	RCODE				rc = NE_FLM_OK;
	LONG				lErr;
	LONG				lSize;

	if( m_bNSS)
	{
		FLMINT	lStatus;
		zInfo_s	Info;

		if( (lStatus = gv_zGetInfoFunc( m_NssKey,
								zGET_STORAGE_USED,
								sizeof( Info), zINFO_VERSION_A,
								&Info)) != zOK)
		{
			rc = MapNSSError( lStatus, NE_FLM_GETTING_FILE_INFO);
			goto Exit;
		}
		
		f_assert( Info.infoVersion == zINFO_VERSION_A);
		*pui64Size = (FLMUINT64)Info.std.logicalEOF;
	}
	else
	{
		if( (lErr = GetFileSize( 0, m_lFileHandle, &lSize)) != 0)
		{
			rc = f_mapPlatformError( lErr, NE_FLM_GETTING_FILE_SIZE);
		}
		
		*pui64Size = (FLMUINT64)lSize;
	}

Exit:

	return( rc);
}
#endif

/****************************************************************************
Desc:		Truncate the file to the indicated size
WARNING: Direct IO methods are calling this method.  Make sure that all
			changes to this method work in direct IO mode.
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE FTKAPI F_FileHdl::truncateFile(
	FLMUINT64		ui64Size)
{
	RCODE				rc = NE_FLM_OK;
	LONG				lErr;

	f_assert( m_bFileOpened);

	if( m_bNSS)
	{
		FLMINT	lStatus;
		
		if( (lStatus = gv_zSetEOFFunc( m_NssKey, 0, ui64Size,
								zSETSIZE_NON_SPARSE_FILE |
								zSETSIZE_NO_ZERO_FILL |
								zSETSIZE_UNDO_ON_ERR)) != zOK)
		{
			rc = MapNSSError( lStatus, NE_FLM_TRUNCATING_FILE);
			goto Exit;
		}
	}
	else
	{
		if( (lErr = SetFileSize( 0, m_lFileHandle, (FLMUINT)ui64Size, TRUE)) != 0)
		{
			rc = f_mapPlatformError( lErr, NE_FLM_TRUNCATING_FILE);
			goto Exit;
		}
	}
	
	if( m_ui64CurrentPos > ui64Size)
	{
		m_ui64CurrentPos = ui64Size;
	}
	
Exit:

	return( rc);
}
#endif

/***************************************************************************
Desc:		Expand a file for writing.
***************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE F_FileHdl::expand(
	LONG			lStartSector,
	LONG			lSectorsToAlloc)
{
	RCODE			rc = NE_FLM_OK;
	LONG			lResult;
	LONG			lBlockNumber;
	LONG			lStartBlockNumber;
	LONG			lNumBlocksToAlloc;
	LONG			lNumBlocksAllocated;
	LONG			lMinToAlloc;
	LONG			lLastBlockNumber;
	LONG			lTotalToAlloc;
	LONG			lExtendSize;
	FLMUINT		uiFileSize;
	FLMUINT		uiRequestedExtendSize = m_uiExtendSize;
	FLMBOOL		bVerifyFileSize = FALSE;

	// If the requested extend size is the "special" value of ~0,
	// we will set the requested size to 0, so that we will use the
	// minimum default below.  This allows us to somewhat emulate what
	// the Window's code does.

	if( uiRequestedExtendSize == (FLMUINT)(~0))
	{
		uiRequestedExtendSize = 0;
	}

	if( m_bNSS)
	{
		lStartBlockNumber = lStartSector / NSS_SECTORS_PER_BLOCK;
		lLastBlockNumber = (lStartSector + lSectorsToAlloc) / NSS_SECTORS_PER_BLOCK;
		if (((lStartSector + lSectorsToAlloc) % NSS_SECTORS_PER_BLOCK) != 0)
		{
			lLastBlockNumber++;
		}
		lExtendSize = uiRequestedExtendSize / NSS_BLOCK_SIZE;
	}
	else
	{
		lStartBlockNumber = lStartSector / m_lSectorsPerBlock;
		lLastBlockNumber = (lStartSector + lSectorsToAlloc) / m_lSectorsPerBlock;
		if (((lStartSector + lSectorsToAlloc) % m_lSectorsPerBlock) != 0)
		{
			lLastBlockNumber++;
		}
		lExtendSize = uiRequestedExtendSize / (m_lSectorsPerBlock * FLM_NLM_SECTOR_SIZE);
	}

	// Last block number better be greater than or equal to
	// start block number.

	f_assert( lLastBlockNumber >= lStartBlockNumber);
	lMinToAlloc = lLastBlockNumber - lStartBlockNumber;

	if( lExtendSize < 5)
	{
		lExtendSize = 5;
	}

	// Allocate up to lExtendSize blocks at a time - hopefully this will be
	// more efficient.

	if( lMinToAlloc < lExtendSize)
	{
		lTotalToAlloc = lExtendSize;
	}
	else if( lMinToAlloc % lExtendSize == 0)
	{
		lTotalToAlloc = lMinToAlloc;
	}
	else
	{
		// Keep the total blocks to allocate a multiple of lExtendSize.

		lTotalToAlloc = lMinToAlloc - 
			(lMinToAlloc % lExtendSize) + lExtendSize;
	}
	
	lNumBlocksToAlloc = lTotalToAlloc;
	lBlockNumber = lStartBlockNumber;
	lNumBlocksAllocated = 0;

	// Must not go over maximum file size.

	if( lStartBlockNumber + lTotalToAlloc > m_lMaxBlocks)
	{
		lNumBlocksToAlloc = lTotalToAlloc = m_lMaxBlocks - lStartBlockNumber;
		
		if( lTotalToAlloc < lMinToAlloc)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_IO_DISK_FULL);
			goto Exit;
		}
	}

	if( m_bNSS)
	{
		FLMINT	lStatus;

		for( ;;)
		{
			if( (lStatus = gv_zSetEOFFunc( m_NssKey, 0,
				(FLMUINT64)lBlockNumber * 65536 + lNumBlocksToAlloc * 65536,
				zSETSIZE_NO_ZERO_FILL | zSETSIZE_NON_SPARSE_FILE)) != zOK)
			{
				if( lStatus == zERR_OUT_OF_SPACE)
				{
					if( lNumBlocksToAlloc > lMinToAlloc)
					{
						lNumBlocksToAlloc--;
						continue;
					}
				}
				
				rc = MapNSSError( lStatus, NE_FLM_EXPANDING_FILE);
				goto Exit;
			}
			else
			{
				break;
			}
		}
	}
	else
	{
		for (;;)
		{
			lResult = ExpandFileInContiguousBlocks( 0, m_lFileHandle, 
									lBlockNumber, lNumBlocksToAlloc, -1, -1);

			// If we couldn't allocate space, see if we can free some of
			// the limbo space on the volume.

			if( lResult == DFSInsufficientSpace || lResult == DFSBoundryError)
			{
				// May not have been able to get contiguous space for
				// multiple blocks.  If we were asking for more than
				// one, reduce the number we are asking for and try
				// again.

				if( lNumBlocksToAlloc > 1)
				{
					lNumBlocksToAlloc--;
					continue;
				}

				// If we could not even get one block, it is time to
				// try and free some limbo space.

				lResult = FreeLimboVolumeSpace( (LONG)m_lVolumeID, 1);
				if( lResult == DFSInsufficientLimboFileSpace)
				{
					// It is not an error to be out of space if
					// we successfully allocated at least the minimum
					// number of blocks needed.

					if( lNumBlocksAllocated >= lMinToAlloc)
					{
						break;
					}
					else
					{
						rc = RC_SET_AND_ASSERT( NE_FLM_IO_DISK_FULL);
						goto Exit;
					}
				}
				
				continue;
			}
			else if( lResult == DFSOverlapError)
			{
				lResult = 0;
				bVerifyFileSize = TRUE;

				// If lNumBlocksToAlloc is greater than one, we
				// don't know exactly where the hole is, so we need
				// to try filling exactly one block right where
				// we are at.

				// If lNumBlocksToAlloc is exactly one, we know that
				// we have a block right where we are at, so we let
				// the code fall through as if the expand had
				// succeeded.

				if( lNumBlocksToAlloc > 1)
				{
					// If we have an overlap, try getting one block at
					// the current block number - need to make sure this
					// is not where the hole is at.

					lNumBlocksToAlloc = 1;
					continue;
				}
			}
			else if( lResult != 0)
			{
				rc = DfsMapError( lResult, NE_FLM_EXPANDING_FILE);
				goto Exit;
			}
			
			lNumBlocksAllocated += lNumBlocksToAlloc;
			lBlockNumber += lNumBlocksToAlloc;
			
			if( lNumBlocksAllocated >= lTotalToAlloc)
			{
				break;
			}
			else if( lNumBlocksToAlloc > lTotalToAlloc - lNumBlocksAllocated)
			{
				lNumBlocksToAlloc = lTotalToAlloc - lNumBlocksAllocated;
			}
		}

		// If bVerifyFileSize is TRUE, we had an overlap error, which means
		// that we may have had a hole in the file.  In that case, we
		// do NOT want to truncate the file to an incorrect size, so we
		// get the current file size to make sure we are not reducing it
		// down inappropriately.  NOTE: This is not foolproof - if we have
		// a hole that is exactly the size we asked for, we will not verify
		// the file size.

		uiFileSize = (FLMUINT)(lStartBlockNumber + lNumBlocksAllocated) *
				(FLMUINT)m_lSectorsPerBlock * (FLMUINT)FLM_NLM_SECTOR_SIZE;
				
		if( bVerifyFileSize)
		{
			LONG	lCurrFileSize;

			lResult = GetFileSize( 0, m_lFileHandle, &lCurrFileSize);
			
			if( lResult != DFSNormalCompletion)
			{
				rc = DfsMapError( lResult, NE_FLM_GETTING_FILE_SIZE);
				goto Exit;
			}
			
			if( (FLMUINT)lCurrFileSize > uiFileSize)
			{
				uiFileSize = (FLMUINT)lCurrFileSize;
			}
		}

		// This call of SetFileSize is done to force the directory entry file size
		// to account for the newly allocated blocks.  It also forces the directory
		// entry to be updated on disk.  If we didn't do this here, the directory
		// entry's file size on disk would not account for this block.
		// Thus, if we crashed after writing data to this
		// newly allocated block, we would lose the data in the block.

		lResult = SetFileSize( 0, m_lFileHandle, uiFileSize, FALSE);
		
		if( lResult != DFSNormalCompletion)
		{
			rc = DfsMapError( lResult, NE_FLM_TRUNCATING_FILE);
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE F_FileHdl::flush( void)
{
	return( NE_FLM_OK);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE FTKAPI F_FileHdl::lock( void)
{
	return( RC_SET_AND_ASSERT( NE_FLM_NOT_IMPLEMENTED));
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE FTKAPI F_FileHdl::unlock( void)
{
	return( RC_SET_AND_ASSERT( NE_FLM_NOT_IMPLEMENTED));
}
#endif

/****************************************************************************
Desc:	Determine if a file or directory exists
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE f_netwareTestIfFileExists(
	const char *	pPath)
{
	RCODE			rc = NE_FLM_OK;
	LONG			unused;
	FLMBYTE		ucPseudoLNamePath[ F_PATH_MAX_SIZE + 1];
	FLMBYTE		ucLNamePath[ F_PATH_MAX_SIZE];
	LONG			lVolumeID;
	LONG			lPathID;
	LONG			lLNamePathCount;
	LONG			lDirectoryID;
	LONG			lErrorCode;

	f_strcpy( (char *)&ucPseudoLNamePath[1], pPath);
	ucPseudoLNamePath[0] = (char)f_strlen( pPath);
	
	if( (lErrorCode = ConvertPathString( 0, 0, ucPseudoLNamePath, &lVolumeID,		
		&lPathID, ucLNamePath, &lLNamePathCount)) != 0)
	{
		goto Exit;
	}

	if( (lErrorCode = MapPathToDirectoryNumber( 0, lVolumeID, 0, ucLNamePath,
		lLNamePathCount, LONGNameSpace, &lDirectoryID, &unused)) != 0)
	{
		goto Exit;
	}

Exit:

	if( lErrorCode == 255 || lErrorCode == 156)
	{
		// Too many error codes map to 255, so we put in a special
		// case check here

		rc = RC_SET( NE_FLM_IO_PATH_NOT_FOUND);
	}
	else if( lErrorCode )
	{
		rc = f_mapPlatformError( lErrorCode, NE_FLM_CHECKING_FILE_EXISTENCE);
	}
	
	return( rc);
}
#endif

/****************************************************************************
Desc:		Delete a file
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE f_netwareDeleteFile(
	const char *	pPath)
{
	RCODE				rc = NE_FLM_OK;
	LONG				lErrorCode;
	char				pszQualifiedPath[ F_PATH_MAX_SIZE];
	FLMBYTE			ucLNamePath[ F_PATH_MAX_SIZE + 1];
	LONG				lLNamePathCount;
	LONG				lVolumeID;
	FLMBOOL			bNssVolume = FALSE;

	ConvertToQualifiedNWPath( pPath, pszQualifiedPath);

	if( (lErrorCode = ConvertPathToLNameFormat( pszQualifiedPath, &lVolumeID,
			&bNssVolume, ucLNamePath, &lLNamePathCount)) != 0)
	{
		rc = f_mapPlatformError( lErrorCode, NE_FLM_RENAMING_FILE);
		goto Exit;
	}

	if( gv_bNSSKeyInitialized && bNssVolume)
	{
		if( (lErrorCode = gv_zDeleteFunc( gv_NssRootKey, 0,
								zNSPACE_LONG | zMODE_UTF8,
								pszQualifiedPath, zMATCH_ALL, 0)) != zOK)
		{
			rc = MapNSSError( lErrorCode, NE_FLM_IO_DELETING_FILE);
			goto Exit;
		}
	}
	else
	{
		if( (lErrorCode = EraseFile( 0, 1, lVolumeID, 0, ucLNamePath,
			lLNamePathCount, LONGNameSpace, 0)) != 0)
		{
			// Too many error codes map to 255, so we put in a special
			// case check here.

			if( lErrorCode == 255)
			{
				rc = RC_SET( NE_FLM_IO_PATH_NOT_FOUND);
			}
			else
			{
				rc = f_mapPlatformError( lErrorCode, NE_FLM_IO_DELETING_FILE);
			}
			
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}
#endif

/****************************************************************************
Desc:	Turn off the rename inhibit bit for a file in an NSS volume.
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
FSTATIC RCODE nssTurnOffRenameInhibit(
	const char *	pszFileName)
{
	RCODE				rc = NE_FLM_OK;
	zInfo_s			Info;
	FLMINT64			NssKey;
	FLMBOOL			bFileOpened = FALSE;
	FLMINT			lStatus;
	FLMUINT			nOpenAttr;

	nOpenAttr = f_getNSSOpenFlags( (FLMUINT)(FLM_IO_RDWR |
															FLM_IO_SH_DENYNONE), FALSE);
															
	if( (lStatus = gv_zOpenFunc( gv_NssRootKey, 1, zNSPACE_LONG | zMODE_UTF8,
					pszFileName, nOpenAttr, &NssKey)) != zOK)
	{
		rc = MapNSSError( lStatus, NE_FLM_OPENING_FILE);
		goto Exit;
	}
	
	bFileOpened = TRUE;

	// Get the file attributes.

	if( (lStatus = gv_zGetInfoFunc( NssKey, zGET_STD_INFO, sizeof( Info),
				zINFO_VERSION_A, &Info)) != zOK)
	{
		rc = MapNSSError( lStatus, NE_FLM_GETTING_FILE_INFO);
		goto Exit;
	}
	
	f_assert( Info.infoVersion == zINFO_VERSION_A);

	// See if the rename inhibit bit is set.

	if( Info.std.fileAttributes & zFA_RENAME_INHIBIT)
	{
		// Turn bit off

		Info.std.fileAttributes = 0;

		// Specify which bits to modify - only rename inhibit in this case

		Info.std.fileAttributesModMask = zFA_RENAME_INHIBIT;

		if( (lStatus = gv_zModifyInfoFunc( NssKey, 0, zMOD_FILE_ATTRIBUTES,
			sizeof( Info), zINFO_VERSION_A, &Info)) != zOK)
		{
			rc = MapNSSError( lStatus, NE_FLM_SETTING_FILE_INFO);
			goto Exit;
		}
	}
	
Exit:

	if( bFileOpened)
	{
		(void)gv_zCloseFunc( NssKey);
	}
	
	return( rc);
}
#endif

/****************************************************************************
Desc:		Rename a file
Notes:	Currently, this function doesn't support moving the file from one
			volume to another.  (There is a CopyFileToFile function that could
			be used to do the move.)  The toolkit function does appear to 
			support moving (copy/delete) the file.
			
			This function does support renaming directories.
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE f_netwareRenameFile(
	const char *	pOldFilePath,
	const char *	pNewFilePath)
{
	RCODE				rc = NE_FLM_OK;
	LONG				unused;
	FLMBYTE			ucOldLNamePath[ F_PATH_MAX_SIZE + 1];
	LONG				lOldLNamePathCount;
	FLMBYTE			ucNewLNamePath[ F_PATH_MAX_SIZE + 1];
	LONG				lNewLNamePathCount;
	LONG				lVolumeID;
	LONG				lErrorCode;
	FLMBYTE			ucPseudoLNamePath[ F_PATH_MAX_SIZE + 1];
	LONG				lPathID;
	LONG				lIsFile;
	FLMBOOL			bIsDirectory;
	struct 			ModifyStructure modifyStruct;
	LONG				lDirectoryID;
	LONG				lFileAttributes;
	LONG				lMatchBits;
	FLMBOOL			bNssVolume =
							(FLMBOOL)(gv_zIsNSSVolumeFunc
								? (gv_zIsNSSVolumeFunc( (const char *)pOldFilePath)
									 ? TRUE
									 : FALSE)
								: FALSE);

	if( gv_bNSSKeyInitialized && bNssVolume)
	{
		FLMINT	lStatus;
		FLMBOOL	bTurnedOffRenameInhibit = FALSE;

Retry_Nss_Rename:

		if( (lStatus = gv_zRenameFunc( gv_NssRootKey, 0,
			zNSPACE_LONG | zMODE_UTF8, pOldFilePath, zMATCH_ALL,
			zNSPACE_LONG | zMODE_UTF8, pNewFilePath, 0)) != zOK)
		{
			if( lStatus == zERR_NO_RENAME_PRIVILEGE && !bTurnedOffRenameInhibit)
			{
				// Attempt to turn off rename inhibit.  This isn't always the
				// reason for zERR_NO_RENAME_PRIVILEGE, but it is one we
				// definitely need to take care of.

				if( RC_BAD( rc = nssTurnOffRenameInhibit( pOldFilePath)))
				{
					goto Exit;
				}
				
				bTurnedOffRenameInhibit = TRUE;
				goto Retry_Nss_Rename;
			}
			
			rc = MapNSSError( lStatus, NE_FLM_RENAMING_FILE);
			goto Exit;
		}
	}
	else
	{
		f_strcpy( (char *)&ucPseudoLNamePath[1], pOldFilePath);
		ucPseudoLNamePath[0] = (char)f_strlen( (const char *)&ucPseudoLNamePath[1] );
		
		if( (lErrorCode = ConvertPathString( 0, 0, ucPseudoLNamePath, &lVolumeID,		
			&lPathID, (BYTE *)ucOldLNamePath, &lOldLNamePathCount)) != 0)
		{
			goto Exit;
		}

		if( (lErrorCode = MapPathToDirectoryNumber( 0, lVolumeID, 0,
			(BYTE *)ucOldLNamePath, lOldLNamePathCount, LONGNameSpace,
			&lDirectoryID, &lIsFile)) != 0)
		{
			goto Exit;
		}
		
		if( lIsFile)
		{
			bIsDirectory = FALSE;
			lMatchBits = 0;
		}
		else
		{
			bIsDirectory = TRUE;
			lMatchBits = SUBDIRECTORY_BIT;
		}
		
		f_strcpy( (char *)&ucPseudoLNamePath[1], pNewFilePath);
		ucPseudoLNamePath[0] = (char)f_strlen( (const char *)&ucPseudoLNamePath[1]);
		
		if( (lErrorCode = ConvertPathString( 0, 0, ucPseudoLNamePath, &unused,
			&lPathID, (BYTE *)ucNewLNamePath, &lNewLNamePathCount)) != 0)
		{
			goto Exit;
		}

		{
			struct DirectoryStructure * pFileInfo;

			if( (lErrorCode = VMGetDirectoryEntry( lVolumeID, 
				lDirectoryID & 0x00ffffff, &pFileInfo)) != 0)
			{
				goto Exit;
			}
			
			lFileAttributes = pFileInfo->DFileAttributes;
		}
		
		if( lFileAttributes & RENAME_INHIBIT_BIT )
		{
			f_memset(&modifyStruct, 0, sizeof(modifyStruct));
			modifyStruct.MFileAttributesMask = RENAME_INHIBIT_BIT;
			
			if( (lErrorCode = ModifyDirectoryEntry( 0, 1, lVolumeID, 0,
				(BYTE *)ucOldLNamePath, lOldLNamePathCount, LONGNameSpace, 
				lMatchBits, LONGNameSpace, &modifyStruct,
				MFileAttributesBit, 0)) != 0)
			{
				goto Exit;
			}
		}

		lErrorCode = RenameEntry( 0, 1, lVolumeID, 0, ucOldLNamePath,
			lOldLNamePathCount, LONGNameSpace, lMatchBits,
			(BYTE)bIsDirectory ? 1 : 0, 0, ucNewLNamePath, lNewLNamePathCount,
			TRUE, TRUE);

		if( lFileAttributes & RENAME_INHIBIT_BIT )
		{
			FLMBYTE *		pFileName;

			if( lErrorCode )
			{
				pFileName = ucOldLNamePath;
				lNewLNamePathCount = lOldLNamePathCount;
			}
			else
			{
				pFileName = ucNewLNamePath;
			}
				
			// Turn the RENAME_INHIBIT_BIT back on
			
			f_memset(&modifyStruct, 0, sizeof(modifyStruct));
			modifyStruct.MFileAttributes = RENAME_INHIBIT_BIT;
			modifyStruct.MFileAttributesMask = RENAME_INHIBIT_BIT;

			(void)ModifyDirectoryEntry( 0, 1, lVolumeID, 0, (BYTE *)pFileName,
				lNewLNamePathCount, LONGNameSpace, lMatchBits, LONGNameSpace,
				&modifyStruct, MFileAttributesBit, 0);
		}
	}

Exit:

	if( !gv_bNSSKeyInitialized || !bNssVolume)
	{
		if( lErrorCode )
		{
			// Too many error codes map to 255, so we put in a special
			// case check here.

			if( lErrorCode == 255)
			{
				rc = RC_SET( NE_FLM_IO_PATH_NOT_FOUND);
			}
			else
			{
				rc = f_mapPlatformError( lErrorCode, NE_FLM_RENAMING_FILE);
			}
		}
	}
	
	return( rc);
}
#endif

/****************************************************************************
Desc:		Convert the given path to NetWare LName format.
Input:	pPath = qualified netware path of the format:
						volume:directory_1\...\directory_n\filename.ext
Output:	plVolumeID = NetWare volume ID
			pLNamePath = NetWare LName format path
											 
				Netware expects paths to be in LName format:
					<L1><C1><L2><C2>...<Ln><Cn>
					where <Lx> is a one-byte length and <Cx> is a path component.
					
					Example: 6SYSTEM4Fred
						note that the 6 and 4 are binary, not ASCII

			plLNamePathCount = number of path components in pLNamePath
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
FSTATIC LONG ConvertPathToLNameFormat(
	const char *	pPath,
	LONG *			plVolumeID,
	FLMBOOL *		pbNssVolume,
	FLMBYTE *		pLNamePath,
	LONG *			plLNamePathCount)
{
	FLMBYTE			ucPseudoLNamePath[ F_PATH_MAX_SIZE + 1];
	LONG				lPathID;
	LONG				lErrorCode = 0;

	*pLNamePath = 0;
	*plLNamePathCount = 0;

	*pbNssVolume = (FLMBOOL)(gv_zIsNSSVolumeFunc
									? (gv_zIsNSSVolumeFunc( (const char *)pPath)
										? TRUE
										: FALSE)
									: FALSE);

	if( gv_bNSSKeyInitialized && *pbNssVolume)
	{
		f_strcpy( (char *)pLNamePath, pPath);
		*plLNamePathCount = 1;
	}
	else
	{
		f_strcpy( (char *)&ucPseudoLNamePath[1], pPath);
		ucPseudoLNamePath[0] = (FLMBYTE)f_strlen( (const char *)&ucPseudoLNamePath[1]);
		
		if( (lErrorCode = ConvertPathString( 0, 0, ucPseudoLNamePath, plVolumeID,		
			&lPathID, (BYTE *)pLNamePath, plLNamePathCount)) != 0)
		{
			goto Exit;
		}
	}

Exit:

	return( lErrorCode );
}
#endif

/****************************************************************************
Desc:		Convert the given path to a NetWare format.  The format isn't 
			critical, it just needs to be consistent.  See below for a 
			description of the format chosen.
Input:	pInputPath = a path to a file
Output:	pszQualifiedPath = qualified netware path of the format:
									volume:directory_1\...\directory_n\filename.ext
									
			If no volume is given, "SYS:" is the default.
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
FSTATIC void ConvertToQualifiedNWPath(
	const char *		pInputPath,
	char *				pszQualifiedPath)
{
	char					ucFileName [F_FILENAME_SIZE];
	char					ucVolume [MAX_NETWARE_VOLUME_NAME+1];
	char					ucPath [F_PATH_MAX_SIZE + 1];
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();

	// Separate path into its components: volume, path...

	pFileSystem->pathParse( pInputPath, NULL, ucVolume, ucPath, ucFileName);

	// Rebuild path to a standard, fully-qualified format, defaulting the
	// volume if one isn't specified.

	*pszQualifiedPath = 0;
	if( ucVolume [0])
	{
		// Append the volume specified by the user.

		f_strcat( pszQualifiedPath, ucVolume );
	}
	else
	{
		// No volume specified, use the default

		f_strcat( pszQualifiedPath, "SYS:");
	}
	
	if( ucPath [0])
	{
		// User specified a path...

		if( ucPath[0] == '\\' || ucPath[0] == '/' )
		{
			// Append the path to the volume without the leading slash

			f_strcat( pszQualifiedPath, &ucPath [1]);
		}
		else
		{
			// Append the path to the volume

			f_strcat( pszQualifiedPath, ucPath);
		}
	}

	if( ucFileName [0])
	{
		// Append the file name to the path

		pFileSystem->pathAppend( pszQualifiedPath, ucFileName);
	}
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
void FTKAPI f_yieldCPU( void)
{
	kYieldIfTimeSliceUp();
}
#endif

/****************************************************************************
Desc: 	Function that must be called within a NLM's startup routine.
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE f_netwareStartup( void)
{
	RCODE		rc = NE_FLM_OK;

	if( f_atomicInc( &gv_NetWareStartupCount) != 1)
	{
		goto Exit;
	}

	gv_MyModuleHandle = CFindLoadModuleHandle( (void *)f_netwareShutdown);

	// Allocate the needed resource tags

	if( (gv_lAllocRTag = AllocateResourceTag( gv_MyModuleHandle,
		"FLAIM Memory", AllocSignature)) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	// Initialize NSS
	
	if( RC_BAD( rc = f_nssInitialize()))
	{
		goto Exit;
	}

Exit:

	if( RC_BAD( rc))
	{
		f_netwareShutdown();
	}

	return( rc);
}
#endif

/****************************************************************************
Desc: 	Closes (Frees) any resources used by FLAIM's clib patches layer.
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
void f_netwareShutdown( void)
{
	// Call exit function.

	if( f_atomicDec( &gv_NetWareStartupCount) != 0)
	{
		goto Exit;
	}
	
	f_nssUninitialize();

	if( gv_lAllocRTag)
	{
		ReturnResourceTag( gv_lAllocRTag, 1);
		gv_lAllocRTag = 0;
	}

	gv_MyModuleHandle = NULL;

Exit:

	return;
}
#endif

/****************************************************************************
Desc: 	
****************************************************************************/
void * f_getNLMHandle( void)
{
#if defined( FLM_RING_ZERO_NLM)
	return( gv_MyModuleHandle);
#else
	return( getnlmhandle());
#endif
}

/**********************************************************************
Desc:
**********************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE FTKAPI f_chdir(
	const char *		pszDir)
{
	F_UNREFERENCED_PARM( pszDir);
	return( RC_SET( NE_FLM_NOT_IMPLEMENTED));
}
#endif
	
/**********************************************************************
Desc:
**********************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE FTKAPI f_getcwd(
	char *			pszDir)
{
	*pszDir = NULL;
	return( RC_SET( NE_FLM_NOT_IMPLEMENTED));
}
#endif


/**********************************************************************
Desc:
**********************************************************************/
#if defined( FLM_RING_ZERO_NLM)
extern "C" void f_fatalRuntimeError( void)
{
	EnterDebugger();
}
#endif
	
#endif // FLM_NLM

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_OSX)
void gv_fnlm()
{
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE F_FileHdl::lowLevelRead(
	FLMUINT64				ui64ReadOffset,
	FLMUINT					uiBytesToRead,
	void *					pvBuffer,
	IF_IOBuffer *			pIOBuffer,
	FLMUINT *				puiBytesRead)
{
	RCODE						rc = NE_FLM_OK;
	FLMUINT					uiBytesRead = 0;

	if( pIOBuffer && pvBuffer && pvBuffer != pIOBuffer->getBufferPtr())
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_ILLEGAL_OP);
		goto Exit;
	}

	if( ui64ReadOffset == FLM_IO_CURRENT_POS)
	{
		ui64ReadOffset = m_ui64CurrentPos;
	}
	else
	{
		m_ui64CurrentPos = ui64ReadOffset;
	}
	
	if( !pvBuffer)
	{
		pvBuffer = pIOBuffer->getBufferPtr();
	}

	if( pIOBuffer)
	{
		pIOBuffer->setPending();
	}

	rc = internalBlockingRead( ui64ReadOffset, uiBytesToRead, pvBuffer, 
		&uiBytesRead);

	m_ui64CurrentPos += uiBytesRead;
	
	if( pIOBuffer)
	{
		pIOBuffer->notifyComplete( rc);
		pIOBuffer = NULL;
	}
	
	if( RC_BAD( rc))
	{
		goto Exit;
	}
	
	if( uiBytesRead < uiBytesToRead)
	{
		rc = RC_SET( NE_FLM_IO_END_OF_FILE);
		goto Exit;
	}
	
Exit:

	f_assert( uiBytesRead || RC_BAD( rc));
	f_assert( uiBytesRead <= uiBytesToRead);

	if( pIOBuffer && !pIOBuffer->isPending())
	{
		f_assert( RC_BAD( rc));
		pIOBuffer->notifyComplete( rc);
	}
	
	if( puiBytesRead)
	{
		*puiBytesRead = uiBytesRead;
	}

	return( rc);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE F_FileHdl::internalBlockingRead(
	FLMUINT64	ui64ReadOffset,
	FLMUINT		uiBytesToRead,	
   void *		pvBuffer,
   FLMUINT *	puiBytesRead)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiBytesRead = 0;
	FLMUINT		uiBytesRemaining = uiBytesToRead;
	
	f_assert( uiBytesToRead);
	
	if( m_bDoDirectIO)
	{
		FLMBYTE *	pucBuffer = (FLMBYTE *)pvBuffer;
		LONG			lStartSector;
		LONG			lSectorCount;
		LONG			lResult;
		BYTE			ucSectorBuf[ FLM_NLM_SECTOR_SIZE];
		FLMUINT		uiBytesToCopy;
		FLMUINT		uiSectorOffset;
		FLMUINT		uiTotal;
		FLMINT		lStatus;
	
		// Calculate the starting sector.
	
		lStartSector = (LONG)(ui64ReadOffset / FLM_NLM_SECTOR_SIZE);
	
		// See if the offset is on a sector boundary.  If not, we must read
		// into the local sector buffer and then copy into the buffer.
		// We must also read into the local buffer if our read size is less
		// than the sector size.
	
		if( (ui64ReadOffset % FLM_NLM_SECTOR_SIZE != 0) ||
			 (uiBytesRemaining < FLM_NLM_SECTOR_SIZE))
		{
			if( m_bNSS)
			{
				if( (lStatus = gv_zDIOReadFunc( m_NssKey, 
										(FLMUINT64)lStartSector, 1,
										(FLMUINT)0, NULL, ucSectorBuf)) != zOK)
				{
					rc = MapNSSError( lStatus, NE_FLM_DIRECT_READING_FILE);
					goto Exit;
				}
			}
			else
			{
				if( (lResult = DirectReadFile( 0, m_lFileHandle, lStartSector,
					1, ucSectorBuf)) != 0)
				{
					rc = DfsMapError( lResult, NE_FLM_DIRECT_READING_FILE);
					goto Exit;
				}
			}
	
			// Copy the part of the sector that was requested into the buffer.
	
			uiSectorOffset = (FLMUINT)(ui64ReadOffset % FLM_NLM_SECTOR_SIZE);
	
			if( (uiBytesToCopy = uiBytesRemaining) > 
					FLM_NLM_SECTOR_SIZE - uiSectorOffset)
			{
				uiBytesToCopy = FLM_NLM_SECTOR_SIZE - uiSectorOffset;
			}
			
			f_memcpy( pucBuffer, &ucSectorBuf[ uiSectorOffset], uiBytesToCopy);
			pucBuffer += uiBytesToCopy;
			uiBytesRemaining -= (FLMUINT)uiBytesToCopy;
			uiBytesRead += uiBytesToCopy;
	
			// See if we got everything we wanted to with this read.
	
			if( !uiBytesRemaining)
			{
				goto Exit;
			}
	
			// Go to the next sector boundary
	
			lStartSector++;
		}
	
		// At this point, we are poised to read on a sector boundary.  See if we
		// have at least one full sector to read.  If so, we can read it directly
		// into the provided buffer.  If not, we must use the temporary sector
		// buffer.
	
		if( uiBytesRemaining >= FLM_NLM_SECTOR_SIZE)
		{
			lSectorCount = (LONG)(uiBytesRemaining / FLM_NLM_SECTOR_SIZE);
	Try_Read:
			if( m_bNSS)
			{
				if( (lStatus = gv_zDIOReadFunc( m_NssKey,
						(FLMUINT64)lStartSector, (FLMUINT)lSectorCount,
						(FLMUINT)0, NULL, pucBuffer)) != zOK)
				{
					if( (lStatus == zERR_END_OF_FILE || lStatus == zERR_BEYOND_EOF) &&
						 (lSectorCount > 1))
					{
	
						// See if we can read one less sector.  We will return
						// NE_FLM_IO_END_OF_FILE in this case.
	
						lSectorCount--;
						rc = RC_SET( NE_FLM_IO_END_OF_FILE);
						goto Try_Read;
					}
					
					rc = MapNSSError( lStatus, NE_FLM_DIRECT_READING_FILE);
					goto Exit;
				}
			}
			else
			{
				if( (lResult = DirectReadFile( 0, m_lFileHandle, lStartSector,
					lSectorCount, pucBuffer)) != 0)
				{
					if( lResult == DFSOperationBeyondEndOfFile && lSectorCount > 1)
					{
						// See if we can read one less sector.  We will return
						// NE_FLM_IO_END_OF_FILE in this case.
	
						lSectorCount--;
						rc = RC_SET( NE_FLM_IO_END_OF_FILE);
						goto Try_Read;
					}
					
					rc = DfsMapError( lResult, NE_FLM_DIRECT_READING_FILE);
					goto Exit;
				}
			}
			
			uiTotal = (FLMUINT)(lSectorCount * FLM_NLM_SECTOR_SIZE);
			pucBuffer += uiTotal;
			uiBytesRead += uiTotal;
			uiBytesRemaining -= uiTotal;
	
			// See if we got everything we wanted to or could with this read.
	
			if( !uiBytesRemaining || rc == NE_FLM_IO_END_OF_FILE)
			{
				goto Exit;
			}
	
			// Go to the next sector after the ones we just read
	
			lStartSector += lSectorCount;
		}
	
		// At this point, we have less than a sector's worth to read, so we must
		// read it into a local buffer.
	
		if( m_bNSS)
		{
			if( (lStatus = gv_zDIOReadFunc( m_NssKey, (FLMUINT64)lStartSector, 1,
				(FLMUINT)0, NULL, ucSectorBuf)) != zOK)
			{
				rc = MapNSSError( lStatus, NE_FLM_DIRECT_READING_FILE);
				goto Exit;
			}
		}
		else
		{
			if( (lResult = DirectReadFile( 0, m_lFileHandle, 
				lStartSector, 1, ucSectorBuf)) != 0)
			{
				rc = DfsMapError( lResult, NE_FLM_DIRECT_READING_FILE);
				goto Exit;
			}
		}
	
		// Copy the part of the sector that was requested into the buffer.
	
		uiBytesRead += uiBytesRemaining;
		f_memcpy( pucBuffer, ucSectorBuf, uiBytesRemaining);
	}
	else
	{
		FCBType *	fcb;
		LONG			lBytesRead;
		LONG			lErr;

		if( m_bNSS)
		{
			FLMINT	lStatus;
	
			if( (lStatus = gv_zReadFunc( m_NssKey, 0, ui64ReadOffset,
					(FLMUINT)uiBytesRemaining, pvBuffer, &uiBytesRead)) != zOK)
			{
				f_assert( 0);
				rc = MapNSSError( lStatus, NE_FLM_READING_FILE);
				goto Exit;
			}
		}
		else
		{
			if( (lErr = MapFileHandleToFCB( m_lFileHandle, &fcb)) != 0)
			{
				rc = f_mapPlatformError( lErr, NE_FLM_SETTING_UP_FOR_READ);
				goto Exit;
			}
			
			if( (lErr = ReadFile( fcb->Station, m_lFileHandle, 
				(LONG)ui64ReadOffset, uiBytesRemaining,
				&lBytesRead, pvBuffer)) != 0)
			{
				rc = f_mapPlatformError( lErr, NE_FLM_READING_FILE);
			}

			uiBytesRead = (FLMUINT)lBytesRead;
		}
	}

	if( uiBytesRead < uiBytesToRead)
	{
		rc = RC_SET( NE_FLM_IO_END_OF_FILE);
		goto Exit;
	}
	
Exit:

	f_assert( uiBytesRead <= uiBytesToRead);

	if( puiBytesRead)
	{
		*puiBytesRead = uiBytesRead;
	}

	return( rc);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE F_FileHdl::lowLevelWrite(
	FLMUINT64				ui64WriteOffset,
	FLMUINT					uiBytesToWrite,
	const void *			pvBuffer,
	IF_IOBuffer *			pIOBuffer,
	FLMUINT *				puiBytesWritten)
{
	RCODE						rc = NE_FLM_OK;
	FLMUINT					uiBytesWritten = 0;
	F_FileAsyncClient *	pAsyncClient = NULL;
	FLMBOOL					bWaitForWrite = FALSE;
	
	f_assert( uiBytesToWrite);
	f_assert( !m_bDoDirectIO || (ui64WriteOffset % FLM_NLM_SECTOR_SIZE) == 0);
	f_assert( !m_bDoDirectIO || (uiBytesToWrite % FLM_NLM_SECTOR_SIZE) == 0);
	
	if( pIOBuffer && pvBuffer && pvBuffer != pIOBuffer->getBufferPtr())
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_ILLEGAL_OP);
		goto Exit;
	}
	
	if( ui64WriteOffset == FLM_IO_CURRENT_POS)
	{
		ui64WriteOffset = m_ui64CurrentPos;
	}
	else
	{
		m_ui64CurrentPos = ui64WriteOffset;
	}

	if( !pvBuffer)
	{
		pvBuffer = pIOBuffer->getBufferPtr();
	}

	if( m_bOpenedInAsyncMode)
	{
		LONG				lStartSector;
		LONG				lSectorCount;

		if( RC_BAD( rc = allocFileAsyncClient( &pAsyncClient)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = pAsyncClient->prepareForAsync( pIOBuffer)))
		{
			goto Exit;
		}
			
		if( !pIOBuffer)
		{
			bWaitForWrite = TRUE;
		}
			
		pAsyncClient->m_uiBytesToDo = uiBytesToWrite;
		pIOBuffer = NULL;
		
		// Calculate the starting sector and number of sectors to write
	
		lStartSector = (LONG)(ui64WriteOffset / FLM_NLM_SECTOR_SIZE);
		lSectorCount = (LONG)(uiBytesToWrite / FLM_NLM_SECTOR_SIZE);
		
		if( RC_BAD( rc = writeSectors( pvBuffer, pAsyncClient, 
			lStartSector, lSectorCount)))
		{
			pAsyncClient->notifyComplete( rc, 0);
			goto Exit;
		}
			
		if( bWaitForWrite)
		{
			if( RC_BAD( rc = pAsyncClient->waitToComplete()))
			{
				if( rc != NE_FLM_IO_DISK_FULL)
				{
					goto Exit;
				}
	
				rc = NE_FLM_OK;
			}
				
			f_assert( pAsyncClient->m_uiBytesDone);
			uiBytesWritten = pAsyncClient->m_uiBytesDone; 
		}
		else
		{
			uiBytesWritten = uiBytesToWrite;
		}
		
		m_ui64CurrentPos += uiBytesWritten;
	}
	else
	{
		if( pIOBuffer)
		{
			pIOBuffer->setPending();
		}
		
		rc = internalBlockingWrite( ui64WriteOffset, uiBytesToWrite, 
					pvBuffer, &uiBytesWritten);
		
		m_ui64CurrentPos += uiBytesWritten;

		if( pIOBuffer)
		{
			pIOBuffer->notifyComplete( rc);
			pIOBuffer = NULL;
		}
		
		if( RC_BAD( rc))
		{
			goto Exit;
		}
	}
	
	if( uiBytesWritten < uiBytesToWrite)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_IO_DISK_FULL);
		goto Exit;
	}
	
Exit:

	if( pAsyncClient)
	{
		pAsyncClient->Release();
	}
	
	if( pIOBuffer && !pIOBuffer->isPending())
	{
		f_assert( RC_BAD( rc));
		pIOBuffer->notifyComplete( rc);
	}

	if( puiBytesWritten)
	{
		*puiBytesWritten = uiBytesWritten;
	}
	
	return( rc);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE F_FileHdl::extendFile(
	FLMUINT64				ui64NewFileSize)
{
	RCODE						rc = NE_FLM_OK;
	FLMUINT64				ui64FileSize;
	FLMUINT					uiStartSector;
	FLMUINT					uiSectorCount;
	
	// Get the current file size
	
	if( RC_BAD( rc = size( &ui64FileSize)))
	{
		goto Exit;
	}
	
	// File is already the requested size
	
	if( ui64FileSize >= ui64NewFileSize)
	{
		goto Exit;
	}
	
	// Determine the number of sectors in the file
	
	uiStartSector = (FLMUINT)(ui64FileSize / FLM_NLM_SECTOR_SIZE);
	uiSectorCount = (FLMUINT)(f_roundUp( ui64NewFileSize - ui64FileSize, 
										FLM_NLM_SECTOR_SIZE) / FLM_NLM_SECTOR_SIZE);
	
	if( RC_BAD( rc = expand( uiStartSector, uiSectorCount)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE F_FileHdl::internalBlockingWrite(
	FLMUINT64			ui64WriteOffset,
	FLMUINT				uiBytesToWrite,	
	const void *		pvBuffer,
	FLMUINT *			puiBytesWritten)
{
	RCODE					rc = NE_FLM_OK;
	LONG					lErr;
	FCBType *			fcb;
	FLMUINT				uiBytesWritten = 0;
	FLMUINT				uiBytesRemaining = uiBytesToWrite;

	if( m_bDoDirectIO)
	{
		FLMBYTE *		pucBuffer = (FLMBYTE *)pvBuffer;
		LONG				lStartSector;
		LONG				lSectorCount;
		LONG				lResult;
		BYTE				ucSectorBuf[ FLM_NLM_SECTOR_SIZE];
		FLMUINT			uiBytesToCopy;
		FLMUINT			uiSectorOffset;
		FLMUINT			uiTotal;
		FLMINT			lStatus;
	
		// Calculate the starting sector
	
		lStartSector = (LONG)(ui64WriteOffset / FLM_NLM_SECTOR_SIZE);
	
		// See if the offset is on a sector boundary.  If not, we must first read
		// the sector into memory, overwrite it with data from the input
		// buffer and write it back out again.
	
		if( (ui64WriteOffset % FLM_NLM_SECTOR_SIZE != 0) || 
			 (uiBytesRemaining < FLM_NLM_SECTOR_SIZE))
		{
			if( m_bNSS)
			{
				if( (lStatus = gv_zDIOReadFunc( m_NssKey, 
					(FLMUINT64)lStartSector,
					(FLMUINT)1, (FLMUINT)0, NULL, ucSectorBuf)) != zOK)
				{
					if( lStatus == zERR_END_OF_FILE || lStatus == zERR_BEYOND_EOF)
					{
						f_memset( ucSectorBuf, 0, sizeof( ucSectorBuf));
	
						// Expand the file
	
						if( RC_BAD( rc = expand( lStartSector, 1)))
						{
							goto Exit;
						}
					}
					else
					{
						rc = MapNSSError( lStatus, NE_FLM_DIRECT_READING_FILE);
						goto Exit;
					}
				}
			}
			else
			{
				lResult = DirectReadFile( 0, m_lFileHandle, lStartSector, 
													1, ucSectorBuf);
													
				if( lResult == DFSHoleInFileError || 
					 lResult == DFSOperationBeyondEndOfFile )
				{
					f_memset( ucSectorBuf, 0, sizeof( ucSectorBuf));
	
					// Expand the file
	
					if( RC_BAD( rc = expand( lStartSector, 1)))
					{
						goto Exit;
					}
				}
				else if( lResult != 0)
				{
					rc = DfsMapError( lResult, NE_FLM_DIRECT_READING_FILE);
					goto Exit;
				}
			}
	
			// Copy the part of the buffer that is being written back into
			// the sector buffer.
	
			uiSectorOffset = (FLMUINT)(ui64WriteOffset % FLM_NLM_SECTOR_SIZE);
	
			if( (uiBytesToCopy = uiBytesRemaining) > 
					(FLM_NLM_SECTOR_SIZE - uiSectorOffset))
			{
				uiBytesToCopy = FLM_NLM_SECTOR_SIZE - uiSectorOffset;
			}
			
			f_memcpy( &ucSectorBuf [uiSectorOffset], pucBuffer, uiBytesToCopy);
			pucBuffer += uiBytesToCopy;
			uiBytesRemaining -= uiBytesToCopy;
			uiBytesWritten += uiBytesToCopy;
	
			// Write the sector buffer back out
	
			if( RC_BAD( rc = writeSectors( &ucSectorBuf [0], NULL,
				lStartSector, 1))) 
			{
				goto Exit;
			}
	
			// See if we wrote everything we wanted to with this write
	
			if( !uiBytesRemaining)
			{
				goto Exit;
			}
	
			// Go to the next sector boundary
	
			lStartSector++;
		}
	
		// At this point, we are poised to write on a sector boundary.  See if we
		// have at least one full sector to write.  If so, we can write it directly
		// from the provided buffer.  If not, we must use the temporary sector
		// buffer.
	
		if( uiBytesRemaining >= FLM_NLM_SECTOR_SIZE)
		{
			lSectorCount = (LONG)(uiBytesRemaining / FLM_NLM_SECTOR_SIZE);
			
			if( RC_BAD( rc = writeSectors( (const void *)pucBuffer, NULL,
				lStartSector, lSectorCount)))
			{
				goto Exit;
			}
			
			uiTotal = (FLMUINT)(lSectorCount * FLM_NLM_SECTOR_SIZE);
			pucBuffer += uiTotal;
			uiBytesWritten += uiTotal;
			uiBytesRemaining -= uiTotal;
	
			// See if we wrote everything we wanted to with this write
	
			if( !uiBytesRemaining)
			{
				goto Exit;
			}
	
			// Go to the next sector after the ones we just wrote
	
			lStartSector += lSectorCount;
		}
	
		// At this point, we have less than a sector's worth to write, so we must
		// first read the sector from disk, alter it, and then write it back out.
	
		if( m_bNSS)
		{
			if( (lStatus = gv_zDIOReadFunc( m_NssKey, (FLMUINT64)lStartSector,
				(FLMUINT)1, (FLMUINT)0, NULL, ucSectorBuf)) != zOK)
			{
				if( lStatus == zERR_END_OF_FILE || lStatus == zERR_BEYOND_EOF)
				{
					f_memset( ucSectorBuf, 0, sizeof( ucSectorBuf));
	
					// Expand the file
	
					if( RC_BAD( rc = expand( lStartSector, 1)))
					{
						goto Exit;
					}
				}
				else
				{
					rc = MapNSSError( lStatus, NE_FLM_DIRECT_READING_FILE);
					goto Exit;
				}
			}
		}
		else
		{
			lResult = DirectReadFile( 0, m_lFileHandle, lStartSector,
												1, ucSectorBuf);
												
			if( lResult == DFSHoleInFileError)
			{
				f_memset( ucSectorBuf, 0, sizeof( ucSectorBuf));
	
				// Expand the file
	
				if( RC_BAD( rc = expand( lStartSector, 1)))
				{
					goto Exit;
				}
			}
			else if( lResult != 0)
			{
				rc = DfsMapError( lResult, NE_FLM_DIRECT_READING_FILE);
				goto Exit;
			}
		}
	
		// Copy the rest of the output buffer into the sector buffer
	
		f_memcpy( ucSectorBuf, pucBuffer, uiBytesRemaining);
	
		// Write the sector back to disk
	
		if( RC_BAD( rc = writeSectors( &ucSectorBuf[ 0], NULL, lStartSector, 1)))
		{
			goto Exit;
		}
	
		uiBytesWritten += uiBytesRemaining;
	}
	else
	{
		if( m_bNSS)
		{
			FLMINT	lStatus;
	
			if( (lStatus = gv_zWriteFunc( m_NssKey, 0, ui64WriteOffset,
				uiBytesRemaining, pvBuffer, &uiBytesWritten)) != zOK)
			{
				rc = MapNSSError( lStatus, NE_FLM_WRITING_FILE);
				goto Exit;
			}
		}
		else
		{
			if( (lErr = MapFileHandleToFCB( m_lFileHandle, &fcb)) != 0)
			{
				rc = f_mapPlatformError( lErr, NE_FLM_SETTING_UP_FOR_WRITE);
				goto Exit;
			}
	
			if( (lErr = WriteFile( fcb->Station, m_lFileHandle,
				(LONG)ui64WriteOffset, uiBytesRemaining, (void *)pvBuffer)) != 0)
			{
				rc = f_mapPlatformError( lErr, NE_FLM_WRITING_FILE);
				goto Exit;
			}
			
			uiBytesWritten = uiBytesRemaining;
		}
	}
	
	if( uiBytesWritten < uiBytesToWrite)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_IO_DISK_FULL);
		goto Exit;
	}
	
Exit:

	if( puiBytesWritten)
	{
		*puiBytesWritten = uiBytesWritten;
	}

	return( rc);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE F_FileHdl::writeSectors(
	const void *			pvBuffer,
	F_FileAsyncClient *	pAsyncClient,
	LONG						lStartSector,
	LONG						lSectorCount)
{
	RCODE				rc = NE_FLM_OK;
	LONG				lResult;
	FLMINT			lStatus;
	FLMBOOL			bExpanded = FALSE;

	// Keep trying write until we succeed or get an error we can't deal with.
	// Actually, this will NOT loop forever.  At most, it will try twice - 
	// and then it is only when we get a hole in the file error.

	for( ;;)
	{
		if( m_bNSS)
		{
			if( pAsyncClient)
			{
				lStatus = gv_zDIOWriteFunc( m_NssKey, (FLMUINT64)lStartSector,
									(FLMUINT)lSectorCount, (FLMUINT)pAsyncClient,
									nssDioCallback, pvBuffer);
			}
			else
			{
				lStatus = gv_zDIOWriteFunc( m_NssKey, (FLMUINT64)lStartSector,
									(FLMUINT)lSectorCount, (FLMUINT)0, NULL, pvBuffer);
			}

			// We may need to allocate space to do this write

			if( lStatus == zERR_END_OF_FILE || lStatus == zERR_BEYOND_EOF || 
				 lStatus == zERR_HOLE_IN_DIO_FILE)
			{
				if( bExpanded)
				{
					f_assert( 0);
					rc = MapNSSError( lStatus, NE_FLM_DIRECT_WRITING_FILE);
					goto Exit;
				}

				// Expand the file

				if( RC_BAD( rc = expand( lStartSector, lSectorCount)))
				{
					goto Exit;
				}
				
				bExpanded = TRUE;
				continue;
			}
			else if( lStatus != 0)
			{
				rc = MapNSSError( lStatus, NE_FLM_DIRECT_WRITING_FILE);
				goto Exit;
			}
			
			break;
		}
		else
		{
			LONG			lSize;
			FLMBOOL		bNeedToWriteEOF;

			// Determine if this write will change the EOF.  If so, pre-expand
			// the file.

			if( (lResult = GetFileSize( 0, m_lFileHandle, &lSize)) != 0)
			{
				rc = f_mapPlatformError( lResult, NE_FLM_GETTING_FILE_SIZE);
				goto Exit;
			}
			
			bNeedToWriteEOF = 
				(lSize < (lStartSector + lSectorCount) * FLM_NLM_SECTOR_SIZE)
											? TRUE
											: FALSE;
											
			if( pAsyncClient)
			{
				lResult = DirectWriteFileNoWait( 0, m_lFileHandle,
									lStartSector,lSectorCount,
									(BYTE *)pvBuffer, DirectIONoWaitCallBack, 
									(LONG)pAsyncClient);
			}
			else
			{
				lResult = DirectWriteFile( 0, m_lFileHandle,
									lStartSector, lSectorCount, (BYTE *)pvBuffer);
			}

			// We may need to allocate space to do this write

			if( lResult == DFSHoleInFileError ||
				 lResult == DFSOperationBeyondEndOfFile)
			{
				if( bExpanded)
				{
					f_assert( 0);
					rc = DfsMapError( lResult, NE_FLM_DIRECT_WRITING_FILE);
					goto Exit;
				}

				// Expand the file

				if( RC_BAD( rc = expand( lStartSector, lSectorCount)))
				{
					goto Exit;
				}
				
				bExpanded = TRUE;

				// The Expand method forces the file EOF in the directory
				// entry to be written to disk.

				bNeedToWriteEOF = FALSE;
				continue;
			}
			else if( lResult != 0)
			{
				rc = DfsMapError( lResult, NE_FLM_DIRECT_WRITING_FILE);
				goto Exit;
			}
			else
			{
				// If bNeedToWriteEOF is TRUE, we need to force EOF to disk.

				if( bNeedToWriteEOF)
				{
					LONG	lFileSizeInSectors;
					LONG	lExtraSectors;

					// Set the EOF to the nearest block boundary - so we don't
					// have to do this very often.

					lFileSizeInSectors = lStartSector + lSectorCount;
					lExtraSectors = lFileSizeInSectors % m_lSectorsPerBlock;
					
					if( lExtraSectors)
					{
						lFileSizeInSectors += (m_lSectorsPerBlock - lExtraSectors);
					}
					
					if( (lResult = SetFileSize( 0, m_lFileHandle,
						(FLMUINT)lFileSizeInSectors * FLM_NLM_SECTOR_SIZE,
						FALSE)) != 0)
					{
						rc = DfsMapError( lResult, NE_FLM_TRUNCATING_FILE);
						goto Exit;
					}
				}

				break;
			}
		}
	}
	
Exit:

	return( rc);
}
#endif

/********************************************************************
Desc: Startup routine for the NLM - that gets the main going in
		its own thread.
*********************************************************************/
#if defined( FLM_RING_ZERO_NLM)
extern "C" void * f_nlmMainStub(
	void *		hThread,
	void *		pData)
{
	ARG_DATA *									pArgData = (ARG_DATA *)pData;
	struct LoadDefinitionStructure *		moduleHandle = pArgData->moduleHandle;

	(void)hThread;

	(void)kSetThreadName( (void *)kCurrentThread(),
		(BYTE *)pArgData->pszThreadName);

	nlm_main( pArgData->iArgC, pArgData->ppszArgV);

	Free( pArgData->ppszArgV);
	Free( pArgData->pszArgs);
	Free( pArgData->pszThreadName);
	Free( pArgData);

	gv_bMainRunning = FALSE;

	if( !gv_bUnloadCalled)
	{
		KillMe( moduleHandle);
	}
	
	kExitThread( NULL);
	return( NULL);
}
#endif
	
/********************************************************************
Desc: Signals the f_nlmEntryPoint thread to release the console.
*********************************************************************/
#if defined( FLM_RING_ZERO_NLM)
void SynchronizeStart( void)
{
	if (gv_lFlmSyncSem)
	{
		(void)kSemaphoreSignal( gv_lFlmSyncSem);
	}
}
#endif

/********************************************************************
Desc: Startup routine for the NLM.
*********************************************************************/
#if defined( FLM_RING_ZERO_NLM)
extern "C" LONG f_nlmEntryPoint(
	struct LoadDefinitionStructure *		moduleHandle,
	struct ScreenStruct *					initScreen,
	char *										commandLine,
	char *										loadDirectoryPath,
	LONG											uninitializedDataLength,
	LONG											fileHandle,
	LONG											(*ReadRoutine)
														(LONG		handle,
														 LONG		offset,
														 char *	buffer,
														 LONG		length),
	LONG											customDataOffset,
	LONG											customDataSize)
{
	char *		pszTmp;
	char *		pszArgStart;
	int			iArgC;
	int			iTotalArgChars;
	int			iArgSize;
	char **		ppszArgV = NULL;
	char *		pszArgs = NULL;
	char *		pszDestArg;
	bool			bFirstPass = true;
	char			cEnd;
	ARG_DATA *	pArgData = NULL;
	LONG			sdRet = 0;
	char *		pszThreadName;
	char *		pszModuleName;
	int			iModuleNameLen;
	int			iThreadNameLen;
	int			iLoadDirPathSize;
	void *		hThread = NULL;
	
	(void)initScreen;
	(void)uninitializedDataLength;
	(void)fileHandle;
	(void)ReadRoutine;
	(void)customDataOffset;
	(void)customDataSize;

	if( f_atomicInc( &gv_NetWareStartupCount) != 1)
	{
		goto Exit;
	}
	
	gv_MyModuleHandle = moduleHandle;
	gv_bUnloadCalled = FALSE;

	// Allocate the needed resource tags
	
	if( (gv_lAllocRTag = AllocateResourceTag( gv_MyModuleHandle,
		"FLAIM Memory", AllocSignature)) == NULL)
	{
		sdRet = 1;
		goto Exit;
	}

	// Syncronized start

	if (moduleHandle->LDFlags & 4)
	{
		gv_lFlmSyncSem = kSemaphoreAlloc( (BYTE *)"FLAIM_SYNC", 0);
	}

	// Initialize NSS
	
	if( RC_BAD( f_nssInitialize()))
	{
		sdRet = 1;
		goto Exit;
	}
	
	pszModuleName = (char *)(&moduleHandle->LDFileName[ 1]);
	iModuleNameLen = (int)(moduleHandle->LDFileName[ 0]);
	
	// First pass: Count the arguments in the command line
	// and determine how big of a buffer we will need.
	// Second pass: Put argments into allocated buffer.

Parse_Args:

	iTotalArgChars = 0;
	iArgC = 0;
	
	iLoadDirPathSize = f_strlen( (const char *)loadDirectoryPath); 
	iArgSize =  iLoadDirPathSize + iModuleNameLen;
	
	if( !bFirstPass)
	{
		ppszArgV[ iArgC] = pszDestArg;
		f_memcpy( pszDestArg, loadDirectoryPath, iLoadDirPathSize);
		f_memcpy( &pszDestArg[ iLoadDirPathSize], pszModuleName, iModuleNameLen);
		pszDestArg[ iArgSize] = 0;
		pszDestArg += (iArgSize + 1);
	}

	iArgC++;
	iTotalArgChars += iArgSize;
	pszTmp = commandLine;

	for (;;)
	{
		// Skip leading blanks.

		while( *pszTmp && *pszTmp == ' ')
		{
			pszTmp++;
		}

		if( *pszTmp == 0)
		{
			break;
		}

		if( *pszTmp == '"' || *pszTmp == '\'')
		{
			cEnd = *pszTmp;
			pszTmp++;
		}
		else
		{
			cEnd = ' ';
		}
		
		pszArgStart = pszTmp;
		iArgSize = 0;

		// Count the characters in the parameter.

		while( *pszTmp && *pszTmp != cEnd)
		{
			iArgSize++;
			pszTmp++;
		}

		if( !iArgSize && cEnd == ' ')
		{
			break;
		}

		// If 2nd pass, save the argument.

		if( !bFirstPass)
		{
			ppszArgV[ iArgC] = pszDestArg;
			
			if( iArgSize)
			{
				f_memcpy( pszDestArg, pszArgStart, iArgSize);
			}
			
			pszDestArg[ iArgSize] = 0;
			pszDestArg += (iArgSize + 1);
		}

		iArgC++;
		iTotalArgChars += iArgSize;

		// Skip trailing quote or blank.

		if( *pszTmp)
		{
			pszTmp++;
		}
	}

	if( bFirstPass)
	{
		if ((ppszArgV = (char **)Alloc(  sizeof( char *) * iArgC, 
			gv_lAllocRTag)) == NULL)
		{
			sdRet = 1;
			goto Exit;
		}

		if( (pszArgs = (char *)Alloc( iTotalArgChars + iArgC, 
			gv_lAllocRTag)) == NULL)
		{
			sdRet = 1;
			goto Exit;
		}
		
		pszDestArg = pszArgs;
		bFirstPass = false;
		goto Parse_Args;
	}

	iThreadNameLen = (int)(moduleHandle->LDName[ 0]);

	if( (pszThreadName = (char *)Alloc( iThreadNameLen + 1, gv_lAllocRTag)) == NULL)
	{
		sdRet = 1;
		goto Exit;
	}
	
	f_memcpy( pszThreadName, (char *)(&moduleHandle->LDName[ 1]), iThreadNameLen);
	pszThreadName[ iThreadNameLen] = 0;

	if( (pArgData = (ARG_DATA *)Alloc( sizeof( ARG_DATA), 
		gv_lAllocRTag)) == NULL)
	{
		sdRet = 1;
		goto Exit;
	}
	
	pArgData->ppszArgV = ppszArgV;
	pArgData->pszArgs = pszArgs;
	pArgData->iArgC = iArgC;
	pArgData->moduleHandle = moduleHandle;
	pArgData->pszThreadName = pszThreadName;

	gv_bMainRunning = TRUE;

	if( (hThread = kCreateThread( (BYTE *)"FTK main",
			f_nlmMainStub, NULL, 32768, (void *)pArgData)) == NULL)
	{
		gv_bMainRunning = FALSE;
		sdRet = 2;
		goto Exit;
	}

	if( kSetThreadLoadHandle( hThread, (LONG)moduleHandle) != 0)
	{
		(void)kDestroyThread( hThread);
		gv_bMainRunning = FALSE;
		sdRet = 2;
		goto Exit;
	}
			
	if( kScheduleThread( hThread) != 0)
	{
		(void)kDestroyThread( hThread);
		gv_bMainRunning = FALSE;
		sdRet = 2;
		goto Exit;
	}
	
	// Synchronized start

	if( moduleHandle->LDFlags & 4)
	{
		(void)kSemaphoreWait( gv_lFlmSyncSem);
	}

Exit:

	if( sdRet != 0)
	{
		f_atomicDec( &gv_NetWareStartupCount);
		
		if( ppszArgV)
		{
			Free( ppszArgV);
		}

		if( pszArgs)
		{
			Free( pszArgs);
		}

		if( pszThreadName)
		{
			Free( pszThreadName);
		}

		if( pArgData)
		{
			Free( pArgData);
		}

		if( gv_lFlmSyncSem)
		{
			kSemaphoreFree( gv_lFlmSyncSem);
			gv_lFlmSyncSem = 0;
		}
		
		if( !gv_bUnloadCalled)
		{
			KillMe( moduleHandle);
		}
	}

	return( sdRet);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
extern "C" void f_nlmExitPoint(void)
{
	if( f_atomicDec( &gv_NetWareStartupCount) > 0)
	{
		return;
	}
	
	gv_bUnloadCalled = TRUE;

	if( gv_fnExit)
	{
		(*gv_fnExit)();
		gv_fnExit = NULL;
	}

	while( gv_bMainRunning)
	{
		kYieldThread();
	}

	f_nssUninitialize();
	
	if( gv_lFlmSyncSem)
	{
		kSemaphoreFree( gv_lFlmSyncSem);
		gv_lFlmSyncSem = 0;
	}

	if( gv_lAllocRTag)
	{
		ReturnResourceTag( gv_lAllocRTag, 1);
		gv_lAllocRTag = 0;
	}
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
extern "C" void exit(
	int		exitCode)
{
	(void)exitCode;
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
extern "C" int atexit(
	F_EXIT_FUNC		fnExit)
{
	gv_fnExit = fnExit;
	return( 0);
}
#endif
	
/****************************************************************************
Desc:
****************************************************************************/
#if !defined( FLM_NLM)
int ftknlmDummy( void)
{
	return( 0);
}
#endif

