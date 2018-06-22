//-------------------------------------------------------------------------
// Desc:	RFL viewer utility - definitions.
// Tabs:	3
//
// Copyright (c) 1998-2007 Novell, Inc. All Rights Reserved.
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

#include "flaim.h"
#include "flaimsys.h"

#ifndef RFLREAD_H
#define RFLREAD_H

extern "C"
{

#ifdef MAIN_MODULE
	#define REXTERN
#else
	#define REXTERN	extern
#endif

#ifndef RFL_BUFFER_SIZE
#define	RFL_BUFFER_SIZE						(65536 * 4)
#endif

REXTERN	IF_FileHdl *	gv_pRflFileHdl;
REXTERN	FLMBYTE		gv_rflBuffer [RFL_BUFFER_SIZE];
REXTERN	FLMUINT64		gv_ui64RflEof;

// Tag numbers for internal fields.

typedef enum
{
	RFL_TRNS_BEGIN_FIELD = 0,
	RFL_TRNS_COMMIT_FIELD,
	RFL_TRNS_ABORT_FIELD,
	RFL_RECORD_ADD_FIELD,
	RFL_RECORD_MODIFY_FIELD,
	RFL_RECORD_DELETE_FIELD,
	RFL_RESERVE_DRN_FIELD,
	RFL_CHANGE_FIELDS_FIELD,
	RFL_DATA_RECORD_FIELD,
	RFL_UNKNOWN_PACKET_FIELD,
	RFL_NUM_BYTES_VALID_FIELD,
	RFL_PACKET_ADDRESS_FIELD,
	RFL_PACKET_CHECKSUM_FIELD,
	RFL_PACKET_CHECKSUM_VALID_FIELD,
	RFL_PACKET_BODY_LENGTH_FIELD,
	RFL_NEXT_PACKET_ADDRESS_FIELD,
	RFL_PREV_PACKET_ADDRESS_FIELD,
	RFL_TRANS_ID_FIELD,
	RFL_START_SECONDS_FIELD,
	RFL_START_MSEC_FIELD,
	RFL_END_SECONDS_FIELD,
	RFL_END_MSEC_FIELD,
	RFL_START_TRNS_ADDR_FIELD,
	RFL_CONTAINER_FIELD,
	RFL_DRN_FIELD,
	RFL_TAG_NUM_FIELD,
	RFL_TYPE_FIELD,
	RFL_LEVEL_FIELD,
	RFL_DATA_LEN_FIELD,
	RFL_DATA_FIELD,
	RFL_MORE_DATA_FIELD,
	RFL_INSERT_FLD_FIELD,
	RFL_MODIFY_FLD_FIELD,
	RFL_DELETE_FLD_FIELD,
	RFL_END_CHANGES_FIELD,
	RFL_UNKNOWN_CHANGE_TYPE_FIELD,
	RFL_POSITION_FIELD,
	RFL_REPLACE_BYTES_FIELD,
	RFL_UNKNOWN_CHANGE_BYTES_FIELD,
	RFL_INDEX_SET_FIELD,
	RFL_INDEX_NUM_FIELD,
	RFL_START_DRN_FIELD,
	RFL_END_DRN_FIELD,
	RFL_START_UNKNOWN_FIELD,
	RFL_UNKNOWN_USER_PACKET_FIELD,
	RFL_HDR_NAME_FIELD,
	RFL_HDR_VERSION_FIELD,
	RFL_HDR_FILE_NUMBER_FIELD,
	RFL_HDR_EOF_FIELD,
	RFL_HDR_DB_SERIAL_NUM_FIELD,
	RFL_HDR_FILE_SERIAL_NUM_FIELD,
	RFL_HDR_NEXT_FILE_SERIAL_NUM_FIELD,
	RFL_HDR_KEEP_SIGNATURE_FIELD,
	RFL_TRNS_BEGIN_EX_FIELD,
	RFL_UPGRADE_PACKET_FIELD,
	RFL_OLD_DB_VERSION_FIELD,
	RFL_NEW_DB_VERSION_FIELD,
	RFL_REDUCE_PACKET_FIELD,
	RFL_BLOCK_COUNT_FIELD,
	RFL_LAST_COMMITTED_TRANS_ID_FIELD,
	RFL_INDEX_SET2_FIELD,
	RFL_INDEX_SUSPEND_FIELD,
	RFL_INDEX_RESUME_FIELD,
	RFL_BLK_CHAIN_FREE_FIELD,
	RFL_TRACKER_REC_FIELD,
	RFL_END_BLK_ADDR_FIELD,
	RFL_FLAGS_FIELD,
	RFL_INSERT_ENC_FLD_FIELD,
	RFL_MODIFY_ENC_FLD_FIELD,
	RFL_ENC_FIELD,
	RFL_ENC_DEF_ID_FIELD,
	RFL_ENC_DATA_LEN_FIELD,
	RFL_DB_KEY_LEN_FIELD,
	RFL_DB_KEY_FIELD,
	RFL_WRAP_KEY_FIELD,
	RFL_ENABLE_ENCRYPTION_FIELD,
	RFL_ENC_DATA_RECORD_FIELD,
	RFL_DATA_RECORD3_FIELD,
	RFL_INSERT_LARGE_FLD_FIELD,
	RFL_INSERT_ENC_LARGE_FLD_FIELD,
	RFL_MODIFY_LARGE_FLD_FIELD,
	RFL_MODIFY_ENC_LARGE_FLD_FIELD,
	RFL_CONFIG_SIZE_EVENT_FIELD,
	RFL_SIZE_THRESHOLD_FIELD,
	RFL_TIME_INTERVAL_FIELD,
	RFL_SIZE_INTERVAL_FIELD,
	
	// IMPORTANT NOTE: WHEN ADDING TO THIS TABLE, BE SURE TO ADD
	// STRING TO gv_szTagNames TABLE BELOW - STRINGS MUST BE
	// IN THE ORDER THEY APPEAR IN THIS TABLE
	
	RFL_TOTAL_FIELDS
} eDispTag;

FINLINE FLMUINT makeTagNum(
	eDispTag	eTagNum)
{
	return( (FLMUINT)eTagNum + 32769);
}

#ifdef MAIN_MODULE

// Put this table in the header file so that the enums
// are right next to the tags that go with them.

const char * gv_szTagNames [] =
{
	"TransBegin",					// RFL_TRNS_BEGIN_FIELD
	"TransCommit",					// RFL_TRNS_COMMIT_FIELD
	"TransAbort",					// RFL_TRNS_ABORT_FIELD
	"RecAdd",						// RFL_RECORD_ADD_FIELD
	"RecModify",					// RFL_RECORD_MODIFY_FIELD
	"RecDelete",					// RFL_RECORD_DELETE_FIELD
	"ReserveDRN",					// RFL_RESERVE_DRN_FIELD
	"ChangeFields",				// RFL_CHANGE_FIELDS_FIELD
	"DataRecord",					// RFL_DATA_RECORD_FIELD
	"UnknownPacket",				// RFL_UNKNOWN_PACKET_FIELD
	"NumBytesValid",				// RFL_NUM_BYTES_VALID_FIELD
	"PacketAddress",				// RFL_PACKET_ADDRESS_FIELD
	"PacketChecksum",				// RFL_PACKET_CHECKSUM_FIELD
	"PacketChecksumValid",		// RFL_PACKET_CHECKSUM_VALID_FIELD
	"PacketBodyLength",			// RFL_PACKET_BODY_LENGTH_FIELD
	"NextPacketAddress",			// RFL_NEXT_PACKET_ADDRESS_FIELD
	"PrevPacketAddress",			// RFL_PREV_PACKET_ADDRESS_FIELD
	"TransID",						// RFL_TRANS_ID_FIELD
	"StartSeconds",				// RFL_START_SECONDS_FIELD
	"StartMillisec",				// RFL_START_MSEC_FIELD
	"EndSeconds",					// RFL_END_SECONDS_FIELD
	"EndMillisec",					// RFL_END_MSEC_FIELD
	"StartTransAddr",				// RFL_START_TRNS_ADDR_FIELD
	"ContainerNum",				// RFL_CONTAINER_FIELD
	"RecordID",						// RFL_DRN_FIELD
	"TagNum",						// RFL_TAG_NUM_FIELD
	"FieldType",					// RFL_TYPE_FIELD
	"Level",							// RFL_LEVEL_FIELD
	"DataLen",						// RFL_DATA_LEN_FIELD
	"Data",							// RFL_DATA_FIELD
	"MoreData",						// RFL_MORE_DATA_FIELD
	"InsertFld",					// RFL_INSERT_FLD_FIELD
	"ModifyFld",					// RFL_MODIFY_FLD_FIELD
	"DeleteFld",					// RFL_DELETE_FLD_FIELD
	"EndChanges",					// RFL_END_CHANGES_FIELD
	"UnknownChangeType",			// RFL_UNKNOWN_CHANGE_TYPE_FIELD
	"Position",						// RFL_POSITION_FIELD
	"ReplaceBytes",				// RFL_REPLACE_BYTES_FIELD
	"UnknownChangeBytes",		// RFL_UNKNOWN_CHANGE_BYTES_FIELD
	"IndexSet",						// RFL_INDEX_SET_FIELD
	"IndexNum",						// RFL_INDEX_NUM_FIELD
	"StartDRN",						// RFL_START_DRN_FIELD
	"EndDRN",						// RFL_END_DRN_FIELD
	"StartUnknown",				// RFL_START_UNKNOWN_FIELD
	"UserUnknown",					// RFL_UNKNOWN_USER_PACKET_FIELD
	"RFLName",						// RFL_HDR_NAME_FIELD
	"RFLVersion",					// RFL_HDR_VERSION_FIELD
	"FileNumber",					// RFL_HDR_FILE_NUMBER_FIELD
	"FileEOF",						// RFL_HDR_EOF_FIELD
	"DBSerialNum",					// RFL_HDR_DB_SERIAL_NUM_FIELD
	"FileSerialNum",				// RFL_HDR_FILE_SERIAL_NUM_FIELD
	"NextFileSerialNum",			// RFL_HDR_NEXT_FILE_SERIAL_NUM_FIELD
	"KeepSignature",				// RFL_HDR_KEEP_SIGNATURE_FIELD
	"TransBeginEx",				// RFL_TRNS_BEGIN_EX_FIELD
	"UpgradeDB",					// RFL_UPGRADE_PACKET_FIELD
	"OldDbVersion",				// RFL_OLD_DB_VERSION_FIELD
	"NewDbVersion",				// RFL_NEW_DB_VERSION_FIELD
	"ReduceDb",						// RFL_REDUCE_PACKET_FIELD
	"BlockCount",					// RFL_BLOCK_COUNT_FIELD
	"LastCommitTransID",			// RFL_LAST_COMMITTED_TRANS_ID_FIELD
	"IndexSet2",					// RFL_INDEX_SET2_FIELD
	"IndexSuspend",				// RFL_INDEX_SUSPEND_FIELD
	"IndexResume",					// RFL_INDEX_RESUME_FIELD
	"BlockChainFree",				// RFL_BLK_CHAIN_FREE_FIELD
	"TrackerRecDRN",				// RFL_TRACKER_REC_FIELD
	"EndBlockAddr",				// RFL_END_BLK_ADDR_FIELD
	"Flags",							// RFL_FLAGS_FIELD
	"InsertEncFld",				// RFL_INSERT_ENC_FLD_FIELD
	"ModifyEncFld",				// RFL_MODIFY_ENC_FLD_FIELD
	"Encrypted",					// RFL_ENC_FIELD
	"EncryptedDefId",				// RFL_ENC_DEF_ID_FIELD
	"EncryptedDataLen",			// RFL_ENC_DATA_LEN_FIELD
	"DataBaseKeyLen",				// RFL_DB_KEY_LEN_FIELD
	"DataBaseKey",					// RFL_DB_KEY_FIELD
	"WrapKey",						// RFL_WRAP_KEY_FIELD
	"EnableEncryption",			// RFL_ENABLE_ENCRYPTION_FIELD
	"EncDataRecord",				// RFL_ENC_DATA_RECORD_FIELD
	"DataRecord3",					// RFL_DATA_RECORD3_FIELD
	"InsertLargeFld",				// RFL_INSERT_LARGE_FLD_FIELD
	"InsertLargeEncFld",			// RFL_INSERT_ENC_LARGE_FLD_FIELD
	"ModifyLargeFld",				// RFL_MODIFY_LARGE_FLD_FIELD
	"ModifyLargeEncFld",			// RFL_MODIFY_ENC_LARGE_FLD_FIELD
	"ConfigSizeEvent",			// RFL_CONFIG_SIZE_EVENT_FIELD
	"SizeThreshold",				// RFL_SIZE_THRESHOLD_FIELD
	"TimeInterval",				// RFL_TIME_INTERVAL_FIELD
	"SizeInterval",				// RFL_SIZE_INTERVAL_FIELD
	NULL
};
#endif


typedef struct Rfl_Packet
{
	FLMUINT	uiFileOffset;					// File offset this packet was read from.
	FLMUINT	uiPacketAddress;				// Packet address that was read
	FLMUINT	uiPacketAddressBytes;		// Bytes that were actually in packet addr.
	FLMUINT	uiPacketChecksum;				// Packet checksum
	FLMBOOL	bHavePacketChecksum;			// Did we actually have a packet checksum?
	FLMBOOL	bValidChecksum;				// Is the checksum valid?
	FLMUINT	uiPacketType;					// Packet type
	FLMBOOL	bHavePacketType;				// Did we actually have a packet type?
	FLMBOOL	bValidPacketType;				// Is the packet type valid?
	FLMBOOL	bHaveTimes;						// Was the time bit set on the packet type?
	FLMUINT	uiPacketBodyLength;			// Packet body length
	FLMUINT	uiPacketBodyLengthBytes;	// Bytes that were in packet body length
	FLMUINT	uiNextPacketAddress;			// Next packet address - zero if no more
	FLMUINT	uiPrevPacketAddress;			// Prev packet address - zero if unknown
	FLMUINT	uiTransID;						// Transaction ID
	FLMUINT	uiTransIDBytes;				// Bytes that were actually in transID
	FLMUINT	uiTransStartAddr;				// Transaction start address
	FLMUINT	uiTransStartAddrBytes;		// Transaction start address bytes
	FLMUINT	uiContainer;					// container
	FLMUINT	uiContainerBytes;				// Bytes that were in container.
	FLMUINT	uiIndex;							// index
	FLMUINT	uiIndexBytes;					// Bytes that were in index.
	FLMUINT	uiDrn;							// DRN
	FLMUINT	uiDrnBytes;						// Bytes that were in DRN.
	FLMUINT	uiEndDrn;						// End DRN
	FLMUINT	uiEndDrnBytes;					// Bytes that were in End DRN.
	FLMUINT	uiStartSeconds;				// Start seconds
	FLMUINT	uiStartSecondsBytes;			// Bytes that were in start seconds
	FLMUINT	uiStartMicro;					// Start micro seconds
	FLMUINT	uiStartMicroBytes;			// Bytes that were in start micro seconds
	FLMUINT	uiEndSeconds;					// End seconds
	FLMUINT	uiEndSecondsBytes;			// Bytes that were in end seconds
	FLMUINT	uiEndMicro;						// End micro seconds
	FLMUINT	uiEndMicroBytes;				// Bytes that were in end micro seconds
	FLMUINT	uiLastCommittedTransID;		// Last committed transaction ID
	FLMUINT	uiLastCommittedTransIDBytes; // Bytes that were in the last committed trans ID
	FLMUINT	uiFlags;							// Operation flags
	FLMUINT	uiFlagsBytes;					// Bytes that were in flags
	FLMUINT	uiCount;							// Count (number of blocks, etc.)
	FLMUINT	uiCountBytes;					// Bytes that were in count
	FLMUINT	uiSizeThreshold;				// Size threshhold
	FLMUINT	uiSizeThresholdBytes;		// Bytes in size threshhold
	FLMUINT	uiTimeInterval;				// Time interval
	FLMUINT	uiTimeIntervalBytes;			// Bytes that were in tine interval
	FLMUINT	uiSizeInterval;				// Size interval
	FLMUINT	uiSizeIntervalBytes;			// Bytes that were in size interval
	FLMUINT	uiMultiFileSearch;			// Is search to span multiple files?
} RFL_PACKET, * RFL_PACKET_p;

RCODE RflPositionToNode(
	FLMUINT		uiFileOffset,
	FLMBOOL		bOperationsOnly,
	F_Pool *		pPool,
	NODE **		ppNodeRV);

RCODE RflGetNextNode(
	NODE *		pCurrOpNode,
	FLMBOOL		bOperationsOnly,
	F_Pool *		pPool,
	NODE **		ppNextNodeRV,
	FLMBOOL		bStopAtEOF = FALSE);

RCODE RflGetPrevNode(
	NODE *		pCurrOpNode,
	FLMBOOL		bOperationsOnly,
	F_Pool *		pPool,
	NODE **		ppPrevNodeRV);

void RflFormatPacket(
	void *			pPacket,
	char *			pszDispBuffer);

RCODE RflExpandPacket(
	NODE *		pPacketNode,
	F_Pool *		pPool,
	NODE **		ppForest);

}	// extern "C"

#endif
