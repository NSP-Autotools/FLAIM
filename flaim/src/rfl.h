//-------------------------------------------------------------------------
// Desc:	Routines for roll-forward logging - definitions.
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

#ifndef RFL_H
#define RFL_H

#include "fpackon.h"

// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

// Packet types for roll forward logging

#define RFL_TRNS_BEGIN_PACKET					1
#define RFL_TRNS_COMMIT_PACKET				2
#define RFL_TRNS_ABORT_PACKET					3
#define RFL_ADD_RECORD_PACKET					4
#define RFL_MODIFY_RECORD_PACKET				5
#define RFL_DELETE_RECORD_PACKET				6
#define RFL_RESERVE_DRN_PACKET				7
#define RFL_CHANGE_FIELDS_PACKET				8
#define RFL_DATA_RECORD_PACKET				9
#define RFL_INDEX_SET_PACKET					10
#define RFL_START_UNKNOWN_PACKET				11
#define RFL_UNKNOWN_PACKET						12
#define RFL_REDUCE_PACKET						13
#define RFL_TRNS_BEGIN_EX_PACKET				14
#define RFL_UPGRADE_PACKET						15
#define RFL_INDEX_SET_PACKET_VER_2			16
#define RFL_INDEX_SUSPEND_PACKET				17
#define RFL_INDEX_RESUME_PACKET				18
#define RFL_ADD_RECORD_PACKET_VER_2			19
#define RFL_MODIFY_RECORD_PACKET_VER_2		20
#define RFL_DELETE_RECORD_PACKET_VER_2		21
#define RFL_BLK_CHAIN_FREE_PACKET			22
#define RFL_ENC_DATA_RECORD_PACKET			23
#define RFL_DATA_RECORD_PACKET_VER_3		24
#define RFL_WRAP_KEY_PACKET					25
#define RFL_ENABLE_ENCRYPTION_PACKET		26
#define RFL_CONFIG_SIZE_EVENT_PACKET		27
#define RFL_TIME_LOGGED_FLAG					0x80
#define RFL_PACKET_TYPE_MASK					0x7F

#define RFL_GET_PACKET_TYPE(uiPacketType)	\
	((FLMUINT)((uiPacketType) & RFL_PACKET_TYPE_MASK))

// Change types for RFL_CHANGE_FIELDS_PACKET.

#define RFL_INSERT_FIELD				1
#define RFL_DELETE_FIELD				2
#define RFL_MODIFY_FIELD				3
#define RFL_END_FIELD_CHANGES			4
#define RFL_INSERT_ENC_FIELD			5
#define RFL_MODIFY_ENC_FIELD			6
#define RFL_INSERT_LARGE_FIELD		7
#define RFL_INSERT_ENC_LARGE_FIELD	8
#define RFL_MODIFY_LARGE_FIELD		9
#define RFL_MODIFY_ENC_LARGE_FIELD	10

// Flags for add, delete, and modify packets
// The flags need to fit in a single byte

#define RFL_UPDATE_BACKGROUND		0x01
#define RFL_UPDATE_SUSPENDED		0x02

// Mini-change types for the RFL_MODIFY_FIELD change type

#define RFL_REPLACE_BYTES			1
#define RFL_INSERT_BYTES			2
#define RFL_DELETE_BYTES			3

// Definitions for ROLL FORWARD LOG file header format

// The following are so we can maintain pre-4.3 databases.
// These are only put into databases prior to 4.3

#define	RFL_NAME_POS		  		0
#define	RFL_NAME						"RFL3"
#define	RFL_NAME_LEN		  		4
#define	RFL_VERSION_POS	  		RFL_NAME_LEN
#define	RFL_VERSION		  			"1.00"
#define	RFL_VERSION_LEN		  	4
#define	RFL_FILE_NUMBER_POS		8
#define	RFL_EOF_POS					12

// The following are new items for 4.3 and greater

#define	RFL_DB_SERIAL_NUM_POS			16
#define	RFL_SERIAL_NUM_POS				(RFL_DB_SERIAL_NUM_POS + F_SERIAL_NUM_SIZE)
#define	RFL_NEXT_FILE_SERIAL_NUM_POS	(RFL_SERIAL_NUM_POS + F_SERIAL_NUM_SIZE)
#define	RFL_KEEP_SIGNATURE_POS			(RFL_NEXT_FILE_SERIAL_NUM_POS + F_SERIAL_NUM_SIZE)

#define	RFL_KEEP_SIGNATURE	"----KeepLog----"
#define	RFL_NOKEEP_SIGNATURE	"--DontKeepLog--"

// Buffer size needs to be a multiple of 512 for direct IO writes.

#define	DEFAULT_RFL_WRITE_BUFFERS		4
#define	DEFAULT_RFL_BUFFER_SIZE			(65536)

// Definitions for packet format and sizes.

#define	RFL_PACKET_ADDRESS_OFFSET			0
#define	RFL_PACKET_CHECKSUM_OFFSET			4
#define	RFL_PACKET_TYPE_OFFSET				5
#define	RFL_PACKET_BODY_LENGTH_OFFSET		6
#define	RFL_PACKET_OVERHEAD					8

// Direct IO requires that we always write on 512 byte boundaries.
// This means that whenever we write out a packet, we may also
// have to write out up to the last 511 bytes of the prior packet
// in order to be on a 512 byte boundary.  Thus, the buffer must
// be able to hold a full packet plus up to 512 bytes of the prior
// packet.

// NOTE: RFL_MAX_PACKET_BODY_SIZE should also be a multiple of 4 for
// encryption reasons.  Thus, it is ANDED with FFFC.

#define	RFL_MAX_PACKET_SIZE					(65536 - 1024)
#define	RFL_MAX_PACKET_BODY_SIZE			((RFL_MAX_PACKET_SIZE - \
														  RFL_PACKET_OVERHEAD) & 0xFFFC)

/****************************************************************************
Desc:
****************************************************************************/
typedef struct RFL_OP_INFO
{
	FLMUINT		uiPacketType;
	FLMUINT		uiContainer;
	FLMUINT		uiDrn;
	FLMUINT		uiIndex;
	FLMUINT		uiEndDrn;
	FLMUINT		uiTransId;
	FLMUINT		uiStartTime;
	FLMUINT		uiLastLoggedCommitTransId;
	FLMUINT		uiFlags;
	FLMUINT		uiCount;
	FLMUINT		uiEndBlock;
	FLMUINT		uiOldVersion;
	FLMUINT		uiNewVersion;
	FLMUINT		uiSizeThreshold;
	FLMUINT		uiSizeInterval;
	FLMUINT		uiTimeInterval;
} RFL_OP_INFO;
														  
/****************************************************************************
Desc:
****************************************************************************/
typedef struct RFL_WAITER
{
	FLMUINT			uiThreadId;
	FLMBOOL			bIsWriter;
	F_SEM				hESem;
	RCODE *			pRc;
	RFL_WAITER *	pNext;
} RFL_WAITER;

/****************************************************************************
Desc:
****************************************************************************/
typedef struct RFL_BUFFER
{
	IF_IOBufferMgr *	pBufferMgr;				// Write buffer manager
	IF_IOBuffer *		pIOBuffer;
	FLMUINT				uiCurrFileNum;			// Current file number.
	FLMUINT				uiRflBufBytes;			// Number of bytes currently in the
														//	pIOBuffer.  Always points to
														// where the last packet ends when
														// writing to the log file.
	FLMUINT				uiRflFileOffset;		// Current offset in file that the
														//	zeroeth byte in the buffer
														//	represents.
	FLMBOOL				bTransInProgress;		// Transaction in progress using
														// these buffers.
	FLMBOOL				bOkToWriteHdrs;		// Is it OK to update the DB
														// headers with this information
	FLMBYTE				ucLogHdr [LOG_HEADER_SIZE];
														// Log header to be written with
														// this buffer.
	FLMBYTE				ucCPHdr [LOG_HEADER_SIZE];
														// Checkpoint header for this
														// buffer.
	RFL_WAITER *		pFirstWaiter;
	RFL_WAITER *		pLastWaiter;
} RFL_BUFFER;

/**************************************************************************
Desc:	This class handles all of the roll-forward logging for FLAIM.  There
		is one of these objects allocated per FFILE.
**************************************************************************/
class F_Rfl : public F_Object
{
public:

	F_Rfl();
	
	virtual ~F_Rfl();

	// Setup for logging - should only be called when
	// database is first opened or created.

	RCODE setup(
		FFILE *				pFile,
		const char *		pszRflDir);

	RCODE finishCurrFile(
		FDB *					pDb,
		FLMBOOL				bNewKeepState);

	// Log transaction begin

	RCODE logBeginTransaction(
		FDB *					pDb);

	// Log transaction commit or abort

	RCODE logEndTransaction(
		FLMUINT			uiPacketType,
		FLMBOOL			bThrowLogAway,
		FLMBOOL *		pbLoggedTransEnd = NULL);

	// Used to log reserve DRN.

	RCODE logUpdatePacket(
		FLMUINT			uiPacketType,
		FLMUINT			uiContainer,
		FLMUINT			uiDrn,
		FLMUINT			uiFlags);

	// Log add, modify, delete

	RCODE logUpdate(
		FLMUINT			uiContainer,
		FLMUINT			uiDrn,
		FLMUINT			uiAutoTrans,
		FlmRecord *		pOldRecord,
		FlmRecord *		pNewRecord);

	// Log index set

	RCODE logIndexSet(
		FLMUINT			uiIndex,
		FLMUINT			uiContainerNum,
		FLMUINT			uiStartDrn,
		FLMUINT			uiEndDrn);

	// Routines for logging unknown packets.

	RCODE startLoggingUnknown( void);

	RCODE logUnknown(
		FLMBYTE *		pucUnknown,
		FLMUINT			uiLen);

	RCODE endLoggingUnknown( void);

	// Routine for logging reduce operation

	RCODE logReduce(
		FLMUINT			uiTransId,
		FLMUINT			uiCount);

	// Routine for logging a block chain delete operation

	RCODE logBlockChainFree(
		FLMUINT			uiTrackerDrn,
		FLMUINT			uiCount,
		FLMUINT			uiEndAddr);

	// Routine for logging index suspend and resume operations

	RCODE logIndexSuspendOrResume(
		FLMUINT			uiIndexNum,
		FLMUINT			uiPacketType);

	// Routine for logging database upgrade

	RCODE logUpgrade(
		FLMUINT			uiTransID,
		FLMUINT			uiOldVersion,
		FLMBYTE *		pucDBKey,
		FLMUINT32		ui32DBKeyLen);
		
	// Routine for logging size event configuration
		
	RCODE logSizeEventConfig(
		FLMUINT		uiTransID,
		FLMUINT		uiSizeThreshold,
		FLMUINT		uiSecondsBetweenEvents,
		FLMUINT		uiBytesBetweenEvents);

	// Routines for restore operations.

	RCODE readUnknown(
		FLMUINT			uiLenToRead,
		FLMBYTE *		pucBuffer,
		FLMUINT *		puiBytesRead);

	RCODE recover(
		FDB *				pDb,
		F_Restore *		pRestore);

	FLMBOOL atEndOfLog( void);

	// Returns full log file name associated with number.
	// Will return the full path name.

	RCODE getFullRflFileName(
		FLMUINT			uiFileNum,
		char *			pszFullRflFileName);

	// Returns base log file name associated with number.

	void getBaseRflFileName(
		FLMUINT			uiFileNum,
		char *			pszBaseRflFileName);

	// Set the RFL directory.  Passing in a NULL or empty
	// string will cause the directory to be to the same
	// directory where the database is located.

	RCODE setRflDir(
		const char *	pszRflDir);

	FINLINE const char * getRflDirPtr( void)
	{
		return( m_szRflDir);
	}
	
	FINLINE const char * getDbPrefixPtr( void)
	{
		return( m_szDbPrefix);
	}

	FINLINE FLMBOOL isRflDirSameAsDbDir( void)
	{
		return m_bRflDirSameAsDb;
	}

	FINLINE FLMUINT getCurrFileNum( void)
	{
		return m_pCurrentBuf->uiCurrFileNum;
	}

	FINLINE FLMUINT getCurrWriteOffset( void)
	{
		return( m_pCurrentBuf->uiRflFileOffset +
				  m_pCurrentBuf->uiRflBufBytes);
	}

	FINLINE FLMUINT getCurrReadOffset( void)
	{
		return( m_uiRflReadOffset + m_pCurrentBuf->uiRflFileOffset);
	}

	FINLINE FLMUINT getCurrPacketAddress( void)
	{
		return( m_uiPacketAddress);
	}

	FINLINE void getCurrSerialNum(
		FLMBYTE *	pucSerialNum)
	{
		f_memcpy( pucSerialNum, m_ucCurrSerialNum, F_SERIAL_NUM_SIZE);
	}

	FINLINE void setCurrSerialNum(
		FLMBYTE *	pucSerialNum)
	{
		f_memcpy( m_ucCurrSerialNum, pucSerialNum, F_SERIAL_NUM_SIZE);
	}

	FINLINE void getNextSerialNum(
		FLMBYTE *	pucSerialNum)
	{
		f_memcpy( pucSerialNum, m_ucNextSerialNum, F_SERIAL_NUM_SIZE);
	}

	FINLINE void setNextSerialNum(
		FLMBYTE *	pucSerialNum)
	{
		f_memcpy( m_ucNextSerialNum, pucSerialNum, F_SERIAL_NUM_SIZE);
	}

	FINLINE FLMUINT getCurrTransID( void)
	{
		return( m_uiCurrTransID);
	}

	FINLINE FLMBOOL loggingIsOff( void)
	{
		return( m_bLoggingOff);
	}

	FINLINE void setLoggingOffState(
		FLMBOOL	bLoggingOff)
	{
		m_bLoggingOff = bLoggingOff;
	}

	RCODE truncate(
		FLMUINT	uiTruncateSize);

	// Public functions, but only intended for internal use

	RCODE makeRoom(
		FLMUINT		uiAdditionalBytesNeeded,
		FLMUINT *	puiCurrPacketLenRV,
		FLMUINT		uiPacketType,
		FLMUINT *	puiBytesAvailableRV,
		FLMUINT *	puiPacketCountRV);

	FINLINE FLMBYTE * getPacketPtr( void)
	{
		return( &(m_pCurrentBuf->pIOBuffer->getBufferPtr()[
						m_pCurrentBuf->uiRflBufBytes]));
	}

	RCODE logData(
		FLMUINT				uiDataLen,
		const FLMBYTE *	pucData,
		FLMUINT				uiPacketType,
		FLMUINT *			puiPacketLenRV,
		FLMUINT *			puiPacketCountRV,
		FLMUINT *			puiMaxLogBytesNeededRV,
		FLMUINT *			puiTotalBytesLoggedRV);

	// Close the current RFL file

	FINLINE void closeFile( void)
	{
		if( m_pCurrentBuf->pBufferMgr)
		{
			flmAssert( !m_pCurrentBuf->pBufferMgr->isIOPending());
		}
		
		if (m_pFileHdl)
		{
			m_pFileHdl->Release();
			m_pFileHdl = NULL;
			m_pCurrentBuf->uiCurrFileNum = 0;
			m_pCurrentBuf->uiRflBufBytes = 0;
			m_pCurrentBuf->uiRflFileOffset = 0;
		}
	}

	FINLINE FLMBOOL seeIfRflVolumeOk( void)
	{
		return m_bRflVolumeOk;
	}

	FINLINE void setRflVolumeOk( void)
	{
		m_bRflVolumeOk = TRUE;
		m_bRflVolumeFull = FALSE;
	}

	FINLINE FLMBOOL isRflVolumeFull( void)
	{
		return m_bRflVolumeFull;
	}

	FINLINE RCODE waitPendingWrites( void)
	{
		if (m_uiRflWriteBufs > 1)
		{
			return( m_pCurrentBuf->pBufferMgr->waitForAllPendingIO());
		}
		else
		{
			return( FERR_OK);
		}
	}

	RCODE waitForWrites(
		RFL_BUFFER *	pBuffer,
		FLMBOOL			bIsWriter);

	RCODE waitForCommit( void);

	FINLINE void commitLogHdrs(
		FLMBYTE *	pucLogHdr,
		FLMBYTE *	pucCPHdr)
	{
		f_memcpy( m_pCurrentBuf->ucLogHdr, pucLogHdr, LOG_HEADER_SIZE);
		f_memcpy( m_pCurrentBuf->ucCPHdr, pucCPHdr, LOG_HEADER_SIZE);
		m_pCurrentBuf->bOkToWriteHdrs = TRUE;
	}

	FINLINE void clearLogHdrs( void)
	{
		m_pCurrentBuf->bOkToWriteHdrs = FALSE;
	}

	FLMBOOL seeIfRflWritesDone(
		FLMBOOL	bForceWait);

	void wakeUpWaiter(
		RCODE				rc,
		FLMBOOL			bIsWriter);

	RCODE completeTransWrites(
		FDB *				pDb,
		FLMBOOL			bCommitting,
		FLMBOOL			bOkToUnlock);

	RCODE logEnableEncryption(
		FLMUINT			uiTransID,
		FLMBYTE *		pucDBKey,
		FLMUINT32		ui32DBKeyLen);

	RCODE logWrappedKey(
		FLMUINT		uiTransID,
		FLMBYTE *	pucDBKey,
		FLMUINT32	ui32DBKeyLen);
		
	FINLINE const char * getRflDir( void)
	{
		return( m_szRflDir);
	}

private:

	FINLINE FLMUINT getEncryptPacketBodyLen(
		FLMUINT	uiPacketType,
		FLMUINT	uiPacketBodyLen)
	{
		if (uiPacketType == RFL_CHANGE_FIELDS_PACKET ||
			 uiPacketType == RFL_DATA_RECORD_PACKET ||
			 uiPacketType == RFL_ENC_DATA_RECORD_PACKET ||
			 uiPacketType == RFL_DATA_RECORD_PACKET_VER_3 ||
			 uiPacketType == RFL_UNKNOWN_PACKET)
		{
			if (uiPacketBodyLen & 0x03)
			{
				uiPacketBodyLen += (4 - (uiPacketBodyLen & 0x0003));
			}
		}
		return( uiPacketBodyLen);
	}

	FINLINE FLMBYTE * getPacketBodyPtr( void)
	{
		return( &(m_pCurrentBuf->pIOBuffer->getBufferPtr()[
			m_pCurrentBuf->uiRflBufBytes + RFL_PACKET_OVERHEAD]));
	}

	FINLINE FLMBOOL haveBuffSpace(
		FLMUINT	uiSpaceNeeded
		)
	{
		return( (FLMBOOL)((m_uiBufferSize - m_pCurrentBuf->uiRflBufBytes >=
										uiSpaceNeeded)
								? (FLMBOOL)TRUE
								: (FLMBOOL)FALSE) );
	}

	// Write the header of an RFL file.

	RCODE writeHeader(
		FLMUINT		uiFileNum,
		FLMUINT		uiEof,
		FLMBYTE *	pucSerialNum,
		FLMBYTE *	pucNextSerialNum,
		FLMBOOL		bKeepSignature);

	// Verify the header of an RFL file.

	RCODE verifyHeader(
		FLMBYTE *	pucHeader,
		FLMUINT		uiFileNum,
		FLMBYTE *	pucSerialNum);

	// Open a new RFL file.

	RCODE openFile(
		FLMUINT		uiFileNum,
		FLMBYTE *	pucSerialNum);

	// Create a new RFL file

	RCODE createFile(
		FLMUINT		uiFileNum,
		FLMBYTE *	pucSerialNum,
		FLMBYTE *	pucNextSerialNum,
		FLMBOOL		bKeepSignature);

	void copyLastBlock(
		RFL_BUFFER *	pBuffer,
		FLMBYTE *		pucOldBuffer,
		FLMBYTE *		pucNewBuffer,
		FLMUINT			uiCurrPacketLen,
		FLMBOOL			bStartingNewFile);

	// Position to an offset in the file.

	RCODE positionTo(
		FLMUINT		uiFileOffset);

	// Flush data to the current RFL file

	RCODE flush(
		RFL_BUFFER *	pBuffer,
		FLMBOOL			bFinalWrite = FALSE,
		FLMUINT			uiCurrPacketLen = 0,
		FLMBOOL			bStartingNewFile = FALSE);

	void switchBuffers( void);

	// Flush all packets except the current one to disk.
	// Shift the current one down to close to or at the
	// beginning of the buffer.

	RCODE shiftPacketsDown(
		FLMUINT		uiCurrPacketLen,
		FLMBOOL		bStartingNewFile);

	// See if we need to generate a new RFL file.

	RCODE seeIfNeedNewFile(
		FLMUINT	uiPacketLen,
		FLMBOOL	bDoNewIfOverLowLimit);

	// Calculate checksum, etc. on current packet.

	RCODE finishPacket(
		FLMUINT		uiPacketType,
		FLMUINT		uiPacketBodyLen,
		FLMBOOL		bDoNewIfOverLowLimit);

	RCODE logChangeFields(
		FlmRecord *	pOldRecord,
		FlmRecord *	pNewRecord);

	RCODE logRecord(
		FlmRecord *	pRecord);

	// Functions for reading log files

	RCODE readPacket(
		FLMUINT	uiMinBytesNeeded);

	RCODE getPacket(
		FLMBOOL		bForceNextFile,
		FLMUINT *	puiPacketTypeRV,
		FLMBYTE **	ppucPacketBodyRV,
		FLMUINT *	puiPacketBodyLenRV,
		FLMBOOL *	pbLoggedTimes);

	RCODE getRecord(
		FDB *			pDb,
		FLMUINT		uiPacketType,
		FLMBYTE *	pucPacketBody,
		FLMUINT		uiPacketBodyLen,
		FlmRecord *	pRecord);

	RCODE modifyRecord(
		HFDB			hDb,
		FlmRecord *	pRecord);

	RCODE readOp(
		FDB *				pDb,
		FLMBOOL			bForceNextFile,
		RFL_OP_INFO *	pOpInfo,
		FlmRecord *		pRecord);

	RCODE setupTransaction( void);

	void finalizeTransaction( void);

	// Member variables

	FFILE *			m_pFile;					// Pointer to FFILE structure
	RFL_BUFFER		m_Buf1;
	RFL_BUFFER		m_Buf2;
	F_MUTEX			m_hBufMutex;
	RFL_BUFFER *	m_pCommitBuf;			// Current buffer being committedout.
													// NULL if no buffer is being committed.
	RFL_BUFFER *	m_pCurrentBuf;			// Current write buffer - points to
													// m_Buf1 or m_Buf2
	FLMUINT			m_uiRflWriteBufs;		// Number of RFL buffers
	FLMUINT			m_uiBufferSize;		// Buffer size
	FLMBOOL			m_bKeepRflFiles;		// Keep RFL files after they are
													// no longer needed?
	FLMUINT			m_uiRflMinFileSize;	// Minimum RFL file size.
	FLMUINT			m_uiRflMaxFileSize;	// Maximum RFL file size.
	IF_FileHdl *	m_pFileHdl;		  		// File handle for writing to roll
													//	forward log file - only need one for
													//	the writer because we can only have
													//	one update transaction at a time.
	FLMUINT			m_uiLastRecoverFileNum;
													// Last file number to go to when
													// doing recovery.
	FLMBYTE			m_ucCurrSerialNum [F_SERIAL_NUM_SIZE];
													// Current file's serial number.
	FLMBOOL			m_bLoggingOff;			// Is logging turned off? Logging
													//	will be off during recovery.
	FLMBOOL			m_bLoggingUnknown;	// Are we in the middle of logging
													// unknown data for the application?
	FLMUINT			m_uiUnknownPacketLen;// Used for logging unknown data.
	FLMBOOL			m_bReadingUnknown;	// Are we in the middle of reading
													// unknown data for the appliation?
	FLMUINT			m_uiUnknownPacketBodyLen;
													// Body length of current
													// unknown packet.
	FLMBYTE *		m_pucUnknownPacketBody;
													// Pointer to current unknown packet.
	FLMUINT			m_uiUnknownBodyLenProcessed;
													// Bytes in unknown packet we have
													// already processed.
	RCODE				m_uiUnknownPacketRc;	// Rcode returned from getting a
													// packet while processing unknown
													// packets.
	FLMUINT			m_uiTransStartFile;	// File the current transaction started
													// in.
	FLMUINT			m_uiTransStartAddr;	// Offset of start transaction packet.
	FLMUINT			m_uiCurrTransID;		// Current transaction ID.
	FLMUINT			m_uiLastTransID;		// Last transaction ID.
	FLMUINT			m_uiLastLoggedCommitTransID;	// Last committed transaction that
																// was logged to the RFL
	FLMUINT			m_uiOperCount;			// Operations that have been logged for
													// this transaction.
	FLMUINT			m_uiRflReadOffset;	// Offset we are reading from in the
													//	buffer - only used when in reading
													//	mode.
	FLMUINT			m_uiPacketAddress;	// Current packet's address in the file.
	FLMUINT			m_uiFileEOF;			// End of file for current file.
													//	Only used when reading.
	F_Restore *		m_pRestore;				// Restore object.
	char				m_szDbPrefix [F_FILENAME_SIZE];		
													// First three characters of DB name.
	char				m_szRflDir [F_PATH_MAX_SIZE];
													// RFL directory
	FLMBOOL			m_bRflDirSameAsDb;
	FLMBOOL			m_bCreateRflDir;
	FLMBYTE			m_ucNextSerialNum [F_SERIAL_NUM_SIZE];
													// Next file's serial number.
	FLMBOOL			m_bRflVolumeOk;		// Did we have a problem accessing the
													// RFL volume?
	FLMBOOL			m_bRflVolumeFull;		// Did we have a problem accessing the
													// RFL volume?
};

/****************************************************************************
Desc:	This object is our implementation of the
		F_UnknownStream object which is used to handle unknown
		objects in the RFL.
****************************************************************************/
class F_RflUnknownStream : public F_UnknownStream
{
public:

	F_RflUnknownStream();
	virtual ~F_RflUnknownStream();

	RCODE setup(
		F_Rfl *			pRfl,
		FLMBOOL			bInputStream);

	RCODE reset( void);

	RCODE read(
		FLMUINT			uiLength,				// Number of bytes to read
		void *			pvBuffer,				// Buffer to place read bytes into
		FLMUINT *		puiBytesRead);			// [out] Number of bytes read

	RCODE write(
		FLMUINT			uiLength,				// Number of bytes to write
		void *			pvBuffer);

	RCODE close( void);							// Reads to the end of the
														// stream and discards any
														// remaining data (if input stream).

private:

	FLMBOOL		m_bSetupCalled;
	F_Rfl *		m_pRfl;							// RFL object.
	FLMBOOL		m_bInputStream;				// TRUE=input stream, FALSE=output stream
	FLMBOOL		m_bStartedWriting;			// Only used for output streams
};

/**************************************************************************
Desc: 	This structure is passed to the callback function that gets the
			differences between an old and a new record when a record
			modify operation is logged.
**************************************************************************/
typedef struct RFL_CHANGE_DATA
{
	RCODE			rc;
	FLMUINT		uiVersionNum;
	F_Rfl *		pRfl;
	FLMUINT		uiCurrPacketLen;
	FLMUINT		uiPacketCount;
	FLMUINT		uiTotalBytesLogged;
	FLMUINT		uiMaxLogBytesNeeded;
} RFL_CHANGE_DATA;

FLMBYTE RflCalcChecksum(
	const FLMBYTE *	pucPacket,
	FLMUINT				uiPacketBodyLen);

void rflGetBaseFileName(
	FLMUINT				uiDbVersion,
	const char *		pszDbPrefix,
	FLMUINT				uiFileNum,
	char *				pszBaseNameOut);

RCODE rflGetDirAndPrefix(
	FLMUINT				uiDbVersionNum,
	const char *		pszDbFileName,
	const char *		pszRflDirIn,
	char *				pszRflDirOut,
	char *				pszDbPrefixOut);

RCODE rflGetFileName(
	FLMUINT				uiDbVersion,
	const char *		pszDbName,
	const char *		pszRflDir,
	FLMUINT				uiFileNum,
	char *				pszRflFileName);

FLMBOOL rflGetFileNum(
	FLMUINT				uiDbVersion,
	const char *		pszPrefix,
	const char *		pszRflFileName,
	FLMUINT *			puiFileNum);

RCODE flmRflCalcDiskUsage(
	const char *		pszRflDir,
	const char *		pszRflPrefix,
	FLMUINT				uiDbVersionNum,
	FLMUINT64 *			pui64DiskUsage);
	
#include "fpackoff.h"

#endif
