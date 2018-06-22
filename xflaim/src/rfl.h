//------------------------------------------------------------------------------
// Desc:	This file contains structures and definitions used for roll
//			forward logging.
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

class IXKeyCompare;

// Packet types for roll forward logging

#define RFL_TRNS_BEGIN_PACKET					1
#define RFL_TRNS_COMMIT_PACKET				2
#define RFL_TRNS_ABORT_PACKET					3
#define RFL_REDUCE_PACKET						4
#define RFL_UPGRADE_PACKET						5
#define RFL_INDEX_SUSPEND_PACKET				6
#define RFL_INDEX_RESUME_PACKET				7
#define RFL_BLK_CHAIN_FREE_PACKET			8
#define RFL_ENABLE_ENCRYPTION_PACKET		9
#define RFL_WRAP_KEY_PACKET					10

#define RFL_NODE_DELETE_PACKET				11
#define RFL_NODE_CHILDREN_DELETE_PACKET	12
#define RFL_NODE_CREATE_PACKET				13
#define RFL_DOCUMENT_DONE_PACKET				14
#define RFL_INSERT_BEFORE_PACKET				15
#define RFL_NODE_FLAGS_UPDATE_PACKET		16
#define RFL_NODE_SET_PREFIX_ID_PACKET		17
#define RFL_SET_NEXT_NODE_ID_PACKET			18
#define RFL_ENC_NODE_UPDATE_PACKET			19
#define RFL_NODE_SET_NUMBER_VALUE_PACKET	20
#define RFL_NODE_SET_TEXT_VALUE_PACKET		21
#define RFL_NODE_SET_BINARY_VALUE_PACKET	22
#define RFL_DATA_PACKET							23
#define RFL_ROLL_OVER_DB_KEY_PACKET			24
#define RFL_ENC_DEF_KEY_PACKET				25
#define RFL_NODE_SET_META_VALUE_PACKET		26
#define RFL_ATTR_SET_VALUE_PACKET			27
#define RFL_ATTR_CREATE_PACKET				28
#define RFL_ATTR_DELETE_PACKET				29
#define RFL_NODE_CLEAR_VALUE_PACKET			30

#define RFL_PACKET_TYPE_MASK					0x7F

// Flags for all packets

#define RFL_HAVE_COUNTS_FLAG					0x01
#define RFL_HAVE_DATA_FLAG						0x02
#define RFL_FIRST_FLAG							0x04
#define RFL_LAST_FLAG							0x08
#define RFL_TRUNCATE_FLAG						0x10

#define RFL_GET_PACKET_TYPE(uiPacketType)	\
	((FLMUINT)((uiPacketType) & RFL_PACKET_TYPE_MASK))

// Definitions for ROLL FORWARD LOG file header format

#define	RFL_NAME_POS		  					0
#define	RFL_NAME									"RFL5"
#define	RFL_NAME_LEN		  					4
#define	RFL_VERSION_POS	  					RFL_NAME_LEN
#define	RFL_VERSION		  						"5.00"
#define	RFL_VERSION_LEN		  				4
#define	RFL_FILE_NUMBER_POS					8
#define	RFL_EOF_POS								12
#define	RFL_DB_SERIAL_NUM_POS				16
#define	RFL_SERIAL_NUM_POS					(RFL_DB_SERIAL_NUM_POS + XFLM_SERIAL_NUM_SIZE)
#define	RFL_NEXT_FILE_SERIAL_NUM_POS		(RFL_SERIAL_NUM_POS + XFLM_SERIAL_NUM_SIZE)
#define	RFL_KEEP_SIGNATURE_POS				(RFL_NEXT_FILE_SERIAL_NUM_POS + XFLM_SERIAL_NUM_SIZE)

#define	RFL_KEEP_SIGNATURE					"----KeepLog----"
#define	RFL_NOKEEP_SIGNATURE					"--DontKeepLog--"

// Buffer size needs to be a multiple of 512 for direct IO writes.

#define	DEFAULT_RFL_WRITE_BUFFERS			8
#define	DEFAULT_RFL_BUFFER_SIZE				(65536)

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

#define RFL_MAX_PACKET_SIZE					(65536 - 1024)
#define RFL_MAX_PACKET_BODY_SIZE				(RFL_MAX_PACKET_SIZE - RFL_PACKET_OVERHEAD)

typedef struct RflWaiterTag *	RFL_WAITER_p;

typedef struct RflWaiterTag
{
	FLMUINT			uiThreadId;
	FLMBOOL			bIsWriter;
	F_SEM				hESem;
	RCODE *			pRc;
	RFL_WAITER_p	pNext;
} RFL_WAITER;

typedef struct
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
	XFLM_DB_HDR			dbHdr;					// DB header to be written with
														// this buffer.
	XFLM_DB_HDR			cpHdr;					// Checkpoint header for this
														// buffer.
	RFL_WAITER *		pFirstWaiter;
	RFL_WAITER *		pLastWaiter;
} RFL_BUFFER;

/**************************************************************************
Desc:	This class handles all of the roll-forward logging for FLAIM.  There
		is one of these objects allocated per F_Database.
**************************************************************************/
class F_Rfl : public F_Object
{
public:

	F_Rfl();
	
	~F_Rfl();

	RCODE setup(
		F_Database *			pDatabase,
		const char *			pszRflDir);

	RCODE finishCurrFile(
		F_Db *					pDb,
		FLMBOOL					bNewKeepState);

	RCODE logBeginTransaction(
		F_Db *					pDb);

	RCODE logEndTransaction(
		F_Db *					pDb,
		FLMUINT					uiPacketType,
		FLMBOOL					bThrowLogAway,
		FLMBOOL *				pbLoggedTransEnd = NULL);

	RCODE logReduce(
		F_Db *					pDb,
		FLMUINT					uiCount);

	RCODE logIndexSuspendOrResume(
		F_Db *					pDb,
		FLMUINT					uiIndexNum,
		FLMUINT					uiPacketType);

	RCODE logUpgrade(
		F_Db *					pDb,
		FLMUINT					uiOldVersion);

	RCODE logBlockChainFree(
		F_Db *					pDb,
		FLMUINT64				ui64MaintDocID,
		FLMUINT					uiStartBlkAddr,
		FLMUINT					uiEndBlkAddr,
		FLMUINT					uiCount);

	RCODE logEncryptionKey(
		F_Db *					pDb,
		FLMUINT					uiPacketType,
		FLMBYTE *				pucKey,
		FLMUINT32				ui32KeyLen);
		
	RCODE logEncDefKey(
		F_Db *					pDb,
		FLMUINT					uiEncDefId,
		FLMBYTE *				pucKeyValue,
		FLMUINT					uiKeyValueLen,
		FLMUINT					uiKeySize);
		
	RCODE logRollOverDbKey(
		F_Db *					pDb);
	
	RCODE logDocumentDone(
		F_Db *					pDb,
		FLMUINT					uiCollection,
		FLMUINT64				ui64RootId);
		
	RCODE logNodeDelete(
		F_Db *					pDb,
		FLMUINT					uiCollection,
		FLMUINT64				ui64NodeId);

	RCODE logAttributeDelete(
		F_Db *					pDb,
		FLMUINT					uiCollection,
		FLMUINT64				ui64ElementId,
		FLMUINT					uiAttrName);
		
	RCODE logNodeChildrenDelete(
		F_Db *					pDb,
		FLMUINT					uiCollection,
		FLMUINT64				ui64NodeId,
		FLMUINT					uiNameId);
		
	RCODE logNodeCreate(
		F_Db *					pDb,
		FLMUINT					uiCollection,
		FLMUINT64				ui64ParentId,
		eDomNodeType			eNodeType,
		FLMUINT					uiNameId,
		eNodeInsertLoc			eLocation,
		FLMUINT64				ui64NodeId);
		
	RCODE logAttributeCreate(
		F_Db *					pDb,
		FLMUINT					uiCollection,
		FLMUINT64				ui64ElementId,
		FLMUINT					uiNameId,
		FLMUINT					uiNextAttrNameId);
	
	RCODE logAttributeSetValue(
		F_Db *					pDb,
		FLMUINT					uiCollection,
		FLMUINT64				ui64ElementNodeId,
		void *					pvValue,
		FLMUINT					uiValueLen);

	RCODE logNodeSetValue(
		F_Db *					pDb,
		FLMUINT					uiPacketType,
		F_CachedNode *			pCachedNode);

	RCODE logEncryptedNodeUpdate(
		F_Db *					pDb,
		F_CachedNode *			pCachedNode);
		
	RCODE logNodeSetNumberValue(
		F_Db *					pDb,
		FLMUINT					uiCollection,
		FLMUINT64				ui64NodeId,
		FLMUINT64				ui64Value,
		FLMBOOL					bNegative);
		
	RCODE logAttrSetValue(
		F_Db *					pDb,
		F_CachedNode *			pCachedNode,
		FLMUINT					uiAttrName);

	RCODE logInsertBefore(
		F_Db *					pDb,
		FLMUINT					uiCollection,
		FLMUINT64				ui64Parent,
		FLMUINT64				ui64Child,
		FLMUINT64				ui64RefChild);
	
	RCODE logNodeFlagsUpdate(
		F_Db *					pDb,
		FLMUINT					uiCollection,
		FLMUINT64				ui64NodeId,
		FLMUINT					uiAttrNameId,
		FLMUINT					uiFlags,
		FLMBOOL					bSetting);
	
	RCODE logNodeSetPrefixId(
		F_Db *					pDb,
		FLMUINT					uiCollection,
		FLMUINT64				ui64NodeId,
		FLMUINT					uiAttrNameId,
		FLMUINT					uiPrefixId);
		
	RCODE logNodeSetMetaValue(
		F_Db *					pDb,
		FLMUINT					uiCollection,
		FLMUINT64				ui64NodeId,
		FLMUINT64				ui64MetaValue);
		
	RCODE logSetNextNodeId(
		F_Db *					pDb,
		FLMUINT					uiCollection,
		FLMUINT64				ui64NextNodeId);
		
	RCODE logNodeClearValue(
		F_Db *					pDb,
		FLMUINT					uiCollection,
		FLMUINT64				ui64NodeId,
		FLMUINT					uiAttrNameId);
		
	RCODE recover(
		F_Db *					pDb,
		IF_RestoreClient *	pRestore,
		IF_RestoreStatus *	pRestoreStatus);

	FLMBOOL atEndOfLog( void);

	// Returns full log file name associated with number.
	// Will return the full path name.

	void getFullRflFileName(
		FLMUINT					uiFileNum,
		char *					pszFullRflFileName,
		FLMUINT *				puiFileNameBufSize,
		FLMBOOL *				pbNameTruncated = NULL);

	// Set the RFL directory.  Passing in a NULL or empty
	// string will cause the directory to be to the same
	// directory where the database is located.

	RCODE setRflDir(
		const char *			pszRflDir);

	FINLINE const char * getRflDirPtr( void)
	{
		return &m_szRflDir [0];
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
		f_memcpy( pucSerialNum, m_ucCurrSerialNum, XFLM_SERIAL_NUM_SIZE);
	}

	FINLINE void setCurrSerialNum(
		FLMBYTE *	pucSerialNum)
	{
		f_memcpy( m_ucCurrSerialNum, pucSerialNum, XFLM_SERIAL_NUM_SIZE);
	}

	FINLINE void getNextSerialNum(
		FLMBYTE *	pucSerialNum)
	{
		f_memcpy( pucSerialNum, m_ucNextSerialNum, XFLM_SERIAL_NUM_SIZE);
	}

	FINLINE void setNextSerialNum(
		FLMBYTE *	pucSerialNum)
	{
		f_memcpy( m_ucNextSerialNum, pucSerialNum, XFLM_SERIAL_NUM_SIZE);
	}

	FINLINE FLMUINT64 getCurrTransID( void)
	{
		return( m_ui64CurrTransID);
	}

	RCODE truncate(
		F_SEM			hWaitSem,
		FLMUINT		uiTruncateSize);

	// Public functions, but only intended for internal use

	RCODE makeRoom(
		F_Db *		pDb,
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

	// Close the current RFL file

	FINLINE void closeFile( void)
	{
		if( m_pCurrentBuf->pBufferMgr)
		{
			flmAssert( !m_pCurrentBuf->pBufferMgr->isIOPending());
		}
		if (m_pFileHdl)
		{
			m_pFileHdl->closeFile();
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
			return( NE_XFLM_OK);
		}
	}

	RCODE waitForWrites(
		F_SEM				hWaitSem,
		RFL_BUFFER *	pBuffer,
		FLMBOOL			bIsWriter);

	RCODE waitForCommit(
		F_SEM				hWaitSem);

	FINLINE void commitDbHdrs(
		XFLM_DB_HDR *	pDbHdr,
		XFLM_DB_HDR *	pCPHdr)
	{
		f_memcpy( &m_pCurrentBuf->dbHdr, pDbHdr, sizeof( XFLM_DB_HDR));
		f_memcpy( &m_pCurrentBuf->cpHdr, pCPHdr, sizeof( XFLM_DB_HDR));
		m_pCurrentBuf->bOkToWriteHdrs = TRUE;
	}

	FINLINE void clearDbHdrs( void)
	{
		m_pCurrentBuf->bOkToWriteHdrs = FALSE;
	}

	FLMBOOL seeIfRflWritesDone(
		F_SEM				hWaitSem,
		FLMBOOL			bForceWait);

	void wakeUpWaiter(
		RCODE				rc,
		FLMBOOL			bIsWriter);

	RCODE completeTransWrites(
		F_Db *			pDb,
		FLMBOOL			bCommitting,
		FLMBOOL			bOkToUnlock);

	FINLINE void disableLogging(
		FLMUINT *			puiToken)
	{
		*puiToken = ++m_uiDisableCount;
	}
	
	FINLINE void enableLogging(
		FLMUINT * 			puiToken)
	{
		flmAssert( m_uiDisableCount);
		flmAssert( *puiToken && *puiToken == m_uiDisableCount);
		
		m_uiDisableCount--;
		*puiToken = 0;
	}
	
	FINLINE FLMBOOL isLoggingEnabled( void)
	{
		return( m_uiDisableCount ? FALSE : TRUE);
	}

private:

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
		FLMUINT			uiFileNum,
		FLMUINT			uiEof,
		FLMBYTE *		pucSerialNum,
		FLMBYTE *		pucNextSerialNum,
		FLMBOOL			bKeepSignature);

	// Verify the header of an RFL file.

	RCODE verifyHeader(
		FLMBYTE *		pucHeader,
		FLMUINT			uiFileNum,
		FLMBYTE *		pucSerialNum);

	// Open a new RFL file.

	RCODE openFile(
		F_SEM				hWaitSem,
		FLMUINT			uiFileNum,
		FLMBYTE *		pucSerialNum);

	// Create a new RFL file

	RCODE createFile(
		F_Db *			pDb,
		FLMUINT			uiFileNum,
		FLMBYTE *		pucSerialNum,
		FLMBYTE *		pucNextSerialNum,
		FLMBOOL			bKeepSignature);

	void copyLastSector(
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
		F_Db *			pDb,
		RFL_BUFFER *	pBuffer,
		FLMBOOL			bFinalWrite = FALSE,
		FLMUINT			uiCurrPacketLen = 0,
		FLMBOOL			bStartingNewFile = FALSE);

	void switchBuffers( void);

	// Flush all packets except the current one to disk.
	// Shift the current one down to close to or at the
	// beginning of the buffer.

	RCODE shiftPacketsDown(
		F_Db *			pDb,
		FLMUINT			uiCurrPacketLen,
		FLMBOOL			bStartingNewFile);

	// See if we need to generate a new RFL file.

	RCODE seeIfNeedNewFile(
		F_Db *			pDb,
		FLMUINT			uiPacketLen,
		FLMBOOL			bDoNewIfOverLowLimit);

	// Calculate checksum, etc. on current packet.

	RCODE finishPacket(
		F_Db *			pDb,
		FLMUINT			uiPacketType,
		FLMUINT			uiPacketBodyLen,
		FLMBOOL			bDoNewIfOverLowLimit);

	// Functions for reading log files

	RCODE readPacket(
		FLMUINT			uiMinBytesNeeded);

	RCODE getPacket(
		F_Db *				pDb,
		FLMBOOL				bForceNextFile,
		FLMUINT *			puiPacketTypeRV,
		const FLMBYTE **	ppucPacketBodyRV,
		FLMUINT *			puiPacketBodyLenRV);

	RCODE setupTransaction(
		F_Db *			pDb);

	void finalizeTransaction( void);

	RCODE recovTransBegin(
		F_Db *				pDb,
		eRestoreAction *	peAction);

	RCODE recovBlockChainFree(
		F_Db *				pDb,
		const FLMBYTE *	pucPacketBody,
		FLMUINT				uiPacketBodyLen,
		eRestoreAction *	peAction);

	RCODE recovIndexSuspendResume(
		F_Db *				pDb,
		FLMUINT				uiPacketType,
		const FLMBYTE *	pucPacketBody,
		FLMUINT				uiPacketBodyLen,
		eRestoreAction *	peAction);

	RCODE recovReduce(
		F_Db *				pDb,
		const FLMBYTE *	pucPacketBody,
		FLMUINT				uiPacketBodyLen,
		eRestoreAction *	peAction);

	RCODE recovUpgrade(
		F_Db *				pDb,
		const FLMBYTE *	pucPacketBody,
		FLMUINT				uiPacketBodyLen,
		eRestoreAction *	peAction);

	RCODE recovEncryptionKey(
		F_Db *				pDb,
		FLMUINT				uiPacketType,
		const FLMBYTE *	pucPacketBody,
		FLMUINT				uiPacketBodyLen,
		eRestoreAction *	peAction);

	RCODE recovEncDefKey(
		F_Db *				pDb,
		const FLMBYTE *	pucPacketBody,
		FLMUINT				uiPacketBodyLen,
		eRestoreAction *	peAction);
		
	RCODE recovRollOverDbKey(
		F_Db *				pDb,
		const FLMBYTE *	pucPacketBody,
		FLMUINT				uiPacketBodyLen,
		eRestoreAction *	peAction);
		
	RCODE recovDocumentDone(
		F_Db *				pDb,
		const FLMBYTE *	pucPacketBody,
		FLMUINT				uiPacketBodyLen,
		eRestoreAction *	peAction);
		
	RCODE recovNodeDelete(
		F_Db *				pDb,
		const FLMBYTE *	pucPacketBody,
		FLMUINT				uiPacketBodyLen,
		eRestoreAction *	peAction);
		
	RCODE recovAttributeDelete(
		F_Db *				pDb,
		const FLMBYTE *	pucPacketBody,
		FLMUINT				uiPacketBodyLen,
		eRestoreAction *	peAction);
		
	RCODE recovNodeChildrenDelete(
		F_Db *				pDb,
		const FLMBYTE *	pucPacketBody,
		FLMUINT				uiPacketBodyLen,
		eRestoreAction *	peAction);
		
	RCODE recovNodeCreate(
		F_Db *				pDb,
		const FLMBYTE *	pucPacketBody,
		FLMUINT				uiPacketBodyLen,
		eRestoreAction *	peAction);
		
	RCODE recovAttributeCreate(
		F_Db *				pDb,
		const FLMBYTE *	pucPacketBody,
		FLMUINT				uiPacketBodyLen,
		eRestoreAction *	peAction);
		
	RCODE recovInsertBefore(
		F_Db *				pDb,
		const FLMBYTE *	pucPacketBody,
		FLMUINT				uiPacketBodyLen,
		eRestoreAction *	peAction);
		
	RCODE recovEncryptedNodeUpdate(
		F_Db *				pDb,
		const FLMBYTE *	pucPacketBody,
		FLMUINT				uiPacketBodyLen,
		eRestoreAction *	peAction);
	
	RCODE recovNodeSetValue(
		F_Db *				pDb,
		FLMUINT				uiPacketType,
		const FLMBYTE *	pucPacketBody,
		FLMUINT				uiPacketBodyLen,
		eRestoreAction *	peAction);

	RCODE recovNodeSetNumberValue(
		F_Db *				pDb,
		const FLMBYTE *	pucPacketBody,
		FLMUINT				uiPacketBodyLen,
		eRestoreAction *	peAction);
		
	RCODE recovAttrSetValue(
		F_Db *				pDb,
		const FLMBYTE *	pucPacketBody,
		FLMUINT				uiPacketBodyLen,
		eRestoreAction *	peAction);

	RCODE recovNodeFlagsUpdate(
		F_Db *				pDb,
		const FLMBYTE *	pucPacketBody,
		FLMUINT				uiPacketBodyLen,
		eRestoreAction *	peAction);
		
	RCODE recovNodeSetPrefixId(
		F_Db *				pDb,
		const FLMBYTE *	pucPacketBody,
		FLMUINT				uiPacketBodyLen,
		eRestoreAction *	peAction);
		
	RCODE recovNodeSetMetaValue(
		F_Db *				pDb,
		const FLMBYTE *	pucPacketBody,
		FLMUINT				uiPacketBodyLen,
		eRestoreAction *	peAction);
		
	RCODE recovSetNextNodeId(
		F_Db *				pDb,
		const FLMBYTE *	pucPacketBody,
		FLMUINT				uiPacketBodyLen,
		eRestoreAction *	peAction);
		
	RCODE recovNodeClearValue(
		F_Db *				pDb,
		const FLMBYTE *	pucPacketBody,
		FLMUINT				uiPacketBodyLen,
		eRestoreAction *	peAction);
		
	FLMBOOL useDataOnlyBlocks(
		F_Db *				pDb,
		FLMUINT				uiDataLen);

	// Member variables

	F_Database *				m_pDatabase;			// Pointer to database
	RFL_BUFFER					m_Buf1;
	RFL_BUFFER					m_Buf2;
	F_MUTEX						m_hBufMutex;
	RFL_BUFFER *				m_pCommitBuf;			// Current buffer being committedout.
																// NULL if no buffer is being committed.
	RFL_BUFFER *				m_pCurrentBuf;			// Current write buffer - points to
																// m_Buf1 or m_Buf2
	FLMUINT						m_uiRflWriteBufs;		// Number of RFL buffers
	FLMUINT						m_uiBufferSize;		// Buffer size
	FLMBOOL						m_bKeepRflFiles;		// Keep RFL files after they are
																// no longer needed?
	FLMUINT						m_uiRflMinFileSize;	// Minimum RFL file size.
	FLMUINT						m_uiRflMaxFileSize;	// Maximum RFL file size.
	IF_FileHdl *				m_pFileHdl;		  		// File handle for writing to roll
																//	forward log file - only need one for
																//	the writer because we can only have
																//	one update transaction at a time.
	FLMUINT						m_uiLastRecoverFileNum;
																// Last file number to go to when
																// doing recovery.
	FLMBYTE						m_ucCurrSerialNum [XFLM_SERIAL_NUM_SIZE];
																// Current file's serial number.
	FLMUINT						m_uiTransStartFile;	// File the current transaction started
																// in.
	FLMUINT						m_uiTransStartAddr;	// Offset of start transaction packet.
	FLMUINT64					m_ui64CurrTransID;	// Current transaction ID.
	FLMUINT64					m_ui64LastTransID;	// Last transaction ID.
	FLMUINT64					m_ui64LastLoggedCommitTransID;
																// Last committed transaction that
																// was logged to the RFL
	FLMUINT						m_uiOperCount;			// Operations that have been logged for
																// this transaction.
	FLMUINT						m_uiRflReadOffset;	// Offset we are reading from in the
																//	buffer - only used when in reading
																//	mode.
	FLMUINT						m_uiPacketAddress;	// Current packet's address in the file.
	FLMUINT						m_uiFileEOF;			// End of file for current file.
																//	Only used when reading.
	IF_RestoreClient *		m_pRestore;				// Restore object.
	IF_RestoreStatus *		m_pRestoreStatus;
																// Restore status object
	char							m_szRflDir [F_PATH_MAX_SIZE];
																// RFL directory
	FLMBOOL						m_bRflDirSameAsDb;
	FLMBOOL						m_bCreateRflDir;
	FLMBYTE						m_ucNextSerialNum [ XFLM_SERIAL_NUM_SIZE];
																// Next file's serial number.
	FLMBOOL						m_bRflVolumeOk;		// Did we have a problem accessing the
																// RFL volume?
	FLMBOOL						m_bRflVolumeFull;		// Did we have a problem accessing the
																// RFL volume?
	FLMUINT						m_uiLastLfNum;			// Last logical file used in restore/recover
	eLFileType					m_eLastLfType;			// The Last Lfile Type
	IXKeyCompare *				m_pIxCompareObject;
	IF_ResultSetCompare *	m_pCompareObject;
	FLMUINT						m_uiDisableCount;

friend class F_RflOStream;
};

// Prototypes for roll forward logging functions.

FLMBYTE RflCalcChecksum(
	FLMBYTE *		pucPacket,
	FLMUINT			uiPacketBodyLen);

void rflGetBaseFileName(
	FLMUINT			uiFileNum,
	char *			pszBaseNameOut,
	FLMUINT *		puiFileNameBufSize,
	FLMBOOL *		pbNameTruncated = NULL);

RCODE rflGetDirAndPrefix(
	const char *	pszDbFileName,
	const char *	pszRflDirIn,
	char *			pszRflDirOut);

RCODE rflGetFileName(
	const char *	szDbName,
	const char *	szRflDir,
	FLMUINT			uiFileNum,
	char *			pszRflFileName);

FLMBOOL rflGetFileNum(
	const char *	pszRflFileName,
	FLMUINT *		puiFileNum);

#endif	// ifdef RFL_H
