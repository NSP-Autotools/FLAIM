//------------------------------------------------------------------------------
// Desc:	Database config get/set functions
// Tabs:	3
//
// Copyright (c) 1996-2007 Novell, Inc. All Rights Reserved.
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

#include "flaimsys.h"

/****************************************************************************
Desc:	Set the RFL keep files flag.
****************************************************************************/
RCODE XFLAPI F_Db::setRflKeepFilesFlag(
	FLMBOOL	bKeepFiles)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bDbLocked = FALSE;

	// See if the database is being forced to close

	if (RC_BAD( rc = checkState( __FILE__, __LINE__)))
	{
		goto Exit;
	}

	// Make sure we don't have a transaction going

	if (m_eTransType != XFLM_NO_TRANS)
	{
		rc = RC_SET( NE_XFLM_TRANS_ACTIVE);
		goto Exit;
	}

	// Make sure there is no active backup running

	m_pDatabase->lockMutex();
	if (m_pDatabase->m_bBackupActive)
	{
		m_pDatabase->unlockMutex();
		rc = RC_SET( NE_XFLM_BACKUP_ACTIVE);
		goto Exit;
	}
	m_pDatabase->unlockMutex();

	// Need to lock the database but not start a transaction yet.

	if (!(m_uiFlags & (FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_SHARED)))
	{
		if (RC_BAD( rc = dbLock( FLM_LOCK_EXCLUSIVE, 0, FLM_NO_TIMEOUT)))
		{
			goto Exit;
		}
		bDbLocked = TRUE;
	}

	// If we aren't changing the keep flag, jump to exit without doing
	// anything.

	if ((bKeepFiles &&
		  m_pDatabase->m_lastCommittedDbHdr.ui8RflKeepFiles) ||
		 (!bKeepFiles &&
		  !m_pDatabase->m_lastCommittedDbHdr.ui8RflKeepFiles))
	{
		goto Exit;	// Will return NE_XFLM_OK;
	}

	// Force a checkpoint and roll to the next RFL file numbers.
	// When changing from keep to no-keep or vice versa, we need to
	// go to a new RFL file so that the new RFL file gets new
	// serial numbers and a new keep or no-keep flag.

	if (RC_BAD( rc = doCheckpoint( FLM_NO_TIMEOUT)))
	{
		goto Exit;
	}

	f_memcpy( &m_pDatabase->m_uncommittedDbHdr,
				 &m_pDatabase->m_lastCommittedDbHdr,
				 sizeof( XFLM_DB_HDR));
	m_pDatabase->m_uncommittedDbHdr.ui8RflKeepFiles =
		(FLMUINT8)(bKeepFiles
					  ? (FLMUINT8)1
					  : (FLMUINT8)0);

	// Force a new RFL file - this will also write out the entire
	// log header - including the changes we made above.

	if (RC_BAD( rc = m_pDatabase->m_pRfl->finishCurrFile( this, TRUE)))
	{
		goto Exit;
	}

Exit:

	if (bDbLocked)
	{
		dbUnlock();
	}

	return( rc);
}

/****************************************************************************
Desc:	Set the RFL directory for a database.
****************************************************************************/
RCODE XFLAPI F_Db::setRflDir(
	const char *	pszNewRflDir)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bDbLocked = FALSE;

	// See if the database is being forced to close

	if (RC_BAD( rc = checkState( __FILE__, __LINE__)))
	{
		goto Exit;
	}

	// Make sure we don't have a transaction going

	if (m_eTransType != XFLM_NO_TRANS)
	{
		rc = RC_SET( NE_XFLM_TRANS_ACTIVE);
		goto Exit;
	}

	// Make sure there is no active backup running

	m_pDatabase->lockMutex();
	if (m_pDatabase->m_bBackupActive)
	{
		m_pDatabase->unlockMutex();
		rc = RC_SET( NE_XFLM_BACKUP_ACTIVE);
		goto Exit;
	}
	m_pDatabase->unlockMutex();

	// Make sure the path exists and that it is a directory
	// rather than a file.

	if (pszNewRflDir && *pszNewRflDir)
	{
		if (!gv_XFlmSysData.pFileSystem->isDir( pszNewRflDir))
		{
			rc = RC_SET( NE_FLM_IO_INVALID_FILENAME);
			goto Exit;
		}
	}

	// Need to lock the database because we can't change the RFL
	// directory until after the checkpoint has completed.  The
	// checkpoint code will unlock the transaction, but not the
	// file if we have an explicit lock.  We need to do this to
	// prevent another transaction from beginning before we have
	// changed the RFL directory.

	if (!(m_uiFlags & (FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_SHARED)))
	{
		if( RC_BAD( rc = dbLock( FLM_LOCK_EXCLUSIVE, 0, FLM_NO_TIMEOUT)))
		{
			goto Exit;
		}
		bDbLocked = TRUE;
	}

	// Force a checkpoint and roll to the next RFL file numbers.  Both
	// of these steps are necessary to ensure that we won't have to do
	// any recovery using the current RFL file - because we do not
	// move the current RFL file to the new directory.  Forcing the
	// checkpoint ensures that we have no transactions that will need
	// to be recovered if we were to crash.  Rolling the RFL file number
	// ensures that no more transactions will be logged to the current
	// RFL file.

	if (RC_BAD( rc = doCheckpoint( FLM_NO_TIMEOUT)))
	{
		goto Exit;
	}

	// Force a new RFL file.

	if (RC_BAD( rc = m_pDatabase->m_pRfl->finishCurrFile( this, FALSE)))
	{
		goto Exit;
	}

	// Set the RFL directory to the new value now that we have
	// finished the checkpoint and rolled to the next RFL file.

	m_pDatabase->lockMutex();
	rc = m_pDatabase->m_pRfl->setRflDir( pszNewRflDir);
	m_pDatabase->unlockMutex();

Exit:

	if (bDbLocked)
	{
		dbUnlock();
	}

	return( rc);
}

/****************************************************************************
Desc:	Set the RFL file size limits for a database.
****************************************************************************/
RCODE XFLAPI F_Db::setRflFileSizeLimits(
	FLMUINT	uiMinRflSize,
	FLMUINT	uiMaxRflSize)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bStartedTrans = FALSE;

	// See if the database is being forced to close

	if (RC_BAD( rc = checkState( __FILE__, __LINE__)))
	{
		goto Exit;
	}

	// Make sure the limits are valid.

	// Maximum must be enough to hold at least one packet plus
	// the RFL header.  Minimum must not be greater than the
	// maximum.  NOTE: Minimum and maximum are allowed to be
	// equal, but in all cases, maximum takes precedence over
	// minimum.  We will first NOT exceed the maximum.  Then,
	// if possible, we will go above the minimum.

	if (uiMaxRflSize < RFL_MAX_PACKET_SIZE + 512)
	{
		uiMaxRflSize = RFL_MAX_PACKET_SIZE + 512;
	}
	if (uiMaxRflSize > gv_XFlmSysData.uiMaxFileSize)
	{
		uiMaxRflSize = gv_XFlmSysData.uiMaxFileSize;
	}
	if (uiMinRflSize > uiMaxRflSize)
	{
		uiMinRflSize = uiMaxRflSize;
	}

	// Start an update transaction.  Must not already be one going.

	if (m_eTransType != XFLM_NO_TRANS)
	{
		rc = RC_SET( NE_XFLM_TRANS_ACTIVE);
		goto Exit;
	}
	if (RC_BAD( rc = beginTrans( XFLM_UPDATE_TRANS)))
	{
		goto Exit;
	}
	bStartedTrans = TRUE;

	// Commit the transaction.

	m_pDatabase->m_uncommittedDbHdr.ui32RflMinFileSize =
		(FLMUINT32)uiMinRflSize;
	m_pDatabase->m_uncommittedDbHdr.ui32RflMaxFileSize =
		(FLMUINT32)uiMaxRflSize;

	bStartedTrans = FALSE;
	if (RC_BAD( rc = commitTrans( 0, FALSE)))
	{
		goto Exit;
	}

Exit:

	if (bStartedTrans)
	{
		abortTrans();
	}

	return( rc);
}

/****************************************************************************
Desc:	Roll to the next RFL file for this database
****************************************************************************/
RCODE XFLAPI F_Db::rflRollToNextFile( void)
{
	RCODE	rc = NE_XFLM_OK;

	// See if the database is being forced to close

	if (RC_BAD( rc = checkState( __FILE__, __LINE__)))
	{
		goto Exit;
	}

	// NOTE: finishCurrFile will not roll to the next file if the current
	// file has not been created.

	if (RC_BAD( rc = m_pDatabase->m_pRfl->finishCurrFile( this, FALSE)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Set keep aborted transactions in RFL flag.
****************************************************************************/
RCODE XFLAPI F_Db::setKeepAbortedTransInRflFlag(
	FLMBOOL	bKeep
	)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bStartedTrans = FALSE;

	// See if the database is being forced to close

	if (RC_BAD( rc = checkState( __FILE__, __LINE__)))
	{
		goto Exit;
	}

	// Start an update transaction.  Must not already be one going.

	if (m_eTransType != XFLM_NO_TRANS)
	{
		rc = RC_SET( NE_XFLM_TRANS_ACTIVE);
		goto Exit;
	}
	if (RC_BAD( rc = beginTrans( XFLM_UPDATE_TRANS)))
	{
		goto Exit;
	}
	bStartedTrans = TRUE;

	// Change the uncommitted log header

	m_pDatabase->m_uncommittedDbHdr.ui8RflKeepAbortedTrans =
						(FLMUINT8)(bKeep
									 ? (FLMUINT8)1
									 : (FLMUINT8)0);

	// Commit the transaction.

	bStartedTrans = FALSE;
	if (RC_BAD( rc = commitTrans( 0, FALSE)))
	{
		goto Exit;
	}

Exit:

	if (bStartedTrans)
	{
		abortTrans();
	}

	return( rc);
}

/****************************************************************************
Desc:	Set auto turn off keep RFL flag.
****************************************************************************/
RCODE XFLAPI F_Db::setAutoTurnOffKeepRflFlag(
	FLMBOOL	bAutoTurnOff
	)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bStartedTrans = FALSE;

	// See if the database is being forced to close

	if (RC_BAD( rc = checkState( __FILE__, __LINE__)))
	{
		goto Exit;
	}

	// Start an update transaction.  Must not already be one going.

	if (m_eTransType != XFLM_NO_TRANS)
	{
		rc = RC_SET( NE_XFLM_TRANS_ACTIVE);
		goto Exit;
	}

	if (RC_BAD( rc = beginTrans( XFLM_UPDATE_TRANS)))
	{
		goto Exit;
	}
	bStartedTrans = TRUE;

	// Change the uncommitted log header

	m_pDatabase->m_uncommittedDbHdr.ui8RflAutoTurnOffKeep =
							(FLMUINT8)(bAutoTurnOff
										 ? (FLMUINT8)1
										 : (FLMUINT8)0);

	// Commit the transaction.

	bStartedTrans = FALSE;
	if (RC_BAD( rc = commitTrans( 0, FALSE)))
	{
		goto Exit;
	}

Exit:

	if (bStartedTrans)
	{
		abortTrans();
	}

	return( rc);
}

/****************************************************************************
Desc: Retrieves the Checkpoint info for the database passed in.  This assumes
		global mutex has already been locked.
*****************************************************************************/
void F_Database::getCPInfo(
	XFLM_CHECKPOINT_INFO *		pCheckpointInfo)
{
	FLMUINT	uiElapTime;
	FLMUINT	uiCurrTime;

	flmAssert( pCheckpointInfo);

	f_memset( pCheckpointInfo, 0, sizeof( XFLM_CHECKPOINT_INFO));
	if (m_pCPInfo)
	{
		pCheckpointInfo->bRunning = m_pCPInfo->bDoingCheckpoint;
		if (pCheckpointInfo->bRunning)
		{
			if (m_pCPInfo->uiStartTime)
			{
				uiCurrTime = FLM_GET_TIMER();

				uiElapTime = FLM_ELAPSED_TIME( uiCurrTime,
							m_pCPInfo->uiStartTime);
				pCheckpointInfo->ui32RunningTime = (FLMUINT32)FLM_TIMER_UNITS_TO_MILLI( uiElapTime);
			}
			else
			{
				pCheckpointInfo->ui32RunningTime = 0;
			}
			pCheckpointInfo->bForcingCheckpoint =
				m_pCPInfo->bForcingCheckpoint;
			if (m_pCPInfo->uiForceCheckpointStartTime)
			{
				uiCurrTime = FLM_GET_TIMER();
				uiElapTime = FLM_ELAPSED_TIME( uiCurrTime,
							m_pCPInfo->uiForceCheckpointStartTime);
				pCheckpointInfo->ui32ForceCheckpointRunningTime = 
					(FLMUINT32)FLM_TIMER_UNITS_TO_MILLI( uiElapTime);
			}
			else
			{
				pCheckpointInfo->ui32ForceCheckpointRunningTime = 0;
			}
			pCheckpointInfo->ui32ForceCheckpointReason =
				(FLMUINT32)m_pCPInfo->iForceCheckpointReason;
			pCheckpointInfo->bWritingDataBlocks =
				m_pCPInfo->bWritingDataBlocks;
			pCheckpointInfo->ui32LogBlocksWritten =
				(FLMUINT32)m_pCPInfo->uiLogBlocksWritten;
			pCheckpointInfo->ui32DataBlocksWritten =
				(FLMUINT32)m_pCPInfo->uiDataBlocksWritten;
		}
		pCheckpointInfo->ui32BlockSize = (FLMUINT32)m_uiBlockSize;
		pCheckpointInfo->ui32DirtyCacheBytes =
			(FLMUINT32)(m_uiDirtyCacheCount * m_uiBlockSize);
		if (m_pCPInfo->uiStartWaitTruncateTime)
		{
			uiCurrTime = FLM_GET_TIMER();

			uiElapTime = FLM_ELAPSED_TIME( uiCurrTime,
						m_pCPInfo->uiStartWaitTruncateTime);
			pCheckpointInfo->ui32WaitTruncateTime = 
				(FLMUINT32)FLM_TIMER_UNITS_TO_MILLI( uiElapTime);
		}
		else
		{
			pCheckpointInfo->ui32WaitTruncateTime = 0;
		}
	}
}

/****************************************************************************
Desc: Retrieves the Checkpoint info for the database.
*****************************************************************************/
void XFLAPI F_Db::getCheckpointInfo(
	XFLM_CHECKPOINT_INFO *	pCheckpointInfo)
{
	m_pDatabase->lockMutex();
	m_pDatabase->getCPInfo( pCheckpointInfo);
	m_pDatabase->unlockMutex();
}

/****************************************************************************
Desc:	Returns current RFL file number
****************************************************************************/
RCODE XFLAPI F_Db::getRflFileNum(
	FLMUINT *	puiRflFileNum
	)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bStartedTrans = FALSE;
	FLMUINT	uiLastCPFile;
	FLMUINT	uiLastTransFile;

	if (m_eTransType == XFLM_READ_TRANS)
	{
		rc = RC_SET( NE_XFLM_ILLEGAL_TRANS_OP);
		goto Exit;
	}
	else if (m_eTransType != XFLM_NO_TRANS)
	{
		if (!okToCommitTrans())
		{
			rc = RC_SET( NE_XFLM_ABORT_TRANS);
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = beginTrans( XFLM_UPDATE_TRANS)))
		{
			goto Exit;
		}
		bStartedTrans = TRUE;
	}

	// Get the CP and last trans RFL file numbers.  Need to
	// return the higher of the two.  No need to lock the
	// mutex because we are in an update transaction.

	uiLastCPFile =
		(FLMUINT)m_pDatabase->m_uncommittedDbHdr.ui32RflLastCPFileNum;

	uiLastTransFile =
		(FLMUINT)m_pDatabase->m_uncommittedDbHdr.ui32RflCurrFileNum;

	*puiRflFileNum = uiLastCPFile > uiLastTransFile
								 ? uiLastCPFile
								 : uiLastTransFile;

Exit:

	if (bStartedTrans)
	{
		abortTrans();
	}

	return( rc);
}

/****************************************************************************
Desc:	Returns highest not used RFL file number
****************************************************************************/
RCODE XFLAPI F_Db::getHighestNotUsedRflFileNum(
	FLMUINT *	puiHighestNotUsedRflFileNum
	)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bStartedTrans = FALSE;
	FLMUINT	uiLastCPFile;
	FLMUINT	uiLastTransFile;

	if (m_eTransType == XFLM_READ_TRANS)
	{
		rc = RC_SET( NE_XFLM_ILLEGAL_TRANS_OP);
		goto Exit;
	}
	else if (m_eTransType != XFLM_NO_TRANS)
	{
		if (!okToCommitTrans())
		{
			rc = RC_SET( NE_XFLM_ABORT_TRANS);
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = beginTrans( XFLM_UPDATE_TRANS)))
		{
			goto Exit;
		}
		bStartedTrans = TRUE;
	}

	// Get the CP and last trans RFL file numbers.  Need to
	// return the lower of the two minus 1.

	uiLastCPFile =
		(FLMUINT)m_pDatabase->m_uncommittedDbHdr.ui32RflLastCPFileNum;

	uiLastTransFile =
		(FLMUINT)m_pDatabase->m_uncommittedDbHdr.ui32RflCurrFileNum;

	*puiHighestNotUsedRflFileNum =
		(FLMUINT)((uiLastCPFile < uiLastTransFile
					? uiLastCPFile
					: uiLastTransFile) - 1);
Exit:

	if (bStartedTrans)
	{
		abortTrans();
	}

	return( rc);
}

/****************************************************************************
Desc:	Returns RFL file size limits for the database
****************************************************************************/
RCODE XFLAPI F_Db::getRflFileSizeLimits(
	FLMUINT *	puiRflMinFileSize,
	FLMUINT *	puiRflMaxFileSize
	)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bStartedTrans = FALSE;

	if (m_eTransType == XFLM_READ_TRANS)
	{
		rc = RC_SET( NE_XFLM_ILLEGAL_TRANS_OP);
		goto Exit;
	}
	else if (m_eTransType != XFLM_NO_TRANS)
	{
		if (!okToCommitTrans())
		{
			rc = RC_SET( NE_XFLM_ABORT_TRANS);
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = beginTrans( XFLM_UPDATE_TRANS)))
		{
			goto Exit;
		}
		bStartedTrans = TRUE;
	}

	if (puiRflMinFileSize)
	{
		*puiRflMinFileSize =
			(FLMUINT)m_pDatabase->m_uncommittedDbHdr.ui32RflMinFileSize;
	}
	if (puiRflMaxFileSize)
	{
		*puiRflMaxFileSize =
			(FLMUINT)m_pDatabase->m_uncommittedDbHdr.ui32RflMaxFileSize;
	}

Exit:

	if (bStartedTrans)
	{
		abortTrans();
	}

	return( rc);
}

/****************************************************************************
Desc:	Returns RFL keep flag for the database
****************************************************************************/
RCODE XFLAPI F_Db::getRflKeepFlag(
	FLMBOOL *	pbKeep
	)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bStartedTrans = FALSE;

	if (m_eTransType == XFLM_READ_TRANS)
	{
		rc = RC_SET( NE_XFLM_ILLEGAL_TRANS_OP);
		goto Exit;
	}
	else if (m_eTransType != XFLM_NO_TRANS)
	{
		if (!okToCommitTrans())
		{
			rc = RC_SET( NE_XFLM_ABORT_TRANS);
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = beginTrans( XFLM_UPDATE_TRANS)))
		{
			goto Exit;
		}
		bStartedTrans = TRUE;
	}

	*pbKeep = m_pDatabase->m_uncommittedDbHdr.ui8RflKeepFiles
				 ? TRUE
				 : FALSE;

Exit:

	if (bStartedTrans)
	{
		abortTrans();
	}

	return( rc);
}

/****************************************************************************
Desc:	Returns last backup transaction ID for the database
****************************************************************************/
RCODE XFLAPI F_Db::getLastBackupTransID(
	FLMUINT64 *	pui64LastBackupTransID
	)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bStartedTrans = FALSE;

	if (m_eTransType == XFLM_READ_TRANS)
	{
		rc = RC_SET( NE_XFLM_ILLEGAL_TRANS_OP);
		goto Exit;
	}
	else if (m_eTransType != XFLM_NO_TRANS)
	{
		if (!okToCommitTrans())
		{
			rc = RC_SET( NE_XFLM_ABORT_TRANS);
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = beginTrans( XFLM_UPDATE_TRANS)))
		{
			goto Exit;
		}
		bStartedTrans = TRUE;
	}

	*pui64LastBackupTransID =
				m_pDatabase->m_uncommittedDbHdr.ui64LastBackupTransID;

Exit:

	if (bStartedTrans)
	{
		abortTrans();
	}

	return( rc);
}

/****************************************************************************
Desc:	Returns blocks changed since the last backup for the database
****************************************************************************/
RCODE XFLAPI F_Db::getBlocksChangedSinceBackup(
	FLMUINT *	puiBlocksChangedSinceBackup
	)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bStartedTrans = FALSE;

	if (m_eTransType == XFLM_READ_TRANS)
	{
		rc = RC_SET( NE_XFLM_ILLEGAL_TRANS_OP);
		goto Exit;
	}
	else if (m_eTransType != XFLM_NO_TRANS)
	{
		if (!okToCommitTrans())
		{
			rc = RC_SET( NE_XFLM_ABORT_TRANS);
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = beginTrans( XFLM_UPDATE_TRANS)))
		{
			goto Exit;
		}
		bStartedTrans = TRUE;
	}

	*puiBlocksChangedSinceBackup =
			(FLMUINT)m_pDatabase->m_uncommittedDbHdr.ui32BlksChangedSinceBackup;

Exit:

	if (bStartedTrans)
	{
		abortTrans();
	}

	return( rc);
}

/****************************************************************************
Desc:	Returns the auto-turn-off-keep-RFL flag for the database
****************************************************************************/
RCODE XFLAPI F_Db::getAutoTurnOffKeepRflFlag(
	FLMBOOL *	pbAutoTurnOff
	)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bStartedTrans = FALSE;

	if (m_eTransType == XFLM_READ_TRANS)
	{
		rc = RC_SET( NE_XFLM_ILLEGAL_TRANS_OP);
		goto Exit;
	}
	else if (m_eTransType != XFLM_NO_TRANS)
	{
		if (!okToCommitTrans())
		{
			rc = RC_SET( NE_XFLM_ABORT_TRANS);
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = beginTrans( XFLM_UPDATE_TRANS)))
		{
			goto Exit;
		}
		bStartedTrans = TRUE;
	}

	*pbAutoTurnOff = m_pDatabase->m_uncommittedDbHdr.ui8RflAutoTurnOffKeep
							? TRUE
							: FALSE;

Exit:

	if (bStartedTrans)
	{
		abortTrans();
	}

	return( rc);
}

/****************************************************************************
Desc:	Returns the keep aborted transactions in RFL flag for the database
****************************************************************************/
RCODE XFLAPI F_Db::getKeepAbortedTransInRflFlag(
	FLMBOOL *	pbKeep
	)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bStartedTrans = FALSE;

	if (m_eTransType == XFLM_READ_TRANS)
	{
		rc = RC_SET( NE_XFLM_ILLEGAL_TRANS_OP);
		goto Exit;
	}
	else if (m_eTransType != XFLM_NO_TRANS)
	{
		if (!okToCommitTrans())
		{
			rc = RC_SET( NE_XFLM_ABORT_TRANS);
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = beginTrans( XFLM_UPDATE_TRANS)))
		{
			goto Exit;
		}
		bStartedTrans = TRUE;
	}

	*pbKeep = m_pDatabase->m_uncommittedDbHdr.ui8RflKeepAbortedTrans
				 ? TRUE
				 : FALSE;

Exit:

	if (bStartedTrans)
	{
		abortTrans();
	}

	return( rc);
}

/****************************************************************************
Desc:	Returns disk space usage for the database
****************************************************************************/
RCODE XFLAPI F_Db::getDiskSpaceUsage(
	FLMUINT64 *		pui64DataSize,
	FLMUINT64 *		pui64RollbackSize,
	FLMUINT64 *		pui64RflSize)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bStartedTrans = FALSE;
	FLMUINT			uiEndAddress;
	FLMUINT			uiLastFileNumber;
	FLMUINT64		ui64LastFileSize;
	char				szTmpName [F_PATH_MAX_SIZE];
	char				szRflDir [F_PATH_MAX_SIZE];
	IF_FileHdl *	pFileHdl = NULL;
	IF_DirHdl *		pDirHdl = NULL;

	if (m_eTransType == XFLM_READ_TRANS)
	{
		rc = RC_SET( NE_XFLM_ILLEGAL_TRANS_OP);
		goto Exit;
	}
	else if (m_eTransType != XFLM_NO_TRANS)
	{
		if (!okToCommitTrans())
		{
			rc = RC_SET( NE_XFLM_ABORT_TRANS);
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = beginTrans( XFLM_UPDATE_TRANS)))
		{
			goto Exit;
		}
		bStartedTrans = TRUE;
	}

	// See if they want the database files sizes.

	if (pui64DataSize)
	{
		uiEndAddress = m_uiLogicalEOF;
		uiLastFileNumber = FSGetFileNumber( uiEndAddress);

		// Last file number better be in the proper range.

		flmAssert( uiLastFileNumber >= 1 &&
					  uiLastFileNumber <= MAX_DATA_BLOCK_FILE_NUMBER);

		// Get the actual size of the last file.

		if (RC_BAD( rc = m_pSFileHdl->getFileSize( uiLastFileNumber,
										&ui64LastFileSize)))
		{
			if (rc == NE_FLM_IO_PATH_NOT_FOUND ||
				 rc == NE_FLM_IO_INVALID_FILENAME)
			{
				if (uiLastFileNumber > 1)
				{
					rc = NE_XFLM_OK;
					ui64LastFileSize = 0;
				}
				else
				{

					// Should always be a data file #1

					RC_UNEXPECTED_ASSERT( rc);
					goto Exit;
				}
			}
			else
			{
				goto Exit;
			}
		}

		// One of two situations exists with respect to the last
		// file: 1) it has not been fully written out yet (blocks
		// are still cached, or 2) it has been written out and
		// extended beyond what the logical EOF shows.  We want
		// the larger of these two possibilities.

		if (FSGetFileOffset( uiEndAddress) > ui64LastFileSize)
		{
			ui64LastFileSize = FSGetFileOffset( uiEndAddress);
		}

		if (uiLastFileNumber == 1)
		{

			// Only one file - use last file's size.

			*pui64DataSize = ui64LastFileSize;
		}
		else
		{

			// Size is the sum of full size for all files except the last one,
			// plus the calculated (or actual) size of the last one.

			(*pui64DataSize) = (FLMUINT64)(uiLastFileNumber - 1) *
							(FLMUINT64)m_pDatabase->m_uiMaxFileSize +
							ui64LastFileSize;
		}
	}

	// See if they want the rollback files sizes.

	if (pui64RollbackSize)
	{
		uiEndAddress = (FLMUINT)m_pDatabase->m_uncommittedDbHdr.ui32RblEOF;
		uiLastFileNumber = FSGetFileNumber( uiEndAddress);

		// Last file number better be in the proper range.

		flmAssert( !uiLastFileNumber ||
					  (uiLastFileNumber >=
							FIRST_LOG_BLOCK_FILE_NUMBER &&
					   uiLastFileNumber <=
							MAX_LOG_BLOCK_FILE_NUMBER));

		// Get the size of the last file number.

		if (RC_BAD( rc = m_pSFileHdl->getFileSize( uiLastFileNumber,
										&ui64LastFileSize)))
		{
			if (rc == NE_FLM_IO_PATH_NOT_FOUND ||
				 rc == NE_FLM_IO_INVALID_FILENAME)
			{
				if (uiLastFileNumber)
				{
					rc = NE_XFLM_OK;
					ui64LastFileSize = 0;
				}
				else
				{

					// Should always have rollback file #0

					RC_UNEXPECTED_ASSERT( rc);
					goto Exit;
				}
			}
			else
			{
				goto Exit;
			}
		}

		// If the EOF offset for the last file is greater than the
		// actual file size, use it instead of the actual file size.

		if (FSGetFileOffset( uiEndAddress) > ui64LastFileSize)
		{
			ui64LastFileSize = FSGetFileOffset( uiEndAddress);
		}

		// Special case handling here because rollback file numbers start with
		// zero and then skip to a file number that is one beyond the
		// highest data file number - so the calculation for file size needs
		// to account for this.

		if (!uiLastFileNumber)
		{
			*pui64RollbackSize = ui64LastFileSize;
		}
		else
		{
			FLMUINT	uiFirstLogFileNum = FIRST_LOG_BLOCK_FILE_NUMBER;

			// Add full size of file zero plus a full size for every file
			// except the last one.

			(*pui64RollbackSize) = (FLMUINT64)(uiLastFileNumber -
															uiFirstLogFileNum + 1) *
							(FLMUINT64)m_pDatabase->m_uiMaxFileSize +
							ui64LastFileSize;
		}
	}

	// See if they want the roll-forward log file sizes.

	if (pui64RflSize)
	{
		char *	pszDbFileName = m_pDatabase->m_pszDbPath;

		*pui64RflSize = 0;

		// Scan the RFL directory for
		// RFL files.  The call below to rflGetDirAndPrefix is done
		// to get the prefix.  It will not return the correct
		// RFL directory name, because we are passing in a NULL
		// RFL directory path (which may or may not be correct).
		// That's OK, because we get the RFL directory directly
		// from the F_Rfl object anyway.

		if (RC_BAD( rc = rflGetDirAndPrefix( pszDbFileName,
									NULL, szRflDir)))
		{
			goto Exit;
		}

		// We need to get the RFL directory from the F_Rfl object.

		m_pDatabase->lockMutex();
		f_strcpy( szRflDir, m_pDatabase->m_pRfl->getRflDirPtr());
		m_pDatabase->unlockMutex();

		// See if the directory exists.  If not, we are done.

		if (gv_XFlmSysData.pFileSystem->isDir( szRflDir))
		{

			// Open the directory and scan for RFL files.

			if (RC_BAD( rc = gv_XFlmSysData.pFileSystem->openDir( szRflDir,
											"*", &pDirHdl)))
			{
				goto Exit;
			}
			for (;;)
			{
				if (RC_BAD( rc = pDirHdl->next()))
				{
					if (rc == NE_FLM_IO_NO_MORE_FILES)
					{
						rc = NE_XFLM_OK;
						break;
					}
					else
					{
						goto Exit;
					}
				}
				pDirHdl->currentItemPath( szTmpName);

				// If the item looks like an RFL file name, get
				// its size.

				if (!pDirHdl->currentItemIsDir() &&
					  rflGetFileNum( szTmpName, &uiLastFileNumber))
				{

					// Open the file and get its size.

					if (RC_BAD( rc = gv_XFlmSysData.pFileSystem->openFile(
							szTmpName, gv_XFlmSysData.uiFileOpenFlags, &pFileHdl)))
					{
						if (rc == NE_FLM_IO_PATH_NOT_FOUND ||
							 rc == NE_FLM_IO_INVALID_FILENAME)
						{
							rc = NE_XFLM_OK;
							ui64LastFileSize = 0;
						}
						else
						{
							goto Exit;
						}
					}
					else
					{
						if (RC_BAD( rc = pFileHdl->size( &ui64LastFileSize)))
						{
							goto Exit;
						}
					}
					if (pFileHdl)
					{
						pFileHdl->Release();
						pFileHdl = NULL;
					}
					(*pui64RflSize) += ui64LastFileSize;
				}
			}
		}
	}

Exit:

	if (pFileHdl)
	{
		pFileHdl->Release();
	}

	if (pDirHdl)
	{
		pDirHdl->Release();
	}

	if (bStartedTrans)
	{
		abortTrans();
	}

	return( rc);
}

/****************************************************************************
Desc:	Returns the next incremental backup sequence number for the database
****************************************************************************/
RCODE XFLAPI F_Db::getNextIncBackupSequenceNum(
	FLMUINT *	puiNextIncBackupSequenceNum
	)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bStartedTrans = FALSE;

	if (m_eTransType == XFLM_READ_TRANS)
	{
		rc = RC_SET( NE_XFLM_ILLEGAL_TRANS_OP);
		goto Exit;
	}
	else if (m_eTransType != XFLM_NO_TRANS)
	{
		if (!okToCommitTrans())
		{
			rc = RC_SET( NE_XFLM_ABORT_TRANS);
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = beginTrans( XFLM_UPDATE_TRANS)))
		{
			goto Exit;
		}
		bStartedTrans = TRUE;
	}

	*puiNextIncBackupSequenceNum =
			(FLMUINT)m_pDatabase->m_uncommittedDbHdr.ui32IncBackupSeqNum;

Exit:

	if (bStartedTrans)
	{
		abortTrans();
	}

	return( rc);
}

/****************************************************************************
Desc:	Returns list of lock waiters in an object that allows caller to
		iterate through the list.
****************************************************************************/
RCODE XFLAPI F_Db::getLockWaiters(
	IF_LockInfoClient *	pLockInfo
	)
{
	RCODE	rc = NE_XFLM_OK;

	if (m_pDatabase->m_pDatabaseLockObj)
	{
		rc = m_pDatabase->m_pDatabaseLockObj->getLockInfo( pLockInfo);
	}
	else
	{
		pLockInfo->setLockCount( 0);
	}
	return( rc);
}

/****************************************************************************
Desc:	Returns RFL directory for the database
****************************************************************************/
void XFLAPI F_Db::getRflDir(
	char *	pszRflDir
	)
{
	m_pDatabase->lockMutex();
	f_strcpy( pszRflDir, m_pDatabase->m_pRfl->getRflDirPtr());
	m_pDatabase->unlockMutex();
}

/****************************************************************************
Desc:	Returns database serial number
****************************************************************************/
void XFLAPI F_Db::getSerialNumber(
	char *	pucSerialNumber)
{
	m_pDatabase->lockMutex();
	f_memcpy( pucSerialNumber, m_pDatabase->m_lastCommittedDbHdr.ucDbSerialNum,
		XFLM_SERIAL_NUM_SIZE);
	m_pDatabase->unlockMutex();
}
