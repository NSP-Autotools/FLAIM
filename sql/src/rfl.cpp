// Desc:	This module contains routine for roll forward logging.
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

#include "flaimsys.h"

#define MOD_512( uiNum)							(FLMUINT)((uiNum) & 511)
#define ON_512_BYTE_BOUNDARY( uiNum)		(!MOD_512(uiNum))
#define ROUND_DOWN_TO_NEAREST_512( uiNum)	\
	(FLMUINT)((uiNum) & (~((FLMUINT)511)))

/********************************************************************
Desc:
*********************************************************************/
FINLINE FLMBOOL F_Rfl::useDataOnlyBlocks(
	F_Db *			pDb,
	FLMUINT			uiDataLen)
{
	if( uiDataLen > (pDb->m_pDatabase->m_uiBlockSize * 8) / 5)
	{
		return( TRUE);
	}
	else
	{
		return( FALSE);
	}
}

/********************************************************************
Desc:
*********************************************************************/
F_Rfl::F_Rfl()
{
	m_pDatabase = NULL;
	m_hBufMutex = F_MUTEX_NULL;
	m_pCommitBuf = NULL;
	m_pCurrentBuf = NULL;
	m_uiRflWriteBufs = DEFAULT_RFL_WRITE_BUFFERS;
	m_uiBufferSize = DEFAULT_RFL_BUFFER_SIZE;
	f_memset( &m_Buf1, 0, sizeof( m_Buf1));
	f_memset( &m_Buf2, 0, sizeof( m_Buf2));
	m_bKeepRflFiles = FALSE;
	m_uiRflMinFileSize = SFLM_DEFAULT_MIN_RFL_FILE_SIZE;
	m_uiRflMaxFileSize = SFLM_DEFAULT_MAX_RFL_FILE_SIZE;
	m_pFileHdl = NULL;
	m_uiLastRecoverFileNum = 0;
	f_memset( m_ucCurrSerialNum, 0, sizeof( m_ucCurrSerialNum));
	m_uiTransStartFile = 0;
	m_uiTransStartAddr = 0;
	m_ui64CurrTransID = 0;
	m_ui64LastTransID = 0;
	m_ui64LastLoggedCommitTransID = 0;
	m_uiOperCount = 0;
	m_uiRflReadOffset = 0;
	m_uiFileEOF = 0;
	m_pRestore = NULL;
	m_pRestoreStatus = NULL;
	f_memset( m_szRflDir, 0, sizeof( m_szRflDir));
	m_bRflDirSameAsDb = FALSE;
	m_bCreateRflDir = FALSE;
	f_memset( m_ucNextSerialNum, 0, sizeof( m_ucNextSerialNum));
	m_bRflVolumeOk = TRUE;
	m_bRflVolumeFull = FALSE;
	m_uiLastLfNum = 0;
	m_eLastLfType = SFLM_LF_INVALID;
	m_pIxCompareObject = NULL;
	m_pCompareObject = NULL;
	m_uiDisableCount = 0;
	m_tmpPool.poolInit( 2048);
}

/********************************************************************
Desc:		Destructor
*********************************************************************/
F_Rfl::~F_Rfl()
{
	if (m_Buf1.pIOBuffer)
	{
		m_Buf1.pIOBuffer->Release();
		m_Buf1.pIOBuffer = NULL;
	}
	if (m_Buf2.pIOBuffer)
	{
		m_Buf2.pIOBuffer->Release();
		m_Buf2.pIOBuffer = NULL;
	}

	if( m_Buf1.pBufferMgr)
	{
		flmAssert( !m_Buf1.pBufferMgr->isIOPending());
		m_Buf1.pBufferMgr->Release();
		m_Buf1.pBufferMgr = NULL;
	}

	if( m_Buf2.pBufferMgr)
	{
		flmAssert( !m_Buf2.pBufferMgr->isIOPending());
		m_Buf2.pBufferMgr->Release();
		m_Buf2.pBufferMgr = NULL;
	}

	if (m_hBufMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hBufMutex);
	}

	if (m_pFileHdl)
	{
		m_pFileHdl->closeFile();
		m_pFileHdl->Release();
		m_pFileHdl = NULL;
		m_pDatabase = NULL;
	}
	
	if (m_pIxCompareObject)
	{
		m_pIxCompareObject->Release();
	}
	m_tmpPool.poolFree();
}

/********************************************************************
Desc:		Returns a boolean indicating whether or not we are at
			the end of the RFL log - will only be TRUE when we are
			doing recovery.
*********************************************************************/
FLMBOOL F_Rfl::atEndOfLog( void)
{
	return( (!m_pRestore &&
				m_uiFileEOF &&
				m_pCurrentBuf->uiRflFileOffset +
				m_pCurrentBuf->uiRflBufBytes >= m_uiFileEOF &&
				m_uiRflReadOffset == m_pCurrentBuf->uiRflBufBytes &&
				m_pCurrentBuf->uiCurrFileNum == m_uiLastRecoverFileNum)
			  ? TRUE
			  : FALSE);
}

/********************************************************************
Desc:	Gets the base RFL file name - does not have directory part.
		This needs to be separate from the F_Rfl object so it can
		be called without having to instantiate and set up an F_Rfl
		object.
*********************************************************************/
void rflGetBaseFileName(
	FLMUINT		uiFileNum,
	char *		pszBaseNameOut,
	FLMUINT *	puiFileNameBufSize,
	FLMBOOL *	pbNameTruncated)
{
	FLMUINT		uiCnt;
	FLMUINT		uiDigit;
	char			szTmpBuf [14];
	char *		pszTmp = &szTmpBuf [0];

	// Output as eight digit hex number

	uiCnt = 0;
	pszTmp += 7;
	while (uiCnt < 8)
	{
		uiDigit = (FLMUINT)(uiFileNum & 0xF);
		uiFileNum >>= 4;
		if (uiDigit <= 9)
		{
			uiDigit += NATIVE_ZERO;
		}
		else
		{
			uiDigit += (NATIVE_LOWER_A - 10);
		}
		*pszTmp = (FLMBYTE)uiDigit;
		pszTmp--;
		uiCnt++;
	}

	// Skip to end of digits and append ".log" to name

	f_strcpy( pszTmp + 9, ".log");
	if (*puiFileNameBufSize >= 13)
	{
		*puiFileNameBufSize = 12;
		f_strcpy( pszBaseNameOut, szTmpBuf);
		if (pbNameTruncated)
		{
			*pbNameTruncated = FALSE;
		}
	}
	else
	{
		flmAssert( *puiFileNameBufSize);
		(*puiFileNameBufSize)--;
		if (*puiFileNameBufSize)
		{
			f_memcpy( pszBaseNameOut, szTmpBuf, *puiFileNameBufSize);
		}
		pszBaseNameOut [*puiFileNameBufSize] = 0;
		if (pbNameTruncated)
		{
			*pbNameTruncated = TRUE;
		}
	}
}

/********************************************************************
Desc:	Generates the full roll forward log file name.
*********************************************************************/
void F_Rfl::getFullRflFileName(
	FLMUINT		uiFileNum,
	char *		pszRflFileName,
	FLMUINT *	puiFileNameBufSize,
	FLMBOOL *	pbNameTruncated)
{
	FLMUINT	uiBaseNameSize;
	FLMUINT	uiLen = f_strlen( m_szRflDir);
	FLMBOOL	bNameTruncated = FALSE;

	// Must at least be room for a null byte to terminate the string.

	flmAssert( *puiFileNameBufSize);
	if (uiLen > *puiFileNameBufSize - 1)
	{
		uiLen = *puiFileNameBufSize - 1;
		if (uiLen)
		{
			f_memcpy( pszRflFileName, m_szRflDir, uiLen);
		}
		bNameTruncated = TRUE;
		goto Exit;
	}

	// Get the directory name.

	if (uiLen)
	{
		f_memcpy( pszRflFileName, m_szRflDir, uiLen);

		// See if we need to append a slash.

#ifdef FLM_UNIX
		if (m_szRflDir [uiLen - 1] != '/')
#else
		if (m_szRflDir [uiLen - 1] != '/' &&
			 m_szRflDir [uiLen - 1] != '\\')
#endif
		{

			// See if we have room for one more character, plus null

			if (uiLen == *puiFileNameBufSize - 1)
			{
				bNameTruncated = TRUE;
				goto Exit;
			}
#ifdef FLM_UNIX
			pszRflFileName [uiLen] = '/';
#else
			pszRflFileName [uiLen] = '\\';
#endif
			uiLen++;
		}
	}

	// See if there is room for at least one more byte plus a
	// null byte.

	if (uiLen == *puiFileNameBufSize - 1)
	{
		bNameTruncated = TRUE;
		goto Exit;
	}

	// Get the base RFL file name.

	uiBaseNameSize = *puiFileNameBufSize - uiLen;
	rflGetBaseFileName( uiFileNum, pszRflFileName + uiLen,
					&uiBaseNameSize, &bNameTruncated);
	uiLen += uiBaseNameSize;

Exit:

	pszRflFileName [uiLen] = 0;
	*puiFileNameBufSize = uiLen;
	if (pbNameTruncated)
	{
		*pbNameTruncated = bNameTruncated;
	}
}

/********************************************************************
Desc:		Positions to the offset specified in the RFL file.
*********************************************************************/
RCODE F_Rfl::positionTo(
	FLMUINT		uiFileOffset)
{
	RCODE		rc = NE_SFLM_OK;
	FLMUINT	uiBytesToRead;
	FLMUINT	uiBytesRead;

	// Should never be attempting to position to something less
	// than 512 - the header is stored in the first 512 bytes.

	flmAssert( uiFileOffset >= 512);

	// If the position is within our current buffer, see if we
	// can adjust things without having to go back and re-read
	// the buffer from disk.

	if (m_pCurrentBuf->uiRflBufBytes &&
		 uiFileOffset >= m_pCurrentBuf->uiRflFileOffset &&
		 uiFileOffset <= m_pCurrentBuf->uiRflFileOffset +
							  m_pCurrentBuf->uiRflBufBytes)
	{

		// Whatever is in the buffer beyond uiFileOffset is irrelevant
		// and can be discarded.

		m_pCurrentBuf->uiRflBufBytes = uiFileOffset -
													m_pCurrentBuf->uiRflFileOffset;
	}
	else
	{

		// Populate the buffer from the 512 byte boundary that is just
		// before the offset we are trying to position to.

		uiBytesToRead = MOD_512( uiFileOffset);
		m_pCurrentBuf->uiRflFileOffset = ROUND_DOWN_TO_NEAREST_512( uiFileOffset);
		m_pCurrentBuf->uiRflBufBytes = MOD_512( uiFileOffset);
		if (m_pCurrentBuf->uiRflBufBytes)
		{
			if (RC_BAD( rc = m_pFileHdl->read( 
				m_pCurrentBuf->uiRflFileOffset, m_pCurrentBuf->uiRflBufBytes,
				m_pCurrentBuf->pIOBuffer->getBufferPtr(), &uiBytesRead)))
			{
				if (rc == NE_FLM_IO_END_OF_FILE)
				{
					rc = RC_SET( NE_SFLM_NOT_RFL);
				}
				else
				{
					m_bRflVolumeOk = FALSE;
				}
				goto Exit;
			}
			else if (uiBytesRead < m_pCurrentBuf->uiRflBufBytes)
			{
				rc = RC_SET( NE_SFLM_NOT_RFL);
				goto Exit;
			}
		}
	}

Exit:

	return( rc);
}

/********************************************************************
Desc:	Get the ACTUAL RFL directory, using as input parameters the
		database version, the name of the database, and the
		user specified RFL directory.  Also return the database
		prefix.
*********************************************************************/
RCODE rflGetDirAndPrefix(
	const char *	pszDbFileName,
	const char *	pszRflDirIn,
	char *			pszRflDirOut)
{
	RCODE			rc = NE_SFLM_OK;
	char			szDbPath [F_PATH_MAX_SIZE];
	char			szBaseName [F_FILENAME_SIZE];
	char			szPrefix [F_FILENAME_SIZE];
	F_DbSystem	dbSystem;

	// Parse the database name into directory and base name

	if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->pathReduce( pszDbFileName,
							szDbPath, szBaseName)))
	{
		goto Exit;
	}

	// Get the base path

	flmGetDbBasePath( szPrefix, szBaseName, NULL);

	// Determine the RFL directory.  If one was
	// specified, it is whatever was specified.
	// Otherwise, it is relative to the database
	// directory.

	if (pszRflDirIn && *pszRflDirIn)
	{
		f_strcpy( pszRflDirOut, pszRflDirIn);
	}
	else
	{
		f_strcpy( pszRflDirOut, szDbPath);
	}

	f_strcpy( szBaseName, szPrefix);
	f_strcat( szBaseName, ".rfl");
	gv_SFlmSysData.pFileSystem->pathAppend( pszRflDirOut, szBaseName);

Exit:

	return( rc);
}

/********************************************************************
Desc:		Set the RFL directory.  If pszRflDir is NULL or empty string,
			the RFL directory is set to the same directory as the
			database.
*********************************************************************/
RCODE F_Rfl::setRflDir(
	const char *	pszRflDir)
{
	// Better have set up the F_Database pointer.

	flmAssert( m_pDatabase != NULL);

	m_bRflDirSameAsDb = (!pszRflDir || !(*pszRflDir))
							  ? TRUE
							  : FALSE;

	flmAssert( m_pDatabase->m_lastCommittedDbHdr.ui32DbVersion);

	m_bCreateRflDir = TRUE;
	return( rflGetDirAndPrefix( m_pDatabase->m_pszDbPath,
						pszRflDir, m_szRflDir));
}

/********************************************************************
Desc:	Gets an RFL file name - based on DB name and RFL directory.
*********************************************************************/
RCODE rflGetFileName(
	const char *	pszDbName,
	const char *	pszRflDir,
	FLMUINT			uiFileNum,
	char *			pszRflFileName)
{
	RCODE			rc = NE_SFLM_OK;
	char			szBaseName [F_FILENAME_SIZE];
	FLMUINT		uiBaseNameSize;

	// Get the full RFL file name.

	if (RC_BAD( rc = rflGetDirAndPrefix( pszDbName, pszRflDir,
								pszRflFileName)))
	{
		goto Exit;
	}
	
	uiBaseNameSize = sizeof( szBaseName);
	rflGetBaseFileName( uiFileNum, szBaseName, &uiBaseNameSize, NULL);
	
	if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->pathAppend( 
		pszRflFileName, szBaseName)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}

/********************************************************************
Desc:	Gets an RFL file number from the RFL file name.
*********************************************************************/
FLMBOOL rflGetFileNum(
	const char *	pszRflFileName,
	FLMUINT *		puiFileNum)
{
	FLMBOOL		bGotNum = FALSE;
	char			szDir[F_PATH_MAX_SIZE];
	char			szBaseName[F_FILENAME_SIZE];
	char *		pszTmp;
	FLMUINT		uiCharCnt;

	if( RC_BAD( gv_SFlmSysData.pFileSystem->pathReduce( 
		pszRflFileName, szDir, szBaseName)))
	{
		goto Exit;
	}

	// See if it has a .log extension.

	pszTmp = &szBaseName [0];
	while (*pszTmp && *pszTmp != '.')
	{
		pszTmp++;
	}

	// If we do not have a .log extension, it is not a legitimate
	// RFL file.

	if (f_stricmp( pszTmp, ".log") != 0)
	{
		goto Exit;
	}

	// Parse out the name according to the rules for this DB version.

	*pszTmp = 0;	// Set period to zero
	pszTmp = &szBaseName [0];
	*puiFileNum = 0;
	uiCharCnt = 0;

	// Name up to the period should be a hex number

	while (*pszTmp)
	{
		(*puiFileNum) <<= 4;
		if (*pszTmp >= NATIVE_ZERO && *pszTmp <= NATIVE_NINE)
		{
			*puiFileNum += (FLMUINT)(*pszTmp - NATIVE_ZERO);
		}
		else if (*pszTmp >= NATIVE_LOWER_A && *pszTmp <= NATIVE_LOWER_F)
		{
			*puiFileNum += ((FLMUINT)(*pszTmp - NATIVE_LOWER_A) + 10);
		}
		else if (*pszTmp >= NATIVE_UPPER_A && *pszTmp <= NATIVE_UPPER_F)
		{
			*puiFileNum += ((FLMUINT)(*pszTmp - NATIVE_UPPER_A) + 10);
		}
		else
		{
			goto Exit;	// Not a hex number
		}
		uiCharCnt++;
		pszTmp++;
	}

	// Better have been exactly 8 hex digits.

	bGotNum = (FLMBOOL)((uiCharCnt == 8)
								? TRUE
								: FALSE);
Exit:

	return( bGotNum);
}

/********************************************************************
Desc:		Sets up the RFL object - associating with a file, etc.
*********************************************************************/
RCODE F_Rfl::setup(
	F_Database *		pDatabase,
	const char *		pszRflDir)
{
	RCODE		rc = NE_SFLM_OK;

	// Better not already be associated with an F_Database object

	flmAssert( m_pDatabase == NULL);
	m_pDatabase = pDatabase;

	// Allocate memory for the RFL buffers

#ifndef FLM_UNIX
	if (!gv_SFlmSysData.bOkToDoAsyncWrites)
#endif
	{
		m_uiRflWriteBufs = 1;
		m_uiBufferSize =
			DEFAULT_RFL_WRITE_BUFFERS * DEFAULT_RFL_BUFFER_SIZE;
	}

	if (RC_BAD( rc = f_mutexCreate( &m_hBufMutex)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = FlmAllocIOBufferMgr( m_uiRflWriteBufs, 
		m_uiRflWriteBufs * m_uiBufferSize, TRUE, &m_Buf1.pBufferMgr)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FlmAllocIOBufferMgr( m_uiRflWriteBufs, 
		m_uiRflWriteBufs * m_uiBufferSize, TRUE, &m_Buf2.pBufferMgr)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = m_Buf1.pBufferMgr->getBuffer( m_uiBufferSize, 
		&m_Buf1.pIOBuffer)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_Buf2.pBufferMgr->getBuffer( m_uiBufferSize,
		&m_Buf2.pIOBuffer)))
	{
		goto Exit;
	}

	m_pCurrentBuf = &m_Buf1;
	m_pCurrentBuf->uiRflBufBytes = 0;

	// Set the RFL directory and prefix if necessary.

	if (RC_BAD( rc = setRflDir( pszRflDir)))
	{
		goto Exit;
	}
	
	// Set up the compare object for comparing index keys.

	if ((m_pIxCompareObject = f_new IXKeyCompare) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}
	
Exit:

	return( rc);
}

/********************************************************************
Desc:	Wait for the writes of a buffer to finish.  This routine assumes
		that the m_hBufMutex is locked when coming in.  It will ALWAYS
		unlock the mutex before exiting.
*********************************************************************/
RCODE F_Rfl::waitForWrites(
	F_SEM				hWaitSem,
	RFL_BUFFER *	pBuffer,
	FLMBOOL			bIsWriter)
{
	RCODE				rc = NE_SFLM_OK;
	RCODE				TempRc;
	RFL_WAITER		Waiter;
	FLMBOOL			bMutexLocked = TRUE;

	// Put self on the wait queue for the buffer.

	Waiter.uiThreadId = f_threadId();
	Waiter.bIsWriter = bIsWriter;
	Waiter.hESem = hWaitSem;

	// Note: rc better be changed to success or write error
	// by the process that signals us.

	rc = RC_SET( NE_SFLM_FAILURE);
	
	Waiter.pRc = &rc;
	Waiter.pNext = NULL;
	
	if (pBuffer->pLastWaiter)
	{
		pBuffer->pLastWaiter->pNext = &Waiter;
	}
	else
	{
		pBuffer->pFirstWaiter = &Waiter;
	
	}
	pBuffer->pLastWaiter = &Waiter;
	
	f_mutexUnlock( m_hBufMutex);
	bMutexLocked = FALSE;

	// Now just wait to be signaled.

	if (RC_BAD( TempRc = f_semWait( Waiter.hESem, F_WAITFOREVER)))
	{
		RC_UNEXPECTED_ASSERT( TempRc);
		rc = TempRc;
	}
	else
	{
		// Process that signaled us better set the rc to something
		// besides NE_SFLM_FAILURE.

		if (rc == NE_SFLM_FAILURE)
		{
			RC_UNEXPECTED_ASSERT( rc);
		}
	}

	if (bMutexLocked)
	{
		f_mutexUnlock( m_hBufMutex);
	}
	
	return( rc);
}

/********************************************************************
Desc:	If a commit is in progress, wait for it to finish.
*********************************************************************/
RCODE F_Rfl::waitForCommit(
	F_SEM 		hWaitSem)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBOOL		bMutexLocked = FALSE;

	// NOTE: If m_pCommitBuf is NULL it cannot be set to something
	// non-NULL except by this thread when this thread ends the
	// transaction.  So, there is no need to lock the mutex and
	// re-check if it is NULL.

	if (m_pCommitBuf)
	{
		f_mutexLock( m_hBufMutex);
		bMutexLocked = TRUE;

		// Check m_pCommitBuf again after locking mutex - may have
		// finished.

		if (m_pCommitBuf)
		{
			bMutexLocked = FALSE;
			rc = waitForWrites( hWaitSem, m_pCommitBuf, FALSE);
		}
	}

	if (bMutexLocked)
	{
		f_mutexUnlock( m_hBufMutex);
	}
	
	return( rc);
}

/********************************************************************
Desc:		Write out the header information for an RFL file.
*********************************************************************/
RCODE F_Rfl::writeHeader(
	FLMUINT		uiFileNum,
	FLMUINT		uiEof,
	FLMBYTE *	pucSerialNum,
	FLMBYTE *	pucNextSerialNum,
	FLMBOOL		bKeepSignature)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBYTE		ucBuf[ 512];
	FLMUINT		uiBytesWritten;

	flmAssert( m_pDatabase);
	flmAssert( m_pFileHdl);

	f_memset( ucBuf, 0, sizeof( ucBuf));
	f_memcpy( &ucBuf [RFL_NAME_POS], RFL_NAME, RFL_NAME_LEN);
	f_memcpy( &ucBuf [RFL_VERSION_POS], RFL_VERSION, RFL_VERSION_LEN);
	UD2FBA( (FLMUINT32)uiFileNum, &ucBuf [RFL_FILE_NUMBER_POS]);
	UD2FBA( (FLMUINT32)uiEof, &ucBuf [RFL_EOF_POS]);

	f_memcpy( &ucBuf [RFL_DB_SERIAL_NUM_POS],
		m_pDatabase->m_lastCommittedDbHdr.ucDbSerialNum,
		SFLM_SERIAL_NUM_SIZE);
	f_memcpy( &ucBuf [RFL_SERIAL_NUM_POS], pucSerialNum,
		SFLM_SERIAL_NUM_SIZE);
	f_memcpy( &ucBuf [RFL_NEXT_FILE_SERIAL_NUM_POS], pucNextSerialNum,
		SFLM_SERIAL_NUM_SIZE);
	f_strcpy( (char *)&ucBuf [RFL_KEEP_SIGNATURE_POS],
		(char *)((bKeepSignature)
						? RFL_KEEP_SIGNATURE
						: RFL_NOKEEP_SIGNATURE));

	// Write out the header

	if (RC_BAD( rc = m_pFileHdl->write( 0L, 512, ucBuf, &uiBytesWritten)))
	{
		// Remap disk full error

		if (rc == NE_FLM_IO_DISK_FULL)
		{
			rc = RC_SET( NE_SFLM_RFL_DISK_FULL);
			m_bRflVolumeFull = TRUE;
		}
		m_bRflVolumeOk = FALSE;
		goto Exit;
	}

	// Flush the file handle to ensure it is forced to disk.

	if (RC_BAD( rc = m_pFileHdl->flush()))
	{
		// Remap disk full error

		if (rc == NE_FLM_IO_DISK_FULL)
		{
			rc = RC_SET( NE_SFLM_RFL_DISK_FULL);
			m_bRflVolumeFull = TRUE;
		}
		m_bRflVolumeOk = FALSE;
		goto Exit;
	}

Exit:

	return( rc);
}

/********************************************************************
Desc:		Verifies the header of an RFL file.
*********************************************************************/
RCODE F_Rfl::verifyHeader(
	FLMBYTE *	pucHeader,
	FLMUINT		uiFileNum,
	FLMBYTE *	pucSerialNum)
{
	RCODE			rc = NE_SFLM_OK;

	flmAssert( m_pDatabase);

	// Check the RFL name and version number

	if( f_memcmp( &pucHeader [RFL_NAME_POS], RFL_NAME,
						RFL_NAME_LEN) != 0)
	{
		rc = RC_SET( NE_SFLM_NOT_RFL);
		goto Exit;
	}
	
	if( f_memcmp( &pucHeader [RFL_VERSION_POS], RFL_VERSION,
					RFL_VERSION_LEN) != 0)
	{
		rc = RC_SET( NE_SFLM_NOT_RFL);
		goto Exit;
	}

	// Verify the database serial number

	if( f_memcmp( &pucHeader [RFL_DB_SERIAL_NUM_POS],
						m_pDatabase->m_lastCommittedDbHdr.ucDbSerialNum,
						SFLM_SERIAL_NUM_SIZE) != 0)
	{
		rc = RC_SET( NE_SFLM_BAD_RFL_DB_SERIAL_NUM);
		goto Exit;
	}

	// Verify the serial number that is expected to be on the
	// RFL file.  If pucSerialNum is NULL, we will not verify
	// it.  This is generally only done during recovery or restore
	// when we are reading through multiple RFL files and we need
	// to verify their serial numbers.

	if( pucSerialNum &&
		 f_memcmp( &pucHeader [RFL_SERIAL_NUM_POS],
						pucSerialNum, SFLM_SERIAL_NUM_SIZE) != 0)
	{
		rc = RC_SET( NE_SFLM_BAD_RFL_SERIAL_NUM);
		goto Exit;
	}

	// Verify the file number.

	if( uiFileNum != (FLMUINT)FB2UD( &pucHeader [RFL_FILE_NUMBER_POS]))
	{
		rc = RC_SET( NE_SFLM_BAD_RFL_FILE_NUMBER);
		goto Exit;
	}

	// Save serial numbers from the header.

	f_memcpy( m_ucCurrSerialNum, &pucHeader [RFL_SERIAL_NUM_POS],
					SFLM_SERIAL_NUM_SIZE);
	f_memcpy( m_ucNextSerialNum, &pucHeader [RFL_NEXT_FILE_SERIAL_NUM_POS],
					SFLM_SERIAL_NUM_SIZE);

	// Save some things from the header.

	m_uiFileEOF = (FLMUINT)FB2UD( &pucHeader [RFL_EOF_POS]);

Exit:

	return( rc);
}

/********************************************************************
Desc:		Opens an RFL file.  Verifies the serial number for 4.3 dbs.
*********************************************************************/
RCODE F_Rfl::openFile(
	F_SEM				hWaitSem,
	FLMUINT			uiFileNum,
	FLMBYTE *		pucSerialNum)
{
	RCODE				rc = NE_SFLM_OK;
	char				szRflFileName [F_PATH_MAX_SIZE];
	FLMUINT			uiFileNameSize;
	FLMBYTE			ucBuf [512];
	FLMUINT			uiBytesRead;

	flmAssert( m_pDatabase);

	// If we have a file open and it is not the file number
	// passed in, close it.

	if (m_pFileHdl)
	{
		if (m_pCurrentBuf->uiCurrFileNum != uiFileNum)
		{
			if (RC_BAD( rc = waitForCommit( hWaitSem)))
			{
				goto Exit;
			}
			closeFile();
		}
		else
		{
			goto Exit;	// Will return NE_SFLM_OK
		}
	}
	else
	{
		// Should not be able to be in the middle of a commit
		// if we don't have a file open!

		flmAssert( !m_pCommitBuf);
	}

	// Generate the log file name.

	uiFileNameSize = sizeof( szRflFileName);
	getFullRflFileName( uiFileNum, szRflFileName, &uiFileNameSize, NULL);

	// Open the file.

	if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->openFile( szRflFileName,
			  			   FLM_IO_RDWR | FLM_IO_SH_DENYNONE | FLM_IO_DIRECT,
							&m_pFileHdl)))
	{
		goto Exit;
	}

	// Read the header.

	if (RC_BAD( rc = m_pFileHdl->read( 0, 512, ucBuf, &uiBytesRead)))
	{
		if (rc == NE_FLM_IO_END_OF_FILE)
		{
			rc = RC_SET( NE_SFLM_NOT_RFL);
		}
		else
		{
			m_bRflVolumeOk = FALSE;
		}
		goto Exit;
	}

	// If there is not enough data in the buffer, it is not an
	// RFL file.

	if (uiBytesRead < 512)
	{
		rc = RC_SET( NE_SFLM_NOT_RFL);
		goto Exit;
	}

	// Verify the header information

	if (RC_BAD( rc = verifyHeader( ucBuf, uiFileNum, pucSerialNum)))
	{
		goto Exit;
	}

	m_pCurrentBuf->uiRflBufBytes = 0;
	m_pCurrentBuf->uiRflFileOffset = 0;
	m_pCurrentBuf->uiCurrFileNum = uiFileNum;
	
Exit:

	if( RC_BAD( rc))
	{
		waitForCommit( hWaitSem);
		closeFile();
	}
	
	return( rc);
}

/********************************************************************
Desc:		Creates a new roll forward log file.
*********************************************************************/
RCODE F_Rfl::createFile(
	F_Db *		pDb,
	FLMUINT		uiFileNum,
	FLMBYTE *	pucSerialNum,
	FLMBYTE *	pucNextSerialNum,
	FLMBOOL		bKeepSignature)
{
	RCODE			rc = NE_SFLM_OK;
	char			szRflFileName [F_PATH_MAX_SIZE];
	FLMUINT		uiFileNameSize;

	flmAssert( m_pDatabase);

	// Better not be trying to create the current file

	flmAssert( uiFileNum != m_pCurrentBuf->uiCurrFileNum);

	// If we have a file open, close it.

	if (RC_BAD( rc = waitForCommit( pDb->m_hWaitSem)))
	{
		goto Exit;
	}
	closeFile();

	// Generate the log file name.

	uiFileNameSize = sizeof( szRflFileName);
	getFullRflFileName( uiFileNum, szRflFileName, &uiFileNameSize, NULL);

	// Delete the file if it already exists - don't care
	// about return code here

	(void)gv_SFlmSysData.pFileSystem->deleteFile( szRflFileName);

	// If DB is 4.3 or greater and we are in the same directory as
	// our database files, see if we need to create the
	// subdirectory.  Otherwise, the RFL directory should already
	// have been created.  If the directory already exists, it is
	// OK - we only try this the first time after setRflDir is
	// called - to either verify that the directory exists, or if
	// it doesn't, to create it.

	if (m_bCreateRflDir)
	{

		// If it already exists, don't attempt to create it.

		if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->doesFileExist( m_szRflDir)))
		{
			if (rc != NE_FLM_IO_PATH_NOT_FOUND && 
				 rc != NE_FLM_IO_INVALID_FILENAME)
			{
				goto Exit;
			}
			else
			{
				if (RC_BAD( rc =
						gv_SFlmSysData.pFileSystem->createDir( m_szRflDir)))
				{
					goto Exit;
				}
			}
		}
		m_bCreateRflDir = FALSE;
	}

	// Create the file

	if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->createFile( szRflFileName,
			 FLM_IO_RDWR | FLM_IO_EXCL | FLM_IO_SH_DENYNONE | FLM_IO_DIRECT,
			 &m_pFileHdl)))
	{
		goto Exit;
	}

	// Initialize the header.

	if (RC_BAD( rc = writeHeader( uiFileNum, 0,
								pucSerialNum, pucNextSerialNum, bKeepSignature)))
	{
		goto Exit;
	}

	m_pCurrentBuf->uiRflBufBytes = 0;
	m_pCurrentBuf->uiRflFileOffset = 512;
	m_pCurrentBuf->uiCurrFileNum = uiFileNum;
Exit:

	// Close the RFL log file AND delete it if we were not successful.

	if (RC_BAD( rc))
	{
		closeFile();
		(void)gv_SFlmSysData.pFileSystem->deleteFile( szRflFileName);
	}
	return( rc);
}

/********************************************************************
Desc:		Copy last partial sector of last buffer written (or to be
			written) into a new buffer.
*********************************************************************/
void F_Rfl::copyLastSector(
	RFL_BUFFER *	pBuffer,
	FLMBYTE *		pucOldBuffer,
	FLMBYTE *		pucNewBuffer,
	FLMUINT			uiCurrPacketLen,
	FLMBOOL			bStartingNewFile)
{
	FLMUINT	uiOldBufBytes = pBuffer->uiRflBufBytes;

	// If we will be starting a new file, no need to keep any of
	// what is in the buffer.  Only the current packet needs to
	// be copied - at the beginning of the buffer.

	// OTHERWISE:

	// If there are fewer than 512 bytes in the buffer, we simply
	// keep them and keep appending to the buffer the next time
	// we output stuff.  The beginning of the buffer must ALWAYS be
	// a 512 byte boundary in the file, because we want to always
	// do our writing on 512 byte boundaries - because of direct IO.

	// If the number of bytes in the buffer is over 512 and it is
	// evenly divisible by 512, we can clear the buffer.  Otherwise,
	// we want to move the extra bytes over the last 512 byte boundary
	// down to the beginning of the buffer and adjust the buffer bytes
	// to reflect just these left-over bytes.

	if (bStartingNewFile)
	{
		pBuffer->uiRflBufBytes = 0;
		pBuffer->uiRflFileOffset = 512;
	}
	else if (pBuffer->uiRflBufBytes >= 512)
	{

		// See if the number of bytes in the buffer is an exact
		// multiple of 512.

		if (pBuffer->uiRflBufBytes & 511)	// Not exact multiple
		{

			// Round m_uiRflBufBytes down to next 512 byte boundary

			FLMUINT		ui512Offset = ROUND_DOWN_TO_NEAREST_512(
										pBuffer->uiRflBufBytes);

			// Move all bytes above the nearest 512 byte boundary
			// down to the beginning of the buffer and adjust
			// pBuffer->uiRflBufBytes and
			// pBuffer->uiRflFileOffset accordingly.

			f_memcpy( pucNewBuffer, &pucOldBuffer[ui512Offset],
						 pBuffer->uiRflBufBytes - ui512Offset);
			pBuffer->uiRflBufBytes -= ui512Offset;
			pBuffer->uiRflFileOffset += ui512Offset;
		}
		else
		{
			pBuffer->uiRflFileOffset += pBuffer->uiRflBufBytes;
			pBuffer->uiRflBufBytes = 0;
		}
	}
	else if (pucNewBuffer != pucOldBuffer)
	{
		f_memcpy( pucNewBuffer, pucOldBuffer, pBuffer->uiRflBufBytes);
	}
	
	if (uiCurrPacketLen)
	{
		flmAssert( uiOldBufBytes + uiCurrPacketLen <= m_uiBufferSize);
		f_memmove( &pucNewBuffer [pBuffer->uiRflBufBytes],
						&pucOldBuffer [uiOldBufBytes],
						uiCurrPacketLen);
	}
}

/********************************************************************
Desc:		Flush the RFL data from the buffer to disk.
*********************************************************************/
RCODE F_Rfl::flush(
	F_Db *			pDb,
	RFL_BUFFER *	pBuffer,
	FLMBOOL			bFinalWrite,
	FLMUINT			uiCurrPacketLen,
	FLMBOOL			bStartingNewFile)
{
	RCODE				rc = NE_SFLM_OK;
	FLMUINT			uiBytesWritten;
	IF_IOBuffer *	pNewBuffer = NULL;
	IF_IOBuffer *	pAsyncBuf = NULL;
	FLMBYTE *		pucOldBuffer;
	FLMUINT			uiFileOffset;
	FLMUINT			uiBufBytes;

	if (m_pFileHdl && pBuffer->uiRflBufBytes)
	{
		// Must wait for stuff in committing buffer, if any, before
		// going ahead here.

		if (pBuffer != m_pCommitBuf)
		{
			if (RC_BAD( rc = waitForCommit( pDb->m_hWaitSem)))
			{
				goto Exit;
			}
		}

		if (m_uiRflWriteBufs > 1 && m_pFileHdl->canDoAsync())
		{
			pAsyncBuf = pBuffer->pIOBuffer;
		}

		if ((FLMUINT)(-1) - pBuffer->uiRflFileOffset <=
								pBuffer->uiRflBufBytes)
		{
			rc = RC_SET( NE_SFLM_DB_FULL);
			goto Exit;
		}

		pucOldBuffer = pBuffer->pIOBuffer->getBufferPtr();
		uiFileOffset = pBuffer->uiRflFileOffset;
		uiBufBytes = pBuffer->uiRflBufBytes;
		if (m_uiRflWriteBufs > 1)
		{
			if( RC_BAD( rc = pBuffer->pBufferMgr->getBuffer( m_uiBufferSize,
				&pNewBuffer)))
			{
				goto Exit;
			}

			// No need to copy data if it is the final write,
			// because it won't be reused anyway - the data for
			// the next transaction has already been copied to
			// another buffer.

			if (!bFinalWrite)
			{
				copyLastSector( pBuffer, pucOldBuffer, pNewBuffer->getBufferPtr(),
									uiCurrPacketLen, bStartingNewFile);
			}
		}
		
		if( pAsyncBuf)
		{
			rc = m_pFileHdl->write( uiFileOffset, uiBufBytes, pAsyncBuf,
							&uiBytesWritten);
		}
		else
		{
			rc = m_pFileHdl->write( uiFileOffset, uiBufBytes, pucOldBuffer,
							&uiBytesWritten);
		}
						
		if( m_uiRflWriteBufs == 1)
		{

			// We are counting on the fact that the write completed.
			// When we only have one buffer, we cannot do async
			// writes.

			flmAssert( !pAsyncBuf);
			if (RC_OK( rc) && !bFinalWrite)
			{
				copyLastSector( pBuffer, pucOldBuffer, pucOldBuffer,
									uiCurrPacketLen, bStartingNewFile);
			}

			// DO NOT call notifyComplete - that would put
			// pBuffer->pIOBuffer into the avail list, and we
			// don't want that.  We simply want to keep
			// reusing it.

		}
		else
		{

			// No need to call copyLastSector, because it was called
			// above before calling SectorWrite.  The part of the
			// old buffer that needs to be transferred to the new
			// buffer has already been transferred.

			if (!pAsyncBuf)
			{
				pBuffer->pIOBuffer->notifyComplete( rc);
			}
			pBuffer->pIOBuffer = pNewBuffer;
		}

		if (RC_BAD( rc))
		{
			// Remap disk full error

			if (rc == NE_FLM_IO_DISK_FULL)
			{
				rc = RC_SET( NE_SFLM_RFL_DISK_FULL);
				m_bRflVolumeFull = TRUE;
			}
			m_bRflVolumeOk = FALSE;
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/********************************************************************
Desc:	Switch buffers.  This routine assumes the m_hBufMutex is locked.
*********************************************************************/
void F_Rfl::switchBuffers( void)
{
	RFL_BUFFER *	pOldBuffer = m_pCurrentBuf;

	if (m_pCurrentBuf == &m_Buf1)
	{
		m_pCurrentBuf = &m_Buf2;
	}
	else
	{
		m_pCurrentBuf = &m_Buf1;
	}
	m_pCurrentBuf->bTransInProgress = pOldBuffer->bTransInProgress;
	m_pCurrentBuf->uiCurrFileNum = pOldBuffer->uiCurrFileNum;
	m_pCurrentBuf->uiRflBufBytes = pOldBuffer->uiRflBufBytes;
	m_pCurrentBuf->uiRflFileOffset = pOldBuffer->uiRflFileOffset;
	
	if (pOldBuffer->uiRflBufBytes)
	{
		copyLastSector( m_pCurrentBuf, pOldBuffer->pIOBuffer->getBufferPtr(),
							 m_pCurrentBuf->pIOBuffer->getBufferPtr(), 0, FALSE);
	}
}

/********************************************************************
Desc: Wait for all RFL transaction writes to be finished.  The caller
		has the write lock on the database, which will prevent further
		writes to the RFL.
*********************************************************************/
FLMBOOL F_Rfl::seeIfRflWritesDone(
	F_SEM			hWaitSem,
	FLMBOOL		bForceWait)
{
	FLMBOOL		bWritesDone;

	f_mutexLock( m_hBufMutex);

	if (!bForceWait)
	{
		bWritesDone = (FLMBOOL)((m_pCurrentBuf->pFirstWaiter || m_pCommitBuf)
										? FALSE
										: TRUE);

		if( bWritesDone)
		{
			m_pCurrentBuf->uiRflBufBytes = 0;
		}

		f_mutexUnlock( m_hBufMutex);
	}
	else
	{
		// If the current buffer has a waiter, add self to that list
		// to wait, because it will be notified after the commit buffer
		// has been notified.  Otherwise, if there is a commit in
		// progress, add self to that list to wait.

		if (m_pCurrentBuf->pFirstWaiter)
		{

			// If bTransInProgress is TRUE and m_pCommitBuf is NULL
			// then this thread is the current transaction, and
			// nobody is going to wake up the first waiter until we
			// are done!  Hence, we must wake him up.

			if (!m_pCommitBuf)
			{

				// If m_pCommitBuf is NULL, this could only be possible if
				// there is a transaction in progress.  Otherwise, there
				// would not have been a pFirstWaiter, because when
				// the commit buffer finishes writing, if there is a
				// waiter, it will set commitbuf=currentbuf if there
				// is no transaction active.

				flmAssert( m_pCurrentBuf->bTransInProgress);

				m_pCommitBuf = m_pCurrentBuf;
				switchBuffers();
				wakeUpWaiter( NE_SFLM_OK, TRUE);
				(void)waitForWrites( hWaitSem, m_pCommitBuf, FALSE);
			}
			else
			{
				FLMBOOL	bSaveTransInProgress = m_pCurrentBuf->bTransInProgress;

				// Must set bTransInProgress to FALSE so that when the writer
				// of m_pCommitBuf finishes, it will signal the first waiter
				// on m_pCurrentBuf.  If we don't do this, m_pCommitBuf will
				// simply be set to NULL, and the first waiter will never
				// be woke up.

				m_pCurrentBuf->bTransInProgress = FALSE;
				(void)waitForWrites( hWaitSem, m_pCurrentBuf, FALSE);

				// It is OK to restore the trans in progress flag to what it
				// was before, because whoever called this routine has a lock
				// on the database, and it is his trans-in-progress state
				// that should be preserved.  No other thread will have been
				// able to change that state because the database is locked.

				f_mutexLock( m_hBufMutex);
				m_pCurrentBuf->bTransInProgress = bSaveTransInProgress;
				f_mutexUnlock( m_hBufMutex);
			}

			m_pCurrentBuf->uiRflBufBytes = 0;
		}
		else if (m_pCommitBuf)
		{
			(void)waitForWrites( hWaitSem, m_pCommitBuf, FALSE);
		}
		else
		{
			f_mutexUnlock( m_hBufMutex);
		}
		bWritesDone = TRUE;
	}
	return( bWritesDone);
}

/********************************************************************
Desc: Wake up the first thread that is waiting on the commit buffer.
*********************************************************************/
void F_Rfl::wakeUpWaiter(
	RCODE				rc,
	FLMBOOL			bIsWriter)
{
	F_SEM			hESem;

#ifndef FLM_DEBUG
	F_UNREFERENCED_PARM( bIsWriter);
#else
	if (bIsWriter)
	{
		flmAssert( m_pCommitBuf->pFirstWaiter->bIsWriter);
	}
	else
	{
		flmAssert( !m_pCommitBuf->pFirstWaiter->bIsWriter);
	}
#endif

	*(m_pCommitBuf->pFirstWaiter->pRc) = rc;
	hESem = m_pCommitBuf->pFirstWaiter->hESem;
	if ((m_pCommitBuf->pFirstWaiter =
			m_pCommitBuf->pFirstWaiter->pNext) == NULL)
	{
		m_pCommitBuf->pLastWaiter = NULL;
	}
	f_semSignal( hESem);
}

/********************************************************************
Desc: Wait for the transaction writes to be finished.
*********************************************************************/
RCODE F_Rfl::completeTransWrites(
	F_Db *		pDb,
	FLMBOOL		bCommitting,
	FLMBOOL		bOkToUnlock)
{
	RCODE					rc	= NE_SFLM_OK;
	RCODE					tmpRc;
	FLMBOOL				bMutexLocked = FALSE;
	FLMBOOL				bNotifyWaiters = FALSE;
	FLMBOOL				bDbUnlocked = FALSE;
	SFLM_DB_STATS *	pDbStats = NULL;
	F_TMSTAMP			StartTime;

	f_mutexLock( m_hBufMutex);
	bMutexLocked = TRUE;
	m_pCurrentBuf->bTransInProgress = FALSE;

	flmAssert( pDb->m_uiFlags & FDB_HAS_WRITE_LOCK);

	// When recovering or restoring all we need to do is write out
	// the database header.

	if (pDb->m_uiFlags & FDB_REPLAYING_RFL)
	{
		if (pDb->m_bHadUpdOper &&
			 m_pCurrentBuf->bOkToWriteHdrs)
		{
			f_mutexUnlock( m_hBufMutex);
			bMutexLocked = FALSE;
			if (RC_BAD( rc = m_pDatabase->writeDbHdr( pDb->m_pDbStats,
												pDb->m_pSFileHdl,
												&m_pCurrentBuf->dbHdr,
												&m_pCurrentBuf->cpHdr, FALSE)))
			{
				m_pDatabase->setMustCloseFlags( rc, FALSE);
			}
		}
		goto Exit;
	}

	// Handle empty transactions differently.
	// These transactions should not do any writing and do not need to
	// wait for all writes to complete, unless the bOkToUnlock flag
	// is set to FALSE.  In that case they must wait for all writes
	// to complete before unlocking.

	if (!pDb->m_bHadUpdOper)
	{
		// If the current buffer has a waiter, add self to that list
		// to wait, because it will be notified after the commit buffer
		// has been notified.  Otherwise, if there is a commit in
		// progress, add self to that list to wait.

		if (m_pCurrentBuf->pFirstWaiter)
		{
			// If m_pCommitBuf is NULL then nobody is going to wake up
			// the first waiter - we must do it.

			if (!m_pCommitBuf)
			{
				if (bOkToUnlock)
				{
					pDb->unlinkFromTransList( bCommitting);
					bDbUnlocked = TRUE;
				}
				m_pCommitBuf = m_pCurrentBuf;
				switchBuffers();
				wakeUpWaiter( NE_SFLM_OK, TRUE);
				if (!bOkToUnlock)
				{
					bMutexLocked = FALSE;
					(void)waitForWrites( pDb->m_hWaitSem, m_pCommitBuf, FALSE);
				}
			}
			else if (!bOkToUnlock)
			{
				bMutexLocked = FALSE;
				(void)waitForWrites( pDb->m_hWaitSem, m_pCurrentBuf, FALSE);
			}
		}
		else if (m_pCommitBuf)
		{
			if (!bOkToUnlock)
			{
				bMutexLocked = FALSE;
				rc = waitForWrites( pDb->m_hWaitSem, m_pCommitBuf, FALSE);
			}
		}
		goto Exit;
	}

	// If there is a transaction committing, put self into
	// the wait list on the current buffer.  When the committer
	// finishes, he will wake up the first thread in the list
	// and that thread will commit the buffer.

	if (m_pCommitBuf)
	{
		FLMBOOL	bIsWriter;

		// Another thread has to be doing the writes to m_pCommitBuf,
		// which means that m_pCurrentBuf better not be equal to
		// m_pCommitBuf.

		flmAssert( m_pCommitBuf != m_pCurrentBuf);

		// If there are no waiters, we are the first one, so when
		// we get signaled, we should proceed and do the write.

		bIsWriter = m_pCurrentBuf->pFirstWaiter ? FALSE : TRUE;
		if (bOkToUnlock)
		{
			pDb->unlinkFromTransList( bCommitting);
			bDbUnlocked = TRUE;
		}
		bMutexLocked = FALSE;
		rc = waitForWrites( pDb->m_hWaitSem, m_pCurrentBuf, bIsWriter);

		// If we were the first one in the queue, we must now
		// do the write.

		if (!bIsWriter)
		{
			goto Exit;
		}

		// First one in the queue, fall through to do the write.

		// The thread that woke me up better have set m_pCommitBuf
		// See below.

		flmAssert( m_pCommitBuf);
	}
	else if (m_pCurrentBuf->pFirstWaiter)
	{

		// Another thread is ready to commit the next set of
		// buffers, but just needs to be woke up.

		if (bOkToUnlock)
		{
			pDb->unlinkFromTransList( bCommitting);
			bDbUnlocked = TRUE;
		}

		// Need to set things up for that first waiter and get him
		// going.

		m_pCommitBuf = m_pCurrentBuf;
		switchBuffers();
		wakeUpWaiter( rc, TRUE);

		// Wait for the write to be completed.

		bMutexLocked = FALSE;
		rc = waitForWrites( pDb->m_hWaitSem, m_pCommitBuf, FALSE);
		goto Exit;
	}
	else
	{
		m_pCommitBuf = m_pCurrentBuf;
		switchBuffers();
		if (bOkToUnlock)
		{
			pDb->unlinkFromTransList( bCommitting);
			bDbUnlocked = TRUE;
		}
		f_mutexUnlock( m_hBufMutex);
		bMutexLocked = FALSE;
	}

	// NOTE: From this point on we use tmpRc because we don't want to
	// lose the rc that may have been set above in the call to
	// waitForWrites

	// At this point the mutex better not be locked.

	flmAssert( !bMutexLocked);
	bNotifyWaiters = TRUE;

	if( (pDbStats = pDb->m_pDbStats) != NULL)
	{
		f_timeGetTimeStamp( &StartTime);
	}

	// Must write out whatever we have in the commit buffer before
	// unlocking the database.

	if (RC_BAD( tmpRc = flush( pDb, m_pCommitBuf, TRUE)))
	{
		if (RC_OK( rc))
		{
			rc = tmpRc;
		}
		goto Exit;
	}

	// Wait for any pending IO off of the log buffer

	if (RC_BAD( tmpRc = m_pCommitBuf->pBufferMgr->waitForAllPendingIO()))
	{
		if (RC_OK( rc))
		{
			rc = tmpRc;
		}
		goto Exit;
	}

	// Force the RFL writes to disk if necessary.
	// NOTE: It is possible for m_pFileHdl to be NULL at this point if
	// there were no operations actually logged.  This happens in
	// FlmDbUpgrade (see flconvrt.cpp).  Even though nothing was logged
	// the transaction is not an empty transaction, because it still needs
	// to write out the log header.

	if (m_pFileHdl)
	{
		if (RC_BAD( tmpRc = m_pFileHdl->flush()))
		{

			// Remap disk full error

			if (tmpRc == NE_FLM_IO_DISK_FULL)
			{
				rc = RC_SET( NE_SFLM_RFL_DISK_FULL);
				m_bRflVolumeFull = TRUE;
			}
			else if (RC_OK( rc))
			{
				rc = tmpRc;
			}
			m_bRflVolumeOk = FALSE;
			goto Exit;
		}
	}

	// Write the log header

	if (m_pCommitBuf->bOkToWriteHdrs)
	{
		if (RC_BAD( tmpRc = m_pDatabase->writeDbHdr( pDb->m_pDbStats,
											pDb->m_pSFileHdl,
											&m_pCommitBuf->dbHdr,
											&m_pCommitBuf->cpHdr, FALSE)))
		{
			if (RC_OK( rc))
			{
				rc = tmpRc;
			}
			m_pDatabase->setMustCloseFlags( tmpRc, FALSE);
			goto Exit;
		}
	}

Exit:

	if (!bDbUnlocked && bOkToUnlock)
	{
		pDb->unlinkFromTransList( bCommitting);
	}

	if (bNotifyWaiters)
	{
		FLMUINT	uiNumFinished = 1;	// For self

		flmAssert( !bMutexLocked);
		f_mutexLock( m_hBufMutex);
		bMutexLocked = TRUE;

		// Wake up any waiters

		while (m_pCommitBuf->pFirstWaiter)
		{
			uiNumFinished++;
			wakeUpWaiter( rc, FALSE);
		}

		// If there are waiters on the current buffer, the first one
		// should be woke up so it can start the next set of writes.

		if (m_pCurrentBuf->pFirstWaiter && !m_pCurrentBuf->bTransInProgress)
		{
			flmAssert( m_pCurrentBuf != m_pCommitBuf);
			m_pCommitBuf = m_pCurrentBuf;
			switchBuffers();
			wakeUpWaiter( rc, TRUE);
		}
		else
		{
			m_pCommitBuf = NULL;
		}
		if (pDbStats)
		{
			flmAddElapTime( &StartTime,
				&pDbStats->UpdateTransStats.GroupCompletes.ui64ElapMilli);
			pDbStats->UpdateTransStats.GroupCompletes.ui64Count++;
			pDbStats->bHaveStats = TRUE;
			pDbStats->UpdateTransStats.ui64GroupFinished += uiNumFinished;
		}
	}

	if (bMutexLocked)
	{
		f_mutexUnlock( m_hBufMutex);
	}
	return( rc);
}

/********************************************************************
Desc:	Flush all completed packets out of the RFL buffer, and shift
		the new partial packet down.  This guarantees that there is
		now room in the buffer for the maximum packet size.
*********************************************************************/
RCODE F_Rfl::shiftPacketsDown(
	F_Db *		pDb,
	FLMUINT		uiCurrPacketLen,
	FLMBOOL		bStartingNewFile)
{
	RCODE			rc = NE_SFLM_OK;

	// The call to flush will move whatever needs to be moved from
	// the current buffer into a new buffer if multiple buffers are
	// being used.  If only one buffer is being used, it will move
	// the part of the packet that needs to be moved down to the
	// beginning of the buffer - AFTER writing out the buffer.

	if (RC_BAD( rc = flush( pDb, m_pCurrentBuf, FALSE,
									uiCurrPacketLen, bStartingNewFile)))
	{
		goto Exit;
	}

	// NOTE: If multiple buffers are being used, whatever was moved
	// to the new buffer has not yet been written out

	if (bStartingNewFile)
	{
		if( RC_BAD( rc = waitPendingWrites()))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/********************************************************************
Desc:		Determine if we should start a new file.  If we are over the
			low limit, and the bDoNewIfOverLowLimit flag is set, we
			will start a new log file.  Or, if this packet size would
			put us over the upper limit, we will start a new log file.
*********************************************************************/
RCODE F_Rfl::seeIfNeedNewFile(
	F_Db *		pDb,
	FLMUINT		uiPacketLen,
	FLMBOOL		bDoNewIfOverLowLimit)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBYTE		ucNextSerialNum [SFLM_SERIAL_NUM_SIZE];

	flmAssert( m_pDatabase);

	// If the keep files flag is FALSE, we won't start
	// a new file.  NOTE: This should ALWAYS be false
	// for pre 4.3 databases.

	if( !m_bKeepRflFiles)
	{
		goto Exit;
	}

	// VERY IMPORTANT NOTE: It is preferrable that we keep transactions
	// entirely contained in the same RFL file if at all possible.  Note
	// that it is NOT a hard and fast requirement.  The system will work
	// just fine if we don't.  However, it would be nice if RFL files
	// always ended with a commit or abort packet.  This preferences is
	// due to what happens after a restore operation.  After a restore
	// operation, we always need to start a new RFL file, but if possible,
	// we would like that new RFL file to be the next one in the sequence
	// after the last RFL file that was restored.  We can only do this if
	// we were able to restore EVERY transaction that was in the last
	// restored RFL file - which we can only do if the last restored RFL
	// file ended with a commit or abort packet.
	// To accomplish this end, we try to roll to new files on the first
	// transaction begin packet that occurs after we have exceeded our
	// low threshold - which is why bDoNewIfOverLowLimit is only set to
	// TRUE on transaction begin packets.  It is set to FALSE on other
	// packets so that we will continue logging the transaction in the
	// same file that we started the transaction in - if possible.  The
	// only thing that will cause a non-transaction-begin packet to roll
	// to a new file is if we would exceed the high limit.

	if ((bDoNewIfOverLowLimit &&
		  m_pCurrentBuf->uiRflFileOffset + m_pCurrentBuf->uiRflBufBytes >=
				m_uiRflMinFileSize) ||
		 (m_pCurrentBuf->uiRflFileOffset + m_pCurrentBuf->uiRflBufBytes +
			uiPacketLen >= m_uiRflMaxFileSize))
	{
		FLMUINT	uiCurrFileEOF = m_pCurrentBuf->uiRflFileOffset +
										 m_pCurrentBuf->uiRflBufBytes;

		// Shift the current packet to the beginning of the buffer.
		// Any packets in the buffer before that one will be written
		// out to the current file.

		if (RC_BAD( rc = shiftPacketsDown( pDb, uiPacketLen, TRUE)))
		{
			goto Exit;
		}

		// Update the header of the current file and close it.

		if (RC_BAD( rc = writeHeader( m_pCurrentBuf->uiCurrFileNum,
									uiCurrFileEOF,
									m_ucCurrSerialNum, m_ucNextSerialNum, TRUE)))
		{
			goto Exit;
		}

		// Truncate the file.

		if (!ON_512_BYTE_BOUNDARY( uiCurrFileEOF))
		{
			uiCurrFileEOF = ROUND_DOWN_TO_NEAREST_512( uiCurrFileEOF) + 512;
		}
		
		if (RC_BAD( rc = m_pFileHdl->truncateFile( uiCurrFileEOF)))
		{
			goto Exit;
		}

		// Close the file handle.

		m_pFileHdl->closeFile();
		m_pFileHdl->Release();
		m_pFileHdl = NULL;

		// Get the next serial number that will be used for the RFL
		// file after this one.

		if (RC_BAD( rc = f_createSerialNumber( ucNextSerialNum)))
		{
			goto Exit;
		}

		// Create next file in the sequence.

		// Use the next serial number stored in the FDB's log header
		// for the serial number on this RFL file.  Use the serial
		// number we just generated as the next RFL serial number.

		if (RC_BAD( rc = createFile( pDb, m_pCurrentBuf->uiCurrFileNum + 1,
				m_ucNextSerialNum, ucNextSerialNum, TRUE)))
		{
			goto Exit;
		}

		// Move the next serial number to the current serial number
		// and the serial number we generated above into the next
		// serial number.

		f_memcpy( m_ucCurrSerialNum, m_ucNextSerialNum, SFLM_SERIAL_NUM_SIZE);
		f_memcpy( m_ucNextSerialNum, ucNextSerialNum, SFLM_SERIAL_NUM_SIZE);
	}

Exit:
	return( rc);
}

/********************************************************************
Desc:		Finish the current RFL file - set up so that next
			transaction will begin a new RFL file.
*********************************************************************/
RCODE F_Rfl::finishCurrFile(
	F_Db *	pDb,
	FLMBOOL	bNewKeepState)
{
	RCODE				rc = NE_SFLM_OK;
	FLMBOOL			bDbLocked = FALSE;
	FLMUINT			uiTransFileNum;
	FLMUINT			uiTransOffset;
	FLMUINT			uiTruncateSize;
	SFLM_DB_HDR *	pUncommittedDbHdr;
	SFLM_DB_HDR		checkpointDbHdr;

	// Make sure we don't have a transaction going

	if (pDb->m_eTransType != SFLM_NO_TRANS)
	{
		rc = RC_SET( NE_SFLM_TRANS_ACTIVE);
		goto Exit;
	}

	// Make sure there is no active backup running

	m_pDatabase->lockMutex();
	if (m_pDatabase->m_bBackupActive)
	{
		m_pDatabase->unlockMutex();
		rc = RC_SET( NE_SFLM_BACKUP_ACTIVE);
		goto Exit;
	}
	m_pDatabase->unlockMutex();

	// Lock the database - need to prevent update
	// transactions and checkpoint thread from running.

	if (RC_BAD( rc = pDb->lockExclusive( FLM_NO_TIMEOUT)))
	{
		goto Exit;
	}
	bDbLocked = TRUE;

	// Must wait for all RFL writes before switching files.

	(void)seeIfRflWritesDone( pDb->m_hWaitSem, TRUE);

	// Better not be in the middle of a transaction.

	flmAssert( !m_ui64CurrTransID);

	pUncommittedDbHdr = &m_pDatabase->m_uncommittedDbHdr;

	// Don't want to copy last committed log header into
	// uncommitted log header if bNewKeepState is TRUE because
	// the caller has already done it, and has made modifications
	// to the uncommitted log header that we don't want to lose.

	if (!bNewKeepState)
	{
		f_memcpy( pUncommittedDbHdr, &m_pDatabase->m_lastCommittedDbHdr,
						sizeof( SFLM_DB_HDR));

		// If we are in a no-keep state, but we were not told that
		// we have a new keep state, we cannot roll to the next
		// RFL file, because a checkpoint has not been done.  This is
		// not an error - it is just the case where FlmDbConfig was
		// asked to roll to the next RFL file when the keep flag was
		// still FALSE.

		if (!pUncommittedDbHdr->ui8RflKeepFiles)
		{
			goto Exit;
		}
	}

	// Get the last committed serial numbers from the file's log header
	// buffer.

	f_memcpy( m_ucCurrSerialNum,
					pUncommittedDbHdr->ucLastTransRflSerialNum,
					SFLM_SERIAL_NUM_SIZE);
	f_memcpy( m_ucNextSerialNum,
					pUncommittedDbHdr->ucNextRflSerialNum,
					SFLM_SERIAL_NUM_SIZE);
	uiTransFileNum = (FLMUINT)pUncommittedDbHdr->ui32RflCurrFileNum;
	uiTransOffset = (FLMUINT)pUncommittedDbHdr->ui32RflLastTransOffset;

	// If ui32RflLastTransOffset is zero, there is no need to go set
	// up to go to the next file, because we are already poised to do so at
	// the beginning of the next transaction.  Just return if this is the
	// case.  Same for if the file does not exist.

	if (!uiTransOffset)
	{
		if (!bNewKeepState)
		{
			goto Exit;
		}
	}
	else if (RC_BAD( rc = openFile( pDb->m_hWaitSem, uiTransFileNum,
		m_ucCurrSerialNum)))
	{
		if (rc == NE_FLM_IO_PATH_NOT_FOUND || rc == NE_FLM_IO_INVALID_FILENAME)
		{
			rc = NE_SFLM_OK;
			if (!bNewKeepState)
			{
				goto Exit;
			}
		}
		else
		{
			goto Exit;
		}
	}
	else
	{

		// At this point, we know the file exists, so we will update
		// its header and then update the log header.  Note that we
		// use the keep RFL state from the last committed log header,
		// not the uncommitted log header - because it will contain
		// the correct keep-state for the current RFL file.

		if (RC_BAD( rc = writeHeader( m_pCurrentBuf->uiCurrFileNum, uiTransOffset,
										m_ucCurrSerialNum, m_ucNextSerialNum,
										m_pDatabase->m_lastCommittedDbHdr.ui8RflKeepFiles
										? TRUE
										: FALSE)))
		{
			goto Exit;
		}

		// Truncate the file down to its EOF size - the nearest 512 byte boundary.

		uiTruncateSize = uiTransOffset;
		if (!ON_512_BYTE_BOUNDARY( uiTruncateSize))
		{
			uiTruncateSize = ROUND_DOWN_TO_NEAREST_512( uiTruncateSize) + 512;
		}
		if (RC_BAD( rc = m_pFileHdl->truncateFile( uiTruncateSize)))
		{
			goto Exit;
		}

		// Close the file handle.

		m_pFileHdl->closeFile();
		m_pFileHdl->Release();
		m_pFileHdl = NULL;

		// Set things up in the log header to go to the next file when
		// we begin the next transaction.  NOTE: NO need to lock the
		// mutex, because nobody but an update transaction looks at
		// the uncommitted log header.

		uiTransFileNum++;
		pUncommittedDbHdr->ui32RflCurrFileNum = (FLMUINT32)uiTransFileNum;
	}

	// Generate a new current serial number if bNewKeepState is
	// TRUE.  Otherwise, move the next serial number into the current
	// serial number.

	if (bNewKeepState)
	{
		if (RC_BAD( rc = f_createSerialNumber( m_ucCurrSerialNum)))
		{
			goto Exit;
		}
	}
	else
	{
		f_memcpy( m_ucCurrSerialNum, m_ucNextSerialNum, SFLM_SERIAL_NUM_SIZE);
	}

	// Always generate a new next serial number.

	if (RC_BAD( rc = f_createSerialNumber( m_ucNextSerialNum)))
	{
		goto Exit;
	}

	// Set transaction offset to zero.  This will force the
	// next RFL file to be created on the next transaction
	// begin.  It will be created even if it is already
	// there.

	pUncommittedDbHdr->ui32RflLastTransOffset = 0;
	f_memcpy( pUncommittedDbHdr->ucLastTransRflSerialNum,
				 m_ucCurrSerialNum, SFLM_SERIAL_NUM_SIZE);
	f_memcpy( pUncommittedDbHdr->ucNextRflSerialNum,
				 m_ucNextSerialNum, SFLM_SERIAL_NUM_SIZE);

	// Set the CP file number and CP offset to point into the new file.
	// The outer code (FlmDbConfig) has done a checkpoint and the database
	// is still locked.  We need to set these values here, otherwise
	// if we crash before the next checkpoint, recovery will start in the
	// old RFL file, causing an NE_SFLM_BAD_RFL_SERIAL_NUM to be returned when
	// traversing from the old RFL file to the new RFL file.
	// NOTE:  These changes must be made to the uncommitted log header AND
	// the CP log header (so that they will be written out even
	// though we are not forcing a checkpoint).

	if (bNewKeepState)
	{
		// Do a quick check to see if it looks like we are in a
		// checkpointed state

		if( !m_pDatabase->m_lastCommittedDbHdr.ui8RflKeepFiles &&
			 m_pDatabase->m_lastCommittedDbHdr.ui32RflLastTransOffset > 512)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_DATA_ERROR);
			goto Exit;
		}

		f_memcpy( &checkpointDbHdr,
			&m_pDatabase->m_checkpointDbHdr, sizeof( SFLM_DB_HDR));
		checkpointDbHdr.ui32RflLastCPFileNum = (FLMUINT32)uiTransFileNum;
		pUncommittedDbHdr->ui32RflLastCPFileNum = (FLMUINT32)uiTransFileNum;
		checkpointDbHdr.ui32RflLastCPOffset = 512;
		pUncommittedDbHdr->ui32RflLastCPOffset = 512;
	}

	// Write out the db header to disk.

	if (RC_BAD( rc = m_pDatabase->writeDbHdr( pDb->m_pDbStats, pDb->m_pSFileHdl,
									pUncommittedDbHdr,
									bNewKeepState
										? &checkpointDbHdr
										: &m_pDatabase->m_checkpointDbHdr, FALSE)))
	{
		goto Exit;
	}

	// Copy the uncommitted log header back to the committed log header and
	// copy the CP log header (if changed above).

	m_pDatabase->lockMutex();
	
	f_memcpy( &m_pDatabase->m_lastCommittedDbHdr, pUncommittedDbHdr,
				sizeof( SFLM_DB_HDR));

	if( bNewKeepState)
	{
		f_memcpy( &m_pDatabase->m_checkpointDbHdr, &checkpointDbHdr,
				sizeof( SFLM_DB_HDR));
	}
	
	m_pDatabase->unlockMutex();

Exit:

	if (bDbLocked)
	{
		(void)pDb->unlockExclusive();
	}
	return( rc);
}

/********************************************************************
Desc:		Finish packet by outputting header information for it.
*********************************************************************/
RCODE F_Rfl::finishPacket(
	F_Db *		pDb,
	FLMUINT		uiPacketType,
	FLMUINT		uiPacketBodyLen,
	FLMBOOL		bDoNewIfOverLowLimit)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiPacketLen;
	FLMBYTE *	pucPacket;

	uiPacketLen = uiPacketBodyLen + RFL_PACKET_OVERHEAD;

	// See if this packet will cause us to overflow the limits on
	// the current file.  If so, create a new file.

	if( RC_BAD( rc = seeIfNeedNewFile( pDb, uiPacketLen, bDoNewIfOverLowLimit)))
	{
		goto Exit;
	}

	// Get a pointer to packet header.

	pucPacket = &(m_pCurrentBuf->pIOBuffer->getBufferPtr()[
											m_pCurrentBuf->uiRflBufBytes]);

	// Set the packet address in the packet header.

	m_uiPacketAddress = m_pCurrentBuf->uiRflFileOffset +
							  m_pCurrentBuf->uiRflBufBytes;
	UD2FBA( (FLMUINT32)m_uiPacketAddress, &pucPacket [RFL_PACKET_ADDRESS_OFFSET]);

	// Set the packet type and packet body length.

	pucPacket [RFL_PACKET_TYPE_OFFSET] = (FLMBYTE)uiPacketType;

	UW2FBA( (FLMUINT16)uiPacketBodyLen,
		&pucPacket [RFL_PACKET_BODY_LENGTH_OFFSET]);

	// Set the checksum for the packet.

	pucPacket [RFL_PACKET_CHECKSUM_OFFSET] = RflCalcChecksum( pucPacket,
																uiPacketBodyLen);

	// Increment bytes in the buffer to reflect the fact that this packet
	// is now complete.

	m_pCurrentBuf->uiRflBufBytes += uiPacketLen;
	
Exit:

	return( rc);
}

/********************************************************************
Desc:		Truncate roll-forward log file to a certain size - only
			do if not keeping RFL files.
*********************************************************************/
RCODE F_Rfl::truncate(
	F_SEM			hWaitSem,
	FLMUINT		uiTruncateSize)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiFileNum;

	flmAssert( uiTruncateSize >= 512);

	// Keeping of log files better not be enabled.

	flmAssert( !m_pDatabase->m_lastCommittedDbHdr.ui8RflKeepFiles);

	// Better not be in the middle of a transaction.

	flmAssert( !m_ui64CurrTransID);

	// Open the current RFL file.  If it does not exist, it is OK - there
	// is nothing to truncate.

	uiFileNum = (FLMUINT)m_pDatabase->m_lastCommittedDbHdr.ui32RflCurrFileNum;
	if (RC_BAD( rc = openFile( hWaitSem, uiFileNum,
				m_pDatabase->m_lastCommittedDbHdr.ucLastTransRflSerialNum)))
	{
		if (rc == NE_FLM_IO_PATH_NOT_FOUND || rc == NE_FLM_IO_INVALID_FILENAME)
		{
			rc = NE_SFLM_OK;
		}
		goto Exit;
	}
	
	if (RC_BAD( rc = m_pFileHdl->truncateFile( uiTruncateSize)))
	{
		m_bRflVolumeOk = FALSE;
		goto Exit;
	}
	
Exit:

	return( rc);
}

/********************************************************************
Desc:		Setup to begin a transaction
*********************************************************************/
RCODE F_Rfl::setupTransaction(
	F_Db *		pDb)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiFileNum;
	FLMUINT		uiLastTransOffset;
	FLMBOOL		bCreateFile;

	f_mutexLock( m_hBufMutex);
	m_pCurrentBuf->bTransInProgress = TRUE;
	f_mutexUnlock( m_hBufMutex);

	// Get the last committed serial numbers from the file's log header
	// buffer.

	f_memcpy( m_ucCurrSerialNum,
				m_pDatabase->m_lastCommittedDbHdr.ucLastTransRflSerialNum,
				SFLM_SERIAL_NUM_SIZE);
	f_memcpy( m_ucNextSerialNum,
				m_pDatabase->m_lastCommittedDbHdr.ucNextRflSerialNum,
				SFLM_SERIAL_NUM_SIZE);
	uiFileNum = (FLMUINT)m_pDatabase->m_lastCommittedDbHdr.ui32RflCurrFileNum;
	uiLastTransOffset =
		(FLMUINT)m_pDatabase->m_lastCommittedDbHdr.ui32RflLastTransOffset;

	// If the LOG_RFL_LAST_TRANS_OFFSET is zero, we need to create the
	// next RFL file number no matter what.  There are two cases where
	// this happens: 1) when the database is first created, and 2) after
	// a restore operation.

	if (!uiLastTransOffset)
	{
		bCreateFile = TRUE;

		// Close the current file, just in case we had opened it before.
		// At this point, it doesn't matter because we are going to
		// overwrite it.

		if (RC_BAD( rc = waitForCommit( pDb->m_hWaitSem)))
		{
			goto Exit;
		}
		closeFile();
	}
	else if (RC_BAD( rc = openFile( pDb->m_hWaitSem, 
		uiFileNum, m_ucCurrSerialNum)))
	{
		if (rc != NE_FLM_IO_PATH_NOT_FOUND && rc != NE_FLM_IO_INVALID_FILENAME)
		{
			goto Exit;
		}
		bCreateFile = TRUE;
	}
	else
	{
		bCreateFile = FALSE;
	}

	if (bCreateFile)
	{

		// If the log header indicates that data has already been logged
		// to the file, we need to return the I/O error rather than just
		// re-creating the file.  This may mean that someone changed the
		// RFL directory without moving the RFL files properly.

		if (uiLastTransOffset > 512)
		{
			rc = RC_SET( NE_SFLM_RFL_FILE_NOT_FOUND);
			goto Exit;
		}

		// Create the RFL file if not found.

		// Use the next serial number stored in the FDB's log header
		// for the serial number on this RFL file.  Use the serial
		// number we just generated as the next RFL serial number.

		if (RC_BAD( rc = createFile( pDb, uiFileNum,
				m_ucCurrSerialNum, m_ucNextSerialNum,
				m_pDatabase->m_lastCommittedDbHdr.ui8RflKeepFiles
					? TRUE
					: FALSE)))
		{
			goto Exit;
		}
	}
	else
	{
		// Read in enough of the buffer from the RFL file so that
		// we are positioned on a 512 byte boundary.

		if (RC_BAD( positionTo( uiLastTransOffset)))
		{
			goto Exit;
		}
	}

	// These can only be changed when starting a transaction.

	m_bKeepRflFiles = m_pDatabase->m_lastCommittedDbHdr.ui8RflKeepFiles
							? TRUE
							: FALSE;

	m_uiRflMaxFileSize =
		(FLMUINT)m_pDatabase->m_lastCommittedDbHdr.ui32RflMaxFileSize;

	// Round maximum down to nearest 512 boundary.  This is necessary
	// because we always write a minimum of 512 byte units in direct IO
	// mode.  If we did not round the maximum down, our last packet could
	// end at an offset that is less than the maximum, but greater than
	// the nearest 512 byte boundary - technically within the user-specified
	// size limit.  However, because we always write a full 512 bytes of data
	// to fill out the last sector when we are in direct IO mode, we would
	// end up with a file that was slightly larger than the user-specified
	// limit.  The EOF in the header of the file would be below the limit,
	// but the actual file size would not be.  Thus, the need to round down.

	m_uiRflMaxFileSize = ROUND_DOWN_TO_NEAREST_512( m_uiRflMaxFileSize);

	// The maximum cannot go below a certain threshold - must have room for
	// least one packet plus the header.

	if (m_uiRflMaxFileSize < RFL_MAX_PACKET_SIZE + 512)
	{
		m_uiRflMaxFileSize = RFL_MAX_PACKET_SIZE + 512;
	}
	else if (m_uiRflMaxFileSize > gv_SFlmSysData.uiMaxFileSize)
	{
		m_uiRflMaxFileSize = gv_SFlmSysData.uiMaxFileSize;
	}

	m_uiRflMinFileSize =
		(FLMUINT)m_pDatabase->m_lastCommittedDbHdr.ui32RflMinFileSize;

	// Minimum RFL file size should not be allowed to be larger than
	// maximum!

	if (m_uiRflMinFileSize > m_uiRflMaxFileSize)
	{
		m_uiRflMinFileSize = m_uiRflMaxFileSize;
	}

	// Set the operation count to zero.

	m_uiOperCount = 0;

	m_pFileHdl->setMaxAutoExtendSize( m_uiRflMaxFileSize);
	m_pFileHdl->setExtendSize( m_pDatabase->m_uiFileExtendSize);
	
Exit:

	return( rc);
}

/********************************************************************
Desc:		Log transaction begin.  This routine will also make sure
			we have opened an RFL file.
			NOTE: The prior version of FLAIM (before 4.3) would log
			a time and set the RFL_TIME_LOGGED_FLAG bit in the packet
			type.  This is no longer done.  Old code should be
			compatible because it reads the flag.
*********************************************************************/
RCODE F_Rfl::logBeginTransaction(
	F_Db *		pDb)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBYTE *	pucPacketStart;
	FLMBYTE *	pucPacketBody;
	FLMUINT		uiPacketBodyLen;
	FLMUINT		uiMaxPacketBodyLen;

	flmAssert( pDb->m_uiFlags & FDB_HAS_FILE_LOCK);
#ifndef FLM_DEBUG
	F_UNREFERENCED_PARM( pDb);
#endif
	
	// Do nothing if logging is disabled.

	if( !isLoggingEnabled())
	{
		goto Exit;
	}
	
	flmAssert( !(pDb->m_uiFlags & FDB_REPLAYING_RFL));

	// Better not be in the middle of a transaction.

	flmAssert( !m_ui64CurrTransID);

	if( RC_BAD( rc = setupTransaction( pDb)))
	{
		goto Exit;
	}

	uiMaxPacketBodyLen = FLM_MAX_SEN_LEN;

	// Make sure we have space in the RFL buffer for a complete packet.

	if (!haveBuffSpace( uiMaxPacketBodyLen + RFL_PACKET_OVERHEAD))
	{
		if (RC_BAD( rc = flush( pDb, m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	// Get a pointer to where we will be laying down the packet body.

	pucPacketBody = pucPacketStart = getPacketBodyPtr();

	// Output the transaction ID.

	f_encodeSEN( pDb->m_ui64CurrTransID, &pucPacketBody);

	// Finish the packet

	uiPacketBodyLen = (FLMUINT)(pucPacketBody - pucPacketStart);
	flmAssert( uiPacketBodyLen <= uiMaxPacketBodyLen);

	if (RC_BAD( rc = finishPacket(
		pDb, RFL_TRNS_BEGIN_PACKET, uiPacketBodyLen, TRUE)))
	{
		goto Exit;
	}

	// Save the file offset for the start transaction packet.

	m_uiTransStartFile = m_pCurrentBuf->uiCurrFileNum;
	m_uiTransStartAddr = m_pCurrentBuf->uiRflFileOffset +
								m_pCurrentBuf->uiRflBufBytes -
									uiPacketBodyLen - RFL_PACKET_OVERHEAD;
	m_ui64CurrTransID = pDb->m_ui64CurrTransID;

Exit:

	return( rc);
}

/********************************************************************
Desc:	Do a transaction begin operation during restore or recovery.
*********************************************************************/
RCODE F_Rfl::recovTransBegin(
	F_Db *				pDb,
	eRestoreAction *	peAction)
{
	RCODE			rc = NE_SFLM_OK;

	if( m_pRestoreStatus)
	{
		if( RC_BAD( rc = m_pRestoreStatus->reportBeginTrans( 
			peAction, m_ui64CurrTransID)))
		{
			goto Exit;
		}

		if (*peAction == SFLM_RESTORE_ACTION_STOP)
		{
			// Need to set m_ui64CurrTransID to 0 since it was
			// set by getPacket().  We are not going to
			// start a transaction because of the user's request
			// to exit.

			m_ui64CurrTransID = 0;
			goto Exit;
		}
	}

	if (RC_BAD( rc = pDb->transBegin( SFLM_UPDATE_TRANS)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/********************************************************************
Desc:		Flushes the RFL and sets some things in the log header.
*********************************************************************/
void F_Rfl::finalizeTransaction( void)
{
	FLMUINT		uiRflTransEndOffset;
	SFLM_DB_HDR *	pDbHdr = &m_pDatabase->m_uncommittedDbHdr;

	// Save the serial numbers and file numbers into the file's
	// uncommitted log header.

	pDbHdr->ui32RflCurrFileNum = (FLMUINT32)m_pCurrentBuf->uiCurrFileNum;

	uiRflTransEndOffset = getCurrWriteOffset();
	pDbHdr->ui32RflLastTransOffset = (FLMUINT32)uiRflTransEndOffset;

	f_memcpy( pDbHdr->ucLastTransRflSerialNum,
		m_ucCurrSerialNum, SFLM_SERIAL_NUM_SIZE);

	f_memcpy( pDbHdr->ucNextRflSerialNum,
		m_ucNextSerialNum, SFLM_SERIAL_NUM_SIZE);
}

/********************************************************************
Desc:		Handles the commit and abort log operations. If aborting
			the transaction, or if the transaction was empty, we will
			simply throw away the entire transaction and not bother
			to log it.  In that case we will reset transaction pointers,
			etc. back to the file and offset where the transaction began.
			We will also delete RFL files that were created during the
			transaction if necessary.  NOTE: It is not essential that
			the RFL files be deleted.  If they are not successfully
			deleted, they will be overwritten if need be when creating
			new ones.
			NOTE: The prior version of FLAIM (before 4.3) would log
			a time and set the RFL_TIME_LOGGED_FLAG bit in the packet
			type.  This is no longer done.  Old code should be
			compatible because it reads the flag.
*********************************************************************/
RCODE F_Rfl::logEndTransaction(
	F_Db *		pDb,
	FLMUINT		uiPacketType,
	FLMBOOL		bThrowLogAway,
	FLMBOOL *	pbLoggedTransEnd)
{
	RCODE			rc = NE_SFLM_OK;
	RCODE			rc2 = NE_SFLM_OK;
	FLMUINT		uiLowFileNum;
	FLMUINT		uiHighFileNum;
	char			szRflFileName [F_PATH_MAX_SIZE];
	FLMBYTE *	pucPacketStart;
	FLMBYTE *	pucPacketBody;
	FLMUINT		uiPacketBodyLen;
	FLMUINT		uiMaxPacketBodyLen;

	flmAssert( pDb->m_uiFlags & FDB_HAS_FILE_LOCK);
#ifndef FLM_DEBUG
	F_UNREFERENCED_PARM( pDb);
#endif
	
	// Initialize the "logged trans end" flag

	if( pbLoggedTransEnd)
	{
		*pbLoggedTransEnd = FALSE;
	}

	// Do nothing if logging is disabled.

	if (!isLoggingEnabled())
	{
		goto Exit;
	}

	flmAssert( m_pFileHdl);
	flmAssert( m_pDatabase);

	// If the transaction had no operations, throw it away - don't
	// even log the packet.  An abort operation may also
	// elect to throw the log away even if there were
	// operations.  That is determined by the bThrowLogAway flag.
	// The bThrowLogAway flag may be TRUE when doing a commit if
	// the caller knows that nothing happened during the transction.

	if (bThrowLogAway || !m_uiOperCount)
	{
Throw_Away_Transaction:

		// If we have switched files, delete all but the file we
		// started in.

		if (m_pCurrentBuf->uiCurrFileNum != m_uiTransStartFile)
		{
			flmAssert( m_pCurrentBuf->uiCurrFileNum > m_uiTransStartFile);

			// File number in uncommitted log header better not
			// have been changed yet.  It is only supposed to
			// be changed when the transaction finishes - i.e., in
			// this routine.  Up until this point, it should only
			// be changed in m_pCurrentBuf->uiCurrFileNum.

			flmAssert( m_uiTransStartFile ==
				(FLMUINT)m_pDatabase->m_uncommittedDbHdr.ui32RflCurrFileNum);

			uiLowFileNum = m_uiTransStartFile + 1;
			uiHighFileNum = m_pCurrentBuf->uiCurrFileNum;

			// Close the current file so it can be deleted.

			if (RC_BAD( rc = waitForCommit( pDb->m_hWaitSem)))
			{
				goto Exit;
			}
			closeFile();

			// Delete as many of the files as possible.  Don't worry
			// about errors here.

			while (uiLowFileNum <= uiHighFileNum)
			{
				FLMUINT	uiFileNameSize = sizeof( szRflFileName);
				FLMBOOL	bNameTruncated;

				getFullRflFileName( uiLowFileNum, szRflFileName,
											&uiFileNameSize, &bNameTruncated);
				if (!bNameTruncated)
				{
					(void)gv_SFlmSysData.pFileSystem->deleteFile( szRflFileName);
				}
				uiLowFileNum++;
			}
		}
		else
		{
			// If we are in the file the transaction started in, simply
			// reset to where the transaction started.

			if (RC_BAD( rc2 = positionTo( m_uiTransStartAddr)))
			{
				// If we got to this point because of a
				// "goto Throw_Away_Transaction", we don't want to
				// clobber the original error code.  So, we use rc2
				// temporarily and then determine if its value should
				// be set into rc.

				if( RC_OK( rc))
				{
					rc = rc2;
				}
				rc2 = NE_SFLM_OK;
				goto Exit;
			}
		}
	}
	else
	{
		// Log a commit or abort packet.

		uiMaxPacketBodyLen = FLM_MAX_SEN_LEN;

		// Make sure we have space in the RFL buffer for a complete packet.

		if( !haveBuffSpace( uiMaxPacketBodyLen + RFL_PACKET_OVERHEAD))
		{
			if (RC_BAD( rc = flush( pDb, m_pCurrentBuf)))
			{
				goto Throw_Away_Transaction;
			}
		}

		// Get a pointer to where we will be laying down the packet body.

		pucPacketBody = pucPacketStart = getPacketBodyPtr();

		// Output the transaction ID.

		f_encodeSEN( m_ui64CurrTransID, &pucPacketBody);

		// Finish the packet

		uiPacketBodyLen = (FLMUINT)(pucPacketBody - pucPacketStart);
		flmAssert(  uiPacketBodyLen <= uiMaxPacketBodyLen);

		if (RC_BAD( rc = finishPacket( pDb, uiPacketType, 
			uiPacketBodyLen, FALSE)))
		{
			goto Throw_Away_Transaction;
		}

		finalizeTransaction();

		if( pbLoggedTransEnd)
		{
			*pbLoggedTransEnd = TRUE;
		}
	}

Exit:

	m_ui64CurrTransID = 0;
	return( RC_BAD( rc) ? rc : rc2);
}

/********************************************************************
Desc:	Log a block chain free packet
*********************************************************************/
RCODE F_Rfl::logBlockChainFree(
	F_Db *		pDb,
	FLMUINT64	ui64MaintDocID,
	FLMUINT		uiStartBlkAddr,
	FLMUINT		uiEndBlkAddr,
	FLMUINT		uiCount)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBYTE *	pucPacketStart;
	FLMBYTE *	pucPacketBody;
	FLMUINT		uiPacketBodyLen;
	FLMUINT		uiMaxPacketBodyLen;

	flmAssert( pDb->m_uiFlags & FDB_HAS_FILE_LOCK);
#ifndef FLM_DEBUG
	F_UNREFERENCED_PARM( pDb);
#endif
	
	// Do nothing if logging is disabled.

	if (!isLoggingEnabled())
	{
		goto Exit;
	}

	// Better be in the middle of a transaction.

	flmAssert( m_ui64CurrTransID);

	// Increment the operation count
	
	m_uiOperCount++;
	
	// Determine the maximum packet size
	
	uiMaxPacketBodyLen = FLM_MAX_SEN_LEN * 4;

	// Make sure we have space in the RFL buffer for a complete packet.

	if( !haveBuffSpace( uiMaxPacketBodyLen + RFL_PACKET_OVERHEAD))
	{
		if( RC_BAD( rc = flush( pDb, m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	// Get a pointer to where we will be laying down the packet body.

	pucPacketBody = pucPacketStart = getPacketBodyPtr();

	// Output the maintenance document ID

	f_encodeSEN( ui64MaintDocID, &pucPacketBody);

	// Output the starting block address

	f_encodeSEN( uiStartBlkAddr, &pucPacketBody);

	// Output the ending block address

	f_encodeSEN( uiEndBlkAddr, &pucPacketBody);

	// Output the block count

	f_encodeSEN( uiCount, &pucPacketBody);

	// Finish the packet

	uiPacketBodyLen = (FLMUINT)(pucPacketBody - pucPacketStart);
	flmAssert( uiPacketBodyLen <= uiMaxPacketBodyLen);

	if( RC_BAD( rc = finishPacket( pDb, RFL_BLK_CHAIN_FREE_PACKET, 
		uiPacketBodyLen, FALSE)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/********************************************************************
Desc:	Free a chain of blocks
*********************************************************************/
RCODE F_Rfl::recovBlockChainFree(
	F_Db *				pDb,
	const FLMBYTE *	pucPacketBody,
	FLMUINT				uiPacketBodyLen,
	eRestoreAction *	peAction)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT64			ui64MaintRowId;
	FLMUINT				uiStartBlkAddr;
	FLMUINT				uiEndBlkAddr;
	FLMUINT				uiCount;
	FLMUINT				uiBlocksFreed;
	const FLMBYTE *	pucEnd = pucPacketBody + uiPacketBodyLen;

	if( RC_BAD( rc = f_decodeSEN64( &pucPacketBody, pucEnd, &ui64MaintRowId)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd, &uiStartBlkAddr)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd, &uiEndBlkAddr)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd, &uiCount)))
	{
		goto Exit;
	}

	if( pucPacketBody != pucEnd)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BAD_RFL_PACKET);
		goto Exit;
	}

	if( m_pRestoreStatus)
	{
		if( RC_BAD( rc = m_pRestoreStatus->reportBlockChainFree( 
			peAction, m_ui64CurrTransID, ui64MaintRowId, uiStartBlkAddr,
			uiEndBlkAddr, uiCount)))
		{
			goto Exit;
		}

		if( *peAction == SFLM_RESTORE_ACTION_STOP)
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = pDb->maintBlockChainFree( 
		ui64MaintRowId, uiStartBlkAddr, uiCount, uiEndBlkAddr, &uiBlocksFreed)))
	{
		goto Exit;
	}

	if( uiCount != uiBlocksFreed)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_DATA_ERROR);
		goto Exit;
	}

Exit:

	return( rc);
}

/********************************************************************
Desc:		Log index suspend and resume packets
*********************************************************************/
RCODE F_Rfl::logIndexSuspendOrResume(
	F_Db *		pDb,
	FLMUINT		uiIndexNum,
	FLMUINT		uiPacketType)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBYTE *	pucPacketStart;
	FLMBYTE *	pucPacketBody;
	FLMUINT		uiPacketBodyLen;
	FLMUINT		uiMaxPacketBodyLen;

	flmAssert( pDb->m_uiFlags & FDB_HAS_FILE_LOCK);
#ifndef FLM_DEBUG
	F_UNREFERENCED_PARM( pDb);
#endif
	
	// Do nothing if logging is disabled.

	if (!isLoggingEnabled())
	{
		goto Exit;
	}

	// Better be in the middle of a transaction.

	flmAssert( m_ui64CurrTransID);

	// Increment the operation count
	
	m_uiOperCount++;
	
	// Determine the maximum packet size

	uiMaxPacketBodyLen = FLM_MAX_SEN_LEN;

	// Make sure we have space in the RFL buffer for a complete packet.

	if( !haveBuffSpace( uiMaxPacketBodyLen + RFL_PACKET_OVERHEAD))
	{
		if( RC_BAD( rc = flush( pDb, m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	// Get a pointer to where we will be laying down the packet body.

	pucPacketBody = pucPacketStart = getPacketBodyPtr();

	// Output the index number.

	flmAssert( uiIndexNum <= FLM_MAX_UINT16);
	f_encodeSEN( uiIndexNum, &pucPacketBody);

	// Finish the packet

	uiPacketBodyLen = (FLMUINT)(pucPacketBody - pucPacketStart);
	flmAssert( uiPacketBodyLen <= uiMaxPacketBodyLen);

	if (RC_BAD( rc = finishPacket( pDb, uiPacketType, uiPacketBodyLen, FALSE)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/********************************************************************
Desc:	Suspend or resume an index during restore or recovery.
*********************************************************************/
RCODE F_Rfl::recovIndexSuspendResume(
	F_Db *				pDb,
	FLMUINT				uiPacketType,
	const FLMBYTE *	pucPacketBody,
	FLMUINT				uiPacketBodyLen,
	eRestoreAction *	peAction)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiIndexNum;
	const FLMBYTE *	pucEnd = pucPacketBody + uiPacketBodyLen;

	if( RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd, &uiIndexNum)))
	{
		goto Exit;
	}

	if( pucPacketBody != pucEnd)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BAD_RFL_PACKET);
		goto Exit;
	}

	if( m_pRestoreStatus)
	{
		if( uiPacketType == RFL_INDEX_SUSPEND_PACKET)
		{
			if( RC_BAD( rc = m_pRestoreStatus->reportIndexSuspend( 
				peAction, m_ui64CurrTransID, uiIndexNum)))
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = m_pRestoreStatus->reportIndexResume(
				peAction, m_ui64CurrTransID, uiIndexNum)))
			{
				goto Exit;
			}
		}

		if( *peAction == SFLM_RESTORE_ACTION_STOP)
		{
			goto Exit;
		}
	}

	if( uiPacketType == RFL_INDEX_SUSPEND_PACKET)
	{
		if( RC_BAD( rc = pDb->indexSuspend( uiIndexNum)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = pDb->indexResume( uiIndexNum)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/********************************************************************
Desc:		Log a reduce packet
*********************************************************************/
RCODE F_Rfl::logReduce(
	F_Db *		pDb,
	FLMUINT		uiCount)
{
	RCODE				rc = NE_SFLM_OK;
	FLMBYTE *		pucPacketStart;
	FLMBYTE *		pucPacketBody;
	FLMUINT			uiPacketBodyLen;
	FLMUINT			uiMaxPacketBodyLen;

	flmAssert( pDb->m_uiFlags & FDB_HAS_FILE_LOCK);
#ifndef FLM_DEBUG
	F_UNREFERENCED_PARM( pDb);
#endif
	
	// Do nothing if logging is disabled.

	if (!isLoggingEnabled())
	{
		goto Exit;
	}

	// We need to set up to log this packet as if we
	// were logging a transaction.  The only difference
	// is that we don't log the begin transaction packet.

	if( RC_BAD( rc = setupTransaction( pDb)))
	{
		goto Exit;
	}

	uiMaxPacketBodyLen = 2 * FLM_MAX_SEN_LEN;

	// Make sure we have space in the RFL buffer for a complete packet.

	if( !haveBuffSpace( uiMaxPacketBodyLen + RFL_PACKET_OVERHEAD))
	{
		if( RC_BAD( rc = flush( pDb, m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	// Get a pointer to where we will be laying down the packet body.

	pucPacketBody = pucPacketStart = getPacketBodyPtr();

	// Output the transaction ID.

	f_encodeSEN( pDb->m_ui64CurrTransID, &pucPacketBody);

	// Output the count

	f_encodeSEN( uiCount, &pucPacketBody);

	// Finish the packet

	uiPacketBodyLen = (FLMUINT)(pucPacketBody - pucPacketStart);
	flmAssert( uiPacketBodyLen <= uiMaxPacketBodyLen);

	if( RC_BAD( rc = finishPacket( pDb, RFL_REDUCE_PACKET, 
		uiPacketBodyLen, TRUE)))
	{
		goto Exit;
	}

	// Finalize the transaction (as if we were committing a transaction)

	finalizeTransaction();

Exit:

	return( rc);
}

/********************************************************************
Desc:	Reduce the database during restore or recovery.
*********************************************************************/
RCODE F_Rfl::recovReduce(
	F_Db *				pDb,
	const FLMBYTE *	pucPacketBody,
	FLMUINT				uiPacketBodyLen,
	eRestoreAction *	peAction)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiCount;
	const FLMBYTE *	pucEnd = pucPacketBody + uiPacketBodyLen;

	if( RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd, &uiCount)))
	{
		goto Exit;
	}

	if( pucPacketBody != pucEnd)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BAD_RFL_PACKET);
		goto Exit;
	}

	if( m_pRestoreStatus)
	{
		if( RC_BAD( rc = m_pRestoreStatus->reportReduce(
			peAction, m_ui64CurrTransID, uiCount)))
		{
			goto Exit;
		}

		if( *peAction == SFLM_RESTORE_ACTION_STOP)
		{
			// Need to set m_ui64CurrTransID to 0 since it was
			// set by getPacket().  We are not going to
			// start a transaction because of the user's request
			// to exit.

			m_ui64CurrTransID = 0;
			goto Exit;
		}
	}
	
	if( RC_BAD( rc = pDb->reduceSize( uiCount, &uiCount)))
	{
		goto Exit;
	}

Exit:

	m_ui64CurrTransID = 0;
	return( rc);
}

/********************************************************************
Desc:		Log a database conversion packet
Note:		This routine performs most of the setup for logging a full
			transaction, but it does not cause begin and commit packets
			to be logged.  It is a "standalone" transaction.
*********************************************************************/
RCODE F_Rfl::logUpgrade(
	F_Db *			pDb,
	FLMUINT			uiOldVersion)
{
	RCODE				rc = NE_SFLM_OK;
	FLMBYTE *		pucPacketStart;
	FLMBYTE *		pucPacketBody;
	FLMUINT			uiPacketBodyLen;
	FLMUINT			uiMaxPacketBodyLen;

	flmAssert( pDb->m_uiFlags & FDB_HAS_FILE_LOCK);
#ifndef FLM_DEBUG
	F_UNREFERENCED_PARM( pDb);
#endif
	
	// Do nothing if logging is disabled.

	if (!isLoggingEnabled())
	{
		goto Exit;
	}

	// We need to set up to log this packet as if we
	// were logging a transaction.  The only difference
	// is that we don't log the begin transaction packet.

	if( RC_BAD( rc = setupTransaction( pDb)))
	{
		goto Exit;
	}

	uiMaxPacketBodyLen = 3 * FLM_MAX_SEN_LEN;

	// Make sure we have space in the RFL buffer for a complete packet.

	if( !haveBuffSpace( uiMaxPacketBodyLen + RFL_PACKET_OVERHEAD))
	{
		if (RC_BAD( rc = flush( pDb, m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	// Get a pointer to where we will be laying down the packet body.

	pucPacketBody = pucPacketStart = getPacketBodyPtr();

	// Output the transaction ID

	f_encodeSEN( pDb->m_ui64CurrTransID, &pucPacketBody);

	// Output the old database version

	f_encodeSEN( uiOldVersion, &pucPacketBody);

	// Output the new database version

	f_encodeSEN( SFLM_CURRENT_VERSION_NUM, &pucPacketBody);

	// Finish the packet

	uiPacketBodyLen = (FLMUINT)(pucPacketBody - pucPacketStart);
	flmAssert( uiPacketBodyLen <= uiMaxPacketBodyLen);

	if( RC_BAD( rc = finishPacket( pDb, RFL_UPGRADE_PACKET,
		uiPacketBodyLen, TRUE)))
	{
		goto Exit;
	}

	// Finalize the transaction (as if we were committing a transaction)

	finalizeTransaction();

Exit:

	m_ui64CurrTransID = 0;
	return( rc);
}

/********************************************************************
Desc:	Upgrade the database during restore or recovery.
*********************************************************************/
RCODE F_Rfl::recovUpgrade(
	F_Db *				pDb,
	const FLMBYTE *	pucPacketBody,
	FLMUINT				uiPacketBodyLen,
	eRestoreAction *	peAction)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiOldDbVersion;
	FLMUINT				uiNewDbVersion;

	if( uiPacketBodyLen != 8)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BAD_RFL_PACKET);
		goto Exit;
	}
	
	uiOldDbVersion = (FLMUINT)FB2UD( pucPacketBody);
	pucPacketBody += 4;
	
	uiNewDbVersion = (FLMUINT)FB2UD( pucPacketBody);
	pucPacketBody += 4;
	
	if( m_pRestoreStatus)
	{
		if( RC_BAD( rc = m_pRestoreStatus->reportUpgrade( 
			peAction, m_ui64CurrTransID, uiOldDbVersion, uiNewDbVersion)))
		{
			goto Exit;
		}

		if( *peAction == SFLM_RESTORE_ACTION_STOP)
		{
			// Need to set m_ui64CurrTransID to 0 since it was
			// set by getPacket().  We are not going to
			// start a transaction because of the user's request
			// to exit.

			m_ui64CurrTransID = 0;
			goto Exit;
		}
	}

	// Attempt the conversion if the current version is less
	// than the target version and the target version is
	// less than or equal to the highest version supported
	// by this code.

	if( uiNewDbVersion > SFLM_CURRENT_VERSION_NUM)
	{
		rc = RC_SET( NE_SFLM_UNALLOWED_UPGRADE);
		goto Exit;
	}
	else if( (FLMUINT)m_pDatabase->m_lastCommittedDbHdr.ui32DbVersion <
					uiNewDbVersion)
	{
		// The logged "new" version may be a lesser version
		// than SFLM_CURRENT_VERSION_NUM, which is what FlmDbUpgrade
		// upgrades to.  This is O.K. because the current version
		// should support all packets in the RFL for versions
		// that are less than it.  Otherwise, the RFL chain
		// would have been broken by the original upgrade and it
		// would not have logged an upgrade packet.

		if( RC_BAD( rc = pDb->upgrade( NULL)))
		{
			goto Exit;
		}
	}

Exit:

	m_ui64CurrTransID = 0;
	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
RCODE F_Rfl::logEncryptionKey(
	F_Db *			pDb,
	FLMUINT			uiPacketType,
	FLMBYTE *		pucKey,
	FLMUINT32		ui32KeyLen)
{	
	RCODE				rc = NE_SFLM_OK;
	FLMBYTE *		pucPacketStart;
	FLMBYTE *		pucPacketBody;
	FLMUINT			uiPacketBodyLen;
	FLMUINT			uiMaxPacketBodyLen;

	flmAssert( pDb->m_uiFlags & FDB_HAS_FILE_LOCK);
#ifndef FLM_DEBUG
	F_UNREFERENCED_PARM( pDb);
#endif

	flmAssert( uiPacketType == RFL_WRAP_KEY_PACKET ||
				  uiPacketType == RFL_ENABLE_ENCRYPTION_PACKET);

	// Do nothing if logging is disabled.

	if (!isLoggingEnabled())
	{
		goto Exit;
	}

	if( RC_BAD( rc = setupTransaction( pDb)))
	{
		goto Exit;
	}

	uiMaxPacketBodyLen = (2 * FLM_MAX_SEN_LEN) + ui32KeyLen;

	// Make sure we have space in the RFL buffer for a complete packet.

	if( !haveBuffSpace( uiMaxPacketBodyLen + RFL_PACKET_OVERHEAD))
	{
		if( RC_BAD( rc = flush( pDb, m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	// Get a pointer to where we will be laying down the packet body.

	pucPacketBody = pucPacketStart = getPacketBodyPtr();

	// Output the transaction ID

	f_encodeSEN( pDb->m_ui64CurrTransID, &pucPacketBody);

	// Store the length of the key
	
	f_encodeSEN( ui32KeyLen, &pucPacketBody);
	
	// If we were built without encryption, the key length will be zero,
	// so no need to store the key.

	if( ui32KeyLen)
	{
		f_memcpy( pucPacketBody, pucKey, ui32KeyLen);
		pucPacketBody += ui32KeyLen;
	}

	// Finish the packet

	uiPacketBodyLen = (FLMUINT)(pucPacketBody - pucPacketStart);
	flmAssert( uiPacketBodyLen <= uiMaxPacketBodyLen);

	if( RC_BAD( rc = finishPacket( pDb, uiPacketType,
		uiPacketBodyLen, TRUE)))
	{
		goto Exit;
	}

	finalizeTransaction();

Exit:

	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
RCODE F_Rfl::recovEncryptionKey(
	F_Db *				pDb,
	FLMUINT				uiPacketType,
	const FLMBYTE *	pucPacketBody,
	FLMUINT				uiPacketBodyLen,
	eRestoreAction *	peAction)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiDBKeyLen;
	FLMBOOL				bTransStarted = FALSE;
	SFLM_DB_HDR *		pUncommittedLogHdr = &m_pDatabase->m_uncommittedDbHdr;
	const FLMBYTE *	pucEnd = pucPacketBody + uiPacketBodyLen;

	if( RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd, &uiDBKeyLen)))
	{
		goto Exit;
	}

	if( pucPacketBody + uiDBKeyLen != pucEnd)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BAD_RFL_PACKET);
		goto Exit;
	}

	if( m_pRestoreStatus)
	{
		if( uiPacketType == RFL_WRAP_KEY_PACKET)
		{
			if( RC_BAD( rc = m_pRestoreStatus->reportWrapKey(
				peAction, m_ui64CurrTransID)))
			{
				goto Exit;
			}
		}
		else if( uiPacketType == RFL_ENABLE_ENCRYPTION_PACKET)
		{
			if( RC_BAD( rc = m_pRestoreStatus->reportEnableEncryption(
				peAction, m_ui64CurrTransID)))
			{
				goto Exit;
			}
		}
		else
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_BAD_RFL_PACKET);
			goto Exit;
		}

		if( *peAction == SFLM_RESTORE_ACTION_STOP)
		{
			m_ui64CurrTransID = 0;
			goto Exit;
		}
	}

	if( uiDBKeyLen)
	{
		// We cannot directly set the key at this point as it may be
		// encrypted using a password, which we do not have here.  We will
		// write the key out to the log header and trust the user to know whether
		// or not a password is needed to open the database.

		if( RC_BAD(rc = pDb->transBegin( SFLM_UPDATE_TRANS)))
		{
			goto Exit;
		}
		
		bTransStarted = TRUE;

		if( uiDBKeyLen > SFLM_MAX_ENC_KEY_SIZE)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_BAD_ENCKEY_SIZE);
			goto Exit;
		}

		f_memcpy( pUncommittedLogHdr->ucDbKey, pucPacketBody, uiDBKeyLen);
		pUncommittedLogHdr->ui32DbKeyLen = (FLMUINT32)uiDBKeyLen;

		pDb->m_bHadUpdOper = TRUE;

		if( RC_BAD( rc = pDb->commitTrans( 0, TRUE)))
		{
			goto Exit;
		}
		
		bTransStarted = FALSE;
	}

Exit:

	if( bTransStarted)
	{
		RCODE			tmpRc;

		if( RC_BAD( tmpRc = pDb->commitTrans( 0, TRUE)))
		{
			pDb->abortTrans();
			
			if (RC_OK( rc))
			{
				rc = tmpRc;
			}
		}
	}

	m_ui64CurrTransID = 0;
	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
RCODE F_Rfl::logEncDefKey(
	F_Db *			pDb,
	FLMUINT			uiEncDefId,
	FLMBYTE *		pucKeyValue,
	FLMUINT			uiKeyValueLen,
	FLMUINT			uiKeySize)
{	
	RCODE				rc = NE_SFLM_OK;
	FLMBYTE *		pucPacketStart;
	FLMBYTE *		pucPacketBody;
	FLMUINT			uiPacketBodyLen;
	FLMUINT			uiMaxPacketBodyLen;

	flmAssert( pDb->m_uiFlags & FDB_HAS_FILE_LOCK);
#ifndef FLM_DEBUG
	F_UNREFERENCED_PARM( pDb);
#endif
	
	// Do nothing if logging is disabled.

	if (!isLoggingEnabled())
	{
		goto Exit;
	}

	// Better be in the middle of a transaction.

	flmAssert( m_ui64CurrTransID);

	// Increment the operation count
	
	m_uiOperCount++;
	
	// Determine the maximum packet size
	
	uiMaxPacketBodyLen = (3 * FLM_MAX_SEN_LEN) + uiKeyValueLen;

	// Make sure we have space in the RFL buffer for a complete packet.

	if( !haveBuffSpace( uiMaxPacketBodyLen + RFL_PACKET_OVERHEAD))
	{
		if( RC_BAD( rc = flush( pDb, m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	// Get a pointer to where we will be laying down the packet body.

	pucPacketBody = pucPacketStart = getPacketBodyPtr();

	// Output the encryption definition ID

	f_encodeSEN( uiEncDefId, &pucPacketBody);
	
	// Output the key size (number of bits)
	
	f_encodeSEN( uiKeySize, &pucPacketBody);

	// Output the key value length

	f_encodeSEN( uiKeyValueLen, &pucPacketBody);

	// Output the key

	f_memcpy( pucPacketBody, pucKeyValue, uiKeyValueLen);
	pucPacketBody += uiKeyValueLen;
	
	// Finish the packet

	uiPacketBodyLen = (FLMUINT)(pucPacketBody - pucPacketStart);
	flmAssert( uiPacketBodyLen <= uiMaxPacketBodyLen);

	if (RC_BAD( rc = finishPacket( pDb, RFL_ENC_DEF_KEY_PACKET,
		uiPacketBodyLen, FALSE)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
RCODE F_Rfl::recovEncDefKey(
	F_Db *,				// pDb,
	const FLMBYTE *	pucPacketBody,
	FLMUINT				uiPacketBodyLen,
	eRestoreAction *)	// peAction
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT64			ui64RowId;
	FLMUINT				uiKeySize;
	FLMUINT				uiKeyValueLen;
	const FLMBYTE *	pucEnd = pucPacketBody + uiPacketBodyLen;

	if( RC_BAD( rc = f_decodeSEN64( &pucPacketBody, pucEnd, &ui64RowId)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd, &uiKeySize)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd, &uiKeyValueLen)))
	{
		goto Exit;
	}
	
	if( pucPacketBody + uiKeyValueLen != pucEnd)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BAD_RFL_PACKET);
		goto Exit;
	}
	
	// Retrieve the encryption definition
	
//VISIT: NEED TO FLESH THIS OUT
	
#if 0
	if( RC_BAD( rc = pDb->getNode( XFLM_DICT_COLLECTION, 
		ui64RootId, FLM_EXACT, &pNode)))
	{
		if( rc == NE_SFLM_ROW_NOT_FOUND)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_BAD_RFL_PACKET);
		}

		goto Exit;
	}
	
	// Set the key in the DOM node as a binary string.
	
	if( RC_BAD( rc = pNode->createAttribute( pDb, ATTR_ENCRYPTION_KEY_TAG,
		(IF_DOMNode **)&pAttr)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pAttr->removeModeFlags( pDb,
							FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pAttr->setBinary( pDb, pucPacketBody, uiKeyValueLen)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pAttr->addModeFlags( pDb,
							FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
	{
		goto Exit;
	}

	// Set the key size
	
	if( RC_BAD( rc = pNode->createAttribute( pDb, ATTR_ENCRYPTION_KEY_SIZE_TAG,
									(IF_DOMNode **)&pAttr)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pAttr->removeModeFlags( pDb,
							FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pAttr->setUINT( pDb, uiKeySize)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pAttr->addModeFlags( pDb,
							FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
	{
		goto Exit;
	}
#endif
	
Exit:

	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
RCODE F_Rfl::logRollOverDbKey(
	F_Db *			pDb)
{	
	RCODE				rc = NE_SFLM_OK;
	FLMBYTE *		pucPacketStart;
	FLMBYTE *		pucPacketBody;
	FLMUINT			uiPacketBodyLen;
	FLMUINT			uiMaxPacketBodyLen;

	flmAssert( pDb->m_uiFlags & FDB_HAS_FILE_LOCK);
#ifndef FLM_DEBUG
	F_UNREFERENCED_PARM( pDb);
#endif

	// Do nothing if logging is disabled.

	if (!isLoggingEnabled())
	{
		goto Exit;
	}

	if( RC_BAD( rc = setupTransaction( pDb)))
	{
		goto Exit;
	}

	uiMaxPacketBodyLen = FLM_MAX_SEN_LEN;

	// Make sure we have space in the RFL buffer for a complete packet.

	if( !haveBuffSpace( uiMaxPacketBodyLen + RFL_PACKET_OVERHEAD))
	{
		if( RC_BAD( rc = flush( pDb, m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	// Get a pointer to where we will be laying down the packet body.

	pucPacketBody = pucPacketStart = getPacketBodyPtr();

	// Output the transaction ID

	f_encodeSEN( pDb->m_ui64CurrTransID, &pucPacketBody);

	// Finish the packet

	uiPacketBodyLen = (FLMUINT)(pucPacketBody - pucPacketStart);
	flmAssert( uiPacketBodyLen <= uiMaxPacketBodyLen);

	if( RC_BAD( rc = finishPacket( pDb, RFL_ROLL_OVER_DB_KEY_PACKET, 
		uiPacketBodyLen, TRUE)))
	{
		goto Exit;
	}

	finalizeTransaction();

Exit:

	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
RCODE F_Rfl::recovRollOverDbKey(
	F_Db *				pDb,
	const FLMBYTE *,	// pucPacketBody,
	FLMUINT				uiPacketBodyLen,
	eRestoreAction *	peAction)
{
	RCODE					rc = NE_SFLM_OK;

	if( uiPacketBodyLen)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BAD_RFL_PACKET);
		goto Exit;
	}
	
	if( m_pRestoreStatus)
	{
		if( RC_BAD( rc = m_pRestoreStatus->reportRollOverDbKey(
			peAction, m_ui64CurrTransID)))
		{
			goto Exit;
		}

		if( *peAction == SFLM_RESTORE_ACTION_STOP)
		{
			m_ui64CurrTransID = 0;
			goto Exit;
		}
	}
	
	if( RC_BAD( rc = pDb->rollOverDbKey()))
	{
		goto Exit;
	}

Exit:

	m_ui64CurrTransID = 0;
	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
RCODE F_Rfl::logColumnDefs(
	F_Db *				pDb,
	F_COLUMN_DEF *		pColumnDefs)
{
	RCODE				rc = NE_SFLM_OK;
	F_COLUMN_DEF *	pColumnDef;
	FLMUINT			uiPacketLen = RFL_PACKET_OVERHEAD;
	FLMBYTE *		pucOut;

	flmAssert( isLoggingEnabled());

	if( !haveBuffSpace( RFL_PACKET_OVERHEAD))
	{
		if( RC_BAD( rc = flush( pDb, m_pCurrentBuf)))
		{
			goto Exit;
		}
	}
	
	// Go through each column's definition
	
	for (pColumnDef = pColumnDefs; pColumnDef; pColumnDef = pColumnDef->pNext)
	{
		
		if( RC_BAD( rc = makeRoom( pDb,
			pColumnDef->uiColumnNameLen + FLM_MAX_SEN_LEN * 4 + 1,
			&uiPacketLen, RFL_COLUMN_DEF_PACKET, NULL, NULL)))
		{
			goto Exit;
		}
		
		pucOut = getPacketPtr() + uiPacketLen;
		
		// Output the column name length and column name.

		f_encodeSEN( pColumnDef->uiColumnNameLen, &pucOut);
		f_memcpy( pucOut, pColumnDef->pszColumnName, pColumnDef->uiColumnNameLen);
		pucOut += pColumnDef->uiColumnNameLen;
		
		// Output the data type.
		
		*pucOut++ = (FLMBYTE)pColumnDef->eColumnDataType;
		
		// Output the maximum length
		
		f_encodeSEN( pColumnDef->uiMaxLen, &pucOut);
		
		// Output the flags
		
		f_encodeSEN( pColumnDef->uiFlags, &pucOut);
		
		// Output the encryption definition number
	
		f_encodeSEN( pColumnDef->uiEncDefNum, &pucOut);
	}
	
	// Add a column number of zero to terminate the columns.

	if (RC_BAD( rc = makeRoom( pDb, 1, &uiPacketLen,
												RFL_COLUMN_DEF_PACKET, NULL, NULL)))
	{
		goto Exit;
	}

	pucOut = getPacketPtr() + uiPacketLen;
	*pucOut++ = 0;
	uiPacketLen++;
	
	// Finish the packet.

	if (RC_BAD( rc = finishPacket( pDb, RFL_COLUMN_DEF_PACKET, 
		uiPacketLen - RFL_PACKET_OVERHEAD, FALSE)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
RCODE F_Rfl::logCreateTable(
	F_Db *				pDb,
	FLMUINT				uiTableNum,
	const char *		pszTableName,
	FLMUINT				uiTableNameLen,
	FLMUINT				uiEncDefNum,
	F_COLUMN_DEF *		pColumnDefs)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiPacketBodyLen;
	FLMBYTE *	pucPacketBody;
	FLMBYTE *	pucPacketStart;
	
	flmAssert( pDb->m_uiFlags & FDB_HAS_FILE_LOCK);
	
	// Do nothing if logging is disabled.

	if( !isLoggingEnabled())
	{
		goto Exit;
	}

	// Better be in the middle of a transaction.

	flmAssert( m_ui64CurrTransID);

	// Increment the operation count
	
	m_uiOperCount++;
	
	// Make sure we have space in the RFL buffer for a complete packet.  NOTE:
	// this is calculating the maximum packet body length that would be needed.

	if( !haveBuffSpace( uiTableNameLen + FLM_MAX_SEN_LEN * 3 + RFL_PACKET_OVERHEAD))
	{
		if( RC_BAD( rc = flush( pDb, m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	// Get a pointer to where we will be laying down the packet body.

	pucPacketBody = pucPacketStart = getPacketBodyPtr();

	// Output the table number.

	f_encodeSEN( uiTableNum, &pucPacketBody);
	
	// Output the table name length and table name.

	f_encodeSEN( uiTableNameLen, &pucPacketBody);
	f_memcpy( pucPacketBody, pszTableName, uiTableNameLen);
	pucPacketBody += uiTableNameLen;
	
	// Output the encryption definition number.

	f_encodeSEN( uiEncDefNum, &pucPacketBody);
	
	// Finish the packet - calculate the actual packet body length.

	uiPacketBodyLen = (FLMUINT)(pucPacketBody - pucPacketStart);

	if (RC_BAD( rc = finishPacket( pDb, RFL_CREATE_TABLE_PACKET,
		uiPacketBodyLen, FALSE)))
	{
		goto Exit;
	}
	
	// Now output the column definitions
	
	if (RC_BAD( rc = logColumnDefs( pDb, pColumnDefs)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}
		
/********************************************************************
Desc:
*********************************************************************/
RCODE F_Rfl::recovCreateTable(
	F_Db *				pDb,
	const FLMBYTE *	pucPacketBody,
	FLMUINT				uiPacketBodyLen,
	eRestoreAction *	peAction)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiTableNum;
	char *				pszTableName;
	FLMUINT				uiTableNameLen;
	FLMUINT				uiEncDefNum;
	char *				pszTmp;
	FLMUINT				uiNumColumnDefs;
	FLMUINT				uiPacketType;
	F_COLUMN_DEF *		pColumnDef;
	F_COLUMN_DEF *		pFirstColDef;
	F_COLUMN_DEF *		pLastColDef;
	FLMUINT				uiColumnNameLen;
	const FLMBYTE *	pucEnd = pucPacketBody + uiPacketBodyLen;
	FLMBOOL				bHitEnd;
	
	// Get the table number from the packet
	
	if (RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd, &uiTableNum)))
	{
		goto Exit;
	}
	
	// Get the table name length and table name from the packet.

	if (RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd, &uiTableNameLen)))
	{
		goto Exit;
	}
	m_tmpPool.poolReset( NULL);
	if (RC_BAD( rc = m_tmpPool.poolAlloc( uiTableNameLen + 1,
				(void **)&pszTableName)))
	{
		goto Exit;
	}
	f_memcpy( pszTableName, pucPacketBody, uiTableNameLen);
	pucPacketBody += uiTableNameLen;
	pszTableName [uiTableNameLen] = 0;
	
	// Get the encryption definition number from the packet
	
	if (RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd, &uiEncDefNum)))
	{
		goto Exit;
	}
	// Get the column definitions.
	
	pucPacketBody = pucEnd = NULL;
	uiNumColumnDefs = 0;
	pFirstColDef = NULL;
	pLastColDef = NULL;
	for (;;)
	{

		// If we have used everything in our current packet, get another.

		if (pucPacketBody == pucEnd)
		{
			if( RC_BAD( rc = getPacket( 
				pDb, FALSE, &uiPacketType, &pucPacketBody, &uiPacketBodyLen,
				&bHitEnd)))
			{
				goto Exit;
			}
			
			// Should not hit the end of the RFL here!
			
			if (bHitEnd)
			{
				rc = RC_SET( NE_SFLM_RFL_INCOMPLETE);
				goto Exit;
			}
			
			if (uiPacketType != RFL_COLUMN_DEF_PACKET)
			{
				rc = RC_SET( NE_SFLM_BAD_RFL_PACKET);
				goto Exit;
			}
			pucEnd = pucPacketBody + uiPacketBodyLen;
		}
		
		// Get the column name length and name.
		
		if (RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd,
									&uiColumnNameLen)))
		{
			goto Exit;
		}
		if (!uiColumnNameLen)
		{
			break;
		}
		if (RC_BAD( rc = m_tmpPool.poolAlloc( sizeof( F_COLUMN_DEF),
									(void **)&pColumnDef)))
		{
			goto Exit;
		}
		uiNumColumnDefs++;
		pColumnDef->pNext = NULL;
		if (pLastColDef)
		{
			pLastColDef->pNext = pColumnDef;
		}
		else
		{
			pFirstColDef = pColumnDef;
		}
		pLastColDef = pColumnDef;
		pColumnDef->uiColumnNameLen = uiColumnNameLen;
		if (RC_BAD( rc = m_tmpPool.poolAlloc( uiColumnNameLen + 1,
					(void **)&pszTmp)))
		{
			goto Exit;
		}
		f_memcpy( pszTmp, pucPacketBody, uiColumnNameLen);
		pucPacketBody += uiColumnNameLen;
		pszTmp [uiColumnNameLen] = 0;
		pColumnDef->pszColumnName = pszTmp;
		
		// Get the data type.
		
		pColumnDef->eColumnDataType = (eDataType)(*pucPacketBody);
		switch (pColumnDef->eColumnDataType)
		{
			case SFLM_STRING_TYPE:
			case SFLM_NUMBER_TYPE:
			case SFLM_BINARY_TYPE:
				break;
			default:
				rc = RC_SET( NE_SFLM_BAD_RFL_PACKET);
				goto Exit;
		}
		pucPacketBody++;
		
		// Get the maximum length
		
		if (RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd,
									&pColumnDef->uiMaxLen)))
		{
			goto Exit;
		}
		
		// Get the flags
		
		if (RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd,
									&pColumnDef->uiFlags)))
		{
			goto Exit;
		}
		
		// Get the encryption definition number
	
		if (RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd,
									&pColumnDef->uiEncDefNum)))
		{
			goto Exit;
		}
	}
	
	if (m_pRestoreStatus)
	{
		if( RC_BAD( rc = m_pRestoreStatus->reportCreateTable(
			peAction, uiTableNum, pszTableName, uiTableNameLen,
			uiEncDefNum, pFirstColDef, uiNumColumnDefs)))
		{
			goto Exit;
		}

		if( *peAction == SFLM_RESTORE_ACTION_STOP)
		{
			m_ui64CurrTransID = 0;
			goto Exit;
		}
	}
	
	if( RC_BAD( rc = pDb->createTable( uiTableNum, pszTableName, uiTableNameLen,
									uiEncDefNum, pFirstColDef, uiNumColumnDefs)))
	{
		goto Exit;
	}

Exit:

	m_tmpPool.poolReset( NULL);

	m_ui64CurrTransID = 0;
	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
RCODE F_Rfl::logDropTable(
	F_Db *				pDb,
	FLMUINT				uiTableNum)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiPacketBodyLen;
	FLMBYTE *	pucPacketBody;
	FLMBYTE *	pucPacketStart;
	
	flmAssert( pDb->m_uiFlags & FDB_HAS_FILE_LOCK);
	
	// Do nothing if logging is disabled.

	if( !isLoggingEnabled())
	{
		goto Exit;
	}

	// Better be in the middle of a transaction.

	flmAssert( m_ui64CurrTransID);

	// Increment the operation count
	
	m_uiOperCount++;
	
	// Make sure we have space in the RFL buffer for a complete packet.  NOTE:
	// this is calculating the maximum packet body length that would be needed.

	if( !haveBuffSpace( FLM_MAX_SEN_LEN + RFL_PACKET_OVERHEAD))
	{
		if( RC_BAD( rc = flush( pDb, m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	// Get a pointer to where we will be laying down the packet body.

	pucPacketBody = pucPacketStart = getPacketBodyPtr();

	// Output the table number

	f_encodeSEN( uiTableNum, &pucPacketBody);

	// Finish the packet - calculate the actual packet body length.

	uiPacketBodyLen = (FLMUINT)(pucPacketBody - pucPacketStart);
	flmAssert( uiPacketBodyLen <= FLM_MAX_SEN_LEN);

	if (RC_BAD( rc = finishPacket( pDb, RFL_DROP_TABLE_PACKET,
		uiPacketBodyLen, FALSE)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}
		
/********************************************************************
Desc:
*********************************************************************/
RCODE F_Rfl::recovDropTable(
	F_Db *				pDb,
	const FLMBYTE *	pucPacketBody,
	FLMUINT				uiPacketBodyLen,
	eRestoreAction *	peAction)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiTableNum;
	const FLMBYTE *	pucEnd = pucPacketBody + uiPacketBodyLen;
	
	// Get the table number from the packet

	if (RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd, &uiTableNum)))
	{
		goto Exit;
	}
	
	if (m_pRestoreStatus)
	{
		if( RC_BAD( rc = m_pRestoreStatus->reportDropTable(
			peAction, uiTableNum)))
		{
			goto Exit;
		}

		if( *peAction == SFLM_RESTORE_ACTION_STOP)
		{
			m_ui64CurrTransID = 0;
			goto Exit;
		}
	}
	
	if( RC_BAD( rc = pDb->dropTable( uiTableNum)))
	{
		goto Exit;
	}

Exit:

	m_tmpPool.poolReset( NULL);

	m_ui64CurrTransID = 0;
	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
RCODE F_Rfl::logIndexColumnDefs(
	F_Db *				pDb,
	F_INDEX_COL_DEF *	pIxColDefs)
{
	RCODE					rc = NE_SFLM_OK;
	F_INDEX_COL_DEF *	pIxColDef;
	FLMUINT				uiPacketLen = RFL_PACKET_OVERHEAD;
	FLMBYTE *			pucOut;

	flmAssert( isLoggingEnabled());

	if( !haveBuffSpace( RFL_PACKET_OVERHEAD))
	{
		if( RC_BAD( rc = flush( pDb, m_pCurrentBuf)))
		{
			goto Exit;
		}
	}
	
	// Go through each column's definition
	
	for (pIxColDef = pIxColDefs; pIxColDef; pIxColDef = pIxColDef->pNext)
	{
		if( RC_BAD( rc = makeRoom( pDb,
			FLM_MAX_SEN_LEN * 4,
			&uiPacketLen, RFL_INDEX_COLUMN_DEF_PACKET, NULL, NULL)))
		{
			goto Exit;
		}
		
		pucOut = getPacketPtr() + uiPacketLen;
		
		// Output the column number

		f_encodeSEN( pIxColDef->uiColumnNum, &pucOut);
		
		// Output the flags.
		
		f_encodeSEN( pIxColDef->uiFlags, &pucOut);
		
		// Output the comparison rules
		
		f_encodeSEN( pIxColDef->uiCompareRules, &pucOut);
		
		// Output the limit
		
		f_encodeSEN( pIxColDef->uiLimit, &pucOut);
	}
	
	// Add a column number of zero to terminate the columns.

	if (RC_BAD( rc = makeRoom( pDb, 1, &uiPacketLen,
												RFL_INDEX_COLUMN_DEF_PACKET, NULL, NULL)))
	{
		goto Exit;
	}

	pucOut = getPacketPtr() + uiPacketLen;
	*pucOut++ = 0;
	uiPacketLen++;
	
	// Finish the packet.

	if (RC_BAD( rc = finishPacket( pDb, RFL_INDEX_COLUMN_DEF_PACKET, 
		uiPacketLen - RFL_PACKET_OVERHEAD, FALSE)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
RCODE F_Rfl::logCreateIndex(
	F_Db *				pDb,
	FLMUINT				uiTableNum,
	FLMUINT				uiIndexNum,
	const char *		pszIndexName,
	FLMUINT				uiIndexNameLen,
	FLMUINT				uiEncDefNum,
	F_INDEX_COL_DEF *	pIxColDefs,
	FLMUINT				uiFlags)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiPacketBodyLen;
	FLMBYTE *	pucPacketBody;
	FLMBYTE *	pucPacketStart;
	
	flmAssert( pDb->m_uiFlags & FDB_HAS_FILE_LOCK);
	
	// Do nothing if logging is disabled.

	if( !isLoggingEnabled())
	{
		goto Exit;
	}

	// Better be in the middle of a transaction.

	flmAssert( m_ui64CurrTransID);

	// Increment the operation count
	
	m_uiOperCount++;
	
	// Make sure we have space in the RFL buffer for a complete packet.  NOTE:
	// this is calculating the maximum packet body length that would be needed.

	if( !haveBuffSpace( uiIndexNameLen + FLM_MAX_SEN_LEN * 5 + RFL_PACKET_OVERHEAD))
	{
		if( RC_BAD( rc = flush( pDb, m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	// Get a pointer to where we will be laying down the packet body.

	pucPacketBody = pucPacketStart = getPacketBodyPtr();

	// Output the table number.

	f_encodeSEN( uiTableNum, &pucPacketBody);
	
	// Output the index number.

	f_encodeSEN( uiIndexNum, &pucPacketBody);
	
	// Output the index name length and index name.

	f_encodeSEN( uiIndexNameLen, &pucPacketBody);
	f_memcpy( pucPacketBody, pszIndexName, uiIndexNameLen);
	pucPacketBody += uiIndexNameLen;
	
	// Output the encryption definition number.

	f_encodeSEN( uiEncDefNum, &pucPacketBody);
	
	// Output the flags.

	f_encodeSEN( uiFlags, &pucPacketBody);
	
	// Finish the packet - calculate the actual packet body length.

	uiPacketBodyLen = (FLMUINT)(pucPacketBody - pucPacketStart);

	if (RC_BAD( rc = finishPacket( pDb, RFL_CREATE_INDEX_PACKET,
		uiPacketBodyLen, FALSE)))
	{
		goto Exit;
	}
	
	// Now output the column definitions for the index
	
	if (RC_BAD( rc = logIndexColumnDefs( pDb, pIxColDefs)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}
		
/********************************************************************
Desc:
*********************************************************************/
RCODE F_Rfl::recovCreateIndex(
	F_Db *				pDb,
	const FLMBYTE *	pucPacketBody,
	FLMUINT				uiPacketBodyLen,
	eRestoreAction *	peAction)
{
	RCODE						rc = NE_SFLM_OK;
	FLMUINT					uiTableNum;
	FLMUINT					uiIndexNum;
	char *					pszIndexName;
	FLMUINT					uiIndexNameLen;
	FLMUINT					uiEncDefNum;
	FLMUINT					uiFlags;
	FLMUINT					uiNumIxColDefs;
	FLMUINT					uiPacketType;
	F_INDEX_COL_DEF *		pIxColDef;
	F_INDEX_COL_DEF *		pFirstIxColDef;
	F_INDEX_COL_DEF *		pLastIxColDef;
	const FLMBYTE *		pucEnd = pucPacketBody + uiPacketBodyLen;
	FLMBOOL					bHitEnd;
	FLMUINT					uiColumnNum;
	
	// Get the table number from the packet
	
	if (RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd, &uiTableNum)))
	{
		goto Exit;
	}
	
	// Get the index number from the packet
	
	if (RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd, &uiIndexNum)))
	{
		goto Exit;
	}
	
	// Get the index name length and index name from the packet.

	if (RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd, &uiIndexNameLen)))
	{
		goto Exit;
	}
	m_tmpPool.poolReset( NULL);
	if (RC_BAD( rc = m_tmpPool.poolAlloc( uiIndexNameLen + 1,
				(void **)&pszIndexName)))
	{
		goto Exit;
	}
	f_memcpy( pszIndexName, pucPacketBody, uiIndexNameLen);
	pucPacketBody += uiIndexNameLen;
	pszIndexName [uiIndexNameLen] = 0;
	
	// Get the encryption definition number from the packet
	
	if (RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd, &uiEncDefNum)))
	{
		goto Exit;
	}
	
	// Get the index flags from the packet
	
	if (RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd, &uiFlags)))
	{
		goto Exit;
	}
	
	// Get the index column definitions.
	
	pucPacketBody = pucEnd = NULL;
	uiNumIxColDefs = 0;
	pFirstIxColDef = NULL;
	pLastIxColDef = NULL;
	for (;;)
	{

		// If we have used everything in our current packet, get another.

		if (pucPacketBody == pucEnd)
		{
			if( RC_BAD( rc = getPacket( 
				pDb, FALSE, &uiPacketType, &pucPacketBody, &uiPacketBodyLen,
				&bHitEnd)))
			{
				goto Exit;
			}
			
			// Should not hit the end of the RFL here!
			
			if (bHitEnd)
			{
				rc = RC_SET( NE_SFLM_RFL_INCOMPLETE);
				goto Exit;
			}
			
			if (uiPacketType != RFL_INDEX_COLUMN_DEF_PACKET)
			{
				rc = RC_SET( NE_SFLM_BAD_RFL_PACKET);
				goto Exit;
			}
			pucEnd = pucPacketBody + uiPacketBodyLen;
		}
		
		// Get the column number
		
		if (RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd,
									&uiColumnNum)))
		{
			goto Exit;
		}
		if (!uiColumnNum)
		{
			break;
		}
		if (RC_BAD( rc = m_tmpPool.poolAlloc( sizeof( F_INDEX_COL_DEF),
									(void **)&pIxColDef)))
		{
			goto Exit;
		}
		uiNumIxColDefs++;
		pIxColDef->pNext = NULL;
		if (pLastIxColDef)
		{
			pLastIxColDef->pNext = pIxColDef;
		}
		else
		{
			pFirstIxColDef = pIxColDef;
		}
		pLastIxColDef = pIxColDef;
		pIxColDef->uiColumnNum = uiColumnNum;

		// Get the flags
		
		if (RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd,
									&pIxColDef->uiFlags)))
		{
			goto Exit;
		}
		
		// Get the comparison rules
		
		if (RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd,
									&pIxColDef->uiCompareRules)))
		{
			goto Exit;
		}
		
		// Get the limit
		
		if (RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd,
									&pIxColDef->uiLimit)))
		{
			goto Exit;
		}
	}
	
	if (m_pRestoreStatus)
	{
		if( RC_BAD( rc = m_pRestoreStatus->reportCreateIndex(
			peAction, uiTableNum, uiIndexNum, pszIndexName, uiIndexNameLen,
			uiEncDefNum, uiFlags, pFirstIxColDef, uiNumIxColDefs)))
		{
			goto Exit;
		}

		if( *peAction == SFLM_RESTORE_ACTION_STOP)
		{
			m_ui64CurrTransID = 0;
			goto Exit;
		}
	}
	
	if( RC_BAD( rc = pDb->createIndex( uiTableNum, uiIndexNum, pszIndexName, uiIndexNameLen,
									uiEncDefNum, uiFlags, pFirstIxColDef, uiNumIxColDefs)))
	{
		goto Exit;
	}

Exit:

	m_tmpPool.poolReset( NULL);

	m_ui64CurrTransID = 0;
	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
RCODE F_Rfl::logDropIndex(
	F_Db *				pDb,
	FLMUINT				uiIndexNum)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiPacketBodyLen;
	FLMBYTE *	pucPacketBody;
	FLMBYTE *	pucPacketStart;
	
	flmAssert( pDb->m_uiFlags & FDB_HAS_FILE_LOCK);
	
	// Do nothing if logging is disabled.

	if( !isLoggingEnabled())
	{
		goto Exit;
	}

	// Better be in the middle of a transaction.

	flmAssert( m_ui64CurrTransID);

	// Increment the operation count
	
	m_uiOperCount++;
	
	// Make sure we have space in the RFL buffer for a complete packet.  NOTE:
	// this is calculating the maximum packet body length that would be needed.

	if( !haveBuffSpace( FLM_MAX_SEN_LEN + RFL_PACKET_OVERHEAD))
	{
		if( RC_BAD( rc = flush( pDb, m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	// Get a pointer to where we will be laying down the packet body.

	pucPacketBody = pucPacketStart = getPacketBodyPtr();

	// Output the index number

	f_encodeSEN( uiIndexNum, &pucPacketBody);

	// Finish the packet - calculate the actual packet body length.

	uiPacketBodyLen = (FLMUINT)(pucPacketBody - pucPacketStart);
	flmAssert( uiPacketBodyLen <= FLM_MAX_SEN_LEN);

	if (RC_BAD( rc = finishPacket( pDb, RFL_DROP_INDEX_PACKET,
		uiPacketBodyLen, FALSE)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}
		
/********************************************************************
Desc:
*********************************************************************/
RCODE F_Rfl::recovDropIndex(
	F_Db *				pDb,
	const FLMBYTE *	pucPacketBody,
	FLMUINT				uiPacketBodyLen,
	eRestoreAction *	peAction)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiIndexNum;
	const FLMBYTE *	pucEnd = pucPacketBody + uiPacketBodyLen;
	
	// Get the index number from the packet

	if (RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd, &uiIndexNum)))
	{
		goto Exit;
	}
	
	if (m_pRestoreStatus)
	{
		if( RC_BAD( rc = m_pRestoreStatus->reportDropIndex(
			peAction, uiIndexNum)))
		{
			goto Exit;
		}

		if( *peAction == SFLM_RESTORE_ACTION_STOP)
		{
			m_ui64CurrTransID = 0;
			goto Exit;
		}
	}
	
	if( RC_BAD( rc = pDb->dropIndex( uiIndexNum)))
	{
		goto Exit;
	}

Exit:

	m_tmpPool.poolReset( NULL);

	m_ui64CurrTransID = 0;
	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
RCODE F_Rfl::logIndexSet(
	F_Db *		pDb,
	FLMUINT		uiIndexNum,
	FLMUINT64	ui64StartRowId,
	FLMUINT64	ui64EndRowId)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiPacketBodyLen;
	FLMBYTE *	pucPacketBody;
	FLMBYTE *	pucPacketStart;
	
	flmAssert( pDb->m_uiFlags & FDB_HAS_FILE_LOCK);
	
	// Do nothing if logging is disabled.

	if( !isLoggingEnabled())
	{
		goto Exit;
	}

	// Better be in the middle of a transaction.

	flmAssert( m_ui64CurrTransID);

	// Increment the operation count
	
	m_uiOperCount++;
	
	// Make sure we have space in the RFL buffer for a complete packet.  NOTE:
	// this is calculating the maximum packet body length that would be needed.

	if (!haveBuffSpace( FLM_MAX_SEN_LEN * 3 + RFL_PACKET_OVERHEAD))
	{
		if( RC_BAD( rc = flush( pDb, m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	// Get a pointer to where we will be laying down the packet body.

	pucPacketBody = pucPacketStart = getPacketBodyPtr();

	// Output the index number.

	f_encodeSEN( uiIndexNum, &pucPacketBody);
	
	// Output the start row id.

	f_encodeSEN( ui64StartRowId, &pucPacketBody);
	
	// Output the end row id.

	f_encodeSEN( ui64EndRowId, &pucPacketBody);

	// Finish the packet - calculate the actual packet body length.

	uiPacketBodyLen = (FLMUINT)(pucPacketBody - pucPacketStart);
	if (RC_BAD( rc = finishPacket( pDb, RFL_INDEX_SET_PACKET,
		uiPacketBodyLen, FALSE)))
	{
		goto Exit;
	}
	
Exit:
	return( rc);
}
		
/********************************************************************
Desc:
*********************************************************************/
RCODE F_Rfl::recovIndexSet(
	F_Db *				pDb,
	const FLMBYTE *	pucPacketBody,
	FLMUINT				uiPacketBodyLen,
	eRestoreAction *	peAction)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiIndexNum;
	FLMUINT64			ui64StartRowId;
	FLMUINT64			ui64EndRowId;
	const FLMBYTE *	pucEnd = pucPacketBody + uiPacketBodyLen;
	
	// Get the index number from the packet
	
	if (RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd, &uiIndexNum)))
	{
		goto Exit;
	}
	
	// Get the start row ID from the packet
	
	if (RC_BAD( rc = f_decodeSEN64( &pucPacketBody, pucEnd, &ui64StartRowId)))
	{
		goto Exit;
	}
	
	// Get the end row ID from the packet
	
	if (RC_BAD( rc = f_decodeSEN64( &pucPacketBody, pucEnd, &ui64EndRowId)))
	{
		goto Exit;
	}
	
	if (m_pRestoreStatus)
	{
		if( RC_BAD( rc = m_pRestoreStatus->reportIndexSet(
			peAction, uiIndexNum, ui64StartRowId, ui64EndRowId)))
		{
			goto Exit;
		}

		if( *peAction == SFLM_RESTORE_ACTION_STOP)
		{
			m_ui64CurrTransID = 0;
			goto Exit;
		}
	}
	
	if (RC_BAD( rc = pDb->indexSetOfRows( uiIndexNum, ui64StartRowId,
			ui64EndRowId, NULL, NULL, NULL, NULL, NULL)))
	{
		goto Exit;
	}

Exit:

	m_ui64CurrTransID = 0;
	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
RCODE F_Rfl::logColumnValues(
	F_Db *				pDb,
	FLMUINT				uiTableNum,
	F_COLUMN_VALUE *	pColumnValues)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiPacketLen = RFL_PACKET_OVERHEAD;
	FLMUINT				uiBytesLeftToWrite;
	FLMUINT				uiBytesToWrite;
	FLMBYTE *			pucValue;
	FLMBYTE *			pucOut;
	FLMBYTE *			pucIV;
	FLMBYTE *			pucStart;
	F_TABLE *			pTable = pDb->m_pDict->getTable( uiTableNum);
	F_COLUMN *			pColumn;
	F_ENCDEF *			pTableEncDef = (pTable->lfInfo.uiEncDefNum)
											? pDb->m_pDict->getEncDef( pTable->lfInfo.uiEncDefNum)
											: (F_ENCDEF *)NULL;
	F_ENCDEF *			pEncDef;
	FLMUINT				uiIVLen;
	FLMUINT				uiEncLen;
	FLMUINT				uiSenLen1;
	FLMUINT				uiSenLen2;
	FLMUINT				uiSpaceNeeded;
	FLMUINT				uiEncOutputLen;
	FLMUINT				uiBytesAvail;
	F_COLUMN_VALUE *	pColumnValue;

	flmAssert( isLoggingEnabled());

	if( !haveBuffSpace( RFL_PACKET_OVERHEAD))
	{
		if( RC_BAD( rc = flush( pDb, m_pCurrentBuf)))
		{
			goto Exit;
		}
	}
	
	// Go through each column's data
	
	for (pColumnValue = pColumnValues; pColumnValue; pColumnValue = pColumnValue->pNext)
	{
		pColumn = pDb->m_pDict->getColumn( pTable, pColumnValue->uiColumnNum);
		
		if( RC_BAD( rc = makeRoom( pDb, FLM_MAX_SEN_LEN * 2,
			&uiPacketLen, RFL_COLUMN_DATA_PACKET, NULL, NULL)))
		{
			goto Exit;
		}
		
		pucOut = pucStart = getPacketPtr() + uiPacketLen;

		// Output the column number.
	
		f_encodeSEN( pColumnValue->uiColumnNum, &pucOut);
		
		// Output the value length
	
		f_encodeSEN( pColumnValue->uiValueLen, &pucOut);
		
		uiPacketLen += (FLMUINT)(pucOut - pucStart);
		
		uiBytesLeftToWrite = pColumnValue->uiValueLen;
		pucValue = pColumnValue->pucColumnValue;
		while (uiBytesLeftToWrite)
		{
			
			// If the column or table are encrypted, we need to make enough room
			// to encrypt the data.
			
			pEncDef = (pColumn->uiEncDefNum)
						 ? pDb->m_pDict->getEncDef( pColumn->uiEncDefNum)
						 : pTableEncDef;
						 
			if (!pEncDef)
			{
				if (RC_BAD( rc = makeRoom( pDb, uiBytesLeftToWrite,
					&uiPacketLen, RFL_COLUMN_DATA_PACKET, &uiBytesToWrite, NULL)))
				{
					goto Exit;
				}
		
				f_memcpy( getPacketPtr() + uiPacketLen, pucValue, uiBytesToWrite);
		
				pucValue += uiBytesToWrite;
				uiBytesLeftToWrite -= uiBytesToWrite;
				uiPacketLen += uiBytesToWrite;
			}
			else
			{
				uiIVLen = pEncDef->pCcs->getIVLen();
				flmAssert( uiIVLen == 8 || uiIVLen == 16);
				uiBytesToWrite = uiBytesLeftToWrite;
				uiEncLen = getEncLen( uiBytesToWrite);
				uiSenLen1 = f_getSENByteCount( uiEncLen);
				uiSenLen2 = f_getSENByteCount( uiBytesToWrite);
				uiSpaceNeeded = uiSenLen1 + uiSenLen2 + uiIVLen + uiEncLen;
				if (RC_BAD( rc = makeRoom( pDb, uiSpaceNeeded,
					&uiPacketLen, RFL_COLUMN_DATA_PACKET, &uiBytesAvail, NULL)))
				{
					goto Exit;
				}
				
				// If we can't hold the entire encrypted chunk, we must
				// recalculate down.
				
				if (uiBytesAvail < uiSpaceNeeded)
				{
					
					// If we cannot hold the encrypted length (as a SEN), the
					// IV, and a minimal encrypted chunk, we need to flush
					// out the current packet.
					
					if (uiBytesAvail < uiSenLen1 + uiSenLen2 + uiIVLen + ENCRYPT_MIN_CHUNK_SIZE)
					{
						goto Finish_Packet;
					}
					
					// Round down to fit whatever data will end right on an
					// encryption boundary.
					
					uiBytesToWrite = uiBytesAvail - uiSenLen1 - uiSenLen2 - uiIVLen;
					uiBytesToWrite -= extraEncBytes( uiBytesToWrite);
					uiEncLen = getEncLen( uiBytesToWrite);
					flmAssert( uiEncLen == uiBytesToWrite);
					
					uiSenLen1 = f_getSENByteCount( uiEncLen);
					uiSenLen2 = f_getSENByteCount( uiBytesToWrite);
					uiSpaceNeeded = uiSenLen1 + uiSenLen2 + uiIVLen + uiEncLen;
					if (RC_BAD( rc = makeRoom( pDb, uiSpaceNeeded,
						&uiPacketLen, RFL_COLUMN_DATA_PACKET, &uiBytesAvail, NULL)))
					{
						goto Exit;
					}
					
					// We should be guaranteed enough room to write everything we
					// asked for at this point.
					
					flmAssert( uiBytesAvail >= uiSpaceNeeded);
				}
				
				// First output the encrypted length, then the IV, and finally
				// the data.  Then encrypt the data in place.
				
				pucOut = getPacketPtr() + uiPacketLen;
				
				// Output the encrypted length
				
				f_encodeSEN( uiEncLen, &pucOut);
				
				// Output the actual data length
				
				f_encodeSEN( uiBytesToWrite, &pucOut);
				
				// Output the IV
				
				pucIV = pucOut;
				if( RC_BAD( rc = pEncDef->pCcs->generateIV( uiIVLen, pucIV)))
				{
					goto Exit;
				}
				pucOut += uiIVLen;
				
				// Output the unencrypted data and encrypt it in place.
				
				f_memcpy( pucOut, pucValue, uiBytesToWrite);
				if (RC_BAD( rc = pDb->encryptData( pEncDef->uiEncDefNum, pucIV, 
					pucOut, uiEncLen, uiBytesToWrite, &uiEncOutputLen)))
				{
					goto Exit;
				}
				flmAssert( uiEncOutputLen == uiEncLen);
				
				pucValue += uiBytesToWrite;
				uiBytesLeftToWrite -= uiBytesToWrite;
				uiPacketLen += (uiSenLen1 + uiSenLen2 + uiIVLen + uiEncLen);
			}

Finish_Packet:

			if( RC_BAD( rc = finishPacket( pDb, RFL_COLUMN_DATA_PACKET,
				uiPacketLen - RFL_PACKET_OVERHEAD, FALSE)))
			{
				goto Exit;
			}
			
			uiPacketLen = RFL_PACKET_OVERHEAD;
		}
	}
	
	// Add a column number of zero to terminate the columns.

	if (RC_BAD( rc = makeRoom( pDb, 1, &uiPacketLen,
												RFL_COLUMN_DATA_PACKET, NULL, NULL)))
	{
		goto Exit;
	}

	pucOut = getPacketPtr() + uiPacketLen;
	*pucOut++ = 0;
	uiPacketLen++;
	
	// Finish the packet.

	if (RC_BAD( rc = finishPacket( pDb, RFL_COLUMN_DATA_PACKET, 
		uiPacketLen - RFL_PACKET_OVERHEAD, FALSE)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
RCODE F_Rfl::logInsertRow(
	F_Db *				pDb,
	FLMUINT				uiTableNum,
	F_COLUMN_VALUE *	pColumnValues)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiPacketBodyLen;
	FLMBYTE *	pucPacketBody;
	FLMBYTE *	pucPacketStart;
	
	flmAssert( pDb->m_uiFlags & FDB_HAS_FILE_LOCK);
	
	// Do nothing if logging is disabled.

	if( !isLoggingEnabled())
	{
		goto Exit;
	}

	// Better be in the middle of a transaction.

	flmAssert( m_ui64CurrTransID);

	// Increment the operation count
	
	m_uiOperCount++;
	
	// Make sure we have space in the RFL buffer for a complete packet.  NOTE:
	// this is calculating the maximum packet body length that would be needed.

	if( !haveBuffSpace( FLM_MAX_SEN_LEN + RFL_PACKET_OVERHEAD))
	{
		if( RC_BAD( rc = flush( pDb, m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	// Get a pointer to where we will be laying down the packet body.

	pucPacketBody = pucPacketStart = getPacketBodyPtr();

	// Output the table number

	f_encodeSEN( uiTableNum, &pucPacketBody);

	// Finish the packet - calculate the actual packet body length.

	uiPacketBodyLen = (FLMUINT)(pucPacketBody - pucPacketStart);
	flmAssert( uiPacketBodyLen <= FLM_MAX_SEN_LEN);

	if (RC_BAD( rc = finishPacket( pDb, RFL_INSERT_ROW_PACKET,
		uiPacketBodyLen, FALSE)))
	{
		goto Exit;
	}
	
	// Now output the column data
	
	if (RC_BAD( rc = logColumnValues( pDb, uiTableNum, pColumnValues)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}
		
/********************************************************************
Desc:
*********************************************************************/
RCODE F_Rfl::getColumnValues(
	F_Db *				pDb,
	F_TABLE *			pTable,
	F_COLUMN_VALUE **	ppFirstColValue,
	FLMUINT *			puiNumColumnValues)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiPacketType;
	F_COLUMN_VALUE *	pColumnValue;
	F_COLUMN_VALUE *	pFirstColValue = NULL;
	F_COLUMN_VALUE *	pLastColValue = NULL;
	FLMUINT				uiNumColumnValues = 0;
	const FLMBYTE *	pucPacketBody = NULL;
	const FLMBYTE *	pucEnd = NULL;
	FLMUINT				uiPacketBodyLen;
	FLMUINT				uiIVLen;
	FLMBOOL				bHitEnd;
	FLMUINT				uiColumnNum;
	F_COLUMN *			pColumn;
	F_ENCDEF *			pTableEncDef;
	F_ENCDEF *			pEncDef;
	FLMBYTE *			pucValue;
	FLMUINT				uiBytesLeftToRead;
	FLMUINT				uiBytesToRead;
	FLMUINT				uiEncLen;
	FLMBYTE				ucIV [16];
	
	pTableEncDef = (pTable->lfInfo.uiEncDefNum)
						? pDb->m_pDict->getEncDef( pTable->lfInfo.uiEncDefNum)
						: (F_ENCDEF *)NULL;
	for (;;)
	{

		// If we have used everything in our current packet, get another.

		if (pucPacketBody == pucEnd)
		{
			if( RC_BAD( rc = getPacket( 
				pDb, FALSE, &uiPacketType, &pucPacketBody, &uiPacketBodyLen,
				&bHitEnd)))
			{
				goto Exit;
			}
			
			// Should not hit the end of the RFL here!
			
			if (bHitEnd)
			{
				rc = RC_SET( NE_SFLM_RFL_INCOMPLETE);
				goto Exit;
			}
			
			if (uiPacketType != RFL_COLUMN_DATA_PACKET)
			{
				rc = RC_SET( NE_SFLM_BAD_RFL_PACKET);
				goto Exit;
			}
			pucEnd = pucPacketBody + uiPacketBodyLen;
		}
		
		// Get the column number.
		
		if (RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd, &uiColumnNum)))
		{
			goto Exit;
		}
		
		// A zero column number means we are at the end
		
		if (!uiColumnNum)
		{
			break;
		}
		if ((pColumn = pDb->m_pDict->getColumn( pTable, uiColumnNum)) == NULL)
		{
			rc = RC_SET( NE_SFLM_BAD_RFL_PACKET);
			goto Exit;
		}
		uiNumColumnValues++;
		if (RC_BAD( rc = m_tmpPool.poolAlloc( sizeof( F_COLUMN_VALUE),
									(void **)&pColumnValue)))
		{
			goto Exit;
		}
		pColumnValue->pNext = NULL;
		if (pLastColValue)
		{
			pLastColValue->pNext = pColumnValue;
		}
		else
		{
			pFirstColValue = pColumnValue;
		}
		pLastColValue = pColumnValue;
		pColumnValue->uiColumnNum = uiColumnNum;
		
		// Get the data type
		
		if (pucPacketBody >= pucEnd)
		{
			rc = RC_SET( NE_SFLM_BAD_RFL_PACKET);
			goto Exit;
		}
		
		// Get the value length.
	
		if (RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd,
								&uiBytesLeftToRead)))
		{
			goto Exit;
		}
		
		// Get the value, if there is one.

		if ((pColumnValue->uiValueLen = uiBytesLeftToRead) != 0)
		{
			pEncDef = (pColumn->uiEncDefNum)
						 ? pDb->m_pDict->getEncDef( pColumn->uiEncDefNum)
						 : pTableEncDef;
						 
			// If data is encrypted, allocate enough so that we can
			// decrypt in place - rounding up to nearest encryption boundary
			// should ensure that.
			
			if (!pEncDef)
			{
				if (RC_BAD( rc = m_tmpPool.poolAlloc( uiBytesLeftToRead,
											(void **)&pucValue)))
				{
					goto Exit;
				}
			}
			else
			{
				if (RC_BAD( rc = m_tmpPool.poolAlloc( getEncLen( uiBytesLeftToRead),
											(void **)&pucValue)))
				{
					goto Exit;
				}
			}
			pColumnValue->pucColumnValue = pucValue;
			
			// Now get the value.
			
			while (uiBytesLeftToRead)
			{
				
				// If we have used up our current packet, get another.
				
				if (pucPacketBody == pucEnd)
				{
					if( RC_BAD( rc = getPacket( 
						pDb, FALSE, &uiPacketType, &pucPacketBody, &uiPacketBodyLen,
						&bHitEnd)))
					{
						goto Exit;
					}
					
					// Should not hit the end of the RFL here!
					
					if (bHitEnd)
					{
						rc = RC_SET( NE_SFLM_RFL_INCOMPLETE);
						goto Exit;
					}
					
					if (uiPacketType != RFL_COLUMN_DATA_PACKET)
					{
						rc = RC_SET( NE_SFLM_BAD_RFL_PACKET);
						goto Exit;
					}
					pucEnd = pucPacketBody + uiPacketBodyLen;
				}
				
				// Getting data is simple if it is not encrypted.
				
				if (!pEncDef)
				{
					uiBytesToRead = (FLMUINT)(pucEnd - pucPacketBody);
					if (uiBytesToRead > uiBytesLeftToRead)
					{
						uiBytesToRead = uiBytesLeftToRead;
					}
					f_memcpy( pucValue, pucPacketBody, uiBytesToRead);
					pucPacketBody += uiBytesToRead;
					pucValue += uiBytesToRead;
					uiBytesLeftToRead -= uiBytesToRead;
				}
				else
				{
					
					// Encrypted data should have the encrypted length, the
					// length of data encrypted, an IV, and the encrypted
					// data - all in the same packet.
					
					if (RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd,
											&uiEncLen)))
					{
						goto Exit;
					}
					if (RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd,
											&uiBytesToRead)))
					{
						goto Exit;
					}
					if (uiBytesToRead > uiEncLen || uiBytesToRead > uiBytesLeftToRead)
					{
						rc = RC_SET( NE_SFLM_BAD_RFL_PACKET);
						goto Exit;
					}
					uiIVLen = pEncDef->pCcs->getIVLen();
					flmAssert( uiIVLen == 8 || uiIVLen == 16);
					
					// Make sure packet has the IV and the encrypted data.
					
					if (pucPacketBody + uiIVLen + uiEncLen > pucEnd)
					{
						rc = RC_SET( NE_SFLM_BAD_RFL_PACKET);
						goto Exit;
					}
					
					// Get the IV from the packet.
					
					f_memcpy( ucIV, pucPacketBody, uiIVLen);
					pucPacketBody += uiIVLen;
					
					// Get the encrypted data into our buffer.
					
					f_memcpy( pucValue, pucPacketBody, uiEncLen);
					
					// Decrypt the data in place.
					
					if (RC_BAD( rc = pDb->decryptData( pEncDef->uiEncDefNum, ucIV,
						pucValue, uiEncLen, pucValue, uiEncLen)))
					{
						goto Exit;
					}
					
					// Increment our value buffer only past the decrypted data.
					// Increment the packet body pointer past the encrypted data.
					
					pucValue += uiBytesToRead;
					uiBytesLeftToRead -= uiBytesToRead;
					pucPacketBody += uiEncLen;
				}
			}
		}
	}
Exit:
	*ppFirstColValue = pFirstColValue;
	*puiNumColumnValues = uiNumColumnValues;
	
	return( rc);
}
	
/********************************************************************
Desc:
*********************************************************************/
RCODE F_Rfl::recovInsertRow(
	F_Db *				pDb,
	const FLMBYTE *	pucPacketBody,
	FLMUINT				uiPacketBodyLen,
	eRestoreAction *	peAction)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiTableNum;
	F_TABLE *			pTable;
	F_COLUMN_VALUE *	pFirstColValue;
	FLMUINT				uiNumColumnValues;
	const FLMBYTE *	pucEnd = pucPacketBody + uiPacketBodyLen;
	
	// Get the table number from the packet

	if (RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd, &uiTableNum)))
	{
		goto Exit;
	}
	if ((pTable = pDb->m_pDict->getTable( uiTableNum)) == NULL ||
		 pTable->bSystemTable)
	{
		rc = RC_SET( NE_SFLM_BAD_RFL_PACKET);
		goto Exit;
	}
	
	// Get the column values.
	
	if (RC_BAD( rc = getColumnValues( pDb, pTable, &pFirstColValue,
								&uiNumColumnValues)))
	{
		goto Exit;
	}
	
	if (m_pRestoreStatus)
	{
		if( RC_BAD( rc = m_pRestoreStatus->reportInsertRow(
			peAction, uiTableNum, pFirstColValue, uiNumColumnValues)))
		{
			goto Exit;
		}

		if( *peAction == SFLM_RESTORE_ACTION_STOP)
		{
			m_ui64CurrTransID = 0;
			goto Exit;
		}
	}
	
	if( RC_BAD( rc = pDb->insertRow( uiTableNum, pFirstColValue)))
	{
		goto Exit;
	}

Exit:

	m_tmpPool.poolReset( NULL);

	m_ui64CurrTransID = 0;
	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
RCODE F_Rfl::logUpdateRow(
	F_Db *				pDb,
	FLMUINT				uiTableNum,
	FLMUINT64			ui64RowId,
	F_COLUMN_VALUE *	pColumnValues)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiPacketBodyLen;
	FLMBYTE *	pucPacketBody;
	FLMBYTE *	pucPacketStart;
	
	flmAssert( pDb->m_uiFlags & FDB_HAS_FILE_LOCK);
	
	// Do nothing if logging is disabled.

	if( !isLoggingEnabled())
	{
		goto Exit;
	}

	// Better be in the middle of a transaction.

	flmAssert( m_ui64CurrTransID);

	// Increment the operation count
	
	m_uiOperCount++;
	
	// Make sure we have space in the RFL buffer for a complete packet.  NOTE:
	// this is calculating the maximum packet body length that would be needed.

	if( !haveBuffSpace( FLM_MAX_SEN_LEN * 2 + RFL_PACKET_OVERHEAD))
	{
		if( RC_BAD( rc = flush( pDb, m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	// Get a pointer to where we will be laying down the packet body.

	pucPacketBody = pucPacketStart = getPacketBodyPtr();

	// Output the table number

	f_encodeSEN( uiTableNum, &pucPacketBody);

	// Output the row ID

	f_encodeSEN( ui64RowId, &pucPacketBody);
	
	// Finish the packet - calculate the actual packet body length.

	uiPacketBodyLen = (FLMUINT)(pucPacketBody - pucPacketStart);
	flmAssert( uiPacketBodyLen <= FLM_MAX_SEN_LEN * 2);

	if (RC_BAD( rc = finishPacket( pDb, RFL_UPDATE_ROW_PACKET,
		uiPacketBodyLen, FALSE)))
	{
		goto Exit;
	}
	
	// Now output the column data
	
	if (RC_BAD( rc = logColumnValues( pDb, uiTableNum, pColumnValues)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}
		
/********************************************************************
Desc:
*********************************************************************/
RCODE F_Rfl::recovUpdateRow(
	F_Db *				pDb,
	const FLMBYTE *	pucPacketBody,
	FLMUINT				uiPacketBodyLen,
	eRestoreAction *	peAction)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiTableNum;
	F_TABLE *			pTable;
	FLMUINT64			ui64RowId;
	F_Row *				pRow = NULL;
	F_COLUMN_VALUE *	pFirstColValue;
	FLMUINT				uiNumColumnValues;
	const FLMBYTE *	pucEnd = pucPacketBody + uiPacketBodyLen;
	
	// Get the table number from the packet

	if (RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd, &uiTableNum)))
	{
		goto Exit;
	}
	if ((pTable = pDb->m_pDict->getTable( uiTableNum)) == NULL ||
		 pTable->bSystemTable)
	{
		rc = RC_SET( NE_SFLM_BAD_RFL_PACKET);
		goto Exit;
	}
	
	// Get the row ID from the packet

	if (RC_BAD( rc = f_decodeSEN64( &pucPacketBody, pucEnd, &ui64RowId)))
	{
		goto Exit;
	}
	
	// Get the column values.

	if (RC_BAD( rc = getColumnValues( pDb, pTable, &pFirstColValue,
								&uiNumColumnValues)))
	{
		goto Exit;
	}
	if (m_pRestoreStatus)
	{
		if( RC_BAD( rc = m_pRestoreStatus->reportUpdateRow(
			peAction, uiTableNum, ui64RowId, pFirstColValue, uiNumColumnValues)))
		{
			goto Exit;
		}

		if( *peAction == SFLM_RESTORE_ACTION_STOP)
		{
			m_ui64CurrTransID = 0;
			goto Exit;
		}
	}
	
	if (RC_BAD( rc = gv_SFlmSysData.pRowCacheMgr->retrieveRow( pDb,
							uiTableNum, ui64RowId, &pRow)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pDb->updateRow( uiTableNum, &pRow, pFirstColValue)))
	{
		goto Exit;
	}

Exit:

	m_tmpPool.poolReset( NULL);
	
	if (pRow)
	{
		pRow->ReleaseRow();
	}

	m_ui64CurrTransID = 0;
	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
RCODE F_Rfl::logDeleteRow(
	F_Db *				pDb,
	FLMUINT				uiTableNum,
	FLMUINT64			ui64RowId)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiPacketBodyLen;
	FLMBYTE *	pucPacketBody;
	FLMBYTE *	pucPacketStart;
	
	flmAssert( pDb->m_uiFlags & FDB_HAS_FILE_LOCK);
	
	// Do nothing if logging is disabled.

	if( !isLoggingEnabled())
	{
		goto Exit;
	}

	// Better be in the middle of a transaction.

	flmAssert( m_ui64CurrTransID);

	// Increment the operation count
	
	m_uiOperCount++;
	
	// Make sure we have space in the RFL buffer for a complete packet.  NOTE:
	// this is calculating the maximum packet body length that would be needed.

	if( !haveBuffSpace( FLM_MAX_SEN_LEN * 2 + RFL_PACKET_OVERHEAD))
	{
		if( RC_BAD( rc = flush( pDb, m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	// Get a pointer to where we will be laying down the packet body.

	pucPacketBody = pucPacketStart = getPacketBodyPtr();

	// Output the table number

	f_encodeSEN( uiTableNum, &pucPacketBody);

	// Output the row ID

	f_encodeSEN( ui64RowId, &pucPacketBody);
	
	// Finish the packet - calculate the actual packet body length.

	uiPacketBodyLen = (FLMUINT)(pucPacketBody - pucPacketStart);
	flmAssert( uiPacketBodyLen <= FLM_MAX_SEN_LEN * 2);

	if (RC_BAD( rc = finishPacket( pDb, RFL_DELETE_ROW_PACKET,
		uiPacketBodyLen, FALSE)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}
		
/********************************************************************
Desc:
*********************************************************************/
RCODE F_Rfl::recovDeleteRow(
	F_Db *				pDb,
	const FLMBYTE *	pucPacketBody,
	FLMUINT				uiPacketBodyLen,
	eRestoreAction *	peAction)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiTableNum;
	FLMUINT64			ui64RowId;
	const FLMBYTE *	pucEnd = pucPacketBody + uiPacketBodyLen;
	
	// Get the table number from the packet

	if (RC_BAD( rc = f_decodeSEN( &pucPacketBody, pucEnd, &uiTableNum)))
	{
		goto Exit;
	}
	
	// Get the row ID from the packet

	if (RC_BAD( rc = f_decodeSEN64( &pucPacketBody, pucEnd, &ui64RowId)))
	{
		goto Exit;
	}
	
	if (m_pRestoreStatus)
	{
		if( RC_BAD( rc = m_pRestoreStatus->reportDeleteRow(
			peAction, uiTableNum, ui64RowId)))
		{
			goto Exit;
		}

		if( *peAction == SFLM_RESTORE_ACTION_STOP)
		{
			m_ui64CurrTransID = 0;
			goto Exit;
		}
	}
	
	if( RC_BAD( rc = pDb->deleteRow( uiTableNum, ui64RowId, FALSE)))
	{
		goto Exit;
	}

Exit:

	m_tmpPool.poolReset( NULL);

	m_ui64CurrTransID = 0;
	return( rc);
}

/********************************************************************
Desc:		Make room in the RFL buffer for the additional bytes.
			This is done by flushing the log buffer and shifting down
			the bytes already used in the current packet.  If that
			doesn't make room, the current packet will be finished and
			a new one started.
*********************************************************************/
RCODE F_Rfl::makeRoom(
	F_Db *		pDb,
	FLMUINT		uiAdditionalBytesNeeded,
	FLMUINT *	puiCurrPacketLenRV,
	FLMUINT		uiPacketType,
	FLMUINT *	puiBytesAvailableRV,
	FLMUINT *	puiPacketCountRV)
{
	RCODE		rc = NE_SFLM_OK;
	FLMUINT	uiBytesNeeded;

	uiBytesNeeded = *puiCurrPacketLenRV + uiAdditionalBytesNeeded;
	if( uiBytesNeeded <= (FLMUINT)RFL_MAX_PACKET_SIZE)
	{
		FLMUINT	uiTmp = uiBytesNeeded;

		if( haveBuffSpace( uiTmp))
		{
			if( puiBytesAvailableRV)
			{
				*puiBytesAvailableRV = uiAdditionalBytesNeeded;
			}
		}
		else
		{
			// Bytes requested will fit into a packet, but not the
			// buffer, so we need to shift the packets in the buffer
			// down.  The shiftPacketsDown guarantees that there
			// is room in the buffer for a full size packet.

			if( RC_BAD( rc = shiftPacketsDown( pDb, *puiCurrPacketLenRV, FALSE)))
			{
				goto Exit;
			}

			// If a non-NULL puwBytesAvailableRV is passed in it means that we
			// are to return the number of bytes that we can actually output.
			// Since we know there is enough for the bytes needed, we simply return
			// the number of bytes that were requested.

			if( puiBytesAvailableRV)
			{
				*puiBytesAvailableRV = uiAdditionalBytesNeeded;
			}
		}
	}
	else // (uiBytesNeeded > RFL_MAX_PACKET_SIZE)
	{
		// This is the case where the bytes needed would overflow the
		// maximum packet size.
		// If puwBytesAvailableRV is NULL, it means that all of the
		// requested additional bytes must fit into the packet.  In that
		// case, since the requested bytes would put us over the packet
		// size limit, we must finish the current packet and then
		// flush the packets out of the buffer so we can start a
		// new packet.

		if( !puiBytesAvailableRV)
		{
			// Finish the current packet and start a new one.

			if( puiPacketCountRV)
			{
				(*puiPacketCountRV)++;
			}
			
			if( RC_BAD( rc = finishPacket( pDb, uiPacketType,
						*puiCurrPacketLenRV - RFL_PACKET_OVERHEAD,
						FALSE)))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = flush( pDb, m_pCurrentBuf)))
			{
				goto Exit;
			}
			
			*puiCurrPacketLenRV = RFL_PACKET_OVERHEAD;
		}
		else
		{
			// When puiBytesAvailableRV is non-NULL, it means we can fill up
			// the rest of the packet with part of the bytes.  In this case
			// we return the number of bytes available and then shift the
			// packets down in the buffer to make sure there is room for
			// a full-size packet.

			*puiBytesAvailableRV = RFL_MAX_PACKET_SIZE - *puiCurrPacketLenRV;
			if( RC_BAD( rc = shiftPacketsDown( pDb, *puiCurrPacketLenRV, FALSE)))
			{
				goto Exit;
			}
		}
	}
	
Exit:

	return( rc);
}

/********************************************************************
Desc:		Reads a full packet, based on what file offset and read
			offset are currently set to.
*********************************************************************/
RCODE F_Rfl::readPacket(
	FLMUINT	uiMinBytesNeeded)
{
	RCODE		rc = NE_SFLM_OK;
	FLMUINT	uiTmpOffset;
	FLMUINT	uiReadLen;
	FLMUINT	uiBytesRead;

	// If we have enough bytes in the buffer for the minimum bytes
	// needed, we don't need to retrieve any more bytes.

	if( m_pCurrentBuf->uiRflBufBytes &&
		 m_pCurrentBuf->uiRflBufBytes >= m_uiRflReadOffset &&
		 m_pCurrentBuf->uiRflBufBytes - m_uiRflReadOffset >= uiMinBytesNeeded)
	{
		goto Exit;
	}

	// If we are doing restore, we have to do only sequential
	// reads - cannot depend on doing reads on 512 byte boundaries.
	// Otherwise, we read directly from disk on 512 byte boundaries.

	if( m_pRestore)
	{
		FLMUINT	uiCurrFilePos = m_pCurrentBuf->uiRflFileOffset +
										 m_pCurrentBuf->uiRflBufBytes;

		if( m_uiRflReadOffset > 0 &&
			 m_pCurrentBuf->uiRflBufBytes >= m_uiRflReadOffset)
		{
			// Move the bytes left in the buffer down to the beginning
			// of the buffer.

			f_memmove( m_pCurrentBuf->pIOBuffer->getBufferPtr(),
				&(m_pCurrentBuf->pIOBuffer->getBufferPtr()[ m_uiRflReadOffset]),
						m_pCurrentBuf->uiRflBufBytes - m_uiRflReadOffset);
			m_pCurrentBuf->uiRflBufBytes -= m_uiRflReadOffset;
			m_pCurrentBuf->uiRflFileOffset += m_uiRflReadOffset;
			m_uiRflReadOffset = 0;
		}
		uiReadLen = m_uiBufferSize - m_pCurrentBuf->uiRflBufBytes;

		// Read enough to fill the rest of the buffer, which is
		// guaranteed to hold at least one full packet.

		if( !m_uiFileEOF)
		{
			if( uiCurrFilePos > (FLMUINT)(-1) - uiReadLen)
			{
				uiReadLen = (FLMUINT)(-1) - uiCurrFilePos;
			}
		}
		else
		{
			if( uiCurrFilePos + uiReadLen > m_uiFileEOF)
			{
				uiReadLen = m_uiFileEOF - uiCurrFilePos;
			}
		}

		// If reading will not give us the minimum bytes needed,
		// we cannot satisfy this request from the current file.

		if( uiReadLen + m_pCurrentBuf->uiRflBufBytes < uiMinBytesNeeded)
		{
			rc = RC_SET( NE_SFLM_BAD_RFL_PACKET);
			goto Exit;
		}

		// Read enough to get the entire packet.

		if( RC_BAD( rc = m_pRestore->read( uiReadLen,
			&(m_pCurrentBuf->pIOBuffer->getBufferPtr()[ 
				m_pCurrentBuf->uiRflBufBytes]), &uiBytesRead)))
		{
			if( rc == NE_FLM_IO_END_OF_FILE)
			{
				rc = NE_SFLM_OK;
			}
			else
			{
				goto Exit;
			}
		}

		if( m_pRestoreStatus)
		{
			eRestoreAction		eAction;

			if( RC_BAD( rc = m_pRestoreStatus->reportRflRead( 
				&eAction, m_pCurrentBuf->uiCurrFileNum, uiBytesRead)))
			{
				goto Exit;
			}

			if( eAction == SFLM_RESTORE_ACTION_STOP)
			{
				rc = RC_SET( NE_SFLM_USER_ABORT);
				goto Exit;
			}
		}

		// If we didn't read enough to satisfy the minimum bytes needed,
		// we cannot satisfy this request from the current file.

		if( uiBytesRead + m_pCurrentBuf->uiRflBufBytes < uiMinBytesNeeded)
		{
			rc = RC_SET( NE_SFLM_BAD_RFL_PACKET);
			goto Exit;
		}
		
		m_pCurrentBuf->uiRflBufBytes += uiBytesRead;
	}
	else
	{
		// Set offsets so we are on a 512 byte boundary for our
		// next read.  No need to move data, since we will be
		// re-reading it anyway.

		if( m_uiRflReadOffset > 0)
		{
			uiTmpOffset = m_uiRflReadOffset - (m_uiRflReadOffset & 511);
			m_pCurrentBuf->uiRflFileOffset += uiTmpOffset;
			m_uiRflReadOffset -= uiTmpOffset;
		}
		else if( m_pCurrentBuf->uiRflFileOffset & 511)
		{
			m_uiRflReadOffset = m_pCurrentBuf->uiRflFileOffset & 511;
			m_pCurrentBuf->uiRflFileOffset -= m_uiRflReadOffset;
		}
		
		m_pCurrentBuf->uiRflBufBytes = 0;

		// Read enough to fill the rest of the buffer, which is
		// guaranteed to hold at least one full packet.

		uiReadLen = m_uiBufferSize;

		// m_uiFileEOF better not be zero at this point - we should
		// always know precisely where the RFL file ends when we
		// are doing recovery as opposed to doing a restore.

		flmAssert( m_uiFileEOF >= 512);
		if( m_pCurrentBuf->uiRflFileOffset + uiReadLen > m_uiFileEOF)
		{
			uiReadLen = m_uiFileEOF - m_pCurrentBuf->uiRflFileOffset;
		}

		// If reading will not give us the minimum number of bytes
		// needed, we have a bad packet.

		if( uiReadLen < m_uiRflReadOffset ||
			 uiReadLen - m_uiRflReadOffset < uiMinBytesNeeded)
		{
			rc = RC_SET( NE_SFLM_BAD_RFL_PACKET);
			goto Exit;
		}

		// Read to get the entire packet.

		if( RC_BAD( rc = m_pFileHdl->read( m_pCurrentBuf->uiRflFileOffset,
									uiReadLen, m_pCurrentBuf->pIOBuffer->getBufferPtr(),
									&uiBytesRead)))
		{
			if( rc == NE_FLM_IO_END_OF_FILE)
			{
				rc = NE_SFLM_OK;
			}
			else
			{
				m_bRflVolumeOk = FALSE;
				goto Exit;
			}
		}
		
		if( uiBytesRead < uiReadLen)
		{
			rc = RC_SET( NE_SFLM_BAD_RFL_PACKET);
			goto Exit;
		}

		m_pCurrentBuf->uiRflBufBytes = uiReadLen;
	}
	
Exit:

	return( rc);
}

/********************************************************************
Desc:		Gets and verifies the next packet from the roll-forward
			log file.  Packet checksum will be verified.
*********************************************************************/
RCODE F_Rfl::getPacket(
	F_Db *				pDb,
	FLMBOOL				bForceNextFile,
	FLMUINT *			puiPacketTypeRV,
	const FLMBYTE **	ppucPacketBodyRV,
	FLMUINT *			puiPacketBodyLenRV,
	FLMBOOL *			pbHitEnd)
{
	RCODE					rc = NE_SFLM_OK;
	FLMBYTE *			pucPacket;
	const FLMBYTE *	pucPacketBody;
	const FLMBYTE *	pucPacketBodyEnd;
	FLMUINT				uiOrigPacketBodyLen;
	FLMUINT				uiPacketBodyLen;
	FLMUINT				uiPacketType;
	FLMBYTE				ucHdr [512];
	FLMUINT				uiBytesRead;

	// See if we need to go to the next file.  Note that we only
	// check for this exactly on packet boundaries.  We do not expect
	// packets to be split across files.  If we are not at the end
	// of processing what is in the buffer, we should be able to
	// read the rest of the packet from the current file.

Get_Next_File:

	if( bForceNextFile ||
		 (m_uiFileEOF && m_uiRflReadOffset == m_pCurrentBuf->uiRflBufBytes &&
		  m_pCurrentBuf->uiRflFileOffset + m_pCurrentBuf->uiRflBufBytes ==
						m_uiFileEOF))
	{
		if( m_bKeepRflFiles)
		{
			if( !m_pRestore)
			{
				// Only doing recovery after a failure, see if we are at
				// the last file already.

				if( m_pCurrentBuf->uiCurrFileNum == m_uiLastRecoverFileNum)
				{
					*pbHitEnd = TRUE;
					goto Exit;
				}
				else if( (m_pCurrentBuf->uiCurrFileNum + 1 ) ==
								m_uiLastRecoverFileNum &&
					!m_pDatabase->m_lastCommittedDbHdr.ui32RflLastTransOffset)
				{
					// We are going to try to open the last file.  Since the log
					// header shows a current offset of 0, the file may have been
					// created but nothing was logged to it.  We don't want to try
					// to open it here because it may not have been initialized
					// fully at the time of the server crash.

					m_pCurrentBuf->uiCurrFileNum = m_uiLastRecoverFileNum;
					*pbHitEnd = TRUE;
					goto Exit;
				}

				// Open the next file in the sequence.

				if( RC_BAD( rc = openFile( pDb->m_hWaitSem,
					m_pCurrentBuf->uiCurrFileNum + 1, m_ucNextSerialNum)))
				{
					if( rc == NE_FLM_IO_PATH_NOT_FOUND)
					{
						rc = NE_SFLM_OK;
						*pbHitEnd = TRUE;
					}
					
					goto Exit;
				}

				// If this is the last RFL file, the EOF is contained
				// in the log header.  Otherwise, it will be in the RFL
				// file's header, and openFile will already have retrieved it.

				if( m_pCurrentBuf->uiCurrFileNum == m_uiLastRecoverFileNum)
				{
					m_uiFileEOF =
						m_pDatabase->m_lastCommittedDbHdr.ui32RflLastTransOffset;

					// Could be zero if RFL file wasn't created yet.

					if( !m_uiFileEOF)
					{
						m_uiFileEOF = 512;
					}
				}

				// By this point, the EOF better be greater than or equal to 512.

				flmAssert( m_uiFileEOF >= 512);
			}
			else
			{
				if( RC_BAD( rc = m_pRestore->close()))
				{
					goto Exit;
				}

				// Ask the recovery object to open the file.

				if( RC_BAD( rc = m_pRestore->openRflFile(
												m_pCurrentBuf->uiCurrFileNum + 1)))
				{
					if( rc == NE_FLM_IO_PATH_NOT_FOUND)
					{
						rc = NE_SFLM_OK;
						*pbHitEnd = TRUE;
					}
					
					goto Exit;
				}

				if( m_pRestoreStatus)
				{
					eRestoreAction		eAction;

					if( RC_BAD( rc = m_pRestoreStatus->reportOpenRflFile( 
						&eAction, m_pCurrentBuf->uiCurrFileNum + 1)))
					{
						goto Exit;
					}

					if( eAction == SFLM_RESTORE_ACTION_STOP)
					{
						rc = RC_SET( NE_SFLM_USER_ABORT);
						goto Exit;
					}
				}

				// Get the first 512 bytes from the file and verify the header.

				if( RC_BAD( rc = m_pRestore->read( 512, ucHdr, &uiBytesRead)))
				{
					goto Exit;
				}

				if( m_pRestoreStatus)
				{
					eRestoreAction		eAction;

					if( RC_BAD( rc = m_pRestoreStatus->reportRflRead( 
						&eAction, m_pCurrentBuf->uiCurrFileNum + 1, uiBytesRead)))
					{
						goto Exit;
					}

					if( eAction == SFLM_RESTORE_ACTION_STOP)
					{
						rc = RC_SET( NE_SFLM_USER_ABORT);
						goto Exit;
					}
				}

				if( uiBytesRead < 512)
				{
					rc = RC_SET( NE_SFLM_NOT_RFL);
					goto Exit;
				}
				
				if( RC_BAD( rc = verifyHeader( ucHdr,
					m_pCurrentBuf->uiCurrFileNum + 1, m_ucNextSerialNum)))
				{
					goto Exit;
				}

				// We may not know the actual EOF of files during restore 
				// operations.  m_uiFileEOF could be zero here.

				m_uiFileEOF = (FLMUINT)FB2UD( &ucHdr[ RFL_EOF_POS]);

				// File EOF may be zero or >= 512 at this point.

				flmAssert( !m_uiFileEOF || m_uiFileEOF >= 512);

				// Need to increment current file number.

				m_pCurrentBuf->uiCurrFileNum++;
			}
			
			m_pCurrentBuf->uiRflFileOffset = 512;
			m_uiRflReadOffset = 0;
			m_pCurrentBuf->uiRflBufBytes = 0;

			// Get the next packet from the new file.

			if( RC_BAD( rc = readPacket( RFL_PACKET_OVERHEAD)))
			{
				if( m_uiFileEOF == 512 && m_bKeepRflFiles)
				{
					// File was empty, try to go to the next file.

					bForceNextFile = TRUE;
					goto Get_Next_File;
				}
				goto Exit;
			}
		}
		else
		{
			// This is the case where we are not keeping the RFL files.
			// So, there is no next file to go to.  If we get to this
			// point, we had better not be doing a restore.

			flmAssert( m_pRestore == NULL && !bForceNextFile);
			*pbHitEnd = TRUE;
			goto Exit;
		}
	}

	// Make sure we at least have a packet header in the buffer.

	if (RC_BAD( rc = readPacket( RFL_PACKET_OVERHEAD)))
	{
		goto Exit;
	}

	// Verify the packet address.

	m_uiPacketAddress = m_pCurrentBuf->uiRflFileOffset + m_uiRflReadOffset;
	pucPacket = &(m_pCurrentBuf->pIOBuffer->getBufferPtr()[m_uiRflReadOffset]);
	if ((FLMUINT)FB2UD( &pucPacket [RFL_PACKET_ADDRESS_OFFSET]) !=
				m_uiPacketAddress)
	{
		rc = RC_SET( NE_SFLM_BAD_RFL_PACKET);
		goto Exit;
	}

	// Get packet type, time flag, and packet body length

	*puiPacketTypeRV = uiPacketType = 
			RFL_GET_PACKET_TYPE( pucPacket [RFL_PACKET_TYPE_OFFSET]);

	uiPacketBodyLen = uiOrigPacketBodyLen =
		(FLMUINT)FB2UW( &pucPacket [RFL_PACKET_BODY_LENGTH_OFFSET]);

	// Make sure we have the entire packet in the buffer.

	if (RC_BAD( rc = readPacket( uiPacketBodyLen + RFL_PACKET_OVERHEAD)))
	{
		goto Exit;
	}
	pucPacket = &(m_pCurrentBuf->pIOBuffer->getBufferPtr()[m_uiRflReadOffset]);

	// At this point, we are guaranteed to have the entire packet
	// in the buffer.

	pucPacketBody = &pucPacket [RFL_PACKET_OVERHEAD];
	pucPacketBodyEnd = pucPacketBody + uiPacketBodyLen;

	// Validate the packet checksum

	if (RflCalcChecksum( pucPacket, uiPacketBodyLen) !=
			pucPacket [RFL_PACKET_CHECKSUM_OFFSET])
	{
		rc = RC_SET( NE_SFLM_BAD_RFL_PACKET);
		goto Exit;
	}

	if (uiPacketType == RFL_TRNS_BEGIN_PACKET ||
		uiPacketType == RFL_UPGRADE_PACKET ||
		uiPacketType == RFL_REDUCE_PACKET ||
		uiPacketType == RFL_WRAP_KEY_PACKET ||
		uiPacketType == RFL_ENABLE_ENCRYPTION_PACKET ||
		uiPacketType == RFL_ROLL_OVER_DB_KEY_PACKET)
	{
		// Current transaction ID better be zero, otherwise, we
		// have two or more begin packets in a row.

		if( m_ui64CurrTransID)
		{
			rc = RC_SET( NE_SFLM_BAD_RFL_PACKET);
			goto Exit;
		}

		if( RC_BAD( rc = f_decodeSEN64( &pucPacketBody, 
			pucPacketBodyEnd, &m_ui64CurrTransID)))
		{
			goto Exit;
		}
		
		uiPacketBodyLen = (FLMUINT)(pucPacketBodyEnd - pucPacketBody);

		// Make sure the transaction numbers are ascending

		if( m_ui64CurrTransID <= m_ui64LastTransID)
		{
			rc = RC_SET( NE_SFLM_BAD_RFL_PACKET);
			goto Exit;
		}
	}
	else
	{
		// If transaction ID is not zero, we are not inside a
		// transaction, and it is likely that we have a corrupt
		// packet.

		if( !m_ui64CurrTransID)
		{
			rc = RC_SET( NE_SFLM_BAD_RFL_PACKET);
			goto Exit;
		}

		if( uiPacketType == RFL_TRNS_COMMIT_PACKET ||
			 uiPacketType == RFL_TRNS_ABORT_PACKET)
		{
			FLMUINT64		ui64Tmp;

			if( RC_BAD( rc = f_decodeSEN64( &pucPacketBody, 
				pucPacketBodyEnd, &ui64Tmp)))
			{
				goto Exit;
			}
			
			if( ui64Tmp != m_ui64CurrTransID)
			{
				rc = RC_SET( NE_SFLM_BAD_RFL_PACKET);
				goto Exit;
			}

			uiPacketBodyLen = (FLMUINT)(pucPacketBodyEnd - pucPacketBody);
		}
	}

	// Set read offset to beginning of next packet.

	m_uiRflReadOffset += (RFL_PACKET_OVERHEAD + uiOrigPacketBodyLen);
	*puiPacketBodyLenRV = uiPacketBodyLen;
	*ppucPacketBodyRV = pucPacketBody;
	
Exit:

	return( rc);
}

/********************************************************************
Desc:	Restore transactions from the roll-forward log to the
		database.
*********************************************************************/
RCODE F_Rfl::recover(
	F_Db *					pDb,
	IF_RestoreClient *	pRestore,
	IF_RestoreStatus *	pRestoreStatus)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiStartFileNum;
	FLMUINT				uiStartOffset;
	FLMUINT				uiOffset;
	FLMUINT				uiReadLen;
	FLMUINT				uiBytesRead;
	FLMBYTE				ucHdr[ 512];
	FLMUINT				uiPacketType;
	const FLMBYTE *	pucPacketBody;
	FLMUINT				uiPacketBodyLen = 0;
	eRestoreAction		eAction;
	FLMBOOL				bTransActive = FALSE;
	FLMBOOL				bHadOperations = FALSE;
	FLMBOOL				bLastTransEndedAtFileEOF = FALSE;
	FLMBOOL				bForceNextFile;
	FLMBOOL				bHitEnd;
	FLMUINT				uiRflToken = 0;

	flmAssert( m_pDatabase);

	m_pCurrentBuf = &m_Buf1;
	m_ui64LastLoggedCommitTransID = 0;

	// We need to allow all updates logged in the RFL
	// (including dictionary updates).

	pDb->m_bItemStateUpdOk = TRUE;

	// Turn off logging.

	disableLogging( &uiRflToken);

	// Set the replay flag on the database.

	pDb->m_uiFlags |= FDB_REPLAYING_RFL;

	// Set the flag as to whether or not we are using multiple RFL files.

	m_bKeepRflFiles = m_pDatabase->m_lastCommittedDbHdr.ui8RflKeepFiles
								? TRUE
								: FALSE;

	// If pRestore is NULL, we are doing a database recovery after
	// open.  In that case, we start from the last checkpoint offset
	// and only run until the last transaction offset.

	m_pRestoreStatus = pRestoreStatus;

	if( (m_pRestore = pRestore) == NULL)
	{
		FLMBYTE *	pucCheckSerialNum;
		FLMUINT		uiEndOffset;

		// Clear the restore status pointer.  Assert that the
		// pointer is NULL so we can catch the improper use
		// of this object.

		if( m_pRestoreStatus)
		{
			flmAssert( 0);
			m_pRestoreStatus = NULL;
		}

		uiStartFileNum =
			(FLMUINT)m_pDatabase->m_lastCommittedDbHdr.ui32RflLastCPFileNum;
		m_uiLastRecoverFileNum =
			(FLMUINT)m_pDatabase->m_lastCommittedDbHdr.ui32RflCurrFileNum;
		uiStartOffset =
			(FLMUINT)m_pDatabase->m_lastCommittedDbHdr.ui32RflLastCPOffset;
		uiEndOffset =
			(FLMUINT)m_pDatabase->m_lastCommittedDbHdr.ui32RflLastTransOffset;

		// Could be zero if the file was created, but no transactions were
		// ever committed to it.

		if( !uiEndOffset)
		{
			uiEndOffset = 512;
		}

		// Start offset better not be less than 512.

		flmAssert( uiStartOffset >= 512);
		flmAssert( uiEndOffset >= 512);

		// If start and end are at the same place, there is nothing
		// to recover.

		if( uiStartFileNum == m_uiLastRecoverFileNum &&
			 uiStartOffset == uiEndOffset)
		{
			goto Finish_Recovery;
		}

		// We have not recorded the serial number of the last checkpoint file
		// number, so we pass in NULL, unless it happens to be the same as the
		// last transaction file number, in which case we can pass in the
		// serial number we have stored in the log header.

		pucCheckSerialNum =
			(uiStartFileNum == m_uiLastRecoverFileNum)
			? &m_pDatabase->m_lastCommittedDbHdr.ucLastTransRflSerialNum [0]
			: NULL;
			
		if( RC_BAD( rc = openFile( pDb->m_hWaitSem,
			uiStartFileNum, pucCheckSerialNum)))
		{
			goto Exit;
		}

		// If this is the last RFL file, the EOF is contained
		// in the log header.  Otherwise, it will be in the RFL
		// file's header, and openFile will already have retrieved it.

		if( uiStartFileNum == m_uiLastRecoverFileNum)
		{
			m_uiFileEOF = uiEndOffset;
		}

		// At this point, file EOF better be greater than or equal to 512.

		flmAssert( m_uiFileEOF >= 512);
	}
	else if( !m_bKeepRflFiles)
	{
		// FlmDbRestore should be checking the "keep" flag and not
		// attempting to do a restore of the RFL.

		rc = RC_SET_AND_ASSERT( NE_SFLM_CANNOT_RESTORE_RFL_FILES);
		goto Exit;
	}
	else
	{
		uiStartFileNum =
			(FLMUINT)m_pDatabase->m_lastCommittedDbHdr.ui32RflCurrFileNum;
		uiStartOffset =
			(FLMUINT)m_pDatabase->m_lastCommittedDbHdr.ui32RflLastTransOffset;

		// Could be zero if the RFL file had never been created.

		if( !uiStartOffset)
		{
			uiStartOffset = 512;
		}

		// Ask the recovery object to open the file.

Retry_Open:

		flmAssert( uiStartFileNum);
		if( RC_BAD( rc = m_pRestore->openRflFile( uiStartFileNum)))
		{
			if( rc == NE_FLM_IO_PATH_NOT_FOUND)
			{
				// Need to set m_pCurrentBuf->uiCurrFileNum in case the first
				// call to openRflFile fails.  This will cause the code at the
				// Finish_Recovery label to correctly set up the log
				// header.

				if( !uiStartOffset)
				{
					m_pCurrentBuf->uiCurrFileNum = uiStartFileNum - 1;
				}
				else
				{
					m_pCurrentBuf->uiCurrFileNum = uiStartFileNum;
				}

				rc = NE_SFLM_OK;
				goto Finish_Recovery;
			}
			else
			{
				goto Exit;
			}
		}

		if( m_pRestoreStatus)
		{
			if( RC_BAD( rc = m_pRestoreStatus->reportOpenRflFile( 
				&eAction, uiStartFileNum)))
			{
				goto Exit;
			}

			if( eAction == SFLM_RESTORE_ACTION_STOP)
			{
				rc = RC_SET( NE_SFLM_USER_ABORT);
				goto Exit;
			}
		}

		// Get the first 512 bytes from the file and verify the header.

		if( RC_BAD( rc = m_pRestore->read( 512, ucHdr, &uiBytesRead)))
		{
			goto Exit;
		}

		if( m_pRestoreStatus)
		{
			if( RC_BAD( rc = m_pRestoreStatus->reportRflRead( 
				&eAction, uiStartFileNum, uiBytesRead)))
			{
				goto Exit;
			}

			if( eAction == SFLM_RESTORE_ACTION_STOP)
			{
				rc = RC_SET( NE_SFLM_USER_ABORT);
				goto Exit;
			}
		}

		if( uiBytesRead < 512)
		{
			rc = RC_SET( NE_SFLM_NOT_RFL);
			goto Exit;
		}

		if( RC_BAD( rc = verifyHeader( ucHdr, uiStartFileNum,
				m_pDatabase->m_lastCommittedDbHdr.ucLastTransRflSerialNum)))
		{
			RCODE		tmpRc;

			if( m_pRestoreStatus)
			{
				if( RC_BAD( tmpRc = m_pRestoreStatus->reportError( &eAction, rc)))
				{
					rc = tmpRc;
					goto Exit;
				}

				if( eAction == SFLM_RESTORE_ACTION_RETRY)
				{
					if( RC_BAD( rc = m_pRestore->close()))
					{
						goto Exit;
					}
					
					goto Retry_Open;
				}
			}
			
			goto Exit;
		}

		// We may not know the actual EOF of files during restore operations.

		if( (m_uiFileEOF = (FLMUINT)FB2UD( &ucHdr [RFL_EOF_POS])) == 0)
		{
			bLastTransEndedAtFileEOF = TRUE;
		}
		else
		{
			bLastTransEndedAtFileEOF = (m_uiFileEOF == uiStartOffset)
												? TRUE
												: FALSE;
		}

		// Position to the start offset.  Unfortunately, this means reading
		// through the data and discarding it.

		uiOffset = 512;
		while( uiOffset < uiStartOffset)
		{
			uiReadLen = (uiStartOffset - uiOffset);
			if( uiReadLen > m_uiBufferSize)
			{
				uiReadLen = m_uiBufferSize;
			}
			
			if( RC_BAD( rc = m_pRestore->read( uiReadLen,
					m_pCurrentBuf->pIOBuffer->getBufferPtr(), &uiBytesRead)))
			{
				goto Exit;
			}

			if( m_pRestoreStatus)
			{
				if( RC_BAD( rc = m_pRestoreStatus->reportRflRead( 
					&eAction, uiStartFileNum, uiBytesRead)))
				{
					goto Exit;
				}

				if( eAction == SFLM_RESTORE_ACTION_STOP)
				{
					rc = RC_SET( NE_SFLM_USER_ABORT);
					goto Exit;
				}
			}

			// RFL file is incomplete if we could not read up to the last
			// committed transaction.

			if( uiBytesRead < uiReadLen)
			{
				rc = RC_SET( NE_SFLM_RFL_INCOMPLETE);
				goto Exit;
			}

			uiOffset += uiBytesRead;
		}

		// Need to set current file number

		m_pCurrentBuf->uiCurrFileNum = uiStartFileNum;

		// Better not be any transactions to recover - last database
		// state needs to be a completed checkpoint.

		flmAssert(
			m_pDatabase->m_lastCommittedDbHdr.ui64RflLastCPTransID ==
			m_pDatabase->m_lastCommittedDbHdr.ui64CurrTransID);

		// Use uiStartOffset here instead of ui32RflLastTransOffset,
		// because ui32RflLastTransOffset may be zero, but we in that
		// case we should be comparing to 512, and uiStartOffset will have
		// been adjusted to 512 if that is the case.

		flmAssert(
			(FLMUINT)m_pDatabase->m_lastCommittedDbHdr.ui32RflLastCPOffset ==
			uiStartOffset);

		flmAssert(
			(FLMUINT)m_pDatabase->m_lastCommittedDbHdr.ui32RflLastCPFileNum ==
			uiStartFileNum);
	}

	// Set last transaction ID to the last transaction
	// that was checkpointed - transaction numbers should ascend
	// from here.

	m_ui64LastTransID =
 		m_pDatabase->m_lastCommittedDbHdr.ui64RflLastCPTransID;

	// Set the last committed trans ID

	m_ui64LastLoggedCommitTransID =
		m_pDatabase->m_lastCommittedDbHdr.ui64LastRflCommitID;

	m_pCurrentBuf->uiRflFileOffset = uiStartOffset;
	m_uiRflReadOffset = 0;
	m_pCurrentBuf->uiRflBufBytes = 0;

	// Now, read until we are done.

	bForceNextFile = FALSE;
	bHitEnd = FALSE;
	for (;;)
	{
		flmAssert( pDb->m_uiFlags & FDB_REPLAYING_RFL);

		// Get the next operation from the file.

		rc = getPacket( pDb, bForceNextFile, &uiPacketType, &pucPacketBody,
					&uiPacketBodyLen, &bHitEnd);
		bForceNextFile = FALSE;
		
		if( RC_BAD( rc))
		{
			if( rc == NE_SFLM_BAD_RFL_PACKET)
			{
				// If we don't know the current file size, and we
				// are doing a restore, it is OK to end on a bad
				// packet - we will simply abort the current
				// transaction, if any.  Then, try to go to the
				// next file, because we really don't know where
				// this file ends.

				if( m_pRestore && !m_uiFileEOF)
				{
					if( bTransActive)
					{
						pDb->transAbort();
						bTransActive = FALSE;
					}

					// Set current transaction ID to zero - as if we had encountered
					// an abort packet.

					m_ui64CurrTransID = 0;
					bLastTransEndedAtFileEOF = TRUE;

					// Force to go to the next file

					bForceNextFile = TRUE;
					rc = NE_SFLM_OK;
					continue;
				}
			}
			
			goto Exit;
		}
		if (bHitEnd)
		{
			if (!m_pRestore)
			{
				// If we didn't end exactly where we should have, we have
				// an incomplete log.  The same is true if we are in the
				// middle of a transaction.

				if( m_pCurrentBuf->uiCurrFileNum != m_uiLastRecoverFileNum ||
					 bTransActive)
				{
					rc = RC_SET( NE_SFLM_RFL_INCOMPLETE);
				}
				else
				{
					rc = NE_SFLM_OK;
					goto Finish_Recovery;
				}
			}
			else
			{
				// If we are doing a restore, and we get to the end of the
				// log, it is OK - even if we are in the middle of a
				// transaction - the transaction will simply be aborted.

				rc = NE_SFLM_OK;
				goto Finish_Recovery;
			}
			goto Exit;
		}
		

		// At this point, we know we have a good packet, see what it
		// is and handle it.

		bHadOperations = TRUE;
		eAction = SFLM_RESTORE_ACTION_CONTINUE;

		switch (uiPacketType)
		{
			case RFL_TRNS_BEGIN_PACKET:
			{
				// If we already have a transaction active, we have
				// a problem.

				flmAssert( !bTransActive);
				
				if( uiPacketBodyLen)
				{
					rc = RC_SET_AND_ASSERT( NE_SFLM_BAD_RFL_PACKET);
					goto Exit;
				}
				
				if( RC_BAD( rc = recovTransBegin( pDb, &eAction)))
				{
					goto Exit;
				}
				
				if (eAction == SFLM_RESTORE_ACTION_STOP)
				{
					bLastTransEndedAtFileEOF = FALSE;
					goto Finish_Recovery;
				}
				
				bTransActive = TRUE;
				break;
			}

			case RFL_TRNS_COMMIT_PACKET:
			{
				// Commit the current transaction.

				if (m_pRestoreStatus)
				{
					if (RC_BAD( rc = m_pRestoreStatus->reportCommitTrans(
						&eAction, m_ui64CurrTransID)))
					{
						goto Exit;
					}
					
					if (eAction == SFLM_RESTORE_ACTION_STOP)
					{
						bLastTransEndedAtFileEOF = FALSE;
						goto Finish_Recovery;
					}
				}

				flmAssert( bTransActive);
				pDb->m_uiFlags |= FDB_REPLAYING_COMMIT;
				rc = pDb->transCommit();
				pDb->m_uiFlags &= ~FDB_REPLAYING_COMMIT;
				bTransActive = FALSE;

				if (RC_BAD( rc))
				{
					goto Exit;
				}

				m_ui64LastLoggedCommitTransID = m_ui64CurrTransID;
				
Finish_Transaction:

				if (!m_uiFileEOF)
				{
					bLastTransEndedAtFileEOF = TRUE;
				}
				else
				{
					bLastTransEndedAtFileEOF =
						(m_uiRflReadOffset == m_pCurrentBuf->uiRflBufBytes &&
						 m_pCurrentBuf->uiRflFileOffset +
						 m_pCurrentBuf->uiRflBufBytes == m_uiFileEOF)
						? TRUE
						: FALSE;
				}
				
				m_ui64CurrTransID = 0;
				break;
			}
			
			case RFL_TRNS_ABORT_PACKET:
			{
				// Abort the current transaction.

				if (m_pRestoreStatus)
				{
					if (RC_BAD( rc = m_pRestoreStatus->reportAbortTrans(
						&eAction, m_ui64CurrTransID)))
					{
						goto Exit;
					}
					
					if (eAction == SFLM_RESTORE_ACTION_STOP)
					{
						bLastTransEndedAtFileEOF = FALSE;
						goto Finish_Recovery;
					}
				}

				flmAssert( bTransActive);

				m_uiLastLfNum = 0;
				m_eLastLfType = SFLM_LF_INVALID;

				rc = pDb->transAbort();
				bTransActive = FALSE;

				if (RC_BAD( rc))
				{
					goto Exit;
				}
				
				goto Finish_Transaction;
			}
			
			case RFL_REDUCE_PACKET:
			{	
				if( RC_BAD( rc = recovReduce( pDb, pucPacketBody,
					uiPacketBodyLen, &eAction)))
				{
					goto Exit;
				}
				
				if( eAction == SFLM_RESTORE_ACTION_STOP)
				{
					bLastTransEndedAtFileEOF = FALSE;
					goto Finish_Recovery;
				}
				
				goto Finish_Transaction;
			}
			
			case RFL_UPGRADE_PACKET:
			{
				if( RC_BAD( rc = recovUpgrade( pDb,
					pucPacketBody, uiPacketBodyLen, &eAction)))
				{
					goto Exit;
				}

				if( eAction == SFLM_RESTORE_ACTION_STOP)
				{
					bLastTransEndedAtFileEOF = FALSE;
					goto Finish_Recovery;
				}
				
				goto Finish_Transaction;
			}
			
			case RFL_INDEX_SUSPEND_PACKET:
			case RFL_INDEX_RESUME_PACKET:
			{
				if( RC_BAD( rc = recovIndexSuspendResume( pDb,
					uiPacketType, pucPacketBody, uiPacketBodyLen, &eAction)))
				{
					goto Exit;
				}
				
				if( eAction == SFLM_RESTORE_ACTION_STOP)
				{
					bLastTransEndedAtFileEOF = FALSE;
					goto Finish_Recovery;
				}
				
				break;
			}
			
			case RFL_BLK_CHAIN_FREE_PACKET:
			{
				if( RC_BAD( rc = recovBlockChainFree( pDb,
					pucPacketBody, uiPacketBodyLen, &eAction)))
				{
					goto Exit;
				}
				
				if( eAction == SFLM_RESTORE_ACTION_STOP)
				{
					bLastTransEndedAtFileEOF = FALSE;
					goto Finish_Recovery;
				}
				
				break;
			}

			case RFL_ENABLE_ENCRYPTION_PACKET:
			case RFL_WRAP_KEY_PACKET:
			{
				if( RC_BAD( rc = recovEncryptionKey( pDb, 
					uiPacketType, pucPacketBody, uiPacketBodyLen, &eAction)))
				{
					goto Exit;
				}
				
				if( eAction == SFLM_RESTORE_ACTION_STOP)
				{
					bLastTransEndedAtFileEOF = FALSE;
					goto Finish_Recovery;
				}
				
				goto Finish_Transaction;
			}
			
			case RFL_ENC_DEF_KEY_PACKET:
			{
				if( RC_BAD( rc = recovEncDefKey( pDb, 
					pucPacketBody, uiPacketBodyLen, &eAction)))
				{
					goto Exit;
				}
				
				if( eAction == SFLM_RESTORE_ACTION_STOP)
				{
					bLastTransEndedAtFileEOF = FALSE;
					goto Finish_Recovery;
				}
				
				break;
			}
			
			case RFL_ROLL_OVER_DB_KEY_PACKET:
			{
				if( RC_BAD( rc = recovRollOverDbKey( pDb, 
					pucPacketBody, uiPacketBodyLen, &eAction)))
				{
					goto Exit;
				}
				
				if( eAction == SFLM_RESTORE_ACTION_STOP)
				{
					bLastTransEndedAtFileEOF = FALSE;
					goto Finish_Recovery;
				}
				
				goto Finish_Transaction;
			}

			case RFL_CREATE_TABLE_PACKET:
			{
				if( RC_BAD( rc = recovCreateTable( pDb, 
					pucPacketBody, uiPacketBodyLen, &eAction)))
				{
					goto Exit;
				}
				
				if( eAction == SFLM_RESTORE_ACTION_STOP)
				{
					bLastTransEndedAtFileEOF = FALSE;
					goto Finish_Recovery;
				}
				
				break;
			}
			
			case RFL_DROP_TABLE_PACKET:
			{
				if( RC_BAD( rc = recovDropTable( pDb, 
					pucPacketBody, uiPacketBodyLen, &eAction)))
				{
					goto Exit;
				}
				
				if( eAction == SFLM_RESTORE_ACTION_STOP)
				{
					bLastTransEndedAtFileEOF = FALSE;
					goto Finish_Recovery;
				}
				
				break;
			}
			
			case RFL_CREATE_INDEX_PACKET:
			{
				if( RC_BAD( rc = recovCreateIndex( pDb, 
					pucPacketBody, uiPacketBodyLen, &eAction)))
				{
					goto Exit;
				}
				
				if( eAction == SFLM_RESTORE_ACTION_STOP)
				{
					bLastTransEndedAtFileEOF = FALSE;
					goto Finish_Recovery;
				}
				
				break;
			}
			
			case RFL_DROP_INDEX_PACKET:
			{
				if( RC_BAD( rc = recovDropIndex( pDb, 
					pucPacketBody, uiPacketBodyLen, &eAction)))
				{
					goto Exit;
				}
				
				if( eAction == SFLM_RESTORE_ACTION_STOP)
				{
					bLastTransEndedAtFileEOF = FALSE;
					goto Finish_Recovery;
				}
				
				break;
			}
			
			case RFL_INDEX_SET_PACKET:			
			{
				if( RC_BAD( rc = recovIndexSet( pDb, 
					pucPacketBody, uiPacketBodyLen, &eAction)))
				{
					goto Exit;
				}
				
				if( eAction == SFLM_RESTORE_ACTION_STOP)
				{
					bLastTransEndedAtFileEOF = FALSE;
					goto Finish_Recovery;
				}
				
				break;
			}
			
			case RFL_INSERT_ROW_PACKET:
			{
				if( RC_BAD( rc = recovInsertRow( pDb, 
					pucPacketBody, uiPacketBodyLen, &eAction)))
				{
					goto Exit;
				}
				
				if( eAction == SFLM_RESTORE_ACTION_STOP)
				{
					bLastTransEndedAtFileEOF = FALSE;
					goto Finish_Recovery;
				}
				
				break;
			}
			
			case RFL_UPDATE_ROW_PACKET:
			{
				if( RC_BAD( rc = recovUpdateRow( pDb, 
					pucPacketBody, uiPacketBodyLen, &eAction)))
				{
					goto Exit;
				}
				
				if( eAction == SFLM_RESTORE_ACTION_STOP)
				{
					bLastTransEndedAtFileEOF = FALSE;
					goto Finish_Recovery;
				}
				
				break;
			}
			
			case RFL_DELETE_ROW_PACKET:
			{
				if( RC_BAD( rc = recovDeleteRow( pDb, 
					pucPacketBody, uiPacketBodyLen, &eAction)))
				{
					goto Exit;
				}
				
				if( eAction == SFLM_RESTORE_ACTION_STOP)
				{
					bLastTransEndedAtFileEOF = FALSE;
					goto Finish_Recovery;
				}
				
				break;
			}

			default:
			{
				// Should not be getting other packet types at this
				// point.

				// If we don't know the current file size, and we
				// are doing a restore, it is OK to end on a bad
				// packet - we will simply abort the current
				// transaction, if any.

				if( m_pRestore && !m_uiFileEOF)
				{
					rc = NE_SFLM_OK;
					goto Finish_Recovery;
				}
				else
				{
					rc = RC_SET_AND_ASSERT( NE_SFLM_BAD_RFL_PACKET);
				}
				
				goto Exit;
			}
		}
	}

Finish_Recovery:

	if( bTransActive)
	{
		pDb->transAbort();
		bTransActive = FALSE;
	}

	if( m_pRestore)
	{
		FLMUINT	uiNextRflFileNum = m_pCurrentBuf->uiCurrFileNum + 1;

		// At the end of the restore operation, we need to set things
		// up so that the next transaction will begin a new RFL file.
		// If we ended the restore in the middle of an RFL file, we
		// need to set it up so that the new RFL file will have a new
		// serial number.  If we ended at the end of an RFL file, we
		// can set it up so that the new RFL file will have the next
		// serial number.

		// Set up the next RFL file number and offset.

		m_pDatabase->m_lastCommittedDbHdr.ui32RflCurrFileNum =
			(FLMINT32)uiNextRflFileNum;

		// Set a zero into the offset, this is a special case which tells
		// us that we should create the file no matter what - even if
		// it already exists - it should be overwritten.

		m_pDatabase->m_lastCommittedDbHdr.ui32RflLastTransOffset = 0;

		if( bLastTransEndedAtFileEOF)
		{
			// Move the next serial number of the last RFL file processed into
			// into the current RFL serial number so that the log header
			// will be correct

			f_memcpy(
				m_pDatabase->m_lastCommittedDbHdr.ucLastTransRflSerialNum,
				m_ucNextSerialNum, SFLM_SERIAL_NUM_SIZE);
		}
		else
		{
			// Must create a new serial number so that when the new RFL
			// file is created, it will have that next serial number.

			if( RC_BAD( rc = f_createSerialNumber(
				m_pDatabase->m_lastCommittedDbHdr.ucLastTransRflSerialNum)))
			{
				goto Exit;
			}
		}

		// Save the last logged commit transaction ID.

		if( m_ui64LastLoggedCommitTransID)
		{
			m_pDatabase->m_lastCommittedDbHdr.ui64LastRflCommitID =
				m_ui64LastLoggedCommitTransID;
		}

		// No matter what, we must generate a new next serial number.  This
		// is what will be written to the new RFL file's header when it is
		// created.

		if( RC_BAD( rc = f_createSerialNumber(
						m_pDatabase->m_lastCommittedDbHdr.ucNextRflSerialNum)))
		{
			goto Exit;
		}
	}

	if( !bHadOperations)
	{
		// No transactions were recovered, but still need to
		// setup a few things.

		m_pDatabase->m_uiFirstLogCPBlkAddress = 0;
		m_pDatabase->m_uiLastCheckpointTime = (FLMUINT)FLM_GET_TIMER();

		// Save the state of the log header into the checkpointDbHdr buffer.

		f_memcpy( &m_pDatabase->m_checkpointDbHdr,
				&m_pDatabase->m_lastCommittedDbHdr,
				sizeof( SFLM_DB_HDR));
	}

	// Force a checkpoint to force the log files to be truncated and
	// everything to be reset.  This is done because during recovery
	// the checkpoints that are executed do NOT truncate the RFL file -
	// because it is using the log file to recover!

	closeFile();
	
	m_pRestore = NULL;
	m_pRestoreStatus = NULL;
	enableLogging( &uiRflToken);
	
	pDb->m_uiFlags &= ~FDB_REPLAYING_RFL;
	
	if( RC_BAD( rc = pDb->doCheckpoint( 0)))
	{
		goto Exit;
	}

Exit:

	if( bTransActive)
	{
		pDb->transAbort();
	}
	
	pDb->m_bItemStateUpdOk = FALSE;
	pDb->m_uiFlags &= ~FDB_REPLAYING_RFL;
	
	m_uiLastLfNum = 0;
	return( rc);
}

/****************************************************************************
Desc:	Returns the name of an RFL file given its number
****************************************************************************/
void F_Db::getRflFileName(
	FLMUINT		uiFileNum,
	FLMBOOL		bBaseOnly,
	char *		pszFileName,
	FLMUINT *	puiFileNameBufSize,
	FLMBOOL *	pbNameTruncated)
{
	if (bBaseOnly)
	{
		rflGetBaseFileName( uiFileNum, pszFileName, puiFileNameBufSize,
					pbNameTruncated);
	}
	else
	{
		m_pDatabase->m_pRfl->getFullRflFileName( uiFileNum,
					pszFileName, puiFileNameBufSize,
					pbNameTruncated);
	}
}

