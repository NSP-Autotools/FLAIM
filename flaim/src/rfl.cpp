//-------------------------------------------------------------------------
// Desc:	Routines for roll-forward logging.
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

#define MOD_512(uiNum) \
	(FLMUINT) ((uiNum) & 511)

#define ON_512_BYTE_BOUNDARY(uiNum) \
	(!MOD_512( uiNum))

#define ROUND_DOWN_TO_NEAREST_512(uiNum) \
	(FLMUINT) ((uiNum) & (~((FLMUINT) 511)))

FSTATIC RCODE RflCheckMaxLogged(
	FLMUINT *	puiMaxBytesNeededRV,
	FLMUINT		uiPacketsLogged,
	FLMUINT *	puiCurrTotalLoggedRV,
	FLMUINT		uiBytesToLog);

FSTATIC void RflChangeCallback(
	GRD_DifferenceData &	DiffData,
	void *					CallbackData);

/********************************************************************
Desc:
*********************************************************************/
F_Rfl::F_Rfl()
{
	m_pFile = NULL;
	m_hBufMutex = F_MUTEX_NULL;
	m_pCommitBuf = NULL;
	m_pCurrentBuf = NULL;
	m_uiRflWriteBufs = DEFAULT_RFL_WRITE_BUFFERS;
	m_uiBufferSize = DEFAULT_RFL_BUFFER_SIZE;
	f_memset( &m_Buf1, 0, sizeof( m_Buf1));
	f_memset( &m_Buf2, 0, sizeof( m_Buf2));
	m_bKeepRflFiles = FALSE;
	m_uiRflMinFileSize = DEFAULT_MIN_RFL_FILE_SIZE;
	m_uiRflMaxFileSize = DEFAULT_MAX_RFL_FILE_SIZE;
	m_pFileHdl = NULL;
	m_uiLastRecoverFileNum = 0;
	f_memset( m_ucCurrSerialNum, 0, sizeof( m_ucCurrSerialNum));
	m_bLoggingOff = FALSE;
	m_bLoggingUnknown = FALSE;
	m_uiUnknownPacketLen = 0;
	m_bReadingUnknown = FALSE;
	m_uiUnknownPacketBodyLen = 0;
	m_pucUnknownPacketBody = NULL;
	m_uiUnknownBodyLenProcessed = 0;
	m_uiUnknownPacketRc = FERR_OK;
	m_uiTransStartFile = 0;
	m_uiTransStartAddr = 0;
	m_uiCurrTransID = 0;
	m_uiLastTransID = 0;
	m_uiLastLoggedCommitTransID = 0;
	m_uiOperCount = 0;
	m_uiRflReadOffset = 0;
	m_uiFileEOF = 0;
	m_pRestore = NULL;
	f_memset( m_szDbPrefix, 0, sizeof(m_szDbPrefix));
	f_memset( m_szRflDir, 0, sizeof(m_szRflDir));
	m_bRflDirSameAsDb = FALSE;
	m_bCreateRflDir = FALSE;
	f_memset( m_ucNextSerialNum, 0, sizeof(m_ucNextSerialNum));
	m_bRflVolumeOk = TRUE;
	m_bRflVolumeFull = FALSE;
}

/********************************************************************
Desc:
*********************************************************************/
F_Rfl::~F_Rfl()
{
	flmAssert( !m_bLoggingUnknown);

	if( m_Buf1.pIOBuffer)
	{
		m_Buf1.pIOBuffer->Release();
		m_Buf1.pIOBuffer = NULL;
	}

	if( m_Buf2.pIOBuffer)
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

	if( m_hBufMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hBufMutex);
	}

	if( m_pFileHdl)
	{
		m_pFileHdl->Release();
		m_pFileHdl = NULL;
		m_pFile = NULL;
	}
}

/********************************************************************
Desc:		Returns a boolean indicating whether or not we are at
			the end of the RFL log - will only be TRUE when we are
			doing recovery.
*********************************************************************/
FLMBOOL F_Rfl::atEndOfLog(void)
{
	return( (!m_pRestore && m_uiFileEOF &&
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
	FLMUINT			uiDbVersion,
	const char *	pszDbPrefix,
	FLMUINT			uiFileNum,
	char *			pszBaseNameOut)
{
	FLMINT		iCnt = 0;
	FLMUINT		uiDigit;
	char *		pszTmp = pszBaseNameOut;

	if (uiDbVersion < FLM_FILE_FORMAT_VER_4_3)
	{

		// Output the database name prefix (up to three characters).

		f_strcpy( pszTmp, pszDbPrefix);
		while (*pszTmp)
		{
			pszTmp++;
		}

		// Output as five digit base 36 number.

		pszTmp += 4;
		while (iCnt < 5)
		{
			uiDigit = (FLMUINT) (uiFileNum % 36);
			uiFileNum /= 36;
			if (uiDigit <= 9)
			{
				uiDigit += NATIVE_ZERO;
			}
			else
			{
				uiDigit += (NATIVE_LOWER_A - 10);
			}

			*pszTmp = (FLMBYTE) uiDigit;
			pszTmp--;
			iCnt++;
		}

		// Skip to end of digits and append ".log" to name

		pszTmp += 6;
		f_strcpy( pszTmp, ".log");
	}
	else
	{

		// Output as eight digit hex number.

		pszTmp += 7;
		while (iCnt < 8)
		{
			uiDigit = (FLMUINT) (uiFileNum & 0xF);
			uiFileNum >>= 4;
			if (uiDigit <= 9)
			{
				uiDigit += NATIVE_ZERO;
			}
			else
			{
				uiDigit += (NATIVE_LOWER_A - 10);
			}

			*pszTmp = (FLMBYTE) uiDigit;
			pszTmp--;
			iCnt++;
		}

		// Skip to end of digits and append ".log" to name

		pszTmp += 9;
		f_strcpy( pszTmp, ".log");
	}
}

/********************************************************************
Desc:		Gets the base RFL file name - does not have directory part.
*********************************************************************/
void F_Rfl::getBaseRflFileName(
	FLMUINT	uiFileNum,
	char *	pszBaseName)
{
	rflGetBaseFileName( m_pFile->FileHdr.uiVersionNum, 
		m_szDbPrefix, uiFileNum, pszBaseName);
}

/********************************************************************
Desc:	Generates the full roll forward log file name. Name is based
		on the sequence number and the first three characters of the
		database if the DB is less than version 4.3. 
		Otherwise, it is a hex number.
*********************************************************************/
RCODE F_Rfl::getFullRflFileName(
	FLMUINT	uiFileNum,
	char *	pszRflFileName)
{
	RCODE 	rc = FERR_OK;
	char		szBaseName[F_FILENAME_SIZE];

	// Get the directory name.

	f_strcpy( pszRflFileName, m_szRflDir);

	// Get the base RFL file name.

	getBaseRflFileName( uiFileNum, szBaseName);

	// Append the two together.

	if (RC_BAD( rc = gv_FlmSysData.pFileSystem->pathAppend( 
		pszRflFileName, szBaseName)))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/********************************************************************
Desc:		Positions to the offset specified in the RFL file.
*********************************************************************/
RCODE F_Rfl::positionTo(
	FLMUINT			uiFileOffset)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiBytesRead;

	// Should never be attempting to position to something less than 512 -
	// the header is stored in the first 512 bytes.

	flmAssert( uiFileOffset >= 512);

	// If the position is within our current buffer, see if we can adjust
	// things without having to go back and re-read the buffer from disk.

	if (m_pCurrentBuf->uiRflBufBytes &&
		 uiFileOffset >= m_pCurrentBuf->uiRflFileOffset &&
		 uiFileOffset <= m_pCurrentBuf->uiRflFileOffset +
		 m_pCurrentBuf->uiRflBufBytes)
	{

		// Whatever is in the buffer beyond uiFileOffset is irrelevant and
		// can be discarded.

		m_pCurrentBuf->uiRflBufBytes = uiFileOffset - 
												 m_pCurrentBuf->uiRflFileOffset;
	}
	else
	{

		// Populate the buffer from the 512 byte boundary that is just
		// before the offset we are trying to position to.

		m_pCurrentBuf->uiRflFileOffset = ROUND_DOWN_TO_NEAREST_512( uiFileOffset);
		m_pCurrentBuf->uiRflBufBytes = MOD_512( uiFileOffset);

		if (m_pCurrentBuf->uiRflBufBytes)
		{
			if (RC_BAD( rc = m_pFileHdl->read( 
				m_pCurrentBuf->uiRflFileOffset, m_pCurrentBuf->uiRflBufBytes,
				m_pCurrentBuf->pIOBuffer->getBufferPtr(), &uiBytesRead)))
			{
				if (rc == FERR_IO_END_OF_FILE)
				{
					rc = RC_SET_AND_ASSERT( FERR_NOT_RFL);
				}
				else
				{
					m_bRflVolumeOk = FALSE;
				}

				goto Exit;
			}
			else if (uiBytesRead < m_pCurrentBuf->uiRflBufBytes)
			{
				rc = RC_SET_AND_ASSERT( FERR_NOT_RFL);
				goto Exit;
			}
		}
	}

Exit:

	return (rc);
}

/********************************************************************
Desc:	Get the ACTUAL RFL directory, using as input parameters the
		database version, the name of the database, and the
		user specified RFL directory.  Also return the database
		prefix.
*********************************************************************/
RCODE rflGetDirAndPrefix(
	FLMUINT			uiDbVersionNum,
	const char *	pszDbFileName,
	const char *	pszRflDirIn,
	char *			pszRflDirOut,
	char *			pszDbPrefixOut)
{
	RCODE 		rc = FERR_OK;
	char			szDbPath[ F_PATH_MAX_SIZE];
	char			szBaseName[ F_FILENAME_SIZE];

	// Parse the database name into directory and base name

	if (RC_BAD( rc = gv_FlmSysData.pFileSystem->pathReduce( 
		pszDbFileName, szDbPath, szBaseName)))
	{
		goto Exit;
	}

	// Get the base path

	flmGetDbBasePath( pszDbPrefixOut, szBaseName, NULL);

	if (uiDbVersionNum >= FLM_FILE_FORMAT_VER_4_3)
	{

		// Determine the RFL directory. If one was specified, it is
		// whatever was specified. Otherwise, it is relative to the database
		// directory.

		if (pszRflDirIn && *pszRflDirIn)
		{
			f_strcpy( pszRflDirOut, pszRflDirIn);
		}
		else
		{
			f_strcpy( pszRflDirOut, szDbPath);
		}

		// For 4.3 and above, the RFL files go in a subdirectory underneath
		// the directory where the database is located or the specified
		// directory.

		f_strcpy( szBaseName, pszDbPrefixOut);
		f_strcat( szBaseName, ".rfl");
		gv_FlmSysData.pFileSystem->pathAppend( pszRflDirOut, szBaseName);
	}
	else
	{
		f_strcpy( pszRflDirOut, szDbPath);
	}

Exit:

	return (rc);
}

/********************************************************************
Desc:		Set the RFL directory.  If pszRflDir is NULL or empty string,
			the RFL directory is set to the same directory as the
			database.
*********************************************************************/
RCODE F_Rfl::setRflDir(
	const char *	pszRflDir)
{

	// Better have set up the FFILE pointer.

	flmAssert( m_pFile != NULL);

	m_bRflDirSameAsDb = (!pszRflDir || !(*pszRflDir)) ? TRUE : FALSE;

	flmAssert( m_pFile->FileHdr.uiVersionNum);

	if (m_pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_3)
	{

		// Don't allow RFL directory to be specified for versions less than
		// 4.3

		pszRflDir = NULL;
		m_bRflDirSameAsDb = TRUE;
	}

	m_bCreateRflDir =
		(m_pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_3)
		? TRUE
		: FALSE;
	return( rflGetDirAndPrefix(
					m_pFile->FileHdr.uiVersionNum, m_pFile->pszDbPath,
					pszRflDir, m_szRflDir, m_szDbPrefix));
}

/********************************************************************
Desc:	Gets an RFL file name - based on DB name and RFL directory.
*********************************************************************/
RCODE rflGetFileName(
	FLMUINT			uiDbVersion,
	const char *	pszDbName,
	const char *	pszRflDir,
	FLMUINT			uiFileNum,
	char *			pszRflFileName)
{
	RCODE			rc = FERR_OK;
	char			szDbPrefix[ F_FILENAME_SIZE];
	char			szBaseName[ F_FILENAME_SIZE];

	// Get the full RFL file name.

	if (RC_BAD( rc = rflGetDirAndPrefix( uiDbVersion, pszDbName, pszRflDir,
				  pszRflFileName, szDbPrefix)))
	{
		goto Exit;
	}

	rflGetBaseFileName( uiDbVersion, szDbPrefix, uiFileNum, szBaseName);
	if (RC_BAD( rc = gv_FlmSysData.pFileSystem->pathAppend(
		pszRflFileName, szBaseName)))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/********************************************************************
Desc:	Gets an RFL file number from the RFL file name.
*********************************************************************/
FLMBOOL rflGetFileNum(
	FLMUINT				uiDbVersion,
	const char *		pszDbPrefix,
	const char *		pszRflFileName,
	FLMUINT *			puiFileNum)
{
	FLMBOOL			bGotNum = FALSE;
	char				szDir[ F_PATH_MAX_SIZE];
	char				szBaseName[ F_FILENAME_SIZE];
	char *			pszTmp;
	FLMUINT			uiCharCnt;

	if (RC_BAD( gv_FlmSysData.pFileSystem->pathReduce( 
		pszRflFileName, szDir, szBaseName)))
	{
		goto Exit;
	}

	// See if it has a .log extension.

	pszTmp = &szBaseName[0];
	while (*pszTmp && *pszTmp != '.')
	{
		pszTmp++;
	}

	// If we do not have a .log extension, it is not a legitimate RFL file.

	if (f_stricmp( pszTmp, ".log") != 0)
	{
		goto Exit;
	}

	// Parse out the name according to the rules for this DB version.

	*pszTmp = 0;			// Set period to zero
	pszTmp = &szBaseName[0];
	*puiFileNum = 0;
	uiCharCnt = 0;
	if (uiDbVersion >= FLM_FILE_FORMAT_VER_4_3)
	{

		// Name up to the period should be a hex number

		while (*pszTmp)
		{
			(*puiFileNum) <<= 4;
			if (*pszTmp >= NATIVE_ZERO && *pszTmp <= NATIVE_NINE)
			{
				*puiFileNum += (FLMUINT) (*pszTmp - NATIVE_ZERO);
			}
			else if (*pszTmp >= NATIVE_LOWER_A && *pszTmp <= NATIVE_LOWER_F)
			{
				*puiFileNum += ((FLMUINT) (*pszTmp - NATIVE_LOWER_A) + 10);
			}
			else if (*pszTmp >= NATIVE_UPPER_A && *pszTmp <= NATIVE_UPPER_F)
			{
				*puiFileNum += ((FLMUINT) (*pszTmp - NATIVE_UPPER_A) + 10);
			}
			else
			{
				goto Exit;	// Not a hex number
			}

			uiCharCnt++;
			pszTmp++;
		}

		// Better have been exactly 8 hex digits.

		bGotNum = (FLMBOOL) ((uiCharCnt == 8) ? TRUE : FALSE);
	}
	else
	{
		FLMUINT	uiLen = f_strlen( pszTmp);
		FLMUINT	uiPrefixLen = f_strlen( pszDbPrefix);

		// Length of base name without the .log extension better be exactly
		// 5 more characters than the length of the prefix.

		if (uiLen != uiPrefixLen + 5)
		{
			flmAssert( 0);
			goto Exit;
		}

		// Prefix better match.

		while (uiPrefixLen)
		{
			if (f_toupper( *pszTmp) != f_toupper( *pszDbPrefix))
			{
				goto Exit;
			}

			uiPrefixLen--;
			pszTmp++;
			pszDbPrefix++;
		}

		// Rest of the name is the five digits that are a base 36 number.

		while (*pszTmp)
		{
			(*puiFileNum) *= 36;
			if (*pszTmp >= NATIVE_ZERO && *pszTmp <= NATIVE_NINE)
			{
				*puiFileNum += (FLMUINT) (*pszTmp - NATIVE_ZERO);
			}
			else if (*pszTmp >= NATIVE_LOWER_A && *pszTmp <= NATIVE_LOWER_Z)
			{
				*puiFileNum += ((FLMUINT) (*pszTmp - NATIVE_LOWER_A) + 10);
			}
			else if (*pszTmp >= NATIVE_UPPER_A && *pszTmp <= NATIVE_UPPER_Z)
			{
				*puiFileNum += ((FLMUINT) (*pszTmp - NATIVE_UPPER_A) + 10);
			}
			else
			{
				goto Exit;	// Not a base 36 number
			}

			pszTmp++;
		}

		bGotNum = TRUE;
	}

Exit:

	return (bGotNum);
}

/********************************************************************
Desc:		Sets up the RFL object - associating with a file, etc.
*********************************************************************/
RCODE F_Rfl::setup(
	FFILE *				pFile,
	const char *		pszRflDir)
{
	RCODE		rc = FERR_OK;

	// Better not already be associated with an FFILE

	flmAssert( m_pFile == NULL);
	m_pFile = pFile;

	// Allocate memory for the RFL buffers

	if( !gv_FlmSysData.pFileSystem->canDoAsync())
	{
		m_uiRflWriteBufs = 1;
		m_uiBufferSize = DEFAULT_RFL_WRITE_BUFFERS * DEFAULT_RFL_BUFFER_SIZE;
	}

	if( RC_BAD( rc = f_mutexCreate( &m_hBufMutex)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = FlmAllocIOBufferMgr( m_uiRflWriteBufs, 
		m_uiRflWriteBufs * m_uiBufferSize, TRUE, &m_Buf1.pBufferMgr)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = m_Buf1.pBufferMgr->getBuffer( 
		m_uiBufferSize, &m_Buf1.pIOBuffer)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FlmAllocIOBufferMgr( m_uiRflWriteBufs, 
		m_uiRflWriteBufs * m_uiBufferSize, TRUE, &m_Buf2.pBufferMgr)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_Buf2.pBufferMgr->getBuffer( 
		m_uiBufferSize, &m_Buf2.pIOBuffer)))
	{
		goto Exit;
	}

	m_bLoggingOff = FALSE;
	m_pCurrentBuf = &m_Buf1;
	m_pCurrentBuf->uiRflBufBytes = 0;

	// Set the RFL directory and prefix if necessary.

	if (RC_BAD( rc = setRflDir( pszRflDir)))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/********************************************************************
Desc:	Wait for the writes of a buffer to finish.  This routine assumes
		that the m_hBufMutex is locked when coming in.  It will ALWAYS
		unlock the mutex before exiting.
*********************************************************************/
RCODE F_Rfl::waitForWrites(
	RFL_BUFFER *	pBuffer,
	FLMBOOL			bIsWriter)
{
	RCODE				rc = FERR_OK;
	RCODE				TempRc;
	RFL_WAITER		Waiter;
	FLMBOOL			bMutexLocked = TRUE;

	// Put self on the wait queue for the buffer.

	Waiter.uiThreadId = f_threadId();
	Waiter.bIsWriter = bIsWriter;
	Waiter.hESem = F_SEM_NULL;
	
	if( RC_BAD( rc = f_semCreate( &Waiter.hESem)))
	{
		goto Exit;
	}

	// Note: rc better be changed to success or write error by the process
	// that signals us.

	rc = RC_SET( FERR_FAILURE);
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

	if( RC_BAD( TempRc = f_semWait( Waiter.hESem, F_WAITFOREVER)))
	{
		flmAssert( 0);
		rc = TempRc;
	}
	else
	{
		// Process that signaled us better set the rc to something besides
		// FERR_FAILURE.
		
		flmAssert( rc != FERR_FAILURE);
	}

Exit:

	if( Waiter.hESem != F_SEM_NULL)
	{
		f_semDestroy( &Waiter.hESem);
	}

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hBufMutex);
	}

	return (rc);
}

/********************************************************************
Desc:	If a commit is in progress, wait for it to finish.
*********************************************************************/
RCODE F_Rfl::waitForCommit( void)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bMutexLocked = FALSE;

	// NOTE: If m_pCommitBuf is NULL it cannot be set to something
	// non-NULL except by this thread when this thread ends the
	// transaction. So, there is no need to lock the mutex and re-check if
	// it is NULL.

	if (m_pCommitBuf)
	{
		f_mutexLock( m_hBufMutex);
		bMutexLocked = TRUE;

		// Check m_pCommitBuf again after locking mutex - may have finished.

		if (m_pCommitBuf)
		{

			// waitForWrites will unlock the mutex.

			bMutexLocked = FALSE;
			rc = waitForWrites( m_pCommitBuf, FALSE);
		}
	}

	if (bMutexLocked)
	{
		f_mutexUnlock( m_hBufMutex);
	}

	return (rc);
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
	RCODE			rc = FERR_OK;
	FLMBYTE *	pucBuffer = NULL;
	FLMUINT		uiBytesWritten;

	flmAssert( m_pFile);
	flmAssert( m_pFileHdl);

	if( RC_BAD( rc = f_allocAlignedBuffer( 512, &pucBuffer)))
	{
		goto Exit;
	}

	f_memset( pucBuffer, 0, 512);
	f_memcpy( &pucBuffer[ RFL_NAME_POS], RFL_NAME, RFL_NAME_LEN);
	f_memcpy( &pucBuffer[ RFL_VERSION_POS], RFL_VERSION, RFL_VERSION_LEN);

	UD2FBA( (FLMUINT32)uiFileNum, &pucBuffer[ RFL_FILE_NUMBER_POS]);
	UD2FBA( (FLMUINT32)uiEof, &pucBuffer[ RFL_EOF_POS]);

	if (m_pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_3)
	{
		f_memcpy( &pucBuffer[ RFL_DB_SERIAL_NUM_POS],
					&m_pFile->ucLastCommittedLogHdr[LOG_DB_SERIAL_NUM],
					F_SERIAL_NUM_SIZE);
		f_memcpy( &pucBuffer[ RFL_SERIAL_NUM_POS], pucSerialNum, F_SERIAL_NUM_SIZE);
		f_memcpy( &pucBuffer[ RFL_NEXT_FILE_SERIAL_NUM_POS], pucNextSerialNum,
					F_SERIAL_NUM_SIZE);
		f_strcpy( (char *) &pucBuffer[ RFL_KEEP_SIGNATURE_POS],
					((bKeepSignature) ? RFL_KEEP_SIGNATURE : RFL_NOKEEP_SIGNATURE));
	}

	// Write out the header

	if( RC_BAD( rc = m_pFileHdl->write( 0L, 512, pucBuffer, &uiBytesWritten)))
	{
		// Remap disk full error

		if( rc == FERR_IO_DISK_FULL)
		{
			rc = RC_SET( FERR_RFL_DEVICE_FULL);
			m_bRflVolumeFull = TRUE;
		}

		m_bRflVolumeOk = FALSE;
		goto Exit;
	}

	// Flush the file handle to ensure it is forced to disk.

	if( RC_BAD( rc = m_pFileHdl->flush()))
	{

		// Remap disk full error

		if (rc == FERR_IO_DISK_FULL)
		{
			rc = RC_SET( FERR_RFL_DEVICE_FULL);
			m_bRflVolumeFull = TRUE;
		}

		m_bRflVolumeOk = FALSE;
		goto Exit;
	}

Exit:

	if( pucBuffer)
	{
		f_freeAlignedBuffer( &pucBuffer);
	}

	return (rc);
}

/********************************************************************
Desc:		Verifies the header of an RFL file.
*********************************************************************/
RCODE F_Rfl::verifyHeader(
	FLMBYTE *	pucHeader,
	FLMUINT		uiFileNum,
	FLMBYTE *	pucSerialNum)
{
	RCODE rc = FERR_OK;

	flmAssert( m_pFile);

	// Check the RFL name and version number

	if (f_memcmp( &pucHeader[RFL_NAME_POS], RFL_NAME, RFL_NAME_LEN) != 0)
	{
		rc = RC_SET_AND_ASSERT( FERR_NOT_RFL);
		goto Exit;
	}

	if (f_memcmp( &pucHeader[RFL_VERSION_POS], 
					  RFL_VERSION, RFL_VERSION_LEN) != 0)
	{
		rc = RC_SET_AND_ASSERT( FERR_NOT_RFL);
		goto Exit;
	}

	if (m_pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_3)
	{

		// Verify the database serial number

		if (f_memcmp( &pucHeader[RFL_DB_SERIAL_NUM_POS],
						 &m_pFile->ucLastCommittedLogHdr[LOG_DB_SERIAL_NUM],
						 F_SERIAL_NUM_SIZE) != 0)
		{
			rc = RC_SET( FERR_BAD_RFL_DB_SERIAL_NUM);
			goto Exit;
		}

		// Verify the serial number that is expected to be on the RFL file.
		// If pucSerialNum is NULL, we will not verify it. This is generally
		// only done during recovery or restore when we are reading through
		// multiple RFL files and we need to verify their serial numbers.

		if (pucSerialNum &&
			 f_memcmp( &pucHeader[RFL_SERIAL_NUM_POS], pucSerialNum,
						 F_SERIAL_NUM_SIZE) != 0)
		{
			rc = RC_SET( FERR_BAD_RFL_SERIAL_NUM);
			goto Exit;
		}

		// Verify the file number.

		if (uiFileNum != (FLMUINT) FB2UD( &pucHeader[RFL_FILE_NUMBER_POS]))
		{
			rc = RC_SET( FERR_BAD_RFL_FILE_NUMBER);
			goto Exit;
		}

		// Save serial numbers from the header.

		f_memcpy( m_ucCurrSerialNum, &pucHeader[RFL_SERIAL_NUM_POS],
					F_SERIAL_NUM_SIZE);
		f_memcpy( m_ucNextSerialNum, &pucHeader[RFL_NEXT_FILE_SERIAL_NUM_POS],
					F_SERIAL_NUM_SIZE);
	}

	// Save some things from the header.

	m_uiFileEOF = (FLMUINT) FB2UD( &pucHeader[RFL_EOF_POS]);

Exit:

	return (rc);
}

/********************************************************************
Desc:		Opens an RFL file.  Verifies the serial number for 4.3 dbs.
*********************************************************************/
RCODE F_Rfl::openFile(
	FLMUINT		uiFileNum,
	FLMBYTE *	pucSerialNum)
{
	RCODE				rc = FERR_OK;
	char				szRflFileName[ F_PATH_MAX_SIZE];
	FLMBYTE *		pucBuffer = NULL;
	FLMUINT			uiBytesRead;

	flmAssert( m_pFile);

	if( RC_BAD( rc = f_allocAlignedBuffer( 512, &pucBuffer)))
	{
		goto Exit;
	}

	// If we have a file open and it is not the file number passed in,
	// close it.

	if( m_pFileHdl)
	{
		if (m_pCurrentBuf->uiCurrFileNum != uiFileNum)
		{
			if (RC_BAD( rc = waitForCommit()))
			{
				goto Exit;
			}

			closeFile();
		}
		else
		{
			goto Exit;	// Will return FERR_OK
		}
	}
	else
	{

		// Should not be able to be in the middle of a commit if we don't
		// have a file open!

		flmAssert( !m_pCommitBuf);
	}

	// Generate the log file name.

	if (RC_BAD( rc = getFullRflFileName( uiFileNum, szRflFileName)))
	{
		goto Exit;
	}

	// Open the file.
	
	f_assert( !m_pFileHdl);

	if (RC_BAD( rc = gv_FlmSysData.pFileSystem->openFile( szRflFileName,
			gv_FlmSysData.uiFileOpenFlags, &m_pFileHdl)))
	{
		goto Exit;
	}

	m_pFileHdl->setMaxAutoExtendSize( m_uiRflMaxFileSize);
	m_pFileHdl->setExtendSize( m_pFile->uiFileExtendSize);
	
	// Read the header.

	if (RC_BAD( rc = m_pFileHdl->read( 0, 512, pucBuffer, &uiBytesRead)))
	{
		if (rc == FERR_IO_END_OF_FILE)
		{
			rc = RC_SET_AND_ASSERT( FERR_NOT_RFL);
		}
		else
		{
			m_bRflVolumeOk = FALSE;
		}

		goto Exit;
	}

	// If there is not enough data in the buffer, it is not an RFL file.

	if (uiBytesRead < 512)
	{
		rc = RC_SET_AND_ASSERT( FERR_NOT_RFL);
		goto Exit;
	}

	// Verify the header information

	if (RC_BAD( rc = verifyHeader( pucBuffer, uiFileNum, pucSerialNum)))
	{
		goto Exit;
	}

	m_pCurrentBuf->uiRflBufBytes = 0;
	m_pCurrentBuf->uiRflFileOffset = 0;
	m_pCurrentBuf->uiCurrFileNum = uiFileNum;
	
Exit:

	if( RC_BAD( rc))
	{
		waitForCommit();
		closeFile();
	}

	if( pucBuffer)
	{
		f_freeAlignedBuffer( &pucBuffer);
	}

	return (rc);
}

/********************************************************************
Desc:		Creates a new roll forward log file.
*********************************************************************/
RCODE F_Rfl::createFile(
	FLMUINT		uiFileNum,
	FLMBYTE *	pucSerialNum,
	FLMBYTE *	pucNextSerialNum,
	FLMBOOL		bKeepSignature)
{
	RCODE			rc = FERR_OK;
	char			szRflFileName[ F_PATH_MAX_SIZE];

	flmAssert( m_pFile);

	// Better not be trying to create the current file

	flmAssert( uiFileNum != m_pCurrentBuf->uiCurrFileNum);

	// If we have a file open close it.

	if (RC_BAD( rc = waitForCommit()))
	{
		goto Exit;
	}

	closeFile();

	// Generate the log file name.

	if (RC_BAD( rc = getFullRflFileName( uiFileNum, szRflFileName)))
	{
		goto Exit;
	}

	// If DB is 4.3 or greater and we are in the same directory as our
	// database files, see if we need to create the subdirectory.
	// Otherwise, the RFL directory should already have been created. If
	// the directory already exists, it is OK - we only try this the first
	// time after setRflDir is called - to either verify that the directory
	// exists, or if it doesn't, to create it.

	if (m_bCreateRflDir)
	{
		// If it already exists, don't attempt to create it.

		if (RC_BAD( rc = gv_FlmSysData.pFileSystem->doesFileExist( m_szRflDir)))
		{
			if (rc != FERR_IO_PATH_NOT_FOUND && rc != FERR_IO_INVALID_PATH)
			{
				goto Exit;
			}
			else
			{
				if (RC_BAD( rc = gv_FlmSysData.pFileSystem->createDir( m_szRflDir)))
				{
					goto Exit;
				}
			}
		}

		m_bCreateRflDir = FALSE;
	}

	// Create the file
	
	f_assert( !m_pFileHdl);

	if( RC_BAD( rc = gv_FlmSysData.pFileSystem->openFile( szRflFileName,
		gv_FlmSysData.uiFileOpenFlags, &m_pFileHdl)))
	{
		if( rc != FERR_IO_PATH_NOT_FOUND)
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = gv_FlmSysData.pFileSystem->createFile( szRflFileName,
			gv_FlmSysData.uiFileCreateFlags, &m_pFileHdl)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = m_pFileHdl->truncateFile( m_pFile->uiFileExtendSize)))
		{
			goto Exit;
		}
	}

	m_pFileHdl->setMaxAutoExtendSize( m_uiRflMaxFileSize);
	m_pFileHdl->setExtendSize( m_pFile->uiFileExtendSize);

	// Initialize the header.

	if (RC_BAD( rc = writeHeader( uiFileNum, 0, pucSerialNum, pucNextSerialNum,
				  bKeepSignature)))
	{
		goto Exit;
	}

	m_pCurrentBuf->uiRflBufBytes = 0;
	m_pCurrentBuf->uiRflFileOffset = 512;
	m_pCurrentBuf->uiCurrFileNum = uiFileNum;
	
	// Update the size of the RFL

	if( m_bKeepRflFiles)
	{
		FLMUINT64		ui64RflDiskUsage;
		
		if( RC_BAD( rc = flmRflCalcDiskUsage( m_szRflDir, m_szDbPrefix,
			m_pFile->FileHdr.uiVersionNum, &ui64RflDiskUsage)))
		{
			goto Exit;
		}
		
		f_mutexLock( gv_FlmSysData.hShareMutex);
		m_pFile->ui64RflDiskUsage = ui64RflDiskUsage;
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}

	
Exit:

	// Close the RFL log file AND delete it if we were not successful.

	if (RC_BAD( rc))
	{
		closeFile();
		(void) gv_FlmSysData.pFileSystem->deleteFile( szRflFileName);
	}

	return (rc);
}

/********************************************************************
Desc:		Copy last partial block of last buffer written (or to be
			written) into a new buffer.
*********************************************************************/
void F_Rfl::copyLastBlock(
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

		// See if the number of bytes in the buffer is an exact multiple of
		// 512.

		if (pBuffer->uiRflBufBytes & 511)	// Not exact multiple
		{

			// Round m_uiRflBufBytes down to next 512 byte boundary

			FLMUINT	ui512Offset = ROUND_DOWN_TO_NEAREST_512( 
												pBuffer->uiRflBufBytes);

			// Move all bytes above the nearest 512 byte boundary down to
			// the beginning of the buffer and adjust pBuffer->uiRflBufBytes
			// and pBuffer->uiRflFileOffset accordingly.

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
		f_memmove( &pucNewBuffer[pBuffer->uiRflBufBytes],
					 &pucOldBuffer[uiOldBufBytes], uiCurrPacketLen);
	}
}

/********************************************************************
Desc:		Flush the RFL data from the buffer to disk.
*********************************************************************/
RCODE F_Rfl::flush(
	RFL_BUFFER *	pBuffer,
	FLMBOOL			bFinalWrite,
	FLMUINT			uiCurrPacketLen,
	FLMBOOL			bStartingNewFile)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiBytesWritten;
	IF_IOBuffer *	pNewBuffer = NULL;
	IF_IOBuffer *	pIOBuffer = NULL;
	FLMBYTE *		pucOldBuffer;
	FLMUINT			uiFileOffset;
	FLMUINT			uiBufBytes;

	if (m_pFileHdl && pBuffer->uiRflBufBytes)
	{

		// Must wait for stuff in committing buffer, if any, before going
		// ahead here.

		if( pBuffer != m_pCommitBuf)
		{
			if (RC_BAD( rc = waitForCommit()))
			{
				goto Exit;
			}
		}

		if( m_uiRflWriteBufs > 1 && m_pFileHdl->canDoAsync())
		{
			pIOBuffer = pBuffer->pIOBuffer;
		}

		if ((FLMUINT) (-1) - pBuffer->uiRflFileOffset <= pBuffer->uiRflBufBytes)
		{
			rc = RC_SET( FERR_DB_FULL);
			goto Exit;
		}

		pucOldBuffer = pBuffer->pIOBuffer->getBufferPtr();
		uiFileOffset = pBuffer->uiRflFileOffset;
		uiBufBytes = pBuffer->uiRflBufBytes;
		
		if (m_uiRflWriteBufs > 1)
		{
			if (RC_BAD( rc = pBuffer->pBufferMgr->getBuffer( 
				m_uiBufferSize, &pNewBuffer)))
			{
				goto Exit;
			}

			// No need to copy data if it is the final write, because it
			// won't be reused anyway - the data for the next transaction has
			// already been copied to another buffer.

			if (!bFinalWrite)
			{
				copyLastBlock( pBuffer, pucOldBuffer, pNewBuffer->getBufferPtr(),
									uiCurrPacketLen, bStartingNewFile);
			}
		}
		
		if( pIOBuffer)
		{
			FLMUINT		uiBytesToWrite =	(FLMUINT)f_roundUp( uiBufBytes, 
														m_pFileHdl->getSectorSize());

			rc = m_pFileHdl->write( uiFileOffset, uiBytesToWrite, pIOBuffer);
		}
		else
		{
			pBuffer->pIOBuffer->setPending();

			rc = m_pFileHdl->write( uiFileOffset, uiBufBytes, 
											  pucOldBuffer, &uiBytesWritten);
		}

		if( RC_OK( rc))
		{
			if( m_bKeepRflFiles)
			{
				f_mutexLock( gv_FlmSysData.hShareMutex);

				if( m_pFile->uiFileExtendSize)
				{
					FLMUINT		uiTmpSize;

					uiTmpSize = (uiFileOffset % m_pFile->uiFileExtendSize) + 
										(FLMUINT)f_roundUp( uiBufBytes, 
												m_pFileHdl->getSectorSize());

					if( uiTmpSize > m_pFile->uiFileExtendSize)
					{
						m_pFile->ui64RflDiskUsage += m_pFile->uiFileExtendSize;
					}
				}
				else
				{
					m_pFile->ui64RflDiskUsage += uiBytesWritten;
				}

				f_mutexUnlock( gv_FlmSysData.hShareMutex);
			}
		}
		
		if (m_uiRflWriteBufs == 1)
		{
			// We are counting on the fact that the write completed. When we
			// only have one buffer, we cannot do async writes.

			flmAssert( !pIOBuffer);
			pBuffer->pIOBuffer->notifyComplete( rc);											  
			
			if( RC_OK( rc) && !bFinalWrite)
			{
				copyLastBlock( pBuffer, pucOldBuffer, pucOldBuffer,
									uiCurrPacketLen, bStartingNewFile);
			}
		}
		else
		{
			// No need to call copyLastBlock, because it was called above
			// before calling write. The part of the old buffer that
			// needs to be transferred to the new buffer has already been
			// transferred.

			if( !pIOBuffer)
			{
				pBuffer->pIOBuffer->notifyComplete( rc);
			}

			pBuffer->pIOBuffer->Release();
			pBuffer->pIOBuffer = pNewBuffer;
		}

		if( RC_BAD( rc))
		{
			if (rc == FERR_IO_DISK_FULL)
			{
				rc = RC_SET( FERR_RFL_DEVICE_FULL);
				m_bRflVolumeFull = TRUE;
			}

			m_bRflVolumeOk = FALSE;
			goto Exit;
		}
	}

Exit:

	return (rc);
}

/********************************************************************
Desc:	Switch buffers.  This routine assumes the m_hBufMutex is locked.
*********************************************************************/
void F_Rfl::switchBuffers(void)
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
		copyLastBlock( m_pCurrentBuf, pOldBuffer->pIOBuffer->getBufferPtr(),
							m_pCurrentBuf->pIOBuffer->getBufferPtr(), 0, FALSE);
	}
}

/********************************************************************
Desc: Wait for all RFL transaction writes to be finished.  The caller
		has the write lock on the database, which will prevent further
		writes to the RFL.
*********************************************************************/
FLMBOOL F_Rfl::seeIfRflWritesDone(
	FLMBOOL	bForceWait)
{
	FLMBOOL	bWritesDone;

	f_mutexLock( m_hBufMutex);

	if (!bForceWait)
	{
		bWritesDone = (FLMBOOL) ((m_pCurrentBuf->pFirstWaiter || m_pCommitBuf) 
													? FALSE 
													: TRUE);
		f_mutexUnlock( m_hBufMutex);
	}
	else
	{

		// If the current buffer has a waiter, add self to that list to
		// wait, because it will be notified after the commit buffer has
		// been notified. Otherwise, if there is a commit in progress, add
		// self to that list to wait.

		if (m_pCurrentBuf->pFirstWaiter)
		{

			// If bTransInProgress is TRUE and m_pCommitBuf is NULL then
			// this thread is the current transaction, and nobody is going to
			// wake up the first waiter until we are done! Hence, we must
			// wake him up.

			if (!m_pCommitBuf)
			{

				// If m_pCommitBuf is NULL, this could only be possible if
				// there is a transaction in progress. Otherwise, there would
				// not have been a pFirstWaiter, because when the commit
				// buffer finishes writing, if there is a waiter, it will set
				// commitbuf=currentbuf if there is no transaction active.

				flmAssert( m_pCurrentBuf->bTransInProgress);

				m_pCommitBuf = m_pCurrentBuf;
				switchBuffers();
				wakeUpWaiter( FERR_OK, TRUE);
				(void) waitForWrites( m_pCommitBuf, FALSE);
			}
			else
			{
				FLMBOOL	bSaveTransInProgress = m_pCurrentBuf->bTransInProgress;

				// Must set bTransInProgress to FALSE so that when the writer
				// of m_pCommitBuf finishes, it will signal the first waiter
				// on m_pCurrentBuf. If we don't do this, m_pCommitBuf will
				// simply be set to NULL, and the first waiter will never be
				// woke up.

				m_pCurrentBuf->bTransInProgress = FALSE;
				(void) waitForWrites( m_pCurrentBuf, FALSE);

				// It is OK to restore the trans in progress flag to what it
				// was before, because whoever called this routine has a lock
				// on the database, and it is his trans-in-progress state that
				// should be preserved. No other thread will have been able to
				// change that state because the database is locked.

				f_mutexLock( m_hBufMutex);
				m_pCurrentBuf->bTransInProgress = bSaveTransInProgress;
				f_mutexUnlock( m_hBufMutex);
			}
		}
		else if (m_pCommitBuf)
		{
			(void) waitForWrites( m_pCommitBuf, FALSE);
		}
		else
		{
			f_mutexUnlock( m_hBufMutex);
		}

		bWritesDone = TRUE;
	}

	return (bWritesDone);
}

/********************************************************************
Desc: Wake up the first thread that is waiting on the commit buffer.
*********************************************************************/
void F_Rfl::wakeUpWaiter(
	RCODE				rc,
	FLMBOOL			bIsWriter	// Only used for debug
	)
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
	if ((m_pCommitBuf->pFirstWaiter = m_pCommitBuf->pFirstWaiter->pNext) == NULL)
	{
		m_pCommitBuf->pLastWaiter = NULL;
	}

	f_semSignal( hESem);
}

/********************************************************************
Desc: Wait for the transaction writes to be finished.
*********************************************************************/
RCODE F_Rfl::completeTransWrites(
	FDB *			pDb,
	FLMBOOL		bCommitting,
	FLMBOOL		bOkToUnlock)
{
	RCODE			rc = FERR_OK;
	RCODE			tmpRc;
	FLMBOOL		bMutexLocked = FALSE;
	FLMBOOL		bNotifyWaiters = FALSE;
	FLMBOOL		bDbUnlocked = FALSE;
	DB_STATS *	pDbStats = NULL;
	F_TMSTAMP	StartTime;

	f_mutexLock( m_hBufMutex);
	bMutexLocked = TRUE;
	m_pCurrentBuf->bTransInProgress = FALSE;

	flmAssert( pDb->uiFlags & FDB_HAS_WRITE_LOCK);

	// If we are not logging, we are probably recovering or restoring the
	// database. All we need to do in this case is write out the log
	// header.

	if (pDb->uiFlags & FDB_REPLAYING_RFL)
	{
		if (pDb->bHadUpdOper && m_pCurrentBuf->bOkToWriteHdrs)
		{
			f_mutexUnlock( m_hBufMutex);
			bMutexLocked = FALSE;
			if (RC_BAD( rc = flmWriteLogHdr( pDb->pDbStats, pDb->pSFileHdl,
						  pDb->pFile, m_pCurrentBuf->ucLogHdr, m_pCurrentBuf->ucCPHdr,
						  FALSE)))
			{
				flmSetMustCloseFlags( pDb->pFile, rc, FALSE);
			}
		}

		goto Exit;
	}

	// Handle empty transactions differently. These transactions should
	// not do any writing and do not need to wait for all writes to
	// complete, unless the bOkToUnlock flag is set to FALSE. In that case
	// they must wait for all writes to complete before unlocking.

	if (!pDb->bHadUpdOper)
	{

		// If the current buffer has a waiter, add self to that list to
		// wait, because it will be notified after the commit buffer has
		// been notified. Otherwise, if there is a commit in progress, add
		// self to that list to wait.

		if (m_pCurrentBuf->pFirstWaiter)
		{

			// If m_pCommitBuf is NULL then nobody is going to wake up the
			// first waiter - we must do it.

			if (!m_pCommitBuf)
			{
				if (bOkToUnlock)
				{
					flmUnlinkDbFromTrans( pDb, bCommitting);
					bDbUnlocked = TRUE;
				}

				m_pCommitBuf = m_pCurrentBuf;
				switchBuffers();
				wakeUpWaiter( FERR_OK, TRUE);

				if (!bOkToUnlock)
				{
					bMutexLocked = FALSE;
					(void) waitForWrites( m_pCommitBuf, FALSE);
				}
			}
			else if (!bOkToUnlock)
			{
				bMutexLocked = FALSE;
				(void) waitForWrites( m_pCurrentBuf, FALSE);
			}
		}
		else if (m_pCommitBuf)
		{
			if (!bOkToUnlock)
			{
				bMutexLocked = FALSE;
				rc = waitForWrites( m_pCommitBuf, FALSE);
			}
		}

		goto Exit;
	}

	// If there is a transaction committing, put self into the wait list
	// on the current buffer. When the committer finishes, he will wake up
	// the first thread in the list and that thread will commit the buffer.

	if (m_pCommitBuf)
	{
		FLMBOOL	bIsWriter;

		// Another thread has to be doing the writes to m_pCommitBuf, which
		// means that m_pCurrentBuf better not be equal to m_pCommitBuf.

		flmAssert( m_pCommitBuf != m_pCurrentBuf);

		// If there are no waiters, we are the first one, so when we get
		// signaled, we should proceed and do the write.

		bIsWriter = m_pCurrentBuf->pFirstWaiter ? FALSE : TRUE;
		if (bOkToUnlock)
		{
			flmUnlinkDbFromTrans( pDb, bCommitting);
			bDbUnlocked = TRUE;
		}

		bMutexLocked = FALSE;
		rc = waitForWrites( m_pCurrentBuf, bIsWriter);

		// If we were the first one in the queue, we must now do the write.

		if (!bIsWriter)
		{
			goto Exit;
		}

		// First one in the queue, fall through to do the write. The thread
		// that woke me up better have set m_pCommitBuf See below.

		flmAssert( m_pCommitBuf);
	}
	else if (m_pCurrentBuf->pFirstWaiter)
	{

		// Another thread is ready to commit the next set of buffers, but
		// just needs to be woke up.

		if (bOkToUnlock)
		{
			flmUnlinkDbFromTrans( pDb, bCommitting);
			bDbUnlocked = TRUE;
		}

		// Need to set things up for that first waiter and get him going.

		m_pCommitBuf = m_pCurrentBuf;
		switchBuffers();
		wakeUpWaiter( rc, TRUE);

		// Wait for the write to be completed.

		bMutexLocked = FALSE;
		rc = waitForWrites( m_pCommitBuf, FALSE);
		goto Exit;
	}
	else
	{
		m_pCommitBuf = m_pCurrentBuf;
		switchBuffers();
		if (bOkToUnlock)
		{
			flmUnlinkDbFromTrans( pDb, bCommitting);
			bDbUnlocked = TRUE;
		}

		f_mutexUnlock( m_hBufMutex);
		bMutexLocked = FALSE;
	}

	// NOTE: From this point on we use tmpRc because we don't want to lose
	// the rc that may have been set above in the call to waitForWrites;
	// At this point the mutex better not be locked.
	flmAssert( !bMutexLocked);
	bNotifyWaiters = TRUE;

	if ((pDbStats = pDb->pDbStats) != NULL)
	{
		f_timeGetTimeStamp( &StartTime);
	}

	// Must write out whatever we have in the commit buffer before
	// unlocking the database.

	if (RC_BAD( tmpRc = flush( m_pCommitBuf, TRUE)))
	{
		if (RC_OK( rc))
		{
			rc = tmpRc;
		}

		goto Exit;
	}

	// Wait for any pending I/O off of the log buffer

	if (RC_BAD( tmpRc = m_pCommitBuf->pBufferMgr->waitForAllPendingIO()))
	{
		if (RC_OK( rc))
		{
			rc = tmpRc;
		}

		goto Exit;
	}

	// Force the RFL writes to disk if necessary. NOTE: It is possible for
	// m_pFileHdl to be NULL at this point if there were no operations
	// actually logged. This happens in FlmDbUpgrade (see flconvrt.cpp).
	// Even though nothing was logged the transaction is not an empty
	// transaction, because it still needs to write out the log header.

	if (m_pFileHdl)
	{
		if (RC_BAD( tmpRc = m_pFileHdl->flush()))
		{

			// Remap disk full error

			if (tmpRc == FERR_IO_DISK_FULL)
			{
				rc = RC_SET( FERR_RFL_DEVICE_FULL);
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
		if (RC_BAD( tmpRc = flmWriteLogHdr( pDb->pDbStats, pDb->pSFileHdl,
					  pDb->pFile, m_pCommitBuf->ucLogHdr, m_pCommitBuf->ucCPHdr,
					  FALSE)))
		{
			if (RC_OK( rc))
			{
				rc = tmpRc;
			}

			flmSetMustCloseFlags( pDb->pFile, tmpRc, FALSE);
			goto Exit;
		}
	}

Exit:

	if (!bDbUnlocked && bOkToUnlock)
	{
		flmUnlinkDbFromTrans( pDb, bCommitting);
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

		// If there are waiters on the current buffer, the first one should
		// be woke up so it can start the next set of writes.

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

	return (rc);
}

/********************************************************************
Desc: Calculate the checksum for a packet.
*********************************************************************/
FLMBYTE RflCalcChecksum(
	const FLMBYTE *	pucPacket,
	FLMUINT				uiPacketBodyLen)
{
	FLMUINT				uiBytesToChecksum;
	const FLMBYTE *	pucStart;
	
	// Checksum is calculated for every byte in the packet that comes
	// after the checksum byte.

	pucStart = &pucPacket[RFL_PACKET_CHECKSUM_OFFSET + 1];
	uiBytesToChecksum = (FLMUINT)(uiPacketBodyLen +
												RFL_PACKET_OVERHEAD -
												(RFL_PACKET_CHECKSUM_OFFSET + 1));

	return( f_calcPacketChecksum( pucStart, uiBytesToChecksum));
}

/********************************************************************
Desc:	Flush all completed packets out of the RFL buffer, and shift
		the new partial packet down.  This guarantees that there is
		now room in the buffer for the maximum packet size.
*********************************************************************/
RCODE F_Rfl::shiftPacketsDown(
	FLMUINT		uiCurrPacketLen,
	FLMBOOL		bStartingNewFile)
{
	RCODE 		rc = FERR_OK;

	// The call to flush will move whatever needs to be moved from the
	// current buffer into a new buffer if multiple buffers are being used.
	// If only one buffer is being used, it will move the part of the
	// packet that needs to be moved down to the beginning of the buffer -
	// AFTER writing out the buffer.

	if( RC_BAD( rc = flush( m_pCurrentBuf, FALSE, uiCurrPacketLen,
				  bStartingNewFile)))
	{
		goto Exit;
	}

	// NOTE: If multiple buffers are being used, whatever was moved to the
	// new buffer has not yet been written out

	if (bStartingNewFile)
	{
		if (RC_BAD( rc = waitPendingWrites()))
		{
			goto Exit;
		}
	}

Exit:

	return (rc);
}

/********************************************************************
Desc:		Determine if we should start a new file.  If we are over the
			low limit, and the bDoNewIfOverLowLimit flag is set, we
			will start a new log file.  Or, if this packet size would
			put us over the upper limit, we will start a new log file.
*********************************************************************/
RCODE F_Rfl::seeIfNeedNewFile(
	FLMUINT	uiPacketLen,
	FLMBOOL	bDoNewIfOverLowLimit)
{
	RCODE		rc = FERR_OK;
	FLMBYTE	ucNextSerialNum[F_SERIAL_NUM_SIZE];

	flmAssert( m_pFile);

	// If the keep files flag is FALSE, we won't start a new file. NOTE:
	// This should ALWAYS be false for pre 4.3 databases.

	if (!m_bKeepRflFiles)
	{
		goto Exit;	// Should return FERR_OK;
	}

	// VERY IMPORTANT NOTE: It is preferrable that we keep transactions
	// entirely contained in the same RFL file if at all possible. Note
	// that it is NOT a hard and fast requirement. The system will work
	// just fine if we don't. However, it would be nice if RFL files always
	// ended with a commit or abort packet. This preferences is due to what
	// happens after a restore operation. After a restore operation, we
	// always need to start a new RFL file, but if possible, we would like
	// that new RFL file to be the next one in the sequence after the last
	// RFL file that was restored. We can only do this if we were able to
	// restore EVERY transaction that was in the last restored RFL file -
	// which we can only do if the last restored RFL file ended with a
	// commit or abort packet. To accomplish this end, we try to roll to
	// new files on the first transaction begin packet that occurs after we
	// have exceeded our low threshold - which is why bDoNewIfOverLowLimit
	// is only set to TRUE on transaction begin packets. It is set to FALSE
	// on other packets so that we will continue logging the transaction in
	// the same file that we started the transaction in - if possible. The
	// only thing that will cause a non-transaction-begin packet to roll to
	// a new file is if we would exceed the high limit.

	if ((bDoNewIfOverLowLimit &&
		  m_pCurrentBuf->uiRflFileOffset + m_pCurrentBuf->uiRflBufBytes >=
				m_uiRflMinFileSize) ||
		 (m_pCurrentBuf->uiRflFileOffset + m_pCurrentBuf->uiRflBufBytes +
			uiPacketLen >= m_uiRflMaxFileSize))
	{
		FLMUINT	uiCurrFileEOF = m_pCurrentBuf->uiRflFileOffset +
										 m_pCurrentBuf->uiRflBufBytes;

		// Shift the current packet to the beginning of the buffer. Any
		// packets in the buffer before that one will be written out to the
		// current file.

		if (RC_BAD( rc = shiftPacketsDown( uiPacketLen, TRUE)))
		{
			goto Exit;
		}

		// Update the header of the current file and close it.

		if (RC_BAD( rc = writeHeader( m_pCurrentBuf->uiCurrFileNum, uiCurrFileEOF,
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

		m_pFileHdl->Release();
		m_pFileHdl = NULL;

		// Get the next serial number that will be used for the RFL file
		// after this one.

		if (RC_BAD( rc = f_createSerialNumber( ucNextSerialNum)))
		{
			goto Exit;
		}

		// Create next file in the sequence. Use the next serial number
		// stored in the FDB's log header for the serial number on this RFL
		// file. Use the serial number we just generated as the next RFL
		// serial number.

		if (RC_BAD( rc = createFile( m_pCurrentBuf->uiCurrFileNum + 1,
					  m_ucNextSerialNum, ucNextSerialNum, TRUE)))
		{
			goto Exit;
		}

		// Move the next serial number to the current serial number and the
		// serial number we generated above into the next serial number.

		f_memcpy( m_ucCurrSerialNum, m_ucNextSerialNum, F_SERIAL_NUM_SIZE);
		f_memcpy( m_ucNextSerialNum, ucNextSerialNum, F_SERIAL_NUM_SIZE);
	}

Exit:

	return (rc);
}

/********************************************************************
Desc: Finish the current RFL file - set up so that next transaction
		will begin a new RFL file.
********************************************************************/
RCODE F_Rfl::finishCurrFile(
	FDB *		pDb,
	FLMBOOL	bNewKeepState)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bDbLocked = FALSE;
	FLMUINT		uiTransFileNum;
	FLMUINT		uiTransOffset;
	FLMUINT		uiTruncateSize;
	FLMBYTE *	pucUncommittedLogHdr;
	FLMBYTE 		ucCheckpointLogHdr[ LOG_HEADER_SIZE];

	// Make sure we don't have a transaction going

	if (pDb->uiTransType != FLM_NO_TRANS)
	{
		rc = RC_SET( FERR_TRANS_ACTIVE);
		goto Exit;
	}

	// Make sure there is no active backup running

	f_mutexLock( gv_FlmSysData.hShareMutex);
	if (m_pFile->bBackupActive)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		rc = RC_SET( FERR_BACKUP_ACTIVE);
		goto Exit;
	}

	f_mutexUnlock( gv_FlmSysData.hShareMutex);

	// Lock the database - need to prevent update transactions and
	// checkpoint thread from running.

	if (RC_BAD( rc = dbLock( pDb, FLM_NO_TIMEOUT)))
	{
		goto Exit;
	}

	bDbLocked = TRUE;

	// Must wait for all RFL writes before switching files.

	(void) seeIfRflWritesDone( TRUE);

	// Better not be in the middle of logging unknown stuff for the
	// application

	flmAssert( !m_bLoggingUnknown);

	// Better not be in the middle of a transaction.

	flmAssert( !m_uiCurrTransID);

	// If DB version is less than 4.3 we cannot do this.

	if (m_pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_3)
	{
		goto Exit;		// Will return FERR_OK
	}

	pucUncommittedLogHdr = &m_pFile->ucUncommittedLogHdr[0];

	// Don't want to copy last committed log header into uncommitted log
	// header if bNewKeepState is TRUE because the caller has already done
	// it, and has made modifications to the uncommitted log header that we
	// don't want to lose.

	if (!bNewKeepState)
	{
		f_memcpy( pucUncommittedLogHdr, m_pFile->ucLastCommittedLogHdr,
					LOG_HEADER_SIZE);

		// If we are in a no-keep state, but we were not told that we have
		// a new keep state, we cannot roll to the next RFL file, because a
		// checkpoint has not been done. This is not an error - it is just
		// the case where FlmDbConfig was asked to roll to the next RFL file
		// when the keep flag was still FALSE.

		if (!pucUncommittedLogHdr[LOG_KEEP_RFL_FILES])
		{
			goto Exit;	// Will return FERR_OK
		}
	}

	// Get the last committed serial numbers from the file's log header
	// buffer.

	f_memcpy( m_ucCurrSerialNum,
				&pucUncommittedLogHdr[LOG_LAST_TRANS_RFL_SERIAL_NUM],
				F_SERIAL_NUM_SIZE);
				
	f_memcpy( m_ucNextSerialNum, &pucUncommittedLogHdr[LOG_RFL_NEXT_SERIAL_NUM],
				F_SERIAL_NUM_SIZE);
				
	uiTransFileNum = (FLMUINT) FB2UD( &pucUncommittedLogHdr[LOG_RFL_FILE_NUM]);
	uiTransOffset = (FLMUINT) FB2UD( &pucUncommittedLogHdr[LOG_RFL_LAST_TRANS_OFFSET]);

	// If the LOG_RFL_LAST_TRANS_OFFSET is zero, there is no need to go
	// set up to go to the next file, because we are already poised to do
	// so at the beginning of the next transaction. Just return if this is
	// the case. Same for if the file does not exist.

	if (!uiTransOffset)
	{
		if (!bNewKeepState)
		{
			goto Exit;	// Will return FERR_OK
		}
	}
	else if (RC_BAD( rc = openFile( uiTransFileNum, m_ucCurrSerialNum)))
	{
		if (rc == FERR_IO_PATH_NOT_FOUND || rc == FERR_IO_INVALID_PATH)
		{
			rc = FERR_OK;
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

		// At this point, we know the file exists, so we will update its
		// header and then update the log header. Note that we use the keep
		// RFL state from the last committed log header, not the uncommitted
		// log header - because it will contain the correct keep-state for
		// the current RFL file.

		if (RC_BAD( rc = writeHeader( m_pCurrentBuf->uiCurrFileNum, uiTransOffset,
										m_ucCurrSerialNum, m_ucNextSerialNum,
										m_pFile->ucLastCommittedLogHdr [LOG_KEEP_RFL_FILES]
											? TRUE
											: FALSE)))
		{
			goto Exit;
		}

		// Truncate the file down to its EOF size - the nearest 512 byte
		// boundary.

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

		m_pFileHdl->Release();
		m_pFileHdl = NULL;

		// Set things up in the log header to go to the next file when we
		// begin the next transaction. NOTE: NO need to lock the mutex,
		// because nobody but an update transaction looks at the uncommitted
		// log header.

		uiTransFileNum++;
		UD2FBA( (FLMUINT32) uiTransFileNum,
				 &pucUncommittedLogHdr[LOG_RFL_FILE_NUM]);
	}

	// Generate a new current serial number if bNewKeepState is TRUE.
	// Otherwise, move the next serial number into the current serial
	// number.

	if (bNewKeepState)
	{
		if (RC_BAD( rc = f_createSerialNumber( m_ucCurrSerialNum)))
		{
			goto Exit;
		}
	}
	else
	{
		f_memcpy( m_ucCurrSerialNum, m_ucNextSerialNum, F_SERIAL_NUM_SIZE);
	}

	// Always generate a new next serial number.

	if (RC_BAD( rc = f_createSerialNumber( m_ucNextSerialNum)))
	{
		goto Exit;
	}

	// Set transaction offset to zero. This will force the next RFL file
	// to be created on the next transaction begin. It will be created even
	// if it is already there.

	UD2FBA( (FLMUINT32) 0, &pucUncommittedLogHdr[LOG_RFL_LAST_TRANS_OFFSET]);
	
	f_memcpy( &pucUncommittedLogHdr[LOG_LAST_TRANS_RFL_SERIAL_NUM],
				m_ucCurrSerialNum, F_SERIAL_NUM_SIZE);
				
	f_memcpy( &pucUncommittedLogHdr[LOG_RFL_NEXT_SERIAL_NUM], m_ucNextSerialNum,
				F_SERIAL_NUM_SIZE);

	// Set the CP file number and CP offset to point into the new file.
	// The outer code (FlmDbConfig) has done a checkpoint and the database
	// is still locked. We need to set these values here, otherwise if we
	// crash before the next checkpoint, recovery will start in the old RFL
	// file, causing an FERR_BAD_RFL_SERIAL_NUM to be returned when
	// traversing from the old RFL file to the new RFL file. NOTE: These
	// changes must be made to the uncommitted log header AND the CP log
	// header (so that they will be written out even though we are not
	// forcing a checkpoint).

	if (bNewKeepState)
	{
#ifdef FLM_DEBUG
		// Do a quick check to see if it looks like we are in a
		// checkpointed state

		if (!m_pFile->ucLastCommittedLogHdr[LOG_KEEP_RFL_FILES] &&
			 (FLMUINT) FB2UD( &m_pFile->ucLastCommittedLogHdr[
										LOG_RFL_LAST_TRANS_OFFSET]) > 512)
		{
			flmAssert( 0);
		}
#endif

		f_memcpy( ucCheckpointLogHdr, m_pFile->ucCheckpointLogHdr,
					LOG_HEADER_SIZE);
		
		UD2FBA( (FLMUINT32) uiTransFileNum,
				 &ucCheckpointLogHdr[LOG_RFL_LAST_CP_FILE_NUM]);
				 
		UD2FBA( (FLMUINT32) uiTransFileNum,
				 &pucUncommittedLogHdr[LOG_RFL_LAST_CP_FILE_NUM]);
				 
		UD2FBA( (FLMUINT32) 512, &ucCheckpointLogHdr[LOG_RFL_LAST_CP_OFFSET]);
		
		UD2FBA( (FLMUINT32) 512, &pucUncommittedLogHdr[LOG_RFL_LAST_CP_OFFSET]);
	}

	// Write out the log header to disk.

	if (RC_BAD( rc = flmWriteLogHdr( pDb->pDbStats, pDb->pSFileHdl,
									m_pFile, pucUncommittedLogHdr,
									bNewKeepState 
										? ucCheckpointLogHdr
										: m_pFile->ucCheckpointLogHdr, FALSE)))
	{
		goto Exit;
	}

	// Copy the uncommitted log header back to the committed log header
	// and copy the CP log header (if changed above).

	f_mutexLock( gv_FlmSysData.hShareMutex);
	f_memcpy( m_pFile->ucLastCommittedLogHdr, pucUncommittedLogHdr,
				LOG_HEADER_SIZE);

	if (bNewKeepState)
	{
		f_memcpy( m_pFile->ucCheckpointLogHdr, ucCheckpointLogHdr,
					LOG_HEADER_SIZE);
	}

	f_mutexUnlock( gv_FlmSysData.hShareMutex);

Exit:

	if (bDbLocked)
	{
		(void) dbUnlock( pDb);
	}

	return (rc);
}

/********************************************************************
Desc:		Finish packet by outputting header information for it.
*********************************************************************/
RCODE F_Rfl::finishPacket(
	FLMUINT	uiPacketType,
	FLMUINT	uiPacketBodyLen,
	FLMBOOL	bDoNewIfOverLowLimit)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiEncryptPacketBodyLen;
	FLMUINT		uiPacketLen;
	FLMBYTE *	pucPacket;

	// Encrypt the packet body, if requested.

	uiEncryptPacketBodyLen = getEncryptPacketBodyLen( uiPacketType,
																	 uiPacketBodyLen);
	uiPacketLen = uiEncryptPacketBodyLen + RFL_PACKET_OVERHEAD;

	// See if this packet will cause us to overflow the limits on the
	// current file. If so, create a new file.

	if (RC_BAD( rc = seeIfNeedNewFile( uiPacketLen, bDoNewIfOverLowLimit)))
	{
		goto Exit;
	}

	// Get a pointer to packet header.

	pucPacket = &(m_pCurrentBuf->pIOBuffer->getBufferPtr()[
											m_pCurrentBuf->uiRflBufBytes]);

	// Set the packet address in the packet header.

	m_uiPacketAddress = m_pCurrentBuf->uiRflFileOffset + 
									m_pCurrentBuf->uiRflBufBytes;
									
	UD2FBA( (FLMUINT32) m_uiPacketAddress, &pucPacket[
															RFL_PACKET_ADDRESS_OFFSET]);

	// Set the packet type and packet body length.

	pucPacket[RFL_PACKET_TYPE_OFFSET] = (FLMBYTE) uiPacketType;
	
	UW2FBA( (FLMUINT16) uiPacketBodyLen,
			 &pucPacket[RFL_PACKET_BODY_LENGTH_OFFSET]);

	// Set the checksum for the packet.

	pucPacket[RFL_PACKET_CHECKSUM_OFFSET] = 
						RflCalcChecksum( pucPacket, uiEncryptPacketBodyLen);

	// Increment bytes in the buffer to reflect the fact that this packet
	// is now complete.

	m_pCurrentBuf->uiRflBufBytes += uiPacketLen;
	
Exit:

	return (rc);
}

/********************************************************************
Desc:	Truncate roll-forward log file to a certain size - only do if 
		not keeping RFL files.
*********************************************************************/
RCODE F_Rfl::truncate(
	FLMUINT		uiTruncateSize)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiFileNum;

	flmAssert( uiTruncateSize >= 512);

	// Keeping of log files better not be enabled.

	flmAssert( !m_pFile->ucLastCommittedLogHdr[LOG_KEEP_RFL_FILES]);

	// Better not be in the middle of logging unknown stuff for the
	// application

	flmAssert( !m_bLoggingUnknown);

	// Better not be in the middle of a transaction.

	flmAssert( !m_uiCurrTransID);

	// Open the current RFL file. If it does not exist, it is OK - there
	// is nothing to truncate.

	uiFileNum = (FLMUINT) FB2UD( &m_pFile->ucLastCommittedLogHdr[
																		LOG_RFL_FILE_NUM]);
	
	if (RC_BAD( rc = openFile( uiFileNum, &m_pFile->ucLastCommittedLogHdr[
					  LOG_LAST_TRANS_RFL_SERIAL_NUM])))
	{
		if (rc == FERR_IO_PATH_NOT_FOUND || rc == FERR_IO_INVALID_PATH)
		{
			rc = FERR_OK;
		}

		goto Exit;
	}

	if (RC_BAD( rc = m_pFileHdl->truncateFile( uiTruncateSize)))
	{
		m_bRflVolumeOk = FALSE;
		goto Exit;
	}

Exit:

	return (rc);
}

/********************************************************************
Desc:		Setup to begin a transaction
*********************************************************************/
RCODE F_Rfl::setupTransaction(void)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiFileNum;
	FLMUINT		uiLastTransOffset;
	FLMBOOL		bCreateFile;

	f_mutexLock( m_hBufMutex);
	m_pCurrentBuf->bTransInProgress = TRUE;
	f_mutexUnlock( m_hBufMutex);

	// Get the last committed serial numbers from the file's log header
	// buffer.

	f_memcpy( m_ucCurrSerialNum,
				&m_pFile->ucLastCommittedLogHdr[LOG_LAST_TRANS_RFL_SERIAL_NUM],
				F_SERIAL_NUM_SIZE);
				
	f_memcpy( m_ucNextSerialNum,
				&m_pFile->ucLastCommittedLogHdr[LOG_RFL_NEXT_SERIAL_NUM],
				F_SERIAL_NUM_SIZE);
				
	uiFileNum = (FLMUINT) FB2UD( &m_pFile->ucLastCommittedLogHdr[
															LOG_RFL_FILE_NUM]);
	
	uiLastTransOffset = (FLMUINT) FB2UD( &m_pFile->ucLastCommittedLogHdr[ 
															LOG_RFL_LAST_TRANS_OFFSET]);

	// If the LOG_RFL_LAST_TRANS_OFFSET is zero, we need to create the
	// next RFL file number no matter what. There are two cases where this
	// happens: 1) when the database is first created, and 2) after a
	// restore operation.

	if (!uiLastTransOffset)
	{
		bCreateFile = TRUE;

		// Close the current file, just in case we had opened it before. At
		// this point, it doesn't matter because we are going to overwrite
		// it.

		if (RC_BAD( rc = waitForCommit()))
		{
			goto Exit;
		}

		closeFile();
	}
	else if (RC_BAD( rc = openFile( uiFileNum, m_ucCurrSerialNum)))
	{
		if (rc != FERR_IO_PATH_NOT_FOUND && rc != FERR_IO_INVALID_PATH)
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

		// If the log header indicates that data has already been logged to
		// the file, we need to return the I/O error rather than just
		// re-creating the file. This may mean that someone changed the RFL
		// directory without moving the RFL files properly.

		if (uiLastTransOffset > 512)
		{
			rc = RC_SET( FERR_RFL_FILE_NOT_FOUND);
			goto Exit;
		}

		// Create the RFL file if not found. Use the next serial number
		// stored in the FDB's log header for the serial number on this RFL
		// file. Use the serial number we just generated as the next RFL
		// serial number.

		if (RC_BAD( rc = createFile( uiFileNum,
									m_ucCurrSerialNum, m_ucNextSerialNum,
				m_pFile->ucLastCommittedLogHdr [LOG_KEEP_RFL_FILES]
				? TRUE
				: FALSE)))
		{
			goto Exit;
		}
	}
	else
	{

		// Read in enough of the buffer from the RFL file so that we are
		// positioned on a 512 byte boundary.

		if (RC_BAD( positionTo( uiLastTransOffset)))
		{
			goto Exit;
		}
	}

	// These can only be changed when starting a transaction.

	if (m_pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_3)
	{
		m_bKeepRflFiles = m_pFile->ucLastCommittedLogHdr[LOG_KEEP_RFL_FILES] 
									? TRUE 
									: FALSE;
		
		m_uiRflMaxFileSize = (FLMUINT) FB2UD( &m_pFile->ucLastCommittedLogHdr[
													LOG_RFL_MAX_FILE_SIZE]);

		// Round maximum down to nearest 512 boundary. This is necessary
		// because we always write a minimum of 512 byte units in direct IO
		// mode. If we did not round the maximum down, our last packet could
		// end at an offset that is less than the maximum, but greater than
		// the nearest 512 byte boundary - technically within the
		// user-specified size limit. However, because we always write a
		// full 512 bytes of data to fill out the last block when we are in
		// direct IO mode, we would end up with a file that was slightly
		// larger than the user-specified limit. The EOF in the header of
		// the file would be below the limit, but the actual file size would
		// not be. Thus, the need to round down.

		m_uiRflMaxFileSize = ROUND_DOWN_TO_NEAREST_512( m_uiRflMaxFileSize);

		// The maximum cannot go below a certain threshold - must have room
		// for least one packet plus the header.

		if (m_uiRflMaxFileSize < RFL_MAX_PACKET_SIZE + 512)
		{
			m_uiRflMaxFileSize = RFL_MAX_PACKET_SIZE + 512;
		}
		else if (m_uiRflMaxFileSize > gv_FlmSysData.uiMaxFileSize)
		{
			m_uiRflMaxFileSize = gv_FlmSysData.uiMaxFileSize;
		}
	}
	else
	{
		m_bKeepRflFiles = FALSE;
		m_uiRflMaxFileSize = gv_FlmSysData.uiMaxFileSize;
	}
	
	m_uiRflMinFileSize = (FLMUINT) FB2UD( &m_pFile->ucLastCommittedLogHdr[
												LOG_RFL_MIN_FILE_SIZE]);

	// Minimum RFL file size should not be allowed to be larger than
	// maximum!

	if (m_uiRflMinFileSize > m_uiRflMaxFileSize)
	{
		m_uiRflMinFileSize = m_uiRflMaxFileSize;
	}

	// Set the operation count to zero.

	m_uiOperCount = 0;

	// Set file extend sizes

	m_pFileHdl->setMaxAutoExtendSize( m_uiRflMaxFileSize);
	m_pFileHdl->setExtendSize( m_pFile->uiFileExtendSize);

Exit:

	return (rc);
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
	FDB *			pDb)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiDbVersion = pDb->pFile->FileHdr.uiVersionNum;
	FLMBYTE *	pucPacketBody;
	FLMUINT		uiPacketBodyLen;
	FLMUINT		uiGMTTime;

	// Do nothing if logging is disabled.

	if (m_bLoggingOff)
	{
		goto Exit;
	}

	// Better not be in the middle of logging unknown stuff for the
	// application

	flmAssert( !m_bLoggingUnknown);

	// Better not be in the middle of a transaction.

	flmAssert( !m_uiCurrTransID);

	if (RC_BAD( rc = setupTransaction()))
	{
		goto Exit;
	}

	uiPacketBodyLen = uiDbVersion >= FLM_FILE_FORMAT_VER_4_31 ? 12 : 8;

	// Make sure we have space in the RFL buffer for a complete packet.

	if (!haveBuffSpace( uiPacketBodyLen + RFL_PACKET_OVERHEAD))
	{
		if (RC_BAD( rc = flush( m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	// Get a pointer to where we will be laying down the packet body.

	pucPacketBody = getPacketBodyPtr();

	// Output the transaction ID.

	UD2FBA( (FLMUINT32) pDb->LogHdr.uiCurrTransID, pucPacketBody);
	pucPacketBody += 4;

	// This used to be a FLM_GET_TIMER() value in pre-4.3 code, but it was
	// never really used. Set it to GMT time now.

	f_timeGetSeconds( &uiGMTTime);
	UD2FBA( (FLMUINT32) uiGMTTime, pucPacketBody);
	pucPacketBody += 4;

	// NOTE: In the pre-4.3 code the next four bytes would be zero, but
	// that is really unnecessary. We will simply no longer set the
	// RFL_TIME_LOGGED_FLAG bit in the packet type. Pre-4.3 code should be
	// compatible.

	if (uiDbVersion >= FLM_FILE_FORMAT_VER_4_31)
	{
		FLMUINT	uiLastLoggedCommitID;

		uiLastLoggedCommitID = FB2UD( 
					&m_pFile->ucLastCommittedLogHdr[LOG_LAST_RFL_COMMIT_ID]);

		UD2FBA( (FLMUINT32) uiLastLoggedCommitID, pucPacketBody);
		pucPacketBody += 4;

		if (RC_BAD( rc = finishPacket( RFL_TRNS_BEGIN_EX_PACKET, uiPacketBodyLen,
					  TRUE)))
		{
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = finishPacket( RFL_TRNS_BEGIN_PACKET, uiPacketBodyLen,
					  TRUE)))
		{
			goto Exit;
		}
	}

	// Save the file offset for the start transaction packet.

	m_uiTransStartFile = m_pCurrentBuf->uiCurrFileNum;
	m_uiTransStartAddr = m_pCurrentBuf->uiRflFileOffset +
		m_pCurrentBuf->uiRflBufBytes -
		uiPacketBodyLen -
		RFL_PACKET_OVERHEAD;
	m_uiCurrTransID = pDb->LogHdr.uiCurrTransID;

Exit:

	return (rc);
}

/********************************************************************
Desc:		Flushes the RFL and sets some things in the log header.
*********************************************************************/
void F_Rfl::finalizeTransaction(void)
{
	FLMUINT		uiRflTransEndOffset;
	FLMBYTE *	pucLogHdr = &m_pFile->ucUncommittedLogHdr[0];

	// Save the serial numbers and file numbers into the file's
	// uncommitted log header.

	UD2FBA( (FLMUINT32) m_pCurrentBuf->uiCurrFileNum,
			 &pucLogHdr[LOG_RFL_FILE_NUM]);

	uiRflTransEndOffset = getCurrWriteOffset();
	UD2FBA( (FLMUINT32) uiRflTransEndOffset,
			 &pucLogHdr[LOG_RFL_LAST_TRANS_OFFSET]);

	f_memcpy( &pucLogHdr[LOG_LAST_TRANS_RFL_SERIAL_NUM], m_ucCurrSerialNum,
				F_SERIAL_NUM_SIZE);

	f_memcpy( &pucLogHdr[LOG_RFL_NEXT_SERIAL_NUM], m_ucNextSerialNum,
				F_SERIAL_NUM_SIZE);
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
Note:		The prior version of FLAIM (before 4.3) would log
			a time and set the RFL_TIME_LOGGED_FLAG bit in the packet
			type.  This is no longer done.  Old code should be
			compatible because it reads the flag.
*********************************************************************/
RCODE F_Rfl::logEndTransaction(
	FLMUINT		uiPacketType,
	FLMBOOL		bThrowLogAway,
	FLMBOOL *	pbLoggedTransEnd)
{
	RCODE			rc = FERR_OK;
	RCODE			rc2 = FERR_OK;
	FLMUINT		uiLowFileNum;
	FLMUINT		uiHighFileNum;
	char			szRflFileName[ F_PATH_MAX_SIZE];
	FLMBYTE *	pucPacketBody;
	FLMUINT		uiPacketBodyLen;

	// Better not be in the middle of logging unknown stuff for the
	// application

	flmAssert( !m_bLoggingUnknown);

	// Initialize the "logged trans end" flag

	if (pbLoggedTransEnd)
	{
		*pbLoggedTransEnd = FALSE;
	}

	// Do nothing if logging is disabled.

	if (m_bLoggingOff)
	{
		goto Exit;
	}

	flmAssert( m_pFileHdl);
	flmAssert( m_pFile);

	// If the transaction had no operations, throw it away - don't even
	// log the packet. An abort operation may also elect to throw the log
	// away even if there were operations. That is determined by the
	// bThrowLogAway flag. The bThrowLogAway flag may be TRUE when doing a
	// commit if the caller knows that nothing happened during the
	// transction.

	if (bThrowLogAway || !m_uiOperCount)
	{
Throw_Away_Transaction:

		// If we have switched files, delete all but the file we started in.

		if (m_pCurrentBuf->uiCurrFileNum != m_uiTransStartFile)
		{
			flmAssert( m_pCurrentBuf->uiCurrFileNum > m_uiTransStartFile);

			// File number in uncommitted log header better not have been
			// changed yet. It is only supposed to be changed when the
			// transaction finishes - i.e., in this routine. Up until this
			// point, it should only be changed in
			// m_pCurrentBuf->uiCurrFileNum.

			flmAssert( m_uiTransStartFile == (FLMUINT) FB2UD(
							 &m_pFile->ucUncommittedLogHdr[LOG_RFL_FILE_NUM]));

			uiLowFileNum = m_uiTransStartFile + 1;
			uiHighFileNum = m_pCurrentBuf->uiCurrFileNum;

			// Close the current file so it can be deleted.

			if (RC_BAD( rc = waitForCommit()))
			{
				goto Exit;
			}

			closeFile();

			// Delete as many of the files as possible. Don't worry about
			// errors here.

			while (uiLowFileNum <= uiHighFileNum)
			{
				if (RC_OK( getFullRflFileName( uiLowFileNum, szRflFileName)))
				{
					(void) gv_FlmSysData.pFileSystem->deleteFile( szRflFileName);
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

				// If we got to this point because of a "goto
				// Throw_Away_Transaction", we don't want to clobber the
				// original error code. So, we use rc2 temporarily and then
				// determine if its value should be set into rc.

				if (RC_OK( rc))
				{
					rc = rc2;
				}

				rc2 = FERR_OK;
				goto Exit;
			}
		}
	}
	else
	{

		// Log a commit or abort packet.

		uiPacketBodyLen = 8;

		// Make sure we have space in the RFL buffer for a complete packet.

		if (!haveBuffSpace( uiPacketBodyLen + RFL_PACKET_OVERHEAD))
		{
			if (RC_BAD( rc = flush( m_pCurrentBuf)))
			{
				goto Throw_Away_Transaction;
			}
		}

		// Get a pointer to where we will be laying down the packet body.

		pucPacketBody = getPacketBodyPtr();

		// Output the transaction ID.

		UD2FBA( (FLMUINT32) m_uiCurrTransID, pucPacketBody);
		pucPacketBody += 4;
		UD2FBA( (FLMUINT32) m_uiTransStartAddr, pucPacketBody);
		pucPacketBody += 4;
		if (RC_BAD( rc = finishPacket( uiPacketType, uiPacketBodyLen, FALSE)))
		{
			goto Throw_Away_Transaction;
		}

		finalizeTransaction();

		if (pbLoggedTransEnd)
		{
			*pbLoggedTransEnd = TRUE;
		}
	}

Exit:

	if (!m_bLoggingOff)
	{
		m_uiCurrTransID = 0;
	}

	return (RC_BAD( rc) ? rc : rc2);
}

/********************************************************************
Desc:		Log add, modify, delete, and reserve DRN packets
*********************************************************************/
RCODE F_Rfl::logUpdatePacket(
	FLMUINT	uiPacketType,
	FLMUINT	uiContainer,
	FLMUINT	uiDrn,
	FLMUINT	uiAutoTrans)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pucPacketBody;
	FLMUINT		uiPacketBodyLen;

	// Do nothing if logging is disabled.

	if (m_bLoggingOff)
	{
		goto Exit;
	}

	// Better not be in the middle of logging unknown stuff for the
	// application

	flmAssert( !m_bLoggingUnknown);

	// Better be in the middle of a transaction.

	flmAssert( m_uiCurrTransID);

	m_uiOperCount++;

	if (uiPacketType == RFL_ADD_RECORD_PACKET_VER_2 ||
		 uiPacketType == RFL_MODIFY_RECORD_PACKET_VER_2 ||
		 uiPacketType == RFL_DELETE_RECORD_PACKET_VER_2)
	{
		uiPacketBodyLen = 11;
	}
	else
	{
		uiPacketBodyLen = 10;
	}

	// Make sure we have space in the RFL buffer for a complete packet.

	if (!haveBuffSpace( uiPacketBodyLen + RFL_PACKET_OVERHEAD))
	{
		if (RC_BAD( rc = flush( m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	// Get a pointer to where we will be laying down the packet body.

	pucPacketBody = getPacketBodyPtr();

	// Output the transaction ID.

	UD2FBA( (FLMUINT32) m_uiCurrTransID, pucPacketBody);
	pucPacketBody += 4;

	// Output the container number.

	UW2FBA( (FLMUINT16) uiContainer, pucPacketBody);
	pucPacketBody += 2;

	// Output the DRN.

	UD2FBA( (FLMUINT32) uiDrn, pucPacketBody);
	pucPacketBody += 4;

	// Output the flags

	if (uiPacketType == RFL_ADD_RECORD_PACKET_VER_2 ||
		 uiPacketType == RFL_MODIFY_RECORD_PACKET_VER_2 ||
		 uiPacketType == RFL_DELETE_RECORD_PACKET_VER_2)
	{
		FLMUINT	uiFlags = 0;

		// For now, these are the only flags we log

		if (uiAutoTrans & FLM_DO_IN_BACKGROUND)
		{
			uiFlags |= RFL_UPDATE_BACKGROUND;
		}

		if (uiAutoTrans & FLM_SUSPENDED)
		{
			uiFlags |= RFL_UPDATE_SUSPENDED;
		}

		*pucPacketBody++ = (FLMBYTE) uiFlags;
	}

	// Finish the packet

	if (RC_BAD( rc = finishPacket( uiPacketType, uiPacketBodyLen, FALSE)))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/********************************************************************
Desc:		Log index suspend and resume packets
*********************************************************************/
RCODE F_Rfl::logIndexSuspendOrResume(
	FLMUINT	uiIndexNum,
	FLMUINT	uiPacketType)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pucPacketBody;
	FLMUINT		uiPacketBodyLen;

	// This call is new with 4.51 databases - not supported in older
	// versions, so don't log it.

	if (m_pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_51)
	{
		goto Exit;
	}

	// Do nothing if logging is disabled.

	if (m_bLoggingOff)
	{
		goto Exit;
	}

	// Better not be in the middle of logging unknown stuff for the
	// application

	flmAssert( !m_bLoggingUnknown);

	// Better be in the middle of a transaction.

	flmAssert( m_uiCurrTransID);

	m_uiOperCount++;
	uiPacketBodyLen = 6;

	// Make sure we have space in the RFL buffer for a complete packet.

	if (!haveBuffSpace( uiPacketBodyLen + RFL_PACKET_OVERHEAD))
	{
		if (RC_BAD( rc = flush( m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	// Get a pointer to where we will be laying down the packet body.

	pucPacketBody = getPacketBodyPtr();

	// Output the transaction ID.

	UD2FBA( (FLMUINT32) m_uiCurrTransID, pucPacketBody);
	pucPacketBody += 4;

	// Output the index number.

	UW2FBA( (FLMUINT16) uiIndexNum, pucPacketBody);
	pucPacketBody += 2;

	// Finish the packet

	if (RC_BAD( rc = finishPacket( uiPacketType, uiPacketBodyLen, FALSE)))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/********************************************************************
Desc:
*********************************************************************/
RCODE F_Rfl::logSizeEventConfig(
	FLMUINT		uiTransID,
	FLMUINT		uiSizeThreshold,
	FLMUINT		uiSecondsBetweenEvents,
	FLMUINT		uiBytesBetweenEvents)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pucPacketBody;
	FLMUINT		uiPacketBodyLen;

	// Don't log the operation if it isn't supported by the current database
	// version

	if (m_pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_61)
	{
		goto Exit;
	}

	// Do nothing if logging is disabled.

	if (m_bLoggingOff)
	{
		goto Exit;
	}

	// Better not be in the middle of logging unknown stuff for the
	// application

	flmAssert( !m_bLoggingUnknown);

	// We need to set up to log this packet as if we were logging a
	// transaction. The only difference is that we don't log the begin
	// transaction packet.

	if (RC_BAD( rc = setupTransaction()))
	{
		goto Exit;
	}

	uiPacketBodyLen = 16;

	// Make sure we have space in the RFL buffer for a complete packet.

	if( !haveBuffSpace( uiPacketBodyLen + RFL_PACKET_OVERHEAD))
	{
		if (RC_BAD( rc = flush( m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	// Get a pointer to where we will be laying down the packet body.

	pucPacketBody = getPacketBodyPtr();

	// Output the transaction ID.

	UD2FBA( (FLMUINT32) uiTransID, pucPacketBody);
	pucPacketBody += 4;

	// Output the size threshold

	UD2FBA( (FLMUINT32) uiSizeThreshold, pucPacketBody);
	pucPacketBody += 4;

	// Output the time frequency

	UD2FBA( (FLMUINT32) uiSecondsBetweenEvents, pucPacketBody);
	pucPacketBody += 4;
	
	// Output the size frequency

	UD2FBA( (FLMUINT32) uiBytesBetweenEvents, pucPacketBody);
	pucPacketBody += 4;
	
	// Finish the packet

	if (RC_BAD( rc = finishPacket( RFL_CONFIG_SIZE_EVENT_PACKET, 
		uiPacketBodyLen, TRUE)))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/********************************************************************
Desc:		Make room in the RFL buffer for the additional bytes.
			This is done by flushing the log buffer and shifting down
			the bytes already used in the current packet.  If that doesn't
			make room, the current packet will be finished and a new one
			started.
*********************************************************************/
RCODE F_Rfl::makeRoom(
	FLMUINT		uiAdditionalBytesNeeded,
	FLMUINT *	puiCurrPacketLenRV,
	FLMUINT		uiPacketType,
	FLMUINT *	puiBytesAvailableRV,
	FLMUINT *	puiPacketCountRV)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiBytesNeeded;

	// Must account for encryption, so round bytes needed to nearest four
	// byte boundary.

	uiBytesNeeded = *puiCurrPacketLenRV + uiAdditionalBytesNeeded;
	if (uiBytesNeeded & 0x3)
	{
		uiBytesNeeded += (4 - (uiBytesNeeded & 0x3));
	}

	if (uiBytesNeeded <= (FLMUINT) RFL_MAX_PACKET_SIZE)
	{
		FLMUINT	uiTmp = uiBytesNeeded;

		if (haveBuffSpace( uiTmp))
		{
			if (puiBytesAvailableRV)
			{
				*puiBytesAvailableRV = uiAdditionalBytesNeeded;
			}
		}
		else
		{

			// Bytes requested will fit into a packet, but not the buffer,
			// so we need to shift the packets in the buffer down. The
			// shiftPacketsDown guarantees that there is room in the buffer
			// for a full size packet.

			if (RC_BAD( rc = shiftPacketsDown( *puiCurrPacketLenRV, FALSE)))
			{
				goto Exit;
			}

			// If a non-NULL puwBytesAvailableRV is passed in it means that
			// we are to return the number of bytes that we can actually
			// output. Since we know there is enough for the bytes needed, we
			// simply return the number of bytes that were requested.

			if (puiBytesAvailableRV)
			{
				*puiBytesAvailableRV = uiAdditionalBytesNeeded;
			}
		}
	}
	else	// (uiBytesNeeded > RFL_MAX_PACKET_SIZE)
	{

		// This is the case where the bytes needed would overflow the
		// maximum packet size. If puwBytesAvailableRV is NULL, it means
		// that all of the requested additional bytes must fit into the
		// packet. In that case, since the requested bytes would put us over
		// the packet size limit, we must finish the current packet and then
		// flush the packets out of the buffer so we can start a new packet.

		if (!puiBytesAvailableRV)
		{

			// Finish the current packet and start a new one.

			if (puiPacketCountRV)
			{
				(*puiPacketCountRV)++;
			}

			if (RC_BAD( rc = finishPacket( uiPacketType,
						  *puiCurrPacketLenRV - RFL_PACKET_OVERHEAD, FALSE)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = flush( m_pCurrentBuf)))
			{
				goto Exit;
			}

			*puiCurrPacketLenRV = RFL_PACKET_OVERHEAD;
		}
		else
		{

			// When puiBytesAvailableRV is non-NULL, it means we can fill up
			// the rest of the packet with part of the bytes. In this case we
			// return the number of bytes available and then shift the
			// packets down in the buffer to make sure there is room for a
			// full-size packet.

			*puiBytesAvailableRV = RFL_MAX_PACKET_SIZE -*puiCurrPacketLenRV;
			if (RC_BAD( rc = shiftPacketsDown( *puiCurrPacketLenRV, FALSE)))
			{
				goto Exit;
			}
		}
	}

Exit:

	return (rc);
}

/********************************************************************
Desc:		Log a chunk of data to the RFL log - typically used to log
			field data.  Will spill over into multiple packets if
			necessary.
*********************************************************************/
RCODE F_Rfl::logData(
	FLMUINT				uiDataLen,
	const FLMBYTE *	pucData,
	FLMUINT				uiPacketType,
	FLMUINT *			puiPacketLenRV,
	FLMUINT *			puiPacketCountRV,
	FLMUINT *			puiMaxLogBytesNeededRV,
	FLMUINT *			puiTotalBytesLoggedRV)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiBytesAvail;
	FLMBYTE *	pucDest;

	while (uiDataLen)
	{
		if (RC_BAD( rc = makeRoom( uiDataLen, puiPacketLenRV, uiPacketType,
					  &uiBytesAvail, puiPacketCountRV)))
		{
			goto Exit;
		}

		if (uiBytesAvail)
		{
			if (puiMaxLogBytesNeededRV)
			{
				if (RC_BAD( rc = RflCheckMaxLogged( puiMaxLogBytesNeededRV,
							  *puiPacketCountRV, puiTotalBytesLoggedRV, uiBytesAvail)))
				{
					goto Exit;
				}
			}

			pucDest = getPacketPtr() + (*puiPacketLenRV);
			f_memcpy( pucDest, pucData, uiBytesAvail);
			uiDataLen -= uiBytesAvail;
			pucData += uiBytesAvail;
			(*puiPacketLenRV) += uiBytesAvail;
		}

		// If we didn't get all of the data into the RFL buffer, finish and
		// flush the current packet.

		if (uiDataLen)
		{
			if (puiPacketCountRV)
			{
				(*puiPacketCountRV)++;
			}

			if (RC_BAD( rc = finishPacket( uiPacketType,
						  *puiPacketLenRV - RFL_PACKET_OVERHEAD, FALSE)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = flush( m_pCurrentBuf)))
			{
				goto Exit;
			}

			*puiPacketLenRV = RFL_PACKET_OVERHEAD;
			if (puiMaxLogBytesNeededRV)
			{
				if (RC_BAD( rc = RflCheckMaxLogged( puiMaxLogBytesNeededRV,
							  *puiPacketCountRV, puiTotalBytesLoggedRV,
							  RFL_PACKET_OVERHEAD)))
				{
					goto Exit;
				}
			}
		}
	}

Exit:

	return (rc);
}

/********************************************************************
Desc: Check to see if by logging the requested number of bytes we
		will end up exceeding the maximum bytes needed.  If so, and
		we have not yet actually logged a packet, return
		FERR_FAILURE so that we will discard this packet that is
		being built.  If we have already logged a packet, it is
		too late to discard what has been done.
*********************************************************************/
FSTATIC RCODE RflCheckMaxLogged(
	FLMUINT *	puiMaxBytesNeededRV,
	FLMUINT		uiPacketsLogged,
	FLMUINT *	puiCurrTotalLoggedRV,
	FLMUINT		uiBytesToLog)
{
	RCODE rc = FERR_OK;

	*puiCurrTotalLoggedRV += uiBytesToLog;

	if ((!uiPacketsLogged) && (*puiCurrTotalLoggedRV > *puiMaxBytesNeededRV))
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

Exit:

	return (rc);
}

/********************************************************************
Desc: Callback function that captures the changes being logged by
		the call to flmRecordDifference.
*********************************************************************/
FSTATIC void RflChangeCallback(
	GRD_DifferenceData&	DiffData,
	void *					CallbackData)
{
	RFL_CHANGE_DATA *	pRflChangeData = (RFL_CHANGE_DATA*) CallbackData;
	F_Rfl *				pRfl = pRflChangeData->pRfl;
	void *				pvField;
	const FLMBYTE *	pucExportPtr;
	FLMBYTE *			pucTmp;
	FLMUINT				uiOverhead = 0;
	FLMUINT				uiBytesToLog;
	FLMUINT				uiPos;
	FLMUINT				uiTagNum;
	FLMUINT				uiDataLen = 0;
	FLMBOOL				bEncrypted = FALSE;
	FLMUINT				uiEncId;
	FLMBYTE				ucChangeType = 0;

	// If we had an error before this callback, do nothing.

	if (RC_BAD( pRflChangeData->rc))
	{
		goto Exit;
	}

	switch (DiffData.type)
	{
		case GRD_Inserted:
		{
			bEncrypted = DiffData.pAfterRecord->isEncryptedField( DiffData.pvAfterField);
			uiDataLen = DiffData.pAfterRecord->getDataLength( DiffData.pvAfterField);
			if (pRflChangeData->uiVersionNum < FLM_FILE_FORMAT_VER_4_61)
			{
				if (bEncrypted)
				{
					uiOverhead = 13;
					ucChangeType = RFL_INSERT_ENC_FIELD;
				}
				else
				{
					uiOverhead = 9;
					ucChangeType = RFL_INSERT_FIELD;
				}
			}
			else
			{
				if (bEncrypted)
				{
					uiOverhead = 17;
					ucChangeType = RFL_INSERT_ENC_LARGE_FIELD;
				}
				else
				{
					uiOverhead = 11;
					ucChangeType = RFL_INSERT_LARGE_FIELD;
				}
			}
			break;
		}
			
		case GRD_Deleted:
		{

			// Ignore these for versions of the database >= 4.60

			if (pRflChangeData->uiVersionNum >= FLM_FILE_FORMAT_VER_4_60)
			{
				goto Exit;
			}

			uiOverhead = 3;
			break;
		}
		
		case GRD_DeletedSubtree:
		{

			// Ignore these for versions of the database < 4.60

			if (pRflChangeData->uiVersionNum < FLM_FILE_FORMAT_VER_4_60)
			{
				goto Exit;
			}

			uiOverhead = 3;
			break;
		}
		
		case GRD_Modified:
		{
			bEncrypted = DiffData.pAfterRecord->isEncryptedField( DiffData.pvAfterField);
			uiDataLen = DiffData.pAfterRecord->getDataLength( DiffData.pvAfterField);
			if (pRflChangeData->uiVersionNum < FLM_FILE_FORMAT_VER_4_61)
			{
				if (bEncrypted)
				{
					uiOverhead = 10;
					ucChangeType = RFL_MODIFY_ENC_FIELD;
				}
				else
				{
					uiOverhead = 6;
					ucChangeType = RFL_MODIFY_FIELD;
				}
			}
			else
			{
				if (bEncrypted)
				{
					uiOverhead = 14;
					ucChangeType = RFL_MODIFY_ENC_LARGE_FIELD;
				}
				else
				{
					uiOverhead = 8;
					ucChangeType = RFL_MODIFY_LARGE_FIELD;
				}
			}
			break;
		}
		
		default:
		{
			flmAssert( 0);
			break;
		}
	}

	// Determine the number of bytes that will actually be logged with
	// this overhead. If it won't fit in the current packet, we will have
	// to create a new packet - hence, we add RFL_PACKET_OVERHEAD to the
	// amount that will be logged.

	uiBytesToLog = uiOverhead;
	if (RFL_MAX_PACKET_SIZE - uiOverhead < pRflChangeData->uiCurrPacketLen)
	{
		uiBytesToLog += RFL_PACKET_OVERHEAD;
	}

	// See if the bytes we are going log will exceed the maximum bytes
	// needed.

	if (RC_BAD( pRflChangeData->rc = RflCheckMaxLogged(
					  &pRflChangeData->uiMaxLogBytesNeeded,
				  pRflChangeData->uiPacketCount,
				  &pRflChangeData->uiTotalBytesLogged, uiBytesToLog)))
	{
		goto Exit;
	}

	// Make room to log the overhead

	if (RC_BAD( pRflChangeData->rc = pRfl->makeRoom( uiOverhead,
				  &pRflChangeData->uiCurrPacketLen, RFL_CHANGE_FIELDS_PACKET, NULL,
				  &pRflChangeData->uiPacketCount)))
	{
		goto Exit;
	}

	pucTmp = pRfl->getPacketPtr() + pRflChangeData->uiCurrPacketLen;
	uiPos = DiffData.uiAbsolutePosition;
	UW2FBA( (FLMUINT16) uiPos, &pucTmp[1]);
	pRflChangeData->uiCurrPacketLen += uiOverhead;
	pvField = DiffData.pvAfterField;

	switch (DiffData.type)
	{
		case GRD_Inserted:
		{
			*pucTmp = ucChangeType;
			pucTmp += 3;
			uiTagNum = DiffData.pAfterRecord->getFieldID( pvField);
			UW2FBA( (FLMUINT16) uiTagNum, pucTmp);
			pucTmp += 2;
			*pucTmp++ = (FLMBYTE) DiffData.pAfterRecord->getDataType( pvField);
			*pucTmp++ = (FLMBYTE) DiffData.pAfterRecord->getLevel( pvField);
			if (pRflChangeData->uiVersionNum < FLM_FILE_FORMAT_VER_4_61)
			{
				UW2FBA( (FLMUINT16)uiDataLen, pucTmp);
				pucTmp += 2;
			}
			else
			{
				UD2FBA( (FLMUINT32)uiDataLen, pucTmp);
				pucTmp += 4;
			}

			if (bEncrypted)
			{
				uiEncId = DiffData.pAfterRecord->getEncryptionID( pvField);
				flmAssert( uiEncId);
				UW2FBA( (FLMUINT16) uiEncId, pucTmp);
				pucTmp += 2;

				uiDataLen = DiffData.pAfterRecord->getEncryptedDataLength( pvField);
				if (pRflChangeData->uiVersionNum < FLM_FILE_FORMAT_VER_4_61)
				{
					UW2FBA( (FLMUINT16)uiDataLen, pucTmp);
					pucTmp += 2;
				}
				else
				{
					UD2FBA( (FLMUINT32)uiDataLen, pucTmp);
					pucTmp += 4;
				}
			}

			// Log the data, if any.

			if (uiDataLen)
			{
				if (bEncrypted)
				{
					pucExportPtr = DiffData.pAfterRecord->getEncryptionDataPtr( pvField);
				}
				else
				{
					pucExportPtr = DiffData.pAfterRecord->getDataPtr( pvField);
				}

				if (!pucExportPtr)
				{
					pRflChangeData->rc = RC_SET( FERR_MEM);
					goto Exit;
				}

				if (RC_BAD( pRflChangeData->rc = pRfl->logData( uiDataLen,
							  pucExportPtr, RFL_CHANGE_FIELDS_PACKET,
							  &pRflChangeData->uiCurrPacketLen,
							  &pRflChangeData->uiPacketCount,
							  &pRflChangeData->uiMaxLogBytesNeeded,
							  &pRflChangeData->uiTotalBytesLogged)))
				{
					goto Exit;
				}
			}
			
			break;
		}
		
		case GRD_Deleted:
		case GRD_DeletedSubtree:
		{
			*pucTmp = RFL_DELETE_FIELD;
			break;
		}
		
		case GRD_Modified:
		{
			*pucTmp = ucChangeType;
			pucTmp += 3;

			// For now, just log the new bytes using RFL_REPLACE_BYTES option

			*pucTmp++ = RFL_REPLACE_BYTES;
			uiDataLen = DiffData.pAfterRecord->getDataLength( pvField);
			if (pRflChangeData->uiVersionNum < FLM_FILE_FORMAT_VER_4_61)
			{
				UW2FBA( (FLMUINT16)uiDataLen, pucTmp);
				pucTmp += 2;
			}
			else
			{
				UD2FBA( (FLMUINT32)uiDataLen, pucTmp);
				pucTmp += 4;
			}

			if (bEncrypted)
			{
				uiEncId = DiffData.pAfterRecord->getEncryptionID( pvField);
				flmAssert( uiEncId);
				UW2FBA( (FLMUINT16) uiEncId, pucTmp);
				pucTmp += 2;

				uiDataLen = DiffData.pAfterRecord->getEncryptedDataLength( pvField);
				if (pRflChangeData->uiVersionNum < FLM_FILE_FORMAT_VER_4_61)
				{
					UW2FBA( (FLMUINT16)uiDataLen, pucTmp);
					pucTmp += 2;
				}
				else
				{
					UD2FBA( (FLMUINT32)uiDataLen, pucTmp);
					pucTmp += 4;
				}
			}

			// Log the data, if any.

			if (uiDataLen)
			{
				if (bEncrypted)
				{
					pucExportPtr = DiffData.pAfterRecord->getEncryptionDataPtr( pvField);
				}
				else
				{
					pucExportPtr = DiffData.pAfterRecord->getDataPtr( pvField);
				}

				if (pucExportPtr == NULL)
				{
					pRflChangeData->rc = RC_SET( FERR_MEM);
					goto Exit;
				}

				if (RC_BAD( pRflChangeData->rc = pRfl->logData( uiDataLen,
							  pucExportPtr, RFL_CHANGE_FIELDS_PACKET,
							  &pRflChangeData->uiCurrPacketLen,
							  &pRflChangeData->uiPacketCount,
							  &pRflChangeData->uiMaxLogBytesNeeded,
							  &pRflChangeData->uiTotalBytesLogged)))
				{
					goto Exit;
				}
			}
			
			break;
		}
		
		default:
		{
			flmAssert( 0);
			break;
		}
	}

Exit:

	return;
}

/********************************************************************
Desc:		Log change fields for a record modify operation.
*********************************************************************/
RCODE F_Rfl::logChangeFields(
	FlmRecord *	pOldRecord,
	FlmRecord *	pNewRecord)
{
	RFL_CHANGE_DATA	RflChangeData;
	FLMUINT				uiTmpBodyLen;
	FLMUINT				uiDataLen;
	void *				pvNewField;
	FLMBOOL				bEncrypted;
	FLMUINT				uiOverhead;

	RflChangeData.rc = FERR_OK;
	RflChangeData.pRfl = this;
	RflChangeData.uiVersionNum = m_pFile->FileHdr.uiVersionNum;

	// Determine the total amount that would have to be logged if we just
	// logged the new record.

	RflChangeData.uiMaxLogBytesNeeded = RFL_PACKET_OVERHEAD;
	uiTmpBodyLen = 0;
	pvNewField = pNewRecord->root();
	for (; pvNewField; pvNewField = pNewRecord->next( pvNewField))
	{
		bEncrypted = pNewRecord->isEncryptedField( pvNewField);
		uiOverhead = (bEncrypted ? 10 : 6);
		if (uiTmpBodyLen + uiOverhead <= RFL_MAX_PACKET_BODY_SIZE)
		{
			uiTmpBodyLen += uiOverhead;
		}
		else
		{
			uiTmpBodyLen = uiOverhead;
			RflChangeData.uiMaxLogBytesNeeded += RFL_PACKET_OVERHEAD;
		}

		RflChangeData.uiMaxLogBytesNeeded += uiOverhead;
		if (bEncrypted)
		{
			uiDataLen = pNewRecord->getEncryptedDataLength( pvNewField);
		}
		else
		{
			uiDataLen = pNewRecord->getDataLength( pvNewField);
		}

		while (uiDataLen)
		{
			FLMUINT	uiTmp;

			uiTmp = RFL_MAX_PACKET_BODY_SIZE - uiTmpBodyLen;
			if (uiTmp >= uiDataLen)
			{
				uiTmp = uiDataLen;
				uiTmpBodyLen += uiDataLen;
			}
			else
			{
				uiTmpBodyLen = 0;
				RflChangeData.uiMaxLogBytesNeeded += RFL_PACKET_OVERHEAD;
			}

			RflChangeData.uiMaxLogBytesNeeded += uiTmp;
			uiDataLen -= uiTmp;
		}
	}

	// Account for terminating 0 at the end.

	if (uiTmpBodyLen + 2 > RFL_MAX_PACKET_BODY_SIZE)
	{
		RflChangeData.uiMaxLogBytesNeeded += RFL_PACKET_OVERHEAD;
	}

	RflChangeData.uiMaxLogBytesNeeded += 2;

	RflChangeData.uiPacketCount = 0;
	RflChangeData.uiTotalBytesLogged = RFL_PACKET_OVERHEAD;
	RflChangeData.uiCurrPacketLen = RFL_PACKET_OVERHEAD;

	if (!haveBuffSpace( RFL_PACKET_OVERHEAD))
	{
		if (RC_BAD( RflChangeData.rc = flush( m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	flmRecordDifference( pOldRecord, pNewRecord, RflChangeCallback,
							  (void *) &RflChangeData);

	// See if we exceeded the maximum log bytes. If so, just log the
	// changed record in its entirety.

	if (RC_BAD( RflChangeData.rc))
	{
		if (RflChangeData.rc == FERR_FAILURE)
		{
			RflChangeData.rc = logRecord( pNewRecord);
		}

		goto Exit;
	}
	else
	{
		FLMBYTE *		pucTmp;

		// Make room to log the 3 bytes of terminator

		if (RC_BAD( RflChangeData.rc = makeRoom( 3,
					  &RflChangeData.uiCurrPacketLen, RFL_CHANGE_FIELDS_PACKET, NULL,
					  &RflChangeData.uiPacketCount)))
		{
			if (RflChangeData.rc == FERR_FAILURE)
			{
				RflChangeData.rc = logRecord( pNewRecord);
			}

			goto Exit;
		}

		pucTmp = getPacketPtr() + RflChangeData.uiCurrPacketLen;
		*pucTmp++ = RFL_END_FIELD_CHANGES;
		UW2FBA( (FLMUINT16) 0, pucTmp);
		RflChangeData.uiCurrPacketLen += 3;

		if (RC_BAD( RflChangeData.rc = finishPacket( RFL_CHANGE_FIELDS_PACKET,
					  RflChangeData.uiCurrPacketLen - RFL_PACKET_OVERHEAD, FALSE)))
		{
			goto Exit;
		}
	}

Exit:

	return (RflChangeData.rc);
}

/********************************************************************
Desc:		Log a record for the record add or modify operations.
*********************************************************************/
RCODE F_Rfl::logRecord(
	FlmRecord *	pRecord)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiPacketLen = RFL_PACKET_OVERHEAD;
	void *		pvField;
	FLMBYTE *	pucTmp;
	FLMUINT		uiTagNum;
	FLMUINT		uiDataLen;
	FLMBOOL		bEncrypted;
	FLMUINT		uiEncId;
	FLMUINT		uiPacketType;
	FLMUINT		uiOverhead;

	if (!haveBuffSpace( RFL_PACKET_OVERHEAD))
	{
		if (RC_BAD( rc = flush( m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	if (m_pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_60)
	{
		uiPacketType = RFL_DATA_RECORD_PACKET;
	}
	else if (m_pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_61)
	{
		uiPacketType = RFL_ENC_DATA_RECORD_PACKET;
	}
	else
	{
		uiPacketType = RFL_DATA_RECORD_PACKET_VER_3;
	}

	pvField = pRecord->root();
	for (; pvField; pvField = pRecord->next( pvField))
	{
		if (uiPacketType == RFL_DATA_RECORD_PACKET)
		{
			bEncrypted = FALSE;
			uiOverhead = 6;
		}
		else if (uiPacketType == RFL_ENC_DATA_RECORD_PACKET)
		{
			bEncrypted = pRecord->isEncryptedField( pvField);
			uiOverhead = (bEncrypted ? 11 : 7);
		}
		else
		{
			bEncrypted = pRecord->isEncryptedField( pvField);
			uiOverhead = (bEncrypted ? 15 : 9);
		}

		if (RC_BAD( rc = makeRoom( uiOverhead, &uiPacketLen, uiPacketType, NULL,
					  NULL)))
		{
			goto Exit;
		}

		pucTmp = getPacketPtr() + uiPacketLen;
		uiPacketLen += uiOverhead;

		uiTagNum = pRecord->getFieldID( pvField);
		UW2FBA( (FLMUINT16) uiTagNum, pucTmp);
		pucTmp += 2;
		*pucTmp++ = (FLMBYTE) pRecord->getDataType( pvField);
		*pucTmp++ = (FLMBYTE) pRecord->getLevel( pvField);
		uiDataLen = pRecord->getDataLength( pvField);
		if (uiPacketType != RFL_DATA_RECORD_PACKET_VER_3)
		{
			UW2FBA( (FLMUINT16) uiDataLen, pucTmp);
			pucTmp += 2;
		}
		else
		{
			UD2FBA( (FLMUINT32) uiDataLen, pucTmp);
			pucTmp += 4;
		}

		// Record if this field is encrypted. If it is, then there will be
		// more data to follow.

		if (uiPacketType != RFL_DATA_RECORD_PACKET)
		{
			*pucTmp = (bEncrypted ? (FLMBYTE) 1 : (FLMBYTE) 0);
			pucTmp++;

			// Check for encrypted field and add the results.

			if (bEncrypted)
			{
				uiEncId = pRecord->getEncryptionID( pvField);
				flmAssert( uiEncId);
				UW2FBA( (FLMUINT16) uiEncId, pucTmp);
				pucTmp += 2;

				uiDataLen = pRecord->getEncryptedDataLength( pvField);
				if (uiPacketType == RFL_DATA_RECORD_PACKET_VER_3)
				{
					UD2FBA( (FLMUINT32)uiDataLen, pucTmp);
					pucTmp += 4;
				}
				else
				{
					UW2FBA( (FLMUINT16)uiDataLen, pucTmp);
					pucTmp += 2;
				}
			}
		}

		// Log the data, if any.

		if (uiDataLen)
		{
			const FLMBYTE *		pucExportPtr;

			if (bEncrypted)
			{
				pucExportPtr = pRecord->getEncryptionDataPtr( pvField);
			}
			else
			{
				pucExportPtr = pRecord->getDataPtr( pvField);
			}

			if (pucExportPtr == NULL)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}

			if (RC_BAD( rc = logData( uiDataLen, pucExportPtr, uiPacketType,
						  &uiPacketLen, NULL, NULL, NULL)))
			{
				goto Exit;
			}
		}
	}

	// Add null to terminate the record.

	if (RC_BAD( rc = makeRoom( 2, &uiPacketLen, uiPacketType, NULL, NULL)))
	{
		goto Exit;
	}

	pucTmp = getPacketPtr() + uiPacketLen;
	uiPacketLen += 2;
	UW2FBA( 0, pucTmp);
	pucTmp += 2;

	// Finish the packet.

	if (RC_BAD( rc = finishPacket( uiPacketType, 
		uiPacketLen - RFL_PACKET_OVERHEAD, FALSE)))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/********************************************************************
Desc:		Log record add, modify, or delete operation
*********************************************************************/
RCODE F_Rfl::logUpdate(
	FLMUINT		uiContainer,
	FLMUINT		uiDrn,
	FLMUINT		uiAutoTrans,
	FlmRecord *	pOldRecord,
	FlmRecord *	pNewRecord)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiPacketType;

	// Do nothing if logging is disabled.

	if (m_bLoggingOff)
	{
		goto Exit;
	}

	// Better not be in the middle of logging unknown stuff for the
	// application

	flmAssert( !m_bLoggingUnknown);

	// Better be in the middle of a transaction.

	flmAssert( m_uiCurrTransID);

	if (pOldRecord && pNewRecord)
	{
		if (m_pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_60)
		{
			uiPacketType = RFL_MODIFY_RECORD_PACKET_VER_2;
		}
		else
		{
			uiPacketType = RFL_MODIFY_RECORD_PACKET;
		}
	}
	else if (pNewRecord)
	{
		if (m_pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_60)
		{
			uiPacketType = RFL_ADD_RECORD_PACKET_VER_2;
		}
		else
		{
			uiPacketType = RFL_ADD_RECORD_PACKET;
		}
	}
	else
	{
		if (m_pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_60)
		{
			uiPacketType = RFL_DELETE_RECORD_PACKET_VER_2;
		}
		else
		{
			uiPacketType = RFL_DELETE_RECORD_PACKET;
		}
	}

	if (RC_BAD( rc = logUpdatePacket( uiPacketType, uiContainer, uiDrn,
				  uiAutoTrans)))
	{
		goto Exit;
	}

	// If it is a record modify, log the change fields. If it is a record
	// add, log the new record.

	if (pOldRecord && pNewRecord)
	{
		if (RC_BAD( rc = logChangeFields( pOldRecord, pNewRecord)))
		{
			goto Exit;
		}
	}
	else if (pNewRecord)
	{
		if (RC_BAD( rc = logRecord( pNewRecord)))
		{
			goto Exit;
		}
	}

Exit:

	return (rc);
}

/********************************************************************
Desc:		Log a set of records that is indexed for a specific index.
*********************************************************************/
RCODE F_Rfl::logIndexSet(
	FLMUINT	uiIndex,
	FLMUINT	uiContainerNum,
	FLMUINT	uiStartDrn,
	FLMUINT	uiEndDrn)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pucPacketBody;
	FLMUINT		uiPacketBodyLen;

	// This call is a new database version. Database better have been
	// upgraded.

	flmAssert( m_pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_3_02);

	// Do nothing if logging is disabled.

	if (m_bLoggingOff)
	{
		goto Exit;
	}

	// Better not be in the middle of logging unknown stuff for the
	// application

	flmAssert( !m_bLoggingUnknown);

	// Better be in the middle of a transaction.

	flmAssert( m_uiCurrTransID);

	m_uiOperCount++;
	uiPacketBodyLen = 
		(FLMUINT)((m_pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_50) 
											? (FLMUINT) 16 
											: (FLMUINT) 14);

	// Make sure we have space in the RFL buffer for a complete packet.

	if (!haveBuffSpace( uiPacketBodyLen + RFL_PACKET_OVERHEAD))
	{
		if (RC_BAD( rc = flush( m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	// Get a pointer to where we will be laying down the packet body.

	pucPacketBody = getPacketBodyPtr();

	// Output the transaction ID.

	UD2FBA( (FLMUINT32) m_uiCurrTransID, pucPacketBody);
	pucPacketBody += 4;

	// Output the container number, if db version is >= 4.50

	if (m_pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_50)
	{
		UW2FBA( (FLMUINT16) uiContainerNum, pucPacketBody);
		pucPacketBody += 2;
	}

	// Output the index number.

	UW2FBA( (FLMUINT16) uiIndex, pucPacketBody);
	pucPacketBody += 2;

	// Output the starting DRN.

	UD2FBA( (FLMUINT32) (uiStartDrn), pucPacketBody);
	pucPacketBody += 4;

	// Output the ending DRN.

	UD2FBA( (FLMUINT32) (uiEndDrn), pucPacketBody);
	pucPacketBody += 4;

	// Finish the packet

	if (RC_BAD( rc = finishPacket( (FLMUINT) (
					  (m_pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_50) 
								? (FLMUINT) RFL_INDEX_SET_PACKET_VER_2
								: (FLMUINT) RFL_INDEX_SET_PACKET), 
					  uiPacketBodyLen, FALSE)))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/********************************************************************
Desc:		Start logging unknown packets.
*********************************************************************/
RCODE F_Rfl::startLoggingUnknown(void)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pucPacketBody;
	FLMUINT		uiPacketBodyLen;

	flmAssert( m_pFile);

	// Do nothing if logging is disabled. Also, ignore these packets if we
	// are operating on a pre-4.3 database.

	if (m_bLoggingOff || m_pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_3)
	{
		goto Exit;
	}

	// Better not already be in the middle of logging unknown stuff for
	// the application

	flmAssert( !m_bLoggingUnknown);

	// Better be inside a transaction.

	flmAssert( m_uiCurrTransID);

	m_uiOperCount++;
	uiPacketBodyLen = 4;

	// Make sure we have space in the RFL buffer for a complete packet.

	if (!haveBuffSpace( uiPacketBodyLen + RFL_PACKET_OVERHEAD))
	{
		if (RC_BAD( rc = flush( m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	// Get a pointer to where we will be laying down the packet body.

	pucPacketBody = getPacketBodyPtr();

	// Output the transaction ID.

	UD2FBA( (FLMUINT32) m_uiCurrTransID, pucPacketBody);
	pucPacketBody += 4;

	// Finish the packet

	if (RC_BAD( rc = finishPacket( RFL_START_UNKNOWN_PACKET, uiPacketBodyLen,
				  FALSE)))
	{
		goto Exit;
	}

	m_bLoggingUnknown = TRUE;
	m_uiUnknownPacketLen = RFL_PACKET_OVERHEAD;
	
Exit:

	return (rc);
}

/********************************************************************
Desc:		Log unknown data.
*********************************************************************/
RCODE F_Rfl::logUnknown(
	FLMBYTE *	pucUnknown,
	FLMUINT		uiLen)
{
	RCODE rc = FERR_OK;

	// Do nothing if logging is disabled. Also, ignore these packets if we
	// are operating on a pre-4.3 database.

	if (m_bLoggingOff || m_pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_3)
	{
		goto Exit;
	}

	flmAssert( m_bLoggingUnknown);
	if (RC_BAD( rc = logData( uiLen, pucUnknown, RFL_UNKNOWN_PACKET,
				  &m_uiUnknownPacketLen, NULL, NULL, NULL)))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/********************************************************************
Desc:		End logging unknown packets.
*********************************************************************/
RCODE F_Rfl::endLoggingUnknown(void)
{
	RCODE rc = FERR_OK;

	flmAssert( m_pFile);

	// Do nothing if logging is disabled. Also, ignore these packets if we
	// are operating on a pre-4.3 database.

	if (m_bLoggingOff || m_pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_3)
	{
		goto Exit;
	}

	// Better be in the middle of logging unknown stuff for the application

	flmAssert( m_bLoggingUnknown);
	if (m_uiUnknownPacketLen > RFL_PACKET_OVERHEAD)
	{
		if (RC_BAD( rc = finishPacket( RFL_UNKNOWN_PACKET,
					  m_uiUnknownPacketLen - RFL_PACKET_OVERHEAD, FALSE)))
		{
			goto Exit;
		}
	}

Exit:

	m_bLoggingUnknown = FALSE;
	m_uiUnknownPacketLen = RFL_PACKET_OVERHEAD;
	return (rc);
}

/********************************************************************
Desc:		Log a reduce packet
*********************************************************************/
RCODE F_Rfl::logReduce(
	FLMUINT	uiTransID,
	FLMUINT	uiCount)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pucPacketBody;
	FLMUINT		uiPacketBodyLen;

	// This call is new with 4.3 databases - not supported in older
	// versions, so don't log it.

	if (m_pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_3)
	{
		goto Exit;
	}

	// Do nothing if logging is disabled.

	if (m_bLoggingOff)
	{
		goto Exit;
	}

	// Better not be in the middle of logging unknown stuff for the
	// application

	flmAssert( !m_bLoggingUnknown);

	// We need to set up to log this packet as if we were logging a
	// transaction. The only difference is that we don't log the begin
	// transaction packet.

	if (RC_BAD( rc = setupTransaction()))
	{
		goto Exit;
	}

	uiPacketBodyLen = 8;

	// Make sure we have space in the RFL buffer for a complete packet.

	if (!haveBuffSpace( uiPacketBodyLen + RFL_PACKET_OVERHEAD))
	{
		if (RC_BAD( rc = flush( m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	// Get a pointer to where we will be laying down the packet body.

	pucPacketBody = getPacketBodyPtr();

	// Output the transaction ID.

	UD2FBA( (FLMUINT32) uiTransID, pucPacketBody);
	pucPacketBody += 4;

	// Output the count

	UD2FBA( (FLMUINT32) uiCount, pucPacketBody);
	pucPacketBody += 4;

	// Finish the packet

	if (RC_BAD( rc = finishPacket( RFL_REDUCE_PACKET, uiPacketBodyLen, TRUE)))
	{
		goto Exit;
	}

	// Finalize the transaction (as if we were committing a transaction)

	finalizeTransaction();

Exit:

	return (rc);
}

/********************************************************************
Desc:		Log a database conversion packet
Note:		This routine performs most of the setup for logging a full
			transaction, but it does not cause begin and commit packets
			to be logged.  It is a "standalone" transaction.
*********************************************************************/
RCODE F_Rfl::logUpgrade(
	FLMUINT		uiTransID,
	FLMUINT		uiOldVersion,
	FLMBYTE *	pucDBKey,
	FLMUINT32	ui32DBKeyLen)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pucPacketBody;
	FLMUINT		uiPacketBodyLen;

	// Do nothing if logging is disabled.

	if (m_bLoggingOff)
	{
		goto Exit;
	}

	// Better not be in the middle of logging unknown stuff for the
	// application

	flmAssert( !m_bLoggingUnknown);

	// We need to set up to log this packet as if we were logging a
	// transaction. The only difference is that we don't log the begin
	// transaction packet.

	if (RC_BAD( rc = setupTransaction()))
	{
		goto Exit;
	}

	uiPacketBodyLen = 14 + ui32DBKeyLen;

	// Make sure we have space in the RFL buffer for a complete packet.

	if (!haveBuffSpace( uiPacketBodyLen + RFL_PACKET_OVERHEAD))
	{
		if (RC_BAD( rc = flush( m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	// Get a pointer to where we will be laying down the packet body.

	pucPacketBody = getPacketBodyPtr();

	// Output the transaction ID

	UD2FBA( (FLMUINT32) uiTransID, pucPacketBody);
	pucPacketBody += 4;

	// Output the old database version

	UD2FBA( (FLMUINT32) uiOldVersion, pucPacketBody);
	pucPacketBody += 4;

	// Output the new database version

	UD2FBA( (FLMUINT32) FLM_CUR_FILE_FORMAT_VER_NUM, pucPacketBody);
	pucPacketBody += 4;

	// For versions >= 4.60, the next two bytes will give the length of
	// the DB Key.

	flmAssert( ui32DBKeyLen <= 0xFFFF);
	UW2FBA( (FLMUINT16) ui32DBKeyLen, pucPacketBody);
	pucPacketBody += 2;

	// If we were built without encryption, the key length will be zero,
	// so no need to store the key.

	if (ui32DBKeyLen)
	{
		f_memcpy( pucPacketBody, pucDBKey, ui32DBKeyLen);
		pucPacketBody += ui32DBKeyLen;
	}

	// Finish the packet

	if (RC_BAD( rc = finishPacket( RFL_UPGRADE_PACKET, uiPacketBodyLen, TRUE)))
	{
		goto Exit;
	}

	// Finalize the transaction (as if we were committing a transaction)

	finalizeTransaction();

Exit:

	if (!m_bLoggingOff)
	{
		m_uiCurrTransID = 0;
	}

	return (rc);
}

/********************************************************************
Desc:		Log the wrapped database key
*********************************************************************/
RCODE F_Rfl::logWrappedKey(
	FLMUINT		uiTransID,
	FLMBYTE *	pucDBKey,
	FLMUINT32	ui32DBKeyLen)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pucPacketBody;
	FLMUINT		uiPacketBodyLen;

	// Do nothing if logging is disabled.

	if (m_bLoggingOff)
	{
		goto Exit;
	}

	// Better not be in the middle of logging unknown stuff for the
	// application

	flmAssert( !m_bLoggingUnknown);

	if (RC_BAD( rc = setupTransaction()))
	{
		goto Exit;
	}

	uiPacketBodyLen = 6 + ui32DBKeyLen;

	// Make sure we have space in the RFL buffer for a complete packet.

	if (!haveBuffSpace( uiPacketBodyLen + RFL_PACKET_OVERHEAD))
	{
		if (RC_BAD( rc = flush( m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	// Get a pointer to where we will be laying down the packet body.

	pucPacketBody = getPacketBodyPtr();

	// Output the transaction ID

	UD2FBA( (FLMUINT32) uiTransID, pucPacketBody);
	pucPacketBody += 4;

	// The next two bytes will give the length of the DB Key.

	flmAssert( ui32DBKeyLen <= 0xFFFF);
	UW2FBA( (FLMUINT16) ui32DBKeyLen, pucPacketBody);
	pucPacketBody += 2;

	// If we were built without encryption, the key length will be zero,
	// so no need to store the key.

	if (ui32DBKeyLen)
	{
		f_memcpy( pucPacketBody, pucDBKey, ui32DBKeyLen);
		pucPacketBody += ui32DBKeyLen;
	}

	// Finish the packet

	if (RC_BAD( rc = finishPacket( RFL_WRAP_KEY_PACKET, uiPacketBodyLen, TRUE)))
	{
		goto Exit;
	}

	finalizeTransaction();

Exit:

	return (rc);
}

/********************************************************************
Desc:		Log that we have enabled encryption
*********************************************************************/
RCODE F_Rfl::logEnableEncryption(
	FLMUINT		uiTransID,
	FLMBYTE *	pucDBKey,
	FLMUINT32	ui32DBKeyLen)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pucPacketBody;
	FLMUINT		uiPacketBodyLen;

	// Do nothing if logging is disabled.

	if (m_bLoggingOff)
	{
		goto Exit;
	}

	// Better not be in the middle of logging unknown stuff for the
	// application

	flmAssert( !m_bLoggingUnknown);

	if (RC_BAD( rc = setupTransaction()))
	{
		goto Exit;
	}

	uiPacketBodyLen = 6 + ui32DBKeyLen;

	// Make sure we have space in the RFL buffer for a complete packet.

	if (!haveBuffSpace( uiPacketBodyLen + RFL_PACKET_OVERHEAD))
	{
		if (RC_BAD( rc = flush( m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	// Get a pointer to where we will be laying down the packet body.

	pucPacketBody = getPacketBodyPtr();

	// Output the transaction ID

	UD2FBA( (FLMUINT32) uiTransID, pucPacketBody);
	pucPacketBody += 4;

	// The next two bytes will give the length of the DB Key.

	flmAssert( ui32DBKeyLen <= 0xFFFF);
	UW2FBA( (FLMUINT16) ui32DBKeyLen, pucPacketBody);
	pucPacketBody += 2;

	// If we were built without encryption, the key length will be zero,
	// so no need to store the key.

	if (ui32DBKeyLen)
	{
		f_memcpy( pucPacketBody, pucDBKey, ui32DBKeyLen);
		pucPacketBody += ui32DBKeyLen;
	}

	// Finish the packet

	if (RC_BAD( rc = finishPacket( RFL_ENABLE_ENCRYPTION_PACKET, uiPacketBodyLen,
				  TRUE)))
	{
		goto Exit;
	}

	finalizeTransaction();

Exit:

	return (rc);
}

/********************************************************************
Desc:		Log a block chain free operation
*********************************************************************/
RCODE F_Rfl::logBlockChainFree(
	FLMUINT	uiTrackerDrn,
	FLMUINT	uiCount,
	FLMUINT	uiEndAddr)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pucPacketBody;
	FLMUINT		uiPacketBodyLen;

	// This call is new with 4.52 databases - not supported in older
	// versions, so don't log it.

	if (m_pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_60)
	{
		flmAssert( 0);
		goto Exit;
	}

	// Do nothing if logging is disabled.

	if (m_bLoggingOff)
	{
		goto Exit;
	}

	// Better not be in the middle of logging unknown stuff for the
	// application

	flmAssert( !m_bLoggingUnknown);

	// Better be in the middle of a transaction.

	flmAssert( m_uiCurrTransID);
	m_uiOperCount++;

	uiPacketBodyLen = 16;

	// Make sure we have space in the RFL buffer for a complete packet.

	if (!haveBuffSpace( uiPacketBodyLen + RFL_PACKET_OVERHEAD))
	{
		if (RC_BAD( rc = flush( m_pCurrentBuf)))
		{
			goto Exit;
		}
	}

	// Get a pointer to where we will be laying down the packet body.

	pucPacketBody = getPacketBodyPtr();

	// Output the transaction ID.

	UD2FBA( (FLMUINT32) m_uiCurrTransID, pucPacketBody);
	pucPacketBody += 4;

	// Output the tracker record number

	UD2FBA( (FLMUINT32) uiTrackerDrn, pucPacketBody);
	pucPacketBody += 4;

	// Output the count

	UD2FBA( (FLMUINT32) uiCount, pucPacketBody);
	pucPacketBody += 4;

	// Output the ending block address

	UD2FBA( (FLMUINT32) uiEndAddr, pucPacketBody);
	pucPacketBody += 4;

	// Finish the packet

	if (RC_BAD( rc = finishPacket( RFL_BLK_CHAIN_FREE_PACKET, uiPacketBodyLen,
				  TRUE)))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/********************************************************************
Desc:		Reads a full packet, based on what file offset and read
			offset are currently set to.
*********************************************************************/
RCODE F_Rfl::readPacket(
	FLMUINT	uiMinBytesNeeded)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiTmpOffset;
	FLMUINT	uiReadLen;
	FLMUINT	uiBytesRead;

	// If we have enough bytes in the buffer for the minimum bytes needed,
	// we don't need to retrieve any more bytes.

	if (m_pCurrentBuf->uiRflBufBytes - m_uiRflReadOffset >= uiMinBytesNeeded)
	{
		goto Exit;
	}

	// If we are doing restore, we have to do only sequential reads -
	// cannot depend on doing reads on 512 byte boundaries. Otherwise, we
	// read directly from disk on 512 byte boundaries.

	if (m_pRestore)
	{
		FLMUINT	uiCurrFilePos = m_pCurrentBuf->uiRflFileOffset +
			m_pCurrentBuf->uiRflBufBytes;

		if (m_uiRflReadOffset > 0)
		{

			// Move the bytes left in the buffer down to the beginning of
			// the buffer.

			f_memmove( m_pCurrentBuf->pIOBuffer->getBufferPtr(),
						 &(m_pCurrentBuf->pIOBuffer->getBufferPtr()[m_uiRflReadOffset]),
							 m_pCurrentBuf->uiRflBufBytes - m_uiRflReadOffset);
			m_pCurrentBuf->uiRflBufBytes -= m_uiRflReadOffset;
			m_pCurrentBuf->uiRflFileOffset += m_uiRflReadOffset;
			m_uiRflReadOffset = 0;
		}

		uiReadLen = m_uiBufferSize - m_pCurrentBuf->uiRflBufBytes;

		// Read enough to fill the rest of the buffer, which is guaranteed
		// to hold at least one full packet.

		if (!m_uiFileEOF)
		{
			if (uiCurrFilePos > (FLMUINT) (-1) - uiReadLen)
			{
				uiReadLen = (FLMUINT) (-1) - uiCurrFilePos;
			}
		}
		else
		{
			if (uiCurrFilePos + uiReadLen > m_uiFileEOF)
			{
				uiReadLen = m_uiFileEOF - uiCurrFilePos;
			}
		}

		// If reading will not give us the minimum bytes needed, we cannot
		// satisfy this request from the current file.

		if (uiReadLen + m_pCurrentBuf->uiRflBufBytes < uiMinBytesNeeded)
		{
			rc = RC_SET( FERR_BAD_RFL_PACKET);
			goto Exit;
		}

		// Read enough to get the entire packet.

		if (RC_BAD( rc = m_pRestore->read( uiReadLen, &(
						  m_pCurrentBuf->pIOBuffer->getBufferPtr()[
						  m_pCurrentBuf->uiRflBufBytes]), &uiBytesRead)))
		{
			if (rc == FERR_IO_END_OF_FILE)
			{
				rc = FERR_OK;
			}
			else
			{
				goto Exit;
			}
		}

		// If we didn't read enough to satisfy the minimum bytes needed, we
		// cannot satisfy this request from the current file.

		if (uiBytesRead + m_pCurrentBuf->uiRflBufBytes < uiMinBytesNeeded)
		{
			rc = RC_SET( FERR_BAD_RFL_PACKET);
			goto Exit;
		}

		m_pCurrentBuf->uiRflBufBytes += uiBytesRead;
	}
	else
	{

		// Set offsets so we are on a 512 byte boundary for our next read.
		// No need to move data, since we will be re-reading it anyway.

		if (m_uiRflReadOffset > 0)
		{
			uiTmpOffset = m_uiRflReadOffset - (m_uiRflReadOffset & 511);
			m_pCurrentBuf->uiRflFileOffset += uiTmpOffset;
			m_uiRflReadOffset -= uiTmpOffset;
		}
		else if (m_pCurrentBuf->uiRflFileOffset & 511)
		{
			m_uiRflReadOffset = m_pCurrentBuf->uiRflFileOffset & 511;
			m_pCurrentBuf->uiRflFileOffset -= m_uiRflReadOffset;
		}

		m_pCurrentBuf->uiRflBufBytes = 0;

		// Read enough to fill the rest of the buffer, which is guaranteed
		// to hold at least one full packet.

		uiReadLen = m_uiBufferSize;

		// m_uiFileEOF better not be zero at this point - we should always
		// know precisely where the RFL file ends when we are doing recovery
		// as opposed to doing a restore.

		flmAssert( m_uiFileEOF >= 512);
		if (m_pCurrentBuf->uiRflFileOffset + uiReadLen > m_uiFileEOF)
		{
			uiReadLen = m_uiFileEOF - m_pCurrentBuf->uiRflFileOffset;
		}

		// If reading will not give us the minimum number of bytes needed,
		// we have a bad packet.

		if (uiReadLen < m_uiRflReadOffset ||
			 uiReadLen - m_uiRflReadOffset < uiMinBytesNeeded)
		{
			rc = RC_SET( FERR_BAD_RFL_PACKET);
			goto Exit;
		}

		// Read to get the entire packet.

		if (RC_BAD( rc = m_pFileHdl->read( m_pCurrentBuf->uiRflFileOffset,
					  uiReadLen, m_pCurrentBuf->pIOBuffer->getBufferPtr(), 
					  &uiBytesRead)))
		{
			if (rc == FERR_IO_END_OF_FILE)
			{
				rc = FERR_OK;
			}
			else
			{
				m_bRflVolumeOk = FALSE;
				goto Exit;
			}
		}

		if (uiBytesRead < uiReadLen)
		{
			rc = RC_SET( FERR_BAD_RFL_PACKET);
			goto Exit;
		}

		m_pCurrentBuf->uiRflBufBytes = uiReadLen;
	}

Exit:

	return (rc);
}

/********************************************************************
Desc:		Gets and verifies the next packet from the roll-forward
			log file.  Packet checksum will be verified.
*********************************************************************/
RCODE F_Rfl::getPacket(
	FLMBOOL		bForceNextFile,
	FLMUINT *	puiPacketTypeRV,
	FLMBYTE **	ppucPacketBodyRV,
	FLMUINT *	puiPacketBodyLenRV,
	FLMBOOL *	pbLoggedTimes)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pucPacket;
	FLMBYTE *	pucPacketBody;
	FLMUINT		uiPacketType;
	FLMUINT		uiEncryptPacketBodyLen;
	FLMBYTE		ucHdr[512];
	FLMUINT		uiBytesRead;

	// See if we need to go to the next file. Note that we only check for
	// this exactly on packet boundaries. We do not expect packets to be
	// split across files. If we are not at the end of processing what is
	// in the buffer, we should be able to read the rest of the packet from
	// the current file.

Get_Next_File:

	if (bForceNextFile || 
		 (m_uiFileEOF && m_uiRflReadOffset == m_pCurrentBuf->uiRflBufBytes &&
		  m_pCurrentBuf->uiRflFileOffset + m_pCurrentBuf->uiRflBufBytes ==
						m_uiFileEOF))
	{
		if (m_bKeepRflFiles)
		{
			if (!m_pRestore)
			{

				// Only doing recovery after a failure, see if we are at the
				// last file already.

				if (m_pCurrentBuf->uiCurrFileNum == m_uiLastRecoverFileNum)
				{
					rc = RC_SET( FERR_END);
					goto Exit;
				}
				else if( (m_pCurrentBuf->uiCurrFileNum + 1 ) ==
								m_uiLastRecoverFileNum &&
					!(FLMUINT)FB2UD( &m_pFile->ucLastCommittedLogHdr[ 
						LOG_RFL_LAST_TRANS_OFFSET]))
				{

					// We are going to try to open the last file. Since the
					// log header shows a current offset of 0, the file may
					// have been created but nothing was logged to it. We don't
					// want to try to open it here because it may not have been
					// initialized fully at the time of the server crash.

					m_pCurrentBuf->uiCurrFileNum = m_uiLastRecoverFileNum;
					rc = RC_SET( FERR_END);
					goto Exit;
				}

				// Open the next file in the sequence.

				if (RC_BAD( rc = openFile( m_pCurrentBuf->uiCurrFileNum + 1,
							  m_ucNextSerialNum)))
				{
					if (rc == FERR_IO_PATH_NOT_FOUND || rc == FERR_IO_INVALID_PATH)
					{
						rc = RC_SET( FERR_END);
					}

					goto Exit;
				}

				// If this is the last RFL file, the EOF is contained in the
				// log header. Otherwise, it will be in the RFL file's header,
				// and openFile will already have retrieved it.

				if (m_pCurrentBuf->uiCurrFileNum == m_uiLastRecoverFileNum)
				{
					m_uiFileEOF = (FLMUINT) FB2UD( &m_pFile->ucLastCommittedLogHdr[
														LOG_RFL_LAST_TRANS_OFFSET]);

					// Could be zero if RFL file wasn't created yet.

					if (!m_uiFileEOF)
					{
						m_uiFileEOF = 512;
					}
				}

				// By this point, the EOF better be greater than or equal to
				// 512.

				flmAssert( m_uiFileEOF >= 512);
			}
			else
			{
				if (RC_BAD( rc = m_pRestore->close()))
				{
					goto Exit;
				}

				// Ask the recovery object to open the file.

				if (RC_BAD( rc = m_pRestore->openRflFile(
								  m_pCurrentBuf->uiCurrFileNum + 1)))
				{
					if (rc == FERR_IO_PATH_NOT_FOUND)
					{
						rc = RC_SET( FERR_END);
					}

					goto Exit;
				}

				// Get the first 512 bytes from the file and verify the
				// header.

				if (RC_BAD( rc = m_pRestore->read( 512, ucHdr, &uiBytesRead)))
				{
					goto Exit;
				}

				if (uiBytesRead < 512)
				{
					rc = RC_SET_AND_ASSERT( FERR_NOT_RFL);
					goto Exit;
				}

				if (RC_BAD( rc = verifyHeader( ucHdr,
							  m_pCurrentBuf->uiCurrFileNum + 1, m_ucNextSerialNum)))
				{
					goto Exit;
				}

				// We may not know the actual EOF of files during restore
				// operations. m_uiFileEOF could be zero here.

				m_uiFileEOF = (FLMUINT) FB2UD( &ucHdr[RFL_EOF_POS]);

				// File EOF may be zero or >= 512 at this point.

				flmAssert( !m_uiFileEOF || m_uiFileEOF >= 512);

				// Need to increment current file number.

				m_pCurrentBuf->uiCurrFileNum++;
			}

			m_pCurrentBuf->uiRflFileOffset = 512;
			m_uiRflReadOffset = 0;
			m_pCurrentBuf->uiRflBufBytes = 0;

			// Get the next packet from the new file.

			if (RC_BAD( rc = readPacket( RFL_PACKET_OVERHEAD)))
			{
				if (m_uiFileEOF == 512 && m_bKeepRflFiles)
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

			// This is the case where we are not keeping the RFL files. So,
			// there is no next file to go to. If we get to this point, we
			// had better not be doing a restore.

			flmAssert( m_pRestore == NULL && !bForceNextFile);
			rc = RC_SET( FERR_END);
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
	if ((FLMUINT) FB2UD( &pucPacket[RFL_PACKET_ADDRESS_OFFSET]) != m_uiPacketAddress)
	{
		rc = RC_SET( FERR_BAD_RFL_PACKET);
		goto Exit;
	}

	// Get packet type, time flag, and packet body length

	*puiPacketTypeRV = uiPacketType = RFL_GET_PACKET_TYPE( 
														pucPacket[RFL_PACKET_TYPE_OFFSET]);

	if (pbLoggedTimes)
	{
		*pbLoggedTimes = (pucPacket[RFL_PACKET_TYPE_OFFSET] & RFL_TIME_LOGGED_FLAG) 
											? TRUE 
											: FALSE;
	}

	*puiPacketBodyLenRV = (FLMUINT) FB2UW( 
									&pucPacket[RFL_PACKET_BODY_LENGTH_OFFSET]);

	// Adjust the packet body length for encryption if necessary. NOTE:
	// This adjusted length is NOT returned to the caller. The actual body
	// length is what will be returned.

	uiEncryptPacketBodyLen = getEncryptPacketBodyLen( uiPacketType,
																	 *puiPacketBodyLenRV);

	// Make sure we have the entire packet in the buffer.

	if (RC_BAD( rc = readPacket( uiEncryptPacketBodyLen + RFL_PACKET_OVERHEAD)))
	{
		goto Exit;
	}

	pucPacket = &(m_pCurrentBuf->pIOBuffer->getBufferPtr()[m_uiRflReadOffset]);

	// At this point, we are guaranteed to have the entire packet in the
	// buffer.

	*ppucPacketBodyRV = pucPacketBody = &pucPacket[RFL_PACKET_OVERHEAD];

	// Validate the packet checksum

	if (RflCalcChecksum( pucPacket, uiEncryptPacketBodyLen) != 
								pucPacket[RFL_PACKET_CHECKSUM_OFFSET])
	{
		rc = RC_SET( FERR_BAD_RFL_PACKET);
		goto Exit;
	}

	if (uiPacketType == RFL_TRNS_BEGIN_PACKET ||
		 uiPacketType == RFL_TRNS_BEGIN_EX_PACKET ||
		 uiPacketType == RFL_UPGRADE_PACKET ||
		 uiPacketType == RFL_REDUCE_PACKET ||
		 uiPacketType == RFL_WRAP_KEY_PACKET ||
		 uiPacketType == RFL_ENABLE_ENCRYPTION_PACKET)
	{

		// Current transaction ID better be zero, otherwise, we have two or
		// more begin packets in a row.

		if (m_uiCurrTransID)
		{
			rc = RC_SET( FERR_BAD_RFL_PACKET);
			goto Exit;
		}

		m_uiCurrTransID = (FLMUINT) FB2UD( pucPacketBody);
		pucPacketBody += 4;

		// Make sure the transaction numbers are ascending

		if (m_uiCurrTransID <= m_uiLastTransID)
		{
			rc = RC_SET( FERR_BAD_RFL_PACKET);
			goto Exit;
		}

		if (uiPacketType == RFL_TRNS_BEGIN_EX_PACKET)
		{
			FLMUINT	uiLastLoggedCommitTransID;

			// Skip past seconds

			pucPacketBody += 4;

			uiLastLoggedCommitTransID = (FLMUINT) FB2UD( pucPacketBody);
			pucPacketBody += 4;

			if (m_pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_31 &&
				 m_uiLastLoggedCommitTransID != uiLastLoggedCommitTransID)
			{
				rc = RC_SET( FERR_RFL_TRANS_GAP);
				goto Exit;
			}
		}
	}
	else
	{

		// If transaction ID is not zero, we are not inside a transaction,
		// and it is likely that we have a corrupt packet.

		if (!m_uiCurrTransID)
		{
			rc = RC_SET( FERR_BAD_RFL_PACKET);
			goto Exit;
		}

		// Decrypt the packet if it is a packet type that was encrypted.

		if (uiPacketType == RFL_TRNS_COMMIT_PACKET ||
			 uiPacketType == RFL_TRNS_ABORT_PACKET)
		{
			if ((FLMUINT) FB2UD( pucPacketBody) != m_uiCurrTransID)
			{
				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}
		}
	}

	// Set read offset to beginning of next packet.

	m_uiRflReadOffset += (RFL_PACKET_OVERHEAD + uiEncryptPacketBodyLen);
	
Exit:

	return (rc);
}

/********************************************************************
Desc:		Get a record from the packets in the roll-forward log.
			This expects a series of RFL_DATA_RECORD_PACKETs.
*********************************************************************/
RCODE F_Rfl::getRecord(
	FDB *			pDb,
	FLMUINT		uiPacketType,
	FLMBYTE *	pucPacketBody,
	FLMUINT		uiPacketBodyLen,
	FlmRecord *	pRecord)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiTagNum;
	FLMUINT		uiDataType;
	FLMUINT		uiLevel;
	FLMUINT		uiDataLen;
	FLMBYTE *	pucFieldData = NULL;
	void *		pvField;
	FLMBOOL		bEncrypted = FALSE;
	FLMUINT		uiEncId = 0;
	FLMUINT		uiEncDataLen = 0;
	F_Pool		pool;

	pool.poolInit( 512);

	// Go into a loop processing packets until we have retrieved all of
	// the fields of the record. At that point, we had better be at the end
	// of the record.

	for (;;)
	{

		// If we don't currently have a packet, get one Packet type had
		// better be RFL_DATA_RECORD_PACKET.

		if (!uiPacketBodyLen)
		{
			if (RC_BAD( rc = getPacket( FALSE, &uiPacketType, &pucPacketBody,
						  &uiPacketBodyLen, NULL)))
			{
				goto Exit;
			}

			if (uiPacketType != RFL_DATA_RECORD_PACKET &&
				 uiPacketType != RFL_ENC_DATA_RECORD_PACKET &&
				 uiPacketType != RFL_DATA_RECORD_PACKET_VER_3)
			{
				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}
		}

		// Packet body length better be at least two or we have an
		// incomplete packet - we need to at least be able to get the tag
		// number at this point.

		if (uiPacketBodyLen < 2)
		{
			rc = RC_SET( FERR_BAD_RFL_PACKET);
			goto Exit;
		}
		else if (uiPacketBodyLen == 2)
		{

			// If the packet body length is only two, we had better be at
			// the end of the record with a tag number of zero. Otherwise, we
			// have an incomplete packet.

			if ((uiTagNum = (FLMUINT) FB2UW( pucPacketBody)) != 0)
			{
				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}
			break;
		}
		else if (uiPacketType == RFL_DATA_RECORD_PACKET)
		{
			flmAssert( m_pFile->FileHdr.uiVersionNum <= FLM_FILE_FORMAT_VER_4_60);
			if (uiPacketBodyLen < 6)
			{

				// If we have a packet body length less than six (for
				// RFL_DATA_RECORD_PACKETs), we have an incomplete field
				// header.

				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}
		}
		else if (uiPacketType == RFL_ENC_DATA_RECORD_PACKET)
		{

			// This type of packet is only valid with versions of flaim >=
			// 4.60

			flmAssert( m_pFile->FileHdr.uiVersionNum == FLM_FILE_FORMAT_VER_4_60);

			if (uiPacketBodyLen < 7)
			{

				// If we have a packet body length less than seven we have an
				// incomplete field header.

				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}
		}
		else
		{
			flmAssert( uiPacketType == RFL_DATA_RECORD_PACKET_VER_3);
			flmAssert( m_pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_61);
			if (uiPacketBodyLen < 9)
			{

				// If we have a packet body length less than nine we have an
				// incomplete field header.

				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}
		}

		// At this point, we have a packet body length that is greater than
		// or equal to seven (or six), meaning we could not possibly be on
		// the last field of the record. Hence, a zero tag number is invalid
		// here.

		if ((uiTagNum = (FLMUINT) FB2UW( pucPacketBody)) == 0)
		{
			rc = RC_SET( FERR_BAD_RFL_PACKET);
			goto Exit;
		}

		pucPacketBody += 2;
		uiDataType = *pucPacketBody++;
		uiLevel = *pucPacketBody++;
		if (uiPacketType != RFL_DATA_RECORD_PACKET_VER_3)
		{
			uiDataLen = (FLMUINT) FB2UW( pucPacketBody);
			pucPacketBody += 2;
			uiPacketBodyLen -= 6;
		}
		else
		{
			uiDataLen = (FLMUINT) FB2UD( pucPacketBody);
			pucPacketBody += 4;
			uiPacketBodyLen -= 8;
		}

		// If the database version supports encryption, we need to check
		// for it.

		if (uiPacketType == RFL_ENC_DATA_RECORD_PACKET)
		{
			bEncrypted = (FLMBOOL) * pucPacketBody++;
			--uiPacketBodyLen;

			if (bEncrypted)
			{
				if (uiPacketBodyLen < 4)
				{
					flmAssert( 0);
					rc = RC_SET( FERR_BAD_RFL_PACKET);
					goto Exit;
				}

				// Extract the encryption ID and the encrypted length.

				uiEncId = FB2UW( pucPacketBody);
				pucPacketBody += 2;

				uiEncDataLen = FB2UW( pucPacketBody);
				pucPacketBody += 2;

				uiPacketBodyLen -= 4;
			}
		}
		else if (uiPacketType == RFL_DATA_RECORD_PACKET_VER_3)
		{
			bEncrypted = (FLMBOOL) * pucPacketBody++;
			--uiPacketBodyLen;

			if (bEncrypted)
			{
				if (uiPacketBodyLen < 6)
				{
					flmAssert( 0);
					rc = RC_SET( FERR_BAD_RFL_PACKET);
					goto Exit;
				}

				// Extract the encryption ID and the encrypted length.

				uiEncId = FB2UW( pucPacketBody);
				pucPacketBody += 2;

				uiEncDataLen = FB2UD( pucPacketBody);
				pucPacketBody += 4;

				uiPacketBodyLen -= 6;
			}
		}

		// Create a new field.

		if (RC_BAD( rc = pRecord->insertLast( uiLevel, uiTagNum, uiDataType,
					  &pvField)))
		{
			goto Exit;
		}

		if (!bEncrypted && uiDataLen)
		{
			if (RC_BAD( rc = pRecord->allocStorageSpace( pvField, uiDataType,
						  uiDataLen, 0, 0, 0, &pucFieldData, NULL)))
			{
				goto Exit;
			}

			while (uiDataLen)
			{
				if (uiDataLen > uiPacketBodyLen)
				{
					f_memcpy( pucFieldData, pucPacketBody, uiPacketBodyLen);
					pucFieldData += uiPacketBodyLen;
					pucPacketBody += uiPacketBodyLen;
					uiDataLen -= uiPacketBodyLen;

					uiPacketBodyLen = 0;

					// Get the next packet. Packet type had better be
					// RFL_DATA_RECORD_PACKET.

					if (RC_BAD( rc = getPacket( FALSE, &uiPacketType, &pucPacketBody,
								  &uiPacketBodyLen, NULL)))
					{
						goto Exit;
					}

					if (uiPacketType != RFL_DATA_RECORD_PACKET &&
						 uiPacketType != RFL_ENC_DATA_RECORD_PACKET &&
						 uiPacketType != RFL_DATA_RECORD_PACKET_VER_3)
					{
						rc = RC_SET( FERR_BAD_RFL_PACKET);
						goto Exit;
					}
				}
				else
				{
					f_memcpy( pucFieldData, pucPacketBody, uiDataLen);
					pucFieldData += uiDataLen;
					uiPacketBodyLen -= uiDataLen;
					pucPacketBody += uiDataLen;
					uiDataLen = 0;
				}
			}

			pucFieldData = NULL;
		}
		else if (bEncrypted)
		{
			FLMBYTE *		pucEncFieldData;

			if (uiEncDataLen)
			{
				if (RC_BAD( rc = pRecord->allocStorageSpace( pvField, uiDataType,
							  uiDataLen, uiEncDataLen, uiEncId,
							  FLD_HAVE_ENCRYPTED_DATA, &pucFieldData, &pucEncFieldData
							  )))
				{
					goto Exit;
				}
			}

			while (uiEncDataLen)
			{
				if (uiEncDataLen > uiPacketBodyLen)
				{
					f_memcpy( pucEncFieldData, pucPacketBody, uiPacketBodyLen);
					pucEncFieldData += uiPacketBodyLen;
					pucPacketBody += uiPacketBodyLen;
					uiEncDataLen -= uiPacketBodyLen;

					uiPacketBodyLen = 0;

					// Get the next packet. Packet type had better be
					// RFL_ENC_DATA_RECORD_PACKET.

					if (RC_BAD( rc = getPacket( FALSE, &uiPacketType, &pucPacketBody,
								  &uiPacketBodyLen, NULL)))
					{
						goto Exit;
					}

					if (uiPacketType != RFL_ENC_DATA_RECORD_PACKET &&
						 uiPacketType != RFL_DATA_RECORD_PACKET_VER_3)
					{
						rc = RC_SET( FERR_BAD_RFL_PACKET);
						goto Exit;
					}
				}
				else
				{
					f_memcpy( pucEncFieldData, pucPacketBody, uiEncDataLen);
					pucEncFieldData += uiEncDataLen;
					uiPacketBodyLen -= uiEncDataLen;
					pucPacketBody += uiEncDataLen;
					uiEncDataLen = 0;
				}
			}

			pucEncFieldData = NULL;

			if (!m_pFile->bInLimitedMode)
			{
				if (RC_BAD( rc = flmDecryptField( pDb->pDict, pRecord, pvField,
							  uiEncId, &pool)))
				{
					goto Exit;
				}
			}
		}
	}

Exit:

	return (rc);
}

/********************************************************************
Desc:		Modify a record using RFL_DATA_RECORD_PACKETs or
			RFL_CHANGE_FIELD_PACKETs.
*********************************************************************/
RCODE F_Rfl::modifyRecord(
	HFDB			hDb,
	FlmRecord *	pRecord)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiPacketType;
	FLMBYTE *	pucPacketBody;
	FLMUINT		uiPacketBodyLen;

	FLMUINT		uiChangeType;
	FLMUINT		uiPosition;
	FLMUINT		uiTagNum = 0;
	FLMUINT		uiDataType = 0;
	FLMUINT		uiLevel = 0;
	FLMUINT		uiDataLen = 0;
	FLMBYTE *	pucData;
	FLMBYTE *	pucEncData;
	FDB *			pDb = (FDB *) hDb;
	FLMBOOL		bEncrypted = FALSE;
	FLMUINT		uiEncDataLen;
	FLMUINT		uiEncId;
	FLMUINT		uiFlags;
	FlmField *	pField;
	FLMUINT		uiCurPos = 1;
	void *		pvField;

	// Get the first packet and see what it is. If it is an
	// RFL_DATA_RECORD_PACKET, just call Rfl3GetRecord to get the entire
	// new record.

	if (RC_BAD( rc = getPacket( FALSE, &uiPacketType, &pucPacketBody,
				  &uiPacketBodyLen, NULL)))
	{
		goto Exit;
	}

	if (uiPacketType == RFL_DATA_RECORD_PACKET ||
		 uiPacketType == RFL_ENC_DATA_RECORD_PACKET ||
		 uiPacketType == RFL_DATA_RECORD_PACKET_VER_3)
	{
		pRecord->clear();
		rc = getRecord( pDb, uiPacketType, pucPacketBody, uiPacketBodyLen, pRecord);
		goto Exit;
	}
	else if (uiPacketType != RFL_CHANGE_FIELDS_PACKET)
	{
		rc = RC_SET( FERR_BAD_RFL_PACKET);
		goto Exit;
	}

	// Go into a loop processing packets until we have processed all of
	// the changed fields for the record.

	pField = pRecord->getFieldPointer( pRecord->root());

	flmAssert( pField);

	for (;;)
	{
		uiEncDataLen = 0;
		uiEncId = 0;

		// If we don't currently have a packet, get one Packet type had
		// better be RFL_CHANGE_FIELDS_PACKET.

		if (!uiPacketBodyLen)
		{
			if (RC_BAD( rc = getPacket( FALSE, &uiPacketType, &pucPacketBody,
						  &uiPacketBodyLen, NULL)))
			{
				goto Exit;
			}

			if (uiPacketType != RFL_CHANGE_FIELDS_PACKET)
			{
				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}
		}

		// Packet body length better be at least three or we have an
		// incomplete packet - we need to at least be able to get the type
		// of change and the absolute position of the change.

		if (uiPacketBodyLen < 3)
		{
			rc = RC_SET( FERR_BAD_RFL_PACKET);
			goto Exit;
		}

		// Get the change type and the absolute position where the change
		// is to be put. A position of zero is illegal.

		uiChangeType = *pucPacketBody++;
		uiPosition = (FLMUINT) FB2UW( pucPacketBody);
		pucPacketBody += 2;
		uiPacketBodyLen -= 3;

		if (uiChangeType == RFL_END_FIELD_CHANGES)
		{

			// If we are not at the end of the packet, it must be a bad
			// packet. Also, uiPosition should be a zero for this packet.

			if (uiPacketBodyLen || uiPosition)
			{
				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}
			break;
		}

		// If not RFL_END_FIELD_CHANGES, a position of zero is illegal.

		if (!uiPosition)
		{
			rc = RC_SET( FERR_BAD_RFL_PACKET);
			goto Exit;
		}

		switch (uiChangeType)
		{
			case RFL_INSERT_FIELD:
			{
				flmAssert( m_pFile->FileHdr.uiVersionNum <= FLM_FILE_FORMAT_VER_4_60);
				bEncrypted = FALSE;
				if (uiPacketBodyLen < 6)
				{
					rc = RC_SET( FERR_BAD_RFL_PACKET);
					goto Exit;
				}

				uiTagNum = (FLMUINT) FB2UW( pucPacketBody);
				pucPacketBody += 2;
				uiDataType = *pucPacketBody++;
				uiLevel = *pucPacketBody++;
				uiDataLen = (FLMUINT)FB2UW( pucPacketBody);
				pucPacketBody += 2;
				uiPacketBodyLen -= 6;
				break;
			}
			case RFL_INSERT_ENC_FIELD:
			{
				flmAssert( m_pFile->FileHdr.uiVersionNum == FLM_FILE_FORMAT_VER_4_60);
				bEncrypted = TRUE;
				if (uiPacketBodyLen < 10)
				{
					rc = RC_SET( FERR_BAD_RFL_PACKET);
					goto Exit;
				}

				uiTagNum = (FLMUINT) FB2UW( pucPacketBody);
				pucPacketBody += 2;
				uiDataType = *pucPacketBody++;
				uiLevel = *pucPacketBody++;
				uiDataLen = (FLMUINT)FB2UW( pucPacketBody);
				pucPacketBody += 2;

				uiEncId = FB2UW( pucPacketBody);
				pucPacketBody += 2;

				uiEncDataLen = FB2UW( pucPacketBody);
				pucPacketBody += 2;

				uiPacketBodyLen -= 10;
				break;
			}
			case RFL_INSERT_LARGE_FIELD:
			{
				flmAssert( m_pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_61);
				bEncrypted = FALSE;
				if (uiPacketBodyLen < 8)
				{
					rc = RC_SET( FERR_BAD_RFL_PACKET);
					goto Exit;
				}
				uiTagNum = (FLMUINT) FB2UW( pucPacketBody);
				pucPacketBody += 2;
				uiDataType = *pucPacketBody++;
				uiLevel = *pucPacketBody++;
				uiDataLen = (FLMUINT)FB2UD( pucPacketBody);
				pucPacketBody += 4;
				uiPacketBodyLen -= 8;
				break;
			}
			case RFL_INSERT_ENC_LARGE_FIELD:
			{
				flmAssert( m_pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_61);
				bEncrypted = TRUE;
				if (uiPacketBodyLen < 14)
				{
					rc = RC_SET( FERR_BAD_RFL_PACKET);
					goto Exit;
				}
				uiTagNum = (FLMUINT) FB2UW( pucPacketBody);
				pucPacketBody += 2;
				uiDataType = *pucPacketBody++;
				uiLevel = *pucPacketBody++;
				uiDataLen = (FLMUINT)FB2UD( pucPacketBody);
				pucPacketBody += 4;

				uiEncId = FB2UW( pucPacketBody);
				pucPacketBody += 2;

				uiEncDataLen = FB2UW( pucPacketBody);
				pucPacketBody += 4;

				uiPacketBodyLen -= 14;
				break;
			}

			case RFL_MODIFY_FIELD:
			{
				flmAssert( m_pFile->FileHdr.uiVersionNum <= FLM_FILE_FORMAT_VER_4_60);
				bEncrypted = FALSE;
				if (uiPacketBodyLen < 3 || *pucPacketBody != RFL_REPLACE_BYTES)
				{
					rc = RC_SET( FERR_BAD_RFL_PACKET);
					goto Exit;
				}

				pucPacketBody++;
				uiDataLen = (FLMUINT) FB2UW( pucPacketBody);
				pucPacketBody += 2;
				uiPacketBodyLen -= 3;
				break;
			}

			case RFL_MODIFY_ENC_FIELD:
			{
				flmAssert( m_pFile->FileHdr.uiVersionNum == FLM_FILE_FORMAT_VER_4_60);
				bEncrypted = TRUE;
				if (uiPacketBodyLen < 7 || *pucPacketBody != RFL_REPLACE_BYTES)
				{
					rc = RC_SET( FERR_BAD_RFL_PACKET);
					goto Exit;
				}

				pucPacketBody++;
				uiDataLen = (FLMUINT) FB2UW( pucPacketBody);
				pucPacketBody += 2;

				uiEncId = FB2UW( pucPacketBody);
				pucPacketBody += 2;

				uiEncDataLen = FB2UW( pucPacketBody);
				pucPacketBody += 2;

				uiPacketBodyLen -= 7;
				break;
			}
			case RFL_MODIFY_LARGE_FIELD:
			{
				flmAssert( m_pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_61);
				bEncrypted = FALSE;
				if (uiPacketBodyLen < 5 || *pucPacketBody != RFL_REPLACE_BYTES)
				{
					rc = RC_SET( FERR_BAD_RFL_PACKET);
					goto Exit;
				}
				pucPacketBody++;
				uiDataLen = (FLMUINT)FB2UD( pucPacketBody);
				pucPacketBody += 4;
				uiPacketBodyLen -= 5;
				break;
			}
			case RFL_MODIFY_ENC_LARGE_FIELD:
			{
				flmAssert( m_pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_61);
				bEncrypted = TRUE;
				if (uiPacketBodyLen < 11 || *pucPacketBody != RFL_REPLACE_BYTES)
				{
					rc = RC_SET( FERR_BAD_RFL_PACKET);
					goto Exit;
				}

				pucPacketBody++;
				uiDataLen = (FLMUINT)FB2UD( pucPacketBody);
				pucPacketBody += 4;

				uiEncId = FB2UW( pucPacketBody);
				pucPacketBody += 2;

				uiEncDataLen = FB2UD( pucPacketBody);
				pucPacketBody += 4;

				uiPacketBodyLen -= 11;
				break;
			}

			case RFL_DELETE_FIELD:
			{
				break;
			}

			default:
			{

				// Bad change type in packet.

				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}
		}

		// Now position to the target field.

		switch (uiChangeType)
		{
			case RFL_DELETE_FIELD:
			case RFL_MODIFY_FIELD:
			case RFL_MODIFY_ENC_FIELD:
			case RFL_MODIFY_LARGE_FIELD:
			case RFL_MODIFY_ENC_LARGE_FIELD:
			{
				while (uiCurPos != uiPosition)
				{
					if (uiPosition < uiCurPos)
					{
						flmAssert( pField->uiPrev);
						pField = pRecord->prevField( pField);
						--uiCurPos;
					}
					else
					{
						flmAssert( pField->uiNext);
						pField = pRecord->nextField( pField);
						uiCurPos++;
					}
				}

				if (uiChangeType != RFL_DELETE_FIELD)
				{

					// Get the data type ... not supplied in the modify field
					// packet.

					uiDataType = pRecord->getFieldDataType( pField);
					uiTagNum = pField->ui16FieldID;
				}
				break;
			}

			case RFL_INSERT_FIELD:
			case RFL_INSERT_ENC_FIELD:
			case RFL_INSERT_LARGE_FIELD:
			case RFL_INSERT_ENC_LARGE_FIELD:
			{
				FlmField *	pNewField;

				// On insert, we may be trying to position to a field that
				// does not exist yet. Therefore we need to position to the
				// field prior to the field position we want to insert.

				flmAssert( uiPosition > 1);	// cannot insert at the root
														///position.

				while (uiCurPos != uiPosition - 1)
				{
					if (uiPosition - 1 < uiCurPos)
					{
						flmAssert( pField->uiPrev);
						pField = pRecord->prevField( pField);
						--uiCurPos;
					}
					else
					{
						flmAssert( pField->uiNext);
						pField = pRecord->nextField( pField);
						uiCurPos++;
					}
				}

				// Insert the new field at the specified position and get
				// back a new field to use later.

				if (RC_BAD( rc = pRecord->createField( pField, &pNewField)))
				{
					goto Exit;
				}

				if (RC_BAD( rc = pRecord->setFieldLevel( pNewField, uiLevel)))
				{
					goto Exit;
				}

				pField = pNewField;
				pField->ui16FieldID = (FLMUINT16) uiTagNum;
				uiCurPos++;							// Bump the position as we have
														///just added a new field and

				// we are positioned on it.

				break;
			}
		}

		if (uiChangeType == RFL_DELETE_FIELD)
		{

			// Remove the specified field or subtree

			pvField = (void *) ((FLMUINT) (pField->uiPrev));
			--uiCurPos;
			if (!pvField)
			{
				pvField = pRecord->root();
				uiCurPos = 1;
			}

			// For versions 4.60 and greater, the interpretation for
			// RFL_DELETE_FIELD is to delete the entire sub-tree. Prior to
			// that, it is to delete only a single field.

			if (m_pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_60)
			{
				if (RC_BAD( rc = pRecord->remove( pField)))
				{
					goto Exit;
				}
			}
			else
			{

				// Passing in the same pointer to removeFields will
				// effectively delete just pField.

				if (RC_BAD( rc = pRecord->removeFields( pField, pField)))
				{
					goto Exit;
				}
			}

			// We need to reset our pField.

			pField = pRecord->getFieldPointer( pvField);

			continue;	// Next field...
		}

		// Both insert & modify need to have space allocated.

		if (bEncrypted)
		{
			uiFlags = FLD_HAVE_ENCRYPTED_DATA;
		}
		else
		{
			uiFlags = 0;
		}

		flmAssert( pField);

		// Allocate space for the data. We call this even if uiDataLen is
		// zero so that the appropriate data type will be set in the node as
		// well. ;
		// Before we allocate storage space, save the field offset. The
		// field buffer may get reallocated, resulting in a nwe address for
		// pField.
		pvField = pRecord->getFieldVoid( pField);
		if (RC_BAD( rc = pRecord->getNewDataPtr( pField, uiDataType, uiDataLen,
					  uiEncDataLen, uiEncId, uiFlags, &pucData, &pucEncData)))
		{
			goto Exit;
		}

		pField = pRecord->getFieldPointer( pvField);

		// Get the data for insert or modify, if any

		if (bEncrypted)
		{
			while (uiEncDataLen)
			{
				if (uiEncDataLen > uiPacketBodyLen)
				{
					f_memcpy( pucEncData, pucPacketBody, uiPacketBodyLen);
					pucEncData += uiPacketBodyLen;
					uiEncDataLen -= uiPacketBodyLen;
					uiPacketBodyLen = 0;

					// Get the next packet. Packet type had better be
					// RFL_CHANGE_FIELDS_PACKET.

					if (RC_BAD( rc = getPacket( FALSE, &uiPacketType, &pucPacketBody,
								  &uiPacketBodyLen, NULL)))
					{
						goto Exit;
					}

					if (uiPacketType != RFL_CHANGE_FIELDS_PACKET)
					{
						rc = RC_SET( FERR_BAD_RFL_PACKET);
						goto Exit;
					}
				}
				else
				{
					f_memcpy( pucEncData, pucPacketBody, uiEncDataLen);
					pucEncData += uiEncDataLen;
					uiPacketBodyLen -= uiEncDataLen;
					pucPacketBody += uiEncDataLen;
					uiEncDataLen = 0;
				}
			}
		}
		else				// Not encrypted
		{
			while (uiDataLen)
			{
				if (uiDataLen > uiPacketBodyLen)
				{
					f_memcpy( pucData, pucPacketBody, uiPacketBodyLen);
					pucData += uiPacketBodyLen;
					uiDataLen -= uiPacketBodyLen;
					uiPacketBodyLen = 0;

					// Get the next packet. Packet type had better be
					// RFL_CHANGE_FIELDS_PACKET.

					if (RC_BAD( rc = getPacket( FALSE, &uiPacketType, &pucPacketBody,
								  &uiPacketBodyLen, NULL)))
					{
						goto Exit;
					}

					if (uiPacketType != RFL_CHANGE_FIELDS_PACKET)
					{
						rc = RC_SET( FERR_BAD_RFL_PACKET);
						goto Exit;
					}
				}
				else
				{
					f_memcpy( pucData, pucPacketBody, uiDataLen);
					pucData += uiDataLen;
					uiPacketBodyLen -= uiDataLen;
					pucPacketBody += uiDataLen;
					uiDataLen = 0;
				}
			}
		}

		// If this field is involved in an index, and it is encrypted, we
		// need to make sure we decrypt it too. If it is not encrypted, we
		// don't care if it involved in an index.

		if (bEncrypted && !(pDb->pFile->bInLimitedMode))
		{
			IFD *	pIfd;

			if (RC_BAD( rc = fdictGetField( pDb->pDict, uiTagNum, NULL, &pIfd, NULL)))
			{
				goto Exit;
			}

			if (pIfd)
			{
				if (RC_BAD( rc = flmDecryptField( pDb->pDict, pRecord,
						pRecord->getFieldVoid( pField), uiEncId, &pDb->TempPool)))
				{
					goto Exit;
				}
			}
		}
	}

Exit:

	return (rc);
}

/********************************************************************
Desc:		Read the next operation from the roll-forward log.
*********************************************************************/
RCODE F_Rfl::readOp(
	FDB *				pDb,
	FLMBOOL			bForceNextFile,
	RFL_OP_INFO *	pOpInfo,
	FlmRecord *		pRecord)
{
	RCODE				rc = FERR_OK;
	FLMBYTE *		pucPacketBody;
	FLMUINT			uiPacketBodyLen;
	FLMUINT			uiExpectedBodyLen;
	FLMBOOL			bLoggedTimes;
	
	f_memset( pOpInfo, 0, sizeof( RFL_OP_INFO));

	// Get the next packet.

	if (RC_BAD( rc = getPacket( bForceNextFile, &pOpInfo->uiPacketType,
				  &pucPacketBody, &uiPacketBodyLen, &bLoggedTimes)))
	{
		goto Exit;
	}

	// Must be one of our packet types that represents an operation.

	switch (pOpInfo->uiPacketType)
	{
		case RFL_TRNS_BEGIN_PACKET:
		{
			uiExpectedBodyLen = 8;
			if (bLoggedTimes)
			{
				uiExpectedBodyLen += 4;
			}

			if (uiExpectedBodyLen != uiPacketBodyLen)
			{
				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}

			pOpInfo->uiTransId = (FLMUINT) FB2UD( pucPacketBody);
			pucPacketBody += 4;
			pOpInfo->uiStartTime = (FLMUINT) FB2UD( pucPacketBody);
			pucPacketBody += 4;
			break;
		}

		case RFL_TRNS_BEGIN_EX_PACKET:
		{
			uiExpectedBodyLen = 12;
			if (uiExpectedBodyLen != uiPacketBodyLen)
			{
				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}

			pOpInfo->uiTransId = (FLMUINT) FB2UD( pucPacketBody);
			pucPacketBody += 4;
			pOpInfo->uiStartTime = (FLMUINT) FB2UD( pucPacketBody);
			pucPacketBody += 4;
			pOpInfo->uiLastLoggedCommitTransId = (FLMUINT) FB2UD( pucPacketBody);
			pucPacketBody += 4;
			break;
		}

		case RFL_TRNS_COMMIT_PACKET:
		case RFL_TRNS_ABORT_PACKET:
		{
			uiExpectedBodyLen = 8;
			if (bLoggedTimes)
			{
				uiExpectedBodyLen += 8;
			}

			if (uiExpectedBodyLen != uiPacketBodyLen)
			{
				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}

			pOpInfo->uiTransId = (FLMUINT) FB2UD( pucPacketBody);
			pucPacketBody += 8;
			break;
		}

		case RFL_ADD_RECORD_PACKET:
		case RFL_MODIFY_RECORD_PACKET:
		case RFL_DELETE_RECORD_PACKET:
		case RFL_RESERVE_DRN_PACKET:
		{
			uiExpectedBodyLen = 10;
			if (bLoggedTimes)
			{
				uiExpectedBodyLen += 16;
			}

			if (uiExpectedBodyLen != uiPacketBodyLen)
			{
				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}

			if ((pOpInfo->uiTransId = 
				(FLMUINT) FB2UD( pucPacketBody)) != m_uiCurrTransID)
			{
				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}
			pucPacketBody += 4;
			
			pOpInfo->uiContainer = (FLMUINT) FB2UW( pucPacketBody);
			pucPacketBody += 2;
			
			pOpInfo->uiDrn = (FLMUINT) FB2UD( pucPacketBody);
			pucPacketBody += 4;
			
			if (pOpInfo->uiPacketType == RFL_ADD_RECORD_PACKET)
			{
				if (RC_BAD( rc = getRecord( pDb, 0, NULL, 0, pRecord)))
				{
					goto Exit;
				}
			}
			break;
		}

		case RFL_ADD_RECORD_PACKET_VER_2:
		case RFL_MODIFY_RECORD_PACKET_VER_2:
		case RFL_DELETE_RECORD_PACKET_VER_2:
		{
			uiExpectedBodyLen = 11;
			if (bLoggedTimes)
			{
				uiExpectedBodyLen += 16;
			}

			if (uiExpectedBodyLen != uiPacketBodyLen)
			{
				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}

			if ((pOpInfo->uiTransId = 
				(FLMUINT) FB2UD( pucPacketBody)) != m_uiCurrTransID)
			{
				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}
			pucPacketBody += 4;

			pOpInfo->uiContainer = (FLMUINT) FB2UW( pucPacketBody);
			pucPacketBody += 2;

			pOpInfo->uiDrn = (FLMUINT) FB2UD( pucPacketBody);
			pucPacketBody += 4;

			pOpInfo->uiFlags = *pucPacketBody;
			pucPacketBody++;

			// Translate the flags

			if (pOpInfo->uiFlags)
			{
				FLMUINT	uiTmp = 0;

				if (pOpInfo->uiFlags & RFL_UPDATE_BACKGROUND)
				{
					uiTmp |= FLM_DO_IN_BACKGROUND;
				}

				if (pOpInfo->uiFlags & RFL_UPDATE_SUSPENDED)
				{
					uiTmp |= FLM_SUSPENDED;
				}

				pOpInfo->uiFlags = uiTmp;
			}

			if (pOpInfo->uiPacketType == RFL_ADD_RECORD_PACKET_VER_2)
			{
				if (RC_BAD( rc = getRecord( pDb, 0, NULL, 0, pRecord)))
				{
					goto Exit;
				}
			}
			break;
		}

		case RFL_INDEX_SET_PACKET:
		case RFL_INDEX_SET_PACKET_VER_2:
		{
			uiExpectedBodyLen = 
					(FLMUINT) ((pOpInfo->uiPacketType == RFL_INDEX_SET_PACKET_VER_2) 
																? (FLMUINT) 16 
																: (FLMUINT) 14);

			if (bLoggedTimes)
			{
				uiExpectedBodyLen += 16;
			}

			if (uiExpectedBodyLen != uiPacketBodyLen)
			{
				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}

			if ((pOpInfo->uiTransId = 
				(FLMUINT) FB2UD( pucPacketBody)) != m_uiCurrTransID)
			{
				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}
			pucPacketBody += 4;
			
			if (pOpInfo->uiPacketType == RFL_INDEX_SET_PACKET_VER_2)
			{
				pOpInfo->uiContainer = (FLMUINT) FB2UW( pucPacketBody);
				pucPacketBody += 2;
			}

			pOpInfo->uiIndex = (FLMUINT) FB2UW( pucPacketBody);
			pucPacketBody += 2;
			
			pOpInfo->uiDrn = (FLMUINT) FB2UD( pucPacketBody);
			pucPacketBody += 4;
			
			pOpInfo->uiEndDrn = (FLMUINT) FB2UD( pucPacketBody);
			pucPacketBody += 4;
			
			break;
		}

		case RFL_START_UNKNOWN_PACKET:
		{
			uiExpectedBodyLen = 4;
			if (uiExpectedBodyLen != uiPacketBodyLen)
			{
				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}

			if ((pOpInfo->uiTransId = 
				(FLMUINT) FB2UD( pucPacketBody)) != m_uiCurrTransID)
			{
				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}

			pucPacketBody += 4;
			break;
		}

		case RFL_REDUCE_PACKET:
		{
			uiExpectedBodyLen = 8;
			if (uiExpectedBodyLen != uiPacketBodyLen)
			{
				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}

			pOpInfo->uiTransId = (FLMUINT) FB2UD( pucPacketBody);
			pOpInfo->uiLastLoggedCommitTransId = pOpInfo->uiTransId;
			pucPacketBody += 4;

			pOpInfo->uiCount = (FLMUINT) FB2UD( pucPacketBody);
			pucPacketBody += 4;
			
			break;
		}

		case RFL_BLK_CHAIN_FREE_PACKET:
		{
			uiExpectedBodyLen = 16;

			if (bLoggedTimes)
			{
				uiExpectedBodyLen += 16;
			}

			if (uiExpectedBodyLen != uiPacketBodyLen)
			{
				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}

			if ((pOpInfo->uiTransId = 
				(FLMUINT) FB2UD( pucPacketBody)) != m_uiCurrTransID)
			{
				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}

			pucPacketBody += 4;

			// Tracker record ID

			pOpInfo->uiDrn = (FLMUINT) FB2UD( pucPacketBody);
			pucPacketBody += 4;

			// Count

			pOpInfo->uiCount = (FLMUINT) FB2UD( pucPacketBody);
			pucPacketBody += 4;

			// Ending block address

			pOpInfo->uiEndBlock = (FLMUINT) FB2UD( pucPacketBody);
			pucPacketBody += 4;

			break;
		}

		case RFL_INDEX_SUSPEND_PACKET:
		case RFL_INDEX_RESUME_PACKET:
		{
			uiExpectedBodyLen = 6;
			if (uiExpectedBodyLen != uiPacketBodyLen)
			{
				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}

			if ((pOpInfo->uiTransId = 
				(FLMUINT) FB2UD( pucPacketBody)) != m_uiCurrTransID)
			{
				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}

			pucPacketBody += 4;

			pOpInfo->uiIndex = (FLMUINT) FB2UW( pucPacketBody);
			pucPacketBody += 2;
			break;
		}

		case RFL_UPGRADE_PACKET:
		{
			FLMUINT	uiDBKeyLen;

			uiExpectedBodyLen = 12;
			if (uiExpectedBodyLen > uiPacketBodyLen)
			{
				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}

			pOpInfo->uiTransId = (FLMUINT) FB2UD( pucPacketBody);
			pucPacketBody += 4;
			uiPacketBodyLen -= 4;
			
			pOpInfo->uiOldVersion = (FLMUINT) FB2UD( pucPacketBody);
			pucPacketBody += 4;
			uiPacketBodyLen -= 4;
			
			pOpInfo->uiNewVersion = (FLMUINT) FB2UD( pucPacketBody);
			pucPacketBody += 4;
			uiPacketBodyLen -= 4;

			// Only look for the wrapping key if the new database version is
			// greater than 4.60 and there isn't already a key.

			if (pOpInfo->uiEndDrn >= FLM_FILE_FORMAT_VER_4_60 && 
				 !m_pFile->pDbWrappingKey)
			{
				if (uiPacketBodyLen < 2)
				{
					rc = RC_SET( FERR_BAD_RFL_PACKET);
					goto Exit;
				}

				uiDBKeyLen = FB2UW( pucPacketBody);
				pucPacketBody += 2;
				uiPacketBodyLen -= 2;
				
				if (uiDBKeyLen)
				{
					if (uiPacketBodyLen != uiDBKeyLen)
					{
						rc = RC_SET( FERR_BAD_RFL_PACKET);
						goto Exit;
					}

					if ((m_pFile->pDbWrappingKey = f_new F_CCS) == NULL)
					{
						rc = RC_SET( FERR_MEM);
						goto Exit;
					}

					if (RC_BAD( rc = m_pFile->pDbWrappingKey->init( TRUE,
								  FLM_NICI_AES)))
					{
						goto Exit;
					}

					if (RC_BAD( rc = m_pFile->pDbWrappingKey->setKeyFromStore(
								pucPacketBody, (FLMUINT32) uiDBKeyLen, NULL, NULL,
								FALSE)))
					{
						goto Exit;
					}

					pucPacketBody += uiDBKeyLen;
					uiPacketBodyLen -= uiDBKeyLen;
					flmAssert( !uiPacketBodyLen);
				}
			}
			break;
		}

		case RFL_CONFIG_SIZE_EVENT_PACKET:
		{
			uiExpectedBodyLen = 16;
			if (uiExpectedBodyLen != uiPacketBodyLen)
			{
				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}

			pOpInfo->uiTransId = (FLMUINT) FB2UD( pucPacketBody);
			pOpInfo->uiLastLoggedCommitTransId = pOpInfo->uiTransId;
			pucPacketBody += 4;

			pOpInfo->uiSizeThreshold = (FLMUINT) FB2UD( pucPacketBody);
			pucPacketBody += 4;
			
			pOpInfo->uiTimeInterval = (FLMUINT) FB2UD( pucPacketBody);
			pucPacketBody += 4;
			
			pOpInfo->uiSizeInterval = (FLMUINT) FB2UD( pucPacketBody);
			pucPacketBody += 4;
			
			break;
		}
		
		case RFL_WRAP_KEY_PACKET:
		case RFL_ENABLE_ENCRYPTION_PACKET:
		{
			FLMUINT					uiDBKeyLen;
			FLMBYTE *				pucUncommittedLogHdr = &m_pFile->ucUncommittedLogHdr[0];
			eRestoreActionType	eRestoreAction;

			uiExpectedBodyLen = 6;
			if (uiExpectedBodyLen >= uiPacketBodyLen)
			{
				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}

			pOpInfo->uiTransId = (FLMUINT) FB2UD( pucPacketBody);
			pOpInfo->uiLastLoggedCommitTransId = pOpInfo->uiTransId;
			pucPacketBody += 4;
			uiPacketBodyLen -= 4;

			if (uiPacketBodyLen < 2)
			{
				rc = RC_SET( FERR_BAD_RFL_PACKET);
				goto Exit;
			}

			uiDBKeyLen = FB2UW( pucPacketBody);
			pucPacketBody += 2;
			uiPacketBodyLen -= 2;

			if (m_pRestore)
			{
				if (RC_BAD( rc = m_pRestore->status( 
						  pOpInfo->uiPacketType == RFL_WRAP_KEY_PACKET 
								? RESTORE_WRAP_KEY 
								: RESTORE_ENABLE_ENCRYPTION,
						  pOpInfo->uiTransId, (void *) uiDBKeyLen, (void *) 0, 
						  (void *) 0, &eRestoreAction)))
				{
					goto Exit;
				}

				if (eRestoreAction == RESTORE_ACTION_STOP)
				{
					m_uiCurrTransID = 0;
					break;
				}
			}

			if (uiDBKeyLen)
			{
				if (uiPacketBodyLen != uiDBKeyLen)
				{
					rc = RC_SET( FERR_BAD_RFL_PACKET);
					goto Exit;
				}

				// We cannot directly set the key at this point as it may be
				// encrypted using a password, which we do not have here. We
				// will write the key out to the log header and trust the user
				// to know whether or not a password is needed to open the
				// database.

				if (RC_BAD( rc = flmBeginDbTrans( pDb, FLM_UPDATE_TRANS, FLM_NO_TIMEOUT, 0)))
				{
					goto Exit;
				}

				f_memcpy( &pucUncommittedLogHdr[ LOG_DATABASE_KEY], pucPacketBody,
							uiDBKeyLen);
							
				UW2FBA( uiDBKeyLen, &pucUncommittedLogHdr[ LOG_DATABASE_KEY_LEN]);

				if (RC_BAD( rc = flmCommitDbTrans( pDb, 0, TRUE)))
				{
					goto Exit;
				}

				pucPacketBody += uiDBKeyLen;
				uiPacketBodyLen -= uiDBKeyLen;
				flmAssert( !uiPacketBodyLen);
			}

			m_uiCurrTransID = 0;
			break;
		}

		default:
		{
			flmAssert( 0);
			rc = RC_SET( FERR_BAD_RFL_PACKET);
			break;
		}
	}

Exit:

	return (rc);
}

/********************************************************************
Desc:		Reads through unknown packets.
*********************************************************************/
RCODE F_Rfl::readUnknown(
	FLMUINT		uiLenToRead,
	FLMBYTE *	pucBuffer,
	FLMUINT *	puiBytesRead)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiPacketType;
	FLMUINT		uiBytesRead = 0;
	FLMUINT		uiBytesToCopy;

	// If we have read through all of the unknown packets, return
	// FERR_EOF_HIT.

	if (!m_bReadingUnknown)
	{
		rc = RC_SET( FERR_EOF_HIT);
		goto Exit;
	}

	// Process packets until we have satisfied the read request or until
	// we run out of unknown packets.

	while (uiLenToRead)
	{

		// Get a packet, if we don't have one.

		if (!m_uiUnknownPacketBodyLen)
		{
			if (RC_BAD( rc = getPacket( FALSE, &uiPacketType,
						  &m_pucUnknownPacketBody, &m_uiUnknownPacketBodyLen, NULL)))
			{
				m_bReadingUnknown = FALSE;
				m_uiUnknownPacketRc = rc;
				goto Exit;
			}

			if (uiPacketType != RFL_UNKNOWN_PACKET)
			{
				if (!uiBytesRead)
				{
					rc = RC_SET( FERR_EOF_HIT);
				}

				m_bReadingUnknown = FALSE;

				// At this point, we know that the entire packet is inside
				// our memory buffer, so it is safe to reset m_uiRflReadOffset
				// back to the beginning of the packet. The call to readOp()
				// will call getPacket again, which will get this exact same
				// packet for processing.

				m_uiRflReadOffset -= (RFL_PACKET_OVERHEAD + m_uiUnknownPacketBodyLen);
				goto Exit;
			}

			m_uiUnknownBodyLenProcessed = 0;
		}

		uiBytesToCopy = uiLenToRead;
		if (uiBytesToCopy > m_uiUnknownPacketBodyLen -
			 m_uiUnknownBodyLenProcessed)
		{
			uiBytesToCopy = m_uiUnknownPacketBodyLen - m_uiUnknownBodyLenProcessed;
		}

		f_memcpy( pucBuffer, m_pucUnknownPacketBody + m_uiUnknownBodyLenProcessed,
					uiBytesToCopy);
		pucBuffer += uiBytesToCopy;
		uiLenToRead -= uiBytesToCopy;
		uiBytesRead += uiBytesToCopy;
		m_uiUnknownBodyLenProcessed += uiBytesToCopy;

		// If we have exhausted the current packet, reset things so that we
		// will get a new packet the next time around.

		if (m_uiUnknownBodyLenProcessed == m_uiUnknownPacketBodyLen)
		{
			m_uiUnknownPacketBodyLen = 0;
			m_uiUnknownBodyLenProcessed = 0;
			m_pucUnknownPacketBody = NULL;
		}
	}

Exit:

	*puiBytesRead = uiBytesRead;
	return (rc);
}

/********************************************************************
Desc:		Restore transactions from the roll-forward log to the
			database.
*********************************************************************/
RCODE F_Rfl::recover(
	FDB *			pDb,
	F_Restore *	pRestore)
{
	RCODE						rc = FERR_OK;
	HFDB						hDb = (HFDB) pDb;
	FLMUINT					uiStartFileNum;
	FLMUINT					uiStartOffset;
	FLMUINT					uiOffset;
	FLMUINT					uiReadLen;
	FLMUINT					uiBytesRead;
	FLMBYTE					ucHdr[ 512];
	FLMUINT					uiCount;
	RFL_OP_INFO				opInfo;
	FlmRecord *				pRecord = NULL;
	FlmRecord *				pTmpRecord = NULL;
	eRestoreActionType	eRestoreAction;
	FLMBOOL					bTransActive = FALSE;
	FLMBOOL					bHadOperations = FALSE;
	FLMBOOL					bLastTransEndedAtFileEOF = FALSE;
	FLMBOOL					bForceNextFile;

	flmAssert( m_pFile);

	m_pCurrentBuf = &m_Buf1;
	m_uiLastLoggedCommitTransID = 0;

	// We need to allow all updates logged in the RFL (including
	// dictionary updates).

	pDb->bFldStateUpdOk = TRUE;

	// If we are less than version 4.3, we cannot do restore.

	if (pRestore && m_pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_3)
	{
		goto Exit;
	}

	// Turn off logging.

	m_bLoggingOff = TRUE;

	// Set the replay flag on the database.

	pDb->uiFlags |= FDB_REPLAYING_RFL;

	// Set the flag as to whether or not we are using multiple RFL files.

	if (m_pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_3)
	{
		m_bKeepRflFiles = FALSE;
	}
	else
	{
		m_bKeepRflFiles = m_pFile->ucLastCommittedLogHdr[LOG_KEEP_RFL_FILES] 
														? TRUE 
														: FALSE;
	}
	
	// Determine the current, on-disk size of the RFL

	if( m_bKeepRflFiles)
	{
		FLMUINT64		ui64RflDiskUsage;
		
		if( RC_BAD( rc = flmRflCalcDiskUsage( m_szRflDir, m_szDbPrefix,
			m_pFile->FileHdr.uiVersionNum, &ui64RflDiskUsage)))
		{
			goto Exit;
		}
		
		f_mutexLock( gv_FlmSysData.hShareMutex);
		m_pFile->ui64RflDiskUsage = ui64RflDiskUsage;
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}

	// If pRestore is NULL, we are doing a database recovery after open.
	// In that case, we start from the last checkpoint offset and only run
	// until the last transaction offset.

	if ((m_pRestore = pRestore) == NULL)
	{
		FLMBYTE *	pucCheckSerialNum;
		FLMUINT		uiEndOffset;

		uiStartFileNum = (FLMUINT) FB2UD( 
				&m_pFile->ucLastCommittedLogHdr[ LOG_RFL_LAST_CP_FILE_NUM]);
														
		m_uiLastRecoverFileNum = (FLMUINT) FB2UD( 
			&m_pFile->ucLastCommittedLogHdr[ LOG_RFL_FILE_NUM]);
			
		uiStartOffset = (FLMUINT) FB2UD( 
			&m_pFile->ucLastCommittedLogHdr[ LOG_RFL_LAST_CP_OFFSET]);
			
		uiEndOffset = (FLMUINT) FB2UD( 
			&m_pFile->ucLastCommittedLogHdr[LOG_RFL_LAST_TRANS_OFFSET]);

		// Could be zero if the file was created, but no transactions were
		// ever committed to it.

		if (!uiEndOffset)
		{
			uiEndOffset = 512;
		}

		// Start offset better not be less than 512.

		flmAssert( uiStartOffset >= 512);
		flmAssert( uiEndOffset >= 512);

		// If start and end are at the same place, there is nothing to
		// recover.

		if (uiStartFileNum == m_uiLastRecoverFileNum &&
			 uiStartOffset == uiEndOffset)
		{
			goto Finish_Recovery;
		}

		// We have not recorded the serial number of the last checkpoint
		// file number, so we pass in NULL, unless it happens to be the same
		// as the last transaction file number, in which case we can pass in
		// the serial number we have stored in the log header.

		pucCheckSerialNum = (uiStartFileNum == m_uiLastRecoverFileNum)
				? &m_pFile->ucLastCommittedLogHdr[LOG_LAST_TRANS_RFL_SERIAL_NUM] 
				: NULL;
					
		if (RC_BAD( rc = openFile( uiStartFileNum, pucCheckSerialNum)))
		{
			goto Exit;
		}

		// If this is the last RFL file, the EOF is contained in the log
		// header. Otherwise, it will be in the RFL file's header, and
		// openFile will already have retrieved it.

		if (uiStartFileNum == m_uiLastRecoverFileNum)
		{
			m_uiFileEOF = uiEndOffset;
		}

		// At this point, file EOF better be greater than or equal to 512.

		flmAssert( m_uiFileEOF >= 512);
	}
	else if (!m_bKeepRflFiles)
	{

		// FlmDbRestore should be checking the "keep" flag and not
		// attempting to do a restore of the RFL.

		flmAssert( 0);
		rc = RC_SET( FERR_CANNOT_RESTORE_RFL_FILES);
		goto Exit;
	}
	else
	{
		uiStartFileNum = (FLMUINT) FB2UD( 
						&m_pFile->ucLastCommittedLogHdr[LOG_RFL_FILE_NUM]);
									
		uiStartOffset = (FLMUINT) FB2UD( 
						&m_pFile->ucLastCommittedLogHdr[LOG_RFL_LAST_TRANS_OFFSET]);

		// Could be zero if the RFL file had never been created.

		if (!uiStartOffset)
		{
			uiStartOffset = 512;
		}

		// Ask the recovery object to open the file.

Retry_Open:

		flmAssert( uiStartFileNum);
		if (RC_BAD( rc = m_pRestore->openRflFile( uiStartFileNum)))
		{
			if (rc == FERR_IO_PATH_NOT_FOUND)
			{

				// Need to set m_pCurrentBuf->uiCurrFileNum in case the first
				// call to openRflFile fails. This will cause the code at the
				// Finish_Recovery label to correctly set up the log header.

				if (!uiStartOffset)
				{
					m_pCurrentBuf->uiCurrFileNum = uiStartFileNum - 1;
				}
				else
				{
					m_pCurrentBuf->uiCurrFileNum = uiStartFileNum;
				}

				rc = FERR_OK;
				goto Finish_Recovery;
			}
			else
			{
				goto Exit;
			}
		}

		// Get the first 512 bytes from the file and verify the header.

		if (RC_BAD( rc = m_pRestore->read( 512, ucHdr, &uiBytesRead)))
		{
			goto Exit;
		}

		if (uiBytesRead < 512)
		{
			rc = RC_SET_AND_ASSERT( FERR_NOT_RFL);
			goto Exit;
		}

		if (RC_BAD( rc = verifyHeader( ucHdr, uiStartFileNum,
				&m_pFile->ucLastCommittedLogHdr[LOG_LAST_TRANS_RFL_SERIAL_NUM])))
		{
			RCODE tmpRc;

			if (RC_BAD( tmpRc = m_pRestore->status( RESTORE_ERROR, 0,
						  (void *) ((FLMUINT) rc), (void *) 0, (void *) 0,
						  &eRestoreAction)))
			{
				rc = tmpRc;
				goto Exit;
			}

			if (eRestoreAction == RESTORE_ACTION_RETRY)
			{
				if (RC_BAD( rc = m_pRestore->close()))
				{
					goto Exit;
				}

				goto Retry_Open;
			}

			goto Exit;
		}

		// We may not know the actual EOF of files during restore
		// operations.

		if ((m_uiFileEOF = (FLMUINT) FB2UD( &ucHdr[RFL_EOF_POS])) == 0)
		{
			bLastTransEndedAtFileEOF = TRUE;
		}
		else
		{
			bLastTransEndedAtFileEOF = (m_uiFileEOF == uiStartOffset) 
														? TRUE 
														: FALSE;
		}

		// Position to the start offset. Unfortunately, this means reading
		// through the data and discarding it.

		uiOffset = 512;
		while (uiOffset < uiStartOffset)
		{
			uiReadLen = (uiStartOffset - uiOffset);
			if (uiReadLen > m_uiBufferSize)
			{
				uiReadLen = m_uiBufferSize;
			}

			if (RC_BAD( rc = m_pRestore->read( uiReadLen,
						  m_pCurrentBuf->pIOBuffer->getBufferPtr(), &uiBytesRead)))
			{
				goto Exit;
			}

			// RFL file is incomplete if we could not read up to the last
			// committed transaction.

			if (uiBytesRead < uiReadLen)
			{
				rc = RC_SET( FERR_RFL_INCOMPLETE);
				goto Exit;
			}

			uiOffset += uiBytesRead;
		}

		// Need to set current file number

		m_pCurrentBuf->uiCurrFileNum = uiStartFileNum;

		// Better not be any transactions to recover - last database state
		// needs to be a completed checkpoint.

		flmAssert(
			FB2UD( &m_pFile->ucLastCommittedLogHdr [LOG_LAST_CP_TRANS_ID]) ==
			FB2UD( &m_pFile->ucLastCommittedLogHdr [LOG_CURR_TRANS_ID]));

		// Use uiStartOffset here instead of LOG_RFL_LAST_TRANS_OFFSET,
		// because LOG_RFL_LAST_TRANS_OFFSET may be zero, but we in that
		// case we should be comparing to 512, and uiStartOffset will have
		// been adjusted to 512 if that is the case.

		flmAssert( FB2UD( &m_pFile->ucLastCommittedLogHdr [
							LOG_RFL_LAST_CP_OFFSET]) == uiStartOffset);

		flmAssert( FB2UD( &m_pFile->ucLastCommittedLogHdr[
							LOG_RFL_LAST_CP_FILE_NUM]) == uiStartFileNum);
	}

	// Set last transaction ID to the last transaction that was
	// checkpointed - transaction numbers should ascend from here.

	m_uiLastTransID = (FLMUINT) FB2UD( 
							&m_pFile->ucLastCommittedLogHdr[LOG_LAST_CP_TRANS_ID]);

	// Set the last committed trans ID if this is a 4.31+ database

	if (m_pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_31)
	{
		m_uiLastLoggedCommitTransID = (FLMUINT) FB2UD( 
							&m_pFile->ucLastCommittedLogHdr[LOG_LAST_RFL_COMMIT_ID]);
	}

	m_pCurrentBuf->uiRflFileOffset = uiStartOffset;
	m_uiRflReadOffset = 0;
	m_pCurrentBuf->uiRflBufBytes = 0;

	// Now, read until we are done.

	bForceNextFile = FALSE;
	for (;;)
	{
		if (!pRecord)
		{
			if ((pRecord = f_new FlmRecord) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}
		}

		// Get the next operation from the file.

		rc = readOp( pDb, bForceNextFile, &opInfo, pRecord); 
		bForceNextFile = FALSE;

		if (RC_BAD( rc))
		{
Handle_Packet_Error:

			if (rc == FERR_END)
			{
				if (!m_pRestore)
				{

					// If we didn't end exactly where we should have, we have
					// an incomplete log. The same is true if we are in the
					// middle of a transaction.

					if (m_pCurrentBuf->uiCurrFileNum != m_uiLastRecoverFileNum ||
						 bTransActive)
					{
						rc = RC_SET( FERR_RFL_INCOMPLETE);
					}
					else
					{
						rc = FERR_OK;
						goto Finish_Recovery;
					}
				}
				else
				{

					// If we are doing a restore, and we get to the end of the
					// log, it is OK - even if we are in the middle of a
					// transaction - the transaction will simply be aborted.

					rc = FERR_OK;
					goto Finish_Recovery;
				}
			}
			else if (rc == FERR_BAD_RFL_PACKET)
			{

				// If we don't know the current file size, and we are doing a
				// restore, it is OK to end on a bad packet - we will simply
				// abort the current transaction, if any. Then, try to go to
				// the next file, because we really don't know where this file
				// ends.

				if (m_pRestore && !m_uiFileEOF)
				{
					if (bTransActive)
					{
						FlmDbTransAbort( hDb);
						bTransActive = FALSE;
					}

					// Set current transaction ID to zero - as if we had
					// encountered an abort packet.

					m_uiCurrTransID = 0;
					bLastTransEndedAtFileEOF = TRUE;

					// Force to go to the next file

					bForceNextFile = TRUE;
					rc = FERR_OK;
					continue;
				}
			}

			goto Exit;
		}

		// At this point, we know we have a good packet, see what it is and
		// handle it.

		bHadOperations = TRUE;
		switch (opInfo.uiPacketType)
		{
			case RFL_TRNS_BEGIN_EX_PACKET:
			case RFL_TRNS_BEGIN_PACKET:
			{
				if (m_pRestore)
				{
					if (RC_BAD( rc = m_pRestore->status( RESTORE_BEGIN_TRANS,
								  opInfo.uiTransId, (void *) opInfo.uiStartTime, 
								  (void *) 0, (void *) 0, &eRestoreAction)))
					{
						goto Exit;
					}

					if (eRestoreAction == RESTORE_ACTION_STOP)
					{

						// Need to set m_uiCurrTransID to 0 since it was set by
						// getPacket(). We are not going to start a transaction
						// because of the user's request to exit.

						m_uiCurrTransID = 0;
						bLastTransEndedAtFileEOF = FALSE;
						goto Finish_Recovery;
					}
				}

				// If we already have a transaction active, we have a problem.

				flmAssert( !bTransActive);

				if (RC_BAD( rc = FlmDbTransBegin( hDb, FLM_UPDATE_TRANS, 0)))
				{
					goto Exit;
				}

				bTransActive = TRUE;
				break;
			}

			case RFL_TRNS_COMMIT_PACKET:
			{

				// Commit the current transaction.

				if (m_pRestore)
				{
					if (RC_BAD( rc = m_pRestore->status( RESTORE_COMMIT_TRANS,
								  opInfo.uiTransId, (void *) 0, (void *) 0, (void *) 0,
								  &eRestoreAction)))
					{
						goto Exit;
					}

					if (eRestoreAction == RESTORE_ACTION_STOP)
					{
						bLastTransEndedAtFileEOF = FALSE;
						goto Finish_Recovery;
					}
				}

				flmAssert( bTransActive);
				pDb->uiFlags |= FDB_REPLAYING_COMMIT;
				rc = FlmDbTransCommit( hDb);
				pDb->uiFlags &= ~FDB_REPLAYING_COMMIT;
				bTransActive = FALSE;

				if (RC_BAD( rc))
				{
					goto Exit;
				}

				m_uiLastLoggedCommitTransID = opInfo.uiTransId;
				
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

				m_uiLastTransID = opInfo.uiTransId;
				m_uiCurrTransID = 0;
				break;
			}

			case RFL_TRNS_ABORT_PACKET:
			{

				// Abort the current transaction.

				if (m_pRestore)
				{
					if (RC_BAD( rc = m_pRestore->status( RESTORE_ABORT_TRANS,
								  opInfo.uiTransId, (void *) 0, (void *) 0, (void *) 0,
								  &eRestoreAction)))
					{
						goto Exit;
					}

					if (eRestoreAction == RESTORE_ACTION_STOP)
					{
						bLastTransEndedAtFileEOF = FALSE;
						goto Finish_Recovery;
					}
				}

				flmAssert( bTransActive);
				rc = FlmDbTransAbort( hDb);
				bTransActive = FALSE;

				if (RC_BAD( rc))
				{
					goto Exit;
				}

				goto Finish_Transaction;
			}

			case RFL_ADD_RECORD_PACKET:
			case RFL_ADD_RECORD_PACKET_VER_2:
			{
				if (m_pRestore)
				{
					if (RC_BAD( rc = m_pRestore->status( RESTORE_ADD_REC, 
								  opInfo.uiTransId, (void *) opInfo.uiContainer,
								  (void *) opInfo.uiDrn, 
								  (void *) pRecord, &eRestoreAction)))
					{
						goto Exit;
					}

					if (eRestoreAction == RESTORE_ACTION_STOP)
					{
						bLastTransEndedAtFileEOF = FALSE;
						goto Finish_Recovery;
					}
				}

				rc = FlmRecordAdd( hDb, opInfo.uiContainer, &opInfo.uiDrn, 
										 pRecord, opInfo.uiFlags);
				pRecord->Release();
				pRecord = NULL;
				if (RC_BAD( rc))
				{
					goto Exit;
				}
				break;
			}

			case RFL_MODIFY_RECORD_PACKET:
			case RFL_MODIFY_RECORD_PACKET_VER_2:
			{

				// Must retrieve the record and then get the modify packet(s)
				// to alter it.

				if (RC_BAD( rc = FlmRecordRetrieve( hDb, opInfo.uiContainer, 
							  opInfo.uiDrn, FO_EXACT, &pRecord, NULL)))
				{
					goto Exit;
				}

				if ((pTmpRecord = pRecord->copy()) == NULL)
				{
					rc = RC_SET( FERR_MEM);
					goto Exit;
				}

				pRecord->Release();
				pRecord = NULL;

				if (RC_BAD( rc = modifyRecord( hDb, pTmpRecord)))
				{
					goto Handle_Packet_Error;
				}

				// Finally, modify the record in the database.

				if (m_pRestore)
				{
					if (RC_BAD( rc = m_pRestore->status( RESTORE_MOD_REC, 
								  opInfo.uiTransId,
								  (void *) opInfo.uiContainer, (void *) opInfo.uiDrn,
								  (void *) pTmpRecord, &eRestoreAction)))
					{
						goto Exit;
					}

					if (eRestoreAction == RESTORE_ACTION_STOP)
					{
						bLastTransEndedAtFileEOF = FALSE;
						goto Finish_Recovery;
					}
				}

				rc = FlmRecordModify( hDb, opInfo.uiContainer, opInfo.uiDrn, 
											 pTmpRecord, opInfo.uiFlags);

				pTmpRecord->Release();
				pTmpRecord = NULL;

				if (RC_BAD( rc))
				{
					goto Exit;
				}
				
				break;
			}

			case RFL_DELETE_RECORD_PACKET:
			case RFL_DELETE_RECORD_PACKET_VER_2:
			{
				if (m_pRestore)
				{
					if (RC_BAD( rc = m_pRestore->status( RESTORE_DEL_REC, 
								  opInfo.uiTransId,
								  (void *) opInfo.uiContainer, (void *) opInfo.uiDrn, 
								  (void *) 0, &eRestoreAction)))
					{
						goto Exit;
					}

					if (eRestoreAction == RESTORE_ACTION_STOP)
					{
						bLastTransEndedAtFileEOF = FALSE;
						goto Finish_Recovery;
					}
				}

				if (RC_BAD( rc = FlmRecordDelete( hDb, opInfo.uiContainer, 
								opInfo.uiDrn, opInfo.uiFlags)))
				{
					goto Exit;
				}
				break;
			}

			case RFL_RESERVE_DRN_PACKET:
			{
				if (m_pRestore)
				{
					if (RC_BAD( rc = m_pRestore->status( RESTORE_RESERVE_DRN,
								  opInfo.uiTransId, (void *) opInfo.uiContainer, 
								  (void *) opInfo.uiDrn, (void *) 0, &eRestoreAction)))
					{
						goto Exit;
					}

					if (eRestoreAction == RESTORE_ACTION_STOP)
					{
						bLastTransEndedAtFileEOF = FALSE;
						goto Finish_Recovery;
					}
				}

				if (RC_BAD( rc = FlmReserveNextDrn( hDb, opInfo.uiContainer, 
					&opInfo.uiDrn)))
				{
					goto Exit;
				}
				
				break;
			}

			case RFL_INDEX_SUSPEND_PACKET:
			{

				if (m_pRestore)
				{
					if (RC_BAD( rc = m_pRestore->status( RESTORE_INDEX_SUSPEND,
								  opInfo.uiTransId, (void *) opInfo.uiIndex, 
								  (void *) 0, (void *) 0, &eRestoreAction)))
					{
						goto Exit;
					}

					if (eRestoreAction == RESTORE_ACTION_STOP)
					{
						bLastTransEndedAtFileEOF = FALSE;
						goto Finish_Recovery;
					}
				}

				if (RC_BAD( rc = FlmIndexSuspend( hDb, opInfo.uiIndex)))
				{
					goto Exit;
				}
				
				break;
			}

			case RFL_INDEX_RESUME_PACKET:
			{

				if (m_pRestore)
				{
					if (RC_BAD( rc = m_pRestore->status( RESTORE_INDEX_RESUME,
								  opInfo.uiTransId, (void *) opInfo.uiIndex, (void *) 0, 
								  (void *) 0, &eRestoreAction)))
					{
						goto Exit;
					}

					if (eRestoreAction == RESTORE_ACTION_STOP)
					{
						bLastTransEndedAtFileEOF = FALSE;
						goto Finish_Recovery;
					}
				}

				if (RC_BAD( rc = FlmIndexResume( hDb, opInfo.uiIndex)))
				{
					goto Exit;
				}
				break;
			}

			case RFL_INDEX_SET_PACKET:
			case RFL_INDEX_SET_PACKET_VER_2:
			{
				if (m_pRestore)
				{
					if (RC_BAD( rc = m_pRestore->status( RESTORE_INDEX_SET,
								  opInfo.uiTransId, (void *) opInfo.uiIndex, 
								  (void *) opInfo.uiDrn, (void *) opInfo.uiEndDrn,
								  &eRestoreAction)))
					{
						goto Exit;
					}

					if (eRestoreAction == RESTORE_ACTION_STOP)
					{
						bLastTransEndedAtFileEOF = FALSE;
						goto Finish_Recovery;
					}
				}

				if (m_pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_50 &&
					 opInfo.uiPacketType != RFL_INDEX_SET_PACKET)
				{
					rc = RC_SET( FERR_BAD_RFL_PACKET);
					goto Exit;
				}
				
				if (RC_BAD( rc = flmDbIndexSetOfRecords( hDb, opInfo.uiIndex, 
							  opInfo.uiContainer, opInfo.uiDrn, opInfo.uiEndDrn)))
				{
					goto Exit;
				}
				
				break;
			}

			case RFL_BLK_CHAIN_FREE_PACKET:
			{
				if (m_pRestore)
				{
					if (RC_BAD( rc = m_pRestore->status( RESTORE_BLK_CHAIN_DELETE,
								  opInfo.uiTransId, (void *) opInfo.uiDrn, 
								  (void *) opInfo.uiCount,
								  (void *) opInfo.uiEndBlock, &eRestoreAction)))
					{
						goto Exit;
					}

					if (eRestoreAction == RESTORE_ACTION_STOP)
					{
						bLastTransEndedAtFileEOF = FALSE;
						goto Finish_Recovery;
					}
				}

				if (RC_BAD( rc = flmMaintFreeBlockChain( pDb, opInfo.uiDrn, 
							  opInfo.uiCount, opInfo.uiEndBlock, NULL)))
				{
					goto Exit;
				}
				break;
			}

			case RFL_START_UNKNOWN_PACKET:
			{
				if (m_pRestore)
				{
					F_RflUnknownStream	unkStrm;

					unkStrm.setup( this, TRUE);
					m_bReadingUnknown = TRUE;
					m_uiUnknownPacketBodyLen = 0;
					m_pucUnknownPacketBody = NULL;
					m_uiUnknownBodyLenProcessed = 0;
					m_uiUnknownPacketRc = FERR_OK;

					if (RC_BAD( rc = m_pRestore->processUnknown(
									  (F_UnknownStream*) &unkStrm)))
					{
						if (m_uiUnknownPacketRc != FERR_OK)
						{
							rc = m_uiUnknownPacketRc;
							goto Handle_Packet_Error;
						}

						goto Exit;
					}

					// If we did not read through all of the unknown packets,
					// skip them at this time.

					if (m_bReadingUnknown)
					{
						goto Skip_Unknown_Packets;
					}
				}
				else
				{
Skip_Unknown_Packets:

					// Skip all unknown packets.

					for (;;)
					{
						FLMUINT		uiPacketType;
						FLMBYTE *	pucPacketBody;
						FLMUINT		uiPacketBodyLen;

						if (RC_BAD( rc = getPacket( FALSE, &uiPacketType,
									  &pucPacketBody, &uiPacketBodyLen, NULL)))
						{
							goto Handle_Packet_Error;
						}

						// If we hit something other than an unknown packet,
						// "push" it back into the pipe so it will be processed
						// by readOp() up above.

						if (uiPacketType != RFL_UNKNOWN_PACKET)
						{

							// At this point, we know that the entire packet is
							// inside our memory buffer, so it is safe to reset
							// m_uiRflReadOffset back to the beginning of the
							// packet. The call to readOp() above will call
							// getPacket again, which will get this exact same
							// packet for processing.

							m_uiRflReadOffset -= 
									(RFL_PACKET_OVERHEAD + uiPacketBodyLen);
							break;
						}
					}
				}
				break;
			}

			case RFL_REDUCE_PACKET:
			{

				if (m_pRestore)
				{
					if (RC_BAD( rc = m_pRestore->status( RESTORE_REDUCE, 
								  opInfo.uiTransId, (void *) opInfo.uiCount,
								  (void *) 0, (void *) 0, &eRestoreAction)))
					{
						goto Exit;
					}

					if (eRestoreAction == RESTORE_ACTION_STOP)
					{

						// Need to set m_uiCurrTransID to 0 since it was set by
						// getPacket(). We are not going to start a transaction
						// because of the user's request to exit.

						m_uiCurrTransID = 0;
						bLastTransEndedAtFileEOF = FALSE;
						goto Finish_Recovery;
					}
				}

				if (RC_BAD( rc = FlmDbReduceSize( hDb, opInfo.uiCount, &uiCount)))
				{
					goto Exit;
				}

				goto Finish_Transaction;
			}

			case RFL_UPGRADE_PACKET:
			{
				if (m_pRestore)
				{
					if (RC_BAD( rc = m_pRestore->status( RESTORE_UPGRADE, 
								  opInfo.uiTransId,
								  (void *) opInfo.uiOldVersion,
								  (void *) opInfo.uiNewVersion,
								  (void *) 0, &eRestoreAction)))
					{
						goto Exit;
					}

					if (eRestoreAction == RESTORE_ACTION_STOP)
					{

						// Need to set m_uiCurrTransID to 0 since it was set by
						// getPacket(). We are not going to start a transaction
						// because of the user's request to exit.

						m_uiCurrTransID = 0;
						bLastTransEndedAtFileEOF = FALSE;
						goto Finish_Recovery;
					}
				}

				// Attempt the conversion if the current version is less than
				// the target version and the target version is less than or
				// equal to the highest version supported by this code.

				if (opInfo.uiNewVersion > FLM_CUR_FILE_FORMAT_VER_NUM)
				{
					rc = RC_SET( FERR_UNALLOWED_UPGRADE);
					goto Exit;
				}
				else
				{
					flmAssert( m_pFile->FileHdr.uiVersionNum < opInfo.uiNewVersion);

					// The logged "new" version may be a lesser version than
					// FLM_CURRENT_FILE_FORMAT_VERSION, which is what FlmDbUpgrade
					// upgrades to. This is OK because the current version
					// should support all packets in the RFL for versions that
					// are less than it. Otherwise, the RFL chain would have
					// been broken by the original upgrade and it would not
					// have logged an upgrade packet.

					if (RC_BAD( rc = FlmDbUpgrade( hDb, opInfo.uiNewVersion, 
						NULL, NULL)))
					{
						goto Exit;
					}
				}

				goto Finish_Transaction;
			}

			case RFL_WRAP_KEY_PACKET:
			case RFL_ENABLE_ENCRYPTION_PACKET:
			{
				goto Finish_Transaction;
			}

			case RFL_CONFIG_SIZE_EVENT_PACKET: // here
			{
				if (m_pRestore)
				{
					if (RC_BAD( rc = m_pRestore->status( RESTORE_CONFIG_SIZE_EVENT, 
								  opInfo.uiTransId,
								  (void *) opInfo.uiSizeThreshold,
								  (void *) opInfo.uiTimeInterval, 
								  (void *) opInfo.uiSizeInterval, &eRestoreAction)))
					{
						goto Exit;
					}

					if (eRestoreAction == RESTORE_ACTION_STOP)
					{

						// Need to set m_uiCurrTransID to 0 since it was set by
						// getPacket(). We are not going to start a transaction
						// because of the user's request to exit.

						m_uiCurrTransID = 0;
						bLastTransEndedAtFileEOF = FALSE;
						goto Finish_Recovery;
					}
				}

				if (RC_BAD( rc = flmSetRflSizeThreshold( 
					hDb, opInfo.uiSizeThreshold, opInfo.uiTimeInterval, 
					opInfo.uiSizeInterval)))
				{
					goto Exit;
				}

				goto Finish_Transaction;
			}
			
			default:
			{

				// Should not be getting other packet types at this point. ;
				// If we don't know the current file size, and we are doing a
				// restore, it is OK to end on a bad packet - we will simply
				// abort the current transaction, if any.
				
				if (m_pRestore && !m_uiFileEOF)
				{
					rc = FERR_OK;
					goto Finish_Recovery;
				}
				else
				{
					rc = RC_SET( FERR_BAD_RFL_PACKET);
				}

				goto Exit;
			}
		}
	}

Finish_Recovery:

	if (bTransActive)
	{
		FlmDbTransAbort( hDb);
		bTransActive = FALSE;
	}

	if (m_pRestore)
	{
		FLMUINT	uiNextRflFileNum = m_pCurrentBuf->uiCurrFileNum + 1;

		// At the end of the restore operation, we need to set things up so
		// that the next transaction will begin a new RFL file. If we ended
		// the restore in the middle of an RFL file, we need to set it up so
		// that the new RFL file will have a new serial number. If we ended
		// at the end of an RFL file, we can set it up so that the new RFL
		// file will have the next serial number. ;
		// Set up the next RFL file number and offset.
		
		UD2FBA( uiNextRflFileNum,
				 &m_pFile->ucLastCommittedLogHdr[LOG_RFL_FILE_NUM]);

		// Set a zero into the offset, this is a special case which tells
		// us that we should create the file no matter what - even if it
		// already exists - it should be overwritten.

		UD2FBA( 0, &m_pFile->ucLastCommittedLogHdr[LOG_RFL_LAST_TRANS_OFFSET]);

		if (bLastTransEndedAtFileEOF)
		{

			// Move the next serial number of the last RFL file processed
			// into into the current RFL serial number so that the log header
			// will be correct

			f_memcpy( &m_pFile->ucLastCommittedLogHdr[
				LOG_LAST_TRANS_RFL_SERIAL_NUM], m_ucNextSerialNum, 
				F_SERIAL_NUM_SIZE);
		}
		else
		{

			// Must create a new serial number so that when the new RFL file
			// is created, it will have that next serial number.

			if (RC_BAD( rc = f_createSerialNumber( &m_pFile->ucLastCommittedLogHdr[
							  LOG_LAST_TRANS_RFL_SERIAL_NUM])))
			{
				goto Exit;
			}
		}

		// Save the last logged commit transaction ID.

		if (m_pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_31 &&
			 m_uiLastLoggedCommitTransID)
		{
			UD2FBA( m_uiLastLoggedCommitTransID,
					 &m_pFile->ucLastCommittedLogHdr[LOG_LAST_RFL_COMMIT_ID]);
		}

		// No matter what, we must generate a new next serial number. This
		// is what will be written to the new RFL file's header when it is
		// created.

		if (RC_BAD( rc = f_createSerialNumber(
						  &m_pFile->ucLastCommittedLogHdr[LOG_RFL_NEXT_SERIAL_NUM])))
		{
			goto Exit;
		}
	}

	if (!bHadOperations)
	{

		// No transactions were recovered, but still need to setup a few
		// things.

		m_pFile->uiFirstLogCPBlkAddress = 0;
		m_pFile->uiLastCheckpointTime = (FLMUINT) FLM_GET_TIMER();

		// Save the state of the log header into the ucCheckpointLogHdr
		// buffer.

		f_memcpy( m_pFile->ucCheckpointLogHdr, m_pFile->ucLastCommittedLogHdr,
					LOG_HEADER_SIZE);
	}

	// Force a checkpoint to force the log files to be truncated and
	// everything to be reset. This is done because during recovery the
	// checkpoints that are executed do NOT truncate the RFL file - because
	// it is using the log file to recover!

	closeFile();
	m_pRestore = NULL;
	m_bLoggingOff = FALSE;
	pDb->uiFlags &= ~FDB_REPLAYING_RFL;
	
	if (RC_BAD( rc = FlmDbCheckpoint( hDb, 0)))
	{
		goto Exit;
	}

Exit:

	if (pRecord)
	{
		pRecord->Release();
	}

	if (pTmpRecord)
	{
		pTmpRecord->Release();
	}

	if (bTransActive)
	{
		FlmDbTransAbort( hDb);
	}

	pDb->bFldStateUpdOk = FALSE;
	pDb->uiFlags &= ~FDB_REPLAYING_RFL;

	return (rc);
}

/********************************************************************
Desc:
********************************************************************/
RCODE flmRflCalcDiskUsage(
	const char *	pszRflDir,
	const char *	pszRflPrefix,
	FLMUINT			uiDbVersionNum,
	FLMUINT64 *		pui64DiskUsage)
{
	RCODE				rc = FERR_OK;
	IF_DirHdl *		pDirHdl = NULL;
	FLMUINT			uiFileNumber;
	FLMUINT64		ui64DiskUsage;
	
	ui64DiskUsage = 0;
	
	if( RC_BAD( rc = gv_FlmSysData.pFileSystem->openDir( pszRflDir,
		(char *) "*", &pDirHdl)))
	{
		if( rc == FERR_IO_PATH_NOT_FOUND)
		{
			rc = FERR_OK;
		}

		goto Exit;
	}
	
	for( ;;)
	{
		if( RC_BAD( rc = pDirHdl->next()))
		{
			if( rc != FERR_IO_NO_MORE_FILES && rc != FERR_IO_PATH_NOT_FOUND)
			{
				goto Exit;
			}
			
			rc = FERR_OK;
			break;
		}
	
		// If the current file is an RFL file, increment the disk usage
	
		if( rflGetFileNum( uiDbVersionNum, pszRflPrefix, 
			pDirHdl->currentItemName(), &uiFileNumber))
		{
			ui64DiskUsage += pDirHdl->currentItemSize();
		}
	}
	
Exit:

	*pui64DiskUsage = ui64DiskUsage;

	if( pDirHdl)
	{
		pDirHdl->Release();
	}
	
	return( rc);
}

/********************************************************************
Desc:	Returns the name of an RFL file given its number
********************************************************************/
FLMEXP RCODE FLMAPI FlmDbGetRflFileName(
	HFDB			hDb,
	FLMUINT		uiFileNum,
	char *		pszFileName)
{
	((FDB *) hDb)->pFile->pRfl->getBaseRflFileName( uiFileNum, pszFileName);
	return (FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
F_RflUnknownStream::F_RflUnknownStream() 
{
	m_pRfl = NULL;
	m_bStartedWriting = FALSE;
	m_bInputStream = FALSE;
	m_bSetupCalled = FALSE;
}

/****************************************************************************
Desc:
****************************************************************************/
F_RflUnknownStream::~F_RflUnknownStream() 
{
	if (m_bSetupCalled)
	{
		(void)close();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_RflUnknownStream::setup(
		F_Rfl *			pRfl,
		FLMBOOL			bInputStream)
{
	RCODE			rc = FERR_OK;

	flmAssert( !m_bSetupCalled);

	if (!pRfl)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_INVALID_PARM);
		goto Exit;
	}
	m_pRfl = pRfl;
	m_bInputStream = bInputStream;
	m_bSetupCalled = TRUE;
	m_bStartedWriting = FALSE;

Exit:
	return( rc);
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE F_RflUnknownStream::close( void)
{
	RCODE			rc = FERR_OK;

	flmAssert( m_bSetupCalled);

	// There is nothing to do for input streams, because the RFL
	// code handles skipping over any unknown data that may not have
	// been read yet.
	// For output streams, we need to call the endLoggingUnknown
	// routine so that the last packet gets written out.

	if (!m_bInputStream)
	{
		if (m_bStartedWriting)
		{
			m_bStartedWriting = FALSE;
			if (RC_BAD( rc = m_pRfl->endLoggingUnknown()))
			{
				goto Exit;
			}
		}
	}
Exit:
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_RflUnknownStream::read(
	FLMUINT			uiLength,
	void *			pvBuffer,
	FLMUINT *		puiBytesRead)
{
	RCODE			rc = FERR_OK;

	flmAssert( m_bSetupCalled);

	if (!m_bInputStream)
	{

		// Cannot read from an output stream.

		flmAssert( 0);
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	if (RC_BAD( rc = m_pRfl->readUnknown( uiLength, (FLMBYTE *)pvBuffer,
										puiBytesRead)))
	{
		goto Exit;
	}

Exit:
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_RflUnknownStream::write(
	FLMUINT			uiLength,
	void *			pvBuffer)
{
	RCODE			rc = FERR_OK;

	flmAssert( m_bSetupCalled);
	flmAssert( m_pRfl);

	if (m_bInputStream)
	{

		// Cannot write to an input stream.

		flmAssert( 0);
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	// Need to start logging on the first write.

	if (!m_bStartedWriting)
	{
		if (RC_BAD( rc = m_pRfl->startLoggingUnknown()))
		{
			goto Exit;
		}
		m_bStartedWriting = TRUE;
	}

	// Log the data.

	if (RC_BAD( rc = m_pRfl->logUnknown( (FLMBYTE *)pvBuffer, uiLength)))
	{
		goto Exit;
	}
Exit:
	return( rc);
}

/****************************************************************************
Desc:	Returns an unknown stream object - suitable for writing unknown
		streams into the roll-forward log.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmDbGetUnknownStreamObj(
	HFDB						hDb,
	F_UnknownStream **	ppUnknownStream)
{
	RCODE						rc = FERR_OK;
	FDB *						pDb = (FDB *)hDb;
	F_RflUnknownStream *	pUnkStream = NULL;

	flmAssert( pDb);
	flmAssert( ppUnknownStream);

	// See if the database is being forced to close

	if( RC_BAD( rc = flmCheckDatabaseState( pDb)))
	{
		goto Exit;
	}

	// This is only valid on 4.3 and greater.

	if (pDb->pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_3)
	{
		goto Exit;	// Will return FERR_OK and a NULL pointer.
	}

	// Must be in an update transaction.

	if (pDb->uiTransType == FLM_NO_TRANS)
	{
		rc = RC_SET( FERR_NO_TRANS_ACTIVE);
		goto Exit;
	}
	if (pDb->uiTransType != FLM_UPDATE_TRANS)
	{
		rc = RC_SET( FERR_ILLEGAL_TRANS_OP);
		goto Exit;
	}

	// Allocate the stream object we want.

	if ((pUnkStream = f_new F_RflUnknownStream) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Setup the unknown stream object.

	if (RC_BAD( rc = pUnkStream->setup( pDb->pFile->pRfl, FALSE)))
	{
		goto Exit;
	}

Exit:

	if (RC_BAD( rc) && pUnkStream)
	{
		pUnkStream->Release();
		pUnkStream = NULL;
	}
	
	*ppUnknownStream = (F_UnknownStream *)pUnkStream;
	return( rc);
}
