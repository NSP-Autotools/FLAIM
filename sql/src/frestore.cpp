//------------------------------------------------------------------------------
// Desc:	Methods used during restore
// Tabs:	3
//
// Copyright (c) 2001-2007 Novell, Inc. All Rights Reserved.
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
Desc: Constructor
****************************************************************************/
F_FSRestore::F_FSRestore()
{
	m_pFileHdl = NULL;
	m_pMultiFileHdl = NULL;
	m_ui64Offset = 0;
	m_bSetupCalled = FALSE;
	m_szDbPath[ 0] = 0;
	m_uiDbVersion = 0;
	m_szBackupSetPath[ 0] = 0;
	m_szRflDir[ 0] = 0;
	m_bOpen = FALSE;
}

/****************************************************************************
Desc: Destructor
****************************************************************************/
F_FSRestore::~F_FSRestore()
{
	if( m_bOpen)
	{
		(void)close();
	}
}


/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FSRestore::setup(
	const char *		pszDbPath,
	const char *		pszBackupSetPath,
	const char *		pszRflDir)
{
	flmAssert( !m_bSetupCalled);
	flmAssert( pszDbPath);
	flmAssert( pszBackupSetPath);

	f_strcpy( m_szDbPath, pszDbPath);
	f_strcpy( m_szBackupSetPath, pszBackupSetPath);

	if( pszRflDir)
	{
		f_strcpy( m_szRflDir, pszRflDir);
	}


	m_bSetupCalled = TRUE;
	return( NE_SFLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FSRestore::openBackupSet( void)
{
	RCODE			rc = NE_SFLM_OK;

	flmAssert( m_bSetupCalled);
	flmAssert( !m_pMultiFileHdl);
	
	if( RC_BAD( rc = FlmAllocMultiFileHdl( &m_pMultiFileHdl)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pMultiFileHdl->openFile( m_szBackupSetPath)))
	{
		m_pMultiFileHdl->Release();
		m_pMultiFileHdl = NULL;
		goto Exit;
	}

	m_ui64Offset = 0;
	m_bOpen = TRUE;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FSRestore::openRflFile(
	FLMUINT			uiFileNum)
{
	RCODE				rc = NE_SFLM_OK;
	char				szRflPath[ F_PATH_MAX_SIZE];
	char				szBaseName[ F_FILENAME_SIZE];
	FLMUINT			uiBaseNameSize;
	SFLM_DB_HDR			dbHdr;
	IF_FileHdl *	pFileHdl = NULL;

	flmAssert( m_bSetupCalled);
	flmAssert( uiFileNum);
	flmAssert( !m_pFileHdl);

	// Read the database header to determine the version number

	if( !m_uiDbVersion)
	{

		if( RC_BAD( rc = gv_SFlmSysData.pFileSystem->openFile( m_szDbPath,
			FLM_IO_RDWR | FLM_IO_SH_DENYNONE | FLM_IO_DIRECT, &pFileHdl)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = flmReadAndVerifyHdrInfo( NULL, pFileHdl, &dbHdr)))
		{
			goto Exit;
		}

		pFileHdl->Release();
		pFileHdl = NULL;

		m_uiDbVersion = (FLMUINT)dbHdr.ui32DbVersion;
	}

	// Generate the log file name.

	if( RC_BAD( rc = rflGetDirAndPrefix( m_szDbPath, m_szRflDir, szRflPath)))
	{
		goto Exit;
	}

	uiBaseNameSize = sizeof( szBaseName);
	rflGetBaseFileName( uiFileNum, szBaseName, &uiBaseNameSize, NULL);
	gv_SFlmSysData.pFileSystem->pathAppend( szRflPath, szBaseName);

	// Open the file.

	if( RC_BAD( rc = gv_SFlmSysData.pFileSystem->openFile( szRflPath,
		FLM_IO_RDWR | FLM_IO_SH_DENYNONE | FLM_IO_DIRECT, &m_pFileHdl)))
	{
		goto Exit;
	}

	m_bOpen = TRUE;
	m_ui64Offset = 0;

Exit:

	if( pFileHdl)
	{
		pFileHdl->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FSRestore::openIncFile(
	FLMUINT			uiFileNum)
{
	char			szIncPath[ F_PATH_MAX_SIZE];
	char			szIncFile[ F_FILENAME_SIZE];
	RCODE			rc = NE_SFLM_OK;

	flmAssert( m_bSetupCalled);
	flmAssert( !m_pMultiFileHdl);

	// Since this is a non-interactive restore, we will "guess"
	// that incremental backups are located in the same parent
	// directory as the main backup set.  We will further assume
	// that the incremental backup sets have been named XXXXXXXX.INC,
	// where X is a hex digit.

	if( RC_BAD( rc = gv_SFlmSysData.pFileSystem->pathReduce( m_szBackupSetPath,
		szIncPath, NULL)))
	{
		goto Exit;
	}

	f_sprintf( szIncFile, "%08X.INC", (unsigned)uiFileNum);
	gv_SFlmSysData.pFileSystem->pathAppend( szIncPath, szIncFile);
	
	if( RC_BAD( rc = FlmAllocMultiFileHdl( &m_pMultiFileHdl)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pMultiFileHdl->openFile( szIncPath)))
	{
		m_pMultiFileHdl->Release();
		m_pMultiFileHdl = NULL;
		goto Exit;
	}

	m_ui64Offset = 0;
	m_bOpen = TRUE;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FSRestore::read(
	FLMUINT			uiLength,
	void *			pvBuffer,
	FLMUINT *		puiBytesRead)
{
	FLMUINT		uiBytesRead = 0;
	RCODE			rc = NE_SFLM_OK;

	flmAssert( m_bSetupCalled);
	flmAssert( m_pFileHdl || m_pMultiFileHdl);

	if( m_pMultiFileHdl)
	{
		if( RC_BAD( rc = m_pMultiFileHdl->read( m_ui64Offset,
			uiLength, pvBuffer, &uiBytesRead)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = m_pFileHdl->read( (FLMUINT)m_ui64Offset,
			uiLength, pvBuffer, &uiBytesRead)))
		{
			goto Exit;
		}
	}

Exit:

	m_ui64Offset += uiBytesRead;

	if( puiBytesRead)
	{
		*puiBytesRead = uiBytesRead;
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FSRestore::close( void)
{
	flmAssert( m_bSetupCalled);

	if( m_pMultiFileHdl)
	{
		m_pMultiFileHdl->Release();
		m_pMultiFileHdl = NULL;
	}

	if( m_pFileHdl)
	{
		m_pFileHdl->Release();
		m_pFileHdl = NULL;
	}

	m_bOpen = FALSE;
	m_ui64Offset = 0;

	return( NE_SFLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FSRestore::abortFile( void)
{
	return( close());
}


