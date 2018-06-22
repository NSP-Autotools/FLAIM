//------------------------------------------------------------------------------
// Desc:	Unit test for testing binary values
// Tabs:	3
//
// Copyright (c) 2003-2007 Novell, Inc. All Rights Reserved.
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

#include "flmunittest.h"

#if defined( FLM_NLM)
	#define DB_NAME_STR					"SYS:\\BIN.DB"
#else
	#define DB_NAME_STR					"bin.db"
#endif

#define	BIN_FILE_SIZE					2000000
#define	BIG_FILE							"BINARYFILE"
#define	BIN_BUFFER_SIZE				6550

/****************************************************************************
Desc:
****************************************************************************/
class IBinaryTestImpl : public TestBase
{
public:

	RCODE verifyData( 
		IF_FileHdl *	pFileHdl, 
		FLMUINT64		ui64NodeId);

	RCODE encryptionTest( 
		IF_FileHdl *	pFileHdl, 
		FLMUINT64		ui64NodeId);

	RCODE	buildBinaryFile( void);

	const char * getName( void);

	RCODE execute( void);
};

/****************************************************************************
Desc:
****************************************************************************/
RCODE getTest( 
	IFlmTest **		ppTest)
{
	RCODE		rc = NE_XFLM_OK;

	if( (*ppTest = f_new IBinaryTestImpl) == NULL)
	{
		rc = NE_XFLM_MEM;
		goto Exit;
	}

Exit:
	return rc;
}

/****************************************************************************
Desc:
****************************************************************************/
const char * IBinaryTestImpl::getName( void)
{
	return( "Binary Test");
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IBinaryTestImpl::execute( void)
{
	RCODE					rc = NE_XFLM_OK;
	FLMBOOL				bDibCreated = FALSE;
	IF_FileSystem *	pFileSystem = NULL;
	IF_FileHdl *		pFileHdl = NULL;
	FLMUINT				uiBytesRead = 0;
	FLMUINT				uiTotalBytesRead = 0;
	FLMUINT				uiNameId = 0;
	IF_DOMNode *		pNode = NULL;
	FLMBOOL				bLast = FALSE;
	char					szBuffer[1024];
	FLMBOOL				bTransActive = FALSE;
	char *				pszBuffer = NULL;
	FLMUINT64			ui64NodeId;

	beginTest(
		"Large Binary Value Test",
		"Insert a large binary value into a DOM Node ",
		"Self-explanatory",
		"");

	if ( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		MAKE_ERROR_STRING( "Failed to initialize test state", m_szDetails, rc);
		goto Exit;
	}
	bDibCreated = TRUE;

	if ( RC_BAD( rc = buildBinaryFile()))
	{
		MAKE_ERROR_STRING( "Failed to create binary test file", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->createElementDef(
		NULL,
		"bin_data",
		XFLM_BINARY_TYPE,
		&uiNameId)))
	{
		MAKE_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->createRootElement(
		XFLM_DATA_COLLECTION, uiNameId, &pNode, &ui64NodeId)))
	{
		MAKE_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
		goto Exit;
	}

	m_pDbSystem->getFileSystem( &pFileSystem);

	if ( RC_BAD( rc = pFileSystem->openFile( 
		BIG_FILE, FLM_IO_RDONLY, &pFileHdl)))
	{
		MAKE_ERROR_STRING( "Failed to open file.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransActive = TRUE;

	// Set a small binary value so we can test to make sure that the code
	// allows an element with an embedded value to be changed to one with
	// a streaming value
	
	f_strcpy( szBuffer, "Hello, World!");
	if ( RC_BAD( rc = pNode->setBinary( m_pDb, szBuffer, 
		f_strlen( szBuffer) + 1, TRUE)))
	{
		MAKE_ERROR_STRING( "setBinary failed.", m_szDetails, rc);
		goto Exit;
	}
	
	for(;;)
	{
		if ( RC_BAD( rc = pFileHdl->read( 
			uiTotalBytesRead, 
			sizeof(szBuffer), 
			szBuffer,
			&uiBytesRead)))
		{
			if  ( rc == NE_FLM_IO_END_OF_FILE)
			{
				bLast = TRUE;
			}
			else
			{
				MAKE_ERROR_STRING( "Failed to read from file", m_szDetails, rc);
				goto Exit;
			}
		}

		if ( RC_BAD( rc = pNode->setBinary( m_pDb, szBuffer, uiBytesRead, bLast)))
		{
			MAKE_ERROR_STRING( "setBinary failed.", m_szDetails, rc);
			goto Exit;
		}
		uiTotalBytesRead += uiBytesRead;

		if ( bLast)
		{
			break;
		}			
	}

	pNode->Release();
	pNode = NULL;

	bTransActive = FALSE;
	if( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = verifyData( pFileHdl, ui64NodeId)))
	{
		goto Exit;
	}

	endTest("PASS");

	beginTest(
		"Large Encrypted Binary Value Test",
		"Insert a large encrypted binary value into a DOM Node ",
		"Self-explanatory",
		"");

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransActive = TRUE;

	if ( RC_BAD( rc = encryptionTest( pFileHdl, ui64NodeId)))
	{
		goto Exit;
	}

	bTransActive = FALSE;
	if( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

	beginTest(
		"6550 Byte Binary Value Test",
		"Insert a binary value of size 6550 into a DOM Node ",
		"Self-explanatory",
		"");

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransActive = TRUE;

	if ( RC_BAD( rc = f_alloc( BIN_BUFFER_SIZE, &pszBuffer)))
	{
		MAKE_ERROR_STRING( "f_alloc failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->getNode( XFLM_DATA_COLLECTION, ui64NodeId, &pNode)))
	{
		MAKE_ERROR_STRING( "getNode failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pNode->setBinary( m_pDb, pszBuffer, BIN_BUFFER_SIZE, TRUE)))
	{
		MAKE_ERROR_STRING( "setBinary failed", m_szDetails, rc);
		goto Exit;
	}

	bTransActive = FALSE;
	if( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}

Exit:

	if( bTransActive)
	{
		if ( RC_BAD( rc))
		{
			m_pDb->transAbort();
		}
		else
		{
			m_pDb->transCommit();
		}
	}

	if( RC_BAD( rc))
	{
		endTest("FAIL");
	}
	else
	{
		endTest("PASS");
	}

	if ( pszBuffer)
	{
		f_free( &pszBuffer);
	}

	if ( pNode)
	{
		pNode->Release();
	}

	pFileSystem->deleteFile( BIG_FILE);
	
	if ( pFileSystem)
	{
		pFileSystem->Release();
	}

	if ( pFileHdl)
	{
		pFileHdl->Release();
	}

	shutdownTestState( DB_NAME_STR, bDibCreated);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IBinaryTestImpl::buildBinaryFile( void)
{
	IF_FileSystem *	pFileSystem = NULL;
	IF_FileHdl *		pFileHdl = NULL;
	FLMUINT				uiBytesToWrite;
	FLMUINT				uiBytesWritten;
	FLMUINT				uiSpaceLeft = BIN_FILE_SIZE;
	char					szChunk[1000];
	char					c = 0;
	RCODE					rc = NE_XFLM_OK;

	m_pDbSystem->getFileSystem( &pFileSystem);

	pFileSystem->deleteFile( BIG_FILE);
	
	if ( RC_BAD( rc = pFileSystem->createFile( 
		BIG_FILE, FLM_IO_RDWR, &pFileHdl)))
	{
		MAKE_ERROR_STRING( "File create failed", m_szDetails, rc);
		goto Exit;
	}

	for( ;;)
	{
		uiBytesToWrite = f_min( uiSpaceLeft, sizeof(szChunk));
		f_memset( szChunk, c++, uiBytesToWrite);

		if ( RC_BAD( rc = pFileHdl->write( 
			FLM_IO_CURRENT_POS,
			uiBytesToWrite,
			szChunk,
			&uiBytesWritten)))
		{
			MAKE_ERROR_STRING( "Write failed", m_szDetails, rc);
			goto Exit;
		}

		uiSpaceLeft -= uiBytesToWrite;

		if ( !uiSpaceLeft)
		{
			break;
		}
	}

Exit:

	if ( pFileHdl)
	{
		pFileHdl->Release();
	}

	if ( pFileSystem)
	{
		pFileSystem->Release();
	}

	return rc;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IBinaryTestImpl::verifyData( 
	IF_FileHdl *	pFileHdl,
	FLMUINT64		ui64NodeId)
{
	RCODE				rc = NE_XFLM_OK;
	char				szBuf1[ 10000];
	char				szBuf2[ sizeof(szBuf1)];
	FLMUINT			uiTotalBytesRead = 0;
	FLMUINT			uiBytesRead1 = 0;
	FLMUINT			uiBytesRead2 = 0;
	FLMBOOL			bLast = FALSE;
	IF_DOMNode *	pNode = NULL;

	m_pDbSystem->clearCache( m_pDb);
	if( RC_BAD( rc = m_pDb->getNode( XFLM_DATA_COLLECTION, ui64NodeId, &pNode)))
	{
		goto Exit;
	}

	for(;;)
	{
		if ( RC_BAD( rc = pFileHdl->read( 
			uiTotalBytesRead, 
			sizeof(szBuf1), 
			szBuf1,
			&uiBytesRead1)))
		{
			if  ( rc == NE_FLM_IO_END_OF_FILE)
			{
				bLast = TRUE;
			}
			else
			{
				MAKE_ERROR_STRING( "Failed to read from file", m_szDetails, rc);
				goto Exit;
			}
		}

		if ( RC_BAD( rc = pNode->getBinary(
			m_pDb,
			szBuf2,
			uiTotalBytesRead,
			sizeof(szBuf2),
			&uiBytesRead2)))
		{
			if ( !( rc == NE_XFLM_EOF_HIT && bLast))
			{
				MAKE_ERROR_STRING( "getBinary failed.", m_szDetails, rc);
				goto Exit;
			}
			else
			{
				rc = NE_XFLM_OK;
			}
		}

		if ( (uiBytesRead1 != uiBytesRead2) ||
			f_memcmp( szBuf1, szBuf2, uiBytesRead1) != 0)
		{
			rc = NE_XFLM_DATA_ERROR;
			MAKE_ERROR_STRING( "Values do not match", m_szDetails, rc);
			goto Exit;
		}

		uiTotalBytesRead += uiBytesRead1;

		if ( bLast)
		{
			break;
		}			

	}

Exit:

	if( pNode)
	{
		pNode->Release();
	}

	return rc;
}

#define START_SEED	123

/****************************************************************************
Desc:
****************************************************************************/
RCODE IBinaryTestImpl::encryptionTest(
	IF_FileHdl *	pFileHdl, 
	FLMUINT64		ui64NodeId)
{
	RCODE						rc = NE_XFLM_OK;
	IF_RandomGenerator * pRand = NULL;
	FLMUINT					uiEncDef = 0;
	FLMUINT64				ui64TotalSize;
	char *					pszBuffer = NULL;
	FLMBOOL					bLast = FALSE;
	FLMUINT					uiTotalBytesRead = 0;
	FLMUINT					uiChunkSize;
	FLMUINT					uiBytesRead;
	IF_DOMNode *			pNode = NULL;

#ifdef FLM_USE_NICI
	if ( RC_BAD( rc = m_pDb->createEncDef(
		"aes",
		"aes_def",
		0,
		&uiEncDef)))
	{
		MAKE_ERROR_STRING( "Failed to create encryption definition", 
			m_szDetails, rc);
		goto Exit;
	}
#endif

	if( RC_BAD( rc = FlmAllocRandomGenerator( &pRand)))
	{
		goto Exit;
	}
	
	pRand->setSeed( START_SEED);

	if( RC_BAD( rc = m_pDb->getNode( XFLM_DATA_COLLECTION, ui64NodeId, &pNode)))
	{
		MAKE_ERROR_STRING( "getNode failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pFileHdl->size( &ui64TotalSize)))
	{
		MAKE_ERROR_STRING( "Failed to get file size", m_szDetails, rc);
		goto Exit;
	}

	for(;;)
	{
		if ( uiTotalBytesRead < ui64TotalSize)
		{
			uiChunkSize = pRand->getUINT32( 1, 
									(FLMUINT32)(ui64TotalSize - uiTotalBytesRead));

			if ( pszBuffer)
			{
				f_free( &pszBuffer);
			}

			if (RC_BAD( rc = f_alloc( uiChunkSize, &pszBuffer)))
			{
				MAKE_ERROR_STRING( "f_alloc failed", m_szDetails, rc);
				goto Exit;
			}

			if ( RC_BAD( rc = pFileHdl->read( 
				uiTotalBytesRead, 
				uiChunkSize, 
				pszBuffer,
				&uiBytesRead)))
			{
				MAKE_ERROR_STRING( "Failed to read from file", m_szDetails, rc);
				goto Exit;
			}
		}
		else
		{
			bLast = TRUE;
			uiBytesRead = 0;
		}

		if( RC_BAD( rc = pNode->setBinary( m_pDb, pszBuffer, uiBytesRead,
			bLast, uiEncDef)))
		{
			MAKE_ERROR_STRING( "setBinary failed.", m_szDetails, rc);
			goto Exit;
		}
		uiTotalBytesRead += uiBytesRead;

		if ( bLast)
		{
			break;
		}			
	}

	pNode->Release();
	pNode = NULL;

	if( RC_BAD( rc = verifyData( pFileHdl, ui64NodeId)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = f_realloc( uiTotalBytesRead, &pszBuffer)))
	{
		MAKE_ERROR_STRING( "f_realloc failed", m_szDetails, rc);
		goto Exit;
	}
	
	if( RC_BAD( rc = m_pDb->getNode( XFLM_DATA_COLLECTION, ui64NodeId, &pNode)))
	{
		MAKE_ERROR_STRING( "getNode failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pFileHdl->read( 0, uiTotalBytesRead, 
		pszBuffer, &uiBytesRead)))
	{
		MAKE_ERROR_STRING( "Failed to read from file", m_szDetails, rc);
		goto Exit;
	}
	
	// Set the node to have a large non-streaming value
	
	if( RC_BAD( rc = pNode->setBinary( m_pDb, pszBuffer, 
		uiBytesRead, TRUE, uiEncDef)))
	{
		goto Exit;
	}

	pNode->Release();
	pNode = NULL;

	if ( RC_BAD( rc = verifyData( pFileHdl, ui64NodeId)))
	{
		goto Exit;
	}
	
Exit:

	if( pszBuffer)
	{
		f_free( &pszBuffer);
	}

	if( pNode)
	{
		pNode->Release();
	}
	
	if( pRand)
	{
		pRand->Release();
	}

	return( rc);
}
