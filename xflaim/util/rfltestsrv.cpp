//------------------------------------------------------------------------------
// Desc:	RFL unit tests
// Tabs:	3
//
// Copyright (c) 2005-2007 Novell, Inc. All Rights Reserved.
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
	#define DB_NAME_STR					"SYS:\\TST.DB"
	#define BACKUP_NAME_STR				"SYS:\\TST.BAK"
	#define NEW_NAME_STR					"SYS:\\NEW.DB"
#else
	#define DB_NAME_STR					"tst.db"
	#define BACKUP_NAME_STR				"tst.bak"
	#define NEW_NAME_STR					"new.db"
#endif

#ifdef FLM_UNIX
	#define LARGE_64BIT_VALUE			0x8a306d2d713a8cfeULL
#else
	#define LARGE_64BIT_VALUE			0x8a306d2d713a8cfe
#endif
		
/********************************************************************
Desc:
*********************************************************************/
class IFlmTestImpl : public TestBase
{
public:

	const char * getName( void);
	
	RCODE execute( void);
};

/********************************************************************
Desc:
*********************************************************************/
RCODE getTest(
	IFlmTest **		ppTest)
{
	RCODE		rc = NE_XFLM_OK;

	if( (*ppTest = f_new IFlmTestImpl) == NULL)
	{
		rc = NE_XFLM_MEM;
		goto Exit;
	}

Exit:

	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
const char * IFlmTestImpl::getName( void)
{
	return( "RFL Test");
}

/********************************************************************
Desc:
*********************************************************************/
RCODE IFlmTestImpl::execute( void)
{
	RCODE						rc = NE_XFLM_OK;
	char						szMsgBuf[ 64];
	IF_Backup *				pBackup = NULL;
	IF_DOMNode *			pDoc = NULL;
	IF_DOMNode *			pRootElement = NULL;
	IF_DOMNode *			pElement = NULL;
	IF_DOMNode *			pData = NULL;
	IF_DOMNode *			pAttr = NULL;
	IF_DOMNode *			pTmpNode = NULL;
	FLMUINT64				ui64NumberNodeId = 0;
	FLMUINT64				ui64Tmp;
	FLMUINT					uiDefNum;
	FLMBOOL					bStartedTrans = FALSE;
	FLMBOOL					bDibCreated = FALSE;

	beginTest( 					
		"rflTestSetup",
		"Create a FLAIM Database to do RFL testing",
		"Enable keepRfl, backup, do a bunch of update operations.",
		"No Additional Details.");

	if ( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to initialize test state.", m_szDetails, rc);
		goto Exit;
	}
	bDibCreated = TRUE;

	if( RC_BAD( rc = m_pDb->setRflKeepFilesFlag( TRUE)))
	{
		MAKE_FLM_ERROR_STRING( "setRflKeepFilesFlag failed", m_szDetails, rc);
		goto Exit;
	}

	// Backup the database

	if( RC_BAD( rc = m_pDb->backupBegin( XFLM_FULL_BACKUP,
		XFLM_READ_TRANS, 0, &pBackup)))
	{
		MAKE_FLM_ERROR_STRING( "backupBegin failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pBackup->backup( BACKUP_NAME_STR,
		NULL, NULL, NULL, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "backup failed", m_szDetails, rc);
		goto Exit;
	}

	pBackup->Release();
	pBackup = NULL;
	
	// Start an update transaction

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed", m_szDetails, rc);
		goto Exit;
	}
	bStartedTrans = TRUE;

	// Create some schema definitions

	uiDefNum = 0;
	if( RC_BAD( rc = m_pDb->createElementDef(
		NULL, "element1", XFLM_NUMBER_TYPE, &uiDefNum)))
	{
		MAKE_FLM_ERROR_STRING( "createElementDef failed", m_szDetails, rc);
		goto Exit;
	}

	// Create a document

	if( RC_BAD( rc = m_pDb->createDocument( XFLM_DATA_COLLECTION, &pDoc)))
	{
		MAKE_FLM_ERROR_STRING( "createDocument failed", m_szDetails, rc);
		goto Exit;
	}
	
	// Insert a large 64-bit value
	
	if( RC_BAD( rc = pDoc->createChildElement( m_pDb, uiDefNum, 
		XFLM_FIRST_CHILD, &pTmpNode, &ui64NumberNodeId)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pTmpNode->setUINT64( m_pDb, LARGE_64BIT_VALUE)))
	{
		goto Exit;
	}
	
	pTmpNode->Release();
	pTmpNode = NULL;

	// Import a document

	if ( RC_BAD( rc = importFile( "7006b90a.xml", XFLM_DATA_COLLECTION)))
	{
		goto Exit;
	}

	// Commit the transaction

	if( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_FLM_ERROR_STRING( "transCommit failed", m_szDetails, rc);
		goto Exit;
	}
	bStartedTrans = FALSE;

	endTest("PASS");

	// Close the database
	
	beginTest(
		"dbRemove",
		"remove the database",
		"Self-explanatory",
		"No Additional Details.");

	m_pDb->Release();
	m_pDb = NULL;

	if( RC_BAD( rc = m_pDbSystem->closeUnusedFiles( 0)))
	{
		MAKE_FLM_ERROR_STRING( "closeUnusedFiles failed", m_szDetails, rc);
		goto Exit;
	}

	// Remove the database

	if( RC_BAD( rc = m_pDbSystem->dbRemove( DB_NAME_STR, NULL, NULL, FALSE)))
	{
		MAKE_FLM_ERROR_STRING( "dbRemove failed", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

	// Restore the database
	
	beginTest( 			
		"dbRestore",
		"restore the database from the backup we just made",
		"Self-explanatory",
		"No Additional Details.");

	if( RC_BAD( rc = m_pDbSystem->dbRestore( DB_NAME_STR,
		NULL, NULL, BACKUP_NAME_STR, NULL, NULL, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "dbRestore failed", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");
	
	// Open the restored database and do some sanity checks
	
	beginTest( 			
		"Restore Sanity Check",
		"Look at values in the database and verify that they are correct",
		"Self-explanatory",
		"No Additional Details.");

	if( RC_BAD( rc = m_pDbSystem->dbOpen( DB_NAME_STR, 
		NULL, NULL, NULL, FALSE, &m_pDb)))
	{
		MAKE_FLM_ERROR_STRING( "dbOpen failed", m_szDetails, rc);
		goto Exit;
	}
	
	if( RC_BAD( rc = m_pDb->getNode( XFLM_DATA_COLLECTION,
		ui64NumberNodeId, &pTmpNode)))
	{
		MAKE_FLM_ERROR_STRING( "getNode failed", m_szDetails, rc);
		goto Exit;
	}
	
	if( RC_BAD( rc = pTmpNode->getUINT64( m_pDb, &ui64Tmp)))
	{
		MAKE_FLM_ERROR_STRING( "getUINT64 failed", m_szDetails, rc);
		goto Exit;
	}
	
	if( ui64Tmp != LARGE_64BIT_VALUE)
	{
		MAKE_FLM_ERROR_STRING( "data comparison failed", 
			m_szDetails, NE_XFLM_DATA_ERROR);
	}
	
	m_pDb->Release();
	m_pDb = NULL;
	
	endTest("PASS");

Exit:

	if ( RC_BAD( rc))
	{
		endTest("FAIL");
	}

	if (bStartedTrans)
	{
		m_pDb->transAbort();
	}

	if( pBackup)
	{
		pBackup->Release();
	}

	if( pDoc)
	{
		pDoc->Release();
	}

	if( pRootElement)
	{
		pRootElement->Release();
	}

	if( pElement)
	{
		pElement->Release();
	}

	if( pData)
	{
		pData->Release();
	}

	if( pAttr)
	{
		pAttr->Release();
	}

	if (pTmpNode)
	{
		pTmpNode->Release();
	}

	if( RC_BAD( rc))
	{
		f_sprintf( szMsgBuf, "Error 0x%04X\n", (unsigned)rc);
		display( szMsgBuf);
		log( szMsgBuf);

		return rc;
	}

	shutdownTestState( DB_NAME_STR, bDibCreated);
	return( rc);
}
