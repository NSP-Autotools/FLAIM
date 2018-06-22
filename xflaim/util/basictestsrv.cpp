//------------------------------------------------------------------------------
// Desc:	This is the main implementation of BasicTest component of our unit
//			test suite.  It handles basic operations such as: creating a db,
//			deleting a db, backup and restore and add and remove nodes.
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
	#define DB_NAME_STR					"SYS:\\TST.DB"
	#define BACKUP_NAME_STR				"SYS:\\TST.BAK"
	#define NEW_NAME_STR					"SYS:\\NEW.DB"
#else
	#define DB_NAME_STR					"tst.db"
	#define BACKUP_NAME_STR				"tst.bak"
	#define NEW_NAME_STR					"new.db"
#endif

#define BACKUP_PASSWORD_STR			"password"

/****************************************************************************
Desc:
****************************************************************************/
class IFlmTestImpl : public TestBase
{
public:

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

	if( (*ppTest = f_new IFlmTestImpl) == NULL)
	{
		rc = NE_XFLM_MEM;
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
const char * IFlmTestImpl::getName()
{
	return( "Basic Test");
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::execute()
{
	RCODE						rc = NE_XFLM_OK;
	char						szMsgBuf[ 64];
	FLMUINT					uiTmp = 0;
	FLMUINT64				ui64Tmp;
	FLMUINT					uiLoop;
	FLMUINT					uiHighNode;
	FLMUINT					uiDefNum;
	FLMUINT					uiDefNum1;
	FLMUINT					uiDefNum2;
	FLMUINT					uiDefNum3;
	FLMUINT					uiLargeTextSize;
	IF_Backup *				pBackup = NULL;
	IF_DOMNode *			pDoc = NULL;
	IF_DOMNode *			pRootElement = NULL;
	IF_DOMNode *			pUniqueElement = NULL;
	IF_DOMNode *			pNonUniqueElement	= NULL;
	IF_DOMNode *			pElement = NULL;
	IF_DOMNode *			pData = NULL;
	IF_DOMNode *			pAttr = NULL;
	IF_DOMNode *			pTmpNode = NULL;
	FLMBYTE *				pucLargeBuf = NULL;
	FLMBOOL					bStartedTrans = FALSE;
	FLMBOOL					bDibCreated = FALSE;

	beginTest(
		"Character conversion test",
		"Test character conversions from UTF8, Unicode, and Native",
		"",
		"");

	endTest("PASS");

	beginTest(
		"dbCreate",
		"Create a FLAIM Database",
		"Get the DbSystemFactory object through COM and call dbCreate",
		"No Additional Details.");

	if( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to initialize test state.", 
			m_szDetails, rc);
		goto Exit;
	}
	bDibCreated = TRUE;

	endTest("PASS");

	beginTest(	
		"createDocument",
		"Create a document and append 50000 child nodes",
		"createDocument; createElement; appendChild",
		"No Additional Details.");

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
		NULL, "element1", XFLM_TEXT_TYPE, &uiDefNum)))
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

	// Create an element

	if( RC_BAD( rc = pDoc->createNode( m_pDb, ELEMENT_NODE, ELM_ELEMENT_TAG,
												  XFLM_FIRST_CHILD, &pRootElement)))
	{
		MAKE_FLM_ERROR_STRING( "createNode failed", m_szDetails, rc);
		goto Exit;
	}
	
	// Generate a large text value

	uiLargeTextSize = 1024;
	if( RC_BAD( rc = f_alloc( uiLargeTextSize, &pucLargeBuf)))
	{
		MAKE_FLM_ERROR_STRING( "f_alloc failed", m_szDetails, rc);
		goto Exit;
	}
	
	for( uiLoop = 0; uiLoop < uiLargeTextSize - 1; uiLoop++)
	{
		pucLargeBuf[ uiLoop] = (FLMBYTE)('A' + (uiLoop % 26));
	}
	
	pucLargeBuf[ uiLargeTextSize - 1] = 0;

	for( uiLoop = 0; uiLoop < 5000; uiLoop++)
	{
		// Create an element

		if( RC_BAD( rc = pRootElement->createNode( m_pDb, ELEMENT_NODE,
			uiDefNum, XFLM_LAST_CHILD, &pElement)))
		{
			MAKE_FLM_ERROR_STRING( "createNode failed", m_szDetails, rc);
			goto Exit;
		}

		// Create an attribute

		if( RC_BAD( rc = pElement->createAttribute(
			m_pDb, ATTR_NEXT_INDEX_NUM_TAG, &pAttr)))
		{
			MAKE_FLM_ERROR_STRING( "createAttribute failed", m_szDetails, rc);
			goto Exit;
		}

		// Set the attribute's value

#ifdef FLM_UNIX
		if( RC_BAD( rc = pAttr->setUINT64( m_pDb, 0x8a306d2d713a8cfeULL)))
#else
		if( RC_BAD( rc = pAttr->setUINT64( m_pDb, 0x8a306d2d713a8cfe)))
#endif
		{
			MAKE_FLM_ERROR_STRING( "setUINT64 failed", m_szDetails, rc);
			goto Exit;
		}

		if( RC_BAD( rc = pAttr->getUINT64( m_pDb, &ui64Tmp)))
		{
			MAKE_FLM_ERROR_STRING( "getUINT failed", m_szDetails, rc);
			goto Exit;
		}
		
#ifdef FLM_UNIX
		if( ui64Tmp != 0x8a306d2d713a8cfeULL)
#else
		if( ui64Tmp != 0x8a306d2d713a8cfe)
#endif
		{
			rc = NE_XFLM_FAILURE;
			MAKE_FLM_ERROR_STRING( "getUINT returned invalid data", 
				m_szDetails, rc);
			goto Exit;
		}

		// Create another attribute

		if( RC_BAD( rc = pElement->createAttribute(
			m_pDb, ATTR_COMPARE_RULES_TAG, &pAttr)))
		{
			MAKE_FLM_ERROR_STRING( "createAttribute failed", m_szDetails, rc);
			goto Exit;
		}

		// Set the attribute's value

		if( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"1234567890", 0)))
		{
			MAKE_FLM_ERROR_STRING( "setNative failed", m_szDetails, rc);
			goto Exit;
		}

		if( RC_BAD( rc = pAttr->getUINT( m_pDb, &uiTmp)))
		{
			MAKE_FLM_ERROR_STRING( "getUINT failed", m_szDetails, rc);
			goto Exit;
		}

		if( uiTmp != 1234567890)
		{
			rc = NE_XFLM_FAILURE;
			MAKE_FLM_ERROR_STRING( "getUINT returned invalid data", 
				m_szDetails, rc);
			goto Exit;
		}

		if( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"987", 0, TRUE)))
		{
			MAKE_FLM_ERROR_STRING( "setNative failed", m_szDetails, rc);
			goto Exit;
		}

		if( RC_BAD( rc = pAttr->getUINT( m_pDb, &uiTmp)))
		{
			MAKE_FLM_ERROR_STRING( "getUINT failed", m_szDetails, rc);
			goto Exit;
		}

		if( uiTmp != 987)
		{
			rc = NE_XFLM_FAILURE;
			MAKE_FLM_ERROR_STRING( "getUINT returned invalid data", 
				m_szDetails, rc);
			goto Exit;
		}

		// Set a long value into the attribute
		
		if( RC_BAD( rc = pAttr->setUTF8( m_pDb, pucLargeBuf, 0, TRUE)))
		{
			MAKE_FLM_ERROR_STRING( "setNative failed", m_szDetails, rc);
			goto Exit;
		}
		
		f_memset( pucLargeBuf, 0, uiLargeTextSize);

		if( RC_BAD( rc = pAttr->getUTF8( m_pDb, pucLargeBuf,
			uiLargeTextSize, 0, uiLargeTextSize - 1)))
		{
			MAKE_FLM_ERROR_STRING( "getNative failed", m_szDetails, rc);
			goto Exit;
		}
		
		// Create a data node

		if( RC_BAD( rc = pElement->createNode( m_pDb, DATA_NODE,
			0, XFLM_LAST_CHILD, &pData)))
		{
			MAKE_FLM_ERROR_STRING( "createNode failed", m_szDetails, rc);
			goto Exit;
		}

		if( (uiLoop % 1000) == 0)
		{
			if ( m_bDisplayVerbose)
			{
				if( RC_BAD( rc = pData->getNodeId( m_pDb, &ui64Tmp)))
				{
					goto Exit;
				}
				
				f_sprintf( szMsgBuf, "Node count = %I64u", ui64Tmp);
				display( szMsgBuf);
			}
		}
	}

	endTest("PASS");

	// Read the nodes
	beginTest(	
		"read nodes",
		"Read and verify the DOM nodes",
		"Self-Explanatory",
		"No Additional Details.");

	if( RC_BAD( rc = pData->getNodeId( m_pDb, &ui64Tmp)))
	{
		goto Exit;
	}
		
	uiHighNode = (FLMUINT)ui64Tmp;
	for( uiLoop = 1; uiLoop < uiHighNode; uiLoop++)
	{
		if( RC_BAD( rc = m_pDb->getNode( XFLM_DATA_COLLECTION,
			uiLoop, &pTmpNode)))
		{
			MAKE_FLM_ERROR_STRING( "getNode failed", m_szDetails, rc);
			goto Exit;
		}

		if( (uiLoop % 1000) == 0)
		{
			if ( m_bDisplayVerbose)
			{
				f_sprintf( szMsgBuf, "Read = %I64u", (FLMUINT64)uiLoop);
				display( szMsgBuf);
			}
		}
	}

	endTest("PASS");

	// Delete the document
	beginTest(	
		"Document Delete",
		"delete the document we just created",
		"Self-explanatory",
		"No Additional Details.");

	if( RC_BAD( rc = pDoc->deleteNode( m_pDb)))
	{
		MAKE_FLM_ERROR_STRING( "deleteNode failed", m_szDetails, rc);
		goto Exit;
	}

	// Commit the transaction

	bStartedTrans = FALSE;
	if( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_FLM_ERROR_STRING( "transCommit failed", m_szDetails, rc);
		goto Exit;
	}

	// Close the database

	m_pDb->Release();
	m_pDb = NULL;

	endTest("PASS");
	
	// Tests for unique child elements.

	beginTest(	
		"Unique Child Elements",
		"Create/Read/Delete unique child elements",
		"Self-explanatory",
		"No Additional Details.");

	if( RC_BAD( rc = m_pDbSystem->dbOpen( DB_NAME_STR, NULL, NULL,
		NULL, FALSE, &m_pDb)))
	{
		MAKE_FLM_ERROR_STRING( "dbOpen failed", m_szDetails, rc);
		goto Exit;
	}
	
	// Start an update transaction

	if (RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed", m_szDetails, rc);
		goto Exit;
	}
	bStartedTrans = TRUE;
	
	// Create two element definitions, one that requires uniqueness,
	// another that does not.
	
	uiDefNum1 = 0;
	if( RC_BAD( rc = m_pDb->createUniqueElmDef(
		NULL, "u_element1", &uiDefNum1)))
	{
		MAKE_FLM_ERROR_STRING( "createUniqueElmDef failed", m_szDetails, rc);
		goto Exit;
	}
	uiDefNum2 = 0;
	if( RC_BAD( rc = m_pDb->createElementDef(
		NULL, "u_element2", XFLM_NODATA_TYPE, &uiDefNum2)))
	{
		MAKE_FLM_ERROR_STRING( "createElementDef failed", m_szDetails, rc);
		goto Exit;
	}
	
	// Create 200 unique element definitions.
	
	uiDefNum3 = uiDefNum2 + 1;
	for (uiLoop = 0; uiLoop < 200; uiLoop++)
	{
		char		szName [80];
		FLMUINT	uiNumToUse = uiDefNum3 + uiLoop;
		
		f_sprintf( szName, "c_element%u", (unsigned)uiLoop);
		if( RC_BAD( rc = m_pDb->createElementDef(
			NULL, szName, XFLM_NODATA_TYPE, &uiNumToUse)))
		{
			MAKE_FLM_ERROR_STRING( "createElementDef failed", m_szDetails, rc);
			goto Exit;
		}
	}
	
	// Create a root node, with two child nodes.
	
	if( RC_BAD( rc = m_pDb->createRootElement( XFLM_DATA_COLLECTION,
		ELM_ELEMENT_TAG, &pRootElement)))
	{
		MAKE_FLM_ERROR_STRING( "createRootElement failed", m_szDetails, rc);
		goto Exit;
	}
	
	if( RC_BAD( rc = pRootElement->createNode( m_pDb,
		ELEMENT_NODE, uiDefNum1, XFLM_FIRST_CHILD, &pUniqueElement, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "createNode failed", m_szDetails, rc);
		goto Exit;
	}
	
	if( RC_BAD( rc = pRootElement->createNode( m_pDb,
		ELEMENT_NODE, uiDefNum2, XFLM_LAST_CHILD, &pNonUniqueElement, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "createNode failed", m_szDetails, rc);
		goto Exit;
	}
	
	// Under each child node create 200 nodes, with a unique ID
	
	for (uiLoop = 0; uiLoop < 200; uiLoop++)
	{
		// Should not allow a data node.
		
		if( RC_BAD( rc = pUniqueElement->createNode( m_pDb,
			DATA_NODE, uiDefNum3 + uiLoop, XFLM_LAST_CHILD, &pElement, NULL)))
		{
			if (rc == NE_XFLM_BAD_DATA_TYPE || rc == NE_XFLM_DOM_INVALID_CHILD_TYPE)
			{
				rc = NE_XFLM_OK;
			}
			else
			{
				MAKE_FLM_ERROR_STRING( "createNode failed", m_szDetails, rc);
				goto Exit;
			}
		}
		else
		{
			// Should not have succeeded.
			
			endTest( "FAIL");
			goto Finish_Test;
		}
		
		// 1st add should work, 2nd add under unique parent should fail
		
		if( RC_BAD( rc = pUniqueElement->createNode( m_pDb,
			ELEMENT_NODE, uiDefNum3 + uiLoop, XFLM_LAST_CHILD, &pElement, NULL)))
		{
			MAKE_FLM_ERROR_STRING( "createNode failed", m_szDetails, rc);
			goto Exit;
		}
		
		if( RC_BAD( rc = pUniqueElement->createNode( m_pDb,
			ELEMENT_NODE, uiDefNum3 + uiLoop, XFLM_LAST_CHILD, &pElement, NULL)))
		{
			if (rc == NE_XFLM_DOM_DUPLICATE_ELEMENT)
			{
				rc = NE_XFLM_OK;
			}
			else
			{
				MAKE_FLM_ERROR_STRING( "createNode failed", m_szDetails, rc);
				goto Exit;
			}
		}
		else
		{
			// Should not have succeeded.
			
			endTest("FAIL");
			goto Finish_Test;
		}
		
		// Should be able to do it twice under non-unique parent
		
		if( RC_BAD( rc = pNonUniqueElement->createNode( m_pDb,
			ELEMENT_NODE, uiDefNum3 + uiLoop, XFLM_LAST_CHILD, &pElement, NULL)))
		{
			MAKE_FLM_ERROR_STRING( "createNode failed", m_szDetails, rc);
			goto Exit;
		}
		
		if( RC_BAD( rc = pNonUniqueElement->createNode( m_pDb,
			ELEMENT_NODE, uiDefNum3 + uiLoop, XFLM_LAST_CHILD, &pElement, NULL)))
		{
			MAKE_FLM_ERROR_STRING( "createNode failed", m_szDetails, rc);
			goto Exit;
		}
		
		// Should not allow a data node.
		
		if( RC_BAD( rc = pNonUniqueElement->createNode( m_pDb,
			DATA_NODE, uiDefNum3 + uiLoop, XFLM_LAST_CHILD, &pElement, NULL)))
		{
			if( rc == NE_XFLM_BAD_DATA_TYPE ||
				 rc == NE_XFLM_DOM_INVALID_CHILD_TYPE)
			{
				rc = NE_XFLM_OK;
			}
			else
			{
				MAKE_FLM_ERROR_STRING( "createNode failed", m_szDetails, rc);
				goto Exit;
			}
		}
		else
		{
			// Should not have succeeded.
			
			endTest("FAIL");
			goto Finish_Test;
		}
	}
	
	// Go through the loop and get all 200, then delete, and get again

	for( uiLoop = 0; uiLoop < 200; uiLoop++)
	{
		// 1st get should work, then delete, 2nd get should fail
		
		if( RC_BAD( rc = pUniqueElement->getChildElement( m_pDb,
			uiDefNum3 + uiLoop, &pElement)))
		{
			MAKE_FLM_ERROR_STRING( "getChildElement failed", m_szDetails, rc);
			goto Exit;
		}
		
		if (RC_BAD( rc = pElement->deleteNode( m_pDb)))
		{
			MAKE_FLM_ERROR_STRING( "deleteNode failed", m_szDetails, rc);
			goto Exit;
		}
		
		if( RC_BAD( rc = pUniqueElement->getChildElement( m_pDb,
			uiDefNum3 + uiLoop, &pElement)))
		{
			if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				rc = NE_XFLM_OK;
			}
			else
			{
				MAKE_FLM_ERROR_STRING( "getChildElement failed", m_szDetails, rc);
				goto Exit;
			}
		}
		else
		{
			// Should not have succeeded.
			
			endTest("FAIL");
			goto Finish_Test;
		}
		
		// 1st get should work, then delete, 2nd should also work
		
		if( RC_BAD( rc = pNonUniqueElement->getChildElement( m_pDb,
			uiDefNum3 + uiLoop, &pElement)))
		{
			MAKE_FLM_ERROR_STRING( "getChildElement failed", m_szDetails, rc);
			goto Exit;
		}
		
		if (RC_BAD( rc = pElement->deleteNode( m_pDb)))
		{
			MAKE_FLM_ERROR_STRING( "deleteNode failed", m_szDetails, rc);
			goto Exit;
		}
		
		if( RC_BAD( rc = pNonUniqueElement->getChildElement( m_pDb,
			uiDefNum3 + uiLoop, &pElement)))
		{
			MAKE_FLM_ERROR_STRING( "getChildElement failed", m_szDetails, rc);
			goto Exit;
		}
	}
	
	// Commit the transaction

	bStartedTrans = FALSE;
	if( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_FLM_ERROR_STRING( "transCommit failed", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

Finish_Test:

	if( bStartedTrans)
	{
		m_pDb->transAbort();
		bStartedTrans = FALSE;
	}

	// Close the database.
	
	m_pDb->Release();
	m_pDb = NULL;	

	// Re-open the database
	
	beginTest(	
		"backup",
		"backup the database",
		"Self-explanatory",
		"No Additional Details.");

	if( RC_BAD( rc = m_pDbSystem->dbOpen( DB_NAME_STR, NULL, NULL, 
		NULL, FALSE, &m_pDb)))
	{
		MAKE_FLM_ERROR_STRING( "dbOpen failed", m_szDetails, rc);
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

	endTest("PASS");

	// Close the database again
	
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

	if( RC_BAD( rc = m_pDbSystem->dbRemove( DB_NAME_STR, NULL, NULL, TRUE)))
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

	beginTest(	
		"backup (w/ password)",
		"backup the database using a password",
		"Self-explanatory",
		"No Additional Details.");

	if( RC_BAD( rc = m_pDbSystem->dbOpen( DB_NAME_STR, NULL, NULL, 
		NULL, FALSE, &m_pDb)))
	{
		MAKE_FLM_ERROR_STRING( "dbOpen failed", m_szDetails, rc);
		goto Exit;
	}

	// Backup the database

	if( RC_BAD( rc = m_pDb->backupBegin( XFLM_FULL_BACKUP,
		XFLM_READ_TRANS, 0, &pBackup)))
	{
		MAKE_FLM_ERROR_STRING( "backupBegin failed", m_szDetails, rc);
		goto Exit;
	}

	 rc = pBackup->backup( BACKUP_NAME_STR,
		BACKUP_PASSWORD_STR, NULL, NULL, NULL);

#ifdef FLM_USE_NICI
	// The criteria for a successful test varies depending on whether we're
	// compiled with NICI or not...
	if( RC_BAD( rc))
#else
	if (rc != NE_XFLM_ENCRYPTION_UNAVAILABLE)
#endif
	{
		MAKE_FLM_ERROR_STRING( "backup failed", m_szDetails, rc);
		goto Exit;
	}

	pBackup->Release();
	pBackup = NULL;
	
	m_pDb->Release();
	m_pDb = NULL;

	endTest("PASS");

#ifdef FLM_USE_NICI	
							// No point in trying to restore with a password when we
							// don't have NICI because there's no way a password
							// based backup could work...
							// Restore the database (using the password)
	beginTest( 			
		"dbRestore (w/ password)",
		"restore the database from the backup we just made",
		"Self-explanatory",
		"No Additional Details.");

	if( RC_BAD( rc = m_pDbSystem->closeUnusedFiles( 0)))
	{
		MAKE_FLM_ERROR_STRING( "closeUnusedFiles failed", m_szDetails, rc);
		goto Exit;
	}

	// Remove the database
	
	if( RC_BAD( rc = m_pDbSystem->dbRemove( DB_NAME_STR, NULL, NULL, TRUE)))
	{
		MAKE_FLM_ERROR_STRING( "dbRemove failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDbSystem->dbRestore( DB_NAME_STR,
		NULL, NULL, BACKUP_NAME_STR, BACKUP_PASSWORD_STR, NULL, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "dbRestore failed", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");
	
	// Wrap the database key back in the server key
	
	beginTest(
		"wrapKey",
		"wrap the database key in the NICI server key",
		"Self-explanatory",
		"No Additional Details.");

	if( RC_BAD( rc = m_pDbSystem->dbOpen( DB_NAME_STR, NULL, NULL, &m_pDb)))
	{
		MAKE_FLM_ERROR_STRING( "dbOpen failed", m_szDetails, rc);
		goto Exit;
	}

	if (RC_BAD( rc = m_pDb->wrapKey()))
	{
		MAKE_FLM_ERROR_STRING( "wrapKey failed", m_szDetails, rc);
		goto Exit;
	}

	m_pDb->Release();
	m_pDb = NULL;	

	endTest("PASS");
#endif // FLM_USE_NICI

	// Rename the database
	
	beginTest(
		"dbRename",
		"rename the database",
		"Self-explanatory",
		"No Additional Details.");

	if( RC_BAD( rc = m_pDbSystem->dbRename( DB_NAME_STR, NULL, NULL,
		NEW_NAME_STR, TRUE, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "dbRename failed", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

	// Copy the database
	
	beginTest(
		"dbCopy",
		"copy the database",
		"Self-explanatory",
		"No Additional Details.");

	if( RC_BAD( rc = m_pDbSystem->dbCopy( NEW_NAME_STR, NULL, NULL,
		DB_NAME_STR, NULL, NULL, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "dbCopy failed", m_szDetails, rc);
		goto Exit;
	}

	// Remove both databases

	if( RC_BAD( rc = m_pDbSystem->dbRemove( DB_NAME_STR, NULL, NULL, TRUE)))
	{
		MAKE_FLM_ERROR_STRING( "dbRemove failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDbSystem->dbRemove( NEW_NAME_STR, NULL, NULL, TRUE)))
	{
		MAKE_FLM_ERROR_STRING( "dbRemove failed", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

Exit:

	if( RC_BAD( rc))
	{
		endTest("FAIL");
	}

	if( bStartedTrans)
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
	
	if( pUniqueElement)
	{
		pUniqueElement->Release();
	}
	
	if( pNonUniqueElement)
	{
		pNonUniqueElement->Release();
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

	if( pTmpNode)
	{
		pTmpNode->Release();
	}
	
	if( pucLargeBuf)
	{
		f_free( &pucLargeBuf);
	}

	if( RC_BAD( rc))
	{
		f_sprintf( szMsgBuf, "Error 0x%04X\n", (unsigned)rc);
		display( szMsgBuf);
		log( szMsgBuf);

		return( rc);
	}
	else
	{
		// Shut down the FLAIM database engine.  This call must be made
		// even if the init call failed.  No more FLAIM calls should be made
		// by the application.
	
		shutdownTestState( DB_NAME_STR, bDibCreated);
		return( NE_XFLM_OK);
	}
}
