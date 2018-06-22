//------------------------------------------------------------------------------
// Desc:	Regression tests for specific defects
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
#else
	#define DB_NAME_STR					"tst.db"
#endif

/****************************************************************************
Desc:
****************************************************************************/
class IRegressionTestImpl : public TestBase
{
public:

	const char * getName( void);
	
	RCODE execute( void);
	
	RCODE nestedElementIndexDefectTest( void);
	
	RCODE leftoverIndexKeyDefectTest( void);
	
	RCODE dataNodeDeletionDefectTest( void);
	
	RCODE truncatedValueFromStoreDefectTest( void);
	
	RCODE rflRecoverDefectTests( void);
	
	RCODE compoundIndexedValueDeleteTest( void);
};

/****************************************************************************
Desc:
****************************************************************************/
RCODE getTest( 
	IFlmTest **		ppTest)
{
	RCODE		rc = NE_XFLM_OK;

	if( (*ppTest = f_new IRegressionTestImpl) == NULL)
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
const char * IRegressionTestImpl::getName()
{
	return( "Regression Test");
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IRegressionTestImpl::execute( void)
{
	RCODE		rc = NE_XFLM_OK;
	
	if( RC_BAD( rc = rflRecoverDefectTests()))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = leftoverIndexKeyDefectTest()))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = compoundIndexedValueDeleteTest()))
	{
		goto Exit;
	}

	if( RC_BAD( rc = dataNodeDeletionDefectTest()))
	{
		goto Exit;
	}

	if( RC_BAD( rc = truncatedValueFromStoreDefectTest()))
	{
		goto Exit;
	}

	/*
	if ( RC_BAD( rc = nestedElementIndexDefectTest()))
	{
		goto Exit;
	}
	*/

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IRegressionTestImpl::dataNodeDeletionDefectTest( void)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bDibCreated = FALSE;
	FLMBOOL			bTransBegun = FALSE;
	FLMUINT			uiTextDef = 0;
	IF_DOMNode *	pTextNode = NULL;

	beginTest(
		"Data Node Deletion Defect Test",
		"",
		"Self-explanatory",
		"");

	if ( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		goto Exit;
	}
	bDibCreated = TRUE;

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		goto Exit;
	}
	bTransBegun = TRUE;

	if ( RC_BAD( rc = m_pDb->createElementDef( 
		NULL, "text_val", XFLM_TEXT_TYPE, &uiTextDef)))
	{
		MAKE_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->createRootElement(
		XFLM_DATA_COLLECTION,
		uiTextDef,
		&pTextNode)))
	{
		MAKE_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
		goto Exit;
	}

	// Stream in the value so a data node will be created

	if ( RC_BAD( rc = pTextNode->setUTF8( m_pDb, 
		(FLMBYTE *)"Streamed ", 9, FALSE)))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pTextNode->setUTF8( m_pDb, 
		(FLMBYTE *)"value", 5, TRUE)))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	// delete the data node within this same transaction

	if ( RC_BAD( rc = pTextNode->deleteChildren( m_pDb)))
	{
		MAKE_ERROR_STRING( "deleteNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->createRootElement(
		XFLM_DATA_COLLECTION,
		uiTextDef,
		&pTextNode)))
	{
		MAKE_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
		goto Exit;
	}

	// Stream in the value so a data node will be created

	if ( RC_BAD( rc = pTextNode->setUTF8( m_pDb, 
		(FLMBYTE *)"Streamed ", 9, FALSE)))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pTextNode->setUTF8( m_pDb, 
		(FLMBYTE *)"value2", 6, TRUE)))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	// delete the data node within a different transaction

	if ( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	if ( RC_BAD( rc = pTextNode->deleteChildren( m_pDb)))
	{
		MAKE_ERROR_STRING( "deleteNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	endTest("PASS");

Exit:
	
	if ( pTextNode)
	{
		pTextNode->Release();
	}

	if ( RC_BAD( rc))
	{
		endTest("FAIL");
	}

	if ( bTransBegun)
	{
		if ( RC_OK( rc))
		{
			rc = m_pDb->transCommit();
		}
		else
		{
			m_pDb->transAbort();
		}
	}

	shutdownTestState( DB_NAME_STR, bDibCreated);
	return rc;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IRegressionTestImpl::leftoverIndexKeyDefectTest( void)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DataVector *	pSearchKey = NULL;
	IF_DOMNode *		pNode = NULL;
	FLMBOOL				bDibCreated = FALSE;
	FLMBOOL				bTransBegun = FALSE;
	IF_DOMNode *		pDoc = NULL;
	IF_DOMNode *		pAttr = NULL;
	IF_DOMNode *		pIndex = NULL;
	IF_DOMNode *		pComp = NULL;
	FLMUINT				uiIxValName = 0;
	char					szBuf[100];
	FLMUINT				uiTmp;

	beginTest(
		"Unlink Node Bad Ix Key Defect Test",
		"",
		"Self-explanatory",
		"");

	if ( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		goto Exit;
	}
	bDibCreated = TRUE;

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	if ( RC_BAD( rc = m_pDb->createElementDef( 
		NULL, "indexed val", XFLM_TEXT_TYPE, &uiIxValName)))
	{
		MAKE_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->createDocument( 
		XFLM_DATA_COLLECTION,
		&pDoc)))
	{
		MAKE_ERROR_STRING( "createDocument failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pDoc->createNode( 
		m_pDb, ELEMENT_NODE, uiIxValName, XFLM_FIRST_CHILD,
		&pNode)))
	{
		MAKE_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->documentDone( pDoc)))
	{
		MAKE_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}

	// create an index definition that references the elem we made

	if ( RC_BAD( rc = m_pDb->createRootElement(
		XFLM_DICT_COLLECTION,
		ELM_INDEX_TAG,
		&pIndex)))
	{
		MAKE_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pIndex->createAttribute( m_pDb, ATTR_NAME_TAG, &pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, 
		(FLMBYTE *)"index_2")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pIndex->createAttribute( m_pDb, ATTR_DICT_NUMBER_TAG, &pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUINT( m_pDb, 123)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	// xflaim:ElementPath must have one or more xflaim:ElementComponent
	// or one or more xflaim:AttributeComponent sub-elements

	if ( RC_BAD( rc = pIndex->createNode(
		m_pDb,
		ELEMENT_NODE,
		ELM_ELEMENT_COMPONENT_TAG,
		XFLM_FIRST_CHILD,
		&pComp)))
	{
		MAKE_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pComp->createAttribute(
		m_pDb,
		ATTR_DICT_NUMBER_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUINT( m_pDb, uiIxValName)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pComp->createAttribute(
		m_pDb,
		ATTR_KEY_COMPONENT_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUINT( m_pDb, 1)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pComp->createAttribute(
		m_pDb,
		ATTR_REQUIRED_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;

	}

	if ( RC_BAD( rc = pAttr->setUINT( m_pDb, 1)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	if (RC_BAD( rc = m_pDb->documentDone( pIndex)))
	{
		MAKE_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDbSystem->createIFDataVector( &pSearchKey)))
	{
		MAKE_FLM_ERROR_STRING( "createIFDataVector failed", m_szDetails, rc);
		goto Exit;
	}

	// we should now have one empty key for this index

	if ( RC_BAD( rc = m_pDb->keyRetrieve( 123, NULL, XFLM_FIRST, pSearchKey)))
	{
		MAKE_ERROR_STRING( "keyRetrieve failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( pSearchKey->getDataLength( 0) != 0)
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		MAKE_ERROR_STRING( "Invalid key found.", m_szDetails, rc);
		goto Exit;
	}

	// now set a value

	if ( RC_BAD( rc = pNode->setUTF8( m_pDb,
		(FLMBYTE *)"new value")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->documentDone( pDoc)))
	{
		MAKE_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}

	// we should have a key with the new value in it

	if ( RC_BAD( rc = m_pDb->keyRetrieve( 123, NULL, XFLM_FIRST, pSearchKey)))
	{
		MAKE_ERROR_STRING( "keyRetrieve failed.", m_szDetails, rc);
		goto Exit;
	}

	uiTmp = sizeof( szBuf);
	if ( RC_BAD( rc = pSearchKey->getUTF8( 0, (FLMBYTE *)szBuf, &uiTmp)))
	{
		MAKE_ERROR_STRING( "getUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( f_strcmp( szBuf, "new value") != 0)
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		MAKE_ERROR_STRING( "invalid key found.", m_szDetails, rc);
		goto Exit;
	}

	// now delete pNode

	if ( RC_BAD( rc = pNode->deleteNode( m_pDb)))
	{
		MAKE_ERROR_STRING( "deleteNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->documentDone( pDoc)))
	{
		MAKE_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}

	// there should be no key now

	if ( RC_OK( rc = m_pDb->keyRetrieve( 123, NULL, XFLM_FIRST, pSearchKey)))
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		MAKE_ERROR_STRING( "Invalid key found.", m_szDetails, rc);
		goto Exit;
	}

	if ( rc != NE_XFLM_EOF_HIT)
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		MAKE_ERROR_STRING( "Unexpected rc from keyRetrieve", m_szDetails, rc);
		goto Exit;
	}

	// create the node again and stream in the data. This will force the
	// creation of a DATA_NODE to hold the data

	if ( RC_BAD( rc = pDoc->createNode( 
		m_pDb, ELEMENT_NODE, uiIxValName, XFLM_FIRST_CHILD,
		&pNode)))
	{
		MAKE_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pNode->setUTF8( m_pDb, 
		(FLMBYTE *)"Streamed ", 9, FALSE)))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pNode->setUTF8( m_pDb, 
		(FLMBYTE *)"in ", 3, FALSE)))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pNode->setUTF8( m_pDb,
		(FLMBYTE *)"value", 5, TRUE)))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->documentDone( pDoc)))
	{
		MAKE_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}

	// we should have a new key now

	if ( RC_BAD( rc = m_pDb->keyRetrieve( 123, NULL, XFLM_FIRST, pSearchKey)))
	{
		MAKE_ERROR_STRING( "keyRetrieve failed.", m_szDetails, rc);
		goto Exit;
	}

	uiTmp = sizeof( szBuf);
	if ( RC_BAD( rc = pSearchKey->getUTF8( 0, (FLMBYTE *)szBuf, &uiTmp)))
	{
		MAKE_ERROR_STRING( "getUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( f_strcmp( szBuf, "Streamed in value") != 0)
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		MAKE_ERROR_STRING( "invalid key found.", m_szDetails, rc);
		goto Exit;
	}

	// delete the node

	if ( RC_BAD( rc = pNode->deleteNode( m_pDb)))
	{
		MAKE_ERROR_STRING( "deleteNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->documentDone( pDoc)))
	{
		MAKE_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}

	// we should have no key left

	if ( RC_OK( rc = m_pDb->keyRetrieve( 123, NULL, XFLM_FIRST, pSearchKey)))
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		MAKE_ERROR_STRING( "Invalid key found.", m_szDetails, rc);
		goto Exit;
	}

	if ( rc != NE_XFLM_EOF_HIT)
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		MAKE_ERROR_STRING( "Unexpected rc from keyRetrieve", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_ERROR_STRING( "commitTrans failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	endTest("PASS");


Exit:

	if( pSearchKey)
	{
		pSearchKey->Release();
	}

	if( pNode)
	{
		pNode->Release();
	}

	if( pAttr)
	{
		pAttr->Release();
	}

	if( pIndex)
	{
		pIndex->Release();
	}

	if( pDoc)
	{
		pDoc->Release();
	}

	if( pComp)
	{
		pComp->Release();
	}

	if( RC_BAD( rc))
	{
		endTest("FAIL");
	}

	if( bTransBegun)
	{
		if( RC_OK( rc))
		{
			rc = m_pDb->transCommit();
		}
		else
		{
			m_pDb->transAbort();
		}
	}

	shutdownTestState( DB_NAME_STR, bDibCreated);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IRegressionTestImpl::nestedElementIndexDefectTest( void)
{
	RCODE					rc = NE_XFLM_OK;
	const char *		pszDoc = 
		"<foo bar=\"dogmaticperipateticaustere\">"
		" <foo>123</foo>"
		"</foo>";
	const char *		pszIndex = "<xflaim:Index "
			"xmlns:xflaim=\"http://www.novell.com/XMLDatabase/Schema\" "
			"xflaim:name=\"foo+bar_IX\" "
			"xflaim:DictNumber=\"99\">"
			"<xflaim:ElementComponent xflaim:name=\"foo\">"
				"<xflaim:AttributeComponent "
					"xflaim:name=\"bar\" "
					"xflaim:KeyComponent=\"1\" "
					"xflaim:Required=\"yes\" "
					"xflaim:type=\"string\" "
					"xflaim:IndexOn=\"value\" "
					"xflaim:Limit=\"18\"/>"
			"</xflaim:ElementComponent>"
		"</xflaim:Index>";

	IF_DataVector *	pSearchKey = NULL;
	FLMBOOL				bTransBegun = FALSE;
	FLMBOOL				bDibCreated = FALSE;

	beginTest( 	
		"Nested Element Index Defect Test",
		"",
		"Self-explanatory",
		"");

	if ( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		goto Exit;
	}
	bDibCreated = TRUE;

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "beginTrans failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	if ( RC_BAD( rc = importBuffer( pszDoc, XFLM_DATA_COLLECTION)))
	{
		goto Exit;
	}

	if ( RC_BAD( rc = importBuffer( pszIndex, XFLM_DICT_COLLECTION)))
	{
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_FLM_ERROR_STRING( "transCommit failed", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	// A key better have been generated...

	if ( RC_BAD( rc = m_pDb->keyRetrieve(
		99, NULL, XFLM_FIRST | XFLM_MATCH_DOC_ID, pSearchKey)))
	{
		MAKE_FLM_ERROR_STRING( "No index keys generated", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

Exit:

	if ( RC_BAD( rc))
	{
		endTest("FAIL");
	}

	if ( pSearchKey)
	{
		pSearchKey->Release();
	}

	if ( bTransBegun)
	{
		if ( RC_OK( rc))
		{
			rc = m_pDb->transCommit();
		}
		else
		{
			m_pDb->transAbort();
		}
	}

	shutdownTestState( DB_NAME_STR, bDibCreated);
	return rc;
}

#define TEXT_VAL		"NATIVE_VALUE"
#define UINT64_VAL	(FLM_MAX_UINT64-7)
#define INT64_VAL 	(-(FLM_MAX_INT64/2))
#define UINT_VAL		(FLM_MAX_INT-13)
#define BIN_VAL		{0x00,0x01,0x02,0x03,0x04,0x05}
#define BIN_VAL_LEN 	6

/****************************************************************************
Desc:
****************************************************************************/
RCODE IRegressionTestImpl::truncatedValueFromStoreDefectTest( void)
{
	RCODE					rc = NE_XFLM_OK;
	FLMBOOL				bTransBegun = FALSE;
	FLMBOOL				bDibCreated = FALSE;
	FLMUINT				uiRootId = 0;
	FLMUINT				uiTextValId = 0;
	FLMUINT				uiNumVal1Id = 0;
	FLMUINT				uiNumVal2Id = 0;
	FLMUINT				uiNumVal3Id = 0;
	FLMUINT				uiBinValId = 0;
	FLMBYTE				pucBinVal[] = BIN_VAL;
	FLMUINT64			ui64RootId = 0;
	FLMUINT				uiNameId = 0;
	IF_DOMNode *		pRootNode = NULL;
	IF_DOMNode *		pValNode = NULL;

	beginTest( 	
		"Truncated Value From Store Defect Test",
		"Make sure values make it back from disk intact",
		"Add values to database/close database/open database/verify values",
		"");

	if ( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		goto Exit;
	}
	bDibCreated = TRUE;

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	if ( RC_BAD( rc = m_pDb->createElementDef(
		NULL,
		"root",
		XFLM_NODATA_TYPE,
		&uiRootId)))
	{
		MAKE_ERROR_STRING( "createElementDef failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->createElementDef(
		NULL,
		"text_val",
		XFLM_TEXT_TYPE,
		&uiTextValId)))
	{
		MAKE_ERROR_STRING( "createElementDef failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->createElementDef(
		NULL,
		"uint64_val",
		XFLM_NUMBER_TYPE,
		&uiNumVal1Id)))
	{
		MAKE_ERROR_STRING( "createElementDef failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->createElementDef(
		NULL,
		"int64_val",
		XFLM_NUMBER_TYPE,
		&uiNumVal2Id)))
	{
		MAKE_ERROR_STRING( "createElementDef failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->createElementDef(
		NULL,
		"uint_val",
		XFLM_NUMBER_TYPE,
		&uiNumVal3Id)))
	{
		MAKE_ERROR_STRING( "createElementDef failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->createElementDef(
		NULL,
		"bin_val",
		XFLM_BINARY_TYPE,
		&uiBinValId)))
	{
		MAKE_ERROR_STRING( "createElementDef failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->createRootElement(
		XFLM_DATA_COLLECTION,
		uiRootId,
		&pRootNode)))
	{
		MAKE_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
		goto Exit;
	}
	
	if( RC_BAD( rc = pRootNode->getNodeId( m_pDb, &ui64RootId)))
	{
		goto Exit;
	}

	if ( RC_BAD( rc = pRootNode->createNode( m_pDb,
		ELEMENT_NODE, uiTextValId, XFLM_FIRST_CHILD,	&pValNode)))
	{
		MAKE_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pValNode->setUTF8( m_pDb, 
		(FLMBYTE *)TEXT_VAL, f_strlen( TEXT_VAL))))
	{
		MAKE_ERROR_STRING( "setUTF8 failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pRootNode->createNode( m_pDb,
		ELEMENT_NODE, uiNumVal1Id, XFLM_FIRST_CHILD,	&pValNode)))
	{
		MAKE_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pValNode->setUINT64( m_pDb, UINT64_VAL)))
	{
		MAKE_ERROR_STRING( "setUINT64 failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pRootNode->createNode( m_pDb,
		ELEMENT_NODE, uiNumVal2Id, XFLM_FIRST_CHILD,	&pValNode)))
	{
		MAKE_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pValNode->setINT64( m_pDb, INT64_VAL)))
	{
		MAKE_ERROR_STRING( "setINT64 failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pRootNode->createNode( m_pDb,
		ELEMENT_NODE, uiNumVal3Id, XFLM_FIRST_CHILD,	&pValNode)))
	{
		MAKE_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pValNode->setUINT( m_pDb, UINT_VAL)))
	{
		MAKE_ERROR_STRING( "setUINT failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pRootNode->createNode( m_pDb,
		ELEMENT_NODE, uiBinValId, XFLM_FIRST_CHILD,	&pValNode)))
	{
		MAKE_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pValNode->setBinary( m_pDb, pucBinVal, BIN_VAL_LEN, TRUE)))
	{
		MAKE_ERROR_STRING( "setBinary failed", m_szDetails, rc);
		goto Exit;
	}

	// close the database to force the values to disk

	if ( RC_BAD( rc = m_pDb->transCommit()))
	{
		goto Exit;
	}
	bTransBegun = FALSE;

	pRootNode->Release();
	pRootNode = NULL;
	pValNode->Release();
	pValNode = NULL;
	m_pDb->Release();
	m_pDb = NULL;

	if( RC_BAD( rc = m_pDbSystem->dbOpen( DB_NAME_STR, NULL, NULL, 
		NULL, FALSE, &m_pDb)))
	{
		MAKE_ERROR_STRING( "dbOpen failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_READ_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	if ( RC_BAD( rc = m_pDb->getNode(
		XFLM_DATA_COLLECTION, ui64RootId, &pRootNode)))
	{
		MAKE_ERROR_STRING( "getNode failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pRootNode->getFirstChild(
		m_pDb, &pValNode)))
	{
		MAKE_ERROR_STRING( "getFirstChild failed", m_szDetails, rc);
		goto Exit;
	}

	for(;;)
	{
		if ( RC_BAD( rc = pValNode->getNameId( m_pDb, &uiNameId)))
		{
			MAKE_ERROR_STRING( "getNameId failed", m_szDetails, rc);
			goto Exit;
		}

		if ( uiNameId == uiTextValId)
		{
			char szTemp[100];

			if ( RC_BAD( rc = pValNode->getUTF8( 
				m_pDb, (FLMBYTE *)szTemp, sizeof( szTemp), 
				0, sizeof(szTemp) -1)))
			{
				MAKE_ERROR_STRING( "getUTF8 failed", m_szDetails, rc);
				goto Exit;
			}

			if ( f_strcmp( szTemp, TEXT_VAL) != 0)
			{
				rc = RC_SET( NE_XFLM_DATA_ERROR);
				MAKE_ERROR_STRING( "Unexpected text value found", m_szDetails, rc);
				goto Exit;
			}
			// flag this name id as visited
			uiTextValId = 0;
		}
		else if ( uiNameId == uiNumVal1Id)
		{
			FLMUINT64	ui64Temp;

			if ( RC_BAD( rc = pValNode->getUINT64( m_pDb, &ui64Temp)))
			{
				MAKE_ERROR_STRING( "getUINT64 Failed", m_szDetails, rc);
				goto Exit;
			}

			if ( ui64Temp != UINT64_VAL)
			{
				rc = RC_SET( NE_XFLM_DATA_ERROR);
				MAKE_ERROR_STRING( "Unexpected uint64 value found", m_szDetails, rc);
				goto Exit;
			}
			uiNumVal1Id = 0;
		}
		else if ( uiNameId == uiNumVal2Id)
		{
			FLMINT64	i64Temp;

			if ( RC_BAD( rc = pValNode->getINT64( m_pDb, &i64Temp)))
			{
				goto Exit;
			}

			if ( i64Temp != INT64_VAL)
			{
				rc = RC_SET( NE_XFLM_DATA_ERROR);
				MAKE_ERROR_STRING( "Unexpected int64 value found", m_szDetails, rc);
				goto Exit;
			}
			uiNumVal2Id = 0;
		}
		else if ( uiNameId == uiNumVal3Id)
		{
			FLMUINT	uiTemp;

			if ( RC_BAD( rc = pValNode->getUINT( m_pDb, &uiTemp)))
			{
				MAKE_ERROR_STRING( "getUINT failed", m_szDetails, rc);
				goto Exit;
			}

			if ( uiTemp != UINT_VAL)
			{
				rc = RC_SET( NE_XFLM_DATA_ERROR);
				MAKE_ERROR_STRING( "Unexpected uint value found", m_szDetails, rc);
				goto Exit;
			}
			uiNumVal3Id = 0;
		}
		else if ( uiNameId == uiBinValId)
		{
			FLMBYTE	pucTemp[BIN_VAL_LEN];
			FLMUINT	uiTmp;

			if ( RC_BAD( rc = pValNode->getBinary( 
				m_pDb,
				pucTemp,
				0,
				sizeof(pucTemp),
				&uiTmp)))
			{
				MAKE_FLM_ERROR_STRING( "getBinary failed.", m_szDetails, rc);
				goto Exit;
			}

			if ( uiTmp != BIN_VAL_LEN ||
				f_memcmp( pucTemp, pucBinVal, uiTmp) != 0)
			{
				rc = RC_SET( NE_XFLM_DATA_ERROR);
				MAKE_ERROR_STRING( "Unexpected binary value found", m_szDetails, rc);
				goto Exit;
			}
			uiBinValId = 0;
		}
		else
		{
			rc = RC_SET( NE_XFLM_DATA_ERROR);
			MAKE_ERROR_STRING( "Unexpected node found", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = pValNode->getNextSibling( m_pDb, &pValNode)))
		{
			if ( rc != NE_XFLM_DOM_NODE_NOT_FOUND || uiTextValId ||
				uiNumVal1Id || uiNumVal2Id || uiNumVal3Id || uiBinValId)
			{
				rc = RC_SET( NE_XFLM_DATA_ERROR);
				MAKE_ERROR_STRING( "Node not found", m_szDetails, rc);
				goto Exit;
			}
			else
			{
				rc = NE_XFLM_OK;
				break;
			}
		}
	}

	endTest("PASS");

Exit:

	if( pRootNode)
	{
		pRootNode->Release();
	}

	if( pValNode)
	{
		pValNode->Release();
	}

	if( RC_BAD( rc))
	{
		endTest("FAIL");
	}

	if( bTransBegun)
	{
		if( RC_OK( rc))
		{
			rc = m_pDb->transCommit();
		}
		else
		{
			m_pDb->transAbort();
		}
	}

	shutdownTestState( DB_NAME_STR, bDibCreated);
	return( rc);
}

#if defined( FLM_NLM)
	#define BACKUP_NAME_STR				"SYS:\\TST.BAK"
	#define NEW_NAME_STR					"SYS:\\NEW.DB"
#else
	#define BACKUP_NAME_STR				"tst.bak"
	#define NEW_NAME_STR					"new.db"
#endif

/****************************************************************************
Desc:
****************************************************************************/
RCODE IRegressionTestImpl::rflRecoverDefectTests( void)
{
	RCODE						rc = NE_XFLM_OK;
	IF_Backup *				pBackup = NULL;
	IF_DOMNode *			pNode = NULL;
	FLMUINT					uiDefNum;
	FLMBOOL					bStartedTrans = FALSE;
	FLMBOOL					bDibCreated = FALSE;
	FLMUINT					uiEncDef = 0;
	IF_DataVector *		pSearchKey = NULL;
	char						szTmp[ 100];
	FLMUINT					uiTmp = 0;
	const char *			pszIndex = "<xflaim:Index "
			"xmlns:xflaim=\"http://www.novell.com/XMLDatabase/Schema\" "
			"xflaim:name=\"text_val_IX\" "
			"xflaim:DictNumber=\"99\">"
			"<xflaim:ElementComponent xflaim:name=\"text_val\" "
					"xflaim:KeyComponent=\"1\" "
					"xflaim:Required=\"yes\" "
					"xflaim:type=\"string\" "
					"xflaim:IndexOn=\"value\" />"
		"</xflaim:Index>";

	beginTest( 					
		"RFL Recover Defect Test",
		"Ensure a bug that was causing a corruption when replaying set text value "
		"operations of zero length has been fixed/Ensure a bug that was causing indexes "
		"to not be updated when recovering node update packets has been fixed",
		"",
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
		NULL, "text_val", XFLM_TEXT_TYPE, &uiDefNum)))
	{
		MAKE_FLM_ERROR_STRING( "createElementDef failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->createRootElement(
		XFLM_DATA_COLLECTION,
		uiDefNum,
		&pNode)))
	{
		MAKE_FLM_ERROR_STRING( "createRootElement failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pNode->setUTF8( m_pDb, (FLMBYTE *)"")))
	{
		MAKE_FLM_ERROR_STRING( "setUTF8 failed", m_szDetails, rc);
		goto Exit;
	}

	// Create an index on text_val

	if ( RC_BAD( rc = importBuffer( pszIndex, XFLM_DICT_COLLECTION)))
	{
		goto Exit;
	}

	// create an encryption definition

#ifdef FLM_USE_NICI
	if ( RC_BAD( rc = m_pDb->createEncDef("aes", "aes_def", 0, &uiEncDef)))
	{
		MAKE_FLM_ERROR_STRING( "createEncDef failed", m_szDetails, rc);
		goto Exit;
	}
#endif

	// modify the node value a few times with encryption. 
	// This will generate node update packets

	if ( RC_BAD( rc = pNode->setUTF8( 
		m_pDb, (FLMBYTE *)"text_val_1", 0, TRUE, uiEncDef)))
	{
		MAKE_FLM_ERROR_STRING( "setUTF8 failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pNode->setUTF8( 
		m_pDb, (FLMBYTE *)"text_val_2", 0, TRUE, uiEncDef)))
	{
		MAKE_FLM_ERROR_STRING( "setUTF8 failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pNode->setUTF8( 
		m_pDb, (FLMBYTE *)"text_val_3", 0, TRUE, uiEncDef)))
	{
		MAKE_FLM_ERROR_STRING( "setUTF8 failed", m_szDetails, rc);
		goto Exit;
	}

	// validate the key

	if ( RC_BAD( rc = m_pDbSystem->createIFDataVector( &pSearchKey)))
	{
		MAKE_FLM_ERROR_STRING( "createIFDataVector failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->keyRetrieve(
		99, NULL, XFLM_FIRST | XFLM_MATCH_DOC_ID, pSearchKey)))
	{
		MAKE_FLM_ERROR_STRING( "No index keys generated", m_szDetails, rc);
		goto Exit;
	}

	uiTmp = sizeof(szTmp);
	if ( RC_BAD( rc = pSearchKey->getUTF8( 0, (FLMBYTE *)szTmp, &uiTmp)))
	{
		MAKE_FLM_ERROR_STRING( "getUTF8 failed", m_szDetails, rc);
		goto Exit;
	}

	pSearchKey->Release();
	pSearchKey = NULL;

	if ( f_strcmp( szTmp, "text_val_3") != 0)
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		MAKE_FLM_ERROR_STRING( "Invalid key found", m_szDetails, rc);
		goto Exit;
	}

	// Commit the transaction

	if( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_FLM_ERROR_STRING( "transCommit failed", m_szDetails, rc);
		goto Exit;
	}
	bStartedTrans = FALSE;

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

	// Restore the database
	
	if( RC_BAD( rc = m_pDbSystem->dbRestore( DB_NAME_STR,
		NULL, NULL, BACKUP_NAME_STR, NULL, NULL, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "dbRestore failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDbSystem->dbOpen( DB_NAME_STR, NULL, NULL, 
		NULL, FALSE, &m_pDb)))
	{
		MAKE_FLM_ERROR_STRING( "dbOpen failed", m_szDetails, rc);
		goto Exit;
	}

	// Validate the key for the index to make sure it was restored properly

	if ( RC_BAD( rc = m_pDbSystem->createIFDataVector( &pSearchKey)))
	{
		MAKE_FLM_ERROR_STRING( "createIFDataVector failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->keyRetrieve(
		99, NULL, XFLM_FIRST | XFLM_MATCH_DOC_ID, pSearchKey)))
	{
		MAKE_FLM_ERROR_STRING( "No index keys generated", m_szDetails, rc);
		goto Exit;
	}

	uiTmp = sizeof(szTmp);
	if ( RC_BAD( rc = pSearchKey->getUTF8( 0, (FLMBYTE *)szTmp, &uiTmp)))
	{
		MAKE_FLM_ERROR_STRING( "getUTF8 failed", m_szDetails, rc);
		goto Exit;
	}

	if ( f_strcmp( szTmp, "text_val_3") != 0)
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		MAKE_FLM_ERROR_STRING( "Invalid key found", m_szDetails, rc);
		goto Exit;
	}

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

	if ( pSearchKey)
	{
		pSearchKey->Release();
	}

	if( pBackup)
	{
		pBackup->Release();
	}

	if( pNode)
	{
		pNode->Release();
	}

	shutdownTestState( DB_NAME_STR, bDibCreated);
	return rc;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IRegressionTestImpl::compoundIndexedValueDeleteTest( void)
{

	RCODE					rc = NE_XFLM_OK;
	IF_DataVector *	pSearchKey = NULL;
	FLMBOOL				bTransBegun = FALSE;
	FLMBOOL				bDibCreated = FALSE;
	IF_DOMNode *		pRoot = NULL;
	IF_DOMNode *		pNode = NULL;
	FLMUINT				uiFooId = 0;
	FLMUINT				uiBarId = 0;
	char					szBuf[100];
	FLMUINT				uiTemp = 0;
	const char *		pszIndex = 
	"<xflaim:Index "
			"xmlns:xflaim=\"http://www.novell.com/XMLDatabase/Schema\" "
			"xflaim:name=\"foo+bar_IX\" "
			"xflaim:DictNumber=\"99\">"
			"<xflaim:ElementComponent xflaim:name=\"foo\" "
				"xflaim:KeyComponent=\"1\" >"
				"<xflaim:ElementComponent "
					"xflaim:name=\"bar\" "
					"xflaim:KeyComponent=\"2\" "
					"xflaim:IndexOn=\"value\">"
				"</xflaim:ElementComponent>"
			"</xflaim:ElementComponent>"
		"</xflaim:Index>";

	beginTest( 	
		"Compound Indexed Value Delete Test",
		"",
		"Self-explanatory",
		"");

	if ( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		goto Exit;
	}
	bDibCreated = TRUE;

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "beginTrans failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	if ( RC_BAD( rc = m_pDb->createElementDef(
		NULL,
		"foo",
		XFLM_NODATA_TYPE,
		&uiFooId)))
	{
		MAKE_FLM_ERROR_STRING( "createElementDef failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->createElementDef(
		NULL,
		"bar",
		XFLM_TEXT_TYPE,
		&uiBarId)))
	{
		MAKE_FLM_ERROR_STRING( "createElementDef failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->createRootElement( 
		XFLM_DATA_COLLECTION, uiFooId, &pRoot)))
	{
		MAKE_FLM_ERROR_STRING( "createRootElement failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pRoot->createNode( 
		m_pDb, ELEMENT_NODE, uiBarId, XFLM_FIRST_CHILD, &pNode)))
	{
		MAKE_FLM_ERROR_STRING( "createNode failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pNode->setUTF8( m_pDb, 
		(FLMBYTE *)"bar ", 4, FALSE)))
	{
		MAKE_FLM_ERROR_STRING( "setUTF8 failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pNode->setUTF8( m_pDb, (FLMBYTE *)"value", 5, TRUE)))
	{
		MAKE_FLM_ERROR_STRING( "setUTF8 failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = importBuffer( pszIndex, XFLM_DICT_COLLECTION)))
	{
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_FLM_ERROR_STRING( "transCommit failed", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	if ( RC_BAD( rc = m_pDbSystem->createIFDataVector( &pSearchKey)))
	{
		MAKE_FLM_ERROR_STRING( "createIFDataVector failed", m_szDetails, rc);
		goto Exit;
	}

	// A key better have been generated...

	if ( RC_BAD( rc = m_pDb->keyRetrieve(
		99, NULL, XFLM_FIRST | XFLM_MATCH_DOC_ID, pSearchKey)))
	{
		MAKE_FLM_ERROR_STRING( "No index keys generated", m_szDetails, rc);
		goto Exit;
	}

	uiTemp = sizeof( szBuf);
	if ( RC_BAD( rc = pSearchKey->getUTF8( 1, (FLMBYTE *)szBuf, &uiTemp)))
	{
		MAKE_FLM_ERROR_STRING( "getUTF8 failed", m_szDetails, rc);
		goto Exit;
	}

	if ( f_strcmp( szBuf, "bar value") != 0)
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		MAKE_FLM_ERROR_STRING( "unexpected index key value", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	if ( RC_BAD( rc = pNode->deleteNode( m_pDb)))
	{
		MAKE_FLM_ERROR_STRING( "deleteNode failed", m_szDetails, rc);
		goto Exit;
	}

	// There better be no second component now

	if ( RC_BAD( rc = m_pDb->keyRetrieve(
		99, NULL, XFLM_FIRST | XFLM_MATCH_DOC_ID, pSearchKey)))
	{
		MAKE_FLM_ERROR_STRING( "No index keys generated", m_szDetails, rc);
		goto Exit;
	}

	uiTemp = sizeof( szBuf);
	if ( RC_OK( rc = pSearchKey->getUTF8( 1, (FLMBYTE *)szBuf, &uiTemp)))
	{
		MAKE_FLM_ERROR_STRING( "getUTF8 failed", m_szDetails, rc);
		goto Exit;
	}

	if ( rc == NE_XFLM_NOT_FOUND)
	{
		rc = NE_XFLM_OK;
	}
	else
	{
		goto Exit;
	}

	endTest("PASS");

Exit:

	if ( RC_BAD( rc))
	{
		endTest("FAIL");
	}

	if ( pSearchKey)
	{
		pSearchKey->Release();
	}

	if ( pNode)
	{
		pNode->Release();
	}

	if ( pRoot)
	{
		pRoot->Release();
	}

	if ( bTransBegun)
	{
		if ( RC_OK( rc))
		{
			rc = m_pDb->transCommit();
		}
		else
		{
			m_pDb->transAbort();
		}
	}

	shutdownTestState( DB_NAME_STR, bDibCreated);
	return rc;
}
