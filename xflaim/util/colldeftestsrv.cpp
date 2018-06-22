//------------------------------------------------------------------------------
// Desc:	Collection definition unit tests
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

static FLMBYTE gv_ucLargeBuffer[8192];

static void buildLargeBuffer( void);

/****************************************************************************
Desc:
****************************************************************************/
class ICollDefTestImpl : public TestBase
{
public:

	const char * getName();
	
	RCODE execute( void);

private:

	RCODE importEncDefs( void);

	RCODE importToCollection( 
		FLMUINT			uiDictNum);

	RCODE verifyDocument(
		FLMUINT			uiDictNum);

	RCODE createDODocument(
		FLMUINT			uiCollNum,
		FLMUINT64 *		pui64NodeId);

	RCODE verifyDODocument(
		FLMUINT			uiCollNum,
		FLMUINT64		ui64NodeId);

	FLMUINT				m_uiAESDef;
	FLMUINT				m_uiDES3Def;
};

/****************************************************************************
Desc:
****************************************************************************/
RCODE getTest( 
	IFlmTest **		ppTest)
{
	RCODE		rc = NE_XFLM_OK;

	if( (*ppTest = f_new ICollDefTestImpl) == NULL)
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
const char * ICollDefTestImpl::getName( void)
{
	return( "Collection Definition Test");
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE ICollDefTestImpl::importEncDefs( void)
{
	RCODE			rc = NE_XFLM_OK;
	const char *ppszEncDefs[] = {XFLM_ENC_AES_OPTION_STR, XFLM_ENC_DES3_OPTION_STR};
	FLMUINT		puiEncDef[ 2];
	char			szEncDef[ 200];
	FLMUINT		uiLoop = 0;

	for( 
		uiLoop = 0; 
		uiLoop < sizeof(ppszEncDefs)/sizeof(ppszEncDefs[0]); 
		uiLoop++)
	{
		f_sprintf( 
			szEncDef, 
			"<xflaim:EncDef "
			"xmlns:xs=\"http://www.w3.org/2001/XMLSchema\" "
			"xmlns:xflaim=\"http://www.novell.com/XMLDatabase/Schema\" "
			"xflaim:name=\"%s definition\" "
			"xflaim:type=\"%s\" />", 
			ppszEncDefs[uiLoop], 
			ppszEncDefs[uiLoop]);

		if ( RC_BAD( rc = importBuffer( szEncDef, XFLM_DICT_COLLECTION)))
		{
			MAKE_ERROR_STRING( "importBuffer failed.", m_szDetails, rc);
			goto Exit;
		}
		
		f_sprintf( szEncDef, "%s definition", ppszEncDefs[uiLoop]);
		if (RC_BAD( rc = m_pDb->getEncDefId( (char *)szEncDef,
														 (FLMUINT *)&puiEncDef[ uiLoop])))
		{
			goto Exit;
		}
	}

	m_uiAESDef = puiEncDef[ 0];
	m_uiDES3Def = puiEncDef[ 1];

Exit:

	return rc;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE ICollDefTestImpl::importToCollection(
	FLMUINT			uiDictNum)
{
	RCODE		rc = NE_XFLM_OK;
	char		szBuffer[ 200];
	FLMBOOL	bTransBegun = FALSE;

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	f_sprintf( 
		szBuffer, 
		"<sample> "
			"<address>"
				"<street>\"2999 Birches Lane\"</street>"
				"<city>\"Anytown\"</city>"
				"<state>\"MyState\"</state>"
				"<zip>\"86507\"</zip>"
			"</address>"
		"</sample>");

	if ( RC_BAD( rc = importBuffer( szBuffer, uiDictNum)))
	{
		MAKE_ERROR_STRING( "importBuffer failed.", m_szDetails, rc);
		goto Exit;
	}
		
	if ( RC_BAD( rc = m_pDb->transCommit( )))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

Exit:

	if (bTransBegun)
	{
		m_pDb->transAbort();
	}

	return rc;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE ICollDefTestImpl::verifyDocument(
	FLMUINT				uiDictNum)
{
	RCODE						rc = NE_XFLM_OK;
	IF_DOMNode *			pNode = NULL;
	char						szValue[ 20];
	char						szLocalName[ 20];
	FLMUINT					uiBufSize = sizeof(szValue);
	FLMUINT					uiCharsReturned;
	
	// Get the first ( and only) document
	
	if (RC_BAD( rc = m_pDb->getFirstDocument( uiDictNum, &pNode)))
	{
		goto Exit;
	}
	
	// Verify the root node - s/b <sample>
	
	if (RC_BAD( rc = pNode->getLocalName( m_pDb, szLocalName, uiBufSize,
							&uiCharsReturned)))
	{
		goto Exit;
	}
	
	if (uiCharsReturned != f_strlen("sample"))
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}
	
	if (f_strcmp( szLocalName, "sample"))
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}

	// Verify the child node - s/b <address>
	
	if (RC_BAD( rc = pNode->getFirstChild( m_pDb, &pNode)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = pNode->getLocalName( m_pDb, szLocalName, uiBufSize,
							&uiCharsReturned)))
	{
		goto Exit;
	}
	
	if (uiCharsReturned != f_strlen("address"))
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}
	
	if (f_strcmp( szLocalName, "address"))
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}

	// Verify the child node - s/b <street>
	
	if (RC_BAD( rc = pNode->getFirstChild( m_pDb, &pNode)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = pNode->getLocalName( m_pDb, szLocalName, uiBufSize,
							&uiCharsReturned)))
	{
		goto Exit;
	}
	
	if (uiCharsReturned != f_strlen("street"))
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}
	
	if (f_strcmp( szLocalName, "street"))
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}
	
	// Verify the value.
	
	if (RC_BAD( rc = pNode->getUTF8( m_pDb, (FLMBYTE *)szValue, uiBufSize,
		0, uiBufSize, &uiCharsReturned)))
	{
		goto Exit;
	}
	
	if (uiCharsReturned != f_strlen("\"2999 Birches Lane\""))
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}
	
	if (f_strcmp( szValue, "\"2999 Birches Lane\""))
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}

	// Verify the sibling node - s/b <city>
	
	if (RC_BAD( rc = pNode->getNextSibling( m_pDb, &pNode)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = pNode->getLocalName( m_pDb, szLocalName, uiBufSize,
							&uiCharsReturned)))
	{
		goto Exit;
	}
	
	if( uiCharsReturned != f_strlen("city"))
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}
	
	if( f_strcmp( szLocalName, "city"))
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}
	
	// Verify the value.
	
	if (RC_BAD( rc = pNode->getUTF8( m_pDb, (FLMBYTE *)szValue, uiBufSize,
		0, uiBufSize, &uiCharsReturned)))
	{
		goto Exit;
	}
	
	if (uiCharsReturned != f_strlen("\"AnyTown\""))
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}
	
	if (f_strcmp( szValue, "\"Anytown\""))
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}

	// Verify the sibling node - s/b <state>
	
	if (RC_BAD( rc = pNode->getNextSibling( m_pDb, &pNode)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = pNode->getLocalName( m_pDb, szLocalName, uiBufSize,
							&uiCharsReturned)))
	{
		goto Exit;
	}
	
	if (uiCharsReturned != f_strlen("state"))
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}
	
	if (f_strcmp( szLocalName, "state"))
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}
	
	// Verify the value.
	
	if (RC_BAD( rc = pNode->getUTF8( m_pDb, (FLMBYTE *)szValue, uiBufSize,
		0, uiBufSize, &uiCharsReturned)))
	{
		goto Exit;
	}
	
	if (uiCharsReturned != f_strlen("\"MyState\""))
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}
	
	if (f_strcmp( szValue, "\"MyState\""))
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}

	// Verify the sibling node - s/b <zip>
	
	if (RC_BAD( rc = pNode->getNextSibling( m_pDb, &pNode)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = pNode->getLocalName( m_pDb, szLocalName, uiBufSize,
							&uiCharsReturned)))
	{
		goto Exit;
	}
	
	if (uiCharsReturned != f_strlen("zip"))
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}
	
	if (f_strcmp( szLocalName, "zip"))
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}
	
	// Verify the value.
	
	if (RC_BAD( rc = pNode->getUTF8( m_pDb, (FLMBYTE *)szValue, uiBufSize,
		0, uiBufSize, &uiCharsReturned)))
	{
		goto Exit;
	}
	
	if (uiCharsReturned != f_strlen("\"86507\""))
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}
	
	if (f_strcmp( szValue, "\"86507\""))
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}

Exit:

	if (pNode)
	{
		pNode->Release();
	}

	return rc;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE ICollDefTestImpl::createDODocument(
	FLMUINT				uiCollNum,
	FLMUINT64 *			pui64NodeId)
{
	RCODE						rc = NE_XFLM_OK;
	IF_DOMNode *			pNode = NULL;
	IF_DOMNode *			pAttr = NULL;
	FLMUINT					uiBuffSize = sizeof(gv_ucLargeBuffer);
	FLMBOOL					bTransBegun = FALSE;
	
	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	// Create a new document
	
	if (RC_BAD( rc = m_pDb->createRootElement( uiCollNum, ELM_ELEMENT_TAG, 
		&pNode, pui64NodeId)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = pNode->createAttribute( m_pDb, 
		ATTR_ENCRYPTION_KEY_TAG, &pAttr)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = pAttr->setBinary( m_pDb, gv_ucLargeBuffer, 
		uiBuffSize, TRUE, 0)))
	{
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

Exit:

	if (pAttr)
	{
		pAttr->Release();
	}

	if (pNode)
	{
		pNode->Release();
	}

	if (bTransBegun)
	{
		m_pDb->transAbort();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE ICollDefTestImpl::verifyDODocument(
	FLMUINT				uiCollNum,
	FLMUINT64			ui64NodeId)
{
	RCODE						rc = NE_XFLM_OK;
	IF_DOMNode *			pNode = NULL;
	IF_DOMNode *			pAttr = NULL;
	FLMBYTE *				pucBuff = NULL;
	FLMUINT					uiLength;
	FLMUINT					uiLengthRV;
	FLMUINT					uiLoop;
	
	// Create a new document
	
	if (RC_BAD( rc = m_pDb->getDocument( uiCollNum, XFLM_EXACT, 
		ui64NodeId, &pNode)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = pNode->getAttribute( m_pDb, 
		ATTR_ENCRYPTION_KEY_TAG, &pAttr)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = pAttr->getDataLength(m_pDb, &uiLength)))
	{
		goto Exit;
	}
	
	if (uiLength != sizeof( gv_ucLargeBuffer))
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}
	
	if (RC_BAD( rc = f_alloc( sizeof( gv_ucLargeBuffer), &pucBuff)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = pAttr->getBinary( m_pDb, pucBuff, 0, uiLength, &uiLengthRV)))
	{
		goto Exit;
	}
	
	if (uiLength != uiLengthRV)
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}
	
	for (uiLoop = 0; uiLoop < sizeof( gv_ucLargeBuffer); uiLoop++)
	{
		if (gv_ucLargeBuffer[ uiLoop] != pucBuff[ uiLoop])
		{
			rc = RC_SET( NE_XFLM_FAILURE);
			goto Exit;
		}
	}

Exit:

	if (pucBuff)
	{
		f_free( &pucBuff);
	}

	if (pAttr)
	{
		pAttr->Release();
	}

	if (pNode)
	{
		pNode->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE ICollDefTestImpl::execute( void)
{
	RCODE						rc = NE_XFLM_OK;
	FLMBOOL					bDibCreated = FALSE;
	FLMUINT					uiAESDictNum = 0;
	FLMUINT					uiDES3DictNum = 0;
	FLMUINT64				ui64AESDoc;
	FLMUINT64				ui64DES3Doc;
	IF_DOMNode *			pNode				= NULL;
	IF_DOMNode *			pAttr				= NULL;
	IF_DOMNode *			pCollDef			= NULL;
	IF_DOMNode *			pTmpNode			= NULL;
	FLMBOOL					bTransBegun = FALSE;

	beginTest(	
		"Encrypted Collection Definition Test", 
		"Define encrypted collections tests.",
		"1) create a database 2) create required encryption definitions "
		"3) create some encrypted collection definitions.",
		"none");

	f_strcpy (m_szDetails, "No Additional Info.");

	if ( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		MAKE_ERROR_STRING( "Failed to initialize test state.", m_szDetails, rc);
		goto Exit;
	}

#ifdef FLM_USE_NICI
	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	if ( RC_BAD( rc = importEncDefs()))
	{
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->transCommit( )))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;
#endif

	// Start an update transaction

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	bDibCreated = TRUE;

	// Create a collection def

	/*
	<Collection name="Encrypted Collection AES" encId=1/>
	*/

	if ( RC_BAD( rc = m_pDb->createRootElement(
		XFLM_DICT_COLLECTION,
		ELM_COLLECTION_TAG,
		&pCollDef)))
	{
		MAKE_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pCollDef->createAttribute(
		m_pDb,
		ATTR_NAME_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb,
		(FLMBYTE *)"Encrypted Collection AES")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

#ifdef FLM_USE_NICI
	if ( RC_BAD( rc = pCollDef->createAttribute(
		m_pDb,
		ATTR_ENCRYPTION_ID_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUINT(
		m_pDb,
		m_uiAESDef)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}
#endif

	if ( RC_BAD( rc = m_pDb->documentDone( pCollDef)))
	{
		MAKE_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}
	

	if ( RC_BAD( rc = pCollDef->getAttribute(
		m_pDb,
		ATTR_DICT_NUMBER_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "getAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	// Get the dictionary number (used by later tests)

	if ( RC_BAD( rc = pAttr->getUINT(
		m_pDb,
		&uiAESDictNum)))
	{
		MAKE_ERROR_STRING( "getUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->transCommit( )))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	// Create an encrypted collection definition for DES3 encryption
	
	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	/*
	<Collection name="Encrypted Collection DES3" encId=2/>
	*/

	if ( RC_BAD( rc = m_pDb->createRootElement(
		XFLM_DICT_COLLECTION,
		ELM_COLLECTION_TAG,
		&pCollDef)))
	{
		MAKE_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pCollDef->createAttribute(
		m_pDb,
		ATTR_NAME_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUTF8(
		m_pDb,
		(FLMBYTE *)"Encrypted Collection DES3")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

#ifdef FLM_USE_NICI
	if ( RC_BAD( rc = pCollDef->createAttribute(
		m_pDb,
		ATTR_ENCRYPTION_ID_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUINT(
		m_pDb,
		m_uiDES3Def)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}
#endif

	if ( RC_BAD( rc = m_pDb->documentDone( pCollDef)))
	{
		MAKE_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pCollDef->getAttribute(
		m_pDb,
		ATTR_DICT_NUMBER_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "getAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	// Get the dictionary number (used by later tests)

	if ( RC_BAD( rc = pAttr->getUINT(
		m_pDb,
		&uiDES3DictNum)))
	{
		MAKE_ERROR_STRING( "getUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->transCommit( )))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	endTest("PASS");

	beginTest(	"Import to Encrypted Collections Tests",
					"Verify that we can import data into an encrypted collection",
					"Import a document into each of the previously defined collections."
				    " Retrieve them and verify them against the original document.",
					 "none");

	if (RC_BAD( rc = importToCollection( uiAESDictNum)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = importToCollection( uiDES3DictNum)))
	{
		goto Exit;
	}

	// Close the database, reopen it and verify the documents.
	
	if( pNode)
	{
		pNode->Release();
		pNode = NULL;
	}
	
	if( pAttr)
	{
		pAttr->Release();
		pAttr = NULL;
	}
	
	if( pCollDef)
	{
		pCollDef->Release();
		pCollDef = NULL;
	}
	
	if( pTmpNode)
	{
		pTmpNode->Release();
		pTmpNode = NULL;
	}

	if( m_pDb->Release())
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}
	
	m_pDb = NULL;

	// Open the database.
	
	if( RC_BAD( rc = m_pDbSystem->dbOpen( DB_NAME_STR, NULL, NULL, NULL, 
		FALSE, &m_pDb)))
	{
		goto Exit;
	}
	
	// Verify the documents at a time and compare them.
	
	if (RC_BAD( rc = verifyDocument( uiAESDictNum)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = verifyDocument( uiDES3DictNum)))
	{
		goto Exit;
	}

	endTest( "PASS");

	beginTest( "Encrypted Data Only Block Test",
				  "Verify that we can encrypt the data only blocks that are part of an encrypted collection",
				  "Create a document with a very large value, then retrieve and verify it.",
				  "none");

	buildLargeBuffer();
	
	if (RC_BAD( rc = createDODocument( uiAESDictNum, &ui64AESDoc)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = createDODocument( uiDES3DictNum, &ui64DES3Doc)))
	{
		goto Exit;
	}

	// Close the database, reopen it and verify the documents.
	
	if ( m_pDb->Release())
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}
	
	m_pDb = NULL;

	// Open the database.
	
	if( RC_BAD( rc = m_pDbSystem->dbOpen( DB_NAME_STR, NULL, NULL,
		NULL, FALSE, &m_pDb)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = verifyDODocument( uiAESDictNum, ui64AESDoc)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = verifyDODocument( uiDES3DictNum, ui64DES3Doc)))
	{
		goto Exit;
	}
	
	endTest( "PASS");
	
Exit:

	if ( RC_BAD( rc))
	{
		endTest( "FAIL");
	}

	if( pNode)
	{
		pNode->Release();
	}
	
	if( pAttr)
	{
		pAttr->Release();
	}
	
	if( pCollDef)
	{
		pCollDef->Release();
	}
	
	if( pTmpNode)
	{
		pTmpNode->Release();
	}

	if( bTransBegun)
	{
		if( RC_OK( rc))
		{
			m_pDb->transCommit();
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
static void buildLargeBuffer( void)
{
	FLMUINT		uiLoop;
	
	for (uiLoop = 0; uiLoop < sizeof(gv_ucLargeBuffer); uiLoop++)
	{
		gv_ucLargeBuffer[ uiLoop] = (FLMBYTE)(uiLoop & 0xFF);
	}
}
