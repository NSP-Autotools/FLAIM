//------------------------------------------------------------------------------
// Desc:	Indexing Unit Test 2
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

#ifndef DB_NAME_STR
	#if defined( FLM_NLM)
		#define DB_NAME_STR					"SYS:\\IX2.DB"
	#else
		#define DB_NAME_STR					"ix2.db"
	#endif
#endif

/****************************************************************************
Desc:
****************************************************************************/
class IIndexTest2Impl : public TestBase
{
public:

	const char * getName( void);
	
	RCODE runSuite1( void);
	
	RCODE runSuite2( void);
	
	RCODE execute( void);
};

/****************************************************************************
Desc:
****************************************************************************/
RCODE getTest( IFlmTest ** ppTest)
{
	RCODE		rc = NE_XFLM_OK;

	if( (*ppTest = f_new IIndexTest2Impl) == NULL)
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
const char * IIndexTest2Impl::getName( void)
{
	return( "Index Test 2");
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IIndexTest2Impl::execute( void)
{
	RCODE		rc = NE_XFLM_OK;

	if( RC_BAD( rc = runSuite1()))
	{
		goto Exit;
	}

	if( RC_BAD( rc = runSuite2()))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IIndexTest2Impl::runSuite1( void)
{
	RCODE						rc = NE_XFLM_OK;
	FLMBYTE					szBuf[ 128];
	FLMBYTE 					pucTemp[256];
	FLMBOOL					bDibCreated = FALSE;
	FLMBOOL					bTransStarted = FALSE;
	IF_DOMNode *			pDocument = NULL;
	IF_DOMNode *			pAttr = NULL;
	IF_DOMNode *			pNameNode = NULL;
	IF_DOMNode *			pPrevResult = NULL;
	IF_DOMNode *			pNextResult = NULL;
	FLMUINT					uiNameDef = 0;
	FLMUINT					uiFirstDef = 0;
	FLMUINT					uiLastDef = 0;
	FLMUINT					i = 0;
	FlagSet					firstNamePlusLastNameFlags;
	IF_Query *				pQuery = NULL;
	KeyIterator *			pKeyIter = f_new KeyIterator();

	const char *			pszIndexDef1 = 
		"<xflaim:Index "
		"	xmlns:xflaim=\"http://www.novell.com/XMLDatabase/Schema\""
		"	xflaim:name=\"first+last\" "
		"	xflaim:DictNumber=\"1\"> " // HARD-CODED Dict Num for easy retrieval
		"	<xflaim:ElementComponent xflaim:name=\"name\">"
		"		<xflaim:AttributeComponent"
		"			xflaim:name=\"first\" "
		"			xflaim:IndexOn=\"value\" "
		"			xflaim:KeyComponent=\"1\" />"
		"		<xflaim:AttributeComponent"
		"			xflaim:name=\"last\" "
		"			xflaim:IndexOn=\"value\" "
		"			xflaim:KeyComponent=\"2\" />"
		"	</xflaim:ElementComponent> "
		"</xflaim:Index> ";

	// NOTE: If you make any modifications here, make sure these arrays have
	// the same number of elements.

	const char * pszFirstNames[] = {
		"Gavin", "James", "John", "Heidi", "Darcey", "Hilary"};
		
	const char * pszLastNames[] = {
		"Jensen", "Stevenson", "Smith", "Christensen", "Miller", "Andersen"};

	char ** pszFirstPlusLastNames = NULL;
	
	if( RC_BAD( rc = f_alloc( sizeof( char *) *
		sizeof( pszFirstNames) / sizeof( pszFirstNames[ 0]), &pszFirstPlusLastNames)))
	{
		goto Exit;
	}
	
	// Concatenate all pairs to
	
	for( i = 0; i < sizeof(pszFirstNames)/sizeof(pszFirstNames[0]); i++)
	{
		if( RC_BAD( rc = f_alloc( f_strlen( pszFirstNames[i]) +
			f_strlen( pszLastNames[i]) + 1, &pszFirstPlusLastNames[ i])))
		{
			goto Exit;
		}
		
		f_strcpy( pszFirstPlusLastNames[i], pszFirstNames[i]);
		f_strcat( pszFirstPlusLastNames[i], pszLastNames[i]);
	}

	firstNamePlusLastNameFlags.init( (FLMBYTE **)pszFirstPlusLastNames, i);

	beginTest( 
		"Attribute Key Removal Test",
		"Create a bunch of documents that will cause the generation of "
		"keys on our indexed attributes. Delete the documents and ensure the keys "
		"are removed as well.",
		"No further details.",
		"");

	if( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to initialize test state.", m_szDetails, rc);
		goto Exit;
	}
	bDibCreated = TRUE;

	if( RC_BAD( rc = pKeyIter->init( 1, m_pDbSystem, m_pDb)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to initialize index key iter.", 
			m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;

	if( RC_BAD( rc = m_pDb->createElementDef(
		NULL, "name", XFLM_NODATA_TYPE, &uiNameDef)))
	{
		MAKE_FLM_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->createAttributeDef(NULL, "first", XFLM_TEXT_TYPE, 
		&uiFirstDef, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to create attribute Def.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->createAttributeDef(NULL, "last", XFLM_TEXT_TYPE, 
		&uiLastDef, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to create attribute Def.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = importBuffer( pszIndexDef1, XFLM_DICT_COLLECTION)))
	{
		 goto Exit;
	}

	// Make a bunch of documents
	
	for( i = 0; i < sizeof(pszFirstNames)/sizeof( pszFirstNames[0]); i++)
	{
		if ( RC_BAD( rc = m_pDb->createDocument( XFLM_DATA_COLLECTION, 
			&pDocument)))
		{
			MAKE_FLM_ERROR_STRING( "createDocument failed.", m_szDetails, rc);
			goto Exit;
		}

		if( RC_BAD( rc = pDocument->createNode( m_pDb, ELEMENT_NODE, uiNameDef,
			XFLM_FIRST_CHILD, &pNameNode)))
		{
			MAKE_FLM_ERROR_STRING( "createNode failed.", m_szDetails, rc);
			goto Exit;
		}

		if( RC_BAD( rc = pNameNode->createAttribute( m_pDb, uiFirstDef, &pAttr)))
		{
			MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
			goto Exit;
		} 

		if( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)pszFirstNames[ i])))
		{
			MAKE_FLM_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
			goto Exit;
		}

		if( RC_BAD( rc = pNameNode->createAttribute( m_pDb, uiLastDef, &pAttr)))
		{
			MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
			goto Exit;
		} 

		if( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)pszLastNames[i])))
		{
			MAKE_FLM_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
			goto Exit;
		}

		if( RC_BAD( rc = m_pDb->documentDone( pDocument)))
		{
			MAKE_FLM_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
			goto Exit;
		}
	}

	// First make sure all expected keys were generated
	
	pKeyIter->reset();
	while( RC_OK( rc = pKeyIter->next()))
	{
		if( RC_BAD( rc = pKeyIter->getCurrentKeyVal( 0, szBuf, sizeof( szBuf),
			NULL)))
		{
			MAKE_FLM_ERROR_STRING( "Unable to get key value.", m_szDetails, rc);
			goto Exit;
		}

		f_strcpy( (char *)pucTemp, (char *)szBuf);

		if( RC_BAD( rc = pKeyIter->getCurrentKeyVal( 1, szBuf, 
			sizeof( szBuf), NULL)))
		{
			MAKE_FLM_ERROR_STRING( "Unable to get key value.", m_szDetails, rc);
			goto Exit;
		}

		f_strcat( (char *)pucTemp, (char *)szBuf);

		if( !firstNamePlusLastNameFlags.setElemFlag( pucTemp))
		{
			MAKE_FLM_ERROR_STRING( "Unexpected key found.", m_szDetails, rc);
			goto Exit;
		}
	}
	
	if ( rc == NE_XFLM_EOF_HIT)
	{
		rc = NE_XFLM_OK;
	}
	else
	{
		MAKE_FLM_ERROR_STRING( "Unexpected rc when iterating keys.", 
			m_szDetails, rc);
		goto Exit;
	}

	if( !firstNamePlusLastNameFlags.allElemFlagsSet())
	{
		rc = NE_XFLM_FAILURE;
		MAKE_FLM_ERROR_STRING( "Expected keys not generated.", 
			m_szDetails, rc);
		goto Exit;
	}

	// Query for all the documents with last == "Miller"
	
	if( RC_BAD( rc = m_pDbSystem->createIFQuery( &pQuery)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to create query object.", m_szDetails, rc);
		goto Exit;
	}

	// This query should return me the document node
	
	if( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, "@last == \"Miller\"")))
	{
		MAKE_FLM_ERROR_STRING( "Failed to set up query expression.", m_szDetails, rc);
		goto Exit;
	}

	// Delete all documents that had a last attr == "Miller"
	
	while( RC_OK( rc = pQuery->getNext( m_pDb, &pNextResult))) 
	{
		if( pPrevResult)
		{
			if( RC_BAD( rc = pPrevResult->deleteNode( m_pDb)))
			{
				MAKE_FLM_ERROR_STRING( "deleteNode failed.", m_szDetails, rc);
				goto Exit;
			}
			
			pPrevResult->Release();
		}
		
		pPrevResult = pNextResult;
		pPrevResult->AddRef();
	}

	// Delete the final node
	
	if( pPrevResult)
	{
		if( RC_BAD( rc = pPrevResult->deleteNode( m_pDb)))
		{
			MAKE_FLM_ERROR_STRING( "deleteNode failed.", m_szDetails, rc);
			goto Exit;
		}
	}

	// remove Miller and make a new cross product
	
	firstNamePlusLastNameFlags.removeElem( (FLMBYTE *)"DarceyMiller");
	firstNamePlusLastNameFlags.unsetAllFlags();

	// Reset the key iterator and make sure we don't get anything unexpected
	// First make sure all expected keys were generated
	
	pKeyIter->reset();
	while( RC_OK( rc = pKeyIter->next()))
	{
		if ( RC_BAD( rc = pKeyIter->getCurrentKeyVal( 0, szBuf, sizeof( szBuf),
			NULL)))
		{
			MAKE_FLM_ERROR_STRING( "Unable to get key value.", m_szDetails, rc);
			goto Exit;
		}

		f_strcpy( (char *)pucTemp, (char *)szBuf);

		if( RC_BAD( rc = pKeyIter->getCurrentKeyVal( 1, szBuf, 
			sizeof( szBuf), NULL)))
		{
			MAKE_FLM_ERROR_STRING( "Unable to get key value.", m_szDetails, rc);
			goto Exit;
		}

		f_strcat( (char *)pucTemp, (char *)szBuf);

		if( !firstNamePlusLastNameFlags.setElemFlag( pucTemp))
		{
			MAKE_FLM_ERROR_STRING( "Unexpected key found.", m_szDetails, rc);
			goto Exit;
		}
	}

	if( rc == NE_XFLM_EOF_HIT)
	{
		rc = NE_XFLM_OK;
	}
	else
	{
		MAKE_FLM_ERROR_STRING( "Unexpected rc when iterating keys.", 
			m_szDetails, rc);
		goto Exit;
	}

	if( !firstNamePlusLastNameFlags.allElemFlagsSet())
	{
		rc = NE_XFLM_FAILURE;
		MAKE_FLM_ERROR_STRING( "Expected keys not generated.", 
			m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

Exit:

	if( RC_BAD( rc))
	{
		endTest("FAIL");
	}

	for( i = 0; i < sizeof(pszFirstNames)/sizeof(pszFirstNames[0]); i++)
	{
		f_free( &pszFirstPlusLastNames[ i]);
	}
	
	f_free( &pszFirstPlusLastNames);

	if( bTransStarted)
	{
		m_pDb->transCommit();
	}

	if( pDocument)
	{
		pDocument->Release();
	}

	if( pAttr)
	{
		pAttr->Release();
	}

	if( pNameNode)
	{
		pNameNode->Release();
	}

	if( pNextResult)
	{
		pNextResult->Release();
	}

	if( pPrevResult)
	{
		pPrevResult->Release();
	}

	if( pQuery)
	{
		pQuery->Release();
	}

	if( pKeyIter)
	{
		pKeyIter->Release();
	}

	shutdownTestState( DB_NAME_STR, bDibCreated);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE	IIndexTest2Impl::runSuite2( void)
{
	RCODE						rc = NE_XFLM_OK;
	FLMBOOL					bDibCreated = FALSE;
	FLMBOOL					bTransStarted = FALSE;
	IF_DataVector *		pSearchKey = NULL;
	FLMUINT					uiExtAttrId = 0;
	char						szTemp[30];
	FLMUINT					uiTemp;
	
	const char *			pszIndexDef = 
		"<prfx0:Index xmlns:prfx0=\"http://www.novell.com/XMLDatabase/Schema\" "
		"	prfx0:DictNumber=\"13\" "
		"	prfx0:CollectionNumber=\"65534\" "
		"	prfx0:name=\"track/ext index\" "
		"	prfx0:State=\"online\">"
		"	<prfx0:ElementComponent prfx0:name=\"track\" prfx0:KeyComponent=\"1\" prfx0:Required=\"1\">"
		"		<prfx0:AttributeComponent "
		"			prfx0:name=\"ext\" "
		"			prfx0:IndexOn=\"presence\" "
		"			prfx0:KeyComponent=\"2\"/>"
		"	</prfx0:ElementComponent>"
		"</prfx0:Index>";

	const char * 			pszDoc = 
		"<?xml version=\"1.0\"?> "
		"<disc>"
		"	<id>00097210</id>"
		"	<length>2420</length>"
		"	<title>Frank Sinatra / Blue skies</title>"
		"	<genre>cddb/jazz</genre>"
		"	<track index=\"1\" offset=\"150\">blue skies</track>"
		"	<track index=\"2\" offset=\"11693\" ext=\"Frank Sinatra\">night and day</track>"
		"</disc>";

	beginTest( 
		"Missing Presence Index Key Test",
		"",
		"No further details.",
		"");

	if( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to initialize test state.", m_szDetails, rc);
		goto Exit;
	}
	bDibCreated = TRUE;

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;

	if( RC_BAD( rc = m_pDb->createAttributeDef( 
		NULL, "ext", XFLM_TEXT_TYPE, &uiExtAttrId)))
	{
		MAKE_FLM_ERROR_STRING( "createAttributeDef failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = importBuffer( pszDoc, XFLM_DATA_COLLECTION)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = importBuffer( pszIndexDef, XFLM_DICT_COLLECTION)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_FLM_ERROR_STRING( "transCommit failed", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = FALSE;

	if( RC_BAD( rc = m_pDbSystem->createIFDataVector( &pSearchKey)))
	{
		MAKE_FLM_ERROR_STRING( "createIFDataVector failed", 
			m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->keyRetrieve( 13, NULL, XFLM_FIRST, pSearchKey)))
	{
		MAKE_FLM_ERROR_STRING( "keyRetrieve failed", 
			m_szDetails, rc);
		goto Exit;
	}

	uiTemp = sizeof(szTemp);
	if( RC_BAD( rc = pSearchKey->getUTF8( 0, (FLMBYTE *)szTemp, &uiTemp)))
	{
		MAKE_FLM_ERROR_STRING( "getNative failed", 
			m_szDetails, rc);
		goto Exit;
	}

	if( f_strcmp( szTemp, "blue skies") != 0)
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		MAKE_FLM_ERROR_STRING( "Invalid key component value", 
			m_szDetails, rc);
		goto Exit;
	}

	if( RC_OK( rc = pSearchKey->getUINT( 1, &uiTemp)))
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		MAKE_FLM_ERROR_STRING( "Invalid key component value", 
			m_szDetails, rc);
		goto Exit;
	}

	// first key - "Blue skies"/<nothing>

	if( RC_BAD( rc = m_pDb->keyRetrieve( 13, pSearchKey, XFLM_EXCL, pSearchKey)))
	{
		MAKE_FLM_ERROR_STRING( "keyRetrieve failed", 
			m_szDetails, rc);
		goto Exit;
	}

	// second key "night and day"/<ext's name id>

	uiTemp = sizeof(szTemp);

	if( RC_BAD( rc = pSearchKey->getUTF8( 0, (FLMBYTE *)szTemp, &uiTemp)))
	{
		MAKE_FLM_ERROR_STRING( "getUTF8 failed", 
			m_szDetails, rc);
		goto Exit;
	}

	if( f_strcmp( szTemp, "night and day") != 0)
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		MAKE_FLM_ERROR_STRING( "Invalid key component value", 
			m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pSearchKey->getUINT( 1, &uiTemp)))
	{
		MAKE_FLM_ERROR_STRING( "getUINT failed", 
			m_szDetails, rc);
		goto Exit;
	}

	if( uiTemp != uiExtAttrId)
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		MAKE_FLM_ERROR_STRING( "Invalid key component value", 
			m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

Exit:

	if( pSearchKey)
	{
		pSearchKey->Release();
	}

	if( RC_BAD( rc))
	{
		endTest("FAIL");
	}

	if( bTransStarted)
	{
		if( RC_OK(rc))
		{
			rc = m_pDb->transCommit();
		}
		else
		{
			m_pDb->transAbort();
		}
	}

	shutdownTestState( DB_NAME_STR, !bDibCreated);
	return( rc);
}
