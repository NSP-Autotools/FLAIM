//------------------------------------------------------------------------------
// Desc:	Indexing unit test 1
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
class IIndexTest1Impl : public TestBase
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

	if( (*ppTest = f_new IIndexTest1Impl) == NULL)
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
static RCODE addName(
	IF_Db *			pDb, 
	IF_DOMNode * 	pRoot,
	FLMUINT 			uiNameDictNum, 
	FLMBYTE *		pszName,
	FLMUINT64 *		pui64NameId)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pName = NULL;
	IF_DOMNode *	pDataNode = NULL;

	if( RC_BAD( rc = pRoot->createNode( pDb, ELEMENT_NODE, uiNameDictNum,
		XFLM_LAST_CHILD, &pName)))
	{
		goto Exit;
	}

	// Create data nodes of type TEXT to store the values. Should I 
	// have to do this manually?
	
	if ( RC_BAD( rc = pName->createNode( pDb, DATA_NODE, 0, XFLM_LAST_CHILD,
		&pDataNode)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pDataNode->setUTF8( pDb, pszName)))
	{
		goto Exit;
	}

	if( pui64NameId)
	{
		if( RC_BAD( rc = pName->getNodeId( pDb, pui64NameId)))
		{
			goto Exit;
		}
	}

Exit:

	if( pDataNode)
	{
		pDataNode->Release();
	}
	
	if( pName)
	{
		pName->Release();
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
const char * IIndexTest1Impl::getName( void)
{
	return( "Index Test 1");
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IIndexTest1Impl::execute( void)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DOMNode *		pDocument = NULL;
	IF_DOMNode *		pNode = NULL;
	IF_DOMNode *		pBeatlesNode = NULL;
	IF_DOMNode *		pAttr = NULL;
	IF_DOMNode *		pAttrToRemove = NULL;
	FLMBOOL				bTransStarted = FALSE;
	FLMBOOL				bDibCreated = FALSE;
	FLMUINT				uiBeatlesDef = 0;
	FLMUINT				uiFNDef = 0;
	FLMUINT				uiLNDef = 0;
	FLMUINT				uiLoop = 0;
	FLMUINT64			ui64JohnId = 0;
	FLMUINT64			ui64LennonId = 0;
	FLMUINT64			ui64NodeId;
	FLMBYTE				szBuf[ 128];
	KeyIterator *		pKeyIter = f_new KeyIterator;
	FlagSet				beatlesFNFlags;
	FlagSet				beatlesLNFlags;
	FlagSet				FNPlusLNFlags;
	FlagSet				contextFlags;
	FLMBYTE *			pucTemp = NULL;

	const char *		pszIndexDef1 = 
		"<xflaim:Index "
		"	xmlns:xflaim=\"http://www.novell.com/XMLDatabase/Schema\""
		"	xflaim:name=\"FirstName+LastName\" "
		"	xflaim:DictNumber=\"1\"> " // HARD-CODED Dict Num for easy retrieval
		"	<xflaim:ElementComponent "
		"		xflaim:name=\"FirstName\" "
		"		xflaim:KeyComponent=\"1\" "
		"		xflaim:IndexOn=\"value\"/> "
		"	<xflaim:ElementComponent "
		"		xflaim:name=\"LastName\" "
		"		xflaim:IndexOn=\"value\" "
		"		xflaim:KeyComponent=\"2\"/> "
		"</xflaim:Index> ";

	const char *		pszIndexDef2 = 
		"<xflaim:Index "
		"	xmlns:xflaim=\"http://www.novell.com/XMLDatabase/Schema\""
		"	xflaim:name=\"X+A+B\" "
		"	xflaim:DictNumber=\"77\"> " // HARD-CODED Dict Num for easy retrieval
		"	<xflaim:ElementComponent "
		"		xflaim:name=\"X\" "
		"		xflaim:KeyComponent=\"1\" "
		"		xflaim:IndexOn=\"value\"/> "
		"	<xflaim:ElementComponent "
		"		xflaim:name=\"Y\"> "
		"			<xflaim:ElementComponent "
		"				xflaim:name=\"A\"	"
		"				xflaim:IndexOn=\"value\" "
		"				xflaim:KeyComponent=\"2\"/>"
		"			<xflaim:ElementComponent "
		"				xflaim:name=\"B\"	"
		"				xflaim:IndexOn=\"value\" "
		"				xflaim:KeyComponent=\"3\"/>"
		"	</xflaim:ElementComponent> "
		"</xflaim:Index> ";

	const char *	pszRec =
		"<DOC> "
			"<X>0</X>"
			"<X>1</X>"
			"<Y>"
			"	<A>2</A>"
			"	<B>3</B>"
			"</Y>"
			"<Y>"
			"	<A>4</A>"
			"	<B>5</B>"
			"</Y>"
		"</DOC>";

	const char *	ppszLegalKeys[] = {"023", "123", "045", "145"};

	const char *	pszFirstNames1[] = {"John", "Paul", "Ringo", "George"};
	
	const char *	pszLastNames1[] = {"Lennon", "McCartney", "Starr", "Harrison"};

	beatlesFNFlags.init( (FLMBYTE **)pszFirstNames1, 
		sizeof( pszFirstNames1)/sizeof( pszFirstNames1[0]));
	beatlesLNFlags.init( (FLMBYTE **)pszLastNames1, 
		sizeof( pszLastNames1)/sizeof( pszLastNames1[0]));

	// We will be using this to test compound indexes (FirstName + LastName)

	FNPlusLNFlags = beatlesFNFlags.crossProduct( beatlesLNFlags);

	beginTest( 		
		"Initialize FLAIM Test",
		"Perform initializations",
		"(1)Get DbSystem class factory (2)call init() (3)create XFLAIM db",
		"No additional info.");

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

	endTest("PASS");

	beginTest( 
		"Index Definition Import Test",
		"Import the index definition",
		"Self-Explanatory",
		"");

	if( RC_BAD( rc = m_pDb->createElementDef(
		NULL, "FirstName", XFLM_TEXT_TYPE, &uiFNDef)))
	{
		MAKE_FLM_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->createElementDef(
		NULL, "LastName", XFLM_TEXT_TYPE, &uiLNDef)))
	{
		MAKE_FLM_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->createElementDef(
		NULL, "Beatles", XFLM_NODATA_TYPE, &uiBeatlesDef)))
	{
		MAKE_FLM_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = importBuffer( pszIndexDef1, XFLM_DICT_COLLECTION)))
	{
		 goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transCommit()))
	{
		goto Exit;
	}
	bTransStarted = FALSE;

	endTest("PASS");

	beginTest( 	
		"Index Key Test",
		"Create Data Documents and verify proper index keys are being generated.",
		"Self-Explanatory",
		"");

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;

	if( RC_BAD( rc = m_pDb->createDocument( 
		XFLM_DATA_COLLECTION, 
		&pDocument)))
	{
		MAKE_FLM_ERROR_STRING( "createDocument failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pDocument->createNode( m_pDb, ELEMENT_NODE, uiBeatlesDef,
		XFLM_FIRST_CHILD, &pBeatlesNode)))
	{
		MAKE_FLM_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	// Add some first names

	for( uiLoop = 0; 
		uiLoop < sizeof( pszFirstNames1)/sizeof( pszFirstNames1[0]); uiLoop++)
	{
		if( RC_BAD( rc = addName( m_pDb, pBeatlesNode, uiFNDef, 
			(FLMBYTE *)pszFirstNames1[ uiLoop],
			( f_strcmp( pszFirstNames1[ uiLoop], "John")  == 0) 
				? &ui64JohnId 
				: NULL)))
		{
			MAKE_FLM_ERROR_STRING( "failed to add name.", m_szDetails, rc);
			goto Exit;
		}
	}

	if( RC_BAD( rc = m_pDb->documentDone( pDocument)))
	{
		MAKE_FLM_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}

	// Check the keys that were created. They should be of the form
	// FirstName + NULL since the LastName piece is optional and
	// there are no LastNames in any documents.

	//	Get the first key

	if( RC_BAD( rc = pKeyIter->init( 1, m_pDbSystem, m_pDb)))
	{
		goto Exit;
	}
	
	while( RC_OK( rc = pKeyIter->next()))
	{
		if( RC_BAD( rc = pKeyIter->getCurrentKeyVal( 
			0, szBuf, sizeof(szBuf), NULL)))
		{
			goto Exit;
		}

		if( !beatlesFNFlags.setElemFlag( szBuf))
		{
			// FALSE indicates that the element was not found in the set

			rc = NE_XFLM_FAILURE;
			MAKE_FLM_ERROR_STRING( "Unexpected index key found.", m_szDetails, rc);
			goto Exit;
		}

		// Second piece should be empty

		if( (rc = pKeyIter->getCurrentKeyVal( 
			1, szBuf, sizeof(szBuf), NULL, &ui64NodeId)) != NE_XFLM_NOT_FOUND)
		{
			if( RC_OK( rc) && ui64NodeId)
			{
				MAKE_FLM_ERROR_STRING( "Unexpected rc from getNative.", m_szDetails, rc);
				rc = NE_XFLM_FAILURE;
			}
			
			goto Exit;
		}
	}

	if( rc != NE_XFLM_EOF_HIT)
	{
		MAKE_FLM_ERROR_STRING( "Unexpected rc from keyRetrieve.", m_szDetails, rc);
		if( RC_OK( rc))
		{
			rc = NE_XFLM_FAILURE;
		}
		goto Exit;
	}

	if( !beatlesFNFlags.allElemFlagsSet())
	{
		rc = NE_XFLM_FAILURE;
		MAKE_FLM_ERROR_STRING( "Wrong number of index keys.", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

	beginTest( 
		"Optional/Required Component Test",

		"Modify the \"Required\" attribute of the index definition "
		"and verify that the index keys are being deleted/added correctly",
	
		"1) Make the LastName portion of the compound index required. "
		"Verify that all the keys of the FirstName+LastName index disappear "
		"since there are not currently any LastNames in the database. 2) Make the "
		"LastName optional again and verify all the index keys return. Also verify "
		"that there are not any keys in the index that shouldn't be there.",
		"");

	// Make the LastName portion of the key required

	if( RC_BAD( rc = m_pDb->getDictionaryDef( ELM_INDEX_TAG, 1, &pNode)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pNode->getFirstChild( m_pDb, &pNode)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pNode->getNextSibling( m_pDb, &pNode)))
	{
		goto Exit;
	}

	if ( RC_BAD( rc = pNode->createAttribute( m_pDb, ATTR_REQUIRED_TAG, &pAttr)))
	{
		MAKE_FLM_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	flmAssert( !pAttrToRemove);
	pAttrToRemove = pAttr;
	pAttr->AddRef();

	// NOTE - "yes" "on" and "1" will also work

	if( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"enable")))
	{
		MAKE_FLM_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pNode->getDocumentId( m_pDb, &ui64NodeId)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->documentDone( XFLM_DICT_COLLECTION, ui64NodeId)))
	{
		MAKE_FLM_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}

	// There should now be no keys in the index

	pKeyIter->reset();
	rc = pKeyIter->next();
	if( rc != NE_XFLM_EOF_HIT)
	{
		MAKE_FLM_ERROR_STRING( "Index should be empty.", m_szDetails, rc);
		rc = NE_XFLM_FAILURE;
		goto Exit;
	}

	// Make the LastName portion optional again

	if( RC_BAD( rc = pAttrToRemove->getDocumentId( m_pDb, &ui64NodeId)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pAttrToRemove->deleteNode( m_pDb)))
	{
		MAKE_FLM_ERROR_STRING( "deleteNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->documentDone( XFLM_DICT_COLLECTION, ui64NodeId)))
	{
		MAKE_FLM_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}

	// Ensure keys have been regenerated

	beatlesFNFlags.clearFlags();
	pKeyIter->reset();
	while( RC_OK( rc = pKeyIter->next()))
	{
		if ( RC_BAD( rc = pKeyIter->getCurrentKeyVal( 
			0, szBuf, sizeof(szBuf), NULL)))
		{
			goto Exit;
		}

		if( !beatlesFNFlags.setElemFlag( szBuf))
		{
			MAKE_FLM_ERROR_STRING( "Unexpected key in index.", m_szDetails, rc);
			rc = NE_XFLM_FAILURE;
			goto Exit;
		}

		// Second piece should be empty

		if( (rc = pKeyIter->getCurrentKeyVal( 
			1, szBuf, sizeof(szBuf), NULL))  != NE_XFLM_NOT_FOUND)
		{
			MAKE_FLM_ERROR_STRING( "Unexpected rc from getNative.", m_szDetails, rc);
			if( RC_OK( rc))
			{
				rc = NE_XFLM_FAILURE;
			}
			goto Exit;
		}
	}

	if( rc != NE_XFLM_EOF_HIT)
	{
		MAKE_FLM_ERROR_STRING( "Unexpected rc from keyRetrieve.", m_szDetails, rc);
		if( RC_OK( rc))
		{
			rc = NE_XFLM_FAILURE;
		}
		goto Exit;
	}

	// We had better have found all of the first names in the index

	if( !beatlesFNFlags.allElemFlagsSet())
	{
		rc = NE_XFLM_FAILURE;
		MAKE_FLM_ERROR_STRING( "Wrong number of index keys.", m_szDetails, rc);
		goto Exit;
	}

	endTest( "PASS");

	beginTest( 	
		"Second Component Test",
		"Add the some LastName elements to the database and make sure "
		"all the index keys for the FirstName+LastName index are correct. ",
		"1) Add LastName elements to the database. 2) Iterate through "
		"all of the keys of the FirstName+LastName index and verify that they "
		"consist solely of the cross-product of the all the FirstNames and "
		"LastNames.",
		"");

	// Add some last names

	for( uiLoop = 0;
		uiLoop < sizeof( pszLastNames1)/sizeof( pszLastNames1[0]); uiLoop++)
	{
		if( RC_BAD( rc = addName( m_pDb, pBeatlesNode, uiLNDef, 
			(FLMBYTE *)pszLastNames1[ uiLoop],  
			(f_strcmp( pszLastNames1[ uiLoop], "Lennon") == 0) 
			? &ui64LennonId
			: NULL)))
		{
			MAKE_FLM_ERROR_STRING( "failed to add name to document.", m_szDetails, rc);
			goto Exit;
		}
	}

	if( RC_BAD( rc = m_pDb->documentDone( pDocument)))
	{
		MAKE_FLM_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}

	// Check the keys that were generated. They should consist solely of the 
	// cross product of pszFirstNames1 and pszLastNames1. 

	pKeyIter->reset();
	f_alloc( sizeof( szBuf) * 2 + 1, &pucTemp);
	
	while( RC_OK( rc = pKeyIter->next()))
	{
		if ( RC_BAD( rc = pKeyIter->getCurrentKeyVal( 0, szBuf, 
			sizeof( szBuf), NULL)))
		{
			MAKE_FLM_ERROR_STRING( "Unable to get key value.", m_szDetails, rc);
			goto Exit;
		}

		f_strcpy( (char *)pucTemp, (char *)szBuf);

		if( RC_BAD( rc = pKeyIter->getCurrentKeyVal(
			1, szBuf, sizeof( szBuf), NULL)))
		{
			MAKE_FLM_ERROR_STRING( "Unable to get key value.", m_szDetails, rc);
			goto Exit;
		}

		f_strcat( (char *)pucTemp, (char *)szBuf);

		if( !FNPlusLNFlags.setElemFlag( pucTemp))
		{
			rc = NE_XFLM_FAILURE;
			MAKE_FLM_ERROR_STRING( "Illegal key found.", m_szDetails, rc);
			goto Exit;
		}
	}
	
	f_free( &pucTemp);

	if( rc != NE_XFLM_EOF_HIT)
	{
		MAKE_FLM_ERROR_STRING( "Unexpected rc from keyRetrieve.", m_szDetails, rc);
		goto Exit;
	}
	
	if( !FNPlusLNFlags.allElemFlagsSet())
	{
		rc = NE_XFLM_FAILURE;
		MAKE_FLM_ERROR_STRING( "Wrong number of index keys.", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

	beginTest( 
		"Indexed Element Removal Test",
		"Remove some elements from the database and ensure that they "
		"are no longer referenced in the Index.",
		"1) Remove the FirstName \"John\". 2)Verify there are not "
		"any index keys containing \"John\". 3) Remove the LastName \"Lennon\" "
		"4) Verify there are no index keys containing \"Lennon\".",
		"");

	// Delete John. 

	if( RC_BAD( rc = m_pDb->getNode( XFLM_DATA_COLLECTION, ui64JohnId, &pNode)))
	{
		MAKE_FLM_ERROR_STRING( "getNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pNode->deleteNode( m_pDb)))
	{
		MAKE_FLM_ERROR_STRING( "deleteNode failed.", m_szDetails, rc);
		goto Exit;
	}

	// Make sure none of the index keys refer to John now

	// Remove "John" from the FirstName Flag set
	
	beatlesFNFlags.removeElem( (FLMBYTE *)"John");

	// Generate a new "John"-less cross product
	
	FNPlusLNFlags = beatlesFNFlags.crossProduct( beatlesLNFlags);
	
	// Now, if we run across a "John" key in the index and try to set its flag
	// in the flag set, we'll get an error.

	pKeyIter->reset();
	f_alloc( sizeof( szBuf) * 2 + 1, &pucTemp);
	
	while( RC_OK( rc = pKeyIter->next()))
	{
		if ( RC_BAD( rc = pKeyIter->getCurrentKeyVal( 
			0, 
			szBuf, 
			sizeof( szBuf),
			NULL)))
		{
			MAKE_FLM_ERROR_STRING( "Unable to get key value.", m_szDetails, rc);
			goto Exit;
		}

		f_strcpy( (char *)pucTemp, (char *)szBuf);

		if( RC_BAD( rc = pKeyIter->getCurrentKeyVal(
			1, szBuf, sizeof( szBuf), NULL)))
		{
			MAKE_FLM_ERROR_STRING( "Unable to get key value.", m_szDetails, rc);
			goto Exit;
		}

		f_strcat( (char *)pucTemp, (char *)szBuf);

		if( !FNPlusLNFlags.setElemFlag( pucTemp))
		{
			MAKE_FLM_ERROR_STRING( "Unexpected key found.", m_szDetails, rc);
			goto Exit;
		}
	}
	
	f_free( &pucTemp);

	if( rc != NE_XFLM_EOF_HIT)
	{
		MAKE_FLM_ERROR_STRING( "Unexpected rc from keyRetrieve.", m_szDetails, rc);
		goto Exit;
	}

	rc = NE_XFLM_OK;

	if( !FNPlusLNFlags.allElemFlagsSet())
	{
		rc = NE_XFLM_FAILURE;
		MAKE_FLM_ERROR_STRING( "Missing index index keys.", m_szDetails, rc);
		goto Exit;
	}

	// Delete Lennon. 

	if( RC_BAD( rc = m_pDb->getNode( XFLM_DATA_COLLECTION,
		ui64LennonId, &pNode)))
	{
		MAKE_FLM_ERROR_STRING( "getNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pNode->deleteNode( m_pDb)))
	{
		MAKE_FLM_ERROR_STRING( "deleteNode failed.", m_szDetails, rc);
		goto Exit;
	}

	// Make sure none of the index keys contain Lennon now

	// Remove "Lennon" from the LastNameName Flag set
	
	beatlesLNFlags.removeElem( (FLMBYTE *)"Lennon");

	// Generate a new "Lennon"-less cross product
	
	FNPlusLNFlags = beatlesFNFlags.crossProduct( beatlesLNFlags);
	
	// Now, if we run across a "Lennon" key in the index and try to set its flag
	// in the flag set, we'll get an error.

	pKeyIter->reset();
	f_alloc( sizeof( szBuf) * 2 + 1, &pucTemp);
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

		if( !FNPlusLNFlags.setElemFlag( pucTemp))
		{
			MAKE_FLM_ERROR_STRING( "Unexpected key found.", m_szDetails, rc);
			goto Exit;
		}
	}
	
	f_free( &pucTemp);

	if( rc != NE_XFLM_EOF_HIT)
	{
		MAKE_FLM_ERROR_STRING( "Unexpected rc from keyRetrieve.", m_szDetails, rc);
		goto Exit;
	}

	rc = NE_XFLM_OK;

	if( !FNPlusLNFlags.allElemFlagsSet())
	{
		rc = NE_XFLM_FAILURE;
		MAKE_FLM_ERROR_STRING( "Missing index index keys.", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

	beginTest( 
		"Key Context Test",
		"Import a record that has indexed elements at varying "
		"contexts and verify that only valid keys are generated",
		"1) Import/create necessary dictionary definitions "
		"2) cycle through all keys in the index and verify that we only "
		"get back the keys we are expecting.",
		"");

	// Do the context test

	if( RC_BAD( rc = m_pDb->createElementDef(
		NULL, "X", XFLM_TEXT_TYPE, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->createElementDef(
		NULL, "Y", XFLM_NODATA_TYPE, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->createElementDef(
		NULL, "A", XFLM_TEXT_TYPE, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->createElementDef(
		NULL, "B", XFLM_TEXT_TYPE, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = importBuffer( pszIndexDef2, XFLM_DICT_COLLECTION)))
	{
		 goto Exit;
	}

	if( RC_BAD( rc = importBuffer( pszRec, XFLM_DATA_COLLECTION)))
	{
		 goto Exit;
	}

	// Iterate over all the keys in the X+A+B index and verify they are all legal

	pKeyIter->setIndexNum( 77);
	contextFlags.init( (FLMBYTE **)ppszLegalKeys, 
		sizeof( ppszLegalKeys)/sizeof( ppszLegalKeys[0]));
	f_alloc( sizeof( szBuf) * 3 + 1, &pucTemp);
	
	while( RC_OK( rc = pKeyIter->next()))
	{
		if( RC_BAD( rc = pKeyIter->getCurrentKeyVal( 0, szBuf, sizeof( szBuf),
			NULL)))
		{
			MAKE_FLM_ERROR_STRING( "Unable to get key value.", m_szDetails, rc);
			goto Exit;
		}

		f_strcpy( (char *)pucTemp, (char *)szBuf);

		if( RC_BAD( rc = pKeyIter->getCurrentKeyVal(
			1, szBuf, sizeof( szBuf), NULL)))
		{
			MAKE_FLM_ERROR_STRING( "Unable to get key value.", m_szDetails, rc);
			goto Exit;
		}

		f_strcat( (char *)pucTemp, (char *)szBuf);

		if( RC_BAD( rc = pKeyIter->getCurrentKeyVal(
			2, szBuf, sizeof( szBuf), NULL)))
		{
			MAKE_FLM_ERROR_STRING( "Unable to get key value.", m_szDetails, rc);
			goto Exit;
		}

		f_strcat( (char *)pucTemp, (char *)szBuf);

		if( !contextFlags.setElemFlag( pucTemp))
		{
			MAKE_FLM_ERROR_STRING( "Unexpected key found.", m_szDetails, rc);
			goto Exit;
		}
	}
	
	f_free( &pucTemp);

	if( rc != NE_XFLM_EOF_HIT)
	{
		MAKE_FLM_ERROR_STRING( "Unexpected rc from keyRetrieve.", m_szDetails, rc);
		goto Exit;
	}

	rc = NE_XFLM_OK;

	if( !contextFlags.allElemFlagsSet())
	{
		rc = NE_XFLM_FAILURE;
		MAKE_FLM_ERROR_STRING( "Missing index index keys.", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

Exit:

	if( bTransStarted)
	{
		m_pDb->transCommit();
	}

	if( RC_BAD( rc))
	{
		endTest("FAIL");
	}

	if( pucTemp)
	{
		f_free( &pucTemp);
	}

	if( pNode)
	{
		pNode->Release();
	}

	if( pBeatlesNode)
	{
		pBeatlesNode->Release();
	}

	if( pAttr)
	{
		pAttr->Release();
	}

	if( pAttrToRemove)
	{
		pAttrToRemove->Release();
	}

	if( pDocument)
	{
		pDocument->Release();
	}

	// Need to force the destruction of the iterator before the pDb and DbSystem
	// are released

	if( pKeyIter)
	{
		pKeyIter->Release();
	}

	shutdownTestState( DB_NAME_STR, !bDibCreated);
	return( rc);
}
