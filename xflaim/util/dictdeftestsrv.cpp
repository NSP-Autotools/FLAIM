//------------------------------------------------------------------------------
// Desc:	Dictionary definition tests
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
class IDictDefTestImpl : public TestBase
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

	if( (*ppTest = f_new IDictDefTestImpl) == NULL)
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
const char * IDictDefTestImpl::getName( void)
{
	return( "Dictionary Definition Test");
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IDictDefTestImpl::execute( void)
{
	RCODE				rc = NE_XFLM_OK;
	int				ps = 0;
	FLMUINT			uiDefNum = 0;
	IF_DOMNode *	pDocument = NULL;
	IF_DOMNode *	pNode = NULL;
	IF_DOMNode *	pAttr = NULL;
	IF_DOMNode *	pAttrDef = NULL;
	IF_DOMNode *	pElemDef = NULL;
	IF_DOMNode *	pIndex = NULL;
	IF_DOMNode *	pCollection = NULL;
	FLMUINT			uiAttrDictNum = 0;
	FLMUINT			uiElemDictNum = 0;
	FLMUINT			uiIndexDictNum = 0;
	FLMBYTE			szBuf[ 128];
	FLMBOOL			bTransStarted = FALSE;
	FLMBOOL			bDibCreated = FALSE;
	FLMUINT64		ui64Tmp;

	beginTest( 	
		"Initialize FLAIM Test",
		"Perform initializations so we can test the dictionary definition",
		"(1)Get DbSystem class factory (2)call init() (3)create XFLAIM db",
		"No additional info.");

	if( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		MAKE_ERROR_STRING( "Failed to initialize test state.", m_szDetails, ps);
		goto Exit;
	}
	bDibCreated = TRUE;

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;

	endTest("PASS");

	beginTest( 	
		"Type Attribute Test",
		"Verify that deleting or modifying of the \"type\""
		"attribute (ATTR_TYPE_TAG) of an element or attribute definition is "
		"denied with a proper error code",
		"(1) Create an element definition (2)Verify the node has a "
		"\"type\" attribute (3)Try to set it to an arbitrary value (4)Try "
		"to delete it.",
		"");

	if( RC_BAD( rc = m_pDb->createElementDef(
		NULL,
		"element1",
		XFLM_TEXT_TYPE,
		&uiDefNum)))
	{
		MAKE_ERROR_STRING( "createElemDef failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->getDictionaryDef(
		ELM_ELEMENT_TAG,
		uiDefNum,
		&pNode)))
	{
		MAKE_ERROR_STRING( "getDictionaryDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pNode->hasAttribute(
		m_pDb,
		ATTR_TYPE_TAG,
		&pAttr)))
	{
		if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			MAKE_ERROR_STRING( "\"Type\" attribute not automatically created.", 
				m_szDetails, rc);
		}
		else
		{
			MAKE_ERROR_STRING( "hasAttribute failed.", m_szDetails, rc);
		}
		goto Exit;
	}

	rc = pAttr->setUINT(
		m_pDb,
		8);
	if ( rc != NE_XFLM_READ_ONLY)
	{
		MAKE_ERROR_STRING( "Succeeded in modifying \"type\" attribute.", 
			m_szDetails, rc);
			
		if ( RC_OK( rc))
		{
			rc = NE_XFLM_FAILURE;
		}
		
		goto Exit;
	}

	// need to begin a new transaction since attempting to delete
	// the Attribute node will probably require the transaction
	// to be aborted and we don't want to lose our modifications
	
	if( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = FALSE;

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = FALSE;

	rc = pAttr->deleteNode(m_pDb);
	if( rc != NE_XFLM_DELETE_NOT_ALLOWED)
	{
		MAKE_ERROR_STRING( "Incorrect rc when trying to delete \"type\" attribute.",
			m_szDetails, rc);
			
		if ( RC_OK( rc))
		{
			rc = NE_XFLM_FAILURE;
		}
		
		goto Exit;
	}

	// The failed delete requires us to abort the trans
	// NOTE: This may no longer be necessary in the future

	if ( RC_BAD( rc = m_pDb->transAbort()))
	{
		MAKE_ERROR_STRING( "transAbort failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = FALSE;

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;

	endTest("PASS");

	beginTest( 	
		"Name Attribute Test",
		"Verify that deleting of the \"name\" "
		"attribute (ATTR_NAME_TAG) of any definition is denied with a proper "
		"error code. Verify that modifying the name is allowed but deleting "
		"it should be denied.",
		"(1) Verify the node has a \"name\" attribute"
		"(2) Set it to an arbitrary value (3)Try to delete it.",
		"");

	if( RC_BAD( rc = pNode->hasAttribute( m_pDb, ATTR_NAME_TAG)))
	{
		MAKE_ERROR_STRING( "hasAttribute failed. \"name\" attribute not created"
			 " automatically.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pNode->getAttribute( m_pDb, ATTR_NAME_TAG, &pAttr)))
	{
		MAKE_ERROR_STRING( "getAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	// This attribute can be modified...
	
	if( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"This should be okay")))
	{
		MAKE_ERROR_STRING( "failed to modify \"name\" attribute.", m_szDetails, rc);
		goto Exit;
	}

	// Need to begin a new transaction since attempting to delete
	// the Attribute node will probably require the transaction
	// to be aborted and we don't want to lose our modifications
	
	if( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = FALSE;

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = FALSE;

	rc = pAttr->deleteNode(m_pDb);
	if( rc != NE_XFLM_DELETE_NOT_ALLOWED)
	{
		MAKE_ERROR_STRING( "Succeeded in deleting \"name\" attribute.", 
			m_szDetails, rc);
			
		if ( RC_OK( rc))
		{
			rc = NE_XFLM_FAILURE;
		}
		
		goto Exit;
	}

	// The failed delete requires us to abort the trans
	// NOTE: This may no longer be necessary in the future

	if( RC_BAD( rc = m_pDb->transAbort()))
	{
		MAKE_ERROR_STRING( "transAbort failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = FALSE;

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;

	endTest("PASS");

	beginTest(
		"State Attribute Test 1",
		" Verify that deleting or modifying the \"state\" attribute "
		"(ATTR_STATE_TAG) on an element attribute or index definition is denied "
		"if it is attemped using the \"regular\" DOM methods for updating nodes. ",	
		"(1) Verify the node has a \"state\" attribute"
		"(2) Try to set it to an arbitrary value (3) Try to delete it.",
		"");

	if( RC_BAD( rc = pNode->hasAttribute( m_pDb, ATTR_STATE_TAG, &pAttr)))
	{
		if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			MAKE_ERROR_STRING( "\"state\" attribute not automatically created.",
				m_szDetails, rc);
			rc = NE_XFLM_FAILURE;
		}
		else
		{
			MAKE_ERROR_STRING( "hasAttribute failed.", m_szDetails, rc);
		}
		
		goto Exit;
	}

	rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"This should not work");
	if( rc != NE_XFLM_READ_ONLY)
	{
		MAKE_ERROR_STRING( "attempt to modify \"state\" attribute returned "
			"incorrect rc.", m_szDetails, rc);
			
		if( RC_OK( rc ))
		{
			rc = NE_XFLM_FAILURE;
		}
		
		goto Exit;
	}

	// Commit the trans here since we may have to abort the trans
	// in which we attempt to delete the pAttr node
	
	if( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = FALSE;

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;

	rc = pAttr->deleteNode(m_pDb);
	if( rc != NE_XFLM_DELETE_NOT_ALLOWED)
	{
		MAKE_ERROR_STRING( "Attempt to delete \"state\" attribute "
			"returned incorrect rc.", m_szDetails, rc);
		goto Exit;
	}

	// The failed delete requires us to abort the trans
	// NOTE: This may no longer be necessary in the future

	if( RC_BAD( rc = m_pDb->transAbort()))
	{
		MAKE_ERROR_STRING( "transAbort failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = FALSE;

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;

	endTest("PASS");

	beginTest(
		"State Attribute Test 2 / Dict Num Test",
	
		"Test creating an element or attribute definition without "
		"a \"state\" attribute (ATTR_STATE_TAG). In this case FLAIM should "
		"automatically create the \"state\" attribute and set its value to "
		"\"active\". Same should happen for indexes if the user did not create "
		"a state attribute for the index definition and value should be set "
		"to \"online\". Make sure that the \"DictNumber\" attribute "
		"(ATTR_DICT_NUMBER) is automatically created and assigned if the user "
		"omits it or sets it to zero. This applies to all types of definitions "
		"(elements attributes indexes collections and prefixes). Once set "
		"to a non-zero value make sure that the user cannot ever modify or "
		"delete the attribute.",

		"(1) Create an attribute definition (2) create an element "
		"definition (3) create an index definition (3) make sure attribute",
		"");


	//create an attribute definition
	/*
	<name="foo" type="string" state="active"/>
	*/
	if ( RC_BAD( rc = m_pDb->createRootElement(
		XFLM_DICT_COLLECTION,
		ELM_ATTRIBUTE_TAG,
		&pAttrDef)))
	{
		MAKE_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttrDef->createAttribute(
		m_pDb,
		ATTR_NAME_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"foo")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->documentDone( pAttrDef)))
	{
		MAKE_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttrDef->getAttribute(
		m_pDb,
		ATTR_STATE_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "getAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->getUTF8( m_pDb, szBuf, sizeof(szBuf), 0,
		sizeof(szBuf) - 1)))
	{
		MAKE_ERROR_STRING( "getUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( f_strcmp( (char *)szBuf, "active"))
	{
		MAKE_ERROR_STRING( "unexpected \"state\" value. Expected: \"active\"",
			m_szDetails, rc);
		goto Exit;
	}

	//create an element definition
	/*
	<name="bar" type="string" state="active" foo />
	*/

	if ( RC_BAD( rc = m_pDb->createRootElement(
		XFLM_DICT_COLLECTION,
		ELM_ELEMENT_TAG,
		&pElemDef)))
	{
		MAKE_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pElemDef->createAttribute(
		m_pDb,
		ATTR_NAME_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"bar")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD ( rc = pAttrDef->getAttribute(
		m_pDb, ATTR_DICT_NUMBER_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "getAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->getUINT( m_pDb, &uiAttrDictNum)))
	{
		MAKE_ERROR_STRING( "getUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	rc = pAttr->setUINT( m_pDb, 123);
	if ( rc != NE_XFLM_READ_ONLY)
	{
		MAKE_ERROR_STRING( "Incorrect error code returned while trying to "
			"delete dict number attribute.", m_szDetails, rc);
		if ( RC_OK( rc))
		{
			rc = NE_XFLM_FAILURE;
		}
		goto Exit;
	}

	//should have been set up automatically

	if ( uiAttrDictNum == 0)
	{
		MAKE_ERROR_STRING( "Invalid dictionary number.", m_szDetails, rc);
		rc = NE_XFLM_FAILURE;
		goto Exit;
	}

	if ( RC_BAD( rc = pElemDef->createAttribute(
		m_pDb,
		uiAttrDictNum,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->documentDone( pElemDef)))
	{
		MAKE_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}

	//make sure state automatically set to "active"

	if ( RC_BAD( rc = pElemDef->getAttribute(
		m_pDb,
		ATTR_STATE_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "getAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->getUTF8( m_pDb, szBuf, sizeof(szBuf), 0,
		sizeof(szBuf) - 1)))
	{
		MAKE_ERROR_STRING( "getUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( f_strcmp( (char *)szBuf, "active"))
	{
		MAKE_ERROR_STRING( "Invalid state value. Should be \"active\".",
			m_szDetails, rc);
		rc = NE_XFLM_FAILURE;
		goto Exit;
	}

	//make sure the dict num was set up automatically

	if ( RC_BAD ( rc = pElemDef->getAttribute(
		m_pDb, ATTR_DICT_NUMBER_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "getAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->getUINT( m_pDb, &uiElemDictNum)))
	{
		MAKE_ERROR_STRING( "getUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( uiElemDictNum == 0)
	{
		MAKE_ERROR_STRING( "Invalid dictionary number.", m_szDetails, rc);
		rc = NE_XFLM_FAILURE;
		goto Exit;
	}

	rc = pAttr->setUINT( m_pDb, 123);
	if ( rc != NE_XFLM_READ_ONLY)
	{
		MAKE_ERROR_STRING( "Incorrect rc when trying to modify dict num.",
			m_szDetails, rc);
		if ( RC_OK( rc))
		{
			rc = NE_XFLM_FAILURE;
		}
		goto Exit;
	}

	//create an index definition that references the elem and attr defs we made
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

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"bar+foo")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	//xflaim:ElementPath must have one or more xflaim:ElementComponent
	//or one or more xflaim:AttributeComponent sub-elements

	if ( RC_BAD( rc = pIndex->createNode(
		m_pDb,
		ELEMENT_NODE,
		ELM_ELEMENT_COMPONENT_TAG,
		XFLM_FIRST_CHILD,
		&pNode)))
	{
		MAKE_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pNode->createAttribute(
		m_pDb,
		ATTR_NAME_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"bar")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pNode->createAttribute(
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

	if ( RC_BAD ( rc = pNode->createAttribute(
		m_pDb,
		ATTR_INDEX_ON_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"value")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pNode->createNode(
		m_pDb,
		ELEMENT_NODE,
		ELM_ATTRIBUTE_COMPONENT_TAG,
		XFLM_FIRST_CHILD,
		&pNode)))
	{
		MAKE_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pNode->createAttribute(
		m_pDb,
		ATTR_NAME_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUTF8(m_pDb, (FLMBYTE *)"foo")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pNode->createAttribute(
		m_pDb,
		ATTR_INDEX_ON_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"value")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pNode->createAttribute(
		m_pDb,
		ATTR_KEY_COMPONENT_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUINT( m_pDb, 2)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	if (RC_BAD( rc = m_pDb->documentDone( pIndex)))
	{
		MAKE_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD ( rc = pIndex->getAttribute(
		m_pDb, ATTR_DICT_NUMBER_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "getAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->getUINT( m_pDb, &uiIndexDictNum)))
	{
		MAKE_ERROR_STRING( "getUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( uiIndexDictNum == 0)
	{
		MAKE_GENERIC_ERROR_STRING( "Invalid dict num", m_szDetails, uiIndexDictNum);
		rc = NE_XFLM_FAILURE;
		goto Exit;
	}

	rc = pAttr->setUINT( m_pDb, 123);
	if ( rc != NE_XFLM_READ_ONLY)
	{
		MAKE_ERROR_STRING( "Invalid rc when trying to change dict num.",
			m_szDetails, rc);
		if ( RC_OK( rc ))
		{
			rc = NE_XFLM_FAILURE;
		}
		goto Exit;
	}

	endTest("PASS");

	beginTest(
		"Referenced Definitions Test",
		"Test to make sure that element and attribute definitions "
		"cannot be deleted if they are referenced from an index definition.",
		"Try to delete the Element and Attribute Definitions we created.",
		"");

	//need to begin a new transaction since attempting to delete
	//the Element Definition node will probably require the transaction
	//to be aborted and we don't want to lose our modifications
	if ( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = FALSE;

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = FALSE;

	rc = pElemDef->deleteNode( m_pDb);
	if ( rc != NE_XFLM_CANNOT_DEL_ELEMENT)
	{
		MAKE_ERROR_STRING( "Expected rc == NE_XFLM_CANNOT_DEL_ELEMENT",
			m_szDetails, rc);
		goto Exit;
	}

	//The failed delete requires us to abort the trans
	//NOTE: This may no longer be necessary in the future

	if ( RC_BAD( rc = m_pDb->transAbort()))
	{
		MAKE_ERROR_STRING( "transAbort failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = FALSE;

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;

	//try to delete attribute
	rc = pAttrDef->deleteNode( m_pDb);
	if ( rc != NE_XFLM_CANNOT_DEL_ATTRIBUTE)
	{
		MAKE_ERROR_STRING( "Exected rc == NE_XFLM_CANNOT_DEL_ATTRIBUTE.",
			m_szDetails, rc);
		if ( RC_OK( rc ))
		{
			rc = NE_XFLM_FAILURE;
		}
		goto Exit;
	}

	//The failed delete requires us to abort the trans
	//NOTE: This may no longer be necessary in the future
	if ( RC_BAD( rc = m_pDb->transAbort()))
	{
		MAKE_ERROR_STRING( "transAbort failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = FALSE;

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;

	endTest("PASS");

	beginTest(
		"State Attribute Test 3",
	
		"Verify that the state attribute can be modified using the "
		"special methods that are made for that purpose. For index definitions "
		"the indexSuspend and indexResume methods may be used to change the state "
		"of the index. For attribute and element definitions the changeItemState "
		"method may be used to change the state of the element or attribute. For "
		"changeItemState we need to verify that the state change is legal. Certain "
		"state transitions are illegal for a user to make even using changeItemState."
		"Test to make sure that element and attribute definitions cannot be deleted "
		"directly using the DOM APIs if the state attribute is not set to \"unused\"."
		"Verify that state is only settable to \"unused\" by the sweep method - users "
		"should not be able to do this. The sweep method should be tested to make sure "
		"it does not set the state to \"unused\" if the element or attribute definition "
		"is actually in use somewhere.",

		"(1) Call indexSuspend and indexResume on the index definition "
		"(2) Verify the state is changed correctly (3) Set state attribute of "
		"the index element and attribute definitions to \"checking\" (4) Call "
		"sweep (5) Verify all states are correct (6) Verify user cannot set state "
		"to \"unused\" directly (7) for each definition in reverse order of creation: "
		"(a) call sweep to set state to \"unused\" (b) delete definition.",
		"");

	//suspend the index
	if ( RC_BAD( rc = m_pDb->indexSuspend( uiIndexDictNum)))
	{
		MAKE_ERROR_STRING( "indexSuspend failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pIndex->getAttribute(
		m_pDb,
		ATTR_STATE_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "getAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->getUTF8( m_pDb, szBuf, sizeof( szBuf), 0,
		sizeof(szBuf) - 1)))
	{
		MAKE_ERROR_STRING( "getUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	//NOTE: There is a define FLM_INDEX_SUSPENDED_STR in flaim.h
	//should I have access to it in flaimpub.h?
	if ( f_strcmp( (char *)szBuf, "suspended"))
	{
		//MAKE_ERROR_STRING macro won't work here.
		f_sprintf(
			(char*)m_szDetails,
			"index state: %s. expected \"suspended\". "
			"rc == %X. File: %s. Line# %u. ",
			szBuf,
			(unsigned)rc,
			__FILE__,
			__LINE__);
		goto Exit;
	}

	//resume the index
	if ( RC_BAD( rc = m_pDb->indexResume( uiIndexDictNum)))
	{
		MAKE_ERROR_STRING( "indexResume failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pIndex->getAttribute(
		m_pDb,
		ATTR_STATE_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "getAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->getUTF8( m_pDb, szBuf, sizeof(szBuf), 0,
		sizeof(szBuf) - 1)))
	{
		MAKE_ERROR_STRING( "getUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}
	if ( f_strcmp( (char *)szBuf, "online") &&
		f_strcmp( (char *)szBuf, "offline")) //VISIT - may still be offline ?
	{
		//MAKE_ERROR_STRING macro won't work here.
		f_sprintf(
			(char*)m_szDetails,
			"index state: %s. expected \"online\" or \"offline\". "
			"rc == %X. File: %s. Line# %u. ",
			szBuf,
			(unsigned)rc,
			__FILE__,
			__LINE__);
		goto Exit;
	}

	//now set it to unused the right way
	if ( RC_BAD( rc = m_pDb->changeItemState(
		ELM_ATTRIBUTE_TAG,
		uiAttrDictNum,
		"checking")))
	{
		MAKE_ERROR_STRING( "changeItemState failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = FALSE;

	for( ;;)
	{
		if ( RC_BAD( rc = pAttrDef->getAttribute(
			m_pDb,
			ATTR_STATE_TAG,
			&pAttr)))
		{
			MAKE_ERROR_STRING( "getAttribute failed.", m_szDetails, rc);
			goto Exit;
		}
	
		if ( RC_BAD( rc = pAttr->getUTF8( m_pDb, szBuf, sizeof(szBuf), 0,
			sizeof(szBuf) - 1)))
		{
			MAKE_ERROR_STRING( "getUTF8 failed.", m_szDetails, rc);
			goto Exit;
		}
	
		if( !f_strcmp( (char *)szBuf, "active"))
		{
			break;
		}
		
		f_sleep( 250);
	}

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;
	
	if ( RC_BAD( rc = m_pDb->changeItemState(
		ELM_ELEMENT_TAG,
		uiElemDictNum,
		"checking")))
	{
		MAKE_ERROR_STRING( "changeItemState failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = FALSE;

	for( ;;)
	{
		if ( RC_BAD( rc = pElemDef->getAttribute(
			m_pDb,
			ATTR_STATE_TAG,
			&pAttr)))
		{
			MAKE_ERROR_STRING( "getAttribute failed.", m_szDetails, rc);
			goto Exit;
		}
		
		if ( RC_BAD( rc = pAttr->getUTF8( m_pDb, szBuf, sizeof(szBuf),
			0, sizeof(szBuf) - 1)))
		{
			MAKE_ERROR_STRING( "getUTF8 failed.", m_szDetails, rc);
			goto Exit;
		}
		
		if( !f_strcmp( (char *)szBuf, "active"))
		{
			break;
		}
		
		f_sleep( 250);
	}

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;
	
	//shouldn't be able to set to "unused" this way
	
	rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"unused");
	if ( rc != NE_XFLM_READ_ONLY)
	{
		MAKE_ERROR_STRING( "Invalid return code when trying to modify "
			"state attribute.", m_szDetails, rc);
		goto Exit;
	}

	//delete the IndexDef
	if ( RC_BAD( rc = pIndex->deleteNode( m_pDb)))
	{
		f_sprintf(
			(char*)m_szDetails,
			"failed to delete index definition."
			"Line# %u. rc == %X.",
			__LINE__,
			(unsigned)rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->changeItemState(
		ELM_ELEMENT_TAG,
		uiElemDictNum,
		"checking")))
	{
		MAKE_ERROR_STRING( "changeItemState failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = FALSE;

	for( ;;)
	{
		if( RC_BAD( rc = pElemDef->getNodeId( m_pDb, &ui64Tmp)))
		{
			if( rc != NE_XFLM_DOM_NODE_DELETED)
			{
				flmAssert( 0);
				goto Exit;
			}
			
			rc = NE_XFLM_OK;
			break;
		}
		
		f_sleep( 250);
	}

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;
	
	if ( RC_BAD( rc = m_pDb->changeItemState(
		ELM_ATTRIBUTE_TAG,
		uiAttrDictNum,
		"checking")))
	{
		MAKE_ERROR_STRING( "changeItemState failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = FALSE;

	for( ;;)
	{
		if( RC_BAD( rc = pAttrDef->getNodeId( m_pDb, &ui64Tmp)))
		{
			if( rc != NE_XFLM_DOM_NODE_DELETED)
			{
				flmAssert( 0);
				goto Exit;
			}
			
			rc = NE_XFLM_OK;
			break;
		}
		
		f_sleep( 250);
	}

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;
	
	endTest("PASS");

	m_pszTestName = "Dictionary Doc #1 Test";
	m_pszTestDesc = "Verify that no nodes in document #1 of the dictionary collection "
		"can be modified or deleted by an application. This is a special document "
		"used internally by FLAIM so it should not be changeable or deletable by "
		"an application. It is readable but not changeable or deletable.";

	m_pszSteps = "(1) Get Doc #1 (2) Try to delete it. ";

	beginTest(	
		"Dictionary Doc #1 Test",
		"Verify that no nodes in document #1 of the dictionary collection "
		"can be modified or deleted by an application. This is a special document "
		"used internally by FLAIM so it should not be changeable or deletable by "
		"an application. It is readable but not changeable or deletable.",
		"(1) Get Doc #1 (2) Try to delete it. ",
		"");

	//get document #1
	if ( RC_BAD( rc = m_pDb->getNode(
		XFLM_DICT_COLLECTION,
		1,
		&pDocument)))
	{
		MAKE_ERROR_STRING( "getNode failed.", m_szDetails, rc);
		goto Exit;
	}

	//should not be deleteable
	rc = pDocument->deleteNode( m_pDb);
	if ( rc != NE_XFLM_DELETE_NOT_ALLOWED)
	{
		MAKE_ERROR_STRING( "Unexpected rc from attempt to delete dict. doc #1."
			, m_szDetails, rc);
		goto Exit;
	}

	//this failed delete forces us to abort the trans
	if ( RC_BAD( rc = m_pDb->transAbort()))
	{
		MAKE_ERROR_STRING( "transAbort failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = FALSE;

	endTest("PASS");

	m_pszTestName = "Collection Definition Test";
	m_pszTestDesc = "Verify rules for collection definitions are enforced";
	m_pszSteps = "(1) Create a collection definition (2) Try to delete its name";

	beginTest(
		"Collection Definition Test",
		"Verify rules for collection definitions are enforced",
		"(1) Create a collection definition (2) Try to delete its name",
		"");

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;

	if ( RC_BAD( rc = m_pDb->createRootElement(
		XFLM_DICT_COLLECTION,
		ELM_COLLECTION_TAG,
		&pCollection)))
	{
		MAKE_ERROR_STRING( "createRootElement failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pCollection->createAttribute(
		m_pDb,
		ATTR_NAME_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"My Collection")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->documentDone( pCollection)))
	{
		MAKE_ERROR_STRING( "documentDone failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pCollection->getAttribute(
		m_pDb,
		ATTR_NAME_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "getAttribute failed", m_szDetails, rc);
		goto Exit;
	}

	rc = pAttr->deleteNode( m_pDb);
	if ( rc != NE_XFLM_DELETE_NOT_ALLOWED)
	{
		MAKE_ERROR_STRING( "Invalid rc returned when trying to delete "
			"\"name\" attribute. Expected: NE_XFLM_DELETE_NOT_ALLOWED",
			m_szDetails, rc);
		if( RC_OK( rc))
		{
			rc = NE_XFLM_FAILURE;
		}
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->transAbort()))
	{
		MAKE_ERROR_STRING( "transAbort failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = FALSE;

	endTest("PASS");

Exit:

	if ( RC_BAD( rc))
	{
		endTest("FAIL");
	}

	if (bTransStarted)
	{
		m_pDb->transCommit();
	}

	if ( pNode)
	{
		pNode->Release();
	}

	if ( pAttr)
	{
		pAttr->Release();
	}

	if ( pAttrDef)
	{
		pAttrDef->Release();
	}

	if ( pElemDef)
	{
		pElemDef->Release();
	}

	if ( pDocument)
	{
		pDocument->Release();
	}

	if ( pIndex )
	{
		pIndex->Release();
	}

	if( pCollection)
	{
		pCollection->Release();
	}

	shutdownTestState( DB_NAME_STR, bDibCreated);
	return( rc);
}
