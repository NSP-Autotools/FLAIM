//------------------------------------------------------------------------------
// Desc:	Index definition unit test.
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
class IIndexDefTestImpl : public TestBase
{
public:

	const char * getName( void);
	
	RCODE execute( void);

private:

	RCODE importEncDefs( void);

	FLMUINT			m_uiAESDef;
	FLMUINT			m_uiDES3Def;
};

/****************************************************************************
Desc:
****************************************************************************/
RCODE getTest( 
	IFlmTest **		ppTest)
{
	RCODE		rc = NE_XFLM_OK;

	if( (*ppTest = f_new IIndexDefTestImpl) == NULL)
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
const char * IIndexDefTestImpl::getName( void)
{
	return "Index Definition Test";
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IIndexDefTestImpl::importEncDefs( void)
{
	RCODE			rc = NE_XFLM_OK;
	const char *ppszEncDefs[] = {XFLM_ENC_AES_OPTION_STR, XFLM_ENC_DES3_OPTION_STR};
	FLMUINT		puiEncDef[2];
	char			szEncDef[200];
	FLMUINT		uiLoop = 0;

	for( 
		uiLoop = 0; 
		uiLoop < sizeof( ppszEncDefs) / sizeof( ppszEncDefs[ 0]); 
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

		if( RC_BAD( rc = importBuffer( szEncDef, XFLM_DICT_COLLECTION)))
		{
			MAKE_ERROR_STRING( "importBuffer failed.", m_szDetails, rc);
			goto Exit;
		}
		
		f_sprintf( szEncDef, "%s definition", ppszEncDefs[uiLoop]);
		if( RC_BAD( rc = m_pDb->getEncDefId( (char *)szEncDef,
														 (FLMUINT *)&puiEncDef[ uiLoop])))
		{
			goto Exit;
		}
	}

	m_uiAESDef = puiEncDef[ 0];
	m_uiDES3Def = puiEncDef[ 1];

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IIndexDefTestImpl::execute( void)
{
	RCODE						rc = NE_XFLM_OK;
	FLMBOOL					bDibCreated = FALSE;
	FLMUINT					uiAttrDefDictNum = 0;
	FLMUINT					uiElemDef1DictNum = 0;
	FLMUINT					uiElemDef2DictNum = 0;
	FLMUINT64				ui64ElemComponent = 0;
	FLMUINT					uiTemp = 0;
	IF_DOMNode *			pIndex			= NULL;
	IF_DOMNode *			pNode				= NULL;
	IF_DOMNode *			pAttr				= NULL;
	IF_DOMNode *			pAttrDef			= NULL;
	IF_DOMNode *			pElemDef			= NULL;
	IF_DOMNode *			pTmpNode			= NULL;
	FLMBOOL					bTransBegun = FALSE;

	beginTest(
		"Index Def Test Init",
		"Perform necessary initializations so we can do the "
		"index definition tests.",
		"1) Get the necessary class factories "
		"2) Create a database 3) create some attribute and element "
		"definitions.",
		"");

	if ( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		MAKE_ERROR_STRING( "Failed to initialize test state.", m_szDetails, rc);
		goto Exit;
	}
	bDibCreated = TRUE;

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

	// Create an attribute def and an element def for us to refer to in the index

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
		ATTR_DICT_NUMBER_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "getAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	// Get the dictionary number (used by later tests)

	if ( RC_BAD( rc = pAttr->getUINT(
		m_pDb,
		&uiAttrDefDictNum)))
	{
		MAKE_ERROR_STRING( "getUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	// Create an element definition

	/*
	<name="bar" type="string" state="active" />
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

	if ( RC_BAD( rc = m_pDb->documentDone( pElemDef)))
	{
		MAKE_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pElemDef->getAttribute(
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
		&uiElemDef1DictNum)))
	{
		MAKE_ERROR_STRING( "getUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	// Create an element definition

	/*
	<name="baz" type="string" state="active" foo />
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

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"baz")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	// Give baz a foo attribute

	if ( RC_BAD( rc = pElemDef->createAttribute(
		m_pDb,
		uiAttrDefDictNum,
		&pAttr)))
	{
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->documentDone( pElemDef)))
	{
		MAKE_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pElemDef->getAttribute(
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
		&uiElemDef2DictNum)))
	{
		MAKE_ERROR_STRING( "getUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	endTest("PASS");

	beginTest(
		"Root Element Tests",
		"Verify rules for having a \"/\" element name are enforced",
		"Verify that an element name of \"/\" is only used once and can "
		"only be used at the root component of an index definition. Verify that it "
		"is not allowed to have any siblings and if it is a key component that "
		"the \"indexOn\" attribute is required to be set to \"presence\". Furthermore "
		"verify that it cannot be used as a data component. ",
		"No additional info");

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	if ( RC_BAD( rc = m_pDb->createRootElement(
		XFLM_DICT_COLLECTION,
		ELM_INDEX_TAG,
		&pIndex)))
	{
		MAKE_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
		goto Exit;
	}

	// Create the root element component

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

	if ( RC_BAD( rc = pIndex->createAttribute(
		m_pDb,
		ATTR_NAME_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb,
		(FLMBYTE *)"root element test index 1")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

#ifdef FLM_USE_NICI
	// Create the encryption Id attribute
	if ( RC_BAD( rc = pIndex->createAttribute(
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

	// Create the key size parameter.
	if ( RC_BAD( rc = pIndex->createAttribute(
		m_pDb,
		ATTR_ENCRYPTION_KEY_SIZE_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUINT(
		m_pDb,
		192)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

#endif

	// Create the name attribute and set it to "/"

	if ( RC_BAD( rc = pNode->createAttribute(
		m_pDb,
		ATTR_NAME_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"/")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	// Make this the first key component

	if ( RC_BAD( rc = pNode->createAttribute(
		m_pDb,
		ATTR_KEY_COMPONENT_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUINT(
		m_pDb,
		1)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
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

	// The indexOn attribute should be required to be set to "presence"

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"value")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	// documentDone should fail since we attempted to set the indexOn attribute
	// to something other than "presence" with an element component of "/"

	rc = m_pDb->documentDone( pIndex);

	if ( rc != NE_XFLM_MUST_INDEX_ON_PRESENCE)
	{
		MAKE_ERROR_STRING( "Invalid rc. Expected: NE_XFLM_MUST_INDEX_ON_PRESENCE.",
			m_szDetails, rc);

		if ( RC_OK( rc))
		{
			rc = NE_XFLM_FAILURE;
		}
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transAbort()))
	{
		MAKE_ERROR_STRING( "transAbort failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	// Try again

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	if ( RC_BAD( rc = m_pDb->createRootElement(
		XFLM_DICT_COLLECTION,
		ELM_INDEX_TAG,
		&pIndex)))
	{
		MAKE_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
		goto Exit;
	}

	// Create the root element component

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

	if ( RC_BAD( rc = pIndex->createAttribute(
		m_pDb,
		ATTR_NAME_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUTF8(
		m_pDb,
		(FLMBYTE *)"root element test index 2")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

#ifdef FLM_USE_NICI
	// Create the encryption Id attribute
	if ( RC_BAD( rc = pIndex->createAttribute(
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

	// Create the key size parameter.
	if ( RC_BAD( rc = pIndex->createAttribute(
		m_pDb,
		ATTR_ENCRYPTION_KEY_SIZE_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUINT(
		m_pDb,
		128)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}
	
#endif

	// Create the name attribute and set it to "/"

	if ( RC_BAD( rc = pNode->createAttribute(
		m_pDb,
		ATTR_NAME_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"/")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	// Make this the first key component

	if ( RC_BAD( rc = pNode->createAttribute(
		m_pDb,
		ATTR_KEY_COMPONENT_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUINT(
		m_pDb,
		1)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
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

	// Set to "presence" as we should

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"presence")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	// We should not be able to create a sibling to this component

	if ( RC_BAD( rc = pIndex->createNode(
		m_pDb,
		ELEMENT_NODE,
		ELM_ELEMENT_COMPONENT_TAG,
		XFLM_FIRST_CHILD,
		&pTmpNode)))
	{
		MAKE_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	// Have to give it at least one attribute or we'll fail in documentDone

	if ( RC_BAD( rc = pTmpNode->createAttribute(
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

	// documentDone should fail since the "/" element has a sibling

	rc = m_pDb->documentDone( pIndex);
	if ( rc != NE_XFLM_BAD_USE_OF_ELM_ROOT_TAG)
	{
		MAKE_ERROR_STRING( "Invalid rc. Expected: NE_XFLM_BAD_USE_OF_ELM_ROOT_TAG.",
			m_szDetails, rc);

		if ( RC_OK( rc))
		{
			rc = NE_XFLM_FAILURE;
		}
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transAbort()))
	{
		MAKE_ERROR_STRING( "transAbort failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;


	// Try again

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	if ( RC_BAD( rc = m_pDb->createRootElement(
		XFLM_DICT_COLLECTION,
		ELM_INDEX_TAG,
		&pIndex)))
	{
		MAKE_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
		goto Exit;
	}

	// Create the root element component

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

	if ( RC_BAD( rc = pIndex->createAttribute(
		m_pDb,
		ATTR_NAME_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUTF8(
		m_pDb,
		(FLMBYTE *)"root element test index 3")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

#ifdef FLM_USE_NICI
	// Create the encryption Id attribute
	if ( RC_BAD( rc = pIndex->createAttribute(
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

	// Create the key size parameter.
	if ( RC_BAD( rc = pIndex->createAttribute(
		m_pDb,
		ATTR_ENCRYPTION_KEY_SIZE_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUINT(
		m_pDb,
		256)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}
	
#endif

	// Create the name attribute and set it to "/"

	if ( RC_BAD( rc = pNode->createAttribute(
		m_pDb,
		ATTR_NAME_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"/")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	// Make this the first key component

	if ( RC_BAD( rc = pNode->createAttribute(
		m_pDb,
		ATTR_KEY_COMPONENT_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUINT(
		m_pDb,
		1)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
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

	// Set to "presence" as we should

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"presence")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	// We should not be able to create a sibling to this component

	if ( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	if ( RC_BAD( rc = m_pDb->transBegin(XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	if ( RC_BAD( rc = pIndex->createNode(
		m_pDb,
		ELEMENT_NODE,
		ELM_ELEMENT_COMPONENT_TAG,
		XFLM_FIRST_CHILD,
		&pTmpNode)))
	{
		MAKE_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	// Have to give it at least one attribute or we'll fail in documentDone

	if ( RC_BAD( rc = pTmpNode->createAttribute(
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

	rc = m_pDb->documentDone( pIndex);

	if ( rc != NE_XFLM_BAD_USE_OF_ELM_ROOT_TAG)
	{
		MAKE_ERROR_STRING( "Incorrect rc from documentDone."
			" Expected: NE_XFLM_BAD_USE_OF_ELM_ROOT_TAG.", m_szDetails, rc);

		if ( RC_OK( rc))
		{
			rc = NE_XFLM_FAILURE;
		}
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->transAbort()))
	{
		MAKE_ERROR_STRING( "transAbort failed", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	// Create a subordinate element component to the "/"

	if ( RC_BAD( rc = pNode->createNode(
		m_pDb,
		ELEMENT_NODE,
		ELM_ELEMENT_COMPONENT_TAG,
		XFLM_FIRST_CHILD,
		&pNode)))
	{
		MAKE_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	// Try to set its name attribute to "/" as well

	if ( RC_BAD( rc = pNode->createAttribute(
		m_pDb,
		ATTR_NAME_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"/")))
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

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"presence")))
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

	if ( RC_BAD( rc = pAttr->setUINT(
		m_pDb,
		2)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	// This call to documentDone should fail since we tried to
	// re-use the "/"
	rc = m_pDb->documentDone( pIndex);
	if ( rc != NE_XFLM_BAD_USE_OF_ELM_ROOT_TAG)
	{
		MAKE_ERROR_STRING( "Incorrect rc from documentDone."
			" Expected: NE_XFLM_BAD_USE_OF_ELM_ROOT_TAG.", m_szDetails, rc);

		if ( RC_OK( rc))
		{
			rc = NE_XFLM_FAILURE;
		}
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transAbort()))
	{
		MAKE_ERROR_STRING( "transAbort failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	// Try one last time

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	if ( RC_BAD( rc = m_pDb->createRootElement(
		XFLM_DICT_COLLECTION,
		ELM_INDEX_TAG,
		&pIndex)))
	{
		MAKE_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
		goto Exit;
	}

	// Create the root element component

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

	if ( RC_BAD( rc = pIndex->createAttribute(
		m_pDb,
		ATTR_NAME_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb,
		(FLMBYTE *)"root element test index 4")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

#ifdef FLM_USE_NICI
	// Create the encryption Id attribute
	if ( RC_BAD( rc = pIndex->createAttribute(
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
	
	// Create the key size parameter.
	if ( RC_BAD( rc = pIndex->createAttribute(
		m_pDb,
		ATTR_ENCRYPTION_KEY_SIZE_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUINT(
		m_pDb,
		256)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}
	
#endif

	// Create the name attribute and set it to "/"

	if ( RC_BAD( rc = pNode->createAttribute(
		m_pDb,
		ATTR_NAME_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"/")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	// Make this the first key component

	if ( RC_BAD( rc = pNode->createAttribute(
		m_pDb,
		ATTR_KEY_COMPONENT_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUINT(
		m_pDb,
		1)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
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

	// Set to "presence" as we should

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"presence")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->documentDone( pIndex)))
	{
		MAKE_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;


	if ( RC_BAD( rc = m_pDb->transBegin(XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	// We should not be able to create a sibling to this component

	if ( RC_BAD( rc = pIndex->createNode(
		m_pDb,
		ELEMENT_NODE,
		ELM_ELEMENT_COMPONENT_TAG,
		XFLM_FIRST_CHILD,
		&pTmpNode)))
	{
		MAKE_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	// Have to give it at least one attribute or we'll fail in documentDone

	if ( RC_BAD( rc = pTmpNode->createAttribute(
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

	// This call should not work
	rc = m_pDb->documentDone( pIndex);

	if ( rc != NE_XFLM_BAD_USE_OF_ELM_ROOT_TAG)
	{
		MAKE_ERROR_STRING( 
			"Unexpected rc from documentDone on bad index def.", 
			m_szDetails, rc);
		if ( RC_OK( rc))
		{
			rc = RC_SET( NE_XFLM_FAILURE);
		}
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->transAbort()))
	{
		MAKE_ERROR_STRING( "transAbort failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	// This call should work since we corrected all of the errors
	if ( RC_BAD( rc = m_pDb->documentDone( pIndex)))
	{
		MAKE_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	endTest("PASS");

	beginTest(
		"Sibling Element Uniqueness Test",
		m_pszTestDesc = "Test to verify uniqueness rules among sibling elements",
		"Verify that sibling elements and attributes are unique "
		"- that we cannot use the same one twice if they are sibling components.",
		"No additional info");

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD ( rc = m_pDb->createRootElement(
		XFLM_DICT_COLLECTION,
		ELM_INDEX_TAG,
		&pIndex)))
	{
		MAKE_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pIndex->createAttribute(
		m_pDb,
		ATTR_NAME_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb,
		(FLMBYTE *)"duplicate components index")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

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

	// Create a duplicate

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
		ATTR_KEY_COMPONENT_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD ( rc = pAttr->setUINT( m_pDb, 2)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
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

	// This should fail because there are duplicate key components

	rc = m_pDb->documentDone( pIndex);
	if ( rc != NE_XFLM_DUP_SIBLING_IX_COMPONENTS)
	{
		MAKE_ERROR_STRING( "Invalid rc from documentDone. Expected: "
			"NE_XFLM_DUP_SIBLING_IX_COMPONENTS.", m_szDetails, rc);
		if ( RC_OK( rc))
		{
			rc = NE_XFLM_FAILURE;
		}
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transAbort()))
	{
		MAKE_ERROR_STRING( "transAbort failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	// Try again

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	if ( RC_BAD ( rc = m_pDb->createRootElement(
		XFLM_DICT_COLLECTION,
		ELM_INDEX_TAG,
		&pIndex)))
	{
		MAKE_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pIndex->createAttribute(
		m_pDb,
		ATTR_NAME_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb,
		(FLMBYTE *)"duplicate components index")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

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

	// Create a duplicate

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
		ATTR_KEY_COMPONENT_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD ( rc = pAttr->setUINT( m_pDb, 2)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
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

	// Clean up our mess

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"baz")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pNode->createNode(
		m_pDb,
		ELEMENT_NODE,
		ELM_ATTRIBUTE_COMPONENT_TAG,
		XFLM_FIRST_CHILD,
		&pTmpNode)))
	{
		MAKE_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pTmpNode->createAttribute(
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

	if ( RC_BAD( rc = pTmpNode->createAttribute(
		m_pDb,
		ATTR_DATA_COMPONENT_TAG,
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

	// Create a duplicate attribute component

	if ( RC_BAD( rc = pNode->createNode(
		m_pDb,
		ELEMENT_NODE,
		ELM_ATTRIBUTE_COMPONENT_TAG,
		XFLM_FIRST_CHILD,
		&pTmpNode)))
	{
		MAKE_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pTmpNode->createAttribute(
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

	if ( RC_BAD( rc = pTmpNode->createAttribute(
		m_pDb,
		ATTR_DATA_COMPONENT_TAG,
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

	// This should fail because of our duplicate attr components

	rc = m_pDb->documentDone( pIndex);
	if ( rc != NE_XFLM_DUP_SIBLING_IX_COMPONENTS)
	{
		MAKE_ERROR_STRING( "Invalid rc from documentDone."
			"Expected: NE_XFLM_DUP_SIBLING_IX_COMPONENTS", m_szDetails, rc);
		if ( RC_OK( rc))
		{
			rc = NE_XFLM_FAILURE;
		}
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transAbort()))
	{
		MAKE_ERROR_STRING( "transAbort failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	// Try one more time

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	if ( RC_BAD ( rc = m_pDb->createRootElement(
		XFLM_DICT_COLLECTION,
		ELM_INDEX_TAG,
		&pIndex)))
	{
		MAKE_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pIndex->createAttribute(
		m_pDb,
		ATTR_NAME_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb,
		(FLMBYTE *)"duplicate components index")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

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

	// Create a duplicate

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
		ATTR_KEY_COMPONENT_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD ( rc = pAttr->setUINT( m_pDb, 2)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
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

	// Clean up our mess

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"baz")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pNode->createNode(
		m_pDb,
		ELEMENT_NODE,
		ELM_ATTRIBUTE_COMPONENT_TAG,
		XFLM_FIRST_CHILD,
		&pTmpNode)))
	{
		MAKE_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pTmpNode->createAttribute(
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

	if ( RC_BAD( rc = pTmpNode->createAttribute(
		m_pDb,
		ATTR_DATA_COMPONENT_TAG,
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
	bTransBegun = TRUE;

	// Create a duplicate attribute component

	if ( RC_BAD( rc = pNode->createNode(
		m_pDb,
		ELEMENT_NODE,
		ELM_ATTRIBUTE_COMPONENT_TAG,
		XFLM_FIRST_CHILD,
		&pTmpNode)))
	{
		MAKE_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pTmpNode->createAttribute(
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

	if ( RC_BAD( rc = pTmpNode->createAttribute(
		m_pDb,
		ATTR_DATA_COMPONENT_TAG,
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

	rc =  m_pDb->documentDone( pIndex);

	if ( rc != NE_XFLM_DUP_SIBLING_IX_COMPONENTS)
	{
		MAKE_ERROR_STRING( "Incorrect rc from documentDone."
			" Expected: NE_XFLM_DUP_SIBLING_IX_COMPONENTS.", m_szDetails, rc);

		if ( RC_OK( rc))
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
	bTransBegun = FALSE;

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	// We should be okay now

	if ( RC_BAD( rc = m_pDb->documentDone( pIndex)))
	{
		MAKE_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	endTest("PASS");

	beginTest(
		"Key/Data Component Value Test",
		"Verify that the key component number attribute "
		"(ATTR_KEY_COMPONENT_TAG) and the data component number attribute "
		"(ATTR_DATA_COMPONENT_TAG) are not set to a value of zero.",
		"Self-Explanatory.",
		"");

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	if ( RC_BAD( rc = pIndex->getFirstChild(
		m_pDb,
		&pNode)))
	{
		MAKE_ERROR_STRING( "getFirstChild failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( pNode->getAttribute(
		m_pDb,
		ATTR_KEY_COMPONENT_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "getAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->getUINT(
		m_pDb,
		&uiTemp)))
	{
		MAKE_ERROR_STRING( "getUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( uiTemp == 0)
	{
		MAKE_ERROR_STRING( "Invalid key component number.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pNode->getFirstChild(
		m_pDb,
		&pNode)))
	{
		MAKE_ERROR_STRING( "getFirstChild failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( pNode->getAttribute(
		m_pDb,
		ATTR_DATA_COMPONENT_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "getAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->getUINT(
		m_pDb,
		&uiTemp)))
	{
		MAKE_ERROR_STRING( "getUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( uiTemp == 0)
	{
		MAKE_ERROR_STRING( "Invalid key component number.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	endTest("PASS");

	beginTest(
		"Component Sequence Number Test",
		"Verify rules regarding key/data component sequence numbers "
		"are enforced. ",
		"Verify that FLAIM disallows a sequence of key component "
		"numbers or data component numbers that has gaps (e.g. 1 3 4 5 - 2 is "
		"missing) or that does not begin with 1. Also verify that each component "
		"number is used once and only once. Also verify that FLAIM requires at "
		"least one key component but does not require at least one data component.",
		"");

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	if ( RC_BAD( rc = m_pDb->createRootElement(
		XFLM_DICT_COLLECTION,
		ELM_INDEX_TAG,
		&pIndex)))
	{
		MAKE_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pIndex->createAttribute(
		m_pDb,
		ATTR_NAME_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"Sequence Test")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	// This should not work because FLAIM requires at least one key component

	rc = m_pDb->documentDone( pIndex);

	if ( rc != NE_XFLM_ILLEGAL_INDEX_DEF)
	{
		MAKE_ERROR_STRING( "Incorrect rc from documentDone. "
			"Expected: NE_XFLM_ILLEGAL_INDEX_DEF.", m_szDetails, rc);
		if ( RC_OK( rc))
		{
			rc = NE_XFLM_FAILURE;
		}
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transAbort()))
	{
		MAKE_ERROR_STRING( "transAbort failed.", m_szDetails, rc);
		goto Exit;
	}

	// Try again

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->createRootElement(
		XFLM_DICT_COLLECTION,
		ELM_INDEX_TAG,
		&pIndex)))
	{
		MAKE_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pIndex->createAttribute(
		m_pDb,
		ATTR_NAME_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"Sequence Test")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

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

	// Try to give this first component a key comp != 1

	if ( RC_BAD( rc = pNode->createAttribute(
		m_pDb,
		ATTR_KEY_COMPONENT_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD ( rc = pAttr->setUINT( m_pDb, 7)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	// This should fail because the first index key component != 1

	rc = m_pDb->documentDone( pIndex);

	if ( rc != NE_XFLM_MISSING_KEY_COMPONENT)
	{
		MAKE_ERROR_STRING( "Invalid rc from documentDone. Expected: "
			"NE_XFLM_MISSING_KEY_COMPONENT.", m_szDetails, rc);
		if ( RC_OK( rc))
		{
			rc = NE_XFLM_FAILURE;
		}
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transAbort()))
	{
		MAKE_ERROR_STRING( "transAbort failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	// Try again

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	if ( RC_BAD( rc = m_pDb->createRootElement(
		XFLM_DICT_COLLECTION,
		ELM_INDEX_TAG,
		&pIndex)))
	{
		MAKE_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pIndex->createAttribute(
		m_pDb,
		ATTR_NAME_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"Sequence Test")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

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

	// Try to give this first component a key comp != 1

	if ( RC_BAD( rc = pNode->createAttribute(
		m_pDb,
		ATTR_KEY_COMPONENT_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD ( rc = pAttr->setUINT( m_pDb, 7)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	// Fix our mistake

	if ( RC_BAD( rc = pAttr->setUINT( m_pDb, 1)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	// Create a new element component

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

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"baz")))
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

	// Try to give this component the same sequence number as the first

	if ( RC_BAD( rc = pAttr->setUINT(
		m_pDb,
		1)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	// This should fail because of duplicate sequence numbers

	rc = m_pDb->documentDone( pIndex);

	if ( rc != NE_XFLM_DUPLICATE_KEY_COMPONENT)
	{
		MAKE_ERROR_STRING( "Invalid rc from documentDone. "
			"Expected: NE_XFLM_DUPLICATE_KEY_COMPONENT.", m_szDetails, rc);
		if ( RC_OK( rc))
		{
			rc = NE_XFLM_FAILURE;
		}
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transAbort()))
	{
		MAKE_ERROR_STRING( "transAbort failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	// Try again

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	if ( RC_BAD( rc = m_pDb->createRootElement(
		XFLM_DICT_COLLECTION,
		ELM_INDEX_TAG,
		&pIndex)))
	{
		MAKE_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pIndex->createAttribute(
		m_pDb,
		ATTR_NAME_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"Sequence Test")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

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

	// Try to give this first component a key comp != 1

	if ( RC_BAD( rc = pNode->createAttribute(
		m_pDb,
		ATTR_KEY_COMPONENT_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD ( rc = pAttr->setUINT( m_pDb, 7)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	// Fix our mistake

	if ( RC_BAD( rc = pAttr->setUINT( m_pDb, 1)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	// Create a new element component

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

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"baz")))
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

	// Try to give this component the same sequence number as the first

	if ( RC_BAD( rc = pAttr->setUINT(
		m_pDb,
		1)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	// Change the number for one out of sequence

	if ( RC_BAD( rc = pAttr->setUINT( m_pDb, 8)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	// This should fail because of holes in the sequence
	rc = m_pDb->documentDone( pIndex);

	if ( rc != NE_XFLM_MISSING_KEY_COMPONENT)
	{
		MAKE_ERROR_STRING( "Invalid rc from documentDone. "
			"Expected: NE_XFLM_MISSING_KEY_COMPONENT.", m_szDetails, rc);
		if ( RC_OK( rc))
		{
			rc = NE_XFLM_FAILURE;
		}
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transAbort()))
	{
		MAKE_ERROR_STRING( "transAbort failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	// Try again

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	if ( RC_BAD( rc = m_pDb->createRootElement(
		XFLM_DICT_COLLECTION,
		ELM_INDEX_TAG,
		&pIndex)))
	{
		MAKE_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pIndex->createAttribute(
		m_pDb,
		ATTR_NAME_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"Sequence Test")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

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

	// Try to give this first component a key comp != 1

	if ( RC_BAD( rc = pNode->createAttribute(
		m_pDb,
		ATTR_KEY_COMPONENT_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD ( rc = pAttr->setUINT( m_pDb, 7)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	// Fix our mistake

	if ( RC_BAD( rc = pAttr->setUINT( m_pDb, 1)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	// Create a new element component

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

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"baz")))
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

	// Try to give this component the same sequence number as the first

	if ( RC_BAD( rc = pAttr->setUINT(
		m_pDb,
		1)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	// Change the number for one out of sequence

	if ( RC_BAD( rc = pAttr->setUINT( m_pDb, 8)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	// Correct our mistake

	if ( RC_BAD( rc = pAttr->setUINT( m_pDb, 2)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	rc = m_pDb->documentDone( pIndex);

	if( RC_BAD( rc))
	{
		MAKE_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	endTest("PASS");

	beginTest(
		"Attribute Component Test",
		"Verify that attribute components are not allowed to have child "
		"components. Verify that attribute components are required to be marked as "
		"either a key component or a data component.",
		"1) Create an Attribute Component 2) Create a child for that "
		"attribute component. 3) Verify documentDone fails with correct rc 4) "
		"Delete the child node and verify documentDone succeeds.",
		"");

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	if( RC_BAD( rc = pNode->getNodeId( m_pDb, &ui64ElemComponent)))
	{
		goto Exit;
	}
	
	// pNode is pointing to an element component node.
	// Create an attr component underneath

	if ( RC_BAD( pNode->createNode(
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

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"foo")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD ( rc = pNode->createAttribute(
		m_pDb,
		ATTR_KEY_COMPONENT_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUINT( m_pDb, 3)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	// Try to add another attribute component under this one

	if ( RC_BAD( rc = pNode->createNode(
		m_pDb,
		ELEMENT_NODE,
		ELM_ATTRIBUTE_COMPONENT_TAG,
		XFLM_FIRST_CHILD,
		&pTmpNode)))
	{
		MAKE_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	// This should not work because of the nested attribute component

	rc = m_pDb->documentDone( pIndex);
	if ( rc != NE_XFLM_ILLEGAL_INDEX_DEF)
	{
		MAKE_ERROR_STRING( "Incorrect rc from documentDone. "
			"Expected: NE_XFLM_ILLEGAL_INDEX_DEF", m_szDetails, rc);
		if ( RC_OK( rc))
		{
			rc = NE_XFLM_FAILURE;
		}
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transAbort()))
	{
		MAKE_ERROR_STRING( "transAbort failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	// Try again

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	if( RC_BAD( rc = m_pDb->getNode( 
		XFLM_DICT_COLLECTION, ui64ElemComponent, &pNode)))
	{
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

	if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"foo")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD ( rc = pNode->createAttribute(
		m_pDb,
		ATTR_KEY_COMPONENT_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUINT( m_pDb, 3)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	// Try to add another attribute component under this one

	if ( RC_BAD( rc = pNode->createNode(
		m_pDb,
		ELEMENT_NODE,
		ELM_ATTRIBUTE_COMPONENT_TAG,
		XFLM_FIRST_CHILD,
		&pTmpNode)))
	{
		MAKE_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pTmpNode->deleteNode( m_pDb)))
	{
		MAKE_ERROR_STRING( "deleteNode failed.", m_szDetails, rc);
		goto Exit;
	}

	// All of our errors should now be corrected
	// NOTE - this works even though we have no data components
	// (they should not be required).

	if ( RC_BAD( rc = m_pDb->documentDone( pIndex)))
	{
		MAKE_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	endTest("PASS");

	beginTest(
		"Name/Dict Number Attribute Tests",
		"Tests to ensure name and dict number rules are being enforced",
		"1) Verify that if both the name and number attributes are "
		"specified that they reference the same element or attribute definition "
		"2) Verify that at least one of these is required to be specified for "
		"each component. ",
		"");

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	//NOTE: pNode still references "foo" attribute component

	if ( RC_BAD( rc = pNode->createAttribute(
		m_pDb,
		ATTR_DICT_NUMBER_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUINT( m_pDb, uiAttrDefDictNum + 17)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	// This should fail because name and dict number don't refer to the same thing

	rc = m_pDb->documentDone( pIndex);

	if ( rc != NE_XFLM_ATTRIBUTE_NAME_MISMATCH)
	{
		MAKE_ERROR_STRING( "Invalid rc from documentDone. Expected: "
			"NE_XFLM_ATTRIBUTE_NAME_MISMATCH", m_szDetails, rc);
		if ( RC_OK( rc))
		{
			rc = NE_XFLM_FAILURE;
		}
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transAbort()))
	{
		MAKE_ERROR_STRING( "transAbort failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	// Try again

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	//NOTE: pNode still references "foo" attribute component

	if ( RC_BAD( rc = pNode->createAttribute(
		m_pDb,
		ATTR_DICT_NUMBER_TAG,
		&pAttr)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUINT( m_pDb, uiAttrDefDictNum + 17)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pAttr->setUINT( m_pDb, uiAttrDefDictNum)))
	{
		MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	// We should be okay now

	if ( RC_BAD( rc = m_pDb->documentDone( pIndex)))
	{
		MAKE_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	endTest("PASS");

Exit:

	if ( RC_BAD( rc))
	{
		 outputAll("FAIL");
	}

	if ( pIndex)
	{
		pIndex->Release();
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
	if ( pTmpNode)
	{
		pTmpNode->Release();
	}

	if ( bTransBegun)
	{
		if ( RC_OK( rc))
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
