//------------------------------------------------------------------------------
// Desc:	Dictionary change tests
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

#define NAMESPACE 						"http://www.bogusnamespace.com"
#define MAX_POLL_COUNT					100

/****************************************************************************
Desc:
****************************************************************************/
class IDictChangeTestImpl : public TestBase
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

	if( (*ppTest = f_new IDictChangeTestImpl) == NULL)
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
const char * IDictChangeTestImpl::getName( void)
{
	return( "Dict Change Test");
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IDictChangeTestImpl::execute( void)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bDibCreated = FALSE;
	FLMUINT			uiElmId = 0;
	IF_DOMNode *	pNode = NULL;
	IF_DOMNode *	pDictDef = NULL;
	IF_DOMNode *	pDoc = NULL;
	IF_DOMNode *	pMyAttrNode = NULL;
	IF_DOMNode *	pMyEntryNode = NULL;
	char				szLocalName[100];
	FLMUINT			uiNumCharsReturned = 0;
	FLMBOOL			bTransStarted = FALSE;
	FLMUINT			uiMyEntryId = 0;
	FLMUINT			uiMyAttrId = 0x80000001;
	FLMUINT64		ui64EntryId;

	beginTest( 	
		"Dictionary Item Change Test",
		"Ensure we can change the name of an element definition then reuse the old name",
		"Create the database/create an element definition/"
		"create root element/rename element/set element to \"purge\"/"
		"reuse the old element name",
		"");

	if( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		MAKE_ERROR_STRING( "Failed to init test state.", m_szDetails, rc);
		goto Exit;
	}
	bDibCreated = TRUE;

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;

	if( RC_BAD( rc = m_pDb->createElementDef( NAMESPACE, "foo", XFLM_TEXT_TYPE,
		&uiElmId)))
	{
		MAKE_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->createRootElement( XFLM_DATA_COLLECTION, uiElmId, 
		&pNode)))
	{
		MAKE_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pNode->setUTF8( m_pDb, (FLMBYTE *)"value1")))
	{
		MAKE_ERROR_STRING( "setUTF8 failed", m_szDetails, rc);
		goto Exit;
	}

	// change the "foo" element's name then set its state to "purge"

	if( RC_BAD( rc = m_pDb->getDictionaryDef( ELM_ELEMENT_TAG, 
		uiElmId, &pDictDef)))
	{
		MAKE_ERROR_STRING( "getDictionaryDef changed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pDictDef->setAttributeValueUTF8( 
		m_pDb, ATTR_NAME_TAG, (FLMBYTE *)"deleted_foo")))
	{
		MAKE_ERROR_STRING( "setAttributeValueUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->documentDone( pDictDef)))
	{
		MAKE_ERROR_STRING( "documentDone failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->changeItemState( ELM_ELEMENT_TAG,
		uiElmId, XFLM_PURGE_OPTION_STR)))
	{
		MAKE_ERROR_STRING( "changeItemState failed", m_szDetails, rc);
		goto Exit;
	}

	uiElmId = 0;
	if ( RC_BAD( rc = m_pDb->createElementDef( NAMESPACE, "foo", 
		XFLM_NUMBER_TYPE, &uiElmId)))
	{
		MAKE_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->createRootElement( XFLM_DATA_COLLECTION, 
		uiElmId, &pNode)))
	{
		MAKE_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pNode->getLocalName( m_pDb, szLocalName,
		sizeof( szLocalName), &uiNumCharsReturned)))
	{
		MAKE_ERROR_STRING( "getLocalName failed.", m_szDetails, rc);
		goto Exit;
	}

	if( bTransStarted)
	{
		if( RC_BAD( rc))
		{
			m_pDb->transAbort();
		}
		else
		{
			m_pDb->transCommit();
		}
	}

  	endTest("PASS");

	beginTest( 	
		"Dictionary Attribute Purge Test",
		"Verify that setting a definition to \"purge\" works correctly",
		"Create the database/create an attribute definition/"
		"Create element/add an attribute/set attribute def to \"purge\"/"
		"Test for presence of attribute definition",
		"");

	if( RC_BAD( rc = m_pDb->transBegin(XFLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;

	if( RC_BAD( rc = m_pDb->createElementDef( NULL, "MyElement",
		XFLM_NODATA_TYPE, &uiMyEntryId)))
	{
		MAKE_FLM_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->createAttributeDef( NULL, "MyAttribute",
		XFLM_NUMBER_TYPE, &uiMyAttrId)))
	{
		MAKE_FLM_ERROR_STRING( "createAttributeDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->createDocument( XFLM_DATA_COLLECTION, &pDoc)))
	{
		MAKE_FLM_ERROR_STRING( "createDocument failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pDoc->createNode( m_pDb, ELEMENT_NODE, uiMyEntryId,
		XFLM_LAST_CHILD, &pMyEntryNode, &ui64EntryId)))
	{
		MAKE_FLM_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pMyEntryNode->createAttribute( m_pDb, uiMyAttrId,
		&pMyAttrNode)))
	{
		MAKE_FLM_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pMyAttrNode->setUINT( m_pDb, 0x18332)))
	{
		MAKE_FLM_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}
	
	if( RC_BAD( rc = pMyEntryNode->setUINT64( m_pDb, 34)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->documentDone( pDoc)))
	{
		MAKE_FLM_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}
	m_pDb->transCommit();
	bTransStarted = FALSE;

	// Start a transaction and set the state of the uiMyAttrId to "purge"

	if( RC_BAD( rc = m_pDb->transBegin(XFLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;

	if( RC_BAD( rc = m_pDb->changeItemState( ELM_ATTRIBUTE_TAG, 
		uiMyAttrId, XFLM_PURGE_OPTION_STR)))
	{
		MAKE_ERROR_STRING( "changeItemState failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( m_pDb->transCommit()))
	{
		goto Exit;
	}

	bTransStarted = FALSE;

	for( FLMINT iCount = 0; iCount < MAX_POLL_COUNT; iCount++)
	{
		if( RC_BAD( rc = m_pDb->getDictionaryDef( ELM_ATTRIBUTE_TAG, 
			uiMyAttrId, &pMyAttrNode)))
		{
			break;
		}
		f_sleep( 100);
	}
	
	if( rc != NE_XFLM_NOT_FOUND)
	{
		MAKE_ERROR_STRING( "purge attribute failed", m_szDetails, rc);
		goto Exit;
	}
	else
	{
		rc = NE_XFLM_OK;
	}
	
	endTest("PASS");
	m_pDb->doCheckpoint( 0);

Exit:

	if( bTransStarted)
	{
		if( RC_BAD( rc))
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

	if( pNode)
	{
		pNode->Release();
	}

  if( pMyEntryNode)
  {
	  pMyEntryNode->Release();
  }
  
  if( pMyAttrNode)
  {
	  pMyAttrNode->Release();
  }
  
  if( pDoc)
  {
	  pDoc->Release();
  }

  if ( pDictDef)
  {
	  pDictDef->Release();
  }

  shutdownTestState( DB_NAME_STR, bDibCreated);
  return rc;
}
