//------------------------------------------------------------------------------
// Desc:	Indexing unit test 3
// Tabs:	3
//
// Copyright (c) 2004-2007 Novell, Inc. All Rights Reserved.
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
	#define DB_NAME_STR					"SYS:\\IX3.DB"
#else
	#define DB_NAME_STR					"ix3.db"
#endif

/****************************************************************************
Desc:
****************************************************************************/
class IndexTest3Impl : public TestBase
{
public:

	IndexTest3Impl()
	{
		m_uiTextId = 0;
		m_uiNumId = 0;
		m_uiBinId = 0;
	}

	const char * getName( void);
	
	RCODE execute( void);
	
	RCODE runSuite1( void);
	
	RCODE runSuite2( void);

	RCODE runSuite3( void);
	
	RCODE runSuite4( void);
	
	RCODE runSuite5( void);
	
private:

	FLMUINT		m_uiTextId;
	FLMUINT		m_uiNumId;
	FLMUINT		m_uiBinId;

	RCODE validateKeys( 
		FLMUINT				uiIndex,
		KEY_COMP_INFO *	pSearchComps,
		FLMUINT				uiNumSearchComps,
		FLMUINT				uiSearchType, 
		FLMUINT64			ui64ExpectedDocId);

	RCODE createOrModifyIndex( 
		FLMUINT			uiIndex,
		FLMBOOL			bComp1SortDescending,
		FLMBOOL			bComp1CaseSensitive,
		FLMBOOL			bComp1SortMissingHigh,
		FLMBOOL			bComp2SortDescending,
		FLMBOOL			bComp2SortMissingHigh);

	 RCODE verifyKeyOrder( 
		 FLMUINT 		uiIndex, 
		 KEY_COMP_INFO	ppCompInfo[][2], 
		 FLMUINT 		uiNumCompInfo);

	RCODE runQueries( 
		FLMUINT 			uiIndexToUse);
};

/****************************************************************************
Desc:
****************************************************************************/
RCODE getTest( IFlmTest ** ppTest)
{
	RCODE		rc = NE_XFLM_OK;

	if ( ( *ppTest = f_new IndexTest3Impl) == NULL)
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
const char * IndexTest3Impl::getName()
{
	return "Index Test 3";
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IndexTest3Impl::execute()
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

	if( RC_BAD( rc = runSuite3()))
	{
		goto Exit;
	}

	if( RC_BAD( rc = runSuite4()))
	{
		goto Exit;
	}

	if( RC_BAD( rc = runSuite5()))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IndexTest3Impl::runSuite1( void)
{
	RCODE					rc = NE_XFLM_OK;
	FLMBOOL				bDibCreated = FALSE;
	IF_DOMNode *		pDoc = NULL;
	IF_DOMNode *		pEntryNode = NULL;
	IF_DOMNode *		pParentAttr = NULL;
	IF_DOMNode *		pFlagsAttr = NULL;
	IF_DataVector *	pDataVector = NULL;
	FLMUINT				uiEntryId = 0;
	FLMUINT				uiParentId = 0;
	FLMUINT				uiFlagsId = 0;
	FLMBOOL				bTransActive = FALSE;
	FLMUINT64			ui64EntryId = 0;
	FLMUINT				uiParent = 0;
	FLMUINT				uiFlags = 0;

	const char *		pszIndexDef =
		"<xflaim:Index "
		"	xmlns:xflaim=\"http://www.novell.com/XMLDatabase/Schema\""
		"	xflaim:name=\"parentID+flags\" "
		"	xflaim:DictNumber=\"1\"> " // HARD-CODED Dict Num for easy retrieval
		"	<xflaim:ElementComponent xflaim:name=\"entry\">"
		"		<xflaim:AttributeComponent"
		"			xflaim:name=\"parentID\" "
		"			xflaim:IndexOn=\"value\" "
		"			xflaim:KeyComponent=\"1\" />"
		"		<xflaim:AttributeComponent"
		"			xflaim:name=\"flags\" "
		"			xflaim:DataComponent=\"1\" />"
		"	</xflaim:ElementComponent> "
		"</xflaim:Index> ";

	beginTest( 	
		"Data Component Modify Test",
		"Create an index with a data component then modify the"
		" value that the data component references.",
		"Self-Explanatory",
		"");

	if( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to init test state.", m_szDetails, rc);
		goto Exit;
	}
	bDibCreated = TRUE;

	if( RC_BAD( rc = m_pDb->transBegin(XFLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransActive = TRUE;

	if( RC_BAD( rc = m_pDb->createElementDef( NULL, "entry", XFLM_NODATA_TYPE,
		&uiEntryId)))
	{
		MAKE_FLM_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->createAttributeDef( NULL, "parentID",
		XFLM_NUMBER_TYPE, &uiParentId)))
	{
		MAKE_FLM_ERROR_STRING( "createAttributeDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->createAttributeDef( NULL, "flags", 
		XFLM_NUMBER_TYPE, &uiFlagsId)))
	{
		MAKE_FLM_ERROR_STRING( "createAttributeDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->createDocument( XFLM_DATA_COLLECTION, &pDoc)))
	{
		MAKE_FLM_ERROR_STRING( "createDocument failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pDoc->createNode( m_pDb, ELEMENT_NODE, uiEntryId,
		XFLM_LAST_CHILD, &pEntryNode, &ui64EntryId)))
	{
		MAKE_FLM_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pEntryNode->createAttribute( 
		m_pDb, uiParentId, &pParentAttr)))
	{
		MAKE_FLM_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pParentAttr->setUINT( m_pDb, 1)))
	{
		MAKE_FLM_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pEntryNode->createAttribute(
		m_pDb, uiFlagsId, &pFlagsAttr)))
	{
		MAKE_FLM_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pFlagsAttr->setUINT( m_pDb, 1)))
	{
		MAKE_FLM_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->documentDone( pDoc)))
	{
		MAKE_FLM_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = importBuffer( pszIndexDef, XFLM_DICT_COLLECTION)))
	{
		MAKE_FLM_ERROR_STRING( "importBuffer failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_FLM_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransActive = FALSE;

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDbSystem->createIFDataVector( &pDataVector)))
	{
		MAKE_FLM_ERROR_STRING( "createIFDataVector failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->keyRetrieve( 1, NULL, XFLM_FIRST, pDataVector)))
	{
		MAKE_FLM_ERROR_STRING( "keyRetrieve failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pDataVector->getUINT( 0, &uiParent)))
	{
		MAKE_FLM_ERROR_STRING( "getUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pDataVector->getUINT( 1, &uiFlags)))
	{
		MAKE_FLM_ERROR_STRING( "getUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	if( uiParent != 1 && uiFlags != 1)
	{
		rc = NE_XFLM_DATA_ERROR;
		MAKE_FLM_ERROR_STRING( "Incorrect key values detected.", 
			m_szDetails, rc);
		goto Exit;
	}

	// change the data component value

	if( RC_BAD( rc = pFlagsAttr->setUINT( m_pDb, 0)))
	{
		MAKE_FLM_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->documentDone( pDoc)))
	{
		MAKE_FLM_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_FLM_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransActive = FALSE;

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransActive = TRUE;

	if( RC_BAD( rc = m_pDb->keyRetrieve( 1, NULL, XFLM_FIRST, pDataVector)))
	{
		MAKE_FLM_ERROR_STRING( "keyRetrieve failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pDataVector->getUINT( 1, &uiFlags)))
	{
		MAKE_FLM_ERROR_STRING( "getUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	if( uiFlags != 0)
	{
		rc = NE_XFLM_DATA_ERROR;
		MAKE_FLM_ERROR_STRING( "Incorrect key values detected.", 
			m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pFlagsAttr->setUINT( m_pDb, 7)))
	{
		MAKE_FLM_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->documentDone( pDoc)))
	{
		MAKE_FLM_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_FLM_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransActive = FALSE;

	if( RC_BAD( rc = m_pDb->keyRetrieve( 1, NULL, XFLM_LAST, pDataVector)))
	{
		MAKE_FLM_ERROR_STRING( "keyRetrieve failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pDataVector->getUINT( 1, &uiFlags)))
	{
		MAKE_FLM_ERROR_STRING( "getUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	if( uiFlags != 7)
	{
		rc = NE_XFLM_DATA_ERROR;
		MAKE_FLM_ERROR_STRING( "Incorrect key values detected.", 
			m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

Exit:

	if( RC_BAD( rc ))
	{
		endTest("FAIL");
	}

	if( bTransActive)
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

	if( pParentAttr)
	{
		pParentAttr->Release();
	}

	if( pFlagsAttr)
	{
		pFlagsAttr->Release();
	}

	if( pDataVector)
	{
		pDataVector->Release();
	}

	if( pDoc)
	{
		pDoc->Release();
	}

	if( pEntryNode)
	{
		pEntryNode->Release();
	}

	shutdownTestState( DB_NAME_STR, bDibCreated);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IndexTest3Impl::runSuite2( void)
{
	RCODE					rc = NE_XFLM_OK;
	FLMBOOL				bDibCreated = FALSE;
	IF_DOMNode *		pIndexRoot = NULL;
	IF_DOMNode *		pComponentNode = NULL;
	IF_DOMNode *		pAttr = NULL;
	FLMBOOL				bTransBegun = FALSE;

	beginTest( 	
		"No Required Data Component Test",
		"Make sure it is illegal to make a data component required",
		"Self-explanatory",
		"");

	if( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to init test state.", m_szDetails, rc);
		goto Exit;
	}
	bDibCreated = TRUE;

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	if( RC_BAD( rc = m_pDb->createRootElement( 
		XFLM_DICT_COLLECTION, ELM_INDEX_TAG, &pIndexRoot)))
	{
		MAKE_FLM_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pIndexRoot->createAttribute( m_pDb, ATTR_NAME_TAG, &pAttr)))
	{
		MAKE_FLM_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)"ix")))
	{
		MAKE_FLM_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pIndexRoot->createNode( m_pDb, ELEMENT_NODE,
		ELM_ELEMENT_COMPONENT_TAG, XFLM_LAST_CHILD, &pComponentNode, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pComponentNode->createAttribute(
		m_pDb, ATTR_DICT_NUMBER_TAG, &pAttr)))
	{
		MAKE_FLM_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pAttr->setUINT( m_pDb, ELM_ELEMENT_COMPONENT_TAG)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pComponentNode->createAttribute( 
		m_pDb, ATTR_REQUIRED_TAG, &pAttr)))
	{
		MAKE_FLM_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pAttr->setUINT( m_pDb, 1)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pComponentNode->createAttribute( 
		m_pDb, ATTR_KEY_COMPONENT_TAG, &pAttr)))
	{
		MAKE_FLM_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pAttr->setUINT( m_pDb, 1)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pComponentNode->createNode( m_pDb, ELEMENT_NODE,
		ELM_ATTRIBUTE_COMPONENT_TAG, XFLM_LAST_CHILD, &pComponentNode,
		NULL)))
	{
		MAKE_FLM_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pComponentNode->createAttribute( 
		m_pDb, ATTR_DICT_NUMBER_TAG, &pAttr)))
	{
		MAKE_FLM_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pAttr->setUINT( m_pDb, ATTR_FIXED_TAG)))
	{
		MAKE_FLM_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pComponentNode->createAttribute( 
		m_pDb, ATTR_DATA_COMPONENT_TAG, &pAttr)))
	{
		MAKE_FLM_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pAttr->setUINT( m_pDb, 1)))
	{
		MAKE_FLM_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pComponentNode->createAttribute( 
		m_pDb, ATTR_REQUIRED_TAG, &pAttr)))
	{
		MAKE_FLM_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pAttr->setUINT( m_pDb, 1)))
	{
		MAKE_FLM_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
		goto Exit;
	}

	rc = m_pDb->documentDone( pIndexRoot);

	if( rc != NE_XFLM_CANNOT_SET_REQUIRED)
	{
		MAKE_FLM_ERROR_STRING( "Unexpected rc returned from documentDone", 
			m_szDetails, rc);
		if ( RC_OK(rc))
		{
			rc = RC_SET( NE_XFLM_FAILURE);
		}
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transAbort()))
	{
		MAKE_FLM_ERROR_STRING( "transAbort failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

   endTest("PASS");

Exit:

	if( RC_BAD( rc))
	{
		endTest( "FAIL");
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

	if( pIndexRoot)
	{
		pIndexRoot->Release();
	}

	if( pComponentNode)
	{
		pComponentNode->Release();
	}

	if( pAttr)
	{
		pAttr->Release();
	}

	shutdownTestState( DB_NAME_STR, bDibCreated);
	return( rc);
}

#define TEXT_INDEX_ID 123
#define BIN_INDEX_ID 456

/****************************************************************************
Desc:
****************************************************************************/
RCODE IndexTest3Impl::validateKeys(
	FLMUINT				uiIndex,
	KEY_COMP_INFO *	pSearchComps,
	FLMUINT				uiNumSearchComps,
	FLMUINT				uiSearchType,
	FLMUINT64			ui64ExpectedDocId)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DataVector *	pSearchKey = NULL;
	IF_DataVector *	pFoundKey = NULL;
	FLMUINT				uiLoop;

	if( RC_BAD( rc = m_pDbSystem->createIFDataVector( &pSearchKey)))
	{
		MAKE_FLM_ERROR_STRING( "createIFDataVector failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDbSystem->createIFDataVector( &pFoundKey)))
	{
		MAKE_FLM_ERROR_STRING( "createIFDataVector failed.", m_szDetails, rc);
		goto Exit;
	}

	for( uiLoop = 0; uiLoop < uiNumSearchComps; uiLoop++)
	{
		switch( pSearchComps[uiLoop].uiDataType)
		{
			case XFLM_TEXT_TYPE:
				if ( RC_BAD( rc = pSearchKey->setUTF8( 
					uiLoop, (FLMBYTE *)pSearchComps[uiLoop].pvComp)))
				{
					goto Exit;
				}
				break;
			case XFLM_BINARY_TYPE:
				if ( RC_BAD( rc = pSearchKey->setBinary(
					uiLoop, (FLMBYTE*)pSearchComps[uiLoop].pvComp,
					pSearchComps[uiLoop].uiDataSize)))
				{
					goto Exit;
				}
				break;
			case XFLM_NUMBER_TYPE:
				if ( RC_BAD( rc = pSearchKey->setUINT(
					uiLoop, (FLMUINT)pSearchComps[uiLoop].pvComp)))
				{
					goto Exit;
				}
				break;
			default:
				flmAssert(0);
		}
	}

	if( RC_BAD( rc = m_pDb->keyRetrieve( 
		uiIndex, pSearchKey, uiSearchType, pFoundKey)))
	{
		if( (rc == NE_XFLM_NOT_FOUND  || rc == NE_XFLM_EOF_HIT) && 
			ui64ExpectedDocId == 0)
		{
			rc = NE_XFLM_OK;
		}
		else
		{
			MAKE_FLM_ERROR_STRING( "keyRetrieve failed.", m_szDetails, rc);
		}
		goto Exit;
	}
	else if( ui64ExpectedDocId == 0)
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		MAKE_FLM_ERROR_STRING( "Unexpected index key found.", m_szDetails, rc);
		goto Exit;
	}

	if( pFoundKey->getDocumentID() != ui64ExpectedDocId)
	{
		rc = RC_SET( NE_XFLM_NOT_FOUND);
		MAKE_FLM_ERROR_STRING( "Unexpected key found.", m_szDetails, rc);
		goto Exit;
	}

Exit:

	if( pSearchKey)
	{
		pSearchKey->Release();
	}

	if( pFoundKey)
	{
		pFoundKey->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IndexTest3Impl::runSuite3( void)
{
	RCODE					rc = NE_XFLM_OK;
	FLMBOOL				bDibCreated = FALSE;
	FLMBOOL				bTransBegun = FALSE;
	IF_DOMNode *		pNode = NULL;
	FLMUINT64			ui64ABCxyz2DocId = 0;
	FLMUINT64			ui64ABCxyz3DocId = 0;
	FLMUINT64			ui64ABCefg5DocId = 0;
	FLMUINT64			ui64ABCdef5DocId = 0;
	FLMUINT64			ui64CB10124DocId = 0;
	FLMUINT64			ui64CB11233DocId = 0;
	FLMUINT64			ui64CB12343DocId = 0;
	FLMUINT64			ui64CB13451DocId = 0;
	
	const char *		pszIndexFormat = 		
		"<xflaim:Index "
			"xmlns:xflaim=\"http://www.novell.com/XMLDatabase/Schema\" "
			"xflaim:name=\"%s_IX\" "
			"xflaim:DictNumber=\"%u\">"
			"<xflaim:ElementComponent "
				"xflaim:targetNameSpace=\"\" "
				"xflaim:DictNumber=\"%u\" "
				"xflaim:IndexOn=\"value\" "
				"xflaim:KeyComponent=\"1\" "
				"xflaim:Limit=\"3\" "
				"xflaim:Required=\"1\"/>"
			"<xflaim:ElementComponent "
				"xflaim:targetNameSpace=\"\" "
				"xflaim:DictNumber=\"%u\" "
				"xflaim:IndexOn=\"value\" "
				"xflaim:KeyComponent=\"2\"/>"
		"</xflaim:Index>";

	char * 			pszIndex = NULL;

	beginTest( 	
		"Truncated Key Test Setup",
		"Setup DIB for the truncated key tests",
		"Self-explanatory",
		"");

	if( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to init test state.", m_szDetails, rc);
		goto Fatal_Error;
	}
	bDibCreated = TRUE;

	if( RC_BAD( rc = m_pDb->createElementDef(
		NULL, "text_val", XFLM_TEXT_TYPE, &m_uiTextId, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Fatal_Error;
	}

	if( RC_BAD( rc = m_pDb->createElementDef(
		NULL, "num_val", XFLM_NUMBER_TYPE, &m_uiNumId, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Fatal_Error;
	}

	if( RC_BAD( rc = m_pDb->createElementDef(
		NULL, "bin_val", XFLM_BINARY_TYPE, &m_uiBinId, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Fatal_Error;
	}

	if( RC_BAD( rc = f_alloc( f_strlen( pszIndexFormat) + 16, &pszIndex)))
	{
		MAKE_FLM_ERROR_STRING( "f_alloc failed.", m_szDetails, rc);
		goto Fatal_Error;
	}

	f_sprintf( pszIndex, pszIndexFormat, "text+num", 
		TEXT_INDEX_ID, m_uiTextId, m_uiNumId);

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Fatal_Error;
	}
	bTransBegun = TRUE;

	// Create the index

	if( RC_BAD( rc = importBuffer( pszIndex, XFLM_DICT_COLLECTION)))
	{
		MAKE_FLM_ERROR_STRING( "importBuffer failed.", m_szDetails, rc);
		goto Fatal_Error;
	}

	f_sprintf( pszIndex, pszIndexFormat, "bin+num", 
		BIN_INDEX_ID, m_uiBinId, m_uiNumId);

	if( RC_BAD( rc = importBuffer( pszIndex, XFLM_DICT_COLLECTION)))
	{
		MAKE_FLM_ERROR_STRING( "importBuffer failed.", m_szDetails, rc);
		goto Fatal_Error;
	}

	// Create documents. I'm doing this in an order reverse to that in which
	// their keys should be sorted.

	{
		ELEMENT_NODE_INFO	pElementNodes[] =
			{
				{(void*)"ABCxyz", XFLM_TEXT_TYPE, 6, m_uiTextId},
				{(void*)3, XFLM_NUMBER_TYPE, sizeof(FLMUINT), m_uiNumId}
			};

		if( RC_BAD( rc = createCompoundDoc( 
			pElementNodes, 2, &ui64ABCxyz3DocId)))
		{
			goto Fatal_Error;
		}
	}

	{
		ELEMENT_NODE_INFO	pElementNodes[] =
			{
				{(void*)"ABCxyz", XFLM_TEXT_TYPE, 6, m_uiTextId},
				{(void*)2, XFLM_NUMBER_TYPE, sizeof(FLMUINT), m_uiNumId}
			};

		if( RC_BAD( rc = createCompoundDoc( 
			pElementNodes, 2,&ui64ABCxyz2DocId)))
		{
			goto Fatal_Error;
		}
	}

	{
		ELEMENT_NODE_INFO	pElementNodes[] =
			{
				{(void*)"ABCefg", XFLM_TEXT_TYPE, 6, m_uiTextId},
				{(void*)5, XFLM_NUMBER_TYPE, sizeof(FLMUINT), m_uiNumId}
			};

		if( RC_BAD( rc = createCompoundDoc( 
			pElementNodes, 2, &ui64ABCefg5DocId)))
		{
			goto Fatal_Error;
		}
	}

	{
		ELEMENT_NODE_INFO	pElementNodes[] =
			{
				{(void*)"ABCdef", XFLM_TEXT_TYPE, 6, m_uiTextId},
				{(void*)5, XFLM_NUMBER_TYPE, sizeof(FLMUINT), m_uiNumId}
			};

		if( RC_BAD( rc = createCompoundDoc( 
			pElementNodes, 2, &ui64ABCdef5DocId)))
		{
			goto Fatal_Error;
		}
	}

	// Documents with binary fields

	{
		ELEMENT_NODE_INFO	pElementNodes[] =
			{
				{(void*)"\xC\xB\x1\x3\x4\x5", XFLM_BINARY_TYPE, 6, m_uiBinId},
				{(void*)1, XFLM_NUMBER_TYPE, sizeof(FLMUINT), m_uiNumId}
			};

		if( RC_BAD( rc = createCompoundDoc( 
			pElementNodes, 2, &ui64CB13451DocId)))
		{
			goto Fatal_Error;
		}
	}

	{
		ELEMENT_NODE_INFO	pElementNodes[] =
			{
				{(void*)"\xC\xB\x1\x2\x3\x4", XFLM_BINARY_TYPE, 6, m_uiBinId},
				{(void*)3, XFLM_NUMBER_TYPE, sizeof(FLMUINT), m_uiNumId}
			};

		if( RC_BAD( rc = createCompoundDoc( 
			pElementNodes, 2, &ui64CB12343DocId)))
		{
			goto Fatal_Error;
		}
	}

	{
		ELEMENT_NODE_INFO	pElementNodes[] =
			{
				{(void*)"\xC\xB\x1\x1\x2\x3", XFLM_BINARY_TYPE, 6, m_uiBinId},
				{(void*)3, XFLM_NUMBER_TYPE, sizeof(FLMUINT), m_uiNumId}
			};

		if( RC_BAD( rc = createCompoundDoc( 
			pElementNodes, 2, &ui64CB11233DocId)))
		{
			goto Fatal_Error;
		}
	}

	{
		ELEMENT_NODE_INFO	pElementNodes[] =
			{
				{(void*)"\xC\xB\x1\x0\x1\x2", XFLM_BINARY_TYPE, 6, m_uiBinId},
				{(void*)4, XFLM_NUMBER_TYPE, sizeof(FLMUINT), m_uiNumId}
			};

		if( RC_BAD( rc = createCompoundDoc( 
			pElementNodes, 2, &ui64CB10124DocId)))
		{
			goto Fatal_Error;
		}
	}

	endTest( "PASS");

	// Keys for the first index should be as follows:
	// ABC|def + 5
	// ABC|efg + 5
	// ABC|xyz + 2
	// ABC|xyz + 3

	beginTest( 	
		"Truncated Key Test #1",
		"Verify key retrieve results for truncated keys",
		"Self-explanatory",
		"");

	// inclusive key retrieve for ABCefg+5. Expected returns:
	//		ABCefg+5
	{
		KEY_COMP_INFO searchKey[] = {	
												{(void*)"ABCefg", XFLM_TEXT_TYPE, 6},
												{(void*)5, XFLM_NUMBER_TYPE, sizeof(FLMUINT)}
											 }; 

		if( RC_BAD( rc = validateKeys(
			TEXT_INDEX_ID,	searchKey, 2, XFLM_INCL, ui64ABCefg5DocId)))
		{
			endTest("FAIL");
		}
		else
		{
			endTest("PASS");
		}
	}

	beginTest( 	
		"Truncated Key Test #2",
		"Verify key retrieve results for truncated keys",
		"Self-explanatory",
		"");

	// exclusive key retrieve for ABCdef+5. Expected returns:
	//		ABCefg+5

	{
		KEY_COMP_INFO searchKey[] = 
			{
				{(void*)"ABCdef", XFLM_TEXT_TYPE, 6},
				{(void*)5, XFLM_NUMBER_TYPE, sizeof(FLMUINT)}
			};

		if( RC_BAD( rc = validateKeys( 
			TEXT_INDEX_ID,	searchKey, 2, XFLM_EXCL, ui64ABCefg5DocId)))
		{
			endTest("FAIL");
		}
		else
		{
			endTest("PASS");
		}
	}

	beginTest( 	
		"Truncated Key Test #3",
		"Verify key retrieve results for truncated keys",
		"Self-explanatory",
		"");

	// inclusive key retrieve for ABCxyz+2. Expected returns:
	//		ABCxyz+2

	{
		KEY_COMP_INFO searchKey[] = 
			{
				{(void*)"ABCxyz", XFLM_TEXT_TYPE, 6},
				{(void*)2, XFLM_NUMBER_TYPE, sizeof(FLMUINT)}
			};

		if( RC_BAD( rc = validateKeys(
			TEXT_INDEX_ID, searchKey, 2, XFLM_INCL, 
			ui64ABCxyz2DocId)))
		{
			endTest("FAIL");
		}
		else
		{
			endTest("PASS");
		}
	}

	beginTest( 	
		"Truncated Key Test #4",
		"Verify key retrieve results for truncated keys",
		"Self-explanatory",
		"");

	// exclusive key retrieve for ABCxyz+2. Expected returns:
	//		ABCxyz+3

	{
		KEY_COMP_INFO searchKey[] = 
			{
				{(void*)"ABCxyz", XFLM_TEXT_TYPE, 6},
				{(void*)2, XFLM_NUMBER_TYPE, sizeof(FLMUINT)}
			};

		if( RC_BAD( rc = validateKeys( 
			TEXT_INDEX_ID,	searchKey, 2, XFLM_EXCL, 
			ui64ABCxyz3DocId)))
		{
			endTest("FAIL");
		}
		else
		{
			endTest("PASS");
		}
	}

	beginTest( 	
		"Truncated Key Test #5",
		"Verify key retrieve results for truncated keys",
		"Self-explanatory",
		"");

	// inclusive key retrieve for ABCyyz+2. Expected returns:
	//		none

	{
		KEY_COMP_INFO searchKey[] = 
			{
				{(void*)"ABCyyz", XFLM_TEXT_TYPE, 6},
				{(void*)2, XFLM_NUMBER_TYPE, sizeof(FLMUINT)}
			};

		if( RC_BAD( rc = validateKeys( 
			TEXT_INDEX_ID, searchKey, 2, XFLM_INCL, 0)))
		{
			endTest("FAIL");
		}
		else
		{
			endTest("PASS");
		}
	}

	beginTest( 	
		"Truncated Key Test #6",
		"Verify key retrieve results for truncated keys",
		"Self-explanatory",
		"");

	// exclusive key retrieve for ABCghi+2. Expected returns:
	//		ABCxyz+2

	{
		KEY_COMP_INFO searchKey[] = 
			{
				{(void*)"ABCghi", XFLM_TEXT_TYPE, 6},
				{(void*)2, XFLM_NUMBER_TYPE, sizeof(FLMUINT)}
			};

		if( RC_BAD( rc = validateKeys( 
			TEXT_INDEX_ID,	searchKey, 2, XFLM_EXCL, ui64ABCxyz2DocId)))
		{
			endTest("FAIL");
		}
		else
		{
			endTest("PASS");
		}
	}

	beginTest( 	
		"Truncated Key Test #7",
		"Verify key retrieve results for truncated keys",
		"Self-explanatory",
		"");

	// inclusive key retrieve for ABCdef+6. Expected returns:
	//		ABCefg+5

	{
		KEY_COMP_INFO searchKey[] = 
			{
				{(void*)"ABCdef", XFLM_TEXT_TYPE, 6},
				{(void*)6, XFLM_NUMBER_TYPE, sizeof(FLMUINT)}
			};

		if( RC_BAD( rc = validateKeys( 
			TEXT_INDEX_ID,	searchKey, 2, XFLM_INCL, ui64ABCefg5DocId)))
		{
			endTest("FAIL");
		}
		else
		{
			endTest("PASS");
		}
	}

	beginTest( 	
		"Truncated Key Test #8",
		"Verify key retrieve results for truncated keys",
		"Self-explanatory",
		"");

	// exclusive key retrieve for ABCdef+5. Expected returns:
	//		ABCefg+5

	{
		KEY_COMP_INFO searchKey[] = 
			{
				{(void*)"ABCdef", XFLM_TEXT_TYPE, 6},
				{(void*)5, XFLM_NUMBER_TYPE, sizeof(FLMUINT)}
			};

		if( RC_BAD( rc = validateKeys( 
			TEXT_INDEX_ID,	searchKey, 2, XFLM_EXCL, ui64ABCefg5DocId)))
		{
			endTest("FAIL");
		}
		else
		{
			endTest("PASS");
		}
	}

	beginTest( 	
		"Truncated Key Test #9",
		"Verify key retrieve results for truncated keys",
		"Self-explanatory",
		"");

	// exclusive key retrieve for ABCxyz+3. Expected returns:
	//		none

	{
		KEY_COMP_INFO searchKey[] = 
			{
				{(void*)"ABCxyz", XFLM_TEXT_TYPE, 6},
				{(void*)3, XFLM_NUMBER_TYPE, sizeof(FLMUINT)}
			};

		if( RC_BAD( rc = validateKeys( 
			TEXT_INDEX_ID,	searchKey, 2, XFLM_EXCL, 0)))
		{
			endTest("FAIL");
		}
		else
		{
			endTest("PASS");
		}
	}

	// Keys for the second index should be as follows:
	// 0xC,0xB,0x1|0x0,0x1,0x2 + 4
	// 0xC,0xB,0x1|0x1,0x2,0x3 + 3
	// 0xC,0xB,0x1|0x2,0x3,0x4 + 3
	// 0xC,0xB,0x1|0x3,0x4,0x5 + 1

	//  inclusive key retrieve for xCxBx1x2x3x4+3. Expected return
	//		xCxBx1x2x3x4+3

	beginTest( 	
		"Truncated Key Test #10",
		"Verify key retrieve results for truncated keys",
		"Self-explanatory",
		"");

	{
		KEY_COMP_INFO searchKey[] = 
			{
				{(void*)"\xC\xB\x1\x2\x3\x4", XFLM_BINARY_TYPE, 6},
				{(void*)3, XFLM_NUMBER_TYPE, sizeof(FLMUINT)}
			};

		if( RC_BAD( rc = validateKeys( 
			BIN_INDEX_ID,	searchKey, 2, XFLM_INCL, ui64CB12343DocId)))
		{
			endTest("FAIL");
		}
		else
		{
			endTest("PASS");
		}
	}

	// exclusive key retrieve for xCxBx1x0x1x2+4. Expected return
	//		xCxBx1x1x2x3+3

	beginTest( 	
		"Truncated Key Test #11",
		"Verify key retrieve results for truncated keys",
		"Self-explanatory",
		"");
	
	{
		KEY_COMP_INFO searchKey[] = 
			{
				{(void*)"\xC\xB\x1\x0\x1\x2", XFLM_BINARY_TYPE, 6},
				{(void*)4, XFLM_NUMBER_TYPE, sizeof(FLMUINT)}
			};

		if( RC_BAD( rc = validateKeys( 
			BIN_INDEX_ID,	searchKey, 2, XFLM_EXCL, ui64CB11233DocId)))
		{
			endTest("FAIL");
		}
		else
		{
			endTest("PASS");
		}
	}

	//  inclusive key retrieve for xCxBx1x3x4x5+1. Expected return
	//		xCxBx1x3x4x5+1

	beginTest( 	
		"Truncated Key Test #12",
		"Verify key retrieve results for truncated keys",
		"Self-explanatory",
		"");

	{
		KEY_COMP_INFO searchKey[] = 
			{
				{(void*)"\xC\xB\x1\x3\x4\x5", XFLM_BINARY_TYPE, 6},
				{(void*)1, XFLM_NUMBER_TYPE, sizeof(FLMUINT)}
			};

		if( RC_BAD( rc = validateKeys( 
			BIN_INDEX_ID,	searchKey, 2, XFLM_INCL, ui64CB13451DocId)))
		{
			endTest("FAIL");
		}
		else
		{
			endTest("PASS");
		}
	}

	//  inclusive key retrieve for xCxBx1x0x1x2+1. Expected return
	//		xCxBx1x0x1x2+4

	beginTest( 	
		"Truncated Key Test #13",
		"Verify key retrieve results for truncated keys",
		"Self-explanatory",
		"");

	{
		KEY_COMP_INFO searchKey[] = 
			{
				{(void*)"\xC\xB\x1\x0\x1\x2", XFLM_BINARY_TYPE, 6},
				{(void*)1, XFLM_NUMBER_TYPE, sizeof(FLMUINT)}
			};

		if( RC_BAD( rc = validateKeys( 
			BIN_INDEX_ID,	searchKey, 2, XFLM_INCL, ui64CB10124DocId)))
		{
			endTest("FAIL");
		}
		else
		{
			endTest("PASS");
		}
	}

	beginTest( 	
		"Truncated Key Modify Test",
		"Modify a value that is indexed by a truncated key and make sure "
		"we do not assert when trying to delete the old key value",
		"Self-explanatory",
		"");

	if( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_FLM_ERROR_STRING( "transCommit failed", m_szDetails, rc);
		goto Fatal_Error;
	}
	bTransBegun = FALSE;

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed", m_szDetails, rc);
		goto Fatal_Error;
	}
	bTransBegun = TRUE;

	if( RC_BAD( rc = m_pDb->getNode( 
		XFLM_DATA_COLLECTION, ui64ABCefg5DocId, &pNode)))
	{
		MAKE_FLM_ERROR_STRING( "getNode failed", m_szDetails, rc);
		goto Fatal_Error;
	}

	if( RC_BAD( rc = pNode->getFirstChild( m_pDb, &pNode)))
	{
		MAKE_FLM_ERROR_STRING( "getFirstChild failed", m_szDetails, rc);
		goto Fatal_Error;
	}

	if( RC_BAD( rc = pNode->setUTF8( m_pDb, (FLMBYTE *)"New Value")))
	{
		MAKE_FLM_ERROR_STRING( "setUTF8 failed", m_szDetails, rc);
		goto Fatal_Error;
	}

	if( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_FLM_ERROR_STRING( "transCommit failed", m_szDetails, rc);
		goto Fatal_Error;
	}
	bTransBegun = FALSE;

	endTest("PASS");
	goto Exit;

Fatal_Error:

	endTest("FAIL");

Exit:

	if( pNode)
	{
		pNode->Release();
	}

	if( pszIndex)
	{
		f_free( &pszIndex);
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
RCODE IndexTest3Impl::createOrModifyIndex( 
	FLMUINT	uiIndex,
	FLMBOOL	bComp1SortDescending,
	FLMBOOL	bComp1CaseSensitive,
	FLMBOOL	bComp1SortMissingHigh,
	FLMBOOL	bComp2SortDescending,
	FLMBOOL	bComp2SortMissingHigh)
{
	RCODE				rc = NE_XFLM_OK;
	char				szCompareRules1[100];
	char				szCompareRules2[100];
	char				szName[100];
	char *			pszIndex = NULL;
	IF_DOMNode *	pNode = NULL;
	const char *	pszIndexFormat = 		
		"<xflaim:Index "
			"xmlns:xflaim=\"http://www.novell.com/XMLDatabase/Schema\" "
			"xflaim:name=\"%s_IX\" "
			"xflaim:DictNumber=\"%u\">"
			"<xflaim:ElementComponent "
				"xflaim:CompareRules=\"%s\" "
				"xflaim:targetNameSpace=\"\" "
				"xflaim:DictNumber=\"%u\" "
				"xflaim:IndexOn=\"value\" "
				"xflaim:KeyComponent=\"1\" />"
			"<xflaim:ElementComponent "
				"xflaim:CompareRules=\"%s\" "
				"xflaim:targetNameSpace=\"\" "
				"xflaim:DictNumber=\"%u\" "
				"xflaim:IndexOn=\"value\" "
				"xflaim:KeyComponent=\"2\"/>"
		"</xflaim:Index>";

	szCompareRules1[0] = '\0';
	szCompareRules2[0] = '\0';

	if( bComp1SortDescending)
	{
		f_strcpy( szCompareRules1, XFLM_DESCENDING_OPTION_STR);
	}

	if( !bComp1CaseSensitive)
	{
		f_strcat( szCompareRules1, " "XFLM_CASE_INSENSITIVE_OPTION_STR);
	}

	if( bComp1SortMissingHigh)
	{
		f_strcat( szCompareRules1, " "XFLM_MISSING_HIGH_OPTION_STR);
	}

	if( bComp2SortDescending)
	{
		f_strcpy( szCompareRules2, XFLM_DESCENDING_OPTION_STR);
	}

	if( bComp2SortMissingHigh)
	{
		f_strcat( szCompareRules2, " "XFLM_MISSING_HIGH_OPTION_STR);
	}

	f_sprintf( szName, "(%s/%s)", szCompareRules1, szCompareRules2); 

	if( RC_BAD( rc = f_alloc( f_strlen( pszIndexFormat) + 
		(f_strlen( szCompareRules1) + f_strlen(szCompareRules2)) * 2 +
		16, &pszIndex)))
	{
		MAKE_FLM_ERROR_STRING( "f_alloc failed", m_szDetails, rc);
		goto Exit;
	}

	f_sprintf( pszIndex, pszIndexFormat, szName, uiIndex, 
		szCompareRules1, m_uiTextId, szCompareRules2, m_uiNumId); 

	// remove the index if it already exists

	if( RC_OK( rc = m_pDb->getDictionaryDef( ELM_INDEX_TAG, uiIndex, &pNode)))
	{
		if( RC_BAD( rc = pNode->deleteNode( m_pDb)))
		{
			MAKE_FLM_ERROR_STRING( "deleteNode failed.", m_szDetails, rc);
			goto Exit;
		}
	}

	if( RC_BAD( rc = importBuffer( pszIndex, XFLM_DICT_COLLECTION)))
	{
		goto Exit;
	}

Exit:

	if( pNode)
	{
		pNode->Release();
	}

	if( pszIndex)
	{
		f_free( &pszIndex);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IndexTest3Impl::verifyKeyOrder( 
	FLMUINT				uiIndex, 
	KEY_COMP_INFO 		ppCompInfo[][2], 
	FLMUINT				uiNumCompInfo)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiLoop1;
	FLMUINT				uiLoop2;
	IF_DataVector *	pKey = NULL;
	char					szTmpBuf[100];
	FLMUINT				uiTmp;

	if( RC_BAD( rc = m_pDbSystem->createIFDataVector( &pKey)))
	{
		MAKE_FLM_ERROR_STRING( "createIFDataVector failed.", m_szDetails, rc);
		goto Exit;
	}

	for( uiLoop1 = 0; uiLoop1 < uiNumCompInfo; uiLoop1++)
	{
		if( uiLoop1 == 0)
		{
			if( RC_BAD( rc = m_pDb->keyRetrieve( 
				uiIndex, NULL, XFLM_FIRST, pKey)))
			{
				MAKE_FLM_ERROR_STRING( "keyRetrieve failed.", m_szDetails, rc);
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = m_pDb->keyRetrieve(
				uiIndex, pKey, XFLM_EXCL, pKey)))
			{
				MAKE_FLM_ERROR_STRING( "keyRetrieve failed.", m_szDetails, rc);
				goto Exit;
			}
		}

		// VISIT - if we want to reuse this code, we'll want to 
		// null-terminate the array. Right now we are simply assuming
		// two components per key

		for( uiLoop2 = 0; uiLoop2 < 2; uiLoop2++)
		{
			switch( ppCompInfo[uiLoop1][uiLoop2].uiDataType)
			{
			case XFLM_TEXT_TYPE:
				uiTmp = sizeof( szTmpBuf);
				if ( RC_BAD( rc = pKey->getUTF8( uiLoop2, 
					(FLMBYTE *)szTmpBuf, &uiTmp)))
				{
					if ( ppCompInfo[uiLoop1][uiLoop2].uiDataSize == 0 &&
						rc == NE_XFLM_NOT_FOUND)
					{
						rc = NE_XFLM_OK;
						break;
					}
					else
					{
						MAKE_FLM_ERROR_STRING( "getNative failed.", m_szDetails, rc);
						goto Exit;
					}
				}

				if ( uiTmp != ppCompInfo[uiLoop1][uiLoop2].uiDataSize ||
					f_strcmp( szTmpBuf,(char*)ppCompInfo[uiLoop1][uiLoop2].pvComp)
						!= 0)
				{
					rc = RC_SET( NE_XFLM_DATA_ERROR);
					MAKE_FLM_ERROR_STRING( "Unexpected key found.", m_szDetails, rc);
					goto Exit;
				}
				// else missing key component verified
				break;
			case XFLM_NUMBER_TYPE:
				if ( RC_BAD( rc = pKey->getUINT( uiLoop2, &uiTmp)))
				{
					if ( ppCompInfo[uiLoop1][uiLoop2].uiDataSize == 0 &&
						rc == NE_XFLM_NOT_FOUND)
					{
						rc = NE_XFLM_OK;
						break;
					}
					else
					{
						MAKE_FLM_ERROR_STRING( "getUINT failed.", m_szDetails, rc);
						goto Exit;
					}
				}
				if ( uiTmp != (FLMUINT)ppCompInfo[uiLoop1][uiLoop2].pvComp)
				{
					rc = RC_SET( NE_XFLM_DATA_ERROR);
					MAKE_FLM_ERROR_STRING( "Unexpected key found", m_szDetails, rc);
					goto Exit;
				}
				break;
			case XFLM_BINARY_TYPE:
				uiTmp = sizeof( szTmpBuf);
				if ( RC_BAD( rc = pKey->getBinary( uiLoop2, szTmpBuf, &uiTmp)))
				{
					if ( ppCompInfo[uiLoop1][uiLoop2].uiDataSize == 0 &&
						rc == NE_XFLM_NOT_FOUND)
					{
						rc = NE_XFLM_OK;
						break;
					}
					else
					{
						MAKE_FLM_ERROR_STRING( "getBinary failed.", m_szDetails, rc);
						goto Exit;
					}
				}

				if ( ( ppCompInfo[uiLoop1][uiLoop2].uiDataSize != uiTmp) ||
					f_memcmp( szTmpBuf,(char*)ppCompInfo[uiLoop1][uiLoop2].pvComp,
					ppCompInfo[uiLoop1][uiLoop2].uiDataSize)
					!= 0)
				{
					rc = RC_SET( NE_XFLM_DATA_ERROR);
					MAKE_FLM_ERROR_STRING( "Unexpected key found", m_szDetails, rc);
					goto Exit;
				}
				break;
			default:
				flmAssert(0);
			}
		}
	}

Exit:

	if( pKey)
	{
		pKey->Release();
	}

	return rc;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IndexTest3Impl::runQueries( 
	FLMUINT			uiIndexToUse)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Query *		pQuery = NULL;
	char 				szQuery[ 100];
	const char *	pszQueryFormat;
	FLMUINT			uiIndexUsed;
	FLMBOOL			bCaseInsensitive = FALSE;

	if( RC_BAD( rc = m_pDbSystem->createIFQuery( &pQuery)))
	{
		goto Exit;
	}

Go_Again:
	{
		pszQueryFormat = "//text_val[. > %s\"ABCD\"]";
		const char *	ppszResults[] = {"EFGH", "EFGH", "ijkl", "ijkl", "IJKL", "IJKL"};

		if ( bCaseInsensitive)
		{
			f_sprintf( szQuery, pszQueryFormat, "{ci}");
		}
		else
		{
			f_sprintf( szQuery, pszQueryFormat, "");
		}

		if ( RC_BAD( rc = doQueryTest( szQuery, ppszResults,
			sizeof( ppszResults) / sizeof( ppszResults[0]), pQuery, 
			m_szDetails, uiIndexToUse, &uiIndexUsed)))
		{
			goto Exit;
		}

		if ( uiIndexToUse != uiIndexUsed)
		{
			rc = RC_SET( NE_XFLM_FAILURE);
			goto Exit;
		}
	}

	{
		pszQueryFormat = "//text_val[. >= %s\"ABCD\"]";
		const char *	ppszResults[] = {"ABCD", "EFGH", "EFGH", "ijkl", "ijkl", "IJKL", "IJKL"};

		if ( bCaseInsensitive)
		{
			f_sprintf( szQuery, pszQueryFormat, "{ci}");
		}
		else
		{
			f_sprintf( szQuery, pszQueryFormat, "");
		}

		if ( RC_BAD( rc = doQueryTest( szQuery, ppszResults,
			sizeof(ppszResults)/sizeof(ppszResults[0]), pQuery, 
			m_szDetails, uiIndexToUse, &uiIndexUsed)))
		{
			goto Exit;
		}

		if ( uiIndexToUse != uiIndexUsed)
		{
			rc = RC_SET( NE_XFLM_FAILURE);
			goto Exit;
		}
	}

	{
		pszQueryFormat = "//text_val[. < %s\"ijkl\"]";
		const char *	ppszResults[] = { "ABCD", "EFGH", "EFGH"};

		if ( bCaseInsensitive)
		{
			f_sprintf( szQuery, pszQueryFormat, "{ci}");
		}
		else
		{
			f_sprintf( szQuery, pszQueryFormat, "");
		}

		if ( RC_BAD( rc = doQueryTest( szQuery, ppszResults,
			sizeof(ppszResults)/sizeof(ppszResults[0]), pQuery, 
			m_szDetails, uiIndexToUse, &uiIndexUsed)))
		{
			goto Exit;
		}

		if ( uiIndexToUse != uiIndexUsed)
		{
			rc = RC_SET( NE_XFLM_FAILURE);
			goto Exit;
		}
	}

	{
		pszQueryFormat = "//text_val[. <= %s\"ijkl\"]";

		if ( bCaseInsensitive)
		{
			const char *	ppszResults[] = {"ABCD", "EFGH", "EFGH", "IJKL", "IJKL", "ijkl", "ijkl"};

			f_sprintf( szQuery, pszQueryFormat, "{ci}");

			if ( RC_BAD( rc = doQueryTest( szQuery, ppszResults,
				sizeof(ppszResults)/sizeof(ppszResults[0]), pQuery, 
				m_szDetails, uiIndexToUse, &uiIndexUsed)))
			{
				goto Exit;
			}
		}
		else
		{
			const char *	ppszResults[] = {"ABCD", "EFGH", "EFGH", "ijkl", "ijkl"};

			f_sprintf( szQuery, pszQueryFormat, "");

			if ( RC_BAD( rc = doQueryTest( szQuery, ppszResults,
				sizeof(ppszResults)/sizeof(ppszResults[0]), pQuery, 
				m_szDetails, uiIndexToUse, &uiIndexUsed)))
			{
				goto Exit;
			}
		}

		if ( uiIndexToUse != uiIndexUsed)
		{
			rc = RC_SET( NE_XFLM_FAILURE);
			goto Exit;
		}
	}

	{
		pszQueryFormat = "//text_val[. >= %s\"ijkl\"]";
		const char *	ppszResults[] = {"IJKL", "IJKL", "ijkl", "ijkl"};

		if ( bCaseInsensitive)
		{

			f_sprintf( szQuery, pszQueryFormat, "{ci}");
		}
		else
		{
			f_sprintf( szQuery, pszQueryFormat, "");
		}

		if ( RC_BAD( rc = doQueryTest( szQuery, ppszResults,
			sizeof(ppszResults)/sizeof(ppszResults[0]), pQuery, 
			m_szDetails, uiIndexToUse, &uiIndexUsed)))
		{
			goto Exit;
		}

		if ( uiIndexToUse != uiIndexUsed)
		{
			rc = RC_SET( NE_XFLM_FAILURE);
			goto Exit;
		}
	}

	{
		pszQueryFormat = "//text_val[. == %s\"IJKL\" && following-sibling::num_val[. == 123]]";

		if ( bCaseInsensitive)
		{
			const char *	ppszResults[] = {"IJKL", "ijkl"};
			f_sprintf( szQuery, pszQueryFormat, "{ci}");

			if ( RC_BAD( rc = doQueryTest( szQuery, ppszResults,
				sizeof(ppszResults)/sizeof(ppszResults[0]), pQuery, 
				m_szDetails, uiIndexToUse, &uiIndexUsed)))
			{
				goto Exit;
			}
		}
		else
		{
			const char *	ppszResults[] = {"IJKL"};
			f_sprintf( szQuery, pszQueryFormat, "");

			if ( RC_BAD( rc = doQueryTest( szQuery, ppszResults,
				sizeof(ppszResults)/sizeof(ppszResults[0]), pQuery, 
				m_szDetails, uiIndexToUse, &uiIndexUsed)))
			{
				goto Exit;
			}
		}

		if ( uiIndexToUse != uiIndexUsed)
		{
			rc = RC_SET( NE_XFLM_FAILURE);
			goto Exit;
		}
	}

	{
		pszQueryFormat = "//text_val[. == %s\"ij*\"]";

		if ( bCaseInsensitive)
		{
			const char *	ppszResults[] = {"ijkl", "ijkl", "IJKL", "IJKL"};
			f_sprintf( szQuery, pszQueryFormat, "{ci}");

			if ( RC_BAD( rc = doQueryTest( szQuery, ppszResults,
				sizeof(ppszResults)/sizeof(ppszResults[0]), pQuery, 
				m_szDetails, uiIndexToUse, &uiIndexUsed)))
			{
				goto Exit;
			}
		}
		else
		{
			const char *	ppszResults[] = {"ijkl", "ijkl"};
	
			f_sprintf( szQuery, pszQueryFormat, "");

			if ( RC_BAD( rc = doQueryTest( szQuery, ppszResults,
				sizeof(ppszResults)/sizeof(ppszResults[0]), pQuery, 
				m_szDetails, uiIndexToUse, &uiIndexUsed)))
			{
				goto Exit;
			}
		}

		if ( uiIndexToUse != uiIndexUsed)
		{
			rc = RC_SET( NE_XFLM_FAILURE);
			goto Exit;
		}
	}

	{
		pszQueryFormat = "//text_val[. == %s\"*GH\"]";
		const char *	ppszResults[] = {"EFGH", "EFGH"};

		if ( bCaseInsensitive)
		{
			f_sprintf( szQuery, pszQueryFormat, "{ci}");
		}
		else
		{
			f_sprintf( szQuery, pszQueryFormat, "");
		}

		if ( RC_BAD( rc = doQueryTest( szQuery, ppszResults,
			sizeof(ppszResults)/sizeof(ppszResults[0]), pQuery, 
			m_szDetails, uiIndexToUse, &uiIndexUsed)))
		{
			goto Exit;
		}

		if ( uiIndexToUse != uiIndexUsed)
		{
			rc = RC_SET( NE_XFLM_FAILURE);
			goto Exit;
		}
	}

	{
		pszQueryFormat = "//text_val[. == %s\"*G*\"]";
		const char *	ppszResults[] = {"EFGH", "EFGH"};

		if ( bCaseInsensitive)
		{
			f_sprintf( szQuery, pszQueryFormat, "{ci}");
		}
		else
		{
			f_sprintf( szQuery, pszQueryFormat, "");
		}

		if ( RC_BAD( rc = doQueryTest( szQuery, ppszResults,
			sizeof(ppszResults)/sizeof(ppszResults[0]), pQuery, 
			m_szDetails, uiIndexToUse, &uiIndexUsed)))
		{
			goto Exit;
		}

		if ( uiIndexToUse != uiIndexUsed)
		{
			rc = RC_SET( NE_XFLM_FAILURE);
			goto Exit;
		}
	}

	if ( !bCaseInsensitive)
	{
		bCaseInsensitive = TRUE;
		goto Go_Again;
	}

Exit:

	if ( pQuery)
	{
		pQuery->Release();
	}

	return rc;
}


#define SUITE4_IX_NUM 42

/****************************************************************************
Desc:
****************************************************************************/
RCODE IndexTest3Impl::runSuite4( void)
{
	RCODE					rc = NE_XFLM_OK;
	FLMBOOL				bDibCreated = FALSE;
	FLMBOOL				bTransBegun = FALSE;
	FLMUINT				uiLoop;
	
	beginTest( 	
		"Truncated Key Test Setup",
		"Setup DIB for the truncated key tests",
		"Self-explanatory",
		"");

	if ( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to init test state.", m_szDetails, rc);
		goto Fatal_Error;
	}
	bDibCreated = TRUE;

	if ( RC_BAD( rc = m_pDb->createElementDef(
		NULL, "text_val", XFLM_TEXT_TYPE, &m_uiTextId, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Fatal_Error;
	}

	if ( RC_BAD( rc = m_pDb->createElementDef(
		NULL, "num_val", XFLM_NUMBER_TYPE, &m_uiNumId, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Fatal_Error;
	}

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Fatal_Error;
	}
	bTransBegun = TRUE;


	// Create documents. 

	{
		ELEMENT_NODE_INFO	pElementNodes[][2] =
			{
				{
					{(void*)"ABCD", XFLM_TEXT_TYPE, 4, m_uiTextId},
					{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT), m_uiNumId}
				},

				{
					{(void*)"EFGH", XFLM_TEXT_TYPE, 4, m_uiTextId},
					{NULL, XFLM_NODATA_TYPE, 0, 0} // Placeholder so the array won't be jagged
				},

				{
					{(void*)"EFGH", XFLM_TEXT_TYPE, 4, m_uiTextId},
					{(void*)456, XFLM_NUMBER_TYPE, sizeof(FLMUINT), m_uiNumId}
				},

				{
					{(void*)"IJKL", XFLM_TEXT_TYPE, 4, m_uiTextId},
					{(void*)456, XFLM_NUMBER_TYPE, sizeof(FLMUINT), m_uiNumId}
				},

				{
					{(void*)"IJKL", XFLM_TEXT_TYPE, 4, m_uiTextId},
					{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT), m_uiNumId}
				},

				{
					{(void*)"ijkl", XFLM_TEXT_TYPE, 4, m_uiTextId},
					{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT), m_uiNumId}
				},

				{
					{(void*)"ijkl", XFLM_TEXT_TYPE, 4, m_uiTextId},
					{NULL, XFLM_NODATA_TYPE, 0, 0} // Placeholder so the array won't be jagged
				},

				{
					{NULL, XFLM_NODATA_TYPE, 0, 0}, // Placeholder so the array won't be jagged
					{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT), m_uiNumId}
				},
			};

		for( uiLoop = 0; uiLoop < 8; uiLoop++)
		{
			if ( RC_BAD( rc = createCompoundDoc( 
				pElementNodes[uiLoop], 2, NULL)))
			{
				goto Fatal_Error;
			}
		}

		// Commit the transaction so we can have something to examine in the 
		// database should any test fail.

		if ( RC_BAD( rc = m_pDb->transCommit()))
		{
			MAKE_FLM_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
			goto Fatal_Error;
		}
		bTransBegun = FALSE;
	}

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Fatal_Error;
	}
	bTransBegun = TRUE;

	endTest("PASS");

	beginTest( 	
		"Key Sort Test (primary key: descending/case-sensitive/missing high; "
		"secondary key: descending/missing high)",
		"Setup DIB for the truncated key tests",
		"Self-explanatory",
		"");

	{
		// Array of key components in sorted order

		KEY_COMP_INFO ppComponents[][2] = 
		{
			{
				{(void*)NULL, XFLM_TEXT_TYPE, 0}, // missing component
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)456, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},			
				
			{
				{(void*)"ijkl", XFLM_TEXT_TYPE, 4},
				{(void*)NULL, XFLM_NUMBER_TYPE, 0}, // missing component
			},

			{
				{(void*)"ijkl", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"EFGH", XFLM_TEXT_TYPE, 4},
				{(void*)0, XFLM_NUMBER_TYPE, 0}, // missing component
			},

			{
				{(void*)"EFGH", XFLM_TEXT_TYPE, 4},
				{(void*)456, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"ABCD", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

		};

		if ( RC_BAD( rc = createOrModifyIndex( 
			SUITE4_IX_NUM,
			TRUE, // Comp #1 sort descending
			TRUE, // Comp #1 case-sensitive
			TRUE, // Comp #1 sort missing high
			TRUE, // Comp #2 sort descending
			TRUE))) // Comp #2 sort missing high
		{
			endTest("FAIL");
			goto Exit1;
		}

		if ( RC_BAD( rc = verifyKeyOrder( SUITE4_IX_NUM, ppComponents, 8)))
		{
			endTest("FAIL");
			goto Exit1;
		}

		// Make sure the new index doesn't break the queries

		if ( RC_BAD( rc = runQueries( SUITE4_IX_NUM)))
		{
			endTest("FAIL");
			goto Exit1;
		}
	}

	endTest("PASS");

Exit1:


	beginTest( 	
		"Key Sort Test (primary key: descending/case-sensitive; "
		"secondary key: descending)",
		"Setup DIB for the truncated key tests",
		"Self-explanatory",
		"");

	{
		// Array of key components in sorted order

		KEY_COMP_INFO ppComponents[][2] = 
		{
			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)456, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},
				
			{
				{(void*)"ijkl", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"ijkl", XFLM_TEXT_TYPE, 4},
				{(void*)0, XFLM_NUMBER_TYPE, 0}, // missing component
			},

			{
				{(void*)"EFGH", XFLM_TEXT_TYPE, 4},
				{(void*)456, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"EFGH", XFLM_TEXT_TYPE, 4},
				{(void*)0, XFLM_NUMBER_TYPE, 0}, // missing component
			},

			{
				{(void*)"ABCD", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)NULL, XFLM_TEXT_TYPE, 0}, // missing component
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},
		};

		if ( RC_BAD( rc = createOrModifyIndex( 
			SUITE4_IX_NUM,
			TRUE, // Comp #1 sort descending
			TRUE, // Comp #1 case-sensitive
			FALSE, // Comp #1 sort missing high
			TRUE, // Comp #2 sort descending
			FALSE))) // Comp #2 sort missing high
		{
			endTest("FAIL");
			goto Exit2;
		}

		if ( RC_BAD( rc = verifyKeyOrder( SUITE4_IX_NUM, ppComponents, 8)))
		{
			endTest("FAIL");
			goto Exit2;
		}

		// Make sure the new index doesn't break the queries

		if ( RC_BAD( rc = runQueries( SUITE4_IX_NUM)))
		{
			endTest("FAIL");
			goto Exit2;
		}
	}

	endTest("PASS");

Exit2:

	beginTest( 	
		"Key Sort Test (primary key: ascending/case-sensitive; "
		"secondary key: descending)",
		"Setup DIB for the truncated key tests",
		"Self-explanatory",
		"");

	{
		// Array of key components in sorted order

		KEY_COMP_INFO ppComponents[][2] = 
		{
			{
				{(void*)NULL, XFLM_TEXT_TYPE, 0}, // missing component
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"ABCD", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"EFGH", XFLM_TEXT_TYPE, 4},
				{(void*)456, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},
		
			{
				{(void*)"EFGH", XFLM_TEXT_TYPE, 4},
				{(void*)NULL, XFLM_NUMBER_TYPE, 0}, // missing component
			},

			{
				{(void*)"ijkl", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"ijkl", XFLM_TEXT_TYPE, 4},
				{(void*)NULL, XFLM_NUMBER_TYPE, 0}, // missing component
			},
				
			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)456, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},
	
		};

		if ( RC_BAD( rc = createOrModifyIndex( 
			SUITE4_IX_NUM,
			FALSE, // Comp #1 sort descending
			TRUE, // Comp #1 case-sensitive
			FALSE, // Comp #1 sort missing high
			TRUE, // Comp #2 sort descending
			FALSE))) // Comp #2 sort missing high
		{
			endTest("FAIL");
			goto Exit3;
		}

		if ( RC_BAD( rc = verifyKeyOrder( SUITE4_IX_NUM, ppComponents, 8)))
		{
			endTest("FAIL");
			goto Exit3;
		}

		// Make sure the new index doesn't break the queries

		if ( RC_BAD( rc = runQueries( SUITE4_IX_NUM)))
		{
			endTest("FAIL");
			goto Exit3;
		}
	}

	endTest("PASS");

Exit3:

	beginTest( 	
		"Key Sort Test (primary key: ascending/case-sensitive; "
		"secondary key: descending/missing high)",
		"Setup DIB for the truncated key tests",
		"Self-explanatory",
		"");

	{
		// Array of key components in sorted order

		KEY_COMP_INFO ppComponents[][2] = 
		{
			{
				{(void*)NULL, XFLM_TEXT_TYPE, 0}, // missing component
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"ABCD", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},
			
			{
				{(void*)"EFGH", XFLM_TEXT_TYPE, 4},
				{(void*)NULL, XFLM_NUMBER_TYPE, 0}, // missing component
			},

			{
				{(void*)"EFGH", XFLM_TEXT_TYPE, 4},
				{(void*)456, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},
	
			{
				{(void*)"ijkl", XFLM_TEXT_TYPE, 4},
				{(void*)NULL, XFLM_NUMBER_TYPE, 0}, // missing component
			},

			{
				{(void*)"ijkl", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},
			
			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)456, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},
		};

		if ( RC_BAD( rc = createOrModifyIndex( 
			SUITE4_IX_NUM,
			FALSE, // Comp #1 sort descending
			TRUE, // Comp #1 case-sensitive
			FALSE, // Comp #1 sort missing high
			TRUE, // Comp #2 sort descending
			TRUE))) // Comp #2 sort missing high
		{
			endTest("FAIL");
			goto Exit4;
		}

		if ( RC_BAD( rc = verifyKeyOrder( SUITE4_IX_NUM, ppComponents, 8)))
		{
			endTest("FAIL");
			goto Exit4;
		}

		// Make sure the new index doesn't break the queries

		if ( RC_BAD( rc = runQueries( SUITE4_IX_NUM)))
		{
			endTest("FAIL");
			goto Exit4;
		}
	}

	endTest("PASS");

Exit4:

	beginTest( 	
		"Key Sort Test (primary key: descending/case-sensitive; "
		"secondary key: ascending)",
		"Setup DIB for the truncated key tests",
		"Self-explanatory",
		"");

	{
		// Array of key components in sorted order

		KEY_COMP_INFO ppComponents[][2] = 
		{
			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)}, // missing component
			},

			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)456, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"ijkl", XFLM_TEXT_TYPE, 4},
				{(void*)NULL, XFLM_NUMBER_TYPE, 0},
			},

			{
				{(void*)"ijkl", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},
		
			{
				{(void*)"EFGH", XFLM_TEXT_TYPE, 4},
				{(void*)NULL, XFLM_NUMBER_TYPE, 0},
			},

			{
				{(void*)"EFGH", XFLM_TEXT_TYPE, 4},
				{(void*)456, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"ABCD", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)}, // missing component
			},

			{
				{(void*)NULL, XFLM_TEXT_TYPE, 0}, // missing component
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

		};

		if ( RC_BAD( rc = createOrModifyIndex( 
			SUITE4_IX_NUM,
			TRUE, // Comp #1 sort descending
			TRUE, // Comp #1 case-sensitive
			FALSE, // Comp #1 sort missing high
			FALSE, // Comp #2 sort descending
			FALSE))) // Comp #2 sort missing high
		{
			endTest("FAIL");
			goto Exit5;
		}

		if ( RC_BAD( rc = verifyKeyOrder( SUITE4_IX_NUM, ppComponents, 8)))
		{
			endTest("FAIL");
			goto Exit5;
		}

		// Make sure the new index doesn't break the queries

		if ( RC_BAD( rc = runQueries( SUITE4_IX_NUM)))
		{
			endTest("FAIL");
			goto Exit5;
		}
	}

	endTest("PASS");

Exit5:

	beginTest( 	
		"Key Sort Test (primary key: descending/case-sensitive; "
		"secondary key: ascending/missing high)",
		"Setup DIB for the truncated key tests",
		"Self-explanatory",
		"");

	{
		// Array of key components in sorted order

		KEY_COMP_INFO ppComponents[][2] = 
		{		
			
			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)}, // missing component
			},

			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)456, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"ijkl", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},
		
			{
				{(void*)"ijkl", XFLM_TEXT_TYPE, 4},
				{(void*)0, XFLM_NUMBER_TYPE, 0},
			},

			{
				{(void*)"EFGH", XFLM_TEXT_TYPE, 4},
				{(void*)456, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"EFGH", XFLM_TEXT_TYPE, 4},
				{(void*)0, XFLM_NUMBER_TYPE, 0},
			},

			{
				{(void*)"ABCD", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)}, // missing component
			},

			{
				{(void*)NULL, XFLM_TEXT_TYPE, 0}, // missing component
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

		};

		if ( RC_BAD( rc = createOrModifyIndex( 
			SUITE4_IX_NUM,
			TRUE, // Comp #1 sort descending
			TRUE, // Comp #1 case-sensitive
			FALSE, // Comp #1 sort missing high
			FALSE, // Comp #2 sort descending
			TRUE))) // Comp #2 sort missing high
		{
			endTest("FAIL");
			goto Exit6;
		}

		if ( RC_BAD( rc = verifyKeyOrder( SUITE4_IX_NUM, ppComponents, 8)))
		{
			endTest("FAIL");
			goto Exit6;
		}

		// Make sure the new index doesn't break the queries

		if ( RC_BAD( rc = runQueries( SUITE4_IX_NUM)))
		{
			endTest("FAIL");
			goto Exit6;
		}
	}

	endTest("PASS");

Exit6:

	// Case-insensitive index tests

	beginTest( 	
		"Key Sort Test (primary key: descending/case-insensitive/missing high; "
		"secondary key: descending/missing high)",
		"Setup DIB for the truncated key tests",
		"Self-explanatory",
		"");

	{
		// Array of key components in sorted order

		KEY_COMP_INFO ppComponents[][2] = 
		{
			{
				{(void*)NULL, XFLM_TEXT_TYPE, 0}, // missing component
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)NULL, XFLM_NUMBER_TYPE, 0}, // missing component
			},

			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)456, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},			
				
			{
				{(void*)"EFGH", XFLM_TEXT_TYPE, 4},
				{(void*)0, XFLM_NUMBER_TYPE, 0}, // missing component
			},

			{
				{(void*)"EFGH", XFLM_TEXT_TYPE, 4},
				{(void*)456, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"ABCD", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

		};

		if ( RC_BAD( rc = createOrModifyIndex( 
			SUITE4_IX_NUM,
			TRUE, // Comp #1 sort descending
			FALSE, // Comp #1 case-sensitive
			TRUE, // Comp #1 sort missing high
			TRUE, // Comp #2 sort descending
			TRUE))) // Comp #2 sort missing high
		{
			endTest("FAIL");
			goto Exit7;
		}

		if ( RC_BAD( rc = verifyKeyOrder( SUITE4_IX_NUM, ppComponents, 7)))
		{
			endTest("FAIL");
			goto Exit7;
		}

		// Make sure the new index doesn't break the queries

		if ( RC_BAD( rc = runQueries( SUITE4_IX_NUM)))
		{
			endTest("FAIL");
			goto Exit7;
		}
	}

	endTest("PASS");

Exit7:

	beginTest( 	
		"Key Sort Test (primary key: descending/case-insensitive; "
		"secondary key: descending)",
		"Setup DIB for the truncated key tests",
		"Self-explanatory",
		"");

	{
		// Array of key components in sorted order

		KEY_COMP_INFO ppComponents[][2] = 
		{
			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)456, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},
				
			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)0, XFLM_NUMBER_TYPE, 0}, // missing component
			},

			{
				{(void*)"EFGH", XFLM_TEXT_TYPE, 4},
				{(void*)456, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"EFGH", XFLM_TEXT_TYPE, 4},
				{(void*)0, XFLM_NUMBER_TYPE, 0}, // missing component
			},

			{
				{(void*)"ABCD", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)NULL, XFLM_TEXT_TYPE, 0}, // missing component
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},
		};

		if ( RC_BAD( rc = createOrModifyIndex( 
			SUITE4_IX_NUM,
			TRUE, // Comp #1 sort descending
			FALSE, // Comp #1 case-sensitive
			FALSE, // Comp #1 sort missing high
			TRUE, // Comp #2 sort descending
			FALSE))) // Comp #2 sort missing high
		{
			endTest("FAIL");
			goto Exit8;
		}

		if ( RC_BAD( rc = verifyKeyOrder( SUITE4_IX_NUM, ppComponents, 7)))
		{
			endTest("FAIL");
			goto Exit8;
		}

		// Make sure the new index doesn't break the queries

		if ( RC_BAD( rc = runQueries( SUITE4_IX_NUM)))
		{
			endTest("FAIL");
			goto Exit8;
		}
	}

	endTest("PASS");

Exit8:

	beginTest( 	
		"Key Sort Test (primary key: ascending/case-insensitive; "
		"secondary key: descending)",
		"Setup DIB for the truncated key tests",
		"Self-explanatory",
		"");

	{
		// Array of key components in sorted order

		KEY_COMP_INFO ppComponents[][2] = 
		{
			{
				{(void*)NULL, XFLM_TEXT_TYPE, 0}, // missing component
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"ABCD", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"EFGH", XFLM_TEXT_TYPE, 4},
				{(void*)456, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},
		
			{
				{(void*)"EFGH", XFLM_TEXT_TYPE, 4},
				{(void*)NULL, XFLM_NUMBER_TYPE, 0}, // missing component
			},

			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)456, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},
			
			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)NULL, XFLM_NUMBER_TYPE, 0}, // missing component
			},
	
		};

		if ( RC_BAD( rc = createOrModifyIndex( 
			SUITE4_IX_NUM,
			FALSE, // Comp #1 sort descending
			FALSE, // Comp #1 case-sensitive
			FALSE, // Comp #1 sort missing high
			TRUE, // Comp #2 sort descending
			FALSE))) // Comp #2 sort missing high
		{
			endTest("FAIL");
			goto Exit9;
		}

		if ( RC_BAD( rc = verifyKeyOrder( SUITE4_IX_NUM, ppComponents, 7)))
		{
			endTest("FAIL");
			goto Exit9;
		}

		// Make sure the new index doesn't break the queries

		if ( RC_BAD( rc = runQueries( SUITE4_IX_NUM)))
		{
			endTest("FAIL");
			goto Exit9;
		}
	}

	endTest("PASS");

Exit9:

	beginTest( 	
		"Key Sort Test (primary key: ascending/case-insensitive; "
		"secondary key: descending/missing high)",
		"Setup DIB for the truncated key tests",
		"Self-explanatory",
		"");

	{
		// Array of key components in sorted order

		KEY_COMP_INFO ppComponents[][2] = 
		{
			{
				{(void*)NULL, XFLM_TEXT_TYPE, 0}, // missing component
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"ABCD", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},
			
			{
				{(void*)"EFGH", XFLM_TEXT_TYPE, 4},
				{(void*)NULL, XFLM_NUMBER_TYPE, 0}, // missing component
			},

			{
				{(void*)"EFGH", XFLM_TEXT_TYPE, 4},
				{(void*)456, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},
	
			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)NULL, XFLM_NUMBER_TYPE, 0}, // missing component
			},

			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)456, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},
		};

		if ( RC_BAD( rc = createOrModifyIndex( 
			SUITE4_IX_NUM,
			FALSE, // Comp #1 sort descending
			FALSE, // Comp #1 case-sensitive
			FALSE, // Comp #1 sort missing high
			TRUE, // Comp #2 sort descending
			TRUE))) // Comp #2 sort missing high
		{
			endTest("FAIL");
			goto Exit10;
		}

		if ( RC_BAD( rc = verifyKeyOrder( SUITE4_IX_NUM, ppComponents, 7)))
		{
			endTest("FAIL");
			goto Exit10;
		}

		// Make sure the new index doesn't break the queries

		if ( RC_BAD( rc = runQueries( SUITE4_IX_NUM)))
		{
			endTest("FAIL");
			goto Exit10;
		}
	}

	endTest("PASS");

Exit10:

	beginTest( 	
		"Key Sort Test (primary key: descending/case-insensitive; "
		"secondary key: ascending)",
		"Setup DIB for the truncated key tests",
		"Self-explanatory",
		"");

	{
		// Array of key components in sorted order

		KEY_COMP_INFO ppComponents[][2] = 
		{
			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)NULL, XFLM_NUMBER_TYPE, 0},
			},

			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)}, // missing component
			},

			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)456, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"EFGH", XFLM_TEXT_TYPE, 4},
				{(void*)NULL, XFLM_NUMBER_TYPE, 0},
			},

			{
				{(void*)"EFGH", XFLM_TEXT_TYPE, 4},
				{(void*)456, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"ABCD", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)}, // missing component
			},

			{
				{(void*)NULL, XFLM_TEXT_TYPE, 0}, // missing component
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

		};

		if ( RC_BAD( rc = createOrModifyIndex( 
			SUITE4_IX_NUM,
			TRUE, // Comp #1 sort descending
			FALSE, // Comp #1 case-sensitive
			FALSE, // Comp #1 sort missing high
			FALSE, // Comp #2 sort descending
			FALSE))) // Comp #2 sort missing high
		{
			endTest("FAIL");
			goto Exit11;
		}

		if ( RC_BAD( rc = verifyKeyOrder( SUITE4_IX_NUM, ppComponents, 7)))
		{
			endTest("FAIL");
			goto Exit11;
		}

		// Make sure the new index doesn't break the queries

		if ( RC_BAD( rc = runQueries( SUITE4_IX_NUM)))
		{
			endTest("FAIL");
			goto Exit11;
		}
	}

	endTest("PASS");

Exit11:

	beginTest( 	
		"Key Sort Test (primary key: descending/case-insensitive; "
		"secondary key: ascending/missing high)",
		"Setup DIB for the truncated key tests",
		"Self-explanatory",
		"");

	{
		// Array of key components in sorted order

		KEY_COMP_INFO ppComponents[][2] = 
		{		
			
			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)}, // missing component
			},

			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)456, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"IJKL", XFLM_TEXT_TYPE, 4},
				{(void*)0, XFLM_NUMBER_TYPE, 0},
			},

			{
				{(void*)"EFGH", XFLM_TEXT_TYPE, 4},
				{(void*)456, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

			{
				{(void*)"EFGH", XFLM_TEXT_TYPE, 4},
				{(void*)0, XFLM_NUMBER_TYPE, 0},
			},

			{
				{(void*)"ABCD", XFLM_TEXT_TYPE, 4},
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)}, // missing component
			},

			{
				{(void*)NULL, XFLM_TEXT_TYPE, 0}, // missing component
				{(void*)123, XFLM_NUMBER_TYPE, sizeof(FLMUINT)},
			},

		};

		if ( RC_BAD( rc = createOrModifyIndex( 
			SUITE4_IX_NUM,
			TRUE, // Comp #1 sort descending
			FALSE, // Comp #1 case-sensitive
			FALSE, // Comp #1 sort missing high
			FALSE, // Comp #2 sort descending
			TRUE))) // Comp #2 sort missing high
		{
			endTest("FAIL");
			goto Exit12;
		}

		if ( RC_BAD( rc = verifyKeyOrder( SUITE4_IX_NUM, ppComponents, 7)))
		{
			endTest("FAIL");
			goto Exit12;
		}

		// Make sure the new index doesn't break the queries

		if ( RC_BAD( rc = runQueries( SUITE4_IX_NUM)))
		{
			endTest("FAIL");
			goto Exit12;
		}
	}

	endTest("PASS");

Exit12:
	goto Exit;


Fatal_Error:
	// We had an error setting up the database so we had to bail out early
	endTest("FAIL");

Exit:

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
RCODE IndexTest3Impl::runSuite5( void)
{
	RCODE					rc = NE_XFLM_OK;
	FLMBOOL				bDibCreated = FALSE;
	FLMBOOL				bTransActive = FALSE;
	IF_DataVector *	pSearchKey = NULL;

	const char *		pszIndexDef =
		"<xflaim:Index "
		"	xmlns:xflaim=\"http://www.novell.com/XMLDatabase/Schema\""
		"	xflaim:name=\"Suite5Index\" "
		"	xflaim:DictNumber=\"1\"> "
		"	<xflaim:ElementComponent xflaim:name=\"Suite5Entry\" xflaim:KeyComponent=\"1\">"
		"		<xflaim:AttributeComponent"
		"			xflaim:name=\"Suite5Attr1\" "
		"			xflaim:IndexOn=\"value\" "
		"			xflaim:Required=\"1\" "
		"			xflaim:KeyComponent=\"2\" />"
		"	</xflaim:ElementComponent> "
		"</xflaim:Index> ";

	const char *	pszObject =
		"<Suite5Entry Suite5Attr1=\"suite5value1\">"
		"	<Suite5Entry/>"
		"	<Suite5Entry2/>"
		"</Suite5Entry>";

	beginTest( 	
		"Test indexDocument context changes",
		"Create a document and then an index that should include that document.",
		"Self-Explanatory",
		"");

	if ( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to init test state.", m_szDetails, rc);
		goto Exit;
	}
	bDibCreated = TRUE;

	if ( RC_BAD( rc = m_pDb->transBegin(XFLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransActive = TRUE;


	if ( RC_BAD( rc = importDocument( pszObject, XFLM_DATA_COLLECTION)))
	{
		MAKE_FLM_ERROR_STRING( "importDocument failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = importDocument( pszIndexDef, XFLM_DICT_COLLECTION)))
	{
		MAKE_FLM_ERROR_STRING( "importDocument failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_FLM_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransActive = FALSE;

	if( RC_BAD( rc = m_pDb->keyRetrieve(
			1, NULL, XFLM_FIRST | XFLM_MATCH_DOC_ID, pSearchKey)))
	{
		MAKE_FLM_ERROR_STRING( "keyRetrieve failed", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

Exit:

	if ( RC_BAD( rc ))
	{
		endTest("FAIL");
	}

	if ( bTransActive)
	{
		if ( RC_OK(rc))
		{
			m_pDb->transCommit();
		}
		else
		{
			m_pDb->transAbort();
		}
	}

	if (pSearchKey)
	{
		pSearchKey->Release();
	}

	shutdownTestState( DB_NAME_STR, bDibCreated);
	return rc;
}
