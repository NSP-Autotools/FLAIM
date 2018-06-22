//------------------------------------------------------------------------------
// Desc:	Namespace unit test
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

#if defined(NLM)
	#define DB_NAME_STR					"SYS:\\TST.DB"
#else
	#define DB_NAME_STR					"tst.db"
#endif

/****************************************************************************
Desc:
****************************************************************************/
class INamespaceTestImpl : public TestBase
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

	if( (*ppTest = f_new INamespaceTestImpl) == NULL)
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
const char * INamespaceTestImpl::getName( void)
{
	return( "Namespace Test");
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE INamespaceTestImpl::execute( void)
{
	RCODE						rc = NE_XFLM_OK;
	IF_PosIStream	*		pPosIStream = NULL;
	IF_Query *				pQuery = NULL;
	IF_DOMNode *			pReturn = NULL;
	char						szNamespace[128];
	FLMBOOL					bTransStarted = FALSE;
	const char *			pszDocument =
		"<person xmlns=\"http://www.novell.com/Bogus\">"
		"	<name first=\"John\" middle =\"Q\" last=\"Doe\" />"
		"	<age>27</age>"
		"	<gender>Male</gender>"
		"	<occupation xmlns=\"\">"
		"		<name>Dishwasher</name>"
		"		<salary>10000</salary>"
		"	</occupation>"
		"	<height>5'10</height>"
		"	<weight>200</weight>"
		"</person>";

	m_szDetails[0] = '\0';

	beginTest( 
		"Default Namespace Test",
		"Set the default namespace of a document and ensure it "
		"is reflected in the various nodes when it is changed or overridden",
		"1) create a database 2) import a document containing "
		"different default namespaces at different levels 3) retrieve various "
		"nodes and ensure their namespaces are correct.",
		"");

	if ( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to begin trans.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;
	
	if ( RC_BAD( rc = importBuffer( pszDocument, XFLM_DATA_COLLECTION)))
	{
		 goto Exit;
	}

	if( RC_BAD( rc = m_pDbSystem->createIFQuery( &pQuery)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to create query object.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, 
		"{xmlns=\"http://www.novell.com/Bogus\"}/person/name")))
	{
		MAKE_FLM_ERROR_STRING( "Failed to setup query expression.", 
			m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pQuery->getFirst( m_pDb, &pReturn)))
	{
		MAKE_FLM_ERROR_STRING( "getFirst failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pReturn->getNamespaceURI( m_pDb, szNamespace, sizeof( szNamespace), NULL)))
	{
		MAKE_FLM_ERROR_STRING( "getNamespaceURI failed", m_szDetails, rc);
		goto Exit;
	}

	if ( f_strcmp( szNamespace, "http://www.novell.com/Bogus") != 0)
	{
		rc = NE_XFLM_FAILURE;
		MAKE_FLM_ERROR_STRING( "element has unexpected namespace", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, "/occupation/name")))
	{
		MAKE_FLM_ERROR_STRING( "Failed to setup query expression.", 
			m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pQuery->getFirst( m_pDb, &pReturn)))
	{
		MAKE_FLM_ERROR_STRING( "getFirst failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pReturn->getNamespaceURI( 
			m_pDb, 
			szNamespace,
			sizeof( szNamespace), 
			NULL)))
	{
		MAKE_FLM_ERROR_STRING( "unexpected rcode from getNamespaceURI.", 
			m_szDetails, rc);
		if ( RC_OK( rc))
		{
			rc = NE_XFLM_FAILURE;
		}
		goto Exit;
	}

	if ( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, 
		"{xmlns=\"http://www.novell.com/Bogus\"}/person/height")))
	{
		MAKE_FLM_ERROR_STRING( "Failed to setup query expression.", 
			m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pQuery->getFirst( m_pDb, &pReturn)))
	{
		MAKE_FLM_ERROR_STRING( "getFirst failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pReturn->getNamespaceURI( m_pDb, szNamespace, 
		sizeof( szNamespace), NULL)))
	{
		MAKE_FLM_ERROR_STRING( "getNamespaceURI failed", m_szDetails, rc);
		goto Exit;
	}

	if ( f_strcmp( szNamespace, "http://www.novell.com/Bogus") != 0)
	{
		rc = NE_XFLM_FAILURE;
		MAKE_FLM_ERROR_STRING( "element has unexpected namespace", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

Exit:

	if( RC_BAD( rc))
	{
		endTest("FAIL");
	}
	
	if( bTransStarted)
	{
		RCODE tmpRc = m_pDb->transCommit();
		if ( RC_OK( rc))
		{
			rc = tmpRc;
		}
	}
	
	if( pPosIStream)
	{
		pPosIStream->Release();
	}

	if( pQuery)
	{
		pQuery->Release();
	}

	if( pReturn)
	{
		pReturn->Release();
	}

	RCODE tmpRc = NE_XFLM_OK;
	tmpRc = shutdownTestState(DB_NAME_STR, TRUE);
	if( RC_OK( rc))
	{
		rc = tmpRc;
	}

	return( rc);
}
