//------------------------------------------------------------------------------
// Desc:	More XPATH Query tests. These tests focus on the META axis and the
//			Node subscript ([]).
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
		#define DB_NAME_STR					"SYS:\\XP3.DB"
	#else
		#define DB_NAME_STR					"xp3.db"
	#endif
#endif

/****************************************************************************
Desc:
****************************************************************************/
class IXPATHTest2Impl : public TestBase
{
public:

	const char * getName( void);
	
	RCODE execute( void);
	
private:

	RCODE runSuite1( void);
	
	RCODE runSuite2( void);
	
	RCODE runSuite3( void);
	
	RCODE runSuite4( void);
	
	RCODE runSuite5( void);
};

/****************************************************************************
Desc:
****************************************************************************/
RCODE getTest( 
	IFlmTest **		ppTest)
{
	RCODE		rc = NE_XFLM_OK;

	if( (*ppTest = f_new IXPATHTest2Impl) == NULL)
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
const char * IXPATHTest2Impl::getName( void)
{
	return( "XPATH Test 2");
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IXPATHTest2Impl::execute( void)
{
	RCODE					rc = NE_XFLM_OK;

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
RCODE IXPATHTest2Impl::runSuite1( void)
{
	RCODE					rc = NE_XFLM_OK;
	FLMBOOL				bDibCreated = FALSE;
	IF_PosIStream *	pPosIStream = NULL;
	IF_Query *			pQuery = NULL;
	IF_DOMNode *		pResult = NULL;
	IF_DOMNode *		pAttr = NULL;
	IF_DOMNode *		pTempNode = NULL;
	char 					szQueryString[256];
	FLMUINT64			ui64ParentId = 0;
	FLMUINT64			ui64ParentId2 = 0;
	FLMUINT64			ui64DocumentId = 0;
	FLMUINT64			ui64DocumentId2 = 0;
	FLMUINT64			ui64NodeId = 0;
	FLMUINT64			ui64FirstChildId = 0;
	FLMUINT64			ui64FirstChildId2 = 0;
	FLMUINT64			ui64LastChildId = 0;
	FLMUINT64			ui64LastChildId2 = 0;
	FLMUINT64			ui64NextSiblingId = 0;
	FLMUINT64			ui64NextSiblingId2 = 0;
	FLMUINT64			ui64PrevSiblingId = 0;
	FLMUINT64			ui64PrevSiblingId2 = 0;
	FLMUINT64			ui64Tmp;
	const char *		ppszQueryThreeResults[] = 
								{"John Thorson","Marea Angela Castaneda"};
	const char *		ppszFollowingQueryResults[] = 
								{"Africa", "Asia", "Europe"};
	const char *		ppszDescendantQueryResults[] = 
								{"Sales","Marketing","Production"};
	const char *		ppszAncestorQueryResults[] =
								{"emp9032", "emp7216", "emp4238", "emp3456"};
	const char *		ppszQuery19Results[] = 
								{"Sales","", "", ""};
	char					szBuffer[ 128];
	const char *		pszQueryResult = NULL;
	FLMBOOL				bTransStarted = FALSE;

	const char *		pszOrgChart = 
		"<chairman empID=\"emp3456\" empdate=\"1976-03-21\">"
		"<name>Kim Akers</name>"
		" <president empID=\"emp4390\" empdate=\"1980-06-18\">"
		"     <name>Steve Masters</name>"
		"     <division>Domestic</division>"
		"     <groupvp empID=\"emp9801\" empdate=\"1981-09-01\">"
		"         <name>Shelly Szymanski</name>"
		"         <department>Sales</department>"
		"         <director empID=\"emp5443\" empdate=\"1993-04-19\">"
		"             <name>Cindy Durkin</name>"
		"             <region>Northeast</region>"
		"         </director>"
		"         <director empID=\"emp2348\" empdate=\"1990-06-02\">"
		"             <name>Michelle Votava</name>"
		"             <region>Southeast</region>"
		"         </director>"
		"         <director empID=\"emp4322\" empdate=\"1995-05-26\">"
		"             <name>John Tippett</name>"
		"             <region>Southwest</region>"
		"         </director>"
		"         <director empID=\"emp4587\" empdate=\"1990-12-20\">"
		"             <name>Alan Steiner</name>"
		"             <region>Northwest</region>"
		"         </director>"
		"     </groupvp>"
		"     <groupvp empID=\"emp4320\" empdate=\"1978-11-09\">"
		"         <name>John Thorson</name>"
		"         <department>Marketing</department>"
		"     </groupvp>"
		"     <groupvp empID=\"emp3961\" empdate=\"1990-01-01\">"
		"         <name>Josh Barnhill</name>"
		"         <department>Production</department>"
		"     </groupvp>"
		" </president>"
		" <president empID=\"emp4238\" empdate=\"1984-08-12\">"
		"     <name>Katie McAskill-White</name>"
		"     <division>International</division>"
		"     <groupvp empID=\"emp7216\" empdate=\"1987-04-13\">"
		"         <name>Neil Charney</name>"
		"         <department>Sales</department>"
		"         <director empID=\"emp9032\" empdate=\"1982-08-31\">"
		"             <name>Beth Silverberg</name>"
		"             <region>Africa</region>"
		"         </director>"
		"         <director empID=\"emp7634\" empdate=\"1994-02-12\">"
		"             <name>Lani Ota</name>"
		"             <region>Asia</region>"
		"         </director>"
		"         <director empID=\"emp9032\" empdate=\"1991-10-03\">"
		"             <name>Peter Porzuczek</name>"
		"             <region>Europe</region>"
		"         </director>"
		"     </groupvp>"
		"     <groupvp empID=\"emp3833\" empdate=\"1991-05-15\">"
		"         <name>Marea Angela Castaneda</name>"
		"         <department>Marketing</department>"
		"     </groupvp>"
		" </president>"
		"</chairman>";

	const char *		pszNameRegionIndex = 
		"<xflaim:Index "
		"	xmlns:xflaim=\"http://www.novell.com/XMLDatabase/Schema\""
		"	xflaim:name=\"name+region\" "
		"	xflaim:DictNumber=\"99\"> " // HARD-CODED Dict Num for easy retrieval
		"	<xflaim:ElementComponent "
		"		xflaim:name=\"name\" "
		"		xflaim:KeyComponent=\"1\" "
		"		xflaim:IndexOn=\"value\"/> "
		"		xflaim:Required=\"1\" "
		"	<xflaim:ElementComponent "
		"		xflaim:name=\"region\" "
		"		xflaim:IndexOn=\"value\" "
		"		xflaim:KeyComponent=\"2\"/> "
		"		xflaim:Required=\"1\" "
		"</xflaim:Index> ";

	m_szDetails[0] = '\0';

	beginTest(
		"XPATH Test 2 Setup",
		"Set up test state for XPATH Query Test Suite #3",
		"",
		"");

	if( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		goto Exit;
	}
	bDibCreated = TRUE;

	
	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;

	if( RC_BAD( rc = importBuffer( pszOrgChart, XFLM_DATA_COLLECTION)))
	{
		 goto Exit;
	}

	if( RC_BAD( rc = importBuffer( pszNameRegionIndex, XFLM_DICT_COLLECTION)))
	{
		 goto Exit;
	}
	
	m_pDb->transCommit();
	bTransStarted = FALSE;

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_READ_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;


	if( RC_BAD( rc = m_pDbSystem->createIFQuery( &pQuery)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to create query object.", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

	/********************* First Query ************************************/
	f_strcpy( szQueryString, 
		"/chairman/president/groupvp[@empID == \"emp9801\"]/director[3]/name");
	pszQueryResult = "John Tippett";

	beginTest( 
		"Node Subscript Test 1",
		"/chairman/president/groupvp[@empID == \"emp9801\"]/director[3]/name",
		"",
		"");

	if ( RC_BAD( rc = doQueryTest( 
		szQueryString,
		&pszQueryResult, 
		  1, 
		pQuery,
		m_szDetails)))
	{
		goto Exit;
	}

	endTest("PASS");

	/********************* Second Query ************************************/
	f_strcpy( szQueryString, "/chairman/president[2]/groupvp[1]/director[3]/name");
	pszQueryResult = "Peter Porzuczek";

	beginTest( 
		"Node Subscript Test 2",
		"/chairman/president[2]/groupvp[1]/director[3]/name",
		"",
		"");

	if ( RC_BAD( rc = doQueryTest( 
		szQueryString,
		&pszQueryResult, 
		  1, 
		pQuery,
		m_szDetails)))
	{
		goto Exit;
	}

	endTest("PASS");

	/********************* Third Query ************************************/
	f_strcpy( szQueryString, "/chairman/president/groupvp[4 - 2]/name");

	beginTest( 
		"Node Subscript Test 2",
		"/chairman/president/groupvp[4 - 2]/name",
		"",
		"");

	if ( RC_BAD( rc = doQueryTest( 
		szQueryString,
		ppszQueryThreeResults, 
		sizeof( ppszQueryThreeResults) / sizeof( ppszQueryThreeResults[0]), 
		pQuery,
		m_szDetails)))
	{
		goto Exit;
	}

	endTest("PASS");

	/********************* Fourth Query ************************************/
	f_strcpy( szQueryString, 
		"/groupvp[ @empID == \"emp7216\"]/director[2 * 3 - 4]/region");
	pszQueryResult = "Asia";

	beginTest( 
		"Node Subscript Test 4",
		"/groupvp[ @empID == \"emp7216\"]/director[2 * 3 - 4]/region",
		"",
		"");
	
	if ( RC_BAD( rc = doQueryTest( 
		szQueryString,
		&pszQueryResult, 
		  1, 
		pQuery,
		m_szDetails)))
	{
		goto Exit;
	}

	endTest("PASS");

	/********************* Fifth Query ************************************/
	beginTest( 
		"Meta Axis Pretest #1",
		"Query for necessary values to run the Meta Axis tests.",
		"Get a node and save its nodeId parentId prevSibId and docId "
		"in preparation for the first meta axis test.",
		"");

	f_strcpy( szQueryString, "/chairman/president[@empID == \"emp4238\"]"
						 "/groupvp/director[@empID == \"emp7634\"]/region[. == \"Asia\"]");

	if ( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, szQueryString)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to set up query expression.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pQuery->getFirst( m_pDb, &pResult)))
	{
		MAKE_FLM_ERROR_STRING( "getFirst failed.", m_szDetails, rc);
		goto Exit;
	}

	// Save the node id
	
	if( RC_BAD( rc = pResult->getNodeId( m_pDb, &ui64NodeId)))
	{
		goto Exit;
	}

	// Save its parent id
	if( RC_BAD( rc = pResult->getParentId( m_pDb, &ui64ParentId)))
	{
		MAKE_FLM_ERROR_STRING( "getParentId failed.", m_szDetails, rc);
		goto Exit;
	}

	// Save its prev sibling id
	if ( RC_BAD( rc = pResult->getPrevSibId( m_pDb, &ui64PrevSiblingId)))
	{
		MAKE_FLM_ERROR_STRING( "getPrevSibId failed.", m_szDetails, rc);
		goto Exit;
	}

	//Save the documentid
	if ( RC_BAD( rc = pResult->getDocumentId(m_pDb, &ui64DocumentId)))
	{
		MAKE_FLM_ERROR_STRING( "getDocumentId failed.", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");
	
	// Generate a new query to validate the meta::parentid axis
	
	f_sprintf( szQueryString, 
				"/chairman/president/groupvp/director/region[meta::parentid == %I64u]", 
				ui64ParentId);

	/********************* Sixth Query ************************************/
	beginTest( 
		"Meta Axis #1",
		szQueryString,
		"Do a meta::parentid query",
		"");

	if ( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, szQueryString)))
	{
		MAKE_FLM_ERROR_STRING( "setupQueryExpr failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pQuery->getFirst( m_pDb, &pResult)))
	{
		MAKE_FLM_ERROR_STRING( "getFirst failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pResult->getParentId( m_pDb, &ui64ParentId2)))
	{
		MAKE_FLM_ERROR_STRING( "getParentId failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( ui64ParentId2 != ui64ParentId)
	{
		rc = NE_XFLM_FAILURE;
		MAKE_FLM_ERROR_STRING( "meta::parentid query returned wrong node.", 
			m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

	/********************* Seventh Query ************************************/

	// Generate a new query to validate the meta::nodeid axis
	
	f_sprintf( szQueryString, "/chairman/president//director/region[meta::nodeid == %I64u]", ui64NodeId);

	beginTest( 
		"Meta Axis #2",
		szQueryString,
		"Do a meta::nodeid query",
		"");

	if ( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, szQueryString)))
	{
		MAKE_FLM_ERROR_STRING( "setupQueryExpr failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pQuery->getFirst( m_pDb, &pResult)))
	{
		MAKE_FLM_ERROR_STRING( "getFirst failed.", m_szDetails, rc);
		goto Exit;
	}
	
	if( RC_BAD( rc = pResult->getNodeId( m_pDb, &ui64Tmp)))
	{
		goto Exit;
	}

	if( ui64Tmp != ui64NodeId)
	{
		rc = NE_XFLM_FAILURE;
		MAKE_FLM_ERROR_STRING( "meta::nodeid query returned wrong node.", 
			m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

	/********************* Eighth Query ************************************/

	// Generate a new query to validate the meta::prevsiblingid axis
	
	f_sprintf( szQueryString, "///region[meta::prevsiblingid == %I64u]", ui64PrevSiblingId);

	beginTest( 
		"Meta Axis #3",
		szQueryString,
		"Do a meta::prevsiblingid query",
		"");

	if ( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, szQueryString)))
	{
		MAKE_FLM_ERROR_STRING( "setupQueryExpr failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pQuery->getFirst( m_pDb, &pResult)))
	{
		MAKE_FLM_ERROR_STRING( "getFirst failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pResult->getPrevSibId( m_pDb, &ui64PrevSiblingId2)))
	{
		MAKE_FLM_ERROR_STRING( "getPrevSibId failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( ui64PrevSiblingId2 != ui64PrevSiblingId)
	{
		rc = NE_XFLM_FAILURE;
		MAKE_FLM_ERROR_STRING( "meta::prevsiblingid query returned wrong node.", 
			m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

	/********************* Ninth Query ************************************/
	// Generate a new query to validate the meta::documentid

	f_sprintf( szQueryString, "///region[meta::documentid == %I64u]", ui64DocumentId);

	beginTest(
		"Meta Axis #9",
		szQueryString,
		"Do a meta::documentid query",
		"");
	
	if ( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, szQueryString)))
	{
		MAKE_FLM_ERROR_STRING( "setupQueryExpr failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pQuery->getFirst( m_pDb, &pResult)))
	{
		MAKE_FLM_ERROR_STRING( "getFirst failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pResult->getDocumentId( m_pDb, &ui64DocumentId2)))
	{
		MAKE_FLM_ERROR_STRING( "getDocumentId failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( ui64DocumentId2 != ui64DocumentId)
	{
		rc = NE_XFLM_FAILURE;
		MAKE_FLM_ERROR_STRING( "meta::documentid query returned wrong node.", 
			m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

	/********************* Tenth Query ************************************/
	f_strcpy( szQueryString, "///director[@empID ==\"emp5443\"]");

	beginTest(
		"Meta Axis Pretest #2",
		"Query for necessary values to run the rest of the Meta Axis"
		" tests.",
		"Get a node and save its firstchildId lastchildId nextSibId"
		" firstAttrId and lastAttrId in preparation for the first meta axis"
		" test.",
		"");

	if ( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, szQueryString)))
	{
		MAKE_FLM_ERROR_STRING( "setupQueryExpr failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pQuery->getFirst( m_pDb, &pResult)))
	{
		MAKE_FLM_ERROR_STRING( "getFirst failed.", m_szDetails, rc);
		goto Exit;
	}

	//Save the first child
	if ( RC_BAD( rc = pResult->getFirstChildId( m_pDb, &ui64FirstChildId)))
	{
		MAKE_FLM_ERROR_STRING( "getFirstChildId failed.", m_szDetails, rc);
		goto Exit;
	}

	//Save the last child

	if ( RC_BAD( rc = pResult->getLastChildId( m_pDb, &ui64LastChildId)))
	{
		MAKE_FLM_ERROR_STRING( "getLastChildId failed.", m_szDetails, rc);
		goto Exit;
	}

	//Save the next sibling
	if ( RC_BAD( rc = pResult->getNextSibling( m_pDb, &pTempNode)))
	{
		MAKE_FLM_ERROR_STRING( "getNextSibling failed.", m_szDetails, rc);
		goto Exit;
	}
	
	if( RC_BAD( rc = pTempNode->getNodeId( m_pDb, &ui64NextSiblingId)))
	{
		goto Exit;
	}

	endTest("PASS");

	/********************* Eleventh Query ************************************/
	// Generate a new query to validate the meta::firstchildid
	
	f_sprintf( szQueryString, "///director[meta::firstchildid == %I64u]", ui64FirstChildId);

	beginTest( "Meta Axis #3",
		szQueryString,
		"Do a meta::firstchildid query",
		"");

	if ( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, szQueryString)))
	{
		MAKE_FLM_ERROR_STRING( "setupQueryExpr failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pQuery->getFirst( m_pDb, &pResult)))
	{
		MAKE_FLM_ERROR_STRING( "getFirst failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pResult->getFirstChildId( m_pDb, &ui64FirstChildId2)))
	{
		MAKE_FLM_ERROR_STRING( "getFirstChildId failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( ui64FirstChildId != ui64FirstChildId2)
	{
		rc = NE_XFLM_FAILURE;
		MAKE_FLM_ERROR_STRING( "meta::firstchildid query returned wrong node.", 
			m_szDetails, rc);		
		goto Exit;
	}

	endTest("PASS");

	/********************* Twelfth Query ************************************/
	// Generate a new query to validate the meta::lastchildid

	f_sprintf( szQueryString, "///director[meta::lastchildid == %I64u]", ui64LastChildId);

	beginTest( "Meta Axis #4",
		szQueryString,
		"Do a meta::lastchildid query",
		"");

	if ( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, szQueryString)))
	{
		MAKE_FLM_ERROR_STRING( "setupQueryExpr failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pQuery->getFirst( m_pDb, &pResult)))
	{
		MAKE_FLM_ERROR_STRING( "getFirst failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pResult->getLastChildId( m_pDb, &ui64LastChildId2)))
	{
		MAKE_FLM_ERROR_STRING( "getLastChildId failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( ui64LastChildId != ui64LastChildId2)
	{
		rc = NE_XFLM_FAILURE;
		MAKE_FLM_ERROR_STRING( "meta::lastchildid query returned wrong node.", 
			m_szDetails, rc);		
		goto Exit;
	}

	endTest("PASS");

	/********************* Thirteenth Query ************************************/
	// Generate a new query to validate the meta::nextsiblingid

	f_sprintf( szQueryString, "///director[meta::nextsiblingid == %I64u]", ui64NextSiblingId);

	beginTest("Meta Axis #5", szQueryString, "Do a meta::nextsiblingid query", "");

	if ( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, szQueryString)))
	{
		MAKE_FLM_ERROR_STRING( "setupQueryExpr failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pQuery->getFirst( m_pDb, &pResult)))
	{
		MAKE_FLM_ERROR_STRING( "getFirst failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pResult->getNextSibling( m_pDb, &pTempNode)))
	{
		MAKE_FLM_ERROR_STRING( "getNextSibling failed.", m_szDetails, rc);
		goto Exit;
	}
	
	if( RC_BAD( rc = pTempNode->getNodeId( m_pDb, &ui64NextSiblingId2)))
	{
		goto Exit;
	}

	if( ui64NextSiblingId != ui64NextSiblingId2)
	{
		rc = NE_XFLM_FAILURE;
		MAKE_FLM_ERROR_STRING( "meta::nextsiblingid query returned wrong node.", 
			m_szDetails, rc);	
		goto Exit;
	}

	endTest("PASS");


	/********************* Seventeenth Query ************************************/
	f_sprintf( szQueryString, 
		"name[. == \"John Thorson\"]/parent::*/[@empID==\"emp4320\"]/attribute::empdate");

	beginTest( "Parent and Attribute Axes Test", szQueryString, "Do a parent axis query", "");
	pszQueryResult =  "1978-11-09";

	if ( RC_BAD( rc = doQueryTest( 
		szQueryString,
		&pszQueryResult, 
		  1, 
		pQuery,
		m_szDetails)))
	{
		goto Exit;
	}

	endTest("PASS");

	/********************* Eighteenth Query ************************************/
	f_sprintf( szQueryString, 
		"division[. == \"Domestic\"]/preceding::name");

	beginTest( "Previous Axis Test", szQueryString, "Do a previous axis query", ""); 

	if ( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, szQueryString)))
	{
		MAKE_FLM_ERROR_STRING( "setupQueryExpr failed.", m_szDetails, rc);
		goto Exit;
	}
	if (RC_BAD( rc = pQuery->setIndex( 0)))
	{
		MAKE_FLM_ERROR_STRING( "setIndex failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pQuery->getLast( m_pDb, &pResult)))
	{
		MAKE_FLM_ERROR_STRING( "getLast failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pResult->getUTF8( m_pDb, (FLMBYTE *)szBuffer,
		sizeof( szBuffer), 0, sizeof( szBuffer) - 1)))
	{
		MAKE_FLM_ERROR_STRING( "getUTF8 failed.", m_szDetails, rc);
		goto Exit;
	} 

	// results are returned in order moving backward from the current node.
	// Therefore, Kim Akers, who appears first in the document is actually
	// last in our query results.

	if ( f_strcmp( szBuffer, "Kim Akers") != 0)
	{
		rc = NE_XFLM_FAILURE;
		MAKE_FLM_ERROR_STRING( "getUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pQuery->getPrev( m_pDb, &pResult)))
	{
		MAKE_FLM_ERROR_STRING( "getPrevious failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pResult->getUTF8( m_pDb, (FLMBYTE *)szBuffer,
		sizeof( szBuffer), 0, sizeof( szBuffer) - 1)))
	{
		MAKE_FLM_ERROR_STRING( "getUTF8 failed.", m_szDetails, rc);
		goto Exit;
	} 

	if ( f_strcmp( szBuffer, "Steve Masters") != 0)
	{
		rc = NE_XFLM_FAILURE;
		MAKE_FLM_ERROR_STRING( "getUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

	/********************* Nineteenth Query ************************************/
	f_sprintf( szQueryString, 
		"name[.==\"Neil Charney\"]/following-sibling::*");

	beginTest( "Following-Sibling Test", szQueryString, 
		"Do a parent axis query", "");

	if ( RC_BAD( rc = doQueryTest( 
		szQueryString,
		ppszQuery19Results, 4,
		pQuery,
		m_szDetails)))
	{
		goto Exit;
	}

	endTest("PASS");

	/********************* Twentieth Query ************************************/
	f_sprintf( szQueryString, 
		"name[.==\"Neil Charney\"]/following::region");

	beginTest("Following Axis Test", szQueryString, "Do a following axis query", "");

	if ( RC_BAD( rc = doQueryTest( 
		szQueryString,
		ppszFollowingQueryResults, 
		sizeof( ppszFollowingQueryResults) / sizeof( ppszFollowingQueryResults[0]), 
		pQuery,
		m_szDetails)))
	{
		goto Exit;
	}

	endTest("PASS");

	/********************* Twenty-first Query ************************************/
	f_sprintf( szQueryString, 
		"president[@empID==\"emp4390\"]/descendant::department");

	beginTest( "Descendant Axis Test", szQueryString, "Do a descendant axis query", "");

	if ( RC_BAD( rc = doQueryTest( 
		szQueryString,
		ppszDescendantQueryResults, 
		sizeof( ppszDescendantQueryResults) / sizeof( ppszDescendantQueryResults[0]), 
		pQuery,
		m_szDetails)))
	{
		goto Exit;
	}

	endTest("PASS");

	/********************* Twenty-second Query ************************************/
	f_sprintf( szQueryString, 
		"region[.==\"Europe\"]/preceding-sibling::*");

	pszQueryResult = "Peter Porzuczek";

	beginTest( "Preceding-sibling Axis Test", szQueryString, 
		"Do a preceding-sibling axis query", "");

	if ( RC_BAD( rc = doQueryTest( 
		szQueryString,
		&pszQueryResult, 
		  1, 
		pQuery,
		m_szDetails)))
	{
		goto Exit;
	}

	endTest("PASS");

	/********************* Twenty-third Query ************************************/
	f_sprintf( szQueryString, 
		"name[.==\"Beth Silverberg\"]/ancestor::*/attribute::empID");

	beginTest( "Ancestor Axis Test", szQueryString, "Do a Ancestor axis query", "");

	if ( RC_BAD( rc = doQueryTest( 
		szQueryString,
		ppszAncestorQueryResults, 
		sizeof( ppszAncestorQueryResults) / sizeof( ppszAncestorQueryResults[0]), 
		pQuery,
		m_szDetails)))
	{
		goto Exit;
	}

	endTest("PASS");

	/********************* Twenty-fourth Query ************************************/
	f_sprintf( szQueryString, 
		"name[.==\"Beth Silverberg\"]/preceding::president/name");

	pszQueryResult = "Steve Masters";

	beginTest( "Preceding Axis Test", szQueryString, "Do a Preceding axis query", "");

	if ( RC_BAD( rc = doQueryTest( 
		szQueryString,
		&pszQueryResult, 
		1, 
		pQuery,
		m_szDetails)))
	{
		goto Exit;
	}

	endTest("PASS");

	/*********************** Twenty-fifth Query ********************************/
	f_sprintf( szQueryString, "true() and ///chairman/president/name[.== \"Steve Masters\"]"
		"/meta::documentid[.==%u]", (unsigned)ui64DocumentId);

	pszQueryResult = "Steve Masters";

	beginTest( "TRUE Value Defect Test #1", szQueryString, 
		"Verify defect involving TRUE values has been fixed.", "");

	if ( RC_BAD( rc = doQueryTest( 
		szQueryString,
		&pszQueryResult, 
		1, 
		pQuery,
		m_szDetails)))
	{
		goto Exit;
	}

	endTest("PASS");

	/*********************** Twenty-sixth Query ********************************/
	f_sprintf( szQueryString, 
		"(///groupvp[@empID==\"emp9801\" and @empdate==\"1981-09-01\"]"
		"/director[@empID==\"emp2348\" or @empdate==\"1995-05-26\"]"
		"/name[.==\"Michelle Votava\" or .==\"John Tippett\"]==\"John Tippett\""
		" and true()) or ///region[.==\"NorthWest\"]/preceding-sibling::*");

	beginTest( "TRUE Value Defect Test #2", szQueryString, 
		"Verify defect involving TRUE values has been fixed.", "");

	if ( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, szQueryString)))
	{
		MAKE_FLM_ERROR_STRING( "setupQueryExpr failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pQuery->getFirst( m_pDb, &pResult)))
	{
		MAKE_FLM_ERROR_STRING( "getFirst failed.", m_szDetails, rc);
		goto Exit;
	}
	
	if( RC_BAD( rc = pResult->getNodeId( m_pDb, &ui64Tmp)))
	{
		goto Exit;
	}

	if( ui64Tmp != ui64DocumentId)
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		MAKE_FLM_ERROR_STRING( "Unexpected query result.", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

	/*********************** Twenty-seventh Query ********************************/

	f_sprintf( szQueryString, 
		"name[.==\"Marea Angela Castaneda\"]/following::department[preceding-sibling::*]");

	pszQueryResult = "Marketing";

	beginTest( "Wildcard in predicate after step", szQueryString, 
		"BUG - Verify that contexts are created properly.", "");

	if ( RC_BAD( rc = doQueryTest( 
		szQueryString,
		&pszQueryResult, 
		1, 
		pQuery,
		m_szDetails)))
	{
		goto Exit;
	}

	endTest("PASS");

Exit:

	if ( RC_BAD( rc))
	{
		endTest("FAIL");
	}

	if ( pPosIStream)
	{
		pPosIStream->Release();
	}

	if ( pQuery)
	{
		pQuery->Release();
	}

	if ( pResult)
	{
		pResult->Release();
	}

	if ( pTempNode)
	{
		pTempNode->Release();
	}

	if ( pAttr)
	{
		pAttr->Release();
	}

	if ( bTransStarted)
	{
		m_pDb->transCommit();
	}

	shutdownTestState( DB_NAME_STR, bDibCreated);
	return rc;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IXPATHTest2Impl::runSuite2( void)
{
	RCODE					rc = NE_XFLM_OK;
	FLMBOOL				bDibCreated = FALSE;
	FLMBOOL				bTransStarted = FALSE;
	IF_PosIStream *	pPosIStream = NULL;
	IF_DOMNode *		pResult = NULL;
	IF_Query *			pQuery = NULL;
	char 					szQueryString[256];
	const char *		pszAlbum = 
		"<disc>"
			"<id>7005c60b</id>"
			"<length>1480</length>"
			"<title>Johnny Cash / The singing storyteller</title>"
			"<genre>cddb/country</genre>"
			"<track index=\"1\" offset=\"150\">Goodbye little darling</track>"
			"<track index=\"2\" offset=\"10262\">Give my love to Rose</track>"
			"<track index=\"3\" offset=\"22712\">Hey good looking</track>"
			"<track index=\"4\" offset=\"30450\">I can't help it</track>"
			"<track index=\"5\" offset=\"38437\">I could never be ashamed of you</track>"
			"<track index=\"6\" offset=\"48512\">I couldn't keep from crying</track>"
			"<track index=\"7\" offset=\"57687\">I love you because</track>"
			"<track index=\"8\" offset=\"68725\">The ways of a woman in love</track>"
			"<track index=\"9\" offset=\"78887\">You're the nearest thing to heaven</track>"
			"<track index=\"10\" offset=\"90962\">Come in, stranger</track>"
			"<track index=\"11\" offset=\"98625\">Next in line</track>"
			"</disc>";

	beginTest(
		"XPATH Defect Tests Setup",
		"Set up test state for XPATH Query Resolved Defect Test Suite",
		"",
		"");

	if ( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		goto Exit;
	}
	bDibCreated = TRUE;
	
	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;

	if ( RC_BAD( rc = importBuffer( pszAlbum, XFLM_DATA_COLLECTION)))
	{
		 goto Exit;
	}

	m_pDb->transCommit();
	bTransStarted = FALSE;

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_READ_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;


	if( RC_BAD( rc = m_pDbSystem->createIFQuery( &pQuery)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to create query object.", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

	/****************************Wildcard Defect Test***************************/

	f_sprintf( szQueryString, 
		"track[.==\"Yo*re the nearest thing*heaven\"]");

	beginTest( "Wildcard Defect Test", szQueryString, 
		"Verify wildcard defect has been fixed", "");

	{
		const char * ppszResults[] = { "You're the nearest thing to heaven"}; 

		if ( RC_BAD( rc = doQueryTest( 
			szQueryString,
			ppszResults, 
			sizeof( ppszResults) / sizeof( ppszResults[0]), 
			pQuery,
			m_szDetails)))
		{
			endTest("FAIL");
		}
		else
		{
			endTest("PASS");
		}
	}

	/****************************Double Period Defect Test**********************/

	f_sprintf( szQueryString, 
		"track[.==\"Hey good looking\"]/../track[@index==\"6\"]");

	beginTest( "Double Period Defect Test", szQueryString, 
		"Verify double period defect has been fixed", "");

	{
		const char * ppszResults[] = { "I couldn't keep from crying"}; 

		if ( RC_BAD( rc = doQueryTest( 
			szQueryString,
			ppszResults, 
			sizeof( ppszResults) / sizeof( ppszResults[0]), 
			pQuery,
			m_szDetails)))
		{
			endTest("FAIL");
		}
		else
		{
			endTest("PASS");
		}
	}

#if 0 // The following queries still do not work

	/****************************Anonymous Parent Axis Defect Test*****************/

	f_sprintf( szQueryString, 
		"//track[parent::*]");

	beginTest( "Anonymous parent axis defect test", szQueryString, 
		"Verify anonymous parent axis defect has been fixed. " 
		"Using the parent axis anonymously in a predicate "
		"will lead to asserts and access violations", "");

	{
		const char * ppszResults[] = 
		{
			"Goodbye little darling",
			"Give my love to Rose",
			"Hey good looking",
			"I can't help it",
			"I could never be ashamed of you",
			"I couldn't keep from crying",
			"I love you because",
			"The ways of a woman in love",
			"You're the nearest thing to heaven",
			"Come in, stranger",
			"Next in line"
		}; 

		if ( RC_BAD( rc = doQueryTest( 
			szQueryString,
			ppszResults, 
			sizeof( ppszResults) / sizeof( ppszResults[0]), 
			pQuery,
			m_szDetails)))
		{
			endTest("FAIL");
		}
		else
		{
			endTest("PASS");	
		}
	}

	/****************************Self Axis Defect Test**************************/

	f_sprintf( szQueryString, 
		"track[self::track==\"Come in, stranger\"]");

	beginTest( "Self Axis Defect Test", szQueryString, 
		"Verify self axis defect has been fixed", "");

	{
		const char * ppszResults[] = { "Come in, stranger"}; 

		if ( RC_BAD( rc = doQueryTest( 
			szQueryString,
			ppszResults, 
			sizeof( ppszResults) / sizeof( ppszResults[0]), 
			pQuery,
			m_szDetails)))
		{
			endTest("FAIL");
		}
		else
		{
			endTest("PASS");
		}
	}

	/****************************Subscript Defect Test***************************/
	// The XPATH specification does allow for multiple predicates per node. There
	// are examples of this in section 2.5 of the XPATH specification.
	// Usually this is not a problem because an equivalent query can be
	// built by and-ing the various expressions together in a single predicate.
	// However, this will not work if one of those expressions is a position.
	// The following query should select the first track node in the document
	// if it has an index attribute with a value equal to 1 (which it does).

	f_sprintf( szQueryString, 
		"track[1][@index==\"1\"]");

	beginTest( "Multiple subscript Defect Test", szQueryString, 
		"Verify multiple subscript defect has been fixed", "");

	{
		const char * ppszResults[] = { "Goodbye little darling"}; 

		if ( RC_BAD( rc = doQueryTest( 
			szQueryString,
			ppszResults, 
			sizeof( ppszResults) / sizeof( ppszResults[0]), 
			pQuery,
			m_szDetails)))
		{
			endTest("FAIL");
		}
		else
		{
			endTest("PASS");	
		}
	}

	/****************************Preceding wildcard Defect Test*****************/

	f_sprintf( szQueryString, 
		"descendant::track[preceding::*]");

	beginTest( "Preceding Wildcard Defect Test", szQueryString, 
		"Verify preceding wildcard defect has been fixed", "");

	{
		const char * ppszResults[] = 
		{
			"Goodbye little darling",
			"Give my love to Rose",
			"Hey good looking",
			"I can't help it",
			"I could never be ashamed of you",
			"I couldn't keep from crying",
			"I love you because",
			"The ways of a woman in love",
			"You're the nearest thing to heaven",
			"Come in, stranger",
			"Next in line"
		}; 

		if ( RC_BAD( rc = doQueryTest( 
			szQueryString,
			ppszResults, 
			sizeof( ppszResults) / sizeof( ppszResults[0]), 
			pQuery,
			m_szDetails)))
		{
			endTest("FAIL");
		}
		else
		{
			endTest("PASS");	
		}
	}

	/****************Descendant or Self Wildcard Defect Test**********************/
	f_sprintf( szQueryString, 
		"//descendant-or-self::*[self::id!=\"gibberish\" or self::id==\"7005c60b\"]");

	beginTest( "Descendant or Self Wildcard Defect Test", szQueryString, 
		"Verify descendant or self wildcard defect has been fixed. "
		"The following query causes an assert in fquery.cpp", "");

	{
		const char * ppszResults[] = { "7005c60b"}; 

		if ( RC_BAD( rc = doQueryTest( 
			szQueryString,
			ppszResults, 
			sizeof( ppszResults) / sizeof( ppszResults[0]), 
			pQuery,
			m_szDetails)))
		{
			endTest("FAIL");
		}
		else
		{
			endTest("PASS");
		}
	}

	/****************Descendant or Self Assert Defect Test**********************/

	f_sprintf( szQueryString, 
		"descendant-or-self::track[.==\"Hey good looking\" or "
		"preceding-sibling::track==\"Give my love to Rose\"]/@offset[preceding::id!=\"gibberish\"]");

	beginTest( "Descendant or Self Wildcard Defect Test", szQueryString, 
		"Verify descendant or self assert defect has been fixed. "
		"The following query causes an assert in fdom.cpp", "");

	{
		const char * ppszResults[] = { "22712"}; 

		if ( RC_BAD( rc = doQueryTest( 
			szQueryString,
			ppszResults, 
			sizeof( ppszResults) / sizeof( ppszResults[0]), 
			pQuery,
			m_szDetails)))
		{
			endTest("FAIL");
		}
		else
		{
			endTest("PASS");	
		}
	}
#endif

Exit:

	if ( pPosIStream)
	{
		pPosIStream->Release();
	}

	if ( pQuery)
	{
		pQuery->Release();
	}

	if ( pResult)
	{
		pResult->Release();
	}

	if ( bTransStarted)
	{
		m_pDb->transCommit();
	}

	shutdownTestState( DB_NAME_STR, bDibCreated);
	return rc;
}

#define IX_DICT_NUM		123
#define NUM_DOCS 			5

/****************************************************************************
Desc:
****************************************************************************/
RCODE IXPATHTest2Impl::runSuite3( void)
{
	RCODE				rc = NE_XFLM_OK;
	char				szIndex[400];
	IF_Query *		pQuery = NULL;
	IF_DOMNode *	pNode = NULL;
	FLMUINT64		ui64DocId = 0;
	IF_DOMNode *	pDoc = NULL;
	FLMUINT			uiLoop;
	FLMBOOL			bTransActive = FALSE;
	FLMBOOL			bDibCreated = FALSE;
	const char *	pszIndexFormat = 
		"<xflaim:Index "
		"	xmlns:xflaim=\"http://www.novell.com/XMLDatabase/Schema\""
		"	xflaim:name=\"foo+bar\" "
		"	xflaim:DictNumber=\"%u\"> "
		"		<xflaim:ElementComponent"
		"			xflaim:name=\"foo\" "
		"			xflaim:IndexOn=\"presence\" "
		"			xflaim:KeyComponent=\"1\">"
		"			<xflaim:ElementComponent"
		"				xflaim:name=\"bar\" "
		"				xflaim:IndexOn=\"presence\" "
		"				xflaim:KeyComponent=\"2\" />"
		"		</xflaim:ElementComponent> "
		"</xflaim:Index> ";
	const char *	pszDoc =
		"<foo>"
		"	<bar/>"
		"</foo>";

	beginTest( 
		"Search/Delete Defect Test", 
		"Search and delete documents to ensure bug that was causing "
		"btree to not be setup properly has been fixed", 
		"Add some documents/index them/search using the index/"
		"occasionally delete a document while iterating",
		"");


	if ( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		goto Exit;
	}
	bDibCreated = TRUE;

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed", m_szDetails, rc);
		goto Exit;
	}
	bTransActive = TRUE;

	// create several documents that will be indexed

	for( uiLoop = 0; uiLoop < NUM_DOCS; uiLoop++)
	{
		if ( RC_BAD( rc = importBuffer( pszDoc, XFLM_DATA_COLLECTION)))
		{
			MAKE_FLM_ERROR_STRING( "importBuffer failed", m_szDetails, rc);
			goto Exit;
		}
	}

	f_sprintf( szIndex, pszIndexFormat, IX_DICT_NUM);

	// create an index

	if ( RC_BAD( rc = importBuffer( szIndex, XFLM_DICT_COLLECTION)))
	{
		MAKE_FLM_ERROR_STRING( "importBuffer failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_FLM_ERROR_STRING( "transCommit failed", m_szDetails, rc);
		goto Exit;
	}
	bTransActive = FALSE;

	// pose a query with the index

	if ( RC_BAD( rc = m_pDbSystem->createIFQuery( &pQuery)))
	{
		MAKE_FLM_ERROR_STRING( "createIFQuery failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pQuery->setIndex( IX_DICT_NUM)))
	{
		MAKE_FLM_ERROR_STRING( "setIndex failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, "bar")))
	{
		MAKE_FLM_ERROR_STRING( "setupQueryExpr failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed", m_szDetails, rc);
		goto Exit;
	}
	bTransActive = TRUE;

	// next, next

	for( uiLoop = 0; uiLoop < NUM_DOCS; uiLoop++)
	{
		if ( RC_BAD( rc = pQuery->getNext( m_pDb, &pNode)))
		{
			MAKE_FLM_ERROR_STRING( "getNext failed", m_szDetails, rc);
			goto Exit;
		}

		if ( ( uiLoop % 2) == 0)
		{
			if ( RC_BAD( rc = pNode->getDocumentId( m_pDb, &ui64DocId)))
			{
				MAKE_FLM_ERROR_STRING( "getDocumentId failed", m_szDetails, rc);
				goto Exit;
			}

			if ( RC_BAD( rc = m_pDb->getDocument( 
				XFLM_DATA_COLLECTION, 
				XFLM_EXACT, 
				ui64DocId,
				&pDoc)))
			{
				MAKE_FLM_ERROR_STRING( "getDocument failed", m_szDetails, rc);
				goto Exit;
			}

			if ( RC_BAD( rc = pDoc->deleteNode( m_pDb)))
			{
				MAKE_FLM_ERROR_STRING( "deleteNode failed", m_szDetails, rc);
				goto Exit;
			}

			// insert a new doc

			if ( RC_BAD( rc = importBuffer( pszDoc, XFLM_DATA_COLLECTION)))
			{
				MAKE_FLM_ERROR_STRING( "importBuffer failed", m_szDetails, rc);
				goto Exit;
			}
		}
	}

	endTest("PASS");

Exit:

	if (RC_BAD( rc))
	{
		endTest("FAIL");
	}

	if ( pQuery)
	{
		pQuery->Release();
	}

	if ( pNode)
	{
		pNode->Release();
	}

	if ( pDoc)
	{
		pDoc->Release();
	}

	if ( bTransActive)
	{
		if (RC_OK( rc))
		{
			m_pDb->transCommit();
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
RCODE IXPATHTest2Impl::runSuite4( void)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bTransActive = FALSE;
	char				szQueryString[100];
	IF_Query *		pQuery = NULL;
	IF_DOMNode *	pDoc = NULL;
	char *			pszResult = NULL;
	FLMUINT			uiResultAttr = 0;
	FLMBOOL			bFoundDesiredDoc = FALSE;
	const char *	pszDoc1 =
		"<foo Result=\"fail\">"
		"	<bar baz=\"1\">Wrong</bar>"
		"	<bar baz=\"1\">Wrong</bar>"
		"	<bar baz=\"1\">Wrong</bar>"
		"	<bar baz=\"1\">Wrong</bar>"
		"	<bar baz=\"1\">Wrong</bar>"
		"	<bar baz=\"1\">Wrong</bar>"
		"	<bar baz=\"1\">Wrong</bar>"
		"	<bar baz=\"1\">Wrong</bar>"
		"	<bar baz=\"1\">Wrong</bar>"
		"	<bar baz=\"1\">Wrong</bar>"
		"</foo>";
	const char *	pszDoc2 =
		"<foo result=\"pass\">"
		"	<bar baz=\"42\">Right</bar>"
		"	<bar baz=\"42\">Right</bar>"
		"	<bar baz=\"42\">Right</bar>"
		"	<bar baz=\"42\">Right</bar>"
		"</foo>";
	const char *	pszDoc3 =
		"<foo result=\"fail\">"
		"	<bar baz=\"42\">Right</bar>"
		"	<bar baz=\"42\">Right</bar>"
		"	<bar baz=\"42\">Right</bar>"
		"	<bar baz=\"42\">Right</bar>"
		"	<bar baz=\"1\">Wrong</bar>"
		"	<bar baz=\"1\">Wrong</bar>"
		"	<bar baz=\"1\">Wrong</bar>"
		"	<bar baz=\"1\">Wrong</bar>"
		"	<bar baz=\"1\">Wrong</bar>"
		"</foo>";

	FLMBOOL		bDibCreated = FALSE;

	f_sprintf( szQueryString, "!(/foo/bar[@baz==1]) && (/foo/bar[@baz==42])");
	beginTest( "Notted Optimization Defect Test", szQueryString, 
		"Verify a defect that was causing an improper optimization "
		"with notted nodes has been fixed", 
		"");

	if ( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		goto Exit;
	}
	bDibCreated = TRUE;

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed", m_szDetails, rc);
		goto Exit;
	}
	bTransActive = TRUE;

	// make sure baz attribute is imported as a number

	if ( RC_BAD( rc = m_pDb->createAttributeDef(
		NULL,
		"baz",
		XFLM_NUMBER_TYPE,
		NULL,
		NULL)))
	{
		MAKE_FLM_ERROR_STRING( "createAttributeDef failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->createAttributeDef(
		NULL,
		"result",
		XFLM_TEXT_TYPE,
		&uiResultAttr,
		NULL)))
	{
		MAKE_FLM_ERROR_STRING( "createAttributeDef failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = importBuffer( pszDoc1, XFLM_DATA_COLLECTION)))
	{
		MAKE_FLM_ERROR_STRING( "importBuffer failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = importBuffer( pszDoc2, XFLM_DATA_COLLECTION)))
	{
		MAKE_FLM_ERROR_STRING( "importBuffer failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = importBuffer( pszDoc3, XFLM_DATA_COLLECTION)))
	{
		MAKE_FLM_ERROR_STRING( "importBuffer failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDbSystem->createIFQuery( &pQuery)))
	{
		MAKE_FLM_ERROR_STRING( "createIFQuery failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, szQueryString)))
	{
		MAKE_FLM_ERROR_STRING( "setQueryExpr failed", m_szDetails, rc);
		goto Exit;
	}

	while ( RC_OK( rc = pQuery->getNext( m_pDb, &pDoc)))
	{
		if ( pszResult)
		{
			f_free( &pszResult);
		}

		if ( RC_BAD( rc = pDoc->getAttributeValueUTF8( m_pDb,
			uiResultAttr, (FLMBYTE **)&pszResult)))
		{
			MAKE_FLM_ERROR_STRING( "getAttributeValueUTF8 failed", 
				m_szDetails, rc);
			goto Exit;
		}

		if ( f_strcmp( pszResult, "pass") != 0)
		{
			rc = NE_XFLM_FAILURE;
			MAKE_FLM_ERROR_STRING( "unexpected document returned", 
				m_szDetails, rc);
			goto Exit;
		}
		bFoundDesiredDoc = TRUE;
	}

	if ( rc == NE_XFLM_EOF_HIT)
	{
		rc = NE_XFLM_OK;
	}
	else
	{
		goto Exit;
	}

	if ( !bFoundDesiredDoc)
	{
		rc = NE_XFLM_FAILURE;
		MAKE_FLM_ERROR_STRING( "expected document returned", 
			m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

Exit:

	if ( pQuery)
	{
		pQuery->Release();
	}

	if ( pDoc)
	{
		pDoc->Release();
	}

	if ( pszResult)
	{
		f_free( &pszResult);
	}

	if (RC_BAD( rc))
	{
		endTest("FAIL");
	}

	if ( bTransActive)
	{
		if (RC_OK( rc))
		{
			m_pDb->transCommit();
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
RCODE IXPATHTest2Impl::runSuite5( void)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bDibCreated = FALSE;
	FLMBOOL			bTransActive = FALSE;
	IF_Query *		pQuery = NULL;
	const char *	pszQueryString = NULL;
	FLMBOOL			bIndexAdded = FALSE;
	char				szTestName[ 100];
	const char *	pszDoc = 
		"<EscapedValues>"
		"	<Text>abc*</Text>"
		"	<Text>abcdef</Text>"
		"	<Text>abc\\</Text>"
		"	<Text>abcd</Text>"
		"</EscapedValues>";
	const char * pszIndex =
		"<xflaim:Index "
		"	xmlns:xflaim=\"http://www.novell.com/XMLDatabase/Schema\""
		"	xflaim:name=\"EscapedValues+Text\"> "
		"			<xflaim:ElementComponent"
		"				xflaim:name=\"Text\" "
		"				xflaim:IndexOn=\"value\" "
		"				xflaim:KeyComponent=\"1\" />"
		"</xflaim:Index> ";

	if ( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		goto Exit;
	}
	bDibCreated = TRUE;

	if ( RC_BAD( rc= m_pDbSystem->createIFQuery( &pQuery)))
	{
		MAKE_FLM_ERROR_STRING( "createIFQuery failed", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed", m_szDetails, rc);
		goto Exit;
	}
	bTransActive = TRUE;

	if (RC_BAD ( rc = importBuffer( pszDoc, XFLM_DATA_COLLECTION)))
	{
		goto Exit;
	}

Run_Again:

	pszQueryString = "///Text[.==\"abc\\*\"]";

	f_sprintf( 
		szTestName, 
		"Escaped wildcard test #1 (indexed=%u)",
		bIndexAdded);

	beginTest( szTestName, pszQueryString, 
		"Verify an escaped wildcard only matches a literal asterisk"
		"Self-explanatory", 
		"");

	{
		const char * ppszResults[] = { "abc*"}; 

		if ( RC_BAD( rc = doQueryTest( 
			pszQueryString,
			ppszResults, 
			sizeof( ppszResults) / sizeof( ppszResults[0]), 
			pQuery,
			m_szDetails)))
		{
			endTest("FAIL");
		}
		else
		{
			endTest("PASS");	
		}
	}

	pszQueryString = "///Text[.==\"abc*\"]";

	f_sprintf( 
		szTestName, 
		"Escaped wildcard test #2 (indexed=%u)",
		bIndexAdded);

	beginTest( szTestName, pszQueryString, 
		"Verify an non-escaped wildcard matches all"
		"Self-explanatory", 
		"");

	{
		const char * ppszResults[] = 
		{
			"abc*",
			"abcdef",
			"abc\\",
			"abcd"
		}; 

		if ( RC_BAD( rc = doQueryTest( 
			pszQueryString,
			ppszResults, 
			sizeof( ppszResults) / sizeof( ppszResults[0]), 
			pQuery,
			m_szDetails)))
		{
			endTest("FAIL");
		}
		else
		{
			endTest("PASS");	
		}
	}

	pszQueryString = "///Text[.==\"abc\\\\\"]";

	f_sprintf( 
		szTestName, 
		"Escaped backslash test #1 (indexed=%u)",
		bIndexAdded);

	beginTest( szTestName, pszQueryString, 
		"Verify an escaped backslash only matches a backslash"
		"Self-explanatory", 
		"");

	{

		const char * ppszResults[] = 
		{
			"abc\\"
		}; 

		if ( RC_BAD( rc = doQueryTest( 
			pszQueryString,
			ppszResults, 
			sizeof( ppszResults) / sizeof( ppszResults[0]), 
			pQuery,
			m_szDetails)))
		{
			endTest("FAIL");
		}
		else
		{
			endTest("PASS");	
		}
	}

	pszQueryString = "///Text[.==\"abc\\d\"]";

	f_sprintf( 
		szTestName, 
		"Escaped backslash test #2 (indexed=%u)",
		bIndexAdded);

	beginTest( szTestName, pszQueryString, 
		"Verify a non-escaped backslash is treated like an escape char"
		"Self-explanatory", 
		"");

	{
		const char * ppszResults[] = 
		{
			"abcd"
		}; 

		if ( RC_BAD( rc = doQueryTest( 
			pszQueryString,
			ppszResults, 
			sizeof( ppszResults) / sizeof( ppszResults[0]), 
			pQuery,
			m_szDetails)))
		{
			endTest("FAIL");
		}
		else
		{
			endTest("PASS");	
		}
	}

	if ( !bIndexAdded)
	{
		if ( RC_BAD( rc = importBuffer( pszIndex, XFLM_DICT_COLLECTION)))
		{
			goto Exit;
		}
		bIndexAdded = TRUE;
		goto Run_Again;
	}

Exit:

	if ( pQuery)
	{
		pQuery->Release();
	}

	if ( bTransActive)
	{
		if (RC_OK( rc))
		{
			m_pDb->transCommit();
		}
		else
		{
			m_pDb->transAbort();
		}
	}

	shutdownTestState( DB_NAME_STR, bDibCreated);
	return rc;
}
