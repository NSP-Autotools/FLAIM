//------------------------------------------------------------------------------
// Desc:	XPATH unit test 1
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

#define CD_NUM								1
#define TITLE_NUM							2
#define ARTIST_NUM						3
#define LOCATION_NUM						4
#define COMPANY_NUM						5
#define PRICE_NUM							6
#define YEAR_NUM							7
#define CATALOG_NUM						8
#define COUNTRY_NUM						9
#define CITY_NUM							10

/****************************************************************************
Desc:
****************************************************************************/
class IXPathTest1Impl : public TestBase
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

	if( (*ppTest = f_new IXPathTest1Impl) == NULL)
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
const char * IXPathTest1Impl::getName( void)
{
	return( "XPath Test 1");
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IXPathTest1Impl::execute( void)
{
	RCODE					rc = NE_XFLM_OK;
	FlagSet				flagSet;
	FLMBOOL				bTransStarted = FALSE;
	FLMBOOL				bDibCreated = FALSE;
	IF_PosIStream	*	pPosIStream = NULL;
	IF_Query *			pQuery = NULL;
	FLMUINT				uiDictNum = 0;
	IF_DOMNode *		pReturn = NULL;
	char					szBuffer[ 128];
	const char *		ppszQueryFourResults[] = 
								{"Hide your heart", "Still got the blues", 
								"One night only", "Sylvias Mother", "Maggie May"};
	const char *		ppszQueryFiveResults[] =
								{"Empire Burlesque", "Greatest Hits", 
								"Unchain my heart"};
	const char *		ppszQuerySixResults[] = 
								{"Gary Moore", "Rod Stewart"};
	const char *		ppszQuerySevenResults[] =
								{"Bob Dylan", "Bonnie Tyler", "Dolly Parton", 
								"Eros Ramazzotti", "Bee Gees", "Dr.Hook", 
								"Andrea Bocelli", "Joe Cocker"};
	const char *		ppszQueryEightResults[] =
								{"CBS Records", "RCA", "Virgin records", 
								"BMG", "CBS", "Pickwick", "EMI"};
	const char *		ppszQueryNineResults[] = 
								{"Still got the blues", "Eros", "One night only",
								"Maggie May", "Romanza"};
	const char *		pszDoc1 =
		"<catalog>"

		"<cd>"
		"<title>Empire Burlesque</title> "
		"<artist>Bob Dylan</artist> "
		"<location country=\"USA\" city=\"Las Angeles\"/> "
		"<company>Columbia</company> "
		"<price>1090</price> "
		"<year>1985</year> "
		"</cd>"

		"<cd>"
		"<title>Hide your heart</title> "
		"<artist>Bonnie Tyler</artist>" 
		"<location country=\"UK\" city=\"Liverpool\"/> "
		"<company>CBS Records</company> "
		"<price>990</price> "
		"<year>1988</year> "
		"</cd>"

		"<cd>"
		"<title>Greatest Hits</title> "
		"<artist>Dolly Parton</artist>" 
		"<location country=\"USA\" city=\"Hicksville\"/>"
		"<company>RCA</company> "
		"<price>990</price> "
		"<year>1982</year> "
		"</cd>"

		"<cd>"
		"<title>Still got the blues</title> "
		"<artist>Gary Moore</artist> "
		"<location country=\"UK\" city=\"London\"/> "
		"<company>Virgin records</company> "
		"<price>1020</price> "
		"<year>1990</year> "
		"</cd>"

		"<cd>"
		"<title>Eros</title> "
		"<artist>Eros Ramazzotti</artist>" 
		"<location country=\"EU\" city=\"Europa\"/> "
		"<company>BMG</company>" 
		"<price>990</price> "
		"<year>1997</year>" 
		"</cd>"

		"<cd>"
		"<title>One night only</title>" 
		"<artist>Bee Gees</artist>" 
		"<location country=\"UK\" city=\"Birmingham\"/> "
		"<company>Polydor</company>" 
		"<price>1090</price>" 
		"<year>1998</year>" 
		"</cd>"

		"<cd>"
		"<title>Sylvias Mother</title>" 
		"<artist>Dr.Hook</artist>" 
		"<location country=\"UK\" city=\"York\"/> "
		"<company>CBS</company>" 
		"<price>810</price>" 
		"<year>1973</year>" 
		"</cd>"

		"<cd>"
		"<title>Maggie May</title>"
		"<artist>Rod Stewart</artist>" 
		"<location country=\"UK\" city=\"London\"/> "
		"<company>Pickwick</company>" 
		"<price>850</price>" 
		"<year>1990</year>" 
		"</cd>"

		"<cd>"
		"<title>Romanza</title>" 
		"<artist>Andrea Bocelli</artist>" 
		"<location country=\"EU\" city=\"Europa\"/> "
		"<company>Polydor</company>" 
		"<price>1080</price>" 
		"<year>1996</year>" 
		"</cd>"

		"<cd>"
		"<title>Unchain my heart</title>" 
		"<artist>Joe Cocker</artist>" 
		"<location country=\"USA\" city=\"Seattle\"/> "
		"<company>EMI</company>" 
		"<price>820</price>" 
		"<year>1987</year>" 
		"</cd>"
		"</catalog>";

	beginTest(	
		"Database Setup",
		"Prep the database for the query tests",
		"1) get the class factory for the dbSystem "
		"2) Create the database 3) Setup element and attribute definitions "
		"4) Import a document 5) get a IF_Query object",
		"No Additional Details.");

	if( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to init test state.", m_szDetails, rc);
		goto Exit;
	}
	bDibCreated = TRUE;
	
	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;

	// Create element defs for easy lookups

	uiDictNum = CD_NUM;
	if( RC_BAD( rc = m_pDb->createElementDef(NULL, "cd",	XFLM_NODATA_TYPE, 
		&uiDictNum, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to create elementDef.", m_szDetails, rc);
		goto Exit;
	}
	
	if( uiDictNum != CD_NUM)
	{
		MAKE_FLM_ERROR_STRING( "Dict num already in use.", m_szDetails, rc);
		goto Exit;
	}

	uiDictNum = TITLE_NUM;
	if( RC_BAD( rc = m_pDb->createElementDef(NULL, "title",
		XFLM_TEXT_TYPE, &uiDictNum, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to create elementDef.", m_szDetails, rc);
		goto Exit;
	}
	
	if( uiDictNum != TITLE_NUM)
	{
		MAKE_FLM_ERROR_STRING( "Dict num already in use.", m_szDetails, rc);
		goto Exit;
	}

	uiDictNum = ARTIST_NUM;
	if( RC_BAD( rc = m_pDb->createElementDef(NULL, "artist",
		XFLM_TEXT_TYPE, &uiDictNum, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to create elementDef.", m_szDetails, rc);
		goto Exit;
	}
	
	if( uiDictNum != ARTIST_NUM)
	{
		MAKE_FLM_ERROR_STRING( "Dict num already in use.", m_szDetails, rc);
		goto Exit;
	}

	uiDictNum = LOCATION_NUM;
	if( RC_BAD( rc = m_pDb->createElementDef(NULL, "location",
		XFLM_TEXT_TYPE, &uiDictNum, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to create elementDef.", m_szDetails, rc);
		goto Exit;
	}
	
	if( uiDictNum != LOCATION_NUM)
	{
		MAKE_FLM_ERROR_STRING( "Dict num already in use.", m_szDetails, rc);
		goto Exit;
	}

	uiDictNum = COMPANY_NUM;
	if( RC_BAD( rc = m_pDb->createElementDef(NULL, "company", XFLM_TEXT_TYPE, 
		&uiDictNum, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to create elementDef.", m_szDetails, rc);
		goto Exit;
	}
	
	if( uiDictNum != COMPANY_NUM)
	{
		MAKE_FLM_ERROR_STRING( "Dict num already in use.", m_szDetails, rc);
		goto Exit;
	}

	uiDictNum = PRICE_NUM;
	if( RC_BAD( rc = m_pDb->createElementDef(NULL, "price",	XFLM_NUMBER_TYPE, 
		&uiDictNum, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to create elementDef.", m_szDetails, rc);
		goto Exit;
	}
	
	if( uiDictNum != PRICE_NUM)
	{
		MAKE_FLM_ERROR_STRING( "Dict num already in use.", m_szDetails, rc);
		goto Exit;
	}

	uiDictNum = YEAR_NUM;
	if( RC_BAD( rc = m_pDb->createElementDef(NULL, "year", XFLM_NUMBER_TYPE, 
		&uiDictNum, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to create elementDef.", m_szDetails, rc);
		goto Exit;
	}
	
	if( uiDictNum != YEAR_NUM)
	{
		MAKE_FLM_ERROR_STRING( "Dict num already in use.", m_szDetails, rc);
		goto Exit;
	}

	uiDictNum = CATALOG_NUM;
	if( RC_BAD( rc = m_pDb->createElementDef(NULL, "catalog", XFLM_NODATA_TYPE, 
		&uiDictNum, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to create elementDef.", m_szDetails, rc);
		goto Exit;
	}
	
	if( uiDictNum != CATALOG_NUM)
	{
		MAKE_FLM_ERROR_STRING( "Dict num already in use.", m_szDetails, rc);
		goto Exit;
	}

	uiDictNum = COUNTRY_NUM;
	if( RC_BAD( rc = m_pDb->createAttributeDef(NULL, "country", XFLM_TEXT_TYPE, 
		&uiDictNum, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to create attribute Def.", m_szDetails, rc);
		goto Exit;
	}
	
	if( uiDictNum != COUNTRY_NUM)
	{
		MAKE_FLM_ERROR_STRING( "Dict num already in use.", m_szDetails, rc);
		goto Exit;
	}

	uiDictNum = CITY_NUM;
	if( RC_BAD( rc = m_pDb->createAttributeDef(NULL, "city", XFLM_TEXT_TYPE, 
		&uiDictNum, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to create attribute Def.", m_szDetails, rc);
		goto Exit;
	}
	
	if( uiDictNum != CITY_NUM)
	{
		MAKE_FLM_ERROR_STRING( "Dict num already in use.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDbSystem->createIFQuery( &pQuery)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to create query object.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = importBuffer( pszDoc1, XFLM_DATA_COLLECTION)))
	{
		 goto Exit;
	}

	endTest("PASS");

	/********************* First Query ************************************/
	beginTest( 
		"Query Test #1",
		"/catalog/cd/title[. ~= \"Won Knight Only\"]",
		"Run the query and validate the results.",
		"");

	if( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, 
		"/catalog/cd/title[. ~= \"Won Knight Only\"]")))
	{
		MAKE_FLM_ERROR_STRING( "Failed to set up query expression.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pQuery->getFirst( m_pDb, &pReturn)))
	{
		MAKE_FLM_ERROR_STRING( "getFirst failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pReturn->getUTF8( m_pDb, (FLMBYTE *)szBuffer, 
		sizeof( szBuffer), 0, sizeof( szBuffer) - 1)))
	{
		MAKE_FLM_ERROR_STRING( "getUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if( f_strcmp( szBuffer, "One night only") != 0)
	{
		rc = NE_XFLM_FAILURE;
		MAKE_FLM_ERROR_STRING( "Unexpected query result.", m_szDetails, rc);
		goto Exit;
	}

	if( (rc = pQuery->getNext( m_pDb, &pReturn)) != NE_XFLM_EOF_HIT)
	{
		rc = NE_XFLM_FAILURE;
		MAKE_FLM_ERROR_STRING( "Unexpected query result.", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

	/********************* Second Query ************************************/

	beginTest( 	
		"Query Test #2",
		"/catalog/cd/price[. == 1080]",
		"Run the query and validate the results.",
		"");

	if( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, "/catalog/cd/price[. == 1080]")))
	{
		MAKE_FLM_ERROR_STRING( "Failed to set up query expression.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pQuery->getFirst( m_pDb, &pReturn)))
	{
		MAKE_FLM_ERROR_STRING( "getFirst failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pReturn->getUTF8( m_pDb, (FLMBYTE *)szBuffer,
		sizeof( szBuffer), 0, sizeof( szBuffer) - 1)))
	{
		MAKE_FLM_ERROR_STRING( "getUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if( f_strcmp( szBuffer, "1080") != 0)
	{
		rc = NE_XFLM_FAILURE;
		MAKE_FLM_ERROR_STRING( "Unexpected query result.", m_szDetails, rc);
		goto Exit;
	}

	if( (rc = pQuery->getNext( m_pDb, &pReturn)) != NE_XFLM_EOF_HIT)
	{
		rc = NE_XFLM_FAILURE;
		MAKE_FLM_ERROR_STRING( "Unexpected query result.", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

	/********************* Third Query ************************************/
	beginTest( 
		"Query Test #3",
		"/catalog/cd[price == 1080]/company",
		"Run the query and validate the results.",
		"");

	if( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, "/catalog/cd[price == 1080]/company")))
	{
		MAKE_FLM_ERROR_STRING( "Failed to set up query expression.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pQuery->getFirst( m_pDb, &pReturn)))
	{
		MAKE_FLM_ERROR_STRING( "getFirst failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pReturn->getUTF8( m_pDb, (FLMBYTE *)szBuffer, 
		sizeof( szBuffer), 0, sizeof( szBuffer) - 1)))
	{
		MAKE_FLM_ERROR_STRING( "getUTF8 failed.", m_szDetails, rc);
		goto Exit;
	}

	if( f_strcmp( szBuffer, "Polydor") != 0)
	{
		rc = NE_XFLM_FAILURE;
		MAKE_FLM_ERROR_STRING( "Unexpected query result.", m_szDetails, rc);
		goto Exit;
	}

	if( (rc = pQuery->getNext( m_pDb, &pReturn)) != NE_XFLM_EOF_HIT)
	{
		rc = NE_XFLM_FAILURE;
		MAKE_FLM_ERROR_STRING( "Unexpected query result.", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");
	
	/********************* Fourth Query ************************************/
	beginTest(
		"Query Test #4",
		"/catalog/cd[location[@country == \"UK\"]]/title",
		"Run the query and validate the results.",
		"");

	if( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, 
		"/catalog/cd[location[@country == \"UK\"]]/title")))
	{
		MAKE_FLM_ERROR_STRING( "Failed to set up query expression.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = checkQueryResults( 
		ppszQueryFourResults, 
		sizeof( ppszQueryFourResults) / sizeof( ppszQueryFourResults[0]), 
		pQuery,
		m_szDetails)))
	{
		goto Exit;
	}

	endTest("PASS");

	/********************* Fifth Query ************************************/

	beginTest( 
		"Query Test #5",
		"//cd[location[@country == \"USA\"]]/title",
		"Run the query and validate the results.",
		"");

	if( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, 
		"//cd[location[@country == \"USA\"]]/title")))
	{
		MAKE_FLM_ERROR_STRING( "Failed to set up query expression.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = checkQueryResults( 
		ppszQueryFiveResults, 
		sizeof( ppszQueryFiveResults) / sizeof( ppszQueryFiveResults[0]), 
		pQuery,
		m_szDetails)))
	{
		goto Exit;
	}

	endTest("PASS");

	/********************* Sixth Query ************************************/

	beginTest( 
		"Query Test #6",
		"//cd[location[@city == \"London\"]]/artist",
		"Run the query and validate the results.",
		"");

	if( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, 
		"//cd[location[@city == \"London\"]]/artist")))
	{
		MAKE_FLM_ERROR_STRING( "Failed to set up query expression.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = checkQueryResults( 
		ppszQuerySixResults, 
		sizeof( ppszQuerySixResults) / sizeof( ppszQuerySixResults[0]), 
		pQuery,
		m_szDetails)))
	{
		goto Exit;
	}

	endTest("PASS");

	/********************* Seventh Query ************************************/

	beginTest( 
		"Query Test #7",
		"//cd[location[@city != \"London\"]]/artist",
		"Run the query and validate the results.",
		"");


	if( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, 
		"//cd[location[@city != \"London\"]]/artist")))
	{
		MAKE_FLM_ERROR_STRING( "Failed to set up query expression.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = checkQueryResults( 
		ppszQuerySevenResults, 
		sizeof( ppszQuerySevenResults) / sizeof( ppszQuerySevenResults[0]), 
		pQuery,
		m_szDetails)))
	{
		goto Exit;
	}

	endTest("PASS");

	/********************* Eighth Query ************************************/

	beginTest( 
		"Query Test #8",
		"//cd[price < 1080]/company",
		"Run the query and validate the results.",
		"");


	if( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, 
		"//cd[price < 1080]/company")))
	{
		MAKE_FLM_ERROR_STRING( "Failed to set up query expression.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = checkQueryResults( 
		ppszQueryEightResults, 
		sizeof( ppszQueryEightResults) / sizeof( ppszQueryEightResults[0]), 
		pQuery,
		m_szDetails)))
	{
		goto Exit;
	}

	endTest("PASS");

	/********************* Ninth Query ************************************/

	beginTest( 
		"Query Test #9",
		"////cd[year >= 1990]/title",
		"Run the query and validate the results.",
		"");

	if( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, 
		"//cd[year >= 1990]/title")))
	{
		MAKE_FLM_ERROR_STRING( "Failed to set up query expression.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = checkQueryResults( 
		ppszQueryNineResults, 
		sizeof( ppszQueryNineResults) / sizeof( ppszQueryNineResults[0]), 
		pQuery,
		m_szDetails)))
	{
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
		m_pDb->transCommit();
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

	shutdownTestState( DB_NAME_STR, bDibCreated);
	return( rc);
}
