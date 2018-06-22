//------------------------------------------------------------------------------
// Desc:	Sort key unit test 2
// Tabs:	3
//
// Copyright (c) 2005-2006 Novell, Inc. All Rights Reserved.
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
	#define DB_NAME_STR			"SYS:\\SKEY2.DB"
#else
	#define DB_NAME_STR			"skey2.db"
#endif

#define FIRST_NAME_ID 			123
#define LAST_NAME_ID 			456

/****************************************************************************
Desc:
****************************************************************************/
class SortKeyTest2Impl : public TestBase
{
public:

	const char * getName( void)
	{
		return( "Sort Key Test 2");
	}
	
	RCODE execute( void);
	
private:

	RCODE createSortKey( 
		IF_Query * 	pQuery, 
		FLMBOOL 		bFirstAscending,
		FLMUINT 		uiFirstCompareRules,
		FLMUINT 		uiFirstLimit,
		FLMBOOL 		bFirstMissingHigh,
		FLMBOOL 		bLastAscending,
		FLMUINT 		uiLastCompareRules,
		FLMUINT 		uiLastLimit,
		FLMBOOL 		bLastMissingHigh);

	RCODE runPositioningQueryTest( 
		IF_Query *	pQuery, 
		FLMUINT64 * pui64ExpectedDocs, 
		FLMUINT		uiNumResults);
};

/****************************************************************************
Desc:
****************************************************************************/
RCODE getTest( IFlmTest ** ppTest)
{
	RCODE		rc = NE_XFLM_OK;

	if( (*ppTest = f_new SortKeyTest2Impl) == NULL)
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
RCODE SortKeyTest2Impl::createSortKey( 
	IF_Query * 	pQuery, 
	FLMBOOL 		bFirstAscending,
	FLMUINT 		uiFirstCompareRules,
	FLMUINT 		uiFirstLimit,
	FLMBOOL 		bFirstMissingHigh,
	FLMBOOL 		bLastAscending,
	FLMUINT 		uiLastCompareRules,
	FLMUINT 		uiLastLimit,
	FLMBOOL 		bLastMissingHigh)
{
	RCODE			rc = NE_XFLM_OK;
	void *		pvKeyContext = NULL;

	// First Name Sort Key Component

	if( RC_BAD( rc = pQuery->addSortKey( NULL, TRUE, TRUE, FIRST_NAME_ID,
		uiFirstCompareRules, uiFirstLimit, 1, !bFirstAscending, 
		bFirstMissingHigh, &pvKeyContext)))
	{
		MAKE_FLM_ERROR_STRING( "addSortKey failed.", m_szDetails, rc);
		goto Exit;
	}

	// LAST Name Sort Key Component

	if( RC_BAD( rc = pQuery->addSortKey( pvKeyContext, FALSE, TRUE,
		LAST_NAME_ID, uiLastCompareRules, uiLastLimit, 2, !bLastAscending,
		bLastMissingHigh, &pvKeyContext)))
	{
		MAKE_FLM_ERROR_STRING( "addSortKey failed.", m_szDetails, rc);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE SortKeyTest2Impl::execute( void)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bDibCreated = FALSE;
	FLMBOOL			bTransActive = FALSE;
	FLMUINT			uiFirstNameId = FIRST_NAME_ID;
	FLMUINT			uiLastNameId = LAST_NAME_ID;
	IF_Query *		pQuery = NULL;
	IF_DOMNode *	pDoc = NULL;
	FLMUINT			uiLoop;
	FLMUINT			uiNumDocsAdded = 0;

	ELEMENT_NODE_INFO	pStandardDocNodes[][2] =
		{
			{//0
				{(void*)"Rebecca", XFLM_TEXT_TYPE, 7, FIRST_NAME_ID},
				{(void*)"Betz", XFLM_TEXT_TYPE, 4, LAST_NAME_ID}
			},

			{//1
				{(void*)"Russell", XFLM_TEXT_TYPE, 7, FIRST_NAME_ID},
				{(void*)"Bakker", XFLM_TEXT_TYPE, 6, LAST_NAME_ID}
			},
	
			{//2
				{(void*)"April", XFLM_TEXT_TYPE, 5, FIRST_NAME_ID},
				{(void*)"Sansone", XFLM_TEXT_TYPE, 7, LAST_NAME_ID}
			},

			{//3
				{(void*)"Julie", XFLM_TEXT_TYPE, 5, FIRST_NAME_ID},
				{(void*)"Betz", XFLM_TEXT_TYPE, 4, LAST_NAME_ID}
			},

			{//4
				{(void*)"Jason", XFLM_TEXT_TYPE, 5, FIRST_NAME_ID},
				{(void*)"Betz", XFLM_TEXT_TYPE, 4, LAST_NAME_ID}
			},

			{//5
				{(void*)"Ronald", XFLM_TEXT_TYPE, 6, FIRST_NAME_ID},
				{(void*)"Betz", XFLM_TEXT_TYPE, 4, LAST_NAME_ID}
			},

			{//6
				{(void*)"Shawn", XFLM_TEXT_TYPE, 5, FIRST_NAME_ID},
				{(void*)"Hafner", XFLM_TEXT_TYPE, 6, LAST_NAME_ID}
			},

			{//7
				{(void*)"Karen", XFLM_TEXT_TYPE, 5, FIRST_NAME_ID},
				{(void*)"Bradley", XFLM_TEXT_TYPE, 7, LAST_NAME_ID}
			},

			{//8
				{(void*)"Mabel", XFLM_TEXT_TYPE, 5, FIRST_NAME_ID},
				{(void*)"Zepeda", XFLM_TEXT_TYPE, 6, LAST_NAME_ID}
			},

			{//9
				{(void*)"Kathleen", XFLM_TEXT_TYPE, 8, FIRST_NAME_ID},
				{(void*)"Ellinger", XFLM_TEXT_TYPE, 8, LAST_NAME_ID}
			},

			{//10
				{(void*)"Victor", XFLM_TEXT_TYPE, 6, FIRST_NAME_ID},
				{(void*)"Tankersley", XFLM_TEXT_TYPE, 10, LAST_NAME_ID}
			},

			{//11
				{(void*)"Leonard", XFLM_TEXT_TYPE, 7, FIRST_NAME_ID},
				{(void*)"Kuehn", XFLM_TEXT_TYPE, 5, LAST_NAME_ID}
			},

			{//12
				{(void*)"Victor", XFLM_TEXT_TYPE, 6, FIRST_NAME_ID},
				{(void*)"Tankersley", XFLM_TEXT_TYPE, 10, LAST_NAME_ID}
			},

			{//13
				{(void*)"Danny", XFLM_TEXT_TYPE, 5, FIRST_NAME_ID},
				{(void*)"Decastro", XFLM_TEXT_TYPE, 8, LAST_NAME_ID}
			},

			{//14
				{(void*)"Harold", XFLM_TEXT_TYPE, 6, FIRST_NAME_ID},
				{(void*)"Bergeron", XFLM_TEXT_TYPE, 8, LAST_NAME_ID}
			},

			{//15
				{(void*)"Annette", XFLM_TEXT_TYPE, 7, FIRST_NAME_ID},
				{(void*)"Sartin", XFLM_TEXT_TYPE, 6, LAST_NAME_ID}
			},

			{//16
				{(void*)"Anthony", XFLM_TEXT_TYPE, 7, FIRST_NAME_ID},
				{(void*)"Glasser", XFLM_TEXT_TYPE, 7, LAST_NAME_ID}
			},

			{//17
				{(void*)"Albert", XFLM_TEXT_TYPE, 6, FIRST_NAME_ID},
				{(void*)"Glasser", XFLM_TEXT_TYPE, 7, LAST_NAME_ID}
			},

			{//18
				{(void*)"Yvette", XFLM_TEXT_TYPE, 6, FIRST_NAME_ID},
				{(void*)"Patch", XFLM_TEXT_TYPE, 5, LAST_NAME_ID}
			},

			{//19
				{(void*)"Joyce", XFLM_TEXT_TYPE, 5, FIRST_NAME_ID},
				{(void*)"Lundberg", XFLM_TEXT_TYPE, 8, LAST_NAME_ID}
			},

			{//20
				{(void*)"John", XFLM_TEXT_TYPE, 4, FIRST_NAME_ID},
				{(void*)"Reinhold", XFLM_TEXT_TYPE, 8, LAST_NAME_ID}
			},

			{//21
				{(void*)"Kristen", XFLM_TEXT_TYPE, 7, FIRST_NAME_ID},
				{(void*)"Hansel", XFLM_TEXT_TYPE, 6, LAST_NAME_ID}
			},

			{//22
				{(void*)"Victor", XFLM_TEXT_TYPE, 6, FIRST_NAME_ID},
				{(void*)"Schell", XFLM_TEXT_TYPE, 6, LAST_NAME_ID}
			},

			{//23
				{(void*)"Patrick", XFLM_TEXT_TYPE, 7, FIRST_NAME_ID},
				{(void*)"Belt", XFLM_TEXT_TYPE, 4, LAST_NAME_ID}
			},

			{//24
				{(void*)"Gina", XFLM_TEXT_TYPE, 4, FIRST_NAME_ID},
				{(void*)"", XFLM_TEXT_TYPE, 0, LAST_NAME_ID}
			},			
			
			{//25
				{(void*)"Gina", XFLM_TEXT_TYPE, 4, FIRST_NAME_ID},
				{(void*)"Belt", XFLM_TEXT_TYPE, 4, LAST_NAME_ID}
			},
			
			{//26
				{(void*)"Brenda", XFLM_TEXT_TYPE, 6, FIRST_NAME_ID},
				{(void*)"Ellis", XFLM_TEXT_TYPE, 5, LAST_NAME_ID}
			},
			
			{//27
				{(void*)"Maryann", XFLM_TEXT_TYPE, 7, FIRST_NAME_ID},
				{(void*)"Sumpter", XFLM_TEXT_TYPE, 7, LAST_NAME_ID}
			},
			
			{//28
				{(void*)"Samantha", XFLM_TEXT_TYPE, 8, FIRST_NAME_ID},
				{(void*)"Beckford", XFLM_TEXT_TYPE, 8, LAST_NAME_ID}
			},
			
			{//29
				{(void*)"Carl", XFLM_TEXT_TYPE, 4, FIRST_NAME_ID},
				{(void*)"Collette", XFLM_TEXT_TYPE, 8, LAST_NAME_ID}
			},
			
			{//30
				{(void*)"Carl", XFLM_TEXT_TYPE, 4, FIRST_NAME_ID},
				{(void*)"Davis", XFLM_TEXT_TYPE, 5, LAST_NAME_ID}
			},
			
			{//31
				{(void*)"", XFLM_TEXT_TYPE, 0, FIRST_NAME_ID},
				{(void*)"", XFLM_TEXT_TYPE, 0, LAST_NAME_ID}
			},

			{//32
				{(void*)"Kristina", XFLM_TEXT_TYPE, 8, FIRST_NAME_ID},
				{(void*)"Tso", XFLM_TEXT_TYPE, 3, LAST_NAME_ID}
			},
			
			{//33
				{(void*)"Jeanne", XFLM_TEXT_TYPE, 6, FIRST_NAME_ID},
				{(void*)"Leonard", XFLM_TEXT_TYPE, 7, LAST_NAME_ID}
			},
			
			{//34
				{(void*)"", XFLM_TEXT_TYPE, 0, FIRST_NAME_ID},
				{(void*)"Leonard", XFLM_TEXT_TYPE, 7, LAST_NAME_ID}
			},
			
			{//35
				{(void*)"Patty", XFLM_TEXT_TYPE, 5, FIRST_NAME_ID},
				{(void*)"Ogletree", XFLM_TEXT_TYPE, 8, LAST_NAME_ID}
			},
			
			{//36
				{(void*)"Alan", XFLM_TEXT_TYPE, 4, FIRST_NAME_ID},
				{(void*)"Villa", XFLM_TEXT_TYPE, 5, LAST_NAME_ID}
			},
			
			{//37
				{(void*)"Carl", XFLM_TEXT_TYPE, 4, FIRST_NAME_ID},
				{(void*)"Wacker", XFLM_TEXT_TYPE, 6, LAST_NAME_ID}
			},
			
			{//38
				{(void*)"Lawrence", XFLM_TEXT_TYPE, 7, FIRST_NAME_ID},
				{(void*)"Kent", XFLM_TEXT_TYPE, 4, LAST_NAME_ID}
			},
		};

	FLMUINT64 pui64DocIDs[ELEMCOUNT(pStandardDocNodes)] = {0};

	ELEMENT_NODE_INFO pMultiKeyDocNodes[][6] =
		{
			{
				{(void*)"Steve", XFLM_TEXT_TYPE, 5, FIRST_NAME_ID},
				{(void*)"Jansen", XFLM_TEXT_TYPE, 6, LAST_NAME_ID},
				{(void*)"James", XFLM_TEXT_TYPE, 5, FIRST_NAME_ID},
				{(void*)"King", XFLM_TEXT_TYPE, 4, LAST_NAME_ID},
				{(void*)"Becky", XFLM_TEXT_TYPE, 5, FIRST_NAME_ID},
				{(void*)"Myer", XFLM_TEXT_TYPE, 4, LAST_NAME_ID}
			},

			{
				{(void*)"Jack", XFLM_TEXT_TYPE, 5, FIRST_NAME_ID},
				{(void*)"Stevens", XFLM_TEXT_TYPE, 6, LAST_NAME_ID},
				{(void*)"James", XFLM_TEXT_TYPE, 5, FIRST_NAME_ID},
				{(void*)"Baker", XFLM_TEXT_TYPE, 4, LAST_NAME_ID},
				{(void*)"Juan", XFLM_TEXT_TYPE, 5, FIRST_NAME_ID},
				{(void*)"Mamani", XFLM_TEXT_TYPE, 4, LAST_NAME_ID}
			},

		};

	FLMUINT64 pui64MultiKeyDocIDs[ELEMCOUNT(pMultiKeyDocNodes)] = {0};

	beginTest( "Missing/Duplicate Key Positionable Query Test Setup",
		"Prepare database to test sorted and positionable queries with "
		"duplicate or missing sort keys",
		"", 
		"");

	if ( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		goto Exit;
	}
	bDibCreated = TRUE;

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransActive = TRUE;

	if ( RC_BAD( rc = m_pDb->createElementDef( 
		NULL, "first_name", XFLM_TEXT_TYPE, &uiFirstNameId, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->createElementDef( 
		NULL, "last_name", XFLM_TEXT_TYPE, &uiLastNameId, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Exit;
	}

	// create all of our documents

	for( uiLoop = 0; uiLoop < ELEMCOUNT(pStandardDocNodes); uiLoop++)
	{
		if ( RC_BAD( rc = createCompoundDoc( 
			pStandardDocNodes[uiLoop], 2, &pui64DocIDs[uiLoop])))
		{
			goto Exit;
		}
		uiNumDocsAdded++;
	}

	for( uiLoop = 0; uiLoop < ELEMCOUNT(pMultiKeyDocNodes); uiLoop++)
	{
		if ( RC_BAD( rc = createCompoundDoc( 
			pMultiKeyDocNodes[uiLoop], 6, &pui64MultiKeyDocIDs[uiLoop])))
		{
			goto Exit;
		}
		uiNumDocsAdded++;
	}

	// Have to commit this transaction or the result set will not see
	// the nodes.

	if ( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_FLM_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransActive = FALSE;

	endTest("PASS");

	beginTest( "Missing Sort Key Test #1",
		"ensure sort key works properly with documents with missing keys",
		"", 
		"");

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransActive = TRUE;

	if ( RC_BAD( rc = m_pDbSystem->createIFQuery( &pQuery)))
	{
		MAKE_FLM_ERROR_STRING( "createIFQuery failed.", m_szDetails, rc);
		goto Exit;
	}

	{
		const char *	pszQuery = "//first_name[.==\"Gina\"] or //last_name[.==\"Leonard\"]";
		FLMUINT64		pui64ExpectedDocs[4];

		pui64ExpectedDocs[0] = pui64DocIDs[25]; // Gina Belt
		pui64ExpectedDocs[1] = pui64DocIDs[24]; // Gina ""
		pui64ExpectedDocs[2] = pui64DocIDs[33]; // Jeanne Leonard
		pui64ExpectedDocs[3] = pui64DocIDs[34]; // "" Leonard

		if ( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, pszQuery)))
		{
			goto Exit;
		}

		if ( RC_BAD( rc = createSortKey( 
			pQuery,
			TRUE, // First Ascending
			0, // compare rules
			0, // no limit
			TRUE, // Sort missing high
			TRUE, // sort last ascending
			0, // compare rules
			0, // no limit
			TRUE  // last missing high
			)))
		{
			goto Exit;
		}

		if ( RC_BAD( rc = runPositioningQueryTest( 
			pQuery, pui64ExpectedDocs, ELEMCOUNT(pui64ExpectedDocs))))
		{
			goto Exit;
		}
	}

	endTest("PASS");

	beginTest( "Missing Sort Key Test #2",
		"ensure sort key works properly with documents with missing keys",
		"", 
		"");

	{
		const char *		pszQuery = "//last_name[.>=\"Zepeda\"] or //first_name[.==\"\"] or //last_name[.==\"\"]";
		FLMUINT64			pui64ExpectedDocs[4];

		pui64ExpectedDocs[0] = pui64DocIDs[31];  // "" ""
		pui64ExpectedDocs[1] = pui64DocIDs[34];  // "" Leonard
		pui64ExpectedDocs[2] = pui64DocIDs[8];   // Mabel Zepeda
		pui64ExpectedDocs[3] = pui64DocIDs[24];  // Gina ""
		
		if ( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, pszQuery)))
		{
			goto Exit;
		}

		if ( RC_BAD( rc = createSortKey( 
			pQuery,
			FALSE,	// First Ascending
			0,			// compare rules
			0,			// no limit
			TRUE,		// Sort missing high
			TRUE,		// sort last ascending
			0,			// compare rules
			0,			// no limit
			FALSE		// last missing high
			)))
		{
			goto Exit;
		}

		if ( RC_BAD( rc = runPositioningQueryTest( 
			pQuery, pui64ExpectedDocs, ELEMCOUNT(pui64ExpectedDocs))))
		{
			goto Exit;
		}
	}

	endTest("PASS");

	beginTest( "Truncated Sort Key Test",
		"ensure sort key works properly with documents with truncated keys",
		"", 
		"");

	{
		const char *		pszQuery = "//last_name[.==\"B*\"]";
		FLMUINT64			pui64ExpectedDocs[11];

		pui64ExpectedDocs[0] = pui64DocIDs[25];  // Gina Belt
		pui64ExpectedDocs[1] = pui64DocIDs[14];  // Harold Bergeron
		pui64ExpectedDocs[2] = pui64MultiKeyDocIDs[1]; // James Baker
		pui64ExpectedDocs[3] = pui64DocIDs[4];  // Jason Betz
		pui64ExpectedDocs[4] = pui64DocIDs[3];  // Julie Betz
		pui64ExpectedDocs[5] = pui64DocIDs[7];  // Karen Bradley
		pui64ExpectedDocs[6] = pui64DocIDs[23];  // Patrick Belt
		pui64ExpectedDocs[7] = pui64DocIDs[0];  // Rebecca Betz
		pui64ExpectedDocs[8] = pui64DocIDs[5];  // Ronald Betz
		pui64ExpectedDocs[9] = pui64DocIDs[1];  // Russell Bakker
		pui64ExpectedDocs[10] = pui64DocIDs[28];  // Samantha Beckford
		
		if ( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, pszQuery)))
		{
			goto Exit;
		}

		if ( RC_BAD( rc = createSortKey( 
			pQuery,
			TRUE,		// First Ascending
			0,			// compare rules
			1,			// one character limit
			TRUE,		// Sort missing high
			TRUE,		// sort last ascending
			0,			// compare rules
			2,			// two character
			FALSE		// last missing high
			)))
		{
			goto Exit;
		}

		if ( RC_BAD( rc = runPositioningQueryTest( 
			pQuery, pui64ExpectedDocs, ELEMCOUNT(pui64ExpectedDocs))))
		{
			goto Exit;
		}
	}

	endTest("PASS");

	beginTest( "Sort By Last Name Test",
		"ensure sort key works properly with documents with missing keys",
		"", 
		"");

	{
		void * 			pvKeyContext = NULL;
		const char *	pszQuery = "//first_name[.==\"A*\"]";
		FLMUINT64		pui64ExpectedDocs[5];

		pui64ExpectedDocs[0] = pui64DocIDs[36];  // Alan Villa
		pui64ExpectedDocs[1] = pui64DocIDs[15];  // Annette Sartin
		pui64ExpectedDocs[2] = pui64DocIDs[2];   // April Sansone
		pui64ExpectedDocs[3] = pui64DocIDs[16];  // Anthony Glasser
		pui64ExpectedDocs[4] = pui64DocIDs[17];  // Albert Glasser
		
		if( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, pszQuery)))
		{
			goto Exit;
		}

		if ( RC_BAD( rc = pQuery->addSortKey(
			pvKeyContext,
			TRUE, //child
			TRUE,
			LAST_NAME_ID,
			0,
			0,
			1,
			TRUE, // descending
			TRUE, // missing high
			&pvKeyContext)))
		{
			MAKE_FLM_ERROR_STRING( "addSortKey failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = runPositioningQueryTest( 
			pQuery, pui64ExpectedDocs, ELEMCOUNT(pui64ExpectedDocs))))
		{
			goto Exit;
		}
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

	if ( RC_BAD( rc))
	{
		endTest("FAIL");
	}

	if( bTransActive)
	{
		if ( RC_BAD( rc))
		{
			m_pDb->transAbort();
		}
		else
		{
			rc = m_pDb->transCommit();
		}
	}

	shutdownTestState( DB_NAME_STR, bDibCreated);
	return rc;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE SortKeyTest2Impl::runPositioningQueryTest( 
	IF_Query * 		pQuery, 
	FLMUINT64 * 	pui64ExpectedDocs, 
	FLMUINT 			uiNumResults)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pDoc = NULL;
	FLMUINT			uiLoop;
	FLMUINT64		ui64Tmp;

	if ( RC_BAD( rc = pQuery->enablePositioning()))
	{
		MAKE_FLM_ERROR_STRING( "enablePositioning failed.", m_szDetails, rc);
		goto Exit;
	}

	for( uiLoop = 0; uiLoop < uiNumResults; uiLoop++)
	{
		if ( RC_BAD( rc = pQuery->getNext( m_pDb, &pDoc)))
		{
			MAKE_FLM_ERROR_STRING( "getNext failed.", m_szDetails, rc);
			goto Exit;
		}
		
		if( RC_BAD( rc = pDoc->getNodeId( m_pDb, &ui64Tmp)))
		{
			goto Exit;
		}

		if( ui64Tmp != pui64ExpectedDocs[uiLoop])
		{
			rc = RC_SET( NE_XFLM_DATA_ERROR);
			MAKE_FLM_ERROR_STRING( "Unexpected query result.", m_szDetails, rc);
			goto Exit;
		}
	}

	// Should be no more results

	if ( RC_OK( rc = pQuery->getNext( m_pDb, &pDoc)))
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		MAKE_FLM_ERROR_STRING( "Unexpected query result.", m_szDetails, rc);
		goto Exit;
	}

	rc = NE_XFLM_OK;

Exit:

	if( pDoc)
	{
		pDoc->Release();
	}

	return( rc);
}
