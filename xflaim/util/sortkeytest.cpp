//------------------------------------------------------------------------------
// Desc:	Sort key unit test
// Tabs:	3
//
// Copyright (c) 2005-2007 Novell, Inc. All Rights Reserved.
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
	#define DB_NAME_STR					"SYS:\\SKEY.DB"
#else
	#define DB_NAME_STR					"skey.db"
#endif

#define FIRST_NAME_ID 					123
#define LAST_NAME_ID 					456
#define IX_NUM 							789

/****************************************************************************
Desc:
****************************************************************************/
class SortKeyTestImpl : public TestBase
{
public:

	const char * getName( void);
	
	RCODE execute( void);

private:

	RCODE	verifyQuery( 
		const char *	pszQuery, 
		char *			ppszNames[][2],
		IF_Query *		pQuery,
		FLMUINT			uiNumNames,
		FLMBOOL			bFirstAscending,
		FLMBOOL			bLastAscending,
		FLMBOOL			bDocsIndexed,
		FLMBOOL			bIxFirstAscending,
		FLMBOOL			bIxLastAscending);

	RCODE	createNameDoc( 
		char * 		pszNames[ 2]);

	RCODE createOrModifyIndex( 
		FLMUINT		uiIndex,
		FLMBOOL		bComp1SortDescending,
		FLMBOOL		bComp1CaseSensitive,
		FLMBOOL		bComp1SortMissingHigh,
		FLMBOOL		bComp2SortDescending,
		FLMBOOL		bComp2SortMissingHigh);

	RCODE createSortKey( 
		IF_Query * 	pQuery, 
		FLMBOOL 		bFirstAscending, 
		FLMBOOL 		bLastAscending);
};

/****************************************************************************
Desc:
****************************************************************************/
RCODE getTest( 
	IFlmTest ** 	ppTest)
{
	RCODE		rc = NE_XFLM_OK;

	if( (*ppTest = f_new SortKeyTestImpl) == NULL)
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
const char * SortKeyTestImpl::getName( void)
{
	return( "SortKey Test");
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE	SortKeyTestImpl::verifyQuery( 
	const char *		pszQuery, 
	char*					ppszNames[][ 2],
	IF_Query *			pQuery,
	FLMUINT				uiNumNames,
	FLMBOOL				bFirstAscending,
	FLMBOOL				bLastAscending,
	FLMBOOL				bDocsIndexed,
	FLMBOOL				bIxFirstAscending,
	FLMBOOL				bIxLastAscending)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DOMNode *		pResultNode = NULL;
	IF_DOMNode *		pFirstNameNode = NULL;
	IF_DOMNode *		pLastNameNode = NULL;
	FLMUINT				uiLoop;
	FLMUINT				uiPos;
	char					szBuf[ 100];
	IF_DataVector *	pSearchKey = NULL;
	char					szTestName[ 200];
	static FLMUINT		uiCallCount = 0;
	const char *		pszTestNameFormat = 
		"Verify Query #%u "
		"(%s/First Name Asc == %u/Last Name Asc ==%u/%s)";


	uiCallCount++;

	f_sprintf( szTestName, pszTestNameFormat, uiCallCount,
		pszQuery, (unsigned)bFirstAscending, (unsigned)bLastAscending,
		"positioning");

	beginTest( szTestName, 
		"Ensure queries return proper results in proper order",
		"No Additional Info",
		"");

	// positioning tests

	if ( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, pszQuery)))
	{
		MAKE_FLM_ERROR_STRING( "setupQueryExpr failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( bDocsIndexed && 
		(bFirstAscending == bIxFirstAscending && 
		bLastAscending == bIxLastAscending))
	{
		pQuery->setIndex( IX_NUM);
	}

	if ( RC_BAD( rc = pQuery->enablePositioning()))
	{
		MAKE_FLM_ERROR_STRING( "enablePositioning failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = createSortKey( 
		pQuery, bFirstAscending, bLastAscending)))
	{
		goto Exit;
	}

	for( uiLoop = 0; uiLoop < uiNumNames; uiLoop++)
	{

		if ( RC_BAD( rc = pQuery->positionTo( 
			m_pDb, &pResultNode, 0, uiLoop)))
		{
			MAKE_FLM_ERROR_STRING( "positionTo failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = pResultNode->getFirstChild( m_pDb, &pFirstNameNode)))
		{
			MAKE_FLM_ERROR_STRING( "getFirstChild failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = pFirstNameNode->getNextSibling( m_pDb, &pLastNameNode)))
		{
			MAKE_FLM_ERROR_STRING( "getNextSibling failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = pFirstNameNode->getUTF8( m_pDb, 
			(FLMBYTE *)szBuf, sizeof( szBuf), 0, sizeof( szBuf) - 1)))
		{
			MAKE_FLM_ERROR_STRING( "getUTF8 failed.", m_szDetails, rc);
			goto Exit;
		} 

		if ( f_strcmp( szBuf, ppszNames[uiLoop][0]) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);
			MAKE_FLM_ERROR_STRING( "Unexpected value.", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = pLastNameNode->getUTF8( m_pDb, 
			(FLMBYTE *)szBuf, sizeof( szBuf), 0, sizeof( szBuf) - 1)))
		{
			MAKE_FLM_ERROR_STRING( "getUTF8 failed.", m_szDetails, rc);
			goto Exit;
		} 

		if ( f_strcmp( szBuf, ppszNames[uiLoop][1]) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);
			MAKE_FLM_ERROR_STRING( "Unexpected value.", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = pQuery->getPosition( m_pDb, &uiPos)))
		{
			MAKE_FLM_ERROR_STRING( "getPosition failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( uiPos != uiLoop)
		{
			rc = RC_SET( NE_XFLM_FAILURE);
			MAKE_FLM_ERROR_STRING( "Unexpected position", m_szDetails, rc);
			goto Exit;
		}

	}

	endTest("PASS");

	f_sprintf( szTestName, pszTestNameFormat, uiCallCount,
		pszQuery, (unsigned)bFirstAscending, (unsigned)bLastAscending,
		"search key positioning");

	beginTest( szTestName, 
		"Ensure queries return proper results in proper order",
		"No Additional Info",
		"");
		
	if( RC_BAD( rc = m_pDbSystem->createIFDataVector( &pSearchKey)))
	{
		goto Exit;
	}

	// positioning tests 2

	for( uiLoop = 0; uiLoop < uiNumNames; uiLoop++)
	{

		if ( uiLoop == 0)
		{
			if ( RC_BAD( rc = pQuery->positionTo( 
				m_pDb, &pResultNode, 0, pSearchKey, XFLM_FIRST)))
			{
				MAKE_FLM_ERROR_STRING( "positionTo failed.", m_szDetails, rc);
				goto Exit;
			}
		}
		else
		{
			if ( RC_BAD( rc = pSearchKey->setUTF8( 0, 
				(FLMBYTE *)ppszNames[uiLoop][0])))
			{
				MAKE_FLM_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
				goto Exit;
			}

			if ( RC_BAD( rc = pSearchKey->setUTF8( 1, 
				(FLMBYTE *)ppszNames[uiLoop][1])))
			{
				MAKE_FLM_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
				goto Exit;
			}

			if ( RC_BAD( rc = pQuery->positionTo( 
				m_pDb, &pResultNode, 0, pSearchKey, XFLM_EXACT)))
			{
				MAKE_FLM_ERROR_STRING( "positionTo failed.", m_szDetails, rc);
				goto Exit;
			}
		}

		if ( RC_BAD( rc = pResultNode->getFirstChild( m_pDb, &pFirstNameNode)))
		{
			MAKE_FLM_ERROR_STRING( "getFirstChild failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = pFirstNameNode->getNextSibling( m_pDb, &pLastNameNode)))
		{
			MAKE_FLM_ERROR_STRING( "getNextSibling failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = pFirstNameNode->getUTF8( m_pDb,
			(FLMBYTE *)szBuf, sizeof( szBuf), 0, sizeof( szBuf) - 1)))
		{
			MAKE_FLM_ERROR_STRING( "getUTF8 failed.", m_szDetails, rc);
			goto Exit;
		} 

		if ( f_strcmp( szBuf, ppszNames[uiLoop][0]) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);
			MAKE_FLM_ERROR_STRING( "Unexpected value.", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = pLastNameNode->getUTF8( m_pDb, 
			(FLMBYTE *)szBuf, sizeof( szBuf), 0, sizeof( szBuf) - 1)))
		{
			MAKE_FLM_ERROR_STRING( "getUTF8 failed.", m_szDetails, rc);
			goto Exit;
		} 

		if ( f_strcmp( szBuf, ppszNames[uiLoop][1]) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);
			MAKE_FLM_ERROR_STRING( "Unexpected value.", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = pQuery->getPosition( m_pDb, &uiPos)))
		{
			MAKE_FLM_ERROR_STRING( "getPosition failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( uiPos != uiLoop)
		{
			rc = RC_SET( NE_XFLM_FAILURE);
			MAKE_FLM_ERROR_STRING( "Unexpected position", m_szDetails, rc);
			goto Exit;
		}
	}

	endTest("PASS");

Exit:

	if( RC_BAD( rc))
	{
		endTest("FAIL");
	}

	if( pResultNode)
	{
		pResultNode->Release();
	}

	if( pFirstNameNode)
	{
		pFirstNameNode->Release();
	}

	if( pLastNameNode)
	{
		pLastNameNode->Release();
	}
	
	if( pSearchKey)
	{
		pSearchKey->Release();
	}

	return rc;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE SortKeyTestImpl::createOrModifyIndex( 
	FLMUINT			uiIndex,
	FLMBOOL			bComp1SortDescending,
	FLMBOOL			bComp1CaseSensitive,
	FLMBOOL			bComp1SortMissingHigh,
	FLMBOOL			bComp2SortDescending,
	FLMBOOL			bComp2SortMissingHigh)
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
				"xflaim:Required=\"1\" " // have to make this required so it will match the sort key ixd
				"xflaim:KeyComponent=\"1\" />"
			"<xflaim:ElementComponent "
				"xflaim:CompareRules=\"%s\" "
				"xflaim:targetNameSpace=\"\" "
				"xflaim:DictNumber=\"%u\" "
				"xflaim:IndexOn=\"value\" "
				"xflaim:Required=\"1\" " // have to make this required so it will match the sort key ixd
				"xflaim:KeyComponent=\"2\"/>"
		"</xflaim:Index>";

	szCompareRules1[0] = '\0';
	szCompareRules2[0] = '\0';

	if( bComp1SortDescending)
	{
		f_strcpy( szCompareRules1, XFLM_DESCENDING_OPTION_STR);
	}

	if ( !bComp1CaseSensitive)
	{
		f_strcat( szCompareRules1, " "XFLM_CASE_INSENSITIVE_OPTION_STR);
	}

	if ( bComp1SortMissingHigh)
	{
		f_strcat( szCompareRules1, " "XFLM_MISSING_HIGH_OPTION_STR);
	}

	if ( bComp2SortDescending)
	{
		f_strcpy( szCompareRules2, XFLM_DESCENDING_OPTION_STR);
	}

	if ( bComp2SortMissingHigh)
	{
		f_strcat( szCompareRules2, " "XFLM_MISSING_HIGH_OPTION_STR);
	}

	f_sprintf( szName, "(%s/%s)", szCompareRules1, szCompareRules2); 

	if ( RC_BAD( rc = f_alloc( f_strlen( pszIndexFormat) + 
		(f_strlen( szCompareRules1) + f_strlen(szCompareRules2)) * 2 +
		16, &pszIndex)))
	{
		MAKE_FLM_ERROR_STRING( "f_alloc failed", m_szDetails, rc);
		goto Exit;
	}

	f_sprintf( pszIndex, pszIndexFormat, szName, uiIndex, 
		szCompareRules1, FIRST_NAME_ID, szCompareRules2, LAST_NAME_ID); 

	// remove the index if it already exists

	if ( RC_OK( rc = m_pDb->getDictionaryDef( ELM_INDEX_TAG, uiIndex, &pNode)))
	{
		if ( RC_BAD( rc = pNode->deleteNode( m_pDb)))
		{
			MAKE_FLM_ERROR_STRING( "deleteNode failed.", m_szDetails, rc);
			goto Exit;
		}
	}

	if ( RC_BAD( rc = importBuffer( pszIndex, XFLM_DICT_COLLECTION)))
	{
		goto Exit;
	}

Exit:

	if ( pNode)
	{
		pNode->Release();
	}

	if ( pszIndex)
	{
		f_free( &pszIndex);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE	SortKeyTestImpl::createNameDoc( 
	char *		pszNames[ 2])
{
	RCODE						rc = NE_XFLM_OK;
	ELEMENT_NODE_INFO 	pNameNodes[2];

	f_memset( pNameNodes, 0, sizeof( pNameNodes));

	if ( RC_BAD( rc = f_alloc( 
		f_strlen( pszNames[0]) + 1, &pNameNodes[0].pvData)))
	{
		MAKE_FLM_ERROR_STRING( "f_alloc failed.", m_szDetails, rc);
		goto Exit;
	}

	f_strcpy( (char*)pNameNodes[0].pvData, pszNames[0]);
	pNameNodes[0].uiDataType = XFLM_TEXT_TYPE;
	pNameNodes[0].uiDataSize = f_strlen( pszNames[0]);
	pNameNodes[0].uiDictNum = FIRST_NAME_ID;

	if ( RC_BAD( rc = f_alloc( 
		f_strlen( pszNames[1]) + 1, &pNameNodes[1].pvData)))
	{
		MAKE_FLM_ERROR_STRING( "f_alloc failed.", m_szDetails, rc);
		goto Exit;
	}

	f_strcpy( (char*)pNameNodes[1].pvData, pszNames[1]);
	pNameNodes[1].uiDataType = XFLM_TEXT_TYPE;
	pNameNodes[1].uiDataSize = f_strlen( pszNames[0]);
	pNameNodes[1].uiDictNum = LAST_NAME_ID;

	if ( RC_BAD( rc = createCompoundDoc( pNameNodes, 2, NULL)))
	{
		goto Exit;
	}

Exit:

	if ( pNameNodes[0].pvData)
	{
		f_free( &pNameNodes[0].pvData);
	}

	if ( pNameNodes[1].pvData)
	{
		f_free( &pNameNodes[1].pvData);
	}

	return rc;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE SortKeyTestImpl::createSortKey( 
	IF_Query * 	pQuery, 
	FLMBOOL 		bFirstAscending, 
	FLMBOOL 		bLastAscending)
{
	RCODE			rc = NE_XFLM_OK;
	void *		pvKeyContext = NULL;

	// First Name Sort Key Component

	if ( RC_BAD( rc = pQuery->addSortKey(
		NULL,
		TRUE,
		TRUE,
		FIRST_NAME_ID,
		0, // VISIT - allow compare rules to be specified
		0, // VISIT - allow limit to be specified
		1,
		!bFirstAscending,
		FALSE, // VISIT - allow sort missing high to be specified
		&pvKeyContext)))
	{
		MAKE_FLM_ERROR_STRING( "addSortKey failed.", m_szDetails, rc);
		goto Exit;
	}

	// LAST Name Sort Key Component

	if ( RC_BAD( rc = pQuery->addSortKey(
		pvKeyContext,
		FALSE, //sibling
		TRUE,
		LAST_NAME_ID,
		0, // VISIT - allow compare rules to be specified
		0, // VISIT - allow limit to be specified
		2,
		!bLastAscending,
		FALSE, // VISIT - allow sort missing high to be specified
		&pvKeyContext)))
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
RCODE SortKeyTestImpl::execute( void)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bDibCreated = FALSE;
	FLMBOOL			bTransActive = FALSE;
	FLMUINT			uiFirstNameId = FIRST_NAME_ID;
	FLMUINT			uiLastNameId = LAST_NAME_ID;
	IF_Query *		pQuery = NULL;
	FLMUINT			uiLoop;
	FLMBOOL			bIxFirstAsc = FALSE;
	FLMBOOL			bIxLastAsc = FALSE;
	FLMBOOL			bIndexCreated = FALSE;

	char* ppszNames[][2] = 
	{
		{"Rebecca","Betz"}, 
		{"Russell","Bakker"},
		{"April","Sansone"},
		{"Julie","Betz"},
		{"Jason","Betz"},
		{"Ronald","Betz"},
		{"Shawn","Hafner"},
		{"Karen","Bradley"},
		{"Mabel","Zepeda"},
		{"Kathleen","Ellinger"},
		{"Victor","Tankersley"},
		{"Leonard","Kuehn"},
		{"Danny","Decastro"},
		{"Harold","Bergeron"},
		{"Annette","Sartin"},
		{"Anthony","Glasser"},
		{"Albert","Glasser"},
		{"Yvette","Patch"},
		{"Joyce","Lundberg"},
		{"John","Reinhold"},
		{"Kristen","Hansel"},
		{"Victor","Schell"},
		{"Patrick","Belt"},
		{"Gina","Belt"},
		{"Brenda","Ellis"},
		{"Maryann","Sumpter"},
		{"Samantha","Beckford"},
		{"Carl","Collette"},
		{"Carl","Davis"},
		{"Kristina","Tso"},
		{"Jeanne","Leonard"},
		{"Patty","Ogletree"},
		{"Alan","Villa"},
		{"Carl","Wacker"},
		{"Lawrence","Kent"},
	};

	beginTest( "Sorting/Positioning Query Test Setup",
		"Prepare database to test sorted and positionable queries",
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

	for ( uiLoop = 0; uiLoop < ELEMCOUNT(ppszNames); uiLoop++)
	{
		if ( RC_BAD( rc = createNameDoc( ppszNames[uiLoop])))
		{
			goto Exit;
		}
	}
	
	if ( RC_BAD( rc = m_pDbSystem->createIFQuery( &pQuery)))
	{
		MAKE_FLM_ERROR_STRING( "createIFQuery failed.", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

	for( uiLoop = 0; uiLoop < 5; uiLoop++)
	{
		// We have to commit all our changes here because when result sets are
		// built in the background, the F_Db the background thread receives to
		// work with is created via a call to dbDup() and will not have knowledge
		// of any uncommitted changes made to the database. If this issue is ever
		// fixed, remove these calls to transCommit and transBegin.

		if ( RC_BAD( rc = m_pDb->transCommit()))
		{
			MAKE_FLM_ERROR_STRING( "createIFQuery failed.", m_szDetails, rc);
			goto Exit;
		}
		bTransActive = FALSE;

		if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
		{
			MAKE_FLM_ERROR_STRING( "createIFQuery failed.", m_szDetails, rc);
			goto Exit;
		}
		bTransActive = TRUE;

		// no index the first time throug...

		{
			const char *	pszQuery = "//first_name[.==\"Carl\"]";
			char * 			pszResults[][3][2] =
			{
				{{"Carl","Wacker"},{"Carl","Davis"},{"Carl","Collette"}}, // desc,desc
				{{"Carl","Collette"},{"Carl","Davis"},{"Carl","Wacker"}}, // desc,asc
				{{"Carl","Wacker"},{"Carl","Davis"},{"Carl","Collette"}}, // asc,desc
				{{"Carl","Collette"},{"Carl","Davis"},{"Carl","Wacker"}}, // asc,asc
			};

			if ( RC_BAD( rc = verifyQuery( 
				pszQuery, pszResults[0], pQuery, ELEMCOUNT(pszResults[0]),
				FALSE, FALSE, bIndexCreated, bIxFirstAsc, bIxLastAsc)))
			{
				goto Exit;
			}

			if ( RC_BAD( rc = verifyQuery( 
				pszQuery, pszResults[1], pQuery, ELEMCOUNT(pszResults[1]),
				FALSE, TRUE, bIndexCreated, bIxFirstAsc, bIxLastAsc)))
			{
				goto Exit;
			}

			if ( RC_BAD( rc = verifyQuery( 
				pszQuery, pszResults[2], pQuery, ELEMCOUNT(pszResults[2]),
				TRUE, FALSE, bIndexCreated, bIxFirstAsc, bIxLastAsc)))
			{
				goto Exit;
			}

			if ( RC_BAD( rc = verifyQuery( 
				pszQuery, pszResults[3], pQuery, ELEMCOUNT(pszResults[3]),
				TRUE, TRUE, bIndexCreated, bIxFirstAsc, bIxLastAsc)))
			{
				goto Exit;
			}
		}

		{
			const char *	pszQuery = "//last_name[.==\"Glasser\"]";
			char * 			pszResults[][2][2] =
			{
				{{"Anthony","Glasser"},{"Albert","Glasser"}}, // desc,desc
				{{"Anthony","Glasser"},{"Albert","Glasser"}}, // desc,asc
				{{"Albert","Glasser"},{"Anthony","Glasser"}}, // asc,desc
				{{"Albert","Glasser"},{"Anthony","Glasser"}}, // asc,asc
			};

			if ( RC_BAD( rc = verifyQuery( 
				pszQuery, pszResults[0], pQuery, ELEMCOUNT(pszResults[0]),
				FALSE, FALSE, bIndexCreated, bIxFirstAsc, bIxLastAsc)))
			{
				goto Exit;
			}

			if ( RC_BAD( rc = verifyQuery( 
				pszQuery, pszResults[1], pQuery, ELEMCOUNT(pszResults[1]),
				FALSE, TRUE, bIndexCreated, bIxFirstAsc, bIxLastAsc)))
			{
				goto Exit;
			}

			if ( RC_BAD( rc = verifyQuery( 
				pszQuery, pszResults[2], pQuery, ELEMCOUNT(pszResults[2]),
				TRUE, FALSE, bIndexCreated, bIxFirstAsc, bIxLastAsc)))
			{
				goto Exit;
			}

			if ( RC_BAD( rc = verifyQuery( 
				pszQuery, pszResults[3], pQuery, ELEMCOUNT(pszResults[3]),
				TRUE, TRUE, bIndexCreated, bIxFirstAsc, bIxLastAsc)))
			{
				goto Exit;
			}
		}

		{
			const char *		pszQuery = "//first_name[. >= \"C\" && . < \"J\"]";
			char *				pszResults[][6][2] =
			{
				{ // Desc, Desc
					{"Harold","Bergeron"},
					{"Gina","Belt"},
					{"Danny","Decastro"},
					{"Carl","Wacker"},
					{"Carl","Davis"},
					{"Carl","Collette"},
				},
				{ // Desc, Asc
					{"Harold","Bergeron"},
					{"Gina","Belt"},
					{"Danny","Decastro"},
					{"Carl","Collette"},
					{"Carl","Davis"},
					{"Carl","Wacker"},
				},
				{ // Asc, Desc
					{"Carl","Wacker"},
					{"Carl","Davis"},
					{"Carl","Collette"},
					{"Danny","Decastro"},
					{"Gina","Belt"},
					{"Harold","Bergeron"},
				},
				{ // Asc, Asc
					{"Carl","Collette"},
					{"Carl","Davis"},
					{"Carl","Wacker"},
					{"Danny","Decastro"},
					{"Gina","Belt"},
					{"Harold","Bergeron"},
				},
			};

			if ( RC_BAD( rc = verifyQuery( 
				pszQuery, pszResults[0], pQuery, ELEMCOUNT(pszResults[0]),
				FALSE, FALSE, bIndexCreated, bIxFirstAsc, bIxLastAsc)))
			{
				goto Exit;
			}

			if ( RC_BAD( rc = verifyQuery( 
				pszQuery, pszResults[1], pQuery, ELEMCOUNT(pszResults[1]),
				FALSE, TRUE, bIndexCreated, bIxFirstAsc, bIxLastAsc)))
			{
				goto Exit;
			}

			if ( RC_BAD( rc = verifyQuery( 
				pszQuery, pszResults[2], pQuery, ELEMCOUNT(pszResults[2]),
				TRUE, FALSE, bIndexCreated, bIxFirstAsc, bIxLastAsc)))
			{
				goto Exit;
			}

			if ( RC_BAD( rc = verifyQuery( 
				pszQuery, pszResults[3], pQuery, ELEMCOUNT(pszResults[3]),
				TRUE, TRUE, bIndexCreated, bIxFirstAsc, bIxLastAsc)))
			{
				goto Exit;
			}

		}

		{
			const char *		pszQuery = "//last_name[. >= \"Betz\" && . <= \"Davis\"]";
			char * 				pszResults[][7][2] =
			{
				{// Desc, Desc
					{"Ronald","Betz"},
					{"Rebecca","Betz"},
					{"Karen","Bradley"},
					{"Julie","Betz"},
					{"Jason","Betz"},
					{"Carl","Davis"},
					{"Carl","Collette"},
				},
				{// Desc, Asc
					{"Ronald","Betz"},
					{"Rebecca","Betz"},
					{"Karen","Bradley"},
					{"Julie","Betz"},
					{"Jason","Betz"},
					{"Carl","Collette"},
					{"Carl","Davis"},
				},
				{// Asc, Desc
					{"Carl","Davis"},
					{"Carl","Collette"},
					{"Jason","Betz"},
					{"Julie","Betz"},
					{"Karen","Bradley"},
					{"Rebecca","Betz"},
					{"Ronald","Betz"},
				},
				{// Asc, Asc
					{"Carl","Collette"},
					{"Carl","Davis"},
					{"Jason","Betz"},
					{"Julie","Betz"},
					{"Karen","Bradley"},
					{"Rebecca","Betz"},
					{"Ronald","Betz"},
				},
			};

			if ( RC_BAD( rc = verifyQuery( 
				pszQuery, pszResults[0], pQuery, ELEMCOUNT(pszResults[0]),
				FALSE, FALSE, bIndexCreated, bIxFirstAsc, bIxLastAsc)))
			{
				goto Exit;
			}

			if ( RC_BAD( rc = verifyQuery( 
				pszQuery, pszResults[1], pQuery, ELEMCOUNT(pszResults[1]),
				FALSE, TRUE, bIndexCreated, bIxFirstAsc, bIxLastAsc)))
			{
				goto Exit;
			}

			if ( RC_BAD( rc = verifyQuery( 
				pszQuery, pszResults[2], pQuery, ELEMCOUNT(pszResults[2]),
				TRUE, FALSE, bIndexCreated, bIxFirstAsc, bIxLastAsc)))
			{
				goto Exit;
			}

			if ( RC_BAD( rc = verifyQuery( 
				pszQuery, pszResults[3], pQuery, ELEMCOUNT(pszResults[3]),
				TRUE, TRUE, bIndexCreated, bIxFirstAsc, bIxLastAsc)))
			{
				goto Exit;
			}
		}

		{
			const char *	pszQuery = "//first_name[.==\"A*\" or .==\"J*\"]";
			char *			pszResults[][10][2] =
			{
				{ //Desc, Desc
					{"Julie","Betz"},
					{"Joyce","Lundberg"},
					{"John","Reinhold"},
					{"Jeanne","Leonard"},
					{"Jason","Betz"},
					{"April","Sansone"},
					{"Anthony","Glasser"},
					{"Annette","Sartin"},
					{"Albert","Glasser"},
					{"Alan","Villa"},
				},
				{ //Desc, Asc
					{"Julie","Betz"},
					{"Joyce","Lundberg"},
					{"John","Reinhold"},
					{"Jeanne","Leonard"},
					{"Jason","Betz"},
					{"April","Sansone"},
					{"Anthony","Glasser"},
					{"Annette","Sartin"},
					{"Albert","Glasser"},
					{"Alan","Villa"},
				},
				{ //Asc, Desc
					{"Alan","Villa"},
					{"Albert","Glasser"},
					{"Annette","Sartin"},
					{"Anthony","Glasser"},
					{"April","Sansone"},
					{"Jason","Betz"},
					{"Jeanne","Leonard"},
					{"John","Reinhold"},
					{"Joyce","Lundberg"},
					{"Julie","Betz"},
				},
				{ //Asc, Asc
					{"Alan","Villa"},
					{"Albert","Glasser"},
					{"Annette","Sartin"},
					{"Anthony","Glasser"},
					{"April","Sansone"},
					{"Jason","Betz"},
					{"Jeanne","Leonard"},
					{"John","Reinhold"},
					{"Joyce","Lundberg"},
					{"Julie","Betz"},
				},
			};

			if ( RC_BAD( rc = verifyQuery( 
				pszQuery, pszResults[0], pQuery, ELEMCOUNT(pszResults[0]),
				FALSE, FALSE, bIndexCreated, bIxFirstAsc, bIxLastAsc)))
			{
				goto Exit;
			}

			if ( RC_BAD( rc = verifyQuery( 
				pszQuery, pszResults[1], pQuery, ELEMCOUNT(pszResults[1]),
				FALSE, TRUE, bIndexCreated, bIxFirstAsc, bIxLastAsc)))
			{
				goto Exit;
			}

			if ( RC_BAD( rc = verifyQuery( 
				pszQuery, pszResults[2], pQuery, ELEMCOUNT(pszResults[2]),
				TRUE, FALSE, bIndexCreated, bIxFirstAsc, bIxLastAsc)))
			{
				goto Exit;
			}

			if ( RC_BAD( rc = verifyQuery( 
				pszQuery, pszResults[3], pQuery, ELEMCOUNT(pszResults[3]),
				TRUE, TRUE, bIndexCreated, bIxFirstAsc, bIxLastAsc)))
			{
				goto Exit;
			}
		}

		switch( uiLoop)
		{
		case 0:
			bIxFirstAsc = FALSE;
			bIxLastAsc = FALSE;
			break;
		case 1:
			bIxFirstAsc = FALSE;
			bIxLastAsc = TRUE;
			break;
		case 2:
			bIxFirstAsc = TRUE;
			bIxLastAsc = FALSE;
			break;
		case 3:
			bIxFirstAsc = TRUE;
			bIxLastAsc = TRUE;
			break;
		default:
			break;
		}

		if ( RC_BAD( rc = createOrModifyIndex( 
			IX_NUM,
			!bIxFirstAsc,
			TRUE,
			FALSE,
			!bIxLastAsc,
			FALSE)))
		{
			goto Exit;
		}
		bIndexCreated = TRUE;
	}

Exit:

	if ( pQuery)
	{
		pQuery->Release();
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
	return( rc);
}
