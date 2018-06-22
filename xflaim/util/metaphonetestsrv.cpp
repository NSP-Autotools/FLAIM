//------------------------------------------------------------------------------
// Desc:	Metaphone unit test
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

struct WordPair
{
	const char * pszWord1;
	const char * pszWord2;
};

/****************************************************************************
Desc:
****************************************************************************/
class IMetaphoneTestImpl : public TestBase
{
public:

	const char * getName( void);
	
	RCODE suite1( void);
	
	RCODE suite2( void);
	
	RCODE execute( void);
};

/****************************************************************************
Desc:
****************************************************************************/
RCODE getTest( IFlmTest ** ppTest)
{
	RCODE		rc = NE_XFLM_OK;

	if( (*ppTest = f_new IMetaphoneTestImpl) == NULL)
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
const char * IMetaphoneTestImpl::getName( void)
{
	return( "MetaPhone Test");
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IMetaphoneTestImpl::execute( void)
{
	RCODE	rc = NE_XFLM_OK;

	if ( RC_BAD( rc = suite1()))
	{
		goto Exit;
	}

	if ( RC_BAD( rc = suite2()))
	{
		goto Exit;
	}

Exit:

	return rc;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IMetaphoneTestImpl::suite1( void)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DOMNode *		pRoot = NULL;
	IF_DOMNode *		pChild = NULL;
	FLMBOOL				bTransStarted = FALSE;
	FLMBOOL				bDibCreated = FALSE;
	FLMUINT				uiLoop = 0;
	FLMUINT				uiWordDictNum = 0;
	IF_PosIStream	*	pMetaphoneIStream = NULL;
	IF_DataVector *	pSearchKey = NULL;
	IF_DataVector *	pFoundKey = NULL;
	FLMUINT				uiMetaphone1 = 0;
	FLMUINT				uiMetaphone2 = 0;
	FLMUINT				uiIxMetaVal;
	const char *		pszIndexDef =
		"<xflaim:Index "
		"	xmlns:xflaim=\"http://www.novell.com/XMLDatabase/Schema\""
		"	xflaim:name=\"Misspelled Words\" "
		"	xflaim:DictNumber=\"77\"> " // HARD-CODED Dict Num for easy retrieval
		"	<xflaim:ElementComponent "
		"		xflaim:name=\"Word\" "
		"		xflaim:KeyComponent=\"1\" "
		"		xflaim:IndexOn=\"metaphone\"/> "
		"</xflaim:Index> ";

	WordPair commonMisspellings[] =
	{
		//correct,incorrect
		
		{"night","nite"},
		{"anoint","annoint"},   
		{"coolly","cooly"}, 
		{"supercede","supersede"},   
		{"irresistible","irresistable"},   
		{"development","developement"},   
		{"separate","seperate"},   
		{"tyranny ","tyrrany"},
		{"harass", "harrass"}, 
		{"desiccate", "dessicate"},
		{"indispensable", "indispensible"}, 
		{"receive","recieve"}, 
		{"pursue", "persue"}, 
		{"recommend","reccomend"},
		{"desperate","desparate"}, 
		{"liquefy","liquify"}, 
		{"seize", "sieze"}, 
		{"cemetery","cemetary"},  
		{"subpoena", "subpena"}, 
		{"definitely", "definately"},
		{"occasion","ocassion"},
		{"consensus", "concensus"}, 
		{"inadvertent","inadvertant"},   
		{"miniscule","minuscule"},
		{"judgment","judgement"},
		{"inoculate","innoculate"}, 
		{"drunkenness","drunkeness"}, 
		{"occurrence","occurence"}, 
		{"dissipate","disippate"}, 
		{"weird","wierd"}, 
		{"accommodate","accomodate"},
		{"embarrassment","embarassment"},
		{"ecstasy","ecstacy"},
		{"repetition","repitition"},
		{"batallion","battalion"},
		{"despair","dispair"},
		{"irritable","irritible"},
		{"accidentally","accidently"},
		{"liaison","liason"},
		{"memento","momento "},
		{"broccoli","brocolli"},
		{"millennium","millenium"},
		{"yield","yeild"},
		{"existence","existance"},
		{"independent","independant"},
		{"sacrilegious","sacreligious"},
		{"insistent","insistant"},
		{"excee","excede"},
		{"privilege","priviledge"},
	};

	FLMUINT		uiNumWords = sizeof( commonMisspellings)/sizeof( commonMisspellings[0]);
 
	if ( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to init test state.", m_szDetails, rc);
		goto Exit;
	}
	bDibCreated = TRUE;
	
	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;

	if ( RC_BAD( rc = m_pDb->createElementDef(		
		NULL,
		"Word",
		XFLM_TEXT_TYPE,
		&uiWordDictNum)))
	{
		MAKE_FLM_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = importBuffer( pszIndexDef, XFLM_DICT_COLLECTION)))
	{
		 goto Exit;
	}

	beginTest( 
		"Metaphone Index Key Creation Test",
		"Make sure mispelled words can be used to retrieve index keys "
		"created for correctly-spelled words.",
		"1) Create nodes with correctly-spelled words "
		"2) search for the index keys using mispelled versions of those words ",
		"");

	//Create a node with a word attribute with a value of a correctly-spelled word
	
	if( RC_BAD( rc = m_pDb->createRootElement(
		XFLM_DATA_COLLECTION, ELM_ELEMENT_TAG, &pRoot)))
	{
		MAKE_FLM_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
		goto Exit;
	}

	for( uiLoop = 0; uiLoop < uiNumWords; uiLoop++)
	{
		if( RC_BAD( rc = pRoot->createNode( 
			m_pDb, ELEMENT_NODE, uiWordDictNum, XFLM_FIRST_CHILD, &pChild)))
		{
			MAKE_FLM_ERROR_STRING( "createNode failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = pChild->setUTF8( m_pDb, 
			(FLMBYTE *)commonMisspellings[uiLoop].pszWord1)))
		{
			MAKE_FLM_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
			goto Exit;
		}
	}

	// Now, we should be able to locate every misspelled word in our list

	if ( RC_BAD( rc = m_pDbSystem->createIFDataVector( &pSearchKey)))
	{
		MAKE_FLM_ERROR_STRING( "createIFDataVector failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDbSystem->createIFDataVector( &pFoundKey)))
	{
		MAKE_FLM_ERROR_STRING( "createIFDataVector failed.", m_szDetails, rc);
		goto Exit;
	}

	f_strcpy( m_szDetails, "Words: ");
	for ( uiLoop = 0; uiLoop < uiNumWords; uiLoop++)
	{
		if ( RC_BAD( rc = m_pDbSystem->openBufferIStream( 
			commonMisspellings[uiLoop].pszWord1,
			f_strlen( commonMisspellings[uiLoop].pszWord1),
			&pMetaphoneIStream)))
		{
			MAKE_FLM_ERROR_STRING( "openBufferIStream failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = m_pDbSystem->getNextMetaphone( 
			pMetaphoneIStream, 
			&uiMetaphone1)))
		{
			MAKE_FLM_ERROR_STRING( "getNextMetaphone failed.", m_szDetails, rc);
			goto Exit;
		}
		
		pMetaphoneIStream->Release();
		pMetaphoneIStream = NULL;

		if ( RC_BAD( rc = m_pDbSystem->openBufferIStream( 
			commonMisspellings[uiLoop].pszWord2,
			f_strlen( commonMisspellings[uiLoop].pszWord2),
			&pMetaphoneIStream)))
		{
			MAKE_FLM_ERROR_STRING( "openBufferIStream failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = m_pDbSystem->getNextMetaphone( 
			pMetaphoneIStream, 
			&uiMetaphone2)))
		{
			MAKE_FLM_ERROR_STRING( "getNextMetaphone failed.", m_szDetails, rc);
			goto Exit;
		}
		
		pMetaphoneIStream->Release();
		pMetaphoneIStream = NULL;

		// No sense in testing the index if the metaphone algorithm itself yields 
		// different codes for these two words.

		if ( uiMetaphone1 == uiMetaphone2)
		{

			if ( (sizeof( m_szDetails) - f_strlen( m_szDetails)) > 
			(f_strlen( commonMisspellings[uiLoop].pszWord1) + 
				f_strlen( " vs. ") + 
						f_strlen( commonMisspellings[uiLoop].pszWord2) + 
							f_strlen( " ")))
			{
				f_strcat( m_szDetails, commonMisspellings[uiLoop].pszWord1);
				f_strcat( m_szDetails, " vs. ");
				f_strcat( m_szDetails, commonMisspellings[uiLoop].pszWord2);
				f_strcat( m_szDetails, " ");
			}

			if ( RC_BAD( rc = pSearchKey->setUTF8( 0, 
				(FLMBYTE *)commonMisspellings[uiLoop].pszWord2)))
			{
				MAKE_FLM_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
				goto Exit;
			}

			if ( RC_BAD( rc = m_pDb->keyRetrieve( 77, pSearchKey, XFLM_EXACT, pFoundKey)))
			{
				char szTemp[128];
				f_sprintf( szTemp, "\n\"%s\" indexed but cannot find \"%s\"!", 
					commonMisspellings[uiLoop].pszWord1, 
					commonMisspellings[uiLoop].pszWord2);

				MAKE_FLM_ERROR_STRING( "keyRetrieve failed. ", m_szDetails, rc);
				f_strcpy( m_szDetails, szTemp);

				goto Exit;
			}

			if ( RC_BAD( rc = pFoundKey->getUINT( 0, &uiIxMetaVal)))
			{
				MAKE_FLM_ERROR_STRING( "getUINT failed. ", m_szDetails, rc);
				goto Exit;
			}

			if ( uiIxMetaVal != uiMetaphone1)
			{
				rc = RC_SET( NE_XFLM_DATA_ERROR);
				MAKE_FLM_ERROR_STRING( "Unexpected metaphone code in index", m_szDetails, rc);
				goto Exit;
			}
		}
	}
	endTest("PASS");

Exit:

	if( RC_BAD( rc))
	{
		endTest("FAIL");
	}

	if( pMetaphoneIStream)
	{
		pMetaphoneIStream->Release();
	}

	if( pSearchKey)
	{
		pSearchKey->Release();
	}

	if( pFoundKey)
	{
		pFoundKey->Release();
	}

	if( pRoot)
	{
		pRoot->Release();
	}

	if( pChild)
	{
		pChild->Release();
	}

	if( bTransStarted)
	{
		m_pDb->transCommit();
	}

	shutdownTestState( DB_NAME_STR, bDibCreated);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IMetaphoneTestImpl::suite2( void)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DOMNode *		pRoot = NULL;
	IF_DOMNode *		pChild = NULL;
	FLMBOOL				bTransStarted = FALSE;
	FLMBOOL				bDibCreated = FALSE;
	FLMUINT				uiWordDictNum = 0;
	IF_DataVector *	pSearchKey = NULL;
	FLMUINT64			pui64DocIds[4];
	FLMUINT				uiLoop;
	FLMUINT				uiLoop2;
	const char *		pszIndexDef =
		"<xflaim:Index "
		"	xmlns:xflaim=\"http://www.novell.com/XMLDatabase/Schema\""
		"	xflaim:name=\"Metaphone Index\" "
		"	xflaim:DictNumber=\"77\"> " // HARD-CODED Dict Num for easy retrieval
		"	<xflaim:ElementComponent "
		"		xflaim:name=\"Word\" "
		"		xflaim:KeyComponent=\"1\" "
		"		xflaim:Required=\"1\" "
		"		xflaim:IndexOn=\"metaphone\"/> "
		"		xflaim:Limit=\"102\" /> "
		"</xflaim:Index> ";

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

	if( RC_BAD( rc = m_pDb->createElementDef( NULL, "Word", XFLM_TEXT_TYPE,
		&uiWordDictNum)))
	{
		MAKE_FLM_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Exit;
	}

	beginTest( 
		"Metaphone Missing Index Key Creation Test",
		"Ensure index keys created properly if node has no value.",
		"Self-explanatory",
		"");

	for( uiLoop = 0; uiLoop < sizeof(pui64DocIds)/sizeof(pui64DocIds[0]); uiLoop++)
	{
		//Create a node with a word attribute with a value of a correctly-spelled word
		if ( RC_BAD( rc = m_pDb->createRootElement(XFLM_DATA_COLLECTION, ELM_ELEMENT_TAG, &pRoot)))
		{
			MAKE_FLM_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
			goto Exit;
		}
		
		if( RC_BAD( rc = pRoot->getNodeId( m_pDb, &pui64DocIds[ uiLoop])))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pRoot->createNode( 
			m_pDb, ELEMENT_NODE, uiWordDictNum, XFLM_FIRST_CHILD, &pChild)))
		{
			MAKE_FLM_ERROR_STRING( "createNode failed.", m_szDetails, rc);
			goto Exit;
		}
	}

	if ( RC_BAD( rc = importBuffer( pszIndexDef, XFLM_DICT_COLLECTION)))
	{
		 goto Exit;
	}

	// Now, we should be able to locate every misspelled word in our list

	if ( RC_BAD( rc = m_pDbSystem->createIFDataVector( &pSearchKey)))
	{
		MAKE_FLM_ERROR_STRING( "createIFDataVector failed.", m_szDetails, rc);
		goto Exit;
	}

	for( uiLoop = 0; uiLoop < sizeof(pui64DocIds)/sizeof(pui64DocIds[0]); uiLoop++)
	{
		if( uiLoop == 0)
		{
			if ( RC_BAD( rc = m_pDb->keyRetrieve( 77, NULL, XFLM_FIRST, pSearchKey)))
			{
				MAKE_FLM_ERROR_STRING( "keyRetrieve failed.", m_szDetails, rc);
				goto Exit;
			}
		}
		else
		{
			rc = m_pDb->keyRetrieve( 
				77, pSearchKey, XFLM_EXCL | XFLM_MATCH_DOC_ID, pSearchKey);
		}

		for( uiLoop2 = 0; 
			uiLoop2 < sizeof(pui64DocIds)/sizeof(pui64DocIds[0]); 
			uiLoop2++)
		{
			if ( pui64DocIds[uiLoop2] == pSearchKey->getDocumentID())
			{
				// "tag" the document ID once we've found it
				pui64DocIds[uiLoop2] = 0;
				break;
			}
		}
	}

	// Make sure we got all the document ids

	for( uiLoop2 = 0; 
		uiLoop2 < sizeof(pui64DocIds)/sizeof(pui64DocIds[0]); 
		uiLoop2++)
	{
		if ( pui64DocIds[uiLoop2] != 0)
		{
			rc = RC_SET( NE_XFLM_DATA_ERROR);
			MAKE_FLM_ERROR_STRING( "Key iteration failed", m_szDetails, rc);
			goto Exit;
		}
	}

	rc = m_pDb->keyRetrieve( 
			77, pSearchKey, XFLM_EXCL | XFLM_MATCH_DOC_ID, pSearchKey);

	if ( rc == NE_XFLM_EOF_HIT)
	{
		rc = NE_XFLM_OK;
	}
	else
	{
		MAKE_FLM_ERROR_STRING( "Too many keys", m_szDetails, rc);
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		goto Exit;
	}
	
	endTest("PASS");

Exit:

	if ( RC_BAD( rc))
	{
		endTest("FAIL");
	}

	if ( pSearchKey)
	{
		pSearchKey->Release();
	}

	if( pRoot)
	{
		pRoot->Release();
	}

	if( pChild)
	{
		pChild->Release();
	}

	if ( bTransStarted)
	{
		m_pDb->transCommit();
	}

	shutdownTestState( DB_NAME_STR, bDibCreated);
	return( rc);
}
