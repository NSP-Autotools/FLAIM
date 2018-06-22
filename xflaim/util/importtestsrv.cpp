//------------------------------------------------------------------------------
// Desc:	Import unit test
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
	#define DB_NAME_STR					"SYS:\\IMP.DB"
#else
	#define DB_NAME_STR					"imp.db"
#endif

/****************************************************************************
Desc:
****************************************************************************/
class IImportTestImpl : public TestBase
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

	if( (*ppTest = f_new IImportTestImpl) == NULL)
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
const char * IImportTestImpl::getName( void)
{
	return( "Import Test");
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IImportTestImpl::execute( void)
{
	RCODE					rc = NE_XFLM_OK;
	FLMBOOL				bDibCreated = FALSE;
	FLMBOOL				bTransStarted = FALSE;
	IF_DirHdl *			pDirHdl = NULL;
	IF_FileSystem *	pFileSystem = NULL;
	FLMUINT				uiCharCount = 0;
	char					szTemp[ 64];
	const char *		pszDoc1 = 
		"<?xml version=\"1.0\"?>"
			"<disc> "
			"<id>00054613</id> "
			"<length>1352</length> "
			"<title>Margret Birkenfeld / Zachaus</title> "
			"<genre>cddb/misc</genre> "
			"<ext> ID3G: 77</ext> "
			"<track index=\"1\" offset=\"150\">Wie erzahlen euch 1. Srophe </track> "
			"<track index=\"2\" offset=\"13065\">Wir erzahlen Euch 2. Strophe </track> "
			"<track index=\"3\" offset=\"14965\">Zachaus ist ein reicher Mann 1+2 Str</track> "
			"<track index=\"4\" offset=\"19980\">Jericho</track> "
			"<track index=\"5\" offset=\"28122\">Haruck, schnauf schnauf 1+2 Strophe</track> "
			"<track index=\"6\" offset=\"33630\">Haruck, schnauf schnauf 3 Strophe</track> "
			"<track index=\"7\" offset=\"37712\">Zachaus ist ein reicher Mann 3. Stophe</track> "
			"<track index=\"8\" offset=\"41502\">Zachaus komm herunter!</track> "
			"<track index=\"9\" offset=\"57627\">Wir erzahlen euch</track> "
			"<track index=\"10\" offset=\"63145\">Leer ab jetzt Playback</track> "
			"<track index=\"11\" offset=\"65687\">Wie erzahlen euch 1. Srophe Pb</track> "
			"<track index=\"12\" offset=\"69212\">Wir erzahlen Euch 2. Strophe Pb</track> "
			"<track index=\"13\" offset=\"71102\">Zachaus ist ein reicher Mann 1+2 Str Pb</track> "
			"<track index=\"14\" offset=\"75622\">Jericho Pb</track> "
			"<track index=\"15\" offset=\"82292\">Haruck, schnauf schnauf 1+2 Strophe Pb</track> "
			"<track index=\"16\" offset=\"86555\">Haruck, schnauf schnauf 3 Strophe Pb</track> "
			"<track index=\"17\" offset=\"89887\">Zachaus ist ein reicher Mann 3. Stophe Pb</track> "
			"<track index=\"18\" offset=\"93067\">Zachaus komm herunter! Pb</track> "
			"<track index=\"19\" offset=\"97797\">Wir erzahlen euch Pb</track> "
			"</disc> ";

const char * pszDoc2 = "<?xml version=\"1.0\"?> "
		"<disc> "
		"<id>0008a40f</id> "
		"<length>2214</length> "
		"<title>rundu... - Visur Ur Vinsabokinni</title> "
		"<genre>cddb/misc</genre> "
		"<track index=\"1\" offset=\"150\">Blessuo Solin Elskar Allt - Ur Augum Stirur Strjukio Fljott</track> "
		"<track index=\"2\" offset=\"13855\">Heioloarkvaeoi</track> "
		"<track index=\"3\" offset=\"27576\">Buxur, Vesti, Brok og Sko</track> "
		"<track index=\"4\" offset=\"33311\">Gekk Eg Upp A Holinn</track>"
		"<track index=\"5\" offset=\"45340\">Nu Blanar Yfir Berjamo - A Berjamo</track> "
		"<track index=\"6\" offset=\"59209\">Orninn Flygur Fugla Haest - Solskrikjan - Min</track> "
		"<track index=\"7\" offset=\"64309\">Nu Er Glatt I Borg Og Bae</track> "
		"<track index=\"8\" offset=\"73018\">Smaladrengurinn - Klappa Saman Lofunum</track> "
		"<track index=\"9\" offset=\"89149\">Stigur Hun Vio Stokkinn</track> "
		"<track index=\"10\" offset=\"91370\">Dansi, Dansi, Dukkan Min</track> "
		"<track index=\"11\" offset=\"104540\">Rioum Heim Til Hola - Gott Er Ao Rioa Sandana Mjuka</track> "
		"<track index=\"12\" offset=\"119232\">Gryla - Jolasveinar Ganga Um Golf</track> "
		"<track index=\"13\" offset=\"133837\">Erla, Gooa Erla</track> "
		"<track index=\"14\" offset=\"146208\">Vio Skulum Ekki Hafa Hatt</track> "
		"<track index=\"15\" offset=\"149899\">Sofa Urtu Born</track> "
		"</disc> ";

		
const char * pszIndexDef1 = "<xflaim:Index "
		"	xmlns:xflaim=\"http://www.novell.com/XMLDatabase/Schema\""
		"	xflaim:name=\"Title+Index+Offset\" "
		"	xflaim:DictNumber=\"1\"> "
		"	<xflaim:ElementComponent "
		"		xflaim:name=\"title\" "
		"		xflaim:KeyComponent=\"1\" "
		"		xflaim:IndexOn=\"value\"/> "
		"	<xflaim:ElementComponent "
		"		xflaim:name=\"track\"> "
		"		<xflaim:AttributeComponent "
		"		  xflaim:name=\"index\" "
		"		  xflaim:KeyComponent=\"2\" "
		"		  xflaim:IndexOn=\"value\"/> "
		"		<xflaim:AttributeComponent "
		"		  xflaim:name=\"offset\" "
		"		  xflaim:KeyComponent=\"3\" "
		"		  xflaim:IndexOn=\"value\"/> "
		"	</xflaim:ElementComponent> "
		"</xflaim:Index> ";

	f_strcpy( m_szDetails, "No additional details.");

	beginTest(	
		"Import Test Init",
		"Perform necessary initializations to carry out the import tests.",
		"(1) Get an F_DbSystem (2) Create a database (3) Start an "
		"update transaction.",
		"");

	if ( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		MAKE_FLM_ERROR_STRING( "Failed to init test state.", m_szDetails, rc);
		goto Exit;
	}
	bDibCreated = TRUE;

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;

	endTest("PASS");

	beginTest( 
		"Document Buffer Import Test",
		"Import some documents from in-memory XML data",
		"Self-Explanatory",
		"");

	if( RC_BAD( rc = importBuffer( pszDoc1, XFLM_DATA_COLLECTION)))
	{
		 goto Exit;
	}

	if( RC_BAD( rc = importBuffer( pszDoc2, XFLM_DATA_COLLECTION)))
	{
		 goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_FLM_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;

	endTest("PASS");

	beginTest( 
		"Document File Import Test",
		"Import some documents from XML data files",
		"Import all .xml files in the current directory.",
		"");

	m_pDbSystem->getFileSystem( &pFileSystem);

	if( RC_BAD( rc = pFileSystem->openDir( ".", ".xml", &pDirHdl)))
	{
		MAKE_FLM_ERROR_STRING( "OpenDir failed.", m_szDetails, rc);
		goto Exit;
	}

	m_szDetails[ 0] = '\0';
	for (;;)
	{
		if( RC_BAD( rc = pDirHdl->next()))
		{
			if ( rc == NE_XFLM_IO_NO_MORE_FILES)
			{
				rc = NE_XFLM_OK;
				break;
			}
			
			MAKE_FLM_ERROR_STRING("F_DirHdl::Next failed", m_szDetails, rc);
			goto Exit;
		}

		if( RC_BAD( rc = importFile( pDirHdl->currentItemName(), 
			XFLM_DATA_COLLECTION)))
		{
			goto Exit;
		}

		f_sprintf( szTemp, "Imported: %s. ", pDirHdl->currentItemName());
		uiCharCount += f_strlen( szTemp);

		if( uiCharCount < DETAILS_BUF_SIZ)
		{
			f_strcat( m_szDetails, szTemp);
		}
	}

	endTest("PASS");

	beginTest( 	
		"Index Definition Import Test",
		"Import an index definition.",
		"Self-explanatory",
		"No Additional Info");

	if( RC_BAD( rc = importBuffer( pszIndexDef1, XFLM_DICT_COLLECTION)))
	{
		 goto Exit;
	}

	endTest("PASS");

Exit:

	if( pDirHdl)
	{
		pDirHdl->Release();
	}

	if( pFileSystem)
	{
		pFileSystem->Release();
	}

	if( RC_BAD( rc))
	{
		endTest("FAIL");
	}
	
	if( bTransStarted)
	{
		m_pDb->transCommit();
	}
	
	shutdownTestState( DB_NAME_STR, bDibCreated);
	return( rc);
}
