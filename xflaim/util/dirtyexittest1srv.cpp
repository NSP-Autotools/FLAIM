//------------------------------------------------------------------------------
// Desc:	Dirty exit test 1
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
	#define DB_NAME_STR					"SYS:\\BAD.DB"
#else
	#define DB_NAME_STR					"bad.db"
#endif

/****************************************************************************
Desc:
****************************************************************************/
class IDirtyExitTest1Impl : public TestBase
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

	if( (*ppTest = f_new IDirtyExitTest1Impl) == NULL)
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
const char * IDirtyExitTest1Impl::getName( void)
{
	return( "Dirty Exit Test Part 1");
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IDirtyExitTest1Impl::execute( void)
{
	RCODE					rc = NE_XFLM_OK;
	FLMBOOL				bTransStarted = FALSE;
	IF_PosIStream *	pBufferIStream = NULL;
	const char *		pszDocument = 
		"<MP3FileDB>"
		"  <MP3FileDesc>"
		"    <File>01 Pull Me Under.mp3</File>"
		"    <Title>Pull Me Under</Title>"
		"    <Artist>Dream Theater</Artist>"
		"    <Album>Images &amp; Words</Album>"
		"    <Year>1992</Year>"
		"    <Genre>Rock</Genre>"
		"    <Track>1</Track>"
		"    <Comment />"
		"  </MP3FileDesc>"
		"  <MP3FileDesc>"
		"    <File>02 Another Day.mp3</File>"
		"    <Title>Another Day</Title>"
		"    <Artist>Dream Theater</Artist>"
		"    <Album>Images &amp; Words</Album>"
		"    <Year>1992</Year>"
		"    <Genre>Rock</Genre>"
		"    <Track>2</Track>"
		"    <Comment />"
		"  </MP3FileDesc>"
		"  <MP3FileDesc>"
		"    <File>03 Take the Time.mp3</File>"
		"    <Title>Take the Time</Title>"
		"    <Artist>Dream Theater</Artist>"
		"    <Album>Images &amp; Words</Album>"
		"    <Year>1992</Year>"
		"    <Genre>Rock</Genre>"
		"    <Track>3</Track>"
		"    <Comment />"
		"  </MP3FileDesc>"
		"  <MP3FileDesc>"
		"    <File>04 Surrounded.mp3</File>"
		"    <Title>Surrounded</Title>"
		"    <Artist>Dream Theater</Artist>"
		"    <Album>Images &amp; Words</Album>"
		"    <Year>1992</Year>"
		"    <Genre>Rock</Genre>"
		"    <Track>4</Track>"
		"    <Comment />"
		"  </MP3FileDesc>"
		"  <MP3FileDesc>"
		"    <File>05 Metropolis, Pt. 1.mp3</File>"
		"    <Title>Metropolis, Pt. 1</Title>"
		"    <Artist>Dream Theater</Artist>"
		"    <Album>Images &amp; Words</Album>"
		"    <Year>1992</Year>"
		"    <Genre>Rock</Genre>"
		"    <Track>5</Track>"
		"    <Comment />"
		"  </MP3FileDesc>"
		"  <MP3FileDesc>"
		"    <File>06 Under a Glass Moon.mp3</File>"
		"    <Title>Under a Glass Moon</Title>"
		"    <Artist>Dream Theater</Artist>"
		"    <Album>Images &amp; Words</Album>"
		"    <Year>1992</Year>"
		"    <Genre>Rock</Genre>"
		"    <Track>6</Track>"
		"    <Comment />"
		"  </MP3FileDesc>"
		"  <MP3FileDesc>"
		"    <File>07 Wait for Sleep.mp3</File>"
		"    <Title>Wait for Sleep</Title>"
		"    <Artist>Dream Theater</Artist>"
		"    <Album>Images &amp; Words</Album>"
		"    <Year>1992</Year>"
		"    <Genre>Rock</Genre>"
		"    <Track>7</Track>"
		"    <Comment />"
		"  </MP3FileDesc>"
		"  <MP3FileDesc>"
		"    <File>08 Learning to Live.mp3</File>"
		"    <Title>Learning to Live</Title>"
		"    <Artist>Dream Theater</Artist>"
		"    <Album>Images &amp; Words</Album>"
		"    <Year>1992</Year>"
		"    <Genre>Rock</Genre>"
		"    <Track>8</Track>"
		"    <Comment />"
		"  </MP3FileDesc>"
		"</MP3FileDB>";

	beginTest(
		"Init Test State",
		"Setup the necessary state for our test ",
		"(1) Get the DbSystem (2) Create a database",
		"No Additional Details.");

	if( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		MAKE_FLM_ERROR_STRING("Failed to open test state", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

	beginTest( 	
		"Import document test",
		"Import a document",
		"Import a document before the dirty shutdown",
		"");


	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_FLM_ERROR_STRING("Failed to begin trans.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = TRUE;

	if ( RC_BAD( rc = m_pDbSystem->openBufferIStream( 
		pszDocument, 
		f_strlen( pszDocument),
		&pBufferIStream)))
	{
		MAKE_FLM_ERROR_STRING("Failed to open file istream", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->import( pBufferIStream, XFLM_DATA_COLLECTION)))
	{
		MAKE_FLM_ERROR_STRING("Failed to import document", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

#ifdef FLM_USE_NICI
	{
		FLMUINT				uiDes3Def = 0;
		IF_DOMNode *		pRoot = NULL;
		IF_DOMNode *		pNode = NULL;

		beginTest( 	
			"Recover Encryption Test",
			"Make sure operations with encrypted values are being recovered properly",
			"(1) Create an encryption definition (2) Add an encrypted value "
			"(3) commit the trans (4) quit the program without calling exit()",
			"");

		if ( RC_BAD( rc = m_pDb->createEncDef(
			"des3",
			"des3 definition",
			0,
			&uiDes3Def)))
		{
			MAKE_FLM_ERROR_STRING("createEncDef failed", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = m_pDb->getFirstDocument( XFLM_DATA_COLLECTION, &pRoot)))
		{
			MAKE_FLM_ERROR_STRING("getFirstDocument", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = pRoot->createNode( m_pDb, ELEMENT_NODE, 
			ELM_DOCUMENT_TITLE_TAG, XFLM_FIRST_CHILD, &pNode)))
		{
			MAKE_FLM_ERROR_STRING("createNode", m_szDetails, rc);
			pRoot->Release();
			goto Exit;
		}

		rc = pNode->setUTF8( 
			m_pDb, (FLMBYTE *)"My big ol' native text value to be encrypted",
			0, TRUE, uiDes3Def);

		pRoot->Release();
		pNode->Release();

		if ( RC_BAD( rc))
		{
			MAKE_FLM_ERROR_STRING("setNative failed", m_szDetails, rc);
			goto Exit;
		}

		endTest("PASS");
	}
#endif

Exit:
	if ( bTransStarted)
	{
		RCODE		tmpRc = m_pDb->transCommit();
		if ( RC_OK( rc))
		{
			rc = tmpRc;
		}
	}

	if ( RC_BAD( rc))
	{
		endTest("FAIL");
	}

	// superficial cleanup. Purposely not calling exit on the dbsystem

	if ( pBufferIStream)
	{
		pBufferIStream->Release();
	}
	return rc;
}
