//------------------------------------------------------------------------------
// Desc:	Encryption tests
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
	#define DB_NAME_STR					"SYS:\\ENC.DB"
	#define BACKUP_NAME_STR				"SYS:\\ENC.BAK"
#else
	#define DB_NAME_STR					"enc.db"
	#define BACKUP_NAME_STR				"enc.bak"
#endif

#define PASSWORD							"password"
#define BAD_PASSWORD						"bad_password"
#define BACKUP_PASSWORD					"backup_password"
#define UINT_VAL							1234
#define INT_VAL							-1234
#define UINT64_VAL						(FLMUINT64)5000000
#define INT64_VAL							(FLMINT64)-5000000
#define NATIVE_VAL						"native_val"
#define BIN_VAL							((FLMBYTE *)"\x01\x02\x03")
#define BIN_VAL_LEN						3

/****************************************************************************
Desc:
****************************************************************************/
class IEncTestImpl : public TestBase
{
public:

	IEncTestImpl();
	~IEncTestImpl();

	const char * getName( void);
	
	RCODE	suite1( void);
	
	RCODE suite2( void);
	
	RCODE execute( void);
	
private:

	FLMUINT			m_uiNumberElm;
	FLMUINT			m_uiTextElm;
	FLMUINT			m_uiBinaryElm;
	FLMUINT			m_uiNoDataElm;
	FLMUINT			m_uiNumberAttr;
	FLMUINT			m_uiTextAttr;
	FLMUINT			m_uiBinaryAttr;
	IF_DOMNode *	m_pRootNode;
	IF_DOMNode *	m_pNumberElmNode;
	IF_DOMNode *	m_pTextElmNode;
	IF_DOMNode *	m_pBinElmNode;
	IF_DOMNode *	m_pAttributedNode;
	IF_DOMNode *	m_pNumberAttrNode;
	IF_DOMNode *	m_pTextAttrNode;
	IF_DOMNode *	m_pBinAttrNode;
	IF_Backup *		m_pBackup;
	FLMUINT			m_uiAESDef;
	FLMUINT			m_uiDES3Def;

	RCODE importEncDefs( void);
	
	RCODE setupElemAndAttrNodes( void);
};

/****************************************************************************
Desc:
****************************************************************************/
RCODE getTest( IFlmTest ** ppTest)
{
	RCODE		rc = NE_XFLM_OK;

	if( (*ppTest = f_new IEncTestImpl) == NULL)
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
const char * IEncTestImpl::getName( void)
{
	return( "Encryption Test");
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IEncTestImpl::setupElemAndAttrNodes( void)
{
	RCODE		rc = NE_XFLM_OK;

	if( RC_BAD( rc = m_pDb->createElementDef( NULL, "generic_nodata_element",
		XFLM_NODATA_TYPE, &m_uiNoDataElm)))
	{
		MAKE_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->createRootElement( XFLM_DATA_COLLECTION,
		m_uiNoDataElm, &m_pRootNode)))
	{
		MAKE_ERROR_STRING( "createRootNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->createElementDef( NULL, "generic_number_element",
		XFLM_NUMBER_TYPE, &m_uiNumberElm)))
	{
		MAKE_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pRootNode->createNode( m_pDb, ELEMENT_NODE,
		m_uiNumberElm, XFLM_LAST_CHILD, &m_pNumberElmNode)))
	{
		MAKE_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->createElementDef( NULL, "generic_text_element",
		XFLM_TEXT_TYPE, &m_uiTextElm)))
	{
		MAKE_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pRootNode->createNode( m_pDb, ELEMENT_NODE,
		m_uiTextElm, XFLM_LAST_CHILD, &m_pTextElmNode)))
	{
		MAKE_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->createElementDef( NULL, "generic_binary_element",
		XFLM_BINARY_TYPE, &m_uiBinaryElm)))
	{
		MAKE_ERROR_STRING( "createElementDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pRootNode->createNode( m_pDb, ELEMENT_NODE,
		m_uiBinaryElm, XFLM_LAST_CHILD, &m_pBinElmNode)))
	{
		MAKE_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	// Create the element node to hang all the attribute nodes off of

	if( RC_BAD( rc = m_pRootNode->createNode( m_pDb, ELEMENT_NODE,
		m_uiNoDataElm, XFLM_LAST_CHILD, &m_pAttributedNode)))
	{
		MAKE_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->createAttributeDef( NULL, "generic_number_attr",
		XFLM_NUMBER_TYPE, &m_uiNumberAttr)))
	{
		MAKE_ERROR_STRING( "createAttributeDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pAttributedNode->createAttribute( 		
		m_pDb, m_uiNumberAttr, &m_pNumberAttrNode)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->createAttributeDef( NULL, "generic_text_attr",
		XFLM_TEXT_TYPE, &m_uiTextAttr)))
	{
		MAKE_ERROR_STRING( "createAttributeDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pAttributedNode->createAttribute( m_pDb,
		m_uiTextAttr, &m_pTextAttrNode)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->createAttributeDef( NULL, "generic_binary_attr",
		XFLM_BINARY_TYPE, &m_uiBinaryAttr)))
	{
		MAKE_ERROR_STRING( "createAttributeDef failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pAttributedNode->createAttribute( m_pDb,
		m_uiBinaryAttr, &m_pBinAttrNode)))
	{
		MAKE_ERROR_STRING( "createAttribute failed.", m_szDetails, rc);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
IEncTestImpl::IEncTestImpl( void)
{
	m_uiNumberElm = 0;
	m_uiTextElm = 0;
	m_uiBinaryElm = 0;
	m_uiNoDataElm = 0;
	m_uiNumberAttr = 0;
	m_uiTextAttr = 0;
	m_uiBinaryAttr = 0;
	m_pRootNode = NULL;
	m_pNumberElmNode = NULL;
	m_pTextElmNode = NULL;
	m_pBinElmNode = NULL;
	m_pAttributedNode = NULL;
	m_pNumberAttrNode = NULL;
	m_pTextAttrNode = NULL;
	m_pBinAttrNode = NULL;
	m_pBackup = NULL;
	m_uiAESDef = 0;
	m_uiDES3Def = 0;
}

/****************************************************************************
Desc:
****************************************************************************/
IEncTestImpl::~IEncTestImpl( void)
{
	if( m_pRootNode)
	{
		m_pRootNode->Release();
	}

	if( m_pNumberElmNode)
	{
		m_pNumberElmNode->Release();
	}
	
	if( m_pTextElmNode)
	{
		m_pTextElmNode->Release();
	}

	if( m_pBinElmNode)
	{
		m_pBinElmNode->Release();
	}

	if( m_pAttributedNode)
	{
		m_pAttributedNode->Release();
	}
	
	if( m_pNumberAttrNode)
	{
		m_pNumberAttrNode->Release();
	}
	
	if( m_pTextAttrNode)
	{
		m_pTextAttrNode->Release();
	}

	if( m_pBinAttrNode)
	{
		m_pBinAttrNode->Release();
	}

	if( m_pBackup)
	{
		m_pBackup->Release();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IEncTestImpl::importEncDefs( void)
{
	RCODE			rc = NE_XFLM_OK;
	const char *ppszEncDefs[] = {XFLM_ENC_AES_OPTION_STR, XFLM_ENC_DES3_OPTION_STR};
	FLMUINT		puiEncDef[ 2];
	char			szEncDef[ 200];
	FLMUINT		uiLoop = 0;

	for( 
		uiLoop = 0; 
		uiLoop < sizeof(ppszEncDefs)/sizeof(ppszEncDefs[0]); 
		uiLoop++)
	{
		f_sprintf( 
			szEncDef, 
			"<xflaim:EncDef "
			"xmlns:xs=\"http://www.w3.org/2001/XMLSchema\" "
			"xmlns:xflaim=\"http://www.novell.com/XMLDatabase/Schema\" "
			"xflaim:name=\"%s definition\" "
			"xflaim:type=\"%s\" />", 
			ppszEncDefs[uiLoop], 
			ppszEncDefs[uiLoop]);

		if ( RC_BAD( rc = importBuffer( szEncDef, XFLM_DICT_COLLECTION)))
		{
			MAKE_ERROR_STRING( "importBuffer failed.", m_szDetails, rc);
			goto Exit;
		}
		
		f_sprintf( szEncDef, "%s definition", ppszEncDefs[uiLoop]);
		if (RC_BAD( rc = m_pDb->getEncDefId( (char *)szEncDef,
														 (FLMUINT *)&puiEncDef[ uiLoop])))
		{
			goto Exit;
		}
	}

	m_uiAESDef = puiEncDef[ 0];
	m_uiDES3Def = puiEncDef[ 1];

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IEncTestImpl::execute( void)
{
	RCODE		rc = NE_XFLM_OK;

	if( RC_BAD( rc = suite1()))
	{
		goto Exit;
	}

	if( RC_BAD( rc = suite2()))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IEncTestImpl::suite1( void)
{
	RCODE				rc = NE_XFLM_OK;
#ifdef FLM_USE_NICI
	FLMBOOL			bDibCreated = FALSE;
	FLMBOOL			bTransBegun = FALSE;
	FLMUINT			puiEncDefs[ 4];
	char *			ppszEncDefs[ 4];
	FLMUINT			uiResult = 0;
	FLMUINT64		ui64Result = 0;
	FLMINT			iResult = 0;
	FLMINT64			i64Result = 0;
	char				szTemp[ 128];
	FLMUINT			uiLoop;
	FLMUINT			uiLoop2;
	char *			ppszTextSizes[] = {"Medium","Large","Small"};
	char *			ppszNativeValues[] = 
		{"MediumMedium","LargeLargeLargeLargeLargeLargeLarge","Small"};

	FLMUNICODE		puzMedium[] = 
		{'M','e','d','i','u','m','M','e','d','i','u','m','\0'};
	FLMUNICODE		puzLarge[] = 
		{'L','a','r','g','e','L','a','r','g','e','L','a','r','g','e', 'L','a','r','g','e','\0'};
	FLMUNICODE		puzSmall[] =
		{'S','m','a','l','l','\0'};

	FLMUNICODE *	ppuzUnicodeValues[3];
	FLMBYTE	* 		ppucBinaryValues[3];

	FLMBYTE			pucMedium[] = {0x01,0x02,0x03};
	FLMBYTE			pucLarge[] = {0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C};
	FLMBYTE			pucSmall[] = {0x0D,0x0E,0x0F,0x10,0x11};
	FLMUINT			uiBinaryLen;
	FLMUNICODE *	puzResult = NULL;
	FLMBYTE			pucResult[100];
	char *			pszResult = NULL;
	FLMBYTE			pucUTF8Result[100];
	FLMUINT			uiBufSize;

	ppuzUnicodeValues[ 0] = puzMedium;
	ppuzUnicodeValues[ 1] = puzLarge;
	ppuzUnicodeValues[ 2] = puzSmall;

	ppucBinaryValues[ 0] = pucMedium;
	ppucBinaryValues[ 1] = pucLarge;
	ppucBinaryValues[ 2] = pucSmall;

	beginTest(	
		"Encryption test setup", 
		"Creates all the necessary database objects to run the encryption tests",
		"Create the database/create necessary element and attribute definitions/"
		"create encryption definitions",
		"none");
	
	if( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		goto Exit;
	}
	bDibCreated = TRUE;

	// Set up element and attribute definitions of all data types

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	if ( RC_BAD( rc = setupElemAndAttrNodes()))
	{
		goto Exit;
	}

	if ( RC_BAD( rc = importEncDefs()))
	{
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->transCommit( )))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	endTest("PASS");

	puiEncDefs[0] = m_uiAESDef;
	puiEncDefs[1] = m_uiDES3Def;
	puiEncDefs[2] = 0; // No encryption;

	ppszEncDefs[0] = XFLM_ENC_AES_OPTION_STR;
	ppszEncDefs[1] = XFLM_ENC_DES3_OPTION_STR;
	ppszEncDefs[2] = "none";

	for ( uiLoop = 0; uiLoop < 3; uiLoop++)
	{

		/***********************************************************************/

		f_sprintf( szTemp, 
			"setAttributeValueUINT Encrypted Value Test (encryption=%s)",
			ppszEncDefs[uiLoop]);

		beginTest(	
		szTemp, 
		"Tests setAttributeValueUINT API using encryption definitions",
		"Self-explanatory",
		"");

		if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
		{
			MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
			goto Exit;
		}
		bTransBegun = TRUE;

		if ( RC_BAD( rc = m_pAttributedNode->setAttributeValueUINT(
			m_pDb,
			m_uiNumberAttr,
			UINT_VAL,
			puiEncDefs[uiLoop])))
		{
			MAKE_ERROR_STRING( "setAttributeValueUINT failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = m_pDb->transCommit( )))
		{
			MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
			goto Exit;
		}
		bTransBegun = FALSE;

		if ( RC_BAD( rc = m_pAttributedNode->getAttributeValueUINT(
			m_pDb,
			m_uiNumberAttr,
			&uiResult)))
		{
			MAKE_ERROR_STRING( "getAttributeValueUINT failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( uiResult != UINT_VAL)
		{
			rc = NE_XFLM_DATA_ERROR;
			MAKE_ERROR_STRING( "UINT value corruption detected.", m_szDetails, rc);
			goto Exit;
		}

		endTest("PASS");

		/***********************************************************************/

		f_sprintf( szTemp, 
			"setAttributeValueUINT64 Encrypted Value Test (encryption=%s)",
			ppszEncDefs[uiLoop]);

		beginTest(	
		szTemp, 
		"Tests setAttributeValueUINT64 API using encryption definitions",
		"Self-explanatory",
		"");

		if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
		{
			MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
			goto Exit;
		}
		bTransBegun = TRUE;

		if ( RC_BAD( rc = m_pAttributedNode->setAttributeValueUINT64(
			m_pDb,
			m_uiNumberAttr,
			UINT64_VAL,
			puiEncDefs[uiLoop])))
		{
			MAKE_ERROR_STRING( "setAttributeValueUINT64 failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = m_pDb->transCommit( )))
		{
			MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
			goto Exit;
		}
		bTransBegun = FALSE;

		if ( RC_BAD( rc = m_pAttributedNode->getAttributeValueUINT64(
			m_pDb,
			m_uiNumberAttr,
			&ui64Result)))
		{
			MAKE_ERROR_STRING( "getAttributeValueUINT64 failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( ui64Result != UINT64_VAL)
		{
			rc = NE_XFLM_DATA_ERROR;
			MAKE_ERROR_STRING( "UINT64 value corruption detected.", m_szDetails, rc);
			goto Exit;
		}

		endTest("PASS");

		/***********************************************************************/

		f_sprintf( szTemp, 
			"setAttributeValueINT Encrypted Value Test (encryption=%s)",
			ppszEncDefs[uiLoop]);

		beginTest(	
		szTemp, 
		"Tests setAttributeValueINT API using encryption definitions",
		"Self-explanatory",
		"");

		if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
		{
			MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
			goto Exit;
		}
		bTransBegun = TRUE;

		if ( RC_BAD( rc = m_pAttributedNode->setAttributeValueINT(
			m_pDb,
			m_uiNumberAttr,
			INT_VAL,
			puiEncDefs[uiLoop])))
		{
			MAKE_ERROR_STRING( "setAttributeValueINT failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = m_pDb->transCommit( )))
		{
			MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
			goto Exit;
		}
		bTransBegun = FALSE;

		if ( RC_BAD( rc = m_pAttributedNode->getAttributeValueINT(
			m_pDb,
			m_uiNumberAttr,
			&iResult)))
		{
			MAKE_ERROR_STRING( "getAttributeValueINT failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( iResult != INT_VAL)
		{
			rc = NE_XFLM_DATA_ERROR;
			MAKE_ERROR_STRING( "INT value corruption detected.", m_szDetails, rc);
			goto Exit;
		}

		endTest("PASS");

		/***********************************************************************/

		f_sprintf( szTemp, 
			"setAttributeValueINT64 Encrypted Value Test (encryption=%s)",
			ppszEncDefs[uiLoop]);

		beginTest(	
		szTemp, 
		"Tests setAttributeValueINT64 API using encryption definitions",
		"Self-explanatory",
		"");

		if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
		{
			MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
			goto Exit;
		}
		bTransBegun = TRUE;

		if ( RC_BAD( rc = m_pAttributedNode->setAttributeValueINT64(
			m_pDb,
			m_uiNumberAttr,
			INT64_VAL,
			puiEncDefs[uiLoop])))
		{
			MAKE_ERROR_STRING( "setAttributeValueINT64 failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = m_pDb->transCommit( )))
		{
			MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
			goto Exit;
		}
		bTransBegun = FALSE;

		if ( RC_BAD( rc = m_pAttributedNode->getAttributeValueINT64(
			m_pDb,
			m_uiNumberAttr,
			&i64Result)))
		{
			MAKE_ERROR_STRING( "getAttributeValueINT64 failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( i64Result != INT64_VAL)
		{
			rc = NE_XFLM_DATA_ERROR;
			MAKE_ERROR_STRING( "INT64 value corruption detected.", m_szDetails, rc);
			goto Exit;
		}

		endTest("PASS");

		/***********************************************************************/

		f_sprintf( szTemp, 
			"setUINT Encrypted Value Test (encryption=%s)",
			ppszEncDefs[uiLoop]);

		beginTest(	
		szTemp, 
		"Tests setUINT API using encryption definitions",
		"Self-explanatory",
		"");

		// "Direct" setter tests

		if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
		{
			MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
			goto Exit;
		}
		bTransBegun = TRUE;

		if ( RC_BAD( rc = m_pNumberElmNode->setUINT(
			m_pDb,
			UINT_VAL)))
		{
			MAKE_ERROR_STRING( "setUINT failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = m_pDb->transCommit( )))
		{
			MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
			goto Exit;
		}
		bTransBegun = FALSE;

		if ( RC_BAD( rc = m_pNumberElmNode->getUINT(
			m_pDb,
			&uiResult)))
		{
			MAKE_ERROR_STRING( "getUINT failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( uiResult != UINT_VAL)
		{
			rc = NE_XFLM_DATA_ERROR;
			MAKE_ERROR_STRING( "UINT value corruption detected.", m_szDetails, rc);
			goto Exit;
		}

		endTest("PASS");

		/***********************************************************************/

		f_sprintf( szTemp, 
			"setUINT64 Encrypted Value Test (encryption=%s)",
			ppszEncDefs[uiLoop]);

		beginTest(	
			szTemp, 
			"Tests setUINT64 API using encryption definitions",
			"Self-explanatory",
			"");

		if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
		{
			MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
			goto Exit;
		}
		bTransBegun = TRUE;

		if ( RC_BAD( rc = m_pNumberElmNode->setUINT64(
			m_pDb,
			UINT64_VAL)))
		{
			MAKE_ERROR_STRING( "setUINT64 failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = m_pDb->transCommit( )))
		{
			MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
			goto Exit;
		}
		bTransBegun = FALSE;

		if ( RC_BAD( rc = m_pNumberElmNode->getUINT64(
			m_pDb,
			&ui64Result)))
		{
			MAKE_ERROR_STRING( "getUINT64 failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( ui64Result != UINT64_VAL)
		{
			rc = NE_XFLM_DATA_ERROR;
			MAKE_ERROR_STRING( "UINT64 value corruption detected.", m_szDetails, rc);
			goto Exit;
		}

		endTest("PASS");

		/***********************************************************************/

		f_sprintf( szTemp, 
			"setINT Encrypted Value Test (encryption=%s)",
			ppszEncDefs[uiLoop]);

		beginTest(	
			szTemp, 
			"Tests setINT API using encryption definitions",
			"Self-explanatory",
			"");

		if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
		{
			MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
			goto Exit;
		}
		bTransBegun = TRUE;

		if ( RC_BAD( rc = m_pNumberElmNode->setINT(
			m_pDb,
			INT_VAL)))
		{
			MAKE_ERROR_STRING( "setINT failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = m_pDb->transCommit( )))
		{
			MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
			goto Exit;
		}
		bTransBegun = FALSE;

		if ( RC_BAD( rc = m_pNumberElmNode->getINT(
			m_pDb,
			&iResult)))
		{
			MAKE_ERROR_STRING( "getINT failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( iResult != INT_VAL)
		{
			rc = NE_XFLM_DATA_ERROR;
			MAKE_ERROR_STRING( "INT value corruption detected.", m_szDetails, rc);
			goto Exit;
		}

		endTest("PASS");

		/***********************************************************************/

		f_sprintf( szTemp, 
			"setINT64 Encrypted Value Test (encryption=%s)",
			ppszEncDefs[uiLoop]);

		beginTest(	
			szTemp, 
			"Tests setINT64 API using encryption definitions",
			"Self-explanatory",
			"");

		if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
		{
			MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
			goto Exit;
		}
		bTransBegun = TRUE;

		if ( RC_BAD( rc = m_pNumberElmNode->setINT64(
			m_pDb,
			INT64_VAL)))
		{
			MAKE_ERROR_STRING( "setINT64 failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = m_pDb->transCommit( )))
		{
			MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
			goto Exit;
		}
		bTransBegun = FALSE;

		if ( RC_BAD( rc = m_pNumberElmNode->getINT64(
			m_pDb,
			&i64Result)))
		{
			MAKE_ERROR_STRING( "getINT64 failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( i64Result != INT64_VAL)
		{
			rc = NE_XFLM_DATA_ERROR;
			MAKE_ERROR_STRING( "INT64 value corruption detected.", m_szDetails, rc);
			goto Exit;
		}

		endTest("PASS");

		/******************************text value tests*************************/

		for ( 
			uiLoop2 = 0; 
			uiLoop2 < sizeof(ppszNativeValues)/sizeof(ppszNativeValues[0]);
			uiLoop2++)
		{
			uiBinaryLen = (uiLoop2 == 0)
									? sizeof(pucMedium)
									: (uiLoop2 == 1)
										? sizeof(pucLarge)
										: sizeof(pucSmall);


			/***********************************************************************/

			f_sprintf( szTemp, 
				"setAttributeValueUnicode Encrypted Value Test (encryption=%s/value size=%s)",
				ppszEncDefs[uiLoop],
				ppszTextSizes[uiLoop2]);

			beginTest(	
				szTemp, 
				"Tests setAttributeValueUnicode API using encryption definitions",
				"Self-explanatory",
				"");

			if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS, 
				FLM_NO_TIMEOUT)))
			{
				MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
				goto Exit;
			}
			bTransBegun = TRUE;
	
			if ( RC_BAD( rc = m_pAttributedNode->setAttributeValueUnicode(
				m_pDb,
				m_uiTextAttr,
				ppuzUnicodeValues[uiLoop2],
				puiEncDefs[uiLoop])))
			{
				MAKE_ERROR_STRING( "setAttributeValueUnicode failed.", 
					m_szDetails, rc);
				goto Exit;
			}

			if ( RC_BAD( rc = m_pDb->transCommit( )))
			{
				MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
				goto Exit;
			}
			bTransBegun = FALSE;

			if ( puzResult)
			{
				f_free( &puzResult);
			}
	
			if ( RC_BAD( rc = m_pAttributedNode->getAttributeValueUnicode(
				m_pDb,
				m_uiTextAttr,
				&puzResult)))
			{
				MAKE_ERROR_STRING( "getAttributeValueUnicode failed.", 
					m_szDetails, rc);
				goto Exit;
			}

			if ( f_unicmp( puzResult, ppuzUnicodeValues[uiLoop2]) != 0)
			{
				rc = NE_XFLM_DATA_ERROR;
				MAKE_ERROR_STRING( "unicode value corruption detection.", 
					m_szDetails, rc);
				goto Exit;
			}

			endTest("PASS");

			/***********************************************************************/

			f_sprintf( szTemp, 
				"setAttributeValueBinary Encrypted Value Test (encryption=%s/value size=%s)",
				ppszEncDefs[uiLoop],
				ppszTextSizes[uiLoop2]);

			beginTest(	
				szTemp, 
				"Tests setAttributeValueBinary API using encryption definitions",
				"Self-explanatory",
				"");

			if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS, 
				FLM_NO_TIMEOUT)))
			{
				MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
				goto Exit;
			}
			bTransBegun = TRUE;
	
			if ( RC_BAD( rc = m_pAttributedNode->setAttributeValueBinary(
				m_pDb,
				m_uiBinaryAttr,
				ppucBinaryValues[uiLoop2],
				uiBinaryLen,
				puiEncDefs[uiLoop])))
			{
				MAKE_ERROR_STRING( "setAttributeValueBinary failed.", 
					m_szDetails, rc);
				goto Exit;
			}

			if ( RC_BAD( rc = m_pDb->transCommit( )))
			{
				MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
				goto Exit;
			}
			bTransBegun = FALSE;
	
			if ( RC_BAD( rc = m_pAttributedNode->getAttributeValueBinary(
				m_pDb,m_uiBinaryAttr, pucResult,	uiBinaryLen, &uiBufSize)))
			{
				MAKE_ERROR_STRING( "getAttributeValueBinary failed.", 
					m_szDetails, rc);
				goto Exit;
			}

			flmAssert( uiBufSize == uiBinaryLen);

			if ( f_memcmp( pucResult, ppucBinaryValues[uiLoop2], uiBinaryLen) != 0)
			{
				rc = NE_XFLM_DATA_ERROR;
				MAKE_ERROR_STRING( "Binary value corruption detected.", 
					m_szDetails, rc);
				goto Exit;
			}

			endTest("PASS");

			/***********************************************************************/

			f_sprintf( szTemp, 
				"setAttributeValueUTF8 Encrypted Value Test (encryption=%s/value size=%s)",
				ppszEncDefs[uiLoop],
				ppszTextSizes[uiLoop2]);

			beginTest(	
				szTemp, 
				"Tests setAttributeValueUTF8 API using encryption definitions",
				"Self-explanatory",
				"");

			if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS, 
				FLM_NO_TIMEOUT)))
			{
				MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
				goto Exit;
			}
			bTransBegun = TRUE;
	
			if ( RC_BAD( rc = m_pAttributedNode->setAttributeValueUTF8(
				m_pDb,
				m_uiTextAttr,
				(FLMBYTE *)ppszNativeValues[uiLoop2],
				strlen(ppszNativeValues[uiLoop2]),
				puiEncDefs[uiLoop])))
			{
				MAKE_ERROR_STRING( "setAttributeValueUTF8 failed.", 
					m_szDetails, rc);
				goto Exit;
			}

			if ( RC_BAD( rc = m_pDb->transCommit( )))
			{
				MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
				goto Exit;
			}
			bTransBegun = FALSE;
	
			uiBufSize = sizeof(pucUTF8Result);
			if ( RC_BAD( rc = m_pAttributedNode->getAttributeValueUTF8(
				m_pDb,
				m_uiTextAttr,
				pucUTF8Result,
				uiBufSize,
				&uiBufSize)))
			{
				MAKE_ERROR_STRING( "getAttributeValueUTF8 failed.", 
					m_szDetails, rc);
				goto Exit;
			}

			if ( f_strcmp( (char *)pucUTF8Result, ppszNativeValues[uiLoop2]) != 0)
			{
				rc = NE_XFLM_DATA_ERROR;
				MAKE_ERROR_STRING( "UTF8 value corruption detected", 
					m_szDetails, rc);
				goto Exit;
			}

			endTest("PASS");

			/***********************************************************************/

			f_sprintf( szTemp, 
				"setUnicode Encrypted Value Test (encryption=%s/value size=%s)",
				ppszEncDefs[uiLoop],
				ppszTextSizes[uiLoop2]);

			beginTest(	
				szTemp, 
				"Tests setUnicode API using encryption definitions",
				"Self-explanatory",
				"");

			if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS, 
				FLM_NO_TIMEOUT)))
			{
				MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
				goto Exit;
			}
			bTransBegun = TRUE;
	
			if ( RC_BAD( rc = m_pTextElmNode->setUnicode(
				m_pDb,
				ppuzUnicodeValues[uiLoop2],
				0,
				TRUE,
				puiEncDefs[uiLoop])))
			{
				MAKE_ERROR_STRING( "setUnicode failed.", 
					m_szDetails, rc);
				goto Exit;
			}

			if ( RC_BAD( rc = m_pDb->transCommit( )))
			{
				MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
				goto Exit;
			}
			bTransBegun = FALSE;
	
			if ( puzResult)
			{
				f_free( &puzResult);
			}

			if ( RC_BAD( rc = m_pTextElmNode->getUnicode(
				m_pDb,
				&puzResult)))
			{
				MAKE_ERROR_STRING( "getUnicode failed.", 
					m_szDetails, rc);
				goto Exit;
			}

			if ( f_unicmp( puzResult, ppuzUnicodeValues[uiLoop2]) != 0)
			{
				rc = NE_XFLM_DATA_ERROR;
				MAKE_ERROR_STRING( "Unicode value corruption detected", 
					m_szDetails, rc);
				goto Exit;
			}

			endTest("PASS");

			/***********************************************************************/

			f_sprintf( szTemp, 
				"setUTF8 Encrypted Value Test (encryption=%s/value size=%s)",
				ppszEncDefs[uiLoop],
				ppszTextSizes[uiLoop2]);

			beginTest(	
				szTemp, 
				"Tests setUTF8 API using encryption definitions",
				"Self-explanatory",
				"");

			if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS, 
				FLM_NO_TIMEOUT)))
			{
				MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
				goto Exit;
			}
			bTransBegun = TRUE;
	
			if ( RC_BAD( rc = m_pTextElmNode->setUTF8(
				m_pDb,
				(FLMBYTE *)ppszNativeValues[uiLoop2],
				0,
				TRUE,
				puiEncDefs[uiLoop])))
			{
				MAKE_ERROR_STRING( "setUTF8 failed.", 
					m_szDetails, rc);
				goto Exit;
			}

			if ( RC_BAD( rc = m_pDb->transCommit( )))
			{
				MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
				goto Exit;
			}
			bTransBegun = FALSE;
	
			uiBufSize = sizeof(pucUTF8Result);
			if ( RC_BAD( rc = m_pTextElmNode->getUTF8(
				m_pDb,
				pucUTF8Result,
				uiBufSize,
				0,
				uiBufSize)))
			{
				MAKE_ERROR_STRING( "getUTF8 failed.", 
					m_szDetails, rc);
				goto Exit;
			}

			if ( f_strcmp( (char *)pucUTF8Result, ppszNativeValues[uiLoop2]) != 0)
			{
				rc = NE_XFLM_DATA_ERROR;
				MAKE_ERROR_STRING( "UTF8 value corruption detected", 
					m_szDetails, rc);
				goto Exit;
			}

			endTest("PASS");

			/***********************************************************************/

			f_sprintf( szTemp, 
				"setBinary Encrypted Value Test (encryption=%s/value size=%s)",
				ppszEncDefs[uiLoop],
				ppszTextSizes[uiLoop2]);

			beginTest(	
				szTemp, 
				"Tests setBinary API using encryption definitions",
				"Self-explanatory",
				"");

			if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS, 
				FLM_NO_TIMEOUT)))
			{
				MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
				goto Exit;
			}
			bTransBegun = TRUE;
	
			if ( RC_BAD( rc = m_pBinElmNode->setBinary(
				m_pDb,
				ppucBinaryValues[uiLoop2],
				uiBinaryLen,
				TRUE,
				puiEncDefs[uiLoop])))
			{
				MAKE_ERROR_STRING( "setBinary failed.", 
					m_szDetails, rc);
				goto Exit;
			}

			if ( RC_BAD( rc = m_pDb->transCommit( )))
			{
				MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
				goto Exit;
			}
			bTransBegun = FALSE;
	
			uiBufSize = sizeof(pucResult);
			if ( RC_BAD( rc = m_pBinElmNode->getBinary(
				m_pDb,
				pucResult,
				0,
				uiBufSize,
				&uiBufSize)))
			{
				if (rc != NE_XFLM_EOF_HIT ||
					 uiBufSize != uiBinaryLen)
				{
					MAKE_ERROR_STRING( "getBinary failed.", 
						m_szDetails, rc);
					goto Exit;
				}
			}

			if ( memcmp( pucResult, ppucBinaryValues[uiLoop2], uiBinaryLen) != 0)
			{
				rc = NE_XFLM_DATA_ERROR;
				MAKE_ERROR_STRING( "binary value corruption detected", 
					m_szDetails, rc);
				goto Exit;
			}

			endTest("PASS");
		}
	}
	
	// DB Key Rollover tests
	//
	// Create a new database key
	// Close the database
	// Verify we can open the database and read encrypted data.

	beginTest(	
		"Database Key Rollover Test", 
		"Replace the DB key with a new one",
		"Self-explanatory",
		"");
	
		if (RC_BAD( rc = m_pDb->rollOverDbKey()))
		{
			MAKE_ERROR_STRING( "rollOverDbKey function failed", m_szDetails, rc);
			goto Exit;
		}
	
		m_pDb->Release();
		m_pDb = NULL;
		if( RC_BAD( rc = m_pDbSystem->closeUnusedFiles( 0)))
		{
			MAKE_FLM_ERROR_STRING( "closeUnusedFiles failed", m_szDetails, rc);
			goto Exit;
		}

		if (RC_BAD( rc = m_pDbSystem->dbOpen( DB_NAME_STR, NULL, NULL, &m_pDb)))
		{
			MAKE_ERROR_STRING( "dbOpen failed", m_szDetails, rc);
			goto Exit;
		}
		
		// VISIT: Add code to read an encrypted value from the database
		
		endTest("PASS");	

	// WRAP KEY TESTS

	// Wrap the database key in a password
	// Close the database
	// Try reopening the database with the incorrect password
	// Open the database with the correct password

	beginTest(	
		"Wrap Key Test", 
		"Wraps the database key in a password",
		"Self-explanatory",
		"");

	if ( RC_BAD( rc = m_pDb->wrapKey( PASSWORD)))
	{
		MAKE_ERROR_STRING( "binary value corruption detected", 
			m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

	beginTest(	
		"Open Database - Incorrect Password Test", 
		"Ensure the dbOpen fails if the password is incorrect",
		"Close the database/call dbOpen with the incorrect password",
		"");

	if ( m_pDb->Release())
	{
		display("Warning, database did not close!");
	}
	m_pDb = NULL;

	rc = m_pDbSystem->dbOpen( 
		DB_NAME_STR, 
		NULL,
		NULL,
		&m_pDb, 
		BAD_PASSWORD);

	if ( rc != NE_XFLM_PBE_DECRYPT_FAILED)
	{
		MAKE_ERROR_STRING( "Unexpected rc from dbOpen with incorrect password", 
			m_szDetails, rc);
		rc = NE_XFLM_FAILURE;
		goto Exit;
	}

	endTest("PASS");

	beginTest(	
		"Open Database - Correct Password Test", 
		"Ensure the dbOpen succeeds if the password is correct",
		"call dbOpen with the correct password",
		"");

	if (m_pDb)
	{
		m_pDb->Release();
		m_pDb = NULL;
	}

	if ( RC_BAD( rc = m_pDbSystem->dbOpen( 
		DB_NAME_STR,
		NULL,
		NULL,
		&m_pDb, 
		PASSWORD)))
	{
		MAKE_ERROR_STRING( "dbOpen Failed", 
			m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

	beginTest(	
		"backup (w/ password)",
		"backup the database using a password",
		"Self-explanatory",
		"No Additional Details.");

	// Backup the database

	if( RC_BAD( rc = m_pDb->backupBegin( XFLM_FULL_BACKUP,
		XFLM_READ_TRANS, 0, &m_pBackup)))
	{
		MAKE_FLM_ERROR_STRING( "backupBegin failed", m_szDetails, rc);
		goto Exit;
	}

	 rc = m_pBackup->backup( BACKUP_NAME_STR,
		BACKUP_PASSWORD, NULL, NULL, NULL);

	if( RC_BAD( rc))
	{
		MAKE_FLM_ERROR_STRING( "backup failed", m_szDetails, rc);
		goto Exit;
	}
	
	m_pBackup->Release();
	m_pBackup = NULL;

	endTest("PASS");

	beginTest( 			
		"dbRestore (w/ password)",
		"restore the database from the backup we just made",
		"Self-explanatory",
		"No Additional Details.");

	if( RC_BAD( rc = m_pDbSystem->closeUnusedFiles( 0)))
	{
		MAKE_FLM_ERROR_STRING( "closeUnusedFiles failed", m_szDetails, rc);
		goto Exit;
	}
	
	m_pDb->Release();
	m_pDb = NULL;

	// Remove the database
	if( RC_BAD( rc = m_pDbSystem->dbRemove( DB_NAME_STR, NULL, NULL, TRUE)))
	{
		MAKE_FLM_ERROR_STRING( "dbRemove failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDbSystem->dbRestore( DB_NAME_STR,
		NULL, NULL, BACKUP_NAME_STR, BACKUP_PASSWORD, NULL, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "dbRestore failed", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

Exit:

	if (puzResult)
	{
		f_free( &puzResult);
	}

	if (pszResult)
	{
		f_free( &pszResult);
	}

	if ( RC_BAD( rc))
	{
		endTest("FAIL");
	}

	if ( bTransBegun)
	{
		if ( RC_OK( rc))
		{
			m_pDb->transCommit();
		}
		else
		{
			m_pDb->transAbort();
		}
	}

	shutdownTestState( DB_NAME_STR, bDibCreated);
#endif
	return rc;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IEncTestImpl::suite2( void)
{
	RCODE				rc = NE_XFLM_OK;
#ifdef FLM_USE_NICI
	FLMBOOL			bDibCreated = FALSE;
	FLMBOOL			bTransBegun = FALSE;
	FLMUINT *		puiEncDefs[] = {&m_uiAESDef,&m_uiDES3Def};
	FLMUINT64		uiLoop;
	FLMUINT64		ui64TextNodeId;
	FLMUINT64		ui64BinNodeId;
	FLMUINT64		ui64NumNodeId;
	char *			pszRetVal = NULL;
	FLMBYTE			pucRetVal[100];
	FLMUINT64		ui64RetVal;
	FLMUINT			uiBinValRetLen;
	FLMUINT			uiRefCount;

	beginTest(	
		"Encryption Force-To-Disk Test Setup", 
		"Creates all the necessary database objects to run the encryption tests",
		"Create the database/create necessary element and attribute definitions/"
		"create encryption definitions",
		"none");
	
	if ( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		goto Exit;
	}
	bDibCreated = TRUE;

	// ENCRYPTED VALUE TESTS

	// Set up element and attribute definitions of all data types

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = TRUE;

	if ( RC_BAD( rc = setupElemAndAttrNodes()))
	{
		goto Exit;
	}

	if ( RC_BAD( rc = importEncDefs()))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = m_pNumberElmNode->getNodeId( m_pDb, &ui64NumNodeId)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pBinElmNode->getNodeId( m_pDb, &ui64BinNodeId)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = m_pTextElmNode->getNodeId( m_pDb, &ui64TextNodeId)))
	{
		goto Exit;
	}
	
	if ( RC_BAD( rc = m_pDb->transCommit( )))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransBegun = FALSE;

	endTest("PASS");

	if( m_pAttributedNode)
	{
		m_pAttributedNode->Release();
		m_pAttributedNode = NULL;
	}

	if( m_pNumberAttrNode)
	{
		m_pNumberAttrNode->Release();
		m_pNumberAttrNode = NULL;
	}

	if( m_pTextAttrNode)
	{
		m_pTextAttrNode->Release();
		m_pTextAttrNode = NULL;
	}

	if( m_pBinAttrNode)
	{
		m_pBinAttrNode->Release();
		m_pBinAttrNode = NULL;
	}

	beginTest(	
		"Encryption Force-To-Disk test", 
		"Force element values to disk to ensure encryption/decryption is working",
		"Create the database/set node values/close database/open database/retrieve and "
		"verify database values",
		"none");


	for ( uiLoop = 0; uiLoop < 2; uiLoop++)
	{
		if ( RC_BAD( rc = m_pNumberElmNode->setUINT64( 
			m_pDb, UINT64_VAL, *puiEncDefs[uiLoop])))
		{
			MAKE_FLM_ERROR_STRING( "setUINT64 failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = m_pBinElmNode->setBinary( 
			m_pDb, BIN_VAL, BIN_VAL_LEN, TRUE, *puiEncDefs[uiLoop])))
		{
			MAKE_FLM_ERROR_STRING( "setBinary failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = m_pTextElmNode->setUTF8( m_pDb, (FLMBYTE *)NATIVE_VAL, 
			0, TRUE, *puiEncDefs[uiLoop])))
		{
			MAKE_FLM_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
			goto Exit;
		}

		if (m_pRootNode)
		{
			m_pRootNode->Release();
			m_pRootNode = NULL;
		}
		
		if (m_pNumberElmNode)
		{
			m_pNumberElmNode->Release();
			m_pNumberElmNode = NULL;
		}
      
		if( m_pTextElmNode)
		{
			m_pTextElmNode->Release();
			m_pTextElmNode = NULL;
		}

		if( m_pBinElmNode)
		{
			m_pBinElmNode->Release();
			m_pBinElmNode = NULL;
		}

		// close the database -- force values to disk
		uiRefCount = m_pDb->Release(); 
		flmAssert( !uiRefCount);

		if ( RC_BAD( rc = m_pDbSystem->dbOpen( DB_NAME_STR, NULL, NULL,
			NULL, FALSE, &m_pDb)))
		{
			MAKE_FLM_ERROR_STRING( "dbOpen failed.", m_szDetails, rc);
			goto Exit;
		}

		if( RC_BAD( rc = m_pDb->getNode( XFLM_DATA_COLLECTION,
			ui64NumNodeId, &m_pNumberElmNode)))
		{
			MAKE_FLM_ERROR_STRING( "getNode failed.", m_szDetails, rc);
			goto Exit;
		}

		if( RC_BAD( rc = m_pDb->getNode( XFLM_DATA_COLLECTION,
			ui64BinNodeId, &m_pBinElmNode)))
		{
			MAKE_FLM_ERROR_STRING( "getNode failed.", m_szDetails, rc);
			goto Exit;
		}

		if( RC_BAD( rc = m_pDb->getNode( XFLM_DATA_COLLECTION,
			ui64TextNodeId, &m_pTextElmNode)))
		{
			MAKE_FLM_ERROR_STRING( "getNode failed.", m_szDetails, rc);
			goto Exit;
		}

		// Retrieve and check values

		if( RC_BAD( rc = m_pNumberElmNode->getUINT64( m_pDb, &ui64RetVal)))
		{
			MAKE_FLM_ERROR_STRING( "getUINT64 failed.", m_szDetails, rc);
			goto Exit;
		}

		if( ui64RetVal != UINT64_VAL)
		{
			rc = RC_SET( NE_XFLM_DATA_ERROR);
			MAKE_FLM_ERROR_STRING( "Unexpected value.", m_szDetails, rc);
			goto Exit;
		}

		if( RC_BAD( rc = m_pBinElmNode->getBinary( m_pDb, pucRetVal, 0,
			BIN_VAL_LEN, &uiBinValRetLen)))
		{
			MAKE_FLM_ERROR_STRING( "getBinary failed.", m_szDetails, rc);
			goto Exit;
		}

		if( uiBinValRetLen != BIN_VAL_LEN ||
			f_memcmp( pucRetVal, BIN_VAL, BIN_VAL_LEN) != 0)
		{
			rc = RC_SET( NE_XFLM_DATA_ERROR);
			MAKE_FLM_ERROR_STRING( "Unexpected value.", m_szDetails, rc);
			goto Exit;
		}

		if( pszRetVal)
		{
			f_free( &pszRetVal);
		}

		if( RC_BAD( rc = m_pTextElmNode->getUTF8( m_pDb, 
			(FLMBYTE **)&pszRetVal)))
		{
			MAKE_FLM_ERROR_STRING( "getUTF8 failed.", m_szDetails, rc);
			goto Exit;
		}

		if( f_strcmp( pszRetVal, NATIVE_VAL) != 0)
		{
			rc = RC_SET( NE_XFLM_DATA_ERROR);
			MAKE_FLM_ERROR_STRING( "Unexpected value.", m_szDetails, rc);
			goto Exit;
		}
	}

	endTest("PASS");

Exit:

	if( bTransBegun)
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

	if( pszRetVal)
	{
		f_free( &pszRetVal);
	}

	shutdownTestState( DB_NAME_STR, bDibCreated);

#endif

	return( rc);
}
