//------------------------------------------------------------------------------
// Desc:	DOM Node tests
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

#define BUILTIN_ATTRIBUTES (XFLM_LAST_RESERVED_ATTRIBUTE_TAG - \
			XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 1)
			
#define NUM_CHILD_NODES					500

struct UTF8LenPair
{
	FLMBYTE *	pucUTF8Str;
	FLMUINT		uiNumChars;
};

/****************************************************************************
Desc:
****************************************************************************/
class IDOMNodeTestImpl : public TestBase
{
public:

	const char * getName( void);
	
	RCODE execute( void);
	
	RCODE	domNodeWorkout( void);
	
	RCODE pendingNodesTest( void);
	
	RCODE strValLengthTests( void);
	
	RCODE uniqueChildTests( void);
	
	RCODE	concurrentValueModifyTests( void);
};

/****************************************************************************
Desc:
****************************************************************************/
RCODE getTest( 
	IFlmTest **		ppTest)
{
	RCODE		rc = NE_XFLM_OK;

	if( (*ppTest = f_new IDOMNodeTestImpl) == NULL)
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
const char * IDOMNodeTestImpl::getName( void)
{
	return( "DOM Node Test");
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IDOMNodeTestImpl::execute( void)
{
	RCODE				rc = NE_XFLM_OK;

	if( RC_BAD( rc = domNodeWorkout()))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pendingNodesTest()))
	{
		goto Exit;
	}

	if( RC_BAD( rc = strValLengthTests()))
	{
		goto Exit;
	}

	if( RC_BAD( rc = uniqueChildTests()))
	{
		goto Exit;
	}

	/* VISIT - enable when concurrent streaming fails with correct error codes
	if ( RC_BAD( rc = concurrentValueModifyTests()))
	{
		goto Exit;
	}
	*/

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IDOMNodeTestImpl::domNodeWorkout( void)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pDocument = NULL;
	IF_DOMNode *	pDocRoot = NULL;
	IF_DOMNode *	pDOMNode1 = NULL;
	IF_DOMNode *	pDOMNode2 = NULL;
	IF_DOMNode *	pAttr = NULL;
	int				ps = 0;
	FLMUINT			uiTag = 0;
	FLMBOOL			b = FALSE;
	FLMUINT			uiDataType = 0;
	FLMUINT64		ui64ParentID = 0;
	FLMUINT			uiLoop = 0;
	FLMBOOL			bTransStarted = FALSE;
	FLMBOOL			bDibCreated = FALSE;
	FLMUINT64		ui64NodeId;
	eDomNodeType	eNodeType;
	FLMUINT			uiValue = 12345;
	FLMUINT64		ui64Value = 123456;
	FLMINT			iValue = -12345;
	FLMINT64			i64Value = -12345;
	const char *	pszValue = "Native\\UTF8 Value";
	FLMUNICODE		puzValue[] = {85,110,105,99,111,100,101,0};
	FLMBYTE			pucBinValue[] = {0x01, 0x02, 0x03, 0x04, 0x05};
	FLMUINT			uiValueRV = 0;
	FLMUINT64		ui64ValueRV = 0;
	FLMINT			iValueRV = 0;
	FLMINT64			i64ValueRV = 0;
	char				pszValueRV[ 128];
	FLMUNICODE		puzValueRV[] = {0,0,0,0,0,0,0,0};
	FLMBYTE			pucBinValueRV[ 64];
	FLMUINT			uiBufferBytes = sizeof( pszValueRV);
	FLMUINT64		ui64Tmp;
	char				szTemp[ 128];

	//PART #1 - Initialize XFLAIM
	
	beginTest(
		"Init",
		"Perform initializations so we can exercise the DOMNode",
		"(1)Get DbSystem class factory (2)call init() (3)create XFLAIM db",
		"No Additional Info.");

	if ( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		MAKE_ERROR_STRING( "Failed to initialize test state.", m_szDetails, ps);
		goto Exit;
	}
	bDibCreated = TRUE;

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.",m_szDetails, ps);
		goto Exit;
	}
	bTransStarted = TRUE;

	endTest("PASS");

	beginTest( 
		"DOMNode Workout",
		"Exercise the DOMNode interface",
		"(1)Create a document (2)Create child nodes (3)Create attributes "
		"(4)Set and verify attribute values (5)Iterate through nodes",
		"");

	if ( RC_BAD( rc = m_pDb->createDocument( 
		XFLM_DATA_COLLECTION, 
		&pDocument)))
	{
		MAKE_ERROR_STRING( "createDocument failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pDocument->createNode( 
		m_pDb,
		ELEMENT_NODE,
		ELM_ELEMENT_TAG,
		XFLM_FIRST_CHILD,
		&pDocRoot)))
	{
		MAKE_ERROR_STRING( "createNode failed.", m_szDetails, rc);
		goto Exit;
	}

	// Create the first child

	if ( RC_BAD( rc = pDocRoot->createNode( 
		m_pDb,
		ELEMENT_NODE,
		ELM_ELEMENT_TAG,
		XFLM_FIRST_CHILD,
		&pDOMNode1)))
	{
		MAKE_ERROR_STRING( "createDocument failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pDocRoot->getDocumentId( m_pDb, &ui64NodeId)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pDocument->getNodeId( m_pDb, &ui64Tmp)))
	{
		goto Exit;
	}

	if( ui64Tmp != ui64NodeId)
	{
		MAKE_GENERIC_ERROR_STRING64( "Incorrect root Id", m_szDetails,
											  ui64NodeId);

		goto Exit;
	}

	// Create a bunch of siblings and add attributes to them
	
	for ( uiLoop = 0; uiLoop < NUM_CHILD_NODES - 1; uiLoop++)
	{
		// Record the iteration number for output purposes.

		f_sprintf( szTemp, " Iteration #%lu.", uiLoop);
		
		// Make a bunch of sibling nodes
		
		if ( RC_BAD( rc = pDOMNode1->createNode(
			m_pDb,
			ELEMENT_NODE,
			ELM_ELEMENT_TAG,
			(uiLoop % 2) ? XFLM_NEXT_SIB : XFLM_PREV_SIB,
			&pDOMNode2)))
		{
			MAKE_ERROR_STRING( "createNode failed.", m_szDetails, rc);
			f_strcat( m_szDetails, szTemp);
			goto Exit;
		}
		
		eNodeType = pDOMNode2->getNodeType();

		if( eNodeType != ELEMENT_NODE)
		{
			rc = RC_SET( NE_XFLM_FAILURE);
			MAKE_ERROR_STRING( "Illegal node type.", 
				m_szDetails, eNodeType);
			f_strcat( m_szDetails, szTemp);
			goto Exit;
		}
	
		if ( RC_BAD( rc = pDOMNode2->hasChildren(m_pDb, &b)))
		{
			MAKE_ERROR_STRING( "hasChildren failed.", m_szDetails, rc);
			goto Exit;
		}

		if (b)
		{
			rc = RC_SET( NE_XFLM_FAILURE);
			MAKE_ERROR_STRING( "Node erroneously claims to have children.", 
				m_szDetails, rc);
			f_strcat( m_szDetails, szTemp);
			goto Exit;
		}

		uiTag = XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + (uiLoop % BUILTIN_ATTRIBUTES);
		
		if ( RC_BAD( rc = pDOMNode2->createAttribute(
			m_pDb,
			uiTag,
			&pAttr)))
		{
			MAKE_ERROR_STRING( "createAttribute failed.", 
				m_szDetails, rc);
			f_strcat( m_szDetails, szTemp);
			goto Exit;
		}

		if ( RC_BAD( rc = pDOMNode2->hasAttribute(
			m_pDb,
			uiTag)))
		{
			if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				MAKE_ERROR_STRING( "Node is missing an attribute", 
					m_szDetails, rc);
			}
			else
			{
				MAKE_ERROR_STRING( "hasAttribute failed.", m_szDetails, rc);
			}
			f_strcat( m_szDetails, szTemp);
			goto Exit;
		}

		// Look up the tag's data type and set an appropriate value
		// Then retrieve it again for verification
		
		if ( RC_BAD( rc = pAttr->getDataType(
			m_pDb,
			&uiDataType)))
		{
			MAKE_ERROR_STRING( "getDataType failed.", 
				m_szDetails, rc);
			f_strcat( m_szDetails, szTemp);
			goto Exit;
		}

		switch( uiDataType)
		{
		case XFLM_NUMBER_TYPE:
			switch( uiLoop % 4)
			{
			case 0:

				if ( RC_BAD( rc = pAttr->setUINT(
					m_pDb,
					uiValue)))
				{
					MAKE_ERROR_STRING( "setUINT failed.", 
						m_szDetails, rc);
					goto Exit;
				}

				if (RC_BAD( rc = pAttr->getUINT( m_pDb, &uiValueRV)))
				{
					MAKE_ERROR_STRING( "getUINT failed.", 
						m_szDetails, rc);
					goto Exit;
				}

				if ( uiValue != uiValueRV)
				{
					rc = RC_SET( NE_XFLM_FAILURE);
					MAKE_ERROR_STRING( "UINT Data corruption detected.", 
						m_szDetails, rc);
					f_strcat( m_szDetails, szTemp);
					goto Exit;
				}
				break;
			case 1:

				if ( RC_BAD( rc = pAttr->setUINT64(
					m_pDb,
					ui64Value
					)))
				{
					MAKE_ERROR_STRING( "setUINT64 failed.", 
						m_szDetails, rc);
					goto Exit;
				}

				if ( RC_BAD( rc = pAttr->getUINT64( m_pDb, &ui64ValueRV)))
				{
					MAKE_ERROR_STRING( "getUINT64 failed.", 
						m_szDetails, rc);
					goto Exit;
				}

				if ( ui64Value != ui64ValueRV)
				{
					rc = RC_SET( NE_XFLM_FAILURE);
					MAKE_ERROR_STRING( "UINT64 Data corruption detected", 
						m_szDetails, rc);
					f_strcat( m_szDetails, szTemp);
					goto Exit;
				}
				break;
			case 2:

				if ( RC_BAD( rc = pAttr->setINT(
					m_pDb,
					iValue
					)))
				{
					MAKE_ERROR_STRING( "setINT failed.", 
						m_szDetails, rc);
					goto Exit;
				}

				if ( RC_BAD( rc = pAttr->getINT( m_pDb, &iValueRV)))
				{
					MAKE_ERROR_STRING( "getINT failed.", 
						m_szDetails, rc);
					goto Exit;
				}

				if ( iValue != iValueRV)
				{
					rc = RC_SET( NE_XFLM_FAILURE);
					MAKE_ERROR_STRING( "INT Data corruption detected", 
						m_szDetails, rc);
					f_strcat( m_szDetails, szTemp);
					goto Exit;
				}
				break;
			case 3:

				if ( RC_BAD( rc = pAttr->setINT64(
					m_pDb,
					i64Value
					)))
				{
					MAKE_ERROR_STRING( "setINT64 failed.", 
						m_szDetails, rc);
					goto Exit;
				}

				if ( RC_BAD ( rc = pAttr->getINT64( m_pDb, &i64ValueRV)))
				{
					MAKE_ERROR_STRING( "getINT64 failed.", 
						m_szDetails, rc);
					goto Exit;
				}

				if ( i64Value != i64ValueRV)
				{
					rc = RC_SET( NE_XFLM_FAILURE);
					MAKE_ERROR_STRING( "INT64 Data corruption detected", 
						m_szDetails, rc);
					f_strcat( m_szDetails, szTemp);
					goto Exit;
				}
				break;
			}
			break;
		case XFLM_TEXT_TYPE:
			switch( uiLoop % 3)
			{
			case 0:
				if ( RC_BAD( rc = pAttr->setUTF8( m_pDb, (FLMBYTE *)pszValue)))
				{
					MAKE_ERROR_STRING( "setUTF8 failed.", 
						m_szDetails, rc);
					goto Exit;
				}

				if (RC_BAD( rc = pAttr->getUTF8( 
					m_pDb, (FLMBYTE *)pszValueRV, 
					sizeof(pszValueRV), 0, ~((FLMUINT)0))))
				{
					MAKE_ERROR_STRING( "getUTF8 error", 
						m_szDetails, rc);
					f_strcat( m_szDetails, szTemp);
					goto Exit;
				}

				if ( f_strcmp( pszValue, pszValueRV) != 0)
				{
					rc = NE_XFLM_FAILURE;
					MAKE_ERROR_STRING( "Native Data corruption detected", 
						m_szDetails, rc);
					f_strcat( m_szDetails, szTemp);
					goto Exit;
				}
				break;
			case 1:

				if (RC_BAD( rc = pAttr->setUnicode(
					m_pDb,
					puzValue
					)))
				{
					MAKE_ERROR_STRING( "setUnicode failed.", 
						m_szDetails, rc);
					goto Exit;
				}

				if (RC_BAD( rc = pAttr->getUnicode( 
					m_pDb, 
					puzValueRV,
					sizeof(puzValueRV),
					0,
					~((FLMUINT)0),
					NULL)))
				{
					MAKE_ERROR_STRING( "getUnicode error", 
						m_szDetails, rc);
					f_strcat( m_szDetails, szTemp);
					goto Exit;
				}

				{
					FLMUNICODE *	puzTmp1 = puzValue;
					FLMUNICODE *	puzTmp2 = puzValueRV;

					while (*puzTmp1 && *puzTmp2 && *puzTmp1 == *puzTmp2)
					{
						puzTmp1++;
						puzTmp2++;
					}
					if ( *puzTmp1 || *puzTmp2)
					{
						rc = NE_XFLM_FAILURE;
						MAKE_ERROR_STRING( "Unicode data corruption detected", 
							m_szDetails, rc);
						f_strcat( m_szDetails, szTemp);
						goto Exit;
					}
				}
				break;
			case 2:

				if (RC_BAD( rc = pAttr->setUTF8(
					m_pDb,
					(FLMBYTE *)pszValue
					)))
				{
					MAKE_ERROR_STRING( "setUTF8 failed.", 
						m_szDetails, rc);
					goto Exit;
				}

				uiBufferBytes = sizeof( pszValueRV);

				if (RC_BAD( rc = pAttr->getUTF8( 
					m_pDb, 
					(FLMBYTE *)pszValueRV, 
					uiBufferBytes,
					0,
					~((FLMUINT)0),
					NULL)))
				{
					MAKE_ERROR_STRING( "getUTF8 error", 
						m_szDetails, rc);
					f_strcat( m_szDetails, szTemp);
					goto Exit;
				}

				if ( f_strcmp( pszValue, pszValueRV) != 0)
				{
					rc = NE_XFLM_FAILURE;
					MAKE_ERROR_STRING( "UTF8 data corruption detected", 
						m_szDetails, rc);
					f_strcat( m_szDetails, szTemp);
					goto Exit;

				}
				break;

			case XFLM_BINARY_TYPE:
				
				if (RC_BAD( rc = pAttr->setBinary(
					m_pDb,
					pucBinValue,
					10
					)))
				{
					MAKE_ERROR_STRING( "setBinary failed.", 
						m_szDetails, rc);
					goto Exit;
				}

				if RC_BAD( rc = pAttr->getBinary( m_pDb, pucBinValueRV, 0, 10, &uiValueRV))
				{
					MAKE_ERROR_STRING( "getBinary failed.", 
						m_szDetails, rc);
					goto Exit;
				}

				if ( f_memcmp( pucBinValue, pucBinValueRV, 10))
				{
					rc = NE_XFLM_FAILURE;
					MAKE_ERROR_STRING( "Binary data corruption detected", 
						m_szDetails, rc);
					f_strcat( m_szDetails, szTemp);
					goto Exit;
				}
				break;

		default:
				break;
			}
			break;
		}
		
		// Since there's only one attribute, either one of these functions will do
		if ( (uiLoop % 2) == 0) 
		{
			if ( RC_BAD( rc = pDOMNode2->getFirstAttribute(
				m_pDb,
				&pAttr)))
			{
				MAKE_ERROR_STRING( "getFirstAttribute failed", 
					m_szDetails, rc);
				f_strcat( m_szDetails, szTemp);
				goto Exit;
			}
		}
		else
		{
			if ( RC_BAD( rc = pDOMNode2->getAttribute(
				m_pDb,
				uiTag,
				&pAttr
				)))
			{
				MAKE_ERROR_STRING( "getAttribute failed", 
					m_szDetails, rc);
				f_strcat( m_szDetails, szTemp);
				goto Exit;
			}
		}

		// We gave these nodes one and only one attribute
		// The attributes should have no siblings

		if ( ( rc = pAttr->getPreviousSibling( 
			m_pDb,
			&pDOMNode2)) != NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			MAKE_ERROR_STRING( "getPreviousSibling returned invalid rc. ", 
					m_szDetails, rc);
			f_strcat( m_szDetails, szTemp);
			if ( RC_OK( rc))
			{
				rc = NE_XFLM_FAILURE;
			}
			goto Exit;
		}

		if ( ( rc = pAttr->getNextSibling( 
			m_pDb,
			&pDOMNode2)) != NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			MAKE_ERROR_STRING( "getNextSibling returned invalid rc. ", 
					m_szDetails, rc);
			f_strcat( m_szDetails, szTemp);
			if ( RC_OK( rc))
			{
				rc = NE_XFLM_FAILURE;
			}
			goto Exit;
		}
	}
	
	if( RC_BAD( rc = pDocRoot->getNodeId( m_pDb, &ui64ParentID)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pDocRoot->hasChildren(m_pDb, &b)))
	{
		MAKE_ERROR_STRING( "hasChildren failed. ", 
				m_szDetails, rc);
		f_strcat( m_szDetails, szTemp);
		goto Exit;
	}
	
	if( !b)
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		MAKE_ERROR_STRING( "Document root erroneously claims to have no children. ", 
			m_szDetails, rc);
		f_strcat( m_szDetails, szTemp);
		goto Exit;
	}

	// Reposition to the first child under the document root,
	// iterate through its children (first->last) and perform 
	// various DOMNode ops

	for ( uiLoop = 0; uiLoop < NUM_CHILD_NODES; uiLoop++)
	{
		f_sprintf( szTemp, " Iteration #%lu.", uiLoop);

		if ( uiLoop == 0)
		{
			// Initialization

			if ( RC_BAD( rc = pDocRoot->getFirstChild(
				m_pDb,
				&pDOMNode1)))
			{
				MAKE_ERROR_STRING( "getFirstChild failed.", 
					m_szDetails, rc);
				f_strcat( m_szDetails, szTemp);
				goto Exit;
			}
		}
		else
		{
			// Move to the next sibling

			if ( RC_BAD( rc = pDOMNode1->getNextSibling(
				m_pDb,
				&pDOMNode1)))
			{
				MAKE_ERROR_STRING( "getNextSibling failed.", 
					m_szDetails, rc);
				f_strcat( m_szDetails, szTemp);
				goto Exit;
			}
		}

		if ( RC_BAD( rc = pDOMNode1->getParentId(m_pDb, &ui64ValueRV)))
		{
				MAKE_ERROR_STRING( "getParentId failed.", 
					m_szDetails, rc);
				f_strcat( m_szDetails, szTemp);
				goto Exit;
		}

		if ( ui64ParentID != ui64ValueRV)
		{
				rc = RC_SET( NE_XFLM_FAILURE);
				MAKE_ERROR_STRING( "Incorrect parent ID.", 
					m_szDetails, rc);
				f_strcat( m_szDetails, szTemp);
				rc = NE_XFLM_FAILURE;
				goto Exit;
		}
	}

	// There should be no more siblings

	if ( (rc = pDOMNode1->getNextSibling(
		m_pDb,
		&pDOMNode2)) != NE_XFLM_DOM_NODE_NOT_FOUND)
	{
		MAKE_ERROR_STRING( "Invalid rc returned from getNextSibling.", 
			m_szDetails, rc);
		f_strcat( m_szDetails, szTemp);
		if ( RC_OK( rc))
		{
			rc = NE_XFLM_FAILURE;
		}
		goto Exit;
	}

	endTest("PASS");

	beginTest(
		"DOMNode Iteration/Deletion",
		"Iterate and delete the DOMNodes",
		"Move backwards through the child nodes and delete them",
		"");

	// Move backwards through the siblings deleting them (except the last one)

	for( uiLoop = 0; uiLoop < NUM_CHILD_NODES; uiLoop++)
	{
		if ( uiLoop == 0)
		{
			// Initialization

			if ( RC_BAD( rc = pDocRoot->getLastChild(
				m_pDb,
				&pDOMNode1)))
			{
				MAKE_ERROR_STRING( "getLastChild failed.", m_szDetails, rc);
				f_strcat( m_szDetails, szTemp);
				goto Exit;
			}
		}
		else
		{
			// Move to the prev sibling

			if ( RC_BAD( rc = pDOMNode1->getPreviousSibling(
				m_pDb,
				&pDOMNode2)))
			{
				MAKE_ERROR_STRING( "getPreviousSibling failed.", m_szDetails, rc);
				f_strcat( m_szDetails, szTemp);
				goto Exit;
			}

			if ( RC_BAD( rc = pDOMNode1->deleteNode( m_pDb)))
			{
				MAKE_ERROR_STRING( "deleteNode failed.", m_szDetails, rc);
				f_strcat( m_szDetails, szTemp);
				goto Exit;
			}
			
			pDOMNode1->Release();
			pDOMNode1 = pDOMNode2;
			pDOMNode2 = NULL;
		}
	}

	// Test error condition

	if ( ( rc = pDOMNode1->getPreviousSibling(
		m_pDb,
		&pDOMNode2)) != NE_XFLM_DOM_NODE_NOT_FOUND)
	{
		MAKE_ERROR_STRING( "getPreviousSibling returned invalid rc.", m_szDetails, rc);
		if ( RC_OK( rc))
		{
			rc = NE_XFLM_FAILURE;
		}
		goto Exit;
	}

	//Delete the last DOM node

	if ( RC_BAD( rc = pDOMNode1->deleteNode(m_pDb)))
	{
		MAKE_ERROR_STRING( "deleteNode failed.", m_szDetails, rc);
		goto Exit;
	}

	pDOMNode1->Release();
	pDOMNode1 = NULL;

	// Test document iteration
	// pDocument - 1st doc, pDOMNode1 - next doc

	if ( RC_BAD( rc = m_pDb->createDocument( 
		XFLM_DATA_COLLECTION, 
		&pDOMNode1)))
	{
		MAKE_ERROR_STRING( "createDocument failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pDocument->getNextDocument(
		m_pDb,
		&pDOMNode2)))
	{
		MAKE_ERROR_STRING( "getNextDocument failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = pDOMNode1->getPreviousDocument(
		m_pDb,
		&pDOMNode2)))
	{
		MAKE_ERROR_STRING( "getPreviousDocument failed.", m_szDetails, rc);
		goto Exit;
	}

	if ( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransStarted = FALSE;

	endTest("PASS");

Exit:	

	if ( RC_BAD( rc ))
	{
		endTest("FAIL");
	}

	if ( bTransStarted)
	{
		m_pDb->transCommit();
	}
	if ( pDOMNode1)
	{
		pDOMNode1->Release();
	}
	if ( pDOMNode2)
	{
		pDOMNode2->Release();
	}
	if ( pDocument)
	{
		pDocument->Release();
	}
	if (pDocRoot)
	{
		pDocRoot->Release();
	}
	if ( pAttr)
	{
		pAttr->Release();
	}

	shutdownTestState( DB_NAME_STR, bDibCreated);

	return rc;
}

#define NUM_DOCS	2000

/****************************************************************************
Desc:
****************************************************************************/
RCODE IDOMNodeTestImpl::pendingNodesTest( void)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bTransActive = FALSE;
	FLMUINT64 *		pui64Docs = NULL;
	FLMUINT			uiLoop;
	IF_DOMNode *	pDoc = NULL;
	FLMBOOL			bDibCreated = FALSE;

	beginTest(
		"Pending Nodes Error Test",
		"Delete 2000 documents within a single transaction "
		"and ensure NE_XFLM_TOO_MANY_PENDING_NODES is not returned "
		"to verify DEFECT000400386 has been fixed",
		"Self-explanatory",
		"");

	if ( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		goto Exit;
	}
	bDibCreated = TRUE;

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransActive = TRUE;

	if ( RC_BAD( rc = f_alloc( 
		NUM_DOCS * sizeof(FLMUINT64), &pui64Docs)))
	{
		MAKE_ERROR_STRING( "Failed to allocate docId array", m_szDetails, rc);
		goto Exit;
	}

	for( uiLoop = 0; uiLoop < NUM_DOCS; uiLoop++)
	{
		if (RC_BAD( rc = m_pDb->createDocument( 
			XFLM_DATA_COLLECTION,
			&pDoc)))
		{
			MAKE_ERROR_STRING( "createDocument failed.", m_szDetails, rc);
			goto Exit;
		}

		if (RC_BAD( rc = m_pDb->documentDone( pDoc)))
		{
			MAKE_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
			goto Exit;
		}
		
		if( RC_BAD( rc = pDoc->getNodeId( m_pDb, &pui64Docs[ uiLoop])))
		{
			MAKE_ERROR_STRING( "getNodeId failed.", m_szDetails, rc);
			goto Exit;
		}
	}

	if ( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransActive = FALSE;

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransActive = TRUE;

	for ( uiLoop = 0; uiLoop < NUM_DOCS; uiLoop++)
	{
		if ( RC_BAD( rc = m_pDb->getNode( 
			XFLM_DATA_COLLECTION,
			pui64Docs[uiLoop],
			&pDoc)))
		{
			MAKE_ERROR_STRING( "getNode failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = pDoc->deleteNode( m_pDb)))
		{
			MAKE_ERROR_STRING( "deleteNode failed.", m_szDetails, rc);
			goto Exit;
		}
	}

	if ( RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_ERROR_STRING( "transCommit failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransActive = FALSE;
	endTest("PASS");

Exit:

	if ( bTransActive)
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

	if ( RC_BAD( rc))
	{
		endTest("FAIL");
	}

	if ( pDoc)
	{
		pDoc->Release();
	}

	if ( pui64Docs)
	{
		f_free( &pui64Docs);
	}

	shutdownTestState( DB_NAME_STR, bDibCreated);
	return rc;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IDOMNodeTestImpl::strValLengthTests( void)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBYTE			pucUTF8Odysseus[] = {
		0xEF, 0xBB, 0xBF, 0xCE, 0xA4, 0xCE, 0xB7, 0x20, 0xCE, 0xB3, 0xCE, 0xBB, 0xCF, 
		0x8E, 0xCF, 0x83, 0xCF, 0x83, 0xCE, 0xB1, 0x20, 0xCE, 0xBC, 0xCE, 0xBF, 0xCF,
		0x85, 0x20, 0xCE, 0xAD, 0xCE, 0xB4, 0xCF, 0x89, 0xCF, 0x83, 0xCE, 0xB1, 0xCE, 
		0xBD, 0x20, 0xCE, 0xB5, 0xCE, 0xBB, 0xCE, 0xBB, 0xCE, 0xB7, 0xCE, 0xBD, 0xCE, 
		0xB9, 0xCE, 0xBA, 0xCE, 0xAE, 0x00}; 
		 // First few words from a Greek poem (29 actual chars + 1 null)
	FLMBYTE			pucUTF8BronzeHorse[] = { 
		0xEF, 0xBB, 0xBF, 0xD0, 0x9D, 0xD0, 0xB0, 0x20, 0xD0, 0xB1, 0xD0, 0xB5, 0xD1,	0x80, 0xD0, 0xB5, 
		0xD0, 0xB3, 0xD1, 0x83, 0x20, 0xD0, 0xBF, 0xD1, 0x83, 0xD1, 0x81, 0xD1, 0x82, 0xD1, 0x8B, 0xD0, 
		0xBD, 0xD0, 0xBD, 0xD1, 0x8B, 0xD1, 0x85, 0x20, 0xD0, 0xB2, 0xD0, 0xBE, 0xD0, 0xBB, 0xD0, 0xBD, 0x00}; 
		// Pushkin's Bronze Horse (Russian) (24 actual chars + 1 null)

	UTF8LenPair		testStrings[] = 
		{
			{ pucUTF8Odysseus, 30},
			{ pucUTF8BronzeHorse, 25}
		};

	IF_DOMNode *	pNode = NULL;
	FLMUINT			uiNumChars;
	FLMUINT			uiLoop;
	FLMBOOL			bTransActive = FALSE;
	char				szTestName[100];
	FLMBOOL			bDibCreated = FALSE;

	beginTest(
		"String Value Length Test Setup",
		"Set up for string value length tests",
		"Self-explanatory",
		"");

	if ( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		goto Exit;
	}
	bDibCreated = TRUE;

	if ( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_ERROR_STRING( "transBegin failed.", m_szDetails, rc);
		goto Exit;
	}
	bTransActive = TRUE;

	if ( RC_BAD( rc = m_pDb->createRootElement(
		XFLM_DATA_COLLECTION,
		ELM_DOCUMENT_TITLE_TAG, // chosen because it is a text type
		&pNode)))
	{
		MAKE_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

	for(	uiLoop = 0; 
			uiLoop < sizeof(testStrings)/sizeof(testStrings[0]); 
			uiLoop++)
	{
		f_sprintf( szTestName, "String Value Length Test %u", uiLoop);

		beginTest(
			szTestName,
			"Set a UTF8 value in a node then call getUTF8 and getUnicode "
			"without buffers to get the character count",
			"Self-explanatory",
			"");

		if ( RC_BAD( rc = pNode->setUTF8( m_pDb, testStrings[uiLoop].pucUTF8Str)))
		{
			MAKE_ERROR_STRING( "setUTF8 failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( RC_BAD( rc = pNode->getUTF8(
			m_pDb, NULL, 0, 0, 0, &uiNumChars)))
		{
			MAKE_ERROR_STRING( "getUTF8 failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( uiNumChars != testStrings[uiLoop].uiNumChars)
		{
			rc = RC_SET( NE_XFLM_FAILURE);
			MAKE_ERROR_STRING( "Unexpected string length.", m_szDetails, rc);
			goto Exit;
		}

		// call getUnicode with no buffer to validate number of chars

		if ( RC_BAD( rc = pNode->getUnicode( 
			m_pDb,
			NULL,
			0, 0, 0, &uiNumChars)))
		{
			MAKE_ERROR_STRING( "getUnicode failed.", m_szDetails, rc);
			goto Exit;
		}

		if ( uiNumChars != testStrings[uiLoop].uiNumChars)
		{
			rc = RC_SET( NE_XFLM_FAILURE);
			MAKE_ERROR_STRING( "Unexpected string length.", m_szDetails, rc);
			goto Exit;
		}

		endTest("PASS");
	}

Exit:

	if ( bTransActive)
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

	if ( RC_BAD( rc))
	{
		endTest("FAIL");
	}

	if ( pNode)
	{
		pNode->Release();
	}

	shutdownTestState( DB_NAME_STR, bDibCreated);
	return rc;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IDOMNodeTestImpl::uniqueChildTests( void)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiDefNum1 = 0;
	FLMUINT			uiDefNum2;
	IF_DOMNode *	pRootElement = NULL;
	IF_DOMNode *	pElement = NULL;
	IF_DOMNode *	pUniqueElement = NULL;
	char				szName [80];
	FLMUINT			uiNumToUse;
	FLMUINT			uiLoop;	
	FLMBOOL			bStartedTrans = FALSE;
	FLMBOOL			bDibCreated = FALSE;
	FLMUINT			uiRefCount;

	beginTest( 
		"Unique Child Elements Database Check Test", 
		"Make sure a check succeeds on a dib with unique child element lists",
		"Self explanatory",
		"No Additional Details.");

	if ( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		goto Exit;
	}
	bDibCreated = TRUE;

	// Start an update transaction

	if (RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed", m_szDetails, rc);
		goto Exit;
	}
	bStartedTrans = TRUE;
	
	if( RC_BAD( rc = m_pDb->createUniqueElmDef(
		NULL, "unique_element", &uiDefNum1)))
	{
		MAKE_FLM_ERROR_STRING( "createUniqueElmDef failed", m_szDetails, rc);
		goto Exit;
	}

	// Create 200 element unique element definitions.
	
	uiDefNum2 = uiDefNum1 + 1;
	for (uiLoop = 0; uiLoop < 200; uiLoop++)
	{
		uiNumToUse = uiDefNum2 + uiLoop;
		
		f_sprintf( szName, "c_element%u", (unsigned)uiLoop);
		if( RC_BAD( rc = m_pDb->createElementDef(
			NULL, szName, XFLM_NODATA_TYPE, &uiNumToUse)))
		{
			MAKE_FLM_ERROR_STRING( "createElementDef failed", m_szDetails, rc);
			goto Exit;
		}
	}
	
	// Create a root node, with two child nodes.
	
	if( RC_BAD( rc = m_pDb->createRootElement( XFLM_DATA_COLLECTION,
										ELM_ELEMENT_TAG, &pRootElement)))
	{
		MAKE_FLM_ERROR_STRING( "createRootElement failed", m_szDetails, rc);
		goto Exit;
	}
	if( RC_BAD( rc = pRootElement->createNode( m_pDb,
										ELEMENT_NODE, uiDefNum1, XFLM_FIRST_CHILD,
										&pUniqueElement, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "createNode failed", m_szDetails, rc);
		goto Exit;
	}
	
	for (uiLoop = 0; uiLoop < 200; uiLoop++)
	{
		// Should not allow a data node.
		
		if( RC_BAD( rc = pUniqueElement->createNode( m_pDb,
											ELEMENT_NODE, uiDefNum2 + uiLoop,
											XFLM_LAST_CHILD, &pElement, NULL)))
		{
			MAKE_FLM_ERROR_STRING( "createNode failed", m_szDetails, rc);
			goto Exit;
		}
		
	}

	pRootElement->Release();
	pRootElement = NULL;

	pElement->Release();
	pElement = NULL;

	pUniqueElement->Release();
	pUniqueElement = NULL;

	if ( RC_BAD( rc = rc = m_pDb->transCommit()))
	{
		goto Exit;
	}
	bStartedTrans = FALSE;

	uiRefCount = m_pDb->Release();
	flmAssert( uiRefCount == 0);
	m_pDb = NULL;

	// do a check

//	if ( RC_BAD( rc = m_pDbSystem->dbCheck( // VISIT
//		DB_NAME_STR,
//		NULL,
//		NULL,
//		XFLM_DO_LOGICAL_CHECK,
//		NULL,
//		NULL)))
//	{
//		MAKE_FLM_ERROR_STRING("dbCheck failed", m_szDetails, rc);
//		goto Exit;
//	}

	endTest("PASS");

Exit:

	if ( bStartedTrans)
	{
		if (RC_OK(rc))
		{
			rc = m_pDb->transCommit();
		}
		else
		{
			m_pDb->transAbort();
		}
	}

	if ( RC_BAD( rc))
	{
		endTest("FAIL");
	}

	if (pRootElement)
	{
		pRootElement->Release();
	}

	if (pElement)
	{
		pElement->Release();
	}

	if ( pUniqueElement)
	{
		pUniqueElement->Release();
	}

	shutdownTestState( DB_NAME_STR, bDibCreated);
	return rc;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IDOMNodeTestImpl::concurrentValueModifyTests( void)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pRootElement = NULL;
	IF_DOMNode *	pBinNode = NULL;
	IF_DOMNode *	pTextNode = NULL;
	IF_DOMNode *	pOtherBinNode = NULL;
	IF_DOMNode *	pOtherNumNode = NULL;
	IF_DOMNode *	pNumNode = NULL;
	FLMBOOL			bStartedTrans = FALSE;
	FLMUINT			uiBinId = 0;
	FLMUINT			uiTextId = 0;
	FLMUINT			uiNumId = 0;
	FLMUINT			uiNewColl = 0;
	FLMBYTE			pucBinVal[] = {0x01,0x02,0x03,0x04,0x05};
	FLMBOOL			bDibCreated = FALSE;

	beginTest( 
		"Simultaneous Streaming Test Setup", 
		"Set up the database for the simultaneous streaming tests",
		"Self explanatory",
		"No Additional Details.");

	if ( RC_BAD( rc = initCleanTestState( DB_NAME_STR)))
	{
		goto Exit;
	}
	bDibCreated = TRUE;

	// Start an update transaction

	if (RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed", m_szDetails, rc);
		goto Exit;
	}
	bStartedTrans = TRUE;
	
	if( RC_BAD( rc = m_pDb->createElementDef(
		NULL, "bin_val", XFLM_BINARY_TYPE, &uiBinId, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "createElementDef failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->createElementDef(
		NULL, "text_val", XFLM_TEXT_TYPE, &uiTextId, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "createElementDef failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->createElementDef(
		NULL, "num_val", XFLM_NUMBER_TYPE, &uiNumId, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "createElementDef failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->createCollectionDef("new_collection", &uiNewColl)))
	{
		MAKE_FLM_ERROR_STRING( "createCollectionDef failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->createRootElement( 
		uiNewColl,	ELM_ELEMENT_TAG, &pRootElement)))
	{
		MAKE_FLM_ERROR_STRING( "createRootElement failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pRootElement->createNode( m_pDb,
										ELEMENT_NODE, uiBinId, XFLM_FIRST_CHILD,
										&pOtherBinNode, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "createNode failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pRootElement->createNode( m_pDb,
										ELEMENT_NODE, uiNumId, XFLM_FIRST_CHILD,
										&pOtherNumNode, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "createNode failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->createRootElement( XFLM_DATA_COLLECTION,
										ELM_ELEMENT_TAG, &pRootElement)))
	{
		MAKE_FLM_ERROR_STRING( "createRootElement failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pRootElement->createNode( m_pDb,
										ELEMENT_NODE, uiBinId, XFLM_FIRST_CHILD,
										&pBinNode, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "createNode failed", m_szDetails, rc);
		goto Exit;
	}
	
	if( RC_BAD( rc = pRootElement->createNode( m_pDb,
										ELEMENT_NODE, uiTextId, XFLM_FIRST_CHILD,
										&pTextNode, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "createNode failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pRootElement->createNode( m_pDb,
										ELEMENT_NODE, uiNumId, XFLM_FIRST_CHILD,
										&pNumNode, NULL)))
	{
		MAKE_FLM_ERROR_STRING( "createNode failed", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

	
	if (RC_BAD( rc = m_pDb->transCommit()))
	{
		MAKE_FLM_ERROR_STRING( "transCommit failed", m_szDetails, rc);
		goto Exit;
	}
	bStartedTrans = FALSE;

	beginTest( 
		"Simultaneous Streaming Test #1", 
		"Make sure we disallow setting of values while a streaming update has not completed",
		"Self explanatory",
		"No Additional Details.");

	if (RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed", m_szDetails, rc);
		goto Exit;
	}
	bStartedTrans = TRUE;

	if ( RC_BAD( rc = pBinNode->setBinary( 
		m_pDb, pucBinVal, sizeof(pucBinVal), FALSE)))
	{
		MAKE_FLM_ERROR_STRING( "setBinary failed", m_szDetails, rc);
		goto Exit;
	}

	if ( ( rc = pNumNode->setUINT( m_pDb, 123)) != NE_XFLM_INPUT_PENDING)
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		MAKE_FLM_ERROR_STRING( "setUINT was allowed", m_szDetails, rc);
		goto Exit;
	}

	if (RC_BAD( rc = m_pDb->transAbort()))
	{
		MAKE_FLM_ERROR_STRING( "transAbort failed", m_szDetails, rc);
		goto Exit;
	}
	bStartedTrans = FALSE;

	if (RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		MAKE_FLM_ERROR_STRING( "transBegin failed", m_szDetails, rc);
		goto Exit;
	}
	bStartedTrans = TRUE;

	if ( ( rc = pBinNode->setBinary( 
		m_pDb, pucBinVal, sizeof(pucBinVal), FALSE)) != NE_XFLM_INPUT_PENDING)
	{
		MAKE_FLM_ERROR_STRING( "unexpected rc from setBinary", m_szDetails, rc);
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}

	if ( ( rc = pOtherNumNode->setUINT( m_pDb, 123)) != NE_XFLM_INPUT_PENDING)
	{
		MAKE_FLM_ERROR_STRING( "unexpected rc from setUINT", m_szDetails, rc);
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}

	if ( ( rc = pOtherBinNode->setBinary( 
		m_pDb, pucBinVal, sizeof(pucBinVal), FALSE)) != NE_XFLM_INPUT_PENDING)
	{
		MAKE_FLM_ERROR_STRING( "unexpected rc from setBinary", m_szDetails, rc);
		rc = NE_XFLM_FAILURE;
		goto Exit;
	}

	if (RC_BAD( rc = m_pDb->transAbort()))
	{
		MAKE_FLM_ERROR_STRING( "transAbort failed", m_szDetails, rc);
		goto Exit;
	}
	bStartedTrans = FALSE;

	endTest("PASS");

Exit:

	if ( pRootElement)
	{
		pRootElement->Release();
	}

	if ( pBinNode)
	{
		pBinNode->Release();
	}

	if ( pOtherBinNode)
	{
		pOtherBinNode->Release();
	}

	if ( pTextNode)
	{
		pTextNode->Release();
	}

	if ( pOtherNumNode)
	{
		pOtherNumNode->Release();
	}

	if ( pNumNode)
	{
		pNumNode->Release();
	}

	if ( bStartedTrans)
	{
		if (RC_OK(rc))
		{
			rc = m_pDb->transCommit();
		}
		else
		{
			m_pDb->transAbort();
		}
	}

	if ( RC_BAD( rc))
	{
		endTest("FAIL");
	}

	shutdownTestState( DB_NAME_STR, bDibCreated);
	return( rc);
}
