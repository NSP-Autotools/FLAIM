//------------------------------------------------------------------------------
// Desc:	Sample application
//
// Tabs:	3
//
//		Copyright (c) 2002-2005 Novell, Inc. All Rights Reserved.
//
//		This program is free software; you can redistribute it and/or
//		modify it under the terms of version 2.1 of the GNU Lesser General Public
//		License as published by the Free Software Foundation.
//
//		This program is distributed in the hope that it will be useful,
//		but WITHOUT ANY WARRANTY; without even the implied warranty of
//		MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//		GNU Lesser General Public License for more details.
//
//		You should have received a copy of the GNU Lesser General Public License
//		along with this program; if not, contact Novell, Inc.
//
//		To contact Novell about this file by physical or electronic mail,
//		you may find current contact information at www.novell.com
//
// $Id: sample.cpp 3102 2006-01-10 10:15:17 -0700 (Tue, 10 Jan 2006) ahodgkinson $
//------------------------------------------------------------------------------

#include "xflaim.h"

#ifdef FLM_WIN
	#define UI64FormatStr			"I64u"
	#define I64FormatStr				"I64d"
	#include <windows.h>
#elif defined( FLM_SOLARIS)
	#define UI64FormatStr			"llu"
	#define I64FormatStr				"lld"
#else
	#define UI64FormatStr			"Lu"
	#define I64FormatStr				"Ld"
#endif
#include <stdio.h>
#include <string.h>

#define DB_NAME_STR					"tst.db"
#define BACKUP_NAME_STR				"tst.bak"
#define NEW_NAME_STR					"new.db"

FLMBOOL	gv_bShutdown = FALSE;

void printMessage(
	const char *		pszMessage,
	FLMUINT				uiLevel = 0);

RCODE printDocument(
	IF_Db *				pDb,
	IF_DOMNode *		pRootNode);

RCODE processAttributes(
	IF_Db *				pDb,
	IF_DOMNode *		pNode,
	FLMBYTE **			ppszLine);

#ifdef FLM_RING_ZERO_NLM
	#define main		nlm_main
#endif

/***************************************************************************
Desc:	Program entry point (main)
****************************************************************************/
extern "C" int main(
	int,			// iArgC,
	char **		// ppucArgV
	)
{
	RCODE						rc = NE_XFLM_OK;
	FLMBYTE					ucMsgBuf[ 200];
	XFLM_CREATE_OPTS		createOpts;
	IF_DbSystem *			pDbSystem = NULL;
	IF_Db *					pDb = NULL;
	IF_Backup *				pBackup = NULL;
	IF_DOMNode *			pTmpNode = NULL;
	FLMUINT					uiMusicElementDef=0;
	FLMUINT					uiCdElementDef=0;
	FLMUINT					uiTitleElementDef=0;
	FLMUINT					uiArtistElementDef=0;
	FLMUINT					uiTracksElementDef=0;
	FLMUINT					uiNoAttrDef=0;
	FLMUINT					uiTitleAttrDef=0;
	FLMUINT					uiCollectionDef=0;
	FLMBOOL					bTranStarted = FALSE;
	IF_DOMNode *			pCdChild = NULL;
	IF_DOMNode *			pCdElement = NULL;
	IF_DOMNode *			pMusicElement = NULL;
	IF_DOMNode *			pCdAttr = NULL;
	IF_DOMNode *			pTrackNode = NULL;
	IF_Query *				pQuery = NULL;
	IF_PosIStream	*		pPosIStream = NULL;
	IF_PosIStream *		pIStream = NULL;
	FLMUINT					uiIndex = 1;
	FLMBYTE					ucUntilKeyBuf[ XFLM_MAX_KEY_SIZE];
	FLMUINT					uiUntilKeyLen;
	FLMBYTE					ucCurrentKeyBuf[ XFLM_MAX_KEY_SIZE];
	FLMUINT					uiCurrentKeyLen;
	FLMUINT64				ui64DocId;
	char						ucTitle[ 200];
	FLMUINT					uiTitleLen;
	const char *			ppszPath[] = 
				{
					"7001e10c.xml",
					"70028663.xml",
					"70037c08.xml",
					"70040b08.xml",
					"70044808.xml",
					"70045109.xml",
					"70045e09.xml",
					"70046709.xml",
					"7004920a.xml",
					""
				};
	FLMUINT					uiLoop;
	IF_DataVector *		pFromKeyV = NULL;
	IF_DataVector *		pUntilKeyV = NULL;
	const char *			pszQueryString = "/music/cd/tracks[@title~=\"we our in luv\"]";
	const char *			pszIndex =
		"<xflaim:Index "
			"xmlns:xflaim=\"http://www.novell.com/XMLDatabase/Schema\" "
			"xflaim:name=\"title_IX\">"
			"<xflaim:ElementComponent "
				"xflaim:name=\"title\" "
				"xflaim:IndexOn=\"value\" "
				"xflaim:KeyComponent=\"1\"/>"
		"</xflaim:Index>";

	// Allocate a DbSystem object

	if( RC_BAD( rc = FlmAllocDbSystem( &pDbSystem)))
	{
		goto Exit;
	}

	// Create the database.  This code will remove the database first if it
	// exists.  Once removed, it will try to create it.  If that fails for
	// any reason other than the database already exists, it will exit.

RetryCreate:

	f_memset( &createOpts, 0, sizeof( XFLM_CREATE_OPTS));
	
	if( RC_BAD( rc = pDbSystem->dbCreate( DB_NAME_STR, NULL, NULL, NULL,
			NULL, &createOpts, &pDb)))
	{
		if( RC_BAD( rc == NE_XFLM_FILE_EXISTS))
		{
			if( RC_BAD( rc = pDbSystem->dbRemove( DB_NAME_STR, NULL, NULL, TRUE)))
			{
				goto Exit;
			}

			goto RetryCreate;
		}
		else
		{
			goto Exit;
		}
	}

	// Start an update transaction.  All access to the database should
	// be done within the confines of a transaction.  When changes are being
	// made to the database, an UPDATE transaction is required.  It is best to
	// keep transactions small if possible.

	if( RC_BAD( rc = pDb->transBegin( XFLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
	{
		goto Exit;
	}
	bTranStarted = TRUE;

	// Let's create a document to demonstrate how we use the DOM...
	// Our document will catalog a music CD.  It will look like:

	// <music>
	// 	<cd>
	// 		<title>We Are In Love</title>
	// 		<artist>Harry Connick Jr.</artist>
	// 		<tracks no="1" title="We Are In Love"/>
	// 		<tracks no="2" title="Only 'Cause I Don't Have You"/>
	// 	</cd>
	// </music>

	// To accomplish this, we must define the elements of the document and a new
	// collection to store the document in.

	// Create some element definitions for our document.
	// Create the "music" element definition.

	if( RC_BAD( rc = pDb->createElementDef( NULL, "music", 
		XFLM_NODATA_TYPE, &uiMusicElementDef)))
	{
		goto Exit;
	}

	// Create the "cd" element definition.

	if( RC_BAD( rc = pDb->createElementDef( NULL, "cd",
		XFLM_NODATA_TYPE, &uiCdElementDef)))
	{
		goto Exit;
	}

	// Create the "title" element definition.

	if( RC_BAD( rc = pDb->createElementDef( NULL, "title",
		XFLM_TEXT_TYPE, &uiTitleElementDef)))
	{
		goto Exit;
	}

	// Create the "artist" element definition.

	if( RC_BAD( rc = pDb->createElementDef( NULL, "artist",
		XFLM_TEXT_TYPE, &uiArtistElementDef)))
	{
		goto Exit;
	}

	// Create the "tracks" element definition.

	if( RC_BAD( rc = pDb->createElementDef( NULL, "tracks",
		XFLM_NODATA_TYPE, &uiTracksElementDef)))
	{
		goto Exit;
	}

	// Create the "no" attribute definition.

	if( RC_BAD( rc = pDb->createAttributeDef( NULL, "no",
		XFLM_NUMBER_TYPE, &uiNoAttrDef)))
	{
		goto Exit;
	}

	// Create the "title" attribute definition.

	if( RC_BAD( rc = pDb->createAttributeDef( NULL, "title",
		XFLM_TEXT_TYPE, &uiTitleAttrDef)))
	{
		goto Exit;
	}

	// Create our special music collection for storing music stuff.

	if( RC_BAD( rc = pDb->createCollectionDef( "Music Collection",
		&uiCollectionDef)))
	{
		goto Exit;
	}

	// We now have all of the definitions we need to build our document.
	// Lets first create a new document, followed by its members...

	// Create the "music" root element

	if( RC_BAD( rc = pDb->createRootElement( uiCollectionDef, uiMusicElementDef,
		&pMusicElement)))
	{
		goto Exit;
	}

	// Create the "cd" element

	if( RC_BAD( rc = pMusicElement->createNode( pDb, ELEMENT_NODE,
		uiCdElementDef, XFLM_FIRST_CHILD, &pCdElement)))
	{
		goto Exit;
	}
	
	// Create the "title" element

	if( RC_BAD( rc = pCdElement->createNode( pDb, ELEMENT_NODE,
		uiTitleElementDef, XFLM_FIRST_CHILD, &pCdChild)))
	{
		goto Exit;
	}

	// Set the value for the title.

	if (RC_BAD( rc = pCdChild->setUTF8( pDb, (FLMBYTE *)"We Are In Love")))
	{
		goto Exit;
	}
	
	// Create the "artist" element

	if( RC_BAD( rc = pCdElement->createNode( pDb, ELEMENT_NODE,
		uiArtistElementDef, XFLM_LAST_CHILD, &pCdChild)))
	{
		goto Exit;
	}

	// Set the value for the title.

	if (RC_BAD( rc = pCdChild->setUTF8( pDb, (FLMBYTE *)"Harry Connick Jr.")))
	{
		goto Exit;
	}
	
	// Create the first "tracks" element

	if( RC_BAD( rc = pCdElement->createNode( pDb, ELEMENT_NODE,
		uiTracksElementDef, XFLM_LAST_CHILD, &pCdChild)))
	{
		goto Exit;
	}

	// Create the "no." attribute, then the "title" attribute.
	
	if (RC_BAD( rc = pCdChild->createAttribute( pDb, uiNoAttrDef, &pCdAttr)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pCdAttr->setUINT( pDb, (FLMUINT)1)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pCdChild->createAttribute( pDb, uiTitleAttrDef, &pCdAttr)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pCdAttr->setUTF8( pDb, (FLMBYTE *)"We Are In Love")))
	{
		goto Exit;
	}

	// Create the next "tracks" element

	if( RC_BAD( rc = pCdElement->createNode( pDb, ELEMENT_NODE,
		uiTracksElementDef, XFLM_LAST_CHILD, &pCdChild)))
	{
		goto Exit;
	}

	// An alternate way to create the attributes and set their values is to use
	// the parent element to set the attribute values.

	// Create the two attributes.

	if (RC_BAD( rc = pCdChild->createAttribute( pDb, uiNoAttrDef, &pCdAttr)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pCdChild->createAttribute( pDb, uiTitleAttrDef, &pCdAttr)))
	{
		goto Exit;
	}

	// Using the parent of the attributes, set their values by specifying
	// which attribute to set.

	if (RC_BAD( rc = pCdChild->setAttributeValueUINT( pDb, uiNoAttrDef,
		(FLMUINT)2)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pCdChild->setAttributeValueUTF8(
		pDb, uiTitleAttrDef, (FLMBYTE *)"Only 'Cause I Don't Have You")))
	{
		goto Exit;
	}

	// It is always good practice to call documentDone whenever updates to a
	// document are completed.

	if (RC_BAD( rc = pDb->documentDone( pMusicElement)))
	{
		goto Exit;
	}

	// Commit the transaction

	if( RC_BAD( rc = pDb->transCommit()))
	{
		goto Exit;
	}
	bTranStarted= FALSE;

	// Now we want to read back our document, and display it for all 
	// the world to see...:^)

	// Start a read transaction

	if( RC_BAD( rc = pDb->transBegin( XFLM_READ_TRANS, 0)))
	{
		goto Exit;
	}
	bTranStarted = TRUE;

	// Read the nodes.  Start with the document node.

	if( RC_BAD( rc = pDb->getFirstDocument( uiCollectionDef, &pTmpNode)))
	{
		goto Exit;
	}

	// printDocument is a simple function that walks through the document
	// and displays it, node by node as an XML document.  It assumes the 
	// node passed in is the root node.  If it is not, then the display will
	// look a bit odd, as you will see later in this sample program.

	if (RC_BAD( rc = printDocument( pDb, pTmpNode)))
	{
		goto Exit;
	}

	// Commit the transaction

	if( RC_BAD( rc = pDb->transCommit()))
	{
		goto Exit;
	}
	bTranStarted = FALSE;

	// Start an update transaction

	if( RC_BAD( rc = pDb->transBegin( XFLM_READ_TRANS, 0)))
	{
		goto Exit;
	}
	bTranStarted = TRUE;

	// Let's do a query on this document...  Our query (above) is
	// deliberately miss-spelling some of the words.  It is looking for
	// the music "tracks" with the title "We Are In Love".
	// The syntax ~= means approximately equals, and is a "sounds like" 
	// search.

	if (RC_BAD( rc = pDbSystem->createIFQuery( &pQuery)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pQuery->setCollection( uiCollectionDef)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pQuery->setupQueryExpr( pDb, pszQueryString)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pQuery->getFirst( pDb, &pTrackNode)))
	{
		goto Exit;
	}

	printMessage( "Query:");
	printMessage( pszQueryString);

	// This will pring the tracks node, followed by all of the closing parent
	// nodes.

	if (RC_BAD( rc = printDocument( pDb, pTrackNode)))
	{
		goto Exit;
	}

	pQuery->Release();
	pQuery = NULL;

	// Commit the transaction

	if( RC_BAD( rc = pDb->transCommit()))
	{
		goto Exit;
	}
	bTranStarted = FALSE;

	// Let's create an index to make searching easier.  An index is essentially
	// just another XML document to XFlaim, however it is created in the dictionary
	// collection rather than in a data collection.  There are two ways to create
	// a document in XFlaim.  One way (demonstrated above) is to create each node
	// of the document, one at a time.  The other is to import the document.
	// Creating the index will demonstrate importing the document.  Our
	// index definition is shown as follows:

	// <xflaim:Index
	// 	xmlns:xflaim="http://www.novell.com/XMLDatabase/Schema"
	// 	xflaim:name="title_IX">
	// 	<xflaim:ElementComponent
	// 		xflaim:name="title"
	// 		xflaim:DictNum="1"
	// 		xflaim:IndexOn="value"/>
	// </xflaim:Index>"

	// For our purposes, we will create a local variable (pszIndex) which
	// holds the index document.  We will import that using the importDocument
	// method of the pDb object.

	// Import the index...
	// We first need to create a BufferIStream object to stream the document
	// from...
	// Start an update transaction

	if( RC_BAD( rc = pDb->transBegin( XFLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
	{
		goto Exit;
	}
	bTranStarted = TRUE;

	if ( RC_BAD( rc = pDbSystem->openBufferIStream( pszIndex, 
		f_strlen( pszIndex), &pPosIStream)))
	{
		goto Exit;
	}

	if ( RC_BAD( rc = pDb->import( pPosIStream, XFLM_DICT_COLLECTION)))
	{
		goto Exit;
	}

	// Commit the transaction

	if( RC_BAD( rc = pDb->transCommit()))
	{
		goto Exit;
	}
	bTranStarted = FALSE;

	// Now, let's get some additional documents in so we can do some more
	// interesting stuff using a IF_DataVector.  The documents we are
	// interested in searching are the CD music documents.
	// They have the following format:

	// <?xml version="1.0"?>
	// <disc>
	// 	<id>00097210</id>
	// 	<length>2420</length>
	// 	<title>Frank Sinatra / Blue skies</title>
	// 	<genre>cddb/jazz</genre>
	// 	<track index="1" offset="150">blue skies</track>
	// 	.
	// 	.
	// 	.
	// </disc>

	if( RC_BAD( rc = pDb->transBegin( XFLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
	{
		goto Exit;
	}
	bTranStarted = TRUE;

	// We will first need an input file stream.

	uiLoop = 0;
	while( f_strlen( ppszPath[ uiLoop]))
	{
		if( RC_BAD( rc = pDbSystem->openFileIStream( 
			ppszPath[ uiLoop], &pIStream)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = pDb->import( pIStream, XFLM_DATA_COLLECTION)))
		{
			goto Exit;
		}
		
		uiLoop++;
	}

	// Commit the transaction

	if( RC_BAD( rc = pDb->transCommit()))
	{
		goto Exit;
	}
	bTranStarted = FALSE;

	// Let's get some IF_DataVectors to work with.

	if( RC_BAD( rc = pDb->transBegin( XFLM_READ_TRANS, 0)))
	{
		goto Exit;
	}
	bTranStarted = TRUE;

	if (RC_BAD( rc = pDbSystem->createIFDataVector( &pFromKeyV)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pDbSystem->createIFDataVector( &pUntilKeyV)))
	{
		goto Exit;
	}

	// We need to get the index.

	// Now to search	the index above for all document titles in the index and
	// display the document Id and title for each node.  Note that the index
	// number is known to be 1.

	if (RC_BAD( rc = pDb->keyRetrieve( uiIndex, NULL, XFLM_FIRST, pFromKeyV)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pDb->keyRetrieve( uiIndex, NULL, XFLM_LAST, pUntilKeyV)))
	{
		goto Exit;
	}

	// Save the UntilKey value to compare with later.

	if (RC_BAD( rc = pUntilKeyV->outputKey( pDb, uiIndex, FALSE,
		&ucUntilKeyBuf[0], XFLM_MAX_KEY_SIZE, &uiUntilKeyLen)))
	{
		goto Exit;
	}

	for (;;)
	{
		// Display the current Document Id and the title.

		ui64DocId = pFromKeyV->getDocumentID();

		uiTitleLen = sizeof(ucTitle);
		if (RC_BAD( rc = pFromKeyV->getUTF8( 0, 
			(FLMBYTE *)&ucTitle[0], &uiTitleLen)))
		{
			goto Exit;
		}

		f_sprintf( (char *)&ucMsgBuf[0], "DocId: %"UI64FormatStr"\n%s\n",
			ui64DocId, ucTitle);
		printMessage( (char *)&ucMsgBuf[0]);

		// Check to see if this key matches the last or UntilKeyV value.

		if (RC_BAD( rc = pFromKeyV->outputKey( pDb, uiIndex, FALSE,
			&ucCurrentKeyBuf[0], XFLM_MAX_KEY_SIZE, &uiCurrentKeyLen)))
		{
			goto Exit;
		}

		if (uiCurrentKeyLen == uiUntilKeyLen)
		{
			if( f_memcmp( ucCurrentKeyBuf, ucUntilKeyBuf, uiCurrentKeyLen) == 0)
			{
				// We are done!

				break;
			}
		}

		// Get the next key.

		if (RC_BAD( rc = pDb->keyRetrieve( uiIndex, pFromKeyV,
			XFLM_EXCL, pFromKeyV)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = pDb->transCommit()))
	{
		goto Exit;
	}
	bTranStarted = FALSE;

	// Close the database

	pDb->Release();
	pDb = NULL;

	// Close all unused files

	if( RC_BAD( rc = pDbSystem->closeUnusedFiles( 0)))
	{
		goto Exit;
	}

	// Re-open the database

	if( RC_BAD( rc = pDbSystem->dbOpen( DB_NAME_STR, NULL, 
		NULL, NULL, FALSE, &pDb)))
	{
		goto Exit;
	}

	// Backup the database

	if( RC_BAD( rc = pDb->backupBegin( XFLM_FULL_BACKUP, XFLM_READ_TRANS,
		0, &pBackup)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pBackup->backup( BACKUP_NAME_STR, NULL, NULL, NULL, NULL)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pBackup->endBackup()))
	{
		goto Exit;
	}

	pBackup->Release();
	pBackup = NULL;

	// Close the database again

	pDb->Release();
	pDb = NULL;

	if( RC_BAD( rc = pDbSystem->closeUnusedFiles( 0)))
	{
		goto Exit;
	}

	// Remove the database

	if( RC_BAD( rc = pDbSystem->dbRemove( DB_NAME_STR, NULL, NULL, TRUE)))
	{
		goto Exit;
	}

	// Restore the database

	if( RC_BAD( rc = pDbSystem->dbRestore( DB_NAME_STR, NULL, NULL,
		BACKUP_NAME_STR, NULL, NULL, NULL)))
	{
		goto Exit;
	}

	// Rename the database

	if( RC_BAD( rc = pDbSystem->dbRename( DB_NAME_STR, NULL, NULL,
			NEW_NAME_STR, TRUE, NULL)))
	{
		goto Exit;
	}

	// Copy the database

	if( RC_BAD( rc = pDbSystem->dbCopy( NEW_NAME_STR, NULL, NULL,
		DB_NAME_STR, NULL, NULL, NULL)))
	{
		goto Exit;
	}

	// Remove the new database

	if( RC_BAD( rc = pDbSystem->dbRemove( NEW_NAME_STR, NULL, NULL, TRUE)))
	{
		goto Exit;
	}


Exit:

	if (bTranStarted && pDb)
	{
		(void)pDb->transAbort();
	}

	if (pCdChild)
	{
		pCdChild->Release();
	}

	if (pCdElement)
	{
		pCdElement->Release();
	}

	if( pMusicElement)
	{
		pMusicElement->Release();
	}

	if (pCdAttr)
	{
		pCdAttr->Release();
	}

	if (pTrackNode)
	{
		pTrackNode->Release();
	}

	if( pTmpNode)
	{
		pTmpNode->Release();
	}

	if (pQuery)
	{
		pQuery->Release();
	}

	if( pPosIStream)
	{
		pPosIStream->Release();
	}

	if (pIStream)
	{
		pIStream->Release();
	}

	if (pFromKeyV)
	{
		pFromKeyV->Release();
	}

	if (pUntilKeyV)
	{
		pUntilKeyV->Release();
	}

	if( pBackup)
	{
		pBackup->Release();
	}

	// Close the database object

	if( pDb)
	{
		pDb->Release();
	}

	if( RC_BAD( rc))
	{
		f_sprintf( (char *)ucMsgBuf, "Error 0x%04X\n",
											(unsigned)rc);
		printMessage( (char *)ucMsgBuf);
	}

	if( pDbSystem)
	{
		pDbSystem->Release();
	}

	return( 0);
}

/***************************************************************************
Desc:	Prints a string to stdout
****************************************************************************/
void printMessage(
	const char *		pszMessage,
	FLMUINT				uiLevel)
{
	unsigned int		uiLoop;

	for (uiLoop = 0; uiLoop < uiLevel; uiLoop++)
	{
		f_printf( " ");
	}
	
	f_printf( "%s\n", pszMessage);
}

/*=============================================================================
Desc:	Simple routine to display the contents of a document.
=============================================================================*/
RCODE printDocument(
	IF_Db *					pDb,
	IF_DOMNode *			pRootNode)
{
	RCODE						rc = NE_XFLM_OK;
	FLMBYTE					ucLine[ 200];
	FLMBYTE *				pszLine;
	FLMBOOL					bHasChildren = FALSE;
	FLMBOOL					bHadAttributes;
	FLMUINT					uiDataType;
	char						szName[ 50];
	FLMUINT					uiNameLen;
	IF_DOMNode *			pNode = pRootNode;
	char						szTemp[ 200];
	FLMUINT					uiTemp;
	FLMBOOL					bUnwinding = FALSE;
	FLMBOOL					bHadValue = FALSE;
	FLMUINT					uiThisLevel = 0;
	FLMUINT					uiNextLevel = 0;
	eDomNodeType			eNodeType;

	pNode->AddRef();
	pszLine = &ucLine[0];
	bHadAttributes = FALSE;

	while (pNode)
	{
		bHadValue = FALSE;

		// Write out the current line.
		
		if (pszLine != &ucLine[0])
		{
			printMessage( (char *)ucLine, uiThisLevel);
			pszLine = &ucLine[0];
		}

		// Adjust to the next level
		
		uiThisLevel = uiNextLevel;
		eNodeType = pNode->getNodeType();

		if( eNodeType == DOCUMENT_NODE)
		{
			if (!bUnwinding)
			{
				f_sprintf( (char *)pszLine, "<doc");
			}
			else
			{
				f_sprintf( (char *)pszLine, "</doc>");
			}
			pszLine += f_strlen( (char *)pszLine);
		}
		else if( eNodeType == ELEMENT_NODE)
		{
			if (RC_BAD( rc = pNode->getQualifiedName( 
				pDb, szName, sizeof(szName), &uiNameLen)))
			{
				goto Exit;
			}

			if (!bUnwinding)
			{
				f_sprintf( (char *)pszLine, "<%s", szName);
			}
			else
			{
				f_sprintf( (char *)pszLine, "</%s>", szName);
			}
			pszLine += f_strlen( (char *)pszLine);
		}

		if (!bUnwinding)
		{
			if (RC_BAD( rc = processAttributes( pDb, pNode, &pszLine)))
			{
				goto Exit;
			} 

			// We need to know if this node has children to 
			// know how to close the line.
			
			if (RC_BAD( rc = pNode->hasChildren( pDb, &bHasChildren)))
			{
				goto Exit;
			}

			if( eNodeType == DATA_NODE || eNodeType == COMMENT_NODE)
			{
				if (RC_BAD( rc = pNode->getDataType( pDb, &uiDataType)))
				{
					goto Exit;
				}

				if( eNodeType == COMMENT_NODE)
				{
					f_sprintf( (char *)pszLine, "<!--");
					pszLine += 4;
				}

				switch (uiDataType)
				{
					case XFLM_TEXT_TYPE:
					case XFLM_NUMBER_TYPE:
					{
						bHadValue = TRUE;
						if (RC_BAD( rc = pNode->getUTF8( pDb, (FLMBYTE *)szTemp, 
							sizeof(szTemp), 0, 200, &uiTemp)))
						{
							goto Exit;
						}
						f_sprintf( (char *)pszLine, "%s", szTemp);
						pszLine += uiTemp;
						break;
					}
					
					default:
					{
						// Do nothing at this time.  Should generate some 
						// type of error response.						
					}
					
					// Could also get binary data, but we won't handle it here.
				}

				if( eNodeType == COMMENT_NODE)
				{
					f_sprintf( (char *)pszLine, "-->");
					pszLine += 3;
				}
			}
			else
			{
				if ( !bHasChildren)
				{
					f_sprintf( (char *)pszLine, "/>");
					pszLine += 2;
				}
				else
				{
					f_sprintf( (char *)pszLine, ">");
					pszLine++;
				}
			}

			// Children
			
			if (bHasChildren)
			{
				// Get the child node.
				
				if (RC_BAD( rc = pNode->getFirstChild( pDb, &pNode)))
				{
					goto Exit;
				}
				
				uiNextLevel++;
				continue;
			}
		}

		bUnwinding = FALSE;
		if (RC_BAD( rc = pNode->getNextSibling( pDb, &pNode)))
		{
			if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				rc = NE_XFLM_OK;
			}
			else
			{
				goto Exit;
			}
		}
		else
		{
			continue;
		}

		// Get the parent if there is one.  Otherwise we are done.

		if (uiNextLevel)
		{
			--uiNextLevel;
		}

		if (RC_BAD( rc = pNode->getParentNode( pDb, &pNode)))
		{
			if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				goto Exit;
			}

			rc = NE_XFLM_OK;

			pNode->Release();
			pNode = NULL;
		}
		else
		{
			bUnwinding = TRUE;
		}
	}

	printMessage( (char *)ucLine);

Exit:

	return( rc);
}

/*=============================================================================
Desc:
=============================================================================*/
RCODE processAttributes(
	IF_Db *			pDb,
	IF_DOMNode *	pNode,
	FLMBYTE **		ppszLine)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pAttrNode = NULL;
	FLMBOOL			bHasAttrs;
	char				szName[ 50];
	FLMUINT			uiNameLen;
	FLMBYTE *		pszLine = *ppszLine;
	FLMUINT			uiDataType;
	FLMBYTE			szTemp[ 200];
	FLMUINT			uiTemp;

	if (RC_BAD( rc = pNode->hasAttributes( pDb, &bHasAttrs)))
	{
		goto Exit;
	}

	if( !bHasAttrs)
	{
		goto Exit;
	}

	// We have attributes.  Let's get them and add them to the line.
	
	if (RC_BAD( rc = pNode->getFirstAttribute( pDb, &pAttrNode)))
	{
		goto Exit;
	}

	for (;;)
	{
		if (RC_BAD( rc = pAttrNode->getQualifiedName( pDb, szName, 
			sizeof(szName), &uiNameLen)))
		{
			goto Exit;
		}

		f_sprintf( (char *)pszLine, " %s", szName);
		pszLine += (uiNameLen + 1);

		if (RC_BAD( rc = pAttrNode->getDataType( pDb, &uiDataType)))
		{
			goto Exit;
		}

		switch (uiDataType)
		{
			case XFLM_TEXT_TYPE:
			{
				if (RC_BAD( rc = pAttrNode->getUTF8( pDb, szTemp, sizeof( szTemp),
					0, 200, &uiTemp)))
				{
					goto Exit;
				}
				f_sprintf( (char *)pszLine, "=\"%s\"", szTemp);
				pszLine += (uiTemp + 3);
				break;
			}
			
			case XFLM_NUMBER_TYPE:
			{
				if (RC_BAD( rc = pAttrNode->getUINT( pDb, &uiTemp)))
				{
					goto Exit;
				}
				
				f_sprintf( (char *)szTemp, "%u", (unsigned)uiTemp);
				f_sprintf( (char *)pszLine, "=\"%s\"", szTemp);
				pszLine += (f_strlen( (char *)szTemp) + 3);
				break;
			}
			// Could also get binary data, but we won't handle it here.
		}

		// Get the next attribute, if any
		
		if (RC_BAD( rc = pAttrNode->getNextSibling( pDb, &pAttrNode)))
		{
			if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				rc = NE_XFLM_OK;
				break;
			}
			goto Exit;
		}
	}

	// Update the line pointer on the way out.

	*ppszLine = pszLine;

Exit:

	if (pAttrNode)
	{
		pAttrNode->Release();
	}

	return( rc);
}
