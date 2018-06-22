//------------------------------------------------------------------------------
// Desc:	DOM Nodes tests
// Tabs:	3
//
// Copyright (c) 2006-2007 Novell, Inc. All Rights Reserved.
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

using System;
using System.IO;
using System.Runtime.InteropServices;
using xflaim;

namespace cstest
{

	//--------------------------------------------------------------------------
	// Open database test.
	//--------------------------------------------------------------------------
	public class DOMNodesTest : Tester
	{
		private const uint NUM_CHILD_NODES = 500;
		private const uint BUILTIN_ATTRIBUTES =
			((uint)ReservedAttrTag.XFLM_LAST_RESERVED_ATTRIBUTE_TAG -
			 (uint)ReservedAttrTag.XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 1);

		public bool createDocumentTest(
			Db	db)
		{
			DOMNode		doc = null;
			DOMNode		docRoot = null;
			DOMNode		node = null;
			DOMNode		node2 = null;
			DOMNode		attr = null;
			ulong			ulDocId;
			ulong			ulNodeId;
			ulong			ulDocRootId;
			uint			uiTag;
			FlmDataType	eDataType;
			uint			uiSetValue = 12345;
			uint			uiGetValue;
			ulong			ulSetValue = 123456;
			ulong			ulGetValue;
			int			iSetValue = -12345;
			int			iGetValue;
			long			lSetValue = -123456;
			long			lGetValue;
			string		sSetValue = "String value";
			string		sGetValue = null;
			byte []		ucSetValue = new byte [] {0x01, 0x02, 0x03, 0x04, 0x05};
			byte []		ucGetValue = null;
			RCODE			rc;

			// Create a document
			
			beginTest( "Create document test");
			try
			{
				doc = db.createDocument( (uint)PredefinedXFlaimCollections.XFLM_DATA_COLLECTION);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "creating document");
				return( false);
			}

			// Create a node - can only be one element node subordinate to a document node.

			try
			{
				docRoot = doc.createNode( eDomNodeType.ELEMENT_NODE,
								(uint)ReservedElmTag.ELM_ELEMENT_TAG,
								eNodeInsertLoc.XFLM_FIRST_CHILD,
								docRoot);

			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "creating node");
				return( false);
			}

			// Create a node subordinate to the root element node.

			try
			{
				node = docRoot.createNode( eDomNodeType.ELEMENT_NODE,
					(uint)ReservedElmTag.ELM_ELEMENT_TAG,
					eNodeInsertLoc.XFLM_FIRST_CHILD,
					node);

			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "creating node");
				return( false);
			}

			// Get the document ID

			try
			{
				ulDocId = node.getDocumentId();
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling getDocumentId");
				return( false);
			}

			// Get the node ID

			try
			{
				ulNodeId = doc.getNodeId();
				ulDocRootId = docRoot.getNodeId();
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling getNodeId");
				return( false);
			}
			if (ulNodeId != ulDocId)
			{
				endTest( false, false);
				System.Console.WriteLine( "Incorrect document ID: NodeID: {0}, DocID: {1}",
						ulNodeId, ulDocId);
				return( false);
			}

			// Create a bunch of siblings and add attributes to them
	
			for (uint uiLoop = 0; uiLoop < NUM_CHILD_NODES - 1; uiLoop++)
			{
				try
				{
					node2 = node.createNode( eDomNodeType.ELEMENT_NODE,
						(uint)ReservedElmTag.ELM_ELEMENT_TAG,
						uiLoop % 2 == 0 ? eNodeInsertLoc.XFLM_NEXT_SIB : eNodeInsertLoc.XFLM_PREV_SIB,
						node2);
				}
				catch (XFlaimException ex)
				{
					endTest( false, ex, "calling createNode");
					return( false);
				}

				// Node better be an element node, and better not have child nodes.

				try
				{
					if (node2.getNodeType() != eDomNodeType.ELEMENT_NODE)
					{
						endTest( false, false);
						System.Console.WriteLine( "Illegal node type returned from getNodeType: {0}",
							node2.getNodeType());
						return( false);
					}
				}
				catch (XFlaimException ex)
				{
					endTest( false, ex, "calling getNodeType");
					return( false);
				}

	
				try
				{
					if (node2.hasChildren())
					{
						endTest( false, false);
						System.Console.WriteLine( "Node erroneously claims to have children");
						return( false);
					}
				}
				catch (XFlaimException ex)
				{
					endTest( false, ex, "calling hasChildren");
					return( false);
				}

				// Create an attribute and make sure it is present.

				uiTag = (uint)ReservedAttrTag.XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + (uiLoop % BUILTIN_ATTRIBUTES);
		
				try
				{
					attr = node2.createAttribute( uiTag, attr);
				}
				catch (XFlaimException ex)
				{
					endTest( false, ex, "calling createAttribute");
					return( false);
				}

				try
				{
					if (!node2.hasAttribute( uiTag))
					{
						endTest( false, false);
						System.Console.WriteLine( "Node is missing an attribute");
						return( false);
					}
				}
				catch (XFlaimException ex)
				{
					endTest( false, ex, "calling hasAttribute");
					return( false);
				}

				// Look up the tag's data type and set an appropriate value
				// Then retrieve it again for verification

				try
				{
					eDataType = attr.getDataType();
				}
				catch (XFlaimException ex)
				{
					endTest( false, ex, "calling getDataType");
					return( false);
				}

				switch (eDataType)
				{
					case FlmDataType.XFLM_NUMBER_TYPE:
						if (uiLoop % 4 == 1)
						{
							ulGetValue = ulSetValue + 1;
							try
							{
								attr.setULong( ulSetValue);
							}
							catch (XFlaimException ex)
							{
								endTest( false, ex, "calling setULong");
								return( false);
							}
							try
							{
								ulGetValue = attr.getULong();
							}
							catch (XFlaimException ex)
							{
								endTest( false, ex, "calling getULong");
								return( false);
							}
							if (ulSetValue != ulGetValue)
							{
								endTest( false, false);
								System.Console.WriteLine( "ulSetValue {0} != ulGetValue {1}",
									ulSetValue, ulGetValue);
								return( false);
							}
						}
						else if (uiLoop % 4 == 2)
						{
							lGetValue = lSetValue + 1;
							try
							{
								attr.setLong( lSetValue);
							}
							catch (XFlaimException ex)
							{
								endTest( false, ex, "calling setLong");
								return( false);
							}
							try
							{
								lGetValue = attr.getLong();
							}
							catch (XFlaimException ex)
							{
								endTest( false, ex, "calling getLong");
								return( false);
							}
							if (lSetValue != lGetValue)
							{
								endTest( false, false);
								System.Console.WriteLine( "lSetValue {0} != lGetValue {1}",
									lSetValue, lGetValue);
								return( false);
							}
						}
						else if (uiLoop % 4 == 3)
						{
							uiGetValue = uiSetValue + 1;
							try
							{
								attr.setUInt( uiSetValue);
							}
							catch (XFlaimException ex)
							{
								endTest( false, ex, "calling setUInt");
								return( false);
							}
							try
							{
								uiGetValue = attr.getUInt();
							}
							catch (XFlaimException ex)
							{
								endTest( false, ex, "calling getUInt");
								return( false);
							}
							if (uiSetValue != uiGetValue)
							{
								endTest( false, false);
								System.Console.WriteLine( "uiSetValue {0} != uiGetValue {1}",
									uiSetValue, uiGetValue);
								return( false);
							}
						}
						else
						{
							iGetValue = iSetValue + 1;
							try
							{
								attr.setInt( iSetValue);
							}
							catch (XFlaimException ex)
							{
								endTest( false, ex, "calling setInt");
								return( false);
							}
							try
							{
								iGetValue = attr.getInt();
							}
							catch (XFlaimException ex)
							{
								endTest( false, ex, "calling getInt");
								return( false);
							}
							if (iSetValue != iGetValue)
							{
								endTest( false, false);
								System.Console.WriteLine( "iSetValue {0} != iGetValue {1}",
									iSetValue, iGetValue);
								return( false);
							}
						}
						break;
					case FlmDataType.XFLM_TEXT_TYPE:
						sGetValue = "";
						try
						{
							attr.setString( sSetValue);
						}
						catch (XFlaimException ex)
						{
							endTest( false, ex, "calling setString");
							return( false);
						}
						try
						{
							sGetValue = attr.getString();
						}
						catch (XFlaimException ex)
						{
							endTest( false, ex, "calling getString");
							return( false);
						}
						if (sSetValue != sGetValue)
						{
							endTest( false, false);
							System.Console.WriteLine( "sSetValue [{0}] != sGetValue [{1}]",
								sSetValue, sGetValue);
							return( false);
						}
						break;
					case FlmDataType.XFLM_BINARY_TYPE:
					{
						bool	bDataSame;

						ucGetValue = null;
						try
						{
							attr.setBinary( ucSetValue);
						}
						catch (XFlaimException ex)
						{
							endTest( false, ex, "calling setBinary");
							return( false);
						}
						try
						{
							ucGetValue = attr.getBinary();
						}
						catch (XFlaimException ex)
						{
							endTest( false, ex, "calling getBinary");
							return( false);
						}
						bDataSame = true;
						if (ucSetValue.Length != ucGetValue.Length)
						{
							bDataSame = false;
						}
						else
						{
							for( uint uiLoop2 = 0; uiLoop2 < ucSetValue.Length; uiLoop2++)
							{
								if (ucSetValue [uiLoop2] != ucGetValue [uiLoop2])
								{
									bDataSame = false;
									break;
								}
							}
						}
						if (!bDataSame)
						{
							endTest( false, false);
							System.Console.WriteLine( "Set binary data does not match get binary data");
							System.Console.Write( "Set Binary Data Length: {0}\n[", ucSetValue.Length);
							for( uint uiLoop2 = 0; uiLoop2 < ucSetValue.Length; uiLoop2++)
							{
								System.Console.Write( "{0} ", ucSetValue [uiLoop2]);
							}
							System.Console.WriteLine( "]");
							System.Console.Write( "Get Binary Data Length: {0}\n[", ucGetValue.Length);
							for( uint uiLoop2 = 0; uiLoop2 < ucGetValue.Length; uiLoop2++)
							{
								System.Console.Write( "{0} ", ucGetValue [uiLoop2]);
							}
							System.Console.WriteLine( "]");
							return( false);
						}
						break;
					}
					default:
						endTest( false, false);
						System.Console.WriteLine( "Invalid data type for attr {0}", eDataType);
						return( false);
				}

				// Since there's only one attribute, either one of these functions will do
				if ( (uiLoop % 2) == 0) 
				{
					try
					{
						attr = node2.getFirstAttribute( attr);
					}
					catch (XFlaimException ex)
					{
						endTest( false, ex, "calling getFirstAttribute");
						return( false);
					}
				}
				else
				{
					try
					{
						attr = node2.getAttribute( uiTag, attr);
					}
					catch (XFlaimException ex)
					{
						endTest( false, ex, "calling getAttribute");
						return( false);
					}
				}

				// We gave these nodes one and only one attribute
				// The attributes should have no siblings

				rc = RCODE.NE_XFLM_OK;
				try
				{
					node2 = attr.getPreviousSibling( node2);
				}
				catch (XFlaimException ex)
				{
					rc = ex.getRCode();
					if (rc != RCODE.NE_XFLM_DOM_NODE_NOT_FOUND)
					{
						endTest( false, ex, "calling getPreviousSibling");
						return( false);
					}
				}
				if (rc == RCODE.NE_XFLM_OK)
				{
					endTest( false, false);
					System.Console.WriteLine( "Attribute should not have a previous sibling");
					return( false);
				}

				rc = RCODE.NE_XFLM_OK;
				try
				{
					node2 = attr.getNextSibling( node2);
				}
				catch (XFlaimException ex)
				{
					rc = ex.getRCode();
					if (rc != RCODE.NE_XFLM_DOM_NODE_NOT_FOUND)
					{
						endTest( false, ex, "calling getNextSibling");
						return( false);
					}
				}
				if (rc == RCODE.NE_XFLM_OK)
				{
					endTest( false, false);
					System.Console.WriteLine( "Attribute should not have a next sibling");
					return( false);
				}
			}

			// Document node should now have children.

			try
			{
				if (!docRoot.hasChildren())
				{
					endTest( false, false);
					System.Console.WriteLine( "Document root erroneously reports that it has no child nodes");
					return( false);
				}
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling hasChildren");
				return( false);
			}

			// Reposition to the first child under the document root,
			// iterate through its children (first->last) and perform 
			// various DOMNode ops

			for (uint uiLoop = 0; uiLoop < NUM_CHILD_NODES; uiLoop++)
			{
				if (uiLoop == 0)
				{
					try
					{
						node = docRoot.getFirstChild( node);
					}
					catch (XFlaimException ex)
					{
						endTest( false, ex, "calling getFirstChild");
						return( false);
					}
				}
				else
				{
					try
					{
						node = node.getNextSibling( node);
					}
					catch (XFlaimException ex)
					{
						endTest( false, ex, "calling getNextSibling");
						return( false);
					}
				}

				try
				{
					if (node.getParentId() != ulDocRootId)
					{
						endTest( false, false);
						System.Console.WriteLine( "Node's parent ID {0} does not match document id {1}",
								node.getParentId(), ulDocId);
						return( false);
					}
				}
				catch (XFlaimException ex)
				{
					endTest( false, ex, "calling getParentId");
					return( false);
				}
			}

			// There should be no more siblings

			rc = RCODE.NE_XFLM_OK;
			try
			{
				node = node.getNextSibling( node);
			}
			catch (XFlaimException ex)
			{
				rc = ex.getRCode();
				if (rc != RCODE.NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					endTest( false, ex, "calling getNextSibling");
					return( false);
				}
			}
			if (rc == RCODE.NE_XFLM_OK)
			{
				endTest( false, false);
				System.Console.WriteLine( "Should have been no more next siblings");
				return( false);
			}
			endTest( false, true);


			beginTest( "Delete DOM nodes test");

			// Move backwards through the siblings deleting them (except the last one)

			for (uint uiLoop = 0; uiLoop < NUM_CHILD_NODES; uiLoop++)
			{
				if (uiLoop == 0)
				{
					try
					{
						node = docRoot.getLastChild( node);
					}
					catch (XFlaimException ex)
					{
						endTest( false, ex, "calling getLastChild");
						return( false);
					}
				}
				else
				{
					try
					{
						node2 = node.getPreviousSibling( node2);
					}
					catch (XFlaimException ex)
					{
						endTest( false, ex, "calling getPreviousSibling");
						return( false);
					}
					try
					{
						node.deleteNode();
					}
					catch (XFlaimException ex)
					{
						endTest( false, ex, "calling deleteNode");
						return( false);
					}
					node = node2;
					node2 = null;
				}
			}

			// There should be no more siblings

			rc = RCODE.NE_XFLM_OK;
			try
			{
				node2 = node.getPreviousSibling( node2);
			}
			catch (XFlaimException ex)
			{
				rc = ex.getRCode();
				if (rc != RCODE.NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					endTest( false, ex, "calling getPreviousSibling");
					return( false);
				}
			}
			if (rc == RCODE.NE_XFLM_OK)
			{
				endTest( false, false);
				System.Console.WriteLine( "Should have been no more previous siblings");
				return( false);
			}
			try
			{
				node.deleteNode();
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling deleteNode");
				return( false);
			}
			node = null;

			endTest( false, true);
			
			// Test next/previous document.

			beginTest( "Next/Previous Document");

			// Create a 2nd document.

			try
			{
				node = db.createDocument( (uint)PredefinedXFlaimCollections.XFLM_DATA_COLLECTION);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "creating document");
				return( false);
			}

			// Make sure 1st document has a next document.

			try
			{
				node2 = doc.getNextDocument( node2);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling getNextDocument");
				return( false);
			}

			// Make sure 2nd document does not have a next document.

			rc = RCODE.NE_XFLM_OK;
			try
			{
				node2 = node.getNextDocument( node2);
			}
			catch (XFlaimException ex)
			{
				rc = ex.getRCode();
				if (rc != RCODE.NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					endTest( false, ex, "calling getNextDocument");
					return( false);
				}
			}
			if (rc == RCODE.NE_XFLM_OK)
			{
				endTest( false, false);
				System.Console.WriteLine( "Document should NOT have a next document");
				return( false);
			}

			// Make sure 2nd document has a previous document

			try
			{
				node2 = node.getPreviousDocument( node2);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling getPreviousDocument");
				return( false);
			}

			// Make sure 1st document does not have a previous document.

			rc = RCODE.NE_XFLM_OK;
			try
			{
				node2 = doc.getPreviousDocument( node2);
			}
			catch (XFlaimException ex)
			{
				rc = ex.getRCode();
				if (rc != RCODE.NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					endTest( false, ex, "calling getPreviousDocument");
					return( false);
				}
			}
			if (rc == RCODE.NE_XFLM_OK)
			{
				endTest( false, false);
				System.Console.WriteLine( "Document should NOT have a previous document");
				return( false);
			}
			endTest( false, true);
			
			return( true);
		}

		public bool domNodesTest(
			string	sDbName,
			DbSystem	dbSystem)
		{
			bool	bOk = false;
			Db		db = null;
			bool	bStartedTrans = false;
			RCODE	rc;

			// Create the database

			beginTest( "Create database \"" + sDbName + "\"");

			for (;;)
			{
				rc = RCODE.NE_XFLM_OK;
				try
				{
					XFLM_CREATE_OPTS	createOpts = new XFLM_CREATE_OPTS();

					createOpts.uiBlockSize = 8192;
					createOpts.uiVersionNum = (uint)DBVersions.XFLM_CURRENT_VERSION_NUM;
					createOpts.uiMinRflFileSize = 2000000;
					createOpts.uiMaxRflFileSize = 20000000;
					createOpts.bKeepRflFiles = 1;
					createOpts.bLogAbortedTransToRfl = 1;
					createOpts.eDefaultLanguage = Languages.FLM_DE_LANG;
					db = dbSystem.dbCreate( sDbName, null, null, null, null, createOpts);
				}
				catch (XFlaimException ex)
				{
					rc = ex.getRCode();

					if (rc != RCODE.NE_XFLM_FILE_EXISTS)
					{
						endTest( false, ex, "creating database");
						return( false);
					}
				}
				if (rc == RCODE.NE_XFLM_OK)
				{
					break;
				}

				// rc better be NE_XFLM_FILE_EXISTS - try to delete the file

				try
				{
					dbSystem.dbRemove( sDbName, null, null, true);
				}
				catch (XFlaimException ex)
				{
					endTest( false, ex, "removing database");
					return( false);
				}
			}
			endTest( false, true);

			// Start a transaction

			beginTest( "Start Update Transaction Test");
			try
			{
				db.transBegin( eDbTransType.XFLM_UPDATE_TRANS, 255, 0);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "starting update transaction");
				goto Exit;
			}
			endTest( false, true);
			bStartedTrans = true;

			// Create a document

			if (!createDocumentTest( db))
			{
				goto Exit;
			}

			// Commit the transaction

			beginTest( "Commit Update Transaction Test");
			try
			{
				bStartedTrans = false;
				db.transCommit();
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "committing update transaction");
				goto Exit;
			}
			endTest( false, true);

			bOk = true;

		Exit:
			if (bStartedTrans)
			{
				db.transAbort();
			}
			if (db != null)
			{
				db.close();
				db = null;
			}
			return( bOk);
		}
	}
}
