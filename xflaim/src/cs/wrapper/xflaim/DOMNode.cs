//------------------------------------------------------------------------------
// Desc:	DOM Node
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
using System.Runtime.InteropServices;

namespace xflaim
{

	// IMPORTANT NOTE: These should be kept in sync with data types defined
	// in xflaim.h
	/// <summary>
	/// Data types supported in an XFLAIM database.
	/// </summary>
	public enum FlmDataType : uint
	{
		/// <summary>No data may be stored with a node of this type</summary>
		XFLM_NODATA_TYPE			= 0,
		/// <summary>String data - UTF8 - unicode 16 supported</summary>
		XFLM_TEXT_TYPE				= 1,
		/// <summary>Integer numbers - 64 bit signed and unsigned supported</summary>
		XFLM_NUMBER_TYPE			= 2,
		/// <summary>Binary data</summary>
		XFLM_BINARY_TYPE			= 3
	}

	// IMPORTANT NOTE: These need to be kept in sync with the corresponding
	// definitions in xflaim.h
	/// <summary>
	/// DOM Node types
	/// </summary>
	public enum eDomNodeType : uint
	{
		/// <summary>Invalid Node</summary>
		INVALID_NODE =							0x00,
		/// <summary>Document Node</summary>
		DOCUMENT_NODE =						0x01,
		/// <summary>Element Node</summary>
		ELEMENT_NODE =							0x02,
		/// <summary>Data Node</summary>
		DATA_NODE =								0x03,
		/// <summary>Comment Node</summary>
		COMMENT_NODE =							0x04,
		/// <summary>CDATA Section Node</summary>
		CDATA_SECTION_NODE =					0x05,
		/// <summary>Annotation Node</summary>
		ANNOTATION_NODE =						0x06,
		/// <summary>Processing Instruction Node</summary>
		PROCESSING_INSTRUCTION_NODE =		0x07,
		/// <summary>Attribute Node</summary>
		ATTRIBUTE_NODE =						0x08,
		/// <summary>Any Node Type</summary>
		ANY_NODE_TYPE =						0xFFFF
	}

	// IMPORTANT NOTE: These need to be kept in sync with the corresponding
	// definitions in xflaim.h
	/// <summary>
	/// Node insert locations - relative to another node.
	/// </summary>
	public enum eNodeInsertLoc : uint
	{
		/// <summary>Insert node as root node of document</summary>
		XFLM_ROOT = 0,
		/// <summary>Insert node as first child of reference node</summary>
		XFLM_FIRST_CHILD,
		/// <summary>Insert node as last child of reference node</summary>
		XFLM_LAST_CHILD,
		/// <summary>Insert node as previous sibling of reference node</summary>
		XFLM_PREV_SIB,
		/// <summary>Insert node as next sibling of reference node</summary>
		XFLM_NEXT_SIB,
		/// <summary>Insert node as attribute of reference node</summary>
		XFLM_ATTRIBUTE
	}

	/// <summary>
	/// The DOMNode class provides a number of methods that allow C#
	/// applications to access DOM nodes in XML documents.
	/// </summary>
	public class DOMNode
	{
		private IntPtr 	m_pNode;			// Pointer to IF_DOMNode object in unmanaged space
		private Db			m_db;

		/// <summary>
		/// DOMNode constructor.
		/// </summary>
		/// <param name="pNode">
		/// Reference to an IF_DOMNode object.
		/// </param>
		/// <param name="db">
		/// Db object that this DOMNode object is associated with.
		/// </param>
		internal DOMNode(
			IntPtr	pNode,
			Db			db)
		{
			if (pNode == IntPtr.Zero)
			{
				throw new XFlaimException( "Invalid IF_DOMNode reference");
			}
			
			m_pNode = pNode;

			if (db == null)
			{
				throw new XFlaimException( "Invalid Db reference");
			}
			
			m_db = db;
			
			// Must call something inside of Db.  Otherwise, the
			// m_db object gets a compiler warning on linux because
			// it is not used anywhere.  Other than that, there is really
			// no need to make the following call.
			if (m_db.getDb() == IntPtr.Zero)
			{
				throw new XFlaimException( "Invalid Db.IF_Db object");
			}
		}

		/// <summary>
		/// Set the IF_DOMNode pointer inside this object.  NOTE: We deliberately
		/// do NOT release the m_pNode in this case, because it will already have
		/// been released by the caller.  Usually, the caller has made a call into
		/// the native C++ code that will have released this pointer if it was
		/// successful.
		/// </summary>
		/// <param name="pNode">
		/// Reference to an IF_DOMNode object.
		/// </param>
		/// <param name="db">
		/// Db object that this DOMNode object is associated with.
		/// </param>
		internal void setNodePtr(
			IntPtr	pNode,
			Db			db)
		{
			m_pNode = pNode;
			m_db = db;
		}

		/// <summary>
		/// Destructor.
		/// </summary>
		~DOMNode()
		{
			close();
		}

		/// <summary>
		/// Return the pointer to the IF_DOMNode object.
		/// </summary>
		/// <returns>Returns a pointer to the IF_DOMNode object.</returns>
		internal IntPtr getNode()
		{
			return( m_pNode);
		}

		/// <summary>
		/// Close this DOM node.
		/// </summary>
		public void close()
		{
			// Release the native pNode!
		
			if (m_pNode != IntPtr.Zero)
			{
				xflaim_DOMNode_Release( m_pNode);
				m_pNode = IntPtr.Zero;
			}
		
			// Remove our reference to the db so it can be released.
		
			m_db = null;
		}

		[DllImport("xflaim")]
		private static extern void xflaim_DOMNode_Release(
			IntPtr	pNode);

		private DOMNode makeNode(
			DOMNode	nodeToReuse,
			IntPtr	pNode)
		{
			if (nodeToReuse == null)
			{
				return( new DOMNode( pNode, m_db));
			}
			else
			{
				nodeToReuse.setNodePtr( pNode, m_db);
				return( nodeToReuse);
			}
		}

//-----------------------------------------------------------------------------
// createNode
//-----------------------------------------------------------------------------

		/// <summary>
		/// Creates a new DOM node and inserts it into the database in the
		/// specified position relative to the current node.  An existing
		/// DOMNode object can optionally be passed in, and it will be reused
		/// instead of a new object being allocated.
		/// </summary>
		/// <param name="eNodeType">
		/// Type of node to create.
		/// </param>
		/// <param name="uiNameId">
		/// The dictionary tag number that represents the node name. This name ID
		/// must exist in the dictionary before it can be used here.  The value
		/// may be one of the predefined ones, or it may be created by calling
		/// <see cref="Db.createElementDef"/>.
		/// </param>
		/// <param name="eInsertLoc">
		/// The relative position to insert the new node with respect to this node.
		/// </param>
		/// <param name="nodeToReuse">
		/// An existing <see cref="DOMNode"/> object can optionally be passed in, and
		/// it will be reused instead of a new object being allocated.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode createNode(
			eDomNodeType	eNodeType,
			uint				uiNameId,
			eNodeInsertLoc	eInsertLoc,
			DOMNode			nodeToReuse)
		{
			RCODE		rc;
			IntPtr	pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;

			if ((rc = xflaim_DOMNode_createNode( m_pNode, m_db.getDb(),
				eNodeType, uiNameId, eInsertLoc, ref pNode)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( makeNode( nodeToReuse, pNode));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_createNode(
			IntPtr			pNode,
			IntPtr			pDb,
			eDomNodeType	eNodeType,
			uint				uiNameId,
			eNodeInsertLoc	eInsertLoc,
			ref IntPtr		ppNode);

//-----------------------------------------------------------------------------
// createChildElement
//-----------------------------------------------------------------------------

		/// <summary>
		/// Creates a new element node and inserts it into the database in the
		/// as either the first or last child of the current node.  An existing
		/// <see cref="DOMNode"/> object can optionally be passed in, and it will be reused
		/// instead of a new object being allocated.
		/// </summary>
		/// <param name="uiChildElementNameId">
		/// The dictionary tag number that represents the node name. This name ID
		/// must exist in the dictionary before it can be used here.  The value
		/// may be one of the predefined ones, or it may be created by calling
		/// <see cref="Db.createElementDef"/>.
		/// </param>
		/// <param name="bFirstChild">
		/// Specifies whether the new element is to be created as a first or last child.
		/// </param>
		/// <param name="nodeToReuse">
		/// An existing <see cref="DOMNode"/> object can optionally be passed in, and
		/// it will be reused instead of a new object being allocated.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode createChildElement(
			uint			uiChildElementNameId,
			bool			bFirstChild,
			DOMNode		nodeToReuse)
		{
			RCODE		rc;
			IntPtr	pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;

			if ((rc = xflaim_DOMNode_createChildElement( m_pNode, m_db.getDb(),
				uiChildElementNameId, (int)(bFirstChild ? 1 : 0), ref pNode)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( makeNode( nodeToReuse, pNode));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_createChildElement(
			IntPtr			pNode,
			IntPtr			pDb,
			uint				uiChildElementNameId,
			int				bFirstChild,
			ref IntPtr		ppNode);

//-----------------------------------------------------------------------------
// deleteNode
//-----------------------------------------------------------------------------

		/// <summary>
		/// Removes this node as well as all of it's descendants from the database.
		/// </summary>
		public void deleteNode()
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_deleteNode( m_pNode, m_db.getDb())) != 0)
			{
				throw new XFlaimException( rc);
			}
			xflaim_DOMNode_Release( m_pNode);
			m_pNode = IntPtr.Zero;
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_deleteNode(
			IntPtr			pNode,
			IntPtr			pDb);

//-----------------------------------------------------------------------------
// deleteChildren
//-----------------------------------------------------------------------------

		/// <summary>
		/// Removes the children of this node from the database.
		/// </summary>
		public void deleteChildren()
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_deleteChildren( m_pNode, m_db.getDb())) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_deleteChildren(
			IntPtr			pNode,
			IntPtr			pDb);

//-----------------------------------------------------------------------------
// getNodeType
//-----------------------------------------------------------------------------

		/// <summary>
		/// Returns the type of node. 
		/// </summary>
		/// <returns>Type of node.</returns>
		public eDomNodeType getNodeType()
		{
			return( xflaim_DOMNode_getNodeType( m_pNode));
		}

		[DllImport("xflaim")]
		private static extern eDomNodeType xflaim_DOMNode_getNodeType(
			IntPtr	pNode);

//-----------------------------------------------------------------------------
// isDataLocalToNode
//-----------------------------------------------------------------------------

		/// <summary>
		/// Determine if data for the current node is associated with the node, or
		/// with a child node.  Element nodes may not have data associated with them,
		/// but with child data nodes instead.
		/// </summary>
		/// <returns>
		/// Returns true if this node's data is associated with it, false otherwise.
		/// </returns>
		public bool isDataLocalToNode()
		{
			RCODE	rc;
			int	bLocal;

			if ((rc = xflaim_DOMNode_isDataLocalToNode( m_pNode, m_db.getDb(), out bLocal)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( bLocal != 0 ? true : false);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_isDataLocalToNode(
			IntPtr	pNode,
			IntPtr	pDb,
			out int	pbLocal);

//-----------------------------------------------------------------------------
// createAttribute
//-----------------------------------------------------------------------------

		/// <summary>
		/// Creates a new attribute node for this node.  Note that only element
		/// nodes are allowed to have attributes.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// The dictionary tag number that represents the attribute name. This name ID
		/// must exist in the dictionary before it can be used here.  The value
		/// may be one of the predefined ones, or it may be created by calling
		/// <see cref="Db.createAttributeDef"/>.
		/// </param>
		/// <param name="nodeToReuse">
		/// An existing <see cref="DOMNode"/> object can optionally be passed in, and
		/// it will be reused instead of a new object being allocated.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode createAttribute(
			uint			uiAttrNameId,
			DOMNode		nodeToReuse)
		{
			RCODE		rc;
			IntPtr	pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;

			if ((rc = xflaim_DOMNode_createAttribute( m_pNode, m_db.getDb(),
											uiAttrNameId, ref pNode)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( makeNode( nodeToReuse, pNode));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_createAttribute(
			IntPtr		pNode,
			IntPtr		pDb,
			uint			uiAttrNameId,
			ref IntPtr	ppNode);

//-----------------------------------------------------------------------------
// getFirstAttribute
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves the first attribute node associated with the current node.
		/// </summary>
		/// <param name="nodeToReuse">
		/// An existing <see cref="DOMNode"/> object can optionally be passed in, and
		/// it will be reused instead of a new object being allocated.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode getFirstAttribute(
			DOMNode		nodeToReuse)
		{
			RCODE		rc;
			IntPtr	pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;

			if ((rc = xflaim_DOMNode_getFirstAttribute( m_pNode, m_db.getDb(),
				ref pNode)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( makeNode( nodeToReuse, pNode));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getFirstAttribute(
			IntPtr		pNode,
			IntPtr		pDb,
			ref IntPtr	ppNode);

//-----------------------------------------------------------------------------
// getLastAttribute
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves the last attribute node associated with the current node.
		/// </summary>
		/// <param name="nodeToReuse">
		/// An existing <see cref="DOMNode"/> object can optionally be passed in, and
		/// it will be reused instead of a new object being allocated.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode getLastAttribute(
			DOMNode		nodeToReuse)
		{
			RCODE		rc;
			IntPtr	pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;

			if ((rc = xflaim_DOMNode_getLastAttribute( m_pNode, m_db.getDb(),
				ref pNode)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( makeNode( nodeToReuse, pNode));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getLastAttribute(
			IntPtr		pNode,
			IntPtr		pDb,
			ref IntPtr	ppNode);

//-----------------------------------------------------------------------------
// getAttribute
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves the requested attribute node associated with this node.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// The dictionary tag number of the requested attribute.  The name id must
		/// exist in the dictionary before it can be used here.  The name id may be
		/// one of the predefined ones, or it may be created with
		/// <see cref="Db.createAttributeDef"/>.
		/// </param>
		/// <param name="nodeToReuse">
		/// An existing <see cref="DOMNode"/> object can optionally be passed in, and
		/// it will be reused instead of a new object being allocated.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode getAttribute(
			uint			uiAttrNameId,
			DOMNode		nodeToReuse)
		{
			RCODE		rc;
			IntPtr	pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;

			if ((rc = xflaim_DOMNode_getAttribute( m_pNode, m_db.getDb(),
				uiAttrNameId, ref pNode)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( makeNode( nodeToReuse, pNode));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getAttribute(
			IntPtr		pNode,
			IntPtr		pDb,
			uint			uiAttrNameId,
			ref IntPtr	ppNode);

//-----------------------------------------------------------------------------
// deleteAttribute
//-----------------------------------------------------------------------------

		/// <summary>
		/// Deletes the specified attribute node associated with this node.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// The dictionary tag number of the attribute to delete.  The name id must
		/// exist in the dictionary before it can be used here.  The name id may be
		/// one of the predefined ones, or it may be created with
		/// <see cref="Db.createAttributeDef"/>.
		/// </param>
		public void deleteAttribute(
			uint	uiAttrNameId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_deleteAttribute( m_pNode, m_db.getDb(),
				uiAttrNameId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_deleteAttribute(
			IntPtr		pNode,
			IntPtr		pDb,
			uint			uiAttrNameId);

//-----------------------------------------------------------------------------
// hasAttribute
//-----------------------------------------------------------------------------

		/// <summary>
		/// Determines if the specified attribute exists for this node.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// The dictionary tag number of the attribute to check.  The name id must
		/// exist in the dictionary before it can be used here.  The name id may be
		/// one of the predefined ones, or it may be created with
		/// <see cref="Db.createAttributeDef"/>.
		/// </param>
		/// <returns>
		/// Returns true if the attribute exists, false otherwise.
		/// </returns>
		public bool hasAttribute(
			uint	uiAttrNameId)
		{
			RCODE		rc;
			int		bHasAttr;

			if ((rc = xflaim_DOMNode_hasAttribute( m_pNode, m_db.getDb(),
				uiAttrNameId, out bHasAttr)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( bHasAttr != 0 ? true : false);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_hasAttribute(
			IntPtr		pNode,
			IntPtr		pDb,
			uint			uiAttrNameId,
			out int		pbHasAttr);

//-----------------------------------------------------------------------------
// hasAttributes
//-----------------------------------------------------------------------------

		/// <summary>
		/// Determines if the node has any attributes.
		/// </summary>
		/// <returns>
		/// Returns true if the node has attributes, false otherwise.
		/// </returns>
		public bool hasAttributes()
		{
			RCODE		rc;
			int		bHasAttrs;

			if ((rc = xflaim_DOMNode_hasAttributes( m_pNode, m_db.getDb(),
				out bHasAttrs)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( bHasAttrs != 0 ? true : false);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_hasAttributes(
			IntPtr		pNode,
			IntPtr		pDb,
			out int		pbHasAttrs);

//-----------------------------------------------------------------------------
// hasNextSibling
//-----------------------------------------------------------------------------

		/// <summary>
		/// Determines if the node has a next sibling.
		/// </summary>
		/// <returns>
		/// Returns true if the node has a next sibling false otherwise.
		/// </returns>
		public bool hasNextSibling()
		{
			RCODE		rc;
			int		bHasNextSibling;

			if ((rc = xflaim_DOMNode_hasNextSibling( m_pNode, m_db.getDb(),
				out bHasNextSibling)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( bHasNextSibling != 0 ? true : false);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_hasNextSibling(
			IntPtr		pNode,
			IntPtr		pDb,
			out int		pbHasNextSibling);

//-----------------------------------------------------------------------------
// hasPreviousSibling
//-----------------------------------------------------------------------------

		/// <summary>
		/// Determines if the node has a previous sibling.
		/// </summary>
		/// <returns>
		/// Returns true if the node has a previous sibling false otherwise.
		/// </returns>
		public bool hasPreviousSibling()
		{
			RCODE		rc;
			int		bHasPreviousSibling;

			if ((rc = xflaim_DOMNode_hasPreviousSibling( m_pNode, m_db.getDb(),
				out bHasPreviousSibling)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( bHasPreviousSibling != 0 ? true : false);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_hasPreviousSibling(
			IntPtr		pNode,
			IntPtr		pDb,
			out int		pbHasPreviousSibling);

//-----------------------------------------------------------------------------
// hasChildren
//-----------------------------------------------------------------------------

		/// <summary>
		/// Determines if the node has child nodes.
		/// </summary>
		/// <returns>
		/// Returns true if the node has child nodes, false otherwise.
		/// </returns>
		public bool hasChildren()
		{
			RCODE		rc;
			int		bHasChildren;

			if ((rc = xflaim_DOMNode_hasChildren( m_pNode, m_db.getDb(),
				out bHasChildren)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( bHasChildren != 0 ? true : false);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_hasChildren(
			IntPtr		pNode,
			IntPtr		pDb,
			out int		pbHasChildren);

//-----------------------------------------------------------------------------
// isNamespaceDecl
//-----------------------------------------------------------------------------

		/// <summary>
		/// Determines if the node is a namespace declaration.
		/// </summary>
		/// <returns>
		/// Returns true if the node is a namespace declaration, false otherwise.
		/// </returns>
		public bool isNamespaceDecl()
		{
			RCODE		rc;
			int		bIsNamespaceDecl;

			if ((rc = xflaim_DOMNode_isNamespaceDecl( m_pNode, m_db.getDb(),
				out bIsNamespaceDecl)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( bIsNamespaceDecl != 0 ? true : false);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_isNamespaceDecl(
			IntPtr		pNode,
			IntPtr		pDb,
			out int		pbIsNamespaceDecl);

//-----------------------------------------------------------------------------
// getParentId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Gets the parent node ID for this node.
		/// </summary>
		/// <returns>
		/// Returns the parent node ID of this node.
		/// </returns>
		public ulong getParentId()
		{
			RCODE		rc;
			ulong		ulParentId;

			if ((rc = xflaim_DOMNode_getParentId( m_pNode, m_db.getDb(),
				out ulParentId)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( ulParentId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getParentId(
			IntPtr		pNode,
			IntPtr		pDb,
			out ulong	pulParentId);

//-----------------------------------------------------------------------------
// getNodeId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Gets the node ID for this node.
		/// </summary>
		/// <returns>
		/// Returns the node ID of this node.
		/// </returns>
		public ulong getNodeId()
		{
			RCODE		rc;
			ulong		ulNodeId;

			if ((rc = xflaim_DOMNode_getNodeId( m_pNode, m_db.getDb(),
				out ulNodeId)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( ulNodeId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getNodeId(
			IntPtr		pNode,
			IntPtr		pDb,
			out ulong	pulNodeId);

//-----------------------------------------------------------------------------
// getDocumentId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Gets the document ID for this node.
		/// </summary>
		/// <returns>
		/// Returns the document ID of this node.
		/// </returns>
		public ulong getDocumentId()
		{
			RCODE		rc;
			ulong		ulDocumentId;

			if ((rc = xflaim_DOMNode_getDocumentId( m_pNode, m_db.getDb(),
				out ulDocumentId)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( ulDocumentId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getDocumentId(
			IntPtr		pNode,
			IntPtr		pDb,
			out ulong	pulDocumentId);

//-----------------------------------------------------------------------------
// getPrevSibId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Gets the node ID of the previous sibling for this node.
		/// </summary>
		/// <returns>
		/// Returns the node ID of the previous sibling for this node.
		/// </returns>
		public ulong getPrevSibId()
		{
			RCODE		rc;
			ulong		ulPrevSibId;

			if ((rc = xflaim_DOMNode_getPrevSibId( m_pNode, m_db.getDb(),
				out ulPrevSibId)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( ulPrevSibId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getPrevSibId(
			IntPtr		pNode,
			IntPtr		pDb,
			out ulong	pulPrevSibId);

//-----------------------------------------------------------------------------
// getNextSibId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Gets the node ID of the next sibling for this node.
		/// </summary>
		/// <returns>
		/// Returns the node ID of the next sibling for this node.
		/// </returns>
		public ulong getNextSibId()
		{
			RCODE		rc;
			ulong		ulNextSibId;

			if ((rc = xflaim_DOMNode_getNextSibId( m_pNode, m_db.getDb(),
				out ulNextSibId)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( ulNextSibId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getNextSibId(
			IntPtr		pNode,
			IntPtr		pDb,
			out ulong	pulNextSibId);

//-----------------------------------------------------------------------------
// getFirstChildId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Gets the node ID of the first child for this node.
		/// </summary>
		/// <returns>
		/// Returns the node ID of the first child for this node.
		/// </returns>
		public ulong getFirstChildId()
		{
			RCODE		rc;
			ulong		ulFirstChildId;

			if ((rc = xflaim_DOMNode_getFirstChildId( m_pNode, m_db.getDb(),
				out ulFirstChildId)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( ulFirstChildId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getFirstChildId(
			IntPtr		pNode,
			IntPtr		pDb,
			out ulong	pulFirstChildId);

//-----------------------------------------------------------------------------
// getLastChildId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Gets the node ID of the last child for this node.
		/// </summary>
		/// <returns>
		/// Returns the node ID of the last child for this node.
		/// </returns>
		public ulong getLastChildId()
		{
			RCODE		rc;
			ulong		ulLastChildId;

			if ((rc = xflaim_DOMNode_getLastChildId( m_pNode, m_db.getDb(),
				out ulLastChildId)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( ulLastChildId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getLastChildId(
			IntPtr		pNode,
			IntPtr		pDb,
			out ulong	pulLastChildId);

//-----------------------------------------------------------------------------
// getNameId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Gets the name ID of this node.
		/// </summary>
		/// <returns>
		/// Returns the name ID of this node.
		/// </returns>
		public uint getNameId()
		{
			RCODE		rc;
			uint		uiNameId;

			if ((rc = xflaim_DOMNode_getNameId( m_pNode, m_db.getDb(),
				out uiNameId)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( uiNameId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getNameId(
			IntPtr		pNode,
			IntPtr		pDb,
			out uint		puiNameId);

//-----------------------------------------------------------------------------
// setULong
//-----------------------------------------------------------------------------

		/// <summary>
		/// Sets the value of the node to an unsigned long integer.
		/// </summary>
		/// <param name="ulValue">
		/// Value to set into the node.
		/// </param>
		public void setULong(
			ulong		ulValue)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setULong( m_pNode, m_db.getDb(),
				ulValue, 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value of the node to an unsigned long integer.
		/// </summary>
		/// <param name="ulValue">
		/// Value to set into the node.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setULong(
			ulong		ulValue,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setULong( m_pNode, m_db.getDb(),
				ulValue, uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_setULong(
			IntPtr		pNode,
			IntPtr		pDb,
			ulong			ulValue,
			uint			uiEncId);

//-----------------------------------------------------------------------------
// setAttributeValueULong
//-----------------------------------------------------------------------------

		/// <summary>
		/// Sets the value for the specified attribute of the node to an unsigned long integer.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of attribute whose value is to be set.
		/// </param>
		/// <param name="ulValue">
		/// Value to set into the attribute.
		/// </param>
		public void setAttributeValueULong(
			uint		uiAttrNameId,
			ulong		ulValue)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setAttributeValueULong( m_pNode, m_db.getDb(),
				uiAttrNameId, ulValue, 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value for the specified attribute of the node to an unsigned long integer.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of attribute whose value is to be set.
		/// </param>
		/// <param name="ulValue">
		/// Value to set into the attribute.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setAttributeValueULong(
			uint		uiAttrNameId,
			ulong		ulValue,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setAttributeValueULong( m_pNode, m_db.getDb(),
				uiAttrNameId, ulValue, uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_setAttributeValueULong(
			IntPtr		pNode,
			IntPtr		pDb,
			uint			uiAttrNameId,
			ulong			ulValue,
			uint			uiEncId);

//-----------------------------------------------------------------------------
// setLong
//-----------------------------------------------------------------------------

		/// <summary>
		/// Sets the value of the node to a signed long integer.
		/// </summary>
		/// <param name="lValue">
		/// Value to set into the node.
		/// </param>
		public void setLong(
			long		lValue)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setLong( m_pNode, m_db.getDb(),
				lValue, 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value of the node to a signed long integer.
		/// </summary>
		/// <param name="lValue">
		/// Value to set into the node.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setLong(
			long		lValue,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setLong( m_pNode, m_db.getDb(),
				lValue, uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_setLong(
			IntPtr		pNode,
			IntPtr		pDb,
			long			lValue,
			uint			uiEncId);

//-----------------------------------------------------------------------------
// setAttributeValueLong
//-----------------------------------------------------------------------------

		/// <summary>
		/// Sets the value for the specified attribute of the node to a signed long integer.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of attribute whose value is to be set.
		/// </param>
		/// <param name="lValue">
		/// Value to set into the attribute.
		/// </param>
		public void setAttributeValueLong(
			uint		uiAttrNameId,
			long		lValue)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setAttributeValueLong( m_pNode, m_db.getDb(),
				uiAttrNameId, lValue, 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value for the specified attribute of the node to a signed long integer.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of attribute whose value is to be set.
		/// </param>
		/// <param name="lValue">
		/// Value to set into the attribute.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setAttributeValueLong(
			uint		uiAttrNameId,
			long		lValue,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setAttributeValueLong( m_pNode, m_db.getDb(),
				uiAttrNameId, lValue, uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_setAttributeValueLong(
			IntPtr		pNode,
			IntPtr		pDb,
			uint			uiAttrNameId,
			long			lValue,
			uint			uiEncId);

//-----------------------------------------------------------------------------
// setUInt
//-----------------------------------------------------------------------------

		/// <summary>
		/// Sets the value of the node to an unsigned integer.
		/// </summary>
		/// <param name="uiValue">
		/// Value to set into the node.
		/// </param>
		public void setUInt(
			uint		uiValue)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setUInt( m_pNode, m_db.getDb(),
				uiValue, 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value of the node to an unsigned integer.
		/// </summary>
		/// <param name="uiValue">
		/// Value to set into the node.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setUInt(
			uint		uiValue,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setUInt( m_pNode, m_db.getDb(),
				uiValue, uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_setUInt(
			IntPtr		pNode,
			IntPtr		pDb,
			uint			uiValue,
			uint			uiEncId);

//-----------------------------------------------------------------------------
// setAttributeValueUInt
//-----------------------------------------------------------------------------

		/// <summary>
		/// Sets the value for the specified attribute of the node to an unsigned integer.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of attribute whose value is to be set.
		/// </param>
		/// <param name="uiValue">
		/// Value to set into the attribute.
		/// </param>
		public void setAttributeValueUInt(
			uint		uiAttrNameId,
			uint		uiValue)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setAttributeValueUInt( m_pNode, m_db.getDb(),
				uiAttrNameId, uiValue, 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value for the specified attribute of the node to an unsigned integer.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of attribute whose value is to be set.
		/// </param>
		/// <param name="uiValue">
		/// Value to set into the attribute.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setAttributeValueUInt(
			uint		uiAttrNameId,
			uint		uiValue,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setAttributeValueUInt( m_pNode, m_db.getDb(),
				uiAttrNameId, uiValue, uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_setAttributeValueUInt(
			IntPtr		pNode,
			IntPtr		pDb,
			uint			uiAttrNameId,
			uint			uiValue,
			uint			uiEncId);

//-----------------------------------------------------------------------------
// setInt
//-----------------------------------------------------------------------------

		/// <summary>
		/// Sets the value of the node to a signed integer.
		/// </summary>
		/// <param name="iValue">
		/// Value to set into the node.
		/// </param>
		public void setInt(
			int		iValue)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setInt( m_pNode, m_db.getDb(),
				iValue, 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value of the node to a signed integer.
		/// </summary>
		/// <param name="iValue">
		/// Value to set into the node.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setInt(
			int		iValue,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setInt( m_pNode, m_db.getDb(),
				iValue, uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_setInt(
			IntPtr		pNode,
			IntPtr		pDb,
			int			iValue,
			uint			uiEncId);

//-----------------------------------------------------------------------------
// setAttributeValueInt
//-----------------------------------------------------------------------------

		/// <summary>
		/// Sets the value for the specified attribute of the node to a signed integer.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of attribute whose value is to be set.
		/// </param>
		/// <param name="iValue">
		/// Value to set into the attribute.
		/// </param>
		public void setAttributeValueInt(
			uint		uiAttrNameId,
			int		iValue)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setAttributeValueInt( m_pNode, m_db.getDb(),
				uiAttrNameId, iValue, 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value for the specified attribute of the node to a signed integer.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of attribute whose value is to be set.
		/// </param>
		/// <param name="iValue">
		/// Value to set into the attribute.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setAttributeValueInt(
			uint		uiAttrNameId,
			int		iValue,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setAttributeValueInt( m_pNode, m_db.getDb(),
				uiAttrNameId, iValue, uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_setAttributeValueInt(
			IntPtr		pNode,
			IntPtr		pDb,
			uint			uiAttrNameId,
			int			iValue,
			uint			uiEncId);

//-----------------------------------------------------------------------------
// setString
//-----------------------------------------------------------------------------

		/// <summary>
		/// Sets the value for the node to a string.
		/// </summary>
		/// <param name="sValue">
		/// Value to set into the node.
		/// </param>
		public void setString(
			string	sValue)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setString( m_pNode, m_db.getDb(),
				sValue, 1, 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value for the node to a string.
		/// </summary>
		/// <param name="sValue">
		/// Value to set into the node.
		/// </param>
		/// <param name="bLast">
		/// Specifies whether sValue is the last text to be appended to this
		/// node.  If false, then another call to setString is expected, and
		/// the new text will be appended to the text currently stored in this
		/// node.  If true, then no more text is expected and  another call to
		/// setString will overwrite the what is currently stored in this node.
		/// </param>
		public void setString(
			string	sValue,
			bool		bLast)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setString( m_pNode, m_db.getDb(),
				sValue, (int)(bLast ? 1 : 0), 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value for the node to a string.
		/// </summary>
		/// <param name="sValue">
		/// Value to set into the node.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setString(
			string	sValue,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setString( m_pNode, m_db.getDb(),
				sValue, 1, uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value for the node to a string.
		/// </summary>
		/// <param name="sValue">
		/// Value to set into the node.
		/// </param>
		/// <param name="bLast">
		/// Specifies whether sValue is the last text to be appended to this
		/// node.  If false, then another call to setString is expected, and
		/// the new text will be appended to the text currently stored in this
		/// node.  If true, then no more text is expected and  another call to
		/// setString will overwrite the what is currently stored in this node.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setString(
			string	sValue,
			bool		bLast,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setString( m_pNode, m_db.getDb(),
				sValue, (int)(bLast ? 1 : 0), uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_setString(
			IntPtr		pNode,
			IntPtr		pDb,
			[MarshalAs(UnmanagedType.LPWStr), In]
			string		sValue,
			int			bLast,
			uint			uiEncId);

//-----------------------------------------------------------------------------
// setAttributeValueString
//-----------------------------------------------------------------------------

		/// <summary>
		/// Sets the value for the specified attribute of the node to a string
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of attribute whose value is to be set.
		/// </param>
		/// <param name="sValue">
		/// Value to set into the attribute.
		/// </param>
		public void setAttributeValueString(
			uint		uiAttrNameId,
			string	sValue)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setAttributeValueString( m_pNode, m_db.getDb(),
				uiAttrNameId, sValue, 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value for the specified attribute of the node to a string.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of attribute whose value is to be set.
		/// </param>
		/// <param name="sValue">
		/// Value to set into the attribute.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setAttributeValueString(
			uint		uiAttrNameId,
			string	sValue,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setAttributeValueString( m_pNode, m_db.getDb(),
				uiAttrNameId, sValue, uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_setAttributeValueString(
			IntPtr		pNode,
			IntPtr		pDb,
			uint			uiAttrNameId,
			[MarshalAs(UnmanagedType.LPWStr), In]
			string		sValue,
			uint			uiEncId);

//-----------------------------------------------------------------------------
// setBinary
//-----------------------------------------------------------------------------

		/// <summary>
		/// Sets the value for the node to a byte array of binary data.
		/// </summary>
		/// <param name="ucValue">
		/// Value to set into the node.
		/// </param>
		public void setBinary(
			byte []	ucValue)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setBinary( m_pNode, m_db.getDb(),
				ucValue, (uint)ucValue.Length, 1, 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value for the node to a byte array of binary data.
		/// </summary>
		/// <param name="ucValue">
		/// Value to set into the node.
		/// </param>
		/// <param name="bLast">
		/// Specifies whether ucValue is the last text to be appended to this
		/// node.  If false, then another call to setBinary is expected, and
		/// the new text will be appended to the text currently stored in this
		/// node.  If true, then no more text is expected and  another call to
		/// setBinary will overwrite the what is currently stored in this node.
		/// </param>
		public void setBinary(
			 byte []	ucValue,
			bool		bLast)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setBinary( m_pNode, m_db.getDb(),
				ucValue, (uint)ucValue.Length, (int)(bLast ? 1 : 0), 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value for the node to a byte array of binary data.
		/// </summary>
		/// <param name="ucValue">
		/// Value to set into the node.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setBinary(
			byte []	ucValue,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setBinary( m_pNode, m_db.getDb(),
				ucValue, (uint)ucValue.Length, 1, uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value for the node to a byte array of binary data.
		/// </summary>
		/// <param name="ucValue">
		/// Value to set into the node.
		/// </param>
		/// <param name="bLast">
		/// Specifies whether ucValue is the last text to be appended to this
		/// node.  If false, then another call to setBinary is expected, and
		/// the new text will be appended to the text currently stored in this
		/// node.  If true, then no more text is expected and  another call to
		/// setBinary will overwrite the what is currently stored in this node.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setBinary(
			byte []	ucValue,
			bool		bLast,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setBinary( m_pNode, m_db.getDb(),
				ucValue, (uint)ucValue.Length, (int)(bLast ? 1 : 0), uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_setBinary(
			IntPtr		pNode,
			IntPtr		pDb,
			[MarshalAs(UnmanagedType.LPArray), In]
			byte []		ucValue,
			uint			uiLen,
			int			bLast,
			uint			uiEncId);

//-----------------------------------------------------------------------------
// setAttributeValueBinary
//-----------------------------------------------------------------------------

		/// <summary>
		/// Sets the value for the specified attribute of the node to a
		/// byte array of binary data.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of attribute whose value is to be set.
		/// </param>
		/// <param name="ucValue">
		/// Value to set into the attribute.
		/// </param>
		public void setAttributeValueBinary(
			uint		uiAttrNameId,
			byte []	ucValue)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setAttributeValueBinary( m_pNode, m_db.getDb(),
				uiAttrNameId, ucValue, (uint)ucValue.Length, 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value for the specified attribute of the node to a
		/// byte array of binary data.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of attribute whose value is to be set.
		/// </param>
		/// <param name="ucValue">
		/// Value to set into the attribute.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setAttributeValueBinary(
			uint		uiAttrNameId,
			byte []	ucValue,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setAttributeValueBinary( m_pNode, m_db.getDb(),
				uiAttrNameId, ucValue, (uint)ucValue.Length, uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_setAttributeValueBinary(
			IntPtr		pNode,
			IntPtr		pDb,
			uint			uiAttrNameId,
			[MarshalAs(UnmanagedType.LPWStr), In]
			byte []		ucValue,
			uint			uiLen,
			uint			uiEncId);

//-----------------------------------------------------------------------------
// getDataLength
//-----------------------------------------------------------------------------

		/// <summary>
		/// Returns the data length for the value of this node.
		/// </summary>
		/// <returns>Node's value data length.</returns>
		public uint getDataLength()
		{
			RCODE		rc;
			uint		uiDataLength;

			if ((rc = xflaim_DOMNode_getDataLength( m_pNode, m_db.getDb(),
				out uiDataLength)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( uiDataLength);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getDataLength(
			IntPtr		pNode,
			IntPtr		pDb,
			out uint		puiDataLength);

//-----------------------------------------------------------------------------
// getDataType
//-----------------------------------------------------------------------------

		/// <summary>
		/// Returns the data type for the value of this node.
		/// </summary>
		/// <returns>Node's value data type.</returns>
		public FlmDataType getDataType()
		{
			RCODE				rc;
			FlmDataType		eDataType;

			if ((rc = xflaim_DOMNode_getDataType( m_pNode, m_db.getDb(),
				out eDataType)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( eDataType);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getDataType(
			IntPtr				pNode,
			IntPtr				pDb,
			out FlmDataType	peDataType);

//-----------------------------------------------------------------------------
// getULong
//-----------------------------------------------------------------------------

		/// <summary>
		/// Returns the value for this node as an unsigned long integer.
		/// </summary>
		/// <returns>Node's value as an unsigned long integer.</returns>
		public ulong getULong()
		{
			RCODE	rc;
			ulong	ulValue;

			if ((rc = xflaim_DOMNode_getULong( m_pNode, m_db.getDb(),
				out ulValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( ulValue);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getULong(
			IntPtr		pNode,
			IntPtr		pDb,
			out ulong	pulValue);

//-----------------------------------------------------------------------------
// getAttributeValueULong
//-----------------------------------------------------------------------------

		/// <summary>
		/// Returns the value for the specified attribute associated with this
		/// node.  Value is returned as an unsigned long integer.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of the attribute whose value is to be returned.
		/// </param>
		/// <param name="bDefaultOk">
		/// If true, specifies that if the attribute is not found then the
		/// value in ulDefaultToUse is to be returned.  If false, and the
		/// attribute is not found, an exception will be thrown.
		/// </param>
		/// <param name="ulDefaultToUse">
		/// Default value to use if the attribute is not found and bDefaultOk is true.
		/// </param>
		/// <returns>Attribute's value is returned as an unsigned long integer.</returns>
		public ulong getAttributeValueULong(
			uint		uiAttrNameId,
			bool		bDefaultOk,
			ulong		ulDefaultToUse)
		{
			RCODE	rc;
			ulong	ulValue;

			if ((rc = xflaim_DOMNode_getAttributeValueULong( m_pNode, m_db.getDb(),
				uiAttrNameId, (int)(bDefaultOk ? 1 : 0), ulDefaultToUse, out ulValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( ulValue);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getAttributeValueULong(
			IntPtr		pNode,
			IntPtr		pDb,
			uint			uiAttrNameId,
			int			bDefaultOk,
			ulong			ulDefaultToUse,
			out ulong	pulValue);

//-----------------------------------------------------------------------------
// getLong
//-----------------------------------------------------------------------------

		/// <summary>
		/// Returns the value for this node as a signed long integer.
		/// </summary>
		/// <returns>Node's value as a signed long integer.</returns>
		public long getLong()
		{
			RCODE	rc;
			long	lValue;

			if ((rc = xflaim_DOMNode_getLong( m_pNode, m_db.getDb(),
				out lValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( lValue);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getLong(
			IntPtr		pNode,
			IntPtr		pDb,
			out long		plValue);

//-----------------------------------------------------------------------------
// getAttributeValueLong
//-----------------------------------------------------------------------------

		/// <summary>
		/// Returns the value for the specified attribute associated with this
		/// node.  Value is returned as a signed long integer.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of the attribute whose value is to be returned.
		/// </param>
		/// <param name="bDefaultOk">
		/// If true, specifies that if the attribute is not found then the
		/// value in lDefaultToUse is to be returned.  If false, and the
		/// attribute is not found, an exception will be thrown.
		/// </param>
		/// <param name="lDefaultToUse">
		/// Default value to use if the attribute is not found and bDefaultOk is true.
		/// </param>
		/// <returns>Attribute's value is returned as an unsigned long integer.</returns>
		public long getAttributeValueLong(
			uint		uiAttrNameId,
			bool		bDefaultOk,
			long		lDefaultToUse)
		{
			RCODE	rc;
			long	lValue;

			if ((rc = xflaim_DOMNode_getAttributeValueLong( m_pNode, m_db.getDb(),
				uiAttrNameId, (int)(bDefaultOk ? 1 : 0), lDefaultToUse, out lValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( lValue);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getAttributeValueLong(
			IntPtr		pNode,
			IntPtr		pDb,
			uint			uiAttrNameId,
			int			bDefaultOk,
			long			lDefaultToUse,
			out long		plValue);

//-----------------------------------------------------------------------------
// getUInt
//-----------------------------------------------------------------------------

		/// <summary>
		/// Returns the value for this node as an unsigned integer.
		/// </summary>
		/// <returns>Node's value as an unsigned integer.</returns>
		public uint getUInt()
		{
			RCODE	rc;
			uint	uiValue;

			if ((rc = xflaim_DOMNode_getUInt( m_pNode, m_db.getDb(),
				out uiValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( uiValue);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getUInt(
			IntPtr		pNode,
			IntPtr		pDb,
			out uint		puiValue);

//-----------------------------------------------------------------------------
// getAttributeValueUInt
//-----------------------------------------------------------------------------

		/// <summary>
		/// Returns the value for the specified attribute associated with this
		/// node.  Value is returned as an unsigned integer.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of the attribute whose value is to be returned.
		/// </param>
		/// <param name="bDefaultOk">
		/// If true, specifies that if the attribute is not found then the
		/// value in uiDefaultToUse is to be returned.  If false, and the
		/// attribute is not found, an exception will be thrown.
		/// </param>
		/// <param name="uiDefaultToUse">
		/// Default value to use if the attribute is not found and bDefaultOk is true.
		/// </param>
		/// <returns>Attribute's value is returned as an unsigned long integer.</returns>
		public uint getAttributeValueUInt(
			uint		uiAttrNameId,
			bool		bDefaultOk,
			uint		uiDefaultToUse)
		{
			RCODE	rc;
			uint	uiValue;

			if ((rc = xflaim_DOMNode_getAttributeValueUInt( m_pNode, m_db.getDb(),
				uiAttrNameId, (int)(bDefaultOk ? 1 : 0), uiDefaultToUse, out uiValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( uiValue);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getAttributeValueUInt(
			IntPtr		pNode,
			IntPtr		pDb,
			uint			uiAttrNameId,
			int			bDefaultOk,
			uint			uiDefaultToUse,
			out uint		puiValue);

//-----------------------------------------------------------------------------
// getInt
//-----------------------------------------------------------------------------

		/// <summary>
		/// Returns the value for this node as a signed integer.
		/// </summary>
		/// <returns>Node's value as a signed integer.</returns>
		public int getInt()
		{
			RCODE	rc;
			int	iValue;

			if ((rc = xflaim_DOMNode_getInt( m_pNode, m_db.getDb(),
				out iValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( iValue);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getInt(
			IntPtr		pNode,
			IntPtr		pDb,
			out int		piValue);

//-----------------------------------------------------------------------------
// getAttributeValueInt
//-----------------------------------------------------------------------------

		/// <summary>
		/// Returns the value for the specified attribute associated with this
		/// node.  Value is returned as a signed integer.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of the attribute whose value is to be returned.
		/// </param>
		/// <param name="bDefaultOk">
		/// If true, specifies that if the attribute is not found then the
		/// value in iDefaultToUse is to be returned.  If false, and the
		/// attribute is not found, an exception will be thrown.
		/// </param>
		/// <param name="iDefaultToUse">
		/// Default value to use if the attribute is not found and bDefaultOk is true.
		/// </param>
		/// <returns>Attribute's value is returned as an unsigned long integer.</returns>
		public int getAttributeValueInt(
			uint		uiAttrNameId,
			bool		bDefaultOk,
			int		iDefaultToUse)
		{
			RCODE	rc;
			int	iValue;

			if ((rc = xflaim_DOMNode_getAttributeValueInt( m_pNode, m_db.getDb(),
				uiAttrNameId, (int)(bDefaultOk ? 1 : 0), iDefaultToUse, out iValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( iValue);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getAttributeValueInt(
			IntPtr		pNode,
			IntPtr		pDb,
			uint			uiAttrNameId,
			int			bDefaultOk,
			int			iDefaultToUse,
			out int		piValue);

//-----------------------------------------------------------------------------
// getString and getSubString
//-----------------------------------------------------------------------------

		/// <summary>
		/// Returns the value for this node as a string
		/// </summary>
		/// <returns>Node's value as a string.</returns>
		public string getString()
		{
			RCODE		rc;
			string	sValue;
			IntPtr	puzValue;

			if ((rc = xflaim_DOMNode_getString( m_pNode, m_db.getDb(),
				0, 0, out puzValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
			sValue = Marshal.PtrToStringUni( puzValue);
			m_db.getDbSystem().freeUnmanagedMem( puzValue);
			return( sValue);
		}

		/// <summary>
		/// Returns a sub-string of the value in this node.
		/// </summary>
		/// <param name="uiStartPos">
		/// Starting character position in string to retrieve sub-string from.
		/// </param>
		/// <param name="uiNumChars">
		/// Maximum number of characters to retrieve.  May return fewer than
		/// this number of characters if there are not that many characters
		/// available from the specified starting position.
		/// </param>
		/// <returns>Node's sub-string value.</returns>
		public string getSubString(
			uint	uiStartPos,
			uint	uiNumChars)
		{
			RCODE		rc;
			string	sValue;
			IntPtr	puzValue;

			if ((rc = xflaim_DOMNode_getString( m_pNode, m_db.getDb(),
				uiStartPos, uiNumChars, out puzValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
			sValue = Marshal.PtrToStringUni( puzValue);
			m_db.getDbSystem().freeUnmanagedMem( puzValue);
			return( sValue);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getString(
			IntPtr		pNode,
			IntPtr		pDb,
			uint			uiStartPos,
			uint			uiNumChars,
			out IntPtr	ppuzValue);

//-----------------------------------------------------------------------------
// getAttributeValueString
//-----------------------------------------------------------------------------

		/// <summary>
		/// Returns the value for the specified attribute of this node as a string
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of the attribute whose value is to be returned.
		/// </param>
		/// <returns>Attribute's value as a string.</returns>
		public string getAttributeValueString(
			uint	uiAttrNameId)
		{
			RCODE		rc;
			string	sValue;
			IntPtr	puzValue;

			if ((rc = xflaim_DOMNode_getAttributeValueString( m_pNode, m_db.getDb(),
				uiAttrNameId, out puzValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
			sValue = Marshal.PtrToStringUni( puzValue);
			m_db.getDbSystem().freeUnmanagedMem( puzValue);
			return( sValue);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getAttributeValueString(
			IntPtr		pNode,
			IntPtr		pDb,
			uint			uiAttrNameId,
			out IntPtr	ppuzValue);

//-----------------------------------------------------------------------------
// getStringLen
//-----------------------------------------------------------------------------

		/// <summary>
		/// Returns the number of characters this node's string value contains.
		/// </summary>
		/// <returns>Number of characters in node's value string.</returns>
		public uint getStringLen()
		{
			RCODE		rc;
			uint		uiNumChars;

			if ((rc = xflaim_DOMNode_getStringLen( m_pNode, m_db.getDb(),
				out uiNumChars)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( uiNumChars);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getStringLen(
			IntPtr		pNode,
			IntPtr		pDb,
			out uint		puiNumChars);

//-----------------------------------------------------------------------------
// getBinary
//-----------------------------------------------------------------------------

		/// <summary>
		/// Returns the value for this node as a binary byte array
		/// </summary>
		/// <returns>Node's value as a string.</returns>
		public byte [] getBinary()
		{
			RCODE		rc;
			byte []	ucValue;
			uint		uiLen;

			uiLen = getDataLength();
			ucValue = new byte [uiLen];

			if ((rc = xflaim_DOMNode_getBinary( m_pNode, m_db.getDb(),
				0, uiLen, ucValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( ucValue);
		}

		/// <summary>
		/// Returns a sub-part of this node's value as a binary byte array.
		/// </summary>
		/// <param name="uiStartPos">
		/// Starting byte position in node's value to retrieve data from.
		/// </param>
		/// <param name="uiNumBytes">
		/// Maximum number of bytes to retrieve.  May return fewer than
		/// this number of bytes if there are not that many bytes
		/// available from the specified starting position.
		/// </param>
		/// <returns>Node's sub-part value.</returns>
		public byte [] getBinary(
			uint	uiStartPos,
			uint	uiNumBytes)
		{
			RCODE		rc;
			byte []	ucValue;
			uint		uiLen;

			uiLen = getDataLength();
			if (uiStartPos >= uiLen)
			{
				return( null);
			}
			uiLen -= uiStartPos;
			if (uiNumBytes > 0 && uiNumBytes < uiLen)
			{
				uiLen = uiNumBytes;
			}
			ucValue = new byte [uiLen];

			if ((rc = xflaim_DOMNode_getBinary( m_pNode, m_db.getDb(),
				uiStartPos, uiLen, ucValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( ucValue);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getBinary(
			IntPtr		pNode,
			IntPtr		pDb,
			uint			uiStartPos,
			uint			uiNumBytes,
			[MarshalAs(UnmanagedType.LPArray), Out] 
			byte []		pucValue);

//-----------------------------------------------------------------------------
// getAttributeValueDataLength
//-----------------------------------------------------------------------------

		/// <summary>
		/// Returns the data length for the specified attribute of this node.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of the attribute whose value length is to be returned.
		/// </param>
		/// <returns>Attribute's value data length.</returns>
		public uint getAttributeValueDataLength(
			uint	uiAttrNameId)
		{
			RCODE		rc;
			uint		uiDataLength;

			if ((rc = xflaim_DOMNode_getAttributeValueDataLength( m_pNode, m_db.getDb(),
				uiAttrNameId, out uiDataLength)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( uiDataLength);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getAttributeValueDataLength(
			IntPtr		pNode,
			IntPtr		pDb,
			uint			uiAttrNameId,
			out uint		puiDataLength);

//-----------------------------------------------------------------------------
// getAttributeValueBinary
//-----------------------------------------------------------------------------

		/// <summary>
		/// Returns the data for the specified attribute of this node as a byte
		/// array of binary data.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of the attribute whose data is to be returned.
		/// </param>
		/// <returns>Attribute's value.</returns>
		public byte [] getAttributeValueBinary(
			uint	uiAttrNameId)
		{
			RCODE		rc;
			byte []	ucValue;
			uint		uiLen;

			uiLen = getAttributeValueDataLength( uiAttrNameId);
			ucValue = new byte [uiLen];

			if ((rc = xflaim_DOMNode_getAttributeValueBinary( m_pNode, m_db.getDb(),
				uiAttrNameId, uiLen, ucValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( ucValue);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getAttributeValueBinary(
			IntPtr		pNode,
			IntPtr		pDb,
			uint			uiAttrNameId,
			uint			uiLen,
			[MarshalAs(UnmanagedType.LPArray), Out] 
			byte []		pucValue);

//-----------------------------------------------------------------------------
// getDocumentNode
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves the document node of the document this node belongs to.
		/// </summary>
		/// <param name="nodeToReuse">
		/// An existing <see cref="DOMNode"/> object can optionally be passed in, and
		/// it will be reused instead of a new object being allocated.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode getDocumentNode(
			DOMNode	nodeToReuse)
		{
			RCODE		rc;
			IntPtr	pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;

			if ((rc = xflaim_DOMNode_getDocumentNode( m_pNode, m_db.getDb(),
				ref pNode)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( makeNode( nodeToReuse, pNode));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getDocumentNode(
			IntPtr		pNode,
			IntPtr		pDb,
			ref IntPtr	ppNode);

//-----------------------------------------------------------------------------
// getParentNode
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves the parent node of this node.
		/// </summary>
		/// <param name="nodeToReuse">
		/// An existing <see cref="DOMNode"/> object can optionally be passed in, and
		/// it will be reused instead of a new object being allocated.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode getParentNode(
			DOMNode	nodeToReuse)
		{
			RCODE		rc;
			IntPtr	pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;

			if ((rc = xflaim_DOMNode_getParentNode( m_pNode, m_db.getDb(),
				ref pNode)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( makeNode( nodeToReuse, pNode));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getParentNode(
			IntPtr		pNode,
			IntPtr		pDb,
			ref IntPtr	ppNode);

//-----------------------------------------------------------------------------
// getFirstChild
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves the first child node of this node.
		/// </summary>
		/// <param name="nodeToReuse">
		/// An existing <see cref="DOMNode"/> object can optionally be passed in, and
		/// it will be reused instead of a new object being allocated.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode getFirstChild(
			DOMNode	nodeToReuse)
		{
			RCODE		rc;
			IntPtr	pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;

			if ((rc = xflaim_DOMNode_getFirstChild( m_pNode, m_db.getDb(),
				ref pNode)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( makeNode( nodeToReuse, pNode));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getFirstChild(
			IntPtr		pNode,
			IntPtr		pDb,
			ref IntPtr	ppNode);

//-----------------------------------------------------------------------------
// getLastChild
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves the last child node of this node.
		/// </summary>
		/// <param name="nodeToReuse">
		/// An existing <see cref="DOMNode"/> object can optionally be passed in, and
		/// it will be reused instead of a new object being allocated.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode getLastChild(
			DOMNode	nodeToReuse)
		{
			RCODE		rc;
			IntPtr	pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;

			if ((rc = xflaim_DOMNode_getLastChild( m_pNode, m_db.getDb(),
				ref pNode)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( makeNode( nodeToReuse, pNode));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getLastChild(
			IntPtr		pNode,
			IntPtr		pDb,
			ref IntPtr	ppNode);

//-----------------------------------------------------------------------------
// getChild
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves the first instance of the specified type of node from this
		/// node's child nodes.
		/// </summary>
		/// <param name="eNodeType">
		/// Type of node to retrieve.
		/// </param>
		/// <param name="nodeToReuse">
		/// An existing <see cref="DOMNode"/> object can optionally be passed in, and
		/// it will be reused instead of a new object being allocated.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode getChild(
			eDomNodeType	eNodeType,
			DOMNode			nodeToReuse)
		{
			RCODE		rc;
			IntPtr	pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;

			if ((rc = xflaim_DOMNode_getChild( m_pNode, m_db.getDb(),
				eNodeType, ref pNode)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( makeNode( nodeToReuse, pNode));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getChild(
			IntPtr			pNode,
			IntPtr			pDb,
			eDomNodeType	eNodeType,
			ref IntPtr		ppNode);

//-----------------------------------------------------------------------------
// getChildElement
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves the first instance of the specified element node from this
		/// node's child nodes.
		/// </summary>
		/// <param name="uiElementNameId">
		/// The element name ID for the node to be retrieved.
		/// </param>
		/// <param name="nodeToReuse">
		/// An existing <see cref="DOMNode"/> object can optionally be passed in, and
		/// it will be reused instead of a new object being allocated.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode getChildElement(
			uint		uiElementNameId,
			DOMNode	nodeToReuse)
		{
			RCODE		rc;
			IntPtr	pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;

			if ((rc = xflaim_DOMNode_getChildElement( m_pNode, m_db.getDb(),
				uiElementNameId, ref pNode)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( makeNode( nodeToReuse, pNode));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getChildElement(
			IntPtr			pNode,
			IntPtr			pDb,
			uint				uiElementNameId,
			ref IntPtr		ppNode);

//-----------------------------------------------------------------------------
// getSiblingElement
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves the first instance of the specified element node from this
		/// node's sibling nodes.
		/// </summary>
		/// <param name="uiElementNameId">
		/// The element name ID for the node to be retrieved.
		/// </param>
		/// <param name="bNext">
		/// If true, will search siblings that follow this node.
		/// If false, will search siblings that precede this node.
		/// </param>
		/// <param name="nodeToReuse">
		/// An existing <see cref="DOMNode"/> object can optionally be passed in, and
		/// it will be reused instead of a new object being allocated.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode getSiblingElement(
			uint		uiElementNameId,
			bool		bNext,
			DOMNode	nodeToReuse)
		{
			RCODE		rc;
			IntPtr	pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;

			if ((rc = xflaim_DOMNode_getSiblingElement( m_pNode, m_db.getDb(),
				uiElementNameId, (int)(bNext ? 1 : 0), ref pNode)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( makeNode( nodeToReuse, pNode));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getSiblingElement(
			IntPtr			pNode,
			IntPtr			pDb,
			uint				uiElementNameId,
			int				bNext,
			ref IntPtr		ppNode);

//-----------------------------------------------------------------------------
// getAncestorElement
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves the first instance of the specified element node from this
		/// node's ancestor nodes.
		/// </summary>
		/// <param name="uiElementNameId">
		/// The element name ID for the node to be retrieved.
		/// </param>
		/// <param name="nodeToReuse">
		/// An existing <see cref="DOMNode"/> object can optionally be passed in, and
		/// it will be reused instead of a new object being allocated.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode getAncestorElement(
			uint		uiElementNameId,
			DOMNode	nodeToReuse)
		{
			RCODE		rc;
			IntPtr	pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;

			if ((rc = xflaim_DOMNode_getAncestorElement( m_pNode, m_db.getDb(),
				uiElementNameId, ref pNode)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( makeNode( nodeToReuse, pNode));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getAncestorElement(
			IntPtr			pNode,
			IntPtr			pDb,
			uint				uiElementNameId,
			ref IntPtr		ppNode);

//-----------------------------------------------------------------------------
// getDescendantElement
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves the first instance of the specified element node from this
		/// node's descendant nodes.
		/// </summary>
		/// <param name="uiElementNameId">
		/// The element name ID for the node to be retrieved.
		/// </param>
		/// <param name="nodeToReuse">
		/// An existing <see cref="DOMNode"/> object can optionally be passed in, and
		/// it will be reused instead of a new object being allocated.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode getDescendantElement(
			uint		uiElementNameId,
			DOMNode	nodeToReuse)
		{
			RCODE		rc;
			IntPtr	pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;

			if ((rc = xflaim_DOMNode_getDescendantElement( m_pNode, m_db.getDb(),
				uiElementNameId, ref pNode)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( makeNode( nodeToReuse, pNode));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getDescendantElement(
			IntPtr			pNode,
			IntPtr			pDb,
			uint				uiElementNameId,
			ref IntPtr		ppNode);

//-----------------------------------------------------------------------------
// getPreviousSibling
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves this node's previous sibling node.
		/// </summary>
		/// <param name="nodeToReuse">
		/// An existing <see cref="DOMNode"/> object can optionally be passed in, and
		/// it will be reused instead of a new object being allocated.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode getPreviousSibling(
			DOMNode	nodeToReuse)
		{
			RCODE		rc;
			IntPtr	pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;

			if ((rc = xflaim_DOMNode_getPreviousSibling( m_pNode, m_db.getDb(),
				ref pNode)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( makeNode( nodeToReuse, pNode));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getPreviousSibling(
			IntPtr			pNode,
			IntPtr			pDb,
			ref IntPtr		ppNode);

//-----------------------------------------------------------------------------
// getNextSibling
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves this node's next sibling node.
		/// </summary>
		/// <param name="nodeToReuse">
		/// An existing <see cref="DOMNode"/> object can optionally be passed in, and
		/// it will be reused instead of a new object being allocated.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode getNextSibling(
			DOMNode	nodeToReuse)
		{
			RCODE		rc;
			IntPtr	pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;

			if ((rc = xflaim_DOMNode_getNextSibling( m_pNode, m_db.getDb(),
				ref pNode)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( makeNode( nodeToReuse, pNode));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getNextSibling(
			IntPtr			pNode,
			IntPtr			pDb,
			ref IntPtr		ppNode);

//-----------------------------------------------------------------------------
// getPreviousDocument
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves the previous document.  This node must be a root node or
		/// a document node.
		/// </summary>
		/// <param name="nodeToReuse">
		/// An existing <see cref="DOMNode"/> object can optionally be passed in, and
		/// it will be reused instead of a new object being allocated.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode getPreviousDocument(
			DOMNode	nodeToReuse)
		{
			RCODE		rc;
			IntPtr	pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;

			if ((rc = xflaim_DOMNode_getPreviousDocument( m_pNode, m_db.getDb(),
				ref pNode)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( makeNode( nodeToReuse, pNode));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getPreviousDocument(
			IntPtr			pNode,
			IntPtr			pDb,
			ref IntPtr		ppNode);

//-----------------------------------------------------------------------------
// getNextDocument
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves the next document.  This node must be a root node or
		/// a document node.
		/// </summary>
		/// <param name="nodeToReuse">
		/// An existing <see cref="DOMNode"/> object can optionally be passed in, and
		/// it will be reused instead of a new object being allocated.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode getNextDocument(
			DOMNode	nodeToReuse)
		{
			RCODE		rc;
			IntPtr	pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;

			if ((rc = xflaim_DOMNode_getNextDocument( m_pNode, m_db.getDb(),
				ref pNode)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( makeNode( nodeToReuse, pNode));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getNextDocument(
			IntPtr			pNode,
			IntPtr			pDb,
			ref IntPtr		ppNode);

//-----------------------------------------------------------------------------
// getPrefix
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves the prefix string for this node.
		/// </summary>
		/// <returns>
		/// Returns a string containing the node's prefix.
		/// </returns>
		public string getPrefix()
		{
			RCODE		rc;
			uint		uiNumChars;
			char []	cPrefix;

			if ((rc = xflaim_DOMNode_getPrefixChars( m_pNode, m_db.getDb(),
				out uiNumChars)) != 0)
			{
				throw new XFlaimException( rc);
			}
			cPrefix = new char [uiNumChars + 1];

			if ((rc = xflaim_DOMNode_getPrefix( m_pNode, m_db.getDb(),
				uiNumChars, cPrefix)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( new string( cPrefix, 0, (int)uiNumChars));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getPrefixChars(
			IntPtr		pNode,
			IntPtr		pDb,
			out uint		puiNumChars);

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getPrefix(
			IntPtr	pNode,
			IntPtr		pDb,
			uint		uiNumChars,
			[MarshalAs(UnmanagedType.LPArray), Out]
			char []	puzPrefix);

//-----------------------------------------------------------------------------
// getPrefixId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves the name id of the prefix for this node.
		/// </summary>
		/// <returns>
		/// Returns prefix name id.
		/// </returns>
		public uint getPrefixId()
		{
			RCODE		rc;
			uint		uiPrefixId;

			if ((rc = xflaim_DOMNode_getPrefixId( m_pNode, m_db.getDb(),
				out uiPrefixId)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( uiPrefixId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getPrefixId(
			IntPtr	pNode,
			IntPtr	pDb,
			out uint	puiPrefixId);

//-----------------------------------------------------------------------------
// getEncDefId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves the id of the encryption definition that is being used
		/// to encrypt this node's data.  Zero is returned if the data is not
		/// being encrypted.
		/// </summary>
		/// <returns>
		/// Returns encryption definition id.
		/// </returns>
		public uint getEncDefId()
		{
			RCODE		rc;
			uint		uiEncDefId;

			if ((rc = xflaim_DOMNode_getEncDefId( m_pNode, m_db.getDb(),
				out uiEncDefId)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( uiEncDefId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getEncDefId(
			IntPtr	pNode,
			IntPtr	pDb,
			out uint	puiEncDefId);

//-----------------------------------------------------------------------------
// setPrefix
//-----------------------------------------------------------------------------

		/// <summary>
		/// Sets the namespace prefix for this node
		/// </summary>
		/// <param name="sPrefix">
		/// The prefix that is to be set for this node
		/// </param>
		public void setPrefix(
			string	sPrefix)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setPrefix( m_pNode, m_db.getDb(),
				sPrefix)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_setPrefix(
			IntPtr	pNode,
			IntPtr	pDb,
			[MarshalAs(UnmanagedType.LPWStr), In]
			string	sPrefix);

//-----------------------------------------------------------------------------
// setPrefixId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Sets the namespace prefix for this node.
		/// </summary>
		/// <param name="uiPrefixId">
		/// The prefix that is to be set for this node
		/// </param>
		public void setPrefixId(
			uint	uiPrefixId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setPrefixId( m_pNode, m_db.getDb(),
				uiPrefixId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_setPrefixId(
			IntPtr	pNode,
			IntPtr	pDb,
			uint		uiPrefixId);

//-----------------------------------------------------------------------------
// getNamespaceURI
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves the namespace URI string for this node.
		/// </summary>
		/// <returns>
		/// Returns a string containing the node's namespace URI.
		/// </returns>
		public string getNamespaceURI()
		{
			RCODE		rc;
			uint		uiNumChars;
			char []	cNamespaceURI;

			if ((rc = xflaim_DOMNode_getNamespaceURIChars( m_pNode, m_db.getDb(),
				out uiNumChars)) != 0)
			{
				throw new XFlaimException( rc);
			}
			cNamespaceURI = new char [uiNumChars + 1];

			if ((rc = xflaim_DOMNode_getNamespaceURI( m_pNode, m_db.getDb(),
				uiNumChars, cNamespaceURI)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( new string( cNamespaceURI, 0, (int)uiNumChars));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getNamespaceURIChars(
			IntPtr		pNode,
			IntPtr		pDb,
			out uint		puiNumChars);

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getNamespaceURI(
			IntPtr	pNode,
			IntPtr	pDb,
			uint		uiNumChars,
			[MarshalAs(UnmanagedType.LPArray), Out]
			char []	puzNamespaceURI);

//-----------------------------------------------------------------------------
// getLocalName
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves the name of this node, without the namespace prefix.
		/// </summary>
		/// <returns>
		/// Returns a string containing the node's local name.
		/// </returns>
		public string getLocalName()
		{
			RCODE		rc;
			uint		uiNumChars;
			char []	cLocalName;

			if ((rc = xflaim_DOMNode_getLocalNameChars( m_pNode, m_db.getDb(),
				out uiNumChars)) != 0)
			{
				throw new XFlaimException( rc);
			}
			cLocalName = new char [uiNumChars + 1];

			if ((rc = xflaim_DOMNode_getLocalName( m_pNode, m_db.getDb(),
				uiNumChars, cLocalName)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( new string( cLocalName, 0, (int)uiNumChars));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getLocalNameChars(
			IntPtr		pNode,
			IntPtr		pDb,
			out uint		puiNumChars);

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getLocalName(
			IntPtr	pNode,
			IntPtr	pDb,
			uint		uiNumChars,
			[MarshalAs(UnmanagedType.LPArray), Out]
			char []	puzLocalName);

//-----------------------------------------------------------------------------
// getQualifiedName
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves the fully qualified name (namespace prefix plus local
		/// name) for this element or attribute.
		/// </summary>
		/// <returns>
		/// Returns a string containing the node's fully qualified name.
		/// </returns>
		public string getQualifiedName()
		{
			RCODE		rc;
			uint		uiNumChars;
			char []	cQualifiedName;

			if ((rc = xflaim_DOMNode_getQualifiedNameChars( m_pNode, m_db.getDb(),
				out uiNumChars)) != 0)
			{
				throw new XFlaimException( rc);
			}
			cQualifiedName = new char [uiNumChars + 1];

			if ((rc = xflaim_DOMNode_getQualifiedName( m_pNode, m_db.getDb(),
				uiNumChars, cQualifiedName)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( new string( cQualifiedName, 0, (int)uiNumChars));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getQualifiedNameChars(
			IntPtr		pNode,
			IntPtr		pDb,
			out uint		puiNumChars);

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getQualifiedName(
			IntPtr	pNode,
			IntPtr	pDb,
			uint		uiNumChars,
			[MarshalAs(UnmanagedType.LPArray), Out]
			char []	puzQualifiedName);

//-----------------------------------------------------------------------------
// getCollection
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves the collection this node belongs to.
		/// </summary>
		/// <returns>
		/// Returns the collection number.
		/// </returns>
		public uint getCollection()
		{
			RCODE		rc;
			uint		uiCollection;

			if ((rc = xflaim_DOMNode_getCollection( m_pNode, m_db.getDb(),
				out uiCollection)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( uiCollection);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getCollection(
			IntPtr	pNode,
			IntPtr	pDb,
			out uint	puiCollection);

//-----------------------------------------------------------------------------
// createAnnotation
//-----------------------------------------------------------------------------

		/// <summary>
		/// Creates an annotation node and associates it with this node.
		/// </summary>
		/// <param name="nodeToReuse">
		/// An existing <see cref="DOMNode"/> object can optionally be passed in, and
		/// it will be reused instead of a new object being allocated.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode createAnnotation(
			DOMNode			nodeToReuse)
		{
			RCODE		rc;
			IntPtr	pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;

			if ((rc = xflaim_DOMNode_createAnnotation( m_pNode, m_db.getDb(),
									ref pNode)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( makeNode( nodeToReuse, pNode));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_createAnnotation(
			IntPtr			pNode,
			IntPtr			pDb,
			ref IntPtr		ppNode);

//-----------------------------------------------------------------------------
// getAnnotation
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieve the annotation node that is associated with this node.
		/// </summary>
		/// <param name="nodeToReuse">
		/// An existing <see cref="DOMNode"/> object can optionally be passed in, and
		/// it will be reused instead of a new object being allocated.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode getAnnotation(
			DOMNode			nodeToReuse)
		{
			RCODE		rc;
			IntPtr	pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;

			if ((rc = xflaim_DOMNode_getAnnotation( m_pNode, m_db.getDb(),
				ref pNode)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( makeNode( nodeToReuse, pNode));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getAnnotation(
			IntPtr			pNode,
			IntPtr			pDb,
			ref IntPtr		ppNode);

//-----------------------------------------------------------------------------
// getAnnotationId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieve the node ID of the annotation node that is associated with this node.
		/// </summary>
		/// <returns>
		/// Returns node ID of the annotation node associated with this node.
		/// </returns>
		public ulong getAnnotationId()
		{
			RCODE		rc;
			ulong		ulAnnotationId;

			if ((rc = xflaim_DOMNode_getAnnotationId( m_pNode, m_db.getDb(),
				out ulAnnotationId)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( ulAnnotationId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getAnnotationId(
			IntPtr		pNode,
			IntPtr		pDb,
			out ulong	pulAnnotationId);

//-----------------------------------------------------------------------------
// hasAnnotation
//-----------------------------------------------------------------------------

		/// <summary>
		/// Determine if a node has an annotation node associated with it.
		/// </summary>
		/// <returns>
		/// Returns true if there is an annotation node assocated with this node,
		/// false otherwise.
		/// </returns>
		public bool hasAnnotation()
		{
			RCODE		rc;
			int		bHasAnnotation;

			if ((rc = xflaim_DOMNode_hasAnnotation( m_pNode, m_db.getDb(),
								out bHasAnnotation)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( bHasAnnotation != 0 ? true : false);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_hasAnnotation(
			IntPtr	pNode,
			IntPtr	pDb,
			out int	pbHasAnnotation);

//-----------------------------------------------------------------------------
// getMetaValue
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get a node's "meta" value.
		/// </summary>
		/// <returns>
		/// Returns node's meta value.
		/// </returns>
		public ulong getMetaValue()
		{
			RCODE		rc;
			ulong		ulValue;

			if ((rc = xflaim_DOMNode_getMetaValue( m_pNode, m_db.getDb(),
				out ulValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( ulValue);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getMetaValue(
			IntPtr		pNode,
			IntPtr		pDb,
			out ulong	pulValue);

//-----------------------------------------------------------------------------
// setMetaValue
//-----------------------------------------------------------------------------

		/// <summary>
		/// Set a node's "meta" value.
		/// </summary>
		/// <param name="ulValue">
		/// Value to set.
		/// </param>
		public void setMetaValue(
			ulong	ulValue)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setMetaValue( m_pNode, m_db.getDb(),
				ulValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_setMetaValue(
			IntPtr	pNode,
			IntPtr	pDb,
			ulong		ulValue);

	}
}
