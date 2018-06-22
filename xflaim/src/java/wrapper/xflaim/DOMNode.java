//------------------------------------------------------------------------------
// Desc:	DOM Node
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

package xflaim;

/**
 * This class encapsulates the XFlaim F_DOMNode interface.
 */
public class DOMNode 
{
	
	DOMNode( 
		long			lThis,
		Db 			jdb) throws XFlaimException
	{
		if (lThis == 0)
		{
			throw new XFlaimException( -1, "No legal reference to a DOMNode");
		}
		
		m_this = lThis;
		
		if (jdb == null)
		{
			throw new XFlaimException( -1, "No legal jDb reference");
		}
		
		m_jdb = jdb;
	}
	
	protected void finalize()
	{
		// The F_DOMNode and F_Db classes are not thread-safe.  The proper way
		// of using XFlaim is to get a new instance of Db for each thread.
		// Unfortunately, the garbage collector runs in its own thread.  This
		// leads to a potential race condition down in the C++ code when one
		// thread tries to create an already existing node (which results in a
		// call to F_DOMNode::AddRef()) and the GC tries to destroy the same
		// node (which results in a call to F_DOMNode::Release()).
		// We protect against this by synchronizing against the instance of
		// Db.  Note that we are not protecting any of the accesses to the
		// node; only creating and destroying.  DOMNode and Db are still
		// not completely thread-safe.
		
		synchronized( m_jdb)
		{
			// Release the associated DOMNode.
			
			if (m_this != 0)
			{
				_release( m_this);
			}
		}
		
		// Free our reference to the Db object.
		
		m_jdb = null;
	}
	
	public void release()
	{
		synchronized( m_jdb)
		{
			if (m_this != 0)
			{
				_release( m_this);
			}
		}
		
		m_jdb = null;
	}
	
	public long getThis()
	{
		return m_this;
	}
	
	/**
	 * Creates a new DOM node and inserts it into the database in the
	 * specified position relative to the current node.  An existing
	 * DOMNode object can optionally be passed in, and it will be reused
	 * instead of a new object being allocated.
	 * @param iNodeType An integer representing the type of node to create.
	 * (Use the constants in {@link xflaim.FlmDomNodeType FlmDomNodeType}.)
	 * @param iNameId The dictionary tag number that represents the node name.
	 * This value must exist in the dictionary before it can be used here.  The
	 * value may be one of the predefined ones, or it may be created with
	 * {@link Db#createElementDef Db::createElementDef}.
	 * @param iInsertLoc An integer representing the relative position to insert
	 * the new node.  (Use the constants in 
	 * {@link xflaim.FlmInsertLoc FlmInsertLoc}.)
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an instance of DOMNode
	 * @throws XFlaimException
	 */
	public DOMNode createNode(
		int			iNodeType,
		int			iNameId,
		int			iInsertLoc,
		DOMNode		ReusedNode) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.m_this
											  : 0;
		
		// See to comments in the finalize function for an explanation of this
		// call synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _createNode( m_this, m_jdb.m_this, iNodeType, iNameId,
									   iInsertLoc, lReusedNodeRef);
		}
		
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
		
		return( NewNode);
	}
	
	/**
	 * Creates a new element node and inserts it into the database in the
	 * as either the first or last child of the current node.  An existing
	 * DOMNode object can optionally be passed in, and it will be reused
	 * instead of a new object being allocated.
	 * @param iChildElementNameId The dictionary tag number that represents the
	 * element node name.
	 * This value must exist in the dictionary before it can be used here.  The
	 * value may be one of the predefined ones, or it may be created with
	 * {@link Db#createElementDef Db::createElementDef}.
	 * @param bFirstChild Specifies whether the new element is to be created as
	 * a first or last child.
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an instance of DOMNode
	 * @throws XFlaimException
	 */
	public DOMNode createChildElement(
		int			iChildElementNameId,
		boolean		bFirstChild,
		DOMNode		ReusedNode) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.m_this
											  : 0;
		
		// See to comments in the finalize function for an explanation of this
		// call synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _createChildElement( m_this, m_jdb.m_this,
										iChildElementNameId, bFirstChild, lReusedNodeRef);
		}
		
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
		
		return( NewNode);
	}

	/**
	 *  Removes this node as well as all of it's descendants from the database.
	 * @throws XFlaimException
	 */
	public void deleteNode() throws XFlaimException
	{
		_deleteNode( m_this, m_jdb.m_this);		
	}
	
	/**
	 * Removes the children of the current node from the database.
	 * @throws XFlaimException
	 */
	public void deleteChildren() throws XFlaimException
	{
		_deleteChildren( m_this, m_jdb.m_this);	
	}

	/**
	 * Returns the type of node.  Returned value will be one of those listed in
	 * {@link xflaim.FlmDomNodeType FlmDomNodeType}.
	 * @return Returns the type of the current node.
	 */
	public int getNodeType() throws XFlaimException
	{
		return _getNodeType( m_this);
	}

	/**
 	* Determine if data for the current node is associated with the node, or
	* with a child node.  Element nodes may not have data associated with them,
	* but with child data nodes instead.
 	* @return Returns true if this node's data is associated with it.
 	*/
	public boolean isDataLocalToNode() throws XFlaimException
	{
		return( _isDataLocalToNode( m_this, m_jdb.m_this));
	}
	
	/**
	 * Creates a new attribute node assigned to the current node.  Note that
	 * some nodes are not allowed to have attributes.
	 * @param iNameId The dictionary tag number that represents the node name.
	 * This value must exist in the dictionary before it can be used here.  The
	 * value may be one of the predefined ones, or it may be created with
	 * {@link Db#createElementDef Db::createElementDef}.
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an instance of DOMNode
	 */
	public DOMNode createAttribute(
		int			iNameId,
		DOMNode		ReusedNode) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.m_this
											  : 0;
		
		// See to comments in the finalize function for an explanation of this
		// call synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _createAttribute( m_this, m_jdb.m_this, iNameId,
											lReusedNodeRef);
		}

		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}

		return( NewNode);
	}

	/**
	 * Retrieves the first attribute node associated with the current node.
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an instance of DOMNode
	 * @throws XFlaimException
	 */
	public DOMNode getFirstAttribute(
		DOMNode ReusedNode) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.m_this
											  : 0;

		// See the comment in the finalize function for an explanation of
		// this synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getFirstAttribute( m_this, 
									m_jdb.m_this, lReusedNodeRef);
		}

		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}

		return( NewNode);
	}
	
	/**
	 * Retrieves the last attribute node associated with the current node.
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an instance of DOMNode.
	 * @throws XFlaimException
	 */
	public DOMNode getLastAttribute(
		DOMNode		ReusedNode) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.m_this
											  : 0;
		
		// See to comments in the finalize function for an explanation of this
		// call synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getLastAttribute( m_this, m_jdb.m_this, lReusedNodeRef);
		}
		
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}

		return( NewNode);
	}
	
	/**
	 * Retrieves the requested attribute node associated with this node.
	 * @param iAttributeId The dictionary tag number of the requested
	 * attribute.  This name id must exist in the dictionary before it can be
	 * used here.  The name id may be one of the predefined ones, or it may be
	 * created with {@link Db#createAttributeDef Db::createAttributeDef}.
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an instance of DOMNode
	 * @throws XFlaimException
	 */
	public DOMNode getAttribute(
		int			iAttributeId,
		DOMNode		ReusedNode) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.m_this
											  : 0;
		
		// See to comments in the finalize function for an explanation of this
		// call synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getAttribute( m_this, m_jdb.m_this, iAttributeId,
										 lReusedNodeRef);
		}
		
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}

		return( NewNode); 
	}
	
	/**
	 * Removes the specified attribute node from the current node.
	 * @param iAttributeId The dictionary tag number representing the
	 * attribute node to be deleted.
	 * @throws XFlaimException
	 */
	public void deleteAttribute(
		int		iAttributeId) throws XFlaimException
	{
		_deleteAttribute( m_this, m_jdb.m_this, iAttributeId);
	}
  
  	/**
  	 * Looks through the list of attributes for the one specified in iNameId.
  	 * Note that this function's semantics differ from its C++ counterpart.
  	 * @return Returns true if the attribute was found; false otherwise.
  	 * @throws XFlaimException
  	 */
	public boolean hasAttribute(
		int		iNameId)  throws XFlaimException
	{
		return _hasAttribute( m_this, m_jdb.m_this, iNameId);
	}
	
	/**
	 * Tests to see if this node as any attributes associated with it.
	 * @return Returns true if the node has any attributes.
	 * @throws XFlaimException
	 */
	public boolean hasAttributes() throws XFlaimException
	{
		return _hasAttributes( m_this, m_jdb.m_this);
	}
	
	/**
	 * Tests to see if this node has a next sibling.
	 * @return Returns true if this node has a next sibling.
	 * @throws XFlaimException
	 */
	public boolean hasNextSibling() throws XFlaimException
	{
		return _hasNextSibling( m_this, m_jdb.m_this);
	}

	/**
	 * Tests to see if this node has a previous sibling.
	 * @return Returns true if this node has a previous sibling.
	 * @throws XFlaimException
	 */
	public boolean hasPreviousSibling() throws XFlaimException
	{
		return _hasPreviousSibling( m_this, m_jdb.m_this);
	}

	/**
	 * Tests to see if this node has any child nodes.
	 * @return Returns true if this node has any children.
	 * @throws XFlaimException
	 */
	public boolean hasChildren() throws XFlaimException
	{
		return _hasChildren( m_this, m_jdb.m_this);
	}	
	/**
	 * Tests to see if this node is an attribute node that defines a namespace.
	 * @return Returns true if this node is an attribute node that defines a
	 * namespace and false if this node not an attribute node or it does not
	 * define a namespace. 
	 * @throws XFlaimException
	 */
	public boolean isNamespaceDecl() throws XFlaimException
	{
		return _isNamespaceDecl( m_this, m_jdb.m_this);
	}

	/**
	 * Retrieves the node ID for this node's parent.
	 * @return Returns the parent node's node ID.
	 * @throws XFlaimException
	 */	
	public long getParentId() throws XFlaimException
	{
		return _getParentId( m_this, m_jdb.m_this);
	}
	
	/**
	 * Retrieves the node ID for this node.
	 * @return Returns the node ID.
	 */
	public long getNodeId()
	{
		return _getNodeId( m_this, m_jdb.m_this);
	}
	
	/**
	 * Retrieves the node ID for the root node in the document
	 * @return Returns the root node's node ID.
	 * @throws XFlaimException
	 */
	public long getDocumentId() throws XFlaimException
	{
		return _getDocumentId( m_this, m_jdb.m_this);
	}
	
	/**
	 * Retrieves the node ID for this node's previous sibling. 
	 * @return Returns the previous sibling node's node ID.
	 * @throws XFlaimException
	 */
	public long getPrevSibId() throws XFlaimException
	{
		return _getPrevSibId(  m_this, m_jdb.m_this);
	}

	/**
	 * Retrieves the node ID for this node's next sibling.
	 * @return Returns the next sibling node's node ID.
	 * @throws XFlaimException
	 */
	public long getNextSibId() throws XFlaimException
	{
		return _getNextSibId(  m_this, m_jdb.m_this);
	}

	/**
	 * Retrieves the node ID for this node's first child. 
	 * @return Returns the first child node's node ID.
	 * @throws XFlaimException
	 */	
	public long getFirstChildId() throws XFlaimException
	{
		return _getFirstChildId(  m_this, m_jdb.m_this);
	}
	
	/**
	 * Retrieves the node ID for this node's next child. 
	 * @return Returns the next child node's node ID.
	 * @throws XFlaimException
	 */	
	public long getLastChildId() throws XFlaimException
	{
		return _getLastChildId(  m_this, m_jdb.m_this);
	}

	/**
	 * Retrieves this node's name ID. 
	 * @return Returns the name ID for this node.
	 * @throws XFlaimException
	 */
	public int getNameId() throws XFlaimException
	{
		return _getNameId( m_this, m_jdb.m_this);
	}

	/**
	 * Assigns a 64-bit value to this node.
	 * @param lValue The value to be assigned.
	 * @throws XFlaimException
	 */
	public void setLong(
		long			lValue) throws XFlaimException
	{
		_setLong( m_this, m_jdb.m_this, lValue, 0);
	}
	
	/**
	 * Assigns a 64-bit value to this node.
	 * @param lValue The value to be assigned.
	 * @param iEncId Encryption definition to use.  If zero is passed, the
	 * data will not be encrypted.
	 * @throws XFlaimException
	 */
	public void setLong(
		long			lValue,
		int			iEncId) throws XFlaimException
	{
		_setLong( m_this, m_jdb.m_this, lValue, iEncId);	
	}
	
	/**
	 * Assigns a 64-bit value to an attribute for this element node (node must
	 * be an element node).
	 * @param iAttrNameId The attribute id whose value is to be assigned.
	 * @param lValue The value to be assigned.
	 * @throws XFlaimException
	 */
	public void setAttributeValueLong(
		int			iAttrNameId,
		long			lValue) throws XFlaimException
	{
		_setAttributeValueLong( m_this, m_jdb.m_this, iAttrNameId, lValue, 0);	
	}
	
	/**
	 * Assigns a 64-bit value to an attribute for this element node (node must
	 * be an element node).
	 * @param iAttrNameId The attribute id whose value is to be assigned.
	 * @param lValue The value to be assigned.
	 * @param iEncId Encryption definition to use.  If zero is passed, the
	 * data will not be encrypted.
	 * @throws XFlaimException
	 */
	public void setAttributeValueLong(
		int			iAttrNameId,
		long			lValue,
		int			iEncId) throws XFlaimException
	{
		_setAttributeValueLong( m_this, m_jdb.m_this, iAttrNameId, lValue, iEncId);	
	}
	
	/**
	 * Assigns a text string to this node.  Existing text is either
	 * overwritten or has the new text appended to it.  See the
	 * explanation for the bLast parameter.
	 * @param sValue The text to be assigned
	 * @param bLast Specifies whether sValue is the last text to be
	 * appended to this node.  If false, then another call to setString
	 * is expected, and the new text will be appended to the text currently
	 * stored in this node.  If true, then no more text is expected and 
	 * another call to setString will overwrite the what is currently
	 * stored in this node.
	 * @throws XFlaimException
	 */
	public void setString(
		String		sValue,
		boolean		bLast) throws XFlaimException
	{
		_setString( m_this, m_jdb.m_this, sValue, bLast, 0);
	}

	/**
	 * Assigns a text string to this node.  Existing text is either
	 * overwritten or has the new text appended to it.  See the
	 * explanation for the bLast parameter.
	 * @param sValue The text to be assigned
	 * @param bLast Specifies whether sValue is the last text to be
	 * appended to this node.  If false, then another call to setString
	 * is expected, and the new text will be appended to the text currently
	 * stored in this node.  If true, then no more text is expected and 
	 * another call to setString will overwrite the what is currently
	 * stored in this node.
	 * @param iEncId Specifies the encryption definition to use to encrypt
	 * this data.  Zero means that the data is not to be encrypted.
	 * @throws XFlaimException
	 */
	public void setString(
		String		sValue,
		boolean		bLast,
		int			iEncId) throws XFlaimException
	{
		_setString( m_this, m_jdb.m_this, sValue, bLast, iEncId);
	}

	/**
	 * Assigns or appends a text string to this node.  This function is
	 * equivalent to setString( sValue, true).
	 * @param sValue The text to be assigned.
	 * @throws XFlaimException
	 */
	public void setString(
		String		sValue) throws XFlaimException
	{
		_setString( m_this, m_jdb.m_this, sValue, true, 0);			
	}
	
	/**
	 * Assigns or appends a text string to this node.  This function is
	 * equivalent to setString( sValue, true, iEncId).
	 * @param sValue The text to be assigned.
	 * @param iEncId The encryption id to be used to encrypt the data.  If zero
	 * the data will not be encrypted.
	 * @throws XFlaimException
	 */
	public void setString(
		String		sValue,
		int			iEncId) throws XFlaimException
	{
		_setString( m_this, m_jdb.m_this, sValue, true, iEncId);			
	}
	
	/**
	 * Assigns a text string to an attribute for this element node (node must be
	 * an element node).
	 * @param iAttrNameId Attribute id of the attribute whose value is to be set.
	 * @param sValue The text to be assigned.
	 * @throws XFlaimException
	 */
	public void setAttributeValueString(
		int			iAttrNameId,
		String		sValue) throws XFlaimException
	{
		_setAttributeValueString( m_this, m_jdb.m_this, iAttrNameId, sValue, 0);			
	}
	
	/**
	 * Assigns a text string to an attribute for this element node (node must be
	 * an element node).
	 * @param iAttrNameId Attribute id of the attribute whose value is to be set.
	 * @param sValue The text to be assigned.
	 * @param iEncId The encryption id to be used to encrypt the data.  If zero
	 * the data will not be encrypted.
	 * @throws XFlaimException
	 */
	public void setAttributeValueString(
		int			iAttrNameId,
		String		sValue,
		int			iEncId) throws XFlaimException
	{
		_setAttributeValueString( m_this, m_jdb.m_this, iAttrNameId, sValue, iEncId);			
	}
	
	/**
	 * Assigns a piece of binary data to this node.
	 * @param Value An array of bytes to be stored in this node.
	 * @throws XFlaimException
	 */
	public void setBinary(
		byte[] 		Value) throws XFlaimException
	{
		_setBinary( m_this, m_jdb.m_this, Value, true, 0);
	}

	/**
	 * Assigns a piece of binary data to this node.
	 * @param Value An array of bytes to be stored in this node.
	 * @param iEncId Encryption id to be used to encrypt the data.  If zero
	 * is passed, data will not be encrypted.
	 * @throws XFlaimException
	 */
	public void setBinary(
		byte[] 		Value,
		int			iEncId) throws XFlaimException
	{
		_setBinary( m_this, m_jdb.m_this, Value, true, iEncId);
	}

	/**
	 * Assigns a piece of binary data to this node.
	 * @param Value An array of bytes to be stored in this node.
	 * @param bLast Specifies whether Value is the last data to be
	 * appended to this node.  If false, then another call to setBinary
	 * is expected, and the new data will be appended to the data currently
	 * stored in this node.  If true, then no more data is expected and 
	 * another call to setBinary will overwrite the what is currently
	 * stored in this node.
	 * @throws XFlaimException
	 */
	public void setBinary(
		byte[] 		Value,
		boolean		bLast) throws XFlaimException
	{
		_setBinary( m_this, m_jdb.m_this, Value, bLast, 0);
	}

	/**
	 * Assigns a piece of binary data to this node.
	 * @param Value An array of bytes to be stored in this node.
	 * @param bLast Specifies whether Value is the last data to be
	 * appended to this node.  If false, then another call to setBinary
	 * is expected, and the new data will be appended to the data currently
	 * stored in this node.  If true, then no more data is expected and 
	 * another call to setBinary will overwrite the what is currently
	 * stored in this node.
	 * @param iEncId Encryption id to be used to encrypt the data.  If zero
	 * is passed, data will not be encrypted.
	 * @throws XFlaimException
	 */
	public void setBinary(
		byte[] 		Value,
		boolean		bLast,
		int			iEncId) throws XFlaimException
	{
		_setBinary( m_this, m_jdb.m_this, Value, bLast, iEncId);
	}

	/**
	 * Assigns a piece of binary data to an attribute of this element node (node
	 * must be an element).
	 * @param iAttrNameId Attribute id of the attribute whose value is to be set.
	 * @param Value An array of bytes to be stored.
	 * @throws XFlaimException
	 */
	public void setAttributeValueBinary(
		int			iAttrNameId,
		byte[] 		Value) throws XFlaimException
	{
		_setAttributeValueBinary( m_this, m_jdb.m_this, iAttrNameId, Value, 0);
	}

	/**
	 * Assigns a piece of binary data to an attribute of this element node (node
	 * must be an element).
	 * @param iAttrNameId Attribute id of the attribute whose value is to be set.
	 * @param Value An array of bytes to be stored.
	 * @param iEncId Encryption id to be used to encrypt the data.  If zero
	 * is passed, data will not be encrypted.
	 * @throws XFlaimException
	 */
	public void setAttributeValueBinary(
		int			iAttrNameId,
		byte[] 		Value,
		int			iEncId) throws XFlaimException
	{
		_setAttributeValueBinary( m_this, m_jdb.m_this, iAttrNameId, Value, iEncId);
	}

	/**
	 * Retrieves the amount of memory occupied by the value of this node. 
	 * @return Returns the length of the data stored in the node (in bytes).
	 * @throws XFlaimException
	 */
	public long getDataLength() throws XFlaimException
	{
		return _getDataLength( m_this, m_jdb.m_this);
	}

	/**
	 * Retrieves the type of the value stored in this node.  The value will
	 * be one of those listed in 
	 * {@link xflaim.FlmDataType FlmDataType}.
	 * @return Returns the type of the value stored in this node.
	 * @throws XFlaimException
	 */
	public int getDataType() throws XFlaimException
	{
		return _getDataType( m_this, m_jdb.m_this);
	}

	/**
	 * Retrieves the value stored in this node as a long. 
	 * @return Returns the value stored in the node.
	 * @throws XFlaimException
	 */
	public long getLong() throws XFlaimException
	{
		return _getLong( m_this, m_jdb.m_this);
	}

	/**
	 * Retrieves the value stored in an attribute associated with this node
	 * (node must be an element node) as a long. 
	 * @param iAttrNameId Name of attribute whose value is to be returned.
	 * @param bDefaultOk If true, specifies that if the attribute is not found
	 * then the value in lDefaultToUse is to be returned.  If false, and the
	 * attribute is not found, an exception will be thrown.
	 * @param lDefaultToUse Default value to use if the attribute is not
	 * found and bDefaultOk is true.
	 * @return Returns the value stored in the element node's attribute.
	 * @throws XFlaimException
	 */
	public long getAttributeValueLong(
		int		iAttrNameId,
		boolean	bDefaultOk,
		long		lDefaultToUse) throws XFlaimException
	{
		return _getAttributeValueLong( m_this, m_jdb.m_this, iAttrNameId,
							bDefaultOk, lDefaultToUse);
	}

	/**
	 * Retrieves a string representation of the value stored in this node. 
	 * @return Returns the value stored in the node.
	 * @throws XFlaimException
	 */	
	public String getString() throws XFlaimException
	{
		return _getString( m_this, m_jdb.m_this, 0, 0);
	}

	/**
	 * Retrieves a sub-string of the value stored in this node. 
	 * @param iStartPos Starting character position in string to retrieve sub-string from.
	 * @param iNumChars Maximum number of characters to retrieve.  May return
	 * fewer than this number of characters if there are not that many
	 * characters available from the specified starting position.
	 * @return Returns the sub-string stored in the node.
	 * @throws XFlaimException
	 */	
	public String getSubString(
		int	iStartPos,
		int	iNumChars) throws XFlaimException
	{
		return _getString( m_this, m_jdb.m_this, iStartPos, iNumChars);
	}

	/**
	 * Retrieves a string representation of the value stored in this element
	 * node's (node must be an element) attribute. 
	 * @param iAttrNameId Name id of attribute whose value is to be returned.
	 * @return Returns the value stored in the element node's attribute.
	 * @throws XFlaimException
	 */	
	public String getAttributeValueString(
		int	iAttrNameId) throws XFlaimException
	{
		return _getAttributeValueString( m_this, m_jdb.m_this, iAttrNameId);
	}

	/**
	 * Retrieves the number of unicode characters a string representation of
	 * the node's value would occupy.
	 * @return Returns the length of a string representation of this node's
	 * data.
	 * @throws XFlaimException
	 */
	public int getStringLen() throws XFlaimException
	{
		return _getStringLen( m_this, m_jdb.m_this);		
	}

	/**
	 * Retrieves the value of the node as raw data.
	 * @return Returns a byte array containing the value of this node. 
	 * @throws XFlaimException
	 */
	public byte[] getBinary() throws XFlaimException
	{
		return _getBinary( m_this, m_jdb.m_this, 0, 0);
	}

	/**
	 * Retrieves the value of the node as raw data.
	 * @param iStartPos Starting byte position in binary data to retrieve from.
	 * @param iNumBytes Maximum number of bytes to retrieve.  May return
	 * fewer than this number of bytes if there are not that many
	 * bytes available from the specified starting position.
	 * @return Returns a byte array containing the requested data from
	 * the value of this node. 
	 * @throws XFlaimException
	 */
	public byte[] getBinary(
		int	iStartPos,
		int	iNumBytes) throws XFlaimException
	{
		return _getBinary( m_this, m_jdb.m_this, iStartPos, iNumBytes);
	}

	/**
	 * Retrieves the value of the element node's (node must be an element)
	 * attribute as raw data.
	 * @param iAttrNameId Name of attribute whose data is to be returned.
	 * @return Returns a byte array containing the value of this element node's
	 * attribute. 
	 * @throws XFlaimException
	 */
	public byte[] getAttributeValueBinary(
		int	iAttrNameId) throws XFlaimException
	{
		return _getAttributeValueBinary( m_this, m_jdb.m_this, iAttrNameId);
	}

	/**
	 * Retrieves the document node of the current node. 
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an instance of DOMNode 
	 * @throws XFlaimException
	 */
	public DOMNode getDocumentNode(
		DOMNode			ReusedNode) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.m_this
											  : 0;
		
		// See to comments in the finalize function for an explanation of this
		// call synchronized call
		synchronized( m_jdb)
		{
			lNewNodeRef = _getDocumentNode( m_this, m_jdb.m_this, lReusedNodeRef);
		}
		
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
		
		return NewNode;
	}

	/**
	 * Retrieves the parent of the current node. 
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an instance of DOMNode 
	 * @throws XFlaimException
	 */
	public DOMNode getParentNode(
		DOMNode			ReusedNode) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.m_this
											  : 0;
		
		// See to comments in the finalize function for an explanation of this
		// call synchronized call
		synchronized( m_jdb)
		{
			lNewNodeRef = _getParentNode( m_this, m_jdb.m_this, lReusedNodeRef);
		}
		
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
		
		return NewNode;
	}

	/**
	 * Retrieves the first node in the current node's list of child nodes.
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an instance of DOMNode
	 * @throws XFlaimException
	 */
	public DOMNode getFirstChild(
		DOMNode		ReusedNode) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.m_this
											  : 0;
		
		// See to comments in the finalize function for an explanation of this
		// call synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getFirstChild( m_this, m_jdb.m_this, lReusedNodeRef);
		}
			
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
			
		return NewNode;
	}
	
	/**
	 * Retrieves the last node in the current node's list of child nodes. 
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an instance of DOMNode;
	 * @throws XFlaimException
	 */
	public DOMNode getLastChild(
		DOMNode		ReusedNode) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.m_this
											  : 0;
		
		// See to comments in the finalize function for an explanation of this
		// synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getLastChild( m_this, m_jdb.m_this, lReusedNodeRef);
		}
			
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
			
		return NewNode;
	}
	
	/**
	 * Retrieves the first instance of the specified type of node from the
	 * current node's list of child nodes.
	 * @param iNodeType The value representing the node type.  
	 * (Use the constants in {@link xflaim.FlmDomNodeType FlmDomNodeType}.)
	 * @param ReusedNode An instance of {@link xflaim.DOMNode DOMNode} which is
	 * no longer needed and can be reassigned to point to different data in the
	 * database.  (Reusing {@link xflaim.DOMNode DOMNode} objects is encouraged
	 * as it saves the system from allocating and freeing memory for each
	 * object.)  Can be null, if no instances are available to be reused.
	 * @return Returns an instance of {@link xflaim.DOMNode DOMNode}
	 * @throws XFlaimException
	 */
	public DOMNode getChild(
		int			iNodeType,
		DOMNode		ReusedNode) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.m_this
											  : 0;
		
		// See to comments in the finalize function for an explanation of this
		// synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getChild( m_this, m_jdb.m_this, 
									iNodeType, lReusedNodeRef);
		}
			
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
			
		return NewNode;
	}

	/**
	 * Retrieves the specified element node from the current node's 
	 * list of child nodes.
	 * @param iNameId The name ID for the desired node
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an  instance of DOMNode
	 * @throws XFlaimException
	 */
	public DOMNode getChildElement(
		int			iNameId,
		DOMNode		ReusedNode) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.m_this
											  : 0;
		
		// See to comments in the finalize function for an explanation of this
		// synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getChildElement( m_this, m_jdb.m_this, 
									iNameId, lReusedNodeRef);
		}
			
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
			
		return NewNode;
	}

	/**
	 * Retrieves the specified element node from the current node's 
	 * list of sibling nodes.
	 * @param iNameId The name ID of the desired node.
	 * @param bNext If true, will search forward following each node's
	 * "next_sibling" link; if false, will follow each node's "prev_sibling"
	 * link. 
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an instance of DOMNode
	 * @throws XFlaimException
	 */
	public DOMNode getSiblingElement(
		int			iNameId,
		boolean		bNext,
		DOMNode		ReusedNode) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.m_this
											  : 0;
		
		// See to comments in the finalize function for an explanation of this
		// synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getSiblingElement( m_this, m_jdb.m_this, iNameId,
											  bNext, lReusedNodeRef);
		}
			
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
			
		return NewNode;		
	}

	/**
	 * Retrieves the specified element node from the current node's 
	 * ancestor nodes.
	 * @param iNameId The name ID of the desired node.
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an instance of DOMNode
	 * @throws XFlaimException
	 */
	public DOMNode getAncestorElement(
		int			iNameId,
		DOMNode		ReusedNode) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.m_this
											  : 0;
			
		// See to comments in the finalize function for an explanation of this
		// synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getAncestorElement( m_this, m_jdb.m_this, iNameId,
											  lReusedNodeRef);
		}
			
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
			
		return NewNode;
	}
	
	/**
	 * Retrieves the specified element node from the current node's 
	 * descendant nodes.
	 * @param iNameId The name ID of the desired node.
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an instance of DOMNode
	 * @throws XFlaimException
	 */
	public DOMNode getDescendantElement(
		int			iNameId,
		DOMNode		ReusedNode) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.m_this
											  : 0;
			
		// See to comments in the finalize function for an explanation of this
		// synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getDescendantElement( m_this, m_jdb.m_this, iNameId,
											  lReusedNodeRef);
		}
			
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
			
		return NewNode;		
	}
	
	/**
	 * Retrieve's the previous node from the current node's list of 
	 * siblings nodes. 
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an instance of DOMNode
	 * @throws XFlaimException
	 */
	public DOMNode getPreviousSibling(
		DOMNode 		ReusedNode) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.m_this
											  : 0;
		
		// See to comments in the finalize function for an explanation of this
		// synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getPreviousSibling( m_this, m_jdb.m_this, 
									lReusedNodeRef);
		}
			
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
			
		return NewNode;
	}
	
	/**
	 * Retrieves the next node from the current node's list of sibling nodes.
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an instance of DOMNode
	 * @throws XFlaimException
	 */
	public DOMNode getNextSibling(
		DOMNode		ReusedNode) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.m_this
											  : 0;
		
		// See to comments in the finalize function for an explanation of this
		// synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getNextSibling( m_this, m_jdb.m_this, lReusedNodeRef);
		}
			
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
			
		return NewNode;
	}

	/**
	 * Retrieves the previous document node.  The current node must be a root
	 * node or a document node.
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)
	 * @return Returns an instance of DOMNode
	 * @throws XFlaimException
	 */
	public DOMNode getPreviousDocument(
		DOMNode		ReusedNode) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.m_this
											  : 0;

		// See to comments in the finalize function for an explanation of this
		// synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getPreviousDocument( m_this, 
									m_jdb.m_this, lReusedNodeRef);
		}
			
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
			
		return NewNode;
	}

	/**
	 * Retrieves the next document node.  The current node must be a root
	 * node or a document node.
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)
	 * @return Returns an instance of DOMNode
	 * @throws XFlaimException
	 */
	public DOMNode getNextDocument(
		DOMNode 		ReusedNode) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.m_this
											  : 0;
		
		// See to comments in the finalize function for an explanation of this
		// synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getNextDocument( m_this, m_jdb.m_this, lReusedNodeRef);
		}
			
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
			
		return NewNode;
	}

	/**
	 * Retrieves the namespace prefix for this node
	 * @return Returns a string containing this node's namespace prefix
	 * @throws XFlaimException
	 */
	public String getPrefix () throws XFlaimException
	{
		return _getPrefix( m_this, m_jdb.m_this);
	}

	/**
	 * Retrieves the namespace prefix ID for this node
	 * @return Returns a number containing this node's namespace prefix id
	 * @throws XFlaimException
	 */
	public int getPrefixId() throws XFlaimException
	{
		return( _getPrefixId( m_this, m_jdb.m_this));
	}
	
	/**
	 * Retrieves the encryption definition ID for this node
	 * @return Returns a number containing this node's encryption definition id
	 * @throws XFlaimException
	 */
	public int getEncDefId() throws XFlaimException
	{
		return( _getEncDefId( m_this, m_jdb.m_this));
	}
	
	/**
	 * Sets the namespace prefix for this node
	 * @param sPrefix The prefix that is to be set for this node
	 * @throws XFlaimException
	 */
	public void setPrefix(
		String	sPrefix) throws XFlaimException
	{
		_setPrefix( m_this, m_jdb.m_this, sPrefix);
	}
	
	/**
	 * Sets the namespace prefix for this node
	 * @param iPrefixId The prefix that is to be set for this node
	 * @throws XFlaimException
	 */
	public void setPrefixId(
		int		iPrefixId) throws XFlaimException
	{
		_setPrefixId( m_this, m_jdb.m_this, iPrefixId);
	}
	
	/**
	 * Retrieves the namespace URI that this node's name belongs to.
	 * @return Returns the namespace URI
	 * @throws XFlaimException
	 */
	public String getNamespaceURI() throws XFlaimException
	{
		return _getNamespaceURI( m_this, m_jdb.m_this);
	}
	
	/**
	 * Retrieves the name of this node, without the namespace prefix.
	 * @return Returns unprefixed element or attribute name.
	 * @throws XFlaimException
	 */
	public String getLocalName() throws XFlaimException
	{
		return _getLocalName( m_this, m_jdb.m_this);
	}
	
	/**
	 * Retrieves the fully qualified name (namespace prefix plus local
	 * name) for this element or attribute.
	 * @return Returns the fully qualified element or attribute name.
	 * @throws XFlaimException
	 */
	public String getQualifiedName() throws XFlaimException
	{
		return _getQualifiedName( m_this, m_jdb.m_this);
	}
	
	/**
	 * Retrieves the collection that this node is stored in.
	 * @return Returns the collection number.
	 * @throws XFlaimException
	 */
	public int getCollection() throws XFlaimException
	{
		return _getCollection( m_this, m_jdb.m_this);
	}

	/**
	 * Creates an annotation node and associates it with this node.
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return  Returns an instance of DOMNode
	 * @throws XFlaimException
	 */
	public DOMNode createAnnotation(
		DOMNode			ReusedNode) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.m_this
											  : 0;

		// See to comments in the finalize function for an explanation of this
		// synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _createAnnotation( m_this, m_jdb.m_this, lReusedNodeRef);
		}
			
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
			
		return NewNode;
	}

	/**
	 * Retrieves the annotation node assigned to this node.
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return returns the annotation node assigned to this node
	 * @throws XFlaimException
	 */
	public DOMNode getAnnotation(
		DOMNode			ReusedNode) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.m_this
											  : 0;
		
		// See to comments in the finalize function for an explanation of this
		// synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getAnnotation( m_this, m_jdb.m_this, lReusedNodeRef);
		}
			
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
			
		return NewNode;		
	}

	/**
	 * Retrieves the id of the annotation associated with this node.
	 * @return returns the id of the annotation assigned to this node
	 * @throws XFlaimException
	 */
	public long getAnnotationId() throws XFlaimException
	{
		return( _getAnnotationId( m_this, m_jdb.m_this));
	}

	/**
	 * Checks to see if this node has an annotation
	 * @return Returns true if the current node has an annotation
	 * @throws XFlaimException
	 */
	public boolean hasAnnotation() throws XFlaimException
	{
		return _hasAnnotation( m_this, m_jdb.m_this);
	}

	/**
	 * Returns the meta-value for the node.
	 * @return Returns meta-value for the node.
	 * @throws XFlaimException
	 */
	public long getMetaValue() throws XFlaimException
	{
		return( _getMetaValue( m_this, m_jdb.m_this));
	}
	
	/**
	 * Set the meta-value for the node.
	 * @param lValue Meta-value to set.
	 * @throws XFlaimException
	 */
	public void setMetaValue(
		long		lValue) throws XFlaimException
	{
		_setMetaValue( m_this, m_jdb.m_this, lValue);
	}
	
	/**
	 * Reassigns the object to "point" to a new F_DOMNode instance and a new
	 * Db.  Called by any of the member functions that take a ReusuedNode
	 * parameter.  Shouldn't be called by outsiders, so it's not public, but
	 * it must be callable for other instances of this class.  (It's also
	 * called by Db.getNode)
	 *
	 * NOTE:  This function does not result in a call to F_DOMNode::Release()
	 * because that is done by the native code when the F_DOMNode object is 
	 * reused.  Calling setRef() in any case except after a DOM node has been
	 * reused will result in a memory leak on the native side!
	*/
	void setRef(
		long	lDomNodeRef,
		Db		jdb)
	{
		m_this = lDomNodeRef;
		m_jdb = jdb;
	}

	long getRef()
	{
		return m_this;
	}
	
	Db getJdb()
	{
		return m_jdb;
	}
	
// PRIVATE NATIVE METHODS
	
	private native void _release(
		long		lThis);
		
	private native long _createNode(
		 long		lThis,
		 long		lDbRef,
		 int		iNodeType,
		 int		iNameId,
		 int		iInsertLoc,
		 long		lReusedNodeRef) throws XFlaimException;

	private native long _createChildElement(
		long		lThis,
		long		lDbRef,
		int		iChildElementNameId,
		boolean	bFirstChild,
		long		lReusedNodeRef) throws XFlaimException;
	
	private native void _deleteNode(
		long		lThis,
		long		lDbRef) throws XFlaimException;
		 
	private native void _deleteChildren(
		long		lThis,
		long		lDbRef) throws XFlaimException;
		
	private native int _getNodeType(
		long		lThis) throws XFlaimException;

	private native boolean _isDataLocalToNode(
		long		lThis,
		long		lDbRef) throws XFlaimException;

	private native long _createAttribute(
		long		lThis,
		long		lDbRef,
		int		iNameId,
		long		lReusedNodeRef) throws XFlaimException;

	private native long _getFirstAttribute(
		long		lThis,
		long		lDbRef,
		long		lReusedNodeRef) throws XFlaimException;

	private native long _getLastAttribute(
		long		lThis,
		long		lDbRef,
		long		lReusedNodeRef) throws XFlaimException;

	private native long _getAttribute(
		long		lThis,
		long		lDbRef,
		int		iAttributeId,
		long		lReusedNodeRef) throws XFlaimException;
	
	private native void _deleteAttribute(
		long		lThis,
		long		lDbRef,
		int		iAttributeId) throws XFlaimException;
		
	private native boolean _hasAttribute(
		long		lThis,
		long		lDbRef,
		int		iAttributeId) throws XFlaimException;

	private native boolean _hasAttributes(
		long		lThis,
		long		lDbRef) throws XFlaimException;
		
	private native boolean _hasNextSibling(
		long		lThis,
		long		lDbRef) throws XFlaimException;
	
	private native boolean _hasPreviousSibling(
		long		lThis,
		long		lDbRef) throws XFlaimException;
	
	private native boolean _hasChildren(
		long		lThis,
		long		lDbRef) throws XFlaimException;
	
	private native boolean _isNamespaceDecl(
		long		lThis,
		long		lDbRef) throws XFlaimException;

	private native long _getParentId(
		long		lThis,
		long		lDbRef) throws XFlaimException;

	private native long _getNodeId(
		long		lThis,
		long		lDbRef);
		
	private native long _getDocumentId(
		long		lThis,
		long		lDbRef) throws XFlaimException;

	private native long _getPrevSibId(
		long		lThis,
		long		lDbRef) throws XFlaimException;

	private native long _getNextSibId(
		long		lThis,
		long		lDbRef) throws XFlaimException;
		
	private native long _getFirstChildId(
		long		lThis,
		long		lDbRef) throws XFlaimException;

	private native long _getLastChildId(
		long		lThis,
		long		lDbRef) throws XFlaimException;

	private native int _getNameId(
		long		lThis,
		long		lDbRef) throws XFlaimException;
		
	private native void _setLong(
		long		lThis,
		long		lDbRef,
		long		lValue,
		int		iEncId) throws XFlaimException;

	private native void _setAttributeValueLong(
		long		lThis,
		long		lDbRef,
		int		iAttrNameId,
		long		lValue,
		int		iEncId) throws XFlaimException;

	private native void _setString(
		long		lThis,
		long		lDbRef,
		String	sValue,
		boolean	bLast,
		int		iEncId) throws XFlaimException;

	private native void _setAttributeValueString(
		long		lThis,
		long		lDbRef,
		int		iAttrNameId,
		String	sValue,
		int		iEncId) throws XFlaimException;

	private native void _setBinary(
		long		lThis,
		long		lDbRef,
		byte[]	Value,
		boolean	bLast,
		int		iEncId) throws XFlaimException;

	private native void _setAttributeValueBinary(
		long		lThis,
		long		lDbRef,
		int		iAttrNameId,
		byte[]	Value,
		int		iEncId) throws XFlaimException;

	private native long _getDataLength(
		long		lThis,
		long		lDbRef) throws XFlaimException;

	private native int _getDataType(
		long		lThis,
		long		lDbRef) throws XFlaimException;
		
	private native long _getLong(
		long		lThis,
		long		lDbRef) throws XFlaimException;

	private native long _getAttributeValueLong(
		long		lThis,
		long		lDbRef,
		int		iAttrNameId,
		boolean	bDefaultOk,
		long		lDefaultToUse) throws XFlaimException;

	private native String _getString(
		long		lThis,
		long		lDbRef,
		int		iStartPos,
		int		iNumChars) throws XFlaimException;
		
	private native String _getAttributeValueString(
		long		lThis,
		long		lDbRef,
		int		iAttrNameId) throws XFlaimException;
		
	private native int _getStringLen(
		long		lThis,
		long		lDbRef) throws XFlaimException;
		
	private native byte[] _getBinary(
		long		lThis,
		long		lDbRef,
		int		iStartPos,
		int		iNumBytes) throws XFlaimException; 

	private native byte[] _getAttributeValueBinary(
		long		lThis,
		long		lDbRef,
		int		iAttrNameId) throws XFlaimException;

	private native long _getDocumentNode(
		long		lThis,
		long		lDbRef,
		long		lReusedNodeRef) throws XFlaimException;

	private native long _getParentNode(
		long		lThis,
		long		lDbRef,
		long		lReusedNodeRef) throws XFlaimException;

	private native long _getFirstChild(
		long		lThis,
		long		lDbRef,
		long		lReusedNodeRef) throws XFlaimException;
		
	private native long _getLastChild(
		long		lThis,
		long		lDbRef,
		long		lReusedNodeRef) throws XFlaimException;

	private native long _getChild(
		long		lThis,
		long		lDbRef,
		int		iNodeType,
		long		lReusedNodeRef) throws XFlaimException;
		
	private native long _getChildElement(
		long		lThis,
		long		lDbRef,
		int		iNameId,
		long		lReusedNodeRef) throws XFlaimException;
		
	private native long _getSiblingElement(
		long		lThis,
		long		lDbRef,
		int		iNameId,
		boolean	bNext,
		long		lReusedNodeRef) throws XFlaimException;

	private native long _getAncestorElement(
		long		lThis,
		long		lDbRef,
		int		iNameId,
		long		lReusedNodeRef) throws XFlaimException;

	private native long _getDescendantElement(
		long		lThis,
		long		lDbRef,
		int		iNameId,
		long		lReusedNodeRef) throws XFlaimException;
		
	private native long _getPreviousSibling(
		long		lThis,
		long		lDbRef,
		long		lReusedNodeRef) throws XFlaimException;
			
	private native long _getNextSibling(
		long		lThis,
		long		lDbRef,
		long		lReusedNodeRef) throws XFlaimException;
		
	private native long _getPreviousDocument(
		long		lThis,
		long		lDbRef,
		long		lReusedNodeRef) throws XFlaimException;
		
	private native long _getNextDocument(
		long		lThis,
		long		lDbRef,
		long		lReusedNodeRef) throws XFlaimException;
		
	private native String _getPrefix(
		long		lThis,
		long		lDbRef) throws XFlaimException;

	private native int _getPrefixId(
		long		lThis,
		long		lDbRef) throws XFlaimException;

	private native int _getEncDefId(
		long		lThis,
		long		lDbRef) throws XFlaimException;

	private native void _setPrefix(
		long		lThis,
		long		lDbRef,
		String	sPrefix) throws XFlaimException;
		
	private native void _setPrefixId(
		long		lThis,
		long		lDbRef,
		int		iPrefixId) throws XFlaimException;

	private native String _getNamespaceURI(
		long		lThis,
		long		lDbRef) throws XFlaimException;
		
	private native String _getLocalName(
		long		lThis,
		long		lDbRef) throws XFlaimException;

	private native String _getQualifiedName(
		long		lThis,
		long		lDbRef) throws XFlaimException;
		
	private native int _getCollection(
		long		lThis,
		long		lDbRef) throws XFlaimException;
		
	private native long _createAnnotation(
		long		lThis,
		long		lDbRef,
		long		lReusedNodeRef) throws XFlaimException;

	private native long _getAnnotation(
		long		lThis,
		long		lDbRef,
		long		lReusedNodeRef) throws XFlaimException;

	private native long _getAnnotationId(
		long		lThis,
		long		lDbRef) throws XFlaimException;

	private native boolean _hasAnnotation(
		long		lThis,
		long		lDbRef) throws XFlaimException;

	private native long _getMetaValue(
		long		lThis,
		long		lDbRef) throws XFlaimException;
	
	private native void _setMetaValue(
		long		lThis,
		long		lDbRef,
		long		lValue) throws XFlaimException;

	private long	m_this;
	private Db		m_jdb;
}

