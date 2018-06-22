//------------------------------------------------------------------------------
// Desc:	Db Class
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
 * The Db class provides a number of methods that allow java applications to
 * access the XFlaim native environment, specifically, the IF_Db interface.
 */
public class Db 
{
	static
	{ 
		System.loadLibrary( "xflaim");
	}
	
	Db( 
		long			ref, 
		DbSystem 	dbSystem) throws XFlaimException
	{
		super();
		
		if( ref == 0)
		{
			throw new XFlaimException( -1, "No legal reference");
		}
		
		m_this = ref;

		if( dbSystem == null)
		{
			throw new XFlaimException( -1, "No legal dbSystem reference");
		}
		
		m_dbSystem = dbSystem;
	}

	/**
	 * Finalize method used to release native resources on garbage collection.
	 */	
	public void finalize()
	{
		close();
	}
	
	public long getThis()
	{
		return m_this;
	}
	
	/**
	 *  Closes the database.
	 */	
	public void close()
	{
		// Release the native pDb!
		
		if( m_this != 0)
		{
			_release( m_this);
			m_this = 0;
		}
		
		// Remove our reference to the dbSystem so it can be released.
		
		m_dbSystem = null;
	}

	/**
	 * Starts a transaction.
	 * 
	 * @param iTransactionType The type of transaction to start (read or
	 * write).  Should be one of the members of {@link
	 * xflaim.TransactionType TransactionType}.
	 * @param iMaxLockWait Maximum lock wait time.  Specifies the amount of
	 * time to wait for lock requests occuring during the transaction to be
	 * granted.  Valid values are 0 through 255 seconds.  Zero is used to
	 * specify no-wait locks.  255 specifies that there is no timeout.
	 * @param iFlags Should be a logical OR'd combination of the members of
	 * the memers of {@link xflaim.TransactionFlags
	 * TransactionFlags}.
	 * @throws XFlaimException
	 */
	public void transBegin(
		int			iTransactionType,
		int			iMaxLockWait,
		int			iFlags) throws XFlaimException
	{
		_transBegin( m_this, iTransactionType, iMaxLockWait, iFlags);
	}
	
	/**
	 * Starts a transaction.  Transaction will be of the same type and same
	 * snapshot as the passed in Db object.  The passed in Db object should
	 * be running a read transaction.
	 * 
	 * @param db Database whose transaction is to be copied.
	 * @throws XFlaimException
	 */
	public void transBegin(
		Db	db) throws XFlaimException
	{
		_transBegin( m_this, db.m_this);
	}

	/**
	 * Commits an existing transaction.  If no transaction is running, or the
	 * transaction commit fails, an XFlaimException exception will be thrown.
	 * @throws XFlaimException
	 */
	public void transCommit() throws XFlaimException
	{
		_transCommit( m_this);
	}
	
	/**
	 * Aborts an existing transaction.
	 * 
	 * @throws XFlaimException
	 */
	public void transAbort() throws XFlaimException
	{
		_transAbort( m_this);
	}
	
	/**
	 * Get the current transaction type.
	 * @return Returns the type of transaction.  Should be one of
	 * the members of {@link xflaim.TransactionType TransactionType}.
	 * @throws XFlaimException
	 */
	public int getTransType() throws XFlaimException
	{
		return( _getTransType( m_this));
	}
		
	/**
	 * Perform a checkpoint on the database.
	 * @param iTimeout lock wait time.  Specifies the amount of
	 * time to wait for database lock.  Valid values are 0 through 255 seconds.
	 * Zero is used to specify no-wait locks. 255 is used to specify that there
	 * is no timeout.
	 * @return Returns the type of transaction.  Should be one of
	 * the members of {@link xflaim.TransactionType TransactionType}.
	 * @throws XFlaimException
	 */
	public void doCheckpoint(
		int	iTimeout) throws XFlaimException
	{
		_doCheckpoint( m_this, iTimeout);
	}

	/**
	 * Lock the database.
	 * @param iLockType Type of lock being requested.  Should be one of the
	 * values in {@link xflaim.LockType LockType}.
	 * @param iPriority Priority of lock being requested.
	 * @param iTimeout lock wait time.  Specifies the amount of
	 * time to wait for database lock.  Valid values are 0 through 255 seconds.
	 * Zero is used to specify no-wait locks. 255 is used to specify that there
	 * is no timeout.
	 * @throws XFlaimException
	 */
	public void dbLock(
		int	iLockType,
		int	iPriority,
		int	iTimeout) throws XFlaimException
	{
		_dbLock( m_this, iLockType, iPriority, iTimeout);
	}
	
	/**
	 * Unlock the database.
	 * @throws XFlaimException
	 */
	public void dbUnlock() throws XFlaimException
	{
		_dbUnlock( m_this);
	}

	/**
	 * Get the type of database lock current held.
	 * @return Returns type of database lock currently held.  Should be one of the
	 * values in {@link xflaim.LockType LockType}.
	 * @throws XFlaimException
	 */
	public int getLockType() throws XFlaimException
	{
		return( _getLockType( m_this));
	}

	/**
	 * Determine if the database lock was implicitly obtained (i.e., obtained
	 * when transBegin was called as opposed to dbLock).
	 * @return Returns whether lock was obtained implicitly or explicitly.
	 * @throws XFlaimException
	 */
	public boolean getLockImplicit() throws XFlaimException
	{
		return( _getLockImplicit( m_this));
	}

	/**
	 * Returns the thread id of the thread that currently holds the database lock.
	 * @return Returns thread ID.
	 * @throws XFlaimException
	 */
	public int getLockThreadId() throws XFlaimException
	{
		return( _getLockThreadId( m_this));
	}
	
	/**
	 * Returns the number of threads that are currently waiting to obtain
	 * an exclusive database lock.
	 * @return Returns number of threads waiting for exclusive lock.
	 * @throws XFlaimException
	 */
	public int getLockNumExclQueued() throws XFlaimException
	{
		return( _getLockNumExclQueued( m_this));
	}
	
	/**
	 * Returns the number of threads that are currently waiting to obtain
	 * a shared database lock.
	 * @return Returns number of threads waiting for shared lock.
	 * @throws XFlaimException
	 */
	public int getLockNumSharedQueued() throws XFlaimException
	{
		return( _getLockNumSharedQueued( m_this));
	}
	
	/**
	 * Returns the number of threads that are currently waiting to obtain
	 * a database lock whose priority is >= iPriority.
	 * @param iPriority Priority to look for - a count of all waiting threads with a
	 * lock priority greater than or equal to this will be returned.
	 * @return Returns number of threads waiting for a database lock whose
	 * priority is >= iPriority.
	 * @throws XFlaimException
	 */
	public int getLockPriorityCount(
		int	iPriority) throws XFlaimException
	{
		return( _getLockPriorityCount( m_this, iPriority));
	}
	
	/**
	 * Suspend indexing on the specified index.
	 * @param iIndex Index to be suspended.
	 * @throws XFlaimException
	 */
	public void indexSuspend(
		int	iIndex) throws XFlaimException
	{
		_indexSuspend( m_this, iIndex);
	}
	
	/**
	 * Resume indexing on the specified index.
	 * @param iIndex Index to be resumed.
	 * @throws XFlaimException
	 */
	public void indexResume(
		int	iIndex) throws XFlaimException
	{
		_indexResume( m_this, iIndex);
	}
	
	/**
	 * This method provides a way to iterate through all of the indexes in the
	 * database.  It returns the index ID of the index that comes after the
	 * passed in index number.  The first index can be obtained by passing in a
	 * zero.
	 * @param iCurrIndex Current index number.  Index that comes after this one
	 * will be returned.
	 * @return Returns the index ID of the index that comes after iCurrIndex.
	 * @throws XFlaimException
	 */
	public int indexGetNext(
		int	iCurrIndex) throws XFlaimException
	{
		return( _indexGetNext( m_this, iCurrIndex));
	}

	/**
	 * Returns status information on an index in an {@link xflaim.IndexStatus IndexStatus}
	 * object.
	 * @param iIndex Index whose status is to be returned.
	 * @return Returns an  {@link xflaim.IndexStatus IndexStatus} object.
	 * @throws XFlaimException
	 */
	public IndexStatus indexStatus(
		int	iIndex) throws XFlaimException
	{
		return( _indexStatus( m_this, iIndex));
	}

	/**
	 * Return unused blocks back to the file system.
	 * @param iCount Maximum number of blocks to be returned.
	 * @return Returns the number of blocks that were actually returned to the
	 * file system.
	 * @throws XFlaimException
	 */
	public int reduceSize(
		int	iCount) throws XFlaimException
	{
		return( _reduceSize( m_this, iCount));
	}
	
	/**
	 * Lookup/retrieve keys in an index.
	 * @param iIndex The index that is being searched.
	 * @param searchKey The search key that is to be looked up.  NOTE: This
	 * parameter may be ignored, depending on the iFlags parameter.  See
	 * {@link xflaim.DataVector DataVector} for information on the DataVector class.
	 * @param iSearchFlags The search flags that direct how the next key will
	 * be determined.  This should be values from
	 * {@link xflaim.SearchFlags SearchFlags} that are ORed together.
	 * @param foundKey Key that was found is returned here.  This parameter may be
	 * used during subsequent calls to keyRetrieve.  The returned DataVector
	 * is passed in as this  parameter so that it may be reused, thus preventing
	 * the unnecessary accumulation of IF_DataVector objects in the C++ environment. 
	 * @throws XFlaimException
	 */
	public void keyRetrieve(
		int				iIndex,
		DataVector		searchKey,
		int				iSearchFlags,
		DataVector		foundKey) throws XFlaimException
	{
		long	lSearchKey = (searchKey == null ? 0 : searchKey.m_this);
		long	lFoundKey = (foundKey == null ? 0 : foundKey.m_this);
		
		_keyRetrieve( m_this, iIndex, lSearchKey, iSearchFlags, lFoundKey);
	}

	/**
	 * Creates a new document node. 
	 * @param iCollection The collection to store the new document in.
	 * @return Returns the {@link xflaim.DOMNode DOMNode} representing the new document.
	 * @throws XFlaimException
	 */
	 public DOMNode createDocument(
	 	int			iCollection) throws XFlaimException
	{
		long 		lNewDocRef;

		// See the comments in the DOMNode::finalize() function for an
		// explanation of this call synchronized call
		
		synchronized( this)
		{
	 		lNewDocRef =  _createDocument( m_this, iCollection);
		}
		
		return (new DOMNode( lNewDocRef, this));
	}
	
	/**
	 * Creates a new root element node. This is the root node of a document
	 * in the XFlaim database.
	 * @param iCollection The collection to store the new node in.
	 * @param iElementNameId Name of the element to be created.
	 * @return Returns the {@link xflaim.DOMNode DOMNode} representing the
	 * root element node.
	 * @throws XFlaimException
	 */
	public DOMNode createRootElement(
		int		iCollection,
		int		iElementNameId) throws XFlaimException
	{
		long 		lNewDocRef;
		
		// See the comments in the DOMNode::finalize() function for an
		// explanation of this call synchronized call
		
		synchronized( this)
		{
			lNewDocRef =  _createRootElement( m_this, iCollection, iElementNameId);
		}
		
		return (new DOMNode( lNewDocRef, this));
	}
	
	/**
	 * Retrieve the first document in a specified collection.
	 * @param iCollection - The collection from which to retrieve the 
	 * first document
	 * @param ReusedNode An existing {@link xflaim.DOMNode DOMNode} object
	 * can optionally be passed in, and it will be reused instead of a new
	 * object being allocated.
	 * @return - Returns a {@link xflaim.DOMNode DOMNode} which is the root node
	 * of the requested document.
	 * @throws XFlaimException
	 */
	public DOMNode getFirstDocument(
		int			iCollection,
		DOMNode		ReusedNode) throws XFlaimException
	 {
		DOMNode		newNode = null;
		long			lNewNodeRef = 0;
		long			lReusedNodeRef = 0;
		
		if (ReusedNode != null)
		{
			lReusedNodeRef = ReusedNode.getRef();
		}
		
		// See the comments in the DOMNode::finalize() function for an
		// explanation of this call synchronized call
		
		synchronized( this)
		{
			lNewNodeRef = _getFirstDocument( m_this, iCollection, lReusedNodeRef);
		}
	
		// If we got a reference to a native DOMNode back, let's 
		// create a new DOMNode.
		
		if (lNewNodeRef != 0)
		{
			if (ReusedNode != null)
			{
				ReusedNode.setRef( lNewNodeRef, this);
				newNode = ReusedNode;
			}
			else
			{
				newNode = new DOMNode( lNewNodeRef, this);
			}
		}
			
		return( newNode);
	}
 
	/**
	 * Retrieve the last document in a specified collection.
	 * @param iCollection - The collection from which to retrieve the 
	 * last document.
	 * @param ReusedNode An existing {@link xflaim.DOMNode DOMNode} object
	 * can optionally be passed in, and it will be reused instead of a new
	 * object being allocated.
	 * @return - Returns a {@link xflaim.DOMNode DOMNode} which is the root node
	 * of the requested document.
	 * @throws XFlaimException
	 */
	public DOMNode getLastDocument(
		int			iCollection,
		DOMNode		ReusedNode) throws XFlaimException
	 {
		DOMNode		newNode = null;
		long			lNewNodeRef = 0;
		long			lReusedNodeRef = 0;
		
		if (ReusedNode != null)
		{
			lReusedNodeRef = ReusedNode.getRef();
		}
		
		// See the comments in the DOMNode::finalize() function for an
		// explanation of this call synchronized call
		
		synchronized( this)
		{
			lNewNodeRef = _getLastDocument( m_this, iCollection, lReusedNodeRef);
		}
	
		// If we got a reference to a native DOMNode back, let's 
		// create a new DOMNode.
		
		if (lNewNodeRef != 0)
		{
			if (ReusedNode != null)
			{
				ReusedNode.setRef( lNewNodeRef, this);
				newNode = ReusedNode;
			}
			else
			{
				newNode = new DOMNode( lNewNodeRef, this);
			}
		}
			
		return( newNode);
	}
 
	/**
	 * Retrieve a document based on document ID.
	 * @param iCollection The collection from which to retrieve the 
	 * last document.
	 * @param iSearchFlags Flags that determine what document should be
	 * returned.  Should be ORed flags from {@link xflaim.SearchFlags SearchFlags}.
	 * @param lDocumentId Document ID to search for.  iSearchFlags determines
	 * how this parameter is to be used.
	 * @param ReusedNode An existing {@link xflaim.DOMNode DOMNode} object
	 * can optionally be passed in, and it will be reused instead of a new
	 * object being allocated.
	 * @return - Returns a {@link xflaim.DOMNode DOMNode} which is the root node
	 * of the requested document.
	 * @throws XFlaimException
	 */
	public DOMNode getDocument(
		int			iCollection,
		int			iSearchFlags,
		long			lDocumentId,
		DOMNode		ReusedNode) throws XFlaimException
	 {
		DOMNode		newNode = null;
		long			lNewNodeRef = 0;
		long			lReusedNodeRef = 0;
		
		if (ReusedNode != null)
		{
			lReusedNodeRef = ReusedNode.getRef();
		}
		
		// See the comments in the DOMNode::finalize() function for an
		// explanation of this call synchronized call
		
		synchronized( this)
		{
			lNewNodeRef = _getDocument( m_this, iCollection, iSearchFlags,
									lDocumentId, lReusedNodeRef);
		}
	
		// If we got a reference to a native DOMNode back, let's 
		// create a new DOMNode.
		
		if (lNewNodeRef != 0)
		{
			if (ReusedNode != null)
			{
				ReusedNode.setRef( lNewNodeRef, this);
				newNode = ReusedNode;
			}
			else
			{
				newNode = new DOMNode( lNewNodeRef, this);
			}
		}
			
		return( newNode);
	}
	
	/**
	 * Indicate that modifications to a document are "done".  This allows
	 * XFLAIM to process the document as needed.
	 * @param iCollection The collection the document belongs to.
	 * @param lDocumentId Document ID of document that is "done".
	 * @throws XFlaimException
	 */
	public void documentDone(
		int			iCollection,
		long			lDocumentId) throws XFlaimException
	{
		_documentDone( m_this, iCollection, lDocumentId);
	}
	
	/**
	 * Indicate that modifications to a document are "done".  This allows
	 * XFLAIM to process the document as needed.
	 * @param domNode The {@link xflaim.DOMNode DOM node} that is the root node
	 * of the document that is "done".
	 * @throws XFlaimException
	 */
	public void documentDone(
		DOMNode		domNode) throws XFlaimException
	{
		_documentDone( m_this, domNode.getThis());
	}
	
	/**
	 * Creates a new element definition in the dictionary.
	 * @param sNamespaceURI The namespace URI that this definition should be
	 * created in.  If null, the default namespace will be used.
	 * @param sElementName The name of the element.
	 * @param iDataType The data type for instances of this element.  Should be
	 * one of the constants listed in {@link xflaim.FlmDataType FlmDataType}.
	 * @param iRequestedId If non-zero, then xflaim will try to use this
	 * number as the name ID of the new definition.
	 * @return Returns the name ID of the new definition.
	 * @throws XFlaimException
	 */
	public int createElementDef(
		String		sNamespaceURI,
		String		sElementName,
		int			iDataType,
		int			iRequestedId) throws XFlaimException
		
	{
		int	iNewNameId;
		
		// See the comments in the DOMNode::finalize() function for an
		// explanation of this call synchronized call
		
		synchronized( this)
		{
			iNewNameId = _createElementDef( m_this, sNamespaceURI,
											sElementName, iDataType,
											iRequestedId);
		}
		
		return( iNewNameId);
	}
	
	/**
	 * Create a "unique" element definition - i.e., an element definition whose
	 * child elements must all be unique.
	 * @param sNamespaceURI The namespace URI for the element.
	 * @param sElementName The name of the element.
	 * @param iRequestedId If non-zero, then xflaim will try to use this
	 * number as the name ID of the new definition.
	 * @return Returns the name ID of the element.
	 * @throws XFlaimException
	 */
	public int createUniqueElmDef(
		String		sNamespaceURI,
		String		sElementName,
		int			iRequestedId) throws XFlaimException
		
	{
		int	iNewNameId;
		
		// See the comments in the DOMNode::finalize() function for an
		// explanation of this call synchronized call
		
		synchronized( this)
		{
			iNewNameId = _createUniqueElmDef( m_this, sNamespaceURI,
											sElementName, iRequestedId);
		}
		
		return( iNewNameId);
	}
	
	/**
	 * Gets the name id for a particular element name.
	 * @param sNamespaceURI The namespace URI for the element.
	 * @param sElementName The name of the element.
	 * @return Returns the name ID of the element.
	 * @throws XFlaimException
	 */
	public int getElementNameId(
		String		sNamespaceURI,
		String		sElementName) throws XFlaimException
	{
		int	iNameId;
		
		// See the comments in the DOMNode::finalize() function for an
		// explanation of this call synchronized call
		
		synchronized( this)
		{
			iNameId = _getElementNameId( m_this, sNamespaceURI,
											sElementName);
		}
		
		return( iNameId);
	}
	
	/**
	 * Creates a new attribute definition in the dictionary.
	 * @param sNamespaceURI The namespace URI that this definition should be
	 * created in.  If null, the default namespace will be used.
	 * @param sAttributeName The name of the attribute.
	 * @param iDataType The data type for instances of this attribute.  Should be
	 * one of the constants listed in {@link xflaim.FlmDataType FlmDataType}.
	 * @param iRequestedId If non-zero, then xflaim will try to use this
	 * number as the name ID of the new definition.
	 * @return Returns the name ID of the new definition.
	 * @throws XFlaimException
	 */
	public int createAttributeDef(
		String		sNamespaceURI,
		String		sAttributeName,
		int			iDataType,
		int			iRequestedId) throws XFlaimException
		
	{
		int	iNewNameId;
		
		// See the comments in the DOMNode::finalize() function for an
		// explanation of this call synchronized call
		
		synchronized( this)
		{
			iNewNameId = _createAttributeDef( m_this, sNamespaceURI,
											sAttributeName, iDataType,
											iRequestedId);
		}
		
		return( iNewNameId);
	}

	/**
	 * Gets the name id for a particular attribute name.
	 * @param sNamespaceURI The namespace URI for the attribute.
	 * @param sAttributeName The name of the attribute.
	 * @return Returns the name ID of the attribute.
	 * @throws XFlaimException
	 */
	public int getAttributeNameId(
		String		sNamespaceURI,
		String		sAttributeName) throws XFlaimException
	{
		int	iNameId;
		
		// See the comments in the DOMNode::finalize() function for an
		// explanation of this call synchronized call
		
		synchronized( this)
		{
			iNameId = _getAttributeNameId( m_this, sNamespaceURI,
											sAttributeName);
		}
		
		return( iNameId);
	}
	
	/**
	 * Creates a new prefix definition in the dictionary.
	 * @param sPrefixName The name of the prefix.
	 * @param iRequestedId If non-zero, then xflaim will try to use this
	 * number as the name ID of the new definition.
	 * @return Returns the name ID of the new definition.
	 * @throws XFlaimException
	 */
	public int createPrefixDef(
		String		sPrefixName,
		int			iRequestedId) throws XFlaimException
		
	{
		int	iNewNameId;
		
		// See the comments in the DOMNode::finalize() function for an
		// explanation of this call synchronized call
		
		synchronized( this)
		{
			iNewNameId = _createPrefixDef( m_this, sPrefixName, iRequestedId);
		}
		
		return( iNewNameId);
	}

	/**
	 * Gets the name id for a particular prefix name.
	 * @param sPrefixName The name of the prefix.
	 * @return Returns the name ID of the prefix.
	 * @throws XFlaimException
	 */
	public int getPrefixId(
		String		sPrefixName) throws XFlaimException
	{
		int	iNameId;
		
		// See the comments in the DOMNode::finalize() function for an
		// explanation of this call synchronized call
		
		synchronized( this)
		{
			iNameId = _getPrefixId( m_this, sPrefixName);
		}
		
		return( iNameId);
	}
	
	/**
	 * Creates a new encryption definition in the dictionary.
	 * @param sEncType Type of encryption key.
	 * @param sEncName The name of the encryption definition.
	 * @param iKeySize Size of the encryption key.
	 * @param iRequestedId If non-zero, then xflaim will try to use this
	 * number as the name ID of the new definition.
	 * @return Returns the name ID of the new definition.
	 * @throws XFlaimException
	 */
	public int createEncDef(
		String		sEncType,
		String		sEncName,
		int			iKeySize,
		int			iRequestedId) throws XFlaimException
		
	{
		int	iNewNameId;
		
		// See the comments in the DOMNode::finalize() function for an
		// explanation of this call synchronized call
		
		synchronized( this)
		{
			iNewNameId = _createEncDef( m_this, sEncType, sEncName, iKeySize,
									iRequestedId);
		}
		
		return( iNewNameId);
	}

	/**
	 * Gets the name id for a particular encryption definition name.
	 * @param sEncName The name of the encryption definition.
	 * @return Returns the name ID of the encryption definition.
	 * @throws XFlaimException
	 */
	public int getEncDefId(
		String		sEncName) throws XFlaimException
	{
		int	iNameId;
		
		// See the comments in the DOMNode::finalize() function for an
		// explanation of this call synchronized call
		
		synchronized( this)
		{
			iNameId = _getEncDefId( m_this, sEncName);
		}
		
		return( iNameId);
	}
	
	/**
	 * Creates a new collection in the dictionary.
	 * @param sCollectionName The name of the collection.
	 * @param iEncNumber ID of the encryption definition that should be used
	 * to encrypt this collection.  Zero means the collection will not be encrypted.
	 * @param iRequestedId If non-zero, then xflaim will try to use this
	 * number as the name ID of the new definition.
	 * @return Returns the name ID of the new definition.
	 * @throws XFlaimException
	 */
	public int createCollectionDef(
		String		sCollectionName,
		int			iEncNumber,
		int			iRequestedId) throws XFlaimException
		
	{
		int	iNewNameId;
		
		// See the comments in the DOMNode::finalize() function for an
		// explanation of this call synchronized call
		
		synchronized( this)
		{
			iNewNameId = _createCollectionDef( m_this, sCollectionName, iEncNumber,
									iRequestedId);
		}
		
		return( iNewNameId);
	}

	/**
	 * Gets the collection number for a particular collection name.
	 * @param sCollectionName The name of the collection.
	 * @return Returns the number of the collection.
	 * @throws XFlaimException
	 */
	public int getCollectionNumber(
		String		sCollectionName) throws XFlaimException
	{
		int	iNameId;
		
		// See the comments in the DOMNode::finalize() function for an
		// explanation of this call synchronized call
		
		synchronized( this)
		{
			iNameId = _getCollectionNumber( m_this, sCollectionName);
		}
		
		return( iNameId);
	}
	
	/**
	 * Gets the index number for a particular index name.
	 * @param sIndexName The name of the index.
	 * @return Returns the number of the index.
	 * @throws XFlaimException
	 */
	public int getIndexNumber(
		String		sIndexName) throws XFlaimException
	{
		int	iNameId;
		
		// See the comments in the DOMNode::finalize() function for an
		// explanation of this call synchronized call
		
		synchronized( this)
		{
			iNameId = _getIndexNumber( m_this, sIndexName);
		}
		
		return( iNameId);
	}
	
	/**
	 * Retrieve a dictionary definition document.  If found, the root node of
	 * the document is returned.
	 * @param iDictType The type of dictionary definition being retrieved.  It
	 * should be one of a {@link xflaim.DictType DictType}.
	 * @param iDictNumber The number the dictionary definition being retrieved.
	 * @param ReusedNode An existing {@link xflaim.DOMNode DOMNode} object
	 * can optionally be passed in, and it will be reused instead of a new
	 * object being allocated.
	 * @return Returns the root {@link xflaim.DOMNode DOM node} of the dictionary
	 * definition document.
	 * @throws XFlaimException
	 */
	public DOMNode getDictionaryDef(
		int			iDictType,
		int			iDictNumber,
		DOMNode		ReusedNode) throws XFlaimException
	 {
		DOMNode		jNode = null;
		long			lNewNodeRef = 0;
		long			lReusedNodeRef = 0;
		
		if (ReusedNode != null)
		{
			lReusedNodeRef = ReusedNode.getRef();
		}
		
		// See the comments in the DOMNode::finalize() function for an
		// explanation of this call synchronized call
		
		synchronized( this)
		{
			lNewNodeRef = _getDictionaryDef( m_this, iDictType, iDictNumber,
										lReusedNodeRef);
		}
	
		// If we got a reference to a native DOMNode back, let's 
		// create a new DOMNode.
		
		if (lNewNodeRef != 0)
		{
			if (ReusedNode != null)
			{
				ReusedNode.setRef( lNewNodeRef, this);
				jNode = ReusedNode;
			}
			else
			{
				jNode = new DOMNode( lNewNodeRef, this);
			}
		}
			
		return( jNode);
	}
	
	/**
	 * Get a dictionary definition's name.
	 * @param iDictType The type of dictionary definition whose name is to be
	 * returned.  It should be one of a {@link xflaim.DictType DictType}.
	 * @param iDictNumber The number of the dictionary definition.
	 * @return Returns the name of the dictionary definition.
	 * @throws XFlaimException
	 */
 	public String getDictionaryName(
 		int	iDictType,
		int	iDictNumber) throws XFlaimException
	{
		return( _getDictionaryName( m_this, iDictType, iDictNumber));
	}
 
	/**
	 * Get an element definition's namespace.
	 * @param iDictNumber The number of the element definition.
	 * @return Returns the namespace for the element definition.
	 * @throws XFlaimException
	 */
	public String getElementNamespace(
		int	iDictNumber) throws XFlaimException
	{
		return( _getElementNamespace( m_this, iDictNumber));
	}
		
	/**
	 * Get an attribute definition's namespace.
	 * @param iDictNumber The number of the attribute definition.
	 * @return Returns the namespace for the attribute definition.
	 * @throws XFlaimException
	 */
	public String getAttributeNamespace(
		int	iDictNumber) throws XFlaimException
	{
		return( _getAttributeNamespace( m_this, iDictNumber));
	}
		
	/**
	 * Retrieves the specified node from the specified collection.
	 * @param iCollection The collection where the node is stored.
	 * @param lNodeId The ID number of the node to be retrieved.
	 * @param ReusedNode An existing instance of {@link xflaim.DOMNode DOMNode} who's
	 * contents will be replaced with that of the new node.  If null, a
	 * new instance will be allocated.
	 * @return Returns a {@link xflaim.DOMNode DOMNode} representing the retrieved node.
	 * @throws XFlaimException
	 */
	public DOMNode getNode(
		int			iCollection,
		long			lNodeId,
		DOMNode		ReusedNode) throws XFlaimException
		
	{
		long			lReusedNodeRef = 0;
		DOMNode		NewNode = null;
		long			lNewNodeRef = 0;
		
		if (ReusedNode != null)
		{
			lReusedNodeRef = ReusedNode.getRef();
		}
		
		// See the comments in DOMNode::finalize() for an explanation
		// of this synchronized call
		
		synchronized( this)
		{
			lNewNodeRef = _getNode( m_this, iCollection, lNodeId, lReusedNodeRef);
		}
		
		if (lNewNodeRef != 0)
		{
			if (ReusedNode == null)
			{
				NewNode = new DOMNode(lNewNodeRef, this);
			}
			else
			{
				NewNode=ReusedNode;
				NewNode.setRef( lNewNodeRef, this);
			}
		}
		
		return( NewNode);		
	}

	/**
	 * Retrieves the specified attribute node from the specified collection.
	 * @param iCollection The collection where the attribute is stored.
	 * @param lElementNodeId The ID number of the element node that contains
	 * the attribute to be retrieved.
	 * @param iAttrNameId The attribute id of the attribute to be retrieved.
	 * @param ReusedNode An existing instance of {@link xflaim.DOMNode DOMNode} who's
	 * contents will be replaced with that of the new node.  If null, a
	 * new instance will be allocated.
	 * @return Returns a {@link xflaim.DOMNode DOMNode} representing the retrieved node.
	 * @throws XFlaimException
	 */
	public DOMNode getAttribute(
		int			iCollection,
		long			lElementNodeId,
		int			iAttrNameId,
		DOMNode		ReusedNode) throws XFlaimException
	{
		long			lReusedNodeRef = 0;
		long			lNewNodeRef = 0;
		DOMNode		NewNode = null;
		
		if (ReusedNode != null)
		{
			lReusedNodeRef = ReusedNode.getRef();
		}
		
		// See the comments in DOMNode::finalize() for an explanation
		// of this synchronized call
		
		synchronized( this)
		{
			lNewNodeRef = _getAttribute( m_this, iCollection, lElementNodeId,
										iAttrNameId, lReusedNodeRef);
		}
		
		if (lNewNodeRef != 0)
		{
			if (ReusedNode == null)
			{
				NewNode = new DOMNode(lNewNodeRef, this);
			}
			else
			{
				NewNode=ReusedNode;
				NewNode.setRef( lNewNodeRef, this);
			}
		}
		
		return( NewNode);		
	}

	/**
	 * Returns the data type that was specified for a particular dictionary
	 * definition.  NOTE: This really only applies to element and attribute
	 * definitions.
	 * @param iDictType The type of dictionary definition whose data type is to be
	 * returned.  It should be one of a {@link xflaim.DictType DictType}.
	 * @param iDictNumber The number of the dictionary definition.
	 * @return Returns the dictionary definition's data type.
	 * @throws XFlaimException
	 */
	public int getDataType(
		int	iDictType,
		int	iDictNumber) throws XFlaimException
	{
		return( _getDataType( m_this, iDictType, iDictNumber));
	}

	/**
	 * Sets up XFlaim to perform a backup operation
	 * @param iBackupType The type of backup to perform.  Must be one of the
	 * members of {@link xflaim.FlmBackupType FlmBackupType}.
	 * @param iTransType The type of transaction in which the backup operation
	 * will take place.   Must be one of the members of
	 * {@link xflaim.TransactionType TransactionType}. 
	 * @param iMaxLockWait  Maximum lock wait time.  Specifies the amount of
	 * time to wait for lock requests occuring during the backup operation to
	 * be granted.  Valid values are 0 through 255 seconds.  Zero is used to
	 * specify no-wait locks. 255 specifies no timeout.
	 * @param ReusedBackup Optional.  An existing instance of
	 * {@link xflaim.Backup Backup} that will be reset with the new settings.
	 * If null, a new instance will be allocated.
	 * @return Returns an instance of {@link xflaim.Backup Backup} configured to
	 * perform the requested backup operation.
	 * @throws XFlaimException
	 */
	public Backup backupBegin(
		int			iBackupType,
		int			iTransType,
		int			iMaxLockWait,
		Backup		ReusedBackup) throws XFlaimException
	{
		long 			lReusedRef = 0;
		long 			lNewRef = 0;
		Backup 		NewBackup;
		
		if (ReusedBackup != null)
		{
			lReusedRef = ReusedBackup.getRef();
		}

		// See to comments in the finalize function for an explanation of this
		// synchronized call
		
		synchronized( this)
		{
			lNewRef = _backupBegin( m_this, iBackupType, iTransType,
									iMaxLockWait, lReusedRef);
		}
		
		if (ReusedBackup == null)
		{
			NewBackup = new Backup(lNewRef, this);
		}
		else
		{
			NewBackup = ReusedBackup;
			NewBackup.setRef( lNewRef, this);
		}
		
		return( NewBackup);
	}

	/**
	 * Imports an XML document into the XFlaim database.  The import requires
	 * an update transaction ({@link xflaim.TransactionType TransactionType}.UPDATE_TRANS).
	 * If the document cannot be imported, an XFlaimEXception exception will be thrown.
	 * @param istream Input stream for importing the document.  Could represent
	 * a file or a buffer.
	 * @param iCollection Collection the document is to be imported into.
	 * @throws XFlaimException
	 */
	public ImportStats Import(
		IStream		istream,
		int			iCollection) throws XFlaimException
	{
		return( _import( m_this, istream.getThis(), iCollection, 0,
						InsertLoc.XFLM_LAST_CHILD));
	}
	
	/**
	 * Imports an XML document into the XFlaim database.  The import requires
	 * an update transaction ({@link xflaim.TransactionType TransactionType}.UPDATE_TRANS).
	 * If the document cannot be imported, an XFlaimEXception exception will be thrown.
	 * @param istream Input stream for importing the document.  Could represent
	 * a file or a buffer.
	 * @param iCollection Collection the document is to be imported into.
	 * @param nodeToLinkTo Node the imported XML should be linked to.
	 * @param iInsertLoc Specifies how the imported document should be linked to
	 * the nodeToLinkTo.  Should be one of the members of {@link
	 * xflaim.InsertLoc InsertLoc}.
	 * @return Returns an {@link xflaim.ImportStats ImportStats} object which holds
	 * statistics about what was imported.
	 * @throws XFlaimException
	 */
	public ImportStats Import(
		IStream		istream,
		int			iCollection,
		DOMNode		nodeToLinkTo,
		int			iInsertLoc) throws XFlaimException
	{
		if (nodeToLinkTo == null)
		{
			return( _import( m_this, istream.getThis(), iCollection, 0, iInsertLoc));
		}
		else
		{
			return( _import( m_this, istream.getThis(), iCollection,
							nodeToLinkTo.getThis(), iInsertLoc));
		}
	}
	
	/**
	 * Change a dictionary definition's state.  This routine is used to determine if
	 * the dictionary item can be deleted.  It may also be used to force the
	 * definition to be deleted - once the database has determined that the
	 * definition is not in use anywhere.  This should only be used for
	 * element definitions and attribute definitions definitions.
	 * @param iDictType Type of dictionary definition whose state is being
	 * changed.  Should be either {@link DictType DictType}.ELEMENT_DEF or
	 * {@link DictType DictType}.ATTRIBUTE_DEF.
	 * @param iDictNum Number of element or attribute definition whose state
	 * is to be changed.
	 * @param sState State the definition is to be changed to.  Must be
	 * "checking", "purge", or "active".
	 * @throws XFlaimException
	 */
	public void changeItemState(
		int				iDictType,
		int				iDictNum,
		String			sState) throws XFlaimException
	{
		_changeItemState( m_this, iDictType, iDictNum, sState);
	}

	/**
	 * Get the name of a roll-forward log file.
	 * @param iFileNum Roll-forward log file number whose name is to be
	 * returned.
	 * @param bBaseOnly If true, only the base name of the file will be returned.
	 * Otherwise, the entire path will be returned.
	 * @return Name of the file.
	 * @throws XFlaimException
	 */
	public String getRflFileName(
		int				iFileNum,
		boolean			bBaseOnly) throws XFlaimException
	{
		return( _getRflFileName( m_this, iFileNum, bBaseOnly));
	}
		
	/**
	 * Set the next node ID for a collection.  This will be the node ID for
	 * the next node that is created in the collection.  NOTE: The node ID must
	 * be greater than or equal to the current next node ID that is already
	 * set for the collection.  Otherwise, it is ignored.
	 * @param iCollection Collection whose next node ID is to be set.
	 * @param lNextNodeId Next node ID for the collection.
	 * @throws XFlaimException
	 */
	public void setNextNodeId(
		int				iCollection,
		long				lNextNodeId) throws XFlaimException
	{
		_setNextNodeId( m_this, iCollection, lNextNodeId);
	}

	/**
	 * Set the next dictionary number that is to be assigned for a particular
	 * type if dictionary definition.  The specified "next dictionary number"
	 * must be greater than the current "next dictionary number".  Otherwise,
	 * no action is taken.
	 * @param iDictType  Type of dictionary definition whose "next dictionary
	 * number" is to be changed.  Should be a valid {@link xflaim.DictType DictType}.
	 * @param iNextDictNumber Next dictionary number.
	 * @throws XFlaimException
	 */
	public void setNextDictNum(
		int	iDictType,
		int	iNextDictNumber) throws XFlaimException
	{
		_setNextDictNum( m_this, iDictType, iNextDictNumber);
	}
	
	/**
	 * Specify whether the roll-forward log should keep or not keep RFL files.
	 * @param bKeep Flag specifying whether to keep or not keep RFL files.
	 * @throws XFlaimException
	 */
	public void setRflKeepFilesFlag(
		boolean	bKeep) throws XFlaimException
	{
		_setRflKeepFilesFlag( m_this, bKeep);
	}
		
	/**
	 * Determine whether or not the roll-forward log files are being kept.
	 * @return Returns true if RFL files are being kept, false otherwise.
	 * @throws XFlaimException
	 */
	public boolean getRflKeepFlag() throws XFlaimException
	{
		return( _getRflKeepFlag( m_this));
	}
	
	/**
	 * Set the RFL directory.
	 * @param sRflDir Name of RFL directory.
	 * @throws XFlaimException
	 */
	public void setRflDir(
		String	sRflDir) throws XFlaimException
	{
		_setRflDir( m_this, sRflDir);
	}
		
	/**
	 * Get the current RFL directory.
	 * @return Returns the current RFL directory name.
	 * @throws XFlaimException
	 */
	public String getRflDir() throws XFlaimException
	{
		return( _getRflDir( m_this));
	}
	
	/**
	 * Get the current RFL file number.
	 * @return Returns the current RFL file number.
	 * @throws XFlaimException
	 */
	public int getRflFileNum() throws XFlaimException
	{
		return( _getRflFileNum( m_this));
	}

	/**
	 * Get the highest RFL file number that is no longer in use by XFLAIM.
	 * This RFL file can be removed from the system if needed.
	 * @return Returns the highest RFL file number that is no longer in use.
	 * @throws XFlaimException
	 */
	public int getHighestNotUsedRflFileNum() throws XFlaimException
	{
		return( _getHighestNotUsedRflFileNum( m_this));
	}

	/**
	 * Set size limits for RFL files.
	 * @param iMinRflSize Minimum RFL file size.  Database will roll to the
	 * next RFL file when the current RFL file reaches this size.  If possible
	 * it will complete the current transaction before rolling to the next file.
	 * @param iMaxRflSize Maximum RFL file size.  Database will not allow an
	 * RFL file to exceed this size.  Even if it is in the middle of a
	 * transaction, it will roll to the next RFL file before this size is allowed
	 * to be exceeded.  Thus, the database first looks for an opportunity to
	 * roll to the next file when the RFL file exceeds iMinRflSize.  If it can
	 * fit the current transaction in without exceeded iMaxRflSize, it will do
	 * so and then roll to the next file.  Otherwise, it will roll to the next
	 * file before iMaxRflSize is exceeded.
	 * @throws XFlaimException
	 */
	public void setRflFileSizeLimits(
		int	iMinRflSize,
		int	iMaxRflSize) throws XFlaimException
	{
		_setRflFileSizeLimits( m_this, iMinRflSize, iMaxRflSize);
	}

	/**
	 * Get the minimum RFL file size.  This is the minimum size an RFL file
	 * must reach before rolling to the next RFL file.
	 * @return Returns minimum RFL file size.
	 * @throws XFlaimException
	 */
	public int getMinRflFileSize() throws XFlaimException
	{
		return( _getMinRflFileSize( m_this));
	}
	
	/**
	 * Get the maximum RFL file size.  This is the maximum size an RFL file
	 * is allowed to grow to.  When the current RFL file exceeds the minimum
	 * RFL file size, the database will attempt to fit the rest of the
	 * transaction in the current file.  If the transaction completes before
	 * the current RFL file grows larger than the maximum RFL file size,
	 * the database will roll to the next RFL file.  However, if the current transaction
	 * would cause the RFL file to grow larger than the maximum RFL file size,
	 * the database will roll to the next file before the transaction completes,
	 * and the transaction will be split across multiple RFL files.
	 * @return Returns maximum RFL file size.
	 * @throws XFlaimException
	 */
	public int getMaxRflFileSize() throws XFlaimException
	{
		return( _getMaxRflFileSize( m_this));
	}

	/**
	 * Force the database to roll to the next RFL file.
	 * @throws XFlaimException
	 */
	public void rflRollToNextFile() throws XFlaimException
	{
		_rflRollToNextFile( m_this);
	}

	/**
	 * Specify whether the roll-forward log should keep or not keep aborted
	 * transactions.
	 * @param bKeep Flag specifying whether to keep or not keep aborted
	 * transactions.
	 * @throws XFlaimException
	 */
	public void setKeepAbortedTransInRflFlag(
		boolean	bKeep) throws XFlaimException
	{
		_setKeepAbortedTransInRflFlag( m_this, bKeep);
	}

	/**
	 * Determine whether or not the roll-forward log is keeping aborted
	 * transactions.
	 * @return Returns true if aborted transactions are being kept, false otherwise.
	 * @throws XFlaimException
	 */
	public boolean getKeepAbortedTransInRflFlag() throws XFlaimException
	{
		return( _getKeepAbortedTransInRflFlag( m_this));
	}

	/**
	 * Specify whether the roll-forward log should automatically turn off the
	 * keeping of RFL files if the RFL volume fills up.
	 * @param bAutoTurnOff Flag specifying whether to automatically turn off the
	 * keeping of RFL files if the RFL volume fills up.
	 * @throws XFlaimException
	 */
	public void setAutoTurnOffKeepRflFlag(
		boolean	bAutoTurnOff) throws XFlaimException
	{
		_setAutoTurnOffKeepRflFlag( m_this, bAutoTurnOff);
	}

	/**
	 * Determine whether or not keeping of RFL files will automatically be
	 * turned off if the RFL volume fills up.
	 * @return Returns true if the keeping of RFL files will automatically be
	 * turned off when the RFL volume fills up, false otherwise.
	 * @throws XFlaimException
	 */
	public boolean getAutoTurnOffKeepRflFlag() throws XFlaimException
	{
		return( _getAutoTurnOffKeepRflFlag( m_this));
	}

	/**
	 * Set the file extend size for the database.  This size specifies how much
	 * to extend a database file when it needs to be extended.
	 * @param iFileExtendSize  File extend size.
	 * @throws XFlaimException
	 */
	public void setFileExtendSize(
		int	iFileExtendSize) throws XFlaimException
	{
		_setFileExtendSize( m_this, iFileExtendSize);
	}

	/**
	 * Get the file extend size for the database.
	 * @return Returns file extend size.
	 * @throws XFlaimException
	 */
	public int getFileExtendSize() throws XFlaimException
	{
		return( _getFileExtendSize( m_this));
	}
	
	/**
	 * Get the database version for the database.  This is the version of the
	 * database, not the code.
	 * @return Returns database version.
	 * @throws XFlaimException
	 */
	public int getDbVersion() throws XFlaimException
	{
		return( _getDbVersion( m_this));
	}

	/**
	 * Get the database block size.
	 * @return Returns database block size.
	 * @throws XFlaimException
	 */
	public int getBlockSize() throws XFlaimException
	{
		return( _getBlockSize( m_this));
	}

	/**
	 * Get the database default language.
	 * @return Returns database default language.
	 * @throws XFlaimException
	 */
	public int getDefaultLanguage() throws XFlaimException
	{
		return( _getDefaultLanguage( m_this));
	}
	
	/**
	 * Get the database's current transaction ID.  If no transaction is
	 * currently running, but this Db object has an exclusive lock on the database,
	 * the transaction ID of the last committed transaction will be returned.
	 * If no transaction is running, and this Db object does not have an
	 * exclusive lock on the database, zero is returned.
	 * @return Returns transaction ID.
	 * @throws XFlaimException
	 */
	public long getTransID() throws XFlaimException
	{
		return( _getTransID( m_this));
	}

	/**
	 * Get the name of the database's control file (e.g.&nbsp;mystuff.db).
	 * @return Returns control file name.
	 * @throws XFlaimException
	 */
	public String getDbControlFileName() throws XFlaimException
	{
		return( _getDbControlFileName( m_this));
	}
	
	/**
	 * Get the transaction ID of the last backup that was taken on the database.
	 * @return Returns last backup transaction ID.
	 * @throws XFlaimException
	 */
	public long getLastBackupTransID() throws XFlaimException
	{
		return( _getLastBackupTransID( m_this));
	}

	/**
	 * Get the number of blocks that have changed since the last backup was
	 * taken.
	 * @return Returns number of blocks that have changed.
	 * @throws XFlaimException
	 */
	public int getBlocksChangedSinceBackup() throws XFlaimException
	{
		return( _getBlocksChangedSinceBackup( m_this));
	}

	/**
	 * Get the next incremental backup sequence number for the database.
	 * @return Returns next incremental backup sequence number.
	 * @throws XFlaimException
	 */
	public int getNextIncBackupSequenceNum() throws XFlaimException
	{
		return( _getNextIncBackupSequenceNum( m_this));
	}
	
	/**
	 * Get the amount of disk space currently being used by data files.
	 * @return Returns disc space used by data files.
	 * @throws XFlaimException
	 */
	public long getDiskSpaceDataSize()throws XFlaimException
	{
		return( _getDiskSpaceDataSize( m_this));
	}

	/**
	 * Get the amount of disk space currently being used by rollback files.
	 * @return Returns disc space used by rollback files.
	 * @throws XFlaimException
	 */
	public long getDiskSpaceRollbackSize() throws XFlaimException
	{
		return( _getDiskSpaceRollbackSize( m_this));
	}
		
	/**
	 * Get the amount of disk space currently being used by RFL files.
	 * @return Returns disc space used by RFL files.
	 * @throws XFlaimException
	 */
	public long getDiskSpaceRflSize() throws XFlaimException
	{
		return( _getDiskSpaceRflSize( m_this));
	}
	
	/**
	 * Get the amount of disk space currently being used by all types of
	 * database files.  This includes the total of data files plus rollback
	 * files plus RFL files.
	 * @return Returns total disc space used by database files of all types.
	 * @throws XFlaimException
	 */
	public long getDiskSpaceTotalSize() throws XFlaimException
	{
		return( _getDiskSpaceTotalSize( m_this));
	}
	
	/**
	 * Get error code that caused the database to force itself to close.  This should
	 * be one of the values in {@link xflaim.RCODE RCODE}.
	 * @return Returns error code that caused the database to force itself to close.
	 * @throws XFlaimException
	 */
	public int getMustCloseRC() throws XFlaimException
	{
		return( _getMustCloseRC( m_this));
	}

	/**
	 * Get error code that caused the current transaction to require an abort.
	 * This may be one of the values in {@link xflaim.RCODE RCODE}, but not
	 * necessarily.
	 * @return Returns error code that caused the current transaction to require
	 * itself to abort.
	 * @throws XFlaimException
	 */
	public int getAbortRC() throws XFlaimException
	{
		return( _getAbortRC( m_this));
	}

	/**
	 * Force the current transaction to abort.  This method should be called
	 * when the code should not be the code that aborts the transation, but
	 * wants to require that the transaction be aborted by whatever module has
	 * the authority to abort or commit the transaction.  An error code may be
	 * set to indicate what error condition is causing the transaction to be
	 * aborted.
	 * @param iRc Error code that indicates why the transaction is aborting.
	 * @throws XFlaimException
	 */
	public void setMustAbortTrans(
		int	iRc) throws XFlaimException
	{
		_setMustAbortTrans( m_this, iRc);
	}

	/**
	 * Enable encryption for this database.
	 * @throws XFlaimException
	 */
	public void enableEncryption() throws XFlaimException
	{
		_enableEncryption( m_this);
	}

	/**
	 * Wrap the database key in a password.  This method is called when it is
	 * desirable to move the database to a different machine.  Normally, the
	 * database key is wrapped in the local NICI storage key - which means that
	 * the database can only be opened and accessed on that machine. -- Once
	 * the database key is wrapped in a password, the password must be
	 * supplied to the dbOpen method to open the database.
	 * @param sPassword Password the database key should be wrapped in.
	 * @throws XFlaimException
	 */
	public void wrapKey(
		String	sPassword) throws XFlaimException
	{
		_wrapKey( m_this, sPassword);
	}
		
	/**
	 * Generate a new database key.  All encryption definition keys will be
	 * re-wrapped in the new database key.
	 * @throws XFlaimException
	 */
	public void rollOverDbKey() throws XFlaimException
	{
		_rollOverDbKey( m_this);
	}

	/**
	 * Get the database serial number.
	 * @return Byte array containing the database serial number.  This number
	 * is generated and stored in the database when the database is created.
	 * @throws XFlaimException
	 */
	public byte[] getSerialNumber() throws XFlaimException
	{
		return( _getSerialNumber( m_this));
	}

	/**
	 * Get information about the checkpoint thread's current state.
	 * @return Checkpoint thread state information is returned in a
	 * {@link xflaim.CheckpointInfo CheckpointInfo} object.
	 * @throws XFlaimException
	 */
	public CheckpointInfo getCheckpointInfo() throws XFlaimException
	{
		return( _getCheckpointInfo( m_this));
	}
		
	/**
	 * Export XML to a text file.
	 * @param startNode The node in the XML document to export.  All of its
	 * sub-tree will be exported.
	 * @param sFileName File the XML is to be exported to.  File will be
	 * overwritten.
	 * @param iFormat Formatting to use when exporting.  Should be one of
	 * {@link xflaim.ExportFormatType ExportFormatType}.
	 * @throws XFlaimException
	 */
	public void exportXML(
		DOMNode	startNode,
		String	sFileName,
		int		iFormat) throws XFlaimException
	{
		_exportXML( m_this, startNode.getThis(), sFileName, iFormat);
	}
			
	/**
	 * Export XML to a string.
	 * @param startNode The node in the XML document to export.  All of its
	 * sub-tree will be exported.
	 * @param iFormat Formatting to use when exporting.  Should be one of
	 * {@link xflaim.ExportFormatType ExportFormatType}.
	 * @throws XFlaimException
	 */
	public String exportXML(
		DOMNode	startNode,
		int		iFormat) throws XFlaimException
	{
		return( _exportXML( m_this, startNode.getThis(), iFormat));
	}
			
	/**
	 * Get the list of threads that are holding the database lock as well as
	 * the threads that are waiting to obtain the database lock.
	 * @return Returns an array of {@link xflaim.LockUser LockUser} objects.  The
	 * zeroeth element in the array is the current holder of the database lock.
	 * All other elements of the array are threads that are waiting to obtain
	 * the lock.
	 * @throws XFlaimException
	 */
	public LockUser[] getLockWaiters() throws XFlaimException
	{
		return( _getLockWaiters( m_this));
	}
			
	/**
	 * Set a callback object that will report the progress of an index or
	 * collection deletion operation.  This object's methods are called only if
	 * the index or collection is deleted in the foreground.  The delete operation
	 * must be performed in the same thread where this method is called.
	 * @param deleteStatusObj An object that implements the {@link xflaim.DeleteStatus
	 * DeleteStatus} interface.
	 * @throws XFlaimException
	 */
	public void setDeleteStatusObject(
		DeleteStatus	deleteStatusObj) throws XFlaimException
	{
		_setDeleteStatusObject( m_this, deleteStatusObj);
	}
	
	/**
	 * Set a callback object that will report each document that is being indexed when
	 * an index definition object is added.  This object's methods are called only if
	 * the index is added in the foreground.  The index definition must be added
	 * in the same thread that sets this object.
	 * @param ixClientObj An object that implements the {@link xflaim.IxClient
	 * IxClient} interface.
	 * @throws XFlaimException
	 */
	public void setIndexingClientObject(
		IxClient			ixClientObj) throws XFlaimException
	{
		_setIndexingClientObject( m_this, ixClientObj);
	}
		
	/**
	 * Set a callback object that will report indexing progress when
	 * an index definition object is added.  This object's methods are called only if
	 * the index is added in the foreground.  The index definition must be added
	 * in the same thread that sets this object.
	 * @param ixStatusObj An object that implements the {@link xflaim.IxStatus
	 * IxStatus} interface.
	 * @throws XFlaimException
	 */
	public void setIndexingStatusObject(
		IxStatus			ixStatusObj) throws XFlaimException
	{
		_setIndexingStatusObject( m_this, ixStatusObj);
	}
	
	/**
	 * Set a callback object that will be called after a transaction commit
	 * has safely saved all transaction data to disk, but before the database
	 * is unlocked.  This allows an application to do anything it may need to do
	 * after a commit but before the database is unlocked.  The thread that
	 * performs the commit must be the thread that sets this object.
	 * @param commitClientObj An object that implements the {@link xflaim.CommitClient
	 * CommitClient} interface.
	 * @throws XFlaimException
	 */
	public void setCommitClientObject(
		CommitClient	commitClientObj) throws XFlaimException
	{
		_setCommitClientObject( m_this, commitClientObj);
	}

	/**
	 * Upgrade the database to the most current database version.
	 * @throws XFlaimException
	 */
	public void upgrade() throws XFlaimException
	{
		_upgrade( m_this);
	}
	
// PRIVATE METHODS

	private native void _release(
		long				lThis);

	private native void _transBegin(
		long				lThis,
		int				iTransactionType,
		int 				iMaxlockWait,
		int 				iFlags) throws XFlaimException;

	private native void _transBegin(
		long			lThis,
		long			lSrcDb) throws XFlaimException;
		
	private native void _transCommit(
		long	lThis) throws XFlaimException;
	
	private native void _transAbort(
		long	lThis) throws XFlaimException;

	private native int _getTransType(
		long	lThis) throws XFlaimException;
	
	private native void _doCheckpoint(
		long	lThis,
		int	iTimeout) throws XFlaimException;
		
	private native void _dbLock(
		long	lThis,
		int	iLockType,
		int	iPriority,
		int	iTimeout) throws XFlaimException;
		
	private native void _dbUnlock(
		long	lThis) throws XFlaimException;

	private native int _getLockType(
		long		lThis) throws XFlaimException;

	private native boolean _getLockImplicit(
		long		lThis) throws XFlaimException;

	private native int _getLockThreadId(
		long		lThis) throws XFlaimException;
		
	private native int _getLockNumExclQueued(
		long		lThis) throws XFlaimException;
		
	private native int _getLockNumSharedQueued(
		long		lThis) throws XFlaimException;
		
	private native int _getLockPriorityCount(
		long		lThis,
		int		iPriority) throws XFlaimException;
		
	private native void _indexSuspend(
		long	lThis,
		int	iIndex) throws XFlaimException;

	private native void _indexResume(
		long	lThis,
		int	iIndex) throws XFlaimException;

	private native int _indexGetNext(
		long	lThis,
		int	iCurrIndex) throws XFlaimException;

	private native IndexStatus _indexStatus(
		long	lThis,
		int	iIndex) throws XFlaimException;
		
	private native int _reduceSize(
		long	lThis,
		int	iCount) throws XFlaimException;

	private native void _keyRetrieve(
		long				lThis,
		int				iIndex,
		long				lSearchKey,
		int				iSearchFlags,
		long				lFoundKey) throws XFlaimException;
 
	private native long _createDocument(
		long				lThis,
		int				iCollection) throws XFlaimException;

	private native long _createRootElement(
		long				lThis,
		int				iCollection,
		int				iElementNameId) throws XFlaimException;
		
 	private native long _getFirstDocument(
 		long				lThis,
 		int				iCollection,
 		long				lOldNodeRef) throws XFlaimException;
 
 	private native long _getLastDocument(
 		long				lThis,
 		int				iCollection,
 		long				lOldNodeRef) throws XFlaimException;
 
 	private native long _getDocument(
 		long				lThis,
 		int				iCollection,
		int				iSearchFlags,
		long				lDocumentId,
 		long				lOldNodeRef) throws XFlaimException;
 
	private native void _documentDone(
		long			lThis,
		int			iCollection,
		long			lDocumentId) throws XFlaimException;

	private native void _documentDone(
		long		lThis,
		long		lNode) throws XFlaimException;

	private native int _createElementDef(
		long				lThis,
		String			sNamespaceURI,
		String			sElementName,
		int				iDataType,
		int				iRequestedId) throws XFlaimException;
		
	private native int _createUniqueElmDef(
		long				lThis,
		String			sNamespaceURI,
		String			sElementName,
		int				iRequestedId) throws XFlaimException;
		
	private native int _getElementNameId(
		long				lThis,
		String			sNamespaceURI,
		String			sElementName) throws XFlaimException;
		
	private native int _createAttributeDef(
		long				lThis,
		String			sNamespaceURI,
		String			sAttributeName,
		int				iDataType,
		int				iRequestedId) throws XFlaimException;
		
	private native int _getAttributeNameId(
		long				lThis,
		String			sNamespaceURI,
		String			sAttributeName) throws XFlaimException;
		
	private native int _createPrefixDef(
		long				lThis,
		String			sPrefixName,
		int				iRequestedId) throws XFlaimException;
		
	private native int _getPrefixId(
		long				lThis,
		String			sPrefixName) throws XFlaimException;
		
	private native int _createEncDef(
		long				lThis,
		String			sEncType,
		String			sEncName,
		int				iKeySize,
		int				iRequestedId) throws XFlaimException;
		
	private native int _getEncDefId(
		long				lThis,
		String			sEncName) throws XFlaimException;
		
	private native int _createCollectionDef(
		long				lThis,
		String			sCollectionName,
		int				iEncNumber,
		int				iRequestedId) throws XFlaimException;
		
	private native int _getCollectionNumber(
		long				lThis,
		String			sCollectionName) throws XFlaimException;
		
	private native int _getIndexNumber(
		long				lThis,
		String			sIndexName) throws XFlaimException;
		
 	private native long _getDictionaryDef(
 		long				lThis,
 		int				iDictType,
		int				iDictNumber,
 		long				lOldNodeRef) throws XFlaimException;
 
	private native String _getDictionaryName(
		long	lThis,
		int	iDictType,
		int	iDictNumber) throws XFlaimException;
		
	private native String _getElementNamespace(
		long	lThis,
		int	iDictNumber) throws XFlaimException;
		
	private native String _getAttributeNamespace(
		long	lThis,
		int	iDictNumber) throws XFlaimException;
		
 	private native long _getNode(
 		long				lThis,
 		int				iCollection,
 		long				lNodeId,
 		long				lOldNodeRef) throws XFlaimException;

	private native long _getAttribute(
		long			lThis,
		int			iCollection,
		long			lElementNodeId,
		int			iAttrNameId,
		long			lOldNodeRef) throws XFlaimException;
		
	private native int _getDataType(
		long	lThis,
		int	iDictType,
		int	iDictNumber) throws XFlaimException;

	private native long _backupBegin(
		long				lThis,
		int				iBackupType,
		int				iTransType,
		int				iMaxLockWait,
		long				lReusedRef) throws XFlaimException;

	private native ImportStats _import(
		long				lThis,
		long				lIStream,
		int				iCollection,
		long				lNodeToLinkTo,
		int				iInsertLoc) throws XFlaimException;

	private native void _changeItemState(
		long				lThis,
		int				iDictType,
		int				iDictNum,
		String			sState) throws XFlaimException;

	private native String _getRflFileName(
		long				lThis,
		int				iFileNum,
		boolean			bBaseOnly) throws XFlaimException;
		
	private native void _setNextNodeId(
		long				lThis,
		int				iCollection,
		long				lNextNodeId) throws XFlaimException;

	private native void _setNextDictNum(
		long				lThis,
		int				iDictType,
		int				iNextDictNumber) throws XFlaimException;

	private native void _setRflKeepFilesFlag(
		long				lThis,
		boolean			bKeep) throws XFlaimException;
		
	private native boolean _getRflKeepFlag(
		long				lThis) throws XFlaimException;
		
	private native void _setRflDir(
		long				lThis,
		String			sRflDir) throws XFlaimException;
		
	private native String _getRflDir(
		long				lThis) throws XFlaimException;
	
	private native int _getRflFileNum(
		long				lThis) throws XFlaimException;

	private native int _getHighestNotUsedRflFileNum(
		long				lThis) throws XFlaimException;

	private native void _setRflFileSizeLimits(
		long				lThis,
		int				iMinRflSize,
		int				iMaxRflSize) throws XFlaimException;

	private native int _getMinRflFileSize(
		long				lThis) throws XFlaimException;
	
	private native int _getMaxRflFileSize(
		long				lThis) throws XFlaimException;

	private native void _rflRollToNextFile(
		long				lThis) throws XFlaimException;

	private native void _setKeepAbortedTransInRflFlag(
		long				lThis,
		boolean			bKeep) throws XFlaimException;

	private native boolean _getKeepAbortedTransInRflFlag(
		long				lThis) throws XFlaimException;

	private native void _setAutoTurnOffKeepRflFlag(
		long				lThis,
		boolean			bAutoTurnOff) throws XFlaimException;

	private native boolean _getAutoTurnOffKeepRflFlag(
		long				lThis) throws XFlaimException;

	private native void _setFileExtendSize(
		long				lThis,
		int				iFileExtendSize) throws XFlaimException;

	private native int _getFileExtendSize(
		long				lThis) throws XFlaimException;

	private native int _getDbVersion(
		long				lThis) throws XFlaimException;

	private native int _getBlockSize(
		long				lThis) throws XFlaimException;

	private native int _getDefaultLanguage(
		long				lThis) throws XFlaimException;

	private native long _getTransID(
		long				lThis) throws XFlaimException;

	private native String _getDbControlFileName(
		long				lThis) throws XFlaimException;

	private native long _getLastBackupTransID(
		long				lThis) throws XFlaimException;

	private native int _getBlocksChangedSinceBackup(
		long				lThis) throws XFlaimException;

	private native int _getNextIncBackupSequenceNum(
		long				lThis) throws XFlaimException;

	private native long _getDiskSpaceDataSize(
		long				lThis) throws XFlaimException;

	private native long _getDiskSpaceRollbackSize(
		long				lThis) throws XFlaimException;
		
	private native long _getDiskSpaceRflSize(
		long				lThis) throws XFlaimException;
	
	private native long _getDiskSpaceTotalSize(
		long				lThis) throws XFlaimException;

	private native int _getMustCloseRC(
		long				lThis) throws XFlaimException;

	private native int _getAbortRC(
		long				lThis) throws XFlaimException;

	private native void _setMustAbortTrans(
		long				lThis,
		int				iRc) throws XFlaimException;
		
	private native void _enableEncryption(
		long				lThis) throws XFlaimException;

	private native void _wrapKey(
		long				lThis,
		String			sPassword) throws XFlaimException;
		
	private native void _rollOverDbKey(
		long				lThis) throws XFlaimException;
			
	private native byte[] _getSerialNumber(
		long				lThis) throws XFlaimException;
		
	private native CheckpointInfo _getCheckpointInfo(
		long				lThis) throws XFlaimException;
		
	private native void _exportXML(
		long		lThis,
		long		lStartNode,
		String	sFileName,
		int		iFormat) throws XFlaimException;
			
	private native String _exportXML(
		long		lThis,
		long		lStartNode,
		int		iFormat) throws XFlaimException;
			
	private native LockUser[] _getLockWaiters(
		long		lThis) throws XFlaimException;
		
	private native void _setDeleteStatusObject(
		long				lThis,
		DeleteStatus	deleteStatusObj) throws XFlaimException;
			
	private native void _setIndexingClientObject(
		long				lThis,
		IxClient			ixClientObj) throws XFlaimException;
		
	private native void _setIndexingStatusObject(
		long				lThis,
		IxStatus			ixStatusObj) throws XFlaimException;
		
	private native void _setCommitClientObject(
		long				lThis,
		CommitClient	commitClientObj) throws XFlaimException;
		
	private native void _upgrade(
		long				lThis) throws XFlaimException;
		
	long 					m_this;
	private DbSystem 	m_dbSystem;
}

