//------------------------------------------------------------------------------
// Desc:	Query object
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
 * This class encapsulates the XFlaim IF_Query interface.
 */
public class Query 
{
	
	public Query(
		Db 			jdb,
		int			iCollection) throws XFlaimException
	{

		m_this = _createQuery( iCollection);
		
		if (jdb == null)
		{
			throw new XFlaimException( -1, "No legal jDb reference");
		}
		
		m_jdb = jdb;
	}
	
	protected void finalize()
	{
		// The F_Query and F_Db classes are not thread-safe.  The proper way
		// of using XFlaim is to get a new instance of Db for each thread.
		// Unfortunately, the garbage collector runs in its own thread.  This
		// leads to a potential race condition down in the C++ code when one
		// thread tries to create an already existing query (which results in a
		// call to F_Query::AddRef()) and the GC tries to destroy the same
		// query (which results in a call to F_Query::Release()).
		// We protect against this by synchronizing against the instance of
		// Db.  Note that we are not protecting any of the accesses to the
		// query; only creating and destroying.  Query and Db are still
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
	
	public long getThis()
	{
		return m_this;
	}
	
	/**
	 * Set the language for the query criteria.  This affects how string
	 * comparisons are done.  Collation is done according to the language
	 * specified.
	 * @param iLanguage Language to be used for string comparisons.
	 * @throws XFlaimException
	 */
	public void setLanguage(
		int	iLanguage) throws XFlaimException
	{
		_setLanguage( m_this, iLanguage);
	}
	
	/**
	 * Setup the query criteria from the passed in string.
	 * @param sQuery String containing the query criteria.
	 * @throws XFlaimException
	 */
	public void setupQueryExpr(
		String	sQuery) throws XFlaimException
	{
		_setupQueryExpr( m_this, m_jdb.m_this, sQuery);
	}
	
	/**
	 * Copy the query criteria from one Query object into this Query object.
	 * @param queryToCopy Query object whose criteria is to be copied.
	 * @throws XFlaimException
	 */
	public void copyCriteria(
		Query	queryToCopy) throws XFlaimException
	{
		_copyCriteria( m_this, queryToCopy.m_this);
	}
	
	/**
	 * Add an XPATH component to a query.
	 * @param iXPathAxis Type of axis for the XPATH component being added.
	 * Should be one of the constants defined in {@link xflaim.XPathAxis XPathAxis}.
	 * @param iNodeType An integer representing the type of node for the
	 * XPATH component.
	 * (Use the constants in {@link xflaim.FlmDomNodeType FlmDomNodeType}.)
	 * @param iNameId  Name ID for the node in the XPATH component.
	 * @throws XFlaimException
	 */
	public void addXPathComponent(
		int		iXPathAxis,
		int		iNodeType,
		int		iNameId) throws XFlaimException
	{
		_addXPathComponent( m_this, iXPathAxis, iNodeType, iNameId);
	}
	
	/**
	 * Add an operator to a query.
	 * @param iOperator Operator to be added.  Should be one of the constants
	 * defined in {@link xflaim.QueryOperators QueryOperators}.
	 * @param iCompareRules  Flags for doing string comparisons.  Should be
	 * logical ORs of the members of {@link xflaim.CompareRules CompareRules}.
	 * @throws XFlaimException
	 */
	public void addOperator(
		int				iOperator,
		int				iCompareRules) throws XFlaimException
	{
		_addOperator( m_this, iOperator, iCompareRules);
	}
		
	/**
	 * Add a string value to the query criteria.
	 * @param sValue String value to be added to criteria.
	 * @throws XFlaimException
	 */
	public void addStringValue(
		String	sValue) throws XFlaimException
	{
		_addStringValue( m_this, sValue);
	}
	
	/**
	 * Add a binary value to the query criteria.
	 * @param Value Binary value to be added to criteria.
	 * @throws XFlaimException
	 */
	public void addBinaryValue(
		byte []	Value) throws XFlaimException
	{
		_addBinaryValue( m_this, Value);
	}
	
	/**
	 * Add a long value to the query criteria.
	 * @param lValue Long value to be added to criteria.
	 * @throws XFlaimException
	 */
	public void addLongValue(
		long		lValue) throws XFlaimException
	{
		_addLongValue( m_this, lValue);
	}
	
	/**
	 * Add a boolean to the query criteria.
	 * @param bValue Boolean value to be added to criteria.
	 * @throws XFlaimException
	 */
	public void addBoolean(
		boolean	bValue) throws XFlaimException
	{
		_addBoolean( m_this, bValue, false);
	}

	/**
	 * Add an "unknown" predicate to the query criteria.
	 * @throws XFlaimException
	 */
	public void addUnknown() throws XFlaimException
	{
		_addBoolean( m_this, false, true);
	}

	/**
	 * Gets the first {@link xflaim.DOMNode DOM node} that satisfies the query criteria.
	 * This may be a document root node, or any node within the document.  What
	 * is returned depends on how the XPATH expression was constructed.
	 * @param ReusedNode An existing {@link xflaim.DOMNode DOMNode} object
	 * can optionally be passed in, and it will be reused instead of a new
	 * object being allocated.
	 * @param iTimeLimit Time limit (in milliseconds) for operation to complete.
	 * A value of zero indicates that the operation should not time out.
	 * @return Returns an instance of {@link xflaim.DOMNode DOMNode}.
	 * @throws XFlaimException
	 */
	public DOMNode getFirst(
		DOMNode	ReusedNode,
		int		iTimeLimit) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.getThis()
											  : 0;
		
		// See to comments in the finalize function for an explanation of this
		// call synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getFirst( m_this, m_jdb.m_this, lReusedNodeRef,
											iTimeLimit);
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
	 * Gets the last {@link xflaim.DOMNode DOM node} that satisfies the query criteria.
	 * This may be a document root node, or any node within the document.  What
	 * is returned depends on how the XPATH expression was constructed.
	 * @param ReusedNode An existing {@link xflaim.DOMNode DOMNode} object
	 * can optionally be passed in, and it will be reused instead of a new
	 * object being allocated.
	 * @param iTimeLimit Time limit (in milliseconds) for operation to complete.
	 * A value of zero indicates that the operation should not time out.
	 * @return Returns an instance of {@link xflaim.DOMNode DOMNode}.
	 * @throws XFlaimException
	 */
	public DOMNode getLast(
		DOMNode	ReusedNode,
		int		iTimeLimit) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.getThis()
											  : 0;
		
		// See to comments in the finalize function for an explanation of this
		// call synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getLast( m_this, m_jdb.m_this, lReusedNodeRef,
											iTimeLimit);
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
	 * Gets the next {@link xflaim.DOMNode DOM node} that satisfies the query criteria.
	 * This may be a document root node, or any node within the document.  What
	 * is returned depends on how the XPATH expression was constructed.
	 * @param ReusedNode An existing {@link xflaim.DOMNode DOMNode} object
	 * can optionally be passed in, and it will be reused instead of a new
	 * object being allocated.
	 * @param iTimeLimit Time limit (in milliseconds) for operation to complete.
	 * A value of zero indicates that the operation should not time out.
	 * @param iNumToSkip Specifies the number of nodes to skip over before
	 * returning a next node.  This includes skipping over the node the
	 * query is currently positioned on.  A value of zero has the same effect
	 * as a value of one - it will position to the next node and return it.
	 * @return Returns an instance of {@link xflaim.DOMNode DOMNode}.
	 * @throws XFlaimException
	 */
	public DOMNode getNext(
		DOMNode	ReusedNode,
		int		iTimeLimit,
		int		iNumToSkip) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.getThis()
											  : 0;
		
		// See to comments in the finalize function for an explanation of this
		// call synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getNext( m_this, m_jdb.m_this, lReusedNodeRef,
											iTimeLimit, iNumToSkip);
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
	 * Gets the previous {@link xflaim.DOMNode DOM node} that satisfies the query criteria.
	 * This may be a document root node, or any node within the document.  What
	 * is returned depends on how the XPATH expression was constructed.
	 * @param ReusedNode An existing {@link xflaim.DOMNode DOMNode} object
	 * can optionally be passed in, and it will be reused instead of a new
	 * object being allocated.
	 * @param iTimeLimit Time limit (in milliseconds) for operation to complete.
	 * A value of zero indicates that the operation should not time out.
	 * @param iNumToSkip Specifies the number of nodes to skip over before
	 * returning a previous node.  This includes skipping over the node the
	 * query is currently positioned on.  A value of zero has the same effect
	 * as a value of one - it will position to the previous node and return it.
	 * @return Returns an instance of {@link xflaim.DOMNode DOMNode}.
	 * @throws XFlaimException
	 */
	public DOMNode getPrev(
		DOMNode	ReusedNode,
		int		iTimeLimit,
		int		iNumToSkip) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.getThis()
											  : 0;
		
		// See to comments in the finalize function for an explanation of this
		// call synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getPrev( m_this, m_jdb.m_this, lReusedNodeRef,
											iTimeLimit, iNumToSkip);
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
	 * Gets the current {@link xflaim.DOMNode DOM node} that was last returned by
	 * calls to getFirst, getLast, getNext, or getPrev.
	 * This may be a document root node, or any node within the document.  What
	 * is returned depends on how the XPATH expression was constructed.
	 * @param ReusedNode An existing {@link xflaim.DOMNode DOMNode} object
	 * can optionally be passed in, and it will be reused instead of a new
	 * object being allocated.
	 * @return Returns an instance of {@link xflaim.DOMNode DOMNode}.
	 * @throws XFlaimException
	 */
	public DOMNode getCurrent(
		DOMNode	ReusedNode) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.getThis()
											  : 0;
		
		// See to comments in the finalize function for an explanation of this
		// call synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getCurrent( m_this, m_jdb.m_this, lReusedNodeRef);
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
	 * Resets the query criteria and results set for the query.
	 * @throws XFlaimException
	 */
	public void resetQuery() throws XFlaimException
	{
		_resetQuery( m_this);
	}
		
	/**
	 * Returns statistics and optimization information for the query.
	 * @return Returns an array of {@link xflaim.OptInfo OptInfo} objects.
	 * @throws XFlaimException
	 */
	public OptInfo [] getStatsAndOptInfo() throws XFlaimException
	{
		return( _getStatsAndOptInfo( m_this));
	}
	
	/**
	 * Set duplicate handling for the query.
	 * @param bRemoveDups Specifies whether duplicates should be removed from
	 * the result set.
	 * @throws XFlaimException
	 */
	public void setDupHandling(
		boolean	bRemoveDups) throws XFlaimException
	{
		_setDupHandling( m_this, bRemoveDups);
	}

	/**
	 * Set an index for the query.
	 * @param iIndex Index that the query should use.
	 * @throws XFlaimException
	 */
	public void setIndex(
		int		iIndex) throws XFlaimException
	{
		_setIndex( m_this, iIndex);
	}

	/**
	 * Get the index, if any, being used by the query.
	 * @return Returns the index being used by the query.  NOTE: Zero is returned
	 * if no indexes are being used.  If multiple indexes are being used, only
	 * the first one will be returned.  Call usesMultipeIndexes to determine
	 * if the query is using multiple indexes.  To see the list of indexes,
	 * call getStatsAndOptInfo.
	 * @throws XFlaimException
	 */
	public int getIndex() throws XFlaimException
	{
		return( _getIndex( m_this, m_jdb.m_this));
	}
		
	/**
	 * Determine if the query is using multiple indexes.
	 * @return Returns a boolean which indicates if the query is using more than
	 * one index.  If zero or one index is being used, will return false.
	 * @throws XFlaimException
	 */
	public boolean usesMultipleIndexes() throws XFlaimException
	{
		return( _usesMultipleIndexes( m_this, m_jdb.m_this));
	}

	/**
	 * Add a sort key to the query.
	 * @param lSortKeyContext Context that the current sort key is to be added
	 * relative to - either as a child or a sibling.  If this is the first
	 * sort key, a zero should be passed in here.  Otherwise, the value returned
	 * from a previous call to addSortKey should be passed in.
	 * @param bChildToContext Indicates whether this sort key should be added as
	 * a child or a sibling to the sort key context that was passed in the
	 * lSortKeyContext parameter.  NOTE: If lSortKeyContext is zero, then the
	 * bChildToContext parameter is ignored.
	 * @param bElement Indicates whether the current key component is an element or an
	 * attribute.
	 * @param iNameId Name ID of the current key component.
	 * @param iCompareRules Flags for doing string comparisons when sorting for
	 * this sort key component.  Should be logical ORs of the members of
	 * {@link xflaim.CompareRules CompareRules}.
	 * @param iLimit Limit on the size of the key component.  If the component
	 * is a string element or attribute, it is the number of characters.  If the
	 * component is a binary element or attribute, it is the number of bytes.
	 * @param iKeyComponent Specifies which key component this sort key component
	 * is.  A value of zero indicates that it is not a key component, but simply
	 * a context component for other key components.
	 * @param bSortDescending Indicates that this key component should be
	 * sorted in descending order.
	 * @param bSortMissingHigh Indicates that when the value for this key
	 * component is missing, it should be sorted high instead of low.
	 * @return Returns a value that can be passed back into subsequent calls
	 * to addSortKey when this component needs to be used as a context for
	 * subsequent components.
	 * @throws XFlaimException
	 */
	public long addSortKey(
		long				lSortKeyContext,
		boolean			bChildToContext,
		boolean			bElement,
		int				iNameId,
		int				iCompareRules,
		int				iLimit,
		int				iKeyComponent,
		boolean			bSortDescending,
		boolean			bSortMissingHigh) throws XFlaimException
	{
		return( _addSortKey( m_this, lSortKeyContext, bChildToContext, bElement,
					iNameId, iCompareRules, iLimit, iKeyComponent, bSortDescending,
					bSortMissingHigh));
	}
	
	/**
	 * Enable absolute positioning in the query result set.
	 * @throws XFlaimException
	 */
	public void enablePositioning() throws XFlaimException
	{
		_enablePositioning( m_this);
	}
	
	/**
	 * Position to the {@link xflaim.DOMNode DOM node} in the result that is at
	 * the absolute position specified by the iPosition parameter.
	 * @param ReusedNode An existing {@link xflaim.DOMNode DOMNode} object
	 * can optionally be passed in, and it will be reused instead of a new
	 * object being allocated.
	 * @param iTimeLimit Time limit (in milliseconds) for operation to complete.
	 * A value of zero indicates that the operation should not time out.
	 * @param iPosition Absolute position in the result set to position to.
	 * @return Returns an instance of {@link xflaim.DOMNode DOMNode}.
	 * @throws XFlaimException
	 */
	public DOMNode positionTo(
		DOMNode			ReusedNode,
		int				iTimeLimit,
		int				iPosition) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.getThis()
											  : 0;
		
		// See to comments in the finalize function for an explanation of this
		// call synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _positionTo( m_this, m_jdb.m_this, lReusedNodeRef,
										iTimeLimit, iPosition);
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
	 * Position to the {@link xflaim.DOMNode DOM node} in the result that is at
	 * the position specified by the searchKey parameter.
	 * @param ReusedNode An existing {@link xflaim.DOMNode DOMNode} object
	 * can optionally be passed in, and it will be reused instead of a new
	 * object being allocated.
	 * @param iTimeLimit Time limit (in milliseconds) for operation to complete.
	 * A value of zero indicates that the operation should not time out.
	 * @param searchKey This is a key that corresponds to the sort key that was
	 * specified using the addSortKey method.  This method looks up the node
	 * in the result set that has this search key and returns it.
	 * @param iFlags The search flags that direct how the key is to be used
	 * to do positioning.  This should be values from
	 * {@link xflaim.SearchFlags SearchFlags} that are ORed together.
	 * @return Returns an instance of {@link xflaim.DOMNode DOMNode}.
	 * @throws XFlaimException
	 */
	public DOMNode positionTo(
		DOMNode			ReusedNode,
		int				iTimeLimit,
		DataVector		searchKey,
		int				iFlags) throws XFlaimException
	{
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		long 			lReusedNodeRef = (ReusedNode != null)
											  ? ReusedNode.getThis()
											  : 0;
		
		// See to comments in the finalize function for an explanation of this
		// call synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _positionTo2( m_this, m_jdb.m_this, lReusedNodeRef,
										iTimeLimit, searchKey.getThis(), iFlags);
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
	 * Returns the absolute position within the result set where the query
	 * is currently positioned.
	 * @return Returns absolute position.
	 * @throws XFlaimException
	 */
	public int getPosition() throws XFlaimException
	{
		return( _getPosition( m_this, m_jdb.m_this));
	}
	
	/**
	 * Build the result set for the query.
	 * @param iTimeLimit Time limit (in milliseconds) for operation to complete.
	 * A value of zero indicates that the operation should not time out.
	 * @throws XFlaimException
	 */
	public void buildResultSet(
		int				iTimeLimit) throws XFlaimException
	{
		_buildResultSet( m_this, m_jdb.m_this, iTimeLimit);
	}
	
	/**
	 * Stop building the result set for the query.
	 * @throws XFlaimException
	 */
	public void stopBuildingResultSet()throws XFlaimException
	{
		_stopBuildingResultSet( m_this);
	}

	/**
	 * Enable encryption for the query result set while it is being built.
	 * Anything that overflows to disk will be encrypted.
	 * @throws XFlaimException
	 */
	public void enableResultSetEncryption() throws XFlaimException
	{
		_enableResultSetEncryption( m_this);
	}
	
	/**
	 * Return counts about the result set that has either been built or is
	 * in the process of being built.
	 * @param iTimeLimit Time limit (in milliseconds) for operation to complete.
	 * A value of zero indicates that the operation should not time out.
	 * @param bPartialCountOk Specifies whether the method should wait for
	 * the result set to be completely built before returning counts.  If true,
	 * the method will return the current counts, even if the result set is
	 * not completely built.
	 * @return {@link xflaim.ResultSetCounts ResultSetCounts} object which
	 * contains various counts pertaining to the result set for the query.
	 * @throws XFlaimException
	 */
	public ResultSetCounts getResultSetCounts(
		int				iTimeLimit,
		boolean			bPartialCountOk) throws XFlaimException
	{
		return( _getResultSetCounts( m_this, m_jdb.m_this,
					iTimeLimit, bPartialCountOk));
	}
		
// PRIVATE METHODS

	private native void _release(
		long		lThis);
		
	private native long _createQuery(
		int		iCollection);

	private native void _setLanguage(
		long	lThis,
		int	iLanguage) throws XFlaimException;
		
	private native void _setupQueryExpr(
		long		lThis,
		long		lDbRef,
		String	sQuery) throws XFlaimException;
		
	private native void _copyCriteria(
		long		lThis,
		long		lQueryToCopy) throws XFlaimException;

	private native void _addXPathComponent(
		long		lThis,
		int		iXPathAxis,
		int		iNodeType,
		int		iNameId) throws XFlaimException;
	
	private native void _addOperator(
		long				lThis,
		int				iOperator,
		int				iCompareRules) throws XFlaimException;
		
	private native void _addStringValue(
		long		lThis,
		String	sValue) throws XFlaimException;
	
	private native void _addBinaryValue(
		long		lThis,
		byte []	Value) throws XFlaimException;
	
	private native void _addLongValue(
		long		lThis,
		long		lValue) throws XFlaimException;
	
	private native void _addBoolean(
		long		lThis,
		boolean	bValue,
		boolean	bUnknown) throws XFlaimException;

	private native long _getFirst(
		long		lThis,
		long		lDbRef,
		long		lReusedNode,
		int		iTimeLimit) throws XFlaimException;
		
	private native long _getLast(
		long		lThis,
		long		lDbRef,
		long		lReusedNode,
		int		iTimeLimit) throws XFlaimException;
		
	private native long _getNext(
		long		lThis,
		long		lDbRef,
		long		lReusedNode,
		int		iTimeLimit,
		int		iNumToSkip) throws XFlaimException;
		
	private native long _getPrev(
		long		lThis,
		long		lDbRef,
		long		lReusedNode,
		int		iTimeLimit,
		int		iNumToSkip) throws XFlaimException;
		
	private native long _getCurrent(
		long		lThis,
		long		lDbRef,
		long		lReusedNode) throws XFlaimException;
		
	private native void _resetQuery(
		long		lThis) throws XFlaimException;
		
	private native OptInfo [] _getStatsAndOptInfo(
		long		lThis) throws XFlaimException;
		
	private native void _setDupHandling(
		long		lThis,
		boolean	bRemoveDups) throws XFlaimException;

	private native void _setIndex(
		long		lThis,
		int		iIndex) throws XFlaimException;

	private native int _getIndex(
		long		lThis,
		long		lDbRef) throws XFlaimException;
		
	private native boolean _usesMultipleIndexes(
		long		lThis,
		long		lDbRef) throws XFlaimException;

	private native long _addSortKey(
		long				lThis,
		long				lSortKeyContext,
		boolean			bChildToContext,
		boolean			bElement,
		int				iNameId,
		int				iCompareRules,
		int				iLimit,
		int				iKeyComponent,
		boolean			bSortDescending,
		boolean			bSortMissingHigh) throws XFlaimException;
	
	private native void _enablePositioning(
		long				lThis) throws XFlaimException;

	private native long _positionTo(
		long				lThis,
		long				lDbRef,
		long				lReusedNode,
		int				iTimeLimit,
		int				iPosition) throws XFlaimException;
		
	private native long _positionTo2(
		long				lThis,
		long				lDbRef,
		long				lReusedNode,
		int				iTimeLimit,
		long				lSearchKeyRef,
		int				iFlags) throws XFlaimException;
	
	private native int _getPosition(
		long				lThis,
		long				lDbRef) throws XFlaimException;
	
	private native void _buildResultSet(
		long				lThis,
		long				lDbRef,
		int				iTimeLimit) throws XFlaimException;
	
	private native void _stopBuildingResultSet(
		long				lThis) throws XFlaimException;

	private native void _enableResultSetEncryption(
		long				lThis) throws XFlaimException;

	private native ResultSetCounts _getResultSetCounts(
		long				lThis,
		long				lDbRef,
		int				iTimeLimit,
		boolean			bPartialCountOk) throws XFlaimException;
		
	private long	m_this;
	private Db		m_jdb;
}

/*

FUNCTIONS NOT YET IMPLEMENTED

virtual RCODE FLMAPI addFunction(
	IF_QueryValFunc *		pFuncObj,
	FLMBOOL					bHasXPathExpr) = 0;

virtual void FLMAPI setQueryStatusObject(
	IF_QueryStatus *		pQueryStatus) = 0;

virtual void FLMAPI setQueryValidatorObject(
	IF_QueryValidator *		pQueryValidator) = 0;
*/

