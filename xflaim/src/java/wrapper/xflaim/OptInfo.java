//------------------------------------------------------------------------------
// Desc:	OptInfo
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

package xflaim;

/**
 * This class encapsulates optimization information returned from a Query object.
 */
public final class OptInfo
{
	public int		iOptType;			// Type of optimization done
	public int		iCost;				// Cost calculated for predicate
	public long		lNodeId;				// Only valid if iOptType is
												// XFLM_QOPT_SINGLE_NODE_ID or
												// XFLM_QOPT_NODE_ID_RANGE
	public long		lEndNodeId;			// Only valid if iOptType is
												// XFLM_QOPT_NODE_ID_RANGE
	public int		iIxNum;				// Index used to execute query if
												// iOptType == XFLM_QOPT_USING_INDEX
	public String	sIxName;
	public boolean	bMustVerifyPath;	// Must verify node path.
	public boolean	bDoNodeMatch;		// Node must be retrieved to exe
												// query.  Only valid if iOptType
												// is XFLM_QOPT_USING_INDEX.
	public boolean	bCanCompareOnKey;	// Can we compare on index keys?  Only
												// valid if iOptType == XFLM_QOPT_USING_INDEX.
	public long		lKeysRead;
	public long		lKeyHadDupDoc;
	public long		lKeysPassed;
	public long		lNodesRead;
	public long		lNodesTested;
	public long		lNodesPassed;
	public long		lDocsRead;
	public long		lDupDocsEliminated;
	public long		lNodesFailedValidation;
	public long		lDocsFailedValidation;
	public long		lDocsPassed;
	
	private static native void initIDs();
}

