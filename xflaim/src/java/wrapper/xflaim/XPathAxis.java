//------------------------------------------------------------------------------
// Desc:	XPathAxis
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
 * Provides list of valid query operators.
 */

public final class XPathAxis
{
	public static final int ROOT_AXIS					= 0;
	public static final int CHILD_AXIS					= 1;
	public static final int PARENT_AXIS					= 2;
	public static final int ANCESTOR_AXIS				= 3;
	public static final int DESCENDANT_AXIS			= 4;
	public static final int FOLLOWING_SIBLING_AXIS	= 5;
	public static final int PRECEDING_SIBLING_AXIS	= 6;
	public static final int FOLLOWING_AXIS				= 7;
	public static final int PRECEDING_AXIS				= 8;
	public static final int ATTRIBUTE_AXIS				= 9;
	public static final int NAMESPACE_AXIS				= 10;
	public static final int SELF_AXIS					= 11;
	public static final int DESCENDANT_OR_SELF_AXIS	= 12;
	public static final int ANCESTOR_OR_SELF_AXIS	= 13;
	public static final int META_AXIS					= 14;
}

