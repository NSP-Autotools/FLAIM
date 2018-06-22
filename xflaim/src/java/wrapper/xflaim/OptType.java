//------------------------------------------------------------------------------
// Desc:	OptType
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
 * Provides list of optimization types for queries
 */

public final class OptType
{
	public static final int XFLM_QOPT_NONE							= 0;
	public static final int XFLM_QOPT_USING_INDEX				= 1;
	public static final int XFLM_QOPT_FULL_COLLECTION_SCAN	= 2;
	public static final int XFLM_QOPT_SINGLE_NODE_ID			= 3;
	public static final int XFLM_QOPT_NODE_ID_RANGE				= 4;
}

