//------------------------------------------------------------------------------
// Desc:	SearchFlags
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
 * Provides bit flags for comparison rules.
 */

public final class SearchFlags
{
	public static final int XFLM_INCL			= 0x0010;
	public static final int XFLM_EXCL			= 0x0020;
	public static final int XFLM_EXACT			= 0x0040;
	public static final int XFLM_KEY_EXACT		= 0x0080;
	public static final int XFLM_FIRST			= 0x0100;
	public static final int XFLM_LAST			= 0x0200;
	public static final int XFLM_MATCH_IDS		= 0x0400;
	public static final int XFLM_MATCH_DOC_ID	= 0x0800;
}

