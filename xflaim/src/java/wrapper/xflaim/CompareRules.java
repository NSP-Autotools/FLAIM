//------------------------------------------------------------------------------
// Desc:	CompareRules
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

public final class CompareRules
{
	public static final int FLM_COMP_CASE_INSENSITIVE			= 0x0001;
	public static final int FLM_COMP_COMPRESS_WHITESPACE		= 0x0002;
	public static final int FLM_COMP_NO_WHITESPACE				= 0x0004;
	public static final int FLM_COMP_NO_UNDERSCORES				= 0x0008;
	public static final int FLM_COMP_NO_DASHES					= 0x0010;
	public static final int FLM_COMP_WHITESPACE_AS_SPACE		= 0x0020;
	public static final int FLM_COMP_IGNORE_LEADING_SPACE		= 0x0040;
	public static final int FLM_COMP_IGNORE_TRAILING_SPACE	= 0x0080;
	public static final int FLM_COMP_WILD							= 0x0100;
}

