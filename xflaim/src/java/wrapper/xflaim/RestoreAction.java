//------------------------------------------------------------------------------
// Desc:	Restore Actions
// Tabs:	3
//
// Copyright (c) 2003, 2005-2007 Novell, Inc. All Rights Reserved.
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
 * Provides enums for all of the possible return codes for the members of the
 * {@link xflaim.RestoreStatus RestoreStatus} interface.
 * NOTE: The values in this class must match *exactly* with the equivalent
 * enum defined in xflaim.h
 */

public final class RestoreAction
{
	public static final int CONTINUE	= 0;	// Continue recovery
	public static final int STOP		= 1;	// Stop recovery
	public static final int SKIP		= 2;	// Skip operation (future)
	public static final int RETRY		= 3;	// Retry the operation
}
