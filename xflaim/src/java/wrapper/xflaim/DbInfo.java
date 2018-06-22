//------------------------------------------------------------------------------
// Desc:	Db Copy Status
// Tabs:	3
//
// Copyright (c) 2004-2007 Novell, Inc. All Rights Reserved.
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
 * To change the template for this generated type comment go to
 * Window&gt;Preferences&gt;Java&gt;Code Generation&gt;Code and Comments
 */
public class DbInfo 
{
	DbInfo(
		long		lRef) throws XFlaimException
	{
		if (lRef == 0)
		{
			throw new XFlaimException( -1, "No legal reference");
		}
		
		m_this = lRef;
	}

	/**
	 * Desc:
	 */
	protected void finalize()
	{
		_release( m_this);
	}

	/**
	 * Desc:
	 */
	private native void _release(
		long		lThis);
		
	private long	m_this;
}
