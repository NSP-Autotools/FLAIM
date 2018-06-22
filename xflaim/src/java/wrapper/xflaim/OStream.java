//------------------------------------------------------------------------------
// Desc:	Output Stream
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
 * The OStream class provides a number of methods that allow java 
 * applications to access the XFlaim native environment, specifically, the
 * IF_OStream interface.
 */
public class OStream 
{
	OStream( 
		long			lRef,
		DbSystem 	dbSystem) throws XFlaimException
	{
		if (lRef == 0)
		{
			throw new XFlaimException( -1, "No legal reference to an OStream");
		}
		
		m_this = lRef;
		
		if (dbSystem==null)
		{
			throw new XFlaimException( -1, "No legal dbSystem reference");
		}
		
		m_dbSystem = dbSystem;
	}
	
	/**
	 * Finalizer method used to release native resources on garbage collection.
	 */	
	public void finalize()
	{
		if (m_this != 0)
		{
			_release( m_this);
			m_this = 0;
		}

		m_dbSystem = null;
	}

	/**
	 *
	 */	
	public void release()
	{
		if (m_this != 0)
		{
			_release( m_this);
			m_this = 0;
		}

		m_dbSystem = null;
	}
	
	public long getThis()
	{
		return( m_this);
	}
	
	/**
	 *
	 */
	private native void _release(
		long	iThis);

	private long			m_this;
	private DbSystem		m_dbSystem;
}

