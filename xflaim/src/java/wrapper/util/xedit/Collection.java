//------------------------------------------------------------------------------
// Desc:	Collections
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

package xedit;

/**
 * The Collection class defines a simple structure that can be used when selection
 * a collection.  When instantiated, it is the object that is inserted into
 * the vector that is added to the combo Box, ultimatly displaying a drop-down list
 * of collections to choose from.
 */
public class Collection
{
	Collection(String sName, int iNumber)
	{
		m_sName = sName;
		m_iNumber = iNumber;
	}
	
	public String toString()
	{
		return m_sName;
	}
	
	public String		m_sName;
	public int			m_iNumber;
}
