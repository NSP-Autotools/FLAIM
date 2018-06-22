//------------------------------------------------------------------------------
// Desc:	Status
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
import javax.swing.JLabel;

/**
 * To change the template for this generated type comment go to
 * Window->Preferences->Java->Code Generation->Code and Comments
 */
public class Status 
{
	private boolean			m_bDatabaseOpen;
	private String			m_sDatabasePath;
	private JLabel			m_label;
	private long			m_lDocId;
	private long			m_lNodeId;
	private boolean			m_bTransaction = false;
	public int				m_iCurrentNode = -2;
	public int				m_iLastNode = -2;
	
	
	// Constructor
	Status()
	{
		m_bDatabaseOpen = false;
		m_sDatabasePath = null;
		m_label = null;
		m_lDocId = 0;
		m_lNodeId = 0;
	}
	
	Status(
		JLabel		label)
	{
		m_bDatabaseOpen = false;
		m_sDatabasePath = null;
		m_lDocId = 0;
		m_lNodeId = 0;
		m_label = label;
		updateLabel();
	}
	
	public void setDatabaseOpen(
		boolean		bOpen)
	{
		m_bDatabaseOpen = bOpen;
		updateLabel();
	}
	
	public boolean getDatabaseOpen()
	{
		return m_bDatabaseOpen;
	}
	
	public void setDatabasePath(
		String		sDatabasePath)
	{
		m_sDatabasePath = sDatabasePath;
		updateLabel();
	}
	
	public String getDatabasePath()
	{
		return m_sDatabasePath;
	}
	
	public void setDocId(
		long		lDocId)
	{
		m_lDocId = lDocId;
		updateLabel();
	}
	
	public long getDocId()
	{
		return m_lDocId;
	}
	
	public void setNodeId(
		long		lNodeId)
	{
		m_lNodeId = lNodeId;
		updateLabel();
	}
	
	public long getNodeId()
	{
		return m_lNodeId;
	}

	public void setLabel(
		JLabel		label)
	{
		m_label = label;
		updateLabel();
	}
	
	public void setTransaction(
		boolean		bTransaction)
	{
		m_bTransaction = bTransaction;
		updateLabel();
	}
	
	public boolean getTransaction()
	{
		return m_bTransaction;
	}
	
	public void updateLabel()
	{
		String	sLabelText;
		String	sDocNode = "";
		String	sTransaction = m_bTransaction ? "Update" : "";
		
		if (m_label != null)
		{
			if (m_bDatabaseOpen)
			{
				sLabelText = "Database: " + m_sDatabasePath;

				if (m_lDocId > 0 && m_lNodeId > 0)
				{
					sDocNode = new String("\t" + new Long(m_lDocId).toString() +
									  "/" + new Long(m_lNodeId).toString());
				}
			}
			else
			{
				sLabelText = "No Database";
			}
//			m_label.setText(sLabelText + "            " + sDocNode + "   [" + m_iCurrentNode + "/" + m_iLastNode + "]");
			m_label.setText(sLabelText + "            " + sDocNode + "          " + sTransaction);
		}
	}
}
