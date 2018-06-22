//------------------------------------------------------------------------------
// Desc:	Open Document Selector
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

/**
 * To change the template for this generated type comment go to
 * Window->Preferences->Java->Code Generation->Code and Comments
 */

package xedit;
import xedit.UITools;

import java.awt.Container;
import java.awt.Dimension;
import java.awt.Frame;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Point;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Vector;

import javax.swing.JButton;
import javax.swing.JList;
import javax.swing.JDialog;
import javax.swing.JScrollPane;

/**
 * To change the template for this generated type comment go to
 * Window->Preferences->Java->Code Generation->Code and Comments
 */
public class OpenDocumentSelector extends JDialog implements ActionListener
{

	private JList				m_lstList;
	private JButton				m_btnOkay;
	private JButton				m_btnCancel;
	private int					m_iItem;

	public OpenDocumentSelector(
		Frame			owner,
		Vector			vDocList)
	{
		super(owner, "Select Open Document", true);

		Container				CP;		// The content pane for this dialog
		GridBagLayout			gridbag;
		GridBagConstraints		constraints = new GridBagConstraints();

		// Coordinates for locating this window in the center of its parent.
		Point					p;
		Dimension				d;
		int						x;
		int						y;

		setDefaultCloseOperation( DISPOSE_ON_CLOSE);
		CP = getContentPane();
		gridbag = new GridBagLayout(); 
		CP.setLayout( gridbag);
	
		m_iItem = -1;

		// Add the list
		m_lstList = new JList( vDocList);
		m_lstList.setSelectedIndex(0);

		JScrollPane scrollPane = new JScrollPane();
		scrollPane.getViewport().setView(m_lstList);

		UITools.buildConstraints(constraints, 0, 0, 2, 4, 100, 100);
		constraints.anchor = GridBagConstraints.NORTHWEST;
		constraints.fill = GridBagConstraints.BOTH;		

		gridbag.setConstraints( scrollPane, constraints);
	
		CP.add( scrollPane);
		
		// Add the Okay button
		m_btnOkay = new JButton("Okay");
		m_btnOkay.setDefaultCapable(true);
		m_btnOkay.addActionListener(this);

		UITools.buildConstraints(constraints, 0, 5, 1, 1, 90, 100);
		constraints.anchor = GridBagConstraints.EAST;
		constraints.fill = GridBagConstraints.NONE;		

		gridbag.setConstraints( m_btnOkay, constraints);
		
		CP.add( m_btnOkay);
		
		// Add the Cancel button
		m_btnCancel = new JButton("Cancel");
		m_btnCancel.addActionListener(this);
		
		UITools.buildConstraints(constraints, 1, 5, 1, 1, 10, 0);
		constraints.anchor = GridBagConstraints.WEST;

		gridbag.setConstraints( m_btnCancel, constraints);
		
		CP.add( m_btnCancel);

		setSize(Math.max( 400, 200), 200);

		p = owner.getLocationOnScreen();
		d = owner.getSize();
		x = (d.width - 400) / 2;
		y = (d.height - 300) / 2;
		setLocation(Math.max(0, p.x + x), Math.max(0, p.y + y));
	}

	public int showDialog()
	{
		setVisible( true);
		return m_iItem;
	}

	/* (non-Javadoc)
	 * @see java.awt.event.ActionListener#actionPerformed(java.awt.event.ActionEvent)
	 */
	public void actionPerformed(ActionEvent e)
	{
		Object obj = (Object)e.getSource();
		if (obj == m_btnOkay)
		{
			m_iItem = m_lstList.getSelectedIndex();
			setVisible(false);
			dispose();
		}
		else
		{
			setVisible(false);
			dispose();
		}
	}
}
