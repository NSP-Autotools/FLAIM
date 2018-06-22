//------------------------------------------------------------------------------
// Desc:	Edit Value Dialog
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

package xedit;

import java.awt.Container;
import java.awt.Dimension;
import java.awt.Frame;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Point;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JScrollPane;
import javax.swing.JTextField;

/**
 * To change the template for this generated type comment go to
 * Window->Preferences->Java->Code Generation->Code and Comments
 */
public class EditValueDialog extends JDialog implements ActionListener
{
	private JButton				m_btnOkay;
	private JButton				m_btnCancel;
	private JTextField			m_textField;
	private NodeValue			m_NodeValue;

	/**
	 * A class for editing the value of a DOM Node.  All values aer treated as strings.
	 * @param owner
	 * @param sValue
	 */
	public EditValueDialog(
		String		sTitle,
		Frame		owner,
		NodeValue	NodeValue)
	{
		super(owner, sTitle, true);
		
		m_NodeValue = NodeValue;

		Container				CP;		// The content pane for this dialog
		GridBagLayout			gridbag;
		GridBagConstraints		constraints = new GridBagConstraints();
		// Coordinates for location this window in the center of its parent.
		Point					p;
		Dimension				d;
		int						x;
		int						y;

		setDefaultCloseOperation( DISPOSE_ON_CLOSE);
		CP = getContentPane();
		gridbag = new GridBagLayout(); 
		CP.setLayout( gridbag);
	
		m_textField = new JTextField(NodeValue.getValue());
		m_textField.setEditable(true);
		JScrollPane sp = new JScrollPane(m_textField);
		
		UITools.buildConstraints(constraints, 0, 0, 3, 1, 100, 100);
		constraints.anchor = GridBagConstraints.NORTHWEST;
		constraints.fill = GridBagConstraints.BOTH;		
		gridbag.setConstraints( sp, constraints);
		
		CP.add( sp);
		
		
		// Add the Okay button
		m_btnOkay = new JButton("Okay");
		m_btnOkay.setDefaultCapable(true);
		m_btnOkay.addActionListener(this);

		UITools.buildConstraints(constraints, 1, 1, 1, 1, 60, 100);		
		constraints.anchor = GridBagConstraints.CENTER;
		constraints.fill = GridBagConstraints.NONE;
		gridbag.setConstraints( m_btnOkay, constraints);
		
		CP.add( m_btnOkay);
		
		// Add the Cancel button
		m_btnCancel = new JButton("Cancel");
		m_btnCancel.addActionListener(this);
		
		UITools.buildConstraints(constraints, 2, 1, 1, 1, 40, 0);

		gridbag.setConstraints( m_btnCancel, constraints);
		
		CP.add( m_btnCancel);

		setSize(200, 100);

		p = owner.getLocationOnScreen();
		d = owner.getSize();
		x = (d.width - 200) / 2;
		y = (d.height - 100) / 2;
		setLocation(Math.max(0, p.x + x), Math.max(0, p.y + y));
		setVisible( true);
	}

	/* (non-Javadoc)
	 * @see java.awt.event.ActionListener#actionPerformed(java.awt.event.ActionEvent)
	 */
	public void actionPerformed(ActionEvent e)
	{
		Object obj = (Object)e.getSource();
		if (obj == m_btnOkay)
		{
			m_NodeValue.setValue(m_textField.getText());
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
