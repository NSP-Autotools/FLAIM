//------------------------------------------------------------------------------
// Desc:	Attribute Selector
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

import xflaim.*;
import java.awt.*;
import java.awt.event.*;
import java.util.Vector;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JDialog;

/**
 * To change the template for this generated type comment go to
 * Window->Preferences->Java->Code Generation->Code and Comments
 */
public class AttributeSelector extends JDialog implements ActionListener
{

	private JComboBox			m_cbAttribute;
	private JButton				m_btnOkay;
	private JButton				m_btnCancel;
	private Attribute			m_Attribute;

	/**
	 * @param owner
	 * @throws java.awt.HeadlessException
	 */
	public AttributeSelector(
	Frame			owner,
	DOMNode		jRefNode,
	Attribute	attribute)
	{
		super(owner, "Select Attribute", true);
		Container				CP;		// The content pane for this dialog
		GridBagLayout			gridbag;
		GridBagConstraints		constraints = new GridBagConstraints();
		Vector					vAttributes;
		// Coordinates for location this window in the center of its parent.
		Point					p;
		Dimension				d;
		int						x;
		int						y;

		setDefaultCloseOperation( DISPOSE_ON_CLOSE);
		CP = getContentPane();
		gridbag = new GridBagLayout(); 
		CP.setLayout( gridbag);
	
		m_Attribute = attribute;
	
		// Add the combobox.
		vAttributes = new Vector();

		// Now get all of the attributes from the node
		try
		{
			boolean		bFirst = true;
			DOMNode		jAttr = null;
			
			for (;;)
			{
				if (bFirst)
				{
					bFirst = false;
					jAttr = jRefNode.getFirstAttribute(jAttr);
				}
				else
				{
					try
					{
						jAttr = jAttr.getNextSibling(jAttr);
					}
					catch (XFlaimException ee)
					{
						if (ee.getRCode() == RCODE.NE_XFLM_DOM_NODE_NOT_FOUND)
						{
							break;
						}
						else
						{
							// Leave it for now.
							System.out.println(ee.getMessage());
						}
					}
				}
				
				String sDesc = "<" + jAttr.getLocalName() + ">";
				long lNodeId = jAttr.getNodeId();
				vAttributes.add(new Attribute(sDesc, lNodeId));
				
			}
		}
		catch (XFlaimException e)
		{
			// Leave it for now.
			System.out.println(e.getMessage());
		}
		

		m_cbAttribute = new JComboBox( vAttributes);
		m_cbAttribute.setSelectedIndex(0);
		m_cbAttribute.addActionListener(this);

		UITools.buildConstraints(constraints, 0, 0, 2, 1, 0, 0);		

		gridbag.setConstraints( m_cbAttribute, constraints);
	
		CP.add( m_cbAttribute);
		
		// Add the Okay button
		m_btnOkay = new JButton("Okay");
		m_btnOkay.setDefaultCapable(true);
		m_btnOkay.addActionListener(this);

		UITools.buildConstraints(constraints, 0, 1, 1, 1, 60, 100);		

		gridbag.setConstraints( m_btnOkay, constraints);
		
		CP.add( m_btnOkay);
		
		// Add the Cancel button
		m_btnCancel = new JButton("Cancel");
		m_btnCancel.addActionListener(this);
		
		UITools.buildConstraints(constraints, 1, 1, 1, 1, 40, 0);

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
		if (obj == m_cbAttribute || obj == m_btnOkay)
		{
			Attribute attr = (Attribute)m_cbAttribute.getSelectedItem();
			m_Attribute.m_lNodeId = attr.m_lNodeId;
			m_Attribute.m_sDesc = new String(attr.m_sDesc);
			if (obj == m_btnOkay)
			{
				setVisible(false);
				dispose();
			}
		}
		else
		{
			m_Attribute.m_lNodeId = -1; // Make sure the caller knows we cancelled
			setVisible(false);
			dispose();
		}
	}
}
