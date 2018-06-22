//------------------------------------------------------------------------------
// Desc:	Document Selector
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

import xedit.*;
import xflaim.*;
import java.awt.*;
import java.awt.event.*;
import java.util.Vector;
import javax.swing.*;

/**
 * The DocumentSelector class provides a means of presenting a list of
 * documents in the database to the user.  It will popup a window with a list
 * of documents, showing their root element.
 */
public class DocumentSelector extends JDialog implements ActionListener
{

	private JList				m_lstList;
	private JButton				m_btnOkay;
	private JButton				m_btnCancel;
	private Document			m_Document;

	public DocumentSelector(
		Frame			owner,
		Db				jDb,
		int			iCollection,
		Document		document)
	{
		super(owner, "Select Document", true);

		Container				CP;		// The content pane for this dialog
		GridBagLayout			gridbag;
		GridBagConstraints		constraints = new GridBagConstraints();
		Vector					vDocuments;
		int						iLength = 0;

		// Coordinates for locating this window in the center of its parent.
		Point						p;
		Dimension				d;
		int						x;
		int						y;
		boolean					bFirst = true;
		DOMNode					jDoc = null;

		setDefaultCloseOperation( DISPOSE_ON_CLOSE);
		CP = getContentPane();
		gridbag = new GridBagLayout(); 
		CP.setLayout( gridbag);
	
		m_Document = document;
	
		// Add the combobox.
		vDocuments = new Vector();
		// Get the list of documents
		for (;;)
		{
			try
			{
				if (bFirst)
				{
					jDoc = jDb.getFirstDocument(iCollection, null);
					bFirst = false;
				}
				else
				{
					jDoc = jDoc.getNextDocument(jDoc);
				}
				String sName = new String("<" + jDoc.getLocalName() + ">");
				long lDocId = jDoc.getNodeId();
				vDocuments.add(new Document(sName, lDocId, iCollection));
				if (sName.length() > iLength)
				{
					iLength = sName.length();
				}
			}
			catch (XFlaimException e)
			{
				if (e.getRCode() == RCODE.NE_XFLM_DOM_NODE_NOT_FOUND ||
					e.getRCode() == RCODE.NE_XFLM_NOT_FOUND)
				{
					break;
				}
			}
		}

		m_lstList = new JList( vDocuments);
		m_lstList.setSelectedIndex(0);
		//m_lstList.addActionListener(this);
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

		setSize(Math.max( iLength, 200), 200);

		p = owner.getLocationOnScreen();
		d = owner.getSize();
		x = (d.width - Math.max( iLength, 200)) / 2;
		y = (d.height - 200) / 2;
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
			Document doc = (Document)m_lstList.getSelectedValue();
			m_Document.m_lDocId = doc.m_lDocId;
			m_Document.m_sName = new String(doc.m_sName);
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
