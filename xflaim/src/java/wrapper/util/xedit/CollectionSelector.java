//------------------------------------------------------------------------------
// Desc:	Collection Selector
// Tabs:	3
//
// Copyright (c) 2003-2007 Novell, Inc. All Rights Reserved.
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
import javax.swing.*;
import java.util.*;

/**
 * The CollectionSelector is used to popup a dialog window that gives the
 * use a choice of collections to choose.  The Collection parameter passed
 * into the constructor will be initialized to hold the collection the user
 * has chosen when this object exits.  All of the work is done inthe constructor.
 * To use this class/object, simply create a new instance of it and check the
 * Collection object that was passed into the constructor.
 */
public class CollectionSelector extends JDialog implements ActionListener
{
	private JComboBox			m_cbCollection;
	private JButton			m_btnOkay;
	private JButton			m_btnCancel;
	private Collection		m_Collection;

	public CollectionSelector(
		Frame				owner,
		DbSystem			dbSystem,
		Db					jDb,
		Collection		collection)
	{
		super(owner, "Select Collection", true);

		Container				CP;		// The content pane for this dialog
		GridBagLayout			gridbag;
		GridBagConstraints	constraints = new GridBagConstraints();
		Vector					vCollections;
		DataVector				SearchKey = null;
		DataVector				FoundKey = null;
		// Coordinates for location this window in the center of its parent.
		Point					p;
		Dimension				d;
		int						x;
		int						y;

		setDefaultCloseOperation( DISPOSE_ON_CLOSE);
		CP = getContentPane();
		gridbag = new GridBagLayout(); 
		CP.setLayout( gridbag);
	
		m_Collection = collection;
	
		// Add the combobox.
		vCollections = new Vector();
		vCollections.add(new Collection("Default Data			", xflaim.Collections.DATA));
		vCollections.add(new Collection("Dictionary				", xflaim.Collections.DICTIONARY));

		// Now get all of the user defined collections from the database
		try
		{
			boolean		bFirst = true;
			int			iFlags;

			SearchKey = dbSystem.createJDataVector();
			FoundKey = dbSystem.createJDataVector();
			
			// Setup the search key.
			jDb.keyRetrieve(FlmDictIndex.NAME_INDEX,
							SearchKey,
							KeyRetrieveFlags.FO_FIRST,
							SearchKey);
			
			SearchKey.setLong(0, ReserveID.ELM_COLLECTION_TAG);
			SearchKey.setString(1, "a");
			
			for (;;)
			{
				if (bFirst)
				{
					bFirst = false;
					iFlags = KeyRetrieveFlags.FO_INCL;
					jDb.keyRetrieve(FlmDictIndex.NAME_INDEX, SearchKey, iFlags, FoundKey);
				
				}
				else
				{
					iFlags = KeyRetrieveFlags.FO_EXCL;
					jDb.keyRetrieve(FlmDictIndex.NAME_INDEX, FoundKey, iFlags, FoundKey);
				
				}
				
				if (FoundKey == null)
				{
					break;
				}
				
				if (FoundKey.getLong(0) != ReserveID.ELM_COLLECTION_TAG)
				{
					break;
				}
				String sName = FoundKey.getString(1);
				int iNumber = (int)FoundKey.getLong(3);
				vCollections.add(new Collection(sName, iNumber));
				
			}
		}
		catch (XFlaimException e)
		{
			// Leave it for now.
			System.out.println(e.getMessage());
		}
		
		




		m_cbCollection = new JComboBox( vCollections);
		m_cbCollection.setSelectedIndex(0);
		m_cbCollection.addActionListener(this);

		UITools.buildConstraints(constraints, 0, 0, 2, 1, 0, 0);		

		gridbag.setConstraints( m_cbCollection, constraints);
	
		CP.add( m_cbCollection);
		
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
		if (obj == m_cbCollection || obj == m_btnOkay)
		{
			Collection coll = (Collection)m_cbCollection.getSelectedItem();
			m_Collection.m_iNumber = coll.m_iNumber;
			m_Collection.m_sName = new String(coll.m_sName);
			if (obj == m_btnOkay)
			{
				setVisible(false);
				dispose();
			}
		}
		else
		{
			setVisible(false);
			dispose();
		}
	}

}
