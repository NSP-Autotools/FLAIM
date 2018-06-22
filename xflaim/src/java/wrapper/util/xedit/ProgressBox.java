//------------------------------------------------------------------------------
// Desc:	Progress Box
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

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import xedit.UITools;

/**
 * To change the template for this generated type comment go to
 * Window->Preferences->Java->Code Generation->Code and Comments
 */
public class ProgressBox extends JDialog implements ActionListener
{
	private JLabel				m_label1;
	private JLabel				m_label2;
	private JCheckBox			m_checkBox;
	private JButton				m_button;
	private JProgressBar		m_bar;
	private boolean				m_bCloseOnExit = false;
	private boolean				m_bToggled = false;
	private boolean				m_bCancelPressed = false;
	
	/**
	 * @throws java.awt.HeadlessException
	 */
	public ProgressBox(
				Frame		owner,
				int			iWidth,
				int			iHeight,
				String		label1,
				String		label2,
				int			iMin,
				int			iMax)
	{
		super();
		JPanel					p1;
		JPanel					p2;
		JPanel					p3;
		JPanel					p4;
		Container				CP;		// The content pane for this dialog
		// Coordinates for location this window in the center of its parent.
		Point							p;
		Dimension					d;
		int								x;
		int								y;
		
		m_label1 = new JLabel(label1);
		m_label2 = new JLabel(label2);

		setDefaultCloseOperation( DISPOSE_ON_CLOSE);
		CP = getContentPane();
		CP.setLayout( new GridLayout( 4, 1));

		p1 = new JPanel();
		p1.add(m_label1);
		CP.add(p1);
		
		p2 = new JPanel();
		p2.add(m_label2);
		CP.add(p2);

		m_bar = new JProgressBar(iMin, iMax);
		m_bar.setStringPainted(true);
		if (iMax == 0)
		{
			m_bar.setIndeterminate(true);
		}
		GridBagLayout gbl = new GridBagLayout();
		GridBagConstraints gbc = new GridBagConstraints();
		UITools.buildConstraints(gbc, 0, 0, 0, 0, 100, 100);
		gbc.fill = GridBagConstraints.BOTH;
		gbl.setConstraints(m_bar, gbc);
		p3 = new JPanel();
		p3.setLayout(gbl);
		p3.add(m_bar);
		CP.add(p3);
		
		p4 = new JPanel();
		GridBagLayout grid = new GridBagLayout();		
		p4.setLayout(grid);
		
		m_checkBox = new JCheckBox();
		m_checkBox.setEnabled(true);
		m_checkBox.addActionListener(this);
		UITools.buildConstraints(gbc, 0, 0, 1, 1, 10, 100);
		gbc.anchor = GridBagConstraints.WEST;
		gbc.fill = GridBagConstraints.NONE;
		grid.setConstraints(m_checkBox, gbc);
		p4.add(m_checkBox);

		JLabel lbl1 = new JLabel("Close when finished");
		UITools.buildConstraints(gbc, 1, 0, 1, 2, 60, 0);
		gbc.anchor = GridBagConstraints.WEST;
		gbc.fill = GridBagConstraints.NONE;
		grid.setConstraints(lbl1, gbc);
		p4.add(lbl1);
		
		m_button = new JButton("Cancel");
		m_button.addActionListener(this);
		UITools.buildConstraints(gbc, 2, 0, 1, 1, 30, 0);
		gbc.anchor = GridBagConstraints.EAST;
		gbc.fill = GridBagConstraints.NONE;
		grid.setConstraints(m_button, gbc);
		p4.add(m_button);
		
		CP.add(p4);
		
		setSize(iWidth,iHeight);
		
		if (owner != null)
		{
			p = owner.getLocationOnScreen();
			d = owner.getSize();
			x = (d.width - iWidth) / 2;
			y = (d.height - iHeight) / 2;
			setLocation(Math.max(0, p.x + x), Math.max(0, p.y + y));
		}

		setVisible(true);
	}


	/* (non-Javadoc)
	 * @see java.awt.event.ActionListener#actionPerformed(java.awt.event.ActionEvent)
	 */
	public void actionPerformed(ActionEvent e)
	{
		Object obj = e.getSource();
		if (obj == m_checkBox)
		{
			m_bCloseOnExit = !m_bCloseOnExit;
		}
		else if (obj == m_button)
		{
			if (m_button.getText().equals("Cancel"))
			{
				m_bCancelPressed = true;
			}
			else
			{
				dispose();
			}
		}		
	}
	
	public void setLabel1( String newLabel1)
	{
		m_label1.setText( newLabel1);
	}
	
	public void setLabel2( String newLabel2)
	{
		m_label2.setText( newLabel2);
	}
	
	public void updateProgress( int iValue)
	{
		m_bar.setValue( iValue);
		if (iValue >= m_bar.getMaximum() && m_bar.getMaximum() > 0)
		{
			if (m_bCloseOnExit)
			{
				dispose();
			}
			else
			{
				toggleOkCancelButton();
			}
		}

	}
	
	public void terminate()
	{
		dispose();
	}

	private void toggleOkCancelButton()
	{
		if (m_bToggled)
		{
			m_button.setText("Cancel");
			m_bToggled = false;
		}
		else
		{
			m_button.setText("Okay");
			m_bToggled = true;		
		}
	}
	
	public boolean Cancelled()
	{
		return m_bCancelPressed;
	}

	public void setMax( int iMax)
	{
		m_bar.setMaximum( iMax);
	}

	public static void main(String[] args)
	{
		ProgressBox b = new ProgressBox(null, 400, 160, "Label 1Label 1Label 1Label 1Label 1Label 1Label 1",
															   "Label 2Label 2Label 2Label 2Label 2Label 2Label 2",
															    0, 999); 
															    
		for (int i = 0; i < 1000; i++)
		{
			if (i % 100 == 0)
			{
				b.setLabel1( "Importing file " + i + "of 1000");
			}
			b.setLabel2("Filename: xxxx" + i + ".xml");
			b.updateProgress(i);
			if (b.Cancelled())
			{
				b.terminate();
				break;
			}
			try
			{
				Thread.sleep(10);
			}
			catch (Exception e)
			{
				// Do nothing...
			}
		}
		System.exit(0);
	}

}
