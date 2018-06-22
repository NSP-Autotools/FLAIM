//------------------------------------------------------------------------------
// Desc:	Node Dialog
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

import xflaim.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;
import java.awt.*;
import java.util.Vector;

/**
 * To change the template for this generated type comment go to
 * Window->Preferences->Java->Code Generation->Code and Comments
 */
public class NodeDialog extends JDialog implements Runnable, ActionListener, ListSelectionListener, KeyListener
{
	private XEdit					m_owner;
	private DbSystem				m_dbSystem;
	private Db						m_jDb;
	private GridBagConstraints	m_gbc;
	private GridBagLayout		m_gbl;
	private long					m_lNodeId;
	private JList					m_NodeTypeList;
	private JList					m_NodeTagList;
	private JRadioButton			m_btnNewDocument;
	private JRadioButton			m_btnNewSibling;
	private JRadioButton			m_btnNewChild;
	private JRadioButton			m_btnAnnotation;
	private final String			m_sNewDocument = "New Document";
	private final String			m_sNewSibling = "New Sibling";
	private final String			m_sNewChild = "New Child";
	private final String			m_sAnnotation = "Annotation";
	private JButton				m_btnCancel;
	private JButton				m_btnAdd;
	private JButton				m_btnDone;
	private JTextField			m_textField;
	private JLabel				m_nodeTagLabel;
	private JScrollPane			m_scrollPane;
	private int					m_iCollection;
	private boolean				m_bTypeSelected = false;
	private boolean				m_bTagRequired = false;
	private boolean				m_bTagSelected = false;
	private boolean				m_bValueRequired = false;
	private boolean				m_bValueEntered = false;
	
	
	/**
	 * @param owner
	 * @throws java.awt.HeadlessException
	 */
	public NodeDialog(
		XEdit			owner,
		DbSystem		dbSystem,
		String		sDbFileName,
		long			lNodeId,
		int			iCollection) throws XFlaimException
	{
		super(owner, "XFLaim Node Dialog", true);
		
		m_owner = owner;
		m_dbSystem = dbSystem;
		m_lNodeId = lNodeId;
		m_iCollection = iCollection;
	
		// Open the database so we can have independant access
		m_jDb = m_dbSystem.dbOpen( sDbFileName, null, null, null, true);

		m_gbl = new GridBagLayout();
		m_gbc = new GridBagConstraints();
		m_gbc.insets = new Insets(3, 6, 3, 6);
		
		setDefaultCloseOperation( DISPOSE_ON_CLOSE);
		Container CP = getContentPane();
		CP.setLayout( m_gbl);
		
		// Select Node Type label
		JLabel lbl1 = new JLabel("Select Node Type");
		UITools.buildConstraints(m_gbc, 0, 0, 2, 1, 0, 0);
		m_gbc.anchor = GridBagConstraints.NORTHWEST;
		m_gbl.addLayoutComponent(lbl1, m_gbc);
		CP.add(lbl1);
		
		// Select Node Tag
		m_nodeTagLabel = new JLabel("Select Node Tag");
		m_nodeTagLabel.setVisible(false);
		UITools.buildConstraints(m_gbc, 3, 0, 2, 1, 0, 0);
		m_gbc.anchor = GridBagConstraints.NORTHWEST;
		m_gbl.addLayoutComponent(m_nodeTagLabel, m_gbc);
		CP.add(m_nodeTagLabel);

		// Select Node Type list box
		Vector vNodeList = new Vector();
		buildNodeTypeList(vNodeList);
		m_NodeTypeList = new JList(vNodeList);
		m_NodeTypeList.addListSelectionListener(this);
		JScrollPane sp = new JScrollPane(m_NodeTypeList);
		UITools.buildConstraints(m_gbc, 0, 1, 2, 1, 0, 0);
		m_gbc.anchor = GridBagConstraints.WEST;
		m_gbl.addLayoutComponent(sp, m_gbc);
		CP.add(sp);
		
		// Select Node Tag list box
		m_NodeTagList = new JList();
		m_NodeTagList.setEnabled(false); // Will be enabled later when a valid type is selected.
		m_NodeTagList.setVisible(false);
		m_NodeTagList.addListSelectionListener(this);
		m_scrollPane = new JScrollPane(m_NodeTagList);
		UITools.buildConstraints(m_gbc, 3, 1, 2, 1, 0, 0);
		m_gbc.anchor = GridBagConstraints.WEST;
		m_gbc.fill = GridBagConstraints.BOTH;
		m_gbl.addLayoutComponent(m_scrollPane, m_gbc);
		m_scrollPane.setVisible(false);
		CP.add(m_scrollPane);

		// Radio Button - New Document
		m_btnNewDocument = new JRadioButton(m_sNewDocument);
		m_btnNewDocument.setActionCommand(m_sNewDocument);
		m_btnNewDocument.setSelected(m_lNodeId > 0 ? false : true);
		UITools.buildConstraints(m_gbc, 0, 2, 1, 1, 0, 0);
		m_gbc.anchor = GridBagConstraints.WEST;
		m_gbc.fill = GridBagConstraints.NONE;
		m_gbl.addLayoutComponent(m_btnNewDocument, m_gbc);
		CP.add(m_btnNewDocument);

		// Radio button - New Sibling
		m_btnNewSibling = new JRadioButton(m_sNewSibling);
		m_btnNewSibling.setActionCommand(m_sNewSibling);
		m_btnNewSibling.setSelected(m_lNodeId > 0 ? true : false);
		if (m_lNodeId == 0)
		{
			m_btnNewSibling.setEnabled(false);
		}
		UITools.buildConstraints(m_gbc, 0, 3, 1, 1, 0, 0);
		m_gbc.anchor = GridBagConstraints.WEST;
		m_gbl.addLayoutComponent(m_btnNewSibling, m_gbc);
		CP.add(m_btnNewSibling);

		// Radio Button - New Child
		m_btnNewChild = new JRadioButton(m_sNewChild);
		m_btnNewChild.setActionCommand(m_sNewChild);
		m_btnNewChild.setSelected(false);
		if (m_lNodeId == 0)
		{
			m_btnNewChild.setEnabled(false);
		}
		UITools.buildConstraints(m_gbc, 0, 4, 1, 1, 0, 0);
		m_gbc.anchor = GridBagConstraints.WEST;
		m_gbl.addLayoutComponent(m_btnNewChild, m_gbc);
		CP.add(m_btnNewChild);

		// Radio Button - Annotation
		m_btnAnnotation = new JRadioButton(m_sAnnotation);
		m_btnAnnotation.setActionCommand(m_sAnnotation);
		m_btnAnnotation.setSelected(false);
		if (m_lNodeId == 0)
		{
			m_btnAnnotation.setEnabled(false);
		}
		UITools.buildConstraints(m_gbc, 0, 5, 1, 1, 0, 0);
		m_gbc.anchor = GridBagConstraints.WEST;
		m_gbl.addLayoutComponent(m_btnAnnotation, m_gbc);
		CP.add(m_btnAnnotation);

		// Group the Radio Buttons
		ButtonGroup btnGroup = new ButtonGroup();
		btnGroup.add(m_btnNewDocument);
		btnGroup.add(m_btnNewSibling);
		btnGroup.add(m_btnNewChild);
		btnGroup.add(m_btnAnnotation);

		// Enter new value
		JLabel lbl3 = new JLabel("Enter new value");
		UITools.buildConstraints(m_gbc, 0, 6, 2, 1, 0, 0);
		m_gbc.anchor = GridBagConstraints.WEST;
		m_gbl.addLayoutComponent(lbl3, m_gbc);
		CP.add(lbl3);
		
		// Text box to hold the new value
		m_textField = new JTextField();
		m_textField.setEditable(true);
		m_textField.addActionListener(this);
		m_textField.setActionCommand("Value");
		m_textField.addKeyListener(this);
		UITools.buildConstraints(m_gbc, 0, 7, 5, 1, 0, 0);
		m_gbc.anchor = GridBagConstraints.WEST;
		m_gbc.fill = GridBagConstraints.HORIZONTAL;
		m_gbl.addLayoutComponent(m_textField, m_gbc);
		CP.add(m_textField);
		

		// Cancel button
		m_btnCancel = new JButton("Cancel");
		m_btnCancel.setActionCommand("Cancel");
		m_btnCancel.addActionListener(this);
		UITools.buildConstraints(m_gbc, 2, 8, 1, 1, 0, 0);
		m_gbl.addLayoutComponent(m_btnCancel, m_gbc);
		CP.add(m_btnCancel);
		
		// Add button
		m_btnAdd = new JButton("Add");
		m_btnAdd.setActionCommand("Add");
		m_btnAdd.addActionListener(this);
		m_btnAdd.setEnabled(false);
		UITools.buildConstraints(m_gbc, 3, 8, 1, 1, 0, 0);
		m_gbl.addLayoutComponent(m_btnAdd, m_gbc);
		CP.add(m_btnAdd);
		
		// Done button
		m_btnDone = new JButton("Done");
		m_btnDone.setActionCommand("Done");
		m_btnDone.addActionListener(this);
		m_btnDone.setEnabled(false);
		UITools.buildConstraints(m_gbc, 4, 8, 1, 1, 0, 0);
		m_gbl.addLayoutComponent(m_btnDone, m_gbc);
		CP.add(m_btnDone);

		this.pack();
		this.setResizable(false);
	}

	/**
	 * @param m_NodeTypeList
	 */
	private void buildNodeTypeList(Vector vNodeList)
	{
		vNodeList.clear();
		vNodeList.add(new NodeType("Annotation", FlmDomNodeType.ANNOTATION_NODE));
		vNodeList.add(new NodeType("Attribute", FlmDomNodeType.ATTRIBUTE_NODE));
		vNodeList.add(new NodeType("CDATA Section", FlmDomNodeType.CDATA_SECTION_NODE));
		vNodeList.add(new NodeType("Comment", FlmDomNodeType.COMMENT_NODE));
		vNodeList.add(new NodeType("Data", FlmDomNodeType.DATA_NODE));
		vNodeList.add(new NodeType("Document", FlmDomNodeType.DOCUMENT_NODE));
		vNodeList.add(new NodeType("Element", FlmDomNodeType.ELEMENT_NODE));
		vNodeList.add(new NodeType("Processing Instruction", FlmDomNodeType.PROCESSING_INSTRUCTION_NODE));
	}

	/* (non-Javadoc)
	 * @see java.lang.Runnable#run()
	 */
	public void run()
	{
		this.setVisible(true);
	}

	/* (non-Javadoc)
	 * @see java.awt.event.ActionListener#actionPerformed(java.awt.event.ActionEvent)
	 */
	public void actionPerformed(ActionEvent e)
	{
		String str = e.getActionCommand();
		
		if (str.equals("Cancel"))
		{
			// Abort everything.
			this.dispose();
		}
		else if (str.equals("Add"))
		{
			// Update the database, but don't quit.
			if (m_btnNewDocument.isSelected())
			{
				try
				{
					NodeTag		ntg = (NodeTag)m_NodeTagList.getSelectedValue();
					int			iNumber = 0;
					
					if (ntg != null)
					{
						iNumber = ntg.m_iNumber;
					}

					m_lNodeId = m_owner.addDocument(
									m_iCollection,
									iNumber,
									m_textField.getText());
					if (m_lNodeId > 0)
					{
						m_btnNewSibling.setEnabled(true);
						m_btnNewChild.setEnabled(true);
						m_btnAnnotation.setEnabled(true);
						m_btnDone.setEnabled(true);
					}
				}
				catch (XFlaimException ee)
				{
					JOptionPane.showMessageDialog(
								this,
								"Database Exception occurred: " + ee.getMessage(),
								"Database Exception",
								JOptionPane.ERROR_MESSAGE);
				}
			}
			else if (m_btnAnnotation.isSelected())
			{
				try
				{
					m_lNodeId = m_owner.addAnnotation(
								m_iCollection,
								m_textField.getText(),
								m_lNodeId);
					if (m_lNodeId > 0)
					{
						m_btnNewSibling.setEnabled(true);
						m_btnNewChild.setEnabled(true);
						m_btnAnnotation.setEnabled(true);
						m_btnDone.setEnabled(true);
					}
				}
				catch (XFlaimException ex)
				{
					JOptionPane.showMessageDialog(
								this,
								"Database Exception occurred: " + ex.getMessage(),
								"Database Exception",
								JOptionPane.ERROR_MESSAGE);
				}
			}
			else
			{
				boolean		bIsSibling = false;
				if (m_btnNewSibling.isSelected())
				{
					bIsSibling = true;
				}

				try
				{
					NodeType	nt = (NodeType)m_NodeTypeList.getSelectedValue();					
					NodeTag		ntg = (NodeTag)m_NodeTagList.getSelectedValue();
					int			iNumber = 0;
					
					if (ntg != null)
					{
						iNumber = ntg.m_iNumber;
					}
					
					m_lNodeId = m_owner.addNode(
								m_iCollection,
								nt.m_iType,
								iNumber,
								m_textField.getText(),
								bIsSibling,
								m_lNodeId);
					if (m_lNodeId > 0)
					{
						m_btnNewSibling.setEnabled(true);
						m_btnNewChild.setEnabled(true);
						m_btnAnnotation.setEnabled(true);
						m_btnDone.setEnabled(true);
					}
				}
				catch (XFlaimException ex)
				{
					JOptionPane.showMessageDialog(
								this,
								"Database Exception occurred: " + ex.getMessage(),
								"Database Exception",
								JOptionPane.ERROR_MESSAGE);
				}
			}
		}
		else if (str.equals("Done"))
		{
			this.dispose();
		}
		else if (str.equals("Value"))
		{
			m_bValueEntered = true;
			enableButtons();
		}
	}

	/**
	 * Finalize method to ensure we close the local copy of the database.
	 */
	public void finalize()
	{
		if (m_jDb != null)
		{
			m_jDb.close();
		}
	}

	/* (non-Javadoc)
	 * @see javax.swing.event.ListSelectionListener#valueChanged(javax.swing.event.ListSelectionEvent)
	 */
	public void valueChanged(ListSelectionEvent e)
	{
		JList list = (JList)e.getSource();

		Object obj = list.getSelectedValue();
		
		if (obj instanceof NodeType)
		{
			NodeType nt = (NodeType)obj;

			switch(nt.m_iType)
			{
				case FlmDomNodeType.ATTRIBUTE_NODE:
				{
					if (buildAttributeTagList())
					{
						m_nodeTagLabel.setText("Select Attribute Tag");
						m_nodeTagLabel.setVisible(true);
						m_bTypeSelected = true;
						m_bTagRequired = true;
						m_bTagSelected = false;
						m_bValueRequired = true;
					}
					break;
				}
				case FlmDomNodeType.ELEMENT_NODE:
				{
					if (buildElementTagList())
					{
						m_nodeTagLabel.setText("Select Element Tag");
						m_nodeTagLabel.setVisible(true);
						m_bTagRequired = true;
						m_bTagSelected = false;
						m_bTypeSelected = true;
					}
					break;
				}
				case FlmDomNodeType.DOCUMENT_NODE: // Same as Element Node - Root node.
				{
					if (buildElementTagList())
					{
						m_nodeTagLabel.setText("Select Root Tag");
						m_nodeTagLabel.setVisible(true);
						m_bTagRequired = true;
						m_bTagSelected = false;
						m_bTypeSelected = true;
					}
					break;
				}
				case FlmDomNodeType.ANNOTATION_NODE:
				{
					m_nodeTagLabel.setVisible(false);
					m_NodeTagList.setVisible(false);
					m_bTypeSelected = true;
					m_bTagRequired = false;
					m_bTagSelected = false;
					m_bValueRequired = true;
					m_scrollPane.setVisible(false);
					break;
				}
				case FlmDomNodeType.CDATA_SECTION_NODE:
				{
					m_nodeTagLabel.setVisible(false);
					m_NodeTagList.setVisible(false);
					m_bValueRequired = true;
					m_bTypeSelected = true;
					m_bTagRequired = false;
					m_bTagSelected = false;
					m_scrollPane.setVisible(false);
					break;
				}
				case FlmDomNodeType.COMMENT_NODE:
				{
					m_nodeTagLabel.setVisible(false);
					m_NodeTagList.setVisible(false);
					m_bTypeSelected = true;
					m_bTagRequired = false;
					m_bTagSelected = false;
					m_bValueRequired = true;
					m_scrollPane.setVisible(false);
					break;
				}
				case FlmDomNodeType.DATA_NODE:
				{
					m_nodeTagLabel.setVisible(false);
					m_NodeTagList.setVisible(false);
					m_bTypeSelected = true;
					m_bTagRequired = false;
					m_bTagSelected = false;
					m_bValueRequired = true;
					m_scrollPane.setVisible(false);
					break;
				}
				case FlmDomNodeType.PROCESSING_INSTRUCTION_NODE:
				{
					m_nodeTagLabel.setVisible(false);
					m_NodeTagList.setVisible(false);
					m_bTypeSelected = true;
					m_bTagRequired = false;
					m_bTagSelected = false;
					m_bValueRequired = true;
					m_scrollPane.setVisible(false);
					break;
				}
				default:
				{
					JOptionPane.showMessageDialog(
								this,
								"Invalid DOM Node Type",
								"Processing Error",
								JOptionPane.WARNING_MESSAGE);
					break;
				}
			}
		}

		if (obj instanceof NodeTag)
		{
			m_bTagSelected = true;
		}
		enableButtons();
	}

	/**
	 * 
	 */
	private void enableButtons()
	{
		if (m_bTypeSelected)
		{
			if (m_bTagRequired)
			{
				if (m_bTagSelected)
				{
					if (m_bValueRequired)
					{
						if (m_bValueEntered)
						{
							m_btnAdd.setEnabled(true);
//							m_btnDone.setEnabled(true);
						}
						else
						{
							m_btnAdd.setEnabled(false);
							m_btnDone.setEnabled(false);
						}
					}
					else
					{
						m_btnAdd.setEnabled(true);
//						m_btnDone.setEnabled(true);
					}
				}
				else
				{
					m_btnAdd.setEnabled(false);
					m_btnDone.setEnabled(false);
				}
			}
			else
			{
				if (m_bValueRequired)
				{
					if (m_bValueEntered)
					{
						m_btnAdd.setEnabled(true);
//						m_btnDone.setEnabled(true);
					}
					else
					{
						m_btnAdd.setEnabled(false);
						m_btnDone.setEnabled(false);
					}
				}
				else
				{
					m_btnAdd.setEnabled(true);
//					m_btnDone.setEnabled(true);
				}
			}
		}
	}

	/**
	 * 
	 */
	private boolean buildElementTagList()
	{
		DataVector			SearchKey = null;
		DataVector			FoundKey = null;
		boolean				bFirst = true;
		int					iFlags;
		Vector				vElements = new Vector();

		try
		{

			SearchKey = m_dbSystem.createJDataVector();
			FoundKey = m_dbSystem.createJDataVector();

			// Setup the search key.
			m_jDb.keyRetrieve(FlmDictIndex.NAME_INDEX,
							  SearchKey,
							  KeyRetrieveFlags.FO_FIRST,
							  SearchKey);
			
			SearchKey.setLong(0, ReserveID.ELM_ELEMENT_TAG);
			SearchKey.setString(1, "a");
			
			for (;;)
			{
				if (bFirst)
				{
					bFirst = false;
					iFlags = KeyRetrieveFlags.FO_INCL;
					m_jDb.keyRetrieve(FlmDictIndex.NAME_INDEX, SearchKey, iFlags, FoundKey);
				
				}
				else
				{
					iFlags = KeyRetrieveFlags.FO_EXCL;
					m_jDb.keyRetrieve(FlmDictIndex.NAME_INDEX, FoundKey, iFlags, FoundKey);
				
				}
				
				if (FoundKey == null)
				{
					break;
				}
				
				if (FoundKey.getLong(0) != ReserveID.ELM_ELEMENT_TAG)
				{
					break;
				}
				String sName = FoundKey.getString(1);
				int iNumber = (int)FoundKey.getLong(3);
				vElements.add(new NodeTag("<" + sName + ">", iNumber));
			}
			
			if (vElements.size() > 0)
			{
				m_NodeTagList.removeAll();
				m_NodeTagList.setListData(vElements);
				m_NodeTagList.setEnabled(true);
				m_NodeTagList.setVisible(true);
				m_scrollPane.setVisible(true);
			}
		}
		catch (XFlaimException e)
		{
			JOptionPane.showMessageDialog(
						this,
						"Database Exception occured: " + e.getMessage(),
						"Database Exception",
						JOptionPane.ERROR_MESSAGE);
			m_NodeTagList.setEnabled(false);
			m_NodeTagList.setVisible(false);
			m_scrollPane.setVisible(false);
			return false;
		}
		return true;
	}

	/**
	 * 
	 */
	private boolean buildAttributeTagList()
	{
		DataVector			SearchKey = null;
		DataVector			FoundKey = null;
		boolean				bFirst = true;
		int					iFlags;
		Vector				vAttributes = new Vector();

		try
		{

			SearchKey = m_dbSystem.createJDataVector();
			FoundKey = m_dbSystem.createJDataVector();

			// Setup the search key.
			m_jDb.keyRetrieve(FlmDictIndex.NAME_INDEX,
							  SearchKey,
							  KeyRetrieveFlags.FO_FIRST,
							  SearchKey);
			
			SearchKey.setLong(0, ReserveID.ELM_ATTRIBUTE_TAG);
			SearchKey.setString(1, "a");
			
			for (;;)
			{
				if (bFirst)
				{
					bFirst = false;
					iFlags = KeyRetrieveFlags.FO_INCL;
					m_jDb.keyRetrieve(FlmDictIndex.NAME_INDEX, SearchKey, iFlags, FoundKey);
				
				}
				else
				{
					iFlags = KeyRetrieveFlags.FO_EXCL;
					m_jDb.keyRetrieve(FlmDictIndex.NAME_INDEX, FoundKey, iFlags, FoundKey);
				
				}
				
				if (FoundKey == null)
				{
					break;
				}
				
				if (FoundKey.getLong(0) != ReserveID.ELM_ATTRIBUTE_TAG)
				{
					break;
				}
				String sName = FoundKey.getString(1);
				int iNumber = (int)FoundKey.getLong(3);
				vAttributes.add(new NodeTag("<" + sName + ">", iNumber));
			}
			
			if (vAttributes.size() > 0)
			{
				m_NodeTagList.removeAll();
				m_NodeTagList.setListData(vAttributes);
				m_NodeTagList.setEnabled(true);
				m_NodeTagList.setVisible(true);
				m_scrollPane.setVisible(true);
			}
		}
		catch (XFlaimException e)
		{
			JOptionPane.showMessageDialog(
						this,
						"Database Exception occured: " + e.getMessage(),
						"Database Exception",
						JOptionPane.ERROR_MESSAGE);
			m_NodeTagList.setEnabled(false);
			m_NodeTagList.setVisible(false);
			m_scrollPane.setVisible(false);
			return false;
		}
		return true;
	}

	/* (non-Javadoc)
	 * @see java.awt.event.KeyListener#keyPressed(java.awt.event.KeyEvent)
	 */
	public void keyPressed(KeyEvent e)
	{
		// BNot implementing
	}

	/* (non-Javadoc)
	 * @see java.awt.event.KeyListener#keyReleased(java.awt.event.KeyEvent)
	 */
	public void keyReleased(KeyEvent e)
	{
		int i = m_textField.getText().length();
		if (i == 0)
		{
			m_bValueEntered = false;
		}
		else
		{
			m_bValueEntered = true;			
		}
		enableButtons();
		
	}

	/* (non-Javadoc)
	 * @see java.awt.event.KeyListener#keyTyped(java.awt.event.KeyEvent)
	 */
	public void keyTyped(KeyEvent e)
	{
		// Not implementing
	}
/*
	public static void main(String[] args)
	{
		Frame f = new Frame();
		NodeDialog nd = null;
//		DbSystem dbSystem;
//		try
//		{
//			dbSystem = new DbSystem();
//		}
//		catch (XFlaimException e)
//		{
//			e.printStackTrace();
//		}
//		String sDbFileName = "myxml.db";
//		NodePanel np = new NodePanel(null, 0);
		try
		{
			nd = new NodeDialog(f);
		}
		catch (XFlaimException e1)
		{
			e1.printStackTrace();
			System.exit(0);
		}
		
		Thread t = new Thread((Runnable)nd);
		
		t.start();
	}
*/
}
