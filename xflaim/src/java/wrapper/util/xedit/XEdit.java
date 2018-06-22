//------------------------------------------------------------------------------
// Desc:	XEdit
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
import java.io.File;
import java.util.Vector;
import javax.swing.*;

/**
 * To change the template for this generated type comment go to
 * Window->Preferences->Java->Code Generation->Code and Comments
 */
public class XEdit extends JFrame implements Runnable, ActionListener, MouseListener, KeyListener
{
	// Package members
	DbSystem 					m_dbSystem;
	Db								m_jDb;

	// Private member variables.
	private JPanel				m_mainDisplay;
	private JPanel				m_mainPanel;
	private Status				m_status;
	private String				m_sDatabaseName;
	private String				m_sDirectory;
	private Vector				m_vDocList;
	private Vector				m_vNodeList;
	private int					m_iExpandedRow;
	private int					m_iLastNode;
	private final int			FIRST_NODE = 0;
	private final int			LAST_NODE = 31;
	private final int			NO_NODES = -1;
	private NodePanel			m_selected;
	private boolean				m_bDocList = false;
	private int					m_iCollection;
	private Document			m_doc;
	private XEdit				m_parent = null;
	private JPopupMenu			m_popup = null;
	private boolean				m_bMustExit = false;


	/**
	 * Basic constructor
	 * @param title
	 */
	XEdit(
		String title)
	{
		super(title);
		init();
	}
	
	/**
	 * Collection List constructor
	 * @param parent
	 * @param title
	 * @param jDb
	 * @param iCollection
	 * @param doc
	 */
	XEdit(
		XEdit			parent,
		String		title,
		Db				jDb,
		int			iCollection,
		Document		doc)
	{
		super(title);
		m_bDocList = true;
		m_parent = parent;

		init();

		m_iCollection = iCollection;
		m_doc = doc;
		m_jDb = jDb;

		if (!showDocumentList())
		{
			JOptionPane.showMessageDialog(
						this,
						"Collection contains no documents",
						"Empty Collection",
						JOptionPane.INFORMATION_MESSAGE);
			m_bMustExit = true;
		}
	}

	/*-------------------------------------------------------------------------
	 * Desc:	Method to execute the common initialization tasks.
	 *-----------------------------------------------------------------------*/
	private void init()
	{
		setSize(660, 625);
		this.setResizable(false);
		this.addKeyListener(this);

		setDefaultCloseOperation(EXIT_ON_CLOSE);

		m_sDatabaseName = null;
		m_sDirectory = System.getProperty("user.dir");
		m_selected = null;

		// Create a dbSystem object.
		try
		{
			m_dbSystem = new DbSystem();
		}
		catch (XFlaimException e)
		{
			e.printStackTrace();
			System.exit(0);
		}

		// Initialize the jDb to null;		
		m_jDb = null;

		// Create the main panel that will hold everything else.
		m_mainPanel = new JPanel();
		
		// Set the layout manager.
		m_mainPanel.setLayout(new BorderLayout());

		// Create the menu and add it to the panel
		if (!m_bDocList)
		{
			JMenuBar menuBar = makeMainMenu();
			m_mainPanel.add("North", menuBar);
		}

		// Create the status bar and add it to the panel
		JLabel statusBar = new JLabel((m_bDocList
									? "Document List"
									: "No Database"));
		m_mainPanel.add("South", statusBar);
		// New Status object to track our status.

		m_status = new Status(statusBar);


		// Create the panel to display the documents and the scrollable panel and add them to the
		// main panel.

		m_mainDisplay = new JPanel();

		m_mainDisplay.setLayout(new GridLayout(32, 1));
		JScrollPane scrollPane = new JScrollPane(m_mainDisplay);
		scrollPane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_NEVER);

		m_mainPanel.add("Center", scrollPane);
		
		m_vDocList = new Vector();
		m_vNodeList = new Vector();

		// Fill up the display with blank panels.
		for (int i = FIRST_NODE; i <= LAST_NODE; i++)
		{
			NodePanel np = new NodePanel(this, i);
			m_mainDisplay.add(np);
			m_vNodeList.addElement(np);
		}
		m_iLastNode = NO_NODES;
		
		m_status.m_iCurrentNode = NO_NODES;
		m_status.m_iLastNode = m_iLastNode;
		m_status.updateLabel();

		// Make it all stick together...
		setContentPane(m_mainPanel);
	}

	/*=========================================================================
	 * Desc:	Method to return the size (Dimension) of the main display panel.
	 *=======================================================================*/
	public Dimension getSizeOfDisplay()
	{
		return m_mainDisplay.getSize();
	}

	/*=========================================================================
	 * Desc:	Method to handle the events from the menu.
	 *=======================================================================*/
	public void actionPerformed(ActionEvent e)
	{
		JMenuItem		source = (JMenuItem) (e.getSource());

		String label = source.getText();

		// Are we exiting?
		if (label.equals("Exit"))
		{
			System.exit(0);
		}
		else if (label.equals("Open"))
		{
			openDatabase();
		}
		else if (label.equals("New"))
		{
			createDatabase();
		}
		else if (label.equals("Close"))
		{
			closeDatabase();
		}
		else if (label.equals("Remove"))
		{
			removeDatabase();
		}
		else if (label.equals("First Document"))
		{
			getFirstDocument();
		}
		else if (label.equals("Open Document"))
		{
			openDocument();
		}
		else if (label.equals("Create Document"))
		{
			createDocument(m_selected);
		}
		else if (label.equals("Close Document"))
		{
			if (m_popup != null)
			{
				NodePanel np = m_selected;
				closeDocument(np);
			}
			else
			{
				closeDocument();
			}
		}
		else if (label.equals("Import Document(s)"))
		{
			importDocument();
		}
		else if (label.equals("Collapse Node"))
		{
			NodePanel np = m_selected;
			if (np.isExpanded())
			{
				collapseNode(np);
			}
		}
		else if (label.equals("Expand Node"))
		{
			NodePanel np = m_selected;
			if (!np.isExpanded())
			{
				expandNode(np, true);
			}
		}
		else if (label.equals("Delete Node"))
		{
			deleteNode(m_selected);
		}
		else if (label.equals("Delete Attribute"))
		{
			deleteAttribute(m_selected);
		}
		else if (label.equals("Delete Annotation"))
		{
			deleteAnnotation(m_selected);
		}
		else if (label.equals("Edit Value"))
		{
			editValue(m_selected);
		}
		else if (label.equals("Edit Attribute"))
		{
			editAttribute(m_selected);
		}
		else if (label.equals("Edit Annotation"))
		{
			editAnnotation(m_selected);
		}
		else if (label.equals("Add Node"))
		{
			createNode(m_selected);
		}
		else if (label.equals("Begin Transaction"))
		{
			beginTransaction();
		}
		else if (label.equals("Commit Transaction"))
		{
			commitTransaction();
		}
		else if (label.equals("Abort Transaction"))
		{
			abortTransaction();
		}

		// If the popup menu is enabled, let's get rid of it.
		if (m_popup != null)
		{
			m_popup = null;
		}
	}

	/**
	 * @param np
	 */
	private void deleteAnnotation(
		NodePanel		np)
	{
		DOMNode			jNode = null;
		DOMNode			jAnnot = null;
		boolean			bTransBegun = false;
		String			sMessage;
	 	
		try
		{
			jNode = m_jDb.getNode(np.getCollection(), np.getNodeId(), null);
			if (!jNode.hasAnnotation())
			{
				return;
			}
			
			jAnnot = jNode.getAnnotation(jAnnot);
			sMessage = new String("Delete annotation: " + jAnnot.getString());

			if (JOptionPane.showConfirmDialog(
							this,
							sMessage) == JOptionPane.OK_OPTION)
			{
				// Begin a transaction
				if (!m_status.getTransaction())
				{
					m_jDb.transBegin(TransactionType.UPDATE_TRANS, 0, 0);
					bTransBegun = true;
				}
	 		
				// Delete the node.
				jAnnot.deleteNode();
	 		
				// Commit
				if (bTransBegun)
				{
					m_jDb.transCommit();
					bTransBegun = false;
				}

				// Refresh the current row
				if (np.isExpanded() && np.isClosing())
				{
					collapseNode(np);
					expandNode(m_selected, true);
				}
				else
				{
					np.buildLabel(jNode, np.isExpanded(), np.isClosing(), false);
				}
				
			}
		}
		catch(XFlaimException e)
		{
			JOptionPane.showMessageDialog(
						this,
						"Database Exception occurred: " + e.getMessage(),
						"Database Exception",
						JOptionPane.ERROR_MESSAGE);
		}
		finally
		{
			if (bTransBegun)
			{
				try
				{
					m_jDb.transAbort();
				}
				catch (XFlaimException ex)
				{
					// ignore this one.
				}
			}
		}
	}

	/**
	 * @param m_selected
	 */
	private void deleteAttribute(
		NodePanel		np)
	{
		DOMNode			jNode = null;
		DOMNode			jAttr = null;
		boolean			bTransBegun = false;
		String			sMessage;
		
		try
		{
			Attribute attr = new Attribute("Empty", -1);

			jNode = m_jDb.getNode(np.getCollection(), np.getNodeId(), jNode);

			AttributeSelector AS = new AttributeSelector(this, jNode, attr);
			AS.dispose();

			if (attr.m_lNodeId == -1)
			{
				return; // Cancelled
			}
						
			jAttr = m_jDb.getNode(np.getCollection(), attr.m_lNodeId, jAttr);
			sMessage = new String("Delete attribute: <" + jAttr.getLocalName() + "> ?");
			
			if (JOptionPane.showConfirmDialog(
							this,
							sMessage) == JOptionPane.OK_OPTION)
			{
				// See if we need to begin an update transaction
				if (!m_status.getTransaction())
				{
					m_jDb.transBegin(
							TransactionType.UPDATE_TRANS, 0, 0);
					bTransBegun = true;
				}
				
				jAttr.deleteNode();
				
				if (bTransBegun)
				{
					m_jDb.transCommit();
				}
				
				// Update the display.
				if (np.isExpanded() && np.isClosing())
				{
					collapseNode(np);
					expandNode(m_selected, true);
				}
				else
				{
					np.buildLabel(jNode, np.isExpanded(), np.isClosing(), false);
				}
			}
		}
		catch (XFlaimException e)
		{
			JOptionPane.showMessageDialog(
						this,
						"Database Exception occurred: " + e.getMessage(),
						"Database Exception",
						JOptionPane.ERROR_MESSAGE);
			if (bTransBegun)
			{
				try
				{
					m_jDb.transAbort();
				}
				catch (XFlaimException ex)
				{
					// Don't do anything here.
				}
			}
		}
	}

	private void editValue(
		NodePanel		np)
	{
		DOMNode			jNode = null;
		boolean			bTransBegun = false;
		String			sValue;
		
		if (!m_status.getDatabaseOpen())
		{
			JOptionPane.showMessageDialog(
						this,
						"No open database",
						"Command Error",
						JOptionPane.INFORMATION_MESSAGE);
			return;
		}
		
		try
		{
			jNode = m_jDb.getNode(np.getCollection(), np.getNodeId(), jNode);
			sValue = jNode.getString();
			NodeValue nv = new NodeValue(sValue);
			EditValueDialog valueDialog = new EditValueDialog("Edit Node Value", this, nv);
			valueDialog.dispose();
			if (!sValue.equals(nv.getValue()))
			{
				// See if we need to begin an update transaction
				if (!m_status.getTransaction())
				{
					m_jDb.transBegin(
							TransactionType.UPDATE_TRANS, 0, 0);
					bTransBegun = true;
				}
				
				jNode.setString(nv.getValue());
				
				if (bTransBegun)
				{
					m_jDb.transCommit();
				}
				
				// Update the display.
				if (np.isExpanded() && np.isClosing())
				{
					collapseNode(np);
					expandNode(m_selected, true);
				}
				else
				{
					np.buildLabel(jNode, np.isExpanded(), np.isClosing(), false);
				}
			}
		}
		catch (XFlaimException e)
		{
			JOptionPane.showMessageDialog(
						this,
						"Database Exception occurred: " + e.getMessage(),
						"Database Exception",
						JOptionPane.ERROR_MESSAGE);
			if (bTransBegun)
			{
				try
				{
					m_jDb.transAbort();
				}
				catch (XFlaimException ex)
				{
					// Don't do anything here.
				}
			}
		}
	}

	private void editAnnotation(
		NodePanel		np)
	{
		DOMNode			jNode = null;
		DOMNode			jAnnot = null;
		boolean			bTransBegun = false;
		String			sValue;
		
		if (!m_status.getDatabaseOpen())
		{
			JOptionPane.showMessageDialog(
						this,
						"No open database",
						"Command Error",
						JOptionPane.INFORMATION_MESSAGE);
			return;
		}
		
		try
		{
			jNode = m_jDb.getNode(np.getCollection(), np.getNodeId(), jNode);
			jAnnot = jNode.getAnnotation(jAnnot);
			sValue = jAnnot.getString();
			NodeValue nv = new NodeValue(sValue);
			EditValueDialog valueDialog = new EditValueDialog("Edit Annotation Value", this, nv);
			valueDialog.dispose();
			if (!sValue.equals(nv.getValue()))
			{
				// See if we need to begin an update transaction
				if (!m_status.getTransaction())
				{
					m_jDb.transBegin(
							TransactionType.UPDATE_TRANS, 0, 0);
					bTransBegun = true;
				}
				
				jAnnot.setString(nv.getValue());
				
				if (bTransBegun)
				{
					m_jDb.transCommit();
				}
				
				// Update the display.
				if (np.isExpanded() && np.isClosing())
				{
					collapseNode(np);
					expandNode(m_selected, true);
				}
				else
				{
					np.buildLabel(jNode, np.isExpanded(), np.isClosing(), false);
				}
			}
		}
		catch (XFlaimException e)
		{
			JOptionPane.showMessageDialog(
						this,
						"Database Exception occurred: " + e.getMessage(),
						"Database Exception",
						JOptionPane.ERROR_MESSAGE);
			if (bTransBegun)
			{
				try
				{
					m_jDb.transAbort();
				}
				catch (XFlaimException ex)
				{
					// Don't do anything here.
				}
			}
		}
	}

	private void editAttribute(
		NodePanel		np)
	{
		DOMNode			jNode = null;
		DOMNode			jAttr = null;
		boolean			bTransBegun = false;
		String			sValue;
		
		if (!m_status.getDatabaseOpen())
		{
			JOptionPane.showMessageDialog(
						this,
						"No open database",
						"Command Error",
						JOptionPane.INFORMATION_MESSAGE);
			return;
		}
		
		try
		{
			Attribute attr = new Attribute("Empty", -1);

			jNode = m_jDb.getNode(np.getCollection(), np.getNodeId(), jNode);

			AttributeSelector AS = new AttributeSelector(this, jNode, attr);
			AS.dispose();

			if (attr.m_lNodeId == -1)
			{
				return; // Cancelled
			}
						
			jAttr = m_jDb.getNode(np.getCollection(), attr.m_lNodeId, jAttr);
			sValue = jAttr.getString();
			NodeValue nv = new NodeValue(sValue);
			EditValueDialog valueDialog = new EditValueDialog("Edit Attribute Value", this, nv);
			valueDialog.dispose();
			if (!sValue.equals(nv.getValue()))
			{
				// See if we need to begin an update transaction
				if (!m_status.getTransaction())
				{
					m_jDb.transBegin(
							TransactionType.UPDATE_TRANS, 0, 0);
					bTransBegun = true;
				}
				
				jAttr.setString(nv.getValue());
				
				if (bTransBegun)
				{
					m_jDb.transCommit();
				}
				
				// Update the display.
				if (np.isExpanded() && np.isClosing())
				{
					collapseNode(np);
					expandNode(m_selected, true);
				}
				else
				{
					np.buildLabel(jNode, np.isExpanded(), np.isClosing(), false);
				}
			}
		}
		catch (XFlaimException e)
		{
			JOptionPane.showMessageDialog(
						this,
						"Database Exception occurred: " + e.getMessage(),
						"Database Exception",
						JOptionPane.ERROR_MESSAGE);
			if (bTransBegun)
			{
				try
				{
					m_jDb.transAbort();
				}
				catch (XFlaimException ex)
				{
					// Don't do anything here.
				}
			}
		}
	}

	/**
	 * @param m_selected
	 */
	private void createNode(
		NodePanel			np)
	{
		if (!m_status.getDatabaseOpen())
		{
			JOptionPane.showMessageDialog(
						this,
						"No open database",
						"Command Error",
						JOptionPane.INFORMATION_MESSAGE);
			return;
		}
		try
		{
			// Need to get the collection.
			long	lNodeId = 0;
			int		iCollection = 0;

			if (np != null)
			{
				lNodeId = np.getNodeId();
				iCollection = np.getCollection();
			}

			if (iCollection > 0)
			{
				NodeDialog nd = new NodeDialog(
										this,
										m_dbSystem,
										m_status.getDatabasePath(),
										lNodeId,
										iCollection);
				GraphicsConfiguration gc = nd.getGraphicsConfiguration();
				Rectangle bounds = gc.getBounds();
				int iHeight = nd.getHeight();
				int iWidth = nd.getWidth();
				nd.setLocation(Math.max(0, (bounds.width - iWidth)/2),
								Math.max(0, (bounds.height - iHeight)/2));
				Thread t = new Thread((Runnable)nd);
				t.start();
			}
		}
		catch (XFlaimException e)
		{
			JOptionPane.showMessageDialog(
						this,
						"Failed to create dialog: " + e.getMessage(),
						"Exception occurred",
						JOptionPane.ERROR_MESSAGE);
		}		
	}

	/**
	 * 
	 */
	private void createDocument(
		NodePanel		np)
	{
		if (!m_status.getDatabaseOpen())
		{
			JOptionPane.showMessageDialog(
						this,
						"No open database",
						"Command Error",
						JOptionPane.INFORMATION_MESSAGE);
			return;
		}
		try
		{
			// Need to get the collection.
			int iCollection = selectCollection();
			if (iCollection > 0)
			{
				long	lNodeId = 0;
				if (np != null)
				{
					lNodeId = np.getNodeId();
				}
				NodeDialog nd = new NodeDialog(
										this,
										m_dbSystem,
										m_status.getDatabasePath(),
										lNodeId,
										iCollection);
				GraphicsConfiguration gc = nd.getGraphicsConfiguration();
				Rectangle bounds = gc.getBounds();
				int iHeight = nd.getHeight();
				int iWidth = nd.getWidth();
				nd.setLocation(Math.max(0, (bounds.width - iWidth)/2),
								Math.max(0, (bounds.height - iHeight)/2));
				Thread t = new Thread((Runnable)nd);
				t.start();
			}
		}
		catch (XFlaimException e)
		{
			JOptionPane.showMessageDialog(
						this,
						"Failed to create dialog: " + e.getMessage(),
						"Exception occurred",
						JOptionPane.ERROR_MESSAGE);
		}		
	}

	/**
	 * 
	 */
	private void beginTransaction()
	{
		if (m_status.getTransaction())
		{
			JOptionPane.showMessageDialog(
						this,
						"Update transaction already in progress",
						"Transaction Error",
						JOptionPane.INFORMATION_MESSAGE);
			return;
		}

		try
		{
			if (m_status.getDatabaseOpen())
			{
				m_jDb.transBegin(TransactionType.UPDATE_TRANS, 0, 0);
				m_status.setTransaction(true);
			}
			else
			{
				JOptionPane.showMessageDialog(
							this,
							"Cannot begin transaction - no open database.",
							"Incomplete Operation",
							JOptionPane.INFORMATION_MESSAGE);
			}
		}
		catch (XFlaimException e)
		{
			JOptionPane.showMessageDialog(
						this,
						"Database Exception occurred: " + e.getMessage(),
						"Database Exception",
						JOptionPane.ERROR_MESSAGE);
			
		}
	}

	/**
	 * 
	 */
	private void commitTransaction()
	{
		if (!m_status.getTransaction())
		{
			JOptionPane.showMessageDialog(
						this,
						"No update transaction in progress",
						"Transaction Error",
						JOptionPane.INFORMATION_MESSAGE);
			return;
		}

		try
		{
			if (m_status.getDatabaseOpen())
			{
				m_jDb.transCommit();
				m_status.setTransaction(false);
			}
		}
		catch (XFlaimException e)
		{
			JOptionPane.showMessageDialog(
						this,
						"Database Exception occurred: " + e.getMessage(),
						"Database Exception",
						JOptionPane.ERROR_MESSAGE);
			
		}
	}

	/**
	 * 
	 */
	private void abortTransaction()
	{
		if (!m_status.getTransaction())
		{
			JOptionPane.showMessageDialog(
						this,
						"No update transaction in progress",
						"Transaction Error",
						JOptionPane.INFORMATION_MESSAGE);
			return;
		}

		try
		{
			if (m_status.getDatabaseOpen())
			{
				m_jDb.transAbort();
				m_status.setTransaction(false);
			}
		}
		catch (XFlaimException e)
		{
			JOptionPane.showMessageDialog(
						this,
						"Database Exception occurred: " + e.getMessage(),
						"Database Exception",
						JOptionPane.ERROR_MESSAGE);
			
		}
	}

	/*=========================================================================
	 * Desc: constructs the main menu for the application.
	 *=======================================================================*/
	private JMenuBar makeMainMenu()
	{
		JMenuBar		menuBar = null;
		JMenu			fileMenu = null;
		JMenu			docMenu = null;
		JMenu			transMenu = null;

		menuBar = new JMenuBar();
		// Create the main menu.
		menuBar = new JMenuBar();

		// Create the File menu
		fileMenu = new JMenu("File");
		fileMenu.setMnemonic(KeyEvent.VK_F);

		// Add the file Menu to the menu bar
		menuBar.add(fileMenu);

		/*------------------------ File menu ---------------------------------*/

		// Create a few items to add to the file menu
		JMenuItem menuItem = new JMenuItem("Open");
		menuItem.setAccelerator(
			KeyStroke.getKeyStroke(KeyEvent.VK_O, ActionEvent.ALT_MASK));
		menuItem.setMnemonic(KeyEvent.VK_O);
		menuItem.addActionListener(this);
		fileMenu.add(menuItem);

		menuItem = new JMenuItem("Close");
		menuItem.setAccelerator(
			KeyStroke.getKeyStroke(KeyEvent.VK_C, ActionEvent.ALT_MASK));
		menuItem.setMnemonic(KeyEvent.VK_C);
		menuItem.addActionListener(this);
		fileMenu.add(menuItem);

		menuItem = new JMenuItem("New");
		menuItem.setAccelerator(
			KeyStroke.getKeyStroke(KeyEvent.VK_N, ActionEvent.ALT_MASK));
		menuItem.setMnemonic(KeyEvent.VK_N);
		menuItem.addActionListener(this);
		fileMenu.add(menuItem);

		fileMenu.addSeparator();

		menuItem = new JMenuItem("Remove");
		menuItem.setAccelerator(
			KeyStroke.getKeyStroke(KeyEvent.VK_R, ActionEvent.ALT_MASK));
		menuItem.setMnemonic(KeyEvent.VK_R);
		menuItem.addActionListener(this);
		fileMenu.add(menuItem);

		fileMenu.addSeparator();

		menuItem = new JMenuItem("Exit");
		menuItem.setAccelerator(
			KeyStroke.getKeyStroke(KeyEvent.VK_X, ActionEvent.ALT_MASK));
		menuItem.setMnemonic(KeyEvent.VK_X);
		menuItem.addActionListener(this);
		fileMenu.add(menuItem);

		/*------------------------ Document menu ---------------------------------*/

		// Create the Document menu
		docMenu = new JMenu("Document");
		docMenu.setMnemonic(KeyEvent.VK_D);

		// Add the Document Menu to the menu bar
		menuBar.add(docMenu);

		// Create a few items to add to the document menu
		menuItem = new JMenuItem("First Document");
		menuItem.setAccelerator(
			KeyStroke.getKeyStroke(KeyEvent.VK_D, ActionEvent.CTRL_MASK));
		menuItem.setMnemonic(KeyEvent.VK_D);
		menuItem.addActionListener(this);
		docMenu.add(menuItem);

		menuItem = new JMenuItem("Last Document");
		menuItem.addActionListener(this);
		menuItem.setEnabled(false);
		docMenu.add(menuItem);

		menuItem = new JMenuItem("Next Document");
		menuItem.addActionListener(this);
		menuItem.setEnabled(false);
		docMenu.add(menuItem);

		menuItem = new JMenuItem("Open Document");
		menuItem.setAccelerator(
			KeyStroke.getKeyStroke(KeyEvent.VK_O, ActionEvent.CTRL_MASK));
		menuItem.setMnemonic(KeyEvent.VK_O);
		menuItem.addActionListener(this);
		docMenu.add(menuItem);

		docMenu.addSeparator();

		menuItem = new JMenuItem("Create Document");
		menuItem.addActionListener(this);
		docMenu.add(menuItem);

		docMenu.addSeparator();

		menuItem = new JMenuItem("Delete Document");
		menuItem.addActionListener(this);
		menuItem.setEnabled(false);
		docMenu.add(menuItem);

		docMenu.addSeparator();

		menuItem = new JMenuItem("Import Document(s)");
		menuItem.setAccelerator(
			KeyStroke.getKeyStroke(KeyEvent.VK_I, ActionEvent.CTRL_MASK));
		menuItem.setMnemonic(KeyEvent.VK_I);
		menuItem.addActionListener(this);
		docMenu.add(menuItem);

		docMenu.addSeparator();

		menuItem = new JMenuItem("Close Document");
		menuItem.addActionListener(this);
		docMenu.add(menuItem);
		
		/*---------------- Transaction Menu ----------------------*/
		// Create the Transaction menu
		transMenu = new JMenu("Transaction");
		transMenu.setMnemonic(KeyEvent.VK_T);

		// Add the transaction Menu to the menu bar
		menuBar.add(transMenu);

		// Create a few items to add to the transaction menu
		menuItem = new JMenuItem("Begin Transaction");
		menuItem.setAccelerator(
			KeyStroke.getKeyStroke(KeyEvent.VK_B, ActionEvent.CTRL_MASK));
		menuItem.setMnemonic(KeyEvent.VK_B);
		menuItem.addActionListener(this);
		transMenu.add(menuItem);

		menuItem = new JMenuItem("Commit Transaction");
		menuItem.setAccelerator(
			KeyStroke.getKeyStroke(KeyEvent.VK_C, ActionEvent.CTRL_MASK));
		menuItem.setMnemonic(KeyEvent.VK_C);
		menuItem.addActionListener(this);
		transMenu.add(menuItem);

		menuItem = new JMenuItem("Abort Transaction");
		menuItem.setAccelerator(
			KeyStroke.getKeyStroke(KeyEvent.VK_A, ActionEvent.CTRL_MASK));
		menuItem.setMnemonic(KeyEvent.VK_A);
		menuItem.addActionListener(this);
		transMenu.add(menuItem);


		return menuBar;

	}

	/*=========================================================================
	 * Desc: 	Method to choose and open a database.
	 *========================================================================*/
	private void openDatabase()
	{
		String				sDatabaseName = null;
		String				sDirectory = null;
		JFileChooser 		fileChooser = new JFileChooser();
		XFileFilter 		dbFilter = new XFileFilter(".db");
		
		fileChooser.setFileFilter(dbFilter);
		fileChooser.setCurrentDirectory(new File(m_sDirectory));
		fileChooser.setDialogTitle("Open Database");
		fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);

		if (fileChooser.showDialog(null, "Open Database") == JFileChooser.APPROVE_OPTION)
		{
			try
			{
				sDatabaseName = new String(fileChooser.getSelectedFile().getPath());
				sDirectory = new String(fileChooser.getCurrentDirectory().getPath());

				if (m_jDb != null)
				{
					closeDatabase();
				}

				// If this fails, we will throw an exception.
				m_jDb = m_dbSystem.dbOpen(sDatabaseName, null, null, null, true);
				m_status.setDatabasePath(sDatabaseName);
				m_status.setDatabaseOpen(true);
				m_sDatabaseName = sDatabaseName;
				m_sDirectory = sDirectory;
			}
			catch (XFlaimException e)
			{
				JOptionPane.showMessageDialog(
							this,
							"Database Exception occurred: " + e.getMessage(),
							"Database Exception",
							JOptionPane.ERROR_MESSAGE);
			}
		}
	}

	/*=========================================================================
	 * Desc: 	Method to close the current database.
	 *========================================================================*/
	private void closeDatabase()
	{
		if (m_jDb == null)
		{
			return;
		}

		// Make sure we clear out the document list.
		if (m_mainDisplay.getComponentCount() > 0)
		{
			closeAllDocuments();
		}

		m_jDb.close();
		m_jDb = null;
		m_status.setDatabaseOpen(false);
		m_status.setDatabasePath(null);
		m_status.setDocId(-1);
		m_status.setNodeId(-1);
		m_sDatabaseName = null;
		if (m_selected != null)
		{
			m_selected.deselectNode();
			m_selected = null;
		}
//		m_iCurrentNode = NO_NODES;
		m_iLastNode = NO_NODES;
	
		m_status.m_iCurrentNode = NO_NODES;
		m_status.m_iLastNode = m_iLastNode;
		m_status.updateLabel();
	
		updateDisplay();

	}

	/*=========================================================================
	 * Desc: 	Method to close the current database.
	 *========================================================================*/
	private void createDatabase()
	{
		String				sDatabaseName = null;
		String				sDirectory = null;
		JFileChooser 	fileChooser = new JFileChooser();
		XFileFilter 		dbFilter = new XFileFilter(".db");

		fileChooser.setFileFilter(dbFilter);
		fileChooser.setCurrentDirectory(new File(m_sDirectory));
		fileChooser.setDialogTitle("Create Database");
		fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);

		if (fileChooser.showDialog(null, "Create Database") == JFileChooser.APPROVE_OPTION)
		{
			sDatabaseName = new String( fileChooser.getSelectedFile().getPath());
			sDirectory = new String( fileChooser.getCurrentDirectory().getPath());

			try
			{
				if (m_jDb != null)
				{
					closeDatabase();
				}

				// If this fails, we will throw an exception.
				m_jDb = m_dbSystem.dbCreate( sDatabaseName, "", "", "", "", null);

				m_status.setDatabasePath(sDatabaseName);
				m_status.setDatabaseOpen(true);
				m_sDatabaseName = sDatabaseName;
				m_sDirectory = sDirectory;
			}
			catch (XFlaimException e)
			{
				JOptionPane.showMessageDialog(
							this,
							"Database Exception occurred: " + e.getMessage(),
							"Database Exception",
							JOptionPane.ERROR_MESSAGE);
			}
		}
	}

	/*=========================================================================
	 * Desc: 	Method to delete document node.
	 *========================================================================*/
	 private void deleteNode(
	 	NodePanel		np)
	 {
	 	DOMNode			jNode = null;
	 	boolean			bTransBegun = false;
	 	Document			doc = null;
	 	boolean			bIsRoot = false;
		String			sMessage;
		long				lDocId;
	 	
	 	try
	 	{
	 		jNode = m_jDb.getNode(np.getCollection(), np.getNodeId(), null);
	 		lDocId = np.getDocId();

			// Check to see if this is a root node.  Ultimately we will call delete document
			// to handle it.	 	
			if (np.getNodeId() == lDocId)
			{
				bIsRoot = true;
				doc = new Document("<" + jNode.getLocalName() + ">",
									np.getDocId(),
									np.getCollection());
			}

	 		if (jNode.getNodeType() == FlmDomNodeType.DATA_NODE)
	 		{
	 			sMessage = new String("Delete data node");
	 		}
			else if (bIsRoot)
			{
				sMessage = new String("Delete document: <" + jNode.getLocalName() + ">");
			}
			else
			{
				sMessage = new String("Delete node: <" + jNode.getLocalName() + ">");
			}

	 		if (JOptionPane.showConfirmDialog(
	 						this,
							sMessage) == JOptionPane.OK_OPTION)
	 		{
				boolean	bHasNextSibling = false;


	 			// Check for a next sibling.
	 			if (jNode.hasNextSibling())
	 			{
	 				try
	 				{
	 					DOMNode jNext = jNode.getNextSibling(null);
	 					bHasNextSibling = true;
						// This is here to silence the warning that jNext is never read.
	 					if (jNext != null)
	 					{
	 						jNext = null;
	 					}
	 				}
	 				catch (XFlaimException e)
	 				{
	 					// Ignore this.
	 				}
	 			}

				// Collapse the current row.
				if (np.isExpanded())
				{
					collapseNode(np);
				}
	 		
				// Begin a transaction
				if (!m_status.getTransaction())
				{
					m_jDb.transBegin(TransactionType.UPDATE_TRANS, 0, 0);
					bTransBegun = true;
				}
	 		
				// Delete the node.
				jNode.deleteNode();
	 		
				// Commit
				if (bTransBegun)
				{
					m_jDb.transCommit();
					bTransBegun = false;
				}

				// Remove the current node from the display
				removeRow(np.getRow());
				
				// If the new row is a parent, then we need to collapse it so
				// the display looks ok.
				if (!bHasNextSibling)
				{
					if (np.isExpanded() && lDocId == np.getDocId())
					{
						collapseNode(np);
					}
				}
				
				m_status.setDocId(np.getDocId());
				m_status.setNodeId(np.getNodeId());
				
				if (bIsRoot)
				{
					m_vDocList.remove(doc);
				}
	 		}
	 	}
	 	catch(XFlaimException e)
	 	{
	 		JOptionPane.showMessageDialog(
	 					this,
						"Database Exception occurred: " + e.getMessage(),
						"Database Exception",
						JOptionPane.ERROR_MESSAGE);
	 	}
	 	finally
	 	{
	 		if (bTransBegun)
	 		{
	 			try
	 			{
	 				m_jDb.transAbort();
	 			}
	 			catch (XFlaimException ex)
	 			{
	 				// ignore this one.
	 			}
	 		}
	 	}
	 }

	/*=========================================================================
	 * Desc: 	Method to remove (delete) a database.
	 *========================================================================*/
	private void removeDatabase()
	{
		String				sDatabaseName = m_sDatabaseName;

		// If there is already an open database close it.
		if (sDatabaseName != null)
		{
			closeDatabase();
		}
		else
		{
			// We need to choose a database to remove.
			JFileChooser 	fileChooser = new JFileChooser();
			XFileFilter 		dbFilter = new XFileFilter(".db");

			fileChooser.setFileFilter(dbFilter);
			fileChooser.setCurrentDirectory(new File(m_sDirectory));
			fileChooser.setDialogTitle("Remove Database");
			fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);

			if (fileChooser.showDialog(null, "Remove Database") == JFileChooser.APPROVE_OPTION)
			{
				sDatabaseName = new String( fileChooser.getSelectedFile().getPath());

				if (m_jDb != null)
				{
					closeDatabase();
				}
			}
		}

		if (sDatabaseName != null)
		{
			// If this fails, we will throw an exception.
			try
			{
				m_dbSystem.dbRemove( sDatabaseName, "", "", true);
			}
			catch (XFlaimException e)
			{
				JOptionPane.showConfirmDialog(
					this,
					"Database Exception occurred: " + e.getMessage(),
					"Database Exception",
					JOptionPane.ERROR_MESSAGE);
			}
		}
	}

	/*=========================================================================
	 * Desc: 	Method to choose and import a document or directory into the current database.
	 *========================================================================*/
	private void importDocument()
	{
		String					sDirectory = null;
		JFileChooser			fileChooser = new JFileChooser();
		XFileFilter				xmlFilter = new XFileFilter(".xml");
		int						iCollection;

		// Make sure we have a database open.
		if (!m_status.getDatabaseOpen())
		{
			JOptionPane.showMessageDialog(this, "There is no open database.  Import operation failed.");
			return;
		}

		fileChooser.setFileFilter(xmlFilter);
		fileChooser.setCurrentDirectory(new File(m_sDirectory));
		fileChooser.setDialogTitle("Import XML File or Directory");
		fileChooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);

		if (fileChooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION)
		{
			sDirectory = new String( fileChooser.getCurrentDirectory().getPath());
			iCollection = selectCollection();
			if (iCollection == 0)
			{
				// Cancelled!
				return;
			}

			try
			{
				FileImporter fi = new FileImporter(
											this,
											m_dbSystem,
											m_sDatabaseName,
											iCollection,
											new String( fileChooser.getSelectedFile().getPath()),
											sDirectory,
											fileChooser.getSelectedFile().list());
				fi.start();

				// Save the last selected directory.
				m_sDirectory = sDirectory;
			}
			catch (XFlaimException e)
			{
				JOptionPane.showConfirmDialog(
							this,
							"Database Exception occurred: " + e.getMessage(),
							"Database Exception",
							JOptionPane.ERROR_MESSAGE);
			}
		}
	}
	

	/*=========================================================================
	 * Desc: 	Method to retrieve the first document from the database.
	 *========================================================================*/
	private void getFirstDocument()
	{
		DOMNode			jDoc = null;
		int				iCollection;
		
		if (m_status.getDatabaseOpen())
		{
			// Need to get the collection.
			iCollection = selectCollection();
			if (iCollection > 0)
			{
				try
				{
					jDoc = m_jDb.getFirstDocument(iCollection, null);
					m_vDocList.add(new Document("<" + jDoc.getLocalName() + ">",
												jDoc.getNodeId(),
												iCollection));
					if (m_iLastNode < LAST_NODE)
					{
						NodePanel np = (NodePanel)m_vNodeList.get(++m_iLastNode);
						np.buildLabel( jDoc, false, false, m_bDocList);
						expandNode(np, true);
					
						m_status.m_iLastNode = m_iLastNode;
						m_status.updateLabel();
					
						if (m_selected == null)
						{
							selectNode(np);
						}
					}
				}
				catch (XFlaimException e)
				{
					JOptionPane.showMessageDialog(
								this,
								"Database Exception occurred: " + e.getMessage(),
								"Database Exception",
								JOptionPane.ERROR_MESSAGE);
				}
			}
		}
		else
		{
			// Put up a dialog to indicate that there is no database open yet.
			JOptionPane.showMessageDialog(
						this,
						"Cannot retrieve a document - no open database.",
						"Incomplete Operation",
						JOptionPane.INFORMATION_MESSAGE);
		}
	}

	/*=========================================================================
	 * Desc: 	Method to present a list of documents to open.
	 *========================================================================*/
	private void openDocument()
	{
		if (m_status.getDatabaseOpen())
		{
			// Need to get the collection.
			int iCollection = selectCollection();
			if (iCollection > 0)
			{
				selectDocument(iCollection);
			}
		}
		else
		{
			// Put up a dialog to indicate that there is no database open yet.
			JOptionPane.showMessageDialog(
						this,
						"Cannot retrieve a document - no open database.",
						"Incomplete Operation",
						JOptionPane.INFORMATION_MESSAGE);
		}
	}

	/*=========================================================================
	 * Desc: 	Method to a document of choice from the database.
	 *========================================================================*/
	private void openDocument(
		int		iCollection,
		long	lDocId)
	{
		DOMNode		jDoc = null;

		try
		{
			if (lDocId > 0)
			{
				jDoc = m_jDb.getNode(iCollection, lDocId, null);
				m_vDocList.add(new Document("<" + jDoc.getLocalName() + ">",
											jDoc.getNodeId(),
											iCollection));
				if (m_iLastNode < LAST_NODE)
				{
					NodePanel np = (NodePanel)m_vNodeList.get(++m_iLastNode);
					np.buildLabel(jDoc, false, false, m_bDocList);
					expandNode(np, true);
					
					m_status.m_iLastNode = m_iLastNode;
					m_status.updateLabel();
					
					if (m_selected == null)
					{
						selectNode(np);
					}
				}
			}
		}
		catch (XFlaimException e)
		{
			JOptionPane.showMessageDialog(
						this,
						"Database Exception occurred:" + e.getMessage(),
						"Database Exception",
						JOptionPane.ERROR_MESSAGE);
		}
	}


	/*=========================================================================
	 * Desc: 	Method to close the document of choice.
	 *========================================================================*/
	private void closeDocument()
	{
		int			iItem;
		boolean		bDocFound = false;

		if (m_vDocList.size() > 1)
		{
			// Need to select the document to close.
			iItem = selectDocumentFromList();
			if (iItem >= 0)
			{
				Document doc = (Document)m_vDocList.get(iItem);

				// Find the root document node and remove it.
				for (int i = 0; i <= m_iLastNode; i++)
				{
					NodePanel np = (NodePanel)m_vNodeList.get(i);
					if (np.getDocId() == doc.m_lDocId)
					{
						closeDocument(np);
						bDocFound = true;
						break;
					}
				}

				// The document may not have been visible on the screen, so we must remove it
				// from the document list.				
				if (!bDocFound)
				{
					m_vDocList.removeElementAt(iItem);
				}
			}
		}
		else
		{
			if (m_vDocList.size() == 1)
			{
				closeAllDocuments();
			}
		}
		updateDisplay();
	}
	
	/*=========================================================================
	 * Desc: 	Method to close the document owned by the node panel passed in.
	 *========================================================================*/
	private void closeDocument(
		NodePanel		np)
	{
		int		iRow = np.getRow();
		int		iItem;

		try
		{
			DOMNode		jNode = m_jDb.getNode( np.getCollection(), np.getNodeId(), null);
			Document 	doc = new Document("<" + jNode.getLocalName() + ">",
										np.getDocId(),
										np.getCollection());

			iItem = m_vDocList.indexOf(doc);
			if (iItem >= 0)
			{
				// Now let's clean up the display.
				// First, if the document is expanded, we need to collapse it.
				if (np.isExpanded())
				{
					collapseNode(np);
				}
				
				m_vDocList.removeElementAt(iItem);
				
				removeRow(iRow);
				
				if (m_vDocList.size() == 0)
				{
					m_status.setDocId(0);
					m_status.setNodeId(0);
				}
				
				if (m_selected != null)
				{
					selectNode(m_selected);
				}
			}
			else
			{
				JOptionPane.showMessageDialog(
							this,
							"Failed to close current document: Document not found",
							"Document Not Closed",
							JOptionPane.WARNING_MESSAGE);
			}
		

		}
		catch (XFlaimException e)
		{
			JOptionPane.showMessageDialog(
						this,
						"Database Exception occurred: " + e.getMessage(),
						"Database Exception",
						JOptionPane.ERROR_MESSAGE);
		}
	}

	/*=========================================================================
	 * Desc: 	Method to close all documents
	 *========================================================================*/
	private void closeAllDocuments()
	{
		for (int i= FIRST_NODE; i <= LAST_NODE; i++)
		{
			NodePanel np = (NodePanel)m_vNodeList.get(i);
			np.reset();
		}
		m_vDocList.removeAllElements();
//		m_iCurrentNode = NO_NODES;
		m_iLastNode = NO_NODES;
		m_selected = null;
		m_status.m_iCurrentNode = NO_NODES;
		m_status.m_iLastNode = m_iLastNode;
		m_status.setDocId(0);
		m_status.setNodeId(0);
	}

	/*=========================================================================
	 * Desc: 	Method to select the collection
	 *========================================================================*/
	private int selectCollection()
	{
		Collection coll = new Collection("invalid", 0);
		CollectionSelector CS = new CollectionSelector(this, m_dbSystem, m_jDb, coll);
		CS.dispose();
		
		return coll.m_iNumber;
	}

	/*=========================================================================
	 * Desc: 	Method to select document
	 *========================================================================*/
	private void selectDocument(
		int		iCollection
		)
	{
		Document doc = null;
		Thread t = null;
		
		doc = new Document("invalid", 0, 0);
		XEdit DS = new XEdit(this, "Document Selector", m_jDb, iCollection, doc);
		GraphicsConfiguration gc = getGraphicsConfiguration();
		Rectangle bounds = gc.getBounds();
		int iHeight = getHeight();
		int iWidth = getWidth();
		DS.setLocation(Math.max(0, (bounds.width - iWidth)/2),
						Math.max(0, (bounds.height - iHeight)/2));
		t = new Thread((Runnable)DS);
		t.start();
	}

	/*=========================================================================
	 * Desc: 	Method to select document from the list of open documents.
	 *========================================================================*/
	private int selectDocumentFromList()
	{
		OpenDocumentSelector ods = new OpenDocumentSelector( this, m_vDocList);
		return ods.showDialog();
	}

	// Default protection means the package can see this method.
	void updateDisplay()
	{
		setContentPane(m_mainPanel);
	}

	// Service method to compare a JPanel object to the main panel
	boolean isMainPanel(
		JPanel		panel)
	{
		if (panel == m_mainDisplay)
		{
			return true;
		}
		else
		{
			return false;
		}
	}


	/* (non-Javadoc)
	 * @see java.awt.event.MouseListener#mouseClicked(java.awt.event.MouseEvent)
	 */
	public void mouseClicked(MouseEvent e)
	{
		NodePanel	np = null;

		np = (NodePanel)e.getSource(); 
		
		selectNode(np);
		
		if (e.getClickCount() == 1 && !m_bDocList)
		{
			if (e.getButton() == MouseEvent.BUTTON1)
			{
				if (np.isExpanded())
				{
					collapseNode(np);
				}
				else
				{
					expandNode(np, false);
				}
			}
			else if (e.getButton() == MouseEvent.BUTTON3)
			{
				JMenuItem		menuItem = null;
				DOMNode			jNode = null;
				
				if (m_status.getDatabaseOpen())
				{
					try
					{
						jNode = m_jDb.getNode(
										np.getCollection(),
										np.getNodeId(),
										jNode);
						
						// popup the document menu.
						m_popup = new JPopupMenu("Document Node Options");
						if (np.getDocId() != NO_NODES)
						{
							if (np.getDocId() == np.getNodeId())
							{
								menuItem = m_popup.add("Close Document");
								menuItem.addActionListener(this);
								m_popup.addSeparator();
							}

							if (np.hasChildren())
							{
								if (np.isExpanded())
								{
									menuItem = m_popup.add("Collapse Node");
								}
								else
								{
									menuItem = m_popup.add("Expand Node");
								}
								menuItem.addActionListener(this);
								m_popup.addSeparator();
							}

							// Make a sub-menu for the delete
							JMenu delMenu = new JMenu("Delete");
							
							menuItem = delMenu.add("Delete Node");
							menuItem.addActionListener(this);
							menuItem = delMenu.add("Delete Attribute");
							menuItem.addActionListener(this);
							if (!jNode.hasAttributes())
							{
								menuItem.setEnabled(false);
							}
							menuItem = delMenu.add("Delete Annotation");
							menuItem.addActionListener(this);
							if (!jNode.hasAnnotation())
							{
								menuItem.setEnabled(false);
							}
							m_popup.add(delMenu);
					
							menuItem = m_popup.add("Add Node");
							menuItem.addActionListener(this);

							// Need to make the edit sub-menu.
							JMenu editMenu = new JMenu("Edit Node");
							menuItem = editMenu.add("Edit Value");
							// Test for a value
							try
							{
								String sValue = jNode.getString();
								if (sValue.length() == 0)
								{
									menuItem.setEnabled(false);
								}
							}
							catch (XFlaimException ee)
							{
								menuItem.setEnabled(false);
							}
							menuItem.addActionListener(this);
							menuItem = editMenu.add("Edit Attribute");
							if (!jNode.hasAttributes())
							{
								menuItem.setEnabled(false);
							}
							menuItem.addActionListener(this);
							menuItem = editMenu.add("Edit Annotation");
							if (!jNode.hasAnnotation())
							{
								menuItem.setEnabled(false);
							}
							menuItem.addActionListener(this);
							m_popup.add(editMenu);
						}
						else
						{
							menuItem = m_popup.add("Add Document");
							menuItem.addActionListener(this);
							menuItem.setEnabled(false);
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

					// Need to make sure the menu appears close to where our window is located.
					Point p = e.getPoint();
					Point p1 = np.getLocation();
					m_popup.show(this, p1.x + p.x, p1.y + p.y);
				}
			}
		}
		
		if (e.getClickCount() == 2)
		{
			if (e.getButton() == MouseEvent.BUTTON1)
			{
				// Are we selecting a document?
				if (m_bDocList)
				{
					m_doc.m_iCollection = np.getCollection();
					m_doc.m_lDocId = np.getDocId();
					try
					{
						DOMNode jDoc = m_jDb.getNode(m_doc.m_iCollection, m_doc.m_lDocId, null);
						m_doc.m_sName = "<" + jDoc.getLocalName() + ">";
						if (m_parent != null)
						{
							// Open the document in the parent
							m_parent.openDocument(m_doc.m_iCollection, m_doc.m_lDocId);
							this.dispose();
						}
					}
					catch (XFlaimException ex)
					{
						JOptionPane.showMessageDialog(
									this,
									"Database Exception ocurred: " + ex.getMessage(),
									"Database Exception",
									JOptionPane.ERROR_MESSAGE);
					}
				}
			}
		}
	}

	/* (non-Javadoc)
	 * @see java.awt.event.MouseListener#mouseEntered(java.awt.event.MouseEvent)
	 */
	public void mouseEntered(MouseEvent e)
	{
		// Do nothing.
		
	}

	/* (non-Javadoc)
	 * @see java.awt.event.MouseListener#mouseExited(java.awt.event.MouseEvent)
	 */
	public void mouseExited(MouseEvent e)
	{
		// Do Nothing
		
	}

	/* (non-Javadoc)
	 * @see java.awt.event.MouseListener#mousePressed(java.awt.event.MouseEvent)
	 */
	public void mousePressed(MouseEvent e)
	{
		// Do Nothing
		
	}

	/* (non-Javadoc)
	 * @see java.awt.event.MouseListener#mouseReleased(java.awt.event.MouseEvent)
	 */
	public void mouseReleased(MouseEvent e)
	{
		// Do Nothing
		
	}

	/* (non-Javadoc)
	 * @see java.awt.event.KeyListener#keyPressed(java.awt.event.KeyEvent)
	 */
	public void keyPressed(KeyEvent e)
	{
		// Do Nothing
		
	}

	/* (non-Javadoc)
	 * @see java.awt.event.KeyListener#keyReleased(java.awt.event.KeyEvent)
	 */
	public void keyReleased(KeyEvent e)
	{
		int		key = e.getKeyCode();

		if (m_vDocList.isEmpty() && !m_bDocList)
		{
			return;
		}
		
		switch (key)
		{
			case KeyEvent.VK_KP_UP:
			case KeyEvent.VK_UP:
			{
				moveUp();
				break;
			}
			case KeyEvent.VK_KP_DOWN:
			case KeyEvent.VK_DOWN:
			{
				moveDown();
				break;
			}
			case KeyEvent.VK_PAGE_UP:
			{
				pageUp();
				break;
			}
			case KeyEvent.VK_PAGE_DOWN:
			{
				pageDown();
				break;
			}
			case KeyEvent.VK_KP_RIGHT:
			case KeyEvent.VK_RIGHT:
			{
				// Expand.
				if (!m_selected.isExpanded())
				{
					expandNode(m_selected, false);
				}
				break;
			}
			case KeyEvent.VK_KP_LEFT:
			case KeyEvent.VK_LEFT:
			{
				// Collapse
				if (m_selected.isExpanded())
				{
					collapseNode(m_selected);
				}
				break;
			}
			case KeyEvent.VK_ENTER:
			{
				// Are we selecting a document?
				if (m_bDocList)
				{
					m_doc.m_iCollection = m_selected.getCollection();
					m_doc.m_lDocId = m_selected.getDocId();
					try
					{
						DOMNode jDoc = m_jDb.getNode(m_doc.m_iCollection, m_doc.m_lDocId, null);
						m_doc.m_sName = "<" + jDoc.getLocalName() + ">";
						if (m_parent != null)
						{
							// Open the document in the parent
							m_parent.openDocument(m_doc.m_iCollection, m_doc.m_lDocId);
							this.dispose();
						}
					}
					catch (XFlaimException ex)
					{
						JOptionPane.showMessageDialog(
									this,
									"Database Exception ocurred: " + ex.getMessage(),
									"Database Exception",
									JOptionPane.ERROR_MESSAGE);
					}
				}
			}
		}
		
	}

	/**
	 * 
	 */
	private void moveUp()
	{
		if (m_selected == null)
		{
			// First row not set yet.
			return;
		}

		if (m_selected.getRow() > 0)
		{
			selectNode((NodePanel)m_vNodeList.get(m_selected.getRow() - 1));
		}
		else
		{
			// We are going to have to move eveyone down one row first.
			// Then get the previous row.
			if (scrollUp())
			{
				selectNode((NodePanel)m_vNodeList.get(m_selected.getRow() - 1));
			}
		}
	}

	/**
	 * 
	 */
	private boolean scrollUp()
	{
		NodePanel		npTo;
		NodePanel		npFrom;
		
		if (m_selected == null)
		{
			return false;
		}
		
		if (hasPreviousNode())
		{
			for (int i = LAST_NODE; i > 0; --i)
			{
				npTo = (NodePanel)m_vNodeList.get(i);
				npFrom = (NodePanel)m_vNodeList.get(i - 1);
				npTo.copyNode(npFrom);
			}
		
			// Move the selected Node down one row too.
			selectNode((NodePanel)m_vNodeList.get(m_selected.getRow() + 1));
		
			// Now we need to get the previous row.  Due to the way this is
			// architected, we should be looking at the currently selected row
			// to determine the previous row.  The currently selected row is supposed to
			// be the second row now.  If there isn't a previous row, we will get a false
			// return value.
			if (!m_bDocList)
			{
				return(getPrevNode());
			}
			else
			{
				return(getPrevDocument());
			}
		}
		else
		{
			return false;
		}
	}

	/**
	 * @return
	 */
	private boolean hasPreviousNode()
	{
		NodePanel	np = m_selected;  // Should be the first row, i.e. row 0.
		DOMNode		jNode = null;
		Document		doc = null;
		
		try
		{
			jNode = m_jDb.getNode(np.getCollection(), np.getDocId(), null);
			
			if (!m_bDocList)
			{
				// With a standard XEdit, we check to see if we have another open
				// document.
				doc = new Document("<" + jNode.getLocalName() + ">",
									np.getDocId(),
									np.getCollection());

				// If we're looking at the root node, then we need to find out if there
				// is another document to view previous to this one.
				if (np.getDocId() == np.getNodeId() && !np.isClosing())
				{
					if (m_vDocList.indexOf(doc) > 0)
					{
						return true;
					}
					else
					{
						return false;
					}
				}
				else
				{
					return true;
				}
			}
			else
			{
				// With a DocList, we check to see if there is a previous document.
				try
				{
					jNode = jNode.getPreviousDocument(jNode);
					return true;
				}
				catch (XFlaimException ex)
				{
					if (ex.getRCode() == RCODE.NE_XFLM_DOM_NODE_NOT_FOUND)
					{
						return false;
					}
					else
					{
						JOptionPane.showMessageDialog(
									this,
									"Database Exception occurred: " + ex.getMessage(),
									"Database Exception",
									JOptionPane.ERROR_MESSAGE);
						return false;
					}
				}
			}
		}
		catch (XFlaimException e)
		{
			JOptionPane.showMessageDialog(
						this,
						"Database Exception occurred: " + e.getMessage(),
						"Database Exception",
						JOptionPane.ERROR_MESSAGE);
			return false;
		}
		
	}

	/**
	 * Fetch the previous node relative to the selected row.  Typically, the
	 * currently selected row is row 0 when this method is invoked.
	 * 
	 * @return boolean, signals whether the previous node was found or not.
	 */
	private boolean getPrevNode()
	{
		NodePanel		np = m_selected;
		int				iSelectedRow = np.getRow();
		DOMNode			jNode = null;
		Document			doc = null;
		
		try
		{
			jNode = m_jDb.getNode(np.getCollection(), np.getNodeId(), null);
			
			if (np.isClosing())
			{
				// See if we can get a child node
				if (jNode.hasChildren())
				{
					jNode = jNode.getLastChild(jNode);
					np = (NodePanel)m_vNodeList.get(iSelectedRow - 1);
					np.reset();
					np.buildLabel(jNode, true, false, m_bDocList);
					return true;
				}
			}
			
			// Check for a previous sibling.
			try
			{
				jNode = jNode.getPreviousSibling(jNode);
				np = (NodePanel)m_vNodeList.get(iSelectedRow - 1);
				np.reset();
				np.buildLabel(jNode, true, jNode.hasChildren() ? true : false, m_bDocList);
				return true;
			}
			catch (XFlaimException e)
			{
				// Do nothing, we just want to make sure we catch it.
			}
			
			// Check for a parent.
			if (np.getNodeId() != np.getDocId())
			{
				jNode = jNode.getParentNode(jNode);
				np = (NodePanel)m_vNodeList.get(iSelectedRow - 1);
				np.reset();
				np.buildLabel(jNode, true, false, m_bDocList);
				return true;
			}
			
			// Is there a previous document to get?
			doc = new Document("<" + jNode.getLocalName() + ">",
								np.getDocId(),
								np.getCollection());
			int iIndex = m_vDocList.indexOf(doc); 
			if (iIndex > 0)
			{
				doc = (Document)m_vDocList.get(iIndex - 1);
				jNode = m_jDb.getNode(doc.m_iCollection, doc.m_lDocId, jNode);
				np = (NodePanel)m_vNodeList.get(iSelectedRow - 1);
				np.reset();
				np.buildLabel(jNode, true, true, m_bDocList);
				return true;
			}
		}
		catch (XFlaimException e)
		{
			JOptionPane.showMessageDialog(
						this,
						"Database Exception occurred: " + e.getMessage(),
						"Database Exception",
						JOptionPane.ERROR_MESSAGE);
			return false;
		}

		return false;
	}

	/**
	 * 
	 */
	private void pageDown()
	{
		// Move down one full page worth.  The moveDown
		// method will stop if we run out of nodes to scroll to.
		for (int i = 0; i < LAST_NODE; i++)
		{
			moveDown();
		}
		
	}

	/**
	 * 
	 */
	private void pageUp()
	{
		// Move up one full page worth. The moveUp method
		// will stop if we reach the top.
		for (int i = 0; i < LAST_NODE; i++)
		{
			moveUp();
		}
		
	}

	/**
	 * 
	 */
	private void moveDown()
	{
		if (m_selected == null)
		{
			return;
		}

		if (m_selected.getRow() < m_iLastNode)
		{
			selectNode((NodePanel)m_vNodeList.get(m_selected.getRow()+ 1));
		}
		else
		{
			// We are going to have to move eveyone up one row first.
			// Then get the next row.
			if (scrollDown())
			{
				selectNode((NodePanel)m_vNodeList.get(m_selected.getRow() + 1));
			}
		}
	}

	/**
	 * Scroll the screen down one row.  This will pick up the
	 * next row from off the screen if there is one.
	 * @return
	 */
	private boolean scrollDown()
	{
		NodePanel		npTo;
		NodePanel		npFrom;
		
		if (m_selected == null)
		{
			return false;
		}
		else
		{
			if (hasNextNode())
			{
				for (int i = 0; i < LAST_NODE; i++)
				{
					npTo = (NodePanel)m_vNodeList.get(i);
					npFrom = (NodePanel)m_vNodeList.get(i + 1);
					npTo.copyNode(npFrom);
				}
				
				// Backup one row and select it.
				selectNode((NodePanel)m_vNodeList.get(m_selected.getRow() - 1));
		
				// Now we need to get the next row.  Due to the way this is
				// architected, we should be looking at the currently selected row
				// to determine the next row.  The currently selected row is supposed to
				// be the next to last row now.  If there isn't a next row, we will get a false
				// return value.
				if (!m_bDocList)
				{
					return(getNextNode(true));
				}
				else
				{
					return(getNextDocument());
				}
			}
			else
			{
				return false;
			}
		}
		
	}

	/**
	 * @return
	 */
	private boolean hasNextNode()
	{
		NodePanel	np = m_selected;
		DOMNode		jNode = null;
		Document		doc = null;
		
		try
		{
			jNode = m_jDb.getNode(np.getCollection(), np.getDocId(), null);
			
			if (!m_bDocList)
			{
				// Standard XEdit - build a Document object to compare with.
				doc = new Document("<" + jNode.getLocalName() + ">",
									np.getDocId(),
									np.getCollection());

				// If we're looking at the root node, then we need to find out if there
				// is another document to view after this one.
				if (np.getDocId() == np.getNodeId() &&
					(np.isClosing() || !np.isExpanded()))
				{
					int i = m_vDocList.indexOf(doc);
					if (i >= 0 && i < m_vDocList.size() - 1)
					{
						return true;
					}
					else
					{
						return false;
					}
				}
				else
				{
					return true;
				}
			}
			else
			{
				// This XEdit is a document list, so just see if we can get a next document.
				try
				{
					jNode = jNode.getNextDocument(jNode);
					return true;
				}
				catch (XFlaimException ex)
				{
					if (ex.getRCode() == RCODE.NE_XFLM_DOM_NODE_NOT_FOUND)
					{
						return false;
					}
					else
					{
						JOptionPane.showMessageDialog(
									this,
									"Database Exception Occurred" + ex.getMessage(),
									"Database Exception",
									JOptionPane.ERROR_MESSAGE);
						return false;
					}
				}
			}
		}
		catch (XFlaimException e)
		{
			JOptionPane.showMessageDialog(
						this,
						"Database Exception occurred: " + e.getMessage(),
						"Database Exception",
						JOptionPane.ERROR_MESSAGE);
			return false;
		}
		
	}

	/**
	 * Fetch the next node, relative to the currently selected node.  This may
	 * be the first child node, or the next sibling node, the parent node or the 
	 * root node of the next document.  If bStartAtChild is false, it will
	 * skip looking for the first child node.
	 * 
	 * @param bStartAtChild boolean parameter to signal whether to look for the
	 * first child or not.
	 * 
	 * @return boolean - next node found or not found.
	 */
	private boolean getNextNode(
		boolean			bStartAtChild)
	{
		NodePanel		np = m_selected;
		int				iSelectedRow = np.getRow();
		DOMNode			jNode = null;
		Document			doc = null;

		try
		{
			jNode = m_jDb.getNode(np.getCollection(), np.getNodeId(), jNode);
			
			if (bStartAtChild && !np.isClosing() && np.isExpanded())
			{
				// See if we can get a child node
				if (jNode.hasChildren())
				{
					jNode = jNode.getFirstChild(jNode);
					np = (NodePanel)m_vNodeList.get(iSelectedRow + 1);
					np.reset();
					np.buildLabel(jNode, true, false, m_bDocList);
					return true;
				}
			}
			
			// Check for a next sibling.
			try
			{
				jNode = jNode.getNextSibling(jNode);
				np = (NodePanel)m_vNodeList.get(iSelectedRow + 1);
				np.reset();
				np.buildLabel(jNode, true, false, m_bDocList);
				return true;
			}
			catch (XFlaimException e)
			{
				// Do nothing, we just want to make sure we catch it.
			}
			
			// Check for a parent.
			if (np.getNodeId() != np.getDocId())
			{
				jNode = jNode.getParentNode(jNode);
				np = (NodePanel)m_vNodeList.get(iSelectedRow + 1);
				np.reset();
				np.buildLabel(jNode, true, true, m_bDocList);
				return true;
			}
			
			// Is there a next document to get?
			doc = new Document("<" + jNode.getLocalName() + ">",
							   np.getDocId(),
							   np.getCollection());
			int iIndex = m_vDocList.indexOf(doc);
			if (iIndex < m_vDocList.size() - 1)
			{
				doc = (Document)m_vDocList.get(iIndex + 1);
				jNode = m_jDb.getNode(doc.m_iCollection, doc.m_lDocId, jNode);
				np = (NodePanel)m_vNodeList.get(iSelectedRow + 1);
				np.reset();
				np.buildLabel(jNode, true, false, m_bDocList);
				return true;
			}
		}
		catch (XFlaimException e)
		{
			JOptionPane.showMessageDialog(
						this,
						"Database Exception occurred: " + e.getMessage(),
						"Database Exception",
						JOptionPane.ERROR_MESSAGE);
			return false;
		}

		return false;

	}

	/* (non-Javadoc)
	 * @see java.awt.event.KeyListener#keyTyped(java.awt.event.KeyEvent)
	 */
	public void keyTyped(KeyEvent e)
	{
		// Do nothing
	}

	/*-------------------------------------------------------------------------
	 * Desc:	Method to highlight a row given the NodePanel.
	 *-----------------------------------------------------------------------*/
	private void selectNode(
		NodePanel		np)
	{
		long	lDocId;
		long	lNodeId;

		// Don't select anything if the NodePanel does not have a document
		// that is displaying.
		if (np.getDocId() == NO_NODES)
		{
			return;
		}

		if (m_selected != null)
		{
			m_selected.deselectNode();
		}

		m_selected = np;
		m_selected.selectNode();
		m_status.m_iCurrentNode = m_selected.getRow();
		lDocId = np.getDocId();
		m_status.setDocId(lDocId);
		lNodeId = np.getNodeId();
		m_status.setNodeId(lNodeId);
	}
	
	/*-------------------------------------------------------------------------
	 * Desc:	Method to expand a node one level given the NodePanel.
	 *-----------------------------------------------------------------------*/
	private void expandNode(
		NodePanel	np,
		boolean		bDeep)
	{
		DOMNode			refNode;
		DOMNode			childNode = null;
		boolean			bFirst = true;
		
		m_iExpandedRow = np.getRow();

		if (!np.hasChildren())
		{
			return;
		}
		
		if (np.isClosing())
		{
			return;
		}
		
		try
		{
			refNode = m_jDb.getNode(np.getCollection(), np.getNodeId(), null);
			np.buildLabel(refNode, true, false, m_bDocList);
			
			for (;m_iExpandedRow < LAST_NODE;)
			{
				if (bFirst)
				{
					bFirst = false;
					childNode = refNode.getFirstChild(null);
				}
				else
				{
					try
					{
						childNode = childNode.getNextSibling(childNode);
					}
					catch (XFlaimException e)
					{
						if (e.getRCode() == RCODE.NE_XFLM_DOM_NODE_NOT_FOUND)
						{
							break;
						}
						else
						{
							JOptionPane.showMessageDialog(
										this,
										"Database Exception occurred: " + e.getMessage(),
										"Database Exception",
										JOptionPane.ERROR_MESSAGE);
						}
					}
				}
					
				insertRow(++m_iExpandedRow, childNode, bDeep, false);
				if (bDeep)
				{
					expandNode((NodePanel)m_vNodeList.get(m_iExpandedRow), bDeep);
				}

			}
			
			// See if we have room to put in a closing row.
			if (m_iExpandedRow + 1 <= LAST_NODE)
			{
				insertRow(++m_iExpandedRow, refNode, true, true);
			}
		}
		catch (XFlaimException e)
		{
			JOptionPane.showMessageDialog(
						this,
						"Database Exception occurred: " + e.getMessage(),
						"Database Exception",
						JOptionPane.ERROR_MESSAGE);
		}
	}

	/*-------------------------------------------------------------------------
	 * Desc:	Method to collapse a node given the NodePanel.  This could result
	 * in multiple levels being collapsed.
	 *-----------------------------------------------------------------------*/
	private void collapseNode(
		NodePanel	np)
	{
		int			iOpeningRow;
		int			iClosingRow = -1;
		long		lNodeId;
		NodePanel	npOpening = np;
		NodePanel	npClosing = null;
		int			iCollection;
		boolean		bHasClosing = false;

		if (!np.isExpanded() || !np.hasChildren())
		{
			return;
		}
		
		lNodeId = np.getNodeId();
		iCollection = np.getCollection();
		
		try
		{
			DOMNode refNode = m_jDb.getNode(iCollection, lNodeId, null);
			if (np.isClosing())
			{
				bHasClosing = true;
				npClosing = np;
				iClosingRow = np.getRow();
				
				// Locate the opening instance of this node, if it is present on
				// the screen.  If it can't be found on the screen, this node will
				// become the opening node and will be displayed as the first row
				// on the screen.
				iOpeningRow = iClosingRow - 1;
				while (iOpeningRow >= 0)
				{
					npOpening = (NodePanel)m_vNodeList.get(iOpeningRow);
					if (npOpening.getNodeId() == lNodeId)
					{
						break;
					}
					--iOpeningRow;
				}
			}
			else
			{
				npOpening = np;
				iOpeningRow = np.getRow();
			}
		
			// Can we short circuit the process by checking for the root node?
			//if (lNodeId == np.getDocId())
			//{
			//	collapseDocument(npOpening);
			//	return;
			//}

			// Find the closing node.
			if (!bHasClosing)
			{
				for (iClosingRow = iOpeningRow + 1;
					 iClosingRow <= m_iLastNode;
					 iClosingRow++)
				{
					npClosing = (NodePanel)m_vNodeList.get(iClosingRow);
					if (npClosing.getNodeId() == lNodeId)
					{
						bHasClosing = true;
						break;
					}
				}
			}
			
			if (!bHasClosing)
			{
				clearNodes(npOpening.getRow() + 1, m_iLastNode);

				// When we build this node, we must first set the closing as true
				// so we can check for another node.  We will rebuild it again
				// when we finish.
				npOpening.buildLabel(refNode, false, false, m_bDocList);

				selectNode(npOpening);
				m_iLastNode = npOpening.getRow();
				m_status.m_iCurrentNode = m_iLastNode;
				m_status.m_iLastNode = m_iLastNode;
				m_status.updateLabel();

				// If we have just collapsed the last row, there is no need to get another row.
				if (m_iLastNode < LAST_NODE)
				{
					if (getNextNode(false))
					{
						m_iLastNode++;
						m_status.m_iLastNode = m_iLastNode;
						
						// Need to expand the new document that just got exposed.
						selectNode((NodePanel)m_vNodeList.get(m_selected.getRow() + 1)); // Will get reset on the way out.
						m_status.m_iCurrentNode = m_selected.getRow();
						m_status.updateLabel();
						np = m_selected;
						// Cannot expand the closing node of a document.
						if (np.isClosing())
						{
							if (m_iLastNode < LAST_NODE && getNextNode(false))
							{
								m_iLastNode++;
								selectNode((NodePanel)m_vNodeList.get(m_selected.getRow() + 1));
								m_status.m_iCurrentNode = m_selected.getRow();
								m_status.m_iLastNode = m_iLastNode;
								m_status.updateLabel();

								expandNode(m_selected, true);
							}
						}
						else
						{
							expandNode(m_selected, true);
						}

						selectNode(npOpening);

					}
				}
			}
			else  // Have a closing node
			{
				// If we have an opening node, then we need to redraw it first,
				// before we collapse the rest.
				if (iOpeningRow >= 0)
				{
					npOpening.buildLabel(refNode, false, false, m_bDocList);
				}

				// We need to remove rows one at a time until we get to the closing row.
				for (int i = iOpeningRow + 1; (iOpeningRow >= 0) ? i <= iClosingRow : i < iClosingRow; i++)
				{
					removeRow(iOpeningRow+1);
				}

				// If we did not have an opening row, we need to build the row now.
				if (iOpeningRow < 0)
				{
					npOpening.buildLabel(refNode, false, false, m_bDocList);
				}

				selectNode(npOpening);

				if (m_iLastNode < npOpening.getRow())
				{
					m_iLastNode = npOpening.getRow();
					m_status.m_iLastNode = m_iLastNode;
					m_status.updateLabel();
				}
			}
		}
		catch (XFlaimException e)
		{
			JOptionPane.showMessageDialog(
						this,
						"Database Exception occurred: " + e.getMessage(),
						"Database Exception",
						JOptionPane.ERROR_MESSAGE);
		}

	}
	
	/*-------------------------------------------------------------------------
	 * Desc:	Method to collapse a document given the root NodePanel.
	 *-----------------------------------------------------------------------*/
/*
	private void collapseDocument(
		NodePanel		npDoc)
	{
		DOMNode		docNode;
		
		try
		{
			docNode = m_jDb.getNode(npDoc.getCollection(), npDoc.getDocId(), null);
			
			clearNodes(npDoc.getRow() + 1, m_iLastNode);

			// When we build this node, we must first set the closing as true
			// so we can check for another document.  We will rebuid it again
			// when we finish.
			npDoc.buildLabel(docNode, false, true);

			selectNode(npDoc);
			m_iLastNode = m_iCurrentNode;
			
			if (getNextNode(false))
			{
				// Need to expand the new document that just got exposed.
				m_iCurrentNode++; // Will get reset on the way out.
				expandNode((NodePanel)m_vNodeList.get(m_iCurrentNode), true);
				
				selectNode(npDoc); // Resets m_iCurrentNode.

			}
			npDoc.buildLabel(docNode, false, false);

		}
		catch (XFlaimException e)
		{
			JOptionPane.showMessageDialog(
						this,
						"Database Exception occurred: " + e.getMessage(),
						"Database Exception",
						JOptionPane.ERROR_MESSAGE);
		}
	}
*/	
	/*-------------------------------------------------------------------------
	 * Desc:	Method to clear the nodes on display within the range specified.
	 *-----------------------------------------------------------------------*/
	private void clearNodes(
		int		iFromNode,
		int		iToNode)
	{
		if (iFromNode > iToNode)
		{
			return;
		}
		
		for (int i = iFromNode; i <= iToNode; i++)
		{
			NodePanel np = (NodePanel)m_vNodeList.get(i);
			np.reset();
		}
	}

	/*-------------------------------------------------------------------------
	 * Desc:	Method to move a row down one and insert a new row in its place.
	 *-----------------------------------------------------------------------*/
	private void insertRow(
		int				iRow,
		DOMNode			refNode,
		boolean			bExpanded,
		boolean			bClosing)
	{
		NodePanel		npCopy = null;
		NodePanel		npInsert = null;
		
		// Move them down one row to make room.
		for (int i = LAST_NODE; i > iRow; i--)
		{
			npInsert = (NodePanel)m_vNodeList.get(i);
			npCopy = (NodePanel)m_vNodeList.get(i-1);
			npInsert.copyNode(npCopy);
		}
		
		// Copy the node that is at the current location into a temporary
		// NodePanel.
		npInsert = (NodePanel)m_vNodeList.get(iRow);
		npCopy = new NodePanel(this, 0);  // The row is not copied across NodePanels.
		npCopy.copyNode(npInsert);
		try
		{
			npInsert.buildLabel(refNode, bExpanded, bClosing, m_bDocList);
			// We cannot hold more than LAST_NODE nodes, so we only increment
			// the count till we get there.
			if (m_iLastNode < LAST_NODE)
			{
				m_iLastNode++;
				m_status.m_iLastNode = m_iLastNode;
				m_status.updateLabel();
			}
		}
		catch (XFlaimException e)
		{
			JOptionPane.showMessageDialog(
						this,
						"Database Exception occurred: " + e.getMessage(),
						"Database Exception",
						JOptionPane.ERROR_MESSAGE);
		}
	}

	/*-------------------------------------------------------------------------
	 * Desc:	Method to remove a row and move all other rows up one position.
	 * 			It will also add a new row to the bottom display if it can.
	 *-----------------------------------------------------------------------*/
	private void removeRow(
		int				iRow
		)
	{
		NodePanel		npCopy = null;
		NodePanel		npInsert = null;
		NodePanel		np;
		NodePanel		npSelected = m_selected;
		
		// Move everybody up one row.
		for (int i = iRow; i < LAST_NODE; i++)
		{
			npInsert = (NodePanel)m_vNodeList.get(i);
			npCopy = (NodePanel)m_vNodeList.get(i+1);
			npInsert.copyNode(npCopy);
		}

		// We have one less node being displayed now.
		m_iLastNode--;
		m_status.m_iLastNode = m_iLastNode;

		// Clear the last line incase there is nothing left to display.
		np = (NodePanel)m_vNodeList.get(m_iLastNode + 1);
		np.reset();
		
		// Since we have one more row, see if we can expand it.
		// Make sure we are looking at a valid row.
		if (m_iLastNode >= 0)
		{
			selectNode((NodePanel)m_vNodeList.get(m_iLastNode));
		}
		else
		{
			m_selected = null;
		}
		m_status.m_iCurrentNode = m_iLastNode;
		m_status.updateLabel();

		if (m_selected != null)
		{
			if (getNextNode(true))
			{
				m_iLastNode++;
				m_status.m_iLastNode = m_iLastNode;
				m_status.updateLabel();
			}
		}

		// If we just removed the first display row and there are no more rows
		// to display, then we should clear the currently selected row (node).
		if (m_selected != null)
		{
			selectNode(npSelected);
			m_status.m_iCurrentNode = npSelected.getRow();
			m_status.updateLabel();
		}
	}

	/*-------------------------------------------------------------------------
	 * Desc:	Method to show the initial document listing.
	 *-----------------------------------------------------------------------*/
	private boolean showDocumentList()
	{
		boolean		bFirst = true;
		DOMNode		jDoc = null;
		NodePanel	np;
		int			iRow = 0;

		// We will clear the screen first.
		clearNodes(0, LAST_NODE);
		
		try
		{
			for (iRow = 0; iRow <= LAST_NODE; iRow++)
			{
				m_status.m_iCurrentNode = iRow;
				m_status.updateLabel();
				if (bFirst)
				{
					bFirst = false;
					jDoc = m_jDb.getFirstDocument(m_iCollection, jDoc);
				}
				else
				{
					jDoc = jDoc.getNextDocument(jDoc);
				}
				np = (NodePanel)m_vNodeList.get(iRow);
				np.buildLabel(jDoc, false, false, m_bDocList);

			}
			m_iLastNode = LAST_NODE;
			m_status.m_iLastNode = m_iLastNode;
			m_status.updateLabel();
		}
		catch (XFlaimException e)
		{
			if (e.getRCode() != RCODE.NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				JOptionPane.showMessageDialog(
							this,
							"Database Exception occurred: " + e.getMessage(),
							"Database Exception",
							JOptionPane.ERROR_MESSAGE);
			}
			else
			{
				if (iRow == 0)
				{
					return false;
				}
				else
				{
					m_iLastNode = iRow - 1;
					m_status.m_iLastNode = m_iLastNode;
					m_status.updateLabel();
				}
			}
		}
		selectNode((NodePanel)m_vNodeList.get(0));
		m_status.m_iCurrentNode = 0;
		m_status.updateLabel();
		
		return true;
	}
	
	private boolean getNextDocument()
	{
		DOMNode		jDoc = null;
		NodePanel	np = m_selected;
		int			iSelectedRow = np.getRow();
		
		try
		{
			jDoc = m_jDb.getNode(m_iCollection, np.getDocId(), jDoc);
			jDoc = jDoc.getNextDocument(jDoc);
			insertRow(iSelectedRow + 1, jDoc, false, false);
			return true;
		}
		catch (XFlaimException e)
		{
			JOptionPane.showMessageDialog(
						this,
						"Database Exception occurred: " + e.getMessage(),
						"Database Exception",
						JOptionPane.ERROR_MESSAGE);
		}
		return false;
	}

	private boolean getPrevDocument()
	{
		DOMNode		jDoc = null;
		NodePanel	np = m_selected;
		int			iSelectedRow = np.getRow();
		
		try
		{
			jDoc = m_jDb.getNode(m_iCollection, np.getDocId(), jDoc);
			jDoc = jDoc.getPreviousDocument(jDoc);
			insertRow(iSelectedRow - 1, jDoc, false, false);
			return true;
		}
		catch (XFlaimException e)
		{
			JOptionPane.showMessageDialog(
						this,
						"Database Exception occurred: " + e.getMessage(),
						"Database Exception",
						JOptionPane.ERROR_MESSAGE);
		}
		return false;
	}

	/* (non-Javadoc)
	 * @see java.lang.Runnable#run()
	 */
	public void run()
	{
		if (m_bMustExit)
		{
			this.dispose();
		}
		else
		{
			setVisible(true);
		}
		
	}

	/**
	 * Method to add a new document node to the current database.  This method
	 * assumes that an UPDATE transaction has already been started.  If one
	 * has not been started, an exception will be thrown.
	 * @param iCollection
	 * @param iTag
	 * @param sNodeValue
	 * @return lDocId The new document root node id.
	 * @throws XFlaimException
	 */
	public long addDocument(
		int					iCollection,
		int					iTag,
		String				sNodeValue) throws XFlaimException
	{
		DOMNode		jNode = null;
		long			lNodeId = 0;
		
		if (m_status.getTransaction())
		{
			
			jNode = m_jDb.createRootElement(iCollection, iTag);
		
			lNodeId = jNode.getNodeId();
		
			if (sNodeValue != null && sNodeValue.length() > 0)
			{
				// Add an annotation note to the document.
				jNode.createAnnotation(jNode);
				jNode.setString(sNodeValue);
			}
		}
		else
		{
			JOptionPane.showMessageDialog(
						this,
						"Illegal State: Action requires Update Transaction",
						"Illegal State",
						JOptionPane.ERROR_MESSAGE);
		}
		
		return lNodeId;
	}
	
	/**
	 * Method to add a new node to an existing document, relative to the
	 * lNodeId parameter.
	 * @param iCollection
	 * @param iNodeType
	 * @param iNodeTag
	 * @param sNodeValue
	 * @param bIsSiblingNode
	 * @param lNodeId
	 * @return
	 * @throws XFlaimException
	 */
	public long addNode(
		int					iCollection,
		int					iNodeType,
		int					iNodeTag,
		String				sNodeValue,
		boolean				bIsSiblingNode,
		long				lNodeId) throws XFlaimException
	{
		long			lAnchorNode = lNodeId;
		boolean			bIsAttribute = false;

		if (m_status.getTransaction())
		{
			DOMNode			jNode = m_jDb.getNode(
									iCollection,
									lNodeId,
									null);
		
			if (iNodeType == FlmDomNodeType.ATTRIBUTE_NODE)
			{
				jNode = jNode.createAttribute(
									iNodeTag,
									jNode);
				bIsAttribute = true;
			}
			else
			{
				if (bIsSiblingNode)
				{
					// Create the sibling node
					jNode = jNode.createNode(
									iNodeType,
									iNodeTag,
									FlmInsertLoc.FLM_NEXT_SIB,
									jNode);
				}
				else
				{
					jNode = jNode.createNode(
									iNodeType,
									iNodeTag,
									FlmInsertLoc.FLM_LAST_CHILD,
									jNode);
				}
			}

			// Set the value.
			if (sNodeValue != null && sNodeValue.length() > 0)
			{
				jNode.setString(sNodeValue);
			}
			
			if (m_selected != null &&
				lAnchorNode == m_selected.getNodeId())
			{
				// See if we can refresh the selected node/row
				NodePanel np = m_selected;
				DOMNode jRefNode = m_jDb.getNode(
										np.getCollection(),
										lAnchorNode,
										null);

				if (np.isExpanded() && np.isClosing())
				{
					collapseNode(np);
					// The original selected node may have changed.  We now want to
					// expand the currently selected node.
					expandNode(m_selected, true);
				}
				else
				{
					np.buildLabel(jRefNode, np.isExpanded(), np.isClosing(), false);
				}
			}
		
			if (bIsAttribute)
			{
				return lAnchorNode;
			}
			else
			{
				return jNode.getNodeId();
			}
		}
		else
		{
			JOptionPane.showMessageDialog(
						this,
						"Illegal State: Action requires Update Transaction",
						"Illegal State",
						JOptionPane.ERROR_MESSAGE);
		}
		
		return 0;

	}

	/**
	 * The main program.
	 * @param args
	 */
	public static void main(String[] args) {
		try {
			UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
		} catch (Exception e) {
			System.err.println(
				"Couldn't use the system look and feel: " + e.getMessage());
			e.printStackTrace();
		}

		JFrame ed = new XEdit("XFlaim Editor");
		GraphicsConfiguration gc = ed.getGraphicsConfiguration();
		Rectangle bounds = gc.getBounds();
		int iHeight = ed.getHeight();
		int iWidth = ed.getWidth();
		ed.setLocation(Math.max(0, (bounds.width - iWidth)/2),
						Math.max(0, (bounds.height - iHeight)/2));
		Thread t = new Thread((Runnable)ed);
		t.start();
	}

	/**
	 * Method to add an annotation node to an existing node.
	 * @param iCollection
	 * @param sAnnotation
	 * @param lNodeId
	 * @return
	 */
	public long addAnnotation(
		int			iCollection,
		String		sAnnotation,
		long		lNodeId) throws XFlaimException
	{

		if (m_status.getTransaction())
		{
			DOMNode		jNode = m_jDb.getNode(
									iCollection,
									lNodeId,
									null);
		
			jNode = jNode.createAnnotation( jNode);
			
			// Set the value.
			if (sAnnotation != null && sAnnotation.length() > 0)
			{
				jNode.setString(sAnnotation);
			}
			
			if (m_selected != null &&
				lNodeId == m_selected.getNodeId())
			{
				// See if we can refresh the selected node/row
				NodePanel np = m_selected;
				DOMNode jRefNode = m_jDb.getNode(
										np.getCollection(),
										lNodeId,
										null);

				if (np.isExpanded() && np.isClosing())
				{
					collapseNode(np);
					// The original selected node may have changed.  We now want to
					// expand the currently selected node.
					expandNode(m_selected, true);
				}
				else
				{
					np.buildLabel(jRefNode, np.isExpanded(), np.isClosing(), false);
				}
			}
		
			return lNodeId;
		}
		else
		{
			JOptionPane.showMessageDialog(
						this,
						"Illegal State: Action requires Update Transaction",
						"Illegal State",
						JOptionPane.ERROR_MESSAGE);
		}
		
		return 0;
	}
}
