//------------------------------------------------------------------------------
// Desc:	Node Panel
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
import javax.swing.*;

/**
 * The Nodepanel class represents the DOM node that is being shown in the XEdit
 * main display window.  It preseves essential state information.  It acts as 
 * as repository for the XEdit main body to interact with.  There is a fixed
 * number of NodePanel objects in the XEdit main display that never changes.
 * As the user scrolls up or down, causing the nodes to be repositioned,
 * the content of each NodePanel is copied to the NodePanel immediately
 * adjacent to it.  The NodePanels themselves are fixed in their location.
 */
public class NodePanel extends JPanel
{
	private boolean				m_bExpanded;
	private boolean				m_bHasChildren;
	private long				m_lNodeId;
	private long				m_lDocId;
	private int					m_iCollection;
	private XEdit				m_root;
	private JLabel				m_label;
	private GridBagLayout		m_gbLayout;
	private GridBagConstraints	m_gbConstraints;
	private int					m_iRow;
	private boolean				m_bClosing;

	/**
	 * Constructor
	 */
	public NodePanel(
		XEdit		root,
		int			iRow)
	{
		m_lNodeId = -1;
		m_lDocId = -1;
		m_iCollection = -1;
		m_root = root;
		m_iRow = iRow;
		m_bHasChildren = false;
		m_bExpanded = false;
		m_bClosing = false;
		addMouseListener(m_root);

		m_label = new JLabel();
		m_label.setFont(new Font("Courier", Font.PLAIN, 16));
		Dimension d = root.getSizeOfDisplay();
		setSize(d.width, 17);
		setMinimumSize(new Dimension(d.width, 17));
		
		m_gbLayout = new GridBagLayout();
		m_gbConstraints = new GridBagConstraints();
		setLayout(m_gbLayout);
		addToPanel(m_label, 0, 0, 1, 1, 100, 100, false);
		
		setVisible(true);
	}
	
	/**
	 * Method reset the node panel.
	 */
	public void reset()
	{
		m_lNodeId = -1;
		m_iCollection = -1;
		m_bHasChildren = false;
		m_bExpanded = false;
		m_bClosing = false;
		m_lDocId = -1;
		setLabelText("");
		m_label.setForeground(Color.BLACK);
		Dimension d = m_root.getSizeOfDisplay();
		setSize(d.width, 17);
	}
	
	/**
	 * Method return the row number that this node occupies
	 */
	public int getRow()
	{
		return m_iRow;
	}
	
	/**
	 * Method to get the expanded state of the node
	 */
	public boolean isExpanded()
	{
		return m_bExpanded;
	}

	/**
	 * Method return the children status of the node (hasChildren)
	 */
	public boolean hasChildren()
	{
		return m_bHasChildren;
	}

	/**
	 * Method return the node Id of the node
	 */
	public long getNodeId()
	{
		return m_lNodeId;
	}

	/**
	 * Method return the document Id of the node
	 */
	public long getDocId()
	{
		return m_lDocId;
	}

	/**
	 * Method return the closing state of the node, i.e. is this a
	 * closing node after expansion.
	 */
	public boolean isClosing()
	{
		return m_bClosing;
	}


	/**
	 * Method to set the forground of the label to show that this node is
	 * selected.
	 */
	public void selectNode()
	{
		if (m_lDocId > 0)
		{
			m_label.setForeground(Color.MAGENTA);
		}
	}

	/**
	 * Method to reset the forground of the label to show that this node is
	 * selected.
	 */
	public void deselectNode()
	{
		m_label.setForeground(Color.BLACK);
	}
	
	/**
	 * Method to retrieve the collection for the current node.
	 */
	public int getCollection()
	{
		return m_iCollection;	
	}
	
	/**
	 * Method to copy a NodePanel.  The Nodepane np will be copied into
	 * this node.
	 */
	public void copyNode(
		NodePanel		np)
	{
		m_bClosing = np.m_bClosing;
		m_bExpanded = np.m_bExpanded;
		m_bHasChildren = np.m_bHasChildren;
		m_iCollection = np.m_iCollection;
		m_lDocId = np.m_lDocId;
		m_lNodeId = np.m_lNodeId;
		m_label.setText(new String(np.m_label.getText()));	
	}

	/**
	 * Method to set the text that is being displayed by the NodePanel.
	 * @param sText
	 */
	private void setLabelText(String sText)
	{
		m_label.setText(sText);
	}
	
	/**
	 * Sets up everything needed to present the DOM node as a text string in
	 * the NodePanel.  All internal state variables are set accordingly.
	 * 
	 * @param jDb
	 * @param refNode
	 * @param bExpanded
	 * @param bClosing
	 * @throws XFlaimException
	 */
	public void buildLabel(
		DOMNode		refNode,
		boolean		bExpanded,
		boolean		bClosing,
		boolean		bList) throws XFlaimException
	{
		String		sText = null;
		String		sExpand;
		String		sTag = null;
		String		sName = null;
		String		sPrefix = null;
		String		sValue = null;
		String[]		saAttrTags = new String[100];
		String[]		saAttrValues = new String[100];
		DOMNode		attrNode = null;
		int			iNumAttrs = 0;
		int			iLevel;
		int			iNodeType = refNode.getNodeType();

		m_bHasChildren = refNode.hasChildren();
		m_bExpanded = bExpanded;
		m_lNodeId = refNode.getNodeId();
		m_lDocId = refNode.getDocumentId();
		if (m_lDocId == 0)
		{
			m_lDocId = m_lNodeId;
		}
		m_bClosing = bClosing;
		m_iCollection = refNode.getCollection();

		iLevel = getLevel(refNode);

		//The first thing on the label is a Plus sign (if the node has children).
		if (m_bHasChildren && !bList)
		{
			if (m_bExpanded)
			{
				sExpand = "-";
			}
			else
			{
				sExpand = "+";
			}
		}
		else
		{
			sExpand = new String(" ");
		}

		if ((iNodeType != FlmDomNodeType.DATA_NODE) &&
			(iNodeType != FlmDomNodeType.COMMENT_NODE))
		{
			sPrefix = refNode.getPrefix();
			sName = refNode.getLocalName();
			if (sPrefix != null && !sPrefix.equals(""))
			{
				sTag = sPrefix + ":" + sName;
			}
			else
			{
				sTag = sName;
			}
		}
		
		// Check here to see if this document has only one node (ignoring attributes)
		// We want to make sure we assign the closing "/" only when the node that is
		// marked as closing, truly is closing off other nodes.  This is due to
		// the way the up/down keys are used.
		if (bClosing)
		{
			if (!m_bHasChildren && m_lDocId == m_lNodeId)
			{
				m_bClosing = bClosing = false;
			}
		}

		if (!bClosing)
		{
			if (refNode.hasAttributes())
			{
				boolean		bFirst = true;
			
				for (iNumAttrs = 0;;iNumAttrs++)
				{
					// We don't know how many attributes this node has, so we will have to rely on catching an exception to 
					// break out of this loop.
					try
					{
						if (bFirst)
						{
							attrNode = refNode.getFirstAttribute(null);
							bFirst = false;
						}
						else
						{
							attrNode = attrNode.getNextSibling(attrNode);
						}
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
										"Exception occurred in Database: " + e.getMessage(),
										"Database Exception",
										JOptionPane.ERROR_MESSAGE);
							return;
						}
					} // try/catch
	
					try
					{
						if (saAttrTags.length < iNumAttrs)
						{
							JOptionPane.showMessageDialog(
										this,
										"Too many attributes",
										"Internal Error",
										JOptionPane.ERROR_MESSAGE);
						}
						sPrefix = attrNode.getPrefix();
						sName = attrNode.getLocalName();
						if (sPrefix != null && !sPrefix.equals(""))
						{
							saAttrTags[iNumAttrs] = sPrefix + ":" + sName;
						}
						else
						{
							saAttrTags[iNumAttrs] = sName;
						}
						saAttrValues[iNumAttrs] = getNodeValue(attrNode);
					}
					catch (XFlaimException e)
					{
						JOptionPane.showMessageDialog(
									this,
									"Exception occurred in Database: " + e.getMessage(),
									"Database Exception",
									JOptionPane.ERROR_MESSAGE);
						return;
					}
				
				} // for - getting attributes.
			}
		}

		// Now build the actual label.
		sText = sExpand + " ";
		if (!bList)
		{
			for (int i=0; i < iLevel; i++)
			{
				sText = sText + "   ";
			}
			sText = sText + new Integer(iLevel).toString() + " ";
		}
		if (iNodeType == FlmDomNodeType.ELEMENT_NODE)
		{
			sText = sText + "<";
			if (bClosing)
			{
				sText = sText + "/";
			}
			sText = sText + sTag;

			for (int i = 0; i < iNumAttrs; i++)
			{
				sText = sText + " " + saAttrTags[i] + "=\"" + saAttrValues[i] + "\"";
			}

			if (!bClosing)
			{
				if (m_bHasChildren && !m_bExpanded)
				{
					sText = sText + "/>";
				}
				else
				{
					sText = sText + ">";
				}
			}
			else
			{
				sText = sText + ">";
			}
			// Check for annotations.
			if (!bClosing && refNode.hasAnnotation())
			{
				DOMNode jAnnot = refNode.getAnnotation(null);
				sText += jAnnot.getString();
			}
		}
		else // Comment or Data Node.  Don't handle other types right now.
		{
			if (iNodeType == FlmDomNodeType.COMMENT_NODE)
			{
				sText = sText + "<!--";
			}
			// Get the value of the data node.
			sValue = getNodeValue(refNode);
			sText = sText + sValue;
			if (iNodeType == FlmDomNodeType.COMMENT_NODE)
			{
				sText = sText + "-->";
			}
		}
		if (bList)
		{
			// Need to make sure we don't let the value get too long on a list.  This
			// is a temporary measure to help find why the list page when listing
			// dictionary documents  hooks the key events onto the horizontal scroll
			// bar.
			if (sText.length() > 65)
			{
				sText = sText.substring(0, 64);
			}
		}
		m_label.setText(sText);
		Dimension d = getSize();
		setSize(m_label.getWidth(), d.height);
	}

	/**
	 * Method to calculate the indentation level of the node.
	 * 
	 * @param refNode
	 * @return
	 * @throws XFlaimException
	 */
	private int getLevel(
		DOMNode			refNode) throws XFlaimException
	{
		DOMNode			parent = null;
		int				iLevel = 0;
		boolean			bFirst = true;
		
		for (;;)
		{
			try
			{
				if (bFirst)
				{
					bFirst = false;
					parent = refNode.getParentNode(null);
				}
				else
				{
					parent = parent.getParentNode(parent);
				}
				iLevel++;
			}
			catch (XFlaimException e)
			{
				if (e.getRCode() == RCODE.NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					break;
				}
				else
				{
					throw e;
				}
			}
		}

		return iLevel;		

	}

	/**
	 * Method to retrieve the node Value as a text string suitable for
	 * displaying.
	 * 
	 * @param jNode
	 * @return
	 */
	private String getNodeValue(
		DOMNode		jNode)
	{
		int			iDataType;
		String		sValue = null;

		try
		{

			iDataType = jNode.getDataType();

			switch (iDataType)
			{
				case FlmDataType.FLM_TEXT_TYPE:
				{
					sValue = jNode.getString();
					break;
				}
				case FlmDataType.FLM_NUMBER_TYPE:
				{
					long		lValue = jNode.getLong();
					sValue = (new Long(lValue)).toString();
					break;
				}
				case FlmDataType.FLM_BINARY_TYPE:
				{
					// No supported yet.
				}
				default:
			
			}
		}
		catch (XFlaimException e)
		{
			JOptionPane.showMessageDialog(this,
										  "Database Exception occurred: " + e.getMessage(),
										  "Database Exception",
										  JOptionPane.ERROR_MESSAGE);
			return null;
		}
		
		
		return sValue;
	}

	/**
	 * Adds the internal component (label) to the NodePanel, setting the
	 * constraints as specified.
	 * 
	 * @param comp
	 * @param iColumn
	 * @param iRow
	 * @param iColSpan
	 * @param iRowSpan
	 * @param iColWeight
	 * @param iRowWeight
	 * @param bUpdateDisplay
	 */
	private void addToPanel(
		Component		comp,
		int				iColumn,
		int				iRow,
		int				iColSpan,
		int				iRowSpan,
		int				iColWeight,
		int				iRowWeight,
		boolean		bUpdateDisplay)
	{
		UITools.buildConstraints(m_gbConstraints, iColumn, iRow, iColSpan, iRowSpan, iColWeight, iRowWeight);
		m_gbConstraints.anchor = GridBagConstraints.LINE_START;
		m_gbConstraints.fill = GridBagConstraints.NONE;
		m_gbLayout.addLayoutComponent(comp, m_gbConstraints);
		add(comp);
		if (bUpdateDisplay)
		{
			m_root.updateDisplay();
		}
	}

	/* (non-Javadoc)
	 * @see java.awt.event.MouseListener#mouseClicked(java.awt.event.MouseEvent)
	 */
//	public void mouseClicked(MouseEvent e)
//	{
//		Point				p;
		
		// Select the row.
//		m_root.selectRow(this);
		
//		if (e.getButton() == MouseEvent.BUTTON3)
//		{
			// popup the document menu.
//			m_popup = new JPopupMenu("Document Node Options");
//			m_popup.addFocusListener(this);
//			if (m_root.isMainPanel((JPanel)this.getParent()))
//			{
//				m_closeItem = m_popup.add("Close Document");
//				m_closeItem.addActionListener(this);
//				m_popup.addSeparator();
//			}

//			if (m_bHasChildren)
//			{
//				if (m_bExpanded)
//				{
//					m_expandCollapseItem = m_popup.add("Collapse Node");
//				}
//				else
//				{
//					m_expandCollapseItem = m_popup.add("Expand Node");
//				}
//				m_expandCollapseItem.addActionListener(this);
//				m_popup.addSeparator();
//			}

//			m_deleteItem = m_popup.add("Delete Node");
//			m_deleteItem.addActionListener(this);
//			m_editItem = m_popup.add("Edit Node");
//			m_editItem.setEnabled(false);
//			m_editItem.addActionListener(this);
			
			// Need to make sure the menu appears close to where our window is located.
//			p = e.getPoint();
//			m_popup.show(this, p.x, p.y);
//		}
		
//		if (e.getButton() == MouseEvent.BUTTON1)
//		{
//			if (e.getClickCount() > 1)
//			{
				// Expand or Collapse the node.
//				if (m_bExpanded)
//				{
//					collapse();
//				}
//				else
//				{
//					expand();
//				}
//			}
//		}
		
//	}


}
