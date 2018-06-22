//------------------------------------------------------------------------------
// Desc:	File Importer
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
import java.io.*;
import javax.swing.*;

/**
 * The FileImporter runs in a separate thre so that it does not block the main
 * XEdit thread while it imports documents into the database.
 */
public class FileImporter extends Thread
{

	/**
	 * 
	 */
	public FileImporter(
		Frame				owner,
		DbSystem			dbSystem,
		String			sDbName,
		int				iCollection,
		String			sFilename,
		String			sDirectory,
		String []		sList) throws XFlaimException
	{
		super();
		m_iCollection = iCollection;
		m_sFilename = sFilename;
		m_sDirectory = sDirectory;
		m_sList = sList;
		m_dbSystem = dbSystem;
		m_owner = owner;

		// Open our own copy of the database.
		m_jDb = dbSystem.dbOpen( sDbName, null, null, null, true);
	}
	
	// Need to close the database we when we are done.
	public void finalize()
	{
		m_jDb.close();
	}

	/* (non-Javadoc)
	 * @see java.lang.Runnable#run()
	 */
	public void run()
	{
		countFiles( m_sFilename,
						 m_sDirectory,
						 m_sList);
		
		m_box = new ProgressBox(m_owner, 500, 160, "Preparing to import files", "Filename:", 0, m_iFileCount);

		importFile( m_iCollection,
						 m_sFilename,
						 m_sDirectory,
						 m_sList);

	}

	/*=========================================================================
	 * Desc: 	Method to iterate through a possible list of files and import every file in the directory.
	 *========================================================================*/
	private void importFile(
		int				iCollection,
		String			sFilename,
		String			sDirectory,
		String []		sList)
	{
		boolean				bTransBegun = false;
		IStream 				jIStream = null;
		
		if (m_box.Cancelled())
		{
			m_box.terminate();
			return;
		}

		// A null sList indicates that we are looking at a file, so we can import it directly.
		if (sList == null)
		{
			m_iCurrentFile++;
			m_box.setLabel1("Importing file " + m_iCurrentFile + " of " + m_iFileCount);
			m_box.setLabel2("Filename: " + sFilename);

			try
			{
				jIStream = m_dbSystem.openFileIStream(sFilename);
				m_jDb.transBegin(TransactionType.UPDATE_TRANS, 0, 0);
				bTransBegun = true;
				m_jDb.Import(jIStream, iCollection);
				m_jDb.transCommit();
				bTransBegun = false;
			}
			catch (XFlaimException e)
			{
				// Abort the transaction, then throw the exception up further.
				if (bTransBegun)
				{
					try
					{
						m_jDb.transAbort();
					}
					catch (XFlaimException ex)
					{
						JOptionPane.showMessageDialog(null, "Exception occurred in import thread: " + ex.getMessage());
					}
				}
				JOptionPane.showMessageDialog(null, "Exception occurred in import thread: " + e.getMessage());
			}
			m_box.updateProgress(m_iCurrentFile);
		}
		// Otherwise we need to find the first file.  If we come across a directory, we need to search it too.
		else
		{
			for (int iLoop = 0; iLoop < sList.length; iLoop++)
			{
				File f = new File(sFilename, sList[iLoop]);
				if (f.isDirectory())
				{
					importFile( iCollection,
									 new String(f.getPath()),
									 new String(f.getAbsolutePath()),
									 f.list());
				}
				else
				{
					importFile( iCollection,
									 new String(f.getPath()),
									 sDirectory,
									 null);
				}
			}
		}
	}

	private void countFiles(
		String			sFilename,
		String			sDirectory,
		String []		sList)
	{
		if (sList == null)
		{
			m_iFileCount++;
		}
		else
		{
			for (int iLoop = 0; iLoop < sList.length; iLoop++)
			{
				File f = new File(sFilename, sList[iLoop]);
				if (f.isDirectory())
				{
					countFiles( new String(f.getPath()),
									 new String(f.getAbsolutePath()),
									 f.list());
				}
				else
				{
					countFiles( new String(f.getPath()),
									 sDirectory,
									 null);
				}
			}

		}
	}
	
	/*----------------------------------------- Private members -----------------------------------*/
	private int					m_iCollection;
	private String				m_sFilename;
	private String				m_sDirectory;
	private String []			m_sList;
	private DbSystem			m_dbSystem;
	private Db					m_jDb;
	private ProgressBox		m_box;
	private int					m_iFileCount = 0;
	private int					m_iCurrentFile = 0;
	private Frame				m_owner;
}
