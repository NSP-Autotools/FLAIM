//------------------------------------------------------------------------------
// Desc:	Default Restore Client
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

package xflaim;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import xflaim.RCODE;

/**
 * This is a simple example of a class that implements the
 * {@link RestoreClient RestoreClient} interface.  It restores from a
 * backup file created with {@link DefaultBackupClient
 * DefaultBackupClient}.  Note that this class is mostly intended as a
 * demonstration.  It only restores full backups; it cannot handle incremental
 * or hot, continuous backups.
 */
public class DefaultRestoreClient implements RestoreClient
{
	public DefaultRestoreClient(
		String		sBackupPath)
	{
		m_sBackupPath = sBackupPath;
	}
	
	/* (non-Javadoc)
	 * @see xflaim.RestoreClient#openBackupSet()
	 */
	public int openBackupSet()
	{
		int iRc = RCODE.NE_XFLM_OK; 

		// Note: This rather odd name comes from the desire to maintain
		// compatibility with C++ default backup client
		
		String sPathName = m_sBackupPath +
						   System.getProperty( "file.separator") +
						   "00000000.64";
		
		try
		{
			m_IStream = new FileInputStream( sPathName);
		}
		catch ( FileNotFoundException e)
		{
			iRc = RCODE.NE_XFLM_IO_PATH_NOT_FOUND;
		}
		
		return( iRc);
	}

	/* (non-Javadoc)
	 * @see xflaim.RestoreClient#openRflFile(int)
	 */
	public int openRflFile(int iFileNum)
	{
		// This function is not, and probably never will be, implemented.
		// If you really want to restore from a hot, continuous backup,
		// you should write your own implementations of RestoreClient
		// (and BackupClient).
		
		return( RCODE.NE_XFLM_FAILURE);
	}

	/* (non-Javadoc)
	 * @see xflaim.RestoreClient#openIncFile(int)
	 */
	public int openIncFile(int iFileNum)
	{		
		// This function is not, and probably never will be, implemented.
		// If you really want to restore from an incremental backup,
		// you should write your own implementations of RestoreClient
		// (and BackupClient).
		// Note that this function will still be called by XFlaim.  Returning
		// PATH_NOT_FOUND is what tells XFLaim that there are no more incremental
		// backups.
		
		return( RCODE.NE_XFLM_IO_PATH_NOT_FOUND);
	}

	/* (non-Javadoc)
	 * @see xflaim.RestoreClient#read(int, byte[])
	 */
	public int read(byte[] Buffer, int[] BytesRead)
	{
		// Try to read Buffer.length bytes from the file.
		// Store the actual bytes read in BytesRead[0]. Note that
		// BytesRead will have a length of 1.
		
		int 	iBytesRead = 0;
		int	iRc = RCODE.NE_XFLM_OK;

		try
		{
			iBytesRead = m_IStream.read( Buffer);
		}
		catch (IOException e)
		{
			iRc = RCODE.NE_XFLM_FAILURE;
		}
		
		if( iBytesRead == -1)
		{
			iBytesRead = 0;
			iRc = RCODE.NE_XFLM_IO_END_OF_FILE;
		}
		
		BytesRead[0] = iBytesRead;
		
		return( iRc);
	}

	/* (non-Javadoc)
	 * @see xflaim.RestoreClient#close()
	 */
	public int close() 
	{
		int	iRc = RCODE.NE_XFLM_OK;
		
		try
		{ 
			m_IStream.close();
		}
		catch ( IOException e)
		{
			iRc = RCODE.NE_XFLM_FAILURE;
		}
		
		m_IStream = null;
		return( iRc);
	}

	/* (non-Javadoc)
	 * @see xflaim.RestoreClient#abortFile()
	 */
	public int abortFile() {
		return close();
	}

	private String					m_sBackupPath;
	private FileInputStream		m_IStream;
}
