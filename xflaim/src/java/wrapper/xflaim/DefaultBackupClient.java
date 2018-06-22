//------------------------------------------------------------------------------
// Desc:	Default Backup Client
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

package xflaim;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import xflaim.RCODE;

/**
 * This is a simple example of an object that implements the <code>
 * BackupClient</code> interface.  It writes the backed up data
 * to a file.  Note that like the C++ default backup, this class can
 * only be used for full backups.  This class cannot be used for an
 * incremental backup.
 */
public class DefaultBackupClient
		implements BackupClient
{
	public DefaultBackupClient(
		String		sBackupPath) throws FileNotFoundException
	{
		File BackupDir = new File( sBackupPath);
		
		BackupDir.mkdirs();
		
		// Note: This rather odd name comes from the desire to maintain
		// compatibility with C++ default backup client
		
		String sPathName = sBackupPath + 
								 System.getProperty( "file.separator") + 
								 "00000000.64";
								 
		m_OStream = new FileOutputStream( sPathName);
	}
	
	/**
	 * Desc:
	 */
	public int WriteData(
		byte[]		Buffer)
	{
		int	iRCode = RCODE.NE_XFLM_OK;
		
		try
		{
				m_OStream.write( Buffer);
		}
		catch (IOException e)
		{
			iRCode = RCODE.NE_FLM_WRITING_FILE;
		}
		
		return( iRCode);
	}

	private FileOutputStream	m_OStream;
}
