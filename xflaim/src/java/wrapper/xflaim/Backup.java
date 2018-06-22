//------------------------------------------------------------------------------
// Desc:	Backup
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
import java.io.FileNotFoundException;

/**
 * This classes provides methods to back up an XFlaim database
 */
public class Backup
{
	// This constructor doesn't need to do much of anything; it's here mostly
	// to ensure that Backup does NOT have a public constructor.  (The
	// application is not supposed to call new on Backup; Backup objects
	// are created by a call to Db.backupBegin
	
	Backup(
		long		lThis,
		Db			jdb)
	{
		m_this = lThis;
		m_jdb = jdb;
	}

	/**
	 * Finalize method used to release native resources on garbage collection.
	 */	
	public void finalize()
	{
		if (m_this != 0)
		{
			_release( m_this);
			m_this = 0;
		}
		
		m_jdb = null;
	}
	
	/**
	 * Get the transaction ID for this backup operation. 
	 * @return Returns the transaction ID for this backup operation.
	 */
	public long getBackupTransId()
	{
		return _getBackupTransId( m_this);
	}
	
	/**
	 * Gets the transaction ID for the last backup job run on this database.
	 * @return Returns the transaction ID for the last backup job run on the
	 * database associated with this Backup object.
	 */
	public long getLastBackupTransId()
	{
		return _getLastBackupTransId( m_this);
	}
	
	/**
	 * Performs the backup operation.  <code>sBackupPath</code> and <code>
	 * backupClient</code> are mutually exclusive.  If <code>backupClient</code> is null,
	 * then an instance of <code>DefaultBackupClient</code> will be created
	 * and <code>sBackupPath</code> passed into its constructor.  If <code>
	 * Client</code> is non-null, <code>sBackupPath</code> is ignored.
	 * @param sBackupPath Optional.  The full pathname of a file to store the
	 * backed up data.
	 * @param backupClient Optional.  If non-null, then it will be used as the backup
	 * client.
	 * @param backupStatus Optional.  If non-null, then <code>backupStatus.backupStatus
	 * </code> will be called periodically to inform the application about the
	 * progress of the backup operation.
	 * @return Returns the sequence number of this backup.  (This is for
	 * informational purposes only; for instance, users can use it to label
	 * their backup tapes.)
	 * @throws XFlaimException
	 */
	public long backup(
		String				sBackupPath,
		String				sPassword,
		BackupClient		backupClient,
		BackupStatus		backupStatus) throws XFlaimException
	{
		BackupClient	backClient;
		
		if (backupClient == null)
		{
			try
			{
				backClient = new DefaultBackupClient( sBackupPath);
			}
			catch (FileNotFoundException e)
			{
				throw new XFlaimException( xflaim.RCODE.NE_FLM_OPENING_FILE,
					"IOException opening " + sBackupPath + ". Message from JVM was" +
					e.getMessage());
			}
		}
		else
		{
			backClient = backupClient;
		}
		
		return _backup( m_this, sBackupPath, sPassword, backClient, backupStatus);
	}

	/**
	 * Ends the backup operation.
	 * @throws XFlaimException
	 */
	public void endBackup() throws XFlaimException
	{
		_endBackup( m_this);
	}
	
	// Reassigns the object to "point" to a new F_Backup instance and a new
	// Db.  Called by any of the member functions that take a
	// Db.backupBegin parameter.  Shouldn't be called by outsiders, so it's
	// not public, but it must be callable for other instances of this class.
	// NOTE:  This function does not result in a call to F_Backup::Release()
	// because that is done by the native code when the F_Backup object is 
	// reused.  Calling setRef() in any case except from within
	// Db.backupBegin will result in a memory leak on the native side!
	void setRef(
		long		lBackupRef,
		Db			jdb)
	{
		m_this = lBackupRef;
		m_jdb = jdb;
	}

	long getRef()
	{
		return m_this;
	}
	
// PRIVATE METHODS
	
	private native void _release(
		long				lThis);
	
	private native long _getBackupTransId(
		long				lThis);
	
	private native long _getLastBackupTransId(
		long				lThis);
		
	private native long _backup(
		long				lThis,
		String			sBackupPath,
		String			sPassword,
		BackupClient	Client,
		BackupStatus	Status) throws XFlaimException;
	
	private native void _endBackup(
		long				lThis) throws XFlaimException;
	
	private long			m_this;
	private Db				m_jdb;
}

