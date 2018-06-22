//------------------------------------------------------------------------------
// Desc:	Backup
// Tabs:	3
//
// Copyright (c) 2006-2007 Novell, Inc. All Rights Reserved.
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

using System;
using System.Runtime.InteropServices;

namespace xflaim
{

	/// <summary>
	/// This class provides methods to backup an XFLAIM database.
	/// </summary>
	public class Backup
	{
		private IntPtr	m_pBackup;	// Pointer to IF_Backup object in unmanaged space
		private Db		m_db;

		/// <summary>
		/// This constructor doesn't need to do much of anything; it's here mostly
		/// to ensure that Backup does NOT have a public constructor.  (The
		/// application is not supposed to call new on Backup; Backup objects
		/// are created by calling <see cref="Db.backupBegin"/>
		/// </summary>
		/// <param name="pBackup">
		/// This is a pointer to the IF_Backup object in C++.
		/// </param>
		/// <param name="db">
		/// This is the database object this backup object is associated with.
		/// We keep a reference to the database object so that it won't go away
		/// in the middle of a backup.
		/// </param>
		public Backup(
			IntPtr	pBackup,
			Db			db)
		{
			if ((m_pBackup = pBackup) == IntPtr.Zero)
			{
				throw new XFlaimException( "Invalid IF_Backup pointer passed into Backup constructor");
			}

			if ((m_db = db) == null)
			{
				throw new XFlaimException( "Invalid Db object passed into Backup constructor");
			}

			// Must call something inside of Db.  Otherwise, the
			// m_db object gets a compiler warning on linux because
			// it is not used anywhere.  Other than that, there is really
			// no need to make the following call.
			if (m_db.getDb() == IntPtr.Zero)
			{
				throw new XFlaimException( "Invalid Db.IF_Db object");
			}
		}

		/// <summary>
		/// Destructor
		/// </summary>
		~Backup()
		{
			if (m_pBackup != IntPtr.Zero)
			{
				xflaim_Backup_Release( m_pBackup);
				m_pBackup = IntPtr.Zero;
			}
			
			m_db = null;
		}
	
		[DllImport("xflaim")]
		private static extern void xflaim_Backup_Release(
			IntPtr	pBackup);

//-----------------------------------------------------------------------------
// getBackupTransId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the transaction ID for this backup operation.
		/// </summary>
		/// <returns>Returns the transaction ID for this backup operation.</returns>
		public ulong getBackupTransId()
		{
			return( xflaim_Backup_getBackupTransId( m_pBackup));
		}
	
		[DllImport("xflaim")]
		private static extern ulong xflaim_Backup_getBackupTransId(
			IntPtr	pBackup);

//-----------------------------------------------------------------------------
// getLastBackupTransId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Gets the transaction ID for the last backup job run on this database.
		/// </summary>
		/// <returns>
		/// Returns the transaction ID for the last backup job run on the
		/// database associated with this Backup object.
		/// </returns>
		public ulong getLastBackupTransId()
		{
			return( xflaim_Backup_getLastBackupTransId( m_pBackup));
		}
	
		[DllImport("xflaim")]
		private static extern ulong xflaim_Backup_getLastBackupTransId(
			IntPtr	pBackup);

//-----------------------------------------------------------------------------
// backup
//-----------------------------------------------------------------------------

		/// <summary>
		/// Performs the backup operation. The <paramref name="sBackupPath"/> and
		/// <paramref name="backupClient"/> parameters are mutually exclusive.  If
		/// backupClient is null, then the backup will be created on disk in the
		/// location specified by sBackupPath.  If backupClient is non-null, the
		/// sBackupPath parameter is ignored.
		/// </summary>
		/// <param name="sBackupPath">
		/// The full pathname where the backup set is to be created.  This parameter
		/// is ignored if the backupClient parameter is non-null.
		/// </param>
		/// <param name="sPassword">
		/// Password to be used for the backup.  A non-empty password allows the backup
		/// to be restored on a machine other than the one where the database exists.  The
		/// database's encryption key (if encryption is enabled) will be wrapped in the
		/// specified password so that the backup can be restored to a different machine.
		/// </param>
		/// <param name="backupClient">
		/// If non-null, the backupClient is an object the provides interfaces for storing
		/// backup data to disk, tape, or other media.  If null, the backup data is stored
		/// to a file set specified by the sBackupPath parameter.
		/// </param>
		/// <param name="backupStatus">
		/// If non-null, the backupStatus object provides an interface that this method
		/// calls to report backup progress.
		/// </param>
		/// <returns>
		/// Returns a sequence number for this backup.  This is for informational
		/// purposes only.  For instance, users can use it to label their backup tapes.
		/// </returns>
		public uint backup(
			string				sBackupPath,
			string				sPassword,
			BackupClient		backupClient,
			BackupStatus		backupStatus)
		{
			RCODE						rc;
			uint						uiSeqNum;
			BackupClientDelegate	backupClientDelegate = null;
			BackupClientCallback	fnBackupClient = null;
			BackupStatusDelegate	backupStatusDelegate = null;
			BackupStatusCallback	fnBackupStatus = null;
			
			if (backupClient != null)
			{
				backupClientDelegate = new BackupClientDelegate( backupClient);
				fnBackupClient = new BackupClientCallback( backupClientDelegate.funcBackupClient);
			}
			if (backupStatus != null)
			{
				backupStatusDelegate = new BackupStatusDelegate( backupStatus);
				fnBackupStatus = new BackupStatusCallback( backupStatusDelegate.funcBackupStatus);
			}
		
			if ((rc = xflaim_Backup_backup( m_pBackup, sBackupPath, sPassword, out uiSeqNum,
				fnBackupClient, fnBackupStatus)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( uiSeqNum);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Backup_backup(
			IntPtr					pBackup,
			[MarshalAs(UnmanagedType.LPStr), In]
			string					sBackupPath,
			[MarshalAs(UnmanagedType.LPStr), In]
			string					sPassword,
			out uint					uiSeqNum,
			BackupClientCallback	fnBackupClient,
			BackupStatusCallback	fnBackupStatus);

		private delegate RCODE BackupClientCallback(
			IntPtr	pvData,
			uint		uiDataLen);
			
		private class BackupClientDelegate
		{
			public BackupClientDelegate(
				BackupClient	backupClient)
			{
				m_backupClient = backupClient; 
			}
			
			~BackupClientDelegate()
			{
			}
			
			public RCODE funcBackupClient(
				IntPtr	pvData,
				uint		uiDataLen)
			{
				return( m_backupClient.writeData( pvData, uiDataLen));
			}
			
			private BackupClient	m_backupClient;
		}

		private delegate RCODE BackupStatusCallback(
			ulong	ulBytesToDo,
			ulong	ulBytesDone);

		private class BackupStatusDelegate
		{
			public BackupStatusDelegate(
				BackupStatus	backupStatus)
			{
				m_backupStatus = backupStatus; 
			}
			
			~BackupStatusDelegate()
			{
			}
			
			public RCODE funcBackupStatus(
				ulong	ulBytesToDo,
				ulong	ulBytesDone)
			{
				return( m_backupStatus.backupStatus( ulBytesToDo, ulBytesDone));
			}
			
			private BackupStatus	m_backupStatus;
		}

//-----------------------------------------------------------------------------
// endBackup
//-----------------------------------------------------------------------------

		/// <summary>
		/// Ends the backup operation.
		/// </summary>
		public void endBackup()
		{
			RCODE	rc;

			if ((rc = xflaim_Backup_endBackup( m_pBackup)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Backup_endBackup(
			IntPtr	pBackup);

	}
}
