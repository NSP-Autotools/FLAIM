//------------------------------------------------------------------------------
// Desc:	Backup database test
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
using System.IO;
using System.Runtime.InteropServices;
using xflaim;

namespace cstest
{

	//--------------------------------------------------------------------------
	// Backup database test.
	//--------------------------------------------------------------------------
	public class BackupDbTest : Tester
	{
		private class MyBackupStatus : BackupStatus
		{
			public MyBackupStatus()
			{
				System.Console.WriteLine( " ");
				m_bOutputLines = false;
			}

			public RCODE backupStatus(
				ulong			ulBytesToDo,
				ulong			ulBytesDone)
			{
				System.Console.Write( "Bytes To Backup: {0}, Bytes Backed Up: {1}\r", ulBytesToDo, ulBytesDone);
				m_bOutputLines = true;
				return( RCODE.NE_XFLM_OK);
			}

			public bool outputLines()
			{
				return( m_bOutputLines);
			}

			private bool	m_bOutputLines;
		}

		public bool backupDbTest(
			string	sDbName,
			string	sBackupPath,
			DbSystem	dbSystem)
		{
			Db					db = null;
			Backup			backup = null;
			MyBackupStatus	backupStatus = null;

			// Try backing up the database

			beginTest( "Backup Database Test (" + sDbName + " to directory \"" + sBackupPath + "\")");

			try
			{
				db = dbSystem.dbOpen( sDbName, null, null, null, false);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "opening database");
				return( false);
			}

			// Backup the database

			try
			{
				backup = db.backupBegin( true, false, 0);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling backupBegin");
				return( false);
			}

			// Perform the backup

			backupStatus = new MyBackupStatus();
			try
			{
				backup.backup( sBackupPath, null, null, backupStatus);
			}
			catch (XFlaimException ex)
			{
				endTest( backupStatus.outputLines(), ex, "calling backup");
				return( false);
			}

			// End the backup

			try
			{
				backup.endBackup();
			}
			catch (XFlaimException ex)
			{
				endTest( backupStatus.outputLines(), ex, "calling endBackup");
				return( false);
			}

			db.close();
			db = null;
			endTest( backupStatus.outputLines(), true);
			return( true);
		}
	}
}
