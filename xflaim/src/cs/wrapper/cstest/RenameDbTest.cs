//------------------------------------------------------------------------------
// Desc:	Rename database test
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
	// Rename database test.
	//--------------------------------------------------------------------------
	public class RenameDbTest : Tester
	{
		private class MyDbRenameStatus : DbRenameStatus
		{
			public MyDbRenameStatus()
			{
				m_bOutputLines = false;
				System.Console.Write( "\n");
			}

			public RCODE dbRenameStatus(
				string		sSrcFileName,
				string		sDestFileName)
			{
				System.Console.WriteLine( "Renaming {0} to {1}", sSrcFileName, sDestFileName);
				m_bOutputLines = true;
				return( RCODE.NE_XFLM_OK);
			}

			public bool outputLines()
			{
				return( m_bOutputLines);
			}

			private bool	m_bOutputLines;
		}

		public bool renameDbTest(
			string	sSrcDbName,
			string	sDestDbName,
			DbSystem	dbSystem)
		{

			// Try renaming the database

			MyDbRenameStatus	renameStatus = new MyDbRenameStatus();

			beginTest( "Rename Database Test (" + sSrcDbName + " --> " + sDestDbName + ")");
			try
			{
				dbSystem.dbRename( sSrcDbName, null, null, sDestDbName, true, renameStatus);
			}
			catch (XFlaimException ex)
			{
				endTest( renameStatus.outputLines(), ex, "renaming database");
				return( false);
			}
			endTest( renameStatus.outputLines(), true);
			return( true);
		}
	}
}
