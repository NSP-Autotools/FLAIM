//------------------------------------------------------------------------------
// Desc:	Copy database test
// Tabs:	3
//
//		Copyright (c) 2006-2007 Novell, Inc. All Rights Reserved.
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
	// Copy database test.
	//--------------------------------------------------------------------------
	public class CopyDbTest : Tester
	{
		private class MyDbCopyStatus : DbCopyStatus
		{
			public MyDbCopyStatus()
			{
				m_bOutputLines = false;
			}

			public RCODE dbCopyStatus(
				ulong			ulBytesToCopy,
				ulong			ulBytesCopied,
				string		sSrcFileName,
				string		sDestFileName)
			{
				if (sSrcFileName != null)
				{
					System.Console.WriteLine( "\nSrc File: {0}, Dest File {1}", sSrcFileName, sDestFileName);
				}
				System.Console.Write( "Bytes To Copy: {0}, Bytes Copied: {1}\r", ulBytesToCopy, ulBytesCopied);
				m_bOutputLines = true;
				return( RCODE.NE_XFLM_OK);
			}

			public bool outputLines()
			{
				return( m_bOutputLines);
			}

			private bool	m_bOutputLines;
		}

		public bool copyDbTest(
			string	sSrcDbName,
			string	sDestDbName,
			DbSystem	dbSystem)
		{

			// Try copying the database

			MyDbCopyStatus	copyStatus = new MyDbCopyStatus();

			beginTest( "Copy Database Test (" + sSrcDbName + " --> " + sDestDbName + ")");
			try
			{
				dbSystem.dbCopy( sSrcDbName, null, null, sDestDbName, null, null, copyStatus);
			}
			catch (XFlaimException ex)
			{
				endTest( copyStatus.outputLines(), ex, "copying database");
				return( false);
			}
			endTest( copyStatus.outputLines(), true);
			return( true);
		}
	}
}
