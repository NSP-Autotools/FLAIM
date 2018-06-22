//------------------------------------------------------------------------------
// Desc:	Check database test
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
	// Check database test.
	//--------------------------------------------------------------------------
	public class CheckDbTest : Tester
	{
		private class MyDbCheckStatus : DbCheckStatus
		{
			public MyDbCheckStatus()
			{
				m_bOutputLines = false;
				System.Console.Write( "\n");
			}

			public RCODE reportProgress(
				XFLM_PROGRESS_CHECK_INFO	progressInfo)
			{
				if (progressInfo.bStartFlag != 0)
				{
					if (progressInfo.eCheckPhase == FlmCheckPhase.XFLM_CHECK_B_TREE)
					{
						System.Console.WriteLine( "\nChecking B-Tree: {0} ({1})",
							progressInfo.uiLfNumber, progressInfo.eLfType);
					}
					else
					{
						System.Console.WriteLine( "\nCheck Phase: {0}", progressInfo.eCheckPhase);
					}
				}
				System.Console.Write( "Bytes To Check: {0}, Bytes Checked: {1}\r",
					progressInfo.ulDatabaseSize, progressInfo.ulBytesExamined);
				m_bOutputLines = true;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportCheckErr(
				XFLM_CORRUPT_INFO	corruptInfo)
			{
				printCorruption( corruptInfo);
				m_bOutputLines = true;
				return( RCODE.NE_XFLM_OK);
			}

			public bool outputLines()
			{
				return( m_bOutputLines);
			}

			private bool	m_bOutputLines;
		}

		public bool checkDbTest(
			string	sDbName,
			DbSystem	dbSystem)
		{
			MyDbCheckStatus	dbCheckStatus = null;
			DbInfo				dbInfo = null;
			XFLM_DB_HDR			dbHdr = new XFLM_DB_HDR();

			// Try restoring the database

			beginTest( "Check Database Test (" + sDbName + ")");

			dbCheckStatus = new MyDbCheckStatus();
			try
			{
				dbInfo = dbSystem.dbCheck( sDbName, null, null, null,
					DbCheckFlags.XFLM_ONLINE | DbCheckFlags.XFLM_DO_LOGICAL_CHECK,
					dbCheckStatus);
			}
			catch (XFlaimException ex)
			{
				endTest( dbCheckStatus.outputLines(), ex, "checking database");
				return( false);
			}

			dbInfo.getDbHdr( dbHdr);
			System.Console.Write( "\n");
			System.Console.WriteLine( "Signature............. {0}", dbHdr.szSignature);
			System.Console.WriteLine( "Database Version...... {0}", dbHdr.ui32DbVersion);
			System.Console.WriteLine( "Block Size............ {0}", dbHdr.ui16BlockSize);

			if (dbHdr.szSignature != "FLAIMDB")
			{
				endTest( true, false);
				System.Console.WriteLine( "Invalid signature in database header");
				return( false);
			}
			if (dbHdr.ui16BlockSize != 8192 && dbHdr.ui16BlockSize != 4096)
			{
				endTest( true, false);
				System.Console.WriteLine( "Invalid block size in database header");
				return( false);
			}
			if ((DBVersions)dbHdr.ui32DbVersion != DBVersions.XFLM_CURRENT_VERSION_NUM)
			{
				endTest( true, false);
				System.Console.WriteLine( "Invalid version in database header");
				return( false);
			}
			endTest( true, true);
			return( true);
		}
	}
}
