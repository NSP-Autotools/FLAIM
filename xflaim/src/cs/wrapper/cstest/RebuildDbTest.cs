//------------------------------------------------------------------------------
// Desc:	Rebuild database test
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
	// Rebuild database test.
	//--------------------------------------------------------------------------
	public class RebuildDbTest : Tester
	{
		private class MyDbRebuildStatus : DbRebuildStatus
		{
			public MyDbRebuildStatus()
			{
				m_bOutputLines = false;
				System.Console.Write( "\n");
			}

			public RCODE reportRebuild(
				XFLM_REBUILD_INFO	rebuildInfo)
			{
				if (rebuildInfo.bStartFlag != 0)
				{
					System.Console.WriteLine( "\nRebuild Phase: {0}", rebuildInfo.eDoingFlag);
				}
				System.Console.Write( "Bytes To Do {0}, Bytes Done: {1}\r",
					rebuildInfo.ulDatabaseSize, rebuildInfo.ulBytesExamined);
				m_bOutputLines = true;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportRebuildErr(
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

		public bool rebuildDbTest(
			string	sSrcDbName,
			string	sDestDbName,
			DbSystem	dbSystem)
		{
			MyDbRebuildStatus	dbRebuildStatus = null;
			XFLM_CREATE_OPTS	createOpts = null;

			// Try restoring the database

			beginTest( "Rebuild Database Test (" + sSrcDbName + " to " + sDestDbName + ")");

			dbRebuildStatus = new MyDbRebuildStatus();
			createOpts = new XFLM_CREATE_OPTS();

			createOpts.uiBlockSize = 8192;
			createOpts.uiVersionNum = (uint)DBVersions.XFLM_CURRENT_VERSION_NUM;
			createOpts.uiMinRflFileSize = 2000000;
			createOpts.uiMaxRflFileSize = 20000000;
			createOpts.bKeepRflFiles = 1;
			createOpts.bLogAbortedTransToRfl = 1;
			createOpts.eDefaultLanguage = Languages.FLM_DE_LANG;
			try
			{
				dbSystem.dbRebuild( sSrcDbName, null, sDestDbName, null, null,
					null, null, createOpts, dbRebuildStatus);
			}
			catch (XFlaimException ex)
			{
				endTest( dbRebuildStatus.outputLines(), ex, "rebuilding database");
				return( false);
			}

			endTest( true, true);
			return( true);
		}
	}
}
