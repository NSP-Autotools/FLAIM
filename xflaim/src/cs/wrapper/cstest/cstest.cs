//------------------------------------------------------------------------------
// Desc:	CSharp Tester
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
	public class Tester
	{
		//--------------------------------------------------------------------------
		// Begin a test.
		//--------------------------------------------------------------------------
		public void beginTest( 
			string	sTestName)
		{
			System.Console.Write( "{0} ... ", sTestName);
		}

		//--------------------------------------------------------------------------
		// End a test.
		//--------------------------------------------------------------------------
		public void endTest(
			bool	bWriteLine,
			bool	bPassed)
		{
			if (bWriteLine)
			{
				System.Console.Write( "\n");
			}
			if (bPassed)
			{
				System.Console.WriteLine( "PASS");
			}
			else
			{
				System.Console.WriteLine( "FAIL");
			}
		}

		//--------------------------------------------------------------------------
		// End a test with an exception
		//--------------------------------------------------------------------------
		public void endTest(
			bool					bWriteLine,
			XFlaimException	ex,
			string				sWhat)
		{
			endTest( bWriteLine, false);
			System.Console.Write( "Error {0}: ", sWhat);
			if (ex.getRCode() == RCODE.NE_XFLM_OK)
			{
				System.Console.WriteLine( "{0}", ex.getString());
			}
			else
			{
				System.Console.WriteLine( "{0}", ex.getRCode());
			}
		}

		//--------------------------------------------------------------------------
		// Print out information on a corruption
		//--------------------------------------------------------------------------
		public static void printCorruption(
			XFLM_CORRUPT_INFO	corruptInfo)
		{
			System.Console.WriteLine( "\nCorruption Found: {0}, Locale: {1}",
				corruptInfo.eErrCode, corruptInfo.eErrLocale);
			if (corruptInfo.uiErrLfNumber != 0)
			{
				System.Console.WriteLine( "  Logical File Number...... {0} ({1})",
					corruptInfo.uiErrLfNumber, corruptInfo.eErrLfType);
				System.Console.WriteLine( "  B-Tree Level............. {0}",
					corruptInfo.uiErrBTreeLevel);
			}
			if (corruptInfo.uiErrBlkAddress != 0)
			{
				System.Console.WriteLine( "  Block Address............ {0:X})",
					corruptInfo.uiErrBlkAddress);
			}
			if (corruptInfo.uiErrParentBlkAddress != 0)
			{
				System.Console.WriteLine( "  Parent Block Address..... {0:X})",
					corruptInfo.uiErrParentBlkAddress);
			}
			if (corruptInfo.uiErrElmOffset != 0)
			{
				System.Console.WriteLine( "  Element Offset........... {0})",
					corruptInfo.uiErrElmOffset);
			}
			if (corruptInfo.ulErrNodeId != 0)
			{
				System.Console.WriteLine( "  Node ID.................. {0})",
					corruptInfo.ulErrNodeId);
			}
		}

		//--------------------------------------------------------------------------
		// Print a cache usage structure.
		//--------------------------------------------------------------------------
		public void printCacheUsage(
			CS_XFLM_CACHE_USAGE	cacheUsage,
			string					sWhat)
		{
			System.Console.WriteLine( "{0}", sWhat);
			System.Console.WriteLine( "  Object Count..................... {0}", cacheUsage.ulCount);
			System.Console.WriteLine( "  Byte Count....................... {0}", cacheUsage.ulByteCount);
			System.Console.WriteLine( "  Old Version Object Count......... {0}", cacheUsage.ulOldVerCount);
			System.Console.WriteLine( "  Old Version Byte Count........... {0}", cacheUsage.ulOldVerBytes);
			System.Console.WriteLine( "  Cache Hits....................... {0}", cacheUsage.uiCacheHits);
			System.Console.WriteLine( "  Cache Hit Looks.................. {0}", cacheUsage.uiCacheHitLooks);
			System.Console.WriteLine( "  Cache Faults..................... {0}", cacheUsage.uiCacheFaults);
			System.Console.WriteLine( "  Cache Fault Looks................ {0}", cacheUsage.uiCacheFaultLooks);
			System.Console.WriteLine( "  Slab Count....................... {0}", cacheUsage.slabUsage.ulSlabs);
			System.Console.WriteLine( "  Slab Bytes Count................. {0}", cacheUsage.slabUsage.ulSlabBytes);
			System.Console.WriteLine( "  Slab Allocated Cells............. {0}", cacheUsage.slabUsage.ulAllocatedCells);
			System.Console.WriteLine( "  Slab Free Cells.................. {0}", cacheUsage.slabUsage.ulFreeCells);
		}
	}

	//--------------------------------------------------------------------------
	// Main for tester program
	//--------------------------------------------------------------------------

	public class RunTests
	{
		private const string CREATE_DB_NAME = "create.db";
		private const string COPY_DB_NAME = "copy.db";
		private const string COPY2_DB_NAME = "copy2.db";
		private const string RENAME_DB_NAME = "rename.db";
		private const string RESTORE_DB_NAME = "restore.db";
		private const string BACKUP_PATH = "backup";
		private const string REBUILD_DB_NAME = "rebuild.db";
		private const string TEST_DB_NAME = "test.db";

		static void Main()
		{

			DbSystem dbSystem = new DbSystem();

			// Database create test

			CreateDbTest createDb = new CreateDbTest();
			if (!createDb.createDbTest( CREATE_DB_NAME, dbSystem))
			{
				return;
			}

			// Database open test

			OpenDbTest openDb = new OpenDbTest();
			if (!openDb.openDbTest( CREATE_DB_NAME, dbSystem))
			{
				return;
			}

			// DOM Nodes test

			DOMNodesTest domNodes = new DOMNodesTest();
			if (!domNodes.domNodesTest( TEST_DB_NAME, dbSystem))
			{
				return;
			}

			// Import tests

			ImportTests importTest = new ImportTests();
			if (!importTest.importTests( TEST_DB_NAME, dbSystem))
			{
				return;
			}

			// Statistics test

#if !mono

// CANT GET THIS TEST TO WORK ON MONO FOR NOW, SO WE COMPILE IT OUT
			StatsTests statsTests = new StatsTests();
			if (!statsTests.statsTests( CREATE_DB_NAME, dbSystem))
			{
				return;
			}
#endif

			// Database copy test

			CopyDbTest copyDb = new CopyDbTest();
			if (!copyDb.copyDbTest( CREATE_DB_NAME, COPY_DB_NAME, dbSystem))
			{
				return;
			}
			if (!copyDb.copyDbTest( TEST_DB_NAME, COPY2_DB_NAME, dbSystem))
			{
				return;
			}

			// Database rename test

			RenameDbTest renameDb = new RenameDbTest();
			if (!renameDb.renameDbTest( COPY2_DB_NAME, RENAME_DB_NAME, dbSystem))
			{
				return;
			}

			// Database backup test

			BackupDbTest backupDb = new BackupDbTest();
			if (!backupDb.backupDbTest( RENAME_DB_NAME, BACKUP_PATH, dbSystem))
			{
				return;
			}

			// Database restore test

			RestoreDbTest restoreDb = new RestoreDbTest();
			if (!restoreDb.restoreDbTest( RESTORE_DB_NAME, BACKUP_PATH, dbSystem))
			{
				return;
			}

			// Database rebuild test

			RebuildDbTest rebuildDb = new RebuildDbTest();
			if (!rebuildDb.rebuildDbTest( RESTORE_DB_NAME, REBUILD_DB_NAME, dbSystem))
			{
				return;
			}

			// Database check test

			CheckDbTest checkDb = new CheckDbTest();
			if (!checkDb.checkDbTest( CREATE_DB_NAME, dbSystem))
			{
				return;
			}
			if (!checkDb.checkDbTest( COPY_DB_NAME, dbSystem))
			{
				return;
			}
			if (!checkDb.checkDbTest( RESTORE_DB_NAME, dbSystem))
			{
				return;
			}
			if (!checkDb.checkDbTest( RENAME_DB_NAME, dbSystem))
			{
				return;
			}
			if (!checkDb.checkDbTest( REBUILD_DB_NAME, dbSystem))
			{
				return;
			}

			// Database remove test

			RemoveDbTest removeDb = new RemoveDbTest();
			if (!removeDb.removeDbTest( CREATE_DB_NAME, dbSystem))
			{
				return;
			}
			if (!removeDb.removeDbTest( COPY_DB_NAME, dbSystem))
			{
				return;
			}
			if (!removeDb.removeDbTest( RESTORE_DB_NAME, dbSystem))
			{
				return;
			}
			if (!removeDb.removeDbTest( RENAME_DB_NAME, dbSystem))
			{
				return;
			}
			if (!removeDb.removeDbTest( REBUILD_DB_NAME, dbSystem))
			{
				return;
			}

			// Input and Output stream tests

			StreamTests streamTests = new StreamTests();
			if (!streamTests.streamTests( dbSystem))
			{
				return;
			}

			// Data vector tests

			VectorTests vectorTests = new VectorTests();
			if (!vectorTests.vectorTests( dbSystem))
			{
				return;
			}

			// Cache tests

			CacheTests cacheTests = new CacheTests();
			if (!cacheTests.cacheTests( dbSystem))
			{
				return;
			}

			// Various settings tests

			SettingsTests settingsTests = new SettingsTests();
			if (!settingsTests.settingsTests( dbSystem))
			{
				return;
			}

			// Various string comparison tests

			CompareStringTests compareStringTests = new CompareStringTests();
			if (!compareStringTests.compareStringTests( dbSystem))
			{
				return;
			}
		}
	}
}
