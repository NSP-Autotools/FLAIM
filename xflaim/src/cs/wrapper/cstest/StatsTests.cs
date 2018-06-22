//------------------------------------------------------------------------------
// Desc:	Statistics tests
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
	// Statistics tests.
	//--------------------------------------------------------------------------
	public class StatsTests : Tester
	{
		private const uint	LABEL_LEN = 35;

		private string makeIndentStr(
			uint	uiIndent)
		{
			string	sIndent = null;

			for (uint uiLoop = 0; uiLoop < uiIndent; uiLoop++)
			{
				sIndent += "  ";
			}
			return( sIndent);
		}

		private string makeLabel(
			uint		uiIndent,
			string	sLabel)
		{
			string	sNewLabel = makeIndentStr( uiIndent) + sLabel;

			while (sNewLabel.Length < LABEL_LEN)
			{
				sNewLabel += ".";
			}
			sNewLabel += " ";
			return( sNewLabel);
		}

		private void printStrStat(
			uint		uiIndent,
			string	sLabel,
			string	sStat)
		{
			System.Console.WriteLine( "{0}{1}", makeLabel( uiIndent, sLabel), sStat);
		}

		private void printUIntStat(
			uint		uiIndent,
			string	sLabel,
			uint		uiStat)
		{
			System.Console.WriteLine( "{0}{1}", makeLabel( uiIndent, sLabel), uiStat);
		}

		private void printULongStat(
			uint		uiIndent,
			string	sLabel,
			ulong		ulStat)
		{
			System.Console.WriteLine( "{0}{1}", makeLabel( uiIndent, sLabel), ulStat);
		}

		private void printCountTimeStat(
			uint					uiIndent,
			string				sLabel,
			F_COUNT_TIME_STAT	stat)
		{
			ulong	ulAvgMilli = (stat.ulCount != 0)
				? stat.ulElapMilli / stat.ulCount
				: 0;
			System.Console.WriteLine( "{0}Count={1},ElapMilli={2},AvgMilli={3}",
				makeLabel( uiIndent, sLabel),
				stat.ulCount, stat.ulElapMilli, ulAvgMilli);
		}

		private void printDiskIOStats(
			uint					uiIndent,
			string				sLabel,
			XFLM_DISKIO_STAT	diskIOStat)
		{
			ulong	ulAvgMilli = (diskIOStat.ulCount != 0)
										? diskIOStat.ulElapMilli / diskIOStat.ulCount
										: 0;
			System.Console.WriteLine( "{0}Count={1},Bytes={2},ElapMilli={3},AvgMilli={4}",
				makeLabel( uiIndent, sLabel),
				diskIOStat.ulCount, diskIOStat.ulTotalBytes,
				diskIOStat.ulElapMilli, ulAvgMilli);
		}

		private void printBlockIOStats(
			uint						uiIndent,
			string					sLabel,
			XFLM_BLOCKIO_STATS	blockIOStats)
		{
			System.Console.WriteLine( "{0}{1}", makeIndentStr( uiIndent), sLabel);
			printDiskIOStats( uiIndent + 1, "Block Reads", blockIOStats.BlockReads);
			printDiskIOStats( uiIndent + 1, "Block Writes", blockIOStats.BlockWrites);
			printDiskIOStats( uiIndent + 1, "Old View Block Reads", blockIOStats.OldViewBlockReads);
			printUIntStat( uiIndent + 1, "Block Checksum Errors", blockIOStats.uiBlockChkErrs);
			printUIntStat( uiIndent + 1, "Old Block Checksum Errors", blockIOStats.uiOldViewBlockChkErrs);
			printUIntStat( uiIndent + 1, "Old View Errors", blockIOStats.uiOldViewErrors);
		}

		public bool statsTests(
			string	sDbName,
			DbSystem	dbSystem)
		{
			Db							db = null;
			DbSystemStats			stats = null;
			uint						uiNumDatabases;
			uint						uiStartTime;
			uint						uiStopTime;
			CS_XFLM_DB_STATS		dbStats = null;
			CS_XFLM_LFILE_STATS	lFileStats = null;

			beginTest( "Start statistics");

			try
			{
				dbSystem.startStats();
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "starting statistics");
				return( false);
			}
			endTest( false, true);

			// Open a database to make some statistics happen

			beginTest( "Open Database Test (" + sDbName + ")");

			try
			{
				db = dbSystem.dbOpen( sDbName, null, null, null, false);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "opening database");
				return( false);
			}
			if (db != null)
			{
				db.close();
				db = null;
			}
			endTest( false, true);

			// Stop collecting statistics

			beginTest( "Stop statistics");

			try
			{
				dbSystem.stopStats();
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "stopping statistics");
				return( false);
			}
			endTest( false, true);

			// Get statistics

			beginTest( "Get statistics");

			try
			{
				stats = dbSystem.getStats();
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "getting statistics");
				return( false);
			}
			endTest( false, true);

			// Get general statistics

			beginTest( "Get general statistics");

			try
			{
				stats.getGeneralStats( out uiNumDatabases, out uiStartTime,
							out uiStopTime);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "getting statistics");
				return( false);
			}
			endTest( false, true);
			printUIntStat( 0, "Databases", uiNumDatabases);
			printUIntStat( 0, "Start Time", uiStartTime);
			printUIntStat( 0, "Stop Time", uiStopTime);

			// Get Database statistics

			for (uint uiLoop = 0; uiLoop < uiNumDatabases; uiLoop++)
			{
				beginTest( "Get database statistics for DB#" + uiLoop);

				try
				{
					dbStats = stats.getDbStats( uiLoop, dbStats);
				}
				catch (XFlaimException ex)
				{
					endTest( false, ex, "getting database statistics");
					return( false);
				}
				endTest( false, true);
				printStrStat( 0, "Database Name", dbStats.sDbName);
				printUIntStat( 0, "Logical File Count", dbStats.uiNumLFiles);
				System.Console.WriteLine( "Read Transactions");
				printCountTimeStat( 1, "Committed Transactions", dbStats.ReadTransStats.CommittedTrans);
				printCountTimeStat( 1, "Aborted Transactions", dbStats.ReadTransStats.AbortedTrans);
				System.Console.WriteLine( "Update Transactions");
				printCountTimeStat( 1, "Committed Transactions", dbStats.UpdateTransStats.CommittedTrans);
				printCountTimeStat( 1, "Aborted Transactions", dbStats.UpdateTransStats.AbortedTrans);
				printCountTimeStat( 1, "Group Completes", dbStats.UpdateTransStats.GroupCompletes);
				printULongStat( 1, "Group Finished", dbStats.UpdateTransStats.ulGroupFinished);
				printBlockIOStats( 0, "LFH Block Stats", dbStats.LFHBlockStats);
				printBlockIOStats( 0, "Avail Block Stats", dbStats.AvailBlockStats);
				printDiskIOStats( 0, "Database Header Writes", dbStats.DbHdrWrites);
				printDiskIOStats( 0, "Log Block Writes", dbStats.LogBlockWrites);
				printDiskIOStats( 0, "Log Block Restores", dbStats.LogBlockRestores);
				printDiskIOStats( 0, "Log Block Reads", dbStats.LogBlockReads);
				printUIntStat( 0, "Log Block Checksum Errors", dbStats.uiLogBlockChkErrs);
				printUIntStat( 0, "Read Errors", dbStats.uiReadErrors);
				printCountTimeStat( 0, "No Locks", dbStats.LockStats.NoLocks);
				printCountTimeStat( 0, "Waiting For Lock", dbStats.LockStats.WaitingForLock);
				printCountTimeStat( 0, "Held Lock", dbStats.LockStats.HeldLock);

				for (uint uiLoop2 = 0; uiLoop2 < dbStats.uiNumLFiles; uiLoop2++)
				{
					beginTest( "  Get database statistics for DB#" + uiLoop + ", LFile#" + uiLoop2);

					try
					{
						lFileStats = stats.getLFileStats( uiLoop, uiLoop2, lFileStats);
					}
					catch (XFlaimException ex)
					{
						endTest( false, ex, "getting logical file statistics");
						return( false);
					}
					endTest( false, true);
					System.Console.WriteLine( "  LOGICAL FILE {0} ({1})",
						lFileStats.uiLFileNum, lFileStats.eLfType);
					printBlockIOStats( 2, "Root Block Stats", lFileStats.RootBlockStats);
					printBlockIOStats( 2, "Middle Block Stats", lFileStats.MiddleBlockStats);
					printBlockIOStats( 2, "Leaf Block Stats", lFileStats.LeafBlockStats);
					printULongStat( 2, "Block Splits", lFileStats.ulBlockSplits);
					printULongStat( 2, "Block Combines", lFileStats.ulBlockCombines);
				}
			}

			return( true);
		}
	}
}
