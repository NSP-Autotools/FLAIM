//------------------------------------------------------------------------------
// Desc:	Settings tests
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
	// Settings tests.
	//--------------------------------------------------------------------------
	public class SettingsTests : Tester
	{

		private bool setTempDirTest(
			DbSystem	dbSystem)
		{
			string	sSetDir = "abc/def/efg";
			string	sGetDir;

			System.IO.Directory.CreateDirectory( sSetDir);

			beginTest( "Set Temporary Directory");

			try
			{
				dbSystem.setTempDir( sSetDir);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling setTempDir");
				return( false);
			}
			try
			{
				sGetDir = dbSystem.getTempDir();
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling getTempDir");
				return( false);
			}
			if (sSetDir != sGetDir)
			{
				endTest( false, false);
				System.Console.WriteLine( "GetDir != SetDir");
				System.Console.WriteLine( "GetDir = [{0}], setDir = [{1}]", sGetDir, sSetDir);
			}
			endTest( false, true);

			return( true);
		}

		private bool setCheckpointIntervalTest(
			DbSystem	dbSystem)
		{
			uint	uiSetValue = 130;
			uint	uiGetValue;

			beginTest( "Set Checkpoint Interval");

			try
			{
				dbSystem.setCheckpointInterval( uiSetValue);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling setCheckpointInterval");
				return( false);
			}
			try
			{
				uiGetValue = dbSystem.getCheckpointInterval();
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling getCheckpointInterval");
				return( false);
			}
			if (uiSetValue != uiGetValue)
			{
				endTest( false, false);
				System.Console.WriteLine( "GetValue [{0}] != SetValue [{1}]",
					uiGetValue, uiSetValue);
			}
			endTest( false, true);

			return( true);
		}

		private bool setCacheAdjustIntervalTest(
			DbSystem	dbSystem)
		{
			uint	uiSetValue = 37;
			uint	uiGetValue;

			beginTest( "Set Cache Adjust Interval");

			try
			{
				dbSystem.setCacheAdjustInterval( uiSetValue);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling setCacheAdjustInterval");
				return( false);
			}
			try
			{
				uiGetValue = dbSystem.getCacheAdjustInterval();
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling getCacheAdjustInterval");
				return( false);
			}
			if (uiSetValue != uiGetValue)
			{
				endTest( false, false);
				System.Console.WriteLine( "GetValue [{0}] != SetValue [{1}]",
					uiGetValue, uiSetValue);
			}
			endTest( false, true);

			return( true);
		}

		private bool setCacheCleanupIntervalTest(
			DbSystem	dbSystem)
		{
			uint	uiSetValue = 33;
			uint	uiGetValue;

			beginTest( "Set Cache Cleanup Interval");

			try
			{
				dbSystem.setCacheCleanupInterval( uiSetValue);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling setCacheCleanupInterval");
				return( false);
			}
			try
			{
				uiGetValue = dbSystem.getCacheCleanupInterval();
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling getCacheCleanupInterval");
				return( false);
			}
			if (uiSetValue != uiGetValue)
			{
				endTest( false, false);
				System.Console.WriteLine( "GetValue [{0}] != SetValue [{1}]",
					uiGetValue, uiSetValue);
			}
			endTest( false, true);

			return( true);
		}

		private bool setUnusedCleanupIntervalTest(
			DbSystem	dbSystem)
		{
			uint	uiSetValue = 31;
			uint	uiGetValue;

			beginTest( "Set Unused Cleanup Interval");

			try
			{
				dbSystem.setUnusedCleanupInterval( uiSetValue);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling setUnusedCleanupInterval");
				return( false);
			}
			try
			{
				uiGetValue = dbSystem.getUnusedCleanupInterval();
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling getUnusedCleanupInterval");
				return( false);
			}
			if (uiSetValue != uiGetValue)
			{
				endTest( false, false);
				System.Console.WriteLine( "GetValue [{0}] != SetValue [{1}]",
					uiGetValue, uiSetValue);
			}
			endTest( false, true);

			return( true);
		}

		private bool setMaxUnusedTimeTest(
			DbSystem	dbSystem)
		{
			uint	uiSetValue = 117;
			uint	uiGetValue;

			beginTest( "Set Max Unused Time");

			try
			{
				dbSystem.setMaxUnusedTime( uiSetValue);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling setMaxUnusedTime");
				return( false);
			}
			try
			{
				uiGetValue = dbSystem.getMaxUnusedTime();
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling getMaxUnusedTime");
				return( false);
			}
			if (uiSetValue != uiGetValue)
			{
				endTest( false, false);
				System.Console.WriteLine( "GetValue [{0}] != SetValue [{1}]",
					uiGetValue, uiSetValue);
			}
			endTest( false, true);

			return( true);
		}

		private bool setQuerySaveMaxTest(
			DbSystem	dbSystem)
		{
			uint	uiSetValue = 117;
			uint	uiGetValue;

			beginTest( "Set Query Save Max");

			try
			{
				dbSystem.setQuerySaveMax( uiSetValue);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling setQuerySaveMax");
				return( false);
			}
			try
			{
				uiGetValue = dbSystem.getQuerySaveMax();
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling getQuerySaveMax");
				return( false);
			}
			if (uiSetValue != uiGetValue)
			{
				endTest( false, false);
				System.Console.WriteLine( "GetValue [{0}] != SetValue [{1}]",
					uiGetValue, uiSetValue);
			}
			endTest( false, true);

			return( true);
		}

		private bool setDirtyCacheLimitsTest(
			DbSystem	dbSystem)
		{
			ulong	ulSetMaxDirty = 117000000;
			ulong	ulGetMaxDirty;
			ulong	ulSetLowDirty = 37000000;
			ulong	ulGetLowDirty;

			beginTest( "Set Dirty Cache Limits");

			try
			{
				dbSystem.setDirtyCacheLimits( ulSetMaxDirty, ulSetLowDirty);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling setDirtyCacheLimits");
				return( false);
			}
			try
			{
				dbSystem.getDirtyCacheLimits( out ulGetMaxDirty, out ulGetLowDirty);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling getDirtyCacheLimits");
				return( false);
			}
			if (ulSetMaxDirty != ulGetMaxDirty || ulSetLowDirty != ulGetLowDirty)
			{
				endTest( false, false);
				if (ulSetMaxDirty != ulGetMaxDirty)
				{
					System.Console.WriteLine( "GetMaxDirty [{0}] != SetMaxDirty [{1}]",
						ulGetMaxDirty, ulSetMaxDirty);
				}
				if (ulSetLowDirty != ulGetLowDirty)
				{
					System.Console.WriteLine( "GetLowDirty [{0}] != SetLowDirty [{1}]",
						ulGetLowDirty, ulSetLowDirty);
				}
			}
			endTest( false, true);

			return( true);
		}

		public bool settingsTests(
			DbSystem	dbSystem)
		{
			if (!setTempDirTest( dbSystem))
			{
				return( false);
			}
			if (!setCheckpointIntervalTest( dbSystem))
			{
				return( false);
			}
			if (!setCacheAdjustIntervalTest( dbSystem))
			{
				return( false);
			}
			if (!setCacheCleanupIntervalTest( dbSystem))
			{
				return( false);
			}
			if (!setUnusedCleanupIntervalTest( dbSystem))
			{
				return( false);
			}
			if (!setMaxUnusedTimeTest( dbSystem))
			{
				return( false);
			}
			if (!setQuerySaveMaxTest( dbSystem))
			{
				return( false);
			}
			if (!setDirtyCacheLimitsTest( dbSystem))
			{
				return( false);
			}
			return( true);
		}
	}
}
