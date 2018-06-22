//------------------------------------------------------------------------------
// Desc:	CSharp Tester
// Tabs:	3
//
// Copyright (c) 2007 Novell, Inc. All Rights Reserved.
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
	// Cache tests
	//--------------------------------------------------------------------------
	public class CacheTests : Tester
	{
		public bool cacheTests(
			DbSystem	dbSystem)
		{
			uint						uiCacheAdjustPercent = 66;
			ulong						ulCacheAdjustMin = 20000000;
			ulong						ulCacheAdjustMax = 1000000000;
			ulong						ulCacheAdjustMinToLeave = 0;
#if !mono
			CS_XFLM_CACHE_INFO	cacheInfo;
#endif

			beginTest( "Set dynamic cache limit test");
			try
			{
				dbSystem.setDynamicMemoryLimit( uiCacheAdjustPercent,
					ulCacheAdjustMin, ulCacheAdjustMax,
					ulCacheAdjustMinToLeave);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling setDynamicMemoryLimit");
				return( false);
			}
			endTest( false, true);

#if !mono
			beginTest( "Get cache info for dynamic cache limit test");
			try
			{
				cacheInfo = dbSystem.getCacheInfo();
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling getCacheInfo");
				return( false);
			}
			endTest( false, true);

			beginTest( "See if cache limits were set");
			if (cacheInfo.bDynamicCacheAdjust == 0 ||
				cacheInfo.uiCacheAdjustPercent != uiCacheAdjustPercent ||
				cacheInfo.ulCacheAdjustMin != ulCacheAdjustMin ||
				cacheInfo.ulCacheAdjustMax != ulCacheAdjustMax ||
				cacheInfo.ulCacheAdjustMinToLeave != ulCacheAdjustMinToLeave)
			{
				endTest( false, false);
				System.Console.WriteLine( "Dynamic cache adjust parameter mismatch");
				System.Console.WriteLine( "Dynamic Adjust Flag..... Set: true Get: {0}",
					cacheInfo.bDynamicCacheAdjust != 0 ? "true" : "false");
				System.Console.WriteLine( "Adjust Percent.......... Set: {0} Get: {1}",
					uiCacheAdjustPercent, cacheInfo.uiCacheAdjustPercent);
				System.Console.WriteLine( "Adjust Min.............. Set: {0} Get: {1}",
					ulCacheAdjustMin, cacheInfo.ulCacheAdjustMin);
				System.Console.WriteLine( "Adjust Max.............. Set: {0} Get: {1}",
					ulCacheAdjustMax, cacheInfo.ulCacheAdjustMax);
				System.Console.WriteLine( "Adjust Min To Leave..... Set: {0} Get: {1}",
					ulCacheAdjustMinToLeave, cacheInfo.ulCacheAdjustMinToLeave);
				return( false);
			}
			endTest( false, true);
#endif

			// SET AND TEST A HARD LIMIT

			beginTest( "Set hard cache limit test");
			try
			{
				dbSystem.setHardMemoryLimit( 0, false, 0, ulCacheAdjustMax,
					0, false);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling setHardMemoryLimit");
				return( false);
			}
			endTest( false, true);

#if !mono
			beginTest( "Get cache info for hard cache limit test");
			try
			{
				cacheInfo = dbSystem.getCacheInfo();
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling getCacheInfo");
				return( false);
			}
			endTest( false, true);

			beginTest( "See if cache limits were set");
			if (cacheInfo.bDynamicCacheAdjust != 0 ||
				cacheInfo.ulCacheAdjustMax != ulCacheAdjustMax ||
				cacheInfo.ulMaxBytes != ulCacheAdjustMax)
			{
				endTest( false, false);
				System.Console.WriteLine( "Hard cache adjust parameter mismatch");
				System.Console.WriteLine( "Dynamic Adjust Flag..... Set: false Get: {0}",
					cacheInfo.bDynamicCacheAdjust != 0 ? "true" : "false");
				System.Console.WriteLine( "Max..................... Set: {0} Get: {1}",
					ulCacheAdjustMax, cacheInfo.ulCacheAdjustMax);
				System.Console.WriteLine( "Max Bytes............... Set: {0} Get: {1}",
					ulCacheAdjustMax, cacheInfo.ulMaxBytes);
				return( false);
			}
			endTest( false, true);

			printCacheUsage( cacheInfo.blockCache, "BLOCK CACHE USAGE");
			printCacheUsage( cacheInfo.nodeCache, "NODE CACHE USAGE");
#endif

			return( true);
		}
	}
}
