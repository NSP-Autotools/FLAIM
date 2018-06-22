//------------------------------------------------------------------------------
// Desc:	Cache information structures.
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

	// IMPORTANT NOTE: This structure must be kept in sync
	// with the corresponding structure in ftk.h
	/// <summary>
	/// Information about slab usage
	/// </summary>
	[StructLayout(LayoutKind.Sequential, Pack = 1, CharSet = CharSet.Ansi)]
	public class FLM_SLAB_USAGE
	{
		/// <summary>Total slabs currently allocated.</summary>
		public ulong			ulSlabs;
		/// <summary>Total bytes currently allocated in slabs.</summary>
		public ulong			ulSlabBytes;
		/// <summary>Total cells allocated within slabs.</summary>
		public ulong			ulAllocatedCells;
		/// <summary>Total cells that are free within slabs.</summary>
		public ulong			ulFreeCells;
	}

	// IMPORTANT NOTE: This structure must be kept in sync
	// with the corresponding structure in xflaim.h
	/// <summary>
	/// Information about usage within a particular cache - either block
	/// cache or node cache.
	/// </summary>
	[StructLayout(LayoutKind.Sequential, Pack = 1, CharSet = CharSet.Ansi)]
	public class CS_XFLM_CACHE_USAGE
	{
		/// <summary>Total bytes allocated for current versions of objects</summary>
		public ulong					ulByteCount;
		/// <summary>Count of current versions of objects</summary>
		public ulong					ulCount;
		/// <summary>Count of older versions of objects</summary>
		public ulong					ulOldVerCount;
		/// <summary>Total bytes allocated for older versions of objects</summary>
		public ulong					ulOldVerBytes;
		/// <summary>Cache hits</summary>
		public uint						uiCacheHits;
		/// <summary>Cache hit looks</summary>
		public uint						uiCacheHitLooks;
		/// <summary>Cache faults</summary>
		public uint						uiCacheFaults;
		/// <summary>cache fault looks</summary>
		public uint						uiCacheFaultLooks;
		/// <summary>Slab usage</summary>
		public FLM_SLAB_USAGE		slabUsage;
	}

	// IMPORTANT NOTE: This structure must be kept in sync
	// with the corresponding structure in xflaim.h
	/// <summary>
	/// Information about the current state of cache.
	/// </summary>
	[StructLayout(LayoutKind.Sequential, Pack = 1, CharSet = CharSet.Ansi)]
	public class CS_XFLM_CACHE_INFO
	{
		/// <summary>Current limit for cache</summary>
		public ulong					ulMaxBytes;
		/// <summary>Total bytes allocated to cache</summary>
		public ulong					ulTotalBytesAllocated;
		/// <summary>Flag indicating if cache limit is being dynamically adjusted</summary>
		public int						bDynamicCacheAdjust;
		/// <summary>
		/// Only used if bDynamicCacheAdjust is non-zero.
		/// Percent of available memory that the cache limit is to be set to. A
		/// new cache limit is periodically recalculated based on this percentage.
		/// </summary>
		public uint						uiCacheAdjustPercent;
		/// <summary>
		/// Only used if bDynamicCacheAdjust is non-zero.
		/// Minimum value that the cache limit is to be set to whenever a new
		/// cache limit is calculated.
		/// </summary>
		public ulong					ulCacheAdjustMin;
		/// <summary>
		/// Only used if bDynamicCacheAdjust is non-zero.
		/// Maximum value that the cache limit is to be set to whenever a new cache
		/// limit is calculated.
		/// </summary>
		public ulong					ulCacheAdjustMax;
		/// <summary>
		/// Only used if bDynamicCacheAdjust is non-zero.
		/// This is an alternative way to specify a maximum cache limit.  If zero, 
		/// this parameter is ignored and uiCacheAdjustMax is used.  If non-zero,
		/// the maximum cache limit is calculated to be the amount of available
		/// memory minus this number - the idea being to leave a certain amount of
		/// memory for other processes to use.
		/// </summary>
		public ulong					ulCacheAdjustMinToLeave;
		/// <summary>Number of blocks in block cache that are dirty</summary>
		public ulong					ulDirtyCount;
		/// <summary>Number of bytes in block cache that are dirty</summary>
		public ulong					ulDirtyBytes;
		/// <summary>
		/// Number of blocks in block cache that are new.  New blocks
		/// are blocks that were allocated at the end of a database when there
		/// were no blocks in the avail list to allocate.
		/// </summary>
		public ulong					ulNewCount;
		/// <summary>Total bytes in new blocks.</summary>
		public ulong					ulNewBytes;
		/// <summary>Number of blocks that need to be written to the rollback log.</summary>
		public ulong					ulLogCount;
		/// <summary>Total bytes for blocks that need to be written to the rollback log.</summary>
		public ulong					ulLogBytes;
		/// <summary>Total blocks that need to be freed</summary>
		public ulong					ulFreeCount;
		/// <summary>Total bytes in freed blocks.</summary>
		public ulong					ulFreeBytes;
		/// <summary>
		/// Total blocks that can be replaced without having to write them
		/// to disk or the rollback log.
		/// </summary>
		public ulong					ulReplaceableCount;
		/// <summary>Total bytes in the replaceable blocks.</summary>
		public ulong					ulReplaceableBytes;
		/// <summary>Statistics on block cache.</summary>
		public CS_XFLM_CACHE_USAGE	blockCache;
		/// <summary>Statistics on node cache.</summary>
		public CS_XFLM_CACHE_USAGE	nodeCache;
		/// <summary>Flag indicating whether cache was preallocated.</summary>
		public int						bPreallocatedCache;
	}
}
