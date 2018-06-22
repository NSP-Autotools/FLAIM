//------------------------------------------------------------------------------
// Desc:	System statistics
// Tabs:	3
//
// Copyright (c) 2006 Novell, Inc. All Rights Reserved.
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

	// IMPORTANT NOTE: This needs to be kept in sync with the
	// corresponding definition in ftk.h
	/// <summary>
	/// Structure used in gathering statistics to hold an operation count elapsed time.
	/// </summary>
	[StructLayout(LayoutKind.Sequential, Pack = 1, CharSet = CharSet.Ansi)]
	public class F_COUNT_TIME_STAT
	{
		/// <summary>Number of times operation was performed</summary>
		public ulong	ulCount;
		/// <summary>Total elamsed time (milliseconds) for the operations</summary>
		public ulong	ulElapMilli;
	}

	// IMPORTANT NOTE: This needs to be kept in sync with the
	// corresponding definition in ftk.h
	/// <summary>
	/// Lock statistics.
	/// </summary>
	[StructLayout(LayoutKind.Sequential, Pack = 1, CharSet = CharSet.Ansi)]
	public class F_LOCK_STATS
	{
		/// <summary>Statistics on times when nobody was holding a lock on the database.</summary>
		public F_COUNT_TIME_STAT	NoLocks;
		/// <summary>Statistics on times threads were waiting to obtain a database lock.</summary>
		public F_COUNT_TIME_STAT	WaitingForLock;
		/// <summary>Statistics on times when a thread was holding a lock on the database.</summary>
		public F_COUNT_TIME_STAT	HeldLock;
	}

	// IMPORTANT NOTE: This needs to be kept in sync with the
	// corresponding definition in xflaim.h
	/// <summary>
	/// Read transaction statistics
	/// </summary>
	[StructLayout(LayoutKind.Sequential, Pack = 1, CharSet = CharSet.Ansi)]
	public class XFLM_RTRANS_STATS
	{
		/// <summary>Committed read transaction statistics</summary>
		public F_COUNT_TIME_STAT		CommittedTrans;
		/// <summary>Aborted read transaction statistics</summary>
		public F_COUNT_TIME_STAT		AbortedTrans;
	}

	// IMPORTANT NOTE: This needs to be kept in sync with the
	// corresponding definition in xflaim.h
	/// <summary>
	/// Update transaction statistics.
	/// </summary>
	[StructLayout(LayoutKind.Sequential, Pack = 1, CharSet = CharSet.Ansi)]
	public class XFLM_UTRANS_STATS
	{
		/// <summary>Committed update transaction statistics</summary>
		public F_COUNT_TIME_STAT		CommittedTrans;
		/// <summary>Group complete statistics</summary>
		public F_COUNT_TIME_STAT		GroupCompletes;
		/// <summary>Transactions that finished in a group</summary>
		public ulong						ulGroupFinished;
		/// <summary>Aborted update transaction statistics</summary>
		public F_COUNT_TIME_STAT		AbortedTrans;
	}

	// IMPORTANT NOTE: This needs to be kept in sync with the
	// corresponding definition in xflaim.h
	/// <summary>
	/// Disk IO statistics.
	/// </summary>
	[StructLayout(LayoutKind.Sequential, Pack = 1, CharSet = CharSet.Ansi)]
	public class XFLM_DISKIO_STAT
	{
		/// <summary>Number of times read or write operation was performed.</summary>
		public ulong	ulCount;
		/// <summary>Total number of bytes read or written.</summary>
		public ulong	ulTotalBytes;
		/// <summary>Total elapsed time (milliseconds) for the read or write operations.</summary>
		public ulong	ulElapMilli;
	}

	// IMPORTANT NOTE: This needs to be kept in sync with the
	// corresponding definition in xflaim.h
	/// <summary>
	/// Block statistics.
	/// </summary>
	[StructLayout(LayoutKind.Sequential, Pack = 1, CharSet = CharSet.Ansi)]
	public class XFLM_BLOCKIO_STATS
	{
		/// <summary>Statistics on block reads</summary>
		public XFLM_DISKIO_STAT	BlockReads;
		/// <summary>Statistics on block writes</summary>
		public XFLM_DISKIO_STAT	BlockWrites;
		/// <summary>Statistics on old view block reads</summary>
		public XFLM_DISKIO_STAT	OldViewBlockReads;
		/// <summary>Number of checksum errors reading blocks.</summary>
		public uint					uiBlockChkErrs;
		/// <summary>Number of checksum errors reading old versions of blocks.</summary>
		public uint					uiOldViewBlockChkErrs;
		/// <summary>Number of times we had an old view error when reading</summary>
		public uint					uiOldViewErrors;
	}

	// IMPORTANT NOTE: This needs to be kept in sync with the
	// corresponding definition in DbSystemStats.cpp
	/// <summary>
	/// Logical file statistics
	/// </summary>
	[StructLayout(LayoutKind.Sequential, Pack = 1, CharSet = CharSet.Ansi)]
	public class CS_XFLM_LFILE_STATS
	{
		/// <summary>Block statistics for root blocks in this logical file.</summary>
		public XFLM_BLOCKIO_STATS	RootBlockStats;
		/// <summary>Block statistics for middle blocks in this logical file.</summary>
		public XFLM_BLOCKIO_STATS	MiddleBlockStats;
		/// <summary>Block statistics for leaf blocks in this logical file.</summary>
		public XFLM_BLOCKIO_STATS	LeafBlockStats;
		/// <summary>Total blocks splits that have been done in this logical file.</summary>
		public ulong					ulBlockSplits;
		/// <summary>Total block combines that have been done in this logical file.</summary>
		public ulong					ulBlockCombines;
		/// <summary>Logical file number.</summary>
		public uint						uiLFileNum;
		/// <summary>Type of logical file.</summary>
		public eLFileType				eLfType;
	}

	// IMPORTANT NOTE: This needs to be kept in sync with the
	// corresponding definition in DbSystemStats.cpp
	/// <summary>
	/// Database statistics
	/// </summary>
	[StructLayout(LayoutKind.Sequential, Pack = 1, CharSet = CharSet.Ansi)]
	public class CS_XFLM_DB_STATS
	{
		/// <summary>Database name</summary>
		[MarshalAs(UnmanagedType.LPStr)]
		public string						sDbName;
		/// <summary>Number of logical files we have statistics for</summary>
		public uint							uiNumLFiles;
		/// <summary>Read transaction statistics</summary>
		public XFLM_RTRANS_STATS		ReadTransStats;
		/// <summary>Update transaction statistics</summary>
		public XFLM_UTRANS_STATS		UpdateTransStats;
		/// <summary>Block statistics for Logical File Header blocks</summary>
		public XFLM_BLOCKIO_STATS		LFHBlockStats;
		/// <summary>Block statistics for avail blocks</summary>
		public XFLM_BLOCKIO_STATS		AvailBlockStats;
		/// <summary>Disk IO statistics for database header writes</summary>
		public XFLM_DISKIO_STAT			DbHdrWrites;
		/// <summary>Disk IO statistics for log block writes</summary>
		public XFLM_DISKIO_STAT			LogBlockWrites;
		/// <summary>Disk IO statistics for log block restores</summary>
		public XFLM_DISKIO_STAT			LogBlockRestores;
		/// <summary>Disk IO statistics for log block reads</summary>
		public XFLM_DISKIO_STAT			LogBlockReads;
		/// <summary>Log block checksum errors</summary>
		public uint							uiLogBlockChkErrs;
		/// <summary>Total read errors</summary>
		public uint							uiReadErrors;
		/// <summary>Total write errors</summary>
		public uint							uiWriteErrors;
		/// <summary>Lock statistics</summary>
		public F_LOCK_STATS				LockStats;
	}

	/// <summary>
	/// The DbSystemStats class provides a number of methods that allow C#
	/// applications to access statistics that have been requested.  A
	/// DbSystemStats object is obtained by calling <see cref="DbSystem.getStats"/>.
	/// </summary>
	public class DbSystemStats
	{
		private IntPtr		m_pStats;			// Pointer to XFLM_STATS object in unmanaged space
		private DbSystem 	m_dbSystem;

		/// <summary>
		/// Constructor.
		/// </summary>
		/// <param name="pStats">
		/// Pointer to an XFLM_STATS object in unmanaged space.
		/// </param>
		/// <param name="dbSystem">
		/// DbSystem object that this DbSystemStats object is associated with.
		/// </param>
		internal DbSystemStats(
			IntPtr	pStats,
			DbSystem	dbSystem)
		{
			if (pStats == IntPtr.Zero)
			{
				throw new XFlaimException( "Invalid pointer to XFLM_STATS structure");
			}
			
			m_pStats = pStats;

			if (dbSystem == null)
			{
				throw new XFlaimException( "Invalid DbSystem reference");
			}
			
			m_dbSystem = dbSystem;
			
			// Must call something inside of DbSystem.  Otherwise, the
			// m_dbSystem object gets a compiler warning on linux because
			// it is not used anywhere.  Other than that, there is really
			// no need to make the following call.
			if (m_dbSystem.getDbSystem() == IntPtr.Zero)
			{
				throw new XFlaimException( "Invalid DbSystem.IF_DbSystem object");
			}
		}

		/// <summary>
		/// Destructor.
		/// </summary>
		~DbSystemStats()
		{
			freeStats();
		}

		/// <summary>
		/// Free the statistics assocated with this object.
		/// </summary>
		public void freeStats()
		{
			// Free the unmanaged XFLM_STATS structure.
		
			if (m_pStats != IntPtr.Zero)
			{
				xflaim_DbSystemStats_freeStats( m_dbSystem.getDbSystem(), m_pStats);
				m_pStats = IntPtr.Zero;
			}
		
			// Remove our reference to the dbSystem so it can be released.
		
			m_dbSystem = null;
		}

		[DllImport("xflaim")]
		private static extern void xflaim_DbSystemStats_freeStats(
			IntPtr	pDbSystem,
			IntPtr	pStats);

//-----------------------------------------------------------------------------
// getGeneralStats
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieve some general statistics.
		/// </summary>
		/// <param name="uiNumDatabases">
		/// Returns number of databases there are statistics for.
		/// </param>
		/// <param name="uiStartTime">
		/// Returns time statistics were started.
		/// </param>
		/// <param name="uiStopTime">
		/// Returns time statistics were stopped.
		/// </param>
		public void getGeneralStats(
			out uint	uiNumDatabases,
			out uint	uiStartTime,
			out uint	uiStopTime)
		{
			xflaim_DbSystemStats_getGeneralStats( m_pStats,
					out uiNumDatabases, out uiStartTime, out uiStopTime);
		}

		[DllImport("xflaim")]
		private static extern void xflaim_DbSystemStats_getGeneralStats(
			IntPtr	pStats,
			out uint	puiNumDatabases,
			out uint	puiStartTime,
			out uint	puiStopTime);

//-----------------------------------------------------------------------------
// getDbStats
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieve statistics for a particular database.
		/// </summary>
		/// <param name="uiDatabaseNum">
		/// Number of the database for which statistics are to be returned.
		/// This number should be between 0 and N - 1, where N = Number of
		/// databases for which we have statistics.
		/// The number of databases may be determined by calling the
		/// <see cref="getGeneralStats"/> method.
		/// </param>
		/// <param name="dbStats">
		/// A <see cref="CS_XFLM_DB_STATS"/> object to return the requested
		/// database statistics in.  If null, an object will be allocated.
		/// </param>
		/// <returns>
		/// A <see cref="CS_XFLM_DB_STATS"/> object is returned which
		/// holds statistics for the specified database.
		/// </returns>
		public CS_XFLM_DB_STATS getDbStats(
			uint					uiDatabaseNum,
			CS_XFLM_DB_STATS	dbStats)
		{
			RCODE	rc;

			if (dbStats == null)
			{
				dbStats = new CS_XFLM_DB_STATS();
			}

			if ((rc = xflaim_DbSystemStats_getDbStats( m_pStats, uiDatabaseNum,
				dbStats)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( dbStats);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DbSystemStats_getDbStats(
			IntPtr				pStats,
			uint					uiDatabaseNum,
			[Out]
			CS_XFLM_DB_STATS	pDbStats);

//-----------------------------------------------------------------------------
// getLFileStats
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieve statistics for a particular logical file in a particular
		/// database.
		/// </summary>
		/// <param name="uiDatabaseNum">
		/// Number of the database for which logical file
		/// statistics are to be returned.
		/// This number should be between 0 and N - 1, where N = Number of
		/// databases for which we have statistics.
		/// The number of databases may be determined by calling the
		/// <see cref="getGeneralStats"/> method.
		/// </param>
		/// <param name="uiLFileNum">
		/// Number of the logical file for which logical file
		/// statistics are to be returned.
		/// This number should be between 0 and N - 1, where N = Number of
		/// logical files for the specified database.  The number of logical
		/// files for a particular database is found in the
		/// <see cref="CS_XFLM_DB_STATS.uiNumLFiles"/> member of the
		/// <see cref="CS_XFLM_DB_STATS"/> object, which is returned from the
		/// <see cref="getDbStats"/> method.
		/// </param>
		/// <param name="lFileStats">
		/// A <see cref="CS_XFLM_LFILE_STATS"/> object to return the requested
		/// statistics in.  If null, an object will be allocated.
		/// </param>
		/// <returns>
		/// A <see cref="CS_XFLM_LFILE_STATS"/> object is returned which
		/// holds statistics for the specified logical file of the
		/// specified database.
		/// </returns>
		public CS_XFLM_LFILE_STATS getLFileStats(
			uint						uiDatabaseNum,
			uint						uiLFileNum,
			CS_XFLM_LFILE_STATS	lFileStats)
		{
			RCODE	rc;

			if (lFileStats == null)
			{
				lFileStats = new CS_XFLM_LFILE_STATS();
			}

			if ((rc = xflaim_DbSystemStats_getLFileStats( m_pStats, uiDatabaseNum,
				uiLFileNum, lFileStats)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( lFileStats);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DbSystemStats_getLFileStats(
			IntPtr					pStats,
			uint						uiDatabaseNum,
			uint						uiLFileNum,
			[Out]
			CS_XFLM_LFILE_STATS	pLFileStats);

	}
}
