//------------------------------------------------------------------------------
// Desc:	Db Info
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

	/// <summary>
	/// Database header - on-disk format.
	/// </summary>
	[StructLayout(LayoutKind.Sequential, Pack = 1, CharSet = CharSet.Ansi)]
	public class XFLM_DB_HDR
	{
		/// <summary>Signature - should be "FLAIMDB"</summary>
		[MarshalAs(UnmanagedType.ByValTStr, SizeConst = 8)]
		public string		szSignature;
		/// <summary>Is header in little-endian format?</summary>
		public byte			ui8IsLittleEndian;
		/// <summary>Default language.  See <see cref="Languages"/></summary>
		public byte			ui8DefaultLanguage;
		/// <summary>Database block size</summary>
		public ushort		ui16BlockSize;
		/// <summary>Database version</summary>
		public uint			ui32DbVersion;
		/// <summary>Is block checksumming enabled for the database?</summary>
		public byte			ui8BlkChkSummingEnabled;
		/// <summary>Is the database keeping roll-forward log files?</summary>
		public byte			ui8RflKeepFiles;
		/// <summary>Will the database automatically turn off keeping RFL files
		/// if the RFL volume gets full?</summary>
		public byte			ui8RflAutoTurnOffKeep;
		/// <summary>Is the database keeping aborted transactions in the roll-forward log?</summary>
		public byte			ui8RflKeepAbortedTrans;
		/// <summary>Current roll-forward log file number</summary>
		public uint			ui32RflCurrFileNum;
		/// <summary>ID of last transaction committed to the roll-forward log.</summary>
		public ulong		ui64LastRflCommitID;
		/// <summary>Last deleted RFL file number</summary>
		public uint			ui32RflLastFileNumDeleted;
		/// <summary>Offset of last transaction in the roll-forward log file</summary>
		public uint			ui32RflLastTransOffset;
		/// <summary>File number of the RFL file that has the last checkpointed transaction</summary>
		public uint			ui32RflLastCPFileNum;
		/// <summary>Offset in the RFL file where last checkpointed transaction ends.</summary>
		public uint			ui32RflLastCPOffset;
		/// <summary>Transaction ID of last checkpointed transaction</summary>
		public ulong		ui64RflLastCPTransID;
		/// <summary>Minimum RFL file size</summary>
		public uint			ui32RflMinFileSize;
		/// <summary>Maximum RFL file size</summary>
		public uint			ui32RflMaxFileSize;
		/// <summary>Current transaction ID (update transaction)</summary>
		public ulong		ui64CurrTransID;
		/// <summary>Total transactions committed</summary>
		public ulong		ui64TransCommitCnt;
		/// <summary>Roll-back log EOF</summary>
		public uint			ui32RblEOF;
		/// <summary>Offset of first checkpoint block in rollback log</summary>
		public uint			ui32RblFirstCPBlkAddr;
		/// <summary>Block address of first block in the avail block list</summary>
		public uint			ui32FirstAvailBlkAddr;
		/// <summary>Block address of first block in logical file header block list</summary>
		public uint			ui32FirstLFBlkAddr;
		/// <summary>Logical EOF for the database (this is a block address)</summary>
		public uint			ui32LogicalEOF;
		/// <summary>Maximum size for files in the database</summary>
		public uint			ui32MaxFileSize;
		/// <summary>Transaction ID of last backup taken on this database</summary>
		public ulong		ui64LastBackupTransID;
		/// <summary>Sequence number for incremental backups</summary>
		public uint			ui32IncBackupSeqNum;
		/// <summary>Number of blocks that have changed in the database since the last backup</summary>
		public uint			ui32BlksChangedSinceBackup;
		/// <summary>Database serial number</summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
		public byte[]		ucDbSerialNum;
		/// <summary>Serial number of the RFL file that contains the last committed transaction</summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
		public byte[]		ucLastTransRflSerialNum;
		/// <summary>Serial number that should be used for the next RFL file that is created</summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
		public byte[]		ucNextRflSerialNum;
		/// <summary>Serial number on the last incremental backup that was taken</summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
		public byte[]		ucIncBackupSerialNum;
		/// <summary>Length of database encrytion key</summary>
		public uint			ui32DbKeyLen;
		/// <summary>Reserved</summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 64)]
		public byte[]		ucReserved;
		/// <summary>CRC for this header</summary>
		public uint			ui32HdrCRC;
		/// <summary>Database encryption key</summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)]
		public byte[]		DbKey;
	}

	/// <summary>
	/// Object returned from <see cref="DbSystem.dbCheck"/> that contains
	/// statistics collected during the check operation.
	/// </summary>
	public class DbInfo
	{
		private IntPtr	m_pDbInfo;		// Pointer to IF_DbInfo object in unmanaged space

		/// <summary>
		/// Constructor
		/// </summary>
		/// <param name="pDbInfo">
		/// Pointer to IF_DbInfo object allocated in unmanaged space.
		/// </param>
		internal DbInfo(
			IntPtr	pDbInfo)
		{
			if (pDbInfo == IntPtr.Zero)
			{
				throw new XFlaimException( "Invalid IF_DbInfo pointer");
			}
			
			m_pDbInfo = pDbInfo;
		}

		/// <summary>
		/// Destructor
		/// </summary>
		~DbInfo()
		{
			if (m_pDbInfo != IntPtr.Zero)
			{
				xflaim_DbInfo_Release( m_pDbInfo);
				m_pDbInfo = IntPtr.Zero;
			}
		}

		[DllImport("xflaim")]
		private static extern void xflaim_DbInfo_Release(
			IntPtr	pDbInfo);

//-----------------------------------------------------------------------------
// getNumCollections
//-----------------------------------------------------------------------------

		/// <summary>
		/// Returns the number of collections in the database.
		/// </summary>
		/// <returns>Number of collections in the database.</returns>
		public uint getNumCollections()
		{
			return( xflaim_DbInfo_getNumCollections( m_pDbInfo));
		}
		
		[DllImport("xflaim")]
		private static extern uint xflaim_DbInfo_getNumCollections(
			IntPtr	pDbInfo);

//-----------------------------------------------------------------------------
// getNumIndexes
//-----------------------------------------------------------------------------

		/// <summary>
		/// Returns the number of indexes in the database.
		/// </summary>
		/// <returns>Number of indexes in the database</returns>
		public uint getNumIndexes()
		{
			return( xflaim_DbInfo_getNumIndexes( m_pDbInfo));
		}

		[DllImport("xflaim")]
		private static extern uint xflaim_DbInfo_getNumIndexes(
			IntPtr	pDbInfo);

//-----------------------------------------------------------------------------
// getNumLogicalFiles
//-----------------------------------------------------------------------------

		/// <summary>
		/// Returns the total number of collections and indexes in the database.
		/// </summary>
		/// <returns>Total number of logical files (collections and indexes)</returns>
		public uint getNumLogicalFiles()
		{
			return( xflaim_DbInfo_getNumLogicalFiles( m_pDbInfo));
		}

		[DllImport("xflaim")]
		private static extern uint xflaim_DbInfo_getNumLogicalFiles(
			IntPtr	pDbInfo);

//-----------------------------------------------------------------------------
// getDatabaseSize
//-----------------------------------------------------------------------------

		/// <summary>
		/// Returns the total size of the database (in bytes).
		/// </summary>
		/// <returns>Total size of the database</returns>
		public ulong getDatabaseSize()
		{
			return( xflaim_DbInfo_getDatabaseSize( m_pDbInfo));
		}

		[DllImport("xflaim")]
		private static extern ulong xflaim_DbInfo_getDatabaseSize(
			IntPtr	pDbInfo);

//-----------------------------------------------------------------------------
// getDbHdr
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the database header
		/// </summary>
		/// <param name="pDbHdr">
		/// Database header is returned here.
		/// </param>
		public void getDbHdr(
			XFLM_DB_HDR	pDbHdr)
		{
			xflaim_DbInfo_getDbHdr( m_pDbInfo, pDbHdr);
		}

		[DllImport("xflaim")]
		private static extern void xflaim_DbInfo_getDbHdr(
			[In]  IntPtr		pDbInfo,
			[Out] XFLM_DB_HDR	pDbHdr);

//-----------------------------------------------------------------------------
// getAvailBlockStats
//-----------------------------------------------------------------------------

		/// <summary>
		/// Return statistics on blocks in the avail list.
		/// </summary>
		/// <param name="pulBytesUsed">
		/// Total bytes of blocks in the avail list.
		/// </param>
		/// <param name="puiBlockCount">
		/// Total blocks in the avail list.
		/// </param>
		/// <param name="peLastError">
		/// Last corruption error that was reported for blocks in the avail list.
		/// </param>
		/// <param name="puiNumErrors">
		/// Total corruptions reported for blocks in the avail list.
		/// </param>
		public void getAvailBlockStats(
			out ulong					pulBytesUsed,
			out uint						puiBlockCount,
			out FlmCorruptionCode	peLastError,
			out uint						puiNumErrors)
		{
			xflaim_DbInfo_getAvailBlockStats( m_pDbInfo,
				out pulBytesUsed, out puiBlockCount,
				out peLastError, out puiNumErrors);
		}

		[DllImport("xflaim")]
		private static extern void xflaim_DbInfo_getAvailBlockStats(
			IntPtr						pDbInfo,
			out ulong					pulBytesUsed,
			out uint						puiBlockCount,
			out FlmCorruptionCode	peLastError,
			out uint						puiNumErrors);

//-----------------------------------------------------------------------------
// getLFHBlockStats
//-----------------------------------------------------------------------------

		/// <summary>
		/// Return statistics for blocks in the logical file header block list.
		/// </summary>
		/// <param name="pulBytesUsed">
		/// Total bytes of blocks in the list.
		/// </param>
		/// <param name="puiBlockCount">
		/// Total blocks in the list.
		/// </param>
		/// <param name="peLastError">
		/// Last corruption error that was reported for blocks in the list.
		/// </param>
		/// <param name="puiNumErrors">
		/// Total corruptions reported for blocks in the list.
		/// </param>
		public void getLFHBlockStats(
			out ulong					pulBytesUsed,
			out uint						puiBlockCount,
			out FlmCorruptionCode	peLastError,
			out uint						puiNumErrors)
		{
			xflaim_DbInfo_getLFHBlockStats( m_pDbInfo,
				out pulBytesUsed, out puiBlockCount,
				out peLastError, out puiNumErrors);
		}

		[DllImport("xflaim")]
		private static extern void xflaim_DbInfo_getLFHBlockStats(
			IntPtr						pDbInfo,
			out ulong					pulBytesUsed,
			out uint						puiBlockCount,
			out FlmCorruptionCode	peLastError,
			out uint						puiNumErrors);

//-----------------------------------------------------------------------------
// getBTreeInfo
//-----------------------------------------------------------------------------

		/// <summary>
		/// Returns information about a particular B-Tree in the database
		/// </summary>
		/// <param name="uiNthLogicalFile">
		/// Logical file for which information is being requested.  Note that
		/// the total number of logical files is the sum of collections and
		/// indexes.  Collections are the numbers between 0 and CollectionCount - 1.
		/// Indexes are the numbers between CollectionCount and CollectionCount + IndexCount - 1.
		/// Thus, if there are 5 collections, and 3 indexes, the collections will
		/// be in elements 0 through 4, and the indexes will be in elements 5 through 7.
		/// </param>
		/// <param name="puiLfNum">
		/// Logical file number for the requested logical file is returned here.
		/// </param>
		/// <param name="peLfType">
		/// Type of logical file is returned here.
		/// </param>
		/// <param name="puiRootBlkAddress">
		/// Root block address for the logical file is returned here.
		/// </param>
		/// <param name="puiNumLevels">
		/// Number of levels in the B-tree is returned here.
		/// </param>
		public void getBTreeInfo(
			uint					uiNthLogicalFile,
			out uint				puiLfNum,
			out eLFileType		peLfType,
			out uint				puiRootBlkAddress,
			out uint				puiNumLevels)
		{
			xflaim_DbInfo_getBTreeInfo( m_pDbInfo, uiNthLogicalFile, out puiLfNum,
				out peLfType, out puiRootBlkAddress, out puiNumLevels);
		}

		[DllImport("xflaim")]
		private static extern void xflaim_DbInfo_getBTreeInfo(
			IntPtr				pDbInfo,
			uint					uiNthLogicalFile,
			out uint				puiLfNum,
			out eLFileType		peLfType,
			out uint				puiRootBlkAddress,
			out uint				puiNumLevels);

//-----------------------------------------------------------------------------
// getBTreeBlockStats
//-----------------------------------------------------------------------------

		/// <summary>
		/// Return the statistics for a specific logical file at a specific level
		/// in the logical file's b-tree.
		/// </summary>
		/// <param name="uiNthLogicalFile">
		/// Logical file for which information is being requested.  Note that
		/// the total number of logical files is the sum of collections and
		/// indexes.  Collections are the numbers between 0 and CollectionCount - 1.
		/// Indexes are the numbers between CollectionCount and CollectionCount + IndexCount - 1.
		/// Thus, if there are 5 collections, and 3 indexes, the collections will
		/// be in elements 0 through 4, and the indexes will be in elements 5 through 7.
		/// </param>
		/// <param name="uiLevel">
		/// Level in b-tree for which information is being requested.
		/// </param>
		/// <param name="pulKeyCount">
		/// Number of keys in this level of the b-tree is returned here.
		/// </param>
		/// <param name="pulBytesUsed">
		/// Total bytes used in blocks at this level of the b-tree is returned here.
		/// </param>
		/// <param name="pulElementCount">
		/// Total elements in blocks at this level of the b-tree is returned here.
		/// </param>
		/// <param name="pulContElementCount">
		/// Total continuation elements in blocks at this level of the b-tree is returned here.
		/// </param>
		/// <param name="pulContElmBytes">
		/// Total bytes in continuation elements in blocks at this level of the b-tree is returned here.
		/// </param>
		/// <param name="puiBlockCount">
		/// Total blocks at this level of the b-tree is returned here.
		/// </param>
		/// <param name="peLastError">
		/// Last corruption found for blocks at this level of the b-tree is returned here.
		/// </param>
		/// <param name="puiNumErrors">
		/// Number of corruptions found for blocks at this level of the b-tree is returned here.
		/// </param>
		public void getBTreeBlockStats(
			uint							uiNthLogicalFile,
			uint							uiLevel,
			out ulong					pulKeyCount,
			out ulong					pulBytesUsed,
			out ulong					pulElementCount,
			out ulong					pulContElementCount,
			out ulong					pulContElmBytes,
			out uint						puiBlockCount,
			out FlmCorruptionCode	peLastError,
			out uint						puiNumErrors)
		{
			xflaim_DbInfo_getBTreeBlockStats( m_pDbInfo, uiNthLogicalFile, uiLevel, out pulKeyCount,
				out pulBytesUsed, out pulElementCount, out pulContElementCount,
				out pulContElmBytes, out puiBlockCount, out peLastError, out puiNumErrors);
		}

		[DllImport("xflaim")]
		private static extern void xflaim_DbInfo_getBTreeBlockStats(
			IntPtr						pDbInfo,
			uint							uiNthLogicalFile,
			uint							uiLevel,
			out ulong					pulKeyCount,
			out ulong					pulBytesUsed,
			out ulong					pulElementCount,
			out ulong					pulContElementCount,
			out ulong					pulContElmBytes,
			out uint						puiBlockCount,
			out FlmCorruptionCode	peLastError,
			out uint						puiNumErrors);
	}
}
