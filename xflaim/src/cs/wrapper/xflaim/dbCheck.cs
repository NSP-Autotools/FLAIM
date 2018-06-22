//------------------------------------------------------------------------------
// Desc:	Db Check Status
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
	/// Flags to be used for checking a database.
	/// </summary>
	[Flags]
	public enum DbCheckFlags : int
	{
		/// <summary>Repair index corruptions on-line.</summary>
		XFLM_ONLINE						= 0x0020,
		/// <summary>Don't execute the redo log before doing check.</summary>
		XFLM_DONT_REDO_LOG			= 0x0040,
		/// <summary>Don't resume background threads while doing check check.</summary>
		XFLM_DONT_RESUME_THREADS	= 0x0080,
		/// <summary>Check for index logical corruptions.</summary>
		XFLM_DO_LOGICAL_CHECK		= 0x0100,
		/// <summary>Don't check DOM links.</summary>
		XFLM_SKIP_DOM_LINK_CHECK	= 0x0200,
		/// <summary>Allow database to be opened in limited mode for check.</summary>
		XFLM_ALLOW_LIMITED_MODE		= 0x0400
	}

	// IMPORTANT NOTE: Any additions or changes made in the C++ code should be
	// synced to here.
	/// <summary>
	/// Corruption codes that can be returned from <see cref="DbSystem.dbCheck"/> when it
	/// reports a corruption.
	/// </summary>
	public enum FlmCorruptionCode : int
	{
		/// <summary></summary>
		FLM_BAD_CHAR							= 1,
		/// <summary></summary>
		FLM_BAD_ASIAN_CHAR					= 2,
		/// <summary></summary>
		FLM_BAD_CHAR_SET						= 3,
		/// <summary></summary>
		FLM_BAD_TEXT_FIELD					= 4,
		/// <summary></summary>
		FLM_BAD_NUMBER_FIELD					= 5,
		/// <summary></summary>
		FLM_BAD_FIELD_TYPE					= 6,
		/// <summary></summary>
		FLM_BAD_IX_DEF							= 7,
		/// <summary></summary>
		FLM_MISSING_REQ_KEY_FIELD			= 8,
		/// <summary></summary>
		FLM_BAD_TEXT_KEY_COLL_CHAR			= 9,
		/// <summary></summary>
		FLM_BAD_TEXT_KEY_CASE_MARKER		= 10,
		/// <summary></summary>
		FLM_BAD_NUMBER_KEY					= 11,
		/// <summary></summary>
		FLM_BAD_BINARY_KEY					= 12,
		/// <summary></summary>
		FLM_BAD_CONTEXT_KEY					= 13,
		/// <summary></summary>
		FLM_BAD_KEY_FIELD_TYPE				= 14,
		/// <summary></summary>
		Not_Used_15								= 15,
		/// <summary></summary>
		Not_Used_16								= 16,
		/// <summary></summary>
		Not_Used_17								= 17,
		/// <summary></summary>
		FLM_BAD_KEY_LEN						= 18,
		/// <summary></summary>
		FLM_BAD_LFH_LIST_PTR					= 19,
		/// <summary></summary>
		FLM_BAD_LFH_LIST_END					= 20,
		/// <summary></summary>
		FLM_INCOMPLETE_NODE					= 21,
		/// <summary></summary>
		FLM_BAD_BLK_END						= 22,
		/// <summary></summary>
		FLM_KEY_COUNT_MISMATCH				= 23,
		/// <summary></summary>
		FLM_REF_COUNT_MISMATCH				= 24,
		/// <summary></summary>
		FLM_BAD_CONTAINER_IN_KEY			= 25,
		/// <summary></summary>
		FLM_BAD_BLK_HDR_ADDR					= 26,
		/// <summary></summary>
		FLM_BAD_BLK_HDR_LEVEL				= 27,
		/// <summary></summary>
		FLM_BAD_BLK_HDR_PREV					= 28,
		/// <summary></summary>
		FLM_BAD_BLK_HDR_NEXT					= 29,
		/// <summary></summary>
		FLM_BAD_BLK_HDR_TYPE					= 30,
		/// <summary></summary>
		FLM_BAD_BLK_HDR_ROOT_BIT			= 31,
		/// <summary></summary>
		FLM_BAD_BLK_HDR_BLK_END				= 32,
		/// <summary></summary>
		FLM_BAD_BLK_HDR_LF_NUM				= 33,
		/// <summary></summary>
		FLM_BAD_AVAIL_LIST_END				= 34,
		/// <summary></summary>
		FLM_BAD_PREV_BLK_NEXT				= 35,
		/// <summary></summary>
		FLM_BAD_FIRST_ELM_FLAG				= 36,
		/// <summary></summary>
		FLM_BAD_LAST_ELM_FLAG				= 37,
		/// <summary></summary>
		FLM_BAD_LEM								= 38,
		/// <summary></summary>
		FLM_BAD_ELM_LEN						= 39,
		/// <summary></summary>
		FLM_BAD_ELM_KEY_SIZE					= 40,
		/// <summary></summary>
		FLM_BAD_ELM_KEY						= 41,
		/// <summary></summary>
		FLM_BAD_ELM_KEY_ORDER				= 42,
		/// <summary></summary>
		FLM_BAD_ELM_KEY_COMPRESS			= 43,
		/// <summary></summary>
		FLM_BAD_CONT_ELM_KEY					= 44,
		/// <summary></summary>
		FLM_NON_UNIQUE_FIRST_ELM_KEY		= 45,
		/// <summary></summary>
		FLM_BAD_ELM_OFFSET					= 46,
		/// <summary></summary>
		FLM_BAD_ELM_INVALID_LEVEL			= 47,
		/// <summary></summary>
		FLM_BAD_ELM_FLD_NUM					= 48,
		/// <summary></summary>
		FLM_BAD_ELM_FLD_LEN					= 49,
		/// <summary></summary>
		FLM_BAD_ELM_FLD_TYPE					= 50,
		/// <summary></summary>
		FLM_BAD_ELM_END						= 51,
		/// <summary></summary>
		FLM_BAD_PARENT_KEY					= 52,
		/// <summary></summary>
		FLM_BAD_ELM_DOMAIN_SEN				= 53,
		/// <summary></summary>
		FLM_BAD_ELM_BASE_SEN					= 54,
		/// <summary></summary>
		FLM_BAD_ELM_IX_REF					= 55,
		/// <summary></summary>
		FLM_BAD_ELM_ONE_RUN_SEN				= 56,
		/// <summary></summary>
		FLM_BAD_ELM_DELTA_SEN				= 57,
		/// <summary></summary>
		FLM_BAD_ELM_DOMAIN					= 58,
		/// <summary></summary>
		FLM_BAD_LAST_BLK_NEXT				= 59,
		/// <summary></summary>
		FLM_BAD_FIELD_PTR						= 60,
		/// <summary></summary>
		FLM_REBUILD_REC_EXISTS				= 61,
		/// <summary></summary>
		FLM_REBUILD_KEY_NOT_UNIQUE			= 62,
		/// <summary></summary>
		FLM_NON_UNIQUE_ELM_KEY_REF			= 63,
		/// <summary></summary>
		FLM_OLD_VIEW							= 64,
		/// <summary></summary>
		FLM_COULD_NOT_SYNC_BLK				= 65,
		/// <summary></summary>
		FLM_IX_REF_REC_NOT_FOUND			= 66,
		/// <summary></summary>
		FLM_IX_KEY_NOT_FOUND_IN_REC		= 67,
		/// <summary></summary>
		FLM_KEY_NOT_IN_KEY_REFSET			= 68,
		/// <summary></summary>
		FLM_BAD_BLK_CHECKSUM					= 69,
		/// <summary></summary>
		FLM_BAD_LAST_DRN						= 70,
		/// <summary></summary>
		FLM_BAD_FILE_SIZE						= 71,
		/// <summary></summary>
		FLM_BAD_FIRST_LAST_ELM_FLAG		= 72,
		/// <summary></summary>
		FLM_BAD_DATE_FIELD					= 73,
		/// <summary></summary>
		FLM_BAD_TIME_FIELD					= 74,
		/// <summary></summary>
		FLM_BAD_TMSTAMP_FIELD				= 75,
		/// <summary></summary>
		FLM_BAD_DATE_KEY						= 76,
		/// <summary></summary>
		FLM_BAD_TIME_KEY						= 77,
		/// <summary></summary>
		FLM_BAD_TMSTAMP_KEY					= 78,
		/// <summary></summary>
		FLM_BAD_BLOB_FIELD					= 79,
		/// <summary></summary>
		FLM_BAD_PCODE_IXD_TBL				= 80,
		/// <summary></summary>
		FLM_NODE_QUARANTINED					= 81,
		/// <summary></summary>
		FLM_BAD_BLK_TYPE						= 82,
		/// <summary></summary>
		FLM_BAD_ELEMENT_CHAIN				= 83,
		/// <summary></summary>
		FLM_BAD_ELM_EXTRA_DATA				= 84,
		/// <summary></summary>
		FLM_BAD_BLOCK_STRUCTURE				= 85,
		/// <summary></summary>
		FLM_BAD_ROOT_PARENT					= 86,
		/// <summary></summary>
		FLM_BAD_ROOT_LINK						= 87,
		/// <summary></summary>
		FLM_BAD_PARENT_LINK					= 88,
		/// <summary></summary>
		FLM_BAD_INVALID_ROOT					= 89,
		/// <summary></summary>
		FLM_BAD_FIRST_CHILD_LINK			= 90,
		/// <summary></summary>
		FLM_BAD_LAST_CHILD_LINK				= 91,
		/// <summary></summary>
		FLM_BAD_PREV_SIBLING_LINK			= 92,
		/// <summary></summary>
		FLM_BAD_NEXT_SIBLING_LINK			= 93,
		/// <summary></summary>
		FLM_BAD_ANNOTATION_LINK				= 94,
		/// <summary></summary>
		FLM_UNSUPPORTED_NODE_TYPE			= 95,
		/// <summary></summary>
		FLM_BAD_INVALID_NAME_ID				= 96,
		/// <summary></summary>
		FLM_BAD_INVALID_PREFIX_ID			= 97,
		/// <summary></summary>
		FLM_BAD_DATA_BLOCK_COUNT			= 98,
		/// <summary></summary>
		FLM_BAD_AVAIL_SIZE					= 99,
		/// <summary></summary>
		FLM_BAD_NODE_TYPE						= 100,
		/// <summary></summary>
		FLM_BAD_CHILD_ELM_COUNT				= 101
	}
		
	// IMPORTANT NOTE: Any additions or changes made in the C++ code should be
	// synced to here.
	/// <summary>
	/// Locations in the database where corruptions can occur.
	/// </summary>
	public enum ErrLocale : int
	{
		/// <summary></summary>
		XFLM_LOCALE_NONE				= 0,
		/// <summary>Error occurred in a logical file header block.</summary>
		XFLM_LOCALE_LFH_LIST			= 1,
		/// <summary>Error occurred in an avail block.</summary>
		XFLM_LOCALE_AVAIL_LIST		= 2,
		/// <summary>Error occurred in a B-tree block.</summary>
		XFLM_LOCALE_B_TREE			= 3,
		/// <summary>Error is a logical index error.</summary>
		XFLM_LOCALE_INDEX				= 4
	}

	// IMPORTANT NOTE: This structure needs to stay in sync with the XFLM_CORRUPT_INFO
	// structure defined in xflaim.h
	/// <summary>
	/// Class that reports a corruption when it is detected by the
	/// <see cref="DbSystem.dbCheck"/> operation.
	/// It is returned in the <see cref="DbCheckStatus.reportCheckErr"/> method.
	/// </summary>
	[StructLayout(LayoutKind.Sequential, Pack = 1, CharSet = CharSet.Ansi)]
	public class XFLM_CORRUPT_INFO
	{
		/// <summary>Corruption error code being reported.</summary>
		public FlmCorruptionCode	eErrCode;
		/// <summary>Locale in database where corruption was found.</summary>
		public ErrLocale				eErrLocale;
		/// <summary>
		/// Logical file number (collection or index) where corruption was found.
		/// Will be zero if the corruption does not pertain to a logical file.
		/// </summary>
		public uint						uiErrLfNumber;
		/// <summary>Type of logical file.</summary>
		public eLFileType				eErrLfType;
		/// <summary>
		/// Level in the B-tree where corruption was found
		/// This is only set if eErrLocale == XFLM_LOCALE_B_TREE.
		/// </summary>
		public uint						uiErrBTreeLevel;
		/// <summary>
		/// Block address where corruption was found.
		/// A value of zero means there was no block address reported.
		/// </summary>
		public uint						uiErrBlkAddress;
		/// <summary>
		/// Parent block address of block where corruption was found.
		/// This is only set if eErrLocale == XFLM_LOCALE_B_TREE.
		/// A value of zero means there was no parent block address reported.
		/// </summary>
		public uint						uiErrParentBlkAddress;
		/// <summary>
		/// Element offset in block where corruption was found.
		/// This is only set if eErrLocale == XFLM_LOCALE_B_TREE.
		/// </summary>
		public uint						uiErrElmOffset;
		/// <summary>
		/// Node ID where corruption was found.  If zero, no node ID was
		/// reported for the corruption.
		/// </summary>
		public ulong					ulErrNodeId;

		// NOTE: This structure does NOT have the index key that is defined
		// in xflaim.h.  That is by design.
	}

	// IMPORTANT NOTE: These enums need to stay in sync with the corresponding
	// definitions in xflaim.h
	/// <summary>
	/// This enum defines the various phases that a <see cref="DbSystem.dbCheck"/>
	/// operation goes through.
	/// </summary>
	public enum FlmCheckPhase : int
	{
		/// <summary>Checking logical file header blocks.</summary>
		XFLM_CHECK_LFH_BLOCKS			= 1,
		/// <summary>Checking B-trees (collections and indexes).</summary>
		XFLM_CHECK_B_TREE					= 2,
		/// <summary>Checking blocks in the avail list.</summary>
		XFLM_CHECK_AVAIL_BLOCKS			= 3,
		/// <summary>Sorting temporary result sets.</summary>
		XFLM_CHECK_RS_SORT				= 4,
		/// <summary>Checking DOM links.</summary>
		XFLM_CHECK_DOM_LINKS				= 5
	}

	// IMPORTANT NOTE: This structure needs to stay in sync with the XFLM_PROGRESS_CHECK_INFO
	// structure defined in xflaim.h
	/// <summary>
	/// Class that reports progress information for a <see cref="DbSystem.dbCheck"/> operation.
	/// It is returned in the <see cref="DbCheckStatus.reportProgress"/> method.
	/// </summary>
	[StructLayout(LayoutKind.Sequential, Pack = 1, CharSet = CharSet.Ansi)]
	public class XFLM_PROGRESS_CHECK_INFO
	{
		/// <summary>
		/// Indicates which phase of the check operation <see cref="DbSystem.dbCheck"/>
		/// is currently in.
		/// </summary>
		public FlmCheckPhase	eCheckPhase;
		/// <summary>
		/// Indicates whether we are just starting this phase of the operation.  Value
		/// will be non-zero if just starting, zero otherwise.
		/// </summary>
		public int				bStartFlag;
		/// <summary>
		/// Total number of bytes in the database.
		/// </summary>
		public ulong			ulDatabaseSize;
		/// <summary>
		/// Total number of logical files (collections and indexes) to be checked.
		/// </summary>
		public uint				uiNumLFs;
		/// <summary>
		/// Logical file being checked (nth of N).
		/// This is only set if eCheckPhase == XFLM_CHECK_B_TREE.
		/// </summary>
		public uint				uiCurrLF;
		/// <summary>
		/// Logical file number of the logical file being checked.
		/// This is only set if eCheckPhase == XFLM_CHECK_B_TREE.
		/// </summary>
		public uint				uiLfNumber;
		/// <summary>
		/// Logical file type of the logical file being checked.
		/// This is only set if eCheckPhase == XFLM_CHECK_B_TREE.
		/// </summary>
		public eLFileType		eLfType;
		/// <summary>
		/// Total bytes checked thus far.
		/// </summary>
		public ulong			ulBytesExamined;
		/// <summary>
		/// Number of corruptions repaired so far.
		/// </summary>
		public uint				uiNumProblemsFixed;
		/// <summary>
		/// Number of DOM nodes traversed so far in the current logical file.
		/// </summary>
		public ulong			ulNumDomNodes;
		/// <summary>
		/// Number of DOM links verified so far in the current logical file.
		/// </summary>
		public ulong			ulNumDomLinksVerified;
		/// <summary>
		/// Number of broken DOM links found so far in the current logical file.
		/// </summary>
		public ulong			ulNumBrokenDomLinks;

		// Fields that report index check progress

		/// <summary>
		/// Number of index keys collected so far
		/// </summary>
		public ulong			ulNumKeys;
		/// <summary>
		/// Number of duplicate index keys found so far
		/// </summary>
		public ulong			ulNumDuplicateKeys;
		/// <summary>
		/// Number of index key checked.
		/// </summary>
		public ulong			ulNumKeysExamined;
		/// <summary>
		/// Number of index keys found in index, but not found in document
		/// </summary>
		public ulong			ulNumKeysNotFound;
		/// <summary>
		/// Number of index keys found in a document but missing from the index
		/// </summary>
		public ulong			ulNumDocKeysNotFound;
		/// <summary>
		/// Number of non-corruption conflicts found
		/// </summary>
		public ulong			ulNumConflicts;
	}

	/// <summary>
	/// This interface allows an application to receive information during a
	/// <see cref="DbSystem.dbCheck"/> operation.  The implementation may
	/// do anything it wants with the information, such as write
	/// it to a log file or display it on the screen.
	/// </summary>
	public interface DbCheckStatus 
	{

		/// <summary>
		/// Called by <see cref="DbSystem.dbCheck"/> to report progress of the
		/// check operation.
		/// </summary>
		/// <param name="progressInfo">
		/// This object contains information about the progress of the
		/// check operation.
		/// </param>
		/// <returns>
		/// If the implementation object returns anything but RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbCheck"/> operation will abort and throw an
		/// <see cref="XFlaimException"/>
		/// </returns>
		RCODE reportProgress(
			XFLM_PROGRESS_CHECK_INFO	progressInfo);

		/// <summary>
		/// Called by <see cref="DbSystem.dbCheck"/> to report corruptions when
		/// they are found.
		/// </summary>
		/// <param name="corruptInfo">
		/// Information about the corruption is contained in this object.
		/// </param>
		/// <returns>
		/// If the implementation object returns anything but RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbCheck"/> operation will abort and throw an
		/// <see cref="XFlaimException"/>
		/// </returns>
		RCODE reportCheckErr(
			XFLM_CORRUPT_INFO	corruptInfo);
	}
}
