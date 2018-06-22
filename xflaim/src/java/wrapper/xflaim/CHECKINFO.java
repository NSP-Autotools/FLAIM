//------------------------------------------------------------------------------
// Desc:	Progress Check Info Structure
// Tabs:	3
//
// Copyright (c) 2004-2007 Novell, Inc. All Rights Reserved.
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

package xflaim;

/**
 * This class contains data about the status of an ongoing database
 * check operation.  It is passed to the
 * <code>DbCheckStatus.reportProgress</code>. 
 */
public final class CHECKINFO
{
	public int		iCheckPhase;
	public boolean	bStartFlag;
	public long		lFileSize;
	public int		iNumLFs;
	public int		iCurrLF;
	public int		iLfNumber;					// Logical File Pass
	public int		iLfType;
	public long		lBytesExamined;
	public int		iNumProblemsFixed;		// Number of corruptions repaired
	public long		lNumDomNodes;				// in the current Lf
	public long		lNumDomLinksVerified;	// in the current Lf
	public long		lNumBrokenDomLinks;		// in the current Lf
	
	// Index check progress

	public long		lNumKeys;					// Number of keys in the result set
	public long		lNumDuplicateKeys;		// Number of duplicate keys generated
	public long		lNumKeysExamined;			// Number of keys checked
	public long		lNumKeysNotFound;			// Extra keys found in indexes
	public long		lNumRecKeysNotFound;		// Keys missing from indexes
	public long		lNumNonUniqueKeys;		// Non-unique keys in indexes
	public long		lNumConflicts;				// # of non-corruption conflicts
	public long		lNumRSUnits;				// Number of rset sort items
	public long		lNumRSUnitsDone;			// Number of rset items sorted

	public static class CheckPhaseCodes
	{
		public static final int		CHECK_GET_DICT_INFO	= 1;
		public static final int		CHECK_B_TREE		= 2;
		public static final int		CHECK_AVAIL_BLOCKS	= 3;
		public static final int		CHECK_RS_SORT		= 4;
		public static final int		CHECK_DOM_LINKS		= 5;
	}
}
