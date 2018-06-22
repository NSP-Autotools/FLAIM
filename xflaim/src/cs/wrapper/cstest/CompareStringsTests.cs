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
	// String comparison tests.
	//--------------------------------------------------------------------------
	public class CompareStringTests : Tester
	{

		private bool compareStrTest(
			string			sLeftStr,
			bool				bLeftWild,
			string			sRightStr,
			bool				bRightWild,
			CompareFlags	compareFlags,
			bool				bExpectedEqual,
			DbSystem			dbSystem)
		{
			int	iCmp;

			beginTest( "Compare Strings, Str1: \"" + sLeftStr +
					"\", Str2: \"" + sRightStr + "\"");

			try
			{
				iCmp = dbSystem.compareStrings( sLeftStr, bLeftWild,
										sRightStr, bRightWild, compareFlags,
										Languages.FLM_US_LANG);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling compareStrings");
				return( false);
			}
			if ((bExpectedEqual && iCmp != 0) ||
				 (!bExpectedEqual && iCmp == 0))
			{
				endTest( false, false);
				System.Console.WriteLine( "Expected Equal [{0}] != Result [{1}]",
						bExpectedEqual, iCmp);
				System.Console.WriteLine( "Compare Flags: {0}", compareFlags);
				System.Console.WriteLine( "Left Wild: {0}", bLeftWild);
				System.Console.WriteLine( "Right Wild: {0}", bRightWild);
			}
			endTest( false, true);

			return( true);
		}

		public bool compareStringTests(
			DbSystem	dbSystem)
		{
			if (!compareStrTest( "ABC", false, "abc", false, CompareFlags.FLM_COMP_CASE_INSENSITIVE, true, dbSystem))
			{
				return( false);
			}
			if (!compareStrTest( "ABC", false, "abc", false, 0, false, dbSystem))
			{
				return( false);
			}
			if (!compareStrTest( "ab  cd", false, "ab     cd", false, CompareFlags.FLM_COMP_COMPRESS_WHITESPACE, true, dbSystem))
			{
				return( false);
			}
			if (!compareStrTest( " ab  cd", false, "ab     cd", false, CompareFlags.FLM_COMP_COMPRESS_WHITESPACE, false, dbSystem))
			{
				return( false);
			}
			if (!compareStrTest( " ab  cd", false, "ab     cd", false,
					CompareFlags.FLM_COMP_COMPRESS_WHITESPACE | CompareFlags.FLM_COMP_IGNORE_LEADING_SPACE,
					true, dbSystem))
			{
				return( false);
			}
			if (!compareStrTest( " ab  cd", false, "   ab     cd", false,
				CompareFlags.FLM_COMP_COMPRESS_WHITESPACE | CompareFlags.FLM_COMP_IGNORE_LEADING_SPACE,
				true, dbSystem))
			{
				return( false);
			}
			if (!compareStrTest( " ab  cd ", false, "   ab     cd", false,
				CompareFlags.FLM_COMP_COMPRESS_WHITESPACE | CompareFlags.FLM_COMP_IGNORE_LEADING_SPACE,
				false, dbSystem))
			{
				return( false);
			}
			if (!compareStrTest( " ab  cd ", false, "   ab     cd", false,
				CompareFlags.FLM_COMP_COMPRESS_WHITESPACE |
				CompareFlags.FLM_COMP_IGNORE_LEADING_SPACE |
				CompareFlags.FLM_COMP_IGNORE_TRAILING_SPACE,
				true, dbSystem))
			{
				return( false);
			}
			if (!compareStrTest( " ab  cd ", false, "   ab     cd   ", false,
				CompareFlags.FLM_COMP_COMPRESS_WHITESPACE |
				CompareFlags.FLM_COMP_IGNORE_LEADING_SPACE |
				CompareFlags.FLM_COMP_IGNORE_TRAILING_SPACE,
				true, dbSystem))
			{
				return( false);
			}
			if (!compareStrTest( "801-224-8888", false, "8012248888", false,
				CompareFlags.FLM_COMP_NO_DASHES,
				true, dbSystem))
			{
				return( false);
			}
			if (!compareStrTest( "801_224_8888", false, "801 224 8888", false,
				CompareFlags.FLM_COMP_NO_UNDERSCORES,
				true, dbSystem))
			{
				return( false);
			}
			if (!compareStrTest( "801_224_8888", false, "801   224    8888", false,
				CompareFlags.FLM_COMP_NO_UNDERSCORES | CompareFlags.FLM_COMP_COMPRESS_WHITESPACE,
				true, dbSystem))
			{
				return( false);
			}
			return( true);
		}
	}
}
