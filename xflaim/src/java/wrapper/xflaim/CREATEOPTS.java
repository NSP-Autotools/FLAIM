//------------------------------------------------------------------------------
// Desc:	Create Options Structure
// Tabs:	3
//
// Copyright (c) 2003-2007 Novell, Inc. All Rights Reserved.
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
 * This class encapsulates the Database create options.
 */
public final class CREATEOPTS
{
	public int 			iBlockSize;
	public int 			iVersionNum;
	public int 			iMinRflFileSize;
	public int 			iMaxRflFileSize;
	public boolean 	bKeepRflFiles;
	public boolean 	bLogAbortedTransToRfl;
	public int 			iDefaultLanguage;
	
	public CREATEOPTS()
	{
		iBlockSize = 4096;
		iVersionNum = 500;
		iMinRflFileSize = (100 * 1024 * 1024);
		iMaxRflFileSize = 0xFFFC0000;
		bKeepRflFiles = false;
		bLogAbortedTransToRfl = false;
		iDefaultLanguage = 0;
	}
	private static native void initIDs();
}
