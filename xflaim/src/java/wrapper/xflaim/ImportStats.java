//------------------------------------------------------------------------------
// Desc:	ImportStats Class
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

package xflaim;

/**
 * The ImportStats class provides members that import statistics.
 */
public class ImportStats 
{
	public int			iLines;
	public int			iChars;
	public int			iAttributes;
	public int			iElements;
	public int			iText;
	public int			iDocuments;
	public int			iErrLineNum;
	public int			iErrLineOffset;
	public int			iErrorType;
	public int			iErrLineFilePos;
	public int			iErrLineBytes;
	public boolean		bUTF8Encoding;
	
	private static native void initIDs();
}

