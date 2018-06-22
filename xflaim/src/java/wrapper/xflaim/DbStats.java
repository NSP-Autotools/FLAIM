//------------------------------------------------------------------------------
// Desc:	DbStats Class
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
 * The DbStats class provides members that give database statistics
 */
public class DbStats
{
	public String					sDbName;
	public RTransStats			readTransStats;
	public UTransStats			updateTransStats;
	public LFileStats[]			lfileStats;
	public BlockIOStats			lfhBlockStats;
	public BlockIOStats			availBlockStats;
	public DiskIOStat				dbHdrWrites;
	public DiskIOStat				logBlockWrites;
	public DiskIOStat				logBlockRestores;
	public DiskIOStat				logBlockReads;
	public int						iLogBlockChkErrs;
	public int						iReadErrors;
	public int						iWriteErrors;
	public LockStats				lockStats;

	private static native void initIDs();
}

