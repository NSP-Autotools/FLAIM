//------------------------------------------------------------------------------
// Desc:	CheckpointInfo Class
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
 * The CheckpointInfo class provides members that give an index's current state.
 */
public class CheckpointInfo 
{
	public boolean		bRunning;
	public int			iRunningTime;
	public boolean		bForcingCheckpoint;
	public int			iForceCheckpointRunningTime;
	public int			iForceCheckpointReason;
	public boolean		bWritingDataBlocks;
	public int			iLogBlocksWritten;
	public int			iDataBlocksWritten;
	public int			iDirtyCacheBytes;
	public int			iBlockSize;
	public int			iWaitTruncateTime;
	
	private static native void initIDs();
}

