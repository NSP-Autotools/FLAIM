//------------------------------------------------------------------------------
// Desc:	Backup Client
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
 * This interface defines the client side interface to XFlaim's backup
 * subsystem.   Clients must pass an object that implements this interface
 * into the call to {@link Backup#backup Backup::backup} 
 * See the documentation regarding Backup/Restore operations for more details.
 * @see DefaultBackupClient
 */
public interface BackupClient 
{
	
	/**
	 * Called by XFlaim's backup subsystem when it has a block of data ready
	 * to be written.  It is up to the implementation to decide what to
	 * do with the data (but presumably, it will write the data to disk,
	 * tape or some other storage medium).
	 * @param Buffer An array of bytes containing the data that needs to be
	 * written.
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * backup operation to abort an XFLaimException to be thrown. 
	 */
	public int WriteData(
		byte[]			Buffer);
}
