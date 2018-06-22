//------------------------------------------------------------------------------
// Desc:	Restore Client
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
 * This interface defines the client side interface to XFlaim's restore
 * subsystem.   Clients must pass an object that implements this interface
 * into the call to {@link DbSystem#dbRestore DbSystem::dbRestore} 
 * See the documentation regarding Backup/Restore operations for more details.
 * @see DefaultRestoreClient
 */
public interface RestoreClient
{
	/**
	 * 
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort an XFLaimException to be thrown.
	 */
	public int openBackupSet();

	/**
	 * 
	 * @param iFileNum
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort an XFLaimException to be thrown.
	 */
	public int openRflFile(
		int		iFileNum);

	/**
	 * 
	 * @param iFileNum
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort an XFLaimException to be thrown.
	 */
	public int openIncFile(
		int		iFileNum);

	/**
	 * 
	 * @param Buffer
	 * @param BytesRead NOTE: Will have a length of 1  Function
	 * must store the number of valid bytes in Buffer into BytesRead[0].
	 * (This allows us to pass that value back to the calling function.)
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort an XFLaimException to be thrown.
	 */
	public int read(
		byte[]	Buffer,
		int[]		BytesRead);

	/**
	 * 
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort an XFLaimException to be thrown.
	 */
	public int close();

	/**
	 * 
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort an XFLaimException to be thrown.
	 */
	public int abortFile();
}
