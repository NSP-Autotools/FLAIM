//------------------------------------------------------------------------------
// Desc:	Db Copy Status
// Tabs:	3
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
 * This interface alows XFlaim to periodically pass information back to the
 * client about the status of an ongoing database copy operation.  The
 * implementor may do anything it wants with the information, such as write
 * it to a log file or display it on the screen.
 */
public interface DbCopyStatus 
{
	/**
	 * Called periodically to inform the client about the status of the copy
	 * operation.
	 * @param ui64BytesToCopy The total number of bytes that this operation
	 * will copy.
	 * @param ui64BytesCopied The number of bytes that have been copied so far.
	 * @param bNewSrcFile Set to true if the copy operation has started
	 * working on a new file. 
	 * @param pszSrcFileName The name of the file that is currently being copied.
	 * @param pszDestFileName The name of the destination file.
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * copy operation to abort and an XFLaimException to be thrown.
	 */
	int dbCopyStatus(
		long			ui64BytesToCopy,
		long			ui64BytesCopied,
		boolean		bNewSrcFile,
		String		pszSrcFileName,
		String		pszDestFileName);
}
