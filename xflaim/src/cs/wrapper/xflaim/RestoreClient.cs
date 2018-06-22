//------------------------------------------------------------------------------
// Desc:	Restore Client
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
	/// This interface defines the client side interface to XFlaim's restore
	/// subsystem.   Clients may pass an object that implements this interface
	/// into the call to <see cref="DbSystem.dbRestore"/>
	/// See the documentation regarding Backup/Restore operations for more details.																																																 * @see DefaultRestoreClient
	/// </summary>
	public interface RestoreClient
	{
		
		/// <summary>
		/// During a restore operation, this method is called when a backup set is
		/// to be opened.  The implementaiton should do whatever is necessary to
		/// open the backup set (open files, etc.).
		/// </summary>
		/// <returns>
		/// Returns an <see cref="RCODE"/>.  Note that returning anything
		/// except RCODE.NE_XFLM_OK will cause the restore operation to abort.  It
		/// will throw an <see cref="XFlaimException"/>
		/// </returns>
		RCODE openBackupSet();

		/// <summary>
		/// During a restore operation, this method is called when the restore
		/// process expects a roll-forward log file to be opened.  Subsequent
		/// read operations are to return data from the specified RFL file.
		/// If the requested roll-forward log file is not available, this method
		/// should return RCODE.NE_FLM_IO_PATH_NOT_FOUND.
		/// </summary>
		/// <param name="uiFileNum">
		/// Roll-forward log file that is to be opened.
		/// </param>
		/// <returns>
		/// Returns an <see cref="RCODE"/>.  To indicate to the restore operation
		/// that there are no more roll-forward log files, the implementation should
		/// return RCODE.NE_FLM_IO_PATH_NOT_FOUND.  Note that returning anything
		/// else will cause the restore operation to abort.  It
		/// will throw an <see cref="XFlaimException"/>
		/// </returns>
		RCODE openRflFile(
			uint		uiFileNum);

		/// <summary>
		/// During a restore operation, this method is called when the restore
		/// process expects an incremental backup file to be opened.  Subsequent
		/// read operations are to return data from the specified incremental backup file.
		/// If the requested incremental backup file is not available, this method
		/// should return RCODE.NE_FLM_IO_PATH_NOT_FOUND.
		/// </summary>
		/// <param name="uiFileNum">
		/// Incremental backup file that is to be opened.
		/// </param>
		/// <returns>
		/// Returns an <see cref="RCODE"/>.  To indicate to the restore operation
		/// that there are no more incremental backup files, the implementation should
		/// return RCODE.NE_FLM_IO_PATH_NOT_FOUND.  Note that returning anything
		/// else will cause the restore operation to abort.  It
		/// will throw an <see cref="XFlaimException"/>
		/// </returns>
		RCODE openIncFile(
			uint		uiFileNum);

		/// <summary>
		/// The restore operation calls this method when it wants to read data.
		/// Previous calls to <see cref="openBackupSet"/>, <see cref="openRflFile"/>,
		/// or <see cref="openIncFile"/> will have been made to tell the
		/// implementation where data should be read from.
		/// </summary>
		/// <param name="uiBytesRequested">
		/// Number of bytes the restore operation would like to read.
		/// </param>
		/// <param name="pvBuffer">
		/// Data buffer that data is to be returned in.
		/// </param>
		/// <param name="uiBytesRead">
		/// Returns the number of bytes actually read.
		/// </param>
		/// <returns>
		/// Returns an <see cref="RCODE"/>.  To indicate to the restore operation
		/// that there is no more data to be read, the implementation should
		/// return RCODE.NE_FLM_IO_END_OF_FILE.  Note that returning anything
		/// else will cause the restore operation to abort.  It
		/// will throw an <see cref="XFlaimException"/>
		/// </returns>
		RCODE read(
			uint		uiBytesRequested,
			IntPtr	pvBuffer,
			ref uint	uiBytesRead);

		/// <summary>
		/// The restore operation will call this method to close the current
		/// file that is being read from.  The implementation should close
		/// whatever file it is currently reading data from.
		/// </summary>
		/// <returns>
		/// Returns an <see cref="RCODE"/>.  Note that returning anything
		/// other than RCODE.NE_XFLM_OK will cause the restore operation to abort.  It
		/// will throw an <see cref="XFlaimException"/>
		/// </returns>
		RCODE close();

		/// <summary>
		/// The restore operation will call this method to abort the current
		/// file that is being read from.  The implementation should close
		/// whatever file it is currently reading data from.
		/// </summary>
		/// <returns>
		/// Returns an <see cref="RCODE"/>.  Note that returning anything
		/// other than RCODE.NE_XFLM_OK will cause the restore operation to abort.  It
		/// will throw an <see cref="XFlaimException"/>
		/// </returns>
		RCODE abortFile();
	}
}
