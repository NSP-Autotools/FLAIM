//------------------------------------------------------------------------------
// Desc:	Backup Status
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
	/// This interface allows XFlaim's backup subsystem to periodicly pass
	/// information about the status of a backup operation (bytes completed and
	/// bytes remaining) while the operation is running.  The implementor may do
	/// anything it wants with the information, such as using it to update a
	/// progress bar or simply ignoring it.
	/// </summary>
	public interface BackupStatus
	{
	
		/// <summary>
		/// This method is called by the <see cref="Backup.backup"/> method to
		/// report progress of a backup.
		/// </summary>
		/// <param name="ulBytesToDo">
		/// Total bytes that are to be backed up.
		/// </param>
		/// <param name="ulBytesDone">
		/// Bytes written out so far.
		/// </param>
		/// <returns>
		/// Returns an <see cref="RCODE"/>.  Note that returning anything
		/// other than RCODE.NE_XFLM_OK will cause the backup operation to abort.  It
		/// will throw an <see cref="XFlaimException"/>
		/// </returns>
		RCODE backupStatus(
			ulong	ulBytesToDo,
			ulong	ulBytesDone);
	}
}
