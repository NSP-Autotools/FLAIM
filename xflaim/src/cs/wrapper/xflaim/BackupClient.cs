//------------------------------------------------------------------------------
// Desc:	Backup Client
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
	/// This interface defines the client side interface to XFlaim's backup
	/// subsystem.   Clients may pass an object that implements this interface
	/// into the call to <see cref="Backup.backup"/>.  The object determines
	/// where backup data will be saved.
	/// </summary>
	public interface BackupClient 
	{
	
		/// <summary>
		/// This method is called by the <see cref="Backup.backup"/> method to
		/// write out data that needs to be backed up.  It is the responsibility
		/// of this method to write that data out - to disk, tape, etc.
		/// </summary>
		/// <param name="pvData">
		/// Pointer to data that is to be written out.
		/// </param>
		/// <param name="uiDataLen">
		/// Length of data to be written out.
		/// </param>
		/// <returns>
		/// Returns an <see cref="RCODE"/>.  Note that returning anything
		/// other than RCODE.NE_XFLM_OK will cause the backup operation to abort.  It
		/// will throw an <see cref="XFlaimException"/>
		/// </returns>
		RCODE writeData(
			IntPtr	pvData,
			uint		uiDataLen);
	}
}
