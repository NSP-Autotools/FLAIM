//------------------------------------------------------------------------------
// Desc:	Db Rename Status
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
	/// This interface allows XFlaim to periodically pass information back to the
	/// client about the status of an ongoing database copy operation.  The
	/// implementor may do anything it wants with the information, such as write
	/// it to a log file or display it on the screen.
	/// </summary>
	public interface DbRenameStatus 
	{

		/// <summary>
		/// The <see cref="DbSystem.dbRename"/> operation calls this method periodically to
		/// inform the implementation object about the status of the rename operation.
		/// </summary>
		/// <param name="sSrcFileName">
		/// The name of the file that is currently being renamed.
		/// </param>
		/// <param name="sDestFileName">
		/// The name the source file is being renamed to.
		/// </param>
		/// <returns>
		/// If the implementation object returns anything but RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRename"/> operation will abort and throw an
		/// <see cref="XFlaimException"/>
		/// </returns>
		RCODE dbRenameStatus(
			string		sSrcFileName,
			string		sDestFileName);
	}
}
