//------------------------------------------------------------------------------
// Desc:	Db Copy Status
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
	public interface DbCopyStatus 
	{

		/// <summary>
		/// The <see cref="DbSystem.dbCopy"/> operation calls this method periodically to
		/// inform the implementation object about the status of the copy operation.
		/// </summary>
		/// <param name="ulBytesToCopy">
		/// The total number of bytes that the <see cref="DbSystem.dbCopy"/> operation
		/// will ultimately copy.
		/// </param>
		/// <param name="ulBytesCopied">
		/// The number of bytes that the <see cref="DbSystem.dbCopy"/> operation has
		/// copied so far.
		/// </param>
		/// <param name="sSrcFileName">
		/// The name of the file that is currently being copied.  This is only passed
		/// when the source file name is changed.  Otherwise, it will be null.
		/// </param>
		/// <param name="sDestFileName">
		/// The name of the current destination file.  This is only passed when the
		/// destination file name is changed.  Otherwise, it will be null.
		/// </param>
		/// <returns>
		/// If the implementation object returns anything but RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbCopy"/> operation will abort and throw an
		/// <see cref="XFlaimException"/>
		/// </returns>
		RCODE dbCopyStatus(
			ulong			ulBytesToCopy,
			ulong			ulBytesCopied,
			string		sSrcFileName,
			string		sDestFileName);
	}
}
