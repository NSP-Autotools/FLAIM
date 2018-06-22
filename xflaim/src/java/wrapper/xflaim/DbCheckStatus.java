//------------------------------------------------------------------------------
// Desc:	Db Check Status
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
 * This interface alows XFlaim to periodically pass information back to the
 * client about the status of an ongoing database check operation.  The
 * implementor may do anything it wants with the information, such as write
 * it to a log file or display it on the screen.  Additionally, it allows
 * the implementor to chose, on a case-by-case basis, whether to attempt
 * to fix problems manually, request XFLaim attempt to fix them or
 * ignore them alltogether.
 */
public interface DbCheckStatus 
{
	/**
	 * Called periodically by XFlaim to inform the client of the status
	 * of an ongoing database check operation.
	 * @param ProgCheck A class who's public data members contain
	 * information about what exactly has been checked so far and 
	 * what problems have been found.
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * check operation to abort and an XFLaimException to be thrown.
	 * @see xflaim.CHECKINFO
	 */
	int reportProgress(
		CHECKINFO		ProgCheck);

	/**
	 * Called by XFlaim when an error has been detected during a database
	 * check.
	 * @param CorruptInfo A class who's public data members contain
	 * information describing the nature of the currption.
	 * @param bFix This is an array containing a single element.  If the
	 * client writes a true into that element, then XFlaim will attempt
	 * to fix the corruption.
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * check operation to abort and an XFLaimException to be thrown.
	 * @see xflaim.CORRUPTINFO
	 */
	int reportCheckErr(
		CORRUPTINFO			CorruptInfo,
		boolean[]			bFix);
}
