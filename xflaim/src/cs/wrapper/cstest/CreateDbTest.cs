//------------------------------------------------------------------------------
// Desc:	Create Database Test
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
using System.IO;
using System.Runtime.InteropServices;
using xflaim;

namespace cstest
{

	//--------------------------------------------------------------------------
	// Create database test
	//--------------------------------------------------------------------------

	public class CreateDbTest : Tester
	{
		public bool createDbTest(
			string	sDbName,
			DbSystem	dbSystem)
		{
			Db			db = null;
			RCODE	rc;

			beginTest( "Create Database Test (" + sDbName + ")");

			for (;;)
			{
				rc = RCODE.NE_XFLM_OK;
				try
				{
					XFLM_CREATE_OPTS	createOpts = new XFLM_CREATE_OPTS();

					createOpts.uiBlockSize = 8192;
					createOpts.uiVersionNum = (uint)DBVersions.XFLM_CURRENT_VERSION_NUM;
					createOpts.uiMinRflFileSize = 2000000;
					createOpts.uiMaxRflFileSize = 20000000;
					createOpts.bKeepRflFiles = 1;
					createOpts.bLogAbortedTransToRfl = 1;
					createOpts.eDefaultLanguage = Languages.FLM_DE_LANG;
					db = dbSystem.dbCreate( sDbName, null, null, null, null, createOpts);
				}
				catch (XFlaimException ex)
				{
					rc = ex.getRCode();

					if (rc != RCODE.NE_XFLM_FILE_EXISTS)
					{
						endTest( false, ex, "creating database");
						return( false);
					}
				}
				if (rc == RCODE.NE_XFLM_OK)
				{
					break;
				}

				// rc better be NE_XFLM_FILE_EXISTS - try to delete the file

				try
				{
					dbSystem.dbRemove( sDbName, null, null, true);
				}
				catch (XFlaimException ex)
				{
					endTest( false, ex, "removing database");
					return( false);
				}
			}
			if (db != null)
			{
				db.close();
				db = null;
			}
			endTest( false, true);
			return( true);
		}
	}
}
