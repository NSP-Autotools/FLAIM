//------------------------------------------------------------------------------
// Desc:	This file contains the flmGetHdrInfo routine -- a routine which
// 		reads the header information from a FLAIM database, verifies it
// 		and returns the header information in a structure.
// Tabs:	3
//
// Copyright (c) 1991-2007 Novell, Inc. All Rights Reserved.
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

#include "flaimsys.h"

/***************************************************************************
Desc:	This routine reads the header information in a FLAIM database,
		verifies the password, and returns the file header and log
		header information.
*****************************************************************************/
RCODE flmGetHdrInfo(
	F_SuperFileHdl *	pSFileHdl,
	SFLM_DB_HDR *		pDbHdr,
	FLMUINT32 *			pui32CalcCRC)
{
	RCODE				rc = NE_SFLM_OK;
	IF_FileHdl *	pCFileHdl;

	if( RC_BAD( rc = pSFileHdl->getFileHdl( 0, FALSE, &pCFileHdl)))
	{
		goto Exit;
	}

	rc = flmReadAndVerifyHdrInfo( NULL, pCFileHdl, pDbHdr, pui32CalcCRC);

Exit:

	return( rc);
}
