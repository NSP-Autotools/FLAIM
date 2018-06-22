//-------------------------------------------------------------------------
// Desc:	Miscellaneous functions.
// Tabs:	3
//
// Copyright (c) 1995-2001, 2003-2007 Novell, Inc. All Rights Reserved.
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

/****************************************************************************
Desc:	Returns TRUE if the passed in RCODE indicates that a corruption
		has occured in a FLAIM database file.
****************************************************************************/
FLMEXP FLMBOOL FLMAPI FlmErrorIsFileCorrupt(
	RCODE			rc)
{
	FLMBOOL		b = FALSE;

	switch( rc)
	{
		case FERR_BTREE_ERROR :
		case FERR_DATA_ERROR :
		case FERR_DD_ERROR :
		case FERR_NOT_FLAIM :
		case FERR_PCODE_ERROR :
		case FERR_BLOCK_CHECKSUM :
		case FERR_INCOMPLETE_LOG :
		case FERR_KEY_NOT_FOUND :
		case FERR_NO_REC_FOR_KEY:
			b = TRUE;
			break;
		default :
			break;
	}

	return( b);
}

/****************************************************************************
Desc:	Returns specific information about the most recent error that
		occured within FLAIM.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmGetDiagInfo(
	HFDB				hDb,
	eDiagInfoType	eDiagCode,
	void *			pvDiagInfo)
{
	RCODE		rc = FERR_OK;
	FDB *		pDb;

	if ((pDb = (FDB *)hDb) == NULL)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}
	fdbUseCheck( pDb);

	/* Now, copy over the data into the users variable */
	switch( eDiagCode)
	{
		case FLM_GET_DIAG_INDEX_NUM :
			if (!(pDb->Diag.uiInfoFlags & FLM_DIAG_INDEX_NUM))
			{
				rc = RC_SET( FERR_NOT_FOUND);
				goto Exit;
			}
			else
			{
				*((FLMUINT *)pvDiagInfo) = pDb->Diag.uiIndexNum;
			}
			break;

		case FLM_GET_DIAG_DRN :
			if (!(pDb->Diag.uiInfoFlags & FLM_DIAG_DRN))
			{
				rc = RC_SET( FERR_NOT_FOUND);
				goto Exit;
			}
			else
			{
				*((FLMUINT *)pvDiagInfo) = pDb->Diag.uiDrn;
			}
			break;

		case FLM_GET_DIAG_FIELD_NUM :
			if (!(pDb->Diag.uiInfoFlags & FLM_DIAG_FIELD_NUM))
			{
				rc = RC_SET( FERR_NOT_FOUND);
				goto Exit;
			}
			else
			{
				*((FLMUINT *)pvDiagInfo) = pDb->Diag.uiFieldNum;
			}
			break;

		case FLM_GET_DIAG_FIELD_TYPE :
			if (!(pDb->Diag.uiInfoFlags & FLM_DIAG_FIELD_TYPE))
			{
				rc = RC_SET( FERR_NOT_FOUND);
				goto Exit;
			}
			else
			{
				*((FLMUINT *)pvDiagInfo) = pDb->Diag.uiFieldType;
			}
			break;

		case FLM_GET_DIAG_ENC_ID :
			if (!(pDb->Diag.uiInfoFlags & FLM_DIAG_ENC_ID))
			{
				rc = RC_SET( FERR_NOT_FOUND);
				goto Exit;
			}
			else
			{
				*((FLMUINT *)pvDiagInfo) = pDb->Diag.uiEncId;
			}
			break;
		default:
			flmAssert( 0);
			rc = RC_SET( FERR_NOT_FOUND);
			goto Exit;

	}

Exit:
	if( pDb)
	{
		fdbUnuse( pDb);
	}
	return( rc);
}

/****************************************************************************
Desc:	Get the total bytes represented by a particular block address.
****************************************************************************/
FLMUINT64 FSGetSizeInBytes(
	FLMUINT		uiMaxFileSize,
	FLMUINT		uiBlkAddress)
{
	FLMUINT		uiFileNum;
	FLMUINT		uiFileOffset;
	FLMUINT64	ui64Size;

	uiFileNum = FSGetFileNumber( uiBlkAddress);
	uiFileOffset = FSGetFileOffset( uiBlkAddress);
	
	if( uiFileNum > 1)
	{
		ui64Size = (FLMUINT64)(((FLMUINT64)uiFileNum - (FLMUINT64)1) *
											(FLMUINT64)uiMaxFileSize +
											(FLMUINT64)uiFileOffset);
	}
	else
	{
		ui64Size = (FLMUINT64)uiFileOffset;
	}
	
	return( ui64Size);
}

/****************************************************************************
Desc:	Converts a UNICODE string consisting of 7-bit ASCII characters to
		a 7-bit ASCII string.  The conversion is done in place, so that
		only one buffer is needed
*****************************************************************************/
RCODE flmUnicodeToAscii(
	FLMUNICODE *	puzString) // Unicode in, Ascii out
{
	FLMBYTE *	pucDest;

	pucDest = (FLMBYTE *)puzString;
	while( *puzString)
	{
		if( *puzString > 0x007F)
		{
			*pucDest = 0xFF;
		}
		else
		{
			*pucDest = (FLMBYTE)*puzString;
		}
		pucDest++;
		puzString++;
	}
	*pucDest = '\0';

	return( FERR_OK);
}
