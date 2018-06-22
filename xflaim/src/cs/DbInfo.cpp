//------------------------------------------------------------------------------
// Desc: Native C routines to support C# DbInfo class
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

#include "xflaim.h"

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DbInfo_Release(
	IF_DbInfo *	pDbInfo)
{
	if (pDbInfo)
	{
		pDbInfo->Release();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMUINT32 XFLAPI xflaim_DbInfo_getNumCollections(
	IF_DbInfo *	pDbInfo)
{
	return( (FLMUINT32)pDbInfo->getNumCollections());
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMUINT32 XFLAPI xflaim_DbInfo_getNumIndexes(
	IF_DbInfo *	pDbInfo)
{
	return( (FLMUINT32)pDbInfo->getNumIndexes());
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMUINT32 XFLAPI xflaim_DbInfo_getNumLogicalFiles(
	IF_DbInfo *	pDbInfo)
{
	return( (FLMUINT32)pDbInfo->getNumLogicalFiles());
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMUINT64 XFLAPI xflaim_DbInfo_getDatabaseSize(
	IF_DbInfo *	pDbInfo)
{
	return( pDbInfo->getFileSize());
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DbInfo_getAvailBlockStats(
	IF_DbInfo *	pDbInfo,
	FLMUINT64 *	pui64BytesUsed,
	FLMUINT32 *	pui32BlockCount,
	FLMINT32 *	pi32LastError,
	FLMUINT32 *	pui32NumErrors)
{
	FLMUINT	uiBlockCount;
	FLMUINT	uiNumErrors;

	pDbInfo->getAvailBlockStats( pui64BytesUsed, &uiBlockCount,
		pi32LastError, &uiNumErrors);
	if (pui32BlockCount)
	{
		*pui32BlockCount = (FLMUINT32)uiBlockCount;
	}
	if (pui32NumErrors)
	{
		*pui32NumErrors = (FLMUINT32)uiNumErrors;
	}
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DbInfo_getLFHBlockStats(
	IF_DbInfo *	pDbInfo,
	FLMUINT64 *	pui64BytesUsed,
	FLMUINT32 *	pui32BlockCount,
	FLMINT32 *	pi32LastError,
	FLMUINT32 *	pui32NumErrors)
{
	FLMUINT	uiBlockCount;
	FLMUINT	uiNumErrors;

	pDbInfo->getLFHBlockStats( pui64BytesUsed, &uiBlockCount,
		pi32LastError, &uiNumErrors);
	if (pui32BlockCount)
	{
		*pui32BlockCount = (FLMUINT32)uiBlockCount;
	}
	if (pui32NumErrors)
	{
		*pui32NumErrors = (FLMUINT32)uiNumErrors;
	}
}


/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DbInfo_getBTreeInfo(
	IF_DbInfo *		pDbInfo,
	FLMUINT32		ui32NthLogicalFile,
	FLMUINT32 *		pui32LfNum,
	FLMINT32 *		pi32LfType,
	FLMUINT32 *		pui32RootBlkAddress,
	FLMUINT32 *		pui32NumLevels)
{
	FLMUINT		uiLfNum;
	eLFileType	eLfType;
	FLMUINT		uiRootBlkAddress;
	FLMUINT		uiNumLevels;

	pDbInfo->getBTreeInfo( (FLMUINT)ui32NthLogicalFile, &uiLfNum,
		&eLfType, &uiRootBlkAddress, &uiNumLevels);
	if (pui32LfNum)
	{
		*pui32LfNum = (FLMUINT32)uiLfNum;
	}
	if (pi32LfType)
	{
		*pi32LfType = (FLMINT32)eLfType;
	}
	if (pui32RootBlkAddress)
	{
		*pui32RootBlkAddress = (FLMUINT32)uiRootBlkAddress;
	}
	if (pui32NumLevels)
	{
		*pui32NumLevels = (FLMUINT32)uiNumLevels;
	}
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DbInfo_getBTreeBlockStats(
	IF_DbInfo *		pDbInfo,
	FLMUINT32		ui32NthLogicalFile,
	FLMUINT32		ui32Level,
	FLMUINT64 *		pui64KeyCount,
	FLMUINT64 *		pui64BytesUsed,
	FLMUINT64 *		pui64ElementCount,
	FLMUINT64 *		pui64ContElementCount,
	FLMUINT64 *		pui64ContElmBytes,
	FLMUINT32 *		pui32BlockCount,
	FLMINT32 *		pi32LastError,
	FLMUINT32 *		pui32NumErrors)
{
	FLMUINT		uiBlockCount;
	FLMUINT		uiNumErrors;

	pDbInfo->getBTreeBlockStats( (FLMUINT)ui32NthLogicalFile, (FLMUINT)ui32Level, pui64KeyCount,
		pui64BytesUsed, pui64ElementCount, pui64ContElementCount, pui64ContElmBytes,
		&uiBlockCount, pi32LastError, &uiNumErrors);
	if (pui32BlockCount)
	{
		*pui32BlockCount = (FLMUINT32)uiBlockCount;
	}
	if (pui32NumErrors)
	{
		*pui32NumErrors = (FLMUINT32)uiNumErrors;
	}
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DbInfo_getDbHdr(
	IF_DbInfo *		pDbInfo,
	XFLM_DB_HDR *	pDbHdr)
{
	f_memcpy( pDbHdr, pDbInfo->getDbHdr(), sizeof( XFLM_DB_HDR));
}
