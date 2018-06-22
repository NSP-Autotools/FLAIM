//------------------------------------------------------------------------------
// Desc:	This modules contains the routines which allow searching for
//			a key or a node.
// Tabs:	3
//
// Copyright (c) 1992-2007 Novell, Inc. All Rights Reserved.
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

#include "view.h"

/********************************************************************
Desc:	Have user enter a key to search for
*********************************************************************/
FLMBOOL ViewGetKey( void)
{
	FLMBOOL			bOk = FALSE;
//	FlmRecord *		pKey = NULL;
//	void *			pvFld;
//	char				szPrompt [80];
	FLMUINT32		ui32Num;
	FLMBOOL			bValEntered;
//	FLMUINT			uiLen;
	char				szTempBuf [80];
//	FLMUINT			uiNumFields;
//	FLMUINT			uiLoop;
//	RCODE				rc;
//	FLMBYTE			szFieldName [80];
//	FLMBYTE			szFieldType [16];
//	FLMBOOL			bKeyEntered = FALSE;
	F_LF_HDR			lfHdr;
	FLMUINT			uiFileOffset;
	LFILE *			pLFile = NULL;
//	IXD_p				pIxd;
//	ICD *				pIcd;
	FLMUINT			uiRootBlkAddress;
//	FLMBOOL			bTruncated;

	if (!gv_bViewHdrRead)
	{
		ViewReadHdr();
	}

	// See if we can get dictionary information.

	(void)ViewGetDictInfo();

	// See if we have a valid logical file

	if (gv_uiViewSearchLfNum == XFLM_DATA_COLLECTION ||
		 gv_uiViewSearchLfNum == XFLM_DICT_COLLECTION ||
		 isDictCollection( gv_uiViewSearchLfNum) ||
		 gv_uiViewSearchLfNum == XFLM_DICT_NUMBER_INDEX ||
		 gv_uiViewSearchLfNum == XFLM_DICT_NAME_INDEX ||
		 pLFile)
	{
		FLMUINT	uiLfType;

		if (gv_uiViewSearchLfNum == XFLM_DATA_COLLECTION ||
			 gv_uiViewSearchLfNum == XFLM_DICT_COLLECTION ||
			 isDictCollection( gv_uiViewSearchLfNum))
		{
			uiLfType = XFLM_LF_COLLECTION;
		}
		else if (gv_uiViewSearchLfNum == XFLM_DICT_NUMBER_INDEX ||
					gv_uiViewSearchLfNum == XFLM_DICT_NAME_INDEX)
		{
			uiLfType = XFLM_LF_INDEX;
		}
		else
		{
			uiLfType = gv_uiViewSearchLfType;
		}

		// Get the LFH information for the logical file

		if (!ViewGetLFH( gv_uiViewSearchLfNum, (eLFileType)uiLfType,
								&lfHdr, &uiFileOffset))
		{
			ViewShowError( "Could not get LFH for logical file");
			goto Exit;
		}
		uiRootBlkAddress = (FLMUINT)lfHdr.ui32RootBlkAddr;

		if (uiRootBlkAddress == 0)
		{
			ViewShowError( "Logical file is empty");
			goto Exit;
		}
	}
	else
	{
		ViewShowError( "Logical file not defined");
		goto Exit;
	}

	if (gv_uiViewSearchLfType == XFLM_LF_COLLECTION)
	{
		if (gv_uiViewSearchLfNum == XFLM_DICT_COLLECTION)
		{
			f_strcpy( szTempBuf, "Enter Dictionary Node Number: ");
		}
		else if (gv_uiViewSearchLfNum == XFLM_DATA_COLLECTION)
		{
			f_strcpy( szTempBuf, "Enter Data Collection Node Number: ");
		}
		else
		{
			f_sprintf( szTempBuf,
				"Enter Node Number For Collection %u: ",
				(unsigned)gv_uiViewSearchLfNum);
		}
		if (!ViewGetNum( szTempBuf, &ui32Num, FALSE, 4,
										 0xFFFFFFFF, &bValEntered) ||
			 !bValEntered)
		{
			goto Exit;
		}
		f_UINT32ToBigEndian( ui32Num, gv_ucViewSearchKey);
		gv_uiViewSearchKeyLen = 4;
		bOk = TRUE;
		goto Exit;
	}

	// At this point, we are dealing with an index.

Exit:

	return( bOk);
}

/********************************************************************
Desc:	Perform a search
*********************************************************************/
void ViewSearch( void)
{
	F_LF_HDR	lfHdr;
	FLMUINT	uiFileOffset;
	FLMUINT	uiRootBlkAddress;
	BLK_EXP	BlkExp;

	if (!gv_bViewHdrRead)
	{
		ViewReadHdr();
	}

	for (;;)
	{
		gv_bViewPoppingStack = FALSE;

		// Get the LFH information for the logical file

		if (!ViewGetLFH( gv_uiViewSearchLfNum, (eLFileType)gv_uiViewSearchLfType,
								&lfHdr, &uiFileOffset))
		{
			ViewShowError( "Could not get LFH for logical file");
			return;
		}
		uiRootBlkAddress = (FLMUINT)lfHdr.ui32RootBlkAddr;
		if (uiRootBlkAddress == 0)
		{
			ViewShowError( "Logical file is empty");
			return;
		}

		BlkExp.uiLevel = 0xFF;
		BlkExp.uiType = 0xFF;
		BlkExp.uiNextAddr = BlkExp.uiPrevAddr = 0;
		BlkExp.uiLfNum = gv_uiViewSearchLfNum;
		gv_bViewEnabled = FALSE;
		gv_bViewSearching = TRUE;
		ViewBlocks( uiRootBlkAddress, uiRootBlkAddress, &BlkExp);

		// Reset Search flag before returning so everything will be back to
		// normal.

		gv_bViewSearching = FALSE;

		// If the ViewBlocks did not set up for another search, we are
		// done, otherwise keep-a-goin

		if (!gv_bViewPoppingStack)
		{
			break;
		}
	}
}
