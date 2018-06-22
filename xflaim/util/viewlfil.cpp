//------------------------------------------------------------------------------
// Desc:	This file contains the routines which display the logical file
//			blocks in a FLAIM database.
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

FSTATIC FLMBOOL ViewOutputLFH(
	FLMUINT		uiCol,
	FLMUINT *	puiRow,
	F_LF_HDR *	pLfHdr,
	FLMUINT		uiLfCount,
	FLMUINT		uiFileOffset);

FSTATIC FLMBOOL ViewSetupLogicalFileMenu(
	FLMUINT		uiLfNum,
	FLMUINT		uiLfType,
	F_LF_HDR *	pLfHdr);

/***************************************************************************
Desc:	This routine formats a logical file type into an ASCII buffer.
*****************************************************************************/
void FormatLFType(
	char *		pszDestBuf,
	FLMUINT		uiLfType)
{
	switch (uiLfType)
	{
		case XFLM_LF_COLLECTION:
			f_strcpy( pszDestBuf, COLLECTION_STRING);
			break;
		case XFLM_LF_INDEX:
			f_strcpy( pszDestBuf, INDEX_STRING);
			break;
		case XFLM_LF_INVALID:
			f_strcpy( pszDestBuf, "Deleted");
			break;
		default:
			f_sprintf( pszDestBuf, "Unknown: %u", (unsigned)uiLfType);
			break;
	}
}

/***************************************************************************
Desc:	This routine outputs the information in a single LFH - for a
		single logical file - FLAIM 1.5 and above.
*****************************************************************************/
FSTATIC FLMBOOL ViewOutputLFH(
	FLMUINT			uiCol,
	FLMUINT *		puiRow,
	F_LF_HDR *		pLfHdr,
	FLMUINT			uiLfCount,
	FLMUINT			uiFileOffset)
{
	FLMBOOL			bOk = FALSE;
	FLMUINT			uiLabelWidth = 35;
	FLMUINT			uiRow = *puiRow;
	FLMUINT			uiBlkAddress;
	char				szTempBuf [80];
	eColorType		uiBackColor = FLM_BLACK;
	eColorType		uiForeColor = FLM_LIGHTGRAY;
	eColorType		uiUnselectBackColor = FLM_BLACK;
	eColorType		uiUnselectForeColor = FLM_WHITE;
	eColorType		uiSelectBackColor = FLM_BLUE;
	eColorType		uiSelectForeColor = FLM_WHITE;
	FLMUINT			uiOption;
	FLMUINT			uiLfNum;

	// Output Logical File Name

	uiLfNum = (FLMUINT)pLfHdr->ui32LfNumber;
	if (pLfHdr->ui32LfType == XFLM_LF_COLLECTION)
	{
		switch (uiLfNum)
		{
			case XFLM_DATA_COLLECTION:
				f_strcpy( szTempBuf, "DATA_COLLECTION");
				break;
			case XFLM_DICT_COLLECTION:
				f_strcpy( szTempBuf, "DICTIONARY_COLLECTION");
				break;
			default:
				f_sprintf( (char *)szTempBuf, "COLLECTION_%u", (unsigned)uiLfNum);
				break;
		}
	}
	else if (pLfHdr->ui32LfType == XFLM_LF_INDEX)
	{
		switch (uiLfNum)
		{
			case XFLM_DICT_NUMBER_INDEX:
				f_strcpy( szTempBuf, "DICT_NUMBER_IX");
				break;
			case XFLM_DICT_NAME_INDEX:
				f_strcpy( szTempBuf, "DICT_NAME_IX");
				break;
			default:
				f_sprintf( (char *)szTempBuf, "INDEX_%u", (unsigned)uiLfNum);
				break;
		}
	}
	else
	{
		f_sprintf( (char *)szTempBuf, "UNKNOWN_TYPE[%u]_%u",
			(unsigned)pLfHdr->ui32LfType, (unsigned)uiLfNum);
	}

	if (!ViewAddMenuItem( LBL_LOGICAL_FILE_NAME, uiLabelWidth,
				VAL_IS_TEXT_PTR,
				(FLMUINT64)((FLMUINT)(&szTempBuf [0])), 0,
				0, VIEW_INVALID_FILE_OFFSET, (FLMUINT)f_strlen( szTempBuf),
				MOD_DISABLED,
				uiCol, uiRow++, 0, FLM_GREEN, FLM_WHITE,
				FLM_GREEN, FLM_WHITE))
	{
		goto Exit;
	}

	// Adjust column and label width so the rest is indented

	uiCol += 2;
	uiLabelWidth -= 2;

	// Output Logical File Number

	if (!ViewAddMenuItem( LBL_LOGICAL_FILE_NUMBER, uiLabelWidth,
			VAL_IS_NUMBER | DISP_DECIMAL,
			(FLMUINT64)uiLfNum, 0,
			0, uiFileOffset + F_LF_HDR_ui32LfNumber_OFFSET, 0,
			MOD_FLMUINT32 | MOD_DECIMAL | MOD_NATIVE,
			uiCol, uiRow++, 0,
			uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Output Logical File Type

	FormatLFType( szTempBuf, (FLMUINT)pLfHdr->ui32LfType);
	if (!ViewAddMenuItem( LBL_LOGICAL_FILE_TYPE, uiLabelWidth,
			VAL_IS_TEXT_PTR,
			(FLMUINT64)((FLMUINT)(&szTempBuf [0])), 0,
			0, uiFileOffset + F_LF_HDR_ui32LfType_OFFSET, 0,
			MOD_FLMUINT32 | MOD_HEX | MOD_NATIVE,
			uiCol, uiRow++, 0,
			uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Output the root block address

	if ((uiBlkAddress = (FLMUINT)pLfHdr->ui32RootBlkAddr) == 0)
	{
		uiOption = 0;
	}
	else
	{
		uiOption = LFH_OPTION_ROOT_BLOCK | uiLfCount;
	}
	if (!ViewAddMenuItem( LBL_ROOT_BLOCK_ADDRESS, uiLabelWidth,
			VAL_IS_NUMBER | DISP_HEX_DECIMAL,
			(FLMUINT64)uiBlkAddress, 0,
			0, uiFileOffset + F_LF_HDR_ui32RootBlkAddr_OFFSET, 0,
			MOD_FLMUINT32 | MOD_HEX | MOD_NATIVE,
			uiCol, uiRow++, uiOption,
			(!uiOption ? uiBackColor : uiUnselectBackColor),
			(!uiOption ? uiForeColor : uiUnselectForeColor),
			(!uiOption ? uiBackColor : uiSelectBackColor),
			(!uiOption ? uiForeColor : uiSelectForeColor)))
	{
		goto Exit;
	}

	// Output the next node id

	if (!ViewAddMenuItem( LBL_NEXT_NODE_ID, uiLabelWidth,
			VAL_IS_NUMBER | DISP_DECIMAL,
			pLfHdr->ui64NextNodeId, 0,
			0, uiFileOffset + F_LF_HDR_ui64NextNodeId_OFFSET, 0,
			MOD_FLMUINT32 | MOD_DECIMAL | MOD_NATIVE,
			uiCol, uiRow++, 0,
			uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Output the next node id

	if (!ViewAddMenuItem( LBL_ENCRYPTION_ID, uiLabelWidth,
			VAL_IS_NUMBER | DISP_DECIMAL,
			pLfHdr->ui32EncId, 0,
			0, uiFileOffset + F_LF_HDR_ui64NextNodeId_OFFSET, 0,
			MOD_FLMUINT32 | MOD_DECIMAL | MOD_NATIVE,
			uiCol, uiRow++, 0,
			uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	*puiRow = uiRow + 1;
	bOk = TRUE;

Exit:

	return( bOk);
}

/***************************************************************************
Desc:	This routine sets up a menu for displaying a single logical file.
*****************************************************************************/
FSTATIC FLMBOOL ViewSetupLogicalFileMenu(
	FLMUINT		uiLfNum,
	FLMUINT		uiLfType,
	F_LF_HDR *	pLfHdr
	)
{
	FLMBOOL	bOk = FALSE;
	FLMUINT	uiRow;
	FLMUINT	uiCol;
	char		szLfName [40];
	FLMUINT	uiFileOffset;

	// Retrieve the information for the logical file

	if (!ViewGetLFName( szLfName, uiLfNum, 
		(eLFileType)uiLfType, pLfHdr, &uiFileOffset))
	{
		ViewShowError( "Could not retrieve LFH information");
		goto Exit;
	}

	ViewMenuInit( "Logical File");
	uiRow = 3;
	uiCol = 5;

	// Output the items in the pLfHdr

	if (!ViewOutputLFH( uiCol, &uiRow, pLfHdr, 0, uiFileOffset))
	{
		goto Exit;
	}

	bOk = TRUE;

Exit:

	return( bOk);
}

/***************************************************************************
Desc:	This routine displays a single logical file and allows the user
		to press menu keys while displaying the information.
*****************************************************************************/
void ViewLogicalFile(
	FLMUINT		uiLfNum,
	FLMUINT		uiLfType
	)
{
	FLMUINT		uiOption;
	VIEW_INFO	SaveView;
	FLMBOOL		bRepaint = TRUE;
	F_LF_HDR		lfHdr;
	BLK_EXP		BlkExp2;
	FLMUINT		uiBlkAddress2;

	// Loop getting commands until the hit the exit key

	ViewReset( &SaveView);
	while (!gv_bViewPoppingStack)
	{
		if (bRepaint)
		{
			if (!ViewSetupLogicalFileMenu( uiLfNum, uiLfType, &lfHdr))
			{
				goto Exit;
			}
		}
		bRepaint = TRUE;
		uiOption = ViewGetMenuOption();
		switch (uiOption)
		{
			case ESCAPE_OPTION:
				goto Exit;
			case SEARCH_OPTION:
				{
					VIEW_MENU_ITEM_p	pMenuItem = gv_pViewMenuCurrItem;

					// Determine which logical file, if any we are pointing at

					while (pMenuItem &&
							 pMenuItem->iLabelIndex != LBL_LOGICAL_FILE_NAME)
					{
						pMenuItem = pMenuItem->pPrevItem;
					}
					if (pMenuItem)
					{
						while (pMenuItem &&
								 pMenuItem->iLabelIndex != LBL_LOGICAL_FILE_NUMBER)
						{
							pMenuItem = pMenuItem->pNextItem;
						}
					}
					if (pMenuItem)
					{
						FLMBYTE *	pszTmp;

						gv_uiViewSearchLfNum = (FLMUINT)pMenuItem->ui64Value;
						while (pMenuItem &&
								 pMenuItem->iLabelIndex != LBL_LOGICAL_FILE_TYPE)
						{
							pMenuItem = pMenuItem->pNextItem;
						}
						pszTmp = (FLMBYTE *)((FLMUINT)pMenuItem->ui64Value);
						if (f_stricmp( (const char *)pszTmp, COLLECTION_STRING) == 0)
						{
							gv_uiViewSearchLfType = XFLM_LF_COLLECTION;
						}
						else
						{
							gv_uiViewSearchLfType = XFLM_LF_INDEX;
						}
						if (ViewGetKey())
						{
							gv_bViewPoppingStack = TRUE;
						}
					}
					else
					{
						ViewShowError(
							"Position cursor to a logical file before searching");
					}
				}
				break;
			default:
				if (uiOption & LFH_OPTION_ROOT_BLOCK)
				{
					BlkExp2.uiLevel = 0xFF;
					BlkExp2.uiType = 0xFF;
					uiBlkAddress2 = (FLMUINT)lfHdr.ui32RootBlkAddr;
					BlkExp2.uiNextAddr = BlkExp2.uiPrevAddr = 0xFFFFFFFF;
					BlkExp2.uiLfNum = (FLMUINT)lfHdr.ui32LfNumber;
					ViewBlocks( uiBlkAddress2, uiBlkAddress2, &BlkExp2);
				}
				else if (uiOption & LOGICAL_INDEX_OPTION)
				{
					ViewLogicalFile( (FLMUINT)(uiOption & (~(LOGICAL_INDEX_OPTION))),
											XFLM_LF_INDEX);
				}
				else if (uiOption & LOGICAL_CONTAINER_OPTION)
				{
					ViewLogicalFile( (FLMUINT)(uiOption & (~(LOGICAL_CONTAINER_OPTION))),
											XFLM_LF_COLLECTION);
				}
				else
				{
					bRepaint = FALSE;
				}
				break;
		}
	}

Exit:

	ViewRestore( &SaveView);
}

/***************************************************************************
Desc:	This routine displays ALL of the logical files in an LFH block
		in the database.
*****************************************************************************/
FLMBOOL ViewLFHBlk(
	FLMUINT			uiReadAddress,
	FLMUINT			uiBlkAddress,
	F_BLK_HDR **	ppBlkHdr,
	BLK_EXP_p		pBlkExp
	)
{
	FLMBOOL		bOk = FALSE;
	FLMUINT		uiRow;
	FLMUINT		uiCol;
	FLMUINT		uiEndOfBlock;
	FLMUINT		uiPos;
	F_LF_HDR *	pLfHdr;
	F_BLK_HDR *	pBlkHdr;
	FLMUINT		uiLfCount = 0;
	FLMUINT32	ui32CalcCRC;
	FLMUINT32	ui32BlkCRC;
	FLMUINT		uiBytesRead;

	// Read the block into memory

	if (!ViewBlkRead( uiReadAddress, ppBlkHdr, TRUE,
							(FLMUINT)gv_ViewDbHdr.ui16BlockSize,
							&ui32CalcCRC, &ui32BlkCRC,
							&uiBytesRead, TRUE))
	{
		goto Exit;
	}
	pBlkHdr = *ppBlkHdr;
	uiPos = SIZEOF_STD_BLK_HDR;
	if (uiBytesRead <= SIZEOF_STD_BLK_HDR)
	{
		uiEndOfBlock = SIZEOF_STD_BLK_HDR;
	}
	else
	{
		uiEndOfBlock = blkGetEnd( (FLMUINT)gv_ViewDbHdr.ui16BlockSize,
								SIZEOF_STD_BLK_HDR, pBlkHdr);
		if (uiEndOfBlock > uiBytesRead)
		{
			uiEndOfBlock = uiBytesRead;
		}
	}

	ViewMenuInit( "LFH Block");

	// Output the block header first

	uiRow = 0;
	uiCol = 5;
	pBlkExp->uiType = BT_LFH_BLK;
	pBlkExp->uiLfNum = 0;
	pBlkExp->uiBlkAddr = uiBlkAddress;
	pBlkExp->uiLevel = 0xFF;
	if (!ViewOutBlkHdr( uiCol, &uiRow, pBlkHdr, pBlkExp, NULL,
								ui32CalcCRC, ui32BlkCRC))
	{
		goto Exit;
	}

	// Now display the items

	uiLfCount = 0;
	pLfHdr = (F_LF_HDR *)((FLMBYTE *)pBlkHdr + SIZEOF_STD_BLK_HDR);
	while (uiPos < uiEndOfBlock)
	{
		FLMUINT	uiLfType = pLfHdr->ui32LfType;

		if (uiLfType != XFLM_LF_INVALID)
		{
			if (!ViewOutputLFH( uiCol, &uiRow, pLfHdr, uiLfCount,
										uiReadAddress + uiPos))
			{
				goto Exit;
			}
		}
		uiLfCount++;
		pLfHdr++;
		uiPos += sizeof( F_LF_HDR);
	}
	bOk = TRUE;

Exit:

	return( bOk);

}

/***************************************************************************
Desc: This routine sets things up to display the logical files in a
		database.
*****************************************************************************/
void ViewLogicalFiles( void)
{
	BLK_EXP	BlkExp;

	if (!gv_bViewHdrRead)
	{
		ViewReadHdr();
	}

	// If there are no LFH blocks, show a message and return

	if (gv_ViewDbHdr.ui32FirstLFBlkAddr == 0)
	{
		ViewShowError( "No LFH blocks in database");
		return;
	}

	BlkExp.uiType = BT_LFH_BLK;
	BlkExp.uiPrevAddr = 0;
	BlkExp.uiNextAddr = 0xFFFFFFFF;
	ViewBlocks( (FLMUINT)gv_ViewDbHdr.ui32FirstLFBlkAddr,
					(FLMUINT)gv_ViewDbHdr.ui32FirstLFBlkAddr, &BlkExp);
}
