//-------------------------------------------------------------------------
// Desc:	View logical file headers (indexes and containers).
// Tabs:	3
//
// Copyright (c) 1992-2001, 2003-2007 Novell, Inc. All Rights Reserved.
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

FSTATIC FLMINT ViewOutputLFH2_0(
	FLMUINT		Col,
	FLMUINT  *	RowRV,
	FLMBYTE *   LFH,
	FLMUINT     Ref,
	FLMUINT     FileOffset
	);

FSTATIC FLMINT ViewSetupLogicalFileMenu(
	FLMUINT     lfNum,
	FLMBYTE *   LFH
	);

/***************************************************************************
Name:    FormatLFType
Desc:    This routine formats a logical file type into an ASCII buffer.
*****************************************************************************/
void FormatLFType(
	FLMBYTE *   DestBuf,
	FLMUINT		lfType
	)
{
	FLMBYTE	TempBuf [40];

	switch( lfType)
	{
		case LF_CONTAINER:
			f_strcpy( (char *)DestBuf, "Container");
			break;
		case LF_INDEX:
			f_strcpy( (char *)DestBuf, "Index");
			break;
		case LF_INVALID:
			f_strcpy( (char *)DestBuf, "Deleted");
			break;
		default:
			f_sprintf( (char *)TempBuf, "Unknown: %u", (unsigned)lfType);
			f_strcpy( (char *)DestBuf, (const char *)TempBuf);
			break;
	}
}

/***************************************************************************
Name:    ViewOutputLFH2_0
Desc:    This routine outputs the information in a single LFH - for a
			single logical file - FLAIM 1.5 and above.
*****************************************************************************/
FSTATIC FLMINT ViewOutputLFH2_0(
	FLMUINT		Col,
	FLMUINT  *	RowRV,
	FLMBYTE *   LFH,
	FLMUINT     Ref,
	FLMUINT     FileOffset
	)
{
	FLMUINT     LabelWidth = 35;
	FLMUINT     Row = *RowRV;
	FLMUINT		BlkAddress;
	FLMBYTE		TempBuf [80];
	eColorType	bc = FLM_BLACK;
	eColorType	fc = FLM_LIGHTGRAY;
	eColorType	mbc = FLM_BLACK;
	eColorType	mfc = FLM_WHITE;
	eColorType	sbc = FLM_BLUE;
	eColorType	sfc = FLM_WHITE;
	FLMUINT     Option;
	FLMUINT		lfNum;

	/* Output Logical File Name */

	lfNum = FB2UW( &LFH [LFH_LF_NUMBER_OFFSET]);
	switch (lfNum)
	{
		case FLM_DICT_CONTAINER:
			f_strcpy( (char *)TempBuf, "LOCAL_DICT");
			break;
		case FLM_DATA_CONTAINER:
			f_strcpy( (char *)TempBuf, "DEFAULT_DATA");
			break;
		case FLM_TRACKER_CONTAINER:
			f_strcpy( (char *)TempBuf, "TRACKER_CONTAINER");
			break;
		case FLM_DICT_INDEX:
			f_strcpy( (char *)TempBuf, "LOCAL_DICT_IX");
			break;
		default:
			switch (LFH [LFH_TYPE_OFFSET])
			{
				case LF_INDEX:
					f_sprintf( (char *)TempBuf, "INDEX_%u", (unsigned)lfNum);
					break;
				case LF_CONTAINER:
					f_sprintf( (char *)TempBuf, "CONTAINER_%u", (unsigned)lfNum);
					break;
				default:
					f_sprintf( (char *)TempBuf, "UNKNOWN_TYPE[%u]_%u",
						(unsigned)LFH [LFH_TYPE_OFFSET], (unsigned)lfNum);
					break;
			}
			break;
	}

	if (!ViewAddMenuItem( LBL_LOGICAL_FILE_NAME, LabelWidth,
				VAL_IS_TEXT_PTR,
				(FLMUINT)((FLMBYTE *)(&TempBuf [0])), 0,
				0, VIEW_INVALID_FILE_OFFSET, (FLMUINT)f_strlen( (const char *)TempBuf),
				MOD_DISABLED,
				Col, Row++, 0, FLM_GREEN, FLM_WHITE,
				FLM_GREEN, FLM_WHITE))
		return( 0);

	/* Adjust column and label width so the rest is indented */

	Col += 2;
	LabelWidth -= 2;

	/* Output Logical File Number */

	if (!ViewAddMenuItem( LBL_LOGICAL_FILE_NUMBER, LabelWidth,
				VAL_IS_NUMBER | DISP_DECIMAL,
				(FLMUINT)(lfNum), 0,
				0, FileOffset + LFH_LF_NUMBER_OFFSET, 0,
				MOD_FLMUINT | MOD_DECIMAL,
				Col, Row++, 0, bc, fc, bc, fc))
		return( 0);

	/* Output Logical File Type */

	FormatLFType( TempBuf, LFH [LFH_TYPE_OFFSET]);
	if (!ViewAddMenuItem( LBL_LOGICAL_FILE_TYPE, LabelWidth,
				VAL_IS_TEXT_PTR,
				(FLMUINT)((FLMBYTE *)(&TempBuf [0])), 0,
				0, FileOffset + LFH_LF_NUMBER_OFFSET, 0,
				MOD_FLMBYTE | MOD_HEX,
				Col, Row++, 0, bc, fc, bc, fc))
		return( 0);

	/* Output the root block address */

	if ((BlkAddress = FB2UD( &LFH [LFH_ROOT_BLK_OFFSET])) == 0xFFFFFFFF)
		Option = 0;
	else
		Option = LFH_OPTION_ROOT_BLOCK | Ref;
	if (!ViewAddMenuItem( LBL_ROOT_BLOCK_ADDRESS, LabelWidth,
				VAL_IS_NUMBER | DISP_HEX_DECIMAL,
				BlkAddress, 0,
				0, FileOffset + LFH_ROOT_BLK_OFFSET, 0,
				MOD_FLMUINT | MOD_HEX,
				Col, Row++, Option,
				!Option ? bc : mbc,
				!Option ? fc : mfc,
				!Option ? bc : sbc,
				!Option ? fc : sfc))
		return( 0);

	/* Output the next DRN */

	if (!ViewAddMenuItem( LBL_NEXT_DRN, LabelWidth,
				VAL_IS_NUMBER | DISP_DECIMAL,
				FB2UD( &LFH [LFH_NEXT_DRN_OFFSET]), 0,
				0, FileOffset + LFH_NEXT_DRN_OFFSET, 0,
				MOD_FLMUINT | MOD_DECIMAL,
				Col, Row++, 0, bc, fc, bc, fc))
		return( 0);

	*RowRV = Row + 1;
	return( 1);
}

/***************************************************************************
Name:    ViewSetupLogicalFileMenu
Desc:    This routine sets up a menu for displaying a single logical
			file.
*****************************************************************************/
FSTATIC FLMINT ViewSetupLogicalFileMenu(
	FLMUINT     lfNum,
	FLMBYTE *   LFH
	)
{
	FLMUINT     Row;
	FLMUINT     Col;
	FLMBYTE		lfName [40];
	FLMUINT		FileOffset;

	/* Retrieve the information for the logical file */

	if (!ViewGetLFName( lfName, lfNum, LFH, &FileOffset))
	{
		ViewShowError( "Could not retrieve LFH information");
		return( 0);
	}

	if (!ViewMenuInit( "Logical File"))
		return( 0);
	Row = 3;
	Col = 5;

	/* Output the items in the LFH */

	if (!ViewOutputLFH2_0( Col, &Row, LFH, 0, FileOffset))
		return( 0);
	return( 1);
}

/***************************************************************************
Name:    ViewLogicalFile
Desc:    This routine displays a single logical file and allows the user
			to press menu keys while displaying the information.
*****************************************************************************/
void ViewLogicalFile(
	FLMUINT		lfNum
	)
{
	FLMUINT     Option;
	VIEW_INFO   SaveView;
	FLMUINT     Done = 0;
	FLMUINT     Repaint = 1;
	FLMBYTE		LFH[ LFH_SIZE];
	BLK_EXP     BlkExp2;
	FLMUINT     BlkAddress2 = 0;

	/* Loop getting commands until the hit the exit key */

	ViewReset( &SaveView);
	while ((!Done) && (!gv_bViewPoppingStack))
	{
		if (Repaint)
		{
			if (!ViewSetupLogicalFileMenu( lfNum, LFH))
				Done = 1;
		}
		if (!Done)
		{
			Repaint = 1;
			Option = ViewGetMenuOption();
			switch( Option)
			{
				case ESCAPE_OPTION:
					Done = 1;
					break;
				case SEARCH_OPTION:
					{
						VIEW_MENU_ITEM_p  vp = gv_pViewMenuCurrItem;

						/* Determine which logical file, if any we are pointing at */

						while ((vp != NULL) &&
									(vp->iLabelIndex != LBL_LOGICAL_FILE_NAME))
							vp = vp->PrevItem;
						if (vp != NULL)
						{
							while ((vp != NULL) &&
										(vp->iLabelIndex != LBL_LOGICAL_FILE_NUMBER))
								vp = vp->NextItem;
						}
						if (vp != NULL)
						{
							gv_uiViewSearchLfNum = (FLMUINT)vp->Value;
							if (ViewGetKey())
								gv_bViewPoppingStack = TRUE;
						}
						else
							ViewShowError( "Position cursor to a logical file before searching");
					}
					break;
				default:
					if ((Option & LFH_OPTION_ROOT_BLOCK) ||
						 (Option & LFH_OPTION_LAST_BLOCK))
					{
						if (Option & LFH_OPTION_ROOT_BLOCK)
						{
							BlkExp2.Level = 0xFF;
							BlkExp2.Type = 0xFF;
							BlkAddress2 = FB2UD( &LFH [LFH_ROOT_BLK_OFFSET]);
							BlkExp2.NextAddr = BlkExp2.PrevAddr = 0xFFFFFFFF;
						}
						else
						{
							flmAssert( 0);
						}
						BlkExp2.LfNum = FB2UW( &LFH [LFH_LF_NUMBER_OFFSET]);
						ViewBlocks( BlkAddress2, BlkAddress2, &BlkExp2);
					}
					else if (Option & LOGICAL_FILE_OPTION)
						ViewLogicalFile( (FLMUINT)(Option & (~(LOGICAL_FILE_OPTION))));
					else
						Repaint = 0;
					break;
			}
		}
	}
	ViewRestore( &SaveView);
}

/***************************************************************************
Name:    ViewLFHBlk
Desc:    This routine displays ALL of the logical files in an LFH block
			in the database.
*****************************************************************************/
FLMINT ViewLFHBlk(
	FLMUINT			ReadAddress,
	FLMUINT        BlkAddress,
	FLMBYTE **		BlkPtrRV,
	BLK_EXP_p      BlkExp
	)
{
	FLMUINT		Row;
	FLMUINT     Col;
	FLMUINT     EndOfBlock;
	FLMUINT     Pos;
	FLMBYTE *   BlkPtr;
	FLMUINT     Ref = 0;
	FLMUINT16	ui16CalcChkSum;
	FLMUINT16	ui16BlkChkSum;
	FLMUINT		uiBytesRead;

	/* Read the block into memory */

	if (!ViewBlkRead( ReadAddress, BlkPtrRV,
										gv_ViewHdrInfo.FileHdr.uiBlockSize,
										&ui16CalcChkSum, &ui16BlkChkSum,
										&uiBytesRead, TRUE, NULL,
										FALSE, NULL))
		return( 0);
	BlkPtr = *BlkPtrRV;
	Pos = BH_OVHD;
	if (uiBytesRead <= BH_OVHD)
		EndOfBlock = BH_OVHD;
	else
	{
		EndOfBlock = FB2UW( &BlkPtr [BH_BLK_END]);
		if (EndOfBlock > uiBytesRead)
			EndOfBlock = uiBytesRead;
		if (EndOfBlock > gv_ViewHdrInfo.FileHdr.uiBlockSize)
			EndOfBlock = gv_ViewHdrInfo.FileHdr.uiBlockSize;
	}

	if (!ViewMenuInit( "LFH Block"))
		return( 0);

	/* Output the block header first */

	Row = 0;
	Col = 5;
	BlkExp->Type = BHT_LFH_BLK;
	BlkExp->LfNum = 0;
	BlkExp->BlkAddr = BlkAddress;
	BlkExp->Level = 0xFF;
	if (!ViewOutBlkHdr( Col, &Row, BlkPtr, BlkExp, NULL,
								ui16CalcChkSum, ui16BlkChkSum))
		return( 0);

	/* Now display the items */

	Ref = 0;
	while (Pos < EndOfBlock)
	{
		FLMBYTE   byLfType = BlkPtr [Pos + LFH_TYPE_OFFSET];

		if (byLfType != LF_INVALID)
		{
			if (!ViewOutputLFH2_0( Col, &Row, &BlkPtr [Pos], Ref,
													ReadAddress + Pos))
				return( 0);
		}
		Ref++;
		Pos += LFH_SIZE;
	}
	return( 1);
}

/***************************************************************************
Name: ViewLogicalFiles
Desc: This routine sets things up to display the logical files in a
		database.
*****************************************************************************/
void ViewLogicalFiles(
	void
	)
{
	BLK_EXP  BlkExp;

	if (!gv_bViewHdrRead)
		ViewReadHdr();

	/* If there are no LFH blocks, show a message and return */

	if (gv_ViewHdrInfo.FileHdr.uiFirstLFHBlkAddr == 0xFFFFFFFF)
	{
		ViewShowError( "No LFH blocks in database");
		return;
	}

	BlkExp.Type = BHT_LFH_BLK;
	BlkExp.PrevAddr = 0xFFFFFFFF;
	BlkExp.NextAddr = 0;
	ViewBlocks( gv_ViewHdrInfo.FileHdr.uiFirstLFHBlkAddr,
							gv_ViewHdrInfo.FileHdr.uiFirstLFHBlkAddr, &BlkExp);
}

