//-------------------------------------------------------------------------
// Desc:	View database header.
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

#define FILE_HEADER_MENU_LOG_HEADER    1
#define FILE_HEADER_MENU_LFH_BLOCKS    2
//#define FILE_HEADER_MENU_PCODE_BLOCKS  3

FSTATIC FLMINT ViewSetupFileHeaderMenu(
	void
	);

/***************************************************************************
Name:    ViewSetupFileHeaderMenu
Desc:    This routine displays the file header of a database.
*****************************************************************************/
FSTATIC FLMINT ViewSetupFileHeaderMenu(
	void
	)
{
#define LABEL_WIDTH  30
	FLMUINT      Row;
	FLMUINT      Col;
	eColorType   bc = FLM_BLACK;
	eColorType   fc = FLM_LIGHTGRAY;
	eColorType   mbc = FLM_BLACK;
	eColorType   mfc = FLM_WHITE;
	eColorType   sbc = FLM_BLUE;
	eColorType   sfc = FLM_WHITE;
	FLMUINT      Option;
	FLMINT       iStatus;
	FLMBYTE      TempBuf[ 100];

	/* Re-read the header information in case it has changed. */

	ViewReadHdr();
	if (!ViewMenuInit( "File Header"))
		goto Zero_Exit;
	Row = 0;
	Col = 5;

	/* Display the application major/minor version numbers */

	if (!ViewAddMenuItem( LBL_PREFIX_MAJOR, LABEL_WIDTH,
			VAL_IS_NUMBER | DISP_DECIMAL,
			(FLMUINT)(gv_ViewHdrInfo.FileHdr.uiAppMajorVer), 0,
			0, 10, 0, MOD_FLMBYTE | MOD_DECIMAL,
			Col, Row++, 0, bc, fc, bc, fc))
		goto Zero_Exit;

	if (!ViewAddMenuItem( LBL_PREFIX_MINOR, LABEL_WIDTH,
			VAL_IS_NUMBER | DISP_DECIMAL,
			(FLMUINT)(gv_ViewHdrInfo.FileHdr.uiAppMinorVer), 0,
			0, 11L, 0, MOD_FLMBYTE | MOD_DECIMAL,
			Col, Row++, 0, bc, fc, bc, fc))
		goto Zero_Exit;

	/* Display the FLAIM Name */

	if (!ViewAddMenuItem( LBL_FLAIM_NAME, LABEL_WIDTH,
			VAL_IS_TEXT_PTR,
			(FLMUINT)((FLMBYTE *)(&gv_szFlaimName[ 0])), 0,
			0, FLAIM_HEADER_START + FLAIM_NAME_POS, FLAIM_NAME_LEN, MOD_TEXT,
			Col, Row++, 0, bc, fc, bc, fc))
		goto Zero_Exit;

	/* Display the FLAIM version number */

	if (!ViewAddMenuItem( LBL_FLAIM_VERSION, LABEL_WIDTH,
			VAL_IS_TEXT_PTR,
			(FLMUINT)((FLMBYTE *)(&gv_szFlaimVersion[ 0])), 0,
			0, FLAIM_HEADER_START + FLM_FILE_FORMAT_VER_POS, 
			FLM_FILE_FORMAT_VER_LEN, MOD_TEXT,
			Col, Row++, 0, bc, fc, bc, fc))
		goto Zero_Exit;

	/* Display the default language */

	f_languageToStr( gv_ViewHdrInfo.FileHdr.uiDefaultLanguage, (char *)TempBuf);
	if (!ViewAddMenuItem( LBL_DEFAULT_LANGUAGE, LABEL_WIDTH,
			VAL_IS_TEXT_PTR,
			(FLMUINT)((FLMBYTE *)(&TempBuf[ 0])), 0,
			0, FLAIM_HEADER_START + DB_DEFAULT_LANGUAGE, 0, MOD_LANGUAGE,
			Col, Row++, 0, bc, fc, bc, fc))
		goto Zero_Exit;

	/* Display the database block size */

	if (!ViewAddMenuItem( LBL_BLOCK_SIZE, LABEL_WIDTH,
			VAL_IS_NUMBER | DISP_DECIMAL,
			(FLMUINT)(gv_ViewHdrInfo.FileHdr.uiBlockSize), 0,
			0, FLAIM_HEADER_START + DB_BLOCK_SIZE, 0, MOD_FLMUINT | MOD_DECIMAL,
			Col, Row++, 0, bc, fc, bc, fc))
		goto Zero_Exit;

	/* Display the first LFH block address */

	if (gv_ViewHdrInfo.FileHdr.uiFirstLFHBlkAddr == 0xFFFFFFFF)
		Option = 0;
	else
		Option = FILE_HEADER_MENU_LFH_BLOCKS;
	if (!ViewAddMenuItem( LBL_FIRST_LFH_BLOCK_ADDRESS, LABEL_WIDTH,
			VAL_IS_NUMBER | DISP_HEX_DECIMAL,
			(FLMUINT)(gv_ViewHdrInfo.FileHdr.uiFirstLFHBlkAddr), 0,
			0, FLAIM_HEADER_START + DB_1ST_LFH_ADDR, 0, MOD_FLMUINT | MOD_HEX,
			Col, Row++, Option,
			!Option ? bc : mbc,
			!Option ? fc : mfc,
			!Option ? bc : sbc,
			!Option ? fc : sfc))
		goto Zero_Exit;

	/* Display the first PCODE block address */

	if (gv_ViewHdrInfo.FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_3)
	{
		if (!ViewAddMenuItem( LBL_FIRST_PCODE_BLOCK_ADDRESS, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_HEX_DECIMAL,
				(FLMUINT)(gv_uiPcodeAddr), 0,
				0, FLAIM_HEADER_START + DB_1ST_PCODE_ADDR,
				0, MOD_FLMUINT | MOD_HEX,
				Col, Row++, 0,
				!Option ? bc : mbc,
				!Option ? fc : mfc,
				!Option ? bc : sbc,
				!Option ? fc : sfc))
			goto Zero_Exit;
	}

	iStatus = 1;
	goto Exit;
Zero_Exit:
	iStatus = 0;
Exit:
	return( iStatus);
}

/***************************************************************************
Name:    ViewFileHeader
Desc:    This routine displays the file header of a database and allows the
			user to select items from the displayed menu.
*****************************************************************************/
void ViewFileHeader(
	void
	)
{
	FLMUINT		Option;
	VIEW_INFO   SaveView;
	FLMUINT     Done = 0;
	FLMUINT     Repaint = 1;
	BLK_EXP     BlkExp;
	FLMUINT     BlkAddress;
	FLMUINT		Type;
	FLMUINT     ViewHexFlag = FALSE;
	FLMBYTE *   BlkPtr = NULL;

	/* Loop getting commands until the ESC key is pressed */

	ViewReset( &SaveView);
	while( !Done)
	{
		if (gv_bViewPoppingStack)
		{
			ViewSearch();
		}
		if (Repaint)
		{
			if (ViewHexFlag)
			{
				ViewHexBlock( 0, &BlkPtr, FALSE, 2048);
			}
			else
			{
				if (!ViewSetupFileHeaderMenu())
					Done = 1;
			}
		}
		if (!Done)
		{
			Repaint = 1;
			ViewEnable();
			Option = ViewGetMenuOption();
			switch( Option)
			{
				case ESCAPE_OPTION:
					Done = 1;
					break;
				case FILE_HEADER_MENU_LOG_HEADER:
					ViewLogHeader();
					break;
				case FILE_HEADER_MENU_LFH_BLOCKS:
					ViewLogicalFiles();
					break;
				case SEARCH_OPTION:
					gv_uiViewSearchLfNum = FLM_DATA_CONTAINER;
					if (ViewGetKey())
						ViewSearch();
					break;
				case GOTO_BLOCK_OPTION:
					if (GetBlockAddrType( &BlkAddress, &Type))
					{
						BlkExp.Type = Type;
						BlkExp.Level = 0xFF;
						BlkExp.NextAddr = 0;
						BlkExp.PrevAddr = 0;
						BlkExp.LfNum = 0;
						ViewBlocks( BlkAddress, BlkAddress, &BlkExp);
					}
					else
						Repaint = 0;
					break;
				case EDIT_OPTION:
				case EDIT_RAW_OPTION:
					if (!ViewEdit( FALSE, TRUE))
						Repaint = 0;
					break;
				case HEX_OPTION:
					ViewDisable();
					ViewHexFlag = !ViewHexFlag;
					break;
				default:
					Repaint = 0;
					break;
			}
		}
	}
	f_free( &BlkPtr);
	ViewRestore( &SaveView);
}

