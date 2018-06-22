//-------------------------------------------------------------------------
// Desc:	View log header.
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

// Menu items

#define LOG_HEADER_MENU_AVAIL_BLOCK       1
#define LOG_HEADER_MENU_BACKCHAIN_BLOCK   2

FSTATIC void viewFormatSerialNum(
	FLMBYTE *	pucBuf,
	FLMBYTE *	pucSerialNum
	);

FSTATIC FLMINT ViewSetupLogHeaderMenu(
	void
	);


/***************************************************************************
Name: viewFormatSerialNum
Desc: Format a serial number for display.
*****************************************************************************/
FSTATIC void viewFormatSerialNum(
	FLMBYTE *	pucBuf,
	FLMBYTE *	pucSerialNum
	)
{
	f_sprintf( (char *)pucBuf,
			"%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
			(unsigned)pucSerialNum[ 0],
			(unsigned)pucSerialNum[ 1],
			(unsigned)pucSerialNum[ 2],
			(unsigned)pucSerialNum[ 3],
			(unsigned)pucSerialNum[ 4],
			(unsigned)pucSerialNum[ 5],
			(unsigned)pucSerialNum[ 6],
			(unsigned)pucSerialNum[ 7],
			(unsigned)pucSerialNum[ 8],
			(unsigned)pucSerialNum[ 9],
			(unsigned)pucSerialNum[ 10],
			(unsigned)pucSerialNum[ 11],
			(unsigned)pucSerialNum[ 12],
			(unsigned)pucSerialNum[ 13],
			(unsigned)pucSerialNum[ 14],
			(unsigned)pucSerialNum[ 15]);
}

/***************************************************************************
Name: ViewSetupLogHeaderMenu
Desc: This routine displays the information found in a log header and
		sets up the menu for the log header display.
*****************************************************************************/
FSTATIC FLMINT ViewSetupLogHeaderMenu(
	void
	)
{
#define LABEL_WIDTH  35
	FLMUINT     Row;
	FLMUINT     Col;
	eColorType	bc = FLM_BLACK;
	eColorType	fc = FLM_LIGHTGRAY;
	eColorType	mbc = FLM_BLACK;
	eColorType	mfc = FLM_WHITE;
	eColorType	sbc = FLM_BLUE;
	eColorType	sfc = FLM_WHITE;
	FLMUINT		Option;
	FLMUINT		uiTmp;
	FLMUINT		uiDbVersion;
	FLMUINT		uiFirstAvailBlk;
	FLMUINT		uiFirstBCAddr;

	/* Re-read the header information in case it has changed. */

	ViewReadHdr();

	if (!ViewMenuInit( "Log Header"))
		return( 0);
	Row = 0;
	Col = 5;
	uiDbVersion = (FLMUINT)FB2UW( &gv_ucViewLogHdr [ LOG_FLAIM_VERSION]);

	/* Display the current roll-forward log file number */

	if (!ViewAddMenuItem( LBL_RFL_FILE_NUM, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_DECIMAL,
				(FLMUINT)FB2UD( &gv_ucViewLogHdr [LOG_RFL_FILE_NUM]), 0,
				0,
				DB_LOG_HEADER_START + LOG_RFL_FILE_NUM,
				0, MOD_FLMUINT | MOD_DECIMAL,
				Col, Row++, 0, bc, fc, bc, fc))
		return( 0);

	/* Display the last transaction offset in the roll forward log file. */

	if (!ViewAddMenuItem( LBL_RFL_LAST_TRANS_OFFSET, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_HEX_DECIMAL,
				(FLMUINT)FB2UD( &gv_ucViewLogHdr [LOG_RFL_LAST_TRANS_OFFSET]), 0,
				0,
				DB_LOG_HEADER_START + LOG_RFL_LAST_TRANS_OFFSET,
				0, MOD_FLMUINT | MOD_HEX,
				Col, Row++, 0, bc, fc, bc, fc))
		return( 0);

	/* Display the last checkpoint roll-forward log file number. */

	if (!ViewAddMenuItem( LBL_RFL_LAST_CP_FILE_NUM, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_DECIMAL,
				(FLMUINT)FB2UD( &gv_ucViewLogHdr [LOG_RFL_LAST_CP_FILE_NUM]), 0,
				0,
				DB_LOG_HEADER_START + LOG_RFL_LAST_CP_FILE_NUM,
				0, MOD_FLMUINT | MOD_DECIMAL,
				Col, Row++, 0, bc, fc, bc, fc))
		return( 0);

	/* Display the last checkpoint roll-forward log file offset. */

	if (!ViewAddMenuItem( LBL_RFL_LAST_CP_OFFSET, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_HEX_DECIMAL,
				(FLMUINT)FB2UD( &gv_ucViewLogHdr [LOG_RFL_LAST_CP_OFFSET]), 0,
				0,
				DB_LOG_HEADER_START + LOG_RFL_LAST_CP_OFFSET,
				0, MOD_FLMUINT | MOD_HEX,
				Col, Row++, 0, bc, fc, bc, fc))
		return( 0);

	/* Display the last RFL file that was deleted. */

	if (!ViewAddMenuItem( LBL_LAST_RFL_FILE_DELETED, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_DECIMAL,
				(FLMUINT)FB2UD( &gv_ucViewLogHdr [LOG_LAST_RFL_FILE_DELETED]), 0,
				0,
				DB_LOG_HEADER_START + LOG_LAST_RFL_FILE_DELETED,
				0, MOD_FLMUINT | MOD_DECIMAL,
				Col, Row++, 0, bc, fc, bc, fc))
		return( 0);

	/* Display the ID of the last checkpoint that was done. */

	if (!ViewAddMenuItem( LBL_LAST_CP_ID, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_DECIMAL,
				(FLMUINT)FB2UD( &gv_ucViewLogHdr [LOG_LAST_CP_TRANS_ID]), 0,
				0,
				DB_LOG_HEADER_START + LOG_LAST_CP_TRANS_ID,
				0, MOD_FLMUINT | MOD_DECIMAL,
				Col, Row++, 0, bc, fc, bc, fc))
		return( 0);

	/* Display the first checkpoint block address. */

	if (!ViewAddMenuItem( LBL_FIRST_CP_BLK_ADDR, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_HEX_DECIMAL,
				(FLMUINT)FB2UD( &gv_ucViewLogHdr [LOG_PL_FIRST_CP_BLOCK_ADDR]), 0,
				0,
				DB_LOG_HEADER_START + LOG_PL_FIRST_CP_BLOCK_ADDR,
				0, MOD_FLMUINT | MOD_HEX,
				Col, Row++, 0, bc, fc, bc, fc))
		return( 0);

	/* Display the end of log address */

	if (!ViewAddMenuItem( LBL_END_OF_LOG_ADDRESS, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_HEX_DECIMAL,
				(FLMUINT)FB2UD( &gv_ucViewLogHdr [LOG_ROLLBACK_EOF]), 0,
				0,
				DB_LOG_HEADER_START + LOG_ROLLBACK_EOF,
				0, MOD_FLMUINT | MOD_HEX,
				Col, Row++, 0, bc, fc, bc, fc))
		return( 0);

	/* Display the minumum roll-forward log file size. */

	if (!ViewAddMenuItem( LBL_RFL_MIN_FILE_SIZE, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_DECIMAL_HEX,
				(FLMUINT)FB2UD( &gv_ucViewLogHdr [LOG_RFL_MIN_FILE_SIZE]), 0,
				0,
				DB_LOG_HEADER_START + LOG_RFL_MIN_FILE_SIZE,
				0, MOD_FLMUINT | MOD_DECIMAL,
				Col, Row++, 0, bc, fc, bc, fc))
		return( 0);

	/* Display the maximum roll-forward log file size. */

	if( uiDbVersion >= FLM_FILE_FORMAT_VER_4_3)
	{
		if (!ViewAddMenuItem( LBL_RFL_MAX_FILE_SIZE, LABEL_WIDTH,
					VAL_IS_NUMBER | DISP_DECIMAL_HEX,
					(FLMUINT)FB2UD( &gv_ucViewLogHdr [LOG_RFL_MAX_FILE_SIZE]), 0,
					0,
					DB_LOG_HEADER_START + LOG_RFL_MAX_FILE_SIZE,
					0, MOD_FLMUINT | MOD_DECIMAL,
					Col, Row++, 0, bc, fc, bc, fc))
			return( 0);
	}

	/* Display the keep-RFL-files flag. */

	if (!ViewAddMenuItem( LBL_KEEP_RFL_FILES, LABEL_WIDTH,
			VAL_IS_LABEL_INDEX,
			(gv_ucViewLogHdr [LOG_KEEP_RFL_FILES])
			? (FLMUINT)LBL_YES
			: (FLMUINT)LBL_NO, 0,
			0, DB_LOG_HEADER_START + LOG_KEEP_RFL_FILES,
			0, MOD_FLMBYTE | MOD_DECIMAL,
			Col, Row++, 0, bc, fc, bc, fc))
		return( 0);

	if (uiDbVersion >= FLM_FILE_FORMAT_VER_4_3)
	{

		/* Display the auto turn off aborted transactions flag. */

		if (!ViewAddMenuItem( LBL_AUTO_TURN_OFF_KEEP_RFL, LABEL_WIDTH,
				VAL_IS_LABEL_INDEX,
				(gv_ucViewLogHdr [LOG_AUTO_TURN_OFF_KEEP_RFL])
				? (FLMUINT)LBL_YES
				: (FLMUINT)LBL_NO, 0,
				0, DB_LOG_HEADER_START + LOG_AUTO_TURN_OFF_KEEP_RFL,
				0, MOD_FLMBYTE | MOD_DECIMAL,
				Col, Row++, 0, bc, fc, bc, fc))
			return( 0);

		/* Display the keep aborted transactions flag. */

		if (!ViewAddMenuItem( LBL_KEEP_ABORTED_TRANS_IN_RFL_FILES, LABEL_WIDTH,
				VAL_IS_LABEL_INDEX,
				(gv_ucViewLogHdr [LOG_KEEP_ABORTED_TRANS_IN_RFL])
				? (FLMUINT)LBL_YES
				: (FLMUINT)LBL_NO, 0,
				0, DB_LOG_HEADER_START + LOG_KEEP_ABORTED_TRANS_IN_RFL,
				0, MOD_FLMBYTE | MOD_DECIMAL,
				Col, Row++, 0, bc, fc, bc, fc))
			return( 0);

	}

	/* Display the current transaction ID */

	if (!ViewAddMenuItem( LBL_CURRENT_TRANS_ID, LABEL_WIDTH,
			VAL_IS_NUMBER | DISP_DECIMAL,
			(FLMUINT)FB2UD( &gv_ucViewLogHdr [LOG_CURR_TRANS_ID]), 0,
			0, DB_LOG_HEADER_START + LOG_CURR_TRANS_ID,
			0, MOD_FLMUINT | MOD_DECIMAL,
			Col, Row++, 0, bc, fc, bc, fc))
		return( 0);

	/* Display the last committed transaction ID */

	if( uiDbVersion >= FLM_FILE_FORMAT_VER_4_3)
	{
		if (!ViewAddMenuItem( LBL_LAST_RFL_COMMIT_ID, LABEL_WIDTH,
					VAL_IS_NUMBER | DISP_DECIMAL,
					(FLMUINT)FB2UD( &gv_ucViewLogHdr [ LOG_LAST_RFL_COMMIT_ID]), 0,
					0, DB_LOG_HEADER_START + LOG_LAST_RFL_COMMIT_ID,
					0, MOD_FLMUINT | MOD_DECIMAL,
					Col, Row++, 0, bc, fc, bc, fc))
			return( 0);
	}

	/* Display the last commit ID */

	if (!ViewAddMenuItem( LBL_COMMIT_COUNT, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_DECIMAL,
				(FLMUINT)FB2UD( &gv_ucViewLogHdr [ LOG_COMMIT_COUNT]), 0,
				0, DB_LOG_HEADER_START + LOG_COMMIT_COUNT,
				0, MOD_FLMUINT | MOD_DECIMAL,
				Col, Row++, 0, bc, fc, bc, fc))
		return( 0);

	/* Display the log header checksum */

	uiTmp = (FLMUINT)FB2UW( &gv_ucViewLogHdr [ LOG_HDR_CHECKSUM]);
	if (!ViewAddMenuItem( LBL_HDR_CHECKSUM, LABEL_WIDTH,
						VAL_IS_NUMBER | DISP_DECIMAL,
						uiTmp, 0,
						0, DB_LOG_HEADER_START + LOG_HDR_CHECKSUM,
						0, MOD_FLMUINT16 | MOD_DECIMAL,
						Col, Row++, 0, bc, fc, bc, fc))
		return( 0);

	/* Display the log header CALCULATED checksum */

	uiTmp = lgHdrCheckSum( gv_ucViewLogHdr, FALSE);
	if (!ViewAddMenuItem( LBL_CALC_HDR_CHECKSUM, LABEL_WIDTH,
						VAL_IS_NUMBER | DISP_DECIMAL,
						uiTmp, 0,
						0, DB_LOG_HEADER_START + LOG_HDR_CHECKSUM,
						0, MOD_DISABLED,
						Col, Row++, 0, bc, fc, bc, fc))
		return( 0);

	/* Display the version number */

	if (!ViewAddMenuItem( LBL_FLAIM_VERSION, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_DECIMAL,
				uiDbVersion, 0,
				0, DB_LOG_HEADER_START + LOG_FLAIM_VERSION,
				0, MOD_FLMUINT16 | MOD_DECIMAL,
				Col, Row++, 0, bc, fc, bc, fc))
		return( 0);

	/* Display the number of blocks in the avail list */

	if (!ViewAddMenuItem( LBL_NUM_AVAIL_BLOCKS, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_DECIMAL,
				(FLMUINT)FB2UD( &gv_ucViewLogHdr [ LOG_PF_NUM_AVAIL_BLKS]), 0,
				0, DB_LOG_HEADER_START + LOG_PF_NUM_AVAIL_BLKS,
				0, MOD_FLMUINT | MOD_DECIMAL,
				Col, Row++, 0, bc, fc, bc, fc))
		return( 0);

	/* Display the first avail block address */

	uiFirstAvailBlk = (FLMUINT)FB2UD( &gv_ucViewLogHdr [LOG_PF_AVAIL_BLKS]);
	if (uiFirstAvailBlk == 0xFFFFFFFF)
		Option = 0;
	else
		Option = LOG_HEADER_MENU_AVAIL_BLOCK;
	if (!ViewAddMenuItem( LBL_FIRST_AVAIL_BLOCK_ADDRESS, LABEL_WIDTH,
						VAL_IS_NUMBER | DISP_HEX_DECIMAL,
						uiFirstAvailBlk, 0,
						0, DB_LOG_HEADER_START + LOG_PF_AVAIL_BLKS,
						0, MOD_FLMUINT | MOD_HEX,
						Col, Row++, Option,
						(!Option ? bc : mbc),
						(!Option ? fc : mfc),
						(!Option ? bc : sbc),
						(!Option ? fc : sfc)))
		return( 0);

	/* Display the back chain address */

	uiFirstBCAddr = (FLMUINT)FB2UD( &gv_ucViewLogHdr [LOG_PF_FIRST_BACKCHAIN]);
	if (uiFirstBCAddr == 0xFFFFFFFF)
		Option = 0;
	else
		Option = LOG_HEADER_MENU_BACKCHAIN_BLOCK;
	if (!ViewAddMenuItem( LBL_FIRST_BACKCHAIN_BLOCK_ADDRESS, LABEL_WIDTH,
					VAL_IS_NUMBER | DISP_HEX_DECIMAL,
					uiFirstBCAddr, 0,
					0, DB_LOG_HEADER_START + LOG_PF_FIRST_BACKCHAIN,
					0, MOD_FLMUINT | MOD_HEX,
					Col, Row++, Option,
					(!Option ? bc : mbc),
					(!Option ? fc : mfc),
					(!Option ? bc : sbc),
					(!Option ? fc : sfc)))
		return( 0);

	/* Display the back chain count */

	if (!ViewAddMenuItem( LBL_NUM_BACKCHAIN_BLOCKS, LABEL_WIDTH,
						VAL_IS_NUMBER | DISP_DECIMAL,
						(FLMUINT)gv_ucViewLogHdr [LOG_PF_FIRST_BC_CNT], 0,
						0, DB_LOG_HEADER_START + LOG_PF_FIRST_BC_CNT,
						0, MOD_FLMBYTE | MOD_DECIMAL,
						Col, Row++, 0, bc, fc, bc, fc))
		return( 0);

	/* Display the logical end of file address */

	if (!ViewAddMenuItem( LBL_LOGICAL_END_OF_FILE, LABEL_WIDTH,
			VAL_IS_NUMBER | DISP_HEX_DECIMAL,
			(FLMUINT)FB2UD( &gv_ucViewLogHdr [LOG_LOGICAL_EOF]), 0,
			0, DB_LOG_HEADER_START + LOG_LOGICAL_EOF,
			0, MOD_FLMUINT | MOD_HEX,
			Col, Row++, 0, bc, fc, bc, fc))
		return( 0);

	if (uiDbVersion >= FLM_FILE_FORMAT_VER_4_3)
	{
		FLMBYTE		ucBuf[ 64];

		Row++;

		viewFormatSerialNum( ucBuf, &gv_ucViewLogHdr [LOG_DB_SERIAL_NUM]);
		if (!ViewAddMenuItem( LBL_DB_SERIAL_NUM, LABEL_WIDTH,
				VAL_IS_TEXT_PTR,
				(FLMUINT)&ucBuf[ 0],
				f_strlen( (const char *)ucBuf),
				0, DB_LOG_HEADER_START + LOG_DB_SERIAL_NUM,
				0, 0,
				Col, Row++, 0, bc, fc, bc, fc))
			return( 0);

		viewFormatSerialNum( ucBuf,
			&gv_ucViewLogHdr [LOG_LAST_TRANS_RFL_SERIAL_NUM]);
		if (!ViewAddMenuItem( LBL_LAST_TRANS_RFL_SERIAL_NUM, LABEL_WIDTH,
				VAL_IS_TEXT_PTR,
				(FLMUINT)&ucBuf[ 0],
				f_strlen( (const char *)ucBuf),
				0, DB_LOG_HEADER_START + 
					LOG_LAST_TRANS_RFL_SERIAL_NUM,
				0, 0,
				Col, Row++, 0, bc, fc, bc, fc))
			return( 0);

		viewFormatSerialNum( ucBuf,
			&gv_ucViewLogHdr [LOG_RFL_NEXT_SERIAL_NUM]);
		if (!ViewAddMenuItem( LBL_RFL_NEXT_SERIAL_NUM, LABEL_WIDTH,
				VAL_IS_TEXT_PTR,
				(FLMUINT)&ucBuf[ 0],
				f_strlen( (const char *)ucBuf),
				0, DB_LOG_HEADER_START + 
					LOG_RFL_NEXT_SERIAL_NUM,
				0, 0,
				Col, Row++, 0, bc, fc, bc, fc))
			return( 0);

		Row++;

		viewFormatSerialNum( ucBuf,
			&gv_ucViewLogHdr [LOG_INC_BACKUP_SERIAL_NUM]);
		if (!ViewAddMenuItem( LBL_INC_BACKUP_SERIAL_NUM, LABEL_WIDTH,
				VAL_IS_TEXT_PTR,
				(FLMUINT)&ucBuf[ 0],
				f_strlen( (const char *)ucBuf),
				0, DB_LOG_HEADER_START + 
					LOG_INC_BACKUP_SERIAL_NUM,
				0, 0,
				Col, Row++, 0, bc, fc, bc, fc))
			return( 0);

		if (!ViewAddMenuItem( LBL_LAST_BACKUP_TRANS_ID, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_HEX_DECIMAL,
				(FLMUINT)FB2UD( &gv_ucViewLogHdr [ LOG_LAST_BACKUP_TRANS_ID]), 0,
				0, DB_LOG_HEADER_START +
				LOG_LAST_BACKUP_TRANS_ID,
				0, MOD_FLMUINT | MOD_HEX,
				Col, Row++, 0, bc, fc, bc, fc))
			return( 0);

		if (!ViewAddMenuItem( LBL_BLK_CHG_SINCE_BACKUP, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_HEX_DECIMAL,
				(FLMUINT)FB2UD( &gv_ucViewLogHdr [ LOG_BLK_CHG_SINCE_BACKUP]), 0,
				0, DB_LOG_HEADER_START + 
					LOG_BLK_CHG_SINCE_BACKUP,
				0, MOD_FLMUINT | MOD_HEX,
				Col, Row++, 0, bc, fc, bc, fc))
			return( 0);

		if (!ViewAddMenuItem( LBL_INC_BACKUP_SEQ_NUM, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_HEX_DECIMAL,
				(FLMUINT)FB2UD( &gv_ucViewLogHdr [LOG_INC_BACKUP_SEQ_NUM]), 0,
				0, DB_LOG_HEADER_START + 
				LOG_INC_BACKUP_SEQ_NUM,
				0, MOD_FLMUINT | MOD_HEX,
				Col, Row++, 0, bc, fc, bc, fc))
			return( 0);

		uiTmp = (FLMUINT)FB2UW( &gv_ucViewLogHdr [ LOG_MAX_FILE_SIZE]);
		if (!ViewAddMenuItem( LBL_MAX_FILE_SIZE, LABEL_WIDTH,
						VAL_IS_NUMBER | DISP_DECIMAL,
						uiTmp, 0,
						0, DB_LOG_HEADER_START + LOG_MAX_FILE_SIZE,
						0, MOD_FLMUINT16 | MOD_DECIMAL,
						Col, Row++, 0, bc, fc, bc, fc))
			return( 0);
	}

	return( 1);
}

/***************************************************************************
Name: ViewLogHeader
Desc: This routine sets up the log header menu and then allows a user to
		press keys while in the menu.
*****************************************************************************/
void ViewLogHeader(
	void
	)
{
	FLMUINT     Option;
	VIEW_INFO   SaveView;
	FLMUINT     Done = 0;
	FLMUINT     Repaint = 1;
	BLK_EXP     BlkExp;
	FLMUINT     BlkAddress;
	FLMUINT		Type;
	FLMUINT     ViewHexFlag = FALSE;
	FLMBYTE *	BlkPtr = NULL;
	FLMUINT		uiAddr;

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
				ViewHexBlock( DB_LOG_HEADER_START, &BlkPtr, FALSE,
									LOG_HEADER_SIZE);
			}
			else
			{
				if (!ViewSetupLogHeaderMenu())
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
				case LOG_HEADER_MENU_AVAIL_BLOCK:
					BlkExp.Type = BHT_FREE;
					BlkExp.PrevAddr = 0;
					BlkExp.NextAddr = 0;
					uiAddr = (FLMUINT)FB2UD( &gv_ucViewLogHdr [LOG_PF_AVAIL_BLKS]);
					ViewBlocks( uiAddr, uiAddr, &BlkExp);
					break;
				case LOG_HEADER_MENU_BACKCHAIN_BLOCK:
					BlkExp.Type = BHT_FREE;
					BlkExp.PrevAddr = 0;
					BlkExp.NextAddr = 0;
					uiAddr =
						(FLMUINT)FB2UD( &gv_ucViewLogHdr [LOG_PF_FIRST_BACKCHAIN]);
					ViewBlocks( uiAddr, uiAddr, &BlkExp);
					break;
				case SEARCH_OPTION:
					if ((gv_pViewMenuCurrItem->iLabelIndex == LBL_DICT_CONTAINER_RECORD_COUNT) ||
							(gv_pViewMenuCurrItem->iLabelIndex == LBL_DICT_CONTAINER_NEXT_RECORD) ||
							(gv_pViewMenuCurrItem->iLabelIndex == LBL_DICT_CONTAINER_LAST_BLOCK_ADDRESS))
						gv_uiViewSearchLfNum = FLM_DICT_CONTAINER;
					else
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

