//------------------------------------------------------------------------------
// Desc:	This file contains the routines which display database header
//			information.
// Tabs:	3
//
// Copyright (c) 2002-2007 Novell, Inc. All Rights Reserved.
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

#define DB_HDR_MENU_AVAIL_BLOCK	1
#define DB_HDR_MENU_LFH_BLOCK	2

// Local Function Prototypes

FSTATIC void viewFormatSerialNum(
	char *		pszBuf,
	FLMBYTE *	pucSerialNum);

FSTATIC void viewFormatDBKey(
	char *		pszBuf,
	FLMBYTE *	pucDBKey);

FSTATIC FLMBOOL ViewSetupDbHdrMenu( void);

/***************************************************************************
Desc: Format a serial number for display.
*****************************************************************************/
FSTATIC void viewFormatSerialNum(
	char *		pszBuf,
	FLMBYTE *	pucSerialNum
	)
{
	f_sprintf( pszBuf,
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
Desc: Format a DB Key.
*****************************************************************************/
FSTATIC void viewFormatDBKey(
	char *		pszBuf,
	FLMBYTE *	pucDBKey
	)
{
	f_sprintf( pszBuf,
			"%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x...",
			(unsigned)pucDBKey[ 0],
			(unsigned)pucDBKey[ 1],
			(unsigned)pucDBKey[ 2],
			(unsigned)pucDBKey[ 3],
			(unsigned)pucDBKey[ 4],
			(unsigned)pucDBKey[ 5],
			(unsigned)pucDBKey[ 6],
			(unsigned)pucDBKey[ 7],
			(unsigned)pucDBKey[ 8],
			(unsigned)pucDBKey[ 9],
			(unsigned)pucDBKey[ 10],
			(unsigned)pucDBKey[ 11],
			(unsigned)pucDBKey[ 12],
			(unsigned)pucDBKey[ 13],
			(unsigned)pucDBKey[ 14],
			(unsigned)pucDBKey[ 15]);
}

/***************************************************************************
Desc: This routine displays the information found in a log header and
		sets up the menu for the log header display.
*****************************************************************************/
FSTATIC FLMBOOL ViewSetupDbHdrMenu( void)
{
#define LABEL_WIDTH  35
	FLMBOOL			bOk = FALSE;
	FLMUINT			uiRow;
	FLMUINT			uiCol;
	eColorType		uiBackColor = FLM_BLACK;
	eColorType		uiForeColor = FLM_LIGHTGRAY;
	eColorType		uiUnselectBackColor = FLM_BLACK;
	eColorType		uiUnselectForeColor = FLM_WHITE;
	eColorType		uiSelectBackColor = FLM_BLUE;
	eColorType		uiSelectForeColor = FLM_WHITE;
	FLMUINT			uiOption;
	char				szBuf [64];
	FLMUINT32		ui32CalcCRC;

	// Re-read the header information in case it has changed.

	ViewReadHdr( &ui32CalcCRC);

	ViewMenuInit( "DB Header");
	uiRow = 0;
	uiCol = 5;

	// Display signature

	if (!ViewAddMenuItem( LBL_SIGNATURE, LABEL_WIDTH,
			VAL_IS_TEXT_PTR,
			(FLMUINT64)((FLMUINT)(&gv_ViewDbHdr.szSignature[ 0])), 0,
			0, XFLM_DB_HDR_szSignature_OFFSET, sizeof( gv_ViewDbHdr.szSignature),
			MOD_TEXT,
			uiCol, uiRow++, 0, uiBackColor, uiForeColor,
			uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display whether this is little-endian or not.

	if (!ViewAddMenuItem( LBL_LITTLE_ENDIAN, LABEL_WIDTH,
			VAL_IS_LABEL_INDEX,
			gv_ViewDbHdr.ui8IsLittleEndian
			? (FLMUINT64)LBL_YES
			: (FLMUINT64)LBL_NO, 0,
			0, XFLM_DB_HDR_ui8IsLittleEndian_OFFSET,
			0, MOD_FLMBYTE | MOD_DECIMAL,
			uiCol, uiRow++, 0, uiBackColor, uiForeColor,
			uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display the default language
	f_languageToStr(	(FLMUINT)gv_ViewDbHdr.ui8DefaultLanguage, szBuf);
	if (!ViewAddMenuItem( LBL_DEFAULT_LANGUAGE, LABEL_WIDTH,
			VAL_IS_TEXT_PTR,
			(FLMUINT64)((FLMUINT)(&szBuf[ 0])), 0,
			0, XFLM_DB_HDR_ui8DefaultLanguage_OFFSET, 0,
			MOD_LANGUAGE,
			uiCol, uiRow++, 0, uiBackColor, uiForeColor,
			uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display the database block size

	if (!ViewAddMenuItem( LBL_BLOCK_SIZE, LABEL_WIDTH,
			VAL_IS_NUMBER | DISP_DECIMAL,
			(FLMUINT64)(gv_ViewDbHdr.ui16BlockSize), 0,
			0, XFLM_DB_HDR_ui16BlockSize_OFFSET, 0,
			MOD_FLMUINT16 | MOD_DECIMAL | MOD_NATIVE,
			uiCol, uiRow++, 0, uiBackColor, uiForeColor,
			uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display the version number

	if (!ViewAddMenuItem( LBL_FLAIM_VERSION, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_DECIMAL,
				(FLMUINT64)gv_ViewDbHdr.ui32DbVersion, 0,
				0, XFLM_DB_HDR_ui32DbVersion_OFFSET,
				0, MOD_FLMUINT32 | MOD_DECIMAL | MOD_NATIVE,
				uiCol, uiRow++, 0,
				uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display whether block checksumming is enabled.

	if (!ViewAddMenuItem( LBL_BLK_CRC_ENABLED, LABEL_WIDTH,
			VAL_IS_LABEL_INDEX,
			gv_ViewDbHdr.ui8BlkChkSummingEnabled
			? (FLMUINT64)LBL_YES
			: (FLMUINT64)LBL_NO, 0,
			0, XFLM_DB_HDR_ui8BlkChkSummingEnabled_OFFSET,
			0, MOD_FLMBYTE | MOD_DECIMAL,
			uiCol, uiRow++, 0, uiBackColor, uiForeColor,
			uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display the keep-RFL-files flag.

	if (!ViewAddMenuItem( LBL_KEEP_RFL_FILES, LABEL_WIDTH,
			VAL_IS_LABEL_INDEX,
			gv_ViewDbHdr.ui8RflKeepFiles
			? (FLMUINT64)LBL_YES
			: (FLMUINT64)LBL_NO, 0,
			0, XFLM_DB_HDR_ui8RflKeepFiles_OFFSET,
			0, MOD_FLMBYTE | MOD_DECIMAL,
			uiCol, uiRow++, 0,
			uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display the auto turn off aborted transactions flag.

	if (!ViewAddMenuItem( LBL_AUTO_TURN_OFF_KEEP_RFL, LABEL_WIDTH,
			VAL_IS_LABEL_INDEX,
			gv_ViewDbHdr.ui8RflAutoTurnOffKeep
			? (FLMUINT64)LBL_YES
			: (FLMUINT64)LBL_NO, 0,
			0, XFLM_DB_HDR_ui8RflAutoTurnOffKeep_OFFSET,
			0, MOD_FLMBYTE | MOD_DECIMAL,
			uiCol, uiRow++, 0,
			uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display the keep aborted transactions flag.

	if (!ViewAddMenuItem( LBL_KEEP_ABORTED_TRANS_IN_RFL_FILES, LABEL_WIDTH,
			VAL_IS_LABEL_INDEX,
			gv_ViewDbHdr.ui8RflKeepAbortedTrans
			? (FLMUINT64)LBL_YES
			: (FLMUINT64)LBL_NO, 0,
			0, XFLM_DB_HDR_ui8RflKeepAbortedTrans_OFFSET,
			0, MOD_FLMBYTE | MOD_DECIMAL,
			uiCol, uiRow++, 0,
			uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display the current roll-forward log file number

	if (!ViewAddMenuItem( LBL_RFL_FILE_NUM, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_DECIMAL,
				(FLMUINT64)gv_ViewDbHdr.ui32RflCurrFileNum, 0,
				0,
				XFLM_DB_HDR_ui32RflCurrFileNum_OFFSET,
				0, MOD_FLMUINT32 | MOD_DECIMAL | MOD_NATIVE,
				uiCol, uiRow++, 0,
				uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display the last committed transaction ID

	if (!ViewAddMenuItem( LBL_LAST_RFL_COMMIT_ID, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_DECIMAL,
				gv_ViewDbHdr.ui64LastRflCommitID, 0,
				0, XFLM_DB_HDR_ui64LastRflCommitID_OFFSET,
				0, MOD_FLMUINT64 | MOD_DECIMAL | MOD_NATIVE,
				uiCol, uiRow++, 0,
				uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display the last RFL file that was deleted.

	if (!ViewAddMenuItem( LBL_LAST_RFL_FILE_DELETED, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_DECIMAL,
				(FLMUINT64)gv_ViewDbHdr.ui32RflLastFileNumDeleted, 0,
				0,
				XFLM_DB_HDR_ui32RflLastFileNumDeleted_OFFSET,
				0, MOD_FLMUINT32 | MOD_DECIMAL | MOD_NATIVE,
				uiCol, uiRow++, 0,
				uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display the last transaction offset in the roll forward log file.

	if (!ViewAddMenuItem( LBL_RFL_LAST_TRANS_OFFSET, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_HEX_DECIMAL,
				(FLMUINT64)gv_ViewDbHdr.ui32RflLastTransOffset, 0,
				0,
				XFLM_DB_HDR_ui32RflLastTransOffset_OFFSET,
				0, MOD_FLMUINT32 | MOD_HEX | MOD_NATIVE,
				uiCol, uiRow++, 0,
				uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display the last checkpoint roll-forward log file number.

	if (!ViewAddMenuItem( LBL_RFL_LAST_CP_FILE_NUM, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_DECIMAL,
				(FLMUINT64)gv_ViewDbHdr.ui32RflLastCPFileNum, 0,
				0,
				XFLM_DB_HDR_ui32RflLastCPFileNum_OFFSET,
				0, MOD_FLMUINT32 | MOD_DECIMAL | MOD_NATIVE,
				uiCol, uiRow++, 0,
				uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display the last checkpoint roll-forward log file offset.

	if (!ViewAddMenuItem( LBL_RFL_LAST_CP_OFFSET, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_HEX_DECIMAL,
				(FLMUINT64)gv_ViewDbHdr.ui32RflLastCPOffset, 0,
				0,
				XFLM_DB_HDR_ui32RflLastCPOffset_OFFSET,
				0, MOD_FLMUINT32 | MOD_HEX | MOD_NATIVE,
				uiCol, uiRow++, 0,
				uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display the transaction ID of the last checkpoint that was done.

	if (!ViewAddMenuItem( LBL_LAST_CP_ID, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_DECIMAL,
				gv_ViewDbHdr.ui64RflLastCPTransID, 0,
				0,
				XFLM_DB_HDR_ui64RflLastCPTransID_OFFSET,
				0, MOD_FLMUINT64 | MOD_DECIMAL | MOD_NATIVE,
				uiCol, uiRow++, 0,
				uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display the minumum roll-forward log file size.

	if (!ViewAddMenuItem( LBL_RFL_MIN_FILE_SIZE, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_DECIMAL_HEX,
				(FLMUINT64)gv_ViewDbHdr.ui32RflMinFileSize, 0,
				0,
				XFLM_DB_HDR_ui32RflMinFileSize_OFFSET,
				0, MOD_FLMUINT32 | MOD_DECIMAL | MOD_NATIVE,
				uiCol, uiRow++, 0,
				uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display the maximum roll-forward log file size.

	if (!ViewAddMenuItem( LBL_RFL_MAX_FILE_SIZE, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_DECIMAL_HEX,
				(FLMUINT64)gv_ViewDbHdr.ui32RflMaxFileSize, 0,
				0,
				XFLM_DB_HDR_ui32RflMaxFileSize_OFFSET,
				0, MOD_FLMUINT32 | MOD_DECIMAL | MOD_NATIVE,
				uiCol, uiRow++, 0,
				uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display the current transaction ID

	if (!ViewAddMenuItem( LBL_CURRENT_TRANS_ID, LABEL_WIDTH,
			VAL_IS_NUMBER | DISP_DECIMAL,
			gv_ViewDbHdr.ui64CurrTransID, 0,
			0, XFLM_DB_HDR_ui64CurrTransID_OFFSET,
			0, MOD_FLMUINT64 | MOD_DECIMAL | MOD_NATIVE,
			uiCol, uiRow++, 0,
			uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display the commit count

	if (!ViewAddMenuItem( LBL_COMMIT_COUNT, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_DECIMAL,
				gv_ViewDbHdr.ui64TransCommitCnt, 0,
				0, XFLM_DB_HDR_ui64TransCommitCnt_OFFSET,
				0, MOD_FLMUINT64 | MOD_DECIMAL | MOD_NATIVE,
				uiCol, uiRow++, 0,
				uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display the end of log address

	if (!ViewAddMenuItem( LBL_END_OF_LOG_ADDRESS, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_HEX_DECIMAL,
				(FLMUINT64)gv_ViewDbHdr.ui32RblEOF, 0,
				0,
				XFLM_DB_HDR_ui32RblEOF_OFFSET,
				0, MOD_FLMUINT32 | MOD_HEX | MOD_NATIVE,
				uiCol, uiRow++, 0,
				uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display the first checkpoint block address.

	if (!ViewAddMenuItem( LBL_FIRST_CP_BLK_ADDR, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_HEX_DECIMAL,
				(FLMUINT64)gv_ViewDbHdr.ui32RblFirstCPBlkAddr, 0,
				0,
				XFLM_DB_HDR_ui32RblFirstCPBlkAddr_OFFSET,
				0, MOD_FLMUINT32 | MOD_HEX | MOD_NATIVE,
				uiCol, uiRow++, 0,
				uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display the first avail block address

	if (gv_ViewDbHdr.ui32FirstAvailBlkAddr == 0)
	{
		uiOption = 0;
	}
	else
	{
		uiOption = DB_HDR_MENU_AVAIL_BLOCK;
	}
	if (!ViewAddMenuItem( LBL_FIRST_AVAIL_BLOCK_ADDRESS, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_HEX_DECIMAL,
				(FLMUINT64)gv_ViewDbHdr.ui32FirstAvailBlkAddr, 0,
				0, XFLM_DB_HDR_ui32FirstAvailBlkAddr_OFFSET,
				0, MOD_FLMUINT32 | MOD_HEX | MOD_NATIVE,
				uiCol, uiRow++, uiOption,
				(!uiOption ? uiBackColor : uiUnselectBackColor),
				(!uiOption ? uiForeColor : uiUnselectForeColor),
				(!uiOption ? uiBackColor : uiSelectBackColor),
				(!uiOption ? uiForeColor : uiSelectForeColor)))
	{
		goto Exit;
	}

	// Display the first LFH block address

	if (gv_ViewDbHdr.ui32FirstLFBlkAddr == 0)
	{
		uiOption = 0;
	}
	else
	{
		uiOption = DB_HDR_MENU_LFH_BLOCK;
	}
	if (!ViewAddMenuItem( LBL_FIRST_LFH_BLOCK_ADDRESS, LABEL_WIDTH,
				VAL_IS_NUMBER | DISP_HEX_DECIMAL,
				(FLMUINT64)gv_ViewDbHdr.ui32FirstLFBlkAddr, 0,
				0, XFLM_DB_HDR_ui32FirstLFBlkAddr_OFFSET,
				0, MOD_FLMUINT32 | MOD_HEX | MOD_NATIVE,
				uiCol, uiRow++, uiOption,
				(!uiOption ? uiBackColor : uiUnselectBackColor),
				(!uiOption ? uiForeColor : uiUnselectForeColor),
				(!uiOption ? uiBackColor : uiSelectBackColor),
				(!uiOption ? uiForeColor : uiSelectForeColor)))
	{
		goto Exit;
	}

	// Display the logical end of file address

	if (!ViewAddMenuItem( LBL_LOGICAL_END_OF_FILE, LABEL_WIDTH,
			VAL_IS_NUMBER | DISP_HEX_DECIMAL,
			(FLMUINT64)gv_ViewDbHdr.ui32LogicalEOF, 0,
			0, XFLM_DB_HDR_ui32LogicalEOF_OFFSET,
			0, MOD_FLMUINT32 | MOD_HEX | MOD_NATIVE,
			uiCol, uiRow++, 0,
			uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	if (!ViewAddMenuItem( LBL_MAX_FILE_SIZE, LABEL_WIDTH,
					VAL_IS_NUMBER | DISP_DECIMAL,
					(FLMUINT64)gv_ViewDbHdr.ui32MaxFileSize, 0,
					0, XFLM_DB_HDR_ui32MaxFileSize_OFFSET,
					0, MOD_FLMUINT32 | MOD_DECIMAL | MOD_NATIVE,
					uiCol, uiRow++, 0,
					uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display the last backup transaction ID

	if (!ViewAddMenuItem( LBL_LAST_BACKUP_TRANS_ID, LABEL_WIDTH,
			VAL_IS_NUMBER | DISP_HEX_DECIMAL,
			gv_ViewDbHdr.ui64LastBackupTransID, 0,
			0, XFLM_DB_HDR_ui64LastBackupTransID_OFFSET,
			0, MOD_FLMUINT32 | MOD_HEX | MOD_NATIVE,
			uiCol, uiRow++, 0,
			uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display the incremental backup sequence number

	if (!ViewAddMenuItem( LBL_INC_BACKUP_SEQ_NUM, LABEL_WIDTH,
			VAL_IS_NUMBER | DISP_HEX_DECIMAL,
			(FLMUINT64)gv_ViewDbHdr.ui32IncBackupSeqNum, 0,
			0, XFLM_DB_HDR_ui32IncBackupSeqNum_OFFSET,
			0, MOD_FLMUINT32 | MOD_HEX | MOD_NATIVE,
			uiCol, uiRow++, 0,
			uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display the number of blocks changed since last backup

	if (!ViewAddMenuItem( LBL_BLK_CHG_SINCE_BACKUP, LABEL_WIDTH,
			VAL_IS_NUMBER | DISP_HEX_DECIMAL,
			(FLMUINT64)gv_ViewDbHdr.ui32BlksChangedSinceBackup, 0,
			0, XFLM_DB_HDR_ui32BlksChangedSinceBackup_OFFSET,
			0, MOD_FLMUINT32 | MOD_HEX | MOD_NATIVE,
			uiCol, uiRow++, 0,
			uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	uiRow++;

	// Display the database serial number

	viewFormatSerialNum( szBuf, gv_ViewDbHdr.ucDbSerialNum);
	if (!ViewAddMenuItem( LBL_DB_SERIAL_NUM, LABEL_WIDTH,
			VAL_IS_TEXT_PTR,
			((FLMUINT64)(FLMUINT)&szBuf[ 0]),
			f_strlen( szBuf),
			0, XFLM_DB_HDR_ucDbSerialNum_OFFSET,
			0, 0,
			uiCol, uiRow++, 0,
			uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display the last RFL transaction serial number

	viewFormatSerialNum( szBuf, gv_ViewDbHdr.ucLastTransRflSerialNum);
	if (!ViewAddMenuItem( LBL_LAST_TRANS_RFL_SERIAL_NUM, LABEL_WIDTH,
			VAL_IS_TEXT_PTR,
			((FLMUINT64)(FLMUINT)&szBuf[ 0]),
			f_strlen( szBuf),
			0, XFLM_DB_HDR_ucLastTransRflSerialNum_OFFSET,
			0, 0,
			uiCol, uiRow++, 0,
			uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display the next RFL serial number

	viewFormatSerialNum( szBuf, gv_ViewDbHdr.ucNextRflSerialNum);
	if (!ViewAddMenuItem( LBL_RFL_NEXT_SERIAL_NUM, LABEL_WIDTH,
			VAL_IS_TEXT_PTR,
			((FLMUINT64)(FLMUINT)&szBuf[ 0]),
			f_strlen( szBuf),
			0, XFLM_DB_HDR_ucNextRflSerialNum_OFFSET,
			0, 0,
			uiCol, uiRow++, 0,
			uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	viewFormatSerialNum( szBuf, gv_ViewDbHdr.ucIncBackupSerialNum);
	if (!ViewAddMenuItem( LBL_INC_BACKUP_SERIAL_NUM, LABEL_WIDTH,
			VAL_IS_TEXT_PTR,
			((FLMUINT64)(FLMUINT)&szBuf[ 0]),
			f_strlen( szBuf),
			0, XFLM_DB_HDR_ucIncBackupSerialNum_OFFSET,
			0, 0,
			uiCol, uiRow++, 0,
			uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	uiRow++;

	// Display the datbase key length

	if (!ViewAddMenuItem( LBL_HDR_KEY_LEN, LABEL_WIDTH,
			VAL_IS_NUMBER | DISP_DECIMAL,
			(FLMUINT64)gv_ViewDbHdr.ui32DbKeyLen, 0,
			0, XFLM_DB_HDR_ui32DbKeyLen,
			0, MOD_FLMUINT32 | MOD_DECIMAL | MOD_NATIVE,
			uiCol, uiRow++, 0,
			uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	if( gv_ViewDbHdr.ui32DbKeyLen)
	{
		// Display the datbase key

		viewFormatDBKey( szBuf, gv_ViewDbHdr.DbKey);
		if (!ViewAddMenuItem( LBL_HDR_DB_KEY, LABEL_WIDTH,
				VAL_IS_TEXT_PTR,
				((FLMUINT64)(FLMUINT)&szBuf[ 0]),
				f_strlen( szBuf),
				0, XFLM_DB_HDR_ucDbSerialNum_OFFSET,
				0, 0,
				uiCol, uiRow++, 0,
				uiBackColor, uiForeColor, uiBackColor, uiForeColor))
		{
			goto Exit;
		}

	}

	uiRow++;

	// Display the database header CRC

	if (!ViewAddMenuItem( LBL_HDR_CRC, LABEL_WIDTH,
			VAL_IS_NUMBER | DISP_DECIMAL,
			(FLMUINT64)gv_ViewDbHdr.ui32HdrCRC, 0,
			0, XFLM_DB_HDR_ui32HdrCRC_OFFSET,
			0, MOD_FLMUINT32 | MOD_DECIMAL | MOD_NATIVE,
			uiCol, uiRow++, 0,
			uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display the database header CALCULATED CRC

	if (!ViewAddMenuItem( LBL_CALC_HDR_CRC, LABEL_WIDTH,
						VAL_IS_NUMBER | DISP_DECIMAL,
						(FLMUINT64)ui32CalcCRC, 0,
						0, XFLM_DB_HDR_ui32HdrCRC_OFFSET,
						0, MOD_DISABLED,
						uiCol, uiRow++, 0,
						uiBackColor, uiForeColor, uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	bOk = TRUE;

Exit:

	return( bOk);
}

/***************************************************************************
Desc: This routine sets up the log header menu and then allows a user to
		press keys while in the menu.
*****************************************************************************/
void ViewDbHeader( void)
{
	FLMUINT		uiOption;
	VIEW_INFO	SaveView;
	FLMBOOL		bRepaint = TRUE;
	BLK_EXP		BlkExp;
	FLMUINT		uiBlkAddress;
	FLMBOOL		bViewHexFlag = FALSE;
	F_BLK_HDR *	pBlkHdr = NULL;

	// Loop getting commands until the ESC key is pressed

	ViewReset( &SaveView);
	for (;;)
	{
		if (gv_bViewPoppingStack)
		{
			ViewSearch();
		}
		if (bRepaint)
		{
			if (bViewHexFlag)
			{
				ViewHexBlock( 0, &pBlkHdr, sizeof( XFLM_DB_HDR));
			}
			else
			{
				if (!ViewSetupDbHdrMenu())
				{
					goto Exit;
				}
			}
		}
		bRepaint = TRUE;
		ViewEnable();
		uiOption = ViewGetMenuOption();
		switch (uiOption)
		{
			case ESCAPE_OPTION:
				goto Exit;
			case DB_HDR_MENU_AVAIL_BLOCK:
				BlkExp.uiType = BT_FREE;
				BlkExp.uiPrevAddr = 0xFFFFFFFF;
				BlkExp.uiNextAddr = 0xFFFFFFFF;
				uiBlkAddress = (FLMUINT)gv_ViewDbHdr.ui32FirstAvailBlkAddr;
				ViewBlocks( uiBlkAddress, uiBlkAddress, &BlkExp);
				break;
			case DB_HDR_MENU_LFH_BLOCK:
				BlkExp.uiType = BT_LFH_BLK;
				BlkExp.uiPrevAddr = 0xFFFFFFFF;
				BlkExp.uiNextAddr = 0xFFFFFFFF;
				uiBlkAddress = (FLMUINT)gv_ViewDbHdr.ui32FirstLFBlkAddr;
				ViewBlocks( uiBlkAddress, uiBlkAddress, &BlkExp);
				break;
			case SEARCH_OPTION:
				gv_uiViewSearchLfNum = XFLM_DATA_COLLECTION;
				gv_uiViewSearchLfType = XFLM_LF_COLLECTION;
				if (ViewGetKey())
				{
					ViewSearch();
				}
				break;
			case GOTO_BLOCK_OPTION:
				if (GetBlockAddrType( &uiBlkAddress))
				{
					BlkExp.uiType = 0xFF;
					BlkExp.uiLevel = 0xFF;
					BlkExp.uiNextAddr = 0xFFFFFFFF;
					BlkExp.uiPrevAddr = 0xFFFFFFFF;
					BlkExp.uiLfNum = 0;
					ViewBlocks( uiBlkAddress, uiBlkAddress, &BlkExp);
				}
				else
				{
					bRepaint = FALSE;
				}
				break;
			case EDIT_OPTION:
			case EDIT_RAW_OPTION:
				if (!ViewEdit( uiOption == EDIT_OPTION ? TRUE : FALSE))
				{
					bRepaint = FALSE;
				}
				break;
			case HEX_OPTION:
				ViewDisable();
				bViewHexFlag = !bViewHexFlag;
				break;
			default:
				bRepaint = FALSE;
				break;
		}
	}

Exit:

	f_free( &pBlkHdr);
	ViewRestore( &SaveView);
}
