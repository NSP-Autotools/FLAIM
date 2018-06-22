//------------------------------------------------------------------------------
// Desc: This file contains global variables, typedefs, and prototypes
// 		for the VIEW program.
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

#ifndef VIEW_H
#define VIEW_H

#include "flaimsys.h"
#include "sharutil.h"

#ifdef MAIN_MODULE
	#define EXTERN
#else
	#define EXTERN extern
#endif

/* Define the area of the screen where menu items may be displayed */

#define LINES_PER_PAGE  ((gv_uiBottomLine) - (gv_uiTopLine) + 1)

/* Common options */

#define ESCAPE_OPTION            0
#define PREV_BLOCK_OPTION        1000
#define NEXT_BLOCK_OPTION        1001
#define PREV_BLOCK_IMAGE_OPTION  1002
#define GOTO_BLOCK_OPTION        1003
#define EDIT_OPTION              1004
#define HEX_OPTION               1005
#define SEARCH_OPTION            1007
#define EDIT_RAW_OPTION          1008
#define LOGICAL_INDEX_OPTION		0x40000
#define LOGICAL_CONTAINER_OPTION	0x80000
#define LFH_OPTION_ROOT_BLOCK    0x80000000
#define BLK_OPTION_CHILD_BLOCK   0x40000000
#define BLK_OPTION_DATA_BLOCK		0x20000000

EXTERN const char * Labels[]
#ifdef MAIN_MODULE
= {
	"Database Header",								// 0
	"Logical Files",									// 1
	"Prior Image Block Address",					// 2
	"Block Logical File Name",						// 3
	"Block Type",										// 4
	"B-Tree Level",									// 5
	"Block Bytes Available",						// 6
	"BLOCK ADDRESS (BLOCK HEADER)",				// 7
	"Previous Block Address",						// 8
	"Next Block Address",							// 9
	"Percent Full",									// 10
	"Block Transaction ID",							// 11
	"Little Endian Format",							// 12
	"Block Status",									// 13
	"Element Number",									// 14
	"Element Length",									// 15
	"First Element Flag",							// 16
	"Last Element Flag",								// 17
	"Previous Key Cont. Len",						// 18
	"Prev. Element Key",								// 19
	"Key Length",										// 20
	"Element Key",										// 21
	"Data Length",										// 22
	"Data",												// 23
	"Overall Data Length",							// 24
	"Domain Number",									// 25
	"Child Block Address",							// 26
	"Block Logical File Type",						// 27
	"Flaim Version",									// 28
	"Default Language",								// 29
	"Block Size",										// 30
	"First LFH Block Address",						// 31
	"Current Transaction ID",						// 32
	"Commit Count",									// 33
	"First Avail Block Address",					// 34
	"Logical End Of File",							// 35
	"LOGICAL FILE NAME",								// 36
	"Logical File Number",							// 37
	"Logical File Type",								// 38
	"Root Block Address",							// 39
	"Next Node Id",									// 40
	"Element Status",									// 41
	"(None)",											// 42
	"Field number",									// 43
	"Field type",										// 44
	"Field length",									// 45
	"Field data",										// 46
	"Field offset",									// 47
	"Field level",										// 48
	"Jump level",										// 49
	"FOP type",											// 50
	"FOP Continued Field",							// 51
	"FOP Standard",									// 52
	"FOP Open",											// 53
	"FOP Reserved",									// 54
	"FOP No Value",									// 55
	"FOP Set Level",									// 56
	"FOP Unknown Type",								// 57
	"TEXT",												// 58
	"NUMBER",											// 59
	"BINARY",											// 60
	"Field Status",									// 61
	"Unknown Type",									// 62
	"Element Offset",									// 63
	"OK",													// 64
	"Yes",												// 65
	"No",													// 66
	"Expected",											// 67
	"Element DRN",										// 68
	"Block CRC",										// 69
	"Block CRC Enabled",								// 70
	"Signature",										// 71
	"Header CRC",										// 72
	"Header Calc CRC",								// 73
	"Next Record Marker",							// 74
	"End of Log",										// 75
	"Current RFL File",								// 76
	"Current RFL Offset",							// 77
	"Last Checkpoint RFL File",					// 78
	"Last Checkpoint RFL Offset",					// 79
	"Last Checkpoint Trans ID",					// 80
	"First Checkpoint Block Addr",				// 81
	"Maximum RFL File Size",						// 82
	"Last RFL File Deleted",						// 83
	"Keep RFL Files",									// 84
	"Child Reference Count",						// 85
	"Last Backup Trans ID",							// 86
	"Blocks Changed Since Last Backup",			// 87
	"Database Serial Number",						// 88
	"Last Trans RFL Serial Number",				// 89
	"Incremental Backup Sequence Num",			// 90
	"Next RFL Serial Number",						// 91
	"Incremental Backup Serial Number",			// 92
	"Minumum RFL File Size",						// 93
	"Keep Aborted Trans. in RFL?",				// 94
	"Auto Turn Off of Keep RFL Files?",			// 95
	"Maximum File Size",								// 96
	"Last Logged Commit ID",						// 97
	"Root Block",										// 98
	"Block End",										// 99
	"Block Flags",										// 100
	"Number Of Keys",									// 101
	"Data Only Block",								// 102
	"Data Only Block Address",						// 103
	"DOM Data Type",									// 104
	"DOM Node Type",									// 105
	"DOM Flags",										// 106
	"DOM Ext Flags",									// 107
	"DOM Name ID",										// 108
	"DOM Prefix ID",									// 109
	"DOM Base ID",										// 110
	"DOM Root ID",										// 111
	"DOM Parent ID",									// 112
	"DOM Previous Sibling ID",						// 113
	"DOM Next Sibling",								// 114
	"DOM First Child ID",							// 115
	"DOM Last Child ID",								// 116
	"DOM First Attribute ID",						// 117
	"DOM LastAttribute ID",							// 118
	"DOM Annotation ID",								// 119
	"DOM Padding",										// 120
	"DOM Value",										// 121
	"Block Heap Size",								// 122
	"DOM Encryption ID",								// 123
	"DOM Ext2 Flags",									// 124
	"DOM DataLength",									// 125
	"DB Key Length",									// 126
	"DB Key",											// 127
	"Encrypted Block",								// 128
	"Encryption Id",									// 129
	"<NoData>",											// 130
	""
}
#endif
	;

enum LabelIndexes {
	LBL_DB_HEADER,													// 0
	LBL_LOGICAL_FILES,											// 1
	LBL_OLD_BLOCK_IMAGE_ADDRESS,								// 2
	LBL_BLOCK_LOGICAL_FILE_NAME,								// 3
	LBL_BLOCK_TYPE,												// 4
	LBL_B_TREE_LEVEL,												// 5
	LBL_BLOCK_BYTES_AVAIL,										// 6
	LBL_BLOCK_ADDRESS_BLOCK_HEADER,							// 7
	LBL_PREVIOUS_BLOCK_ADDRESS,								// 8
	LBL_NEXT_BLOCK_ADDRESS,										// 9
	LBL_PERCENT_FULL,												// 10
	LBL_BLOCK_TRANS_ID,											// 11
	LBL_LITTLE_ENDIAN,											// 12
	LBL_BLOCK_STATUS,												// 13
	LBL_ELEMENT_NUMBER,											// 14
	LBL_ELEMENT_LENGTH,											// 15
	LBL_FIRST_ELEMENT_FLAG,										// 16
	LBL_LAST_ELEMENT_FLAG,										// 17
	LBL_PREVIOUS_KEY_CONT_LEN,									// 18
	LBL_PREV_ELEMENT_KEY,										// 19
	LBL_KEY_LENGTH,												// 20
	LBL_ELEMENT_KEY,												// 21
	LBL_DATA_LENGTH,												// 22
	LBL_DATA,														// 23
	LBL_OA_DATA_LENGTH,											// 24
	LBL_DOMAIN_NUMBER,											// 25
	LBL_CHILD_BLOCK_ADDRESS,									// 26
	LBL_BLOCK_LOGICAL_FILE_TYPE,								// 27
	LBL_FLAIM_VERSION,											// 28
	LBL_DEFAULT_LANGUAGE,										// 29
	LBL_BLOCK_SIZE,												// 30
	LBL_FIRST_LFH_BLOCK_ADDRESS,								// 31
	LBL_CURRENT_TRANS_ID,										// 32
	LBL_COMMIT_COUNT,												// 33
	LBL_FIRST_AVAIL_BLOCK_ADDRESS,							// 34
	LBL_LOGICAL_END_OF_FILE,									// 35
	LBL_LOGICAL_FILE_NAME,										// 36
	LBL_LOGICAL_FILE_NUMBER,									// 37
	LBL_LOGICAL_FILE_TYPE,										// 38
	LBL_ROOT_BLOCK_ADDRESS,										// 39
	LBL_NEXT_NODE_ID,												// 40
	LBL_ELEMENT_STATUS,											// 41
	LBL_NONE,														// 42
	LBL_FIELD_NUMBER,												// 43
	LBL_FIELD_TYPE,												// 44
	LBL_FIELD_LENGTH,												// 45
	LBL_FIELD_DATA,												// 46
	LBL_FIELD_OFFSET,												// 47
	LBL_FIELD_LEVEL,												// 48
	LBL_JUMP_LEVEL,												// 49
	LBL_FOP_TYPE,													// 50
	LBL_FOP_CONT,													// 51
	LBL_FOP_STD,													// 52
	LBL_FOP_OPEN,													// 53
	LBL_FOP_RESERVED,												// 54
	LBL_FOP_NO_VALUE,												// 55
	LBL_FOP_SET_LEVEL,											// 56
	LBL_FOP_BAD,													// 57
	LBL_TYPE_TEXT,													// 58
	LBL_TYPE_NUMBER,												// 59
	LBL_TYPE_BINARY,												// 60
	LBL_FIELD_STATUS,												// 61
	LBL_TYPE_UNKNOWN,												// 62
	LBL_ELEMENT_OFFSET,											// 63
	LBL_OK,															// 64
	LBL_YES,															// 65
	LBL_NO,															// 66
	LBL_EXPECTED,													// 67
	LBL_ELEMENT_DRN,												// 68
	LBL_BLOCK_CRC,													// 69
	LBL_BLK_CRC_ENABLED,											// 70
	LBL_SIGNATURE,													// 71
	LBL_HDR_CRC,													// 72
	LBL_CALC_HDR_CRC,												// 73
	LBL_NEXT_DRN_MARKER,											// 74
	LBL_END_OF_LOG_ADDRESS,										// 75
	LBL_RFL_FILE_NUM,												// 76
	LBL_RFL_LAST_TRANS_OFFSET,									// 77
	LBL_RFL_LAST_CP_FILE_NUM,									// 78
	LBL_RFL_LAST_CP_OFFSET,										// 79
	LBL_LAST_CP_ID,												// 80
	LBL_FIRST_CP_BLK_ADDR,										// 81
	LBL_RFL_MAX_FILE_SIZE,										// 82
	LBL_LAST_RFL_FILE_DELETED,									// 83
	LBL_KEEP_RFL_FILES,											// 84
	LBL_CHILD_REFERENCE_COUNT,									// 85
	LBL_LAST_BACKUP_TRANS_ID,									// 86
	LBL_BLK_CHG_SINCE_BACKUP,									// 87
	LBL_DB_SERIAL_NUM,											// 88
	LBL_LAST_TRANS_RFL_SERIAL_NUM,							// 89
	LBL_INC_BACKUP_SEQ_NUM,										// 90
	LBL_RFL_NEXT_SERIAL_NUM,									// 91
	LBL_INC_BACKUP_SERIAL_NUM,									// 92
	LBL_RFL_MIN_FILE_SIZE,										// 93
	LBL_KEEP_ABORTED_TRANS_IN_RFL_FILES,					// 94
	LBL_AUTO_TURN_OFF_KEEP_RFL,								// 95
	LBL_MAX_FILE_SIZE,											// 96
	LBL_LAST_RFL_COMMIT_ID,										// 97
	LBL_BLOCK_ROOT,												// 98
	LBL_BLOCK_END,													// 99
	LBL_BLOCK_FLAGS,												// 100
	LBL_BLOCK_NUM_KEYS,											// 101
	LBL_DATA_BLOCK_FLAG,											// 102
	LBL_DATA_BLOCK_ADDRESS,										// 103
	LBL_DOM_DATA_TYPE,											// 104
	LBL_DOM_NODE_TYPE,											// 105
	LBL_DOM_FLAGS,													// 106
	LBL_DOM_EXT_FLAGS,											// 107
	LBL_DOM_NAME_ID,												// 108
	LBL_DOM_PREFIX_ID,											// 109
	LBL_DOM_BASE_ID,												// 110
	LBL_DOM_ROOT_ID,												// 111
	LBL_DOM_PARENT_ID,											// 112
	LBL_DOM_PREV_SIB_ID,											// 113
	LBL_DOM_NEXT_SIB_ID,											// 114
	LBL_DOM_FIRST_CHILD_ID,										// 115
	LBL_DOM_LAST_CHILD_ID,										// 116
	LBL_DOM_FIRST_ATTR_ID,										// 117
	LBL_DOM_LAST_ATTR_ID,										// 118
	LBL_DOM_ANNOTATION_ID,										// 119
	LBL_DOM_PADDING,												// 120
	LBL_DOM_VALUE,													// 121
	LBL_BLOCK_HEAP_SIZE,											// 122
	LBL_DOM_ENCDEF_ID,											// 123
	LBL_DOM_EXT2_FLAGS,											// 124
	LBL_DOM_DATA_LENGTH,											// 125
	LBL_HDR_KEY_LEN,												// 126
	LBL_HDR_DB_KEY,												// 127
	LBL_BLOCK_ENCRYPTED,											// 128
	LBL_ENCRYPTION_ID,											// 129
	LBL_NO_VALUE													// 130
};

#define NUM_STATUS_BYTES   (FLM_NUM_CORRUPT_ERRORS / 8 + 1)

typedef struct View_Menu_Item  *VIEW_MENU_ITEM_p;

typedef struct View_Menu_Item
{
	FLMUINT		uiItemNum;
	FLMUINT		uiCol;
	FLMUINT		uiRow;
	FLMUINT		uiOption;
	eColorType	uiUnselectBackColor;
	eColorType	uiUnselectForeColor;
	eColorType	uiSelectBackColor;
	eColorType	uiSelectForeColor;
	FLMINT		iLabelIndex;	// Signed number
	FLMUINT		uiLabelWidth;
	FLMUINT		uiHorizCurPos;
	FLMUINT		uiValueType;

	// Lower four bits contain data type

#define				VAL_IS_LABEL_INDEX	0x01
#define				VAL_IS_ERR_INDEX		0x02
#define				VAL_IS_TEXT_PTR		0x03
#define				VAL_IS_BINARY_PTR		0x04
#define				VAL_IS_BINARY			0x05
#define				VAL_IS_BINARY_HEX		0x06
#define				VAL_IS_NUMBER			0x07
#define				VAL_IS_EMPTY			0x08

	// Upper four bits contain display format for numbers

#define				DISP_DECIMAL			0x00
#define				DISP_HEX					0x10
#define				DISP_HEX_DECIMAL		0x20
#define				DISP_DECIMAL_HEX		0x30

	FLMUINT64					ui64Value;
	FLMUINT						uiValueLen;
	VIEW_MENU_ITEM_p			pNextItem;
	VIEW_MENU_ITEM_p			pPrevItem;

	// Modification parameters

	FLMUINT		uiModFileNumber;	// Number of block file value is in.
	FLMUINT		uiModFileOffset;	// Zero means it cannot be modified
	FLMUINT		uiModBufLen;		// For binary only
	FLMUINT		uiModType;

	// Lower four bits contains modification type

#define				MOD_FLMUINT32		0x01
#define				MOD_FLMUINT16		0x02
#define				MOD_FLMBYTE			0x03
#define				MOD_BINARY			0x04
#define				MOD_TEXT				0x05
#define				MOD_LANGUAGE		0x06
#define				MOD_CHILD_BLK		0x07
#define				MOD_BITS				0x08
#define				MOD_FLMUINT64		0x0A
#define				MOD_DATA_BLK		0x0B
#define				MOD_SEN5				0x0C
#define				MOD_SEN9				0x0D

	// Upper four bits contains how number is to be entered

#define				MOD_HEX				0x10
#define				MOD_DECIMAL			0x20
#define				MOD_NATIVE			0x40
#define				MOD_DISABLED		0xF0

} VIEW_MENU_ITEM;

typedef struct BLK_EXP
{
	FLMUINT	uiBlkAddr;
	FLMUINT	uiType;
	FLMUINT	uiLfNum;
	FLMUINT	uiNextAddr;
	FLMUINT	uiPrevAddr;
	FLMUINT	uiLevel;
} BLK_EXP;

typedef struct BLK_EXP  *BLK_EXP_p;

typedef struct VIEW_INFO
{
	FLMUINT		uiCurrItem;
	FLMUINT		uiTopRow;
	FLMUINT		uiBottomRow;
	FLMUINT		uiCurrFileNumber;
	FLMUINT		uiCurrFileOffset;
} VIEW_INFO;

typedef struct VIEW_INFO  *VIEW_INFO_p;

#define HAVE_HORIZ_CUR(vm)    ((((vm)->uiValueType & 0x0F) == VAL_IS_BINARY_PTR) || \
										 (((vm)->uiValueType & 0x0F) == VAL_IS_BINARY_HEX))
#define HORIZ_SIZE(vm)           ((vm)->uiValueLen)
#define MAX_HORIZ_SIZE(Col)      ((79 - ((Col) + 4)) / 4)

#define COLLECTION_STRING	"Collection"
#define INDEX_STRING			"Index"

// Global variables

EXTERN FLMBOOL						gv_bViewPoppingStack
#ifdef MAIN_MODULE
	= FALSE
#endif
	;
EXTERN FLMBOOL						gv_bViewSearching
#ifdef MAIN_MODULE
	= FALSE
#endif
	;
EXTERN IF_Db *						gv_hViewDb
#ifdef MAIN_MODULE
	= NULL
#endif
	;
EXTERN FLMBOOL						gv_bViewDbInitialized
#ifdef MAIN_MODULE
	= FALSE
#endif
	;
EXTERN FLMBOOL						gv_bShutdown;
EXTERN FLMBOOL						gv_bRunning;
EXTERN FLMBOOL						gv_bViewExclusive;
EXTERN VIEW_MENU_ITEM_p			gv_pViewSearchItem;
EXTERN FLMUINT						gv_uiViewSearchLfNum;
EXTERN FLMUINT						gv_uiViewSearchLfType;
EXTERN FLMBYTE						gv_ucViewSearchKey[ XFLM_MAX_KEY_SIZE];
EXTERN FLMUINT						gv_uiViewSearchKeyLen;
EXTERN F_Pool *					gv_pViewPool;
EXTERN FLMUINT						gv_uiViewTopRow;
EXTERN FLMUINT						gv_uiViewBottomRow;
EXTERN F_TMSTAMP					gv_ViewLastTime;
EXTERN XFLM_DB_HDR				gv_ViewDbHdr;
EXTERN char							gv_szFlaimName [10];
EXTERN char							gv_szFlaimVersion [10];
EXTERN FLMBYTE						gv_uiPcodeAddr;
EXTERN char							gv_szViewFileName[ F_PATH_MAX_SIZE];
EXTERN char							gv_szDataDir[ F_PATH_MAX_SIZE];
EXTERN char							gv_szRflDir[ F_PATH_MAX_SIZE];
EXTERN F_SuperFileHdl *			gv_pSFileHdl;
EXTERN FLMBOOL						gv_bViewFileOpened;
EXTERN FLMBOOL						gv_bViewHaveDictInfo;
EXTERN char							gv_szPassword[ 80];
EXTERN FLMBOOL						gv_bViewOkToUsePassword;
EXTERN FLMUINT						gv_bViewFixHeader;
EXTERN XFLM_CREATE_OPTS				gv_ViewFixOptions;
EXTERN FLMBOOL						gv_bViewHdrRead;
EXTERN FLMUINT						gv_bViewEnabled
#ifdef MAIN_MODULE
	= TRUE
#endif
	;

EXTERN FLMUINT						gv_uiViewCurrFileNumber
#ifdef MAIN_MODULE
	= 0
#endif
	;

#define	VIEW_INVALID_FILE_OFFSET		(0xFFFFFFFF)

EXTERN FLMUINT						gv_uiViewCurrFileOffset
#ifdef MAIN_MODULE
	= 0
#endif
	;

EXTERN FLMUINT						gv_uiViewLastFileNumber
#ifdef MAIN_MODULE
	= 0xFFFFFFFF
#endif
	;

EXTERN FLMUINT						gv_uiViewLastFileOffset
#ifdef MAIN_MODULE
	= 0xFFFFFFFF
#endif
	;

EXTERN FLMUINT						gv_uiViewMenuCurrItemNum
#ifdef MAIN_MODULE
	= 0
#endif
	;

EXTERN VIEW_MENU_ITEM_p			gv_pViewMenuCurrItem
#ifdef MAIN_MODULE
	= NULL
#endif
	;

EXTERN VIEW_MENU_ITEM_p			gv_pViewMenuFirstItem
#ifdef MAIN_MODULE
	= NULL
#endif
	;

EXTERN VIEW_MENU_ITEM_p			gv_pViewMenuLastItem
#ifdef MAIN_MODULE
	= NULL
#endif
	;

EXTERN IF_DbSystem *				gv_pDbSystem
#ifdef MAIN_MODULE
	= NULL
#endif
	;

// Function prototypes

#ifdef FLM_NLM
	#define viewGiveUpCPU()			f_yieldCPU()
#else
	#define viewGiveUpCPU()			f_sleep( 10)
#endif

RCODE ViewGetDictInfo( void);			// Source: view.cpp

void ViewReadHdr(							// Source: view.cpp
	FLMUINT32 *	pui32CalcCRC = NULL);

void ViewShowError(						// Source: viewdisp.cpp
	const char *	pszMessage);

void ViewShowRCError(					// Source: viewdisp.cpp
	const char *	pszWhat,
	RCODE				rc);

void ViewFreeMenuMemory( void);		// Source: viewmenu.cpp

void ViewMenuInit(						// Source: viewmenu.cpp
	const char *	pszTitle);

FLMBOOL ViewAddMenuItem(				// Source: viewmenu.cpp
	FLMINT		iLabelIndex,
	FLMUINT		uiLabelWidth,
	FLMUINT		uiValueType,
	FLMUINT64	ui64Value,
	FLMUINT		uiValueLen,
	FLMUINT		uiModFileNumber,
	FLMUINT		uiModFileOffset,
	FLMUINT		uiModBufLen,
	FLMUINT		uiModType,
	FLMUINT		uiCol,
	FLMUINT		uiRow,
	FLMUINT		uiOption,
	eColorType	uiUnselectBackColor,
	eColorType	uiUnselectForeColor,
	eColorType	uiSelectBackColor,
	eColorType	uiSelectForeColor);

FLMUINT ViewGetMenuOption( void);	// Source: viewmenu.cpp

void ViewUpdateDate(						// Source: viewmenu.cpp
	FLMBOOL		bUpdateFlag,
	F_TMSTAMP *	pLastTime);

void ViewEscPrompt( void);				// Source: viewmenu.cpp

void ViewReset(							// Source: viewmenu.cpp
	VIEW_INFO_p pSaveView);

void ViewRestore(							// Source: viewmenu.cpp
	VIEW_INFO_p	pSaveView);

void ViewDisable( void);				// Source: viewmenu.cpp

void ViewEnable( void);					// Source: viewmenu.cpp

void ViewDbHeader( void);				// Source: viewhdr.cpp

void ViewLogicalFile(					// Source: viewlfil.cpp
	FLMUINT	uiLfNum,
	FLMUINT	uiLfType);

void ViewLogicalFiles( void);			// Source: viewlfil.cpp

FLMBOOL ViewBlkRead(						// Source: viewblk.cpp
	FLMUINT			uiBlkAddress,
	F_BLK_HDR **	ppBlkHdr,
	FLMBOOL			bOkToConvert,
	FLMUINT			uiReadLen,
	FLMUINT32 *		pui32CalcCRC,
	FLMUINT32 *		pu32BlkCRC,
	FLMUINT *		puiBytesRead,
	FLMBOOL			bShowPartialReadError);

FLMBOOL ViewGetLFH(						// Source: viewblk.cpp
	FLMUINT			uiLfNum,
	eLFileType		eLfType,
	F_LF_HDR *		pLfHdr,
	FLMUINT *		puiFileOffset);

FLMBOOL ViewGetLFName(					// Source: viewblk.cpp
	char *			pszName,
	FLMUINT			uiLfNum,
	eLFileType		eLfType,
	F_LF_HDR *		pLfHdr,
	FLMUINT *		puiFileOffset);

FLMBOOL ViewOutBlkHdr(					// Source: viewblk.cpp
	FLMUINT		uiCol,
	FLMUINT *	puiRow,
	F_BLK_HDR *	pBlkHdr,
	BLK_EXP_p	pBlkExp,
	FLMBYTE *	pucBlkStatus,
	FLMUINT32	ui32CalcCRC,
	FLMUINT32	ui32BlkCRC);

FLMBOOL ViewAvailBlk(					// Source: viewblk.cpp
	FLMUINT			uiReadAddress,
	FLMUINT			uiBlkAddress,
	F_BLK_HDR **	ppBlkHdr,
	BLK_EXP_p		pBlkExp);

FLMBOOL ViewLeafBlk(						// Source: viewblk.cpp
	FLMUINT			uiReadAddress,
	FLMUINT			uiBlkAddress,
	F_BLK_HDR **	ppBlkHdr,
	BLK_EXP_p		pBlkExp);

FLMBOOL ViewDataBlk(						// Source: viewblk.cpp
	FLMUINT			uiReadAddress,
	FLMUINT			uiBlkAddress,
	F_BLK_HDR **	ppBlkHdr,
	BLK_EXP_p		pBlkExp);

FLMBOOL ViewNonLeafBlk(					// Source: viewblk.cpp
	FLMUINT			uiReadAddress,
	FLMUINT			uiBlkAddress,
	F_BLK_HDR **	ppBlkHdr,
	BLK_EXP_p		pBlkExp);

void ViewBlocks(							// Source: viewblk.cpp
	FLMUINT			uiReadAddress,
	FLMUINT			uiBlkAddress,
	BLK_EXP_p		pBlkExp);

FLMBOOL GetBlockAddrType(				// Source: viewblk.cpp
	FLMUINT *	puiBlkAddress);

void ViewHexBlock(						// Source: viewblk.cpp
	FLMUINT			uiReadAddress,
	F_BLK_HDR **	ppBlkHdr,
	FLMUINT			uiViewLen);

FLMBOOL OutputHexValue(					// Source: viewblk.cpp
	FLMUINT		uiCol,
	FLMUINT *	puiRow,
	FLMUINT		uiBackColor,
	FLMUINT		uiForeColor,
	FLMINT		iLabelIndex,
	FLMUINT		uiFileNumber,
	FLMUINT		uiFileOffset,
	FLMBYTE *	pucVal,
	FLMUINT		uiValLen,
	FLMBOOL		bCopyVal);

FLMBOOL ViewLFHBlk(						// Source: viewlfil.cpp
	FLMUINT			uiReadAddress,
	FLMUINT			uiTargBlkAddress,
	F_BLK_HDR **	ppBlkHdr,
	BLK_EXP_p		pBlkExp);

void FormatLFType(						// Source: viewlfil.cpp
	char *			pszDestBuf,
	FLMUINT			uiLfType);

void ViewAskInput(						// Source: view.cpp
	const char *	pszPrompt,
	char *			pszBuffer,
	FLMUINT			uiBufLen);

FLMBOOL ViewEdit(							// Source: viewedit.cpp
	FLMBOOL	bWriteEntireBlock);

FLMBOOL ViewGetNum(						// Source: viewedit.cpp
	const char *	pszPrompt,
	void *			pvNum,
	FLMBOOL			bEnterHexFlag,
	FLMUINT			uiNumBytes,
	FLMUINT64		ui64MaxValue,
	FLMBOOL *		pbValEntered);

FLMBOOL ViewEditNum(						// Source: viewedit.cpp
	void *			pvNum,
	FLMBOOL			bEnterHexFlag,
	FLMUINT			uiNumBytes,
	FLMUINT64		ui64MaxValue);

FLMBOOL ViewEditText(					// Source: viewedit.cpp
	const char *	pszPrompt,
	char *			pszText,
	FLMUINT			uiTextLen,
	FLMBOOL *		pbValEntered);

FLMBOOL ViewEditBinary(					// Source: viewedit.cpp
	const char *	pszPrompt,
	FLMBYTE *		pucBuf,
	FLMUINT *		puiByteCount,
	FLMBOOL *		pbValEntered);

FLMBOOL ViewEditLanguage(				// Source: viewedit.cpp
	FLMUINT *		puiLang);

FLMBOOL ViewEditBits(					// Source: viewedit.cpp
	FLMBYTE *		pucBit,
	FLMBOOL			bEnterHexFlag,
	FLMBYTE			ucMask);

FLMBOOL ViewGetKey( void);				// Source: viewsrch.cpp

void ViewSearch( void);					// Source: viewsrch.cpp

#endif

