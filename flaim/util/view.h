//-------------------------------------------------------------------------
// Desc: Database viewer utility - definitions.
// Tabs: 3
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

#ifndef VIEW_H
#define VIEW_H

#include "flaimsys.h"

#ifdef MAIN_MODULE
	#define EXTERN
#else
	#define EXTERN extern
#endif

// Define the area of the screen where menu items may be displayed

#define LINES_PER_PAGE	((gv_uiBottomLine) - (gv_uiTopLine) + 1)

// Common options

#define ESCAPE_OPTION				0
#define PREV_BLOCK_OPTION			1000
#define NEXT_BLOCK_OPTION			1001
#define PREV_BLOCK_IMAGE_OPTION	1002
#define GOTO_BLOCK_OPTION			1003
#define EDIT_OPTION					1004
#define HEX_OPTION					1005
#define DECRYPT_OPTION				1006
#define SEARCH_OPTION				1007
#define EDIT_RAW_OPTION				1008
#define LOGICAL_FILE_OPTION		0x8000
#define LFH_OPTION_ROOT_BLOCK		0x4000
#define LFH_OPTION_LAST_BLOCK		0x2000
#define BLK_OPTION_CHILD_BLOCK	0x1000

EXTERN const char * Labels[]
#ifdef MAIN_MODULE
= {
	"File Header",								  /*0*/
	"Log Header",								  /*1*/
	"PCode",										  /*2*/
	"Logical Files",							  /*3*/
	"Old Block Image Address",				  /*4*/
	"Block Logical File Name",				  /*5*/
	"Block Type",								  /*6*/
	"B-Tree Level",							  /*7*/
	"Block End",								  /*8*/
	"BLOCK ADDRESS (BLOCK HEADER)",		  /*9*/
	"Previous Block Address",				  /*10*/
	"Next Block Address",					  /*11*/
	"Block Index Container",				  /*12*/
	"Percent Full",							  /*13*/
	"Block Transaction ID",					  /*14*/
	"Block Encrypted",						  /*15*/
	"Old Block Image Transaction ID",	  /*16*/
	"Block Status",							  /*17*/
	"Element Number",							  /*18*/
	"Element Length",							  /*19*/
	"First Element Flag",					  /*20*/
	"Last Element Flag",						  /*21*/
	"Previous Key Cont. Len",				  /*22*/
	"Prev. Element Key",						  /*23*/
	"Key Length",								  /*24*/
	"Element Key",								  /*25*/
	"Record Length",							  /*26*/
	"Record",									  /*27*/
	"Domain Present Flag",					  /*28*/
	"Domain Number",							  /*29*/
	"Child Block Address",					  /*30*/
	"Block Logical File Type",				  /*31*/
	"Flaim Name",								  /*32*/
	"Flaim Version",							  /*33*/
	"PCODE Data",								  /*34*/
	"Default Language",						  /*35*/
	"Block Size",								  /*36*/
	"Initial Log Segment Size",			  /*37*/
	"Log Segment Extent Size",				  /*38*/
	"Initial Log Segment Address",		  /*39*/
	"Log Header Address",					  /*40*/
	"First LFH Block Address",				  /*41*/
	"First PCODE Block Address",			  /*42*/
	"Encryption Version",					  /*43*/
	"First Log Segment Extent Address",	  /*44*/
	"Last Log Segment Extent Address",	  /*45*/
	"Start of Log Segment Address",		  /*46*/
	"Start of Log Segment Offset",		  /*47*/
	"End of Log Segment Address",			  /*48*/
	"End of Log Segment Offset",			  /*49*/
	"Last Transaction ID",					  /*50*/
	"Current Transaction ID",				  /*51*/
	"Commit Count",							  /*52*/
	"Data Container Record Count",		  /*53*/
	"Data Container Next Record",			  /*54*/
	"Data Container Last Block Address",  /*55*/
	"Dict Container Record Count",		  /*56*/
	"Dict Container Next Record",			  /*57*/
	"Dict Container Last Block Address",  /*58*/
	"First Avail Block Address",			  /*59*/
	"Logical End Of File",					  /*60*/
	"Transaction Active",					  /*61*/
	"LOGICAL FILE NAME",						  /*62*/
	"Logical File Number",					  /*63*/
	"Logical File Type",						  /*64*/
	"Index Container",						  /*65*/
	"Root Block Address",					  /*66*/
	"Last Block Address",					  /*67*/
	"B-Tree Levels",							  /*68*/
	"Next DRN",									  /*69*/
	"Logical File Status",					  /*70*/
	"Logical File Block Size",				  /*71*/
	"Update Seq Number",						  /*72*/
	"Min Fill",									  /*73*/
	"Max Fill",									  /*74*/
	"Maximum Number of DNA Entries",		  /*75*/
	"ISK Count",								  /*76*/
	"FPL Count",								  /*77*/
	"LFD Count",								  /*78*/
	"Field/Domain Count",					  /*79*/
	"Field",										  /*80*/
	"Container",								  /*81*/
	"Index",										  /*82*/
	"Index Language",							  /*83*/
	"Index Attributes",						  /*84*/
	"Index Field Count",						  /*85*/
	"Field Path",								  /*86*/
	"Field Index Attributes",				  /*87*/
	"Element Status",							  /*88*/
	"(None)",									  /*89*/
	"Block modified in transaction",		  /*90*/
	"Field number",							  /*91*/
	"Field type",								  /*92*/
	"Field length",							  /*93*/
	"Field data",								  /*94*/
	"Field offset",							  /*95*/
	"Field level",								  /*96*/
	"Jump level",								  /*97*/
	"FOP type",									  /*98*/
	"FOP Continued Field",					  /*99*/
	"FOP Standard",							  /*100*/
	"FOP Open",									  /*101*/
	"FOP Tagged",								  /*102*/
	"FOP No Value",							  /*103*/
	"FOP Set Level",							  /*104*/
	"FOP Unknown Type",						  /*105*/
	"TEXT",										  /*106*/
	"NUMBER",									  /*107*/
	"BINARY",									  /*108*/
	"CONTEXT",									  /*109*/
	"REAL",										  /*110*/
	"DATE",										  /*111*/
	"TIME",										  /*112*/
	"TMSTAMP",									  /*113*/
	"Field Status",							  /*114*/
	"Could Not Read PCODE",					  /*115*/
	"Element Offset",							  /*116*/
	"Avail Block Count",						  /*117*/
	"Backchain Count",						  /*118*/
	"Backchain Block Address",				  /*119*/
	"OK",											  /*120*/
	"Yes",										  /*121*/
	"No",											  /*122*/
	"Expected",									  /*123*/
	"Element DRN",								  /*124*/
	"Low Checksum Byte",						  /*125*/
	"Encryption Block",						  /*126*/
	"Sync Checkpoint",						  /*127*/
	"Product Code",							  /*128*/
	"File Type",								  /*129*/
	"Major Version",							  /*130*/
	"Minor Version",							  /*131*/
	"RFL Max Size",							  /*132*/
	"RFL Sequence",							  /*133*/
	"RFL Options",								  /*134*/
	"High Checksum Byte",					  /*135*/
	"Header Checksum",						  /*136*/
	"Header Calc Checksum",					  /*137*/
	"Sync Checksum",							  /*138*/
	"Sync Calc Checksum",					  /*139*/
	"Maint. In Progress",					  /*140*/
	"Maximum Occurrences",					  /*141*/
	"Record Template",						  /*142*/
	"FOP Next Drn",							  /*143*/
	"Field ID",									  /*144*/
	"Next Record Marker",					  /*145*/
	"Stamped",									  /*146*/
	"Value Required",							  /*147*/
	"Shared Dict Version",					  /*148*/
	"Shared Dict Num",						  /*149*/
	"Store Number",							  /*150*/
	"Guardian File Name Len",				  /*151*/
	"Guardian File Name",					  /*152*/
	"Guardian Password",						  /*153*/
	"Guardian Checksum",						  /*154*/
	"Guardian Calc Checksum",				  /*155*/
	"FOP Record Info",						  /*156*/
	"Global Dictionary ID",					  /*157*/
	"Init Local Dict ID",					  /*158*/
	"Index (IXD)",								  /*159*/
	"Index Field (IFD)",						  /*160*/
	"Index Field Path (IFP)",				  /*161*/
	"Record Field (RFD)",					  /*162*/
	"Area (FAREA)",							  /*163*/
	"Area Machine",							  /*164*/
	"Record Template (RTD)",				  /*165*/
	"Item Type (ITT)",						  /*166*/
	"Field Index Link (FIL)",				  /*167*/
	"Container (COD)",						  /*168*/
	"PCODE Table Type",						  /*169*/
	"Table Sub-Type",							  /*170*/
	"Table Item Count",						  /*171*/
	"Table Item Size",						  /*172*/
	"Table Extra Overhead Size",			  /*173*/
	"Table Extra Overhead",					  /*174*/
	"Table Base",								  /*175*/
	"Table High",								  /*176*/
	"Table Alloc Size",						  /*177*/
	"First IFD Offset",						  /*178*/
	"Area ID",									  /*179*/
	"IFP Offset",								  /*180*/
	"Next IFD Offset",						  /*181*/
	"Base Area ID",							  /*182*/
	"Threshold",								  /*183*/
	"Subdir Count",							  /*184*/
	"Area Flags",								  /*185*/
	"Subdir Prefix",							  /*186*/
	"RFD Offset",								  /*187*/
	"Template Flags",							  /*188*/
	"Template Field Count",					  /*189*/
	"Template Field Flags",					  /*190*/
	"Minimum Occurrences",					  /*191*/
	"Dictionary Stamp",						  /*192*/
	"Unknown Table Type",					  /*193*/
	"Field Path Count",						  /*194*/
	"Reserved",									  /*195*/
	"Area",										  /*196*/
	"Empty",										  /*197*/
	"Compound Position",							/*198*/
	"Item Type (ITT) Range",					/*199*/
	"Maintenance Sequence Number",			/*200*/
	"Pending Threshold",							/*201*/
	"End of Log",									/*202*/
	"Current RFL File",							/*203*/
	"Current RFL Offset",						/*204*/
	"Last Checkpoint RFL File",				/*205*/
	"Last Checkpoint RFL Offset",				/*206*/
	"Last Checkpoint Trans ID",				/*207*/
	"First Checkpoint Block Addr",			/*208*/
	"Maximum RFL File Size",					/*209*/
	"Last RFL File Deleted",					/*210*/
	"Keep RFL Files",								/*211*/
	"Child Reference Count",					/*212*/
	"Last Backup Trans ID",						/*213*/
	"Blocks Changed Since Last Backup",		/*214*/
	"Database Serial Number",					/*215*/
	"Last Trans RFL Serial Number",			/*216*/
	"Incremental Backup Sequence Num",		/*217*/
	"Next RFL Serial Number",					/*218*/
	"Incremental Backup Serial Number",		/*219*/
	"Minumum RFL File Size",					/*220*/
	"Keep Aborted Trans. in RFL?",			/*221*/
	"Auto Turn Off of Keep RFL Files?",		/*222*/
	"Maximum File Size (64K units)",			/*223*/
	"Last Logged Commit ID",					/*224*/
	"FOP Encrypted",								/*225*/
	"Encryption ID",								/*226*/
	"Encrypted Field Length",					/*227*/
	"Encrypted Data",								/*228*/
	""
}
#endif
	;

enum LabelIndexes {
	LBL_FILE_HEADER,											/*0*/
	LBL_LOG_HEADER,											/*1*/
	LBL_PCODE,													/*2*/
	LBL_LOGICAL_FILES,										/*3*/
	LBL_OLD_BLOCK_IMAGE_ADDRESS,							/*4*/
	LBL_BLOCK_LOGICAL_FILE_NAME,							/*5*/
	LBL_BLOCK_TYPE,											/*6*/
	LBL_B_TREE_LEVEL,											/*7*/
	LBL_BLOCK_END,												/*8*/
	LBL_BLOCK_ADDRESS_BLOCK_HEADER,						/*9*/
	LBL_PREVIOUS_BLOCK_ADDRESS,							/*10*/
	LBL_NEXT_BLOCK_ADDRESS,									/*11*/
	LBL_BLOCK_INDEX_CONTAINER,								/*12*/
	LBL_PERCENT_FULL,											/*13*/
	LBL_BLOCK_TRANS_ID,										/*14*/
	LBL_BLOCK_ENCRYPTED,										/*15*/
	LBL_OLD_BLOCK_IMAGE_TRANS_ID,							/*16*/
	LBL_BLOCK_STATUS,											/*17*/
	LBL_ELEMENT_NUMBER,										/*18*/
	LBL_ELEMENT_LENGTH,										/*19*/
	LBL_FIRST_ELEMENT_FLAG,									/*20*/
	LBL_LAST_ELEMENT_FLAG,									/*21*/
	LBL_PREVIOUS_KEY_CONT_LEN,								/*22*/
	LBL_PREV_ELEMENT_KEY,									/*23*/
	LBL_KEY_LENGTH,											/*24*/
	LBL_ELEMENT_KEY,											/*25*/
	LBL_RECORD_LENGTH,										/*26*/
	LBL_RECORD,													/*27*/
	LBL_DOMAIN_PRESENT_FLAG,								/*28*/
	LBL_DOMAIN_NUMBER,										/*29*/
	LBL_CHILD_BLOCK_ADDRESS,								/*30*/
	LBL_BLOCK_LOGICAL_FILE_TYPE,							/*31*/
	LBL_FLAIM_NAME,											/*32*/
	LBL_FLAIM_VERSION,										/*33*/
	LBL_PCODE_DATA,											/*34*/
	LBL_DEFAULT_LANGUAGE,									/*35*/
	LBL_BLOCK_SIZE,											/*36*/
	LBL_INITIAL_LOG_SEGMENT_SIZE,							/*37*/
	LBL_LOG_SEGMENT_EXTENT_SIZE,							/*38*/
	LBL_INITIAL_LOG_SEGMENT_ADDRESS,						/*39*/
	LBL_LOG_HEADER_ADDRESS,									/*40*/
	LBL_FIRST_LFH_BLOCK_ADDRESS,							/*41*/
	LBL_FIRST_PCODE_BLOCK_ADDRESS,						/*42*/
	LBL_ENCRYPTION_VERSION,									/*43*/
	LBL_FIRST_LOG_SEGMENT_EXTENT_ADDRESS,				/*44*/
	LBL_LAST_LOG_SEGMENT_EXTENT_ADDRESS,				/*45*/
	LBL_START_OF_LOG_SEGMENT_ADDRESS,					/*46*/
	LBL_START_OF_LOG_SEGMENT_OFFSET,						/*47*/
	LBL_END_OF_LOG_SEGMENT_ADDRESS,						/*48*/
	LBL_END_OF_LOG_SEGMENT_OFFSET,						/*49*/
	LBL_LAST_TRANSACTION_ID,								/*50*/
	LBL_CURRENT_TRANS_ID,									/*51*/
	LBL_COMMIT_COUNT,											/*52*/
	LBL_DATA_CONTAINER_RECORD_COUNT,						/*53*/
	LBL_DATA_CONTAINER_NEXT_RECORD,						/*54*/
	LBL_DATA_CONTAINER_LAST_BLOCK_ADDRESS,				/*55*/
	LBL_DICT_CONTAINER_RECORD_COUNT,						/*56*/
	LBL_DICT_CONTAINER_NEXT_RECORD,						/*57*/
	LBL_DICT_CONTAINER_LAST_BLOCK_ADDRESS,				/*58*/
	LBL_FIRST_AVAIL_BLOCK_ADDRESS,						/*59*/
	LBL_LOGICAL_END_OF_FILE,								/*60*/
	LBL_TRANSACTION_ACTIVE,									/*61*/
	LBL_LOGICAL_FILE_NAME,									/*62*/
	LBL_LOGICAL_FILE_NUMBER,								/*63*/
	LBL_LOGICAL_FILE_TYPE,									/*64*/
	LBL_INDEX_CONTAINER,										/*65*/
	LBL_ROOT_BLOCK_ADDRESS,									/*66*/
	LBL_LAST_BLOCK_ADDRESS,									/*67*/
	LBL_B_TREE_LEVELS,										/*68*/
	LBL_NEXT_DRN,												/*69*/
	LBL_LOGICAL_FILE_STATUS,								/*70*/
	LBL_LOGICAL_FILE_BLOCK_SIZE,							/*71*/
	LBL_UPDATE_SEQ_NUMBER,									/*72*/
	LBL_MIN_FILL,												/*73*/
	LBL_MAX_FILL,												/*74*/
	LBL_MAXIMUM_NUMBER_OF_DNA_ENTRIES,					/*75*/
	LBL_ISK_COUNT,												/*76*/
	LBL_FPL_COUNT,												/*77*/
	LBL_LFD_COUNT,												/*78*/
	LBL_FIELD_DOMAIN_COUNT,									/*79*/
	LBL_FIELD,													/*80*/
	LBL_CONTAINER,												/*81*/
	LBL_INDEX,													/*82*/
	LBL_INDEX_LANGUAGE,										/*83*/
	LBL_INDEX_ATTRIBUTES,									/*84*/
	LBL_INDEX_FIELD_COUNT,									/*85*/
	LBL_FIELD_PATH,											/*86*/
	LBL_FIELD_INDEX_ATTRIBUTES,							/*87*/
	LBL_ELEMENT_STATUS,										/*88*/
	LBL_NONE,													/*89*/
	LBL_BLOCK_MODIFIED,										/*90*/
	LBL_FIELD_NUMBER,											/*91*/
	LBL_FIELD_TYPE,											/*92*/
	LBL_FIELD_LENGTH,											/*93*/
	LBL_FIELD_DATA,											/*94*/
	LBL_FIELD_OFFSET,											/*95*/
	LBL_FIELD_LEVEL,											/*96*/
	LBL_JUMP_LEVEL,											/*97*/
	LBL_FOP_TYPE,												/*98*/
	LBL_FOP_CONT,												/*99*/
	LBL_FOP_STD,												/*100*/
	LBL_FOP_OPEN,												/*101*/
	LBL_FOP_TAGGED,											/*102*/
	LBL_FOP_NO_VALUE,											/*103*/
	LBL_FOP_SET_LEVEL,										/*104*/
	LBL_FOP_BAD,												/*105*/
	LBL_TYPE_TEXT,												/*106*/
	LBL_TYPE_NUMBER,											/*107*/
	LBL_TYPE_BINARY,											/*108*/
	LBL_TYPE_CONTEXT,											/*109*/
	LBL_TYPE_REAL,												/*110*/
	LBL_TYPE_DATE,												/*111*/
	LBL_TYPE_TIME,												/*112*/
	LBL_TYPE_TMSTAMP,											/*113*/
	LBL_FIELD_STATUS,											/*114*/
	LBL_TYPE_UNKNOWN,											/*115*/
	LBL_ELEMENT_OFFSET,										/*116*/
	LBL_NUM_AVAIL_BLOCKS,									/*117*/
	LBL_NUM_BACKCHAIN_BLOCKS,								/*118*/
	LBL_FIRST_BACKCHAIN_BLOCK_ADDRESS,					/*119*/
	LBL_OK,														/*120*/
	LBL_YES,														/*121*/
	LBL_NO,														/*122*/
	LBL_EXPECTED,												/*123*/
	LBL_ELEMENT_DRN,											/*124*/
	LBL_BLOCK_CHECKSUM_LOW,									/*125*/
	LBL_ENCRYPTION_BLOCK,									/*126*/
	LBL_SYNC_CHECKPOINT,										/*127*/
	LBL_PREFIX_PRODUCT,										/*128*/
	LBL_PREFIX_FILE_TYPE,									/*129*/
	LBL_PREFIX_MAJOR,											/*130*/
	LBL_PREFIX_MINOR,											/*131*/
	LBL_RFL_MAX_SIZE,											/*132*/
	LBL_RFL_SEQUENCE,											/*133*/
	LBL_RFL_OPTIONS,											/*134*/
	LBL_BLOCK_CHECKSUM_HIGH,								/*135*/
	LBL_HDR_CHECKSUM,											/*136*/
	LBL_CALC_HDR_CHECKSUM,									/*137*/
	LBL_SYNC_CHECKSUM,										/*138*/
	LBL_CALC_SYNC_CHECKSUM,									/*139*/
	LBL_MAINT_IN_PROGRESS,									/*140*/
	LBL_MAX_OCCURS,											/*141*/
	LBL_RECORD_TEMPLATE,										/*142*/
	LBL_FOP_NEXT_DRN,											/*143*/
	LBL_FIELD_ID,												/*144*/
	LBL_NEXT_DRN_MARKER,										/*145*/
	LBL_STAMPED,												/*146*/
	LBL_VALUE_REQUIRED,										/*147*/
	LBL_SH_DICT_VER,											/*148*/
	LBL_SH_DICT_NUM,											/*149*/
	LBL_STORE_NUM,												/*150*/
	LBL_GUAR_FILE_NAME_LEN,									/*151*/
	LBL_GUAR_FILE_NAME,										/*152*/
	LBL_GUAR_PASSWORD,										/*153*/
	LBL_GUAR_CHECKSUM,										/*154*/
	LBL_GUAR_CALC_CHECKSUM,									/*155*/
	LBL_FOP_REC_INFO,											/*156*/
	LBL_GLOBAL_DICT_ID,										/*157*/
	LBL_INIT_DICT_ID,											/*158*/
	LBL_IXD_TYPE,												/*159*/
	LBL_IFD_TYPE,												/*160*/
	LBL_IFP_TYPE,												/*161*/
	LBL_RFD_TYPE,												/*162*/
	LBL_AREA_TYPE,												/*163*/
	LBL_MACHINE_TYPE,											/*164*/
	LBL_RTD_TYPE,												/*165*/
	LBL_ITT_TYPE,												/*166*/
	LBL_FIL_TYPE,												/*167*/
	LBL_COD_TYPE,												/*168*/
	LBL_PCODE_TYPE,											/*169*/
	LBL_PCODE_SUBTYPE,										/*170*/
	LBL_PCODE_COUNT,											/*171*/
	LBL_PCODE_SIZE,											/*172*/
	LBL_PCODE_EXTRA_OVERHEAD_SIZE,						/*173*/
	LBL_PCODE_EXTRA_OVERHEAD,								/*174*/
	LBL_PCODE_BASE_VALUE,									/*175*/
	LBL_PCODE_HIGH_VALUE,									/*176*/
	LBL_PCODE_ALLOC_VALUE,									/*177*/
	LBL_FIRST_IFD_OFFSET,									/*178*/
	LBL_AREA_ID,												/*179*/
	LBL_FIELD_PATH_OFFSET,									/*180*/
	LBL_NEXT_IFD_OFFSET,										/*181*/
	LBL_BASE_AREA_ID,											/*182*/
	LBL_THRESHOLD,												/*183*/
	LBL_SUBDIR_COUNT,											/*184*/
	LBL_AREA_FLAGS,											/*185*/
	LBL_SUBDIR_PREFIX,										/*186*/
	LBL_RFD_OFFSET,											/*187*/
	LBL_RTD_FLAGS,												/*188*/
	LBL_RTD_FLD_CNT,											/*189*/
	LBL_RFD_FLAGS,												/*190*/
	LBL_MIN_OCCURS,											/*191*/
	LBL_DICT_STAMP,											/*192*/
	LBL_UNKNOWN_TYPE,											/*193*/
	LBL_FLD_PATH_COUNT,										/*194*/
	LBL_RESERVED,												/*195*/
	LBL_AREA,													/*196*/
	LBL_EMPTY,													/*197*/
	LBL_COMPOUND_POS,											/*198*/
	LBL_ITT_RANGE_TYPE,										/*199*/
	LBL_MAINT_SEQ_NUM,										/*200*/
	LBL_PENDING_THRESHOLD,									/*201*/
	LBL_END_OF_LOG_ADDRESS,									/*202*/
	LBL_RFL_FILE_NUM,											/*203*/
	LBL_RFL_LAST_TRANS_OFFSET,								/*204*/
	LBL_RFL_LAST_CP_FILE_NUM,								/*205*/
	LBL_RFL_LAST_CP_OFFSET,									/*206*/
	LBL_LAST_CP_ID,											/*207*/
	LBL_FIRST_CP_BLK_ADDR,									/*208*/
	LBL_RFL_MAX_FILE_SIZE,									/*209*/
	LBL_LAST_RFL_FILE_DELETED,								/*210*/
	LBL_KEEP_RFL_FILES,										/*211*/
	LBL_CHILD_REFERENCE_COUNT,								/*212*/
	LBL_LAST_BACKUP_TRANS_ID,								/*213*/
	LBL_BLK_CHG_SINCE_BACKUP,								/*214*/
	LBL_DB_SERIAL_NUM,										/*215*/
	LBL_LAST_TRANS_RFL_SERIAL_NUM,						/*216*/
	LBL_INC_BACKUP_SEQ_NUM,									/*217*/
	LBL_RFL_NEXT_SERIAL_NUM,								/*218*/
	LBL_INC_BACKUP_SERIAL_NUM,								/*219*/
	LBL_RFL_MIN_FILE_SIZE,									/*220*/
	LBL_KEEP_ABORTED_TRANS_IN_RFL_FILES,				/*221*/
	LBL_AUTO_TURN_OFF_KEEP_RFL,							/*222*/
	LBL_MAX_FILE_SIZE,										/*223*/
	LBL_LAST_RFL_COMMIT_ID,									/*224*/
	LBL_FOP_ENCRYPTED,										/*225*/
	LBL_ENC_ID,													/*226*/
	LBL_ENC_LENGTH,											/*227*/
	LBL_ENC_DATA												/*228*/
};

#define NUM_STATUS_BYTES	(FLMUINT)((FLMUINT)FLM_LAST_CORRUPT_ERROR / 8 + 1)

typedef struct View_Menu_Item	 *VIEW_MENU_ITEM_p;

typedef struct View_Menu_Item
{
	FLMUINT						ItemNum;
	FLMUINT						Col;
	FLMUINT						Row;
	FLMUINT						Option;
	eColorType					UnselectBackColor;
	eColorType					UnselectForeColor;
	eColorType					SelectBackColor;
	eColorType					SelectForeColor;
	FLMINT						iLabelIndex;	// Signed number
	FLMUINT						LabelWidth;
	FLMUINT						HorizCurPos;
	FLMUINT						ValueType;

	/* Lower four bits contain data type */

#define								VAL_IS_LABEL_INDEX	0x01
#define								VAL_IS_ERR_INDEX		0x02
#define								VAL_IS_TEXT_PTR		0x03
#define								VAL_IS_BINARY_PTR		0x04
#define								VAL_IS_BINARY			0x05
#define								VAL_IS_BINARY_HEX		0x06
#define								VAL_IS_NUMBER			0x07
#define								VAL_IS_EMPTY			0x08

	/* Upper four bits contain display format for numbers */

#define								DISP_DECIMAL			0x00
#define								DISP_HEX					0x10
#define								DISP_HEX_DECIMAL		0x20
#define								DISP_DECIMAL_HEX		0x30

	FLMUINT						Value;
	FLMUINT						ValueLen;
	VIEW_MENU_ITEM_p			NextItem;
	VIEW_MENU_ITEM_p			PrevItem;

	/* Modification parameters */

	FLMUINT						ModFileNumber;		/* Number of block file value is in. */
	FLMUINT						ModFileOffset;		/* Zero means it cannot be modified */
	FLMUINT						ModBufLen;			/* For binary only */
	FLMUINT						ModType;

	/* Lower four bits contains modification type */

#define								MOD_FLMUINT				0x01
#define								MOD_FLMUINT16			0x02
#define								MOD_FLMBYTE				0x03
#define								MOD_BINARY				0x04
#define								MOD_TEXT					0x05
#define								MOD_LANGUAGE			0x06
#define								MOD_CHILD_BLK			0x07
#define								MOD_BITS					0x08
#define								MOD_KEY_LEN				0x09
#define								MOD_BH_ADDR				0x0A
#define								MOD_BINARY_ENC			0x0B

	/* Upper four bits contains how number is to be entered */

#define								MOD_HEX					0x10
#define								MOD_DECIMAL				0x20
#define								MOD_DISABLED			0xF0

} VIEW_MENU_ITEM;

typedef struct BLK_EXP
{
	FLMUINT		BlkAddr;
	FLMUINT		Type;
	FLMUINT		LfNum;
	FLMUINT		NextAddr;
	FLMUINT		PrevAddr;
	FLMUINT		Level;
} BLK_EXP;

typedef struct BLK_EXP	*BLK_EXP_p;

typedef struct VIEW_INFO
{
	FLMINT		CurrItem;
	FLMUINT		TopRow;
	FLMUINT		BottomRow;
	FLMUINT		CurrFileNumber;
	FLMUINT		CurrFileOffset;
} VIEW_INFO;

typedef struct VIEW_INFO  *VIEW_INFO_p;

#define HAVE_HORIZ_CUR(vm)		((((vm)->ValueType & 0x0F) == VAL_IS_BINARY_PTR) || \
															 (((vm)->ValueType & 0x0F) == VAL_IS_BINARY_HEX))
#define HORIZ_SIZE(vm)				((vm)->ValueLen)
#define MAX_HORIZ_SIZE(Col)		((79 - ((Col) + 4)) / 4)

/* Global variables */

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
EXTERN HFDB							gv_hViewDb
#ifdef MAIN_MODULE
	= HFDB_NULL
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
EXTERN FLMBYTE						gv_ucViewSearchKey[ MAX_KEY_SIZ];
EXTERN FLMUINT						gv_uiViewSearchKeyLen;
EXTERN F_Pool						gv_ViewPool;
EXTERN FLMUINT						gv_uiViewTopRow;
EXTERN FLMUINT						gv_uiViewBottomRow;
EXTERN F_TMSTAMP					gv_ViewLastTime;
EXTERN HDR_INFO					gv_ViewHdrInfo;
EXTERN char							gv_szFlaimName [10];
EXTERN char							gv_szFlaimVersion [10];
EXTERN FLMUINT						gv_uiPcodeAddr;
EXTERN char							gv_szViewFileName[ F_PATH_MAX_SIZE];
EXTERN char							gv_szDataDir[ F_PATH_MAX_SIZE];
EXTERN char							gv_szRflDir[ F_PATH_MAX_SIZE];
EXTERN F_SuperFileHdl *			gv_pSFileHdl;
EXTERN FLMBOOL						gv_bViewFileOpened;
EXTERN FLMBOOL						gv_bViewHaveDictInfo;
EXTERN char							gv_szViewPassword[ 80];
EXTERN FLMBOOL						gv_bViewOkToUsePassword;
EXTERN FLMUINT						gv_bViewFixHeader;
EXTERN CREATE_OPTS				gv_ViewFixOptions;
EXTERN FLMBOOL						gv_bViewHdrRead;
EXTERN FLMBYTE						gv_ucViewLogHdr[ LOG_HEADER_SIZE];
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
EXTERN IF_FileSystem *			gv_pFileSystem
#ifdef MAIN_MODULE
	= NULL
#endif
	;

/* Function prototypes */

#ifdef FLM_NLM
	#define viewGiveUpCPU()			f_yieldCPU()
#else
	#define viewGiveUpCPU()			f_sleep( 10)
#endif

void ViewShowError(
	const char *	Message);

void ViewShowRCError(
	const char *	pszWhat,
	RCODE				rc);

void ViewFreeMenuMemory( void);

FLMINT ViewMenuInit(
	const char *	pszTitle);

FLMINT ViewAddMenuItem(
	FLMINT			LabelIndex,
	FLMUINT			LabelWidth,
	FLMUINT			ValueType,
	FLMUINT			Value,
	FLMUINT			ValueLen,
	FLMUINT			ModFileNumber,
	FLMUINT			ModFileOffset,
	FLMUINT			ModBufLen,
	FLMUINT			ModType,
	FLMUINT			Col,
	FLMUINT			Row,
	FLMUINT			Option,
	eColorType		UnselectBackColor,
	eColorType		UnselectForeColor,
	eColorType		SelectBackColor,
	eColorType		SelectForeColor);

FLMINT ViewGetMenuOption( void);

void ViewFileHeader( void);

void ViewLogHeader( void);

void ViewLogicalFile(
	FLMUINT			lfNum);

void ViewLogicalFiles( void);

void ViewUpdateDate(
	FLMUINT			UpdateFlag,
	F_TMSTAMP  *	LastTime);

FLMINT ViewBlkRead(
	FLMUINT			BlkAddress,
	FLMBYTE **		BlkPtrRV,
	FLMUINT			ReadLen,
	FLMUINT16 *		puwCalcChkSum,
	FLMUINT16 *		puwBlkChkSum,
	FLMUINT *		pwBytesReadRV,
	FLMBOOL			bShowPartialReadError,
	FLMBOOL *		pbIsEncBlock,
	FLMBOOL			bDecryptBlock,
	FLMBOOL *		pbEncrypted);

FLMINT ViewGetLFH(
	FLMUINT			lfNum,
	FLMBYTE *		lfhRV,
	FLMUINT	*		FileOffset);

FLMINT ViewGetLFName(
	FLMBYTE *		lfName,
	FLMUINT			lfNum,
	FLMBYTE *		LFH,
	FLMUINT	*		FileOffset);

FLMINT ViewOutBlkHdr(
	FLMUINT			Col,
	FLMUINT	*		RowRV,
	FLMBYTE *		BlkPtr,
	BLK_EXP_p		BlkExp,
	FLMBYTE *		BlkStatus,
	FLMUINT16		ui16CalcChkSum,
	FLMUINT16		ui16BlkChkSum);

void ViewEscPrompt( void);

FLMINT ViewLFHBlk(
	FLMUINT			ReadAddress,
	FLMUINT			TargBlkAddress,
	FLMBYTE **		BlkPtrRV,
	BLK_EXP_p		BlkExp);

FLMINT ViewAvailBlk(
	FLMUINT			ReadAddress,
	FLMUINT			BlkAddress,
	FLMBYTE **		BlkPtrRV,
	BLK_EXP_p		BlkExp);

FLMINT ViewLeafBlk(
	FLMUINT			ReadAddress,
	FLMUINT			BlkAddress,
	FLMBYTE **		BlkPtrRV,
	BLK_EXP_p		BlkExp);

FLMINT ViewNonLeafBlk(
	FLMUINT			ReadAddress,
	FLMUINT			BlkAddress,
	FLMBYTE **		BlkPtrRV,
	BLK_EXP_p		BlkExp);

void ViewBlocks(
	FLMUINT			ReadAddress,
	FLMUINT			BlkAddress,
	BLK_EXP_p		BlkExp);

void ViewReset(
	VIEW_INFO_p		SaveView);

void ViewRestore(
	VIEW_INFO_p		SaveView);

void FormatLFType(
	FLMBYTE *		DestBuf,
	FLMUINT			lfType);

FLMINT GetBlockAddrType(
	FLMUINT	*		BlkAddressRV,
	FLMUINT *		BlkTypeRV);

void ViewAskInput(
	const char *	Prompt,
	char *			Buffer,
	FLMUINT			BufLen);

void ViewGetDictInfo( void);

FLMINT ViewEdit(
	FLMUINT			WriteEntireBlock,
	FLMBOOL			bRecalcChecksum);

FLMINT viewLineEdit(
	char *			psStrRV,
	FLMUINT			iMaxLen);

void ViewReadHdr( void);

void ViewHexBlock(
	FLMUINT			ReadAddress,
	FLMBYTE **		BlkPtrRV,
	FLMBOOL			bViewDecrypted,
	FLMUINT			ViewLen);

void ViewDisable( void);

void ViewEnable( void);

FLMINT ViewGetNum(
	const char *	Prompt,
	void *			NumRV,
	FLMUINT			EnterHexFlag,
	FLMUINT			NumBytes,
	FLMUINT			MaxValue,
	FLMUINT *		ValEntered);

FLMINT ViewEditNum(
	void *			NumRV,
	FLMUINT			EnterHexFlag,
	FLMUINT			NumBytes,
	FLMUINT			MaxValue);

FLMINT ViewEditText(
	const char *	Prompt,
	char *			TextRV,
	FLMUINT			TextLen,
	FLMUINT *		ValEntered);

FLMINT ViewEditBinary(
	const char *	Prompt,
	char *			Buf,
	FLMUINT	*		ByteCountRV,
	FLMUINT *		ValEntered);

FLMINT ViewEditLanguage(
	FLMUINT *		LangRV);

FLMINT ViewEditBits(
	FLMBYTE *		BitRV,
	FLMUINT			EnterHexFlag,
	FLMBYTE			Mask);

FLMINT ViewGetKey( void);

void ViewSearch( void);

#endif
