//------------------------------------------------------------------------------
// Desc:	Routines for building collation keys
// Tabs:	3
//
// Copyright (c) 1993-2007 Novell, Inc. All Rights Reserved.
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

#include "ftksys.h"

// Collating Sequence Equates

#define COLLS						32					// first collating number (space/end of line)
#define COLS0						255				//  graphics/misc - chars without a collate value
#define COLS1						(COLLS + 9)		// quotes
#define COLS2						(COLS1 + 5)		// parens
#define COLS3						(COLS2 + 6)		// money
#define COLS4						(COLS3 + 6)		// math ops
#define COLS5						(COLS4 + 8)		// math others
#define COLS6						(COLS5 + 14)	// others: %#&@\_|~
#define COLS7						(COLS6 + 13)	// greek
#define COLS8						(COLS7 + 25)	// numbers
#define COLS9						(COLS8 + 10)	// alphabet
#define COLS10						(COLS9 + 60)	// cyrillic
#define COLS10h					(COLS9 + 42)	// hebrew - writes over european & cyrilic
#define COLS10a					(COLS10h + 28)	// arabic - inclusive from 198(C6)-252(FC)
#define COLS11						253				//	End of list - arabic goes to the end
#define COLS0_ARABIC				COLS11			// Set if arabic accent marking
#define COLS0_HEBREW				COLS11			// Set if hebrew accent marking
#define COLS_ASIAN_MARKS		0x140
#define COLS_ASIAN_MARK_VAL	0x40				// Without 0x100

#define SET_CASE_BIT				0x01
#define SET_KATAKANA_BIT		0x01
#define SET_WIDTH_BIT			0x02

#define UNK_UNICODE_CODE		0xFFFE

#define MAX_SUBCOL_BUF			(500)
#define MAX_CASE_BYTES			(150)

#define ASCTBLLEN					95
#define MNTBLLEN					219
#define SYMTBLLEN					9
#define GRKTBLLEN					219
#define CYRLTBLLEN				200
#define HEBTBL1LEN				27
#define HEBTBL2LEN				35
#define AR1TBLLEN					158
#define AR2TBLLEN					179

#define Upper_JP_A				0x2520
#define Upper_JP_Z				0x2539
#define Upper_KR_A				0x5420
#define Upper_KR_Z				0x5439
#define Upper_CS_A				0x82FC
#define Upper_CS_Z				0x8316
#define Upper_CT_A				0xA625
#define Upper_CT_Z				0xA63E

#define Lower_JP_a				0x2540
#define Lower_JP_z				0x2559
#define Lower_KR_a				0x5440
#define Lower_KR_z				0x5459
#define Lower_CS_a				0x82DC
#define Lower_CS_z				0x82F5
#define Lower_CT_a				0xA60B
#define Lower_CT_z				0xA624

// # of characters in each character set.
// CHANGING ANY OF THESE DEFINES WILL CAUSE BUGS!

#define ASC_N						95
#define ML1_N						242
#define ML2_N						145
#define BOX_N						88
#define TYP_N						103
#define ICN_N						255
#define MTH_N						238
#define MTX_N						229
#define GRK_N						219
#define HEB_N						123
#define CYR_N						250
#define KAN_N						63
#define USR_N						255
#define ARB_N						196
#define ARS_N						220

// TOTAL:	1447 WP + 255 User Characters

#define	C_N 						ASC_N + ML1_N + ML2_N + BOX_N +\
										MTH_N + MTX_N + TYP_N + ICN_N +\
										GRK_N + HEB_N + CYR_N + KAN_N +\
										USR_N + ARB_N + ARS_N

// State table constants for double character sorting

#define STATE1						1
#define STATE2						2
#define STATE3						3
#define STATE4						4
#define STATE5						5
#define STATE6						6
#define STATE7						7
#define STATE8						8
#define STATE9						9
#define STATE10					10
#define STATE11					11
#define AFTERC						12
#define AFTERH						13
#define AFTERL						14
#define INSTAE						15
#define INSTOE						16
#define INSTSG						17
#define INSTIJ						18
#define WITHAA						19

#define START_COL					12
#define START_ALL					(START_COL + 1)	// all US and european
#define START_DK					(START_COL + 2)	// Danish
#define START_IS					(START_COL + 3)	// Icelandic
#define START_NO					(START_COL + 4)	// Norwegian
#define START_SU					(START_COL + 5)	// Finnish
#define START_SV					(START_COL + 5)	// Swedish
#define START_YK					(START_COL + 6)	// Ukrain
#define START_TK					(START_COL + 7)	// Turkish
#define START_CZ					(START_COL + 8)	// Czech
#define START_SL					(START_COL + 8)	// Slovak

#define FIXUP_AREA_SIZE			24						// Number of characters to fix up

FSTATIC FLMUINT16 flmWPAsiaGetCollation(
	FLMUINT16			ui16WpChar,
	FLMUINT16			ui16NextWpChar,
	FLMUINT16   		ui16PrevColValue,
	FLMUINT16 *			pui16ColValue,
	FLMUINT16 * 		pui16SubColVal,
	FLMBYTE *			pucCaseBits,
	FLMBOOL				bUppercaseFlag);

FSTATIC FLMUINT16 flmWPGetSubCol(
	FLMUINT16			ui16WPValue,
	FLMUINT16			ui16ColValue,
	FLMUINT				uiLanguage);

FSTATIC RCODE flmWPCmbSubColBuf(
	FLMBYTE *			pucWPStr,
	FLMUINT *			puiWPStrLen,
	FLMUINT				uiMaxWPBytes,
	const FLMBYTE *	pucSubColBuf,
	FLMBOOL				bHebrewArabic,
	FLMUINT *			puiSubColBitPos);
	
FSTATIC RCODE flmAsiaParseCase(
	FLMBYTE *			pucWPStr,
	FLMUINT *			puiWPStrLen,
	FLMUINT				uiMaxWPBytes,
	const FLMBYTE *	pucCaseBits,
	FLMUINT *			puiColBytesProcessed);

// Global data

static FLMUINT16 *	gv_pUnicodeToWP60 = NULL;
static FLMUINT16 *	gv_pWP60ToUnicode = NULL;
static FLMUINT			gv_uiMinUniChar = 0;
static FLMUINT			gv_uiMaxUniChar = 0;
static FLMUINT			gv_uiMinWPChar = 0;
static FLMUINT			gv_uiMaxWPChar = 0;
FLMUINT16 *				gv_pui16USCollationTable = NULL;

// Typedefs

typedef struct
{
	FLMBYTE				base;
	FLMBYTE				diacrit;
} BASE_DIACRIT_TABLE;

typedef struct
{
	FLMUINT16					char_count;		// # of characters in table
	FLMUINT16					start_char;		// start char.
	BASE_DIACRIT_TABLE *		table;

} BASE_DIACRIT;

typedef struct
{
	FLMBYTE				key;						// character key to search on
	FLMBYTE *			charPtr;					// character pointer for matched key
} TBL_B_TO_BP;

typedef struct
{
	FLMBYTE				ByteValue;
	FLMUINT16			WordValue;
} BYTE_WORD_TBL;

// Collation tables

/****************************************************************************
Desc:		Base character location table
				Bit mapped table.	(1) - corresponding base char is in same
				set as combined
				(0) - corresponding base char is in ascii set

Notes:		In the following table, the bits are numbered from left
				to right relative to each individual byte.
						EX. 00000000b   ;0-7
						bit#   01234567
****************************************************************************/
static FLMBYTE fwp_ml1_cb60[] =
{
	0x00,    // 0-7
	0x00,    // 8-15
	0x00,    // 16-23
	0x00,    // 24-31
	0x00,    // 32-39
	0x00,    // 40-47
	0x55,    // 48-55
	0x00,    // 56-63
	0x00,    // 64-71
	0x00,    // 72-79
	0x00,    // 80-87
	0x00,    // 88-95
	0x00,    // 96-103
	0x00,    // 104-111
	0x00,    // 112-119
	0x00,    // 120-127
	0x14,    // 128-135
	0x44,    // 136-143
	0x00,    // 144-151
	0x00,    // 152-159
	0x00,    // 160-167
	0x00,    // 168-175
	0x00,    // 176-183
	0x00,    // 184-191
	0x00,    // 192-199
	0x00,    // 200-207
	0x00,    // 208-215
	0x00,    // 216-223
	0x00,    // 224-231
	0x04,    // 232-239
	0x00,    // 240-241
};

/****************************************************************************
Desc:		Format of index:
				2 words before = count.
				word before = start character.
				db code for base char.
				db code for diacritic
Notes:	Diacritical char is always in same set as composed char
			base is in same set if other table indicates, else in ASCII
****************************************************************************/
static BASE_DIACRIT_TABLE fwp_ml1c_table[] =
{
	{'A',			F_ACUTE},
	{'a',			F_ACUTE},
	{'A',			F_CIRCUM},
	{'a',			F_CIRCUM},
	{'A',			F_UMLAUT},
	{'a',			F_UMLAUT},
	{'A',			F_GRAVE},
	{'a',			F_GRAVE},
	{'A',			F_RING},
	{'a',			F_RING},
	{0xff,		0xff},      // no AE diagraph
	{0xff,		0xff},      // no ae diagraph
	{'C',			F_CEDILLA},
	{'c',			F_CEDILLA},
	{'E',			F_ACUTE},
	{'e',			F_ACUTE},
	{'E',			F_CIRCUM},
	{'e',			F_CIRCUM},
	{'E',			F_UMLAUT},
	{'e',			F_UMLAUT},
	{'E',			F_GRAVE},
	{'e',			F_GRAVE},
	{'I',			F_ACUTE},
	{F_DOTLESI,	F_ACUTE},
	{'I',			F_CIRCUM},
	{F_DOTLESI,	F_CIRCUM},
	{'I',			F_UMLAUT},
	{F_DOTLESI,	F_UMLAUT},
	{'I',			F_GRAVE},
	{F_DOTLESI,	F_GRAVE},
	{'N',			F_TILDE},
	{'n',			F_TILDE},
	{'O',			F_ACUTE},
	{'o',			F_ACUTE},
	{'O',			F_CIRCUM},
	{'o',			F_CIRCUM},
	{'O',			F_UMLAUT},
	{'o',			F_UMLAUT},
	{'O',			F_GRAVE},
	{'o',			F_GRAVE},
	{'U',			F_ACUTE},
	{'u',			F_ACUTE},
	{'U',			F_CIRCUM},
	{'u',			F_CIRCUM},
	{'U',			F_UMLAUT},
	{'u',			F_UMLAUT},
	{'U',			F_GRAVE},
	{'u',			F_GRAVE},
	{'Y',			F_UMLAUT},
	{'y',			F_UMLAUT},
	{'A',			F_TILDE},
	{'a',			F_TILDE},
	{'D',			F_CROSSB},
	{'d',			F_CROSSB},
	{'O',			F_SLASH},
	{'o',			F_SLASH},
	{'O',			F_TILDE},
	{'o',			F_TILDE},
	{'Y',			F_ACUTE},
	{'y',			F_ACUTE},
	{0xff,		0xff},		// no eth
	{0xff,		0xff},		// no eth
	{0xff,		0xff},		// no Thorn
	{0xff,		0xff},		// no Thorn
	{'A',			F_BREVE},
	{'a',			F_BREVE},
	{'A',			F_MACRON},
	{'a',			F_MACRON},
	{'A',			F_OGONEK},
	{'a',			F_OGONEK},
	{'C',			F_ACUTE},
	{'c',			F_ACUTE},
	{'C',			F_CARON},
	{'c',			F_CARON},
	{'C',			F_CIRCUM},
	{'c',			F_CIRCUM},
	{'C',			F_DOTA},
	{'c',			F_DOTA},
	{'D',			F_CARON},
	{'d',			F_CARON},
	{'E',			F_CARON},
	{'e',			F_CARON},
	{'E',			F_DOTA},
	{'e',			F_DOTA},
	{'E',			F_MACRON},
	{'e',			F_MACRON},
	{'E',			F_OGONEK},
	{'e',			F_OGONEK},
	{'G',			F_ACUTE},
	{'g',			F_ACUTE},
	{'G',			F_BREVE},
	{'g',			F_BREVE},
	{'G',			F_CARON},
	{'g',			F_CARON},
	{'G',			F_CEDILLA},
	{'g',			F_APOSAB},
	{'G',			F_CIRCUM},
	{'g',			F_CIRCUM},
	{'G',			F_DOTA},
	{'g',			F_DOTA},
	{'H',			F_CIRCUM},
	{'h',			F_CIRCUM},
	{'H',			F_CROSSB},
	{'h',			F_CROSSB},
	{'I',			F_DOTA},
	{F_DOTLESI,	F_DOTA},
	{'I',			F_MACRON},
	{F_DOTLESI,	F_MACRON},
	{'I',			F_OGONEK},
	{'i',			F_OGONEK},
	{'I',			F_TILDE},
	{F_DOTLESI,	F_TILDE},
	{0xff,		0xff},		// no IJ digraph
	{0xff,		0xff},		// no ij digraph
	{'J',			F_CIRCUM},
	{F_DOTLESJ,	F_CIRCUM},
	{'K',			F_CEDILLA},
	{'k',			F_CEDILLA},
	{'L',			F_ACUTE},
	{'l',			F_ACUTE},
	{'L',			F_CARON},
	{'l',			F_CARON},
	{'L',			F_CEDILLA},
	{'l',			F_CEDILLA},
	{'L',			F_CENTERD},
	{'l',			F_CENTERD},
	{'L',			F_STROKE},
	{'l',			F_STROKE},
	{'N',			F_ACUTE},
	{'n',			F_ACUTE},
	{'N',			F_APOSBA},
	{'n',			F_APOSBA},
	{'N',			F_CARON},
	{'n',			F_CARON},
	{'N',			F_CEDILLA},
	{'n',			F_CEDILLA},
	{'O',			F_DACUTE},
	{'o',			F_DACUTE},
	{'O',			F_MACRON},
	{'o',			F_MACRON},
	{0xff,		0xff},		// OE digraph
	{0xff,		0xff},		// oe digraph
	{'R',			F_ACUTE},
	{'r',			F_ACUTE},
	{'R',			F_CARON},
	{'r',			F_CARON},
	{'R',			F_CEDILLA},
	{'r',			F_CEDILLA},
	{'S',			F_ACUTE},
	{'s',			F_ACUTE},
	{'S',			F_CARON},
	{'s',			F_CARON},
	{'S',			F_CEDILLA},
	{'s',			F_CEDILLA},
	{'S',			F_CIRCUM},
	{'s',			F_CIRCUM},
	{'T',			F_CARON},
	{'t',			F_CARON},
	{'T',			F_CEDILLA},
	{'t',			F_CEDILLA},
	{'T',			F_CROSSB},
	{'t',			F_CROSSB},
	{'U',			F_BREVE},
	{'u',			F_BREVE},
	{'U',			F_DACUTE},
	{'u',			F_DACUTE},
	{'U',			F_MACRON},
	{'u',			F_MACRON},
	{'U',			F_OGONEK},
	{'u',			F_OGONEK},
	{'U',			F_RING},
	{'u',			F_RING},
	{'U',			F_TILDE},
	{'u',			F_TILDE},
	{'W',			F_CIRCUM},
	{'w',			F_CIRCUM},
	{'Y',			F_CIRCUM},
	{'y',			F_CIRCUM},
	{'Z',			F_ACUTE},
	{'z',			F_ACUTE},
	{'Z',			F_CARON},
	{'z',			F_CARON},
	{'Z',			F_DOTA},
	{'z',			F_DOTA},
	{0xff,		0xff},		// no Eng
	{0xff,		0xff},		// no eng
	{'D',			F_MACRON},
	{'d',			F_MACRON},
	{'L',			F_MACRON},
	{'l',			F_MACRON},
	{'N',			F_MACRON},
	{'n',			F_MACRON},
	{'R',			F_GRAVE},
	{'r',			F_GRAVE},
	{'S',			F_MACRON},
	{'s',			F_MACRON},
	{'T',			F_MACRON},
	{'t',			F_MACRON},
	{'Y',			F_BREVE},
	{'y',			F_BREVE},
	{'Y',			F_GRAVE},
	{'y',			F_GRAVE},
	{'D',			F_APOSBES},
	{'d',			F_APOSBES},
	{'O',			F_APOSBES},
	{'o',			F_APOSBES},
	{'U',			F_APOSBES},
	{'u',			F_APOSBES},
	{'E',			F_BREVE},
	{'e',			F_BREVE},
	{'I',			F_BREVE},
	{F_DOTLESI,	F_BREVE},
	{0xff,		0xff},		// no dotless I
	{0xff,		0xff},		// no dotless i
	{'O',			F_BREVE},
	{'o',			F_BREVE}
};

/****************************************************************************
Desc:
****************************************************************************/
static BASE_DIACRIT fwp_ml1c =
{
	216,    	// # of characters in table
	26,      // start char
	fwp_ml1c_table,
};

/****************************************************************************
Desc:		Format of index:
				2 words before = count.
				word before = start character.
				db	code for base char.
				db	code for diacritic
Notes:	Diacritical char is always in same set as composed char
			base is in same set
****************************************************************************/
static BASE_DIACRIT_TABLE fwp_grk_c_table[] =
{
	{  0, 			F_GHPRIME },					// ALPHA High Prime
	{  1, 			F_GACUTE },						// alpha acute
	{ 10, 			F_GHPRIME },					// EPSILON High Prime
	{ 11, 			F_GACUTE },						// epsilon Acute
	{ 14, 			F_GHPRIME },					// ETA High Prime
	{ 15, 			F_GACUTE },						// eta Acute
	{ 18, 			F_GHPRIME },					// IOTA High Prime
	{ 19, 			F_GACUTE },						// iota Acute
	{ 0xFF, 			0xFF },							// IOTA Diaeresis
	{ 19, 			F_GDIA },						// iota Diaeresis
	{ 30, 			F_GHPRIME },					// OMICRON High Prime
	{ 31, 			F_GACUTE },						// omicron Acute
	{ 42, 			F_GHPRIME },					// UPSILON High Prime
	{ 43, 			F_GACUTE },						// upsilon Acute
	{ 0xFF, 			0xFF }, 							// UPSILON Diaeresis
	{ 43, 			F_GDIA }, 						// upsilon Diaeresis
	{ 50, 			F_GHPRIME }, 					// OMEGA High Prime
	{ 51, 			F_GACUTE }, 					// omega Acute
	{ 0xFF, 			0xFF },							// epsilon (Variant)
	{ 0xFF, 			0xFF },							// theta (Variant)
	{ 0xFF, 			0xFF },							// kappa (Variant)
	{ 0xFF, 			0xFF },							// pi (Variant)
	{ 0xFF, 			0xFF },							// rho (Variant)
	{ 0xFF, 			0xFF },							// sigma (Variant)
	{ 0xFF, 			0xFF },							// UPSILON (Variant)
	{ 0xFF, 			0xFF },							// phi (Variant)
	{ 0xFF, 			0xFF },							// omega (Variant)
	{ 0xFF, 			0xFF },							// Greek Question Mark
	{ 0xFF, 			0xFF },							// Greek Semicolon
	{ 0xFF, 			0xFF },							// High Prime
	{ 0xFF, 			0xFF },							// Low Prime
	{ 0xFF, 			0xFF },							// Acute (Greek)
	{ 0xFF, 			0xFF },							// Diaeresis (Greek)
	{ F_GACUTE, 	F_GDIA },						// Acute Diaeresis
	{ F_GGRAVE, 	F_GDIA },						// Grave Diaeresis
	{ 0xFF, 			0xFF },							// Grave (Greek)
	{ 0xFF, 			0xFF },							// Circumflex (Greek)
	{ 0xFF, 			0xFF },							// Smooth Breathing
	{ 0xFF, 			0xFF },							// Rough Breathing
	{ 0xFF, 			0xFF },							// Iota Subscript
	{ F_GSMOOTH,	F_GACUTE },						// Smooth Breathing Acute
	{ F_GROUGH, 	F_GACUTE },						// Rough Breathing Acute
	{ F_GSMOOTH, 	F_GGRAVE },						// Smooth Breathing Grave
	{ F_GROUGH, 	F_GGRAVE },						// Rough Breathing Grave
	{ F_GSMOOTH, 	F_GCIRCM },						// Smooth Breathing Circumflex
	{ F_GROUGH, 	F_GCIRCM },						// Rough Breathing Circumflex
	{ F_GACUTE, 	F_GIOTA },						// Acute w/Iota Subscript
	{ F_GGRAVE, 	F_GIOTA },						// Grave w/Iota Subscript
	{ F_GCIRCM, 	F_GIOTA },						// Circumflex w/Iota Subscript
	{ F_GSMOOTH, 	F_GIOTA },						// Smooth Breathing w/Iota Subscript
	{ F_GROUGH, 	F_GIOTA },						// Rough Breathing w/Iota Subscript
	{ F_GSMACT, 	F_GIOTA },						// Smooth Breathing Acute w/Iota Subscript
	{ F_GRGACT, 	F_GIOTA },						// Rough Breathing Acute w/Iota Subscript
	{ F_GSMGRV, 	F_GIOTA },						// Smooth Breathing Grave w/Iota Subscript
	{ F_GRGGRV, 	F_GIOTA },						// Rough Breathing Grave w/Iota Subscript
	{ F_GSMCIR, 	F_GIOTA },						// Smooth Breathing Circumflex w/Iota Sub
	{ F_GRGCIR, 	F_GIOTA },						// Rough Breathing Circumflex w/Iota Sub
	{ 1, 				F_GGRAVE },						// alpha Grave
	{ 1, 				F_GCIRCM },						// alpha Circumflex
	{ 1, 				F_GIOTA },						// alpha w/Iota
	{ 1, 				F_GACTIO },						// alpha Acute w/Iota
	{ 1, 				F_GGRVIO },						// alpha Grave w/Iota
	{ 1, 				F_GCIRIO },						// alpha Circumflex w/Iota
	{ 1, 				F_GSMOOTH },					// alpha Smooth
	{ 1, 				F_GSMACT },						// alpha Smooth Acute
	{ 1, 				F_GSMGRV },						// alpha Smooth Grave
	{ 1, 				F_GSMCIR },						// alpha Smooth Circumflex
	{ 1, 				F_GSMIO },						// alpha Smooth w/Iota
	{ 1, 				F_GSMAIO },						// alpha Smooth Acute w/Iota
	{ 1, 				F_GSMGVIO },					// alpha Smooth Grave w/Iota
	{ 1, 				F_GSMCIO },						// alpha Smooth Circumflex w/Iota
	{ 1, 				F_GROUGH },						// alpha Rough
	{ 1, 				F_GRGACT },						// alpha Rough Acute
	{ 1, 				F_GRGGRV },						// alpha Rough Grave
	{ 1, 				F_GRGCIR },						// alpha Rough Circumflex
	{ 1, 				F_GRGIO	 },					// alpha Rough w/Iota
	{ 1, 				F_GRGAIO },						// alpha Rough Acute w/Iota
	{ 1, 				F_GRGGVIO },					// alpha Rough Grave w/Iota
	{ 1, 				F_GRGCIO },						// alpha Rough Circumflex w/Iota
	{ 11, 			F_GGRAVE },						// epsilon Grave
	{ 11, 			F_GSMOOTH },					// epsilon Smooth
	{ 11, 			F_GSMACT },						// epsilon Smooth Acute
	{ 11, 			F_GSMGRV },						// epsilon Smooth Grave
	{ 11, 			F_GROUGH },						// epsilon Rough
	{ 11, 			F_GRGACT },						// epsilon Rough Acute
	{ 11, 			F_GRGGRV },						// epsilon Rough Grave
	{ 15, 			F_GGRAVE },						// eta Grave
	{ 15, 			F_GCIRCM },						// eta Circumflex
	{ 15, 			F_GIOTA },						// eta w/Iota
	{ 15, 			F_GACTIO },						// eta Acute w/Iota
	{ 15, 			F_GGRVIO },						// eta Grave w/Iota
	{ 15, 			F_GCIRIO },						// eta Circumflex w/Iota
	{ 15,				F_GSMOOTH },					// eta Smooth
	{ 15, 			F_GSMACT },						// eta Smooth Acute
	{ 15, 			F_GSMGRV },						// eta Smooth Grave
	{ 15, 			F_GSMCIR },						// eta Smooth Circumflex
	{ 15, 			F_GSMIO },						// eta Smooth w/Iota
	{ 15, 			F_GSMAIO },						// eta Smooth Acute w/Iota
	{ 15, 			F_GSMGVIO },					// eta Smooth Grave w/Iota
	{ 15, 			F_GSMCIO },						// eta Smooth Circumflex w/Iota
	{ 15, 			F_GROUGH },						// eta Rough
	{ 15, 			F_GRGACT },						// eta Rough Acute
	{ 15, 			F_GRGGRV },						// eta Rough Grave
	{ 15, 			F_GRGCIR },						// eta Rough Circumflex
	{ 15, 			F_GRGIO },						// eta Rough w/Iota
	{ 15, 			F_GRGAIO },						// eta Rough Acute w/Iota
	{ 15, 			F_GRGGVIO },					// eta Rough Grave w/Iota
	{ 15, 			F_GRGCIO },						// eta Rough Circumflex w/Iota
	{ 19, 			F_GGRAVE },						// iota Grave
	{ 19, 			F_GCIRCM },						// iota Circumflex
	{ 19, 			F_GACTDIA },					// iota Acute Diaeresis
	{ 19, 			F_GGRVDIA },					// iota Grave Diaeresis
	{ 19, 			F_GSMOOTH },					// iota Smooth
	{ 19, 			F_GSMACT },						// iota Smooth Acute
	{ 19, 			F_GSMGRV },						// iota Smooth Grave
	{ 19, 			F_GSMCIR },						// iota Smooth Circumflex
	{ 19, 			F_GROUGH },						// iota Rough
	{ 19, 			F_GRGACT },						// iota Rough Acute
	{ 19, 			F_GRGGRV },						// iota Rough Grave
	{ 19, 			F_GRGCIR },						// iota Rough Circumflex
	{ 31, 			F_GGRAVE },						// omicron Grave
	{ 31, 			F_GSMOOTH },					// omicron Smooth
	{ 31, 			F_GSMACT },						// omicron Smooth Acute
	{ 31, 			F_GSMGRV },						// omicron Smooth Grave
	{ 31, 			F_GROUGH },						// omicron Rough
	{ 31, 			F_GRGACT },						// omicron Rough Acute
	{ 31, 			F_GRGGRV },						// omicron Rough Grave
	{ 0xFF, 			0xFF },							// rho rough
	{ 0xFF, 			0xFF },							// rho smooth
	{ 43, 			F_GGRAVE },						// upsilon Grave
	{ 43, 			F_GCIRCM },						// upsilon Circumflex
	{ 43, 			F_GACTDIA },					// upsilon Acute Diaeresis
	{ 43, 			F_GGRVDIA },					// upsilon Grave Diaeresis
	{ 43, 			F_GSMOOTH },					// upsilon Smooth
	{ 43, 			F_GSMACT },						// upsilon Smooth Acute
	{ 43, 			F_GSMGRV },						// upsilon Smooth Grave
	{ 43, 			F_GSMCIR },						// upsilon Smooth Circumflex
	{ 43, 			F_GROUGH },						// upsilon Rough
	{ 43, 			F_GRGACT },						// upsilon Rough Acute
	{ 43, 			F_GRGGRV },						// upsilon Rough Grave
	{ 43, 			F_GRGCIR },						// upsilon Rough Circumflex
	{ 51, 			F_GGRAVE },						// omega Grave
	{ 51, 			F_GCIRCM },						// omega Circumflex
	{ 51, 			F_GIOTA },						// omega w/Iota
	{ 51, 			F_GACTIO },						// omega Acute w/Iota
	{ 51, 			F_GGRVIO },						// omega Grave w/Iota
	{ 51, 			F_GCIRIO },						// omega Circumflex w/Iota
	{ 51, 			F_GSMOOTH },					// omega Smooth
	{ 51, 			F_GSMACT },						// omega Smooth Acute
	{ 51, 			F_GSMGRV },						// omega Smooth Grave
	{ 51, 			F_GSMCIR },						// omega Smooth Circumflex
	{ 51, 			F_GSMIO },						// omega Smooth w/Iota
	{ 51, 			F_GSMAIO },						// omega Smooth Acute w/Iota
	{ 51, 			F_GSMGVIO },					// omega Smooth Grave w/Iota
	{ 51, 			F_GSMCIO },						// omega Smooth Circumflex w/Iota
	{ 51, 			F_GROUGH },						// omega Rough
	{ 51, 			F_GRGACT },						// omega Rough Acute
	{ 51, 			F_GRGGRV },						// omega Rough Grave
	{ 51, 			F_GRGCIR },						// omega Rough Circumflex
	{ 51, 			F_GRGIO },						// omega Rough w/Iota
	{ 51, 			F_GRGAIO },						// omega Rough Acute w/Iota
	{ 51, 			F_GRGGVIO },					// omega Rough Grave w/Iota
	{ 51, 			F_GRGCIO}						// omega Rough Circumflex w/Iota
};

/****************************************************************************
Desc:
****************************************************************************/
static BASE_DIACRIT fwp_grk_c =
{
	163,	// # of characters in table.
	52,	// start char.
	fwp_grk_c_table
};

/****************************************************************************
Desc:		Format of index:
				2 words before = count.
				word before = start character.
				db	code for base char.
				db code for diacritic
Notes:	Diacritical char is always in same set as composed char
			base is in same set
****************************************************************************/
static BASE_DIACRIT_TABLE fwp_rus_c_table[] =
{
	{ 14, 			204 },					// ZHE with right descender
	{ 15, 			204 },					// zhe with right descender
	{ 0xFF, 			0xFF },					// DZE
	{ 0xFF, 			0xFF },					// dze
	{ 0xFF, 			0xFF },					// Z
	{ 0xFF, 			0xFF },					// z
	{ 18, 			206 },					// II with macron
	{ 19, 			206 },					// ii with macron
	{ 0xFF, 			0xFF },					// I
	{ 0xFF, 			0xFF },					// i
	{ 0xFF, 			0xFF },					// YI
	{ 0xFF, 			0xFF },					// yi
	{ 0xFF, 			0xFF },					// I ligature
	{ 0xFF, 			0xFF },					// i ligature
	{ 0xFF, 			0xFF },					// JE
	{ 0xFF, 			0xFF },					// je
	{ 0xFF, 			0xFF },					// KJE
	{ 0xFF, 			0xFF },					// kje
	{ 22, 			204 },					// KA with right descender
	{ 23, 			204 },					// ka with right descender
	{ 22, 			205 },					// KA ogonek
	{ 23, 			205 },					// ka ogonek
	{ 0xFF, 			0xFF },					// KA vertical bar
	{ 0xFF, 			0xFF },					// ka vertical bar
	{ 0xFF, 			0xFF },					// LJE
	{ 0xFF, 			0xFF },					// lje
	{ 28, 			204 },					// EN with right descender
	{ 29, 			204 },					// en with right descender
	{ 0xFF, 			0xFF },					// NJE
	{ 0xFF, 			0xFF },					// nje
	{ 0xFF, 			0xFF },					// ROUND OMEGA
	{ 0xFF, 			0xFF },					// round omega
	{ 0xFF, 			0xFF },					// OMEGA
	{ 0xFF, 			0xFF },					// omega
	{ 0xFF, 			0xFF },					// TSHE
	{ 0xFF, 			0xFF },					// tshe
	{ 0xFF, 			0xFF },					// SHORT U
	{ 0xFF, 			0xFF },					// short u
	{ 40, 			206 },					// U with macron
	{ 41, 			206 },					// u with macron
	{ 0xFF, 			0xFF },					// STRAIGHT U
	{ 0xFF, 			0xFF },					// straight u
	{ 0xFF, 			0xFF },					// STRAIGHT U BAR
	{ 0xFF, 			0xFF },					// straight u bar
	{ 0xFF, 			0xFF },					// OU ligature
	{ 0xFF, 			0xFF },					// ou ligature
	{ 44, 			204 },					// KHA with right descender
	{ 45, 			204 },					// kha with right descender
	{ 44, 			205 },					// KHA ogonek
	{ 45, 			205 },					// kha ogonek
	{ 0xFF, 			0xFF },					// H
	{ 0xFF, 			0xFF },					// h
	{ 0xFF, 			0xFF },					// OMEGA titlo
	{ 0xFF, 			0xFF },					// omega titlo
	{ 0xFF, 			0xFF },					// DZHE
	{ 0xFF, 			0xFF },					// dzhe
	{ 48, 			204 },					// CHE with right descender
	{ 49, 			204 },					// che with right descender
	{ 0xFF, 			0xFF },					// CHE vertical bar
	{ 0xFF, 			0xFF },					// che vertical bar
	{ 0xFF, 			0xFF },					// SHCHA (variant)
	{ 0xFF, 			0xFF },					// shcha (variant)
	{ 0xFF, 			0xFF },					// YAT
	{ 0xFF, 			0xFF },					// yat
	{ 0xFF, 			0xFF },					// YUS BOLSHOI
	{ 0xFF, 			0xFF },					// yus bolshoi
	{ 0xFF, 			0xFF },					// BIG MALYI
	{ 0xFF, 			0xFF },					// big malyi
	{ 0xFF, 			0xFF },					// KSI
	{ 0xFF, 			0xFF },					// ksi
	{ 0xFF, 			0xFF },					// PSI
	{ 0xFF, 			0xFF },					// psi
	{ 0xFF, 			0xFF },					// FITA
	{ 0xFF, 			0xFF },					// fita
	{ 0xFF, 			0xFF },					// IZHITSA
	{ 0xFF, 			0xFF },					// izhitsa
	{ 00, 			F_RACUTE },				// Russian A acute
	{ 01, 			F_RACUTE },				// Russian a acute
	{ 10, 			F_RACUTE },				// Russian IE acute
	{ 11, 			F_RACUTE },				// Russian ie acute
	{ 78, 			F_RACUTE },				// Russian E acute
	{ 79, 			F_RACUTE },				// Russian e acute
	{ 18, 			F_RACUTE },				// Russian II acute
	{ 19, 			F_RACUTE },				// Russian ii acute
	{ 88, 			F_RACUTE },				// Russian I acute
	{ 89, 			F_RACUTE },				// Russian i acute
	{ 90, 			F_RACUTE },				// Russian YI acute
	{ 91, 			F_RACUTE },				// Russian yi acute
	{ 30, 			F_RACUTE },				// Russian O acute
	{ 31, 			F_RACUTE },				// Russian o acute
	{ 40, 			F_RACUTE },				// Russian U acute
	{ 41, 			F_RACUTE },				// Russian u acute
	{ 56, 			F_RACUTE },				// Russian YERI acute
	{ 57, 			F_RACUTE },				// Russian yeri acute
	{ 60, 			F_RACUTE },				// Russian REVERSED E acute
	{ 61, 			F_RACUTE },				// Russian reversed e acute
	{ 62, 			F_RACUTE },				// Russian IU acute
	{ 63, 			F_RACUTE },				// Russian iu acute
	{ 64, 			F_RACUTE },				// Russian IA acute
	{ 65, 			F_RACUTE },				// Russian ia acute
	{ 00, 			F_RGRAVE },				// Russian A grave
	{ 01, 			F_RGRAVE },				// Russian a grave
	{ 10, 			F_RGRAVE },				// Russian IE grave
	{ 11, 			F_RGRAVE },				// Russian ie grave
	{ 12, 			F_RGRAVE },				// Russian YO grave
	{ 13, 			F_RGRAVE },				// Russian yo grave
	{ 18, 			F_RGRAVE },				// Russian I grave
	{ 19, 			F_RGRAVE },				// Russian i grave
	{ 30, 			F_RGRAVE },				// Russian O grave
	{ 31, 			F_RGRAVE },				// Russian o grave
	{ 40, 			F_RGRAVE },				// Russian U grave
	{ 41, 			F_RGRAVE },				// Russian u grave
	{ 56, 			F_RGRAVE },				// Russian YERI grave
	{ 57, 			F_RGRAVE },				// Russian yeri grave
	{ 60, 			F_RGRAVE },				// Russian REVERSED E grave
	{ 61, 			F_RGRAVE },				// Russian reversed e grave
	{ 62, 			F_RGRAVE },				// Russian IU grave
	{ 63, 			F_RGRAVE },				// Russian iu grave
	{ 64, 			F_RGRAVE },				// Russian IA grave
	{ 65, 			F_RGRAVE }					// Russian ia grave
};

/****************************************************************************
Desc:
****************************************************************************/
static BASE_DIACRIT fwp_rus_c =
{
	120,				// # of characters in table.
	156,				// start char.
	fwp_rus_c_table,
};

/****************************************************************************
Desc:		Table of pointers to character component tables.
****************************************************************************/
static BASE_DIACRIT * fwp_car60_c[ F_NCHSETS] =
{
	(BASE_DIACRIT*)0,    // no composed characters for ascii.
	&fwp_ml1c,
	(BASE_DIACRIT*)0,    // no composed characters for multinational 2
	(BASE_DIACRIT*)0,    // no composed characters for line draw.
	(BASE_DIACRIT*)0,    // no composed characters for typographic.
	(BASE_DIACRIT*)0,    // no composed characters for icons.
	(BASE_DIACRIT*)0,    // no composed characters for math.
	(BASE_DIACRIT*)0,    // no composed characters for math extension.
	&fwp_grk_c,				// Greek
	(BASE_DIACRIT*)0,		// Hebrew
	&fwp_rus_c,				// Cyrillic - Russian
	(BASE_DIACRIT*)0,		// Hiragana or Katakana (Japanese)
	(BASE_DIACRIT*)0,		// no composed characters for user.
	(BASE_DIACRIT*)0,		// no composed characters for Arabic.
	(BASE_DIACRIT*)0,		// no composed characters for Arabic Script .
};

/****************************************************************************
Desc:		Map special chars in CharSet (x24) to collation values
****************************************************************************/
static BYTE_WORD_TBL fwp_Ch24ColTbl[] =
{
	{1,	COLLS+2},					// comma
	{2,	COLLS+1},					// maru
	{5,	COLS_ASIAN_MARKS+2},		// chuuten
	{10,	COLS_ASIAN_MARKS},		// dakuten
	{11,	COLS_ASIAN_MARKS+1},		// handakuten
	{43,	COLS2+2},					// angled brackets
	{44,	COLS2+3},					//
	{49,	COLS2+2},					// pointy brackets
	{50,	COLS2+3},
	{51,	COLS2+2},					// double pointy brackets
	{52,	COLS2+3},
	{53,	COLS1},						// Japanese quotes
	{54,	COLS1},
	{55,	COLS1},						// hollow Japanese quotes
	{56,	COLS1},
	{57,	COLS2+2},					// filled rounded brackets
	{58,	COLS2+3}
};

/****************************************************************************
Desc:		Kana subcollation values
		 		BIT 0: set if large char
				 BIT 1: set if voiced
				 BIT 2: set if half voiced
Notes:
			To save space should be nibbles
			IMPORTANT:
				The '1' entries that do not have
				a matching '0' entry have been
				changed to zero to save space in
				the subcollation area.
				The original table is listed below.
****************************************************************************/
static FLMBYTE KanaSubColTbl[] =
{
	0,1,0,1,0,1,0,1,0,1,				// a    A   i   I   u   U   e   E   o   O
	1,3,0,3,0,3,1,3,0,3,				// KA  GA  KI  GI  KU  GU  KE  GE  KO  GO
	0,3,0,3,0,3,0,3,0,3,				// SA  ZA SHI  JI  SU  ZU  SE  ZE  SO  ZO
	0,3,0,3,0,1,3,0,3,0,3,			// TA  DA CHI  JI tsu TSU  ZU  TE DE TO DO
	0,0,0,0,0,							// NA NI NU NE NO
	0,3,5,0,3,5,0,3,5,				// HA BA PA HI BI PI FU BU PU
	0,3,5,0,3,5,						// HE BE PE HO BO PO
	0,0,0,0,0,							// MA MI MU ME MO
	0,1,0,1,0,1,						// ya YA yu YU yo YO
	0,0,0,0,0,							// RA RI RU RE RO
	0,1,0,0,0,							// wa WA WI WE WO
	0,3,0,0								//  N VU ka ke
};

/****************************************************************************
Desc:		Map katakana (CharSet x26) to collation values
			kana collating values are two byte values
			where the high byte is 0x01.
****************************************************************************/
static FLMBYTE KanaColTbl[] =
{
	 0, 0, 1, 1, 2, 2, 3, 3, 4, 4,		// a    A   i   I   u   U   e   E   o   O
 	 5, 5, 6, 6, 7, 7, 8, 8, 9, 9,		// KA  GA  KI  GI  KU  GU  KE  GE  KO  GO
	10,10,11,11,12,12,13,13,14,14,		// SA  ZA SHI  JI  SU  ZU  SE  ZE  SO  ZO
	15,15,16,16,17,17,17,18,18,19,19,	// TA DA CHI JI tsu TSU  ZU  TE DE TO DO
	20,21,22,23,24,							// NA NI NU NE NO
	25,25,25,26,26,26,27,27,27,			// HA BA PA HI BI PI FU BU PU
	28,28,28,29,29,29,						// HE BE PE HO BO PO
	30,31,32,33,34,							// MA MI MU ME MO
	35,35,36,36,37,37,						// ya YA yu YU yo YO
	38,39,40,41,42,							// RA RI RU RE RO
	43,43,44,45,46,							// wa WA WI WE WO
	47, 2, 5, 8									//  N VU ka ke
};

/****************************************************************************
Desc:		Map KataKana collated value to vowel value for
			use for the previous char.
****************************************************************************/
static FLMBYTE KanaColToVowel[] =
{
	0,1,2,3,4,		//  a   i   u  e  o
	0,1,2,3,4,		// ka  ki  ku ke ko
	0,1,2,3,4,		// sa shi  su se so
	0,1,2,3,4,		// ta chi tsu te to
	0,1,2,3,4,		// na  ni  nu ne no
	0,1,2,3,4,		// ha  hi  hu he ho
	0,1,2,3,4,		// ma  mi  mu me mo
	0,2,4,			// ya  yu  yo
	0,1,2,3,4,		// ra  ri  ru re ro
	0,1,3,4,			// wa  wi  we wo
};

/****************************************************************************
Desc:		Convert Zenkaku (double wide) to Hankaku (single wide)
			Character set 0x24 maps to single wide chars in other char sets.
			This enables collation values to be found on some symbols.
			This is also used to convert symbols from hankaku to Zen24.
****************************************************************************/
static BYTE_WORD_TBL Zen24ToHankaku[] =
{
	{	0  ,0x0020 },		// space
	{	1  ,0x0b03 },		// japanese comma
	{	2  ,0x0b00 },		// circle period
	{	3  ,  44	 },		// comma
	{	4  ,  46	 },		// period
	{	5  ,0x0b04 },		// center dot
	{	6  ,  58	 },		// colon
	{	7  ,  59	 },		// semicolon
	{	8  ,  63	 },		// question mark
	{	9  ,  33	 },		// exclamation mark
	{	10 ,0x0b3d },		// dakuten
	{	11 ,0x0b3e },		// handakuten
	{	12 ,0x0106 },		// accent mark
	{	13 ,  96	 },		// accent mark
	{	14 ,0x0107 },		// umlat
	{	15 ,  94	 },		// caret
	{	16 ,0x0108 },		// macron
	{	17 ,  95	 },		// underscore
	{	27 ,0x0b0f },		// extend vowel
	{	28 ,0x0422 },		// mdash
	{	29 ,  45	 },		// hyphen
	{	30 ,  47  },     	// slash
	{	31 ,0x0607 },		// backslash
	{	32 , 126	 },		// tilde
	{	33 ,0x0611 },		// doubleline
	{	34 ,0x0609 },		// line
	{	37 ,0x041d },		// left apostrophe
	{	38 ,0x041c },		// right apostrophe
	{	39 ,0x0420 },		// left quote
	{	40 ,0x041f },		// right quote
	{	41 ,  40	 },		// left paren
	{	42 ,  41	 },		// right paren
	{	45 ,  91	 },		// left bracket
	{	46 ,  93	 },		// right bracket
	{	47 , 123	 },		// left curly bracket
	{	48 , 125	 },		// right curly bracket
	{	53 ,0x0b01 },		// left j quote
	{	54 ,0x0b02 },		// right j quote
	{	59 ,  43	 },		// plus
	{	60 ,0x0600 },		// minus
	{	61 ,0x0601 },		// plus/minus
	{	62 ,0x0627 },		// times
	{	63 ,0x0608 },		// divide
	{	64 ,  61	 },		// equal
	{	65 ,0x0663 },		// unequal
	{	66 ,  60	 },		// less
	{	67 ,  62	 },		// greater
	{	68 ,0x0602 },		// less/equal
	{	69 ,0x0603 },		// greater/equal
	{	70 ,0x0613 },		// infinity
	{	71 ,0x0666 },		// traingle dots
	{	72 ,0x0504 },		// man
	{	73 ,0x0505 },		// woman
	{	75 ,0x062d },		// prime
	{	76 ,0x062e },		// double prime
	{	78 ,0x040c },		// yen
	{	79 ,  36	 },		// $
	{	80 ,0x0413 },		// cent
	{	81 ,0x040b },		// pound
	{	82 ,  37	 },		// %
	{	83 ,  35	 },		// #
	{	84 ,  38	 },		// &
	{	85 ,  42	 },		// *
	{	86 ,  64	 },		// @
	{	87 ,0x0406 },		// squiggle
	{	89 ,0x06b8 },		// filled star
	{	90 ,0x0425 },		// hollow circle
	{	91 ,0x042c },		// filled circle
	{	93 ,0x065f },		// hollow diamond
	{	94 ,0x0660 },		// filled diamond
	{	95 ,0x0426 },		// hollow box
	{	96 ,0x042e },		// filled box
	{	97 ,0x0688 },		// hollow triangle
	{	99 ,0x0689 },		// hollow upside down triangle
	{	103,0x0615 },		// right arrow
	{	104,0x0616 },		// left arrow
	{	105,0x0617 },		// up arrow
	{	106,0x0622 },		// down arrow
	{	119,0x060f },
	{	121,0x0645 },
	{	122,0x0646 },
	{	123,0x0643 },
	{	124,0x0644 },
	{	125,0x0642 },		// union
	{	126,0x0610 },		// intersection
	{	135,0x0655 },
	{	136,0x0656 },
	{	138,0x0638 },		// right arrow
	{	139,0x063c },		// left/right arrow
	{	140,0x067a },
	{	141,0x0679 },
	{	153,0x064f },		// angle
	{	154,0x0659 },
	{	155,0x065a },
	{	156,0x062c },
	{	157,0x062b },
	{	158,0x060e },
	{	159,0x06b0 },
	{	160,0x064d },
	{	161,0x064e },
	{	162,0x050e },		// square root
	{	164,0x0604 },
	{	175,0x0623 },		// angstrom
	{	176,0x044b },		// percent
	{	177,0x051b },		// sharp
	{	178,0x051c },		// flat
	{	179,0x0509 },		// musical note
	{	180,0x0427 },		// dagger
	{	181,0x0428 },		// double dagger
	{	182,0x0405 },		// paragraph
	{	187,0x068f }		// big hollow circle
};

/****************************************************************************
Desc:		Maps CS26 to CharSet 11
			Used to uncollate characters for FLAIM - placed here for consistency
				0x80 - add dakuten
				0xC0 - add handakuten
				0xFF - no mapping exists
****************************************************************************/
static FLMBYTE MapCS26ToCharSet11[ 86] =
{
	0x06,	// 0     a
	0x10,	// 1     A
	0x07,	// 2     i
	0x11,	//	3     I
	0x08,	//	4     u
	0x12,	//	5     U
	0x09,	//	6     e
	0x13,	//	7     E
	0x0a,	//	8     o
	0x14,	//	9     O

	0x15,	//	0x0a  KA
	0x95,	//       GA - 21 followed by 0x3D dakuten

	0x16,	// 0x0c  KI
	0x96,	//       GI
	0x17,	//	0x0e  KU
	0x97,	//       GU
	0x18,	// 0x10  KE
	0x98,	//       GE
	0x19,	// 0x12  KO
	0x99,	//       GO

	0x1a,	//	0x14  SA
	0x9a,	//       ZA
	0x1b,	//	0x16  SHI
	0x9b,	//       JI
	0x1c,	//	0x18  SU
	0x9c,	//       ZU
	0x1d,	//	0x1a  SE
	0x9d,	//       ZE
	0x1e,	//	0x1c  SO
	0x9e,	//       ZO

	0x1f,	//	0x1e  TA
	0x9f,	//       DA
	0x20,	//	0x20  CHI
	0xa0,	//       JI
	0x0e,	//	0x22  small tsu
	0x21,	//	0x23  TSU
	0xa1,	//       ZU
	0x22,	//	0x25  TE
	0xa2,	//       DE
	0x23,	//	0x27  TO
	0xa3,	//       DO

	0x24,	//	0x29  NA
	0x25,	//	0x2a  NI
	0x26,	// 0x2b  NU
	0x27,	//	0x2c  NE
	0x28,	//	0x2d  NO

	0x29,	//	0x2e  HA
	0xa9,	// 0x2f  BA
	0xe9,	// 0x30  PA
	0x2a,	//	0x31  HI
	0xaa,	// 0x32  BI
	0xea,	// 0x33  PI
	0x2b,	//	0x34  FU
	0xab,	// 0x35  BU
	0xeb,	// 0x36  PU
	0x2c,	//	0x37  HE
	0xac,	// 0x38  BE
	0xec,	// 0x39  PE
	0x2d,	//	0x3a  HO
	0xad,	// 0x3b  BO
	0xed,	// 0x3c  PO

	0x2e,	//	0x3d  MA
	0x2f,	//	0x3e  MI
	0x30,	//	0x3f  MU
	0x31,	//	0x40  ME
	0x32,	//	0x41  MO

	0x0b,	//	0x42  small ya
	0x33,	//	0x43  YA
	0x0c,	//	0x44  small yu
	0x34,	//	0x45  YU
	0x0d,	// 0x46  small yo
	0x35,	//	0x47  YO

	0x36,	//	0x48  RA
	0x37,	//	0x49  RI
	0x38,	//	0x4a  RU
	0x39,	//	0x4b  RE
	0x3a,	//	0x4c  RO

	0xff,	// 0x4d  small wa
	0x3b,	//	0x4e  WA
	0xff,	// 0x4f  WI
	0xff,	// 0x50  WE
	0x05,	//	0x51	WO

	0x3c,	//	0x52	N
	0xff,	// 0x53  VU
	0xff, // 0x54  ka
	0xff 	// 0x55  ke
};

/****************************************************************************
Desc:		Conversion from single (Hankaku) to double (Zenkaku) wide characters
			Used in f_wpHanToZenkaku()
			Maps from charset 11 to CS24 (punctuation) (starting from 11,0)
****************************************************************************/
static FLMBYTE From0AToZen[] =
{
 	0, 	9,		40,	0x53, 		// sp ! " #
 	0x4f, 0x52, 0x54,	38, 			// $ % & '
 											// Was 187 for ! and 186 for '
	0x29,	0x2a,	0x55,	0x3b, 		// ( ) * +
	3,		0x1d,	4,		0x1e	 		// , - . /
};

/****************************************************************************
Desc:
****************************************************************************/
static FLMBYTE From0BToZen[] =
{
	6,		7,		0x42,	0x40,			// : ; < =
	0x43,	8,		0x56					// > ? @
};

/****************************************************************************
Desc:
****************************************************************************/
static FLMBYTE From0CToZen[] =
{
	0x2d,	0x1f,	0x2e,	0x0f,	0x11,	0x0d	// [ BACKSLASH ] ^ _ `
};

/****************************************************************************
Desc:
****************************************************************************/
static FLMBYTE From0DToZen[] =
{
	0x2f,	0x22,	0x30,	0x20 			// { | } ~
};

/****************************************************************************
Desc:
****************************************************************************/
static FLMBYTE  From8ToZen[] =
{
	0x5e, 0x7e, 0x5f, 0x7f, 0x5f, 0xFF, 0x60, 0x80,
	0x61, 0x81, 0x62, 0x82, 0x63, 0x83, 0x64, 0x84,
	0x65, 0x85, 0x66, 0x86, 0x67, 0x87, 0x68, 0x88,
	0x69, 0x89, 0x6a, 0x8a, 0x6b, 0x8b, 0x6c, 0x8c,
	0x6d, 0x8d, 0x6e, 0x8e, 0x6f, 0x8f, 0x6f, 0xFF,
	0x70, 0x90, 0x71, 0x91, 0x72, 0x92, 0x73, 0x93,
	0x74, 0x94, 0x75, 0x95
};

/****************************************************************************
Desc:
****************************************************************************/
static FLMBYTE From11AToZen[] =
{
	2,								// japanese period
	0x35,							// left bracket
	0x36,							// right bracket
	0x01,							// comma
	0x05							// chuuten
};

/****************************************************************************
Desc:
****************************************************************************/
static FLMBYTE From11BToZen[] =
{
	0x51,										// wo
	0,2,4,6,8,0x42,0x44,0x46,0x22,	// small a i u e o ya yu yo tsu
	0xFF, 1, 3, 5, 7, 9,					// dash (x241b) a i u e o
	0x0a, 0x0c, 0x0e, 0x10, 0x12,		// ka ki ku ke ko
	0x14, 0x16, 0x18, 0x1a, 0x1c,		// sa shi su se so
	0x1e, 0x20, 0x23, 0x25, 0x27,		// ta chi tsu te to
	0x29, 0x2a, 0x2b, 0x2c, 0x2d,		// na ni nu ne no
	0x2e, 0x31, 0x34, 0x37, 0x3a,		// ha hi fu he ho
	0x3d, 0x3e, 0x3f, 0x40, 0x41,		// ma mi mu me mo
	0x43, 0x45, 0x47,						// ya yu yo
	0x48, 0x49, 0x4a, 0x4b, 0x4c,		// ra ri ru re ro
	0x4e, 0x52								// WA N
};												// does not have wa WI WE VU ka ke

/****************************************************************************
Desc:
****************************************************************************/
static FLMUINT16 fwp_indexi[] =
{
	0,11,14,15,17,18,19,21,22,23,24,25,26,35,59
};

/****************************************************************************
Desc:
****************************************************************************/
static FLMUINT16 fwp_indexj[] =
{
	FLM_CA_LANG,	// Catalan (0)
	FLM_CF_LANG,	// Canadian French
	FLM_CZ_LANG,	// Czech
	FLM_SL_LANG,	// Slovak
	FLM_DE_LANG,	// German
	FLM_SD_LANG,	// Swiss German
	FLM_ES_LANG,	// Spanish (Spain)
	FLM_FR_LANG,	// French
	FLM_NL_LANG,	// Netherlands
	0xFFFF,			// DK_LANG,	Danish    - support for 'aa' -> a-ring out
	0xFFFF,			// NO_LANG,	Norwegian - support for 'aa' -> a-ring out
	0x0063,			// c						 - DOUBLE CHARACTERS - STATE ENTRIES
	0x006c,			// l
	0x0197,			// l with center dot
	0x0063,			// c
	0x0125,			// ae digraph
	0x01a7,			// oe digraph
	0x0068,			// h
	0x0068,			// h
	0x006c,			// l
	0x0101,			// center dot alone
	0x006c,			// l
	0x0117,			// ?	(for German)
	0x018b,			// ij digraph
	0x0000,			// was 'a' - will no longer map 'aa' to a-ring
	0x0000,			// was 'a'

	FLM_CZ_LANG,	// SINGLE CHARS - LANGUAGES
	FLM_DK_LANG,
	FLM_NO_LANG,
	FLM_SL_LANG,
	FLM_TK_LANG,
	FLM_SU_LANG,
	FLM_IS_LANG,
	FLM_SV_LANG,
	FLM_YK_LANG,
						// SINGLE CHARS
	0x011e,			// A Diaeresis					- alternate collating sequences
	0x011f,			// a Diaeresis
	0x0122,			// A Ring						- 2
	0x0123,			// a Ring
	0x0124,			// AE Diagraph					- 4
	0x0125,			// ae diagraph
	0x013e,			// O Diaeresis					- 6
	0x013f,			// o Diaeresis
	0x0146,			// U Diaeresis					- 8
	0x0147,			// u Diaeresis
	0x0150,			// O Slash						- 10
	0x0151,			// o Slash

	0x0A3a,			// CYRILLIC SOFT SIGN		- 12
	0x0A3b,			// CYRILLIC soft sign
	0x01ee,			// dotless i - turkish		- 14
	0x01ef,			// dotless I - turkish
	0x0162,			// C Hacek/caron - 1,98		- 16
	0x0163,			// c Hacek/caron - 1,99
	0x01aa,			// R Hacek/caron - 1,170	- 18
	0x01ab,			// r Hacek/caron - 1,171
	0x01b0,			// S Hacek/caron - 1,176	- 20
	0x01b1,			// s Hacek/caron - 1,177
	0x01ce,			// Z Hacek/caron - 1,206	- 22
	0x01cf,			// z Hacek/caron - 1,207
};

/****************************************************************************
Desc:
****************************************************************************/
static FLMUINT16 fwp_valuea[] =
{
//	DOUBLE CHAR STATE VALUES
	STATE1,		// 00
	STATE3,
	STATE2,
	STATE2,
	STATE8,
	STATE8,
	STATE1,
	STATE3,
	STATE9,
	STATE10,		// No longer in use
	STATE10,		// No longer in use
	STATE4,
	STATE6,
	STATE6,
	STATE5,
	INSTAE,
	INSTOE,
	AFTERC,
	AFTERH,
	AFTERL,
	STATE7,
	STATE6,
	INSTSG,		// ss for German
	INSTIJ,
	STATE11,		// aa - no longer in use
	WITHAA,		// aa - no longer in use

// SINGLE CHARS - LANGUAGES
	START_CZ,	// Czech
	START_DK,	// Danish
	START_NO,	// Norwegian
	START_SL,	// Slovak
	START_TK,	// Turkish
	START_SU,	// Finnish
	START_IS,	// Icelandic
	START_SV,	// Swedish
	START_YK,	// Ukrainian

// SINGLE CHARS FIXUP AREAS
	COLS9,		COLS9,		COLS9,		COLS9,		// US & OTHERS
	COLS9+1,		COLS9+1,		COLS9+21,	COLS9+21,
	COLS9+30,	COLS9+30,	COLS9+21,	COLS9+21,
	COLS10+43,	COLS10+43,	COLS9+12,	COLS9+12,
	COLS9+3,		COLS9+3,		COLS9+25,	COLS9+25,
	COLS9+27,	COLS9+27,	COLS9+35,	COLS9+35,

	COLS9+45,	COLS9+45,	COLS9+55,	COLS9+55,	// DANISH
	COLS9+42,	COLS9+42,	COLS9+53,	COLS9+53,
	COLS9+30,	COLS9+30,	COLS9+49,	COLS9+49,	// Oct98 U Diaer no longer to y Diaer
	COLS10+43,	COLS10+43,	COLS9+12,	COLS9+12,
	COLS9+3,		COLS9+3,		COLS9+25,	COLS9+25,
	COLS9+27,	COLS9+27,	COLS9+35,	COLS9+35,

	COLS9,		COLS9,		COLS9,		COLS9,		// Icelandic
	COLS9+46,	COLS9+46,	COLS9+50,	COLS9+50,
	COLS9+30,	COLS9+30,	COLS9+54,	COLS9+54,
	COLS10+43,	COLS10+43,	COLS9+12,	COLS9+12,
	COLS9+3,		COLS9+3,		COLS9+25,	COLS9+25,
	COLS9+27,	COLS9+27,	COLS9+35,	COLS9+35,

	COLS9,		COLS9,		COLS9+51,	COLS9+51,	// Norwegian
	COLS9+43,	COLS9+43,	COLS9+21,	COLS9+21,
	COLS9+30,	COLS9+30,	COLS9+47,	COLS9+47,
	COLS10+43,	COLS10+43,	COLS9+12,	COLS9+12,
	COLS9+3,		COLS9+3,		COLS9+25,	COLS9+25,
	COLS9+27,	COLS9+27,	COLS9+35,	COLS9+35,

	COLS9+48,	COLS9+48,	COLS9+44,	COLS9+44,	// Finnish/Swedish
	COLS9+1,		COLS9+1,		COLS9+52,	COLS9+52,
	COLS9+30,	COLS9+30,	COLS9+21,	COLS9+21,	// Oct98 U Diaer no longer to y Diaer
	COLS10+43,	COLS10+43,	COLS9+12,	COLS9+12,
	COLS9+3,		COLS9+3,		COLS9+25,	COLS9+25,
	COLS9+27,	COLS9+27,	COLS9+35,	COLS9+35,

	COLS9,		COLS9,		COLS9,		COLS9,		// Ukrain
	COLS9+1,		COLS9+1,		COLS9+21,	COLS9+21,
	COLS9+30,	COLS9+30,	COLS9+21,	COLS9+21,
	COLS10+48,	COLS10+48,	COLS9+12,	COLS9+12,
	COLS9+3,		COLS9+3,		COLS9+25,	COLS9+25,
	COLS9+27,	COLS9+27,	COLS9+35,	COLS9+35,

	COLS9,		COLS9,		COLS9,		COLS9,		// Turkish
	COLS9+1,		COLS9+1,		COLS9+21,	COLS9+21,
	COLS9+30,	COLS9+30,	COLS9+21,	COLS9+21,
	COLS9+43,	COLS9+43,	COLS9+11,	COLS9+11,	// dotless i same as
	COLS9+3,		COLS9+3,		COLS9+25,	COLS9+25,	// the "CH" in Czech
	COLS9+27,	COLS9+27,	COLS9+35,	COLS9+35,	// works because char
																	// fails brkcar()

	COLS9,		COLS9,		COLS9,		COLS9,		// Czech / Slovak
	COLS9+1,		COLS9+1,		COLS9+21,	COLS9+21,
	COLS9+30,	COLS9+30,	COLS9+21,	COLS9+21,
	COLS10+43,	COLS10+43,	COLS9+12,	COLS9+12,
	COLS9+5,		COLS9+5,		COLS9+26,	COLS9+26,	// carons
	COLS9+28,	COLS9+28,	COLS9+36,	COLS9+36
};

/****************************************************************************
Desc:
****************************************************************************/
static FLMBYTE fwp_asc60Tbl[ ASCTBLLEN + 2] =
{
	0x20,			// initial character offset!!
	ASCTBLLEN,	// len of this table
	COLLS,		// <Spc>
	COLLS+5,		// !
	COLS1,		// "
	COLS6+1,		// #
	COLS3,		// $
	COLS6,		// %
	COLS6+2,		// &
	COLS1+1,		// '
	COLS2,		// (
	COLS2+1,		// )
	COLS4+2,		// *
	COLS4,		// +
	COLLS+2,		// ,
	COLS4+1,		// -
	COLLS+1,		// .
	COLS4+3,		// /
	COLS8,		// 0
	COLS8+1,		// 1
	COLS8+2,		// 2
	COLS8+3,		// 3
	COLS8+4,		// 4
	COLS8+5,		// 5
	COLS8+6,		// 6
	COLS8+7,		// 7
	COLS8+8,		// 8
	COLS8+9,		// 9
	COLLS+3,		// :
	COLLS+4,		// ;
	COLS5,		// <
	COLS5+2,		// =
	COLS5+4,		// >
	COLLS+7,		// ?
	COLS6+3,		// @
	COLS9,		// A
	COLS9+2,		// B
	COLS9+3,		// C
	COLS9+6,		// D
	COLS9+7,		// E
	COLS9+8,		// F
	COLS9+9,		// G
	COLS9+10,	// H
	COLS9+12,	// I
	COLS9+14,	// J
	COLS9+15,	// K
	COLS9+16,	// L
	COLS9+18,	// M
	COLS9+19,	// N
	COLS9+21,	// O
	COLS9+23,	// P
	COLS9+24,	// Q
	COLS9+25,	// R
	COLS9+27,	// S
	COLS9+29,	// T
	COLS9+30,	// U
	COLS9+31,	// V
	COLS9+32,	// W
	COLS9+33,	// X
	COLS9+34,	// Y
	COLS9+35,	// Z
	COLS9+40,	// [ (note: alphabetic - end of list)
	COLS6+4,		// Backslash
	COLS9+41,	// ] (note: alphabetic - end of list)
	COLS4+4,		// ^
	COLS6+5,		// _
	COLS1+2,		// `
	COLS9,		// a
	COLS9+2,		// b
	COLS9+3,		// c
	COLS9+6,		// d
	COLS9+7,		// e
	COLS9+8,		// f
	COLS9+9,		// g
	COLS9+10,	// h
	COLS9+12,	// i
	COLS9+14,	// j
	COLS9+15,	// k
	COLS9+16,	// l
	COLS9+18,	// m
	COLS9+19,	// n
	COLS9+21,	// o
	COLS9+23,	// p
	COLS9+24,	// q
	COLS9+25,	// r
	COLS9+27,	// s
	COLS9+29,	// t
	COLS9+30,	// u
	COLS9+31,	// v
	COLS9+32,	// w
	COLS9+33,	// x
	COLS9+34,	// y
	COLS9+35,	// z
	COLS2+4,		// {
	COLS6+6,		// |
	COLS2+5,		// }
	COLS6+7		// ~
};

/****************************************************************************
Desc:
****************************************************************************/
static FLMBYTE fwp_mn60Tbl[ MNTBLLEN + 2] =
{
	23,			// initial character offset!!
	MNTBLLEN,	// len of this table
	COLS9+27,	// German Double s
	COLS9+15,	// Icelandic k
	COLS9+14,	// Dotless j

// IBM Charset

	COLS9,		// A Acute
	COLS9,		// a Acute
	COLS9,		// A Circumflex
	COLS9,		// a Circumflex
	COLS9,		// A Diaeresis or Umlaut
	COLS9,		// a Diaeresis or Umlaut
	COLS9,		// A Grave
	COLS9,		// a Grave
	COLS9,		// A Ring
	COLS9,		// a Ring
	COLS9+1,		// AE digraph
	COLS9+1,		// ae digraph
	COLS9+3,		// C Cedilla
	COLS9+3,		// c Cedilla
	COLS9+7,		// E Acute
	COLS9+7,		// e Acute
	COLS9+7,		// E Circumflex
	COLS9+7,		// e Circumflex
	COLS9+7,		// E Diaeresis or Umlaut
	COLS9+7,		// e Diaeresis or Umlaut
	COLS9+7,		// E Grave
	COLS9+7,		// e Grave
	COLS9+12,	// I Acute
	COLS9+12,	// i Acute
	COLS9+12,	// I Circumflex
	COLS9+12,	// i Circumflex
	COLS9+12,	// I Diaeresis or Umlaut
	COLS9+12,	// i Diaeresis or Umlaut
	COLS9+12,	// I Grave
	COLS9+12,	// i Grave
	COLS9+20,	// N Tilde
	COLS9+20,	// n Tilde
	COLS9+21,	// O Acute
	COLS9+21,	// o Acute
	COLS9+21,	// O Circumflex
	COLS9+21,	// o Circumflex
	COLS9+21,	// O Diaeresis or Umlaut
	COLS9+21,	// o Diaeresis or Umlaut
	COLS9+21,	// O Grave
	COLS9+21,	// o Grave
	COLS9+30,	// U Acute
	COLS9+30,	// u Acute
	COLS9+30,	// U Circumflex
	COLS9+30,	// u Circumflex
	COLS9+30,	// U Diaeresis or Umlaut
	COLS9+30,	// u Diaeresis or Umlaut
	COLS9+30,	// U Grave
	COLS9+30,	// u Grave
	COLS9+34,	// Y Diaeresis or Umlaut
	COLS9+34,	// y Diaeresis or Umlaut

// IBM foreign

	COLS9,		// A Tilde
	COLS9,		// a Tilde
	COLS9+6,		// D Cross Bar
	COLS9+6,		// d Cross Bar
	COLS9+21,	// O Slash
	COLS9+21,	// o Slash
	COLS9+21,	// O Tilde
	COLS9+21,	// o Tilde
	COLS9+34,	// Y Acute
	COLS9+34,	// y Acute
	COLS9+6,		// Uppercase Eth
	COLS9+6,		// Lowercase Eth
	COLS9+37,	// Uppercase Thorn
	COLS9+37,	// Lowercase Thorn

// Teletex chars

	COLS9,		// A Breve
	COLS9,		// a Breve
	COLS9,		// A Macron
	COLS9,		// a Macron
	COLS9,		// A Ogonek
	COLS9,		// a Ogonek
	COLS9+3,		// C Acute
	COLS9+3,		// c Acute
	COLS9+3,		// C Caron or Hachek
	COLS9+3,		// c Caron or Hachek
	COLS9+3,		// C Circumflex
	COLS9+3,		// c Circumflex
	COLS9+3,		// C Dot Above
	COLS9+3,		// c Dot Above
	COLS9+6,		// D Caron or Hachek (Apostrophe Beside)
	COLS9+6,		// d Caron or Hachek (Apostrophe Beside)
	COLS9+7,		// E Caron or Hachek
	COLS9+7,		// e Caron or Hachek
	COLS9+7,		// E Dot Above
	COLS9+7,		// e Dot Above
	COLS9+7,		// E Macron
	COLS9+7,		// e Macron
	COLS9+7,		// E Ogonek
	COLS9+7,		// e Ogonek
	COLS9+9,		// G Acute
	COLS9+9,		// g Acute
	COLS9+9,		// G Breve
	COLS9+9,		// g Breve
	COLS9+9,		// G Caron or Hachek
	COLS9+9,		// g Caron or Hachek
	COLS9+9,		// G Cedilla (Apostrophe Under)
	COLS9+9,		// g Cedilla (Apostrophe Over)
	COLS9+9,		// G Circumflex
	COLS9+9,		// g Circumflex
	COLS9+9,		// G Dot Above
	COLS9+9,		// g Dot Above
	COLS9+10,	// H Circumflex
	COLS9+10,	// h Circumflex
	COLS9+10,	// H Cross Bar
	COLS9+10,	// h Cross Bar
	COLS9+12,	// I Dot Above (Sharp Accent)
	COLS9+12,	// i Dot Above (Sharp Accent)
	COLS9+12,	// I Macron
	COLS9+12,	// i Macron
	COLS9+12,	// I Ogonek
	COLS9+12,	// i Ogonek
	COLS9+12,	// I Tilde
	COLS9+12,	// i Tilde
	COLS9+13,	// IJ Digraph
	COLS9+13,	// ij Digraph
	COLS9+14,	// J Circumflex
	COLS9+14,	// j Circumflex
	COLS9+15,	// K Cedilla (Apostrophe Under)
	COLS9+15,	// k Cedilla (Apostrophe Under)
	COLS9+16,	// L Acute
	COLS9+16,	// l Acute
	COLS9+16,	// L Caron or Hachek (Apostrophe Beside)
	COLS9+16,	// l Caron or Hachek (Apostrophe Beside)
	COLS9+16,	// L Cedilla (Apostrophe Under)
	COLS9+16,	// l Cedilla (Apostrophe Under)
	COLS9+16,	// L Center Dot
	COLS9+16,	// l Center Dot
	COLS9+16,	// L Stroke
	COLS9+16,	// l Stroke
	COLS9+19,	// N Acute
	COLS9+19,	// n Acute
	COLS9+19,	// N Apostrophe
	COLS9+19,	// n Apostrophe
	COLS9+19,	// N Caron or Hachek
	COLS9+19,	// n Caron or Hachek
	COLS9+19,	// N Cedilla (Apostrophe Under)
	COLS9+19,	// n Cedilla (Apostrophe Under)
	COLS9+21,	// O Double Acute
	COLS9+21,	// o Double Acute
	COLS9+21,	// O Macron
	COLS9+21,	// o Macron
	COLS9+22,	// OE digraph
	COLS9+22,	// oe digraph
	COLS9+25,	// R Acute
	COLS9+25,	// r Acute
	COLS9+25,	// R Caron or Hachek
	COLS9+25,	// r Caron or Hachek
	COLS9+25,	// R Cedilla (Apostrophe Under)
	COLS9+25,	// r Cedilla (Apostrophe Under)
	COLS9+27,	// S Acute
	COLS9+27,	// s Acute
	COLS9+27,	// S Caron or Hachek
	COLS9+27,	// s Caron or Hachek
	COLS9+27,	// S Cedilla
	COLS9+27,	// s Cedilla
	COLS9+27,	// S Circumflex
	COLS9+27,	// s Circumflex
	COLS9+29,	// T Caron or Hachek (Apostrophe Beside)
	COLS9+29,	// t Caron or Hachek (Apostrophe Beside)
	COLS9+29,	// T Cedilla (Apostrophe Under)
	COLS9+29,	// t Cedilla (Apostrophe Under)
	COLS9+29,	// T Cross Bar
	COLS9+29,	// t Cross Bar
	COLS9+30,	// U Breve
	COLS9+30,	// u Breve
	COLS9+30,	// U Double Acute
	COLS9+30,	// u Double Acute
	COLS9+30,	// U Macron
	COLS9+30,	// u Macron
	COLS9+30,	// U Ogonek
	COLS9+30,	// u Ogonek
	COLS9+30,	// U Ring
	COLS9+30,	// u Ring
	COLS9+30,	// U Tilde
	COLS9+30,	// u Tilde
	COLS9+32,	// W Circumflex
	COLS9+32,	// w Circumflex
	COLS9+34,	// Y Circumflex
	COLS9+34,	// y Circumflex
	COLS9+35,	// Z Acute
	COLS9+35,	// z Acute
	COLS9+35,	// Z Caron or Hachek
	COLS9+35,	// z Caron or Hachek
	COLS9+35,	// Z Dot Above
	COLS9+35,	// z Dot Above
	COLS9+19,	// Uppercase Eng
	COLS9+19,	// Lowercase Eng

// Other

	COLS9+6,		// D Macron
	COLS9+6,		// d Macron
	COLS9+16,	// L Macron
	COLS9+16,	// l Macron
	COLS9+19,	// N Macron
	COLS9+19,	// n Macron
	COLS9+25,	// R Grave
	COLS9+25,	// r Grave
	COLS9+27,	// S Macron
	COLS9+27,	// s Macron
	COLS9+29,	// T Macron
	COLS9+29,	// t Macron
	COLS9+34,	// Y Breve
	COLS9+34,	// y Breve
	COLS9+34,	// Y Grave
	COLS9+34,	// y Grave
	COLS9+6,		// D Apostrophe Beside
	COLS9+6,		// d Apostrophe Beside
	COLS9+21,	// O Apostrophe Beside
	COLS9+21,	// o Apostrophe Beside
	COLS9+30,	// U Apostrophe Beside
	COLS9+30,	// u Apostrophe Beside
	COLS9+7,		// E breve
	COLS9+7,		// e breve
	COLS9+12,	// I breve
	COLS9+12,	// i breve
	COLS9+12,	// dotless I
	COLS9+12,	// dotless i
	COLS9+21,	// O breve
	COLS9+21		// o breve
};

/****************************************************************************
Desc:
****************************************************************************/
static FLMBYTE fwp_sym60Tbl[ SYMTBLLEN + 2] =
{
	11,			// initial character offset!!
	SYMTBLLEN,	// len of this table
	COLS3+2,		// pound
	COLS3+3,		// yen
	COLS3+4,		// pacetes
	COLS3+5,		// floren
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS3+1,		// cent
};

/****************************************************************************
Desc:
****************************************************************************/
static FLMBYTE fwp_grk60Tbl[ GRKTBLLEN + 2] =
{
	0,					// starting offset
	GRKTBLLEN,		// length
	COLS7,			// Uppercase Alpha
	COLS7,			// Lowercase Alpha
	COLS7+1,			// Uppercase Beta
	COLS7+1,			// Lowercase Beta
	COLS7+1,			// Uppercase Beta Medial
	COLS7+1,			// Lowercase Beta Medial
	COLS7+2,			// Uppercase Gamma
	COLS7+2,			// Lowercase Gamma
	COLS7+3,			// Uppercase Delta
	COLS7+3,			// Lowercase Delta
	COLS7+4,			// Uppercase Epsilon
	COLS7+4,			// Lowercase Epsilon
	COLS7+5,			// Uppercase Zeta
	COLS7+5,			// Lowercase Zeta
	COLS7+6,			// Uppercase Eta
	COLS7+6,			// Lowercase Eta
	COLS7+7,			// Uppercase Theta
	COLS7+7,			// Lowercase Theta
	COLS7+8,			// Uppercase Iota
	COLS7+8,			// Lowercase Iota
	COLS7+9,			// Uppercase Kappa
	COLS7+9,			// Lowercase Kappa
	COLS7+10,		// Uppercase Lambda
	COLS7+10,		// Lowercase Lambda
	COLS7+11,		// Uppercase Mu
	COLS7+11,		// Lowercase Mu
	COLS7+12,		// Uppercase Nu
	COLS7+12,		// Lowercase Nu
	COLS7+13,		// Uppercase Xi
	COLS7+13,		// Lowercase Xi
	COLS7+14,		// Uppercase Omicron
	COLS7+14,		// Lowercase Omicron
	COLS7+15,		// Uppercase Pi
	COLS7+15,		// Lowercase Pi
	COLS7+16,		// Uppercase Rho
	COLS7+16,		// Lowercase Rho
	COLS7+17,		// Uppercase Sigma
	COLS7+17,		// Lowercase Sigma
	COLS7+17,		// Uppercase Sigma Terminal
	COLS7+17,		// Lowercase Sigma Terminal
	COLS7+18,		// Uppercase Tau
	COLS7+18,		// Lowercase Tau
	COLS7+19,		// Uppercase Upsilon
	COLS7+19,		// Lowercase Upsilon
	COLS7+20,		// Uppercase Phi
	COLS7+20,		// Lowercase Phi
	COLS7+21,		// Uppercase Chi
	COLS7+21,		// Lowercase Chi
	COLS7+22,		// Uppercase Psi
	COLS7+22,		// Lowercase Psi
	COLS7+23,		// Uppercase Omega
	COLS7+23,		// Lowercase Omega

// Other Modern Greek Characters [8,52]

	COLS7,			// Uppercase ALPHA Tonos high prime
	COLS7,			// Lowercase Alpha Tonos - acute
	COLS7+4,			// Uppercase EPSILON Tonos - high prime
	COLS7+4,			// Lowercase Epslion Tonos - acute
	COLS7+6,			// Uppercase ETA Tonos - high prime
	COLS7+6,			// Lowercase Eta Tonos - acute
	COLS7+8,			// Uppercase IOTA Tonos - high prime
	COLS7+8,			// Lowercase iota Tonos - acute
	COLS7+8,			// Uppercase IOTA Diaeresis
	COLS7+8,			// Lowercase iota diaeresis
	COLS7+14,		// Uppercase OMICRON Tonos - high prime
	COLS7+14,		// Lowercase Omicron Tonos - acute
	COLS7+19,		// Uppercase UPSILON Tonos - high prime
	COLS7+19,		// Lowercase Upsilon Tonos - acute
	COLS7+19,		// Uppercase UPSILON Diaeresis
	COLS7+19,		// Lowercase Upsilon diaeresis
	COLS7+23,		// Uppercase OMEGA Tonos - high prime
	COLS7+23,		// Lowercase Omega Tonso - acute

// Variants [8,70]

	COLS7+4,			// epsilon (variant)
	COLS7+7,			// theta (variant)
	COLS7+9,			// kappa (variant)
	COLS7+15,		// pi (variant)
	COLS7+16,		// rho (variant)
	COLS7+17,		// sigma (variant)
	COLS7+19,		// upsilon (variant)
	COLS7+20,		// phi (variant)
	COLS7+23,		// omega (variant)

// Greek Diacritic marks [8,79]

	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,		// 8,108 end of diacritic marks

// Ancient Greek [8,109]

	COLS7,		// alpha grave
	COLS7,		// alpha circumflex
	COLS7,		// alpha w/iota
	COLS7,		// alpha acute w/iota
	COLS7,		// alpha grave w/iota
	COLS7,		// alpha circumflex w/Iota
	COLS7,		// alpha smooth
	COLS7,		// alpha smooth acute
	COLS7,		// alpha smooth grave
	COLS7,		// alpha smooth circumflex
	COLS7,		// alpha smooth w/Iota
	COLS7,		// alpha smooth acute w/Iota
	COLS7,		// alpha smooth grave w/Iota
	COLS7,		// alpha smooth circumflex w/Iota
// [8,123]
	COLS7,		// alpha rough
	COLS7,		// alpha rough acute
	COLS7,		// alpha rough grave
	COLS7,		// alpha rough circumflex
	COLS7,		// alpha rough w/Iota
	COLS7,		// alpha rough acute w/Iota
	COLS7,		// alpha rough grave w/Iota
	COLS7,		// alpha rough circumflex w/Iota
// [8,131]
	COLS7+4,		// epsilon grave
	COLS7+4,		// epsilon smooth
	COLS7+4,		// epsilon smooth acute
	COLS7+4,		// epsilon smooth grave
	COLS7+4,		// epsilon rough
	COLS7+4,		// epsilon rough acute
	COLS7+4,		// epsilon rough grave
// [8,138]
	COLS7+6,		// eta grave
	COLS7+6,		// eta circumflex
	COLS7+6,		// eta w/iota
	COLS7+6,		// eta acute w/iota
	COLS7+6,		// eta grave w/Iota
	COLS7+6,		// eta circumflex w/Iota
	COLS7+6,		// eta smooth
	COLS7+6,		// eta smooth acute
	COLS7+6,		// eta smooth grave
	COLS7+6,		// eta smooth circumflex
	COLS7+6,		// eta smooth w/Iota
	COLS7+6,		// eta smooth acute w/Iota
	COLS7+6,		// eta smooth grave w/Iota
	COLS7+6,		// eta smooth circumflex w/Iota
	COLS7+6,		// eta rough
	COLS7+6,		// eta rough acute
	COLS7+6,		// eta rough grave
	COLS7+6,		// eta rough circumflex
	COLS7+6,		// eta rough w/Iota
	COLS7+6,		// eta rough acute w/Iota
	COLS7+6,		// eta rough grave w/Iota
	COLS7+6,		// eta rough circumflex w/Iota
// [8,160]
	COLS7+8,		// iota grave
	COLS7+8,		// iota circumflex
	COLS7+8,		// iota acute diaeresis
	COLS7+8,		// iota grave diaeresis
	COLS7+8,		// iota smooth
	COLS7+8,		// iota smooth acute
	COLS7+8,		// iota smooth grave
	COLS7+8,		// iota smooth circumflex
	COLS7+8,		// iota rough
	COLS7+8,		// iota rough acute
	COLS7+8,		// iota rough grave
	COLS7+8,		// iota rough circumflex
// [8,172]
	COLS7+14,	// omicron grave
	COLS7+14,	// omicron smooth
	COLS7+14,	// omicron smooth acute
	COLS7+14,	// omicron smooth grave
	COLS7+14,	// omicron rough
	COLS7+14,	// omicron rough acute
	COLS7+14,	// omicron rough grave
// [8,179]
	COLS7+16,	// rho smooth
	COLS7+16,	// rho rough
// [8,181]
	COLS7+19,	// upsilon grave
	COLS7+19,	// upsilon circumflex
	COLS7+19,	// upsilon acute diaeresis
	COLS7+19,	// upsilon grave diaeresis
	COLS7+19,	// upsilon smooth
	COLS7+19,	// upsilon smooth acute
	COLS7+19,	// upsilon smooth grave
	COLS7+19,	// upsilon smooth circumflex
	COLS7+19,	// upsilon rough
	COLS7+19,	// upsilon rough acute
	COLS7+19,	// upsilon rough grave
	COLS7+19,	// upsilon rough circumflex
// [8,193]
	COLS7+23,	// omega grave
	COLS7+23,	// omega circumflex
	COLS7+23,	// omega w/Iota
	COLS7+23,	// omega acute w/Iota
	COLS7+23,	// omega grave w/Iota
	COLS7+23,	// omega circumflex w/Iota
	COLS7+23,	// omega smooth
	COLS7+23,	// omega smooth acute
	COLS7+23,	// omega smooth grave
	COLS7+23,	// omega smooth circumflex
	COLS7+23,	// omega smooth w/Iota
	COLS7+23,	// omega smooth acute w/Iota
	COLS7+23,	// omega smooth grave w/Iota
	COLS7+23,	// omega smooth circumflex w/Iota
	COLS7+23,	// omega rough
	COLS7+23,	// omega rough acute
	COLS7+23,	// omega rough grave
	COLS7+23,	// omega rough circumflex
	COLS7+23,	// omega rough w/Iota
	COLS7+23,	// omega rough acute w/Iota
	COLS7+23,	// omega rough grave w/Iota
	COLS7+23,	// omega rough circumflex w/Iota
// [8,215]
	COLS7+24,	// Uppercase Stigma--the number 6
	COLS7+24,	// Uppercase Digamma--Obsolete letter used as 6
	COLS7+24,	// Uppercase Koppa--Obsolete letter used as 90
	COLS7+24		// Uppercase Sampi--Obsolete letter used as 900
};

/****************************************************************************
Desc:
****************************************************************************/
static FLMBYTE fwp_cyrl60Tbl[ CYRLTBLLEN + 2] =
{
	0,					// starting offset
	CYRLTBLLEN,		// len of table

	COLS10,			// Russian uppercase A
	COLS10,			// Russian lowercase A
	COLS10+1,		// Russian uppercase BE
	COLS10+1,		// Russian lowercase BE
	COLS10+2,		// Russian uppercase VE
	COLS10+2,		// Russian lowercase VE
	COLS10+3,		// Russian uppercase GHE
	COLS10+3,		// Russian lowercase GHE
	COLS10+5,		// Russian uppercase DE
	COLS10+5,		// Russian lowercase DE

	COLS10+8,		// Russian uppercase E
	COLS10+8,		// Russian lowercase E
	COLS10+9,		// Russian lowercase YO
	COLS10+9,		// Russian lowercase YO
	COLS10+11,		// Russian uppercase ZHE
	COLS10+11,		// Russian lowercase ZHE
	COLS10+12,		// Russian uppercase ZE
	COLS10+12,		// Russian lowercase ZE
	COLS10+14,		// Russian uppercase I
	COLS10+14,		// Russian lowercase I

	COLS10+17,		// Russian uppercase SHORT I
	COLS10+17,		// Russian lowercase SHORT I
	COLS10+19,		// Russian uppercase KA
	COLS10+19,		// Russian lowercase KA
	COLS10+20,		// Russian uppercase EL
	COLS10+20,		// Russian lowercase EL
	COLS10+22,		// Russian uppercase EM
	COLS10+22,		// Russian lowercase EM
	COLS10+23,		// Russian uppercase EN
	COLS10+23,		// Russian lowercase EN

	COLS10+25,		// Russian uppercase O
	COLS10+25,		// Russian lowercase O
	COLS10+26,		// Russian uppercase PE
	COLS10+26,		// Russian lowercase PE
	COLS10+27,		// Russian uppercase ER
	COLS10+27,		// Russian lowercase ER
	COLS10+28,		// Russian uppercase ES
	COLS10+28,		// Russian lowercase ES
	COLS10+29,		// Russian uppercase TE
	COLS10+29,		// Russian lowercase TE

	COLS10+32,		// Russian uppercase U
	COLS10+32,		// Russian lowercase U
	COLS10+34,		// Russian uppercase EF
	COLS10+34,		// Russian lowercase EF
	COLS10+35,		// Russian uppercase HA
	COLS10+35,		// Russian lowercase HA
	COLS10+36,		// Russian uppercase TSE
	COLS10+36,		// Russian lowercase TSE
	COLS10+37,		// Russian uppercase CHE
	COLS10+37,		// Russian lowercase CHE

	COLS10+39,		// Russian uppercase SHA
	COLS10+39,		// Russian lowercase SHA
	COLS10+40,		// Russian uppercase SHCHA
	COLS10+40,		// Russian lowercase SHCHA
	COLS10+41,		// Russian lowercase ER (also hard sign)
	COLS10+41,		// Russian lowercase ER (also hard sign)
	COLS10+42,		// Russian lowercase ERY
	COLS10+42,		// Russian lowercase ERY
	COLS10+43,		// Russian lowercase SOFT SIGN
	COLS10+43,		// Russian lowercase SOFT SIGN

	COLS10+45,		// Russian uppercase REVERSE E
	COLS10+45,		// Russian lowercase REVERSE E
	COLS10+46,		// Russian uppercase YU
	COLS10+46,		// Russian lowercase yu
	COLS10+47,		// Russian uppercase YA
	COLS10+47,		// Russian lowercase ya

	COLS0,			// Russian uppercase EH
	COLS0,			// Russian lowercase eh
	COLS10+7,		// Macedonian uppercase SOFT DJ
	COLS10+7,		// Macedonian lowercase soft dj

	COLS10+4,		// Ukrainian uppercase HARD G
	COLS10+4,		// Ukrainian lowercase hard g
	COLS0,			// GE bar
	COLS0,			// ge bar
	COLS10+6,		// Serbian uppercase SOFT DJ
	COLS10+6,		// Serbian lowercase SOFT DJ
	COLS0,			// IE (variant)
	COLS0,			// ie (variant)
	COLS10+10,		// Ukrainian uppercase YE
	COLS10+10,		// Ukrainian lowercase YE

	COLS0,			// ZHE with right descender
	COLS0,			// zhe with right descender
	COLS10+13,		// Macedonian uppercase ZELO
	COLS10+13,		// Macedonian lowercase ZELO
	COLS0,			// Old Slovanic uppercase Z
	COLS0,			// Old Slovanic uppercase z
	COLS0,			// II with macron
	COLS0,			// ii with mscron
	COLS10+15,		// Ukrainian uppercase I
	COLS10+15,		// Ukrainian lowercase I

	COLS10+16,		// Ukrainian uppercase I with Two Dots
	COLS10+16,		// Ukrainian lowercase I with Two Dots
	COLS0,			// Old Slovanic uppercase I ligature
	COLS0,			// Old Slovanic lowercase I ligature
	COLS10+18,		// Serbian--Macedonian uppercase JE
	COLS10+18,		// Serbian--Macedonian lowercase JE
	COLS10+31,		// Macedonian uppercase SOFT K
	COLS10+31,		// Macedonian lowercase SOFT K
	COLS0,			// KA with right descender
	COLS0,			// ka with right descender

	COLS0,			// KA ogonek
	COLS0,			// ka ogonek
	COLS0,			// KA vertical bar
	COLS0,			// ka vertical bar
	COLS10+21,		// Serbian--Macedonian uppercase SOFT L
	COLS10+21,		// Serbian--Macedonian lowercase SOFT L
	COLS0,			// EN with right descender
	COLS0,			// en with right descender
	COLS10+24,		// Serbian--Macedonian uppercase SOFT N
	COLS10+24,		// Serbian--Macedonian lowercase SOFT N

	COLS0,			// ROUND OMEGA
	COLS0,			// round omega
	COLS0,			// OMEGA
	COLS0,			// omega
	COLS10+30,		// Serbian uppercase SOFT T
	COLS10+30,		// Serbian lowercase SOFT T
	COLS10+33,		// Byelorussian uppercase SHORT U
	COLS10+33,		// Byelorussian lowercase SHORT U
	COLS0,			// U with macron
	COLS0,			// u with macron

	COLS0,			// STRAIGHT U
	COLS0,			// straight u
	COLS0,			// STRAIGHT U bar
	COLS0,			// straight u bar
	COLS0,			// OU ligature
	COLS0,			// ou ligature
	COLS0,			// KHA with right descender
	COLS0,			// kha with right descender
	COLS0,			// KHA ogonek
	COLS0,			// kha ogonek

	COLS0,			// H
	COLS0,			// h
	COLS0,			// OMEGA titlo
	COLS0,			// omega titlo
	COLS10+38,		// Serbian uppercase HARD DJ
	COLS10+38,		// Serbian lowercase HARD DJ
	COLS0,			// CHE with right descender
	COLS0,			// che with right descender
	COLS0,			// CHE vertical bar
	COLS0,			// che vertical bar

	COLS0,			// Old Slavonic SHCHA (variant)
	COLS0,			// old SLAVONIC shcha (variant)
	COLS10+44,		// Old Russian uppercase YAT
	COLS10+44,		// Old Russian lowercase YAT

// END OF UNIQUE COLLATED BYTES
// CHARACTERS BELOW MUST HAVE HAVE THEIR OWN
// SUB-COLLATION VALUE TO COMPARE CORRECTLY.

	COLS0,			// Old Bulgarian uppercase YUS
	COLS0,			// Old Bulgarian lowercase YUS
	COLS0,			// Old Slovanic uppercase YUS MALYI
	COLS0,			// Old Slovanic uppercase YUS MALYI
	COLS0,			// KSI
	COLS0,			// ksi

	COLS0,			// PSI
	COLS0,			// psi
	COLS0,			// Old Russian uppercase FITA
	COLS0,			// Old Russian lowercase FITA
	COLS0,			// Old Russian uppercase IZHITSA
	COLS0,			// Old Russian lowercase IZHITSA
	COLS0,			// Russian uppercase A acute
	COLS0,			// Russian lowercase A acute
	COLS10+8,		// Russian uppercase E acute
	COLS10+8,		// Russian lowercase E acute

// 160-below all characters are russian to 199

	COLS0,			// E acute
	COLS0,			// e acute
	COLS10+14,		// II acute
	COLS10+14,		// ii acute
	COLS0,			// I acute
	COLS0,			// i acute
	COLS0,			// YI acute
	COLS0,			// yi acute
	COLS10+25,		// O acute
	COLS10+25,		// o acute

	COLS10+32,		// U acute
	COLS10+32,		// u acute
	COLS10+42,		// YERI acute
	COLS10+42,		// YERI acute
	COLS10+45,		// REVERSED E acute
	COLS10+45,		// reversed e acute
	COLS10+46,		// YU acute
	COLS10+46,		// yu acute
	COLS10+47,		// YA acute
	COLS10+47,		// ya acute

	COLS10,			// A grave
	COLS10,			// a grave
	COLS10+8,		// E grave
	COLS10+8,		// e grave
	COLS10+9,		// YO grave
	COLS10+9,		// yo grave
	COLS10+14,		// I grave
	COLS10+14,		// i grave
	COLS10+25,		// O grave
	COLS10+25,		// o grave

	COLS10+32,		// U grave
	COLS10+32,		// u grave
	COLS10+42,		// YERI grave
	COLS10+42,		// yeri grave
	COLS10+45,		// REVERSED E grave
	COLS10+45,		// reversed e grave
	COLS10+46,		// IU (YU) grave
	COLS10+46,		// iu (yu) grave
	COLS10+47,		// ia (YA) grave
	COLS10+47,		// ia (ya) grave ******* [10,199]
};

/****************************************************************************
Desc:		The Hebrew characters are collated over the Russian characters
			Therefore sorting both Hebrew and Russian is impossible to do.
****************************************************************************/
static FLMBYTE fwp_heb60TblA[ HEBTBL1LEN + 2] =
{
	0,					// starting offset
	HEBTBL1LEN,		// len of table
	COLS10h+0,		// Alef
	COLS10h+1,		// Bet
	COLS10h+2,		// Gimel
	COLS10h+3,		// Dalet
	COLS10h+4,		// He
	COLS10h+5,		// Vav
	COLS10h+6,		// Zayin
	COLS10h+7,		// Het
	COLS10h+8,		// Tet
	COLS10h+9,		// Yod
	COLS10h+10,		// Kaf (final) [9,10]
	COLS10h+11,		// Kaf
	COLS10h+12,		// Lamed
	COLS10h+13,		// Mem (final)
	COLS10h+14,		// Mem
	COLS10h+15,		// Nun (final)
	COLS10h+16,		// Nun
	COLS10h+17,		// Samekh
	COLS10h+18,		// Ayin
	COLS10h+19,		// Pe (final)
	COLS10h+20,		// Pe [9,20]
	COLS10h+21,		// Tsadi (final)
	COLS10h+22,		// Tsadi
	COLS10h+23,		// Qof
	COLS10h+24,		// Resh
	COLS10h+25,		// Shin
	COLS10h+26		// Tav [9,26]
};

/****************************************************************************
Desc:		This is the ANCIENT HEBREW SCRIPT piece.
			The actual value will be stored in the subcollation.
			This way we don't play diacritic/subcollation games.
****************************************************************************/
static FLMBYTE fwp_heb60TblB[ HEBTBL2LEN + 2] =
{
	84,
	HEBTBL2LEN,

// [9,84]
	COLS10h+0,		// Alef Dagesh [9,84]
	COLS10h+1,		// Bet Dagesh
	COLS10h+1,		// Vez - looks like a bet
	COLS10h+2,		// Gimel Dagesh
	COLS10h+3,		// Dalet Dagesh
	COLS10h+4,		// He Dagesh
	COLS10h+5,		// Vav Dagesh [9,90]
	COLS10h+5,		// Vav Holem
	COLS10h+6,		// Zayin Dagesh
	COLS10h+7,		// Het Dagesh
	COLS10h+8,		// Tet Dagesh
	COLS10h+9,		// Yod Dagesh
	COLS10h+9,		// Yod Hiriq [9,96] - not on my list

	COLS10h+11,		// Kaf Dagesh
	COLS10h+10,		// Kaf Dagesh (final)
	COLS10h+10,		// Kaf Sheva (final)
	COLS10h+10,		// Kaf Tsere (final) [9,100]
	COLS10h+10,		// Kaf Segol (final)
	COLS10h+10,		// Kaf Patah (final)
	COLS10h+10,		// Kaf Qamats (final)
	COLS10h+10,		// Kaf Dagesh Qamats (final)
	COLS10h+12,		// Lamed Dagesh
	COLS10h+14,		// Mem Dagesh
	COLS10h+16,		// Nun Dagesh
	COLS10h+15,		// Nun Qamats (final)
	COLS10h+17,		// Samekh Dagesh
	COLS10h+20,		// Pe Dagesh [9,110]
	COLS10h+20,		// Fe - just guessing this is like Pe - was +21
	COLS10h+22,		// Tsadi Dagesh
	COLS10h+23,		// Qof Dagesh
	COLS10h+25,		// Sin (with sin dot)
	COLS10h+25,		// Sin Dagesh (with sin dot)
	COLS10h+25,		// Shin
	COLS10h+25,		// Shin Dagesh
	COLS10h+26		// Tav Dagesh [9,118]
};

/****************************************************************************
Desc:		The Arabic characters are collated OVER the Russian characters
			Therefore sorting both Arabic and Russian in the same database
			is not supported.

			Arabic starts with a bunch of accents/diacritic marks that are
			Actually placed OVER a preceeding character.  These accents are
			ignored while sorting the first pass - when collation == COLS0.

			There are 4 possible states for all/most arabic characters:
				?? - occurs as the only character in a word
				?? - appears at the first of the word
				?? - appears at the middle of a word
				?? - appears at the end of the word

			Usually only the simple version of the letter is stored.
			Therefore we should not have to worry about sub-collation
			of these characters.

			The arabic characters with diacritics differ however.  The alef has
			sub-collation values to sort correctly.  There is not any more room
			to add more collation values.  Some chars in CS14 are combined when
			urdu, pashto and sindhi characters overlap.
****************************************************************************/
static FLMBYTE fwp_ar160Tbl[ AR1TBLLEN + 2] =
{
	38,				// starting offset
	AR1TBLLEN,		// len of table
// [13,38]
	COLLS+2,			// , comma
	COLLS+3,			// : colon
// [13,40]
	COLLS+7,			// ? question mark
	COLS4+2,			// * asterick
	COLS6,			// % percent
	COLS9+41,		// >> alphabetic - end of list)
	COLS9+40,		// << alphabetic - end of list)
	COLS2,			// (
	COLS2+1,			// )
// [13,47]
	COLS8+1,			// ?? One
	COLS8+2,			// ?? Two
	COLS8+3,			// ?? Three
// [13,50]
	COLS8+4,			// ?? Four
	COLS8+5,			// ?? Five
	COLS8+6,			// ?? Six
	COLS8+7,			// ?? Seven
	COLS8+8,			// ?? Eight
	COLS8+9,			// ?? Nine
	COLS8+0,			// ?? Zero
	COLS8+2,			// ?? Two (Handwritten)

	COLS10a+1,		// ?? alif
	COLS10a+1,		// ?? alif
// [13,60]
	COLS10a+2,		// ?? ba
	COLS10a+2,		// ?? ba
	COLS10a+2,		// ?? ba
	COLS10a+2,		// ?? ba
	COLS10a+6,		// ?? ta
	COLS10a+6,		// ?? ta
	COLS10a+6,		// ?? ta
	COLS10a+6,		// ?? ta
	COLS10a+8,		// ?? tha
	COLS10a+8,		// ?? tha
// [13,70]
	COLS10a+8,		// ?? tha
	COLS10a+8,		// ?? tha
	COLS10a+12,		// ?? jiim
	COLS10a+12,		// ?? jiim
	COLS10a+12,		// ?? jiim
	COLS10a+12,		// ?? jiim
	COLS10a+16,		// ?? Ha
	COLS10a+16,		// ?? Ha
	COLS10a+16,		// ?? Ha
	COLS10a+16,		// ?? Ha
// [13,80]
	COLS10a+17,		// ?? kha
	COLS10a+17,		// ?? kha
	COLS10a+17,		// ?? kha
	COLS10a+17,		// ?? kha
	COLS10a+20,		// ?? dal
	COLS10a+20,		// ?? dal
	COLS10a+22,		// ?? dhal
	COLS10a+22,		// ?? dhal
	COLS10a+27,		// ?? ra
	COLS10a+27,		// ?? ra
// [13,90]
	COLS10a+29,		// ?? ziin
	COLS10a+29,		// ?? ziin
	COLS10a+31,		// ?? siin
	COLS10a+31,		// ?? siin
	COLS10a+31,		// ?? siin
	COLS10a+31,		// ?? siin
	COLS10a+32,		// ?? shiin
	COLS10a+32,		// ?? shiin
	COLS10a+32,		// ?? shiin
	COLS10a+32,		// ?? shiin
// [13,100]
	COLS10a+34,		// ?? Sad
	COLS10a+34,		// ?? Sad
	COLS10a+34,		// ?? Sad
	COLS10a+34,		// ?? Sad
	COLS10a+35,		// ?? Dad
	COLS10a+35,		// ?? Dad
	COLS10a+35,		// ?? Dad
	COLS10a+35,		// ?? Dad
	COLS10a+36,		// ?? Ta
	COLS10a+36,		// ?? Ta
// [13,110]
	COLS10a+36,		// ?? Ta
	COLS10a+36,		// ?? Ta
	COLS10a+37,		// ?? Za
	COLS10a+37,		// ?? Za
	COLS10a+37,		// ?? Za
	COLS10a+37,		// ?? Za
	COLS10a+38,		// ?? 'ain
	COLS10a+38,		// ?? 'ain
	COLS10a+38,		// ?? 'ain
	COLS10a+38,		// ?? 'ain
// [13,120]
	COLS10a+39,		// ?? ghain
	COLS10a+39,		// ?? ghain
	COLS10a+39,		// ?? ghain
	COLS10a+39,		// ?? ghain
	COLS10a+40,		// ?? fa
	COLS10a+40,		// ?? fa
	COLS10a+40,		// ?? fa
	COLS10a+40,		// ?? fa
	COLS10a+42,		// ?? Qaf
	COLS10a+42,		// ?? Qaf
// [13,130]
	COLS10a+42,		// ?? Qaf
	COLS10a+42,		// ?? Qaf
	COLS10a+43,		// ?? kaf
	COLS10a+43,		// ?? kaf
	COLS10a+43,		// ?? kaf
	COLS10a+43,		// ?? kaf
	COLS10a+46,		// ?? lam
	COLS10a+46,		// ?? lam
	COLS10a+46,		// ?? lam
	COLS10a+46,		// ?? lam
// [13,140]
	COLS10a+47,		// ?? miim
	COLS10a+47,		// ?? miim
	COLS10a+47,		// ?? miim
	COLS10a+47,		// ?? miim
	COLS10a+48,		// ?? nuun
	COLS10a+48,		// ?? nuun
	COLS10a+48,		// ?? nuun
	COLS10a+48,		// ?? nuun
	COLS10a+49,		// ?? ha
	COLS10a+49,		// ?? ha
// [13,150]
	COLS10a+49,		// ?? ha
	COLS10a+49,		// ?? ha
						// ha is also 51 for non-arabic
	COLS10a+6, 		// ?? ta marbuuTah
	COLS10a+6, 		// ?? ta marbuuTah
	COLS10a+50,		// ?? waw
	COLS10a+50,		// ?? waw
	COLS10a+53,		// ?? ya
	COLS10a+53,		// ?? ya
	COLS10a+53,		// ?? ya
	COLS10a+53,		// ?? ya
// [13,160]
	COLS10a+52,		// ?? alif maqSuurah
	COLS10a+52,		// ?? ya   maqSuurah?
	COLS10a+52,		// ?? ya   maqSuurah?
	COLS10a+52,		// ?? alif maqSuurah

	COLS10a+0,		// ?? hamzah accent - never appears alone
// [13,165]

// Store the sub-collation as the actual
// character value from this point on

	COLS10a+1,		// ?? alif hamzah
	COLS10a+1,		// ?? alif hamzah
	COLS10a+1,		// ?? hamzah-under-alif
	COLS10a+1,		// ?? hamzah-under-alif
	COLS10a+1,		// ?? waw hamzah
// [13,170]
	COLS10a+1,		// ?? waw hamzah
	COLS10a+1,		// ?? ya hamzah
	COLS10a+1,		// ?? ya hamzah
	COLS10a+1,		// ?? ya hamzah
	COLS10a+1,		// ?? ya hamzah
	COLS10a+1,		// ?? alif fatHataan
	COLS10a+1,		// ?? alif fatHataan
	COLS10a+1,		// ?? alif maddah
	COLS10a+1,		// ?? alif maddah
	COLS10a+1,		// ?? alif waSlah
// [13,180]
	COLS10a+1,		// ?? alif waSlah (final)

//  LIGATURES
//    Should NEVER be stored so will not worry
//    about breaking up into pieces for collation.
//  NOTE:
//    Let's store the "Lam" collation value (+42)
//    below and in the sub-collation store the
//    actual character.  This will sort real close.
//    The best implementation is to
//    break up ligatures into its base pieces.

	COLS10a+46,		// ?? lamalif
	COLS10a+46,		// ?? lamalif
	COLS10a+46,		// ?? lamalif hamzah
	COLS10a+46,		// ?? lamalif hamzah
	COLS10a+46,		// ?? hamzah-under-lamalif
	COLS10a+46,		// ?? hamzah-under-lamalif
	COLS10a+46,		// ?? lamalif fatHataan
	COLS10a+46,		// ?? lamalif fatHataan
	COLS10a+46,		// ?? lamalif maddah
// [13,190]
	COLS10a+46,		// ?? lamalif maddah
	COLS10a+46,		// ?? lamalif waSlah
	COLS10a+46,		// ?? lamalif waSlah
	COLS10a+46,		// ?? Allah - khaDalAlif
	COLS0_ARABIC,	// ?? taTwiil     - character extension - throw out
	COLS0_ARABIC	// ?? taTwiil 1/6 - character extension - throw out
};

/****************************************************************************
Desc:		Alef needs a subcollation table.
			If colval==COLS10a+1 & char>=165
			index through this table.  Otherwise
			the alef value is [13,58] and subcol
			value should be 7.  Alef maddah is default (0)

			Handcheck if colval==COLS10a+6
			Should sort:
				[13,152]..[13,153] - taa marbuuTah - nosubcoll
				[13,64] ..[13,67]  - taa - subcoll of 1
****************************************************************************/
static FLMBYTE fwp_alefSubColTbl[] =
{
// [13,165]
	1,		// ?? alif hamzah
	1,		// ?? alif hamzah
	3,		// ?? hamzah-under-alif
	3,		// ?? hamzah-under-alif
	2,		// ?? waw hamzah
// [13,170]
	2,		// ?? waw hamzah
	4,		// ?? ya hamzah
	4,		// ?? ya hamzah
	4,		// ?? ya hamzah
	4,		// ?? ya hamzah
	5,		// ?? alif fatHataan
	5,		// ?? alif fatHataan
	0,		// ?? alif maddah
	0,		// ?? alif maddah
	6,		// ?? alif waSlah
// [13,180]
	6		// ?? alif waSlah (final)
};

/****************************************************************************
Desc:
****************************************************************************/
static FLMBYTE fwp_ar260Tbl[ AR2TBLLEN + 2] =
{
	41,				// starting offset
	AR2TBLLEN,		// len of table
// [14,41]
	COLS8+4,			// Farsi and Urdu Four
	COLS8+4,			// Urdu Four
	COLS8+5,			// Farsi and Urdu Five
	COLS8+6,			// Farsi Six
	COLS8+6,			// Farsi and Urdu Six
	COLS8+7,			// Urdu Seven
	COLS8+8,			// Urdu Eight

	COLS10a+3,		// Sindhi bb - baa /w 2 dots below (67b)
	COLS10a+3,
	COLS10a+3,
	COLS10a+3,
	COLS10a+4,		// Sindhi bh - baa /w 4 dots below (680)
	COLS10a+4,
	COLS10a+4,
	COLS10a+4,
// [14,56]
	COLS10a+5,		// Malay, Kurdish, Pashto, Farsi, Sindhi, and Urdu p
	COLS10a+5,		// =peh - taa /w 3 dots below (67e)
	COLS10a+5,
	COLS10a+5,
	COLS10a+7,		// Urdu T - taa /w small tah
	COLS10a+7,
	COLS10a+7,
	COLS10a+7,
	COLS10a+7,		// Pashto T - taa /w ring (forced to combine)
	COLS10a+7,
	COLS10a+7,
	COLS10a+7,
	COLS10a+9,		// Sindhi th - taa /w 4 dots above (67f)
	COLS10a+9,
// [14,70]
	COLS10a+9,
	COLS10a+9,
	COLS10a+10,		// Sindhi Tr - taa /w 3 dots above (67d)
	COLS10a+10,
	COLS10a+10,
	COLS10a+10,
	COLS10a+11,		// Sindhi Th - taa /w 2 dots above (67a)
	COLS10a+11,
	COLS10a+11,
	COLS10a+11,
	COLS10a+13,		// Sindhi jj - haa /w 2 middle dots verticle (684)
	COLS10a+13,
	COLS10a+13,
	COLS10a+13,
	COLS10a+14,		// Sindhi ny - haa /w 2 middle dots (683)
	COLS10a+14,
	COLS10a+14,
	COLS10a+14,
// [14,88]
	COLS10a+15,		// Malay, Kurdish, Pashto, Farsi, Sindhi, and Urdu ch
	COLS10a+15,		// =tcheh (686)
	COLS10a+15,
	COLS10a+15,
	COLS10a+15,		// Sindhi chh - haa /w middle 4 dots (687)
	COLS10a+15,		// forced to combine
	COLS10a+15,
	COLS10a+15,
	COLS10a+18,		// Pashto ts - haa /w 3 dots above (685)
	COLS10a+18,
	COLS10a+18,
	COLS10a+18,
	COLS10a+19,		// Pashto dz - hamzah on haa (681)
	COLS10a+19,
	COLS10a+19,
	COLS10a+19,
// [14,104]
	COLS10a+21,		// Urdu D - dal /w small tah (688)
	COLS10a+21,
	COLS10a+21,		// Pashto D - dal /w ring (689) forced to combine
	COLS10a+21,
	COLS10a+23,		// Sindhi dh - dal /w 2 dots above (68c)
	COLS10a+23,
	COLS10a+24,		// Sindhi D - dal /w 3 dots above (68e)
	COLS10a+24,
	COLS10a+25,		// Sindhi Dr - dal /w dot below (68a)
	COLS10a+25,
	COLS10a+26,		// Sindhi Dh - dal /w 2 dots below (68d)
	COLS10a+26,
	COLS10a+28,		// Pashto r - ra /w ring (693)
	COLS10a+28,
// [14,118]
	COLS10a+28,		// Urdu R - ra /w small tah (691) forced to combine
	COLS10a+28,
	COLS10a+28,		// Sindhi r - ra /w 4 dots above (699) forced to combine
	COLS10a+28,
	COLS10a+27,		// Kurdish rolled r - ra /w 'v' below (695)
	COLS10a+27,
	COLS10a+27,
	COLS10a+27,
// [14,126]
	COLS10a+30,		// Kurdish, Pashto, Farsi, Sindhi, and Urdu Z
	COLS10a+30,		// = jeh - ra /w 3 dots above (698)
	COLS10a+30,		// Pashto zz - ra /w dot below & dot above (696)
	COLS10a+30,		// forced to combine
	COLS10a+30,		// Pashto g - not in unicode! - forced to combine
	COLS10a+30,
	COLS10a+33,		// Pashto x - seen dot below & above (69a)
	COLS10a+33,
	COLS10a+33,
	COLS10a+33,
	COLS10a+39,		// Malay ng - old maly ain /w 3 dots above (6a0)
	COLS10a+39,		// forced to combine
	COLS10a+39,
	COLS10a+39,
// [14,140]
	COLS10a+41,		// Malay p, Kurdish v - Farsi ? - fa /w 3 dots above
	COLS10a+41,		// = veh - means foreign words (6a4)
	COLS10a+41,
	COLS10a+41,
	COLS10a+41,		// Sindhi ph - fa /w 4 dots above (6a6) forced to combine
	COLS10a+41,
	COLS10a+41,
	COLS10a+41,
// [14,148]
	COLS10a+43,		// Misc k - open caf (6a9)
	COLS10a+43,
	COLS10a+43,
	COLS10a+43,
	COLS10a+43,		// misc k - no unicode - forced to combine
	COLS10a+43,
	COLS10a+43,
	COLS10a+43,
	COLS10a+43,		// Sindhi k - swash caf (various) (6aa) -forced to combine
	COLS10a+43,
	COLS10a+43,
	COLS10a+43,
// [14,160]
	COLS10a+44,		// Persian/Urdu g - gaf (6af)
	COLS10a+44,
	COLS10a+44,
	COLS10a+44,
	COLS10a+44,		// Persian/Urdu g - no unicode
	COLS10a+44,
	COLS10a+44,
	COLS10a+44,
	COLS10a+44,		// malay g - gaf /w ring (6b0)
	COLS10a+44,
	COLS10a+44,
	COLS10a+44,
	COLS10a+44,		// Sindhi ng  - gaf /w 2 dots above (6ba)
	COLS10a+44,		// forced to combine ng only
	COLS10a+44,
	COLS10a+44,
	COLS10a+45,		// Sindhi gg - gaf /w 2 dots vertical below (6b3)
	COLS10a+45,
	COLS10a+45,
	COLS10a+45,
// [14,180]
	COLS10a+46,		// Kurdish velar l - lam /w small v (6b5)
	COLS10a+46,
	COLS10a+46,
	COLS10a+46,
	COLS10a+46,		// Kurdish Lamalif with diacritic - no unicode
	COLS10a+46,
// [14,186]
	COLS10a+48,		// Urdu n - dotless noon (6ba)
	COLS10a+48,
	COLS10a+48,
	COLS10a+48,
	COLS10a+48,		// Pashto N - noon /w ring (6bc) - forced to combine
	COLS10a+48,
	COLS10a+48,
	COLS10a+48,
	COLS10a+48,		// Sindhi N - dotless noon/w small tah (6bb)
	COLS10a+48,		// forced to combine
	COLS10a+48,
	COLS10a+48,
	COLS10a+50,		// Kurdish o - waw /w small v (6c6)
	COLS10a+50,
// [14,200]
	COLS10a+50,		// Kurdish o - waw /w bar above (6c5)
	COLS10a+50,
	COLS10a+50,		// Kurdish o - waw /w 2 dots above (6ca)
	COLS10a+50,
// [14,204]
	COLS10a+51,		// Urdu h - no unicode
	COLS10a+51,
	COLS10a+51,
	COLS10a+51,
	COLS10a+52,		// Kurdish ? - ya /w small v (6ce)
	COLS10a+52,
	COLS10a+52,
	COLS10a+52,
// [14,212]
	COLS10a+54,		// Urdu y - ya barree (6d2)
	COLS10a+54,
	COLS10a+54,		// Malay ny - ya /w 3 dots below (6d1) forced to combine
	COLS10a+54,
	COLS10a+54,
	COLS10a+54,
// [14,218]
	COLS10a+51,		// Farsi hamzah - hamzah on ha (6c0) forced to combine
	COLS10a+51
};

/****************************************************************************
Desc:		If the bit position is set then save the character in the sub-col
			area.  The bit values are determined by looking at the
			FLAIM COLTBL1 to see which characters are combined with other
			Arabic characters.
****************************************************************************/
static FLMBYTE fwp_ar2BitTbl[] =
{
	// Start at character 64
	// The only 'clean' areas uncollate to the correct place, they are...
						// 48..63
						// 68..91
						// 96..117
						// 126..127
						// 140..143
						// 160..163
						// 176..179
						// 212..213

	0xF0,				// 64..71
	0x00,				// 72..79
	0x00,				// 80..87
	0x0F,				// 88..95 - 92..95
	0x00,				// 96..103
	0x00,				// 104..111
	0x03,				// 112..119
	0xFC,				// 120..127
	0xFF,				// 128..135
	0xF0,				// 136..143 - 136..139
	0xFF,				// 144..151 - 144..147, 148..159
	0xFF,				// 152..159
	0x0F,				// 160..167 - 164..175
	0xFF,				// 168..175
	0x0F,				// 176..183 - 180..185
	0xFF,				// 184..191 - 186..197
	0xFF,				// 192..199 - 198..203
	0xFF,				// 200..207 - 204..207
	0xF3,				// 208..215 - 208..211 , 214..217
	0xF0				// 216..219 - 218..219
};

/****************************************************************************
Desc:		This table describes and gives addresses for collating 5.0
			character sets.  Each line corresponds with a character set.
***************************************************************************/
static TBL_B_TO_BP fwp_col60Tbl[] =
{
	{F_CHSASCI, fwp_asc60Tbl},
	{F_CHSMUL1, fwp_mn60Tbl},
	{F_CHSSYM1, fwp_sym60Tbl},
	{F_CHSGREK, fwp_grk60Tbl},
	{F_CHSCYR,  fwp_cyrl60Tbl},
	{0xFF, 	 	0}
};

/****************************************************************************
Desc:		This table is for sorting the hebrew/arabic languages.
			These values overlap the end of ASC/european and cyrillic tables.
****************************************************************************/
static TBL_B_TO_BP fwp_HebArabicCol60Tbl[] =
{
	{F_CHSASCI,	fwp_asc60Tbl},
	{F_CHSMUL1,	fwp_mn60Tbl},
	{F_CHSSYM1,	fwp_sym60Tbl},
	{F_CHSGREK,	fwp_grk60Tbl},
	{F_CHSHEB,	fwp_heb60TblA},
	{F_CHSHEB,	fwp_heb60TblB},
	{F_CHSARB1,	fwp_ar160Tbl},
	{F_CHSARB2,	fwp_ar260Tbl},
	{0xff, 		0}
};

/****************************************************************************
Desc:		The diacritical to collated table translates the first 26
			characters of WP character set #1 into a 5 bit value for "correct"
			sorting sequence for that diacritical (DCV) - diacritic collated
			value.

			The attempt here is to convert the collated character value
			along with the DCV to form the original WP character.

			The diacriticals are in an order to fit the most languages.
			Czech, Swedish, and Finnish will have to manual reposition the
			ring above (assign it a value greater then the umlaut)

			This table is index by the diacritical value.
****************************************************************************/
static FLMBYTE fwp_dia60Tbl[] =
{
	2,			// grave		offset = 0
	16,		//	centerd	offset = 1
	7,			//	tilde		offset = 2
	4,			//	circum	offset = 3
	12,		//	crossb	offset = 4
	10,		//	slash		offset = 5
	1,			//	acute		offset = 6
	6,			//	umlaut	offset = 7
				// In SU, SV and CZ will = 9
	17,		//	macron	offset = 8
	18,		//	aposab	offset = 9
	19,		//	aposbes	offset = 10
	20,		//	aposba	offset = 11
	21,		//	aposbc	offset = 12
	22,		//	abosbl	offset = 13
	8,			// ring		offset = 14
	13,		//	dota		offset = 15
	23,		//	dacute	offset = 16
	11,		//	cedilla	offset = 17
	14,		//	ogonek	offset = 18
	5,			//	caron		offset = 19
	15,		//	stroke	offset = 20
	24,		//	bara 		offset = 21
	3,			//	breve		offset = 22
	0,			// dbls		offset = 23 sorts as 'ss'
	25,		//	dotlesi	offset = 24
	26			// dotlesj	offset = 25
};

/****************************************************************************
Desc:		This table defines the range of characters within the set
			which are case convertible.
****************************************************************************/
static FLMBYTE fwp_caseConvertableRange[] =
{
	26,241,		// Multinational 1
	0,0,			// Multinational 2
	0,0,			// Box Drawing
	0,0,			// Symbol 1
	0,0,			// Symbol 2
	0,0,			// Math 1
	0,0,			// Math 2
	0,69,			// Greek 1
	0,0,			// Hebrew
	0,199,		// Cyrillic
	0,0,			// Japanese Kana
	0,0,			// User-defined
	0,0,			// Not defined
	0,0,			// Not defined
	0,0,			// Not defined
};

/****************************************************************************
Desc:
****************************************************************************/
static FLMUINT16 colToWPChr[ COLS11 - COLLS] =
{
	0x20,			// colls	-	<Spc>
	0x2e,			// colls+1	-	.
	0x2c,			// colls+2	-	,
	0x3a,			// colls+3	-	:
	0x3b,			// colls+4	-	;
	0x21,			// colls+5	-	!
	0,				// colls+6	-	NO VALUE
	0x3f,			// colls+7	-	?
	0,				// colls+8	-	NO VALUE

	0x22,			// cols1		-	"
	0x27,			// cols1+1	-	'
	0x60,			// cols1+2	-	`
	0,				// cols1+3	-	NO VALUE
	0,				// cols1+4	-	NO VALUE

	0x28,			// cols2		-	(
	0x29,			// cols2+1	-	)
	0x5b,			// cols2+2	-	japanese angle brackets
	0x5d,			// cols2+3	-	japanese angle brackets
	0x7b,			// cols2+4	-	{
	0x7d,			// cols2+5	-	}

	0x24,			// cols3		-	$
	0x413,		// cols3+1	-	cent
	0x40b,		// cols3+2	-	pound
	0x40c,		// cols3+3	-	yen
	0x40d,		// cols3+4	-	pacetes
	0x40e,		// cols3+5	-	floren

	0x2b,			// cols4		-	+
	0x2d,			// cols4+1	-	-
	0x2a,			// cols4+2	-	*
	0x2f,			// cols4+3	-	/
	0x5e,			// cols4+4	-	^
	0,				// cols4+5	-	NO VALUE
	0,				// cols4+6	-	NO VALUE
	0,				// cols4+7	-	NO VALUE

	0x3c,			// cols5		-	<
	0,				// cols5+1	-	NO VALUE
	0x3d,			// cols5+2	-	=
	0,				// cols5+3	-	NO VALUE
	0x3e,			// cols5+4	-	>
	0,				// cols5+5	-	NO VALUE
	0,				// cols5+6	-	NO VALUE
	0,				// cols5+7	-	NO VALUE
	0,				// cols5+8	-	NO VALUE
	0,				// cols5+9	-	NO VALUE
	0,				// cols5+10	-	NO VALUE
	0,				// cols5+11	-	NO VALUE
	0,				// cols5+12	-	NO VALUE
	0,				// cols5+13	-	NO VALUE

	0x25,			// cols6		-	%
	0x23,			// cols6+1	-	#
	0x26,			// cols6+2	-	&
	0x40,			// cols6+3	-	@
	0x5c,			// cols6+4	-	Backslash
	0x5f,			// cols6+5	-	_
	0x7c,			// cols6+6	-	|
	0x7e,			// cols6+7	-	~
	0,				// cols6+8	- NO VALUE
	0,				// cols6+9	- NO VALUE
	0,				// cols6+10	- NO VALUE
	0,				// cols6+11	- NO VALUE
	0,				// cols6+12	- NO VALUE

	0x800,		// cols7		-	Uppercase Alpha
	0x802,		// cols7+1	-	Uppercase Beta
	0x806,		// cols7+2	-	Uppercase Gamma
	0x808,		// cols7+3	-	Uppercase Delta
	0x80a,		// cols7+4	-	Uppercase Epsilon
	0x80c,		// cols7+5	-	Uppercase Zeta
	0x80e,		// cols7+6	-	Uppercase Eta
	0x810,		// cols7+7	-	Uppercase Theta
	0x812,		// cols7+8	-	Uppercase Iota
	0x814,		// cols7+9	-	Uppercase Kappa
	0x816,		// cols7+10	-	Uppercase Lambda
	0x818,		// cols7+11	-	Uppercase Mu
	0x81a,		// cols7+12	-	Uppercase Nu
	0x81c,		// cols7+13	-	Uppercase Xi
	0x81e,		// cols7+14	-	Uppercase Omicron
	0x820,		// cols7+15	-	Uppercase Pi
	0x822,		// cols7+16	-	Uppercase Rho
	0x824,		// cols7+17	-	Uppercase Sigma
	0x828,		// cols7+18	-	Uppercase Tau
	0x82a,		// cols7+19	-	Uppercase Upsilon
	0x82c,		// cols7+20	-	Uppercase Phi
	0x82e,		// cols7+21	-	Uppercase Chi
	0x830,		// cols7+22	-	Uppercase Psi
	0x832,		// cols7+23	-	Uppercase Omega
	0,				// cols7+24 - NO VALUE

	0x30,			// cols8		-	0
	0x31,			// cols8+1	-	1
	0x32,			// cols8+2	-	2
	0x33,			// cols8+3	-	3
	0x34,			// cols8+4	-	4
	0x35,			// cols8+5	-	5
	0x36,			// cols8+6	-	6
	0x37,			// cols8+7	-	7
	0x38,			// cols8+8	-	8
	0x39,			// cols8+9	-	9

	0x41,			// cols9		-	A
	0x124,		// cols9+1	-	AE digraph
	0x42,			// cols9+2	-	B
	0x43,			// cols9+3	-	C
	0xffff,		// cols9+4	-	CH in spanish
	0x162,		// cols9+5	-	Holder for C caron in Czech
	0x44,			// cols9+6	-	D
	0x45,			// cols9+7	-	E
	0x46,			// cols9+8	-	F
	0x47,			// cols9+9	-	G
	0x48,			// cols9+10	-	H
	0xffff,		// cols9+11	-	CH in czech or dotless i in turkish
	0x49,			// cols9+12	-	I
	0x18a,		// cols9+13	-	IJ Digraph
	0x4a,			// cols9+14	-	J
	0x4b,			// cols9+15	-	K
	0x4c,			// cols9+16	-	L
	0xffff,		// cols9+17	-	LL in spanish
	0x4d,			// cols9+18	-	M
	0x4e,			// cols9+19	-	N
	0x138,		// cols9+20	-	N Tilde
	0x4f,			// cols9+21	-	O
	0x1a6,		// cols9+22	-	OE digraph
	0x50,			// cols9+23	-	P
	0x51,			// cols9+24	-	Q
	0x52,			// cols9+25	-	R
	0x1aa,		// cols9+26	-	Holder for R caron in Czech
	0x53,			// cols9+27	-	S
	0x1b0,		// cols9+28	-	Holder for S caron in Czech
	0x54,			// cols9+29	-	T
	0x55,			// cols9+30	-	U
	0x56,			// cols9+31	-	V

	0x57,			// cols9+32	-	W
	0x58,			// cols9+33	-	X
	0x59,			// cols9+34	-	Y
	0x5a,			// cols9+35	-	Z
	0x1ce,		// cols9+36	-	Holder for Z caron in Czech
	0x158,		// cols9+37	-	Uppercase Thorn
	0,				// cols9+38	-	???
	0,				// cols9+39	-	???
	0x5b,			// cols9+40	-	[ (note: alphabetic - end of list)
	0x5d,			// cols9+41	-	] (note: alphabetic - end of list)
// 0xAA - also start of Hebrew
	0x124,		// cols9+42	- AE diagraph - DK
	0x124,		// cols9+43 - AE diagraph - NO
	0x122,		// cols9+44 - A ring      - SW
	0x11E,		// cols9+45 - A diaeresis - DK
	0x124,		// cols9+46	- AE diagraph - IC
	0x150,		// cols9+47 - O slash     - NO
	0x11e,		// cols9+48	- A diaeresis - SW
	0x150,		// cols9+49	- O slash     - DK
	0x13E,		// cols9+50	- O Diaeresis - IC
	0x122,		// cols9+51	- A ring      - NO
	0x13E,		// cols9+52	- O Diaeresis - SW
	0x13E,		// cols9+53	- O Diaeresis - DK
	0x150,		// cols9+54 - O slash     - IC
	0x122,		// cols9+55	- A ring      - DK
	0x124,		// cols9+56	- AE diagraph future
	0x13E,		// cols9+57 - O Diaeresis future
	0x150,		// cols9+58 - O slash     future
	0,				// cols9+59 - NOT USED    future

	0xA00,		// cols10		-	Russian A
	0xA02,		// cols10+1		-	Russian BE
	0xA04,		// cols10+2		-	Russian VE
	0xA06,		// cols10+3		-	Russian GHE
	0xA46,		// cols10+4		-	Ukrainian HARD G
	0xA08,		// cols10+5		-	Russian DE
	0xA4a,		// cols10+6		-	Serbian SOFT DJ
	0xA44,		// cols10+7		-	Macedonian SOFT DJ
	0xA0a,		// cols10+8		-	Russian E
	0xA0c,		// cols10+9		-	Russian YO
	0xA4e,		// cols10+10	-	Ukrainian YE
	0xA0e,		// cols10+11	-	Russian ZHE
	0xA10,		// cols10+12	-	Russian ZE
	0xA52,		// cols10+13	-	Macedonian ZELO
	0xA12,		// cols10+14	-	Russian I
	0xA58,		// cols10+15	-	Ukrainian I
	0xA5a,		// cols10+16	-	Ukrainian I with Two dots
	0xA14,		// cols10+17	-	Russian SHORT I
	0xA5e,		// cols10+18	-	Serbian--Macedonian JE
	0xA16,		// cols10+19	-	Russian KA
	0xA18,		// cols10+20	-	Russian EL
	0xA68,		// cols10+21	-	Serbian--Macedonian SOFT L
	0xA1a,		// cols10+22	-	Russian EM
	0xA1c,		// cols10+23	-	Russian EN
	0xA6c,		// cols10+24	-	Serbian--Macedonian SOFT N
	0xA1e,		// cols10+25	-	Russian O
	0xA20,		// cols10+26	-	Russian PE
	0xA22,		// cols10+27	-	Russian ER
	0xA24,		// cols10+28	-	Russian ES
	0xA26,		// cols10+29	-	Russian TE
	0xA72,		// cols10+30	-	Serbian SOFT T
	0xA60,		// cols10+31	-	Macedonian SOFT K
	0xA28,		// cols10+32	-	Russian U
	0xA74,		// cols10+33	-	Byelorussian SHORT U
	0xA2a,		// cols10+34	-	Russian EF
	0xA2c,		// cols10+35	-	Russian HA
	0xA2e,		// cols10+36	-	Russian TSE
	0xA30,		// cols10+37	-	Russian CHE
	0xA86,		// cols10+38	-	Serbian HARD DJ
	0xA32,		// cols10+39	-	Russian SHA
	0xA34,		// cols10+40	-	Russian SHCHA
	0xA36,		// cols10+41	-	Russian ER (also hard
	0xA38,		// cols10+42	-	Russian ERY
	0xA3a,		// cols10+43	-	Russian SOFT SIGN
	0xA8e,		// cols10+44	-	Old Russian YAT
	0xA3c,		// cols10+45	-	Russian uppercase	REVERSE E
	0xA3e,		// cols10+46	-	Russian YU
	0xA40,		// cols10+47	-	Russian YA
	0xA3a,		// cols10+48	-	Russian SOFT SIGN - UKRAIN ONLY
 	0				// cols10+49	- future
};

/****************************************************************************
Desc:
****************************************************************************/
static FLMUINT16 HebArabColToWPChr[] =
{
	// Start at COLS10a+0
// [0]
	0x0D00 +164,	// hamzah
	0x0D00 + 58,	// [13,177] alef maddah
						// Read subcollation to get other alef values
	0x0D00 + 60,	// baa
	0x0E00 + 48,	// Sindhi bb
	0x0E00 + 52,	// Sindhi bh
	0x0E00 + 56,	// Misc p = peh
	0x0D00 +152,	// taa marbuuTah
						// subcollation of 1 is taa [13,64]
	0x0E00 + 60,	// Urdu T   [14,60]
						// Pashto T [14,64]
// [8]
	0x0D00 + 68,	// thaa
	0x0E00 + 68,	// Sindhi th
	0x0E00 + 72,	// Sindhi tr
	0x0E00 + 76,	// Sindhi Th
	0x0D00 + 72,	// jiim - jeem
	0x0E00 + 80,	// Sindhi jj
	0x0E00 + 84,	// Sindhi ny
	0x0E00 + 88,	// Misc ch
						// Sinhi chh [14,92]
// [16]
	0x0D00 + 76,	// Haa
	0x0D00 + 80,	// khaa
	0x0E00 + 96,	// Pashto ts
	0x0E00 +100,	// Pashto dz

	0x0D00 + 84,	// dal
	0x0E00 +104,	// Urdu D
						// Pashto D
	0x0D00 + 86,	// thal
	0x0E00 +108,	// Sindhi dh

// [24]
	0x0E00 +110,	// Sindhi D
	0x0E00 +112,	// Sindhi Dr
	0x0E00 +114,	// Sindhi Dh

	0x0D00 + 88,	// ra
						// Kurdish rolled r [14,122]
	0x0E00 +116,	// Pashto r [14,116] - must pick this!
						// Urdu R   [14,118]
						// Sindhi r [14,120]

	0x0D00 + 90,	// zain
	0x0E00 +126,	// Mizc Z=jeh [14,126]
						// Pashto zz  [14,128]
						// Pashto g   [14,130]

	0x0D00 + 92,	// seen

// [32]
	0x0D00 + 96,	// sheen
	0x0E00 +132,	// Pashto x
	0x0D00 +100,	// Sad
	0x0D00 +104,	// Dad
	0x0D00 +108,	// Tah
	0x0D00 +112,	// Za (dhah)
	0x0D00 +116,	// 'ain
	0x0D00 +120,	// ghain
						// malay ng [14,136]
// [40]
	0x0D00 +124,	// fa
	0x0E00 +140,	// Malay p, kurdish v = veh
						// Sindhi ph [14,144]
	0x0D00 +128,	// Qaf
	0x0D00 +132,	// kaf (caf)
						// Misc k  [14,148]
						// misc k - no unicode [14,152]
						// Sindhi k [14,156]

	0x0E00 +160,	// Persian/Urdu gaf
						// gaf - no unicode [14,164]
						// malay g [14,168]
						// Sindhi ng [14,172]
	0x0E00 +176,	// Singhi gg

	0x0D00 +136,	// lam - all ligature variants
						// Kurdish valar lam [14,180]
						// Kurdish lamalef - no unicode [14,184]

	0x0D00 +140,	// meem

// [48]
	0x0D00 +144,	// noon
						// Urdu n [14,186]
						// Pashto N [14,190]
						// Sindhi N [14,194]
	0x0D00 +148,	// ha - arabic language only!
	0x0D00 +154,	// waw
						// Kurdish o [14,198]
						// Kurdish o with bar [14,200]
						// Kurdish o with 2 dots [14,202]
	0x0D00 +148,	// ha - non-arabic language
						// Urdu h [14,204]
						// Farsi hamzah on ha [14,218]
	0x0D00 +160,	// alef maqsurah
						// Kurdish e - ya /w small v

	0x0D00 +156,	// ya
	0x0E00 +212		// Urdu ya barree
						// Malay ny [14,214]
};

/****************************************************************************
Desc:
****************************************************************************/
static FLMUINT16 ArabSubColToWPChr[] =
{
	0x0D00 +177,	// Alef maddah - default value - here for documentation
	0x0D00 +165,	// Alef Hamzah
	0x0D00 +169,	// Waw hamzah
	0x0D00 +167,	// Hamzah under alef
	0x0D00 +171,	// ya hamzah
	0x0D00 +175,	// alef fathattan
	0x0D00 +179,	// alef waslah
	0x0D00 + 58,	// alef
	0x0D00 + 64		// taa - after taa marbuuTah
};

/****************************************************************************
Desc:		Turns a collated diacritic value into the original diacritic value
****************************************************************************/
static FLMBYTE ml1_COLtoD[27] =
{
	23,		// dbls	sort value = 0  sorts as 'ss'
	6,			// acute	sort value = 1
	0,			// grave	sort value = 2
	22,		// breve	sort value = 3
	3,			// circum	sort value = 4
	19,		// caron	sort value = 5
	7,			// umlaut	sort value = 6
	2,			// tilde	sort value = 7
	14,		// ring	sort value = 8
	7,			// umlaut in SU,SV & CZ after ring = 9
	5,			// slash	sort value = 10
	17,	 	// cedilla	sort value = 11
	4,			// crossb	sort value = 12
	15,	 	// dota	sort value = 13
	18,	 	// ogonek	sort value = 14
	20,	 	// stroke	sort value = 15
	1, 	 	// centerd	sort value = 16
	8,			// macron	sort value = 17
	9,			// aposab	sort value = 18
	10,	 	// aposbes	sort value = 19
	11,	 	// aposba	sort value = 20
	12,	 	// aposbc	sort value = 21
	13,	 	// abosbl	sort value = 22
	16,	 	// dacute	sort value = 23
	21,	 	// bara 	sort value = 24
	24,	 	// dotlesi	sort value = 25
	25			// dotlesj	sort value = 26
};

/****************************************************************************
Desc:
Notes:		Only 48 values + 0x40, 0x41, 0x42 (169..171)
****************************************************************************/
static FLMBYTE ColToKanaTbl[ 48] =
{
	 0,	// a=0, A=1
	 2,	// i=2, I=3
	 4,	// u=4, U=5, VU=83
	 6,	// e=6, E=7
 	 8,	// o=8, O=9
 	84,	// KA=10, GA=11, ka=84 - remember voicing table is optimized
 			//                       so that zero value is position and
 			//                       if voice=1 and no 0 is changed to 0
 	12,	// KI=12, GI=13
 	14,	// KU=14, GU=15
 	85,	// KE=16, GE=17, ke=85
 	18,	// KO=18, GO=19
 	20,	// SA=20, ZA=21
 	22,	// SHI=22, JI=23
 	24,	// SU=24, ZU=25
 	26,	// SE=26, ZE=27
 	28,	// SO=28, ZO=29
 	30,	// TA=30, DA=31
	32,	// CHI=32, JI=33
	34,	// tsu=34, TSU=35, ZU=36
	37,	// TE=37, DE=38
	39,	// TO=39, DO=40
	41,	// NA
	42,	// NI
	43,	// NU
	44,	// NE
	45,	// NO
	46,	// HA, BA, PA
	49,	// HI, BI, PI
	52,	// FU, BU, PU
	55,	// HE, BE, PE
	58,	// HO, BO, PO
	61,	// MA
	62,	// MI
	63,	// MU
	64,	// ME
	65,	// MO
	66,	// ya, YA
	68,	// yu, YU
	70,	// yo, YO
	72,	// RA
	73,	// RI
	74,	// RU
	75,	// RE
	76,	// RO
	77,	// wa, WA
	79,	// WI
	80,	// WE
	81,	// WO
	82		//  N
};

/****************************************************************************
Desc:
****************************************************************************/
static FLMBYTE f_langtbl[ FLM_LAST_LANG + FLM_LAST_LANG] =
{
	'U', 'S',	// English, United States
	'A', 'F',	// Afrikaans
	'A', 'R',	// Arabic
	'C', 'A',	// Catalan
	'H', 'R',	// Croatian
	'C', 'Z',	// Czech
	'D', 'K',	// Danish
	'N', 'L',	// Dutch
	'O', 'Z',	// English, Australia
	'C', 'E',	// English, Canada
	'U', 'K',	// English, United Kingdom
	'F', 'A',	// Farsi
	'S', 'U',	// Finnish
	'C', 'F',	// French, Canada
	'F', 'R',	// French, France
	'G', 'A',	// Galician
	'D', 'E',	// German, Germany
	'S', 'D',	// German, Switzerland
	'G', 'R',	// Greek
	'H', 'E',	// Hebrew
	'M', 'A',	// Hungarian
	'I', 'S',	// Icelandic
	'I', 'T',	// Italian
	'N', 'O',	// Norwegian
	'P', 'L',	// Polish
	'B', 'R',	// Portuguese, Brazil
	'P', 'O',	// Portuguese, Portugal
	'R', 'U',	// Russian
	'S', 'L',	// Slovak
	'E', 'S',	// Spanish
	'S', 'V',	// Swedish
	'Y', 'K',	// Ukrainian
	'U', 'R',	// Urdu
	'T', 'K',	// Turkey
	'J', 'P',	// Japanese
	'K', 'R',	// Korean
	'C', 'T',	// Chinese-Traditional
	'C', 'S',	// Chinese-Simplified
	'L', 'A'		// Future asian language
};

/****************************************************************************
Desc:		UNICODE to WP6 character mapping table
Notes:	This table is used to convert a subset of Unicode characters to
			their WordPerfect equivalents so that the WP collation routines
			can be used for indexing.  This contains characters that can be
			mapped 1:1 from Unicode->WP and from WP->Unicode.  There is
			no ambiguity and there are no character expansions or
			contractions.
****************************************************************************/
#define UTOWP60_ENTRIES			1502
static FLMUINT16 WP_UTOWP60[ UTOWP60_ENTRIES][2] =
{
	{ 0x00A1, 0x0407 },		//   7 ,  4
	{ 0x00A2, 0x0413 },		//  19 ,  4
	{ 0x00A3, 0x040b },		//  11 ,  4
	{ 0x00A4, 0x0418 },		//  24 ,  4
	{ 0x00A5, 0x040c },		//  12 ,  4
	{ 0x00A7, 0x0406 },		//   6 ,  4
	{ 0x00A9, 0x0417 },		//  23 ,  4
	{ 0x00AA, 0x040f },		//  15 ,  4
	{ 0x00AB, 0x0409 },		//   9 ,  4
	{ 0x00AC, 0x0614 },		//  20 ,  6
	{ 0x00AE, 0x0416 },		//  22 ,  4
	{ 0x00B0, 0x0624 },		//  36 ,  6
	{ 0x00B1, 0x0601 },		//   1 ,  6
	{ 0x00B2, 0x0414 },		//  20 ,  4
	{ 0x00B3, 0x041a },		//  26 ,  4
	{ 0x00B5, 0x0625 },		//  37 ,  6
	{ 0x00B6, 0x0405 },		//   5 ,  4
	{ 0x00B7, 0x0101 },		//  101,  1
	{ 0x00B9, 0x044e },		//  78 ,  4
	{ 0x00BA, 0x0410 },		//  16 ,  4
	{ 0x00BB, 0x040a },		//  10 ,  4
	{ 0x00BC, 0x0412 },		//  18 ,  4
	{ 0x00BD, 0x0411 },		//  17 ,  4
	{ 0x00BE, 0x0419 },		//  25 ,  4
	{ 0x00BF, 0x0408 },		//   8 ,  4
	{ 0x00C0, 0x0120 },		//  32 ,  1
	{ 0x00C1, 0x011a },		//  26 ,  1
	{ 0x00C2, 0x011c },		//  28 ,  1
	{ 0x00C3, 0x014c },		//  76 ,  1
	{ 0x00C4, 0x011e },		//  30 ,  1
	{ 0x00C5, 0x0122 },		//  34 ,  1
	{ 0x00C6, 0x0124 },		//  36 ,  1
	{ 0x00C7, 0x0126 },		//  38 ,  1
	{ 0x00C8, 0x012e },		//  46 ,  1
	{ 0x00C9, 0x0128 },		//  40 ,  1
	{ 0x00CA, 0x012a },		//  42 ,  1
	{ 0x00CB, 0x012c },		//  44 ,  1
	{ 0x00CC, 0x0136 },		//  54 ,  1
	{ 0x00CD, 0x0130 },		//  48 ,  1
	{ 0x00CE, 0x0132 },		//  50 ,  1
	{ 0x00CF, 0x0134 },		//  52 ,  1
	{ 0x00D0, 0x0156 },		//  86 ,  1
	{ 0x00D1, 0x0138 },		//  56 ,  1
	{ 0x00D2, 0x0140 },		//  64 ,  1
	{ 0x00D3, 0x013a },		//  58 ,  1
	{ 0x00D4, 0x013c },		//  60 ,  1
	{ 0x00D5, 0x0152 },		//  82 ,  1
	{ 0x00D6, 0x013e },		//  62 ,  1
	{ 0x00D7, 0x0627 },		//  39 ,  6
	{ 0x00D8, 0x0150 },		//  80 ,  1
	{ 0x00D9, 0x0148 },		//  72 ,  1
	{ 0x00DA, 0x0142 },		//  66 ,  1
	{ 0x00DB, 0x0144 },		//  68 ,  1
	{ 0x00DC, 0x0146 },		//  70 ,  1
	{ 0x00DD, 0x0154 },		//  84 ,  1
	{ 0x00DE, 0x0158 },		//  88 ,  1
	{ 0x00DF, 0x0117 },		//  23 ,  1
	{ 0x00E0, 0x0121 },		//  33 ,  1
	{ 0x00E1, 0x011b },		//  27 ,  1
	{ 0x00E2, 0x011d },		//  29 ,  1
	{ 0x00E3, 0x014d },		//  77 ,  1
	{ 0x00E4, 0x011f },		//  31 ,  1
	{ 0x00E5, 0x0123 },		//  35 ,  1
	{ 0x00E6, 0x0125 },		//  37 ,  1
	{ 0x00E7, 0x0127 },		//  39 ,  1
	{ 0x00E8, 0x012f },		//  47 ,  1
	{ 0x00E9, 0x0129 },		//  41 ,  1
	{ 0x00EA, 0x012b },		//  43 ,  1
	{ 0x00EB, 0x012d },		//  45 ,  1
	{ 0x00EC, 0x0137 },		//  55 ,  1
	{ 0x00ED, 0x0131 },		//  49 ,  1
	{ 0x00EE, 0x0133 },		//  51 ,  1
	{ 0x00EF, 0x0135 },		//  53 ,  1
	{ 0x00F0, 0x0157 },		//  87 ,  1
	{ 0x00F1, 0x0139 },		//  57 ,  1
	{ 0x00F2, 0x0141 },		//  65 ,  1
	{ 0x00F3, 0x013b },		//  59 ,  1
	{ 0x00F4, 0x013d },		//  61 ,  1
	{ 0x00F5, 0x0153 },		//  83 ,  1
	{ 0x00F6, 0x013f },		//  63 ,  1
	{ 0x00F7, 0x0608 },		//   8 ,  6
	{ 0x00F8, 0x0151 },		//  81 ,  1
	{ 0x00F9, 0x0149 },		//  73 ,  1
	{ 0x00FA, 0x0143 },		//  67 ,  1
	{ 0x00FB, 0x0145 },		//  69 ,  1
	{ 0x00FC, 0x0147 },		//  71 ,  1
	{ 0x00FD, 0x0155 },		//  85 ,  1
	{ 0x00FE, 0x0159 },		//  89 ,  1
	{ 0x00FF, 0x014b },		//  75 ,  1
	{ 0x0100, 0x015c },		//  92 ,  1
	{ 0x0101, 0x015d },		//  93 ,  1
	{ 0x0102, 0x015a },		//  90 ,  1
	{ 0x0103, 0x015b },		//  91 ,  1
	{ 0x0104, 0x015e },		//  94 ,  1
	{ 0x0105, 0x015f },		//  95 ,  1
	{ 0x0106, 0x0160 },		//  96 ,  1
	{ 0x0107, 0x0161 },		//  97 ,  1
	{ 0x0108, 0x0164 },		//  100,  1
	{ 0x0109, 0x0165 },		//  101,  1
	{ 0x010A, 0x0166 },		//  102,  1
	{ 0x010B, 0x0167 },		//  103,  1
	{ 0x010C, 0x0162 },		//  98 ,  1
	{ 0x010D, 0x0163 },		//  99 ,  1
	{ 0x010E, 0x0168 },		//  104,  1
	{ 0x010F, 0x0169 },		//  105,  1
	{ 0x0110, 0x014e },		//  78 ,  1
	{ 0x0111, 0x014f },		//  79 ,  1
	{ 0x0112, 0x016e },		//  110,  1
	{ 0x0113, 0x016f },		//  111,  1
	{ 0x0114, 0x01ea },		//  234,  1
	{ 0x0115, 0x01eb },		//  235,  1
	{ 0x0116, 0x016c },		//  108,  1
	{ 0x0117, 0x016d },		//  109,  1
	{ 0x0118, 0x0170 },		//  112,  1
	{ 0x0119, 0x0171 },		//  113,  1
	{ 0x011A, 0x016a },		//  106,  1
	{ 0x011B, 0x016b },		//  107,  1
	{ 0x011C, 0x017a },		//  122,  1
	{ 0x011D, 0x017b },		//  123,  1
	{ 0x011E, 0x0174 },		//  116,  1
	{ 0x011F, 0x0175 },		//  117,  1
	{ 0x0120, 0x017c },		//  124,  1
	{ 0x0121, 0x017d },		//  125,  1
	{ 0x0122, 0x0178 },		//  120,  1
	{ 0x0123, 0x0179 },		//  121,  1
	{ 0x0124, 0x017e },		//  126,  1
	{ 0x0125, 0x017f },		//  127,  1
	{ 0x0126, 0x0180 },		//  128,  1
	{ 0x0127, 0x0181 },		//  129,  1
	{ 0x0128, 0x0188 },		//  136,  1
	{ 0x0129, 0x0189 },		//  137,  1
	{ 0x012A, 0x0184 },		//  132,  1
	{ 0x012B, 0x0185 },		//  133,  1
	{ 0x012C, 0x01ec },		//  236,  1
	{ 0x012D, 0x01ed },		//  237,  1
	{ 0x012E, 0x0186 },		//  134,  1
	{ 0x012F, 0x0187 },		//  135,  1
	{ 0x0130, 0x0182 },		//  130,  1
	{ 0x0131, 0x01ef },		//  239,  1
	{ 0x0132, 0x018a },		//  138,  1
	{ 0x0133, 0x018b },		//  139,  1
	{ 0x0134, 0x018c },		//  140,  1
	{ 0x0135, 0x018d },		//  141,  1
	{ 0x0136, 0x018e },		//  142,  1
	{ 0x0137, 0x018f },		//  143,  1
	{ 0x0138, 0x0118 },		//  24 ,  1
	{ 0x0139, 0x0190 },		//  144,  1
	{ 0x013A, 0x0191 },		//  145,  1
	{ 0x013B, 0x0194 },		//  148,  1
	{ 0x013C, 0x0195 },		//  149,  1
	{ 0x013D, 0x0192 },		//  146,  1
	{ 0x013E, 0x0193 },		//  147,  1
	{ 0x013F, 0x0196 },		//  150,  1
	{ 0x0140, 0x0197 },		//  151,  1
	{ 0x0141, 0x0198 },		//  152,  1
	{ 0x0142, 0x0199 },		//  153,  1
	{ 0x0143, 0x019a },		//  154,  1
	{ 0x0144, 0x019b },		//  155,  1
	{ 0x0145, 0x01a0 },		//  160,  1
	{ 0x0146, 0x01a1 },		//  161,  1
	{ 0x0147, 0x019e },		//  158,  1
	{ 0x0148, 0x019f },		//  159,  1
	{ 0x0149, 0x019d },		//  157,  1
	{ 0x014A, 0x01d2 },		//  210,  1
	{ 0x014B, 0x01d3 },		//  211,  1
	{ 0x014C, 0x01a4 },		//  164,  1
	{ 0x014D, 0x01a5 },		//  165,  1
	{ 0x014E, 0x01f0 },		//  240,  1
	{ 0x014F, 0x01f1 },		//  241,  1
	{ 0x0150, 0x01a2 },		//  162,  1
	{ 0x0151, 0x01a3 },		//  163,  1
	{ 0x0152, 0x01a6 },		//  166,  1
	{ 0x0153, 0x01a7 },		//  167,  1
	{ 0x0154, 0x01a8 },		//  168,  1
	{ 0x0155, 0x01a9 },		//  169,  1
	{ 0x0156, 0x01ac },		//  172,  1
	{ 0x0157, 0x01ad },		//  173,  1
	{ 0x0158, 0x01aa },		//  170,  1
	{ 0x0159, 0x01ab },		//  171,  1
	{ 0x015A, 0x01ae },		//  174,  1
	{ 0x015B, 0x01af },		//  175,  1
	{ 0x015C, 0x01b4 },		//  180,  1
	{ 0x015D, 0x01b5 },		//  181,  1
	{ 0x015E, 0x01b2 },		//  178,  1
	{ 0x015F, 0x01b3 },		//  179,  1
	{ 0x0160, 0x01b0 },		//  176,  1
	{ 0x0161, 0x01b1 },		//  177,  1
	{ 0x0162, 0x01b8 },		//  184,  1
	{ 0x0163, 0x01b9 },		//  185,  1
	{ 0x0164, 0x01b6 },		//  182,  1
	{ 0x0165, 0x01b7 },		//  183,  1
	{ 0x0166, 0x01ba },		//  186,  1
	{ 0x0167, 0x01bb },		//  187,  1
	{ 0x0168, 0x01c6 },		//  198,  1
	{ 0x0169, 0x01c7 },		//  199,  1
	{ 0x016A, 0x01c0 },		//  192,  1
	{ 0x016B, 0x01c1 },		//  193,  1
	{ 0x016C, 0x01bc },		//  188,  1
	{ 0x016D, 0x01bd },		//  189,  1
	{ 0x016E, 0x01c4 },		//  196,  1
	{ 0x016F, 0x01c5 },		//  197,  1
	{ 0x0170, 0x01be },		//  190,  1
	{ 0x0171, 0x01bf },		//  191,  1
	{ 0x0172, 0x01c2 },		//  194,  1
	{ 0x0173, 0x01c3 },		//  195,  1
	{ 0x0174, 0x01c8 },		//  200,  1
	{ 0x0175, 0x01c9 },		//  201,  1
	{ 0x0176, 0x01ca },		//  202,  1
	{ 0x0177, 0x01cb },		//  203,  1
	{ 0x0178, 0x014a },		//  74 ,  1
	{ 0x0179, 0x01cc },		//  204,  1
	{ 0x017A, 0x01cd },		//  205,  1
	{ 0x017B, 0x01d0 },		//  208,  1
	{ 0x017C, 0x01d1 },		//  209,  1
	{ 0x017D, 0x01ce },		//  206,  1
	{ 0x017E, 0x01cf },		//  207,  1
	{ 0x0192, 0x040e },		//  14 ,  4
	{ 0x0194, 0x0a7c },		//  124, 10
	{ 0x01A0, 0x01e6 },		//  230,  1
	{ 0x01A1, 0x01e7 },		//  231,  1
	{ 0x01AF, 0x01e8 },		//  232,  1
	{ 0x01B0, 0x01e9 },		//  233,  1
	{ 0x01C0, 0x0605 },		//   5 ,  6
	{ 0x0250, 0x0237 },		//  55 ,  2
	{ 0x0251, 0x0238 },		//  56 ,  2
	{ 0x0252, 0x0239 },		//  57 ,  2
	{ 0x0253, 0x023a },		//  58 ,  2
	{ 0x0254, 0x023c },		//  60 ,  2
	{ 0x0255, 0x023d },		//  61 ,  2
	{ 0x0256, 0x023f },		//  63 ,  2
	{ 0x0257, 0x0240 },		//  64 ,  2
	{ 0x0258, 0x0241 },		//  65 ,  2
	{ 0x0259, 0x0242 },		//  66 ,  2
	{ 0x025A, 0x0243 },		//  67 ,  2
	{ 0x025B, 0x0244 },		//  68 ,  2
	{ 0x025C, 0x0245 },		//  69 ,  2
	{ 0x025D, 0x0246 },		//  70 ,  2
	{ 0x025E, 0x0248 },		//  72 ,  2
	{ 0x025F, 0x0249 },		//  73 ,  2
	{ 0x0260, 0x024c },		//  76 ,  2
	{ 0x0261, 0x024b },		//  75 ,  2
	{ 0x0262, 0x024d },		//  77 ,  2
	{ 0x0263, 0x024f },		//  79 ,  2
	{ 0x0264, 0x0250 },		//  80 ,  2
	{ 0x0265, 0x0251 },		//  81 ,  2
	{ 0x0266, 0x0252 },		//  82 ,  2
	{ 0x0267, 0x0253 },		//  83 ,  2
	{ 0x0268, 0x0255 },		//  85 ,  2
	{ 0x0269, 0x0257 },		//  87 ,  2
	{ 0x026A, 0x0256 },		//  86 ,  2
	{ 0x026B, 0x025a },		//  90 ,  2
	{ 0x026C, 0x025b },		//  91 ,  2
	{ 0x026D, 0x025c },		//  92 ,  2
	{ 0x026E, 0x025e },		//  94 ,  2
	{ 0x026F, 0x0260 },		//  96 ,  2
	{ 0x0270, 0x0261 },		//  97 ,  2
	{ 0x0271, 0x0262 },		//  98 ,  2
	{ 0x0272, 0x0263 },		//  99 ,  2
	{ 0x0273, 0x0264 },		//  100,  2
	{ 0x0274, 0x0265 },		//  101,  2
	{ 0x0275, 0x0279 },		//  121,  2
	{ 0x0276, 0x0266 },		//  102,  2
	{ 0x0277, 0x0267 },		//  103,  2
	{ 0x0278, 0x024a },		//  74 ,  2
	{ 0x0279, 0x0269 },		//  105,  2
	{ 0x027A, 0x026a },		//  106,  2
	{ 0x027B, 0x026b },		//  107,  2
	{ 0x027C, 0x026c },		//  108,  2
	{ 0x027D, 0x026d },		//  109,  2
	{ 0x027E, 0x026e },		//  110,  2
	{ 0x027F, 0x026f },		//  111,  2
	{ 0x0280, 0x0270 },		//  112,  2
	{ 0x0281, 0x0271 },		//  113,  2
	{ 0x0282, 0x0272 },		//  114,  2
	{ 0x0283, 0x0273 },		//  115,  2
	{ 0x0284, 0x0274 },		//  116,  2
	{ 0x0285, 0x0275 },		//  117,  2
	{ 0x0286, 0x0276 },		//  118,  2
	{ 0x0287, 0x0277 },		//  119,  2
	{ 0x0288, 0x0278 },		//  120,  2
	{ 0x0289, 0x027a },		//  122,  2
	{ 0x028A, 0x027b },		//  123,  2
	{ 0x028B, 0x027d },		//  125,  2
	{ 0x028C, 0x027c },		//  124,  2
	{ 0x028D, 0x027e },		//  126,  2
	{ 0x028E, 0x025f },		//  95 ,  2
	{ 0x028F, 0x0280 },		//  128,  2
	{ 0x0290, 0x0281 },		//  129,  2
	{ 0x0291, 0x0282 },		//  130,  2
	{ 0x0292, 0x0283 },		//  131,  2
	{ 0x0293, 0x0284 },		//  132,  2
	{ 0x0294, 0x0285 },		//  133,  2
	{ 0x0295, 0x0286 },		//  134,  2
	{ 0x0296, 0x0287 },		//  135,  2
	{ 0x0297, 0x023e },		//  62 ,  2
	{ 0x0298, 0x028a },		//  138,  2
	{ 0x0299, 0x023b },		//  59 ,  2
	{ 0x029A, 0x0247 },		//  71 ,  2
	{ 0x029B, 0x024e },		//  78 ,  2
	{ 0x029C, 0x0254 },		//  84 ,  2
	{ 0x029D, 0x0258 },		//  88 ,  2
	{ 0x029E, 0x0259 },		//  89 ,  2
	{ 0x029F, 0x025d },		//  93 ,  2
	{ 0x02A0, 0x0268 },		//  104,  2
	{ 0x02A1, 0x0288 },		//  136,  2
	{ 0x02A2, 0x0289 },		//  137,  2
	{ 0x02A3, 0x028b },		//  139,  2
	{ 0x02A4, 0x028c },		//  140,  2
	{ 0x02A5, 0x028d },		//  141,  2
	{ 0x02A6, 0x028e },		//  142,  2
	{ 0x02A7, 0x028f },		//  143,  2
	{ 0x02A8, 0x0290 },		//  144,  2
	{ 0x02B0, 0x0235 },		//  53 ,  2
	{ 0x02B6, 0x0236 },		//  54 ,  2
	{ 0x02B9, 0x0200 },		//   0 ,  2
	{ 0x02BA, 0x0201 },		//   1 ,  2
	{ 0x02BB, 0x0202 },		//   2 ,  2
	{ 0x02BC, 0x0205 },		//   5 ,  2
	{ 0x02BD, 0x0204 },		//   4 ,  2
	{ 0x02BE, 0x0207 },		//   7 ,  2
	{ 0x02BF, 0x0208 },		//   8 ,  2
	{ 0x02C6, 0x0217 },		//  23 ,  2
	{ 0x02C7, 0x0218 },		//  24 ,  2
	{ 0x02C8, 0x020f },		//  15 ,  2
	{ 0x02C9, 0x0211 },		//  17 ,  2
	{ 0x02CA, 0x0212 },		//  18 ,  2
	{ 0x02CB, 0x0213 },		//  19 ,  2
	{ 0x02CC, 0x0210 },		//  16 ,  2
	{ 0x02CD, 0x0214 },		//  20 ,  2
	{ 0x02CE, 0x0215 },		//  21 ,  2
	{ 0x02CF, 0x0216 },		//  22 ,  2
	{ 0x02D0, 0x020a },		//  10 ,  2
	{ 0x02D1, 0x020b },		//  11 ,  2
	{ 0x02D2, 0x022a },		//  42 ,  2
	{ 0x02D3, 0x022b },		//  43 ,  2
	{ 0x02DA, 0x021b },		//  27 ,  2
	{ 0x02DB, 0x0231 },		//  49 ,  2
	{ 0x02DC, 0x0219 },		//  25 ,  2
	{ 0x02DE, 0x0233 },		//  51 ,  2
	{ 0x0300, 0x0100 },		//   0 ,  1
	{ 0x0301, 0x0106 },		//   6 ,  1
	{ 0x0302, 0x0103 },		//   3 ,  1
	{ 0x0303, 0x0102 },		//   2 ,  1
	{ 0x0304, 0x0108 },		//   8 ,  1
	{ 0x0305, 0x0115 },		//  21 ,  1
	{ 0x0306, 0x0116 },		//  22 ,  1
	{ 0x0307, 0x010f },		//  15 ,  1
	{ 0x0308, 0x0107 },		//   7 ,  1
	{ 0x030A, 0x010e },		//  14 ,  1
	{ 0x030B, 0x0110 },		//  16 ,  1
	{ 0x030C, 0x0113 },		//  19 ,  1
	{ 0x0310, 0x0209 },		//   9 ,  2
	{ 0x0311, 0x0858 },		//  88 ,  8
	{ 0x0313, 0x0109 },		//   9 ,  1
	{ 0x0314, 0x085a },		//  90 ,  8
	{ 0x0315, 0x010a },		//  10 ,  1
	{ 0x031C, 0x0221 },		//  33 ,  2
	{ 0x031D, 0x0222 },		//  34 ,  2
	{ 0x031E, 0x0223 },		//  35 ,  2
	{ 0x031F, 0x0224 },		//  36 ,  2
	{ 0x0320, 0x0225 },		//  37 ,  2
	{ 0x0321, 0x0226 },		//  38 ,  2
	{ 0x0322, 0x0227 },		//  39 ,  2
	{ 0x0323, 0x021e },		//  30 ,  2
	{ 0x0324, 0x0220 },		//  32 ,  2
	{ 0x0325, 0x021a },		//  26 ,  2
	{ 0x0326, 0x010c },		//  12 ,  1
	{ 0x0327, 0x0111 },		//  17 ,  1
	{ 0x0328, 0x0112 },		//  18 ,  1
	{ 0x0329, 0x020e },		//  14 ,  2
	{ 0x032A, 0x0228 },		//  40 ,  2
	{ 0x032B, 0x0229 },		//  41 ,  2
	{ 0x032C, 0x021d },		//  29 ,  2
	{ 0x032D, 0x021c },		//  28 ,  2
	{ 0x032E, 0x020d },		//  13 ,  2
	{ 0x0335, 0x0104 },		//   4 ,  1
	{ 0x0337, 0x0114 },		//  20 ,  1
	{ 0x0338, 0x0105 },		//   5 ,  1
	{ 0x033E, 0x0230 },		//  48 ,  2
	{ 0x0345, 0x085b },		//  91 ,  8
	{ 0x0374, 0x0851 },		//  81 ,  8
	{ 0x0375, 0x0852 },		//  82 ,  8
	{ 0x0391, 0x0800 },		//   0 ,  8
	{ 0x0392, 0x0802 },		//   2 ,  8
	{ 0x0393, 0x0806 },		//   6 ,  8
	{ 0x0394, 0x0808 },		//   8 ,  8
	{ 0x0395, 0x080a },		//  10 ,  8
	{ 0x0396, 0x080c },		//  12 ,  8
	{ 0x0397, 0x080e },		//  14 ,  8
	{ 0x0398, 0x0810 },		//  16 ,  8
	{ 0x0399, 0x0812 },		//  18 ,  8
	{ 0x039A, 0x0814 },		//  20 ,  8
	{ 0x039B, 0x0816 },		//  22 ,  8
	{ 0x039C, 0x0818 },		//  24 ,  8
	{ 0x039D, 0x081a },		//  26 ,  8
	{ 0x039E, 0x081c },		//  28 ,  8
	{ 0x039F, 0x081e },		//  30 ,  8
	{ 0x03A0, 0x0820 },		//  32 ,  8
	{ 0x03A1, 0x0822 },		//  34 ,  8
	{ 0x03A3, 0x0824 },		//  36 ,  8
	{ 0x03A4, 0x0828 },		//  40 ,  8
	{ 0x03A5, 0x082a },		//  42 ,  8
	{ 0x03A6, 0x082c },		//  44 ,  8
	{ 0x03A7, 0x082e },		//  46 ,  8
	{ 0x03A8, 0x0830 },		//  48 ,  8
	{ 0x03A9, 0x0832 },		//  50 ,  8
	{ 0x03AA, 0x083c },		//  60 ,  8
	{ 0x03AB, 0x0842 },		//  66 ,  8
	{ 0x03AC, 0x0835 },		//  53 ,  8
	{ 0x03AD, 0x0837 },		//  55 ,  8
	{ 0x03AE, 0x0839 },		//  57 ,  8
	{ 0x03AF, 0x083b },		//  59 ,  8
	{ 0x03B1, 0x0801 },		//   1 ,  8
	{ 0x03B2, 0x0803 },		//   3 ,  8
	{ 0x03B3, 0x0807 },		//   7 ,  8
	{ 0x03B4, 0x0809 },		//   9 ,  8
	{ 0x03B5, 0x080b },		//  11 ,  8
	{ 0x03B6, 0x080d },		//  13 ,  8
	{ 0x03B7, 0x080f },		//  15 ,  8
	{ 0x03B8, 0x0811 },		//  17 ,  8
	{ 0x03B9, 0x0813 },		//  19 ,  8
	{ 0x03BA, 0x0815 },		//  21 ,  8
	{ 0x03BB, 0x0817 },		//  23 ,  8
	{ 0x03BC, 0x0819 },		//  25 ,  8
	{ 0x03BD, 0x081b },		//  27 ,  8
	{ 0x03BE, 0x081d },		//  29 ,  8
	{ 0x03BF, 0x081f },		//  31 ,  8
	{ 0x03C0, 0x0821 },		//  33 ,  8
	{ 0x03C1, 0x0823 },		//  35 ,  8
	{ 0x03C2, 0x0827 },		//  39 ,  8
	{ 0x03C3, 0x0825 },		//  37 ,  8
	{ 0x03C4, 0x0829 },		//  41 ,  8
	{ 0x03C5, 0x082b },		//  43 ,  8
	{ 0x03C6, 0x082d },		//  45 ,  8
	{ 0x03C7, 0x082f },		//  47 ,  8
	{ 0x03C8, 0x0831 },		//  49 ,  8
	{ 0x03C9, 0x0833 },		//  51 ,  8
	{ 0x03CA, 0x083d },		//  61 ,  8
	{ 0x03CB, 0x0843 },		//  67 ,  8
	{ 0x03CC, 0x083f },		//  63 ,  8
	{ 0x03CD, 0x0841 },		//  65 ,  8
	{ 0x03CE, 0x0845 },		//  69 ,  8
	{ 0x03D0, 0x0805 },		//   5 ,  8
	{ 0x03D1, 0x0847 },		//  71 ,  8
	{ 0x03D2, 0x084c },		//  76 ,  8
	{ 0x03D5, 0x084d },		//  77 ,  8
	{ 0x03D6, 0x0849 },		//  73 ,  8
	{ 0x03D7, 0x084f },		//  79 ,  8
	{ 0x03DA, 0x08d7 },		//  215,  8
	{ 0x03DB, 0x084B },		//  75 ,  8
	{ 0x03DC, 0x08d8 },		//  216,  8
	{ 0x03DE, 0x08d9 },		//  217,  8
	{ 0x03E0, 0x08da },		//  218,  8
	{ 0x03F0, 0x0848 },		//  72 ,  8
	{ 0x03F1, 0x084a },		//  74 ,  8
	{ 0x0401, 0x0a0c },		//  12 , 10
	{ 0x0402, 0x0a4a },		//  74 , 10
	{ 0x0403, 0x0a44 },		//  68 , 10
	{ 0x0404, 0x0a4e },		//  78 , 10
	{ 0x0405, 0x0a52 },		//  82 , 10
	{ 0x0406, 0x0a58 },		//  88 , 10
	{ 0x0407, 0x0a5a },		//  90 , 10
	{ 0x0408, 0x0a5e },		//  94 , 10
	{ 0x0409, 0x0a68 },		//  104, 10
	{ 0x040A, 0x0a6c },		//  108, 10
	{ 0x040B, 0x0a72 },		//  114, 10
	{ 0x040C, 0x0a60 },		//  96 , 10
	{ 0x040E, 0x0a74 },		//  116, 10
	{ 0x040F, 0x0a86 },		//  134, 10
	{ 0x0410, 0x0a00 },		//   0 , 10
	{ 0x0411, 0x0a02 },		//   2 , 10
	{ 0x0412, 0x0a04 },		//   4 , 10
	{ 0x0413, 0x0a06 },		//   6 , 10
	{ 0x0414, 0x0a08 },		//   8 , 10
	{ 0x0415, 0x0a0a },		//  10 , 10
	{ 0x0416, 0x0a0e },		//  14 , 10
	{ 0x0417, 0x0a10 },		//  16 , 10
	{ 0x0418, 0x0a12 },		//  18 , 10
	{ 0x0419, 0x0a14 },		//  20 , 10
	{ 0x041A, 0x0a16 },		//  22 , 10
	{ 0x041B, 0x0a18 },		//  24 , 10
	{ 0x041C, 0x0a1a },		//  26 , 10
	{ 0x041D, 0x0a1c },		//  28 , 10
	{ 0x041E, 0x0a1e },		//  30 , 10
	{ 0x041F, 0x0a20 },		//  32 , 10
	{ 0x0420, 0x0a22 },		//  34 , 10
	{ 0x0421, 0x0a24 },		//  36 , 10
	{ 0x0422, 0x0a26 },		//  38 , 10
	{ 0x0423, 0x0a28 },		//  40 , 10
	{ 0x0424, 0x0a2a },		//  42 , 10
	{ 0x0425, 0x0a2c },		//  44 , 10
	{ 0x0426, 0x0a2e },		//  46 , 10
	{ 0x0427, 0x0a30 },		//  48 , 10
	{ 0x0428, 0x0a32 },		//  50 , 10
	{ 0x0429, 0x0a34 },		//  52 , 10
	{ 0x042A, 0x0a36 },		//  54 , 10
	{ 0x042B, 0x0a38 },		//  56 , 10
	{ 0x042C, 0x0a3a },		//  58 , 10
	{ 0x042D, 0x0a3c },		//  60 , 10
	{ 0x042E, 0x0a3e },		//  62 , 10
	{ 0x042F, 0x0a40 },		//  64 , 10
	{ 0x0430, 0x0a01 },		//   1 , 10
	{ 0x0431, 0x0a03 },		//   3 , 10
	{ 0x0432, 0x0a05 },		//   5 , 10
	{ 0x0433, 0x0a07 },		//   7 , 10
	{ 0x0434, 0x0a09 },		//   9 , 10
	{ 0x0435, 0x0a0b },		//  11 , 10
	{ 0x0436, 0x0a0f },		//  15 , 10
	{ 0x0437, 0x0a11 },		//  17 , 10
	{ 0x0438, 0x0a13 },		//  19 , 10
	{ 0x0439, 0x0a15 },		//  21 , 10
	{ 0x043A, 0x0a17 },		//  23 , 10
	{ 0x043B, 0x0a19 },		//  25 , 10
	{ 0x043C, 0x0a1b },		//  27 , 10
	{ 0x043D, 0x0a1d },		//  29 , 10
	{ 0x043E, 0x0a1f },		//  31 , 10
	{ 0x043F, 0x0a21 },		//  33 , 10
	{ 0x0440, 0x0a23 },		//  35 , 10
	{ 0x0441, 0x0a25 },		//  37 , 10
	{ 0x0442, 0x0a27 },		//  39 , 10
	{ 0x0443, 0x0a29 },		//  41 , 10
	{ 0x0444, 0x0a2b },		//  43 , 10
	{ 0x0445, 0x0a2d },		//  45 , 10
	{ 0x0446, 0x0a2f },		//  47 , 10
	{ 0x0447, 0x0a31 },		//  49 , 10
	{ 0x0448, 0x0a33 },		//  51 , 10
	{ 0x0449, 0x0a35 },		//  53 , 10
	{ 0x044A, 0x0a37 },		//  55 , 10
	{ 0x044B, 0x0a39 },		//  57 , 10
	{ 0x044C, 0x0a3b },		//  59 , 10
	{ 0x044D, 0x0a3d },		//  61 , 10
	{ 0x044E, 0x0a3f },		//  63 , 10
	{ 0x044F, 0x0a41 },		//  65 , 10
	{ 0x0451, 0x0a0d },		//  13 , 10
	{ 0x0452, 0x0a4b },		//  75 , 10
	{ 0x0453, 0x0a45 },		//  69 , 10
	{ 0x0454, 0x0a4f },		//  79 , 10
	{ 0x0455, 0x0a53 },		//  83 , 10
	{ 0x0456, 0x0a59 },		//  89 , 10
	{ 0x0457, 0x0a5b },		//  91 , 10
	{ 0x0458, 0x0a5f },		//  95 , 10
	{ 0x0459, 0x0a69 },		//  105, 10
	{ 0x045A, 0x0a6d },		//  109, 10
	{ 0x045B, 0x0a73 },		//  115, 10
	{ 0x045C, 0x0a61 },		//  97 , 10
	{ 0x045E, 0x0a75 },		//  117, 10
	{ 0x045F, 0x0a87 },		//  135, 10
	{ 0x0460, 0x0a70 },		//  112, 10
	{ 0x0461, 0x0a71 },		//  113, 10
	{ 0x0462, 0x0a8e },		//  142, 10
	{ 0x0463, 0x0a8f },		//  143, 10
	{ 0x0466, 0x0a90 },		//  144, 10
	{ 0x0467, 0x0a91 },		//  145, 10
	{ 0x046A, 0x0a92 },		//  146, 10
	{ 0x046B, 0x0a93 },		//  147, 10
	{ 0x046E, 0x0a94 },		//  148, 10
	{ 0x046F, 0x0a95 },		//  149, 10
	{ 0x0470, 0x0a96 },		//  150, 10
	{ 0x0471, 0x0a97 },		//  151, 10
	{ 0x0472, 0x0a98 },		//  152, 10
	{ 0x0473, 0x0a99 },		//  153, 10
	{ 0x0474, 0x0a9a },		//  154, 10
	{ 0x0475, 0x0a9b },		//  155, 10
	{ 0x047A, 0x0a6e },		//  110, 10
	{ 0x047B, 0x0a6f },		//  111, 10
	{ 0x047E, 0x0a84 },		//  132, 10
	{ 0x047F, 0x0a85 },		//  133, 10
	{ 0x0490, 0x0a46 },		//  70 , 10
	{ 0x0491, 0x0a47 },		//  71 , 10
	{ 0x0492, 0x0a48 },		//  72 , 10
	{ 0x0493, 0x0a49 },		//  73 , 10
	{ 0x0496, 0x0a50 },		//  80 , 10
	{ 0x0497, 0x0a51 },		//  81 , 10
	{ 0x049A, 0x0a62 },		//  98 , 10
	{ 0x049B, 0x0a63 },		//  99 , 10
	{ 0x049C, 0x0a66 },		//  102, 10
	{ 0x049D, 0x0a67 },		//  103, 10
	{ 0x04A2, 0x0a6a },		//  106, 10
	{ 0x04A3, 0x0a6b },		//  107, 10
	{ 0x04AE, 0x0a78 },		//  120, 10
	{ 0x04AF, 0x0a79 },		//  121, 10
	{ 0x04B0, 0x0a7a },		//  122, 10
	{ 0x04B1, 0x0a7b },		//  123, 10
	{ 0x04B2, 0x0a7e },		//  126, 10
	{ 0x04B3, 0x0a7f },		//  127, 10
	{ 0x04B6, 0x0a88 },		//  136, 10
	{ 0x04B7, 0x0a89 },		//  137, 10
	{ 0x04B8, 0x0a8a },		//  138, 10
	{ 0x04B9, 0x0a8b },		//  139, 10
	{ 0x04BA, 0x0a82 },		//  130, 10
	{ 0x04BB, 0x0a83 },		//  131, 10
	{ 0x04D8, 0x0a42 },		//  66 , 10
	{ 0x04D9, 0x0a43 },		//  67 , 10
	{ 0x04EE, 0x0a76 },		//  118, 10
	{ 0x04EF, 0x0a77 },		//  119, 10
	{ 0x05B0, 0x0920 },		//  32 ,  9
	{ 0x05B1, 0x0921 },		//  33 ,  9
	{ 0x05B2, 0x0922 },		//  34 ,  9
	{ 0x05B3, 0x0923 },		//  35 ,  9
	{ 0x05B4, 0x0924 },		//  36 ,  9
	{ 0x05B5, 0x0925 },		//  37 ,  9
	{ 0x05B6, 0x0926 },		//  38 ,  9
	{ 0x05B7, 0x0927 },		//  39 ,  9
	{ 0x05B8, 0x0928 },		//  40 ,  9
	{ 0x05B9, 0x0929 },		//  41 ,  9
	{ 0x05BB, 0x092b },		//  43 ,  9
	{ 0x05BC, 0x092c },		//  44 ,  9
	{ 0x05BD, 0x092d },		//  45 ,  9
	{ 0x05BF, 0x092e },		//  46 ,  9
	{ 0x05C0, 0x091c },		//  28 ,  9
	{ 0x05C3, 0x091d },		//  29 ,  9
	{ 0x05D0, 0x0900 },		//   0 ,  9
	{ 0x05D1, 0x0901 },		//   1 ,  9
	{ 0x05D2, 0x0902 },		//   2 ,  9
	{ 0x05D3, 0x0903 },		//   3 ,  9
	{ 0x05D4, 0x0904 },		//   4 ,  9
	{ 0x05D5, 0x0905 },		//   5 ,  9
	{ 0x05D6, 0x0906 },		//   6 ,  9
	{ 0x05D7, 0x0907 },		//   7 ,  9
	{ 0x05D8, 0x0908 },		//   8 ,  9
	{ 0x05D9, 0x0909 },		//   9 ,  9
	{ 0x05DA, 0x090a },		//  10 ,  9
	{ 0x05DB, 0x090b },		//  11 ,  9
	{ 0x05DC, 0x090c },		//  12 ,  9
	{ 0x05DD, 0x090d },		//  13 ,  9
	{ 0x05DE, 0x090e },		//  14 ,  9
	{ 0x05DF, 0x090f },		//  15 ,  9
	{ 0x05E0, 0x0910 },		//  16 ,  9
	{ 0x05E1, 0x0911 },		//  17 ,  9
	{ 0x05E2, 0x0912 },		//  18 ,  9
	{ 0x05E3, 0x0913 },		//  19 ,  9
	{ 0x05E4, 0x0914 },		//  20 ,  9
	{ 0x05E5, 0x0915 },		//  21 ,  9
	{ 0x05E6, 0x0916 },		//  22 ,  9
	{ 0x05E7, 0x0917 },		//  23 ,  9
	{ 0x05E8, 0x0918 },		//  24 ,  9
	{ 0x05E9, 0x0919 },		//  25 ,  9
	{ 0x05EA, 0x091a },		//  26 ,  9
	{ 0x05F0, 0x0931 },		//  49 ,  9
	{ 0x05F1, 0x0932 },		//  50 ,  9
	{ 0x05F2, 0x0933 },		//  51 ,  9
	{ 0x05F3, 0x091e },		//  30 ,  9
	{ 0x05F4, 0x091f },		//  31 ,  9
	{ 0x060C, 0x0d26 },		//  38 , 13
	{ 0x061B, 0x0d27 },		//  39 , 13
	{ 0x061F, 0x0d28 },		//  40 , 13
	{ 0x0621, 0x0da4 },		//  164, 13
	{ 0x0622, 0x0db1 },		//  177, 13
	{ 0x0623, 0x0da5 },		//  165, 13
	{ 0x0624, 0x0da9 },		//  169, 13
	{ 0x0625, 0x0da7 },		//  167, 13
	{ 0x0626, 0x0dab },		//  171, 13
	{ 0x0627, 0x0d3a },		//  58 , 13
	{ 0x0628, 0x0d3c },		//  60 , 13
	{ 0x0629, 0x0d98 },		//  152, 13
	{ 0x062A, 0x0d40 },		//  64 , 13
	{ 0x062B, 0x0d44 },		//  68 , 13
	{ 0x062C, 0x0d48 },		//  72 , 13
	{ 0x062D, 0x0d4c },		//  76 , 13
	{ 0x062E, 0x0d50 },		//  80 , 13
	{ 0x062F, 0x0d54 },		//  84 , 13
	{ 0x0630, 0x0d56 },		//  86 , 13
	{ 0x0631, 0x0d58 },		//  88 , 13
	{ 0x0632, 0x0d5a },		//  90 , 13
	{ 0x0633, 0x0d5c },		//  92 , 13
	{ 0x0634, 0x0d60 },		//  96 , 13
	{ 0x0635, 0x0d64 },		//  100, 13
	{ 0x0636, 0x0d68 },		//  104, 13
	{ 0x0637, 0x0d6c },		//  108, 13
	{ 0x0638, 0x0d70 },		//  112, 13
	{ 0x0639, 0x0d74 },		//  116, 13
	{ 0x063A, 0x0d78 },		//  120, 13
	{ 0x0640, 0x0dc2 },		//  194, 13
	{ 0x0641, 0x0d7c },		//  124, 13
	{ 0x0642, 0x0d80 },		//  128, 13
	{ 0x0643, 0x0d84 },		//  132, 13
	{ 0x0644, 0x0d88 },		//  136, 13
	{ 0x0645, 0x0d8c },		//  140, 13
	{ 0x0646, 0x0d90 },		//  144, 13
	{ 0x0647, 0x0d94 },		//  148, 13
	{ 0x0648, 0x0d9a },		//  154, 13
	{ 0x0649, 0x0da0 },		//  160, 13
	{ 0x064A, 0x0d9c },		//  156, 13
	{ 0x064B, 0x0d10 },		//  16 , 13
	{ 0x064C, 0x0d11 },		//  17 , 13
	{ 0x064E, 0x0d0a },		//  10 , 13
	{ 0x064F, 0x0d0c },		//  12 , 13
	{ 0x0650, 0x0d0e },		//  14 , 13
	{ 0x0651, 0x0d16 },		//  22 , 13
	{ 0x0652, 0x0d14 },		//  20 , 13
	{ 0x0660, 0x0d38 },		//  56 , 13
	{ 0x0661, 0x0d2f },		//  47 , 13
	{ 0x0662, 0x0d30 },		//  48 , 13
	{ 0x0663, 0x0d31 },		//  49 , 13
	{ 0x0664, 0x0d32 },		//  50 , 13
	{ 0x0665, 0x0d33 },		//  51 , 13
	{ 0x0666, 0x0d34 },		//  52 , 13
	{ 0x0667, 0x0d35 },		//  53 , 13
	{ 0x0668, 0x0d36 },		//  54 , 13
	{ 0x0669, 0x0d37 },		//  55 , 13
	{ 0x066A, 0x0d2a },		//  42 , 13
	{ 0x0671, 0x0db3 },		//  179, 13
	{ 0x0674, 0x0d24 },		//  36 , 13
	{ 0x0679, 0x0e3c },		//  60 , 14
	{ 0x067A, 0x0e4c },		//  76 , 14
	{ 0x067B, 0x0e30 },		//  48 , 14
	{ 0x067C, 0x0e40 },		//  64 , 14
	{ 0x067D, 0x0e48 },		//  72 , 14
	{ 0x067E, 0x0e38 },		//  56 , 14
	{ 0x067F, 0x0e44 },		//  68 , 14
	{ 0x0680, 0x0e34 },		//  52 , 14
	{ 0x0681, 0x0e64 },		//  100, 14
	{ 0x0683, 0x0e54 },		//  84 , 14
	{ 0x0684, 0x0e50 },		//  80 , 14
	{ 0x0685, 0x0e60 },		//  96 , 14
	{ 0x0686, 0x0e58 },		//  88 , 14
	{ 0x0687, 0x0e5c },		//  92 , 14
	{ 0x0688, 0x0e68 },		//  104, 14
	{ 0x0689, 0x0e6a },		//  106, 14
	{ 0x068A, 0x0e70 },		//  112, 14
	{ 0x068C, 0x0e6c },		//  108, 14
	{ 0x068D, 0x0e72 },		//  114, 14
	{ 0x068E, 0x0e6e },		//  110, 14
	{ 0x0691, 0x0e76 },		//  118, 14
	{ 0x0692, 0x0e7C },		//  124, 14
	{ 0x0693, 0x0e74 },		//  116, 14
	{ 0x0695, 0x0e7a },		//  122, 14
	{ 0x0696, 0x0e80 },		//  128, 14
	{ 0x0698, 0x0e7e },		//  126, 14
	{ 0x0699, 0x0e78 },		//  120, 14
	{ 0x069A, 0x0e84 },		//  132, 14
	{ 0x06A0, 0x0e88 },		//  136, 14
	{ 0x06A4, 0x0e8c },		//  140, 14
	{ 0x06A6, 0x0e90 },		//  144, 14
	{ 0x06A9, 0x0e94 },		//  148, 14
	{ 0x06AA, 0x0e9c },		//  156, 14
	{ 0x06AB, 0x0ea8 },		//  168, 14
	{ 0x06AF, 0x0ea0 },		//  160, 14
	{ 0x06B1, 0x0eac },		//  172, 14
	{ 0x06B3, 0x0eb0 },		//  176, 14
	{ 0x06B5, 0x0eb4 },		//  180, 14
	{ 0x06BA, 0x0eba },		//  186, 14
	{ 0x06BB, 0x0ec2 },		//  194, 14
	{ 0x06BC, 0x0ebe },		//  190, 14
	{ 0x06C0, 0x0eda },		//  218, 14
	{ 0x06C6, 0x0ec6 },		//  198, 14
	{ 0x06CA, 0x0ec8 },		//  200, 14
	{ 0x06CE, 0x0ed0 },		//  208, 14
	{ 0x06D1, 0x0ed6 },		//  214, 14
	{ 0x06D2, 0x0ed4 },		//  212, 14
	{ 0x06D6, 0x0d25 },		//  37 , 13
	{ 0x06E4, 0x0d22 },		//  34 , 13
	{ 0x06F4, 0x0e29 },		//  41 , 14
	{ 0x06F5, 0x0e2b },		//  43 , 14
	{ 0x06F6, 0x0e2c },		//  44 , 14
	{ 0x06F7, 0x0e2e },		//  46 , 14
	{ 0x06F8, 0x0e2f },		//  47 , 14
	{ 0x10D0, 0x0ad2 },		//  210, 10
	{ 0x10D1, 0x0ad3 },		//  211, 10
	{ 0x10D2, 0x0ad4 },		//  212, 10
	{ 0x10D3, 0x0ad5 },		//  213, 10
	{ 0x10D4, 0x0ad6 },		//  214, 10
	{ 0x10D5, 0x0ad7 },		//  215, 10
	{ 0x10D6, 0x0ad8 },		//  216, 10
	{ 0x10D7, 0x0ada },		//  218, 10
	{ 0x10D8, 0x0adb },		//  219, 10
	{ 0x10D9, 0x0adc },		//  220, 10
	{ 0x10DA, 0x0add },		//  221, 10
	{ 0x10DB, 0x0ade },		//  222, 10
	{ 0x10DC, 0x0adf },		//  223, 10
	{ 0x10DD, 0x0ae1 },		//  225, 10
	{ 0x10DE, 0x0ae2 },		//  226, 10
	{ 0x10DF, 0x0ae3 },		//  227, 10
	{ 0x10E0, 0x0ae4 },		//  228, 10
	{ 0x10E1, 0x0ae5 },		//  229, 10
	{ 0x10E2, 0x0ae6 },		//  230, 10
	{ 0x10E3, 0x0ae7 },		//  231, 10
	{ 0x10E4, 0x0ae9 },		//  233, 10
	{ 0x10E5, 0x0aea },		//  234, 10
	{ 0x10E6, 0x0aeb },		//  235, 10
	{ 0x10E7, 0x0aec },		//  236, 10
	{ 0x10E8, 0x0aed },		//  237, 10
	{ 0x10E9, 0x0aee },		//  238, 10
	{ 0x10EA, 0x0aef },		//  239, 10
	{ 0x10EB, 0x0af0 },		//  240, 10
	{ 0x10EC, 0x0af1 },		//  241, 10
	{ 0x10ED, 0x0af2 },		//  242, 10
	{ 0x10EE, 0x0af3 },		//  243, 10
	{ 0x10EF, 0x0af5 },		//  245, 10
	{ 0x10F0, 0x0af6 },		//  246, 10
	{ 0x10F1, 0x0ad9 },		//  217, 10
	{ 0x10F2, 0x0ae0 },		//  224, 10
	{ 0x10F3, 0x0ae8 },		//  232, 10
	{ 0x10F4, 0x0af4 },		//  244, 10
	{ 0x10F5, 0x0af7 },		//  247, 10
	{ 0x10F6, 0x0af8 },		//  248, 10
	{ 0x1F00, 0x0873 },		//  115,  8
	{ 0x1F01, 0x087b },		//  123,  8
	{ 0x1F02, 0x0875 },		//  117,  8
	{ 0x1F03, 0x087d },		//  125,  8
	{ 0x1F04, 0x0874 },		//  116,  8
	{ 0x1F05, 0x087c },		//  124,  8
	{ 0x1F10, 0x0884 },		//  132,  8
	{ 0x1F11, 0x0887 },		//  135,  8
	{ 0x1F12, 0x0886 },		//  134,  8
	{ 0x1F13, 0x0889 },		//  137,  8
	{ 0x1F14, 0x0885 },		//  133,  8
	{ 0x1F15, 0x0888 },		//  136,  8
	{ 0x1F20, 0x0890 },		//  144,  8
	{ 0x1F21, 0x0898 },		//  152,  8
	{ 0x1F22, 0x0892 },		//  146,  8
	{ 0x1F23, 0x089a },		//  154,  8
	{ 0x1F24, 0x0891 },		//  145,  8
	{ 0x1F25, 0x0899 },		//  153,  8
	{ 0x1F30, 0x08a4 },		//  164,  8
	{ 0x1F31, 0x08a8 },		//  168,  8
	{ 0x1F32, 0x08a6 },		//  166,  8
	{ 0x1F33, 0x08aa },		//  170,  8
	{ 0x1F34, 0x08a5 },		//  165,  8
	{ 0x1F35, 0x08a9 },		//  169,  8
	{ 0x1F40, 0x08ad },		//  173,  8
	{ 0x1F41, 0x08b0 },		//  176,  8
	{ 0x1F42, 0x08af },		//  175,  8
	{ 0x1F43, 0x08b2 },		//  178,  8
	{ 0x1F44, 0x08ae },		//  174,  8
	{ 0x1F45, 0x08b1 },		//  177,  8
	{ 0x1F50, 0x08b9 },		//  185,  8
	{ 0x1F51, 0x08bd },		//  189,  8
	{ 0x1F52, 0x08bb },		//  187,  8
	{ 0x1F53, 0x08bf },		//  191,  8
	{ 0x1F54, 0x08ba },		//  186,  8
	{ 0x1F55, 0x08be },		//  190,  8
	{ 0x1F60, 0x08c7 },		//  199,  8
	{ 0x1F61, 0x08cf },		//  207,  8
	{ 0x1F62, 0x08c9 },		//  201,  8
	{ 0x1F63, 0x08d1 },		//  209,  8
	{ 0x1F64, 0x08c8 },		//  200,  8
	{ 0x1F65, 0x08d0 },		//  208,  8
	{ 0x1F70, 0x086d },		//  109,  8
	{ 0x1F72, 0x0883 },		//  131,  8
	{ 0x1F74, 0x088a },		//  138,  8
	{ 0x1F76, 0x08a0 },		//  160,  8
	{ 0x1F78, 0x08ac },		//  172,  8
	{ 0x1F7A, 0x08b5 },		//  181,  8
	{ 0x1F7C, 0x08c1 },		//  193,  8
	{ 0x1F80, 0x0877 },		//  119,  8
	{ 0x1F81, 0x087f },		//  127,  8
	{ 0x1F82, 0x0879 },		//  121,  8
	{ 0x1F83, 0x0881 },		//  129,  8
	{ 0x1F84, 0x0878 },		//  120,  8
	{ 0x1F85, 0x0880 },		//  128,  8
	{ 0x1F90, 0x0894 },		//  148,  8
	{ 0x1F91, 0x089c },		//  156,  8
	{ 0x1F92, 0x0896 },		//  150,  8
	{ 0x1F93, 0x089e },		//  158,  8
	{ 0x1F94, 0x0895 },		//  149,  8
	{ 0x1F95, 0x089d },		//  157,  8
	{ 0x1FA0, 0x08cb },		//  203,  8
	{ 0x1FA1, 0x08d3 },		//  211,  8
	{ 0x1FA2, 0x08cd },		//  205,  8
	{ 0x1FA3, 0x08d5 },		//  213,  8
	{ 0x1FA4, 0x08cc },		//  204,  8
	{ 0x1FA5, 0x08d4 },		//  212,  8
	{ 0x1FB2, 0x0871 },		//  113,  8
	{ 0x1FB3, 0x086f },		//  111,  8
	{ 0x1FB4, 0x0870 },		//  112,  8
	{ 0x1FC2, 0x088e },		//  142,  8
	{ 0x1FC3, 0x088c },		//  140,  8
	{ 0x1FC4, 0x088d },		//  141,  8
	{ 0x1FCD, 0x085e },		//  94 ,  8
	{ 0x1FCE, 0x085c },		//  92 ,  8
	{ 0x1FDD, 0x085f },		//  95 ,  8
	{ 0x1FDE, 0x085d },		//  93 ,  8
	{ 0x1FE4, 0x08B4 },		//  180,  8
	{ 0x1FE5, 0x08B3 },		//  179,  8
	{ 0x1FF2, 0x08c5 },		//  197,  8
	{ 0x1FF3, 0x08c3 },		//  195,  8
	{ 0x1FF4, 0x08c4 },		//  196,  8
	{ 0x2007, 0x0517 },		//  23 ,  5
	{ 0x2012, 0x0432 },		//  50 ,  4
	{ 0x2013, 0x0421 },		//  33 ,  4
	{ 0x2014, 0x0422 },		//  34 ,  4
	{ 0x2017, 0x022f },		//  47 ,  2
	{ 0x2018, 0x041d },		//  29 ,  4
	{ 0x2019, 0x041c },		//  28 ,  4
	{ 0x201A, 0x043e },		//  62 ,  4
	{ 0x201B, 0x041b },		//  27 ,  4
	{ 0x201C, 0x0420 },		//  32 ,  4
	{ 0x201D, 0x041f },		//  31 ,  4
	{ 0x201E, 0x043f },		//  63 ,  4
	{ 0x201F, 0x041e },		//  30 ,  4
	{ 0x2020, 0x0427 },		//  39 ,  4
	{ 0x2021, 0x0428 },		//  40 ,  4
	{ 0x2022, 0x0403 },		//   3 ,  4
	{ 0x2026, 0x0438 },		//  56 ,  4
	{ 0x2030, 0x044b },		//  75 ,  4
	{ 0x2033, 0x0580 },		//  128,  5
	{ 0x2034, 0x0671 },		//  113,  6
	{ 0x2036, 0x057f },		//  127,  5
	{ 0x2039, 0x0423 },		//  35 ,  4
	{ 0x203A, 0x0424 },		//  36 ,  4
	{ 0x203C, 0x050d },		//  13 ,  5
	{ 0x203E, 0x0626 },		//  38 ,  6
	{ 0x207F, 0x0415 },		//  21 ,  4
	{ 0x20A0, 0x043c },		//  60 ,  4
	{ 0x20A2, 0x043b },		//  59 ,  4
	{ 0x20A3, 0x043a },		//  58 ,  4
	{ 0x20A4, 0x043d },		//  61 ,  4
	{ 0x20A6, 0x0457 },		//  87 ,  4
	{ 0x20A7, 0x040d },		//  13 ,  4
	{ 0x20A8, 0x0458 },		//  88 ,  4
	{ 0x20A9, 0x0456 },		//  86 ,  4
	{ 0x20AA, 0x097A },		//  122,  9
	{ 0x20AC, 0x0466 },	   //  102,  4, Euro Sign - GW assigned x448 [4,72]
	{ 0x20DD, 0x066d },		//  109,  6
	{ 0x20E1, 0x06e1 },		//  225,  6
	{ 0x2102, 0x06d5 },		//  213,  6
	{ 0x2104, 0x0515 },		//  21 ,  5
	{ 0x2105, 0x0449 },		//  73 ,  4
	{ 0x2106, 0x044a },		//  74 ,  4
	{ 0x210C, 0x06e9 },		//  233,  6
	{ 0x210F, 0x0632 },		//  50 ,  6
	{ 0x2111, 0x0633 },		//  51 ,  6
	{ 0x2112, 0x0669 },		//  105,  6
	{ 0x2113, 0x0631 },		//  49 ,  6
	{ 0x2115, 0x06d7 },		//  215,  6
	{ 0x2116, 0x044c },		//  76 ,  4
	{ 0x2118, 0x0635 },		//  53 ,  6
	{ 0x211C, 0x0634 },		//  52 ,  6
	{ 0x211D, 0x06d8 },		//  216,  6
	{ 0x211E, 0x042b },		//  43 ,  4
	{ 0x2120, 0x042a },		//  42 ,  4
	{ 0x2122, 0x0429 },		//  41 ,  4
	{ 0x2127, 0x06a7 },		//  167,  6
	{ 0x2128, 0x066b },		//  107,  6
	{ 0x212B, 0x0623 },		//  35 ,  6
	{ 0x212D, 0x066a },		//  106,  6
	{ 0x212F, 0x0630 },		//  48 ,  6
	{ 0x2130, 0x06d3 },		//  211,  6
	{ 0x2131, 0x06d4 },		//  212,  6
	{ 0x2153, 0x0440 },		//  64 ,  4
	{ 0x2154, 0x0441 },		//  65 ,  4
	{ 0x215B, 0x0442 },		//  66 ,  4
	{ 0x215C, 0x0443 },		//  67 ,  4
	{ 0x215D, 0x0444 },		//  68 ,  4
	{ 0x215E, 0x0445 },		//  69 ,  4
	{ 0x2190, 0x0590 },		//  144,  5
	{ 0x2191, 0x0617 },		//  23 ,  6
	{ 0x2192, 0x05d5 },		//  213,  5
	{ 0x2193, 0x0618 },		//  24 ,  6
	{ 0x2194, 0x05d6 },		//  214,  5
	{ 0x2195, 0x05d7 },		//  215,  5
	{ 0x2196, 0x0640 },		//  64 ,  6
	{ 0x2197, 0x063e },		//  62 ,  6
	{ 0x2198, 0x063f },		//  63 ,  6
	{ 0x2199, 0x0641 },		//  65 ,  6
	{ 0x219D, 0x0690 },		//  144,  6
	{ 0x21A3, 0x0693 },		//  147,  6
	{ 0x21A8, 0x050f },		//  15 ,  5
	{ 0x21A9, 0x0691 },		//  145,  6
	{ 0x21AA, 0x0692 },		//  146,  6
	{ 0x21B5, 0x0514 },		//  20 ,  5
	{ 0x21BC, 0x0694 },		//  148,  6
	{ 0x21BD, 0x0695 },		//  149,  6
	{ 0x21BE, 0x069b },		//  155,  6
	{ 0x21BF, 0x069a },		//  154,  6
	{ 0x21C0, 0x0696 },		//  150,  6
	{ 0x21C1, 0x0697 },		//  151,  6
	{ 0x21C2, 0x069d },		//  157,  6
	{ 0x21C3, 0x069c },		//  156,  6
	{ 0x21C4, 0x0636 },		//  54 ,  6
	{ 0x21C6, 0x0637 },		//  55 ,  6
	{ 0x21C7, 0x069f },		//  159,  6
	{ 0x21C9, 0x069e },		//  158,  6
	{ 0x21CB, 0x0699 },		//  153,  6
	{ 0x21CC, 0x0698 },		//  152,  6
	{ 0x21D0, 0x0639 },		//  57 ,  6
	{ 0x21D1, 0x063a },		//  58 ,  6
	{ 0x21D2, 0x0638 },		//  56 ,  6
	{ 0x21D3, 0x063b },		//  59 ,  6
	{ 0x21D4, 0x063c },		//  60 ,  6
	{ 0x21D5, 0x063d },		//  61 ,  6
	{ 0x21E6, 0x0597 },		//  151,  5
	{ 0x21E8, 0x0596 },		//  150,  5
	{ 0x2200, 0x067a },		//  122,  6
	{ 0x2202, 0x062c },		//  44 ,  6
	{ 0x2203, 0x0679 },		//  121,  6
	{ 0x2204, 0x06d0 },		//  208,  6
	{ 0x2205, 0x0648 },		//  72 ,  6
	{ 0x2207, 0x062b },		//  43 ,  6
	{ 0x2208, 0x060f },		//  15 ,  6
	{ 0x2209, 0x06d1 },		//  209,  6
	{ 0x220B, 0x06db },		//  219,  6
	{ 0x220D, 0x0647 },		//  71 ,  6
	{ 0x220F, 0x0629 },		//  41 ,  6
	{ 0x2210, 0x0672 },		//  114,  6
	{ 0x2211, 0x0612 },		//  18 ,  6
	{ 0x2212, 0x0600 },		//   0 ,  6
	{ 0x2213, 0x062a },		//  42 ,  6
	{ 0x2214, 0x06ae },		//  174,  6
	{ 0x2215, 0x0606 },		//   6 ,  6
	{ 0x2216, 0x0607 },		//   7 ,  6
	{ 0x2218, 0x0621 },		//  33 ,  6
	{ 0x2219, 0x0622 },		//  34 ,  6
	{ 0x221A, 0x0704 },		//   4 ,  7
	{ 0x221D, 0x0604 },		//   4 ,  6
	{ 0x221E, 0x0613 },		//  19 ,  6
	{ 0x221F, 0x06da },		//  218,  6
	{ 0x2220, 0x064f },		//  79 ,  6
	{ 0x2221, 0x06a8 },		//  168,  6
	{ 0x2222, 0x06a9 },		//  169,  6
	{ 0x2223, 0x0609 },		//   9 ,  6
	{ 0x2224, 0x06ce },		//  206,  6
	{ 0x2225, 0x0611 },		//  17 ,  6
	{ 0x2226, 0x06cd },		//  205,  6
	{ 0x2227, 0x0655 },		//  85 ,  6
	{ 0x2228, 0x0656 },		//  86 ,  6
	{ 0x2229, 0x0610 },		//  16 ,  6
	{ 0x222A, 0x0642 },		//  66 ,  6
	{ 0x222B, 0x0628 },		//  40 ,  6
	{ 0x222E, 0x0668 },		//  104,  6
	{ 0x2234, 0x0666 },		//  102,  6
	{ 0x2235, 0x0665 },		//  101,  6
	{ 0x2237, 0x0667 },		//  103,  6
	{ 0x223C, 0x060c },		//  12 ,  6
	{ 0x2241, 0x06bd },		//  189,  6
	{ 0x2243, 0x0673 },		//  115,  6
	{ 0x2244, 0x06be },		//  190,  6
	{ 0x2245, 0x0674 },		//  116,  6
	{ 0x2247, 0x06bf },		//  191,  6
	{ 0x2248, 0x060d },		//  13 ,  6
	{ 0x2249, 0x06c0 },		//  192,  6
	{ 0x224D, 0x06b3 },		//  179,  6
	{ 0x224E, 0x06b2 },		//  178,  6
	{ 0x2250, 0x06af },		//  175,  6
	{ 0x2252, 0x06b0 },		//  176,  6
	{ 0x2253, 0x06b1 },		//  177,  6
	{ 0x225F, 0x06d9 },		//  217,  6
	{ 0x2260, 0x0663 },		//  99 ,  6
	{ 0x2261, 0x060e },		//  14 ,  6
	{ 0x2262, 0x0664 },		//  100,  6
	{ 0x2264, 0x0602 },		//   2 ,  6
	{ 0x2265, 0x0603 },		//   3 ,  6
	{ 0x226A, 0x064d },		//  77 ,  6
	{ 0x226B, 0x064e },		//  78 ,  6
	{ 0x226C, 0x06b6 },		//  182,  6
	{ 0x226D, 0x06cf },		//  207,  6
	{ 0x226E, 0x06b9 },		//  185,  6
	{ 0x226F, 0x06bb },		//  187,  6
	{ 0x2270, 0x06ba },		//  186,  6
	{ 0x2271, 0x06bc },		//  188,  6
	{ 0x2272, 0x06eb },		//  235,  6
	{ 0x2273, 0x06ec },		//  236,  6
	{ 0x227A, 0x0675 },		//  117,  6
	{ 0x227B, 0x0677 },		//  119,  6
	{ 0x227C, 0x0676 },		//  118,  6
	{ 0x227D, 0x0678 },		//  120,  6
	{ 0x2280, 0x06c1 },		//  193,  6
	{ 0x2281, 0x06c3 },		//  195,  6
	{ 0x2282, 0x0643 },		//  67 ,  6
	{ 0x2283, 0x0644 },		//  68 ,  6
	{ 0x2284, 0x06c5 },		//  197,  6
	{ 0x2285, 0x06c6 },		//  198,  6
	{ 0x2286, 0x0645 },		//  69 ,  6
	{ 0x2287, 0x0646 },		//  70 ,  6
	{ 0x2288, 0x06c7 },		//  199,  6
	{ 0x2289, 0x06c8 },		//  200,  6
	{ 0x228A, 0x067e },		//  126,  6
	{ 0x228B, 0x067f },		//  127,  6
	{ 0x228E, 0x067d },		//  125,  6
	{ 0x228F, 0x0682 },		//  130,  6
	{ 0x2290, 0x0685 },		//  133,  6
	{ 0x2291, 0x0683 },		//  131,  6
	{ 0x2292, 0x0686 },		//  134,  6
	{ 0x2293, 0x0680 },		//  128,  6
	{ 0x2294, 0x0681 },		//  129,  6
	{ 0x2295, 0x0651 },		//  81 ,  6
	{ 0x2296, 0x0652 },		//  82 ,  6
	{ 0x2297, 0x0650 },		//  80 ,  6
	{ 0x2299, 0x0654 },		//  84 ,  6
	{ 0x229A, 0x06a4 },		//  164,  6
	{ 0x229B, 0x06a5 },		//  165,  6
	{ 0x229D, 0x06a6 },		//  166,  6
	{ 0x22A2, 0x065b },		//  91 ,  6
	{ 0x22A3, 0x065c },		//  92 ,  6
	{ 0x22A4, 0x0658 },		//  88 ,  6
	{ 0x22A5, 0x0659 },		//  89 ,  6
	{ 0x22A8, 0x06b4 },		//  180,  6
	{ 0x22BB, 0x0657 },		//  87 ,  6
	{ 0x22C5, 0x061f },		//  31 ,  6
	{ 0x22C6, 0x0670 },		//  112,  6
	{ 0x22C8, 0x068c },		//  140,  6
	{ 0x22D0, 0x06a2 },		//  162,  6
	{ 0x22D1, 0x06a3 },		//  163,  6
	{ 0x22D2, 0x06a1 },		//  161,  6
	{ 0x22D3, 0x06a0 },		//  160,  6
	{ 0x22D8, 0x067b },		//  123,  6
	{ 0x22D9, 0x067c },		//  124,  6
	{ 0x22E0, 0x06c2 },		//  194,  6
	{ 0x22E1, 0x06c4 },		//  196,  6
	{ 0x22E2, 0x06cb },		//  203,  6
	{ 0x22E3, 0x06cc },		//  204,  6
	{ 0x22E4, 0x0684 },		//  132,  6
	{ 0x22E5, 0x0687 },		//  135,  6
	{ 0x22EE, 0x06de },		//  222,  6
	{ 0x22EF, 0x06dc },		//  220,  6
	{ 0x22F1, 0x06df },		//  223,  6
	{ 0x2302, 0x050c },		//  12 ,  5
	{ 0x2308, 0x0649 },		//  73 ,  6
	{ 0x2309, 0x064a },		//  74 ,  6
	{ 0x230A, 0x064b },		//  75 ,  6
	{ 0x230B, 0x064c },		//  76 ,  6
	{ 0x2310, 0x0510 },		//  16 ,  5
	{ 0x2312, 0x065a },		//  90 ,  6
	{ 0x2319, 0x0511 },		//  17 ,  5
	{ 0x231A, 0x051f },		//  31 ,  5
	{ 0x231B, 0x0520 },		//  32 ,  5
	{ 0x2320, 0x0700 },		//   0 ,  7
	{ 0x2321, 0x0701 },		//   1 ,  7
	{ 0x2322, 0x068e },		//  142,  6
	{ 0x2323, 0x068d },		//  141,  6
	{ 0x2329, 0x060a },		//  10 ,  6
	{ 0x232A, 0x060b },		//  11 ,  6
	{ 0x2409, 0x044f },		//  79 ,  4
	{ 0x240A, 0x0452 },		//  82 ,  4
	{ 0x240B, 0x0454 },		//  84 ,  4
	{ 0x240C, 0x0450 },		//  80 ,  4
	{ 0x240D, 0x0451 },		//  81 ,  4
	{ 0x2424, 0x0453 },		//  83 ,  4
	{ 0x24C2, 0x0446 },		//  70 ,  4
	{ 0x24C5, 0x0447 },		//  71 ,  4
	{ 0x24CA, 0x0448 },		//  72 ,  4,  - circled U
	{ 0x2500, 0x0308 },		//   8 ,  3
	{ 0x2502, 0x0309 },		//   9 ,  3
	{ 0x250C, 0x030a },		//  10 ,  3
	{ 0x2510, 0x030b },		//  11 ,  3
	{ 0x2514, 0x030d },		//  13 ,  3
	{ 0x2518, 0x030c },		//  12 ,  3
	{ 0x251C, 0x030e },		//  14 ,  3
	{ 0x251E, 0x033e },		//  62 ,  3
	{ 0x251F, 0x033c },		//  60 ,  3
	{ 0x2521, 0x033f },		//  63 ,  3
	{ 0x2522, 0x033d },		//  61 ,  3
	{ 0x2524, 0x0310 },		//  16 ,  3
	{ 0x2526, 0x0345 },		//  69 ,  3
	{ 0x2527, 0x0344 },		//  68 ,  3
	{ 0x2529, 0x0347 },		//  71 ,  3
	{ 0x252A, 0x0346 },		//  70 ,  3
	{ 0x252C, 0x030f },		//  15 ,  3
	{ 0x252D, 0x0342 },		//  66 ,  3
	{ 0x252E, 0x0340 },		//  64 ,  3
	{ 0x2531, 0x0343 },		//  67 ,  3
	{ 0x2532, 0x0341 },		//  65 ,  3
	{ 0x2534, 0x0311 },		//  17 ,  3
	{ 0x2535, 0x034a },		//  74 ,  3
	{ 0x2536, 0x0348 },		//  72 ,  3
	{ 0x2539, 0x034b },		//  75 ,  3
	{ 0x253A, 0x0349 },		//  73 ,  3
	{ 0x253C, 0x0312 },		//  18 ,  3
	{ 0x253D, 0x0352 },		//  82 ,  3
	{ 0x253E, 0x034e },		//  78 ,  3
	{ 0x2540, 0x034f },		//  79 ,  3
	{ 0x2541, 0x034c },		//  76 ,  3
	{ 0x2543, 0x0355 },		//  85 ,  3
	{ 0x2544, 0x0350 },		//  80 ,  3
	{ 0x2545, 0x0353 },		//  83 ,  3
	{ 0x2546, 0x034d },		//  77 ,  3
	{ 0x2547, 0x0357 },		//  87 ,  3
	{ 0x2548, 0x0354 },		//  84 ,  3
	{ 0x2549, 0x0356 },		//  86 ,  3
	{ 0x254A, 0x0351 },		//  81 ,  3
	{ 0x2550, 0x0313 },		//  19 ,  3
	{ 0x2551, 0x0314 },		//  20 ,  3
	{ 0x2552, 0x031e },		//  30 ,  3
	{ 0x2553, 0x0322 },		//  34 ,  3
	{ 0x2554, 0x0315 },		//  21 ,  3
	{ 0x2555, 0x031f },		//  31 ,  3
	{ 0x2556, 0x0323 },		//  35 ,  3
	{ 0x2557, 0x0316 },		//  22 ,  3
	{ 0x2558, 0x0321 },		//  33 ,  3
	{ 0x2559, 0x0325 },		//  37 ,  3
	{ 0x255A, 0x0318 },		//  24 ,  3
	{ 0x255B, 0x0320 },		//  32 ,  3
	{ 0x255C, 0x0324 },		//  36 ,  3
	{ 0x255D, 0x0317 },		//  23 ,  3
	{ 0x255E, 0x0326 },		//  38 ,  3
	{ 0x255F, 0x032a },		//  42 ,  3
	{ 0x2560, 0x0319 },		//  25 ,  3
	{ 0x2561, 0x0328 },		//  40 ,  3
	{ 0x2562, 0x032c },		//  44 ,  3
	{ 0x2563, 0x031b },		//  27 ,  3
	{ 0x2564, 0x032b },		//  43 ,  3
	{ 0x2565, 0x0327 },		//  39 ,  3
	{ 0x2566, 0x031a },		//  26 ,  3
	{ 0x2567, 0x032d },		//  45 ,  3
	{ 0x2568, 0x0329 },		//  41 ,  3
	{ 0x2569, 0x031c },		//  28 ,  3
	{ 0x256A, 0x032f },		//  47 ,  3
	{ 0x256B, 0x032e },		//  46 ,  3
	{ 0x256C, 0x031d },		//  29 ,  3
	{ 0x2574, 0x0330 },		//  48 ,  3
	{ 0x2575, 0x0331 },		//  49 ,  3
	{ 0x2576, 0x0332 },		//  50 ,  3
	{ 0x2577, 0x0333 },		//  51 ,  3
	{ 0x2578, 0x0334 },		//  52 ,  3
	{ 0x2579, 0x0335 },		//  53 ,  3
	{ 0x257A, 0x0336 },		//  54 ,  3
	{ 0x257B, 0x0337 },		//  55 ,  3
	{ 0x257C, 0x0338 },		//  56 ,  3
	{ 0x257D, 0x033a },		//  58 ,  3
	{ 0x257E, 0x0339 },		//  57 ,  3
	{ 0x257F, 0x033b },		//  59 ,  3
	{ 0x2580, 0x0305 },		//   5 ,  3
	{ 0x2584, 0x0307 },		//   7 ,  3
	{ 0x2588, 0x0303 },		//   3 ,  3
	{ 0x258C, 0x0304 },		//   4 ,  3
	{ 0x2590, 0x0306 },		//   6 ,  3
	{ 0x2591, 0x0300 },		//   0 ,  3
	{ 0x2592, 0x0301 },		//   1 ,  3
	{ 0x2593, 0x0302 },		//   2 ,  3
	{ 0x25A0, 0x0402 },		//   2 ,  4
	{ 0x25A1, 0x0426 },		//  38 ,  4
	{ 0x25AA, 0x042f },		//  47 ,  4
	{ 0x25AB, 0x0431 },		//  49 ,  4
	{ 0x25AC, 0x050b },		//  11 ,  5
	{ 0x25B2, 0x0573 },		//  115,  5
	{ 0x25B3, 0x0688 },		//  136,  6
	{ 0x25B4, 0x061d },		//  29 ,  6
	{ 0x25B5, 0x06ac },		//  172,  6
	{ 0x25B8, 0x061b },		//  27 ,  6
	{ 0x25B9, 0x068b },		//  139,  6
	{ 0x25BC, 0x0574 },		//  116,  5
	{ 0x25BD, 0x0689 },		//  137,  6
	{ 0x25BE, 0x061e },		//  30 ,  6
	{ 0x25BF, 0x06ad },		//  173,  6
	{ 0x25C2, 0x061c },		//  28 ,  6
	{ 0x25C3, 0x068a },		//  138,  6
	{ 0x25C6, 0x0575 },		//  117,  5
	{ 0x25C7, 0x066f },		//  111,  6
	{ 0x25CA, 0x065f },		//  95 ,  6
	{ 0x25CB, 0x0401 },		//   1 ,  4
	{ 0x25CF, 0x0400 },		//   0 ,  4
	{ 0x25D6, 0x059e },		//  158,  5
	{ 0x25D7, 0x0577 },		//  119,  5
	{ 0x25D8, 0x0512 },		//  18 ,  5
	{ 0x25D9, 0x0513 },		//  19 ,  5
	{ 0x25E6, 0x042d },		//  45 ,  4
	{ 0x2605, 0x0548 },		//  72,   5
	{ 0x260E, 0x051e },		//  30 ,  5
	{ 0x2610, 0x0518 },		//  24 ,  5
	{ 0x2612, 0x0519 },		//  25 ,  5
	{ 0x261B, 0x052a },		//  42 ,  5
	{ 0x261C, 0x0516 },		//  22 ,  5
	{ 0x261E, 0x052b },		//  43 ,  5
	{ 0x2639, 0x051a },		//  26 ,  5
	{ 0x263A, 0x0507 },		//   7 ,  5
	{ 0x263B, 0x0508 },		//   8 ,  5
	{ 0x263C, 0x0506 },		//   6 ,  5
	{ 0x2640, 0x0505 },		//   5 ,  5
	{ 0x2642, 0x0504 },		//   4 ,  5
	{ 0x2660, 0x05ab },		//  171,  5
	{ 0x2661, 0x0500 },		//   0 ,  5
	{ 0x2662, 0x0501 },		//   1 ,  5
	{ 0x2663, 0x05a8 },		//  168,  5
	{ 0x2664, 0x0503 },		//   3 ,  5
	{ 0x2665, 0x05aa },		//  170,  5
	{ 0x2666, 0x05a9 },		//  169,  5
	{ 0x2667, 0x0502 },		//   2 ,  5
	{ 0x266A, 0x0509 },		//   9 ,  5
	{ 0x266C, 0x050a },		//  10 ,  5
	{ 0x266D, 0x051c },		//  28 ,  5
	{ 0x266E, 0x051d },		//  29 ,  5
	{ 0x266F, 0x051b },		//  27 ,  5
	{ 0x2701, 0x0521 },		//  33 ,  5
	{ 0x2702, 0x0522 },		//  34 ,  5
	{ 0x2703, 0x0523 },		//  35 ,  5
	{ 0x2704, 0x0524 },		//  36 ,  5
	{ 0x2706, 0x0526 },		//  38 ,  5
	{ 0x2707, 0x0527 },		//  39 ,  5
	{ 0x2708, 0x0528 },		//  40 ,  5
	{ 0x2709, 0x0529 },		//  41 ,  5
	{ 0x270C, 0x052c },		//  44 ,  5
	{ 0x270D, 0x052d },		//  45 ,  5
	{ 0x270E, 0x052e },		//  46 ,  5
	{ 0x270F, 0x052f },		//  47 ,  5
	{ 0x2710, 0x0530 },		//  48 ,  5
	{ 0x2711, 0x0531 },		//  49 ,  5
	{ 0x2712, 0x0532 },		//  50 ,  5
	{ 0x2713, 0x0533 },		//  51 ,  5
	{ 0x2714, 0x0534 },		//  52 ,  5
	{ 0x2715, 0x0535 },		//  53 ,  5
	{ 0x2716, 0x0536 },		//  54 ,  5
	{ 0x2717, 0x0537 },		//  55 ,  5
	{ 0x2718, 0x0538 },		//  56 ,  5
	{ 0x2719, 0x0539 },		//  57 ,  5
	{ 0x271A, 0x053a },		//  58 ,  5
	{ 0x271B, 0x053b },		//  59 ,  5
	{ 0x271C, 0x053c },		//  60 ,  5
	{ 0x271D, 0x053d },		//  61 ,  5
	{ 0x271E, 0x053e },		//  62 ,  5
	{ 0x271F, 0x053f },		//  63 ,  5
	{ 0x2720, 0x0540 },		//  64 ,  5
	{ 0x2721, 0x0541 },		//  65 ,  5
	{ 0x2722, 0x0542 },		//  66 ,  5
	{ 0x2723, 0x0543 },		//  67 ,  5
	{ 0x2724, 0x0544 },		//  68 ,  5
	{ 0x2725, 0x0545 },		//  69 ,  5
	{ 0x2726, 0x0546 },		//  70 ,  5
	{ 0x2727, 0x0547 },		//  71 ,  5
	{ 0x2729, 0x0549 },		//  73 ,  5
	{ 0x272A, 0x054a },		//  74 ,  5
	{ 0x272B, 0x054b },		//  75 ,  5
	{ 0x272C, 0x054c },		//  76 ,  5
	{ 0x272D, 0x054d },		//  77 ,  5
	{ 0x272E, 0x054e },		//  78 ,  5
	{ 0x272F, 0x054f },		//  79 ,  5
	{ 0x2730, 0x0550 },		//  80 ,  5
	{ 0x2731, 0x0551 },		//  81 ,  5
	{ 0x2732, 0x0552 },		//  82 ,  5
	{ 0x2733, 0x0553 },		//  83 ,  5
	{ 0x2734, 0x0554 },		//  84 ,  5
	{ 0x2735, 0x0555 },		//  85 ,  5
	{ 0x2736, 0x0556 },		//  86 ,  5
	{ 0x2737, 0x0557 },		//  87 ,  5
	{ 0x2738, 0x0558 },		//  88 ,  5
	{ 0x2739, 0x0559 },		//  89 ,  5
	{ 0x273A, 0x055a },		//  90 ,  5
	{ 0x273B, 0x055b },		//  91 ,  5
	{ 0x273C, 0x055c },		//  92 ,  5
	{ 0x273D, 0x055d },		//  93 ,  5
	{ 0x273E, 0x055e },		//  94 ,  5
	{ 0x273F, 0x055f },		//  95 ,  5
	{ 0x2740, 0x0560 },		//  96 ,  5
	{ 0x2741, 0x0561 },		//  97 ,  5
	{ 0x2742, 0x0562 },		//  98 ,  5
	{ 0x2743, 0x0563 },		//  99 ,  5
	{ 0x2744, 0x0564 },		//  100,  5
	{ 0x2745, 0x0565 },		//  101,  5
	{ 0x2746, 0x0566 },		//  102,  5
	{ 0x2747, 0x0567 },		//  103,  5
	{ 0x2748, 0x0568 },		//  104,  5
	{ 0x2749, 0x0569 },		//  105,  5
	{ 0x274A, 0x056a },		//  106,  5
	{ 0x274B, 0x056b },		//  107,  5
	{ 0x274D, 0x056d },		//  109,  5
	{ 0x274F, 0x056f },		//  111,  5
	{ 0x2750, 0x0570 },		//  112,  5
	{ 0x2751, 0x0571 },		//  113,  5
	{ 0x2752, 0x0572 },		//  114,  5
	{ 0x2756, 0x0576 },		//  118,  5
	{ 0x2758, 0x0578 },		//  120,  5
	{ 0x2759, 0x0579 },		//  121,  5
	{ 0x275A, 0x057a },		//  122,  5
	{ 0x275B, 0x057b },		//  123,  5
	{ 0x275C, 0x057c },		//  124,  5
	{ 0x275D, 0x057d },		//  125,  5
	{ 0x275E, 0x057e },		//  126,  5
	{ 0x2761, 0x05a1 },		//  161,  5
	{ 0x2762, 0x05a2 },		//  162,  5
	{ 0x2763, 0x05a3 },		//  163,  5
	{ 0x2764, 0x05a4 },		//  164,  5
	{ 0x2765, 0x05a5 },		//  165,  5
	{ 0x2766, 0x05a6 },		//  166,  5
	{ 0x2767, 0x05a7 },		//  167,  5
	{ 0x2776, 0x05b6 },		//  182,  5
	{ 0x2777, 0x05b7 },		//  183,  5
	{ 0x2778, 0x05b8 },		//  184,  5
	{ 0x2779, 0x05b9 },		//  185,  5
	{ 0x277A, 0x05ba },		//  186,  5
	{ 0x277B, 0x05bb },		//  187,  5
	{ 0x277C, 0x05bc },		//  188,  5
	{ 0x277D, 0x05bd },		//  189,  5
	{ 0x277E, 0x05be },		//  190,  5
	{ 0x277F, 0x05bf },		//  191,  5
	{ 0x2780, 0x05c0 },		//  192,  5
	{ 0x2781, 0x05c1 },		//  193,  5
	{ 0x2782, 0x05c2 },		//  194,  5
	{ 0x2783, 0x05c3 },		//  195,  5
	{ 0x2784, 0x05c4 },		//  196,  5
	{ 0x2785, 0x05c5 },		//  197,  5
	{ 0x2786, 0x05c6 },		//  198,  5
	{ 0x2787, 0x05c7 },		//  199,  5
	{ 0x2788, 0x05c8 },		//  200,  5
	{ 0x2789, 0x05c9 },		//  201,  5
	{ 0x278A, 0x05ca },		//  202,  5
	{ 0x278B, 0x05cb },		//  203,  5
	{ 0x278C, 0x05cc },		//  204,  5
	{ 0x278D, 0x05cd },		//  205,  5
	{ 0x278E, 0x05ce },		//  206,  5
	{ 0x278F, 0x05cf },		//  207,  5
	{ 0x2790, 0x05d0 },		//  208,  5
	{ 0x2791, 0x05d1 },		//  209,  5
	{ 0x2792, 0x05d2 },		//  210,  5
	{ 0x2793, 0x05d3 },		//  211,  5
	{ 0x2794, 0x05d4 },		//  212,  5
	{ 0x2798, 0x05d8 },		//  216,  5
	{ 0x2799, 0x05d9 },		//  217,  5
	{ 0x279A, 0x05da },		//  218,  5
	{ 0x279B, 0x05db },		//  219,  5
	{ 0x279C, 0x05dc },		//  220,  5
	{ 0x279D, 0x05dd },		//  221,  5
	{ 0x279E, 0x05de },		//  222,  5
	{ 0x279F, 0x05df },		//  223,  5
	{ 0x27A0, 0x05e0 },		//  224,  5
	{ 0x27A1, 0x05e1 },		//  225,  5
	{ 0x27A2, 0x05e2 },		//  226,  5
	{ 0x27A3, 0x05e3 },		//  227,  5
	{ 0x27A4, 0x05e4 },		//  228,  5
	{ 0x27A5, 0x05e5 },		//  229,  5
	{ 0x27A6, 0x05e6 },		//  230,  5
	{ 0x27A7, 0x05e7 },		//  231,  5
	{ 0x27A8, 0x05e8 },		//  232,  5
	{ 0x27A9, 0x05e9 },		//  233,  5
	{ 0x27AA, 0x05ea },		//  234,  5
	{ 0x27AB, 0x05eb },		//  235,  5
	{ 0x27AC, 0x05ec },		//  236,  5
	{ 0x27AD, 0x05ed },		//  237,  5
	{ 0x27AE, 0x05ee },		//  238,  5
	{ 0x27AF, 0x05ef },		//  239,  5
	{ 0x27B1, 0x05f1 },		//  241,  5
	{ 0x27B2, 0x05f2 },		//  242,  5
	{ 0x27B3, 0x05f3 },		//  243,  5
	{ 0x27B4, 0x05f4 },		//  244,  5
	{ 0x27B5, 0x05f5 },		//  245,  5
	{ 0x27B6, 0x05f6 },		//  246,  5
	{ 0x27B7, 0x05f7 },		//  247,  5
	{ 0x27B8, 0x05f8 },		//  248,  5
	{ 0x27B9, 0x05f9 },		//  249,  5
	{ 0x27BA, 0x05fa },		//  250,  5
	{ 0x27BB, 0x05fb },		//  251,  5
	{ 0x27BC, 0x05fc },		//  252,  5
	{ 0x27BD, 0x05fd },		//  253,  5
	{ 0x27BE, 0x05fe },		//  254,  5

	// Range 0xE000 through 0xF8FF is reserved for private use.
	// We cannot try to interpret characters in this range nor
	// assign any default collation or meaning.

	{ 0xFB00, 0x0433 },		//  51 ,  4
	{ 0xFB01, 0x0436 },		//  54 ,  4
	{ 0xFB02, 0x0437 },		//  55 ,  4
	{ 0xFB03, 0x0434 },		//  52 ,  4
	{ 0xFB04, 0x0435 },		//  53 ,  4
	{ 0xFB1E, 0x0930 },		//  48 ,  9
	{ 0xFF61, 0x0b00 },		//   0 , 11
	{ 0xFF62, 0x0b01 },		//   1 , 11
	{ 0xFF63, 0x0b02 },		//   2 , 11
	{ 0xFF64, 0x0b03 },		//   3 , 11
	{ 0xFF65, 0x0b04 },		//   4 , 11
	{ 0xFF66, 0x0b05 },		//   5 , 11
	{ 0xFF67, 0x0b06 },		//   6 , 11
	{ 0xFF68, 0x0b07 },		//   7 , 11
	{ 0xFF69, 0x0b08 },		//   8 , 11
	{ 0xFF6A, 0x0b09 },		//   9 , 11
	{ 0xFF6B, 0x0b0a },		//  10 , 11
	{ 0xFF6C, 0x0b0b },		//  11 , 11
	{ 0xFF6D, 0x0b0c },		//  12 , 11
	{ 0xFF6E, 0x0b0d },		//  13 , 11
	{ 0xFF6F, 0x0b0e },		//  14 , 11
	{ 0xFF70, 0x0b0f },		//  15 , 11
	{ 0xFF71, 0x0b10 },		//  16 , 11
	{ 0xFF72, 0x0b11 },		//  17 , 11
	{ 0xFF73, 0x0b12 },		//  18 , 11
	{ 0xFF74, 0x0b13 },		//  19 , 11
	{ 0xFF75, 0x0b14 },		//  20 , 11
	{ 0xFF76, 0x0b15 },		//  21 , 11
	{ 0xFF77, 0x0b16 },		//  22 , 11
	{ 0xFF78, 0x0b17 },		//  23 , 11
	{ 0xFF79, 0x0b18 },		//  24 , 11
	{ 0xFF7A, 0x0b19 },		//  25 , 11
	{ 0xFF7B, 0x0b1a },		//  26 , 11
	{ 0xFF7C, 0x0b1b },		//  27 , 11
	{ 0xFF7D, 0x0b1c },		//  28 , 11
	{ 0xFF7E, 0x0b1d },		//  29 , 11
	{ 0xFF7F, 0x0b1e },		//  30 , 11
	{ 0xFF80, 0x0b1f },		//  31 , 11
	{ 0xFF81, 0x0b20 },		//  32 , 11
	{ 0xFF82, 0x0b21 },		//  33 , 11
	{ 0xFF83, 0x0b22 },		//  34 , 11
	{ 0xFF84, 0x0b23 },		//  35 , 11
	{ 0xFF85, 0x0b24 },		//  36 , 11
	{ 0xFF86, 0x0b25 },		//  37 , 11
	{ 0xFF87, 0x0b26 },		//  38 , 11
	{ 0xFF88, 0x0b27 },		//  39 , 11
	{ 0xFF89, 0x0b28 },		//  40 , 11
	{ 0xFF8A, 0x0b29 },		//  41 , 11
	{ 0xFF8B, 0x0b2a },		//  42 , 11
	{ 0xFF8C, 0x0b2b },		//  43 , 11
	{ 0xFF8D, 0x0b2c },		//  44 , 11
	{ 0xFF8E, 0x0b2d },		//  45 , 11
	{ 0xFF8F, 0x0b2e },		//  46 , 11
	{ 0xFF90, 0x0b2f },		//  47 , 11
	{ 0xFF91, 0x0b30 },		//  48 , 11
	{ 0xFF92, 0x0b31 },		//  49 , 11
	{ 0xFF93, 0x0b32 },		//  50 , 11
	{ 0xFF94, 0x0b33 },		//  51 , 11
	{ 0xFF95, 0x0b34 },		//  52 , 11
	{ 0xFF96, 0x0b35 },		//  53 , 11
	{ 0xFF97, 0x0b36 },		//  54 , 11
	{ 0xFF98, 0x0b37 },		//  55 , 11
	{ 0xFF99, 0x0b38 },		//  56 , 11
	{ 0xFF9A, 0x0b39 },		//  57 , 11
	{ 0xFF9B, 0x0b3a },		//  58 , 11
	{ 0xFF9C, 0x0b3b },		//  59 , 11
	{ 0xFF9D, 0x0b3c },		//  60 , 11
	{ 0xFF9E, 0x0b3d },		//  61 , 11
	{ 0xFF9F, 0x0b3e }		//  62 , 11
};

/****************************************************************************
Desc:
****************************************************************************/
FINLINE FLMUINT bytesInBits(
	FLMUINT		uiBits)
{
	return( (uiBits + 7) >> 3);
}

/****************************************************************************
Desc:
****************************************************************************/
FINLINE FLMBOOL testOneBit(
	const FLMBYTE *	pucBuf,
	FLMUINT				uiBit)
{
	return( (((pucBuf[ uiBit >> 3]) >> (7 - (uiBit & 7))) & 1)
		? TRUE
		: FALSE);
}

/****************************************************************************
Desc:
****************************************************************************/
FINLINE FLMUINT getNBits(
	FLMUINT				uiNumBits,
	const FLMBYTE *	pucBuf,
	FLMUINT				uiBit)
{
	return(((FLMUINT)(
		((FLMUINT)pucBuf[ uiBit >> 3] << 8) |		// append high bits (byte 1) to ...
		(FLMUINT)pucBuf[ (uiBit >> 3) + 1]) >>		// ... overflow bits in 2nd byte
		(16 - uiNumBits - (uiBit & 7))) &  			// reposition to low end of value
		((1 << uiNumBits) - 1));				  		// mask off high bits
}

/****************************************************************************
Desc:
****************************************************************************/
FINLINE void setBit(
	FLMBYTE *	pucBuf,
	FLMUINT		uiBit)
{
	pucBuf[ uiBit >> 3] |= (FLMBYTE)(1 << (7 - (uiBit & 7)));
}

/****************************************************************************
Desc:
****************************************************************************/
FINLINE void setBits(
	FLMUINT		uiCount,
	FLMBYTE *	pucBuf,
	FLMUINT		uiBit,
	FLMUINT		uiVal)
{
	pucBuf[ uiBit >> 3] |= 		  					// 1st byte
			(FLMBYTE)((uiVal << (8 - uiCount)) 	// Align to bit 0
			>>
			(uiBit & 7)); 				  				// Re-align to actual bit position

	pucBuf[ (uiBit >> 3) + 1] = 					// 2nd byte
			(FLMBYTE)(uiVal
			<<
			(16 - uiCount - (uiBit & 7))); 		// Align spill-over bits
}

/****************************************************************************
Desc: Returns TRUE if the character is upper case, FALSE if lower case.
****************************************************************************/
FINLINE FLMBOOL charIsUpper(
	FLMUINT16	ui16Char)
{
	return( (FLMBOOL)((ui16Char < 0x7F)
							? (FLMBOOL)((ui16Char >= ASCII_LOWER_A &&
											 ui16Char <= ASCII_LOWER_Z)
											 ? (FLMBOOL)FALSE
											 : (FLMBOOL)TRUE)
							: f_wpIsUpper( ui16Char)));
}

/****************************************************************************
Desc:		flmGetNextCharState can be thought of as a 2 dimentional array with
			i and j as the row and column indicators respectively.  If a value
			exists at the intersection of i and j, it is returned.  Sparse array
			techniques are used to minimize memory usage.

Return:	0 = no valid next state
			non-zero = valid next state, offset for action, or collating value
****************************************************************************/
FINLINE FLMUINT16 flmGetNextCharState(
	FLMUINT		i,
	FLMUINT		j)
{
	FLMUINT		k;
	FLMUINT		x;

	for( k = fwp_indexi[ x = (i > START_COL) ? (START_ALL) : i];
		  k <= (FLMUINT) (fwp_indexi[ x + 1] - 1);
		  k++ )
	{
		if(  j == fwp_indexj[ k])
		{
			return( fwp_valuea[ (i > START_COL)
				?	(k + (FIXUP_AREA_SIZE * (i - START_ALL)))
				: k]);
		}
	}

	return( 0);
}

/****************************************************************************
Desc:		Convert a Unicode character to its WP equivalent
Ret:		Returns TRUE if the character could be converted
****************************************************************************/
FLMBOOL FTKAPI f_unicodeToWP(
	FLMUNICODE			uUniChar,		// Unicode character to convert
	FLMUINT16 *			pui16WPChar)	// Returns 0 or WPChar converted.
{
	if( uUniChar <= 127)
	{
		// Character is in the ASCII conversion range

		*pui16WPChar = uUniChar;
		return( TRUE);
	}

	if( uUniChar < gv_uiMinUniChar || uUniChar > gv_uiMaxUniChar)
	{
		*pui16WPChar = 0;
		return( FALSE);
	}

	if( (*pui16WPChar = gv_pUnicodeToWP60[ uUniChar - gv_uiMinUniChar]) != 0)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Convert a Unicode character to its WP equivalent using the
			depricated FLAIM conversion rules
Ret:		Returns TRUE if the character could be converted
****************************************************************************/
FLMBOOL FTKAPI f_depricatedUnicodeToWP(
	FLMUNICODE			uUniChar,		// Unicode character to convert
	FLMUINT16 *			pui16WPChar)	// Returns 0 or WPChar converted.
{
	if( uUniChar < 127)
	{
		*pui16WPChar = uUniChar;
		return( TRUE);
	}

	if( uUniChar < gv_uiMinUniChar || 
		 uUniChar > gv_uiMaxUniChar ||
		 uUniChar > 0x222E)
	{
		*pui16WPChar = 0;
		return( FALSE);
	}

	if( (*pui16WPChar = gv_pUnicodeToWP60[ uUniChar - gv_uiMinUniChar]) != 0)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Convert a WP character to its Unicode equivalent
****************************************************************************/
RCODE FTKAPI f_wpToUnicode(
	FLMUINT16			ui16WPChar,
	FLMUNICODE *		puUniChar)
{
	if( ui16WPChar <= 127)
	{
		// Character is in the ASCII conversion range

		*puUniChar = (FLMUNICODE)ui16WPChar;
		return( NE_FLM_OK);
	}

	if( ui16WPChar < gv_uiMinWPChar || ui16WPChar > gv_uiMaxWPChar)
	{
		*puUniChar = 0;
		return( RC_SET( NE_FLM_CONV_ILLEGAL));
	}

	if( (*puUniChar = gv_pWP60ToUnicode[ ui16WPChar - gv_uiMinWPChar]) == 0)
	{
		return( RC_SET( NE_FLM_CONV_ILLEGAL));
	}

	return( NE_FLM_OK);
}

/****************************************************************************
Desc:	Reads the next character from the storage buffer
****************************************************************************/
FINLINE RCODE flmGetCharFromUTF8Buf(
	const FLMBYTE **		ppucBuf,
	const FLMBYTE *		pucEnd,
	FLMUNICODE *			puChar)
{
	const FLMBYTE *	pucBuf = *ppucBuf;
	FLMUINT				uiMaxLen = pucEnd ? (FLMUINT)(pucEnd - *ppucBuf) : 3;

	if( !uiMaxLen)
	{
		*puChar = 0;
		return( NE_FLM_OK);
	}
	
	if( pucBuf[ 0] <= 0x7F)
	{
		if( (*puChar = (FLMUNICODE)pucBuf[ 0]) != 0)
		{
			(*ppucBuf)++;
		}
		return( NE_FLM_OK);
	}

	if( uiMaxLen < 2 || (pucBuf[ 1] >> 6) != 0x02)
	{
		return( RC_SET( NE_FLM_BAD_UTF8));
	}

	if( (pucBuf[ 0] >> 5) == 0x06)
	{
		*puChar = 
			(FLMUNICODE)(((FLMUNICODE)( pucBuf[ 0] - 0xC0) << 6) +
							(FLMUNICODE)(pucBuf[ 1] - 0x80));
		(*ppucBuf) += 2;
		return( NE_FLM_OK);
	}

	if( uiMaxLen < 3 ||
		 (pucBuf[ 0] >> 4) != 0x0E ||
		 (pucBuf[ 2] >> 6) != 0x02)
	{
		return( RC_SET( NE_FLM_BAD_UTF8));
	}

	*puChar = 
		(FLMUNICODE)(((FLMUNICODE)(pucBuf[ 0] - 0xE0) << 12) +
			((FLMUNICODE)(pucBuf[ 1] - 0x80) << 6) +
						(FLMUNICODE)(pucBuf[ 2] - 0x80));
	(*ppucBuf) += 3;

	return( NE_FLM_OK);
}

/****************************************************************************
Desc: 	Convert a Unicode character to UTF-8
*****************************************************************************/
FINLINE RCODE flmUni2UTF8(
	FLMUNICODE		uChar,
	FLMBYTE *		pucBuf,
	FLMUINT *		puiBufSize)
{
	if( uChar <= 0x007F)
	{
		if( pucBuf)
		{
			if( *puiBufSize < 1)
			{
				return( RC_SET( NE_FLM_CONV_DEST_OVERFLOW));
			}

			*pucBuf = (FLMBYTE)uChar;
		}
		*puiBufSize = 1;
	}
	else if( uChar <= 0x07FF)
	{
		if( pucBuf)
		{
			if( *puiBufSize < 2)
			{
				return( RC_SET( NE_FLM_CONV_DEST_OVERFLOW));
			}

			*pucBuf++ = (FLMBYTE)(0xC0 | (FLMBYTE)(uChar >> 6));
			*pucBuf = (FLMBYTE)(0x80 | (FLMBYTE)(uChar & 0x003F));
		}
		*puiBufSize = 2;
	}
	else
	{
		if( pucBuf)
		{
			if( *puiBufSize < 3)
			{
				return( RC_SET( NE_FLM_CONV_DEST_OVERFLOW));
			}

			*pucBuf++ = (FLMBYTE)(0xE0 | (FLMBYTE)(uChar >> 12));
			*pucBuf++ = (FLMBYTE)(0x80 | (FLMBYTE)((uChar & 0x0FC0) >> 6));
			*pucBuf = (FLMBYTE)(0x80 | (FLMBYTE)(uChar & 0x003F));
		}
		*puiBufSize = 3;
	}

	return( NE_FLM_OK);
}

/****************************************************************************
Desc:		Reads the next UTF-8 character from a UTF-8 buffer
Notes:	This routine assumes that the destination buffer can hold at least
			three bytes
****************************************************************************/
FINLINE RCODE flmGetUTF8CharFromUTF8Buf(
	FLMBYTE **		ppucBuf,
	FLMBYTE *		pucEnd,
	FLMBYTE *		pucDestBuf,
	FLMUINT *		puiLen)
{
	FLMBYTE *	pucBuf = *ppucBuf;
	FLMUINT		uiMaxLen = pucEnd ? (FLMUINT)(pucEnd - *ppucBuf) : 3;

	if( !uiMaxLen || !pucBuf[ 0])
	{
		*puiLen = 0;
		return( NE_FLM_OK);
	}
	
	if( pucBuf[ 0] <= 0x7F)
	{
		*pucDestBuf = pucBuf[ 0];
		(*ppucBuf)++;
		*puiLen = 1;
		return( NE_FLM_OK);
	}

	if( uiMaxLen < 2 || (pucBuf[ 1] >> 6) != 0x02)
	{
		return( RC_SET( NE_FLM_BAD_UTF8));
	}

	if( (pucBuf[ 0] >> 5) == 0x06)
	{
		pucDestBuf[ 0] = pucBuf[ 0];
		pucDestBuf[ 1] = pucBuf[ 1];
		(*ppucBuf) += 2;
		*puiLen = 2;
		return( NE_FLM_OK);
	}

	if( uiMaxLen < 3 ||
		 (pucBuf[ 0] >> 4) != 0x0E || 
		 (pucBuf[ 2] >> 6) != 0x02)
	{
		return( RC_SET( NE_FLM_BAD_UTF8));
	}

	pucDestBuf[ 0] = pucBuf[ 0];
	pucDestBuf[ 1] = pucBuf[ 1];
	pucDestBuf[ 2] = pucBuf[ 2];
	(*ppucBuf) += 3;
	*puiLen = 3;

	return( NE_FLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FINLINE RCODE flmGetUTF8Length(
	const FLMBYTE *	pucBuf,
	FLMUINT				uiBufLen,
	FLMUINT *			puiBytes,
	FLMUINT *			puiChars)
{
	const FLMBYTE *	pucStart = pucBuf;
	const FLMBYTE *	pucEnd = uiBufLen ? (pucStart + uiBufLen) : NULL;
	FLMUINT				uiChars = 0;

	if (!pucBuf)
	{
		goto Exit;
	}

	while( (!pucEnd || pucBuf < pucEnd) && *pucBuf)
	{
		if( *pucBuf <= 0x7F)
		{
			pucBuf++;
			uiChars++;
			continue;
		}
	
		if( (pucEnd && pucBuf + 1 >= pucEnd) ||
			 (pucBuf[ 1] >> 6) != 0x02)
		{
			return( RC_SET( NE_FLM_BAD_UTF8));
		}
	
		if( ((*pucBuf) >> 5) == 0x06)
		{
			pucBuf += 2;
			uiChars++;
			continue;
		}
	
		if( (pucEnd && pucBuf + 2 >= pucEnd) ||
			 (pucBuf[ 0] >> 4) != 0x0E || 
			 (pucBuf[ 2] >> 6) != 0x02)
		{
			return( RC_SET( NE_FLM_BAD_UTF8));
		}
		
		pucBuf += 3;
		uiChars++;
	}

Exit:

	*puiChars = uiChars;
	if (pucEnd && pucBuf == pucEnd)
	{
		*puiBytes = (FLMUINT)(pucBuf - pucStart);
	}
	else
	{
		// Hit a null byte
		*puiBytes = (FLMUINT)(pucBuf - pucStart) + 1;
	}

	return( NE_FLM_OK);
}

/****************************************************************************
Desc:	Converts a character to upper case (if possible)
****************************************************************************/
FLMUINT16 FTKAPI f_wpUpper(
	FLMUINT16	ui16WpChar)
{
	if( ui16WpChar < 256)
	{
		if( ui16WpChar >= ASCII_LOWER_A && ui16WpChar <= ASCII_LOWER_Z)
		{
			// Return ASCII upper case

			return( ui16WpChar & 0xdf);
		}
	}
	else
	{
		FLMBYTE	ucCharSet = (FLMBYTE)(ui16WpChar >> 8);

		if( ucCharSet == F_CHSMUL1)
		{
			FLMBYTE	ucChar = (FLMBYTE)(ui16WpChar & 0xFF);

			if( ucChar >= fwp_caseConvertableRange[ (F_CHSMUL1 - 1) * 2] &&
				 ucChar <= fwp_caseConvertableRange[ ((F_CHSMUL1 - 1) * 2) + 1])
			{
				return( ui16WpChar & 0xFFFE);
			}
		}
		else if( ucCharSet == F_CHSGREK)
		{
			if( (ui16WpChar & 0xFF) <=
						fwp_caseConvertableRange[ ((F_CHSGREK - 1) * 2) + 1])
			{
				return( ui16WpChar & 0xFFFE);
			}
		}
		else if( ucCharSet == F_CHSCYR)
		{
			if( (ui16WpChar & 0xFF) <=
						fwp_caseConvertableRange[ ((F_CHSCYR - 1) * 2) + 1])
			{
				return( ui16WpChar & 0xFFFE);
			}
		}
		else if( ui16WpChar >= Lower_JP_a)
		{
			// Possible double byte character set alphabetic character?

			if( ui16WpChar <= Lower_JP_z)
			{
				// Japanese?

				ui16WpChar = (ui16WpChar - Lower_JP_a) + Upper_JP_A;
			}
			else if( ui16WpChar >= Lower_KR_a && ui16WpChar <= Lower_KR_z)
			{
				// Korean?

				ui16WpChar = (ui16WpChar - Lower_KR_a) + Upper_KR_A;
			}
			else if( ui16WpChar >= Lower_CS_a && ui16WpChar <= Lower_CS_z)
			{
				// Chinese Simplified?

				ui16WpChar = (ui16WpChar - Lower_CS_a) + Upper_CS_A;
			}
			else if( ui16WpChar >= Lower_CT_a && ui16WpChar <= Lower_CT_z)
			{
				// Chinese Traditional?

				ui16WpChar = (ui16WpChar - Lower_CT_a) + Upper_CT_A;
			}
		}
	}

	// Return original character - original not in lower case.

	return( ui16WpChar);
}

/****************************************************************************
Desc:	Checks to see if WP character is upper case
****************************************************************************/
FLMBOOL FTKAPI f_wpIsUpper(
	FLMUINT16	ui16WpChar)
{
	FLMBYTE	ucChar;
	FLMBYTE	ucCharSet;

	// Get character

	ucChar = (FLMBYTE)(ui16WpChar & 0xFF);

	// Test if ASCII character set

	if( !(ui16WpChar & 0xFF00))
	{
		return( (ucChar >= ASCII_LOWER_A && ucChar <= ASCII_LOWER_Z)
				  ? FALSE
				  : TRUE);
	}

	// Get the character set

	ucCharSet = (FLMBYTE) (ui16WpChar >> 8);

	if( (ucCharSet == F_CHSMUL1 && ucChar >= 26 && ucChar <= 241) ||
		 (ucCharSet == F_CHSGREK && ucChar <= 69) ||
		 (ucCharSet == F_CHSCYR && ucChar <= 199))
	{
		return( (ucChar & 1) ? FALSE : TRUE);
	}

	// Don't care that double ss is lower

	return( TRUE);
}

/****************************************************************************
Desc:	Converts a character to lower case (if possible)
****************************************************************************/
FLMUINT16 FTKAPI f_wpLower(
	FLMUINT16	ui16WpChar)
{
	if( ui16WpChar < 256)
	{
		if( ui16WpChar >= ASCII_UPPER_A && ui16WpChar <= ASCII_UPPER_Z)
		{
			return( ui16WpChar | 0x20);
		}
	}
	else
	{
		FLMBYTE	ucCharSet = (FLMBYTE)(ui16WpChar >> 8);

		if( ucCharSet == F_CHSMUL1)
		{
			FLMBYTE	ucChar = (FLMBYTE)(ui16WpChar & 0xFF);

			if( ucChar >= fwp_caseConvertableRange[ (F_CHSMUL1 - 1) * 2] &&
				 ucChar <= fwp_caseConvertableRange[ ((F_CHSMUL1 - 1) * 2) + 1] )
			{
				return( ui16WpChar | 1);
			}
		}
		else if( ucCharSet == F_CHSGREK)
		{
			if( (ui16WpChar & 0xFF) <=
						fwp_caseConvertableRange[ ((F_CHSGREK - 1) * 2) + 1])
			{
				return( ui16WpChar | 1);
			}
		}
		else if( ucCharSet == F_CHSCYR)
		{
			if( (ui16WpChar & 0xFF) <=
						fwp_caseConvertableRange[ ((F_CHSCYR-1) * 2) + 1])
			{
				return( ui16WpChar | 1);
			}
		}
		else if( ui16WpChar >= Upper_JP_A)
		{
			// Possible double byte character set alphabetic character?

			if( ui16WpChar <= Upper_JP_Z)
			{
				// Japanese?

				ui16WpChar = ui16WpChar - Upper_JP_A + Lower_JP_a;
			}
			else if( ui16WpChar >= Upper_KR_A && ui16WpChar <= Upper_KR_Z)
			{
				// Korean?

				ui16WpChar = ui16WpChar - Upper_KR_A + Lower_KR_a;
			}
			else if( ui16WpChar >= Upper_CS_A && ui16WpChar <= Upper_CS_Z)
			{
				// Chinese Simplified?

				ui16WpChar = ui16WpChar - Upper_CS_A + Lower_CS_a;
			}
			else if( ui16WpChar >= Upper_CT_A && ui16WpChar <= Upper_CT_Z)
			{
				// Chinese Traditional?

				ui16WpChar = ui16WpChar - Upper_CT_A + Lower_CT_a;
			}
		}
	}

	// Return original character, original not in upper case

	return( ui16WpChar);
}

/****************************************************************************
Desc:	Break a WP character into a base and a diacritical char.
****************************************************************************/
FLMBOOL FTKAPI f_breakWPChar(
	FLMUINT16		ui16WpChar,
	FLMUINT16 *		pui16BaseChar,
	FLMUINT16 *		pui16DiacriticChar)
{
	BASE_DIACRIT *		pBaseDiacritic;
	FLMINT				iTableIndex;

	if( HI(ui16WpChar) >= F_NCHSETS ||
		 (pBaseDiacritic = fwp_car60_c[ HI(ui16WpChar)]) == 0)
	{
		return( TRUE);
	}

	iTableIndex = ((FLMBYTE)ui16WpChar) - pBaseDiacritic->start_char;
	
	if( iTableIndex < 0 ||
		 iTableIndex >= pBaseDiacritic->char_count ||
		 pBaseDiacritic->table [iTableIndex].base == (FLMBYTE)0xFF)
	{
		return( TRUE);
	}

	if( (HI( ui16WpChar) != F_CHSMUL1) ||
		 ((fwp_ml1_cb60[ ((FLMBYTE) ui16WpChar) >> 3] >>
			(7 - (ui16WpChar & 0x07))) & 0x01))
	{

		// normal case, same base as same as characters

		*pui16BaseChar = (ui16WpChar & 0xFF00) |
								pBaseDiacritic->table [iTableIndex].base;
		*pui16DiacriticChar = (ui16WpChar & 0xFF00) |
								pBaseDiacritic->table[iTableIndex].diacrit;
	}
	else
	{

		// Multi-national where base is ascii value.

		*pui16BaseChar = pBaseDiacritic->table [iTableIndex].base;
		*pui16DiacriticChar = (ui16WpChar & 0xFF00) |
										pBaseDiacritic->table[iTableIndex].diacrit;
	}

	return( FALSE);
}

/****************************************************************************
Desc:	Take a base and a diacritic and compose a WP character.
		Note on base character: i's and j's must be dotless i's and j's (for
		those which use them) or they will not be found.
Ret: 	TRUE - if not found
		FALSE  - if found
Notes: ascii characters with diacriticals are in multi-national if anywhere;
		 all other base chars with diacritics are found in their own sets.
****************************************************************************/
FLMBOOL FTKAPI f_combineWPChar(
	FLMUINT16 *		pui16WpChar,
	FLMUINT16		ui16BaseChar,
	FLMINT16			ui16DiacriticChar)
{
	FLMUINT						uiRemaining;
	FLMBYTE						ucCharSet;
	FLMBYTE						ucChar;
	BASE_DIACRIT *				pBaseDiacritic;
	BASE_DIACRIT_TABLE *		pTable;

	ucCharSet = HI( ui16BaseChar);
	
	if( ucCharSet >= F_NCHSETS)
	{
		return( TRUE);
	}

	// Is base ASCII?  If so, look in multinational 1

	if( !ucCharSet)
	{
		ucCharSet = F_CHSMUL1;
	}

	if( ucCharSet >= F_NCHSETS ||
		 (pBaseDiacritic = fwp_car60_c[ ucCharSet]) == 0)
	{
		return( TRUE);
	}

	ucChar = LO( ui16BaseChar);
	ui16DiacriticChar = LO( ui16DiacriticChar);
	pTable = pBaseDiacritic->table;

	for( uiRemaining = pBaseDiacritic->char_count;
		  uiRemaining;
		  uiRemaining--, pTable++ )
	{
		// Same base?

		if( pTable->base == ucChar &&
			 (pTable->diacrit & 0x7F) == ui16DiacriticChar)
		{
			// Same diacritic?

			*pui16WpChar = (FLMUINT16) (((FLMUINT16) ucCharSet << 8) +
					(pBaseDiacritic->start_char +
					 (FLMUINT16)(pTable - pBaseDiacritic->table)));
			return( FALSE);
		}
	}

	return( TRUE);
}

/**************************************************************************
Desc:	Find the collating value of a WP character
ret:	Collating value (COLS0 is high value - undefined WP char)
***********************************************************************/
FLMUINT16 FTKAPI f_wpGetCollationImp(
	FLMUINT16		ui16WpChar,
	FLMUINT			uiLanguage)
{
	FLMUINT16		ui16State;
	FLMBYTE			ucCharVal;
	FLMBYTE			ucCharSet;
	FLMBOOL			bHebrewArabicFlag;
	TBL_B_TO_BP *	pColTbl;

	if( uiLanguage == FLM_US_LANG)
	{
		return( gv_pui16USCollationTable[ ui16WpChar]);
	}
	else if( uiLanguage == FLM_AR_LANG || uiLanguage == FLM_FA_LANG ||
				uiLanguage == FLM_HE_LANG || uiLanguage == FLM_UR_LANG)
	{
		pColTbl = fwp_HebArabicCol60Tbl;
		bHebrewArabicFlag = TRUE;
	}
	else
	{
		// Check if uiLanguage candidate for alternate double collating

		ui16State = flmGetNextCharState( START_COL, uiLanguage);
		if( 0 != (ui16State = flmGetNextCharState( (ui16State
						?	ui16State		// look at special case languages
						:	START_ALL),		// look at US and European
						(FLMUINT) ui16WpChar)))
		{
			return( ui16State);
		}

		pColTbl = fwp_col60Tbl;
		bHebrewArabicFlag = FALSE;
	}

	ucCharVal = (FLMBYTE)ui16WpChar;
	ucCharSet = (FLMBYTE)(ui16WpChar >> 8);

	do
	{
		if( pColTbl->key == ucCharSet)
		{
			FLMBYTE *	pucColVals = pColTbl->charPtr;

			// Check if the value is in the range of collated chars
			// Above lower range of table?

			if( ucCharVal >= *pucColVals)
			{
				// Make value zero based to index

				ucCharVal -= *pucColVals++;

				// Below maximum number of table entries?

				if( ucCharVal < *pucColVals++)
				{
					// Return collated value.

					return( pucColVals[ ucCharVal]);
				}
			}
		}

		// Go to next table entry

		pColTbl++;

	} while( pColTbl->key != 0xFF);

	if( bHebrewArabicFlag)
	{
		if( ucCharSet == F_CHSHEB || ucCharSet == F_CHSARB1 || 
			 ucCharSet == F_CHSARB2)
		{
			return( COLS0_ARABIC);
		}
	}

	// Defaults for characters that don't have a collation value.

	return( COLS0);
}

/****************************************************************************
Desc:	Check for double characters that sort as 1 (like ch in Spanish) or
		1 character that should sort as 2 (like ? sorts as ae in French).
Return:	0 = nothing changes
			1 if sorting 2 characters as 1 - *pui16WpChar is the one character.
			second character value if 1 character sorts as 2,
			*pui16WpChar changes to first character in sequence
****************************************************************************/
RCODE FTKAPI f_wpCheckDoubleCollation(
	IF_PosIStream *	pIStream,
	FLMBOOL				bUnicodeStream,
	FLMBOOL				bAllowTwoIntoOne,
	FLMUNICODE *		puzChar,
	FLMUNICODE *		puzChar2,
	FLMBOOL *			pbTwoIntoOne,
	FLMUINT				uiLanguage)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT16			ui16CurState;
	FLMUINT16			ui16WpChar;
	FLMUNICODE			uzLastChar = 0;
	FLMUNICODE			uChar = *puzChar;
	FLMUNICODE			uDummy;
	FLMBOOL				bUpperFlag;
	FLMUINT64			ui64SavePosition = pIStream->getCurrPosition();

	if (!f_unicodeToWP( *puzChar, &ui16WpChar))
	{
		ui16WpChar = UNK_UNICODE_CODE;
	}
	bUpperFlag = f_wpIsUpper( ui16WpChar);

	*pbTwoIntoOne = FALSE;
	*puzChar2 = 0;
	if ((ui16CurState = flmGetNextCharState( 0, uiLanguage)) == 0)
	{
		goto Exit;
	}

	for (;;)
	{
		switch (ui16CurState)
		{
			case INSTSG:
			{
				*puzChar = *puzChar2 = (FLMUNICODE)f_toascii( 's');
				*pbTwoIntoOne = FALSE;
				goto Exit;
			}
			
			case INSTAE:
			{
				if (bUpperFlag)
				{
					*puzChar = (FLMUNICODE)f_toascii( 'A');
					*puzChar2 = (FLMUNICODE)f_toascii( 'E');
				}
				else
				{
					*puzChar = (FLMUNICODE)f_toascii( 'a');
					*puzChar2 = (FLMUNICODE)f_toascii( 'e');
				}
				*pbTwoIntoOne = FALSE;
				goto Exit;
			}
			
			case INSTIJ:
			{
				if (bUpperFlag)
				{
					*puzChar = (FLMUNICODE)f_toascii( 'I');
					*puzChar2 = (FLMUNICODE)f_toascii( 'J');
				}
				else
				{
					*puzChar = (FLMUNICODE)f_toascii( 'i');
					*puzChar2 = (FLMUNICODE)f_toascii( 'j');
				}
				*pbTwoIntoOne = FALSE;
				goto Exit;
			}
			
			case INSTOE:
			{
				if (bUpperFlag)
				{
					*puzChar = (FLMUNICODE)f_toascii( 'O');
					*puzChar2 = (FLMUNICODE)f_toascii( 'E');
				}
				else
				{
					*puzChar = (FLMUNICODE)f_toascii( 'o');
					*puzChar2 = (FLMUNICODE)f_toascii( 'e');
				}
				*pbTwoIntoOne = FALSE;
				goto Exit;
			}
			
			case WITHAA:
			{
				*puzChar = (FLMUNICODE)(bUpperFlag
													? (FLMUNICODE)0xC5
													: (FLMUNICODE)0xE5);
													
				if (RC_BAD( rc = pIStream->positionTo( ui64SavePosition)))
				{
					goto Exit;
				}

				if( bUnicodeStream)
				{
					rc = pIStream->read( &uDummy, sizeof( FLMUNICODE), NULL);
				}
				else
				{
					rc = f_readUTF8CharAsUnicode( pIStream, &uDummy);
				}
				
				if( RC_BAD( rc))
				{
					if (rc == NE_FLM_EOF_HIT)
					{
						rc = NE_FLM_OK;
					}
					else
					{
						goto Exit;
					}
				}

				ui64SavePosition = pIStream->getCurrPosition();
				break;
			}
			
			case AFTERC:
			{
				*puzChar = (FLMUINT16)(bUpperFlag
													? (FLMUNICODE)f_toascii( 'C')
													: (FLMUNICODE)f_toascii( 'c'));
Position_After_2nd:

				if( bAllowTwoIntoOne)
				{
					*puzChar2 = uzLastChar;
					*pbTwoIntoOne = TRUE;

					if (RC_BAD( rc = pIStream->positionTo( ui64SavePosition)))
					{
						goto Exit;
					}
					
					if( bUnicodeStream)
					{
						rc = pIStream->read( &uChar, sizeof( FLMUNICODE), NULL);
					}
					else
					{
						rc = f_readUTF8CharAsUnicode( pIStream, &uChar);
					}

					if (RC_BAD( rc))
					{
						if (rc == NE_FLM_EOF_HIT)
						{
							rc = NE_FLM_OK;
						}
						else
						{
							goto Exit;
						}
					}

					ui64SavePosition = pIStream->getCurrPosition();
				}
				goto Exit;
			}
			
			case AFTERH:
			{
				*puzChar = (FLMUINT16)(bUpperFlag
													? (FLMUNICODE)f_toascii( 'H')
													: (FLMUNICODE)f_toascii( 'h'));
				goto Position_After_2nd;
			}
			
			case AFTERL:
			{
				*puzChar = (FLMUINT16)(bUpperFlag
													? (FLMUNICODE)f_toascii( 'L')
													: (FLMUNICODE)f_toascii( 'l'));
				goto Position_After_2nd;
			}
			
			default:
			{
				// Handles STATE1 through STATE11 also
				break;
			}
		}

		if ((ui16CurState = flmGetNextCharState( ui16CurState,
									f_wpLower( ui16WpChar))) == 0)
		{
			break;
		}

		uzLastChar = uChar;
		
		if( bUnicodeStream)
		{
			rc = pIStream->read( &uChar, sizeof( FLMUNICODE), NULL);
		}
		else
		{
			rc = f_readUTF8CharAsUnicode( pIStream, &uChar);
		}

		if (RC_BAD( rc))
		{
			if (rc == NE_FLM_EOF_HIT)
			{
				rc = NE_FLM_OK;
			}
			else
			{
				goto Exit;
			}
		}

		if (!f_unicodeToWP( uChar, &ui16WpChar))
		{
			ui16WpChar = UNK_UNICODE_CODE;
		}
	}

Exit:

	if (RC_OK( rc))
	{
		rc = pIStream->positionTo( ui64SavePosition);
	}

	return( rc);
}

/****************************************************************************
Desc:		Check for double characters that sort as 1 (like ch in Spanish) or
			1 character that should sort as 2 (like � sorts as ae in French).
Return:	0 = nothing changes.  Otherwise, *pui16WpChar is the first
			character, and the return value contains the 2nd character.
			In addition, *pbTwoIntoOne will be TRUE if we should take two
			characters and treat as one (i.e, change the collation on the
			outside to one more than the collation of the first character).
****************************************************************************/
FLMUINT16 FTKAPI f_wpCheckDoubleCollation(
	FLMUINT16 *			pui16WpChar,
	FLMBOOL *			pbTwoIntoOne,
	const FLMBYTE **	ppucInputStr,
	FLMUINT				uiLanguage)
{
	FLMUINT16			ui16CurState;
	FLMUINT16			ui16WpChar;
	FLMUINT16			ui16SecondChar;
	FLMUINT16			ui16LastChar = 0;
	FLMUINT				uiInLen;
	FLMBOOL				bUpperFlag;

	ui16WpChar = *pui16WpChar;
	bUpperFlag = f_wpIsUpper( ui16WpChar);

	uiInLen = 0;
	ui16SecondChar = 0;

	// Primer read

	if ((ui16CurState = flmGetNextCharState( 0, uiLanguage)) == 0)
	{
		goto Exit;
	}

	for (;;)
	{
		switch (ui16CurState)
		{
			case INSTSG:
			{
				*pui16WpChar = ui16SecondChar = (FLMUINT16) f_toascii( 's');
				*pbTwoIntoOne = FALSE;
				goto Exit;
			}
			
			case INSTAE:
			{
				if (bUpperFlag)
				{
					*pui16WpChar = (FLMUINT16) f_toascii( 'A');
					ui16SecondChar = (FLMUINT16) f_toascii( 'E');
				}
				else
				{
					*pui16WpChar = (FLMUINT16) f_toascii( 'a');
					ui16SecondChar = (FLMUINT16) f_toascii( 'e');
				}

				*pbTwoIntoOne = FALSE;
				goto Exit;
			}
			
			case INSTIJ:
			{
				if (bUpperFlag)
				{
					*pui16WpChar = (FLMUINT16) f_toascii( 'I');
					ui16SecondChar = (FLMUINT16) f_toascii( 'J');
				}
				else
				{
					*pui16WpChar = (FLMUINT16) f_toascii( 'i');
					ui16SecondChar = (FLMUINT16) f_toascii( 'j');
				}

				*pbTwoIntoOne = FALSE;
				goto Exit;
			}
			
			case INSTOE:
			{
				if (bUpperFlag)
				{
					*pui16WpChar = (FLMUINT16) f_toascii( 'O');
					ui16SecondChar = (FLMUINT16) f_toascii( 'E');
				}
				else
				{
					*pui16WpChar = (FLMUINT16) f_toascii( 'o');
					ui16SecondChar = (FLMUINT16) f_toascii( 'e');
				}

				*pbTwoIntoOne = FALSE;
				goto Exit;
			}
			
			case WITHAA:
			{
				*pui16WpChar = (FLMUINT16) (bUpperFlag 
														? (FLMUINT16) 0x122 
														: (FLMUINT16) 0x123);
				(*ppucInputStr)++;
				break;
			}
			
			case AFTERC:
			{
				*pui16WpChar = (FLMUINT16) (bUpperFlag 
														? (FLMUINT16) f_toascii( 'C') 
														: (FLMUINT16) f_toascii( 'c'));
				ui16SecondChar = ui16LastChar;
				*pbTwoIntoOne = TRUE;
				(*ppucInputStr)++;
				goto Exit;
			}
			
			case AFTERH:
			{
				*pui16WpChar = (FLMUINT16) (bUpperFlag 
														? (FLMUINT16) f_toascii( 'H') 
														: (FLMUINT16) f_toascii( 'h'));
				ui16SecondChar = ui16LastChar;
				*pbTwoIntoOne = TRUE;
				(*ppucInputStr)++;
				goto Exit;
			}
			
			case AFTERL:
			{
				*pui16WpChar = (FLMUINT16) (bUpperFlag 
														? (FLMUINT16) f_toascii( 'L') 
														: (FLMUINT16) f_toascii( 'l'));
				ui16SecondChar = ui16LastChar;
				*pbTwoIntoOne = TRUE;
				(*ppucInputStr)++;
				goto Exit;
			}
			
			default:
			{

				// Handles STATE1 through STATE11 also

				break;
			}
		}

		if ((ui16CurState = flmGetNextCharState( ui16CurState, 
				f_wpLower( ui16WpChar))) == 0)
		{
			goto Exit;
		}

		ui16LastChar = ui16WpChar;
		ui16WpChar = (FLMUINT16) * ((*ppucInputStr) + (uiInLen++));
	}

Exit:

	return (ui16SecondChar);
}

/****************************************************************************
Desc:	Returns the collation value of the input WP character.
		If in charset 11 will convert the character to Zenkaku (double wide).
In:	ui16WpChar - Char to collate off of - could be in CS0..14 or x24..up
		ui16NextWpChar - next WP char for CS11 voicing marks
		ui16PrevColValue - previous collating value - for repeat/vowel repeat
		pui16ColValue - returns 2 byte collation value
		pui16SubColVal - 0, 6 or 16 bit value for the latin sub collation
								or the kana size & vowel voicing
								001 - set if large (upper) character
								010 - set if voiced
								100 - set if half voiced

		pucCaseBits - returns 2 bits
				Latin/Greek/Cyrillic
					01 - case bit set if character is uppercase
					10 - double wide character in CS 0x25xx, 0x26xx and 0x27xx
				Japanese
					00 - double wide hiragana 0x255e..25b0
					01 - double wide katakana 0x2600..2655
					10 - double wide symbols that map to charset 11
					11 - single wide katakana from charset 11
Ret:	0 - no valid collation value
				high values set for pui16ColValue
				Sub-collation gets original WP character value
		1 - valid collation value
		2 - valid collation value and used the ui16NextWpChar

Notes:	Code taken from XCH2COL.ASM - routine xch2col_f
			also from CMPWS.ASM - routine getcase
Terms:
	HANKAKU - single wide characters in charsets 0..14
	ZENKAKU - double wide characters in charsets 0x24..end of kanji
	KANJI   - collation values are 0x2900 less than WPChar value

****************************************************************************/
FLMUINT16 flmWPAsiaGetCollation(
	FLMUINT16		ui16WpChar,				// WP char to get collation values
	FLMUINT16		ui16NextWpChar,		// Next WP char - for CS11 voicing marks
	FLMUINT16   	ui16PrevColValue,		// Previous collating value
	FLMUINT16 *		pui16ColValue,			// Returns collation value
	FLMUINT16 * 	pui16SubColVal,		// Returns sub-collation value
	FLMBYTE *		pucCaseBits,		 	// Returns case bits value
	FLMBOOL			bUppercaseFlag)		// Set if to convert to uppercase
{
	FLMUINT16		ui16ColValue;
	FLMUINT16		ui16SubColVal;
	FLMBYTE			ucCaseBits = 0;
	FLMBYTE			ucCharSet = (FLMBYTE)(ui16WpChar >> 8);
	FLMBYTE			ucCharVal = (FLMBYTE)(ui16WpChar & 0xFF);
	FLMUINT16		ui16Hankaku;
	FLMUINT			uiLoop;
	FLMUINT16		ui16ReturnValue = 1;

	ui16ColValue = ui16SubColVal = 0;

	// Kanji or above

	if( ucCharSet >= 0x2B)
	{
		// Puts 2 or above into high byte.

		ui16ColValue = ui16WpChar - 0x2900;

		// No subcollation or case bits need to be set

		goto	Exit;
	}

	// Single wide character? (HANKAKU)

	if( ucCharSet < 11)
	{
		// Get the values from a non-asian character
		// LATIN, GREEK or CYRILLIC
		// The width bit may have been set on a jump to
		// label from below.

Latin_Greek_Cyrillic:

		// YES: Pass FLM_US_LANG because this is what we want -
		// Prevents double character sorting.

		ui16ColValue = f_wpGetCollation( ui16WpChar, FLM_US_LANG);

		if (bUppercaseFlag || f_wpIsUpper( ui16WpChar))
		{
			// Uppercase - set case bit

			ucCaseBits |= SET_CASE_BIT;
		}

		// Character for which there is no collation value?

		if( ui16ColValue == COLS0)
		{
			ui16ReturnValue = 0;
			if( !f_wpIsUpper( ui16WpChar))
			{
				// Convert to uppercase

				ui16WpChar--;
			}
			ui16ColValue = 0xFFFF;
			ui16SubColVal = ui16WpChar;
		}
		else if( ucCharSet) 				// Don't bother with ascii
		{
			if( !f_wpIsUpper( ui16WpChar))
			{
				// Convert to uppercase

				ui16WpChar--;
			}

        	if( ucCharSet == F_CHSMUL1)
			{
				FLMUINT16	ui16Base;
				FLMUINT16	ui16Diacritic;

				ui16SubColVal = !f_breakWPChar( ui16WpChar, &ui16Base,
															&ui16Diacritic)
									  ? fwp_dia60Tbl[ ui16Diacritic & 0xFF]
									  : ui16WpChar;
			}
			else if( ucCharSet == F_CHSGREK)
         {
         	if( ui16WpChar >= 0x834 ||		// [8,52] or above
            	 ui16WpChar == 0x804 ||		// [8,4] BETA Medial | Terminal
					 ui16WpChar == 0x826)		// [8,38] SIGMA terminal
				{
					ui16SubColVal = ui16WpChar;
				}
			}
			else if( ucCharSet == F_CHSCYR)
			{
           	if( ui16WpChar >= 0xA90)		// [10, 144] or above
				{
              	ui16SubColVal = ui16WpChar;	// Dup collation values
				}
         }
         // else don't need a sub collation value
      }
		goto	Exit;
	}

	// Single wide Japanese character?

 	if( ucCharSet == 11)
	{
		FLMUINT16	ui16KanaChar;

		// Convert charset 11 to Zenkaku (double wide) CS24 or CS26 hex.
		// All characters in charset 11 will convert to CS24 or CS26.
		// when combining the collation and the sub-collation values.

		if( f_wpHanToZenkaku( ui16WpChar,
			ui16NextWpChar, &ui16KanaChar ) == 2)
		{
			// Return 2

			ui16ReturnValue++;
		}

		ucCaseBits |= SET_WIDTH_BIT;	// Set so will allow to go back
		ui16WpChar = ui16KanaChar;		// If in CS24 will fall through to ZenKaku
		ucCharSet = (FLMBYTE)(ui16KanaChar >> 8);
		ucCharVal = (FLMBYTE)(ui16KanaChar & 0xFF);
	}

	if( ui16WpChar < 0x2400)
	{
		// In some other character set

		goto Latin_Greek_Cyrillic;
	}
	else if( ui16WpChar >= 0x255e &&	// Hiragana?
				ui16WpChar <= 0x2655)	// Katakana?
	{
		if( ui16WpChar >= 0x2600)
		{
			ucCaseBits |= SET_KATAKANA_BIT;
		}

		// HIRAGANA & KATAKANA
		//		Kana contains both hiragana and katakana.
		//		The tables contain the same characters in same order

		if( ucCharSet == 0x25)
		{
			// Change value to be in character set 26

			ucCharVal -= 0x5E;
		}

		ui16ColValue = 0x0100 + KanaColTbl[ ucCharVal ];
		ui16SubColVal = KanaSubColTbl[ ucCharVal ];
		goto Exit;
	}

	// ZenKaku - means any double wide character
	// Hankaku - single wide character

	//		Inputs:	0x2400..2559	symbols..latin  - Zenkaku
	//					0x265B..2750	greek..cyrillic - Zenkaku

	//	SET_WIDTH_BIT may have been set if original char
	// was in 11 and got converted to CS24. [1,2,5,27(extendedVowel),53,54]
	// Original chars from CS11 will have some collation value that when
	// combined with the sub-collation value will format a character in
	// CS24.  The width bit will then convert back to CS11.

	if( (ui16Hankaku = f_wpZenToHankaku( ui16WpChar, NULL)) != 0)
	{
		if( (ui16Hankaku >> 8) != 11)			// if CharSet11 was a CS24 symbol
		{
			ui16WpChar = ui16Hankaku;			// May be CS24 symbol/latin/gk/cy
			ucCharSet = (FLMBYTE)(ui16WpChar >> 8);
			ucCharVal = (FLMBYTE)(ui16WpChar & 0xFF);
			ucCaseBits |= SET_WIDTH_BIT;		// Latin symbols double wide
			goto Latin_Greek_Cyrillic;
		}
	}

	// 0x2400..0x24bc Japanese symbols that cannot be converted to Hankaku.
	// All 6 original symbol chars from 11 will also be here.
	// First try to find a collation value of the symbol.
	// The sub-collation value will be the position in the CS24 table + 1.

	for( uiLoop = 0;
		  uiLoop < (sizeof( fwp_Ch24ColTbl) / sizeof( BYTE_WORD_TBL));
	  	  uiLoop++ )
	{
		if( ucCharVal == fwp_Ch24ColTbl[ uiLoop].ByteValue)
		{
			if( (ui16ColValue = fwp_Ch24ColTbl[ uiLoop].WordValue) < 0x100)
			{
				// Don't save for chuuten, dakuten, handakuten

				ui16SubColVal = (FLMUINT16)(uiLoop + 1);
			}
			break;
		}
	}

	if( !ui16ColValue)
	{
		// Now see if it's a repeat or repeat-vowel character

		if( (((ucCharVal >= 0x12) && (ucCharVal <= 0x15)) ||
			   (ucCharVal == 0x17) ||
			   (ucCharVal == 0x18)) &&
		  		((ui16PrevColValue >> 8) == 1))
		{
			ui16ColValue = ui16PrevColValue;

			// Store original WP character

			ui16SubColVal = ui16WpChar;
		}
		else if( (ucCharVal == 0x1B) &&						// repeat vowel?
					(ui16PrevColValue >= 0x100) &&
					(ui16PrevColValue < COLS_ASIAN_MARKS))	// Previous kana char?
		{
			ui16ColValue = 0x0100 + KanaColToVowel[ ui16PrevColValue & 0xFF ];

			// Store original WP character

			ui16SubColVal = ui16WpChar;
		}
		else
		{
			ui16ReturnValue = 0;
			ui16ColValue = 0xFFFF;			// No collation value
			ui16SubColVal = ui16WpChar;	// Never have changed if gets here
		}
	}

Exit:

	// Set return values

	*pui16ColValue = ui16ColValue;
	*pui16SubColVal = ui16SubColVal;
	*pucCaseBits = ucCaseBits;

	return( ui16ReturnValue);
}

/****************************************************************************
Desc:		Convert a zenkaku (double wide) char to a hankaku (single wide) char
Ret:		Hankaku char or 0 if a conversion doesn't exist
Notes:	Taken from CHAR.ASM -  zen2han_f routine
****************************************************************************/
FLMUINT16 FTKAPI f_wpZenToHankaku(
	FLMUINT16	ui16WpChar,
	FLMUINT16 * pui16DakutenOrHandakuten)
{
	FLMUINT16	ui16Hankaku = 0;
	FLMBYTE		ucCharSet = (FLMBYTE)(ui16WpChar >> 8);
	FLMBYTE		ucCharVal = (FLMBYTE)(ui16WpChar & 0xFF);
	FLMUINT		uiLoop;

	switch( ucCharSet)
	{
		// SYMBOLS

		case 0x24:
		{
			for( uiLoop = 0;
				  uiLoop < (sizeof( Zen24ToHankaku) / sizeof( BYTE_WORD_TBL));
				  uiLoop++)
			{
				// List is sorted so table entry is more you are done

				if( Zen24ToHankaku[ uiLoop].ByteValue >= ucCharVal)
				{
					if( Zen24ToHankaku[ uiLoop].ByteValue == ucCharVal)
					{
						ui16Hankaku = Zen24ToHankaku[ uiLoop].WordValue;
					}

					break;
				}
			}
			break;
		}

		// ROMAN - 0x250F..2559
		// Hiragana - 0x255E..2580

		case 0x25:
		{
			if( ucCharVal >= 0x0F && ucCharVal < 0x5E)
			{
				ui16Hankaku = ucCharVal + 0x21;
			}
			break;
		}

		// Katakana - 0x2600..2655
		// Greek - 0x265B..2695

		case 0x26:
		{
			if( ucCharVal <= 0x55)		// Katakana range
			{
				FLMBYTE		ucCS11CharVal;
				FLMUINT16	ui16NextWpChar = 0;

				if( (ucCS11CharVal = MapCS26ToCharSet11[ ucCharVal ]) != 0xFF)
				{
					if( ucCS11CharVal & 0x80)
					{
						if( ucCS11CharVal & 0x40)
						{
							// Handakuten voicing

							ui16NextWpChar = 0xB3E;
						}
						else
						{
							// Dakuten voicing

							ui16NextWpChar = 0xB3D;
						}
						ucCS11CharVal &= 0x3F;
					}
					ui16Hankaku = 0x0b00 + ucCS11CharVal;
					if( ui16NextWpChar && pui16DakutenOrHandakuten)
					{
						*pui16DakutenOrHandakuten = ui16NextWpChar;
					}
				}
			}
			else if( ucCharVal <= 0x95)	// Greek
			{
				FLMBYTE	ucGreekChar = ucCharVal;

				// Make a zero based number.

				ucGreekChar -= 0x5E;

				// Check for lowercase
				if( ucGreekChar >= 0x20)
				{
					// Convert to upper case for now

					ucGreekChar -= 0x20;
				}

				if( ucGreekChar >= 2)
				{
					ucGreekChar++;
				}

				if (ucGreekChar >= 19)
				{
					ucGreekChar++;
				}

				// Convert to character set 8

				ui16Hankaku = (ucGreekChar << 1) + 0x800;
				if( ucCharVal >= (0x5E + 0x20))
				{
					// Adjust to lower case character

					ui16Hankaku++;
				}
			}
			break;
		}

		// Cyrillic

		case 0x27:
		{
			// Uppercase?

			if( ucCharVal <= 0x20)
			{
				ui16Hankaku = (ucCharVal << 1) + 0xa00;
			}
			else if( ucCharVal >= 0x30 && ucCharVal <= 0x50)
			{
				// Lower case

				ui16Hankaku = ((ucCharVal - 0x30) << 1) + 0xa01;
			}
			break;
		}
	}

	return( ui16Hankaku);
}

/****************************************************************************
Desc:		Convert a WPChar from hankaku (single wide) to zenkaku (double wide).
			1) Used to see if a char in CS11 can map to a double wide character
			2) Used to convert keys into original data.
Ret:		0 = no conversion
			1 = converted character to zenkaku
			2 = ui16NextWpChar dakuten or handakuten voicing got combined
Notes:	Taken from char.asm - han2zen()
			From8ToZen could be taken out and placed in code.
****************************************************************************/
FLMUINT16 FTKAPI f_wpHanToZenkaku(
	FLMUINT16	ui16WpChar,
	FLMUINT16	ui16NextWpChar,
	FLMUINT16 *	pui16Zenkaku)
{
	FLMUINT16	ui16Zenkaku = 0;
	FLMBYTE		ucCharSet = (FLMBYTE)(ui16WpChar >> 8);
	FLMBYTE		ucCharVal = (FLMBYTE)(ui16WpChar & 0xFF);
	FLMUINT		uiLoop;
	FLMUINT16	ui16CharsUsed = 1;

	switch( ucCharSet)
	{
		// Character set 0 - symbols

		case 0:
		{
			// Invalid? - all others are used.

			if( ucCharVal < 0x20)
			{
				;
			}
			else if( ucCharVal <= 0x2F)
			{
				// Symbols A
				ui16Zenkaku = 0x2400 + From0AToZen[ ucCharVal - 0x20 ];
			}
			else if( ucCharVal <= 0x39)
			{
				// 0..9
				ui16Zenkaku = 0x2500 + (ucCharVal - 0x21);
			}
			else if( ucCharVal <= 0x40)
			{
				// Symbols B
				ui16Zenkaku = 0x2400 + From0BToZen[ ucCharVal - 0x3A ];
			}
			else if( ucCharVal <= 0x5A)
			{
				// A..Z
				ui16Zenkaku = 0x2500 + (ucCharVal - 0x21);
			}
			else if( ucCharVal <= 0x60)
			{
				// Symbols C
				ui16Zenkaku = 0x2400 + From0CToZen[ ucCharVal - 0x5B ];
			}
			else if( ucCharVal <= 0x7A)
			{
				// a..z
				ui16Zenkaku = 0x2500 + (ucCharVal - 0x21);
			}
			else if( ucCharVal <= 0x7E)
			{
				// Symbols D
				ui16Zenkaku = 0x2400 + From0DToZen[ ucCharVal - 0x7B ];
			}
			break;
		}

		// GREEK

		case 8:
		{
			if( (ucCharVal >= sizeof( From8ToZen)) ||
				 ((ui16Zenkaku = 0x2600 + From8ToZen[ ucCharVal ]) == 0x26FF))
			{
				ui16Zenkaku = 0;
			}
			break;
		}

		// CYRILLIC

		case 10:
		{
			// Check range

			ui16Zenkaku = 0x2700 + (ucCharVal >> 1);	// Uppercase value

			// Convert to lower case?

			if( ucCharVal & 0x01)
			{
				ui16Zenkaku += 0x30;
			}
			break;
		}

		// JAPANESE

		case 11:
		{
			if( ucCharVal < 5)
			{
				ui16Zenkaku = 0x2400 + From11AToZen[ ucCharVal];
			}
			else if( ucCharVal < 0x3D)		// katakana?
			{
				if( (ui16Zenkaku = 0x2600 +
							From11BToZen[ ucCharVal - 5 ]) == 0x26FF)
				{
					// Dash - convert to this
					ui16Zenkaku = 0x241b;
				}
				else
				{
					if( ui16NextWpChar == 0xB3D)
					{
						// First check exception(s) then
						// check if voicing exists! - will NOT access out of table

						if( (ui16Zenkaku != 0x2652) &&	// is not 'N'?
							 (KanaSubColTbl[ ui16Zenkaku - 0x2600 + 1 ] == 3))
						{
							ui16Zenkaku++;

							// Return 2

							ui16CharsUsed++;
						}
					}
					else if( ui16NextWpChar == 0xB3E)	// handakuten? - voicing
					{
						// Check if voicing exists! - will NOT access out of table

						if( KanaSubColTbl [ui16Zenkaku - 0x2600 + 2 ] == 5)
						{
							ui16Zenkaku += 2;

							// Return 2

							ui16CharsUsed++;
						}
					}
				}
			}
			else if( ucCharVal == 0x3D)		// dakuten?
			{
				// Convert to voicing symbol

				ui16Zenkaku = 0x240A;
			}
			else if( ucCharVal == 0x3E)		// handakuten?
			{
				// Convert to voicing symbol

				ui16Zenkaku = 0x240B;
			}
			// else cannot convert

			break;
		}

		// Other character sets
		// CS 1,4,5,6 - symbols

		default:
		{
			// Look in the Zen24Tohankaku table for a matching value

			for( uiLoop = 0;
				  uiLoop < (sizeof( Zen24ToHankaku) / sizeof( BYTE_WORD_TBL));
				  uiLoop++)
			{
				if( Zen24ToHankaku[ uiLoop].WordValue == ui16WpChar)
				{
					ui16Zenkaku = 0x2400 + Zen24ToHankaku[ uiLoop].ByteValue;
					break;
				}
			}
			break;
		}
	}

	if( !ui16Zenkaku)
	{
		// Change return value

		ui16CharsUsed = 0;
	}

	*pui16Zenkaku = ui16Zenkaku;
	return( ui16CharsUsed);
}

/****************************************************************************
Desc:		Converts a 2-byte language code into its corresponding language ID
****************************************************************************/
FLMUINT FTKAPI f_languageToNum(
	const char *	pszLanguage)
{
	FLMBYTE		ucFirstChar  = (FLMBYTE)(*pszLanguage);
	FLMBYTE		ucSecondChar = (FLMBYTE)(*(pszLanguage + 1));
	FLMUINT		uiTablePos;

	for( uiTablePos = 0; 
		uiTablePos < (FLM_LAST_LANG + FLM_LAST_LANG); uiTablePos += 2)
	{
		if( f_langtbl [uiTablePos]   == ucFirstChar &&
			 f_langtbl [uiTablePos+1] == ucSecondChar)
		{
			return( uiTablePos >> 1);
		}
	}

	// Language not found, return default US language

	return( FLM_US_LANG);
}

/****************************************************************************
Desc:		Converts a language ID to its corresponding 2-byte language code
****************************************************************************/
void FTKAPI f_languageToStr(
	FLMINT		iLangNum,
	char *		pszLanguage)
{
	// iLangNum could be negative

	if( iLangNum < 0 || iLangNum >= FLM_LAST_LANG)
	{
		iLangNum = FLM_US_LANG;
	}

	iLangNum += iLangNum;
	*pszLanguage++ = (char)f_langtbl [iLangNum ];
	*pszLanguage++ = (char)f_langtbl [iLangNum+1];
	*pszLanguage = 0;
}

/***************************************************************************
Desc:	Return the sub-collation value of a WP character.  Unconverted
		unicode values always have a sub-collation value of
		11110+UnicodeChar
***************************************************************************/
FLMUINT16 flmWPGetSubCol(
	FLMUINT16		ui16WPValue,		// [in] WP Character value.
	FLMUINT16		ui16ColValue,		// [in] Collation Value (for arabic)
	FLMUINT			uiLanguage)			// [in] WP Language ID.
{
	FLMUINT16		ui16SubColVal;
	FLMBYTE			ucCharVal;
	FLMBYTE			ucCharSet;
	FLMUINT16		ui16Base;

	// Easy case first - ascii characters.

	ui16SubColVal = 0;
	if (ui16WPValue <= 127)
	{
		goto Exit;
	}

	// From here down default ui16SubColVal is WP value.

	ui16SubColVal = ui16WPValue;
	ucCharVal = (FLMBYTE) ui16WPValue;
	ucCharSet = (FLMBYTE) (ui16WPValue >> 8);

	// Convert char to uppercase because case information
	// is stored above.  This will help
	// ensure that the "ETA" doesn't sort before "eta"
	// could use is lower code here for added performance.

	// This just happens to work with all WP character values.

	if (!f_wpIsUpper( ui16WPValue))
	{
		ui16WPValue &= ~1;
	}

	switch (ucCharSet)
	{
		case F_CHSMUL1:
		{
			// If you cannot break down a char into base and
			// diacritic then you cannot combine the charaacter
			// later when converting back the key.  So, write
			// the entire WP char in the sub-collation area.
			// We can ONLY SUPPORT MULTINATIONAL 1 for brkcar()

			if (f_breakWPChar( ui16WPValue, &ui16Base, &ui16SubColVal))
			{

				// WordPerfect character cannot be broken down.
				// If we had a collation value other than 0xFF (COLS0), don't
				// return a sub-collation value.  This will allow things like
				// upper and lower AE digraphs to compare properly.

				if (ui16ColValue != COLS0)
				{
					ui16SubColVal = 0;
				}
				goto Exit;
			}

			// Write the FLAIM diacritic sub-collation value.
			// Prefix is 2 bits "10".  Remember to leave
			// "111" alone for the future.
			// Bug 11/16/92 = was only writing a "1" and not "10"

			ui16SubColVal = (
					(ui16SubColVal & 0xFF) == F_UMLAUT
					&& ( (uiLanguage == FLM_SU_LANG) ||
						  (uiLanguage == FLM_SV_LANG) ||
						  (uiLanguage == FLM_CZ_LANG) ||
						  (uiLanguage == FLM_SL_LANG)
						)
					)
				?	(FLMUINT16)(fwp_dia60Tbl[ F_RING] + 1)	// umlaut must be after ring above
				:	(FLMUINT16)(fwp_dia60Tbl[ ui16SubColVal & 0xFF]);

			break;
		}

		case F_CHSGREK:
		{
			if( (ucCharVal >= 52)  ||		// Keep case bit for 52-69 else ignore
          	 (ui16WPValue == 0x804) ||	// [ 8,4] BETA Medial | Terminal
				 (ui16WPValue == 0x826)) 	// [ 8,38] SIGMA termainal
			{
				ui16SubColVal = ui16WPValue;
			}
			// else no subcollation to worry about
			break;
		}

		case F_CHSCYR:
		{
			if (ucCharVal >= 144)
			{
				ui16SubColVal = ui16WPValue;
			}

			break;
		}

		case F_CHSHEB:
		{

			// Three sections in Hebrew:
			//		0..26 - main characters
			//		27..83 - accents that apear over previous character
			//		84..118- dagesh (ancient) hebrew with accents

			// Because the ancient is only used for sayings & scriptures
			// we will support a collation value and in the sub-collation
			// store the actual character because sub-collation is in
			// character order.

         if (ucCharVal >= 84) 		// Save ancient - value 84 and above
			{
				ui16SubColVal = ui16WPValue;
			}
			break;
		}

		case F_CHSARB1:	// Arabic 1
		{

			// Three sections in Arabic:
			//		00..37  - accents that display OVER a previous character
			//		38..46  - symbols
			//		47..57  - numbers
			//		58..163 - characters
			//		164     - hamzah accent
			//		165..180- common characters with accents
			//		181..193- ligatures - common character combinations
			//		194..195- extensions - throw away when sorting

			if (ucCharVal <= 46)
			{
				ui16SubColVal = ui16WPValue;
			}
			else
			{
				if (ui16ColValue == COLS10a+1)	// Alef?
				{
					ui16SubColVal = (ucCharVal >= 165)
						? (FLMUINT16)(fwp_alefSubColTbl[ ucCharVal - 165 ])
						: (FLMUINT16)7;		// Alef subcol value
				}
				else
				{
					if (ucCharVal >= 181)		// Ligatures - char combination
					{
						ui16SubColVal = ui16WPValue;
					}
					else if (ucCharVal == 64)	// taa exception
					{
						ui16SubColVal = 8;
					}
				}
			}
			break;
		}

		case F_CHSARB2:			// Arabic 2
		{

			// There are some characters that share the same slot
			// Check the bit table if above character 64

			if ((ucCharVal >= 64) &&
				 (fwp_ar2BitTbl[(ucCharVal-64)>> 3] & (0x80 >> (ucCharVal&0x07))))
			{
				ui16SubColVal = ui16WPValue;
			}
			break;
		}
	}

Exit:

	return( ui16SubColVal);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CollIStream::read(
	FLMBOOL			bAllowTwoIntoOne,
	FLMUNICODE *	puChar,
	FLMBOOL *		pbCharIsWild,
	FLMUINT16 *		pui16Col,
	FLMUINT16 *		pui16SubCol,
	FLMBYTE *		pucCase)
{
	RCODE					rc = NE_FLM_OK;
	FLMUNICODE			uChar;
	FLMUINT16			ui16WpChar;
	FLMUINT16			ui16NextWpChar;
	FLMUINT16			ui16Col;
	FLMUINT16			ui16SubCol;
	FLMBOOL				bTwoIntoOne;
	FLMBYTE				ucCase;
	FLMBOOL				bAsian;
	FLMBOOL				bLastCharWasSpace = FALSE;
	FLMUINT64			ui64AfterLastSpacePos = 0;
	FLMUINT64			ui64CurrCharPos = 0;

	if (pbCharIsWild)
	{
		*pbCharIsWild = FALSE;
	}

	// Is this a double-byte (Asian) character set?

	bAsian = (m_uiLanguage >= FLM_FIRST_DBCS_LANG && 
					m_uiLanguage <= FLM_LAST_DBCS_LANG)
							? TRUE
							: FALSE;

	// Get the next character from the stream

GetNextChar:

	ui16WpChar = 0;
	ui16NextWpChar = 0;
	ui16Col = 0;
	ui16SubCol = 0;
	bTwoIntoOne = FALSE;
	ucCase = 0;

	if (m_uNextChar)
	{
		uChar = m_uNextChar;
		m_uNextChar = 0;
	}
	else
	{
		ui64CurrCharPos = m_pIStream->getCurrPosition();
		if( RC_BAD( rc = readCharFromStream( &uChar)))
		{
			if (rc != NE_FLM_EOF_HIT)
			{
				goto Exit;
			}
			
			// If we were skipping spaces, we need to
			// process a single space character, unless we are
			// ignoring trailing white space.
			
			if (bLastCharWasSpace &&
				 !(m_uiCompareRules & FLM_COMP_IGNORE_TRAILING_SPACE))
			{
				// bLastCharWasSpace flag can only be TRUE if either
				// FLM_COMP_IGNORE_TRAILING_SPACE is set or
				// FLM_COMP_COMPRESS_WHITESPACE is set.
				
				flmAssert( m_uiCompareRules & FLM_COMP_COMPRESS_WHITESPACE);
				uChar = ASCII_SPACE;
				rc = NE_FLM_OK;
				goto Process_Char;
			}
			goto Exit;
		}
	}

	if ((uChar = f_convertChar( uChar, m_uiCompareRules)) == 0)
	{
		goto GetNextChar;
	}

	// Deal with spaces

	if (uChar == ASCII_SPACE)
	{
		if (m_uiCompareRules & FLM_COMP_COMPRESS_WHITESPACE)
		{
			bLastCharWasSpace = TRUE;
			ui64AfterLastSpacePos = m_pIStream->getCurrPosition();
			goto GetNextChar;
		}
		else if (m_uiCompareRules & FLM_COMP_IGNORE_TRAILING_SPACE)
		{
			if (!bLastCharWasSpace)
			{
				bLastCharWasSpace = TRUE;
				
				// Save where we are at so that if this doesn't turn out
				// to be trailing spaces, we can restore this position.
				
				ui64AfterLastSpacePos = m_pIStream->getCurrPosition();
			}
			
			goto GetNextChar;
		}
	}
	else
	{
		if (m_uiCompareRules & FLM_COMP_IGNORE_LEADING_SPACE)
		{
			m_ui64EndOfLeadingSpacesPos = ui64CurrCharPos;
			m_uiCompareRules &= (~(FLM_COMP_IGNORE_LEADING_SPACE));
		}
		
		// If the last character was a space, we need to process it.
		
		if (bLastCharWasSpace)
		{
			
			// Position back to after the last space, and process a space
			// character.
			
			if (RC_BAD( rc = m_pIStream->positionTo( ui64AfterLastSpacePos)))
			{
				goto Exit;
			}
			
			uChar = ASCII_SPACE;
			bLastCharWasSpace = FALSE;
		}
		else if (uChar == ASCII_BACKSLASH)
		{
			// If wildcards are allowed, the backslash should be treated
			// as an escape character, and the next character is the one
			// we want.  Otherwise, it should be treated as
			// the actual character we want returned.
			
			if (m_bMayHaveWildCards)
			{
			
				// Got a backslash.  Means the next character is to be taken
				// no matter what because it is escaped.
			
				if (RC_BAD( rc = readCharFromStream( &uChar)))
				{
					if (rc != NE_FLM_EOF_HIT)
					{
						goto Exit;
					}
					rc = NE_FLM_OK;
					uChar = ASCII_BACKSLASH;
				}
			}
		}
		else if (uChar == ASCII_WILDCARD)
		{
			if (m_bMayHaveWildCards && pbCharIsWild)
			{
				*pbCharIsWild = TRUE;
			}
		}
	}
	
Process_Char:

	if (!bAsian)
	{
		
		// Must check for double characters if non-US and non-Asian
		// character set

		if (m_uiLanguage != FLM_US_LANG)
		{
			if (RC_BAD( rc = f_wpCheckDoubleCollation( 
				m_pIStream, m_bUnicodeStream, bAllowTwoIntoOne, 
				&uChar, &m_uNextChar, &bTwoIntoOne, m_uiLanguage)))
			{
				goto Exit;
			}
		}
	}
	else
	{
		if (RC_BAD( rc = readCharFromStream( &m_uNextChar)))
		{
			if (rc == NE_FLM_EOF_HIT)
			{
				rc = NE_FLM_OK;
				m_uNextChar = 0;
			}
			else
			{
				RC_UNEXPECTED_ASSERT( rc);
				goto Exit;
			}
		}
	}

	// Convert each character to its WP equivalent

	if (!f_unicodeToWP( uChar, &ui16WpChar))
	{
		ui16WpChar = 0;
	}

	if (!f_unicodeToWP( m_uNextChar, &ui16NextWpChar))
	{
		ui16NextWpChar = 0;
	}

	// If we have an unconvertible UNICODE character, the collation
	// value for it will be COLS0

	if (!ui16WpChar)
	{
		if (!bAsian)
		{
			ui16Col = COLS0;
		}
		else
		{
			if (uChar < 0x20)
			{
				ui16Col = 0xFFFF;
				ui16SubCol = uChar;
			}
			else
			{
				ui16Col = uChar;
				ui16SubCol = 0;
			}
		}
	}
	else
	{
		if (!bAsian)
		{
			ui16Col = f_wpGetCollation( ui16WpChar, m_uiLanguage);
			if (bTwoIntoOne)
			{
				// Since two characters were merged into one, increment
				// the collation value by one.  In the case of something
				// like 'ch', there is a collation value between 'c' and
				// 'd'.  f_wpGetCollation would have returned the
				// collation value for 'c' ... incrementing by one gives
				// us the proper collation value for 'ch' (i.e., the
				// collation value between 'c' and 'd').

				ui16Col++;
			}
		}
		else
		{
			if (flmWPAsiaGetCollation( ui16WpChar, ui16NextWpChar, ui16Col,
					&ui16Col, &ui16SubCol, &ucCase, !m_bCaseSensitive) == 2)
			{
				
				// Next character was consumed by collation

				m_uNextChar = 0;
			}
		}
	}

	if (pui16Col)
	{
		*pui16Col = ui16Col;
	}

	// Consume m_uNextChar if two characters merged into one

	if (bTwoIntoOne)
	{
		m_uNextChar = 0;
	}
	
	// Subcollation

	if( pui16SubCol)
	{
		if( uChar > 127 && !bAsian)
		{
			ui16SubCol = ui16WpChar
							  ? flmWPGetSubCol( ui16WpChar, ui16Col, m_uiLanguage)
							  : uChar;

			if( !m_bCaseSensitive)
			{
				// If the sub-collation value is the original
				// character, it means that the collation could not
				// distinguish the characters and sub-collation is being
				// used to do it.  However, this creates a problem when the
				// characters are the same character except for case.  In that
				// scenario, we incorrectly return a not-equal when we are
				// doing a case-insensitive comparison.  So, at this point,
				// we need to use the sub-collation for the upper-case of the
				// character instead of the sub-collation for the character
				// itself.

				if( ui16WpChar && ui16SubCol == ui16WpChar)
				{
					ui16SubCol = flmWPGetSubCol(
												f_wpUpper( ui16WpChar),
												ui16Col, m_uiLanguage);
				}
			}
		}

		*pui16SubCol = ui16SubCol;
	}

	// Case

	if( pucCase)
	{
		if (!m_bCaseSensitive)
		{
			*pucCase = 0;
		}
		else
		{
			if (!bAsian && ui16WpChar)
			{
				// f_wpIsUpper() returns FALSE if the character is lower or
				// TRUE if the character is not lower case.
	
				if( f_wpIsUpper( ui16WpChar))
				{
					if( bTwoIntoOne)
					{
						if( f_wpIsUpper( ui16NextWpChar))
						{
							ucCase = 0x03;
						}
						else
						{
							ucCase = 0x10;
						}
					}
					else
					{
						ucCase = 0x01;
					}
				}
			}
			*pucCase = ucCase;
		}
	}

	if (puChar)
	{
		*puChar = uChar;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:  	Compare two entire strings.
****************************************************************************/
RCODE FTKAPI f_compareCollStreams(
	IF_CollIStream *	pLStream,
	IF_CollIStream *	pRStream,
	FLMBOOL				bOpIsMatch,
	FLMUINT				uiLanguage,
	FLMINT *				piResult)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT16			ui16RCol;
	FLMUINT16			ui16LCol;
	FLMUINT16			ui16RSubCol;
	FLMUINT16			ui16LSubCol;
	FLMBYTE				ucRCase;
	FLMBYTE				ucLCase;
	F_CollStreamPos	savedRPos;
	F_CollStreamPos	savedLPos;
	F_CollStreamPos	startLPos;
	FLMUNICODE			uLChar = 0;
	FLMBOOL				bLCharIsWild = FALSE;
	FLMUNICODE			uRChar = 0;
	FLMBOOL				bRCharIsWild = FALSE;
	FLMBOOL				bPrevLWasWild = FALSE;
	FLMBOOL				bPrevRWasWild = FALSE;
	FLMBOOL				bAllowTwoIntoOne;

	// If we are doing a "match" operation, we don't want two
	// character sequences like Ch, ae, etc. turned into a single
	// a single collation, because then matches that involve wildcards
	// like "aetna == a*" would not match properly.
	// When not doing a match operation, we WANT two character sequences
	// turned into a single collation value so that we can know if
	// something is > or <.  When doing match operations, all we care
	// about is if they are equal or not, so there is no need to look
	// at double character collation properties.

	bAllowTwoIntoOne = bOpIsMatch ? FALSE : TRUE;

	for( ;;)
	{
GetNextLChar:

		if( bLCharIsWild)
		{
			bPrevLWasWild = TRUE;
		}

		pLStream->getCurrPosition( &startLPos);
		if( RC_BAD( rc = pLStream->read( 
			bAllowTwoIntoOne,
			&uLChar, &bLCharIsWild, &ui16LCol, &ui16LSubCol, &ucLCase)))
		{
			if( rc == NE_FLM_EOF_HIT)
			{
				rc = NE_FLM_OK;

				// If the last character was a wildcard, we have a match!

				if( bPrevLWasWild)
				{
					*piResult = 0;
					goto Exit;
				}

				for( ;;)
				{
					if( RC_BAD( rc = pRStream->read( 
						bAllowTwoIntoOne,
						&uRChar, &bRCharIsWild, &ui16RCol, &ui16RSubCol, &ucRCase)))
					{
						if( rc == NE_FLM_EOF_HIT)
						{
							rc = NE_FLM_OK;
							*piResult = 0;
						}

						goto Exit;
					}

					// Break out when we hit a non-wild character

					if( !bRCharIsWild)
					{
						break;
					}
				}

				*piResult = -1;
			}

			goto Exit;
		}

		if( bLCharIsWild)
		{
			// Consume multiple wildcards

			if( bPrevLWasWild)
			{
				goto GetNextLChar;
			}

			// See if we match anywhere on the remaining right string

			for( ;;)
			{
				pRStream->getCurrPosition( &savedRPos);
				pLStream->getCurrPosition( &savedLPos);

				if( RC_BAD( rc = f_compareCollStreams( pLStream, pRStream,
					bOpIsMatch, uiLanguage, piResult)))
				{
					goto Exit;
				}

				if( !(*piResult))
				{
					goto Exit;
				}

				if( RC_BAD( rc = pRStream->positionTo( &savedRPos)))
				{
					goto Exit;
				}

				if( RC_BAD( rc = pRStream->read( 
					bAllowTwoIntoOne, 
					NULL, NULL, NULL, NULL, NULL)))
				{
					if( rc == NE_FLM_EOF_HIT)
					{
						rc = NE_FLM_OK;
						break;
					}
					goto Exit;
				}

				if( RC_BAD( rc = pLStream->positionTo( &savedLPos)))
				{
					goto Exit;
				}
			}

			*piResult = 1;
			goto Exit;
		}

GetNextRChar:

		if( bRCharIsWild)
		{
			bPrevRWasWild = TRUE;
		}

		if( RC_BAD( rc = pRStream->read( 
			bAllowTwoIntoOne, 
			&uRChar, &bRCharIsWild, &ui16RCol, &ui16RSubCol, &ucRCase)))
		{
			if( rc == NE_FLM_EOF_HIT)
			{
				rc = NE_FLM_OK;

				// If the last character was a wildcard, we have a match!

				if( bPrevRWasWild)
				{
					*piResult = 0;
				}
				else
				{
					*piResult = 1;
				}
			}

			goto Exit;
		}

		if( bRCharIsWild)
		{
			if( bPrevRWasWild)
			{
				goto GetNextRChar;
			}

			// See if we match anywhere on the remaining left string

			if( RC_BAD( rc = pLStream->positionTo( &startLPos)))
			{
				goto Exit;
			}

			for( ;;)
			{
				pLStream->getCurrPosition( &savedLPos);
				pRStream->getCurrPosition( &savedRPos);

				if( RC_BAD( rc = f_compareCollStreams( pLStream, pRStream,
					bOpIsMatch, uiLanguage, piResult)))
				{
					goto Exit;
				}

				if( !(*piResult))
				{
					goto Exit;
				}

				if( RC_BAD( rc = pRStream->positionTo( &savedRPos)))
				{
					goto Exit;
				}

				if( RC_BAD( rc = pLStream->positionTo( &savedLPos)))
				{
					goto Exit;
				}

				// Skip the character we just processed

				if( RC_BAD( rc = pLStream->read( 
					bAllowTwoIntoOne,
					NULL, NULL, NULL, NULL, NULL)))
				{
					if( rc == NE_FLM_EOF_HIT)
					{
						rc = NE_FLM_OK;
						break;
					}
					goto Exit;
				}
			}

			*piResult = -1;
			goto Exit;
		}

		if( ui16LCol != ui16RCol)
		{
			*piResult = ui16LCol < ui16RCol ? -1 : 1;
			goto Exit;
		}
		else if( ui16LSubCol != ui16RSubCol)
		{
			*piResult = ui16LSubCol < ui16RSubCol ? -1 : 1;
			goto Exit;
		}
		else if( ucLCase != ucRCase) 
		{
			// NOTE: If we are doing a case insensitive comparison,
			// ucLCase and ucRCase should be equal (both will have been
			// set to zero
			
			*piResult = ucLCase < ucRCase ? -1 : 1;
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
FLMUNICODE FTKAPI f_convertChar(
	FLMUNICODE		uzChar,
	FLMUINT			uiCompareRules)
{
	if (uzChar == ASCII_SPACE ||
		 (uzChar == ASCII_UNDERSCORE &&
		  (uiCompareRules & FLM_COMP_NO_UNDERSCORES)) ||
		 (f_isWhitespace( uzChar) &&
		  (uiCompareRules & FLM_COMP_WHITESPACE_AS_SPACE)))
	{
		return( (FLMUNICODE)((uiCompareRules &
									 (FLM_COMP_NO_WHITESPACE |
									  FLM_COMP_IGNORE_LEADING_SPACE))
									 ? (FLMUNICODE)0
									 : (FLMUNICODE)ASCII_SPACE));
	}
	else if (uzChar == ASCII_DASH && (uiCompareRules & FLM_COMP_NO_DASHES))
	{
		return( (FLMUNICODE)0);
	}
	else
	{
		return( uzChar);
	}
}


/****************************************************************************
Desc:		Called by ftkStartup, this routine initializes the Unicode to
			WP and WP to Unicode mapping tables.
****************************************************************************/
RCODE f_initCharMappingTables( void)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT16 *		puStaticPtr;
	FLMUINT			uiLoop;
	FLMUINT			uiEntries;
	FLMUINT			uiOffset;

	if( gv_pUnicodeToWP60 || gv_pWP60ToUnicode || gv_pui16USCollationTable)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_FAILURE);
		goto Exit;
	}

	gv_uiMinUniChar = 0;
	gv_uiMaxUniChar = 0;

	gv_uiMinWPChar = 0;
	gv_uiMaxWPChar = 0;

	// Make an initial pass over the table to determine
	// what our allocation sizes will need to be.

	for( uiLoop = 0, puStaticPtr = (FLMUINT16 *)WP_UTOWP60;
		uiLoop < UTOWP60_ENTRIES;
		uiLoop++, puStaticPtr += 2)
	{
		// Unicode

		if( (FLMUINT)puStaticPtr[ 0] < gv_uiMinUniChar ||
			!gv_uiMinUniChar)
		{
			flmAssert( puStaticPtr[ 0] != 0);
			gv_uiMinUniChar = (FLMUINT)puStaticPtr[ 0];
		}

		if( (FLMUINT)puStaticPtr[ 0] > gv_uiMaxUniChar)
		{
			gv_uiMaxUniChar = (FLMUINT)puStaticPtr[ 0];
		}

		// WordPerfect

		if( (FLMUINT)puStaticPtr[ 1] < gv_uiMinWPChar ||
			!gv_uiMinWPChar)
		{
			flmAssert( puStaticPtr[ 1] != 0);
			gv_uiMinWPChar = (FLMUINT)puStaticPtr[ 1];
		}

		if( (FLMUINT)puStaticPtr[ 1] > gv_uiMaxWPChar)
		{
			gv_uiMaxWPChar = (FLMUINT)puStaticPtr[ 1];
		}
	}

	// Allocate the Unicode table

	uiEntries = (gv_uiMaxUniChar - gv_uiMinUniChar) + 1;
	if (RC_BAD( rc = f_calloc( uiEntries * sizeof( FLMUINT16),
		&gv_pUnicodeToWP60)))
	{
		goto Exit;
	}

	// Populate the Unicode table

	for( uiLoop = 0, puStaticPtr = (FLMUINT16 *)WP_UTOWP60;
		uiLoop < UTOWP60_ENTRIES; uiLoop++, puStaticPtr += 2)
	{
		uiOffset = (FLMUINT)puStaticPtr[ 0] - gv_uiMinUniChar;

		flmAssert( gv_pUnicodeToWP60[ uiOffset] == 0);
		gv_pUnicodeToWP60[ uiOffset] = puStaticPtr[ 1];
	}

	// Allocate the WordPerfect table

	uiEntries = (gv_uiMaxWPChar - gv_uiMinWPChar) + 1;
	if (RC_BAD( rc = f_calloc( uiEntries * sizeof( FLMUINT16),
		&gv_pWP60ToUnicode)))
	{
		goto Exit;
	}

	// Populate the WordPerfect table

	for( uiLoop = 0, puStaticPtr = (FLMUINT16 *)WP_UTOWP60;
		uiLoop < UTOWP60_ENTRIES; uiLoop++, puStaticPtr += 2)
	{
		uiOffset = (FLMUINT)puStaticPtr[ 1] - gv_uiMinWPChar;

		flmAssert( gv_pWP60ToUnicode[ uiOffset] == 0);
		gv_pWP60ToUnicode[ uiOffset] = puStaticPtr[ 0];
	}

	// Allocate the US collation mapping table

	uiEntries = 0x10000;
	if (RC_BAD( rc = f_calloc( uiEntries * sizeof( FLMUINT16),
		&gv_pui16USCollationTable)))
	{
		goto Exit;
	}

	// Populate the US collation mapping table

	for( uiLoop = 0; uiLoop < uiEntries; uiLoop++)
	{
		FLMBYTE			ucCharVal = (FLMBYTE)uiLoop;
		FLMBYTE			ucCharSet = (FLMBYTE)(uiLoop >> 8);
		TBL_B_TO_BP *	pColTbl = fwp_col60Tbl;

		do
		{
			if( pColTbl->key == ucCharSet)
			{
				FLMBYTE *	pucColVals = pColTbl->charPtr;

				// Check if the value is in the range of collated chars
				// Above lower range of table?

				if( ucCharVal >= *pucColVals)
				{
					// Make value zero based to index

					ucCharVal -= *pucColVals++;

					// Below maximum number of table entries?

					if( ucCharVal < *pucColVals++)
					{
						// Return collated value.

						gv_pui16USCollationTable[ uiLoop] = pucColVals[ ucCharVal];
						break;
					}
				}
			}

			// Go to next table entry

			pColTbl++;

		} while( pColTbl->key != 0xFF);

		if( pColTbl->key == 0xFF)
		{
			gv_pui16USCollationTable[ uiLoop] = COLS0;
		}
	}

Exit:

	if( RC_BAD( rc))
	{
		if( gv_pUnicodeToWP60)
		{
			f_free( &gv_pUnicodeToWP60);
		}

		if( gv_pWP60ToUnicode)
		{
			f_free( &gv_pWP60ToUnicode);
		}

		if( gv_pui16USCollationTable)
		{
			f_free( &gv_pui16USCollationTable);
		}

		gv_uiMinUniChar = 0;
		gv_uiMaxUniChar = 0;

		gv_uiMinWPChar = 0;
		gv_uiMaxWPChar = 0;
	}

	return( rc);
}

/****************************************************************************
Desc:		Called by ftkShutdown, this routine frees the Unicode to WP and
			WP to Unicode mapping tables.
****************************************************************************/
void f_freeCharMappingTables( void)
{
	if( gv_pUnicodeToWP60)
	{
		f_free( &gv_pUnicodeToWP60);
	}

	if( gv_pWP60ToUnicode)
	{
		f_free( &gv_pWP60ToUnicode);
	}

	if( gv_pui16USCollationTable)
	{
		f_free( &gv_pui16USCollationTable);
	}

	gv_uiMinUniChar = 0;
	gv_uiMaxUniChar = 0;

	gv_uiMinWPChar = 0;
	gv_uiMaxWPChar = 0;
}

/**************************************************************************
Desc: 	Convert the WP string to lower case chars given low/up bit string
Out:	 	WP characters that have been modified to their original case
Ret:		Number of bytes used in the lower/upper buffer
Notes:	Only WP to lower case conversion is done here for each bit NOT set.
***************************************************************************/
FLMUINT FTKAPI f_wpToMixed(
	FLMBYTE *			pucWPStr,			// Existing WP string to modify
	FLMUINT				uiWPStrLen,			// Length of the WP string in bytes
	const FLMBYTE *	pucLowUpBitStr,	// Lower/upper case bit string
	FLMUINT				uiLang)
{
	FLMUINT		uiNumChars;
	FLMUINT		uiTempWord;
	FLMBYTE		ucTempByte = 0;
	FLMBYTE		ucMaskByte;
	FLMBYTE		ucXorByte;	// Used to reverse GR, bits

	ucXorByte = (uiLang == FLM_US_LANG)	// Do most common compare first
						? (FLMBYTE)0
						: (uiLang == FLM_GR_LANG)	// Greek has uppercase first
							? (FLMBYTE)0xFF
							: (FLMBYTE)0 ;

	// For each character (two bytes) in the word string ...

	for( uiNumChars = uiWPStrLen >> 1, ucMaskByte = 0;
		  uiNumChars--;
		  pucWPStr += 2, ucMaskByte >>= 1)
	{
		if( ucMaskByte == 0)
		{
			// Time to get another byte

			ucTempByte = ucXorByte ^ *pucLowUpBitStr++;
			ucMaskByte = 0x80;
		}

		// If lowercase convert, else is upper

		if( (ucTempByte & ucMaskByte) == 0)
		{
			// Convert to lower case - COLL -> WP is already in upper case

			uiTempWord = (FLMUINT) FB2UW( pucWPStr);
			if( uiTempWord >= ASCII_UPPER_A && uiTempWord <= ASCII_UPPER_Z)
			{
				uiTempWord |= 0x20;
			}
			else
			{
				FLMBYTE ucCharVal = (FLMBYTE)( uiTempWord & 0xFF);
				FLMBYTE ucCharSet = (FLMBYTE)( uiTempWord >> 8);

				// Check if charact within region of character set

				if( ((ucCharSet == F_CHSMUL1) &&
						((ucCharVal >= 26) && (ucCharVal <= 241))) ||
					((ucCharSet == F_CHSGREK) && (ucCharVal <= 69)) ||
					((ucCharSet == F_CHSCYR) && (ucCharVal <= 199)))
				{
					uiTempWord |= 0x01;		// Set the bit ... don't increment!
				}
			}
			UW2FBA( (FLMUINT16)uiTempWord, pucWPStr);
		}
	}

	uiNumChars = uiWPStrLen >> 1;
	return( bytesInBits( uiNumChars));
}

/****************************************************************************
Desc:  	Convert a text string to a collated string.
			If NE_FLM_CONV_DEST_OVERFLOW is returned the string is truncated as
			best as it can be.  The caller must decide to return the error up
			or deal with the truncation.
VISIT:	If the string is EXACTLY the length of the truncation
			length then it should, but doesn't, set the truncation flag.
			The code didn't match the design intent.  Fix next major
			version.
****************************************************************************/
RCODE FTKAPI flmUTF8ToColText(
	IF_PosIStream *	pIStream,
	FLMBYTE *			pucCollatedStr,		// Returns collated string
	FLMUINT *			puiCollatedStrLen,	// Returns total collated string length
														// Input is maximum bytes in buffer
	FLMBOOL  			bCaseInsensitive,		// Set if to convert to uppercase
	FLMUINT *			puiCollationLen,		// Returns the collation bytes length
	FLMUINT *			puiCaseLen,				// Returns length of case bytes
	FLMUINT				uiLanguage,				// Language
	FLMUINT				uiCharLimit,			// Max number of characters in this key piece
	FLMBOOL				bFirstSubstring,		// TRUE is this is the first substring key
	FLMBOOL				bDataTruncated,		// TRUE if data is coming in truncated.
	FLMBOOL *			pbOriginalCharsLost,
	FLMBOOL *			pbDataTruncated)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT16	ui16Base;				// Value of the base character
	FLMUINT16	ui16SubColVal;			// Sub-collated value (diacritic)
	FLMUINT 		uiLength;				// Temporary variable for length
	FLMUINT 		uiTargetColLen = *puiCollatedStrLen - 8;	// 4=ovhd,4=worse char

	// Need to increase the buffer sizes to not overflow.
	// Characaters without COLL values will take up 3 bytes in
	// the ucSubColBuf[] and easily overflow the buffer.
	// Hard coded the values so as to minimize changes.

	FLMBYTE		ucSubColBuf[ MAX_SUBCOL_BUF + 301];	// Holds sub-collated values(diac)
	FLMBYTE		ucCaseBits[ MAX_CASE_BYTES + 81];	// Holds case bits
	FLMUINT16	ui16WpChr;			// Current WP character
	FLMUNICODE	uChar = 0;			// Current unconverted Unicode character
	FLMUNICODE	uChar2;
	FLMUINT16	ui16WpChr2;			// 2nd character if any; default 0 for US lang
	FLMUINT		uiColLen;			// Return value of collated length
	FLMUINT		uiSubColBitPos;	// Sub-collation bit position
	FLMUINT	 	uiCaseBitPos;		// Case bit position
	FLMUINT		uiFlags;				// Clear all bit flags
	FLMBOOL		bHebrewArabic = FALSE;	// Set if language is hebrew, arabic, farsi
	FLMBOOL		bTwoIntoOne = FALSE;
	FLMUINT		uiUppercaseFlag;

	uiColLen = 0;
	uiSubColBitPos = 0;
	uiCaseBitPos = 0;
	uiFlags = 0;
	ui16WpChr2 = 0;

	// We don't want any single key piece to "pig out" more
	// than 256 bytes of the key

	if( uiTargetColLen > 256 - 8)
	{
		uiTargetColLen = 256 - 8;
	}

	// Code below sets ucSubColBuf[] and ucCaseBits[] values to zero.

	if (uiLanguage != FLM_US_LANG)
	{
		if (uiLanguage == FLM_AR_LANG ||		// Arabic
			 uiLanguage == FLM_FA_LANG ||		// Farsi - persian
			 uiLanguage == FLM_HE_LANG ||		// Hebrew
			 uiLanguage == FLM_UR_LANG)			// Urdu
		{
			bHebrewArabic = TRUE;
		}
	}

	for (;;)
	{
		// Set the case bits and sub-collation bits to zero when
		// on the first bit of the byte.

		if (!(uiCaseBitPos & 0x07))
		{
			ucCaseBits [uiCaseBitPos >> 3] = 0;
		}
		if (!(uiSubColBitPos & 0x07))
		{
			ucSubColBuf [uiSubColBitPos >> 3] = 0;
		}

		ui16SubColVal = 0; // Default sub-collation value

		// Get the next character from the string.

		if( RC_BAD( rc = f_readUTF8CharAsUnicode( pIStream, &uChar)))
		{
			if (rc == NE_FLM_EOF_HIT)
			{
				rc = NE_FLM_OK;
				break;
			}
			goto Exit;
		}

		// f_wpCheckDoubleCollation modifies ui16WpChr if a digraph or a double
		// character sequence is found.  If a double character is found, pucStr
		// is incremented past the next character and ui16WpChr2 is set to 1.
		// If a digraph is found, pucStr is not changed, but ui16WpChr
		// contains the first character and ui16WpChr2 contains the second
		// character of the digraph.

		if (uiLanguage != FLM_US_LANG)
		{
			if( RC_BAD( rc = f_wpCheckDoubleCollation( 
				pIStream, FALSE, TRUE, &uChar, &uChar2, &bTwoIntoOne, uiLanguage)))
			{
				goto Exit;
			}
			if (!f_unicodeToWP( uChar, &ui16WpChr))
			{
				ui16WpChr = UNK_UNICODE_CODE;
			}
			if (uChar2)
			{
				if (!f_unicodeToWP( uChar2, &ui16WpChr2))
				{
					ui16WpChr2 = UNK_UNICODE_CODE;
				}
			}
			else
			{
				ui16WpChr2 = 0;
			}
		}
		else
		{

			// Convert the character to its WP equivalent

			if( !f_unicodeToWP( uChar, &ui16WpChr))
			{
				ui16WpChr = UNK_UNICODE_CODE;
			}
		}

		// Save the case bit if not case-insensitive

		if (!bCaseInsensitive)
		{

			// charIsUpper returns TRUE if upper case, 0 if lower case.

			if (!charIsUpper( ui16WpChr))
			{
				uiFlags |= F_HAD_LOWER_CASE;
			}
			else
			{
				// Set if upper case.

				setBit( ucCaseBits, uiCaseBitPos);
			}
			uiCaseBitPos++;
		}

		// Handle non-collating characters with subcollating values,
		// Get the collated value from the WP character-if not collating value

		if ((pucCollatedStr[ uiColLen++] =
				(FLMBYTE)(f_wpGetCollation( ui16WpChr, uiLanguage))) >= COLS11)
		{
			FLMUINT	uiTemp;

			// If lower case, convert to upper case.

			if (!charIsUpper( ui16WpChr))
			{
				ui16WpChr &= ~1;
			}

			// No collating value given for this WP char.
			// Save original WP char (2 bytes) in subcollating
			// buffer.

			// 1110 is a new code that will store an insert over
			// the character OR a non-convertable unicode character.
			// Store with the same alignment as "store_extended_char"
			// below.

			// 11110 is code for unmappable UNICODE value.
			// A value 0xFE will be the collation value.  The sub-collation
			// value will be 0xFFFF followed by the UNICODE value.
			// Be sure to eat an extra case bit.

			// See specific Hebrew and Arabic comments in the
			//	switch statement below.

			// Set the next byte that follows in the sub collation buffer.

			ucSubColBuf [(uiSubColBitPos + 8) >> 3] = 0;
			if (bHebrewArabic && (pucCollatedStr[ uiColLen - 1] == COLS0_ARABIC))
			{
				// Store first bit of 1110, fall through & store remaining 3 bits

				setBit( ucSubColBuf, uiSubColBitPos);
				uiSubColBitPos++;

				// Don't store collation value

				uiColLen--;
			}
			else if( uChar)
			{
				ui16WpChr = uChar;
				uChar = 0;

				// Store 11 out of 11110

				setBit( ucSubColBuf, uiSubColBitPos);
				uiSubColBitPos++;
				setBit( ucSubColBuf, uiSubColBitPos);
				uiSubColBitPos++;
				if (!bCaseInsensitive)
				{
					ucCaseBits [(uiCaseBitPos + 7) >> 3] = 0;

					// Set upper case bit.

					setBit( ucCaseBits, uiCaseBitPos);
					uiCaseBitPos++;
				}
			}
store_extended_char:

			// Set the next byte that follows in the sub collation buffer.

			ucSubColBuf [(uiSubColBitPos + 8) >> 3] = 0;
			ucSubColBuf [(uiSubColBitPos + 16) >> 3] = 0;
			uiFlags |= F_HAD_SUB_COLLATION;

			// Set 110 bits in sub-collation - continued from above.
			// No need to explicitly set the zero, but must increment
			// for it.

			setBit( ucSubColBuf, uiSubColBitPos);
			uiSubColBitPos++;
			setBit( ucSubColBuf, uiSubColBitPos);
			uiSubColBitPos += 2;

			// store_aligned_word: This label is not referenced.
			// Go to the next byte boundary to write the character.

			uiSubColBitPos = (uiSubColBitPos + 7) & (~7);
			uiTemp = bytesInBits( uiSubColBitPos);

			// Need to big-endian - so it will sort correctly.

			ucSubColBuf [uiTemp] = (FLMBYTE)(ui16WpChr >> 8);
			ucSubColBuf [uiTemp + 1] = (FLMBYTE)(ui16WpChr);
			uiSubColBitPos += 16;
			ucSubColBuf [uiSubColBitPos >> 3] = 0;
		}
		else
		{
			// Had a collation value
			// Add the lower/uppercase bit if a mixed case output.
			// If not lower ASCII set - check diacritic value for sub-collation

			if( !(ui16WpChr & 0xFF00))
			{
				// ASCII character set - set a single 0 bit - just need to
				// increment to do this.

				uiSubColBitPos++;
			}
			else
			{
				FLMBYTE	ucChar = (FLMBYTE)ui16WpChr;
				FLMBYTE	ucCharSet = (FLMBYTE)(ui16WpChr >> 8);

				// Convert char to uppercase because case information
				// is stored above.  This will help
				// ensure that the "ETA" doesn't sort before "eta"

				if( !charIsUpper( ui16WpChr))
				{
					ui16WpChr &= ~1;
				}

				switch( ucCharSet)
				{
					case F_CHSMUL1:	// Multinational 1
					{
						// If we cannot break down a char into base and
						// diacritic we cannot combine the charaacter
						// later when converting back the key.  In that case,
						// write the entire WP char in the sub-collation area.

						if( f_breakWPChar( ui16WpChr, &ui16Base, &ui16SubColVal))
						{
							goto store_extended_char;
						}

						// Write the FLAIM diacritic sub-collation value.
						// Prefix is 2 bits "10".  Remember to leave
						// "111" alone for the future.
						// NOTE: The "unlaut" character must sort after the "ring"
						// character.

						ui16SubColVal = ((ui16SubColVal & 0xFF) == F_UMLAUT &&
											  (uiLanguage == FLM_SU_LANG ||
												uiLanguage == FLM_SV_LANG ||
												uiLanguage == FLM_CZ_LANG ||
												uiLanguage == FLM_SL_LANG))
							?	(FLMUINT16)(fwp_dia60Tbl[ F_RING] + 1)
							:	(FLMUINT16)(fwp_dia60Tbl[ ui16SubColVal & 0xFF]);

store_sub_col:
						// Set the next byte that follows in the sub collation buffer.

						ucSubColBuf[ (uiSubColBitPos + 8) >> 3] = 0;
						uiFlags |= F_HAD_SUB_COLLATION;

						// Set the 10 bits - no need to explicitly set the zero, but
						// must increment for it.

						setBit( ucSubColBuf, uiSubColBitPos);
						uiSubColBitPos += 2;

						// Set sub-collation bits.

						setBits( 5, ucSubColBuf, uiSubColBitPos, ui16SubColVal);
						uiSubColBitPos += 5;
						break;
					}

					case F_CHSGREK:		// Greek
					{
						if (ucChar >= 52  ||			// Keep case bit for 52-69 else ignore
          				 ui16WpChr == 0x804 ||	// [ 8,4] BETA Medial | Terminal
							 ui16WpChr == 0x826) 	// [ 8,38] SIGMA terminal
						{
							goto store_extended_char;
						}

						// No subcollation to worry about - set a zero bit by
						// incrementing the bit position.

						uiSubColBitPos++;
						break;
					}

					case F_CHSCYR:
					{
						if (ucChar >= 144)
						{
							goto store_extended_char;
						}

						// No subcollation to worry about - set a zero bit by
						// incrementing the bit position.

						uiSubColBitPos++;

						// Georgian covers 208-249 - no collation defined yet

						break;
					}

					case F_CHSHEB:		// Hebrew
					{
						// Three sections in Hebrew:
						//		0..26 - main characters
						//		27..83 - accents that apear over previous character
						//		84..118- dagesh (ancient) hebrew with accents

						// Because the ancient is only used for sayings & scriptures
						// we will support a collation value and in the sub-collation
						// store the actual character because sub-collation is in
						// character order.

            		if (ucChar >= 84)		// Save ancient - value 84 and above
						{
							goto store_extended_char;
						}

						// No subcollation to worry about - set a zero bit by
						// incrementing the bit position.

						uiSubColBitPos++;
						break;
					}

					case F_CHSARB1:		// Arabic 1
					{
						// Three sections in Arabic:
						//		00..37  - accents that display OVER a previous character
						//		38..46  - symbols
						//		47..57  - numbers
						//		58..163 - characters
						//		164     - hamzah accent
						//		165..180- common characters with accents
						//		181..193- ligatures - common character combinations
						//		194..195- extensions - throw away when sorting

						if( ucChar <= 46)
						{
							goto store_extended_char;	// save original character
						}

						if( pucCollatedStr[ uiColLen - 1] == COLS10a + 1) // Alef?
						{
							ui16SubColVal = (ucChar >= 165)
								? (FLMUINT16)(fwp_alefSubColTbl[ ucChar - 165 ])
								: (FLMUINT16)7;			// Alef subcol value
							goto store_sub_col;
						}

						if (ucChar >= 181)			// Ligatures - char combination
						{
							goto store_extended_char;	// save original character
						}

						if (ucChar == 64)				// taa exception
						{
							ui16SubColVal = 8;
							goto store_sub_col;
						}

						// No subcollation to worry about - set a zero bit by
						// incrementing the bit position.

						uiSubColBitPos++;
						break;
					}

					case F_CHSARB2:			// Arabic 2
					{
						// There are some characters that share the same slot
						// Check the bit table if above character 64

						if (ucChar >= 64 &&
							 fwp_ar2BitTbl[(ucChar-64)>> 3] & (0x80 >> (ucChar&0x07)))
						{
							goto store_extended_char;	// Will save original
						}

						// No subcollation to worry about - set a zero bit by
						// incrementing the bit position.

						uiSubColBitPos++;
						break;
					}

					default:
					{
						// Increment bit position to set a zero bit.

						uiSubColBitPos++;
						break;
					}
				}
			}

			// Now let's worry about double character sorting

			if (ui16WpChr2)
			{
				if (pbOriginalCharsLost)
				{
					*pbOriginalCharsLost = TRUE;
				}

				// Set the next byte that follows in the sub collation buffer.

				ucSubColBuf[ (uiSubColBitPos + 7) >> 3] = 0;

				if (bTwoIntoOne)
				{

					// Sorts after character in ui16WpChr after call to
					// f_wpCheckDoubleCollation
					// Write the char 2 times so lower/upper bits are correct.
					// Could write infinite times because of collation rules.

					pucCollatedStr[ uiColLen] = ++pucCollatedStr[ uiColLen - 1];
					uiColLen++;

					// If original was upper case, set one more upper case bit

					if( !bCaseInsensitive)
					{
						ucCaseBits[ (uiCaseBitPos + 7) >> 3] = 0;
						if( !charIsUpper( ui16WpChr2))
						{
							uiFlags |= F_HAD_LOWER_CASE;
						}
						else
						{
							setBit( ucCaseBits, uiCaseBitPos);
						}
						uiCaseBitPos++;
					}

					// Take into account the diacritical space

					uiSubColBitPos++;
				}
				else
				{

					// We have a digraph, get second collation value

					pucCollatedStr[ uiColLen++] =
						(FLMBYTE)(f_wpGetCollation( ui16WpChr2, uiLanguage));

					// Normal case, assume no diacritics set

					uiSubColBitPos++;

					// If first was upper, set one more upper bit.

					if( !bCaseInsensitive)
					{
						ucCaseBits [(uiCaseBitPos + 7) >> 3] = 0;
						if (charIsUpper( ui16WpChr))
						{
							setBit( ucCaseBits, uiCaseBitPos);
						}
						uiCaseBitPos++;

						// no need to reset the uiFlags
					}
				}
			}
		}

		// Check to see if uiColLen is at some overflow limit.

		if (uiColLen >= uiCharLimit ||
			 uiColLen + bytesInBits( uiSubColBitPos) +
						  bytesInBits( uiCaseBitPos) >= uiTargetColLen)
		{

			// We hit the maximum number of characters.  See if we hit the
			// end of the string.

			if (RC_BAD( rc = f_readUTF8CharAsUnicode( pIStream, &uChar)))
			{
				if (rc == NE_FLM_EOF_HIT)
				{
					rc = NE_FLM_OK;
				}
				else
				{
					goto Exit;
				}
			}
			else
			{
				bDataTruncated = TRUE;
			}
			break;
		}
	}

	if (puiCollationLen)
	{
		*puiCollationLen = uiColLen;
	}

	// Add the first substring marker - also serves as making the string non-null.

	if (bFirstSubstring)
	{
		pucCollatedStr[ uiColLen++] = F_COLL_FIRST_SUBSTRING;
	}

	if (bDataTruncated)
	{
		pucCollatedStr[ uiColLen++ ] = F_COLL_TRUNCATED;
	}

	// Return NOTHING if no values found

	if (!uiColLen && !uiSubColBitPos)
	{
		if (puiCaseLen)
		{
			*puiCaseLen = 0;
		}
		goto Exit;
	}

	// Store extra zero bit in the sub-collation area for Hebrew/Arabic

	if (bHebrewArabic)
	{
		uiSubColBitPos++;
	}

	// Done putting the string into 4 sections - build the COLLATED KEY
	// Don't set uiUppercaseFlag earlier than here because F_SC_LOWER 
	// may be zero

	uiUppercaseFlag = (uiLanguage == FLM_GR_LANG)
								? F_SC_LOWER
								: F_SC_UPPER;

	// Did we write anything to the subcollation area?
	// The default terminating characters is (F_COLL_MARKER | F_SC_UPPER)

	if (uiFlags & F_HAD_SUB_COLLATION)
	{
		// Writes out a 0x7

		pucCollatedStr[ uiColLen++] = F_COLL_MARKER | F_SC_SUB_COL;

		// Move the sub-collation into the collating string

		uiLength = bytesInBits( uiSubColBitPos);
		f_memcpy( &pucCollatedStr[ uiColLen], ucSubColBuf, uiLength);
		uiColLen += uiLength;
	}

	// Move the upper/lower case stuff - force bits for Greek ONLY
	// This is such a small size that a memcpy is not worth it

	if( uiFlags & F_HAD_LOWER_CASE)
	{
		FLMUINT		uiNumBytes = bytesInBits( uiCaseBitPos);
		FLMBYTE *	pucCasePtr = ucCaseBits;

		// Output the 0x5

		pucCollatedStr[ uiColLen++] = (FLMBYTE)(F_COLL_MARKER | F_SC_MIXED);
		if( puiCaseLen)
		{
			*puiCaseLen = uiNumBytes + 1;
		}

		if( uiUppercaseFlag == F_SC_LOWER)
		{
			// Negate case bits for languages (like GREEK) that sort
			// upper case before lower case.

			while( uiNumBytes--)
			{
				pucCollatedStr[ uiColLen++] = ~(*pucCasePtr++);
			}
		}
		else
		{
			while( uiNumBytes--)
			{
				pucCollatedStr[ uiColLen++] = *pucCasePtr++;
			}
		}
	}
	else
	{
		// All characters are either upper or lower case, as determined
		// by uiUppercaseFlag.

		pucCollatedStr[ uiColLen++] = (FLMBYTE)(F_COLL_MARKER | uiUppercaseFlag);
		if( puiCaseLen)
		{
			*puiCaseLen = 1;
		}
	}

Exit:

	if( pbDataTruncated)
	{
		*pbDataTruncated = bDataTruncated;
	}

	*puiCollatedStrLen = uiColLen;
	return( rc);
}

/*****************************************************************************
Desc:		Convert a collated string to a WP word string
*****************************************************************************/
RCODE FTKAPI f_colStr2WPStr(
	const FLMBYTE *	pucColStr,			  	// Points to the collated string
	FLMUINT				uiColStrLen,		  	// Length of the collated string
	FLMBYTE *			pucWPStr,			  	// Output string to build - WP word string
	FLMUINT *			puiWPStrLen,
	FLMUINT				uiLang,
	FLMUINT *			puiUnconvChars,
	FLMBOOL *			pbDataTruncated,		// Set to TRUE if truncated
	FLMBOOL *			pbFirstSubstring)		// Sets to TRUE if first substring
{
	FLMBYTE *	pucWPPtr = pucWPStr;			// Points to the word string data area
	FLMBYTE *	pucWPEnd = &pucWPPtr[ *puiWPStrLen];
	FLMUINT		uiMaxWPBytes = *puiWPStrLen;
	FLMUINT		uiLength = uiColStrLen;		// May optimize as a register
	FLMUINT		uiPos = 0;						// Position in pucColStr
	FLMUINT		uiBitPos;						// Computed bit position
	FLMUINT		uiColChar;						// Not portable if a FLMBYTE value
	FLMUINT		uiWPStrLen;
	FLMUINT		uiUnconvChars = 0;
	FLMBOOL		bHebrewArabic = FALSE;
	RCODE			rc = NE_FLM_OK;

	//  WARNING:
	//  The code is duplicated for performance reasons.
	//  The US code below is much more optimized so
	//  any changes must be done twice.

	if( uiLang == FLM_US_LANG)
	{
		while( uiLength && (pucColStr[ uiPos] > F_MAX_COL_OPCODE))
		{
			uiLength--;

			// Move in the WP value given uppercase collated value

			uiColChar = (FLMUINT)pucColStr[ uiPos++];
			if( uiColChar == COLS0)
			{
				uiColChar = (FLMUINT)0xFFFF;
				uiUnconvChars++;
			}
			else
			{
				uiColChar = (FLMUINT)colToWPChr[ uiColChar - COLLS];
			}

			// Put the WP char in the word string

			if( pucWPPtr + 2 >= pucWPEnd)
			{
				rc = RC_SET( NE_FLM_CONV_DEST_OVERFLOW);
				goto Exit;
			}

			UW2FBA( (FLMUINT16)uiColChar, pucWPPtr);
			pucWPPtr += 2;
		}
	}
	else // Non-US collation
	{
		if( (uiLang == FLM_AR_LANG ) ||		// Arabic
			 (uiLang == FLM_FA_LANG ) ||		// Farsi - Persian
			 (uiLang == FLM_HE_LANG ) ||		// Hebrew
			 (uiLang == FLM_UR_LANG))			// Urdu
		{
			bHebrewArabic = TRUE;
		}

		while( uiLength && (pucColStr[ uiPos] > F_MAX_COL_OPCODE))
		{
			uiLength--;
			uiColChar = (FLMUINT)pucColStr[ uiPos++];

			switch( uiColChar)
			{
				case COLS9+4:		// ch in spanish
				case COLS9+11:		// ch in czech
				{
					// Put the WP char in the word string

					if( pucWPPtr + 2 >= pucWPEnd)
					{
						rc = RC_SET( NE_FLM_CONV_DEST_OVERFLOW);
						goto Exit;
					}

					UW2FBA( (FLMUINT16) 'C', pucWPPtr);
					pucWPPtr += 2;
					uiColChar = (FLMUINT)'H';
					uiPos++;	// Move past second duplicate char
					break;
				}

				case COLS9+17:		// ll in spanish
				{
					// Put the WP char in the word string

					if( pucWPPtr + 2 >= pucWPEnd)
					{
						rc = RC_SET( NE_FLM_CONV_DEST_OVERFLOW);
						goto Exit;
					}

					UW2FBA( (FLMUINT16)'L', pucWPPtr);
					pucWPPtr += 2;
					uiColChar = (FLMUINT)'L';
					uiPos++;	// Move past duplicate character
					break;
				}

				case COLS0:			// Non-collating character or OEM character
				{
					// Actual character is in sub-collation area

					uiColChar = (FLMUINT)0xFFFF;
					uiUnconvChars++;
					break;
				}

				default:
				{
					// Watch out COLS10h has () around it for subtraction

					if( bHebrewArabic && (uiColChar >= COLS10h))
					{
						uiColChar = (uiColChar < COLS10a)	// Hebrew only?
					 			? (FLMUINT) (0x900 + (uiColChar - (COLS10h)))	// Hebrew
					 			: (FLMUINT) (HebArabColToWPChr[ uiColChar - (COLS10a)]);	// Arabic
					}
					else
					{
						uiColChar = (FLMUINT)colToWPChr[ uiColChar - COLLS];
					}
					break;
				}
			}

			// Put the WP char in the word string

			if( pucWPPtr + 2 >= pucWPEnd)
			{
				rc = RC_SET( NE_FLM_CONV_DEST_OVERFLOW);
				goto Exit;
			}

			UW2FBA( (FLMUINT16)uiColChar, pucWPPtr);
			pucWPPtr += 2;
		}
	}

	// Terminate the string

	if( pucWPPtr + 2 >= pucWPEnd)
	{
		rc = RC_SET( NE_FLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}

	UW2FBA( (FLMUINT16)0, pucWPPtr);
	uiWPStrLen = uiPos + uiPos;	// Multiply by 2

	// Parse through the sub-collation and case information.
	//  Here are values for some of the codes:
	//   [ 0x04] - case information is all uppercase (IS,DK,GR)
	//   [ 0x05] - case bits follow
	//   [ 0x06] - case information is all uppercase
	//   [ 0x07] - beginning of sub-collation information
	//   [ 0x08] - first substring field that is made
	//   [ 0x09] - truncation marker for text and binary
	//
	//  Below are some cases to consider...
	//
	// [ COLLATION][ 0x07 sub-collation][ 0x05 case info]
	// [ COLLATION][ 0x07 sub-collation][ 0x05 case info]
	// [ COLLATION][ 0x07 sub-collation]
	// [ COLLATION][ 0x07 sub-collation]
	// [ COLLATION][ 0x05 case info]
	// [ COLLATION][ 0x05 case info]
	// [ COLLATION]
	// [ COLLATION]
	//
	//  In the future still want[ 0x06] to be compressed out for uppercase
	//  only indexes.

	// Check first substring before truncated

	if( uiLength && pucColStr[ uiPos] == F_COLL_FIRST_SUBSTRING)
	{
		if( pbFirstSubstring)
		{
			*pbFirstSubstring = TRUE;	// Don't need to initialize to FALSE.
		}
		uiLength--;
		uiPos++;
	}

	// Is the key truncated?

	if( uiLength && pucColStr[ uiPos] == F_COLL_TRUNCATED)
	{
		if( pbDataTruncated)
		{
			*pbDataTruncated = TRUE;	// Don't need to initialize to FALSE.
		}
		uiLength--;
		uiPos++;
	}

	// Does sub-collation follow?
	// Still more to process - first work on the sub-collation (diacritics)
	// Hebrew/Arabic may have empty collation area

	if( uiLength && (pucColStr[ uiPos] == (F_COLL_MARKER | F_SC_SUB_COL)))
	{
		FLMUINT	uiTempLen;

		// Do another pass on the word string adding the diacritics

		if( RC_BAD( rc = flmWPCmbSubColBuf( pucWPStr, &uiWPStrLen, uiMaxWPBytes,
			&pucColStr[ ++uiPos], bHebrewArabic, &uiBitPos)))
		{
			goto Exit;
		}

		// Move pos to next byte value

		uiTempLen = bytesInBits( uiBitPos);
		uiPos += uiTempLen;
		uiLength -= uiTempLen + 1; // The 1 includes the 0x07 byte
	}

	// Does the case info follow?

	if( uiLength && (pucColStr[ uiPos] >= 0x04))
	{
		// Take care of the lower and upper case conversion
		// If mixed case then convert using case bits

		if( pucColStr[ uiPos++] & F_SC_MIXED)	// Increment pos here!
		{
			// Don't pre-increment pos on line below!
			uiPos += f_wpToMixed( pucWPStr, uiWPStrLen,
				&pucColStr[ uiPos], uiLang);
		}
		// else 0x04 or 0x06 - all characters already in uppercase
	}
	
	// Should end perfectly at the end of the collation buffer.

	if (uiPos != uiColStrLen)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_DATA_ERROR);
		goto Exit;
	}
	
	*puiWPStrLen = uiWPStrLen;
	*puiUnconvChars = uiUnconvChars;

Exit:

	return( rc);
}

/****************************************************************************
Desc:  	Convert a text string to a collated string.
****************************************************************************/
RCODE FTKAPI f_asiaUTF8ToColText(
	IF_PosIStream *	pIStream,
	FLMBYTE *			pucColStr,			// Output collated string
	FLMUINT *			puiColStrLen,		// Collated string length return value
													// Input value is MAX num of bytes in buffer
	FLMBOOL				bCaseInsensitive,	// Set if to convert to uppercase
	FLMUINT *			puiCollationLen,	// Returns the collation bytes length
	FLMUINT *			puiCaseLen,			// Returns length of case bytes
	FLMUINT				uiCharLimit,		// Max number of characters in this key piece
	FLMBOOL				bFirstSubstring,	// TRUE is this is the first substring key
	FLMBOOL				bDataTruncated,	// Was input data already truncated.
	FLMBOOL *			pbDataTruncated)
{
	RCODE			rc = NE_FLM_OK;
	FLMBOOL		bEndOfStr = FALSE;
	FLMUINT		uiLength;
	FLMUINT 		uiTargetColLen = *puiColStrLen - 12; // 6=ovhd,6=worst char
	FLMBYTE		ucSubColBuf[ MAX_SUBCOL_BUF + 1]; // Holds Sub-col values (diac)
	FLMBYTE		ucLowUpBuf[ MAX_CASE_BYTES + MAX_CASE_BYTES + 2]; // 2 case bits/wpchar
	FLMUINT		uiColLen;
	FLMUINT		uiSubColBitPos;
	FLMUINT 		uiLowUpBitPos;
	FLMUINT		uiFlags;
	FLMUNICODE	uChar;
	FLMUINT16	ui16NextWpChar;
	FLMUINT16	ui16ColValue;

	uiColLen = uiSubColBitPos = uiLowUpBitPos = uiFlags = 0;
	uChar = ui16ColValue = 0;

	// We don't want any single key piece to "pig out" more
	// than 256 bytes of the key

	if( uiTargetColLen > 256 - 12)
	{
		uiTargetColLen = 256 - 12;
	}

	// Make sure ucSubColBuf and ucLowUpBuf are set to 0

	f_memset( ucSubColBuf, 0, sizeof( ucSubColBuf));
	f_memset( ucLowUpBuf,  0, sizeof( ucLowUpBuf));

	ui16NextWpChar = 0;

	while( !bEndOfStr || ui16NextWpChar || uChar)
	{
		FLMUINT16	ui16WpChar;			// Current WP character
		FLMUINT16	ui16SubColVal;		// Sub-collated value (diacritic)
		FLMBYTE		ucCaseFlags;
		FLMUINT16	ui16CurWpChar;

		// Get the next character from the string.

		ui16WpChar = ui16NextWpChar;
		for( ui16NextWpChar = 0;
			  (!ui16WpChar || !ui16NextWpChar) &&
				  !uChar && !bEndOfStr;)
		{
			if (!bEndOfStr)
			{
				if( RC_BAD( rc = f_readUTF8CharAsUnicode( pIStream, &uChar)))
				{
					if (rc == NE_FLM_EOF_HIT)
					{
						rc = NE_FLM_OK;
						bEndOfStr = TRUE;
					}
					else
					{
						goto Exit;
					}
				}
			}
			else
			{
				uChar = 0;
			}

			if( f_unicodeToWP( uChar, &ui16CurWpChar))
			{
				uChar = 0;
			}

			if( !ui16WpChar)
			{
				ui16WpChar = ui16CurWpChar;
			}
			else
			{
				ui16NextWpChar = ui16CurWpChar;
			}
		}

		// If we didn't get a character, break out of the outer
		// processing loop.

		if( !ui16WpChar && !uChar)
		{
			break;
		}

		if( ui16WpChar)
		{
			if( flmWPAsiaGetCollation( ui16WpChar, ui16NextWpChar, ui16ColValue,
				&ui16ColValue, &ui16SubColVal, &ucCaseFlags, bCaseInsensitive) == 2)
			{
				// Took the ui16NextWpChar value
				// Force to skip this value

				ui16NextWpChar = 0;
			}
		}
		else // Use the uChar value for this pass
		{
			// This handles all of the UNICODE characters that could not
			// be converted to WP characters - which will include most
			// of the Asian characters.

			ucCaseFlags = 0;
			if( uChar < 0x20)
			{
				ui16ColValue = 0xFFFF;

				// Setting ui16SubColVal to a high code will ensure
				// that the code that the uChar value will be stored
				// in in the sub-collation area.

				ui16SubColVal = 0xFFFF;

				// NOTE: uChar SHOULD NOT be set to zero here.
				// It will be set to zero below.
			}
			else
			{
				ui16ColValue = uChar;
				ui16SubColVal = 0;
				uChar = 0;
			}
		}

		// Store the values in 2 bytes

		pucColStr[ uiColLen++] = (FLMBYTE)(ui16ColValue >> 8);
		pucColStr[ uiColLen++] = (FLMBYTE)(ui16ColValue & 0xFF);

		if( ui16SubColVal)
		{
			uiFlags |= F_HAD_SUB_COLLATION;
			if( ui16SubColVal <= 31)	// 5 bit - store bits 10
			{
				setBit( ucSubColBuf, uiSubColBitPos);
				uiSubColBitPos += 1 + 1; // Stores a zero
				setBits( 5, ucSubColBuf, uiSubColBitPos, ui16SubColVal);
				uiSubColBitPos += 5;
			}
			else	// 2 bytes - store bits 110 or 11110
			{
				FLMUINT		 uiTemp;

				setBit( ucSubColBuf, uiSubColBitPos);
				uiSubColBitPos++;
				setBit( ucSubColBuf, uiSubColBitPos);
				uiSubColBitPos++;

				if( !ui16WpChar && uChar) // Store as "11110"
				{
					ui16SubColVal = uChar;
					uChar = 0;
					setBit( ucSubColBuf, uiSubColBitPos);
					uiSubColBitPos++;
					setBit( ucSubColBuf, uiSubColBitPos);
					uiSubColBitPos++;
				}
				uiSubColBitPos++;	// Skip past the zero

				// Go to the next byte boundary to write the WP char
				uiSubColBitPos = (uiSubColBitPos + 7) & (~7);
				uiTemp = bytesInBits( uiSubColBitPos);

				// Need to store HIGH-Low - PC format is Low-high!
				ucSubColBuf[ uiTemp ] = (FLMBYTE)(ui16SubColVal >> 8);
				ucSubColBuf[ uiTemp + 1] = (FLMBYTE)(ui16SubColVal);

				uiSubColBitPos += 16;
			}
		}
		else
		{
			uiSubColBitPos++;
		}

		// Save case information - always 2 bits worth for Asian

		if( ucCaseFlags & 0x02)
		{
			setBit( ucLowUpBuf, uiLowUpBitPos);
		}

		uiLowUpBitPos++;

		if( ucCaseFlags & 0x01)
		{
			setBit( ucLowUpBuf, uiLowUpBitPos);
		}
		uiLowUpBitPos++;

		// Check to see if uiColLen is within 1 byte of max

		if( (uiColLen >= uiCharLimit) ||
			 (uiColLen + bytesInBits( uiSubColBitPos) +
					 bytesInBits( uiLowUpBitPos) >= uiTargetColLen))
		{
			// Still something left?

			if (ui16NextWpChar || uChar)
			{
				bDataTruncated = TRUE;
			}
			else if (!bEndOfStr)
			{
				if (RC_BAD( rc = f_readUTF8CharAsUnicode( pIStream, &uChar)))
				{
					if (rc == NE_FLM_EOF_HIT)
					{
						bEndOfStr = TRUE;
						rc = NE_FLM_OK;
					}
					else
					{
						goto Exit;
					}
				}
				else
				{
					bDataTruncated = TRUE;
				}
			}
			break; // Hit the max. number of characters
		}
	}

	if( puiCollationLen)
	{
		*puiCollationLen = uiColLen;
	}

	// Add the first substring marker - also serves
	// as making the string non-null.

	if( bFirstSubstring)
	{
		pucColStr[ uiColLen++] = 0;
		pucColStr[ uiColLen++] = F_COLL_FIRST_SUBSTRING;
	}

	if( bDataTruncated)
	{
		pucColStr[ uiColLen++] = 0;
		pucColStr[ uiColLen++] = F_COLL_TRUNCATED;
	}

	// Return NOTHING if no values found

	if( !uiColLen && !uiSubColBitPos)
	{
		if( puiCaseLen)
		{
			*puiCaseLen = 0;
		}
		goto Exit;
	}

	// Done putting the String into 3 sections - build the COLLATED KEY

	if( uiFlags & F_HAD_SUB_COLLATION)
	{
		pucColStr[ uiColLen++] = 0;
		pucColStr[ uiColLen++] = F_COLL_MARKER | F_SC_SUB_COL;

		// Move the Sub-collation (diacritics) into the collating string

		uiLength = (FLMUINT)(bytesInBits( uiSubColBitPos));
		f_memcpy( &pucColStr[ uiColLen], ucSubColBuf, uiLength);
		uiColLen += uiLength;
	}

	// Always represent the marker as 2 bytes and case bits in Asia

	pucColStr[ uiColLen++] = 0;
	pucColStr[ uiColLen++] = F_COLL_MARKER | F_SC_MIXED;

	uiLength = (FLMUINT)(bytesInBits( uiLowUpBitPos));
	f_memcpy( &pucColStr[ uiColLen ], ucLowUpBuf, uiLength);

	if( puiCaseLen)
	{
		*puiCaseLen = (FLMUINT)(uiLength + 2);
	}
	uiColLen += uiLength;

Exit:

	if( pbDataTruncated)
	{
		*pbDataTruncated = bDataTruncated;
	}

	*puiColStrLen = uiColLen;
	return( rc);
}

/****************************************************************************
Desc:		Combine the diacritic 5 and 16 bit values to an existing word string.
Ret:		FLMUINT - Number of bytes parsed
Notes:	For each bit in the sub-collation section:
	0 - no subcollation information
	10 - take next 5 bits - will tell about diacritics or japanese vowel
	110 - align to next byte & take word value as extended character

****************************************************************************/
RCODE FTKAPI f_asiaParseSubCol(
	FLMBYTE *			pucWPStr,
	FLMUINT *			puiWPStrLen,
	FLMUINT				uiMaxWPBytes,
	const FLMBYTE *	pucSubColBuf,
	FLMUINT *			puiSubColBitPos)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT 		uiSubColBitPos = 0;
	FLMUINT 		uiNumChars = *puiWPStrLen >> 1;
	FLMUINT16 	ui16Diac;
	FLMUINT16 	ui16WpChar;

	// For each character (16 bits) in the WP string ...

	while( uiNumChars--)
	{
		// Have to skip 0, because it is not accounted for
		// in the sub-collation bits.  It was inserted when we
		// encountered unconverted unicode characters (Asian).
		// Will be converted to something else later on.
		// SEE NOTE ABOVE.

		if( FB2UW( pucWPStr) == 0)
		{
			pucWPStr += 2;
			continue;
		}

		// This macro DOESN'T increment uiBitPos

		if( testOneBit( pucSubColBuf, uiSubColBitPos))
		{
			//  Bits 10 - take next 5 bits
			//  Bits 110 align and take next word
			//  Bits 11110 align and take unicode value

			uiSubColBitPos++;
			if( !testOneBit( pucSubColBuf, uiSubColBitPos))
			{
				uiSubColBitPos++;
				ui16Diac = (FLMUINT16)(getNBits( 5, pucSubColBuf, uiSubColBitPos));
				uiSubColBitPos += 5;

				if( (ui16WpChar = FB2UW( pucWPStr)) < 0x100)
				{
					if( (ui16WpChar >= 'A') && (ui16WpChar <= 'Z'))
					{
						// Convert to WP diacritic and combine characters

						f_combineWPChar( &ui16WpChar, ui16WpChar,
							(FLMUINT16)ml1_COLtoD[ ui16Diac]);

						// Even if cmbcar fails, WpChar is still set to a valid value
					}
					else
					{
						// Symbols from charset 0x24

						ui16WpChar = (FLMUINT16)(0x2400 +
							fwp_Ch24ColTbl[ ui16Diac - 1 ].ByteValue);
					}
				}
				else if( ui16WpChar >= 0x2600)	// Katakana
				{
					//  Voicings - will allow to select original char
					//		000 - some 001 are changed to 000 to save space
					//		001 - set if large char (uppercase)
					//		010 - set if voiced
					//		100 - set if half voiced
					//
					//  Should NOT match voicing or wouldn't be here!

					FLMBYTE ucChar = (FLMBYTE)(ui16WpChar & 0xFF);

					// Try exceptions first so don't access out of bounds

					if( ucChar == 84)
					{
						ui16WpChar = (FLMUINT16)(0x2600 +
												((ui16Diac == 1)
												? (FLMUINT16)10
												: (FLMUINT16)11));
					}
					else if( ucChar == 85)
					{
						ui16WpChar = (FLMUINT16)(0x2600 +
												((ui16Diac == 1)
												 ? (FLMUINT16)16
												 : (FLMUINT16)17));
					}

					// Try the next 2 slots, if not then
					// value is 83, 84 or 85

					else if( KanaSubColTbl[ ucChar + 1 ] == ui16Diac)
					{
						ui16WpChar++;
					}
					else if( KanaSubColTbl[ ucChar + 2 ] == ui16Diac)
					{
						ui16WpChar += 2;
					}
					else if( ucChar == 4) // Last exception
					{
						ui16WpChar = 0x2600 + 83;
					}

					// else, leave alone! - invalid storage
				}

				UW2FBA( ui16WpChar, pucWPStr);	// Set if changed or not
			}
			else		// "110"
			{
				FLMUINT    uiTemp;

				uiSubColBitPos++;	// Skip second '1'
				if( testOneBit( pucSubColBuf, uiSubColBitPos))	// 11?10 ?
				{
					if( (*puiWPStrLen) + 2 > uiMaxWPBytes)
					{
						rc = RC_SET( NE_FLM_CONV_DEST_OVERFLOW);
						goto Exit;
					}

					// Unconvertable UNICODE character
					// The format will be 4 bytes, 0xFF, 0xFF, 2 byte Unicode

					shiftN( pucWPStr,
						(FLMUINT16)(uiNumChars + uiNumChars + 4), 2);

					pucWPStr += 2;	// Skip the 0xFFFF for now
					uiSubColBitPos += 2;	// Skip next "11"
					(*puiWPStrLen) += 2;
				}
				uiSubColBitPos++;	// Skip the zero

				// Round up to next byte
				uiSubColBitPos = (uiSubColBitPos + 7) & (~7);
				uiTemp = bytesInBits( uiSubColBitPos);
				pucWPStr[ 1] = pucSubColBuf[ uiTemp];	// Character set
				pucWPStr[ 0] = pucSubColBuf[ uiTemp + 1];	// Character
				uiSubColBitPos += 16;
			}
		}
		else
		{
			uiSubColBitPos++;	// Be sure to increment this!
		}

		pucWPStr += 2; // Next WP character
	}

	*puiSubColBitPos = bytesInBits( uiSubColBitPos);

Exit:

	return( rc);
}

/****************************************************************************
Desc:		The case bits for asia are:
				Latin/Greek/Cyrillic
					01 - case bit set if character is uppercase
					10 - double wide character in CS 0x25xx, 0x26xx and 0x27xx
				Japanese
					00 - double wide hiragana 0x255e..25b0
					01 - double wide katakana 0x2600..2655
					10 - single wide symbols from charset 11 that map to CS24??
					11 - single wide katakana from charset 11
Ret:
Notes:	This is tricky to really understand the inputs.
			This looks at the bits according to the current character value.
****************************************************************************/
FSTATIC RCODE flmAsiaParseCase(
	FLMBYTE *			pucWPStr,
	FLMUINT *			puiWPStrLen,
	FLMUINT				uiMaxWPBytes,
	const FLMBYTE *	pucCaseBits,
	FLMUINT *			puiColBytesProcessed)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiWPStrLen = *puiWPStrLen;
	FLMUINT		uiCharCnt;
	FLMUINT		uiExtraBytes = 0;
	FLMUINT16	ui16WpChar;
	FLMBYTE		ucTempByte = 0;
	FLMBYTE		ucMaskByte;

	// For each character (two bytes) in the string ...

	for( uiCharCnt = uiWPStrLen >> 1, ucMaskByte = 0; uiCharCnt--;)
	{
		FLMBYTE	ucChar;
		FLMBYTE	ucCharSet;

		ui16WpChar = FB2UW( pucWPStr);	// Get the next character

		// Must skip any 0xFFFFs or zeroes that were inserted.

		if( ui16WpChar == 0xFFFF || ui16WpChar == 0)
		{
			// Put back 0xFFFF in case it was a zero.

			UW2FBA( 0xFFFF, pucWPStr);
			pucWPStr += 2;
			uiExtraBytes += 2;
			continue;
		}

		// Time to get another byte?

		if( ucMaskByte == 0)
		{
			ucTempByte = *pucCaseBits++;
			ucMaskByte = 0x80;
		}

		ucCharSet = (FLMBYTE)(ui16WpChar >> 8);
		ucChar = (FLMBYTE)(ui16WpChar & 0xFF);

		// SINGLE WIDE - NORMAL CHARACTERS

		if( ui16WpChar < 0x2400)
		{
			// Convert to double wide?

			if( ucTempByte & ucMaskByte)
			{
				// Latin/greek/cyrillic
				// Convert to uppercase double wide char

				if( ucCharSet == 0) // Latin - uppercase
				{
					// May convert to 0x250F (Latin) or CS24

					if( ui16WpChar >= ASCII_UPPER_A && ui16WpChar <= ASCII_UPPER_Z)
					{
						// Convert to double wide

						ui16WpChar = (FLMUINT16)(ui16WpChar - 0x30 + 0x250F);
					}
					else
					{
						f_wpHanToZenkaku( ui16WpChar, 0, &ui16WpChar);
					}
				}
				else if( ucCharSet == 8)	// Greek
				{
					if( ucChar > 38)	// Adjust for spaces in Greek
					{
						ucChar -= 2;
					}

					if( ucChar > 4)
					{
						ucChar -= 2;
					}

					ui16WpChar = (FLMUINT16)((ucChar >> 1) + 0x265E);
				}
				else if( ucCharSet == 10)	// Cyrillic
				{
					ui16WpChar = (FLMUINT16)((ucChar >> 1) + 0x2700);
				}
				else
				{
					f_wpHanToZenkaku( ui16WpChar, 0, &ui16WpChar);
				}

				ucCharSet = (FLMBYTE)(ui16WpChar >> 8);
				ucChar = (FLMBYTE)(ui16WpChar & 0xFF);
			}

			ucMaskByte >>= 1; // Next bit

			// Change to lower case?

			if( (ucTempByte & ucMaskByte) == 0)
			{
				// Convert ui16WpChar to lower case

				switch( ucCharSet)
				{
					case	0:
					{
						// Bit zero only if lower case

						ui16WpChar |= 0x20;
						break;
					}

					case	1:
					{
						// In upper/lower case region?

						if( ucChar >= 26)
						{
							ui16WpChar++;
						}
						break;
					}

					case	8:
					{
						// All lowercase after 69

						if( ucChar <= 69)
						{
							ui16WpChar++;
						}
						break;
					}

					case	10:
					{
						// No cases after 199

						if( ucChar <= 199)
						{
							ui16WpChar++;
						}
						break;
					}

					case	0x25:
					case	0x26:
					{
						// Should be double wide latin or Greek
						// Add offset to convert to lowercase

						ui16WpChar += 0x20;
						break;
					}

					case	0x27:
					{
						// Double wide cyrillic only
						// Add offset to convert to lowercase

						ui16WpChar += 0x30;
						break;
					}
				}
			}
		}
		else // JAPANESE CHARACTERS
		{
			if( ucTempByte & ucMaskByte)	// Original chars from CharSet 11
			{
				if( ucCharSet == 0x26)	// Convert to Zen to Hankaku
				{
					FLMUINT16	ui16NextChar = 0;

					ui16WpChar = f_wpZenToHankaku( ui16WpChar, &ui16NextChar);
					if( ui16NextChar)	// Move everyone down
					{
						if( (*puiWPStrLen) + 2 > uiMaxWPBytes)
						{
							rc = RC_SET( NE_FLM_CONV_DEST_OVERFLOW);
							goto Exit;
						}

						uiCharCnt++;
						shiftN( pucWPStr, uiCharCnt + uiCharCnt + 2, 2);
						UW2FBA( ui16WpChar, pucWPStr);
						pucWPStr += 2;
						ui16WpChar = ui16NextChar;	// This will be stored below

						// Adjust the length
						*puiWPStrLen = *puiWPStrLen + 2;
					}
				}
				else if( ucCharSet == 0x24)
				{
					ui16WpChar = f_wpZenToHankaku( ui16WpChar, NULL);
				}
				ucMaskByte >>= 1;	// Eat the next bit
			}
			else
			{
				ucMaskByte >>= 1;	// Next bit
				if( (ucTempByte & ucMaskByte) == 0)	// Convert to Hiragana?
				{
					// Kanji will also fall through here

					if( ucCharSet == 0x26)
					{
						// Convert to Hiragana
						ui16WpChar = (FLMUINT16)(0x255E + ucChar);
					}
				}
			}
		}
		UW2FBA( ui16WpChar, pucWPStr);
		pucWPStr += 2;
		ucMaskByte >>= 1;
	}

	uiCharCnt = uiWPStrLen - uiExtraBytes;	// Should be 2 bits for each character.
	*puiColBytesProcessed = bytesInBits( uiCharCnt);

Exit:

	return( rc);
}

/***************************************************************************
Desc:		Get the original string from an asian collation string
Ret:		Length of the word string in bytes
****************************************************************************/
RCODE FTKAPI f_asiaColStr2WPStr(
	const FLMBYTE *	pucColStr,			  	// Points to the collated string
	FLMUINT				uiColStrLen,		  	// Length of the collated string
	FLMBYTE *			pucWPStr,			  	// Output string to build - WP word string
	FLMUINT *			puiWPStrLen,
	FLMUINT *			puiUnconvChars,
	FLMBOOL *			pbDataTruncated,		// Set to TRUE if truncated
	FLMBOOL *			pbFirstSubstring)		// Sets to TRUE if first substring
{
	FLMBYTE *	pucWPStrPtr = pucWPStr;
	FLMBYTE *	pucWPEnd = &pucWPStr[ *puiWPStrLen];
	FLMUINT		uiLength = uiColStrLen;
	FLMUINT		uiMaxWPBytes = *puiWPStrLen;
	FLMUINT		uiColStrPos = 0;
	FLMBOOL		bHadExtended = FALSE;
	FLMUINT		uiWPStrLen;
	FLMUINT16	ui16ColChar;
	FLMUINT		uiUnconvChars = 0;
	FLMUINT		uiColBytesProcessed;
	RCODE			rc = NE_FLM_OK;

	while( uiLength)
	{
		FLMBYTE	ucChar = pucColStr[ uiColStrPos + 1];
		FLMBYTE	ucCharSet = pucColStr[ uiColStrPos];

		ui16ColChar = (FLMUINT16)((ucCharSet << 8) + ucChar);
		if( ui16ColChar <= F_MAX_COL_OPCODE)
		{
			break;
		}

		uiColStrPos += 2;
		uiLength -= 2;
		if( ucCharSet == 0)	// Normal Latin/Greek/Cyrillic value
		{
			ui16ColChar = colToWPChr[ ucChar - COLLS];
		}
		else if( ucCharSet == 1)	// Katakana or Hiragana character
		{
			if( ucChar > sizeof( ColToKanaTbl))	// Special cases below
			{
				if( ucChar == COLS_ASIAN_MARK_VAL) // Dakuten
				{
					ui16ColChar = 0x240a;
				}
				else if( ucChar == COLS_ASIAN_MARK_VAL + 1)	// Handakuten
				{
					ui16ColChar = 0x240b;
				}
				else if( ucChar == COLS_ASIAN_MARK_VAL + 2)	// Chuuten
				{
					ui16ColChar = 0x2405;
				}
				else
				{
					ui16ColChar = 0xFFFF;	// Error
				}
			}
			else
			{
				ui16ColChar = (FLMUINT16)(0x2600 + ColToKanaTbl[ ucChar]);
			}
		}
		else if( ucCharSet != 0xFF || ucChar != 0xFF)	// Asian characters
		{
			// Insert zeroes that will be treated as a signal for
			// uncoverted unicode characters later on.  NOTE: Cannot
			// use 0xFFFF, because we need to be able to detect this
			// case in the sub-collation stuff, and we don't want
			// to confuse it with the 0xFFFF that may have been inserted
			// in another case.
			// THIS IS A REALLY BAD HACK, BUT IT IS THE BEST WE CAN DO
			// FOR NOW!

			if( pucWPStrPtr + 2 >= pucWPEnd)
			{
				rc = RC_SET( NE_FLM_CONV_DEST_OVERFLOW);
				goto Exit;
			}

			*pucWPStrPtr++ = 0;
			*pucWPStrPtr++ = 0;
			uiUnconvChars++;
			bHadExtended = TRUE;
		}
		// else, there is no collation value - found in sub-collation part

		if( pucWPStrPtr + 2 >= pucWPEnd)
		{
			rc = RC_SET( NE_FLM_CONV_DEST_OVERFLOW);
			goto Exit;
		}

		UW2FBA( ui16ColChar, pucWPStrPtr);	// Put the uncollation value back
		pucWPStrPtr += 2;
	}

	if( pucWPStrPtr + 2 >= pucWPEnd)
	{
		rc = RC_SET( NE_FLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}

	UW2FBA( 0, pucWPStrPtr);	// Terminate the string
	uiWPStrLen = (FLMUINT)(pucWPStrPtr - pucWPStr);

	//  Parse through the sub-collation and case information.
	//  Here are values for some of the codes:
	//   [ 0x05] - case bits follow
	//   [ 0x06] - case information is all uppercase
	//   [ 0x07] - beginning of sub-collation information
	//   [ 0x08] - first substring field that is made
	//   [ 0x09] - truncation marker for text and binary
	//
	//  Asian chars the case information should always be there and not
	//  compressed out.  This is because the case information could change
	//  the actual width of the character from 0x26xx to charset 11.

	//  Does truncation marker or sub-collation follow?

	if( uiLength)
	{
		ui16ColChar = (FLMUINT16)((pucColStr[ uiColStrPos] << 8) +
									pucColStr[ uiColStrPos + 1]);

		// First substring is before truncated.
		
		if( ui16ColChar == F_COLL_FIRST_SUBSTRING)
		{
			if( pbFirstSubstring)
			{
				*pbFirstSubstring = TRUE;	// Don't need to initialize to FALSE.
			}

			uiLength -= 2;
			uiColStrPos += 2;
			ui16ColChar = (FLMUINT16)((pucColStr[ uiColStrPos] << 8) +
										pucColStr[ uiColStrPos + 1]);
		}

		if( ui16ColChar == F_COLL_TRUNCATED)
		{
			if( pbDataTruncated)
			{
				*pbDataTruncated = TRUE;	// Don't need to initialize to FALSE.
			}
			uiLength -= 2;
			uiColStrPos += 2;
			ui16ColChar = (FLMUINT16)((pucColStr[ uiColStrPos] << 8) +
										pucColStr[ uiColStrPos+1]);
		}

		if( ui16ColChar == (F_COLL_MARKER | F_SC_SUB_COL))
		{
			FLMUINT 	uiTempLen;

			// Do another pass on the word string adding diacritics/voicings

			uiColStrPos += 2;
			uiLength -= 2;
			if( RC_BAD( rc = f_asiaParseSubCol( pucWPStr, &uiWPStrLen,
				uiMaxWPBytes, &pucColStr[ uiColStrPos], &uiTempLen)))
			{
				goto Exit;
			}

			uiColStrPos += uiTempLen;
			uiLength -= uiTempLen;
		}
		else
		{
			goto check_case;
		}
	}

	// Does the case info follow?

	if( uiLength)
	{
		ui16ColChar = (FLMUINT16)((pucColStr[ uiColStrPos] << 8) +
									pucColStr[ uiColStrPos + 1]);
check_case:

		if( ui16ColChar == (F_COLL_MARKER | F_SC_MIXED))
		{
			uiColStrPos += 2;

			if( RC_BAD( rc = flmAsiaParseCase( pucWPStr, &uiWPStrLen,
				uiMaxWPBytes, &pucColStr[ uiColStrPos], &uiColBytesProcessed)))
			{
				goto Exit;
			}

			uiColStrPos += uiColBytesProcessed;

			// Set bHadExtended to FALSE, because they will have
			// been taken care of in this pass.

			bHadExtended = FALSE;
		}
	}

	// Change embedded zeroes to 0xFFFFs

	if (bHadExtended)
	{
		FLMUINT		uiCnt;
		FLMBYTE *	pucTmp;

		for( uiCnt = 0, pucTmp = pucWPStr;
			  uiCnt < uiWPStrLen;
			  uiCnt += 2, pucTmp += 2)
		{
			if( FB2UW( pucTmp) == 0)
			{
				UW2FBA( 0xFFFF, pucTmp);
			}
		}
	}
	
	if (uiColStrLen != uiColStrPos)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_DATA_ERROR);
		goto Exit;
	}

	*puiUnconvChars = uiUnconvChars;
	*puiWPStrLen = uiWPStrLen;

Exit:

	return( rc);
}

/**************************************************************************
Desc: 	Combine the diacritic 5-bit values to an existing WP string
***************************************************************************/
FSTATIC RCODE flmWPCmbSubColBuf(
	FLMBYTE *			pucWPStr,					// Existing WP string to modify
	FLMUINT *			puiWPStrLen,				// WP string length in bytes
	FLMUINT				uiMaxWPBytes,
	const FLMBYTE *	pucSubColBuf,				// Diacritic values in 5 bit sets
	FLMBOOL				bHebrewArabic,				// Set if language is Hebrew or Arabic
	FLMUINT *			puiSubColBitPos)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT 		uiSubColBitPos = 0;
	FLMUINT 		uiNumChars = *puiWPStrLen >> 1;
	FLMUINT16 	ui16Diac;
	FLMUINT16 	ui16WPChar;
	FLMUINT		uiTemp;

	// For each character (two bytes) in the WP string ...

	while( uiNumChars--)
	{
		// Label used for hebrew/arabic - additional subcollation can follow
		// This macro DOESN'T increment bitPos

		if( testOneBit( pucSubColBuf, uiSubColBitPos))
		{
			// If "11110" - unmappable unicode char - 0xFFFF is before it
			// If "1110" then INDEX extended char is inserted
			// If "110" then extended char follows that replaces collation
			// If "10"  then take next 5 bits which
			// contain the diacritic subcollation value.

after_last_character:

			uiSubColBitPos++;	// Eat the first 1 bit
			if( !testOneBit( pucSubColBuf, uiSubColBitPos))
			{
				uiSubColBitPos++;	// Eat the 0 bit
				ui16Diac = (FLMUINT16)(getNBits( 5, pucSubColBuf, uiSubColBitPos));
				uiSubColBitPos += 5;

				// If not extended base

				if( (ui16WPChar = FB2UW( pucWPStr)) < 0x100)
				{
					// Convert to WP diacritic and combine characters

					f_combineWPChar( &ui16WPChar, ui16WPChar,
						(FLMUINT16)ml1_COLtoD[ ui16Diac]);

					// Even if cmbcar fails, wpchar is still set to a valid value

					UW2FBA( ui16WPChar, pucWPStr);
				}
				else if( (ui16WPChar & 0xFF00) == 0x0D00)	// Arabic?
				{
					ui16WPChar = ArabSubColToWPChr[ ui16Diac];
					UW2FBA( ui16WPChar, pucWPStr);
				}
				// else diacritic is extra info
				// cmbcar should not handle extended chars for this design
			}
			else		// "110"  or "1110" or "11110"
			{
				uiSubColBitPos++;	// Eat the 2nd '1' bit
				if( testOneBit( pucSubColBuf, uiSubColBitPos))	// Test the 3rd bit
				{
					if( (*puiWPStrLen) + 2 > uiMaxWPBytes)
					{
						rc = RC_SET( NE_FLM_CONV_DEST_OVERFLOW);
						goto Exit;
					}

					// 1110 - shift wpchars down 1 word and insert value below
					uiSubColBitPos++;			// Eat the 3rd '1' bit
					*puiWPStrLen += 2;		// Return 2 more bytes

					if( testOneBit( pucSubColBuf, uiSubColBitPos))	// Test 4th bit
					{
						// Unconvertable UNICODE character
						// The format will be 4 bytes, 0xFF, 0xFF, 2 byte Unicode

						shiftN( pucWPStr, uiNumChars + uiNumChars + 4, 2);
						uiSubColBitPos++;	// Eat the 4th '1' bit
						pucWPStr += 2;	// Skip the 0xFFFF for now
					}
					else
					{
						// Move down 2 byte NULL and rest of the 2 byte characters
						// The extended character does not have a 0xFF col value

						shiftN( pucWPStr, uiNumChars + uiNumChars + 2, 2);
						uiNumChars++;	// Increment because inserted

						// Fall through reading the actual charater value
					}
				}

				uiSubColBitPos++;	// Skip past the zero bit
				uiSubColBitPos = (uiSubColBitPos + 7) & (~7);	// roundup to next byte
				uiTemp = bytesInBits( uiSubColBitPos);				// compute position
				pucWPStr[ 1] = pucSubColBuf[ uiTemp];				// Character set
				pucWPStr[ 0] = pucSubColBuf[ uiTemp + 1];			// Character
				uiSubColBitPos += 16;
			}
		}
		else
		{
			uiSubColBitPos++;
		}

		pucWPStr += 2;	// Next WP character
	}

	if( bHebrewArabic)
	{
		if( testOneBit( pucSubColBuf, uiSubColBitPos))
		{
			// Hebrew/Arabic can have trailing accents that
			// don't have a matching collation value.
			// Keep looping in this case.
			// Note that subColBitPos isn't incremented above.

			uiNumChars = 0;	// Set so we won't loop forever!
			goto after_last_character;	// process trailing bit
		}
		
		uiSubColBitPos++;	// Eat the last '0' bit
	}

	*puiSubColBitPos = uiSubColBitPos;

Exit:

	return( rc);
}

/**************************************************************************
Desc:
***************************************************************************/
void FTKAPI F_CollIStream::getCurrPosition(
	F_CollStreamPos *		pPos)
{
	pPos->uNextChar = m_uNextChar;
	pPos->ui64Position = m_pIStream->getCurrPosition();
}
