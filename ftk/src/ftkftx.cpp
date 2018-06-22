//------------------------------------------------------------------------------
// Desc: This file contains functions for supporting a cross-platform
//			text-based user interface.
// Tabs: 3
//
// Copyright (c) 1996-2007 Novell, Inc. All Rights Reserved.
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

#define FTX_MAX_WINNAME_LEN			128
#define FTX_KEYBUF_SIZE					128

/****************************************************************************
Desc:
****************************************************************************/
typedef struct
{
	char				charValue;
	char				attribute;
} NLM_CHAR_INFO;

/****************************************************************************
Desc:
****************************************************************************/
typedef struct FTX_WINDOW
{
	char *					pszBuffer;
	FLMBYTE *				pucForeAttrib;
	FLMBYTE *				pucBackAttrib;
	eColorType				backgroundColor;
	eColorType				foregroundColor;
	FLMUINT					uiUlx;
	FLMUINT					uiUly;
	FLMUINT					uiCols;
	FLMUINT					uiRows;
	FLMUINT					uiCurX;
	FLMUINT					uiCurY;
	FLMUINT					uiOffset;
	FLMUINT					uiCursorType;
	char						szName[ FTX_MAX_WINNAME_LEN + 4];
	FLMBOOL					bScroll;
	FLMBOOL					bOpen;
	FLMBOOL					bInitialized;
	FLMBOOL					bForceOutput;
	FLMBOOL					bNoLineWrap;
	FLMUINT					uiId;
	FTX_WINDOW *			pWinPrev;
	FTX_WINDOW *			pWinNext;
	struct FTX_SCREEN *	pScreen;
} FTX_WINDOW;

/****************************************************************************
Desc:
****************************************************************************/
typedef struct FTX_SCREEN
{
	F_MUTEX				hScreenMutex;
	F_SEM					hKeySem;
	FLMUINT				uiRows;
	FLMUINT				uiCols;
	eColorType			backgroundColor;
	eColorType			foregroundColor;
	FLMUINT				uiCursorType;
	char					szName[ FTX_MAX_WINNAME_LEN + 4];
	FLMBOOL				bInitialized;
	FLMBOOL				bChanged;
	FLMBOOL				bActive;
	FLMBOOL				bUpdateCursor;
	FLMUINT				uiSequence;
	FLMUINT				uiId;
	FLMBOOL *			pbShutdown;
	FTX_WINDOW *		pWinImage;
	FTX_WINDOW *		pWinScreen;
	FTX_WINDOW *		pWinCur;
	FTX_SCREEN *		pScreenPrev;
	FTX_SCREEN *		pScreenNext;
} FTX_SCREEN;

/****************************************************************************
Desc:
****************************************************************************/
typedef struct FTX_INFO
{
	F_MUTEX				hFtxMutex;
	IF_Thread *			pBackgroundThrd;
	IF_Thread *			pKeyboardThrd;
	IF_Thread *			pDisplayThrd;
	KEY_HANDLER 		pKeyHandler;
	void *				pvKeyHandlerData;
	FLMUINT				uiRows;
	FLMUINT				uiCols;
	eColorType			backgroundColor;
	eColorType			foregroundColor;
	FLMUINT				uiCursorType;
	FLMUINT				puiKeyBuffer[ FTX_KEYBUF_SIZE];
	FLMUINT				uiCurKey;
	FLMUINT				uiSequence;
	FLMBOOL				bExiting;
	FLMBOOL				bScreenSwitch;
	FLMBOOL				bRefreshDisabled;
	FLMBOOL				bEnablePingChar;
	FLMBOOL *			pbShutdown;
	FTX_SCREEN *		pScreenCur;
#if defined( FLM_WIN)
	PCHAR_INFO   		pCells;
#elif defined( FLM_NLM)
	scr_t					hScreen;
	NLM_CHAR_INFO *	pCells;
#endif

} FTX_INFO;

#if defined( FLM_WIN)

	typedef struct
	{
		unsigned char LeadChar;
		unsigned char SecondChar;
	} ftxWin32CharPair;

	typedef struct
	{
		unsigned short ScanCode;
		ftxWin32CharPair RegChars;
		ftxWin32CharPair ShiftChars;
		ftxWin32CharPair CtrlChars;
		ftxWin32CharPair AltChars;
	} ftxWin32EnhKeyVals;

	typedef struct
	{
		ftxWin32CharPair RegChars;
		ftxWin32CharPair ShiftChars;
		ftxWin32CharPair CtrlChars;
		ftxWin32CharPair AltChars;
	} ftxWin32NormKeyVals;

	static ftxWin32EnhKeyVals ftxWin32EnhancedKeys[] = 
	{
		{ 28, {  13,	  0 }, {	 13,	 0 }, {	10,	0 }, {	0, 166 } },
		{ 53, {  47,	  0 }, {	 63,	 0 }, {	 0, 149 }, {	0, 164 } },
		{ 71, { 224,	 71 }, { 224,	71 }, { 224, 119 }, {	0, 151 } },
		{ 72, { 224,	 72 }, { 224,	72 }, { 224, 141 }, {	0, 152 } },
		{ 73, { 224,	 73 }, { 224,	73 }, { 224, 134 }, {	0, 153 } },
		{ 75, { 224,	 75 }, { 224,	75 }, { 224, 115 }, {	0, 155 } },
		{ 77, { 224,	 77 }, { 224,	77 }, { 224, 116 }, {	0, 157 } },
		{ 79, { 224,	 79 }, { 224,	79 }, { 224, 117 }, {	0, 159 } },
		{ 80, { 224,	 80 }, { 224,	80 }, { 224, 145 }, {	0, 160 } },
		{ 81, { 224,	 81 }, { 224,	81 }, { 224, 118 }, {	0, 161 } },
		{ 82, { 224,	 82 }, { 224,	82 }, { 224, 146 }, {	0, 162 } },
		{ 83, { 224,	 83 }, { 224,	83 }, { 224, 147 }, {	0, 163 } }
	};

	#define FTX_WIN32_NUM_EKA_ELTS	(sizeof( ftxWin32EnhancedKeys) / sizeof( ftxWin32EnhKeyVals))

	static ftxWin32NormKeyVals ftxWin32NormalKeys[] = 
	{
		{ /*  0 */ {	  0,	 0 }, {	 0,	0 }, {	0,	  0 }, {	  0,	 0 } },
		{ /*  1 */ {	 27,	 0 }, {	27,	0 }, {  27,	  0 }, {	  0,	 1 } },
		{ /*  2 */ {	 49,	 0 }, {	33,	0 }, {	0,	  0 }, {	  0, 120 } },
		{ /*  3 */ {	 50,	 0 }, {	64,	0 }, {	0,	  3 }, {	  0, 121 } },
		{ /*  4 */ {	 51,	 0 }, {	35,	0 }, {	0,	  0 }, {	  0, 122 } },
		{ /*  5 */ {	 52,	 0 }, {	36,	0 }, {	0,	  0 }, {	  0, 123 } },
		{ /*  6 */ {	 53,	 0 }, {	37,	0 }, {	0,	  0 }, {	  0, 124 } },
		{ /*  7 */ {	 54,	 0 }, {	94,	0 }, {  30,	  0 }, {	  0, 125 } },
		{ /*  8 */ {	 55,	 0 }, {	38,	0 }, {	0,	  0 }, {	  0, 126 } },
		{ /*  9 */ {	 56,	 0 }, {	42,	0 }, {	0,	  0 }, {	  0, 127 } },
		{ /* 10 */ {	 57,	 0 }, {	40,	0 }, {	0,	  0 }, {	  0, 128 } },
		{ /* 11 */ {	 48,	 0 }, {	41,	0 }, {	0,	  0 }, {	  0, 129 } },
		{ /* 12 */ {	 45,	 0 }, {	95,	0 }, {  31,	  0 }, {	  0, 130 } },
		{ /* 13 */ {	 61,	 0 }, {	43,	0 }, {	0,	  0 }, {	  0, 131 } },
		{ /* 14 */ {	  8,	 0 }, {	 8,	0 }, { 127,	  0 }, {	  0,	14 } },
		{ /* 15 */ {	  9,	 0 }, {	 0,  15 }, {	0, 148 }, {	  0,	15 } },
		{ /* 16 */ {	113,	 0 }, {	81,	0 }, {  17,	  0 }, {	  0,	16 } },
		{ /* 17 */ {	119,	 0 }, {	87,	0 }, {  23,	  0 }, {	  0,	17 } },
		{ /* 18 */ {	101,	 0 }, {	69,	0 }, {	5,	  0 }, {	  0,	18 } },
		{ /* 19 */ {	114,	 0 }, {	82,	0 }, {  18,	  0 }, {	  0,	19 } },
		{ /* 20 */ {	116,	 0 }, {	84,	0 }, {  20,	  0 }, {	  0,	20 } },
		{ /* 21 */ {	121,	 0 }, {	89,	0 }, {  25,	  0 }, {	  0,	21 } },
		{ /* 22 */ {	117,	 0 }, {	85,	0 }, {  21,	  0 }, {	  0,	22 } },
		{ /* 23 */ {	105,	 0 }, {	73,	0 }, {	9,	  0 }, {	  0,	23 } },
		{ /* 24 */ {	111,	 0 }, {	79,	0 }, {  15,	  0 }, {	  0,	24 } },
		{ /* 25 */ {	112,	 0 }, {	80,	0 }, {  16,	  0 }, {	  0,	25 } },
		{ /* 26 */ {	 91,	 0 }, { 123,	0 }, {  27,	  0 }, {	  0,	26 } },
		{ /* 27 */ {	 93,	 0 }, { 125,	0 }, {  29,	  0 }, {	  0,	27 } },
		{ /* 28 */ {	 13,	 0 }, {	13,	0 }, {  10,	  0 }, {	  0,	28 } },
		{ /* 29 */ {	  0,	 0 }, {	 0,	0 }, {	0,	  0 }, {	  0,	 0 } },
		{ /* 30 */ {	 97,	 0 }, {	65,	0 }, {	1,	  0 }, {	  0,	30 } },
		{ /* 31 */ {	115,	 0 }, {	83,	0 }, {  19,	  0 }, {	  0,	31 } },
		{ /* 32 */ {	100,	 0 }, {	68,	0 }, {	4,	  0 }, {	  0,	32 } },
		{ /* 33 */ {	102,	 0 }, {	70,	0 }, {	6,	  0 }, {	  0,	33 } },
		{ /* 34 */ {	103,	 0 }, {	71,	0 }, {	7,	  0 }, {	  0,	34 } },
		{ /* 35 */ {	104,	 0 }, {	72,	0 }, {	8,	  0 }, {	  0,	35 } },
		{ /* 36 */ {	106,	 0 }, {	74,	0 }, {  10,	  0 }, {	  0,	36 } },
		{ /* 37 */ {	107,	 0 }, {	75,	0 }, {  11,	  0 }, {	  0,	37 } },
		{ /* 38 */ {	108,	 0 }, {	76,	0 }, {  12,	  0 }, {	  0,	38 } },
		{ /* 39 */ {	 59,	 0 }, {	58,	0 }, {	0,	  0 }, {	  0,	39 } },
		{ /* 40 */ {	 39,	 0 }, {	34,	0 }, {	0,	  0 }, {	  0,	40 } },
		{ /* 41 */ {	 96,	 0 }, { 126,	0 }, {	0,	  0 }, {	  0,	41 } },
		{ /* 42 */ {	  0,	 0 }, {	 0,	0 }, {	0,	  0 }, {	  0,	 0 } },
		{ /* 43 */ {	 92,	 0 }, { 124,	0 }, {  28,	  0 }, {	  0,	 0 } },
		{ /* 44 */ {	122,	 0 }, {	90,	0 }, {  26,	  0 }, {	  0,	44 } },
		{ /* 45 */ {	120,	 0 }, {	88,	0 }, {  24,	  0 }, {	  0,	45 } },
		{ /* 46 */ {	 99,	 0 }, {	67,	0 }, {	3,	  0 }, {	  0,	46 } },
		{ /* 47 */ {	118,	 0 }, {	86,	0 }, {  22,	  0 }, {	  0,	47 } },
		{ /* 48 */ {	 98,	 0 }, {	66,	0 }, {	2,	  0 }, {	  0,	48 } },
		{ /* 49 */ {	110,	 0 }, {	78,	0 }, {  14,	  0 }, {	  0,	49 } },
		{ /* 50 */ {	109,	 0 }, {	77,	0 }, {  13,	  0 }, {	  0,	50 } },
		{ /* 51 */ {	 44,	 0 }, {	60,	0 }, {	0,	  0 }, {	  0,	51 } },
		{ /* 52 */ {	 46,	 0 }, {	62,	0 }, {	0,	  0 }, {	  0,	52 } },
		{ /* 53 */ {	 47,	 0 }, {	63,	0 }, {	0,	  0 }, {	  0,	53 } },
		{ /* 54 */ {	  0,	 0 }, {	 0,	0 }, {	0,	  0 }, {	  0,	 0 } },
		{ /* 55 */ {	 42,	 0 }, {	 0,	0 }, { 114,	  0 }, {	  0,	 0 } },
		{ /* 56 */ {	  0,	 0 }, {	 0,	0 }, {	0,	  0 }, {	  0,	 0 } },
		{ /* 57 */ {	 32,	 0 }, {	32,	0 }, {  32,	  0 }, {	 32,	 0 } },
		{ /* 58 */ {	  0,	 0 }, {	 0,	0 }, {	0,	  0 }, {	  0,	 0 } },
		{ /* 59 */ {	  0,	59 }, {	 0,  84 }, {	0,	 94 }, {	  0, 104 } },
		{ /* 60 */ {	  0,	60 }, {	 0,  85 }, {	0,	 95 }, {	  0, 105 } },
		{ /* 61 */ {	  0,	61 }, {	 0,  86 }, {	0,	 96 }, {	  0, 106 } },
		{ /* 62 */ {	  0,	62 }, {	 0,  87 }, {	0,	 97 }, {	  0, 107 } },
		{ /* 63 */ {	  0,	63 }, {	 0,  88 }, {	0,	 98 }, {	  0, 108 } },
		{ /* 64 */ {	  0,	64 }, {	 0,  89 }, {	0,	 99 }, {	  0, 109 } },
		{ /* 65 */ {	  0,	65 }, {	 0,  90 }, {	0, 100 }, {	  0, 110 } },
		{ /* 66 */ {	  0,	66 }, {	 0,  91 }, {	0, 101 }, {	  0, 111 } },
		{ /* 67 */ {	  0,	67 }, {	 0,  92 }, {	0, 102 }, {	  0, 112 } },
		{ /* 68 */ {	  0,	68 }, {	 0,  93 }, {	0, 103 }, {	  0, 113 } },
		{ /* 69 */ {	  0,	 0 }, {	 0,	0 }, {	0,	  0 }, {	  0,	 0 } },
		{ /* 70 */ {	  0,	 0 }, {	 0,	0 }, {	0,	  0 }, {	  0,	 0 } },
		{ /* 71 */ {	  0,	71 }, {	55,	0 }, {	0, 119 }, {	  0,	 0 } },
		{ /* 72 */ {	  0,	72 }, {	56,	0 }, {	0, 141 }, {	  0,	 0 } },
		{ /* 73 */ {	  0,	73 }, {	57,	0 }, {	0, 132 }, {	  0,	 0 } },
		{ /* 74 */ {	  0,	 0 }, {	45,	0 }, {	0,	  0 }, {	  0,	 0 } },
		{ /* 75 */ {	  0,	75 }, {	52,	0 }, {	0, 115 }, {	  0,	 0 } },
		{ /* 76 */ {	  0,	 0 }, {	53,	0 }, {	0,	  0 }, {	  0,	 0 } },
		{ /* 77 */ {	  0,	77 }, {	54,	0 }, {	0, 116 }, {	  0,	 0 } },
		{ /* 78 */ {	  0,	 0 }, {	43,	0 }, {	0,	  0 }, {	  0,	 0 } },
		{ /* 79 */ {	  0,	79 }, {	49,	0 }, {	0, 117 }, {	  0,	 0 } },
		{ /* 80 */ {	  0,	80 }, {	50,	0 }, {	0, 145 }, {	  0,	 0 } },
		{ /* 81 */ {	  0,	81 }, {	51,	0 }, {	0, 118 }, {	  0,	 0 } },
		{ /* 82 */ {	  0,	82 }, {	48,	0 }, {	0, 146 }, {	  0,	 0 } },
		{ /* 83 */ {	  0,	83 }, {	46,	0 }, {	0, 147 }, {	  0,	 0 } },
		{ /* 84 */ {	  0,	 0 }, {	 0,	0 }, {	0,	  0 }, {	  0,	 0 } },
		{ /* 85 */ {	  0,	 0 }, {	 0,	0 }, {	0,	  0 }, {	  0,	 0 } },
		{ /* 86 */ {	  0,	 0 }, {	 0,	0 }, {	0,	  0 }, {	  0,	 0 } },
		{ /* 87 */ {	224, 133 }, { 224, 135 }, { 224, 137 }, { 224, 139 } },
		{ /* 88 */ {	224, 134 }, { 224, 136 }, { 224, 138 }, { 224, 140 } }
	};

	static HANDLE								gv_hStdOut;
	static HANDLE								gv_hStdIn;
	static FLMBOOL								gv_bAllocatedConsole = FALSE;
	static CONSOLE_SCREEN_BUFFER_INFO	gv_ConsoleScreenBufferInfo;

	FSTATIC FLMUINT ftxWin32KBGetChar( void);

	FSTATIC ftxWin32CharPair * ftxWin32GetExtendedKeycode(
		KEY_EVENT_RECORD *	pKE);

	static int chbuf = -1;

#elif defined( FLM_UNIX)

	#if defined( FLM_HPUX) || defined( FLM_OSF)
		#ifndef _XOPEN_CURSES
			#define _XOPEN_CURSES
		#endif
		#define _XOPEN_SOURCE_EXTENDED 1
	#endif
	
	// curses.h pollutes name spaces like crazy; these definitions
	// are required to get the code to compile cleanly on all platforms.
	
	#if defined( bool)
		#undef bool
	#endif
	
	#if defined( EO)
		#undef EO
	#endif
	
	#if defined( ERR)
		#undef ERR
	#endif
	
	#if defined( FLM_SOLARIS)
		#define _WIDEC_H
	#endif
	
	#include <curses.h>
	
	#ifdef FLM_AIX
		#ifdef wgetch
			#undef wgetch
		#endif
		
		extern "C"
		{
			extern int wgetch( WINDOW *);
			extern int clear( void);
		}
	#endif
	
	static int ungetChar;
	
	// Curses gives us only a limited number of color pairs.  We use this
	// static color_pairs array for only the colors we need. flm2curses is
	// used to convert from flaim colors to curses colors and last_pair
	// is the last_pair that we used.
	
	static short flm2curses[FLM_NUM_COLORS];
	static short color_pairs[FLM_NUM_COLORS][FLM_NUM_COLORS];
	short last_pair = 0;

	FSTATIC void ftxUnixDisplayChar(
		FLMUINT			uiChar,
		FLMUINT			uiAttr);
		
	FSTATIC void ftxUnixDisplayRefresh( void);
	
	FSTATIC void ftxUnixDisplayReset( void);

	FSTATIC void ftxUnixDisplayInit( void);
	
	FSTATIC void ftxUnixDisplayFree( void);
	
	FSTATIC void ftxUnixDisplayGetSize(
		FLMUINT *		puiNumColsRV,
		FLMUINT *		puiNumRowsRV);
		
	FSTATIC void ftxUnixDisplaySetCursorPos(
		FLMUINT			uiCol,
		FLMUINT			uiRow);
		
	FSTATIC FLMUINT ftxUnixKBGetChar( void);
		
	FSTATIC FLMBOOL ftxUnixKBTest( void);
	
#endif

static FLMATOMIC		gv_ftxInitCount = 0;
static FLMBOOL			gv_bDisplayInitialized = FALSE;
static FTX_INFO *		gv_pFtxInfo = NULL;
static FLMATOMIC		gv_conInitCount = 0;
static FTX_SCREEN *	gv_pConScreen = NULL;
static FTX_WINDOW *	gv_pConWindow = NULL;
static F_MUTEX			gv_hConMutex = F_MUTEX_NULL;

#if defined( FLM_WIN)

	FSTATIC void ftxWin32Refresh( void);

#elif defined( FLM_NLM)

	FSTATIC void ftxNLMRefresh( void);

#else

	FSTATIC void ftxRefresh( void);

#endif

FSTATIC void ftxSyncImage( void);

FSTATIC void ftxWinReset(
	FTX_WINDOW *	pWindow);

FSTATIC void ftxCursorUpdate( void);

FSTATIC void ftxWinPrintChar(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiChar);

FINLINE void ftxKeyboardFlush( void)
{
	gv_pFtxInfo->uiCurKey = 0;
	f_memset( gv_pFtxInfo->puiKeyBuffer, (FLMBYTE)0,
		sizeof( FLMUINT) * FTX_KEYBUF_SIZE);
}

FSTATIC void ftxWinClearLine(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiCol,
	FLMUINT			uiRow);

FSTATIC void ftxWinClose(
	FTX_WINDOW *	pWindow);

FSTATIC void ftxWinFree(
	FTX_WINDOW *	pWindow);

FSTATIC RCODE ftxWinOpen(
	FTX_WINDOW *	pWindow);

FSTATIC void ftxScreenFree(
	FTX_SCREEN *	pScreen);

FSTATIC void ftxWinSetCursorPos(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiCol,
	FLMUINT			uiRow);

FSTATIC RCODE ftxDisplayInit(
	FLMUINT			uiRows,
	FLMUINT			uiCols,
	const char *	pszTitle);

FSTATIC void ftxDisplayReset( void);

FSTATIC void ftxDisplayGetSize(
	FLMUINT *		puiNumColsRV,
	FLMUINT *		puiNumRowsRV);

FSTATIC FLMBOOL ftxDisplaySetCursorType(
	FLMUINT			uiType);

FSTATIC void ftxDisplayExit( void);

FSTATIC void ftxDisplaySetCursorPos(
	FLMUINT			uiCol,
	FLMUINT			uiRow);

FSTATIC void ftxDisplaySetBackFore(
	eColorType		backgroundColor,
	eColorType		foregroundColor);

#ifdef FLM_WIN
FSTATIC FLMUINT ftxMapFlmColorToWin32(
	eColorType		uiColor);
#endif
	
RCODE FTKAPI _ftxBackgroundThread(
	IF_Thread *		pThread);

FLMBOOL ftxKBTest( void);

FLMUINT ftxKBGetChar( void);

RCODE FTKAPI _ftxDefaultDisplayHandler(
	IF_Thread *		pThread);

RCODE FTKAPI _ftxDefaultKeyboardHandler(
	IF_Thread *		pThread);

#if defined( FLM_UNIX)
FSTATIC FLMUINT ftxDisplayStrOut(
	const char *	pszString,
	FLMUINT			uiAttribute);
#endif

/****************************************************************************
Desc:	Scan code conversion tables
****************************************************************************/
#if defined( FLM_WIN) || defined( FLM_NLM)
static FLMUINT ScanCodeToFKB[] = {
	0,					0,					0,					0,					/* 00..03 */
	0,					0,					0,					0,					/* 04 */
	0,					0,					0,					0,					/* 08 */
	0,					0,					0,					FKB_STAB,		/* 0C */
	FKB_ALT_Q,		FKB_ALT_W,		FKB_ALT_E,		FKB_ALT_R,		/* 10 */
	FKB_ALT_T,		FKB_ALT_Y,		FKB_ALT_U,		FKB_ALT_I,		/* 14 */
	FKB_ALT_O,		FKB_ALT_P,		0,					0,					/* 18 */
	0,					0,					FKB_ALT_A,		FKB_ALT_S,		/* 1C */
	FKB_ALT_D,		FKB_ALT_F,		FKB_ALT_G,		FKB_ALT_H,		/* 20 */
	FKB_ALT_J,		FKB_ALT_K,		FKB_ALT_L,		0,					/* 24 */
	0,					0,					0,					0,					/* 28 */
	FKB_ALT_Z,		FKB_ALT_X,		FKB_ALT_C,		FKB_ALT_V,		/* 2C */
	FKB_ALT_B,		FKB_ALT_N,		FKB_ALT_M,		0,					/* 30 */
	0,					0,					0,					0,					/* 34 */
	0,					0,					0,					FKB_F1,			/* 38 */
	FKB_F2,			FKB_F3,			FKB_F4,			FKB_F5,			/* 3C */
	FKB_F6,			FKB_F7,			FKB_F8,			FKB_F9,			/* 40 */
										/* F8 MAY BE BAD*/
	FKB_F10,			FKB_F11,			FKB_F12,			FKB_HOME,		/* 44 */
	FKB_UP,			FKB_PGUP,		0,					FKB_LEFT,		/* 48 */
	0,					FKB_RIGHT,		0,					FKB_END,			/* 4C */
	FKB_DOWN,		FKB_PGDN,		FKB_INSERT,		FKB_DELETE,		/* 50 */

	FKB_SF1,			FKB_SF2,			FKB_SF3,			FKB_SF4,			/* 54 */
	FKB_SF5,			FKB_SF6,			FKB_SF7,			FKB_SF8,			/* 58 */
	FKB_SF9,			FKB_SF10,		FKB_CTRL_F1,	FKB_CTRL_F2,	/* 5C */
	FKB_CTRL_F3,	FKB_CTRL_F4,	FKB_CTRL_F5,	FKB_CTRL_F6,	/* 60 */
	FKB_CTRL_F7,	FKB_CTRL_F8,	FKB_CTRL_F9,	FKB_CTRL_F10,	/* 64 */

	FKB_ALT_F1,		FKB_ALT_F2,		FKB_ALT_F3,		FKB_ALT_F4,		/* 68 */
	FKB_ALT_F5,		FKB_ALT_F6,		FKB_ALT_F7,		FKB_ALT_F8,		/* 6C */
	FKB_ALT_F9,		FKB_ALT_F10,	0,					FKB_CTRL_LEFT, /* 70 */
	FKB_CTRL_RIGHT,FKB_CTRL_END,	FKB_CTRL_PGDN, FKB_CTRL_HOME, /* 74 */

	FKB_CTRL_1,		FKB_CTRL_2,		FKB_CTRL_3,		FKB_CTRL_4,		/* 78 */
	FKB_CTRL_5,		FKB_CTRL_6,		FKB_CTRL_7,		FKB_CTRL_8,		/* 7C */
	FKB_CTRL_9,		FKB_CTRL_0,		FKB_CTRL_MINUS,FKB_CTRL_EQUAL,/* 80 */
	FKB_CTRL_PGUP, 0,					0,					0,					/* 84 */
	0,					0,					0,					0,					/* 88 */
	0,					FKB_CTRL_UP,	0,					0,					/* 8C */
	0,					FKB_CTRL_DOWN, 0,					0					/* 90 */
};
#endif

#ifdef FLM_NLM
	void *	g_pvScreenTag;
#endif

/****************************************************************************
Desc:		Initializes the FTX environment.
****************************************************************************/
RCODE FTKAPI FTXInit(
	const char *		pszAppName,
	FLMUINT				uiCols,
	FLMUINT				uiRows,
	eColorType			backgroundColor,
	eColorType			foregroundColor,
	KEY_HANDLER			pKeyHandler,
	void *				pvKeyHandlerData)
{
	RCODE					rc = NE_FLM_OK;
	FTX_INFO *			pFtxInfo;
	IF_ThreadMgr *		pThreadMgr = f_getThreadMgrPtr();

	if( f_atomicInc( &gv_ftxInitCount) > 1)
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_calloc( sizeof( FTX_INFO), &pFtxInfo)))
	{
		goto Exit;
	}
	gv_pFtxInfo = pFtxInfo;

	if( RC_BAD( rc = f_mutexCreate( &(gv_pFtxInfo->hFtxMutex))))
	{
		goto Exit;
	}

#ifdef FLM_NLM

		g_pvScreenTag = AllocateResourceTag( 
			f_getNLMHandle(), "Screen", ScreenSignature);

		(void)OpenScreen( pszAppName,
			g_pvScreenTag, &gv_pFtxInfo->hScreen);
		ActivateScreen( gv_pFtxInfo->hScreen);

#endif

	if( RC_BAD( rc = ftxDisplayInit( uiRows, uiCols, pszAppName)))
	{
		goto Exit;
	}

	ftxDisplayReset();
	ftxDisplayGetSize( &(gv_pFtxInfo->uiCols), &(gv_pFtxInfo->uiRows));

	if( uiCols && gv_pFtxInfo->uiCols > uiCols)
	{
		gv_pFtxInfo->uiCols = uiCols;
	}

	if( uiRows && gv_pFtxInfo->uiRows > uiRows)
	{
		gv_pFtxInfo->uiRows = uiRows;
	}

	gv_pFtxInfo->uiCursorType = FLM_CURSOR_INVISIBLE;
	ftxDisplaySetCursorType( gv_pFtxInfo->uiCursorType);

#if defined( FLM_WIN)

	if( RC_BAD( rc = f_calloc( (FLMUINT)(sizeof( CHAR_INFO) * (gv_pFtxInfo->uiCols *
		gv_pFtxInfo->uiRows)), &gv_pFtxInfo->pCells)))
	{
		goto Exit;
	}

#elif !defined( FLM_NLM) || !defined( FLM_UNIX)

	gv_pFtxInfo->uiRows--;

#endif

	if( RC_BAD( rc = pThreadMgr->createThread( &gv_pFtxInfo->pBackgroundThrd,
		_ftxBackgroundThread, "ftx_background")))
	{
		goto Exit;
	}

	gv_pFtxInfo->backgroundColor = backgroundColor;
	gv_pFtxInfo->foregroundColor = foregroundColor;

	if( RC_BAD( rc = pThreadMgr->createThread( &gv_pFtxInfo->pDisplayThrd,
		_ftxDefaultDisplayHandler, "ftx_display")))
	{
		goto Exit;
	}

	// Start the keyboard handler

	gv_pFtxInfo->uiCurKey = 0;
	f_memset( gv_pFtxInfo->puiKeyBuffer, 0, sizeof( FLMUINT) * FTX_KEYBUF_SIZE);
	gv_pFtxInfo->pKeyHandler = pKeyHandler;
	gv_pFtxInfo->pvKeyHandlerData = pvKeyHandlerData;

	if( RC_BAD( rc = pThreadMgr->createThread( &gv_pFtxInfo->pKeyboardThrd,
		_ftxDefaultKeyboardHandler, "ftx_keyboard")))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Frees all resources allocated to the FTX environment
Notes:	All screens and windows are freed automatically
****************************************************************************/
void FTKAPI FTXExit( void)
{
	FTX_SCREEN *		pScreen;

	if( !gv_ftxInitCount || 
		 f_atomicDec( &gv_ftxInitCount) > 0 ||
		 !gv_pFtxInfo)
	{
		return;
	}

	if( gv_pFtxInfo->pKeyboardThrd)
	{
		gv_pFtxInfo->pKeyboardThrd->stopThread();
		gv_pFtxInfo->pKeyboardThrd->Release();
		gv_pFtxInfo->pKeyboardThrd = NULL;
	}
	
	if( gv_pFtxInfo->pDisplayThrd)
	{
		gv_pFtxInfo->pDisplayThrd->stopThread();
		gv_pFtxInfo->pDisplayThrd->Release();
		gv_pFtxInfo->pDisplayThrd = NULL;
	}
	
	if( gv_pFtxInfo->pBackgroundThrd)
	{
		gv_pFtxInfo->pBackgroundThrd->stopThread();
		gv_pFtxInfo->pBackgroundThrd->Release();
		gv_pFtxInfo->pBackgroundThrd = NULL;
	}
	
	if( gv_pFtxInfo->hFtxMutex != F_MUTEX_NULL)
	{
		f_mutexLock( gv_pFtxInfo->hFtxMutex);
	
		gv_pFtxInfo->bExiting = TRUE;
	
		while( (pScreen = gv_pFtxInfo->pScreenCur) != NULL)
		{
			ftxScreenFree( pScreen);
		}
	
		ftxDisplayReset();
		ftxDisplayExit();
	
	#if defined( FLM_WIN)
	
		f_free( &gv_pFtxInfo->pCells);
	
	#elif defined( FLM_NLM)
	
		CloseScreen( gv_pFtxInfo->hScreen);
	
	#endif
	
		f_mutexUnlock( gv_pFtxInfo->hFtxMutex);
		f_mutexDestroy( &(gv_pFtxInfo->hFtxMutex));
	}
	
	f_free( &gv_pFtxInfo);
}

/****************************************************************************
Desc:		Refreshes the current screen
****************************************************************************/
void FTKAPI FTXRefresh( void)
{
	FTX_WINDOW *	pWinScreen;

	f_mutexLock( gv_pFtxInfo->hFtxMutex);

	if( !gv_pFtxInfo->bRefreshDisabled && gv_pFtxInfo->pScreenCur)
	{
		f_mutexLock( gv_pFtxInfo->pScreenCur->hScreenMutex);
		if( gv_pFtxInfo->pScreenCur->bChanged || gv_pFtxInfo->bScreenSwitch)
		{
			if( gv_pFtxInfo->bScreenSwitch)
			{
				pWinScreen = gv_pFtxInfo->pScreenCur->pWinScreen;
				f_memset( pWinScreen->pszBuffer, 0,
					pWinScreen->uiRows * pWinScreen->uiCols);
				#ifdef FLM_UNIX
					ftxUnixDisplayReset();
				#endif
			}

#if defined( FLM_WIN)
			ftxWin32Refresh();
#elif defined( FLM_NLM)
			ftxNLMRefresh();
#else
			ftxRefresh();
#endif
			gv_pFtxInfo->pScreenCur->bChanged = FALSE;
			gv_pFtxInfo->bScreenSwitch = FALSE;
			gv_pFtxInfo->pScreenCur->bUpdateCursor = TRUE;
		}

		if( gv_pFtxInfo->pScreenCur->bUpdateCursor)
		{
			ftxCursorUpdate();
		}
		
		f_mutexUnlock( gv_pFtxInfo->pScreenCur->hScreenMutex);
	}

	f_mutexUnlock( gv_pFtxInfo->hFtxMutex);
}

/****************************************************************************
Desc:		Enables or disables refresh
****************************************************************************/
void FTKAPI FTXSetRefreshState(
	FLMBOOL			bDisable)
{
	f_mutexLock( gv_pFtxInfo->hFtxMutex);
	gv_pFtxInfo->bRefreshDisabled = bDisable;
	f_mutexUnlock( gv_pFtxInfo->hFtxMutex);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMBOOL FTKAPI FTXRefreshDisabled( void)
{
	FLMBOOL		bDisabled;
	
	f_mutexLock( gv_pFtxInfo->hFtxMutex);
	bDisabled = gv_pFtxInfo->bRefreshDisabled;
	f_mutexUnlock( gv_pFtxInfo->hFtxMutex);
	
	return( bDisabled);
}

/****************************************************************************
Desc:		Allows a keyboard handler to add a key to the FTX key buffer
****************************************************************************/
RCODE FTKAPI FTXAddKey(
	FLMUINT			uiKey)
{
	RCODE			rc = NE_FLM_OK;
	FLMBOOL		bSet = FALSE;
	FLMUINT		uiLoop;

	f_mutexLock( gv_pFtxInfo->hFtxMutex);

	uiLoop = gv_pFtxInfo->uiCurKey;
	while( uiLoop < FTX_KEYBUF_SIZE)
	{
		if( gv_pFtxInfo->puiKeyBuffer[ uiLoop] == 0)
		{
			gv_pFtxInfo->puiKeyBuffer[ uiLoop] = uiKey;
			bSet = TRUE;
			goto Exit;
		}
		uiLoop++;
	}

	if( !bSet)
	{
		uiLoop = 0;
		while( uiLoop < gv_pFtxInfo->uiCurKey)
		{
			if( gv_pFtxInfo->puiKeyBuffer[ uiLoop] == 0)
			{
				gv_pFtxInfo->puiKeyBuffer[ uiLoop] = uiKey;
				bSet = TRUE;
				goto Exit;
			}
			uiLoop++;
		}
	}

Exit:

	f_mutexUnlock( gv_pFtxInfo->hFtxMutex);

	if( !bSet)
	{
		rc = RC_SET( NE_FLM_CONV_DEST_OVERFLOW);
	}
	else
	{
		if( gv_pFtxInfo->pScreenCur != NULL)
		{
			f_semSignal( gv_pFtxInfo->pScreenCur->hKeySem);
		}
	}

	return( rc);
}

/****************************************************************************
Desc:		Cycles to the next screen in the FTX environment
****************************************************************************/
void FTKAPI FTXCycleScreensNext( void)
{
	FTX_SCREEN *		pScreenTmp;
	FTX_SCREEN *		pScreenLast;

	f_mutexLock( gv_pFtxInfo->hFtxMutex);

	if( gv_pFtxInfo->pScreenCur && gv_pFtxInfo->pScreenCur->pScreenNext)
	{
		pScreenTmp = gv_pFtxInfo->pScreenCur;
		gv_pFtxInfo->pScreenCur = gv_pFtxInfo->pScreenCur->pScreenNext;

		pScreenLast = gv_pFtxInfo->pScreenCur;
		while( pScreenLast->pScreenNext)
		{
			pScreenLast = pScreenLast->pScreenNext;
		}

		pScreenLast->pScreenNext = pScreenTmp;
		pScreenTmp->pScreenPrev = pScreenLast;
		pScreenTmp->pScreenNext = NULL;
		gv_pFtxInfo->pScreenCur->pScreenPrev = NULL;
		gv_pFtxInfo->bScreenSwitch = TRUE;
		ftxKeyboardFlush();
	}

	f_mutexUnlock( gv_pFtxInfo->hFtxMutex);
}

/****************************************************************************
Desc:		Cycles to the previous screen in the FTX environment
****************************************************************************/
void FTKAPI FTXCycleScreensPrev( void)
{
	FTX_SCREEN *	pScreenPreviousFront;
	FTX_SCREEN *	pScreenLast;

	f_mutexLock( gv_pFtxInfo->hFtxMutex);

	if( gv_pFtxInfo->pScreenCur && gv_pFtxInfo->pScreenCur->pScreenNext)
	{
		pScreenPreviousFront = gv_pFtxInfo->pScreenCur;
		pScreenLast = pScreenPreviousFront;

		while( pScreenLast->pScreenNext)
		{
			pScreenLast = pScreenLast->pScreenNext;
		}
		
		pScreenLast->pScreenPrev->pScreenNext = NULL;
		pScreenLast->pScreenPrev = NULL;
		pScreenLast->pScreenNext = pScreenPreviousFront;
		pScreenPreviousFront->pScreenPrev = pScreenLast;
		gv_pFtxInfo->pScreenCur = pScreenLast;
		gv_pFtxInfo->bScreenSwitch = TRUE;
		ftxKeyboardFlush();
	}

	f_mutexUnlock( gv_pFtxInfo->hFtxMutex);
}

/****************************************************************************
Desc:		Force cursor refresh
****************************************************************************/
void FTKAPI FTXRefreshCursor( void)
{
	f_mutexLock( gv_pFtxInfo->hFtxMutex);

	if( gv_pFtxInfo->pScreenCur)
	{
		gv_pFtxInfo->pScreenCur->bUpdateCursor = TRUE;
	}

	f_mutexUnlock( gv_pFtxInfo->hFtxMutex);
}

/****************************************************************************
Desc:		Invalidates the current screen so that it will be completly redrawn
****************************************************************************/
void FTKAPI FTXInvalidate( void)
{
	FTX_WINDOW *	pWinScreen;

	f_mutexLock( gv_pFtxInfo->hFtxMutex);

	if( gv_pFtxInfo->pScreenCur)
	{
		f_mutexLock( gv_pFtxInfo->pScreenCur->hScreenMutex);
		pWinScreen = gv_pFtxInfo->pScreenCur->pWinScreen;
		f_memset( pWinScreen->pszBuffer, 0,
			pWinScreen->uiRows * pWinScreen->uiCols);
		gv_pFtxInfo->pScreenCur->bChanged = TRUE;
		f_mutexUnlock( gv_pFtxInfo->pScreenCur->hScreenMutex);
	}

#ifdef FLM_UNIX
	ftxUnixDisplayReset();
#endif
	f_mutexUnlock( gv_pFtxInfo->hFtxMutex);
}


/****************************************************************************
Desc:		Allocates and initializes a new screen object
****************************************************************************/
RCODE FTKAPI FTXScreenInit(
	const char *	pszName,
	FTX_SCREEN **	ppScreen)
{
	RCODE				rc = NE_FLM_OK;
	FTX_SCREEN *	pScreen;
	FTX_SCREEN *	pScreenTmp;

	f_mutexLock( gv_pFtxInfo->hFtxMutex);

	*ppScreen = NULL;
	if( RC_BAD( rc = f_calloc( sizeof( FTX_SCREEN), &pScreen)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_mutexCreate( &(pScreen->hScreenMutex))))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_semCreate( &(pScreen->hKeySem))))
	{
		goto Exit;
	}

	pScreen->uiRows = gv_pFtxInfo->uiRows;
	pScreen->uiCols = gv_pFtxInfo->uiCols;
	pScreen->backgroundColor = gv_pFtxInfo->backgroundColor;
	pScreen->foregroundColor = gv_pFtxInfo->foregroundColor;
	pScreen->uiCursorType = FLM_CURSOR_VISIBLE | FLM_CURSOR_UNDERLINE;

	if( f_strlen( pszName) <= FTX_MAX_WINNAME_LEN)
	{
		f_strcpy( pScreen->szName, pszName);
	}
	else
	{
		f_strcpy( pScreen->szName, "?");
	}

	pScreen->bInitialized = TRUE;

	if( RC_BAD( rc = FTXWinInit( pScreen, pScreen->uiCols, pScreen->uiRows,
		&(pScreen->pWinScreen))))
	{
		goto Exit;
	}

	pScreen->pWinScreen->backgroundColor = pScreen->backgroundColor;
	pScreen->pWinScreen->foregroundColor = pScreen->foregroundColor;

	if( RC_BAD( rc = FTXWinInit( pScreen, pScreen->uiCols, pScreen->uiRows,
		&(pScreen->pWinImage))))
	{
		goto Exit;
	}

	f_memset( pScreen->pWinScreen->pszBuffer, 0,
		pScreen->pWinScreen->uiRows *
			pScreen->pWinScreen->uiCols);

Exit:

	if( RC_BAD( rc))
	{
		pScreen->bInitialized = FALSE;
	}
	else
	{
		if( gv_pFtxInfo->pScreenCur)
		{
			pScreenTmp = gv_pFtxInfo->pScreenCur;
			while( pScreenTmp->pScreenNext)
			{
				pScreenTmp = pScreenTmp->pScreenNext;
			}
			pScreenTmp->pScreenNext = pScreen;
			pScreen->pScreenPrev = pScreenTmp;
		}
		else
		{
			gv_pFtxInfo->pScreenCur = pScreen;
			gv_pFtxInfo->bScreenSwitch = TRUE;
		}

		pScreen->uiId = gv_pFtxInfo->uiSequence++;
		*ppScreen = pScreen;
	}

	f_mutexUnlock( gv_pFtxInfo->hFtxMutex);
	return( rc);
}


/****************************************************************************
Desc:		Frees all resources allocated to a screen, including all window
			objects
****************************************************************************/
void FTKAPI FTXScreenFree(
	FTX_SCREEN **	ppScreen)
{
	FTX_SCREEN *	pScreen;

	if( !ppScreen)
	{
		goto Exit;
	}

	pScreen = *ppScreen;
	if( !pScreen)
	{
		goto Exit;
	}

	if( !pScreen->bInitialized)
	{
		goto Exit;
	}

	f_mutexLock( gv_pFtxInfo->hFtxMutex);
	ftxScreenFree( pScreen);
	gv_pFtxInfo->bScreenSwitch = TRUE;
	f_mutexUnlock( gv_pFtxInfo->hFtxMutex);

Exit:

	*ppScreen = NULL;
}


/****************************************************************************
Desc:		Makes the passed-in screen the visible screen
****************************************************************************/
RCODE FTKAPI FTXScreenDisplay(
	FTX_SCREEN *	pScreen)
{
	RCODE				rc = NE_FLM_OK;
	FLMBOOL			bScreenValid = FALSE;
	FTX_SCREEN *	pTmpScreen;

	f_mutexLock( gv_pFtxInfo->hFtxMutex);

	// Make sure the screen is still in the list.  If it isn't, the thread
	// that owned the screen may have terminated.

	pTmpScreen = gv_pFtxInfo->pScreenCur;
	while( pTmpScreen)
	{
		if( pTmpScreen == pScreen)
		{
			bScreenValid = TRUE;
			break;
		}

		pTmpScreen = pTmpScreen->pScreenPrev;
	}

	pTmpScreen = gv_pFtxInfo->pScreenCur;
	while( pTmpScreen)
	{
		if( pTmpScreen == pScreen)
		{
			bScreenValid = TRUE;
			break;
		}

		pTmpScreen = pTmpScreen->pScreenNext;
	}

	if( !bScreenValid)
	{
		rc = RC_SET( NE_FLM_INVALID_PARM);
		goto Exit;
	}

	if( pScreen != gv_pFtxInfo->pScreenCur)
	{
		if( pScreen->pScreenNext != NULL)
		{
			pScreen->pScreenNext->pScreenPrev = pScreen->pScreenPrev;
		}

		if( pScreen->pScreenPrev != NULL)
		{
			pScreen->pScreenPrev->pScreenNext = pScreen->pScreenNext;
		}

		pScreen->pScreenPrev = NULL;
		pScreen->pScreenNext = gv_pFtxInfo->pScreenCur;
		gv_pFtxInfo->pScreenCur->pScreenPrev = pScreen;
		gv_pFtxInfo->pScreenCur = pScreen;
		gv_pFtxInfo->bScreenSwitch = TRUE;
		ftxKeyboardFlush();
	}

Exit:

	f_mutexUnlock( gv_pFtxInfo->hFtxMutex);
	return( rc);
}

/****************************************************************************
Desc:		Retrieves the size of the passed-in screen
****************************************************************************/
RCODE FTKAPI FTXScreenGetSize(
	FTX_SCREEN *	pScreen,
	FLMUINT *		puiNumCols,
	FLMUINT *		puiNumRows)
{
	RCODE				rc = NE_FLM_OK;

	if( !pScreen)
	{
		rc = RC_SET( NE_FLM_INVALID_PARM);
		goto Exit;
	}

	if( puiNumCols)
	{
		*puiNumCols = pScreen->uiCols;
	}

	if( puiNumRows)
	{
		*puiNumRows = pScreen->uiRows;
	}

Exit:

	return( rc);

}

/****************************************************************************
Desc:		Sets the screen's shutdown flag
****************************************************************************/
void FTKAPI FTXScreenSetShutdownFlag(
	FTX_SCREEN *	pScreen,
	FLMBOOL *		pbShutdownFlag)
{
	if( !pScreen)
	{
		goto Exit;
	}

	pScreen->pbShutdown = pbShutdownFlag;

Exit:

	return;
}

/****************************************************************************
Desc:		Creates a title window and main window (with border)
****************************************************************************/
RCODE FTKAPI FTXScreenInitStandardWindows(
	FTX_SCREEN *	pScreen,
	eColorType		titleBackColor,
	eColorType		titleForeColor,
	eColorType		mainBackColor,
	eColorType		mainForeColor,
	FLMBOOL			bBorder,
	FLMBOOL			bBackFill,
	const char *	pszTitle,
	FTX_WINDOW **	ppTitleWin,
	FTX_WINDOW **	ppMainWin)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiScreenCols;
	FLMUINT			uiScreenRows;
	FTX_WINDOW *	pTitleWin;
	FTX_WINDOW *	pMainWin;

	if( RC_BAD( rc = FTXScreenGetSize( pScreen,
		&uiScreenCols, &uiScreenRows)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FTXWinInit( pScreen, 0, 1, &pTitleWin)))
	{
		goto Exit;
	}

	FTXWinSetBackFore( pTitleWin, titleBackColor, titleForeColor);
	FTXWinClear( pTitleWin);
	FTXWinSetCursorType( pTitleWin, FLM_CURSOR_INVISIBLE);

	if( pszTitle)
	{
		FTXWinPrintf( pTitleWin, "%s", pszTitle);
	}

	if( RC_BAD( rc = FTXWinOpen( pTitleWin)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FTXWinInit( pScreen, uiScreenCols,
		uiScreenRows - 1, &pMainWin)))
	{
		goto Exit;
	}

	FTXWinMove( pMainWin, 0, 1);
	FTXWinSetBackFore( pMainWin, mainBackColor, mainForeColor);
	FTXWinClear( pMainWin);

	if( bBorder)
	{
		FTXWinDrawBorder( pMainWin);
	}

#if defined( FLM_WIN) || defined( FLM_NLM)
	if( bBackFill)
	{
		FTXWinSetChar( pMainWin, 176);
	}
#else
	F_UNREFERENCED_PARM( bBackFill);
#endif

	if( RC_BAD( rc = FTXWinOpen( pMainWin)))
	{
		goto Exit;
	}

	if( ppTitleWin)
	{
		*ppTitleWin = pTitleWin;
	}

	if( ppMainWin)
	{
		*ppMainWin = pMainWin;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Allocates and initializes a window object
****************************************************************************/
RCODE FTKAPI FTXWinInit(
	FTX_SCREEN *	pScreen,
	FLMUINT			uiCols,
	FLMUINT			uiRows,
	FTX_WINDOW **	ppWindow)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiSize;
	FTX_WINDOW *	pWindow;
	FTX_WINDOW *	pWinTmp;

	*ppWindow = NULL;

	if( !pScreen->bInitialized)
	{
		rc = RC_SET( NE_FLM_INVALID_PARM);
		goto Exit;
	}

	f_mutexLock( pScreen->hScreenMutex);

	if( uiRows > pScreen->uiRows || uiCols > pScreen->uiCols)
	{
		rc = RC_SET( NE_FLM_INVALID_PARM);
		goto Exit;
	}

	if( uiRows == 0)
	{
		uiRows = pScreen->uiRows;
	}

	if( uiCols == 0)
	{
		uiCols = pScreen->uiCols;
	}

	if( RC_BAD( rc = f_calloc( sizeof( FTX_WINDOW), &pWindow)))
	{
		goto Exit;
	}

	uiSize = (FLMUINT)((uiRows * uiCols) + 1);

	if( RC_BAD( rc = f_calloc( (FLMUINT)(sizeof( FLMBYTE) * uiSize),
		&pWindow->pszBuffer)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_calloc( (FLMUINT)(sizeof( FLMBYTE) * uiSize),
		&pWindow->pucForeAttrib)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_calloc( (FLMUINT)(sizeof( FLMBYTE) * uiSize),
		&pWindow->pucBackAttrib)))
	{
		goto Exit;
	}

	f_memset( pWindow->pucForeAttrib, (FLMBYTE)pScreen->foregroundColor, uiSize);
	f_memset( pWindow->pucBackAttrib, (FLMBYTE)pScreen->backgroundColor, uiSize);

	pWindow->uiRows = uiRows;
	pWindow->uiCols = uiCols;

	pWindow->uiCursorType = FLM_CURSOR_VISIBLE | FLM_CURSOR_UNDERLINE;
	pWindow->bScroll = TRUE;
	pWindow->bOpen = FALSE;
	pWindow->bInitialized = TRUE;
	pWindow->bForceOutput = FALSE;

	pWindow->pScreen = pScreen;
	pWindow->uiId = pScreen->uiSequence++;

	ftxWinReset( pWindow);

	if( pScreen->pWinCur)
	{
		pWinTmp = pScreen->pWinCur;
		while( pWinTmp->pWinNext)
		{
			pWinTmp = pWinTmp->pWinNext;
		}

		pWindow->pWinPrev = pWinTmp;
		pWinTmp->pWinNext = pWindow;
	}
	else
	{
		pScreen->pWinCur = pWindow;
	}
	
	*ppWindow = pWindow;

Exit:

	f_mutexUnlock( pScreen->hScreenMutex);
	return( rc);
}

/****************************************************************************
Desc:		Frees all resources associated with the passed-in window object
****************************************************************************/
void FTKAPI FTXWinFree(
	FTX_WINDOW **	ppWindow)
{
	FTX_WINDOW *	pWindow;
	FTX_SCREEN *	pScreen;

	if( !ppWindow)
	{
		goto Exit;
	}

	pWindow = *ppWindow;

	if( pWindow->bInitialized == FALSE)
	{
		goto Exit;
	}

	pScreen = pWindow->pScreen;
	f_mutexLock( pScreen->hScreenMutex);
	ftxWinFree( pWindow);
	f_mutexUnlock( pScreen->hScreenMutex);
	
Exit:

	*ppWindow = NULL;
}

/****************************************************************************
Desc:		Opens the specified window and makes it visible
****************************************************************************/
RCODE FTKAPI FTXWinOpen(
	FTX_WINDOW *	pWindow)
{
	RCODE				rc = NE_FLM_OK;

	if( !pWindow || !pWindow->bInitialized)
	{
		rc = RC_SET( NE_FLM_INVALID_PARM);
		goto Exit;
	}

	f_mutexLock( pWindow->pScreen->hScreenMutex);
	rc = ftxWinOpen( pWindow);
	f_mutexUnlock( pWindow->pScreen->hScreenMutex);

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Closes (or hides) the specified window
****************************************************************************/
void FTKAPI FTXWinClose(
	FTX_WINDOW *	pWindow)
{
	if( !pWindow)
	{
		goto Exit;
	}

	if( !pWindow->bInitialized || !pWindow->bOpen)
	{
		goto Exit;
	}

	f_mutexLock( pWindow->pScreen->hScreenMutex);
	ftxWinClose( pWindow);
	f_mutexUnlock( pWindow->pScreen->hScreenMutex);

Exit:

	return;
}

/****************************************************************************
Desc:		Sets the specified window's name
****************************************************************************/
RCODE FTKAPI FTXWinSetName(
	FTX_WINDOW *	pWindow,
	const char *	pszName)
{
	RCODE				rc = NE_FLM_OK;

	f_mutexLock( pWindow->pScreen->hScreenMutex);

	if( f_strlen( pszName) > FTX_MAX_WINNAME_LEN)
	{
		rc = RC_SET( NE_FLM_INVALID_PARM);
		goto Exit;
	}
	
	f_strcpy( pWindow->szName, pszName);

Exit:

	f_mutexUnlock( pWindow->pScreen->hScreenMutex);
	return( rc);
}

/****************************************************************************
Desc:		Moves the specified window to a new location on the screen
****************************************************************************/
void FTKAPI FTXWinMove(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiCol,
	FLMUINT			uiRow)
{
	f_mutexLock( pWindow->pScreen->hScreenMutex);

	if( (FLMUINT)uiCol + (FLMUINT)pWindow->uiCols >
		(FLMUINT)pWindow->pScreen->uiCols)
	{
		goto Exit;
	}

	if( uiRow + pWindow->uiRows > pWindow->pScreen->uiRows)
	{
		goto Exit;
	}

	if( pWindow->uiUlx != uiCol || pWindow->uiUly != uiRow)
	{
		pWindow->uiUlx = uiCol;
		pWindow->uiUly = uiRow;
		if( pWindow->bOpen)
		{
			pWindow->pScreen->bChanged = TRUE;
		}
	}

Exit:

	f_mutexUnlock( pWindow->pScreen->hScreenMutex);
	return;
}

/****************************************************************************
Desc:		Sets the input focus to the specified window
****************************************************************************/
void FTKAPI FTXWinSetFocus(
	FTX_WINDOW *	pWindow)
{
	f_mutexLock( pWindow->pScreen->hScreenMutex);

	if( pWindow->bOpen && pWindow->pScreen->pWinCur != pWindow)
	{
		ftxWinClose( pWindow);
		ftxWinOpen( pWindow);
	}

	f_mutexUnlock( pWindow->pScreen->hScreenMutex);
}

/****************************************************************************
Desc:		Sets the background color of all characters in the specified window
			to the same color
****************************************************************************/
void FTKAPI FTXWinPaintBackground(
	FTX_WINDOW *	pWindow,
	eColorType		backgroundColor)
{
	FLMUINT			uiSize;

	f_mutexLock( pWindow->pScreen->hScreenMutex);

	uiSize = (FLMUINT)(pWindow->uiRows * pWindow->uiCols);
	f_memset( pWindow->pucBackAttrib, (FLMBYTE)backgroundColor, uiSize);
	pWindow->backgroundColor = backgroundColor;

	if( pWindow->bOpen)
	{
		pWindow->pScreen->bChanged = TRUE;
	}

	f_mutexUnlock( pWindow->pScreen->hScreenMutex);
}

/****************************************************************************
Desc:		Sets the background and/or foreground color of a row in the
			specified window
****************************************************************************/
void FTKAPI FTXWinPaintRow(
	FTX_WINDOW *	pWindow,
	eColorType *	pBackground,
	eColorType *	pForeground,
	FLMUINT			uiRow)
{
	f_mutexLock( pWindow->pScreen->hScreenMutex);

	if( uiRow < (pWindow->uiRows - (2 * pWindow->uiOffset)))
	{
		if( pBackground != NULL)
		{
			f_memset( pWindow->pucBackAttrib +
				(pWindow->uiCols * (uiRow + pWindow->uiOffset)) + pWindow->uiOffset,
				(FLMBYTE)*pBackground, pWindow->uiCols - (2 * pWindow->uiOffset));
		}
		
		if( pForeground != NULL)
		{
			f_memset( pWindow->pucForeAttrib +
				(pWindow->uiCols * (uiRow + pWindow->uiOffset)) + pWindow->uiOffset,
				(FLMBYTE)*pForeground, pWindow->uiCols - (2 * pWindow->uiOffset));
		}

		if( pWindow->bOpen)
		{
			pWindow->pScreen->bChanged = TRUE;
		}
	}

	f_mutexUnlock( pWindow->pScreen->hScreenMutex);
}

/****************************************************************************
Desc:		Sets all of the characters in the window to the specified character
****************************************************************************/
void FTKAPI FTXWinSetChar(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiChar)
{
	FLMUINT			uiSize;

	f_mutexLock( pWindow->pScreen->hScreenMutex);

	uiSize = (FLMUINT)(pWindow->uiCols - pWindow->uiOffset) *
		(FLMUINT)(pWindow->uiRows - pWindow->uiOffset);

	f_memset( pWindow->pszBuffer, (FLMBYTE)uiChar, uiSize);
	
	if( pWindow->bOpen)
	{
		pWindow->pScreen->bChanged = TRUE;
	}

	f_mutexUnlock( pWindow->pScreen->hScreenMutex);
}

/****************************************************************************
Desc:		Sets the background color of a row in the specified window.
****************************************************************************/
void FTKAPI FTXWinPaintRowBackground(
	FTX_WINDOW *	pWindow,
	eColorType		backgroundColor,
	FLMUINT			uiRow)
{
	FTXWinPaintRow( pWindow, &backgroundColor, NULL, uiRow);
}

/****************************************************************************
Desc:		Sets the foreground color of all characters in the specified window
****************************************************************************/
void FTKAPI FTXWinPaintForeground(
	FTX_WINDOW *	pWindow,
	eColorType		foregroundColor)
{
	FLMUINT			uiSize;

	f_mutexLock( pWindow->pScreen->hScreenMutex);

	uiSize = (FLMUINT)(pWindow->uiRows * pWindow->uiCols);
	f_memset( pWindow->pucForeAttrib, (FLMBYTE)foregroundColor, uiSize);
	pWindow->foregroundColor = foregroundColor;

	if( pWindow->bOpen)
	{
		pWindow->pScreen->bChanged = TRUE;
	}

	f_mutexUnlock( pWindow->pScreen->hScreenMutex);
}

/****************************************************************************
Desc:		Sets the foreground color of a row in the specified window.
****************************************************************************/
void FTKAPI FTXWinPaintRowForeground(
	FTX_WINDOW *	pWindow,
	eColorType		foregroundColor,
	FLMUINT			uiRow)
{
	FTXWinPaintRow( pWindow, NULL, &foregroundColor, uiRow);
}

/****************************************************************************
Desc:		Sets the background and foreground color of the pen associated
			with the current window
****************************************************************************/
void FTKAPI FTXWinSetBackFore(
	FTX_WINDOW *	pWindow,
	eColorType		backgroundColor,
	eColorType		foregroundColor)
{
	f_mutexLock( pWindow->pScreen->hScreenMutex);

	pWindow->backgroundColor = backgroundColor;
	pWindow->foregroundColor = foregroundColor;

	f_mutexUnlock( pWindow->pScreen->hScreenMutex);
}

/****************************************************************************
Desc:		Retrieves the current background and/or foreground color of
			the pen associated with the specified window
****************************************************************************/
void FTKAPI FTXWinGetBackFore(
	FTX_WINDOW *	pWindow,
	eColorType *	pBackgroundColor,
	eColorType *	pForegroundColor)
{
	f_mutexLock( pWindow->pScreen->hScreenMutex);

	if( pBackgroundColor)
	{
		*pBackgroundColor = pWindow->backgroundColor;
	}

	if( pForegroundColor != NULL)
	{
		*pForegroundColor = pWindow->foregroundColor;
	}

	f_mutexUnlock( pWindow->pScreen->hScreenMutex);
}

/****************************************************************************
Desc:		Prints a character at the current cursor location in the
			specified window.
****************************************************************************/
void FTKAPI FTXWinPrintChar(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiChar)
{
	f_mutexLock( pWindow->pScreen->hScreenMutex);
	ftxWinPrintChar( pWindow, uiChar);
	f_mutexUnlock( pWindow->pScreen->hScreenMutex);
}

/****************************************************************************
Desc:	Prints a string starting at the current cursor location in the
		specified window.
****************************************************************************/
void FTKAPI FTXWinPrintStr(
	FTX_WINDOW *	pWindow,
	const char *	pszString)
{
	FLMBOOL			bMutexLocked = FALSE;

	if( !pszString)
	{
		goto Exit;
	}

	f_mutexLock( pWindow->pScreen->hScreenMutex);
	bMutexLocked = TRUE;

	while( *pszString != '\0')
	{
		ftxWinPrintChar( pWindow, (FLMUINT)*pszString);
		pszString++;
	}

Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( pWindow->pScreen->hScreenMutex);
	}
}

/****************************************************************************
Desc:		Output a formatted string at present cursor location.
****************************************************************************/
void FTKAPI FTXWinPrintf(
	FTX_WINDOW *	pWindow,
	const char *	pszFormat, ...)
{
	char			pszBuffer[ 512];
	f_va_list	args;

	f_va_start( args, pszFormat);
	f_vsprintf( pszBuffer, pszFormat, &args);
	f_va_end( args);
	FTXWinPrintStr( pWindow, pszBuffer);
}

/****************************************************************************
Desc:		Output a formatted string (with color) at present cursor location.
****************************************************************************/
void FTKAPI FTXWinCPrintf(
	FTX_WINDOW *	pWindow,
	eColorType		backgroundColor,
	eColorType		foregroundColor,
	const char *	pszFormat, ...)
{
	char				szBuffer[ 512];
	eColorType		oldBackground;
	eColorType		oldForeground;
	f_va_list		args;

	oldBackground = pWindow->backgroundColor;
	oldForeground = pWindow->foregroundColor;
	
	pWindow->backgroundColor = backgroundColor;
	pWindow->foregroundColor = foregroundColor;

	f_va_start( args, pszFormat);
	f_vsprintf( szBuffer, pszFormat, &args);
	f_va_end( args);

	FTXWinPrintStr( pWindow, szBuffer);

	pWindow->backgroundColor = oldBackground;
	pWindow->foregroundColor = oldForeground;
}

/****************************************************************************
Desc:		Prints a string at a specific offset in the specified window.
****************************************************************************/
void FTKAPI FTXWinPrintStrXY(
	FTX_WINDOW *	pWindow,
	const char *	pszString,
	FLMUINT			uiCol,
	FLMUINT			uiRow)
{
	FTXWinSetCursorPos( pWindow, uiCol, uiRow);
	FTXWinPrintStr( pWindow, pszString);
}

/****************************************************************************
Desc:		Sets the cursor position in the specified window
****************************************************************************/
void FTKAPI FTXWinSetCursorPos(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiCol,
	FLMUINT			uiRow)
{
	f_mutexLock( pWindow->pScreen->hScreenMutex);
	ftxWinSetCursorPos( pWindow, uiCol, uiRow);
	f_mutexUnlock( pWindow->pScreen->hScreenMutex);
}

/****************************************************************************
Desc:		Enables or disables scrolling in the specified window
****************************************************************************/
void FTKAPI FTXWinSetScroll(
	FTX_WINDOW *	pWindow,
	FLMBOOL			bScroll)
{
	if( !pWindow)
	{
		goto Exit;
	}

	if( pWindow->bInitialized == FALSE)
	{
		goto Exit;
	}

	pWindow->bScroll = bScroll;
	
Exit:

	return;
}

/****************************************************************************
Desc:		Enables or disables line wrap
****************************************************************************/
void FTKAPI FTXWinSetLineWrap(
	FTX_WINDOW *	pWindow,
	FLMBOOL			bLineWrap)
{
	if( !pWindow)
	{
		goto Exit;
	}

	if( pWindow->bInitialized == FALSE)
	{
		goto Exit;
	}

	pWindow->bNoLineWrap = !bLineWrap;

Exit:

	return;
}

/****************************************************************************
Desc:		Retrieves the scroll flag for the specified window
****************************************************************************/
void FTKAPI FTXWinGetScroll(
	FTX_WINDOW *	pWindow,
	FLMBOOL *		pbScroll)
{
	if( !pWindow || !pbScroll)
	{
		goto Exit;
	}

	if( pWindow->bInitialized == FALSE)
	{
		goto Exit;
	}

	*pbScroll = pWindow->bScroll;

Exit:

	return;
}

/****************************************************************************
Desc:		Retrieves the screen of the current window
****************************************************************************/
RCODE FTKAPI FTXWinGetScreen(
	FTX_WINDOW *	pWindow,
	FTX_SCREEN **	ppScreen)
{
	RCODE				rc = NE_FLM_OK;

	if( !pWindow || !ppScreen)
	{
		rc = RC_SET( NE_FLM_INVALID_PARM);
		goto Exit;
	}

	*ppScreen = NULL;

	if( pWindow->bInitialized == FALSE)
	{
		rc = RC_SET( NE_FLM_INVALID_PARM);
		goto Exit;
	}

	*ppScreen = pWindow->pScreen;

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Retrieves the windows position on the screen
****************************************************************************/
RCODE FTKAPI FTXWinGetPosition(
	FTX_WINDOW *	pWindow,
	FLMUINT *		puiCol,
	FLMUINT *		puiRow)
{
	RCODE				rc = NE_FLM_OK;

	if( !pWindow)
	{
		rc = RC_SET( NE_FLM_INVALID_PARM);
		goto Exit;
	}

	if( !pWindow->bInitialized)
	{
		rc = RC_SET( NE_FLM_INVALID_PARM);
		goto Exit;
	}

	if( puiCol)
	{
		*puiCol = pWindow->uiUlx;
	}

	if( puiRow)
	{
		*puiRow = pWindow->uiUly;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Clears from the specified column and row to the end of the row in
			the specified window
****************************************************************************/
void FTKAPI FTXWinClearLine(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiCol,
	FLMUINT			uiRow)
{
	f_mutexLock( pWindow->pScreen->hScreenMutex);
	ftxWinClearLine( pWindow, uiCol, uiRow);
	f_mutexUnlock( pWindow->pScreen->hScreenMutex);
}

/****************************************************************************
Desc:		Clears from the current cursor position to the end of the current
			line
****************************************************************************/
void FTKAPI FTXWinClearToEOL(
	FTX_WINDOW *	pWindow)
{
	f_mutexLock( pWindow->pScreen->hScreenMutex);
	ftxWinClearLine( pWindow,
		(FLMUINT)(pWindow->uiCurX - pWindow->uiOffset),
		(FLMUINT)(pWindow->uiCurY - pWindow->uiOffset));
	f_mutexUnlock( pWindow->pScreen->hScreenMutex);
}

/****************************************************************************
Desc:		Clears the canvas of the specified window starting at the requested
			row and column offset
****************************************************************************/
void FTKAPI FTXWinClearXY(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiCol,
	FLMUINT			uiRow)
{
	FLMUINT			uiSaveCol;
	FLMUINT			uiSaveRow;
	FLMUINT			uiLoop;

	f_mutexLock( pWindow->pScreen->hScreenMutex);

	uiSaveCol = pWindow->uiCurX;
	uiSaveRow = pWindow->uiCurY;

	ftxWinClearLine( pWindow, uiCol, uiRow);
	uiLoop = (FLMUINT)(uiRow + 1);

	while( uiLoop < pWindow->uiRows - pWindow->uiOffset)
	{
		ftxWinClearLine( pWindow, 0, uiLoop);
		uiLoop++;
	}

	pWindow->uiCurY = uiSaveRow;
	pWindow->uiCurX = uiSaveCol;

	f_mutexUnlock( pWindow->pScreen->hScreenMutex);
}

/****************************************************************************
Desc:		Clears the canvas area of the specified window
****************************************************************************/
void FTKAPI FTXWinClear(
	FTX_WINDOW *	pWindow)
{
	FTXWinClearXY( pWindow, 0, 0);
	FTXWinSetCursorPos( pWindow, 0, 0);
}

/****************************************************************************
Desc:		Draws a border around the canvas area of the specified window
****************************************************************************/
void FTKAPI FTXWinDrawBorder(
	FTX_WINDOW *	pWindow)
{
	FLMUINT			uiLoop;
	FLMBOOL			bScroll;
	FLMUINT			uiCols;
	FLMUINT			uiRows;

	f_mutexLock( pWindow->pScreen->hScreenMutex);

	uiCols = pWindow->uiCols;
	uiRows = pWindow->uiRows;

	if( (uiRows > 2 && uiCols > 2))
	{
		pWindow->bForceOutput = TRUE;

		pWindow->uiOffset = 0;
		bScroll = pWindow->bScroll;

		pWindow->uiOffset = 0;
		pWindow->bScroll = FALSE;

		ftxWinSetCursorPos( pWindow, 0, 0);
#if defined( FLM_WIN) || defined( FLM_NLM)
		ftxWinPrintChar( pWindow, (FLMUINT)201);
#else
		ftxWinPrintChar( pWindow, (FLMUINT)'+');
#endif

		ftxWinSetCursorPos( pWindow, (FLMUINT)(uiCols - 1), 0);
#if defined( FLM_WIN) || defined( FLM_NLM)
		ftxWinPrintChar( pWindow, (FLMUINT)187);
#else
		ftxWinPrintChar( pWindow, (FLMUINT)'+');
#endif

		ftxWinSetCursorPos( pWindow, 0, (FLMUINT)(uiRows - 1));
#if defined( FLM_WIN) || defined( FLM_NLM)
		ftxWinPrintChar( pWindow, (FLMUINT)200);
#else
		ftxWinPrintChar( pWindow, (FLMUINT)'+');
#endif

		ftxWinSetCursorPos( pWindow, (FLMUINT)(uiCols - 1),
			(FLMUINT)(uiRows - 1));
#if defined( FLM_WIN) || defined( FLM_NLM)
		ftxWinPrintChar( pWindow, (FLMUINT)188);
#else
		ftxWinPrintChar( pWindow, (FLMUINT)'+');
#endif

		for( uiLoop = 1; uiLoop < uiCols - 1; uiLoop++)
		{
			ftxWinSetCursorPos( pWindow, uiLoop, 0);
#if defined( FLM_WIN) || defined( FLM_NLM)
			ftxWinPrintChar( pWindow, (FLMUINT)205);
#else
			ftxWinPrintChar( pWindow, (FLMUINT)'-');
#endif

			ftxWinSetCursorPos( pWindow, uiLoop,
				(FLMUINT)(uiRows - 1));
#if defined( FLM_WIN) || defined( FLM_NLM)
			ftxWinPrintChar( pWindow, (FLMUINT)205);
#else
			ftxWinPrintChar( pWindow, (FLMUINT)'-');
#endif
		}

		for( uiLoop = 1; uiLoop < uiRows - 1; uiLoop++)
		{
			ftxWinSetCursorPos( pWindow, 0, uiLoop);
#if defined( FLM_WIN) || defined( FLM_NLM)
			ftxWinPrintChar( pWindow, (FLMUINT)186);
#else
			ftxWinPrintChar( pWindow, (FLMUINT)'|');
#endif

			ftxWinSetCursorPos( pWindow, (FLMUINT)(uiCols - 1),
				uiLoop);
#if defined( FLM_WIN) || defined( FLM_NLM)
			ftxWinPrintChar( pWindow, (FLMUINT)186);
#else
			ftxWinPrintChar( pWindow, (FLMUINT)'|');
#endif
		}

		pWindow->uiOffset = 1;
		pWindow->bScroll = bScroll;
		pWindow->bForceOutput = FALSE;

		ftxWinSetCursorPos( pWindow, 0, 0);
	}

	f_mutexUnlock( pWindow->pScreen->hScreenMutex);
}

/****************************************************************************
Desc:		Draws a border around the canvas area of the specified window
****************************************************************************/
void FTKAPI FTXWinSetTitle(
	FTX_WINDOW *	pWindow,
	const char *	pszTitle,
	eColorType		backgroundColor,
	eColorType		foregroundColor)
{
	FLMBOOL			bScroll = FALSE;
	FLMUINT			uiCols;
	FLMUINT			uiRows;
	FLMUINT			uiStrLen;
	eColorType		saveForegroundColor;
	eColorType		saveBackgroundColor;

	f_mutexLock( pWindow->pScreen->hScreenMutex);

	uiCols = pWindow->uiCols;
	uiRows = pWindow->uiRows;

	if( (uiRows > 2 && uiCols > 2))
	{
		pWindow->bForceOutput = TRUE;

		pWindow->uiOffset = 0;
		bScroll = pWindow->bScroll;

		pWindow->uiOffset = 0;
		pWindow->bScroll = FALSE;
		saveBackgroundColor = pWindow->backgroundColor;
		pWindow->backgroundColor = backgroundColor;
		saveForegroundColor = pWindow->foregroundColor;
		pWindow->foregroundColor = foregroundColor;
		uiStrLen = f_strlen( pszTitle);
		
		if( uiStrLen < uiCols)
		{
			ftxWinSetCursorPos( pWindow, (FLMUINT)((uiCols - uiStrLen) / 2), 0);
		}
		else
		{
			ftxWinSetCursorPos( pWindow, 0, 0);
		}

		while( *pszTitle != '\0')
		{
			ftxWinPrintChar( pWindow, (FLMUINT)*pszTitle);
			pszTitle++;
		}
		
		pWindow->backgroundColor = saveBackgroundColor;
		pWindow->foregroundColor = saveForegroundColor;
	}

	pWindow->uiOffset = 1;
	pWindow->bScroll = bScroll;
	pWindow->bForceOutput = FALSE;
	ftxWinSetCursorPos( pWindow, 0, 0);
	f_mutexUnlock( pWindow->pScreen->hScreenMutex);
}

/****************************************************************************
Desc:		Draws a border around the canvas area of the specified window
****************************************************************************/
void FTKAPI FTXWinSetHelp(
	FTX_WINDOW *	pWindow,
	const char *	pszHelp,
	eColorType		backgroundColor,
	eColorType		foregroundColor)
{
	FLMBOOL			bScroll = FALSE;
	FLMUINT			uiCols;
	FLMUINT			uiRows;
	FLMUINT			uiStrLen;
	eColorType		saveForegroundColor;
	eColorType		saveBackgroundColor;

	f_mutexLock( pWindow->pScreen->hScreenMutex);

	uiCols = pWindow->uiCols;
	uiRows = pWindow->uiRows;

	if( (uiRows > 2 && uiCols > 2))
	{
		pWindow->bForceOutput = TRUE;

		pWindow->uiOffset = 0;
		bScroll = pWindow->bScroll;

		pWindow->uiOffset = 0;
		pWindow->bScroll = FALSE;
		saveBackgroundColor = pWindow->backgroundColor;
		pWindow->backgroundColor = backgroundColor;
		saveForegroundColor = pWindow->foregroundColor;
		pWindow->foregroundColor = foregroundColor;

		uiStrLen = f_strlen( pszHelp);
		if( uiStrLen < uiCols)
		{
			ftxWinSetCursorPos( pWindow, (FLMUINT)((uiCols - uiStrLen) / 2), uiRows - 1);
		}
		else
		{
			ftxWinSetCursorPos( pWindow, 0, uiRows-1);
		}

		while( *pszHelp != '\0')
		{
			ftxWinPrintChar( pWindow, (FLMUINT)*pszHelp);
			pszHelp++;
		}
		
		pWindow->backgroundColor = saveBackgroundColor;
		pWindow->foregroundColor = saveForegroundColor;
	}

	pWindow->uiOffset = 1;
	pWindow->bScroll = bScroll;
	pWindow->bForceOutput = FALSE;
	ftxWinSetCursorPos( pWindow, 0, 0);
	f_mutexUnlock( pWindow->pScreen->hScreenMutex);
}

/****************************************************************************
Desc:		Tests the key buffer for an available key
****************************************************************************/
RCODE FTKAPI FTXWinTestKB(
	FTX_WINDOW *	pWindow)
{
	RCODE				rc = NE_FLM_OK;

	f_mutexLock( gv_pFtxInfo->hFtxMutex);

	if( !pWindow->bOpen || pWindow->pScreen->pWinCur != pWindow ||
		gv_pFtxInfo->pScreenCur != pWindow->pScreen)
	{
		rc = RC_SET( NE_FLM_EOF_HIT);
		goto Exit;
	}

	if( gv_pFtxInfo->puiKeyBuffer[ gv_pFtxInfo->uiCurKey] == 0)
	{
		rc = RC_SET( NE_FLM_EOF_HIT);
		goto Exit;
	}

Exit:

	f_mutexUnlock( gv_pFtxInfo->hFtxMutex);
	return( rc);
}

/****************************************************************************
Desc:		Gets a character from the keyboard
****************************************************************************/
RCODE FTKAPI FTXWinInputChar(
	FTX_WINDOW *	pWindow,
	FLMUINT *		puiChar)
{
	RCODE				rc = NE_FLM_OK;
	FLMBOOL			bLocked = FALSE;

	if( puiChar)
	{
		*puiChar = 0;
	}

	if( !pWindow)
	{
		rc = RC_SET( NE_FLM_INVALID_PARM);
		goto Exit;
	}

	if( !pWindow->bInitialized)
	{
		rc = RC_SET( NE_FLM_INVALID_PARM);
		goto Exit;
	}

	if( !pWindow->bOpen || pWindow->pScreen->pWinCur != pWindow)
	{
		rc = RC_SET( NE_FLM_EOF_HIT);
		goto Exit;
	}

	for( ;;)
	{
		f_mutexLock( gv_pFtxInfo->hFtxMutex);
		bLocked = TRUE;

		if( (gv_pFtxInfo->pbShutdown != NULL && 
			*(gv_pFtxInfo->pbShutdown) == TRUE) ||
			(pWindow->pScreen->pbShutdown != NULL &&
			*(pWindow->pScreen->pbShutdown) == TRUE))
		{
			rc = RC_SET( NE_FLM_EOF_HIT);
			goto Exit;
		}

		if( gv_pFtxInfo->pScreenCur == pWindow->pScreen)
		{
			if( gv_pFtxInfo->puiKeyBuffer[ gv_pFtxInfo->uiCurKey])
			{
				if( puiChar)
				{
					*puiChar = gv_pFtxInfo->puiKeyBuffer[ gv_pFtxInfo->uiCurKey];
				}
				gv_pFtxInfo->puiKeyBuffer[ gv_pFtxInfo->uiCurKey] = 0;
				gv_pFtxInfo->uiCurKey++;
				if( gv_pFtxInfo->uiCurKey >= FTX_KEYBUF_SIZE)
				{
					gv_pFtxInfo->uiCurKey = 0;
				}
				break;
			}
		}
		f_mutexUnlock( gv_pFtxInfo->hFtxMutex);
		bLocked = FALSE;
		(void)f_semWait( pWindow->pScreen->hKeySem, 1000);
	}

Exit:

	if( bLocked)
	{
		f_mutexUnlock( gv_pFtxInfo->hFtxMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:		Line editor routine
****************************************************************************/
RCODE FTKAPI FTXLineEdit(
	FTX_WINDOW *	pWindow,
	char *			pszBuffer,
	FLMUINT			uiBufSize,
	FLMUINT			uiMaxWidth,
	FLMUINT *		puiCharCount,
	FLMUINT *		puiTermChar)
{
	RCODE				rc = NE_FLM_OK;
	char				szLineBuf[ 256];
	char				szSnapBuf[ 256];
	FLMUINT			uiCharCount;
	FLMUINT			uiBufPos;
	FLMUINT			uiStartCol;
	FLMUINT			uiStartRow;
	FLMUINT			uiChar;
	FLMUINT			uiCursorOutputPos = 0;
	FLMUINT			uiNumRows;
	FLMUINT			uiNumCols;
	FLMUINT			uiSaveCursor;
	FLMUINT			uiLoop;
	FLMUINT			uiCharsLn;
	FLMUINT			uiOutputStart = 0;
	FLMUINT			uiCursorPos;
	FLMUINT			uiOutputEnd = 0;
	FLMBOOL			bDone;
	FLMBOOL			bInsert;
	FLMBOOL			bRefresh;
	FLMBOOL			bGotChar = FALSE;
	FLMBOOL			bSaveScroll = FALSE;

	if( puiCharCount)
	{
		*puiCharCount = 0;
	}

	uiSaveCursor = FTXWinGetCursorType( pWindow);
	FTXWinGetCanvasSize( pWindow, &uiNumCols, &uiNumRows);
	uiStartCol = FTXWinGetCurrCol( pWindow);
	uiStartRow = FTXWinGetCurrRow( pWindow);
	FTXWinGetScroll( pWindow, &bSaveScroll);

	if( uiBufSize < 2 || uiMaxWidth < 2 || (uiNumCols - uiStartCol) < 3)
	{
		return( 0);
	}

	FTXWinSetScroll( pWindow, FALSE);
	FTXWinSetFocus( pWindow);
	FTXRefresh();

	uiCharsLn = (FLMUINT)(uiNumCols - uiStartCol);
	if( uiCharsLn > uiMaxWidth)
	{
		uiCharsLn = uiMaxWidth;
	}

	f_memset( szLineBuf, (FLMBYTE)32, uiCharsLn);

	szLineBuf[ uiCharsLn] = '\0';
	pszBuffer[ uiBufSize - 1] = '\0';
	uiCharCount = f_strlen( pszBuffer);
	if( uiCharCount > 0)
	{
		bGotChar = TRUE;
		uiBufPos = uiCharCount;
		uiCursorPos = (uiBufPos < uiCharsLn) ? uiBufPos : (uiCharsLn - 1);
	}
	else
	{
		uiBufPos = 0;
		uiCursorPos = 0;
	}

	bDone = FALSE;
	bInsert = TRUE;
	bRefresh = FALSE;
	uiChar = 0;

	while( !bDone)
	{
		if( gv_pFtxInfo->pbShutdown && *(gv_pFtxInfo->pbShutdown))
		{
			pszBuffer[ 0] = '\0';
			uiCharCount = 0;
			rc = RC_SET( NE_FLM_EOF_HIT);
			break;
		}

		if( !bGotChar)
		{
			if( RC_BAD( rc = FTXWinInputChar( pWindow, &uiChar)))
			{
				goto Exit;
			}
			bGotChar = TRUE;

			switch( uiChar)
			{
				case FKB_HOME:
				{
					uiBufPos = 0;
					uiCursorPos = 0;
					break;
				}
				case FKB_LEFT:
				{
					if( uiBufPos > 0)
					{
						uiBufPos--;
						if( uiCursorPos > 0)
						{
							uiCursorPos--;
						}
					}
					break;
				}
				case FKB_RIGHT:
				{
					if( uiBufPos < uiCharCount)
					{
						uiBufPos++;
						if( uiCursorPos < (uiCharsLn - 1))
						{
							uiCursorPos++;
						}
					}
					break;
				}
				case FKB_END:
				{
					if( uiBufPos != uiCharCount)
					{
						if( uiCharCount < (uiCharsLn - 1))
						{
							uiCursorPos = uiCharCount;
						}
						else
						{
							uiCursorPos = (FLMUINT)(uiCharsLn - 1);
						}
						uiBufPos = uiCharCount;
					}
					break;
				}
				case FKB_CTRL_LEFT:
				{
					if( uiBufPos > 0)
					{
						if( pszBuffer[ uiBufPos - 1] == ' ')
						{
							if( uiCursorPos > 0)
							{
								uiCursorPos--;
							}
							uiBufPos--;
						}
						
						while( uiBufPos > 0 && pszBuffer[ uiBufPos] == ' ')
						{
							uiBufPos--;
							if( uiCursorPos > 0)
							{
								uiCursorPos--;
							}
						}
						
						while( uiBufPos > 0 && pszBuffer[ uiBufPos] != ' ')
						{
							uiBufPos--;
							if( uiCursorPos > 0)
							{
								uiCursorPos--;
							}
						}
					}
					
					if( uiBufPos > 0 && pszBuffer[ uiBufPos] == ' ' &&
						uiBufPos < uiCharCount)
					{
						uiBufPos++;
						if( uiCursorPos < (uiCharsLn - 1))
						{
							uiCursorPos++;
						}
					}
					
					break;
				}
				
				case FKB_CTRL_RIGHT:
				{
					if( uiBufPos < uiCharCount)
					{
						while( uiBufPos < uiCharCount && pszBuffer[ uiBufPos] != ' ')
						{
							uiBufPos++;
							if( uiCursorPos < (uiCharsLn - 1))
							{
								uiCursorPos++;
							}
						}
						
						while( uiBufPos < uiCharCount && pszBuffer[ uiBufPos] == ' ')
						{
							uiBufPos++;
							if( uiCursorPos < (uiCharsLn - 1))
							{
								uiCursorPos++;
							}
						}
					}
					
					break;
				}
				
				case FKB_INSERT:
				{
					if( bInsert == TRUE)
					{
						bInsert = FALSE;
						FTXWinSetCursorType( pWindow,
							FLM_CURSOR_VISIBLE | FLM_CURSOR_BLOCK);
					}
					else
					{
						bInsert = TRUE;
						FTXWinSetCursorType( pWindow,
							FLM_CURSOR_VISIBLE | FLM_CURSOR_UNDERLINE);
					}
					
					ftxCursorUpdate();
					break;
				}
				
				case FKB_DELETE:
				{
					if( uiBufPos < uiCharCount)
					{
						f_memmove( &(pszBuffer[ uiBufPos]),
							&(pszBuffer[ uiBufPos + 1]), uiCharCount - uiBufPos);
						uiCharCount--;
					}
					break;
				}
				
				case FKB_BACKSPACE:
				{
					if( uiBufPos > 0)
					{
						if( uiCursorPos > 0)
						{
							uiCursorPos--;
						}
						uiBufPos--;
						f_memmove( &(pszBuffer[ uiBufPos]),
							&(pszBuffer[ uiBufPos + 1]), uiCharCount - uiBufPos);
						uiCharCount--;
					}
					break;
				}
				
				case FKB_CTRL_B:
				{
					if( uiBufPos > 0)
					{
						uiCharCount -= uiBufPos;
						f_memmove( pszBuffer,
							&(pszBuffer[ uiBufPos]), uiCharCount + 1);
						uiBufPos = 0;
						uiCursorPos = 0;
					}
					break;
				}
				
				case FKB_CTRL_D:
				{
					if( uiBufPos < uiCharCount)
					{
						uiCharCount = uiBufPos;
						pszBuffer[ uiCharCount] = '\0';
					}
					break;
				}
				
				default:
				{
					if( (uiChar & 0xFF00) == 0)
					{
						if( bInsert && uiBufPos < uiCharCount &&
							uiCharCount < (uiBufSize - 1))
						{
							for( uiLoop = 0; uiLoop < uiCharCount - uiBufPos; uiLoop++)
							{
								pszBuffer[ uiCharCount - uiLoop] =
									pszBuffer[ uiCharCount - uiLoop - 1];
							}

							pszBuffer[ uiBufPos] = (char)uiChar;
							if( uiCursorPos < (uiCharsLn - 1))
							{
								uiCursorPos++;
							}
							pszBuffer[ ++uiCharCount] = '\0';
							uiBufPos++;
						}
						else if( (uiBufPos < uiCharCount && !bInsert) ||
							uiCharCount < (uiBufSize - 1))
						{
							pszBuffer[ uiBufPos] = (char)uiChar;
							if( uiBufPos == uiCharCount)
							{
								pszBuffer[ ++uiCharCount] = '\0';
							}
							
							if( uiCursorPos < (uiCharsLn - 1))
							{
								uiCursorPos++;
							}
							uiBufPos++;
						}
					}
					else if( uiChar & 0xFF00)
					{
						bDone = TRUE;
						bGotChar = FALSE;
					}
				}
			}
		}

		if( bGotChar)
		{
			uiOutputStart = (FLMUINT)(uiBufPos - uiCursorPos);
			uiOutputEnd = (FLMUINT)(uiOutputStart + uiCharsLn);
			
			if( uiOutputEnd > uiCharCount)
			{
				uiOutputEnd = uiCharCount;
			}

			f_memset( szSnapBuf, (FLMBYTE)32, uiCharsLn);
			szSnapBuf[ uiCharsLn] = '\0';
			f_memmove( szSnapBuf, &(pszBuffer[ uiOutputStart]),
				(FLMUINT)(uiOutputEnd - uiOutputStart));

			uiCursorOutputPos = 0;
			uiLoop = 0;
			while( uiLoop < uiCharsLn)
			{
				if( szSnapBuf[ uiLoop] != szLineBuf[ uiLoop])
				{
					bRefresh = TRUE;
					uiCursorOutputPos = uiLoop;
					break;
				}
				uiLoop++;
			}

			uiLoop = uiCharsLn;
			while( uiLoop > uiCursorOutputPos)
			{
				if( szSnapBuf[ uiLoop - 1] != szLineBuf[ uiLoop - 1])
				{
					bRefresh = TRUE;
					break;
				}
				uiLoop--;
			}
			szSnapBuf[ uiLoop] = '\0';
			bGotChar = FALSE;
		}

		if( bRefresh)
		{
			f_memset( szLineBuf, (FLMBYTE)32, uiCharsLn);
			szLineBuf[ uiCharsLn] = '\0';
			f_memmove( szLineBuf, &(pszBuffer[ uiOutputStart]),
				(FLMUINT)(uiOutputEnd - uiOutputStart));

			FTXWinSetCursorPos( pWindow,
				(FLMUINT)(uiStartCol + uiCursorOutputPos), uiStartRow);
			FTXWinPrintStr( pWindow, &(szSnapBuf[ uiCursorOutputPos]));
			FTXWinSetCursorPos( pWindow,
				(FLMUINT)(uiStartCol + uiCursorPos), uiStartRow);

			FTXRefresh();
			bRefresh = FALSE;
		}
		else
		{
			FLMUINT	uiTmpCol = FTXWinGetCurrCol( pWindow);
			FLMUINT	uiTmpRow = FTXWinGetCurrRow( pWindow);

			if( uiTmpCol != uiStartCol + uiCursorPos || uiTmpRow != uiStartRow)
			{
				FTXWinSetCursorPos( pWindow, (FLMUINT)(uiStartCol + uiCursorPos),
					uiStartRow);
				ftxCursorUpdate();
			}
		}
	}

	if( puiTermChar)
	{
		*puiTermChar = uiChar;
	}

	if( puiCharCount)
	{
		*puiCharCount = uiCharCount;
	}

Exit:

	FTXWinSetCursorType( pWindow, uiSaveCursor);
	FTXWinSetScroll( pWindow, bSaveScroll);

	return( rc);
}

/****************************************************************************
Desc:		Line editor routine which assumes some defaults
****************************************************************************/
FLMUINT FTKAPI FTXLineEd(
	FTX_WINDOW *	pWindow,
	char *			pszBuffer,
	FLMUINT			uiBufSize)
{
	FLMUINT		uiTermChar;
	FLMUINT		uiStartCol;
	FLMUINT		uiStartRow;
	FLMUINT		uiCharsInput;
	FLMBOOL		bDone = FALSE;

	uiStartCol = FTXWinGetCurrCol( pWindow);
	uiStartRow = FTXWinGetCurrRow( pWindow);

	while( !bDone)
	{
		if( gv_pFtxInfo->pbShutdown && *(gv_pFtxInfo->pbShutdown))
		{
			pszBuffer[ 0] = '\0';
			uiCharsInput = 0;
			break;
		}

		pszBuffer[ 0] = '\0';
		if( RC_BAD( FTXLineEdit( pWindow, pszBuffer, uiBufSize, 
			255, &uiCharsInput, &uiTermChar)))
		{
			uiCharsInput = 0;
			*pszBuffer = '\0';
			goto Exit;
		}

		switch( uiTermChar)
		{
			case FKB_ENTER:
			{
				FTXWinPrintChar( pWindow, '\n');
				bDone = TRUE;
				break;
			}
			
			case FKB_ESC:
			{
				pszBuffer[ 0] = '\0';
				bDone = TRUE;
				break;
			}
			
			default:
			{
				FTXWinClearLine( pWindow, uiStartCol, uiStartRow);
				FTXWinSetCursorPos( pWindow, uiStartCol, uiStartRow);
				break;
			}
		}
	}

Exit:

	return( uiCharsInput);
}

/****************************************************************************
Desc: Displays a message window
*****************************************************************************/
RCODE FTKAPI FTXMessageWindow(
	FTX_SCREEN *		pScreen,
	eColorType			backgroundColor,
	eColorType			foregroundColor,
	const char *		pszMessage1,
	const char *		pszMessage2,
	FTX_WINDOW **		ppWindow)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiNumCols;
	FLMUINT			uiNumRows;
	FLMUINT			uiNumWinRows = 10;
	FLMUINT			uiNumWinCols;
	FLMUINT			uiNumCanvCols;
	char				pucTmpBuf[ 128];
	FLMUINT			uiMessageLen;
	FTX_WINDOW *	pWindow = NULL;

	if( RC_BAD( rc = FTXScreenGetSize( pScreen, &uiNumCols, &uiNumRows)))
	{
		goto Exit;
	}

	uiNumWinCols = uiNumCols - 8;

	if( RC_BAD( rc = FTXWinInit( pScreen, uiNumWinCols, uiNumWinRows, &pWindow)))
	{
		goto Exit;
	}

	FTXWinSetScroll( pWindow, FALSE);
	FTXWinSetCursorType( pWindow, FLM_CURSOR_INVISIBLE);
	FTXWinSetBackFore( pWindow, backgroundColor, foregroundColor);
	FTXWinClear( pWindow);
	FTXWinDrawBorder( pWindow);
	FTXWinMove( pWindow, (FLMUINT)((uiNumCols - uiNumWinCols) / 2),
		(FLMUINT)((uiNumRows - uiNumWinRows) / 2));
	FTXWinGetCanvasSize( pWindow, &uiNumCanvCols, NULL);

	if( RC_BAD( rc = FTXWinOpen( pWindow)))
	{
		goto Exit;
	}

	if( pszMessage1)
	{
		f_strncpy( pucTmpBuf, pszMessage1, uiNumCanvCols);
		pucTmpBuf[ uiNumCanvCols] = '\0';
		uiMessageLen = f_strlen( pucTmpBuf);

		FTXWinSetCursorPos( pWindow,
			(FLMUINT)((uiNumCanvCols - uiMessageLen) / 2), 3);
		FTXWinPrintf( pWindow, "%s", pucTmpBuf);
	}

	if( pszMessage2)
	{
		f_strncpy( pucTmpBuf, pszMessage2, uiNumCanvCols);
		pucTmpBuf[ uiNumCanvCols] = '\0';
		uiMessageLen = f_strlen( pucTmpBuf);

		FTXWinSetCursorPos( pWindow,
			(FLMUINT)((uiNumCanvCols - uiMessageLen) / 2), 4);
		FTXWinPrintf( pWindow, "%s", pucTmpBuf);
	}

	FTXRefresh();
	
Exit:

	if( RC_BAD( rc) && pWindow)
	{
		*ppWindow = NULL;
		FTXWinFree( &pWindow);
	}
	else
	{
		*ppWindow = pWindow;
	}

	return( rc);
}


/****************************************************************************
Desc: Displays a dialog-style message box
*****************************************************************************/
RCODE FTKAPI FTXDisplayMessage(
	FTX_SCREEN *		pScreen,
	eColorType			backgroundColor,
	eColorType			foregroundColor,
	const char *		pszMessage1,
	const char *		pszMessage2,
	FLMUINT *			puiTermChar)
{
	RCODE				rc = NE_FLM_OK;
	FTX_WINDOW *	pWindow = NULL;

	if( puiTermChar)
	{
		*puiTermChar = 0;
	}

	if( RC_BAD( rc = FTXMessageWindow( pScreen, backgroundColor, foregroundColor,
						pszMessage1, pszMessage2, &pWindow)))
	{
		goto Exit;
	}

	for( ;;)
	{
		if( gv_pFtxInfo->pbShutdown && *(gv_pFtxInfo->pbShutdown))
		{
			rc = RC_SET( NE_FLM_EOF_HIT);
			goto Exit;
		}

		if( RC_OK( FTXWinTestKB( pWindow)))
		{
			FLMUINT		uiChar;

			FTXWinInputChar( pWindow, &uiChar);

			if( uiChar == FKB_ESCAPE || uiChar == FKB_ENTER)
			{
				if( puiTermChar)
				{
					*puiTermChar = uiChar;
				}
				break;
			}

		}
		else
		{
			f_sleep( 10);
		}
	}

Exit:

	if( pWindow)
	{
		FTXWinFree( &pWindow);
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FTKAPI FTXGetInput(
	FTX_SCREEN *	pScreen,
	const char *	pszMessage,
	char *			pszResponse,
	FLMUINT			uiMaxRespLen,
	FLMUINT *		puiTermChar)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiNumCols;
	FLMUINT			uiNumRows;
	FLMUINT			uiNumWinRows = 3;
	FLMUINT			uiNumWinCols;
	FTX_WINDOW *	pWindow = NULL;

	if( RC_BAD( rc = FTXScreenGetSize( pScreen, &uiNumCols, &uiNumRows)))
	{
		goto Exit;
	}

	uiNumWinCols = uiNumCols - 8;

	if( RC_BAD( rc = FTXWinInit( pScreen, uiNumWinCols, uiNumWinRows, &pWindow)))
	{
		goto Exit;
	}

	FTXWinSetScroll( pWindow, FALSE);
	FTXWinSetCursorType( pWindow, FLM_CURSOR_UNDERLINE);
	FTXWinSetBackFore( pWindow, FLM_CYAN, FLM_WHITE);
	FTXWinClear( pWindow);
	FTXWinDrawBorder( pWindow);
	FTXWinMove( pWindow, (uiNumCols - uiNumWinCols) / 2,
		(uiNumRows - uiNumWinRows) / 2);

	if( RC_BAD( rc = FTXWinOpen( pWindow)))
	{
		goto Exit;
	}

	FTXWinClear( pWindow);
	FTXWinPrintf( pWindow, "%s: ", pszMessage);

	if( RC_BAD( rc = FTXLineEdit( pWindow, pszResponse, uiMaxRespLen, 
			uiMaxRespLen, NULL, puiTermChar)))
	{
		goto Exit;
	}

Exit:

	if( pWindow)
	{
		FTXWinFree( &pWindow);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI FTXWinGetCanvasSize(
	FTX_WINDOW *	pWindow,
	FLMUINT *		puiNumCols,
	FLMUINT *		puiNumRows)
{
	f_assert( pWindow);
	
	if( puiNumCols)
	{
		*puiNumCols = (FLMUINT)(pWindow->uiCols - ((FLMUINT)2 * pWindow->uiOffset));
	}
	if( puiNumRows)
	{
		*puiNumRows = (FLMUINT)(pWindow->uiRows - ((FLMUINT)2 * pWindow->uiOffset));
	}
}

/****************************************************************************
Desc:		Sets or changes the appearance of the cursor in the specified
			window.
****************************************************************************/
void FTKAPI FTXWinSetCursorType(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiType)
{
	f_mutexLock( pWindow->pScreen->hScreenMutex);
	pWindow->uiCursorType = uiType;
	pWindow->pScreen->bUpdateCursor = TRUE;
	f_mutexUnlock( pWindow->pScreen->hScreenMutex);
}

/****************************************************************************
Desc:		Retrieves the cursor type of the specified window
****************************************************************************/
FLMUINT FTKAPI FTXWinGetCursorType(
	FTX_WINDOW *	pWindow)
{
	return( pWindow->uiCursorType);
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI FTXSetShutdownFlag(
	FLMBOOL *		pbShutdownFlag)
{
	f_mutexLock( gv_pFtxInfo->hFtxMutex);
	gv_pFtxInfo->pbShutdown = pbShutdownFlag;
	f_mutexUnlock( gv_pFtxInfo->hFtxMutex);
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI FTXWinGetSize(
	FTX_WINDOW *	pWindow,
	FLMUINT *		puiNumCols,
	FLMUINT *		puiNumRows)
{
	if( puiNumCols)
	{
		*puiNumCols = pWindow->uiCols;
	}

	if( puiNumRows)
	{
		*puiNumRows = pWindow->uiRows;
	}
}

/****************************************************************************
Desc: Retrieves the current cursor row in the specified window
****************************************************************************/
FLMUINT FTKAPI FTXWinGetCurrRow(
	FTX_WINDOW *	pWindow)
{
	return( (FLMUINT)(pWindow->uiCurY - pWindow->uiOffset));
}

/****************************************************************************
Desc:		Retrieves the current cursor column in the specified window
****************************************************************************/
FLMUINT FTKAPI FTXWinGetCurrCol(
	FTX_WINDOW *	pWindow)
{
	return( (FLMUINT)(pWindow->uiCurX - pWindow->uiOffset));
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI FTXWinGetCursorPos(
	FTX_WINDOW *	pWindow,
	FLMUINT *		puiCol,
	FLMUINT *		puiRow)
{
	f_assert( pWindow);
	
	if( puiCol != NULL)
	{
		*puiCol = (FLMUINT)(pWindow->uiCurX - pWindow->uiOffset);
	}

	if( puiRow != NULL)
	{
		*puiRow = (FLMUINT)(pWindow->uiCurY - pWindow->uiOffset);
	}
}

/****************************************************************************
Desc: Synchronizes the "camera-ready" display image with the "in-memory"
		image
****************************************************************************/
FSTATIC void ftxSyncImage( void)
{
	FTX_WINDOW *	pWin;
	FTX_SCREEN *	pScreenCur;
	char *			pszWTBuf;
	char *			pszSBuf;
	FLMBYTE *		pucWTBackAttrib;
	FLMBYTE *		pucWTForeAttrib;
	FLMBYTE *		pucSBackAttrib;
	FLMBYTE *		pucSForeAttrib;
	FLMUINT			uiLoop;
	FLMUINT			uiOffset;

	pScreenCur = gv_pFtxInfo->pScreenCur;

	ftxWinReset( pScreenCur->pWinImage);
	pWin = pScreenCur->pWinCur;

	if( pWin)
	{
		while( pWin->pWinNext)
		{
			pWin = pWin->pWinNext;
		}
	}

	while( pWin != NULL)
	{
		if( pWin->bOpen)
		{
			pszSBuf = pWin->pszBuffer;
			pucSBackAttrib = pWin->pucBackAttrib;
			pucSForeAttrib = pWin->pucForeAttrib;

			uiOffset = (FLMUINT)(((FLMUINT)pScreenCur->pWinImage->uiCols *
				(FLMUINT)pWin->uiUly) + (FLMUINT)pWin->uiUlx);

			pszWTBuf = pScreenCur->pWinImage->pszBuffer + uiOffset;
			pucWTBackAttrib = pScreenCur->pWinImage->pucBackAttrib + uiOffset;
			pucWTForeAttrib = pScreenCur->pWinImage->pucForeAttrib + uiOffset;

			for( uiLoop = 0; uiLoop < pWin->uiRows; uiLoop++)
			{
				f_memmove( pszWTBuf, pszSBuf, pWin->uiCols);
				f_memmove( pucWTBackAttrib, pucSBackAttrib, pWin->uiCols);
				f_memmove( pucWTForeAttrib, pucSForeAttrib, pWin->uiCols);

				pszSBuf += pWin->uiCols;
				pucSBackAttrib += pWin->uiCols;
				pucSForeAttrib += pWin->uiCols;

				pszWTBuf += pScreenCur->pWinImage->uiCols;
				pucWTBackAttrib += pScreenCur->pWinImage->uiCols;
				pucWTForeAttrib += pScreenCur->pWinImage->uiCols;
			}
		}
		pWin = pWin->pWinPrev;
	}
}

/****************************************************************************
Desc:		Win32 display update
****************************************************************************/
#if defined( FLM_WIN)
FSTATIC void ftxWin32Refresh( void)
{
	PCHAR_INFO			paCell;
	COORD					size;
	COORD					coord;
	SMALL_RECT			region;
	HANDLE				hStdOut;
	FLMUINT				uiLoop;
	FLMUINT				uiSubloop;
	FLMUINT				uiOffset;
	FLMUINT				uiLeft = 0;
	FLMUINT				uiRight = 0;
	FLMUINT				uiTop = 0;
	FLMUINT				uiBottom = 0;
	FLMBOOL				bTopSet = FALSE;
	FLMBOOL				bLeftSet = FALSE;
	FLMBOOL				bChanged = FALSE;
	FTX_WINDOW *		pWinImage;
	FTX_WINDOW *		pWinScreen;

	ftxSyncImage();
	pWinImage = gv_pFtxInfo->pScreenCur->pWinImage;
	pWinScreen = gv_pFtxInfo->pScreenCur->pWinScreen;

	for( uiLoop = 0; uiLoop < pWinImage->uiRows; uiLoop++)
	{
		for( uiSubloop = 0; uiSubloop < pWinImage->uiCols; uiSubloop++)
		{
			uiOffset = (FLMUINT)((uiLoop * (FLMUINT)(pWinImage->uiCols)) + uiSubloop);

			if( pWinImage->pszBuffer[ uiOffset] !=
					pWinScreen->pszBuffer[ uiOffset] ||
				pWinImage->pucForeAttrib[ uiOffset] !=
					pWinScreen->pucForeAttrib[ uiOffset] ||
				pWinImage->pucBackAttrib[ uiOffset] !=
					pWinScreen->pucBackAttrib[ uiOffset])
			{
				pWinScreen->pszBuffer[ uiOffset] =
					pWinImage->pszBuffer[ uiOffset];
				pWinScreen->pucForeAttrib[ uiOffset] =
					pWinImage->pucForeAttrib[ uiOffset];
				pWinScreen->pucBackAttrib[ uiOffset] =
					pWinImage->pucBackAttrib[ uiOffset];

				if( uiSubloop > uiRight)
				{
					uiRight = uiSubloop;
				}
				
				if( uiLoop > uiBottom)
				{
					uiBottom = uiLoop;
				}
				
				if( !bTopSet)
				{
					uiTop = uiLoop;
					bTopSet = TRUE;
				}
				
				if( !bLeftSet || uiLeft > uiSubloop)
				{
					uiLeft = uiSubloop;
					bLeftSet = TRUE;
				}
				
				if( !bChanged)
				{
					bChanged = TRUE;
				}
			}

			paCell = &(gv_pFtxInfo->pCells[ ((uiLoop + pWinImage->uiUly) *
				pWinScreen->uiCols) + (uiSubloop + pWinImage->uiUlx)]);
			paCell->Char.AsciiChar = pWinImage->pszBuffer[ uiOffset];
			paCell->Attributes =
				(ftxMapFlmColorToWin32( 
					(eColorType)pWinImage->pucForeAttrib[ uiOffset]) & 0x8F) |
				((ftxMapFlmColorToWin32( 
					(eColorType)pWinImage->pucBackAttrib[ uiOffset]) << 4) & 0x7F);
		}
	}

	if( bChanged)
	{
		if( (hStdOut = GetStdHandle( STD_OUTPUT_HANDLE)) ==
			INVALID_HANDLE_VALUE)
		{
			goto Exit;
		}

		size.X = (SHORT)pWinScreen->uiCols;
		size.Y = (SHORT)pWinScreen->uiRows;
		coord.X = (SHORT)uiLeft;
		coord.Y = (SHORT)uiTop;
		region.Left = (SHORT)uiLeft;
		region.Right = (SHORT)uiRight;
		region.Top = (SHORT)uiTop;
		region.Bottom = (SHORT)uiBottom;
		WriteConsoleOutput( hStdOut, gv_pFtxInfo->pCells, size, coord, &region);
	}

Exit:

	return;
}
#endif

/****************************************************************************
Desc:		NLM display update
****************************************************************************/
#if defined( FLM_NLM)
FSTATIC void ftxNLMRefresh( void)
{
	FLMUINT				uiLoop;
	FLMUINT				uiSubLoop;
	FLMUINT				uiOffset;
	FTX_WINDOW *		pWinImage;
	FTX_WINDOW *		pWinScreen;
	FLMBOOL				bModified;
	FLMUINT				uiCnt;
	FLMUINT				uiStartColumn;
	FLMUINT				uiStartOffset;
	char *				pucStartValue;
	char					attribute;
	char					ucStartAttr;

	ftxSyncImage();
	pWinImage = gv_pFtxInfo->pScreenCur->pWinImage;
	pWinScreen = gv_pFtxInfo->pScreenCur->pWinScreen;

	for( uiLoop = 0; uiLoop < (FLMUINT)pWinImage->uiRows; uiLoop++)
	{
		bModified = FALSE;
		for( uiSubLoop = 0; uiSubLoop < pWinImage->uiCols; uiSubLoop++)
		{
			uiOffset = (FLMUINT)((uiLoop * (FLMUINT)(pWinImage->uiCols)) + uiSubLoop);
			if((pWinImage->pszBuffer[ uiOffset] !=
					pWinScreen->pszBuffer[ uiOffset] ||
				pWinImage->pucForeAttrib[ uiOffset] !=
					pWinScreen->pucForeAttrib[ uiOffset] ||
				pWinImage->pucBackAttrib[ uiOffset] !=
					pWinScreen->pucBackAttrib[ uiOffset]))
			{
				attribute = (pWinImage->pucBackAttrib[ uiOffset] << 4) +
						pWinImage->pucForeAttrib[ uiOffset];
				if (!bModified || attribute != ucStartAttr)
				{
					if (bModified)
					{
						DisplayScreenTextWithAttribute( gv_pFtxInfo->hScreen,
							uiLoop, uiStartColumn, uiCnt, ucStartAttr,
							pucStartValue);
					}
					ucStartAttr = attribute;
					uiCnt = 0;
					uiStartOffset = uiOffset;
					uiStartColumn = uiSubLoop;
					bModified = TRUE;
					pucStartValue = &pWinImage->pszBuffer[ uiOffset];
				}
				uiCnt++;
			}
			else
			{
				if (bModified)
				{
					bModified = FALSE;
					(void)DisplayScreenTextWithAttribute( gv_pFtxInfo->hScreen,
						uiLoop, uiStartColumn, uiCnt, ucStartAttr,
						pucStartValue);
				}
			}
		}
		if (bModified)
		{
			bModified = FALSE;
			DisplayScreenTextWithAttribute( gv_pFtxInfo->hScreen,
				uiLoop, uiStartColumn, uiCnt, ucStartAttr, pucStartValue);
		}
	}
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_UNIX
FSTATIC void ftxRefresh( void)
{
	char *			pszWTBuf;
	char *			pszSBuf;
	FLMBYTE *		pucWTBackAttrib;
	FLMBYTE *		pucWTForeAttrib;
	FLMBYTE *		pucSBackAttrib;
	FLMBYTE *		pucSForeAttrib;
	FLMUINT			uiChangeStart = 0;
	FLMUINT			uiChangeEnd = 0;
	FLMUINT			uiSaveChar;
	FLMBOOL			bChange;
	FLMUINT			uiLoop;
	FLMUINT			uiSubloop;
	FLMUINT			uiTempAttrib;
	FTX_WINDOW *	pWinImage;
	FTX_WINDOW *	pWinScreen;

	ftxSyncImage();
	pWinImage = gv_pFtxInfo->pScreenCur->pWinImage;
	pWinScreen = gv_pFtxInfo->pScreenCur->pWinScreen;

	gv_pFtxInfo->uiCursorType = FLM_CURSOR_INVISIBLE;
	ftxDisplaySetCursorType( gv_pFtxInfo->uiCursorType);

	pszSBuf = pWinScreen->pszBuffer;
	pucSBackAttrib = pWinScreen->pucBackAttrib;
	pucSForeAttrib = pWinScreen->pucForeAttrib;

	pszWTBuf = pWinImage->pszBuffer;
	pucWTBackAttrib = pWinImage->pucBackAttrib;
	pucWTForeAttrib = pWinImage->pucForeAttrib;

	for( uiLoop = 0; uiLoop < pWinScreen->uiRows; uiLoop++)
	{
		uiSubloop = 0;
		while( uiSubloop < pWinScreen->uiCols)
		{
			bChange = FALSE;
			if( pszSBuf[ uiSubloop] != pszWTBuf[ uiSubloop] ||
				pucSBackAttrib[ uiSubloop] != pucWTBackAttrib[ uiSubloop] ||
				pucSForeAttrib[ uiSubloop] != pucWTForeAttrib[ uiSubloop])
			{
				bChange = TRUE;
				uiChangeStart = uiSubloop;
				uiChangeEnd = uiSubloop;

				while( pucWTBackAttrib[ uiChangeStart] ==
					pucWTBackAttrib[ uiSubloop] &&
					pucWTForeAttrib[ uiChangeStart] == pucWTForeAttrib[ uiSubloop] &&
					uiSubloop < pWinScreen->uiCols)
				{
					if( pszSBuf[ uiSubloop] != pszWTBuf[ uiSubloop] ||
						pucSBackAttrib[ uiSubloop] != pucWTBackAttrib[ uiSubloop] ||
						pucSForeAttrib[ uiSubloop] != pucWTForeAttrib[ uiSubloop])
					{
						uiChangeEnd = uiSubloop;
					}
					pszSBuf[ uiSubloop] = pszWTBuf[ uiSubloop];
					pucSBackAttrib[ uiSubloop] = pucWTBackAttrib[ uiSubloop];
					pucSForeAttrib[ uiSubloop] = pucWTForeAttrib[ uiSubloop];
					uiSubloop++;
				}
				uiSubloop--;
			}

			if( bChange)
			{
				ftxDisplaySetCursorPos( uiChangeStart, uiLoop);
				uiSaveChar = pszSBuf[ uiChangeEnd + 1];
				pszSBuf[ uiChangeEnd + 1] = '\0';
				uiTempAttrib = (pucSBackAttrib [uiChangeStart] << 4) +
						pucSForeAttrib [uiChangeStart];
				ftxDisplaySetBackFore( (eColorType)pucSBackAttrib[ uiChangeStart],
					(eColorType)pucSForeAttrib[ uiChangeStart]);
				ftxDisplayStrOut( &(pszSBuf[ uiChangeStart]), uiTempAttrib);
				pszSBuf[ uiChangeEnd + 1] = (char)uiSaveChar;
			}

			uiSubloop++;
		}

		pszSBuf += pWinScreen->uiCols;
		pucSBackAttrib += pWinScreen->uiCols;
		pucSForeAttrib += pWinScreen->uiCols;

		pszWTBuf += pWinImage->uiCols;
		pucWTBackAttrib += pWinImage->uiCols;
		pucWTForeAttrib += pWinImage->uiCols;
	}
}
#endif

/****************************************************************************
Desc:		Initializes / resets a window object
****************************************************************************/
FSTATIC void ftxWinReset(
	FTX_WINDOW *	pWindow)
{
	FLMUINT			uiSize;

	uiSize = (FLMUINT)(pWindow->uiRows * pWindow->uiCols);

	f_memset( pWindow->pszBuffer, (FLMBYTE)' ', uiSize);
	f_memset( pWindow->pucBackAttrib, (FLMBYTE)pWindow->pScreen->backgroundColor, uiSize);
	f_memset( pWindow->pucForeAttrib, (FLMBYTE)pWindow->pScreen->foregroundColor, uiSize);

	pWindow->backgroundColor = pWindow->pScreen->backgroundColor;
	pWindow->foregroundColor = pWindow->pScreen->foregroundColor;

	pWindow->uiCurX = pWindow->uiOffset;
	pWindow->uiCurY = pWindow->uiOffset;
}

/****************************************************************************
Desc:		Low-level routine for freeing a screen object
****************************************************************************/
FSTATIC void ftxScreenFree(
	FTX_SCREEN *	pScreen)
{
	FTX_WINDOW *	pWin;

	while( (pWin = pScreen->pWinCur) != NULL)
	{
		ftxWinFree( pWin);
	}

	if( pScreen == gv_pFtxInfo->pScreenCur)
	{
		gv_pFtxInfo->pScreenCur = pScreen->pScreenNext;
		if( gv_pFtxInfo->pScreenCur)
		{
			gv_pFtxInfo->pScreenCur->pScreenPrev = NULL;
		}
	}
	else
	{
		if( pScreen->pScreenNext)
		{
			pScreen->pScreenNext->pScreenPrev = pScreen->pScreenPrev;
		}

		if( pScreen->pScreenPrev)
		{
			pScreen->pScreenPrev->pScreenNext = pScreen->pScreenNext;
		}
	}

	f_mutexDestroy( &(pScreen->hScreenMutex));
	f_semDestroy( &(pScreen->hKeySem));
	f_free( &pScreen);
}

/****************************************************************************
Desc:		Low-level routine for freeing a window object
****************************************************************************/
FSTATIC void ftxWinFree(
	FTX_WINDOW *	pWindow)
{
	FTX_WINDOW *	pWin;

	if( pWindow->bOpen)
	{
		ftxWinClose( pWindow);
	}

	pWin = pWindow->pScreen->pWinCur;
	while( pWin != pWindow)
	{
		pWin = pWin->pWinNext;
		if( pWin == NULL)
		{
			break;
		}
	}

	if( pWin)
	{
		if( pWin == pWindow->pScreen->pWinCur)
		{
			pWindow->pScreen->pWinCur = pWin->pWinNext;
			if( pWindow->pScreen->pWinCur)
			{
				pWindow->pScreen->pWinCur->pWinPrev = NULL;
			}
		}
		else
		{
			if( pWin->pWinNext)
			{
				pWin->pWinNext->pWinPrev = pWin->pWinPrev;
			}

			if( pWin->pWinPrev)
			{
				pWin->pWinPrev->pWinNext = pWin->pWinNext;
			}
		}
	}

	f_free( &(pWindow->pszBuffer));
	f_free( &(pWindow->pucForeAttrib));
	f_free( &(pWindow->pucBackAttrib));
	f_free( &pWindow);
}

/****************************************************************************
Desc:		Low-level routine for opening a window
****************************************************************************/
FSTATIC RCODE ftxWinOpen(
	FTX_WINDOW *	pWindow)
{
	RCODE				rc = NE_FLM_OK;

	if( pWindow->bOpen)
	{
		goto Exit;
	}

	if( pWindow != pWindow->pScreen->pWinCur)
	{
		if( pWindow->pWinNext != NULL)
		{
			pWindow->pWinNext->pWinPrev = pWindow->pWinPrev;
		}

		if( pWindow->pWinPrev != NULL)
		{
			pWindow->pWinPrev->pWinNext = pWindow->pWinNext;
		}

		pWindow->pWinPrev = NULL;
		pWindow->pWinNext = pWindow->pScreen->pWinCur;
		
		if( pWindow->pWinNext)
		{
			pWindow->pWinNext->pWinPrev = pWindow;
		}
		pWindow->pScreen->pWinCur = pWindow;
	}
	
	pWindow->bOpen = TRUE;

Exit:

	pWindow->pScreen->bChanged = TRUE;
	return( rc);
}

/****************************************************************************
Desc:		Low-level routine for closing a window
****************************************************************************/
FSTATIC void ftxWinClose(
	FTX_WINDOW *	pWindow)
{
	FTX_WINDOW *	pWinTmp;
	
	if( pWindow->pScreen->pWinCur == pWindow &&
		pWindow->pWinNext != NULL)
	{
		pWindow->pScreen->pWinCur = pWindow->pWinNext;
	}

	if( pWindow->pWinNext != NULL)
	{
		pWindow->pWinNext->pWinPrev = pWindow->pWinPrev;
	}

	if( pWindow->pWinPrev != NULL)
	{
		pWindow->pWinPrev->pWinNext = pWindow->pWinNext;
	}

	pWinTmp = pWindow->pScreen->pWinCur;
	while( pWinTmp->pWinNext)
	{
		pWinTmp = pWinTmp->pWinNext;
	}

	pWindow->pWinPrev = pWinTmp;
	pWinTmp->pWinNext = pWindow;
	pWindow->pWinNext = NULL;
	pWindow->bOpen = FALSE;
	pWindow->pScreen->bChanged = TRUE;
}

/****************************************************************************
Desc:		Low-level routine for printing a character
****************************************************************************/
FSTATIC void ftxWinPrintChar(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiChar)
{
	FLMBOOL			bChanged = FALSE;
	FLMUINT			uiOffset;
	FLMUINT			uiRow;

	uiOffset = (FLMUINT)((FLMUINT)(pWindow->uiCurY * pWindow->uiCols) +
					pWindow->uiCurX);

	if( uiOffset >= ((FLMUINT)(pWindow->uiCols) * pWindow->uiRows))
	{
		goto Exit;
	}

	if( (uiChar > 31 && uiChar <= 126) || pWindow->bForceOutput)
	{
		if( (FLMUINT)pWindow->pszBuffer[ uiOffset] != uiChar ||
			pWindow->pucForeAttrib[ uiOffset] != pWindow->foregroundColor ||
			pWindow->pucBackAttrib[ uiOffset] != pWindow->backgroundColor)
		{
			pWindow->pszBuffer[ uiOffset] = (FLMBYTE)uiChar;
			pWindow->pucForeAttrib[ uiOffset] = (FLMBYTE)pWindow->foregroundColor;
			pWindow->pucBackAttrib[ uiOffset] = (FLMBYTE)pWindow->backgroundColor;
			bChanged = TRUE;
		}
		
		pWindow->uiCurX++;
	}
	else
	{
		switch( uiChar)
		{
			case 9: /* TAB */
			{
				pWindow->uiCurX += (FLMUINT)(8 - (pWindow->uiCurX % 8));

				if( pWindow->uiCurX > pWindow->uiCols)
				{
					pWindow->uiCurX = pWindow->uiOffset;
					pWindow->uiCurY++;
				}
				break;
			}
			
			case 10: /* LF */
			{
				pWindow->uiCurX = pWindow->uiOffset;
				pWindow->uiCurY++;
				break;
			}
			
			case 13: /* CR */
			{
				pWindow->uiCurX = pWindow->uiOffset;
				break;
			}
		}
	}

	if( pWindow->uiCurX + pWindow->uiOffset >= pWindow->uiCols)
	{
		if( pWindow->bNoLineWrap)
		{
			pWindow->uiCurX = (pWindow->uiCols - pWindow->uiOffset) - 1;
		}
		else
		{
			pWindow->uiCurY++;
			pWindow->uiCurX = pWindow->uiOffset;
		}
	}

	if( pWindow->uiCurY + pWindow->uiOffset >= pWindow->uiRows)
	{
		pWindow->uiCurY = (FLMUINT)(pWindow->uiRows - pWindow->uiOffset - 1);
		if( pWindow->bScroll)
		{
			if( pWindow->uiRows - pWindow->uiOffset > 1)
			{
				if( pWindow->uiOffset)
				{
					for( uiRow = pWindow->uiOffset;
						uiRow < pWindow->uiRows - (2 * pWindow->uiOffset); uiRow++)
					{
						uiOffset = (FLMUINT)((FLMUINT)(uiRow * pWindow->uiCols) +
							pWindow->uiOffset);
						f_memmove( pWindow->pszBuffer + uiOffset,
							pWindow->pszBuffer + uiOffset + pWindow->uiCols,
							pWindow->uiCols - (2 * pWindow->uiOffset));

						f_memmove( pWindow->pucForeAttrib + uiOffset,
							pWindow->pucForeAttrib + uiOffset + pWindow->uiCols,
							pWindow->uiCols - (2 * pWindow->uiOffset));

						f_memmove( pWindow->pucBackAttrib + uiOffset,
							pWindow->pucBackAttrib + uiOffset + pWindow->uiCols,
							pWindow->uiCols - (2 * pWindow->uiOffset));
					}
				}
				else
				{
					f_memmove( pWindow->pszBuffer,
						pWindow->pszBuffer + pWindow->uiCols,
						(pWindow->uiRows - 1) * pWindow->uiCols);

					f_memmove( pWindow->pucForeAttrib,
						pWindow->pucForeAttrib + pWindow->uiCols,
						(pWindow->uiRows - 1) * pWindow->uiCols);

					f_memmove( pWindow->pucBackAttrib,
						pWindow->pucBackAttrib + pWindow->uiCols,
						(pWindow->uiRows - 1) * pWindow->uiCols);
				}
			}

			uiOffset = (FLMUINT)(((FLMUINT)(pWindow->uiRows - pWindow->uiOffset - 1) *
				pWindow->uiCols) + pWindow->uiOffset);

			f_memset( pWindow->pszBuffer + uiOffset, (FLMBYTE)' ',
				pWindow->uiCols - (2 * pWindow->uiOffset));
			f_memset( pWindow->pucForeAttrib + uiOffset, (FLMBYTE)pWindow->foregroundColor,
				pWindow->uiCols - (2 * pWindow->uiOffset));
			f_memset( pWindow->pucBackAttrib + uiOffset, (FLMBYTE)pWindow->backgroundColor,
				pWindow->uiCols - (2 * pWindow->uiOffset));
			bChanged = TRUE;
		}
	}

Exit:

	if( pWindow->bOpen && bChanged)
	{
		pWindow->pScreen->bChanged = TRUE;
	}
}

/****************************************************************************
Desc:		Low-level routine for updating the cursor
****************************************************************************/
FSTATIC void ftxCursorUpdate( void)
{
	FLMUINT			uiCurX;
	FLMUINT			uiCurY;
	FTX_WINDOW *	pWinCur;

	if( gv_pFtxInfo->pScreenCur && gv_pFtxInfo->pScreenCur->bUpdateCursor)
	{
		pWinCur = gv_pFtxInfo->pScreenCur->pWinCur;
		if( pWinCur && pWinCur->bOpen)
		{
			uiCurX = (FLMUINT)(pWinCur->uiUlx + pWinCur->uiCurX);
			uiCurY = (FLMUINT)(pWinCur->uiUly + pWinCur->uiCurY);

			ftxDisplaySetCursorPos( uiCurX, uiCurY);
			ftxDisplaySetCursorType( pWinCur->uiCursorType);
			gv_pFtxInfo->uiCursorType = pWinCur->uiCursorType;
		}
		else
		{
			ftxDisplaySetCursorType( FLM_CURSOR_INVISIBLE);
			gv_pFtxInfo->uiCursorType = FLM_CURSOR_INVISIBLE;
		}
		gv_pFtxInfo->pScreenCur->bUpdateCursor = FALSE;
	}
}

/****************************************************************************
Desc:		Low-level routine for clearing a line
****************************************************************************/
FSTATIC void ftxWinClearLine(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiCol,
	FLMUINT			uiRow)
{
	FLMUINT		uiOffset;
	FLMUINT		uiSize;

	if( (pWindow->uiRows - (2 * pWindow->uiOffset)) > uiRow &&
		(pWindow->uiCols - (2 * pWindow->uiOffset)) > uiCol)
	{
		pWindow->uiCurY = (FLMUINT)(uiRow + pWindow->uiOffset);
		pWindow->uiCurX = (FLMUINT)(uiCol + pWindow->uiOffset);

		uiOffset = ((FLMUINT)(pWindow->uiCurY) * pWindow->uiCols) +
			pWindow->uiCurX;

		uiSize = (FLMUINT)(pWindow->uiCols - pWindow->uiOffset) - pWindow->uiCurX;

		f_memset( pWindow->pszBuffer + uiOffset, (FLMBYTE)' ', uiSize);
		f_memset( pWindow->pucForeAttrib + uiOffset, (FLMBYTE)pWindow->foregroundColor, uiSize);
		f_memset( pWindow->pucBackAttrib + uiOffset, (FLMBYTE)pWindow->backgroundColor, uiSize);

		pWindow->uiCurY = (FLMUINT)(uiRow + pWindow->uiOffset);
		pWindow->uiCurX = (FLMUINT)(uiCol + pWindow->uiOffset);
		if( pWindow->bOpen)
		{
			pWindow->pScreen->bChanged = TRUE;
		}
	}
}

/****************************************************************************
Desc:		Low-level routine for setting the cursor's position
****************************************************************************/
FSTATIC void ftxWinSetCursorPos(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiCol,
	FLMUINT			uiRow)
{
	if( (pWindow->uiRows - (2 * pWindow->uiOffset)) > uiRow &&
		(pWindow->uiCols - (2 * pWindow->uiOffset)) > uiCol)
	{
		pWindow->uiCurY = (FLMUINT)(uiRow + pWindow->uiOffset);
		pWindow->uiCurX = (FLMUINT)(uiCol + pWindow->uiOffset);
		pWindow->pScreen->bUpdateCursor = TRUE;
	}
}

/****************************************************************************
Desc:		Initializes the "physical" screen
****************************************************************************/
FSTATIC RCODE ftxDisplayInit(
	FLMUINT			uiRows,		// 0 means use current screen height.
	FLMUINT			uiCols,		// 0 means use current screen width.
	const char *	pszTitle)
{
#if defined( FLM_WIN)

	// Allocate a console if the application does not already have
	// one.

	if( AllocConsole())
	{
		gv_bAllocatedConsole = TRUE;
	}

	// Set up the console.

	if( (gv_hStdOut = GetStdHandle( STD_OUTPUT_HANDLE)) ==
		INVALID_HANDLE_VALUE)
	{
		return( RC_SET( NE_FLM_MEM));
	}

	// If FTX allocated the console, re-size the console to match
	// the requested size

	if( gv_bAllocatedConsole)
	{
		COORD			conSize;

		conSize.X = (SHORT)uiCols;
		conSize.Y = (SHORT)uiRows;
		
		SetConsoleScreenBufferSize( gv_hStdOut, conSize);
	}

	SMALL_RECT	conRec;

	conRec.Left = 0;
	conRec.Top = 0;
	conRec.Right = (SHORT)(uiCols - 1);
	conRec.Bottom = (SHORT)(uiRows - 1);
	SetConsoleWindowInfo( gv_hStdOut, TRUE, &conRec);

	if( (gv_hStdIn = GetStdHandle( STD_INPUT_HANDLE)) ==
		INVALID_HANDLE_VALUE)
	{
		return( RC_SET( NE_FLM_MEM));
	}

	// Save information about the screen attributes

	if( !GetConsoleScreenBufferInfo( gv_hStdOut, &gv_ConsoleScreenBufferInfo))
	{
		return( RC_SET( NE_FLM_MEM));
	}

	FlushConsoleInputBuffer( gv_hStdIn);
	SetConsoleMode( gv_hStdIn, 0);
	SetConsoleTitle( (LPCTSTR)pszTitle);

#elif defined( FLM_UNIX)

	F_UNREFERENCED_PARM( uiRows);
	F_UNREFERENCED_PARM( uiCols);
	F_UNREFERENCED_PARM( pszTitle);
	
	ftxUnixDisplayInit();

#else

	F_UNREFERENCED_PARM( uiRows);
	F_UNREFERENCED_PARM( uiCols);
	F_UNREFERENCED_PARM( pszTitle);

#endif

	// Set default cursor type

	ftxDisplaySetCursorType( FLM_CURSOR_VISIBLE | FLM_CURSOR_UNDERLINE);

	// Set default foreground/background colors

	ftxDisplaySetBackFore( FLM_BLACK, FLM_LIGHTGRAY);

	gv_bDisplayInitialized = TRUE;
	return( NE_FLM_OK);
}

/****************************************************************************
Desc:		Restores the "physical" screen to an initial state
****************************************************************************/
FSTATIC void ftxDisplayExit( void)
{
	if( gv_bDisplayInitialized)
	{
#if defined( FLM_UNIX)

		ftxUnixDisplayFree();

#elif defined( FLM_WIN)

		// Reset the console cursor

		CONSOLE_CURSOR_INFO		CursorInfo;

		CursorInfo.bVisible = TRUE;
		CursorInfo.dwSize = (DWORD)25;
		SetConsoleCursorInfo( gv_hStdOut, &CursorInfo);

		// Reset the screen attributes

		SetConsoleTextAttribute( gv_hStdOut,
			gv_ConsoleScreenBufferInfo.wAttributes);

		// Free the console if the application allocated one.

		if( gv_bAllocatedConsole)
		{
			(void)FreeConsole();
			gv_bAllocatedConsole = FALSE;
		}

#endif
	}

	gv_bDisplayInitialized = FALSE;
}

/****************************************************************************
Desc:		Resets (clears) the "physical" screen and positions the cursor
			at the origin
****************************************************************************/
FSTATIC void ftxDisplayReset( void)
{
#if defined( FLM_WIN)
	{
		COORD									coord;
		DWORD									dCharWritten;
		DWORD									dCharsToWrite;
		CONSOLE_SCREEN_BUFFER_INFO		Console;

		if( GetConsoleScreenBufferInfo( gv_hStdOut, &Console) == FALSE)
			return;

		dCharsToWrite = Console.dwMaximumWindowSize.X *
			Console.dwMaximumWindowSize.Y;

		coord.X = 0;
		coord.Y = 0;

		// Fill the screen with spaces
		
		FillConsoleOutputCharacter( gv_hStdOut, ' ',
				dCharsToWrite, coord, &dCharWritten);

		// Set the screen colors back to default colors.
		
		FillConsoleOutputAttribute( gv_hStdOut, FOREGROUND_INTENSITY,
				dCharsToWrite, coord, &dCharWritten);
	}

#elif defined( FLM_UNIX)
	ftxUnixDisplayReset();
#elif defined( FLM_NLM)
	ClearScreen( gv_pFtxInfo->hScreen);
#else
	clrscr();
#endif
}

/****************************************************************************
Desc: Returns the size of the "physical" screen in columns and rows
****************************************************************************/
FSTATIC void ftxDisplayGetSize(
	FLMUINT *	puiNumColsRV,
	FLMUINT *	puiNumRowsRV)
{
#if defined( FLM_WIN)
	CONSOLE_SCREEN_BUFFER_INFO Console;

	if( GetConsoleScreenBufferInfo( gv_hStdOut, &Console) == FALSE)
	{
		return;
	}

	*puiNumColsRV = (FLMUINT)Console.dwSize.X;
	*puiNumRowsRV = (FLMUINT)Console.dwSize.Y;

#elif defined( FLM_UNIX)
	ftxUnixDisplayGetSize( puiNumColsRV, puiNumRowsRV);
#else

	FLMUINT16	ui16ScreenHeight;
	FLMUINT16	ui16ScreenWidth;

	GetScreenSize( &ui16ScreenHeight, &ui16ScreenWidth);

	*puiNumColsRV = ui16ScreenWidth;
	*puiNumRowsRV = ui16ScreenHeight;
#endif
}

/****************************************************************************
Desc : Sets the "physical" cursor attributes
****************************************************************************/
FSTATIC FLMBOOL ftxDisplaySetCursorType(
	FLMUINT		uiType)
{
#if defined( FLM_WIN)
	{
		CONSOLE_CURSOR_INFO		CursorInfo;

		if( uiType & FLM_CURSOR_INVISIBLE)
		{
			CursorInfo.dwSize = (DWORD)99;
			CursorInfo.bVisible = FALSE;
		}
		else
		{
			CursorInfo.bVisible = TRUE;
			if( uiType & FLM_CURSOR_BLOCK)

			{
				CursorInfo.dwSize = (DWORD)99;
			}
			else
			{
				CursorInfo.dwSize = (DWORD)25;
			}
		}

		return( (FLMBOOL)SetConsoleCursorInfo( gv_hStdOut, &CursorInfo));
	}

#elif defined( FLM_NLM)

	if (uiType & FLM_CURSOR_INVISIBLE)
	{
		DisableInputCursor( gv_pFtxInfo->hScreen);
	}
	else if (uiType & FLM_CURSOR_BLOCK)
	{
		EnableInputCursor( gv_pFtxInfo->hScreen);
		SetCursorStyle( gv_pFtxInfo->hScreen, CURSOR_BLOCK);
	}
	else
	{
		EnableInputCursor( gv_pFtxInfo->hScreen);
		SetCursorStyle( gv_pFtxInfo->hScreen, CURSOR_NORMAL);
	}

	return( TRUE);
#else
	F_UNREFERENCED_PARM( uiType);
	return( FALSE);
#endif
}

/****************************************************************************
Desc:		Sets the "physical" cursor to the column and row specified
****************************************************************************/
FSTATIC void ftxDisplaySetCursorPos(
	FLMUINT		uiCol,
	FLMUINT		uiRow)
{
	if( uiCol == (FLMUINT)255 || uiRow == (FLMUINT)255)
	{
		return;
	}

#if defined( FLM_NLM)
	PositionOutputCursor( gv_pFtxInfo->hScreen, 
		(FLMUINT16)uiRow, (FLMUINT16)uiCol);

	// Wake up the input thread and send it a special code to
	// cause the cursor to be re-positioned.
	
	UngetKey( gv_pFtxInfo->hScreen, 0xFE, (FLMBYTE)uiRow, (FLMBYTE)uiCol, 0);
		
#elif defined( FLM_WIN)

	{
		COORD		coord;

		coord.X = (SHORT)uiCol;
		coord.Y = (SHORT)uiRow;
		SetConsoleCursorPosition( gv_hStdOut, coord);
	}

#elif defined( FLM_UNIX)
	ftxUnixDisplaySetCursorPos( uiCol, uiRow);
	ftxUnixDisplayRefresh();
#else
	gotoxy( (FLMUINT)(uiCol + 1), (FLMUINT)(uiRow + 1));
#endif
}

/****************************************************************************
Desc:		Outputs a string to the "physical" screen
****************************************************************************/
#if defined( FLM_UNIX)
FSTATIC FLMUINT ftxDisplayStrOut(
	const char *	pszString,
	FLMUINT			uiAttr)
{
	while( *pszString)
	{
		ftxUnixDisplayChar( (FLMUINT)*pszString, uiAttr);
		pszString++;
	}

	return( (FLMUINT)0);
}
#endif

/****************************************************************************
Desc:		Set the background and foreground colors of the "physical" screen
****************************************************************************/
#ifdef FLM_WIN
FSTATIC FLMUINT ftxMapFlmColorToWin32(
	eColorType		uiColor)
{
	switch( uiColor)
	{
		case FLM_BLACK:
			return( 0);
		case FLM_BLUE:
			return( 1);
		case FLM_GREEN:
			return( 2);
		case FLM_CYAN:
			return( 3);
		case FLM_RED:
			return( 4);
		case FLM_MAGENTA:
			return( 5);
		case FLM_BROWN:
			return( 6);
		case FLM_LIGHTGRAY:
			return( 7);
		case FLM_DARKGRAY:
			return( 8);
		case FLM_LIGHTBLUE:
			return( 9);
		case FLM_LIGHTGREEN:
			return( 10);
		case FLM_LIGHTCYAN:
			return( 11);
		case FLM_LIGHTRED:
			return( 12);
		case FLM_LIGHTMAGENTA:
			return( 13);
		case FLM_YELLOW:
			return( 14);
		case FLM_WHITE:
			return( 15);
		default:
			return( 0);
	}
}
#endif

/****************************************************************************
Desc:		Set the background and foreground colors of the "physical" screen
****************************************************************************/
FSTATIC void ftxDisplaySetBackFore(
	eColorType	backgroundColor,
	eColorType	foregroundColor)
{

#if defined( FLM_WIN)

	FLMUINT	uiAttrib = 0;

	uiAttrib = (ftxMapFlmColorToWin32( foregroundColor) & 0x8F) |
				  ((ftxMapFlmColorToWin32( backgroundColor) << 4) & 0x7F);
	SetConsoleTextAttribute( gv_hStdOut, (WORD)uiAttrib);

#else

	F_UNREFERENCED_PARM( backgroundColor);
	F_UNREFERENCED_PARM( foregroundColor);

#endif
}

/****************************************************************************
Desc:		Gets a character from the "physical" keyboard and converts keyboard
			sequences/scan codes to FKB key strokes.
****************************************************************************/
FLMUINT ftxKBGetChar( void)
{
	FLMUINT	uiChar = 0;
#ifdef FLM_NLM
	BYTE		scanCode;
	BYTE		keyType;
	BYTE		keyValue;
	BYTE		keyStatus;
#endif

#if defined( FLM_NLM)

get_key:

	// Are we exiting?

	if( gv_pFtxInfo->pbShutdown != NULL)
	{
		if( *(gv_pFtxInfo->pbShutdown) == TRUE)
		{
			return( uiChar);
		}
	}

	// Get a key

	GetKey( gv_pFtxInfo->hScreen, &keyType, &keyValue, &keyStatus, &scanCode, 0);

	switch (keyType)
	{
		case 0:	// NORMAL_KEY
			if (keyStatus & 4)		// CTRL key pressed
			{
				uiChar = FKB_CTRL_A + keyValue - 1;
			}
			else if (keyStatus & 8) // ALT key pressed
			{
				uiChar = ScanCodeToFKB[scanCode];
			}
			else	// Handles SHIFT key case.
			{
				uiChar = (FLMUINT)keyValue;
			}
			break;
		case 1:	// FUNCTION_KEY
			uiChar = ScanCodeToFKB[scanCode];
			if (keyStatus & 4)		// CTRL key pressed
			{
				uiChar = FKB_CTRL_F1 + (uiChar - FKB_F1);
			}
			else if (keyStatus & 8) // ALT key pressed
			{
				uiChar = FKB_ALT_F1 + (uiChar - FKB_F1);
			}
			else if (keyStatus & 2) // SHIFT key pressed
			{
				uiChar = FKB_SF1 + (uiChar - FKB_F1);
			}
			break;
		case 2:	// ENTER_KEY
			if (keyStatus & 4)
			{
				uiChar = FKB_CTRL_ENTER;
			}
			else
			{
				uiChar = FKB_ENTER;
			}
			break;
		case 3:	// ESCAPE_KEY
			uiChar = FKB_ESCAPE;
			break;
		case 4:	// BACKSPACE_KEY
			uiChar = FKB_BACKSPACE;
			break;
		case 5:	// DELETE_KEY
			if (keyStatus & 4)
			{
				uiChar = FKB_CTRL_DELETE;
			}
			else
			{
				uiChar = FKB_DELETE;
			}
			break;
		case 6:	// INSERT_KEY
			if (keyStatus & 4)
			{
				uiChar = FKB_CTRL_INSERT;
			}
			else
			{
				uiChar = FKB_INSERT;
			}
			break;
		case 7:	// CURSOR_UP_KEY
			if (keyStatus & 4)
			{
				uiChar = FKB_CTRL_UP;
			}
			else
			{
				uiChar = FKB_UP;
			}
			break;
		case 8:	// CURSOR_DOWN_KEY
			if (keyStatus & 4)
			{
				uiChar = FKB_CTRL_DOWN;
			}
			else
			{
				uiChar = FKB_DOWN;
			}
			break;
		case 9:	// CURSOR_RIGHT_KEY
			if (keyStatus & 4)
			{
				uiChar = FKB_CTRL_RIGHT;
			}
			else
			{
				uiChar = FKB_RIGHT;
			}
			break;
		case 10: // CURSOR_LEFT_KEY
			if (keyStatus & 4)
			{
				uiChar = FKB_CTRL_LEFT;
			}
			else
			{
				uiChar = FKB_LEFT;
			}
			break;
		case 11: // CURSOR_HOME_KEY
			if (keyStatus & 4)
			{
				uiChar = FKB_CTRL_HOME;
			}
			else
			{
				uiChar = FKB_HOME;
			}
			break;
		case 12: // CURSOR_END_KEY
			if (keyStatus & 4)
			{
				uiChar = FKB_CTRL_END;
			}
			else
			{
				uiChar = FKB_END;
			}
			break;
		case 13: // CURSOR_PUP_KEY
			if (keyStatus & 4)
			{
				uiChar = FKB_CTRL_PGUP;
			}
			else
			{
				uiChar = FKB_PGUP;
			}
			break;
		case 14: // CURSOR_PDOWN_KEY
			if (keyStatus & 4)
			{
				uiChar = FKB_CTRL_PGDN;
			}
			else
			{
				uiChar = FKB_PGDN;
			}
			break;
		case 0xFE:
			// Re-position the input cursor
			PositionInputCursor( gv_pFtxInfo->hScreen, keyValue, keyStatus);
			goto get_key;
		case 0xFF:
			// Ping
			uiChar = (FLMUINT)0xFFFF;
			break;
		default:
			uiChar = (FLMUINT)keyValue;
			break;
	}
#elif defined( FLM_WIN)
	uiChar = (FLMUINT) ftxWin32KBGetChar();
#elif defined( FLM_UNIX)
	uiChar = ftxUnixKBGetChar();
#else
	uiChar = (FLMUINT) getch();
#endif

#if defined( FLM_WIN)
	if( uiChar == 0 || uiChar == 0x00E0)
	{
		FLMUINT	scanCode = (FLMUINT)ftxWin32KBGetChar();
		if( scanCode < (sizeof( ScanCodeToFKB) / sizeof( FLMUINT)))
		{
			uiChar = ScanCodeToFKB[ scanCode ];
		}
	}
	else if( (uiChar > 0) && (uiChar < 0x20))
	{
		switch( uiChar)
		{
			case	0x0D:
				uiChar = FKB_ENTER;				 break;
			case	0x1B:
				uiChar = FKB_ESCAPE;				 break;
			case	0x08:
				uiChar = FKB_BACKSPACE;			 break;
			case	0x09:
				uiChar = FKB_TAB;					 break;
			case	0x0A:
				uiChar = FKB_CTRL_ENTER;		 break;

			/* Default is a ctrl-letter code */
			default:
				uiChar = (FLMUINT)((FKB_CTRL_A - 1) + uiChar);
				break;
		}
	}
#endif
	return( uiChar);
}

/****************************************************************************
Desc:		Returns TRUE if a key is available from the "physical" keyboard
****************************************************************************/
FLMBOOL ftxKBTest( void)
{
#if defined( FLM_UNIX)

	return( ftxUnixKBTest());

#elif defined( FLM_WIN)

	DWORD				lRecordsRead;
	INPUT_RECORD	inputRecord;
	FLMBOOL			bKeyHit = FALSE;

	// VISIT: If a keyboard handler has not been started, need
	// to protect this code with a critical section?

	for( ;;)
	{
		if( PeekConsoleInput( gv_hStdIn, &inputRecord, 1, &lRecordsRead))
		{
			if( !lRecordsRead)
			{
				break;
			}

			if( inputRecord.EventType == KEY_EVENT)
			{
				if( inputRecord.Event.KeyEvent.bKeyDown &&
					(inputRecord.Event.KeyEvent.uChar.AsciiChar ||
					ftxWin32GetExtendedKeycode( &(inputRecord.Event.KeyEvent))))
				{
					bKeyHit = TRUE;
					break;
				}
			}

			if( !ReadConsoleInput( gv_hStdIn, &inputRecord, 1, &lRecordsRead))
			{
				goto Exit;
			}
		}
		else
		{
			break;
		}
	}

Exit:

	return( bKeyHit);
#elif defined( FLM_NLM)

	return( (FLMBOOL)CheckKeyStatus( gv_pFtxInfo->hScreen));

#else

	return( kbhit());

#endif
}

/****************************************************************************
Desc:		Causes the console to "beep"
Ret:		If the console does not support this feature, FALSE is returned.
****************************************************************************/
void FTKAPI FTXBeep( void)
{
#if defined( FLM_WIN)

	Beep( (DWORD)2000, (DWORD)250);

#endif
}

/****************************************************************************
Desc:		Gets a character from the Win32 console
****************************************************************************/
#if defined( FLM_WIN)
FSTATIC FLMUINT ftxWin32KBGetChar( void)
{
	INPUT_RECORD				ConInpRec;
	DWORD							NumRead;
	ftxWin32CharPair *		pCP;
	int							uiChar = 0;

	// Check pushback buffer (chbuf) a for character
	
	if( chbuf != -1 )
	{
		// Something there, clear buffer and return the character.
		
		uiChar = (unsigned char)(chbuf & 0xFF);
		chbuf = -1;
		return( uiChar);
	}

	for( ;;)
	{
		if( gv_pFtxInfo->pbShutdown != NULL)
		{
			if( *(gv_pFtxInfo->pbShutdown) == TRUE)
			{
				return( 0);
			}
		}

		// Get a console input event.
		
		if( !ReadConsoleInput( gv_hStdIn,
			&ConInpRec, 1L, &NumRead) || (NumRead == 0L))
		{
			uiChar = -1;
			break;
		}

		// Look for, and decipher, key events.
		
		if( ConInpRec.EventType == KEY_EVENT)
		{
			if( ConInpRec.Event.KeyEvent.bKeyDown)
			{
				if ( !(ConInpRec.Event.KeyEvent.dwControlKeyState &
					(LEFT_ALT_PRESSED | RIGHT_ALT_PRESSED | SHIFT_PRESSED | CAPSLOCK_ON)))
				{
					if( (uiChar = (FLMUINT)ConInpRec.Event.KeyEvent.uChar.AsciiChar) != 0)
					{
					  break;
					}
				}

				if( (pCP = ftxWin32GetExtendedKeycode(
					&(ConInpRec.Event.KeyEvent))) != NULL)
				{
					uiChar = pCP->LeadChar;
					if( pCP->SecondChar)
					{
						chbuf = pCP->SecondChar;
					}
					
					break;
				}
			}
			else
			{
				if( ConInpRec.Event.KeyEvent.uChar.AsciiChar == (unsigned char)0xFF &&
					ConInpRec.Event.KeyEvent.wRepeatCount == 0)
				{
					// Ping
					
					uiChar = (FLMUINT)0xFFFF;
					break;
				}
			}
		}
	}

	return( uiChar);
}
#endif

#if defined( FLM_WIN)
/****************************************************************************
Desc:
****************************************************************************/
FSTATIC ftxWin32CharPair * ftxWin32GetExtendedKeycode(
	KEY_EVENT_RECORD *	pKE)
{
	DWORD						CKS;
	ftxWin32CharPair *	pCP;
	int						iLoop;

	if( (CKS = pKE->dwControlKeyState) & ENHANCED_KEY )
	{
		// Find the appropriate entry in EnhancedKeys[]

		for( pCP = NULL, iLoop = 0; iLoop < FTX_WIN32_NUM_EKA_ELTS; iLoop++)
		{
			if( ftxWin32EnhancedKeys[ iLoop].ScanCode == pKE->wVirtualScanCode)
			{
				if( CKS & (LEFT_ALT_PRESSED | RIGHT_ALT_PRESSED))
				{
					pCP = &(ftxWin32EnhancedKeys[ iLoop].AltChars);
				}
				else if( CKS & (LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED))
				{
					pCP = &(ftxWin32EnhancedKeys[ iLoop].CtrlChars);
				}
				else if( CKS & SHIFT_PRESSED)
				{
					pCP = &(ftxWin32EnhancedKeys[ iLoop].ShiftChars);
				}
				else
				{
					pCP = &(ftxWin32EnhancedKeys[ iLoop].RegChars);
				}
				break;
			}
		}
	}
	else
	{
		// Regular key or a keyboard event which shouldn't be recognized.
		// Determine which by getting the proper field of the proper
		// entry in NormalKeys[], and examining the extended code.

		if( CKS & (LEFT_ALT_PRESSED | RIGHT_ALT_PRESSED))
		{
			pCP = &(ftxWin32NormalKeys[pKE->wVirtualScanCode].AltChars);
		}
		else if( CKS & (LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED))
		{
			pCP = &(ftxWin32NormalKeys[pKE->wVirtualScanCode].CtrlChars);
		}
		else if( CKS & SHIFT_PRESSED)
		{
			pCP = &(ftxWin32NormalKeys[pKE->wVirtualScanCode].ShiftChars);
		}
		else
		{
			pCP = &(ftxWin32NormalKeys[pKE->wVirtualScanCode].RegChars);
			if( (CKS & CAPSLOCK_ON) && pCP->SecondChar == 0)
			{
				if( pCP->LeadChar >= 'a' && pCP->LeadChar <= 'z')
				{
					pCP->LeadChar = pCP->LeadChar - 'a' + 'A';
				}
			}
		}

		if( pCP->LeadChar == 0 && pCP->SecondChar == 0)
		{
			pCP = NULL;
		}
  }

  return( pCP);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI _ftxDefaultDisplayHandler(
	IF_Thread *		pThread)
{
#if defined( FLM_WIN)
	FLMUINT		uiRefreshCount = 0;
#endif

	for( ;;)
	{
		if( pThread->getShutdownFlag())
		{
			break;
		}

		if( gv_pFtxInfo->pbShutdown != NULL)
		{
			if( *(gv_pFtxInfo->pbShutdown) == TRUE)
			{
				break;
			}
		}

#if defined( FLM_WIN)
		if( ++uiRefreshCount > 60)
		{
			uiRefreshCount = 0;

			// Update the cursor to work around a bug in NT where the
			// cursor is set to visible when the console is made
			// full-screen.

			FTXRefreshCursor();
		}
#endif

		FTXRefresh();
		f_sleep( 50); // Refresh 20 times a second
	}

	return( NE_FLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI _ftxDefaultKeyboardHandler(
	IF_Thread *		pThread)
{
	FLMUINT		uiChar;

	f_mutexLock( gv_pFtxInfo->hFtxMutex);
	gv_pFtxInfo->bEnablePingChar = TRUE;
	f_mutexUnlock( gv_pFtxInfo->hFtxMutex);

	for( ;;)
	{
		if( pThread->getShutdownFlag())
		{
			break;
		}

		if( gv_pFtxInfo->pbShutdown != NULL)
		{
			if( *(gv_pFtxInfo->pbShutdown) == TRUE)
			{
				break;
			}
		}

		uiChar = 0;

#if !defined( FLM_NLM) && !defined( FLM_WIN)

		// NetWare and Windows will wake up periodically
		// to check for shutdown and therefore do not
		// need to poll the keyboard.
		
		if( ftxKBTest())
#endif
		{
			uiChar = ftxKBGetChar();
			if( gv_pFtxInfo->pKeyHandler && uiChar != 0xFFFF)
			{
				gv_pFtxInfo->pKeyHandler( uiChar,
					&uiChar, gv_pFtxInfo->pvKeyHandlerData);
			}

			switch( uiChar)
			{
				case 0:
				{
					// Ignore the keystroke
					break;
				}

				case FKB_CTRL_A:
				{
					FTXCycleScreensNext();
					FTXRefresh();
					break;
				}

				case FKB_CTRL_B:
				{
					// Enter the debugger
					f_assert( 0);
					break;
				}

				case FKB_CTRL_L:
				{
					FTXInvalidate();
					FTXRefresh();
					break;
				}

				case FKB_CTRL_S:
				{
					FTXCycleScreensPrev();
					FTXRefresh();
					break;
				}

				case 0xFFFF:
				{
					// Ping
					break;
				}

				default:
				{
					FTXAddKey( uiChar);
					break;
				}
			}
		}

#if !defined( FLM_NLM) && !defined( FLM_WIN)
		f_sleep( 0);
#endif
	}

	return( NE_FLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI _ftxBackgroundThread(
	IF_Thread *		pThread)
{
	for( ;;)
	{
		// Ping the keyboard handler to cause it to wake up
		// periodically to check for the shutdown flag
		
		if( gv_pFtxInfo->bEnablePingChar)
		{
#if defined( FLM_NLM)
			UngetKey( gv_pFtxInfo->hScreen, 0xFF, 0, 0, 0);
#elif defined( FLM_WIN)
			{
				INPUT_RECORD		inputRec;
				DWORD					numWritten;

				f_memset( &inputRec, (FLMBYTE)0, sizeof( INPUT_RECORD));
				inputRec.EventType = KEY_EVENT;
				inputRec.Event.KeyEvent.bKeyDown = FALSE;
				inputRec.Event.KeyEvent.wRepeatCount = 0;
				inputRec.Event.KeyEvent.wVirtualKeyCode = 0;
				inputRec.Event.KeyEvent.wVirtualScanCode = 0;
				inputRec.Event.KeyEvent.uChar.AsciiChar = (unsigned char)0xFF;
				inputRec.Event.KeyEvent.dwControlKeyState = 0;

				WriteConsoleInput( gv_hStdIn, &inputRec, 1, &numWritten);
			}
#endif
		}

		if( pThread->getShutdownFlag())
		{
			break;
		}

		f_sleep( 250);
	}

	return( NE_FLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI FTXDisplayScrollWindow(
	FTX_SCREEN *	pScreen,
	const char *	pszTitle,
	const char *	pszMessage,
	FLMUINT			uiCols,
	FLMUINT			uiRows)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiScreenCols = 0;
	FLMUINT			uiScreenRows = 0;
	FLMUINT			uiCanvasCols = 0;
	FLMUINT			uiCanvasRows = 0;
	FLMUINT			uiMessageLen = f_strlen( pszMessage);
	FLMUINT			uiNumLines	= 0;
	FLMUINT			uiLineLen = 0;
	FLMUINT			uiLoop = 0;
	char **			ppszRows = NULL;
	FLMUINT			uiCurrentLine = 0;
	const char *	pszBeginLine = NULL;
	FLMUINT			uiLastSpace;
	FTX_WINDOW *	pWin = NULL;
	FLMUINT			uiChar = 0;

	if( RC_BAD( rc = FTXScreenGetSize( pScreen, &uiScreenCols, &uiScreenRows)))
	{
		goto Exit;
	}

	// Make sure the window will fit on the screen
	
	if( ( uiCols > uiScreenCols) || ( uiRows > uiScreenRows))
	{
		rc = RC_SET( NE_FLM_ILLEGAL_OP);
		goto Exit;
	}

	if( RC_BAD( rc = FTXWinInit( pScreen, uiCols, uiRows, &pWin)))
	{
		goto Exit;
	}

	// Center the window
	
	FTXWinMove( pWin, (FLMUINT)((uiScreenCols - uiCols) / 2),
		(FLMUINT)((uiScreenRows - uiRows) / 2));

	FTXWinClear( pWin);
	FTXWinSetFocus( pWin);
	FTXWinDrawBorder( pWin);
	FTXWinSetTitle( pWin, pszTitle, FLM_BLUE, FLM_WHITE);
	
	if( RC_BAD( rc = FTXWinOpen( pWin)))
	{
		goto Exit;
	}
	
	FTXWinSetScroll( pWin, FALSE);
	FTXWinSetCursorType( pWin, FLM_CURSOR_INVISIBLE);
	FTXWinGetCanvasSize( pWin, &uiCanvasCols, &uiCanvasRows);

	uiMessageLen = f_strlen( pszMessage);
	
	// Count the number of lines
	
	for ( uiLoop = 0, uiLastSpace = (FLMUINT)~0;
			uiLoop < (uiMessageLen + 1);
			uiLoop++)
	{
		uiLineLen++;
		
		if( (pszMessage[ uiLoop] == '\n') || 
			(uiLineLen >= uiCanvasCols) ||
			(pszMessage[ uiLoop] == '\0'))
		{
			uiNumLines++;
			if( uiLastSpace != (FLMUINT)~0 && uiLineLen >= uiCanvasCols)
			{
				uiLoop = uiLastSpace;
				uiLastSpace = (FLMUINT)~0;
			}
			uiLineLen = 0;
		}
		else
		{
			if( pszMessage[ uiLoop] <= ' ')
			{
				uiLastSpace = uiLoop;
			}
		}
	}

	if( RC_BAD( rc = f_alloc( sizeof( FLMBYTE *) * uiNumLines, &ppszRows)))
	{
		goto Exit;
	}

	uiLineLen = 0;
	pszBeginLine = pszMessage;
	
	for( uiLoop = 0, uiLastSpace = (FLMUINT)~(0);
		  uiLoop < (uiMessageLen + 1);
		  uiLoop++)
	{
		if ( uiLineLen >= uiCanvasCols ||
			  pszMessage[ uiLoop] == '\n' ||
			  pszMessage[ uiLoop] == '\0')
		{

			if (uiLastSpace != (FLMUINT)~(0) && uiLineLen >= uiCanvasCols)
			{
				uiLineLen = (FLMUINT)(&pszMessage[ uiLastSpace] - pszBeginLine + 1);
				uiLoop = uiLastSpace;
				uiLastSpace = (FLMUINT)~(0);
			}

			if( RC_BAD( rc = f_alloc( uiLineLen + 1, &ppszRows[ uiCurrentLine])))
			{
				goto Exit;
			}
			f_memcpy( ppszRows[ uiCurrentLine], pszBeginLine, uiLineLen);
			ppszRows[ uiCurrentLine++][uiLineLen] = '\0';
			pszBeginLine += uiLineLen;
			uiLineLen = 0;
			continue;
		}
		else
		{
			if (uiLastSpace != (FLMUINT)~0)
			{
				f_assert( (FLMUINT)(&pszMessage[ uiLastSpace] - pszBeginLine) <= uiCanvasCols);
			}
			if (pszMessage[ uiLoop] <= ' ')
			{
				uiLastSpace = uiLoop;
			}
		}
		uiLineLen++;
	}

	uiCurrentLine = 0;

	for ( uiLoop = 0; uiLoop < uiNumLines && uiLoop < uiRows - 2; uiLoop++)
	{
		FTXWinSetCursorPos( pWin, 0, uiLoop);
		FTXWinPrintStr( pWin, ppszRows[ uiLoop]);
	}
	
	FTXRefresh();

	for( ;;)
	{
		if( FTXWinTestKB( pWin))
		{
			FTXWinInputChar( pWin, &uiChar);
			switch( uiChar)
			{
				case FKB_DOWN:
				{
					if( uiCurrentLine < ( uiNumLines - 1))
					{
						uiCurrentLine++;
						FTXWinClear( pWin);
						
						for ( uiLoop = 0;
								uiLoop < (uiNumLines - uiCurrentLine) &&
										uiLoop < uiRows - 2;
								uiLoop++)
						{
							FTXWinSetCursorPos( pWin, 0, uiLoop);
							FTXWinClearToEOL( pWin);
							FTXWinPrintStr( pWin, ppszRows[ uiLoop + uiCurrentLine]);
						}
						
						FTXRefresh();
					}
					
					break;
				}
				
				case FKB_UP:
				{
					if ( uiCurrentLine > 0)
					{
						uiCurrentLine--;
						FTXWinClear( pWin);
						
						for ( uiLoop = 0;
								uiLoop < (uiNumLines - uiCurrentLine) &&
										uiLoop < uiRows - 2;
								uiLoop++)
						{
							FTXWinSetCursorPos( pWin, 0, uiLoop);
							FTXWinClearToEOL( pWin);
							FTXWinPrintStr( pWin, ppszRows[ uiLoop + uiCurrentLine]);
						}
						
						FTXRefresh();
					}
					
					break;
				}
				
				case FKB_ENTER:
				case FKB_ESC:
				{
					goto Exit;
				}
			}
		}
		
		f_yieldCPU();
	}

Exit:

	if( pWin)
	{
		FTXWinFree( &pWin);
		FTXRefresh();
	}
	
	if( ppszRows)
	{
		for( uiLoop = 0; uiLoop < uiNumLines; uiLoop++)
		{
			f_free( &ppszRows[ uiLoop]);
		}
		
		f_free( &ppszRows);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_UNIX
FSTATIC void ftxUnixDisplayInit( void)
{
	initscr();
	noecho();
	cbreak();
	halfdelay( 4);
	meta( stdscr, TRUE);
	keypad( stdscr, TRUE);
	scrollok( stdscr, FALSE);
	move( 0, 0);
	refresh();
	
	ungetChar = (int)ERR;
	
	if( has_colors())
	{
		start_color();

		f_memset( flm2curses, 0, sizeof( flm2curses));
		f_memset( color_pairs, 0, sizeof( color_pairs));

		flm2curses[ FLM_BLACK] = COLOR_BLACK;
		flm2curses[ FLM_BLUE] = COLOR_BLUE;
		flm2curses[ FLM_GREEN] = COLOR_GREEN;
		flm2curses[ FLM_CYAN] = COLOR_CYAN;
		flm2curses[ FLM_RED] = COLOR_RED;
		flm2curses[ FLM_MAGENTA] = COLOR_MAGENTA;
		flm2curses[ FLM_BROWN] = COLOR_YELLOW;
		flm2curses[ FLM_LIGHTGRAY] = COLOR_WHITE;
		flm2curses[ FLM_DARKGRAY] = COLOR_WHITE;
		flm2curses[ FLM_LIGHTBLUE] = COLOR_BLUE;
		flm2curses[ FLM_LIGHTGREEN] = COLOR_GREEN;
		flm2curses[ FLM_LIGHTCYAN] = COLOR_CYAN;
		flm2curses[ FLM_LIGHTRED] = COLOR_RED;
		flm2curses[ FLM_LIGHTMAGENTA] = COLOR_MAGENTA;
		flm2curses[ FLM_YELLOW] = COLOR_YELLOW;
		flm2curses[ FLM_WHITE] = COLOR_WHITE;

		bkgd( A_NORMAL | ' ');
	}
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_UNIX
FSTATIC void ftxUnixDisplayFree( void)
{
	endwin();
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_UNIX
FSTATIC void ftxUnixDisplayGetSize(
	FLMUINT *		puiNumColsRV,
	FLMUINT *		puiNumRowsRV)
{
	*puiNumColsRV = (FLMUINT)COLS;
	*puiNumRowsRV = (FLMUINT)LINES;
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_UNIX
static int flm_to_curses_attr(
	int		attr)
{
	int		fg;
	int		bg;
	int		curses_attr = 0;

	fg = attr & 0x0F;
	bg = (attr >> 4) & 0x0F;

	curses_attr = (fg > FLM_LIGHTGRAY) ? A_BOLD : 0;
	
	if (has_colors())
	{
		if (color_pairs[bg][fg] == 0)
		{
			if (last_pair >= COLOR_PAIRS)
			{
				color_pairs[bg][fg] = 1;
			}
			else
			{
				color_pairs[bg][fg] = ++last_pair;
				init_pair(last_pair, flm2curses[fg], flm2curses[bg]);
			}
		}
		
		curses_attr |= COLOR_PAIR(color_pairs[bg][fg]);
	}
	else
	{
		curses_attr |= (bg != FLM_BLUE) ? A_REVERSE : 0;
	}

	return( curses_attr);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_UNIX
FSTATIC void ftxUnixDisplayChar(
	FLMUINT			uiChar,
	FLMUINT			uiAttr)
{
	addch( (chtype)(uiChar | flm_to_curses_attr( (int)uiAttr)));
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_UNIX
FSTATIC void ftxUnixDisplayRefresh( void)
{
	refresh();
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_UNIX
FSTATIC void ftxUnixDisplayReset( void)
{
	clearok( stdscr, TRUE);
	refresh();
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_UNIX
FSTATIC void ftxUnixDisplaySetCursorPos(
	FLMUINT			uiCol,
	FLMUINT			uiRow)
{
	move( (int)uiRow, (int)uiCol);
	refresh();
}
#endif

/****************************************************************************
Desc:		Simulate Ctrl + Shift + character keys
****************************************************************************/
#ifdef FLM_UNIX
static FLMUINT ftxUnixSimulatedKey(
	FLMUINT			c)
{
	// We simulate Insert, Delete, Home, End, PgUp and PgDn. We can
	// also simulate the CTRL- combinations of these if needed

	chtype ch = (chtype) c + 'a' - 1;
	
	switch( ch)
	{
		case 'i':
		{
			c = FKB_INSERT;
			break;
		}
	
		case 'd':
		{
			c = FKB_DELETE;
			break;
		}
	
		case 'b':
		{
			c = FKB_PGUP;
			break;
		}
	
		case 'f':
		{
			c = FKB_PGDN;
			break;
		}
	
		case 'h':
		{
			c = FKB_HOME;
			break;
		}
	
		case 'e':
		{
			c = FKB_END;
			break;
		}
	
		default:
		{
			c = (FLMUINT)ERR;
			break;
		}
	}
	
	return( c);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_UNIX
static FLMUINT ftxUnixHandleEsc( void)
{
	// On unix ESC is the prefix for many function keys. It's a bad
	// idea to use ESC as a character by itself because it can result
	// in a delay of as much as a second.	If we don't handle all
	// escape sequences, the first escape character can cause FLAIM to
	// exit!	 So, we discard unrecognized escape sequences here.

	int c = FKB_ESCAPE;
	int c2;

	if( (c2 = getch()) == ERR)
	{
		goto Exit;
	}

	switch( c2)
	{
		case '1':
		{
			c = FKB_F1;
			break;
		}

		case '2':
		{
			c = FKB_F2;
			break;
		}

		case '3':
		{
			c = FKB_F3;
			break;
		}

		case '4':
		{
			c = FKB_F4;
			break;
		}

		case '5':
		{
			c = FKB_F5;
			break;
		}

		case '6':
		{
			c = FKB_F6;
			break;
		}

		case '7':
		{
			c = FKB_F7;
			break;
		}

		case '8':
		{
			c = FKB_F8;
			break;
		}

		case '9':
		{
			c = FKB_F9;
			break;
		}

		case '0':
		{
			c = FKB_F10;
			break;
		}

		case 'i':
		{
			c = FKB_INSERT;
			break;
		}

		case 'd':
		{
			c = FKB_DELETE;
			break;
		}

		case KEY_F( 0):
		{
			c = FKB_ALT_F10;
			break;
		}

		case '\t':
		{
			c = FKB_STAB;
			break;
		}

		case KEY_FIND:
		case KEY_HOME:
		{
			c = FKB_CTRL_HOME;
			break;
		}

		case KEY_END:
		case KEY_SELECT:
		case KEY_LL:
		{
			c = FKB_CTRL_END;
			break;
		}

		case KEY_LEFT:
		{
			c = FKB_CTRL_LEFT;
			break;
		}

		case KEY_RIGHT:
		{
			c = FKB_CTRL_RIGHT;
			break;
		}

		case KEY_DOWN:
		{
			c = FKB_CTRL_DOWN;
			break;
		}

		case KEY_UP:
		{
			c = FKB_CTRL_UP;
			break;
		}

		case 0x000A:
		case 0x000D:
		case KEY_ENTER:
		{
			c = FKB_CTRL_ENTER;
			break;
		}

		case KEY_NPAGE:
		{
			c = FKB_CTRL_PGDN;
			break;
		}

		case KEY_PPAGE:
		{
			c = FKB_CTRL_PGUP;
			break;
		}

		case KEY_IC:
		{
			c = FKB_CTRL_INSERT;
			break;
		}

		default:
		{
			if( c2 >= '0' && c2 <= '9')
			{
				c = FKB_ALT_0 + c2 - '0';
			}
			else if( c2 >= 'a' && c2 <= 'z')
			{
				c = FKB_ALT_A + c2 - 'a';
			}
			else if( (c2 >= 1) && (c <= 032))
			{
				c = (int)ftxUnixSimulatedKey( (int)c2);
			}
			else if( c2 >= KEY_F(1) && c2 <= KEY_F( 10))
			{
				c = FKB_ALT_F1 + c - KEY_F(1);
			}
			else if( c2 == erasechar() || c2 == 0127)
			{
				c = FKB_ESCAPE;
			}
			else if( c2 == 033)
			{
				c = FKB_ESCAPE;
				break;
			}
			else
			{
				c = ERR;
				while( getch() != ERR);
			}
			
			break;
		}
	}
	
Exit:

	return( c);
}
#endif

/****************************************************************************
Desc:
Notes:	On Unix some terminal types (notably Solaris xterm) do not generate
			proper key codes for Insert, Home, PgUp, PgDn etc. Use a different
			terminal emulator (rxvt for eg). They can also be simulated by the
			key combination Meta-Shift-I, Meta-Shift-U, Meta-Shift-D etc.
****************************************************************************/
#ifdef FLM_UNIX
FSTATIC FLMUINT ftxUnixKBGetChar( void)
{
	int	c;

Again:

	if( ungetChar != ERR)
	{
		c = ungetChar;
		ungetChar = ERR;
	}
	else
	{
		while( (c = getch()) == ERR);
	}

	if (c == killchar())
	{
		c = FKB_DELETE;
	}
	else if (c == erasechar())
	{
		c = FKB_BACKSPACE;
	}
	else if (c == '\t')
	{
		c = FKB_TAB;
	}
	else if( c >= 1 && c <= 032 && c != 10 && c != 13)
	{
		c = FKB_CTRL_A + (c - 1);
	}
	else if ((c >= (128 + '0')) && (c <= (128 + '9')))
	{
		c = FKB_ALT_0 + (c - 128 - '0');
	}
	else if ((c >= (128 + 'a')) && (c <= (128 + 'z')))
	{
		c = FKB_ALT_A + (c - 128 - 'a');
	}
	else if ((c >= 128) && (c <= (128 + 032)))
	{
		c = (int)ftxUnixSimulatedKey( (int)(c - 128));
	}
	else if (c >= KEY_F(1) && c <= KEY_F(10))
	{
		c = FKB_F1 + c - KEY_F(1);
	}
	else if (c >= KEY_F(11) && c <= KEY_F(20))
	{
		c = FKB_SF1 + c - KEY_F(11);
	}
	else
	{
		switch( c)
		{
			case KEY_F( 0):
			{
				c = FKB_ALT_F10;
				break;
			}
	
			case KEY_BACKSPACE:
			{
				c = FKB_BACKSPACE;
				break;
			}
	
			case 033:
			{
				c = (int)ftxUnixHandleEsc();
				break;
			}
			
			case 0127:
			case KEY_DC:
			{
				c = FKB_DELETE;
				break;
			}
	
			case KEY_FIND:
			case KEY_HOME:
			{
				c = FKB_HOME;
				break;
			}
	
			case KEY_END:
			case KEY_SELECT:
			case KEY_LL:
			{
				c = FKB_END;
				break;
			}
	
			case KEY_LEFT:
			{
				c = FKB_LEFT;
				break;
			}
	
			case KEY_RIGHT:
			{
				c = FKB_RIGHT;
				break;
			}
	
			case KEY_DOWN:
			{
				c = FKB_DOWN;
				break;
			}
	
			case KEY_UP:
			{
				c = FKB_UP;
				break;
			}
	
			case 0x000A:
			case 0x000D:
			case KEY_ENTER:
			{
				c = FKB_ENTER;
				break;
			}
	
			case KEY_NPAGE:
			{
				c = FKB_PGDN;
				break;
			}
	
			case KEY_PPAGE:
			{
				c = FKB_PGUP;
				break;
			}
	
			case KEY_IC:
			{
				c = FKB_INSERT;
				break;
			}
		}
	}

	if( c == ERR)
	{
		goto Again;
	}
	
	return( (FLMUINT)c);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_UNIX
FSTATIC FLMBOOL ftxUnixKBTest( void)
{
	int	c;

	if( ungetChar != ERR)
	{
		c = ungetChar;
	}
	else
	{
		if( (c = getch()) != ERR)
		{
			ungetChar = c;
		}
	}
	
	return( (c == ERR) ? FALSE : TRUE);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI f_conInit(
	FLMUINT			uiRows,
	FLMUINT			uiCols,
	const char *	pszTitle)
{
	RCODE				rc = NE_FLM_OK;
	
	if( f_atomicInc( &gv_conInitCount) > 1)
	{
		goto Exit;
	}

	if( RC_BAD( rc = FTXInit( pszTitle, uiCols, uiRows, 
		FLM_BLACK, FLM_LIGHTGRAY, NULL, NULL)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_mutexCreate( &gv_hConMutex)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = FTXScreenInit( pszTitle, &gv_pConScreen)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = FTXWinInit( gv_pConScreen, gv_pConScreen->uiCols, 
		gv_pConScreen->uiRows, &gv_pConWindow)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FTXWinOpen( gv_pConWindow)))
	{
		goto Exit;
	}

Exit:

	if( RC_BAD( rc))
	{
		f_conExit();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI f_conExit( void)
{
	if( !gv_conInitCount || f_atomicDec( &gv_conInitCount) > 0)
	{
		return;
	}
	
	if( gv_pConWindow)
	{
		FTXWinFree( &gv_pConWindow);
	}
	
	if( gv_pConScreen)
	{
		FTXScreenFree( &gv_pConScreen);
	}

	if( gv_hConMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &gv_hConMutex);
	}
	
	FTXExit();
}

/****************************************************************************
Desc: Returns the size of the screen in columns and rows.
****************************************************************************/
void FTKAPI f_conGetScreenSize(
	FLMUINT *	puiNumColsRV,
	FLMUINT *	puiNumRowsRV)
{
	f_mutexLock( gv_hConMutex);
	FTXWinGetCanvasSize( gv_pConWindow, puiNumColsRV, puiNumRowsRV);
	f_mutexUnlock( gv_hConMutex);
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI f_conStrOut(
	const char *	pszString)
{
	f_mutexLock( gv_hConMutex);
	FTXWinPrintStr( gv_pConWindow, pszString);
	f_mutexUnlock( gv_hConMutex);
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI f_conStrOutXY(
	const char *	pszString,
	FLMUINT			uiCol,
	FLMUINT			uiRow)
{
	f_mutexLock( gv_hConMutex);
	FTXWinPrintStrXY( gv_pConWindow, pszString, uiCol, uiRow);
	f_mutexUnlock( gv_hConMutex);
}

/****************************************************************************
Desc:	Output a formatted string at present cursor location.
****************************************************************************/
void FTKAPI f_conPrintf(
	const char *	pszFormat, ...)
{
	char			szBuffer[ 512];
	f_va_list	args;

	f_va_start( args, pszFormat);
	f_vsprintf( szBuffer, pszFormat, &args);
	f_va_end( args);

	f_mutexLock( gv_hConMutex);
	FTXWinPrintStr( gv_pConWindow, szBuffer);
	f_mutexUnlock( gv_hConMutex);
}

/****************************************************************************
Desc:	Output a formatted string at present cursor location with color
****************************************************************************/
void FTKAPI f_conCPrintf(
	eColorType			back,
	eColorType			fore,
	const char *		pszFormat, ...)
{
	char				szBuffer[ 512];
	f_va_list		args;
	eColorType		oldBack;
	eColorType		oldFore;

	f_va_start( args, pszFormat);
	f_vsprintf( szBuffer, pszFormat, &args);
	f_va_end( args);

	f_mutexLock( gv_hConMutex);
	FTXWinGetBackFore( gv_pConWindow, &oldBack, &oldFore);
	FTXWinSetBackFore( gv_pConWindow, back, fore);
	FTXWinPrintStr( gv_pConWindow, szBuffer);
	FTXWinSetBackFore( gv_pConWindow, oldBack, oldFore);
	f_mutexUnlock( gv_hConMutex);
}

/****************************************************************************
Desc:    Clear the screen from the col/row down
****************************************************************************/
void FTKAPI f_conClearScreen(
	FLMUINT			uiCol,
	FLMUINT			uiRow)
{
	FLMUINT			uiCurrCol;
	FLMUINT			uiCurrRow;

	f_mutexLock( gv_hConMutex);

	FTXWinGetCursorPos( gv_pConWindow, &uiCurrCol, &uiCurrRow);

	if( uiCol == 255)
	{
		uiCol = uiCurrCol;
	}

	if( uiRow == 255)
	{
		uiRow = uiCurrRow;
	}

	FTXWinClearXY( gv_pConWindow, uiCol, uiRow);
	FTXWinSetCursorPos( gv_pConWindow, uiCol, uiRow);
	
	f_mutexUnlock( gv_hConMutex);
}

/****************************************************************************
Desc:    Position to the column and row specified.
Notes:   The NLM could call GetPositionOfOutputCursor(&r,&c);
****************************************************************************/
void FTKAPI f_conSetCursorPos(
	FLMUINT			uiCol,
	FLMUINT			uiRow)
{
	FLMUINT			uiCurrCol;
	FLMUINT			uiCurrRow;

	f_mutexLock( gv_hConMutex);
	
	FTXWinGetCursorPos( gv_pConWindow, &uiCurrCol, &uiCurrRow);

	if( uiCol == 255)
	{
		uiCol = uiCurrCol;
	}

	if( uiRow == 255)
	{
		uiRow = uiCurrRow;
	}

	FTXWinSetCursorPos( gv_pConWindow, uiCol, uiRow);
	
	f_mutexUnlock( gv_hConMutex);
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI f_conClearLine(
	FLMUINT			uiCol,
	FLMUINT			uiRow)
{
	FLMUINT			uiCurrCol;
	FLMUINT			uiCurrRow;

	f_mutexLock( gv_hConMutex);
	
	FTXWinGetCursorPos( gv_pConWindow, &uiCurrCol, &uiCurrRow);

	if( uiCol == 255)
	{
		uiCol = uiCurrCol;
	}

	if( uiRow == 255)
	{
		uiRow = uiCurrRow;
	}

	FTXWinClearLine( gv_pConWindow, uiCol, uiRow);
	
	f_mutexUnlock( gv_hConMutex);
}

/****************************************************************************
Desc:    Edit a line of data like gets(s).  Newline replaced by NULL character.
****************************************************************************/
FLMUINT FTKAPI f_conLineEdit(
	char *		pszString,
	FLMUINT		uiMaxLen)
{
	FLMUINT		uiCharCount;
	FLMUINT		uiCursorType;

	f_mutexLock( gv_hConMutex);
	
	uiCursorType = FTXWinGetCursorType( gv_pConWindow);
	FTXWinSetCursorType( gv_pConWindow, FLM_CURSOR_UNDERLINE);
	uiCharCount = FTXLineEd( gv_pConWindow, pszString, uiMaxLen);
	FTXWinSetCursorType( gv_pConWindow, uiCursorType);

	f_mutexUnlock( gv_hConMutex);

	return( uiCharCount);
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI f_conSetShutdown(
	FLMBOOL *    pbShutdown)
{
	FTXSetShutdownFlag( pbShutdown);
}

/****************************************************************************
Desc:    Edit a line of data with advanced features.
Ret:     Number of characters input.
****************************************************************************/
FLMUINT FTKAPI f_conLineEditExt(
	char *		pszBuffer,
	FLMUINT		uiBufSize,
	FLMUINT		uiMaxWidth,
	FLMUINT *	puiTermChar)
{
	FLMUINT		uiCharCount = 0;
	FLMUINT		uiCursorType;

	f_mutexLock( gv_hConMutex);
	
	uiCursorType = FTXWinGetCursorType( gv_pConWindow);
	FTXWinSetCursorType( gv_pConWindow, FLM_CURSOR_UNDERLINE);
	FTXLineEdit( gv_pConWindow, pszBuffer, uiBufSize, uiMaxWidth,
		&uiCharCount, puiTermChar);
	FTXWinSetCursorType( gv_pConWindow, uiCursorType);
	f_mutexUnlock( gv_hConMutex);

	return( (FLMINT)uiCharCount);
}

/****************************************************************************
Desc:	Get the current X coordinate of the cursor
****************************************************************************/
FLMUINT FTKAPI f_conGetCursorColumn( void)
{
	FLMUINT		uiCol;

	f_mutexLock( gv_hConMutex);
	FTXWinGetCursorPos( gv_pConWindow, &uiCol, NULL);
	f_mutexUnlock( gv_hConMutex);

	return( uiCol);
}

/****************************************************************************
Desc:	Get the current Y coordinate of the cursor
****************************************************************************/
FLMUINT FTKAPI f_conGetCursorRow( void)
{
	FLMUINT		uiRow;

	f_mutexLock( gv_hConMutex);
	FTXWinGetCursorPos( gv_pConWindow, NULL, &uiRow);
	f_mutexUnlock( gv_hConMutex);

	return( uiRow);
}

/****************************************************************************
Desc:    Set the background and foreground colors
****************************************************************************/
void FTKAPI f_conSetBackFore(
	eColorType		backColor,
	eColorType		foreColor)
{
	f_mutexLock( gv_hConMutex);
	FTXWinSetBackFore( gv_pConWindow, backColor, foreColor);
	f_mutexUnlock( gv_hConMutex);
}

/****************************************************************************
Desc : Sets the cursor attributes.
****************************************************************************/
void FTKAPI f_conSetCursorType(
	FLMUINT		uiType)
{
	f_mutexLock( gv_hConMutex);
	FTXWinSetCursorType( gv_pConWindow, uiType);
	f_mutexUnlock( gv_hConMutex);
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI f_conDrawBorder( void)
{
	f_mutexLock( gv_hConMutex);
	FTXWinDrawBorder( gv_pConWindow);
	f_mutexUnlock( gv_hConMutex);
}

/****************************************************************************
Desc:
Not*************************************************************************/
FLMUINT FTKAPI f_conGetKey( void)
{
	FLMUINT		uiChar;

	f_mutexLock( gv_hConMutex);
	FTXWinInputChar( gv_pConWindow, &uiChar);
	f_mutexUnlock( gv_hConMutex);
	
	return( uiChar);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMBOOL FTKAPI f_conHaveKey( void)
{
	FLMBOOL		bHaveKey;

	f_mutexLock( gv_hConMutex);
	bHaveKey = FTXWinTestKB( gv_pConWindow) == NE_FLM_OK ? TRUE : FALSE;
	f_mutexUnlock( gv_hConMutex);

	return( bHaveKey);
}
