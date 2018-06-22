//------------------------------------------------------------------------------
// Desc:	This file contains the routines which initialize and setup
//			menus in the VIEW program.
// Tabs:	3
//
// Copyright (c) 1992-1995, 1998-2000, 2002-2007 Novell, Inc.
// All Rights Reserved.
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

// These variables are for displaying date and time

FSTATIC const char * Months[] = {
	"Jan",
	"Feb",
	"Mar",
	"Apr",
	"May",
	"Jun",
	"Jul",
	"Aug",
	"Sep",
	"Oct",
	"Nov",
	"Dec"
};

// Local Function Prototypes

FSTATIC void ViewDispMenuItem(
	VIEW_MENU_ITEM_p	pMenuItem);

FSTATIC void ViewRefreshMenu(
	VIEW_MENU_ITEM_p	pPrevItem);

FSTATIC void UpdateHorizCursor(
	FLMBOOL	bOnFlag);

FSTATIC void DoUpArrow( void);

FSTATIC void DoDownArrow( void);

FSTATIC void DoPageDown( void);

FSTATIC void DoPageUp( void);

FSTATIC void DoHome( void);

FSTATIC void DoEnd( void);

FSTATIC void DoRightArrow( void);

FSTATIC void DoLeftArrow( void);

FSTATIC void ByteToHex(
	FLMBYTE		ucChar,
	char *		pszDestBuf,
	FLMBOOL		bUpperCaseFlag);

FSTATIC void ViewHelpScreen( void);

extern FLMUINT	gv_uiTopLine;
extern FLMUINT	gv_uiBottomLine;


/***************************************************************************
Desc:	This routine frees the memory used to hold menu items.
*****************************************************************************/
void ViewFreeMenuMemory( void)
{
	if (gv_pViewMenuFirstItem)
	{
		gv_pViewPool->poolFree();
		gv_pViewMenuFirstItem = NULL;
		gv_pViewMenuLastItem = NULL;
		gv_pViewMenuCurrItem = NULL;
	}
}

/***************************************************************************
Desc:	This routine initializes variables to start a new menu.
*****************************************************************************/
void ViewMenuInit(
	const char *	pszTitle)
{
	FLMUINT	uiCol;

	// Clear the screen and display the menu title

	f_conSetBackFore( FLM_BLACK, FLM_WHITE);
	f_conClearScreen( 0, 0);

	// Display the title in the middle of the top line of the screen

	uiCol = (80 - f_strlen( pszTitle)) / 2;

	f_conStrOutXY( pszTitle, uiCol, 0);
	ViewUpdateDate( TRUE, &gv_ViewLastTime);

	// Deallocate any old memory, if any

	ViewFreeMenuMemory();
}

/***************************************************************************
Desc:	This routine converts a FLMBYTE value to two ASCII Hex characters.
*****************************************************************************/
FSTATIC void ByteToHex(
	FLMBYTE		ucChar,
	char *		pszDestBuf,
	FLMBOOL		bUpperCaseFlag
	)
{
	FLMBYTE	ucTempChar;

	// Convert the upper four bits

	if ((ucTempChar = ((ucChar & 0xF0) >> 4)) <= 9)
	{
		*pszDestBuf = '0' + ucTempChar;
	}
	else if (bUpperCaseFlag)
	{
		*pszDestBuf = 'A' + ucTempChar - 10;
	}
	else
	{
		*pszDestBuf = 'a' + ucTempChar - 10;
	}
	pszDestBuf++;

	// Convert the lower four bits

	if ((ucTempChar = (ucChar & 0x0F)) <= 9)
	{
		*pszDestBuf = '0' + ucTempChar;
	}
	else if (bUpperCaseFlag)
	{
		*pszDestBuf = 'A' + ucTempChar - 10;
	}
	else
	{
		*pszDestBuf = 'a' + ucTempChar - 10;
	}
	pszDestBuf++;

	// Terminate with a NULL character

	*pszDestBuf = 0;
}

/***************************************************************************
Desc:	This routine displays a menu item on the screen.
*****************************************************************************/
FSTATIC void ViewDispMenuItem(
	VIEW_MENU_ITEM_p  pMenuItem
	)
{
	FLMUINT		uiRow;
	FLMUINT		uiCol = pMenuItem->uiCol;
	FLMUINT		uiLoop;
	char			szTempBuf [80];

	if (!gv_bViewEnabled)
	{
		return;
	}

	// Calculate row and column where the item is to be displayed

	uiRow = gv_uiTopLine + (pMenuItem->uiRow - gv_uiViewTopRow);

	// If it is a HEX display, output the address first

	if ((pMenuItem->uiValueType & 0x0F) == VAL_IS_BINARY_HEX)
	{
		f_sprintf( (char *)szTempBuf, "%03u:%08X",
				(unsigned)pMenuItem->uiModFileNumber,
				(unsigned)pMenuItem->uiModFileOffset);
		f_conSetBackFore( FLM_WHITE, FLM_GREEN);
		f_conStrOutXY( szTempBuf, 0, uiRow);
	}

	// If the item is the current item, display it using the
	// select colors.  Otherwise, display using the unselect colors

	if (pMenuItem->uiItemNum == gv_uiViewMenuCurrItemNum)
	{
		if (!gv_pViewMenuCurrItem)
		{
			gv_pViewMenuCurrItem = pMenuItem;
		}
		if (pMenuItem->uiOption)
		{
			f_conSetBackFore( pMenuItem->uiSelectBackColor,
											 pMenuItem->uiSelectForeColor);
		}
		else
		{
			f_conSetBackFore( pMenuItem->uiUnselectForeColor,
											 pMenuItem->uiUnselectBackColor);
		}
	}
	else
	{
		f_conSetBackFore( pMenuItem->uiUnselectBackColor,
										 pMenuItem->uiUnselectForeColor);
	}

	if (pMenuItem->iLabelIndex < 0)
	{
		uiCol += (pMenuItem->uiLabelWidth + 1);
		szTempBuf[ 0] = 0;
	}
	else if (!pMenuItem->uiLabelWidth)
	{
		f_strcpy( szTempBuf, Labels[ pMenuItem->iLabelIndex]);
		f_strcpy( &szTempBuf[ f_strlen( szTempBuf)], " ");
	}
	else
	{
		for( uiLoop = 0; uiLoop < pMenuItem->uiLabelWidth; uiLoop++)
		{
			szTempBuf[ uiLoop] = '.';
		}
		szTempBuf[ pMenuItem->uiLabelWidth] = ' ';
		szTempBuf[ pMenuItem->uiLabelWidth + 1] = 0;
		f_memcpy( szTempBuf, Labels[ pMenuItem->iLabelIndex],
							(FLMSIZET)f_strlen( Labels[ pMenuItem->iLabelIndex]));
	}
	if (pMenuItem->uiOption)
	{
		if (pMenuItem->uiItemNum == gv_uiViewMenuCurrItemNum)
		{
			f_conStrOutXY( "*>", (uiCol - 2), uiRow);
		}
		else
		{
			f_conStrOutXY( "* ", (uiCol - 2), uiRow);
		}
	}
	else
	{
		if (pMenuItem->uiItemNum == gv_uiViewMenuCurrItemNum)
		{
			f_conStrOutXY( " >", (uiCol - 2), uiRow);
		}
		else
		{
			f_conStrOutXY( "  ", (uiCol - 2), uiRow);
		}
	}
	if (szTempBuf[ 0])
	{
		f_conStrOutXY( szTempBuf, uiCol, uiRow);
	}

	// Now output the value

	uiCol += f_strlen( szTempBuf);
	switch (pMenuItem->uiValueType & 0x0F)
	{
		case VAL_IS_LABEL_INDEX:
			f_conStrOutXY( (Labels[ pMenuItem->ui64Value]),
									 uiCol, uiRow);
			break;
		case VAL_IS_ERR_INDEX:
			{
			FLMUINT  ValIndex = (FLMUINT)pMenuItem->ui64Value;
			f_conStrOutXY( gv_pDbSystem->checkErrorToStr( ValIndex),
							uiCol, uiRow);
			break;
			}
		case VAL_IS_TEXT_PTR:
			f_conStrOutXY( ((char *)(FLMUINT)pMenuItem->ui64Value), uiCol, uiRow);
			break;
		case VAL_IS_BINARY_HEX:
		case VAL_IS_BINARY_PTR:
		{
			FLMUINT		uiBytesPerLine = MAX_HORIZ_SIZE( uiCol);
			FLMUINT		uiBytesProcessed = 0;
			FLMUINT		uiLoopJ;
			FLMUINT		uiLoopK;
			FLMUINT		uiNumBytes;
			FLMBYTE *	pucVal = (FLMBYTE *)(FLMUINT)pMenuItem->ui64Value;
			FLMUINT		uiValLen = pMenuItem->uiValueLen;

			// Process each character in the value

			uiLoop = 0;
			uiLoopJ = 0;
			uiLoopK = uiBytesPerLine * 3 + 5;
			uiNumBytes = 0;

			// Fill up a single line with whatever will fit on the line in
			// hex format.

			f_memset( szTempBuf, ' ', 80);
			szTempBuf[ uiLoopK - 3] = '|';
			while (uiBytesProcessed < uiValLen && uiLoop < uiBytesPerLine)
			{
				ByteToHex( pucVal[ uiBytesProcessed], &szTempBuf[ uiLoopJ], TRUE);
				if (pucVal[ uiBytesProcessed] > ' ' &&
					 pucVal[ uiBytesProcessed] <= 127)
				{
					szTempBuf [uiLoopK] = pucVal [uiBytesProcessed];
				}
				else
				{
					szTempBuf [uiLoopK] = '.';
				}
				uiLoopK++;
				uiNumBytes++;
				uiBytesProcessed++;
				uiLoop++;
				uiLoopJ += 2;
				szTempBuf [uiLoopJ] = ' ';
				uiLoopJ++;
			}
			szTempBuf [uiLoopK] = 0;

			// Output the line

			f_conStrOutXY( szTempBuf, uiCol, uiRow);
			if (pMenuItem->uiItemNum == gv_uiViewMenuCurrItemNum)
			{
				UpdateHorizCursor( TRUE);
			}
			break;
		}
		case VAL_IS_NUMBER:
			switch (pMenuItem->uiValueType & 0xF0)
			{
				case DISP_DECIMAL:
					f_sprintf( (char *)szTempBuf, "%I64u", pMenuItem->ui64Value);
					break;
				case DISP_HEX:
					if (pMenuItem->ui64Value == 0xFFFFFFFF)
					{
						f_strcpy( szTempBuf, "None");
					}
					else if (pMenuItem->ui64Value == 0)
					{
						f_strcpy( szTempBuf, "0");
					}
					else
					{
						szTempBuf [0] = '0';
						szTempBuf [1] = 'x';
						f_sprintf( (char *)&szTempBuf[ 2], "%I64X",
							pMenuItem->ui64Value);
					}
					break;
				case DISP_DECIMAL_HEX:
					f_sprintf( (char *)szTempBuf, "%I64u (0x%I64X)",
						pMenuItem->ui64Value, pMenuItem->ui64Value);
					break;
				case DISP_HEX_DECIMAL:
				default:
					if (pMenuItem->ui64Value == 0xFFFFFFFF)
					{
						f_strcpy( szTempBuf, "None");
					}
					else if (pMenuItem->ui64Value == 0)
					{
						f_strcpy( szTempBuf, "0");
					}
					else
					{
						f_sprintf( (char *)szTempBuf, "0x%I64X (%I64u)",
							pMenuItem->ui64Value, pMenuItem->ui64Value);
					}
					break;
			}
			f_conStrOutXY( szTempBuf, uiCol, uiRow);
			break;
		case VAL_IS_EMPTY:
		default:
			break;
	}
}

/***************************************************************************
Desc:	This routine adds a menu item to the item list.
*****************************************************************************/
FLMBOOL ViewAddMenuItem(
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
	eColorType	uiSelectForeColor)
{
	FLMBOOL				bOk = FALSE;
	VIEW_MENU_ITEM_p	pMenuItem;
	FLMUINT				uiSize = sizeof( VIEW_MENU_ITEM);

	// Allocate the memory for the item

	if ((uiValueType & 0x0F) == VAL_IS_TEXT_PTR)
	{
		uiSize += (f_strlen( (const char *)((FLMUINT)ui64Value)) + 1);
	}
	else if ((uiValueType & 0x0F) == VAL_IS_BINARY)
	{
		uiSize += uiValueLen;
	}
	if (RC_BAD( gv_pViewPool->poolAlloc(uiSize, (void **)&pMenuItem)))
	{
		ViewShowError( "Could not allocate memory for menu value");
		goto Exit;
	}

	// Link the item into the array and set some values in the item

	pMenuItem->pNextItem = NULL;
	pMenuItem->pPrevItem = gv_pViewMenuLastItem;
	if (gv_pViewMenuLastItem)
	{
		gv_pViewMenuLastItem->pNextItem = pMenuItem;
		pMenuItem->uiItemNum = gv_pViewMenuLastItem->uiItemNum + 1;
	}
	else
	{
		gv_pViewMenuFirstItem = pMenuItem;
		pMenuItem->uiItemNum = 0;
	}
	gv_pViewMenuLastItem = pMenuItem;
	pMenuItem->iLabelIndex = iLabelIndex;
	pMenuItem->uiLabelWidth = uiLabelWidth;
	pMenuItem->uiValueType = uiValueType;
	if ((uiValueType & 0x0F) == VAL_IS_TEXT_PTR)
	{
		pMenuItem->ui64Value = (FLMUINT64)((FLMUINT)&pMenuItem[ 1]);
		f_strcpy( (char *)(FLMUINT)pMenuItem->ui64Value,
			(const char *)(FLMUINT)ui64Value);
	}
	else if ((uiValueType & 0x0F) == VAL_IS_BINARY)
	{
		pMenuItem->uiValueType = VAL_IS_BINARY_PTR;
		pMenuItem->ui64Value = (FLMUINT64)((FLMUINT)&pMenuItem[ 1]);
		f_memcpy( (void *)(FLMUINT)pMenuItem->ui64Value, 
			(void *)(FLMUINT)ui64Value, uiValueLen);
	}
	else
	{
		pMenuItem->ui64Value = ui64Value;
	}
	pMenuItem->uiValueLen = uiValueLen;
	pMenuItem->uiModFileOffset = uiModFileOffset;
	pMenuItem->uiModFileNumber = uiModFileNumber;
	pMenuItem->uiModBufLen = uiModBufLen;
	pMenuItem->uiModType = uiModType;
	pMenuItem->uiCol = uiCol;
	pMenuItem->uiRow = uiRow;
	pMenuItem->uiOption = uiOption;
	pMenuItem->uiUnselectBackColor = uiUnselectBackColor;
	pMenuItem->uiUnselectForeColor = uiUnselectForeColor;
	pMenuItem->uiSelectBackColor = uiSelectBackColor;
	pMenuItem->uiSelectForeColor = uiSelectForeColor;
	pMenuItem->uiHorizCurPos = 0;
	if (pMenuItem->uiRow >= gv_uiViewTopRow &&
		 pMenuItem->uiRow <= gv_uiViewBottomRow)
	{
		ViewDispMenuItem( pMenuItem);
	}
	bOk = TRUE;

Exit:

	return( bOk);
}

/***************************************************************************
Desc:	This routine displays the prompt to press ESCAPE.  This prompt
		appears at the bottom of every screen.
*****************************************************************************/
void ViewEscPrompt( void)
{
	FLMUINT	uiNumCols;
	FLMUINT	uiNumRows;

	f_conGetScreenSize( &uiNumCols, &uiNumRows);
	f_conSetBackFore( FLM_BLACK, FLM_WHITE);
	f_conClearScreen( 0, uiNumRows - 1);
	f_conSetBackFore( FLM_RED, FLM_WHITE);
	f_conStrOutXY( "ESC=Exit, ?=Help", 0, uiNumRows - 1);
	f_conSetBackFore( FLM_BLACK, FLM_WHITE);
	f_conStrOutXY( "File: ", 20, uiNumRows - 1);
	f_conStrOutXY( gv_szViewFileName, 26, uiNumRows - 1);
	gv_uiViewLastFileOffset = VIEW_INVALID_FILE_OFFSET;
}

/***************************************************************************
Desc:	This routine refreshes the menu display.  If NULL is passed in
		the entire screen is refreshed.  Otherwise, the item passed as
		well as the current item are refreshed.
*****************************************************************************/
FSTATIC void ViewRefreshMenu(
	VIEW_MENU_ITEM_p	pPrevItem
	)
{
	VIEW_MENU_ITEM_p  pMenuItem;

	gv_uiViewMenuCurrItemNum = gv_pViewMenuCurrItem->uiItemNum;
	if (pPrevItem)
	{
		ViewDispMenuItem( pPrevItem);
		ViewDispMenuItem( gv_pViewMenuCurrItem);
	}
	else
	{

		// Refresh the entire screen

		f_conSetBackFore( FLM_BLACK, FLM_WHITE);
		f_conClearScreen( 0, 1);

		pMenuItem = gv_pViewMenuFirstItem;
		while( pMenuItem)
		{
			if (pMenuItem->uiRow >= gv_uiViewTopRow &&
				 pMenuItem->uiRow <= gv_uiViewBottomRow)
			{
				ViewDispMenuItem( pMenuItem);
			}
			pMenuItem = pMenuItem->pNextItem;
		}
		ViewEscPrompt();
	}
}

/********************************************************************
Desc:	Update the horizontal cursor
*********************************************************************/
FSTATIC void UpdateHorizCursor(
	FLMBOOL	bOnFlag
	)
{
	unsigned char		szTempBuf[ 4];
	FLMUINT		uiLoop;
	FLMUINT		uiRow;
	FLMUINT		uiCol = gv_pViewMenuCurrItem->uiCol + 1;
	FLMBYTE *	pucVal = ((FLMBYTE *)(FLMUINT)gv_pViewMenuCurrItem->ui64Value);

	if (gv_pViewMenuCurrItem->uiHorizCurPos >
			HORIZ_SIZE( gv_pViewMenuCurrItem) - 1)
	{
		gv_pViewMenuCurrItem->uiHorizCurPos =
			HORIZ_SIZE( gv_pViewMenuCurrItem) - 1;
	}
	uiLoop = gv_pViewMenuCurrItem->uiHorizCurPos;
	ByteToHex( *(pucVal + uiLoop), (char *)szTempBuf, TRUE);
	szTempBuf[ 2] = 0;
	if (bOnFlag)
	{
		f_conSetBackFore( FLM_RED, FLM_WHITE);
	}
	else
	{
		f_conSetBackFore( gv_pViewMenuCurrItem->uiUnselectForeColor,
									 gv_pViewMenuCurrItem->uiUnselectBackColor);
	}

	// Calculate row and column where the item is to be displayed

	uiRow = gv_uiTopLine + (gv_pViewMenuCurrItem->uiRow - gv_uiViewTopRow);
	f_conStrOutXY( (char *)szTempBuf, (uiCol + uiLoop * 3), uiRow);
	if (((szTempBuf[ 0] = pucVal[ uiLoop]) < ' ') ||
			(szTempBuf[ 0] > 127))
	{
		szTempBuf[ 0] = ' ';
	}
#if defined( FLM_WIN)
	if (bOnFlag)
	{
		szTempBuf [0] = 128;
	}
#endif
	szTempBuf[ 1] = 0;
	f_conStrOutXY( (char *)szTempBuf,
			(uiCol + MAX_HORIZ_SIZE( uiCol) * 3 + 5 + uiLoop), uiRow);
	f_conSetBackFore( gv_pViewMenuCurrItem->uiUnselectForeColor,
								 gv_pViewMenuCurrItem->uiUnselectBackColor);
	if (bOnFlag)
	{
		f_conStrOutXY( ">", (uiCol + uiLoop * 3 - 1), uiRow);
	}
	else
	{
		f_conStrOutXY( " ", (uiCol + uiLoop * 3 - 1), uiRow);
	}
}

/********************************************************************
Desc:	Handle up arrow key
*********************************************************************/
FSTATIC void DoUpArrow( void)
{
	VIEW_MENU_ITEM_p	pPrevItem;

	if ((pPrevItem = gv_pViewMenuCurrItem->pPrevItem) != NULL)
	{
		if (pPrevItem->uiHorizCurPos != gv_pViewMenuCurrItem->uiHorizCurPos)
		{
			pPrevItem->uiHorizCurPos = gv_pViewMenuCurrItem->uiHorizCurPos;
		}
		gv_pViewMenuCurrItem = pPrevItem;
		if (gv_pViewMenuCurrItem->uiRow < gv_uiViewTopRow)
		{
			gv_uiViewTopRow--;
			gv_uiViewBottomRow--;
			if (gv_pViewMenuCurrItem->uiRow < gv_uiViewTopRow)
			{
				gv_pViewMenuCurrItem = gv_pViewMenuCurrItem->pNextItem;
			}
			ViewRefreshMenu( NULL);
		}
		else
		{
			ViewRefreshMenu( gv_pViewMenuCurrItem->pNextItem);
		}
	}
}

/********************************************************************
Desc:	Handle down arrow key
*********************************************************************/
FSTATIC void DoDownArrow( void)
{
	VIEW_MENU_ITEM_p	pNextItem;

	if ((pNextItem = gv_pViewMenuCurrItem->pNextItem) != NULL)
	{
		if (pNextItem->uiHorizCurPos != gv_pViewMenuCurrItem->uiHorizCurPos)
		{
			pNextItem->uiHorizCurPos = gv_pViewMenuCurrItem->uiHorizCurPos;
		}
		gv_pViewMenuCurrItem = pNextItem;
		if (gv_pViewMenuCurrItem->uiRow > gv_uiViewBottomRow)
		{
			gv_uiViewTopRow++;
			gv_uiViewBottomRow++;
			if (gv_pViewMenuCurrItem->uiRow > gv_uiViewBottomRow)
			{
				gv_pViewMenuCurrItem = gv_pViewMenuCurrItem->pPrevItem;
			}
			ViewRefreshMenu( NULL);
		}
		else
		{
			ViewRefreshMenu( gv_pViewMenuCurrItem->pPrevItem);
		}
	}
}

/********************************************************************
Desc:	Handle page down key
*********************************************************************/
FSTATIC void DoPageDown( void)
{
	FLMUINT				uiTargetRow;
	VIEW_MENU_ITEM_p	pSaveItem;

	if (gv_uiViewBottomRow < gv_pViewMenuLastItem->uiRow)
	{
		gv_uiViewBottomRow += LINES_PER_PAGE;
		if (gv_uiViewBottomRow > gv_pViewMenuLastItem->uiRow)
		{
			gv_uiViewBottomRow = gv_pViewMenuLastItem->uiRow;
		}
		gv_uiViewTopRow = gv_uiViewBottomRow - LINES_PER_PAGE + 1;
		uiTargetRow = gv_pViewMenuCurrItem->uiRow + LINES_PER_PAGE;
		if (uiTargetRow > gv_uiViewBottomRow)
		{
			uiTargetRow = gv_uiViewBottomRow;
		}
		while (gv_pViewMenuCurrItem->pNextItem &&
				 gv_pViewMenuCurrItem->uiRow < uiTargetRow)
		{
			gv_pViewMenuCurrItem = gv_pViewMenuCurrItem->pNextItem;
		}
		if (gv_pViewMenuCurrItem->uiRow > gv_uiViewBottomRow)
		{
			gv_pViewMenuCurrItem = gv_pViewMenuCurrItem->pPrevItem;
		}
		ViewRefreshMenu( NULL);
	}
	else if (gv_pViewMenuCurrItem->pNextItem)
	{
		pSaveItem = gv_pViewMenuCurrItem;
		gv_pViewMenuCurrItem = gv_pViewMenuLastItem;
		ViewRefreshMenu( pSaveItem);
	}
}

/********************************************************************
Desc:	Handle the page up key
*********************************************************************/
FSTATIC void DoPageUp( void)
{
	FLMUINT				uiTargetRow;
	VIEW_MENU_ITEM_p	pSaveItem;

	if (gv_uiViewTopRow > 0)
	{
		if (gv_uiViewTopRow < LINES_PER_PAGE)
		{
			gv_uiViewTopRow = 0;
		}
		else
		{
			gv_uiViewTopRow -= LINES_PER_PAGE;
		}
		gv_uiViewBottomRow = gv_uiViewTopRow + LINES_PER_PAGE - 1;
		uiTargetRow = gv_pViewMenuCurrItem->uiRow - LINES_PER_PAGE;
		if (uiTargetRow < gv_uiViewTopRow)
		{
			uiTargetRow = gv_uiViewTopRow;
		}
		while (gv_pViewMenuCurrItem->pPrevItem &&
				 gv_pViewMenuCurrItem->uiRow > uiTargetRow)
		{
			gv_pViewMenuCurrItem = gv_pViewMenuCurrItem->pPrevItem;
		}
		if (gv_pViewMenuCurrItem->uiRow < gv_uiViewTopRow)
		{
			gv_pViewMenuCurrItem = gv_pViewMenuCurrItem->pNextItem;
		}
		ViewRefreshMenu( NULL);
	}
	else if (gv_pViewMenuCurrItem->pPrevItem != NULL)
	{
		pSaveItem = gv_pViewMenuCurrItem;
		gv_pViewMenuCurrItem = gv_pViewMenuFirstItem;
		ViewRefreshMenu( pSaveItem);
	}
}

/********************************************************************
Desc:	Handle home key
*********************************************************************/
FSTATIC void DoHome( void)
{
	VIEW_MENU_ITEM_p	pSaveItem;

	if (gv_uiViewTopRow != 0)
	{
		gv_uiViewTopRow = 0;
		gv_uiViewBottomRow = gv_uiViewTopRow + LINES_PER_PAGE - 1;
		gv_pViewMenuCurrItem = gv_pViewMenuFirstItem;
		ViewRefreshMenu( NULL);
	}
	else if (gv_pViewMenuCurrItem->pPrevItem)
	{
		pSaveItem = gv_pViewMenuCurrItem;
		gv_pViewMenuCurrItem = gv_pViewMenuFirstItem;
		ViewRefreshMenu( pSaveItem);
	}
}

/********************************************************************
Desc:	Handle end key
*********************************************************************/
FSTATIC void DoEnd( void)
{
	VIEW_MENU_ITEM_p	pSaveItem;

	if (gv_uiViewBottomRow < gv_pViewMenuLastItem->uiRow)
	{
		gv_uiViewBottomRow = gv_pViewMenuLastItem->uiRow;
		gv_uiViewTopRow = gv_uiViewBottomRow - LINES_PER_PAGE + 1;
		gv_pViewMenuCurrItem = gv_pViewMenuLastItem;
		ViewRefreshMenu( NULL);
	}
	else if (gv_pViewMenuCurrItem->pNextItem)
	{
		pSaveItem = gv_pViewMenuCurrItem;
		gv_pViewMenuCurrItem = gv_pViewMenuLastItem;
		ViewRefreshMenu( pSaveItem);
	}
}

/********************************************************************
Desc:	Handle right arrow key
*********************************************************************/
FSTATIC void DoRightArrow( void)
{
	if (!HAVE_HORIZ_CUR( gv_pViewMenuCurrItem) ||
		 gv_pViewMenuCurrItem->uiHorizCurPos == HORIZ_SIZE( gv_pViewMenuCurrItem) - 1)
	{
		if (gv_pViewMenuCurrItem->uiItemNum != gv_pViewMenuLastItem->uiItemNum)
		{
			gv_pViewMenuCurrItem->uiHorizCurPos = 0;
			DoDownArrow();
		}
	}
	else if (gv_pViewMenuCurrItem->uiHorizCurPos <
					 HORIZ_SIZE( gv_pViewMenuCurrItem) - 1)
	{
		UpdateHorizCursor( FALSE);
		gv_pViewMenuCurrItem->uiHorizCurPos++;
		UpdateHorizCursor( TRUE);
	}
}

/********************************************************************
Desc:	Handle left arrow key
*********************************************************************/
FSTATIC void DoLeftArrow( void)
{
	if (!HAVE_HORIZ_CUR( gv_pViewMenuCurrItem) ||
		 gv_pViewMenuCurrItem->uiHorizCurPos == 0)
	{
		if (gv_pViewMenuCurrItem->pPrevItem)
		{
			gv_pViewMenuCurrItem->uiHorizCurPos =
				HORIZ_SIZE( gv_pViewMenuCurrItem->pPrevItem) - 1;
		}
		DoUpArrow();
	}
	else if (gv_pViewMenuCurrItem->uiHorizCurPos > 0)
	{
		UpdateHorizCursor( FALSE);
		gv_pViewMenuCurrItem->uiHorizCurPos--;
		UpdateHorizCursor( TRUE);
	}
}

/***************************************************************************
Desc:	This routine displays a help screen showing available commands.
*****************************************************************************/
FSTATIC void ViewHelpScreen( void)
{
	FLMUINT	uiChar;

	// Clear the screen and display the menu title

	f_conSetBackFore( FLM_BLACK, FLM_WHITE);
	f_conClearScreen( 0, 1);
	f_conSetCursorPos( 0, 3);
	f_conStrOut( "     RECOGNIZED KEYBOARD CHARACTERS\n");
	f_conStrOut( "\n");
	f_conStrOut( "     ESCAPE              - Exit Screen\n");
	f_conStrOut( "     U,u,8               - Up Arrow\n");
	f_conStrOut( "     D,d,2               - Down Arrow\n");
	f_conStrOut( "     +,3                 - Page Down\n");
	f_conStrOut( "     R,r,6               - Right Arrow\n");
	f_conStrOut( "     L,l,5               - Left Arrow\n");
	f_conStrOut( "     -,9                 - Page Up\n");
	f_conStrOut( "     H,h,7               - Home\n");
	f_conStrOut( "     Z,z,1               - End\n");
	f_conStrOut( "     E,e                 - Edit Data\n");
	f_conStrOut( "     A,a                 - Edit Data in RAW Mode (no checksum)\n");
	f_conStrOut( "     G,g                 - Goto Block\n");
	f_conStrOut( "     X,x                 - Display Hex\n");
	f_conStrOut( "     Y,y                 - Display Decrypted\n");
	f_conStrOut( "     S,s                 - Search\n");
	f_conStrOut( "     ?                   - Show this help screen\n");
	f_conStrOut( "\n");
	f_conStrOut( "     PRESS ANY CHARACTER TO EXIT HELP SCREEN\n");

	for (;;)
	{

		// Update date and time

		ViewUpdateDate( FALSE, &gv_ViewLastTime);

		// See what character was pressed

		uiChar = (!f_conHaveKey()) ? 0 : (f_conGetKey());
		if (gv_bShutdown)
		{
			return;
		}
		if (uiChar)
		{
			break;
		}
		viewGiveUpCPU();
	}
	ViewRefreshMenu( NULL);
}

/***************************************************************************
Desc:	This routine allows the user to press keys while in a menu.
		Keys for navigating through the menu are handled inside this
		routine.  Other keys are passed to the calling routine or are
		ignored altogether.
*****************************************************************************/
FLMUINT ViewGetMenuOption( void)
{
	FLMUINT	uiChar;

	// Make sure we have a pointer to the current item

	if (!gv_pViewMenuCurrItem)
	{
		gv_pViewMenuCurrItem = gv_pViewMenuFirstItem;
		while (gv_pViewMenuCurrItem->pNextItem &&
				 gv_pViewMenuCurrItem->uiItemNum < gv_uiViewMenuCurrItemNum)
		{
			gv_pViewMenuCurrItem = gv_pViewMenuCurrItem->pNextItem;
		}
		if (gv_pViewMenuCurrItem->uiItemNum != gv_uiViewMenuCurrItemNum)
		{
			gv_uiViewMenuCurrItemNum = gv_pViewMenuCurrItem->uiItemNum;
			ViewDispMenuItem( gv_pViewMenuCurrItem);
		}
	}

	// Loop getting user input

	ViewEscPrompt();
	for (;;)
	{

		// Set the file position of the current object

		if (gv_pViewMenuCurrItem->uiModFileOffset != VIEW_INVALID_FILE_OFFSET)
		{
			gv_uiViewCurrFileOffset = gv_pViewMenuCurrItem->uiModFileOffset;
			gv_uiViewCurrFileNumber = gv_pViewMenuCurrItem->uiModFileNumber;
			if (HAVE_HORIZ_CUR( gv_pViewMenuCurrItem))
			{
				gv_uiViewCurrFileOffset += gv_pViewMenuCurrItem->uiHorizCurPos;
			}
		}

		// Update date and time

		ViewUpdateDate( FALSE, &gv_ViewLastTime);

		// See what character was pressed

		viewGiveUpCPU();
		uiChar = (FLMUINT)(!f_conHaveKey()
								 ? (FLMUINT)0
								 : (FLMUINT)f_conGetKey());
		if (gv_bShutdown)
		{
			return( ESCAPE_OPTION);
		}
		switch( uiChar)
		{
			case FKB_ESCAPE:
				return( ESCAPE_OPTION);
			case FKB_UP:
			case 'U':
			case 'u':
			case '8':
				DoUpArrow();
				break;
			case FKB_DOWN:
			case 'D':
			case 'd':
			case '2':
				DoDownArrow();
				break;
			case FKB_PGDN:
			case '+':
			case '3':
				DoPageDown();
				break;
			case FKB_PGUP:
			case '-':
			case '9':
				DoPageUp();
				break;
			case FKB_HOME:
			case 'H':
			case 'h':
			case '7':
				DoHome();
				break;
			case FKB_END:
			case 'Z':
			case 'z':
			case '1':
				DoEnd();
				break;
			case '\n':
			case '\r':
			case FKB_ENTER:
				if (gv_pViewMenuCurrItem->uiOption)
				{
					return( gv_pViewMenuCurrItem->uiOption);
				}
				break;
			case 'G':
			case 'g':
			case 7:  /* Control-G */
				return( GOTO_BLOCK_OPTION);
			case FKB_RIGHT:
			case 'R':
			case 'r':
			case '6':
				DoRightArrow();
				break;
			case FKB_LEFT:
			case 'L':
			case 'l':
			case '4':
				DoLeftArrow();
				break;
			case 'E':
			case 'e':
				return( EDIT_OPTION);
			case 'A':
			case 'a':
				return( EDIT_RAW_OPTION);
			case 'x':
			case 'X':
				return( HEX_OPTION);
			case 'S':
			case 's':
				return( SEARCH_OPTION);
			case '?':
				ViewHelpScreen();
				break;
			default:
				break;
		}
	}
}

/***************************************************************************
Desc:	This routine updates the date and time on the screen.
*****************************************************************************/
void ViewUpdateDate(
	FLMBOOL			bUpdateFlag,
	F_TMSTAMP  *	pLastTime
	)
{
	F_TMSTAMP	CurrTime;
	char			szTempBuf [64];
	FLMUINT		uiHour;
	char			szAmPm [4];
	FLMUINT		uiNumCols;
	FLMUINT		uiNumRows;

	f_conGetScreenSize( &uiNumCols, &uiNumRows);
	f_timeGetTimeStamp( &CurrTime);

	// Update the date, if it has changed or the bUpdateFlag is set

	if (bUpdateFlag ||
		 pLastTime->year != CurrTime.year ||
		 pLastTime->month != CurrTime.month ||
		 pLastTime->day != CurrTime.day)
	{
		f_sprintf( (char *)szTempBuf, "%s %u, %u",
						 Months [CurrTime.month],
						 (unsigned)CurrTime.day,
						 (unsigned)CurrTime.year);
		f_conSetBackFore( FLM_BLACK, FLM_WHITE);
		f_conStrOutXY( szTempBuf, 0, 0);
	}

	// Update the time, if it has changed or the bUpdateFlag is set

	if (bUpdateFlag ||
		 pLastTime->hour != CurrTime.hour ||
		 pLastTime->minute != CurrTime.minute ||
		 pLastTime->second != CurrTime.second)
	{
		if (CurrTime.hour == 0)
		{
			uiHour = 12;
		}
		else if (CurrTime.hour > 12)
		{
			uiHour = (FLMUINT)CurrTime.hour - 12;
		}
		else
		{
			uiHour = (FLMUINT)CurrTime.hour;
		}
		if (CurrTime.hour >= 12)
		{
			f_strcpy( szAmPm, "pm");
		}
		else
		{
			f_strcpy( szAmPm, "am");
		}
		f_sprintf( (char *)szTempBuf, "%2u:%02u:%02u %s",
						(unsigned)uiHour,
						(unsigned)CurrTime.minute,
						(unsigned)CurrTime.second, szAmPm);
		f_conSetBackFore( FLM_BLACK, FLM_WHITE);
		f_conStrOutXY( szTempBuf, 66, 0);
	}

	if (bUpdateFlag ||
		 gv_uiViewLastFileOffset != gv_uiViewCurrFileOffset ||
		 (gv_pViewMenuCurrItem->uiModFileOffset == VIEW_INVALID_FILE_OFFSET &&
		  gv_uiViewLastFileOffset != VIEW_INVALID_FILE_OFFSET))
	{
		if (!gv_pViewMenuCurrItem)
		{
			gv_uiViewLastFileOffset = VIEW_INVALID_FILE_OFFSET;
		}
		else if (gv_pViewMenuCurrItem->uiModFileOffset == VIEW_INVALID_FILE_OFFSET)
		{
			gv_uiViewLastFileNumber = gv_pViewMenuCurrItem->uiModFileNumber;
			gv_uiViewLastFileOffset = gv_pViewMenuCurrItem->uiModFileOffset;
		}
		else
		{
			gv_uiViewLastFileNumber = gv_uiViewCurrFileNumber;
			gv_uiViewLastFileOffset = gv_uiViewCurrFileOffset;
		}

		if (gv_uiViewLastFileOffset == VIEW_INVALID_FILE_OFFSET)
		{
			f_strcpy( szTempBuf, "File: N/A   File Pos: N/A       ");
		}
		else
		{
			f_sprintf( (char *)szTempBuf, "File: %03u  File Pos: 0x%08X",
								(unsigned)gv_uiViewLastFileNumber,
								(unsigned)gv_uiViewLastFileOffset);
		}
		f_conSetBackFore( FLM_BLACK, FLM_WHITE);
		f_conStrOutXY( szTempBuf, 47, uiNumRows - 1);
	}

	// Save the date and time

	f_memcpy( pLastTime, &CurrTime, sizeof( F_TMSTAMP));
}

/***************************************************************************
Desc: This routine resets the view parameters for a menu and saves
		the parameters for the current menu.  This is done whenever a new
		menu is being entered.  It allows the previous menu to be restored
		to its original state upon returning.
*****************************************************************************/
void ViewReset(
	VIEW_INFO_p	pSaveView
	)
{
	pSaveView->uiCurrItem = gv_uiViewMenuCurrItemNum;
	pSaveView->uiTopRow = gv_uiViewTopRow;
	pSaveView->uiBottomRow = gv_uiViewBottomRow;
	pSaveView->uiCurrFileOffset = gv_uiViewCurrFileOffset;
	pSaveView->uiCurrFileNumber = gv_uiViewCurrFileNumber;
	gv_pViewMenuCurrItem = NULL;
	gv_uiViewMenuCurrItemNum = 0;
	gv_uiViewTopRow = 0;
	gv_uiViewBottomRow = gv_uiViewTopRow + LINES_PER_PAGE - 1;
	gv_uiViewCurrFileOffset = 0;
	gv_uiViewCurrFileNumber = 0;
}

/********************************************************************
Desc: ?
*********************************************************************/
void ViewDisable( void)
{
	gv_bViewEnabled = FALSE;
}

/********************************************************************
Desc: ?
*********************************************************************/
void ViewEnable( void)
{
	VIEW_MENU_ITEM_p	pMenuItem;
	FLMUINT				uiDistance = 0xFFFFFFFF;
	VIEW_MENU_ITEM_p	pClosestMenuItem = NULL;
	FLMUINT				uiStartOffset;
	FLMUINT				uiEndOffset;

	if (!gv_bViewEnabled)
	{
		if (gv_uiViewCurrFileOffset != VIEW_INVALID_FILE_OFFSET)
		{
			pMenuItem = gv_pViewMenuFirstItem;
			while (pMenuItem)
			{
				if (pMenuItem->uiModFileOffset != VIEW_INVALID_FILE_OFFSET)
				{
					uiStartOffset = pMenuItem->uiModFileOffset;
					switch (pMenuItem->uiModType & 0x0F)
					{
						case MOD_FLMUINT32:
							uiEndOffset = uiStartOffset + 3;
							break;
						case MOD_FLMUINT16:
							uiEndOffset = uiStartOffset + 1;
							break;
						case MOD_BINARY:
						case MOD_TEXT:
							uiEndOffset = uiStartOffset + pMenuItem->uiModBufLen - 1;
							break;
						case MOD_CHILD_BLK:
							uiEndOffset = uiStartOffset + 2;
							break;
						case MOD_FLMUINT64:
							uiEndOffset = uiStartOffset + 7;
							break;
						default:
							uiEndOffset = uiStartOffset;
							break;
					}
					if (gv_uiViewCurrFileOffset >= uiStartOffset &&
						 gv_uiViewCurrFileOffset <= uiEndOffset)
					{
						if ((pMenuItem->uiModType & 0xF0) == MOD_DISABLED)
						{
							pClosestMenuItem = pMenuItem;
							uiDistance = 0;
						}
						else
						{
							pClosestMenuItem = pMenuItem;
							break;
						}
					}
					else if (gv_uiViewCurrFileOffset < uiStartOffset)
					{
						if (uiStartOffset - gv_uiViewCurrFileOffset < uiDistance)
						{
							pClosestMenuItem = pMenuItem;
							uiDistance = uiStartOffset - gv_uiViewCurrFileOffset;
						}
					}
					else
					{
						if (gv_uiViewCurrFileOffset - uiStartOffset < uiDistance)
						{
							pClosestMenuItem = pMenuItem;
							uiDistance = gv_uiViewCurrFileOffset - uiStartOffset;
						}
					}
				}
				pMenuItem = pMenuItem->pNextItem;
			}
		}
		if (pClosestMenuItem)
		{
			gv_pViewMenuCurrItem = pMenuItem = pClosestMenuItem;
			gv_uiViewMenuCurrItemNum = pMenuItem->uiItemNum;
			if (pMenuItem->uiRow < LINES_PER_PAGE)
			{
				gv_uiViewTopRow = 0;
			}
			else
			{
				gv_uiViewTopRow = pMenuItem->uiRow - LINES_PER_PAGE / 2 + 1;
			}
			gv_uiViewBottomRow = gv_uiViewTopRow + LINES_PER_PAGE - 1;
			if (gv_uiViewBottomRow > gv_pViewMenuLastItem->uiRow)
			{
				gv_uiViewBottomRow = gv_pViewMenuLastItem->uiRow;
			}
			if (gv_uiViewBottomRow - gv_uiViewTopRow + 1 < LINES_PER_PAGE)
			{
				if (gv_uiViewBottomRow < LINES_PER_PAGE + 1)
				{
					gv_uiViewTopRow = 0;
				}
				else
				{
					gv_uiViewTopRow = gv_uiViewBottomRow + 1 - LINES_PER_PAGE;
				}
			}
			if (HAVE_HORIZ_CUR( pMenuItem) &&
				 gv_uiViewCurrFileOffset - pMenuItem->uiModFileOffset <=
						(FLMUINT)pMenuItem->uiModBufLen)
			{
				pMenuItem->uiHorizCurPos =
					(FLMUINT)(gv_uiViewCurrFileOffset - pMenuItem->uiModFileOffset);
			}
		}
		gv_bViewEnabled = TRUE;
		ViewRefreshMenu( NULL);
	}
}

/***************************************************************************
Desc:	This routine restores the view parameters for a menu which were
		previously saved by the ViewReset routine.
*****************************************************************************/
void ViewRestore(
	VIEW_INFO_p	pSaveView
	)
{
	gv_uiViewMenuCurrItemNum = pSaveView->uiCurrItem;
	gv_pViewMenuCurrItem = NULL;
	gv_uiViewTopRow = pSaveView->uiTopRow;
	gv_uiViewBottomRow = pSaveView->uiBottomRow;
	gv_uiViewCurrFileOffset = pSaveView->uiCurrFileOffset;
	gv_uiViewCurrFileNumber = pSaveView->uiCurrFileNumber;
}
