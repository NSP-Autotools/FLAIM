//-------------------------------------------------------------------------
// Desc:	Display routines for the database viewer utility.
// Tabs:	3
//
// Copyright (c) 1992-1995, 1998-2000, 2003-2007 Novell, Inc.
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

/***************************************************************************
Name:    ViewShowError
Desc:    This routine displays an error message to the screen.
*****************************************************************************/
void ViewShowError(
	const char *	Message)
{
	FLMUINT		uiChar;
	FLMUINT		uiNumCols;
	FLMUINT		uiNumRows;

	f_conGetScreenSize( &uiNumCols, &uiNumRows);
	f_conSetBackFore( FLM_BLACK, FLM_WHITE);
	f_conClearScreen( 0, uiNumRows - 2);
	f_conSetBackFore( FLM_RED, FLM_WHITE);
	f_conStrOutXY( Message, 0, uiNumRows - 2);
	f_conStrOutXY( "Press ENTER to continue: ", 0, 23);
	for( ;;)
	{
		uiChar = f_conGetKey();
		if( (uiChar == FKB_ENTER) || (uiChar == FKB_ESCAPE))
		{
			break;
		}
	}
	f_conSetBackFore( FLM_BLACK, FLM_WHITE);
	f_conClearScreen( 0, uiNumRows - 2);
	ViewEscPrompt();
}

/***************************************************************************
Name:    ViewShowRCError
Desc:    This routine displays a FLAIM error message to the screen.  It
				formats the RCODE into a message and calls ViewShowError
				to display the error.
*****************************************************************************/
void ViewShowRCError(
	const char *	szWhat,
	RCODE       	rc)
{
	char	TBuf[ 100];

	f_strcpy( TBuf, "Error ");
	f_strcpy( &TBuf [f_strlen( TBuf)], szWhat);
	f_strcpy( &TBuf [f_strlen( TBuf)], ": ");
	f_strcpy( &TBuf [f_strlen( TBuf)], FlmErrorString( rc));
	ViewShowError( TBuf);
}
