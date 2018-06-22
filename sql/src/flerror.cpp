//------------------------------------------------------------------------------
// Desc:	This file contains error routines that are used throughout FLAIM.
// Tabs:	3
//
// Copyright (c) 1997-2000, 2002-2007 Novell, Inc. All Rights Reserved.
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

#include "flaimsys.h"

/****************************************************************************
Desc:
****************************************************************************/
const char * FlmCorruptStrings[ SFLM_NUM_CORRUPT_ERRORS] =
{
	
	// WARNING:	ANY CHANGES MADE TO THE FlmCorruptStrings TABLE MUST BE
	// REFLECTED IN THE CHECK CODE DEFINES FOUND IN flaimsys.h

	"OK",											// 0
	"BAD_CHAR",									// 1
	"BAD_ASIAN_CHAR",							// 2
	"BAD_CHAR_SET",							// 3
	"BAD_TEXT_FIELD",							// 4
	"BAD_NUMBER_FIELD",						// 5
	"BAD_FIELD_TYPE",							// 6
	"BAD_IX_DEF",								// 7
	"BAD_NUMBER_KEY",							// 8
	"BAD_BINARY_KEY",							// 9
	"BAD_KEY_FIELD_TYPE",					// 10
	"BAD_KEY_LEN",								// 11
	"BAD_LFH_LIST_PTR",						// 12
	"BAD_LFH_LIST_END",						// 13
	"BAD_BLK_END",								// 14
	"KEY_COUNT_MISMATCH",					// 15
	"REF_COUNT_MISMATCH",					// 16
	"BAD_BLK_HDR_ADDR",						// 17
	"BAD_BLK_HDR_LEVEL",						// 18
	"BAD_BLK_HDR_PREV",						// 19

	// WARNING:	ANY CHANGES MADE TO THE FlmCorruptStrings TABLE MUST BE
	// REFLECTED IN THE CHECK CODE DEFINES FOUND IN flaimsys.h

	"BAD_BLK_HDR_NEXT",						// 20
	"BAD_BLK_HDR_TYPE",						// 21
	"BAD_BLK_HDR_ROOT_BIT",					// 22
	"BAD_BLK_HDR_BLK_END",					// 23
	"BAD_BLK_HDR_LF_NUM",					// 24
	"BAD_AVAIL_LIST_END",					// 25
	"BAD_PREV_BLK_NEXT",						// 26
	"BAD_FIRST_ELM_FLAG",					// 27
	"BAD_LEM",									// 28
	"BAD_ELM_LEN",								// 29
	"BAD_ELM_KEY_SIZE",						// 30
	"BAD_ELM_KEY",								// 31
	"BAD_ELM_KEY_ORDER",						// 32
	"BAD_CONT_ELM_KEY",						// 34
	"NON_UNIQUE_FIRST_ELM_KEY",			// 35
	"BAD_ELM_OFFSET",							// 36
	"BAD_ELM_INVALID_LEVEL",				// 37
	"BAD_ELM_FLD_NUM",						// 38
	"BAD_ELM_FLD_LEN",						// 39

	// WARNING:	ANY CHANGES MADE TO THE FlmCorruptStrings TABLE MUST BE
	// REFLECTED IN THE CHECK CODE DEFINES FOUND IN flaimsys.h

	"BAD_ELM_FLD_TYPE",						// 40
	"BAD_ELM_END",								// 41
	"BAD_PARENT_KEY",							// 42
	"BAD_ELM_IX_REF",							// 43
	"BAD_LAST_BLK_NEXT",						// 44
	"REBUILD_REC_EXISTS",					// 45
	"REBUILD_KEY_NOT_UNIQUE",				// 46
	"NON_UNIQUE_ELM_KEY_REF",				// 47
	"OLD_VIEW",									// 48
	"COULD_NOT_SYNC_BLK",					// 49
	"IX_KEY_NOT_FOUND_IN_REC",				// 50
	"KEY_NOT_IN_KEY_REFSET",				// 51
	"BAD_BLK_CHECKSUM",						// 52
	"BAD_FILE_SIZE",							// 53
	"BAD_BLK_TYPE",							// 54
	"BAD_ELEMENT_CHAIN",						// 55
	"BAD_ELM_EXTRA_DATA",					// 56
	"BAD_BLOCK_STRUCTURE",					// 57
	"BAD_DATA_BLOCK_COUNT",					// 58
	"BAD_AVAIL_SIZE",							// 59
	"BAD_CHILD_ELM_COUNT",					// 60
};

/****************************************************************************
Desc:	The primary purpose of this function is to provide a way to easily
		trap errors when they occur.  Just put a breakpoint in this function
		to catch them.
Note:	Some of the most common errors will be coded so the use can set a
		break point.
****************************************************************************/
#ifdef FLM_DEBUG
RCODE flmMakeErr(
	RCODE				rc,
	const char *	pszFile,
	int				iLine,
	FLMBOOL			bAssert)
{
	F_UNREFERENCED_PARM( pszFile);
	F_UNREFERENCED_PARM( iLine);
	
	if( bAssert)
	{
		flmAssert( 0);
	}

	return( rc);
}
#endif

#if defined( FLM_WATCOM_NLM)
	int gv_iFlerrorDummy(void)
	{
		return( 0);
	}
#endif

/****************************************************************************
Desc:	Returns a pointer to the string representation of a corruption
		error code.
****************************************************************************/
const char * F_DbSystem::checkErrorToStr(
	eCorruptionType	eCorruption)
{
	if (eCorruption < SFLM_NUM_CORRUPT_ERRORS)
	{
		return( FlmCorruptStrings [(FLMINT)eCorruption]);
	}
	else
	{
		return( "Unknown Error");
	}
}

