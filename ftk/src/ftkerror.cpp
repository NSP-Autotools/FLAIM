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

#include "ftksys.h"

/****************************************************************************
Desc:	The primary purpose of this function is to provide a way to easily
		trap errors when they occur.  Just put a breakpoint in this function
		to catch them.
****************************************************************************/
#ifdef FLM_DEBUG
RCODE FTKAPI f_makeErr(
	RCODE				rc,
	const char *,	// pszFile,
	int,				// iLine,
	FLMBOOL			bAssert)
{
	if( rc == NE_FLM_OK)
	{
		return( NE_FLM_OK);
	}
	
	f_assert( rc != NE_FLM_MEM);

#if defined( FLM_DEBUG)
	if( bAssert)
	{
		f_assert( 0);
	}
#else
	F_UNREFERENCED_PARM( bAssert);
#endif

	return( rc);
}
#endif

/***************************************************************************
Desc:   Map POSIX errno to Flaim IO errors.
***************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
RCODE FTKAPI f_mapPlatformError(
	FLMINT	iError,
	RCODE		defaultRc)
{
	switch (iError)
	{
		case 0:
		{
			return( NE_FLM_OK);
		}

		case ENOENT:
		{
			return( RC_SET( NE_FLM_IO_PATH_NOT_FOUND));
		}

		case EACCES:
		case EEXIST:
		{
			return( RC_SET( NE_FLM_IO_ACCESS_DENIED));
		}

		case EINVAL:
		{
			return( RC_SET_AND_ASSERT( NE_FLM_INVALID_PARM));
		}

		case EIO:
		{
			return( RC_SET( NE_FLM_IO_DISK_FULL));
		}

		case ENOTDIR:
		{
			return( RC_SET( NE_FLM_IO_DIRECTORY_ERR));
		}

#ifdef EBADFD
		case EBADFD:
		{
			return( RC_SET( NE_FLM_IO_BAD_FILE_HANDLE));
		}
#endif

#ifdef EOF
		case EOF:
		{
			return( RC_SET( NE_FLM_IO_END_OF_FILE));
		}
#endif
			
		case EMFILE:
		{
			return( RC_SET( NE_FLM_IO_NO_MORE_FILES));
		}

		default:
		{
			return( RC_SET( defaultRc));
		}
	}
}
#endif

/***************************************************************************
Desc:
***************************************************************************/
#ifdef FLM_RING_ZERO_NLM
RCODE FTKAPI f_mapPlatformError(
	FLMINT	iErrCode,
	RCODE		defaultRc)
{
	RCODE		rc;
	
	switch (iErrCode)
	{
		case 128: // ERR_LOCK_FAIL
		case 147: // ERR_NO_READ_PRIVILEGE
		case 148: // ERR_NO_WRITE_PRIVILEGE
		case 168: // ERR_ACCESS_DENIED
			rc = RC_SET( NE_FLM_IO_ACCESS_DENIED);
			break;

		case 136: //ERR_INVALID_FILE_HANDLE
			rc = RC_SET( NE_FLM_IO_BAD_FILE_HANDLE);
			break;

		case 001: //ERR_INSUFFICIENT_SPACE
		case 153: //ERR_DIRECTORY_FULL
			rc = RC_SET( NE_FLM_IO_DISK_FULL);
			break;

		case 130: //ERR_NO_OPEN_PRIVILEGE
		case 165: //ERR_INVALID_OPENCREATE_MODE
			rc = RC_SET( NE_FLM_IO_OPEN_ERR);
			break;

		case 156: //ERR_INVALID_PATH
		case 158: //ERR_BAD_FILE_NAME
			rc = RC_SET( NE_FLM_IO_PATH_NOT_FOUND);
			break;

		case 129: //ERR_OUT_OF_HANDLES
			rc = RC_SET( NE_FLM_IO_TOO_MANY_OPEN_FILES);
			break;

		case 139: //ERR_NO_RENAME_PRIVILEGE
		case 154: //ERR_RENAME_ACROSS_VOLUME
		case 164: //ERR_RENAME_DIR_INVALID
			rc = RC_SET( NE_FLM_IO_RENAME_FAILURE);
			break;

		case 222: //ERR_BAD_PASSWORD
		case 223: //ERR_PASSWORD_EXPIRED
			rc = RC_SET( NE_FLM_IO_INVALID_PASSWORD);
			break;

		case 122: //ERR_CONNECTION_ALREADY_TEMPORARY
		case 123: //ERR_CONNECTION_ALREADY_LOGGED_IN
		case 124: //ERR_CONNECTION_NOT_AUTHENTICATED
		case 125: //ERR_CONNECTION_NOT_LOGGED_IN
		case 224: //ERR_NO_LOGIN_CONNECTIONS_AVAILABLE
			rc = RC_SET( NE_FLM_IO_CONNECT_ERROR);
			break;

		default:
			rc = RC_SET( defaultRc);
			break;
	}
	
	return( rc);
}
#endif

/***************************************************************************
Desc:
***************************************************************************/
#ifdef FLM_WIN
RCODE FTKAPI f_mapPlatformError(
	FLMINT	iErrCode,
	RCODE		defaultRc)
{
	switch( iErrCode)
	{
		case ERROR_NOT_ENOUGH_MEMORY:
		case ERROR_OUTOFMEMORY:
			return( RC_SET( NE_FLM_MEM));
			
		case ERROR_BAD_NETPATH:
		case ERROR_BAD_PATHNAME:
		case ERROR_DIRECTORY:
		case ERROR_FILE_NOT_FOUND:
		case ERROR_INVALID_DRIVE:
		case ERROR_INVALID_NAME:
		case ERROR_NO_NET_OR_BAD_PATH:
		case ERROR_PATH_NOT_FOUND:
			return( RC_SET( NE_FLM_IO_PATH_NOT_FOUND));

		case ERROR_ACCESS_DENIED:
		case ERROR_SHARING_VIOLATION:
		case ERROR_FILE_EXISTS:
		case ERROR_ALREADY_EXISTS:
			return( RC_SET( NE_FLM_IO_ACCESS_DENIED));

		case ERROR_BUFFER_OVERFLOW:
		case ERROR_FILENAME_EXCED_RANGE:
			return( RC_SET( NE_FLM_IO_PATH_TOO_LONG));

		case ERROR_DISK_FULL:
		case ERROR_HANDLE_DISK_FULL:
			return( RC_SET( NE_FLM_IO_DISK_FULL));

		case ERROR_CURRENT_DIRECTORY:
		case ERROR_DIR_NOT_EMPTY:
			return( RC_SET( NE_FLM_IO_DIRECTORY_ERR));

		case ERROR_DIRECT_ACCESS_HANDLE:
		case ERROR_INVALID_HANDLE:
		case ERROR_INVALID_TARGET_HANDLE:
			return( RC_SET( NE_FLM_IO_BAD_FILE_HANDLE));

		case ERROR_HANDLE_EOF:
			return( RC_SET( NE_FLM_IO_END_OF_FILE));

		case ERROR_OPEN_FAILED:
			return( RC_SET( NE_FLM_IO_OPEN_ERR));

		case ERROR_CANNOT_MAKE:
			return( RC_SET( NE_FLM_IO_PATH_CREATE_FAILURE));

		case ERROR_LOCK_FAILED:
		case ERROR_LOCK_VIOLATION:
			return( RC_SET( NE_FLM_IO_FILE_LOCK_ERR));

		case ERROR_NEGATIVE_SEEK:
		case ERROR_SEEK:
		case ERROR_SEEK_ON_DEVICE:
			return( RC_SET( NE_FLM_IO_SEEK_ERR));

		case ERROR_NO_MORE_FILES:
		case ERROR_NO_MORE_SEARCH_HANDLES:
			return( RC_SET( NE_FLM_IO_NO_MORE_FILES));

		case ERROR_TOO_MANY_OPEN_FILES:
			return( RC_SET( NE_FLM_IO_TOO_MANY_OPEN_FILES));

		case NO_ERROR:
			return( NE_FLM_OK);

		case ERROR_DISK_CORRUPT:
		case ERROR_DISK_OPERATION_FAILED:
		case ERROR_FILE_CORRUPT:
		case ERROR_FILE_INVALID:
		case ERROR_NOT_SAME_DEVICE:
		case ERROR_IO_DEVICE:
		default:
			return( RC_SET( defaultRc));

   }
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
FLMINT FTKAPI f_enterDebugger(
	const char *	pszFile,
	int				iLine)
{
#ifdef FLM_WIN
	fprintf( stderr, "Assertion failed in %s on line %d\n", pszFile, iLine);
	fflush( stderr);
	DebugBreak();
#elif defined( FLM_NLM)
	(void)pszFile;
	(void)iLine;
	EnterDebugger();
#else
	fprintf( stderr, "Assertion failed in %s on line %d\n", pszFile, iLine);
	fflush( stderr);
	assert( 0);
#endif

	return( 0);
}

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_WATCOM_NLM)
int gv_ftkerrorDummy(void)
{
	return( 0);
}
#endif
