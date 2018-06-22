//------------------------------------------------------------------------------
// Desc:	This file contains misc toolkit functions
// Tabs:	3
//
// Copyright (c) 2000-2007 Novell, Inc. All Rights Reserved.
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

static FLMATOMIC					gv_startupCount = 0;
static FLMUINT						gv_uiRandomGenInitCount = 0;
static F_MUTEX						gv_hRandomGenMutex = F_MUTEX_NULL;
static IF_RandomGenerator *	gv_pRandomGenerator = NULL;
static IF_ThreadMgr *			gv_pThreadMgr = NULL;
static IF_FileSystem *			gv_pFileSystem = NULL;
static FLMUINT						gv_uiMaxFileSize = FLM_MAXIMUM_FILE_SIZE;
static F_XML *						gv_pXml = NULL;

FLMATOMIC							gv_openFiles = 0;
F_MUTEX 								F_FileHdl::m_hAsyncListMutex = F_MUTEX_NULL;
F_FileAsyncClient *				F_FileHdl::m_pFirstAvailAsync = NULL;
FLMUINT								F_FileHdl::m_uiAvailAsyncCount = 0;

#ifdef FLM_WIN
	SET_FILE_VALID_DATA_FUNC 	gv_SetFileValidDataFunc = NULL;
#endif

#ifdef FLM_AIX
	#ifndef nsleep
		extern "C"
		{
			extern int nsleep( struct timestruc_t *, struct timestruc_t *);
		}
	#endif
#endif

FSTATIC RCODE f_initRandomGenerator( void);

FSTATIC void f_freeRandomGenerator( void);

#define F_VECTOR_START_AMOUNT		16
#define F_VECTOR_GROW_AMOUNT		2

/****************************************************************************
Desc:
****************************************************************************/
static FLMBYTE gv_ucSENLengthArray[] =
{
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,		// 0   - 15
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,		// 16  - 31
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,		// 32  - 47
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,		// 48  - 63
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,		// 64  - 79
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,		// 80  - 95
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,		// 96  - 111
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,		// 112 - 127
	2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,		// 128 - 143
	2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,		// 144 - 159
	2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,		// 160 - 175
	2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,		// 176 - 191
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,		// 192 - 207
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,		// 208 - 223
	4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,		// 224 - 239
	5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 8, 9		// 240 - 255
};

/****************************************************************************
Desc:
****************************************************************************/
static FLMBYTE ucSENPrefixArray[] =
{
	0,
	0,
	0x80,
	0xC0,
	0xE0,
	0xF0,
	0xF8,
	0xFC,
	0xFE,
	0xFF
};

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI ftkStartup( void)
{
	RCODE		rc = NE_FLM_OK;
	
	if( f_atomicInc( &gv_startupCount) > 1)
	{
		goto Exit;
	}
	
	// Sanity check -- make sure we are using the correct
	// byte-swap macros for this platform

	f_assert( FB2UD( (FLMBYTE *)"\x0A\x0B\x0C\x0D") == 0x0D0C0B0A);
	f_assert( FB2UW( (FLMBYTE *)"\x0A\x0B") == 0x0B0A);
	
	// Verify that the platform word size is correct
	
#ifdef FLM_64BIT
	f_assert( sizeof( FLMUINT) == 8);
#else
	f_assert( sizeof( FLMUINT) == 4);
#endif

#if defined( FLM_RING_ZERO_NLM)
	if( RC_BAD( rc = f_netwareStartup()))
	{
		goto Exit;
	}
#endif

	f_memoryInit();

#if !defined( FLM_RING_ZERO_NLM)
	f_assert( sizeof( f_va_list) == sizeof( va_list));
#endif
	
	if( RC_BAD( rc = f_initCharMappingTables()))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = f_verifyDiskStructOffsets()))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = f_allocFileSystem( &gv_pFileSystem)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = f_initFileAsyncClientList()))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = f_allocThreadMgr( &gv_pThreadMgr)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = f_initRandomGenerator()))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = f_initCRCTable()))
	{
		goto Exit;
	}
	
	f_initFastCheckSum();
	
	if( (gv_pXml = f_new F_XML) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
		
	if( RC_BAD( rc = gv_pXml->setup()))
	{
		goto Exit;
	}

#ifdef FLM_DEBUG
	if( RC_BAD( rc = f_verifyMetaphoneRoutines()))
	{
		goto Exit;
	}
#endif
	
#if defined( FLM_LINUX)
	f_setupLinuxKernelVersion();
	gv_uiMaxFileSize = f_getLinuxMaxFileSize();
#elif defined( FLM_AIX)

	// Call setrlimit to increase the max allowed file size.
	// We don't have a good way to deal with any errors returned by
	// setrlimit(), so we just hope that there aren't any ...
	
	struct rlimit rlim;
	
	rlim.rlim_cur = RLIM_INFINITY;
	rlim.rlim_max = RLIM_INFINITY;
	
	setrlimit( RLIMIT_FSIZE, &rlim);
#endif

#if defined( FLM_WIN)
	{
		HINSTANCE		hLibrary; 
	
		// Get a handle to the DLL.  If the handle is valid, try to get the
		// function address. 
	 
		if( (hLibrary = LoadLibrary( TEXT( "kernel32.dll"))) != NULL) 
		{
			gv_SetFileValidDataFunc = (SET_FILE_VALID_DATA_FUNC)GetProcAddress( 
				hLibrary, TEXT( "SetFileValidData")); 
			FreeLibrary( hLibrary);
		}
	}
#endif

#if defined( FLM_OPENSSL)
	// Initialize OpenSSL

	SSL_load_error_strings();
	SSL_library_init();	
	ERR_load_BIO_strings();
#endif

	// Setup logger

	if (RC_BAD( rc = f_loggerInit()))
	{
		goto Exit;
	}

Exit:

	if( RC_BAD( rc))
	{
		ftkShutdown();
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI ftkShutdown( void)
{
	if( !gv_startupCount || f_atomicDec( &gv_startupCount) > 0)
	{
		return;
	}
	
	f_assert( !gv_openFiles);
	
	if( gv_pThreadMgr)
	{
		gv_pThreadMgr->Release();
		gv_pThreadMgr = NULL;
	}
	
	f_freeFileAsyncClientList();
	
	if( gv_pFileSystem)
	{
		gv_pFileSystem->Release();
		gv_pFileSystem = NULL;
	}
	
	f_freeCRCTable();
	
	if( gv_pXml)
	{
		gv_pXml->Release();
	}

	f_loggerShutdown();
	f_freeRandomGenerator();
	f_freeCharMappingTables();
	f_memoryCleanup();
	
#if defined( FLM_RING_ZERO_NLM)
	f_netwareShutdown();
#endif
}

/****************************************************************************
Desc: This routine causes the calling process to delay the given number
		of milliseconds.  Due to the nature of the call, the actual sleep
		time is almost guaranteed to be different from requested sleep time.
****************************************************************************/
#ifdef FLM_UNIX
void FTKAPI f_sleep(
	FLMUINT		uiMilliseconds)
{
#ifdef FLM_AIX
	struct timestruc_t timeout;
	struct timestruc_t remain;
#else
	struct timespec timeout;
#endif

	timeout.tv_sec = (uiMilliseconds / 1000);
	timeout.tv_nsec = (uiMilliseconds % 1000) * 1000000;

#ifdef FLM_AIX
	nsleep(&timeout, &remain);
#else
	nanosleep(&timeout, 0);
#endif
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_WIN
void FTKAPI f_sleep(
	FLMUINT		uiMilliseconds)
{
	SleepEx( (DWORD)uiMilliseconds, true);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_LIBC_NLM
void FTKAPI f_sleep( 
	FLMUINT		uiMilliseconds)
{
	if( !uiMilliseconds )
	{
		pthread_yield();
	}
	else
	{
		delay( uiMilliseconds);
	}
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_RING_ZERO_NLM
void FTKAPI f_sleep( 
	FLMUINT	uiMilliseconds)
{
	if( !uiMilliseconds)
	{
		kYieldThread();
	}
	else
	{
		kDelayThread( uiMilliseconds);
	}
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE f_initRandomGenerator( void)
{
	FLMUINT					uiTime;
	RCODE						rc = NE_FLM_OK;

	if (++gv_uiRandomGenInitCount > 1)
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_mutexCreate( &gv_hRandomGenMutex)))
	{
		goto Exit;
	}

	f_timeGetSeconds( &uiTime );

	if( RC_BAD( rc = FlmAllocRandomGenerator( &gv_pRandomGenerator)))
	{
		goto Exit;
	}

	gv_pRandomGenerator->setSeed( (FLMUINT32)(uiTime ^ (FLMUINT)f_getpid()));

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC void f_freeRandomGenerator( void)
{
	if( (--gv_uiRandomGenInitCount) > 0)
	{
		return;
	}
	
	if( gv_pRandomGenerator)
	{
		gv_pRandomGenerator->Release();
		gv_pRandomGenerator = NULL;
	}

	if( gv_hRandomGenMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &gv_hRandomGenMutex);
	}
}

/****************************************************************************
Desc:		This routine will use the operating system calls to generate a
			"globally unique" identifier.  Typically, this is based on the
			MAC address of an ethernet card installed in the machine.  If the
			machine does not have an ethernet card, or if the OS does not
			support generating GUIDs, this routine will generate a pseudo-GUID
			using a random number generator.  A serial number is 16-bytes.
****************************************************************************/
RCODE FTKAPI f_createSerialNumber(
	FLMBYTE *		pszSerialNum)
{
	RCODE						rc = NE_FLM_OK;

#if defined( FLM_WIN)

	UUID			uuidVal;
	RPC_STATUS	err = UuidCreate( &uuidVal);

	if (err == RPC_S_OK || err == RPC_S_UUID_LOCAL_ONLY)
	{
		UD2FBA( (FLMUINT32)uuidVal.Data1, &pszSerialNum[ 0]);
		UW2FBA( (FLMUINT16)uuidVal.Data2, &pszSerialNum[ 4]);
		UW2FBA( (FLMUINT16)uuidVal.Data3, &pszSerialNum[ 6]);
		f_memcpy( &pszSerialNum[ 8], (FLMBYTE *)uuidVal.Data4, 8);
		goto Exit;
	}

#elif defined( FLM_UNIX) || defined( FLM_NLM)

	// Generate a pseudo GUID value

	UD2FBA( f_getRandomUINT32(), &pszSerialNum[ 0]);
	UD2FBA( f_getRandomUINT32(), &pszSerialNum[ 4]);
	UD2FBA( f_getRandomUINT32(), &pszSerialNum[ 8]);
	UD2FBA( f_getRandomUINT32(), &pszSerialNum[ 12]);

#endif

#if defined( FLM_WIN)
Exit:
#endif

	return( rc);
}

/****************************************************************************
Desc: 	
****************************************************************************/
void FTKAPI f_getenv(
	const char *	pszKey,
	FLMBYTE *		pszBuffer,
	FLMUINT			uiBufferSize,
	FLMUINT *		puiValueLen)
{
	FLMUINT			uiValueLen = 0;

	if( !uiBufferSize)
	{
		goto Exit;
	}
	
	pszBuffer[ 0] = 0;
	
#if defined( FLM_WIN) || defined( FLM_UNIX)
	char *	pszValue;
	
   if( (pszValue = getenv( pszKey)) != NULL &&
		 (uiValueLen = f_strlen( pszValue)) < uiBufferSize)
	{
		f_strcpy( (char *)pszBuffer, pszValue);
	}
#else
	F_UNREFERENCED_PARM( pszKey);
#endif

Exit:

	if( puiValueLen)
	{
		*puiValueLen = uiValueLen;
	}

	return;
}

/***************************************************************************
Desc:		Sort an array of items
****************************************************************************/
void FTKAPI f_qsort(
	void *					pvBuffer,
	FLMUINT					uiLowerBounds,
	FLMUINT					uiUpperBounds,
	F_SORT_COMPARE_FUNC	fnCompare,
	F_SORT_SWAP_FUNC		fnSwap)
{
	FLMUINT			uiLBPos;
	FLMUINT			uiUBPos;
	FLMUINT			uiMIDPos;
	FLMUINT			uiCurrentPos;
	FLMUINT			uiLeftItems;
	FLMUINT			uiRightItems;
	FLMINT			iCompare;

Iterate_Larger_Half:

	uiUBPos = uiUpperBounds;
	uiLBPos = uiLowerBounds;
	uiMIDPos = (uiUpperBounds + uiLowerBounds + 1) / 2;
	uiCurrentPos = uiMIDPos;

	for (;;)
	{
		while (uiLBPos == uiMIDPos ||
					((iCompare = 
						fnCompare( pvBuffer, uiLBPos, uiCurrentPos)) < 0))
		{
			if( uiLBPos >= uiUpperBounds)
			{
				break;
			}
			uiLBPos++;
		}

		while( uiUBPos == uiMIDPos ||
					(((iCompare = 
						fnCompare( pvBuffer, uiCurrentPos, uiUBPos)) < 0)))
		{
			if (!uiUBPos)
			{
				break;
			}
			uiUBPos--;
		}
		
		if( uiLBPos < uiUBPos)
		{
			// Exchange [uiLBPos] with [uiUBPos].

			fnSwap( pvBuffer, uiLBPos, uiUBPos);
			uiLBPos++;
			uiUBPos--;
		}
		else
		{
			break;
		}
	}

	// Check for swap( LB, MID ) - cases 3 and 4

	if( uiLBPos < uiMIDPos )
	{

		// Exchange [uiLBPos] with [uiMIDPos]

		fnSwap( pvBuffer, uiMIDPos, uiLBPos);
		uiMIDPos = uiLBPos;
	}
	else if( uiMIDPos < uiUBPos )
	{
		// Exchange [uUBPos] with [uiMIDPos]

		fnSwap( pvBuffer, uiMIDPos, uiUBPos);
		uiMIDPos = uiUBPos;
	}

	// Check the left piece.

	uiLeftItems = (uiLowerBounds + 1 < uiMIDPos)
							? uiMIDPos - uiLowerBounds
							: 0;

	uiRightItems = (uiMIDPos + 1 < uiUpperBounds)
							? uiUpperBounds - uiMIDPos
							: 0;

	if( uiLeftItems < uiRightItems)
	{
		// Recurse on the LEFT side and goto the top on the RIGHT side.

		if( uiLeftItems)
		{
			f_qsort( pvBuffer, uiLowerBounds, uiMIDPos - 1, fnCompare, fnSwap);
		}

		uiLowerBounds = uiMIDPos + 1;
		goto Iterate_Larger_Half;
	}
	else if( uiLeftItems)
	{
		// Recurse on the RIGHT side and goto the top for the LEFT side.

		if (uiRightItems )
		{
			f_qsort( pvBuffer, uiMIDPos + 1, uiUpperBounds, fnCompare, fnSwap);
		}

		uiUpperBounds = uiMIDPos - 1;
		goto Iterate_Larger_Half;
	}
}

/***************************************************************************
Desc:
****************************************************************************/
FLMINT FTKAPI f_qsortUINTCompare(
	void *		pvBuffer,
	FLMUINT		uiPos1,
	FLMUINT		uiPos2)
{
	FLMUINT		uiLeft = *(((FLMUINT *)pvBuffer) + uiPos1);
	FLMUINT		uiRight = *(((FLMUINT *)pvBuffer) + uiPos2);

	if( uiLeft < uiRight)
	{
		return( -1);
	}
	else if( uiLeft > uiRight)
	{
		return( 1);
	}

	return( 0);
}

/***************************************************************************
Desc:
****************************************************************************/
void FTKAPI f_qsortUINTSwap(
	void *		pvBuffer,
	FLMUINT		uiPos1,
	FLMUINT		uiPos2)
{
	FLMUINT *	puiArray = (FLMUINT *)pvBuffer;
	FLMUINT		uiTmp = puiArray[ uiPos1];

	puiArray[ uiPos1] = puiArray[ uiPos2];
	puiArray[ uiPos2] = uiTmp;
}

/****************************************************************************
Desc:
****************************************************************************/
void * FTKAPI f_memcpy(
	void *			pvDest,
	const void *	pvSrc,
	FLMSIZET			iSize)
{
	if( iSize == 1)
	{
		*((FLMBYTE *)pvDest) = *((FLMBYTE *)pvSrc);
		return( pvDest);
	}
	
#ifdef FLM_RING_ZERO_NLM
		CMoveFast( pvSrc, pvDest, iSize);
		return( pvDest);
#else
		return( memcpy( pvDest, pvSrc, iSize));
#endif
}

/****************************************************************************
Desc:
****************************************************************************/
void * FTKAPI f_memmove(
	void *			pvDest,
	const void *	pvSrc,
	FLMSIZET			uiLength)
{
#ifndef FLM_RING_ZERO_NLM

	return( memmove( pvDest, pvSrc, uiLength));
	
#else

	#define CMOVB_THRESHOLD		16
	char			*s = (char *)pvSrc;
	char			*d = (char *)pvDest;
	unsigned		uDiff;

	if( (char *)(s + uiLength) < d || (char *)(d + uiLength) < s)
	{
		// The source and destination do not overlap.

		CMoveFast( (void *)s, d, (LONG)uiLength);
	}
	else if( s < d)
	{
		// Source preceeds the destination, with overlap.

		uDiff = (unsigned)(d - s);
		d += uiLength;
		s += uiLength;
		if( uDiff >= CMOVB_THRESHOLD)
		{
			for( ;;)
			{
				if( uiLength < uDiff)
				{
					break;
				}

				// Copy the tail

				s -= uDiff;
				d -= uDiff;
				uiLength -= uDiff;
				CMoveFast( (void *)s, d, (LONG)uDiff);
			}
		}

		// Copy remaining bytes.

		while( uiLength--)
		{
			*--d = *--s;
		}
	}
	else if( s > d)
	{
		// Source follows the destination, with overlap.

		uDiff = (unsigned)(s - d);
		if( uDiff >= CMOVB_THRESHOLD)
		{
			for( ;;)
			{
				if( uiLength < uDiff)
				{
					break;
				}

				// Copy the head

				CMoveFast( (void *)s, d, (LONG)uDiff);
				uiLength -= uDiff;
				d += uDiff;
				s += uDiff;
			}
		}

		// Copy the remaining bytes

		while( uiLength--)
		{
			*d++ = *s++;
		}
	}

	// Else, the regions overlap completely (s == d).  Do nothing.

	return( pvDest);

#endif
}

/****************************************************************************
Desc:
****************************************************************************/
void * FTKAPI f_memset(
	void *				pvMem,
	unsigned char		ucByte,
	FLMSIZET				uiLength)
{
#ifndef FLM_RING_ZERO_NLM
	return( memset( pvMem, ucByte, uiLength));
#else
	char *			cp = (char *)pvMem;
	unsigned			dwordLength;
	unsigned long	dwordVal;

	dwordVal = ((unsigned long)ucByte << 24) |
		((unsigned long)ucByte << 16) |
		((unsigned long)ucByte << 8) |
		(unsigned long)ucByte;

	while( uiLength && ((long)cp & 3L))
	{
		*cp++ = (char)ucByte;
		uiLength--;
	}

	dwordLength = uiLength >> 2;
	if(  dwordLength != 0)
	{
		CSetD( dwordVal, (void *)cp, dwordLength);
		cp += (dwordLength << 2);
		uiLength -= (dwordLength << 2);
	}

	while( uiLength)
	{
		*cp++ = (char)ucByte;
		uiLength--;
	}

	return( pvMem);
#endif
}

/****************************************************************************
Desc:
****************************************************************************/
FLMINT FTKAPI f_memcmp(
	const void *		pvMem1,
	const void *		pvMem2,
	FLMSIZET				uiLength)
{
#ifndef FLM_NLM
	return( memcmp( pvMem1, pvMem2, uiLength));
#else
	unsigned char *	s1;
	unsigned char *	s2;

	for (s1 = (unsigned char *)pvMem1, s2 = (unsigned char *)pvMem2; 
		uiLength > 0; uiLength--, s1++, s2++)
	{
		if (*s1 == *s2)
		{
			continue;
		}
		else if( *s1 > *s2)
		{
			return( 1);
		}
		else
		{
			return( -1);
		}
	}

	return( 0);
#endif
}

/****************************************************************************
Desc:
****************************************************************************/
char * FTKAPI f_strcpy(
	char *			pszDest,
	const char *	pszSrc)
{
#ifndef FLM_NLM
	return( strcpy( pszDest, pszSrc));
#else
	while ((*pszDest++ = *pszSrc++) != 0);
	return( pszDest);
#endif
}

/****************************************************************************
Desc:
****************************************************************************/
char * FTKAPI f_strncpy(
	char *			pszDest,
	const char *	pszSrc,
	FLMSIZET			uiLength)
{
#ifndef FLM_NLM
	return( strncpy( pszDest, pszSrc, uiLength));
#else
	while( uiLength)
	{
		*pszDest++ = *pszSrc;
		if( *pszSrc)
		{
			pszSrc++;
		}
		
		uiLength--;
	}

	*pszDest = 0;
	return( pszDest);
#endif
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT FTKAPI f_strlen(
	const char *	pszStr)
{
#ifndef FLM_NLM
	return( strlen( pszStr));
#else
	const char *	pszStart = pszStr;

	while( *pszStr)
	{
		pszStr++;
	}

	return( pszStr - pszStart);
#endif
}

/****************************************************************************
Desc:
****************************************************************************/
FLMINT FTKAPI f_strcmp(
	const char *		pszStr1,
	const char *		pszStr2)
{
#ifndef FLM_NLM
	return( strcmp( pszStr1, pszStr2));
#else
	while( *pszStr1 == *pszStr2 && *pszStr1)
	{
		pszStr1++;
		pszStr2++;
	}
	
	return( (FLMINT)(*pszStr1 - *pszStr2));
#endif
}
	
/****************************************************************************
Desc:
****************************************************************************/
FLMINT FTKAPI f_stricmp(
	const char *		pszStr1,
	const char *		pszStr2)
{
#ifdef FLM_WIN
	return( _stricmp( pszStr1, pszStr2));
#else 
	while( f_toupper( *pszStr1) == f_toupper( *pszStr2) && *pszStr1)
	{
		pszStr1++;
		pszStr2++;
	}
	return( (FLMINT)( f_toupper( *pszStr1) - f_toupper( *pszStr2)));
#endif
}

/****************************************************************************
Desc:
****************************************************************************/
FLMINT FTKAPI f_strncmp(
	const char *		pszStr1,
	const char *		pszStr2,
	FLMSIZET				uiLength)
{
#ifndef FLM_NLM
	return( strncmp( pszStr1, pszStr2, uiLength));
#else
	while( *pszStr1 == *pszStr2 && *pszStr1 && uiLength)
	{
		pszStr1++;
		pszStr2++;
		uiLength--;
	}

	if( uiLength)
	{
		return( (*pszStr1 - *pszStr2));
	}

	return( 0);
#endif
}
	
/****************************************************************************
Desc:
****************************************************************************/
FLMINT FTKAPI f_strnicmp(
	const char *		pszStr1,
	const char *		pszStr2,
	FLMSIZET				uiLength)
{
#ifdef FLM_WIN
	return( _strnicmp( pszStr1, pszStr2, uiLength));
#else
	FLMINT				iLen = (FLMINT)uiLength;

	if( !pszStr1 || !pszStr2)
	{
		return( (pszStr1 == pszStr2) 
						? 0 
						: (pszStr1 ? 1 : -1));
	}

	while( iLen-- && *pszStr1 && *pszStr2 && 
		(f_toupper( *pszStr1) == f_toupper( *pszStr2)))
	{
		pszStr1++;
		pszStr2++;
	}

	return(	(iLen == -1)
					?	0
					:	(f_toupper( *pszStr1) - f_toupper( *pszStr2)));

#endif
}

/****************************************************************************
Desc:
****************************************************************************/
char * FTKAPI f_strcat(
	char *				pszDest,
	const char *		pszSrc)
{
#ifndef FLM_NLM
	return( strcat( pszDest, pszSrc));
#else
	const char *	p = pszSrc;
	char * 			q = pszDest;
	
	while (*q++);
	q--;
	while( (*q++ = *p++) != 0);
	
	return( pszDest);
#endif
}

/****************************************************************************
Desc:
****************************************************************************/
char * FTKAPI f_strncat(
	char *				pszDest,
	const char *		pszSrc,
	FLMSIZET				uiLength)
{
#ifndef FLM_NLM
	return( strncat( pszDest, pszSrc, uiLength));
#else
	const char *		p = pszSrc;
	char *				q = pszDest;
	
	while (*q++);
	
	q--;
	uiLength++;
	
	while( --uiLength)
	{
		if( (*q++ = *p++) == 0)
		{
			q--;
			break;
		}
	}
	
	*q = 0;
	return( pszDest);
#endif
}

/****************************************************************************
Desc:
****************************************************************************/
char * FTKAPI f_strchr(
	const char *		pszStr,
	unsigned char		ucByte)
{
#ifndef FLM_NLM
	return( (char *)strchr( pszStr, ucByte));
#else
	if( !pszStr)
	{
		return( NULL);
	}

	while (*pszStr && *pszStr != ucByte)
	{
		pszStr++;
	}

	return( (char *)((*pszStr == ucByte) 
								? pszStr
								: NULL));
#endif
}

/****************************************************************************
Desc:
****************************************************************************/
char * FTKAPI f_strrchr(
	const char *		pszStr,
	unsigned char		ucByte)
{
#ifndef FLM_NLM
	return( (char *)strrchr( pszStr, ucByte));
#else
	const char * pszLast = NULL;

	if( !pszStr)
	{
		return( NULL);
	}

	while (*pszStr)
	{
		if( *pszStr == ucByte)
		{
			pszLast = pszStr;
		}
		
		pszStr++;
	}
	
	if( ucByte == '\0')
	{
		pszLast = pszStr;
	}

	return( (char *)pszLast);
#endif
}

/****************************************************************************
Desc:
****************************************************************************/
char * FTKAPI f_strstr(
	const char *		pszStr1,
	const char *		pszStr2)
{
#ifndef FLM_NLM
	return( (char *)strstr( pszStr1, pszStr2));
#else
	FLMUINT 			i;
	FLMUINT			j;
	FLMUINT			k;

	if ( !pszStr1 || !pszStr2)
	{
		return( NULL);
	}

	for( i = 0; pszStr1[i] != '\0'; i++)
	{
		for( j=i, k=0; pszStr2[k] != '\0' &&
			pszStr1[j] == pszStr2[k]; j++, k++)
		{
			;
		}

		if ( k > 0 && pszStr2[k] == '\0')
		{
			return( (char *)&pszStr1[i]);
		}
	}

	return( NULL);
#endif
}

/****************************************************************************
Desc:		Turn a base 24 digit's ordinal value into a native
			alphanumeric value.
Notes:	This is a base 24 alphanumeric value where 
			{a, b, c, d, e, f, i, l, o, r, u, v } values are removed.
****************************************************************************/
FLMBYTE FTKAPI f_getBase24DigitChar( 
	FLMBYTE		ucValue)
{
	f_assert( ucValue <= 23);

	if( ucValue <= 9)
	{
		ucValue += NATIVE_ZERO;
	}
	else
	{
		ucValue = f_toascii( ucValue) - 10 + f_toascii( 'g');
		if( ucValue >= (FLMBYTE)'i')
		{
			ucValue++;
			if( ucValue >= (FLMBYTE)'l')
			{
				ucValue++;
				if( ucValue >= (FLMBYTE)'o')
				{
					ucValue++;
					if( ucValue >= (FLMBYTE)'r')
					{
						ucValue++;
						if( ucValue >= (FLMBYTE)'u')
						{
							ucValue++;
							if( ucValue >= (FLMBYTE)'v')
							{
								ucValue++;
							}
						}
					}
				}
			}
		}
	}
	
	return( ucValue);
}

/****************************************************************************
Desc:
****************************************************************************/
char * FTKAPI f_strupr(
	char *				pszStr)
{
#ifdef FLM_WIN
	return( _strupr( pszStr));
#else
	while( *pszStr)
	{
		*pszStr = f_toupper( *pszStr);
		pszStr++;
	}

	return( pszStr);
#endif
}

/**********************************************************************
Desc:
**********************************************************************/
FLMINT32 FTKAPI f_atomicInc(
	FLMATOMIC *			piTarget)
{
	#if defined( FLM_LIBC_NLM)
	{
		return( (FLMINT32)atomic_retadd( (unsigned long *)piTarget, 1));
	}
	#elif defined( FLM_RING_ZERO_NLM)
	{
		return( nlm_AtomicIncrement( (volatile LONG *)piTarget)); 
	}
	#elif defined( FLM_WIN)
	{
		return( (FLMINT32)InterlockedIncrement( (volatile LONG *)piTarget));
	}
	#elif defined( FLM_AIX)
	{
		return( (FLMINT32)aix_atomic_add( piTarget, 1));
	}
	#elif defined( FLM_OSX)
	{
		return( (FLMINT32)OSAtomicIncrement32Barrier( (int32_t *)piTarget));
	}
	#elif defined( FLM_SPARC_PLUS)
	{
		return( sparc_atomic_add_32( piTarget, 1));
	}
	#elif (defined( __i386__) || defined( __x86_64__)) && defined( FLM_GNUC)
	{
		FLMINT32 			i32Tmp;
		
		__asm__ __volatile__ (
						"lock;"
						"xaddl %0, %1"
							: "=r" (i32Tmp), "=m" (*piTarget)
							: "0" (1), "m" (*piTarget));
	
		return( i32Tmp + 1);
	}
	#elif defined( FLM_PPC) && defined( FLM_GNUC) && defined( FLM_LINUX)
	{
		return( ppc_atomic_add( piTarget, 1));
	}
	#elif defined( FLM_UNIX)
		return( posix_atomic_add_32( piTarget, 1));
	#else
		#error Atomic operations are not supported
	#endif
}

/**********************************************************************
Desc:
**********************************************************************/
FLMINT32 FTKAPI f_atomicDec(
	FLMATOMIC *			piTarget)
{
	#if defined( FLM_LIBC_NLM)
	{
		return( (FLMINT32)atomic_retadd( (unsigned long *)piTarget, -1));
	}
	#elif defined( FLM_RING_ZERO_NLM)
	{
		return( nlm_AtomicDecrement( (volatile LONG *)piTarget)); 
	}
	#elif defined( FLM_WIN)
	{
		return( (FLMINT32)InterlockedDecrement( (volatile LONG *)piTarget));
	}
	#elif defined( FLM_AIX)
	{
		return( (FLMINT32)aix_atomic_add( piTarget, -1));
	}
	#elif defined( FLM_OSX)
	{
		return( (FLMINT32)OSAtomicDecrement32Barrier( (int32_t *)piTarget));
	}
	#elif defined( FLM_SPARC_PLUS)
	{
		return( sparc_atomic_add_32( piTarget, -1));
	}
	#elif (defined( __i386__) || defined( __x86_64__)) && defined( FLM_GNUC)
	{
		FLMINT32				i32Tmp;
		
		__asm__ __volatile__ (
						"lock;" 
						"xaddl %0, %1"
							: "=r" (i32Tmp), "=m" (*piTarget)
							: "0" (-1), "m" (*piTarget));
	
		return( i32Tmp - 1);
	}
	#elif defined( FLM_PPC) && defined( FLM_GNUC) && defined( FLM_LINUX)
	{
		return( ppc_atomic_add( piTarget, -1));
	}
	#elif defined( FLM_UNIX)
		return( posix_atomic_add_32( piTarget, -1));
	#else
		#error Atomic operations are not supported
	#endif
}

/**********************************************************************
Desc:
**********************************************************************/
FLMINT32 FTKAPI f_atomicExchange(
	FLMATOMIC *			piTarget,
	FLMINT32				i32NewVal)
{
	#if defined( FLM_NLM)
	{
		return( (FLMINT32)atomic_xchg( (unsigned long *)piTarget, i32NewVal));
	}
	#elif defined( FLM_WIN)
	{
		return( (FLMINT32)InterlockedExchange( (volatile LONG *)piTarget,
			i32NewVal));
	}
	#elif defined( FLM_AIX)
	{
		int		iOldVal;
		
		for( ;;)
		{ 
			iOldVal = (int)*piTarget;
			
			if( compare_and_swap( (int *)piTarget, &iOldVal, i32NewVal))
			{
				break;
			}
		}
		
		return( (FLMINT32)iOldVal);
	}
	#elif defined( FLM_OSX)
	{
		int32_t		iOldVal;

		for( ;;)
		{
			iOldVal = (int32_t)*piTarget;

			if( OSAtomicCompareAndSwap32Barrier( iOldVal, i32NewVal, 
					(int32_t *)piTarget))
			{
				break;
			}
		}
		
		return( (FLMINT32)iOldVal);
	}
	#elif defined( FLM_SPARC_PLUS)
	{
		return( sparc_atomic_xchg_32( piTarget, i32NewVal));
	}
	#elif (defined( __i386__) || defined( __x86_64__)) && defined( FLM_GNUC)
	{
		FLMINT32 			i32OldVal;
		
		__asm__ __volatile__ (
						"1:	lock;"
						"		cmpxchgl %2, %0;"
						"		jne 1b"
							: "=m" (*piTarget), "=a" (i32OldVal)
							: "r" (i32NewVal), "m" (*piTarget), "a" (*piTarget));
	
		return( i32OldVal);
	}
	#elif defined( FLM_PPC) && defined( FLM_GNUC) && defined( FLM_LINUX)
	{
		return( ppc_atomic_xchg( piTarget, i32NewVal));
	}
	#elif defined( FLM_UNIX)
		return( posix_atomic_xchg_32( piTarget, i32NewVal));
	#else
		#error Atomic operations are not supported
	#endif
}

/**********************************************************************
Desc:
**********************************************************************/
FLMINT FTKAPI F_Object::getRefCount( void)
{
	return( m_refCnt);
}

/**********************************************************************
Desc:
**********************************************************************/
FLMINT FTKAPI F_Object::AddRef( void)
{
	return( ++m_refCnt);
}

/**********************************************************************
Desc:
**********************************************************************/
FLMINT FTKAPI F_Object::Release( void)
{
	FLMINT		iRefCnt = --m_refCnt;

	if( !iRefCnt)
	{
		delete this;
	}

	return( iRefCnt);
}

/**********************************************************************
Desc:
**********************************************************************/
IF_FileSystem * FTKAPI f_getFileSysPtr( void)
{
	return( gv_pFileSystem);
}

/**********************************************************************
Desc:
**********************************************************************/
FLMUINT FTKAPI f_getOpenFileCount( void)
{
	return( gv_openFiles);
}

/**********************************************************************
Desc:
**********************************************************************/
IF_ThreadMgr * f_getThreadMgrPtr( void)
{
	return( gv_pThreadMgr);
}

/**********************************************************************
Desc:
**********************************************************************/
FLMUINT FTKAPI f_getMaxFileSize( void)
{
	return( gv_uiMaxFileSize);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE FTKAPI f_readSEN(
	IF_IStream *	pIStream,
	FLMUINT *		puiValue,
	FLMUINT *		puiLength)
{
	RCODE				rc;
	FLMUINT64		ui64Tmp;

	if( RC_BAD( rc = f_readSEN64( pIStream, &ui64Tmp, puiLength)))
	{
		goto Exit;
	}

	if( ui64Tmp > ~((FLMUINT)0))
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}

	if( puiValue)
	{
		*puiValue = (FLMUINT)ui64Tmp;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE FTKAPI f_readSEN64(
	IF_IStream *		pIStream,
	FLMUINT64 *			pui64Value,
	FLMUINT *			puiLength)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT				uiLen;
	FLMUINT				uiSENLength;
	FLMBYTE				ucBuffer[ 16];
	const FLMBYTE *	pucBuffer;

	uiLen = 1;
	if( RC_BAD( rc = pIStream->read( 
		(char *)&ucBuffer[ 0], uiLen, &uiLen)))
	{
		goto Exit;
	}

	uiSENLength = 	gv_ucSENLengthArray[ ucBuffer[ 0]];
	uiLen = uiSENLength - 1;

	if( puiLength)
	{
		*puiLength = uiSENLength;
	}

	if( pui64Value)
	{
		pucBuffer = &ucBuffer[ 1];
	}
	else
	{
		pucBuffer = NULL;
	}

	if( uiLen)
	{
		if( RC_BAD( rc = pIStream->read( 
			(char *)pucBuffer, uiLen, &uiLen)))
		{
			goto Exit;
		}
	}

	if( pui64Value)
	{
		pucBuffer = &ucBuffer[ 0];
		if( RC_BAD( rc = f_decodeSEN64( &pucBuffer,
			&ucBuffer[ sizeof( ucBuffer)], pui64Value)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
FLMUINT FTKAPI f_getSENLength(
	FLMBYTE 					ucByte)
{
	return( gv_ucSENLengthArray[ ucByte]);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI f_decodeSEN64(
	const FLMBYTE **		ppucBuffer,
	const FLMBYTE *		pucEnd,
	FLMUINT64 *				pui64Value)
{
	RCODE						rc = NE_FLM_OK;
	FLMUINT					uiSENLength;
	const FLMBYTE *		pucBuffer = *ppucBuffer;

	uiSENLength = gv_ucSENLengthArray[ *pucBuffer];
	if( pucBuffer + uiSENLength > pucEnd)
	{
		if (pui64Value)
		{
			*pui64Value = 0;
		}
		rc = RC_SET( NE_FLM_BAD_SEN);
		goto Exit;
	}

	if (pui64Value)
	{
		switch( uiSENLength)
		{
			case 1:
				*pui64Value = *pucBuffer;
				break;
	
			case 2:
				*pui64Value = (((FLMUINT64)(*pucBuffer & 0x3F)) << 8) + pucBuffer[ 1];
				break;
	
			case 3:
				*pui64Value = (((FLMUINT64)(*pucBuffer & 0x1F)) << 16) +
					(((FLMUINT64)pucBuffer[ 1]) << 8) + pucBuffer[ 2];
				break;
	
			case 4:
				*pui64Value = (((FLMUINT64)(*pucBuffer & 0x0F)) << 24) +
					(((FLMUINT64)pucBuffer[ 1]) << 16) +
					(((FLMUINT64)pucBuffer[ 2]) << 8) + pucBuffer[ 3];
				break;
	
			case 5:
				*pui64Value = (((FLMUINT64)(*pucBuffer & 0x07)) << 32) +
					(((FLMUINT64)pucBuffer[ 1]) << 24) +
					(((FLMUINT64)pucBuffer[ 2]) << 16) +
					(((FLMUINT64)pucBuffer[ 3]) << 8) + pucBuffer[ 4];
				break;
	
			case 6:
				*pui64Value = (((FLMUINT64)(*pucBuffer & 0x03)) << 40) +
					(((FLMUINT64)pucBuffer[ 1]) << 32) +
					(((FLMUINT64)pucBuffer[ 2]) << 24) +
					(((FLMUINT64)pucBuffer[ 3]) << 16) +
					(((FLMUINT64)pucBuffer[ 4]) << 8) + pucBuffer[ 5];
				break;
	
			case 7:
				*pui64Value = (((FLMUINT64)(*pucBuffer & 0x01)) << 48) +
					(((FLMUINT64)pucBuffer[ 1]) << 40) +
					(((FLMUINT64)pucBuffer[ 2]) << 32) +
					(((FLMUINT64)pucBuffer[ 3]) << 24) +
					(((FLMUINT64)pucBuffer[ 4]) << 16) +
					(((FLMUINT64)pucBuffer[ 5]) << 8) + pucBuffer[ 6];
				break;
	
			case 8:
				*pui64Value = (((FLMUINT64)pucBuffer[ 1]) << 48) +
					(((FLMUINT64)pucBuffer[ 2]) << 40) +
					(((FLMUINT64)pucBuffer[ 3]) << 32) +
					(((FLMUINT64)pucBuffer[ 4]) << 24) +
					(((FLMUINT64)pucBuffer[ 5]) << 16) +
					(((FLMUINT64)pucBuffer[ 6]) << 8) + pucBuffer[ 7];
				break;
	
			case 9:
				*pui64Value = (((FLMUINT64)pucBuffer[ 1]) << 56) +
					(((FLMUINT64)pucBuffer[ 2]) << 48) +
					(((FLMUINT64)pucBuffer[ 3]) << 40) +
					(((FLMUINT64)pucBuffer[ 4]) << 32) +
					(((FLMUINT64)pucBuffer[ 5]) << 24) +
					(((FLMUINT64)pucBuffer[ 6]) << 16) +
					(((FLMUINT64)pucBuffer[ 7]) << 8) + pucBuffer[ 8];
				break;
	
			default:
				*pui64Value = 0;
				f_assert( 0);
				break;
		}
	}

Exit:

	*ppucBuffer = pucBuffer + uiSENLength;

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI f_decodeSEN(
	const FLMBYTE **		ppucBuffer,
	const FLMBYTE *		pucEnd,
	FLMUINT *				puiValue)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT64		ui64Value;
	
	if( RC_BAD( rc = f_decodeSEN64( ppucBuffer, pucEnd, &ui64Value)))
	{
		return( rc);
	}
	
	if( ui64Value > FLM_MAX_UINT)
	{
		return( RC_SET_AND_ASSERT( NE_FLM_CONV_NUM_OVERFLOW));
	}
	
	if( puiValue)
	{
		*puiValue = (FLMUINT)ui64Value;
	}

	return( rc);
}
	
/****************************************************************************
Desc:
****************************************************************************/
FINLINE FLMBYTE f_shiftRightRetByte(
	FLMUINT64	ui64Num,
	FLMBYTE		ucBits)
{
	return( ucBits < 64 ? (FLMBYTE)(ui64Num >> ucBits) : 0);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT FTKAPI f_getSENByteCount(
	FLMUINT64	ui64Num)
{
	FLMUINT		uiCount = 0;

	if( ui64Num < 0x80)
	{
		return( 1);
	}

	while( ui64Num)
	{
		uiCount++;
		ui64Num >>= 7;
	}

	// If the high bit is set, the counter will be incremented 1 beyond
	// the actual number of bytes need to represent the SEN.  We will need
	// to re-visit this if we ever go beyond 64-bits.

	return( uiCount < FLM_MAX_SEN_LEN ? uiCount : FLM_MAX_SEN_LEN);
}

/****************************************************************************
Desc:		Encodes a number as a SEN
****************************************************************************/
FLMUINT FTKAPI f_encodeSEN(
	FLMUINT64		ui64Value,
	FLMBYTE **		ppucBuffer,
	FLMUINT			uiSizeWanted)
{
	FLMBYTE *		pucBuffer = *ppucBuffer;
	FLMUINT			uiSenLen = f_getSENByteCount( ui64Value);

	f_assert( uiSizeWanted <= FLM_MAX_SEN_LEN && 
				  (!uiSizeWanted || uiSizeWanted >= uiSenLen));

	uiSenLen = uiSizeWanted > uiSenLen ? uiSizeWanted : uiSenLen;

	if( uiSenLen == 1)
	{
		*pucBuffer++ = (FLMBYTE)ui64Value;
	}
	else
	{
		FLMUINT			uiTmp = (uiSenLen - 1) << 3;

		*pucBuffer++ = ucSENPrefixArray[ uiSenLen] + 
							f_shiftRightRetByte( ui64Value, (FLMBYTE)uiTmp);
		while( uiTmp)
		{
			uiTmp -= 8;
			*pucBuffer++ = f_shiftRightRetByte( ui64Value, (FLMBYTE)uiTmp);
		}
	}

	*ppucBuffer = pucBuffer;
	return( uiSenLen);
}

/****************************************************************************
Desc:		Encodes a number as a SEN
****************************************************************************/
RCODE FTKAPI f_encodeSEN(
	FLMUINT64		ui64Value,
	FLMBYTE **		ppucBuffer,
	FLMBYTE *		pucEnd)
{
	RCODE				rc = NE_FLM_OK;
	FLMBYTE *		pucBuffer = *ppucBuffer;
	FLMUINT			uiSenLen = f_getSENByteCount( ui64Value);
	
	if( *ppucBuffer + uiSenLen > pucEnd)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}

	if( uiSenLen == 1)
	{
		*pucBuffer++ = (FLMBYTE)ui64Value;
	}
	else
	{
		FLMUINT			uiTmp = (uiSenLen - 1) << 3;

		*pucBuffer++ = ucSENPrefixArray[ uiSenLen] + 
							f_shiftRightRetByte( ui64Value, (FLMBYTE)uiTmp);
		while( uiTmp)
		{
			uiTmp -= 8;
			*pucBuffer++ = f_shiftRightRetByte( ui64Value, (FLMBYTE)uiTmp);
		}
	}

	*ppucBuffer = pucBuffer;
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:		Encodes a number as a SEN
****************************************************************************/
FLMUINT FTKAPI f_encodeSENKnownLength(
	FLMUINT64		ui64Value,
	FLMUINT			uiSenLen,
	FLMBYTE **		ppucBuffer)
{
	FLMBYTE *			pucBuffer = *ppucBuffer;

	if( uiSenLen == 1)
	{
		*pucBuffer++ = (FLMBYTE)ui64Value;
	}
	else
	{
		FLMUINT			uiTmp = (uiSenLen - 1) << 3;

		*pucBuffer++ = ucSENPrefixArray[ uiSenLen] + 
							f_shiftRightRetByte( ui64Value, (FLMBYTE)uiTmp);
		while( uiTmp)
		{
			uiTmp -= 8;
			*pucBuffer++ = f_shiftRightRetByte( ui64Value, (FLMBYTE)uiTmp);
		}
	}

	*ppucBuffer = pucBuffer;
	return( uiSenLen);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI FlmGetXMLObject(
	IF_XML **				ppXmlObject)
{
	*ppXmlObject = gv_pXml;
	(*ppXmlObject)->AddRef();
	
	return( NE_FLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
IF_XML * f_getXmlObjPtr( void)
{
	return( gv_pXml);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT32 FTKAPI f_getRandomUINT32(
	FLMUINT32		ui32Low,
	FLMUINT32		ui32High)
{
	FLMUINT32		ui32Value;

	f_mutexLock( gv_hRandomGenMutex);	
	ui32Value = gv_pRandomGenerator->getUINT32( ui32Low, ui32High);
	f_mutexUnlock( gv_hRandomGenMutex);	
	
	return( ui32Value);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMBYTE FTKAPI f_getRandomByte( void)
{
	FLMBYTE		ucValue;

	f_mutexLock( gv_hRandomGenMutex);	
	ucValue = (FLMBYTE)(gv_pRandomGenerator->getUINT32( 0, 0xFF));
	f_mutexUnlock( gv_hRandomGenMutex);	
	
	return( ucValue);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_ListManager::insertFirst(
	FLMUINT				uiList,
	F_ListItem *		pNewFirstItem)
{
	F_ListNode *		pListNode;
	
	f_assert( uiList < m_uiListNodeCnt);
	
	pNewFirstItem->AddRef();
	pListNode = &m_pListNodes[ uiList];
	
	if( !pListNode->pNextItem)
	{
		pListNode->pPrevItem = pNewFirstItem;
		pNewFirstItem->setNextListItem( uiList, NULL);
	}
	else
	{
		// Add this new item to the first of the list.
		
		pListNode->pNextItem->setPrevListItem( uiList, pNewFirstItem);
		pNewFirstItem->setNextListItem( uiList, pListNode->pNextItem);
	}

	pListNode->pNextItem = pNewFirstItem;
	pNewFirstItem->setPrevListItem( uiList, NULL);
	pNewFirstItem->m_bInList = TRUE;
	pListNode->uiListCount++;
}

/****************************************************************************
Desc:
****************************************************************************/
void F_ListManager::insertLast(
	FLMUINT				uiList,
	F_ListItem *		pNewLastItem)
{
	F_ListNode *		pListNode;
	
	f_assert( uiList < m_uiListNodeCnt);
	
	pNewLastItem->AddRef();
	pListNode = &m_pListNodes[ uiList];
	
	if( !pListNode->pPrevItem)
	{
		pListNode->pNextItem = pNewLastItem;
		pNewLastItem->setPrevListItem( uiList, NULL);
	}
	else
	{
		// Add this new item to the end of the list.
		
		pListNode->pPrevItem->setNextListItem( uiList, pNewLastItem);
		pNewLastItem->setPrevListItem( uiList, pListNode->pPrevItem);
	}

	pListNode->pPrevItem = pNewLastItem;
	pNewLastItem->setNextListItem( uiList, NULL);
	pNewLastItem->m_bInList = TRUE;
	pListNode->uiListCount++;
}

/****************************************************************************
Desc:
****************************************************************************/
F_ListItem * F_ListManager::getItem(
	FLMUINT				uiList,
	FLMUINT				nth)
{
	F_ListNode *		pListNode;
	F_ListItem *		pListItem;
	
	// Check bounds with assert.  
	
	f_assert( uiList < m_uiListNodeCnt );

	pListNode = &m_pListNodes[ uiList ];
	pListItem = pListNode ? pListNode->pNextItem : NULL;
	
	while( nth--)
	{
		pListItem = pListItem->getNextListItem( uiList);
	}
	
	return( pListItem);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_ListManager::removeItem(
	FLMUINT				uiList,
	F_ListItem *		pItem)
{
	F_ListNode *		pMgrListNode;
	F_ListItem *		pPrevItem;
	F_ListItem *		pNextItem;
	
	f_assert( uiList < m_uiListNodeCnt);

	pMgrListNode = &m_pListNodes[ uiList];

	// Get this item's prev and next items

	pPrevItem = pItem->getPrevListItem( uiList);
	pNextItem = pItem->getNextListItem( uiList);

	if( !pPrevItem && !pNextItem && pMgrListNode->pPrevItem != pItem && 
		 pMgrListNode->pNextItem != pItem)
	{
		// If the item is not within the list then skip to the end
		
		goto Exit;
	}

	// Determine if this item is pointed to by the head or tail pointers
	// that the list manager maintains

	if( pMgrListNode->pPrevItem == pItem)
	{
		pMgrListNode->pPrevItem = pItem->getPrevListItem( uiList);
	}

	if( pMgrListNode->pNextItem == pItem)
	{
		pMgrListNode->pNextItem = pItem->getNextListItem( uiList);
	}

	// If there is a prev item - change it's next ptr to be items next ptr
	
	if( pPrevItem)
	{
		pPrevItem->setNextListItem( uiList, pItem->getNextListItem( uiList));
	}

	// If there is a next item - change it's prev ptr to be items prev ptr
	
	if( pNextItem)
	{
		pNextItem->setPrevListItem( uiList, pItem->getPrevListItem( uiList));
	}

	// Clear out this items prev and next links
	
	pItem->setPrevListItem( uiList, NULL);
	pItem->setNextListItem( uiList, NULL);
	pItem->m_bInList = FALSE;
	pItem->Release();
	pMgrListNode->uiListCount--;

Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
void F_ListManager::clearList(
	FLMUINT			uiList)
{
	FLMUINT			uiListCnt;
	F_ListNode *	pListNode;

	f_assert( (FLM_ALL_LISTS == uiList) || (uiList < m_uiListNodeCnt));

	if( uiList == FLM_ALL_LISTS)
	{
		uiList = 0;
		uiListCnt = m_uiListNodeCnt;
		pListNode = m_pListNodes;
	}
	else
	{
		uiListCnt = 1;
		pListNode = &m_pListNodes[ uiList ];
	}
	
	for( ; uiListCnt--; pListNode++, uiList++)
	{
		F_ListItem *		pItem;
		F_ListItem *		pNextItem;
		
		// Go through the list Releasing every list item.
		
		for( pItem = pListNode->pNextItem; pItem; pItem = pNextItem)
		{
			pNextItem = pItem->getNextListItem( uiList);
			removeItem( uiList, pItem);
		}

		// At this point the ListCount should be at 0.
		
		f_assert( !pListNode->uiListCount);

		// Clear the managers head and tail list pointers.
		
		pListNode->pNextItem = pListNode->pPrevItem = NULL;
	}
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT F_ListManager::getItemCount(
	FLMUINT			uiList)
{
	FLMUINT			uiListNodeCnt;
	FLMUINT			uiCount = 0;
	F_ListNode *	pListNode;

	f_assert( (FLM_ALL_LISTS == uiList) || (uiList < m_uiListNodeCnt));

	if( uiList == FLM_ALL_LISTS)
	{
		uiListNodeCnt = m_uiListNodeCnt;
		pListNode = m_pListNodes;
	}
	else
	{
		uiListNodeCnt = 1;
		pListNode = &m_pListNodes[ uiList];
	}

	for( ; uiListNodeCnt--; pListNode++)
	{
		uiCount += pListNode->uiListCount;
	}

	return( uiCount);
}

/****************************************************************************
Desc:
****************************************************************************/
F_ListItem::~F_ListItem()
{
#ifdef FLM_DEBUG
	FLMUINT			uiLoop;
	F_ListNode *	pTmpNd;

	f_assert( !m_bInList);

	for( uiLoop = 0; uiLoop < m_uiListNodeCnt; uiLoop++)
	{
		pTmpNd = &m_pListNodes[ uiLoop];
		f_assert( !pTmpNd->pPrevItem && !pTmpNd->pNextItem);
	}
#endif
}

/****************************************************************************
Desc:
****************************************************************************/
void F_ListItem::setup(
	F_ListManager *	pListMgr,
	F_ListNode *		pListNodes,
	FLMUINT				uiListNodeCnt)
{
	f_assert( pListMgr);
	f_assert( pListNodes);
	f_assert( uiListNodeCnt);

	m_pListManager = pListMgr;
	m_uiListNodeCnt = uiListNodeCnt;
	m_pListNodes = pListNodes;
	
	f_memset( pListNodes, 0, sizeof( F_ListNode) * uiListNodeCnt );
}

/****************************************************************************
Desc:
****************************************************************************/
void F_ListItem::removeFromList(
	FLMUINT			uiList)
{

	f_assert( (uiList < m_uiListNodeCnt) || (uiList == FLM_ALL_LISTS));

	if( uiList == FLM_ALL_LISTS)
	{
		FLMUINT			uiListCnt = m_uiListNodeCnt;
		F_ListNode *	pListNode = m_pListNodes;

		uiList = 0;

		// Remove this item from all lists

		for( ; uiListCnt--; uiList++, pListNode++)
		{
			m_pListManager->removeItem( uiList, this);
		}
	}
	else
	{
		// Remove item from a specific list
		
		m_pListManager->removeItem( uiList, this);
	}
}

/****************************************************************************
Desc: This routine allocates and initializes a hash table.
****************************************************************************/
RCODE FTKAPI f_allocHashTable(
	FLMUINT					uiHashTblSize,
	F_BUCKET **				ppHashTblRV)
{
	RCODE						rc = NE_FLM_OK;
	F_BUCKET *				pHashTbl = NULL;
	IF_RandomGenerator *	pRandGen = NULL;
	FLMUINT					uiCnt;
	FLMUINT					uiRandVal;
	FLMUINT					uiTempVal;
	
	// Allocate memory for the hash table

	if (RC_BAD( rc = f_calloc(
		(FLMUINT)(sizeof( F_BUCKET)) * uiHashTblSize, &pHashTbl)))
	{
		goto Exit;
	}

	// Set up the random number generator
	
	if( RC_BAD( rc = FlmAllocRandomGenerator( &pRandGen)))
	{
		goto Exit;
	}

	pRandGen->setSeed( 1);

	for (uiCnt = 0; uiCnt < uiHashTblSize; uiCnt++)
	{
		pHashTbl [uiCnt].uiHashValue = (FLMBYTE)uiCnt;
		pHashTbl [uiCnt].pFirstInBucket = NULL;
	}

	if( uiHashTblSize <= 256)
	{
		for( uiCnt = 0; uiCnt < uiHashTblSize - 1; uiCnt++)
		{
			uiRandVal = (FLMBYTE) pRandGen->getUINT32( (FLMUINT32)uiCnt,
										(FLMUINT32)(uiHashTblSize - 1));
			if( uiRandVal != uiCnt)
			{
				uiTempVal = (FLMBYTE)pHashTbl [uiCnt].uiHashValue;
				pHashTbl [uiCnt].uiHashValue = pHashTbl [uiRandVal].uiHashValue;
				pHashTbl [uiRandVal].uiHashValue = uiTempVal;
			}
		}
	}

Exit:

	if( pRandGen)
	{
		pRandGen->Release();
	}

	*ppHashTblRV = pHashTbl;
	return( rc);
}

/****************************************************************************
Desc: This routine determines the hash bucket for a string.
****************************************************************************/
FLMUINT FTKAPI f_strHashBucket(
	char *		pszStr,
	F_BUCKET *	pHashTbl,
	FLMUINT		uiNumBuckets)
{
	FLMUINT	uiHashIndex;

	if ((uiHashIndex = (FLMUINT)*pszStr) >= uiNumBuckets)
	{
		uiHashIndex -= uiNumBuckets;
	}

	while (*pszStr)
	{
		if ((uiHashIndex = (FLMUINT)((pHashTbl [uiHashIndex].uiHashValue) ^ 
			(FLMUINT)(f_toupper( *pszStr)))) >= uiNumBuckets)
		{
			uiHashIndex -= uiNumBuckets;
		}
		pszStr++;
	}

	return( uiHashIndex);
}

/****************************************************************************
Desc: This routine determines the hash bucket for a binary array of
		characters.
****************************************************************************/
FLMUINT FTKAPI f_binHashBucket(
	void *		pBuf,
	FLMUINT		uiBufLen,
	F_BUCKET *	pHashTbl,
	FLMUINT		uiNumBuckets)
{
	FLMUINT		uiHashIndex;
	FLMBYTE *	ptr = (FLMBYTE *)pBuf;

	if ((uiHashIndex = (FLMUINT)*ptr) >= uiNumBuckets)
		uiHashIndex -= uiNumBuckets;
	while (uiBufLen)
	{
		if ((uiHashIndex =
				(FLMUINT)((pHashTbl [uiHashIndex].uiHashValue) ^ (FLMUINT)(*ptr))) >=
					uiNumBuckets)
			uiHashIndex -= uiNumBuckets;
		ptr++;
		uiBufLen--;
	}
	return( uiHashIndex);
}

/****************************************************************************
Desc:
****************************************************************************/
F_HashTable::F_HashTable()
{
	m_hMutex = F_MUTEX_NULL;
	m_pMRUObject = NULL;
	m_pLRUObject = NULL;
	m_ppHashTable = NULL;
	m_uiBuckets = 0;
	m_uiObjects = 0;
	m_uiMaxObjects = 0;
}

/****************************************************************************
Desc:
****************************************************************************/
F_HashTable::~F_HashTable()
{
	F_HashObject *		pCur;
	F_HashObject *		pNext;

	pCur = m_pMRUObject;
	while( pCur)
	{
		pNext = pCur->m_pNextInGlobal;
		unlinkObject( pCur);
		pCur->Release();
		pCur = pNext;
	}

	f_assert( !m_uiObjects);
	f_assert( !m_pMRUObject);
	f_assert( !m_pLRUObject);

	if( m_ppHashTable)
	{
		f_free( &m_ppHashTable);
	}

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}
}

/****************************************************************************
Desc:	Configures the hash table prior to first use
****************************************************************************/
RCODE FTKAPI F_HashTable::setupHashTable(
	FLMBOOL			bMultithreaded,
	FLMUINT			uiNumBuckets,
	FLMUINT			uiMaxObjects)
{
	RCODE			rc = NE_FLM_OK;

	f_assert( uiNumBuckets);

	// Create the hash table

	if( RC_BAD( rc = f_alloc( 
		sizeof( F_HashObject *) * uiNumBuckets, &m_ppHashTable)))
	{
		goto Exit;
	}
	
	m_uiObjects = 0;
	m_uiMaxObjects = uiMaxObjects;
	m_uiBuckets = uiNumBuckets;
	f_memset( m_ppHashTable, 0, sizeof( F_HashObject *) * uiNumBuckets);

	if( bMultithreaded)
	{
		// Initialize the mutex

		if( RC_BAD( rc = f_mutexCreate( &m_hMutex)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Retrieves an object from the hash table with the specified key.
		This routine assumes the table's mutex has already been locked.
		A reference IS NOT added to the object for the caller.
****************************************************************************/
RCODE F_HashTable::findObject(
	const void *		pvKey,
	FLMUINT				uiKeyLen,
	F_HashObject **	ppObject)
{
	RCODE					rc = NE_FLM_OK;
	F_HashObject *		pObject = NULL;
	FLMUINT				uiBucket;
	FLMUINT32			ui32CRC = 0;

	*ppObject = NULL;

	// Calculate the hash bucket and mutex offset

	uiBucket = getHashBucket( pvKey, uiKeyLen, &ui32CRC);

	// Search the bucket for an object with a matching
	// key.

	pObject = m_ppHashTable[ uiBucket];
	while( pObject)
	{
		if( pObject->getKeyCRC() == ui32CRC)
		{
			const void *	pvTmpKey = pObject->getKey();
			FLMUINT			uiTmpKeyLen = pObject->getKeyLength();

			if( uiTmpKeyLen == uiKeyLen &&
				f_memcmp( pvTmpKey, pvKey, uiKeyLen) == 0)
			{
				break;
			}
		}
		
		pObject = pObject->m_pNextInBucket;
	}

	if( !pObject)
	{
		rc = RC_SET( NE_FLM_NOT_FOUND);
		goto Exit;
	}

	*ppObject = pObject;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Adds an object to the hash table
****************************************************************************/
RCODE FTKAPI F_HashTable::addObject(
	F_HashObject *		pObject,
	FLMBOOL				bAllowDuplicates)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT				uiBucket;
	F_HashObject *		pTmp;
	const void *		pvKey;
	FLMUINT				uiKeyLen;
	FLMUINT32			ui32CRC;
	FLMBOOL				bMutexLocked = FALSE;

	// Calculate and set the objects hash bucket

	f_assert( pObject->getHashBucket() == F_INVALID_HASH_BUCKET);

	pvKey = pObject->getKey();
	uiKeyLen = pObject->getKeyLength();
	f_assert( uiKeyLen);

	uiBucket = getHashBucket( pvKey, uiKeyLen, &ui32CRC);
	pObject->m_ui32KeyCRC = ui32CRC;

	// Lock the mutex

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexLock( m_hMutex);
		bMutexLocked = TRUE;
	}

	// Make sure the object doesn't already exist

	if( !bAllowDuplicates)
	{
		if( RC_BAD( rc = findObject( pvKey, uiKeyLen, &pTmp)))
		{
			if( rc != NE_FLM_NOT_FOUND)
			{
				goto Exit;
			}
			rc = NE_FLM_OK;
		}
		else
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_EXISTS);
			goto Exit;
		}
	}

	// Add a reference to the object

	pObject->AddRef();

	// Link the object into the appropriate lists

	linkObject( pObject, uiBucket);
	
	// Make sure the maximum number of objects hasn't been exceeded
	
	if( m_uiMaxObjects)
	{
		while( m_uiObjects > m_uiMaxObjects)
		{
			if( (pTmp = m_pLRUObject) == NULL)
			{
				break;
			}

			unlinkObject( pTmp);
		}
	}

Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:	Returns the next object in the linked list of objects in the hash
		table.  If *ppObject == NULL, the first object will be returned.
****************************************************************************/
RCODE FTKAPI F_HashTable::getNextObjectInGlobal(
	F_HashObject **	ppObject)
{
	RCODE					rc = NE_FLM_OK;
	FLMBOOL				bMutexLocked = FALSE;
	F_HashObject *		pOldObj;

	// Lock the mutex

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexLock( m_hMutex);
		bMutexLocked = TRUE;
	}

	if( !(*ppObject))
	{
		*ppObject = m_pMRUObject;
	}
	else
	{
		pOldObj = *ppObject;
		*ppObject = (*ppObject)->m_pNextInGlobal;
		pOldObj->Release();
	}

	if( *ppObject == NULL)
	{
		rc = RC_SET( NE_FLM_EOF_HIT);
		goto Exit;
	}

	(*ppObject)->AddRef();

Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_HashTable::getNextObjectInBucket(
	F_HashObject **	ppObject)
{
	RCODE					rc = NE_FLM_OK;
	FLMBOOL				bMutexLocked = FALSE;
	F_HashObject *		pOldObj;

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexLock( m_hMutex);
		bMutexLocked = TRUE;
	}

	if( !(*ppObject))
	{
		rc = RC_SET( NE_FLM_EOF_HIT);
		goto Exit;
	}
	
	pOldObj = *ppObject;
	*ppObject = (*ppObject)->m_pNextInBucket;
	pOldObj->Release();

	if( *ppObject == NULL)
	{
		rc = RC_SET( NE_FLM_EOF_HIT);
		goto Exit;
	}

	(*ppObject)->AddRef();

Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:	Retrieves an object from the hash table with the specified key
****************************************************************************/
RCODE FTKAPI F_HashTable::getObject(
	const void *		pvKey,
	FLMUINT				uiKeyLen,
	F_HashObject **	ppObject,
	FLMBOOL				bRemove)
{
	RCODE				rc = NE_FLM_OK;
	F_HashObject *	pObject;
	FLMBOOL			bMutexLocked = FALSE;

	// Lock the mutex

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexLock( m_hMutex);
		bMutexLocked = TRUE;
	}

	// Search for an object with a matching key.

	if( RC_BAD( rc = findObject( pvKey, uiKeyLen, &pObject)))
	{
		goto Exit;
	}

	if( pObject && bRemove)
	{
		unlinkObject( pObject);
		if( !ppObject)
		{
			pObject->Release();
			pObject = NULL;
		}
	}

	if( ppObject)
	{
		if( !bRemove)
		{
			pObject->AddRef();
		}
		
		*ppObject = pObject;
		pObject = NULL;
	}

Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:	Removes an object from the hash table by key
****************************************************************************/
RCODE FTKAPI F_HashTable::removeObject(
	void *			pvKey,
	FLMUINT			uiKeyLen)
{
	return( getObject( pvKey, uiKeyLen, NULL, TRUE));
}

/****************************************************************************
Desc:	Removes an object from the hash table by object pointer
****************************************************************************/
RCODE FTKAPI F_HashTable::removeObject(
	F_HashObject *		pObject)
{
	const void *	pvKey = pObject->getKey();
	FLMUINT			uiKeyLen = pObject->getKeyLength();

	return( getObject( pvKey, uiKeyLen, NULL, TRUE));
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI F_HashTable::removeAllObjects( void)
{
	F_HashObject *		pCur;
	FLMBOOL				bMutexLocked = FALSE;

	for( ;;)
	{
		if( m_hMutex != F_MUTEX_NULL)
		{
			f_mutexLock( m_hMutex);
			bMutexLocked = TRUE;
		}
		
		if( (pCur = m_pMRUObject) == NULL)
		{
			break;
		}

		unlinkObject( pCur);

		if( bMutexLocked)
		{
			f_mutexUnlock( m_hMutex);
			bMutexLocked = FALSE;
		}

		pCur->Release();
	}

	f_assert( !m_pMRUObject);
	f_assert( !m_pLRUObject);

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI F_HashTable::removeAgedObjects(
	FLMUINT				uiMaxAge)
{
	F_HashObject *		pCur;
	FLMBOOL				bMutexLocked = FALSE;
	FLMUINT				uiCurrentTime = FLM_GET_TIMER();

	for( ;;)
	{
		if( m_hMutex != F_MUTEX_NULL)
		{
			f_mutexLock( m_hMutex);
			bMutexLocked = TRUE;
		}
		
		if( (pCur = m_pLRUObject) == NULL)
		{
			break;
		}
		
		if( FLM_TIMER_UNITS_TO_SECS( 
			FLM_ELAPSED_TIME( uiCurrentTime, pCur->m_uiTimeAdded)) < uiMaxAge)
		{
			break;
		}

		unlinkObject( pCur);

		if( bMutexLocked)
		{
			f_mutexUnlock( m_hMutex);
			bMutexLocked = FALSE;
		}

		pCur->Release();
	}

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT FTKAPI F_HashTable::getMaxObjects( void)
{
	FLMUINT		uiMaxObjects;
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexLock( m_hMutex);
	}
	
	uiMaxObjects = m_uiMaxObjects;
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( m_hMutex);
	}
	
	return( m_uiMaxObjects);
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_HashTable::setMaxObjects(
	FLMUINT				uiMaxObjects)
{
	F_HashObject *		pCur;
	FLMBOOL				bMutexLocked = FALSE;
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexLock( m_hMutex);
		bMutexLocked = TRUE;
	}
	
	m_uiMaxObjects = uiMaxObjects;
	
	while( m_uiObjects > m_uiMaxObjects)
	{
		if( !bMutexLocked && m_hMutex != F_MUTEX_NULL)
		{
			f_mutexLock( m_hMutex);
			bMutexLocked = TRUE;
		}
		
		if( (pCur = m_pLRUObject) == NULL)
		{
			break;
		}
		
		unlinkObject( pCur);

		if( bMutexLocked)
		{
			f_mutexUnlock( m_hMutex);
			bMutexLocked = FALSE;
		}

		pCur->Release();
	}
	
	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}
	
	return( NE_FLM_OK);
}
	
/****************************************************************************
Desc:	Calculates the hash bucket of a key and optionally returns the key's
		CRC.
****************************************************************************/
FLMUINT F_HashTable::getHashBucket(
	const void *	pvKey,
	FLMUINT			uiLen,
	FLMUINT32 *		pui32KeyCRC)
{
	FLMUINT32		ui32CRC = 0;

	f_updateCRC( (FLMBYTE *)pvKey, uiLen, &ui32CRC);
	
	if( pui32KeyCRC)
	{
		*pui32KeyCRC = ui32CRC;
	}
	
	return( ui32CRC % m_uiBuckets);
}

/****************************************************************************
Desc:		Links an object to the global list and also to its bucket
Notes:	This routine assumes that the bucket's mutex is already locked
			if the hash table is multi-threaded.
****************************************************************************/
void F_HashTable::linkObject(
	F_HashObject *		pObject,
	FLMUINT				uiBucket)
{
	f_assert( uiBucket < m_uiBuckets);
	f_assert( pObject->getHashBucket() == F_INVALID_HASH_BUCKET);

	// Set the object's bucket

	pObject->setHashBucket( uiBucket);

	// Link the object to its hash bucket

	pObject->m_pNextInBucket = m_ppHashTable[ uiBucket];
	if( m_ppHashTable[ uiBucket])
	{
		m_ppHashTable[ uiBucket]->m_pPrevInBucket = pObject;
	}
	m_ppHashTable[ uiBucket] = pObject;

	// Link to the global list

	if( (pObject->m_pNextInGlobal = m_pMRUObject) != NULL)
	{
		m_pMRUObject->m_pPrevInGlobal = pObject;
	}
	else
	{
		m_pLRUObject = pObject;
	}
	
	pObject->m_uiTimeAdded = FLM_GET_TIMER();
	m_pMRUObject = pObject;
	m_uiObjects++;
}

/****************************************************************************
Desc:		Unlinks an object from its bucket and the global list.
Notes:	This routine assumes that the bucket's mutex is already locked
			if the hash table is multi-threaded.
****************************************************************************/
void F_HashTable::unlinkObject(
	F_HashObject *		pObject)
{
	FLMUINT		uiBucket = pObject->getHashBucket();

	// Is the bucket valid?

	f_assert( uiBucket < m_uiBuckets);

	// Unlink from the hash bucket

	if( pObject->m_pNextInBucket)
	{
		pObject->m_pNextInBucket->m_pPrevInBucket = pObject->m_pPrevInBucket;
	}

	if( pObject->m_pPrevInBucket)
	{
		pObject->m_pPrevInBucket->m_pNextInBucket = pObject->m_pNextInBucket;
	}
	else
	{
		m_ppHashTable[ uiBucket] = pObject->m_pNextInBucket;
	}

	pObject->m_pPrevInBucket = NULL;
	pObject->m_pNextInBucket = NULL;
	pObject->setHashBucket( F_INVALID_HASH_BUCKET);

	// Unlink from the global list

	if( pObject->m_pNextInGlobal)
	{
		pObject->m_pNextInGlobal->m_pPrevInGlobal = pObject->m_pPrevInGlobal;
	}
	else
	{
		m_pLRUObject = pObject->m_pPrevInGlobal;
	}

	if( pObject->m_pPrevInGlobal)
	{
		pObject->m_pPrevInGlobal->m_pNextInGlobal = pObject->m_pNextInGlobal;
	}
	else
	{
		m_pMRUObject = pObject->m_pNextInGlobal;
	}

	pObject->m_pPrevInGlobal = NULL;
	pObject->m_pNextInGlobal = NULL;
	pObject->m_uiTimeAdded = 0;

	f_assert( m_uiObjects);
	m_uiObjects--;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE f_initFileAsyncClientList( void)
{
	RCODE			rc = NE_FLM_OK;
	
	if( RC_BAD( rc = f_mutexCreate( &F_FileHdl::m_hAsyncListMutex)))
	{
		goto Exit;
	}
	
	F_FileHdl::m_pFirstAvailAsync = NULL;
	F_FileHdl::m_uiAvailAsyncCount = 0;
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void f_freeFileAsyncClientList( void)
{
	F_FileAsyncClient *		pAsyncClient;
	
	while( F_FileHdl::m_pFirstAvailAsync)
	{
		pAsyncClient = F_FileHdl::m_pFirstAvailAsync;
		F_FileHdl::m_pFirstAvailAsync = F_FileHdl::m_pFirstAvailAsync->m_pNext;
		pAsyncClient->m_pNext = NULL;
		pAsyncClient->Release( FALSE);
	}
	
	if( F_FileHdl::m_hAsyncListMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &F_FileHdl::m_hAsyncListMutex);
	}
	
	F_FileHdl::m_uiAvailAsyncCount = 0;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI f_stripCRLF( 
	const FLMBYTE *	pucSourceBuf,
	FLMUINT				uiSourceLength,
	F_DynaBuf *			pDestBuf)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT				uiLoop;
	FLMBYTE				ucSourceChar;
	
	for( uiLoop = 0; uiLoop < uiSourceLength; uiLoop++)
	{
		ucSourceChar = pucSourceBuf[ uiLoop];
		
		if( ucSourceChar != ASCII_CR && ucSourceChar != ASCII_NEWLINE)
		{
			if( RC_BAD( rc = pDestBuf->appendByte( ucSourceChar)))
			{
				goto Exit;
			}
			
			if( !ucSourceChar)
			{
				break;
			}
		}
	}
	
Exit:

	return( rc);
}	

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI f_base64Encode(
	const char *		pData,
	FLMUINT				uiDataLength,
	F_DynaBuf *			pBuffer)
{
	RCODE					rc = NE_FLM_OK;
	IF_PosIStream *	pBufferStream = NULL;
	IF_IStream *		pEncodedStream = NULL;
	
	if( RC_BAD( rc = FlmOpenBufferIStream( pData, 
		uiDataLength, &pBufferStream)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = FlmOpenBase64EncoderIStream( pBufferStream, 
		FALSE, &pEncodedStream)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = FlmReadFully( pEncodedStream, pBuffer)))
	{
		goto Exit;
	}
	
Exit:

	if( pEncodedStream)
	{
		pEncodedStream->Release();
	}

	if( pBufferStream)
	{
		pBufferStream->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMBOOL FTKAPI f_isNumber(
	const char *	pszStr,
	FLMBOOL *		pbNegative,
	FLMBOOL *		pbHex)
{
	FLMUINT			uiLen = f_strlen( pszStr);
	FLMBYTE			ucChar0 = f_toupper( pszStr[ 0]);
	FLMBYTE			ucChar1 = 0;
	FLMBOOL			bIsNumber = FALSE;
	FLMBOOL			bNegative = FALSE;
	FLMBOOL			bHex = FALSE;
	FLMUINT 			uiLoop;

	if( uiLen > 1)
	{
		ucChar1 = f_toupper( pszStr[ 1]);
	}

	if( (ucChar0 == 'X') || 
		 ((uiLen > 1) && (ucChar0 == '0') && (ucChar1 == 'X')))
	{
		const char * pszHexStart;
		
		if( ucChar0 == 'X')
		{
			pszHexStart = pszStr + 1;
		}
		else
		{
			pszHexStart = pszStr + 2;
		}
		
		uiLen = f_strlen( pszHexStart);
		
		for( uiLoop = 0; uiLoop < uiLen; uiLoop++)
		{
			FLMBYTE ucChar = f_toupper( pszHexStart[ uiLoop]);

			if( !((ucChar >= '0' && ucChar <= '9') ||
					(ucChar >= 'A' && ucChar <= 'F')))
			{
				goto Exit;
			}
		}
		
		bIsNumber = TRUE;
		bHex = TRUE;
	}
	else
	{
		for( FLMUINT uiStr = 0; uiStr < f_strlen( pszStr); uiStr++)
		{
			FLMBYTE	ucChar = pszStr[ uiStr];
			
			if( !uiStr && (ucChar == '+' || ucChar == '-'))
			{
				if( ucChar == '-')
				{
					bNegative = TRUE;
				}
			}
			else if( (ucChar < '0' || ucChar > '9') )
			{
				goto Exit;
			}
		}
		
		bIsNumber = TRUE;
	}
	
Exit:

	if( pbNegative)
	{
		*pbNegative = bNegative;
	}
	
	if( pbHex)
	{
		*pbHex = bHex;
	}

	return( bIsNumber);
}

/****************************************************************************
Desc:  
****************************************************************************/
RCODE FTKAPI F_Vector::setElementAt( 
	void * 			pData,
	FLMUINT 			uiIndex)
{
	RCODE				rc = NE_FLM_OK;
	
	if( !m_pElementArray)
	{		
		if( RC_BAD( rc = f_calloc( sizeof( void *) * F_VECTOR_START_AMOUNT,
			&m_pElementArray)))
		{
			goto Exit;
		}
		
		m_uiArraySize = F_VECTOR_START_AMOUNT;
	}

	if( uiIndex >= m_uiArraySize)
	{		
		if( RC_BAD( rc = f_recalloc(
			sizeof( void *) * m_uiArraySize * F_VECTOR_GROW_AMOUNT,
			&m_pElementArray)))
		{
			goto Exit;
		}
		
		m_uiArraySize *= F_VECTOR_GROW_AMOUNT;
	}

	m_pElementArray[ uiIndex] = pData;
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void * FTKAPI F_Vector::getElementAt( 
	FLMUINT		uiIndex)
{
	f_assert( uiIndex < m_uiArraySize);
	return( m_pElementArray[ uiIndex]);
}


/****************************************************************************
Desc:	Append a char (or the same char many times) to the string
****************************************************************************/
RCODE FTKAPI F_StringAcc::appendCHAR( 
	char				ucChar,
	FLMUINT			uiHowMany)
{
	RCODE				rc = NE_FLM_OK;
	
	if( uiHowMany == 1)
	{
		FLMBYTE 		szStr[ 2];
		
		szStr[ 0] = ucChar;
		szStr[ 1] = 0;
		
		rc = appendTEXT( (const FLMBYTE*)szStr);
	}
	else
	{
		FLMBYTE * 	pszStr;
		
		if( RC_BAD( rc = f_alloc( uiHowMany + 1, &pszStr)))
		{
			goto Exit;
		}
		
		f_memset( pszStr, ucChar, uiHowMany);
		
		pszStr[ uiHowMany] = 0;
		rc = appendTEXT( pszStr);
		f_free( &pszStr);
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	appending text to the accumulator safely.  all other methods in
		the class funnel through this one, as this one contains the logic
		for making sure storage requirements are met.
****************************************************************************/
RCODE FTKAPI F_StringAcc::appendTEXT( 
	const FLMBYTE * 	pszVal)
{	
	RCODE 				rc = NE_FLM_OK;
	FLMUINT 				uiIncomingStrLen;
	FLMUINT 				uiStrLen;

	if( !pszVal)
	{
		goto Exit;
	}
	else if( (uiIncomingStrLen = f_strlen( (const char *)pszVal)) == 0)
	{
		goto Exit;
	}
	
	// Compute total size we need to store the new total
	
	if( m_bQuickBufActive || m_pszVal)
	{
		uiStrLen = uiIncomingStrLen + m_uiValStrLen;
	}
	else
	{
		uiStrLen = uiIncomingStrLen;
	}

	// Just use small buffer if it's small enough
	
	if( uiStrLen < sizeof( m_szQuickBuf))
	{
		f_strcat( m_szQuickBuf, (const char *)pszVal);
		m_bQuickBufActive = TRUE;
	}
	else
	{
		// Ensure storage requirements are met (and then some)
		
		if( m_pszVal == NULL)
		{
			FLMUINT 		uiNewBytes = (uiStrLen + 1) * 4;
			
			if( RC_BAD( rc = f_alloc( (FLMUINT)(sizeof( FLMBYTE) * uiNewBytes),
				&m_pszVal)))
			{
				goto Exit;
			}
			
			m_uiBytesAllocatedForPszVal = uiNewBytes;
			m_pszVal[ 0] = 0;
		}
		else if( (m_uiBytesAllocatedForPszVal - 1) < uiStrLen)
		{
			FLMUINT 		uiNewBytes = (uiStrLen + 1) * 4;
			
			if( RC_BAD( rc = f_realloc( (FLMUINT)(sizeof( FLMBYTE) * uiNewBytes),
				&m_pszVal)))
			{
				goto Exit;
			}
			
			m_uiBytesAllocatedForPszVal = uiNewBytes;
		}

		// If transitioning from quick buf to heap buf, we need to
		// transfer over the quick buf contents and unset the flag
		
		if( m_bQuickBufActive)
		{
			m_bQuickBufActive = FALSE;
			f_strcpy( m_pszVal, m_szQuickBuf);
			
			// No need to zero out m_szQuickBuf because it will never
			// be used again, unless a clear() is issued, in which
			// case it will be zeroed out then.
		}		

		// Copy over the string
		
		f_strcat( m_pszVal, (const char *)pszVal);
	}
	
	m_uiValStrLen = uiStrLen;
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_StringAcc::printf(
	const char * pszFormatString,
	...)
{
	RCODE				rc = NE_FLM_OK;
	f_va_list		args;
	char *			pDestStr = NULL;
	FLMSIZET			iSize = 4096;

	if( RC_BAD( rc = f_alloc( iSize, &pDestStr)))
	{
		goto Exit;
	}

	f_va_start( args, pszFormatString);
	f_vsprintf( pDestStr, pszFormatString, &args);
	f_va_end( args);

	clear();
	
	if( RC_BAD( rc = appendTEXT( (FLMBYTE *)pDestStr)))
	{
		goto Exit;
	}

Exit:

	if( pDestStr)
	{
		f_free( &pDestStr);
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_StringAcc::appendf(
	const char * pszFormatString,
	...)
{
	RCODE				rc = NE_FLM_OK;
	f_va_list		args;
	char *			pDestStr = NULL;
	FLMSIZET			iSize = 4096;

	if( RC_BAD( rc = f_alloc( iSize, &pDestStr)))
	{
		goto Exit;
	}

	f_va_start( args, pszFormatString);
	f_vsprintf( pDestStr, pszFormatString, &args);
	f_va_end( args);

	if( RC_BAD( rc = appendTEXT( (FLMBYTE *)pDestStr)))
	{
		goto Exit;
	}

Exit:

	if( pDestStr)
	{
		f_free( &pDestStr);
	}
	
	return( rc);
}

