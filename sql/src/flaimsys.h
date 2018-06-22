//------------------------------------------------------------------------------
// Desc:	This is the master header file for FLAIM.  It is included by nearly
//			all of the source files.
// Tabs:	3
//
// Copyright (c) 1991-2007 Novell, Inc. All Rights Reserved.
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

#ifndef  FLAIMSYS_H
#define  FLAIMSYS_H

// Public includes

#ifdef HAVE_CONFIG_H
	#include <config.h>
#endif

#include "flaimsql.h"

// Collation bits

#define HAD_SUB_COLLATION				0x01		// Set if had sub-collating values-diacritics
#define HAD_LOWER_CASE					0x02		// Set if you hit a lowercase character
#define COMPOUND_MARKER					0x02		// Compound key marker between each piece
#define END_COMPOUND_MARKER			0x01		// Last of all compound markers - for post
#define NULL_KEY_MARKER					0x03
#define COLL_FIRST_SUBSTRING			0x03		// First substring marker
#define COLL_MARKER 						0x04		// Marks place of sub-collation
#define SC_LOWER							0x00		// Only lowercase characters exist
#define SC_MIXED							0x01		// Lower/uppercase flags follow in next byte
#define SC_UPPER							0x02		// Only upper characters exist
#define SC_SUB_COL						0x03		// Sub-collation follows (diacritics|extCh)
#define UNK_UNICODE_CODE				0xFFFE	// Used for collation
#define COLL_TRUNCATED					0x0C		// This key piece has been truncated from original
#define MAX_COL_OPCODE					COLL_TRUNCATED

#if defined( FLM_WIN)
	// Conversion from XXX to YYY, possible loss of data
	#pragma warning( disable : 4244) 

	// Local variable XXX may be used without having been initialized
	#pragma warning( disable : 4701)

	// Function XXX not inlined
	#pragma warning( disable : 4710) 
#endif
	
#if defined( FLM_WATCOM_NLM)

	// Disable "Warning! W549: col(XX) 'sizeof' operand contains
	// compiler generated information"
	
	#pragma warning 549 9
#endif
		
// Put all forward references here

class F_Database;
class F_Dict;
class F_Db;
class F_NameTable;
class F_Rfl;
class F_Btree;
class F_Query;
class F_DbRebuild;
class F_DbCheck;
class F_DbInfo;
class F_KeyCollector;
class FSIndexCursor;
class FSTableCursor;
class FDynSearchSet;
class F_CachedBlock;
class F_Row;
class F_GlobalCacheMgr;
class F_BlockCacheMgr;
class F_RowCacheMgr;
class F_BTreeIStream;
class F_BTreeIStreamPool;
class F_QueryResultSet;
class F_BTreeInfo;
class F_RebuildRowIStream;
class F_DbInfo;

// Some in-line functions included by other header files.

// Misc. global constants

#ifdef DEFINE_NUMBER_MAXIMUMS
	#define GV_EXTERN
#else
	#define GV_EXTERN		extern
#endif

GV_EXTERN FLMBOOL	gv_b32BitPlatform
#ifdef DEFINE_NUMBER_MAXIMUMS
	= (FLMBOOL)(sizeof( FLMUINT) == 4 ? TRUE : FALSE)
#endif
	;

GV_EXTERN FLMUINT	gv_uiMaxUInt32Val
#ifdef DEFINE_NUMBER_MAXIMUMS
		= (FLMUINT)0xFFFFFFFF
#endif
	;

GV_EXTERN FLMUINT	gv_uiMaxUIntVal
#ifdef DEFINE_NUMBER_MAXIMUMS
		= (FLMUINT)(~(FLMUINT)0)
#endif
		;

GV_EXTERN FLMUINT gv_uiMaxSignedIntVal
#ifdef DEFINE_NUMBER_MAXIMUMS
		= (FLMUINT)((((~(FLMUINT)0) << 1) >> 1))
#endif
	;

GV_EXTERN FLMUINT64 gv_ui64MaxSignedIntVal
#ifdef DEFINE_NUMBER_MAXIMUMS
	= (FLMUINT64)((((~(FLMUINT64)0) << 1) >> 1))
#endif
	;

/*****************************************************************************
Desc:
******************************************************************************/
FINLINE RCODE convertToUINT(
	FLMUINT64		ui64Num,
	FLMBOOL			bNeg,
	FLMUINT *		puiNum)
{
	if( bNeg)
	{
		return( RC_SET( NE_SFLM_CONV_NUM_UNDERFLOW));
	}

	if( gv_b32BitPlatform && ui64Num > 0xFFFFFFFF)
	{
		return( RC_SET( NE_SFLM_CONV_NUM_OVERFLOW));
	}

	*puiNum = (FLMUINT)ui64Num;
	return( NE_SFLM_OK);
}

/*****************************************************************************
Desc:
******************************************************************************/
FINLINE RCODE convertToUINT32(
	FLMUINT64		ui64Num,
	FLMBOOL			bNeg,
	FLMUINT32 *		pui32Num)
{
	if( bNeg)
	{
		return( RC_SET( NE_SFLM_CONV_NUM_UNDERFLOW));
	}

	if( ui64Num > 0xFFFFFFFF)
	{
		return( RC_SET( NE_SFLM_CONV_NUM_OVERFLOW));
	}

	*pui32Num = (FLMUINT32)ui64Num;
	return( NE_SFLM_OK);
}

/*****************************************************************************
Desc:
******************************************************************************/
FINLINE RCODE convertToUINT64(
	FLMUINT64		ui64Num,
	FLMBOOL			bNeg,
	FLMUINT64 *		pui64Num)
{
	if( bNeg)
	{
		return( RC_SET( NE_SFLM_CONV_NUM_UNDERFLOW));
	}

	*pui64Num = ui64Num;
	return( NE_SFLM_OK);
}

/*****************************************************************************
Desc:
******************************************************************************/
FINLINE RCODE convertToINT(
	FLMUINT64		ui64Num,
	FLMBOOL			bNeg,
	FLMINT *			piNum)
{
	if( bNeg)
	{
		if (ui64Num == (FLMUINT64)(FLM_MAX_INT) + 1)
		{
			*piNum = FLM_MIN_INT;
		}
		else if( ui64Num > (FLMUINT64)(FLM_MAX_INT) + 1)
		{
			return( RC_SET( NE_SFLM_CONV_NUM_UNDERFLOW));
		}
		else
		{
			*piNum = -((FLMINT)ui64Num);
		}
	}
	else
	{
		if( ui64Num > (FLMUINT64)FLM_MAX_INT)
		{
			return( RC_SET( NE_SFLM_CONV_NUM_OVERFLOW));
		}

		*piNum = (FLMINT)ui64Num;
	}

	return( NE_SFLM_OK);
}

/*****************************************************************************
Desc:
******************************************************************************/
FINLINE RCODE convertToINT32(
	FLMUINT64		ui64Num,
	FLMBOOL			bNeg,
	FLMINT32 *		pi32Num)
{
	if( bNeg)
	{
		if (ui64Num == (FLMUINT64)(FLM_MAX_INT32) + 1)
		{
			*pi32Num = FLM_MIN_INT32;
		}
		else if( ui64Num > (FLMUINT64)(FLM_MAX_INT32) + 1)
		{
			return( RC_SET( NE_SFLM_CONV_NUM_UNDERFLOW));
		}
		else
		{
			*pi32Num = -((FLMINT32)ui64Num);
		}
	}
	else
	{
		if( ui64Num > (FLMUINT64)FLM_MAX_INT32)
		{
			return( RC_SET( NE_SFLM_CONV_NUM_OVERFLOW));
		}

		*pi32Num = (FLMINT32)ui64Num;
	}

	return( NE_SFLM_OK);
}

/*****************************************************************************
Desc:
******************************************************************************/
FINLINE RCODE convertToINT64(
	FLMUINT64		ui64Num,
	FLMBOOL			bNeg,
	FLMINT64 *		pi64Num)
{
	if( bNeg)
	{
		if( ui64Num > gv_ui64MaxSignedIntVal + 1)
		{
			return( RC_SET( NE_SFLM_CONV_NUM_UNDERFLOW));
		}

		*pi64Num = -(FLMINT64)ui64Num;
	}
	else
	{
		if( ui64Num > gv_ui64MaxSignedIntVal)
		{
			return( RC_SET( NE_SFLM_CONV_NUM_OVERFLOW));
		}

		*pi64Num = (FLMINT64)ui64Num;
	}

	return( NE_SFLM_OK);
}

// NOTE: ENCRYPT_MIN_CHUNK_SIZE should always be a power of 2
// getEncLen supposes that it is.

#define ENCRYPT_MIN_CHUNK_SIZE			16
#define ENCRYPT_BOUNDARY_MASK				(~(ENCRYPT_MIN_CHUNK_SIZE - 1))

/*****************************************************************************
Desc: Calculate the number of bytes extra there are beyond the closest
		encryption boundary.
******************************************************************************/
FINLINE FLMUINT extraEncBytes(
	FLMUINT	uiDataLen)
{
	// This works if ENCRYPT_MIN_CHUNK_SIZE is a power of 2
	// Otherwise, it needs to be changed to uiDataLen % ENCRYPT_MIN_CHUNK_SIZE
	
	return( uiDataLen & (ENCRYPT_MIN_CHUNK_SIZE - 1));
}

/*****************************************************************************
Desc: Calculate the encryption length for a piece of data.
******************************************************************************/
FINLINE FLMUINT getEncLen(
	FLMUINT		uiDataLen)
{
	return( extraEncBytes( uiDataLen)
				? ((uiDataLen + ENCRYPT_MIN_CHUNK_SIZE) & ENCRYPT_BOUNDARY_MASK)
				: uiDataLen);
}

/****************************************************************************
Storage conversion functions.
****************************************************************************/

#define FLM_MAX_NUM_BUF_SIZE		9

RCODE flmStorage2Number(
	eDataType				eDataTyp,
	FLMUINT					uiBufLen,
	const FLMBYTE *		pucBuf,
	FLMUINT *				puiNum,
	FLMINT *					piNum);

RCODE flmStorage2Number64(
	eDataType				eDataTyp,
	FLMUINT					uiBufLen,
	const FLMBYTE *		pucBuf,
	FLMUINT64 *				pui64Num,
	FLMINT64 *				pi64Num);

RCODE flmNumber64ToStorage(
	FLMUINT64				ui64Num,
	FLMUINT *				puiBufLen,
	FLMBYTE *				pucBuf,
	FLMBOOL					bNegative,
	FLMBOOL					bCollation);

RCODE FlmUINT2Storage(
	FLMUINT					uiNum,
	FLMUINT *				puiBufLen,
	FLMBYTE *				pucBuf);

RCODE FlmINT2Storage(
	FLMINT					iNum,
	FLMUINT *				puiBufLen,
	FLMBYTE *				pucBuf);

RCODE	flmUTF8ToStorage(
	const FLMBYTE *		pucUTF8,
	FLMUINT					uiBytesInBuffer,
	FLMBYTE *				pucBuf,
	FLMUINT *				puiBufLength);

RCODE flmGetCharCountFromStorageBuf(
	const FLMBYTE **		ppucBuf, 
	FLMUINT					uiBufSize,
	FLMUINT *				puiNumChars,
	FLMUINT *				puiSenLen = NULL);

RCODE flmStorage2UTF8(
	FLMUINT					uiType,
	FLMUINT					uiBufLength,
	const FLMBYTE *		pucBuffer,
	FLMUINT *				puiOutBufLen,
	FLMBYTE *				pucOutBuf);

RCODE flmStorage2UTF8(
	FLMUINT					uiType,
	FLMUINT					uiBufLength,
	const FLMBYTE *		pucBuffer,
	F_DynaBuf *				pBuffer);

RCODE flmStorage2Unicode(
	FLMUINT					uiType,
	FLMUINT					uiBufLength,
	const FLMBYTE *		pucBuffer,
	FLMUINT *				puiOutBufLen,
	void *					pOutBuf);

RCODE flmStorage2Unicode(
	FLMUINT					uiType,
	FLMUINT					uiStorageLength,
	const FLMBYTE *		pucStorageBuffer,
	F_DynaBuf *				pBuffer);

RCODE	flmUnicode2Storage(
	const FLMUNICODE *	puzStr,
	FLMUINT					uiStrLen,
	FLMBYTE *				pucBuf,
	FLMUINT *				puiBufLength,
	FLMUINT *				puiCharCount);

RCODE flmStorageNum2StorageText(
	const FLMBYTE *		pucNum,
	FLMUINT					uiNumLen,
	FLMBYTE *				pucBuffer,
	FLMUINT *				puiBufLen);

/****************************************************************************
Desc: 	Returns the size of buffer needed to hold the unicode string in 
			FLAIM's storage format.
****************************************************************************/
FINLINE RCODE FlmGetUnicodeStorageLength(
	FLMUNICODE *	puzStr,
	FLMUINT *		puiByteCount)
{
	FLMUINT	uiByteCount;
	RCODE		rc;

	if( RC_BAD( rc = flmUnicode2Storage( puzStr, 0, NULL, 
		&uiByteCount, NULL)))
	{
		return( rc);
	}

	*puiByteCount = uiByteCount + sizeof( FLMUNICODE);
	return( NE_SFLM_OK);
}

/****************************************************************************
Desc: 	Copies and formats a Unicode string into FLAIM's storage format.
			The Unicode string must be in little-endian byte order.
			Unicode values that are not represented as WordPerfect 6.x characters
			are preserved as non-WP characters.
****************************************************************************/
FINLINE RCODE FlmUnicode2Storage(
	FLMUNICODE *	puzStr,
	FLMUINT *		puiBufLength,	// [IN] size of pBuf,
											// [OUT] amount of pBuf used to hold puzStr
	FLMBYTE *		pBuf)
{
	return( flmUnicode2Storage( puzStr, 0, pBuf, puiBufLength, NULL));
}

/****************************************************************************
Desc: 	Converts from FLAIM's internal storage format to a Unicode string
****************************************************************************/
FINLINE RCODE FlmStorage2Unicode(
	FLMUINT			uiType,
	FLMUINT			uiBufLength,
	FLMBYTE *		pBuffer,
	FLMUINT *		puiOutBufLen,
	FLMUNICODE *	puzOutBuf)
{
	return( flmStorage2Unicode( uiType, uiBufLength, pBuffer,
		puiOutBufLen, puzOutBuf));
}

/****************************************************************************
Desc: 	Convert storage text into a null-terminated UTF-8 string
****************************************************************************/
FINLINE RCODE FlmStorage2UTF8(
	FLMUINT		uiType,
	FLMUINT		uiBufLength,
	FLMBYTE *	pBuffer, 
	FLMUINT *	puiOutBufLenRV,
			// [IN] Specified the number of bytes available in buffer.
			// [OUT] Returns the number of bytes of UTF-8 text.  This value
			// does not include a terminating NULL byte in the buffer.
	FLMBYTE *	pOutBuffer)
			// [OUT] The buffer that will hold the output UTF-8 text.
			// If this value is NULL then only bufLenRV will computed so that
			// the caller may know how many bytes to allocate for a buffer.
{
	return( flmStorage2UTF8( uiType, uiBufLength, 
		pBuffer, puiOutBufLenRV, pOutBuffer));
}

// Internal includes

#include "fcollate.h"
#include "fdict.h"
#include "fstructs.h"
#include "fcache.h"
#include "flmstat.h"
#include "fbtrset.h"
#include "fcollate.h"
#include "f_btree.h"
#include "f_btpool.h"
#include "rfl.h"
#include "filesys.h"
#include "flog.h"
#include "f_nici.h"
#include "sqlstatement.h"

/****************************************************************************
Desc:
****************************************************************************/
typedef struct FlmVectorElementTag
{
	FLMUINT		uiColumnNum;
	FLMUINT		uiFlags;
#define				VECT_SLOT_HAS_DATA			0x01
#define				VECT_SLOT_HAS_ID				0x02
#define				VECT_SLOT_RIGHT_TRUNCATED	0x04
#define				VECT_SLOT_LEFT_TRUNCATED	0x08
#define				VECT_SLOT_HAS_COLUMN_NUM		0x10
#define				VECT_SLOT_IS_ATTR				0x20
#define				VECT_SLOT_IS_DATA				0x40
	eDataType	eDataTyp;
	FLMUINT		uiDataLength;
	FLMUINT		uiDataOffset;
} F_VECTOR_ELEMENT;

/*****************************************************************************
Desc:	Used to build keys and data components
*****************************************************************************/
class F_DataVector : public F_Object
{
public:

	// Constructor/Destructor

	F_DataVector();
	virtual ~F_DataVector();

	// Setter methods

	FINLINE void setRowID(
		FLMUINT64	ui64RowId)
	{
		m_ui64RowId = ui64RowId;
	}

	RCODE setColumnNum(
		FLMUINT		uiElementNumber,
		FLMUINT		uiColumnNum,
		FLMBOOL		bIsData);

	RCODE setINT(
		FLMUINT	uiElementNumber,
		FLMINT	iNum);

	RCODE setINT64(
		FLMUINT		uiElementNumber,
		FLMINT64		i64Num);

	RCODE setUINT(
		FLMUINT	uiElementNumber,
		FLMUINT	uiNum);

	RCODE setUINT64(
		FLMUINT		uiElementNumber,
		FLMUINT64	ui64Num);

	RCODE setUnicode(
		FLMUINT					uiElementNumber,
		const FLMUNICODE *	puzUnicode);

	RCODE setUTF8(
		FLMUINT				uiElementNumber,
		const FLMBYTE *	pszUtf8,
		FLMUINT				uiBytesInBuffer = 0);

	FINLINE RCODE setBinary(
		FLMUINT				uiElementNumber,
		const void *		pvBinary,
		FLMUINT				uiBinaryLen)
	{
		return( storeValue( uiElementNumber,
							SFLM_BINARY_TYPE, (FLMBYTE *)pvBinary, uiBinaryLen));
	}

	FINLINE void setRightTruncated(
		FLMUINT	uiElementNumber)
	{
		F_VECTOR_ELEMENT *	pVector;

		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_DATA)) != NULL)
		{
			setRightTruncated( pVector);
		}
		else
		{
			// Need to set some data value before setting
			// truncated.
			flmAssert( 0);
		}
	}

	FINLINE void setLeftTruncated(
		FLMUINT	uiElementNumber)
	{
		F_VECTOR_ELEMENT *	pVector;

		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_DATA)) != NULL)
		{
			setLeftTruncated( pVector);
		}
		else
		{
			// Need to set some data value before setting
			// truncated.
			flmAssert( 0);
		}
	}

	FINLINE void clearRightTruncated(
		FLMUINT	uiElementNumber)
	{
		F_VECTOR_ELEMENT *	pVector;

		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_DATA)) != NULL)
		{
			clearRightTruncated( pVector);
		}
		else
		{
			// Need to set some data value before clearing
			// truncated.
			flmAssert( 0);
		}
	}

	FINLINE void clearLeftTruncated(
		FLMUINT	uiElementNumber)
	{
		F_VECTOR_ELEMENT *	pVector;

		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_DATA)) != NULL)
		{
			clearLeftTruncated( pVector);
		}
		else
		{
			// Need to set some data value before clearing
			// truncated.
			flmAssert( 0);
		}
	}

	FINLINE FLMBOOL isRightTruncated(
		FLMUINT	uiElementNumber)
	{
		F_VECTOR_ELEMENT *	pVector;

		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_DATA)) != NULL)
		{
			return( isRightTruncated( pVector));
		}

		return( FALSE);
	}

	FINLINE FLMBOOL isLeftTruncated(
		FLMUINT	uiElementNumber)
	{
		F_VECTOR_ELEMENT *	pVector;

		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_DATA)) != NULL)
		{
			return( isLeftTruncated( pVector));
		}

		return( FALSE);
	}

	// Getter methods

	FINLINE FLMUINT64 getRowId( void)
	{
		return( m_ui64RowId);
	}

	FINLINE FLMUINT getColumnNum(
		FLMUINT		uiElementNumber)
	{
		F_VECTOR_ELEMENT *	pVector;

		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_COLUMN_NUM)) == NULL)
		{
			return( 0);
		}
		else
		{
			return( pVector->uiColumnNum);
		}
	}

	FINLINE FLMBOOL isDataComponent(
		FLMUINT		uiElementNumber)
	{
		F_VECTOR_ELEMENT *	pVector;

		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_COLUMN_NUM)) == NULL)
		{
			return( FALSE);
		}
		else
		{
			return( (pVector->uiFlags & VECT_SLOT_IS_DATA) ? TRUE : FALSE);
		}
	}

	FINLINE FLMBOOL isKeyComponent(
		FLMUINT		uiElementNumber)
	{
		F_VECTOR_ELEMENT *	pVector;

		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_COLUMN_NUM)) == NULL)
		{
			return( FALSE);
		}
		else
		{
			return( (pVector->uiFlags & VECT_SLOT_IS_DATA) ? FALSE : TRUE);
		}
	}

	FLMUINT getDataLength(
		FLMUINT	uiElementNumber)
	{
		F_VECTOR_ELEMENT *	pVector;

		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_DATA)) == NULL)
		{
			return( 0);
		}
		else
		{
			return( pVector->uiDataLength);
		}
	}

	eDataType getDataType(
		FLMUINT	uiElementNumber)
	{
		F_VECTOR_ELEMENT *	pVector;

		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_DATA)) == NULL)
		{
			return( SFLM_UNKNOWN_TYPE);
		}
		else
		{
			return( pVector->eDataTyp);
		}
	}

	RCODE getUTF8Ptr(
		FLMUINT				uiElementNumber,
		const FLMBYTE **	ppszUTF8,
		FLMUINT *			puiBufLen);

	FINLINE RCODE getINT(
		FLMUINT	uiElementNumber,
		FLMINT *	piNum)
	{
		F_VECTOR_ELEMENT *	pVector;

		return( (RCODE)((pVector = getVector( uiElementNumber,
												VECT_SLOT_HAS_DATA)) != NULL
							 ? flmStorage2Number( pVector->eDataTyp,
									pVector->uiDataLength,
									(FLMBYTE *)getDataPtr( pVector),
									NULL, piNum)
							 : RC_SET( NE_SFLM_NOT_FOUND)));
	}

	FINLINE RCODE getINT64(
		FLMUINT		uiElementNumber,
		FLMINT64 *	pi64Num)
	{
		F_VECTOR_ELEMENT *	pVector;

		return( (RCODE)((pVector = getVector( uiElementNumber,
												VECT_SLOT_HAS_DATA)) != NULL
							 ? flmStorage2Number64( pVector->eDataTyp,
									pVector->uiDataLength,
									(FLMBYTE *)getDataPtr( pVector),
									NULL, pi64Num)
							 : RC_SET( NE_SFLM_NOT_FOUND)));
	}

	FINLINE RCODE getUINT(
		FLMUINT		uiElementNumber,
		FLMUINT *	puiNum)
	{
		F_VECTOR_ELEMENT *	pVector;

		return( (RCODE)((pVector = getVector( uiElementNumber,
												VECT_SLOT_HAS_DATA)) != NULL
							 ? flmStorage2Number( pVector->eDataTyp,
									pVector->uiDataLength,
									(FLMBYTE *)getDataPtr( pVector),
									puiNum, NULL)
							 : RC_SET( NE_SFLM_NOT_FOUND)));
	}

	FINLINE RCODE getUINT64(
		FLMUINT		uiElementNumber,
		FLMUINT64 *	pui64Num)
	{
		F_VECTOR_ELEMENT *	pVector;

		return( (RCODE)((pVector = getVector( uiElementNumber,
												VECT_SLOT_HAS_DATA)) != NULL
							 ? flmStorage2Number64( pVector->eDataTyp,
									pVector->uiDataLength,
									(FLMBYTE *)getDataPtr( pVector),
									pui64Num, NULL)
							 : RC_SET( NE_SFLM_NOT_FOUND)));
	}

	RCODE getUnicode(
		FLMUINT			uiElementNumber,
		FLMUNICODE **	ppuzUnicode);

	FINLINE RCODE getUnicode(
		FLMUINT			uiElementNumber,
		F_DynaBuf *		pBuffer)
	{
		F_VECTOR_ELEMENT *	pVector;

		return( (RCODE)((pVector = getVector( uiElementNumber,
												VECT_SLOT_HAS_DATA)) != NULL
							 ? flmStorage2Unicode( pVector->eDataTyp,
									pVector->uiDataLength,
									(FLMBYTE *)getDataPtr( pVector),
									pBuffer)
							 : RC_SET( NE_SFLM_NOT_FOUND)));
	}
		
	FINLINE RCODE getUnicode(
		FLMUINT			uiElementNumber,
		FLMUNICODE *	puzUnicode,
		FLMUINT *		puiBufLen)
	{
		F_VECTOR_ELEMENT *	pVector;

		return( (RCODE)((pVector = getVector( uiElementNumber,
												VECT_SLOT_HAS_DATA)) != NULL
							 ? flmStorage2Unicode( pVector->eDataTyp,
									pVector->uiDataLength,
									(FLMBYTE *)getDataPtr( pVector),
									puiBufLen, puzUnicode)
							 : RC_SET( NE_SFLM_NOT_FOUND)));
	}

	FINLINE RCODE getUTF8(
		FLMUINT			uiElementNumber,
		FLMBYTE *		pszUTF8,
		FLMUINT *		puiBufLen)
	{
		F_VECTOR_ELEMENT *	pVector;

		return( (RCODE)((pVector = getVector( uiElementNumber,
												VECT_SLOT_HAS_DATA)) != NULL
							 ? flmStorage2UTF8( pVector->eDataTyp,
									pVector->uiDataLength,
									(FLMBYTE *)getDataPtr( pVector),
									puiBufLen, pszUTF8)
							 : RC_SET( NE_SFLM_NOT_FOUND)));
	}

	FINLINE RCODE getUTF8(
		FLMUINT			uiElementNumber,
		F_DynaBuf *		pBuffer)
	{
		F_VECTOR_ELEMENT *	pVector;

		return( (RCODE)((pVector = getVector( uiElementNumber,
												VECT_SLOT_HAS_DATA)) != NULL
							 ? flmStorage2UTF8( pVector->eDataTyp,
									pVector->uiDataLength,
									(FLMBYTE *)getDataPtr( pVector),
									pBuffer)
							 : RC_SET( NE_SFLM_NOT_FOUND)));
	}
		
	FINLINE RCODE getBinary(
		FLMUINT			uiElementNumber,
		void *			pvBuffer,
		FLMUINT *		puiLength)
	{
		F_VECTOR_ELEMENT *	pVector;

		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_DATA)) != NULL)
		{
			*puiLength = f_min( (*puiLength), pVector->uiDataLength);
			if (pvBuffer && *puiLength)
			{
				f_memcpy( pvBuffer, getDataPtr( pVector), *puiLength);
			}
			
			return( NE_SFLM_OK);
		}
		else
		{
			*puiLength = 0;
		}

		return( RC_SET( NE_SFLM_NOT_FOUND));
	}

	FINLINE RCODE getBinary(
		FLMUINT				uiElementNumber,
		F_DynaBuf *			pBuffer)
	{
		F_VECTOR_ELEMENT *	pVector;

		pBuffer->truncateData( 0);
		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_DATA)) != NULL)
		{
			return( pBuffer->appendData( getDataPtr( pVector),
													pVector->uiDataLength));
		}

		return( RC_SET( NE_SFLM_NOT_FOUND));
	}

	RCODE outputKey(
		F_Db *			pDb,
		FLMUINT			uiIndexNum,
		FLMUINT			uiMatchFlags,
		FLMBYTE *		pucKeyBuf,
		FLMUINT			uiKeyBufSize,
		FLMUINT *		puiKeyLen,
		FLMUINT			uiSearchKeyFlag);
		
	RCODE outputData(
		F_Db *				pDb,
		FLMUINT				uiIndexNum,
		FLMBYTE *			pucDataBuf,
		FLMUINT				uiDataBufSize,
		FLMUINT *			puiDataLen);

	RCODE inputKey(
		F_Db *				pDb,
		FLMUINT				uiIndexNum,
		const FLMBYTE *	pucKey,
		FLMUINT				uiKeyLen);
		
	RCODE inputData(
		F_Db *				pDb,
		FLMUINT				uiIndexNum,
		const FLMBYTE *	pucData,
		FLMUINT				uiDataLen);

	// Miscellaneous methods

	void reset( void);

	FINLINE const void * getDataPtr(
		FLMUINT	uiElementNumber)
	{
		return( getDataPtr( getVector( uiElementNumber, VECT_SLOT_HAS_DATA)));
	}
	
	RCODE copyVector(
		F_DataVector *	pSrcVector);

private:

	RCODE allocVectorArray(
		FLMUINT	uiElementNumber);

	RCODE storeValue(
		FLMINT				uiElementNumber,
		eDataType			eDataTyp,
		const FLMBYTE *	pucData,
		FLMUINT				uiDataLen,
		FLMBYTE **			ppucDataPtr = NULL);

	FINLINE F_VECTOR_ELEMENT * getVector(
		FLMUINT	uiElementNumber,
		FLMUINT	uiTestFlags)
	{
		F_VECTOR_ELEMENT *	pVector;

		if (uiElementNumber >= m_uiNumElements)
		{
			return( NULL);
		}
		pVector = &m_pVectorElements [uiElementNumber];
		if (!(pVector->uiFlags & uiTestFlags))
		{
			return( NULL);
		}
		else
		{
			return( pVector);
		}
	}

	FINLINE FLMBOOL isRightTruncated(
		F_VECTOR_ELEMENT *	pVector)
	{
		return( (pVector->uiFlags & VECT_SLOT_RIGHT_TRUNCATED)
					? TRUE
					: FALSE);
	}

	FINLINE void setRightTruncated(
		F_VECTOR_ELEMENT *	pVector)
	{
		pVector->uiFlags |= VECT_SLOT_RIGHT_TRUNCATED;
	}

	FINLINE void clearRightTruncated(
		F_VECTOR_ELEMENT *	pVector)
	{
		pVector->uiFlags &= (~(VECT_SLOT_RIGHT_TRUNCATED));
	}

	FINLINE FLMBOOL isLeftTruncated(
		F_VECTOR_ELEMENT *	pVector)
	{
		return( (pVector->uiFlags & VECT_SLOT_LEFT_TRUNCATED)
					? TRUE
					: FALSE);
	}

	FINLINE void setLeftTruncated(
		F_VECTOR_ELEMENT *	pVector)
	{
		pVector->uiFlags |= VECT_SLOT_LEFT_TRUNCATED;
	}

	FINLINE void clearLeftTruncated(
		F_VECTOR_ELEMENT *	pVector)
	{
		pVector->uiFlags &= (~(VECT_SLOT_LEFT_TRUNCATED));
	}

	FINLINE void * getDataPtr(
		F_VECTOR_ELEMENT *	pVector)
	{
		if (!pVector || !pVector->uiDataLength)
		{
			return( NULL);
		}
		else if (pVector->uiDataLength <= sizeof( FLMUINT))
		{
			return( (void *)&pVector->uiDataOffset);
		}
		else
		{
			return( (void *)(m_pucDataBuf + pVector->uiDataOffset));
		}
	}

#define MIN_VECTOR_ELEMENTS	6
	F_VECTOR_ELEMENT		m_VectorArray [MIN_VECTOR_ELEMENTS];
	F_VECTOR_ELEMENT *	m_pVectorElements;	// Pointer to vector elements
	FLMUINT					m_uiVectorArraySize;	// Size of vector array
	FLMUINT					m_uiNumElements;		// Number of elements actually
															// populated in the array.

	FLMBYTE					m_ucIntDataBuf[ 32];	// Internal data buffer
	FLMBYTE *				m_pucDataBuf;			// Values stored here
	FLMUINT					m_uiDataBufLength;	// Bytes of data allocated
	FLMUINT					m_uiDataBufOffset;	// Current offset into allocated
															// data buffer.
	FLMUINT64				m_ui64RowId;			// Row ID;

friend class F_Db;
friend class FSIndexCursor;
friend class FSTableCursor;
friend class F_QueryResultSet;
};

// Flags for the m_uiFlags member of the F_Db object

#define FDB_UPDATED_DICTIONARY	0x0001
												// Flag indicating whether the file's
												// local dictionary was updated
												// during the transaction.
#define FDB_DO_TRUNCATE				0x0002
												// Truncate log extents at the end
												// of a transaction.
#define FDB_HAS_FILE_LOCK			0x0004
												//	FDB has a file lock.
#define FDB_FILE_LOCK_SHARED		0x0008
												// File lock is shared.  Update
												// transactions are not allowed when
												// the lock is shared.
#define FDB_FILE_LOCK_IMPLICIT	0x0010
												// File lock is implicit - means file
												// lock was obtained when the update
												// transaction began and cannot be
												// released by a call to FlmDbUnlock.
#define FDB_DONT_KILL_TRANS		0x0020
												// Do not attempt to kill an active
												// read transaction on this database
												// handle.  This is used by FlmDbBackup.
#define FDB_INTERNAL_OPEN			0x0040
												// FDB is an internal one used by a
												// background thread.
#define FDB_DONT_POISON_CACHE		0x0080
												// If blocks are read from disk during
												// a transaction, release them at the LRU
												// end of the cache chain.
#define FDB_UPGRADING				0x0100
												// Database is being upgraded.
#define FDB_REPLAYING_RFL			0x0200
												// Database is being recovered
#define FDB_REPLAYING_COMMIT		0x0400
												// During replay of the RFL, this
												// is an actual call to FlmDbTransCommit.
#define FDB_BACKGROUND_INDEXING	0x0800
												// FDB is being used by a background indexing
												// thread
#define FDB_HAS_WRITE_LOCK			0x1000
												// FDB has the write lock
#define FDB_REBUILDING_DATABASE	0x2000
												// Database is being rebuilt
#define FDB_SWEEP_SCHEDULED		0x4000
												// Sweep operation scheduled due to a
												// dictionary change during the transaction

/*****************************************************************************
Desc:	Class for performing database backup.
*****************************************************************************/
class F_Backup : public F_Object
{
public:

	F_Backup();
	virtual ~F_Backup();

	FINLINE FLMUINT64 getBackupTransId( void)
	{
		return( m_ui64TransId);
	}

	FINLINE FLMUINT64 getLastBackupTransId( void)
	{
		return( m_ui64LastBackupTransId);
	}

	RCODE backup(
		const char *			pszBackupPath,
		const char *			pszPassword,
		IF_BackupClient *		pClient,
		IF_BackupStatus *		pStatus,
		FLMUINT *				puiIncSeqNum);

	RCODE endBackup( void);

private:

	void reset( void);

	F_Db *			m_pDb;
	eDbTransType	m_eTransType;
	FLMUINT64		m_ui64TransId;
	FLMUINT64		m_ui64LastBackupTransId;
	FLMUINT			m_uiDbVersion;
	FLMUINT			m_uiBlkChgSinceLastBackup;
	FLMBOOL			m_bTransStarted;
	FLMUINT			m_uiBlockSize;
	FLMUINT			m_uiLogicalEOF;
	FLMUINT			m_uiFirstReqRfl;
	FLMUINT			m_uiIncSeqNum;
	FLMBOOL			m_bCompletedBackup;
	eDbBackupType	m_eBackupType;
	RCODE				m_backupRc;
	FLMBYTE			m_ucNextIncSerialNum[ SFLM_SERIAL_NUM_SIZE];
	char				m_szDbPath[ F_PATH_MAX_SIZE];
	SFLM_DB_HDR		m_dbHdr;

friend class F_Db;
};


/*****************************************************************************
Desc:		An implementation of IF_Backup_Client that backs up to the
			local hard disk.
*****************************************************************************/
class F_DefaultBackupClient : public IF_BackupClient
{
public:

	F_DefaultBackupClient(
		const char *	pszBackupPath);

	virtual ~F_DefaultBackupClient();

	RCODE WriteData(
		const void *	pvBuffer,
		FLMUINT			uiBytesToWrite);

	virtual FLMINT SQFAPI getRefCount( void)
	{
		return( IF_BackupClient::getRefCount());
	}

	virtual FLMINT SQFAPI AddRef( void)
	{
		return( IF_BackupClient::AddRef());
	}

	virtual FLMINT SQFAPI Release( void)
	{
		return( IF_BackupClient::Release());
	}

private:

	char						m_szPath[ F_PATH_MAX_SIZE];
	IF_MultiFileHdl *		m_pMultiFileHdl;
	FLMUINT64				m_ui64Offset;
	RCODE						m_rc;
};

/*****************************************************************************
Desc:		The F_FSRestore class is used to read backup and RFL files from 
			a disk file system.
*****************************************************************************/
class F_FSRestore : public IF_RestoreClient
{
public:

	virtual ~F_FSRestore();
	F_FSRestore();

	RCODE setup(
		const char *	pszDbPath,
		const char *	pszBackupSetPath,
		const char *	pszRflDir);

	RCODE openBackupSet( void);

	RCODE openIncFile(
		FLMUINT			uiFileNum);

	RCODE openRflFile(
		FLMUINT			uiFileNum);

	RCODE read(
		FLMUINT			uiLength,
		void *			pvBuffer,
		FLMUINT *		puiBytesRead);

	RCODE close( void);

	RCODE abortFile( void);

	virtual FLMINT SQFAPI getRefCount( void)
	{
		return( IF_RestoreClient::getRefCount());
	}

	virtual FLMINT SQFAPI AddRef( void)
	{
		return( IF_RestoreClient::AddRef());
	}

	virtual FLMINT SQFAPI Release( void)
	{
		return( IF_RestoreClient::Release());
	}

protected:

	IF_FileHdl *			m_pFileHdl;
	IF_MultiFileHdl *		m_pMultiFileHdl;
	FLMUINT64				m_ui64Offset;
	FLMUINT					m_uiDbVersion;
	char						m_szDbPath[ F_PATH_MAX_SIZE];
	char						m_szBackupSetPath[ F_PATH_MAX_SIZE];
	char						m_szRflDir[ F_PATH_MAX_SIZE];
	FLMBOOL					m_bSetupCalled;
	FLMBOOL					m_bOpen;
};

/*****************************************************************************
Desc:		Default implementation of a restore status object than can
			be inherited by a user implementation.
*****************************************************************************/
class F_DefaultRestoreStatus : public IF_RestoreStatus
{
public:

	F_DefaultRestoreStatus()
	{
	}

	RCODE reportProgress(
		eRestoreAction *	peAction,
		FLMUINT64,			// ui64BytesToDo,
		FLMUINT64)			// ui64BytesDone
	{
		*peAction = SFLM_RESTORE_ACTION_CONTINUE;
		return( NE_SFLM_OK);
	}

	RCODE reportError(
		eRestoreAction *	peAction,
		RCODE)				// rcErr
	{
		*peAction = SFLM_RESTORE_ACTION_CONTINUE;
		return( NE_SFLM_OK);
	}

	RCODE reportBeginTrans(
		eRestoreAction *	peAction,
		FLMUINT64)			// ui64TransId
	{
		*peAction = SFLM_RESTORE_ACTION_CONTINUE;
		return( NE_SFLM_OK);
	}

	RCODE reportCommitTrans(
		eRestoreAction *	peAction,
		FLMUINT64)			// ui64TransId
	{
		*peAction = SFLM_RESTORE_ACTION_CONTINUE;
		return( NE_SFLM_OK);
	}

	RCODE reportAbortTrans(
		eRestoreAction *	peAction,
		FLMUINT64)			// ui64TransId
	{
		*peAction = SFLM_RESTORE_ACTION_CONTINUE;
		return( NE_SFLM_OK);
	}

	RCODE reportRemoveData(
		eRestoreAction *	peAction,
		FLMUINT64,			// ui64TransId,
		FLMUINT,				// uiLfNum,
		FLMUINT,				// uiKeyLen,
		FLMBYTE *)			// pucKey
	{
		*peAction = SFLM_RESTORE_ACTION_CONTINUE;
		return( NE_SFLM_OK);
	}

	RCODE reportInsertData(
		eRestoreAction *	peAction,
		FLMUINT64,			// ui64TransId,
		FLMUINT,				// uiLfNum,
		FLMUINT,				// uiKeyLen,
		FLMBYTE *)			// pucKey
	{
		*peAction = SFLM_RESTORE_ACTION_CONTINUE;
		return( NE_SFLM_OK);
	}

	RCODE reportReplaceData(
		eRestoreAction *	peAction,
		FLMUINT64,			// ui64TransId,
		FLMUINT,				// uiLfNum,
		FLMUINT,				// uiKeyLen,
		FLMBYTE *)			// pucKey
	{
		*peAction = SFLM_RESTORE_ACTION_CONTINUE;
		return( NE_SFLM_OK);
	}

	RCODE reportLFileCreate(
		eRestoreAction *	peAction,
		FLMUINT64,			// ui64TransId,
		FLMUINT)				// uiLfNum
	{
		*peAction = SFLM_RESTORE_ACTION_CONTINUE;
		return( NE_SFLM_OK);
	}

	RCODE reportLFileUpdate(
		eRestoreAction *	peAction,
		FLMUINT64,			// ui64TransId,
		FLMUINT,				// uiLfNum,
		FLMUINT,				// uiRootBlk,
		FLMUINT64,			// ui64NextNodeId,
		FLMUINT64,			// ui64FirstDocId,
		FLMUINT64			// ui64LastDocId
		)
	{
		*peAction = SFLM_RESTORE_ACTION_CONTINUE;
		return( NE_SFLM_OK);
	}

	RCODE reportUpdateDict(
		eRestoreAction *	peAction,
		FLMUINT64,			// ui64TransId,
		FLMUINT,				// uiDictType,
		FLMUINT,				// uiDictNum,
		FLMBOOL)				// bDeleting
	{
		*peAction = SFLM_RESTORE_ACTION_CONTINUE;
		return( NE_SFLM_OK);
	}

	RCODE reportIndexSuspend(
		eRestoreAction *	peAction,
		FLMUINT64,			// ui64TransId,
		FLMUINT)				// uiIndexNum
	{
		*peAction = SFLM_RESTORE_ACTION_CONTINUE;
		return( NE_SFLM_OK);
	}

	RCODE reportIndexResume(
		eRestoreAction *	peAction,
		FLMUINT64,			// ui64TransId,
		FLMUINT)				// uiIndexNum
	{
		*peAction = SFLM_RESTORE_ACTION_CONTINUE;
		return( NE_SFLM_OK);
	}

	RCODE reportReduce(
		eRestoreAction *	peAction,
		FLMUINT64,			// ui64TransId,
		FLMUINT)				// uiCount
	{
		*peAction = SFLM_RESTORE_ACTION_CONTINUE;
		return( NE_SFLM_OK);
	}

	RCODE reportUpgrade(
		eRestoreAction *	peAction,
		FLMUINT64,			// ui64TransId,
		FLMUINT,				// uiOldDbVersion,
		FLMUINT)				// uiNewDbVersion
	{
		*peAction = SFLM_RESTORE_ACTION_CONTINUE;
		return( NE_SFLM_OK);
	}

	RCODE reportEnableEncryption(
		eRestoreAction *	peAction,
		FLMUINT64			// ui64TransId
		)
	{
		*peAction = SFLM_RESTORE_ACTION_CONTINUE;
		return( NE_SFLM_OK);
	}

	RCODE reportWrapKey(
		eRestoreAction *	peAction,
		FLMUINT64)			// ui64TransId
	{
		*peAction = SFLM_RESTORE_ACTION_CONTINUE;
		return( NE_SFLM_OK);
	}

	RCODE reportOpenRflFile(
		eRestoreAction *	peAction,
		FLMUINT)				// uiFileNum
	{
		*peAction = SFLM_RESTORE_ACTION_CONTINUE;
		return( NE_SFLM_OK);
	}

	RCODE reportRflRead(
		eRestoreAction *	peAction,
		FLMUINT,				// uiFileNum,
		FLMUINT)				// uiBytesRead
	{
		*peAction = SFLM_RESTORE_ACTION_CONTINUE;
		return( NE_SFLM_OK);
	}

	virtual FLMINT SQFAPI getRefCount( void)
	{
		return( IF_RestoreStatus::getRefCount());
	}

	virtual FLMINT SQFAPI AddRef( void)
	{
		return( IF_RestoreStatus::AddRef());
	}

	virtual FLMINT SQFAPI Release( void)
	{
		return( IF_RestoreStatus::Release());
	}
};

typedef struct KEY_GEN_INFO
{
	F_TABLE *			pTable;
	F_INDEX *			pIndex;
	F_Row *				pRow;
	F_COLUMN_VALUE *	pColumnValues;
	FLMBOOL				bIsAsia;
	FLMBOOL				bIsCompound;
	FLMBOOL				bAddKeys;
	FLMBYTE *			pucKeyBuf;
	FLMBYTE *			pucData;
	FLMUINT				uiDataBufSize;
	FLMBOOL				bDataBufAllocated;
} KEY_GEN_INFO;

#define FLM_NO_TIMEOUT 0xFF

/*****************************************************************************
Desc: Thread's database object - returned by openDatabase, createDatabase in F_DbSystem class
*****************************************************************************/
class F_Db : public F_Object
{
public:

	F_Db(
		FLMBOOL	bInternalOpen);
		
	virtual ~F_Db();
	
	RCODE transBegin(
		eDbTransType			eTransType,
		FLMUINT					uiMaxLockWait = FLM_NO_TIMEOUT,
		FLMUINT					uiFlags = 0,
		SFLM_DB_HDR *			pDbHeader = NULL);

	RCODE transBegin(
		F_Db *					pDb);

	RCODE transCommit(
		FLMBOOL *				pbEmpty = NULL);

	RCODE transAbort( void);

	FINLINE eDbTransType getTransType( void)
	{
		return( m_eTransType);
	}

	RCODE doCheckpoint(
		FLMUINT					uiTimeout);

	RCODE dbLock(
		eLockType				eLockType,
		FLMINT					iPriority,
		FLMUINT					uiTimeout);

	RCODE dbUnlock( void);

	RCODE getLockType(
		eLockType *				peLockType,
		FLMBOOL *				pbImplicit);

	RCODE getLockInfo(
		FLMINT					iPriority,
		eLockType *				peCurrLockType,
		FLMUINT *				puiThreadId,
		FLMUINT *				puiNumExclQueued,
		FLMUINT *				puiNumSharedQueued,
		FLMUINT *				puiPriorityCount);

	RCODE dupTrans(
		FLMUINT64	ui64TransId);

	RCODE demoteTrans( void);

	RCODE cancelTrans(
		FLMUINT64				ui64TransId);

	RCODE getCommitCnt(
		FLMUINT64 *				pui64CommitCount);

	// Index methods

	RCODE indexStatus(
		FLMUINT					uiIndexNum,
		SFLM_INDEX_STATUS *	pIndexStatus);

	RCODE indexGetNext(
		FLMUINT *				puiIndexNum);

	RCODE indexSuspend(
		FLMUINT					uiIndexNum);

	RCODE indexResume(
		FLMUINT					uiIndexNum);

	// Retrieval Functions

	RCODE	keyRetrieve(
		FLMUINT				uiIndex,
		F_DataVector *		pSearchKey,
		FLMUINT				uiFlags,
		F_DataVector *		pFoundKey);

	RCODE enableEncryption( void);
		
	RCODE wrapKey(
		const char *	pszPassword = NULL);

	RCODE rollOverDbKey( void);
			
	RCODE reduceSize(
		FLMUINT     uiCount,
		FLMUINT *	puiCountRV);

	RCODE upgrade(
		IF_UpgradeClient *	pUpgradeClient);

	RCODE backupBegin(
		eDbBackupType			eBackupType,
		eDbTransType			eTransType,
		FLMUINT					uiMaxLockWait,
		F_Backup **				ppBackup);

	void getRflFileName(
		FLMUINT					uiFileNum,
		FLMBOOL					bBaseOnly,
		char *					pszFileName,
		FLMUINT *				puiFileNameBufSize,
		FLMBOOL *				pbNameTruncated = NULL);

	// Configuration methods

	RCODE setRflKeepFilesFlag(
		FLMBOOL					bKeep);

	RCODE getRflKeepFlag(
		FLMBOOL *				pbKeep);

	RCODE setRflDir(
		const char *			pszNewRflDir);

	void getRflDir(
		char *					pszRflDir);

	RCODE getRflFileNum(
		FLMUINT *				puiRflFileNum);

	RCODE getHighestNotUsedRflFileNum(
		FLMUINT *				puiHighestNotUsedRflFileNum);

	RCODE setRflFileSizeLimits(
		FLMUINT					uiMinRflSize,
		FLMUINT					uiMaxRflSize);

	RCODE getRflFileSizeLimits(
		FLMUINT *				puiRflMinFileSize,
		FLMUINT *				puiRflMaxFileSize);

	RCODE rflRollToNextFile( void);

	RCODE setKeepAbortedTransInRflFlag(
		FLMBOOL					bKeep);

	RCODE getKeepAbortedTransInRflFlag(
		FLMBOOL *				pbKeep);

	RCODE setAutoTurnOffKeepRflFlag(
		FLMBOOL					bAutoTurnOff);

	RCODE getAutoTurnOffKeepRflFlag(
		FLMBOOL *				pbAutoTurnOff);

	FINLINE void setFileExtendSize(
		FLMUINT					uiFileExtendSize)
	{
		m_pDatabase->m_uiFileExtendSize = uiFileExtendSize;
	}

	FINLINE FLMUINT getFileExtendSize( void)
	{
		return( m_pDatabase->m_uiFileExtendSize);
	}

	FINLINE void setAppData(
		void *			pvAppData)
	{
		m_pvAppData = pvAppData;
	}

	FINLINE void * getAppData( void)
	{
		return( m_pvAppData);
	}

	FINLINE void setDeleteStatusObject(
		IF_DeleteStatus *		pDeleteStatus)
	{
		if (m_pDeleteStatus)
		{
			m_pDeleteStatus->Release();
		}
		if ((m_pDeleteStatus = pDeleteStatus) != NULL)
		{
			m_pDeleteStatus->AddRef();
		}
	}

	FINLINE void setCommitClientObject(
		IF_CommitClient *		pCommitClient)
	{
		if (m_pCommitClient)
		{
			m_pCommitClient->Release();
		}
		
		m_pCommitClient = pCommitClient;
		
		if (m_pCommitClient)
		{
			m_pCommitClient->AddRef();
		}
	}

	FINLINE void setIndexingClientObject(
		IF_IxClient *	pIxClient)
	{
		if (m_pIxClient)
		{
			m_pIxClient->Release();
		}
		m_pIxClient = pIxClient;
		if (m_pIxClient)
		{
			m_pIxClient->AddRef();
		}
	}

	FINLINE void setIndexingStatusObject(
		IF_IxStatus *			ifpIxStatus)
	{
		if (m_pIxStatus)
		{
			m_pIxStatus->Release();
		}
		m_pIxStatus = ifpIxStatus;
		if (m_pIxStatus)
		{
			m_pIxStatus->AddRef();
		}
	}

	// Configuration information getting methods

	FINLINE FLMUINT getDbVersion( void)
	{
		return( (FLMUINT)m_pDatabase->m_lastCommittedDbHdr.ui32DbVersion);
	}

	FINLINE FLMUINT getBlockSize( void)
	{
		return( m_pDatabase->m_uiBlockSize);
	}

	FINLINE FLMUINT getDefaultLanguage( void)
	{
		return( m_pDatabase->m_uiDefaultLanguage);
	}

	FINLINE FLMUINT64 getTransID( void)
	{
		if (m_eTransType != SFLM_NO_TRANS)
		{
			return( m_ui64CurrTransID);
		}
		else if (m_uiFlags & FDB_HAS_FILE_LOCK)
		{
			return( m_pDatabase->m_lastCommittedDbHdr.ui64CurrTransID);
		}

		return( 0);
	}

	void getCheckpointInfo(
		SFLM_CHECKPOINT_INFO *	pCheckpointInfo);

	RCODE getDbControlFileName(
		char *					pszControlFileName,
		FLMUINT					uiControlFileBufSize)
	{
		RCODE		rc = NE_SFLM_OK;
		FLMUINT	uiLen = f_strlen( m_pDatabase->m_pszDbPath);

		if (uiLen + 1 > uiControlFileBufSize)
		{
			uiLen = uiControlFileBufSize - 1;
			rc = RC_SET( NE_SFLM_BUFFER_OVERFLOW);
		}
		f_memcpy( pszControlFileName, m_pDatabase->m_pszDbPath, uiLen);
		pszControlFileName [uiLen] = 0;
		return( rc);
	}

	RCODE getLockWaiters(
		IF_LockInfoClient *	pLockInfo);

	RCODE getLastBackupTransID(
		FLMUINT64 *				pui64LastBackupTransID);

	RCODE getBlocksChangedSinceBackup(
		FLMUINT *				puiBlocksChangedSinceBackup);

	RCODE getNextIncBackupSequenceNum(
		FLMUINT *				puiNextIncBackupSequenceNum);

	void getSerialNumber(
		char *					pucSerialNumber);

	RCODE getDiskSpaceUsage(
		FLMUINT64 *				pui64DataSize,
		FLMUINT64 *				pui64RollbackSize,
		FLMUINT64 *				pui64RflSize);

	FINLINE RCODE getMustCloseRC( void)
	{
		return( m_pDatabase->m_rcMustClose);
	}

	FINLINE RCODE getAbortRC( void)
	{
		return( m_AbortRc);
	}

	FINLINE RCODE startTransaction(
		eDbTransType	eReqTransType,
		FLMBOOL *		pbStartedTrans)
	{
		RCODE		rc;

		if( m_eTransType != SFLM_NO_TRANS)
		{
			return( RC_SET_AND_ASSERT( NE_SFLM_ILLEGAL_TRANS_OP));
		}

		if( !pbStartedTrans)
		{
			return( RC_SET( NE_SFLM_NO_TRANS_ACTIVE));
		}

		if( RC_BAD( rc = transBegin( eReqTransType)))
		{
			return( rc);
		}

		*pbStartedTrans = TRUE;
		return( NE_SFLM_OK);
	}

	FINLINE RCODE checkTransaction(
		eDbTransType	eReqTransType,
		FLMBOOL *		pbStartedTrans)
	{
		if( m_AbortRc)
		{
			return( m_AbortRc);
		}
		else if( m_eTransType >= eReqTransType)
		{
			return( NE_SFLM_OK);
		}

		return( startTransaction( eReqTransType, pbStartedTrans));
	}

	FINLINE void setMustAbortTrans(
		RCODE		rc)
	{
		if( RC_BAD( rc) && RC_OK( m_AbortRc))
		{
			m_AbortRc = rc;
		}
	}

	FINLINE RCODE checkState(
		const char *			pszFileName,
		FLMINT					iLineNumber)
	{
		RCODE	rc = NE_SFLM_OK;

		if (m_bMustClose)
		{
			m_pDatabase->logMustCloseReason( pszFileName, iLineNumber);
			rc = RC_SET( NE_SFLM_MUST_CLOSE_DATABASE);
		}
		return( rc);
	}

	FINLINE F_Database * getDatabase( void)
	{
		return m_pDatabase;
	}

	RCODE backgroundIndexBuild(
		IF_Thread *				pThread,
		FLMBOOL *				pbShutdown,
		FLMINT *					piErrorLine);

	FINLINE FLMUINT getLogicalEOF( void)
	{
		return( m_uiLogicalEOF);
	}

	// Key Collector object, used when checking indexes

	FINLINE void setKeyCollector(
		F_KeyCollector *		pKeyColl)
	{
		m_pKeyColl = pKeyColl;
	}

	FINLINE F_KeyCollector * getKeyCollector( void)
	{
		return m_pKeyColl;
	}
	
	RCODE waitForMaintenanceToComplete( void);

	FINLINE F_Dict * getDict( void)
	{
		return( m_pDict);
	}
	
	void removeTableRows(
		FLMUINT		uiTableNum,
		FLMUINT64	ui64TransId);
		
	RCODE insertRow(
		FLMUINT				uiTableNum,
		F_COLUMN_VALUE *	pColumnValues);
		
	RCODE updateRow(
		FLMUINT				uiTableNum,
		F_Row **				ppRow,
		F_COLUMN_VALUE *	pColumnValues);
		
	RCODE deleteRow(
		FLMUINT				uiTableNum,
		FLMUINT64			ui64RowId,
		FLMBOOL				bLogDelete);
		
	RCODE deleteSelectedRows(
		FLMUINT		uiTableNum,
		SQLQuery *	pSqlQuery);
		
	RCODE updateSelectedRows(
		FLMUINT			uiTableNum,
		SQLQuery *		pSqlQuery,
		COLUMN_SET *	pFirstColumnSet,
		FLMUINT			uiNumColumnsToSet);
		
	RCODE createTable(
		FLMUINT				uiTableNum,
		const char *		pszTableName,
		FLMUINT				uiTableNameLen,
		FLMUINT				uiEncDefNum,
		F_COLUMN_DEF *		pColumnDefs,
		FLMUINT				uiNumColumnDefs);
		
	RCODE dropTable(
		FLMUINT				uiTableNum);
		
	RCODE createIndex(
		FLMUINT				uiTableNum,
		FLMUINT				uiIndexNum,
		const char *		pszIndexName,
		FLMUINT				uiIndexNameLen,
		FLMUINT				uiEncDefNum,
		FLMUINT				uiFlags,
		F_INDEX_COL_DEF *	pIxColDefs,
		FLMUINT				uiNumIxColDefs);

	RCODE dropIndex(
		FLMUINT				uiIndexNum);
		
private:

	// This routine assumes that the database mutex is locked
	FINLINE void linkToDict(
		F_Dict *					pDict)
	{
		if (pDict != m_pDict)
		{
			if (m_pDict)
			{
				unlinkFromDict();
			}
			if ((m_pDict = pDict) != NULL)
			{
				pDict->incrUseCount();
			}
		}
	}

	// This routine assumes the database mutex is locked.
	FINLINE void unlinkFromDict( void)
	{
		if (m_pDict)
		{

			// If the use count goes to zero and the F_Dict is not the first one
			// in the file's list or it is not linked to a file, unlink the F_Dict
			// object from its database and delete it.

			if (!m_pDict->decrUseCount() &&
		 		(m_pDict->getPrev() || !m_pDict->getDatabase()))
			{
				m_pDict->unlinkFromDatabase();
			}
			m_pDict = NULL;
		}
	}

	RCODE linkToDatabase(
		F_Database *		pDatabase);

	void unlinkFromDatabase();

	RCODE initDbFiles(
		const char *			pszRflDir,
		SFLM_CREATE_OPTS *	pCreateOpts);

	RCODE beginBackgroundTrans(
		IF_Thread *			pThread);

	RCODE beginTrans(
		eDbTransType		eTransType,
		FLMUINT				uiMaxLockWait = FLM_NO_TIMEOUT,
		FLMUINT				uiFlags = 0,
		SFLM_DB_HDR *		pDbHdr = NULL);

	RCODE beginTrans(
		F_Db *	pDb);

	RCODE	commitTrans(
		FLMUINT				uiNewLogicalEOF,
		FLMBOOL				bForceCheckpoint,
		FLMBOOL *			pbEmpty = NULL);

	RCODE	abortTrans(
		FLMBOOL				bOkToLogAbort = TRUE);

	RCODE readRollbackLog(
		FLMUINT				uiLogEOF,
		FLMUINT *			puiCurrAddr,
		F_BLK_HDR *			pBlkHdr,
		FLMBOOL *			pbIsBeforeImageBlk);

	RCODE processBeforeImage(
		FLMUINT				uiLogEOF,
		FLMUINT *			puiCurrAddrRV,
		F_BLK_HDR *			pBlkHdr,
		FLMBOOL				bDoingRecovery,
		FLMUINT64			ui64MaxTransID);

	RCODE physRollback(
		FLMUINT				uiLogEOF,
		FLMUINT				uiFirstLogBlkAddr,
		FLMBOOL				bDoingRecovery,
		FLMUINT64			ui64MaxTransID);

	void completeOpenOrCreate(
		RCODE					rc,
		FLMBOOL				bNewDatabase);

	RCODE startBackgroundIndexing( void);

	void unlinkFromTransList(
		FLMBOOL				bCommitting);

	RCODE lockExclusive(
		FLMUINT				uiMaxLockWait);

	void unlockExclusive( void);

	RCODE readDictionary( void);

	RCODE dictCreate( void);

	RCODE dictReadLFH( void);

	RCODE dictReadEncDefs( void);
	
	RCODE dictReadTables( void);
	
	RCODE dictReadColumns( void);
	
	RCODE dictReadIndexes( void);
	
	RCODE dictReadIndexComponents( void);
	
	RCODE dictOpen( void);

	RCODE dictClone( void);

	RCODE createNewDict( void);

	FINLINE void getDbHdrInfo(
		SFLM_DB_HDR *		pDbHdr)
	{
		// IMPORTANT NOTE: Any changes to this method should also be
		// mirrored with changes to the other getDbHdrInfo call - see below.
		
		m_ui64CurrTransID = pDbHdr->ui64CurrTransID;
		m_uiLogicalEOF = (FLMUINT)pDbHdr->ui32LogicalEOF;

		// If we are doing a read transaction, this is only needed
		// if we are checking the database.

		m_uiFirstAvailBlkAddr = (FLMUINT)pDbHdr->ui32FirstAvailBlkAddr;
	}
	
	FINLINE void getDbHdrInfo(
		F_Db *	pDb)
	{
		m_ui64CurrTransID = pDb->m_ui64CurrTransID;
		m_uiLogicalEOF = pDb->m_uiLogicalEOF;

		// If we are doing a read transaction, this is only needed
		// if we are checking the database.

		m_uiFirstAvailBlkAddr = pDb->m_uiFirstAvailBlkAddr;
	}
	
	FINLINE FLMBOOL okToCommitTrans( void)
	{
		return( m_eTransType == SFLM_READ_TRANS ||
				  m_AbortRc == NE_SFLM_OK
				  ? TRUE
				  : FALSE);
	}

	RCODE processDupKeys(
		F_INDEX *	pIndex);

	RCODE keysCommit(
		FLMBOOL				bCommittingTrans,
		FLMBOOL				bSortKeys = TRUE);

	RCODE refUpdate(
		F_INDEX *			pIndex,
		KREF_ENTRY *		pKrefEntry,
		FLMBOOL				bNormalUpdate);

	FINLINE RCODE flushKeys( void)
	{
		RCODE	rc = NE_SFLM_OK;

		if( m_bKrefSetup)
		{
			if( m_uiKrefCount)
			{
				if (RC_BAD( rc = keysCommit( FALSE)))
				{
					goto Exit;
				}
			}
			
			m_pKrefReset = m_pKrefPool->poolMark();
		}

	Exit:

		return( rc);
	}

	RCODE krefCntrlCheck( void);

	void krefCntrlFree( void);

	FINLINE FLMBOOL isKrefOverThreshold( void)
	{
		if( (((m_pKrefPool->getBlockSize() * 3) - 250) <= m_uiTotalKrefBytes) ||
			m_uiKrefCount > (m_uiKrefTblSize - 128))
		{
			return( TRUE);
		}
	
		return( FALSE);
	}

	RCODE addToKrefTbl(
		FLMUINT				uiKeyLen,
		FLMUINT				uiDataLen);

	RCODE buildData(
		ICD *			pIcd,
		FLMUINT		uiDataComponent,
		FLMUINT		uiKeyLen,
		FLMUINT		uiDataLen);
		
	RCODE finishKeyComponent(
		ICD *		pIcd,
		FLMUINT	uiKeyComponent,
		FLMUINT	uiKeyLen);
		
	RCODE genTextKeyComponents(
		F_COLUMN *	pColumn,
		ICD *			pIcd,
		FLMUINT		uiKeyComponent,
		FLMUINT		uiKeyLen,
		FLMBYTE **	ppucTmpBuf,
		FLMUINT *	puiTmpBufSize,
		void **		ppvMark);
		
	RCODE genOtherKeyComponent(
		F_COLUMN *	pColumn,
		ICD *			pIcd,
		FLMUINT		uiKeyComponent,
		FLMUINT		uiKeyLen);
		
	RCODE buildKeys(
		ICD *		pIcd,
		FLMUINT	uiKeyComponent,
		FLMUINT	uiKeyLen);
		
	RCODE buildKeys(
		F_INDEX *			pIndex,
		F_TABLE *			pTable,
		F_Row *				pRow,
		FLMBOOL				bAddKeys,
		F_COLUMN_VALUE *	pColumnValues);
		
	RCODE updateIndexKeys(
		FLMUINT				uiTableNum,
		F_Row *				pRow,
		FLMBOOL				bAddKeys,
		F_COLUMN_VALUE *	pColumnValues);
		
	void indexingAfterCommit( void);

	void indexingAfterAbort( void);

	RCODE	addToStopList(
		FLMUINT				uiIndexNum);

	RCODE	addToStartList(
		FLMUINT				uiIndexNum);

	void stopBackgroundIndexThread(
		FLMUINT				uiIndexNum,
		FLMBOOL				bWait,
		FLMBOOL *			pbStopped);

	RCODE startIndexBuild(
		FLMUINT				uiIndexNum);

	RCODE setIxStateInfo(
		FLMUINT				uiIndexNum,
		FLMUINT64			ui64LastRowIndexed,
		FLMUINT				uiState);

	RCODE indexSetOfRows(
		FLMUINT					uiIndexNum,
		FLMUINT64				ui64StartRowId,
		FLMUINT64				ui64EndRowId,
		IF_IxStatus *			pIxStatus,
		IF_IxClient *			pIxClient,
		SFLM_INDEX_STATUS *	pIndexStatus,
		FLMBOOL *				pbHitEnd,
		IF_Thread *				pThread);
		
	RCODE readBlkHdr(
		FLMUINT				uiBlkAddress,
		F_BLK_HDR *			pBlkHdr,
		FLMINT *				piType);

	SFLM_LFILE_STATS * getLFileStatPtr(
		LFILE *				pLFile);

#define FLM_UPD_ADD					0x00001
#define FLM_UPD_INTERNAL_CHANGE	0x00004

	RCODE getCachedBTree(
		FLMUINT				uiTableNum,
		F_Btree **			ppBTree);

	RCODE flushDirtyRows( void);

	RCODE maintBlockChainFree(
		FLMUINT64		ui64MaintRowId,
		FLMUINT			uiStartBlkAddr,
		FLMUINT 			uiBlocksToFree,
		FLMUINT			uiExpectedEndBlkAddr,
		FLMUINT *		puiBlocksFreed);
		
	RCODE encryptData(
		FLMUINT				uiEncDefNum,
		FLMBYTE *			pucIV,
		FLMBYTE *			pucBuffer,
		FLMUINT				uiBufferSize,
		FLMUINT				uiDataLen,
		FLMUINT *			puiEncryptedLength);
		
	RCODE decryptData(
		FLMUINT				uiEncDefNum,
		FLMBYTE *			pucIV,
		void *				pvInBuf,
		FLMUINT				uiInLen,
		void *				pvOutBuf,
		FLMUINT				uiOutBufLen);
	
	RCODE createDbKey();

	// Private data members.

	F_Database *			m_pDatabase;		// Pointer to F_Database object
	F_Dict *					m_pDict;				// Pointer to dictionary object
	F_Db *					m_pNextForDatabase;	// Next F_Db associated with F_Database
														// NOTE: gv_SFlmSysData.hShareMutex
														// must be locked to set this
	F_Db *					m_pPrevForDatabase;	// Prev F_Db associated with F_Database
														// NOTE: gv_SFlmSysData.hShareMutex
														// must be locked to set this
	void *					m_pvAppData;		// Application data that is used
														// to associate this F_Db with
														// an object in the application
														// space.
	FLMUINT					m_uiThreadId;		// Thread that started the current
														// transaction, if any.  NOTE:
														// Only set on transaction begin.
														// Hence, if operations are performed
														// by multiple threads, within the
														// transaction, it will not necessarily
														// reflect the thread that is currently
														// using the F_Db.
	FLMBOOL					m_bMustClose;		// An error has occurred that requires
														// the application to stop using (close)
														// this FDB
	F_SuperFileHdl *		m_pSFileHdl;		// Pointer to the super file handle
	FLMUINT					m_uiFlags;			// Flags for this F_Db.

	// TRANSACTION STATE STUFF

	FLMUINT					m_uiTransCount;	// Transaction counter for the F_Db.
														// Incremented whenever a transaction
														// is started on this F_Db.  Used so
														// that FLAIM can tell if an implicit
														// transaction it started is still in
														// effect.  This should NOT be
														// confused with update transaction
														// IDs.
	eDbTransType			m_eTransType;		// Type of transaction
	RCODE						m_AbortRc;			// If not NE_SFLM_OK, transaction must be
														// aborted.
	FLMUINT64  				m_ui64CurrTransID;// Current transaction ID.
	FLMUINT					m_uiFirstAvailBlkAddr;	// Address of first block in avail list
	FLMUINT					m_uiLogicalEOF;			// Current logical end of file.  New
																// blocks are allocated at this address.
	FLMUINT					m_uiUpgradeCPFileNum;
	FLMUINT					m_uiUpgradeCPOffset;
														// RFL file number and offset to set
														// RFL to during an upgrade operation
														// that happens during a restore or
														// recovery.
	FLMUINT					m_uiTransEOF;			// Address of logical end of file
														// when the last transaction
														// committed. A block beyond this
														// point in the file is going to be
														// a new block and will not need to
														// be logged.
	F_TMSTAMP				m_TransStartTime;	// Transaction start time, for stats

	// KREF STUFF
														
	KEY_GEN_INFO			m_keyGenInfo;		// Information for generating index
														// keys.
	KREF_ENTRY **			m_pKrefTbl;			// Pointer to KREF table, which is an array
														// of KREF_ENTRY * pointers.
	FLMUINT					m_uiKrefTblSize;	// KREF table size.
	FLMUINT					m_uiKrefCount;		// Number of entries in KREF table that
														// are currently used.
	FLMUINT					m_uiTotalKrefBytes;	// Total number of entries allocated
															// in the pool.
	FLMBYTE *				m_pucKrefKeyBuf;	// Pointer to temporary key buffer.
	FLMBOOL					m_bKrefSetup;		// True if the KRef table has been initialized.
	F_Pool *					m_pKrefPool;		// Memory pool to use
	FLMBOOL					m_bReuseKrefPool;	// Reuse pool instead of free it?
	FLMBOOL					m_bKrefCompoundKey;	// True if a compound key has been processed.
	void *					m_pKrefReset;			// Used to reset the Kref pool on
														// indexing failures
	F_Pool					m_tmpKrefPool;		// KREF pool to be used during
														// read transactions - only used when
														// checking indexes.

	// UPDATE TRANSACTION STUFF

	FLMBOOL					m_bHadUpdOper;		// Did this transaction have any
														// updates?
	FLMUINT					m_uiBlkChangeCnt;	// Number of times ScaLogPhysBlk has
														// been called during this transaction.
														// This is used by the cursor code to
														// know when it is necessary to
														// re-position in the B-Tree.0
	IXD_FIXUP *				m_pIxdFixups;		// List of indexes whose IXD needs
														// to be restored to its prior
														// state if the transaction aborts
	// READ TRANSACTION STUFF

	F_Db *					m_pNextReadTrans;	// Next active read transaction for
														// this database.
														// NOTE: If uiKilledTime (see below)
														// is non-zero, then transaction is
														// in killed list.
	F_Db *					m_pPrevReadTrans;	// Previous active read transaction
														// for this database.
														// NOTE: If m_uiKilledTime (see below)
														// is non-zero, then transaction is
														// in killed list.
	FLMUINT					m_uiInactiveTime;	// If non-zero, this is the last time
														// the checkpoint thread marked this
														// transaction as inactive.  If zero,
														// it means that the transaction is
														// active, or it has not been marked
														// by the checkpoint thread as
														// inactive.  If it stays non-zero for
														// five or more minutes, it will be
														// killed.
	FLMUINT					m_uiKilledTime;	// Time transaction was killed, if
														// non-zero.
	// Misc. DB Info.

	FLMBOOL					m_bItemStateUpdOk;//	This variable is used to ensure
														// that FlmDbSweep / recovery are the
														// only ways that:
														// 1) an element or attribute's state
														//		can be changed to 'unused'
														// 2) a 'purge' element or attribute
														//		can be deleted

	F_Pool					m_tempPool;			// Temporary memory pool.  It
														// is only used for the duration of
														// a FLAIM operation and then reset.
														// The first block in the pool is
														// retained between operations to
														// help performance.

	// Callback functions.
	IF_DeleteStatus *		m_pDeleteStatus;	// Handles status info coming back
															// from deleting a BTree
	IF_IxClient *			m_pIxClient;		// Indexing callback
	IF_IxStatus *			m_pIxStatus;		// Indexing status callback
	IF_CommitClient *		m_pCommitClient;	// Commit callback

	SFLM_STATS *			m_pStats;
	SFLM_DB_STATS *		m_pDbStats;			// DB statistics pointer.
	SFLM_LFILE_STATS *	m_pLFileStats;		// LFILE statistics pointer.
	FLMUINT					m_uiLFileAllocSeq;// Allocation sequence number for
														// LFILE statistics array so we
														// can tell if the array has been
														// reallocated and we need to reset
														// our pLFileStats pointer.
	SFLM_STATS				m_Stats;				// Statistics kept here until end
														// of transaction.
	FLMBOOL					m_bStatsInitialized;// Has statistics structure been
														// initialized?
	F_BKGND_IX *			m_pIxStartList;	// Indexing threads to start at 
														// the conclusion of the transaction.
	F_BKGND_IX *			m_pIxStopList;		// Indexing threads to stop at 
														// the conclusion of the transaction.
	F_Btree *				m_pCachedBTree;	// BTree object used for node operations
	F_KeyCollector *		m_pKeyColl;			// Special purpose object used when checking
														// indexes in the F_DbCheck class.
	FLMUINT					m_uiDirtyRowCount;
	F_SEM						m_hWaitSem;			// Semaphore that is used when
														// waiting for reads to complete

friend class F_Database;
friend class F_Dict;
friend class F_DbSystem;
friend class F_Rfl;
friend class F_Btree;
friend class F_Backup;
friend class F_Row;
friend class F_BTreeIStream;
friend class F_DataVector;
friend class F_DbRebuild;
friend class F_DbCheck;
friend class F_Query;
friend class FSIndexCursor;
friend class FSTableCursor;
friend class F_BtRSFactory;
friend class F_BtResultSet;
friend class F_CachedBlock;
friend class F_BlockCacheMgr;
friend class F_RowCacheMgr;
friend class F_GlobalCacheMgr;
friend class F_QueryResultSet;
friend class F_BTreeInfo;
friend class SQLQuery;
friend class SQLStatement;
};

typedef struct BTREE_INFO
{
	FLMUINT						uiLfNum;
	char *						pszLfName;
	FLMUINT						uiNumLevels;
	SFLM_BTREE_LEVEL_INFO	levelInfo [MAX_LEVELS];
} BTREE_INFO;

/*****************************************************************************
Desc:	Object for gathering B-Tree information.
*****************************************************************************/
class F_BTreeInfo : public F_Object
{
public:
	F_BTreeInfo()
	{
		m_pIndexArray = NULL;
		m_uiIndexArraySize = 0;
		m_uiNumIndexes = 0;
		m_pTableArray = NULL;
		m_uiTableArraySize = 0;
		m_uiNumTables = 0;
		m_pool.poolInit( 512);
	}
	
	virtual ~F_BTreeInfo()
	{
		if (m_pIndexArray)
		{
			f_free( &m_pIndexArray);
		}
		if (m_pTableArray)
		{
			f_free( &m_pTableArray);
		}
		m_pool.poolFree();
	}
	
	FINLINE void clearBTreeInfo( void)
	{
		m_uiNumIndexes = 0;
		m_uiNumTables = 0;
	}
	
	RCODE collectIndexInfo(
		F_Db *					pDb,
		FLMUINT					uiIndexNum,
		IF_BTreeInfoStatus *	pInfoStatus);
		
	RCODE collectTableInfo(
		F_Db *					pDb,
		FLMUINT					uiTableNum,
		IF_BTreeInfoStatus *	pInfoStatus);
			
	FINLINE FLMUINT getNumIndexes( void)
	{
		return( m_uiNumIndexes);
	}
		
	FINLINE FLMUINT getNumTables( void)
	{
		return( m_uiNumTables);
	}
		
	FINLINE FLMBOOL getIndexInfo(
		FLMUINT		uiNthIndex,
		FLMUINT *	puiIndexNum,
		char **		ppszIndexName,
		FLMUINT *	puiNumLevels)
	{
		if (uiNthIndex < m_uiNumIndexes)
		{
			*puiIndexNum = m_pIndexArray [uiNthIndex].uiLfNum;
			*puiNumLevels = m_pIndexArray [uiNthIndex].uiNumLevels;
			*ppszIndexName = m_pIndexArray [uiNthIndex].pszLfName;
			return( TRUE);
		}
		else
		{
			*puiIndexNum = 0;
			*ppszIndexName = NULL;
			*puiNumLevels = 0;
			return( FALSE);
		}
	}
		
	FINLINE FLMBOOL getTableInfo(
		FLMUINT		uiNthTable,
		FLMUINT *	puiTableNum,
		char **		ppszTableName,
		FLMUINT *	puiNumLevels)
	{
		if (uiNthTable < m_uiNumTables)
		{
			*puiTableNum = m_pTableArray [uiNthTable].uiLfNum;
			*puiNumLevels = m_pTableArray [uiNthTable].uiNumLevels;
			*ppszTableName = m_pTableArray [uiNthTable].pszLfName;
			return( TRUE);
		}
		else
		{
			*puiTableNum = 0;
			*puiNumLevels = 0;
			*ppszTableName = NULL;
			return( FALSE);
		}
	}
		
	FINLINE FLMBOOL getIndexLevelInfo(
		FLMUINT						uiNthIndex,
		FLMUINT						uiBTreeLevel,
		SFLM_BTREE_LEVEL_INFO *	pLevelInfo)
	{
		if (uiNthIndex < m_uiNumIndexes &&
			 uiBTreeLevel < m_pIndexArray [uiNthIndex].uiNumLevels)
		{
			f_memcpy( pLevelInfo,
				&(m_pIndexArray [uiNthIndex].levelInfo [uiBTreeLevel]),
				sizeof( SFLM_BTREE_LEVEL_INFO));
			return( TRUE);
		}
		else
		{
			return( FALSE);
		}
	}

	FINLINE FLMBOOL getTableLevelInfo(
		FLMUINT						uiNthTable,
		FLMUINT						uiBTreeLevel,
		SFLM_BTREE_LEVEL_INFO *	pLevelInfo)
	{
		if (uiNthTable < m_uiNumTables &&
			 uiBTreeLevel < m_pTableArray [uiNthTable].uiNumLevels)
		{
			f_memcpy( pLevelInfo,
				&(m_pTableArray [uiNthTable].levelInfo [uiBTreeLevel]),
				sizeof( SFLM_BTREE_LEVEL_INFO));
			return( TRUE);
		}
		else
		{
			return( FALSE);
		}
	}

private:

	RCODE collectBlockInfo(
		F_Db *					pDb,
		LFILE *					pLFile,
		BTREE_INFO *			pBTreeInfo,
		F_BTREE_BLK_HDR *		pBlkHdr,
		F_INDEX *				pIndex);
		
	RCODE collectBTreeInfo(
		F_Db *					pDb,
		LFILE *					pLFile,
		BTREE_INFO *			pBTreeInfo,
		const char *			pszName,
		F_INDEX *				pIndex);

	FINLINE RCODE doCallback( void)
	{
		if (m_pInfoStatus)
		{
			return( m_pInfoStatus->infoStatus( m_uiCurrLfNum, m_bIsTable,
						m_pszCurrLfName, m_uiCurrLevel,
						m_ui64CurrLfBlockCount, m_ui64CurrLevelBlockCount,
						m_ui64TotalBlockCount));
		}
		else
		{
			return( NE_SFLM_OK);
		}
	}
	
	BTREE_INFO *			m_pIndexArray;
	FLMUINT					m_uiIndexArraySize;
	FLMUINT					m_uiNumIndexes;
	BTREE_INFO *			m_pTableArray;
	FLMUINT					m_uiTableArraySize;
	FLMUINT					m_uiNumTables;
	F_Pool					m_pool;
	
	// Items for the callback function.
	
	IF_BTreeInfoStatus *	m_pInfoStatus;
	FLMUINT					m_uiBlockSize;
	FLMUINT					m_uiCurrLfNum;
	FLMBOOL					m_bIsTable;
	char *					m_pszCurrLfName;
	FLMUINT					m_uiCurrLevel;
	FLMUINT64				m_ui64CurrLfBlockCount;
	FLMUINT64				m_ui64CurrLevelBlockCount;
	FLMUINT64				m_ui64TotalBlockCount;
};

RCODE ixKeyCompare(
	F_Db *				pDb,
	F_INDEX *			pIndex,
	FLMBOOL				bCompareRowId,
	F_DataVector *		pSearchKey1,
	F_Row *				pRow1,
	const void *		pvKey1,
	FLMUINT				uiKeyLen1,
	F_DataVector *		pSearchKey2,
	F_Row *				pRow2,
	const void *		pvKey2,
	FLMUINT				uiKeyLen2,
	FLMINT *				piCompare);
	
/********************************************************************
Desc:	Class for comparing two keys in an index.
********************************************************************/
class IXKeyCompare : public IF_ResultSetCompare
{
public:

	IXKeyCompare()
	{
		
		// m_pDb is used to sort truncated keys if necessary.
		// m_pIxd is used for comparison
		
		m_pDb = NULL;
		m_pIndex = NULL;
		m_pSearchKey = NULL;
		m_bCompareRowId = TRUE;
		m_pOldRow = NULL;
	}

	virtual ~IXKeyCompare()
	{
	}

	FINLINE RCODE SQFAPI compare(
		const void *	pvKey1,
		FLMUINT			uiKeyLen1,
		const void *	pvKey2,
		FLMUINT			uiKeyLen2,
		FLMINT *			piCompare)
	{
		return( ixKeyCompare( m_pDb, m_pIndex, m_bCompareRowId,
						m_pSearchKey, m_pOldRow, pvKey1, uiKeyLen1,
						m_pSearchKey, m_pOldRow, pvKey2, uiKeyLen2,
						piCompare));
	}
	
	FINLINE void setIxInfo(
		F_Db *		pDb,
		F_INDEX *	pIndex)
	{
		m_pDb = pDb;
		m_pIndex = pIndex;
	}
	
	FINLINE void setSearchKey(
		F_DataVector *	pSearchKey)
	{
		m_pSearchKey = pSearchKey;
	}

	FINLINE void setCompareRowId(
		FLMBOOL	bCompareRowId)
	{
		m_bCompareRowId = bCompareRowId;
	}

	FINLINE void setOldRow(
		F_Row *	pOldRow)
	{
		m_pOldRow = pOldRow;
	}
	
	virtual FLMINT SQFAPI getRefCount( void)
	{
		return( IF_ResultSetCompare::getRefCount());
	}

	virtual FLMINT SQFAPI AddRef( void)
	{
		return( IF_ResultSetCompare::AddRef());
	}

	virtual FLMINT SQFAPI Release( void)
	{
		return( IF_ResultSetCompare::Release());
	}
	
private:

	F_Db *				m_pDb;
	F_INDEX *			m_pIndex;
	F_DataVector *		m_pSearchKey;
	FLMBOOL				m_bCompareRowId;
	F_Row *				m_pOldRow;
};

/*=============================================================================
Desc:	Result set class for queries that do sorting.
=============================================================================*/
class F_QueryResultSet : public F_Object
{
public:

	F_QueryResultSet()
	{
		m_pBTree = NULL;
		m_pResultSetDb = NULL;
		m_pSrcDb = NULL;
		m_pIndex = NULL;
		m_uiCurrPos = FLM_MAX_UINT;
		m_uiCount = 0;
		m_bPositioned = FALSE;
		m_hMutex = F_MUTEX_NULL;
	}

	~F_QueryResultSet();
	
	// Initialize the result set
	
	RCODE initResultSet(
		FLMBOOL	bUseIxCompareObj,
		FLMBOOL	bEnableEncryption);
	
	FINLINE void setIxInfo(
		F_Db *		pSrcDb,
		F_INDEX *	pIndex)
	{
		m_pSrcDb = pSrcDb;
		m_pIndex = pIndex;
		m_compareObj.setIxInfo( pSrcDb, pIndex);
	}

	// Entry Add and Sort Methods

	RCODE addEntry(						// Variable or fixed length entry coming in
		FLMBYTE *	pucKey,				// key for sorting.
		FLMUINT		uiKeyLength,
		FLMBOOL		bLockMutex);

	// Methods to read entries.

	RCODE getFirst(
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyBufSize,
		FLMUINT *	puiKeyLen,
		FLMBOOL		bLockMutex);

	RCODE getLast(
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyBufSize,
		FLMUINT *	puiKeyLen,
		FLMBOOL		bLockMutex);

	RCODE getNext(
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyBufSize,
		FLMUINT *	puiKeyLen,
		FLMBOOL		bLockMutex);

	RCODE getPrev(
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyBufSize,
		FLMUINT *	puiKeyLen,
		FLMBOOL		bLockMutex);

	RCODE getCurrent(
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyBufSize,
		FLMUINT *	puiKeyLen,
		FLMBOOL		bLockMutex);

	RCODE positionToEntry(
		FLMBYTE *		pucKey,
		FLMUINT			uiKeyBufSize,
		FLMUINT *		puiKeyLen,
		F_DataVector *	pSearchKey,
		FLMUINT			uiFlags,
		FLMBOOL			bLockMutex);

	RCODE positionToEntry(
		FLMUINT		uiPosition,
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyBufSize,
		FLMUINT *	puiKeyLen,
		FLMBOOL		bLockMutex);

	FINLINE FLMUINT getCount( void)
	{
		return( m_uiCount);
	}
	
	FINLINE FLMUINT getCurrPos( void)
	{
		return( m_uiCurrPos);
	}
	
	FINLINE void lockMutex( void)
	{
		f_mutexLock( m_hMutex);
	}
		
	FINLINE void unlockMutex( void)
	{
		f_mutexUnlock( m_hMutex);
	}
		
private:

	char				m_szResultSetDibName [F_PATH_MAX_SIZE];
	F_Db *			m_pResultSetDb;
	F_Btree *		m_pBTree;
	LFILE				m_LFile;
	F_Db *			m_pSrcDb;
	F_INDEX *		m_pIndex;
	IXKeyCompare	m_compareObj;
	FLMUINT			m_uiCurrPos;
	FLMUINT			m_uiCount;
	FLMBOOL			m_bPositioned;
	F_MUTEX			m_hMutex;
};

typedef struct RS_WAITER
{
	FLMUINT					uiThreadId;		// Thread of waiter
	F_SEM						hESem;			// Semaphore to signal to wake up thread.
	RCODE *					pRc;				// Pointer to return code that is to
													// be set.
	FLMUINT					uiWaitStartTime;
													// Time we started waiting.
	FLMUINT					uiTimeLimit;	// Maximum time (milliseconds) to wait
													// before timing out.
	FLMUINT					uiNumToWaitFor;// Wait until we get at least this many
													// in the result set - or until the
													// result set is complete.
	RS_WAITER *				pNext;			// Next lock waiter in list.
	RS_WAITER *				pPrev;			// Previous lock waiter in list.
} RS_WAITER;

/*****************************************************************************
Desc:		FLAIM database system object
******************************************************************************/
class F_DbSystem : public F_Object
{
public:

	F_DbSystem() 
	{
		m_refCnt = 1;
	}

	virtual ~F_DbSystem()
	{
	}

	virtual FLMINT SQFAPI AddRef( void);

	virtual FLMINT SQFAPI Release( void);

	RCODE init( void);

	RCODE updateIniFile(
		const char *			pszParamName,
		const char *			pszValue);

	void exit();

	
	void getFileSystem(
		IF_FileSystem **		ppFileSystem);
		
	RCODE createDatabase(
		const char *			pszDbFileName,
		const char *			pszDataDir,
		const char *			pszRflDir,
		SFLM_CREATE_OPTS *	pCreateOpts,
		FLMBOOL					bTempDb,
		F_Db **					ppDb);

	FINLINE RCODE createDatabase(
		const char *			pszDbFileName,
		const char *			pszDataDir,
		const char *			pszRflDir,
		SFLM_CREATE_OPTS *	pCreateOpts,
		F_Db **					ppDb)
	{
		return( createDatabase( pszDbFileName, pszDataDir, pszRflDir,
								pCreateOpts, FALSE, ppDb));
	}

	RCODE openDatabase(
		const char *	pszDbFileName,
		const char *	pszDataDir,
		const char *	pszRflDir,
		const char *	pszPassword,
		FLMUINT			uiOpenFlags,
		F_Db **			ppDb);
	
	RCODE dbRebuild(						
		const char *				pszSourceDbPath,
		const char *				pszSourceDataDir,
		const char *				pszDestDbPath,
		const char *				pszDestDataDir,
		const char *				pszDestRflDir,
		const char *				pszDictPath,
		const char *				pszPassword,
		SFLM_CREATE_OPTS *		pCreateOpts,
		FLMUINT64 *					pui64TotRows,
		FLMUINT64 *					pui64RowsRecov,
		IF_DbRebuildStatus *		pRebuildStatus);

	RCODE dbCheck(
		const char *				pszDbFileName,
		const char *				pszDataDir,
		const char *				pszRflDir,
		const char *				pszPassword,
		FLMUINT						uiFlags,
		F_DbInfo **					ppDbInfo,
		IF_DbCheckStatus *		pDbCheck);

	FINLINE RCODE dbDup(
		F_Db *			pDb,
		F_Db **			ppDb)
	{
		return( openDatabase( pDb->m_pDatabase, NULL, NULL, NULL, NULL, 0,
									FALSE, NULL, NULL, NULL, ppDb));
	}

	FINLINE RCODE setDynamicMemoryLimit(
		FLMUINT					uiCacheAdjustPercent,
		FLMUINT					uiCacheAdjustMin,
		FLMUINT					uiCacheAdjustMax,
		FLMUINT					uiCacheAdjustMinToLeave)
	{
		return( gv_SFlmSysData.pGlobalCacheMgr->setDynamicMemoryLimit(
						uiCacheAdjustPercent, uiCacheAdjustMin,
						uiCacheAdjustMax, uiCacheAdjustMinToLeave));
	}

	FINLINE RCODE setHardMemoryLimit(
		FLMUINT					uiPercent,
		FLMBOOL					bPercentOfAvail,
		FLMUINT					uiMin,
		FLMUINT					uiMax,
		FLMUINT					uiMinToLeave,
		FLMBOOL					bPreallocate)
	{
		return( gv_SFlmSysData.pGlobalCacheMgr->setHardMemoryLimit( uiPercent,
						bPercentOfAvail, uiMin, uiMax, uiMinToLeave, bPreallocate));
	}

	// Determine if dyamic cache adjusting is supported.

	FINLINE FLMBOOL getDynamicCacheSupported( void)
	{
#ifdef FLM_CAN_GET_PHYS_MEM
		return( TRUE);
#else
		return( FALSE);
#endif
	}
			
	FINLINE void getCacheInfo(
		SFLM_CACHE_INFO *		pCacheInfo)
	{
		gv_SFlmSysData.pGlobalCacheMgr->getCacheInfo( pCacheInfo);
	}

	// Enable/disable cache debugging mode

	void enableCacheDebug(
		FLMBOOL		bDebug);

	FLMBOOL cacheDebugEnabled( void);

	// Clear cache

	FINLINE RCODE clearCache(
		F_Db *		pDb)
	{
		return( gv_SFlmSysData.pGlobalCacheMgr->clearCache( pDb));
	}

	// Close all files that have not been used for the specified number of
	// seconds.

	RCODE closeUnusedFiles(
		FLMUINT		uiSeconds);

	// Start gathering statistics.
	
	void startStats( void);

	// Stop gathering statistics.
	
	void stopStats( void);

	// Reset statistics.
	
	void resetStats( void);

	RCODE getStats(
		SFLM_STATS *			pFlmStats);

	void freeStats(
		SFLM_STATS *			pFlmStats);

	// Set the maximum number of queries to save.
	
	void setQuerySaveMax(
		FLMUINT					uiMaxToSave);

	FLMUINT getQuerySaveMax( void);

	// Set temporary directory.
	
	RCODE setTempDir(
		const char *			pszPath);

	RCODE getTempDir(
		char *					pszPath);

	// Maximum seconds between checkpoints.	

	void setCheckpointInterval(
		FLMUINT					uiSeconds);

	FLMUINT getCheckpointInterval( void);

	// Set interval for dynamically adjusting cache limit.

	void setCacheAdjustInterval(
		FLMUINT					uiSeconds);

	FLMUINT getCacheAdjustInterval( void);

	// Set interval for dynamically cleaning out old cache blocks and records.
	
	void setCacheCleanupInterval(
		FLMUINT					uiSeconds);

	FLMUINT getCacheCleanupInterval( void);

	// Set interval for cleaning up unused structures.

	void setUnusedCleanupInterval(
		FLMUINT					uiSeconds);

	FLMUINT getUnusedCleanupInterval( void);

	// Set maximum time for an item to be unused.
	
	void setMaxUnusedTime(
		FLMUINT					uiSeconds);

	FLMUINT getMaxUnusedTime( void);
	
	// Specify the logger object

	void setLogger(
		IF_LoggerClient *		pLogger);
		
	// Enable or disable use of ESM
	
	void enableExtendedServerMemory(
		FLMBOOL					bEnable);

	FLMBOOL extendedServerMemoryEnabled( void);

	void deactivateOpenDb(
		const char *			pszDbFileName,
		const char *			pszDataDir);

	// Maximum dirty cache.
	
	void setDirtyCacheLimits(
		FLMUINT					uiMaxDirty,
		FLMUINT					uiLowDirty);

	void getDirtyCacheLimits(
		FLMUINT *			puiMaxDirty,
		FLMUINT *			puiLowDirty);

	RCODE getThreadInfo(
		IF_ThreadInfo **	ppThreadInfo);

	RCODE registerForEvent(
		eEventCategory		eCategory,
		IF_EventClient *	pEventClient);

	void deregisterForEvent(
		eEventCategory		eCategory,
		IF_EventClient *	pEventClient);

	RCODE getNextMetaphone(
		IF_IStream *		pIStream,
		FLMUINT *			puiMetaphone,
		FLMUINT *			puiAltMetaphone = NULL);

	RCODE dbCopy(
		const char *		pszSrcDbName,
		const char *		pszSrcDataDir,
		const char *		pszSrcRflDir,
		const char *		pszDestDbName,
		const char *		pszDestDataDir,
		const char *		pszDestRflDir,
		IF_DbCopyStatus *	ifpStatus);

	RCODE dropDatabase(
		const char *		pszDbName,
		const char *		pszDataDir,
		const char *		pszRflDir,
		FLMBOOL				bRemoveRflFiles);

	RCODE dbRename(
		const char *			pszDbName,
		const char *			pszDataDir,
		const char *			pszRflDir,
		const char *			pszNewDbName,
		FLMBOOL					bOverwriteDestOk,
		IF_DbRenameStatus *	ifpStatus);

	RCODE dbRestore(
		const char *			pszDbPath,
		const char *			pszDataDir,
		const char *			pszRflDir,
		const char *			pszBackupPath,
		const char *			pszPassword,
		IF_RestoreClient *	pRestoreObj,
		IF_RestoreStatus *	pRestoreStatus);

	RCODE strCmp(
		FLMUINT					uiCompFlags,
		FLMUINT					uiLanguage,
		FLMUNICODE *			uzStr1,
		FLMUNICODE *			uzStr2,
		FLMINT *					piCmp);

	FLMBOOL errorIsFileCorrupt(
		RCODE						rc);

	static FLMBOOL _errorIsFileCorrupt(
		RCODE						rc)
	{
		F_DbSystem		dbSystem;

		return( dbSystem.errorIsFileCorrupt( rc));
	}

	const char * checkErrorToStr(
		eCorruptionType	eCorruption);

	FINLINE void freeMem(
		void **					ppMem)
	{
		f_free( ppMem);
	}

	FINLINE RCODE internalDbOpen(
		F_Database *			pDatabase,
		F_Db **					ppDb)
	{
		RCODE		rc = NE_SFLM_OK;
		F_Db *	pDb;

		if (RC_OK( rc = openDatabase( pDatabase, NULL, NULL, NULL,
				NULL, 0, TRUE, NULL, NULL, NULL, &pDb)))
		{
			*ppDb = (F_Db *)pDb;
		}
		return( rc);
	}

	static FINLINE FLMBOOL validBlockSize(
		FLMUINT			uiBlockSize)
	{
		if( uiBlockSize == 4096 || uiBlockSize == 8192)
		{
			return( TRUE);
		}

		return( FALSE);
	}

	RCODE waitToClose(
		const char *	pszDbPath);
	
private:

	// Methods

	RCODE readIniFile( void);

	RCODE setCacheParams(
		IF_IniFile *	pIniFile);

	void cleanup( void);
	
	FINLINE RCODE internalDbDup(
		F_Db *	pDb,
		F_Db **	ppDb)
	{
		return( openDatabase( pDb->m_pDatabase, NULL, NULL,
				NULL, NULL, 0, TRUE, NULL, NULL, NULL, ppDb));
	}

	void initFastBlockCheckSum( void);

	RCODE allocDb(
		F_Db **				ppDb,
		FLMBOOL				bInternalOpen);

	RCODE findDatabase(
		const char *		pszDbPath,
		const char *		pszDataDir,
		F_Database **		ppDatabase);

	RCODE checkDatabaseClosed(
		const char *		pszDbName,
		const char *		pszDataDir);

	RCODE allocDatabase(
		const char *		pszDbPath,
		const char *		pszDataDir,
		FLMBOOL				bTempDb,
		F_Database **		ppDatabase);

	RCODE openDatabase(
		F_Database *			pDatabase,
		const char *			pszDbPath,
		const char *			pszDataDir,
		const char *			pszRflDir,
		const char *			pszPassword,
		FLMUINT					uiOpenFlags,
		FLMBOOL					bInternalOpen,
		IF_RestoreClient *	pRestoreObj,
		IF_RestoreStatus *	pRestoreStatus,
		IF_FileHdl *			pLockFileHdl,
		F_Db **					pDb);

	RCODE copyDb(
		const char *			pszSrcDbName,
		const char *			pszSrcDataDir,
		const char *			pszSrcRflDir,
		const char *			pszDestDbName,
		const char *			pszDestDataDir,
		const char *			pszDestRflDir,
		IF_DbCopyStatus *		ifpStatus);

	static RCODE SQFAPI monitorThrd(
		IF_Thread *		pThread);
		
	static RCODE SQFAPI cacheCleanupThrd(
		IF_Thread *		pThread);

	static void checkNotUsedObjects( void);

	static FLMATOMIC			m_flmSysSpinLock;
	static FLMUINT				m_uiFlmSysStartupCount;

friend class F_Db;
friend class F_Database;
friend class F_DbRebuild;
friend class F_DbCheck;
};

// Supported text types

typedef enum
{
	SFLM_UNICODE_TEXT = 1,
	SFLM_UTF8_TEXT
} eSFlmTextType;

void flmGetDbBasePath(
	char *			pszBaseDbName,
	const char *	pszDbName,
	FLMUINT *		puiBaseDbNameLen);
	
/*------------------------------------------------------
	FLAIM Processing Hooks (call-backs)
-------------------------------------------------------*/

#define FLM_DATA_LEFT_TRUNCATED	0x10	// Data is left truncated
#define FLM_DATA_RIGHT_TRUNCATED	0x20	// Data is right truncated

RCODE flmReadStorageAsText(
	IF_IStream *		pIStream,
	FLMBYTE *			pucStorageData,
	FLMUINT				uiDataLen,
	eDataType			eDataTyp,
	void *				pvBuffer,
	FLMUINT 				uiBufLen,
	eSFlmTextType		eTextType,
	FLMUINT				uiMaxCharsToRead,
	FLMUINT				uiCharOffset,
	FLMUINT *			puiCharsRead,
	FLMUINT *			puiBufferBytesUsed);

RCODE flmReadStorageAsBinary(
	IF_IStream *		pIStream,
	void *				pvBuffer,
	FLMUINT 				uiBufLen,
	FLMUINT				uiByteOffset,
	FLMUINT *			puiBytesRead);

RCODE flmReadStorageAsNumber(
	IF_IStream *		pIStream,
	eDataType			eDataTyp,
	FLMUINT64 *			pui64Number,
	FLMBOOL *			pbNeg);

RCODE flmReadLine(
	IF_IStream *		pIStream,
	FLMBYTE *			pucBuffer,
	FLMUINT *			puiSize);

#define FLM_ENCRYPT_CHUNK_SIZE 512

/*****************************************************************************
Desc:
******************************************************************************/
class F_BTreeIStream : public IF_PosIStream
{
public:

	F_BTreeIStream()
	{
		m_pucBuffer = NULL;
		m_pBTree = NULL;
		m_bReleaseBTree = FALSE;
		reset();
	}

	virtual ~F_BTreeIStream()
	{
		reset();
	}

	FINLINE void reset( void)
	{
		m_pNextInPool = NULL;
		if( m_pBTree && m_bReleaseBTree)
		{
			m_pBTree->btClose();
			gv_SFlmSysData.pBtPool->btpReturnBtree( &m_pBTree);
			m_pBTree = NULL;
		}

		if( m_pucBuffer != &m_ucBuffer [0])
		{
			f_free( &m_pucBuffer);
		}

		m_pDb = NULL;
		m_uiTableNum = 0;
		m_ui64RowId = 0;
		m_pBTree = NULL;
		m_bReleaseBTree = FALSE;
		m_uiKeyLength = 0;
		m_uiStreamSize = 0;
		m_uiBufferBytes = 0;
		m_uiBufferOffset = 0;
		m_uiBufferStartOffset = 0;
		m_uiBufferSize = sizeof( m_ucBuffer);
		m_pucBuffer = &m_ucBuffer [0];
		m_ui32BlkAddr = 0;
		m_uiOffsetIndex = 0;
		m_uiDataLength = 0;
	}

	RCODE openStream(
		F_Db *			pDb,
		FLMUINT			uiTableNum,
		FLMUINT64		ui64RowId,
		FLMUINT32		ui32BlkAddr = 0,
		FLMUINT			uiOffsetIndex = 0);

	RCODE openStream(
		F_Db *			pDb,
		F_Btree *		pBTree,
		FLMUINT			uiFlags,
		FLMUINT			uiTableNum,
		FLMUINT64		ui64RowId,
		FLMUINT32		ui32BlkAddr = 0,
		FLMUINT			uiOffsetIndex = 0);

	FINLINE FLMUINT64 SQFAPI totalSize( void)
	{
		return( m_uiStreamSize);
	}

	FINLINE FLMUINT64 SQFAPI remainingSize( void)
	{
		return( m_uiStreamSize - (m_uiBufferStartOffset + m_uiBufferOffset));
	}

	FINLINE RCODE SQFAPI closeStream( void)
	{
		reset();
		return( NE_SFLM_OK);
	}

	RCODE SQFAPI positionTo(
		FLMUINT64		ui64Position);

	FINLINE FLMUINT64 SQFAPI getCurrPosition( void)
	{
		return( m_uiBufferStartOffset + m_uiBufferOffset);
	}

	RCODE SQFAPI read(
		void *			pvBuffer,
		FLMUINT			uiBytesToRead,
		FLMUINT *		puiBytesRead);

	FLMINT SQFAPI Release( void);
	
	FINLINE FLMUINT32 getBlkAddr( void)
	{
		return( m_ui32BlkAddr);
	}

	FINLINE FLMUINT getOffsetIndex( void)
	{
		return( m_uiOffsetIndex);
	}
	
private:

	F_BTreeIStream *	m_pNextInPool;
	F_Db *				m_pDb;
	F_Btree *			m_pBTree;
	FLMUINT				m_uiTableNum;
	FLMUINT64			m_ui64RowId;
	FLMUINT				m_uiStreamSize;
	FLMUINT				m_uiKeyLength;
	FLMUINT				m_uiBufferBytes;
	FLMUINT				m_uiBufferSize;
	FLMUINT				m_uiBufferOffset;
	FLMUINT				m_uiBufferStartOffset;
	FLMUINT				m_uiDataLength;
	FLMBYTE				m_ucBuffer[ FLM_ENCRYPT_CHUNK_SIZE];
	FLMBYTE *			m_pucBuffer;
	FLMUINT				m_uiOffsetIndex;
	FLMUINT32			m_ui32BlkAddr;
	FLMBOOL				m_bReleaseBTree;
	FLMBYTE				m_ucKey[ FLM_MAX_NUM_BUF_SIZE];
friend class F_Row;
friend class F_Db;
friend class F_BTreeIStreamPool;
};

/*===========================================================================
Desc: Pool manager for b-tree istream objects
===========================================================================*/
class F_BTreeIStreamPool : public F_Object
{
public:

	F_BTreeIStreamPool()
	{
		m_pFirstBTreeIStream = NULL;
		m_hMutex = F_MUTEX_NULL;
	}

	~F_BTreeIStreamPool();

	RCODE setup( void);

	RCODE allocBTreeIStream(
		F_BTreeIStream **	ppBTreeIStream);
		
	void insertBTreeIStream(
		F_BTreeIStream *	pBTreeIStream);
		
private:

	F_BTreeIStream *			m_pFirstBTreeIStream;
	F_MUTEX						m_hMutex;

friend class F_BTreeIStream;
};

typedef struct RSIxKeyTag
{
	FLMBYTE			pucRSKeyBuf[ SFLM_MAX_KEY_SIZE];
	FLMUINT			uiRSKeyLen;
	FLMBYTE			pucRSDataBuf[ SFLM_MAX_KEY_SIZE];
	FLMUINT			uiRSDataLen;
} RS_IX_KEY;

/******************************************************************************
Desc:
******************************************************************************/
class F_KeyCollector : public F_Object
{

public:

	F_KeyCollector(
		F_DbCheck *	pDbCheck)
	{
		m_pDbCheck = pDbCheck;
		m_ui64TotalKeys = 0;
	}

	~F_KeyCollector(){}

	RCODE addKey(
		F_Db *				pDb,
		F_INDEX *			pIndex,
		KREF_ENTRY *		pKref);

	FLMUINT64 getTotalKeys()
	{
		return m_ui64TotalKeys;
	}

private:

	F_DbCheck *				m_pDbCheck;
	FLMUINT64				m_ui64TotalKeys;
	
friend class F_DbCheck;
};

typedef struct
{
	FLMUINT64	ui64BytesUsed;
	FLMUINT64	ui64ElementCount;
	FLMUINT64 	ui64ContElementCount;
	FLMUINT64 	ui64ContElmBytes;
	FLMUINT		uiBlockCount;
	FLMINT		iErrCode;
	FLMUINT		uiNumErrors;
} BLOCK_INFO;

typedef struct
{
	FLMUINT64		ui64KeyCount;
	BLOCK_INFO		BlockInfo;
} LEVEL_INFO;

typedef struct
{
	FLMUINT			uiLfNum;
	eLFileType		eLfType;
	FLMUINT			uiRootBlk;
	FLMUINT			uiNumLevels;
	LEVEL_INFO *	pLevelInfo;
} LF_HDR;

/******************************************************************************
Desc:
******************************************************************************/
typedef struct State_Info
{
	F_Db *				pDb;
	FLMUINT32			ui32BlkAddress;
	FLMUINT32			ui32NextBlkAddr;
	FLMUINT32			ui32PrevBlkAddr;
	FLMUINT32			ui32LastChildAddr;
	FLMUINT				uiElmLastFlag;
	FLMUINT64			ui64KeyCount;
	FLMUINT64			ui64KeyRefs;
	FLMUINT				uiBlkType;
	FLMUINT				uiLevel;
	FLMUINT				uiRootLevel;
	FLMUINT				uiElmOffset;
	FLMBYTE *			pucElm;
	FLMUINT				uiElmLen;
	FLMUINT				uiElmKeyLenOffset;	// New
	FLMUINT				uiElmKeyOffset;		// New
	FLMBYTE *			pucElmKey;
	FLMUINT				uiElmKeyLen;
	FLMUINT				uiCurKeyLen;			// Used in Rebuild...
	FLMUINT				uiElmDataLenOffset;	// New
	FLMUINT				uiElmDataOffset;		// uiElmRecOffset;
	FLMUINT				uiElmDataLen;			// uiElmRecLen;
	FLMBYTE *			pucElmData;				// pucElmRec;
	FLMUINT				uiElmCounts;			// New
	FLMUINT				uiElmCountsOffset;	// New
	FLMUINT				uiElmOADataLenOffset;
	FLMUINT				uiElmOADataLen;
	FLMUINT64			ui64ElmNodeId;
	FLMBOOL				bValidKey;
	F_BLK_HDR *			pBlkHdr;
	BLOCK_INFO			BlkInfo;
	F_BtResultSet *	pNodeRS;
	F_BtResultSet *	pXRefRS;
	FLMUINT				uiCurrLf;
} STATE_INFO;

/******************************************************************************
Desc:
******************************************************************************/
typedef struct 
{
	FLMUINT		uiStartOfEntry;
	FLMUINT		uiEndOfEntry;
} BlkStruct;

/******************************************************************************
Desc:	DbCheck object for verifying the condition of the database.
******************************************************************************/
class F_DbCheck : public F_Object
{

public:
	F_DbCheck( void)
	{
		m_pXRefRS = NULL;
		m_pDb = NULL;
		m_pIndex = NULL;
		m_pTable = NULL;
		m_pLFile = NULL;
		m_pDbInfo = NULL;
		m_pBtPool = NULL;
		m_pResultSetDb = NULL;
		m_pRandGen = NULL;
		m_pDbCheckStatus = NULL;		
		f_memset( m_szResultSetDibName, 0, sizeof( m_szResultSetDibName));
		m_bPhysicalCorrupt = FALSE;
		m_bIndexCorrupt = FALSE;
		m_LastStatusRc = NE_SFLM_OK;
		m_uiFlags = 0;
		m_bStartedUpdateTrans = FALSE;
		m_puiIxArray = NULL;
		m_pIxRSet = NULL;
		m_bGetNextRSKey = FALSE;
		f_memset( &m_IxKey1, 0, sizeof( m_IxKey1));
		f_memset( &m_IxKey2, 0, sizeof( m_IxKey2));
		m_pCurrRSKey = NULL;
		m_pPrevRSKey = NULL;
		m_pBlkEntries = NULL;
		m_uiBlkEntryArraySize = 0;
	}

	~F_DbCheck();

	RCODE dbCheck(
		const char *			pszDbFileName,
		const char *			pszDataDir,
		const char *			pszRflDir,
		const char *			pszPassword,
		FLMUINT					uiFlags,
		F_DbInfo **				ppDbInfo,
		IF_DbCheckStatus *	pDbCheck);

private:

	FINLINE RCODE chkCallProgFunc( void)
	{
		if (m_pDbCheckStatus && RC_OK( m_LastStatusRc))
		{
			m_LastStatusRc = m_pDbCheckStatus->reportProgress( &m_Progress);
		}
		return( m_LastStatusRc);
	}

	RCODE chkReportError(
		FLMINT			iErrCode,
		FLMUINT			uiErrLocale,
		FLMUINT			uiErrLfNumber,
		FLMUINT			uiErrLfType,
		FLMUINT			uiErrBTreeLevel,
		FLMUINT			uiErrBlkAddress,
		FLMUINT			uiErrParentBlkAddress,
		FLMUINT			uiErrElmOffset,
		FLMUINT64		ui64ErrNodeId);

	FINLINE SFLM_PROGRESS_CHECK_INFO * getProgress( void)
	{
		return( &m_Progress);
	}

	RCODE getBtResultSet(
		F_BtResultSet **	ppBtRSet);
		
	RCODE createAndOpenResultSetDb( void);

	RCODE closeAndDeleteResultSetDb( void);

	RCODE getDictInfo( void);

	RCODE verifyBlkChain(
		BLOCK_INFO *		pBlkInfo,
		FLMUINT				uiLocale,
		FLMUINT				uiFirstBlkAddr,
		FLMUINT				uiBlkType,
		FLMBOOL *			pbStartOverRV);

	RCODE verifyLFHBlocks(
		FLMBOOL *			pbStartOverRV);

	RCODE verifyAvailList(
		FLMBOOL *			pbStartOverRV);

	RCODE blkRead(
		FLMUINT				uiBlkAddress,
		F_BLK_HDR **		ppBlkHdr,
		F_CachedBlock **	ppSCache,
		FLMINT *				piBlkErrCodeRV);

	RCODE verifySubTree(
		STATE_INFO *		pParentState,
		STATE_INFO *		pStateInfo,
		FLMUINT				uiBlkAddress,
		FLMBYTE **			ppucResetKey,
		FLMUINT				uiResetKeyLen,
		FLMUINT64			ui64ResetNodeId);

	RCODE verifyBTrees(
		FLMBOOL *			pbStartOverRV);

	RCODE setupLfTable();

	RCODE setupIxInfo( void);

	RCODE getLfInfo(
		LF_HDR *				pLogicalFile,
		LFILE *				pLFile);
		
	RCODE verifyNodePointers(
		STATE_INFO *		pStateInfo,
		FLMINT *				piErrCode);

	RCODE verifyDOChain(
		STATE_INFO *		pParentState,
		FLMUINT				uiBlkAddr,
		FLMINT *				piElmErrCode);

	RCODE chkGetNextRSKey( void);
		
	RCODE verifyIXRSet(
		STATE_INFO *		pStateInfo);

	RCODE resolveIXMissingKey(
		STATE_INFO *		pStateInfo);

	RCODE verifyComponentInDoc(
		ICD *					pIcd,
		FLMUINT				uiComponent,
		F_DataVector *		pKey,
		FLMBOOL *			pbInDoc);
		
	RCODE getKeySource(
		FLMBYTE *			pucKey,
		FLMUINT				uiKeyLen,
		FLMBOOL *			pbKeyInDoc,
		FLMBOOL *			pbKeyInIndex);

	RCODE resolveRSetMissingKey(
		STATE_INFO *		pStateInfo);

	RCODE chkVerifyKeyExists(
		FLMBYTE *			pucKey,
		FLMUINT				uiKeyLen,
		FLMBOOL *			pbFoundRV);
		
	RCODE addDelKeyRef(
		FLMBYTE *			pucKey,
		FLMUINT				uiKeyLen,
		FLMBOOL				bDelete);

	RCODE reportIxError(
		STATE_INFO *		pStateInfo,
		FLMINT				iErrCode,
		FLMBYTE *			pucErrKey,
		FLMUINT				uiErrKeyLen,
		FLMBOOL *			pbFixErrRV);

	RCODE startUpdate( void);

	RCODE keyToVector(
		FLMBYTE *			pucKey,
		FLMUINT				uiKeyLen,
		F_DataVector **	ppKeyRV);

	RCODE verifyIXRefs(
		STATE_INFO *	pStateInfo,
		FLMUINT64		ui64ResetNodeId);

	RCODE verifyBlockStructure(
		FLMUINT					uiBlockSize,
		F_BTREE_BLK_HDR *		pBlkHdr);

	RCODE chkEndUpdate( void);
		
	F_Db *							m_pDb;
	F_TABLE *						m_pTable;
	F_INDEX *						m_pIndex;
	LFILE *							m_pLFile;
	F_DbInfo *						m_pDbInfo;
	F_BtResultSet *				m_pXRefRS;
	F_BtPool *						m_pBtPool;
	IF_RandomGenerator *			m_pRandGen;
	char								m_szResultSetDibName [F_PATH_MAX_SIZE];
	F_Db *							m_pResultSetDb;
	IF_DbCheckStatus *			m_pDbCheckStatus;
	FLMBOOL							m_bPhysicalCorrupt;
	FLMBOOL							m_bIndexCorrupt;
	SFLM_PROGRESS_CHECK_INFO	m_Progress;
	RCODE								m_LastStatusRc;
	FLMUINT							m_uiFlags;
	FLMBOOL							m_bStartedUpdateTrans;
	FLMUINT *						m_puiIxArray;
	F_BtResultSet *				m_pIxRSet;
	FLMBOOL							m_bGetNextRSKey;
	RS_IX_KEY						m_IxKey1;
	RS_IX_KEY						m_IxKey2;
	RS_IX_KEY *						m_pCurrRSKey;
	RS_IX_KEY *						m_pPrevRSKey;
	BlkStruct *						m_pBlkEntries;
	FLMUINT							m_uiBlkEntryArraySize;
friend class F_DbInfo;
friend class F_KeyCollector;
};

// Check Error Codes

/*
**  WARNING:	ANY CHANGES MADE TO THE CHECK ERROR CODE DEFINES MUST BE
**					REFLECTED IN THE FlmCorruptStrings TABLE FOUND IN FLERRSTR.CPP
*/
#define FLM_BAD_CHAR							1
#define FLM_BAD_ASIAN_CHAR					2
#define FLM_BAD_CHAR_SET					3
#define FLM_BAD_TEXT_FIELD					4
#define FLM_BAD_NUMBER_FIELD				5
#define FLM_BAD_FIELD_TYPE					6
#define FLM_BAD_IX_DEF						7
#define FLM_MISSING_REQ_KEY_FIELD		8
#define FLM_BAD_TEXT_KEY_COLL_CHAR		9
#define FLM_BAD_TEXT_KEY_CASE_MARKER	10
#define FLM_BAD_NUMBER_KEY					11
#define FLM_BAD_BINARY_KEY					12
#define FLM_BAD_CONTEXT_KEY				13
#define FLM_BAD_KEY_FIELD_TYPE			14
//#define Not_Used_15						15
//#define Not_Used_16						16
//#define Not_Used_17						17
#define FLM_BAD_KEY_LEN						18
#define FLM_BAD_LFH_LIST_PTR				19
#define FLM_BAD_LFH_LIST_END				20
#define FLM_INCOMPLETE_NODE				21	// Not really an error.  Part of a field has been split across entries
#define FLM_BAD_BLK_END						22
#define FLM_KEY_COUNT_MISMATCH			23
#define FLM_REF_COUNT_MISMATCH			24
#define FLM_BAD_CONTAINER_IN_KEY			25
#define FLM_BAD_BLK_HDR_ADDR				26
#define FLM_BAD_BLK_HDR_LEVEL				27
#define FLM_BAD_BLK_HDR_PREV				28
/*
**  WARNING:	ANY CHANGES MADE TO THE CHECK ERROR CODE DEFINES MUST BE
**					REFLECTED IN THE FlmCorruptStrings TABLE FOUND IN FLERRSTR.CPP
*/
#define FLM_BAD_BLK_HDR_NEXT				29
#define FLM_BAD_BLK_HDR_TYPE				30
#define FLM_BAD_BLK_HDR_ROOT_BIT			31
#define FLM_BAD_BLK_HDR_BLK_END			32
#define FLM_BAD_BLK_HDR_LF_NUM			33
#define FLM_BAD_AVAIL_LIST_END			34
#define FLM_BAD_PREV_BLK_NEXT				35
#define FLM_BAD_FIRST_ELM_FLAG			36	// NOTE: This is only needed during rebuild
#define FLM_BAD_LAST_ELM_FLAG				37	// NOTE: This is only needed during rebuild
#define FLM_BAD_LEM							38
#define FLM_BAD_ELM_LEN						39
#define FLM_BAD_ELM_KEY_SIZE				40
#define FLM_BAD_ELM_KEY						41
#define FLM_BAD_ELM_KEY_ORDER				42
#define FLM_BAD_ELM_KEY_COMPRESS			43	// NOTE: 5.x keys are not compressed
#define FLM_BAD_CONT_ELM_KEY				44
#define FLM_NON_UNIQUE_FIRST_ELM_KEY	45
#define FLM_BAD_ELM_OFFSET					46
#define FLM_BAD_ELM_INVALID_LEVEL		47
#define FLM_BAD_ELM_FLD_NUM				48
#define FLM_BAD_ELM_FLD_LEN				49
#define FLM_BAD_ELM_FLD_TYPE				50
#define FLM_BAD_ELM_END						51
#define FLM_BAD_PARENT_KEY					52
#define FLM_BAD_ELM_DOMAIN_SEN			53
#define FLM_BAD_ELM_BASE_SEN				54
#define FLM_BAD_ELM_IX_REF					55
#define FLM_BAD_ELM_ONE_RUN_SEN			56
#define FLM_BAD_ELM_DELTA_SEN				57
#define FLM_BAD_ELM_DOMAIN					58
/*
**  WARNING:	ANY CHANGES MADE TO THE CHECK ERROR CODE DEFINES MUST BE
**					REFLECTED IN THE FlmCorruptStrings TABLE FOUND IN FLERRSTR.CPP
*/
#define FLM_BAD_LAST_BLK_NEXT				59
#define FLM_BAD_FIELD_PTR					60
#define FLM_REBUILD_REC_EXISTS			61
#define FLM_REBUILD_KEY_NOT_UNIQUE		62
#define FLM_NON_UNIQUE_ELM_KEY_REF		63
#define FLM_OLD_VIEW							64
#define FLM_COULD_NOT_SYNC_BLK			65
#define FLM_IX_REF_REC_NOT_FOUND			66
#define FLM_IX_KEY_NOT_FOUND_IN_REC		67
#define FLM_KEY_NOT_IN_KEY_REFSET		68
#define FLM_BAD_BLK_CHECKSUM				69
#define FLM_BAD_LAST_DRN					70
#define FLM_BAD_FILE_SIZE					71
#define FLM_BAD_FIRST_LAST_ELM_FLAG		72
#define FLM_BAD_DATE_FIELD					73
#define FLM_BAD_TIME_FIELD					74
#define FLM_BAD_TMSTAMP_FIELD				75
#define FLM_BAD_DATE_KEY					76
#define FLM_BAD_TIME_KEY					77
#define FLM_BAD_TMSTAMP_KEY				78
#define FLM_BAD_BLOB_FIELD					79

/*
**  WARNING:	ANY CHANGES MADE TO THE CHECK ERROR CODE DEFINES MUST BE
**					REFLECTED IN THE FlmCorruptStrings TABLE FOUND IN FLERRSTR.CPP
*/

#define FLM_BAD_PCODE_IXD_TBL				80
#define FLM_NODE_QUARANTINED				81
#define FLM_BAD_BLK_TYPE					82
#define FLM_BAD_ELEMENT_CHAIN				83
#define FLM_BAD_ELM_EXTRA_DATA			84
#define FLM_BAD_BLOCK_STRUCTURE			85
#define FLM_BAD_ROOT_PARENT				86
#define FLM_BAD_ROOT_LINK					87
#define FLM_BAD_PARENT_LINK				88
#define FLM_BAD_INVALID_ROOT				89
#define FLM_BAD_FIRST_CHILD_LINK			90
#define FLM_BAD_LAST_CHILD_LINK			91
#define FLM_BAD_PREV_SIBLING_LINK		92
#define FLM_BAD_NEXT_SIBLING_LINK		93
#define FLM_BAD_ANNOTATION_LINK			94
#define FLM_UNSUPPORTED_NODE_TYPE		95
#define FLM_BAD_INVALID_NAME_ID			96
#define FLM_BAD_INVALID_PREFIX_ID		97
#define FLM_BAD_DATA_BLOCK_COUNT			98
#define FLM_BAD_AVAIL_SIZE					99
#define FLM_BAD_NODE_TYPE					100
#define FLM_BAD_CHILD_ELM_COUNT			101
#define FLM_NUM_CORRUPT_ERRORS			101

/*
**  WARNING:	ANY CHANGES MADE TO THE CHECK ERROR CODE DEFINES MUST BE
**					REFLECTED IN THE FlmCorruptStrings TABLE FOUND IN FLERRSTR.CPP
*/

class F_DbInfo : public F_Object
{
public:

	F_DbInfo()
	{
		m_uiLogicalCorruptions = 0;
		m_uiLogicalRepairs = 0;
		m_ui64FileSize = 0;
		m_uiNumIndexes = 0;
		m_uiNumTables = 0;
		m_uiNumLogicalFiles = 0;
		m_pLogicalFiles = NULL;
		f_memset( &m_dbHdr, 0, sizeof( m_dbHdr));
		f_memset( &m_AvailBlocks, 0, sizeof( m_AvailBlocks));
		f_memset( &m_LFHBlocks, 0, sizeof( m_LFHBlocks));
	}

	virtual ~F_DbInfo()
	{
		freeLogicalFiles();
	}

	FINLINE void freeLogicalFiles( void)
	{
		FLMUINT	uiLoop;
		
		if (m_pLogicalFiles)
		{
			for (uiLoop = 0; uiLoop < m_uiNumLogicalFiles; uiLoop++)
			{
				if (m_pLogicalFiles [uiLoop].pLevelInfo)
				{
					f_free( &m_pLogicalFiles [uiLoop].pLevelInfo);
				}
			}
			f_free( &m_pLogicalFiles);
		}
		m_uiNumLogicalFiles = 0;
		m_uiNumIndexes = 0;
		m_uiNumTables = 0;
	}
	
	FINLINE FLMUINT getNumTables( void)
	{
		return( m_uiNumTables);
	}

	FINLINE FLMUINT getNumIndexes( void)
	{
		return( m_uiNumIndexes);
	}

	FINLINE FLMUINT getNumLogicalFiles( void)
	{
		return( m_uiNumLogicalFiles);
	}

	FINLINE FLMUINT64 getFileSize( void)
	{
		return( m_ui64FileSize);
	}

	FINLINE SFLM_DB_HDR * getDbHdr( void)
	{
		return( &m_dbHdr);
	}

	FINLINE void getAvailBlockStats(
		FLMUINT64 *		pui64BytesUsed,
		FLMUINT *		puiBlockCount,
		FLMINT *			piLastError,
		FLMUINT *	puiNumErrors)
	{
		*pui64BytesUsed = m_LFHBlocks.ui64BytesUsed;
		*puiBlockCount = m_AvailBlocks.uiBlockCount;
		*piLastError = m_AvailBlocks.iErrCode;
		*puiNumErrors = m_AvailBlocks.uiNumErrors;
	}

	FINLINE void getLFHBlockStats(
		FLMUINT64 *		pui64BytesUsed,
		FLMUINT *		puiBlockCount,
		FLMINT *			piLastError,
		FLMUINT *	puiNumErrors)
	{
		*pui64BytesUsed = m_LFHBlocks.ui64BytesUsed;
		*puiBlockCount = m_LFHBlocks.uiBlockCount;
		*piLastError = m_LFHBlocks.iErrCode;
		*puiNumErrors = m_LFHBlocks.uiNumErrors;
	}

	void getBTreeInfo(
		FLMUINT			uiNthLogicalFile,
		FLMUINT *		puiLfNum,
		eLFileType *	peLfType,
		FLMUINT *		puiRootBlkAddress,
		FLMUINT *		puiNumLevels);

	void getBTreeBlockStats(
		FLMUINT			uiNthLogicalFile,
		FLMUINT			uiLevel,
		FLMUINT64 *		pui64KeyCount,
		FLMUINT64 *		pui64BytesUsed,
		FLMUINT64 *		pui64ElementCount,
		FLMUINT64 *		pui64ContElementCount,
		FLMUINT64 *		pui64ContElmBytes,
		FLMUINT *		puiBlockCount,
		FLMINT *			piLastError,
		FLMUINT *		puiNumErrors);

private:

	FLMUINT							m_uiLogicalCorruptions;
	FLMUINT							m_uiLogicalRepairs;
	FLMUINT64						m_ui64FileSize;
	FLMUINT							m_uiNumIndexes;
	FLMUINT							m_uiNumTables;
	FLMUINT							m_uiNumLogicalFiles;
	LF_HDR *							m_pLogicalFiles;
	SFLM_DB_HDR						m_dbHdr;
	BLOCK_INFO						m_AvailBlocks;
	BLOCK_INFO						m_LFHBlocks;
friend class F_DbCheck;
};

#define REBUILD_BLK_SIZE				(1024 * 50)
#define REBUILD_RSET_ENTRY_SIZE		21

/*=============================================================================
Desc: Class to rebuild a broken database.  This class is used by
		F_DbSystem::dbRebuild()
=============================================================================*/
class F_DbRebuild : public F_Object
{
public:

	// Constructor and Destructor

	F_DbRebuild( void)
	{
		m_pDb = NULL;
		m_pSFileHdl = NULL;
	}

	~F_DbRebuild()
	{
	}

	RCODE dbRebuild(
		const char *				pszSourceDbPath,
		const char *				pszSourceDataDir,
		const char *				pszDestDbPath,
		const char *				pszDestDataDir,
		const char *				pszDestRflDir,
		const char *				pDictPath,
		const char *				pszPassword,
		SFLM_CREATE_OPTS *		pCreateOpts,
		FLMUINT64 *					pui64TotNodes,
		FLMUINT64 *					pui64NodesRecov,
		FLMUINT64 *					pui64QuarantinedNodes,
		IF_DbRebuildStatus *		pRebuildStatus);

	FINLINE FLMUINT getBlockSize( void)
	{
		return( m_dbHdr.ui16BlockSize);
	}

	FINLINE RCODE reportStatus(
		FLMBOOL			bForce = FALSE)
	{
		RCODE		rc = NE_SFLM_OK;

		if( m_pRebuildStatus)
		{
			FLMUINT		uiCurrentTime = FLM_GET_TIMER();
			FLMUINT		uiElapTime = FLM_ELAPSED_TIME( uiCurrentTime, m_uiLastStatusTime);

			uiElapTime = FLM_TIMER_UNITS_TO_SECS( uiElapTime);

			if( bForce || uiElapTime >= 1)
			{
				m_uiLastStatusTime = uiCurrentTime;
				m_callbackData.bStartFlag = FALSE;
				if( RC_BAD( rc = m_pRebuildStatus->reportRebuild( &m_callbackData)))
				{
					m_cbrc = rc;
					goto Exit;
				}
			}
		}

	Exit:

		return( rc);
	}

private:

	RCODE rebuildDatabase( void);

	RCODE recoverNodes(
		FLMBOOL				bRecoverDictionary);

	FINLINE void buildRSetEntry(
		FLMBYTE		ucPrefix,
		FLMUINT		uiCollection,
		FLMUINT64	ui64NodeId,
		FLMUINT		uiBlockAddr,
		FLMUINT		uiElmNumber,
		FLMBYTE *	pucBuffer)
	{
		pucBuffer[ 0] = ucPrefix;
		
		UD2FBA( (FLMUINT32)uiCollection, &pucBuffer[ 1]);
		U642FBA( ui64NodeId, &pucBuffer[ 5]);
		UD2FBA( (FLMUINT32)uiBlockAddr, &pucBuffer[ 13]);
		UD2FBA( (FLMUINT32)uiElmNumber, &pucBuffer[ 17]);
	}

	FINLINE void extractRSetEntry(
		FLMBYTE *	pucBuffer,
		FLMUINT *	puiCollection,
		FLMUINT64 *	pui64NodeId,
		FLMUINT *	puiBlockAddr,
		FLMUINT *	puiElmNumber)
	{
		if( puiCollection)
		{
			*puiCollection = FB2UD( &pucBuffer[ 1]);
		}

		if( pui64NodeId)
		{
			*pui64NodeId = FB2U64( &pucBuffer[ 5]);
		}

		if( puiBlockAddr)
		{
			*puiBlockAddr = FB2UD( &pucBuffer[ 13]);
		}

		if( puiElmNumber)
		{
			*puiElmNumber = FB2UD( &pucBuffer[ 17]);
		}
	}

	RCODE recoverTree(
		F_RebuildRowIStream *	pIStream,
		IF_ResultSet *				pNonRootRSet,
		F_Row *						pRecovCachedNode,
		FLMBYTE *					pucNodeIV);

	FINLINE RCODE reportCorruption(
		eCorruptionType	eCorruption,
		FLMUINT				uiErrBlkAddress,
		FLMUINT				uiErrElmOffset,
		FLMUINT64			ui64ErrRowId)
	{
		RCODE		rc;

		if( m_pRebuildStatus)
		{
			m_corruptInfo.eCorruption = eCorruption;
			m_corruptInfo.uiErrBlkAddress = uiErrBlkAddress;
			m_corruptInfo.uiErrElmOffset = uiErrElmOffset;
			m_corruptInfo.ui64ErrRowId = ui64ErrRowId;
			rc = m_pRebuildStatus->reportRebuildErr( &m_corruptInfo);
			m_corruptInfo.eCorruption = SFLM_NO_CORRUPTION;
			return( rc);
		}
		
		return( NE_SFLM_OK);
	}

	RCODE determineBlkSize(
		FLMUINT *			puiBlkSizeRV);

	F_Db *						m_pDb;
	F_SuperFileHdl *			m_pSFileHdl;
	IF_DbRebuildStatus *		m_pRebuildStatus;
	FLMBOOL						m_bBadHeader;
	FLMUINT						m_uiLastStatusTime;
	SFLM_DB_HDR					m_dbHdr;
	SFLM_CREATE_OPTS			m_createOpts;
	SFLM_REBUILD_INFO			m_callbackData;
	SFLM_CORRUPT_INFO			m_corruptInfo;
	RCODE							m_cbrc;

friend class F_RebuildRowIStream;
};

RCODE chkBlkRead(
	F_DbInfo *			pDbInfo,
	FLMUINT				uiBlkAddress,
	F_BLK_HDR **		ppBlkHdr,
	F_CachedBlock **	ppSCache,
	FLMINT *				piBlkErrCodeRV);

FLMINT flmCompareKeys(
	FLMBYTE *	pBuf1,
	FLMUINT		uiBuf1Len,
	FLMBYTE *	pBuf2,
	FLMUINT		uiBuf2Len);

void flmInitReadState(
	STATE_INFO *	pStateInfo,
	FLMBOOL *		pbStateInitialized,
	FLMUINT			uiVersionNum,
	F_Db *			pDb,
	LF_HDR *			pLogicalFile,
	FLMUINT			uiLevel,
	FLMUINT			uiBlkType,
	FLMBYTE *		pucKeyBuffer);

FLMINT flmVerifyBlockHeader(
	STATE_INFO *	pStateInfo,
	BLOCK_INFO *	pBlockInfoRV,
	FLMUINT			uiBlockSize,
	FLMUINT			uiExpNextBlkAddr,
	FLMUINT			uiExpPrevBlkAddr,
	FLMBOOL			bCheckEOF);

RCODE flmVerifyElement(
	STATE_INFO *	pStateInfo,
	LFILE *			pLFile,
	F_INDEX *		pIndex,
	FLMINT *			piErrCode);

void getEntryInfo(
	F_BTREE_BLK_HDR *		pBlkHdr,
	FLMUINT					uiOffset,
	FLMBYTE **				ppucElm,
	FLMUINT *				puiElmLen,
	FLMUINT *				puiElmKeyLen,
	FLMUINT *				puiElmDataLen,
	FLMBYTE **				ppucElmKey,
	FLMBYTE **				ppucElmData);
	
FINLINE RCODE F_RowCacheMgr::makeWriteCopy(
	F_Db *	pDb,
	F_Row **	ppRow)
{
	if ((*ppRow)->getLowTransId() < pDb->m_ui64CurrTransID)
	{
		return( gv_SFlmSysData.pRowCacheMgr->_makeWriteCopy( pDb, ppRow));
	}
	
	return( NE_SFLM_OK);
}

/****************************************************************************
Desc:
*****************************************************************************/
class SQFEXP F_SuperFileClient : public IF_SuperFileClient
{
public:

	F_SuperFileClient();
	
	virtual ~F_SuperFileClient();
	
	RCODE setup(
		const char *			pszCFileName,
		const char *			pszDataDir);
	
	FLMUINT SQFAPI getFileNumber(
		FLMUINT					uiBlockAddr);
		
	FLMUINT SQFAPI getFileOffset(
		FLMUINT					uiBlockAddr);
		
	FLMUINT SQFAPI getBlockAddress(
		FLMUINT					uiFileNumber,
		FLMUINT					uiFileOffset);
			
	RCODE SQFAPI getFilePath(
		FLMUINT					uiFileNumber,
		char *					pszPath);
		
	FLMUINT64 SQFAPI getMaxFileSize( void);

	static void bldSuperFileExtension(
		FLMUINT					uiFileNum,
		char *					pszFileExtension);
		
private:

	char *						m_pszCFileName;
	char *						m_pszDataFileBaseName;
	FLMUINT						m_uiExtOffset;
	FLMUINT						m_uiDataExtOffset;
};

// More includes

#include "sqlquery.h"
#include "fscursor.h"

#endif // FLAIMSYS_H
