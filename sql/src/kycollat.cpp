//------------------------------------------------------------------------------
// Desc:	Index collation routines
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

#include "flaimsys.h"

FSTATIC RCODE KYFormatUTF8Text(
	IF_PosIStream *	pIStream,
	FLMUINT				uiFlags,
	FLMUINT				uiCompareRules,
	F_DynaBuf *			pDynaBuf);

/****************************************************************************
Desc:	Build a collated key value piece.
****************************************************************************/
RCODE KYCollateValue(
	FLMBYTE *			pucDest,
	FLMUINT *			puiDestLen,
	IF_PosIStream *	pIStream,
	FLMUINT				uiDataType,
	FLMUINT				uiFlags,
	FLMUINT				uiCompareRules,
	FLMUINT				uiLimit,
	FLMUINT *			puiCollationLen,
	FLMUINT *			puiLuLen,
	FLMUINT				uiLanguage,
	FLMBOOL				bFirstSubstring,
	FLMBOOL				bDataTruncated,
	FLMBOOL *			pbDataTruncated,
	FLMBOOL *			pbOriginalCharsLost)
{
	RCODE						rc = NE_SFLM_OK;
	FLMUINT					uiDestLen;
	IF_BufferIStream *	pBufferIStream = NULL;
	FLMUINT					uiCharLimit;
	FLMUINT					uiLength;
	FLMBYTE *				pucTmpDest;
	FLMUINT					uiBytesRead;
	FLMBOOL					bHaveData = TRUE;
	FLMUNICODE				uChar;
	FLMBYTE					ucDynaBuf[ 64];
	F_DynaBuf				dynaBuf( ucDynaBuf, sizeof( ucDynaBuf));

	if (puiLuLen)
	{
		*puiLuLen = 0;
	}

	if ((uiDestLen = *puiDestLen) == 0)
	{
		rc = RC_SET( NE_SFLM_KEY_OVERFLOW);
		goto Exit;
	}

	if (uiDataType != SFLM_STRING_TYPE)
	{
		if( !pIStream->remainingSize())
		{
			bHaveData = FALSE;
		}
	}
	else
	{
		FLMUINT64	ui64SavePosition = pIStream->getCurrPosition();

		if( RC_BAD( rc = f_readUTF8CharAsUnicode( 
			pIStream, &uChar)))
		{
			if (rc == NE_SFLM_EOF_HIT)
			{
				bHaveData = FALSE;
				rc = NE_SFLM_OK;
			}
			else
			{
				goto Exit;
			}
		}

		if( RC_BAD( rc = pIStream->positionTo( ui64SavePosition)))
		{
			goto Exit;
		}

		// The text is expected to be 0-terminated UTF-8

		if ((uiFlags & ICD_ESC_CHAR) ||
			 (uiCompareRules &
				(FLM_COMP_COMPRESS_WHITESPACE |
				 FLM_COMP_NO_WHITESPACE |
				 FLM_COMP_NO_UNDERSCORES |
				 FLM_COMP_NO_DASHES |
				 FLM_COMP_WHITESPACE_AS_SPACE |
				 FLM_COMP_IGNORE_LEADING_SPACE |
				 FLM_COMP_IGNORE_TRAILING_SPACE)))
		{
			dynaBuf.truncateData( 0);
			if (RC_BAD( rc = KYFormatUTF8Text( pIStream,
					uiFlags, uiCompareRules, &dynaBuf)))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = FlmAllocBufferIStream( &pBufferIStream)))
			{
				goto Exit;
			}
			
			if (RC_BAD( rc = pBufferIStream->openStream( 
				(const char *)dynaBuf.getBufferPtr(), dynaBuf.getDataLength())))
			{
				goto Exit;
			}
			pIStream = pBufferIStream;
		}

		uiCharLimit = uiLimit ? uiLimit : ICD_DEFAULT_LIMIT;

		if( (uiLanguage >= FLM_FIRST_DBCS_LANG ) && 
			 (uiLanguage <= FLM_LAST_DBCS_LANG))
		{
			if( RC_BAD( rc = f_asiaUTF8ToColText( pIStream, pucDest, &uiDestLen,
								(uiCompareRules & FLM_COMP_CASE_INSENSITIVE)
								? TRUE
								: FALSE,
								puiCollationLen, puiLuLen,
								uiCharLimit, bFirstSubstring,
								bDataTruncated, pbDataTruncated)))
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = flmUTF8ToColText( pIStream, pucDest, &uiDestLen,
								(uiCompareRules & FLM_COMP_CASE_INSENSITIVE)
								? TRUE
								: FALSE,
								puiCollationLen, puiLuLen,
								uiLanguage, uiCharLimit, bFirstSubstring,
								bDataTruncated,
								pbOriginalCharsLost, pbDataTruncated)))
			{
				goto Exit;
			}
		}
	}

	// TRICKY: uiDestLen could be set to zero if text and no value.

	if (!bHaveData || !uiDestLen)
	{
		uiDestLen = 0;
		goto Exit;
	}

 	switch (uiDataType)
	{
		case SFLM_STRING_TYPE:
			break;

		case SFLM_NUMBER_TYPE:
		{
			FLMBYTE	ucTmpBuf [FLM_MAX_NUM_BUF_SIZE];
			
			uiLength = (FLMUINT)pIStream->remainingSize();
			
			flmAssert( uiLength <= sizeof( ucTmpBuf));

			if (RC_BAD( rc = pIStream->read( ucTmpBuf, uiLength, &uiBytesRead)))
			{
				goto Exit;
			}
			flmAssert( uiBytesRead == uiLength);
			if (RC_BAD( rc = flmStorageNum2CollationNum( ucTmpBuf,
										uiBytesRead, pucDest, &uiDestLen)))
			{
				goto Exit;
			}
			break;
		}

		case SFLM_BINARY_TYPE:
		{
			uiLength = (FLMUINT)pIStream->remainingSize();
			pucTmpDest = pucDest;

			if (uiLength >= uiLimit)
			{
				uiLength = uiLimit;
				bDataTruncated = TRUE;
			}

			// We don't want any single key piece to "pig out" more
			// than 256 bytes of the key

			if (uiDestLen > 256)
			{
				uiDestLen = 256;
			}

			if (uiLength > uiDestLen)
			{

				// Compute length so will not overflow

				uiLength = uiDestLen;
				bDataTruncated = TRUE;
			}
			else
			{
				uiDestLen = uiLength;
			}

			// Store as is.

			if (RC_BAD( rc = pIStream->read( pucTmpDest, uiDestLen, &uiBytesRead)))
			{
				goto Exit;
			}

			if (bDataTruncated && pbDataTruncated)
			{
				*pbDataTruncated = TRUE;
			}
			break;
		}

		default:
		{
			rc = RC_SET( NE_SFLM_CANNOT_INDEX_DATA_TYPE);
			break;
		}
	}

Exit:

	if( pBufferIStream)
	{
		pBufferIStream->Release();
	}

	*puiDestLen = uiDestLen;
	return( rc);
}

/****************************************************************************
Desc:		Format text removing leading and trailing spaces.  Treat
			underscores as spaces.  As options, remove all spaces and dashes.
Ret:		NE_SFLM_OK always.  WIll truncate so text will fill SFLM_MAX_KEY_SIZE.
			Allocate 8 more than SFLM_MAX_KEY_SIZE for psDestBuf.
Visit:	Pass in uiLimit and pass back a truncated flag when the
			string is truncated.  This was not done because we will have
			to get the exact truncated count that is done in f_tocoll.cpp
			and that could introduce some bugs.
****************************************************************************/
FSTATIC RCODE KYFormatUTF8Text(
	IF_PosIStream *	pIStream,
	FLMUINT				uiFlags,					// ICD flags
	FLMUINT				uiCompareRules,		// ICD compare rules
	F_DynaBuf *			pDynaBuf)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiFirstSpaceCharPos = FLM_MAX_UINT;
	FLMUNICODE	uChar;
	FLMUINT		uiSize;
	FLMUINT		uiStrSize = 0;
	FLMBYTE *	pucTmp;

	if( !pIStream->remainingSize())
	{
		pDynaBuf->truncateData( 0);
		goto Exit;
	}

	for (;;)
	{
		if (RC_BAD( rc = f_readUTF8CharAsUnicode( pIStream, &uChar)))
		{
			if (rc == NE_SFLM_EOF_HIT)
			{
				rc = NE_SFLM_OK;
				break;
			}
			goto Exit;
		}
		if ((uChar = f_convertChar( uChar, uiCompareRules)) == 0)
		{
			continue;
		}

		if (uChar == ASCII_SPACE)
		{
			if (uiCompareRules &
				 (FLM_COMP_COMPRESS_WHITESPACE |
				  FLM_COMP_IGNORE_TRAILING_SPACE))
			{
				
				// Remember the position of the first space.
				// When we come to the end of the spaces, we may reset
				// the size to compress out spaces if necessary.  Or,
				// we may opt to get rid of all of them.

				if (uiFirstSpaceCharPos == FLM_MAX_UINT)
				{
					uiFirstSpaceCharPos = uiStrSize;
				}
			}
		}
		else
		{
			
			// Once we hit a non-space character, we can turn off the
			// ignore leading spaces flag.
			
			uiCompareRules &= (~(FLM_COMP_IGNORE_LEADING_SPACE));
			
			// See if we need to compress spaces.
			
			if (uiFirstSpaceCharPos != FLM_MAX_UINT)
			{
				
				// Output exactly one ASCII_SPACE character if we are compressing
				// spaces.  If we are not compressing spaces, then the only other
				// way uiFirstSpaceCharPos would have been set is if we were
				// ignoring trailing spaces.  In that case, since the spaces
				// were not trailing spaces, we need to leave them as is.
				
				if (uiCompareRules & FLM_COMP_COMPRESS_WHITESPACE)
				{
					
					// A space will already have been encoded into the string.
					// Since we know a space takes exactly one byte in the UTF8
					// space, we can simply set our pointer one byte past where
					// the last non-space character was found.
					
					uiStrSize = uiFirstSpaceCharPos + 1;
					pDynaBuf->truncateData( uiStrSize);
				}
				uiFirstSpaceCharPos = FLM_MAX_UINT;
			}
			
			// If we are allowing escaped characters, backslash is treated
			// always as an escape character.  Whatever follows the
			// backslash is the character we need to process.

			if (uChar == ASCII_BACKSLASH && (uiFlags & ICD_ESC_CHAR))
			{
				if (RC_BAD( rc = f_readUTF8CharAsUnicode( pIStream, &uChar)))
				{
					if (rc == NE_SFLM_EOF_HIT)
					{
						rc = NE_SFLM_OK;
					}
					else
					{
						goto Exit;
					}
				}
			}
		}
		
		// Output the character - need at most three bytes
		
		if (RC_BAD( rc = pDynaBuf->allocSpace( 3, (void **)&pucTmp)))
		{
			goto Exit;
		}
		uiSize = 3;
		if (RC_BAD( rc = f_uni2UTF8( uChar, pucTmp, &uiSize)))
		{
			goto Exit;
		}
		uiStrSize += uiSize;
		pDynaBuf->truncateData( uiStrSize);
	}

	// If uiFirstSpaceCharPos != FLM_MAX_UINT, it means that all of the
	// characters at the end of the string were spaces.  If we
	// are ignoring trailing spaces, we need to truncate the string so
	// they will be ignored.  Otherwise, we need to compress them into
	// a single space.
	
	if (uiFirstSpaceCharPos != FLM_MAX_UINT)
	{
		if (uiCompareRules & FLM_COMP_IGNORE_TRAILING_SPACE)
		{
			uiStrSize = uiFirstSpaceCharPos;
		}
		else
		{
			flmAssert( uiCompareRules & FLM_COMP_COMPRESS_WHITESPACE);
			
			// A space will already have been encoded into the string.
			// Since we know a space takes exactly one byte in the UTF8
			// space, we can simply set our pointer one byte past where
			// the last non-space character was found.

			uiStrSize = uiFirstSpaceCharPos + 1;
		}
		pDynaBuf->truncateData( uiStrSize);
	}
	
	// Terminate the UTF-8 string
	
	if (RC_BAD( rc = pDynaBuf->appendByte( 0)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}
