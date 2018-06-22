//------------------------------------------------------------------------------
// Desc:	This file contains the routines which allow editing and entering
//			of data.
// Tabs:	3
//
// Copyright (c) 1992-2007 Novell, Inc. All Rights Reserved.
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

/********************************************************************
Desc:	Get a number from the user
*********************************************************************/
FLMBOOL ViewGetNum(
	const char *	pszPrompt,
	void *			pvNum,
	FLMBOOL			bEnterHexFlag,
	FLMUINT			uiNumBytes,
	FLMUINT64		ui64MaxValue,
	FLMBOOL *		pbValEntered)
{
	FLMBOOL			bOk = FALSE;
	char				szTempBuf[ 20];
	FLMUINT     	uiLoop;
	FLMUINT			uiChar;
	FLMBOOL			bGetOK;
	FLMUINT64		ui64Num;
	FLMUINT			uiMaxDigits;
	FLMUINT			uiNumCols;
	FLMUINT			uiNumRows;

	f_conGetScreenSize( &uiNumCols, &uiNumRows);

	if (bEnterHexFlag)
	{
		uiMaxDigits = uiNumBytes * 2;
	}
	else
	{
		uiMaxDigits = (uiNumBytes == 8
						  ? 20
						  : (uiNumBytes == 4
							  ? 10
							  : (uiNumBytes == 2
								  ? 5
								  : 3)));
	}

	for (;;)
	{
		bGetOK = TRUE;
		f_conSetBackFore( FLM_BLACK, FLM_WHITE);
		f_conClearScreen( 0, uiNumRows - 2);
		ViewAskInput( pszPrompt, szTempBuf, sizeof( szTempBuf));
		f_conSetBackFore( FLM_BLACK, FLM_WHITE);
		f_conClearScreen( 0, uiNumRows - 2);
		if (f_stricmp( szTempBuf, "\\") == 0)
		{
			*pbValEntered = FALSE;
			goto Exit;
		}
		if( !szTempBuf[ 0])
		{
			*pbValEntered = FALSE;
			bOk = TRUE;
			goto Exit;
		}
		uiLoop = 0;
		ui64Num = 0;
		while (szTempBuf[ uiLoop] && uiLoop < uiMaxDigits)
		{
			uiChar = szTempBuf[ uiLoop];
			if (bEnterHexFlag)
			{
				ui64Num <<= 4;
				if (uiChar >= '0' && uiChar <= '9')
				{
					ui64Num += (FLMUINT64)(uiChar - '0');
				}
				else if (uiChar >= 'a' && uiChar <= 'f')
				{
					ui64Num += (FLMUINT64)(uiChar - 'a' + 10);
				}
				else if (uiChar >= 'A' && uiChar <= 'F')
				{
					ui64Num += (FLMUINT64)(uiChar - 'A' + 10);
				}
				else
				{
					ViewShowError(
						"Illegal digit in number - must be hex digits");
					bGetOK = FALSE;
					break;
				}
			}
			else if (uiChar < '0' || uiChar > '9')
			{
				ViewShowError(
					"Illegal digit in number - must be 0 through 9");
				bGetOK = FALSE;
				break;
			}
			else
			{
				if (ui64MaxValue / 10 < ui64Num)
				{
					ViewShowError( "Number is too large");
					bGetOK = FALSE;
					break;
				}
				else
				{
					ui64Num *= 10;
					if (ui64MaxValue - (FLMUINT64)(uiChar - '0') < ui64Num)
					{
						ViewShowError( "Number is too large");
						bGetOK = FALSE;
						break;
					}
					else
					{
						ui64Num += (FLMUINT64)(uiChar - '0');
					}
				}
			}
			uiLoop++;
		}
		if (bGetOK)
		{
			if (uiNumBytes == 8)
			{
				*((FLMUINT64 *)(pvNum)) = ui64Num;
			}
			else if (uiNumBytes == 4)
			{
				*((FLMUINT32 *)(pvNum)) = (FLMUINT32)ui64Num;
			}
			else if( uiNumBytes == 2)
			{
				*((FLMUINT16 *)(pvNum)) = (FLMUINT16)ui64Num;
			}
			else
			{
				*((FLMBYTE *)(pvNum)) = (FLMBYTE)ui64Num;
			}
			*pbValEntered = TRUE;
			bOk = TRUE;
			goto Exit;
		}
	}

Exit:

	return( bOk);
}

/********************************************************************
Desc:	Edit a number
*********************************************************************/
FLMBOOL ViewEditNum(
	void *		pvNum,
	FLMBOOL		bEnterHexFlag,
	FLMUINT		uiNumBytes,
	FLMUINT64	ui64MaxValue
	)
{
	char		szPrompt[ 80];
	FLMBOOL	bValEntered;

	f_strcpy( szPrompt, "Enter Value (in ");
	if (bEnterHexFlag)
	{
		f_strcpy( &szPrompt[ f_strlen( szPrompt)], "hex): ");
	}
	else
	{
		f_strcpy( &szPrompt[ f_strlen( szPrompt)], "decimal): ");
	}
	if (!ViewGetNum( szPrompt, pvNum, bEnterHexFlag, uiNumBytes, ui64MaxValue,
											&bValEntered) ||
		 !bValEntered)
	{
		return( FALSE);
	}
	else
	{
		return( TRUE);
	}
}

/********************************************************************
Desc:	Edit Text
*********************************************************************/
FLMBOOL ViewEditText(
	const char *	pszPrompt,
	char *			pszText,
	FLMUINT			uiTextLen,
	FLMBOOL *		pbValEntered)
{
	char		szTempBuf [100];
	FLMUINT	uiNumCols;
	FLMUINT	uiNumRows;

	f_conGetScreenSize( &uiNumCols, &uiNumRows);
	f_conSetBackFore( FLM_BLACK, FLM_WHITE);
	f_conClearScreen( 0, uiNumRows - 2);
	ViewAskInput( pszPrompt, szTempBuf, sizeof( szTempBuf) - 1);
	f_conSetBackFore( FLM_BLACK, FLM_WHITE);
	f_conClearScreen( 0, uiNumRows - 2);
	
	if (f_stricmp( szTempBuf, "\\") == 0)
	{
		*pbValEntered = FALSE;
		return( FALSE);
	}
	if( !szTempBuf[ 0])
	{
		*pbValEntered = FALSE;
		return( TRUE);
	}
	if (f_strlen( szTempBuf) >= uiTextLen)
	{
		szTempBuf [uiTextLen - 1] = 0;
	}
	f_strcpy( pszText, szTempBuf);
	*pbValEntered = TRUE;
	return( TRUE);
}

/********************************************************************
Desc:	Edit language
*********************************************************************/
FLMBOOL ViewEditLanguage(
	FLMUINT *	puiLang
	)
{
	char		szTempBuf[ 80];
	FLMUINT	uiTempNum;
	FLMBOOL	bValEntered;

	for (;;)
	{
		if (!ViewEditText( "Enter Language Code: ", 
						szTempBuf, 3, &bValEntered) ||
			 !bValEntered)
		{
			return( FALSE);
		}
		if (f_strlen( szTempBuf) != 2)
		{
			uiTempNum = 0;
			szTempBuf [0] = 0;
		}
		else
		{
			if (szTempBuf [0] >= 'a' && szTempBuf [0] <= 'z')
			{
				szTempBuf [0] = szTempBuf [0] - 'a' + 'A';
			}
			if (szTempBuf [1] >= 'a' && szTempBuf [1] <= 'z')
			{
				szTempBuf [1] = szTempBuf [1] - 'a' + 'A';
			}
			uiTempNum = f_languageToNum( szTempBuf);
		}
		if (uiTempNum == 0 &&
			 (szTempBuf [0] != 'U' || szTempBuf [1] != 'S'))
		{
			ViewShowError( "Illegal language code");
		}
		else
		{
			*puiLang = uiTempNum;
			return( TRUE);
		}
	}
}

/********************************************************************
Desc:	Edit binary data
*********************************************************************/
FLMBOOL ViewEditBinary(
	const char *	pszPrompt,
	FLMBYTE *		pucBuf,
	FLMUINT *		puiByteCount,
	FLMBOOL *		pbValEntered)
{
	FLMUINT		uiMaxBytes = *puiByteCount;
	FLMUINT		uiByteCount;
	FLMBOOL		bOdd;
	char			szTempBuf [300];
	FLMUINT		uiLoop;
	FLMUINT		uiNumCols;
	FLMUINT		uiNumRows;

	f_conGetScreenSize( &uiNumCols, &uiNumRows);

	if (!pszPrompt)
	{
		pszPrompt = "Enter Binary Values (in hex): ";
	}
	for (;;)
	{
		f_conSetBackFore( FLM_BLACK, FLM_WHITE);
		f_conClearScreen( 0, uiNumRows - 2);
		ViewAskInput( pszPrompt, szTempBuf, sizeof( szTempBuf));
		f_conSetBackFore( FLM_BLACK, FLM_WHITE);
		f_conClearScreen( 0, uiNumRows - 2);
		if( f_stricmp( szTempBuf, "\\") == 0)
		{
			*pbValEntered = FALSE;
			return( FALSE);
		}
		if (!szTempBuf[ 0])
		{
			*pbValEntered = FALSE;
			return( TRUE);
		}

		bOdd = FALSE;
		uiByteCount = 0;
		uiLoop = 0;
		while (szTempBuf [uiLoop])
		{
			FLMBYTE	ucValue;

			ucValue = szTempBuf [uiLoop];
			if (ucValue >= '0' && ucValue <= '9')
			{
				ucValue -= '0';
			}
			else if (ucValue >= 'a' && ucValue <= 'f')
			{
				ucValue = ucValue - 'a' + 10;
			}
			else if (ucValue >= 'A' && ucValue <= 'F')
			{
				ucValue = ucValue - 'A' + 10;
			}
			else if (ucValue == ' ' || ucValue == '\t')
			{
				bOdd = FALSE;
				uiLoop++;
				continue;
			}
			else
			{
				uiByteCount = 0;
				ViewShowError( "Non-HEX digits are illegal");
				break;
			}

			// If we get here, we have another digit

			if (bOdd)
			{
				bOdd = FALSE;
				(*pucBuf) <<= 4;
				(*pucBuf) |= ucValue;
			}
			else
			{
				if (uiByteCount == uiMaxBytes)
				{
					break;
				}

				// Don't increment pucBuf the first time through

				if (uiByteCount)
				{
					pucBuf++;
				}
				uiByteCount++;
				*pucBuf = ucValue;
				bOdd = TRUE;
			}
			uiLoop++;
		}
		if (!uiByteCount)
		{
			if (!szTempBuf[ uiLoop])
			{
				ViewShowError( "No HEX digits entered");
			}
		}
		else if (szTempBuf[ uiLoop])
		{
			ViewShowError( "Too many digits entered");
		}
		else
		{
			*puiByteCount = uiByteCount;
			*pbValEntered = TRUE;
			return( TRUE);
		}
	}
}

/********************************************************************
Desc:	Edit bits
*********************************************************************/
FLMBOOL ViewEditBits(
	FLMBYTE *	pucBit,
	FLMBOOL		bEnterHexFlag,
	FLMBYTE		ucMask
	)
{
	FLMBYTE	ucShiftBits = 0;

	// Determine the maximum value that can be entered

	while (!(ucMask & 0x01))
	{
		ucShiftBits++;
		ucMask >>= 1;
	}

	if (!ViewEditNum( pucBit, bEnterHexFlag, 1, (FLMUINT64)ucMask))
	{
		return( FALSE);
	}
	if (ucShiftBits)
	{
		(*pucBit) <<= ucShiftBits;
	}
	return( TRUE);
}

/********************************************************************
Desc: Edit a field
*********************************************************************/
FLMBOOL ViewEdit(
	FLMBOOL	bWriteEntireBlock
	)
{
	FLMUINT			uiBytesToWrite;
	FLMUINT			uiBytesWritten;
	FLMUINT			uiNum;
	FLMUINT64		ui64Num;
	FLMUINT32		ui32Num;
	FLMUINT16		ui16Num;
	char				szTempBuf[ 100];
	F_BLK_HDR *		pBlkHdr = NULL;
	RCODE				rc;
	FLMUINT			uiFileOffset;
	FLMUINT			uiFileNumber;
	FLMBOOL			bValEntered;
	FLMUINT			uiBytesRead;
	IF_FileHdl *	pFileHdl;
	FLMUINT32		ui32CRC;
	FLMUINT			uiBlockOffset;
	FLMUINT			uiBlkAddress = 0;
	FLMBYTE			ucSENValue[ 9];
	FLMBYTE *		pucSENBuffer = &ucSENValue[0];
	FLMUINT			uiSENLen = 0;

	if ((gv_pViewMenuCurrItem->uiModType & 0xF0) == MOD_DISABLED)
	{
		ViewShowError( "Cannot modify this value");
		return( FALSE);
	}
	uiFileOffset = gv_pViewMenuCurrItem->uiModFileOffset;
	uiFileNumber = gv_pViewMenuCurrItem->uiModFileNumber;

	switch (gv_pViewMenuCurrItem->uiModType & 0x0F)
	{
		case MOD_SEN5:
			// The SEN value is at most 5 bytes.
			ui64Num = 0;
			if (!ViewEditNum( &ui64Num,
						((gv_pViewMenuCurrItem->uiModType & 0xF0) == MOD_HEX), 5,
						~((FLMUINT64)0)))
			{
				return( FALSE);
			}

			// Need to know make a SEN out of this first.
			uiSENLen = f_encodeSEN( ui64Num, &pucSENBuffer);
			break;

		case MOD_SEN9:
			// The SEN value is at most 5 bytes.
			ui64Num = 0;
			if (!ViewEditNum( &ui64Num,
						((gv_pViewMenuCurrItem->uiModType & 0xF0) == MOD_HEX), 9,
						~((FLMUINT64)0)))
			{
				return( FALSE);
			}

			// Need to know make a SEN out of this first.
			uiSENLen = f_encodeSEN( ui64Num, &pucSENBuffer);
			break;

		case MOD_FLMUINT64:
			uiBytesToWrite = 8;
			if (!ViewEditNum( &ui64Num,
						((gv_pViewMenuCurrItem->uiModType & 0xF0) == MOD_HEX), 8,
						~((FLMUINT64)0)))
			{
				return( FALSE);
			}

			if (gv_pViewMenuCurrItem->uiModType & MOD_NATIVE)
			{
				f_memcpy( szTempBuf, &ui64Num, 8);
			}
			else
			{
				U642FBA( ui64Num, (FLMBYTE *)szTempBuf);
			}
			break;
		case MOD_FLMUINT32:
			uiBytesToWrite = 4;
			if (!ViewEditNum( &ui32Num,
						((gv_pViewMenuCurrItem->uiModType & 0xF0) == MOD_HEX), 4,
						(FLMUINT64)0xFFFFFFFF))
			{
				return( FALSE);
			}
			if (gv_pViewMenuCurrItem->uiModType & MOD_NATIVE)
			{
				f_memcpy( szTempBuf, &ui32Num, 4);
			}
			else
			{
				UD2FBA( ui32Num, (FLMBYTE *)szTempBuf);
			}
			break;
		case MOD_FLMUINT16:
			uiBytesToWrite = 2;
			if (!ViewEditNum( &ui16Num,
						((gv_pViewMenuCurrItem->uiModType & 0xF0) == MOD_HEX), 2,
						(FLMUINT64)0xFFFF))
			{
				return( FALSE);
			}
			if (gv_pViewMenuCurrItem->uiModType & MOD_NATIVE)
			{
				f_memcpy( szTempBuf, &ui16Num, 2);
			}
			else
			{
				UW2FBA( ui16Num, (FLMBYTE *)szTempBuf);
			}
			break;
		case MOD_FLMBYTE:
			uiBytesToWrite = 1;
			if (!ViewEditNum( &szTempBuf [0],
					((gv_pViewMenuCurrItem->uiModType & 0xF0) == MOD_HEX), 1,
					(FLMUINT64)0xFF))
			{
				return( FALSE);
			}
			break;
		case MOD_BINARY:
			uiBytesToWrite = gv_pViewMenuCurrItem->uiModBufLen;
			if (HAVE_HORIZ_CUR( gv_pViewMenuCurrItem))
			{
				uiFileOffset += gv_pViewMenuCurrItem->uiHorizCurPos;
				uiBytesToWrite -= gv_pViewMenuCurrItem->uiHorizCurPos;
			}
			if (!ViewEditBinary( NULL, (FLMBYTE *)szTempBuf,
							&uiBytesToWrite, &bValEntered) ||
				 !bValEntered)
			{
				return( FALSE);
			}
			break;
		case MOD_TEXT:
			if (!ViewEditText( "Enter Value: ", 
						szTempBuf, gv_pViewMenuCurrItem->uiModBufLen,
						&bValEntered) ||
				 !bValEntered)
			{
				return( FALSE);
			}
			uiBytesToWrite = gv_pViewMenuCurrItem->uiModBufLen;
			break;
		case MOD_LANGUAGE:
			if( !ViewEditLanguage( &uiNum))
			{
				return( FALSE);
			}
			szTempBuf[0] = (FLMBYTE)uiNum;
			uiBytesToWrite = 1;
			break;
		case MOD_CHILD_BLK:
			if( !ViewEditNum( &uiNum, TRUE, 4, (FLMUINT64)0xFFFFFFFF))
			{
				return( FALSE);
			}
			uiBytesToWrite = 4;
			UD2FBA( (FLMUINT32)uiNum, (FLMBYTE *)szTempBuf);
			break;
		case MOD_BITS:
			if (!ViewEditBits( (FLMBYTE *)&szTempBuf[ 0],
						((gv_pViewMenuCurrItem->uiModType & 0xF0) == MOD_HEX),
						(FLMBYTE)gv_pViewMenuCurrItem->uiModBufLen))
			{
				return( FALSE);
			}
			uiBytesToWrite = 1;
			break;
	}

	// Read in the block if necessary

	if (!bWriteEntireBlock)
	{
		pBlkHdr = (F_BLK_HDR *)(&szTempBuf [0]);
	}
	else
	{
		uiBlockOffset = (FLMUINT)(uiFileOffset %
						(FLMUINT)gv_ViewDbHdr.ui16BlockSize);
		uiBlkAddress = FSBlkAddress( uiFileNumber, 
											uiFileOffset - uiBlockOffset);
		uiFileOffset = uiFileOffset - uiBlockOffset;

		// Don't convert the block if the address is zero - means we
		// are updating the database header.

		if (!ViewBlkRead( uiBlkAddress, &pBlkHdr,
									!uiBlkAddress ? FALSE : TRUE,
									(FLMUINT)gv_ViewDbHdr.ui16BlockSize,
									NULL, NULL, &uiBytesRead, FALSE))
		{
			return( FALSE);
		}

		uiBytesToWrite = uiBytesRead;

		// Convert to native format

		if (!uiBlkAddress)
		{
			if (hdrIsNonNativeFormat( (XFLM_DB_HDR *)pBlkHdr))
			{
				convertDbHdr( (XFLM_DB_HDR *)pBlkHdr);
			}
		}
		else
		{
			if (blkIsNonNativeFormat( pBlkHdr))
			{
				convertBlk( (FLMUINT)gv_ViewDbHdr.ui16BlockSize, pBlkHdr);
			}
		}

		// Put the data in the appropriate place in the block

		if ((gv_pViewMenuCurrItem->uiModType & 0x0F) == MOD_BITS)
		{
			FLMBYTE		ucMask = (FLMBYTE)gv_pViewMenuCurrItem->uiModBufLen;
			FLMBYTE *	pucBuf = (FLMBYTE *)pBlkHdr;

			// Unset the bits, then OR in the new bits

			pucBuf [uiBlockOffset] &= (~(ucMask));
			pucBuf [uiBlockOffset] |= szTempBuf[ 0];
		}
		else if ((gv_pViewMenuCurrItem->uiModType & 0x0F) == MOD_SEN5 ||
					(gv_pViewMenuCurrItem->uiModType & 0x0F) == MOD_SEN9)
		{
			// Need to make sure the size of the original SEN is the same as the
			// new SEN.
			const FLMBYTE *	pucOrigSEN = (FLMBYTE *)pBlkHdr + uiBlockOffset;
			FLMBYTE				ucBuffer[9];
			FLMBYTE *			pucBuffer = &ucBuffer[0];
			FLMUINT64			ui64Value;
			FLMUINT				uiOrigSENLen;

			if (RC_BAD( rc = f_decodeSEN64( &pucOrigSEN,
													 (FLMBYTE *)pBlkHdr +
															gv_ViewDbHdr.ui16BlockSize,
													 &ui64Value)))
			{
				ViewShowRCError( "Decoding original SEN value", rc);
			}
			uiOrigSENLen = f_encodeSEN( ui64Value, &pucBuffer);

			if (uiOrigSENLen != uiSENLen)
			{
				ViewShowRCError( "SEN Length does not match original",
									  NE_XFLM_FAILURE);
			}
			else
			{
				f_memcpy( (FLMBYTE *)pBlkHdr + uiBlockOffset, ucSENValue,
							uiSENLen);
			}
		}
		else
		{
			f_memcpy( (FLMBYTE *)pBlkHdr + uiBlockOffset, szTempBuf,
							uiBytesToWrite);
		}

		// Calculate CRC

		if (!uiBlkAddress)
		{
			ui32CRC = calcDbHdrCRC( (XFLM_DB_HDR *)pBlkHdr);
			((XFLM_DB_HDR *)pBlkHdr)->ui32HdrCRC = ui32CRC;
			uiBytesToWrite = sizeof( XFLM_DB_HDR);
		}
		else
		{
			FLMUINT	uiBlkLen;
			uiBlkLen = gv_ViewDbHdr.ui16BlockSize;

#if 0
			if ((FLMUINT)pBlkHdr->ui16BlkBytesAvail >
					(FLMUINT)gv_ViewDbHdr.ui16BlockSize - blkHdrSize( pBlkHdr))
			{
				uiBlkLen = blkHdrSize( pBlkHdr);
			}
			else
			{
				uiBlkLen = (FLMUINT)(gv_ViewDbHdr.ui16BlockSize - 
											pBlkHdr->ui16BlkBytesAvail);
			}
#endif
			// Calculate and set the block CRC.

			ui32CRC = calcBlkCRC( pBlkHdr, uiBlkLen);
			pBlkHdr->ui32BlkCRC = ui32CRC;
		}
	}
	
	if (RC_BAD( rc = gv_pSFileHdl->getFileHdl( uiFileNumber, TRUE, &pFileHdl)))
	{
		ViewShowRCError( "getting file handle", rc);
	}

	// Write the data out to the file

	else if (RC_BAD( rc = pFileHdl->write( uiFileOffset, uiBytesToWrite,
							pBlkHdr, &uiBytesWritten)))
	{
		ViewShowRCError( "updating file", rc);
	}
	else if (RC_BAD( rc = pFileHdl->flush()))
	{
		ViewShowRCError( "flushing data to file", rc);
	}
	else if (bWriteEntireBlock && !uiBlkAddress)
	{
		f_memcpy( &gv_ViewDbHdr, pBlkHdr, sizeof( XFLM_DB_HDR));
	}

	// Free any memory used to read in the data block

	if (pBlkHdr && pBlkHdr != (F_BLK_HDR *)(&szTempBuf [0]))
	{
		f_free( &pBlkHdr);
	}
	return( TRUE);
}
