//-------------------------------------------------------------------------
// Desc: Editing routines for the database viewer utility.
// Tabs: 3
//
// Copyright (c) 1992-2001, 2003-2007 Novell, Inc. All Rights Reserved.
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
Desc: ?
*********************************************************************/
FLMINT ViewGetNum(
	const char *	Prompt,
	void *			NumRV,
	FLMUINT			EnterHexFlag,
	FLMUINT			NumBytes,
	FLMUINT			MaxValue,
	FLMUINT *		ValEntered)
{
	char			TempBuf[ 20];
	FLMUINT		i;
	FLMUINT		c;
	FLMINT		GetOK;
	FLMUINT		Num;
	FLMUINT		MaxDigits;
	FLMUINT		uiNumCols;
	FLMUINT		uiNumRows;

	f_conGetScreenSize( &uiNumCols, &uiNumRows);

	if( EnterHexFlag)
		MaxDigits = (NumBytes == 4) ? 8 : ((NumBytes == 2) ? 4 : 2);
	else
		MaxDigits = (NumBytes == 4) ? 10 : ((NumBytes == 2) ? 5 : 3);

	for( ;;)
	{
		GetOK = TRUE;
		f_conSetBackFore( FLM_BLACK, FLM_WHITE);
		f_conClearScreen( 0, uiNumRows - 2);
		ViewAskInput( Prompt, TempBuf, sizeof( TempBuf));
		f_conSetBackFore( FLM_BLACK, FLM_WHITE);
		f_conClearScreen( 0, uiNumRows - 2);
		if( f_stricmp( TempBuf, "\\") == 0)
		{
			*ValEntered = FALSE;
			return( FALSE);
		}
		if( !TempBuf[ 0])
		{
			*ValEntered = FALSE;
			return( TRUE);
		}
		i = 0;
		Num = 0;
		while( (TempBuf[ i]) && (i < MaxDigits))
		{
			c = TempBuf[ i];
			if( EnterHexFlag)
			{
				Num <<= 4;
				if( (c >= '0') && (c <= '9'))
					Num += (FLMUINT)(c - '0');
				else if( (c >= 'a') && (c <= 'f'))
					Num += (FLMUINT)(c - 'a' + 10);
				else if( (c >= 'A') && (c <= 'F'))
					Num += (FLMUINT)(c - 'A' + 10);
				else
				{
					ViewShowError( "Illegal digit in number - must be hex digits");
					GetOK = FALSE;
					break;
				}
			}
			else if( (c < '0') || (c > '9'))
			{
				ViewShowError( "Illegal digit in number - must be 0 through 9");
				GetOK = FALSE;
				break;
			}
			else
			{
				if( MaxValue / 10 < Num)
				{
					ViewShowError( "Number is too large");
					GetOK = FALSE;
					break;
				}
				else
				{
					Num *= 10;
					if( MaxValue - (FLMUINT)(c - '0') < Num)
					{
						ViewShowError( "Number is too large");
						GetOK = FALSE;
						break;
					}
					else
						Num += (FLMUINT)(c - '0');
				}
			}
			i++;
		}
		if( GetOK)
		{
			if( NumBytes == 4)
				*((FLMUINT	*)(NumRV)) = Num;
			else if( NumBytes == 2)
				*((FLMUINT	*)(NumRV)) = (FLMUINT)Num;
			else
				*((FLMBYTE *)(NumRV)) = (FLMBYTE)Num;
			*ValEntered = TRUE;
			return( TRUE);
		}
	}
}

/********************************************************************
Desc: ?
*********************************************************************/
FLMINT ViewEditNum(
	void *		NumRV,
	FLMUINT		EnterHexFlag,
	FLMUINT		NumBytes,	/* WAS FLMUINT */
	FLMUINT		MaxValue
	)
{
	char			Prompt[ 80];
	FLMUINT		ValEntered;

	f_strcpy( Prompt, "Enter Value (in ");
	if( EnterHexFlag)
		f_strcpy( &Prompt[ f_strlen( Prompt)], "hex): ");
	else
		f_strcpy( &Prompt[ f_strlen( Prompt)], "decimal): ");
	if( (!ViewGetNum( Prompt, NumRV, EnterHexFlag, (FLMUINT)NumBytes, MaxValue,
											&ValEntered)) ||
			 (!ValEntered))
		return( FALSE);
	else
		return( TRUE);
}

/********************************************************************
Desc: ?
*********************************************************************/
FLMINT ViewEditText(
	const char *	Prompt,
	char *			TextRV,
	FLMUINT			TextLen,
	FLMUINT *		ValEntered)
{
	char	 		TempBuf[ 100];
	FLMUINT		uiNumCols;
	FLMUINT		uiNumRows;

	f_conGetScreenSize( &uiNumCols, &uiNumRows);
	f_conSetBackFore( FLM_BLACK, FLM_WHITE);
	f_conClearScreen( 0, uiNumRows - 2);
	ViewAskInput( Prompt, TempBuf, sizeof( TempBuf) - 1);
	f_conSetBackFore( FLM_BLACK, FLM_WHITE);
	f_conClearScreen( 0, uiNumRows - 2);
	if( f_stricmp( TempBuf, "\\") == 0)
	{
		*ValEntered = FALSE;
		return( FALSE);
	}
	if( !TempBuf[ 0])
	{
		*ValEntered = FALSE;
		return( TRUE);
	}
	f_memset( TextRV, 0, TextLen);
	if( f_strlen( TempBuf) >= TextLen)
		f_memcpy( TextRV, TempBuf, TextLen);
	else
		f_strcpy( TextRV, TempBuf);
	*ValEntered = TRUE;
	return( TRUE);
}

/********************************************************************
Desc: ?
*********************************************************************/
FLMINT ViewEditLanguage(
	FLMUINT *	LangRV)
{
	char			TempBuf[ 80];
	FLMUINT		TempNum;
	FLMUINT		ValEntered;

	for( ;;)
	{
		if( (!ViewEditText( "Enter Language Code: ", 
			TempBuf, 3, &ValEntered)) || (!ValEntered))
			return( FALSE);
		if( f_strlen( TempBuf) != 2)
		{
			TempNum = 0;
			TempBuf[ 0] = 0;
		}
		else
		{
			if( (TempBuf[ 0] >= 'a') && (TempBuf[ 0] <= 'z'))
				TempBuf[ 0] = TempBuf[ 0] - 'a' + 'A';
			if( (TempBuf[ 1] >= 'a') && (TempBuf[ 1] <= 'z'))
				TempBuf[ 1] = TempBuf[ 0] - 'a' + 'A';
			TempNum = f_languageToNum( (char *)TempBuf);
		}
		if( (TempNum == 0) &&
				((TempBuf[ 0] != 'U') || (TempBuf[ 1] != 'S')))
			ViewShowError( "Illegal language code");
		else
		{
			*LangRV = TempNum;
			return( TRUE);
		}
	}
}

/********************************************************************
Desc: ?
*********************************************************************/
FLMINT ViewEditBinary(
	const char *	Prompt,
	char *			Buf,
	FLMUINT	*		ByteCountRV,
	FLMUINT *		ValEntered)
{
	FLMUINT		MaxBytes = *ByteCountRV;
	FLMUINT		ByteCount;
	FLMUINT		Odd;
	char			TempBuf[ 300];
	FLMUINT		i;
	char			TempPrompt[ 80];
	FLMUINT		uiNumCols;
	FLMUINT		uiNumRows;

	f_conGetScreenSize( &uiNumCols, &uiNumRows);

	if( Prompt == NULL)
	{
		f_strcpy( TempPrompt, "Enter Binary Values (in hex): ");
		Prompt = &TempPrompt[ 0];
	}
	for( ;;)
	{
		f_conSetBackFore( FLM_BLACK, FLM_WHITE);
		f_conClearScreen( 0, uiNumRows - 2);
		ViewAskInput( Prompt, TempBuf, sizeof( TempBuf));
		f_conSetBackFore( FLM_BLACK, FLM_WHITE);
		f_conClearScreen( 0, uiNumRows - 2);
		if( f_stricmp( TempBuf, "\\") == 0)
		{
			*ValEntered = FALSE;
			return( FALSE);
		}
		if( !TempBuf[ 0])
		{
			*ValEntered = FALSE;
			return( TRUE);
		}

		Odd = 0;
		ByteCount = 0;
		i = 0;
		while( TempBuf[ i])
		{
			FLMBYTE	 Value;

			Value = TempBuf[ i];
			if( (Value >= '0') && (Value <= '9'))
				Value -= '0';
			else if( (Value >= 'a') && (Value <= 'f'))
				Value = Value - 'a' + 10;
			else if( (Value >= 'A') && (Value <= 'F'))
				Value = Value - 'A' + 10;
			else if( Value == ' ' || Value == '\t')
			{
				Odd = 0;
				i++;
				continue;
			}
			else
			{
				ByteCount = 0;
				ViewShowError( "Non-HEX digits are illegal");
				break;
			}

			/* If we get here, we have another digit */

			if( Odd)
			{
				Odd = 0;
				(*Buf) <<= 4;
				(*Buf) |= Value;
			}
			else
			{
				if( ByteCount == MaxBytes)
					break;

				/* Don't increment Buf the first time through */

				if( ByteCount)
					Buf++;
				ByteCount++;
				*Buf = Value;
				Odd = 1;
			}
			i++;
		}
		if( !ByteCount)
		{
			if( !TempBuf[ i])
				ViewShowError( "No HEX digits entered");
		}
		else if( TempBuf[ i])
			ViewShowError( "Too many digits entered");
		else
		{
			*ByteCountRV = ByteCount;
			*ValEntered = TRUE;
			return( TRUE);
		}
	}
}

/********************************************************************
Desc: ?
*********************************************************************/
FLMINT ViewEditBits(
	FLMBYTE *	 BitRV,
	FLMUINT		 EnterHexFlag,
	FLMBYTE		 Mask)
{
	FLMBYTE	 ShiftBits = 0;

	/* Determine the maximum value that can be entered */

	while( !(Mask & 0x01))
	{
		ShiftBits++;
		Mask >>= 1;
	}

	if( !ViewEditNum( BitRV, EnterHexFlag, 1, (FLMUINT)Mask))
		return( FALSE);
	if( ShiftBits)
		(*BitRV) <<= ShiftBits;
	return( TRUE);
}

/********************************************************************
Desc: ?
*********************************************************************/
FLMINT ViewEdit(
	FLMUINT	WriteEntireBlock,
	FLMBOOL	bRecalcChecksum
	)
{
	FLMUINT			BytesToWrite;
	FLMUINT			BytesWritten;
	FLMUINT			Num;
	char				TempBuf[ 100];
	char *			BufPtr = NULL;
	RCODE				rc;
	FLMUINT			FileOffset;
	FLMUINT			FileNumber;
	FLMUINT			ValEntered;
	FLMUINT			wBytesRead;
	IF_FileHdl * 	pFileHdl = NULL;
	FLMBOOL			bEncrypted;
	FLMBOOL			bIsEncBlock;
	FLMBOOL			bModEnc = FALSE;

	if( (gv_pViewMenuCurrItem->ModType & 0xF0) == MOD_DISABLED)
	{
		ViewShowError( "Cannot modify this value");
		return( FALSE);
	}
	FileOffset = gv_pViewMenuCurrItem->ModFileOffset;
	FileNumber = gv_pViewMenuCurrItem->ModFileNumber;

	switch( gv_pViewMenuCurrItem->ModType & 0x0F)
	{
		case MOD_FLMUINT:
			BytesToWrite = 4;
			if( !ViewEditNum( &Num,
						((gv_pViewMenuCurrItem->ModType & 0xF0) == MOD_HEX), 4,
						0xFFFFFFFF))
				return( FALSE);
			UD2FBA( Num, (FLMBYTE *)TempBuf);
			break;
		case MOD_FLMUINT16:
			BytesToWrite = 2;
			if( !ViewEditNum( &Num,
						((gv_pViewMenuCurrItem->ModType & 0xF0) == MOD_HEX), 2,
						0xFFFF))
				return( FALSE);
			UW2FBA( Num, (FLMBYTE *)TempBuf);
			break;
		case MOD_FLMBYTE:
			BytesToWrite = 1;
			if( !ViewEditNum( &TempBuf[ 0],
						((gv_pViewMenuCurrItem->ModType & 0xF0) == MOD_HEX), 1, 0xFF))
				return( FALSE);
			break;
		case MOD_BINARY_ENC:
			bModEnc = TRUE;
			goto Mod_Binary;
		case MOD_BINARY:
Mod_Binary:
			BytesToWrite = gv_pViewMenuCurrItem->ModBufLen;
			if( HAVE_HORIZ_CUR( gv_pViewMenuCurrItem))
			{
				FileOffset += gv_pViewMenuCurrItem->HorizCurPos;
				BytesToWrite -= gv_pViewMenuCurrItem->HorizCurPos;
			}
			if( (!ViewEditBinary( NULL, TempBuf, &BytesToWrite, &ValEntered)) ||
					(!ValEntered))
				return( FALSE);
			break;
		case MOD_TEXT:
			if( (!ViewEditText( "Enter Value: ", 
				TempBuf, gv_pViewMenuCurrItem->ModBufLen, &ValEntered)) ||
					(!ValEntered))
				return( FALSE);
			BytesToWrite = gv_pViewMenuCurrItem->ModBufLen;
			break;
		case MOD_LANGUAGE:
			if( !ViewEditLanguage( &Num))
				return( FALSE);
			TempBuf[0] = (FLMBYTE) Num;
			BytesToWrite = 1;
			break;
		case MOD_CHILD_BLK:
			if( !ViewEditNum( &Num, TRUE, 4, 0xFFFFFFFF))
				return( FALSE);
			BytesToWrite = 4;
			UD2FBA( Num, (FLMBYTE *)TempBuf);
			break;
		case MOD_BITS:
			if( !ViewEditBits( (FLMBYTE *)&TempBuf[ 0],
						((gv_pViewMenuCurrItem->ModType & 0xF0) == MOD_HEX),
						(FLMBYTE)gv_pViewMenuCurrItem->ModBufLen))
				return( FALSE);
			BytesToWrite = 1;
			break;
		case MOD_KEY_LEN:
			if( !ViewEditNum( &Num,
					((gv_pViewMenuCurrItem->ModType & 0xF0) == MOD_HEX),
					2, 0x000003FF))
				return( FALSE);
			TempBuf[ 0] = (FLMBYTE)((Num >> 8) & 0x0003) << 4;
			TempBuf[ 1] = (FLMBYTE)(Num & 0x00FF);
			break;
	}

	/* Read in the block if necessary */

	if( !WriteEntireBlock)
	{
		BufPtr = &TempBuf[ 0];
	}
	else
	{
		FLMUINT		BlockOffset;
		FLMUINT		BlkAddress;
		FLMUINT16	ui16BlkChkSum;
		FLMBOOL		bNeedToEncrypt;

		BlockOffset = (FLMUINT)(FileOffset %
						(FLMUINT)gv_ViewHdrInfo.FileHdr.uiBlockSize);
		BlkAddress = FSBlkAddress( FileNumber, 
											FileOffset - BlockOffset);
		FileOffset = FileOffset - BlockOffset;

		if( !ViewBlkRead( BlkAddress, (FLMBYTE **)&BufPtr,
									gv_ViewHdrInfo.FileHdr.uiBlockSize,
									NULL, &ui16BlkChkSum, &wBytesRead, FALSE,
									&bIsEncBlock, bModEnc ? FALSE : TRUE, &bEncrypted))
			return( FALSE);

		bNeedToEncrypt = FALSE;
		if (bEncrypted)
		{
			flmAssert( bIsEncBlock);
			flmAssert( bModEnc);
		}
		else
		{
			// bModEnc would only be TRUE if the original read returned
			// the data encrypted, but if that is the case, this read should
			// also have returned the data encrypted.
			
			flmAssert( !bModEnc);
			if (bIsEncBlock)
			{
				bNeedToEncrypt = TRUE;
			}
		}

		/* Put the data in the appropriate place in the block */

		if( (gv_pViewMenuCurrItem->ModType & 0x0F) == MOD_BITS)
		{
			FLMBYTE	 Mask = (FLMBYTE)gv_pViewMenuCurrItem->ModBufLen;

			/* Unset the bits, then OR in the new bits */

			BufPtr[ BlockOffset] &= (~(Mask));
			BufPtr[ BlockOffset] |= TempBuf[ 0];
		}
		else if( (gv_pViewMenuCurrItem->ModType & 0x0F) == MOD_KEY_LEN)
		{

			/* Unset the high bits of the key length, then OR in the new bits */

			BufPtr[ BlockOffset] &= ~(0x30);
			BufPtr[ BlockOffset] |= TempBuf[ 0];

			/* Set the low bits of the key length. */

			BufPtr[ BlockOffset + BBE_KL] = TempBuf[ 1];
		}
		else
		{
			f_memcpy( BufPtr + BlockOffset, TempBuf, BytesToWrite);
		}
		
		// Re-encrypt the data, if necessary
		
		if (bNeedToEncrypt)
		{
#ifndef FLM_USE_NICI
			// Should not be possible to get here
			flmAssert( 0);
#else
			IXD *				pIxd;
			FLMUINT			uiIxNum = FB2UW( (const FLMBYTE *)&BufPtr [BH_LOG_FILE_NUM]);
			FLMUINT			uiEncLen = getEncryptSize( (FLMBYTE *)BufPtr) - BH_OVHD;
			FDB *				pDb = (FDB *)gv_hViewDb;
			FFILE *			pFile = pDb->pFile;
			
			flmAssert( uiEncLen);
			flmAssert( !pFile->bInLimitedMode);

			// Get the index.
			
			if (RC_OK( fdictGetIndex( pFile->pDictList,
						pFile->bInLimitedMode, uiIxNum, NULL,
						&pIxd, TRUE)) &&
						pIxd && pIxd->uiEncId)
			{
				F_CCS *	pCcs = (F_CCS *)pFile->pDictList->pIttTbl[ pIxd->uiEncId].pvItem;
				
				flmAssert( pCcs);
				flmAssert( !(uiEncLen % 16));
				// Encrypt the buffer in place.
				(void)pCcs->encryptToStore( (FLMBYTE *)&BufPtr [BH_OVHD], uiEncLen,
												(FLMBYTE *)&BufPtr [BH_OVHD], &uiEncLen);
			}
#endif
		}

		/* Recalculate the checksum */

		if (bRecalcChecksum)
		{
			if (FB2UW( (FLMBYTE *)&BufPtr [BH_BLK_END]) >
					gv_ViewHdrInfo.FileHdr.uiBlockSize)
			{
				UW2FBA( gv_ViewHdrInfo.FileHdr.uiBlockSize,
							(FLMBYTE *)&BufPtr [BH_BLK_END]);
			}
			BlkCheckSum( (FLMBYTE *)BufPtr,
							FALSE, BlkAddress,
							gv_ViewHdrInfo.FileHdr.uiBlockSize);
		}
		else
		{

			/*
			Restore checksum bytes to whatever we read. This is necessary
			because ViewBlkRead for version 3.x and greater will alter
			the low checksum byte after reading the block.	It is not
			really necessary to restore the high checksum byte, but we
			do anyway for consistency.
			*/

			BufPtr [BH_CHECKSUM_HIGH] = (FLMBYTE)(ui16BlkChkSum >> 8);
			BufPtr [BH_CHECKSUM_LOW] = (FLMBYTE)(ui16BlkChkSum & 0x00FF);
		}
		BytesToWrite = wBytesRead;
	}
	
	if (RC_BAD( rc = gv_pSFileHdl->getFileHdl( FileNumber, TRUE, &pFileHdl)))
	{
		ViewShowRCError( "getting file handle", rc);
	}

	// Write the data out to the file

	else if( RC_BAD( rc = pFileHdl->write( FileOffset, BytesToWrite,
		BufPtr, &BytesWritten)))
	{
		ViewShowRCError( "updating file", rc);
	}
	else if( RC_BAD( rc = pFileHdl->flush()))
	{
		ViewShowRCError( "flushing data to file", rc);
	}

	// Free any memory used to read in the data block

	if( BufPtr != &TempBuf[ 0])
	{
		f_free( &BufPtr);
	}
	return( TRUE);
}
