//------------------------------------------------------------------------------
// Desc: This file contains routines for viewing blocks in the database.
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

FSTATIC void InitStatusBits(
	FLMBYTE *	pucStatusBytes);

FSTATIC void OrStatusBits(
	FLMBYTE *	pucDestStatusBytes,
	FLMBYTE *	pucSrcStatusBytes);

FSTATIC FLMBOOL TestStatusBit(
	FLMBYTE *	pucStatusBytes,
	FLMINT		iErrorCode);

FSTATIC FLMBOOL OutBlkHdrExpNum(
	FLMUINT		uiCol,
	FLMUINT *	puiRow,
	eColorType	uiBackColor,
	eColorType	uiForeColor,
	eColorType	uiUnselectBackColor,
	eColorType	uiUnselectForeColor,
	eColorType	uiSelectBackColor,
	eColorType	uiSelectForeColor,
	FLMUINT		uiLabelWidth,
	FLMINT		iLabelIndex,
	FLMUINT		uiFileNumber,
	FLMUINT		uiFileOffset,
	void *		pvcValue,
	FLMUINT		uiValueType,
	FLMUINT		uiModType,
	FLMUINT64	ui64ExpNum,
	FLMUINT64	ui64IgnoreExpNum,
	FLMUINT		uiOption);

FSTATIC void FormatBlkType(
	char *		pszTempBuf,
	FLMUINT		uiBlkType
	);

FSTATIC FLMBOOL OutputStatus(
	FLMUINT		uiCol,
	FLMUINT *	puiRow,
	eColorType	uiBackColor,
	eColorType	uiForeColor,
	FLMUINT		uiLabelWidth,
	FLMINT		iLabelIndex,
	FLMBYTE *	pucStatusFlags
	);

FSTATIC FLMBOOL OutputLeafElements(
	FLMUINT				uiCol,
	FLMUINT  *			puiRow,
	F_BTREE_BLK_HDR *	pBlkHdr,
	BLK_EXP_p			pBlkExp,
	FLMBYTE *			pucBlkStatus,
	FLMBOOL				bStatusOnlyFlag);

FSTATIC FLMBOOL OutputDataOnlyElements(
	FLMUINT				uiCol,
	FLMUINT  *			puiRow,
	F_BLK_HDR *			pBlkHdr,
	BLK_EXP_p			pBlkExp,
	FLMBYTE *			pucBlkStatus,
	FLMBOOL				bStatusOnlyFlag);

FSTATIC FLMBOOL OutputNonLeafElements(
	FLMUINT				uiCol,
	FLMUINT *			puiRow,
	F_BTREE_BLK_HDR *	pBlkHdr,
	BLK_EXP_p			pBlkExp,
	FLMBYTE *			pucBlkStatus,
	FLMBOOL				bStatusOnlyFlag);

FSTATIC FLMBOOL OutputDomNode(
	FLMUINT			uiCol,
	FLMUINT *		puiRow,
	eColorType		uiBackColor,
	eColorType		uiForeColor,
	FLMBYTE *		pucVal,
	STATE_INFO *	pStateInfo);

FSTATIC void SetSearchTopBottom( void);

FSTATIC void GetElmInfo(
	FLMBYTE *			pucEntry,
	F_BTREE_BLK_HDR *	pBlkHdr,
	STATE_INFO *		pStateInfo);

extern FLMUINT	gv_uiTopLine;
extern FLMUINT	gv_uiBottomLine;

/********************************************************************
Desc:	Initialize status bits
*********************************************************************/
FSTATIC void InitStatusBits(
	FLMBYTE *	pucStatusBytes
	)
{
	if (pucStatusBytes)
	{
		f_memset( pucStatusBytes, 0, NUM_STATUS_BYTES);
	}
}

/********************************************************************
Desc:	OR together two sets of status bytes
*********************************************************************/
FSTATIC void OrStatusBits(
	FLMBYTE *	pucDestStatusBytes,
	FLMBYTE *	pucSrcStatusBytes
	)
{
	FLMUINT	uiLoop;

	if (pucDestStatusBytes && pucSrcStatusBytes)
	{
		for (uiLoop = 0; uiLoop < NUM_STATUS_BYTES; uiLoop++)
		{
			pucDestStatusBytes [uiLoop] |= pucSrcStatusBytes [uiLoop];
		}
	}
}

/********************************************************************
Desc:	TestStatusBit
*********************************************************************/
FSTATIC FLMBOOL TestStatusBit(
	FLMBYTE *	pucStatusBytes,
	FLMINT		iErrorCode
	)
{

	if (!pucStatusBytes)
	{
		return( FALSE);
	}
	else
	{
		return( (FLMBOOL)(pucStatusBytes [(iErrorCode - 1) / 8] & 
			(FLMBYTE)((FLMBYTE)0x80 >> (FLMBYTE)((iErrorCode - 1) % 8))
			? TRUE
			: FALSE));
	}
}

/***************************************************************************
Desc: This routine outputs all of the status bits which were set for
		a block.  Each error discovered in a block will be displayed on
		a separate line.
*****************************************************************************/
FSTATIC FLMBOOL OutputStatus(
	FLMUINT		uiCol,
	FLMUINT  *	puiRow,
	eColorType	uiBackColor,
	eColorType	uiForeColor,
	FLMUINT		uiLabelWidth,
	FLMINT		iLabelIndex,
	FLMBYTE *	pucStatusFlags
	)
{
	FLMBOOL	bOk = FALSE;
	FLMUINT	uiRow = *puiRow;
	FLMINT	iError;
	FLMBOOL	bHadError = FALSE;

	// Output each error on a separate line

	if (pucStatusFlags)
	{
		for (iError = 0; iError < FLM_NUM_CORRUPT_ERRORS; iError++)
		{
			if (TestStatusBit( pucStatusFlags, iError))
			{
				bHadError = TRUE;
				if (!ViewAddMenuItem( iLabelIndex, uiLabelWidth,
							VAL_IS_ERR_INDEX, (FLMUINT64)iError, 0,
							0, VIEW_INVALID_FILE_OFFSET, 0, MOD_DISABLED,
							uiCol, uiRow++, 0,
							FLM_RED, FLM_LIGHTGRAY,
							FLM_RED, FLM_LIGHTGRAY))
				{
					goto Exit;
				}

				// Set iLabelIndex to -1 so that it will not be displayed after
				// the first one.

				iLabelIndex = -1;
			}
		}
	}

	// If there were no errors in the block, just output an OK status

	if (!bHadError)
	{
		if (!ViewAddMenuItem( iLabelIndex, uiLabelWidth,
					VAL_IS_LABEL_INDEX, (FLMUINT64)LBL_OK, 0,
					0, VIEW_INVALID_FILE_OFFSET, 0, MOD_DISABLED,
					uiCol, uiRow++, 0, uiBackColor, uiForeColor,
					uiBackColor, uiForeColor))
		{
			goto Exit;
		}
	}
	*puiRow = uiRow;
	bOk = TRUE;

Exit:

	return( bOk);
}


/***************************************************************************
Desc: This routine reads into memory a database block.  It will also
		allocate memory to hold the block if necessary.  The block will
		also be decrypted if necessary.
*****************************************************************************/
FLMBOOL ViewBlkRead(
	FLMUINT			uiBlkAddress,
	F_BLK_HDR **	ppBlkHdr,
	FLMBOOL			bOkToConvert,
	FLMUINT			uiReadLen,
	FLMUINT32 *		pui32CalcCRC,
	FLMUINT32 *		pui32BlkCRC,
	FLMUINT *		puiBytesRead,
	FLMBOOL			bShowPartialReadError
	)
{
	FLMBOOL		bOk = FALSE;
	RCODE       rc;
	F_BLK_HDR *	pBlkHdr;
	char			szErrMsg [80];
	FLMUINT		uiBlkEnd;
	FLMUINT16	ui16BlkBytesAvail;

	// First allocate memory to read the block into
	// if not already allocated

	if (!(*ppBlkHdr))
	{
		if (RC_BAD( rc = f_alloc( uiReadLen, ppBlkHdr)))
		{
			ViewShowRCError( "allocating memory to read block", rc);
			goto Exit;
		}
	}
	pBlkHdr = *ppBlkHdr;

	// Read the block into memory
	
	if (RC_BAD( rc = gv_pSFileHdl->readBlock( uiBlkAddress, uiReadLen,
												pBlkHdr, puiBytesRead)))
	{
		if (rc == NE_XFLM_IO_END_OF_FILE)
		{
			rc = NE_XFLM_OK;
			f_memset( (FLMBYTE *)pBlkHdr + *puiBytesRead, 0xEE,
								uiReadLen - *puiBytesRead);
			if (bShowPartialReadError)
			{
				if (!(*puiBytesRead))
				{
					ViewShowRCError( "reading block", NE_XFLM_IO_END_OF_FILE);
					goto Exit;
				}
				else if (*puiBytesRead < uiReadLen)
				{
					f_sprintf( szErrMsg,
						"Only %u bytes of data were read (requested %u)",
							(unsigned)*puiBytesRead, (unsigned)uiReadLen);
					ViewShowError( szErrMsg);
				}
			}
		}
		else
		{
			ViewShowRCError( "reading block", rc);
			goto Exit;
		}
	}

	// Calculate the CRC on the block.

	if (bOkToConvert)
	{
		if (pui32CalcCRC)
		{
			ui16BlkBytesAvail = pBlkHdr->ui16BlkBytesAvail;
			if (blkIsNonNativeFormat( pBlkHdr))
			{
				convert16( &ui16BlkBytesAvail);
			}

			if( ui16BlkBytesAvail > gv_ViewDbHdr.ui16BlockSize)
			{
				*pui32CalcCRC = 0;
			}
			else
			{
				uiBlkEnd = (blkIsNewBTree( pBlkHdr)
								? gv_ViewDbHdr.ui16BlockSize
								: gv_ViewDbHdr.ui16BlockSize - (FLMUINT)ui16BlkBytesAvail);
				*pui32CalcCRC = calcBlkCRC( pBlkHdr, uiBlkEnd);
			}
		}

		if (blkIsNonNativeFormat( pBlkHdr))
		{
			convertBlk( (FLMUINT)gv_ViewDbHdr.ui16BlockSize, pBlkHdr);
		}

		if (pui32BlkCRC)
		{
			*pui32BlkCRC = pBlkHdr->ui32BlkCRC;
		}
	}

	bOk = TRUE;

Exit:

	return( bOk);
}

/***************************************************************************
Desc: This routine searches through the LFH blocks searching for the
		LFH of a particular logical file.
*****************************************************************************/
FLMBOOL ViewGetLFH(
	FLMUINT		uiLfNum,
	eLFileType	eLfType,
	F_LF_HDR *	pLfHdr,
	FLMUINT *	puiFileOffset)
{
	FLMUINT		uiBlkAddress;
	FLMUINT		uiBlkCount = 0;
	F_BLK_HDR *	pBlkHdr = NULL;
	F_LF_HDR *	pTmpLfHdr;
	FLMUINT		uiEndOfBlock;
	FLMUINT		uiPos;
	FLMUINT		uiBytesRead;
	FLMBOOL		bGotLFH = FALSE;

	// Read the LFH blocks and get the information needed
	// If we read too many, the file is probably corrupt.

	*puiFileOffset = 0;
	uiBlkAddress = (FLMUINT)gv_ViewDbHdr.ui32FirstLFBlkAddr;
	while (uiBlkAddress)
	{
		if (!ViewBlkRead( uiBlkAddress, &pBlkHdr, TRUE,
								(FLMUINT)gv_ViewDbHdr.ui16BlockSize,
								NULL, NULL, &uiBytesRead, FALSE) ||
			 !uiBytesRead)
		{
			break;
		}

		// Count the blocks read to prevent too many from being read.
		// We don't want to get into an infinite loop if the database
		// is corrupted.

		uiBlkCount++;

		// Search through the block for the particular LFH which matches
		// the one we are looking for.

		if (uiBytesRead <= SIZEOF_STD_BLK_HDR)
		{
			uiEndOfBlock = SIZEOF_STD_BLK_HDR;
		}
		else
		{
			uiEndOfBlock = blkGetEnd( (FLMUINT)gv_ViewDbHdr.ui16BlockSize,
									SIZEOF_STD_BLK_HDR, pBlkHdr);
			if (uiEndOfBlock > uiBytesRead)
			{
				uiEndOfBlock = uiBytesRead;
			}
		}
		uiPos = SIZEOF_STD_BLK_HDR;
		pTmpLfHdr = (F_LF_HDR *)((FLMBYTE *)pBlkHdr + SIZEOF_STD_BLK_HDR);
		while (uiPos < uiEndOfBlock)
		{

			// See if we got the one we wanted

			if (pTmpLfHdr->ui32LfType != XFLM_LF_INVALID &&
				 (FLMUINT)pTmpLfHdr->ui32LfNumber == uiLfNum &&
				 (FLMUINT)pTmpLfHdr->ui32LfType == (FLMUINT)eLfType)
			{
				f_memcpy( pLfHdr, pTmpLfHdr, sizeof( F_LF_HDR));
				bGotLFH = TRUE;
				*puiFileOffset = uiBlkAddress + uiPos;
				break;
			}

			uiPos += sizeof( F_LF_HDR);
			pTmpLfHdr++;
		}

		// If we didn't end right on end of block, return

		if (uiPos != uiEndOfBlock || bGotLFH)
		{
			break;
		}

		// If we have traversed too many blocks, things are probably corrupt

		if (uiBlkCount > 1000)
		{
			break;
		}
		if (F_BLK_HDR_ui32NextBlkInChain_OFFSET + 4 <= uiBytesRead)
		{
			uiBlkAddress = pBlkHdr->ui32NextBlkInChain;
		}
		else
		{
			uiBlkAddress = 0;
		}
	}

	// Be sure to free the block -- if one was allocated

	f_free( &pBlkHdr);
	return( bGotLFH);
}

/***************************************************************************
Desc: This routine attempts to find an LFH for a particular logical
		file.  It then extracts the name from the LFH.
*****************************************************************************/
FLMBOOL ViewGetLFName(
	char *			pszName,
	FLMUINT			uiLfNum,
	eLFileType		eLfType,
	F_LF_HDR *		pLfHdr,
	FLMUINT *		puiFileOffset)
{
	f_sprintf( pszName, "lfNum=%u", (unsigned)uiLfNum);
	if (!uiLfNum || !ViewGetLFH( uiLfNum, eLfType, pLfHdr, puiFileOffset))
	{
		*puiFileOffset = 0;
		f_memset( pLfHdr, 0, sizeof( F_LF_HDR));
		return( FALSE);
	}
	return( TRUE);
}

/***************************************************************************
Desc: This routine outputs one of the number fields in the block
		header.  It checks the number against an expected value, and if
		the number does not match the expected value also outputs the
		value which was expected.  This routine is used to output values
		in the block header where we expect certain values.
*****************************************************************************/
FSTATIC FLMBOOL OutBlkHdrExpNum(
	FLMUINT			uiCol,
	FLMUINT *		puiRow,
	eColorType		uiBackColor,
	eColorType		uiForeColor,
	eColorType		uiUnselectBackColor,
	eColorType		uiUnselectForeColor,
	eColorType		uiSelectBackColor,
	eColorType		uiSelectForeColor,
	FLMUINT			uiLabelWidth,
	FLMINT			iLabelIndex,
	FLMUINT			uiFileNumber,
	FLMUINT			uiFileOffset,
	void *			pvValue,
	FLMUINT			uiValueType,
	FLMUINT			uiModType,
	FLMUINT64		ui64ExpNum,
	FLMUINT64		ui64IgnoreExpNum,
	FLMUINT			uiOption)
{
	FLMBOOL		bOk = FALSE;
	FLMUINT		uiRow = *puiRow;
	FLMUINT64	ui64Num = 0;

	if (!uiOption)
	{
		uiUnselectBackColor = uiSelectBackColor = uiBackColor;
		uiUnselectForeColor = uiSelectForeColor = uiForeColor;
	}
	switch (uiModType & 0x0F)
	{
		case MOD_FLMUINT64:
			ui64Num = *((FLMUINT64 *)pvValue);
			break;
		case MOD_FLMUINT32:
			ui64Num = (FLMUINT64)(*((FLMUINT32 *)pvValue));
			break;
		case MOD_FLMUINT16:
			ui64Num = (FLMUINT64)(*((FLMUINT16 *)pvValue));
			break;
		case MOD_FLMBYTE:
			ui64Num = (FLMUINT64)(*((FLMUINT8 *)pvValue));
			break;
	}
	if (!ViewAddMenuItem( iLabelIndex, uiLabelWidth,
				uiValueType, ui64Num, 0,
				uiFileNumber, uiFileOffset, 0,
				uiModType | MOD_NATIVE,
				uiCol, uiRow++, uiOption,
				uiUnselectBackColor, uiUnselectForeColor,
				uiSelectBackColor, uiSelectForeColor))
	{
		goto Exit;
	}

	if (ui64ExpNum != ui64IgnoreExpNum && ui64Num != ui64ExpNum)
	{
		if (!ViewAddMenuItem( LBL_EXPECTED, 0,
					uiValueType, ui64ExpNum, 0,
					0, VIEW_INVALID_FILE_OFFSET, 0, MOD_DISABLED,
					uiCol + uiLabelWidth + 1, uiRow++, 0,
					FLM_RED, FLM_LIGHTGRAY,
					FLM_RED, FLM_LIGHTGRAY))
		{
			goto Exit;
		}
	}
	*puiRow = uiRow;
	bOk = TRUE;

Exit:

	return( bOk);
}

/***************************************************************************
Desc:	This routine formats a block's type into ASCII.
*****************************************************************************/
FSTATIC void FormatBlkType(
	char *		pszTempBuf,
	FLMUINT		uiBlkType
	)
{
	switch (uiBlkType)
	{
		case BT_FREE:
			f_strcpy( pszTempBuf, "Free");
			break;
		case BT_LEAF:
			f_strcpy( pszTempBuf, "Leaf");
			break;
		case BT_NON_LEAF:
			f_strcpy( pszTempBuf, "Non-Leaf");
			break;
		case BT_NON_LEAF_COUNTS:
			f_strcpy( pszTempBuf, "Non-Leaf /w Counts");
			break;
		case BT_LEAF_DATA:
			f_strcpy( pszTempBuf, "Leaf /w Data");
			break;
		case BT_DATA_ONLY:
			f_strcpy( pszTempBuf, "Data Only");
			break;
		case BT_LFH_BLK:
			f_strcpy( pszTempBuf, "LFH");
			break;
		default:
			f_sprintf( pszTempBuf,
				"Unknown Type: %u", (unsigned)uiBlkType);
			break;
	}
}

/***************************************************************************
Desc:	This routine outputs a block's header.
*****************************************************************************/
FLMBOOL ViewOutBlkHdr(
	FLMUINT		uiCol,
	FLMUINT *	puiRow,
	F_BLK_HDR *	pBlkHdr,
	BLK_EXP_p	pBlkExp,
	FLMBYTE *	pucBlkStatus,
	FLMUINT32	ui32CalcCRC,
	FLMUINT32	ui32BlkCRC
	)
{
	FLMBOOL			bOk = FALSE;
	FLMUINT			uiLabelWidth = 35;
	FLMUINT			uiRow = *puiRow;
	char				szTempBuf [80];
	FLMUINT			uiBlkAddress;
	FLMUINT			uiEndOfBlock;
	FLMUINT			uiBytesUsed;
	FLMUINT			uiPercentFull;
	FLMUINT			uiOption;
	eColorType		uiBackColor = FLM_BLACK;
	eColorType		uiForeColor = FLM_LIGHTGRAY;
	eColorType		uiUnselectBackColor = FLM_BLACK;
	eColorType		uiUnselectForeColor = FLM_WHITE;
	eColorType		uiSelectBackColor = FLM_BLUE;
	eColorType		uiSelectForeColor = FLM_WHITE;
	F_LF_HDR			lfHdr;
	char				szLfName [80];
	FLMUINT			uiLfNum;
	eLFileType		eLfType;
	FLMUINT			uiTempFileOffset;

	// Output the block Header address

	if (!OutBlkHdrExpNum( uiCol, &uiRow, FLM_RED, FLM_LIGHTGRAY,
			FLM_RED, FLM_WHITE, uiSelectBackColor, uiSelectForeColor,
			uiLabelWidth, LBL_BLOCK_ADDRESS_BLOCK_HEADER,
			FSGetFileNumber( pBlkExp->uiBlkAddr), 
			FSGetFileOffset( pBlkExp->uiBlkAddr),
			&pBlkHdr->ui32BlkAddr,
			VAL_IS_NUMBER | DISP_HEX_DECIMAL,
			MOD_FLMUINT32 | MOD_HEX | MOD_NATIVE,
			(FLMUINT64)pBlkExp->uiBlkAddr, 0, 0))
	{
		goto Exit;
	}

	// Adjust column so rest of header is indented

	uiCol += 2;
	uiLabelWidth -= 2;

	// Output the previous block address

	if (!pBlkHdr->ui32PrevBlkInChain)
	{
		uiOption = 0;
	}
	else
	{
		uiOption = PREV_BLOCK_OPTION;
	}
	if (!OutBlkHdrExpNum( uiCol, &uiRow, uiBackColor, uiForeColor,
				uiUnselectBackColor, uiUnselectForeColor,
				uiSelectBackColor, uiSelectForeColor,
				uiLabelWidth, LBL_PREVIOUS_BLOCK_ADDRESS,
				FSGetFileNumber( pBlkExp->uiBlkAddr), 
				FSGetFileOffset( pBlkExp->uiBlkAddr) +
				F_BLK_HDR_ui32PrevBlkInChain_OFFSET,
				&pBlkHdr->ui32PrevBlkInChain,
				VAL_IS_NUMBER | DISP_HEX_DECIMAL,
				MOD_FLMUINT32 | MOD_HEX | MOD_NATIVE,
				(FLMUINT64)pBlkExp->uiPrevAddr, 0xFFFFFFFF, uiOption))
	{
		goto Exit;
	}

	// Output the next block address

	if (!pBlkHdr->ui32NextBlkInChain)
	{
		uiOption = 0;
	}
	else
	{
		uiOption = NEXT_BLOCK_OPTION;
	}
	if (!OutBlkHdrExpNum( uiCol, &uiRow, uiBackColor, uiForeColor,
				uiUnselectBackColor, uiUnselectForeColor,
				uiSelectBackColor, uiSelectForeColor,
				uiLabelWidth, LBL_NEXT_BLOCK_ADDRESS,
				FSGetFileNumber( pBlkExp->uiBlkAddr), 
				FSGetFileOffset( pBlkExp->uiBlkAddr) +
				F_BLK_HDR_ui32NextBlkInChain_OFFSET,
				&pBlkHdr->ui32NextBlkInChain,
				VAL_IS_NUMBER | DISP_HEX_DECIMAL,
				MOD_FLMUINT32 | MOD_HEX | MOD_NATIVE,
				(FLMUINT64)pBlkExp->uiNextAddr, 0xFFFFFFFF, uiOption))
	{
		goto Exit;
	}

	// Output the prior block image address

	uiBlkAddress = (FLMUINT)pBlkHdr->ui32PriorBlkImgAddr;
	if (uiBlkAddress == 0 ||
			pBlkHdr->ui64TransID <= gv_ViewDbHdr.ui64CurrTransID)
	{
		uiOption = 0;
	}
	else
	{
		uiOption = PREV_BLOCK_IMAGE_OPTION;
	}
	if (!ViewAddMenuItem( LBL_OLD_BLOCK_IMAGE_ADDRESS, uiLabelWidth,
				VAL_IS_NUMBER | DISP_HEX_DECIMAL,
				(FLMUINT64)uiBlkAddress, 0,
				FSGetFileNumber( pBlkExp->uiBlkAddr),
				FSGetFileOffset( pBlkExp->uiBlkAddr) +
				F_BLK_HDR_ui32PriorBlkImgAddr_OFFSET, 0,
				MOD_FLMUINT32 | MOD_HEX | MOD_NATIVE,
				uiCol, uiRow++, uiOption,
				(!uiOption ? uiBackColor : uiUnselectBackColor),
				(!uiOption ? uiForeColor : uiUnselectForeColor),
				(!uiOption ? uiBackColor : uiSelectBackColor),
				(!uiOption ? uiForeColor : uiSelectForeColor)))
	{
		goto Exit;
	}

	// Output the block transaction ID

	if (!ViewAddMenuItem( LBL_BLOCK_TRANS_ID, uiLabelWidth,
				VAL_IS_NUMBER | DISP_DECIMAL,
				pBlkHdr->ui64TransID, 0,
				FSGetFileNumber( pBlkExp->uiBlkAddr),
				FSGetFileOffset( pBlkExp->uiBlkAddr) +
				F_BLK_HDR_ui64TransID_OFFSET, 0,
				MOD_FLMUINT32 | MOD_DECIMAL | MOD_NATIVE,
				uiCol, uiRow++, 0, uiBackColor, uiForeColor,
				uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Output the block CRC

	if (!OutBlkHdrExpNum( uiCol, &uiRow, uiBackColor, uiForeColor,
			uiUnselectBackColor, uiUnselectForeColor,
			uiSelectBackColor, uiSelectForeColor,
			uiLabelWidth, LBL_BLOCK_CRC,
			FSGetFileNumber( pBlkExp->uiBlkAddr), 
			FSGetFileOffset( pBlkExp->uiBlkAddr) +
			F_BLK_HDR_ui32BlkCRC_OFFSET,
			&ui32BlkCRC,
			VAL_IS_NUMBER | DISP_DECIMAL,
			MOD_FLMUINT32 | MOD_DECIMAL | MOD_NATIVE,
			(FLMUINT64)ui32CalcCRC, 0, 0))
	{
		return( 0);
	}

	// Output the available bytes in the block

	if (!ViewAddMenuItem( LBL_BLOCK_BYTES_AVAIL, uiLabelWidth,
				VAL_IS_NUMBER | DISP_DECIMAL,
				(FLMUINT64)pBlkHdr->ui16BlkBytesAvail, 0,
				FSGetFileNumber( pBlkExp->uiBlkAddr),
				FSGetFileOffset( pBlkExp->uiBlkAddr) +
				F_BLK_HDR_ui16BlkBytesAvail_OFFSET, 0,
				MOD_FLMUINT16 | MOD_DECIMAL | MOD_NATIVE,
				uiCol, uiRow++, 0, uiBackColor, uiForeColor,
				uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Output the heap bytes in the block
	if ((pBlkHdr->ui8BlkType == BT_LEAF) ||
		 (pBlkHdr->ui8BlkType == BT_LEAF_DATA) ||
		 (pBlkHdr->ui8BlkType == BT_NON_LEAF) ||
		 (pBlkHdr->ui8BlkType == BT_NON_LEAF_COUNTS))
	{

		if (!ViewAddMenuItem( LBL_BLOCK_HEAP_SIZE, uiLabelWidth,
					VAL_IS_NUMBER | DISP_DECIMAL,
					(FLMUINT64)((F_BTREE_BLK_HDR *)pBlkHdr)->ui16HeapSize, 0,
					FSGetFileNumber( pBlkExp->uiBlkAddr),
					FSGetFileOffset( pBlkExp->uiBlkAddr) +
					F_BTREE_BLK_HDR_ui16HeapSize_OFFSET, 0,
					MOD_FLMUINT16 | MOD_DECIMAL | MOD_NATIVE,
					uiCol, uiRow++, 0, uiBackColor, uiForeColor,
					uiBackColor, uiForeColor))
		{
			goto Exit;
		}
	}

	uiEndOfBlock = blkGetEnd( (FLMUINT)gv_ViewDbHdr.ui16BlockSize,
							blkHdrSize( pBlkHdr), pBlkHdr);

	// Output the percent full

	if (blkIsNewBTree( pBlkHdr))
	{
		uiBytesUsed = gv_ViewDbHdr.ui16BlockSize -
			pBlkHdr->ui16BlkBytesAvail - blkHdrSize( pBlkHdr);
	}
	else
	{
		uiBytesUsed = uiEndOfBlock - blkHdrSize( pBlkHdr);
	}

	if (!uiBytesUsed || uiEndOfBlock < blkHdrSize( pBlkHdr))
	{
		uiPercentFull = 0;
	}
	else if (uiEndOfBlock > (FLMUINT)gv_ViewDbHdr.ui16BlockSize)
	{
		uiPercentFull = 100;
	}
	else
	{
		uiPercentFull = ((FLMUINT)(uiBytesUsed) * (FLMUINT)(100)) /
				 ((FLMUINT)(gv_ViewDbHdr.ui16BlockSize) -
				  blkHdrSize( pBlkHdr));
	}
	if (!ViewAddMenuItem( LBL_PERCENT_FULL, uiLabelWidth,
			VAL_IS_NUMBER | DISP_DECIMAL,
			(FLMUINT64)uiPercentFull, 0,
			0, VIEW_INVALID_FILE_OFFSET, 0, MOD_DISABLED,
			uiCol, uiRow++, 0, uiBackColor, uiForeColor,
			uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	if (!ViewAddMenuItem( LBL_BLOCK_END, uiLabelWidth,
				VAL_IS_NUMBER | DISP_HEX_DECIMAL,
				(FLMUINT64)(FSGetFileOffset( pBlkExp->uiBlkAddr) + uiEndOfBlock),
				0,
				FSGetFileNumber( pBlkExp->uiBlkAddr),
				FSGetFileOffset( pBlkExp->uiBlkAddr) + uiEndOfBlock,
				0, MOD_DISABLED, uiCol, uiRow++,
				0, uiBackColor, uiForeColor,
				uiBackColor, uiForeColor))
	{
		goto Exit;
	}


	// Output block flags here

	if (pBlkHdr->ui8BlkFlags & BLK_FORMAT_IS_LITTLE_ENDIAN)
	{
		f_strcpy( szTempBuf, "Little-Endian");
	}
	else
	{
		f_strcpy( szTempBuf, "Big-Endian");
	}
	if (pBlkHdr->ui8BlkFlags & BLK_IS_BEFORE_IMAGE)
	{
		f_strcpy( &szTempBuf [f_strlen( szTempBuf)], ", Before-Image");
	}
	if (!ViewAddMenuItem( LBL_BLOCK_FLAGS, uiLabelWidth,
			VAL_IS_TEXT_PTR,
			(FLMUINT)&szTempBuf[0], 0,
			FSGetFileNumber( pBlkExp->uiBlkAddr),
			FSGetFileOffset( pBlkExp->uiBlkAddr) +
			F_BLK_HDR_ui8BlkFlags_OFFSET, 0,
			MOD_FLMBYTE | MOD_HEX | MOD_NATIVE,
			uiCol, uiRow++, 0, uiBackColor, uiForeColor,
			uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Output the block type

	FormatBlkType( szTempBuf, (FLMUINT)pBlkHdr->ui8BlkType);
	if (pBlkHdr->ui8BlkType != (FLMUINT8)pBlkExp->uiType)
	{
		f_strcpy( &szTempBuf [f_strlen( szTempBuf)], ", Expecting ");
		FormatBlkType( &szTempBuf [f_strlen( szTempBuf)], pBlkExp->uiType);
	}
	if (!ViewAddMenuItem( LBL_BLOCK_TYPE, uiLabelWidth,
			VAL_IS_TEXT_PTR,
			(FLMUINT)(&szTempBuf [0]), 0,
			FSGetFileNumber( pBlkExp->uiBlkAddr),
			FSGetFileOffset( pBlkExp->uiBlkAddr) +
			F_BLK_HDR_ui8BlkType_OFFSET, 0,
			MOD_FLMBYTE | MOD_HEX | MOD_NATIVE,
			uiCol, uiRow++, 0, uiBackColor, uiForeColor,
			uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	if (blkIsBTree( pBlkHdr) && (pBlkHdr->ui8BlkType != BT_DATA_ONLY))
	{

		// Output the logical file this block belongs to - if any

		uiLfNum = (FLMUINT)(((F_BTREE_BLK_HDR *)pBlkHdr)->ui16LogicalFile);
		eLfType = getBlkLfType( (F_BTREE_BLK_HDR *)pBlkHdr);
		if (!ViewGetLFName( szLfName, uiLfNum, eLfType,
									&lfHdr, &uiTempFileOffset))
		{
			eLfType = XFLM_LF_INVALID;
			uiOption = 0;
		}
		else
		{
			if (eLfType == XFLM_LF_INDEX)
			{
				uiOption = LOGICAL_INDEX_OPTION | uiLfNum;
			}
			else
			{
				uiOption = LOGICAL_CONTAINER_OPTION | uiLfNum;
			}
		}
		if (pBlkExp->uiLfNum && uiLfNum != pBlkExp->uiLfNum)
		{
			f_sprintf( (char *)szTempBuf, "%s (Expected %u)", szLfName,
						(unsigned)pBlkExp->uiLfNum);
		}
		else
		{
			f_strcpy( szTempBuf, szLfName);
		}
		if (!ViewAddMenuItem( LBL_BLOCK_LOGICAL_FILE_NAME, uiLabelWidth,
					VAL_IS_TEXT_PTR,
					(FLMUINT)((FLMBYTE *)(&szTempBuf [0])), 0,
					FSGetFileNumber( pBlkExp->uiBlkAddr),
					FSGetFileOffset( pBlkExp->uiBlkAddr) +
					F_BTREE_BLK_HDR_ui16LogicalFile_OFFSET, 0,
					MOD_FLMUINT16 | MOD_DECIMAL | MOD_NATIVE,
					uiCol, uiRow++, uiOption,
					(!uiOption ? uiBackColor : uiUnselectBackColor),
					(!uiOption ? uiForeColor : uiUnselectForeColor),
					(!uiOption ? uiBackColor : uiSelectBackColor),
					(!uiOption ? uiForeColor : uiSelectForeColor)))
		{
			goto Exit;
		}

		// Output the logical file type

		FormatLFType( szTempBuf, eLfType);
		if (!ViewAddMenuItem( LBL_BLOCK_LOGICAL_FILE_TYPE, uiLabelWidth,
					VAL_IS_TEXT_PTR,
					(FLMUINT)((FLMBYTE *)(&szTempBuf [0])), 0,
					FSGetFileNumber( pBlkExp->uiBlkAddr),
					FSGetFileOffset( pBlkExp->uiBlkAddr) +
					F_BTREE_BLK_HDR_ui8BTreeFlags_OFFSET, 0,
					MOD_FLMBYTE | MOD_HEX | MOD_NATIVE,
					uiCol, uiRow++, 0, uiBackColor, uiForeColor,
					uiBackColor, uiForeColor))
		{
			goto Exit;
		}

		// Output number of keys

		if (!ViewAddMenuItem( LBL_BLOCK_NUM_KEYS, uiLabelWidth,
					VAL_IS_NUMBER | DISP_DECIMAL,
					(FLMUINT64)(((F_BTREE_BLK_HDR *)pBlkHdr)->ui16NumKeys), 0,
					FSGetFileNumber( pBlkExp->uiBlkAddr),
					FSGetFileOffset( pBlkExp->uiBlkAddr) +
					F_BTREE_BLK_HDR_ui16NumKeys_OFFSET, 0,
					MOD_FLMUINT16 | MOD_DECIMAL | MOD_NATIVE,
					uiCol, uiRow++, 0, uiBackColor, uiForeColor,
					uiBackColor, uiForeColor))
		{
			goto Exit;
		}

		// Output block level

		if (!OutBlkHdrExpNum( uiCol, &uiRow,
				uiBackColor, uiForeColor,
				uiUnselectBackColor, uiUnselectForeColor,
				uiSelectBackColor, uiSelectForeColor,
				uiLabelWidth, LBL_B_TREE_LEVEL,
				FSGetFileNumber( pBlkExp->uiBlkAddr), 
				FSGetFileOffset( pBlkExp->uiBlkAddr) +
				F_BTREE_BLK_HDR_ui8BlkLevel_OFFSET,
				&(((F_BTREE_BLK_HDR *)pBlkHdr)->ui8BlkLevel),
				VAL_IS_NUMBER | DISP_DECIMAL,
				MOD_FLMBYTE | MOD_DECIMAL | MOD_NATIVE,
				(FLMUINT64)pBlkExp->uiLevel, (FLMUINT64)0xFF, 0))
		{
			goto Exit;
		}

		// Output if block is a root block

		if (!ViewAddMenuItem( LBL_BLOCK_ROOT, uiLabelWidth,
					VAL_IS_LABEL_INDEX,
					(FLMUINT64)(isRootBlk( (F_BTREE_BLK_HDR *)pBlkHdr)
									? (FLMUINT64)LBL_YES
									: (FLMUINT64)LBL_NO), 0,
					FSGetFileNumber( pBlkExp->uiBlkAddr),
					FSGetFileOffset( pBlkExp->uiBlkAddr) +
					F_BTREE_BLK_HDR_ui8BTreeFlags_OFFSET, 0,
					MOD_FLMBYTE | MOD_HEX | MOD_NATIVE,
					uiCol, uiRow++, 0, uiBackColor, uiForeColor,
					uiBackColor, uiForeColor))
		{
			goto Exit;
		}

		// Output if block is encrypted

		if (!ViewAddMenuItem( LBL_BLOCK_ENCRYPTED, uiLabelWidth,
					VAL_IS_LABEL_INDEX,
					(FLMUINT64)(isEncryptedBlk( pBlkHdr)
									? (FLMUINT64)LBL_YES
									: (FLMUINT64)LBL_NO), 0,
					FSGetFileNumber( pBlkExp->uiBlkAddr),
					FSGetFileOffset( pBlkExp->uiBlkAddr) +
					F_BTREE_BLK_HDR_ui8BTreeFlags_OFFSET, 0,
					MOD_FLMBYTE | MOD_HEX | MOD_NATIVE,
					uiCol, uiRow++, 0, uiBackColor, uiForeColor,
					uiBackColor, uiForeColor))
		{
			goto Exit;
		}

	}

	// Output the flags which indicate the state of the block

	if (!OutputStatus( uiCol, &uiRow, uiBackColor, uiForeColor,
					uiLabelWidth, LBL_BLOCK_STATUS, pucBlkStatus))
	{
		goto Exit;
	}

	*puiRow = uiRow + 1;
	bOk = TRUE;

Exit:

	return( bOk);
}

/***************************************************************************
Desc:	This routine displays a block in the AVAIL list.
*****************************************************************************/
FLMBOOL ViewAvailBlk(
	FLMUINT			uiReadAddress,
	FLMUINT			uiBlkAddress,
	F_BLK_HDR **	ppBlkHdr,
	BLK_EXP_p		pBlkExp
	)
{
	FLMBOOL			bOk = FALSE;
	FLMUINT			uiRow;
	FLMUINT			uiCol;
	F_BLK_HDR *		pBlkHdr;
	FLMBYTE			ucBlkStatus [NUM_STATUS_BYTES];
	STATE_INFO		StateInfo;
//	FLMBOOL			bStateInitialized = FALSE;
//	BLOCK_INFO		BlockInfo;
//	FLMINT			iErrorCode;
	FLMUINT32		ui32CalcCRC;
	FLMUINT32		ui32BlkCRC;
	FLMUINT			uiBytesRead;

	// Read the block into memory

	if (!ViewBlkRead( uiReadAddress, ppBlkHdr, TRUE,
							(FLMUINT)gv_ViewDbHdr.ui16BlockSize,
							&ui32CalcCRC, &ui32BlkCRC, &uiBytesRead, TRUE))
	{
		goto Exit;
	}
	pBlkHdr = *ppBlkHdr;

	ViewMenuInit( "AVAIL Block");

	// Output the block header first

	uiRow = 0;
	uiCol = 5;
	pBlkExp->uiType = BT_FREE;
	pBlkExp->uiLfNum = 0;
	pBlkExp->uiBlkAddr = uiBlkAddress;
	pBlkExp->uiLevel = 0xFF;

	// Setup the STATE variable for processing through the block

	InitStatusBits( ucBlkStatus);
#if 0
	flmInitReadState( &StateInfo, &bStateInitialized,
							(FLMUINT)gv_ViewDbHdr.ui32DbVersion,
							(gv_bViewDbInitialized)
							? gv_hViewDb
							: NULL,
							NULL, 0, BT_FREE, NULL);
#endif
	StateInfo.ui32BlkAddress = (FLMUINT32)uiBlkAddress;
	StateInfo.pBlkHdr = pBlkHdr;
#if 0
	if ((iErrorCode = flmVerifyBlockHeader( &StateInfo, &BlockInfo,
											 (FLMUINT)gv_ViewDbHdr.ui16BlockSize,
											 pBlkExp->uiNextAddr,
											 0xFFFFFFFF,
											 (FLMBOOL)(StateInfo.pDb != NULL
															  ? TRUE
															  : FALSE))) != 0)
	{
		SetStatusBit( ucBlkStatus, iErrorCode);
	}
#endif
	if (!ViewOutBlkHdr( uiCol, &uiRow, pBlkHdr, pBlkExp, ucBlkStatus,
								ui32CalcCRC, ui32BlkCRC))
	{
		goto Exit;
	}
	bOk = TRUE;

Exit:

	return( bOk);
}

/***************************************************************************
Desc: This routine outputs a stream of FLMBYTEs in hex format.  This
		routine is used to output key values and records within an
		element.
*****************************************************************************/
FLMBOOL OutputHexValue(
	FLMUINT			uiCol,
	FLMUINT *		puiRow,
	eColorType		uiBackColor,
	eColorType		uiForeColor,
	FLMINT			iLabelIndex,
	FLMUINT			uiFileNumber,
	FLMUINT			uiFileOffset,
	FLMBYTE *		pucVal,
	FLMUINT			uiValLen,
	FLMBOOL			bCopyVal)
{
	FLMBOOL		bOk = FALSE;
	FLMUINT		uiRow = *puiRow;
	FLMUINT		uiBytesPerLine = MAX_HORIZ_SIZE( uiCol + 3);
	FLMBOOL		bFirstTime = TRUE;
	FLMUINT		uiBytesProcessed = 0;
	FLMUINT		uiNumBytes;

	while (uiBytesProcessed < uiValLen)
	{
		if ((uiNumBytes = uiValLen - uiBytesProcessed) > uiBytesPerLine)
		{
			uiNumBytes = uiBytesPerLine;
		}

		// Output the line

		if (bFirstTime)
		{
			if (!ViewAddMenuItem( iLabelIndex, 0,
										VAL_IS_EMPTY, 0, 0,
										uiFileNumber, uiFileOffset, 0, MOD_DISABLED,
										uiCol, uiRow++, 0, uiBackColor, uiForeColor,
										uiBackColor, uiForeColor))
			{
				goto Exit;
			}
			bFirstTime = FALSE;
			uiCol += 2;
		}
		if (!ViewAddMenuItem( -1, 0,
					(FLMBYTE)((bCopyVal)
								? (FLMBYTE)VAL_IS_BINARY
								: (FLMBYTE)VAL_IS_BINARY_PTR),
					(FLMUINT)pucVal, uiNumBytes,
					uiFileNumber, uiFileOffset, uiNumBytes, MOD_BINARY,
					uiCol, uiRow++, 0, uiBackColor, uiForeColor,
					uiBackColor, uiForeColor))
		{
			goto Exit;
		}
		uiFileOffset += uiNumBytes;
		uiBytesProcessed += uiNumBytes;
		pucVal += uiNumBytes;
	}
	*puiRow = uiRow;
	bOk = TRUE;

Exit:

	return( bOk);
}

/***************************************************************************
Desc: This routine outputs the content of a DOM node in the Element
*****************************************************************************/
FSTATIC FLMBOOL OutputDomNode(
	FLMUINT				uiCol,
	FLMUINT *			puiRow,
	eColorType			uiBackColor,
	eColorType			uiForeColor,
	FLMBYTE *			pucVal,
	STATE_INFO *		pStateInfo)
{
	RCODE					rc = NE_XFLM_OK;
	FLMBOOL				bOk = FALSE;
	FLMUINT				uiLabelWidth = 30;
	FLMUINT				uiRow = *puiRow;
	FLMUINT				uiNumBytes;
	FLMBYTE *			pucData = pucVal;
	FLMUINT				uiDataOffset = pStateInfo->uiElmDataOffset;
	FLMBYTE				ucFlags = 0;
	FLMBYTE				ucExtFlags = 0;
	FLMUINT				uiSENLength = 0;
	const FLMBYTE *	pucTmp;
	FLMUINT64			ui64Value;
	FLMUINT				uiValue;
	
	// VISIT: change to use flmReadNodeInfo

	if ((uiNumBytes = pStateInfo->uiElmDataLen) == 0)
	{
		bOk = TRUE;
		goto Exit;
	}

	// Display the Node Type.  The first byte is encoded as RDDDNNNN
	// where R = Reserved, DDD = Data Type, NNNN = Node Type

	// Display the Data Type
	if (!ViewAddMenuItem( LBL_DOM_DATA_TYPE, uiLabelWidth,
			VAL_IS_NUMBER | DISP_HEX_DECIMAL,
			(*pucData & 0x70) >> 4, 0,
			FSGetFileNumber( pStateInfo->ui32BlkAddress),
			FSGetFileOffset( pStateInfo->ui32BlkAddress) + uiDataOffset,
			0, MOD_DISABLED,
			uiCol, uiRow++, 0,
			uiBackColor, uiForeColor,
			uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	// Display the Node Type
	if (!ViewAddMenuItem( LBL_DOM_NODE_TYPE, uiLabelWidth,
			VAL_IS_NUMBER | DISP_HEX_DECIMAL,
			*pucData & 0x0F, 0,
			FSGetFileNumber( pStateInfo->ui32BlkAddress),
			FSGetFileOffset( pStateInfo->ui32BlkAddress) + uiDataOffset,
			0, MOD_DISABLED,
			uiCol, uiRow++, 0,
			uiBackColor, uiForeColor,
			uiBackColor, uiForeColor))
	{
		goto Exit;
	}

	pucData++;
	uiDataOffset++;
	uiNumBytes--;

	if (!uiNumBytes)
	{
		if (!ViewAddMenuItem( LBL_DOM_FLAGS, uiLabelWidth,
				VAL_IS_LABEL_INDEX,
				LBL_NO_VALUE, 0,
				0, VIEW_INVALID_FILE_OFFSET,
				0, MOD_DISABLED,
				uiCol, uiRow++, 0, uiBackColor, uiForeColor,
				uiBackColor, uiForeColor))
		{
			goto Exit;
		}
	}
	else
	{
		ucFlags = *pucData;
	
		// Display the flags
		if (!ViewAddMenuItem( LBL_DOM_FLAGS, uiLabelWidth,
				VAL_IS_NUMBER | DISP_HEX_DECIMAL,
				*pucData, 0,
				FSGetFileNumber( pStateInfo->ui32BlkAddress),
				FSGetFileOffset( pStateInfo->ui32BlkAddress) + uiDataOffset,
				0, MOD_DISABLED,
				uiCol, uiRow++, 0,
				uiBackColor, uiForeColor,
				uiBackColor, uiForeColor))
		{
			goto Exit;
		}
	
		pucData++;
		uiDataOffset++;
		uiNumBytes--;
	}

	// Name ID
	if (!uiNumBytes)
	{
		if (!ViewAddMenuItem( LBL_DOM_NAME_ID, uiLabelWidth,
				VAL_IS_LABEL_INDEX,
				LBL_NO_VALUE, 0,
				0, VIEW_INVALID_FILE_OFFSET,
				0, MOD_DISABLED,
				uiCol, uiRow++, 0, uiBackColor, uiForeColor,
				uiBackColor, uiForeColor))
		{
			goto Exit;
		}
	}
	else
	{
		pucTmp = pucData;
		uiSENLength = f_getSENLength( *pucTmp);
		if (uiNumBytes >= uiSENLength)
		{
			if( RC_BAD( rc = f_decodeSEN( &pucTmp, 
				&pucTmp[ FLM_MAX_NUM_BUF_SIZE], &uiValue)))
			{
				goto Exit;
			}

			if (!ViewAddMenuItem( LBL_DOM_NAME_ID, uiLabelWidth,
					VAL_IS_NUMBER | DISP_HEX_DECIMAL,
					uiValue, 0,
					FSGetFileNumber( pStateInfo->ui32BlkAddress),
					FSGetFileOffset( pStateInfo->ui32BlkAddress) + uiDataOffset,
					0, MOD_SEN5 | MOD_HEX,
					uiCol, uiRow++, 0,
					uiBackColor, uiForeColor,
					uiBackColor, uiForeColor))
			{
				goto Exit;
			}
		}
		else
		{
			// If we can't display the number, at least display what we have left.
			if (!OutputHexValue( uiCol, &uiRow,
						uiBackColor, uiForeColor, LBL_DOM_NAME_ID,
						FSGetFileNumber( pStateInfo->ui32BlkAddress),
						FSGetFileOffset( pStateInfo->ui32BlkAddress) + uiDataOffset,
						pucData, uiNumBytes, FALSE))
			{
				goto Exit;
			}
			uiSENLength = uiNumBytes;
		}

		pucData += uiSENLength;
		uiNumBytes -= uiSENLength;
		uiDataOffset += uiSENLength;
	}

	// Prefix ID
	if (ucFlags & NSF_EXT_HAVE_PREFIX_BIT)
	{
		if (!uiNumBytes)
		{
			if (!ViewAddMenuItem( LBL_DOM_PREFIX_ID, uiLabelWidth,
					VAL_IS_LABEL_INDEX,
					LBL_NO_VALUE, 0,
					0, VIEW_INVALID_FILE_OFFSET,
					0, MOD_DISABLED,
					uiCol, uiRow++, 0, uiBackColor, uiForeColor,
					uiBackColor, uiForeColor))
			{
				goto Exit;
			}
		}
		else
		{
			pucTmp = pucData;
			uiSENLength = f_getSENLength( *pucTmp);
			if (uiNumBytes >= uiSENLength)
			{
				if( RC_BAD( rc = f_decodeSEN( &pucTmp, 
					&pucTmp[ FLM_MAX_NUM_BUF_SIZE], &uiValue)))
				{
					goto Exit;
				}
	
				if (!ViewAddMenuItem( LBL_DOM_PREFIX_ID, uiLabelWidth,
						VAL_IS_NUMBER | DISP_HEX_DECIMAL,
						uiValue, 0,
						FSGetFileNumber( pStateInfo->ui32BlkAddress),
						FSGetFileOffset( pStateInfo->ui32BlkAddress) + uiDataOffset,
						0, MOD_SEN5 | MOD_HEX,
						uiCol, uiRow++, 0,
						uiBackColor, uiForeColor,
						uiBackColor, uiForeColor))
				{
					goto Exit;
				}
			}
			else
			{
				// If we can't display the number, at least display what we have left.
				if (!OutputHexValue( uiCol, &uiRow,
							uiBackColor, uiForeColor, LBL_DOM_PREFIX_ID,
							FSGetFileNumber( pStateInfo->ui32BlkAddress),
							FSGetFileOffset( pStateInfo->ui32BlkAddress) +
								uiDataOffset,
							pucData, uiNumBytes, FALSE))
				{
					goto Exit;
				}
				uiSENLength = uiNumBytes;
			}
	
			pucData += uiSENLength;
			uiNumBytes -= uiSENLength;
			uiDataOffset += uiSENLength;
		}
	}

	if (!uiNumBytes)
	{
		if (!ViewAddMenuItem( LBL_DOM_BASE_ID, uiLabelWidth,
				VAL_IS_LABEL_INDEX,
				LBL_NO_VALUE, 0,
				0, VIEW_INVALID_FILE_OFFSET,
				0, MOD_DISABLED,
				uiCol, uiRow++, 0, uiBackColor, uiForeColor,
				uiBackColor, uiForeColor))
		{
			goto Exit;
		}
	}
	else
	{
		// Base ID (required)
		pucTmp = pucData;
		uiSENLength = f_getSENLength( *pucTmp);
		if (uiNumBytes >= uiSENLength)
		{
			if( RC_BAD( rc = f_decodeSEN64( &pucTmp, 
				&pucTmp[ FLM_MAX_NUM_BUF_SIZE], &ui64Value)))
			{
				goto Exit;
			}
	
			if (!ViewAddMenuItem( LBL_DOM_BASE_ID, uiLabelWidth,
					VAL_IS_NUMBER | DISP_HEX_DECIMAL,
					ui64Value, 0,
					FSGetFileNumber( pStateInfo->ui32BlkAddress),
					FSGetFileOffset( pStateInfo->ui32BlkAddress) + uiDataOffset,
					0, MOD_SEN9 | MOD_HEX,
					uiCol, uiRow++, 0,
					uiBackColor, uiForeColor,
					uiBackColor, uiForeColor))
			{
				goto Exit;
			}
		}
		else
		{
			// If we can't display the number, at least display what we have left.
			if (!OutputHexValue( uiCol, &uiRow,
						uiBackColor, uiForeColor, LBL_DOM_BASE_ID,
						FSGetFileNumber( pStateInfo->ui32BlkAddress),
						FSGetFileOffset( pStateInfo->ui32BlkAddress) + uiDataOffset,
						pucData, uiNumBytes, FALSE))
			{
				goto Exit;
			}
			uiSENLength = uiNumBytes;			
		}

		pucData += uiSENLength;
		uiDataOffset += uiSENLength;
		uiNumBytes -= uiSENLength;
	}

	if (!uiNumBytes)
	{
		if (!ViewAddMenuItem( LBL_DOM_ROOT_ID, uiLabelWidth,
				VAL_IS_LABEL_INDEX,
				LBL_NO_VALUE, 0,
				0, VIEW_INVALID_FILE_OFFSET,
				0, MOD_DISABLED,
				uiCol, uiRow++, 0, uiBackColor, uiForeColor,
				uiBackColor, uiForeColor))
		{
			goto Exit;
		}
	}
	else
	{
		pucTmp = pucData;
		uiSENLength = f_getSENLength( *pucTmp);
		if (uiNumBytes >= uiSENLength)
		{
			if( RC_BAD( rc = f_decodeSEN64( &pucTmp, 
				&pucTmp[ FLM_MAX_NUM_BUF_SIZE], &ui64Value)))
			{
				goto Exit;
			}

			if (!ViewAddMenuItem( LBL_DOM_ROOT_ID, uiLabelWidth,
					VAL_IS_NUMBER | DISP_HEX_DECIMAL,
					ui64Value, 0,
					FSGetFileNumber( pStateInfo->ui32BlkAddress),
					FSGetFileOffset( pStateInfo->ui32BlkAddress) + uiDataOffset,
					0, MOD_SEN9 | MOD_HEX,
					uiCol, uiRow++, 0,
					uiBackColor, uiForeColor,
					uiBackColor, uiForeColor))
			{
				goto Exit;
			}
		}
		else
		{
			// If we can't display the number, at least display what we have left.
			if (!OutputHexValue( uiCol, &uiRow,
						uiBackColor, uiForeColor, LBL_DOM_ROOT_ID,
						FSGetFileNumber( pStateInfo->ui32BlkAddress),
						FSGetFileOffset( pStateInfo->ui32BlkAddress) + uiDataOffset,
						pucData, uiNumBytes, FALSE))
			{
				goto Exit;
			}			
			uiSENLength = uiNumBytes;
		}
		pucData += uiSENLength;
		uiDataOffset += uiSENLength;
		uiNumBytes -= uiSENLength;
	}

	if (!uiNumBytes)
	{
		if (!ViewAddMenuItem( LBL_DOM_PARENT_ID, uiLabelWidth,
				VAL_IS_LABEL_INDEX,
				LBL_NO_VALUE, 0,
				0, VIEW_INVALID_FILE_OFFSET,
				0, MOD_DISABLED,
				uiCol, uiRow++, 0, uiBackColor, uiForeColor,
				uiBackColor, uiForeColor))
		{
			goto Exit;
		}
	}
	else
	{
		pucTmp = pucData;
		uiSENLength = f_getSENLength( *pucTmp);
		if (uiNumBytes >= uiSENLength)
		{
			if( RC_BAD( rc = f_decodeSEN64( &pucTmp, 
				&pucTmp[ FLM_MAX_NUM_BUF_SIZE], &ui64Value)))
			{
				goto Exit;
			}

			if (!ViewAddMenuItem( LBL_DOM_PARENT_ID, uiLabelWidth,
					VAL_IS_NUMBER | DISP_HEX_DECIMAL,
					ui64Value, 0,
					FSGetFileNumber( pStateInfo->ui32BlkAddress),
					FSGetFileOffset( pStateInfo->ui32BlkAddress) + uiDataOffset,
					0, MOD_SEN9 | MOD_HEX,
					uiCol, uiRow++, 0,
					uiBackColor, uiForeColor,
					uiBackColor, uiForeColor))
			{
				goto Exit;
			}
		}
		else
		{
			// If we can't display the number, at least display what we have left.
			if (!OutputHexValue( uiCol, &uiRow,
						uiBackColor, uiForeColor, LBL_DOM_PARENT_ID,
						FSGetFileNumber( pStateInfo->ui32BlkAddress),
						FSGetFileOffset( pStateInfo->ui32BlkAddress) + uiDataOffset,
						pucData, uiNumBytes, FALSE))
			{
				goto Exit;
			}
			uiSENLength = uiNumBytes;
		}
		pucData += uiSENLength;
		uiDataOffset += uiSENLength;
		uiNumBytes -= uiSENLength;
	}

	// Prev+Next Siblings
	if (ucFlags & NSF_HAVE_SIBLINGS_BIT)
	{
		if (!uiNumBytes)
		{
			if (!ViewAddMenuItem( LBL_DOM_PREV_SIB_ID, uiLabelWidth,
					VAL_IS_LABEL_INDEX,
					LBL_NO_VALUE, 0,
					0, VIEW_INVALID_FILE_OFFSET,
					0, MOD_DISABLED,
					uiCol, uiRow++, 0, uiBackColor, uiForeColor,
					uiBackColor, uiForeColor))
			{
				goto Exit;
			}
		}
		else
		{
			// Previous
			pucTmp = pucData;
			uiSENLength = f_getSENLength( *pucTmp);
			if (uiNumBytes >= uiSENLength)
			{
				if( RC_BAD( rc = f_decodeSEN64( &pucTmp, 
					&pucTmp[ FLM_MAX_NUM_BUF_SIZE], &ui64Value)))
				{
					goto Exit;
				}
	
				if (!ViewAddMenuItem( LBL_DOM_PREV_SIB_ID, uiLabelWidth,
						VAL_IS_NUMBER | DISP_HEX_DECIMAL,
						ui64Value, 0,
						FSGetFileNumber( pStateInfo->ui32BlkAddress),
						FSGetFileOffset( pStateInfo->ui32BlkAddress) + uiDataOffset,
						0, MOD_SEN9 | MOD_HEX,
						uiCol, uiRow++, 0,
						uiBackColor, uiForeColor,
						uiBackColor, uiForeColor))
				{
					goto Exit;
				}
			}
			else
			{
				// If we can't display the number, at least display what we have left.
				if (!OutputHexValue( uiCol, &uiRow,
							uiBackColor, uiForeColor, LBL_DOM_PREV_SIB_ID,
							FSGetFileNumber( pStateInfo->ui32BlkAddress),
							FSGetFileOffset( pStateInfo->ui32BlkAddress) + uiDataOffset,
							pucData, uiNumBytes, FALSE))
				{
					goto Exit;
				}
				uiSENLength = uiNumBytes;
			}
	
			pucData += uiSENLength;
			uiDataOffset += uiSENLength;
			uiNumBytes -= uiSENLength;
		}

		if (!uiNumBytes)
		{
			if (!ViewAddMenuItem( LBL_DOM_NEXT_SIB_ID, uiLabelWidth,
					VAL_IS_LABEL_INDEX,
					LBL_NO_VALUE, 0,
					0, VIEW_INVALID_FILE_OFFSET,
					0, MOD_DISABLED,
					uiCol, uiRow++, 0, uiBackColor, uiForeColor,
					uiBackColor, uiForeColor))
			{
				goto Exit;
			}
		}
		else
		{
			// Next
			pucTmp = pucData;
			uiSENLength = f_getSENLength( *pucTmp);
			if (uiNumBytes >= uiSENLength)
			{
				if( RC_BAD( rc = f_decodeSEN64( &pucTmp, 
					&pucTmp[ FLM_MAX_NUM_BUF_SIZE], &ui64Value)))
				{
					goto Exit;
				}
	
				if (!ViewAddMenuItem( LBL_DOM_NEXT_SIB_ID, uiLabelWidth,
						VAL_IS_NUMBER | DISP_HEX_DECIMAL,
						ui64Value, 0,
						FSGetFileNumber( pStateInfo->ui32BlkAddress),
						FSGetFileOffset( pStateInfo->ui32BlkAddress) + uiDataOffset,
						0, MOD_SEN9 | MOD_HEX,
						uiCol, uiRow++, 0,
						uiBackColor, uiForeColor,
						uiBackColor, uiForeColor))
				{
					goto Exit;
				}
			}
			else
			{
				// If we can't display the number, at least display what we have left.
				if (!OutputHexValue( uiCol, &uiRow,
							uiBackColor, uiForeColor, LBL_DOM_NEXT_SIB_ID,
							FSGetFileNumber( pStateInfo->ui32BlkAddress),
							FSGetFileOffset( pStateInfo->ui32BlkAddress) + uiDataOffset,
							pucData, uiNumBytes, FALSE))
				{
					goto Exit;
				}
				uiSENLength = uiNumBytes;
			}
			pucData += uiSENLength;
			uiDataOffset += uiSENLength;
			uiNumBytes -= uiSENLength;
		}
	}

	// First+Last Child ID
	if (ucFlags & NSF_HAVE_CHILDREN_BIT)
	{
		// First
		if (!uiNumBytes)
		{
			if (!ViewAddMenuItem( LBL_DOM_FIRST_CHILD_ID, uiLabelWidth,
					VAL_IS_LABEL_INDEX,
					LBL_NO_VALUE, 0,
					0, VIEW_INVALID_FILE_OFFSET,
					0, MOD_DISABLED,
					uiCol, uiRow++, 0, uiBackColor, uiForeColor,
					uiBackColor, uiForeColor))
			{
				goto Exit;
			}
		}
		else
		{
			pucTmp = pucData;
			uiSENLength = f_getSENLength( *pucTmp);
			if (uiNumBytes >= uiSENLength)
			{
				if( RC_BAD( rc = f_decodeSEN64( &pucTmp, 
					&pucTmp[ FLM_MAX_NUM_BUF_SIZE], &ui64Value)))
				{
					goto Exit;
				}
	
				if (!ViewAddMenuItem( LBL_DOM_FIRST_CHILD_ID, uiLabelWidth,
						VAL_IS_NUMBER | DISP_HEX_DECIMAL,
						ui64Value, 0,
						FSGetFileNumber( pStateInfo->ui32BlkAddress),
						FSGetFileOffset( pStateInfo->ui32BlkAddress) + uiDataOffset,
						0, MOD_SEN9 | MOD_HEX,
						uiCol, uiRow++, 0,
						uiBackColor, uiForeColor,
						uiBackColor, uiForeColor))
				{
					goto Exit;
				}
			}
			else
			{
				// If we can't display the number, at least display what we have left.
				if (!OutputHexValue( uiCol, &uiRow,
							uiBackColor, uiForeColor, LBL_DOM_FIRST_CHILD_ID,
							FSGetFileNumber( pStateInfo->ui32BlkAddress),
							FSGetFileOffset( pStateInfo->ui32BlkAddress) + uiDataOffset,
							pucData, uiNumBytes, FALSE))
				{
					goto Exit;
				}
				uiSENLength = uiNumBytes;
			}
	
			pucData += uiSENLength;
			uiDataOffset += uiSENLength;
			uiNumBytes -= uiSENLength;
		}

		// Last
		if (!uiNumBytes)
		{
			if (!ViewAddMenuItem( LBL_DOM_LAST_CHILD_ID, uiLabelWidth,
					VAL_IS_LABEL_INDEX,
					LBL_NO_VALUE, 0,
					0, VIEW_INVALID_FILE_OFFSET,
					0, MOD_DISABLED,
					uiCol, uiRow++, 0, uiBackColor, uiForeColor,
					uiBackColor, uiForeColor))
			{
				goto Exit;
			}
		}
		else
		{
			pucTmp = pucData;
			uiSENLength = f_getSENLength( *pucTmp);
			if (uiNumBytes >= uiSENLength)
			{
				if( RC_BAD( rc = f_decodeSEN64( &pucTmp, 
					&pucTmp[ FLM_MAX_NUM_BUF_SIZE], &ui64Value)))
				{
					goto Exit;
				}
	
				if (!ViewAddMenuItem( LBL_DOM_LAST_CHILD_ID, uiLabelWidth,
						VAL_IS_NUMBER | DISP_HEX_DECIMAL,
						ui64Value, 0,
						FSGetFileNumber( pStateInfo->ui32BlkAddress),
						FSGetFileOffset( pStateInfo->ui32BlkAddress) + uiDataOffset,
						0, MOD_SEN9 | MOD_HEX,
						uiCol, uiRow++, 0,
						uiBackColor, uiForeColor,
						uiBackColor, uiForeColor))
				{
					goto Exit;
				}
			}
			else
			{
				// If we can't display the number, at least display what we have left.
				if (!OutputHexValue( uiCol, &uiRow,
							uiBackColor, uiForeColor, LBL_DOM_LAST_CHILD_ID,
							FSGetFileNumber( pStateInfo->ui32BlkAddress),
							FSGetFileOffset( pStateInfo->ui32BlkAddress) + uiDataOffset,
							pucData, uiNumBytes, FALSE))
				{
					goto Exit;
				}
				uiSENLength = uiNumBytes;
			}
			pucData += uiSENLength;
			uiDataOffset += uiSENLength;
			uiNumBytes -= uiSENLength;
		}
	}

	// Annotation
	if (ucExtFlags & NSF_EXT_ANNOTATION_BIT)
	{
		if (!uiNumBytes)
		{
			if (!ViewAddMenuItem( LBL_DOM_ANNOTATION_ID, uiLabelWidth,
					VAL_IS_LABEL_INDEX,
					LBL_NO_VALUE, 0,
					0, VIEW_INVALID_FILE_OFFSET,
					0, MOD_DISABLED,
					uiCol, uiRow++, 0, uiBackColor, uiForeColor,
					uiBackColor, uiForeColor))
			{
				goto Exit;
			}
		}
		else
		{
			pucTmp = pucData;
			uiSENLength = f_getSENLength( *pucTmp);
			if (uiNumBytes >= uiSENLength)
			{
				if( RC_BAD( rc = f_decodeSEN64( &pucTmp, 
					&pucTmp[ FLM_MAX_NUM_BUF_SIZE], &ui64Value)))
				{
					goto Exit;
				}
	
				if (!ViewAddMenuItem( LBL_DOM_ANNOTATION_ID, uiLabelWidth,
						VAL_IS_NUMBER | DISP_HEX_DECIMAL,
						ui64Value, 0,
						FSGetFileNumber( pStateInfo->ui32BlkAddress),
						FSGetFileOffset( pStateInfo->ui32BlkAddress) + uiDataOffset,
						0, MOD_SEN9 | MOD_HEX,
						uiCol, uiRow++, 0,
						uiBackColor, uiForeColor,
						uiBackColor, uiForeColor))
				{
					goto Exit;
				}
			}
			else
			{
				// If we can't display the number, at least display what we have left.
				if (!OutputHexValue( uiCol, &uiRow,
							uiBackColor, uiForeColor, LBL_DOM_ANNOTATION_ID,
							FSGetFileNumber( pStateInfo->ui32BlkAddress),
							FSGetFileOffset( pStateInfo->ui32BlkAddress) + uiDataOffset,
							pucData, uiNumBytes, FALSE))
				{
					goto Exit;
				}
				uiSENLength = uiNumBytes;
			}
			pucData += uiSENLength;
			uiDataOffset += uiSENLength;
			uiNumBytes -= uiSENLength;
		}
	}

	if (uiNumBytes)
	{
		// Output whatever is left as a HEX dump.
		if (!OutputHexValue( uiCol, &uiRow,
					uiBackColor, uiForeColor, LBL_DOM_VALUE,
					FSGetFileNumber( pStateInfo->ui32BlkAddress),
					FSGetFileOffset( pStateInfo->ui32BlkAddress) +
						uiDataOffset,
					pucData, uiNumBytes, FALSE))
		{
			goto Exit;
		}
	}

	bOk = TRUE;

Exit:

	*puiRow = uiRow;

	return( bOk);
}


/***************************************************************************
Desc:	This routine outputs the elements in a LEAF block.
*****************************************************************************/
FSTATIC FLMBOOL OutputLeafElements(
	FLMUINT				uiCol,
	FLMUINT *			puiRow,
	F_BTREE_BLK_HDR *	pBlkHdr,
	BLK_EXP_p			pBlkExp,
	FLMBYTE *			pucBlkStatus,
	FLMBOOL				bStatusOnlyFlag
	)
{
// VISIT:	There are a number of places throughout that have been commentd out
//				that have to do with the block status.  Currently, the pucBlkStatus
//				flag is not being updated in a meaningful way.  Need to investigate
//				what this is used for and find a way to make it meaningful.
	FLMBOOL		bOk = FALSE;
	FLMUINT		uiLabelWidth = 30;
	eColorType	uiBackColor = FLM_BLACK;
	eColorType	uiForeColor = FLM_LIGHTGRAY;
	eColorType	uiUnselectBackColor = FLM_BLACK;
	eColorType	uiUnselectForeColor = FLM_WHITE;
	eColorType	uiSelectBackColor = FLM_BLUE;
	eColorType	uiSelectForeColor = FLM_WHITE;
	FLMUINT		uiRow = *puiRow;
	FLMUINT		uiElementCount = 0;
	FLMINT		iErrorCode;
//	LFileType	eLfType = LF_INVALID;
	FLMBYTE		ucElmStatus [NUM_STATUS_BYTES];
	STATE_INFO	StateInfo;
//	LF_HDR_p		pLogicalFile = NULL;
	FLMUINT		uiCurOffset;
	FLMUINT16 *	pui16OffsetArray;
	FLMUINT		uiOption;
	FLMBOOL		bHasData = (pBlkHdr->stdBlkHdr.ui8BlkType == BT_LEAF_DATA);

	if (pBlkExp->uiLfNum == 0)
	{
		pBlkExp->uiLfNum = (FLMUINT)pBlkHdr->ui16LogicalFile;
	}

	// Setup the STATE variable for processing through the block

	(void)ViewGetDictInfo();

#if 0
	if ((iErrorCode = flmVerifyBlockHeader( &StateInfo,
														 &BlockInfo,
														 (FLMUINT)gv_ViewDbHdr.ui16BlockSize,
														 pBlkExp->uiNextAddr,
														 pBlkExp->uiPrevAddr,
														 (FLMBOOL)(StateInfo.pDb != NULL
															 ? TRUE
															 : FALSE))) != 0)
	{
		SetStatusBit( pucBlkStatus, iErrorCode);
	}
#endif
	uiCurOffset = 0;
	pui16OffsetArray = (FLMUINT16 *)((FLMBYTE *)pBlkHdr + sizeofBTreeBlkHdr( pBlkHdr));

	// Read through the elements in the block

	while( uiCurOffset <= (FLMUINT)(pBlkHdr->ui16NumKeys - 1))
	{
		InitStatusBits( ucElmStatus);
		uiElementCount++;
		GetElmInfo( (FLMBYTE *)pBlkHdr + pui16OffsetArray[ uiCurOffset], pBlkHdr, &StateInfo);
		iErrorCode = 0;  // ??

#if 0
		if ((iErrorCode = flmVerifyElement( &StateInfo,
										FO_DO_EXTENDED_DATA_CHECK)) != 0)
		{
			SetStatusBit( ucElmStatus, iErrorCode);
		}
		else if (eLfType == XFLM_LF_INDEX)
		{
			if (StateInfo.uiCurKeyLen)
			{
				if (RC_BAD( flmVerifyIXRefs( &StateInfo, NULL, 0,
									&iErrorCode)) || iErrorCode != 0)
				{
					SetStatusBit( ucElmStatus, iErrorCode);
				}
			}
		}
#endif

		// Output the element

		if (!bStatusOnlyFlag)
		{
			uiRow++;

			// Output the element number

			if (!ViewAddMenuItem( LBL_ELEMENT_NUMBER, uiLabelWidth,
					VAL_IS_NUMBER | DISP_DECIMAL,
					(FLMUINT64)uiElementCount, 0,
					0,
					VIEW_INVALID_FILE_OFFSET,
					0, MOD_DISABLED,
					uiCol, uiRow++, 0,
					FLM_GREEN, FLM_WHITE,
					FLM_GREEN, FLM_WHITE))
			{
				goto Exit;
			}

			// Remember this item if we are searching
#if 0
			if ((gv_bViewSearching) &&
					(flmCompareKeys( StateInfo.pucCurKey, StateInfo.uiCurKeyLen,
													gv_ucViewSearchKey,
													gv_uiViewSearchKeyLen) >= 0) &&
					(gv_pViewSearchItem == NULL))
			{
				gv_pViewSearchItem = gv_pViewMenuLastItem;
			}
#endif
			// Output the element offset within the block

			if (!ViewAddMenuItem( LBL_ELEMENT_OFFSET, uiLabelWidth,
						VAL_IS_NUMBER | DISP_DECIMAL,
						(FLMUINT64)StateInfo.uiElmOffset, 0,
						FSGetFileNumber( StateInfo.ui32BlkAddress),
						FSGetFileOffset( StateInfo.ui32BlkAddress) + StateInfo.uiElmOffset,
						0, MOD_DISABLED,
						uiCol, uiRow++, 0, uiBackColor, uiForeColor,
						uiBackColor, uiForeColor))
			{
				goto Exit;
			}

			// Output the element length

			if (!ViewAddMenuItem( LBL_ELEMENT_LENGTH, uiLabelWidth,
						VAL_IS_NUMBER | DISP_DECIMAL,
						(FLMUINT64)StateInfo.uiElmLen, 0,
						FSGetFileNumber( StateInfo.ui32BlkAddress),
						FSGetFileOffset( StateInfo.ui32BlkAddress) + StateInfo.uiElmOffset,
						0, MOD_DISABLED,
						uiCol, uiRow++, 0, uiBackColor, uiForeColor,
						uiBackColor, uiForeColor))
			{
				goto Exit;
			}

			// Display the first element flag if present
			if (bHasData)
			{

				if (!ViewAddMenuItem( LBL_FIRST_ELEMENT_FLAG, uiLabelWidth,
						VAL_IS_LABEL_INDEX,
						(bteFirstElementFlag( StateInfo.pucElm)
						? (FLMUINT64)LBL_YES
						: (FLMUINT64)LBL_NO), 0,
						FSGetFileNumber( StateInfo.ui32BlkAddress),
						FSGetFileOffset( StateInfo.ui32BlkAddress) + StateInfo.uiElmOffset,
						0x80, MOD_BITS | MOD_DECIMAL,
						uiCol, uiRow++, 0, uiBackColor, uiForeColor,
						uiBackColor, uiForeColor))
				{
					goto Exit;
				}

				// Display the last element flag

				if (!ViewAddMenuItem( LBL_LAST_ELEMENT_FLAG, uiLabelWidth,
							VAL_IS_LABEL_INDEX,
							(bteLastElementFlag( StateInfo.pucElm)
							? (FLMUINT64)LBL_YES
							: (FLMUINT64)LBL_NO), 0,
							FSGetFileNumber( StateInfo.ui32BlkAddress),
							FSGetFileOffset( StateInfo.ui32BlkAddress) + StateInfo.uiElmOffset,
							0x40, MOD_BITS | MOD_DECIMAL,
							uiCol, uiRow++, 0, uiBackColor, uiForeColor,
							uiBackColor, uiForeColor))
				{
					goto Exit;
				}

				// Display the Data Only flag

				if (!ViewAddMenuItem( LBL_DATA_BLOCK_FLAG, uiLabelWidth,
							VAL_IS_LABEL_INDEX,
							(bteDataBlockFlag( StateInfo.pucElm)
							? (FLMUINT64)LBL_YES
							: (FLMUINT64)LBL_NO), 0,
							FSGetFileNumber( StateInfo.ui32BlkAddress),
							FSGetFileOffset( StateInfo.ui32BlkAddress) + StateInfo.uiElmOffset,
							0x40, MOD_BITS | MOD_DECIMAL,
							uiCol, uiRow++, 0, uiBackColor, uiForeColor,
							uiBackColor, uiForeColor))
				{
					goto Exit;
				}
			}

			// Display the key length

			if (!ViewAddMenuItem( LBL_KEY_LENGTH, uiLabelWidth,
						VAL_IS_NUMBER | DISP_DECIMAL,
						(FLMUINT64)StateInfo.uiElmKeyLen, 0,
						FSGetFileNumber( StateInfo.ui32BlkAddress),
						FSGetFileOffset( StateInfo.ui32BlkAddress) +
													StateInfo.uiElmKeyLenOffset,
						0, MOD_DECIMAL,
						uiCol, uiRow++, 0, uiBackColor, uiForeColor,
						uiBackColor, uiForeColor))
			{
				goto Exit;
			}

			// Output the current key portion, if any

			if (!OutputHexValue( uiCol, &uiRow,
						uiBackColor, uiForeColor, LBL_ELEMENT_KEY,
						FSGetFileNumber( StateInfo.ui32BlkAddress),
						FSGetFileOffset( StateInfo.ui32BlkAddress) +
														StateInfo.uiElmKeyOffset,
						StateInfo.pucElmKey, StateInfo.uiElmKeyLen,
						FALSE))
			{
				goto Exit;
			}



			// Display the data length

			if (bHasData)
			{
				// Output the overall data field (if present)
				if (bteOADataLenFlag( StateInfo.pucElm))
				{
					if (!ViewAddMenuItem( LBL_OA_DATA_LENGTH, uiLabelWidth,
							VAL_IS_NUMBER | DISP_DECIMAL,
							(FLMUINT64)StateInfo.uiElmOADataLen, 0,
							FSGetFileNumber( StateInfo.ui32BlkAddress),
							FSGetFileOffset( StateInfo.ui32BlkAddress) + 
														StateInfo.uiElmOADataLenOffset,
							0, MOD_FLMBYTE | MOD_DECIMAL,
							uiCol, uiRow++, 0, uiBackColor, uiForeColor,
							uiBackColor, uiForeColor))
					{
						goto Exit;
					}
				}

				if (!ViewAddMenuItem( LBL_DATA_LENGTH, uiLabelWidth,
						VAL_IS_NUMBER | DISP_DECIMAL,
						(FLMUINT64)StateInfo.uiElmDataLen, 0,
						FSGetFileNumber( StateInfo.ui32BlkAddress),
						FSGetFileOffset( StateInfo.ui32BlkAddress) + 
													StateInfo.uiElmDataLenOffset,
						0, MOD_FLMBYTE | MOD_DECIMAL,
						uiCol, uiRow++, 0, uiBackColor, uiForeColor,
						uiBackColor, uiForeColor))
				{
					goto Exit;
				}


				// Output the data portion
				// Check for a data only block.
				if ( bteDataBlockFlag( StateInfo.pucElm))
				{
					FLMUINT		uiTempAddress = FB2UD( StateInfo.pucElmData);
					if (uiTempAddress == 0)
					{
						uiOption = 0;
					}
					else
					{
						uiOption = BLK_OPTION_DATA_BLOCK | StateInfo.uiElmOffset;
					}
					if (!ViewAddMenuItem( LBL_DATA_BLOCK_ADDRESS, uiLabelWidth,
								VAL_IS_NUMBER | DISP_HEX_DECIMAL,
								(FLMUINT64)uiTempAddress, 0,
								FSGetFileNumber( StateInfo.ui32BlkAddress),
								FSGetFileOffset( StateInfo.ui32BlkAddress) +
									StateInfo.uiElmDataOffset, 
								0, MOD_DISABLED, uiCol, uiRow++, uiOption,
								(!uiOption ? uiBackColor : uiUnselectBackColor),
								(!uiOption ? uiForeColor : uiUnselectForeColor),
								(!uiOption ? uiBackColor : uiSelectBackColor),
								(!uiOption ? uiForeColor : uiSelectForeColor)))
					{
						goto Exit;
					}

				}
				else
				{
					if ( bteFirstElementFlag( StateInfo.pucElm) &&
						  !isIndexBlk((F_BTREE_BLK_HDR *)StateInfo.pBlkHdr))
					{
						if (!OutputDomNode( uiCol, &uiRow,
								uiBackColor, uiForeColor, 
								StateInfo.pucElmData + 1, &StateInfo))
						{
							goto Exit;
						}

					}
					else
					{

						if (!OutputHexValue( uiCol, &uiRow,
								uiBackColor, uiForeColor, LBL_DATA,
								FSGetFileNumber( StateInfo.ui32BlkAddress),
								FSGetFileOffset( StateInfo.ui32BlkAddress) +
									StateInfo.uiElmDataOffset,
								StateInfo.pucElmData, StateInfo.uiElmDataLen,
								FALSE))
						{
							goto Exit;
						}
					}
				}
			}
		}

		// Go to the next element
		uiCurOffset++;
		OrStatusBits( pucBlkStatus, ucElmStatus);
	}

	if (!bStatusOnlyFlag)
	{
		*puiRow = uiRow;

		// If we were searching and did not find a key, set it on the
		// last key found

		if (gv_bViewSearching && !gv_pViewSearchItem)
		{
			gv_pViewSearchItem = gv_pViewMenuLastItem;
			while (gv_pViewSearchItem &&
					 gv_pViewSearchItem->iLabelIndex != LBL_ELEMENT_NUMBER)
			{
				gv_pViewSearchItem = gv_pViewSearchItem->pPrevItem;
			}
		}
	}
	bOk = TRUE;

Exit:

	*puiRow = uiRow;

#if 0
	if (bStateInitialized && StateInfo.pRecord)
	{
		StateInfo.pRecord->release( &StateInfo.pRecord);
	}
#endif
	return( bOk);
}

/***************************************************************************
Desc:	This routine outputs the elements in a DATA-ONLY block.
*****************************************************************************/
FSTATIC FLMBOOL OutputDataOnlyElements(
	FLMUINT				uiCol,
	FLMUINT *			puiRow,
	F_BLK_HDR *			pBlkHdr,
	BLK_EXP_p,			//pBlkExp,
	FLMBYTE *,			//pucBlkStatus,
	FLMBOOL				bStatusOnlyFlag
	)
{
	FLMBOOL		bOk = FALSE;
	FLMUINT		uiLabelWidth = 30;
	eColorType	uiBackColor = FLM_BLACK;
	eColorType	uiForeColor = FLM_LIGHTGRAY;
	FLMUINT		uiRow = *puiRow;
	FLMINT		iErrorCode;
//	LFileType	eLfType = LF_INVALID;
	FLMBYTE		ucElmStatus [NUM_STATUS_BYTES];
	STATE_INFO	StateInfo;

	// Setup the STATE variable for processing through the block

	(void)ViewGetDictInfo();

#if 0
	if ((iErrorCode = flmVerifyBlockHeader( &StateInfo, &BlockInfo,
						(FLMUINT)gv_ViewDbHdr.ui16BlockSize,
												pBlkExp->uiNextAddr,
												pBlkExp->uiPrevAddr,
												(FLMBOOL)(StateInfo.pDb != NULL
															 ? TRUE
															 : FALSE))) != 0)
	{
		SetStatusBit( pucBlkStatus, iErrorCode);
	}
#endif

	// Read through the block

	InitStatusBits( ucElmStatus);
	
	GetElmInfo( (FLMBYTE *)pBlkHdr + SIZEOF_STD_BLK_HDR,
					(F_BTREE_BLK_HDR *)pBlkHdr, &StateInfo);
	
	iErrorCode = 0;  // ??

#if 0
	if ((iErrorCode = flmVerifyElement( &StateInfo,
									FO_DO_EXTENDED_DATA_CHECK)) != 0)
	{
		SetStatusBit( ucElmStatus, iErrorCode);
	}
	else if (eLfType == XFLM_LF_INDEX)
	{
		if (StateInfo.uiCurKeyLen)
		{
			if (RC_BAD( flmVerifyIXRefs( &StateInfo, NULL, 0,
								&iErrorCode)) || iErrorCode != 0)
			{
				SetStatusBit( ucElmStatus, iErrorCode);
			}
		}
	}
#endif

	// Output the element

	if (!bStatusOnlyFlag)
	{
		uiRow++;


		// Remember this item if we are searching
#if 0
		if ((gv_bViewSearching) &&
				(flmCompareKeys( StateInfo.pucCurKey, StateInfo.uiCurKeyLen,
												gv_ucViewSearchKey,
												gv_uiViewSearchKeyLen) >= 0) &&
				(gv_pViewSearchItem == NULL))
		{
			gv_pViewSearchItem = gv_pViewMenuLastItem;
		}
#endif

		// Output the element length

		if (!ViewAddMenuItem( LBL_ELEMENT_LENGTH, uiLabelWidth,
								VAL_IS_NUMBER | DISP_DECIMAL,
								(FLMUINT64)StateInfo.uiElmLen, 0,
								FSGetFileNumber( StateInfo.ui32BlkAddress),
								FSGetFileOffset( StateInfo.ui32BlkAddress) + 
									StateInfo.uiElmOffset,
								0, MOD_DISABLED,
								uiCol, uiRow++, 0, uiBackColor, uiForeColor,
								uiBackColor, uiForeColor))
		{
			goto Exit;
		}


		// Display the key length if there is a key...
		if ( StateInfo.uiElmKeyLen)
		{
			if (!ViewAddMenuItem( LBL_KEY_LENGTH, uiLabelWidth,
						VAL_IS_NUMBER | DISP_DECIMAL,
						(FLMUINT64)StateInfo.uiElmKeyLen, 0,
						FSGetFileNumber( StateInfo.ui32BlkAddress),
						FSGetFileOffset( StateInfo.ui32BlkAddress) + 
							StateInfo.uiElmKeyOffset, 0,
						MOD_DECIMAL,
						uiCol, uiRow++, 0, uiBackColor, uiForeColor,
						uiBackColor, uiForeColor))
			{
				goto Exit;
			}

			// Output the key portion

			if (!OutputHexValue( uiCol, &uiRow,
						uiBackColor, uiForeColor, LBL_ELEMENT_KEY,
						FSGetFileNumber( StateInfo.ui32BlkAddress),
						FSGetFileOffset( StateInfo.ui32BlkAddress) +
							StateInfo.uiElmKeyOffset,
						StateInfo.pucElmKey, StateInfo.uiElmKeyLen,
						FALSE))
			{
				goto Exit;
			}
		}

		// Display the data length

		if (!ViewAddMenuItem( LBL_DATA_LENGTH, uiLabelWidth,
				VAL_IS_NUMBER | DISP_DECIMAL,
				(FLMUINT64)StateInfo.uiElmDataLen, 0,
				FSGetFileNumber( StateInfo.ui32BlkAddress),
				FSGetFileOffset( StateInfo.ui32BlkAddress) + 
					StateInfo.uiElmDataLenOffset,
				0,
				MOD_FLMBYTE | MOD_DECIMAL,
				uiCol, uiRow++, 0, uiBackColor, uiForeColor,
				uiBackColor, uiForeColor))
		{
			goto Exit;
		}

		// Output the data portion.
		if (!OutputHexValue( uiCol, &uiRow,
				uiBackColor, uiForeColor, LBL_DATA,
				FSGetFileNumber( StateInfo.ui32BlkAddress),
				FSGetFileOffset( StateInfo.ui32BlkAddress) +
					StateInfo.uiElmDataOffset,
				StateInfo.pucElmData, StateInfo.uiElmDataLen,
				FALSE))
		{
			goto Exit;
		}
	}


	if (!bStatusOnlyFlag)
	{
		*puiRow = uiRow;

		// If we were searching and did not find a key, set it on the
		// last key found

		if (gv_bViewSearching && !gv_pViewSearchItem)
		{
			gv_pViewSearchItem = gv_pViewMenuLastItem;
			while (gv_pViewSearchItem &&
					 gv_pViewSearchItem->iLabelIndex != LBL_ELEMENT_NUMBER)
			{
				gv_pViewSearchItem = gv_pViewSearchItem->pPrevItem;
			}
		}
	}
	bOk = TRUE;

Exit:

	return( bOk);
}


/***************************************************************************
Desc:	This routine outputs a LEAF block, including the block header.
*****************************************************************************/
FLMBOOL ViewLeafBlk(
	FLMUINT			uiReadAddress,
	FLMUINT			uiBlkAddress,
	F_BLK_HDR **	ppBlkHdr,
	BLK_EXP_p		pBlkExp
	)
{
	FLMBOOL			bOk = FALSE;
	FLMUINT			uiRow;
	FLMUINT			uiCol;
	F_BLK_HDR *		pBlkHdr;
	FLMBYTE			ucBlkStatus [NUM_STATUS_BYTES];
	FLMUINT32		ui32CalcCRC;
	FLMUINT32		ui32BlkCRC;
	FLMUINT			uiBytesRead;

	InitStatusBits( ucBlkStatus);

	// Read the block into memory

	if (!ViewBlkRead( uiReadAddress, ppBlkHdr, TRUE,
							(FLMUINT)gv_ViewDbHdr.ui16BlockSize,
							&ui32CalcCRC, &ui32BlkCRC,
							&uiBytesRead, TRUE))
	{
		goto Exit;
	}
	pBlkHdr = *ppBlkHdr;

	ViewMenuInit( "LEAF Block");

	// Output the block header first

	uiRow = 0;
	uiCol = 5;
	pBlkExp->uiType = pBlkHdr->ui8BlkType;
	pBlkExp->uiBlkAddr = uiBlkAddress;
	pBlkExp->uiLevel = 0;

	OutputLeafElements( uiCol, &uiRow, (F_BTREE_BLK_HDR *)pBlkHdr,
								pBlkExp, ucBlkStatus, TRUE);
	if (!ViewOutBlkHdr( uiCol, &uiRow, pBlkHdr, pBlkExp, ucBlkStatus,
								ui32CalcCRC, ui32BlkCRC))
	{
		goto Exit;
	}

	// Now output the leaf data

	if (!OutputLeafElements( uiCol, &uiRow, (F_BTREE_BLK_HDR *)pBlkHdr,
							pBlkExp, ucBlkStatus, FALSE))
	{
		goto Exit;
	}
	bOk = TRUE;

Exit:

	return( bOk);
}

/***************************************************************************
Desc:	This routine outputs a Data Only block, including the block header.
*****************************************************************************/
FLMBOOL ViewDataBlk(
	FLMUINT			uiReadAddress,
	FLMUINT			uiBlkAddress,
	F_BLK_HDR **	ppBlkHdr,
	BLK_EXP_p		pBlkExp
	)
{
	FLMBOOL			bOk = FALSE;
	FLMUINT			uiRow;
	FLMUINT			uiCol;
	F_BLK_HDR *		pBlkHdr;
	FLMBYTE			ucBlkStatus [NUM_STATUS_BYTES];
	FLMUINT32		ui32CalcCRC;
	FLMUINT32		ui32BlkCRC;
	FLMUINT			uiBytesRead;

	InitStatusBits( ucBlkStatus);

	// Read the block into memory

	if (!ViewBlkRead( uiReadAddress, ppBlkHdr, TRUE,
							(FLMUINT)gv_ViewDbHdr.ui16BlockSize,
							&ui32CalcCRC, &ui32BlkCRC,
							&uiBytesRead, TRUE))
	{
		goto Exit;
	}
	pBlkHdr = *ppBlkHdr;

	ViewMenuInit( "DATA Block");

	// Output the block header first

	uiRow = 0;
	uiCol = 5;
	pBlkExp->uiType = pBlkHdr->ui8BlkType;
	pBlkExp->uiBlkAddr = uiBlkAddress;
	pBlkExp->uiLevel = 0;

	OutputDataOnlyElements( uiCol, &uiRow, pBlkHdr,
								pBlkExp, ucBlkStatus, TRUE);
	if (!ViewOutBlkHdr( uiCol, &uiRow, pBlkHdr, pBlkExp, ucBlkStatus,
								ui32CalcCRC, ui32BlkCRC))
	{
		goto Exit;
	}

	// Now output the leaf data

	if (!OutputDataOnlyElements( uiCol, &uiRow, pBlkHdr,
							pBlkExp, ucBlkStatus, FALSE))
	{
		goto Exit;
	}
	bOk = TRUE;

Exit:

	return( bOk);
}


/***************************************************************************
Desc:	This routine outputs the elements of a NON-LEAF block.
*****************************************************************************/
FSTATIC FLMBOOL OutputNonLeafElements(
	FLMUINT					uiCol,
	FLMUINT *				puiRow,
	F_BTREE_BLK_HDR *		pBlkHdr,
	BLK_EXP_p				pBlkExp,
	FLMBYTE *				pucBlkStatus,
	FLMBOOL					bStatusOnlyFlag
	)
{
	FLMBOOL		bOk = FALSE;
	FLMUINT		uiLabelWidth = 30;
	eColorType	uiBackColor = FLM_BLACK;
	eColorType	uiForeColor = FLM_LIGHTGRAY;
	eColorType	uiUnselectBackColor = FLM_BLACK;
	eColorType	uiUnselectForeColor = FLM_WHITE;
	eColorType	uiSelectBackColor = FLM_BLUE;
	eColorType	uiSelectForeColor = FLM_WHITE;
	FLMUINT		uiRow = *puiRow;
	FLMUINT		uiElementCount = 0;
//	FLMBYTE *	pucTmpAddr;
	FLMUINT		uiTempAddress;
	FLMUINT		iErrorCode=0;
	FLMUINT		uiOption;
	eLFileType	eLfType;
	FLMUINT		uiBlkType = (FLMUINT)pBlkHdr->stdBlkHdr.ui8BlkType;
	FLMBYTE		ucElmStatus [NUM_STATUS_BYTES];
	STATE_INFO	StateInfo;
//	FLMBOOL		bStateInitialized = FALSE;
//	BLOCK_INFO	BlockInfo;
//	FLMBYTE		ucKeyBuffer [MAX_KEY_SIZ];
//	LF_HDR		LogicalFile;
	LF_HDR *		pLogicalFile = NULL;
//	LFILE_p		pLFile = NULL;
//	FLMUINT		uiFixedDrn;
	FLMUINT16 *	pui16OffsetArray;
	FLMUINT		uiCurOffset;

	if (pBlkExp->uiLfNum == 0)
	{
		pBlkExp->uiLfNum = (FLMUINT)pBlkHdr->ui16LogicalFile;
	}

	pui16OffsetArray = (FLMUINT16 *)((FLMBYTE *)pBlkHdr + sizeofBTreeBlkHdr( pBlkHdr));
	uiCurOffset = 0;

	// Setup the STATE variable for processing through the block

	(void)ViewGetDictInfo();

#if 0
	if (gv_bViewHaveDictInfo)
	{
		F_Dict *	pDict = gv_hViewDb->pDict;

		if (isContainerBlk( pBlkHdr))
		{
			if (RC_OK( pDict->getContainer( pBlkExp->uiLfNum, &pLFile)))
			{
				f_memset( &LogicalFile, 0, sizeof( LF_HDR));
				pLogicalFile = &LogicalFile;
				LogicalFile.pLFile = pLFile;
			}
		}
		else
		{
			IXD *	pIxd;

			if (RC_OK( pDict->getIndex( pBlkExp->uiLfNum, &pLFile, &pIxd)))
			{
				f_memset( &LogicalFile, 0, sizeof( LF_HDR));
				pLogicalFile = &LogicalFile;
				LogicalFile.pLFile = pLFile;
				LogicalFile.pIxd = pIxd;
				LogicalFile.pIfd = pIxd->pFirstIfd;
			}
		}
	}
#endif

	eLfType = (pLogicalFile) ? pLogicalFile->eLfType : XFLM_LF_INVALID;

#if 0
	flmInitReadState( &StateInfo, &bStateInitialized,
							(FLMUINT)gv_ViewDbHdr.ui32DbVersion,
							(gv_bViewDbInitialized)
							? gv_hViewDb
							: NULL,
							pLogicalFile, pBlkExp->uiLevel,
							uiBlkType, ucKeyBuffer);

	if ((iErrorCode = flmVerifyBlockHeader( &StateInfo, &BlockInfo,
						(FLMUINT)gv_ViewDbHdr.ui16BlockSize,
												pBlkExp->uiNextAddr,
												pBlkExp->uiPrevAddr,
												(FLMBOOL)(StateInfo.pDb != NULL
															 ? TRUE
															 : FALSE))) != 0)
	{
		SetStatusBit( pucBlkStatus, iErrorCode);
	}
#endif

	// Output each element in the block

	while (uiCurOffset <= (FLMUINT)(pBlkHdr->ui16NumKeys - 1))
	{
		InitStatusBits( ucElmStatus);
		uiElementCount++;
		GetElmInfo( (FLMBYTE *)pBlkHdr +
										pui16OffsetArray[ uiCurOffset],
						pBlkHdr, &StateInfo);
/*
		if ((iErrorCode = flmVerifyElement( &StateInfo,
									FO_DO_EXTENDED_DATA_CHECK)) != 0)
		{
			SetStatusBit( ucElmStatus, iErrorCode);
		}
*/
		if (!bStatusOnlyFlag)
		{
			uiRow++;

			// Output the element number

			if (!ViewAddMenuItem( LBL_ELEMENT_NUMBER, uiLabelWidth,
						VAL_IS_NUMBER | DISP_DECIMAL,
						(FLMUINT64)uiElementCount, 0,
						0,
						VIEW_INVALID_FILE_OFFSET,
						0, MOD_DISABLED,
						uiCol, uiRow++, 0,
						FLM_GREEN, FLM_WHITE,
						FLM_GREEN, FLM_WHITE))
			{
				goto Exit;
			}

			// Output the element offset within the block

			if (!ViewAddMenuItem( LBL_ELEMENT_OFFSET, uiLabelWidth,
						VAL_IS_NUMBER | DISP_DECIMAL,
						(FLMUINT64)StateInfo.uiElmOffset, 0,
						FSGetFileNumber( StateInfo.ui32BlkAddress),
						FSGetFileOffset( StateInfo.ui32BlkAddress) + 
								StateInfo.uiElmOffset,
						0, MOD_DISABLED,
						uiCol, uiRow++, 0, uiBackColor, uiForeColor,
						uiBackColor, uiForeColor))
			{
				goto Exit;
			}

			// Output the element length

			if (!ViewAddMenuItem( LBL_ELEMENT_LENGTH, uiLabelWidth,
						VAL_IS_NUMBER | DISP_DECIMAL,
						(FLMUINT64)StateInfo.uiElmLen, 0,
						FSGetFileNumber( StateInfo.ui32BlkAddress),
						FSGetFileOffset( StateInfo.ui32BlkAddress) + 
							StateInfo.uiElmOffset,
						0, MOD_DISABLED,
						uiCol, uiRow++, 0, uiBackColor, uiForeColor,
						uiBackColor, uiForeColor))
			{
				goto Exit;
			}

			// Display the child block address

			uiTempAddress = FB2UD( StateInfo.pucElm);
			if (uiTempAddress == 0)
			{
				uiOption = 0;
			}
			else
			{
				uiOption = BLK_OPTION_CHILD_BLOCK | StateInfo.uiElmOffset;
			}
			if (!ViewAddMenuItem( LBL_CHILD_BLOCK_ADDRESS, uiLabelWidth,
						VAL_IS_NUMBER | DISP_HEX_DECIMAL,
						(FLMUINT64)uiTempAddress, 0,
						FSGetFileNumber( StateInfo.ui32BlkAddress),
						FSGetFileOffset( StateInfo.ui32BlkAddress) +
																StateInfo.uiElmOffset,
						0, MOD_CHILD_BLK, uiCol, uiRow++, uiOption,
						(!uiOption ? uiBackColor : uiUnselectBackColor),
						(!uiOption ? uiForeColor : uiUnselectForeColor),
						(!uiOption ? uiBackColor : uiSelectBackColor),
						(!uiOption ? uiForeColor : uiSelectForeColor)))
			{
				goto Exit;
			}

			// Display the counts info if any.
			if (uiBlkType == BT_NON_LEAF_COUNTS)
			{
				if (!ViewAddMenuItem( LBL_CHILD_REFERENCE_COUNT, uiLabelWidth,
							VAL_IS_NUMBER,
							(FLMUINT64) StateInfo.uiElmCounts, 0,
							FSGetFileNumber( StateInfo.ui32BlkAddress),
							FSGetFileOffset( StateInfo.ui32BlkAddress) +
								StateInfo.uiElmCountsOffset,
							0, MOD_FLMUINT32 | MOD_DECIMAL,
							uiCol, uiRow++, uiOption, uiBackColor, uiForeColor,
							uiBackColor, uiForeColor))
				{
					goto Exit;
				}
			}

			// Remember this item if we are searching
#if 0
			if (gv_bViewSearching &&
				 flmCompareKeys( StateInfo.pucCurKey, StateInfo.uiCurKeyLen,
											gv_ucViewSearchKey, gv_uiViewSearchKeyLen) >= 0 &&
				 !gv_pViewSearchItem)
			{
				gv_pViewSearchItem = gv_pViewMenuLastItem;
			}
#endif
			if (iErrorCode != 0)
			{
				if (!OutputStatus( uiCol, &uiRow, uiBackColor, uiForeColor,
							uiLabelWidth, LBL_ELEMENT_STATUS, ucElmStatus))
				{
					goto Exit;
				}
			}

			// Display the key length

			if (!ViewAddMenuItem( LBL_KEY_LENGTH, uiLabelWidth,
					VAL_IS_NUMBER | DISP_DECIMAL,
					(FLMUINT64)StateInfo.uiElmKeyLen, 0,
					FSGetFileNumber( StateInfo.ui32BlkAddress),
					FSGetFileOffset( StateInfo.ui32BlkAddress) + 
						StateInfo.uiElmKeyLenOffset, 0,
					MOD_DECIMAL,
					uiCol, uiRow++, 0, uiBackColor, uiForeColor,
					uiBackColor, uiForeColor))
			{
				goto Exit;
			}

			// Output the key

			if (!OutputHexValue( uiCol, &uiRow,
					uiBackColor, uiForeColor, LBL_ELEMENT_KEY,
					FSGetFileNumber( StateInfo.ui32BlkAddress),
					FSGetFileOffset( StateInfo.ui32BlkAddress) + 
													StateInfo.uiElmKeyOffset,
					StateInfo.pucElmKey, StateInfo.uiElmKeyLen,
					FALSE))
			{
				goto Exit;
			}
		}

		// Go to the next element

		uiCurOffset++;
		OrStatusBits( pucBlkStatus, ucElmStatus);
	}

	if (!bStatusOnlyFlag)
	{
		*puiRow = uiRow;

		// If we were searching and did not find a key, set it on the
		// last key found

		if (gv_bViewSearching && !gv_pViewSearchItem)
		{
			gv_pViewSearchItem = gv_pViewMenuLastItem;
			while (gv_pViewSearchItem &&
					 gv_pViewSearchItem->iLabelIndex != LBL_CHILD_BLOCK_ADDRESS)
			{
				gv_pViewSearchItem = gv_pViewSearchItem->pPrevItem;
			}
		}
	}
	bOk = TRUE;

Exit:

#if 0
	if (bStateInitialized && StateInfo.pRecord)
	{
		StateInfo.pRecord->release( &StateInfo.pRecord);
	}
#endif
	return( bOk);
}

/***************************************************************************
Desc:	This routine outputs a NON-LEAF block, including the block header.
*****************************************************************************/
FLMBOOL ViewNonLeafBlk(
	FLMUINT			uiReadAddress,
	FLMUINT			uiBlkAddress,
	F_BLK_HDR **	ppBlkHdr,
	BLK_EXP_p		pBlkExp
	)
{
	FLMBOOL		bOk = FALSE;
	FLMUINT		uiRow;
	FLMUINT		uiCol;
	F_BLK_HDR *	pBlkHdr;
	FLMBYTE		ucBlkStatus [NUM_STATUS_BYTES];
	FLMUINT32	ui32CalcCRC;
	FLMUINT32	ui32BlkCRC;
	FLMUINT		uiBytesRead;

	InitStatusBits( ucBlkStatus);

	// Read the block into memory

	if (!ViewBlkRead( uiReadAddress, ppBlkHdr, TRUE,
							(FLMUINT)gv_ViewDbHdr.ui16BlockSize,
							&ui32CalcCRC, &ui32BlkCRC,
							&uiBytesRead, TRUE))
	{
		goto Exit;
	}
	pBlkHdr = *ppBlkHdr;

	ViewMenuInit( "NON-LEAF Block");

	// Output the block header first

	uiRow = 0;
	uiCol = 5;
	pBlkExp->uiType = (FLMUINT)pBlkHdr->ui8BlkType;
	pBlkExp->uiBlkAddr = uiBlkAddress;
	OutputNonLeafElements( uiCol, &uiRow, (F_BTREE_BLK_HDR *)pBlkHdr,
									pBlkExp, ucBlkStatus, TRUE);
	if (!ViewOutBlkHdr( uiCol, &uiRow, pBlkHdr, pBlkExp, ucBlkStatus,
							ui32CalcCRC, ui32BlkCRC))
	{
		goto Exit;
	}

	// Now output the non-leaf data

	if (!OutputNonLeafElements( uiCol, &uiRow, (F_BTREE_BLK_HDR *)pBlkHdr,
						pBlkExp, ucBlkStatus, FALSE))
	{
		goto Exit;
	}
	bOk = TRUE;

Exit:

	return( bOk);
}

/********************************************************************
Desc:	View a block in HEX
*********************************************************************/
void ViewHexBlock(
	FLMUINT			uiReadAddress,
	F_BLK_HDR **	ppBlkHdr,
	FLMUINT			uiViewLen
	)
{
	FLMBYTE *	pucBlk;
	FLMUINT		uiRow = 0;
	FLMUINT		uiCol = 13;
	FLMUINT		uiBytesPerLine = MAX_HORIZ_SIZE( uiCol);
	FLMUINT		uiBytesProcessed = 0;
	FLMUINT		uiNumBytes;
	FLMUINT		uiFileOffset;
	FLMUINT		uiFileNumber;
	char			szTitle [80];
	FLMUINT		uiBytesRead;

	uiFileOffset = FSGetFileOffset( uiReadAddress);
	uiFileNumber = FSGetFileNumber( uiReadAddress);

	if (!ViewBlkRead( uiReadAddress, ppBlkHdr, FALSE, uiViewLen,
						NULL, NULL, &uiBytesRead, TRUE))
	{
		return;
	}
	pucBlk = (FLMBYTE *)(*ppBlkHdr);

	f_sprintf( szTitle, "HEX DISPLAY OF BLOCK %08X", (unsigned)uiReadAddress);
	ViewMenuInit( szTitle);

	while (uiBytesProcessed < uiViewLen)
	{
		if ((uiNumBytes = uiViewLen - uiBytesProcessed) > uiBytesPerLine)
		{
			uiNumBytes = uiBytesPerLine;
		}

		// Output the line

		if (!ViewAddMenuItem( -1, 0,
				VAL_IS_BINARY_HEX,
				(FLMUINT)pucBlk, uiNumBytes,
				uiFileNumber, uiFileOffset, uiNumBytes, MOD_BINARY,
				uiCol, uiRow++, 0,
				FLM_BLACK, FLM_LIGHTGRAY,
				FLM_BLACK, FLM_LIGHTGRAY))
		{
			return;
		}
		uiFileOffset += uiNumBytes;
		uiBytesProcessed += uiNumBytes;
		pucBlk += uiNumBytes;
	}
}

/***************************************************************************
Desc: This routine outputs a block in the database.  Depending on the
		type of block, it will call a different routine to display
		the block.  The routine then allows the user to press keys to
		navigate to other blocks in the database if desired.
*****************************************************************************/
void ViewBlocks(
	FLMUINT		uiReadAddress,
	FLMUINT		uiBlkAddress,
	BLK_EXP_p	pBlkExp
	)
{
	FLMUINT		uiOption;
	VIEW_INFO   SaveView;
	VIEW_INFO   DummySave;
	FLMBOOL		bRepaint = TRUE;
	F_BLK_HDR *	pBlkHdr = NULL;
	FLMUINT		uiBlkAddress2;
	BLK_EXP     BlkExp2;
	FLMBOOL		bSetExp = FALSE;
	FLMBOOL		bViewHexFlag = FALSE;
	FLMUINT		uiBytesRead;

	// Loop getting commands until hit the exit key

	if (!gv_bViewHdrRead)
	{
		ViewReadHdr();
	}
	gv_pViewSearchItem = NULL;
	ViewReset( &SaveView);
	while (!gv_bViewPoppingStack)
	{

		// Display the type of block expected

		if (bRepaint)
		{
			if (bViewHexFlag)
			{
				ViewHexBlock( uiReadAddress, &pBlkHdr,
											(FLMUINT)gv_ViewDbHdr.ui16BlockSize);
			}
			else
			{
Switch_Statement:
				switch (pBlkExp->uiType)
				{
					case BT_NON_LEAF_COUNTS:
					case BT_NON_LEAF:
						if (!ViewNonLeafBlk( uiReadAddress, uiBlkAddress,
																 &pBlkHdr, pBlkExp))
						{
							goto Exit;
						}
						break;
					case BT_LEAF:
					case BT_LEAF_DATA:
						if (!ViewLeafBlk( uiReadAddress, uiBlkAddress,
															&pBlkHdr, pBlkExp))
						{
							goto Exit;
						}
						break;
					case BT_DATA_ONLY:
						if (!ViewDataBlk( uiReadAddress, uiBlkAddress,
															&pBlkHdr, pBlkExp))
						{
							goto Exit;
						}
						break;
					case BT_FREE:
						if (!ViewAvailBlk( uiReadAddress, uiBlkAddress,
															 &pBlkHdr, pBlkExp))
						{
							goto Exit;
						}
						break;
					case BT_LFH_BLK:
						if (!ViewLFHBlk( uiReadAddress, uiBlkAddress,
														 &pBlkHdr, pBlkExp))
						{
							goto Exit;
						}
						break;
					case 0xFF:
						if (!ViewBlkRead( uiReadAddress, &pBlkHdr, TRUE,
										(FLMUINT)gv_ViewDbHdr.ui16BlockSize,
												NULL, NULL, &uiBytesRead, FALSE))
						{
							goto Exit;
						}
						else
						{
							pBlkExp->uiType = (FLMUINT)pBlkHdr->ui8BlkType;
							goto Switch_Statement;
						}
				}
			}
		}

		// See what the user wants to do next.

		if (bSetExp &&
			 (pBlkExp->uiType == BT_LEAF || 
			  pBlkExp->uiType == BT_NON_LEAF_COUNTS ||
			  pBlkExp->uiType == BT_NON_LEAF ||
			  pBlkExp->uiType == BT_LEAF_DATA))
		{
			pBlkExp->uiLfNum =
				(FLMUINT)(((F_BTREE_BLK_HDR *)pBlkHdr)->ui16LogicalFile);
			pBlkExp->uiLevel =
				(FLMUINT)(((F_BTREE_BLK_HDR *)pBlkHdr)->ui8BlkLevel);
		}
		bSetExp = FALSE;
		bRepaint = TRUE;
		if (gv_bViewSearching)
		{
			SetSearchTopBottom();
			if (pBlkExp->uiType == BT_LEAF ||
				 pBlkExp->uiType == BT_LEAF_DATA ||
				 !gv_pViewSearchItem)
			{
				gv_bViewSearching = FALSE;
				ViewEnable();
				uiOption = ViewGetMenuOption();
			}
			else
			{
				uiOption = gv_pViewSearchItem->uiOption;
			}
		}
		else
		{
			ViewEnable();
			uiOption = ViewGetMenuOption();
		}
		switch (uiOption)
		{
			case ESCAPE_OPTION:
				goto Exit;
			case PREV_BLOCK_OPTION:
				ViewReset( &DummySave);
				pBlkExp->uiNextAddr = uiBlkAddress;
				pBlkExp->uiPrevAddr = 0xFFFFFFFF;
				uiReadAddress = uiBlkAddress =
						(FLMUINT)pBlkHdr->ui32PrevBlkInChain;
				break;
			case NEXT_BLOCK_OPTION:
				ViewReset( &DummySave);
				pBlkExp->uiNextAddr = 0xFFFFFFFF;
				pBlkExp->uiPrevAddr = uiBlkAddress;
				uiReadAddress = uiBlkAddress =
						(FLMUINT)pBlkHdr->ui32NextBlkInChain;
				break;
			case PREV_BLOCK_IMAGE_OPTION:
				f_memcpy( &BlkExp2, pBlkExp, sizeof( BLK_EXP));
				BlkExp2.uiNextAddr = 0xFFFFFFFF;
				BlkExp2.uiPrevAddr = 0xFFFFFFFF;
				ViewBlocks( (FLMUINT)pBlkHdr->ui32PriorBlkImgAddr,
										uiBlkAddress, &BlkExp2);
				break;
			case GOTO_BLOCK_OPTION:
				if (GetBlockAddrType( &uiBlkAddress2))
				{
					ViewReset( &DummySave);
					uiReadAddress = uiBlkAddress = uiBlkAddress2;
					pBlkExp->uiType = 0xFF;
					pBlkExp->uiLevel = 0xFF;
					pBlkExp->uiNextAddr = 0xFFFFFFFF;
					pBlkExp->uiPrevAddr = 0xFFFFFFFF;
					pBlkExp->uiLfNum = 0;
					bSetExp = TRUE;
					if (uiBlkAddress < 2048)
					{
						bViewHexFlag = TRUE;
					}
				}
				else
				{
					bRepaint = FALSE;
				}
				break;
			case EDIT_OPTION:
			case EDIT_RAW_OPTION:
				if (!ViewEdit( (uiOption == EDIT_OPTION) ? TRUE : FALSE))
				{
					bRepaint = FALSE;
				}
				break;
			case HEX_OPTION:
				ViewDisable();
				bViewHexFlag = !bViewHexFlag;
				break;
			case SEARCH_OPTION:
				switch (pBlkHdr->ui8BlkType)
				{
					case BT_NON_LEAF_COUNTS:
					case BT_NON_LEAF:
					case BT_LEAF_DATA:
					case BT_LEAF:
						gv_uiViewSearchLfNum =
							(FLMUINT)(((F_BTREE_BLK_HDR *)pBlkHdr)->ui16LogicalFile);
						gv_uiViewSearchLfType =
							getBlkLfType( (F_BTREE_BLK_HDR *)pBlkHdr);
						if (ViewGetKey())
						{
							gv_bViewPoppingStack = TRUE;
						}
						break;
					case BT_LFH_BLK:
						{
							VIEW_MENU_ITEM_p  pMenuItem = gv_pViewMenuCurrItem;

							// Determine which logical file, if any we are pointing at

							while (pMenuItem &&
									 pMenuItem->iLabelIndex != LBL_LOGICAL_FILE_NAME)
							{
								pMenuItem = pMenuItem->pPrevItem;
							}
							if (pMenuItem)
							{
								while (pMenuItem &&
										 pMenuItem->iLabelIndex != LBL_LOGICAL_FILE_NUMBER)
								{
									pMenuItem = pMenuItem->pNextItem;
								}
							}
							if (pMenuItem)
							{
								FLMBYTE *	pszTmp;

								gv_uiViewSearchLfNum = (FLMUINT)pMenuItem->ui64Value;
								while (pMenuItem &&
										 pMenuItem->iLabelIndex != LBL_LOGICAL_FILE_TYPE)
								{
									pMenuItem = pMenuItem->pNextItem;
								}
								pszTmp = (FLMBYTE *)((FLMUINT)pMenuItem->ui64Value);
								if (f_stricmp( (const char *)pszTmp, COLLECTION_STRING) == 0)
								{
									gv_uiViewSearchLfType = XFLM_LF_COLLECTION;
								}
								else
								{
									gv_uiViewSearchLfType = XFLM_LF_INDEX;
								}
								if (ViewGetKey())
								{
									gv_bViewPoppingStack = TRUE;
								}
							}
							else
							{
								ViewShowError(
									"Position cursor to a logical file before searching");
							}
						}
						break;
					default:
						ViewShowError(
							"This block does not belong to a logical file - cannot search");
						break;
				}
				break;
			default:
				if (uiOption & LOGICAL_INDEX_OPTION)
				{
					ViewLogicalFile( (FLMUINT)(uiOption &
											(~(LOGICAL_INDEX_OPTION))), XFLM_LF_INDEX);
				}
				else if (uiOption & LOGICAL_CONTAINER_OPTION)
				{
					ViewLogicalFile( (FLMUINT)(uiOption &
											(~(LOGICAL_CONTAINER_OPTION))), XFLM_LF_COLLECTION);
				}
				else if (uiOption & LFH_OPTION_ROOT_BLOCK)
				{
					F_LF_HDR *	pLfHdr = (F_LF_HDR *)((FLMBYTE *)pBlkHdr +
																	SIZEOF_STD_BLK_HDR);
						
					pLfHdr += (uiOption & 0xFFFF);

					BlkExp2.uiLevel = 0xFF;
					BlkExp2.uiType = 0xFF;
					uiBlkAddress2 = (FLMUINT)pLfHdr->ui32RootBlkAddr;
					BlkExp2.uiLfNum = (FLMUINT)pLfHdr->ui32LfNumber;
					BlkExp2.uiNextAddr = BlkExp2.uiPrevAddr = 0;
					ViewBlocks( uiBlkAddress2, uiBlkAddress2, &BlkExp2);
				}
				else if (uiOption & BLK_OPTION_CHILD_BLOCK)
				{
					uiBlkAddress2 = (FLMUINT)gv_pViewMenuCurrItem->ui64Value;
					f_memcpy( &BlkExp2, pBlkExp, sizeof( BLK_EXP));
					BlkExp2.uiNextAddr = 0xFFFFFFFF;
					BlkExp2.uiPrevAddr = 0xFFFFFFFF;
					BlkExp2.uiLevel = 0xFF;
					BlkExp2.uiType = 0xFF;
					ViewBlocks( uiBlkAddress2, uiBlkAddress2, &BlkExp2);
				}
				else if (uiOption & BLK_OPTION_DATA_BLOCK)
				{
					uiBlkAddress2 = (FLMUINT)gv_pViewMenuCurrItem->ui64Value;
					f_memcpy( &BlkExp2, pBlkExp, sizeof( BLK_EXP));
					BlkExp2.uiNextAddr = 0xFFFFFFFF;
					BlkExp2.uiPrevAddr = 0xFFFFFFFF;
					BlkExp2.uiLevel = 0xFF;
					BlkExp2.uiType = 0xFF;
					ViewBlocks( uiBlkAddress2, uiBlkAddress2, &BlkExp2);
				}
				else
				{
					bRepaint = FALSE;
				}
				break;
		}
	}

Exit:

	f_free( &pBlkHdr);
	ViewRestore( &SaveView);
}

/********************************************************************
Desc:	Have user enter a block address.
*********************************************************************/
FLMBOOL GetBlockAddrType(
	FLMUINT *	puiBlkAddress
	)
{
	FLMBOOL	bGotAddress = FALSE;
	FLMBOOL	bBadDigit;
	char		szTempBuf [20];
	FLMUINT	uiLoop;
	FLMUINT	uiChar;
	FLMUINT	uiNumCols;
	FLMUINT	uiNumRows;

	f_conGetScreenSize( &uiNumCols, &uiNumRows);

	// Get the block address

	for (;;)
	{
		bBadDigit = FALSE;
		f_conSetBackFore( FLM_BLACK, FLM_WHITE);
		f_conClearScreen( 0, uiNumRows - 2);
		ViewAskInput( 
			"Enter Block Address (in hex): ", 
			szTempBuf, sizeof( szTempBuf));
		if (f_stricmp( szTempBuf, "\\") == 0 || !szTempBuf [0])
		{
			break;
		}
		uiLoop = 0;
		*puiBlkAddress = 0;
		while (szTempBuf [uiLoop] && uiLoop < 8)
		{
			(*puiBlkAddress) <<= 4;
			uiChar = (FLMUINT)szTempBuf [uiLoop];
			if (uiChar >= '0' && uiChar <= '9')
			{
				(*puiBlkAddress) += (FLMUINT)(uiChar - '0');
			}
			else if (uiChar >= 'a' && uiChar <= 'f')
			{
				(*puiBlkAddress) += (FLMUINT)(uiChar - 'a' + 10);
			}
			else if (uiChar >= 'A' && uiChar<= 'F')
			{
				(*puiBlkAddress) += (FLMUINT)(uiChar - 'A' + 10);
			}
			else
			{
				bBadDigit = TRUE;
				break;
			}
			uiLoop++;
		}
		if (bBadDigit)
		{
			ViewShowError( 
				"Illegal digit in number - must be hex digits");
		}
		else if (szTempBuf [uiLoop])
		{
			ViewShowError( "Too many characters in number");
		}
		else
		{
			bGotAddress = TRUE;
			break;
		}
	}

	f_conClearScreen( 0, uiNumRows - 2);
	ViewEscPrompt();
	return( bGotAddress);
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC void SetSearchTopBottom( void)
{
	if (!gv_pViewSearchItem)
	{
		gv_pViewMenuCurrItem = NULL;
		gv_uiViewMenuCurrItemNum = 0;
		gv_uiViewCurrFileOffset = 0;
		gv_uiViewCurrFileNumber = 0;
		gv_uiViewTopRow = 0;
	}
	else
	{
		gv_pViewMenuCurrItem = gv_pViewSearchItem;
		gv_uiViewMenuCurrItemNum = gv_pViewSearchItem->uiItemNum;
		gv_uiViewCurrFileOffset = gv_pViewSearchItem->uiModFileOffset;
		gv_uiViewCurrFileNumber = gv_pViewSearchItem->uiModFileNumber;
		gv_uiViewTopRow = gv_pViewSearchItem->uiRow;
		if (gv_uiViewTopRow < LINES_PER_PAGE / 2)
		{
			gv_uiViewTopRow = 0;
		}
		else
		{
			gv_uiViewTopRow -= (LINES_PER_PAGE / 2);
		}
	}
	gv_uiViewBottomRow = gv_uiViewTopRow + LINES_PER_PAGE - 1;
	if (gv_uiViewBottomRow > gv_pViewMenuLastItem->uiRow)
	{
		gv_uiViewBottomRow = gv_pViewMenuLastItem->uiRow;
		if (gv_uiViewBottomRow < LINES_PER_PAGE)
		{
			gv_uiViewTopRow = 0;
		}
		else
		{
			gv_uiViewTopRow = gv_uiViewBottomRow - LINES_PER_PAGE + 1;
		}
	}
}

/*=============================================================================
Desc:
=============================================================================*/
FSTATIC void GetElmInfo(
	FLMBYTE *			pucEntry,
	F_BTREE_BLK_HDR *	pBlkHdr,
	STATE_INFO *		pStateInfo
	)
{
	F_NodeVerifier *		pNodeVerifier = pStateInfo->pNodeVerifier;
	
	f_memset( pStateInfo, 0, sizeof(STATE_INFO));

	pStateInfo->pNodeVerifier = pNodeVerifier;
	pStateInfo->pBlkHdr = (F_BLK_HDR *)pBlkHdr;
	pStateInfo->ui32BlkAddress = pBlkHdr->stdBlkHdr.ui32BlkAddr;
	pStateInfo->pucElm = pucEntry;
	pStateInfo->uiElmOffset = (FLMUINT)pucEntry - (FLMUINT)pBlkHdr;

	switch ( pBlkHdr->stdBlkHdr.ui8BlkType)
	{

		// Leaf node - no data.  Only element and key.
		case BT_LEAF:
		{
			pStateInfo->uiElmKeyLenOffset = pStateInfo->uiElmOffset; // No flags.
			pStateInfo->uiElmKeyLen = FB2UW( pucEntry);
			pStateInfo->uiElmLen = pStateInfo->uiElmKeyLen + 2;  // 2 byte key len
			pStateInfo->uiElmKeyOffset = pStateInfo->uiElmOffset + 2;	// ditto
			pStateInfo->pucElmKey = &pucEntry[ 2];  // Key
			break;
		}
		
		// Leaf node with data
		case BT_LEAF_DATA:
		{
			FLMBYTE		ucFlag;
			FLMBYTE *	pucTmp;
			FLMUINT		uiElmLen;

			pucTmp = &pucEntry[ 1];
			ucFlag = pucEntry[ 0];
			uiElmLen = 1;  // Flag

			pStateInfo->uiElmKeyLenOffset = (FLMUINT)pucTmp - (FLMUINT)pBlkHdr;
			
			if (bteKeyLenFlag( pucEntry))
			{
				uiElmLen += 2;
				pStateInfo->uiElmKeyLen = FB2UW( pucTmp);
				uiElmLen += pStateInfo->uiElmKeyLen;
				pucTmp += 2;
			}
			else
			{
				uiElmLen++;
				pStateInfo->uiElmKeyLen = *pucTmp;
				uiElmLen += pStateInfo->uiElmKeyLen;
				pucTmp++;
			}

			pStateInfo->uiElmDataLenOffset = (FLMUINT)pucTmp - (FLMUINT)pBlkHdr;

			if (bteDataLenFlag( pucEntry))
			{
				uiElmLen += 2;
				pStateInfo->uiElmDataLen = FB2UW( pucTmp);
				uiElmLen += pStateInfo->uiElmDataLen;
				pucTmp += 2;
			}
			else
			{
				uiElmLen++;
				pStateInfo->uiElmDataLen = *pucTmp;
				uiElmLen += pStateInfo->uiElmDataLen;
				pucTmp++;
			}
			
			if (bteOADataLenFlag( pucEntry))
			{
				pStateInfo->uiElmOADataLen = FB2UD( pucTmp);
				pStateInfo->uiElmOADataLenOffset = (FLMUINT)pucTmp - (FLMUINT)pBlkHdr;
				uiElmLen += 4; // Add the OA Data Length
				pucTmp += 4;  // Skip over the OA Data Length
			}
			// Save the key pointer...
			pStateInfo->pucElmKey = pucTmp;
			pStateInfo->uiElmKeyOffset = (FLMUINT)pucTmp - (FLMUINT)pBlkHdr;

			// Now the data
			pucTmp += pStateInfo->uiElmKeyLen;
			pStateInfo->pucElmData = pucTmp;
			pStateInfo->uiElmDataOffset = (FLMUINT)pucTmp - (FLMUINT)pBlkHdr;

			// The element length
			pStateInfo->uiElmLen = uiElmLen;
			break;
		}

		case BT_NON_LEAF:
		{
			FLMBYTE *	pucTmp = pucEntry;

			// Skip over the child block address for now.
			pucTmp += 4;
			
			// Get the key length & offset
			pStateInfo->uiElmKeyLen = FB2UW( pucTmp);
			pStateInfo->uiElmKeyLenOffset = (FLMUINT)pucTmp - (FLMUINT)pBlkHdr;
			
			// Get the key and offset
			pucTmp += 2;
			pStateInfo->pucElmKey = pucTmp;
			pStateInfo->uiElmKeyOffset = (FLMUINT)pucTmp - (FLMUINT)pBlkHdr;

			// The element length
			pStateInfo->uiElmLen = pStateInfo->uiElmKeyLen + 6;
			break;
		}

		case BT_NON_LEAF_COUNTS:
		{
			FLMBYTE *	pucTmp = pucEntry;
			
			// Skip over the child block address for now.
			pucTmp += 4;

			// Get the counts and offset
			pStateInfo->uiElmCounts = FB2UD( pucTmp);
			pStateInfo->uiElmCountsOffset = (FLMUINT)pucTmp - (FLMUINT)pBlkHdr;
			
			// Get the key length & offset
			pucTmp += 4;
			pStateInfo->uiElmKeyLen = FB2UW( pucTmp);
			pStateInfo->uiElmKeyLenOffset = (FLMUINT)pucTmp - (FLMUINT)pBlkHdr;
			
			// Get the key and offset
			pucTmp += 2;
			pStateInfo->pucElmKey = pucTmp;
			pStateInfo->uiElmKeyOffset = (FLMUINT)pucTmp - (FLMUINT)pBlkHdr;

			// The element length
			pStateInfo->uiElmLen = pStateInfo->uiElmKeyLen + 10;
			break;
		}

		case BT_DATA_ONLY:
		{
			FLMBYTE *		pucTmp = pucEntry;
			if (!pBlkHdr->stdBlkHdr.ui32PrevBlkInChain)
			{
				pStateInfo->uiElmKeyLen = (FLMUINT)FB2UW( pucTmp);
				pStateInfo->uiElmKeyLenOffset = (FLMUINT)pucTmp - (FLMUINT)pBlkHdr;
				pucTmp += 2;
				pStateInfo->pucElmKey = pucTmp;
				pStateInfo->uiElmKeyOffset = (FLMUINT)pucTmp - (FLMUINT)pBlkHdr;
				pucTmp += pStateInfo->uiElmKeyLen;
			}
			else
			{
				pStateInfo->uiElmKeyLen = 0;
			}
			pStateInfo->uiElmLen = gv_ViewDbHdr.ui16BlockSize - 
						  pBlkHdr->stdBlkHdr.ui16BlkBytesAvail;
			pStateInfo->uiElmDataLen = pStateInfo->uiElmLen -
												pStateInfo->uiElmKeyLen;
			pStateInfo->pucElmData = pucTmp +
				(pStateInfo->uiElmKeyLen ? pStateInfo->uiElmKeyLen + 2 : 0);

		}
	}
}

