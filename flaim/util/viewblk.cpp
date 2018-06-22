//-------------------------------------------------------------------------
// Desc:	View database blocks.
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
	FLMBYTE * StatusBytes);

FSTATIC void OrStatusBits(
	FLMBYTE * DestStatusBytes,
	FLMBYTE * SrcStatusBytes);

FSTATIC void SetStatusBit(
	FLMBYTE *			StatusBytes,
	eCorruptionType	eCorruptionCode);

FSTATIC FLMBOOL TestStatusBit(
	FLMBYTE *			StatusBytes,
	eCorruptionType	eCorruptionCode);

FSTATIC FLMINT OutputElmRecord(
	FLMUINT       Col,
	FLMUINT  *    RowRV,
	eColorType	  bc,
	eColorType	  fc,
	FLMUINT       LabelWidth,
	STATE_INFO *  StateInfo,
	FLMBYTE *     StatusBytes,
	FLMUINT       StatusOnlyFlag);

FSTATIC FLMINT OutBlkHdrExpNum(
	FLMUINT			Col,
	FLMUINT  *		RowRV,
	eColorType		bc,
	eColorType		fc,
	eColorType		mbc,
	eColorType		mfc,
	eColorType		sbc,
	eColorType		sfc,
	FLMUINT     	LabelWidth,
	FLMINT      	iLabelIndex,
	FLMUINT			FileNumber,
	FLMUINT     	FileOffset,
	FLMBYTE *   	ValuePtr,
	FLMUINT     	ValueType,
	FLMUINT     	ModType,
	FLMUINT     	ExpNum,
	FLMUINT     	IgnoreExpNum,
	FLMUINT     	Option);

FSTATIC void FormatBlkType(
	FLMBYTE *    TempBuf,
	FLMUINT      BlkType);

FSTATIC FLMINT OutputStatus(
	FLMUINT     Col,
	FLMUINT  *	RowRV,
	eColorType  bc,
	eColorType  fc,
	FLMUINT     LabelWidth,
	FLMINT      iLabelIndex,
	FLMBYTE *   StatusFlags);

FSTATIC FLMINT OutputHexValue(
	FLMUINT     Col,
	FLMUINT  *  RowRV,
	eColorType  bc,
	eColorType  fc,
	FLMINT		LabelIndex,
	FLMUINT		FileNumber,
	FLMUINT     FileOffset,
	FLMBYTE *   ValPtr,
	FLMUINT     ValLen,
	FLMUINT     CopyVal,
	FLMUINT		uiModFlag);

FSTATIC FLMINT OutputLeafElements(
	FLMUINT     Col,
	FLMUINT  *	RowRV,
	FLMBYTE *   BlkPtr,
	BLK_EXP_p   BlkExp,
	FLMBYTE *   BlkStatusRV,
	FLMUINT		StatusOnlyFlag,
	FLMBOOL		bEncrypted);

FSTATIC FLMINT OutputNonLeafElements(
	FLMUINT     Col,
	FLMUINT  *	RowRV,
	FLMBYTE *   BlkPtr,
	BLK_EXP_p   BlkExp,
	FLMBYTE *   BlkStatusRV,
	FLMUINT     StatusOnlyFlag,
	FLMBOOL		bEncrypted);

FSTATIC void SetSearchTopBottom(
	void
	);

extern FLMUINT	gv_uiTopLine;
extern FLMUINT	gv_uiBottomLine;

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC void InitStatusBits(
	FLMBYTE * StatusBytes
	)
{
	FLMUINT  i;

	if (StatusBytes != NULL)
	{
		for( i = 0; i < NUM_STATUS_BYTES; i++)
			StatusBytes [i] = 0;
	}
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC void OrStatusBits(
	FLMBYTE * DestStatusBytes,
	FLMBYTE * SrcStatusBytes
	)
{
	FLMUINT  i;

	if ((DestStatusBytes != NULL) && (SrcStatusBytes != NULL))
	{
		for( i = 0; i < NUM_STATUS_BYTES; i++)
			DestStatusBytes [i] |= SrcStatusBytes [i];
	}
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC void SetStatusBit(
	FLMBYTE *			StatusBytes,
	eCorruptionType	eCorruptionCode
	)
{
	FLMUINT   ByteOffset = ((FLMUINT)eCorruptionCode - 1) / 8;
	FLMBYTE   BitToSet = (FLMBYTE)0x80 >> (FLMBYTE)(((FLMUINT)eCorruptionCode - 1) % 8);

	if (StatusBytes != NULL)
		StatusBytes [ByteOffset] |= BitToSet;
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC FLMBOOL TestStatusBit(
	FLMBYTE *			StatusBytes,
	eCorruptionType	eCorruptionCode
	)
{
	if (eCorruptionCode == FLM_NO_CORRUPTION)
	{
		return( FALSE);
	}
	else
	{
		FLMUINT   ByteOffset = ((FLMUINT)eCorruptionCode - 1) / 8;
		FLMBYTE   BitToTest = (FLMBYTE)0x80 >> (FLMBYTE)(((FLMUINT)eCorruptionCode - 1) % 8);

		if (StatusBytes == NULL)
		{
			return( FALSE);
		}
		else
		{
			return( (StatusBytes [ByteOffset] & BitToTest) ? TRUE : FALSE);
		}
	}
}

/***************************************************************************
Name: OutputStatus
Desc: This routine outputs all of the status bits which were set for
		a block.  Each error discovered in a block will be displayed on
		a separate line.
*****************************************************************************/
FSTATIC FLMINT OutputStatus(
	FLMUINT     Col,
	FLMUINT  *	RowRV,
	eColorType	bc,
	eColorType	fc,
	FLMUINT     LabelWidth,
	FLMINT      iLabelIndex,
	FLMBYTE *   StatusFlags)
{
	FLMUINT	Row = *RowRV;
	FLMUINT	HadError = FALSE;
	FLMUINT	uiLoop;

	/* Output each error on a separate line */

	if (StatusFlags)
	{
		for( uiLoop = (FLMUINT)FLM_NO_CORRUPTION + 1; uiLoop < (FLMUINT)FLM_LAST_CORRUPT_ERROR; uiLoop++)
		{
			if (TestStatusBit( StatusFlags, (eCorruptionType)uiLoop))
			{
				HadError = TRUE;
				if (!ViewAddMenuItem( iLabelIndex, LabelWidth,
							VAL_IS_ERR_INDEX, (FLMUINT)uiLoop, 0,
							0, VIEW_INVALID_FILE_OFFSET, 0, MOD_DISABLED,
							Col, Row++, 0,
							FLM_RED, FLM_LIGHTGRAY,
							FLM_RED, FLM_LIGHTGRAY))
					return( 0);

				/* Set iLabelIndex to -1 so that it will not be displayed after */
				/* the first one. */

				iLabelIndex = -1;
			}
		}
	}

	/* If there were no errors in the block, just output an OK status */

	if (!HadError)
	{
		if (!ViewAddMenuItem( iLabelIndex, LabelWidth,
					VAL_IS_LABEL_INDEX, (FLMUINT)LBL_OK, 0,
					0, VIEW_INVALID_FILE_OFFSET, 0, MOD_DISABLED,
					Col, Row++, 0, bc, fc, bc, fc))
			return( 0);
	}
	*RowRV = Row;
	return( 1);
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC FLMINT OutputElmRecord(
	FLMUINT			Col,
	FLMUINT  *		RowRV,
	eColorType		bc,
	eColorType		fc,
	FLMUINT        LabelWidth,
	STATE_INFO *   StateInfo,
	FLMBYTE *      StatusBytes,
	FLMUINT        StatusOnlyFlag)
{
	eCorruptionType	eCorruptionCode;
	FLMUINT				SaveElmRecOffset;
	FLMBYTE				FOPStatusBytes [NUM_STATUS_BYTES];
	FLMUINT				FOPType;
	FLMUINT				FieldType;
	FLMUINT				Row = *RowRV;
	FLMUINT				FileOffset;
	FLMUINT				FileNumber;
	FLMUINT				FldOverhead;
	FLMBYTE *			FieldPtr;
	FLMUINT				FldFlags;

	for( ;;)
	{
		InitStatusBits( FOPStatusBytes);
		SaveElmRecOffset = StateInfo->uiElmRecOffset;
		FieldPtr = &StateInfo->pElmRec [SaveElmRecOffset];
		FileOffset = StateInfo->uiBlkAddress +
								 (StateInfo->pElmRec - StateInfo->pBlk) +
								 SaveElmRecOffset;
		FileNumber = FSGetFileNumber( StateInfo->uiBlkAddress);

		if ((eCorruptionCode = flmVerifyElmFOP( StateInfo)) != FLM_NO_CORRUPTION)
		{
			SetStatusBit( FOPStatusBytes, eCorruptionCode);
			SetStatusBit( StatusBytes, eCorruptionCode);
		}

		/* Output the field overhead */

		if (!StatusOnlyFlag)
		{
			Row++;

			/* First output the FOP type */

			switch( StateInfo->uiFOPType)
			{
				case FLM_FOP_CONT_DATA:
					FOPType = LBL_FOP_CONT;
					FldOverhead = 0;
					break;
				case FLM_FOP_STANDARD:
					FOPType = LBL_FOP_STD;
					FldOverhead = 2;
					break;
				case FLM_FOP_OPEN:
					FOPType = LBL_FOP_OPEN;
					FldFlags = FOP_GET_FLD_FLAGS( FieldPtr);
					FldOverhead = ((FOP_2BYTE_FLDNUM( FldFlags)) ? 3 : 2) +
									  ((FOP_2BYTE_FLDLEN( FldFlags)) ? 2 : 1);
					break;
				case FLM_FOP_TAGGED:
					FOPType = LBL_FOP_TAGGED;
					FldFlags = FOP_GET_FLD_FLAGS( FieldPtr);
					FldOverhead = ((FOP_2BYTE_FLDNUM( FldFlags)) ? 4 : 3) +
									  ((FOP_2BYTE_FLDLEN( FldFlags)) ? 2 : 1);
					break;
				case FLM_FOP_NO_VALUE:
					FOPType = LBL_FOP_NO_VALUE;
					FldFlags = FOP_GET_FLD_FLAGS( FieldPtr);
					FldOverhead = ((FOP_2BYTE_FLDNUM( FldFlags)) ? 3 : 2);
					break;
				case FLM_FOP_JUMP_LEVEL:
					FOPType = LBL_FOP_SET_LEVEL;
					FldOverhead = 1;
					break;
				case FLM_FOP_REC_INFO:
					FOPType = LBL_FOP_REC_INFO;
					FldFlags = FOP_GET_FLD_FLAGS( FieldPtr);
					FldOverhead = ((FOP_2BYTE_FLDLEN( FldFlags)) ? 3 : 2);
					break;
				case FLM_FOP_ENCRYPTED:
					FOPType = LBL_FOP_ENCRYPTED;
					FldFlags = 0;
					FldOverhead = 2;
					break;
				case FLM_FOP_BAD:
				default:
					FOPType = LBL_FOP_BAD;
					FldOverhead = 0;
					break;
			}
			if (!ViewAddMenuItem( LBL_FOP_TYPE, LabelWidth,
						VAL_IS_LABEL_INDEX, (FLMUINT)FOPType, 0,
						FileNumber, FileOffset, FldOverhead,
						(FLMBYTE)(MOD_BINARY | (FLMBYTE)(!FldOverhead
																 ? (FLMBYTE)MOD_DISABLED
																 : (FLMBYTE)0)),
						Col, Row++, 0, FLM_BLUE, FLM_LIGHTGRAY,
						FLM_BLUE, FLM_LIGHTGRAY))
				return( 0);

			/* Output the offset of the field within the element record */

			if (!ViewAddMenuItem( LBL_FIELD_OFFSET, LabelWidth,
						VAL_IS_NUMBER | DISP_DECIMAL,
						(FLMUINT)SaveElmRecOffset, 0,
						FileNumber, FileOffset, 0, MOD_DISABLED,
						Col, Row++, 0, bc, fc, bc, fc))
				return( 0);

			/* Output the field number, field type, and field level if applicable */

			if ((StateInfo->uiFOPType != FLM_FOP_CONT_DATA) &&
					(StateInfo->uiFOPType != FLM_FOP_JUMP_LEVEL) &&
					(StateInfo->uiFOPType != FLM_FOP_BAD) &&
					(StateInfo->uiFOPType != FLM_FOP_NEXT_DRN) &&
					(StateInfo->uiFOPType != FLM_FOP_REC_INFO) &&
					(eCorruptionCode != FLM_BAD_ELM_FLD_OVERHEAD))
			{
				if (!ViewAddMenuItem( LBL_FIELD_NUMBER, LabelWidth,
							VAL_IS_NUMBER | DISP_DECIMAL,
							(FLMUINT)StateInfo->uiFieldNum, 0,
							FileNumber, FileOffset, 0, MOD_DISABLED,
							Col, Row++, 0, bc, fc, bc, fc))
					return( 0);

				/* Output the field type */

				switch( StateInfo->uiFieldType)
				{
					case FLM_TEXT_TYPE:
						FieldType = LBL_TYPE_TEXT;
						break;
					case FLM_NUMBER_TYPE:
						FieldType = LBL_TYPE_NUMBER;
						break;
					case FLM_BINARY_TYPE:
						FieldType = LBL_TYPE_BINARY;
						break;
					case FLM_CONTEXT_TYPE:
						FieldType = LBL_TYPE_CONTEXT;
						break;
					default:
						FieldType = LBL_TYPE_UNKNOWN;
						break;
				}
				if (!ViewAddMenuItem( LBL_FIELD_TYPE, LabelWidth,
							VAL_IS_LABEL_INDEX, (FLMUINT)FieldType, 0,
							FileNumber, FileOffset, 0, MOD_DISABLED,
							Col, Row++, 0, bc, fc, bc, fc))
						return( 0);

				/* Output the field level */

				if (!ViewAddMenuItem( LBL_FIELD_LEVEL, LabelWidth,
							VAL_IS_NUMBER | DISP_DECIMAL,
							(FLMUINT)StateInfo->uiFieldLevel, 0,
							FileNumber, FileOffset, 0, MOD_DISABLED,
							Col, Row++, 0, bc, fc, bc, fc))
					return( 0);

				if (StateInfo->uiEncId)
				{
					if (!ViewAddMenuItem( LBL_ENC_ID, LabelWidth,
								VAL_IS_NUMBER | DISP_DECIMAL,
								(FLMUINT)StateInfo->uiEncId, 0,
								FileNumber, FileOffset, 0, MOD_DISABLED,
								Col, Row++, 0, bc, fc, bc, fc))
						return( 0);
				}
			}

			/* Output the jump level for the jump FOP */

			if (StateInfo->uiFOPType == FLM_FOP_JUMP_LEVEL)
			{
				if (!ViewAddMenuItem( LBL_JUMP_LEVEL, LabelWidth,
						VAL_IS_NUMBER | DISP_DECIMAL,
						(FLMUINT)StateInfo->uiJumpLevel, 0,
						FileNumber, FileOffset, 0x07, MOD_BITS | MOD_DECIMAL,
						Col, Row++, 0, bc, fc, bc, fc))
					return( 0);
			}

			/* Output the field data length */

			if (eCorruptionCode != FLM_BAD_ELM_FLD_OVERHEAD)
			{
				if (!ViewAddMenuItem( LBL_FIELD_LENGTH, LabelWidth,
							VAL_IS_NUMBER | DISP_DECIMAL,
							StateInfo->uiFOPDataLen, 0,
							FileNumber, FileOffset, 0, MOD_DISABLED,
							Col, Row++, 0, bc, fc, bc, fc))
					return( 0);
			}
		}

		/* Verify the field if it is entirely contained right here */

		if ((eCorruptionCode == FLM_NO_CORRUPTION) &&
			(StateInfo->uiFOPDataLen == (StateInfo->uiEncId
										? StateInfo->uiEncFieldLen
										: StateInfo->uiFieldLen) &&
				(StateInfo->uiFOPDataLen > 0) &&
				(StateInfo->uiFOPType != FLM_FOP_CONT_DATA)))
		{
			if (StateInfo->uiFOPType == FLM_FOP_REC_INFO)
			{
				eCorruptionCode = FLM_NO_CORRUPTION;
			}
			else
			{
				eCorruptionCode = flmVerifyField( StateInfo, StateInfo->pFOPData,
													 StateInfo->uiFOPDataLen,
													 StateInfo->uiFieldType);
			}
			if (eCorruptionCode != FLM_NO_CORRUPTION)
			{
				SetStatusBit( FOPStatusBytes, eCorruptionCode);
				SetStatusBit( StatusBytes, eCorruptionCode);
			}
		}

		/* Output the field status */

		if ((!StatusOnlyFlag) && (eCorruptionCode != FLM_NO_CORRUPTION))
		{
			if (!OutputStatus( Col, &Row, bc, fc, LabelWidth,
												 LBL_FIELD_STATUS, FOPStatusBytes))
				return( 0);
		}

		/* Output the data.  If we had a bad overhead error, output */
		/* the rest of the data in the element. */

		if ((eCorruptionCode == FLM_BAD_ELM_FLD_OVERHEAD) ||
			 ((eCorruptionCode != FLM_NO_CORRUPTION) &&
			  (StateInfo->uiElmRecOffset == SaveElmRecOffset)))
		{
			*RowRV = Row;
			if (StatusOnlyFlag)
				return( 1);
			return( OutputHexValue( Col, RowRV, bc, fc,
							(FLMUINT)((SaveElmRecOffset == 0)
									 ? (FLMUINT)LBL_RECORD
									 : (FLMUINT)LBL_FIELD_DATA),
							FileNumber, FileOffset,
							&StateInfo->pElmRec [SaveElmRecOffset],
							(FLMUINT)(StateInfo->uiElmRecLen - SaveElmRecOffset),
							FALSE, MOD_BINARY));
		}
		else if ((!StatusOnlyFlag) && (StateInfo->uiFOPDataLen))
		{
			if (!OutputHexValue( Col, &Row, bc, fc, LBL_FIELD_DATA,
						FSGetFileNumber( StateInfo->uiBlkAddress),
						FSGetFileOffset( StateInfo->uiBlkAddress) +
						(FLMUINT)(StateInfo->pFOPData - StateInfo->pBlk),
						StateInfo->pFOPData,
						StateInfo->uiFOPDataLen, FALSE, MOD_BINARY))
				return( 0);
		}

		/* See if we have reached the end of the element - or quit */
		/* if the element record offset is not changing */

		if (StateInfo->uiElmRecOffset >= StateInfo->uiElmRecLen)
		{
			*RowRV = Row;
			return( 1);
		}
	}
}


/***************************************************************************
Desc: This routine reads into memory a database block.  It will also
		allocate memory to hold the block if necessary.  The block will
		also be decrypted if necessary.
*****************************************************************************/
FLMINT ViewBlkRead(
	FLMUINT			BlkAddress,
	FLMBYTE **		BlkPtrRV,
	FLMUINT			ReadLen,
	FLMUINT16 *		pui16CalcChkSum,
	FLMUINT16 *		pui16BlkChkSum,
	FLMUINT *		puiBytesReadRV,
	FLMBOOL			bShowPartialReadError,
	FLMBOOL *		pbIsEncBlock,
	FLMBOOL			bDecryptBlock,
	FLMBOOL *		pbEncrypted
	)
{
	RCODE       rc;
	FLMBYTE *   BlkPtr;
	char			szErrMsg [80];

	/* First allocate memory to read the block into */
	/* if not already allocated */

	if (!(*BlkPtrRV))
	{
		if (RC_BAD( rc = f_alloc( ReadLen, BlkPtrRV)))
		{
			ViewShowRCError( "allocating memory to read block", rc);
			return( 0);
		}
	}
	BlkPtr = *BlkPtrRV;

	/* Read the block into memory - if block address is zero, don't */
	/* read the first byte of the file. */

	if (BlkAddress == 0)
	{
		BlkAddress++;
		*BlkPtr++ = 0xFF;
		ReadLen--;
	}
	
	if (RC_BAD( rc = gv_pSFileHdl->readBlock( BlkAddress, ReadLen,
												BlkPtr, puiBytesReadRV)))
	{
		if (rc == FERR_IO_END_OF_FILE)
		{
			rc = FERR_OK;
			f_memset( &BlkPtr [*puiBytesReadRV], 0xEE, ReadLen - *puiBytesReadRV);
			if (bShowPartialReadError)
			{
				if (!(*puiBytesReadRV))
				{
					ViewShowRCError( "reading block", FERR_IO_END_OF_FILE);
					return( 0);
				}
				else if (*puiBytesReadRV < ReadLen)
				{
					f_sprintf( szErrMsg,
						"Only %u bytes of data were read (requested %u)",
							(unsigned)*puiBytesReadRV, (unsigned)ReadLen);
					ViewShowError( szErrMsg);
				}
			}
		}
		else
		{
			ViewShowRCError( "reading block", rc);
			return( 0);
		}
	}

	if (pui16BlkChkSum)
	{
		*pui16BlkChkSum = (FLMUINT16)(((FLMUINT)BlkPtr [BH_CHECKSUM_HIGH] << 8) + 
												 BlkPtr [BH_CHECKSUM_LOW]);
	}

	/* Decrypt the block if necessary */

	if (pui16CalcChkSum != NULL)
	{
		if (FB2UW( &BlkPtr [BH_BLK_END]) >
				gv_ViewHdrInfo.FileHdr.uiBlockSize)
		{
			*pui16CalcChkSum = 0;
		}
		else
		{
			FLMBYTE		ucChecksumHigh = BlkPtr [BH_CHECKSUM_HIGH];
			FLMBYTE		ucChecksumLow  = BlkPtr [BH_CHECKSUM_LOW];

			BlkCheckSum( BlkPtr, CHECKSUM_SET, BlkAddress,
											gv_ViewHdrInfo.FileHdr.uiBlockSize);

			*pui16CalcChkSum = (FLMUINT16)(((FLMUINT)BlkPtr [BH_CHECKSUM_HIGH] << 8) + 
												 BlkPtr [BH_CHECKSUM_LOW]);

			BlkPtr [BH_CHECKSUM_HIGH] = ucChecksumHigh;
			BlkPtr [BH_CHECKSUM_LOW] = ucChecksumLow;

			// 3x Format demands we write over the checksum value.

			BlkPtr [BH_CHECKSUM_LOW] = (FLMBYTE) BlkAddress;
		}
	}
	
	if (pbEncrypted || pbIsEncBlock || bDecryptBlock)
	{
		FLMUINT	uiBlkType = BH_GET_TYPE( BlkPtr);
		FLMUINT	uiLfNum = FB2UW( &BlkPtr [BH_LOG_FILE_NUM]);
		FLMUINT	uiBufLen = getEncryptSize( BlkPtr);
		FLMUINT	uiEncLen = uiBufLen - BH_OVHD;
		LFILE *	pLFile;
		IXD *		pIxd;
		FDB *		pDb;
		FFILE *	pFile;

		if (pbEncrypted)
		{
			*pbEncrypted = FALSE;
		}
		if (pbIsEncBlock)
		{
			*pbIsEncBlock = FALSE;
		}
		
		if (uiEncLen && uiLfNum &&
			 uiBlkType != BHT_FREE &&
			 uiBlkType != BHT_LFH_BLK &&
		 	 uiBlkType != BHT_PCODE_BLK)
		{
			ViewGetDictInfo();
			if (gv_bViewHaveDictInfo)
			{
				pDb = (FDB *)gv_hViewDb;
				pFile = pDb->pFile;
				if (RC_OK( fdictGetIndex( pDb->pDict, pFile->bInLimitedMode,
								uiLfNum, &pLFile, &pIxd, TRUE)) &&
					 pIxd && pIxd->uiEncId)
				{
					if (pbEncrypted)
					{
						*pbEncrypted = TRUE;
					}
					if (pbIsEncBlock)
					{
						*pbIsEncBlock = TRUE;
					}
#ifdef FLM_USE_NICI
					if (!pFile->bInLimitedMode && bDecryptBlock)
					{
						F_CCS *	pCcs = (F_CCS *)pFile->pDictList->pIttTbl[ pIxd->uiEncId].pvItem; 

						flmAssert( pCcs);
						if (RC_OK( pCcs->decryptFromStore( &BlkPtr [BH_OVHD],
												uiEncLen, &BlkPtr [BH_OVHD], &uiEncLen)))
						{
							if (pbEncrypted)
							{
								*pbEncrypted = FALSE;
							}
						}
					}
#endif
				}
			}
		}
	}

	return( 1);
}

/***************************************************************************
Name: ViewGetLFH
Desc: This routine searches through the LFH blocks searching for the
		LFH of a particular logical file.
*****************************************************************************/
FLMINT ViewGetLFH(
	FLMUINT     lfNum,
	FLMBYTE *   lfhRV,
	FLMUINT  *	FileOffset
	)
{
	FLMUINT     BlkAddress;
	FLMUINT     BlkCount = 0;
	FLMBYTE *   BlkPtr = NULL;
	FLMUINT     EndOfBlock;
	FLMUINT     Pos;
	FLMUINT		uiBytesRead;
	FLMUINT     GotLFH = 0;

	/* Read the LFH blocks and get the information needed */
	/* If we read too many, the file is probably corrupt. */

	*FileOffset = 0;
	BlkAddress = gv_ViewHdrInfo.FileHdr.uiFirstLFHBlkAddr;
	while( BlkAddress != BT_END)
	{
		if ((!ViewBlkRead( BlkAddress, &BlkPtr,
											gv_ViewHdrInfo.FileHdr.uiBlockSize,
											NULL, NULL, &uiBytesRead, FALSE,
											NULL, FALSE, NULL)) ||
			 (!uiBytesRead))
			break;

		/* Count the blocks read to prevent too many from being read. */
		/* We don't want to get into an infinite loop if the database */
		/* is corrupted. */

		BlkCount++;

		/* Search through the block for the particular LFH which matches */
		/* the one we are looking for. */

		if (uiBytesRead <= BH_OVHD)
			EndOfBlock = BH_OVHD;
		else
		{
			EndOfBlock = FB2UW( &BlkPtr [BH_BLK_END]);
			if (EndOfBlock > gv_ViewHdrInfo.FileHdr.uiBlockSize)
				EndOfBlock = gv_ViewHdrInfo.FileHdr.uiBlockSize;
			if (EndOfBlock > uiBytesRead)
				EndOfBlock = uiBytesRead;
		}
		Pos = BH_OVHD;
		while( Pos < EndOfBlock)
		{
			FLMUINT		uiLfType;
			FLMUINT		uiLfNum;

			/* See if we got the one we wanted */

			uiLfType = BlkPtr [Pos + LFH_TYPE_OFFSET];
			uiLfNum = FB2UW( &BlkPtr [Pos + LFH_LF_NUMBER_OFFSET]);

			if ((uiLfType != LF_INVALID) && (lfNum == uiLfNum))
			{
				f_memcpy( lfhRV, &BlkPtr [Pos], LFH_SIZE);
				GotLFH = 1;
				*FileOffset = BlkAddress + Pos;
				break;
			}

			Pos += LFH_SIZE;
		}

		/* If we didn't end right on end of block, return */

		if ((Pos != EndOfBlock) || (GotLFH))
			break;

		/* If we have traversed too many blocks, things are probably corrupt */

		if (BlkCount > 100)
			break;
		if (BH_NEXT_BLK + 4 <= uiBytesRead)
			BlkAddress = FB2UD( &BlkPtr [BH_NEXT_BLK]);
		else
			BlkAddress = BT_END;
	}

	/* Be sure to free the block handle -- if one was allocated */

	f_free( &BlkPtr);
	return( GotLFH);
}

/***************************************************************************
Name: ViewGetLFName
Desc: This routine attempts to find an LFH for a particular logical
		file.  It then extracts the name from the LFH.
*****************************************************************************/
FLMINT ViewGetLFName(
	FLMBYTE *      lfName,
	FLMUINT        lfNum,
	FLMBYTE *      LFH,
	FLMUINT  *		FileOffset)
{
	FLMBYTE   TempBuf [40];

	if ((lfNum == 0) || (!ViewGetLFH( lfNum, LFH, FileOffset)))
	{
		*FileOffset = 0;
		f_sprintf( (char *)TempBuf, "lfNum=%u", (unsigned)lfNum);
		f_strcpy( (char *)lfName, (const char *)TempBuf);
		return( 0);
	}
	else
	{
		TempBuf [0] = 0;
		f_sprintf( (char *)(&TempBuf [f_strlen( (const char *)TempBuf)]), ", lfNum=%u", (unsigned)lfNum);
		f_strcpy( (char *)lfName, (const char *)TempBuf);
		return( 1);
	}
}

/***************************************************************************
Name: OutBlkHdrExpNum
Desc: This routine outputs one of the number fields in the block
		header.  It checks the number against an expected value, and if
		the number does not match the expected value also outputs the
		value which was expected.  This routine is used to output values
		in the block header where we are expected certain values.
*****************************************************************************/
FSTATIC FLMINT OutBlkHdrExpNum(
	FLMUINT		Col,
	FLMUINT  *	RowRV,
	eColorType	bc,
	eColorType	fc,
	eColorType	mbc,
	eColorType	mfc,
	eColorType	sbc,
	eColorType	sfc,
	FLMUINT     LabelWidth,
	FLMINT      iLabelIndex,
	FLMUINT		FileNumber,
	FLMUINT     FileOffset,
	FLMBYTE *   ValuePtr,
	FLMUINT     ValueType,
	FLMUINT     ModType,
	FLMUINT     ExpNum,
	FLMUINT     IgnoreExpNum,
	FLMUINT     Option)
{
	FLMUINT     Row = *RowRV;
	FLMUINT		Num = 0;

	if (!Option)
	{
		mbc = sbc = bc;
		mfc = sfc = fc;
	}
	switch( ModType & 0x0F)
	{
		case MOD_FLMUINT:
			Num = FB2UD( ValuePtr);
			break;
		case MOD_FLMUINT16:
			Num = FB2UW( ValuePtr);
			break;
		case MOD_FLMBYTE:
			Num = *ValuePtr;
			break;
		case MOD_BH_ADDR:
			Num = FB2UD( ValuePtr );/* & 0xFFFFFF00;*/  /* Could use GET_BH_ADDR() */
																/* But pass in offset */
			break;   
	}
	if (!ViewAddMenuItem( iLabelIndex, LabelWidth,
				ValueType, Num, 0,
				FileNumber, FileOffset, 0, ModType,
				Col, Row++, Option, mbc, mfc, sbc, sfc))
		return( 0);

	if ((ExpNum != IgnoreExpNum) && (Num != ExpNum))
	{
		if (!ViewAddMenuItem( LBL_EXPECTED, 0,
					ValueType, ExpNum, 0,
					0, VIEW_INVALID_FILE_OFFSET, 0, MOD_DISABLED,
					Col + LabelWidth + 1, Row++, 0,
					FLM_RED, FLM_LIGHTGRAY,
					FLM_RED, FLM_LIGHTGRAY))
			return( 0);
	}
	*RowRV = Row;
	return( 1);
}

/***************************************************************************
Name:    FormatBlkType
Desc:    This routine formats a block's type into ASCII.
*****************************************************************************/
FSTATIC void FormatBlkType(
	FLMBYTE *   TempBuf,
	FLMUINT		BlkType
	)
{
	FLMBYTE   TempBuf1 [30];

	switch( BlkType)
	{
		case BHT_FREE:
			f_strcpy( (char *)TempBuf, "Free");
			break;
		case BHT_LEAF:
			f_strcpy( (char *)TempBuf, "Leaf");
			break;
		case BHT_NON_LEAF:
			f_strcpy( (char *)TempBuf, "Non-Leaf 3x");
			break;
		case BHT_NON_LEAF_DATA:
			f_strcpy( (char *)TempBuf, "Non-Leaf Data");
			break;
		case BHT_NON_LEAF_COUNTS:
			f_strcpy( (char *)TempBuf, "Non-Leaf /w Counts");
			break;
		case BHT_LFH_BLK:
			f_strcpy( (char *)TempBuf, "LFH");
			break;
		case BHT_PCODE_BLK:
			f_strcpy( (char *)TempBuf, "PCODE");
			break;
		default:
			f_sprintf( (char *)TempBuf1, "Unknown Type: %u", (unsigned)BlkType);
			f_strcpy( (char *)TempBuf, (const char *)TempBuf1);
			break;
	}
}

/***************************************************************************
Name:    ViewOutBlkHdr
Desc:    This routine outputs a block's header.
*****************************************************************************/
FLMINT ViewOutBlkHdr(
	FLMUINT     Col,
	FLMUINT  *	RowRV,
	FLMBYTE *   BlkPtr,
	BLK_EXP_p   BlkExp,
	FLMBYTE *   BlkStatus,
	FLMUINT16   ui16CalcChkSum,
	FLMUINT16	ui16BlkChkSum
	)
{
	FLMUINT     LabelWidth = 35;
	FLMUINT     Row = *RowRV;
	FLMBYTE		TempBuf [80];
	FLMUINT		BlkAddress;
	FLMUINT		EndOfBlock;
	FLMUINT		BytesUsed;
	FLMUINT		PercentFull;
	FLMUINT     Option;
	eColorType	bc = FLM_BLACK;
	eColorType	fc = FLM_LIGHTGRAY;
	eColorType	mbc = FLM_BLACK;
	eColorType	mfc = FLM_WHITE;
	eColorType	sbc = FLM_BLUE;
	eColorType	sfc = FLM_WHITE;
	FLMBYTE		lfLFH [LFH_SIZE];
	FLMBYTE		lfName [80];
	FLMUINT		lfNum;
	FLMUINT		lfType;
	FLMUINT		TempFileOffset;
	FLMBYTE		bySaveChar;

	/* Output the block Header address */

	if (!OutBlkHdrExpNum( Col, &Row, FLM_RED, FLM_LIGHTGRAY,
			FLM_RED, FLM_WHITE, sbc, sfc,
			LabelWidth, LBL_BLOCK_ADDRESS_BLOCK_HEADER,
			FSGetFileNumber( BlkExp->BlkAddr), 
			FSGetFileOffset( BlkExp->BlkAddr), &BlkPtr [BH_ADDR],
			VAL_IS_NUMBER | DISP_HEX_DECIMAL,
			MOD_BH_ADDR | MOD_HEX,
			BlkExp->BlkAddr, 0, 0))
		return( 0);

	/* Adjust column so rest of header is indented */

	Col += 2;
	LabelWidth -= 2;

	/* Output the previous block address */

	if (FB2UD( &BlkPtr [BH_PREV_BLK]) == BT_END)
		Option = 0;
	else
		Option = PREV_BLOCK_OPTION;
	if (!OutBlkHdrExpNum( Col, &Row, bc, fc, mbc, mfc, sbc, sfc,
				LabelWidth, LBL_PREVIOUS_BLOCK_ADDRESS,
				FSGetFileNumber( BlkExp->BlkAddr), 
				FSGetFileOffset( BlkExp->BlkAddr) + BH_PREV_BLK,
				&BlkPtr [BH_PREV_BLK],
				VAL_IS_NUMBER | DISP_HEX_DECIMAL,
				MOD_FLMUINT | MOD_HEX,
				BlkExp->PrevAddr, 0, Option))
		return( 0);

	/* Output the next block address */

	if (FB2UD( &BlkPtr [BH_NEXT_BLK]) == BT_END)
		Option = 0;
	else
		Option = NEXT_BLOCK_OPTION;
	if (!OutBlkHdrExpNum( Col, &Row, bc, fc, mbc, mfc, sbc, sfc,
				LabelWidth, LBL_NEXT_BLOCK_ADDRESS,
				FSGetFileNumber( BlkExp->BlkAddr), 
				FSGetFileOffset( BlkExp->BlkAddr) + BH_NEXT_BLK,
				&BlkPtr [BH_NEXT_BLK],
				VAL_IS_NUMBER | DISP_HEX_DECIMAL,
				MOD_FLMUINT | MOD_HEX,
				BlkExp->NextAddr, 0, Option))
		return( 0);

	/* Output the logical file this block belongs to - if any */

	lfNum = FB2UW( &BlkPtr [BH_LOG_FILE_NUM]);
	if (!ViewGetLFName( lfName, lfNum, lfLFH, &TempFileOffset))
	{
		lfType = LF_INVALID;
		Option = 0;
	}
	else
	{
		lfType = lfLFH [LFH_TYPE_OFFSET];
		Option = LOGICAL_FILE_OPTION | lfNum;
	}
	if ((BlkExp->LfNum != 0) && (lfNum != BlkExp->LfNum))
		f_sprintf( (char *)TempBuf, "%s (Expected %u)", lfName, (unsigned)BlkExp->LfNum);
	else
		f_strcpy( (char *)TempBuf, (const char *)lfName);
	if (!ViewAddMenuItem( LBL_BLOCK_LOGICAL_FILE_NAME, LabelWidth,
				VAL_IS_TEXT_PTR,
				(FLMUINT)((FLMBYTE *)(&TempBuf [0])), 0,
				FSGetFileNumber( BlkExp->BlkAddr),
				FSGetFileOffset( BlkExp->BlkAddr) + BH_LOG_FILE_NUM, 0,
				MOD_FLMUINT | MOD_DECIMAL,
				Col, Row++, Option,
				!Option ? bc : mbc,
				!Option ? fc : mfc,
				!Option ? bc : sbc,
				!Option ? fc : sfc))
		return( 0);

	/* Output the logical file type */

	if (lfType != LF_INVALID)
	{
		FormatLFType( TempBuf, lfType);
		if (!ViewAddMenuItem( LBL_BLOCK_LOGICAL_FILE_TYPE, LabelWidth,
					VAL_IS_TEXT_PTR,
					(FLMUINT)((FLMBYTE *)(&TempBuf [0])), 0,
					0, VIEW_INVALID_FILE_OFFSET, 0, MOD_DISABLED,
					Col, Row++, 0, bc, fc, bc, fc))
			return( 0);
	}

	/* Output the block type */

	FormatBlkType( TempBuf, BH_GET_TYPE( BlkPtr));
	if (BH_IS_ROOT_BLK( BlkPtr))
		f_strcpy( (char *)&TempBuf [f_strlen( (const char *)TempBuf)], " (Root)");
	if (BH_GET_TYPE( BlkPtr) != (FLMBYTE) BlkExp->Type)
	{
		f_strcpy( (char *)&TempBuf [f_strlen( (const char *)TempBuf)], ", Expecting ");
		FormatBlkType( &TempBuf [f_strlen( (const char *)TempBuf)], BlkExp->Type);
	}
	if (!ViewAddMenuItem( LBL_BLOCK_TYPE, LabelWidth,
			VAL_IS_TEXT_PTR,
			(FLMUINT)((FLMBYTE *)(&TempBuf [0])), 0,
			FSGetFileNumber( BlkExp->BlkAddr),
			FSGetFileOffset( BlkExp->BlkAddr) + BH_TYPE, 0,
			MOD_FLMBYTE | MOD_HEX,
			Col, Row++, 0, bc, fc, bc, fc))
		return( 0);

	/* Output the level in the B-TREE */

	if (!OutBlkHdrExpNum( Col, &Row, bc, fc, mbc, mfc, sbc, sfc,
			LabelWidth, LBL_B_TREE_LEVEL,
			FSGetFileNumber( BlkExp->BlkAddr), 
			FSGetFileOffset( BlkExp->BlkAddr) + BH_LEVEL,
			&BlkPtr [BH_LEVEL],
			VAL_IS_NUMBER | DISP_DECIMAL,
			MOD_FLMBYTE | MOD_DECIMAL,
			(FLMUINT)BlkExp->Level, (FLMUINT)0xFF, 0))
		return( 0);

	/* Output the end of the block */

	EndOfBlock = FB2UW( &BlkPtr [BH_BLK_END]);
	if (!ViewAddMenuItem( LBL_BLOCK_END, LabelWidth,
				VAL_IS_NUMBER | DISP_DECIMAL,
				(FLMUINT)EndOfBlock, 0,
				FSGetFileNumber( BlkExp->BlkAddr),
				FSGetFileOffset( BlkExp->BlkAddr) + BH_BLK_END, 0,
				MOD_FLMUINT | MOD_DECIMAL,
				Col, Row++, 0, bc, fc, bc, fc))
		return( 0);

	/* Output the percent full */

	BytesUsed = EndOfBlock - BH_OVHD;
	if ((!BytesUsed) || (EndOfBlock < BH_OVHD))
		PercentFull = 0;
	else if (EndOfBlock > gv_ViewHdrInfo.FileHdr.uiBlockSize)
		PercentFull = 100;
	else
		PercentFull = ((FLMUINT)(BytesUsed) * (FLMUINT)(100)) /
				 (FLMUINT)(gv_ViewHdrInfo.FileHdr.uiBlockSize - BH_OVHD);
	if (!ViewAddMenuItem( LBL_PERCENT_FULL, LabelWidth,
			VAL_IS_NUMBER | DISP_DECIMAL,
			(FLMUINT)PercentFull, 0,
			0, VIEW_INVALID_FILE_OFFSET, 0, MOD_DISABLED,
			Col, Row++, 0, bc, fc, bc, fc))
		return( 0);

	/* Output the block transaction ID */

	if (!ViewAddMenuItem( LBL_BLOCK_TRANS_ID, LabelWidth,
				VAL_IS_NUMBER | DISP_DECIMAL,
				FB2UD( &BlkPtr [BH_TRANS_ID]), 0,
				FSGetFileNumber( BlkExp->BlkAddr),
				FSGetFileOffset( BlkExp->BlkAddr) + BH_TRANS_ID, 0,
				MOD_FLMUINT | MOD_DECIMAL,
				Col, Row++, 0, bc, fc, bc, fc))
		return( 0);

	/* Output the encryption flag */

	if (!ViewAddMenuItem( LBL_BLOCK_ENCRYPTED, LabelWidth,
			VAL_IS_LABEL_INDEX,
			(BlkPtr [BH_ENCRYPTED])
			? (FLMUINT)LBL_YES
			: (FLMUINT)LBL_NO, 0,
			FSGetFileNumber( BlkExp->BlkAddr),
			FSGetFileOffset( BlkExp->BlkAddr) + BH_ENCRYPTED, 0, 
			MOD_DISABLED | MOD_FLMBYTE,
			Col, Row++, 0, bc, fc, bc, fc))
		return( 0);

	/* Output the old block image address */

	BlkAddress = FB2UD( &BlkPtr [BH_PREV_BLK_ADDR]);
	if ((BlkAddress == BT_END) || (BlkAddress == 0) ||
			(FB2UD( &BlkPtr [BH_TRANS_ID]) <=
				gv_ViewHdrInfo.LogHdr.uiCurrTransID))
		Option = 0;
	else
		Option = PREV_BLOCK_IMAGE_OPTION;
	if (!ViewAddMenuItem( LBL_OLD_BLOCK_IMAGE_ADDRESS, LabelWidth,
				VAL_IS_NUMBER | DISP_HEX_DECIMAL,
				BlkAddress, 0,
				FSGetFileNumber( BlkExp->BlkAddr),
				FSGetFileOffset( BlkExp->BlkAddr) + BH_PREV_BLK_ADDR, 0,
				MOD_FLMUINT | MOD_HEX,
				Col, Row++, Option,
				!Option ? bc : mbc,
				!Option ? fc : mfc,
				!Option ? bc : sbc,
				!Option ? fc : sfc))
		return( 0);

	/* Output the old block image transaction ID */

	if (!ViewAddMenuItem( LBL_OLD_BLOCK_IMAGE_TRANS_ID, LabelWidth,
				VAL_IS_NUMBER | DISP_DECIMAL,
				FB2UD( &BlkPtr [BH_PREV_TRANS_ID]), 0,
				FSGetFileNumber( BlkExp->BlkAddr),
				FSGetFileOffset( BlkExp->BlkAddr) + BH_PREV_TRANS_ID, 0,
				MOD_FLMUINT | MOD_DECIMAL,
				Col, Row++, 0, bc, fc, bc, fc))
		return( 0);

	/* Output the low byte of the block checksum */

	bySaveChar = (FLMBYTE)(ui16BlkChkSum & 0x00FF);
	if (!OutBlkHdrExpNum( Col, &Row, bc, fc, mbc, mfc, sbc, sfc,
			LabelWidth, LBL_BLOCK_CHECKSUM_LOW,
			FSGetFileNumber( BlkExp->BlkAddr), 
			FSGetFileOffset( BlkExp->BlkAddr) + BH_CHECKSUM_LOW,
			&bySaveChar,
			VAL_IS_NUMBER | DISP_DECIMAL,
			MOD_FLMBYTE | MOD_DECIMAL,
			(FLMUINT)((ui16BlkChkSum)
						 ? (FLMUINT)(ui16CalcChkSum & 0x00FF)
						 : (FLMUINT)0), 0, 0))
	{
		BlkPtr [BH_CHECKSUM_LOW] = bySaveChar;
		return( 0);
	}

	/* Output the high byte of the block checksum */

	bySaveChar = (FLMBYTE)(ui16BlkChkSum >> 8);
	if (!OutBlkHdrExpNum( Col, &Row, bc, fc, mbc, mfc, sbc, sfc,
			LabelWidth, LBL_BLOCK_CHECKSUM_HIGH,
			FSGetFileNumber( BlkExp->BlkAddr), 
			FSGetFileOffset( BlkExp->BlkAddr) + BH_CHECKSUM_HIGH,
			&bySaveChar,
			VAL_IS_NUMBER | DISP_DECIMAL,
			MOD_FLMBYTE | MOD_DECIMAL,
			(FLMUINT)((ui16BlkChkSum)
						 ? (FLMUINT)(ui16CalcChkSum >> 8)
						 : (FLMUINT)0), 0, 0))
		return( 0);

	/* Output the flags which indicate the state of the block */

	if (!OutputStatus( Col, &Row, bc, fc, LabelWidth, LBL_BLOCK_STATUS,
										 BlkStatus))
		return( 0);

	*RowRV = Row + 1;
	return( 1);
}

/***************************************************************************
Name: ViewAvailBlk
Desc: This routine displays a block in the AVAIL list.
*****************************************************************************/
FLMINT ViewAvailBlk(
	FLMUINT		ReadAddress,
	FLMUINT		BlkAddress,
	FLMBYTE **	BlkPtrRV,
	BLK_EXP_p	BlkExp
	)
{
	FLMINT				iRc = 0;
	FLMUINT				Row;
	FLMUINT				Col;
	FLMBYTE *			BlkPtr;
	FLMBYTE				BlkStatus [NUM_STATUS_BYTES];
	STATE_INFO			StateInfo;
	FLMBOOL				bStateInitialized = FALSE;
	BLOCK_INFO			BlockInfo;
	eCorruptionType	eCorruptionCode;
	FLMUINT16			ui16CalcChkSum;
	FLMUINT16			ui16BlkChkSum;
	FLMUINT				uiBytesRead;

	/* Read the block into memory */

	if (!ViewBlkRead( ReadAddress, BlkPtrRV,
										gv_ViewHdrInfo.FileHdr.uiBlockSize,
										&ui16CalcChkSum, &ui16BlkChkSum, &uiBytesRead, TRUE,
										NULL, FALSE, NULL))
	{
		goto Exit;
	}
	BlkPtr = *BlkPtrRV;

	if (!ViewMenuInit( "AVAIL Block"))
	{
		goto Exit;
	}

	/* Output the block header first */

	Row = 0;
	Col = 5;
	BlkExp->Type = BHT_FREE;
	BlkExp->LfNum = 0;
	BlkExp->BlkAddr = BlkAddress;
	BlkExp->Level = 0xFF;

	/* Setup the STATE variable for processing through the block */

	InitStatusBits( BlkStatus);
	flmInitReadState( &StateInfo, &bStateInitialized,
							gv_ViewHdrInfo.FileHdr.uiVersionNum,
							(gv_bViewDbInitialized)
							? (FDB *)gv_hViewDb
							: (FDB *)NULL, NULL, 0, BHT_FREE, NULL);
	StateInfo.uiBlkAddress = BlkAddress;
	StateInfo.pBlk = BlkPtr;

	if ((eCorruptionCode = flmVerifyBlockHeader( &StateInfo, &BlockInfo,
											 gv_ViewHdrInfo.FileHdr.uiBlockSize,
											 BlkExp->NextAddr,
											 0, (FLMBOOL)(StateInfo.pDb != NULL
															  ? TRUE
															  : FALSE), TRUE)) != FLM_NO_CORRUPTION)
		SetStatusBit( BlkStatus, eCorruptionCode);

	if (!ViewOutBlkHdr( Col, &Row, BlkPtr, BlkExp, BlkStatus,
								ui16CalcChkSum, ui16BlkChkSum))
	{
		goto Exit;
	}
	iRc = 1;
Exit:
	if (bStateInitialized && StateInfo.pRecord)
	{
		StateInfo.pRecord->Release();
		StateInfo.pRecord = NULL;
	}
	return( iRc);
}

/***************************************************************************
Name: OutputHexValue
Desc: This routine outputs a stream of FLMBYTEs in hex format.  This
		routine is used to output key values and records within an
		element.
*****************************************************************************/
FSTATIC FLMINT OutputHexValue(
	FLMUINT     Col,
	FLMUINT  *	RowRV,
	eColorType	bc,
	eColorType	fc,
	FLMINT      iLabelIndex,
	FLMUINT		FileNumber,
	FLMUINT     FileOffset,
	FLMBYTE *   ValPtr,
	FLMUINT     ValLen,
	FLMUINT     CopyVal,
	FLMUINT		uiModFlag)
{
	FLMUINT     Row = *RowRV;
	FLMUINT     BytesPerLine = MAX_HORIZ_SIZE( Col + 3);
	FLMUINT		BytesProcessed = 0;
	FLMUINT		NumBytes;
	
	if (!ValLen)
	{
		return( 1);
	}

	if (!ViewAddMenuItem( iLabelIndex, 0,
								VAL_IS_EMPTY, 0, 0,
								FileNumber, FileOffset, 0, MOD_DISABLED,
								Col, Row++, 0, bc, fc, bc, fc))
		return( 0);
	Col += 2;
	while( BytesProcessed < ValLen)
	{
		if ((NumBytes = ValLen - BytesProcessed) > BytesPerLine)
			NumBytes = BytesPerLine;

		/* Output the line */

		if (!ViewAddMenuItem( -1, 0,
					(FLMBYTE)((CopyVal)
								? (FLMBYTE)VAL_IS_BINARY
								: (FLMBYTE)VAL_IS_BINARY_PTR),
					(FLMUINT)ValPtr, NumBytes,
					FileNumber, FileOffset, NumBytes, uiModFlag,
					Col, Row++, 0, bc, fc, bc, fc))
			return( 0);
		FileOffset += (FLMUINT)NumBytes;
		BytesProcessed += NumBytes;
		ValPtr += NumBytes;
	}
	*RowRV = Row;
	return( 1);
}

/***************************************************************************
Name:    OutputLeafElements
Desc:    This routine outputs the elements in a LEAF block.
*****************************************************************************/
FSTATIC FLMINT OutputLeafElements(
	FLMUINT     Col,
	FLMUINT  *  RowRV,
	FLMBYTE *   BlkPtr,
	BLK_EXP_p   BlkExp,
	FLMBYTE *   BlkStatusRV,
	FLMUINT     StatusOnlyFlag,
	FLMBOOL		bEncrypted
	)
{
	FLMINT				iRc = 0;
	FLMUINT				LabelWidth = 30;
	eColorType			bc = FLM_BLACK;
	eColorType			fc = FLM_LIGHTGRAY;
	FLMUINT				Row = *RowRV;
	FLMUINT				ElementCount = 0;
	eCorruptionType	eCorruptionCode;
	FLMUINT				LfType = LF_INVALID;
	FLMBYTE				ElmStatus [NUM_STATUS_BYTES];
	STATE_INFO			StateInfo;
	FLMBOOL				bStateInitialized = FALSE;
	BLOCK_INFO			BlockInfo;
	FLMBYTE				KeyBuffer [MAX_KEY_SIZ];
	LFILE *				pLFile = NULL;
	LF_HDR				LogicalFile;
	LF_STATS				LfStats;
	LF_HDR *				pLogicalFile = NULL;

	if (BlkExp->LfNum == 0)
		BlkExp->LfNum = FB2UW( &BlkPtr [BH_LOG_FILE_NUM]);

	/* Setup the STATE variable for processing through the block */

	ViewGetDictInfo();
	if (gv_bViewHaveDictInfo)
	{
		if ((RC_OK( fdictGetIndex(
				((FDB *)gv_hViewDb)->pDict,
				((FDB *)gv_hViewDb)->pFile->bInLimitedMode,
				BlkExp->LfNum, &pLFile, NULL))) ||
			 (RC_OK( fdictGetContainer(
			 	((FDB *)gv_hViewDb)->pDict,
				BlkExp->LfNum, &pLFile))))
		{
			f_memset( &LogicalFile, 0, sizeof( LF_HDR));
			f_memset( &LfStats, 0, sizeof( LF_STATS));
			pLogicalFile = &LogicalFile;
			pLogicalFile->pLfStats = &LfStats;
			LogicalFile.pLFile = pLFile;
			if (pLFile->uiLfType == LF_INDEX)
			{
				if (RC_BAD( fdictGetIndex(
					((FDB *)gv_hViewDb)->pDict,
					((FDB *)gv_hViewDb)->pFile->bInLimitedMode,
					 pLFile->uiLfNum, &LogicalFile.pLFile,
					 &LogicalFile.pIxd)))
				{
					pLogicalFile = NULL;
				}
				LogicalFile.pIfd = LogicalFile.pIxd->pFirstIfd;
			}
		}
	}
	LfType = (pLogicalFile) ? pLogicalFile->pLFile->uiLfType : LF_INVALID;

	flmInitReadState( &StateInfo, &bStateInitialized,
							gv_ViewHdrInfo.FileHdr.uiVersionNum,
							(gv_bViewDbInitialized)
							? (FDB *)gv_hViewDb
							: (FDB *)NULL, pLogicalFile, 0,
							BHT_LEAF, KeyBuffer);
	StateInfo.uiBlkAddress = BlkExp->BlkAddr;
	StateInfo.pBlk = BlkPtr;

	if ((eCorruptionCode = flmVerifyBlockHeader( &StateInfo, &BlockInfo,
								gv_ViewHdrInfo.FileHdr.uiBlockSize,
														BlkExp->NextAddr,
														BlkExp->PrevAddr,
														(FLMBOOL)(StateInfo.pDb != NULL
																	 ? TRUE
																	 : FALSE), TRUE)) != FLM_NO_CORRUPTION)
		SetStatusBit( BlkStatusRV, eCorruptionCode);

	if (bEncrypted)
	{
		if (!StatusOnlyFlag)
		{
			FLMUINT	uiEncSize = getEncryptSize( StateInfo.pBlk) - BH_OVHD;
			
			if (uiEncSize)
			{
			
				// Output the rest of the block as HEX - cannot be parsed through
				// when it is encrypted
		
				OutputHexValue( Col, &Row, bc, fc, LBL_ENC_DATA,
						FSGetFileNumber(StateInfo.uiBlkAddress),
						FSGetFileOffset(StateInfo.uiBlkAddress) + BH_OVHD,
						&StateInfo.pBlk [BH_OVHD], uiEncSize, FALSE, MOD_BINARY_ENC);
			}
		}
		iRc = 1;
		goto Exit;
	}
	
	/* Read through the elements in the block */

	while( StateInfo.uiElmOffset < StateInfo.uiEndOfBlock)
	{
		ElementCount++;
		InitStatusBits( ElmStatus);

		if ((eCorruptionCode = flmVerifyElement( &StateInfo, FLM_CHK_FIELDS)) != FLM_NO_CORRUPTION)
			SetStatusBit( ElmStatus, eCorruptionCode);
		else if (LfType == LF_INDEX)
		{
			if (StateInfo.uiCurKeyLen)
			{
				if( RC_BAD( flmVerifyIXRefs( &StateInfo, NULL, 0,
					&eCorruptionCode)) || eCorruptionCode != FLM_NO_CORRUPTION)
				{
					SetStatusBit( ElmStatus, eCorruptionCode);
				}
			}
		}

		/* Output the element */

		if (!StatusOnlyFlag)
		{
			Row++;

			/* Output the element number */

			if (!ViewAddMenuItem( LBL_ELEMENT_NUMBER, LabelWidth,
					VAL_IS_NUMBER | DISP_DECIMAL,
					(FLMUINT)ElementCount, 0,
					FSGetFileNumber( StateInfo.uiBlkAddress),
					FSGetFileOffset( StateInfo.uiBlkAddress) + StateInfo.uiElmOffset,
					0, MOD_DISABLED,
					Col, Row++, 0,
					FLM_GREEN, FLM_WHITE,
					FLM_GREEN, FLM_WHITE))
				goto Exit;

			/* Remember this item if we are searching */

			if ((gv_bViewSearching) &&
					(flmCompareKeys( StateInfo.pCurKey, StateInfo.uiCurKeyLen,
													gv_ucViewSearchKey,
													gv_uiViewSearchKeyLen) >= 0) &&
					(gv_pViewSearchItem == NULL))
			{
				gv_pViewSearchItem = gv_pViewMenuLastItem;
			}

			/* Output the element offset within the block */

			if (!ViewAddMenuItem( LBL_ELEMENT_OFFSET, LabelWidth,
									VAL_IS_NUMBER | DISP_DECIMAL,
									(FLMUINT)StateInfo.uiElmOffset, 0,
									FSGetFileNumber( StateInfo.uiBlkAddress),
									FSGetFileOffset( StateInfo.uiBlkAddress) +
										StateInfo.uiElmOffset,
									0, MOD_DISABLED,
									Col, Row++, 0, bc, fc, bc, fc))
				goto Exit;

			/* Output the element length */

			if (!ViewAddMenuItem( LBL_ELEMENT_LENGTH, LabelWidth,
									VAL_IS_NUMBER | DISP_DECIMAL,
									(FLMUINT)StateInfo.uiElmLen, 0,
									FSGetFileNumber( StateInfo.uiBlkAddress),
									FSGetFileOffset( StateInfo.uiBlkAddress) + 
										StateInfo.uiElmOffset,
									0, MOD_DISABLED,
									Col, Row++, 0, bc, fc, bc, fc))
				goto Exit;

			/* Display the first element flag */

			if (!ViewAddMenuItem( LBL_FIRST_ELEMENT_FLAG, LabelWidth,
					VAL_IS_LABEL_INDEX,
					(BBE_IS_FIRST( StateInfo.pElm))
					? (FLMUINT)LBL_YES
					: (FLMUINT)LBL_NO, 0,
					FSGetFileNumber( StateInfo.uiBlkAddress),
					FSGetFileOffset( StateInfo.uiBlkAddress) + StateInfo.uiElmOffset, 
					0x80, MOD_BITS | MOD_DECIMAL,
					Col, Row++, 0, bc, fc, bc, fc))
				goto Exit;

			/* Display the last element flag */

			if (!ViewAddMenuItem( LBL_LAST_ELEMENT_FLAG, LabelWidth,
						VAL_IS_LABEL_INDEX,
						(BBE_IS_LAST( StateInfo.pElm))
						? (FLMUINT)LBL_YES
						: (FLMUINT)LBL_NO, 0,
						FSGetFileNumber( StateInfo.uiBlkAddress),
						FSGetFileOffset( StateInfo.uiBlkAddress) + 
								StateInfo.uiElmOffset, 0x40,
						MOD_BITS | MOD_DECIMAL,
						Col, Row++, 0, bc, fc, bc, fc))
				goto Exit;

			if (eCorruptionCode != FLM_NO_CORRUPTION)
			{
				if ((LfType != LF_INDEX) &&
					 (LfType != LF_INVALID) &&
					 (StateInfo.uiCurKeyLen) &&
					 (StateInfo.uiElmDrn != DRN_LAST_MARKER))
				{
					FLMUINT        r = 0;
					STATE_INFO  DummyState;
					FLMBYTE      TempKeyBuffer [MAX_KEY_SIZ];

					f_memcpy( &DummyState, &StateInfo, sizeof( STATE_INFO));
					f_memcpy( TempKeyBuffer, KeyBuffer, MAX_KEY_SIZ);
					DummyState.pCurKey = &TempKeyBuffer [0];
					if (!OutputElmRecord( Col, &r, bc, fc, LabelWidth,
											&DummyState, ElmStatus, TRUE))
						goto Exit;
				}
				if (!OutputStatus( Col, &Row, bc, fc, LabelWidth,
													 LBL_ELEMENT_STATUS, ElmStatus))
					goto Exit;
			}

			/* Display the Previous Key Count */

			if (!ViewAddMenuItem( LBL_PREVIOUS_KEY_CONT_LEN, LabelWidth,
						VAL_IS_NUMBER | DISP_DECIMAL,
						(FLMUINT)StateInfo.uiElmPKCLen, 0,
						FSGetFileNumber( StateInfo.uiBlkAddress),
						FSGetFileOffset( StateInfo.uiBlkAddress) + StateInfo.uiElmOffset, 
						0x0F, MOD_BITS | MOD_DECIMAL,
						Col, Row++, 0, bc, fc, bc, fc))
				goto Exit;

			/* Output the previous key portion, if any */

			if (!OutputHexValue( Col, &Row, bc, fc,
					LBL_PREV_ELEMENT_KEY,
					0, VIEW_INVALID_FILE_OFFSET,
					StateInfo.pCurKey, StateInfo.uiElmPKCLen,
					TRUE, MOD_BINARY))
				goto Exit;

			/* Display the key length */

			if (!ViewAddMenuItem( LBL_KEY_LENGTH, LabelWidth,
						VAL_IS_NUMBER | DISP_DECIMAL,
						(FLMUINT)StateInfo.uiElmKeyLen, 0,
						FSGetFileNumber( StateInfo.uiBlkAddress),
						FSGetFileOffset( StateInfo.uiBlkAddress) + 
							StateInfo.uiElmOffset, 0,
						MOD_KEY_LEN | MOD_DECIMAL,
						Col, Row++, 0, bc, fc, bc, fc))
				goto Exit;

			/* Output the current key portion, if any */

			if (!OutputHexValue( Col, &Row, bc, fc, LBL_ELEMENT_KEY,
						FSGetFileNumber( StateInfo.uiBlkAddress),
						FSGetFileOffset( StateInfo.uiBlkAddress) +
							StateInfo.uiElmOffset + BBE_KEY,
						StateInfo.pElmKey, StateInfo.uiElmKeyLen,
						FALSE, MOD_BINARY))
				goto Exit;
			if ((LfType != LF_INDEX) && (LfType != LF_INVALID))
			{
				if (StateInfo.uiElmDrn != DRN_LAST_MARKER)
				{
					if (!ViewAddMenuItem( LBL_ELEMENT_DRN, LabelWidth,
							VAL_IS_NUMBER | DISP_DECIMAL_HEX,
							StateInfo.uiElmDrn, 0,
							FSGetFileNumber( StateInfo.uiBlkAddress),
							FSGetFileOffset( StateInfo.uiBlkAddress) + 
								StateInfo.uiElmOffset + BBE_KEY,
							StateInfo.uiElmKeyLen,
							MOD_DISABLED | MOD_BINARY,
							Col, Row++, 0, bc, fc, bc, fc))
					goto Exit;
				}

				/* Display the Next DRN marker if there is one */

				if (StateInfo.uiElmDrn == DRN_LAST_MARKER)
				{
					if (!ViewAddMenuItem( LBL_NEXT_DRN_MARKER, LabelWidth,
							VAL_IS_NUMBER | DISP_DECIMAL_HEX,
							StateInfo.uiLastElmDrn, 0,
							FSGetFileNumber( StateInfo.uiBlkAddress),
							FSGetFileOffset( StateInfo.uiBlkAddress) + 
								StateInfo.uiElmOffset + BBE_KEY+4,
							StateInfo.uiElmKeyLen,
							MOD_DISABLED | MOD_BINARY,
							Col, Row++, 0, bc, fc, bc, fc))
					goto Exit;
				}
			}

			/* Display the record length */

			if (!ViewAddMenuItem( LBL_RECORD_LENGTH, LabelWidth,
					VAL_IS_NUMBER | DISP_DECIMAL,
					(FLMUINT)StateInfo.uiElmRecLen, 0,
					FSGetFileNumber( StateInfo.uiBlkAddress),
					FSGetFileOffset( StateInfo.uiBlkAddress) + 
						StateInfo.uiElmOffset + BBE_RL,
					0,
					MOD_FLMBYTE | MOD_DECIMAL,
					Col, Row++, 0, bc, fc, bc, fc))
				goto Exit;

			if ((LfType == LF_INDEX) || (LfType == LF_INVALID))
			{

				/* Output the record portion */

				if (!OutputHexValue( Col, &Row, bc, fc, LBL_RECORD,
						FSGetFileNumber(StateInfo.uiBlkAddress),
						FSGetFileOffset(StateInfo.uiBlkAddress) +
							(FLMUINT)(StateInfo.pElmRec - StateInfo.pBlk),
						StateInfo.pElmRec, StateInfo.uiElmRecLen,
						FALSE, MOD_BINARY))
					goto Exit;
			}
		}
		if ((LfType != LF_INDEX) && 
			 (LfType != LF_INVALID) &&
			 (StateInfo.uiCurKeyLen) &&
			 ( StateInfo.uiElmDrn != DRN_LAST_MARKER))
		{
			if (!OutputElmRecord( Col, &Row, bc, fc, LabelWidth,
										&StateInfo, ElmStatus, StatusOnlyFlag))
				goto Exit;
		}

		/* Go to the next element */

		StateInfo.uiElmOffset += StateInfo.uiElmLen;
		OrStatusBits( BlkStatusRV, ElmStatus);
	}

	/* Verify that we ended exactly on the end of the block */

	if ((!TestStatusBit( BlkStatusRV, FLM_BAD_BLK_HDR_BLK_END)) &&
			(StateInfo.uiElmOffset > StateInfo.uiEndOfBlock))
		SetStatusBit( BlkStatusRV, FLM_BAD_ELM_END);

	if (!StatusOnlyFlag)
	{
		*RowRV = Row;

		/* If we were searching and did not find a key, set it on the */
		/* last key found */

		if ((gv_bViewSearching) && (gv_pViewSearchItem == NULL))
		{
			gv_pViewSearchItem = gv_pViewMenuLastItem;
			while ((gv_pViewSearchItem != NULL) &&
					 (gv_pViewSearchItem->iLabelIndex != LBL_ELEMENT_NUMBER))
				gv_pViewSearchItem = gv_pViewSearchItem->PrevItem;
		}
	}
	iRc = 1;
Exit:
	if (bStateInitialized && StateInfo.pRecord)
	{
		StateInfo.pRecord->Release();
		StateInfo.pRecord = NULL;
	}
	return( iRc);
}

/***************************************************************************
Name:    ViewLeafBlk
Desc:    This routine outputs a LEAF block, including the block header.
*****************************************************************************/
FLMINT ViewLeafBlk(
	FLMUINT		ReadAddress,
	FLMUINT		BlkAddress,
	FLMBYTE **	BlkPtrRV,
	BLK_EXP_p	BlkExp
	)
{
	FLMUINT        Row;
	FLMUINT        Col;
	FLMBYTE *		BlkPtr;
	FLMBYTE			BlkStatus [NUM_STATUS_BYTES];
	FLMUINT16      ui16CalcChkSum;
	FLMUINT16		ui16BlkChkSum;
	FLMUINT			uiBytesRead;
	FLMBOOL			bEncrypted;

	InitStatusBits( BlkStatus);

	/* Read the block into memory */

	if (!ViewBlkRead( ReadAddress, BlkPtrRV,
										gv_ViewHdrInfo.FileHdr.uiBlockSize,
										&ui16CalcChkSum, &ui16BlkChkSum,
										&uiBytesRead, TRUE, NULL, TRUE, &bEncrypted))
		return( 0);
	BlkPtr = *BlkPtrRV;

	if (!ViewMenuInit( "LEAF Block"))
		return( 0);

	/* Output the block header first */

	Row = 0;
	Col = 5;
	BlkExp->Type = BHT_LEAF;
	BlkExp->BlkAddr = BlkAddress;
	BlkExp->Level = 0;

	OutputLeafElements( Col, &Row, BlkPtr, BlkExp, BlkStatus, TRUE,
								bEncrypted);
	if (!ViewOutBlkHdr( Col, &Row, BlkPtr, BlkExp, BlkStatus,
								ui16CalcChkSum, ui16BlkChkSum))
		return( 0);

	/* Now output the leaf data */

	if (!OutputLeafElements( Col, &Row, BlkPtr, BlkExp,
							BlkStatus, FALSE, bEncrypted))
		return( 0);
	return( 1);
}

/***************************************************************************
Name:    OutputNonLeafElements
Desc:    This routine outputs the elements of a NON-LEAF block.
*****************************************************************************/
FSTATIC FLMINT OutputNonLeafElements(
	FLMUINT        Col,
	FLMUINT  *		RowRV,
	FLMBYTE *		BlkPtr,
	BLK_EXP_p		BlkExp,
	FLMBYTE *		BlkStatusRV,
	FLMUINT        StatusOnlyFlag,
	FLMBOOL			bEncrypted
	)
{
	FLMINT				iRc = 0;
	FLMUINT				LabelWidth = 30;
	eColorType			bc = FLM_BLACK;
	eColorType			fc = FLM_LIGHTGRAY;
	eColorType			mbc = FLM_BLACK;
	eColorType			mfc = FLM_WHITE;
	eColorType			sbc = FLM_BLUE;
	eColorType			sfc = FLM_WHITE;
	FLMUINT				Row = *RowRV;
	FLMUINT				ElementCount = 0;
	FLMBYTE *			TempAddrPtr;
	FLMUINT				TempAddress;
	eCorruptionType	eCorruptionCode;
	FLMUINT				Option;
	FLMUINT				LfType;
	FLMUINT				uiBlkType = BlkExp->Type;
	FLMBYTE				ElmStatus [NUM_STATUS_BYTES];
	STATE_INFO			StateInfo;
	FLMBOOL				bStateInitialized = FALSE;
	BLOCK_INFO			BlockInfo;
	FLMBYTE				KeyBuffer [MAX_KEY_SIZ];
	LF_HDR				LogicalFile;
	LF_HDR *				pLogicalFile = NULL;
	LFILE *				pLFile = NULL;
	FLMUINT				uiFixedDrn = 0;

	if (BlkExp->LfNum == 0)
		BlkExp->LfNum = FB2UW( &BlkPtr [BH_LOG_FILE_NUM]);

	/* Setup the STATE variable for processing through the block */

	ViewGetDictInfo();
	if (gv_bViewHaveDictInfo)
	{
		if ((RC_OK( fdictGetIndex(
				((FDB *)gv_hViewDb)->pDict,
				((FDB *)gv_hViewDb)->pFile->bInLimitedMode,
				BlkExp->LfNum, &pLFile, NULL))) ||
			 (RC_OK( fdictGetContainer(
			 		((FDB *)gv_hViewDb)->pDict,
					BlkExp->LfNum, &pLFile))))
		{
			f_memset( &LogicalFile, 0, sizeof( LF_HDR));
			pLogicalFile = &LogicalFile;
			LogicalFile.pLFile = pLFile;
			if (pLFile->uiLfType == LF_INDEX)
			{
				if (RC_BAD( fdictGetIndex(
						((FDB *)gv_hViewDb)->pDict,
						((FDB *)gv_hViewDb)->pFile->bInLimitedMode,
						pLFile->uiLfNum, &LogicalFile.pLFile,
						&LogicalFile.pIxd)))
				{
					pLogicalFile = NULL;
				}
				LogicalFile.pIfd = LogicalFile.pIxd->pFirstIfd;
			}
		}
	}
	LfType = (pLogicalFile) ? pLogicalFile->pLFile->uiLfType : LF_INVALID;

	if (uiBlkType != BHT_NON_LEAF &&
		 uiBlkType != BHT_NON_LEAF_COUNTS &&
		 uiBlkType != BHT_NON_LEAF_DATA)
	{
		if (pLogicalFile)
		{
			if (LfType == LF_INDEX)
			{
				if (pLogicalFile->pIxd &&
					 (pLogicalFile->pIxd->uiFlags & IXD_POSITIONING))
				{
					uiBlkType = BHT_NON_LEAF_COUNTS;
				}
				else
				{
					uiBlkType = BHT_NON_LEAF;
				}
			}
			else
			{
				uiBlkType = BHT_NON_LEAF_DATA;
			}
		}
		else
		{
			uiBlkType = BHT_NON_LEAF;
		}
	}

	flmInitReadState( &StateInfo, &bStateInitialized,
							gv_ViewHdrInfo.FileHdr.uiVersionNum,
							(gv_bViewDbInitialized)
							? (FDB *)gv_hViewDb
							: (FDB *)NULL, pLogicalFile, BlkExp->Level,
							uiBlkType, KeyBuffer);
	StateInfo.uiBlkAddress = BlkExp->BlkAddr;
	StateInfo.pBlk = BlkPtr;

	if ((eCorruptionCode = flmVerifyBlockHeader( &StateInfo, &BlockInfo,
								gv_ViewHdrInfo.FileHdr.uiBlockSize,
														BlkExp->NextAddr,
														BlkExp->PrevAddr,
														(FLMBOOL)(StateInfo.pDb != NULL
																	 ? TRUE
																	 : FALSE), TRUE)) != FLM_NO_CORRUPTION)
		SetStatusBit( BlkStatusRV, eCorruptionCode);

	if (bEncrypted)
	{
		if (!StatusOnlyFlag)
		{
			FLMUINT	uiEncSize = getEncryptSize( StateInfo.pBlk) - BH_OVHD;
			
			if (uiEncSize)
			{
			
				// Output the rest of the block as HEX - cannot be parsed through
				// when it is encrypted
		
				OutputHexValue( Col, &Row, bc, fc, LBL_ENC_DATA,
						FSGetFileNumber(StateInfo.uiBlkAddress),
						FSGetFileOffset(StateInfo.uiBlkAddress) + BH_OVHD,
						&StateInfo.pBlk [BH_OVHD], uiEncSize, FALSE, MOD_BINARY_ENC);
			}
		}
		iRc = 1;
		goto Exit;
	}
	
	/* Output each element in the block */

	while (StateInfo.uiElmOffset < StateInfo.uiEndOfBlock)
	{
		InitStatusBits( ElmStatus);
		ElementCount++;

		if ((eCorruptionCode = flmVerifyElement( &StateInfo, FLM_CHK_FIELDS)) != FLM_NO_CORRUPTION)
			SetStatusBit( ElmStatus, eCorruptionCode);

		if (!StatusOnlyFlag)
		{
			Row++;

			/* Output the element number */

			if (!ViewAddMenuItem( LBL_ELEMENT_NUMBER, LabelWidth,
						VAL_IS_NUMBER | DISP_DECIMAL,
						(FLMUINT)ElementCount, 0,
						FSGetFileNumber( StateInfo.uiBlkAddress),
						FSGetFileOffset( StateInfo.uiBlkAddress) + 
							StateInfo.uiElmOffset,
						0, MOD_DISABLED,
						Col, Row++, 0,
						FLM_GREEN, FLM_WHITE,
						FLM_GREEN, FLM_WHITE))
				goto Exit;

			/* Output the element offset within the block */

			if (!ViewAddMenuItem( LBL_ELEMENT_OFFSET, LabelWidth,
						VAL_IS_NUMBER | DISP_DECIMAL,
						(FLMUINT)StateInfo.uiElmOffset, 0,
						FSGetFileNumber( StateInfo.uiBlkAddress),
						FSGetFileOffset( StateInfo.uiBlkAddress) + 
								StateInfo.uiElmOffset,
						0, MOD_DISABLED,
						Col, Row++, 0, bc, fc, bc, fc))
				goto Exit;

			/* Output the element length */

			if (!ViewAddMenuItem( LBL_ELEMENT_LENGTH, LabelWidth,
						VAL_IS_NUMBER | DISP_DECIMAL,
						(FLMUINT)StateInfo.uiElmLen, 0,
						FSGetFileNumber( StateInfo.uiBlkAddress),
						FSGetFileOffset( StateInfo.uiBlkAddress) + 
							StateInfo.uiElmOffset,
						0, MOD_DISABLED,
						Col, Row++, 0, bc, fc, bc, fc))
				goto Exit;

			/* Display the domain flag */

			if (!ViewAddMenuItem( LBL_DOMAIN_PRESENT_FLAG, LabelWidth,
						VAL_IS_LABEL_INDEX,
						(BNE_IS_DOMAIN( StateInfo.pElm))
						? (FLMUINT)LBL_YES
						: (FLMUINT)LBL_NO, 0,
						FSGetFileNumber( StateInfo.uiBlkAddress),
						FSGetFileOffset( StateInfo.uiBlkAddress) + 
							StateInfo.uiElmOffset,
						0x80, MOD_BITS | MOD_DECIMAL,
						Col, Row++, 0, bc, fc, bc, fc))
				goto Exit;

			/* Display the domain number */

			if (uiBlkType != BHT_NON_LEAF_DATA && (BNE_IS_DOMAIN( StateInfo.pElm)))
			{
				TempAddrPtr = &StateInfo.pElmKey [StateInfo.uiElmKeyLen];
				TempAddress =  ((FLMUINT) (*TempAddrPtr++)) << 16;
				TempAddress |= (FLMUINT) ((*TempAddrPtr++) << 8);
				TempAddress |=  (FLMUINT) (*TempAddrPtr);
				TempAddress <<= 8;
			}
			else
				TempAddress = 0;
			if (!ViewAddMenuItem( LBL_DOMAIN_NUMBER, LabelWidth,
						VAL_IS_NUMBER | DISP_HEX_DECIMAL,
						(FLMUINT)TempAddress, 0,
						FSGetFileNumber( StateInfo.uiBlkAddress),
						FSGetFileOffset( StateInfo.uiBlkAddress) +
								((FLMUINT)(StateInfo.pElmKey - StateInfo.pBlk) +
								StateInfo.uiElmKeyLen),
						0,
						(FLMBYTE)((TempAddress) 
									? (FLMBYTE)MOD_CHILD_BLK
									: (FLMBYTE)MOD_DISABLED),
						Col, Row++, 0, bc, fc, bc, fc))
				goto Exit;

			/* Display the child block address */

			if (uiBlkType == BHT_NON_LEAF_DATA)
			{
				uiFixedDrn = f_bigEndianToUINT32( StateInfo.pElm);
				TempAddrPtr = &StateInfo.pElm [BNE_DATA_CHILD_BLOCK];
			}
			else
			{
				TempAddrPtr = &StateInfo.pElm [BNE_CHILD_BLOCK];
			}
			TempAddress = FB2UD( TempAddrPtr);
			if (TempAddress == BT_END)
				Option = 0;
			else
				Option = BLK_OPTION_CHILD_BLOCK | StateInfo.uiElmOffset;
			if (!ViewAddMenuItem( LBL_CHILD_BLOCK_ADDRESS, LabelWidth,
						VAL_IS_NUMBER | DISP_HEX_DECIMAL,
						(FLMUINT)TempAddress, 0,
						FSGetFileNumber( StateInfo.uiBlkAddress),
						FSGetFileOffset( StateInfo.uiBlkAddress) +
							StateInfo.uiElmOffset +
							(FLMUINT) (TempAddrPtr - StateInfo.pElm),
						0, MOD_CHILD_BLK,
						Col, Row++, Option,
						!Option ? bc : mbc,
						!Option ? fc : mfc,
						!Option ? bc : sbc,
						!Option ? fc : sfc))
				goto Exit;

			if (uiBlkType == BHT_NON_LEAF_COUNTS)
			{
				TempAddrPtr = &StateInfo.pElm [BNE_CHILD_COUNT];
				if (!ViewAddMenuItem( LBL_CHILD_REFERENCE_COUNT, LabelWidth,
							VAL_IS_NUMBER,
							(FLMUINT) StateInfo.uiChildCount, 0,
							FSGetFileNumber( StateInfo.uiBlkAddress),
							FSGetFileOffset( StateInfo.uiBlkAddress) +
								StateInfo.uiElmOffset +
								(FLMUINT) (TempAddrPtr - StateInfo.pElm),
							0, MOD_FLMUINT | MOD_DECIMAL,
							Col, Row++, Option, bc, fc, bc, fc))
					goto Exit;
			}

			/* Remember this item if we are searching */

			if ((gv_bViewSearching) &&
					(flmCompareKeys( StateInfo.pCurKey, StateInfo.uiCurKeyLen,
											gv_ucViewSearchKey, gv_uiViewSearchKeyLen) >= 0) &&
					(gv_pViewSearchItem == NULL))
			{
				gv_pViewSearchItem = gv_pViewMenuLastItem;
			}

			if (eCorruptionCode != FLM_NO_CORRUPTION)
			{
				if (!OutputStatus( Col, &Row, bc, fc, LabelWidth, LBL_ELEMENT_STATUS,
													 ElmStatus))
					goto Exit;
			}

			/* Display the Previous Key Count */

			if (!ViewAddMenuItem( LBL_PREVIOUS_KEY_CONT_LEN, LabelWidth,
						VAL_IS_NUMBER | DISP_DECIMAL,
						(FLMUINT)StateInfo.uiElmPKCLen, 0,
						FSGetFileNumber( StateInfo.uiBlkAddress),
						FSGetFileOffset( StateInfo.uiBlkAddress) + 
							StateInfo.uiElmOffset, 0x0F,
						MOD_BITS | MOD_DECIMAL,
						Col, Row++, 0, bc, fc, bc, fc))
				goto Exit;

			if (uiBlkType != BHT_NON_LEAF_DATA)
			{
				/* Output the previous key portion, if any */

				if (!OutputHexValue( Col, &Row, bc, fc,
							LBL_PREV_ELEMENT_KEY,
							0, VIEW_INVALID_FILE_OFFSET,
							StateInfo.pCurKey, StateInfo.uiElmPKCLen,
							TRUE, MOD_BINARY))
					goto Exit;

				/* Display the key length */

				if (!ViewAddMenuItem( LBL_KEY_LENGTH, LabelWidth,
						VAL_IS_NUMBER | DISP_DECIMAL,
						(FLMUINT)StateInfo.uiElmKeyLen, 0,
						FSGetFileNumber( StateInfo.uiBlkAddress),
						FSGetFileOffset( StateInfo.uiBlkAddress) + 
							StateInfo.uiElmOffset, 0,
						MOD_KEY_LEN | MOD_DECIMAL,
						Col, Row++, 0, bc, fc, bc, fc))
					goto Exit;

				/* Output the current key portion, if any */

				if (!OutputHexValue( Col, &Row, bc, fc, LBL_ELEMENT_KEY,
						FSGetFileNumber( StateInfo.uiBlkAddress),
						FSGetFileOffset( StateInfo.uiBlkAddress) + 
							(FLMUINT)(StateInfo.pElmKey - StateInfo.pBlk),
						StateInfo.pElmKey, StateInfo.uiElmKeyLen,
						FALSE, MOD_BINARY))
					goto Exit;
			}

			if (((LfType != LF_INDEX) && (LfType != LF_INVALID)) ||
				 (uiBlkType == BHT_NON_LEAF_DATA))
			{
				FLMUINT	uiDrn = StateInfo.uiElmDrn;

				if (uiBlkType == BHT_NON_LEAF_DATA)
				{
					uiDrn = uiFixedDrn;
				}
				if (!ViewAddMenuItem( LBL_ELEMENT_DRN, LabelWidth,
						VAL_IS_NUMBER | DISP_DECIMAL_HEX,
						uiFixedDrn, 0,
						FSGetFileNumber( StateInfo.uiBlkAddress),
						FSGetFileOffset( StateInfo.uiBlkAddress) + 
							(FLMUINT)(StateInfo.pElmKey - StateInfo.pBlk),
						StateInfo.uiElmKeyLen,
						MOD_DISABLED | MOD_BINARY,
						Col, Row++, 0, bc, fc, bc, fc))
					goto Exit;
			}
		}

		/* Go to the next element */

		StateInfo.uiElmOffset += StateInfo.uiElmLen;
		OrStatusBits( BlkStatusRV, ElmStatus);
	}

	/* Verify that we ended exactly on the end of the block */

	if ((!TestStatusBit( BlkStatusRV, FLM_BAD_BLK_HDR_BLK_END)) &&
			(StateInfo.uiElmOffset > StateInfo.uiEndOfBlock))
		SetStatusBit( BlkStatusRV, FLM_BAD_ELM_END);

	if (!StatusOnlyFlag)
	{
		*RowRV = Row;

		/* If we were searching and did not find a key, set it on the */
		/* last key found */

		if ((gv_bViewSearching) && (gv_pViewSearchItem == NULL))
		{
			gv_pViewSearchItem = gv_pViewMenuLastItem;
			while ((gv_pViewSearchItem != NULL) &&
						 (gv_pViewSearchItem->iLabelIndex != LBL_CHILD_BLOCK_ADDRESS))
				gv_pViewSearchItem = gv_pViewSearchItem->PrevItem;
		}
	}
	iRc = 1;
Exit:
	if (bStateInitialized && StateInfo.pRecord)
	{
		StateInfo.pRecord->Release();
		StateInfo.pRecord = NULL;
	}
	return( iRc);
}

/***************************************************************************
Name:    ViewNonLeafBlk
Desc:    This routine outputs a NON-LEAF block, including the block header.
*****************************************************************************/
FLMINT ViewNonLeafBlk(
	FLMUINT		ReadAddress,
	FLMUINT		BlkAddress,
	FLMBYTE **	BlkPtrRV,
	BLK_EXP_p	BlkExp
	)
{
	FLMUINT		Row;
	FLMUINT     Col;
	FLMBYTE *   BlkPtr;
	FLMBYTE     BlkStatus [NUM_STATUS_BYTES];
	FLMUINT16   ui16CalcChkSum;
	FLMUINT16	ui16BlkChkSum;
	FLMUINT		uiBytesRead;
	FLMBOOL		bEncrypted;

	InitStatusBits( BlkStatus);

	/* Read the block into memory */

	if (!ViewBlkRead( ReadAddress, BlkPtrRV,
										gv_ViewHdrInfo.FileHdr.uiBlockSize,
										&ui16CalcChkSum, &ui16BlkChkSum,
										&uiBytesRead, TRUE, NULL, TRUE, &bEncrypted))
		return( 0);
	BlkPtr = *BlkPtrRV;

	if (!ViewMenuInit( "NON-LEAF Block"))
		return( 0);

	/* Output the block header first */

	Row = 0;
	Col = 5;
	BlkExp->Type = BH_GET_TYPE( BlkPtr); 
	BlkExp->BlkAddr = BlkAddress;
	OutputNonLeafElements( Col, &Row, BlkPtr, BlkExp, BlkStatus, TRUE,
				bEncrypted);
	if (!ViewOutBlkHdr( Col, &Row, BlkPtr, BlkExp, BlkStatus,
							ui16CalcChkSum, ui16BlkChkSum))
		return( 0);

	/* Now output the non-leaf data */

	if (!OutputNonLeafElements( Col, &Row, BlkPtr, BlkExp, BlkStatus, FALSE,
				bEncrypted))
		return( 0);
	return( 1);
}

/********************************************************************
Desc: ?
*********************************************************************/
void ViewHexBlock(
	FLMUINT			ReadAddress,
	FLMBYTE **		BlkPtrRV,
	FLMBOOL        bViewDecrypted,
	FLMUINT        uiViewLen
	)
{
	FLMBYTE *	BlkPtr;
	FLMUINT     Row = 0;
	FLMUINT     Col = 9;
	FLMUINT     BytesPerLine = MAX_HORIZ_SIZE( Col);
	FLMUINT     BytesProcessed = 0;
	FLMUINT     NumBytes;
	FLMUINT     FileOffset;
	FLMUINT		FileNumber;
	char     	Title [80];
	FLMUINT16	ui16CalcChkSum;
	FLMUINT16	ui16BlkChkSum;
	FLMUINT		uiBytesRead;
	FLMBOOL		bIsEncBlock;
	FLMBOOL		bEncrypted;
	FLMUINT		uiModFlag;

	FileOffset = FSGetFileOffset( ReadAddress);
	FileNumber = FSGetFileNumber( ReadAddress);

	if (!ViewBlkRead( ReadAddress, BlkPtrRV, uiViewLen,
										&ui16CalcChkSum, &ui16BlkChkSum,
										&uiBytesRead, TRUE, &bIsEncBlock,
										bViewDecrypted, &bEncrypted))
		return;
	BlkPtr = *BlkPtrRV;

	f_sprintf( (char *)Title, "HEX DISPLAY OF BLOCK %08X (%s)", (unsigned)ReadAddress,
		(char *)(bIsEncBlock
					? (char *)(bEncrypted
								  ? (char *)"DECRYPTED"
								  : (char *)"ENCRYPTED (RAW)")
					: (char *)"RAW"));
	if (!ViewMenuInit( Title))
		return;

	uiModFlag = bEncrypted ? MOD_BINARY_ENC : MOD_BINARY;
	while (BytesProcessed < uiViewLen)
	{
		if ((NumBytes = uiViewLen - BytesProcessed) > BytesPerLine)
			NumBytes = BytesPerLine;

		/* Output the line */

		if (!ViewAddMenuItem( -1, 0,
				VAL_IS_BINARY_HEX,
				(FLMUINT)BlkPtr, NumBytes,
				FileNumber, FileOffset, NumBytes, uiModFlag,
				Col, Row++, 0,
				FLM_BLACK, FLM_LIGHTGRAY,
				FLM_BLACK, FLM_LIGHTGRAY))
			return;
		FileOffset += (FLMUINT)NumBytes;
		BytesProcessed += NumBytes;
		BlkPtr += NumBytes;
	}
}


/***************************************************************************
Name: ViewBlocks
Desc: This routine outputs a block in the database.  Depending on the
		type of block, it will call a different routine to display
		the block.  The routine then allows the user to press keys to
		navigate to other blocks in the database if desired.
*****************************************************************************/
void ViewBlocks(
	FLMUINT     ReadAddress,
	FLMUINT     BlkAddress,
	BLK_EXP_p   BlkExp
	)
{
	FLMUINT     Option;
	VIEW_INFO   SaveView;
	VIEW_INFO   DummySave;
	FLMUINT		Done = 0;
	FLMUINT     Repaint = 1;
	FLMBYTE *   BlkPtr = NULL;
	FLMUINT     BlkAddress2;
	BLK_EXP     BlkExp2;
	FLMUINT     SetExp = FALSE;
	FLMUINT		Type;
	FLMBOOL		bViewHex = FALSE;
	FLMBOOL     bViewDecrypted = FALSE;
	FLMUINT		uiBytesRead;

	/* Loop getting commands until hit the exit key */

	if (!gv_bViewHdrRead)
		ViewReadHdr();
	gv_pViewSearchItem = NULL;
	ViewReset( &SaveView);
	while ((!Done) && (!gv_bViewPoppingStack))
	{

		/* Display the type of block expected */

		if (Repaint)
		{
			if (bViewHex)
			{
				ViewHexBlock( ReadAddress, &BlkPtr, bViewDecrypted,
											gv_ViewHdrInfo.FileHdr.uiBlockSize);
			}
			else
			{
Switch_Statement:
				switch( BlkExp->Type)
				{
					case BHT_NON_LEAF_COUNTS:
					case BHT_NON_LEAF:
					case BHT_NON_LEAF_DATA:
						if (!ViewNonLeafBlk( ReadAddress, BlkAddress,
																 &BlkPtr, BlkExp))
							Done = 1;
						break;
					case BHT_LEAF:
						if (!ViewLeafBlk( ReadAddress, BlkAddress,
															&BlkPtr, BlkExp))
							Done = 1;
						break;
					case BHT_FREE:
						if (!ViewAvailBlk( ReadAddress, BlkAddress,
															 &BlkPtr, BlkExp))
							Done = 1;
						break;
					case BHT_LFH_BLK:
						if (!ViewLFHBlk( ReadAddress, BlkAddress,
														 &BlkPtr, BlkExp))
							Done = 1;
						break;
					case 0xFF:
						if (!ViewBlkRead( ReadAddress, &BlkPtr,
												gv_ViewHdrInfo.FileHdr.uiBlockSize,
												NULL, NULL, &uiBytesRead, FALSE,
												NULL, FALSE, NULL))
						{
							Done = 1;
						}
						else
						{
							BlkExp->Type = BH_GET_TYPE( BlkPtr );
							goto Switch_Statement;
						}
						break;
				}
			}
		}

		/* See what the user wants to do next. */

		if (!Done)
		{
			if ((SetExp) &&
				 ((BlkExp->Type == BHT_LEAF) || 
				  (BlkExp->Type == BHT_NON_LEAF_COUNTS) ||
				  (BlkExp->Type == BHT_NON_LEAF) ||
				  (BlkExp->Type == BHT_NON_LEAF_DATA) ))
			{
				BlkExp->LfNum = FB2UW( &BlkPtr [BH_LOG_FILE_NUM]);
				BlkExp->Level = BlkPtr [BH_LEVEL];
			}
			SetExp = FALSE;
			Repaint = 1;
			if (gv_bViewSearching)
			{
				SetSearchTopBottom();
				if ((BlkExp->Type == BHT_LEAF) || (gv_pViewSearchItem == NULL))
				{
					gv_bViewSearching = FALSE;
					ViewEnable();
					Option = ViewGetMenuOption();
				}
				else
					Option = gv_pViewSearchItem->Option;
			}
			else
			{
				ViewEnable();
				Option = ViewGetMenuOption();
			}
			switch( Option)
			{
				case ESCAPE_OPTION:
					Done = 1;
					break;
				case PREV_BLOCK_OPTION:
					ViewReset( &DummySave);
					BlkExp->NextAddr = BlkAddress;
					BlkExp->PrevAddr = 0;
					ReadAddress = BlkAddress = FB2UD( &BlkPtr [BH_PREV_BLK]);
					break;
				case NEXT_BLOCK_OPTION:
					ViewReset( &DummySave);
					BlkExp->NextAddr = 0;
					BlkExp->PrevAddr = BlkAddress;
					ReadAddress = BlkAddress = FB2UD( &BlkPtr [BH_NEXT_BLK]);
					break;
				case PREV_BLOCK_IMAGE_OPTION:
					if (BlkExp->Type == BHT_PCODE_BLK)
					{
						ViewShowError( 
							"This option not supported for PCODE blocks");
						Repaint = 0;
					}
					else
					{
						f_memcpy( &BlkExp2, BlkExp, sizeof( BLK_EXP));
						BlkExp2.NextAddr = 0;
						BlkExp2.PrevAddr = 0;
						ViewBlocks( FB2UD( &BlkPtr [BH_PREV_BLK_ADDR]),
												BlkAddress, &BlkExp2);
					}
					break;
				case GOTO_BLOCK_OPTION:
					if (GetBlockAddrType( &BlkAddress2, &Type))
					{
						ViewReset( &DummySave);
						ReadAddress = BlkAddress = BlkAddress2;
						BlkExp->Type = Type;
						BlkExp->Level = 0xFF;
						BlkExp->NextAddr = 0;
						BlkExp->PrevAddr = 0;
						BlkExp->LfNum = 0;
						SetExp = TRUE;
						if (BlkAddress < 2048)
						{
							bViewDecrypted = FALSE;
							bViewHex = TRUE;
						}
					}
					else
						Repaint = 0;
					break;
				case EDIT_OPTION:
				case EDIT_RAW_OPTION:
					if (!ViewEdit( TRUE,
									(Option == EDIT_OPTION) ? TRUE : FALSE))
						Repaint = 0;
					break;
				case HEX_OPTION:
					ViewDisable();
					bViewHex = bViewHex ? FALSE : TRUE;
					if (!bViewHex)
					{
						bViewDecrypted = FALSE;
					}
					break;
				case DECRYPT_OPTION:
					if (bViewHex)
					{
						ViewDisable();
						bViewDecrypted = bViewDecrypted ? FALSE : TRUE;
					}
					else
					{
						Repaint = 0;
					}
					break;
				case SEARCH_OPTION:
					switch( BH_GET_TYPE( BlkPtr))
					{
						case BHT_NON_LEAF_COUNTS:
						case BHT_NON_LEAF:
						case BHT_NON_LEAF_DATA:
						case BHT_LEAF:
							gv_uiViewSearchLfNum = FB2UW( &BlkPtr [BH_LOG_FILE_NUM]);
							if (ViewGetKey())
								gv_bViewPoppingStack = TRUE;
							break;
						case BHT_LFH_BLK:
							{
								VIEW_MENU_ITEM_p  vp = gv_pViewMenuCurrItem;

								/* Determine which logical file, if any we are pointing at */

								while ((vp != NULL) &&
											(vp->iLabelIndex != LBL_LOGICAL_FILE_NAME))
									vp = vp->PrevItem;
								if (vp != NULL)
								{
									while ((vp != NULL) &&
											(vp->iLabelIndex != LBL_LOGICAL_FILE_NUMBER))
										vp = vp->NextItem;
								}
								if (vp != NULL)
								{
									gv_uiViewSearchLfNum = (FLMUINT)vp->Value;
									if (ViewGetKey())
										gv_bViewPoppingStack = TRUE;
								}
								else
									ViewShowError(
										"Position cursor to a logical file before searching");
							}
							break;
						case BHT_PCODE_BLK:
							{
								VIEW_MENU_ITEM_p  vp = gv_pViewMenuCurrItem;

								/* Determine which logical file, if any we are pointing at */

								if ((vp->iLabelIndex != LBL_CONTAINER) ||
									(vp->iLabelIndex != LBL_INDEX_CONTAINER))
								{
									while ((vp != NULL) &&
											 (vp->iLabelIndex != LBL_INDEX))
										vp = vp->PrevItem;
								}
								if (vp != NULL)
								{
									gv_uiViewSearchLfNum = (FLMUINT)(vp->Option & (~(LOGICAL_FILE_OPTION)));
									if (ViewGetKey())
										gv_bViewPoppingStack = TRUE;
								}
								else
									ViewShowError( 
										"Position cursor to a logical file before searching");
							}
							break;
						default:
							ViewShowError(
								"This block does not belong to a logical file - cannot search");
							break;
					}
					break;
				default:
					if (Option & LOGICAL_FILE_OPTION)
						ViewLogicalFile( (FLMUINT)(Option & (~(LOGICAL_FILE_OPTION))));
					else if ((Option & LFH_OPTION_ROOT_BLOCK) ||
								(Option & LFH_OPTION_LAST_BLOCK))
					{
						FLMUINT   Offset = (FLMUINT)(BH_OVHD +
													(Option & 0x0FFF) * LFH_SIZE);

						FLMBYTE * LFHPtr = &BlkPtr [Offset];

						if (Option & LFH_OPTION_ROOT_BLOCK)
						{
							BlkExp2.Level = 0xFF;
							BlkExp2.Type = 0xFF;
							BlkAddress2 = FB2UD( &LFHPtr [LFH_ROOT_BLK_OFFSET]);
							BlkExp2.LfNum = FB2UW( &LFHPtr [LFH_LF_NUMBER_OFFSET]);
							BlkExp2.NextAddr = BlkExp2.PrevAddr = BT_END;
						}
						else
						{
							flmAssert( 0);
						}
						ViewBlocks( BlkAddress2, BlkAddress2, &BlkExp2);
					}
					else if (Option & BLK_OPTION_CHILD_BLOCK)
					{
						FLMBYTE * TempAddrPtr;
			
						if (BlkExp->Type == BHT_NON_LEAF_DATA)
							TempAddrPtr = &BlkPtr [(Option & 0xFFF) + BNE_DATA_CHILD_BLOCK];
						else
							TempAddrPtr = &BlkPtr [(Option & 0xFFF) + BNE_CHILD_BLOCK];
	
						BlkAddress2 = FB2UD( TempAddrPtr);
						f_memcpy( &BlkExp2, BlkExp, sizeof( BLK_EXP));
						BlkExp2.NextAddr = 0;
						BlkExp2.PrevAddr = 0;
						if (BlkExp2.Level == 0xFF)
							BlkExp2.Level = BlkPtr [BH_LEVEL] - 1;
						else
							BlkExp2.Level--;
						if (!BlkExp2.Level)
							BlkExp2.Type = BHT_LEAF;
						ViewBlocks( BlkAddress2, BlkAddress2, &BlkExp2);
					}
					else
						Repaint = 0;
					break;
			}
		}
	}
	f_free( &BlkPtr);
	ViewRestore( &SaveView);
}

/********************************************************************
Desc: ?
*********************************************************************/
FLMINT GetBlockAddrType(
	FLMUINT  *	BlkAddressRV,
	FLMUINT *	BlkTypeRV)
{
	char			TempBuf [20];
	FLMUINT     i;
	FLMUINT     c;
	FLMUINT     BadDigit;
	FLMUINT     GotAddress = FALSE;
	FLMUINT     GotType = FALSE;
	FLMUINT     GetOK = 1;
	FLMUINT		uiNumCols;
	FLMUINT		uiNumRows;

	f_conGetScreenSize( &uiNumCols, &uiNumRows);

	/* First get the block address */

	while ((!GotAddress) && (GetOK))
	{
		BadDigit = FALSE;
		f_conSetBackFore( FLM_BLACK, FLM_WHITE);
		f_conClearScreen( 0, uiNumRows - 2);
		ViewAskInput( "Enter Block Address (in hex): ", 
			TempBuf, sizeof( TempBuf));
		if ((f_stricmp( TempBuf, "\\") == 0) ||
				(!TempBuf [0]))
		{
			GetOK = 0;
			break;
		}
		i = 0;
		*BlkAddressRV = 0;
		while ((TempBuf [i]) && (i < 8))
		{
			(*BlkAddressRV) <<= 4;
			c = TempBuf [i];
			if ((c >= '0') && (c <= '9'))
				(*BlkAddressRV) += (FLMUINT)(c - '0');
			else if ((c >= 'a') && (c <= 'f'))
				(*BlkAddressRV) += (FLMUINT)(c - 'a' + 10);
			else if ((c >= 'A') && (c <= 'F'))
				(*BlkAddressRV) += (FLMUINT)(c - 'A' + 10);
			else
			{
				BadDigit = TRUE;
				break;
			}
			i++;
		}
		if (BadDigit)
			ViewShowError( 
				"Illegal digit in number - must be hex digits");
		else if (TempBuf [i])
			ViewShowError(
				"Too many characters in number");
		else
			GotAddress = TRUE;
	}

	/* Next get the block type */

	if (GetOK)
	{
		f_conSetBackFore( FLM_BLACK, FLM_WHITE);
		f_conClearScreen( 0, uiNumRows - 1);
		f_conStrOutXY( "View: 1=Leaf, 2=NLeafCnts, 3=NLeafVar, 4=NLeafFix, 5=Avail, 6=LFH: ",
			0, uiNumRows - 1);
	}
	while ((GetOK) && (!GotType))
	{
		c = f_conGetKey();
		switch( c)
		{
			case FKB_ESCAPE:
			case '0':
				GetOK = 0;
				break;
			case '1':
				GotType = TRUE;
				*BlkTypeRV = BHT_LEAF;
				break;
			case '2':
				GotType = TRUE;
				*BlkTypeRV = BHT_NON_LEAF_COUNTS;
				break;
			case '3':
				GotType = TRUE;
				*BlkTypeRV = BHT_NON_LEAF;
				break;
			case '4':
				GotType = TRUE;
				*BlkTypeRV = BHT_NON_LEAF_DATA;
				break;
			case '5':
				GotType = TRUE;
				*BlkTypeRV = BHT_FREE;
				break;
			case '6':
				GotType = TRUE;
				*BlkTypeRV = BHT_LFH_BLK;
				break;
			default:
				break;
		}
	}
	f_conClearScreen( 0, uiNumRows - 2);
	ViewEscPrompt();
	return( GetOK);
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC void SetSearchTopBottom(
	void
	)
{
	if (gv_pViewSearchItem == NULL)
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
		gv_uiViewMenuCurrItemNum = gv_pViewSearchItem->ItemNum;
		gv_uiViewCurrFileOffset = gv_pViewSearchItem->ModFileOffset;
		gv_uiViewCurrFileNumber = gv_pViewSearchItem->ModFileNumber;
		gv_uiViewTopRow = gv_pViewSearchItem->Row;
		if (gv_uiViewTopRow < LINES_PER_PAGE / 2)
			gv_uiViewTopRow = 0;
		else
			gv_uiViewTopRow -= (LINES_PER_PAGE / 2);
	}
	gv_uiViewBottomRow = gv_uiViewTopRow + LINES_PER_PAGE - 1;
	if (gv_uiViewBottomRow > gv_pViewMenuLastItem->Row)
	{
		gv_uiViewBottomRow = gv_pViewMenuLastItem->Row;
		if (gv_uiViewBottomRow < LINES_PER_PAGE)
			gv_uiViewTopRow = 0;
		else
			gv_uiViewTopRow = gv_uiViewBottomRow - LINES_PER_PAGE + 1;
	}
}
