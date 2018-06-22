//-------------------------------------------------------------------------
// Desc:	Insert or modify a record into a container b-tree.
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

FSTATIC RCODE FSBldRecElement(
	FDB *			pDb,
	LFILE *		pLFile,
	FlmRecord *	pRecord,
	UCUR * 		uCur);

/**************************************************************************
Desc:		Add, Delete or Replace a gedcom record pointed to by gt and
			identified by a drn
**************************************************************************/
RCODE FSRecUpdate(
	FDB *			pDb,
	LFILE *		pLFile,
	FlmRecord *	pRecord,
	FLMUINT		uiDrn,							// Data record number - never 0	
	FLMUINT		uiAddAppendFlags)				// REC_UPD_ ADD | NEW_RECCORD | MODIFY | DELETE
{
	RCODE			rc;
	BTSK			stackBuf[ BH_MAX_LEVELS ];	// Stack to hold b-tree variables
	BTSK *		pStack = stackBuf;			// Points to a stack element
	FLMBYTE		pKeyBuf[ DIN_KEY_SIZ + 4 ];// Key buffer pointed to by stack
	UCUR			updCur;							// Update cursor
	FLMBYTE *	pElmBuf;							// Points to updCur.buffer

	// Set up the stack

	FSInitStackCache( &stackBuf [0], BH_MAX_LEVELS);
	pStack->pKeyBuf = pKeyBuf;
	f_UINT32ToBigEndian( (FLMUINT32)uiDrn, updCur.pKeyBuf);

	// Position to the element in the b-tree.

	if( uiAddAppendFlags & REC_UPD_NEW_RECORD)
	{
		if( RC_BAD( rc = FSBtSearchEnd( pDb, pLFile, &pStack, uiDrn)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = FSBtSearch( pDb, pLFile, &pStack,
							updCur.pKeyBuf, DIN_KEY_SIZ, 0)))
		{
			goto Exit;
		}
	}
	
	// If the record exists, and we are adding, it is an error.

	if( (pStack->uiCmpStatus == BT_EQ_KEY) && (uiAddAppendFlags & REC_UPD_ADD))
	{
		rc = RC_SET( FERR_EXISTS);
		goto Exit;
	}

	// The LFILE may need to be initialized.

	if( pLFile->uiRootBlk == BT_END)
	{
		if( RC_BAD( rc = flmLFileInit( pDb, pLFile)))
		{
			goto Exit;
		}

		// Position to the element in the b-tree

		if( RC_BAD( rc = FSBtSearch( pDb, pLFile, &pStack, updCur.pKeyBuf,
											 DIN_KEY_SIZ,0)))
		{
			goto Exit;
		}
	}

	updCur.uiFlags = (pStack->uiCmpStatus != BT_EQ_KEY) 
										? UCUR_INSERT 
										: UCUR_REPLACE;
	updCur.uiDrn = uiDrn;

	// Update or Insert the record.
	// The stack is positioned to the correct place.

	if( pRecord)
	{
		if( FB2UD( &pStack->pBlk[ BH_NEXT_BLK]) == BT_END)
		{
			if( RC_BAD( rc = FSSetNextDrn( pDb, pStack, uiDrn, FALSE)))
			{
				goto Exit;
			}
		}

		// Setup the element buffer 

		pElmBuf = updCur.pElmBuf;
		pElmBuf[ BBE_PKC] = BBE_FIRST_FLAG;
		pElmBuf[ BBE_KL] = DIN_KEY_SIZ;
		f_UINT32ToBigEndian( (FLMUINT32)uiDrn, &pElmBuf[ BBE_KEY]);

		// BBE_RL is set in the flush routine 

		updCur.uiBufLen  = ELM_DIN_OVHD + MAX_REC_ELM;
		updCur.uiUsedLen = ELM_DIN_OVHD;
		updCur.pStack = pStack;
		
		rc = FSBldRecElement( pDb, pLFile, pRecord, &updCur);
		updCur.uiFlags |= UCUR_LAST_TIME;
		if( RC_BAD( rc))
		{
			goto Exit;
		}

		if( RC_BAD( rc = FSFlushElement( pDb, pLFile, &updCur)))
		{
			goto Exit;
		}
	}
	else if( updCur.uiFlags & UCUR_INSERT)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}
	else
	{
		// Delete all remaining elements by deleting all of the 
		// elements that have the BBE_NOT_LAST flag set then deleting
		// the last element.

		while( BBE_NOT_LAST( CURRENT_ELM( pStack)))
		{
			if( RC_BAD( rc = FSBtDelete( pDb, pLFile, &pStack)))
			{
				goto Exit;
			}
		}

		if( RC_BAD( rc = FSBtDelete( pDb, pLFile, &pStack)))
		{
			goto Exit;
		}
	}

Exit:

	FSReleaseStackCache( stackBuf, BH_MAX_LEVELS, FALSE);
	return( rc);
}

/***************************************************************************
Desc:		Saves a record to the file. The record may overflow & span elements.
*****************************************************************************/
FSTATIC RCODE FSBldRecElement(
	FDB *			pDb,
	LFILE *		pLFile,
	FlmRecord *	pRecord,
	UCUR * 		updCur)
{
	RCODE					rc = FERR_OK;
	void *				pvField;
	FLMBYTE *			pBuf = updCur->pElmBuf;				// Make pointer for performance
	FLMBYTE *			pField;									// Points to the output field
	FLMBYTE *			pBufEnd;									// Points to the end of the buffer
	const FLMBYTE *	pValue = NULL;							// Points to the value on the line
	FLMBYTE *			pOvhd;
	FLMUINT				uiBufLen = updCur->uiBufLen;		// Used to optimize
	FLMUINT				uiValuePos;								// Position in the value buffer
	FLMUINT				uiBytesLeft;							// Number of bytes left in element
	FLMUINT				uiValueLen;								// Length of the value
	FLMUINT				uiUsedLen = updCur->uiUsedLen;
	FLMUINT				uiTagNum;
	FLMUINT				uiLevel;
	FLMUINT				uiFldType;
	FLMINT				iLevelCntx;
	FLMINT				iLastNdLevel = -1;					// Last node's level number
	FLMBYTE				ucBaseFlags;
	FLMUINT				uiEncValueLen;
	FLMUINT				uiEncTagNum;

	// Loop on each field writing as the element buffer fills up.

	for( pvField = pRecord->root(); pvField; pvField = pRecord->next( pvField))
	{
		FLMBOOL		bFldEncrypted;
		FLMUINT		uiEncFlags = 0;
		FLMUINT		uiEncId = 0;

		// Check for encryption.

		bFldEncrypted = pRecord->isEncryptedField( pvField);
		if( bFldEncrypted)
		{
			// May still proceed if the field is already encrypted.

			uiEncFlags = pRecord->getEncFlags( pvField);

			// Cannot add encrypted field while in limited mode.

			if (pDb->pFile->bInLimitedMode &&
				 (!(uiEncFlags & FLD_HAVE_ENCRYPTED_DATA)))
			{
				rc = RC_SET( FERR_ENCRYPTION_UNAVAILABLE);
				goto Exit;
			}

			// Only encrypt the field if it isn't already encrypted.

			if( !(uiEncFlags & FLD_HAVE_ENCRYPTED_DATA))
			{
				uiEncId = pRecord->getEncryptionID( pvField);

				if (RC_BAD( rc = flmEncryptField( 
					pDb->pDict, pRecord, pvField, uiEncId, &pDb->TempPool)))
				{
					goto Exit;
				}
			}
		}

		// First determine if there is enough room for just the tag overhead

		uiBytesLeft = (FLMUINT) (uiBufLen - uiUsedLen);

		// Check for overflow - want all tagged fields with field overhead

		if( uiBytesLeft <= MAX_FLD_OVHD)
		{
			// Flush element

			if( RC_BAD( rc = FSFlushElement( pDb, pLFile, updCur)))
			{
				goto Exit;
			}

			uiUsedLen = updCur->uiUsedLen;
		}

		pField = &pBuf[ uiUsedLen];
		pBufEnd = &pBuf[ uiBufLen];

		if( bFldEncrypted)
		{
			if( RC_BAD( rc = pRecord->getFieldInfo( pvField,
				&uiTagNum, &uiLevel, &uiFldType, &uiValueLen,
				&uiEncValueLen, &uiEncTagNum)))
			{
				goto Exit;
			}
		}
		else
		{
			uiEncValueLen = 0;
			uiEncTagNum = 0;

			if( RC_BAD( rc = pRecord->getFieldInfo( pvField, &uiTagNum, &uiLevel,
						&uiFldType, &uiValueLen, NULL, NULL)))
			{
				goto Exit;
			}
		}

		// Do data sanity checks

		if( pDb->pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_61)
		{
			if( uiValueLen > 0x0000FFFF)
			{
				rc = RC_SET( FERR_VALUE_TOO_LARGE);
				goto Exit;
			}
		}

		if( !uiTagNum)
		{
			flmAssert( 0);
			rc = RC_SET( FERR_BAD_FIELD_NUM);
			goto Exit;
		}

		if (!bFldEncrypted)
		{
			if( uiFldType == FLM_NUMBER_TYPE)
			{
				if( uiValueLen > 11)
				{
					flmAssert( 0);
					rc = RC_SET( FERR_BAD_DATA_LENGTH);
					goto Exit;
				}
			}
			else if( uiFldType == FLM_CONTEXT_TYPE)
			{
				if( uiValueLen != 0 && uiValueLen != 4)
				{
					flmAssert( 0);
					rc = RC_SET( FERR_BAD_DATA_LENGTH);
					goto Exit;
				}
			}
		}
		
		if( uiFldType > FLM_CONTEXT_TYPE && uiFldType != FLM_BLOB_TYPE)
		{
			rc = RC_SET( FERR_BAD_FIELD_TYPE);
			goto Exit;
		}

		// Level number is growing correctly. 

		if( uiLevel > (FLMUINT)iLastNdLevel && iLastNdLevel != -1)				
		{
			if( uiLevel != (FLMUINT)(iLastNdLevel + 1))
			{
				flmAssert( 0);
				rc = RC_SET( FERR_BAD_FIELD_LEVEL);
				goto Exit;
			}
		}
		
		flmAssert( uiTagNum != 0);

		if( bFldEncrypted)
		{
			flmAssert( uiEncTagNum != 0);
		}

		if( (iLevelCntx = iLastNdLevel) == -1)
		{
			iLevelCntx = uiLevel;
		}

		if( (iLevelCntx -= uiLevel) > 0)
		{
			do
			{
				*pField++ = (FLMBYTE)(FOP_SET_LEVEL +
									((iLevelCntx >= FOP_LEVEL_MAX)
										? (FLMBYTE)FOP_LEVEL_MAX
										: (FLMBYTE)iLevelCntx));
				iLevelCntx -= FOP_LEVEL_MAX;
			} while( iLevelCntx > 0 );
			iLevelCntx = 0;
		}
		iLastNdLevel = (FLMINT) uiLevel;

		// Get a pointer to the data (if any)

		if( uiValueLen)
		{
			pValue = pRecord->getDataPtr( pvField);
		}

		if( uiEncValueLen)
		{
			flmAssert( bFldEncrypted);
			pValue = pRecord->getEncryptionDataPtr( pvField);
			flmAssert( pValue);
		}

		// Determine the FOP used for storing the field

		if( uiValueLen > 0xFFFF)
		{
			goto StoreLargeField;
		}
		else if( uiTagNum < FLM_DICT_FIELD_NUMS)
		{
			// Normal field (not dictionary and not unregistered)

			if( !bFldEncrypted)
			{
				goto StoreDefinedField;
			}
			else
			{
				goto StoreEncryptedField;
			}
		}
		else if( uiTagNum >= FLM_UNREGISTERED_TAGS)
		{
			if( !bFldEncrypted)
			{
				goto StoreTaggedField;
			}
			else
			{
				goto StoreEncryptedField;
			}
		}
		else
		{
			// Dictionary fields 

			if( !bFldEncrypted)
			{
				if( uiFldType == FLM_TEXT_TYPE || uiFldType == FLM_CONTEXT_TYPE)
				{
					goto StoreDefinedField;
				}
				else
				{
					goto StoreTaggedField;
				}
			}
			else
			{
				goto StoreEncryptedField;
			}
		}

		// Store the field

StoreDefinedField:

		flmAssert( uiValueLen <= 0xFFFF);

		if( uiTagNum <= FSTA_MAX_FLD_NUM && uiValueLen <= FSTA_MAX_FLD_LEN)
		{
			// FOP_STANDARD

			*pField++ = (FLMBYTE)((iLevelCntx ? 0x40 : 0) + uiValueLen);
			*pField++ = (FLMBYTE)uiTagNum;
		}
		else
		{
			pOvhd = pField++;

			if( uiValueLen)
			{
				// FOP_OPEN

				ucBaseFlags = iLevelCntx ? (FOP_OPEN + 0x08) : FOP_OPEN;
				*pField++ = (FLMBYTE)uiTagNum;

				if( uiTagNum > 0xFF)
				{
					ucBaseFlags |= 0x02;
					*pField++ = (FLMBYTE)(uiTagNum >> 8);
				}

				*pField++ = (FLMBYTE) uiValueLen;
				if( uiValueLen > 0x000000FF)
				{
					ucBaseFlags |= 0x01;
					*pField++ = (FLMBYTE)(uiValueLen >> 8);
				}
			}
			else
			{
				// FOP_NO_VALUE

				ucBaseFlags = iLevelCntx ? (FOP_NO_VALUE + 0x04) : FOP_NO_VALUE;
				*pField++ = (FLMBYTE) uiTagNum;

				if( uiTagNum > 0xFF)
				{
					ucBaseFlags |= 0x02;
					*pField++ = (FLMBYTE) (uiTagNum >> 8);
				}
			}

			*pOvhd = ucBaseFlags;
		}
		goto WriteValuePortion;
	
StoreTaggedField:

		// FOP_TAGGED
		//
		// Used for data dictionary records, unregistered, and local fields.
		// Format similar to FOP_OPEN except the field type is stored with
		// the data.
		//
		// When storing the unregistered fields we cleared 
		// the high bit to save on storage for VER11.  The problem is 
		// that if a tag that is not in the unregistered range 
		// (FLAIM TAGS) cannot be represented.  SO, we will XOR the 
		// high bit so 0x0111 is stored as 0x8111 and 0x8222 is stored 
		// as 0x0222.

		pOvhd = pField++;

		ucBaseFlags = iLevelCntx 
							? (FOP_TAGGED + 0x08) 
							: FOP_TAGGED;

		uiTagNum ^= 0x8000;	// Clear high bit if SET (VER11) or set
									// if it is in the FLM_TAGs range

		*pField++ = (FLMBYTE) uiFldType;
		*pField++ = (FLMBYTE) uiTagNum;

		if( uiTagNum > 0xFF)
		{
			ucBaseFlags |= 0x02;
			*pField++ = (FLMBYTE) (uiTagNum >> 8);
		}

		*pField++ = (FLMBYTE) uiValueLen;
		if( uiValueLen > 0xFF)
		{
			ucBaseFlags |= 0x01;
			*pField++ = (FLMBYTE)(uiValueLen >> 8);
		}

		*pOvhd = ucBaseFlags;
		goto WriteValuePortion;
		
StoreEncryptedField:

		// FOP_ENCRYPTED

		*pField++ = iLevelCntx ? (FOP_ENCRYPTED + 0x01) : FOP_ENCRYPTED;
		*pField = (FLMBYTE)0;
		*pField = (FLMBYTE)(uiFldType << 4);

		if (uiTagNum > 0xFF)
		{
			*pField |= 0x08;
		}

		if (uiValueLen > 0xFF)
		{
			*pField |= 0x04;
		}

		if (uiEncTagNum > 0xFF)
		{
			*pField |= 0x02;
		}

		if (uiEncValueLen > 0xFF)
		{
			*pField |= 0x01;
		}

		pField++;

		*pField++ = (FLMBYTE)uiTagNum;
		if (uiTagNum > 0xFF)
		{
			*pField++ = (FLMBYTE)(uiTagNum >> 8);
		}

		*pField++ = (FLMBYTE) uiValueLen;
		if( uiValueLen > 0xFF)
		{
			*pField++ = (FLMBYTE) (uiValueLen >> 8);
		}

		*pField++ = (FLMBYTE)uiEncTagNum;
		if (uiEncTagNum > 0xFF)
		{
			*pField++ = (FLMBYTE)(uiEncTagNum >> 8);
		}

		*pField++ = (FLMBYTE) uiEncValueLen;
		if( uiEncValueLen > 0xFF)
		{
			*pField++ = (FLMBYTE) (uiEncValueLen >> 8);
		}

		// Copy the encrypted value length (uiEncValueLen) into the value length
		// (uiValueLength).  This will be use in the next section so that we can
		// copy the right amount of data.

		uiValueLen = uiEncValueLen;
		goto WriteValuePortion;
		
StoreLargeField:

		// FOP_LARGE
	
		// 1101 xxec
		// fieldType (1 byte, 0000 ffff)
		// tagNum (2 bytes)
		// dataLen (4 bytes)
		//
		// If encrypted, the following are also present:
		//
		// encryptionId (2 bytes)
		// encryptionLength (4 bytes)
	
		*pField = FOP_LARGE;
	
		if( iLevelCntx)
		{
			*pField |= 0x01;
		}
	
		if( bFldEncrypted)
		{
			*pField |= 0x02;
		}
		pField++;
	
		*pField = (FLMBYTE)(uiFldType & 0x0F);
		pField++;
	
		UW2FBA( (FLMUINT16)uiTagNum, pField);
		pField += 2;
	
		UD2FBA( (FLMUINT32)uiValueLen, pField);
		pField += 4;
	
		if( bFldEncrypted)
		{
			UW2FBA( (FLMUINT16)uiEncTagNum, pField);
			pField += 2;
	
			UD2FBA( (FLMUINT32)uiEncValueLen, pField);
			pField += 4;
	
			// Copy the encrypted value length (uiEncValueLen) into the value length
			// (uiValueLength).  This will be use in the next section so that we can
			// copy the right amount of data.
	
			uiValueLen = uiEncValueLen;
		}

WriteValuePortion:

		// Write out the data a chunk at a time
	
		uiBytesLeft = (FLMUINT)(pBufEnd - pField);
		if( uiValueLen <= uiBytesLeft)
		{
			// YES - enough room to copy the node and go on 
	
			if( uiValueLen)
			{
				f_memcpy( pField, pValue, uiValueLen);
				pField += uiValueLen;
			}
		}
		else
		{
			// The node's data will overflow the buffer.
			// Write until it is full, flush and write again.
	
			uiValuePos = 0;
			for( ;;)
			{
				FLMUINT		uiTemp;
	
				f_memcpy( pField, &pValue[ uiValuePos], uiBytesLeft);
				
				pField  += uiBytesLeft;
				uiValuePos += uiBytesLeft;
	
				if( uiValuePos >= uiValueLen)
				{
					break;
				}
	
	
				updCur->uiUsedLen = uiUsedLen = uiBufLen;
	
				if( RC_BAD( rc = FSFlushElement( pDb, pLFile, updCur )))
				{
					goto Exit;
				}
				
				uiUsedLen = updCur->uiUsedLen;
	
				// Reset local variables 
	
				pField = &pBuf[ uiUsedLen];
	
				// Compute number of bytes to move for the next element
	
				uiBytesLeft = (FLMUINT)(pBufEnd - pField);
				uiTemp = (FLMUINT)(uiValueLen - uiValuePos);
				uiBytesLeft = f_min( uiBytesLeft, uiTemp);
			}
		}
	
		// Set the used length and get the next node 
	
		updCur->uiUsedLen = uiUsedLen = (FLMUINT)(pField - pBuf);
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:		Flush the current element to the file.  Set updCur->flag to state.
*****************************************************************************/
RCODE FSFlushElement(
	FDB *			pDb,
	LFILE *		pLFile,
	UCUR *		updCur)
{
	RCODE			rc = FERR_OK;
	BTSK *		pStack = updCur->pStack;
	FLMBYTE *	pElmBuf = updCur->pElmBuf;
	FLMUINT		uiFlags = updCur->uiFlags;
	FLMBOOL		bIsLast = FALSE;

	if( uiFlags & UCUR_LAST_TIME)
	{
		BBE_SET_LAST( pElmBuf);
	}
	BBE_SET_RL( pElmBuf, (updCur->uiUsedLen - ELM_DIN_OVHD));

	if( uiFlags & UCUR_REPLACE)
	{
		FLMBYTE * curElm = CURRENT_ELM( pStack);

		bIsLast = (FLMBYTE)(BBE_IS_LAST( curElm ));

		if( bIsLast && (( uiFlags & UCUR_LAST_TIME ) == 0))
		{
			// Log the block before modifying it. 

			if( RC_BAD( rc = FSLogPhysBlk( pDb, pStack)))
			{
				goto Exit;
			}

			curElm = CURRENT_ELM( pStack);
			BBE_CLR_LAST( curElm);
		}
		else if( !bIsLast && (uiFlags & UCUR_LAST_TIME))
		{
			// Log the block before modifying it. 

			if( RC_BAD( rc = FSLogPhysBlk( pDb, pStack)))
			{
				goto Exit;
			}

			curElm = CURRENT_ELM( pStack );
			BBE_SET_LAST( curElm);
		}

		if( RC_BAD( rc = FSBtReplace( pDb, pLFile, 
			&pStack, pElmBuf, updCur->uiUsedLen)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = FSBtInsert( pDb, pLFile, &pStack, 
			pElmBuf, updCur->uiUsedLen)))
		{
			goto Exit;
		}
	}

	if( BBE_IS_FIRST( pElmBuf))
	{
		BBE_CLR_FIRST( pElmBuf);
	}
		
	// Should always setup the pStack to the next element (or LEM)

	if( RC_BAD( rc = FSBtNextElm( pDb, pLFile, pStack)))
	{
		if( rc != FERR_BT_END_OF_DATA)
		{
			goto Exit;
		}

		updCur->pStack = pStack;
		rc = FERR_OK;
	}

	if( !(uiFlags & UCUR_LAST_TIME))
	{
		// If you are replacing the current element (continuation),
		// go to the next element and keep replacing to your
		// hearts content.  Check if the doesContinue flag is set
		// so that you will replace next time FSFlushElement() is called.  

		if( uiFlags & UCUR_REPLACE)
		{
			// If the element DOES NOT continue then insert next time 

			if( bIsLast)
			{
				updCur->uiFlags = uiFlags = UCUR_INSERT;
			}
		}

		// Flags could have just changed above 

		if( uiFlags & UCUR_INSERT)
		{
			if( RC_BAD( rc = FSBtScanTo( pStack, &pElmBuf[ BBE_KEY],
												 DIN_KEY_SIZ, 0)))
			{
				goto Exit;
			}
		}

		// Reset element to build again

		updCur->uiUsedLen = ELM_DIN_OVHD;
	}
	else
	{
 		// This is the last time - kill additional continuation elements
		
		if( uiFlags & UCUR_REPLACE)
		{
			// Need to delete all other continuation records.

			while( !bIsLast)
			{
				bIsLast = (FLMBYTE)(BBE_IS_LAST( CURRENT_ELM( pStack)));
					
				if( RC_BAD( rc = FSBtDelete( pDb, pLFile, &pStack)))
				{
					break;
				}
			}
		}
	}

	updCur->pStack = pStack;

Exit:

	return( rc );
}

/****************************************************************************
Desc:		Sets or verifies the next DRN for the domain supplied. 
			If the domain has no data blocks then LFH area will be updated.
			Must have the stack positioned to the right most tree.
Notes:	All checks are very explicit to make clear for maintenance.
****************************************************************************/
RCODE FSSetNextDrn(
	FDB *			pDb,					// Pointer to operation context struct.
	BTSK *		pStack,				// Points to a stack element
	FLMUINT		uiDrn,				// (IN) Value to update next counter
	FLMBOOL		bManditoryChange)	// (IN) If true then must be set.
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiNextDrn;
	FLMBYTE *	ptr;
	FLMBYTE *	pBlk = GET_CABLKPTR( pStack );

	// Next DRN marker element in b-tree.  By this time, the root
	// block will have been initialized. The '11' below is the size
	// of the next DRN marker element.
	
	if( FB2UD( &pBlk[ BH_NEXT_BLK ]) == BT_END && 
		pStack->uiCurElm + 11 + BBE_LEM_LEN >= pStack->uiBlkEnd)
	{
		ptr = CURRENT_ELM( pStack);
		ptr += BBE_GETR_KL( ptr) + BBE_KEY;
		uiNextDrn = FB2UD( ptr);
	
		// Check if the DRN is 0 or more than the NEXT DRN marker

		if( uiDrn >= uiNextDrn)
		{
			uiNextDrn = uiDrn + 1;

			if( RC_BAD( rc = FSLogPhysBlk( pDb, pStack)))
			{
				goto Exit;
			}

			ptr = CURRENT_ELM( pStack);
			ptr += BBE_GETR_KL( ptr) + BBE_KEY;

			// Update with the next DRN value and dirty the block

			UD2FBA( uiNextDrn, ptr);
			goto Exit;
		}
	}

	if( bManditoryChange)
	{
		rc = RC_SET( FERR_BTREE_ERROR);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Sets or verifies the next DRN for the domain supplied. 
			If the domain is non-default domain then LFH area will be updated
			for FLM_1_1 databases ONLY.  FLM_1_2 stores next DRN in last block.
****************************************************************************/
RCODE FSGetNextDrn(
	FDB *			pDb,					// Pointer to operation context struct.
	LFILE *		pLFile,				// Domain Logical File Definition
	FLMBOOL		bUpdateNextDrn,	// (IN) TRUE then update next drn value 
	FLMUINT *	puiDrnRV)			// (IN) 0 or value to check if higher
											// (OUT) Returns input value or next value
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiNextDrn = *puiDrnRV;
	BTSK			stackBuf[ BH_MAX_LEVELS];
	FLMBOOL		bUsedStack = FALSE;

	if( uiNextDrn == DRN_LAST_MARKER)
	{
		rc = RC_SET( FERR_BAD_DRN);
		goto Exit;
	}

	// All containers have next DRN marker

	if( !uiNextDrn)
	{
		BTSK *		pStack = stackBuf;				
		FLMBYTE		pKeyBuf[ DIN_KEY_SIZ + 4];
		FLMBYTE *	ptr;

		// Set up the stack

		bUsedStack = TRUE;
		FSInitStackCache( &stackBuf [0], BH_MAX_LEVELS);
		pStack->pKeyBuf = pKeyBuf;

		// Make sure pLFile is up to date

		if( RC_BAD( rc = FSBtSearchEnd( pDb, pLFile, &pStack, DRN_LAST_MARKER)))
		{
			goto Exit;
		}

		if( pLFile->uiRootBlk == BT_END)
		{
			*puiDrnRV = pLFile->uiNextDrn;
			if( bUpdateNextDrn)
			{
				pLFile->uiNextDrn++;
				if( RC_BAD( rc = flmLFileWrite( pDb, pLFile)))
				{
					pLFile->uiNextDrn--;
					goto Exit;
				}
			}
		}
		else
		{
			if( pStack->uiCmpStatus != BT_EQ_KEY || 
				pLFile->uiLfNum != FB2UW( &pStack->pBlk[ BH_LOG_FILE_NUM]))
			{
				rc = RC_SET( FERR_BTREE_ERROR);
				goto Exit;
			}

			ptr = CURRENT_ELM( pStack);
			ptr += BBE_GETR_KL( ptr) + BBE_KEY;
			*puiDrnRV = FB2UD( ptr);

			if( bUpdateNextDrn)
			{
				FLMUINT32	ui32NextDrn = (FLMUINT32)(*puiDrnRV + 1);

				if( RC_BAD( rc = FSLogPhysBlk( pDb, pStack)))
				{
					goto Exit;
				}

				ptr = CURRENT_ELM( pStack);
				ptr += BBE_GETR_KL( ptr) + BBE_KEY;

				// Update with the next DRN value and dirty the block 

				UD2FBA( ui32NextDrn, ptr);
			}
		}
	}

	if( *puiDrnRV == DRN_LAST_MARKER)
	{
		rc = RC_SET( FERR_NO_MORE_DRNS);
		goto Exit;
	}

Exit:

	if( bUsedStack)
	{
		FSReleaseStackCache( stackBuf, BH_MAX_LEVELS, FALSE);
	}

	return( rc );
}
