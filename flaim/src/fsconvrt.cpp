//-------------------------------------------------------------------------
// Desc:	Database format conversion routines.
// Tabs:	3
//
// Copyright (c) 1999-2001, 2003-2007 Novell, Inc. All Rights Reserved.
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

FSTATIC RCODE FSConvertNonLeafTree(
	FDB *			pDb,
	LFILE *		pLFile,
	BTSK *		pOldStack,	
	BTSK *		pOldStackBase,
	FLMUINT		uiNewVersion,
	STATUS_HOOK	fnStatusCallback,
	void *	 	UserData,
	DB_UPGRADE_INFO * pDbConvertInfo);

FSTATIC void FSBuildNonLeafDataElement(
	BTSK *		pStack,
	FLMBYTE *	pElement,
	FLMUINT *	puiElmLen,
	FLMUINT		uiNewElmOvhd,
	FLMBYTE **	ppKey);
	
/***************************************************************************
Desc:		File system conversions from one version to another.
*****************************************************************************/
RCODE FSVersionConversion40(
	FDB *			pDb,
	FLMUINT		uiNewVersion,
	STATUS_HOOK	fnStatusCallback,
	void *	 	UserData)
{
	RCODE			rc = FERR_OK;
	FFILE *		pFile = pDb->pFile;
	LFILE *		pLFileTbl;
	FLMUINT		uiPos, uiTblSize;
	FLMUINT		uiCurrentVersion;
	BTSK *		pStack;						
	BTSK			stackBuf[ BH_MAX_LEVELS ];	
	FLMBYTE		pKeyBuf[ DIN_KEY_SIZ + 4 ];
	FLMBYTE		pDrnKey[ DIN_KEY_SIZ];
	DB_UPGRADE_INFO DbConvertInfo;

	f_memset( &DbConvertInfo, 0, sizeof( DB_UPGRADE_INFO));
	// Supported conversions...
	uiCurrentVersion = pFile->FileHdr.uiVersionNum;

	// Loop through all of the data blocks in the lfile.

	pLFileTbl =  pDb->pDict->pLFileTbl;
	uiTblSize = pDb->pDict->uiLFileCnt;
	FSInitStackCache( &stackBuf [0], BH_MAX_LEVELS);

	for( uiPos = 0; uiPos < uiTblSize; uiPos++)
	{
		LFILE *	pLFile;

		pLFile = &pLFileTbl [uiPos];

		if( pLFile->uiLfType == LF_CONTAINER)
		{
			if( fnStatusCallback)
			{
				DbConvertInfo.uiLastDrn = 0;
				if( RC_BAD( rc = FSGetNextDrn( pDb, pLFile, FALSE, 
						&DbConvertInfo.uiLastDrn)))
					goto Exit;
			}
			// Set up the stack
			FSInitStackCache( &stackBuf [0], BH_MAX_LEVELS);
			pStack = stackBuf;
			pStack->pKeyBuf = pKeyBuf;
			f_UINT32ToBigEndian( 0, pDrnKey);
			if( RC_BAD(rc = FSBtSearch( pDb, pLFile, &pStack,
												 pDrnKey, DIN_KEY_SIZ,0)))
				goto Exit;

			// If pStack isn't at stackBuf[] we have a b-tree of 2 or more levels.
			
			// VISIT: the change may not be necessary - may be new version.
			// Look at the root block type.
			
			if( pStack != stackBuf)
			{
				DbConvertInfo.uiContainer = pLFile->uiLfNum;

				if( RC_BAD( rc = FSConvertNonLeafTree( pDb, pLFile, pStack - 1, 
						stackBuf, uiNewVersion, 
						fnStatusCallback, UserData, &DbConvertInfo)))
					goto Exit;
			}
			FSReleaseStackCache( stackBuf, BH_MAX_LEVELS, FALSE);
		}
	}
			// Loop through all of the data b-trees and look for non-leaf blocks.
Exit:
	FSReleaseStackCache( stackBuf, BH_MAX_LEVELS, FALSE);
	return( rc);
}

/***************************************************************************
Desc:		Convert the non-leaf data blocks from one version to another version.
			pOldStack points to the level 1 non-leaf data blocks.  
			Read all of the elements and build a new non-leaf b-tree
			and make this the new tree while freeing up the old non-leaf blocks.
*****************************************************************************/
FSTATIC RCODE FSConvertNonLeafTree(
	FDB *			pDb,
	LFILE *		pLFile,
	BTSK *		pOldStack,
	BTSK *		pOldStackBase,
	FLMUINT		uiNewVersion,
	STATUS_HOOK	fnStatusCallback,
	void *	 	UserData,
	DB_UPGRADE_INFO * pDbConvertInfo)
{
	RCODE			rc = FERR_OK;
	BTSK			stackBuf[ BH_MAX_LEVELS ];	
	BTSK *		pStack;
	FLMBYTE		pKeyBuf[ DIN_KEY_SIZ + 4 ];
	FLMBYTE		pDrnKey[ DIN_KEY_SIZ];
	FLMBYTE		pElement[ DIN_KEY_SIZ + 16];	// Enough bytes for either format
	SCACHE *		pTempSCache;
	FLMUINT		uiBlkAddr;
	
	FSInitStackCache( &stackBuf [0], BH_MAX_LEVELS);
	f_UINT32ToBigEndian( 0, pDrnKey);

	/*
	Free the b-tree blocks that are above this lowest non-leaf level.
	This will reuse the blocks instead of allocating from the avail list or end
	of file.
	*/

	pOldStack->uiFlags &= ~FULL_STACK;
	for( ; pOldStackBase != pOldStack; pOldStackBase++ )
	{
		uiBlkAddr = pOldStackBase->uiBlkAddr;

		// Release block so that caller doesn't release an avail block.
		FSReleaseBlock( pOldStackBase, FALSE);

		while( uiBlkAddr != BT_END)
		{
			if( RC_BAD( rc = ScaGetBlock( pDb, pLFile, BHT_LEAF,
										uiBlkAddr, NULL, &pTempSCache)))
				goto Exit;
			uiBlkAddr = FB2UD( &pTempSCache->pucBlk[ BH_NEXT_BLK ]);

			if( RC_BAD( FSBlockFree( pDb, pTempSCache)))
				goto Exit;
		}
	}

	// Allocate a new root block in the new NON-LEAF format.

	pStack = stackBuf;
	pStack->pKeyBuf = pKeyBuf;	
	pLFile->uiRootBlk = BT_END;

	if( RC_BAD( flmLFileWrite( pDb, pLFile)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = flmLFileInit( pDb, pLFile)))
	{
		goto Exit;
	}

	if( RC_BAD(rc = FSGetBlock( pDb, pLFile, pLFile->uiRootBlk, pStack)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FSLogPhysBlk( pDb, pStack)))
	{
		goto Exit;
	}
	
	// Remove both leading elements in the block and setup as a non-leaf block.

	UW2FBA( BH_OVHD, &pStack->pBlk[ BH_BLK_END]);

	if( uiNewVersion >= FLM_FILE_FORMAT_VER_4_0)
	{
		pStack->pBlk[ BH_TYPE ] = BHT_NON_LEAF_DATA + BHT_ROOT_BLK;
	}
	else
	{
		pStack->pBlk[ BH_TYPE ] = BHT_NON_LEAF + BHT_ROOT_BLK;
	}

	pStack->pBlk[ BH_LEVEL ] = 1;
	pStack->uiKeyBufSize = DIN_KEY_SIZ;
	FSBlkToStack( pStack);
	pStack->uiFlags = FULL_STACK;

	// Build the key for the first element in the old stack.

	if( RC_BAD( rc = FSBtScanTo( pOldStack, NULL, 0, 0)))
	{
		goto Exit;
	}

	// Start inserting from pOldStack to pStack until pOldStack is done.

	for(;;)
	{
		FLMUINT		uiElmLen;
		FLMBYTE *	pKey;

		FSBuildNonLeafDataElement( pOldStack, pElement, 
			&uiElmLen, pStack->uiElmOvhd, &pKey);
		
		if( uiNewVersion <= FLM_FILE_FORMAT_VER_3_02)
		{
			flmAssert( pStack->uiCurElm == pStack->uiBlkEnd);
			if( pStack != stackBuf)
			{
				flmAssert( stackBuf[0].uiCurElm + 6 == stackBuf[0].uiBlkEnd);
			}

			pStack->uiCurElm = pStack->uiBlkEnd;

			// Call scanto to place the last element's key in keybuf.
			if( RC_BAD( rc = FSBtScanTo( pStack, NULL, 0, 0)))
				goto Exit;
			if( pStack->uiBlkEnd > BH_OVHD)
			{
				if( uiElmLen > BNE_KEY_START)
				{
					pStack->uiPrevElmPKC = 
						FSElmComparePKC( pKeyBuf, DIN_KEY_SIZ, pKey, DIN_KEY_SIZ);
				}
			}
		}
		if( RC_BAD( rc = FSBtInsert( pDb, pLFile, &pStack, pElement, uiElmLen)))
			goto Exit;

		// Go to next element so we always insert at the end.
		(void) FSBtNextElm( pDb, pLFile, pStack );

		uiBlkAddr = pOldStack->uiBlkAddr;
		if( RC_BAD( rc = FSBtNextElm( pDb, pLFile, pOldStack )))
		{
			// All done?  Free the last block in the old b-tree.
			if( rc == FERR_BT_END_OF_DATA)
			{
				if( RC_BAD( rc = ScaGetBlock( pDb, pLFile, BHT_LEAF,
											uiBlkAddr, NULL, &pTempSCache)))
					goto Exit;

				// No need to release pTempSCache.  FSBlockFree does it.
				if( RC_BAD( FSBlockFree( pDb, pTempSCache)))
					goto Exit;
				rc = FERR_OK;
				break;
			}
			goto Exit;
		}

		// Did next element position to a new block?
		if( uiBlkAddr != pOldStack->uiBlkAddr)
		{
			// Do callback to report progress.
			if (fnStatusCallback)
			{
				pDbConvertInfo->uiDrn = f_bigEndianToUINT32( pOldStack->pKeyBuf);
				if (RC_BAD( rc = (*fnStatusCallback)( FLM_DB_UPGRADE_STATUS,
											(void *) pDbConvertInfo,
											(void *)0, UserData)))
				{
					goto Exit;
				}
			}

			// Free the previous block back into the avail list.
			if( RC_BAD( rc = ScaGetBlock( pDb, pLFile, BHT_LEAF,
										uiBlkAddr, NULL, &pTempSCache)))
				goto Exit;
			if( RC_BAD( FSBlockFree( pDb, pTempSCache)))
				goto Exit;
			// No need to release pTempSCache.  FSBlockFree does it.
		}
	}

Exit:
	FSReleaseStackCache( stackBuf, BH_MAX_LEVELS, FALSE);
	return( rc);
}

/***************************************************************************
Desc:		Build a non-leaf data element from the current stack position.
*****************************************************************************/
FSTATIC void FSBuildNonLeafDataElement(
	BTSK *		pStack,
	FLMBYTE *	pElement,
	FLMUINT *	puiElmLen,
	FLMUINT		uiNewElmOvhd,
	FLMBYTE **	ppKey)
{
	FLMUINT		uiElmLen;

	// Grab the key from the key buffer,

	if( uiNewElmOvhd == BNE_DATA_OVHD)		// New fixed length 4.0 format.
	{
		FLMBYTE *	pOldElm = CURRENT_ELM( pStack);

		uiElmLen = BNE_DATA_OVHD; 
		*ppKey = pElement;
		
		// Check for last element marker.
		if( pOldElm[ BBE_PKC ] == 0 && pOldElm[ BBE_KL] == 0)
		{
			f_UINT32ToBigEndian( (FLMUINT32)DRN_LAST_MARKER, pElement);
		}
		else
		{
			f_memcpy( pElement, pStack->pKeyBuf, DIN_KEY_SIZ);
		}
	}
	else	// Old variable length 3.x format.
	{
		pElement [ BBE_PKC ] = 0;		// Set PKC, DOMAIN to zero 
		if( FB2UD( pStack->pKeyBuf) == DRN_LAST_MARKER)
		{
			pElement [ BBE_KL ] = 0;		
			uiElmLen = BNE_KEY_START;
		}
		else
		{
			pElement [ BBE_PKC ] = 0;		// Set PKC, DOMAIN to zero 
			pElement [ BBE_KL ] = DIN_KEY_SIZ; // sizeof( FLMUINT32) or 4
			uiElmLen = BNE_KEY_START + DIN_KEY_SIZ;
			f_memcpy( pElement + BNE_KEY_START, pStack->pKeyBuf, DIN_KEY_SIZ);
		}
		*ppKey = pElement + BNE_KEY_START;
	}
	// Set the child block address.
	FSSetChildBlkAddr( pElement, FSChildBlkAddr( pStack), uiNewElmOvhd);
	*puiElmLen = uiElmLen;
	return;
}
