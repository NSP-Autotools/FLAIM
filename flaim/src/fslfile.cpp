//-------------------------------------------------------------------------
// Desc:	Routines for accessing/updating an LFILE structure.
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

FSTATIC RCODE flmLFileToBuffer(
	LFILE *			pLFile,
	FLMBYTE *	pucBuf);

/***************************************************************************
Desc: 	Searches a block for an empty LFILE slot.  This is called whenever
			a new logical file is create so we re-use the slots.
			Supports VER11 and VER15 formats
****************************************************************************/
FINLINE FLMUINT flmLFileFindEmpty(
	FLMBYTE *	pucBlk)
{
	FLMUINT		uiPos = BH_OVHD;
	FLMUINT		uiEndPos = (FLMUINT) FB2UW( &pucBlk[ BH_ELM_END ]);

	while( (uiPos < uiEndPos ) && 
		(pucBlk[ uiPos + LFH_TYPE_OFFSET ] != LF_INVALID ))
	{
		uiPos += LFH_SIZE;
	}

	return( (uiPos < uiEndPos) ? uiPos : 0);
}

/***************************************************************************
Desc: 	Initialize an existing LFILE.  Right now the only LFILE data 
			structure is a b-tree so the root block will be allocated and
			initialized.  
****************************************************************************/
RCODE	flmLFileInit(
	FDB *			pDb,
	LFILE *		pLFile)
{
	RCODE			rc = FERR_OK;
	SCACHE *		pSCache;
	FLMBYTE *	pucBlk;
	FLMBOOL		bReleaseCache = FALSE;
	FLMUINT		uiBlkAddress;
	FLMUINT		uiBlkPos;

	if( RC_BAD( rc = flmLFileRead( pDb, pLFile)))
	{
		goto Exit;
	}

	if( pLFile->uiRootBlk != BT_END)
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = ScaCreateBlock( pDb, pLFile, &pSCache)))
	{
		goto Exit;
	}

	bReleaseCache = TRUE;
	pucBlk = pSCache->pucBlk;
	uiBlkAddress = GET_BH_ADDR( pucBlk);

	// Have the logical file header point to the root block

	pLFile->uiRootBlk = uiBlkAddress;
	
	// Initialize some other fields in the block header & log it 
	// before modifying.  The only type supported at this time is
	// a b-tree structure 

	pucBlk[ BH_TYPE ] = BHT_LEAF + BHT_ROOT_BLK;
	pucBlk[ BH_LEVEL] = 0;
	UD2FBA( BT_END, &pucBlk[ BH_PREV_BLK ]);
	UD2FBA( BT_END, &pucBlk[ BH_NEXT_BLK ]);
	UW2FBA( (FLMUINT16) pLFile->uiLfNum, &pucBlk[ BH_LOG_FILE_NUM ]);
	uiBlkPos = BH_OVHD;

	// Check for encrypted index
	if (pLFile->uiLfType == LF_INDEX)
	{
		IXD *		pIxd;

		if (RC_BAD( rc = fdictGetIndex(
					pDb->pDict, pDb->pFile->bInLimitedMode,
					pLFile->uiLfNum, NULL, &pIxd, TRUE)))
		{
			goto Exit;
		}
	}

	// Add the next DRN element

	if( pLFile->uiLfType == LF_CONTAINER)
	{
		FLMBYTE *	pElm = &pucBlk[ BH_OVHD ];
		
		// Set the nextDRN value in the block
		
		*pElm = BBE_FIRST_FLAG | BBE_LAST_FLAG;
		pElm[ BBE_KL ] = 4;
		pElm[ BBE_RL ] = 4;
		UD2FBA( (FLMUINT32)DRN_LAST_MARKER, &pElm[ BBE_KEY ]);
		UD2FBA( (FLMUINT32)pLFile->uiNextDrn, &pElm[ BBE_KEY + 4 ]);
		uiBlkPos += DRN_LAST_MARKER_LEN;
	}

	// Write the Last element marker
	
	pucBlk[ uiBlkPos] = BBE_FIRST_FLAG | BBE_LAST_FLAG;
	pucBlk[ uiBlkPos + BBE_KL] = 
	pucBlk[ uiBlkPos + BBE_RL] = 0;
	uiBlkPos += BBE_LEM_LEN;
	UW2FBA( (FLMUINT16)uiBlkPos, &pucBlk[ BH_ELM_END]);

	// Release the cache block, because we are done with it

	ScaReleaseCache( pSCache, FALSE);
	bReleaseCache = FALSE;

	// Update the logical save area and return

	if( RC_BAD( rc = flmLFileWrite( pDb, pLFile)))
	{
		goto Exit;
	}

Exit:

	if (bReleaseCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc: 	Retrieves the logical file header record from disk & updates LFILE.
		 	Called when it is discovered that the LFH for a
		 	particular logical file is out of date.
*****************************************************************************/
RCODE flmLFileRead(
	FDB *			pDb,
	LFILE *		pLFile)
{
	RCODE			rc = FERR_OK;
	SCACHE *		pSCache;
	FLMBOOL		bReleaseCache = FALSE;

	// Read in the block containing the logical file header

	if (RC_BAD( rc = ScaGetBlock( pDb, NULL, BHT_LFH_BLK,
								pLFile->uiBlkAddress, NULL, &pSCache)))
	{
		goto Exit;
	}
	bReleaseCache = TRUE;

	// Copy the LFH from the block to the LFD

	if( RC_BAD( rc = flmBufferToLFile( 
		&pSCache->pucBlk[ pLFile->uiOffsetInBlk],
		pLFile, pLFile->uiBlkAddress, pLFile->uiOffsetInBlk)))
	{
		goto Exit;
	}

Exit:

	if (bReleaseCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE flmBufferToLFile(
	FLMBYTE *	pucBuf,
	LFILE *		pLFile,
	FLMUINT		uiBlkAddress,
	FLMUINT		uiOffsetInBlk)
{
	RCODE			rc = FERR_OK;

	pLFile->uiBlkAddress	= uiBlkAddress;
	pLFile->uiOffsetInBlk= uiOffsetInBlk;

	if( (pLFile->uiLfType = (FLMUINT)pucBuf[ LFH_TYPE_OFFSET]) == LF_INVALID)
	{
		pLFile->uiLfType = LF_INVALID;
		goto Exit;
	}

	pLFile->uiLfNum = (FLMUINT)FB2UW( &pucBuf[ LFH_LF_NUMBER_OFFSET]);
	pLFile->uiRootBlk = (FLMUINT)FB2UD( &pucBuf[ LFH_ROOT_BLK_OFFSET]);
	pLFile->uiNextDrn = (FLMUINT)FB2UD( &pucBuf[ LFH_NEXT_DRN_OFFSET]);

Exit:

	return( rc);
}

/***************************************************************************
Desc: 	Update the LFILE data on disk
*****************************************************************************/
RCODE flmLFileWrite(
	FDB *			pDb,
	LFILE *		pLFile)
{
	RCODE			rc = FERR_OK;
	SCACHE *		pSCache;
	FLMBOOL		bReleaseCache = FALSE;

	// Read in the block containing the logical file header

	if( RC_BAD( rc = ScaGetBlock( pDb, NULL, BHT_LFH_BLK,
							pLFile->uiBlkAddress, NULL, &pSCache)))
	{
		goto Exit;
	}
	bReleaseCache = TRUE;

	// Log the block before modifying it

	if( RC_BAD( rc = ScaLogPhysBlk( pDb, &pSCache)))
	{
		goto Exit;
	}

	// Now modify the block and set its status to dirty

	if( RC_BAD( rc = flmLFileToBuffer( pLFile, 
		&pSCache->pucBlk[ pLFile->uiOffsetInBlk])))
	{
		goto Exit;
	}

Exit:

	if( bReleaseCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc:		Copy the data from the LFILE OUT into the disk block buffer.
			Supports VER11 and VER15 formats.
*****************************************************************************/
FSTATIC RCODE flmLFileToBuffer(
	LFILE *		pLFile,
	FLMBYTE *	pucBuf)
{
	RCODE			rc = FERR_OK;

	// If deleted, fill with 0xFF, except for type - it is set below

	if( pLFile->uiLfType == LF_INVALID)
	{
		f_memset( pucBuf, 0xFF, LFH_SIZE );
		pucBuf[ LFH_TYPE_OFFSET ] = LF_INVALID;
		goto Exit;
	}

	UW2FBA( (FLMUINT16) pLFile->uiLfNum, &pucBuf[ LFH_LF_NUMBER_OFFSET]);
	pucBuf[ LFH_TYPE_OFFSET] = (FLMBYTE) pLFile->uiLfType;
	UD2FBA( (FLMUINT32)pLFile->uiRootBlk, &pucBuf[ LFH_ROOT_BLK_OFFSET]);
	UD2FBA( (FLMUINT32)pLFile->uiNextDrn, &pucBuf[ LFH_NEXT_DRN_OFFSET]);

	// Set these for backwards compatibility.

	pucBuf[ LFH_STATUS_OFFSET] = 0;
	pucBuf[ LFH_MIN_FILL_OFFSET] = (FLMBYTE)(FFILE_MIN_FILL * 128 / 100);
	pucBuf[ LFH_MAX_FILL_OFFSET] = (FLMBYTE)(FFILE_MAX_FILL * 128 / 100);
	
Exit:

	return( rc);
}

/***************************************************************************
Desc: 	Creates and initializes a LFILE structure on disk and in memory.
*****************************************************************************/
RCODE flmLFileCreate(
	FDB *			pDb,
	LFILE *		pLFile,
	FLMUINT		uiLfNum,
	FLMUINT		uiLfType)
{
	RCODE			rc = FERR_OK;
	SCACHE *		pNewSCache;
	SCACHE *		pSCache = NULL;
	FLMBYTE *	pucBlk;
	FLMUINT		uiBlkAddress = 0;
	FLMUINT		uiNextBlkAddress;
	FLMUINT		uiEndPos = 0;
	FLMUINT		uiPos = 0;
	FLMBOOL		bReleaseCache2 = FALSE;
	FLMBOOL		bReleaseCache = FALSE;

	// Find an available slot to create the LFH -- follow the linked list 
	// of LFH blocks to find one.  Supports uiFirstLFHBlkAddr to be BT_END
	// so this routine can create the first block.
	
	for( uiNextBlkAddress = pDb->pFile->FileHdr.uiFirstLFHBlkAddr, pucBlk = NULL;
 		(uiNextBlkAddress != BT_END) && (uiNextBlkAddress != 0 ); )
	{
		if (bReleaseCache)
		{
			ScaReleaseCache( pSCache, FALSE);
			bReleaseCache = FALSE;
		}
		uiBlkAddress = uiNextBlkAddress;
	
		if( RC_BAD( rc = ScaGetBlock( pDb, NULL, BHT_LFH_BLK,
								uiBlkAddress, NULL, &pSCache)))
		{
			goto Exit;
		}
		bReleaseCache = TRUE;

		pucBlk = pSCache->pucBlk;
		uiNextBlkAddress = FB2UD( &pucBlk[ BH_NEXT_BLK]);
		uiEndPos = (FLMUINT) FB2UW( &pucBlk[ BH_ELM_END]);

		if( (uiPos = flmLFileFindEmpty( pucBlk)) != 0)
		{
			break;
	}
	}

	// pucBlk will be defined unless the file header is corrupt
	
	if( !pucBlk)
	{
		rc = RC_SET( FERR_DATA_ERROR);
		goto Exit;
	}

	// If we did not find a deleted slot we can use, see if there
	// is room for a new logical file in the last block
	// in the chain.  If not, allocate a new block.

	if( !uiPos)
	{
		uiEndPos = (FLMUINT)FB2UW( &pucBlk[ BH_ELM_END]);
			
		// Allocate a new block?
		
		if( uiEndPos + LFH_SIZE >= pDb->pFile->FileHdr.uiBlockSize)
		{
			if( RC_BAD( rc = ScaCreateBlock( pDb, NULL, &pNewSCache)))
			{
				goto Exit;
			}
			bReleaseCache2 = TRUE;
			
			pucBlk = pNewSCache->pucBlk;
			uiNextBlkAddress = GET_BH_ADDR( pucBlk);

			// Modify the new block's next pointer and other fields

			UD2FBA( (FLMUINT32) BT_END, &pucBlk[ BH_NEXT_BLK]);
			pucBlk[ BH_TYPE] = BHT_LFH_BLK;
			pucBlk[ BH_LEVEL] = 0;
			UD2FBA( uiBlkAddress, &pucBlk[ BH_PREV_BLK]);
			UW2FBA( BH_OVHD, &pucBlk[ BH_ELM_END]);
			UW2FBA( 0, &pucBlk[ BH_LOG_FILE_NUM]);

			if( RC_BAD( rc = ScaLogPhysBlk( pDb, &pSCache)))
			{
				goto Exit;
			}
			pucBlk = pSCache->pucBlk;
			UD2FBA( uiNextBlkAddress, &pucBlk[ BH_NEXT_BLK]);

			// Set everything up so we are pointing to the new block

			ScaReleaseCache( pSCache, FALSE);
			pSCache = pNewSCache;
			bReleaseCache2 = FALSE;
			pucBlk = pSCache->pucBlk;
			uiEndPos = (FLMUINT) FB2UW( &pucBlk[ BH_ELM_END]);
			uiBlkAddress = uiNextBlkAddress;
		}

		// Modify the end of block pointer -- log block before modifying

		uiPos = uiEndPos;
		uiEndPos += LFH_SIZE;
	}

	// Call memset to ensure unused bytes are zero. 
	// pucBlk, uiPos and uiEndPos should ALL be set.
	
	if( RC_BAD( rc = ScaLogPhysBlk( pDb, &pSCache)))
	{
		goto Exit;
	}

	pucBlk = pSCache->pucBlk;
	f_memset( &pucBlk[ uiPos], 0, LFH_SIZE);
	
	UW2FBA( (FLMUINT16)uiEndPos, &pucBlk[ BH_ELM_END]);

	// Done with block in this routine

	ScaReleaseCache( pSCache, FALSE);
	bReleaseCache = FALSE;

	// Set the variables in the LFILE structure to later save to disk

	pLFile->uiLfNum = uiLfNum;
	pLFile->uiLfType = uiLfType;
	pLFile->uiBlkAddress	= uiBlkAddress;
	pLFile->uiOffsetInBlk = uiPos;
	pLFile->uiRootBlk = (FLMUINT)BT_END;
	pLFile->uiNextDrn = 1;
	pLFile->pIxd = NULL;
	
	if( RC_BAD( rc = flmLFileWrite( pDb, pLFile)))
	{
		goto Exit;
	}

Exit:
	
	if( bReleaseCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	if( bReleaseCache2)
	{
		ScaReleaseCache( pNewSCache, FALSE);
	}

	return( rc);
}
