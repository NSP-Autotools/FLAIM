//------------------------------------------------------------------------------
// Desc:	This file contains the F_Db::keyRetrieve method.
// Tabs:	3
//
// Copyright (c) 1998-2007 Novell, Inc. All Rights Reserved.
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

/*API~***********************************************************************
Area :	RETRIEVAL
Desc:		Retrieves a key from an index based on a passed-in GEDCOM tree and DRN.
*END************************************************************************/
RCODE	F_Db::keyRetrieve(
	FLMUINT					uiIndex,
		// [IN] Index number.
	IF_DataVector *		ifpSearchKey,
		// [IN] Pointer to the search key.
	FLMUINT					uiFlags,
		// [IN] Flag (XFLM_FIRST, XFLM_LAST, XFLM_EXCL, XFLM_INCL, 
		// XFLM_EXACT, XFLM_KEY_EXACT
	IF_DataVector *		ifpFoundKey
		// [OUT] Returns key that was found - may also have data components.
	)
{
	RCODE				rc = NE_XFLM_OK;
	IXD *				pIxd = NULL;
	LFILE *			pLFile;
	FLMBYTE *		pucSearchKey = NULL;
	FLMBYTE *		pucFoundKey = NULL;
	void *			pvMark = m_tempPool.poolMark();
	FLMUINT			uiSearchKeyLen = 0;
	FLMUINT			uiFoundKeyLen;
	FLMUINT			uiOriginalFlags;
	F_Btree *		pbtree = NULL;
	FLMUINT			uiDataLen;
	FLMBOOL			bStartedTrans = FALSE;
	FLMUINT			uiIdMatchFlags = uiFlags & (XFLM_MATCH_IDS | XFLM_MATCH_DOC_ID);
	IXKeyCompare	compareObject;
	FLMBOOL			bCompareDocId = FALSE;
	FLMBOOL			bCompareNodeIds = FALSE;
	F_DataVector *	pSearchKey = (F_DataVector *)ifpSearchKey;
	F_DataVector *	pFoundKey = (F_DataVector *)ifpFoundKey;

	// See if the database is being forced to close

	if (RC_BAD( rc = checkState( __FILE__, __LINE__)))
	{
		goto Exit;
	}

	// If we are not in a transaction, we cannot read.

	if (m_eTransType == XFLM_NO_TRANS)
	{
		if( RC_BAD( rc = transBegin( XFLM_READ_TRANS)))
		{
			goto Exit;
		}

		bStartedTrans = TRUE;
	}

	// See if we have a transaction going which should be aborted.

	if( RC_BAD( m_AbortRc))
	{
		rc = RC_SET( NE_XFLM_ABORT_TRANS);
		goto Exit;
	}

	// Allocate key buffers.

	if (pSearchKey)
	{
		if (RC_BAD( rc = m_tempPool.poolAlloc( XFLM_MAX_KEY_SIZE,
												(void **)&pucSearchKey)))
		{
			goto Exit;
		}
	}

	if (RC_BAD( rc = m_tempPool.poolAlloc( XFLM_MAX_KEY_SIZE, (void **)&pucFoundKey)))
	{
		goto Exit;
	}

	// Make sure it is a valid index definition

	if (RC_BAD( rc = m_pDict->getIndex( uiIndex, &pLFile, &pIxd)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = flushKeys()))
	{
		goto Exit;
	}

	if (uiFlags & XFLM_FIRST || (!pSearchKey &&
				!(uiFlags & XFLM_FIRST) && !(uiFlags & XFLM_LAST)))
	{
		uiOriginalFlags = uiFlags = XFLM_FIRST;
	}
	else if (uiFlags & XFLM_LAST)
	{
		uiOriginalFlags = uiFlags = XFLM_LAST;
	}
	else
	{
		uiOriginalFlags = uiFlags;
		if( !(uiIdMatchFlags & XFLM_MATCH_IDS))
		{
			flmAssert( !(uiFlags & XFLM_KEY_EXACT));
			if (uiFlags & XFLM_EXCL)
			{
				uiFlags = XFLM_EXCL;
			}
			else if (uiFlags & XFLM_EXACT)
			{
				uiOriginalFlags = XFLM_EXACT | XFLM_KEY_EXACT;
				uiFlags = XFLM_INCL;
			}
			else
			{
				uiFlags = XFLM_INCL;
			}
		}
		else
		{
			if (uiFlags & XFLM_EXACT)
			{
				flmAssert( !(uiFlags & XFLM_KEY_EXACT));
				uiFlags = XFLM_EXACT;
			}
			else if (uiFlags & XFLM_EXCL)
			{
				uiFlags = XFLM_EXCL;
			}
			else
			{
				uiFlags = XFLM_INCL;
			}
		}

		if (RC_BAD( rc = pSearchKey->outputKey( pIxd, uiIdMatchFlags,
				pucSearchKey, XFLM_MAX_KEY_SIZE, &uiSearchKeyLen, SEARCH_KEY_FLAG)))
		{
			goto Exit;
		}

		// If we are not matching on the IDs and this is an XFLM_EXCL
		// search, tack on a 0xFF for the IDs, which should get us past
		// all keys that match.  We need to turn on the match IDs flags
		// in this case so that the comparison routine will match on the
		// 0xFF.

		if (!uiIdMatchFlags && (uiFlags & XFLM_EXCL))
		{
			pucSearchKey [uiSearchKeyLen++] = 0xFF;
			bCompareDocId = TRUE;
			bCompareNodeIds = TRUE;
		}
		else
		{
			if (uiIdMatchFlags & XFLM_MATCH_IDS)
			{
				bCompareNodeIds = TRUE;
				bCompareDocId = TRUE;
			}
			else if (uiIdMatchFlags & XFLM_MATCH_DOC_ID)
			{
				bCompareDocId = TRUE;
			}
		}
	}

	compareObject.setIxInfo( this, pIxd);
	compareObject.setCompareNodeIds( bCompareNodeIds);
	compareObject.setCompareDocId( bCompareDocId);
	compareObject.setSearchKey( pSearchKey);
	
	// Get a btree

	if (RC_BAD( rc = gv_XFlmSysData.pBtPool->btpReserveBtree( &pbtree)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pbtree->btOpen( this, pLFile,
		(pIxd->uiFlags & IXD_ABS_POS)
		? TRUE
		: FALSE,
		(pIxd->pFirstData)
		? TRUE
		: FALSE, &compareObject)))
	{
		goto Exit;
	}

	// Search the for the key

	if (uiSearchKeyLen)
	{
		f_memcpy( pucFoundKey, pucSearchKey, uiSearchKeyLen);
	}
	uiFoundKeyLen = uiSearchKeyLen;

	if( RC_BAD( rc = pbtree->btLocateEntry(
		pucFoundKey, XFLM_MAX_KEY_SIZE, &uiFoundKeyLen, uiFlags, NULL,
		&uiDataLen)))
	{
		if (rc == NE_XFLM_EOF_HIT && uiOriginalFlags & XFLM_EXACT)
		{
			rc = RC_SET( NE_XFLM_NOT_FOUND);
		}
		goto Exit;
	}

	// See if we are in the same key

	if (uiOriginalFlags & XFLM_KEY_EXACT)
	{
		FLMINT	iTmpCmp;
		
		if (RC_BAD( rc = ixKeyCompare( this, pIxd,
						(F_DataVector *)pSearchKey, NULL, NULL,
						(uiIdMatchFlags == XFLM_MATCH_DOC_ID) ? TRUE : FALSE,
						FALSE, pucFoundKey, uiFoundKeyLen,
						pucSearchKey, uiSearchKeyLen, &iTmpCmp)))
		{
			goto Exit;
		}
									
		if (iTmpCmp != 0)
		{
			rc = (uiOriginalFlags & (XFLM_INCL | XFLM_EXCL))
				? RC_SET( NE_XFLM_EOF_HIT)
				: RC_SET( NE_XFLM_NOT_FOUND);
			goto Exit;
		}
	}

	// Parse the found key into its individual components

	if (pFoundKey)
	{
		pFoundKey->reset();
		if (RC_BAD( rc = pFoundKey->inputKey( pIxd, pucFoundKey, uiFoundKeyLen)))
		{
			goto Exit;
		}

		// See if there is a data part

		if (pIxd->pFirstData)
		{
			FLMUINT		uiDataBufSize;
			FLMBYTE *	pucData;

			// If the data will fit in the search key buffer, just
			// reuse it since we are not going to do anything with
			// it after this.  Otherwise, allocate a new buffer.

			if (uiDataLen <= XFLM_MAX_KEY_SIZE && pucSearchKey)
			{
				uiDataBufSize = XFLM_MAX_KEY_SIZE;
				pucData = pucSearchKey;
			}
			else
			{
				uiDataBufSize = uiDataLen;
				if (RC_BAD( rc = m_tempPool.poolAlloc( uiDataBufSize,
											(void **)&pucData)))
				{
					goto Exit;
				}
			}

			// Retrieve the data

			if (RC_BAD( rc = pbtree->btGetEntry(
				pucFoundKey, XFLM_MAX_KEY_SIZE, uiFoundKeyLen,
				pucData, uiDataBufSize, &uiDataLen)))
			{
				goto Exit;
			}

			// Parse the data

			if (RC_BAD( rc = pFoundKey->inputData( pIxd, pucData, uiDataLen)))
			{
				goto Exit;
			}
		}
	}

Exit:

	m_tempPool.poolReset( pvMark);

	if (pbtree)
	{
		gv_XFlmSysData.pBtPool->btpReturnBtree( &pbtree);
	}

	if( bStartedTrans)
	{
		transAbort();
	}

	return( rc);
}
