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

/*****************************************************************************
Desc:	Retrieves a key from an index.
******************************************************************************/
RCODE	F_Db::keyRetrieve(
	FLMUINT					uiIndex,
	F_DataVector *			pSearchKey,
	FLMUINT					uiFlags,
	F_DataVector *			pFoundKey)
{
	RCODE				rc = NE_SFLM_OK;
	F_INDEX *		pIndex = NULL;
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
	FLMUINT			uiIdMatchFlags = uiFlags & FLM_MATCH_ROW_ID;
	IXKeyCompare	compareObject;
	FLMBOOL			bCompareRowId = FALSE;

	// See if the database is being forced to close

	if (RC_BAD( rc = checkState( __FILE__, __LINE__)))
	{
		goto Exit;
	}

	// If we are not in a transaction, we cannot read.

	if (m_eTransType == SFLM_NO_TRANS)
	{
		if( RC_BAD( rc = transBegin( SFLM_READ_TRANS)))
		{
			goto Exit;
		}

		bStartedTrans = TRUE;
	}

	// See if we have a transaction going which should be aborted.

	if( RC_BAD( m_AbortRc))
	{
		rc = RC_SET( NE_SFLM_ABORT_TRANS);
		goto Exit;
	}

	// Allocate key buffers.

	if (pSearchKey)
	{
		if (RC_BAD( rc = m_tempPool.poolAlloc( SFLM_MAX_KEY_SIZE,
												(void **)&pucSearchKey)))
		{
			goto Exit;
		}
	}

	if (RC_BAD( rc = m_tempPool.poolAlloc( SFLM_MAX_KEY_SIZE, (void **)&pucFoundKey)))
	{
		goto Exit;
	}

	// Make sure it is a valid index definition

	pIndex = m_pDict->getIndex( uiIndex);
	pLFile = &pIndex->lfInfo;

	if (RC_BAD( rc = flushKeys()))
	{
		goto Exit;
	}

	if (uiFlags & FLM_FIRST || (!pSearchKey &&
				!(uiFlags & FLM_FIRST) && !(uiFlags & FLM_LAST)))
	{
		uiOriginalFlags = uiFlags = FLM_FIRST;
	}
	else if (uiFlags & FLM_LAST)
	{
		uiOriginalFlags = uiFlags = FLM_LAST;
	}
	else
	{
		uiOriginalFlags = uiFlags;
		if (uiFlags & FLM_EXACT)
		{
			flmAssert( !(uiFlags & FLM_KEY_EXACT));
			uiFlags = FLM_EXACT;
		}
		else if (uiFlags & FLM_EXCL)
		{
			uiFlags = FLM_EXCL;
		}
		else
		{
			uiFlags = FLM_INCL;
		}

		if (RC_BAD( rc = pSearchKey->outputKey( this, uiIndex, uiIdMatchFlags,
				pucSearchKey, SFLM_MAX_KEY_SIZE, &uiSearchKeyLen, SEARCH_KEY_FLAG)))
		{
			goto Exit;
		}

		// If we are not matching on the IDs and this is an FLM_EXCL
		// search, tack on a 0xFF for the IDs, which should get us past
		// all keys that match.  We need to turn on the match IDs flags
		// in this case so that the comparison routine will match on the
		// 0xFF.

		if (!uiIdMatchFlags && (uiFlags & FLM_EXCL))
		{
			pucSearchKey [uiSearchKeyLen++] = 0xFF;
			bCompareRowId = TRUE;
		}
		else
		{
			if (uiIdMatchFlags & FLM_MATCH_ROW_ID)
			{
				bCompareRowId = TRUE;
			}
		}
	}

	compareObject.setIxInfo( this, pIndex);
	compareObject.setCompareRowId( bCompareRowId);
	compareObject.setSearchKey( pSearchKey);
	
	// Get a btree

	if (RC_BAD( rc = gv_SFlmSysData.pBtPool->btpReserveBtree( &pbtree)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pbtree->btOpen( this, pLFile,
		(pIndex->uiFlags & IXD_ABS_POS)
		? TRUE
		: FALSE,
		(pIndex->pDataIcds)
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
		pucFoundKey, SFLM_MAX_KEY_SIZE, &uiFoundKeyLen, uiFlags, NULL,
		&uiDataLen)))
	{
		if (rc == NE_SFLM_EOF_HIT && uiOriginalFlags & FLM_EXACT)
		{
			rc = RC_SET( NE_SFLM_NOT_FOUND);
		}
		goto Exit;
	}

	// See if we are in the same key

	if (uiOriginalFlags & FLM_KEY_EXACT)
	{
		FLMINT	iTmpCmp;
		
		if (RC_BAD( rc = ixKeyCompare( this, pIndex,
						(uiIdMatchFlags == FLM_MATCH_ROW_ID) ? TRUE : FALSE,
						pSearchKey, NULL,
						pucFoundKey, uiFoundKeyLen,
						pSearchKey, NULL,
						pucSearchKey, uiSearchKeyLen, &iTmpCmp)))
		{
			goto Exit;
		}
									
		if (iTmpCmp != 0)
		{
			rc = (uiOriginalFlags & (FLM_INCL | FLM_EXCL))
				? RC_SET( NE_SFLM_EOF_HIT)
				: RC_SET( NE_SFLM_NOT_FOUND);
			goto Exit;
		}
	}

	// Parse the found key into its individual components

	if (pFoundKey)
	{
		pFoundKey->reset();
		if (RC_BAD( rc = pFoundKey->inputKey( this, uiIndex,
											pucFoundKey, uiFoundKeyLen)))
		{
			goto Exit;
		}

		// See if there is a data part

		if (pIndex->pDataIcds)
		{
			FLMUINT		uiDataBufSize;
			FLMBYTE *	pucData;

			// If the data will fit in the search key buffer, just
			// reuse it since we are not going to do anything with
			// it after this.  Otherwise, allocate a new buffer.

			if (uiDataLen <= SFLM_MAX_KEY_SIZE && pucSearchKey)
			{
				uiDataBufSize = SFLM_MAX_KEY_SIZE;
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
				pucFoundKey, SFLM_MAX_KEY_SIZE, uiFoundKeyLen,
				pucData, uiDataBufSize, &uiDataLen)))
			{
				goto Exit;
			}

			// Parse the data

			if (RC_BAD( rc = pFoundKey->inputData( this, uiIndex,
													pucData, uiDataLen)))
			{
				goto Exit;
			}
		}
	}

Exit:

	m_tempPool.poolReset( pvMark);

	if (pbtree)
	{
		gv_SFlmSysData.pBtPool->btpReturnBtree( &pbtree);
	}

	if( bStartedTrans)
	{
		transAbort();
	}

	return( rc);
}
