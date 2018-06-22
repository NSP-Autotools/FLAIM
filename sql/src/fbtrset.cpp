//------------------------------------------------------------------------------
// Desc:	This file contains routines that implement a result set using
//			a temporary XFLAIM database.
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

/****************************************************************************
Desc:
****************************************************************************/
F_BtResultSet::~F_BtResultSet()
{
	// Free the collection table if it was ever created.

	if (m_ppTableTbl)
	{
		FLMUINT	uiLoop;

		for (uiLoop = 0; uiLoop < BT_MAX_TABLE_TBL_SIZ; uiLoop++)
		{
			if (m_ppTableTbl[ uiLoop] != NULL)
			{
				BT_TABLE_XREF * pTmp;

				while (m_ppTableTbl[ uiLoop] != NULL)
				{
					pTmp = m_ppTableTbl[ uiLoop];
					m_ppTableTbl[ uiLoop] = pTmp->pNext;
					if (pTmp && pTmp->pCompare)
					{
						pTmp->pCompare->Release();
					}
					f_free( &pTmp);
				}
			}
		}
		f_free( &m_ppTableTbl);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_BtResultSet::getBTree(
	F_Db *		pSrcDb,
	F_INDEX *	pSrcIndex,
	F_Btree **	ppBTree)
{
	RCODE							rc = NE_SFLM_OK;
	FLMUINT						uiCollHash;
	BT_TABLE_XREF *			pCollPtr = NULL;
	F_Database *				pDatabase;
	
	if (RC_BAD( rc = m_pBtPool->btpReserveBtree( ppBTree)))
	{
		goto Exit;
	}

	if (pSrcIndex)
	{
		if (!m_ppTableTbl)
		{
			if (RC_BAD( rc = f_calloc(
							BT_MAX_TABLE_TBL_SIZ * sizeof(BT_TABLE_XREF),
							&m_ppTableTbl)))
			{
				goto Exit;
			}
		}

		uiCollHash = pSrcIndex->uiIndexNum % BT_MAX_TABLE_TBL_SIZ;

		pCollPtr = m_ppTableTbl[ uiCollHash];

		// Verify that we have the right collection
		while (pCollPtr && pCollPtr->uiKeyNum != pSrcIndex->uiIndexNum)
		{
			pCollPtr = pCollPtr->pNext;
		}

		if (!pCollPtr)
		{
			pDatabase = m_pResultSetDb->m_pDatabase;

			// Allocate a new collection key context and create a new
			// collection for it.

			if (RC_BAD( rc = f_calloc( sizeof(BT_TABLE_XREF), &pCollPtr)))
			{
				goto Exit;
			}

			// Insert into the table at the head of the list.

			pCollPtr->pCompare = NULL;
			pCollPtr->pNext = m_ppTableTbl[ uiCollHash];
			m_ppTableTbl[ uiCollHash] = pCollPtr;

			// Check to see if it already exists.
			if (RC_BAD( rc = pDatabase->lFileCreate( m_pResultSetDb,
				&pCollPtr->table.lfInfo, 101, SFLM_LF_TABLE, FALSE, TRUE,
				pSrcIndex->lfInfo.uiEncDefNum)))
			{
				goto Exit;
			}

			pCollPtr->uiKeyNum = pSrcIndex->uiIndexNum;
			
			// Set up the comparison object.
			
			if ((pCollPtr->pCompare = f_new IXKeyCompare) == NULL)
			{
				rc = RC_SET( NE_SFLM_MEM);
				goto Exit;
			}
		}
		pCollPtr->pCompare->setIxInfo( pSrcDb, pSrcIndex);

		// Open the btree and use the specified collection.

		if (RC_BAD( rc = (*ppBTree)->btOpen( m_pResultSetDb,
													&pCollPtr->table.lfInfo,
													FALSE, TRUE, pCollPtr->pCompare)))
		{
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = (*ppBTree)->btOpen( m_pResultSetDb,
													&m_table.lfInfo,
													FALSE, TRUE, NULL)))
		{
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_BtResultSet::addEntry(
	F_Db *		pSrcDb,
	F_INDEX *	pSrcIndex,
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyLength,
	FLMBYTE *	pucEntry,
	FLMUINT		uiEntryLength)
{
	RCODE			rc = NE_SFLM_OK;
	F_Btree *	pBTree = NULL;

	if( RC_BAD( rc = getBTree( pSrcDb, pSrcIndex, &pBTree)))
	{
		goto Exit;
	}

	flmAssert( uiKeyLength <= SFLM_MAX_KEY_SIZE);

	if( RC_BAD( rc = pBTree->btInsertEntry( pucKey,
		uiKeyLength, pucEntry, uiEntryLength, TRUE, TRUE)))
	{
		if (rc == NE_SFLM_NOT_UNIQUE)
		{
			rc = NE_SFLM_OK;
		}
		else
		{
			goto Exit;
		}
	}

Exit:

	if (pBTree)
	{
		m_pBtPool->btpReturnBtree( &pBTree);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_BtResultSet::modifyEntry(
	F_Db *		pSrcDb,
	F_INDEX *	pSrcIndex,
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyLength,
	FLMBYTE *	pucEntry,
	FLMUINT		uiEntryLength)
{
	RCODE			rc = NE_SFLM_OK;
	F_Btree *	pBTree = NULL;

	if (RC_BAD( rc = getBTree( pSrcDb, pSrcIndex, &pBTree)))
	{
		goto Exit;
	}

	flmAssert( uiKeyLength <= SFLM_MAX_KEY_SIZE);

	if( RC_BAD( rc = pBTree->btReplaceEntry( pucKey, uiKeyLength,
		pucEntry, uiEntryLength, TRUE, TRUE)))
	{
		goto Exit;
	}

Exit:

	if( pBTree)
	{
		m_pBtPool->btpReturnBtree( &pBTree);
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_BtResultSet::deleteEntry(
	F_Db *		pSrcDb,
	F_INDEX *	pSrcIndex,
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyLength)
{
	RCODE			rc = NE_SFLM_OK;
	F_Btree *	pBTree = NULL;

	if (RC_BAD( rc = getBTree( pSrcDb, pSrcIndex, &pBTree)))
	{
		goto Exit;
	}

	flmAssert( uiKeyLength <= SFLM_MAX_KEY_SIZE);

	if (RC_BAD( rc = pBTree->btRemoveEntry( pucKey, uiKeyLength)))
	{
		goto Exit;
	}

Exit:

	if (pBTree)
	{
		m_pBtPool->btpReturnBtree( &pBTree);
	}
	return rc;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_BtResultSet::findEntry(
	F_Db *		pSrcDb,
	F_INDEX *	pSrcIndex,
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyBufLen,
	FLMUINT *	puiKeyLen,
	FLMBYTE *	pucBuffer,
	FLMUINT		uiBufferLength,
	FLMUINT *	puiReturnLength)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiLengthRV;
	F_Btree *	pBTree = NULL;

	if( RC_BAD( rc = getBTree( pSrcDb, pSrcIndex, &pBTree)))
	{
		goto Exit;
	}

	flmAssert( uiKeyBufLen <= SFLM_MAX_KEY_SIZE);

	if( RC_BAD( rc = pBTree->btLocateEntry( pucKey, uiKeyBufLen, puiKeyLen,
		FLM_EXACT, NULL, &uiLengthRV)))
	{
		goto Exit;
	}

	if( pucBuffer)
	{
		// Get the entry ...
		
		if( RC_BAD( rc = pBTree->btGetEntry( pucKey, uiKeyBufLen, *puiKeyLen,
			pucBuffer, uiBufferLength, puiReturnLength)))
		{
			goto Exit;
		}
	}
	else if( puiReturnLength)
	{
		*puiReturnLength = uiLengthRV;
	}

Exit:

	if( pBTree)
	{
		m_pBtPool->btpReturnBtree( &pBTree);
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_BtResultSet::getCurrent(
	F_Db *		pSrcDb,
	F_INDEX *	pSrcIndex,
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyLength,
	FLMBYTE *	pucEntry,
	FLMUINT		uiEntryLength,
	FLMUINT *	puiReturnLength)
{
	RCODE			rc = NE_SFLM_OK;
	F_Btree *	pBTree = NULL;

	if( RC_BAD( rc = getBTree( pSrcDb, pSrcIndex, &pBTree)))
	{
		goto Exit;
	}

	flmAssert( uiKeyLength <= SFLM_MAX_KEY_SIZE);

	if( RC_BAD( rc = pBTree->btGetEntry( pucKey, uiKeyLength, uiKeyLength,
		pucEntry, uiEntryLength, puiReturnLength)))
	{
		goto Exit;
	}

Exit:

	if( pBTree)
	{
		m_pBtPool->btpReturnBtree( &pBTree);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_BtResultSet::getNext(
	F_Db *		pSrcDb,
	F_INDEX *	pSrcIndex,
	F_Btree *	pBTree,
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyBufLen,
	FLMUINT *	puiKeyLen,
	FLMBYTE *	pucEntry,
	FLMUINT		uiEntryLength,
	FLMUINT *	puiReturnLength)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBOOL		bFreeBTree = FALSE;

	if(	!pBTree)
	{
		if( RC_BAD( rc = getBTree( pSrcDb, pSrcIndex, &pBTree)))
		{
			goto Exit;
		}
		bFreeBTree = TRUE;
	}

	flmAssert( uiKeyBufLen <= SFLM_MAX_KEY_SIZE);

	if( RC_BAD( rc = pBTree->btNextEntry( pucKey, uiKeyBufLen, puiKeyLen,
		puiReturnLength)))
	{
		goto Exit;
	}

	if( pucEntry)
	{
		if( RC_BAD( rc = pBTree->btGetEntry( pucKey, *puiKeyLen, *puiKeyLen,
			pucEntry, uiEntryLength, puiReturnLength)))
		{
			goto Exit;
		}
	}

Exit:

	if( bFreeBTree)
	{
		m_pBtPool->btpReturnBtree( &pBTree);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_BtResultSet::getPrev(
	F_Db *		pSrcDb,
	F_INDEX *	pSrcIndex,
	F_Btree *	pBTree,
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyBufLen,
	FLMUINT *	puiKeyLen,
	FLMBYTE *	pucEntry,
	FLMUINT		uiEntryLength,
	FLMUINT *	puiReturnLength)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBOOL		bFreeBTree = FALSE;

	if( !pBTree)
	{
		if( RC_BAD( rc = getBTree( pSrcDb, pSrcIndex, &pBTree)))
		{
			goto Exit;
		}
		bFreeBTree = TRUE;
	}

	flmAssert( uiKeyBufLen <= SFLM_MAX_KEY_SIZE);

	if( RC_BAD( rc = pBTree->btPrevEntry( pucKey, uiKeyBufLen, puiKeyLen,
		puiReturnLength)))
	{
		goto Exit;
	}

	if( pucEntry)
	{
		if( RC_BAD( rc = pBTree->btGetEntry( pucKey, *puiKeyLen, *puiKeyLen,
			pucEntry, uiEntryLength, puiReturnLength)))
		{
			goto Exit;
		}
	}

Exit:

	if( bFreeBTree)
	{
		m_pBtPool->btpReturnBtree( &pBTree);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_BtResultSet::getFirst(
	F_Db *		pSrcDb,
	F_INDEX *	pSrcIndex,
	F_Btree *	pBTree,
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyBufLen,
	FLMUINT *	puiKeyLen,
	FLMBYTE *	pucEntry,
	FLMUINT		uiEntryLength,
	FLMUINT *	puiReturnLength)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBOOL		bFreeBTree = FALSE;

	if( !pBTree)
	{
		if( RC_BAD( rc = getBTree( pSrcDb, pSrcIndex, &pBTree)))
		{
			goto Exit;
		}
		bFreeBTree = TRUE;
	}

	flmAssert( uiKeyBufLen <= SFLM_MAX_KEY_SIZE);

	pBTree->btResetBtree();

	if( RC_BAD( rc = pBTree->btFirstEntry( pucKey, uiKeyBufLen, puiKeyLen,
		puiReturnLength)))
	{
		goto Exit;
	}

	if( pucEntry)
	{
		if( RC_BAD( rc = pBTree->btGetEntry( pucKey, *puiKeyLen, *puiKeyLen,
			pucEntry, uiEntryLength, puiReturnLength)))
		{
			goto Exit;
		}
	}

Exit:

	if( bFreeBTree)
	{
		m_pBtPool->btpReturnBtree( &pBTree);
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_BtResultSet::getLast(
	F_Db *		pSrcDb,
	F_INDEX *	pSrcIndex,
	F_Btree *	pBTree,
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyBufLen,
	FLMUINT *	puiKeyLen,
	FLMBYTE *	pucEntry,
	FLMUINT		uiEntryLength,
	FLMUINT *	puiReturnLength)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBOOL		bFreeBTree = FALSE;

	if( !pBTree)
	{
		if( RC_BAD( rc = getBTree( pSrcDb, pSrcIndex, &pBTree)))
		{
			goto Exit;
		}
		bFreeBTree = TRUE;
	}

	flmAssert( uiKeyBufLen <= SFLM_MAX_KEY_SIZE);

	if( RC_BAD( rc = pBTree->btLastEntry( pucKey, uiKeyBufLen, puiKeyLen,
		puiReturnLength)))
	{
		goto Exit;
	}

	if( pucEntry)
	{
		if( RC_BAD( rc = pBTree->btGetEntry( pucKey, *puiKeyLen, *puiKeyLen,
			pucEntry, uiEntryLength, puiReturnLength)))
		{
			goto Exit;
		}
	}

Exit:

	if( bFreeBTree)
	{
		m_pBtPool->btpReturnBtree( &pBTree);
	}
	
	return( rc);
}
