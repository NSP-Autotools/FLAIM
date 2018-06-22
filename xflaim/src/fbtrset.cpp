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

	if (m_ppCollectionTbl)
	{
		FLMUINT	uiLoop;

		for (uiLoop = 0; uiLoop < BT_MAX_COLLECTION_TBL_SIZ; uiLoop++)
		{
			if (m_ppCollectionTbl[ uiLoop] != NULL)
			{
				BT_COLLECTION_XREF * pTmp;

				while (m_ppCollectionTbl[ uiLoop] != NULL)
				{
					pTmp = m_ppCollectionTbl[ uiLoop];
					m_ppCollectionTbl[ uiLoop] = pTmp->pNext;
					if (pTmp && pTmp->pCompare)
					{
						pTmp->pCompare->Release();
					}
					f_free( &pTmp);
				}
			}
		}
		f_free( &m_ppCollectionTbl);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_BtResultSet::getBTree(
	F_Db *		pSrcDb,
	IXD *			pSrcIxd,
	F_Btree **	ppBTree)
{
	RCODE							rc = NE_XFLM_OK;
	FLMUINT						uiCollHash;
	BT_COLLECTION_XREF *		pCollPtr = NULL;
	IF_RandomGenerator *		pRandGen  = NULL;
	F_Database *				pDatabase;
	FLMUINT						uiCollection;
	
	if (RC_BAD( rc = m_pBtPool->btpReserveBtree( ppBTree)))
	{
		goto Exit;
	}

	if (pSrcIxd)
	{
		if (!m_ppCollectionTbl)
		{
			if (RC_BAD( rc = f_calloc(
							BT_MAX_COLLECTION_TBL_SIZ * sizeof(BT_COLLECTION_XREF),
							&m_ppCollectionTbl)))
			{
				goto Exit;
			}
		}

		uiCollHash = pSrcIxd->uiIndexNum % BT_MAX_COLLECTION_TBL_SIZ;

		pCollPtr = m_ppCollectionTbl[ uiCollHash];

		// Verify that we have the right collection
		while (pCollPtr && pCollPtr->uiKeyNum != pSrcIxd->uiIndexNum)
		{
			pCollPtr = pCollPtr->pNext;
		}

		if (!pCollPtr)
		{
			pDatabase = m_pResultSetDb->m_pDatabase;

			// Will need a random number generator.
			
			if( RC_BAD( rc = FlmAllocRandomGenerator( &pRandGen)))
			{
				goto Exit;
			}

			pRandGen->setSeed( (FLMINT32)pSrcIxd->uiIndexNum);

			// Allocate a new collection key context and create a new
			// collection for it.

			if (RC_BAD( rc = f_calloc( sizeof(BT_COLLECTION_XREF), &pCollPtr)))
			{
				goto Exit;
			}

			// Insert into the table at the head of the list.

			pCollPtr->pCompare = NULL;
			pCollPtr->pNext = m_ppCollectionTbl[ uiCollHash];
			m_ppCollectionTbl[ uiCollHash] = pCollPtr;

TryAgain:

			// Randomly select a collection number to use.

			uiCollection = pRandGen->getUINT32( 100, XFLM_MAX_COLLECTION_NUM);

			// Check to see if it already exists.
			if (RC_BAD( rc = pDatabase->lFileCreate( m_pResultSetDb,
				&pCollPtr->Collection.lfInfo, &pCollPtr->Collection,
				uiCollection, XFLM_LF_COLLECTION, FALSE, TRUE,
				pSrcIxd->lfInfo.uiEncId)))
			{
				if (rc != NE_XFLM_EXISTS)
				{
					goto Exit;
				}
				rc = NE_XFLM_OK;
				goto TryAgain;
			}

			pCollPtr->uiKeyNum = pSrcIxd->uiIndexNum;
			pCollPtr->uiCollection = uiCollection;
			
			// Set up the comparison object.
			
			if ((pCollPtr->pCompare = f_new IXKeyCompare) == NULL)
			{
				rc = RC_SET( NE_XFLM_MEM);
				goto Exit;
			}
		}
		pCollPtr->pCompare->setIxInfo( pSrcDb, pSrcIxd);

		// Open the btree and use the specified collection.

		if (RC_BAD( rc = (*ppBTree)->btOpen( m_pResultSetDb,
													&pCollPtr->Collection.lfInfo,
													FALSE, TRUE, pCollPtr->pCompare)))
		{
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = (*ppBTree)->btOpen( m_pResultSetDb,
													&m_Collection.lfInfo,
													FALSE, TRUE, NULL)))
		{
			goto Exit;
		}
	}
	
Exit:

	if (pRandGen)
	{
		pRandGen->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_BtResultSet::addEntry(
	F_Db *		pSrcDb,
	IXD *			pSrcIxd,
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyLength,
	FLMBYTE *	pucEntry,
	FLMUINT		uiEntryLength)
{
	RCODE			rc = NE_XFLM_OK;
	F_Btree *	pBTree = NULL;

	if( RC_BAD( rc = getBTree( pSrcDb, pSrcIxd, &pBTree)))
	{
		goto Exit;
	}

	flmAssert( uiKeyLength <= XFLM_MAX_KEY_SIZE);

	if( RC_BAD( rc = pBTree->btInsertEntry( pucKey,
		uiKeyLength, pucEntry, uiEntryLength, TRUE, TRUE)))
	{
		if (rc == NE_XFLM_NOT_UNIQUE)
		{
			rc = NE_XFLM_OK;
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
	IXD *			pSrcIxd,
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyLength,
	FLMBYTE *	pucEntry,
	FLMUINT		uiEntryLength)
{
	RCODE			rc = NE_XFLM_OK;
	F_Btree *	pBTree = NULL;

	if (RC_BAD( rc = getBTree( pSrcDb, pSrcIxd, &pBTree)))
	{
		goto Exit;
	}

	flmAssert( uiKeyLength <= XFLM_MAX_KEY_SIZE);

	if( RC_BAD( rc = pBTree->btReplaceEntry( pucKey,
		uiKeyLength, pucEntry, uiEntryLength, TRUE, TRUE)))
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
	IXD *			pSrcIxd,
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyLength)
{
	RCODE			rc = NE_XFLM_OK;
	F_Btree *	pBTree = NULL;

	if (RC_BAD( rc = getBTree( pSrcDb, pSrcIxd, &pBTree)))
	{
		goto Exit;
	}

	flmAssert( uiKeyLength <= XFLM_MAX_KEY_SIZE);

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
	IXD *			pSrcIxd,
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyBufLen,
	FLMUINT *	puiKeyLen,
	FLMBYTE *	pucBuffer,
	FLMUINT		uiBufferLength,
	FLMUINT *	puiReturnLength)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiLengthRV;
	F_Btree *	pBTree = NULL;

	if( RC_BAD( rc = getBTree( pSrcDb, pSrcIxd, &pBTree)))
	{
		goto Exit;
	}

	flmAssert( uiKeyBufLen <= XFLM_MAX_KEY_SIZE);

	if( RC_BAD( rc = pBTree->btLocateEntry( pucKey, uiKeyBufLen, puiKeyLen,
		XFLM_EXACT, NULL, &uiLengthRV)))
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
	IXD *			pSrcIxd,
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyLength,
	FLMBYTE *	pucEntry,
	FLMUINT		uiEntryLength,
	FLMUINT *	puiReturnLength)
{
	RCODE			rc = NE_XFLM_OK;
	F_Btree *	pBTree = NULL;

	if( RC_BAD( rc = getBTree( pSrcDb, pSrcIxd, &pBTree)))
	{
		goto Exit;
	}

	flmAssert( uiKeyLength <= XFLM_MAX_KEY_SIZE);

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
	IXD *			pSrcIxd,
	F_Btree *	pBTree,
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyBufLen,
	FLMUINT *	puiKeyLen,
	FLMBYTE *	pucEntry,
	FLMUINT		uiEntryLength,
	FLMUINT *	puiReturnLength)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBOOL		bFreeBTree = FALSE;

	if(	!pBTree)
	{
		if( RC_BAD( rc = getBTree( pSrcDb, pSrcIxd, &pBTree)))
		{
			goto Exit;
		}
		bFreeBTree = TRUE;
	}

	flmAssert( uiKeyBufLen <= XFLM_MAX_KEY_SIZE);

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
	IXD *			pSrcIxd,
	F_Btree *	pBTree,
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyBufLen,
	FLMUINT *	puiKeyLen,
	FLMBYTE *	pucEntry,
	FLMUINT		uiEntryLength,
	FLMUINT *	puiReturnLength)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBOOL		bFreeBTree = FALSE;

	if( !pBTree)
	{
		if( RC_BAD( rc = getBTree( pSrcDb, pSrcIxd, &pBTree)))
		{
			goto Exit;
		}
		bFreeBTree = TRUE;
	}

	flmAssert( uiKeyBufLen <= XFLM_MAX_KEY_SIZE);

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
	IXD *			pSrcIxd,
	F_Btree *	pBTree,
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyBufLen,
	FLMUINT *	puiKeyLen,
	FLMBYTE *	pucEntry,
	FLMUINT		uiEntryLength,
	FLMUINT *	puiReturnLength)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBOOL		bFreeBTree = FALSE;

	if( !pBTree)
	{
		if( RC_BAD( rc = getBTree( pSrcDb, pSrcIxd, &pBTree)))
		{
			goto Exit;
		}
		bFreeBTree = TRUE;
	}

	flmAssert( uiKeyBufLen <= XFLM_MAX_KEY_SIZE);

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
	IXD *			pSrcIxd,
	F_Btree *	pBTree,
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyBufLen,
	FLMUINT *	puiKeyLen,
	FLMBYTE *	pucEntry,
	FLMUINT		uiEntryLength,
	FLMUINT *	puiReturnLength)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBOOL		bFreeBTree = FALSE;

	if( !pBTree)
	{
		if( RC_BAD( rc = getBTree( pSrcDb, pSrcIxd, &pBTree)))
		{
			goto Exit;
		}
		bFreeBTree = TRUE;
	}

	flmAssert( uiKeyBufLen <= XFLM_MAX_KEY_SIZE);

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
