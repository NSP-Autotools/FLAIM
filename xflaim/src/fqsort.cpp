//------------------------------------------------------------------------------
// Desc:	Contains the methods for doing sorting in the F_Query class.
// Tabs:	3
//
// Copyright (c) 2005-2007 Novell, Inc. All Rights Reserved.
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
#include "fquery.h"

FSTATIC RCODE fqGetDocId(
	IXD *					pIxd,
	const FLMBYTE *	pucKey,
	FLMUINT				uiKeyLen,
	FLMUINT64 *			pui64DocId);
	
/***************************************************************************
Desc:	Add a sort key to the query.
***************************************************************************/
RCODE XFLAPI F_Query::addSortKey(
	void *			pvSortKeyContext,
	FLMBOOL			bChildToContext,
	FLMBOOL			bElement,
	FLMUINT			uiDictNum,
	FLMUINT			uiCompareRules,
	FLMUINT			uiLimit,
	FLMUINT			uiKeyComponent,
	FLMBOOL			bSortDescending,
	FLMBOOL			bSortMissingHigh,
	void **			ppvContext)
{
	RCODE	rc = NE_XFLM_OK;
	ICD *	pSortIcd;
	ICD *	pSortIcdContext;
	ICD *	pTmpIcd;
	
	// If an error has already occurred, cannot add more to query.

	if (RC_BAD( rc = m_rc))
	{
		goto Exit;
	}

	if (m_bOptimized)
	{
		rc = RC_SET( NE_XFLM_Q_TOO_LATE_TO_ADD_SORT_KEYS);
		goto Exit;
	}
	
	// Verify that the sort key component number is legal
	
	if (uiKeyComponent > XFLM_MAX_SORT_KEYS)
	{
		rc = RC_SET( NE_XFLM_Q_INVALID_SORT_KEY_COMPONENT);
		goto Exit;
	}
	
	// If we have not created an IXD, create one now.
	
	if (!m_pSortIxd)
	{
		if (RC_BAD( rc = m_pool.poolCalloc( sizeof( IXD), (void **)&m_pSortIxd)))
		{
			goto Exit;
		}
		
		// Assume single path - can be unset later on.
		
		m_pSortIxd->uiFlags |= IXD_SINGLE_PATH;
		m_pSortIxd->uiCollectionNum = m_uiCollection;
	}
	
	// Allocate an ICD structure.
	
	if (RC_BAD( rc = m_pool.poolCalloc( sizeof( ICD),
										(void **)&pSortIcd)))
	{
		goto Exit;
	}
	
	pSortIcd->uiCdl = m_pSortIxd->uiNumIcds;
	m_pSortIxd->uiNumIcds++;
	pSortIcd->pIxd = m_pSortIxd;
	pSortIcd->uiIndexNum = m_pSortIxd->uiIndexNum;
	pSortIcd->uiDictNum = uiDictNum;
	if (!bElement)
	{
		pSortIcd->uiFlags |= ICD_IS_ATTRIBUTE;
	}
	if ((pSortIcd->uiKeyComponent = uiKeyComponent) > 0)
	{
		pSortIcd->uiFlags |= (ICD_VALUE | ICD_REQUIRED_PIECE | ICD_REQUIRED_IN_SET);
		pSortIcd->uiCompareRules = uiCompareRules;
		if (bSortDescending)
		{
			pSortIcd->uiFlags |= ICD_DESCENDING;
		}
		if (bSortMissingHigh)
		{
			pSortIcd->uiFlags |= ICD_MISSING_HIGH;
		}
		if (!uiLimit)
		{
			pSortIcd->uiLimit = ICD_DEFAULT_LIMIT;
		}
		else
		{
			pSortIcd->uiLimit = uiLimit;
		}
		
		m_pSortIxd->uiNumKeyComponents++;
		
		// Link into list of key components where it goes.
		
		pTmpIcd = m_pSortIxd->pFirstKey;
		while (pTmpIcd &&
					pSortIcd->uiKeyComponent > pTmpIcd->uiKeyComponent)
		{
			pTmpIcd = pTmpIcd->pNextKeyComponent;
		}
		
		// Cannot have two components with the same value.
		
		if (pTmpIcd && pSortIcd->uiKeyComponent == pTmpIcd->uiKeyComponent)
		{
			rc = RC_SET( NE_XFLM_Q_DUPLICATE_SORT_KEY_COMPONENT);
			goto Exit;
		}

		if ((pSortIcd->pNextKeyComponent = pTmpIcd) == NULL)
		{
			
			// Link at end of list.

			if ((pSortIcd->pPrevKeyComponent = m_pSortIxd->pLastKey) != NULL)
			{
				m_pSortIxd->pLastKey->pNextKeyComponent = pSortIcd;
			}
			else
			{
				m_pSortIxd->pFirstKey = pSortIcd;
			}
			m_pSortIxd->pLastKey = pSortIcd;
		}
		else
		{
			
			// Link in front of pTmpIcd

			flmAssert( pSortIcd->uiKeyComponent < pTmpIcd->uiKeyComponent);
			if ((pSortIcd->pPrevKeyComponent =
						pTmpIcd->pPrevKeyComponent) == NULL)
			{
				m_pSortIxd->pFirstKey = pSortIcd;
			}
			else
			{
				pTmpIcd->pPrevKeyComponent->pNextKeyComponent = pSortIcd;
			}
			pTmpIcd->pPrevKeyComponent = pSortIcd;
		}
	}
	else
	{
		m_pSortIxd->uiNumContextComponents++;
		if ((pSortIcd->pPrevKeyComponent = m_pSortIxd->pLastContext) == NULL)
		{
			m_pSortIxd->pFirstContext = pSortIcd;
		}
		m_pSortIxd->pLastContext = pSortIcd;
	}
	
	if ((pSortIcdContext = (ICD *)(pvSortKeyContext)) != NULL)
	{
		flmAssert( m_pSortIxd->pIcdTree);
		if (bChildToContext)
		{
			
			// Attributes cannot be parents to another sort ICD.
			
			if (pSortIcdContext->uiFlags & ICD_IS_ATTRIBUTE)
			{
				rc = RC_SET( NE_XFLM_Q_SORT_KEY_CONTEXT_MUST_BE_ELEMENT);
				goto Exit;
			}
			pSortIcd->pParent = pSortIcdContext;
			if ((pSortIcd->pNextSibling = pSortIcdContext->pFirstChild) != NULL)
			{
				pSortIcd->pNextSibling->pPrevSibling = pSortIcd;
			}
			pSortIcdContext->pFirstChild = pSortIcd;
			if (pSortIcdContext->pNextSibling || pSortIcd->pPrevSibling)
			{
				m_pSortIxd->uiFlags &= (~(IXD_SINGLE_PATH));
			}
		}
		else
		{
			pSortIcd->pParent = pSortIcdContext->pParent;
			pSortIcd->pPrevSibling = pSortIcdContext;
			if ((pSortIcd->pNextSibling = pSortIcdContext->pNextSibling) != NULL)
			{
				pSortIcd->pNextSibling->pPrevSibling = pSortIcd;
			}
			pSortIcdContext->pNextSibling = pSortIcd;
			if (pSortIcdContext->pFirstChild)
			{
				m_pSortIxd->uiFlags &= (~(IXD_SINGLE_PATH));
			}
		}
	}
	else if (m_pSortIxd->pIcdTree)
	{
		pSortIcdContext = m_pSortIxd->pIcdTree;
		while (pSortIcdContext->pNextSibling)
		{
			if (pSortIcdContext->pFirstChild)
			{
				m_pSortIxd->uiFlags &= (~(IXD_SINGLE_PATH));
			}
			pSortIcdContext = pSortIcdContext->pNextSibling;
		}
		if (pSortIcdContext->pFirstChild)
		{
			m_pSortIxd->uiFlags &= (~(IXD_SINGLE_PATH));
		}
		pSortIcdContext->pNextSibling = pSortIcd;
		pSortIcd->pPrevSibling = pSortIcdContext;
	}
	else
	{
		m_pSortIxd->pIcdTree = pSortIcd;
	}
	
	if (ppvContext)
	{
		*ppvContext = (void *)pSortIcd;
	}
	
Exit:

	m_rc = rc;
	return( rc);
}

/***************************************************************************
Desc:	Verify the sort keys, if any.  This method is called during
		optimization.
***************************************************************************/
RCODE F_Query::verifySortKeys( void)
{
	RCODE			rc = NE_XFLM_OK;
	ICD *			pSortIcd;
	FLMUINT		uiExpectedKeyComponent;
	
	// This routine should not be called if there were no sort keys specified.
	
	flmAssert( m_pSortIxd);
	
	if ((pSortIcd = m_pSortIxd->pFirstKey) == NULL)
	{
		
		// No sort keys were really specified.  The user called addSortKey
		// without ever setting one of the sort key components.
		
		m_pSortIxd = NULL;
		rc = RC_SET( NE_XFLM_Q_NO_SORT_KEY_COMPONENTS_SPECIFIED);
		goto Exit;
	}
	
	// Set the language for the sort keys
	
	m_pSortIxd->uiLanguage = m_pDb->m_pDatabase->m_uiDefaultLanguage;

	// Verify that we don't have any missing sort key components,
	// and that we can get all of their data types.

	uiExpectedKeyComponent = 1;
	while (pSortIcd)
	{
		if (pSortIcd->uiKeyComponent != uiExpectedKeyComponent)
		{
			rc = RC_SET( NE_XFLM_Q_MISSING_SORT_KEY_COMPONENT);
			goto Exit;
		}
		
		// Verify the dictionary number and get its data type.
		
		if (pSortIcd->uiDictNum != ELM_ROOT_TAG)
		{
			F_AttrElmInfo	info;
			
			if (!(pSortIcd->uiFlags & ICD_IS_ATTRIBUTE))
			{
				if (RC_BAD( rc = m_pDb->m_pDict->getElement( m_pDb,
											pSortIcd->uiDictNum, &info)))
				{
					if (rc == NE_XFLM_BAD_ELEMENT_NUM)
					{
						rc = RC_SET( NE_XFLM_Q_INVALID_ELEMENT_NUM_IN_SORT_KEYS);
					}
					goto Exit;
				}
			}
			else
			{
				if (RC_BAD( rc = m_pDb->m_pDict->getAttribute( m_pDb,
											pSortIcd->uiDictNum, &info)))
				{
					if (rc == NE_XFLM_BAD_ATTRIBUTE_NUM)
					{
						rc = RC_SET( NE_XFLM_Q_INVALID_ATTR_NUM_IN_SORT_KEYS);
					}
					goto Exit;
				}
			}
			icdSetDataType( pSortIcd, info.m_uiDataType);
		}
		
		// Go to the next sort key component
		
		pSortIcd = pSortIcd->pNextKeyComponent;
		uiExpectedKeyComponent++;
	}
	
Exit:

	return( rc);
}

/***************************************************************************
Desc:	Add the current document to the result set.
***************************************************************************/
RCODE F_Query::addToResultSet( void)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBYTE		ucKey [XFLM_MAX_SORT_KEYS * 3 + 20];
	FLMBYTE *	pucTmp;
	FLMUINT		uiKeyLen;
	FLMUINT64	ui64DocId;
	FLMUINT		uiIDLen;
	
	if (m_pSortIxd)
	{
		// This routine should only be called if we have sort keys.
		
		m_pSortResultSet->setIxInfo( m_pDb, m_pSortIxd);
		
		// Call index document to generate the sort keys for the document.
		
		if (RC_BAD( rc = m_pDb->indexDocument( m_pSortIxd, (F_DOMNode *)m_pCurrDoc)))
		{
			goto Exit;
		}
	
		// Take only the lowest sort key for the document.
		
		if (m_pDb->m_uiKrefCount)
		{
			KREF_ENTRY *	pKref;
														
			pKref = m_pDb->m_pKrefTbl [0];
			if (RC_BAD( rc = m_pSortResultSet->addEntry( (FLMBYTE *)&pKref [1],
										(FLMUINT)pKref->ui16KeyLen, TRUE)))
			{
				goto Exit;
			}
		}
		else
		{
			uiKeyLen = 2 * m_pSortIxd->uiNumKeyComponents;
			
			// Generate an empty key and add to the result set.
			
			f_memset( ucKey, 0, uiKeyLen);
			
			// Put in document ID - so we can pull it out when traversing
			// through the result set.
			
			if (RC_BAD( rc = m_pCurrDoc->getDocumentId( (IF_Db *)m_pDb, &ui64DocId)))
			{
				goto Exit;
			}
			pucTmp = &ucKey [uiKeyLen];
			uiIDLen = f_encodeSEN( ui64DocId, &pucTmp);
			
			// Output a zero SEN (which is one byte) for all of the other node IDs.
			
			f_memset( &ucKey [uiKeyLen + uiIDLen],
								0, m_pSortIxd->uiNumKeyComponents);
			uiIDLen += m_pSortIxd->uiNumKeyComponents;
			uiKeyLen += uiIDLen;
			
			if (RC_BAD( rc = m_pSortResultSet->addEntry( ucKey, uiKeyLen,
									TRUE)))
			{
				goto Exit;
			}
		}
		
		// Empty the table out for next document.
		
		m_pDb->m_pKrefPool->poolReset( NULL, TRUE);
		m_pDb->m_uiKrefCount = 0;
		m_pDb->m_uiTotalKrefBytes = 0;
	}
	else
	{
		FLMUINT32	ui32Count = (FLMUINT32)m_pSortResultSet->getCount();
		
		// If there is no sort key, the m_bPositioningEnabled flag better
		// be set - meaning we are just keeping a result set with the objects
		// in the order we found them.
		
		flmAssert( m_bPositioningEnabled);
		ui32Count++;
		f_UINT32ToBigEndian( ui32Count, ucKey);
		pucTmp = &ucKey [4];
		if (RC_BAD( rc = m_pCurrDoc->getDocumentId( (IF_Db *)m_pDb, &ui64DocId)))
		{
			goto Exit;
		}
		uiIDLen = f_encodeSEN( ui64DocId, &pucTmp);
		if (RC_BAD( rc = m_pSortResultSet->addEntry( ucKey, uiIDLen + 4,
								TRUE)))
		{
			goto Exit;
		}
	}
	m_ui64RSDocsPassed++;
	
Exit:

	return( rc);
}

/***************************************************************************
Desc:	Destructor for query result set.
***************************************************************************/
F_QueryResultSet::~F_QueryResultSet()
{
	if (m_pBTree)
	{
		m_pBTree->btClose();
		m_pBTree->Release();
	}
	if (m_pResultSetDb)
	{
		if (m_pResultSetDb->getTransType() != XFLM_NO_TRANS)
		{
			m_pResultSetDb->transAbort();
		}
		m_pResultSetDb->Release();
		m_pResultSetDb = NULL;
		gv_pXFlmDbSystem->dbRemove( m_szResultSetDibName, NULL, NULL, TRUE);
	}
	
	if (m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}
}

/***************************************************************************
Desc:	Initialize a query result set.
***************************************************************************/
RCODE F_QueryResultSet::initResultSet(
	FLMBOOL	bUseIxCompareObj,
	FLMBOOL	bEnableEncryption)
{
	RCODE					rc = NE_XFLM_OK;
	XFLM_CREATE_OPTS	createOpts;
	FLMUINT				uiNum = (FLMUINT)this;
	
	flmAssert( !m_pResultSetDb);
	
	// Create a mutex.
	
	if (RC_BAD( f_mutexCreate( &m_hMutex)))
	{
		goto Exit;
	}

	f_memset( &createOpts, 0, sizeof( XFLM_CREATE_OPTS));
	for (;;)
	{

		// Generate a random file name
		
		f_sprintf( m_szResultSetDibName, "%x.db", (unsigned)uiNum);
		if (RC_OK( rc = gv_pXFlmDbSystem->dbCreate( m_szResultSetDibName, 
			NULL, NULL, NULL, NULL, &createOpts, 
			TRUE, (IF_Db **)&m_pResultSetDb)))
		{
			break;
		}
		if (rc == NE_XFLM_FILE_EXISTS || rc == NE_FLM_IO_ACCESS_DENIED)
		{
			
			// Try again with a slightly altered number.
			
			uiNum -= 10;
			rc = NE_XFLM_OK;
		}
		else
		{
			goto Exit;
		}
	}

	// Shouldn't have an RFL object - don't want anything
	// logged by this database.
	
	flmAssert( !m_pResultSetDb->getDatabase()->m_pRfl);

	// Create a b-tree that will hold our result set data.
	// Will always be #1.
	// Should be one that keeps counts so we can do absolute positioning.

	if (RC_BAD( rc = m_pResultSetDb->m_pDatabase->lFileCreate( m_pResultSetDb,
		&m_LFile, NULL, 1, XFLM_LF_INDEX, TRUE, FALSE,
		(FLMUINT)(bEnableEncryption ? 1 : 0))))
	{
		goto Exit;
	}
	
	// Open a b-tree object to read from and write to the btree.
	
	if ((m_pBTree = f_new F_Btree) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	// Open the btree and use the specified collection.

	if (RC_BAD( rc = m_pBTree->btOpen( m_pResultSetDb,
													&m_LFile, TRUE, FALSE,
													bUseIxCompareObj
													? &m_compareObj
													: (IF_ResultSetCompare *)NULL)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}

/***************************************************************************
Desc:	Add an entry to the result set.
***************************************************************************/
RCODE F_QueryResultSet::addEntry(
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyLen,
	FLMBOOL		bLockMutex)
{
	RCODE	rc = NE_XFLM_OK;
	
	// May need to lock the mutex when adding an entry - this will hold off
	// any other thread that may be trying to read the result set at the
	// same time.
	
	if (bLockMutex)
	{
		lockMutex();
	}
	
	m_pBTree->btResetBtree();
	
	if (RC_BAD( rc = m_pBTree->btInsertEntry( pucKey, uiKeyLen, NULL,
								0, TRUE, TRUE)))
	{
		goto Exit;
	}
	m_uiCount++;
	m_bPositioned = FALSE;
	
Exit:

	if (bLockMutex)
	{
		unlockMutex();
	}

	return( rc);
}

/***************************************************************************
Desc:	Get the first entry in the result set.
***************************************************************************/
RCODE F_QueryResultSet::getFirst(
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyBufSize,
	FLMUINT *	puiKeyLen,
	FLMBOOL		bLockMutex)
{
	RCODE	rc = NE_XFLM_OK;
	
	if (bLockMutex)
	{
		lockMutex();
	}
	
	if (RC_BAD( rc = m_pBTree->btFirstEntry( pucKey, uiKeyBufSize, puiKeyLen)))
	{
		goto Exit;
	}
	
	// Remember our current position, so we can reposition there if necessary.
	
	if (RC_BAD( rc = m_pBTree->btGetPosition( pucKey, *puiKeyLen, &m_uiCurrPos)))
	{
		goto Exit;
	}
	m_bPositioned = TRUE;
	
Exit:

	if (bLockMutex)
	{
		unlockMutex();
	}
	return( rc);
}

/***************************************************************************
Desc:	Get the last entry in the result set.
***************************************************************************/
RCODE F_QueryResultSet::getLast(
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyBufSize,
	FLMUINT *	puiKeyLen,
	FLMBOOL		bLockMutex)
{
	RCODE	rc = NE_XFLM_OK;
	
	if (bLockMutex)
	{
		lockMutex();
	}
	if (RC_BAD( rc = m_pBTree->btLastEntry( pucKey, uiKeyBufSize, puiKeyLen)))
	{
		goto Exit;
	}
	
	// Remember our current position, so we can reposition there if necessary.
	
	if (RC_BAD( rc = m_pBTree->btGetPosition( pucKey, *puiKeyLen, &m_uiCurrPos)))
	{
		goto Exit;
	}
	m_bPositioned = TRUE;
	
Exit:

	if (bLockMutex)
	{
		unlockMutex();
	}
	return( rc);
}

/***************************************************************************
Desc:	Get the next entry from the current position in the result set.
***************************************************************************/
RCODE F_QueryResultSet::getNext(
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyBufSize,
	FLMUINT *	puiKeyLen,
	FLMBOOL		bLockMutex)
{
	RCODE	rc = NE_XFLM_OK;
	
	if (bLockMutex)
	{
		lockMutex();
	}
	if (m_uiCurrPos == FLM_MAX_UINT)
	{
		if (RC_BAD( rc = getFirst( pucKey, uiKeyBufSize, puiKeyLen, FALSE)))
		{
			goto Exit;
		}
	}
	else
	{
		
		// m_bPositioned will be set to FALSE if addEntry is called - because
		// that changes the positioning of the b-tree.  If that happens, we need
		// to reposition before calling btNextEntry.
		
		if (!m_bPositioned)
		{
			if (RC_BAD( rc = m_pBTree->btPositionTo( m_uiCurrPos,
													pucKey, uiKeyBufSize, puiKeyLen)))
			{
				goto Exit;
			}
		}
		if (RC_BAD( rc = m_pBTree->btNextEntry( pucKey, uiKeyBufSize, puiKeyLen)))
		{
			goto Exit;
		}
		
		// Remember our current position, so we can reposition there if necessary.
		
		if (RC_BAD( rc = m_pBTree->btGetPosition( pucKey, *puiKeyLen, &m_uiCurrPos)))
		{
			goto Exit;
		}
	}
	m_bPositioned = TRUE;
	
Exit:

	if (bLockMutex)
	{
		unlockMutex();
	}
	return( rc);
}

/***************************************************************************
Desc:	Get the previous entry from the current position in the result set.
***************************************************************************/
RCODE F_QueryResultSet::getPrev(
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyBufSize,
	FLMUINT *	puiKeyLen,
	FLMBOOL		bLockMutex)
{
	RCODE	rc = NE_XFLM_OK;
	
	if (bLockMutex)
	{
		lockMutex();
	}
	if (m_uiCurrPos == FLM_MAX_UINT)
	{
		if (RC_BAD( rc = getLast( pucKey, uiKeyBufSize, puiKeyLen, FALSE)))
		{
			goto Exit;
		}
	}
	else
	{
		
		// m_bPositioned will be set to FALSE if addEntry is called - because
		// that changes the positioning of the b-tree.  If that happens, we need
		// to reposition before calling btPrevEntry.
		
		if (!m_bPositioned)
		{
			if (RC_BAD( rc = m_pBTree->btPositionTo( m_uiCurrPos,
													pucKey, uiKeyBufSize, puiKeyLen)))
			{
				goto Exit;
			}
		}
		if (RC_BAD( rc = m_pBTree->btPrevEntry( pucKey, uiKeyBufSize, puiKeyLen)))
		{
			goto Exit;
		}
		
		// Remember our current position, so we can reposition there if necessary.
		
		if (RC_BAD( rc = m_pBTree->btGetPosition( pucKey, *puiKeyLen, &m_uiCurrPos)))
		{
			goto Exit;
		}
	}
	m_bPositioned = TRUE;
	
Exit:

	if (bLockMutex)
	{
		unlockMutex();
	}
	return( rc);
}

/***************************************************************************
Desc:	Get the entry from the current position in the result set.
***************************************************************************/
RCODE F_QueryResultSet::getCurrent(
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyBufSize,
	FLMUINT *	puiKeyLen,
	FLMBOOL		bLockMutex)	
{
	RCODE	rc = NE_XFLM_OK;
	
	if (bLockMutex)
	{
		lockMutex();
	}
	if (m_uiCurrPos == FLM_MAX_UINT)
	{
		rc = RC_SET( NE_XFLM_BOF_HIT);
		goto Exit;
	}
	else
	{
		if (RC_BAD( rc = m_pBTree->btPositionTo( m_uiCurrPos,
												pucKey, uiKeyBufSize, puiKeyLen)))
		{
			goto Exit;
		}
	}
	m_bPositioned = TRUE;

Exit:

	if (bLockMutex)
	{
		unlockMutex();
	}
	return( rc);
}

/***************************************************************************
Desc:	Position to a particular entry in the result set.
***************************************************************************/
RCODE F_QueryResultSet::positionToEntry(
	FLMBYTE *		pucKey,
	FLMUINT			uiKeyBufSize,
	FLMUINT *		puiKeyLen,
	F_DataVector *	pSearchKey,
	FLMUINT			uiFlags,
	FLMBOOL			bLockMutex)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBYTE	ucSearchKey [XFLM_MAX_KEY_SIZE];
	FLMUINT	uiSearchKeyLen = 0;
	FLMUINT	uiOriginalFlags;
	FLMUINT	uiIdMatchFlags = uiFlags & (XFLM_MATCH_IDS | XFLM_MATCH_DOC_ID);
	FLMBOOL	bCompareDocId = FALSE;
	FLMBOOL	bCompareNodeIds = FALSE;
	FLMUINT	uiPos;
	
	if (bLockMutex)
	{
		lockMutex();
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

		if (RC_BAD( rc = pSearchKey->outputKey( m_pIxd, uiIdMatchFlags,
				ucSearchKey, sizeof( ucSearchKey), &uiSearchKeyLen, SEARCH_KEY_FLAG)))
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
			ucSearchKey [uiSearchKeyLen++] = 0xFF;
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

	m_compareObj.setCompareNodeIds( bCompareNodeIds);
	m_compareObj.setCompareDocId( bCompareDocId);
	m_compareObj.setSearchKey( pSearchKey);
	
	// Search the for the key

	if (uiSearchKeyLen)
	{
		f_memcpy( pucKey, ucSearchKey, uiSearchKeyLen);
	}
	*puiKeyLen = uiSearchKeyLen;
	
	// NOTE: We don't pass m_uiCurrPos into btLocateEntry, because we may have to
	// test below to see if we really got to something we can stay on.
	// m_uiCurrPos should remain unchanged if we did not.

	if (RC_BAD( rc = m_pBTree->btLocateEntry(
		pucKey, uiKeyBufSize, puiKeyLen, uiFlags, &uiPos,
		NULL, NULL, NULL)))
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
		
		if (RC_BAD( rc = ixKeyCompare( m_pSrcDb, m_pIxd,
						pSearchKey, NULL, NULL,
						(uiIdMatchFlags == XFLM_MATCH_DOC_ID) ? TRUE : FALSE,
						FALSE, pucKey, *puiKeyLen,
						ucSearchKey, uiSearchKeyLen, &iTmpCmp)))
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

	m_bPositioned = TRUE;
	m_uiCurrPos = uiPos;

Exit:

	m_compareObj.setSearchKey( NULL);
	m_compareObj.setCompareNodeIds( FALSE);
	m_compareObj.setCompareDocId( FALSE);

	if (bLockMutex)
	{
		unlockMutex();
	}
	return( rc);
}

/***************************************************************************
Desc:	Position to a particular entry in the result set.
***************************************************************************/
RCODE F_QueryResultSet::positionToEntry(
	FLMUINT		uiPosition,
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyBufSize,
	FLMUINT *	puiKeyLen,
	FLMBOOL		bLockMutex)
{
	RCODE	rc = NE_XFLM_OK;
	
	if (bLockMutex)
	{
		lockMutex();
	}
	if (uiPosition >= m_uiCount)
	{
		rc = RC_SET( NE_XFLM_Q_INVALID_POSITION);
		goto Exit;
	}
	if (RC_BAD( rc = m_pBTree->btPositionTo( uiPosition,
											pucKey, uiKeyBufSize, puiKeyLen)))
	{
		goto Exit;
	}
	m_uiCurrPos = uiPosition;
	m_bPositioned = TRUE;

Exit:

	if (bLockMutex)
	{
		unlockMutex();
	}
	return( rc);
}

/***************************************************************************
Desc:	Determine how much time is remaining on timer.
***************************************************************************/
FINLINE RCODE getRemainingTimeMilli(
	FLMUINT		uiStartTimeTU,
	FLMUINT		uiTimeLimitTU,
	FLMUINT *	puiRemainingTimeMilli)
{
	RCODE		rc = NE_XFLM_OK;
	FLMUINT	uiCurrTimeTU;
	FLMUINT	uiElapsedTimeTU;
	FLMUINT	uiRemainingTimeTU;
	
	uiCurrTimeTU = FLM_GET_TIMER();
	uiElapsedTimeTU = FLM_ELAPSED_TIME( uiCurrTimeTU, uiStartTimeTU);
	if (uiElapsedTimeTU >= uiTimeLimitTU)
	{
		rc = RC_SET( NE_XFLM_TIMEOUT);
		goto Exit;
	}
	else
	{
		uiRemainingTimeTU = uiTimeLimitTU - uiElapsedTimeTU;
		*puiRemainingTimeMilli = FLM_TIMER_UNITS_TO_MILLI( uiRemainingTimeTU);
	}
Exit:
	return( rc);
}

/***************************************************************************
Desc:	Create and initialize the query result set.
***************************************************************************/
RCODE F_Query::createResultSet( void)
{
	RCODE	rc = NE_XFLM_OK;
	
	// Create a result set for the sort keys
	
	if ((m_pSortResultSet = f_new F_QueryResultSet) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	if (RC_BAD( rc = m_pSortResultSet->initResultSet(
										m_pSortIxd ? TRUE : FALSE,
										m_bEncryptResultSet)))
	{
		goto Exit;
	}

	m_ui64RSDocsRead = 0;
	m_ui64RSDocsPassed = 0;
	if (!m_pSortIxd)
	{
		m_bEntriesAlreadyInOrder = TRUE;
	}
	
Exit:

	return( rc);
}

/***************************************************************************
Desc:	Build the query result set.
***************************************************************************/
RCODE F_Query::buildResultSet(
	IF_Db *	pDb,
	FLMUINT	uiTimeLimit,
	FLMUINT	uiNumToWaitFor)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pNode = NULL;
	FLMUINT			uiStartTimeTU = 0;
	FLMUINT			uiRemainingTimeMilli = 0;
	FLMUINT			uiTimeLimitTU = 0;
	RS_WAITER *		pWaiter;
	FLMBOOL			bMutexLocked = FALSE;
	FLMBOOL			bDoneBuilding = FALSE;
	FLMBOOL			bNotifyWaiters = FALSE;
	
	// NOTE: uiTimeLimit will always be zero when building the result
	// set in another thread.
	
	if (uiTimeLimit)
	{
		uiTimeLimitTU = FLM_MILLI_TO_TIMER_UNITS( uiTimeLimit);
		uiStartTimeTU = FLM_GET_TIMER();
		uiRemainingTimeMilli = uiTimeLimit;
	}
	
	m_pSortResultSet->lockMutex();
	bMutexLocked = TRUE;
	
	// See if we have already built enough of the result set to get what
	// we need from it.
	
	if (m_pSortResultSet->getCount() >= uiNumToWaitFor || m_bResultSetPopulated)
	{
		goto Exit;
	}
		
	// Only one thread at a time can be actually be building the result set.
	// This for loop waits for such a thread to build what it needs, or, if
	// there is no thread running and the result set is not populated yet,
	// it will build the result set out to the point it needs.
	
	for (;;)
	{
	
		// If no other thread is building the result set, we can take over and
		// do it.  Otherwise, we will wait for that thread.
		
		if (!m_uiBuildThreadId)
		{
			m_uiBuildThreadId = f_threadId();
			bNotifyWaiters = TRUE;
			break;
		}
			
		// Thread is currently building the result set.  It cannot
		// be our thread.
		
		flmAssert( m_uiBuildThreadId != f_threadId());
		
		// Just need to wait now.
		
		if (RC_BAD( rc = waitResultSetBuild( pDb, uiTimeLimit, uiNumToWaitFor)))
		{
			goto Exit;
		}
		
		// See if the other thread built enough of the result set to get what
		// we need from it.
		
		if (m_pSortResultSet->getCount() >= uiNumToWaitFor || m_bResultSetPopulated)
		{
			goto Exit;
		}
		
		// At this point, we know that the result set is not built out far enough
		// for us to get what we need, so we need to adjust our timeout.
		
		if (uiTimeLimit)
		{
			if (RC_BAD( rc = getRemainingTimeMilli( uiStartTimeTU, uiTimeLimitTU,
										&uiRemainingTimeMilli)))
			{
				goto Exit;
			}
		}
	}
	
	if (bMutexLocked)
	{
		m_pSortResultSet->unlockMutex();
		bMutexLocked = FALSE;
	}
	
	for (;;)
	{
		if (m_bStopBuildingResultSet)
		{
			bDoneBuilding = TRUE;
			rc = RC_SET( NE_XFLM_USER_ABORT);
			break;
		}
		if (RC_BAD( rc = getNext( pDb, &pNode, uiRemainingTimeMilli, 0, NULL)))
		{
			if (rc == NE_XFLM_EOF_HIT)
			{
				bDoneBuilding = TRUE;
				rc = NE_XFLM_OK;
				break;
			}
			goto Exit;
		}
		
		// See how much time is left, if we are operating with a time limit.
		
		if (uiTimeLimit)
		{
			if (RC_BAD( rc = getRemainingTimeMilli( uiStartTimeTU, uiTimeLimitTU,
										&uiRemainingTimeMilli)))
			{
				goto Exit;
			}
		}
		
		// See if any of the waiters timed out.

		checkResultSetWaiters( NE_XFLM_OK);
		if (m_pSortResultSet->getCount() >= uiNumToWaitFor)
		{
			break;
		}
	}
	
Exit:

	if (pNode)
	{
		pNode->Release();
	}
	
	if (!bMutexLocked)
	{
		m_pSortResultSet->lockMutex();
		bMutexLocked = TRUE;
	}
	
	if (!m_bResultSetPopulated && bDoneBuilding)
	{
		m_bResultSetPopulated = TRUE;
		m_bPositioningEnabled = TRUE;
		if (RC_OK( rc))
		{
			if (m_pQueryStatus)
			{
				flmAssert( m_ui64RSDocsPassed == (FLMUINT64)m_pSortResultSet->getCount());
				rc = m_pQueryStatus->resultSetComplete(
										m_ui64RSDocsRead, m_ui64RSDocsPassed);
			}
		}
	}
	
	if (bNotifyWaiters)
	{
		
		// Notify any waiters that are waiting.
		
		pWaiter = m_pFirstWaiter;
		m_pFirstWaiter = NULL;
		while (pWaiter)
		{
			F_SEM	hESem = pWaiter->hESem;
			
			// Set waiter's return code to whatever we are returning.
			
			*(pWaiter->pRc) = rc;
			
			// Must get next waiter before signaling semaphore.
		
			pWaiter = pWaiter->pNext;
			f_semSignal( hESem);
		}
	}
	
	// Must set m_uiBuildThreadId to 0 inside the
	// mutex lock, because stopResultSetBuild() locks the mutex
	// one last time to ensure that this method is no longer
	// accessing anything inside the F_Query object.  That is
	// because it is fair game to destroy the F_Query object upon
	// returning from this method.
	// IMPORTANT NOTE: NOTHING SHOULD BE ACCESSED INSIDE THE F_Query
	// OBJECT AFTER THE MUTEX IS UNLOCKED.
		
	m_uiBuildThreadId = 0;
	
	if (bMutexLocked)
	{
		m_pSortResultSet->unlockMutex();
	}

	return( rc);
}

/***************************************************************************
Desc:	Stop building the result set.
***************************************************************************/
void XFLAPI F_Query::stopBuildingResultSet( void)
{
	if (m_pSortResultSet)
	{
		m_pSortResultSet->lockMutex();
		
		// If we have a thread currently building the
		// result set, signal it to stop.
		
		if (m_uiBuildThreadId)
		{
			m_bStopBuildingResultSet = TRUE;
			(void)waitResultSetBuild( m_pDb, 0, FLM_MAX_UINT);
			
			// waitResultSetBuild may temporarily unlock the
			// mutex, but it will always be relocked upon
			// returning.  When we return this time,
			// m_uiBuildThreadId should be zero.
			
			flmAssert( !m_uiBuildThreadId);
		}
		else
		{
			m_bResultSetPopulated = TRUE;
		}
		m_pSortResultSet->unlockMutex();
	}
}
		
/***************************************************************************
Desc:	Build the query result set.  This is the method that applications
		can call.  It implies enabling of positioning.
***************************************************************************/
RCODE XFLAPI F_Query::buildResultSet(
	IF_Db *	pDb,
	FLMUINT	uiTimeLimit)
{
	RCODE	rc = NE_XFLM_OK;
	
	m_pDb = (F_Db *)pDb;

	if (m_pDatabase && m_pDb->m_pDatabase != m_pDatabase)
	{

		// Make sure the passed in F_Db matches the one associated with
		// the query.

		rc = RC_SET( NE_XFLM_Q_MISMATCHED_DB);
		goto Exit;
	}
	
	// See if the database is being forced to close

	if (RC_BAD( rc = m_pDb->checkState( __FILE__, __LINE__)))
	{
		goto Exit;
	}

	// If we are not in a transaction, we cannot read.

	if (m_pDb->m_eTransType == XFLM_NO_TRANS)
	{
		rc = RC_SET( NE_XFLM_NO_TRANS_ACTIVE);
		goto Exit;
	}

	// See if we have a transaction going which should be aborted.

	if (RC_BAD( m_pDb->m_AbortRc))
	{
		rc = RC_SET( NE_XFLM_ABORT_TRANS);
		goto Exit;
	}

	if (!m_bOptimized)
	{
		m_bPositioningEnabled = TRUE;
		if (RC_BAD( rc = optimize()))
		{
			goto Exit;
		}
	}
	else if (!m_pSortResultSet)
	{
		
		// Cannot build a result set for a query that was not
		// set up to use result sets.
		
		rc = RC_SET( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}
	
	// If the query can never evaluate to TRUE, return EOF without
	// doing anything.

	if (m_bEmpty)
	{
		m_eState = XFLM_QUERY_AT_BOF;
		rc = RC_SET( NE_XFLM_BOF_HIT);
		goto Exit;
	}
	
	// At this point, we should have a result set created.
	
	flmAssert( m_pSortResultSet);
	
	// Build the entire result set.
	
	if (RC_BAD( rc = buildResultSet( pDb, uiTimeLimit, FLM_MAX_UINT)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}
		
/***************************************************************************
Desc:	Extract the document ID from the key.
***************************************************************************/
FSTATIC RCODE fqGetDocId(
	IXD *					pIxd,
	const FLMBYTE *	pucKey,
	FLMUINT				uiKeyLen,
	FLMUINT64 *			pui64DocId)
{
	RCODE					rc = NE_XFLM_OK;
	const FLMBYTE *	pucKeyEnd = pucKey + uiKeyLen;
	
	if (pIxd)
	{
		ICD *			pIcd = pIxd->pFirstKey;
		FLMUINT		uiComponentLen;
		
		// Skip past all key components.  Document ID will be right after them.
		
		while (pIcd && pucKey < pucKeyEnd)
		{
			if (pucKey + 2 > pucKeyEnd)
			{
				rc = RC_SET( NE_XFLM_BAD_COLLATED_KEY);
				goto Exit;
			}
			uiComponentLen = getKeyComponentLength( pucKey);
			pucKey += (2 + uiComponentLen);
			pIcd = pIcd->pNextKeyComponent;
		}
		
		if (pucKey >= pucKeyEnd)
		{
			rc = RC_SET( NE_XFLM_BAD_COLLATED_KEY);
			goto Exit;
		}
		
		// Should be positioned on document ID
		
		if (RC_BAD( rc = f_decodeSEN64( &pucKey, pucKeyEnd, pui64DocId)))
		{
			goto Exit;
		}
	}
	else
	{
		
		// If no sort index was defined, the key is a 4 byte sequence number,
		// followed by the document id stored as a SEN.  This is done so that
		// it can be sorted with a memcmp.

		pucKey += 4;
		if (pucKey >= pucKeyEnd)
		{
			rc = RC_SET( NE_XFLM_BAD_COLLATED_KEY);
			goto Exit;
		}
		if (RC_BAD( rc = f_decodeSEN64( &pucKey, pucKeyEnd, pui64DocId)))
		{
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}

/***************************************************************************
Desc:	Check result set waiters to see if any should be notified that they
		are done waiting or that they timed out or that there was an error.
***************************************************************************/
void F_Query::checkResultSetWaiters(
	RCODE	rc)
{
	RS_WAITER *	pWaiter;
	FLMUINT		uiCurrTime;
	F_SEM			hESem;
	
	if (m_pSortResultSet && !m_bResultSetPopulated && m_pFirstWaiter)
	{
		uiCurrTime = FLM_GET_TIMER();
		m_pSortResultSet->lockMutex();
		pWaiter = m_pFirstWaiter;
		while (pWaiter)
		{
			
			// First see if their count has been satisfied.
			
			if (RC_BAD( rc) ||
				 m_pSortResultSet->getCount() >= pWaiter->uiNumToWaitFor)
			{
				hESem = pWaiter->hESem;
				*(pWaiter->pRc) = rc;
				
				// Unlink the waiter from the list before signaling.
				
				if (pWaiter->pNext)
				{
					pWaiter->pNext->pPrev = pWaiter->pPrev;
				}
				if (pWaiter->pPrev)
				{
					pWaiter->pPrev->pNext = pWaiter->pNext;
				}
				else
				{
					m_pFirstWaiter = pWaiter->pNext;
				}
				pWaiter = pWaiter->pNext;
				f_semSignal( hESem);
			}
			else if (pWaiter->uiTimeLimit &&
						FLM_ELAPSED_TIME( uiCurrTime, pWaiter->uiWaitStartTime) >
							pWaiter->uiTimeLimit)
			{
				hESem = pWaiter->hESem;
				*(pWaiter->pRc) = RC_SET( NE_XFLM_TIMEOUT);
				
				// Unlink the waiter from the list before signaling.
				
				if (pWaiter->pNext)
				{
					pWaiter->pNext->pPrev = pWaiter->pPrev;
				}
				if (pWaiter->pPrev)
				{
					pWaiter->pPrev->pNext = pWaiter->pNext;
				}
				else
				{
					m_pFirstWaiter = pWaiter->pNext;
				}
				pWaiter = pWaiter->pNext;
				f_semSignal( hESem);
			}
			else
			{
				pWaiter = pWaiter->pNext;
			}
		}
		m_pSortResultSet->unlockMutex();
	}
}
	
/***************************************************************************
Desc:	Wait for the result set to get completely built.  This routine assumes
		that the result set mutex has already been locked.
***************************************************************************/
RCODE F_Query::waitResultSetBuild(
	IF_Db *		pDb,
	FLMUINT		uiTimeLimit,
	FLMUINT		uiNumToWaitFor)
{
	RCODE			rc = NE_XFLM_OK;
	RCODE			TempRc;
	RS_WAITER	waiter;
	FLMBOOL		bMutexLocked = TRUE;
	
	// See if we are still building the result set, or if our
	// count has been reached.
	
	if (m_pSortResultSet->getCount() >= uiNumToWaitFor || m_bResultSetPopulated)
	{
		goto Exit;
	}
	
	// This should only be called if there is actually a thread currently
	// building the result set.
	
	flmAssert( m_uiBuildThreadId && m_uiBuildThreadId != f_threadId());
	
	waiter.uiThreadId = f_threadId();
	waiter.hESem = ((F_Db *)pDb)->m_hWaitSem;
	waiter.pRc = &rc;
	
	if (uiTimeLimit)
	{
		waiter.uiWaitStartTime = FLM_GET_TIMER();
		waiter.uiTimeLimit = FLM_MILLI_TO_TIMER_UNITS( uiTimeLimit);
	}
	else
	{
		waiter.uiWaitStartTime = 0;
		waiter.uiTimeLimit = 0;
	}
	
	waiter.uiNumToWaitFor = uiNumToWaitFor;
	waiter.pPrev = NULL;
	
	if ((waiter.pNext = m_pFirstWaiter) != NULL)
	{
		m_pFirstWaiter->pPrev = &waiter;
	}
	
	m_pFirstWaiter = &waiter;
	rc = NE_XFLM_FAILURE;
	m_pSortResultSet->unlockMutex();
	bMutexLocked = FALSE;
	
	if (RC_BAD( TempRc = f_semWait( waiter.hESem, F_WAITFOREVER)))
	{
		flmAssert( 0);
		rc = TempRc;
	}
	else
	{

		// Process that signaled us better set the rc to something
		// besides NE_XFLM_FAILURE.

		if( rc == NE_XFLM_FAILURE)
		{
			flmAssert( 0);
		}
	}

Exit:

	if (!bMutexLocked)
	{
		m_pSortResultSet->lockMutex();
	}
	
	return( rc);
}
	
/***************************************************************************
Desc:	Get the first item in the result set.
***************************************************************************/
RCODE F_Query::getFirstFromResultSet(
	IF_Db *			ifpDb,
	IF_DOMNode **	ppNode,
	FLMUINT			uiTimeLimit)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBYTE		ucKey [XFLM_MAX_KEY_SIZE];
	FLMUINT		uiKeyLen;
	FLMUINT64	ui64DocId;
	FLMUINT		uiNumToWaitFor = 1;
	FLMUINT		uiStartTimeTU = 0;
	FLMUINT		uiRemainingTimeMilli = 0;
	FLMUINT		uiTimeLimitTU = 0;
	
	// NOTE: uiTimeLimit will always be zero when building the result
	// set in another thread.
	
	if (uiTimeLimit)
	{
		uiTimeLimitTU = FLM_MILLI_TO_TIMER_UNITS( uiTimeLimit);
		uiStartTimeTU = FLM_GET_TIMER();
		uiRemainingTimeMilli = uiTimeLimit;
	}
	
	for (;;)
	{
		
		// If we are in the middle of populating the result set, we may need to wait
		// for entries to come into the result set.
		
		if (!m_bResultSetPopulated)
		{
			
			// If the entries are not in order, we must wait for the entire result
			// set to be populated.  NOTE: If we are not sorting, they are,
			// by definition, in order - we order them with a sequence number.
			
			if (!m_bEntriesAlreadyInOrder)
			{
				if (RC_BAD( rc = buildResultSet( ifpDb, uiRemainingTimeMilli,
												FLM_MAX_UINT)))
				{
					goto Exit;
				}
			}
			
			// If there are no entries in the result set yet, in order to get
			// the first entry we must at least wait for the first one to
			// appear.
			
			else if (m_pSortResultSet->getCount() < uiNumToWaitFor)
			{
				if (RC_BAD( rc = buildResultSet( ifpDb, uiRemainingTimeMilli,
												uiNumToWaitFor)))
				{
					goto Exit;
				}
			}
		}

		if (uiNumToWaitFor == 1)
		{
			
			// If the result set is not yet populated, we need to lock the mutex
			// before accessing it - because another thread is trying to populate it.
			
			if (RC_BAD( rc = m_pSortResultSet->getFirst( ucKey, sizeof( ucKey),
											&uiKeyLen, m_bResultSetPopulated ? FALSE : TRUE)))
			{
				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = m_pSortResultSet->getNext( ucKey, sizeof( ucKey),
											&uiKeyLen,
											m_bResultSetPopulated ? FALSE : TRUE)))
			{
				goto Exit;
			}
		}
	
		if (RC_BAD( rc = fqGetDocId( m_pSortIxd, ucKey, uiKeyLen, &ui64DocId)))
		{
			goto Exit;
		}
		if (RC_OK( rc = ifpDb->getNode( m_uiCollection, ui64DocId, ppNode)))
		{
			break;
		}
			
		if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			goto Exit;
		}
		rc = NE_XFLM_OK;
			
		// See how much time is left, if we are operating with a time limit.
		
		if (uiTimeLimit)
		{
			if (RC_BAD( rc = getRemainingTimeMilli( uiStartTimeTU, uiTimeLimitTU,
										&uiRemainingTimeMilli)))
			{
				goto Exit;
			}
		}
		uiNumToWaitFor++;
	}
	
Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get the last item in the result set.
***************************************************************************/
RCODE F_Query::getLastFromResultSet(
	IF_Db *			ifpDb,
	IF_DOMNode **	ppNode,
	FLMUINT			uiTimeLimit)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBYTE		ucKey [XFLM_MAX_KEY_SIZE];
	FLMUINT		uiKeyLen;
	FLMUINT64	ui64DocId;
	FLMUINT		uiStartTimeTU = 0;
	FLMUINT		uiRemainingTimeMilli = 0;
	FLMUINT		uiTimeLimitTU = 0;
	FLMBOOL		bGetLast = TRUE;
	
	// NOTE: uiTimeLimit will always be zero when building the result
	// set in another thread.
	
	if (uiTimeLimit)
	{
		uiTimeLimitTU = FLM_MILLI_TO_TIMER_UNITS( uiTimeLimit);
		uiStartTimeTU = FLM_GET_TIMER();
		uiRemainingTimeMilli = uiTimeLimit;
	}
	
	for (;;)
	{
	
		// If we are in the middle of populating the result set, we may need to wait
		// for entries to come into the result set.
		
		if (!m_bResultSetPopulated)
		{
			
			// When getting the last entry, we must always wait for the result set
			// to become fully populated.
			
			if (RC_BAD( rc = buildResultSet( ifpDb, uiRemainingTimeMilli,
										FLM_MAX_UINT)))
			{
				goto Exit;
			}
		}
		
		// If the result set is not yet populated, we need to lock the mutex
		// before accessing it - because another thread is trying to populate it.
		
		if (bGetLast)
		{
			if (RC_BAD( rc = m_pSortResultSet->getLast( ucKey, sizeof( ucKey),
											&uiKeyLen, m_bResultSetPopulated ? FALSE : TRUE)))
			{
				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = m_pSortResultSet->getPrev( ucKey, sizeof( ucKey),
											&uiKeyLen,
											m_bResultSetPopulated ? FALSE : TRUE)))
			{
				goto Exit;
			}
		}
		if (RC_BAD( rc = fqGetDocId( m_pSortIxd, ucKey, uiKeyLen, &ui64DocId)))
		{
			goto Exit;
		}
		if (RC_OK( rc = ifpDb->getNode( m_uiCollection, ui64DocId, ppNode)))
		{
			break;
		}
		if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			goto Exit;
		}
		rc = NE_XFLM_OK;
		
		// See how much time is left, if we are operating with a time limit.
		
		if (uiTimeLimit)
		{
			if (RC_BAD( rc = getRemainingTimeMilli( uiStartTimeTU, uiTimeLimitTU,
										&uiRemainingTimeMilli)))
			{
				goto Exit;
			}
		}
		bGetLast = FALSE;
	}
	
Exit:

	return( rc);
}
	
/***************************************************************************
Desc:	Get the next item from the result set.
***************************************************************************/
RCODE F_Query::getNextFromResultSet(
	IF_Db *			ifpDb,
	IF_DOMNode **	ppNode,
	FLMUINT			uiTimeLimit,
	FLMUINT			uiNumToSkip,
	FLMUINT *		puiNumSkipped)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBYTE		ucKey [XFLM_MAX_KEY_SIZE];
	FLMUINT		uiKeyLen;
	FLMUINT64	ui64DocId;
	FLMUINT		uiStartTimeTU = 0;
	FLMUINT		uiRemainingTimeMilli = 0;
	FLMUINT		uiTimeLimitTU = 0;
	FLMUINT		uiNumSkipped;
	
	// NOTE: uiTimeLimit will always be zero when building the result
	// set in another thread.
	
	if (uiTimeLimit)
	{
		uiTimeLimitTU = FLM_MILLI_TO_TIMER_UNITS( uiTimeLimit);
		uiStartTimeTU = FLM_GET_TIMER();
		uiRemainingTimeMilli = uiTimeLimit;
	}
	
	if (!puiNumSkipped)
	{

		// puiNumSkipped has to be non-NULL so it can be incremented only
		// if uiNumToSkip > 1

		if (uiNumToSkip > 1)
		{
			uiNumSkipped = 0;
			puiNumSkipped = &uiNumSkipped;
		}
	}
	else
	{
		*puiNumSkipped = 0;
	}
	for (;;)
	{
	
		// If we are in the middle of populating the result set, we may need to wait
		// for entries to come into the result set.
		
		if (!m_bResultSetPopulated)
		{
			
			// If the entries are not in order, we must wait for the entire result
			// set to be populated.  NOTE: If we are not sorting, they are,
			// by definition, in order - we order them with a sequence number.
			
			if (!m_bEntriesAlreadyInOrder)
			{
				if (RC_BAD( rc = buildResultSet( ifpDb, uiRemainingTimeMilli,
											FLM_MAX_UINT)))
				{
					goto Exit;
				}
			}
			else
			{
				FLMUINT	uiCurrPos = m_pSortResultSet->getCurrPos();
				
				// See if we have enough entries in the result set to position
				// to the next entry.
				
				// If we are not yet positioned (uiCurrPos == FLM_MAX_UINT),
				// we need to have at least one item in the result set.
				// Otherwise, we need to have one more beyond uiCurrPos - which
				// is (uiCurrPos + 1) + 1 --> uiCurrPos + 2.
				
				if (RC_BAD( rc = buildResultSet( ifpDb, uiRemainingTimeMilli,
										(uiCurrPos == FLM_MAX_UINT) ? 1 : uiCurrPos + 2)))
				{
					goto Exit;
				}
			}
		}
	
		// If the result set is not yet populated, we need to lock the mutex
		// before accessing it - because another thread is trying to populate it.
		
		if (RC_BAD( rc = m_pSortResultSet->getNext( ucKey, sizeof( ucKey),
										&uiKeyLen, m_bResultSetPopulated ? FALSE : TRUE)))
		{
			goto Exit;
		}
		if (puiNumSkipped)
		{
			(*puiNumSkipped)++;
		}
		if (uiNumToSkip > 1)
		{

			// puiNumSkipped will always be non-NULL in the case
			// where uiNumToSkip > 1

			flmAssert( puiNumSkipped);
			if (*puiNumSkipped < uiNumToSkip)
			{
				continue;
			}
		}
		if (RC_BAD( rc = fqGetDocId( m_pSortIxd, ucKey, uiKeyLen, &ui64DocId)))
		{
			goto Exit;
		}
		if (RC_OK( rc = ifpDb->getNode( m_uiCollection, ui64DocId, ppNode)))
		{
			break;
		}
		if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			goto Exit;
		}
		rc = NE_XFLM_OK;
		
		// See how much time is left, if we are operating with a time limit.
		
		if (uiTimeLimit)
		{
			if (RC_BAD( rc = getRemainingTimeMilli( uiStartTimeTU, uiTimeLimitTU,
										&uiRemainingTimeMilli)))
			{
				goto Exit;
			}
		}
	}
	
Exit:

	return( rc);
}
	
/***************************************************************************
Desc:	Get the previous item in the result set.
***************************************************************************/
RCODE F_Query::getPrevFromResultSet(
	IF_Db *			ifpDb,
	IF_DOMNode **	ppNode,
	FLMUINT			uiTimeLimit,
	FLMUINT			uiNumToSkip,
	FLMUINT *		puiNumSkipped)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBYTE		ucKey [XFLM_MAX_KEY_SIZE];
	FLMUINT		uiKeyLen;
	FLMUINT64	ui64DocId;
	FLMUINT		uiStartTimeTU = 0;
	FLMUINT		uiRemainingTimeMilli = 0;
	FLMUINT		uiTimeLimitTU = 0;
	FLMUINT		uiNumSkipped;
	
	if (!puiNumSkipped)
	{

		// puiNumSkipped has to be non-NULL so it can be incremented only
		// if uiNumToSkip > 1

		if (uiNumToSkip > 1)
		{
			uiNumSkipped = 0;
			puiNumSkipped = &uiNumSkipped;
		}
	}
	else
	{
		*puiNumSkipped = 0;
	}

	// NOTE: uiTimeLimit will always be zero when building the result
	// set in another thread.
	
	if (uiTimeLimit)
	{
		uiTimeLimitTU = FLM_MILLI_TO_TIMER_UNITS( uiTimeLimit);
		uiStartTimeTU = FLM_GET_TIMER();
		uiRemainingTimeMilli = uiTimeLimit;
	}
	
	// If we are in the middle of populating the result set, we may need to wait
	// for entries to come into the result set.
	
	if (!m_bResultSetPopulated)
	{
		
		// If the entries are not in order, we must wait for the entire result
		// set to be populated.  NOTE: If we are not sorting, they are,
		// by definition, in order - we order them with a sequence number.
		
		if (!m_bEntriesAlreadyInOrder)
		{
			if (RC_BAD( rc = buildResultSet( ifpDb, uiRemainingTimeMilli,
										FLM_MAX_UINT)))
			{
				goto Exit;
			}
		}
		
		// If there are no entries in the result set yet, in order to get
		// the first entry we must at least wait for the first one to
		// appear.
		
		else
		{
			FLMUINT	uiCurrPos = m_pSortResultSet->getCurrPos();
			
			// See if we have enough entries in the result set to position
			// to the previous entry.  We should never be positioned beyond
			// the number that are currently in there.  So, at most, we should
			// only have to wait for one to appear.
			
			if (uiCurrPos == FLM_MAX_UINT)
			{
				if (RC_BAD( rc = buildResultSet( ifpDb, uiRemainingTimeMilli, 1)))
				{
					goto Exit;
				}
			}
			else
			{
				
				// Better never be beyond the current count.
				
				flmAssert( uiCurrPos < m_pSortResultSet->getCount());
			}
		}
	}

	for (;;)
	{
		
		// If the result set is not yet populated, we need to lock the mutex
		// before accessing it - because another thread is trying to populate it.
		
		if (RC_BAD( rc = m_pSortResultSet->getPrev( ucKey, sizeof( ucKey),
										&uiKeyLen, m_bResultSetPopulated ? FALSE : TRUE)))
		{
			goto Exit;
		}
		if (puiNumSkipped)
		{
			(*puiNumSkipped)++;
		}
		if (uiNumToSkip > 1)
		{

			// puiNumSkipped will always be non-NULL in the case
			// where uiNumToSkip > 1

			flmAssert( puiNumSkipped);
			if (*puiNumSkipped < uiNumToSkip)
			{
				continue;
			}
		}
		if (RC_BAD( rc = fqGetDocId( m_pSortIxd, ucKey, uiKeyLen, &ui64DocId)))
		{
			goto Exit;
		}
		if (RC_OK( rc = ifpDb->getNode( m_uiCollection, ui64DocId, ppNode)))
		{
			break;
		}
		if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			goto Exit;
		}
		rc = NE_XFLM_OK;
		
		// See how much time is left, if we are operating with a time limit.
		
		if (uiTimeLimit)
		{
			if (RC_BAD( rc = getRemainingTimeMilli( uiStartTimeTU, uiTimeLimitTU,
										&uiRemainingTimeMilli)))
			{
				goto Exit;
			}
		}
	}
	
Exit:

	return( rc);
}
	
/***************************************************************************
Desc:	Get the current item in the result set.
***************************************************************************/
RCODE F_Query::getCurrentFromResultSet(
	IF_Db *				ifpDb,
	IF_DOMNode **		ppNode)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBYTE		ucKey [XFLM_MAX_KEY_SIZE];
	FLMUINT		uiKeyLen;
	FLMUINT64	ui64DocId;
	
	// If we are in the middle of populating the result set, we may need to wait
	// for entries to come into the result set.
	
	if (!m_bResultSetPopulated)
	{
		
		// If the entries are not in order, we must wait for the entire result
		// set to be populated.  But that means that we have never positioned
		// anywhere yet, so we should return an error.
		
		if (!m_bEntriesAlreadyInOrder)
		{
			rc = RC_SET( NE_XFLM_Q_NOT_POSITIONED);
			goto Exit;
		}
		
		// If there are no entries in the result set yet, in order to get
		// the first entry we must at least wait for the first one to
		// appear.
		
		else
		{
			FLMUINT	uiCurrPos = m_pSortResultSet->getCurrPos();
			
			// It is an error to call this if we have not yet positioned
			// anywhere in the result set.
			
			if (uiCurrPos == FLM_MAX_UINT)
			{
				rc = RC_SET( NE_XFLM_Q_NOT_POSITIONED);
				goto Exit;
			}
			else
			{
				
				// Better never be beyond the current count.
				
				flmAssert( uiCurrPos < m_pSortResultSet->getCount());
			}
		}
	}
	
	// If the result set is not yet populated, we need to lock the mutex
	// before accessing it - because another thread is trying to populate it.
	
	if (RC_BAD( rc = m_pSortResultSet->getCurrent( ucKey, sizeof( ucKey),
									&uiKeyLen, m_bResultSetPopulated ? FALSE : TRUE)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = fqGetDocId( m_pSortIxd, ucKey, uiKeyLen, &ui64DocId)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = ifpDb->getNode( m_uiCollection, ui64DocId, ppNode)))
	{
		if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			rc = RC_SET( NE_XFLM_Q_NOT_POSITIONED);
		}
		goto Exit;
	}
	
Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get previous node/document that passes query expression.
***************************************************************************/
RCODE XFLAPI F_Query::positionTo(
	IF_Db *			ifpDb,
	IF_DOMNode **	ppNode,
	FLMUINT			uiTimeLimit,
	FLMUINT			uiPosition
	)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBYTE		ucKey [XFLM_MAX_KEY_SIZE];
	FLMUINT		uiKeyLen;
	FLMUINT64	ui64DocId;
	FLMUINT		uiStartTimeTU = 0;
	FLMUINT		uiRemainingTimeMilli = 0;
	FLMUINT		uiTimeLimitTU = 0;
	
	// If we have a result set, or are in the middle of creating it,
	// we don't want to change the member variables because a background
	// thread may be using them.
	
	if (!m_pSortResultSet)
	{
	
		// NOTE: uiTimeLimit will always be zero when building the result
		// set in another thread.
		
		if (uiTimeLimit)
		{
			uiTimeLimitTU = FLM_MILLI_TO_TIMER_UNITS( uiTimeLimit);
			uiStartTimeTU = FLM_GET_TIMER();
			uiRemainingTimeMilli = uiTimeLimit;
		}
		
		m_pDb = (F_Db *)ifpDb;
		if (ppNode && *ppNode)
		{
			(*ppNode)->Release();
			*ppNode = NULL;
		}
	
		if (m_pDatabase && m_pDb->m_pDatabase != m_pDatabase)
		{
	
			// Make sure the passed in F_Db matches the one associated with
			// the query.
	
			rc = RC_SET( NE_XFLM_Q_MISMATCHED_DB);
			goto Exit;
		}
		
		// See if the database is being forced to close
	
		if (RC_BAD( rc = m_pDb->checkState( __FILE__, __LINE__)))
		{
			goto Exit;
		}
	
		// If we are not in a transaction, we cannot read.
	
		if (m_pDb->m_eTransType == XFLM_NO_TRANS)
		{
			rc = RC_SET( NE_XFLM_NO_TRANS_ACTIVE);
			goto Exit;
		}
	
		// See if we have a transaction going which should be aborted.
	
		if (RC_BAD( m_pDb->m_AbortRc))
		{
			rc = RC_SET( NE_XFLM_ABORT_TRANS);
			goto Exit;
		}
	
		if (!m_bOptimized)
		{
			if (RC_BAD( rc = optimize()))
			{
				goto Exit;
			}
		}
		
		// If we are not creating a result set, this is not a positionable
		// query.
		
		if (!m_pSortResultSet)
		{
			rc = RC_SET( NE_XFLM_Q_NON_POSITIONABLE_QUERY);
			goto Exit;
		}
	}
	
	// If we are in the middle of populating the result set, we may need to wait
	// for entries to come into the result set.
	
	if (!m_bResultSetPopulated)
	{
		
		// If the entries are not in order, we must wait for the entire result
		// set to be populated before we can determine what will be
		// the nth entry.  NOTE: If we are not sorting, they are,
		// by definition, in order - we order them with a sequence number.
		
		if (!m_bEntriesAlreadyInOrder)
		{
			if (RC_BAD( rc = buildResultSet( ifpDb, uiRemainingTimeMilli,
										FLM_MAX_UINT)))
			{
				goto Exit;
			}
		}
		else
		{
			FLMUINT	uiCurrPos = m_pSortResultSet->getCurrPos();
			
			// See if we have the nth entry yet.
			
			if (uiCurrPos == FLM_MAX_UINT ||
				 m_pSortResultSet->getCount() < uiPosition + 1)
			{
				// Must wait for uiPosition + 1, because uiPosition is zero based.
				
				if (RC_BAD( rc = buildResultSet( ifpDb, uiRemainingTimeMilli,
											uiPosition + 1)))
				{
					goto Exit;
				}
			}
		}
	}

	if (m_pSortIxd)
	{
		m_pSortResultSet->setIxInfo( m_pDb, m_pSortIxd);
	}

	// If the result set is not yet populated, we need to lock the mutex
	// before accessing it - because another thread is trying to populate it.
	
	if (RC_BAD( rc = m_pSortResultSet->positionToEntry( uiPosition,
									ucKey, sizeof( ucKey),
									&uiKeyLen, m_bResultSetPopulated ? FALSE : TRUE)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = fqGetDocId( m_pSortIxd, ucKey, uiKeyLen, &ui64DocId)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = ifpDb->getNode( m_uiCollection, ui64DocId, ppNode)))
	{
		if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			rc = RC_SET( NE_XFLM_Q_NOT_POSITIONED);
		}
		goto Exit;
	}
		
Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get previous node/document that passes query expression.
***************************************************************************/
RCODE XFLAPI F_Query::positionTo(
	IF_Db *				ifpDb,
	IF_DOMNode **		ppNode,
	FLMUINT				uiTimeLimit,
	IF_DataVector *	pSearchKey,
	FLMUINT				uiFlags
	)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBYTE		ucKey [XFLM_MAX_KEY_SIZE];
	FLMUINT		uiKeyLen;
	FLMUINT64	ui64DocId;
	FLMUINT		uiStartTimeTU = 0;
	FLMUINT		uiRemainingTimeMilli = 0;
	FLMUINT		uiTimeLimitTU = 0;
	
	// If we have a result set, or are in the middle of creating it,
	// we don't want to change the member variables because a background
	// thread may be using them.
	
	if (!m_pSortResultSet)
	{
	
		// NOTE: uiTimeLimit will always be zero when building the result
		// set in another thread.
		
		if (uiTimeLimit)
		{
			uiTimeLimitTU = FLM_MILLI_TO_TIMER_UNITS( uiTimeLimit);
			uiStartTimeTU = FLM_GET_TIMER();
			uiRemainingTimeMilli = uiTimeLimit;
		}
		
		m_pDb = (F_Db *)ifpDb;
		if (ppNode && *ppNode)
		{
			(*ppNode)->Release();
			*ppNode = NULL;
		}
	
		if (m_pDatabase && m_pDb->m_pDatabase != m_pDatabase)
		{
	
			// Make sure the passed in F_Db matches the one associated with
			// the query.
	
			rc = RC_SET( NE_XFLM_Q_MISMATCHED_DB);
			goto Exit;
		}
		
		// See if the database is being forced to close
	
		if (RC_BAD( rc = m_pDb->checkState( __FILE__, __LINE__)))
		{
			goto Exit;
		}
	
		// If we are not in a transaction, we cannot read.
	
		if (m_pDb->m_eTransType == XFLM_NO_TRANS)
		{
			rc = RC_SET( NE_XFLM_NO_TRANS_ACTIVE);
			goto Exit;
		}
	
		// See if we have a transaction going which should be aborted.
	
		if (RC_BAD( m_pDb->m_AbortRc))
		{
			rc = RC_SET( NE_XFLM_ABORT_TRANS);
			goto Exit;
		}
	
		if (!m_bOptimized)
		{
			if (RC_BAD( rc = optimize()))
			{
				goto Exit;
			}
		}
		
		// If no sort keys were specified, we cannot do this kind of
		// positioning.
		
		if (!m_pSortIxd)
		{
			rc = RC_SET( NE_XFLM_Q_NON_POSITIONABLE_QUERY);
			goto Exit;
		}
		
		if (!m_pSortResultSet)
		{
	//visit: Need to allow something here for indexes where we really don't
	//care about a result set.  Positioning to a particular key is still possible
	//in that case.
			rc = RC_SET( NE_XFLM_Q_NON_POSITIONABLE_QUERY);
			goto Exit;
		}
	}
	
	// If we are in the middle of populating the result set, and the entries
	// are not in order, we need to wait for all entries to
	// come into the result set.
	//VISIT: Should we just wait for the entire result set even if the entries
	//are in order?
	
	if (!m_bResultSetPopulated && !m_bEntriesAlreadyInOrder)
	{
		if (RC_BAD( rc = buildResultSet( ifpDb, uiRemainingTimeMilli,
									FLM_MAX_UINT)))
		{
			goto Exit;
		}
	}
	
	// Need to set the result set's index and db.
		
	m_pSortResultSet->setIxInfo( m_pDb, m_pSortIxd);

	// If the result set is not yet populated, we need to lock the mutex
	// before accessing it - because another thread is trying to populate it.
	
	if (RC_BAD( rc = m_pSortResultSet->positionToEntry( ucKey, sizeof( ucKey),
									&uiKeyLen, (F_DataVector *)pSearchKey, uiFlags,
									m_bResultSetPopulated ? FALSE : TRUE)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = fqGetDocId( m_pSortIxd, ucKey, uiKeyLen, &ui64DocId)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = ifpDb->getNode( m_uiCollection, ui64DocId, ppNode)))
	{
		if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			rc = RC_SET( NE_XFLM_Q_NOT_POSITIONED);
		}
		goto Exit;
	}
		
Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get current position.
***************************************************************************/
RCODE XFLAPI F_Query::getPosition(
	IF_Db *			ifpDb,
	FLMUINT *		puiPosition
	)
{
	RCODE	rc = NE_XFLM_OK;
	
	// If we do not have a result set, we will not be positioned anywhere.
	
	if (!m_pSortResultSet)
	{
		m_pDb = (F_Db *)ifpDb;
	
		if (m_pDatabase && m_pDb->m_pDatabase != m_pDatabase)
		{
	
			// Make sure the passed in F_Db matches the one associated with
			// the query.
	
			rc = RC_SET( NE_XFLM_Q_MISMATCHED_DB);
			goto Exit;
		}
		
		// See if the database is being forced to close
	
		if (RC_BAD( rc = m_pDb->checkState( __FILE__, __LINE__)))
		{
			goto Exit;
		}
	
		// If we are not in a transaction, we cannot read.
	
		if (m_pDb->m_eTransType == XFLM_NO_TRANS)
		{
			rc = RC_SET( NE_XFLM_NO_TRANS_ACTIVE);
			goto Exit;
		}
	
		// See if we have a transaction going which should be aborted.
	
		if (RC_BAD( m_pDb->m_AbortRc))
		{
			rc = RC_SET( NE_XFLM_ABORT_TRANS);
			goto Exit;
		}
	
		if (!m_bOptimized)
		{
			if (RC_BAD( rc = optimize()))
			{
				goto Exit;
			}
		}
		
		if (!m_pSortResultSet)
		{
			rc = RC_SET( NE_XFLM_Q_NON_POSITIONABLE_QUERY);
			goto Exit;
		}
		*puiPosition = 0;
		rc = RC_SET( NE_XFLM_BOF_HIT);
		goto Exit;
	}
	
	if ((*puiPosition = m_pSortResultSet->getCurrPos()) == FLM_MAX_UINT)
	{
		*puiPosition = 0;
		rc = RC_SET( NE_XFLM_BOF_HIT);
		goto Exit;
	}
			
Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get count.  Only works for queries that are building result sets.
***************************************************************************/
RCODE XFLAPI F_Query::getCounts(
	IF_Db *		ifpDb,
	FLMUINT		uiTimeLimit,
	FLMBOOL		bPartialCountOk,
	FLMUINT *	puiReadCount,
	FLMUINT *	puiPassedCount,
	FLMUINT *	puiPositionableToCount,
	FLMBOOL *	pbDoneBuildingResultSet
	)
{
	RCODE	rc = NE_XFLM_OK;
	
	// If we do not have a result set, we will not have a count.
	
	if (!m_pSortResultSet)
	{
		m_pDb = (F_Db *)ifpDb;
	
		if (m_pDatabase && m_pDb->m_pDatabase != m_pDatabase)
		{
	
			// Make sure the passed in F_Db matches the one associated with
			// the query.
	
			rc = RC_SET( NE_XFLM_Q_MISMATCHED_DB);
			goto Exit;
		}
		
		// See if the database is being forced to close
	
		if (RC_BAD( rc = m_pDb->checkState( __FILE__, __LINE__)))
		{
			goto Exit;
		}
	
		// If we are not in a transaction, we cannot read.
	
		if (m_pDb->m_eTransType == XFLM_NO_TRANS)
		{
			rc = RC_SET( NE_XFLM_NO_TRANS_ACTIVE);
			goto Exit;
		}
	
		// See if we have a transaction going which should be aborted.
	
		if (RC_BAD( m_pDb->m_AbortRc))
		{
			rc = RC_SET( NE_XFLM_ABORT_TRANS);
			goto Exit;
		}
	
		if (!m_bOptimized)
		{
			if (RC_BAD( rc = optimize()))
			{
				goto Exit;
			}
		}
		
		if (!m_pSortResultSet)
		{
			rc = RC_SET( NE_XFLM_Q_NON_POSITIONABLE_QUERY);
			goto Exit;
		}
	}
	
	// If the result set is not yet populated, we need to let it become
	// completely populated before we return a count.
	
	if (!m_bResultSetPopulated)
	{
		if (bPartialCountOk)
		{
			if (m_bEntriesAlreadyInOrder)
			{
				*puiPositionableToCount = *puiPassedCount = m_pSortResultSet->getCount();
			}
			else
			{
				*puiPassedCount = m_pSortResultSet->getCount();
				*puiPositionableToCount = 0;
			}
			if (pbDoneBuildingResultSet)
			{
				*pbDoneBuildingResultSet = FALSE;
			}
		}
		else
		{
			if (RC_BAD( rc = buildResultSet( ifpDb, uiTimeLimit, FLM_MAX_UINT)))
			{
				goto Exit;
			}
			*puiPositionableToCount = *puiPassedCount = m_pSortResultSet->getCount();
			if (pbDoneBuildingResultSet)
			{
				*pbDoneBuildingResultSet = TRUE;
			}
		}
	}
	else
	{
		*puiPositionableToCount = *puiPassedCount = m_pSortResultSet->getCount();
		if (pbDoneBuildingResultSet)
		{
			*pbDoneBuildingResultSet = TRUE;
		}
	}
	
	// Always set *puiReadCount last, in case it is being incremented on
	// another thread.  It will be ok for it to advance a little after having
	// set the other two counters.  It would look weird, however, if it was
	// set first, and the other two counters ended up being greater than
	// this one.
	
	*puiReadCount = (FLMUINT)m_ui64RSDocsRead;
			
Exit:

	return( rc);
}

