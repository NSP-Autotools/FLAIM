//------------------------------------------------------------------------------
// Desc:	Check logical integrity of indexes.
// Tabs:	3
//
// Copyright (c) 1992, 1994-2007 Novell, Inc. All Rights Reserved.
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

/********************************************************************
Desc:	This routine builds a key tree from a collated key
********************************************************************/
RCODE F_DbCheck::keyToVector(
	FLMBYTE *			pucKey,
	FLMUINT				uiKeyLen,
	IF_DataVector **	ppKeyRV
	)
{
	RCODE	rc = NE_XFLM_OK;

	// Generate the key tree

	if ((*ppKeyRV = f_new F_DataVector) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	(*ppKeyRV)->reset();

	rc = (*ppKeyRV)->inputKey( m_pDb, m_pIxd->uiIndexNum, pucKey, uiKeyLen);

Exit:

	return( rc);
}

/********************************************************************
Desc: Retrieves the next key from the sorted result set
*********************************************************************/
RCODE F_DbCheck::chkGetNextRSKey( void)
{
	RCODE			rc = NE_XFLM_OK;
	RS_IX_KEY *	pCurrRSKey;

	// Swap current and last key pointers - this allows us to always keep
	// the last key without having to memcpy the keys.

	pCurrRSKey = m_pCurrRSKey;
	m_pCurrRSKey = m_pPrevRSKey;
	m_pPrevRSKey = pCurrRSKey;
	pCurrRSKey = m_pCurrRSKey;

	if (pCurrRSKey == NULL)
	{
		rc = RC_SET( NE_XFLM_EOF_HIT);
		goto Exit;
	}

	// Get the next key - call getFirst because we are deleting the
	// entry after we process it.
	
	if (RC_BAD( rc = m_pIxRSet->getFirst( m_pDb, m_pIxd, NULL,
										pCurrRSKey->pucRSKeyBuf,
										XFLM_MAX_KEY_SIZE,
										&pCurrRSKey->uiRSKeyLen,
										pCurrRSKey->pucRSDataBuf,
										XFLM_MAX_KEY_SIZE,
										&pCurrRSKey->uiRSDataLen)))
	{
		goto Exit;
	}

	// Verify that the key meets the minimum length requirements

	flmAssert( pCurrRSKey->uiRSKeyLen);

Exit:

	return( rc);

}

/********************************************************************
Desc: Verifies the current index key against the result set.
*********************************************************************/
RCODE F_DbCheck::verifyIXRSet(
	STATE_INFO *	pStateInfo)
{
	RCODE			rc = NE_XFLM_OK;
	FLMINT		iCmpVal = 0;
	FLMUINT		uiIteration = 0;
	FLMBOOL		bRSetEmpty = FALSE;
	RS_IX_KEY *	pCurrRSKey;
	RS_IX_KEY *	pPrevRSKey;

	if (!m_pCurrRSKey)
	{
		m_pCurrRSKey = &m_IxKey1;
		m_pPrevRSKey = &m_IxKey2;
	}

	// Compare index and result set keys
	
	while (!bRSetEmpty)
	{
		if (m_bGetNextRSKey)
		{

			// Get the next key from the result set.  If the result set
			// is empty, then m_uiRSKeyLen will be set to
			// zero, forcing the problem to be resolved below.

			if (RC_BAD( rc = chkGetNextRSKey()))
			{
				if (rc == NE_XFLM_EOF_HIT || rc == NE_XFLM_NOT_FOUND)
				{

					// Set bRSetEmpty to TRUE so that the loop will exit after the
					// current key is resolved.  Otherwise, conflict resolution on
					// the current key will be repeated forever (infinite loop).

					bRSetEmpty = TRUE;
					rc = NE_XFLM_OK;
				}
				else
				{
					goto Exit;
				}
			}
			else
			{

				// Update statistics
				
				m_Progress.ui64NumKeysExamined++;
			}
		}
		pCurrRSKey = m_pCurrRSKey;
		pPrevRSKey = m_pPrevRSKey;

		if (pCurrRSKey->uiRSKeyLen == 0 || bRSetEmpty)
		{

			// We don't have a key because we got an EOF when
			// reading the result set.  Need to resolve the
			// fact that the result set does not have a key
			// that is found in the index.  Set iCmpVal to
			// 1 to force this resolution.

			iCmpVal = 1;
		}
		else
		{

			// Compare the index key and result set key.
			
			if (RC_BAD( rc = ixKeyCompare( m_pDb, m_pIxd, NULL, NULL, NULL,
										TRUE, TRUE,
										pCurrRSKey->pucRSKeyBuf,
										pCurrRSKey->uiRSKeyLen,
										pStateInfo->pucElmKey,
										pStateInfo->uiElmKeyLen, &iCmpVal)))
			{
				goto Exit;
			}
		}

		if (iCmpVal < 0)
		{

			// The result set key is less than the index key.  This could mean
			// that the result set key needs to be added to the index.

			if ( RC_BAD( rc = resolveIXMissingKey( pStateInfo)))
			{

				// If the key was added to the index (bReposition == TRUE)
				// or we got some other error, we don't want to get the next
				// result set key.

				m_bGetNextRSKey = (bRSetEmpty ? TRUE : FALSE);
				goto Exit;
			}
			else
			{

				// False alarm.  The index is missing the key because of
				// a concurrent update.  We want to get the next RS key.

				m_bGetNextRSKey = TRUE;
				
				// Delete the current key in the result set so we don't hit it again.
				
				if (RC_BAD( rc = m_pIxRSet->deleteEntry( m_pDb, m_pIxd,
										pCurrRSKey->pucRSKeyBuf, pCurrRSKey->uiRSKeyLen)))
				{
					goto Exit;
				}
			}
		}
		else if (iCmpVal > 0)
		{

			// The result set key is greater than the index key.  This could mean
			// that the index key needs to be deleted from the index.  Whether we
			// delete the index key or not, we don't need to get the next result
			// set key, but we do want to reposition and get the next index key.

			m_bGetNextRSKey = (bRSetEmpty ? TRUE : FALSE);
			if ( RC_BAD( rc = resolveRSetMissingKey( pStateInfo)))
			{
				goto Exit;
			}
			break;
		}
		else
		{

			// The index and result set keys are equal.  We want to get
			// the next result set key.

			m_bGetNextRSKey = TRUE;

			// Delete the current key in the result set so we don't hit it again.
			
			if (RC_BAD( rc = m_pIxRSet->deleteEntry( m_pDb, m_pIxd,
									pCurrRSKey->pucRSKeyBuf, pCurrRSKey->uiRSKeyLen)))
			{
				goto Exit;
			}

			break;
		}

		// Call the yield function periodically

		uiIteration++;
		if (!(uiIteration & 0x1F) )
		{
			f_yieldCPU();
		}
	}

Exit:

	return( rc);
}

/********************************************************************
Desc: Resolves the case of a key found in the result set but not in
		the current index.
*********************************************************************/
RCODE F_DbCheck::resolveIXMissingKey(
	STATE_INFO *	pStateInfo)
{
	FLMBOOL					bKeyInDoc;
	FLMBOOL					bKeyInIndex;
	RCODE						rc = NE_XFLM_OK;
	FLMBOOL					bFixCorruption = FALSE;
	RS_IX_KEY *				pCurrRSKey = m_pCurrRSKey;
	XFLM_INDEX_STATUS		ixStatus;

	// Determine if the record generates the key and if the
	// key is found in the index.

	if (RC_BAD( rc = getKeySource( pCurrRSKey->pucRSKeyBuf,
											 pCurrRSKey->uiRSKeyLen,
											 &bKeyInDoc, &bKeyInIndex)))
	{
		if (rc == NE_XFLM_INDEX_OFFLINE)
		{
			rc = NE_XFLM_OK;
		}
		goto Exit;
	}

	// If the record does not generate the key or the key+ref is in the index,
	// the index is not corrupt.

	if (!bKeyInDoc || bKeyInIndex)
	{
		m_Progress.ui64NumConflicts++;
		goto Exit;
	}

	// Otherwise, the index is corrupt.

	// Update statistics
	
	m_Progress.ui64NumDocKeysNotFound++;
	m_pDbInfo->m_uiLogicalCorruptions++;

	// Report the error
	
	if (RC_BAD( rc = reportIxError( pStateInfo,
											  FLM_KEY_NOT_IN_KEY_REFSET,
											  pCurrRSKey->pucRSKeyBuf,
											  pCurrRSKey->uiRSKeyLen,
											  &bFixCorruption)))
	{
		goto Exit;
	}

	// Exit if the application does not want to repair the corruption.

	if (!bFixCorruption)
	{

		// Set the logical corruption flag

		m_bIndexCorrupt = TRUE;
		goto Exit;
	}

	// Fix the corruption
	
	if (RC_BAD( rc = m_pDb->indexStatus( m_pIxd->uiIndexNum, &ixStatus)))
	{
		goto Exit;
	}
	
	if (ixStatus.ui64LastDocumentIndexed == (FLMUINT64)~0 &&
		 ixStatus.eState != XFLM_INDEX_SUSPENDED)
	{

		// Update statistics

		m_pDbInfo->m_uiLogicalRepairs++;
			
		// Add the key
		
		if (RC_OK( rc = addDelKeyRef( pCurrRSKey->pucRSKeyBuf,
												pCurrRSKey->uiRSKeyLen,
												FALSE)))
		{
			goto Exit;
		}
		else
		{

			// Set the logical corruption flag

			m_bIndexCorrupt = TRUE;
		}
	}
	else
	{

		// Set the logical corruption flag

		m_bIndexCorrupt = TRUE;
	}

Exit:

	return( rc);
}


/********************************************************************
Desc: Resolves the case of a key found in the current index but not
		in the result set.
*********************************************************************/
RCODE F_DbCheck::resolveRSetMissingKey(
	STATE_INFO *	pStateInfo)
{
	RCODE						rc = NE_XFLM_OK;
	FLMBOOL					bKeyInDoc;
	FLMBOOL					bKeyInIndex;
	FLMBOOL					bFixCorruption = FALSE;
	XFLM_INDEX_STATUS		ixStatus;

	// See if the key is found in the index and/or generated
	// by the document.
	
	if (RC_BAD( rc = getKeySource( pStateInfo->pucElmKey,
											 pStateInfo->uiElmKeyLen,
											 &bKeyInDoc, &bKeyInIndex)))
	{
		if (rc == NE_XFLM_INDEX_OFFLINE)
		{
			rc = NE_XFLM_OK;
		}
		goto Exit;
	}

	// If the key is generated by the record or the key is not found
	// in the index, the index is not corrupt.
	
	if (bKeyInDoc || !bKeyInIndex)
	{
		m_Progress.ui64NumConflicts++;
		goto Exit;
	}

	// Otherwise, the index is corrupt.

	// Update statistics
	
	m_Progress.ui64NumKeysNotFound++;
	m_pDbInfo->m_uiLogicalCorruptions++;

	// Report the error

	if (RC_BAD( rc = reportIxError( pStateInfo,
											  FLM_IX_KEY_NOT_FOUND_IN_REC, 
											  pStateInfo->pucElmKey,
											  pStateInfo->uiElmKeyLen,
											  &bFixCorruption)))
	{
		goto Exit;
	}

	// Exit if the application does not want to repair the corruption.

	if (!bFixCorruption)
	{

		// Set the logical corruption flag

		m_bIndexCorrupt = TRUE;
		goto Exit;
	}

	// Fix the corruption

	if (RC_BAD( rc = m_pDb->indexStatus( m_pIxd->uiIndexNum, &ixStatus)))
	{
		goto Exit;
	}
	
	if (ixStatus.ui64LastDocumentIndexed == (FLMUINT64)~0 &&
		 ixStatus.eState != XFLM_INDEX_SUSPENDED)
	{

		// Update statistics

		m_pDbInfo->m_uiLogicalRepairs++;
			
		// Delete the reference from the index
		
		if (RC_BAD( rc = addDelKeyRef( pStateInfo->pucElmKey,
												pStateInfo->uiElmKeyLen,
												TRUE)))
		{
			// Set the logical corruption flag

			m_bIndexCorrupt = TRUE;
		}
		goto Exit;
	}
	else
	{

		// Set the logical corruption flag

		m_bIndexCorrupt = TRUE;
	}

Exit:

	return( rc);
}

/********************************************************************
Desc: Verifies that a key component is actually in the document.
*********************************************************************/
RCODE F_DbCheck::verifyComponentInDoc(
	ICD *				pIcd,
	FLMUINT			uiComponent,
	F_DataVector *	pKey,
	FLMBOOL *		pbInDoc
	)
{
	RCODE				rc = NE_XFLM_OK;
	F_DOMNode *		pDOMNode = NULL;
	FLMUINT64		ui64NodeId;
	FLMUINT			uiNameId;
	
	// Get the nodeId.
	
	if ((ui64NodeId = pKey->getID( uiComponent)) != 0)
	{
		if (RC_BAD( rc = m_pDb->getNode( m_pIxd->uiCollectionNum, ui64NodeId,
												 &pDOMNode)))
		{
			if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				rc = NE_XFLM_OK;
			}
			*pbInDoc = FALSE;
			goto Exit;
		}
		
		// No need to verify the name ID if it is an element root tag.
		
		if( pIcd->uiFlags & ICD_IS_ATTRIBUTE)
		{
			if( RC_BAD( rc = pDOMNode->hasAttribute( m_pDb, pIcd->uiDictNum)))
			{
				if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					rc = NE_XFLM_OK;
				}

				*pbInDoc = FALSE;
				goto Exit;
			}
		}
		else if( pIcd->uiDictNum != ELM_ROOT_TAG)
		{
			if (RC_BAD( rc = pDOMNode->getNameId( m_pDb, &uiNameId)))
			{
				goto Exit;
			}
	
			if (uiNameId != pIcd->uiDictNum)
			{
				*pbInDoc = FALSE;
				goto Exit;
			}
		}

		// Make sure these are the same type.
		
		if ((pKey->isAttr( uiComponent) && !(pIcd->uiFlags & ICD_IS_ATTRIBUTE)) ||
			 (!pKey->isAttr( uiComponent) && (pIcd->uiFlags & ICD_IS_ATTRIBUTE)))
		{
			goto Exit;
		}

		// Verify that the node belongs to the document.
		
		if (pKey->getDocumentID() != pDOMNode->getDocumentId())
		{
			*pbInDoc = FALSE;
			goto Exit;
		}
	}
	
Exit:

	if (pDOMNode)
	{
		pDOMNode->Release();
	}

	return( rc);
}

/********************************************************************
Desc: Determines if a key is generated by the current document
		and/or if the key is found in the current index
*********************************************************************/
RCODE F_DbCheck::getKeySource(
	FLMBYTE *		pucKey,
	FLMUINT			uiKeyLen,
	FLMBOOL *		pbKeyInDoc,
	FLMBOOL *		pbKeyInIndex
	)
{
	RCODE				rc = NE_XFLM_OK;
	ICD *				pIcd;
	FLMUINT			uiComponent;
	F_DataVector	key;

	// Initialize return values.

	*pbKeyInDoc = FALSE;
	*pbKeyInIndex = FALSE;
	
	if (m_pIxd->uiFlags & IXD_OFFLINE)
	{
		rc = RC_SET( NE_XFLM_INDEX_OFFLINE);
		goto Exit;
	}
	
	// See if the key is in the index.
		
	if (RC_BAD( rc = chkVerifyKeyExists( pucKey, uiKeyLen, pbKeyInIndex)))
	{
		goto Exit;
	}

	// Put the key into a data vector structure.
	
	if (RC_BAD( rc = key.inputKey( m_pDb, m_pIxd->uiIndexNum, pucKey, uiKeyLen)))
	{
		goto Exit;
	}

	// See if all of the nodes referenced from the key actually are in the
	// document.  This includes data nodes and context nodes.

	*pbKeyInDoc = TRUE;
	uiComponent = 0;
	pIcd = m_pIxd->pFirstKey;
	while (pIcd)
	{
		if (RC_BAD( rc = verifyComponentInDoc( pIcd, uiComponent, &key, pbKeyInDoc)))
		{
			goto Exit;
		}
		if (!(*pbKeyInDoc))
		{
			goto Exit;
		}
		uiComponent++;
		pIcd = pIcd->pNextKeyComponent;
	}
	
	// Go through data components.
	
	pIcd = m_pIxd->pFirstData;
	while (pIcd)
	{
		if (RC_BAD( rc = verifyComponentInDoc( pIcd, uiComponent, &key, pbKeyInDoc)))
		{
			goto Exit;
		}
		if (!(*pbKeyInDoc))
		{
			goto Exit;
		}
		uiComponent++;
		pIcd = pIcd->pNextDataComponent;
	}
	
	// Go through context components

	pIcd = m_pIxd->pFirstContext;
	while (pIcd)
	{
		if (RC_BAD( rc = verifyComponentInDoc( pIcd, uiComponent, &key, pbKeyInDoc)))
		{
			goto Exit;
		}
		if (!(*pbKeyInDoc))
		{
			goto Exit;
		}
		uiComponent++;
		pIcd = pIcd->pNextKeyComponent;
	}
	
Exit:

	return( rc);
}

/********************************************************************
Desc: Verify that a key is (or is not) found in an index.
*********************************************************************/
RCODE F_DbCheck::chkVerifyKeyExists(
	FLMBYTE *		pucKey,
	FLMUINT			uiKeyLen,
	FLMBOOL *		pbFoundRV)
{
	RCODE				rc = NE_XFLM_OK;
	F_Btree *		pbtree = NULL;
	IXKeyCompare	compareObject;
	
	compareObject.setIxInfo( m_pDb, m_pIxd);
	compareObject.setCompareNodeIds( TRUE);
	compareObject.setCompareDocId( TRUE);

	*pbFoundRV = FALSE;

	// Get a btree
	
	if (RC_BAD( rc = gv_XFlmSysData.pBtPool->btpReserveBtree( &pbtree)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pbtree->btOpen( m_pDb, &m_pIxd->lfInfo, 
		(m_pIxd->uiFlags & IXD_ABS_POS) ? TRUE : FALSE, FALSE,
		&compareObject)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pbtree->btLocateEntry(
								pucKey, uiKeyLen, &uiKeyLen, XFLM_EXACT)))
	{
		if( rc == NE_XFLM_NOT_FOUND)
		{
			rc = NE_XFLM_OK;
		}
		goto Exit;
	}

	*pbFoundRV = TRUE;

Exit:

	if (pbtree)
	{
		gv_XFlmSysData.pBtPool->btpReturnBtree( &pbtree);
	}

	return( rc );
}

/***************************************************************************
Desc:	This routine adds or deletes an index key and/or reference.
*****************************************************************************/
RCODE F_DbCheck::addDelKeyRef(
	FLMBYTE *		pucKey,
	FLMUINT			uiKeyLen,
	FLMBOOL			bDelete)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBYTE			ucKeyBuf[ sizeof( KREF_ENTRY) + XFLM_MAX_KEY_SIZE];
	KREF_ENTRY *	pKrefEntry = (KREF_ENTRY *)(&ucKeyBuf[ 0]);
	FLMBOOL			bStartedUpdate = FALSE;
	FLMBOOL			bKeyInDoc;
	FLMBOOL			bKeyInIndex;

	// Start an update transaction, if necessary

	if (RC_BAD( rc = startUpdate()))
	{
		goto Exit;
	}
	bStartedUpdate = TRUE;
	
	// Verify that the state has not changed

	if (RC_BAD( rc = getKeySource( pucKey, uiKeyLen,
											 &bKeyInDoc, &bKeyInIndex)))
	{
		goto Exit;
	}

	if ((bKeyInIndex && bDelete) || (!bKeyInIndex && !bDelete))
	{
		// Setup the KrefEntry structure

		f_memcpy( &(ucKeyBuf[ sizeof( KREF_ENTRY)]), pucKey, uiKeyLen);
		pKrefEntry->ui16KeyLen = (FLMUINT16)uiKeyLen;
		pKrefEntry->uiDataLen = 0;
		pKrefEntry->ui16IxNum = (FLMUINT16)m_pIxd->uiIndexNum;
		pKrefEntry->uiSequence = 1;
		pKrefEntry->bDelete = bDelete;

		// Add or delete the key/reference.
		
		if (RC_BAD( rc = m_pDb->refUpdate( &m_pIxd->lfInfo, m_pIxd, pKrefEntry,
															FALSE)))
		{
			goto Exit;
		}

		// Update statistics

		m_Progress.ui32NumProblemsFixed++;
	}

Exit:

	// End the update

	if (bStartedUpdate)
	{
		RCODE	rc2;
		
		if (RC_BAD( rc2 = chkEndUpdate()))
		{
			if (RC_OK( rc))
			{
				rc = rc2;
			}
		}
	}

	return( rc);
}

/********************************************************************
Desc: Populates the XFLM_CORRUPT_INFO structure and calls the user's
		callback routine.
*********************************************************************/
RCODE F_DbCheck::reportIxError(
	STATE_INFO *	pStateInfo,
	FLMINT32			i32ErrCode,
	FLMBYTE *		pucErrKey,
	FLMUINT			uiErrKeyLen,
	FLMBOOL *		pbFixErrRV
	)
{
	RCODE						rc = NE_XFLM_OK;
	void *					pDbPoolMark = NULL;
	FLMBOOL					bResetKRef = FALSE;
	XFLM_CORRUPT_INFO		CorruptInfo;

	f_memset( &CorruptInfo, 0, sizeof( XFLM_CORRUPT_INFO));

	// Need to mark the DB's temporary pool.  The index code allocates
	// memory for new CDL entries from the DB pool.  If the pool is not
	// reset, it grows during the check and becomes VERY large.

	pDbPoolMark = m_pDb->m_tempPool.poolMark();

	// Set up the KRef so that flmGetRecKeys will work
	
	if (RC_BAD( rc = m_pDb->krefCntrlCheck()))
	{
		goto Exit;
	}
	bResetKRef = TRUE;

	// Fix corruptions by default unless the app says not to.

	CorruptInfo.ui32ErrLocale = XFLM_LOCALE_INDEX;
	CorruptInfo.i32ErrCode = i32ErrCode;
	CorruptInfo.ui32ErrLfNumber = (FLMUINT32)m_pIxd->uiIndexNum;
	CorruptInfo.ui32ErrElmOffset = (FLMUINT32)pStateInfo->uiElmOffset;

	// Generate the key tree using the key that caused the error

	if (RC_BAD( rc = keyToVector( pucErrKey, uiErrKeyLen, &CorruptInfo.ifpErrIxKey)))
	{
		goto Exit;
	}

	// Report the error

	*pbFixErrRV = FALSE;
	if (m_pDbCheckStatus && RC_OK( m_LastStatusRc))
	{
		m_LastStatusRc = m_pDbCheckStatus->reportCheckErr( &CorruptInfo, pbFixErrRV);
	}

Exit:

	if (CorruptInfo.ifpErrIxKey)
	{
		CorruptInfo.ifpErrIxKey->Release();
		CorruptInfo.ifpErrIxKey = NULL;
	}

	// Remove any keys added to the KRef
	
	if (bResetKRef)
	{
		m_pDb->krefCntrlFree(); // VISIT:  Is this correct?
	}

	// Reset the index check pool

	m_pDb->m_tempPool.poolReset(pDbPoolMark);

	return( rc);
}

/***************************************************************************
Desc:	Start an update transaction
*****************************************************************************/
RCODE F_DbCheck::startUpdate( void)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bAbortedReadTrans = FALSE;
	FLMUINT	uiSaveIndexNum = m_pIxd->uiIndexNum;

	// This routine should never be called unless
	// XFLM_ONLINE flag was passed in to the check.

	flmAssert( m_uiFlags & XFLM_ONLINE);
	
	if (m_pDb->getTransType() == XFLM_READ_TRANS)
	{

		// Free the KrefCntrl

		m_pDb->krefCntrlCheck();

		// Abort the read transaction

		m_pIxd = NULL;
		if (RC_BAD( rc = m_pDb->transAbort()))
		{
			goto Exit;
		}
		bAbortedReadTrans = TRUE;

		// Try to start an update transaction
	
		if (RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS, FLM_NO_TIMEOUT,
													 XFLM_DONT_POISON_CACHE)))
		{
			goto Exit;
		}
		m_bStartedUpdateTrans = TRUE;
		
		// Must reget the IXD.
		
		if (RC_BAD( rc = m_pDb->getDict()->getIndex( uiSaveIndexNum, &m_pLFile,
										&m_pIxd, TRUE)))
		{
			goto Exit;
		}
	}

	if (RC_BAD( m_LastStatusRc))
	{
		rc = m_LastStatusRc;
		goto Exit;
	}

Exit:

	// If something went wrong after the update transaction was started,
	// abort the transaction.

	if (RC_BAD( rc))
	{
		if (m_bStartedUpdateTrans)
		{
			m_pDb->transAbort();
			m_bStartedUpdateTrans = FALSE;
		}
	}

	// Re-start the read transaction.
	
	if (bAbortedReadTrans && !m_bStartedUpdateTrans)
	{
		RCODE	rc2;
		
		m_pIxd = NULL;
		if (RC_BAD( rc2 = m_pDb->transBegin( XFLM_READ_TRANS, FLM_NO_TIMEOUT,
									  XFLM_DONT_POISON_CACHE)))
		{
			if (RC_OK( rc))
			{
				rc = rc2;
			}
		}
		else
		{
			if (RC_BAD( rc2 = m_pDb->getDict()->getIndex( uiSaveIndexNum, &m_pLFile,
												&m_pIxd, TRUE)))
			{
				if (RC_OK( rc))
				{
					rc = rc2;
				}
			}
		}
	}

	return( rc);
}

/***************************************************************************
Desc:	End an update transaction.
*****************************************************************************/
RCODE F_DbCheck::chkEndUpdate( void)
{
	RCODE		rc = NE_XFLM_OK;
	FLMUINT	uiSaveIndexNum = m_pIxd->uiIndexNum;

	if (m_bStartedUpdateTrans)
	{

		// Commit the update transaction that was started.  If the transaction
		// started by the application, do not commit it.

		m_pIxd = NULL;		
		m_bStartedUpdateTrans = FALSE;
		if (RC_BAD( rc = m_pDb->transCommit()))
		{
			goto Exit;
		}
	}

Exit:
	
	// Re-start read transaction
	
	if (m_pDb->getTransType() == XFLM_NO_TRANS)
	{
		RCODE	rc2;
		
		if (RC_BAD( rc2 = m_pDb->transBegin( 
			XFLM_READ_TRANS, FLM_NO_TIMEOUT, XFLM_DONT_POISON_CACHE)))
		{
			if (RC_OK( rc))
			{
				rc = rc2;
			}
		}
		else
		{
			if (RC_BAD( rc2 = m_pDb->getDict()->getIndex( uiSaveIndexNum, &m_pLFile,
												&m_pIxd, TRUE)))
			{
				if (RC_OK( rc))
				{
					rc = rc2;
				}
			}
		}
	}

	return( rc);
}

