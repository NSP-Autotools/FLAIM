//-------------------------------------------------------------------------
// Desc:	Various cursor/query functions
// Tabs:	3
//
// Copyright (c) 1994-2007 Novell, Inc. All Rights Reserved.
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

#define FLM_WILD_MASK		0x03
#define FLM_CASE_MASK		0x0C
#define FLM_GRAN_MASK		0xF00

POOL_STATS	g_SQPoolStats = {0,0};
POOL_STATS	g_QueryPoolStats = {0,0};

// Local Function Prototypes

FSTATIC RCODE flmCurCopyQTInfo(
	QTINFO *		pSrc,
	QTINFO *		pDest,
	F_Pool *		pPool);

FSTATIC void flmCurClearSelect(
	CURSOR *		pCursor);

FSTATIC RCODE flmCurPosToEOF(
	CURSOR *		pCursor);

FSTATIC RCODE flmCurPosToBOF(
	CURSOR *		pCursor);

FSTATIC RCODE flmCurSetPos(
	CURSOR *		pDestCursor,
	CURSOR *		pSrcCursor	);

FSTATIC RCODE flmCurSetAbsolutePos(
	CURSOR *		pCursor,
	FLMUINT		uiPosition,
	FLMBOOL		bFallForward,
	FLMUINT *	puiPosition);

FSTATIC RCODE flmCurPositionable(
	CURSOR *		pCursor,
	FLMBOOL *	pbPositionable);

FSTATIC RCODE flmCurAbsPositionable(
	CURSOR *		pCursor,
	FLMBOOL *	pbAbsPositionable);

FSTATIC RCODE flmCurGetAbsolutePos(
	CURSOR *		pCursor,
	FLMUINT *	puiPosition);

FSTATIC RCODE flmCurGetAbsoluteCount(
	CURSOR *		pCursor,
	FLMUINT *	puiCount);

FSTATIC FLMBOOL flmCurMatchIndexPath( 
	IFD *			pIfd, 
	FLMUINT *	puiField, 
	FLMUINT		uiPos);

/****************************************************************************
Desc: Finishes a source's invisible transaction, if any.
****************************************************************************/
void flmCurFinishTrans(
	CURSOR *		pCursor)
{
	FLMBOOL		bIgnore;

	if (pCursor->bInvTrans && pCursor->pDb)
	{
		if (RC_OK( fdbInit( pCursor->pDb, FLM_NO_TRANS, 0, 0,
									&bIgnore)))
		{
			if (pCursor->pDb->uiTransType != FLM_NO_TRANS &&
				 pCursor->pDb->uiTransCount == pCursor->uiTransSeq)
			{

				// If the commit fails, then do the abort.

				if (RC_BAD( flmCommitDbTrans( pCursor->pDb, 0, FALSE)))
				{
					(void)flmAbortDbTrans( pCursor->pDb);
				}
			}
		}
		fdbExit( pCursor->pDb);
		pCursor->bInvTrans = FALSE;
	}
}

/****************************************************************************
Desc: Initializes an FDB for a source.
****************************************************************************/
RCODE flmCurDbInit(
	CURSOR *		pCursor)
{
	RCODE			rc;
	FLMBOOL		bStartedTrans;

	if (RC_OK( rc = fdbInit( pCursor->pDb, FLM_READ_TRANS,
								FDB_TRANS_GOING_OK | FDB_INVISIBLE_TRANS_OK, 0,
								&bStartedTrans)))
	{
		if (bStartedTrans)
		{
			pCursor->pDb->uiFlags |= FDB_INVISIBLE_TRANS;
			pCursor->uiTransSeq = pCursor->pDb->uiTransCount;
			pCursor->bInvTrans = TRUE;
		}
	}
	return( rc);
}

/****************************************************************************
Desc:	Initializes a cursor for subsequent definition and navigation of
		a record set.  A cursor must be initialized before it can
		be used.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmCursorInit(
	HFDB			hDb,
	FLMUINT		uiContainer,
	HFCURSOR *	phCursor)
{
	RCODE       rc = FERR_OK;
	CURSOR *    pCursor = NULL;
	FDB *			pDb = (FDB *)hDb;

	flmAssert( hDb != HFDB_NULL);
	flmAssert( uiContainer != 0);

	// See if the database is being forced to close

	if( RC_BAD( rc = flmCheckDatabaseState( pDb)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = f_calloc( (FLMUINT)sizeof( CURSOR), &pCursor)))
	{
		goto Exit;
	}
	pCursor->QTInfo.uiMaxPredicates = MAX_USER_PREDICATES;
	pCursor->QTInfo.ppPredicates = &pCursor->QTInfo.Predicates [0];

	// Initialize cursor members

	pCursor->QueryPool.smartPoolInit( &g_QueryPoolStats);
	pCursor->SQPool.smartPoolInit( &g_SQPoolStats);

	pCursor->pDb = pDb;
	pCursor->uiContainer = uiContainer;

	// Default is to have FLAIM select an index.

	pCursor->uiIndexNum = FLM_SELECT_INDEX;
	pCursor->pCSContext = pCursor->pDb->pCSContext;
	pCursor->uiCursorId = FCS_INVALID_ID;

	pCursor->QTInfo.uiExpecting = FLM_Q_OPERAND;
	pCursor->QTInfo.uiFlags = FLM_COMP_CASE_INSENSITIVE | FLM_COMP_WILD;

Exit:
	if (RC_BAD( rc))
	{
		if (pCursor)
		{
			pCursor->QueryPool.poolFree();
			pCursor->SQPool.poolFree();
			f_free( &pCursor);
		}
	}
	*phCursor = (HFCURSOR)pCursor;
	return( rc);
}

/****************************************************************************
Desc:	Copies a passed-in query tree into a new tree, using the passed-in
		memory pool.
****************************************************************************/
FSTATIC RCODE flmCurCopyQTInfo(
	QTINFO *		pSrc,
	QTINFO *		pDest,
	F_Pool *		pPool)
{
	RCODE			rc = FERR_OK;
	FQNODE *		pDestParentNode;
	FQNODE *		pSrcCurrNode;
	FQNODE *		pDestCurrNode;
	FLMBOOL		bGoingUp = FALSE;
	FLMBOOL		bTreeComplete;

	// If the source query has been optimized, the query is found
	// in pSrc->pSaveQuery.  Otherwise, it is found in
	// pSrc->pTopNode.

	if (pSrc->pSaveQuery)
	{
		pSrcCurrNode = pSrc->pSaveQuery;
		bTreeComplete = TRUE;
	}
	else
	{
		pSrcCurrNode = pSrc->pTopNode;
		bTreeComplete = FALSE;
		pDest->pCurOpNode = NULL;
		pDest->pCurAtomNode = NULL;
	}

	// If there is no query tree, don't need to copy.

	if (pSrcCurrNode)
	{

		// Must not do a recursive copy, because tree may not
		// have been flattened, and the recursion might go
		// too deep.

		if (RC_BAD( rc = flmCurCopyQNode( pSrcCurrNode, pDest,
										&pDest->pTopNode, pPool)))
		{
			goto Exit;
		}
		pDestParentNode = NULL;
		pDestCurrNode = pDest->pTopNode;
		if (!bTreeComplete)
		{
			if (pSrcCurrNode == pSrc->pCurOpNode)
			{
				pDest->pCurOpNode = pDestCurrNode;
			}
			else if (pSrcCurrNode == pSrc->pCurAtomNode)
			{
				pDest->pCurAtomNode = pDestCurrNode;
			}
		}
		for (;;)
		{
			if (bGoingUp)
			{
				if (pSrcCurrNode->pNextSib)
				{
					pSrcCurrNode = pSrcCurrNode->pNextSib;
					bGoingUp = FALSE;
				}
				else if ((pSrcCurrNode = pSrcCurrNode->pParent) == NULL)
				{
					break;
				}
				else
				{
					pDestCurrNode = pDestCurrNode->pParent;
					pDestParentNode = pDestCurrNode->pParent;
					continue;
				}
			}
			else
			{
				if (pSrcCurrNode->pChild)
				{
					pSrcCurrNode = pSrcCurrNode->pChild;
					pDestParentNode = pDestCurrNode;
				}
				else
				{
					bGoingUp = TRUE;
					continue;
				}
			}
			if (RC_BAD( rc = flmCurCopyQNode( pSrcCurrNode, pDest,
										&pDestCurrNode, pPool)))
			{
				goto Exit;
			}

			flmCurLinkLastChild( pDestParentNode, pDestCurrNode);
			if (!bTreeComplete)
			{
				if (pSrcCurrNode == pSrc->pCurOpNode)
				{
					pDest->pCurOpNode = pDestCurrNode;
				}
				else if( pSrcCurrNode == pSrc->pCurAtomNode)
				{
					pDest->pCurAtomNode = pDestCurrNode;
				}
			}
		}
	}

	if (bTreeComplete)
	{
		pDest->pCurOpNode = pDest->pTopNode;
		pDest->uiNestLvl = 0;
		pDest->pCurAtomNode = NULL;
		if (pDest->pTopNode)
		{
			pDest->uiExpecting = FLM_Q_OPERATOR;
		}
		else
		{
			pDest->uiExpecting = FLM_Q_OPERAND;
		}
	}
	else
	{
		pDest->uiNestLvl = pSrc->uiNestLvl;
		pDest->uiExpecting = pSrc->uiExpecting;

		// Defect #84610 -- Need to force pCurOpNode and pCurAtomNode into pDest
		// in cases where the query has no operator (i.e., existence query).

		if (!pDest->pTopNode)
		{
			if (pSrc->pCurOpNode)
			{
				if (RC_BAD( rc = flmCurCopyQNode( pSrc->pCurOpNode,
											pDest, &pDest->pCurOpNode, pPool)))
				{
					goto Exit;
				}
			}
			if (pSrc->pCurAtomNode)
			{
				if (RC_BAD( rc = flmCurCopyQNode( pSrc->pCurAtomNode,
					pDest, &pDest->pCurAtomNode, pPool)))
				{
					goto Exit;
				}
			}
		}
	}

	pDest->uiFlags = pSrc->uiFlags;

Exit:
	return( rc);
}

/****************************************************************************
Desc:	Initializes a new cursor and sets its selection criteria and record
		sources to be the same as those of the passed-in cursor.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmCursorClone(
	HFCURSOR      hSource,
	HFCURSOR  *   phCursor)
{
	RCODE				rc = FERR_OK;
	CURSOR *    	pSrcCursor;
	CURSOR *    	pDestCursor = NULL;

	if ((pSrcCursor = (CURSOR *)hSource) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if (pSrcCursor->pCSContext)
	{
		rc = RC_SET( FERR_NOT_IMPLEMENTED);
		goto Exit;
	}

	// See if the database is being forced to close

	if( RC_BAD( rc = flmCheckDatabaseState( pSrcCursor->pDb)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = f_calloc( (FLMUINT)sizeof( CURSOR), &pDestCursor)))
	{
		goto Exit;
	}
	pDestCursor->QTInfo.uiMaxPredicates = MAX_USER_PREDICATES;
	pDestCursor->QTInfo.ppPredicates = &pDestCursor->QTInfo.Predicates [0];

	// Initialize cursor members

	pDestCursor->QueryPool.smartPoolInit( &g_QueryPoolStats);
	pDestCursor->SQPool.smartPoolInit( &g_SQPoolStats);

	// Set up a tree info structure for query declaration.

	if (RC_BAD( rc = flmCurCopyQTInfo( &pSrcCursor->QTInfo,
									 &pDestCursor->QTInfo,
									 &pDestCursor->QueryPool)))
	{
		goto Exit;
	}

	// Copy source information.

	pDestCursor->pDb = pSrcCursor->pDb;
	pDestCursor->uiContainer = pSrcCursor->uiContainer;
	pDestCursor->pCSContext = pSrcCursor->pDb->pCSContext;
	pDestCursor->uiCursorId = FCS_INVALID_ID;

	// Copy index information.

	pDestCursor->uiIndexNum = pSrcCursor->uiIndexNum;
	pDestCursor->uiRecType = pSrcCursor->uiRecType;

	// Initialize various structure elements

	pDestCursor->bOkToReturnKeys = pSrcCursor->bOkToReturnKeys;
	pDestCursor->bOptimized = FALSE;

Exit:
	if (RC_BAD( rc))
	{
		if (pDestCursor)
		{
			(void)flmCurFree( pDestCursor, TRUE);
			pDestCursor = NULL;
		}
	}
	*phCursor = (HFCURSOR)pDestCursor;
	return( rc);
}

/****************************************************************************
Desc: Frees up memory associated with a subquery structure.
****************************************************************************/
void flmSQFree(
	SUBQUERY *     pSubQuery,
	FLMBOOL			bFreeEverything)
{
	if (!bFreeEverything)
	{
		if (pSubQuery->pFSIndexCursor)
		{
			pSubQuery->pFSIndexCursor->releaseBlocks();
		}
		if (pSubQuery->pFSDataCursor)
		{
			pSubQuery->pFSDataCursor->releaseBlocks();
		}
	}
	else
	{
		FQNODE *	pCurrNode = pSubQuery->pTree;
		QTYPES	eType;

		// Free the memory associated with callbacks in the query tree.

		while (pCurrNode)
		{
			eType = GET_QNODE_TYPE( pCurrNode);
			if (IS_FLD_CB( eType, pCurrNode))
			{
				(void)pCurrNode->pQAtom->val.QueryFld.fnGetField(
					pCurrNode->pQAtom->val.QueryFld.pvUserData, NULL, HFDB_NULL,
					pCurrNode->pQAtom->val.QueryFld.puiFldPath, FLM_FLD_CLEANUP,
					NULL, NULL, NULL);
			}

			// Find the next node to process

			if (pCurrNode->pChild)
			{
				pCurrNode = pCurrNode->pChild;
			}
			else
			{

				// Travel back up the tree until we find a node
				// that has a sibling.

				for (;;)
				{
					if (pCurrNode->pNextSib)
					{
						pCurrNode = pCurrNode->pNextSib;
						break;
					}
					if ((pCurrNode = pCurrNode->pParent) == NULL)
						break;
				}
			}
		}

		pSubQuery->OptPool.poolFree();

		// Free up the file system cursors, if any.

		if (pSubQuery->pFSIndexCursor)
		{
			pSubQuery->pFSIndexCursor->Release();
			pSubQuery->pFSIndexCursor = NULL;
		}
		if (pSubQuery->pFSDataCursor)
		{
			pSubQuery->pFSDataCursor->Release();
			pSubQuery->pFSDataCursor = NULL;
		}
	}
	if (pSubQuery->pRec)
	{
		pSubQuery->pRec->Release();
		pSubQuery->pRec = NULL;
	}
}

/****************************************************************************
Desc: Frees up memory associated with a cursor.
****************************************************************************/
void flmCurFree(
	CURSOR *		pCursor,
	FLMBOOL		bFinishTrans)
{
	FLMUINT			uiCnt;
	CS_CONTEXT *	pCSContext;

	if (bFinishTrans)
	{
		flmCurFinishTransactions( pCursor, TRUE);
	}

	// Free the memory associated with positioning keys.

	flmCurFreePosKeys( pCursor);

	// Free the memory associated with any subqueries.

	flmCurFreeSQList( pCursor, TRUE);
	pCursor->SQPool.poolFree();

	// Free the memory associated with the pool structures

	pCursor->QueryPool.poolFree();
	if (pCursor->pDRNSet)
	{
		pCursor->pDRNSet->Release();
		pCursor->pDRNSet = NULL;
	}

	for (uiCnt = 0; uiCnt < pCursor->QTInfo.uiNumPredicates; uiCnt++)
	{
		pCursor->QTInfo.ppPredicates [uiCnt]->Release();
		pCursor->QTInfo.ppPredicates [uiCnt] = NULL;
	}
	if (pCursor->QTInfo.uiMaxPredicates > MAX_USER_PREDICATES)
	{
		f_free( &pCursor->QTInfo.ppPredicates);
	}
	f_memset( &pCursor->QTInfo, 0, sizeof( QTINFO));
	pCursor->QTInfo.uiMaxPredicates = MAX_USER_PREDICATES;
	pCursor->QTInfo.ppPredicates = &pCursor->QTInfo.Predicates [0];

	if ((pCSContext = pCursor->pCSContext) != NULL)
	{
		// Send message to free the cursor - if one was allocated.

		if ((pCursor->uiCursorId != FCS_INVALID_ID) &&
			 (pCSContext->bConnectionGood))
		{
			FCL_WIRE		Wire( pCSContext);

			// Send a request to free the cursor.

			if (RC_BAD( Wire.sendOp(
				FCS_OPCLASS_ITERATOR, FCS_OP_ITERATOR_FREE)))
			{
				goto CS_Exit;
			}

			if (RC_BAD( Wire.sendNumber(
				WIRE_VALUE_ITERATOR_ID, pCursor->uiCursorId)))
			{
				pCSContext->bConnectionGood = FALSE;
				goto CS_Exit;
			}

			if (RC_BAD( Wire.sendTerminate()))
			{
				pCSContext->bConnectionGood = FALSE;
				goto CS_Exit;
			}

			// Read the response - just discard it.

			if (RC_BAD( Wire.read()))
			{
				pCSContext->bConnectionGood = FALSE;
				goto CS_Exit;
			}
		}
		pCursor->pCSContext = NULL;
	}

CS_Exit:
	f_free( &pCursor);
	return;
}

/****************************************************************************
Desc:	Frees resources of the cursor without actually freeing the cursor.
		Keeps around the stuff that is needed to display the cursor
		information for debugging purposes after the fact.  At this point
		the cursor is no longer usable.
****************************************************************************/
FLMEXP void FLMAPI FlmCursorReleaseResources(
	HFCURSOR			hCursor)
{
	FLMUINT			uiCnt;
	CURSOR *			pCursor = (CURSOR *)hCursor;

	flmCurFinishTransactions( pCursor, TRUE);
	flmCurFreeSQList( pCursor, FALSE);
	
	for (uiCnt = 0; uiCnt < pCursor->QTInfo.uiNumPredicates; uiCnt++)
	{
		pCursor->QTInfo.ppPredicates [uiCnt]->releaseResources();
	}
}

/****************************************************************************
Desc:	Frees memory allocated to an initialized cursor.  The cursor handle
		cannot be used for additional cursor operations unless it is
		re-initialized.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmCursorFree(
	HFCURSOR  *   	phCursor)
{
	CURSOR *						pCursor = (CURSOR *)*phCursor;
	IF_LogMessageClient *	pLogMsg = NULL;

	flmAssert( pCursor);
	
	if( gv_FlmSysData.pLogger)
	{
		if ((pLogMsg = gv_FlmSysData.pLogger->beginMessage( FLM_QUERY_MESSAGE,
								F_DEBUG_MESSAGE)) != NULL)
		{
			flmLogQuery( pLogMsg, 0, pCursor);
			pLogMsg->endMessage();
			pLogMsg->Release();
		}
	}

	if (!pCursor->pCSContext && gv_FlmSysData.uiMaxQueries)
	{
		FlmCursorReleaseResources( (HFCURSOR)pCursor);
		flmSaveQuery( *phCursor);
		*phCursor = HFCURSOR_NULL;
	}
	else
	{
		flmCurFree( pCursor, TRUE);
		*phCursor = HFCURSOR_NULL;
	}

	return( FERR_OK);
}

/****************************************************************************
Desc:	Sets flags in the query that determine text comparision modes
		and the granularity for QuickFinder indexes.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmCursorSetMode(
	HFCURSOR    hCursor,
	FLMUINT		uiFlags)
{
	CURSOR *    pCursor = (CURSOR *)hCursor;

	flmAssert( pCursor != NULL);

	if ((pCursor->pCSContext) && (pCursor->uiCursorId != FCS_INVALID_ID))
	{
		return( RC_SET( FERR_NOT_IMPLEMENTED));
	}

	pCursor->QTInfo.uiFlags = uiFlags;
	return( FERR_OK);
}

/****************************************************************************
Desc: Clears the selection criteria in a query.
****************************************************************************/
FSTATIC void flmCurClearSelect(
	CURSOR *     pCursor)
{
	flmCurFreeSQList( pCursor, TRUE);
	pCursor->pTree = NULL;

	pCursor->bOptimized = FALSE;

	flmCurFinishTransactions( pCursor, FALSE);

	pCursor->QTInfo.pTopNode = NULL;
	pCursor->QTInfo.pCurOpNode = NULL;
	pCursor->QTInfo.pCurAtomNode = NULL;
	pCursor->QTInfo.uiNestLvl = 0;
	pCursor->QTInfo.uiExpecting = FLM_Q_OPERAND;
	pCursor->QTInfo.uiFlags = FLM_COMP_CASE_INSENSITIVE | FLM_COMP_WILD;

	pCursor->QueryPool.poolReset();

	pCursor->uiLastRecID = 0;
	pCursor->ReadRc = FERR_OK;
	pCursor->rc = FERR_OK;
}

/****************************************************************************
Desc: Positions a cursor to EOF.
****************************************************************************/
FSTATIC RCODE flmCurPosToEOF(
	CURSOR *		pCursor)
{
	RCODE			rc = FERR_OK;
	FlmRecord *	pRecord = NULL;

	if (RC_BAD( rc = flmCurPerformRead( FLM_CURSOR_LAST, 
						(HFCURSOR)pCursor, FALSE, TRUE, 0, &pRecord, NULL)))
	{
		if (rc == FERR_BOF_HIT)
		{

			// BOF HIT means that all entries were rejected. Change ReadRc
			// of cursor to be EOF hit.

			pCursor->ReadRc = RC_SET( FERR_EOF_HIT);
			rc = FERR_OK;
		}
	}
	else
	{
		rc = flmCurPerformRead( FLM_CURSOR_NEXT, (HFCURSOR)pCursor,
						TRUE, FALSE, 0, &pRecord, NULL);
		if (rc == FERR_EOF_HIT)
		{
			rc = FERR_OK;
		}
	}
	if (pRecord)
	{
		pRecord->Release();
	}
	return( rc);
}

/****************************************************************************
Desc: Positions a cursor to BOF.
****************************************************************************/
FSTATIC RCODE flmCurPosToBOF(
	CURSOR *	pCursor
	)
{
	RCODE			rc = FERR_OK;
	FlmRecord *	pRecord = NULL;

	if (RC_BAD( rc = flmCurPerformRead( FLM_CURSOR_FIRST, 
				(HFCURSOR)pCursor, TRUE, TRUE, 0, &pRecord, NULL)))
	{
		if (rc == FERR_EOF_HIT)
		{

			// EOF HIT means that all entries were rejected. Change ReadRc
			// of destination to be BOF hit.

			pCursor->ReadRc = RC_SET( FERR_BOF_HIT);
			rc = FERR_OK;
		}
	}
	else
	{
		rc = flmCurPerformRead( FLM_CURSOR_PREV, (HFCURSOR)pCursor,
						FALSE, FALSE, 0, &pRecord, NULL);
		if (rc == FERR_BOF_HIT)
		{
			rc = FERR_OK;
		}
	}
	if (pRecord)
	{
		pRecord->Release();
	}
	return( rc);
}

/****************************************************************************
Desc: Sets the positioning information in a query to be the same as that of
		another query.
****************************************************************************/
FSTATIC RCODE flmCurSetPos(
	CURSOR *		pDestCursor,
	CURSOR *		pSrcCursor)
{
	RCODE			rc = FERR_OK;
	FDB *			pDb = NULL;
	FlmRecord *	pRecord = NULL;
	FLMUINT		uiRecordDrn;
	FLMUINT		uiContainerNum;
	SUBQUERY *	pSrcSubQuery;
	SUBQUERY *	pDestSubQuery;
	FLMUINT		uiRecMatch;
	FLMBYTE *	pucKeyBuffer = NULL;
	FLMUINT		uiKeyLen;
	IXD *			pIxd;
	void *		pvMark = NULL;
	FLMBOOL		bSavedInvisTrans;

	// Must be at most one index used for optimizing - which means that
	// there will only be one sub-query.

	if ((pSrcSubQuery = pSrcCursor->pSubQueryList) != NULL)
	{
		// If the source query has been optimized, but there is either
		// no index or multiple sub-queries, we can't use it for
		// positioning.

		if (pSrcSubQuery->pNext)
		{
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
		}
	}
	
	if ((!pDestCursor->bOptimized))
	{
		if( RC_BAD( rc = flmCurPrep( pDestCursor)))
		{
			goto Exit;
		}
	}

	// Destination query has same restrictions as source query.
	// Must be at most one index used for optimizing.  This means
	// there must be at most one sub-query.

	if ((pDestSubQuery = pDestCursor->pSubQueryList) != NULL)
	{
		if (pDestSubQuery->pNext)
		{
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
		}
	}

	// If there is no current record in the source cursor, it is for one of
	// three reasons: the source cursor has not yet been positioned, it is at
	// EOF, or it is at BOF. Position the destination cursor to reflect these
	// cases.

	if (!pSrcCursor->uiLastRecID || !pSrcCursor->bOptimized)
	{
		if (pSrcCursor->ReadRc == FERR_EOF_HIT && pSrcCursor->bOptimized)
		{
			rc = flmCurPosToEOF( pDestCursor);
		}
		else
		{

			// Anything else we will just position it to FIRST - don't
			// really know what else to do.

			rc = flmCurPosToBOF( pDestCursor);
		}
		goto Exit;
	}

	// Initialize an FDB structure for various and sundry operations.

	pDb = pDestCursor->pDb;
	pvMark = pDb->TempPool.poolMark();
	
	if( RC_BAD( rc = flmCurDbInit( pDestCursor)))
	{
		goto Exit;
	}

	// At this point, we know we have an optimized source cursor and
	// an optimized destination cursor and both of them have exactly
	// one sub-query.  We also know that we are not positioned on
	// EOF or BOF.

	// Get the current record from the source sub-query.

	switch (pSrcSubQuery->OptInfo.eOptType)
	{
		case QOPT_USING_INDEX:

			// Get the current key and DRN.

			if (RC_BAD( rc = pSrcSubQuery->pFSIndexCursor->currentKeyBuf( pDb,
										&pDb->TempPool, &pucKeyBuffer, &uiKeyLen,
										&uiRecordDrn, &uiContainerNum)))
			{

				// Although currentKeyBuf is coded to allow for returning
				// EOF or BOF, by this point we should have already taken
				// care of that case.

				flmAssert( rc != FERR_BOF_HIT && rc != FERR_EOF_HIT);
				goto Exit;
			}

			// Retrieve the full record - must have for evaluation.

			flmAssert( uiContainerNum == pSrcCursor->uiContainer);
			if (RC_BAD( rc = flmRcaRetrieveRec( pDb, NULL,
										uiContainerNum, uiRecordDrn,
										TRUE, NULL, NULL, &pRecord)))
			{
				goto Exit;
			}
			break;
		case QOPT_USING_PREDICATE:
			if (pDestSubQuery->OptInfo.eOptType != QOPT_USING_PREDICATE)
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
				goto Exit;
			}
			CB_ENTER( pDb, &bSavedInvisTrans);
			rc = pDestSubQuery->pPredicate->positionTo( (HFDB)pDb,
											pSrcSubQuery->pPredicate);
			CB_EXIT( pDb, bSavedInvisTrans);
			goto Exit;
		case QOPT_SINGLE_RECORD_READ:

			// Retrieve the record.

			if (!pSrcSubQuery->pRec)
			{
				if (RC_BAD( rc = flmRcaRetrieveRec( pDb, NULL,
											pSrcCursor->uiContainer,
											pSrcSubQuery->OptInfo.uiDrn,
											TRUE, NULL, NULL, &pSrcSubQuery->pRec)))
				{
					goto Exit;
				}
				pSrcSubQuery->uiDrn = pSrcSubQuery->OptInfo.uiDrn;
				pSrcSubQuery->bRecIsAKey = FALSE;
			}
			pRecord = pSrcSubQuery->pRec;
			pRecord->AddRef();
			uiRecordDrn = pSrcSubQuery->uiDrn;
			break;
		case QOPT_PARTIAL_CONTAINER_SCAN:
		case QOPT_FULL_CONTAINER_SCAN:

			// Get the current record and DRN

			if (!pSrcSubQuery->pRec)
			{
				if (RC_BAD( rc = pSrcSubQuery->pFSDataCursor->currentRec( pDb,
										&pSrcSubQuery->pRec,
										&pSrcSubQuery->uiDrn)))
				{
					goto Exit;
				}
			}
			pRecord = pSrcSubQuery->pRec;
			pRecord->AddRef();
			uiRecordDrn = pSrcSubQuery->uiDrn;
			pSrcSubQuery->bRecIsAKey = FALSE;
			break;
		default:
			break;
	}

	// If the destination sub-query is optimized with a user predicate,
	// cannot position source into it at this point.  If the source
	// was a user predicate, the positioning will already have been
	// taken care of up above.
	// Also, source and destination cursors must be on the same
	// container.

	if ((pDestSubQuery->OptInfo.eOptType == QOPT_USING_PREDICATE) ||
		 (pDestCursor->uiContainer != pSrcCursor->uiContainer))
	{
		rc = RC_SET( FERR_NOT_IMPLEMENTED);
		goto Exit;
	}

	// Verify that the record passes the destination criteria.

	if (RC_BAD( rc = flmCurEvalCriteria( pDestCursor, pDestSubQuery,
								pRecord, FALSE, &uiRecMatch)))
	{
		goto Exit;
	}
	if (uiRecMatch != FLM_TRUE)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	// Record passes destination criteria - do the positioning.

	switch (pDestSubQuery->OptInfo.eOptType)
	{
		case QOPT_USING_INDEX:

			// VISIT: Should we insist that if the source is
			// also using an index that it be using the same
			// index as the destination?

			if (RC_BAD( rc = fdictGetIndex(
						pDb->pDict,
						pDb->pFile->bInLimitedMode,
						pDestSubQuery->OptInfo.uiIxNum,
						NULL, &pIxd)))
			{
				goto Exit;
			}

			// If the source is not using an index, we need to generate
			// our key buffer from pRecord.

			if (pSrcSubQuery->OptInfo.eOptType != QOPT_USING_INDEX)
			{
				if (RC_BAD( rc = flmCurMakeKeyFromRec( pDb, pIxd, &pDb->TempPool,
											pRecord, &pucKeyBuffer, &uiKeyLen)))
				{

					// FERR_ILLEGAL_OP means the record had multiple keys.  We
					// don't know where to position to in that case.

					if (rc == FERR_ILLEGAL_OP)
					{
						rc = RC_SET( FERR_NOT_IMPLEMENTED);
					}
					goto Exit;
				}
			}
			if (RC_BAD( rc = pDestSubQuery->pFSIndexCursor->positionTo( pDb,
										pucKeyBuffer, uiKeyLen, uiRecordDrn)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = pDestSubQuery->pFSIndexCursor->currentKey( pDb,
											&pDestSubQuery->pRec,
											&pDestSubQuery->uiDrn)))
			{
				goto Exit;
			}

			// We already know this key matches the criteria because we
			// tested it above.

			pDestSubQuery->bFirstReference = FALSE;

			// These should have been set by the call to currentKey.

			flmAssert( pDestSubQuery->pRec->getContainerID() ==
							pDestCursor->uiContainer);
			flmAssert( pDestSubQuery->pRec->getID() ==
							pDestSubQuery->uiDrn);

			pDestSubQuery->bRecIsAKey = TRUE;
			pDestSubQuery->uiCurrKeyMatch = FLM_UNK;
			break;
		case QOPT_SINGLE_RECORD_READ:
			if (uiRecordDrn != pDestSubQuery->OptInfo.uiDrn)
			{
				rc = RC_SET( FERR_NOT_FOUND);
				goto Exit;
			}
			break;
		case QOPT_PARTIAL_CONTAINER_SCAN:
		case QOPT_FULL_CONTAINER_SCAN:
			if (RC_BAD( rc = pDestSubQuery->pFSDataCursor->positionTo( pDb,
										uiRecordDrn)))
			{
				goto Exit;
			}
			break;
		default:
			flmAssert( 0);
			break;
	}
	pDestCursor->pCurrSubQuery = pDestSubQuery;

Exit:

	if (pRecord)
	{
		pRecord->Release();
	}
	
	if (pDb)
	{
		pDb->TempPool.poolReset( pvMark);
		fdbExit( pDb);
	}
	
	return( rc);
}

/****************************************************************************
Desc:	Saves the current cursor position.
****************************************************************************/
RCODE flmCurSavePosition(
	CURSOR *	pCursor
	)
{
	RCODE			rc = FERR_OK;
	SUBQUERY *	pSaveSubQuery;

	pCursor->pSaveSubQuery = pSaveSubQuery = pCursor->pCurrSubQuery;
	if (pSaveSubQuery)
	{
		switch (pSaveSubQuery->OptInfo.eOptType)
		{
			case QOPT_USING_INDEX:
				if (RC_BAD( rc = pSaveSubQuery->pFSIndexCursor->savePosition()))
				{
					goto Exit;
				}
				break;
			case QOPT_USING_PREDICATE:
				if (RC_BAD( rc = pSaveSubQuery->pPredicate->savePosition()))
				{
					goto Exit;
				}
				break;
			case QOPT_SINGLE_RECORD_READ:
				pSaveSubQuery->bSaveRecReturned = pSaveSubQuery->bRecReturned;
				break;
			case QOPT_PARTIAL_CONTAINER_SCAN:
			case QOPT_FULL_CONTAINER_SCAN:
				if (RC_BAD( rc = pSaveSubQuery->pFSDataCursor->savePosition()))
				{
					goto Exit;
				}
				break;
			default:
				break;
		}
	}
Exit:
	return( rc);
}

/****************************************************************************
Desc:	Restores the last cursor position that was saved.
****************************************************************************/
RCODE flmCurRestorePosition(
	CURSOR *	pCursor
	)
{
	RCODE			rc = FERR_OK;
	SUBQUERY *	pSaveSubQuery = pCursor->pSaveSubQuery;

	if ((pCursor->pCurrSubQuery =
			pSaveSubQuery =
				pCursor->pSaveSubQuery) != NULL)
	{
		switch (pSaveSubQuery->OptInfo.eOptType)
		{
			case QOPT_USING_INDEX:
				if (RC_BAD( rc =
						pSaveSubQuery->pFSIndexCursor->restorePosition()))
				{
					goto Exit;
				}
				break;
			case QOPT_USING_PREDICATE:
				if (RC_BAD( rc = pSaveSubQuery->pPredicate->restorePosition()))
				{
					goto Exit;
				}
				break;
			case QOPT_SINGLE_RECORD_READ:
				pSaveSubQuery->bRecReturned = pSaveSubQuery->bSaveRecReturned;
				break;
			case QOPT_PARTIAL_CONTAINER_SCAN:
			case QOPT_FULL_CONTAINER_SCAN:
				if (RC_BAD( rc =
						pSaveSubQuery->pFSDataCursor->restorePosition()))
				{
					goto Exit;
				}
				break;
			default:
				break;
		}
	}
Exit:
	return( rc);
}

/****************************************************************************
Desc:	Sets a cursor to an absolute position.
****************************************************************************/
FSTATIC RCODE flmCurSetAbsolutePos(
	CURSOR *		pCursor,
	FLMUINT		uiPosition,
	FLMBOOL		bFallForward,
	FLMUINT *	puiPosition
	)
{
	RCODE					rc = FERR_OK;
	FDB *					pDb = NULL;
	FLMBOOL				bAbsPositionable;
	SUBQUERY *			pSubQuery;
	FSIndexCursor *	pFSIndexCursor = NULL;
	FLMBOOL				bSavedInvisTrans;
	FLMBOOL				bPassedCriteria;
	FLMBOOL				bDoRecMatch;
	FLMUINT				uiDrn;

	// Verify that this is an absolute positionable query.

	if (RC_BAD( rc = flmCurAbsPositionable( pCursor, &bAbsPositionable)))
	{
		goto Exit;
	}
	if (!bAbsPositionable)
	{
		rc = RC_SET( FERR_NOT_IMPLEMENTED);
		goto Exit;
	}

	pDb = pCursor->pDb;
	if (RC_BAD(rc = flmCurDbInit( pCursor)))
	{
		goto Exit;
	}

	// See if we are to position to EOF or BOF.

	if (uiPosition == (FLMUINT)(-1))
	{
		if (RC_BAD( rc = flmCurPosToEOF( pCursor)))
		{
			goto Exit;
		}
		*puiPosition = (FLMUINT)(-1);
	}
	else if (uiPosition == 0)
	{
		if (RC_BAD( rc = flmCurPosToBOF( pCursor)))
		{
			goto Exit;
		}
		*puiPosition = 0;
	}
	else
	{

		// Set absolute position

		pSubQuery = pCursor->pSubQueryList;
		if (pSubQuery->OptInfo.eOptType == QOPT_USING_INDEX)
		{
			pFSIndexCursor = pSubQuery->pFSIndexCursor;

			if (RC_OK( rc = pFSIndexCursor->setAbsolutePosition( pDb, uiPosition)))
			{

				// Get the current key so we can test it below

				if (RC_BAD( rc = pFSIndexCursor->currentKey( pDb, &pSubQuery->pRec,
												&pSubQuery->uiDrn)))
				{
					goto Exit;
				}
				pSubQuery->bFirstReference = FALSE;

				// These should have been set by the call to currentKey.

				flmAssert( pSubQuery->pRec->getContainerID() ==
								pCursor->uiContainer);
				flmAssert( pSubQuery->pRec->getID() == pSubQuery->uiDrn);

				pSubQuery->bRecIsAKey = TRUE;
				pSubQuery->uiCurrKeyMatch = FLM_UNK;
			}
			else	// RC_BAD( rc)
			{
				if (rc != FERR_EOF_HIT)
				{
					flmAssert( rc != FERR_BOF_HIT);
					goto Exit;
				}

				// rc == FERR_EOF_HIT.  If bFallForward is TRUE, we simply
				// set our position to EOF and return.  Otherwise, we must
				// fall backward to find the last key that passes the query.

				if (bFallForward)
				{
					*puiPosition = (FLMUINT)(-1);
					pCursor->uiLastRecID = 0;
					pCursor->ReadRc = rc;
					rc = FERR_OK;
					goto Exit;
				}

				// Position to last - use CB_ENTER and CB_EXIT so that
				// transaction state of pDb is preserved.  We need
				// to pass pDb into the getAbsolutePosition call below, and
				// if we had an invisible transaction going, it still needs
				// to be going below.

				CB_ENTER( pDb, &bSavedInvisTrans);
				rc = flmCurSearch( FLM_CURSOR_CONFIG, pCursor, TRUE,
										FALSE, NULL, NULL, NULL, &uiDrn);
				CB_EXIT( pDb, bSavedInvisTrans);
				if (RC_BAD( rc))
				{
					if (rc == FERR_BOF_HIT)
					{
						*puiPosition = 0;
						pCursor->uiLastRecID = 0;
						pCursor->ReadRc = rc;
						rc = FERR_OK;
					}
				}
				else
				{
					pCursor->ReadRc = FERR_OK;
					pCursor->uiLastRecID = uiDrn;
					rc = pFSIndexCursor->getAbsolutePosition( pDb,
												puiPosition);
				}
				goto Exit;
			}

			// Need to verify that the key we are on will pass the query criteria.
			// If not, fall forward or backward until we find one that does.

			bPassedCriteria = FALSE;
			bDoRecMatch = TRUE;
			if (pSubQuery->OptInfo.bDoKeyMatch)
			{
				if (RC_BAD( rc = flmCurEvalCriteria( pCursor, pSubQuery,
											pSubQuery->pRec, TRUE,
											&pSubQuery->uiCurrKeyMatch)))
				{
					if (rc == FERR_TRUNCATED_KEY)
					{
						pSubQuery->uiCurrKeyMatch = FLM_UNK;
						rc = FERR_OK;
					}
					else
					{
						goto Exit;
					}
				}
				else if (pSubQuery->uiCurrKeyMatch == FLM_TRUE)
				{
					bDoRecMatch = FALSE;
					bPassedCriteria = TRUE;
				}
				else if (pSubQuery->uiCurrKeyMatch != FLM_UNK)
				{
					flmAssert( pSubQuery->uiCurrKeyMatch == FLM_FALSE);
					bDoRecMatch = FALSE;

					// If we must evaluate DRN fields, we need to go to the
					// next reference, so we set bFirstReference to FALSE.
					// Otherwise, we set it to TRUE so it will skip to the
					// next key below.

					pSubQuery->bFirstReference = !pSubQuery->bHaveDrnFlds;
				}
			}
		}
		else	// eOptType == QOPT_USING_PREDICATE
		{
			FLMUINT	uiSeconds;

			flmAssert( pSubQuery->OptInfo.eOptType == QOPT_USING_PREDICATE);

			if (pCursor->uiTimeLimit)
			{
				uiSeconds = FLM_TIMER_UNITS_TO_SECS( pCursor->uiTimeLimit);
				if (!uiSeconds)
				{
					uiSeconds = 1;
				}
			}
			else
			{
				uiSeconds = 0;
			}
			CB_ENTER( pDb, &bSavedInvisTrans);
			rc = pSubQuery->pPredicate->positionToAbs( (HFDB)pDb, uiPosition,
										bFallForward, uiSeconds,
										puiPosition, &pSubQuery->uiDrn);
			CB_EXIT( pDb, bSavedInvisTrans);
			if (RC_BAD( rc))
			{
				goto Exit;
			}

			// If we get back EOF or BOF, set things up accordingly.

			if (*puiPosition == 0)
			{
				pCursor->ReadRc = RC_SET( FERR_BOF_HIT);
				pCursor->uiLastRecID = 0;
				goto Exit;
			}
			if (*puiPosition == (FLMUINT)(-1))
			{
				pCursor->ReadRc = RC_SET( FERR_EOF_HIT);
				pCursor->uiLastRecID = 0;
				goto Exit;
			}

			// Verify that the record passes the rest of criteria.

			bPassedCriteria = FALSE;
			bDoRecMatch = TRUE;
		}

		// Need to verify that the record we positioned to
		// passes the query criteria, unless we were able to
		// pass the key up above.  If we are using a predicate,
		// bDoRecMatch is always set to TRUE.

		if (bDoRecMatch)
		{
			FLMUINT	uiRecMatch;

			// Retrieve the record

			if (RC_BAD( rc = flmRcaRetrieveRec( pDb, NULL, pCursor->uiContainer,
							pSubQuery->uiDrn, TRUE, NULL, NULL, &pSubQuery->pRec)))
			{
				if (rc == FERR_NOT_FOUND && pFSIndexCursor)
				{
					rc = RC_SET( FERR_NO_REC_FOR_KEY);
				}
				goto Exit;
			}
			if (pFSIndexCursor)
			{
				pSubQuery->bRecIsAKey = FALSE;
			}

			// Evaluate the record against the query criteria

			if (RC_BAD( rc = flmCurEvalCriteria( pCursor, pSubQuery,
									pSubQuery->pRec, FALSE, &uiRecMatch)))
			{
				goto Exit;
			}

			bPassedCriteria = (FLMBOOL)((uiRecMatch == FLM_TRUE)
													 ? (FLMBOOL)TRUE
													 : (FLMBOOL)FALSE);
		}

		// At this point, we know that the key passed our
		// query criteria.  Verify that it passes the record
		// validator callback, if any.

		if (pCursor->fnRecValidator && bPassedCriteria)
		{
			CB_ENTER( pDb, &bSavedInvisTrans);
			bPassedCriteria = (pCursor->fnRecValidator)( FLM_CURSOR_CONFIG,
							(HFDB)pDb, pCursor->uiContainer,
							pSubQuery->pRec, NULL, pCursor->RecValData, &rc);
			CB_EXIT( pDb, bSavedInvisTrans);
		}

		if (bPassedCriteria)
		{
			pCursor->ReadRc = FERR_OK;
			pCursor->uiLastRecID = pSubQuery->uiDrn;
			*puiPosition = uiPosition;
		}
		else
		{

			// Position to next or previous - use CB_ENTER and CB_EXIT
			// so that transaction state of pDb is preserved.  We need
			// to pass pDb into the getAbsolutePosition call below, and
			// if we had an invisible transaction going, it still needs
			// to be going below.

			CB_ENTER( pDb, &bSavedInvisTrans);
			rc = flmCurSearch( FLM_CURSOR_CONFIG, pCursor, FALSE,
									bFallForward, NULL, NULL, NULL, &uiDrn);
			CB_EXIT( pDb, bSavedInvisTrans);
			if (RC_BAD( rc))
			{
				if (rc == FERR_BOF_HIT || rc == FERR_EOF_HIT)
				{
					*puiPosition = (FLMUINT)((rc == FERR_BOF_HIT)
													 ? (FLMUINT)0
													 : (FLMUINT)(-1));
					pCursor->uiLastRecID = 0;
					pCursor->ReadRc = rc;
					rc = FERR_OK;
				}
				goto Exit;
			}
			else
			{
				pCursor->ReadRc = FERR_OK;
				pCursor->uiLastRecID = uiDrn;
				if (pFSIndexCursor)
				{
					if (RC_BAD( rc = pFSIndexCursor->getAbsolutePosition( pDb,
												puiPosition)))
					{
						goto Exit;
					}
				}
				else
				{
					CB_ENTER( pDb, &bSavedInvisTrans);
					rc = pSubQuery->pPredicate->getAbsPosition( (HFDB)pDb,
										puiPosition);
					CB_EXIT( pDb, bSavedInvisTrans);
					if (RC_BAD( rc))
					{
						goto Exit;
					}
				}
			}
		}
	}

Exit:
	if (pDb)
	{
		flmExit( FLM_CURSOR_CONFIG, pDb, rc);
	}
	return( pCursor->rc = rc);
}

/****************************************************************************
Desc : Allows configuration of cursor attributes, including assignment of
		 QuickFinder strings, indexes and search records.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmCursorConfig(
	HFCURSOR				hCursor,
	eCursorConfigType	eConfigType,
	void *				Value1,
	void *				Value2
	)
{
	RCODE			rc = FERR_OK;
	CURSOR *		pCursor = (CURSOR *)hCursor;

	if (!pCursor)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_INVALID_PARM);
		goto Exit;
	}

	if( RC_BAD( rc = flmCheckDatabaseState( pCursor->pDb)))
	{
		goto Exit;
	}

	switch( eConfigType)
	{
		case FCURSOR_ALLOW_DUPS:
			if (pCursor->pCSContext)
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
				goto Exit;
			}

			if (pCursor->pDRNSet)
			{
				pCursor->pDRNSet->Release();
			}
			pCursor->pDRNSet = NULL;
			pCursor->bEliminateDups = FALSE;
			break;

		case FCURSOR_ELIMINATE_DUPS:
			if (pCursor->pCSContext)
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
				goto Exit;
			}

			if (pCursor->pDRNSet)
			{
				pCursor->pDRNSet->Release();
			}
			pCursor->pDRNSet = NULL;
			pCursor->bEliminateDups = TRUE;
			break;

		case FCURSOR_CLEAR_QUERY:
			if ((pCursor->pCSContext) && (pCursor->uiCursorId != FCS_INVALID_ID))
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
				goto Exit;
			}
			flmCurClearSelect( pCursor);
			break;

		case FCURSOR_GEN_POS_KEYS:
			if ((pCursor->pCSContext) && (pCursor->uiCursorId != FCS_INVALID_ID))
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
				goto Exit;
			}
			rc = flmCurSetupPosKeyArray( pCursor);
			break;

		case FCURSOR_SET_HDB:
			if ((pCursor->pCSContext) && (pCursor->uiCursorId != FCS_INVALID_ID))
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
				goto Exit;
			}
			flmCurFinishTrans( pCursor);
			pCursor->pDb = (FDB *)Value1;
			break;

		case FCURSOR_DISCONNECT:
			flmCurFinishTransactions( pCursor, TRUE);
			break;

		case FCURSOR_RETURN_KEYS_OK:
			pCursor->bOkToReturnKeys = (FLMBOOL)((FLMBOOL)(FLMUINT)Value1 ? TRUE : FALSE);
			break;

		case FCURSOR_SET_FLM_IX:
			if ((pCursor->pCSContext) && (pCursor->uiCursorId != FCS_INVALID_ID))
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
				goto Exit;
			}

			if (pCursor->bOptimized)
			{
				rc = RC_SET( FERR_ILLEGAL_OP);
			}
			else
			{
				pCursor->uiIndexNum = (FLMUINT)Value1;
			}
			break;

		case FCURSOR_SET_OP_TIME_LIMIT:
		{
			FLMUINT		uiTimeLimit;

			if (pCursor->pCSContext)
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
				goto Exit;
			}

			uiTimeLimit = pCursor->uiTimeLimit = (FLMUINT)Value1;
			if (uiTimeLimit)
			{
				pCursor->uiTimeLimit = FLM_SECS_TO_TIMER_UNITS( uiTimeLimit);
			}
			break;
		}
		case FCURSOR_SET_PERCENT_POS:
			if (pCursor->pCSContext)
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
				goto Exit;
			}

			if (RC_OK( rc = flmCurSetPercentPos( pCursor, (FLMUINT)Value1)))
			{
				pCursor->ReadRc = FERR_OK;
			}
			break;

		case FCURSOR_SET_ABS_POS:
			if (pCursor->pCSContext)
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
				goto Exit;
			}
			rc = flmCurSetAbsolutePos( pCursor,
									*((FLMUINT *)Value1),
									(FLMBOOL)(FLMUINT)Value2,
									(FLMUINT *)Value1);
			break;

		case FCURSOR_SET_POS:
		{
			CURSOR *      pPosCursor;

			if ((pPosCursor = (CURSOR *)Value1) == NULL)
			{
				flmAssert( 0);
				rc = RC_SET( FERR_INVALID_PARM);
				goto Exit;
			}

			if ((pCursor->pCSContext) || (pPosCursor->pCSContext))
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
				goto Exit;
			}

			if (RC_OK( rc = flmCurSetPos( pCursor, pPosCursor)))
			{
				pCursor->uiLastRecID = pPosCursor->uiLastRecID;
			}
			break;
		}
		case FCURSOR_SET_POS_FROM_DRN:
			rc = flmCurSetPosFromDRN( pCursor, (FLMUINT)Value1);
			break;

		case FCURSOR_SET_REC_TYPE:
			if ((pCursor->pCSContext) && (pCursor->uiCursorId != FCS_INVALID_ID))
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
				goto Exit;
			}

			if (pCursor->bOptimized)
			{
				rc = RC_SET( FERR_ILLEGAL_OP);
			}
			else
			{
				pCursor->uiRecType = (FLMUINT)Value1;
			}
			break;

		case FCURSOR_SET_REC_VALIDATOR:
			pCursor->fnRecValidator = (REC_VALIDATOR_HOOK)((FLMUINT)Value1);
			pCursor->RecValData = Value2;
			break;

		case FCURSOR_SET_STATUS_HOOK:
			pCursor->fnStatus = (STATUS_HOOK)((FLMUINT)Value1);
			pCursor->StatusData = Value2;
			pCursor->uiLastCBTime = FLM_GET_TIMER();
			break;

		case FCURSOR_SAVE_POSITION:
			rc = flmCurSavePosition( pCursor);
			break;

		case FCURSOR_RESTORE_POSITION:
			rc = flmCurRestorePosition( pCursor);
			break;

		default:
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			break;
	}

Exit:

	return( rc);
}

/************************************************************************
Desc : Returns whether or not a query is positionable.
*************************************************************************/
FSTATIC RCODE flmCurPositionable(
	CURSOR *		pCursor,
	FLMBOOL *	pbPositionable)
{
	RCODE			rc = FERR_OK;
	SUBQUERY *	pSubQuery;

	*pbPositionable = FALSE;

	// Optimize if necessary.

	if (!pCursor->bOptimized)
	{
		if( RC_BAD( rc = flmCurPrep( pCursor)))
		{
			goto Exit;
		}
	}

	// Must use an index and only one index.  Also, the bDoRecMatch
	// flag must be FALSE, because we must be able to evaluate the
	// query entirely using only the fields in the index.  Finally,
	// we must not need to evaluate DRN fields. Normally the
	// bHaveDrnFlds would not be a problem for evaluating a criteria
	// using only an index key, but we do not have DRNs at the higher
	// levels of the B-Tree, so having a DRN in the selection criteria
	// ruins positionability.

	if (((pSubQuery = pCursor->pSubQueryList) != NULL) &&
		 (!pSubQuery->pNext) &&
		 (pSubQuery->OptInfo.eOptType == QOPT_USING_INDEX) &&
		 (!pSubQuery->OptInfo.bDoRecMatch) &&
		 (!pSubQuery->bHaveDrnFlds))
	{
		*pbPositionable = TRUE;
	}
Exit:
	return( rc);
}

/************************************************************************
Desc : Returns whether or not a query is absolute positionable.
*************************************************************************/
FSTATIC RCODE flmCurAbsPositionable(
	CURSOR *		pCursor,
	FLMBOOL *	pbAbsPositionable)
{
	RCODE			rc = FERR_OK;
	SUBQUERY *	pSubQuery;
	FLMBOOL		bSavedInvisTrans;

	*pbAbsPositionable = FALSE;

	// Optimize if necessary.

	if (!pCursor->bOptimized)
	{
		if( RC_BAD( rc = flmCurPrep( pCursor)))
		{
			goto Exit;
		}
	}

	// Must have only one subquery that either uses an index that
	// supports absolute positioning or uses a predicate that
	// supports it.

	if (((pSubQuery = pCursor->pSubQueryList) != NULL) &&
		 !pSubQuery->pNext)
	{
		if (pSubQuery->OptInfo.eOptType == QOPT_USING_INDEX)
		{
			*pbAbsPositionable =
				pSubQuery->pFSIndexCursor->isAbsolutePositionable();
		}
		else if (pSubQuery->OptInfo.eOptType == QOPT_USING_PREDICATE)
		{
			CB_ENTER( pCursor->pDb, &bSavedInvisTrans);
			rc = pSubQuery->pPredicate->isAbsPositionable( (HFDB)pCursor->pDb,
										pbAbsPositionable);
			CB_EXIT( pCursor->pDb, bSavedInvisTrans);
			if (RC_BAD( rc))
			{
				goto Exit;
			}
		}
	}
Exit:
	return( rc);
}

/************************************************************************
Desc : Returns absolute position of cursor.
*************************************************************************/
FSTATIC RCODE flmCurGetAbsolutePos(
	CURSOR *		pCursor,
	FLMUINT *	puiPosition
	)
{
	RCODE		rc = FERR_OK;
	FDB *		pDb = NULL;
	FLMBOOL	bAbsPositionable;
	FLMBOOL	bSavedInvisTrans;

	// Verify that this is an absolute positionable query.

	if (RC_BAD( rc = flmCurAbsPositionable( pCursor, &bAbsPositionable)))
	{
		goto Exit;
	}
	if (!bAbsPositionable)
	{
		rc = RC_SET( FERR_NOT_IMPLEMENTED);
		goto Exit;
	}

	pDb = pCursor->pDb;
	if (RC_BAD(rc = flmCurDbInit( pCursor)))
	{
		goto Exit;
	}

	// See if we are positioned on EOF or BOF.  Return special values
	// for these.

	if (!pCursor->uiLastRecID)
	{
		if (pCursor->ReadRc == FERR_EOF_HIT)
		{
			*puiPosition = (FLMUINT)(-1);	// EOF
		}
		else
		{
			*puiPosition = 0;		// BOF
		}
	}
	else
	{
		if (pCursor->pSubQueryList->OptInfo.eOptType == QOPT_USING_INDEX)
		{

			// Get absolute position from the first sub-query's index cursor.

			if (RC_BAD( rc =
					pCursor->pSubQueryList->pFSIndexCursor->getAbsolutePosition( pDb,
							puiPosition)))
			{
				goto Exit;
			}
		}
		else	// eOptType == QOPT_USING_PREDICATE
		{
			flmAssert( pCursor->pSubQueryList->OptInfo.eOptType ==
							QOPT_USING_PREDICATE);

			CB_ENTER( pDb, &bSavedInvisTrans);
			rc = pCursor->pSubQueryList->pPredicate->getAbsPosition(
								(HFDB)pDb, puiPosition);
			CB_EXIT( pDb, bSavedInvisTrans);
		}
	}

Exit:
	if (pDb)
	{
		flmExit( FLM_CURSOR_GET_CONFIG, pDb, rc);
	}
	return( pCursor->rc = rc);
}

/************************************************************************
Desc : Returns absolute count for the cursor's index.  NOTE: This is
		 not necessarily the same thing as the number of records that
		 will pass the filter criteria.
*************************************************************************/
FSTATIC RCODE flmCurGetAbsoluteCount(
	CURSOR *		pCursor,
	FLMUINT *	puiCount
	)
{
	RCODE		rc = FERR_OK;
	FDB *		pDb = NULL;
	FLMBOOL	bAbsPositionable;
	FLMBOOL	bTotalEstimated;
	FLMBOOL	bSavedInvisTrans;

	// Verify that this is an absolute positionable query.

	if (RC_BAD( rc = flmCurAbsPositionable( pCursor, &bAbsPositionable)))
	{
		goto Exit;
	}
	if (!bAbsPositionable)
	{
		rc = RC_SET( FERR_NOT_IMPLEMENTED);
		goto Exit;
	}

	pDb = pCursor->pDb;
	if (RC_BAD(rc = flmCurDbInit( pCursor)))
	{
		goto Exit;
	}

	if (pCursor->pSubQueryList->OptInfo.eOptType == QOPT_USING_INDEX)
	{

		// Get absolute count from the first sub-query's index cursor.

		if (RC_BAD( rc =
				pCursor->pSubQueryList->pFSIndexCursor->getTotalReferences( pDb,
							puiCount, &bTotalEstimated)))
		{
			goto Exit;
		}
	}
	else	// eOptType == QOPT_USING_PREDICATE
	{
		flmAssert( pCursor->pSubQueryList->OptInfo.eOptType ==
						QOPT_USING_PREDICATE);

		CB_ENTER( pDb, &bSavedInvisTrans);
		rc = pCursor->pSubQueryList->pPredicate->getAbsCount( (HFDB)pDb,
								puiCount);
		CB_EXIT( pDb, bSavedInvisTrans);
	}

Exit:
	if (pDb)
	{
		flmExit( FLM_CURSOR_GET_CONFIG, pDb, rc);
	}
	return( pCursor->rc = rc);
}

/****************************************************************************
Desc:	Returns FLAIM cursor configuration values.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmCursorGetConfig(
	HFCURSOR						hCursor,
	eCursorGetConfigType		eGetConfigType,
	void *						Value1,
	void *						Value2)
{
	RCODE			rc = FERR_OK;
	CURSOR *		pCursor = (CURSOR *)hCursor;

	if (!pCursor)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_INVALID_PARM);
		goto Exit;
	}

	if( RC_BAD( rc = flmCheckDatabaseState( pCursor->pDb)))
	{
		goto Exit;
	}

	switch( eGetConfigType)
	{
		case FCURSOR_GET_PERCENT_POS:
		{
			if (pCursor->pCSContext)
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
			}
			else
			{
				rc = flmCurGetPercentPos( pCursor, (FLMUINT *)Value1);
			}
			
			break;
		}
			
		case FCURSOR_GET_ABS_POS:
		{
			if (pCursor->pCSContext)
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
			}
			else
			{
				rc = flmCurGetAbsolutePos( pCursor, (FLMUINT *)Value1);
			}
			
			break;
		}
			
		case FCURSOR_GET_ABS_COUNT:
		{
			if (pCursor->pCSContext)
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
			}
			else
			{
				rc = flmCurGetAbsoluteCount( pCursor, (FLMUINT *)Value1);
			}
			
			break;
		}
		
		case FCURSOR_GET_OPT_INFO_LIST:
		{
			if (pCursor->pCSContext)
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
			}
			else
			{
				OPT_INFO *	pOptInfoArray = (OPT_INFO *)Value1;
				FLMUINT *	puiSubQueryCnt = (FLMUINT *)Value2;
				FLMUINT		uiSubQueryCnt = 0;
				SUBQUERY	*	pSubQuery;

				if (!pCursor->bOptimized)
				{
					if (RC_BAD( rc = flmCurPrep( pCursor)))
					{
						goto Exit;
					}
				}
				
				for (pSubQuery = pCursor->pSubQueryList;
					  pSubQuery;
					  pSubQuery = pSubQuery->pNext)
				{
					if (pOptInfoArray)
					{
						f_memcpy( &pOptInfoArray[ uiSubQueryCnt],
										&pSubQuery->OptInfo, sizeof( OPT_INFO));   
					}
					
					uiSubQueryCnt++;
				}
				
				*puiSubQueryCnt = uiSubQueryCnt;
			}
			
			break;
		}
		
		case FCURSOR_GET_OPT_INFO:
		{
			if (pCursor->pCSContext)
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
			}
			else
			{
				if (!pCursor->bOptimized)
				{
					if (RC_BAD( rc = flmCurPrep( pCursor)))
					{
						goto Exit;
					}
				}
				
				if (pCursor->pSubQueryList)
				{
					f_memcpy( (OPT_INFO *)Value2,
						&pCursor->pSubQueryList->OptInfo, sizeof( OPT_INFO));
				}
			}
			
			break;
		}
		
		case FCURSOR_GET_FLM_IX:
		{
			if (pCursor->pCSContext)
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
			}
			else
			{
				FLMUINT	uiIxNum = 0;
				FLMUINT	uiIndexInfo = HAVE_NO_INDEX;
				FLMUINT	uiSubqueryCount = 0;

				if (!pCursor->bOptimized)
				{
					if (RC_BAD( rc = flmCurPrep( pCursor)))
					{
						goto Exit;
					}
				}

				if( pCursor->pSubQueryList)
				{
					SUBQUERY *		pTmpSubQuery = pCursor->pSubQueryList;
					FLMUINT			uiSQIndex = 0;

					while( pTmpSubQuery)
					{
						uiSubqueryCount++;

						if (pTmpSubQuery->OptInfo.eOptType == QOPT_USING_INDEX)
						{
							uiSQIndex = pTmpSubQuery->OptInfo.uiIxNum;
						}
						else if (pTmpSubQuery->OptInfo.eOptType ==
										QOPT_USING_PREDICATE)
						{
							FLMUINT	uiTmpIndexInfo;

							uiSQIndex = pTmpSubQuery->pPredicate->getIndex( 
												&uiTmpIndexInfo);
												
							if (uiTmpIndexInfo == HAVE_MULTIPLE_INDEXES)
							{
								if (!uiIxNum)
								{
									uiIxNum = uiSQIndex;
								}
								uiIndexInfo = HAVE_MULTIPLE_INDEXES;
								break;
							}
							else if (uiTmpIndexInfo == HAVE_ONE_INDEX_MULT_PARTS)
							{
								if (uiIxNum && uiSQIndex != uiIxNum)
								{
									uiIndexInfo = HAVE_MULTIPLE_INDEXES;
									break;
								}
								else
								{
									// NOTE: At this point we know that uiIxNum is either
									// zero or equal to uiSQIndex, so by
									// assigning uiIxNum to uiSQIndex it will cause us to
									// assign uiIndexInfo to HAVE_ONE_INDEX_MULT_PARTS
									// below.

									flmAssert( uiSQIndex);
									uiIxNum = uiSQIndex;
								}
							}
						}

						if (uiSQIndex)
						{
							if( !uiIxNum)
							{
								uiIxNum = uiSQIndex;
								if( uiSubqueryCount > 1)
								{
									uiIndexInfo = HAVE_ONE_INDEX_MULT_PARTS;
								}
								else
								{
									uiIndexInfo = HAVE_ONE_INDEX;
								}
							}
							else if( uiIxNum != uiSQIndex)
							{
								uiIndexInfo = HAVE_MULTIPLE_INDEXES;
								break;
							}
							else
							{
								uiIndexInfo = HAVE_ONE_INDEX_MULT_PARTS;
							}
						}
						else if( uiIxNum)
						{
							uiIndexInfo = HAVE_ONE_INDEX_MULT_PARTS;
						}
						pTmpSubQuery = pTmpSubQuery->pNext;
					}
				}

				if( Value1)
				{
					*((FLMUINT *)Value1) = uiIxNum;
				}

				if( Value2)
				{
					*((FLMUINT *)Value2) = uiIndexInfo;
				}
			}
			
			break;
		}
		
		case FCURSOR_GET_REC_TYPE:
		{
			*((FLMUINT *)Value1) = pCursor->uiRecType;
			break;
		}
		
		case FCURSOR_GET_FLAGS:
		{
			*((FLMUINT *)Value1) = pCursor->QTInfo.uiFlags;
			break;
		}
		
		case FCURSOR_GET_STATE:
		{
			*((FLMUINT *)Value1) = 0;

			if (pCursor->QTInfo.pTopNode ||
				 pCursor->QTInfo.pCurOpNode ||
				 pCursor->QTInfo.pCurAtomNode)
			{
				*((FLMUINT *)Value1) |= FCURSOR_HAVE_CRITERIA;
			}
			
			if (pCursor->QTInfo.uiExpecting & FLM_Q_OPERATOR)
			{
				*((FLMUINT *)Value1) |= FCURSOR_EXPECTING_OPERATOR;
			}
			
			if ((pCursor->QTInfo.uiNestLvl == 0) ||
				 ((pCursor->QTInfo.uiExpecting & FLM_Q_OPERATOR) &&
					pCursor->QTInfo.pTopNode))
			{
				*((FLMUINT *)Value1) |= FCURSOR_QUERY_COMPLETE;
			}
			
			if (pCursor->bOptimized)
			{
				*((FLMUINT *)Value1) |=
					(FCURSOR_QUERY_OPTIMIZED | FCURSOR_READ_PERFORMED);
			}
			
			break;
		}
		
		case FCURSOR_GET_POSITIONABLE:
		{
			if (pCursor->pCSContext)
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
			}
			else
			{
				rc = flmCurPositionable( pCursor, (FLMBOOL *)Value1);
			}
			
			break;
		}
		
		case FCURSOR_GET_ABS_POSITIONABLE:
		{
			if (pCursor->pCSContext)
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
			}
			else
			{
				rc = flmCurAbsPositionable( pCursor, (FLMBOOL *)Value1);
			}
			
			break;
		}
		
		case FCURSOR_AT_BOF:
		{
			if (pCursor->pCSContext)
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
			}
			else
			{
				if (!pCursor->bOptimized)
				{
					if (RC_BAD( rc = flmCurPrep( pCursor)))
					{
						goto Exit;
					}
				}
				
				*((FLMBOOL *)Value1) = (FLMBOOL)((!pCursor->uiLastRecID &&
															 pCursor->ReadRc == FERR_BOF_HIT)
															? (FLMBOOL)TRUE
															: (FLMBOOL)FALSE);
			}
			
			break;
		}
		
		case FCURSOR_AT_EOF:
		{
			if (pCursor->pCSContext)
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
			}
			else
			{
				if (!pCursor->bOptimized)
				{
					if (RC_BAD( rc = flmCurPrep( pCursor)))
					{
						goto Exit;
					}
				}
				
				*((FLMBOOL *)Value1) = (FLMBOOL)((!pCursor->uiLastRecID &&
															 pCursor->ReadRc == FERR_EOF_HIT)
															? (FLMBOOL)TRUE
															: (FLMBOOL)FALSE);
			}
			
			break;
		}
		
		default:
		{
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			break;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: Given a IFD this function will verify that the input path matches and
		that it is in the same position (for compound keys)
****************************************************************************/
FSTATIC FLMBOOL flmCurMatchIndexPath( 
	IFD *			pIfd, 
	FLMUINT *	puiField, 
	FLMUINT		uiPos)
{
	FLMUINT *	puiIndexFieldPath;
	FLMBOOL		bIsMatch;

	// Are both fields at the same position within a compound key

	if (pIfd->uiCompoundPos != uiPos)
	{
		bIsMatch = FALSE;
		goto Exit;
	}

	// Check in PARENT to CHILD order.

	puiIndexFieldPath = pIfd->pFieldPathPToC;
	while (*puiIndexFieldPath)
	{
		if (*puiField != *puiIndexFieldPath)
		{
			bIsMatch = FALSE;
			goto Exit;
		}
		puiIndexFieldPath++;
		puiField++;
	}
	bIsMatch = (*puiField == 0) ? TRUE : FALSE;
Exit:
	return( bIsMatch);
}

/****************************************************************************
Desc:		Uses the specified field path[s] to find a matching ordering index.
Note:		FlmCursorConfig( type == FCURSOR_SET_FLM_IX) and FlmCursorSetOrderIndex
			cannot both be called for the same cursor (they will override each 
			other).
Warning:	The index selected from this call will be used for query optimization
			in addition to ordering the results. This could result in slower
			query performance.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmCursorSetOrderIndex(
	HFCURSOR		hCursor,
	FLMUINT *	puiFieldPaths,		/* List of field paths to match on. Each path
												is terminated with a single 0, and the 
												entire list is terminated by two 0's */	
	FLMUINT *	puiIndexRV)			/* [optional] index id of matching index.
												A value of 0 indicates that no match
												was found. */
{
	RCODE			rc = FERR_OK;
	CURSOR *    pCursor = (CURSOR *)hCursor;
	FDB *			pDb = NULL;
	IFD *			pIfd;
	IFD *			pIfd2;
	IXD *			pIxd;
	FLMUINT *	puiField = puiFieldPaths;
	FLMUINT		uiPos;
	FLMUINT		uiScore;
	FLMUINT		uiBestScore = 0;
	FLMUINT		uiBestIndex = 0;

	if (!pCursor)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_INVALID_PARM);
		goto Exit;
	}

	flmAssert( puiFieldPaths != NULL);
	flmAssert( puiFieldPaths[ 0] != 0);	// must give at least one field 

	if (puiIndexRV)
	{
		*puiIndexRV = 0;
	}

	pDb = pCursor->pDb;
	if (RC_BAD( rc = flmCurDbInit( pCursor)))
	{
		goto Exit;
	}

	// Position to the last child field in the first path.

	while (*puiField)
	{
		puiField++;
	}
	puiField--;	

	// Is the first field even indexed?

	if (RC_BAD( rc = fdictGetField( pDb->pDict, *puiField, NULL,
								&pIfd, NULL)))
	{
		goto Exit;
	}
	if (!pIfd)
	{	
		// First field was not indexed (at all) so return a index id of ZERO.

		goto Exit;
	}

	// Loop through all indexes on the first field.

	for ( ; pIfd; pIfd = pIfd->pNextInChain)
	{
		uiScore = 0;
		
		pIxd = pIfd->pIxd;

		// If index is on a different container then skip this index.
		// NOTE: If pIxd->uiContainerNum is zero, it covers all
		// containers.

		if (pIxd->uiContainerNum &&
			 pIxd->uiContainerNum != pCursor->uiContainer)
		{
			continue;
		}

		// Verify that this index contains all requested field paths

		pIfd2 = pIxd->pFirstIfd;
		for (uiPos = 0, puiField = puiFieldPaths;
			  *puiField;
			  pIfd2++, uiPos++)
		{
			if (!flmCurMatchIndexPath( pIfd2, puiField, uiPos))
			{
				goto NextIndex;
			}

			// The path matching requirements has been meet now score this field
			// Scoring Rules for each field path:
			//		Value Index			4 points 
			//		SubSring/EachWord	2 points - index provides some ordering
			//		Context Index		0 points - index provides no ordering
			// Scoring Boost			1 point - when # of field paths in index match
			//												# of field paths supplied. 

			if (pIfd2->uiFlags & IFD_CONTEXT)
			{
				;		// Context index gets no score. 
			}
			else if (pIfd2->uiFlags & IFD_SUBSTRING)
			{
				uiScore += 2;
			}
			else if (pIfd2->uiFlags & IFD_EACHWORD)
			{
				uiScore += 2;
			}
			else
			{
				uiScore += 4;
			}

			// Position to the the next field path

			while (*puiField)
			{
				puiField++;
			}
			puiField++;				// Skip single null terminator.

			// See if all components for index have been visited.

			if (pIfd2->uiFlags & IFD_LAST)
			{
				break;
			}
		}

		// If the number of requested fields equals the number of fields
		//	contained within the index then add in a exact match score.

		if (pIfd2->uiFlags & IFD_LAST)
		{	
			uiScore += 1;
		}

		// Always remember the highest index score

		if (uiBestScore < uiScore)
		{
			uiBestScore = uiScore;
			uiBestIndex = pIxd->uiIndexNum;
		}

NextIndex:
		;
	}

Exit:
	if (puiIndexRV)
	{
		*puiIndexRV = uiBestIndex;
		if (uiBestIndex && RC_OK( rc))
		{
			rc = FlmCursorConfig( hCursor, FCURSOR_SET_FLM_IX, 
						(void *) uiBestIndex, (void *) 0);
		}
	}

	if (pDb)
	{
		(void)fdbExit( pDb);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void flmCurFreeSQList(
	CURSOR *	pCursor,
	FLMBOOL	bFreeEverything)
{
	SUBQUERY *	pSubQuery;

	for( pSubQuery = pCursor->pSubQueryList;
		  pSubQuery;
		  pSubQuery = pSubQuery->pNext)
	{
		flmSQFree( pSubQuery, bFreeEverything);
	}

	if (bFreeEverything)
	{
		pCursor->SQPool.poolReset();
		pCursor->pSubQueryList = NULL;
	}
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT flmGetPathLen(
	FLMUINT *		pFldPath)
{
	FLMUINT			uiPathLen = 0;

	if( pFldPath)
	{
		for( ; uiPathLen < GED_MAXLVLNUM + 1; uiPathLen++)
		{
			if( !pFldPath[ uiPathLen])
				break;
		}
	}
	return uiPathLen;
}
