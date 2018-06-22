//------------------------------------------------------------------------------
// Desc: Native C routines to support C# Query class
// Tabs:	3
//
// Copyright (c) 2006-2007 Novell, Inc. All Rights Reserved.
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

#include "xflaim.h"
#include "flaimsys.h"

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_createQuery(
	FLMUINT32	ui32Collection,
	F_Query **	ppQuery)
{
	RCODE			rc = NE_XFLM_OK;
	F_Query *	pQuery;

	if ((pQuery = f_new F_Query) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	pQuery->setCollection( (FLMUINT)ui32Collection);

Exit:

	*ppQuery = pQuery;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_Query_Release(
	IF_Query *	pQuery)
{
	if (pQuery)
	{
		pQuery->Release();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_setLanguage(
	IF_Query *	pQuery,
	FLMUINT32	ui32Language)
{
	return( pQuery->setLanguage( (FLMUINT)ui32Language));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_setupQueryExpr(
	IF_Query *				pQuery,
	IF_Db *					pDb,
	const FLMUNICODE *	puzQueryExpr)
{
	return( pQuery->setupQueryExpr( pDb, puzQueryExpr));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_copyCriteria(
	IF_Query *	pQuery,
	IF_Query *	pQueryToCopy)
{
	return( pQuery->copyCriteria( pQueryToCopy));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_addXPathComponent(
	IF_Query *	pQuery,
	FLMUINT32	ui32XPathAxis,
	FLMUINT32	ui32NodeType,
	FLMUINT32	ui32NameId)
{
	return( pQuery->addXPathComponent( (eXPathAxisTypes)ui32XPathAxis,
							(eDomNodeType)ui32NodeType, (FLMUINT)ui32NameId));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_addOperator(
	IF_Query *	pQuery,
	FLMUINT32	ui32Operator,
	FLMUINT32	ui32CompareFlags)
{
	return( pQuery->addOperator( (eQueryOperators)ui32Operator,
							(FLMUINT)ui32CompareFlags));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_addStringValue(
	IF_Query *				pQuery,
	const FLMUNICODE *	puzValue)
{
	return( pQuery->addUnicodeValue( puzValue));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_addBinaryValue(
	IF_Query *		pQuery,
	const void *	pvValue,
	FLMINT32			i32ValueLen)
{
	return( pQuery->addBinaryValue( pvValue, (FLMUINT)i32ValueLen));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_addULongValue(
	IF_Query *	pQuery,
	FLMUINT64	ui64Value)
{
	return( pQuery->addUINT64Value( ui64Value));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_addLongValue(
	IF_Query *	pQuery,
	FLMINT64		i64Value)
{
	return( pQuery->addINT64Value( i64Value));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_addUIntValue(
	IF_Query *	pQuery,
	FLMUINT32	ui32Value)
{
	return( pQuery->addUINTValue( (FLMUINT)ui32Value));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_addIntValue(
	IF_Query *	pQuery,
	FLMINT32		i32Value)
{
	return( pQuery->addINTValue( (FLMINT)i32Value));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_addBoolean(
	IF_Query *	pQuery,
	FLMBOOL		bValue)
{
	return( pQuery->addBoolean( bValue, FALSE));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_addUnknown(
	IF_Query *	pQuery)
{
	return( pQuery->addBoolean( FALSE, TRUE));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_getFirst(
	IF_Query *		pQuery,
	IF_Db *			pDb,
	FLMUINT32		ui32TimeLimit,
	IF_DOMNode **	ppNode)
{
	return( pQuery->getFirst( pDb, ppNode, (FLMUINT)ui32TimeLimit));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_getLast(
	IF_Query *		pQuery,
	IF_Db *			pDb,
	FLMUINT32		ui32TimeLimit,
	IF_DOMNode **	ppNode)
{
	return( pQuery->getLast( pDb, ppNode, (FLMUINT)ui32TimeLimit));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_getNext(
	IF_Query *		pQuery,
	IF_Db *			pDb,
	FLMUINT32		ui32TimeLimit,
	IF_DOMNode **	ppNode)
{
	return( pQuery->getNext( pDb, ppNode, (FLMUINT)ui32TimeLimit));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_getPrev(
	IF_Query *		pQuery,
	IF_Db *			pDb,
	FLMUINT32		ui32TimeLimit,
	IF_DOMNode **	ppNode)
{
	return( pQuery->getPrev( pDb, ppNode, (FLMUINT)ui32TimeLimit));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_getCurrent(
	IF_Query *		pQuery,
	IF_Db *			pDb,
	IF_DOMNode **	ppNode)
{
	return( pQuery->getCurrent( pDb, ppNode));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_Query_resetQuery(
	IF_Query *	pQuery)
{
	pQuery->resetQuery();
}

// IMPORTANT NOTE: This structure must be kept in sync with the
// corresponding structure in C# code.
typedef struct
{
	FLMUINT32	ui32OptType;
	FLMUINT32	ui32Cost;
	FLMUINT64	ui64NodeId;
	FLMUINT64	ui64EndNodeId;
	char			szIxName [80];
	FLMUINT32	ui32IxNum;
	FLMBOOL		bMustVerifyPath;
	FLMBOOL		bDoNodeMatch;
	FLMBOOL		bCanCompareOnKey;
	FLMUINT64	ui64KeysRead;
	FLMUINT64	ui64KeyHadDupDoc;
	FLMUINT64	ui64KeysPassed;
	FLMUINT64	ui64NodesRead;
	FLMUINT64	ui64NodesTested;
	FLMUINT64	ui64NodesPassed;
	FLMUINT64	ui64DocsRead;
	FLMUINT64	ui64DupDocsEliminated;
	FLMUINT64	ui64NodesFailedValidation;
	FLMUINT64	ui64DocsFailedValidation;
	FLMUINT64	ui64DocsPassed;
} CS_XFLM_OPT_INFO;

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_getStatsAndOptInfo(
	IF_Query *			pQuery,
	XFLM_OPT_INFO **	ppOptInfoArray,
	FLMUINT32 *			pui32NumOptInfos)
{
	RCODE		rc = NE_XFLM_OK;
	FLMUINT	uiNumOptInfos = 0;

	*ppOptInfoArray = NULL;
	if (RC_BAD( rc = pQuery->getStatsAndOptInfo( &uiNumOptInfos, ppOptInfoArray)))
	{
		goto Exit;
	}

Exit:

	*pui32NumOptInfos = (FLMUINT32)uiNumOptInfos;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_Query_getOptInfo(
	XFLM_OPT_INFO *		pOptInfoArray,
	FLMUINT32				ui32InfoToGet,
	CS_XFLM_OPT_INFO *	pCSOptInfo)
{
	XFLM_OPT_INFO *	pOptInfo = &pOptInfoArray [ui32InfoToGet];

	pCSOptInfo->ui32OptType = (FLMUINT32)pOptInfo->eOptType;
	pCSOptInfo->ui32Cost = (FLMUINT32)pOptInfo->uiCost;
	pCSOptInfo->ui64NodeId = pOptInfo->ui64NodeId;
	pCSOptInfo->ui64EndNodeId = pOptInfo->ui64EndNodeId;
	f_memcpy( pCSOptInfo->szIxName, pOptInfo->szIxName, sizeof( pCSOptInfo->szIxName));
	pCSOptInfo->ui32IxNum = (FLMUINT32)pOptInfo->uiIxNum;
	pCSOptInfo->bMustVerifyPath = pOptInfo->bMustVerifyPath;
	pCSOptInfo->bDoNodeMatch = pOptInfo->bDoNodeMatch;
	pCSOptInfo->bCanCompareOnKey = pOptInfo->bCanCompareOnKey;
	pCSOptInfo->ui64KeysRead = pOptInfo->ui64KeysRead;
	pCSOptInfo->ui64KeyHadDupDoc = pOptInfo->ui64KeyHadDupDoc;
	pCSOptInfo->ui64KeysPassed = pOptInfo->ui64KeysPassed;
	pCSOptInfo->ui64NodesRead = pOptInfo->ui64NodesRead;
	pCSOptInfo->ui64NodesTested = pOptInfo->ui64NodesTested;
	pCSOptInfo->ui64NodesPassed = pOptInfo->ui64NodesPassed;
	pCSOptInfo->ui64DocsRead = pOptInfo->ui64DocsRead;
	pCSOptInfo->ui64DupDocsEliminated = pOptInfo->ui64DupDocsEliminated;
	pCSOptInfo->ui64NodesFailedValidation = pOptInfo->ui64NodesFailedValidation;
	pCSOptInfo->ui64DocsFailedValidation = pOptInfo->ui64DocsFailedValidation;
	pCSOptInfo->ui64DocsPassed = pOptInfo->ui64DocsPassed;
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_Query_setDupHandling(
	IF_Query *	pQuery,
	FLMBOOL		bRemoveDups)
{
	pQuery->setDupHandling( bRemoveDups);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_setIndex(
	IF_Query *	pQuery,
	FLMUINT32	ui32Index)
{
	return( pQuery->setIndex( (FLMUINT)ui32Index));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_getIndex(
	IF_Query *	pQuery,
	IF_Db *		pDb,
	FLMUINT32 *	pui32Index,
	FLMBOOL *	pbHaveMultiple)
{
	RCODE			rc;
	FLMUINT		uiIndex;
	
	rc = pQuery->getIndex( pDb, &uiIndex, pbHaveMultiple);
	*pui32Index = (FLMUINT32)uiIndex;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_addSortKey(
	IF_Query *	pQuery,
	void *		pvSortKeyContext,
	FLMBOOL		bChildToContext,
	FLMBOOL		bElement,
	FLMUINT32	ui32NameId,
	FLMUINT32	ui32CompareFlags,
	FLMUINT32	ui32Limit,
	FLMUINT32	ui32KeyComponent,
	FLMBOOL		bSortDescending,
	FLMBOOL		bSortMissingHigh,
	void **		ppvContext)
{
	return( pQuery->addSortKey( pvSortKeyContext,
				bChildToContext, bElement, (FLMUINT)ui32NameId,
				(FLMUINT)ui32CompareFlags, (FLMUINT)ui32Limit,
				(FLMUINT)ui32KeyComponent, bSortDescending, bSortMissingHigh,
				ppvContext));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_enablePositioning(
	IF_Query *	pQuery)
{
	return( pQuery->enablePositioning());
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_positionTo(
	IF_Query *		pQuery,
	IF_Db *			pDb,
	FLMUINT32		ui32TimeLimit,
	FLMUINT32		ui32Position,
	IF_DOMNode **	ppNode)
{
	return( pQuery->positionTo( pDb, ppNode, (FLMUINT)ui32TimeLimit, (FLMUINT)ui32Position));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_positionToByKey(
	IF_Query *			pQuery,
	IF_Db *				pDb,
	FLMUINT32			ui32TimeLimit,
	IF_DataVector *	pSearchKey,
	FLMUINT32			ui32RetrieveFlags,
	IF_DOMNode **		ppNode)
{
	return( pQuery->positionTo( pDb, ppNode, (FLMUINT)ui32TimeLimit,
						pSearchKey, (FLMUINT)ui32RetrieveFlags));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_getPosition(
	IF_Query *		pQuery,
	IF_Db *			pDb,
	FLMUINT32 *		pui32Position)
{
	RCODE			rc;
	FLMUINT		uiPosition;

	rc = pQuery->getPosition( pDb, &uiPosition);
	*pui32Position = (FLMUINT32)uiPosition;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_buildResultSet(
	IF_Query *	pQuery,
	IF_Db *		pDb,
	FLMUINT32	ui32TimeLimit)
{
	return( pQuery->buildResultSet( pDb, (FLMUINT)ui32TimeLimit));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_Query_stopBuildingResultSet(
	IF_Query *	pQuery)
{
	pQuery->stopBuildingResultSet();
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_Query_enableResultSetEncryption(
	IF_Query *	pQuery)
{
	pQuery->enableResultSetEncryption();
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_Query_getCounts(
	IF_Query *		pQuery,
	IF_Db *			pDb,
	FLMUINT32		ui32TimeLimit,
	FLMBOOL			bPartialCountOk,
	FLMUINT32 *		pui32ReadCount,
	FLMUINT32 *		pui32PassedCount,
	FLMUINT32 *		pui32PositionableToCount,
	FLMBOOL *		pbDoneBuildingResultSet)
{
	RCODE			rc;
	FLMUINT		uiReadCount;
	FLMUINT		uiPassedCount;
	FLMUINT		uiPositionableToCount;

	rc = pQuery->getCounts( pDb, (FLMUINT)ui32TimeLimit, bPartialCountOk,
							&uiReadCount, &uiPassedCount, &uiPositionableToCount,
							pbDoneBuildingResultSet);
	*pui32ReadCount = (FLMUINT32)uiReadCount;
	*pui32PassedCount = (FLMUINT32)uiPassedCount;
	*pui32PositionableToCount = (FLMUINT32)uiPositionableToCount;
	return( rc);
}
