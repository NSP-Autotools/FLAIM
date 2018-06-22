//------------------------------------------------------------------------------
// Desc:	Contains specific q-sort code to sort FLAIM's KREF structures.
// Tabs:	3
//
// Copyright (c) 1990-2000, 2002-2007 Novell, Inc. All Rights Reserved.
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

#define	KY_SWAP( pKrefTbl, leftP, rightP)	\
	pTempKref = pKrefTbl [leftP]; \
	pKrefTbl [leftP] = pKrefTbl [rightP]; \
	pKrefTbl [rightP] = pTempKref
	
FSTATIC RCODE ixKeyGetRowId(
	F_INDEX *			pIndex,
	const FLMBYTE *	pucKey,
	const FLMBYTE *	pucKeyEnd,
	FLMUINT				uiKeyComponent,
	FLMUINT64 *			pui64RowId);
	
FSTATIC RCODE ixKeyGetUTF8(
	F_Db *				pDb,
	FLMUINT				uiTableNum,
	ICD *					pIcd,
	F_Row *				pOldRow,
	FLMUINT64			ui64RowId,
	F_DataVector *		pSearchKey,
	FLMUINT				uiKeyComponent,
	F_DynaBuf *			pDynaBuf);
	
FSTATIC RCODE ixKeyGetBinary(
	F_Db *				pDb,
	FLMUINT				uiTableNum,
	ICD *					pIcd,
	F_Row *				pOldRow,
	FLMUINT64			ui64RowId,
	F_DataVector *		pSearchKey,
	FLMUINT				uiKeyComponent,
	F_DynaBuf *			pDynaBuf);
	
FSTATIC RCODE krefQuickSort(
	F_Db *				pDb,
	F_INDEX *			pIndex,
	KREF_ENTRY **		pEntryTbl,
	FLMUINT				uiLowerBounds,
	FLMUINT				uiUpperBounds);

FSTATIC RCODE krefKillDups(
	F_Db *				pDb,
	F_INDEX *			pIndex,
	KREF_ENTRY **		pKrefTbl,
	FLMUINT *			puiKrefTotal);

/***************************************************************************
Desc:	Get the row ID from a key.
*****************************************************************************/
FSTATIC RCODE ixKeyGetRowId(
	F_INDEX *			pIndex,
	const FLMBYTE *	pucKey,
	const FLMBYTE *	pucKeyEnd,
	FLMUINT				uiKeyComponent,
	FLMUINT64 *			pui64RowId)
{
	RCODE				rc = NE_SFLM_OK;
	FLMUINT			uiComponent;
	FLMUINT			uiComponentLen;

	// Skip past all of the remaining key components so we can get to the
	// row ID.  We are currently positioned on the key component
	// specified in uiKeyComponent.  NOTE: uiKeyComponent is zero-based,
	// 0=1st component, 1=2nd component, etc.

	uiComponent = uiKeyComponent;
	while (pucKey < pucKeyEnd && uiComponent < pIndex->uiNumKeyComponents)
	{
		uiComponentLen = getKeyComponentLength( pucKey);
		if (uiComponentLen != KEY_HIGH_VALUE && uiComponentLen != KEY_LOW_VALUE)
		{
			pucKey += (uiComponentLen + 2);
		}
		else
		{
			pucKey += 2;
		}
		uiComponent++;
	}
	
	// See if there is a row ID in the key.  A 0xFF could be present if
	// we have set a "high" row ID.
	
	if (pucKey >= pucKeyEnd || *pucKey == 0xFF)
	{
		*pui64RowId = 0;
		goto Exit;
	}
	
	// At this point, we better have a row ID.

	if (RC_BAD( rc = f_decodeSEN64( &pucKey, pucKeyEnd, pui64RowId)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}

/***************************************************************************
Desc:	Do binary comparison.
*****************************************************************************/
FINLINE FLMINT ixKeyCompareBinary(
	const void *	pvData1,
	FLMUINT			uiLen1,
	const void *	pvData2,
	FLMUINT			uiLen2,
	FLMBOOL			bSortAscending)
{
	FLMINT			iCompare;
	
	if (uiLen1 > uiLen2)
	{
		if ((iCompare = f_memcmp( pvData1, pvData2, uiLen2)) >= 0)
		{
			return( bSortAscending ? 1 : -1);
		}
		else
		{
			return( bSortAscending ? -1 : 1);
		}
	}
	else if (uiLen1 < uiLen2)
	{
		if ((iCompare = f_memcmp( pvData1, pvData2, uiLen1)) <= 0)
		{
			return( bSortAscending ? -1 : 1);
		}
		else
		{
			return( bSortAscending ? 1 : -1);
		}
	}
	else
	{
		if ((iCompare = f_memcmp( pvData1, pvData2, uiLen1)) != 0)
		{
			if (iCompare < 0)
			{
				return( bSortAscending ? -1 : 1);
			}
			else
			{
				return( bSortAscending ? 1 : -1);
			}
		}
	}
	
	return( 0);
}
	
/***************************************************************************
Desc:	Get the UTF8 value for a particular key component.
*****************************************************************************/
FSTATIC RCODE ixKeyGetUTF8(
	F_Db *				pDb,
	FLMUINT				uiTableNum,
	ICD *					pIcd,
	F_Row *				pOldRow,
	FLMUINT64			ui64RowId,
	F_DataVector *		pSearchKey,
	FLMUINT				uiKeyComponent,
	F_DynaBuf *			pDynaBuf)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiBufSize;
	FLMBOOL		bIsNull;
	char *		pszDestBuffer;
	F_Row *		pRow = NULL;
	
	if (ui64RowId)
	{
		if (!pOldRow || pOldRow->getRowId() != ui64RowId)
		{
			if( RC_BAD( rc = gv_SFlmSysData.pRowCacheMgr->retrieveRow( pDb,
										uiTableNum, ui64RowId, &pRow)))
			{
				goto Exit;
			}
		}
		else
		{
			pRow = pOldRow;
		}
		
		if (RC_BAD( rc = pRow->getUTF8( pDb, pIcd->uiColumnNum, NULL,
									0, &bIsNull, NULL, &uiBufSize)))
		{
			goto Exit;
		}
		if (bIsNull || !uiBufSize)
		{
			pDynaBuf->truncateData( 0);
		}
		else
		{
			if( RC_BAD( rc = pDynaBuf->allocSpace( uiBufSize, 
				(void **)&pszDestBuffer)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = pRow->getUTF8( pDb, pIcd->uiColumnNum,
									pszDestBuffer, uiBufSize,
									&bIsNull, NULL, NULL)))
			{
				goto Exit;
			}
		}
	}
	else
	{
		if (RC_BAD( rc = pSearchKey->getUTF8( uiKeyComponent, pDynaBuf)))
		{
			goto Exit;
		}
	}
	
Exit:

	if (pRow && pRow != pOldRow)
	{
		pRow->ReleaseRow();
	}

	return( rc);
}
	
/***************************************************************************
Desc:	Get the binary value for a particular key component.
*****************************************************************************/
FSTATIC RCODE ixKeyGetBinary(
	F_Db *				pDb,
	FLMUINT				uiTableNum,
	ICD *					pIcd,
	F_Row *				pOldRow,
	FLMUINT64			ui64RowId,
	F_DataVector *		pSearchKey,
	FLMUINT				uiKeyComponent,
	F_DynaBuf *			pDynaBuf)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiBufSize;
	FLMBOOL		bIsNull;
	FLMBYTE *	pucDestBuffer;
	F_Row *		pRow = NULL;

	if (ui64RowId)
	{
		if (!pOldRow || pOldRow->getRowId() != ui64RowId)
		{
			if( RC_BAD( rc = gv_SFlmSysData.pRowCacheMgr->retrieveRow( pDb,
										uiTableNum, ui64RowId, &pRow)))
			{
				goto Exit;
			}
		}
		else
		{
			pRow = pOldRow;
		}
	
		pRow->getDataLen( pDb, pIcd->uiColumnNum, &uiBufSize, &bIsNull);
		if (bIsNull || !uiBufSize)
		{
			pDynaBuf->truncateData( 0);
		}
		else
		{
			if( RC_BAD( rc = pDynaBuf->allocSpace( uiBufSize, 
				(void **)&pucDestBuffer)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = pRow->getBinary( pDb, pIcd->uiColumnNum,
									pucDestBuffer, uiBufSize, NULL, &bIsNull)))
			{
				goto Exit;
			}
		}
	}
	else
	{
		if (RC_BAD( rc = pSearchKey->getBinary( uiKeyComponent, pDynaBuf)))
		{
			goto Exit;
		}
	}
	
Exit:

	if (pRow && pRow != pOldRow)
	{
		pRow->ReleaseRow();
	}

	return( rc);
}
	
/***************************************************************************
Desc:	Compares result set entries during the finalization stage to allow
		the result set to be sorted and to remove duplicates.
*****************************************************************************/
RCODE ixKeyCompare(
	F_Db *				pDb,
	F_INDEX *			pIndex,
	FLMBOOL				bCompareRowId,
	F_DataVector *		pSearchKey1,
	F_Row *				pRow1,
	const void *		pvKey1,
   FLMUINT           uiKeyLen1,
	F_DataVector *		pSearchKey2,
	F_Row *				pRow2,
	const void *		pvKey2,
	FLMUINT				uiKeyLen2,
	FLMINT *				piCompare)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiKeyComponent;
	ICD *					pIcd;
	FLMUINT				uiComponentLen1;
	FLMUINT				uiComponentLen2;
	FLMBOOL				bTruncated1;
	FLMBOOL				bTruncated2;
	const FLMBYTE *	pucKey1 = (const FLMBYTE *)pvKey1;
	const FLMBYTE *	pucKey2 = (const FLMBYTE *)pvKey2;
	const FLMBYTE *	pucKeyEnd1 = pucKey1 + uiKeyLen1;
	const FLMBYTE *	pucKeyEnd2 = pucKey2 + uiKeyLen2;
	FLMBOOL				bSortAscending;
	FLMBOOL				bSortMissingHigh;
	FLMUINT64			ui64RowId1;
	FLMUINT64			ui64RowId2;
	
	flmAssert( uiKeyLen1 && uiKeyLen2);
	
	// Loop for each compound piece of key

	uiKeyComponent = 0;
	pIcd = pIndex->pKeyIcds;
	for (;;)
	{
		bSortAscending = (pIcd->uiFlags & ICD_DESCENDING) ? FALSE : TRUE;
		bSortMissingHigh = (pIcd->uiFlags & ICD_MISSING_HIGH) ? TRUE : FALSE;
		uiComponentLen1 = getKeyComponentLength( pucKey1);
		uiComponentLen2 = getKeyComponentLength( pucKey2);
		
		// See if either component is a "high" key
		// NOTE: KEY_HIGH_VALUE always sorts highest, regardless of
		// ascending or descending.  It is never actually stored.  It is
		// only passed in for searching.
		
		if (uiComponentLen1 == KEY_HIGH_VALUE)
		{
			if (uiComponentLen2 == KEY_HIGH_VALUE)
			{
				uiComponentLen1 = uiComponentLen2 = 0;
				goto Test_Exclusive;
			}
			else
			{
				*piCompare = 1;
				goto Exit;
			}
		}
		else if (uiComponentLen2 == KEY_HIGH_VALUE)
		{
			*piCompare = -1;
			goto Exit;
		}
		
		// See if either component is a "low" key
		// NOTE: KEY_LOW_VALUE always sorts lowest, regardless of
		// ascending or descending.  It is never actually stored.  It is
		// only passed in for searching.
		
		if (uiComponentLen1 == KEY_LOW_VALUE)
		{
			if (uiComponentLen2 == KEY_LOW_VALUE)
			{
				uiComponentLen1 = uiComponentLen2 = 0;
				goto Test_Exclusive;
			}
			else
			{
				*piCompare = -1;
				goto Exit;
			}
		}
		else if (uiComponentLen2 == KEY_LOW_VALUE)
		{
			*piCompare = 1;
			goto Exit;
		}
		
		// See if either component is missing.  Need to apply the rules for
		// sorting missing components in that case.
	
		if (!uiComponentLen1)
		{
			if (uiComponentLen2)
			{
				if (bSortMissingHigh)
				{
					*piCompare = bSortAscending ? 1 : -1;
				}
				else
				{
					*piCompare = bSortAscending ? -1 : 1;
				}
				goto Exit;
			}
			else
			{
				goto Test_Exclusive;
			}
		}
		else if (!uiComponentLen2)
		{
			if (bSortMissingHigh)
			{
				*piCompare = bSortAscending ? -1 : 1;
			}
			else
			{
				*piCompare = bSortAscending ? 1 : -1;
			}
			goto Exit;
		}
		else
		{
		
			// Component length must not exceed remaining length of key.
			
			flmAssert( pucKey1 + 2 + uiComponentLen1 <= pucKeyEnd1 &&
						  pucKey2 + 2 + uiComponentLen2 <= pucKeyEnd2);
	
			if ((*piCompare = ixKeyCompareBinary( pucKey1 + 2, uiComponentLen1,
										pucKey2 + 2, uiComponentLen2, bSortAscending)) != 0)
			{
				goto Exit;
			}
			
			// Data is equal, see if one or the other is truncated.
			
			bTruncated1 = isKeyComponentTruncated( pucKey1);
			bTruncated2 = isKeyComponentTruncated( pucKey2);
			
			if (bTruncated1 || bTruncated2)
			{
				if (!bTruncated2)
				{
					*piCompare = bSortAscending ? 1 : -1;
					goto Exit;
				}
				else if (!bTruncated1)
				{
					*piCompare = bSortAscending ? -1 : 1;
					goto Exit;
				}
				
				// Need to get the row that holds the data for the 1st key.
				
				if (isSearchKeyComponent( pucKey1))
				{
					flmAssert( pSearchKey1);
					ui64RowId1 = pSearchKey1->getRowId();
					
					// The search key better have a row ID or the untruncated
					// value.
					
					flmAssert( ui64RowId1 ||
									!pSearchKey1->isRightTruncated( uiKeyComponent));
				}
				else
				{
					if (RC_BAD( rc = ixKeyGetRowId( pIndex, pucKey1, pucKeyEnd1,
												uiKeyComponent, &ui64RowId1)))
					{
						goto Exit;
					}
					flmAssert( ui64RowId1);
				}
				
				// Get the row that holds the data for the 2nd key.
				
				if (isSearchKeyComponent( pucKey2))
				{
					flmAssert( pSearchKey2);
					ui64RowId2 = pSearchKey2->getRowId();
					
					// The search key better have a row ID or the untruncated
					// value.
					
					flmAssert( ui64RowId2 ||
									!pSearchKey2->isRightTruncated( uiKeyComponent));
				}
				else
				{
					if (RC_BAD( rc = ixKeyGetRowId( pIndex, pucKey2, pucKeyEnd2,
												uiKeyComponent, &ui64RowId2)))
					{
						goto Exit;
					}
					flmAssert( ui64RowId2);
				}
	
				// If the row IDs are equal, we can skip fetching the data, because
				// it will be the same.
				
				if (ui64RowId1 != ui64RowId2)
				{
					FLMBYTE		ucDynaBuf1[ 64];
					FLMBYTE		ucDynaBuf2[ 64];
					F_DynaBuf	dynaBuf1( ucDynaBuf1, sizeof( ucDynaBuf1));
					F_DynaBuf	dynaBuf2( ucDynaBuf2, sizeof( ucDynaBuf2));
					F_TABLE *	pTable = pDb->getDict()->getTable( pIndex->uiTableNum);
					F_COLUMN *	pColumn = pDb->getDict()->getColumn( pTable, pIcd->uiColumnNum);
					
					// Better be binary data or text data.
					
					switch (pColumn->eDataTyp)
					{
						case SFLM_STRING_TYPE:
						{
							if (RC_BAD( rc = ixKeyGetUTF8( pDb, pIndex->uiTableNum,
													pIcd, pRow1, ui64RowId1,
													pSearchKey1, uiKeyComponent, &dynaBuf1)))
							{
								goto Exit;
							}
							if (RC_BAD( rc = ixKeyGetUTF8( pDb, pIndex->uiTableNum,
													pIcd, pRow2, ui64RowId2,
													pSearchKey2, uiKeyComponent, &dynaBuf2)))
							{
								goto Exit;
							}
							
							if (RC_BAD( rc = f_compareUTF8Strings(
														dynaBuf1.getBufferPtr(),
														dynaBuf1.getDataLength(),
														FALSE,
														dynaBuf2.getBufferPtr(),
														dynaBuf2.getDataLength(),
														FALSE, pIcd->uiCompareRules,
														pIndex->uiLanguage, piCompare)))
							{
								goto Exit;
							}
							if (*piCompare < 0)
							{
								*piCompare = bSortAscending ? -1 : 1;
								goto Exit;
							}
							else if (*piCompare > 0)
							{
								*piCompare = bSortAscending ? 1 : -1;
								goto Exit;
							}
							break;
						}
				
						case SFLM_BINARY_TYPE:
						{
							if (RC_BAD( rc = ixKeyGetBinary( pDb, pIndex->uiTableNum,
													pIcd, pRow1, ui64RowId1,
													pSearchKey1, uiKeyComponent, &dynaBuf1)))
							{
								goto Exit;
							}
							if (RC_BAD( rc = ixKeyGetBinary( pDb, pIndex->uiTableNum,
													pIcd, pRow2, ui64RowId2,
													pSearchKey2, uiKeyComponent, &dynaBuf2)))
							{
								goto Exit;
							}
							
							if ((*piCompare = ixKeyCompareBinary(
														dynaBuf1.getBufferPtr(),
														dynaBuf1.getDataLength(),
														dynaBuf2.getBufferPtr(),
														dynaBuf2.getDataLength(),
														bSortAscending)) != 0)
							{
								goto Exit;
							}
							break;
						}
	
						default:
							rc = RC_SET_AND_ASSERT( NE_SFLM_DATA_ERROR);
							goto Exit;
					}
				}
			}
		}

Test_Exclusive:
			
		// See if either component is exclusive - everything else is
		// equal up to this point.
		
		if (isKeyComponentLTExclusive( pucKey1))
		{
			if (!isKeyComponentLTExclusive( pucKey2))
			{
				*piCompare = bSortAscending ? -1 : 1;
				goto Exit;
			}
		}
		else if (isKeyComponentGTExclusive( pucKey1))
		{
			if (!isKeyComponentGTExclusive( pucKey2))
			{
				*piCompare = bSortAscending ? 1 : -1;
				goto Exit;
			}
		}
		else if (isKeyComponentLTExclusive( pucKey2))
		{
			*piCompare = bSortAscending ? 1 : -1;
			goto Exit;
		}
		else if (isKeyComponentGTExclusive( pucKey2))
		{
			*piCompare = bSortAscending ? -1 : 1;
			goto Exit;
		}
		
		// Position to the end of this component

		pucKey1 += (2 + uiComponentLen1);
		pucKey2 += (2 + uiComponentLen2);

		// If there are no more ICDs, we are done with the key
		// components.

		if (uiKeyComponent < pIndex->uiNumKeyComponents)
		{
			break;
		}
		pIcd++;
		uiKeyComponent++;

		// See if we are out of key components - this may be a search that
		// passed in only a partial key.
		
		if (pucKey1 >= pucKeyEnd1)
		{
			*piCompare = (pucKey2 >= pucKeyEnd2) ? 0 : -1;
			goto Exit;
		}
		else if (pucKey2 >= pucKeyEnd2)
		{
			*piCompare = 1;
			goto Exit;
		}
	}
	
	// Compare the row ID, if being requested to.  Includes comparing of the
	// last byte, which is the total number of bytes in the row ID.
	
	if (bCompareRowId)
	{
		
		// See if we have a row ID - this may be a search that
		// passed in only a partial key and there is now ROW id on it.
		
		if (pucKey1 >= pucKeyEnd1)
		{
			*piCompare = (pucKey2 >= pucKeyEnd2) ? 0 : -1;
			goto Exit;
		}
		else if (pucKey2 >= pucKeyEnd2)
		{
			*piCompare = 1;
			goto Exit;
		}
		
		// See if either one has an ID buffer of "high"
		
		if (*pucKey1 == 0xFF)
		{
			
			// Key1 has a "high" set of node IDs, see what key2 has.
			
			*piCompare = (*pucKey2 == 0xFF) ? 0 : 1;
			goto Exit;
		}
		else if (*pucKey2 == 0xFF)
		{
			// Key2 has a "high" set of node IDs, key1 does not.
			
			*piCompare = -1;
			goto Exit;
		}
		else
		{
			FLMUINT64	ui64RowId1;
			FLMUINT64	ui64RowId2;
			
			// Get the document ID and compare it, and only it.
			// At this point, both keys should be positioned to
			// get the document ID.
			
			if (RC_BAD( rc = f_decodeSEN64( &pucKey1, pucKeyEnd1, &ui64RowId1)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = f_decodeSEN64( &pucKey2, pucKeyEnd2, &ui64RowId2)))
			{
				goto Exit;
			}
			if (ui64RowId1 == ui64RowId2)
			{
				*piCompare = 0;
			}
			else if (ui64RowId1 < ui64RowId2)
			{
				*piCompare = -1;
			}
			else
			{
				*piCompare = 1;
			}
		}
	}
	else
	{
		*piCompare = 0;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Compare function used to compare index number and key
****************************************************************************/
FINLINE RCODE krefCompareIxAndKey(
	F_Db *			pDb,
	F_INDEX *		pIndex,
	KREF_ENTRY * 	pKrefA,
	KREF_ENTRY *	pKrefB,
	FLMINT *			piCompare)
{
	RCODE		rc = NE_SFLM_OK;
	
	// Compare index numbers

	if ((*piCompare = ((FLMINT) pKrefA->ui16IxNum) - 
		((FLMINT) pKrefB->ui16IxNum)) != 0)
	{
		goto Exit;
	}
	
	if (!pIndex || pIndex->uiIndexNum != (FLMUINT)pKrefA->ui16IxNum)
	{
		pIndex = pDb->getDict()->getIndex( (FLMUINT)pKrefA->ui16IxNum);
	}

	if (RC_BAD( rc = ixKeyCompare( pDb, pIndex, TRUE,
							NULL, pKrefA->pRow,
							&pKrefA [1], (FLMUINT)pKrefA->ui16KeyLen,
							NULL, pKrefB->pRow,
							&pKrefB [1], (FLMUINT)pKrefB->ui16KeyLen, piCompare)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Compare function used to compare key data
****************************************************************************/
FINLINE FLMBOOL krefIsKeyDataEqual(
	KREF_ENTRY * 	pKrefA,
	KREF_ENTRY *	pKrefB)
{
	if( pKrefA->uiDataLen != pKrefB->uiDataLen)
	{
		return( FALSE);
	}
	
	if( pKrefA->uiDataLen)
	{
		if( f_memcmp( (FLMBYTE *)(&pKrefA [1]) +
												pKrefA->ui16KeyLen + 1,
									(FLMBYTE *)(&pKrefB [1]) +
												pKrefB->ui16KeyLen + 1,
									pKrefA->uiDataLen) != 0)
		{
			return( FALSE);
		}
	}
	
	return( TRUE);
}
		
/****************************************************************************
Desc:	Compare function used to compare two keys.
****************************************************************************/
FINLINE RCODE krefSortCompare(
	F_Db *			pDb,
	F_INDEX *		pIndex,
	KREF_ENTRY * 	pKrefA,
	KREF_ENTRY *	pKrefB,
	FLMINT *			piCompare)
{
	RCODE		rc = NE_SFLM_OK;
	
	if (RC_BAD( rc = krefCompareIxAndKey( pDb, pIndex, pKrefA, pKrefB, piCompare)))
	{
		goto Exit;
	}

	if (*piCompare == 0)
	{
		*piCompare = (pKrefA->uiSequence < pKrefB->uiSequence) ? -1 : 1;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Checks if the current database has any UNIQUE indexes that need 
		to checked. Also does duplicate processing for the record.
****************************************************************************/
RCODE F_Db::processDupKeys(
	F_INDEX *	pIndex)
{
	RCODE	rc = NE_SFLM_OK;
	
	//  Sort and remove duplicates

	if (m_uiKrefCount > 1)
	{
		if (RC_BAD( rc = krefQuickSort( this, pIndex, m_pKrefTbl,
									0, m_uiKrefCount - 1)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = krefKillDups( this, pIndex,
										m_pKrefTbl, &m_uiKrefCount)))
		{
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Commit (write out) all keys that have built up in the KREF table.
****************************************************************************/
RCODE F_Db::keysCommit(
	FLMBOOL			bCommittingTrans,
	FLMBOOL			bSortKeys)
{
	RCODE	rc = NE_SFLM_OK;
	
	// If the Kref has not been initialized, there is no
	// work to do.

	if (m_bKrefSetup)
	{
		F_INDEX *		pIndex = NULL;
		FLMUINT			uiTotal = m_uiKrefCount;
		KREF_ENTRY *	pKref;
		KREF_ENTRY **	pKrefTbl = m_pKrefTbl;
		FLMUINT			uiKrefNum;
		FLMUINT			uiLastIxNum;

		// We should not have reached this point if bAbortTrans is TRUE

		if( RC_BAD( m_AbortRc))
		{
			rc = RC_SET_AND_ASSERT( m_AbortRc);
			goto Exit;
		}

		// Sort the KREF table, if it contains more than one key.
		// This will sort all keys from the same index the same.

		if (uiTotal > 1 && bSortKeys)
		{
			processDupKeys( NULL);
			uiTotal = m_uiKrefCount;
		}

		// Loop through the KREF table outputting all keys

		uiLastIxNum = 0;
		for (uiKrefNum = 0; uiKrefNum < uiTotal; uiKrefNum++)
		{
			pKref = pKrefTbl [uiKrefNum];

			// See if the LFILE changed

			flmAssert( pKref->ui16IxNum);

			if (pKref->ui16IxNum != uiLastIxNum)
			{
				uiLastIxNum = pKref->ui16IxNum;
				pIndex = m_pDict->getIndex( uiLastIxNum);
			}

			// Flush the key to the index
			if (m_pKeyColl)
			{
				m_pKeyColl->addKey( this, pIndex, pKref);
			}
			else
			{
				if (RC_BAD(rc = refUpdate( pIndex, pKref, TRUE)))
				{
					if (rc != NE_SFLM_NOT_UNIQUE)
					{
						RC_UNEXPECTED_ASSERT( rc);
					}
					goto Exit;
				}
			}
		}
						
		if (bCommittingTrans)
		{
			krefCntrlFree();
		}
		else
		{
			// Empty the table out so we can add more keys in this trans.

			m_pKrefPool->poolReset( NULL, TRUE);
			m_uiKrefCount = 0;
			m_uiTotalKrefBytes = 0;
		}
	}

Exit:

	if( RC_BAD( rc))
	{
		setMustAbortTrans( rc);
	}

	return( rc);
}

/***************************************************************************
Desc:	Quick sort an array of KREF_ENTRY * values.
****************************************************************************/
FSTATIC RCODE krefQuickSort(
	F_Db *			pDb,
	F_INDEX *		pIndex,
	KREF_ENTRY **	pEntryTbl,
	FLMUINT			uiLowerBounds,
	FLMUINT			uiUpperBounds)
{
	RCODE				rc = NE_SFLM_OK;
	FLMUINT			uiLBPos;
	FLMUINT			uiUBPos;
	FLMUINT			uiMIDPos;
	FLMUINT			uiLeftItems;
	FLMUINT			uiRightItems;
	KREF_ENTRY *	pCurEntry;
	KREF_ENTRY *	pTempKref;
	FLMINT			iCompare;

Iterate_Larger_Half:

	uiUBPos = uiUpperBounds;
	uiLBPos = uiLowerBounds;
	uiMIDPos = (uiUpperBounds + uiLowerBounds + 1) / 2;
	pCurEntry = pEntryTbl[ uiMIDPos ];
	for( ;;)
	{
		for (;;)
		{
			if (uiLBPos != uiMIDPos)
			{
				if (RC_BAD( rc = krefSortCompare( pDb, pIndex,
											pEntryTbl[ uiLBPos], pCurEntry, &iCompare)))
				{
					goto Exit;
				}
				if (iCompare >= 0)
				{
					break;
				}
			}
			if (uiLBPos >= uiUpperBounds)
			{
				break;
			}
			uiLBPos++;
		}

		for (;;)
		{
			if (uiUBPos != uiMIDPos)
			{
				if (RC_BAD( rc = krefSortCompare( pDb, pIndex, pCurEntry,
												pEntryTbl[ uiUBPos], &iCompare)))
				{
					goto Exit;
				}
				if (iCompare >= 0)
				{
					break;
				}
			}
			if (!uiUBPos)
			{
				break;
			}
			uiUBPos--;
		}
		
		if (uiLBPos < uiUBPos)			// Interchange and continue loop.
		{

			// Interchange [uiLBPos] with [uiUBPos].

			KY_SWAP( pEntryTbl, uiLBPos, uiUBPos );
			uiLBPos++;						// Scan from left to right.
			uiUBPos--;						// Scan from right to left.
		}
		else									// Past each other - done
		{
			break;
		}
	}

	// Check for swap( LB, MID ) - cases 3 and 4

	if (uiLBPos < uiMIDPos)
	{

		// Interchange [uiLBPos] with [uiMIDPos]

		KY_SWAP( pEntryTbl, uiMIDPos, uiLBPos );
		uiMIDPos = uiLBPos;
	}
	else if (uiMIDPos < uiUBPos)
	{

		// Interchange [uUBPos] with [uiMIDPos]

		KY_SWAP( pEntryTbl, uiMIDPos, uiUBPos );
		uiMIDPos = uiUBPos;
	}

	// Check the left piece.

	uiLeftItems = (uiLowerBounds + 1 < uiMIDPos )
							? uiMIDPos - uiLowerBounds		// 2 or more
							: 0;
	uiRightItems = (uiMIDPos + 1 < uiUpperBounds )
							? uiUpperBounds - uiMIDPos 		// 2 or more
							: 0;

	if (uiLeftItems < uiRightItems)
	{

		// Recurse on the LEFT side and goto the top on the RIGHT side.

		if (uiLeftItems)
		{
			if (RC_BAD( rc = krefQuickSort( pDb, pIndex, pEntryTbl,
										uiLowerBounds, uiMIDPos - 1)))
			{
				goto Exit;
			}
		}
		uiLowerBounds = uiMIDPos + 1;
		goto Iterate_Larger_Half;
	}
	else if (uiLeftItems)	// Compute a truth table to figure out this check.
	{

		// Recurse on the RIGHT side and goto the top for the LEFT side.

		if (uiRightItems)
		{
			if (RC_BAD( rc = krefQuickSort( pDb, pIndex, pEntryTbl,
										uiMIDPos + 1, uiUpperBounds)))
			{
				goto Exit;
			}
		}
		uiUpperBounds = uiMIDPos - 1;
		goto Iterate_Larger_Half;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:		Kill all duplicate keys in the KREF table.
****************************************************************************/
FSTATIC RCODE krefKillDups(
	F_Db *			pDb,
	F_INDEX *		pIndex,
	KREF_ENTRY **	pKrefTbl,
	FLMUINT *		puiKrefTotal)
{
	RCODE				rc = NE_SFLM_OK;
	FLMUINT			uiCurKref = 0;
	FLMUINT			uiLastKref = *puiKrefTotal;
	FLMUINT			uiFirstForKey;
	FLMUINT			uiLastForKey;
	FLMUINT			uiNewPosOffset = 0;
	FLMINT			iCompare;

	while( uiCurKref < uiLastKref)
	{
		uiFirstForKey = uiLastForKey = uiCurKref;
		uiCurKref = uiFirstForKey + 1;

		while( uiCurKref < uiLastKref)
		{
			if (RC_BAD( rc = krefCompareIxAndKey( pDb, pIndex, pKrefTbl[ uiFirstForKey],
												pKrefTbl[ uiCurKref], &iCompare)))
			{
				goto Exit;
			}
			
			if (iCompare)
			{
				break;
			}
			
			uiLastForKey = uiCurKref++;
		}
		
		if( uiFirstForKey == uiLastForKey)
		{
			pKrefTbl[ uiNewPosOffset++] = pKrefTbl[ uiFirstForKey];
			continue;
		}
		
		if( pKrefTbl[ uiFirstForKey]->bDelete)
		{
			if( pKrefTbl[ uiLastForKey]->bDelete)
			{
				pKrefTbl[ uiNewPosOffset++] = pKrefTbl[ uiFirstForKey];
			}
			else
			{
TestCancel:
				// See if the operations cancel each other.  If they don't, we
				// need to keep both operations
				
				if( !krefIsKeyDataEqual( pKrefTbl[ uiFirstForKey], pKrefTbl[ uiLastForKey]))
				{
					pKrefTbl[ uiNewPosOffset++] = pKrefTbl[ uiFirstForKey];
					pKrefTbl[ uiNewPosOffset++] = pKrefTbl[ uiLastForKey];
				}
			}
		}
		else
		{
			if( pKrefTbl[ uiLastForKey]->bDelete)
			{
				goto TestCancel;
			}
			else
			{
				pKrefTbl[ uiNewPosOffset++] = pKrefTbl[ uiLastForKey];
			}
		}
	}
	
	*puiKrefTotal = uiNewPosOffset;
	
Exit:

	return( rc);
}

