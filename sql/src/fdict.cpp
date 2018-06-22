//------------------------------------------------------------------------------
// Desc:	Routines to access anything in the dictionary
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

#include "flaimsys.h"

// Local function prototypes

FSTATIC FLMBOOL validBooleanValue(
	const char *	pszValue,
	FLMBOOL *		pbTrue);
	
FSTATIC FLMBOOL validDataType(
	const char *	pszDataType,
	eDataType *		peDataTyp);
	
FSTATIC FLMBOOL validEncAlgorithm(
	const char *		pszEncAlgorithm,
	eEncAlgorithm *	peEncAlg);
	
FSTATIC FLMBOOL validEncKeySize(
	eEncAlgorithm	eEncAlg,
	FLMUINT			uiKeySize);
	
FSTATIC void sortNameTbl(
	NAME_INFO *	pNameInfoTbl,
	FLMUINT		uiLowerBounds,
	FLMUINT		uiUpperBounds,
	FLMBOOL *	pbDuplicateNames);

FSTATIC FLMBOOL validIndexState(
	const char *	pszIndexState,
	FLMUINT *		puiFlags);
	
FSTATIC FLMBOOL validIndexOnValue(
	const char *	pszIndexOn,
	FLMUINT *		puiFlags);
	
FSTATIC FLMBOOL validCompareRulesValue(
	char *			pszCompareRules,
	FLMUINT *		puiCompareRules);
	
/***************************************************************************
Desc:	Constructor
***************************************************************************/
F_Dict::F_Dict()
{
	m_pNext = NULL;
	m_pPrev = NULL;
	m_pDatabase = NULL;
	m_uiDictSeq = 0;
	m_dictPool.poolInit( 1024);

	m_pTableTbl = NULL;
	m_uiTableTblSize = 0;
	m_uiHighestTableNum = 0;
	m_pTableNames = NULL;
	
	m_pIndexTbl = NULL;
	m_uiIndexTblSize = 0;
	m_uiHighestIndexNum = 0;
	m_pIndexNames = NULL;
	
	m_pEncDefTbl = NULL;
	m_uiEncDefTblSize = 0;
	m_uiHighestEncDefNum = 0;
	m_pEncDefNames = NULL;
	
	// Whenever an F_Dict is allocated, it is always immediately
	// used by an F_Db.

	m_uiUseCount = 1;
}

/***************************************************************************
Desc:	Destructor
***************************************************************************/
F_Dict::~F_Dict()
{
	resetDict();
}

/***************************************************************************
Desc:	Clear the dictionary object so it can be reused.  NOTE: This function
		also needs to do all the freeing up that the destructor would do
		because it is called by the destructor.
***************************************************************************/
void F_Dict::resetDict( void)
{
	FLMUINT		uiLoop;

	for (uiLoop = 0; uiLoop < m_uiHighestTableNum; uiLoop++)
	{
		if (m_pTableTbl [uiLoop].uiTableNum &&
			 m_pTableTbl [uiLoop].pColumnNames)
		{
			m_pTableTbl [uiLoop].pColumnNames->Release();
		}
	}
	
	f_free( &m_pTableTbl);
	m_uiTableTblSize = 0;
	m_uiHighestTableNum = 0;
	if (m_pTableNames)
	{
		m_pTableNames->Release();
		m_pTableNames = NULL;
	}
	
	f_free( &m_pIndexTbl);
	m_uiIndexTblSize = 0;
	m_uiHighestIndexNum = 0;
	if (m_pIndexNames)
	{
		m_pIndexNames->Release();
		m_pIndexNames = NULL;
	}
	
	for (uiLoop = 0; uiLoop < m_uiHighestEncDefNum; uiLoop++)
	{
		if (m_pEncDefTbl [uiLoop].uiEncDefNum &&
			 m_pEncDefTbl [uiLoop].pCcs)
		{
			m_pEncDefTbl[ uiLoop].pCcs->Release();
		}
	}
	f_free( &m_pEncDefTbl);
	m_uiEncDefTblSize = 0;
	m_uiHighestEncDefNum = 0;
	if (m_pEncDefNames)
	{
		m_pEncDefNames->Release();
		m_pEncDefNames = NULL;
	}

	m_dictPool.poolFree();
	m_dictPool.poolInit( 1024);
}

/***************************************************************************
Desc:	Get the table given a table name.
***************************************************************************/
RCODE F_Dict::getTable(
	const char *	pszTableName,
	F_TABLE **		ppTable,
	FLMBOOL			bOfflineOk)
{
	RCODE			rc = NE_SFLM_OK;
	F_TABLE *	pTable;

	if ((pTable = findTable( pszTableName)) == NULL)
	{
		rc = RC_SET( NE_SFLM_BAD_TABLE);
		goto Exit;
	}
	
	// If the table is encrypted, and we are in limited mode, then we must
	// treat is as an offline table.

	if (pTable->lfInfo.uiEncDefNum && m_pDatabase &&
		 m_pDatabase->inLimitedMode() && !bOfflineOk)
	{
		rc = RC_SET( NE_SFLM_TABLE_OFFLINE);
	}

Exit:

	if (ppTable)
	{
		*ppTable = pTable;
	}

	return( rc);
}

/***************************************************************************
Desc:	Get the index given an index name.
***************************************************************************/
RCODE F_Dict::getIndex(
	const char *	pszIndexName,
	F_INDEX **		ppIndex,
	FLMBOOL			bOfflineOk)
{
	RCODE			rc = NE_SFLM_OK;
	F_INDEX *	pIndex;
	
	if ((pIndex = findIndex( pszIndexName)) == NULL)
	{
		rc = RC_SET( NE_SFLM_BAD_IX);
		goto Exit;
	}

	// If the index is suspended the IXD_OFFLINE flag
	// will be set, so it is sufficient to just test
	// the IXD_OFFLINE for both suspended and offline
	// conditions.

	if ((pIndex->uiFlags & IXD_OFFLINE) && !bOfflineOk)
	{
		rc = RC_SET( NE_SFLM_INDEX_OFFLINE);
		goto Exit;
	}

	// An encrypted index is offline if we are in limited mode.

	if (pIndex->lfInfo.uiEncDefNum && m_pDatabase &&
		 m_pDatabase->inLimitedMode() && !bOfflineOk)
	{
		rc = RC_SET( NE_SFLM_INDEX_OFFLINE);
		goto Exit;
	}

Exit:

	if (ppIndex)
	{
		*ppIndex = pIndex;
	}

	return( rc);
}

/****************************************************************************
Desc:	Link the dictionary to an F_Database object
		NOTE: This routine assumes the global mutex is locked.
****************************************************************************/
void F_Dict::linkToDatabase(
	F_Database *	pDatabase)
{
	if ((m_pNext = pDatabase->m_pDictList) != NULL)
	{
		m_uiDictSeq = m_pNext->m_uiDictSeq + 1;
		m_pNext->m_pPrev = this;
	}
	else
	{
		m_uiDictSeq = 1;
	}
	pDatabase->m_pDictList = this;
	m_pDatabase = pDatabase;
}

/****************************************************************************
Desc:	Unlink the dictionary from its F_Database object
		NOTE: This routine assumes the database mutex is locked.
****************************************************************************/
void F_Dict::unlinkFromDatabase( void)
{

	// Unlink the local dictionary from its database - if it is connected
	// to one.

	if (m_pDatabase)
	{
		if (m_pPrev)
		{
			m_pPrev->m_pNext = m_pNext;
		}
		else
		{
			m_pDatabase->m_pDictList = m_pNext;
		}
		if (m_pNext)
		{
			m_pNext->m_pPrev = m_pPrev;
		}
	}

	// Free the local dictionary and its associated tables.

	Release();
}

/****************************************************************************
Desc:	Copies an encryption def (F_ENCDEF)
****************************************************************************/
RCODE F_Dict::copyEncDef(
	F_ENCDEF *	pDestEncDef,
	F_ENCDEF *	pSrcEncDef)
{
	RCODE	rc = NE_SFLM_OK;

	f_memcpy( pDestEncDef, pSrcEncDef, sizeof( F_ENCDEF));
	if (RC_BAD( rc = m_pEncDefNames->copyName( pSrcEncDef->pszEncDefName,
								pDestEncDef->uiEncDefNum, &pDestEncDef->pszEncDefName,
								&m_dictPool)))
	{
		goto Exit;
	}
	pDestEncDef->pCcs->AddRef();

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Copies a table column.  NOTE: This routine assumes that
		the encryption definitions have already been copied.
****************************************************************************/
RCODE F_Dict::copyColumn(
	F_NameTable *	pDestColumnNameTable,
	F_COLUMN *		pDestColumn,
	F_COLUMN *		pSrcColumn)
{
	RCODE	rc = NE_SFLM_OK;

	f_memcpy( pDestColumn, pSrcColumn, sizeof( F_COLUMN));
	if (RC_BAD( rc = pDestColumnNameTable->copyName( pSrcColumn->pszColumnName,
								pDestColumn->uiColumnNum, &pDestColumn->pszColumnName,
								&m_dictPool)))
	{
		goto Exit;
	}
	
	// ICDs will be fixed up and set when the indexes are copied later on.
	
	pDestColumn->pFirstIcd = NULL;
	pDestColumn->pFirstDataIcd = NULL;
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Copies a table and all of its columns.  NOTE: This routine assumes that
		the encryption definitions have already been copied.
****************************************************************************/
RCODE F_Dict::copyTable(
	F_TABLE *		pDestTable,
	F_TABLE *		pSrcTable)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiLoop;
	F_COLUMN *	pSrcColumn;
	F_COLUMN *	pDestColumn;
	
	f_memcpy( pDestTable, pSrcTable, sizeof( F_TABLE));
	
	// Add the table name to the table name table.

	if (RC_BAD( rc = m_pTableNames->copyName( pSrcTable->pszTableName,
								pDestTable->uiTableNum, &pDestTable->pszTableName,
								&m_dictPool)))
	{
		goto Exit;
	}
	
	// Allocate a name table for the column names.
	
	if ((pDestTable->pColumnNames = f_new F_NameTable) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}
	if (RC_BAD( rc = f_alloc( sizeof( NAME_INFO) * pSrcTable->pColumnNames->m_uiNumNames,
									&pDestTable->pColumnNames->m_pNames)))
	{
		goto Exit;
	}
	pDestTable->pColumnNames->m_uiTblSize = pSrcTable->pColumnNames->m_uiNumNames;
	
	// Copy the columns. - There is always at least one column, so no
	// need to check for a column count.
	
	if (RC_BAD( rc = m_dictPool.poolAlloc( sizeof( F_COLUMN) * pSrcTable->uiNumColumns,
								(void **)&pDestTable->pColumns)))
	{
		goto Exit;
	}
	
	for (uiLoop = 0, pSrcColumn = pSrcTable->pColumns, pDestColumn = pDestTable->pColumns;
		  uiLoop < pSrcTable->uiNumColumns;
		  uiLoop++, pSrcColumn++, pDestColumn++)
	{
		if (RC_BAD( rc = copyColumn( pDestTable->pColumnNames, pDestColumn, pSrcColumn)))
		{
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Copies an index and all of its ICDs.  NOTE: This routine assumes that
		the tables and columns have already been copied.
****************************************************************************/
RCODE F_Dict::copyIndex(
	F_INDEX *	pDestIndex,
	F_INDEX *	pSrcIndex)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiTotalIcds;
	ICD *			pIcd;
	F_COLUMN *	pColumn;
	F_TABLE *	pDestTable;
	FLMUINT		uiLoop;
	
	f_memcpy( pDestIndex, pSrcIndex, sizeof( F_INDEX));
	
	// Tables should already have been set up, and pSrcIndex->uiTableNum
	// better be referencing a valid table!
	
	pDestTable = getTable( pDestIndex->uiTableNum);
	flmAssert( pDestTable);
	
	// Add the index name to the index name table.

	if (RC_BAD( rc = m_pIndexNames->copyName( pSrcIndex->pszIndexName,
								pDestIndex->uiIndexNum, &pDestIndex->pszIndexName,
								&m_dictPool)))
	{
		goto Exit;
	}
	
	// Allocate space for the ICDs.
	
	uiTotalIcds = pDestIndex->uiNumKeyComponents + pDestIndex->uiNumDataComponents;
	if (RC_BAD( rc = m_dictPool.poolAlloc( sizeof( ICD) * uiTotalIcds,
					(void **)&pDestIndex->pKeyIcds)))
	{
		goto Exit;
	}
	f_memcpy( pDestIndex->pKeyIcds, pSrcIndex->pKeyIcds,
				sizeof( ICD) * pDestIndex->uiNumKeyComponents);
	if (pDestIndex->pDataIcds)
	{
		pDestIndex->pDataIcds = pDestIndex->pKeyIcds + pDestIndex->uiNumKeyComponents;
		f_memcpy( pDestIndex->pDataIcds, pSrcIndex->pDataIcds,
				sizeof( ICD) * pDestIndex->uiNumDataComponents);
	}
	
	// Fixup the index and column pointers in the destination ICD.
	
	for (uiLoop = 0, pIcd = pDestIndex->pKeyIcds;
		  uiLoop < pDestIndex->uiNumKeyComponents;
		  uiLoop++, pIcd++)
	{
		
		// Columns should already be set up and column number better be
		// referencing a valid column!
		
		pColumn = getColumn( pDestTable, pIcd->uiColumnNum);
		flmAssert( pColumn);
		
		pIcd->pNextInChain = pColumn->pFirstIcd;
		pColumn->pFirstIcd = pIcd->pNextInChain;
	}
	for (uiLoop = 0, pIcd = pDestIndex->pDataIcds;
		  uiLoop < pDestIndex->uiNumDataComponents;
		  uiLoop++, pIcd++)
	{
		
		// Columns should already be set up and column number better be
		// referencing a valid column!
		
		pColumn = getColumn( pDestTable, pIcd->uiColumnNum);
		flmAssert( pColumn);
		pIcd->pNextInDataChain = pColumn->pFirstDataIcd;
		pColumn->pFirstDataIcd = pIcd->pNextInDataChain;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Clones a dictionary from another one.
****************************************************************************/
RCODE F_Dict::cloneDict(
	F_Dict *	pSrcDict)
{
	RCODE			rc = NE_SFLM_OK;
	F_ENCDEF *	pSrcEncDef;
	F_ENCDEF *	pDestEncDef;
	F_TABLE *	pSrcTable;
	F_TABLE *	pDestTable;
	F_INDEX *	pSrcIndex;
	F_INDEX *	pDestIndex;
	FLMUINT		uiLoop;

	resetDict();
	
	// Set up all of the name tables to be large enough to hold the
	// names in the source dictionary.
	
	if (pSrcDict->m_pTableNames)
	{
		if ((m_pTableNames = f_new F_NameTable) == NULL)
		{
			rc = RC_SET( NE_SFLM_MEM);
			goto Exit;
		}
		if (RC_BAD( rc = f_alloc( sizeof( NAME_INFO) * pSrcDict->m_pTableNames->m_uiNumNames,
										&m_pTableNames->m_pNames)))
		{
			goto Exit;
		}
		m_pTableNames->m_uiTblSize = pSrcDict->m_pTableNames->m_uiNumNames;
	}
	
	if (pSrcDict->m_pIndexNames)
	{
		if ((m_pIndexNames = f_new F_NameTable) == NULL)
		{
			rc = RC_SET( NE_SFLM_MEM);
			goto Exit;
		}
		if (RC_BAD( rc = f_alloc( sizeof( NAME_INFO) * pSrcDict->m_pIndexNames->m_uiNumNames,
										&m_pIndexNames->m_pNames)))
		{
			goto Exit;
		}
		m_pIndexNames->m_uiTblSize = pSrcDict->m_pIndexNames->m_uiNumNames;
	}
	
	if (pSrcDict->m_pEncDefNames)
	{
		if ((m_pEncDefNames = f_new F_NameTable) == NULL)
		{
			rc = RC_SET( NE_SFLM_MEM);
			goto Exit;
		}
		if (RC_BAD( rc = f_alloc( sizeof( NAME_INFO) * pSrcDict->m_pEncDefNames->m_uiNumNames,
										&m_pEncDefNames->m_pNames)))
		{
			goto Exit;
		}
		m_pEncDefNames->m_uiTblSize = pSrcDict->m_pEncDefNames->m_uiNumNames;
	}
	
	// Copy encryption definitions first.

	if (pSrcDict->m_uiHighestEncDefNum)
	{
		if (RC_BAD( rc = f_alloc( sizeof( F_ENCDEF) * pSrcDict->m_uiHighestEncDefNum,
									&m_pEncDefTbl)))
		{
			goto Exit;
		}
		m_uiEncDefTblSize = m_uiHighestEncDefNum = pSrcDict->m_uiHighestEncDefNum;
	
		for (uiLoop = 0, pSrcEncDef = pSrcDict->m_pEncDefTbl, pDestEncDef = m_pEncDefTbl;
			  uiLoop < m_uiHighestEncDefNum;
			  uiLoop++, pSrcEncDef++, pDestEncDef++)
		{
			if (RC_BAD( rc = copyEncDef( pDestEncDef, pSrcEncDef)))
			{
				goto Exit;
			}
		}
	}
	else
	{
		m_uiEncDefTblSize = 0;
		m_uiHighestEncDefNum = 0;
	}
	
	// Copy tables and columns. - There is always at least one table, so no
	// need to check for a table count.
	
	if (RC_BAD( rc = f_alloc( sizeof( F_TABLE) * pSrcDict->m_uiHighestTableNum,
								&m_pTableTbl)))
	{
		goto Exit;
	}
	m_uiTableTblSize = m_uiHighestTableNum = pSrcDict->m_uiHighestTableNum;
	
	for (uiLoop = 0, pSrcTable = pSrcDict->m_pTableTbl, pDestTable = m_pTableTbl;
		  uiLoop < m_uiHighestTableNum;
		  uiLoop++, pSrcTable++, pDestTable++)
	{
		if (RC_BAD( rc = copyTable( pDestTable, pSrcTable)))
		{
			goto Exit;
		}
	}
	
	// Copy indexes. - there is always at least one index, because we have
	// internal indexes - so no need to check for an index count.
	
	if (RC_BAD( rc = f_alloc( sizeof( F_ENCDEF) * pSrcDict->m_uiHighestIndexNum,
								&m_pIndexTbl)))
	{
		goto Exit;
	}
	m_uiIndexTblSize = m_uiHighestIndexNum = pSrcDict->m_uiHighestIndexNum;
	
	for (uiLoop = 0, pSrcIndex = pSrcDict->m_pIndexTbl, pDestIndex = m_pIndexTbl;
		  uiLoop < m_uiHighestIndexNum;
		  uiLoop++, pSrcIndex++, pDestIndex++)
	{
		if (RC_BAD( rc = copyIndex( pDestIndex, pSrcIndex)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	See if the read-only value is valid.
****************************************************************************/
FSTATIC FLMBOOL validBooleanValue(
	const char *	pszValue,
	FLMBOOL *		pbTrue)
{
	if (f_stricmp( pszValue, "true") == 0 ||
		 f_stricmp( pszValue, "yes") == 0 ||
		 f_stricmp( pszValue, "1") == 0 ||
		 f_stricmp( pszValue, "enabled") == 0 ||
		 f_stricmp( pszValue, "on") == 0)
	{
		*pbTrue = TRUE;
		return( TRUE);
	}
	else if (f_stricmp( pszValue, "false") == 0 ||
				f_stricmp( pszValue, "no") == 0 ||
				f_stricmp( pszValue, "0") == 0 ||
				f_stricmp( pszValue, "disabled") == 0 ||
				f_stricmp( pszValue, "off") == 0)
	{
		*pbTrue = FALSE;
		return( TRUE);
	}
	return( FALSE);
}

/***************************************************************************
Desc:	Maps a string to an element or attribute data type.
***************************************************************************/
FSTATIC FLMBOOL validDataType(
	const char *	pszDataType,
	eDataType *		peDataTyp)
{
	if (f_stricmp( pszDataType, SFLM_STRING_OPTION_STR) == 0)
	{
		*peDataTyp = SFLM_STRING_TYPE;
		return( TRUE);
	}
	else if (f_stricmp( pszDataType, SFLM_INTEGER_OPTION_STR) == 0)
	{
		*peDataTyp = SFLM_NUMBER_TYPE;
		return( TRUE);
	}
	else if (f_stricmp( pszDataType, SFLM_BINARY_OPTION_STR) == 0)
	{
		*peDataTyp = SFLM_BINARY_TYPE;
		return( TRUE);
	}
	return( FALSE);
}

/***************************************************************************
Desc:	Determine if an encryption algorithm is valid.
***************************************************************************/
FSTATIC FLMBOOL validEncAlgorithm(
	const char *		pszEncAlgorithm,
	eEncAlgorithm *	peEncAlg)
{
	if (f_stricmp( pszEncAlgorithm, SFLM_ENC_AES_OPTION_STR) == 0)
	{
		*peEncAlg = SFLM_AES_ENCRYPTION;
		return( TRUE);
	}
	else if (f_stricmp( pszEncAlgorithm, SFLM_ENC_DES3_OPTION_STR))
	{
		*peEncAlg = SFLM_DES3_ENCRYPTION;
		return( TRUE);
	}
	return( FALSE);
}

/***************************************************************************
Desc:	Determine if the key size for an encryption algorithm is valid.
***************************************************************************/
FSTATIC FLMBOOL validEncKeySize(
	eEncAlgorithm	eEncAlg,
	FLMUINT			uiKeySize)
{
	switch (eEncAlg)
	{
		case SFLM_AES_ENCRYPTION:
			if (uiKeySize == SFLM_AES128_KEY_SIZE ||
				 uiKeySize == SFLM_AES192_KEY_SIZE ||
				 uiKeySize == SFLM_AES256_KEY_SIZE)
			{
				return( TRUE);
			}
			break;
		case SFLM_DES3_ENCRYPTION:
			if (uiKeySize == SFLM_DES3_168_KEY_SIZE)
			{
				return( TRUE);
			}
		default:
			// Should never hit this case.
			flmAssert( 0);
			break;
	}
	return( FALSE);
}

/****************************************************************************
Desc:	Add an encryption definition to the dictionary.
****************************************************************************/
RCODE F_Dict::addEncDef(
	FLMUINT			uiEncDefNum,
	FLMUINT64		ui64DefRowId,
	const char *	pszEncDefName,
	eEncAlgorithm	eEncAlg,
	FLMUINT			uiEncKeySize,
	FLMBYTE *		pucEncKey,
	FLMUINT)		// uiEncKeyLen)
{
	RCODE			rc = NE_SFLM_OK;
	F_ENCDEF *	pEncDef;
	F_CCS *		pCcs = NULL;
	
	if (!uiEncDefNum)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_ENCDEF_NUM);
		goto Exit;
	}
	
	// Add a table entry to the array of tables.
	// See if there is room in the table for another table.
	
	if (uiEncDefNum >= m_uiEncDefTblSize)
	{
		FLMUINT	uiNewTblSize = uiEncDefNum + 10;
		
		if (RC_BAD( rc = f_realloc( sizeof( F_ENCDEF) * uiNewTblSize,
									&m_pEncDefTbl)))
		{
			goto Exit;
		}
		
		// Memset the new part of the table to all zeroes - so we can tell
		// which slots are empty.
		
		f_memset( &m_pEncDefTbl [m_uiEncDefTblSize], 0,
					sizeof( F_ENCDEF) * (uiNewTblSize - m_uiEncDefTblSize));
		m_uiEncDefTblSize = uiNewTblSize;
	}
	pEncDef = &m_pEncDefTbl [uiEncDefNum - 1];
	
	// Make sure we have not already defined this slot.
	
	if (pEncDef->uiEncDefNum)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_DUPLICATE_ENCDEF_NUM);
		goto Exit;
	}
	
	// Add the table name to the table name table.

	if (RC_BAD( rc = m_pEncDefNames->copyName( pszEncDefName,
								uiEncDefNum, &pEncDef->pszEncDefName, &m_dictPool)))
	{
		goto Exit;
	}
	pEncDef->eEncAlg = eEncAlg;
	pEncDef->uiEncKeySize = uiEncKeySize;
	
	// Create the CCS object
	
	if ((pCcs = f_new( F_CCS)) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = pCcs->init( FALSE, eEncAlg)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pCcs->setKeyFromStore(
		pucEncKey, NULL, m_pDatabase->m_pWrappingKey)))
	{
		goto Exit;
	}
		
	pEncDef->pCcs = pCcs;
	pEncDef->pCcs->AddRef();
	
	pEncDef->uiEncDefNum = uiEncDefNum;
	pEncDef->ui64DefRowId = ui64DefRowId;
	if (uiEncDefNum > m_uiHighestEncDefNum)
	{
		m_uiHighestEncDefNum = uiEncDefNum;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Add a table to the dictionary, reserving room for its columns.
****************************************************************************/
RCODE F_Dict::addTable(
	FLMUINT			uiTableNum,
	FLMUINT64		ui64DefRowId,
	const char *	pszTableName,
	FLMBOOL			bSystemTable,
	FLMUINT			uiNumColumns,
	FLMUINT			uiEncDefNum)
{
	RCODE			rc = NE_SFLM_OK;
	F_TABLE *	pTable;
	
	if (!uiTableNum)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_TABLE_NUM);
		goto Exit;
	}
	
	// Verify that the encryption definition is valid.

	if (uiEncDefNum && getEncDef( uiEncDefNum) == NULL)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_ENCDEF_NUM);
		goto Exit;
	}
	
	// Add a table entry to the array of tables.
	// See if there is room in the table for another table.
	
	if (uiTableNum >= m_uiTableTblSize)
	{
		FLMUINT	uiNewTblSize = uiTableNum + 20;
		
		if (RC_BAD( rc = f_realloc( sizeof( F_TABLE) * uiNewTblSize,
									&m_pTableTbl)))
		{
			goto Exit;
		}
		
		// Memset the new part of the table to all zeroes - so we can tell
		// which slots are empty.
		
		f_memset( &m_pTableTbl [m_uiTableTblSize], 0,
					sizeof( F_TABLE) * (uiNewTblSize - m_uiTableTblSize));
		m_uiTableTblSize = uiNewTblSize;
	}
	pTable = &m_pTableTbl [uiTableNum - 1];
	
	// Make sure we have not already defined this slot.
	
	if (pTable->uiTableNum)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_DUPLICATE_TABLE_NUM);
		goto Exit;
	}
	
	// Add the table name to the table name table.

	if (RC_BAD( rc = m_pTableNames->copyName( pszTableName,
								uiTableNum, &pTable->pszTableName, &m_dictPool)))
	{
		goto Exit;
	}
	
	// Allocate a name table for the column names.
	
	if ((pTable->pColumnNames = f_new F_NameTable) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}
	if (RC_BAD( rc = f_calloc( sizeof( NAME_INFO) * uiNumColumns,
									&pTable->pColumnNames->m_pNames)))
	{
		goto Exit;
	}
	pTable->pColumnNames->m_uiTblSize = uiNumColumns;
	
	// Make space for the columns.

	if (RC_BAD( rc = m_dictPool.poolAlloc( sizeof( F_COLUMN) * uiNumColumns,
								(void **)&pTable->pColumns)))
	{
		goto Exit;
	}
	pTable->uiNumColumns = uiNumColumns;
	pTable->bSystemTable = bSystemTable;
	f_memset( &pTable->lfInfo, 0, sizeof( LFILE));
	pTable->lfInfo.uiLfNum = uiTableNum;
	pTable->lfInfo.uiEncDefNum = uiEncDefNum;
	pTable->uiTableNum = uiTableNum;
	pTable->ui64DefRowId = ui64DefRowId;
	pTable->uiFirstIndexNum = 0;
	
	if (uiTableNum > m_uiHighestTableNum)
	{
		m_uiHighestTableNum = uiTableNum;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Add a column to a table in the dictionary.
****************************************************************************/
RCODE F_Dict::addColumn(
	FLMUINT			uiTableNum,
	FLMUINT64		ui64DefRowId,
	FLMUINT			uiColumnNum,
	const char *	pszColumnName,
	FLMUINT			uiFlags,
	eDataType		eDataTyp,
	FLMUINT			uiMaxLen,
	FLMUINT			uiEncDefNum)
{
	RCODE			rc = NE_SFLM_OK;
	F_TABLE *	pTable;
	F_COLUMN *	pColumn;
	
	// Verify the table number
	
	if ((pTable = getTable( uiTableNum)) == NULL)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_TABLE_NUM);
		goto Exit;
	}
	
	// Verify that the encryption definition is valid.
	
	if (uiEncDefNum && getEncDef( uiEncDefNum) == NULL)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_ENCDEF_NUM);
		goto Exit;
	}
	
	// Make sure the column number is valid.
	
	if (!uiColumnNum || uiColumnNum > pTable->uiNumColumns)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_COLUMN_NUM);
		goto Exit;
	}
	pColumn = &pTable->pColumns [uiColumnNum - 1];
	
	// Column number should not yet be set up.
	
	if (pColumn->uiColumnNum)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_DUPLICATE_COLUMN_NUM);
		goto Exit;
	}
	
	// Add the column name to the table's column name table.

	if (RC_BAD( rc = pTable->pColumnNames->copyName( pszColumnName,
								uiColumnNum, &pColumn->pszColumnName, &m_dictPool)))
	{
		goto Exit;
	}
	pColumn->uiColumnNum = uiColumnNum;
	pColumn->ui64DefRowId = ui64DefRowId;
	pColumn->uiFlags = uiFlags;
	pColumn->eDataTyp = eDataTyp;
	pColumn->uiMaxLen = uiMaxLen;
	pColumn->uiEncDefNum = uiEncDefNum;
	pColumn->pFirstIcd = NULL;
	pColumn->pFirstDataIcd = NULL;
	if (!(uiFlags & COL_NULL_ALLOWED))
	{
		pTable->uiNumReqColumns++;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Add an index to the dictionary, reserving room for its key and data
		components.
****************************************************************************/
RCODE F_Dict::addIndex(
	FLMUINT			uiIndexNum,
	FLMUINT64		ui64DefRowId,
	const char *	pszIndexName,
	FLMUINT			uiTableNum,
	FLMUINT			uiEncDefNum,
	FLMUINT			uiFlags,
	FLMUINT			uiNumKeyComponents,
	FLMUINT			uiNumDataComponents,
	FLMUINT			uiLanguage,
	FLMUINT64		ui64LastRowIndexed)
{
	RCODE			rc = NE_SFLM_OK;
	F_INDEX *	pIndex;
	F_TABLE *	pTable;
	
	if (!uiIndexNum)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_INDEX_NUM);
		goto Exit;
	}
	
	// Verify the table number
	
	if ((pTable = getTable( uiTableNum)) == NULL)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_TABLE_NUM);
		goto Exit;
	}
	
	// Verify that the encryption definition is valid.
	
	if (uiEncDefNum && getEncDef( uiEncDefNum) == NULL)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_ENCDEF_NUM);
		goto Exit;
	}
	
	// Add an index entry to the array of indexes.
	// See if there is room in the table for another index.
	
	if (uiIndexNum >= m_uiIndexTblSize)
	{
		FLMUINT	uiNewTblSize = uiIndexNum + 20;
		
		if (RC_BAD( rc = f_realloc( sizeof( F_INDEX) * uiNewTblSize,
									&m_pIndexTbl)))
		{
			goto Exit;
		}
		
		// Memset the new part of the table to all zeroes - so we can tell
		// which slots are empty.
		
		f_memset( &m_pIndexTbl [m_uiIndexTblSize], 0,
					sizeof( F_INDEX) * (uiNewTblSize - m_uiIndexTblSize));
		m_uiIndexTblSize = uiNewTblSize;
	}
	pIndex = &m_pIndexTbl [uiIndexNum - 1];
	
	// Make sure we have not already defined this slot.
	
	if (pIndex->uiIndexNum)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_DUPLICATE_INDEX_NUM);
		goto Exit;
	}
	
	// Add the index name to the index name table.

	if (RC_BAD( rc = m_pIndexNames->copyName( pszIndexName,
								uiIndexNum, &pIndex->pszIndexName, &m_dictPool)))
	{
		goto Exit;
	}
	
	// Allocate space for the key components and data components
	// Must be at least one key component.
	
	flmAssert( uiNumKeyComponents);
	if (RC_BAD( rc = m_dictPool.poolCalloc(
					sizeof( ICD) * (uiNumKeyComponents + uiNumDataComponents),
					(void **)&pIndex->pKeyIcds)))
	{
		goto Exit;
	}
	if (uiNumDataComponents)
	{
		pIndex->pDataIcds = pIndex->pKeyIcds + uiNumKeyComponents;
	}
	else
	{
		pIndex->pDataIcds = NULL;
	}
	pIndex->uiNumKeyComponents = uiNumKeyComponents;
	pIndex->uiNumDataComponents = uiNumDataComponents;
	
	// Set other members of the index structure.
	
	pIndex->uiTableNum = uiTableNum;
	
	// NOTE: The substring flag should only be set when components are added
	// We AND it off here in case it was set by the caller.
	
	pIndex->uiFlags = uiFlags & (~(IXD_HAS_SUBSTRING));
	pIndex->uiLanguage = uiLanguage;
	pIndex->ui64LastRowIndexed = ui64LastRowIndexed;
	f_memset( &pIndex->lfInfo, 0, sizeof( LFILE));
	pIndex->lfInfo.uiLfNum = uiIndexNum;
	pIndex->lfInfo.uiEncDefNum = uiEncDefNum;
	pIndex->uiIndexNum = uiIndexNum;
	pIndex->ui64DefRowId = ui64DefRowId;
	
	// Link the index into the list of indexes for the table
	
	pIndex->uiNextIndexNum = pTable->uiFirstIndexNum;
	pTable->uiFirstIndexNum = uiIndexNum;
	
	if (uiIndexNum > m_uiHighestIndexNum)
	{
		m_uiHighestIndexNum = uiIndexNum;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Add an index column to an index's key component ICD array and/or to
		the index's data component ICD array.
****************************************************************************/
RCODE F_Dict::addIndexComponent(
	FLMUINT			uiIndexNum,
	FLMUINT64		ui64DefRowId,
	FLMUINT			uiColumnNum,
	FLMUINT			uiFlags,
	FLMUINT			uiCompareRules,
	FLMUINT			uiLimit,
	FLMUINT			uiKeyComponent,
	FLMUINT			uiDataComponent)
{
	RCODE			rc = NE_SFLM_OK;
	F_INDEX *	pIndex;
	F_TABLE *	pTable;
	ICD *			pIcd;
	F_COLUMN *	pColumn;
	
	// Verify the index number
	
	if ((pIndex = getIndex( uiIndexNum)) == NULL)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_INDEX_NUM);
		goto Exit;
	}
	
	// If we have a valid pIndex, it's table number should have already
	// been validated.
	
	pTable = getTable( pIndex->uiTableNum);
	flmAssert( pTable);
	
	// Verify the column number.

	if ((pColumn = getColumn( pTable, uiColumnNum)) == NULL)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_COLUMN_NUM);
		goto Exit;
	}
	
	// See if it is a key component.
	// NOTE: It is possible for the column to be both a key component and
	// a data component.
	
	if (uiKeyComponent)
	{
		if (uiKeyComponent > pIndex->uiNumKeyComponents)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_KEY_COMPONENT);
			goto Exit;
		}
		pIcd = &pIndex->pKeyIcds [uiKeyComponent - 1];
		
		// Make sure this component hasn't already been defined.
		
		if (pIcd->uiIndexNum)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_DUPLICATE_KEY_COMPONENT);
			goto Exit;
		}
		
		pIcd->uiIndexNum = uiIndexNum;
		pIcd->uiColumnNum = uiColumnNum;
		pIcd->ui64DefRowId = ui64DefRowId;
		pIcd->uiFlags = uiFlags;
		if (uiFlags & ICD_SUBSTRING)
		{
			pIndex->uiFlags |= IXD_HAS_SUBSTRING;
		}
		pIcd->uiCompareRules = uiCompareRules;
		pIcd->uiLimit = uiLimit;
		pIcd->pNextInChain = pColumn->pFirstIcd;
		pColumn->pFirstIcd = pIcd;
	}
	
	// See if it is a data component.
	
	if (uiDataComponent)
	{
		if (uiDataComponent > pIndex->uiNumDataComponents)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_DATA_COMPONENT);
			goto Exit;
		}
		
		pIcd = &pIndex->pDataIcds [uiDataComponent - 1];
		
		// Make sure this component hasn't already been defined.
		
		if (pIcd->uiIndexNum)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_DUPLICATE_DATA_COMPONENT);
			goto Exit;
		}
		
		pIcd->uiIndexNum = uiIndexNum;
		pIcd->uiColumnNum = uiColumnNum;
		pIcd->ui64DefRowId = ui64DefRowId;
		pIcd->uiFlags = 0;
		pIcd->uiCompareRules = 0;
		pIcd->pNextInChain = NULL;
		pIcd->uiLimit = 0;
		pIcd->pNextInDataChain = pColumn->pFirstDataIcd;
		pColumn->pFirstDataIcd = pIcd;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Setup the encryption definition table and its associated indexes.
****************************************************************************/
RCODE F_Dict::setupEncDefTable( void)
{
	RCODE	rc = NE_SFLM_OK;
	
	// Create the table
		
	if (RC_BAD( rc = addTable( SFLM_TBLNUM_ENCDEFS, 0,
										SFLM_TBLNAM_ENCDEFS, TRUE,
										SFLM_ENCDEFS_NUM_COLUMNS, 0)))
	{
		goto Exit;
	}
	
	// Add Columns
	
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_ENCDEFS, 0,
								SFLM_COLNUM_ENCDEFS_ENCDEF_NAME,
								SFLM_COLNAM_ENCDEFS_ENCDEF_NAME,
								0, SFLM_STRING_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_ENCDEFS, 0,
								SFLM_COLNUM_ENCDEFS_ENCDEF_NUM,
								SFLM_COLNAM_ENCDEFS_ENCDEF_NUM,
								COL_READ_ONLY, SFLM_NUMBER_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_ENCDEFS, 0,
								SFLM_COLNUM_ENCDEFS_ENC_ALGORITHM,
								SFLM_COLNAM_ENCDEFS_ENC_ALGORITHM,
								COL_READ_ONLY, SFLM_STRING_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_ENCDEFS, 0,
								SFLM_COLNUM_ENCDEFS_ENC_KEY_SIZE,
								SFLM_COLNAM_ENCDEFS_ENC_KEY_SIZE,
								COL_READ_ONLY, SFLM_NUMBER_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_ENCDEFS, 0,
								SFLM_COLNUM_ENCDEFS_ENC_KEY,
								SFLM_COLNAM_ENCDEFS_ENC_KEY,
								COL_READ_ONLY, SFLM_BINARY_TYPE, 0, 0)))
	{
		goto Exit;
	}
	
	// Add an index on the encdef table, encdef name column
	
	if (RC_BAD( rc = addIndex( SFLM_IXNUM_ENCDEF_NAME, 0,
										SFLM_IXNAM_ENCDEF_NAME,
										SFLM_TBLNUM_ENCDEFS, 0,
										IXD_SYSTEM, 1, 0, FLM_US_LANG, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addIndexComponent( SFLM_IXNUM_ENCDEF_NAME, 0,
							SFLM_COLNUM_ENCDEFS_ENCDEF_NAME,
							ICD_VALUE, FLM_COMP_CASE_INSENSITIVE, 0,
							1, 0)))
	{
		goto Exit;
	}
		
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Setup the table table and its associated indexes.
****************************************************************************/
RCODE F_Dict::setupTableTable( void)
{
	RCODE	rc = NE_SFLM_OK;
	
	// Create table
	
	if (RC_BAD( rc = addTable( SFLM_TBLNUM_TABLES, 0,
										SFLM_TBLNAM_TABLES, TRUE,
										SFLM_TABLES_NUM_COLUMNS, 0)))
	{
		goto Exit;
	}
	
	// Add Columns
	
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_TABLES, 0,
								SFLM_COLNUM_TABLES_TABLE_NAME,
								SFLM_COLNAM_TABLES_TABLE_NAME,
								0, SFLM_STRING_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_TABLES, 0,
								SFLM_COLNUM_TABLES_TABLE_NUM,
								SFLM_COLNAM_TABLES_TABLE_NUM,
								COL_READ_ONLY, SFLM_NUMBER_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_TABLES, 0,
								SFLM_COLNUM_TABLES_ENCDEF_NUM,
								SFLM_COLNAM_TABLES_ENCDEF_NUM,
								0, SFLM_NUMBER_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_TABLES, 0,
								SFLM_COLNUM_TABLES_NUM_COLUMNS,
								SFLM_COLNAM_TABLES_NUM_COLUMNS,
								0, SFLM_NUMBER_TYPE, 0, 0)))
	{
		goto Exit;
	}
	
	// Add an index on the table name column
	
	if (RC_BAD( rc = addIndex( SFLM_IXNUM_TABLE_NAME, 0,
										SFLM_IXNAM_TABLE_NAME,
										SFLM_TBLNUM_TABLES, 0,
										IXD_SYSTEM, 1, 0, FLM_US_LANG, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addIndexComponent( SFLM_IXNUM_TABLE_NAME, 0,
							SFLM_COLNUM_TABLES_TABLE_NAME,
							ICD_VALUE, FLM_COMP_CASE_INSENSITIVE, 0,
							1, 0)))
	{
		goto Exit;
	}
	
	// Add an index on the encdef number column
	
	if (RC_BAD( rc = addIndex( SFLM_IXNUM_TABLE_ENCDEF_NUM, 0,
										SFLM_IXNAM_TABLE_ENCDEF_NUM,
										SFLM_TBLNUM_TABLES, 0,
										IXD_SYSTEM, 1, 0, FLM_US_LANG, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addIndexComponent( SFLM_IXNUM_TABLE_ENCDEF_NUM, 0,
							SFLM_COLNUM_TABLES_ENCDEF_NUM,
							ICD_VALUE, 0, 0,
							1, 0)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}
	
/****************************************************************************
Desc:	Setup the column table and its associated indexes.
****************************************************************************/
RCODE F_Dict::setupColumnTable( void)
{
	RCODE	rc = NE_SFLM_OK;
	
	// Create table
	
	if (RC_BAD( rc = addTable( SFLM_TBLNUM_COLUMNS, 0,
										SFLM_TBLNAM_COLUMNS, TRUE,
										SFLM_COLUMNS_NUM_COLUMNS, 0)))
	{
		goto Exit;
	}
	
	// Add columns

	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_COLUMNS, 0,
								SFLM_COLNUM_COLUMNS_TABLE_NUM,
								SFLM_COLNAM_COLUMNS_TABLE_NUM,
								COL_READ_ONLY, SFLM_NUMBER_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_COLUMNS, 0,
								SFLM_COLNUM_COLUMNS_COLUMN_NAME,
								SFLM_COLNAM_COLUMNS_COLUMN_NAME,
								0, SFLM_STRING_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_COLUMNS, 0,
								SFLM_COLNUM_COLUMNS_COLUMN_NUM,
								SFLM_COLNAM_COLUMNS_COLUMN_NUM,
								COL_READ_ONLY, SFLM_NUMBER_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_COLUMNS, 0,
								SFLM_COLNUM_COLUMNS_DATA_TYPE,
								SFLM_COLNAM_COLUMNS_DATA_TYPE,
								COL_READ_ONLY, SFLM_STRING_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_COLUMNS, 0,
								SFLM_COLNUM_COLUMNS_MAX_LEN,
								SFLM_COLNAM_COLUMNS_MAX_LEN,
								COL_READ_ONLY, SFLM_NUMBER_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_COLUMNS, 0,
								SFLM_COLNUM_COLUMNS_ENCDEF_NUM,
								SFLM_COLNAM_COLUMNS_ENCDEF_NUM,
								COL_READ_ONLY | COL_NULL_ALLOWED,
								SFLM_NUMBER_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_COLUMNS, 0,
								SFLM_COLNUM_COLUMNS_READ_ONLY,
								SFLM_COLNAM_COLUMNS_READ_ONLY,
								COL_NULL_ALLOWED, SFLM_STRING_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_COLUMNS, 0,
								SFLM_COLNUM_COLUMNS_NULL_ALLOWED,
								SFLM_COLNAM_COLUMNS_NULL_ALLOWED,
								COL_NULL_ALLOWED, SFLM_STRING_TYPE, 0, 0)))
	{
		goto Exit;
	}
	
	// Add an index on the table number column
	
	if (RC_BAD( rc = addIndex( SFLM_IXNUM_COLUMN_TABLE_NUM, 0,
										SFLM_IXNAM_COLUMN_TABLE_NUM,
										SFLM_TBLNUM_COLUMNS, 0,
										IXD_SYSTEM, 1, 0, FLM_US_LANG, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addIndexComponent( SFLM_IXNUM_COLUMN_TABLE_NUM, 0,
								SFLM_COLNUM_COLUMNS_TABLE_NUM,
								ICD_VALUE, 0, 0,
								1, 0)))
	{
		goto Exit;
	}
		
	// Add an index on the encdef number column
	
	if (RC_BAD( rc = addIndex( SFLM_IXNUM_COLUMN_ENCDEF_NUM, 0,
										SFLM_IXNAM_COLUMN_ENCDEF_NUM,
										SFLM_TBLNUM_COLUMNS, 0,
										IXD_SYSTEM, 1, 0, FLM_US_LANG, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addIndexComponent( SFLM_IXNUM_COLUMN_ENCDEF_NUM, 0,
								SFLM_COLNUM_COLUMNS_ENCDEF_NUM,
								ICD_VALUE, 0, 0,
								1, 0)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Setup the index table and its associated indexes.
****************************************************************************/
RCODE F_Dict::setupIndexTable( void)
{
	RCODE	rc = NE_SFLM_OK;
	
	// Create the table

	if (RC_BAD( rc = addTable( SFLM_TBLNUM_INDEXES, 0,
										SFLM_TBLNAM_INDEXES, TRUE,
										SFLM_INDEXES_NUM_COLUMNS, 0)))
	{
		goto Exit;
	}
	
	// Add columns
	
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_INDEXES, 0,
								SFLM_COLNUM_INDEXES_INDEX_NAME,
								SFLM_COLNAM_INDEXES_INDEX_NAME,
								0, SFLM_STRING_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_INDEXES, 0,
								SFLM_COLNUM_INDEXES_INDEX_NUM,
								SFLM_COLNAM_INDEXES_INDEX_NUM,
								COL_READ_ONLY, SFLM_NUMBER_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_INDEXES, 0,
								SFLM_COLNUM_INDEXES_TABLE_NUM,
								SFLM_COLNAM_INDEXES_TABLE_NUM,
								COL_READ_ONLY, SFLM_NUMBER_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_INDEXES, 0,
								SFLM_COLNUM_INDEXES_ENCDEF_NUM,
								SFLM_COLNAM_INDEXES_ENCDEF_NUM,
								COL_READ_ONLY | COL_NULL_ALLOWED,
								SFLM_NUMBER_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_INDEXES, 0,
								SFLM_COLNUM_INDEXES_LANGUAGE,
								SFLM_COLNAM_INDEXES_LANGUAGE,
								COL_NULL_ALLOWED, SFLM_STRING_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_INDEXES, 0,
								SFLM_COLNUM_INDEXES_NUM_KEY_COMPONENTS,
								SFLM_COLNAM_INDEXES_NUM_KEY_COMPONENTS,
								COL_NULL_ALLOWED, SFLM_NUMBER_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_INDEXES, 0,
								SFLM_COLNUM_INDEXES_NUM_DATA_COMPONENTS,
								SFLM_COLNAM_INDEXES_NUM_DATA_COMPONENTS,
								COL_NULL_ALLOWED, SFLM_NUMBER_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_INDEXES, 0,
								SFLM_COLNUM_INDEXES_LAST_ROW_INDEXED,
								SFLM_COLNAM_INDEXES_LAST_ROW_INDEXED,
								COL_NULL_ALLOWED, SFLM_NUMBER_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_INDEXES, 0,
								SFLM_COLNUM_INDEXES_INDEX_STATE,
								SFLM_COLNAM_INDEXES_INDEX_STATE,
								COL_NULL_ALLOWED, SFLM_STRING_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_INDEXES, 0,
								SFLM_COLNUM_INDEXES_KEEP_ABS_POS_INFO,
								SFLM_COLNAM_INDEXES_KEEP_ABS_POS_INFO,
								COL_NULL_ALLOWED, SFLM_STRING_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_INDEXES, 0,
								SFLM_COLNUM_INDEXES_KEYS_UNIQUE,
								SFLM_COLNAM_INDEXES_KEYS_UNIQUE,
								COL_NULL_ALLOWED, SFLM_STRING_TYPE, 0, 0)))
	{
		goto Exit;
	}
	
	// Add an index on the index name column
	
	if (RC_BAD( rc = addIndex( SFLM_IXNUM_INDEX_NAME, 0,
										SFLM_IXNAM_INDEX_NAME,
										SFLM_TBLNUM_INDEXES, 0,
										IXD_SYSTEM, 1, 0, FLM_US_LANG, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addIndexComponent( SFLM_IXNUM_INDEX_NAME, 0,
									SFLM_COLNUM_INDEXES_INDEX_NAME,
									ICD_VALUE, FLM_COMP_CASE_INSENSITIVE, 0,
									1, 0)))
	{
		goto Exit;
	}
		
	// Add an index on the table number column
	
	if (RC_BAD( rc = addIndex( SFLM_IXNUM_INDEX_TABLE_NUM, 0,
										SFLM_IXNAM_INDEX_TABLE_NUM,
										SFLM_TBLNUM_INDEXES, 0,
										IXD_SYSTEM, 1, 0, FLM_US_LANG, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addIndexComponent( SFLM_IXNUM_INDEX_TABLE_NUM, 0,
									SFLM_COLNUM_INDEXES_TABLE_NUM,
									ICD_VALUE, 0, 0,
									1, 0)))
	{
		goto Exit;
	}
		
	// Add an index on the encdef number column
	
	if (RC_BAD( rc = addIndex( SFLM_IXNUM_INDEX_ENCDEF_NUM, 0,
										SFLM_IXNAM_INDEX_ENCDEF_NUM,
										SFLM_TBLNUM_INDEXES, 0,
										IXD_SYSTEM, 1, 0, FLM_US_LANG, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addIndexComponent( SFLM_IXNUM_INDEX_ENCDEF_NUM, 0,
									SFLM_COLNUM_INDEXES_ENCDEF_NUM,
									ICD_VALUE, 0, 0,
									1, 0)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Setup the index component table and its associated indexes.
****************************************************************************/
RCODE F_Dict::setupIndexComponentTable( void)
{
	RCODE	rc = NE_SFLM_OK;
	
	// Create the table

	if (RC_BAD( rc = addTable( SFLM_TBLNUM_INDEX_COMPONENTS, 0,
										SFLM_TBLNAM_INDEX_COMPONENTS, TRUE,
										SFLM_INDEX_COMP_NUM_COLUMNS, 0)))
	{
		goto Exit;
	}
	
	// Add Columns
	
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_INDEX_COMPONENTS, 0,
								SFLM_COLNUM_INDEX_COMP_INDEX_NUM,
								SFLM_COLNAM_INDEX_COMP_INDEX_NUM,
								COL_READ_ONLY, SFLM_NUMBER_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_INDEX_COMPONENTS, 0,
								SFLM_COLNUM_INDEX_COMP_COLUMN_NUM,
								SFLM_COLNAM_INDEX_COMP_COLUMN_NUM,
								COL_READ_ONLY, SFLM_NUMBER_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_INDEX_COMPONENTS, 0,
								SFLM_COLNUM_INDEX_COMP_KEY_COMPONENT,
								SFLM_COLNAM_INDEX_COMP_KEY_COMPONENT,
								COL_NULL_ALLOWED, SFLM_NUMBER_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_INDEX_COMPONENTS, 0,
								SFLM_COLNUM_INDEX_COMP_DATA_COMPONENT,
								SFLM_COLNAM_INDEX_COMP_DATA_COMPONENT,
								COL_NULL_ALLOWED, SFLM_NUMBER_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_INDEX_COMPONENTS, 0,
								SFLM_COLNUM_INDEX_COMP_INDEX_ON,
								SFLM_COLNAM_INDEX_COMP_INDEX_ON,
								COL_NULL_ALLOWED, SFLM_STRING_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_INDEX_COMPONENTS, 0,
								SFLM_COLNUM_INDEX_COMP_COMPARE_RULES,
								SFLM_COLNAM_INDEX_COMP_COMPARE_RULES,
								COL_NULL_ALLOWED, SFLM_STRING_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_INDEX_COMPONENTS, 0,
								SFLM_COLNUM_INDEX_COMP_SORT_DESCENDING,
								SFLM_COLNAM_INDEX_COMP_SORT_DESCENDING,
								COL_NULL_ALLOWED, SFLM_STRING_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_INDEX_COMPONENTS, 0,
								SFLM_COLNUM_INDEX_COMP_SORT_MISSING_HIGH,
								SFLM_COLNAM_INDEX_COMP_SORT_MISSING_HIGH,
								COL_NULL_ALLOWED, SFLM_STRING_TYPE, 0, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_INDEX_COMPONENTS, 0,
								SFLM_COLNUM_INDEX_COMP_LIMIT,
								SFLM_COLNAM_INDEX_COMP_LIMIT,
								COL_NULL_ALLOWED, SFLM_NUMBER_TYPE, 0, 0)))
	{
		goto Exit;
	}

	// Add an index on the index number column
	
	if (RC_BAD( rc = addIndex( SFLM_IXNUM_INDEX_COMP_INDEX_NUM, 0,
										SFLM_IXNAM_INDEX_COMP_INDEX_NUM,
										SFLM_TBLNUM_INDEX_COMPONENTS, 0,
										IXD_SYSTEM, 1, 0, FLM_US_LANG, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addIndexComponent( SFLM_IXNUM_INDEX_COMP_INDEX_NUM, 0,
								SFLM_COLNUM_INDEX_COMP_INDEX_NUM,
								ICD_VALUE, 0, 0,
								1, 0)))
	{
		goto Exit;
	}
		
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Setup the block chain table.
****************************************************************************/
RCODE F_Dict::setupBlockChainTable( void)
{
	RCODE	rc = NE_SFLM_OK;
	
	// Create the table

	if (RC_BAD( rc = addTable( SFLM_TBLNUM_BLOCK_CHAINS, 0,
										SFLM_TBLNAM_BLOCK_CHAINS, TRUE,
										SFLM_BLOCK_CHAINS_NUM_COLUMNS, 0)))
	{
		goto Exit;
	}
	
	// Add columns
	
	if (RC_BAD( rc = addColumn( SFLM_TBLNUM_BLOCK_CHAINS, 0,
								SFLM_COLNUM_BLOCK_CHAINS_BLOCK_ADDRESS,
								SFLM_COLNAM_BLOCK_CHAINS_BLOCK_ADDRESS,
								0, SFLM_NUMBER_TYPE, 0, 0)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Setup the predefined tables and indexes.
****************************************************************************/
RCODE F_Dict::setupPredefined( void)
{
	RCODE	rc = NE_SFLM_OK;
	
	// Set up the encryption definition table
	
	if (RC_BAD( rc = setupEncDefTable()))
	{
		goto Exit;
	}
	
	// Set up the table table
	
	if (RC_BAD( rc = setupTableTable()))
	{
		goto Exit;
	}

	// Set up the column table
	
	if (RC_BAD( rc = setupColumnTable()))
	{
		goto Exit;
	}
	
	// Set up the index table
	
	if (RC_BAD( rc = setupIndexTable()))
	{
		goto Exit;
	}
	
	// Set up the index component table
	
	if (RC_BAD( rc = setupIndexComponentTable()))
	{
		goto Exit;
	}
	
	// Set up the block chain deletion table
	
	if (RC_BAD( rc= setupBlockChainTable()))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Verify that a dictionary is good after having read it in.  These are
		things that didn't get verified as we were reading in.
****************************************************************************/
RCODE F_Dict::verifyDict( void)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiTableNum;
	F_TABLE *	pTable;
	FLMUINT		uiColumnNum;
	F_COLUMN *	pColumn;
	FLMUINT		uiIndexNum;
	F_INDEX *	pIndex;
	ICD *			pIcd;
	FLMUINT		uiKeyComponent;
	FLMUINT		uiDataComponent;
	
	// Make sure we have no duplicate table names, index names, or encryption
	// definition names.
	
	if (m_pTableNames)
	{
		m_pTableNames->sortNames();
		if (m_pTableNames->m_bDuplicateNames)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_DUPLICATE_TABLE_NAME);
			goto Exit;
		}
	}
	if (m_pIndexNames)
	{
		m_pIndexNames->sortNames();
		if (m_pTableNames->m_bDuplicateNames)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_DUPLICATE_INDEX_NAME);
			goto Exit;
		}
	}
	if (m_pEncDefNames)
	{
		m_pEncDefNames->sortNames();
		if (m_pTableNames->m_bDuplicateNames)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_DUPLICATE_ENCDEF_NAME);
			goto Exit;
		}
	}
	
	// Loop through all of the tables.  Verify that all of the column numbers
	// are defined, and that for each column name table there are no duplicate
	// column names.
	
	for (uiTableNum = 0, pTable = m_pTableTbl;
		  uiTableNum < m_uiHighestTableNum;
		  uiTableNum++, pTable++)
	{
		if (!pTable->uiTableNum)
		{
			continue;
		}
		
		// Sort the column names - make sure there are no duplicate column names.
		
		pTable->pColumnNames->sortNames();
		if (pTable->pColumnNames->m_bDuplicateNames)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_DUPLICATE_COLUMN_NAME);
			goto Exit;
		}
		
		// Make sure each column is defined.
		
		for (uiColumnNum = 0, pColumn = pTable->pColumns;
			  uiColumnNum < pTable->uiNumColumns;
			  uiColumnNum++, pColumn++)
		{
			
			// uiColumnNum will be zero if it never got defined.
			
			if (!pColumn->uiColumnNum)
			{
				rc = RC_SET_AND_ASSERT( NE_SFLM_UNDEFINED_COLUMN_NUM);
				goto Exit;
			}
		}
	}
	
	// Verify that for each index all of the key components and data components
	// are defined.

	for (uiIndexNum = 0, pIndex = m_pIndexTbl;
		  uiIndexNum < m_uiHighestIndexNum;
		  uiIndexNum++, pIndex++)
	{
		if (!pIndex->uiIndexNum)
		{
			continue;
		}
		
		// Make sure all key components are defined.
		
		for (uiKeyComponent = 0, pIcd = pIndex->pKeyIcds;
			  uiKeyComponent < pIndex->uiNumKeyComponents;
			  uiKeyComponent++, pIcd++)
		{
			
			// If the ICD is not defined, the uiIndexNum field will be 0
			
			if (!pIcd->uiIndexNum)
			{
				rc = RC_SET_AND_ASSERT( NE_SFLM_UNDEFINED_KEY_COMPONENT);
				goto Exit;
			}
		}
		
		// Make sure all data components are defined.
		
		for (uiDataComponent = 0, pIcd = pIndex->pDataIcds;
			  uiDataComponent < pIndex->uiNumDataComponents;
			  uiDataComponent++, pIcd++)
		{
			
			// If the ICD is not defined, the uiIndexNum field will be 0
			
			if (!pIcd->uiIndexNum)
			{
				rc = RC_SET_AND_ASSERT( NE_SFLM_UNDEFINED_DATA_COMPONENT);
				goto Exit;
			}
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Read in LFH headers.
****************************************************************************/
RCODE F_Db::dictReadLFH( void)
{
	RCODE					rc = NE_SFLM_OK;
	F_TABLE *			pTable;
	F_INDEX *			pIndex;
	F_CachedBlock *	pSCache;
	FLMBOOL				bReleaseCache = FALSE;
	F_BLK_HDR *			pBlkHdr;
	FLMUINT				uiBlkAddress;
	FLMUINT				uiPos;
	FLMUINT				uiEndPos;
	FLMUINT				uiBlkSize = m_pDatabase->m_uiBlockSize;
	LFILE					TmpLFile;
	
	f_memset( &TmpLFile, 0, sizeof( LFILE));

	// Read through all of the LFILE blocks.
	
	uiBlkAddress =
			(FLMUINT)m_pDatabase->m_lastCommittedDbHdr.ui32FirstLFBlkAddr;
	while (uiBlkAddress)
	{
		if (RC_BAD( rc = m_pDatabase->getBlock( this, NULL, 
			uiBlkAddress, NULL, &pSCache)))
		{
			goto Exit;
		}
		bReleaseCache = TRUE;

		pBlkHdr = pSCache->m_pBlkHdr;
		uiPos = SIZEOF_STD_BLK_HDR;
		uiEndPos = blkGetEnd( uiBlkSize, SIZEOF_STD_BLK_HDR, pBlkHdr);

		// Read through all of the logical file definitions in the block

		for( ; uiPos + sizeof( F_LF_HDR) <= uiEndPos; uiPos += sizeof( F_LF_HDR))
		{
			F_LF_HDR *	pLfHdr = (F_LF_HDR *)((FLMBYTE *)(pBlkHdr) + uiPos);
			eLFileType	eLfType = (eLFileType)pLfHdr->ui32LfType;

			// Have to fix up the offsets later when they are read in

			if (eLfType == SFLM_LF_INVALID)
			{
				continue;
			}

			// Populate the LFILE in the dictionary, if one has been set up.

			if (eLfType == SFLM_LF_INDEX)
			{
				FSLFileIn( (FLMBYTE *)pLfHdr, &TmpLFile, uiBlkAddress, uiPos);

				if ((pIndex = m_pDict->getIndex( TmpLFile.uiLfNum)) != NULL)
				{
					f_memcpy( &pIndex->lfInfo, &TmpLFile, sizeof( LFILE));
				}

				// LFILE better have a non-zero root block.

				if (!TmpLFile.uiRootBlk)
				{
					rc = RC_SET_AND_ASSERT( NE_SFLM_DATA_ERROR);
					goto Exit;
				}
			}
			else
			{

				// Better be a container

				flmAssert( eLfType == SFLM_LF_TABLE);

				FSLFileIn( (FLMBYTE *)pLfHdr, &TmpLFile, uiBlkAddress, uiPos);

				if ((pTable = m_pDict->getTable( TmpLFile.uiLfNum)) != NULL)
				{
					f_memcpy( &pTable->lfInfo, &TmpLFile, sizeof( LFILE));
				}

				// LFILE better have a non-zero root block.

				if (!TmpLFile.uiRootBlk)
				{
					rc = RC_SET_AND_ASSERT( NE_SFLM_DATA_ERROR);
					goto Exit;
				}
			}
		}

		// Get the next block in the chain

		uiBlkAddress = (FLMUINT)pBlkHdr->ui32NextBlkInChain;
		ScaReleaseCache( pSCache, FALSE);
		bReleaseCache = FALSE;
	}

Exit:

	if (bReleaseCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	return( rc );
}

/****************************************************************************
Desc:	Read in all rows from the encryption definition table.
		Create the in-memory structures for each encryption definition.
****************************************************************************/
RCODE F_Db::dictReadEncDefs( void)
{
	RCODE					rc = NE_SFLM_OK;
	F_Row *				pRow = NULL;
	FLMUINT64			ui64DefRowId;
	FSTableCursor *	pTableCursor = NULL;
	char					szEncDefName [MAX_ENCDEF_NAME_LEN + 1];
	FLMUINT				uiEncDefNameLen;
	FLMUINT				uiEncDefNum;
	char					szEncAlgorithm [20];
	eEncAlgorithm		eEncAlg;
	FLMUINT				uiEncKeySize;
	FLMBYTE *			pucEncKey = NULL;
	FLMUINT				uiEncKeyLen;
	FLMBOOL				bIsNull;
	
	if ((pTableCursor = f_new FSTableCursor) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = pTableCursor->setupRange( this, SFLM_TBLNUM_ENCDEFS,
											1, FLM_MAX_UINT64, FALSE)))
	{
		goto Exit;
	}
	
	// Read through all rows in the table.  Each row defines a table in
	// the database, so we add it to the in-memory dictionary.
	
	for (;;)
	{
		if (RC_BAD( rc = pTableCursor->nextRow( this, &pRow, &ui64DefRowId)))
		{
			if (rc == NE_SFLM_EOF_HIT)
			{
				rc = NE_SFLM_OK;
			}
			goto Exit;
		}
		
		// Get the encryption definition name - required.
		
		if (RC_BAD( rc = pRow->getUTF8( this, SFLM_COLNUM_ENCDEFS_ENCDEF_NAME,
											szEncDefName, sizeof( szEncDefName),
											&bIsNull, &uiEncDefNameLen, NULL)))
		{
			goto Exit;
		}
		if (bIsNull || !uiEncDefNameLen)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_NULL_ENCDEF_NAME);
			goto Exit;
		}
		
		// Get the encryption definition number - required.
		
		if (RC_BAD( rc = pRow->getUINT( this, SFLM_COLNUM_ENCDEFS_ENCDEF_NUM,
											&uiEncDefNum, &bIsNull)))
		{
			goto Exit;
		}
		if (bIsNull)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_NULL_ENCDEF_NUM);
			goto Exit;
		}
		
		// Get the encryption algorithm - required.
		
		if (RC_BAD( rc = pRow->getUTF8( this, SFLM_COLNUM_ENCDEFS_ENC_ALGORITHM,
											szEncAlgorithm, sizeof( szEncAlgorithm),
											&bIsNull, NULL, NULL)))
		{
			goto Exit;
		}
		if (bIsNull || !szEncAlgorithm [0])
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_NULL_ENC_ALGORITHM);
			goto Exit;
		}
		
		// Make sure the encryption algorithm and key size are valid.
		
		if (!validEncAlgorithm( szEncAlgorithm, &eEncAlg))
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_ENC_ALGORITHM);
			goto Exit;
		}
		
		// Get the encryption key size - required.
		
		if (RC_BAD( rc = pRow->getUINT( this, SFLM_COLNUM_ENCDEFS_ENC_KEY_SIZE,
											&uiEncKeySize, &bIsNull)))
		{
			goto Exit;
		}
		if (bIsNull)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_NULL_ENC_KEY_SIZE);
			goto Exit;
		}
		if (!validEncKeySize( eEncAlg, uiEncKeySize))
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_ENC_KEY_SIZE);
			goto Exit;
		}
		
		// Get the encryption key.  There better be one at this point.
		
		pRow->getDataLen( this, SFLM_COLNUM_ENCDEFS_ENC_KEY,
											&uiEncKeyLen, &bIsNull);
		if (bIsNull || !uiEncKeyLen)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_NULL_ENC_KEY);
			goto Exit;
		}
		if (RC_BAD( rc = f_alloc( uiEncKeyLen, &pucEncKey)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = pRow->getBinary( this, SFLM_COLNUM_ENCDEFS_ENC_KEY,
											pucEncKey, uiEncKeyLen, &uiEncKeyLen,
											&bIsNull)))
		{
			goto Exit;
		}
		flmAssert( !bIsNull);	// If encryption key length > 0, this better never be TRUE!
		
		if (RC_BAD( rc = m_pDict->addEncDef( uiEncDefNum, ui64DefRowId,
											szEncDefName, eEncAlg,
											uiEncKeySize, pucEncKey, uiEncKeyLen)))
		{
			goto Exit;
		}
	}

Exit:

	if (pRow)
	{
		pRow->ReleaseRow();
	}
	if (pucEncKey)
	{
		f_free( &pucEncKey);
	}
	if (pTableCursor)
	{
		pTableCursor->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Read in all rows from the table table.  Create the in-memory structures
		for each defined table.
****************************************************************************/
RCODE F_Db::dictReadTables( void)
{
	RCODE					rc = NE_SFLM_OK;
	F_Row *				pRow = NULL;
	FLMUINT64			ui64DefRowId;
	FSTableCursor *	pTableCursor = NULL;
	char					szTableName [MAX_TABLE_NAME_LEN + 1];
	FLMUINT				uiTableNameLen;
	FLMUINT				uiTableNum;
	FLMUINT				uiEncDefNum;
	FLMUINT				uiNumColumns;
	FLMBOOL				bIsNull;

	if ((pTableCursor = f_new FSTableCursor) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = pTableCursor->setupRange( this, SFLM_TBLNUM_TABLES,
											1, FLM_MAX_UINT64, FALSE)))
	{
		goto Exit;
	}
	
	// Read through all rows in the table.  Each row defines a table in
	// the database, so we add it to the in-memory dictionary.
	
	for (;;)
	{
		if (RC_BAD( rc = pTableCursor->nextRow( this, &pRow, &ui64DefRowId)))
		{
			if (rc == NE_SFLM_EOF_HIT)
			{
				rc = NE_SFLM_OK;
			}
			goto Exit;
		}
		
		// Get the table name - required.
		
		if (RC_BAD( rc = pRow->getUTF8( this, SFLM_COLNUM_TABLES_TABLE_NAME,
											szTableName, sizeof( szTableName),
											&bIsNull, &uiTableNameLen, NULL)))
		{
			goto Exit;
		}
		if (bIsNull || !uiTableNameLen)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_NULL_TABLE_NAME);
			goto Exit;
		}
		
		// Get the table number - required.
		
		if (RC_BAD( rc = pRow->getUINT( this, SFLM_COLNUM_TABLES_TABLE_NUM,
											&uiTableNum, &bIsNull)))
		{
			goto Exit;
		}
		if (bIsNull)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_NULL_TABLE_NUM);
			goto Exit;
		}
		
		// Get the encryption name - optional.  But if present, it must be
		// defined in the dictionary already.
		
		if (RC_BAD( rc = pRow->getUINT( this, SFLM_COLNUM_TABLES_ENCDEF_NUM,
											&uiEncDefNum, &bIsNull)))
		{
			goto Exit;
		}
		if (bIsNull)
		{
			uiEncDefNum = 0;
		}
		
		// Get the number of columns.  Must be non-zero.
		
		if (RC_BAD( rc = pRow->getUINT( this, SFLM_COLNUM_TABLES_NUM_COLUMNS,
											&uiNumColumns, &bIsNull)))
		{
			goto Exit;
		}
		if (bIsNull)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_NULL_NUM_COLUMNS);
			goto Exit;
		}
		if (!uiNumColumns)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_NUM_COLUMNS);
			goto Exit;
		}
		
		// Add an entry to the in-memory dictionary for the table.

		if (RC_BAD( rc = m_pDict->addTable( uiTableNum, ui64DefRowId,
														szTableName, FALSE,
														uiNumColumns, uiEncDefNum)))
		{
			goto Exit;
		}
	}

Exit:

	if (pRow)
	{
		pRow->ReleaseRow();
	}

	if (pTableCursor)
	{
		pTableCursor->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Read in all rows from the column table.  Create the in-memory structures
		for each defined column.
****************************************************************************/
RCODE F_Db::dictReadColumns( void)
{
	RCODE					rc = NE_SFLM_OK;
	F_Row *				pRow = NULL;
	FLMUINT64			ui64DefRowId;
	FSTableCursor *	pTableCursor = NULL;
	FLMUINT				uiEncDefNum;
	FLMUINT				uiTableNum;
	char					szColumnName [MAX_COLUMN_NAME_LEN + 1];
	FLMUINT				uiColumnNameLen;
	FLMUINT				uiColumnNum;
	eDataType			eDataTyp;
	FLMUINT				uiFlags;
	char					szTmp [100];
	FLMUINT				uiTmpLen;
	FLMBOOL				bTmp;
	FLMBOOL				bIsNull;
	FLMUINT				uiMaxLen;

	if ((pTableCursor = f_new FSTableCursor) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = pTableCursor->setupRange( this, SFLM_TBLNUM_COLUMNS,
											1, FLM_MAX_UINT64, FALSE)))
	{
		goto Exit;
	}
	
	// Read through all rows in the table.  Each row defines a column in
	// a table in the database, so we add it to the in-memory dictionary.
	
	for (;;)
	{
		if (RC_BAD( rc = pTableCursor->nextRow( this, &pRow, &ui64DefRowId)))
		{
			if (rc == NE_SFLM_EOF_HIT)
			{
				rc = NE_SFLM_OK;
			}
			goto Exit;
		}
		
		// Get the table number - required.
		
		if (RC_BAD( rc = pRow->getUINT( this, SFLM_COLNUM_COLUMNS_TABLE_NUM,
											&uiTableNum, &bIsNull)))
		{
			goto Exit;
		}
		if (bIsNull)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_NULL_TABLE_NUM);
			goto Exit;
		}
		
		// Get the column name.  There must be a column name.
		
		if (RC_BAD( rc = pRow->getUTF8( this, SFLM_COLNUM_COLUMNS_COLUMN_NAME,
											szColumnName, sizeof( szColumnName),
											&bIsNull, &uiColumnNameLen, NULL)))
		{
			goto Exit;
		}
		if (bIsNull || !uiColumnNameLen)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_NULL_COLUMN_NAME);
			goto Exit;
		}
		
		// Get column number - must be non-zero.
		
		if (RC_BAD( rc = pRow->getUINT( this, SFLM_COLNUM_COLUMNS_COLUMN_NUM,
											&uiColumnNum, &bIsNull)))
		{
			goto Exit;
		}
		if (bIsNull)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_NULL_COLUMN_NUM);
			goto Exit;
		}
		flmAssert( uiColumnNum);
		
		// Get the encryption definition number - optional.
		
		if (RC_BAD( rc = pRow->getUINT( this, SFLM_COLNUM_COLUMNS_ENCDEF_NUM,
											&uiEncDefNum, &bIsNull)))
		{
			goto Exit;
		}
		if (bIsNull)
		{
			uiEncDefNum = 0;
		}
		
		// Get the data type - required.
		
		if (RC_BAD( rc = pRow->getUTF8( this, SFLM_COLNUM_COLUMNS_DATA_TYPE,
											szTmp, sizeof( szTmp),
											&bIsNull, &uiTmpLen, NULL)))
		{
			goto Exit;
		}
		if (bIsNull || !uiTmpLen)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_NULL_DATA_TYPE);
			goto Exit;
		}
		if (!validDataType( szTmp, &eDataTyp))
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_ILLEGAL_DATA_TYPE);
			goto Exit;
		}
		
		uiFlags = 0;
		
		// Get the maximum length - optional
		
		if (RC_BAD( rc = pRow->getUINT( this, SFLM_COLNUM_COLUMNS_MAX_LEN,
										&uiMaxLen, &bIsNull)))
		{
			goto Exit;
		}
		if (bIsNull)
		{
			uiMaxLen = 0;
		}
		
		// Get the read-only flag - optional.
		
		if (RC_BAD( rc = pRow->getUTF8( this, SFLM_COLNUM_COLUMNS_READ_ONLY,
											szTmp, sizeof( szTmp),
											&bIsNull, &uiTmpLen, NULL)))
		{
			goto Exit;
		}
		if (!bIsNull && uiTmpLen)
		{
			if (!validBooleanValue( szTmp, &bTmp))
			{
				rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_READ_ONLY_VALUE);
				goto Exit;
			}
			if (bTmp)
			{
				uiFlags |= COL_READ_ONLY;
			}
		}
		
		// Get the null-allowed flag
		
		if (RC_BAD( rc = pRow->getUTF8( this, SFLM_COLNUM_COLUMNS_NULL_ALLOWED,
											szTmp, sizeof( szTmp),
											&bIsNull, &uiTmpLen, NULL)))
		{
			goto Exit;
		}
		if (!bIsNull && uiTmpLen)
		{
			if (!validBooleanValue( szTmp, &bTmp))
			{
				rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_NULL_ALLOWED_VALUE);
				goto Exit;
			}
			if (bTmp)
			{
				uiFlags |= COL_NULL_ALLOWED;
			}
		}
		
		// Add the column to the table entry in the in-memory dictionary.

		if (RC_BAD( rc = m_pDict->addColumn( uiTableNum, ui64DefRowId,
												uiColumnNum, szColumnName, uiFlags,
												eDataTyp, uiMaxLen, uiEncDefNum)))
		{
			goto Exit;
		}
	}

Exit:

	if (pRow)
	{
		pRow->ReleaseRow();
	}

	if (pTableCursor)
	{
		pTableCursor->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Test the index state string to see if it is valid.  Set index flags
		accordingly.
****************************************************************************/
FSTATIC FLMBOOL validIndexState(
	const char *	pszIndexState,
	FLMUINT *		puiFlags)
{
	if (f_stricmp( pszIndexState, SFLM_INDEX_SUSPENDED_STR) == 0)
	{
		(*puiFlags) |= IXD_SUSPENDED;
		(*puiFlags) &= (~(IXD_OFFLINE));
		return( TRUE);
	}
	else if (f_stricmp( pszIndexState, SFLM_INDEX_OFFLINE_STR) == 0)
	{
		(*puiFlags) |= IXD_OFFLINE;
		(*puiFlags) &= (~(IXD_SUSPENDED));
		return( TRUE);
	}
	else if (f_stricmp( pszIndexState, SFLM_INDEX_ONLINE_STR) == 0)
	{
		(*puiFlags) &= (~(IXD_SUSPENDED | IXD_OFFLINE));
		return( TRUE);
	}
	return( FALSE);
}
		
/****************************************************************************
Desc:	Read in all rows from the index table.  Create the in-memory structures
		for each index.
****************************************************************************/
RCODE F_Db::dictReadIndexes( void)
{
	RCODE					rc = NE_SFLM_OK;
	F_Row *				pRow = NULL;
	FLMUINT64			ui64DefRowId;
	FSTableCursor *	pTableCursor = NULL;
	FLMUINT				uiEncDefNum;
	FLMUINT				uiTableNum;
	char					szIndexName [MAX_INDEX_NAME_LEN + 1];
	FLMUINT				uiIndexNameLen;
	FLMUINT				uiIndexNum;
	char					szTmp [50];
	FLMBOOL				bTmp;
	FLMUINT				uiTmpLen;
	FLMUINT				uiLanguage;
	FLMUINT64			ui64LastRowIndexed;
	FLMUINT				uiNumKeyComponents;
	FLMUINT				uiNumDataComponents;
	FLMUINT				uiFlags;
	FLMBOOL				bIsNull;

	if ((pTableCursor = f_new FSTableCursor) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = pTableCursor->setupRange( this, SFLM_TBLNUM_INDEXES,
											1, FLM_MAX_UINT64, FALSE)))
	{
		goto Exit;
	}
	
	// Read through all rows in the table.  Each row defines an index on
	// a table in the database, so we add it to the in-memory dictionary.
	
	for (;;)
	{
		if (RC_BAD( rc = pTableCursor->nextRow( this, &pRow, &ui64DefRowId)))
		{
			if (rc == NE_SFLM_EOF_HIT)
			{
				rc = NE_SFLM_OK;
			}
			goto Exit;
		}
		
		// Get the index name - required.
		
		if (RC_BAD( rc = pRow->getUTF8( this, SFLM_COLNUM_INDEXES_INDEX_NAME,
											szIndexName, sizeof( szIndexName),
											&bIsNull, &uiIndexNameLen, NULL)))
		{
			goto Exit;
		}
		if (bIsNull || !uiIndexNameLen)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_NULL_INDEX_NAME);
			goto Exit;
		}
		
		// Get the index number - required.
		
		if (RC_BAD( rc = pRow->getUINT( this, SFLM_COLNUM_INDEXES_INDEX_NUM,
											&uiIndexNum, &bIsNull)))
		{
			goto Exit;
		}
		if (bIsNull)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_NULL_INDEX_NUM);
			goto Exit;
		}
		
		// Get the table number - required.
		
		if (RC_BAD( rc = pRow->getUINT( this, SFLM_COLNUM_INDEXES_TABLE_NUM,
											&uiTableNum, &bIsNull)))
		{
			goto Exit;
		}
		if (bIsNull)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_NULL_TABLE_NUM);
			goto Exit;
		}
		
		// Get the encryption definition number - optional.
		
		if (RC_BAD( rc = pRow->getUINT( this, SFLM_COLNUM_INDEXES_ENCDEF_NUM,
											&uiEncDefNum, &bIsNull)))
		{
			goto Exit;
		}
		if (bIsNull)
		{
			uiEncDefNum = 0;
		}

		// Get the number of key components - optional, defaults to zero.
		
		if (RC_BAD( rc = pRow->getUINT( this, SFLM_COLNUM_INDEXES_NUM_KEY_COMPONENTS,
											&uiNumKeyComponents, &bIsNull)))
		{
			goto Exit;
		}
		if (bIsNull)
		{
			uiNumKeyComponents = 0;
		}
		
		// Get the number of data components - optional, defaults to zero
		
		if (RC_BAD( rc = pRow->getUINT( this, SFLM_COLNUM_INDEXES_NUM_DATA_COMPONENTS,
											&uiNumDataComponents, &bIsNull)))
		{
			goto Exit;
		}
		if (bIsNull)
		{
			uiNumDataComponents = 0;
		}
	
		// Get index language - optional.
		
		uiLanguage = FLM_US_LANG;
		if (RC_BAD( rc = pRow->getUTF8( this, SFLM_COLNUM_INDEXES_LANGUAGE,
											szTmp, sizeof( szTmp),
											&bIsNull, &uiTmpLen, NULL)))
		{
			goto Exit;
		}
		if (!bIsNull && uiTmpLen)
		{
			if (uiTmpLen != 2)
			{
				rc = RC_SET_AND_ASSERT( NE_SFLM_ILLEGAL_LANGUAGE);
				goto Exit;
			}
			
			uiLanguage = f_languageToNum( szTmp);
			
			// f_languageToNum returns FLM_US_LANG for all strings it doesn't
			// recognize.  If that is what is returned make sure that the string
			// is "US".
			
			if (uiLanguage == FLM_US_LANG && f_stricmp( szTmp, "US") != 0)
			{
				rc = RC_SET_AND_ASSERT( NE_SFLM_ILLEGAL_LANGUAGE);
				goto Exit;
			}
		}
		
		// Get the last row indexed, optional - defaults to zero.
		
		if (RC_BAD( rc = pRow->getUINT64( this, SFLM_COLNUM_INDEXES_LAST_ROW_INDEXED,
											&ui64LastRowIndexed, &bIsNull)))
		{
			goto Exit;
		}
		if (bIsNull)
		{
			ui64LastRowIndexed = 0;
		}
		
		// Get the index state - optional.
		
		if (RC_BAD( rc = pRow->getUTF8( this, SFLM_COLNUM_INDEXES_INDEX_STATE,
											szTmp, sizeof( szTmp), &bIsNull, &uiTmpLen, NULL)))
		{
			goto Exit;
		}
		uiFlags = 0;
		if (!bIsNull && uiTmpLen)
		{
			if (!validIndexState( szTmp, &uiFlags))
			{
				rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_INDEX_STATE);
				goto Exit;
			}
		}
			
		// Get the keep absolute positioning flag - optional.
		
		if (RC_BAD( rc = pRow->getUTF8( this, SFLM_COLNUM_INDEXES_KEEP_ABS_POS_INFO,
											szTmp, sizeof( szTmp), &bIsNull, &uiTmpLen, NULL)))
		{
			goto Exit;
		}
		if (!bIsNull && uiTmpLen)
		{
			if (!validBooleanValue( szTmp, &bTmp))
			{
				rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_KEEP_ABS_POS_INFO_VALUE);
				goto Exit;
			}
			if (bTmp)
			{
				uiFlags |= IXD_ABS_POS;
			}
		}
		
		// Get the keys unique flag - optional.
		
		if (RC_BAD( rc = pRow->getUTF8( this, SFLM_COLNUM_INDEXES_KEYS_UNIQUE,
											szTmp, sizeof( szTmp), &bIsNull, &uiTmpLen, NULL)))
		{
			goto Exit;
		}
		if (!bIsNull && uiTmpLen)
		{
			if (!validBooleanValue( szTmp, &bTmp))
			{
				rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_KEYS_UNIQUE_VALUE);
				goto Exit;
			}
			if (bTmp)
			{
				uiFlags |= IXD_KEYS_UNIQUE;
			}
		}
		
		// Add the index to the the in-memory dictionary.
		
		if (RC_BAD( rc = m_pDict->addIndex( uiIndexNum, ui64DefRowId,
												szIndexName, uiTableNum, uiEncDefNum,
												uiFlags, uiNumKeyComponents,
												uiNumDataComponents, uiLanguage,
												ui64LastRowIndexed)))
		{
			goto Exit;
		}
	}

Exit:

	if (pRow)
	{
		pRow->ReleaseRow();
	}

	if (pTableCursor)
	{
		pTableCursor->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Determine if the "index on" value for an index component is valid.
****************************************************************************/
FSTATIC FLMBOOL validIndexOnValue(
	const char *	pszIndexOn,
	FLMUINT *		puiFlags)
{
	if (f_stricmp( pszIndexOn, SFLM_VALUE_OPTION_STR) == 0)
	{
		(*puiFlags) |= ICD_VALUE;
		return( TRUE);
	}
	else if (f_stricmp( pszIndexOn, SFLM_PRESENCE_OPTION_STR) == 0)
	{
		(*puiFlags) |= ICD_PRESENCE;
		return( TRUE);
	}
	else if (f_stricmp( pszIndexOn, SFLM_SUBSTRING_OPTION_STR) == 0)
	{
		(*puiFlags) |= ICD_SUBSTRING;
		return( TRUE);
	}
	else if (f_stricmp( pszIndexOn, SFLM_EACHWORD_OPTION_STR) == 0)
	{
		(*puiFlags) |= ICD_EACHWORD;
		return( TRUE);
	}
	else if (f_stricmp( pszIndexOn, SFLM_METAPHONE_OPTION_STR) == 0)
	{
		(*puiFlags) |= ICD_METAPHONE;
		return( TRUE);
	}
	return( FALSE);
}

/****************************************************************************
Desc:	Determine if the "compare rules" value for an index component is valid.
		NOTE: pszCompareRules will be modified, but it doesn't matter because
		the caller should have passed a temporary buffer.
****************************************************************************/
FSTATIC FLMBOOL validCompareRulesValue(
	char *			pszCompareRules,
	FLMUINT *		puiCompareRules)
{
	char *	pszRuleStart;
	char *	pszRuleEnd;
	
	pszRuleStart = pszCompareRules;
	while (*pszRuleStart)
	{
		// Skip leading spaces, tabs, newlines, and commas.
		
		while (*pszRuleStart == ' ' || *pszRuleStart == ',' ||
				 *pszRuleStart == '\t' || *pszRuleStart == '\n')
		{
			pszRuleStart++;
		}
		
		// If we are at the end, there are no more rules to look at.
		
		if (*pszRuleStart == 0)
		{
			break;
		}
		
		// Go until we hit a comma or whitespace.
		
		pszRuleEnd = pszRuleStart;
		while (*pszRuleEnd && *pszRuleEnd != ' ' && *pszRuleEnd != ',' &&
				 *pszRuleEnd != '\t' && *pszRuleEnd != '\n')
		{
			pszRuleEnd++;
		}
		if (*pszRuleEnd)
		{
			*pszRuleEnd = 0;
			pszRuleEnd++;
		}
		
		// See if the rule is valid.
		
		if (f_stricmp( SFLM_CASE_INSENSITIVE_OPTION_STR, pszRuleStart) == 0)
		{
			(*puiCompareRules) |= FLM_COMP_CASE_INSENSITIVE;
		}
		else if (f_stricmp( SFLM_COMPRESS_WHITESPACE_OPTION_STR, pszRuleStart) == 0)
		{
			(*puiCompareRules) |= FLM_COMP_COMPRESS_WHITESPACE;
		}
		else if (f_stricmp( SFLM_WHITESPACE_AS_SPACE_STR, pszRuleStart) == 0)
		{
			(*puiCompareRules) |= FLM_COMP_WHITESPACE_AS_SPACE;
		}
		else if (f_stricmp( SFLM_IGNORE_LEADINGSPACES_OPTION_STR, pszRuleStart) == 0)
		{
			(*puiCompareRules) |= FLM_COMP_IGNORE_LEADING_SPACE;
		}
		else if (f_stricmp( SFLM_IGNORE_TRAILINGSPACES_OPTION_STR, pszRuleStart) == 0)
		{
			(*puiCompareRules) |= FLM_COMP_IGNORE_TRAILING_SPACE;
		}
		else if (f_stricmp( SFLM_NOUNDERSCORE_OPTION_STR, pszRuleStart) == 0)
		{
			(*puiCompareRules) |= FLM_COMP_NO_UNDERSCORES;
		}
		else if (f_stricmp( SFLM_NO_WHITESPACE_OPTION_STR, pszRuleStart) == 0)
		{
			(*puiCompareRules) |= FLM_COMP_NO_WHITESPACE;
		}
		else if (f_stricmp( SFLM_NODASH_OPTION_STR, pszRuleStart) == 0)
		{
			(*puiCompareRules) |= FLM_COMP_NO_DASHES;
		}
		else
		{
			return( FALSE);
		}
	}
	return( TRUE);
}

/****************************************************************************
Desc:	Read in all rows from the index component table.  Create the in-memory
		structure for each index component.
****************************************************************************/
RCODE F_Db::dictReadIndexComponents( void)
{
	RCODE					rc = NE_SFLM_OK;
	F_Row *				pRow = NULL;
	FLMUINT64			ui64DefRowId;
	FSTableCursor *	pTableCursor = NULL;
	FLMUINT				uiColumnNum;
	FLMUINT				uiIndexNum;
	char					szTmp [200];
	FLMUINT				uiTmpLen;
	FLMBOOL				bTmp;
	FLMUINT				uiKeyComponent;
	FLMUINT				uiDataComponent;
	FLMUINT				uiFlags;
	FLMUINT				uiCompareRules;
	FLMUINT				uiLimit;
	FLMBOOL				bIsNull;

	if ((pTableCursor = f_new FSTableCursor) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = pTableCursor->setupRange( this, SFLM_TBLNUM_INDEX_COMPONENTS,
											1, FLM_MAX_UINT64, FALSE)))
	{
		goto Exit;
	}
	
	// Read through all rows in the table.  Each row defines an index component
	// for an index in the database, so we add it to the in-memory dictionary.
	
	for (;;)
	{
		if (RC_BAD( rc = pTableCursor->nextRow( this, &pRow, &ui64DefRowId)))
		{
			if (rc == NE_SFLM_EOF_HIT)
			{
				rc = NE_SFLM_OK;
			}
			goto Exit;
		}
		
		// Get the index number - required.
		
		if (RC_BAD( rc = pRow->getUINT( this, SFLM_COLNUM_INDEX_COMP_INDEX_NUM,
											&uiIndexNum, &bIsNull)))
		{
			goto Exit;
		}
		if (bIsNull)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_NULL_INDEX_NUM);
			goto Exit;
		}
		
		// Get the column number - required.
		
		if (RC_BAD( rc = pRow->getUINT( this, SFLM_COLNUM_INDEX_COMP_COLUMN_NUM,
											&uiColumnNum, &bIsNull)))
		{
			goto Exit;
		}
		if (bIsNull)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_NULL_COLUMN_NUM);
			goto Exit;
		}
		
		// Get the key component number - optional, defaults to zero.
		
		if (RC_BAD( rc = pRow->getUINT( this, SFLM_COLNUM_INDEX_COMP_KEY_COMPONENT,
											&uiKeyComponent, &bIsNull)))
		{
			goto Exit;
		}
		if (bIsNull)
		{
			uiKeyComponent = 0;
		}
		
		// Get the data component number - optional, defaults to zero.
		
		if (RC_BAD( rc = pRow->getUINT( this, SFLM_COLNUM_INDEX_COMP_DATA_COMPONENT,
											&uiDataComponent, &bIsNull)))
		{
			goto Exit;
		}
		if (bIsNull)
		{
			uiDataComponent = 0;
		}
		
		// We must have at least one of the components non-zero.
		
		if (!uiKeyComponent && !uiDataComponent)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_NO_COMP_NUM_FOR_INDEX_COLUMN);
			goto Exit;
		}
	
		uiFlags = 0;
		uiCompareRules = 0;
		uiLimit = 0;
		
		// For data components we just ignore the index-on, compare-rules,
		// sort-descending, sort-missing-high, and limit columns.
		
		if (uiKeyComponent)
		{
			
			// Get what we are indexing on - optional, defaults to value.
			
			if (RC_BAD( rc = pRow->getUTF8( this, SFLM_COLNUM_INDEX_COMP_INDEX_ON,
												szTmp, sizeof( szTmp),
												&bIsNull, &uiTmpLen, NULL)))
			{
				goto Exit;
			}
			if (bIsNull || !uiTmpLen)
			{
				uiFlags |= ICD_VALUE; 
			}
			else
			{
				if (!validIndexOnValue( szTmp, &uiFlags))
				{
					rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_INDEX_ON_VALUE);
					goto Exit;
				}
			}
			
			// Get compare rules - optional, defaults to none.
			
			if (RC_BAD( rc = pRow->getUTF8( this, SFLM_COLNUM_INDEX_COMP_COMPARE_RULES,
												szTmp, sizeof( szTmp), &bIsNull, &uiTmpLen, NULL)))
			{
				goto Exit;
			}
			if (!bIsNull && uiTmpLen)
			{
				if (!validCompareRulesValue( szTmp, &uiCompareRules))
				{
					rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_COMPARE_RULES_VALUE);
					goto Exit;
				}
			}
			
			// Get the sort-descending flag - optional, defaults to sort ascending.
			
			if (RC_BAD( rc = pRow->getUTF8( this, SFLM_COLNUM_INDEX_COMP_SORT_DESCENDING,
												szTmp, sizeof( szTmp), &bIsNull, &uiTmpLen, NULL)))
			{
				goto Exit;
			}
			if (!bIsNull && uiTmpLen)
			{
				if (!validBooleanValue( szTmp, &bTmp))
				{
					rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_SORT_DESCENDING_VALUE);
					goto Exit;
				}
				if (bTmp)
				{
					uiFlags |= ICD_DESCENDING;
				}
			}
		
			// Get the sort-missing-high flag - optional, defaults to sort missing low.
			
			if (RC_BAD( rc = pRow->getUTF8( this, SFLM_COLNUM_INDEX_COMP_SORT_MISSING_HIGH,
												szTmp, sizeof( szTmp), &bIsNull, &uiTmpLen, NULL)))
			{
				goto Exit;
			}
			if (!bIsNull && uiTmpLen)
			{
				if (!validBooleanValue( szTmp, &bTmp))
				{
					rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_SORT_MISSING_HIGH_VALUE);
					goto Exit;
				}
				if (bTmp)
				{
					uiFlags |= ICD_MISSING_HIGH;
				}
			}
			
			// Get the limit - defaults to different things depending on
			// what we are indexing.

			if (RC_BAD( rc = pRow->getUINT( this, SFLM_COLNUM_INDEX_COMP_LIMIT,
												&uiLimit, &bIsNull)))
			{
				goto Exit;
			}
			if (bIsNull)
			{
				uiLimit = (FLMUINT)((uiFlags & ICD_SUBSTRING)
										  ? (FLMUINT)ICD_DEFAULT_SUBSTRING_LIMIT
										  : (FLMUINT)ICD_DEFAULT_LIMIT);
			}
		}
			
		// Add the index component to the the in-memory dictionary.
		
		if (RC_BAD( rc = m_pDict->addIndexComponent( uiIndexNum, ui64DefRowId,
												uiColumnNum,
												uiFlags, uiCompareRules, uiLimit,
												uiKeyComponent, uiDataComponent)))
		{
			goto Exit;
		}
	}

Exit:

	if (pRow)
	{
		pRow->ReleaseRow();
	}

	if (pTableCursor)
	{
		pTableCursor->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Open a dictionary by reading in all of the dictionary tables
		from the dictionaries.
****************************************************************************/
RCODE F_Db::dictOpen( void)
{
	RCODE	rc = NE_SFLM_OK;

	// At this point, better not be pointing to a dictionary.

	flmAssert( !m_pDict);

	// Should never get here for a temporary database.

	flmAssert( !m_pDatabase->m_bTempDb);

	// Allocate a new F_Dict object for reading the dictionary
	// into memory.

	if ((m_pDict = f_new F_Dict) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}

	// Allocate the fixed collections and indexes and set them up

	if (RC_BAD( rc = m_pDict->setupPredefined()))
	{
		goto Exit;
	}

	// Read in the LFH's for the predefined stuff.

	if (RC_BAD( rc = dictReadLFH()))
	{
		goto Exit;
	}

	// If dictionary tables are not yet set up, do nothing.

	if (m_pDict->m_pTableTbl [SFLM_TBLNUM_TABLES - 1].lfInfo.uiBlkAddress)
	{

		// Read in definitions in the following order:
		// 1) encryption definitions
		// 2) table definitions
		// 3) column definitions
		// 4) index definitions
		// 5) index components
		// This guarantees that things will be defined by the
		// time they are referenced.

		if (RC_BAD( rc = dictReadEncDefs()))
		{
			goto Exit;
		}
		if (RC_BAD( rc = dictReadTables()))
		{
			goto Exit;
		}
		if (RC_BAD( rc = dictReadColumns()))
		{
			goto Exit;
		}
		if (RC_BAD( rc = dictReadIndexes()))
		{
			goto Exit;
		}
		if (RC_BAD( rc = dictReadIndexComponents()))
		{
			goto Exit;
		}

		// Must read LFHs to get the LFILE information for the
		// tables and indexes we have just added.

		if (RC_BAD( rc = dictReadLFH()))
		{
			goto Exit;
		}
	}
	
	// Verify the dictionary after it is all read into memory.
	
	if (RC_BAD( rc = m_pDict->verifyDict()))
	{
		goto Exit;
	}

Exit:

	if (RC_BAD( rc) && m_pDict)
	{
		m_pDict->Release();
		m_pDict = NULL;
	}

	return( rc);
}

/****************************************************************************
Desc:	Creates a new dictionary for a database.
		This occurs on database create and on a dictionary change.
****************************************************************************/
RCODE F_Db::createNewDict( void)
{
	RCODE	rc = NE_SFLM_OK;

	// Unlink the DB from the current F_Dict object, if any.

	if (m_pDict)
	{
		m_pDatabase->lockMutex();
		unlinkFromDict();
		m_pDatabase->unlockMutex();
	}

	// Allocate a new F_Dict object for the new dictionary we
	// are going to create.

	if (RC_BAD( rc = dictOpen()))
	{
		goto Exit;
	}

	// Update the F_Db flags to indicate that the dictionary
	// was updated.

	m_uiFlags |= FDB_UPDATED_DICTIONARY;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Create the tables and indexes needed for storing dictionary
		definitions.
****************************************************************************/
RCODE F_Db::dictCreate( void)
{
	RCODE rc = NE_SFLM_OK;
	LFILE	TempLFile;

	// This should never be called for a temporary database.

	flmAssert( !m_pDatabase->m_bTempDb);

	// Create the dictionary tables

	if (RC_BAD(rc = m_pDatabase->lFileCreate( this, &TempLFile,
							SFLM_TBLNUM_ENCDEFS, SFLM_LF_TABLE, FALSE, TRUE, 0)))
	{
		goto Exit;
	}
	if (RC_BAD(rc = m_pDatabase->lFileCreate( this, &TempLFile,
							SFLM_TBLNUM_TABLES, SFLM_LF_TABLE, FALSE, TRUE, 0)))
	{
		goto Exit;
	}
	if (RC_BAD(rc = m_pDatabase->lFileCreate( this, &TempLFile,
							SFLM_TBLNUM_COLUMNS, SFLM_LF_TABLE, FALSE, TRUE, 0)))
	{
		goto Exit;
	}
	if (RC_BAD(rc = m_pDatabase->lFileCreate( this, &TempLFile,
							SFLM_TBLNUM_INDEXES, SFLM_LF_TABLE, FALSE, TRUE, 0)))
	{
		goto Exit;
	}
	if (RC_BAD(rc = m_pDatabase->lFileCreate( this, &TempLFile,
							SFLM_TBLNUM_INDEX_COMPONENTS, SFLM_LF_TABLE, FALSE, TRUE, 0)))
	{
		goto Exit;
	}
	if (RC_BAD(rc = m_pDatabase->lFileCreate( this, &TempLFile,
							SFLM_TBLNUM_BLOCK_CHAINS, SFLM_LF_TABLE, FALSE, TRUE, 0)))
	{
		goto Exit;
	}
	
	// Create the dictionary indexes

	if (RC_BAD(rc = m_pDatabase->lFileCreate( this, &TempLFile,
							SFLM_IXNUM_ENCDEF_NAME, SFLM_LF_INDEX, FALSE, FALSE, 0)))
	{
		goto Exit;
	}
	if (RC_BAD(rc = m_pDatabase->lFileCreate( this, &TempLFile,
							SFLM_IXNUM_TABLE_NAME, SFLM_LF_INDEX, FALSE, FALSE, 0)))
	{
		goto Exit;
	}
	if (RC_BAD(rc = m_pDatabase->lFileCreate( this, &TempLFile,
							SFLM_IXNUM_TABLE_ENCDEF_NUM, SFLM_LF_INDEX, FALSE, FALSE, 0)))
	{
		goto Exit;
	}
	if (RC_BAD(rc = m_pDatabase->lFileCreate( this, &TempLFile,
							SFLM_IXNUM_COLUMN_TABLE_NUM, SFLM_LF_INDEX, FALSE, FALSE, 0)))
	{
		goto Exit;
	}
	if (RC_BAD(rc = m_pDatabase->lFileCreate( this, &TempLFile,
							SFLM_IXNUM_COLUMN_ENCDEF_NUM, SFLM_LF_INDEX, FALSE, FALSE, 0)))
	{
		goto Exit;
	}
	if (RC_BAD(rc = m_pDatabase->lFileCreate( this, &TempLFile,
							SFLM_IXNUM_INDEX_NAME, SFLM_LF_INDEX, FALSE, FALSE, 0)))
	{
		goto Exit;
	}
	if (RC_BAD(rc = m_pDatabase->lFileCreate( this, &TempLFile,
							SFLM_IXNUM_INDEX_TABLE_NUM, SFLM_LF_INDEX, FALSE, FALSE, 0)))
	{
		goto Exit;
	}
	if (RC_BAD(rc = m_pDatabase->lFileCreate( this, &TempLFile,
							SFLM_IXNUM_INDEX_ENCDEF_NUM, SFLM_LF_INDEX, FALSE, FALSE, 0)))
	{
		goto Exit;
	}
	if (RC_BAD(rc = m_pDatabase->lFileCreate( this, &TempLFile,
							SFLM_IXNUM_INDEX_COMP_INDEX_NUM, SFLM_LF_INDEX, FALSE, FALSE, 0)))
	{
		goto Exit;
	}

	// Create a new dictionary we can work with.

	if (RC_BAD( rc = createNewDict()))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Lookup a name in the name table
****************************************************************************/
NAME_INFO * F_NameTable::findName(
	const char *	pszName,
	FLMUINT *		puiInsertPos)
{
	NAME_INFO *		pNameInfo = NULL;
	const char *	pszTblName;
	FLMUINT			uiTblSize;
	FLMUINT			uiLow;
	FLMUINT			uiMid;
	FLMUINT			uiHigh;
	FLMINT			iCmp;

	if (!m_bTableSorted)
	{
		sortNames();
	}
	
	// Do binary search in the table

	if ((uiTblSize = m_uiNumNames) == 0)
	{
		if (puiInsertPos)
		{
			*puiInsertPos = 0;
		}
		goto Exit;
	}

	uiHigh = --uiTblSize;
	uiLow = 0;
	for (;;)
	{
		uiMid = (uiLow + uiHigh) / 2;

		pszTblName = m_pNames [uiMid].pszName;
		if ((iCmp = (FLMINT)f_strcmp( pszName, pszTblName)) == 0)
		{

			// Found Match

			pNameInfo = &m_pNames [uiMid];
			if (puiInsertPos)
			{
				*puiInsertPos = uiMid;
			}
			goto Exit;
		}

		// Check if we are done

		if (uiLow >= uiHigh)
		{

			// Done, item not found

			if (puiInsertPos)
			{
				*puiInsertPos = (iCmp < 0)
									 ? uiMid
									 : uiMid + 1;
			}
			goto Exit;
		}

		if (iCmp < 0)
		{
			if (uiMid == 0)
			{
				if (puiInsertPos)
				{
					*puiInsertPos = 0;
				}
				goto Exit;
			}
			uiHigh = uiMid - 1;
		}
		else
		{
			if (uiMid == uiTblSize)
			{
				if (puiInsertPos)
				{
					*puiInsertPos = uiMid + 1;
				}
				goto Exit;
			}
			uiLow = uiMid + 1;
		}
	}

Exit:

	return( pNameInfo);
}

/***************************************************************************
Desc:	Swap two entries in tag info table during sort.
*****************************************************************************/
FINLINE void nameInfoSwap(
	NAME_INFO *	pNameInfoTbl,
	FLMUINT		uiPos1,
	FLMUINT		uiPos2)
{
	NAME_INFO	tmpNameInfo;
	
	tmpNameInfo.pszName = pNameInfoTbl [uiPos1].pszName;
	tmpNameInfo.uiItemNum = pNameInfoTbl [uiPos1].uiItemNum;
	pNameInfoTbl [uiPos1].pszName = pNameInfoTbl [uiPos2].pszName;
	pNameInfoTbl [uiPos1].uiItemNum = pNameInfoTbl [uiPos2].uiItemNum;
	pNameInfoTbl [uiPos2].pszName = tmpNameInfo.pszName;
	pNameInfoTbl [uiPos2].uiItemNum = tmpNameInfo.uiItemNum;
}

/***************************************************************************
Desc:	Comparison function for sorting name table
****************************************************************************/
FINLINE FLMINT compareNameInfo(
	NAME_INFO *	pNameInfo1,
	NAME_INFO *	pNameInfo2,
	FLMBOOL *	pbDuplicateNames)
{
	FLMINT	iCmp = (FLMINT)f_strcmp( pNameInfo1->pszName, pNameInfo2->pszName);
	
	if (iCmp == 0)
	{
		*pbDuplicateNames = TRUE;
		return( 0);
	}
	return( (iCmp < 0) ? -1 : 1);
}

/***************************************************************************
Desc:	Sort a name table.
****************************************************************************/
FSTATIC void sortNameTbl(
	NAME_INFO *	pNameInfoTbl,
	FLMUINT		uiLowerBounds,
	FLMUINT		uiUpperBounds,
	FLMBOOL *	pbDuplicateNames)
{
	FLMUINT			uiLBPos;
	FLMUINT			uiUBPos;
	FLMUINT			uiMIDPos;
	FLMUINT			uiLeftItems;
	FLMUINT			uiRightItems;
	NAME_INFO *		pCurNameInfo;
	FLMINT			iCompare;

Iterate_Larger_Half:

	uiUBPos = uiUpperBounds;
	uiLBPos = uiLowerBounds;
	uiMIDPos = (uiUpperBounds + uiLowerBounds + 1) / 2;
	pCurNameInfo = &pNameInfoTbl [uiMIDPos];
	for (;;)
	{
		while (uiLBPos == uiMIDPos ||				// Don't compare with target
					((iCompare =
						compareNameInfo( &pNameInfoTbl [uiLBPos], pCurNameInfo,
												pbDuplicateNames)) < 0))
		{
			if (uiLBPos >= uiUpperBounds)
			{
				break;
			}
			uiLBPos++;
		}

		while (uiUBPos == uiMIDPos ||				// Don't compare with target
					(((iCompare =
						compareNameInfo( pCurNameInfo, &pNameInfoTbl [uiUBPos],
							pbDuplicateNames)) < 0)))
		{
			if (!uiUBPos)
			{
				break;
			}
			uiUBPos--;
		}

		if (uiLBPos < uiUBPos )			// Interchange and continue loop.
		{

			// Exchange [uiLBPos] with [uiUBPos].

			nameInfoSwap( pNameInfoTbl, uiLBPos, uiUBPos);
			uiLBPos++;						// Scan from left to right.
			uiUBPos--;						// Scan from right to left.
		}
		else									// Past each other - done
		{
			break;
		}
	}

	// Check for swap( LB, MID ) - cases 3 and 4

	if( uiLBPos < uiMIDPos )
	{

		// Exchange [uiLBPos] with [uiMIDPos]

		nameInfoSwap( pNameInfoTbl, uiMIDPos, uiLBPos);
		uiMIDPos = uiLBPos;
	}
	else if( uiMIDPos < uiUBPos )
	{

		// Exchange [uUBPos] with [uiMIDPos]

		nameInfoSwap( pNameInfoTbl, uiMIDPos, uiUBPos);
		uiMIDPos = uiUBPos;
	}

	// Check the left piece.

	uiLeftItems = (uiLowerBounds + 1 < uiMIDPos )
							? uiMIDPos - uiLowerBounds		// 2 or more
							: 0;
	uiRightItems = (uiMIDPos + 1 < uiUpperBounds )
							? uiUpperBounds - uiMIDPos 		// 2 or more
							: 0;

	if( uiLeftItems < uiRightItems )
	{

		// Recurse on the LEFT side and goto the top on the RIGHT side.

		if (uiLeftItems )
		{
			sortNameTbl( pNameInfoTbl, uiLowerBounds, uiMIDPos - 1, pbDuplicateNames);
		}
		uiLowerBounds = uiMIDPos + 1;
		goto Iterate_Larger_Half;
	}
	else if (uiLeftItems )	// Compute a truth table to figure out this check.
	{

		// Recurse on the RIGHT side and goto the top for the LEFT side.

		if (uiRightItems )
		{
			sortNameTbl( pNameInfoTbl, uiMIDPos + 1, uiUpperBounds, pbDuplicateNames);
		}
		uiUpperBounds = uiMIDPos - 1;
		goto Iterate_Larger_Half;
	}
}

/****************************************************************************
Desc:	Add a name to the table.
****************************************************************************/
RCODE F_NameTable::addName(
	const char *	pszName,
	FLMUINT			uiItemNum,
	FLMBOOL			bCheckDuplicates,
	RCODE				rcDuplicateError,
	FLMUINT			uiTableGrowSize)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiInsertPos;

	if (!pszName || !uiItemNum)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_PARM);
		goto Exit;
	}

	if (bCheckDuplicates)
	{

		// Make sure that the name is not already used.

		if (findName( pszName, &uiInsertPos))
		{
			rc = RC_SET( rcDuplicateError);
			goto Exit;
		}
	}
	else
	{
		uiInsertPos = m_uiNumNames;
		m_bTableSorted = FALSE;
	}
	
	// See if we need to grow the table.
	
	if (m_uiNumNames == m_uiTblSize)
	{
		FLMUINT	uiNewSize = m_uiTblSize + uiTableGrowSize;
		
		if (RC_BAD( rc = f_realloc( sizeof( NAME_INFO) * uiNewSize,
									&m_pNames)))
		{
			goto Exit;
		}
		m_uiTblSize = uiNewSize;
	}
	
	// If necessary, move names up in the table to make room for the
	// new name.
	
	if (uiInsertPos < m_uiNumNames)
	{
		f_memmove( &m_pNames [uiInsertPos + 1],
					  &m_pNames [uiInsertPos],
					  sizeof( NAME_INFO) * (m_uiNumNames - uiInsertPos));
	}
	m_pNames [uiInsertPos].pszName = pszName;
	m_pNames [uiInsertPos].uiItemNum = uiItemNum;
	m_uiNumNames++;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Copy a name into the name table.
****************************************************************************/
RCODE F_NameTable::copyName(
	const char *	pszName,
	FLMUINT			uiItemNum,
	const char **	ppszDestName,
	F_Pool *			pPool)
{
	RCODE		rc = NE_SFLM_OK;
	FLMUINT	uiNameLen;
	char *	pszDestName;
	
	uiNameLen = f_strlen( pszName) + 1;
	if (RC_BAD( rc = pPool->poolAlloc( uiNameLen, (void **)&pszDestName)))
	{
		goto Exit;
	}
	f_memcpy( pszDestName, pszName, uiNameLen);
	*ppszDestName = pszDestName;

	if (RC_BAD( rc = addName( pszDestName, uiItemNum,
								FALSE, NE_SFLM_OK, 0)))
	{
		goto Exit;
	}
Exit:
	return( rc);
}

/****************************************************************************
Desc:	Sort the name table.
****************************************************************************/
void F_NameTable::sortNames( void)
{
	if (!m_bTableSorted)
	{
		m_bDuplicateNames = FALSE;
		if (m_uiNumNames > 1)
		{
			sortNameTbl( m_pNames, 0, m_uiNumNames - 1, &m_bDuplicateNames);
		}
		m_bTableSorted = TRUE;
	}
}

/****************************************************************************
Desc:	Remove a name from the table
****************************************************************************/
void F_NameTable::removeName(
	const char *	pszName)
{
	FLMUINT	uiPos;

	if (findName( pszName, &uiPos) != NULL)
	{
		if (uiPos < m_uiNumNames - 1)
		{
			f_memmove( &m_pNames [uiPos], &m_pNames [uiPos + 1],
					sizeof( NAME_INFO) * (m_uiNumNames - uiPos - 1));
		}
		m_uiNumNames--;
	}
}

