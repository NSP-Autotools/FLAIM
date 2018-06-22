//------------------------------------------------------------------------------
// Desc:	F_Dict class definitions - internal object for database's
//			dictionary.
// Tabs:	3
//
// Copyright (c) 2002-2007 Novell, Inc. All Rights Reserved.
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

#ifndef FDICT_H
#define FDICT_H

struct ICD;
struct F_INDEX;
class F_Dict;
class F_NameTable;
class F_Database;
class F_CCS;

/****************************************************************************
Desc:	Encryption definition
****************************************************************************/
typedef struct F_ENCDEF
{
	FLMUINT			uiEncDefNum;		// Encryption definition number.  This is
												// also the place in the encryption definition
												// table minus one.
	FLMUINT64		ui64DefRowId;		// Definition row ID.
	const char *	pszEncDefName;		// Encryption definition name.
	eEncAlgorithm	eEncAlg;				// AES or DES3
	FLMUINT			uiEncKeySize;		// Key size
	F_CCS *			pCcs;					// Encryption object
} F_ENCDEF;

/****************************************************************************
Desc:	Table column definition.
****************************************************************************/
typedef struct F_COLUMN
{
	FLMUINT			uiColumnNum;		// Column number.  This is also the place in
												// the table's column table minus one.
	FLMUINT64		ui64DefRowId;		// Definition row ID.
	const char *	pszColumnName;		// Column name.
	FLMUINT			uiFlags;				// Column flags
#define COL_READ_ONLY		0x0001
#define COL_NULL_ALLOWED	0x0002
	eDataType		eDataTyp;			// Column data type
	FLMUINT			uiMaxLen;			// Maximum length (for strings or binary)
	FLMUINT			uiEncDefNum;		// If column is encrypted, this is the
												// encryption definition number.  Zero if
												// column is not encrypted.
	ICD *				pFirstIcd;			// First index component where this column
												// is a key component.
	ICD *				pFirstDataIcd;		// First index component where this column
												// is a data component.
} F_COLUMN;

/****************************************************************************
Desc:	Logical File
****************************************************************************/
typedef struct LFILE
{
	FLMUINT		uiLfNum;				// Index number or table number.
	FLMUINT		uiEncDefNum;		// If index or table is encrypted, this is
											// the table's encryption definition number.
											// Zero if the index or table is not encrypted.
	eLFileType	eLfType; 			// Type of logical file
	FLMUINT   	uiRootBlk;			// Address of root block.
	FLMUINT		uiBlkAddress;		// Block address of LFile entry.
	FLMUINT		uiOffsetInBlk;		// Offset within block of entry.
	FLMUINT64	ui64NextRowId;		// Next row ID in table, if eLfType is a table
											// Not used for indexes.
	FLMBOOL		bNeedToWriteOut;	// If TRUE, this LFILE needs to be written
											// to disk.  NOTE: This is only used for
											// tables - so that we can update ui64NextRowId
											// many times in memory, but only write it out
											// at commit time.
} LFILE;

/*****************************************************************************
Desc:	Table definition
*****************************************************************************/
typedef struct F_TABLE
{
	FLMUINT			uiTableNum;			// Table number.  This is also this entry's place
												// in the table table, minus one.
	FLMUINT64		ui64DefRowId;		// Definition row ID.
	const char *	pszTableName;		// Table name.
	FLMBOOL			bSystemTable;		// System table - cannot be dropped or altered by
												// an application.
	FLMUINT			uiNumColumns;		// Number of columns in the table.
	FLMUINT			uiNumReqColumns;	// Number of columns that are required to
												// have values.
	F_COLUMN *		pColumns;			// Contains all columns for the table.
	F_NameTable *	pColumnNames;		// Contains all column names for the table
	LFILE				lfInfo;				// Logical file information.
	FLMUINT			uiFirstIndexNum;	// Number of first index on this table
} F_TABLE;

#define MAX_INDEX_COLUMNS		((SFLM_MAX_KEY_SIZE - FLM_MAX_NUM_BUF_SIZE) / 2)
#define MAX_ORDER_BY_COLUMNS	MAX_INDEX_COLUMNS
#define MAX_SELECT_TABLES	50

/*****************************************************************************
Desc:	Index definition
*****************************************************************************/
typedef struct F_INDEX
{
	FLMUINT			uiIndexNum;					// Index number.  This also corresponds
														// to the position of this entry in
														// the index table minus one.
	FLMUINT64		ui64DefRowId;				// Definition row ID.
	FLMUINT			uiNextIndexNum;			// Next index on this same table.														
	FLMUINT			uiTableNum;					// Table this index is on.
	const char *	pszIndexName;				// Index name.
	ICD *				pKeyIcds;					// Array of key components for the index.
	FLMUINT			uiNumKeyComponents;		// Number of key components.
	ICD *				pDataIcds;					// Array of data components for the index.
	FLMUINT			uiNumDataComponents;		// Number of data components.
	FLMUINT			uiFlags;						// Index flags.
		#define IXD_ABS_POS				0x00001	// Maintain absolute positioning info.
		#define IXD_HAS_SUBSTRING		0x00002	// At least one key component is substring.
		#define IXD_OFFLINE				0x00004	// Index is offline - may or may
															// not be suspended.
		#define IXD_SUSPENDED			0x00008	// IXD_OFFLINE should also be set
		#define IXD_SYSTEM				0x00010	// Index is an internal system index
		#define IXD_KEYS_UNIQUE			0x00020	// Index keys must be unique
	FLMUINT			uiLanguage;					// Language for the index.
	LFILE				lfInfo;						// Logical file information.
	FLMUINT64		ui64LastRowIndexed;		// Last row indexed, if indexing in
														// the background.
} F_INDEX;

/*****************************************************************************
Desc:	Index component
*****************************************************************************/
typedef struct ICD
{
	FLMUINT		uiIndexNum;			// Index this component belongs to
	FLMUINT		uiColumnNum;		// Column in table this component refers to
	FLMUINT64	ui64DefRowId;		// Definition row ID.
	FLMUINT		uiFlags;				// Flags for component
	FLMUINT		uiCompareRules;	// Comparison rules for this component.
	FLMUINT		uiLimit;				// Limit for this component.
#define ICD_DEFAULT_LIMIT					128
#define ICD_DEFAULT_SUBSTRING_LIMIT		48
	ICD *			pNextInChain;
	ICD *			pNextInDataChain;
} ICD;

/**************************************************************************
Desc:	Structure kept in name table for sorting names.
**************************************************************************/
typedef struct NAME_INFO
{
	const char *	pszName;
	FLMUINT			uiItemNum;
} NAME_INFO;

/**************************************************************************
Desc:	This class is the name table class.
**************************************************************************/
class F_NameTable : public F_Object
{
public:

	// Constructor and destructor

	F_NameTable()
	{
		m_pNames = NULL;
		m_uiTblSize = 0;
		m_uiNumNames = 0;
		m_bTableSorted = FALSE;
		m_bDuplicateNames = FALSE;
	}

	~F_NameTable()
	{
		f_free( &m_pNames);
	}
	
	NAME_INFO * findName(
		const char *	pszName,
		FLMUINT *		puiInsertPos);

	RCODE addName(
		const char *	pszName,
		FLMUINT			uiItemNum,
		FLMBOOL			bCheckDuplicates,
		RCODE				rcDuplicateError,
		FLMUINT			uiTableGrowSize);

	RCODE copyName(
		const char *	pszName,
		FLMUINT			uiItemNum,
		const char **	ppszDestName,
		F_Pool *			pPool);
		
	void sortNames( void);
	
	void removeName(
		const char *	pszName);
		
private:

	NAME_INFO *		m_pNames;
	FLMUINT			m_uiTblSize;
	FLMUINT			m_uiNumNames;
	FLMBOOL			m_bTableSorted;
	FLMBOOL			m_bDuplicateNames;

friend class F_Db;
friend class F_Dict;
};

/**************************************************************************
Desc:	This class is the FLAIM dictionary class.
**************************************************************************/
class F_Dict : public F_Object
{
public:

	// Constructor and destructor

	F_Dict();

	~F_Dict();
	
	void resetDict( void);

	RCODE getTable(
		const char *	pszTableName,
		F_TABLE **		ppTable,
		FLMBOOL			bOfflineOk);
		
	RCODE getIndex(
		const char *	pszIndexName,
		F_INDEX **		ppIndex,
		FLMBOOL			bOfflineOk);
		
	FINLINE F_ENCDEF * getEncDef(
		FLMUINT	uiEncDefNum)
	{
		return( (F_ENCDEF *)((uiEncDefNum && uiEncDefNum <= m_uiHighestEncDefNum &&
									 m_pEncDefTbl [uiEncDefNum - 1].uiEncDefNum)
									? &m_pEncDefTbl [uiEncDefNum - 1]
									: (F_ENCDEF *)NULL));
	}

	FINLINE F_TABLE * getTable(
		FLMUINT	uiTableNum)
	{
		return( (F_TABLE *)((uiTableNum && uiTableNum <= m_uiHighestTableNum &&
									m_pTableTbl [uiTableNum - 1].uiTableNum)
									? &m_pTableTbl [uiTableNum - 1]
									: (F_TABLE *)NULL));
	}
	
	FINLINE F_INDEX * getIndex(
		FLMUINT	uiIndexNum)
	{
		return( (F_INDEX *)((uiIndexNum && uiIndexNum <= m_uiHighestIndexNum &&
									m_pIndexTbl [uiIndexNum - 1].uiIndexNum)
									? &m_pIndexTbl [uiIndexNum - 1]
									: (F_INDEX *)NULL));
	}
	
	FINLINE F_COLUMN * getColumn(
		F_TABLE *	pTable,
		FLMUINT		uiColumnNum)
	{
		return( (F_COLUMN *)((uiColumnNum && uiColumnNum <= pTable->uiNumColumns)
									? &pTable->pColumns [uiColumnNum - 1]
									: (F_COLUMN *)NULL));
	}
	
	FINLINE F_TABLE * findTable(
		const char *	pszTableName)
	{
		NAME_INFO *	pNameInfo;
		
		if (m_pTableNames)
		{
			if ((pNameInfo = m_pTableNames->findName( pszTableName, NULL)) != NULL)
			{
				return( getTable( pNameInfo->uiItemNum));
			}
		}
		return( NULL);
	}
	
	FINLINE F_COLUMN * findColumn(
		F_TABLE *		pTable,
		const char *	pszColumnName)
	{
		NAME_INFO *	pNameInfo;
		
		if (pTable->pColumnNames)
		{
			if ((pNameInfo = pTable->pColumnNames->findName( pszColumnName,
													NULL)) != NULL)
			{
				return( getColumn( pTable, pNameInfo->uiItemNum));
			}
		}
		return( NULL);
	}
	
	FINLINE F_INDEX * findIndex(
		const char *	pszIndexName)
	{
		NAME_INFO *	pNameInfo;
		
		if (m_pIndexNames)
		{
			if ((pNameInfo = m_pIndexNames->findName( pszIndexName, NULL)) != NULL)
			{
				return( getIndex( pNameInfo->uiItemNum));
			}
		}
		return( NULL);
	}
	
	FINLINE F_ENCDEF * findEncDef(
		const char *	pszEncDefName)
	{
		NAME_INFO *	pNameInfo;
		
		if (m_pEncDefNames)
		{
			if ((pNameInfo = m_pEncDefNames->findName( pszEncDefName, NULL)) != NULL)
			{
				return( getEncDef( pNameInfo->uiItemNum));
			}
		}
		return( NULL);
	}
	
	void linkToDatabase(
		F_Database *	pDatabase);

	void unlinkFromDatabase( void);

	FINLINE FLMUINT getUseCount( void)
	{
		return( m_uiUseCount);
	}

	FINLINE FLMUINT decrUseCount( void)
	{
		return( --m_uiUseCount);
	}

	FINLINE void incrUseCount( void)
	{
		m_uiUseCount++;
	}

	FINLINE F_Dict * getPrev( void)
	{
		return( m_pPrev);
	}

	FINLINE F_Dict * getNext( void)
	{
		return( m_pNext);
	}

	FINLINE F_Database * getDatabase( void)
	{
		return( m_pDatabase);
	}

	RCODE copyEncDef(
		F_ENCDEF *	pDestEncDef,
		F_ENCDEF *	pSrcEncDef);
		
	RCODE copyColumn(
		F_NameTable *	pDestColumnNameTable,
		F_COLUMN *		pDestColumn,
		F_COLUMN *		pSrcColumn);
		
	RCODE copyTable(
		F_TABLE *		pDestTable,
		F_TABLE *		pSrcTable);
		
	RCODE copyIndex(
		F_INDEX *	pDestIndex,
		F_INDEX *	pSrcIndex);
		
	RCODE cloneDict(
		F_Dict *	pSrcDict);
		
	RCODE addEncDef(
		FLMUINT			uiEncDefNum,
		FLMUINT64		ui64DefRowId,
		const char *	pszEncDefName,
		eEncAlgorithm	eEncAlg,
		FLMUINT			uiEncKeySize,
		FLMBYTE *		pucEncKey,
		FLMUINT			uiEncKeyLen);
		
	RCODE addTable(
		FLMUINT			uiTableNum,
		FLMUINT64		ui64DefRowId,
		const char *	pszTableName,
		FLMBOOL			bSystemTable,
		FLMUINT			uiNumColumns,
		FLMUINT			uiEncDefNum);
		
	RCODE addColumn(
		FLMUINT			uiTableNum,
		FLMUINT64		ui64DefRowId,
		FLMUINT			uiColumnNum,
		const char *	pszColumnName,
		FLMUINT			uiFlags,
		eDataType		eDataTyp,
		FLMUINT			uiMaxLen,
		FLMUINT			uiEncDefNum);
		
	RCODE addIndex(
		FLMUINT			uiIndexNum,
		FLMUINT64		ui64DefRowId,
		const char *	pszIndexName,
		FLMUINT			uiTableNum,
		FLMUINT			uiEncDefNum,
		FLMUINT			uiFlags,
		FLMUINT			uiNumKeyComponents,
		FLMUINT			uiNumDataComponents,
		FLMUINT			uiLanguage,
		FLMUINT64		ui64LastRowIndexed);
		
	RCODE addIndexComponent(
		FLMUINT			uiIndexNum,
		FLMUINT64		ui64DefRowId,
		FLMUINT			uiColumnNum,
		FLMUINT			uiFlags,
		FLMUINT			uiCompareRules,
		FLMUINT			uiLimit,
		FLMUINT			uiKeyComponent,
		FLMUINT			uiDataComponent);
		
	RCODE setupEncDefTable( void);
	
	RCODE setupTableTable( void);
	
	RCODE setupColumnTable( void);
	
	RCODE setupIndexTable( void);
	
	RCODE setupIndexComponentTable( void);
	
	RCODE setupBlockChainTable( void);
	
	RCODE setupPredefined( void);
	
	RCODE verifyDict( void);
	
private:

	F_Dict *					m_pNext;			// Pointer to next F_Dict object in the list,
													// if any.  All versions of a dictionary that
													// are currently in use are linked together.
													// Usually, there will be only one local
													// dictionary in the list.
	F_Dict *					m_pPrev;			// Previous F_Dict object in the list.
	F_Database *			m_pDatabase;	// database this dictionary is associated with.
													// A null value means it is not yet linked
													// to a database.
	FLMUINT					m_uiDictSeq;	// This is the sequence number of the
													// dictionary
	F_Pool  					m_dictPool;		// Pool for all allocations except tables.
	
	// Encryption definition table
	
	F_ENCDEF *				m_pEncDefTbl;
	FLMUINT					m_uiEncDefTblSize;
	FLMUINT					m_uiHighestEncDefNum;
	F_NameTable *			m_pEncDefNames;
	
	// Table table
	
	F_TABLE *				m_pTableTbl;
	FLMUINT					m_uiTableTblSize;
	FLMUINT					m_uiHighestTableNum;
	F_NameTable *			m_pTableNames;

	// Index table
	
	F_INDEX *				m_pIndexTbl;
	FLMUINT					m_uiIndexTblSize;
	FLMUINT					m_uiHighestIndexNum;
	F_NameTable *			m_pIndexNames;
	
	FLMUINT					m_uiUseCount;	// Number of F_Db structures currently
													// pointing to this dictionary.

friend class F_Database;
friend class F_Db;
friend class F_Query;
friend class F_DbCheck;
friend class F_BTreeInfo;
};

#endif // #ifndef FDICT_H
