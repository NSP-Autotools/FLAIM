//------------------------------------------------------------------------------
// Desc:	This module contains routines that will create an index.
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

//------------------------------------------------------------------------------
// Desc:	Create a new index in the database.  Caller should have already
//			verified that the index name is unique and that the table number and
//			column numbers are valid.  This routine will assign an index number.
//------------------------------------------------------------------------------
RCODE F_Db::createIndex(
	FLMUINT				uiTableNum,
	FLMUINT				uiIndexNum,
	const char *		pszIndexName,
	FLMUINT				uiIndexNameLen,
	FLMUINT				uiEncDefNum,
	FLMUINT				uiFlags,
	F_INDEX_COL_DEF *	pIxColDefs,
	FLMUINT				uiNumIxColDefs)
{
	RCODE					rc = NE_SFLM_OK;
	F_Row *				pRow = NULL;
	F_TABLE *			pTable;
	F_COLUMN *			pColumn;
	F_INDEX_COL_DEF *	pIxColDef;
	char					szLanguage [10];
	FLMUINT				uiLanguageLen;
	FLMUINT				uiKeyComponent;
	const char *		pszIndexOn;
	FLMUINT				uiIndexOnLen;
	F_INDEX *			pIndex;
	FLMBOOL				bStartedTrans = FALSE;
	
	// Make sure we are in an update transaction.
	
	if (RC_BAD( rc = checkTransaction( SFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}
	
	// Create a new dictionary, if we don't already have one.
	
	if (!(m_uiFlags & FDB_UPDATED_DICTIONARY))
	{
		if (RC_BAD( rc = dictClone()))
		{
			goto Exit;
		}
	}
	
	// Create a row for the table in the table definition table.
	
	if (RC_BAD( rc = gv_SFlmSysData.pRowCacheMgr->createRow( this,
								SFLM_TBLNUM_INDEXES, &pRow)))
	{
		goto Exit;
	}
	
	// Determine the index number to use - find lowest non-used index
	// number.
	
	if (!uiIndexNum)
	{
		uiIndexNum = 1;
		while (uiIndexNum <= m_pDict->m_uiHighestIndexNum)
		{
			if (!m_pDict->m_pIndexTbl [uiIndexNum - 1].uiIndexNum)
			{
				break;
			}
			uiIndexNum++;
		}
	}
	
	// The call to addIndex will initialize either the empty slot we found, or
	// the next slot at the end of the index table.  It will reallocate the
	// index array if necessary.
	
	// Remove the IXD_HAS_SUBSTRING flag - will reset later based on the
	// column definitions for the index.  Also remove the IXD_SYSTEM flag if
	// it was set.
	
	uiFlags &= (~(IXD_HAS_SUBSTRING | IXD_SYSTEM));
	if (RC_BAD( rc = m_pDict->addIndex( uiIndexNum, pRow->m_ui64RowId,
										pszIndexName, uiTableNum, uiEncDefNum, uiFlags,
										uiNumIxColDefs, 0,
										m_pDatabase->m_uiDefaultLanguage, 0)))
	{
		goto Exit;
	}
	
	pTable = m_pDict->getTable( uiTableNum);
	pIndex = m_pDict->getIndex( uiIndexNum);
	
	// Populate the columns for the row in the table definition table.
	
	if (RC_BAD( rc = pRow->setUTF8( this, SFLM_COLNUM_INDEXES_INDEX_NAME,
								pszIndexName, uiIndexNameLen, uiIndexNameLen)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pRow->setUINT( this, SFLM_COLNUM_INDEXES_INDEX_NUM,
								uiIndexNum)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pRow->setUINT( this, SFLM_COLNUM_INDEXES_TABLE_NUM,
								uiTableNum)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pRow->setUINT( this, SFLM_COLNUM_INDEXES_ENCDEF_NUM,
								uiEncDefNum)))
	{
		goto Exit;
	}
	f_languageToStr( pIndex->uiLanguage, szLanguage);
	uiLanguageLen = f_strlen( szLanguage);
	if (RC_BAD( rc = pRow->setUTF8( this, SFLM_COLNUM_INDEXES_LANGUAGE,
								szLanguage, uiLanguageLen, uiLanguageLen)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pRow->setUINT( this, SFLM_COLNUM_INDEXES_NUM_KEY_COMPONENTS,
								uiNumIxColDefs)))
	{
		goto Exit;
	}
	if (uiFlags & IXD_ABS_POS)
	{
		if (RC_BAD( rc = pRow->setUTF8( this, SFLM_COLNUM_INDEXES_KEEP_ABS_POS_INFO,
									"yes", 3, 3)))
		{
			goto Exit;
		}
	}
	if (uiFlags & IXD_KEYS_UNIQUE)
	{
		if (RC_BAD( rc = pRow->setUTF8( this, SFLM_COLNUM_INDEXES_KEYS_UNIQUE,
									"yes", 3, 3)))
		{
			goto Exit;
		}
	}
	if (uiFlags & IXD_SUSPENDED)
	{
		if (RC_BAD( rc = pRow->setUTF8( this, SFLM_COLNUM_INDEXES_INDEX_STATE,
									SFLM_INDEX_SUSPENDED_STR,
									SFLM_INDEX_SUSPENDED_STR_LEN,
									SFLM_INDEX_SUSPENDED_STR_LEN)))
		{
			goto Exit;
		}
	}
	else if (uiFlags & IXD_OFFLINE)
	{
		if (RC_BAD( rc = pRow->setUTF8( this, SFLM_COLNUM_INDEXES_INDEX_STATE,
									SFLM_INDEX_OFFLINE_STR,
									SFLM_INDEX_OFFLINE_STR_LEN,
									SFLM_INDEX_OFFLINE_STR_LEN)))
		{
			goto Exit;
		}
	}
	
	// Add all of the index columns to the index column table.
	
	for (pIxColDef = pIxColDefs, uiKeyComponent = 1;
		  pIxColDef;
		  uiKeyComponent++, pIxColDef = pIxColDef->pNext)
	{
		
		// Create a row for the column in the column definition table.
		
		if (pRow)
		{
			pRow->ReleaseRow();
			pRow = NULL;
		}
		if (RC_BAD( rc = gv_SFlmSysData.pRowCacheMgr->createRow( this,
									SFLM_TBLNUM_INDEX_COMPONENTS, &pRow)))
		{
			goto Exit;
		}
		
		// Verify that the column number is valid.  Caller is responsible
		// to pass in valid column numbers.
		
		pColumn = m_pDict->getColumn( pTable, pIxColDef->uiColumnNum);
		flmAssert( pColumn);
		
		// Add the index component to the in-memory dictionary structures.
		
		if (RC_BAD( rc = m_pDict->addIndexComponent( uiIndexNum, pRow->m_ui64RowId,
										pIxColDef->uiColumnNum, pIxColDef->uiFlags,
										pIxColDef->uiCompareRules,
										pIxColDef->uiLimit, uiKeyComponent, 0)))
		{
			goto Exit;
		}

		// Populate the columns for the row in the column definition table.
		
		if (RC_BAD( rc = pRow->setUINT( this, SFLM_COLNUM_INDEX_COMP_INDEX_NUM,
											uiIndexNum)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = pRow->setUINT( this, SFLM_COLNUM_INDEX_COMP_COLUMN_NUM,
											pIxColDef->uiColumnNum)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = pRow->setUINT( this, SFLM_COLNUM_INDEX_COMP_KEY_COMPONENT,
											uiKeyComponent)))
		{
			goto Exit;
		}
		if (pIxColDef->uiFlags & ICD_VALUE)
		{
			pszIndexOn = SFLM_VALUE_OPTION_STR;
			uiIndexOnLen = SFLM_VALUE_OPTION_STR_LEN;
		}
		else if (pIxColDef->uiFlags & ICD_EACHWORD)
		{
			pszIndexOn = SFLM_EACHWORD_OPTION_STR;
			uiIndexOnLen = SFLM_EACHWORD_OPTION_STR_LEN;
		}
		else if (pIxColDef->uiFlags & ICD_PRESENCE)
		{
			pszIndexOn = SFLM_PRESENCE_OPTION_STR;
			uiIndexOnLen = SFLM_PRESENCE_OPTION_STR_LEN;
		}
		else if (pIxColDef->uiFlags & ICD_METAPHONE)
		{
			pszIndexOn = SFLM_METAPHONE_OPTION_STR;
			uiIndexOnLen = SFLM_METAPHONE_OPTION_STR_LEN;
		}
		else
		{
			
			// Assert here, because it has to be one of these five options to
			// index on.
			
			flmAssert( pIxColDef->uiFlags & ICD_SUBSTRING);
			pszIndexOn = SFLM_SUBSTRING_OPTION_STR;
			uiIndexOnLen = SFLM_SUBSTRING_OPTION_STR_LEN;
		}
		if (RC_BAD( rc = pRow->setUTF8( this, SFLM_COLNUM_INDEX_COMP_INDEX_ON,
											pszIndexOn, uiIndexOnLen, uiIndexOnLen)))
		{
			goto Exit;
		}
		
		if (pIxColDef->uiCompareRules)
		{
			char		szCompareRules [200];
			char *	pszCompareRules = &szCompareRules [0];
			FLMUINT	uiCompareRulesLen;
			
			if (pIxColDef->uiCompareRules & FLM_COMP_CASE_INSENSITIVE)
			{
				f_memcpy( pszCompareRules, SFLM_CASE_INSENSITIVE_OPTION_STR,
					SFLM_CASE_INSENSITIVE_OPTION_STR_LEN);
				pszCompareRules += SFLM_CASE_INSENSITIVE_OPTION_STR_LEN;
				*pszCompareRules++ = ' ';
			}
			if (pIxColDef->uiCompareRules & FLM_COMP_COMPRESS_WHITESPACE)
			{
				f_memcpy( pszCompareRules, SFLM_COMPRESS_WHITESPACE_OPTION_STR,
					SFLM_COMPRESS_WHITESPACE_OPTION_STR_LEN);
				pszCompareRules += SFLM_COMPRESS_WHITESPACE_OPTION_STR_LEN;
				*pszCompareRules++ = ' ';
			}
			if (pIxColDef->uiCompareRules & FLM_COMP_NO_WHITESPACE)
			{
				f_memcpy( pszCompareRules, SFLM_NO_WHITESPACE_OPTION_STR,
					SFLM_NO_WHITESPACE_OPTION_STR_LEN);
				pszCompareRules += SFLM_NO_WHITESPACE_OPTION_STR_LEN;
				*pszCompareRules++ = ' ';
			}
			if (pIxColDef->uiCompareRules & FLM_COMP_NO_UNDERSCORES)
			{
				f_memcpy( pszCompareRules, SFLM_NOUNDERSCORE_OPTION_STR,
					SFLM_NOUNDERSCORE_OPTION_STR_LEN);
				pszCompareRules += SFLM_NOUNDERSCORE_OPTION_STR_LEN;
				*pszCompareRules++ = ' ';
			}
			if (pIxColDef->uiCompareRules & FLM_COMP_NO_DASHES)
			{
				f_memcpy( pszCompareRules, SFLM_NODASH_OPTION_STR,
					SFLM_NODASH_OPTION_STR_LEN);
				pszCompareRules += SFLM_NODASH_OPTION_STR_LEN;
				*pszCompareRules++ = ' ';
			}
			if (pIxColDef->uiCompareRules & FLM_COMP_WHITESPACE_AS_SPACE)
			{
				f_memcpy( pszCompareRules, SFLM_WHITESPACE_AS_SPACE_STR,
					SFLM_WHITESPACE_AS_SPACE_STR_LEN);
				pszCompareRules += SFLM_WHITESPACE_AS_SPACE_STR_LEN;
				*pszCompareRules++ = ' ';
			}
			if (pIxColDef->uiCompareRules & FLM_COMP_IGNORE_LEADING_SPACE)
			{
				f_memcpy( pszCompareRules, SFLM_IGNORE_LEADINGSPACES_OPTION_STR,
					SFLM_IGNORE_LEADINGSPACES_OPTION_STR_LEN);
				pszCompareRules += SFLM_IGNORE_LEADINGSPACES_OPTION_STR_LEN;
				*pszCompareRules++ = ' ';
			}
			if (pIxColDef->uiCompareRules & FLM_COMP_IGNORE_TRAILING_SPACE)
			{
				f_memcpy( pszCompareRules, SFLM_IGNORE_TRAILINGSPACES_OPTION_STR,
					SFLM_IGNORE_TRAILINGSPACES_OPTION_STR_LEN);
				pszCompareRules += SFLM_IGNORE_TRAILINGSPACES_OPTION_STR_LEN;
				*pszCompareRules++ = ' ';
			}
			
			// See if anything was set - extraneous bits are ignored.
			
			if (pszCompareRules != &szCompareRules [0])
			{
				
				// Get rid of the trailing space - every option added a trailing
				// space to separate it from the next option, if any.
				
				pszCompareRules--;
				*pszCompareRules = 0;
				uiCompareRulesLen = (FLMUINT)(pszCompareRules - &szCompareRules [0]);
				if (RC_BAD( rc = pRow->setUTF8( this, SFLM_COLNUM_INDEX_COMP_COMPARE_RULES,
													szCompareRules,
													uiCompareRulesLen, uiCompareRulesLen)))
				{
					goto Exit;
				}
			}
		}
	
		if (pIxColDef->uiFlags & ICD_DESCENDING)
		{
			if (RC_BAD( rc = pRow->setUTF8( this, SFLM_COLNUM_INDEX_COMP_SORT_DESCENDING,
												"yes", 3, 3)))
			{
				goto Exit;
			}
		}
		if (pIxColDef->uiFlags & ICD_MISSING_HIGH)
		{
			if (RC_BAD( rc = pRow->setUTF8( this, SFLM_COLNUM_INDEX_COMP_SORT_MISSING_HIGH,
												"yes", 3, 3)))
			{
				goto Exit;
			}
		}
		if (pIxColDef->uiLimit)
		{
			if (RC_BAD( rc = pRow->setUINT( this, SFLM_COLNUM_INDEX_COMP_LIMIT,
												pIxColDef->uiLimit)))
			{
				goto Exit;
			}
		}
	}
	
	if (RC_BAD( rc = m_pDatabase->m_pRfl->logCreateIndex( this, uiTableNum,
										uiIndexNum, pszIndexName, uiIndexNameLen,
										uiEncDefNum, pIxColDefs, uiFlags)))
	{
		goto Exit;
	}
	
	// Build the index - but only if this is not a replay of the
	// roll-forward log.  If we are replaying the roll-forward log,
	// it will have packets for building the index.

	if (!(uiFlags & IXD_SUSPENDED) && !(m_uiFlags & FDB_REPLAYING_RFL))
	{
		// Unique indexes cannot be built off-line, because we need to make
		// sure there are no non-unique keys that will get built.
		
		if ((uiFlags & IXD_OFFLINE) && !(uiFlags & IXD_KEYS_UNIQUE))
		{
			if (RC_BAD( rc = addToStartList( uiIndexNum)))
			{
				goto Exit;
			}
		}
		else
		{
			
			// There may be "new" rows in the row cache.
			// Need to flush them to the database so that
			// the B-Tree lookups done by the indexing code will
			// work correctly
		
			if( RC_BAD( rc = flushDirtyRows()))
			{
				goto Exit;
			}
		
			// Build index in foreground.
		
			if (RC_BAD( rc = indexSetOfRows( uiIndexNum, 1,
					FLM_MAX_UINT64, m_pIxStatus, m_pIxClient, NULL, NULL, NULL)))
			{
				goto Exit;
			}
		}
	}
	
	// Commit the transaction if we started it
	
	if (bStartedTrans)
	{
		bStartedTrans = FALSE;
		if (RC_BAD( rc = transCommit()))
		{
			goto Exit;
		}
	}

Exit:

	if (bStartedTrans)
	{
		transAbort();
	}

	if (pRow)
	{
		pRow->ReleaseRow();
	}

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Process the create index statement.  The "CREATE INDEX" keywords
//			or "CREATE UNIQUE INDEX" keywords have already been parsed.
//------------------------------------------------------------------------------
RCODE SQLStatement::processCreateIndex(
	FLMBOOL	bUnique)
{
	RCODE						rc = NE_SFLM_OK;
	FLMBOOL					bStartedTrans = FALSE;
	char						szIndexName [MAX_SQL_NAME_LEN + 1];
	FLMUINT					uiIndexNameLen;
	char						szTableName [MAX_SQL_NAME_LEN + 1];
	FLMUINT					uiTableNameLen;
	char						szColumnName [MAX_SQL_NAME_LEN + 1];
	FLMUINT					uiColumnNameLen;
	char						szEncDefName [MAX_SQL_NAME_LEN + 1];
	FLMUINT					uiEncDefNameLen;
	F_INDEX_COL_DEF *		pIxColDef;
	F_INDEX_COL_DEF *		pFirstIxColDef;
	F_INDEX_COL_DEF *		pLastIxColDef;
	FLMUINT					uiNumIxColumns;
	F_TABLE *				pTable;
	F_INDEX *				pIndex;
	F_ENCDEF *				pEncDef;
	FLMUINT					uiEncDefNum;
	F_COLUMN *				pColumn;
	FLMBOOL					bDone;
	FLMUINT					uiFlags;
	char						szToken [MAX_SQL_TOKEN_SIZE + 1];
	FLMUINT					uiTokenLineOffset;
	
	// If we are in a read transaction, we cannot do this operation
	
	if (RC_BAD( rc = m_pDb->checkTransaction( SFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// SYNTAX: CREATE [UNIQUE] INDEX <indexname> ON <tablename>
	//		(<column_name> [DESC], ...)
	//		[ABSPOS] [OFFLINE] [SUSPENDED] [ENCRYPT_WITH <EncDefName>]
	
	if (bUnique)
	{
		uiFlags = IXD_KEYS_UNIQUE;
	}
	else
	{
		uiFlags = 0;
	}

	// Whitespace must follow the "CREATE [UNIQUE] INDEX"

	if (RC_BAD( rc = skipWhitespace( TRUE)))
	{
		goto Exit;
	}

	// Get the index name - index name must NOT already exist
	
	if (RC_BAD( rc = getIndexName( FALSE, NULL, szIndexName, sizeof( szIndexName),
							&uiIndexNameLen, &pIndex)))
	{
		goto Exit;
	}
	
	// Whitespace must follow index name.
	
	if (RC_BAD( rc = skipWhitespace( TRUE)))
	{
		goto Exit;
	}
	
	// The keyword "ON" must follow
	
	if (RC_BAD( rc = haveToken( "on", FALSE, SQL_ERR_EXPECTING_ON)))
	{
		goto Exit;
	}

	// Get the table name - must exist.
	
	if (RC_BAD( rc = getTableName( TRUE, szTableName, sizeof( szTableName),
								&uiTableNameLen, &pTable)))
	{
		goto Exit;
	}
	
	// Left paren must follow table name
	
	if (RC_BAD( rc = haveToken( "(", FALSE, SQL_ERR_EXPECTING_LPAREN)))
	{
		goto Exit;
	}
	
	// Get the columns to be indexed.

	pFirstIxColDef = NULL;
	pLastIxColDef = NULL;
	uiNumIxColumns = 0;
	for (;;)
	{
		
		// Get the column name
		
		if (RC_BAD( rc = getName( szColumnName, sizeof( szColumnName),
										&uiColumnNameLen, &uiTokenLineOffset)))
		{
			goto Exit;
		}
		flmAssert( uiColumnNameLen);
		
		// Make sure it is a valid column for the table.

		if ((pColumn = m_pDb->m_pDict->findColumn( pTable, szColumnName)) == NULL)
		{
			setErrInfo( m_uiCurrLineNum,
					uiTokenLineOffset,
					SQL_ERR_UNDEFINED_COLUMN,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
			goto Exit;
		}
		
		// See if too many columns have been specified.
		
		if (uiNumIxColumns == MAX_INDEX_COLUMNS)
		{
			setErrInfo( m_uiCurrLineNum,
					uiTokenLineOffset,
					SQL_ERR_TOO_MANY_INDEX_COLUMNS,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
			goto Exit;
		}
		
		// Allocate an index column def structure
		
		if (RC_BAD( rc = m_tmpPool.poolAlloc( sizeof( F_INDEX_COL_DEF),
								(void **)&pIxColDef)))
		{
			goto Exit;
		}
		uiNumIxColumns++;
		
		pIxColDef->uiColumnNum = pColumn->uiColumnNum;
		pIxColDef->uiFlags = 0;
		pIxColDef->uiCompareRules = 0;
		pIxColDef->uiLimit = 0;
		pIxColDef->pNext = NULL;
		if (pLastIxColDef)
		{
			pLastIxColDef->pNext = pIxColDef;
		}
		else
		{
			pFirstIxColDef = pIxColDef;
		}
		pLastIxColDef = pIxColDef;
		
		// Get all of the index options and comparison rules.
		
		bDone = FALSE;
		for (;;)
		{
			if (RC_BAD( rc = getToken( szToken, sizeof( szToken), FALSE,
											&uiTokenLineOffset, NULL)))
			{
				goto Exit;
			}
			if (f_stricmp( szToken, ",") == 0)
			{
				break;
			}
			else if (f_stricmp( szToken, ")") == 0)
			{
				bDone = TRUE;
				break;
			}
			else if (f_stricmp( szToken, SFLM_VALUE_OPTION_STR) == 0)
			{
				if ((pIxColDef->uiFlags &
					 (ICD_VALUE | ICD_EACHWORD | ICD_PRESENCE | ICD_METAPHONE | ICD_SUBSTRING)))
				{
Multiple_Ix_Options:
					setErrInfo( m_uiCurrLineNum,
							m_uiCurrLineOffset - 1,
							SQL_ERR_MULTIPLE_INDEX_OPTIONS,
							m_uiCurrLineFilePos,
							m_uiCurrLineBytes);
					rc = RC_SET( NE_SFLM_INVALID_SQL);
					goto Exit;
				}
				pIxColDef->uiFlags |= ICD_VALUE;
			}
			else if (f_stricmp( szToken, SFLM_EACHWORD_OPTION_STR) == 0)
			{
				if ((pIxColDef->uiFlags &
					 (ICD_VALUE | ICD_EACHWORD | ICD_PRESENCE | ICD_METAPHONE | ICD_SUBSTRING)))
				{
					goto Multiple_Ix_Options;
				}
				if (pColumn->eDataTyp != SFLM_STRING_TYPE)
				{
Invalid_Ix_Option:
					setErrInfo( m_uiCurrLineNum,
							m_uiCurrLineOffset - 1,
							SQL_ERR_INVALID_COL_INDEX_OPTION,
							m_uiCurrLineFilePos,
							m_uiCurrLineBytes);
					rc = RC_SET( NE_SFLM_INVALID_SQL);
					goto Exit;
				}
				pIxColDef->uiFlags |= ICD_EACHWORD;
			}
			else if (f_stricmp( szToken, SFLM_PRESENCE_OPTION_STR) == 0)
			{
				if ((pIxColDef->uiFlags &
					 (ICD_VALUE | ICD_EACHWORD | ICD_PRESENCE | ICD_METAPHONE | ICD_SUBSTRING)))
				{
					goto Multiple_Ix_Options;
				}
				pIxColDef->uiFlags |= ICD_PRESENCE;
			}
			else if (f_stricmp( szToken, SFLM_METAPHONE_OPTION_STR) == 0)
			{
				if ((pIxColDef->uiFlags &
					 (ICD_VALUE | ICD_EACHWORD | ICD_PRESENCE | ICD_METAPHONE | ICD_SUBSTRING)))
				{
					goto Multiple_Ix_Options;
				}
				if (pColumn->eDataTyp != SFLM_STRING_TYPE)
				{
					goto Invalid_Ix_Option;
				}
				pIxColDef->uiFlags |= ICD_METAPHONE;
			}
			else if (f_stricmp( szToken, SFLM_SUBSTRING_OPTION_STR) == 0)
			{
				if ((pIxColDef->uiFlags &
					 (ICD_VALUE | ICD_EACHWORD | ICD_PRESENCE | ICD_METAPHONE | ICD_SUBSTRING)))
				{
					goto Multiple_Ix_Options;
				}
				if (pColumn->eDataTyp != SFLM_STRING_TYPE)
				{
					goto Invalid_Ix_Option;
				}
				pIxColDef->uiFlags |= ICD_SUBSTRING;
			}
			else if (f_stricmp( szToken, SFLM_DESCENDING_OPTION_STR) == 0 ||
						f_stricmp( szToken, "desc") == 0)
			{
				pIxColDef->uiFlags |= ICD_DESCENDING;
			}
			else if (f_stricmp( szToken, SFLM_ASCENDING_OPTION_STR) == 0 ||
						f_stricmp( szToken, "asc") == 0)
			{
				pIxColDef->uiFlags &= (~(ICD_DESCENDING));
			}
			else if (f_stricmp( szToken, SFLM_SORT_MISSING_HIGH_OPTION_STR) == 0)
			{
				pIxColDef->uiFlags |= ICD_MISSING_HIGH;
			}
			else if (f_stricmp( szToken, SFLM_SORT_MISSING_LOW_OPTION_STR) == 0)
			{
				pIxColDef->uiFlags &= (~(ICD_MISSING_HIGH));
			}
			else if (f_stricmp( szToken, SFLM_CASE_INSENSITIVE_OPTION_STR) == 0)
			{
				if (pColumn->eDataTyp != SFLM_STRING_TYPE)
				{
					goto Invalid_Ix_Option;
				}
				pIxColDef->uiCompareRules |= FLM_COMP_CASE_INSENSITIVE;
			}
			else if (f_stricmp( szToken, SFLM_COMPRESS_WHITESPACE_OPTION_STR) == 0)
			{
				if (pColumn->eDataTyp != SFLM_STRING_TYPE)
				{
					goto Invalid_Ix_Option;
				}
				pIxColDef->uiCompareRules |= FLM_COMP_COMPRESS_WHITESPACE;
			}
			else if (f_stricmp( szToken, SFLM_NO_WHITESPACE_OPTION_STR) == 0)
			{
				if (pColumn->eDataTyp != SFLM_STRING_TYPE)
				{
					goto Invalid_Ix_Option;
				}
				pIxColDef->uiCompareRules |= FLM_COMP_NO_WHITESPACE;
			}
			else if (f_stricmp( szToken, SFLM_NOUNDERSCORE_OPTION_STR) == 0)
			{
				if (pColumn->eDataTyp != SFLM_STRING_TYPE)
				{
					goto Invalid_Ix_Option;
				}
				pIxColDef->uiCompareRules |= FLM_COMP_NO_UNDERSCORES;
			}
			else if (f_stricmp( szToken, SFLM_NODASH_OPTION_STR) == 0)
			{
				if (pColumn->eDataTyp != SFLM_STRING_TYPE)
				{
					goto Invalid_Ix_Option;
				}
				pIxColDef->uiCompareRules |= FLM_COMP_NO_DASHES;
			}
			else if (f_stricmp( szToken, SFLM_WHITESPACE_AS_SPACE_STR) == 0)
			{
				if (pColumn->eDataTyp != SFLM_STRING_TYPE)
				{
					goto Invalid_Ix_Option;
				}
				pIxColDef->uiCompareRules |= FLM_COMP_WHITESPACE_AS_SPACE;
			}
			else if (f_stricmp( szToken, SFLM_IGNORE_LEADINGSPACES_OPTION_STR) == 0)
			{
				if (pColumn->eDataTyp != SFLM_STRING_TYPE)
				{
					goto Invalid_Ix_Option;
				}
				pIxColDef->uiCompareRules |= FLM_COMP_IGNORE_LEADING_SPACE;
			}
			else if (f_stricmp( szToken, SFLM_IGNORE_TRAILINGSPACES_OPTION_STR) == 0)
			{
				if (pColumn->eDataTyp != SFLM_STRING_TYPE)
				{
					goto Invalid_Ix_Option;
				}
				pIxColDef->uiCompareRules |= FLM_COMP_IGNORE_TRAILING_SPACE;
			}
			else if (f_stricmp( szToken, SFLM_LIMIT_OPTION_STR) == 0)
			{
				if (pColumn->eDataTyp != SFLM_STRING_TYPE &&
					 pColumn->eDataTyp != SFLM_BINARY_TYPE)
				{
					goto Invalid_Ix_Option;
				}
				if (RC_BAD( rc = getUINT( TRUE, &pIxColDef->uiLimit)))
				{
					goto Exit;
				}
			}
			else
			{
				setErrInfo( m_uiCurrLineNum,
						uiTokenLineOffset,
						SQL_ERR_INVALID_INDEX_OPTION,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
		}
		if (!(pIxColDef->uiFlags &
			(ICD_VALUE | ICD_EACHWORD | ICD_PRESENCE | ICD_METAPHONE | ICD_SUBSTRING)))
		{
			pIxColDef->uiFlags |= ICD_VALUE;
		}
		if (!pIxColDef->uiLimit)
		{
			if (pIxColDef->uiFlags & ICD_SUBSTRING)
			{
				pIxColDef->uiLimit = ICD_DEFAULT_SUBSTRING_LIMIT;
			}
			else
			{
				pIxColDef->uiLimit = ICD_DEFAULT_LIMIT;
			}
		}
		
		if (bDone)
		{
			break;
		}
	}
	
	// See if there are any options for the index itself
	
	uiEncDefNum = 0;
	for (;;)
	{
		if (RC_BAD( rc = getToken( szToken, sizeof( szToken), TRUE,
									&uiTokenLineOffset, NULL)))
		{
			if (rc == NE_SFLM_EOF_HIT)
			{
				rc = NE_SFLM_OK;
				break;
			}
			else
			{
				goto Exit;
			}
		}
		
		if (f_stricmp( szToken, SFLM_INDEX_SUSPENDED_STR) == 0)
		{
			
			// Ignore offline option if suspended was specified.
			
			if (!(uiFlags & IXD_SUSPENDED))
			{
				uiFlags |= IXD_OFFLINE;
			}
		}
		else if (f_stricmp( szToken, SFLM_INDEX_SUSPENDED_STR) == 0)
		{
			uiFlags &= (~(IXD_OFFLINE));
			uiFlags |= IXD_SUSPENDED;
		}
		else if (f_stricmp( szToken, SFLM_ABS_POS_OPTION_STR) == 0)
		{
			uiFlags |= IXD_ABS_POS;
		}
		else if (f_stricmp( szToken, SFLM_ENCRYPT_WITH_STR) == 0)
		{
			if (RC_BAD( rc = getEncDefName( TRUE, szEncDefName, sizeof( szEncDefName),
										&uiEncDefNameLen, &pEncDef)))
			{
				goto Exit;
			}
			uiEncDefNum = pEncDef->uiEncDefNum;
		}
		else
		{
			
			// Move the line offset back to the beginning of the token
			// so it can be processed by the next SQL statement in the
			// stream.
			
			m_uiCurrLineOffset = uiTokenLineOffset;
			break;
		}
	}
	
	// Create the index.

	if (RC_BAD( rc = m_pDb->createIndex( pTable->uiTableNum, 0,
										szIndexName, uiIndexNameLen, uiEncDefNum,
										uiFlags, pFirstIxColDef, uiNumIxColumns)))
	{
		goto Exit;
	}

	// Commit the transaction if we started it
	
	if (bStartedTrans)
	{
		bStartedTrans = FALSE;
		if (RC_BAD( rc = m_pDb->transCommit()))
		{
			goto Exit;
		}
	}

Exit:

	if (bStartedTrans)
	{
		m_pDb->transAbort();
	}

	return( rc);
}

