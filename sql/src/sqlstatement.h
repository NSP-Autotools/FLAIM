//------------------------------------------------------------------------------
// Desc:	This file contains SQL statement class.
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

#ifndef SQLSTATEMENT_H
#define SQLSTATEMENT_H

// Maximum name length for table names, column names, and index names

#define MAX_SQL_NAME_LEN	128

class SQLQuery;

typedef struct SELECT_EXPR
{
	SQLQuery *		pSqlQuery;
	SELECT_EXPR *	pNext;
} SELECT_EXPR;

typedef struct COLUMN_SET
{
	FLMUINT			uiColumnNum;
	SQLQuery *		pSqlQuery;
	COLUMN_SET *	pNext;
} COLUMN_SET;

typedef struct TABLE_ITEM
{
	FLMUINT			uiTableNum;
	const char *	pszTableAlias;
	FLMUINT			uiIndexNum;
	FLMBOOL			bScan;
} TABLE_ITEM;

typedef enum
{
	SQL_PARSE_STATS
} eSQLStatus;

typedef RCODE (* SQL_STATUS_HOOK)(
	eSQLStatus		eStatusType,
	void *			pvArg1,
	void *			pvArg2,
	void *			pvArg3,
	void *			pvUserData);

//------------------------------------------------------------------------------
// Desc: Base object that defines methods which must be present for all ODBC
//			handle objects.
//------------------------------------------------------------------------------
class ODBCObject : public F_Object
{
public:

	ODBCObject()
	{
		m_hMutex = F_MUTEX_NULL;
		m_pszStateInfo = NULL;
		m_uiErrMsgLen = 0;
		m_bHaveError = FALSE;
		m_uiNumDiagRecs = 0;
	}
	
	virtual ~ODBCObject()
	{
		if (m_hMutex != F_MUTEX_NULL)
		{
			f_mutexDestroy( &m_hMutex);
		}
	}
	
	FINLINE FLMINT SQFAPI AddRef( void)
	{
		return( f_atomicInc( &m_refCnt));
	}

	FINLINE FLMINT SQFAPI Release( void)
	{
		FLMINT	iRefCnt;
		
		if ((iRefCnt = f_atomicDec( &m_refCnt)) == 0)
		{
			delete this;
		}
		return( iRefCnt);
	}

	FINLINE RCODE setupObject( void)
	{
		return( f_mutexCreate( &m_hMutex));
	}
	
	FINLINE void lockObject( void)
	{
		f_mutexLock( m_hMutex);
	}
	
	FINLINE void unlockObject( void)
	{
		f_mutexUnlock( m_hMutex);
	}
	
	FINLINE const char * getStateInfo( void)
	{
		return( m_pszStateInfo);
	}
	
	FINLINE FLMUINT getErrMsgLen( void)
	{
		return( m_uiErrMsgLen);
	}
	
	FINLINE const char * getErrMsg( void)
	{
		if (m_uiErrMsgLen)
		{
			return( m_szErrMsg [0]);
		}
		else
		{
			return( NULL);
		}
	}
	
	FINLINE RCODE getRCODE( void)
	{
		return( m_rc);
	}
	
	FINLINE FLMBOOL haveError( void)
	{
		return( m_bHaveError);
	}
	
	FINLINE void clearState( void)
	{
		m_bHaveError = FALSE;
		m_uiNumDiagRecs = 0;
	}
	
	FINLINE void setStateInfo(
		const char *	pszStateInfo)
	{
		m_pszStateInfo = pszStateInfo;
		m_uiErrMsgLen = 0;
		m_rc = NE_SFLM_OK;
		m_bHaveError = TRUE;
	}
	
	FINLINE void setGeneralErrMsg(
		const char *	pszErrMsg,
		RCODE				rc = NE_SFLM_OK)
	{
		m_pszStateInfo = "HY000";
		m_uiErrMsgLen = f_strlen( pszErrMsg);
		
		// Copy null terminator character too.
		
		f_memcpy( m_szErrMsg, pszErrMsg, m_uiErrMsgLen + 1);
		m_rc = rc;
		m_bHaveError = TRUE;
	}

	// Must be implemented by inheriting class.
	
	virtual FLMBOOL canRelease( void) = 0;
	
private:

	F_MUTEX			m_hMutex;
	const char *		m_pszStateInfo;
	char *			m_szErrMsg [200];
	FLMUINT			m_uiErrMsgLen;
	RCODE				m_rc;
	FLMUINT			m_bHaveError;
	FLMUINT			m_uiNumDiagRecs;
};

/*============================================================================
Desc:	SQL statement class.  Parses and executes SQL statements.  This object
		type is returned for ODBC for handles of type SQL_HANDLE_STMT or
		SQLHSTMT.
============================================================================*/
class SQLStatement : public ODBCObject
{
public:

#define MAX_SQL_TOKEN_SIZE		80

	SQLStatement();

	virtual ~SQLStatement();
	
	RCODE setupStatement( void);

	void resetStatement( void);

	RCODE executeSQL(
		IF_IStream *	pStream,
		F_Db *			pDb,
		SQL_STATS *		pSQLStats);
		
	FINLINE FLMBOOL canRelease( void)
	{
		// VISIT: Need to determine whether or not this is possible.
		
		return( TRUE);
	}

	SQLConnection * getConnection( void)
	{
		return( m_pConnection);
	}

private:

	// Methods

	RCODE getByte(
		FLMBYTE *	pucByte);
		
	FINLINE void ungetByte(
		FLMBYTE	ucByte)
	{
		// Can only unget a single byte.
		
		flmAssert( !m_ucUngetByte);
		m_ucUngetByte = ucByte;
		m_sqlStats.uiChars--;
	}
		
	RCODE getLine( void);
	
	FINLINE FLMBYTE getChar( void)
	{
		if (m_uiCurrLineOffset == m_uiCurrLineBytes)
		{
			return( (FLMBYTE)0);
		}
		else
		{
			FLMBYTE	ucChar = m_pucCurrLineBuf [m_uiCurrLineOffset++];
			return( ucChar);
		}
	}
	
	FINLINE FLMBYTE peekChar( void)
	{
		if (m_uiCurrLineOffset == m_uiCurrLineBytes)
		{
			return( (FLMBYTE)0);
		}
		else
		{
			return( m_pucCurrLineBuf [m_uiCurrLineOffset]);
		}
	}
	
	FINLINE void ungetChar( void)
	{
		
		// There should never be a reason to unget past the beginning of the current
		// line.
		
		flmAssert( m_uiCurrLineOffset);
		m_uiCurrLineOffset--;
	}

	RCODE skipWhitespace(
		FLMBOOL	bRequired);

	RCODE haveToken(
		const char *	pszToken,
		FLMBOOL			bEofOK,
		SQLParseError	eNotHaveErr = SQL_NO_ERROR);

	RCODE getToken(
		char *		pszToken,
		FLMUINT		uiTokenBufSize,
		FLMBOOL		bEofOK,
		FLMUINT *	puiTokenLineOffset,
		FLMUINT *	puiTokenLen);
		
	FINLINE void setErrInfo(
		FLMUINT			uiErrLineNum,
		FLMUINT			uiErrLineOffset,
		SQLParseError	eErrorType,
		FLMUINT			uiErrLineFilePos,
		FLMUINT			uiErrLineBytes)
	{
		m_sqlStats.uiErrLineNum = uiErrLineNum;
		m_sqlStats.uiErrLineOffset = uiErrLineOffset;
		m_sqlStats.eErrorType = eErrorType;
		m_sqlStats.uiErrLineFilePos = uiErrLineFilePos;
		m_sqlStats.uiErrLineBytes = uiErrLineBytes;
	}

	RCODE getBinaryValue(
		F_DynaBuf *	pDynaBuf);
		
	RCODE getUTF8String(
		FLMBOOL		bMustHaveEqual,
		FLMBOOL		bStripWildcardEscapes,
		FLMBYTE *	pszStr,
		FLMUINT		uiStrBufSize,
		FLMUINT *	puiStrLen,
		FLMUINT *	puiNumChars,
		F_DynaBuf *	pDynaBuf);
		
	RCODE getNumber(
		FLMBOOL		bMustHaveEqual,
		FLMUINT64 *	pui64Num,
		FLMBOOL *	pbNeg,
		FLMBOOL		bNegAllowed);
		
	RCODE getBool(
		FLMBOOL		bMustHaveEqual,
		FLMBOOL *	pbBool);
		
	RCODE getUINT(
		FLMBOOL		bMustHaveEqual,
		FLMUINT *	puiNum);
		
	RCODE getName(
		char *		pszName,
		FLMUINT		uiNameBufSize,
		FLMUINT *	puiNameLen,
		FLMUINT *	puiTokenLineOffset);
		
	RCODE getEncDefName(
		FLMBOOL		bMustExist,
		char *		pszEncDefName,
		FLMUINT		uiEncDefNameBufSize,
		FLMUINT *	puiEncDefNameLen,
		F_ENCDEF **	ppEncDef);

	RCODE getTableName(
		FLMBOOL		bMustExist,
		char *		pszTableName,
		FLMUINT		uiTableNameBufSize,
		FLMUINT *	puiTableNameLen,
		F_TABLE **	ppTable);

	RCODE getIndexName(
		FLMBOOL		bMustExist,
		F_TABLE *	pTable,
		char *		pszIndexName,
		FLMUINT		uiIndexNameBufSize,
		FLMUINT *	puiIndexNameLen,
		F_INDEX **	ppIndex);
		
	RCODE getStringValue(
		F_COLUMN *			pColumn,
		F_COLUMN_VALUE *	pColumnValue);

	RCODE getNumberValue(
		F_COLUMN_VALUE *	pColumnValue);

	RCODE getBinaryValue(
		F_COLUMN *			pColumn,
		F_COLUMN_VALUE *	pColumnValue);
		
	RCODE getValue(
		F_COLUMN *			pColumn,
		F_COLUMN_VALUE *	pColumnValue);
		
	RCODE insertRow( void);

	RCODE processCreateDatabase( void);
	
	RCODE processOpenDatabase( void);
	
	RCODE processDropDatabase( void);
	
	RCODE getDataType(
		eDataType *	peDataType,
		FLMUINT *	puiMax,
		FLMUINT *	puiEncDefNum,
		FLMUINT *	puiFlags);
		
	RCODE processCreateTable( void);
	
	RCODE processDropTable( void);
	
	RCODE processCreateIndex(
		FLMBOOL	bUnique);
	
	RCODE processDropIndex( void);
	
	RCODE processInsertRow( void);
	
	RCODE parseSetColumns(
		TABLE_ITEM *	pTableList,
		COLUMN_SET **	ppFirstColumnSet,
		COLUMN_SET **	ppLastColumnSet,
		FLMUINT *		puiNumColumnsToSet,
		FLMBOOL *		pbHadWhere);
		
	RCODE processUpdateRows( void);
	
	RCODE processDeleteRows( void);
	
	RCODE processAlphaToken(
		TABLE_ITEM *	pTableList,
		const char **	ppszTerminatingTokens,
		const char **	ppszTerminator,
		SQLQuery *		pSqlQuery,
		FLMBOOL *		pbDone);
		
	RCODE parseCriteria(
		TABLE_ITEM *	pTableList,
		const char **	ppszTerminatingTokens,
		FLMBOOL			bEofOK,
		const char **	ppszTerminator,
		SQLQuery *		pSqlQuery);
		
	RCODE parseSelectExpressions(
		SELECT_EXPR **	ppFirstSelectExpr,
		SELECT_EXPR **	ppLastSelectExpr);
		
	RCODE processSelect( void);
	
	// Data

	F_Db *						m_pDb;
	IF_XML *						m_pXml;
	FLMBYTE						m_ucUngetByte;
	FLMBYTE *					m_pucCurrLineBuf;
	FLMUINT						m_uiCurrLineBufMaxBytes;
	FLMUINT						m_uiCurrLineOffset;
	FLMUINT						m_uiCurrLineNum;
	FLMUINT						m_uiCurrLineFilePos;
	FLMUINT						m_uiCurrLineBytes;
	IF_IStream *				m_pStream;
	FLMUINT						m_uiFlags;
	SQL_STATUS_HOOK			m_fnStatus;
	void *						m_pvCallbackData;
	SQL_STATS					m_sqlStats;
	F_Pool						m_tmpPool;
	SQLConnection *			m_pConnection;
	SQLStatement *				m_pNextInConnection;
	SQLStatement *				m_pPrevInConnection;

friend class F_Db;
friend class F_Database;
friend class SQLConnection;
friend class SQLEnv;
friend class SQLDesc;
};

RCODE resolveColumnName(
	F_Db *				pDb,
	TABLE_ITEM *		pTableList,
	const char *		pszTableAlias,
	const char *		pszColumnName,
	FLMUINT *			puiTableNum,
	FLMUINT *			puiColumnNum,
	SQLParseError *	peParseError);
	
#endif // SQLSTATEMENT_H
