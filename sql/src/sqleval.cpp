//-------------------------------------------------------------------------
// Desc:	Parse SQL
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

FSTATIC RCODE sqlCompareText(
	SQL_VALUE *		pLValue,
	SQL_VALUE *		pRValue,
	FLMUINT			uiCompareRules,
	FLMBOOL			bOpIsMatch,
	FLMUINT			uiLanguage,
	FLMINT *			piResult);
	
FSTATIC RCODE sqlApproxCompare(
	SQL_VALUE *		pLValue,
	SQL_VALUE *		pRValue,
	FLMINT *			piResult);
	
FSTATIC RCODE sqlCompareBinary(
	SQL_VALUE *		pLValue,
	SQL_VALUE *		pRValue,
	FLMINT *			piResult);
	
FSTATIC void sqlArithOpUUBitAND(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpUUBitOR(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpUUBitXOR(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpUUMult(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpUSMult(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpSSMult(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpSUMult(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpUUDiv(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpUSDiv(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpSSDiv(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpSUDiv(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpUUMod(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpUSMod(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpSSMod(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpSUMod(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpUUPlus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpUSPlus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpSSPlus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpSUPlus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpUUMinus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpUSMinus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpSSMinus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpSUMinus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC RCODE sqlCompareOperands(
	FLMUINT					uiLanguage,
	SQL_VALUE *				pLValue,
	SQL_VALUE *				pRValue,
	eSQLQueryOperators	eOperator,
	FLMUINT					uiCompareRules,
	FLMBOOL					bNotted,
	SQLBoolType *			peBool);
	
FSTATIC RCODE sqlGetColumnValue(
	F_Db *		pDb,
	F_Row *		pRow,
	FLMUINT		uiTableNum,
	FLMUINT		uiColumnNum,
	F_Pool *		pPool,
	SQL_VALUE *	pSqlValue);
	
FSTATIC RCODE sqlEvalOperator(
	FLMUINT		uiLanguage,
	SQL_NODE *	pQNode);
	
typedef void SQL_ARITH_OP(
	SQL_VALUE *		pLValue,
	SQL_VALUE *		pRValue,
	SQL_VALUE *		pResult);

SQL_ARITH_OP * SQL_ArithOpTable[ 
	((SQL_LAST_ARITH_OP - SQL_FIRST_ARITH_OP) + 1) * 4 ] =
{
/*	U = Unsigned		S = Signed
					U + U					U + S
						S + U					S + S */
/* BITAND */	sqlArithOpUUBitAND,		sqlArithOpUUBitAND,
						sqlArithOpUUBitAND,		sqlArithOpUUBitAND,
/* BITOR  */	sqlArithOpUUBitOR,		sqlArithOpUUBitOR,
						sqlArithOpUUBitOR,		sqlArithOpUUBitOR,
/* BITXOR */	sqlArithOpUUBitXOR,		sqlArithOpUUBitXOR,
						sqlArithOpUUBitXOR,		sqlArithOpUUBitXOR,
/* MULT   */	sqlArithOpUUMult,			sqlArithOpUSMult,
						sqlArithOpSUMult,			sqlArithOpSSMult,
/* DIV    */	sqlArithOpUUDiv,			sqlArithOpUSDiv,
						sqlArithOpSUDiv,			sqlArithOpSSDiv,
/* MOD    */	sqlArithOpUUMod,			sqlArithOpUSMod,
						sqlArithOpSUMod,			sqlArithOpSSMod,
/* PLUS   */	sqlArithOpUUPlus,			sqlArithOpUSPlus,
						sqlArithOpSUPlus,			sqlArithOpSSPlus,
/* MINUS  */	sqlArithOpUUMinus,		sqlArithOpUSMinus,
						sqlArithOpSUMinus,		sqlArithOpSSMinus
};

//-------------------------------------------------------------------------
// Desc:	Compare two entire strings.
//-------------------------------------------------------------------------
FSTATIC RCODE sqlCompareText(
	SQL_VALUE *		pLValue,
	SQL_VALUE *		pRValue,
	FLMUINT			uiCompareRules,
	FLMBOOL			bOpIsMatch,
	FLMUINT			uiLanguage,
	FLMINT *			piResult)
{
	RCODE					rc = NE_SFLM_OK;
	F_BufferIStream	bufferLStream;
	IF_PosIStream *	pLStream;
	F_BufferIStream	bufferRStream;
	IF_PosIStream *	pRStream;

	// Types must be text

	if (pLValue->eValType != SQL_UTF8_VAL || pRValue->eValType != SQL_UTF8_VAL)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_Q_COMPARE_OPERAND_TYPE_MISMATCH);
		goto Exit;
	}

	// Open the streams

	if (RC_BAD( rc = bufferLStream.openStream( 
		(const char *)pLValue->val.str.pszStr, pLValue->val.str.uiByteLen)))
	{
		goto Exit;
	}

	pLStream = &bufferLStream;

	if( RC_BAD( rc = bufferRStream.openStream( 
		(const char *)pRValue->val.str.pszStr, pRValue->val.str.uiByteLen)))
	{
		goto Exit;
	}
	pRStream = &bufferRStream;

	if( RC_BAD( rc = f_compareUTF8Streams( 
		pLStream, 
		(bOpIsMatch && (pLValue->uiFlags & SQL_VAL_IS_CONSTANT)) ? TRUE : FALSE,
		pRStream,
		(bOpIsMatch && (pRValue->uiFlags & SQL_VAL_IS_CONSTANT)) ? TRUE : FALSE,
		uiCompareRules, uiLanguage, piResult)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Performs approximate compare on two strings.
//-------------------------------------------------------------------------
FSTATIC RCODE sqlApproxCompare(
	SQL_VALUE *				pLValue,
	SQL_VALUE *				pRValue,
	FLMINT *					piResult)
{
	RCODE						rc = NE_SFLM_OK;
	FLMUINT					uiLMeta;
	FLMUINT					uiRMeta;
	FLMUINT64				ui64StartPos;
	F_BufferIStream		bufferLStream;
	IF_PosIStream *		pLStream;
	F_BufferIStream		bufferRStream;
	IF_PosIStream *		pRStream;

	// Types must be text

	if (pLValue->eValType != SQL_UTF8_VAL ||
		 pRValue->eValType != SQL_UTF8_VAL)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_NOT_IMPLEMENTED);
		goto Exit;
	}

	// Open the streams

	if (RC_BAD( rc = bufferLStream.openStream( 
		(const char *)pLValue->val.str.pszStr, pLValue->val.str.uiByteLen)))
	{
		goto Exit;
	}
	pLStream = &bufferLStream;

	if( RC_BAD( rc = bufferRStream.openStream( 
		(const char *)pRValue->val.str.pszStr, pRValue->val.str.uiByteLen)))
	{
		goto Exit;
	}
	pRStream = &bufferRStream;
	
	if ((pLValue->uiFlags & SQL_VAL_IS_CONSTANT) ||
		 !(pRValue->uiFlags & SQL_VAL_IS_CONSTANT))
	{
		for( ;;)
		{
			if (RC_BAD( rc = f_getNextMetaphone( pLStream, &uiLMeta)))
			{
				if( rc == NE_FLM_EOF_HIT)
				{
					*piResult = 0;
					rc = NE_SFLM_OK;
				}
				goto Exit;
			}

			ui64StartPos = pRStream->getCurrPosition();

			for( ;;)
			{
				if( RC_BAD( rc = f_getNextMetaphone( pRStream, &uiRMeta)))
				{
					if( rc == NE_FLM_EOF_HIT)
					{
						rc = NE_SFLM_OK;
						*piResult = -1;
					}

					goto Exit;
				}

				if( uiLMeta == uiRMeta)
				{
					break;
				}

			}

			if( RC_BAD( rc = pRStream->positionTo( ui64StartPos)))
			{
				goto Exit;
			}
		}
	}
	else
	{
		for( ;;)
		{
			if( RC_BAD( rc = f_getNextMetaphone( pRStream, &uiRMeta)))
			{
				if( rc == NE_FLM_EOF_HIT)
				{
					*piResult = 0;
					rc = NE_SFLM_OK;
				}
				goto Exit;
			}

			ui64StartPos = pLStream->getCurrPosition();

			for( ;;)
			{
				if( RC_BAD( rc = f_getNextMetaphone( pLStream, &uiLMeta)))
				{
					if( rc == NE_FLM_EOF_HIT)
					{
						rc = NE_SFLM_OK;
						*piResult = 1;
					}

					goto Exit;
				}

				if( uiLMeta == uiRMeta)
				{
					break;
				}

			}

			if( RC_BAD( rc = pLStream->positionTo( ui64StartPos)))
			{
				goto Exit;
			}
		}
	}

Exit:

	return( rc);
}
	
//-------------------------------------------------------------------------
// Desc:	Performs binary comparison on two streams - may be text or binary,
//			it really doesn't matter.  Returns SQL_TRUE or SQL_FALSE.
//-------------------------------------------------------------------------
FSTATIC RCODE sqlCompareBinary(
	SQL_VALUE *		pLValue,
	SQL_VALUE *		pRValue,
	FLMINT *			piResult)
{
	RCODE					rc = NE_SFLM_OK;
	F_BufferIStream	bufferLStream;
	IF_PosIStream *	pLStream;
	F_BufferIStream	bufferRStream;
	IF_PosIStream *	pRStream;
	FLMBYTE				ucLByte;
	FLMBYTE				ucRByte;
	FLMUINT				uiOffset = 0;
	FLMBOOL				bLEmpty = FALSE;

	*piResult = 0;

	// Types must be binary

	if (pLValue->eValType != SQL_BINARY_VAL ||
		 pRValue->eValType != SQL_BINARY_VAL)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_Q_COMPARE_OPERAND_TYPE_MISMATCH);
		goto Exit;
	}

	// Open the streams

	if (RC_BAD( rc = bufferLStream.openStream( 
		(const char *)pLValue->val.str.pszStr, pLValue->val.str.uiByteLen)))
	{
		goto Exit;
	}
	pLStream = &bufferLStream;

	if( RC_BAD( rc = bufferRStream.openStream( 
		(const char *)pRValue->val.str.pszStr, pRValue->val.str.uiByteLen)))
	{
		goto Exit;
	}
	pRStream = &bufferRStream;

	for (;;)
	{
		if (RC_BAD( rc = flmReadStorageAsBinary( 
			pLStream, &ucLByte, 1, uiOffset, NULL)))
		{
			if (rc == NE_SFLM_EOF_HIT)
			{
				rc = NE_SFLM_OK;
				bLEmpty = TRUE;
			}
			else
			{
				goto Exit;
			}
		}

		if (RC_BAD( rc = flmReadStorageAsBinary( 
			pRStream, &ucRByte, 1, uiOffset, NULL)))
		{
			if (rc == NE_SFLM_EOF_HIT)
			{
				rc = NE_SFLM_OK;
				if( bLEmpty)
				{
					*piResult = 0;
				}
				else
				{
					*piResult = 1;
				}
			}
			goto Exit;
		}
		else if( bLEmpty)
		{
			*piResult = -1;
			goto Exit;
		}

		if( ucLByte != ucRByte)
		{
			*piResult = ucLByte < ucRByte ? -1 : 1;
			goto Exit;
		}

		uiOffset++;
	}

Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Compare two values.  This routine assumes that pValue1 and pValue2
//			are non-null.
//-------------------------------------------------------------------------
RCODE sqlCompare(
	SQL_VALUE *		pValue1,
	SQL_VALUE *		pValue2,
	FLMUINT			uiCompareRules,
	FLMUINT			uiLanguage,
	FLMINT *			piCmp)
{
	RCODE		rc = NE_SFLM_OK;

	// We have already called sqlCanCompare, so no need to do it here

	switch (pValue1->eValType)
	{
		case SQL_BOOL_VAL:
			*piCmp = pValue1->val.eBool > pValue2->val.eBool
					 ? 1
					 : pValue1->val.eBool < pValue2->val.eBool
						? -1
						: 0;
			break;
		case SQL_UINT_VAL:
			switch (pValue2->eValType)
			{
				case SQL_UINT_VAL:
					*piCmp = pValue1->val.uiVal > pValue2->val.uiVal
							 ? 1
							 : pValue1->val.uiVal < pValue2->val.uiVal
							   ? -1
								: 0;
					break;
				case SQL_UINT64_VAL:
					*piCmp = (FLMUINT64)pValue1->val.uiVal > pValue2->val.ui64Val
							 ? 1
							 : (FLMUINT64)pValue1->val.uiVal < pValue2->val.ui64Val
								? -1
								: 0;
					break;
				case SQL_INT_VAL:
					*piCmp = pValue2->val.iVal < 0 ||
							 pValue1->val.uiVal > (FLMUINT)pValue2->val.iVal
							 							 ? 1
							 : pValue1->val.uiVal < (FLMUINT)pValue2->val.iVal
							   ? -1
								: 0;
					break;
				case SQL_INT64_VAL:
					*piCmp = pValue2->val.i64Val < 0 ||
							 (FLMUINT64)pValue1->val.uiVal >
							 (FLMUINT64)pValue2->val.i64Val
							 ? 1
							 : (FLMUINT64)pValue1->val.uiVal <
								(FLMUINT64)pValue2->val.i64Val
								? -1
								: 0;
					break;
				default:
					rc = RC_SET_AND_ASSERT( NE_SFLM_Q_COMPARE_OPERAND_TYPE_MISMATCH);
					goto Exit;
			}
			break;
		case SQL_UINT64_VAL:
			switch (pValue2->eValType)
			{
				case SQL_UINT_VAL:
					*piCmp = pValue1->val.ui64Val > (FLMUINT64)pValue2->val.uiVal
							 ? 1
							 : pValue1->val.ui64Val < (FLMUINT64)pValue2->val.uiVal
							   ? -1
								: 0;
					break;
				case SQL_UINT64_VAL:
					*piCmp = pValue1->val.ui64Val > pValue2->val.ui64Val
							 ? 1
							 : pValue1->val.ui64Val < pValue2->val.ui64Val
							   ? -1
								: 0;
					break;
				case SQL_INT_VAL:
					*piCmp = pValue2->val.iVal < 0 ||
							 pValue1->val.ui64Val > (FLMUINT64)pValue2->val.iVal
							 ? 1
							 : pValue1->val.ui64Val < (FLMUINT64)pValue2->val.iVal
							   ? -1
								: 0;
					break;
				case SQL_INT64_VAL:
					*piCmp = pValue2->val.i64Val < 0 ||
							 pValue1->val.ui64Val > (FLMUINT64)pValue2->val.i64Val
							 ? 1
							 : pValue1->val.ui64Val < (FLMUINT64)pValue2->val.i64Val
							   ? -1
								: 0;
					break;
            default:
					rc = RC_SET_AND_ASSERT( NE_SFLM_Q_COMPARE_OPERAND_TYPE_MISMATCH);
					goto Exit;
			}
			break;
		case SQL_INT_VAL:
			switch (pValue2->eValType)
			{
				case SQL_UINT_VAL:
					*piCmp = pValue1->val.iVal < 0 ||
							 (FLMUINT)pValue1->val.iVal < pValue2->val.uiVal
							 ? -1
							 : (FLMUINT)pValue1->val.iVal > pValue2->val.uiVal
							   ? 1
								: 0;
					break;
				case SQL_UINT64_VAL:
					*piCmp = pValue1->val.iVal < 0 ||
							 (FLMUINT64)pValue1->val.iVal < pValue2->val.ui64Val
							 ? -1
							 : (FLMUINT64)pValue1->val.iVal > pValue2->val.ui64Val
							   ? 1
								: 0;
					break;
				case SQL_INT_VAL:
					*piCmp = pValue1->val.iVal < pValue2->val.iVal
							 ? -1
							 : pValue1->val.iVal > pValue2->val.iVal
							   ? 1
								: 0;
					break;
				case SQL_INT64_VAL:
					*piCmp = (FLMINT64)pValue1->val.iVal < pValue2->val.i64Val
							 ? -1
							 : (FLMINT64)pValue1->val.iVal > pValue2->val.i64Val
							   ? 1
								: 0;
					break;
            default:
					rc = RC_SET_AND_ASSERT( NE_SFLM_Q_COMPARE_OPERAND_TYPE_MISMATCH);
					goto Exit;
			}
			break;
		case SQL_INT64_VAL:
			switch (pValue2->eValType)
			{
				case SQL_UINT_VAL:
					*piCmp = pValue1->val.i64Val < 0 ||
							 (FLMUINT64)pValue1->val.i64Val <
							 (FLMUINT64)pValue2->val.uiVal
							 ? -1
							 : (FLMUINT64)pValue1->val.i64Val >
							   (FLMUINT64)pValue2->val.uiVal
							   ? 1
								: 0;
					break;
				case SQL_UINT64_VAL:
					*piCmp = pValue1->val.i64Val < 0 ||
							 (FLMUINT64)pValue1->val.i64Val < pValue2->val.ui64Val
							 ? -1
							 : (FLMUINT64)pValue1->val.i64Val > pValue2->val.ui64Val
							   ? 1
								: 0;
					break;
				case SQL_INT_VAL:
					*piCmp = pValue1->val.i64Val < (FLMINT64)pValue2->val.iVal
							 ? -1
							 : pValue1->val.i64Val > (FLMINT64)pValue2->val.iVal
							   ? 1
								: 0;
					break;
				case SQL_INT64_VAL:
					*piCmp = pValue1->val.i64Val < pValue2->val.i64Val
							 ? -1
							 : pValue1->val.i64Val > pValue2->val.i64Val
							   ? 1
								: 0;
					break;
				default:
					rc = RC_SET_AND_ASSERT( NE_SFLM_Q_COMPARE_OPERAND_TYPE_MISMATCH);
					goto Exit;
			}
			break;
		case SQL_BINARY_VAL:
			if (RC_BAD( rc = sqlCompareBinary( pValue1, pValue2, piCmp)))
			{
				goto Exit;
			}
			break;
		case SQL_UTF8_VAL:
			if (RC_BAD( rc = sqlCompareText( pValue1, pValue2,
				uiCompareRules, FALSE, uiLanguage, piCmp)))
			{
				goto Exit;
			}
			break;
		default:
			break;
	}

Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Returns a 64-bit unsigned integer
//-------------------------------------------------------------------------
FINLINE FLMUINT64 sqlGetUInt64(
	SQL_VALUE *		pValue)
{
	if (pValue->eValType == SQL_UINT_VAL)
	{
		return( (FLMUINT64)pValue->val.uiVal);
	}
	else if( pValue->eValType == SQL_UINT64_VAL)
	{
		return( pValue->val.ui64Val);
	}
	else if( pValue->eValType == SQL_INT64_VAL)
	{
		if( pValue->val.i64Val >= 0)
		{
			return( (FLMUINT64)pValue->val.i64Val);
		}
	}
	else if( pValue->eValType == SQL_INT_VAL)
	{
		if( pValue->val.iVal >= 0)
		{
			return( (FLMUINT64)pValue->val.iVal);
		}
	}
	
	flmAssert( 0);
	return( 0);
}

//-------------------------------------------------------------------------
// Desc:	Returns a 64-bit signed integer
//-------------------------------------------------------------------------
FINLINE FLMINT64 sqlGetInt64(
	SQL_VALUE *		pValue)
{
	if (pValue->eValType == SQL_INT_VAL)
	{
		return( (FLMINT64)pValue->val.iVal);
	}
	else if( pValue->eValType == SQL_INT64_VAL)
	{
		return( pValue->val.i64Val);
	}
	else if( pValue->eValType == SQL_UINT_VAL)
	{
		return( (FLMINT64)pValue->val.uiVal);
	}
	else if( pValue->eValType == SQL_UINT64_VAL)
	{
		if( pValue->val.ui64Val <= (FLMUINT64)FLM_MAX_INT64)
		{
			return( (FLMINT64)pValue->val.ui64Val);
		}
	}
		
	flmAssert( 0);
	return( 0);
}

//-------------------------------------------------------------------------
// Desc:	Performs the bit and operation
//-------------------------------------------------------------------------
FSTATIC void sqlArithOpUUBitAND(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		pResult->val.uiVal = pLValue->val.uiVal & pRValue->val.uiVal;
		pResult->eValType = SQL_UINT_VAL;
	}
	else
	{
		pResult->val.ui64Val = 
			sqlGetUInt64( pLValue) & sqlGetUInt64( pRValue);
		pResult->eValType = SQL_UINT64_VAL;
	}
}
	
/***************************************************************************
Desc:		Performs the bit or operation
***************************************************************************/
FSTATIC void sqlArithOpUUBitOR(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		pResult->val.uiVal = pLValue->val.uiVal | pRValue->val.uiVal;
		pResult->eValType = SQL_UINT_VAL;
	}
	else
	{
		pResult->val.ui64Val = 
			sqlGetUInt64( pLValue) | sqlGetUInt64( pRValue);
		pResult->eValType = SQL_UINT64_VAL;
	}
}

/***************************************************************************
Desc:		Performs the bit xor operation
***************************************************************************/
FSTATIC void sqlArithOpUUBitXOR(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		pResult->val.uiVal = pLValue->val.uiVal ^ pRValue->val.uiVal;
		pResult->eValType = SQL_UINT_VAL;
	}
	else
	{
		pResult->val.ui64Val = 
			sqlGetUInt64( pLValue) ^ sqlGetUInt64( pRValue);
		pResult->eValType = SQL_UINT64_VAL;
	}
}

/***************************************************************************
Desc:		Performs the multiply operation
***************************************************************************/
FSTATIC void sqlArithOpUUMult(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{	
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		pResult->val.uiVal = pLValue->val.uiVal * pRValue->val.uiVal;
		pResult->eValType = SQL_UINT_VAL;
	}
	else
	{
		pResult->val.ui64Val = 
			sqlGetUInt64( pLValue) * sqlGetUInt64( pRValue);
		pResult->eValType = SQL_UINT64_VAL;
	}
}
	
/***************************************************************************
Desc:		Performs the multiply operation
***************************************************************************/
FSTATIC void sqlArithOpUSMult(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{	
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		pResult->val.iVal = (FLMINT)pLValue->val.uiVal * pRValue->val.iVal;
		pResult->eValType = SQL_INT_VAL;
	}
	else
	{
		pResult->val.i64Val = (FLMINT64)
			sqlGetUInt64( pLValue) * sqlGetInt64( pRValue);
		pResult->eValType = SQL_INT64_VAL;
	}
}
	
/***************************************************************************
Desc:		Performs the multiply operation
***************************************************************************/
FSTATIC void sqlArithOpSSMult(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		pResult->val.iVal = pLValue->val.iVal * pRValue->val.iVal;
		pResult->eValType = (pResult->val.iVal < 0) 
									? SQL_INT_VAL 
									: SQL_UINT_VAL;
	}
	else
	{
		pResult->val.i64Val = (FLMINT64)(sqlGetInt64( pLValue) *
										sqlGetInt64( pRValue));

		pResult->eValType = (pResult->val.i64Val < 0) 
									? SQL_INT64_VAL 
									: SQL_UINT64_VAL;
	}
}

/***************************************************************************
Desc:		Performs the multiply operation
***************************************************************************/
FSTATIC void sqlArithOpSUMult(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		pResult->val.iVal = pLValue->val.iVal * 
			(FLMINT)pRValue->val.uiVal;
		pResult->eValType = SQL_INT_VAL;
	}
	else
	{
		pResult->val.i64Val = (FLMINT64)
			(sqlGetInt64( pLValue) * sqlGetUInt64( pRValue));
		pResult->eValType = SQL_INT64_VAL;
	}
}

/***************************************************************************
Desc:		Performs the divide operation
***************************************************************************/
FSTATIC void sqlArithOpUUDiv(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{	
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if( pRValue->val.uiVal)
		{
			pResult->val.uiVal = pLValue->val.uiVal / pRValue->val.uiVal;
			pResult->eValType = SQL_UINT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
	else
	{
		FLMUINT64	ui64LValue = sqlGetUInt64( pLValue);
		FLMUINT64	ui64RValue = sqlGetUInt64( pRValue);

		if( ui64RValue)
		{
			pResult->val.ui64Val = ui64LValue / ui64RValue;
			pResult->eValType = SQL_UINT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
}
	
/***************************************************************************
Desc:		Performs the divide operation
***************************************************************************/
FSTATIC void sqlArithOpUSDiv(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{	
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if( pRValue->val.iVal)
		{
			pResult->val.iVal = pLValue->val.uiVal / pRValue->val.iVal;
			pResult->eValType = SQL_INT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
	else
	{
		FLMUINT64	ui64LValue = sqlGetUInt64( pLValue);
		FLMINT64		i64RValue = sqlGetInt64( pRValue);

		if( i64RValue)
		{
			pResult->val.i64Val = ui64LValue  / i64RValue;
			pResult->eValType = SQL_INT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
}
	
/***************************************************************************
Desc:		Performs the divide operation
***************************************************************************/
FSTATIC void sqlArithOpSSDiv(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if( pRValue->val.iVal)
		{
			pResult->val.iVal = pLValue->val.iVal / pRValue->val.iVal;
			pResult->eValType = (pResult->val.iVal < 0) 
										? SQL_INT_VAL : SQL_UINT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = sqlGetInt64( pLValue);
		FLMINT64		i64RValue = sqlGetInt64( pRValue);

		if( i64RValue)
		{
			pResult->val.i64Val = i64LValue  / i64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
										? SQL_INT64_VAL : SQL_UINT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs the divide operation
***************************************************************************/
FSTATIC void sqlArithOpSUDiv(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if( pRValue->val.uiVal)
		{
			pResult->val.iVal = pLValue->val.iVal / pRValue->val.uiVal;
			pResult->eValType = SQL_INT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = sqlGetInt64( pLValue);
		FLMUINT64	ui64RValue = sqlGetUInt64( pRValue);

		if( ui64RValue)
		{
			pResult->val.i64Val = i64LValue  / ui64RValue;
			pResult->eValType = SQL_INT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs the modulo operation
***************************************************************************/
FSTATIC void sqlArithOpUUMod(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{	
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if( pRValue->val.uiVal)
		{
			pResult->val.uiVal = pLValue->val.uiVal % pRValue->val.uiVal;
			pResult->eValType = SQL_UINT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
	else
	{
		FLMUINT64	ui64LValue = sqlGetUInt64( pLValue);
		FLMUINT64	ui64RValue = sqlGetUInt64( pRValue);

		if( ui64RValue)
		{
			pResult->val.ui64Val = ui64LValue  % ui64RValue;
			pResult->eValType = SQL_UINT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
}
	
/***************************************************************************
Desc:		Performs the modulo operation
***************************************************************************/
FSTATIC void sqlArithOpUSMod(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{	
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if( pRValue->val.iVal)
		{
			pResult->val.iVal = pLValue->val.uiVal % pRValue->val.iVal;
			pResult->eValType = SQL_INT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
	else
	{
		FLMUINT64	ui64LValue = sqlGetUInt64( pLValue);
		FLMINT64		i64RValue = sqlGetInt64( pRValue);

		if( i64RValue)
		{
			pResult->val.i64Val = ui64LValue  % i64RValue;
			pResult->eValType = SQL_INT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
}
	
/***************************************************************************
Desc:		Performs the modulo operation
***************************************************************************/
FSTATIC void sqlArithOpSSMod(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if( pRValue->val.iVal)
		{
			pResult->val.iVal = pLValue->val.iVal % pRValue->val.iVal;
			pResult->eValType = (pResult->val.iVal < 0) 
										? SQL_INT_VAL : SQL_UINT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = sqlGetInt64( pLValue);
		FLMINT64		i64RValue = sqlGetInt64( pRValue);

		if( i64RValue)
		{
			pResult->val.i64Val = i64LValue % i64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
										? SQL_INT64_VAL : SQL_UINT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs the modulo operation
***************************************************************************/
FSTATIC void sqlArithOpSUMod(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if( pRValue->val.uiVal)
		{
			pResult->val.iVal = pLValue->val.iVal % pRValue->val.uiVal;
			pResult->eValType = SQL_INT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = sqlGetInt64( pLValue);
		FLMUINT64	ui64RValue = sqlGetUInt64( pRValue);

		if( ui64RValue)
		{
			pResult->val.i64Val = i64LValue  % ui64RValue;
			pResult->eValType = SQL_INT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs an addition operation
***************************************************************************/
FSTATIC void sqlArithOpUUPlus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{	
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		pResult->val.uiVal = pLValue->val.uiVal + pRValue->val.uiVal;
		pResult->eValType = SQL_UINT_VAL;
	}
	else
	{
		pResult->val.ui64Val = 
			sqlGetUInt64( pLValue) + sqlGetUInt64( pRValue);
		pResult->eValType = SQL_UINT64_VAL;
	}
}

/***************************************************************************
Desc:		Performs an addition operation
***************************************************************************/
FSTATIC void sqlArithOpUSPlus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{	
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if( (pRValue->val.iVal >= 0) || 
			 (pLValue->val.uiVal > gv_uiMaxSignedIntVal))
		{
			pResult->val.uiVal = pLValue->val.uiVal + (FLMUINT)pRValue->val.iVal;
			pResult->eValType = SQL_UINT_VAL;
		}
		else
		{
			pResult->val.iVal = (FLMINT)pLValue->val.uiVal + pRValue->val.iVal;
			pResult->eValType = (pResult->val.iVal < 0) 
									? SQL_INT_VAL : SQL_UINT_VAL;
		}
	}
	else
	{
		FLMUINT64		ui64LValue = sqlGetUInt64( pLValue);
		FLMINT64			i64RValue = sqlGetInt64( pRValue);

		if( (i64RValue >= 0) || (ui64LValue > gv_ui64MaxSignedIntVal))
		{			pResult->val.ui64Val = ui64LValue + (FLMUINT64)i64RValue;
			pResult->eValType = SQL_UINT64_VAL;
		}
		else
		{
			pResult->val.i64Val = (FLMINT64)ui64LValue + i64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
									? SQL_INT64_VAL : SQL_UINT64_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs an addition operation
***************************************************************************/
FSTATIC void sqlArithOpSSPlus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		pResult->val.iVal = pLValue->val.iVal + pRValue->val.iVal;
		pResult->eValType = (pResult->val.iVal < 0) 
									? SQL_INT_VAL : SQL_UINT_VAL;
	}
	else
	{
		pResult->val.i64Val = 
			sqlGetInt64( pLValue) + sqlGetInt64( pRValue);
		pResult->eValType = (pResult->val.i64Val < 0) 
									? SQL_INT64_VAL : SQL_UINT64_VAL;
	}
}

/***************************************************************************
Desc:		Performs an addition operation
***************************************************************************/
FSTATIC void sqlArithOpSUPlus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if( (pLValue->val.iVal >= 0) ||
			 (pRValue->val.uiVal > gv_uiMaxSignedIntVal))
		{
			pResult->val.uiVal = (FLMUINT)pLValue->val.iVal + pRValue->val.uiVal;
			pResult->eValType = SQL_UINT_VAL;
		}
		else
		{
			pResult->val.iVal = pLValue->val.iVal + (FLMINT)pRValue->val.uiVal;
			pResult->eValType = (pResult->val.iVal < 0) 
									? SQL_INT_VAL : SQL_UINT_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = sqlGetInt64( pLValue);
		FLMUINT64	ui64RValue = sqlGetUInt64( pRValue);

		if( (i64LValue >= 0) || (ui64RValue > gv_ui64MaxSignedIntVal))
		{
			pResult->val.ui64Val = (FLMUINT64)i64LValue + ui64RValue;
			pResult->eValType = SQL_UINT64_VAL;
		}
		else
		{
			pResult->val.i64Val = i64LValue + (FLMINT64)ui64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
									? SQL_INT64_VAL : SQL_UINT64_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs a subtraction operation
***************************************************************************/
FSTATIC void sqlArithOpUUMinus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{	
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if( pLValue->val.uiVal >= pRValue->val.uiVal)
		{
			pResult->val.uiVal = pLValue->val.uiVal - pRValue->val.uiVal;
			pResult->eValType = SQL_UINT_VAL;
		}
		else
		{
			pResult->val.iVal = (FLMINT)(pLValue->val.uiVal - pRValue->val.uiVal);
			pResult->eValType = SQL_INT_VAL;
		}
	}
	else
	{
		FLMUINT64	ui64LValue = sqlGetUInt64( pLValue);
		FLMUINT64	ui64RValue = sqlGetUInt64( pRValue);

		if( ui64LValue >= ui64RValue)
		{
			pResult->val.ui64Val = ui64LValue - ui64RValue;
			pResult->eValType = SQL_UINT64_VAL;
		}
		else
		{
			pResult->val.i64Val = (FLMINT64)(ui64LValue - ui64RValue);
			pResult->eValType = SQL_INT64_VAL;
		}
	}
}
	
/***************************************************************************
Desc:		Performs a subtraction operation
***************************************************************************/
FSTATIC void sqlArithOpUSMinus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{	
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if( pRValue->val.iVal < 0) 
		{
			pResult->val.uiVal = pLValue->val.uiVal - pRValue->val.iVal;
			pResult->eValType = SQL_UINT_VAL;
		}
		else
		{
			pResult->val.iVal = (FLMINT)pLValue->val.uiVal - pRValue->val.iVal;
			pResult->eValType = (pResult->val.iVal < 0) 
									? SQL_INT_VAL : SQL_UINT_VAL;
		}
	}
	else
	{
		FLMUINT64	ui64LValue = sqlGetUInt64( pLValue);
		FLMINT64		i64RValue = sqlGetInt64( pRValue);

		if( i64RValue < 0)
		{
			pResult->val.ui64Val = ui64LValue - i64RValue;
			pResult->eValType = SQL_UINT64_VAL;
		}
		else
		{
			pResult->val.i64Val = (FLMINT64)ui64LValue - i64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
									? SQL_INT64_VAL : SQL_UINT64_VAL;
		}
	}
}
	
/***************************************************************************
Desc:		Performs a subtraction operation
***************************************************************************/
FSTATIC void sqlArithOpSSMinus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if(( pLValue->val.iVal > 0) && ( pRValue->val.iVal < 0))
		{
			pResult->val.uiVal = (FLMUINT)(pLValue->val.iVal - pRValue->val.iVal);
			pResult->eValType = SQL_UINT_VAL;
		}
		else
		{
			pResult->val.iVal = pLValue->val.iVal - pRValue->val.iVal;
			pResult->eValType = (pResult->val.iVal < 0) 
									? SQL_INT_VAL : SQL_UINT_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = sqlGetInt64( pLValue);
		FLMINT64		i64RValue = sqlGetInt64( pRValue);

		if( (i64LValue > 0) && (i64RValue < 0))
		{
			pResult->val.ui64Val = (FLMUINT64)( i64LValue - i64RValue);
			pResult->eValType = SQL_UINT64_VAL;
		}
		else
		{
			pResult->val.i64Val = i64LValue - i64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
									? SQL_INT64_VAL : SQL_UINT64_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs a subtraction operation
***************************************************************************/
FSTATIC void sqlArithOpSUMinus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if( pRValue->val.uiVal > gv_uiMaxSignedIntVal)
		{
			pResult->val.iVal = (pLValue->val.iVal - gv_uiMaxSignedIntVal) - 
				(FLMINT)(pRValue->val.uiVal - gv_uiMaxSignedIntVal);
			pResult->eValType = SQL_INT_VAL;
		}
		else
		{
			pResult->val.iVal = pLValue->val.iVal - (FLMINT)pRValue->val.uiVal;
			pResult->eValType = (pResult->val.iVal < 0) 
									? SQL_INT_VAL : SQL_UINT_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = sqlGetInt64( pLValue);
		FLMUINT64	ui64RValue = sqlGetUInt64( pRValue);

		if( ui64RValue > gv_ui64MaxSignedIntVal)
		{
			pResult->val.i64Val = (i64LValue - gv_ui64MaxSignedIntVal) -
				(FLMINT64)(ui64RValue - gv_ui64MaxSignedIntVal);
			pResult->eValType = SQL_INT64_VAL;
		}
		else
		{
			pResult->val.i64Val = i64LValue - (FLMINT64)ui64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
									? SQL_INT64_VAL : SQL_UINT64_VAL;
		}
	}
}

/***************************************************************************
Desc:	Do a comparison operator.
***************************************************************************/
FSTATIC RCODE sqlCompareOperands(
	FLMUINT					uiLanguage,
	SQL_VALUE *				pLValue,
	SQL_VALUE *				pRValue,
	eSQLQueryOperators	eOperator,
	FLMUINT					uiCompareRules,
	FLMBOOL,					// bNotted,
	SQLBoolType *			peBool)
{
	RCODE			rc = NE_SFLM_OK;
	FLMINT		iCmp;

	if (!pLValue || pLValue->eValType == SQL_MISSING_VAL ||
		 !pRValue || pRValue->eValType == SQL_MISSING_VAL ||
		 !sqlCanCompare( pLValue, pRValue))
	{
		*peBool = SQL_UNKNOWN;
	}

	// At this point, both operands are known to be present and are of
	// types that can be compared.  The comparison
	// will therefore be performed according to the
	// operator specified.
	
	else
	{
		switch (eOperator)
		{
			case SQL_EQ_OP:
			case SQL_NE_OP:
				if (pLValue->eValType == SQL_UTF8_VAL ||
					 pRValue->eValType == SQL_UTF8_VAL)
				{
					if (RC_BAD( rc = sqlCompareText( pLValue, pRValue,
						uiCompareRules, TRUE, uiLanguage, &iCmp)))
					{
						goto Exit;
					}
				}
				else
				{
					if (RC_BAD( rc = sqlCompare( pLValue, pRValue, 
						uiCompareRules, uiLanguage, &iCmp)))
					{
						goto Exit;
					}
				}
				if (eOperator == SQL_EQ_OP)
				{
					*peBool = (iCmp == 0 ? SQL_TRUE : SQL_FALSE);
				}
				else
				{
					*peBool = (iCmp != 0 ? SQL_TRUE : SQL_FALSE);
				}
				break;

			case SQL_APPROX_EQ_OP:
				if (RC_BAD( rc = sqlApproxCompare( pLValue, pRValue, &iCmp)))
				{
					goto Exit;
				}
				*peBool = (iCmp == 0 ? SQL_TRUE : SQL_FALSE);
				break;

			case SQL_LT_OP:
				if (RC_BAD( rc = sqlCompare( pLValue, pRValue, 
					uiCompareRules, uiLanguage, &iCmp)))
				{
					goto Exit;
				}
				*peBool = (iCmp < 0 ? SQL_TRUE : SQL_FALSE);
				break;

			case SQL_LE_OP:
				if (RC_BAD( rc = sqlCompare( pLValue, pRValue, 
					uiCompareRules, uiLanguage, &iCmp)))
				{
					goto Exit;
				}
				*peBool = (iCmp <= 0 ? SQL_TRUE : SQL_FALSE);
				break;

			case SQL_GT_OP:
				if (RC_BAD( rc = sqlCompare( pLValue, pRValue, 
					uiCompareRules, uiLanguage, &iCmp)))
				{
					goto Exit;
				}
				*peBool = (iCmp > 0 ? SQL_TRUE : SQL_FALSE);
				break;

			case SQL_GE_OP:
				if (RC_BAD( rc = sqlCompare( pLValue, pRValue, 
					uiCompareRules, uiLanguage, &iCmp)))
				{
					goto Exit;
				}
				*peBool = (iCmp >= 0 ? SQL_TRUE : SQL_FALSE);
				break;

			default:
				*peBool = SQL_UNKNOWN;
				rc = RC_SET_AND_ASSERT( NE_SFLM_QUERY_SYNTAX);
				goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Do an arithmetic operator.
***************************************************************************/
RCODE sqlEvalArithOperator(
	SQL_VALUE *				pLValue,
	SQL_VALUE *				pRValue,
	eSQLQueryOperators	eOperator,
	SQL_VALUE *				pResult)
{
	RCODE					rc = NE_SFLM_OK;
	SQL_ARITH_OP *		fnOp;
	FLMUINT				uiOffset = 0;

	if (!isSQLArithOp( eOperator))
	{
		rc = RC_SET( NE_SFLM_Q_INVALID_OPERATOR);
		goto Exit;
	}

	if (pLValue->eValType == SQL_MISSING_VAL ||
		 pRValue->eValType == SQL_MISSING_VAL)
	{
		pResult->eValType = SQL_MISSING_VAL;
		goto Exit;
	}

	if (isSQLValUnsigned( pLValue->eValType))
	{
		if (isSQLValUnsigned( pRValue->eValType))
		{
			uiOffset = 0;
		}
		else if (isSQLValSigned( pRValue->eValType))
		{
			uiOffset = 1;
		}
		else
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_NOT_IMPLEMENTED);
			goto Exit;
		}
	}
	else if (isSQLValSigned( pLValue->eValType))
	{
		if (isSQLValUnsigned( pRValue->eValType))
		{
			uiOffset = 2;
		}
		else if (isSQLValSigned( pRValue->eValType))
		{
			uiOffset = 3;
		}
		else
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_NOT_IMPLEMENTED);
			goto Exit;
		}
	}

	fnOp = SQL_ArithOpTable[ ((((FLMUINT)eOperator) - 
					SQL_FIRST_ARITH_OP) * 4) + uiOffset];
	fnOp( pLValue, pRValue, pResult);

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get a column's value from the passed in row.  Return it in pSqlValue.
***************************************************************************/
FSTATIC RCODE sqlGetColumnValue(
	F_Db *		pDb,
	F_Row *		pRow,
	FLMUINT		uiTableNum,
	FLMUINT		uiColumnNum,
	F_Pool *		pPool,
	SQL_VALUE *	pSqlValue)
{
	RCODE					rc = NE_SFLM_OK;
	F_TABLE *			pTable = pDb->getDict()->getTable( uiTableNum);
	F_COLUMN *			pColumn = pDb->getDict()->getColumn( pTable, uiColumnNum);
	FLMBOOL				bNeg;
	FLMUINT64			ui64Value;
	FLMBOOL				bIsNull;
	FLMUINT				uiDataLen;
	const FLMBYTE *	pucColumnData;
	const FLMBYTE *	pucEnd;
	
	flmAssert( pTable->uiTableNum == uiTableNum);
	
	pRow->getDataLen( pDb, uiColumnNum, &uiDataLen, &bIsNull);
	if (bIsNull)
	{
		pSqlValue->eValType = SQL_MISSING_VAL;
		goto Exit;
	}
	pucColumnData = (const FLMBYTE *)pRow->getColumnDataPtr( uiColumnNum);
	switch (pColumn->eDataTyp)
	{
		case SFLM_STRING_TYPE:
		
			// Decode the number of characters directly from the column's
			// data buffer.  Then copy only whatever part remains after that.

			pSqlValue->eValType = SQL_UTF8_VAL;			
			pucEnd = pucColumnData + uiDataLen;
			if (RC_BAD( rc = f_decodeSEN( &pucColumnData, pucEnd,
										&pSqlValue->val.str.uiNumChars)))
			{
				goto Exit;
			}
			uiDataLen = (FLMUINT)(pucEnd - pucColumnData);
			pSqlValue->val.str.uiByteLen = uiDataLen;
			if (RC_BAD( rc = pPool->poolAlloc( uiDataLen,
										(void **)&pSqlValue->val.str.pszStr)))
			{
				goto Exit;
			}
			f_memcpy( pSqlValue->val.str.pszStr, pucColumnData, uiDataLen);
			break;
			
		case SFLM_NUMBER_TYPE:
			if (RC_BAD( rc = flmStorageNumberToNumber( pucColumnData, uiDataLen,
										&ui64Value, &bNeg)))
			{
				goto Exit;
			}
			if (!bNeg)
			{
				if (ui64Value <= (FLMUINT64)(FLM_MAX_UINT))
				{
					pSqlValue->eValType = SQL_UINT_VAL;
					pSqlValue->val.uiVal = (FLMUINT)ui64Value;
				}
				else
				{
					pSqlValue->eValType = SQL_UINT64_VAL;
					pSqlValue->val.ui64Val = ui64Value;
				}
			}
			else
			{
				if (-((FLMINT64)ui64Value) <= (FLMINT64)(FLM_MIN_INT))
				{
					pSqlValue->eValType = SQL_INT_VAL;
					pSqlValue->val.iVal = (FLMINT)(-((FLMINT64)ui64Value));
				}
				else
				{
					pSqlValue->eValType = SQL_INT64_VAL;
					pSqlValue->val.i64Val = -((FLMINT64)ui64Value);
				}
			}
			break;
			
		case SFLM_BINARY_TYPE:
			pSqlValue->eValType = SQL_BINARY_VAL;
			if (RC_BAD( rc = pPool->poolAlloc( uiDataLen,
										(void **)&pSqlValue->val.bin.pucValue)))
			{
				goto Exit;
			}
			pSqlValue->val.bin.uiByteLen = uiDataLen;
			f_memcpy( pSqlValue->val.bin.pucValue, pucColumnData, uiDataLen);
			break;
		
		default:
			flmAssert( 0);
			rc = RC_SET( NE_SFLM_FAILURE);
			goto Exit;
	}
	
Exit:

	return( rc);
}

/***************************************************************************
Desc:	Evaluate a simple operator.
***************************************************************************/
FSTATIC RCODE sqlEvalOperator(
	FLMUINT		uiLanguage,
	SQL_NODE *	pQNode)
{
	RCODE				rc = NE_SFLM_OK;
	SQL_NODE *		pLeftOperand;
	SQL_NODE *		pRightOperand;
	SQLBoolType		eBool;

	// Right now we are only able to do operator nodes.

	flmAssert( pQNode->eNodeType == SQL_OPERATOR_NODE);

	pLeftOperand = pQNode->pFirstChild;
	pRightOperand = pQNode->pLastChild;

	pQNode->currVal.eValType = SQL_MISSING_VAL;

	switch (pQNode->nd.op.eOperator)
	{
		case SQL_AND_OP:
		case SQL_OR_OP:
			pQNode->currVal.eValType = SQL_BOOL_VAL;
		
			// There may be multiple operands here.  We have already looked
			// at all of the operands.  If this is an AND we know that all
			// of them were either TRUE or UNKNOWN.  If this is an OR, we know
			// that all of the previous operands were either FALSE or UNKNOWN.
			// If we had hit a FALSE for an AND or a TRUE for an OR, we would
			// not have come to this point.  Now we just need to determine if
			// any of the operands are UNKNOWN.  If so, that is what we will
			// set this node's value to.  Otherwise, we will set it to TRUE for
			// AND and FALSE for OR.
			
			while (pLeftOperand)
			{
			
				// Get the left operand
	
				if (pLeftOperand->eNodeType == SQL_OPERATOR_NODE)
				{
	
					// This operator may not have been evaluated because of missing
					// column values in one or both operands, in which case
					// its state will be SQL_MISSING_VALUE.  If it was evaluated,
					// its state should show a boolean value.
	
					if (pLeftOperand->currVal.eValType == SQL_MISSING_VAL)
					{
						eBool = (pLeftOperand->bNotted ? SQL_TRUE : SQL_FALSE);
					}
					else
					{
						flmAssert( pLeftOperand->currVal.eValType == SQL_BOOL_VAL);
						eBool = pLeftOperand->currVal.val.eBool;
					}
				}
				else if (pLeftOperand->eNodeType == SQL_COLUMN_NODE)
				{
					if (!pLeftOperand->bNotted)
					{
						eBool = (pLeftOperand->currVal.eValType != SQL_MISSING_VAL)
										  ? SQL_TRUE
										  : SQL_FALSE;
					}
					else
					{
						eBool = (pLeftOperand->currVal.eValType != SQL_MISSING_VAL)
										  ? SQL_FALSE
										  : SQL_TRUE;
					}
				}
				else
				{
					flmAssert( pLeftOperand->eNodeType == SQL_VALUE_NODE);
					flmAssert( pLeftOperand->currVal.eValType == SQL_BOOL_VAL);
					eBool = pLeftOperand->currVal.val.eBool;
				}
				
				// eBool better not have FALSE for an AND operator or
				// TRUE for an OR operator.
				
				flmAssert( (pQNode->nd.op.eOperator == SQL_AND_OP &&
							    eBool != SQL_FALSE) ||
								(pQNode->nd.op.eOperator == SQL_OR_OP &&
								 eBool != SQL_TRUE));
								 
				
				if (eBool == SQL_UNKNOWN)
				{
					pQNode->currVal.val.eBool = SQL_UNKNOWN;
					break;
				}
				pLeftOperand = pLeftOperand->pNextSib;
			}
			
			// If we didn't hit an UNKNOWN, set the node's value to TRUE for
			// an AND operator and FALSE for an OR operator.
			
			if (!pLeftOperand)
			{
				pQNode->currVal.val.eBool = pQNode->nd.op.eOperator == SQL_AND_OP
													 ? SQL_TRUE
													 : SQL_FALSE;
			}
			break;

		case SQL_EQ_OP:
		case SQL_APPROX_EQ_OP:
		case SQL_NE_OP:
		case SQL_LT_OP:
		case SQL_LE_OP:
		case SQL_GT_OP:
		case SQL_GE_OP:
			pQNode->currVal.eValType = SQL_BOOL_VAL;
			if (RC_BAD( rc = sqlCompareOperands( uiLanguage,
										&pLeftOperand->currVal,
										&pRightOperand->currVal,
										pQNode->nd.op.eOperator,
										pQNode->nd.op.uiCompareRules,
										pQNode->bNotted,
										&pQNode->currVal.val.eBool)))
			{
				goto Exit;
			}
			break;

		case SQL_BITAND_OP:
		case SQL_BITOR_OP:
		case SQL_BITXOR_OP:
		case SQL_MULT_OP:
		case SQL_DIV_OP:
		case SQL_MOD_OP:
		case SQL_PLUS_OP:
		case SQL_MINUS_OP:
		case SQL_NEG_OP:
		
			if (RC_BAD( rc = sqlEvalArithOperator( &pLeftOperand->currVal,
										&pRightOperand->currVal,
										pQNode->nd.op.eOperator,
										&pQNode->currVal)))
			{
				goto Exit;
			}
			break;
		
		default:
			break;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Evaluate a query expression.
***************************************************************************/
RCODE sqlEvalCriteria(
	F_Db *			pDb,
	SQL_NODE *		pQueryExpr,
	SQL_VALUE **	ppSqlValue,
	F_Pool *			pPool,
	F_Row *			pRow,
	FLMUINT			uiLanguage)
{
	RCODE				rc = NE_SFLM_OK;
	SQL_NODE *		pCurrNode;
	SQL_VALUE *		pSqlValue;
	SQLBoolType		eBoolVal;
	SQLBoolType		eBoolPartialEval;

	// If the query is empty, return a value of SQL_TRUE.

	if (!pQueryExpr)
	{
		if (RC_BAD( rc = pPool->poolAlloc( sizeof( SQL_VALUE),
									(void **)ppSqlValue)))
		{
			goto Exit;
		}
		pSqlValue = *ppSqlValue;
		pSqlValue->eValType = SQL_BOOL_VAL;
		pSqlValue->uiFlags = SQL_VAL_IS_CONSTANT;
		pSqlValue->val.eBool = SQL_TRUE;
		goto Exit;
	}
	
	// If the query is a constant, return pointer to its value.
	
	if (pQueryExpr->eNodeType == SQL_VALUE_NODE)
	{
		*ppSqlValue = &pQueryExpr->currVal;
		goto Exit;
	}
	
	// If the query is a column, get the column's value from the
	// row that was passed in.
	
	if (pQueryExpr->eNodeType == SQL_COLUMN_NODE)
	{
		*ppSqlValue = &pQueryExpr->currVal;
		rc = sqlGetColumnValue( pDb, pRow,
									pQueryExpr->nd.column.pSQLTable->uiTableNum,
									pQueryExpr->nd.column.uiColumnNum,
									pPool, *ppSqlValue);
		goto Exit;
	}

	// Perform the evaluation

	pCurrNode = pQueryExpr;
	for (;;)
	{
		while (pCurrNode->pFirstChild)
		{
			pCurrNode = pCurrNode->pFirstChild;
		}

		// We should be positioned on a leaf node that is either a
		// value or a column

		if (pCurrNode->eNodeType == SQL_COLUMN_NODE)
		{
			if (RC_BAD( rc = sqlGetColumnValue( pDb, pRow,
									pCurrNode->nd.column.pSQLTable->uiTableNum,
									pCurrNode->nd.column.uiColumnNum,
									pPool, &pCurrNode->currVal)))
			{
				goto Exit;
			}
		}
		else
		{
			
			// Better be a constant

			flmAssert( pCurrNode->eNodeType == SQL_VALUE_NODE);
		}
			
		// When we get to this point, we have at least one leaf
		// level operand in hand - pCurrNode.
		// See if we can evaluate the operator of pCurrNode.
		// This will take care of any short-circuiting evaluation
		// that can be done.
		
		for (;;)
		{
			if (pCurrNode == pQueryExpr)
			{
				*ppSqlValue = &pQueryExpr->currVal;
				goto Exit;
			}
	
			// If the current node's parent is an AND or OR
			// operator, see if we even need to go to the next
			// sibling.
	
			flmAssert( pCurrNode->pParent->eNodeType == SQL_OPERATOR_NODE);
			if (isSQLLogicalOp( pCurrNode->pParent->nd.op.eOperator))
			{
				// All NOT operators should have been weeded out of the tree
				// by now.
				
				flmAssert( pCurrNode->pParent->nd.op.eOperator != SQL_NOT_OP);
				eBoolVal = SQL_UNKNOWN;
				eBoolPartialEval = pCurrNode->pParent->nd.op.eOperator == SQL_AND_OP
										  ? SQL_FALSE
										  : SQL_TRUE;
				if (pCurrNode->eNodeType == SQL_OPERATOR_NODE)
				{
	
					// It may not have been evaluated because of missing
					// values in one or both operands, in which case
					// its state will be SQL_MISSING_VALUE.  If it was
					// evaluated, its state should show a boolean value.
	
					if (pCurrNode->currVal.eValType == SQL_MISSING_VAL)
					{
						eBoolVal = (pCurrNode->bNotted ? SQL_TRUE : SQL_FALSE);
					}
					else
					{
						flmAssert( pCurrNode->currVal.eValType == SQL_BOOL_VAL);
						eBoolVal = pCurrNode->currVal.val.eBool;
					}
				}
				else if (pCurrNode->eNodeType == SQL_COLUMN_NODE)
				{
					if (!pCurrNode->bNotted)
					{
						eBoolVal = (pCurrNode->currVal.eValType == SQL_MISSING_VAL)
										? SQL_FALSE
										: SQL_TRUE;
					}
					else
					{
						eBoolVal = (pCurrNode->currVal.eValType == SQL_MISSING_VAL)
										? SQL_TRUE
										: SQL_FALSE;
					}
				}
				else
				{
					flmAssert( pCurrNode->eNodeType == SQL_VALUE_NODE);
					
					// Only allowed value node underneath a logical operator is
					// a boolean value that has a value of SQL_UNKNOWN.
					// SQL_FALSE and SQL_TRUE will already have been weeded out.
	
					flmAssert( pCurrNode->currVal.eValType == SQL_BOOL_VAL &&
								  pCurrNode->currVal.val.eBool == SQL_UNKNOWN);
	
					// No need to set eBoolVal to SQL_UNKNOWN, because it will never
					// match eBoolPartialEval in the test below.  eBoolPartialEval
					// is always either SQL_FALSE or SQL_TRUE.
	
					// eBoolVal = SQL_UNKNOWN;
	
				}
				if (eBoolVal == eBoolPartialEval)
				{
					pCurrNode = pCurrNode->pParent;
					pCurrNode->currVal.eValType = SQL_BOOL_VAL;
					pCurrNode->currVal.val.eBool = eBoolVal;
				}
				else
				{
					goto Check_Sibling_Operand;
				}
			}
			else if (isSQLCompareOp( pCurrNode->pParent->nd.op.eOperator))
			{
				
				// We can short-circuit the comparison - avoid getting the
				// sibling operand - if the current node is a missing value.
				// If the value is missing, the comparison operator will
				// return a boolean SQL_UNKNOWN, regardless of what the
				// other operand is.
				
				if (pCurrNode->currVal.eValType == SQL_MISSING_VAL)
				{
					pCurrNode = pCurrNode->pParent;
					pCurrNode->currVal.eValType = SQL_BOOL_VAL;
					pCurrNode->currVal.val.eBool = SQL_UNKNOWN;
				}
				else
				{
					goto Check_Sibling_Operand;
				}
			}
			else if (isSQLArithOp(  pCurrNode->pParent->nd.op.eOperator))
			{
				
				// We can short-circuit the arithmetic operation - avoid getting the
				// sibling operand - if the current node is a missing value.
				// If the value is missing, the arithmetic operation will
				// return a missing value, regardless of what the
				// other operand is.
				
				if (pCurrNode->currVal.eValType == SQL_MISSING_VAL)
				{
					pCurrNode = pCurrNode->pParent;
					pCurrNode->currVal.eValType = SQL_MISSING_VAL;
				}
				else
				{
					goto Check_Sibling_Operand;
				}
			}
			else
			{

Check_Sibling_Operand:

				if (pCurrNode->pNextSib)
				{
					pCurrNode = pCurrNode->pNextSib;
					break;
				}
				pCurrNode = pCurrNode->pParent;
	
				// All operands are now present - do evaluation
	
				if (RC_BAD( rc = sqlEvalOperator( uiLanguage, pCurrNode)))
				{
					goto Exit;
				}
			}
		}
	}
	
Exit:

	return( rc);
}

