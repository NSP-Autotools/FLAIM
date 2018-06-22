//------------------------------------------------------------------------------
// Desc:	Contains the methods for doing evaluation of query expressions.
// Tabs:	3
//
// Copyright (c) 2003-2007 Novell, Inc. All Rights Reserved.
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

FSTATIC RCODE fqApproxCompare(
	FQVALUE *			pLValue,
	FQVALUE *			pRValue,
	FLMINT *				piResult);

FSTATIC RCODE fqCompareBinary(
	IF_OperandComparer *	pOpComparer,
	FQVALUE *				pLValue,
	FQVALUE *				pRValue,
	FLMINT *					piResult);

FSTATIC RCODE fqCompareText(
	IF_OperandComparer *	pOpComparer,
	FQVALUE *				pLValue,
	FQVALUE *				pRValue,
	FLMUINT					uiCompareRules,
	FLMBOOL					bOpIsMatch,
	FLMUINT					uiLanguage,
	FLMINT *					piResult);

FSTATIC void fqOpUUBitAND(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpUUBitOR(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpUUBitXOR(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpUUMult(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpUSMult(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpSSMult(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpSUMult(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpUUDiv(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpUSDiv(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpSSDiv(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpSUDiv(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpUUMod(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpUSMod(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpSSMod(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpSUMod(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpUUPlus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpUSPlus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpSSPlus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpSUPlus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpUUMinus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpUSMinus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpSSMinus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpSUMinus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

typedef void FQ_OPERATION(
	FQVALUE *		pLValue,
	FQVALUE *		pRValue,
	FQVALUE *		pResult);

FQ_OPERATION * FQ_ArithOpTable[ 
	((XFLM_LAST_ARITH_OP - XFLM_FIRST_ARITH_OP) + 1) * 4 ] =
{
/*	U = Unsigned		S = Signed
					U + U					U + S
						S + U					S + S */
/* BITAND */	fqOpUUBitAND,		fqOpUUBitAND,
						fqOpUUBitAND,		fqOpUUBitAND,
/* BITOR  */	fqOpUUBitOR,		fqOpUUBitOR,
						fqOpUUBitOR,		fqOpUUBitOR,
/* BITXOR */	fqOpUUBitXOR,		fqOpUUBitXOR,
						fqOpUUBitXOR,		fqOpUUBitXOR,
/* MULT   */	fqOpUUMult,			fqOpUSMult,
						fqOpSUMult,			fqOpSSMult,
/* DIV    */	fqOpUUDiv,			fqOpUSDiv,
						fqOpSUDiv,			fqOpSSDiv,
/* MOD    */	fqOpUUMod,			fqOpUSMod,
						fqOpSUMod,			fqOpSSMod,
/* PLUS   */	fqOpUUPlus,			fqOpUSPlus,
						fqOpSUPlus,			fqOpSSPlus,
/* MINUS  */	fqOpUUMinus,		fqOpUSMinus,
						fqOpSUMinus,		fqOpSSMinus
};

/***************************************************************************
Desc:		Returns a 64-bit unsigned integer
***************************************************************************/
FINLINE FLMUINT64 fqGetUInt64(
	FQVALUE *		pValue)
{
	if (pValue->eValType == XFLM_UINT_VAL)
	{
		return( (FLMUINT64)pValue->val.uiVal);
	}
	else if( pValue->eValType == XFLM_UINT64_VAL)
	{
		return( pValue->val.ui64Val);
	}
	else if( pValue->eValType == XFLM_INT64_VAL)
	{
		if( pValue->val.i64Val >= 0)
		{
			return( (FLMUINT64)pValue->val.i64Val);
		}
	}
	else if( pValue->eValType == XFLM_INT_VAL)
	{
		if( pValue->val.iVal >= 0)
		{
			return( (FLMUINT64)pValue->val.iVal);
		}
	}
	
	flmAssert( 0);
	return( 0);
}

/***************************************************************************
Desc:		Returns a 64-bit signed integer
***************************************************************************/
FINLINE FLMINT64 fqGetInt64(
	FQVALUE *		pValue)
{
	if (pValue->eValType == XFLM_INT_VAL)
	{
		return( (FLMINT64)pValue->val.iVal);
	}
	else if( pValue->eValType == XFLM_INT64_VAL)
	{
		return( pValue->val.i64Val);
	}
	else if( pValue->eValType == XFLM_UINT_VAL)
	{
		return( (FLMINT64)pValue->val.uiVal);
	}
	else if( pValue->eValType == XFLM_UINT64_VAL)
	{
		if( pValue->val.ui64Val <= (FLMUINT64)FLM_MAX_INT64)
		{
			return( (FLMINT64)pValue->val.ui64Val);
		}
	}
		
	flmAssert( 0);
	return( 0);
}

/***************************************************************************
Desc:		Performs the bit and operation
***************************************************************************/
FSTATIC void fqOpUUBitAND(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		pResult->val.uiVal = pLValue->val.uiVal & pRValue->val.uiVal;
		pResult->eValType = XFLM_UINT_VAL;
	}
	else
	{
		pResult->val.ui64Val = 
			fqGetUInt64( pLValue) & fqGetUInt64( pRValue);
		pResult->eValType = XFLM_UINT64_VAL;
	}
}
	
/***************************************************************************
Desc:		Performs the bit or operation
***************************************************************************/
FSTATIC void fqOpUUBitOR(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		pResult->val.uiVal = pLValue->val.uiVal | pRValue->val.uiVal;
		pResult->eValType = XFLM_UINT_VAL;
	}
	else
	{
		pResult->val.ui64Val = 
			fqGetUInt64( pLValue) | fqGetUInt64( pRValue);
		pResult->eValType = XFLM_UINT64_VAL;
	}
}

/***************************************************************************
Desc:		Performs the bit xor operation
***************************************************************************/
FSTATIC void fqOpUUBitXOR(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		pResult->val.uiVal = pLValue->val.uiVal ^ pRValue->val.uiVal;
		pResult->eValType = XFLM_UINT_VAL;
	}
	else
	{
		pResult->val.ui64Val = 
			fqGetUInt64( pLValue) ^ fqGetUInt64( pRValue);
		pResult->eValType = XFLM_UINT64_VAL;
	}
}

/***************************************************************************
Desc:		Performs the multiply operation
***************************************************************************/
FSTATIC void fqOpUUMult(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{	
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		pResult->val.uiVal = pLValue->val.uiVal * pRValue->val.uiVal;
		pResult->eValType = XFLM_UINT_VAL;
	}
	else
	{
		pResult->val.ui64Val = 
			fqGetUInt64( pLValue) * fqGetUInt64( pRValue);
		pResult->eValType = XFLM_UINT64_VAL;
	}
}
	
/***************************************************************************
Desc:		Performs the multiply operation
***************************************************************************/
FSTATIC void fqOpUSMult(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{	
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		pResult->val.iVal = (FLMINT)pLValue->val.uiVal * pRValue->val.iVal;
		pResult->eValType = XFLM_INT_VAL;
	}
	else
	{
		pResult->val.i64Val = (FLMINT64)
			fqGetUInt64( pLValue) * fqGetInt64( pRValue);
		pResult->eValType = XFLM_INT64_VAL;
	}
}
	
/***************************************************************************
Desc:		Performs the multiply operation
***************************************************************************/
FSTATIC void fqOpSSMult(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		pResult->val.iVal = pLValue->val.iVal * pRValue->val.iVal;
		pResult->eValType = (pResult->val.iVal < 0) 
									? XFLM_INT_VAL 
									: XFLM_UINT_VAL;
	}
	else
	{
		pResult->val.i64Val = (FLMINT64)(fqGetInt64( pLValue) *
										fqGetInt64( pRValue));

		pResult->eValType = (pResult->val.i64Val < 0) 
									? XFLM_INT64_VAL 
									: XFLM_UINT64_VAL;
	}
}

/***************************************************************************
Desc:		Performs the multiply operation
***************************************************************************/
FSTATIC void fqOpSUMult(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		pResult->val.iVal = pLValue->val.iVal * 
			(FLMINT)pRValue->val.uiVal;
		pResult->eValType = XFLM_INT_VAL;
	}
	else
	{
		pResult->val.i64Val = (FLMINT64)
			(fqGetInt64( pLValue) * fqGetUInt64( pRValue));
		pResult->eValType = XFLM_INT64_VAL;
	}
}

/***************************************************************************
Desc:		Performs the divide operation
***************************************************************************/
FSTATIC void fqOpUUDiv(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{	
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if( pRValue->val.uiVal)
		{
			pResult->val.uiVal = pLValue->val.uiVal / pRValue->val.uiVal;
			pResult->eValType = XFLM_UINT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
	else
	{
		FLMUINT64	ui64LValue = fqGetUInt64( pLValue);
		FLMUINT64	ui64RValue = fqGetUInt64( pRValue);

		if( ui64RValue)
		{
			pResult->val.ui64Val = ui64LValue / ui64RValue;
			pResult->eValType = XFLM_UINT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
}
	
/***************************************************************************
Desc:		Performs the divide operation
***************************************************************************/
FSTATIC void fqOpUSDiv(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{	
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if( pRValue->val.iVal)
		{
			pResult->val.iVal = pLValue->val.uiVal / pRValue->val.iVal;
			pResult->eValType = XFLM_INT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
	else
	{
		FLMUINT64	ui64LValue = fqGetUInt64( pLValue);
		FLMINT64		i64RValue = fqGetInt64( pRValue);

		if( i64RValue)
		{
			pResult->val.i64Val = ui64LValue  / i64RValue;
			pResult->eValType = XFLM_INT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
}
	
/***************************************************************************
Desc:		Performs the divide operation
***************************************************************************/
FSTATIC void fqOpSSDiv(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if( pRValue->val.iVal)
		{
			pResult->val.iVal = pLValue->val.iVal / pRValue->val.iVal;
			pResult->eValType = (pResult->val.iVal < 0) 
										? XFLM_INT_VAL : XFLM_UINT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = fqGetInt64( pLValue);
		FLMINT64		i64RValue = fqGetInt64( pRValue);

		if( i64RValue)
		{
			pResult->val.i64Val = i64LValue  / i64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
										? XFLM_INT64_VAL : XFLM_UINT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs the divide operation
***************************************************************************/
FSTATIC void fqOpSUDiv(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if( pRValue->val.uiVal)
		{
			pResult->val.iVal = pLValue->val.iVal / pRValue->val.uiVal;
			pResult->eValType = XFLM_INT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = fqGetInt64( pLValue);
		FLMUINT64	ui64RValue = fqGetUInt64( pRValue);

		if( ui64RValue)
		{
			pResult->val.i64Val = i64LValue  / ui64RValue;
			pResult->eValType = XFLM_INT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs the modulo operation
***************************************************************************/
FSTATIC void fqOpUUMod(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{	
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if( pRValue->val.uiVal)
		{
			pResult->val.uiVal = pLValue->val.uiVal % pRValue->val.uiVal;
			pResult->eValType = XFLM_UINT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
	else
	{
		FLMUINT64	ui64LValue = fqGetUInt64( pLValue);
		FLMUINT64	ui64RValue = fqGetUInt64( pRValue);

		if( ui64RValue)
		{
			pResult->val.ui64Val = ui64LValue  % ui64RValue;
			pResult->eValType = XFLM_UINT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
}
	
/***************************************************************************
Desc:		Performs the modulo operation
***************************************************************************/
FSTATIC void fqOpUSMod(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{	
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if( pRValue->val.iVal)
		{
			pResult->val.iVal = pLValue->val.uiVal % pRValue->val.iVal;
			pResult->eValType = XFLM_INT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
	else
	{
		FLMUINT64	ui64LValue = fqGetUInt64( pLValue);
		FLMINT64		i64RValue = fqGetInt64( pRValue);

		if( i64RValue)
		{
			pResult->val.i64Val = ui64LValue  % i64RValue;
			pResult->eValType = XFLM_INT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
}
	
/***************************************************************************
Desc:		Performs the modulo operation
***************************************************************************/
FSTATIC void fqOpSSMod(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if( pRValue->val.iVal)
		{
			pResult->val.iVal = pLValue->val.iVal % pRValue->val.iVal;
			pResult->eValType = (pResult->val.iVal < 0) 
										? XFLM_INT_VAL : XFLM_UINT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = fqGetInt64( pLValue);
		FLMINT64		i64RValue = fqGetInt64( pRValue);

		if( i64RValue)
		{
			pResult->val.i64Val = i64LValue % i64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
										? XFLM_INT64_VAL : XFLM_UINT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs the modulo operation
***************************************************************************/
FSTATIC void fqOpSUMod(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if( pRValue->val.uiVal)
		{
			pResult->val.iVal = pLValue->val.iVal % pRValue->val.uiVal;
			pResult->eValType = XFLM_INT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = fqGetInt64( pLValue);
		FLMUINT64	ui64RValue = fqGetUInt64( pRValue);

		if( ui64RValue)
		{
			pResult->val.i64Val = i64LValue  % ui64RValue;
			pResult->eValType = XFLM_INT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs an addition operation
***************************************************************************/
FSTATIC void fqOpUUPlus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{	
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		pResult->val.uiVal = pLValue->val.uiVal + pRValue->val.uiVal;
		pResult->eValType = XFLM_UINT_VAL;
	}
	else
	{
		pResult->val.ui64Val = 
			fqGetUInt64( pLValue) + fqGetUInt64( pRValue);
		pResult->eValType = XFLM_UINT64_VAL;
	}
}

/***************************************************************************
Desc:		Performs an addition operation
***************************************************************************/
FSTATIC void fqOpUSPlus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{	
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if( (pRValue->val.iVal >= 0) || 
			 (pLValue->val.uiVal > gv_uiMaxSignedIntVal))
		{
			pResult->val.uiVal = pLValue->val.uiVal + (FLMUINT)pRValue->val.iVal;
			pResult->eValType = XFLM_UINT_VAL;
		}
		else
		{
			pResult->val.iVal = (FLMINT)pLValue->val.uiVal + pRValue->val.iVal;
			pResult->eValType = (pResult->val.iVal < 0) 
									? XFLM_INT_VAL : XFLM_UINT_VAL;
		}
	}
	else
	{
		FLMUINT64		ui64LValue = fqGetUInt64( pLValue);
		FLMINT64			i64RValue = fqGetInt64( pRValue);

		if( (i64RValue >= 0) || (ui64LValue > gv_ui64MaxSignedIntVal))
		{
			pResult->val.ui64Val = ui64LValue + (FLMUINT64)i64RValue;
			pResult->eValType = XFLM_UINT64_VAL;
		}
		else
		{
			pResult->val.i64Val = (FLMINT64)ui64LValue + i64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
									? XFLM_INT64_VAL : XFLM_UINT64_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs an addition operation
***************************************************************************/
FSTATIC void fqOpSSPlus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		pResult->val.iVal = pLValue->val.iVal + pRValue->val.iVal;
		pResult->eValType = (pResult->val.iVal < 0) 
									? XFLM_INT_VAL : XFLM_UINT_VAL;
	}
	else
	{
		pResult->val.i64Val = 
			fqGetInt64( pLValue) + fqGetInt64( pRValue);
		pResult->eValType = (pResult->val.i64Val < 0) 
									? XFLM_INT64_VAL : XFLM_UINT64_VAL;
	}
}

/***************************************************************************
Desc:		Performs an addition operation
***************************************************************************/
FSTATIC void fqOpSUPlus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if( (pLValue->val.iVal >= 0) ||
			 (pRValue->val.uiVal > gv_uiMaxSignedIntVal))
		{
			pResult->val.uiVal = (FLMUINT)pLValue->val.iVal + pRValue->val.uiVal;
			pResult->eValType = XFLM_UINT_VAL;
		}
		else
		{
			pResult->val.iVal = pLValue->val.iVal + (FLMINT)pRValue->val.uiVal;
			pResult->eValType = (pResult->val.iVal < 0) 
									? XFLM_INT_VAL : XFLM_UINT_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = fqGetInt64( pLValue);
		FLMUINT64	ui64RValue = fqGetUInt64( pRValue);

		if( (i64LValue >= 0) || (ui64RValue > gv_ui64MaxSignedIntVal))
		{
			pResult->val.ui64Val = (FLMUINT64)i64LValue + ui64RValue;
			pResult->eValType = XFLM_UINT64_VAL;
		}
		else
		{
			pResult->val.i64Val = i64LValue + (FLMINT64)ui64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
									? XFLM_INT64_VAL : XFLM_UINT64_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs a subtraction operation
***************************************************************************/
FSTATIC void fqOpUUMinus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{	
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if( pLValue->val.uiVal >= pRValue->val.uiVal)
		{
			pResult->val.uiVal = pLValue->val.uiVal - pRValue->val.uiVal;
			pResult->eValType = XFLM_UINT_VAL;
		}
		else
		{
			pResult->val.iVal = (FLMINT)(pLValue->val.uiVal - pRValue->val.uiVal);
			pResult->eValType = XFLM_INT_VAL;
		}
	}
	else
	{
		FLMUINT64	ui64LValue = fqGetUInt64( pLValue);
		FLMUINT64	ui64RValue = fqGetUInt64( pRValue);

		if( ui64LValue >= ui64RValue)
		{
			pResult->val.ui64Val = ui64LValue - ui64RValue;
			pResult->eValType = XFLM_UINT64_VAL;
		}
		else
		{
			pResult->val.i64Val = (FLMINT64)(ui64LValue - ui64RValue);
			pResult->eValType = XFLM_INT64_VAL;
		}
	}
}
	
/***************************************************************************
Desc:		Performs a subtraction operation
***************************************************************************/
FSTATIC void fqOpUSMinus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{	
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if( pRValue->val.iVal < 0) 
		{
			pResult->val.uiVal = pLValue->val.uiVal - pRValue->val.iVal;
			pResult->eValType = XFLM_UINT_VAL;
		}
		else
		{
			pResult->val.iVal = (FLMINT)pLValue->val.uiVal - pRValue->val.iVal;
			pResult->eValType = (pResult->val.iVal < 0) 
									? XFLM_INT_VAL : XFLM_UINT_VAL;
		}
	}
	else
	{
		FLMUINT64	ui64LValue = fqGetUInt64( pLValue);
		FLMINT64		i64RValue = fqGetInt64( pRValue);

		if( i64RValue < 0)
		{
			pResult->val.ui64Val = ui64LValue - i64RValue;
			pResult->eValType = XFLM_UINT64_VAL;
		}
		else
		{
			pResult->val.i64Val = (FLMINT64)ui64LValue - i64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
									? XFLM_INT64_VAL : XFLM_UINT64_VAL;
		}
	}
}
	
/***************************************************************************
Desc:		Performs a subtraction operation
***************************************************************************/
FSTATIC void fqOpSSMinus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if(( pLValue->val.iVal > 0) && ( pRValue->val.iVal < 0))
		{
			pResult->val.uiVal = (FLMUINT)(pLValue->val.iVal - pRValue->val.iVal);
			pResult->eValType = XFLM_UINT_VAL;
		}
		else
		{
			pResult->val.iVal = pLValue->val.iVal - pRValue->val.iVal;
			pResult->eValType = (pResult->val.iVal < 0) 
									? XFLM_INT_VAL : XFLM_UINT_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = fqGetInt64( pLValue);
		FLMINT64		i64RValue = fqGetInt64( pRValue);

		if( (i64LValue > 0) && (i64RValue < 0))
		{
			pResult->val.ui64Val = (FLMUINT64)( i64LValue - i64RValue);
			pResult->eValType = XFLM_UINT64_VAL;
		}
		else
		{
			pResult->val.i64Val = i64LValue - i64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
									? XFLM_INT64_VAL : XFLM_UINT64_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs a subtraction operation
***************************************************************************/
FSTATIC void fqOpSUMinus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if( pRValue->val.uiVal > gv_uiMaxSignedIntVal)
		{
			pResult->val.iVal = (pLValue->val.iVal - gv_uiMaxSignedIntVal) - 
				(FLMINT)(pRValue->val.uiVal - gv_uiMaxSignedIntVal);
			pResult->eValType = XFLM_INT_VAL;
		}
		else
		{
			pResult->val.iVal = pLValue->val.iVal - (FLMINT)pRValue->val.uiVal;
			pResult->eValType = (pResult->val.iVal < 0) 
									? XFLM_INT_VAL : XFLM_UINT_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = fqGetInt64( pLValue);
		FLMUINT64	ui64RValue = fqGetUInt64( pRValue);

		if( ui64RValue > gv_ui64MaxSignedIntVal)
		{
			pResult->val.i64Val = (i64LValue - gv_ui64MaxSignedIntVal) -
				(FLMINT64)(ui64RValue - gv_ui64MaxSignedIntVal);
			pResult->eValType = XFLM_INT64_VAL;
		}
		else
		{
			pResult->val.i64Val = i64LValue - (FLMINT64)ui64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
									? XFLM_INT64_VAL : XFLM_UINT64_VAL;
		}
	}
}

/***************************************************************************
Desc:  	Compare two entire strings.
****************************************************************************/
FSTATIC RCODE fqCompareText(
	IF_OperandComparer *		pOpComparer,
	FQVALUE *					pLValue,
	FQVALUE *					pRValue,
	FLMUINT						uiCompareRules,
	FLMBOOL						bOpIsMatch,
	FLMUINT						uiLanguage,
	FLMINT *						piResult)
{
	RCODE							rc = NE_XFLM_OK;
	IF_BufferIStream *		pBufferLStream = NULL;
	IF_PosIStream *			pLStream;
	IF_BufferIStream *		pBufferRStream = NULL;
	IF_PosIStream *			pRStream;

	// Types must be text

	if (pLValue->eValType != XFLM_UTF8_VAL || 
		pRValue->eValType != XFLM_UTF8_VAL)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}

	// Open the streams

	if( !(pLValue->uiFlags & VAL_IS_STREAM))
	{
		if( RC_BAD( rc = FlmAllocBufferIStream( &pBufferLStream)))
		{
			goto Exit;
		}
		
		if (RC_BAD( rc = pBufferLStream->openStream( 
			(const char *)pLValue->val.pucBuf, pLValue->uiDataLen)))
		{
			goto Exit;
		}

		pLStream = pBufferLStream;
	}
	else
	{
		pLStream = pLValue->val.pIStream;
	}

	if( !(pRValue->uiFlags & VAL_IS_STREAM))
	{
		if( RC_BAD( rc = FlmAllocBufferIStream( &pBufferRStream)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = pBufferRStream->openStream( 
			(const char *)pRValue->val.pucBuf, pRValue->uiDataLen)))
		{
			goto Exit;
		}
		pRStream = pBufferRStream;
	}
	else
	{
		pRStream = pRValue->val.pIStream;
	}

	if (pOpComparer)
	{
		rc = pOpComparer->compare( pLStream, pRStream, piResult);
		goto Exit;
	}
	
	if( RC_BAD( rc = f_compareUTF8Streams( 
		pLStream, 
		(bOpIsMatch && (pLValue->uiFlags & VAL_IS_CONSTANT)) ? TRUE : FALSE,
		pRStream,
		(bOpIsMatch && (pRValue->uiFlags & VAL_IS_CONSTANT)) ? TRUE : FALSE,
		uiCompareRules, uiLanguage, piResult)))
	{
		goto Exit;
	}

Exit:

	if( pBufferLStream)
	{
		pBufferLStream->Release();
	}
	
	if( pBufferRStream)
	{
		pBufferRStream->Release();
	}
	
	return( rc);
}

/***************************************************************************
Desc:	Approximate compare - only works for strings right now.
****************************************************************************/
FSTATIC RCODE fqApproxCompare(
	FQVALUE *				pLValue,
	FQVALUE *				pRValue,
	FLMINT *					piResult)
{
	RCODE						rc = NE_XFLM_OK;
	FLMUINT					uiLMeta;
	FLMUINT					uiRMeta;
	FLMUINT64				ui64StartPos;
	IF_BufferIStream *	pBufferLStream = NULL;
	IF_PosIStream *		pLStream;
	IF_BufferIStream *	pBufferRStream = NULL;
	IF_PosIStream *		pRStream;

	// Types must be text

	if (pLValue->eValType != XFLM_UTF8_VAL ||
		 pRValue->eValType != XFLM_UTF8_VAL)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}

	// Open the streams

	if (!(pLValue->uiFlags & VAL_IS_STREAM))
	{
		if( RC_BAD( rc = FlmAllocBufferIStream( &pBufferLStream)))
		{
			goto Exit;
		}
		
		if (RC_BAD( rc = pBufferLStream->openStream( 
			(const char *)pLValue->val.pucBuf, pLValue->uiDataLen)))
		{
			goto Exit;
		}

		pLStream = pBufferLStream;
	}
	else
	{
		pLStream = pLValue->val.pIStream;
	}

	if (!(pRValue->uiFlags & VAL_IS_STREAM))
	{
		if( RC_BAD( rc = FlmAllocBufferIStream( &pBufferRStream)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = pBufferRStream->openStream( 
			(const char *)pRValue->val.pucBuf, pRValue->uiDataLen)))
		{
			goto Exit;
		}
		
		pRStream = pBufferRStream;
	}
	else
	{
		pRStream = pRValue->val.pIStream;
	}

	if ((pLValue->uiFlags & VAL_IS_CONSTANT) ||
		 !(pRValue->uiFlags & VAL_IS_CONSTANT))
	{
		for( ;;)
		{
			if( RC_BAD( rc = f_getNextMetaphone( pLStream, &uiLMeta)))
			{
				if( rc == NE_XFLM_EOF_HIT)
				{
					*piResult = 0;
					rc = NE_XFLM_OK;
				}
				goto Exit;
			}

			ui64StartPos = pRStream->getCurrPosition();

			for( ;;)
			{
				if( RC_BAD( rc = f_getNextMetaphone( pRStream, &uiRMeta)))
				{
					if( rc == NE_XFLM_EOF_HIT)
					{
						rc = NE_XFLM_OK;
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
				if( rc == NE_XFLM_EOF_HIT)
				{
					*piResult = 0;
					rc = NE_XFLM_OK;
				}
				goto Exit;
			}

			ui64StartPos = pLStream->getCurrPosition();

			for( ;;)
			{
				if( RC_BAD( rc = f_getNextMetaphone( pLStream, &uiLMeta)))
				{
					if( rc == NE_XFLM_EOF_HIT)
					{
						rc = NE_XFLM_OK;
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

	if( pBufferLStream)
	{
		pBufferLStream->Release();
	}
	
	if( pBufferRStream)
	{
		pBufferRStream->Release();
	}

	return( rc);
}
	
/***************************************************************************
Desc:	Performs binary comparison on two streams - may be text or binary,
		it really doesn't matter.  Returns XFLM_TRUE or XFLM_FALSE.
***************************************************************************/
FSTATIC RCODE fqCompareBinary(
	IF_OperandComparer *	pOpComparer,
	FQVALUE *				pLValue,
	FQVALUE *				pRValue,
	FLMINT *					piResult)
{
	RCODE						rc = NE_XFLM_OK;
	IF_BufferIStream *	pBufferLStream = NULL;
	IF_PosIStream *		pLStream;
	IF_BufferIStream *	pBufferRStream = NULL;
	IF_PosIStream *		pRStream;
	FLMBYTE					ucLByte;
	FLMBYTE					ucRByte;
	FLMUINT					uiOffset = 0;
	FLMBOOL					bLEmpty = FALSE;

	*piResult = 0;

	// Types must be binary

	if ( pLValue->eValType != XFLM_BINARY_VAL ||
		  pRValue->eValType != XFLM_BINARY_VAL)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}

	// Open the streams

	if( !(pLValue->uiFlags & VAL_IS_STREAM))
	{
		if( RC_BAD( rc = FlmAllocBufferIStream( &pBufferLStream)))
		{
			goto Exit;
		}
		
		if (RC_BAD( rc = pBufferLStream->openStream( 
			(const char *)pLValue->val.pucBuf, pLValue->uiDataLen)))
		{
			goto Exit;
		}

		pLStream = pBufferLStream;
	}
	else
	{
		pLStream = pLValue->val.pIStream;
	}

	if( !(pRValue->uiFlags & VAL_IS_STREAM))
	{
		if( RC_BAD( rc = pBufferRStream->openStream( 
			(const char *)pRValue->val.pucBuf, pRValue->uiDataLen)))
		{
			goto Exit;
		}
		pRStream = pBufferRStream;
	}
	else
	{
		pRStream = pRValue->val.pIStream;
	}

	if (pOpComparer)
	{
		rc = pOpComparer->compare( pLStream, pRStream, piResult);
		goto Exit;
	}

	for (;;)
	{
		if (RC_BAD( rc = flmReadStorageAsBinary( 
			pLStream, &ucLByte, 1, uiOffset, NULL)))
		{
			if (rc == NE_XFLM_EOF_HIT)
			{
				rc = NE_XFLM_OK;
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
			if (rc == NE_XFLM_EOF_HIT)
			{
				rc = NE_XFLM_OK;
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

	if( pBufferLStream)
	{
		pBufferLStream->Release();
	}
	
	if( pBufferRStream)
	{
		pBufferRStream->Release();
	}
	
	return( rc);
}

/***************************************************************************
Desc:	Compare two values.  This routine assumes that pValue1 and pValue2
		are non-null.
***************************************************************************/
RCODE fqCompare(
	FQVALUE *				pValue1,
	FQVALUE *				pValue2,
	FLMUINT					uiCompareRules,
	IF_OperandComparer *	pOpComparer,
	FLMUINT					uiLanguage,
	FLMINT *					piCmp)
{
	RCODE		rc = NE_XFLM_OK;

	// We have already called fqCanCompare, so no need to do it here

	switch (pValue1->eValType)
	{
		case XFLM_BOOL_VAL:
			*piCmp = pValue1->val.eBool > pValue2->val.eBool
					 ? 1
					 : pValue1->val.eBool < pValue2->val.eBool
						? -1
						: 0;
			break;
		case XFLM_UINT_VAL:
			switch (pValue2->eValType)
			{
				case XFLM_UINT_VAL:
					*piCmp = pValue1->val.uiVal > pValue2->val.uiVal
							 ? 1
							 : pValue1->val.uiVal < pValue2->val.uiVal
							   ? -1
								: 0;
					break;
				case XFLM_UINT64_VAL:
					*piCmp = (FLMUINT64)pValue1->val.uiVal > pValue2->val.ui64Val
							 ? 1
							 : (FLMUINT64)pValue1->val.uiVal < pValue2->val.ui64Val
								? -1
								: 0;
					break;
				case XFLM_INT_VAL:
					*piCmp = pValue2->val.iVal < 0 ||
							 pValue1->val.uiVal > (FLMUINT)pValue2->val.iVal
							 							 ? 1
							 : pValue1->val.uiVal < (FLMUINT)pValue2->val.iVal
							   ? -1
								: 0;
					break;
				case XFLM_INT64_VAL:
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
					rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
					goto Exit;
			}
			break;
		case XFLM_UINT64_VAL:
			switch (pValue2->eValType)
			{
				case XFLM_UINT_VAL:
					*piCmp = pValue1->val.ui64Val > (FLMUINT64)pValue2->val.uiVal
							 ? 1
							 : pValue1->val.ui64Val < (FLMUINT64)pValue2->val.uiVal
							   ? -1
								: 0;
					break;
				case XFLM_UINT64_VAL:
					*piCmp = pValue1->val.ui64Val > pValue2->val.ui64Val
							 ? 1
							 : pValue1->val.ui64Val < pValue2->val.ui64Val
							   ? -1
								: 0;
					break;
				case XFLM_INT_VAL:
					*piCmp = pValue2->val.iVal < 0 ||
							 pValue1->val.ui64Val > (FLMUINT64)pValue2->val.iVal
							 ? 1
							 : pValue1->val.ui64Val < (FLMUINT64)pValue2->val.iVal
							   ? -1
								: 0;
					break;
				case XFLM_INT64_VAL:
					*piCmp = pValue2->val.i64Val < 0 ||
							 pValue1->val.ui64Val > (FLMUINT64)pValue2->val.i64Val
							 ? 1
							 : pValue1->val.ui64Val < (FLMUINT64)pValue2->val.i64Val
							   ? -1
								: 0;
					break;
            default:
					rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
					goto Exit;
			}
			break;
		case XFLM_INT_VAL:
			switch (pValue2->eValType)
			{
				case XFLM_UINT_VAL:
					*piCmp = pValue1->val.iVal < 0 ||
							 (FLMUINT)pValue1->val.iVal < pValue2->val.uiVal
							 ? -1
							 : (FLMUINT)pValue1->val.iVal > pValue2->val.uiVal
							   ? 1
								: 0;
					break;
				case XFLM_UINT64_VAL:
					*piCmp = pValue1->val.iVal < 0 ||
							 (FLMUINT64)pValue1->val.iVal < pValue2->val.ui64Val
							 ? -1
							 : (FLMUINT64)pValue1->val.iVal > pValue2->val.ui64Val
							   ? 1
								: 0;
					break;
				case XFLM_INT_VAL:
					*piCmp = pValue1->val.iVal < pValue2->val.iVal
							 ? -1
							 : pValue1->val.iVal > pValue2->val.iVal
							   ? 1
								: 0;
					break;
				case XFLM_INT64_VAL:
					*piCmp = (FLMINT64)pValue1->val.iVal < pValue2->val.i64Val
							 ? -1
							 : (FLMINT64)pValue1->val.iVal > pValue2->val.i64Val
							   ? 1
								: 0;
					break;
            default:
					rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
					goto Exit;
			}
			break;
		case XFLM_INT64_VAL:
			switch (pValue2->eValType)
			{
				case XFLM_UINT_VAL:
					*piCmp = pValue1->val.i64Val < 0 ||
							 (FLMUINT64)pValue1->val.i64Val <
							 (FLMUINT64)pValue2->val.uiVal
							 ? -1
							 : (FLMUINT64)pValue1->val.i64Val >
							   (FLMUINT64)pValue2->val.uiVal
							   ? 1
								: 0;
					break;
				case XFLM_UINT64_VAL:
					*piCmp = pValue1->val.i64Val < 0 ||
							 (FLMUINT64)pValue1->val.i64Val < pValue2->val.ui64Val
							 ? -1
							 : (FLMUINT64)pValue1->val.i64Val > pValue2->val.ui64Val
							   ? 1
								: 0;
					break;
				case XFLM_INT_VAL:
					*piCmp = pValue1->val.i64Val < (FLMINT64)pValue2->val.iVal
							 ? -1
							 : pValue1->val.i64Val > (FLMINT64)pValue2->val.iVal
							   ? 1
								: 0;
					break;
				case XFLM_INT64_VAL:
					*piCmp = pValue1->val.i64Val < pValue2->val.i64Val
							 ? -1
							 : pValue1->val.i64Val > pValue2->val.i64Val
							   ? 1
								: 0;
					break;
				default:
					rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
					goto Exit;
			}
			break;
		case XFLM_BINARY_VAL:
			if (RC_BAD( rc = fqCompareBinary( pOpComparer, pValue1,
												pValue2, piCmp)))
			{
				goto Exit;
			}
			break;
		case XFLM_UTF8_VAL:
			if (RC_BAD( rc = fqCompareText( pOpComparer,
				pValue1, pValue2,
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

/***************************************************************************
Desc:	Do a comparison operator.
***************************************************************************/
RCODE fqCompareOperands(
	FLMUINT					uiLanguage,
	FQVALUE *				pLValue,
	FQVALUE *				pRValue,
	eQueryOperators		eOperator,
	FLMUINT					uiCompareRules,
	IF_OperandComparer *	pOpComparer,
	FLMBOOL					bNotted,
	XFlmBoolType *			peBool)
{
	RCODE			rc = NE_XFLM_OK;
	FLMINT		iCmp;

	if (!pLValue || pLValue->eValType == XFLM_MISSING_VAL ||
		 !pRValue || pRValue->eValType == XFLM_MISSING_VAL ||
		 !fqCanCompare( pLValue, pRValue))
	{
		*peBool = (bNotted ? XFLM_TRUE : XFLM_FALSE);
	}

	// At this point, both operands are known to be present and are of
	// types that can be compared.  The comparison
	// will therefore be performed according to the
	// operator specified.
	
	else
	{
		switch (eOperator)
		{
			case XFLM_EQ_OP:
			case XFLM_NE_OP:
				if (pLValue->eValType == XFLM_UTF8_VAL ||
					 pRValue->eValType == XFLM_UTF8_VAL)
				{
					if (RC_BAD( rc = fqCompareText( pOpComparer, pLValue, pRValue,
						uiCompareRules, TRUE, uiLanguage, &iCmp)))
					{
						goto Exit;
					}
				}
				else
				{
					if (RC_BAD( rc = fqCompare( pLValue, pRValue, 
						uiCompareRules, pOpComparer, uiLanguage, &iCmp)))
					{
						goto Exit;
					}
				}
				if (eOperator == XFLM_EQ_OP)
				{
					*peBool = (iCmp == 0 ? XFLM_TRUE : XFLM_FALSE);
				}
				else
				{
					*peBool = (iCmp != 0 ? XFLM_TRUE : XFLM_FALSE);
				}
				break;

			case XFLM_APPROX_EQ_OP:
				if (RC_BAD( rc = fqApproxCompare( pLValue, pRValue, &iCmp)))
				{
					goto Exit;
				}
				*peBool = (iCmp == 0 ? XFLM_TRUE : XFLM_FALSE);
				break;

			case XFLM_LT_OP:
				if (RC_BAD( rc = fqCompare( pLValue, pRValue, 
					uiCompareRules, pOpComparer, uiLanguage, &iCmp)))
				{
					goto Exit;
				}
				*peBool = (iCmp < 0 ? XFLM_TRUE : XFLM_FALSE);
				break;

			case XFLM_LE_OP:
				if (RC_BAD( rc = fqCompare( pLValue, pRValue, 
					uiCompareRules, pOpComparer, uiLanguage, &iCmp)))
				{
					goto Exit;
				}
				*peBool = (iCmp <= 0 ? XFLM_TRUE : XFLM_FALSE);
				break;

			case XFLM_GT_OP:
				if (RC_BAD( rc = fqCompare( pLValue, pRValue, 
					uiCompareRules, pOpComparer, uiLanguage, &iCmp)))
				{
					goto Exit;
				}
				*peBool = (iCmp > 0 ? XFLM_TRUE : XFLM_FALSE);
				break;

			case XFLM_GE_OP:
				if (RC_BAD( rc = fqCompare( pLValue, pRValue, 
					uiCompareRules, pOpComparer, uiLanguage, &iCmp)))
				{
					goto Exit;
				}
				*peBool = (iCmp >= 0 ? XFLM_TRUE : XFLM_FALSE);
				break;

			default:
				*peBool = XFLM_UNKNOWN;
				rc = RC_SET_AND_ASSERT( NE_XFLM_QUERY_SYNTAX);
				goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Do an arithmetic operator.
***************************************************************************/
RCODE fqArithmeticOperator(
	FQVALUE *			pLValue,
	FQVALUE *			pRValue,
	eQueryOperators	eOperator,
	FQVALUE *			pResult)
{
	RCODE					rc = NE_XFLM_OK;
	FQ_OPERATION *		fnOp;
	FLMUINT				uiOffset = 0;

	if( !isArithOp( eOperator))
	{
		rc = RC_SET( NE_XFLM_SYNTAX);
		goto Exit;
	}

	if (pLValue->eValType == XFLM_MISSING_VAL ||
		 pRValue->eValType == XFLM_MISSING_VAL)
	{
		pResult->eValType = XFLM_MISSING_VAL;
		goto Exit;
	}

	if( isUnsigned( pLValue))
	{
		if( isUnsigned( pRValue))
		{
			uiOffset = 0;
		}
		else if( isSigned( pRValue))
		{
			uiOffset = 1;
		}
		else
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
			goto Exit;
		}
	}
	else if( isSigned( pLValue))
	{
		if( isUnsigned( pRValue))
		{
			uiOffset = 2;
		}
		else if( isSigned( pRValue))
		{
			uiOffset = 3;
		}
		else
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
			goto Exit;
		}
	}

	fnOp = FQ_ArithOpTable[ ((((FLMUINT)eOperator) - 
					XFLM_FIRST_ARITH_OP) * 4) + uiOffset];
	fnOp( pLValue, pRValue, pResult);

Exit:

	return( rc);
}
