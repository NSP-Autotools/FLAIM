//-------------------------------------------------------------------------
// Desc:	Routines to handle numbers.
// Tabs:	3
//
// Copyright (c) 1999-2001, 2003-2007 Novell, Inc. All Rights Reserved.
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

/****************************************************************************
Desc:		Given an unsigned number create the matching FLAIM-specific BCD
			number.
Note:		If terminating byte is half-full, low-nibble value is
			undefined.  Example: -125 creates B1-25-FX
Method:	Using a MOD algorithm, stack BCD values -- popping to
			destination reverses the order for correct final sequence
****************************************************************************/
FLMEXP RCODE FLMAPI FlmUINT2Storage(
	FLMUINT		uiNum,
	FLMUINT *	puiBufLength,
	FLMBYTE *	pBuf)
{
	FLMBYTE		ucNibStk[ F_MAX_NUM64_BUF + 1];
	FLMBYTE *	pucNibStk;
#ifdef FLM_DEBUG
	FLMBYTE *	pucNibStkEnd = &ucNibStk[ sizeof( ucNibStk)];
#endif

	flmAssert( *puiBufLength >= F_MAX_NUM_BUF);

	// push spare (undefined) nibble for possible half-used terminating byte

	pucNibStk = &ucNibStk[ 1];

	// push terminator nibble -- popped last

	*pucNibStk++ = 0x0F;

	// push digits
	// do 32 bit division until we get down to last digit

	while( uiNum >= 10)
	{
		// push BCD nibbles in reverse order
		
		*pucNibStk++ = (FLMBYTE)(uiNum % 10);
		uiNum /= 10;
	}
	
	// push last nibble of number
	
	*pucNibStk++ = (FLMBYTE)uiNum;
	f_assert( pucNibStk <= pucNibStkEnd);

	// count: nibbleCount / 2 and truncate

	*puiBufLength =  ((pucNibStk - ucNibStk) >> 1);		

	// Pop stack and pack nibbles into byte stream a pair at a time

	do
	{
		*pBuf++ = (FLMBYTE)((pucNibStk[ -1] << 4) | pucNibStk[ -2]);
	}
	while( (pucNibStk -= 2) > &ucNibStk[ 1]);

	return( FERR_OK);
}

/****************************************************************************
Desc:		Given a 64 bit unsigned number create the matching FLAIM-specific BCD
			number.
Note:		If terminating byte is half-full, low-nibble value is
			undefined.  Example: -125 creates B1-25-FX
Method:	Using a MOD algorithm, stack BCD values -- popping to
			destination reverses the order for correct final sequence
****************************************************************************/
FLMEXP RCODE FLMAPI FlmUINT64ToStorage(
	FLMUINT64	ui64Num,
	FLMUINT *	puiBufLength,
	FLMBYTE *	pBuf)
{
	FLMBYTE		ucNibStk[ F_MAX_NUM64_BUF + 1];
	FLMBYTE *	pucNibStk;

	flmAssert( *puiBufLength >= F_MAX_NUM64_BUF);

	// push spare (undefined) nibble for possible half-used terminating byte

	pucNibStk = &ucNibStk[ 1];

	// push terminator nibble -- popped last

	*pucNibStk++ = 0x0F;

	// push digits
	// do 64 bit division until we get down to last digit.

	while( ui64Num >= 10)
	{
		// push BCD nibbles in reverse order
		
		*pucNibStk++ = (FLMBYTE)(ui64Num % 10);
		ui64Num /= 10;
	}
	
	// push last nibble of number
	
	*pucNibStk++ = (FLMBYTE)ui64Num;

	// count: nibbleCount / 2 and truncate

	*puiBufLength = ((pucNibStk - ucNibStk) >> 1);		

	// Pop stack and pack nibbles into byte stream a pair at a time

	do
	{
		*pBuf++ = (FLMBYTE)((pucNibStk[ -1] << 4) | pucNibStk[ -2]);
	}
	while( (pucNibStk -= 2) > &ucNibStk[ 1]);

	return( FERR_OK);
}

/****************************************************************************
Desc: 	Given an signed number create the matching FLAIM-specific BCD
			number.
Note:		If terminating byte is half-full, low-nibble value is
			undefined.  Example: -125 creates B1-25-FX
Method:	Using a MOD algorithm, stack BCD values -- popping to
			destination reverses the order for correct final sequence
****************************************************************************/
FLMEXP RCODE FLMAPI FlmINT2Storage(
	FLMINT		iNum,
	FLMUINT *	puiBufLength,
	FLMBYTE *	pBuf)
{
	FLMUINT		uiNum;
	FLMBYTE		ucNibStk[ F_MAX_NUM64_BUF + 1];
	FLMBYTE *	pucNibStk;
	FLMBOOL		bNegFlag;
#ifdef FLM_DEBUG
	FLMBYTE *	pucNibStkEnd = &ucNibStk[ sizeof( ucNibStk)];
#endif

	flmAssert( *puiBufLength >= F_MAX_NUM_BUF);

	pucNibStk = &ucNibStk[ 1];
	*pucNibStk++ = 0x0F;

	if (iNum < 0)
	{
		bNegFlag = TRUE;
		if (iNum == FLM_MIN_INT)
		{
			uiNum = (FLMUINT)(FLM_MAX_INT) + 1;
		}
		else
		{
			uiNum = (FLMUINT)(-iNum);
		}
	}
	else
	{
		bNegFlag = FALSE;
		uiNum = (FLMUINT)iNum;
	}

	while( uiNum >= 10)
	{
		*pucNibStk++ = (FLMBYTE)(uiNum % 10);
		uiNum /= 10;
	}
	
	*pucNibStk++ = (FLMBYTE)uiNum;

	if( bNegFlag)
	{
		*pucNibStk++ = 0x0B;
	}

	f_assert( pucNibStk <= pucNibStkEnd);
	*puiBufLength = ((pucNibStk - ucNibStk) >> 1); 	

	do
	{
		*pBuf++ = (FLMBYTE)((pucNibStk[ -1] << 4) | pucNibStk[ -2]);
	}
	while( (pucNibStk -= 2) > &ucNibStk[ 1]);

	return( FERR_OK);
}

/****************************************************************************
Desc: 	Given a 64 bit signed number create the matching FLAIM-specific BCD
			number.
Note:		If terminating byte is half-full, low-nibble value is
			undefined.  Example: -125 creates B1-25-FX
Method:	Using a MOD algorithm, stack BCD values -- popping to
			destination reverses the order for correct final sequence
****************************************************************************/
FLMEXP RCODE FLMAPI FlmINT64ToStorage(
	FLMINT64		i64Num,
	FLMUINT *	puiBufLength,
	FLMBYTE *	pBuf)
{
	FLMUINT64	ui64Num;
	FLMBYTE		ucNibStk[ F_MAX_NUM64_BUF + 1];
	FLMBYTE *	pucNibStk;
	FLMBOOL		bNegFlag;

	flmAssert( *puiBufLength >= F_MAX_NUM64_BUF);

	pucNibStk = &ucNibStk[ 1];
	*pucNibStk++ = 0x0F;

	if (i64Num < 0)
	{
		bNegFlag = TRUE;
		if (i64Num == FLM_MIN_INT64)
		{
			ui64Num = (FLMUINT64)(FLM_MAX_INT64) + 1;
		}
		else
		{
			ui64Num = (FLMUINT64)(-i64Num);
		}
	}
	else
	{
		bNegFlag = FALSE;
		ui64Num = (FLMUINT64)i64Num;
	}

	while( ui64Num >= 10)
	{
		*pucNibStk++ = (FLMBYTE)(ui64Num % 10);
		ui64Num /= 10;
	}
	
	*pucNibStk++ = (FLMBYTE)ui64Num;

	if (bNegFlag)
	{
		*pucNibStk++ = 0x0B;
	}

	*puiBufLength = ((pucNibStk - ucNibStk) >> 1); 	

	do
	{
		*pBuf++ = (FLMBYTE)((pucNibStk[ -1] << 4) | pucNibStk[ -2]);
	}
	while( (pucNibStk -= 2) > &ucNibStk[ 1]);

	return( FERR_OK);
}

/****************************************************************************
Desc: 	Returns a signed value from a BCD value.
			The data may be a number type, or context type. 
****************************************************************************/
FLMEXP RCODE FLMAPI FlmStorage2INT(
	FLMUINT				uiValueType,
	FLMUINT				uiValueLength,
	const FLMBYTE *	pucValue,
	FLMINT *				piNum)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiNum;
	FLMBOOL	bNegFlag;

	if( RC_OK(rc = flmBcd2Num( uiValueType, uiValueLength, pucValue,
							&uiNum, &bNegFlag)))
	{
		if (bNegFlag)
		{
			
			// If bNegFlag is set, we will have already checked to make sure
			// the value is in range inside of flmBcd2Num.

			if (uiNum == (FLMUINT)(FLM_MAX_INT) + 1)
			{
				*piNum = FLM_MIN_INT;
			}
			else
			{
				*piNum = -((FLMINT)uiNum);
			}
		}
		
		// If the value is positive, we will have checked to make sure the
		// number did not overflow FLM_MAX_UINT, but not FLM_MAX_INT.
		
		else if (uiNum > (FLMUINT)(FLM_MAX_INT))
		{
			rc = RC_SET( FERR_CONV_NUM_OVERFLOW);
		}
		else
		{
			*piNum = (FLMINT)uiNum;
		}
	}
	
	return( rc);
}

/****************************************************************************
Desc: 	Returns a signed value from a BCD value.
			The data may be a number type, or context type. 
****************************************************************************/
FLMEXP RCODE FLMAPI FlmStorage2INT32(
	FLMUINT				uiValueType,
	FLMUINT				uiValueLength,
	const FLMBYTE *	pucValue,
	FLMINT32 *			pi32Num)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiNum;
	FLMBOOL	bNegFlag;

	if( RC_OK(rc = flmBcd2Num( uiValueType, uiValueLength, pucValue,
							&uiNum, &bNegFlag)))
	{
		if (bNegFlag)
		{
			if (uiNum > (FLMUINT)(FLM_MAX_INT32) + 1)
			{
				rc = RC_SET( FERR_CONV_NUM_UNDERFLOW);
			}
			else if (uiNum == (FLMUINT)(FLM_MAX_INT32) + 1)
			{
				*pi32Num = FLM_MIN_INT32;
			}
			else
			{
				*pi32Num = -((FLMINT32)uiNum);
			}
		}
		
		// If the value is positive, we will have checked to make sure the
		// number did not overflow FLM_MAX_UINT, but not FLM_MAX_INT32.
		
		else if (uiNum > (FLMUINT)(FLM_MAX_INT32))
		{
			rc = RC_SET( FERR_CONV_NUM_OVERFLOW);
		}
		else
		{
			*pi32Num = (FLMINT32)uiNum;
		}
	}
	
	return( rc);
}

/****************************************************************************
Desc: 	Returns a 64 bit signed value from a BCD value.
			The data may be a number type, or context type. 
****************************************************************************/
FLMEXP RCODE FLMAPI FlmStorage2INT64(
	FLMUINT				uiValueType,
	FLMUINT				uiValueLength,
	const FLMBYTE *	pucValue,
	FLMINT64 *			pi64Num)
{
	RCODE			rc = FERR_OK;
	FLMUINT64	ui64Num;
	FLMBOOL		bNegFlag;

	if( RC_OK(rc = flmBcd2Num64( uiValueType, uiValueLength, pucValue,
							&ui64Num, &bNegFlag)))
	{
		if (bNegFlag)
		{
			// If bNegFlag is set, we will have already checked to make sure
			// the value is in range inside of flmBcd2Num64.
			
			if (ui64Num == (FLMUINT64)(FLM_MAX_INT64) + 1)
			{
				*pi64Num = FLM_MIN_INT64;
			}
			else
			{
				*pi64Num = -((FLMINT64)ui64Num);
			}
		}
		
		// If the value is positive, we will have checked to make sure the
		// number did not overflow FLM_MAX_UINT64, but not FLM_MAX_INT64.
		
		else if (ui64Num > (FLMUINT64)(FLM_MAX_INT64))
		{
			rc = RC_SET( FERR_CONV_NUM_OVERFLOW);
		}
		else
		{
			*pi64Num = (FLMINT64)ui64Num;
		}
	}
	
	return( rc);
}

/****************************************************************************
Desc: 	Returns a unsigned value from a BCD value.
			The data may be a number type, or context type. 
****************************************************************************/
FLMEXP RCODE FLMAPI FlmStorage2UINT(
	FLMUINT				uiValueType,
	FLMUINT				uiValueLength,
	const FLMBYTE *	pucValue,
	FLMUINT *			puiNum)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bNegFlag;

	if( RC_OK( rc = flmBcd2Num( uiValueType, uiValueLength, pucValue,
								puiNum, &bNegFlag)))
	{
		if (bNegFlag)
		{
			rc = RC_SET( FERR_CONV_NUM_UNDERFLOW);
		}
	}
	
	return( rc);
}

/****************************************************************************
Desc: 	Returns a unsigned value from a BCD value.
			The data may be a number type, or context type. 
****************************************************************************/
FLMEXP RCODE FLMAPI FlmStorage2UINT32(
	FLMUINT				uiValueType,
	FLMUINT				uiValueLength,
	const FLMBYTE *	pucValue,
	FLMUINT32 *			pui32Num)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiNum;
	FLMBOOL	bNegFlag;

	if( RC_OK(rc = flmBcd2Num( uiValueType, uiValueLength, pucValue,
							&uiNum, &bNegFlag)))
	{
		if (bNegFlag)
		{
			rc = RC_SET( FERR_CONV_NUM_UNDERFLOW);
		}
		
// On 64 bit platforms FLM_MAX_UINT32 will be less than FLM_MAX_UINT
// so we need to test against it.  Otherwise, we have already tested
// against FLM_MAX_UINT, and it is the same as FLM_MAX_UINT32, so there
// is no need to test against it.

#ifdef FLM_64BIT
		else if (uiNum > FLM_MAX_UINT32)
		{
			rc = RC_SET( FERR_CONV_NUM_OVERFLOW);
		}
#endif

		else 
		{
			*pui32Num = (FLMUINT32)uiNum;
		}
	}
	
	return( rc);
}


/****************************************************************************
Desc: 	Returns a unsigned value from a BCD value.
			The data may be a number type, or context type. 
****************************************************************************/
FLMEXP RCODE FLMAPI FlmStorage2UINT64(
	FLMUINT				uiValueType,
	FLMUINT				uiValueLength,
	const FLMBYTE *	pucValue,
	FLMUINT64 *			pui64Num)
{
	RCODE					rc = FERR_OK;
	FLMBOOL				bNegFlag;

	if( RC_OK(rc = flmBcd2Num64( uiValueType, uiValueLength, pucValue,
							pui64Num, &bNegFlag)))
	{
		if (bNegFlag)
		{
			rc = RC_SET( FERR_CONV_NUM_UNDERFLOW);
		}
	}
	
	return( rc);
}


/****************************************************************************
Desc: 	Converts FT_NUMBER and FT_CONTEXT storage buffers to a number
****************************************************************************/
RCODE flmBcd2Num(
	FLMUINT				uiValueType,
	FLMUINT				uiValueLength,
	const FLMBYTE *	pucValue,
	FLMUINT *			puiNum,
	FLMBOOL *			pbNegFlag)
{
	RCODE			rc = FERR_OK;
	FLMUINT 		uiTotalNum;
	FLMUINT		uiByte;
	FLMUINT		uiNibble;
	FLMUINT		uiMaxBeforeMultValue;
	FLMUINT		uiMaxValue;
	
	if (pucValue == NULL)
	{
		rc = RC_SET( FERR_CONV_NULL_SRC);
		goto Exit;
	}

	switch (uiValueType)
	{
		case FLM_NUMBER_TYPE:
		{
			uiTotalNum = 0;
			if ((*pucValue & 0xF0) == 0xB0)
			{
				*pbNegFlag = TRUE;
				uiNibble = 1;
				uiMaxBeforeMultValue = ((FLMUINT)(FLM_MAX_INT) + 1) / 10;
				uiMaxValue = ((FLMUINT)(FLM_MAX_INT) + 1);
			}
			else
			{
				*pbNegFlag = FALSE;
				uiNibble = 0;
				uiMaxBeforeMultValue = (FLM_MAX_UINT) / 10;
				uiMaxValue = FLM_MAX_UINT;
			}

			// Get each nibble and use to create the number

			while (uiValueLength)
			{
				
				// An odd value for uiNibble means we are on the 2nd nibble of
				// the byte.
				
				if (uiNibble & 1)
				{
					uiByte = (FLMINT)(*pucValue & 0x0F);
					pucValue++;
					uiValueLength--;
				}
				else
				{
					uiByte = (FLMUINT)(*pucValue >> 4);
				}
				uiNibble++;
				if (uiByte == 0x0F)
				{
					break;
				}

				if (uiTotalNum > uiMaxBeforeMultValue)
				{
					if (*pbNegFlag)
					{
						rc = RC_SET( FERR_CONV_NUM_UNDERFLOW);
					}
					else
					{
						rc = RC_SET( FERR_CONV_NUM_OVERFLOW);
					}
					goto Exit;
				}
				uiTotalNum = (uiTotalNum << 3) + (uiTotalNum << 1);
				if (uiTotalNum > uiMaxValue - uiByte)
				{
					if (*pbNegFlag)
					{
						rc = RC_SET( FERR_CONV_NUM_UNDERFLOW);
					}
					else
					{
						rc = RC_SET( FERR_CONV_NUM_OVERFLOW);
					}
					goto Exit;
				}
				uiTotalNum += uiByte;
			}
			*puiNum = uiTotalNum;
			break;
		}

		case FLM_TEXT_TYPE : 
		{
			uiTotalNum = 0;
			if (*pucValue == '-')
			{
				*pbNegFlag = TRUE;
				uiMaxBeforeMultValue = ((FLMUINT)(FLM_MAX_INT) + 1) / 10;
				uiMaxValue = (FLMUINT)(FLM_MAX_INT) + 1;
			}
			else
			{
				*pbNegFlag = FALSE;
				uiMaxBeforeMultValue = (FLM_MAX_UINT) / 10;
				uiMaxValue = FLM_MAX_UINT;
			}
			while (uiValueLength--)
			{
				if( *pucValue < ASCII_ZERO || *pucValue > ASCII_NINE)
				{
					break;
				}
				uiByte = (FLMUINT)(*pucValue - ASCII_ZERO);
				
				if (uiTotalNum > uiMaxBeforeMultValue)
				{
					if (*pbNegFlag)
					{
						rc = RC_SET( FERR_CONV_NUM_UNDERFLOW);
					}
					else
					{
						rc = RC_SET( FERR_CONV_NUM_OVERFLOW);
					}
					goto Exit;
				}
				uiTotalNum = (uiTotalNum << 3) + (uiTotalNum << 1);
				if (uiTotalNum > uiMaxValue - uiByte)
				{
					if (*pbNegFlag)
					{
						rc = RC_SET( FERR_CONV_NUM_UNDERFLOW);
					}
					else
					{
						rc = RC_SET( FERR_CONV_NUM_OVERFLOW);
					}
					goto Exit;
				}
				uiTotalNum += uiByte;
				pucValue++;
			}
			
			*puiNum = uiTotalNum;
			break;
		}

		case FLM_CONTEXT_TYPE :
		{
			if (uiValueLength == sizeof( FLMUINT32))
			{
				*puiNum = (FLMUINT)( FB2UD( pucValue));
				*pbNegFlag = FALSE;
			}
			
			break;
		}

		default:
		{
			flmAssert( 0);
			return( RC_SET( FERR_CONV_ILLEGAL));
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc: 	Converts FT_NUMBER and FT_CONTEXT storage buffers to a 64 bit number
****************************************************************************/
RCODE flmBcd2Num64(
	FLMUINT				uiValueType,
	FLMUINT				uiValueLength,
	const FLMBYTE *	pucValue,
	FLMUINT64 *			pui64Num,
	FLMBOOL *			pbNegFlag)
{
	RCODE			rc = FERR_OK;
	FLMUINT64 	ui64TotalNum;
	FLMUINT		uiByte;
	FLMUINT		uiNibble;
	FLMUINT64	ui64MaxBeforeMultValue;
	FLMUINT64	ui64MaxValue;
	
	if (pucValue == NULL)
	{
		rc = RC_SET( FERR_CONV_NULL_SRC);
		goto Exit;
	}

	switch (uiValueType)
	{
		case FLM_NUMBER_TYPE:
		{
			ui64TotalNum = 0;
			if ((*pucValue & 0xF0) == 0xB0)
			{
				*pbNegFlag = TRUE;
				uiNibble = 1;
				ui64MaxBeforeMultValue = ((FLMUINT64)(FLM_MAX_INT64) + 1) / 10;
				ui64MaxValue = (FLMUINT64)(FLM_MAX_INT64) + 1;
			}
			else
			{
				*pbNegFlag = FALSE;
				uiNibble = 0;
				ui64MaxBeforeMultValue = (FLM_MAX_UINT64) / 10;
				ui64MaxValue = FLM_MAX_UINT64;
			}

			// Get each nibble and use to create the number

			while (uiValueLength)
			{
				
				// An odd value for uiNibble means we are on the 2nd nibble of
				// the byte.
				
				if (uiNibble & 1)
				{
					uiByte = (FLMINT)(*pucValue & 0x0F);
					pucValue++;
					uiValueLength--;
				}
				else
				{
					uiByte = (FLMUINT)(*pucValue >> 4);
				}
				uiNibble++;
				if (uiByte == 0x0F)
				{
					break;
				}

				if (ui64TotalNum > ui64MaxBeforeMultValue)
				{
					if (*pbNegFlag)
					{
						rc = RC_SET( FERR_CONV_NUM_UNDERFLOW);
					}
					else
					{
						rc = RC_SET( FERR_CONV_NUM_OVERFLOW);
					}
					goto Exit;
				}
				ui64TotalNum = (ui64TotalNum << 3) + (ui64TotalNum << 1);
				if (ui64TotalNum > ui64MaxValue - (FLMUINT64)uiByte)
				{
					if (*pbNegFlag)
					{
						rc = RC_SET( FERR_CONV_NUM_UNDERFLOW);
					}
					else
					{
						rc = RC_SET( FERR_CONV_NUM_OVERFLOW);
					}
					goto Exit;
				}
				ui64TotalNum += (FLMUINT64)uiByte;
			}
			*pui64Num = ui64TotalNum;
			break;
		}

		case FLM_TEXT_TYPE : 
		{
			ui64TotalNum = 0;
			if (*pucValue == '-')
			{
				*pbNegFlag = TRUE;
				ui64MaxBeforeMultValue = ((FLMUINT64)(FLM_MAX_INT64) + 1) / 10;
				ui64MaxValue = (FLMUINT64)(FLM_MAX_INT64) + 1;
			}
			else
			{
				*pbNegFlag = FALSE;
				ui64MaxBeforeMultValue = (FLM_MAX_UINT64) / 10;
				ui64MaxValue = FLM_MAX_UINT64;
			}
			while (uiValueLength--)
			{
				if( *pucValue < ASCII_ZERO || *pucValue > ASCII_NINE)
				{
					break;
				}
				uiByte = (FLMUINT)(*pucValue - ASCII_ZERO);
				
				if (ui64TotalNum > ui64MaxBeforeMultValue)
				{
					if (*pbNegFlag)
					{
						rc = RC_SET( FERR_CONV_NUM_UNDERFLOW);
					}
					else
					{
						rc = RC_SET( FERR_CONV_NUM_OVERFLOW);
					}
					goto Exit;
				}
				ui64TotalNum = (ui64TotalNum << 3) + (ui64TotalNum << 1);
				if (ui64TotalNum > ui64MaxValue - (FLMUINT64)uiByte)
				{
					if (*pbNegFlag)
					{
						rc = RC_SET( FERR_CONV_NUM_UNDERFLOW);
					}
					else
					{
						rc = RC_SET( FERR_CONV_NUM_OVERFLOW);
					}
					goto Exit;
				}
				ui64TotalNum += (FLMUINT64)uiByte;
				pucValue++;
			}
			
			*pui64Num = ui64TotalNum;
			break;
		}

		case FLM_CONTEXT_TYPE :
		{
			if (uiValueLength == sizeof( FLMUINT32))
			{
				*pui64Num = (FLMUINT64)( FB2UD( pucValue));
				*pbNegFlag = FALSE;
			}
			
			break;
		}

		default:
		{
			flmAssert( 0);
			return( RC_SET( FERR_CONV_ILLEGAL));
		}
	}
	
Exit:

	return( rc);
}

