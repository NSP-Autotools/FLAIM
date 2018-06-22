//------------------------------------------------------------------------------
// Desc: Native C routines to support C# DataVector class
// Tabs:	3
//
//	Copyright (c) 2006-2007 Novell, Inc. All Rights Reserved.
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

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DataVector_Release(
	IF_DataVector *	pDataVector)
{
	if (pDataVector)
	{
		pDataVector->Release();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DataVector_setDocumentID(
	IF_DataVector *	pDataVector,
	FLMUINT64			ui64DocumentID)
{
	pDataVector->setDocumentID( ui64DocumentID);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DataVector_setID(
	IF_DataVector *	pDataVector,
	FLMUINT32			ui32ElementNumber,
	FLMUINT64			ui64ID)
{
	return( pDataVector->setID( (FLMUINT)ui32ElementNumber, ui64ID));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DataVector_setNameId(
	IF_DataVector *	pDataVector,
	FLMUINT32			ui32ElementNumber,
	FLMUINT32			ui32NameId,
	FLMBOOL				bIsAttr,
	FLMBOOL				bIsData)
{
	return( pDataVector->setNameId( (FLMUINT)ui32ElementNumber,
					(FLMUINT)ui32NameId, bIsAttr, bIsData));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DataVector_setULong(
	IF_DataVector *	pDataVector,
	FLMUINT32			ui32ElementNumber,
	FLMUINT64			ui64Value)
{
	return( pDataVector->setUINT64( (FLMUINT)ui32ElementNumber, ui64Value));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DataVector_setLong(
	IF_DataVector *	pDataVector,
	FLMUINT32			ui32ElementNumber,
	FLMINT64				i64Value)
{
	return( pDataVector->setINT64( (FLMUINT)ui32ElementNumber, i64Value));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DataVector_setUInt(
	IF_DataVector *	pDataVector,
	FLMUINT32			ui32ElementNumber,
	FLMUINT32			ui32Value)
{
	return( pDataVector->setUINT( (FLMUINT)ui32ElementNumber, (FLMUINT)ui32Value));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DataVector_setInt(
	IF_DataVector *	pDataVector,
	FLMUINT32			ui32ElementNumber,
	FLMINT32				i32Value)
{
	return( pDataVector->setINT( (FLMUINT)ui32ElementNumber, (FLMINT)i32Value));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DataVector_setString(
	IF_DataVector *		pDataVector,
	FLMUINT32				ui32ElementNumber,
	const FLMUNICODE *	puzValue)
{
	return( pDataVector->setUnicode( (FLMUINT)ui32ElementNumber, puzValue));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DataVector_setBinary(
	IF_DataVector *	pDataVector,
	FLMUINT32			ui32ElementNumber,
	const void *		pvValue,
	FLMUINT32			ui32Len)
{
	return( pDataVector->setBinary( (FLMUINT)ui32ElementNumber, pvValue, (FLMUINT)ui32Len));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DataVector_setRightTruncated(
	IF_DataVector *	pDataVector,
	FLMUINT32			ui32ElementNumber)
{
	pDataVector->setRightTruncated( (FLMUINT)ui32ElementNumber);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DataVector_setLeftTruncated(
	IF_DataVector *	pDataVector,
	FLMUINT32			ui32ElementNumber)
{
	pDataVector->setLeftTruncated( (FLMUINT)ui32ElementNumber);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DataVector_clearRightTruncated(
	IF_DataVector *	pDataVector,
	FLMUINT32			ui32ElementNumber)
{
	pDataVector->clearRightTruncated( (FLMUINT)ui32ElementNumber);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DataVector_clearLeftTruncated(
	IF_DataVector *	pDataVector,
	FLMUINT32			ui32ElementNumber)
{
	pDataVector->clearLeftTruncated( (FLMUINT)ui32ElementNumber);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMBOOL XFLAPI xflaim_DataVector_isRightTruncated(
	IF_DataVector *	pDataVector,
	FLMUINT32			ui32ElementNumber)
{
	return( pDataVector->isRightTruncated( (FLMUINT)ui32ElementNumber));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMBOOL XFLAPI xflaim_DataVector_isLeftTruncated(
	IF_DataVector *	pDataVector,
	FLMUINT32			ui32ElementNumber)
{
	return( pDataVector->isLeftTruncated( (FLMUINT)ui32ElementNumber));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMUINT64 XFLAPI xflaim_DataVector_getDocumentID(
	IF_DataVector *	pDataVector)
{
	return( pDataVector->getDocumentID());
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMUINT64 XFLAPI xflaim_DataVector_getID(
	IF_DataVector *	pDataVector,
	FLMUINT32			ui32ElementNumber)
{
	return( pDataVector->getID( (FLMUINT)ui32ElementNumber));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMUINT32 XFLAPI xflaim_DataVector_getNameId(
	IF_DataVector *	pDataVector,
	FLMUINT32			ui32ElementNumber)
{
	return( (FLMUINT32)pDataVector->getNameId( (FLMUINT)ui32ElementNumber));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMBOOL XFLAPI xflaim_DataVector_isAttr(
	IF_DataVector *	pDataVector,
	FLMUINT32			ui32ElementNumber)
{
	return( pDataVector->isAttr( (FLMUINT)ui32ElementNumber));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMBOOL XFLAPI xflaim_DataVector_isDataComponent(
	IF_DataVector *	pDataVector,
	FLMUINT32			ui32ElementNumber)
{
	return( pDataVector->isDataComponent( (FLMUINT)ui32ElementNumber));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMBOOL XFLAPI xflaim_DataVector_isKeyComponent(
	IF_DataVector *	pDataVector,
	FLMUINT32			ui32ElementNumber)
{
	return( pDataVector->isKeyComponent( (FLMUINT)ui32ElementNumber));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMUINT32 XFLAPI xflaim_DataVector_getDataLength(
	IF_DataVector *	pDataVector,
	FLMUINT32			ui32ElementNumber)
{
	return( (FLMUINT32)pDataVector->getDataLength( (FLMUINT)ui32ElementNumber));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMUINT32 XFLAPI xflaim_DataVector_getDataType(
	IF_DataVector *	pDataVector,
	FLMUINT32			ui32ElementNumber)
{
	return( (FLMUINT32)pDataVector->getDataType( (FLMUINT)ui32ElementNumber));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DataVector_getULong(
	IF_DataVector *	pDataVector,
	FLMUINT32			ui32ElementNumber,
	FLMUINT64 *			pui64Value)
{
	return( pDataVector->getUINT64( (FLMUINT)ui32ElementNumber, pui64Value));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DataVector_getLong(
	IF_DataVector *	pDataVector,
	FLMUINT32			ui32ElementNumber,
	FLMINT64 *			pi64Value)
{
	return( pDataVector->getINT64( (FLMUINT)ui32ElementNumber, pi64Value));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DataVector_getUInt(
	IF_DataVector *	pDataVector,
	FLMUINT32			ui32ElementNumber,
	FLMUINT32 *			pui32Value)
{
	RCODE		rc;
	FLMUINT	uiValue;
	
	rc = pDataVector->getUINT( (FLMUINT)ui32ElementNumber, &uiValue);
	*pui32Value = (FLMUINT32)uiValue;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DataVector_getInt(
	IF_DataVector *	pDataVector,
	FLMUINT32			ui32ElementNumber,
	FLMINT32 *			pi32Value)
{
	RCODE		rc;
	FLMINT	iValue;
	
	rc = pDataVector->getINT( (FLMUINT)ui32ElementNumber, &iValue);
	*pi32Value = (FLMINT32)iValue;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DataVector_getString(
	IF_DataVector *	pDataVector,
	FLMUINT32			ui32ElementNumber,
	FLMUNICODE **		ppuzValue)
{
	return( pDataVector->getUnicode( (FLMUINT)ui32ElementNumber, ppuzValue));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DataVector_getBinary(
	IF_DataVector *	pDataVector,
	FLMUINT32			ui32ElementNumber,
	FLMUINT32			ui32Len,
	void *				pvValue)
{
	RCODE		rc = NE_XFLM_OK;
	FLMUINT	uiLength = (FLMUINT)ui32Len;
	
	if (RC_BAD( rc = pDataVector->getBinary( (FLMUINT)ui32ElementNumber,
		pvValue, &uiLength)))
	{
		goto Exit;
	}
	flmAssert( uiLength == (FLMUINT)ui32Len);

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DataVector_outputKey(
	IF_DataVector *	pDataVector,
	IF_Db *				pDb,
	FLMUINT32			ui32IndexNum,
	FLMBOOL				bOutputIds,
	FLMBYTE *			pucKey,
	FLMINT32 *			pi32Len)
{
	RCODE		rc = NE_XFLM_OK;
	FLMUINT	uiLength;

	if (RC_BAD( rc = pDataVector->outputKey( pDb, (FLMUINT)ui32IndexNum,
			bOutputIds, pucKey, XFLM_MAX_KEY_SIZE, &uiLength)))
	{
		goto Exit;
	}
	*pi32Len = (FLMINT32)uiLength;
	
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DataVector_outputData(
	IF_DataVector *	pDataVector,
	IF_Db *				pDb,
	FLMUINT32			ui32IndexNum,
	FLMBYTE *			pucData,
	FLMINT32				i32BufSize,
	FLMINT32 *			pi32Len)
{
	RCODE		rc = NE_XFLM_OK;
	FLMUINT	uiLength;

	if (RC_BAD( rc = pDataVector->outputData( pDb, (FLMUINT)ui32IndexNum,
			pucData, (FLMUINT)i32BufSize, &uiLength)))
	{
		goto Exit;
	}
	*pi32Len = (FLMINT32)uiLength;
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DataVector_inputKey(
	IF_DataVector *	pDataVector,
	IF_Db *				pDb,
	FLMUINT32			ui32IndexNum,
	FLMBYTE *			pucKey,
	FLMINT32				i32KeyLen)
{
	return( pDataVector->inputKey( pDb, (FLMUINT)ui32IndexNum,
			pucKey, (FLMUINT)i32KeyLen));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DataVector_inputData(
	IF_DataVector *	pDataVector,
	IF_Db *				pDb,
	FLMUINT32			ui32IndexNum,
	FLMBYTE *			pucData,
	FLMINT32				i32DataLen)
{
	return( pDataVector->inputData( pDb, (FLMUINT)ui32IndexNum,
			pucData, (FLMUINT)i32DataLen));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DataVector_reset(
	IF_DataVector *	pDataVector)
{
	pDataVector->reset();
}
