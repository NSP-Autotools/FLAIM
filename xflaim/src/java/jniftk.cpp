//------------------------------------------------------------------------------
// Desc:
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

#include "xflaim.h"
#include "jniftk.h"

/****************************************************************************
Desc:
****************************************************************************/
void ThrowError(
	RCODE				rc,
	JNIEnv *			pEnv)
{
	char 				szMsg[ 128];
	jclass 			class_XFlaimException;
	jmethodID 		id_Constructor;
	jobject 			Exception;
	
	f_sprintf( szMsg, "Error code from XFLAIM was %08X", (unsigned)rc);
	
	class_XFlaimException = pEnv->FindClass( "xflaim/XFlaimException");
	
	id_Constructor = pEnv->GetMethodID( class_XFlaimException,
							"<init>", "(ILjava/lang/String;)V");
	
	Exception = pEnv->NewObject( class_XFlaimException, id_Constructor,
										(jint)rc, pEnv->NewStringUTF( szMsg));
	
	pEnv->Throw( reinterpret_cast<jthrowable>(Exception));
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE getUniString(
	JNIEnv *		pEnv,
	jstring		sStr,
	F_DynaBuf *	pDynaBuf)
{
	RCODE						rc = NE_XFLM_OK;
	const FLMUNICODE *	puzStr = NULL;
	FLMUINT					uiStrCharCount;
	
	if (sStr)
	{
		puzStr = (const FLMUNICODE *)pEnv->GetStringChars( sStr, NULL);
		uiStrCharCount = (FLMUINT)pEnv->GetStringLength( sStr);
		if (RC_BAD( rc = pDynaBuf->appendData( puzStr,
									sizeof( FLMUNICODE) * uiStrCharCount)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = pDynaBuf->appendUniChar( 0)))
		{
			goto Exit;
		}
	}
	else
	{
		pDynaBuf->truncateData( 0);
	}
	
Exit:

	if (puzStr)
	{
		pEnv->ReleaseStringChars( sStr, puzStr);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE getUTF8String(
	JNIEnv *		pEnv,
	jstring		sStr,
	F_DynaBuf *	pDynaBuf)
{
	RCODE				rc = NE_XFLM_OK;
	const char *	pszStr = NULL;
	FLMUINT			uiStrCharCount;
	
	if (sStr)
	{
		pszStr = pEnv->GetStringUTFChars( sStr, NULL);
		uiStrCharCount = (FLMUINT)pEnv->GetStringUTFLength( sStr);
		if (RC_BAD( rc = pDynaBuf->appendData( pszStr, uiStrCharCount)))
		{
			goto Exit;
		}
	}
	else
	{
		pDynaBuf->truncateData( 0);
	}
	if (RC_BAD( rc = pDynaBuf->appendByte( 0)))
	{
		goto Exit;
	}
	
Exit:

	if (pszStr)
	{
		pEnv->ReleaseStringUTFChars( sStr, pszStr);
	}

	return( rc);
}

