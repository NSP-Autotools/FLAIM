//-------------------------------------------------------------------------
// Desc:	Encryption/decryption methods for interfacing to NICI.
// Tabs:	3
//
// Copyright (c) 2004-2007 Novell, Inc. All Rights Reserved.
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

#ifdef FLM_USE_NICI
	FSTATIC void GetIV(
		FLMBYTE *		pucIV,
		FLMUINT			uiLen);
#endif

/****************************************************************************
Desc:	wrapNiciKey - Save the wrapped key in m_pKey.  NOTE:  Make sure
		there is a buffer allocated for the wrapped key (m_pucWrappedKey).
****************************************************************************/
F_CCS::~F_CCS()
{

	if (m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}

#ifdef FLM_USE_NICI
	if ( m_keyHandle)
	{
		NICI_CC_HANDLE			context = 0;
		
		// Create NICI Context
		
		if( RC_OK( CCS_CreateContext(0, &context)))
		{
			// Get rid of the key handle.
			
			CCS_DestroyObject( context, m_keyHandle);
			CCS_DestroyContext( context);
		}
		else
		{
			flmAssert( 0);
		}
	}
#endif
}

/****************************************************************************
Desc:	wrapNiciKey - Save the wrapped key in m_pKey.  NOTE:  Make sure
		there is a buffer allocated for the wrapped key (m_pucWrappedKey).
****************************************************************************/
RCODE F_CCS::wrapKey(
	FLMBYTE **				ppucWrappedKey,
	FLMUINT32 *				pui32Length,
	NICI_OBJECT_HANDLE	masterWrappingKey)
{
	RCODE						rc = FERR_OK;

#ifndef FLM_USE_NICI
	F_UNREFERENCED_PARM( ppucWrappedKey);
	F_UNREFERENCED_PARM( pui32Length);
	F_UNREFERENCED_PARM( masterWrappingKey);
	rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
	goto Exit;
#else
	NICI_CC_HANDLE			context =0;
	NICI_ATTRIBUTE			wKey;
	NICI_ALGORITHM			algorithm;
	NICI_PARAMETER_INFO	parm[1];
	FLMBYTE					oid_aes[] = {IDV_AES128CBC};
	FLMBYTE					oid_3des[] = {IDV_DES_EDE3_CBCPadIV8};
	FLMBYTE					oid_des[] = {IDV_DES_CBCPadIV8};
	NICI_OBJECT_HANDLE	wrappingKeyHandle;

	if( masterWrappingKey)
	{
		wrappingKeyHandle = masterWrappingKey;
	}
	else
	{
		if( RC_BAD( rc = getWrappingKey( &wrappingKeyHandle)))
		{
			goto Exit;
		}
	}

	// Create NICI Context
	
	if( CCS_CreateContext(0, &context) != 0)
	{
		rc = RC_SET( FERR_NICI_CONTEXT);
		goto Exit;
	}
	
	f_memset( &wKey, 0, sizeof( NICI_ATTRIBUTE));

	wKey.type = NICI_A_KEY_TYPE;
	if( CCS_GetAttributeValue( context, wrappingKeyHandle, &wKey, 1) != 0)
	{
		rc = RC_SET( FERR_NICI_ATTRIBUTE_VALUE);
		goto Exit;
	}

	if( !wKey.u.f.hasValue)
	{
		rc = RC_SET( FERR_NICI_BAD_ATTRIBUTE);
		goto ExitCtx;
	}

	switch (wKey.u.f.value)
	{
		case NICI_K_AES:
		{
			algorithm.algorithm = oid_aes;
			algorithm.parameterLen = sizeof(algorithm.parameter->parms[0])+
									sizeof(algorithm.parameter->count);
			algorithm.parameter = parm;
			algorithm.parameter->count = 1;
			algorithm.parameter->parms[0].parmType = NICI_P_IV;
			algorithm.parameter->parms[0].u.b.len = IV_SZ;
			algorithm.parameter->parms[0].u.b.ptr = m_pucIV;
			break;
		}

		case NICI_K_DES3X:
		{
			algorithm.algorithm = oid_3des;
			algorithm.parameterLen = sizeof(algorithm.parameter->parms[0])+
									sizeof(algorithm.parameter->count);
			algorithm.parameter = parm;
			algorithm.parameter->count = 1;
			algorithm.parameter->parms[0].parmType = NICI_P_IV;
			algorithm.parameter->parms[0].u.b.len = IV_SZ8;
			algorithm.parameter->parms[0].u.b.ptr = m_pucIV;
			break;
		}

		case NICI_K_DES:
		{
			// Set up alogrithm now to do DES for encryption
			
			algorithm.algorithm = oid_des;
			algorithm.parameterLen = sizeof(algorithm.parameter->parms[0])+
									sizeof(algorithm.parameter->count);
			algorithm.parameter = parm;
			algorithm.parameter->count = 1;
			algorithm.parameter->parms[0].parmType = NICI_P_IV;
			algorithm.parameter->parms[0].u.b.len = IV_SZ8;
			algorithm.parameter->parms[0].u.b.ptr = m_pucIV;
			break;
		}

		default:
		{
			rc = RC_SET( FERR_NICI_WRAPKEY_FAILED);
			goto ExitCtx;
		}
	}

	// We should be able to call this with NULL for the wrapped 
	// key, to get the length.
	
	if( CCS_WrapKey( context, &algorithm, NICI_KM_UNSPECIFIED, 0,
		wrappingKeyHandle, m_keyHandle, NULL, (NICI_ULONG *)pui32Length) != 0)
	{
		rc = RC_SET( FERR_NICI_WRAPKEY_FAILED);
		goto Exit;
	}

	if( RC_BAD( rc = f_calloc( *pui32Length, ppucWrappedKey)))
	{
		goto ExitCtx;
	}

	if( CCS_WrapKey( context, &algorithm, NICI_KM_UNSPECIFIED, 0,
		wrappingKeyHandle, m_keyHandle, *ppucWrappedKey,
		(NICI_ULONG *)pui32Length) != 0)
	{
		rc = RC_SET( FERR_NICI_WRAPKEY_FAILED);
		goto Exit;
	}

ExitCtx:

	CCS_DestroyContext(context);

#endif

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_CCS::unwrapKey(
	FLMBYTE *				pucWrappedKey,
	FLMUINT32				ui32WrappedKeyLength,
	NICI_OBJECT_HANDLE	masterWrappingKey)
{
	RCODE						rc = FERR_OK;

#ifndef FLM_USE_NICI
	F_UNREFERENCED_PARM( pucWrappedKey);
	F_UNREFERENCED_PARM( ui32WrappedKeyLength);
	F_UNREFERENCED_PARM( masterWrappingKey);
	rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
	goto Exit;
#else
	NICI_CC_HANDLE			context = 0;
	NICI_OBJECT_HANDLE	wrappingKeyHandle;

	if( masterWrappingKey)
	{
		wrappingKeyHandle = masterWrappingKey;
	}
	else
	{
		if( RC_BAD( rc = getWrappingKey( &wrappingKeyHandle)))
		{
			goto Exit;
		}
	}

	// Create NICI Context
	
	if( CCS_CreateContext( 0, &context) != 0)
	{
		rc = RC_SET( FERR_NICI_CONTEXT);
		goto Exit;
	}

	if( CCS_UnwrapKey( context, wrappingKeyHandle, pucWrappedKey,
		ui32WrappedKeyLength, &m_keyHandle) != 0)
	{
		rc = RC_SET_AND_ASSERT( FERR_NICI_UNWRAPKEY_FAILED);
		goto Exit;
	}

	CCS_DestroyContext(context);

#endif

Exit:

	return(rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_CCS::generateEncryptionKey( void)
{
	RCODE			rc = FERR_OK;

#ifndef FLM_USE_NICI
	rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
	goto Exit;
#else

	switch( m_uiAlgType)
	{
		case FLM_NICI_AES:
		{
			rc = generateEncryptionKeyAES();
			break;
		}
		
		case FLM_NICI_DES3:
		{
			rc = generateEncryptionKeyDES3();
			break;
		}
		
		case FLM_NICI_DES:
		{
			rc = generateEncryptionKeyDES();
			break;
		}
		
		default:
		{
			rc = RC_SET( FERR_NICI_INVALID_ALGORITHM);
			goto Exit;
		}
	}

#endif

Exit:

	return( rc);

}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_CCS::generateEncryptionKeyAES( void)
{
	RCODE						rc = FERR_OK;

#ifndef FLM_USE_NICI
	rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
	goto Exit;
#else
	NICI_CC_HANDLE			context = 0;
	NICI_ALGORITHM			algorithm;
	NICI_ATTRIBUTE			keyAttr[3];
	FLMUINT8					keySizeChanged;
	FLMBYTE					oid_aes[] = {IDV_AES128CBC};

	// Create NICI Context
	
	if( CCS_CreateContext(0, &context) != 0)
	{
		rc = RC_SET( FERR_NICI_CONTEXT);
		goto Exit;
	}

	// Set up algorithm
	
	algorithm.algorithm = oid_aes;
	algorithm.parameterLen = 0;

	// Set up key attributes
	
	keyAttr[0].type = NICI_A_KEY_USAGE;
	keyAttr[0].u.f.hasValue = 1;
	keyAttr[0].u.f.value = NICI_F_DATA_ENCRYPT | NICI_F_DATA_DECRYPT | NICI_F_EXTRACT;
	keyAttr[0].u.f.valueInfo = 0;

	keyAttr[1].type = NICI_A_KEY_SIZE;
	keyAttr[1].u.f.hasValue = 1;
	keyAttr[1].u.f.value = 128;
	keyAttr[1].u.f.valueInfo = 0;

	keyAttr[2].type = NICI_A_GLOBAL;
	keyAttr[2].u.f.hasValue = 1;
	keyAttr[2].u.f.value = N_TRUE;
	keyAttr[2].u.f.valueInfo = 0;

	// Generate a key
	
	if( CCS_GenerateKey( context, &algorithm, keyAttr, 3,
		(NICI_BBOOL *)&keySizeChanged, &m_keyHandle, NICI_H_INVALID) != 0)
	{
		rc = RC_SET( FERR_NICI_GENKEY_FAILED);
		goto Exit;
	}

	// Generate some IV to use with this key.
	
	if( CCS_GetRandom( context, m_pucIV, IV_SZ) != 0)
	{
		rc = RC_SET( FERR_NICI_BAD_RANDOM);
		goto Exit;
	}

	CCS_DestroyContext(context);

#endif

Exit:

	return(rc);

}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_CCS::generateEncryptionKeyDES3( void)
{
	RCODE						rc = FERR_OK;

#ifndef FLM_USE_NICI
	rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
	goto Exit;
#else
	NICI_CC_HANDLE			context = 0;
	NICI_ALGORITHM			algorithm;
	NICI_ATTRIBUTE			keyAttr[3];
	FLMUINT8					keySizeChanged;
	FLMBYTE					oid_des3[] = {IDV_DES_EDE3_CBC_IV8};

	// Create NICI Context
	
	if( CCS_CreateContext(0, &context) != 0)
	{
		rc = RC_SET( FERR_NICI_CONTEXT);
		goto Exit;
	}

	// Set up algorithm
	
	algorithm.algorithm = oid_des3;
	algorithm.parameterLen = 0;

	// Set up key attributes
	
	keyAttr[0].type = NICI_A_KEY_USAGE;
	keyAttr[0].u.f.hasValue = 1;
	keyAttr[0].u.f.value = NICI_F_DATA_ENCRYPT | NICI_F_DATA_DECRYPT | NICI_F_EXTRACT;
	keyAttr[0].u.f.valueInfo = 0;

	keyAttr[1].type = NICI_A_KEY_SIZE;
	keyAttr[1].u.f.hasValue = 1;
	keyAttr[1].u.f.value = 168;
	keyAttr[1].u.f.valueInfo = 0;

	keyAttr[2].type = NICI_A_GLOBAL;
	keyAttr[2].u.f.hasValue = 1;
	keyAttr[2].u.f.value = N_TRUE;
	keyAttr[2].u.f.valueInfo = 0;

	// Generate a DES3 key
	
	if( CCS_GenerateKey( context, &algorithm, keyAttr, 3,
		(NICI_BBOOL *)&keySizeChanged, &m_keyHandle, NICI_H_INVALID) != 0)
	{
		rc = RC_SET( FERR_NICI_GENKEY_FAILED);
		goto Exit;
	}

	// Generate some IV to use with this key
	
	if( CCS_GetRandom( context, m_pucIV, IV_SZ) != 0)
	{
		rc = RC_SET( FERR_NICI_BAD_RANDOM);
		goto Exit;
	}

	CCS_DestroyContext(context);

#endif

Exit:

	return(rc);

}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_CCS::generateEncryptionKeyDES( void)
{
	RCODE						rc = FERR_OK;

#ifndef FLM_USE_NICI
	rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
	goto Exit;
#else
	NICI_CC_HANDLE			context = 0;
	NICI_ALGORITHM			algorithm;
	NICI_ATTRIBUTE			keyAttr[3];
	FLMUINT8					keySizeChanged;
	FLMBYTE					oid_des[] = {IDV_DES_CBC_IV8};

	// Create NICI Context
	
	if( CCS_CreateContext(0, &context) != 0)
	{
		rc = RC_SET( FERR_NICI_CONTEXT);
		goto Exit;
	}

	// Set up AES Algorithm
	
	algorithm.algorithm = oid_des;
	algorithm.parameterLen = 0;

	// Set up key attributes
	keyAttr[0].type = NICI_A_KEY_USAGE;
	keyAttr[0].u.f.hasValue = 1;
	keyAttr[0].u.f.value = NICI_F_DATA_ENCRYPT | NICI_F_DATA_DECRYPT | NICI_F_EXTRACT;
	keyAttr[0].u.f.valueInfo = 0;

	keyAttr[1].type = NICI_A_KEY_SIZE;
	keyAttr[1].u.f.hasValue = 1;
	keyAttr[1].u.f.value = 56;
	keyAttr[1].u.f.valueInfo = 0;

	keyAttr[2].type = NICI_A_GLOBAL;
	keyAttr[2].u.f.hasValue = 1;
	keyAttr[2].u.f.value = N_TRUE;
	keyAttr[2].u.f.valueInfo = 0;

	// Generate a AES key
	
	if( CCS_GenerateKey( context, &algorithm, keyAttr, 3,
		(NICI_BBOOL *)&keySizeChanged, &m_keyHandle, NICI_H_INVALID) != 0)
	{
		rc = RC_SET( FERR_NICI_GENKEY_FAILED);
		goto Exit;
	}

	// Generate some IV to use with this key.
	
	if( CCS_GetRandom( context, m_pucIV, IV_SZ) != 0)
	{
		rc = RC_SET( FERR_NICI_BAD_RANDOM);
		goto Exit;
	}

	CCS_DestroyContext(context);

#endif

Exit:

	return(rc);

}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_CCS::generateWrappingKey( void)
{
	RCODE						rc = FERR_OK;

#ifndef FLM_USE_NICI
	rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
	goto Exit;
#else
	NICI_CC_HANDLE			context = 0;
	NICI_ALGORITHM			algorithm;
	NICI_ATTRIBUTE			keyAttr[6];
	FLMUINT8					keySizeChanged;
	FLMBYTE					oid_des3[] = {IDV_DES_EDE3_CBC_IV8};
	FLMUINT					uiIndx;

	// Create NICI Context
	
	if( CCS_CreateContext(0, &context) != 0)
	{
		rc = RC_SET( FERR_NICI_CONTEXT);
		goto Exit;
	}

	// Set up AES Algorithm
	
	algorithm.algorithm = oid_des3;
	algorithm.parameterLen = 0;

	// Set up key attributes
	
	uiIndx = 0;
	keyAttr[uiIndx].type = NICI_A_KEY_TYPE;
	keyAttr[uiIndx].u.f.hasValue = 1;
	keyAttr[uiIndx].u.f.value = NICI_K_DES3X;
	keyAttr[uiIndx].u.f.valueInfo = 0;

	uiIndx++;
	keyAttr[uiIndx].type = NICI_A_KEY_FORMAT;
	keyAttr[uiIndx].u.v.valuePtr = oid_des3;
	keyAttr[uiIndx].u.v.valueLen = sizeof( oid_des3);
	keyAttr[uiIndx].u.v.valueInfo = 0;

	uiIndx++;
	keyAttr[uiIndx].type = NICI_A_KEY_USAGE;
	keyAttr[uiIndx].u.f.hasValue = 1;
	keyAttr[uiIndx].u.f.value = NICI_F_WRAP | NICI_F_UNWRAP | NICI_F_KM_ENCRYPT | NICI_F_KM_DECRYPT | NICI_F_EXTRACT;
	keyAttr[uiIndx].u.f.valueInfo = 0;

	uiIndx++;
	keyAttr[uiIndx].type = NICI_A_KEY_SIZE;
	keyAttr[uiIndx].u.f.hasValue = 1;
	keyAttr[uiIndx].u.f.value = 168;
	keyAttr[uiIndx].u.f.valueInfo = 0;

	uiIndx++;
	keyAttr[uiIndx].type = NICI_A_GLOBAL;
	keyAttr[uiIndx].u.f.hasValue = 1;
	keyAttr[uiIndx].u.f.value = N_TRUE;
	keyAttr[uiIndx].u.f.valueInfo = 0;

	uiIndx++;
	keyAttr[uiIndx].type = NICI_A_CLASS;
	keyAttr[uiIndx].u.f.hasValue = 1;
	keyAttr[uiIndx].u.f.value = NICI_O_SECRET_KEY;
	keyAttr[uiIndx].u.f.valueInfo = 0;

	// Generate an AES wrapping key
	
	if( CCS_GenerateKey( context, &algorithm, keyAttr, 6,
		(NICI_BBOOL *)&keySizeChanged, &m_keyHandle, NICI_H_INVALID) != 0)
	{
		rc = RC_SET( FERR_NICI_GENKEY_FAILED);
		goto Exit;
	}

	// Generate some IV to use with this key
	
	if (CCS_GetRandom( context, m_pucIV, IV_SZ) != 0)
	{
		rc = RC_SET( FERR_NICI_BAD_RANDOM);
		goto Exit;
	}

	// If we generated a wrap;ping key, then this object's key handle is
	// actually a wrapping key.  This means that we will us it to wrap
	// the other keys in the system.
	
	m_bKeyIsWrappingKey = TRUE;
	CCS_DestroyContext(context);

#endif

Exit:

	return(rc);

}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_CCS::encryptToStore(
	FLMBYTE *			pucIn,
	FLMUINT				uiInLen,
	FLMBYTE *			pucOut,
	FLMUINT *			puiOutLen)
{
	RCODE			rc = FERR_OK;

#ifndef FLM_USE_NICI
	F_UNREFERENCED_PARM( pucIn);
	F_UNREFERENCED_PARM( uiInLen);
	F_UNREFERENCED_PARM( pucOut);
	F_UNREFERENCED_PARM( puiOutLen);
	rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
	goto Exit;
#else

	switch (m_uiAlgType)
	{
		case FLM_NICI_AES:
		{
			rc = encryptToStoreAES( pucIn, uiInLen, pucOut, puiOutLen);
			break;
		}
		case FLM_NICI_DES3:
		{
			rc = encryptToStoreDES3( pucIn, uiInLen, pucOut, puiOutLen);
			break;
		}
		
		case FLM_NICI_DES:
		{
			rc = encryptToStoreDES( pucIn, uiInLen, pucOut, puiOutLen);
			break;
		}
		
		default:
		{
			rc = RC_SET( FERR_NICI_INVALID_ALGORITHM);
			goto Exit;
		}
	}

#endif

Exit:

	return( rc);

}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_CCS::decryptFromStore(
	FLMBYTE *				pucIn,
	FLMUINT					uiInLen,
	FLMBYTE *				pucOut,
	FLMUINT *				puiOutLen)
{
	RCODE				rc = FERR_OK;

#ifndef FLM_USE_NICI
	F_UNREFERENCED_PARM( pucIn);
	F_UNREFERENCED_PARM( uiInLen);
	F_UNREFERENCED_PARM( pucOut);
	F_UNREFERENCED_PARM( puiOutLen);
	rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
	goto Exit;
#else

	switch( m_uiAlgType)
	{
		case FLM_NICI_AES:
		{
			rc = decryptFromStoreAES( pucIn, uiInLen, pucOut, puiOutLen);
			break;
		}
		
		case FLM_NICI_DES3:
		{
			rc = decryptFromStoreDES3( pucIn, uiInLen, pucOut, puiOutLen);
			break;
		}
		
		case FLM_NICI_DES:
		{
			rc = decryptFromStoreDES( pucIn, uiInLen, pucOut, puiOutLen);
			break;
		}
		
		default:
		{
			rc = RC_SET( FERR_NICI_INVALID_ALGORITHM);
			goto Exit;
		}
	}

#endif

Exit:

	return( rc);

}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_CCS::encryptToStoreAES(
	FLMBYTE *			pucIn,
	FLMUINT				uiInLen,
	FLMBYTE *			pucOut,
	FLMUINT *			puiOutLen)
{
	RCODE					rc = FERR_OK;

#ifndef FLM_USE_NICI
	F_UNREFERENCED_PARM( pucIn);
	F_UNREFERENCED_PARM( uiInLen);
	F_UNREFERENCED_PARM( pucOut);
	F_UNREFERENCED_PARM( puiOutLen);
	rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
	goto Exit;
#else
	NICI_CC_HANDLE				context = 0;
	NICI_ALGORITHM				algorithm;
	NICI_PARAMETER_INFO		parm[1];
	FLMBYTE						oid_aes[] = {IDV_AES128CBC};

	// Create NICI Context
	
	if( CCS_CreateContext(0, &context) != 0)
	{
		rc = RC_SET( FERR_NICI_CONTEXT);
		goto Exit;
	}

	algorithm.algorithm = oid_aes;
	algorithm.parameterLen = sizeof(algorithm.parameter->parms[0])+
								sizeof(algorithm.parameter->count);
	algorithm.parameter = parm;
	algorithm.parameter->count = 1;
	algorithm.parameter->parms[0].parmType = NICI_P_IV;
	algorithm.parameter->parms[0].u.b.len = IV_SZ;
	algorithm.parameter->parms[0].u.b.ptr = m_pucIV;

	if( CCS_DataEncryptInit(context, &algorithm, m_keyHandle) != 0)
	{
		rc = RC_SET( FERR_NICI_ENC_INIT_FAILED);
	 	goto Exit;
	}

	if( CCS_Encrypt( context, pucIn, uiInLen, pucOut, puiOutLen) != 0)
	{
		rc = RC_SET( FERR_NICI_ENCRYPT_FAILED);
		goto Exit;
	}

	CCS_DestroyContext(context);

#endif

Exit:

	return(rc);

}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_CCS::decryptFromStoreAES(
	FLMBYTE *				pucIn,
	FLMUINT					uiInLen,
	FLMBYTE *				pucOut,
	FLMUINT *				puiOutLen)
{
	RCODE						rc = FERR_OK;

#ifndef FLM_USE_NICI
	F_UNREFERENCED_PARM( pucIn);
	F_UNREFERENCED_PARM( uiInLen);
	F_UNREFERENCED_PARM( pucOut);
	F_UNREFERENCED_PARM( puiOutLen);
	rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
	goto Exit;
#else
	NICI_CC_HANDLE				context = 0;
	NICI_ALGORITHM				algorithm;
	NICI_PARAMETER_INFO		parm[1];
	FLMBYTE						oid_aes[] = {IDV_AES128CBC};

	// Create NICI Context
	
	if (CCS_CreateContext(0, &context) != 0)
	{
		rc = RC_SET( FERR_NICI_CONTEXT);
		goto Exit;
	}

	algorithm.algorithm = oid_aes;
	algorithm.parameterLen = sizeof(algorithm.parameter->parms[0])+
								sizeof(algorithm.parameter->count);
	algorithm.parameter = parm;
	algorithm.parameter->count = 1;
	algorithm.parameter->parms[0].parmType = NICI_P_IV;
	algorithm.parameter->parms[0].u.b.len = IV_SZ;
	algorithm.parameter->parms[0].u.b.ptr = m_pucIV;

	// Init encryption

	if (CCS_DataDecryptInit(context, &algorithm, m_keyHandle) != 0)
	{
		rc = RC_SET( FERR_NICI_DECRYPT_INIT_FAILED);
	 	goto Exit;
	}

	if( CCS_Decrypt( context, pucIn, uiInLen, pucOut, puiOutLen) != 0)
	{
		rc = RC_SET( FERR_NICI_DECRYPT_FAILED);
		goto Exit;
	}

	CCS_DestroyContext(context);

#endif

Exit:

	return(rc);

}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_CCS::encryptToStoreDES3(
	FLMBYTE *			pucIn,
	FLMUINT				uiInLen,
	FLMBYTE *			pucOut,
	FLMUINT *			puiOutLen)
{
	RCODE					rc = FERR_OK;

#ifndef FLM_USE_NICI
	F_UNREFERENCED_PARM( pucIn);
	F_UNREFERENCED_PARM( uiInLen);
	F_UNREFERENCED_PARM( pucOut);
	F_UNREFERENCED_PARM( puiOutLen);
	rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
	goto Exit;
#else
	NICI_CC_HANDLE				context = 0;
	NICI_ALGORITHM				algorithm;
	NICI_PARAMETER_INFO		parm[1];
	FLMBYTE						oid_des3[] = {IDV_DES_EDE3_CBC_IV8};

	// Create NICI Context
	
	if (CCS_CreateContext(0, &context) != 0)
	{
		rc = RC_SET( FERR_NICI_CONTEXT);
		goto Exit;
	}

	algorithm.algorithm = oid_des3;
	algorithm.parameterLen = sizeof(algorithm.parameter->parms[0])+
								sizeof(algorithm.parameter->count);
	algorithm.parameter = parm;
	algorithm.parameter->count = 1;
	algorithm.parameter->parms[0].parmType = NICI_P_IV;
	algorithm.parameter->parms[0].u.b.len = IV_SZ8;
	algorithm.parameter->parms[0].u.b.ptr = m_pucIV;

	// Init encryption
	
	if (CCS_DataEncryptInit(context, &algorithm, m_keyHandle) != 0)
	{
		rc = RC_SET( FERR_NICI_ENC_INIT_FAILED);
	 	goto Exit;
	}

	if( CCS_Encrypt( context, pucIn, uiInLen, pucOut, puiOutLen) != 0)
	{
		rc = RC_SET( FERR_NICI_ENCRYPT_FAILED);
		goto Exit;
	}

	CCS_DestroyContext(context);

#endif

Exit:

	return(rc);

}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_CCS::decryptFromStoreDES3(
	FLMBYTE *				pucIn,
	FLMUINT					uiInLen,
	FLMBYTE *				pucOut,
	FLMUINT *				puiOutLen)
{
	RCODE						rc = FERR_OK;

#ifndef FLM_USE_NICI
	F_UNREFERENCED_PARM( pucIn);
	F_UNREFERENCED_PARM( uiInLen);
	F_UNREFERENCED_PARM( pucOut);
	F_UNREFERENCED_PARM( puiOutLen);
	rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
	goto Exit;
#else
	NICI_CC_HANDLE				context = 0;
	NICI_ALGORITHM				algorithm;
	NICI_PARAMETER_INFO		parm[1];
	FLMBYTE						oid_des3[] = {IDV_DES_EDE3_CBC_IV8};

	// Create NICI Context
	
	if( CCS_CreateContext(0, &context) != 0)
	{
		rc = RC_SET( FERR_NICI_CONTEXT);
		goto Exit;
	}

	// Set up alogrithm now to do triple des decryption
	
	algorithm.algorithm = oid_des3;
	algorithm.parameterLen = sizeof(algorithm.parameter->parms[0])+
								sizeof(algorithm.parameter->count);
	algorithm.parameter = parm;
	algorithm.parameter->count = 1;
	algorithm.parameter->parms[0].parmType = NICI_P_IV;
	algorithm.parameter->parms[0].u.b.len = IV_SZ8;
	algorithm.parameter->parms[0].u.b.ptr = m_pucIV;

	// Init encryption
	
	if( CCS_DataDecryptInit(context, &algorithm, m_keyHandle) != 0)
	{
		rc = RC_SET( FERR_NICI_DECRYPT_INIT_FAILED);
	 	goto Exit;
	}

	if( CCS_Decrypt( context, pucIn, uiInLen, pucOut, puiOutLen) != 0)
	{
		rc = RC_SET( FERR_NICI_DECRYPT_FAILED);
		goto Exit;
	}

	CCS_DestroyContext(context);

#endif

Exit:

	return(rc);

}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_CCS::encryptToStoreDES(
	FLMBYTE *			pucIn,
	FLMUINT				uiInLen,
	FLMBYTE *			pucOut,
	FLMUINT *			puiOutLen)
{
	RCODE					rc = FERR_OK;

#ifndef FLM_USE_NICI
	F_UNREFERENCED_PARM( pucIn);
	F_UNREFERENCED_PARM( uiInLen);
	F_UNREFERENCED_PARM( pucOut);
	F_UNREFERENCED_PARM( puiOutLen);
	rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
	goto Exit;
#else
	NICI_CC_HANDLE				context = 0;
	NICI_ALGORITHM				algorithm;
	NICI_PARAMETER_INFO		parm[1];
	FLMBYTE						oid_des[] = {IDV_DES_CBC_IV8};

	// Create NICI Context
	
	if (CCS_CreateContext(0, &context) != 0)
	{
		rc = RC_SET( FERR_NICI_CONTEXT);
		goto Exit;
	}

	algorithm.algorithm = oid_des;
	algorithm.parameterLen = sizeof(algorithm.parameter->parms[0])+
								sizeof(algorithm.parameter->count);
	algorithm.parameter = parm;
	algorithm.parameter->count = 1;
	algorithm.parameter->parms[0].parmType = NICI_P_IV;
	algorithm.parameter->parms[0].u.b.len = IV_SZ8;
	algorithm.parameter->parms[0].u.b.ptr = m_pucIV;

	// Init encryption
	
	if( CCS_DataEncryptInit(context, &algorithm, m_keyHandle) != 0)
	{
		rc = RC_SET( FERR_NICI_ENC_INIT_FAILED);
	 	goto Exit;
	}

	if( CCS_Encrypt(context, pucIn, uiInLen, pucOut, puiOutLen) != 0)
	{
		rc = RC_SET( FERR_NICI_ENCRYPT_FAILED);
		goto Exit;
	}

	CCS_DestroyContext(context);

#endif

Exit:

	return(rc);

}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_CCS::decryptFromStoreDES(
	FLMBYTE *				pucIn,
	FLMUINT					uiInLen,
	FLMBYTE *				pucOut,
	FLMUINT *				puiOutLen)
{
	RCODE						rc = FERR_OK;

#ifndef FLM_USE_NICI
	F_UNREFERENCED_PARM( pucIn);
	F_UNREFERENCED_PARM( uiInLen);
	F_UNREFERENCED_PARM( pucOut);
	F_UNREFERENCED_PARM( puiOutLen);
	rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
	goto Exit;
#else
	NICI_CC_HANDLE				context = 0;
	NICI_ALGORITHM				algorithm;
	NICI_PARAMETER_INFO		parm[1];
	FLMBYTE						oid_des[] = {IDV_DES_CBC_IV8};

	// Create NICI Context
	
	if( CCS_CreateContext( 0, &context) != 0)
	{
		rc = RC_SET( FERR_NICI_CONTEXT);
		goto Exit;
	}

	// Set up alogrithm now to do triple des decryption
	
	algorithm.algorithm = oid_des;
	algorithm.parameterLen = sizeof(algorithm.parameter->parms[0])+
								sizeof(algorithm.parameter->count);
	algorithm.parameter = parm;
	algorithm.parameter->count = 1;
	algorithm.parameter->parms[0].parmType = NICI_P_IV;
	algorithm.parameter->parms[0].u.b.len = IV_SZ8;
	algorithm.parameter->parms[0].u.b.ptr = m_pucIV;

	// Init encryption
	
	if( CCS_DataDecryptInit(context, &algorithm, m_keyHandle) != 0)
	{
		rc = RC_SET( FERR_NICI_DECRYPT_INIT_FAILED);
	 	goto Exit;
	}

	if( CCS_Decrypt( context, pucIn, uiInLen, pucOut, puiOutLen) != 0)
	{
		rc = RC_SET( FERR_NICI_DECRYPT_FAILED);
		goto Exit;
	}

	CCS_DestroyContext(context);

#endif

Exit:

	return(rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_CCS::init(
	FLMBOOL		bKeyIsWrappingKey,
	FLMUINT		uiAlgType)
{
	RCODE			rc = FERR_OK;

	if (m_bInitCalled)
	{
		flmAssert(0);
		goto Exit;
	}

	m_bKeyIsWrappingKey = bKeyIsWrappingKey;

	if (uiAlgType != FLM_NICI_AES &&
		 uiAlgType != FLM_NICI_DES3 &&
		 uiAlgType != FLM_NICI_DES)
	{
		rc = RC_SET( FERR_NICI_INVALID_ALGORITHM);
		goto Exit;
	}

	if (RC_BAD( rc = f_mutexCreate( &m_hMutex)))
	{
		goto Exit;
	}

	m_uiAlgType = uiAlgType;

	m_bInitCalled = TRUE;

Exit:

	return( rc);

}

/****************************************************************************
Desc:	Pick a wrapping key that we can use to wrap and
		unwrap the encryption key with.
****************************************************************************/
RCODE F_CCS::getWrappingKey(
	NICI_OBJECT_HANDLE *		pWrappingKeyHandle)
{
	RCODE						rc = FERR_OK;

#ifndef FLM_USE_NICI
	F_UNREFERENCED_PARM( pWrappingKeyHandle);
	rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
	goto Exit;
#else
	NICI_ATTRIBUTE			find[2];
	NICI_CC_HANDLE			context =0;
	FLMUINT					uiCount;

	// Create NICI Context
	
	if( CCS_CreateContext(0, &context) != 0)
	{
		rc = RC_SET( FERR_NICI_CONTEXT);
		goto Exit;
	}

	find[0].type = NICI_A_GLOBAL;
	find[0].u.f.hasValue = 1;
	find[0].u.f.value = 1;
	find[0].u.f.valueInfo = 0;

	find[1].type = NICI_A_FEATURE;
	find[1].u.f.hasValue = 1;
	find[1].u.f.value = NICI_AV_STORAGE;
	find[1].u.f.valueInfo = 0;

	if( CCS_FindObjectsInit(context, find, 2) != 0)
	{
		rc = RC_SET( FERR_NICI_FIND_INIT);
		goto Exit;
	}

	uiCount = 1;
	if (CCS_FindObjects(context, pWrappingKeyHandle, &uiCount) != 0)
	{
		rc = RC_SET( FERR_NICI_FIND_OBJECT);
		goto Exit;
	}

	if (uiCount < 1)
	{
		rc = RC_SET( FERR_NICI_WRAPKEY_NOT_FOUND);
		goto ExitCtx;
	}

ExitCtx:

	CCS_DestroyContext(context);

#endif

Exit:

	return(rc);
}

/****************************************************************************
Desc:	Function used to obtain the key information in the
		format that will be stored on disk.
****************************************************************************/
RCODE F_CCS::getKeyToStore(
	FLMBYTE **				ppucKeyInfo,
	FLMUINT32 *				pui32BufLen,
	const char *			pszEncKeyPasswd,
	F_CCS *					pWrappingCcs,
	FLMBOOL					bBase64Encode)
{
	RCODE						rc = FERR_OK;

#ifndef FLM_USE_NICI
	F_UNREFERENCED_PARM( ppucKeyInfo);
	F_UNREFERENCED_PARM( pui32BufLen);
	F_UNREFERENCED_PARM( pszEncKeyPasswd);
	F_UNREFERENCED_PARM( pWrappingCcs);
	F_UNREFERENCED_PARM( bBase64Encode);
	rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
	goto Exit;
#else
	FLMBYTE *				pucTmp = NULL;
	FLMBYTE *				pucPtr = NULL;
	void *					pvB64Buffer = NULL;
	FLMUINT32				ui32PaddedLength;
	NICI_CC_HANDLE			context = 0;
	IF_PosIStream *		pBufferIStream = NULL;
	IF_IStream *			pB64Encoder = NULL;
	FLMBYTE *				pucWrappedKey = NULL;
	FLMUINT32				ui32WrappedKeyLen = 0;
	char *					pszFormattedEncKeyPasswd = NULL;
	NICI_OBJECT_HANDLE	wrappingKeyHandle = 0;
	FLMUINT					uiB64Length;

	*ppucKeyInfo = NULL;
	*pui32BufLen = 0;

	if (pWrappingCcs)
	{
		flmAssert(m_bKeyIsWrappingKey == FALSE);
		wrappingKeyHandle = pWrappingCcs->m_keyHandle;
	}
	else if (!pszEncKeyPasswd)
	{
		flmAssert( m_bKeyIsWrappingKey);
	}

	// Either extract the key or wrap the key.
	
	if( pszEncKeyPasswd && pszEncKeyPasswd[0])
	{
		// The password that is passed in to CCS_pbeEncrypt is NOT actually
		// unicode.  It must be treated as a sequence of bytes that that is
		// terminated with 2 nulls and has an even length.  If we treat it
		// as unicode, then we'll have endian issues if we move the database
		// to machines with different byte ordering.
		
		if (RC_BAD( rc = f_calloc( f_strlen(pszEncKeyPasswd) +
											(f_strlen(pszEncKeyPasswd) % 2) + 2,
											&pszFormattedEncKeyPasswd)))
		{
			goto Exit;
		}
		
		f_strcpy( pszFormattedEncKeyPasswd, pszEncKeyPasswd);

		if( RC_BAD( rc = extractKey( &pucWrappedKey, &ui32WrappedKeyLen,
											  (FLMUNICODE *)pszFormattedEncKeyPasswd)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = wrapKey( &pucWrappedKey, &ui32WrappedKeyLen,
			wrappingKeyHandle)))
		{
			goto Exit;
		}
	}

	// The shrouded or wrapped key will be stored in m_pKey.
	
	ui32PaddedLength = (ui32WrappedKeyLen + sizeof( FLMUINT32) +
								sizeof( FLMUINT32) + IV_SZ );

	// Make sure our buffer size is padded to a 16 byte boundary.
	
	if ((ui32PaddedLength % 16) != 0)
	{
		ui32PaddedLength += (16 - (ui32PaddedLength % 16));
	}

	// Add one extra byte for a NULL terminator
	
	if (RC_BAD(rc = f_alloc( ui32PaddedLength + 1, &pucTmp)))
	{
		goto Exit;
	}

	if (CCS_CreateContext( 0, &context))
	{
		rc = RC_SET( FERR_NICI_CONTEXT);
		goto Exit;
	}

	pucPtr = pucTmp;

	// Save a flag indicating whether the key is wrapped or encoded in
	// a password.
	
	UD2FBA( (pszEncKeyPasswd && pszEncKeyPasswd[0]) ? 1 : 0, pucPtr); 
	pucPtr += sizeof( FLMUINT32);

	// Copy the key length.
	
	UD2FBA(ui32WrappedKeyLen, pucPtr);
	pucPtr += sizeof( FLMUINT32);

	// Copy the IV too.
	
	f_memcpy( pucPtr, m_pucIV, IV_SZ);
	pucPtr += IV_SZ;

	// Copy the wrapped key value
	
	f_memcpy( pucPtr, pucWrappedKey, ui32WrappedKeyLen);
	pucPtr += ui32WrappedKeyLen;

	// Fill the remainder of the buffer with random data.
	
	if( CCS_GetRandom( context, pucPtr,
							((FLMUINT)pucTmp + ui32PaddedLength) - (FLMUINT)pucPtr))
	{
		rc = RC_SET( FERR_NICI_BAD_RANDOM);
		goto Exit;
	}

	if( bBase64Encode)
	{
		// The resulting length will not be more than doubled.
		
		uiB64Length = (FLMUINT)(ui32PaddedLength * 2);
		if( RC_BAD( rc = f_calloc( uiB64Length, &pvB64Buffer)))
		{
			goto ExitCtx;
		}
		
		if( RC_BAD( rc = FlmOpenBufferIStream( (const char *)pucTmp, 
			ui32PaddedLength, &pBufferIStream)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = FlmOpenBase64EncoderIStream( pBufferIStream, 
			FALSE, &pB64Encoder)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = pB64Encoder->read( pvB64Buffer, 
			0xFFFFFFFF, &uiB64Length)))
		{
			if( rc != NE_FLM_EOF_HIT)
			{
				goto ExitCtx;
			}
			
			rc = NE_FLM_OK;
		}

		flmAssert( uiB64Length < (FLMUINT)(ui32PaddedLength * 2));

		((FLMBYTE *)pvB64Buffer)[ uiB64Length] = '\0';
		*ppucKeyInfo = (FLMBYTE *)pvB64Buffer;
		pvB64Buffer = NULL;
		*pui32BufLen = (FLMUINT32)uiB64Length;
	}
	else
	{
		pucTmp[ ui32PaddedLength] = '\0';
		*ppucKeyInfo = pucTmp;
		*pui32BufLen = ui32PaddedLength;
		pucTmp = NULL;
	}

ExitCtx:

	CCS_DestroyContext( context);

#endif

Exit:

#ifdef FLM_USE_NICI
	if (pucTmp)
	{
		f_free(&pucTmp);
	}

	if (pvB64Buffer)
	{
		f_free(&pvB64Buffer);
	}

	if (pB64Encoder)
	{
		pB64Encoder->Release();
	}
	
	if (pBufferIStream)
	{
		pBufferIStream->Release();
	}

	if (pucWrappedKey)
	{
		f_free( &pucWrappedKey);
	}

	if (pszFormattedEncKeyPasswd)
	{
		f_free( &pszFormattedEncKeyPasswd);
	}
#endif

	return( rc);
}

/****************************************************************************
Desc: Function used to set the key info using the binary key stored
		on the disk.
****************************************************************************/
RCODE F_CCS::setKeyFromStore(
	FLMBYTE *			pucKeyInfo,
	FLMUINT32			ui32BufLen,
	const char *		pszEncKeyPasswd,
	F_CCS *				pWrappingCcs,
	FLMBOOL				bBase64Encoded)
{
	RCODE					rc = FERR_OK;

#ifndef FLM_USE_NICI
	F_UNREFERENCED_PARM( pucKeyInfo);
	F_UNREFERENCED_PARM( ui32BufLen);
	F_UNREFERENCED_PARM( pszEncKeyPasswd);
	F_UNREFERENCED_PARM( pWrappingCcs);
	F_UNREFERENCED_PARM( bBase64Encoded);
	rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
	goto Exit;
#else
	FLMBYTE *				pTmpKey = pucKeyInfo;
	FLMBYTE *				pucTmp;
	FLMBYTE *				pucBuffer = NULL;
	FLMBOOL					bShrouded = FALSE;
	FLMUINT					uiLength;
	FLMBYTE *				pucKeyBuf = NULL;
	char *					pszFormattedEncKeyPasswd = NULL;
	NICI_OBJECT_HANDLE	wrappingKeyHandle = 0;

	if (pWrappingCcs)
	{
		flmAssert(m_bKeyIsWrappingKey == FALSE);
		wrappingKeyHandle = pWrappingCcs->m_keyHandle;
	}


	if (bBase64Encoded)
	{
		F_BufferIStream			bufferStream;
		{
			F_Base64DecoderIStream	B64Decoder;
	
			// Need a temporary buffer to translate the Base64 encoded buffer into
	
			if (RC_BAD( rc = f_alloc( ui32BufLen, &pucKeyBuf)))
			{
				goto Exit;
			}
	
			if (RC_BAD( rc = bufferStream.openStream( (const char *)pTmpKey, (FLMUINT)ui32BufLen)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = B64Decoder.openStream( &bufferStream)))
			{
				goto Exit;
			}
	
			// Buffer is Base64 encoded.  We must first decode it.
			
			// Decode the buffer
			
			if( RC_BAD( rc = B64Decoder.read(
				(void *)pucKeyBuf, ui32BufLen, &uiLength)))
			{
				goto Exit;
			}
			pucTmp = pucKeyBuf;
		}
	}
	else
	{
		// Buffer is not base 64 encoded
		
		pucTmp = pTmpKey;
	}

	// Extract the fields from the buffer
	
	bShrouded = FB2UD( pucTmp);
	pucTmp += sizeof( FLMUINT32);

	// Actual length - note that the passed buffer is padded to 16 byte boundary.
	
	uiLength = FB2UD( pucTmp);
	pucTmp += sizeof( FLMUINT32);

	// Get the IV
	
	f_memcpy( m_pucIV, pucTmp, IV_SZ);
	pucTmp += IV_SZ;

	// Need another temporary buffer to hold the encrypted / shrouded key.
	
	if (RC_BAD( rc = f_alloc( uiLength, &pucBuffer)))
	{
		goto Exit;
	}

	f_memcpy( pucBuffer, pucTmp, uiLength);

	if (bShrouded)
	{
		if (pszEncKeyPasswd == NULL)
		{
			rc = RC_SET( FERR_REQUIRE_PASSWD);
			goto Exit;
		}
		
		// The password that is passed in to CCS_pbeDecrypt is NOT actually
		// unicode.  It must be treated as a sequence of bytes that that is
		// terminated with 2 nulls and has an even length.  If we treat it
		// as unicode, then we'll have endian issues if we move the database
		// to machines with different byte ordering.
		
		if( RC_BAD( rc = f_calloc( f_strlen(pszEncKeyPasswd) +
			(f_strlen( pszEncKeyPasswd) % 2) + 2, &pszFormattedEncKeyPasswd)))
		{
			goto Exit;
		}
		
		f_strcpy( pszFormattedEncKeyPasswd, pszEncKeyPasswd);

		// Unshroud the key using the password.
		// Key handle is always kept in m_keyHandle.
		
		if( RC_BAD( rc = injectKey( pucBuffer, (FLMUINT32)uiLength,
			(FLMUNICODE *)pszFormattedEncKeyPasswd)))
		{
			goto Exit;
		}
	}
	else
	{
		if (pszEncKeyPasswd)
		{
			flmAssert( pszEncKeyPasswd[0] == '\0');
		}

		// Unwrap the key.  The Key handle is always store in m_keyHandle.
		
		if (RC_BAD( rc = unwrapKey( pucBuffer, (FLMUINT32)uiLength, wrappingKeyHandle)))
		{
			goto Exit;
		}
	}

	m_bKeyVerified = TRUE;

#endif

Exit:

#ifdef FLM_USE_NICI

	if (pucBuffer)
	{
		f_free( &pucBuffer);
	}

	if (pucKeyBuf)
	{
		f_free( &pucKeyBuf);
	}
	
	if (pszFormattedEncKeyPasswd)
	{
		f_free( &pszFormattedEncKeyPasswd);
	}

#endif

	return( rc);

}

typedef struct
{
	FLMUINT		uiKeyType;
	FLMUINT		uiFormatLen;
	FLMUINT		uiKeyLen;
} EXTRACTED_KEY;

/****************************************************************************
Desc:	Extract the key by encrypting it in a supplied password.  The
		buffer ppucExtractedKey buffer is allocated and returned, thus *MUST*
		be released after it is no longer needed.
****************************************************************************/
RCODE F_CCS::extractKey(
	FLMBYTE **		ppucExtractedKey,
	FLMUINT32 *		pui32Length,
	FLMUNICODE *	puzEncKeyPasswd)
{
	RCODE				rc = FERR_OK;

#ifndef FLM_USE_NICI
	F_UNREFERENCED_PARM( ppucExtractedKey);
	F_UNREFERENCED_PARM( pui32Length);
	F_UNREFERENCED_PARM( puzEncKeyPasswd);
	rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
	goto Exit;
#else
	NICI_CC_HANDLE				context =0;
	NICI_ALGORITHM				algorithm;
	NICI_ATTRIBUTE				keyAttr[2];
	NICI_ATTRIBUTE				attr[2];
	FLMBYTE						oid_sha1[] = {IDV_SHA1};
	FLMBYTE						oid_pbe[] = {IDV_pbeWithSHA1And3Key3xDES_CBC};
	FLMBYTE						ucDigest[ 20];
	FLMUINT						uiDigestLen = sizeof(ucDigest);
	FLMUINT						uiBufferSize;
	FLMBYTE *					pucKey = NULL;
	FLMBYTE *					pucFormat = NULL;
	EXTRACTED_KEY *			pExtractedKey = NULL;
	FLMUINT						uiEncLen;
	FLMBYTE *					pTemp = NULL;
	NICI_PARAMETER_INFO *	pParmInfo;
	FLMBYTE *					pucSalt;
	FLMUINT						uiAllocSize;
	FLMUINT						uiIndx;
	FLMBYTE *					pucTempPtr;

	// Create NICI Context
	
	if (CCS_CreateContext(0, &context) != 0)
	{
		rc = RC_SET( FERR_NICI_CONTEXT);
		goto Exit;
	}

	f_memset( &attr[0], 0, sizeof(NICI_ATTRIBUTE) * 2);
	
	attr[0].type = NICI_A_KEY_TYPE;
	attr[1].type = NICI_A_KEY_FORMAT;
	
	if (CCS_GetAttributeValue(context, m_keyHandle, &attr[0], 2) != 0)
	{
		rc = RC_SET( FERR_NICI_ATTRIBUTE_VALUE);
		goto Exit;
	}

	if (!attr[0].u.f.hasValue)
	{
		rc = RC_SET( FERR_NICI_BAD_ATTRIBUTE);
		goto ExitCcs;
	}

	f_memset( &keyAttr[0], 0, sizeof(NICI_ATTRIBUTE) * 2);

	switch (attr[0].u.f.value)
	{
		case NICI_K_AES:
		{
			uiIndx = 0;
			keyAttr[uiIndx].type = NICI_A_KEY_VALUE;
			keyAttr[uiIndx].u.v.valueLen = 16;

			uiIndx++;
			keyAttr[uiIndx].type = NICI_A_KEY_FORMAT;
			keyAttr[uiIndx].u.v.valueLen = attr[1].u.v.valueLen;

			break;
		}
		
		case NICI_K_DES3X:
		{
			uiIndx = 0;
			keyAttr[uiIndx].type = NICI_A_KEY_VALUE;
			keyAttr[uiIndx].u.v.valueLen = 24;

			uiIndx++;
			keyAttr[uiIndx].type = NICI_A_KEY_FORMAT;
			keyAttr[uiIndx].u.v.valueLen = attr[1].u.v.valueLen;
			break;

		}
		case NICI_K_DES:
		default:
		{
			rc = RC_SET( FERR_NICI_INVALID_ALGORITHM);
			goto ExitCcs;
		}
	}

	// Make one allocation that we can then use to hold several different things

	uiBufferSize = sizeof( EXTRACTED_KEY) + attr[1].u.v.valueLen +
						keyAttr[0].u.v.valueLen + sizeof (ucDigest);
	uiAllocSize = uiBufferSize + SALT_SZ +
					  (sizeof( NICI_PARAMETER_DATA) * 2) + sizeof( FLMUINT32);

	// Make sure the allocation size is on a 8 byte boundary
	
	if( (uiAllocSize % 8) != 0)
	{
		uiAllocSize += (8 - (uiAllocSize % 8));
	}

	if (RC_BAD( rc = f_calloc( uiAllocSize, &pExtractedKey)))
	{
		goto ExitCcs;
	}

	keyAttr[1].u.v.valuePtr = (FLMBYTE *)&pExtractedKey[1];
	pucFormat = (FLMBYTE *)keyAttr[1].u.v.valuePtr;
	keyAttr[0].u.v.valuePtr = pucFormat + attr[1].u.v.valueLen;
	pucKey = (FLMBYTE *)keyAttr[0].u.v.valuePtr;

	pucSalt = (FLMBYTE *)pExtractedKey + uiBufferSize;

	pParmInfo = (NICI_PARAMETER_INFO *)(pucSalt + SALT_SZ);

	// Make sure that pParmInfo is 8 byte alligned.
	
	if ((FLMUINT)pParmInfo % 8)
	{
		FLMBYTE *	pucTemp = (FLMBYTE *)pParmInfo + 
			(8 - ((FLMUINT)pParmInfo % 8));
		pParmInfo = (NICI_PARAMETER_INFO *)pucTemp;
	}

	// Extracted the key value now

	if (CCS_ExtractKey( context, m_keyHandle, &keyAttr[0], 2) != 0)
	{
		rc = RC_SET( FERR_EXTRACT_KEY_FAILED);
		goto Exit;
	}

	// Calculate a SHA1 checksum.

	algorithm.algorithm = oid_sha1;
	algorithm.parameter = NULL;
	algorithm.parameterLen = 0;
	
	if (CCS_DigestInit( context, &algorithm) != 0)
	{
		rc = RC_SET( FERR_DIGEST_INIT_FAILED);
		goto Exit;
	}
	
	if( CCS_Digest( context, pucFormat, 
		keyAttr[0].u.v.valueLen + attr[1].u.v.valueLen, ucDigest,
		&uiDigestLen) != 0)
	{
		rc = RC_SET( FERR_DIGEST_FAILED);
		goto Exit;
	}
	
	flmAssert( uiDigestLen == sizeof( ucDigest));

	pucTempPtr = (FLMBYTE *)pExtractedKey;

	UD2FBA( attr[0].u.f.value, pucTempPtr);
	pucTempPtr += 4;

	UD2FBA( attr[1].u.v.valueLen, pucTempPtr);
	pucTempPtr += 4;

	UD2FBA( keyAttr[0].u.v.valueLen, pucTempPtr);
	
	// Point to the digest ...

	pTemp = (FLMBYTE *)&pExtractedKey[1] +
									attr[1].u.v.valueLen +
									keyAttr[0].u.v.valueLen;
	f_memcpy( pTemp, ucDigest, uiDigestLen);

	// Generate some salt.

	if (CCS_GetRandom( context, pucSalt, SALT_SZ) != 0)
	{
		rc = RC_SET( FERR_NICI_BAD_RANDOM);
		goto Exit;
	}

	// This buffer needs to be a separate allocation because it is returned to
	// the caller.  We will be returning the value of the SALT with the
	// encrypted key.  The call to CCS_pbeEncrypt may return an extra 8 bytes.

	if (RC_BAD( rc = f_alloc( uiBufferSize + SALT_SZ + 8, &pTemp)))
	{
		goto ExitCcs;
	}
	
	// Now to encrypt the buffer.

	algorithm.algorithm = oid_pbe;
	
	pParmInfo->count = 2;
	
	pParmInfo->parms[0].parmType = NICI_P_SALT;
	pParmInfo->parms[0].u.b.len = SALT_SZ;
	pParmInfo->parms[0].u.b.ptr = pucSalt;
	
	pParmInfo->parms[1].parmType = NICI_P_COUNT;
	pParmInfo->parms[1].u.value = SALT_COUNT;

	algorithm.parameter = pParmInfo;
	algorithm.parameterLen = sizeof(NICI_PARAMETER_DATA) * 2 + sizeof(FLMUINT32);

	uiEncLen = uiBufferSize + 8;
	
	if( CCS_pbeEncrypt( context, &algorithm, puzEncKeyPasswd,
		(FLMBYTE *)pExtractedKey, uiBufferSize, pTemp, &uiEncLen) != 0)
	{
		rc = RC_SET( FERR_PBE_ENCRYPT_FAILED);
		goto Exit;
	}

	*ppucExtractedKey = pTemp;

	// Now add the salt to the end of the buffer.

	pTemp += uiEncLen;
	
	f_memcpy( pTemp, pucSalt, SALT_SZ);
	
	pTemp = NULL;
	
	*pui32Length = uiEncLen + SALT_SZ;
	
ExitCcs:

	CCS_DestroyContext(context);

#endif

Exit:
#ifdef FLM_USE_NICI
	if (pTemp)
	{
		f_free( &pTemp);
	}

	if (pucKey)
	{
		f_free( &pExtractedKey);
	}
#endif
	return(rc);
}

/****************************************************************************
Desc:	Inject the encrypting key using the supplied password.
****************************************************************************/
RCODE F_CCS::injectKey(
	FLMBYTE *		pszExtractedKey,
	FLMUINT32		ui32Length,
	FLMUNICODE *	puzEncKeyPasswd)
{
	RCODE						rc = FERR_OK;

#ifndef FLM_USE_NICI
	F_UNREFERENCED_PARM( pszExtractedKey);
	F_UNREFERENCED_PARM( ui32Length);
	F_UNREFERENCED_PARM( puzEncKeyPasswd);
	rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
	goto Exit;
#else
	NICI_CC_HANDLE				context =0;
	NICI_ALGORITHM				algorithm;
	NICI_ATTRIBUTE				keyAttr[7];
	FLMBYTE						oid_sha1[] = {IDV_SHA1};
	FLMBYTE						oid_pbe[] = {IDV_pbeWithSHA1And3Key3xDES_CBC};
	FLMUINT						uiIndx;
	FLMBYTE						ucDigest[ 20];
	FLMUINT						uiDigestLen = sizeof(ucDigest);
	FLMBYTE *					pKey;
	FLMBYTE *					pucFormat;
	EXTRACTED_KEY *			pExtractedKey;
	FLMUINT						uiEncLen;
	FLMBYTE *					pTemp;
	FLMBYTE *					pucBuffer = NULL;
	FLMBYTE *					pucSalt;
	FLMUINT						uiAllocSize;
	NICI_PARAMETER_INFO *	pParmInfo = NULL;
	FLMBYTE *					pucTempPtr;
	FLMUINT						uiKeyType;
	FLMUINT						uiFormatLen;
	FLMUINT						uiKeyLen;

	// Extract the SALT from the key buffer.
	
	pucSalt = pszExtractedKey + (ui32Length - SALT_SZ);
	ui32Length -= SALT_SZ;

	// Make one allocation and point into it for the different buffers we need.

	uiAllocSize = ui32Length +
					  sizeof(NICI_PARAMETER_DATA) * 2 + sizeof(FLMUINT32);
	
	if (RC_BAD( rc = f_calloc( uiAllocSize, &pucBuffer)))
	{
		goto Exit;
	}
	
	pParmInfo = (NICI_PARAMETER_INFO *)(pucBuffer + ui32Length);

	// Create NICI context

	if( CCS_CreateContext(0, &context) != 0)
	{
		rc = RC_SET( FERR_NICI_CONTEXT);
		goto Exit;
	}
	
	// Now to decrypt the buffer.

	algorithm.algorithm = oid_pbe;
	
	pParmInfo->count = 2;
	
	pParmInfo->parms[0].parmType = NICI_P_SALT;
	pParmInfo->parms[0].u.b.len = SALT_SZ;
	pParmInfo->parms[0].u.b.ptr = pucSalt;
	
	pParmInfo->parms[1].parmType = NICI_P_COUNT;
	pParmInfo->parms[1].u.value = SALT_COUNT;

	algorithm.parameter = pParmInfo;
	algorithm.parameterLen = sizeof(NICI_PARAMETER_DATA) * 2 + sizeof(FLMUINT32);

	uiEncLen = ui32Length;
	if( CCS_pbeDecrypt( context, &algorithm, puzEncKeyPasswd, pszExtractedKey,
		ui32Length, pucBuffer, &uiEncLen) != 0)
	{
		rc = RC_SET( FERR_PBE_DECRYPT_FAILED);
		goto Exit;
	}

	// For cross platform compatibility, we need to first extract the KeyType,
	// FormatLen and KeyLen values then we will set them back again.  They are
	// stored in a specific byte order, which may not match the native order for
	// referencing integers on the local platform.

	pExtractedKey = (EXTRACTED_KEY *)pucBuffer;
	pucTempPtr = pucBuffer;

	uiKeyType = FB2UD( pucTempPtr);
	pucTempPtr += 4;
	pExtractedKey->uiKeyType = uiKeyType;

	uiFormatLen = FB2UD( pucTempPtr);
	pucTempPtr += 4;
	pExtractedKey->uiFormatLen = uiFormatLen;

	uiKeyLen = FB2UD( pucTempPtr);
	pExtractedKey->uiKeyLen = uiKeyLen;

	// Calculate a SHA1 checksum.

	algorithm.algorithm = oid_sha1;
	algorithm.parameter = NULL;
	algorithm.parameterLen = 0;
	
	if (CCS_DigestInit( context, &algorithm) != 0)
	{
		rc = RC_SET( FERR_DIGEST_INIT_FAILED);
		goto Exit;
	}
	
	pTemp = (FLMBYTE *)&pExtractedKey[1];
	if( CCS_Digest( context, pTemp, pExtractedKey->uiFormatLen +
		pExtractedKey->uiKeyLen, ucDigest, &uiDigestLen) != 0)
	{
		rc = RC_SET( FERR_DIGEST_FAILED);
		goto Exit;
	}
	
	flmAssert( uiDigestLen == sizeof( ucDigest));

	// Now compare the two digests.  They must be equal!
	
	pTemp += pExtractedKey->uiKeyLen + pExtractedKey->uiFormatLen;
	
	if (f_memcmp( pTemp, ucDigest, uiDigestLen))
	{
		rc = RC_SET( FERR_INVALID_CRC);
		goto ExitCcs;
	}

	pucFormat = (FLMBYTE *)&pExtractedKey[1];
	pKey = pucFormat + pExtractedKey->uiFormatLen;
	
	uiIndx = 0;
	f_memset( &keyAttr[0], 0, sizeof(NICI_ATTRIBUTE) * 7);
			
	switch (pExtractedKey->uiKeyType)
	{
		case NICI_K_AES:
		{
			uiIndx = 0;
			keyAttr[uiIndx].type = NICI_A_KEY_TYPE;
			keyAttr[uiIndx].u.f.hasValue = 1;
			keyAttr[uiIndx].u.f.value = NICI_K_AES;
			keyAttr[uiIndx].u.f.valueInfo = 0;
		
			uiIndx++;
			keyAttr[uiIndx].type = NICI_A_KEY_FORMAT;
			keyAttr[uiIndx].u.v.valuePtr = pucFormat;
			keyAttr[uiIndx].u.v.valueLen = pExtractedKey->uiFormatLen;
			keyAttr[uiIndx].u.v.valueInfo = 0;
		
			uiIndx++;
			keyAttr[uiIndx].type = NICI_A_KEY_USAGE;
			keyAttr[uiIndx].u.f.hasValue = 1;
			keyAttr[uiIndx].u.f.value = NICI_F_WRAP | NICI_F_UNWRAP | NICI_F_KM_ENCRYPT | NICI_F_KM_DECRYPT | NICI_F_EXTRACT;
			keyAttr[uiIndx].u.f.valueInfo = 0;
		
			uiIndx++;
			keyAttr[uiIndx].type = NICI_A_KEY_SIZE;
			keyAttr[uiIndx].u.f.hasValue = 1;
			keyAttr[uiIndx].u.f.value = 128;
			keyAttr[uiIndx].u.f.valueInfo = 0;
		
			uiIndx++;
			keyAttr[uiIndx].type = NICI_A_KEY_VALUE;
			keyAttr[uiIndx].u.v.valuePtr = pKey;
			keyAttr[uiIndx].u.v.valueLen = pExtractedKey->uiKeyLen;
			keyAttr[uiIndx].u.v.valueInfo = 0;
		
			uiIndx++;
			keyAttr[uiIndx].type = NICI_A_CLASS;
			keyAttr[uiIndx].u.f.hasValue = 1;
			keyAttr[uiIndx].u.f.value = NICI_O_SECRET_KEY;
			keyAttr[uiIndx].u.f.valueInfo = 0;

			uiIndx++;
			keyAttr[uiIndx].type = NICI_A_GLOBAL;
			keyAttr[uiIndx].u.f.hasValue = 1;
			keyAttr[uiIndx].u.f.value = N_TRUE;
			keyAttr[uiIndx].u.f.valueInfo = 0;
			break;
		}
		case NICI_K_DES3X:
		{
			uiIndx = 0;
			keyAttr[uiIndx].type = NICI_A_KEY_TYPE;
			keyAttr[uiIndx].u.f.hasValue = 1;
			keyAttr[uiIndx].u.f.value = NICI_K_DES3X;
			keyAttr[uiIndx].u.f.valueInfo = 0;
		
			uiIndx++;
			keyAttr[uiIndx].type = NICI_A_KEY_FORMAT;
			keyAttr[uiIndx].u.v.valuePtr = pucFormat;
			keyAttr[uiIndx].u.v.valueLen = pExtractedKey->uiFormatLen;
			keyAttr[uiIndx].u.v.valueInfo = 0;
		
			uiIndx++;
			keyAttr[uiIndx].type = NICI_A_KEY_USAGE;
			keyAttr[uiIndx].u.f.hasValue = 1;
			keyAttr[uiIndx].u.f.value = NICI_F_WRAP | NICI_F_UNWRAP | NICI_F_KM_ENCRYPT | NICI_F_KM_DECRYPT | NICI_F_EXTRACT;
			keyAttr[uiIndx].u.f.valueInfo = 0;
		
			uiIndx++;
			keyAttr[uiIndx].type = NICI_A_KEY_SIZE;
			keyAttr[uiIndx].u.f.hasValue = 1;
			keyAttr[uiIndx].u.f.value = 168;
			keyAttr[uiIndx].u.f.valueInfo = 0;

			uiIndx++;
			keyAttr[uiIndx].type = NICI_A_KEY_VALUE;
			keyAttr[uiIndx].u.v.valuePtr = pKey;
			keyAttr[uiIndx].u.v.valueLen = pExtractedKey->uiKeyLen;
			keyAttr[uiIndx].u.v.valueInfo = 0;
		
			uiIndx++;
			keyAttr[uiIndx].type = NICI_A_CLASS;
			keyAttr[uiIndx].u.f.hasValue = 1;
			keyAttr[uiIndx].u.f.value = NICI_O_SECRET_KEY;
			keyAttr[uiIndx].u.f.valueInfo = 0;

			uiIndx++;
			keyAttr[uiIndx].type = NICI_A_GLOBAL;
			keyAttr[uiIndx].u.f.hasValue = 1;
			keyAttr[uiIndx].u.f.value = N_TRUE;
			keyAttr[uiIndx].u.f.valueInfo = 0;
			break;
		}
		case NICI_K_DES:
		default:
		{
			rc = RC_SET( FERR_NICI_INVALID_ALGORITHM);
			goto ExitCcs;
		}
	}


	if (CCS_InjectKey( context,
							  &keyAttr[0],
							  7,
							  &m_keyHandle) != 0)
	{
		rc = RC_SET( FERR_INJECT_KEY_FAILED);
		goto Exit;
	}

	ExitCcs:

	CCS_DestroyContext(context);

#endif

Exit:
#ifdef FLM_USE_NICI
	if (pucBuffer)
	{
		f_free( &pucBuffer);
	}
#endif
	return(rc);
}

/****************************************************************************
Desc: flmDecryptBuffer - assumes aes
****************************************************************************/
RCODE flmDecryptBuffer(
	FLMBYTE *	pucBuffer,
	FLMUINT *	puiBufLen)
{
	RCODE			rc = FERR_OK;

#ifndef FLM_USE_NICI
	F_UNREFERENCED_PARM( pucBuffer);
	F_UNREFERENCED_PARM( puiBufLen);
	rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
	goto Exit;
#else
	NICI_ATTRIBUTE				find[2];
	NICI_CC_HANDLE				context =0;
	NICI_OBJECT_HANDLE		serverKeyHdl = 0;
	FLMUINT						uiCount;
	NICI_ALGORITHM				algorithm;
	NICI_PARAMETER_INFO		parm[1];
	FLMBYTE						oid_aes[] = {IDV_AES128CBC};
	FLMBYTE						pucIV[ IV_SZ];

	// Create NICI Context
	
	if( CCS_CreateContext(0, &context) != 0)
	{
		rc = RC_SET( FERR_NICI_CONTEXT);
		goto Exit;
	}

	find[0].type = NICI_A_GLOBAL;
	find[0].u.f.hasValue = 1;
	find[0].u.f.value = 1;
	find[0].u.f.valueInfo = 0;

	find[1].type = NICI_A_FEATURE;
	find[1].u.f.hasValue = 1;
	find[1].u.f.value = NICI_F_DATA_ENCRYPT | NICI_F_DATA_DECRYPT;
	find[1].u.f.valueInfo = 0;

	if (CCS_FindObjectsInit(context, find, 2) != 0)
	{
		rc = RC_SET( FERR_NICI_FIND_INIT);
		goto Exit;
	}

	uiCount = 1;
	if (CCS_FindObjects(context, &serverKeyHdl, &uiCount) != 0)
	{
		rc = RC_SET( FERR_NICI_FIND_OBJECT);
		goto Exit;
	}

	if (uiCount < 1)
	{
		rc = RC_SET( FERR_NICI_KEY_NOT_FOUND);
		goto ExitCtx;
	}

	// Set up alogrithm now to do AES and pading for encryption
	
	algorithm.algorithm = oid_aes;
	algorithm.parameterLen = sizeof(algorithm.parameter->parms[0])+
								sizeof(algorithm.parameter->count);
	algorithm.parameter = parm;
	algorithm.parameter->count = 1;
	algorithm.parameter->parms[0].parmType = NICI_P_IV;
	algorithm.parameter->parms[0].u.b.len = IV_SZ;
	algorithm.parameter->parms[0].u.b.ptr = pucIV;

	// Init encryption
	
	if( CCS_DataDecryptInit(context, &algorithm, serverKeyHdl) != 0)
	{
		rc = RC_SET( FERR_NICI_DECRYPT_INIT_FAILED);
	 	goto Exit;
	}

	if( CCS_Decrypt( context, pucBuffer, *puiBufLen, pucBuffer,
		puiBufLen) != 0)
	{
		rc = RC_SET( FERR_NICI_DECRYPT_FAILED);
		goto Exit;
	}

ExitCtx:

	CCS_DestroyContext(context);

#endif

Exit:

	return( rc);

}

/****************************************************************************
Desc:
****************************************************************************/
RCODE flmEncryptBuffer(
	FLMBYTE *				pucBuffer,
	FLMUINT *				puiBufLen)
{
	RCODE							rc = FERR_OK;

#ifndef FLM_USE_NICI
	F_UNREFERENCED_PARM( pucBuffer);
	F_UNREFERENCED_PARM( puiBufLen);
	rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
	goto Exit;
#else
	NICI_ATTRIBUTE				find[2];
	NICI_CC_HANDLE				context =0;
	NICI_OBJECT_HANDLE		serverKeyHdl = 0;
	FLMUINT						uiCount;
	NICI_ALGORITHM				algorithm;
	NICI_PARAMETER_INFO		parm[1];
	FLMBYTE						oid_aes[] = {IDV_AES128CBC};
	FLMBYTE						pucIV[ IV_SZ];

	// Create NICI Context
	
	if (CCS_CreateContext(0, &context) != 0)
	{
		rc = RC_SET( FERR_NICI_CONTEXT);
		goto Exit;
	}

	find[0].type = NICI_A_GLOBAL;
	find[0].u.f.hasValue = 1;
	find[0].u.f.value = 1;
	find[0].u.f.valueInfo = 0;

	find[1].type = NICI_A_FEATURE;
	find[1].u.f.hasValue = 1;
	find[1].u.f.value = NICI_F_DATA_ENCRYPT | NICI_F_DATA_DECRYPT;
	find[1].u.f.valueInfo = 0;

	if (CCS_FindObjectsInit(context, find, 2) != 0)
	{
		rc = RC_SET( FERR_NICI_FIND_INIT);
		goto Exit;
	}

	uiCount = 1;
	if (CCS_FindObjects(context, &serverKeyHdl, &uiCount) != 0)
	{
		rc = RC_SET( FERR_NICI_FIND_OBJECT);
		goto Exit;
	}

	if (uiCount < 1)
	{
		rc = RC_SET( FERR_NICI_KEY_NOT_FOUND);
		goto ExitCtx;
	}


	algorithm.algorithm = oid_aes;
	algorithm.parameterLen = sizeof(algorithm.parameter->parms[0])+
								sizeof(algorithm.parameter->count);
	algorithm.parameter = parm;
	algorithm.parameter->count = 1;
	algorithm.parameter->parms[0].parmType = NICI_P_IV;
	algorithm.parameter->parms[0].u.b.len = IV_SZ;
	algorithm.parameter->parms[0].u.b.ptr = pucIV;

	GetIV(pucIV, IV_SZ);

	// Init encryption
	
	if (CCS_DataEncryptInit(context, &algorithm, serverKeyHdl) != 0)
	{
		rc = RC_SET( FERR_NICI_ENC_INIT_FAILED);
	 	goto Exit;
	}

	if (CCS_Encrypt(context,
						  pucBuffer,
						  *puiBufLen,
						  pucBuffer,
						  puiBufLen) != 0)
	{
		rc = RC_SET( FERR_NICI_ENCRYPT_FAILED);
		goto Exit;
	}

ExitCtx:

	CCS_DestroyContext( context);

#endif

Exit:

	return( rc);

}

/*****************************************************************************
Desc:
*****************************************************************************/
#ifdef FLM_USE_NICI
FSTATIC void GetIV(
	FLMBYTE *		pucIV,
	FLMUINT			//uiLen
	)
{
	FLMUINT			uiLoop;
	FLMUINT			uiLoop2;

	f_strcpy( (char *)pucIV, "3587903781145935");

	for (uiLoop = 0; uiLoop < 100; uiLoop++)
	{
		for ( uiLoop2 = 0; uiLoop2 < IV_SZ; uiLoop2++)
		{
			pucIV[IV_SZ - uiLoop2] ^= pucIV[ uiLoop2];
			pucIV[IV_SZ - uiLoop2] += pucIV[ uiLoop2];
			pucIV[IV_SZ - uiLoop2] ^= pucIV[ uiLoop2];
		}

	}
}
#endif

/*****************************************************************************
Desc:
*****************************************************************************/
#if defined( FLM_USE_NICI) && !defined( FLM_UNIX)
int CCSX_SetNewIV(
	int				,	// MODULEID,
	FLMUINT32		,	// hContext,
	pnuint8			,	// IV,
	nuint32)				// IVLen
{
	return( NICI_E_FUNCTION_NOT_SUPPORTED);
}
#endif
