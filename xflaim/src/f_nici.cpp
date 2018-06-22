//------------------------------------------------------------------------------
// Desc:	This file contains the functions needed for the NICI interface
//			functions. Adapted from ss_crypto.c written by Cameron Mashayekhi.
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

#ifdef FLM_NLM
	#define N_PLAT_NLM
#endif

#include "nwccs.h"

#ifndef IDV_NOV_AES128CBCPad
	#define IDV_NOV_AES128CBCPad			NICI_AlgorithmPrefix(1), 97
#endif

/****************************************************************************
Desc:
****************************************************************************/

#define IV_SZ						16
#define IV_SZ8						8
#define SALT_SZ					8
#define SALT_COUNT				895

/****************************************************************************
Desc:
****************************************************************************/
typedef struct
{
	FLMUINT		uiKeyType;
	FLMUINT		uiFormatLen;
	FLMUINT		uiKeyLen;
	FLMUINT		uiKeySize;
} EXTRACTED_KEY;

/****************************************************************************
Desc:
****************************************************************************/
class F_NICICCS : public IF_CCS
{
public:

	F_NICICCS()
	{
		m_bInitCalled = FALSE;
		m_bKeyVerified = FALSE;
		f_memset( m_ucIV, 0, IV_SZ);
		m_keyHandle = 0;
		m_hContext = 0;
		m_uiEncKeySize = 0;
		m_hMutex = F_MUTEX_NULL;
	}

	~F_NICICCS();

	RCODE init(
		FLMBOOL						bKeyIsWrappingKey,
		FLMUINT						uiAlgType);

	RCODE generateEncryptionKey(
		FLMUINT						uiEncKeySize);

	RCODE generateWrappingKey(
		FLMUINT						uiEncKeySize);

	RCODE encryptToStore(
		FLMBYTE *					pucIn,
		FLMUINT						uiInLen,
		FLMBYTE *					pucOut,
		FLMUINT *					puiOutLen,
		FLMBYTE *					pucIV = NULL);

	RCODE decryptFromStore(
		FLMBYTE *					pucIn,
		FLMUINT						uiInLen,
		FLMBYTE *					pucOut,
		FLMUINT *					puiOutLen,
		FLMBYTE *					pucIV = NULL);

	RCODE getKeyToStore(
		FLMBYTE **					ppucKeyInfo,
		FLMUINT32 *					pui32BufLen,
		FLMBYTE *					pzEncKeyPasswd,
		IF_CCS *						pWrappingCcs);

	RCODE setKeyFromStore(
		FLMBYTE *					pucKeyInfo,
		FLMBYTE *					pszEncKeyPasswd = NULL,
		IF_CCS *						pWrappingCcs = NULL);

	FINLINE FLMBOOL keyVerified( void)
	{
		return( m_bKeyVerified);
	}

	FINLINE FLMUINT getEncType( void)
	{
		return( m_uiAlgType);
	}
	
	FLMUINT getIVLen( void);
	
	RCODE generateIV(
		FLMUINT						uiIVLen,
		FLMBYTE *					pucIV);

	RCODE getWrappingKey(
		NICI_OBJECT_HANDLE *		pWrappingKeyHandle);

	RCODE wrapKey(
		FLMBYTE **					ppucWrappedKey,
		FLMUINT32 *					pui32Length,
		NICI_OBJECT_HANDLE		masterWrappingKey = 0 );

	RCODE unwrapKey(
		FLMBYTE *					pucWrappedKey,
		FLMUINT32					ui32WrappedKeyLength,
		NICI_OBJECT_HANDLE		masterWrappingKey = 0);

	RCODE extractKey(
		FLMBYTE **					ppucShroudedKey,
		FLMUINT32 *					pui32Length,
		FLMUNICODE *				puzEncKeyPasswd );

	RCODE injectKey(
		FLMBYTE *					pucBuffer,
		FLMUINT32					ui32Length,
		FLMUNICODE *				puzEncKeyPasswd );

	RCODE encryptToStoreAES(
		FLMBYTE *					pucIn,
		FLMUINT						uiInLen,
		FLMBYTE *					pucOut,
		FLMUINT *					puiOutLen,
		FLMBYTE *					pucIV);

	RCODE encryptToStoreDES3(
		FLMBYTE *					pucIn,
		FLMUINT						uiInLen,
		FLMBYTE *					pucOut,
		FLMUINT *					puiOutLen,
		FLMBYTE *					pucIV);

	RCODE encryptToStoreDES(
		FLMBYTE *					pucIn,
		FLMUINT						uiInLen,
		FLMBYTE *					pucOut,
		FLMUINT *					puiOutLen,
		FLMBYTE *					pucIV);

	RCODE decryptFromStoreAES(
		FLMBYTE *					pucIn,
		FLMUINT						uiInLen,
		FLMBYTE *					pucOut,
		FLMUINT *					puiOutLen,
		FLMBYTE *					pucIV);

	RCODE decryptFromStoreDES3(
		FLMBYTE *					pucIn,
		FLMUINT						uiInLen,
		FLMBYTE *					pucOut,
		FLMUINT *					puiOutLen,
		FLMBYTE *					pucIV);

	RCODE decryptFromStoreDES(
		FLMBYTE *					pucIn,
		FLMUINT						uiInLen,
		FLMBYTE *					pucOut,
		FLMUINT *					puiOutLen,
		FLMBYTE *					pucIV);

	RCODE generateEncryptionKeyAES(
		FLMUINT						uiEncKeySize);

	RCODE generateEncryptionKeyDES3(
		FLMUINT						uiEncKeySize);

	RCODE generateEncryptionKeyDES(
		FLMUINT						uiEncKeySize);

	RCODE generateWrappingKeyAES(
		FLMUINT						uiEncKeySize);

	RCODE generateWrappingKeyDES3(
		FLMUINT						uiEncKeySize);

	RCODE generateWrappingKeyDES(
		FLMUINT						uiEncKeySize);

	FLMUINT							m_uiAlgType;
	FLMBOOL							m_bInitCalled;
	FLMBOOL							m_bKeyIsWrappingKey;
	FLMBOOL							m_bKeyVerified;
	NICI_OBJECT_HANDLE			m_keyHandle;			// Handle to the clear key - we don't ever get the actual key.
	FLMBYTE							m_ucIV[ IV_SZ];		// Used when the algorithm type is DES, 3DES or AES
	FLMBYTE							m_ucRndIV[ IV_SZ];	// Used when the IV is stored with the data.
	FLMUINT							m_uiIVFactor;
	NICI_CC_HANDLE					m_hContext;
	FLMUINT							m_uiEncKeySize;
	F_MUTEX							m_hMutex;
};
	
/****************************************************************************
Desc:
****************************************************************************/
F_NICICCS::~F_NICICCS()
{
	if( m_keyHandle)
	{
		if( !m_hContext)
		{
			if( RC_BAD( CCS_CreateContext(0, &m_hContext)))
			{
				flmAssert( 0);
			}
		}

		// Get rid of the key handle.
		
		if( m_hContext)
		{
			CCS_DestroyObject( m_hContext, m_keyHandle);
			CCS_DestroyContext( m_hContext);
		}
	}

	if (m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}
}

/****************************************************************************
Desc:	Save the wrapped key in m_pKey.
Note:	Make sure there is a buffer allocated for the wrapped key.
****************************************************************************/
RCODE F_NICICCS::wrapKey(
	FLMBYTE **				ppucWrappedKey,
	FLMUINT32 *				pui32Length,
	NICI_OBJECT_HANDLE	masterWrappingKey)
{
	RCODE						rc = NE_XFLM_OK;
	NICI_ATTRIBUTE			wKey[2];
	NICI_ALGORITHM			algorithm;
	NICI_PARAMETER_INFO	parm[1];
	FLMBYTE					oid_aes128[] = {IDV_NOV_AES128CBCPad};
	FLMBYTE					oid_aes192[] = {IDV_NOV_AES192CBCPad};
	FLMBYTE					oid_aes256[] = {IDV_NOV_AES256CBCPad};
	FLMBYTE					oid_3des[] = {IDV_DES_EDE3_CBCPadIV8};
	NICI_OBJECT_HANDLE	wrappingKeyHandle;
	FLMBOOL					bLocked = FALSE;

	if (masterWrappingKey)
	{
		wrappingKeyHandle = masterWrappingKey;
	}
	else
	{
		if (RC_BAD( rc = getWrappingKey( &wrappingKeyHandle)))
		{
			goto Exit;
		}
	}

	f_mutexLock( m_hMutex);
	bLocked = TRUE;

	// Create NICI Context
	
	if( !m_hContext)
	{
		if( RC_BAD( rc = CCS_CreateContext( 0, &m_hContext)))
		{
			rc = RC_SET( NE_XFLM_NICI_CONTEXT);
			m_hContext = 0;
			goto Exit;
		}
	}

	f_memset( &wKey, 0, sizeof(NICI_ATTRIBUTE) * 2);

	wKey[0].type = NICI_A_KEY_TYPE;
	wKey[1].type = NICI_A_KEY_SIZE;
	
	if( RC_BAD( rc = CCS_GetAttributeValue( m_hContext, wrappingKeyHandle,
		&wKey[0], 2)))
	{
		rc = RC_SET( NE_XFLM_NICI_ATTRIBUTE_VALUE);
		m_hContext = 0;
		goto Exit;
	}

	if( !wKey[0].u.f.hasValue || !wKey[1].u.f.hasValue)
	{
		rc = RC_SET( NE_XFLM_NICI_BAD_ATTRIBUTE);
		goto Exit;
	}

	switch( wKey[0].u.f.value)
	{
		case NICI_K_AES:
		{
			switch( wKey[1].u.f.value)
			{
				case XFLM_NICI_AES128:
				{
					algorithm.algorithm = (nuint8 *)oid_aes128;
					break;
				}
				
				case XFLM_NICI_AES192:
				{
					algorithm.algorithm = (nuint8 *)oid_aes192;
					break;
				}
				
				case XFLM_NICI_AES256:
				{
					algorithm.algorithm = (nuint8 *)oid_aes256;
					break;
				}
				
				default:
				{
					rc = RC_SET( NE_XFLM_INVALID_ENC_KEY_SIZE);
					goto Exit;
				}
			}
			
			algorithm.parameter = parm;
			algorithm.parameterLen = sizeof(algorithm.parameter->parms[0])+
									sizeof(algorithm.parameter->count);
			algorithm.parameter->count = 1;
			algorithm.parameter->parms[0].parmType = NICI_P_IV;
			algorithm.parameter->parms[0].u.b.len = IV_SZ; /* 16-byte IV */
			algorithm.parameter->parms[0].u.b.ptr = (nuint8 *)m_ucIV;
			break;
		}

		case NICI_K_DES3X:
		{
			algorithm.algorithm = (nuint8 *)oid_3des;
			algorithm.parameter = parm;
			algorithm.parameterLen = sizeof(algorithm.parameter->parms[0])+
									sizeof(algorithm.parameter->count);
			algorithm.parameter->count = 1;
			algorithm.parameter->parms[0].parmType = NICI_P_IV;
			algorithm.parameter->parms[0].u.b.len = IV_SZ8; /* 8-byte IV */
			algorithm.parameter->parms[0].u.b.ptr = (nuint8 *)m_ucIV;
			break;
		}

		default:
		{
			rc = RC_SET( NE_XFLM_NICI_WRAPKEY_FAILED);
			goto Exit;
		}
	}

	// We should be able to call this with NULL for the 
	// wrapped key, to get the length.
	
	if( RC_BAD( rc = CCS_WrapKey( m_hContext, &algorithm, NICI_KM_UNSPECIFIED,
		0, wrappingKeyHandle, m_keyHandle, (nuint8 *)NULL, 
		(pnuint32)pui32Length)))
	{
		rc = RC_SET( NE_XFLM_NICI_WRAPKEY_FAILED);
		m_hContext = 0;
		goto Exit;
	}

	if( RC_BAD( rc = f_calloc( *pui32Length, ppucWrappedKey)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = CCS_WrapKey( m_hContext, &algorithm, NICI_KM_UNSPECIFIED,
		0, wrappingKeyHandle, m_keyHandle, (nuint8 *)*ppucWrappedKey,
		(pnuint32)pui32Length)))
	{
		rc = RC_SET( NE_XFLM_NICI_WRAPKEY_FAILED);
		m_hContext = 0;
		goto Exit;
	}

Exit:

	if (bLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return(rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_NICICCS::unwrapKey(
	FLMBYTE *				pucWrappedKey,
	FLMUINT32				ui32WrappedKeyLength,
	NICI_OBJECT_HANDLE	masterWrappingKey)
{
	RCODE						rc = NE_XFLM_OK;
	NICI_ATTRIBUTE			wKey;
	NICI_OBJECT_HANDLE	wrappingKeyHandle;
	FLMBOOL					bLocked = FALSE;

	if (masterWrappingKey)
	{
		wrappingKeyHandle = masterWrappingKey;
	}
	else
	{
		if (RC_BAD( rc = getWrappingKey( &wrappingKeyHandle)))
		{
			goto Exit;
		}
	}

	f_mutexLock( m_hMutex);
	bLocked = TRUE;

	// Create NICI Context
	
	if( !m_hContext)
	{
		if( RC_BAD( rc = CCS_CreateContext( 0, &m_hContext)))
		{
			rc = RC_SET( NE_XFLM_NICI_CONTEXT);
			m_hContext = 0;
			goto Exit;
		}
	}

	if( RC_BAD( rc = CCS_UnwrapKey( m_hContext, wrappingKeyHandle,
		(nuint8 *)pucWrappedKey, ui32WrappedKeyLength, &m_keyHandle)))
	{
		rc = RC_SET( NE_XFLM_NICI_UNWRAPKEY_FAILED);
		m_hContext = 0;
		goto Exit;
	}

	// We need to get the key size...
	
	f_memset( &wKey, 0, sizeof( NICI_ATTRIBUTE));

	wKey.type = NICI_A_KEY_SIZE;
	
	if( RC_BAD( rc = CCS_GetAttributeValue( m_hContext, m_keyHandle, &wKey, 1)))
	{
		rc = RC_SET( NE_XFLM_NICI_ATTRIBUTE_VALUE);
		m_hContext = 0;
		goto Exit;
	}

	if( !wKey.u.f.hasValue)
	{
		rc = RC_SET( NE_XFLM_NICI_BAD_ATTRIBUTE);
		goto Exit;
	}
	
	m_uiEncKeySize = wKey.u.f.value;

Exit:

	if( bLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return(rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_NICICCS::generateEncryptionKey(
	FLMUINT			uiEncKeySize)
{
	RCODE			rc = NE_XFLM_OK;

	switch( m_uiAlgType)
	{
		case FLM_NICI_AES:
		{
			rc = generateEncryptionKeyAES( uiEncKeySize);
			break;
		}
		
		case FLM_NICI_DES3:
		{
			rc = generateEncryptionKeyDES3( uiEncKeySize);
			break;
		}
		
		default:
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_NICI_INVALID_ALGORITHM);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_NICICCS::generateEncryptionKeyAES(
	FLMUINT				uiEncKeySize)
{
	RCODE					rc = NE_XFLM_OK;
	NICI_ALGORITHM		algorithm;
	NICI_ATTRIBUTE		keyAttr[3];
	nbool8				keySizeChanged;
	FLMBYTE				oid_aes128[] = {IDV_AES128CBC};
	FLMBYTE				oid_aes192[] = {IDV_AES192CBC};
	FLMBYTE				oid_aes256[] = {IDV_AES256CBC};

	f_mutexLock( m_hMutex);

	// Create NICI Context
	
	if( !m_hContext)
	{
		if( RC_BAD( rc = CCS_CreateContext( 0, &m_hContext)))
		{
			rc = RC_SET( NE_XFLM_NICI_CONTEXT);
			m_hContext = 0;
			goto Exit;
		}
	}

	// Set up AES Algorithm
	
	switch( uiEncKeySize)
	{
		case XFLM_NICI_AES128:
		{
			algorithm.algorithm = (nuint8 *)oid_aes128;
			break;
		}
		
		case XFLM_NICI_AES192:
		{
			algorithm.algorithm = (nuint8 *)oid_aes192;
			break;
		}
		
		case XFLM_NICI_AES256:
		{
			algorithm.algorithm = (nuint8 *)oid_aes256;
			break;
		}
		
		default:
		{
			rc = RC_SET( NE_XFLM_INVALID_ENC_KEY_SIZE);
			goto Exit;
		}
	}
	
	algorithm.parameterLen = 0;
	algorithm.parameter = NULL;

	// Set up key attributes
	
	keyAttr[0].type = NICI_A_KEY_USAGE;
	keyAttr[0].u.f.hasValue = 1;
	keyAttr[0].u.f.value = NICI_F_DATA_ENCRYPT | NICI_F_DATA_DECRYPT | NICI_F_EXTRACT;
	keyAttr[0].u.f.valueInfo = 0;

	keyAttr[1].type = NICI_A_KEY_SIZE;
	keyAttr[1].u.f.hasValue = 1;
	keyAttr[1].u.f.value = uiEncKeySize;
	keyAttr[1].u.f.valueInfo = 0;

	keyAttr[2].type = NICI_A_GLOBAL;
	keyAttr[2].u.f.hasValue = 1;
	keyAttr[2].u.f.value = N_TRUE;
	keyAttr[2].u.f.valueInfo = 0;

	// Generate a AES key

	if( RC_BAD( rc = CCS_GenerateKey( m_hContext, &algorithm, keyAttr, 3,
		&keySizeChanged, &m_keyHandle, NICI_H_INVALID)))
	{
		rc = RC_SET( NE_XFLM_NICI_GENKEY_FAILED);
		m_hContext = 0;
		goto Exit;
	}

	// Generate some IV to use with this key.

	if( RC_BAD( rc = CCS_GetRandom( m_hContext, (nuint8 *)m_ucIV, IV_SZ)))
	{
		rc = RC_SET( NE_XFLM_NICI_BAD_RANDOM);
		m_hContext = 0;
		goto Exit;
	}

	m_uiEncKeySize = uiEncKeySize;

Exit:

	f_mutexUnlock( m_hMutex);
	return(rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_NICICCS::generateEncryptionKeyDES3(
	FLMUINT				uiEncKeySize)
{
	RCODE					rc = NE_XFLM_OK;
	NICI_ALGORITHM		algorithm;
	NICI_ATTRIBUTE		keyAttr[3];
	nbool8				keySizeChanged;
	FLMBYTE				oid_des3[] = {IDV_DES_EDE3_CBC_IV8};

	f_mutexLock( m_hMutex);

	// Only one DES3 key size supported.

	if( uiEncKeySize != XFLM_NICI_DES3X)
	{
		rc = RC_SET( NE_XFLM_INVALID_ENC_KEY_SIZE);
		goto Exit;
	}

	// Create NICI Context
	
	if( !m_hContext)
	{
		if( RC_BAD( rc = CCS_CreateContext( 0, &m_hContext)))
		{
			rc = RC_SET( NE_XFLM_NICI_CONTEXT);
			m_hContext = 0;
			goto Exit;
		}
	}

	// Set up AES Algorithm
	
	algorithm.algorithm = (nuint8 *)oid_des3;
	algorithm.parameterLen = 0;
	algorithm.parameter = NULL;

	// Set up key attributes
	
	keyAttr[0].type = NICI_A_KEY_USAGE;
	keyAttr[0].u.f.hasValue = 1;
	keyAttr[0].u.f.value = NICI_F_DATA_ENCRYPT | NICI_F_DATA_DECRYPT | NICI_F_EXTRACT;
	keyAttr[0].u.f.valueInfo = 0;

	keyAttr[1].type = NICI_A_KEY_SIZE;
	keyAttr[1].u.f.hasValue = 1;
	keyAttr[1].u.f.value = uiEncKeySize;
	keyAttr[1].u.f.valueInfo = 0;

	keyAttr[2].type = NICI_A_GLOBAL;
	keyAttr[2].u.f.hasValue = 1;
	keyAttr[2].u.f.value = N_TRUE;
	keyAttr[2].u.f.valueInfo = 0;

	// Generate a AES key

	if( RC_BAD( rc = CCS_GenerateKey( m_hContext, &algorithm, keyAttr,
		3, &keySizeChanged, &m_keyHandle, NICI_H_INVALID)))
	{
		rc = RC_SET( NE_XFLM_NICI_GENKEY_FAILED);
		m_hContext = 0;
		goto Exit;
	}

	// Generate some IV to use with this key.

	if( RC_BAD( rc = CCS_GetRandom( m_hContext, (nuint8 *)m_ucIV, IV_SZ)))
	{
		rc = RC_SET( NE_XFLM_NICI_BAD_RANDOM);
		m_hContext = 0;
		goto Exit;
	}

	m_uiEncKeySize = uiEncKeySize;

Exit:

	f_mutexUnlock( m_hMutex);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_NICICCS::generateWrappingKey(
	FLMUINT			uiEncKeySize)
{
	RCODE				rc = NE_XFLM_OK;

	switch( m_uiAlgType)
	{
		case FLM_NICI_AES:
		{
			rc = generateWrappingKeyAES( uiEncKeySize);
			break;
		}
		
		case FLM_NICI_DES3:
		{
			rc = generateWrappingKeyDES3( uiEncKeySize);
			break;
		}
		
		default:
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_NICI_INVALID_ALGORITHM);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_NICICCS::generateWrappingKeyAES(
	FLMUINT				uiEncKeySize)
{
	RCODE					rc = NE_XFLM_OK;
	NICI_ALGORITHM		algorithm;
	NICI_ATTRIBUTE		keyAttr[6];
	nbool8				keySizeChanged;
	FLMBYTE				oid_aes128[] = {IDV_AES128CBC};
	FLMBYTE				oid_aes192[] = {IDV_AES192CBC};
	FLMBYTE				oid_aes256[] = {IDV_AES256CBC};

	f_mutexLock( m_hMutex);

	// Create NICI Context
	
	if( !m_hContext)
	{
		if( RC_BAD( rc = CCS_CreateContext( 0, &m_hContext)))
		{
			rc = RC_SET( NE_XFLM_NICI_CONTEXT);
			m_hContext = 0;
			goto Exit;
		}
	}

	// Set up AES Algorithm
	
	switch( uiEncKeySize)
	{
		case XFLM_NICI_AES128:
		{
			algorithm.algorithm = (nuint8 *)oid_aes128;
			keyAttr[1].u.v.valuePtr = oid_aes128;
			keyAttr[1].u.v.valueLen = (nuint32)sizeof( oid_aes128);
			break;
		}
		
		case XFLM_NICI_AES192:
		{
			algorithm.algorithm = (nuint8 *)oid_aes192;
			keyAttr[1].u.v.valuePtr = oid_aes192;
			keyAttr[1].u.v.valueLen = (nuint32)sizeof( oid_aes192);
			break;
		}
		
		case XFLM_NICI_AES256:
		{
			algorithm.algorithm = (nuint8 *)oid_aes256;
			keyAttr[1].u.v.valuePtr = oid_aes256;
			keyAttr[1].u.v.valueLen = (nuint32)sizeof( oid_aes256);
			break;
		}
		
		default:
		{
			rc = RC_SET( NE_XFLM_INVALID_ENC_KEY_SIZE);
			goto Exit;
		}
	}
	
	algorithm.parameterLen = 0;
	algorithm.parameter = NULL;

	// Set up key attributes
	
	keyAttr[0].type = NICI_A_KEY_TYPE;
	keyAttr[0].u.f.hasValue = 1;
	keyAttr[0].u.f.value = NICI_K_AES;
	keyAttr[0].u.f.valueInfo = 0;

	keyAttr[1].type = NICI_A_KEY_FORMAT;
	keyAttr[1].u.v.valueInfo = 0;

	keyAttr[2].type = NICI_A_KEY_USAGE;
	keyAttr[2].u.f.hasValue = 1;
	keyAttr[2].u.f.value = NICI_F_WRAP | NICI_F_UNWRAP | NICI_F_KM_ENCRYPT | 
								  NICI_F_KM_DECRYPT | NICI_F_EXTRACT | 
								  NICI_F_DATA_ENCRYPT | NICI_F_DATA_DECRYPT;
	keyAttr[2].u.f.valueInfo = 0;

	keyAttr[3].type = NICI_A_KEY_SIZE;
	keyAttr[3].u.f.hasValue = 1;
	keyAttr[3].u.f.value = uiEncKeySize;
	keyAttr[3].u.f.valueInfo = 0;

	keyAttr[4].type = NICI_A_GLOBAL;
	keyAttr[4].u.f.hasValue = 1;
	keyAttr[4].u.f.value = N_TRUE;
	keyAttr[4].u.f.valueInfo = 0;

	keyAttr[5].type = NICI_A_CLASS;
	keyAttr[5].u.f.hasValue = 1;
	keyAttr[5].u.f.value = NICI_O_SECRET_KEY;
	keyAttr[5].u.f.valueInfo = 0;

	// Generate an AES wrapping key

	if( RC_BAD( rc = CCS_GenerateKey( m_hContext, &algorithm, keyAttr, 6,
		&keySizeChanged, &m_keyHandle, NICI_H_INVALID)))
	{
		rc = RC_SET( NE_XFLM_NICI_GENKEY_FAILED);
		m_hContext = 0;
		goto Exit;
	}

	// Generate some IV to use with this key.

	if( RC_BAD( rc = CCS_GetRandom( m_hContext, (nuint8 *)m_ucIV, IV_SZ)))
	{
		rc = RC_SET( NE_XFLM_NICI_BAD_RANDOM);
		m_hContext = 0;
		goto Exit;
	}

	// If we generated a wrapping key, then this object's key handle is 
	// actually a wrapping key. This means that we will use it to wrap the 
	// other keys in the system.

	m_bKeyIsWrappingKey = TRUE;
	m_uiEncKeySize = uiEncKeySize;

Exit:

	f_mutexUnlock( m_hMutex);
	return(rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_NICICCS::generateWrappingKeyDES3(
	FLMUINT				uiEncKeySize)
{
	RCODE					rc = NE_XFLM_OK;
	NICI_ALGORITHM		algorithm;
	NICI_ATTRIBUTE		keyAttr[ 6];
	nbool8				keySizeChanged;
	FLMBYTE				oid_des3[] = {IDV_DES_EDE3_CBC_IV8};

	f_mutexLock( m_hMutex);

	if( uiEncKeySize != XFLM_NICI_DES3X)
	{
		rc = RC_SET( NE_XFLM_INVALID_ENC_KEY_SIZE);
		goto Exit;
	}

	// Create NICI Context
	
	if( !m_hContext)
	{
		if( RC_BAD( rc = CCS_CreateContext( 0, &m_hContext)))
		{
			rc = RC_SET( NE_XFLM_NICI_CONTEXT);
			m_hContext = 0;
			goto Exit;
		}
	}

	// Set up AES Algorithm
	
	algorithm.algorithm = (nuint8 *)oid_des3;
	algorithm.parameterLen = 0;
	algorithm.parameter = NULL;

	// Set up key attributes

	keyAttr[0].type = NICI_A_KEY_TYPE;
	keyAttr[0].u.f.hasValue = 1;
	keyAttr[0].u.f.value = NICI_K_DES3X;
	keyAttr[0].u.f.valueInfo = 0;

	keyAttr[1].type = NICI_A_KEY_FORMAT;
	keyAttr[1].u.v.valuePtr = oid_des3;
	keyAttr[1].u.v.valueLen = (nuint32)sizeof( oid_des3);
	keyAttr[1].u.v.valueInfo = 0;

	keyAttr[2].type = NICI_A_KEY_USAGE;
	keyAttr[2].u.f.hasValue = 1;
	keyAttr[2].u.f.value = NICI_F_WRAP | NICI_F_UNWRAP | NICI_F_KM_ENCRYPT | 
								  NICI_F_KM_DECRYPT | NICI_F_EXTRACT | 
								  NICI_F_DATA_ENCRYPT | NICI_F_DATA_DECRYPT;
	keyAttr[2].u.f.valueInfo = 0;

	keyAttr[3].type = NICI_A_KEY_SIZE;
	keyAttr[3].u.f.hasValue = 1;
	keyAttr[3].u.f.value = uiEncKeySize;
	keyAttr[3].u.f.valueInfo = 0;

	keyAttr[4].type = NICI_A_GLOBAL;
	keyAttr[4].u.f.hasValue = 1;
	keyAttr[4].u.f.value = N_TRUE;
	keyAttr[4].u.f.valueInfo = 0;

	keyAttr[5].type = NICI_A_CLASS;
	keyAttr[5].u.f.hasValue = 1;
	keyAttr[5].u.f.value = NICI_O_SECRET_KEY;
	keyAttr[5].u.f.valueInfo = 0;

	// Generate an AES wrapping key

	if( RC_BAD( rc = CCS_GenerateKey( m_hContext, &algorithm, keyAttr, 6,
		&keySizeChanged, &m_keyHandle, NICI_H_INVALID)))
	{
		rc = RC_SET( NE_XFLM_NICI_GENKEY_FAILED);
		m_hContext = 0;
		goto Exit;
	}

	// Generate some IV to use with this key.

	if( RC_BAD( rc = CCS_GetRandom( m_hContext, (nuint8 *)m_ucIV, IV_SZ)))
	{
		rc = RC_SET( NE_XFLM_NICI_BAD_RANDOM);
		m_hContext = 0;
		goto Exit;
	}

	// If we generated a wrapping key, then this object's key handle is 
	// actually a wrapping key.  This means that we will use it to wrap the
	// other keys in the system.
	
	m_bKeyIsWrappingKey = TRUE;
	m_uiEncKeySize = uiEncKeySize;

Exit:

	f_mutexUnlock( m_hMutex);
	return(rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_NICICCS::encryptToStore(
	FLMBYTE *			pucIn,
	FLMUINT				uiInLen,
	FLMBYTE *			pucOut,
	FLMUINT *			puiOutLen,
	FLMBYTE *			pucIV)
{
	RCODE					rc = NE_XFLM_OK;

	switch( m_uiAlgType)
	{
		case FLM_NICI_AES:
		{
			rc = encryptToStoreAES( pucIn, uiInLen, pucOut, puiOutLen, pucIV);
			break;
		}
		
		case FLM_NICI_DES3:
		{
			rc = encryptToStoreDES3( pucIn, uiInLen, pucOut, puiOutLen, pucIV);
			break;
		}
		
		default:
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_NICI_INVALID_ALGORITHM);
			goto Exit;
		}
	}

Exit:

	return rc;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_NICICCS::decryptFromStore(
	FLMBYTE *		pucIn,
	FLMUINT			uiInLen,
	FLMBYTE *		pucOut,
	FLMUINT *		puiOutLen,
	FLMBYTE *		pucIV)
{
	RCODE				rc = NE_XFLM_OK;

	switch( m_uiAlgType)
	{
		case FLM_NICI_AES:
		{
			rc = decryptFromStoreAES( pucIn, uiInLen, pucOut, puiOutLen, pucIV);
			break;
		}
		
		case FLM_NICI_DES3:
		{
			rc = decryptFromStoreDES3( pucIn, uiInLen, pucOut, puiOutLen, pucIV);
			break;
		}
		
		default:
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_NICI_INVALID_ALGORITHM);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_NICICCS::encryptToStoreAES(
	FLMBYTE *				pucIn,
	FLMUINT					uiInLen,
	FLMBYTE *				pucOut,
	FLMUINT *				puiOutLen,
	FLMBYTE *				pucIV)
{
	RCODE						rc = NE_XFLM_OK;
	NICI_ALGORITHM			algorithm;
	NICI_PARAMETER_INFO	parm[1];
	FLMBYTE					oid_aes128[] = {IDV_AES128CBC};
	FLMBYTE					oid_aes192[] = {IDV_AES192CBC};
	FLMBYTE					oid_aes256[] = {IDV_AES256CBC};

	f_mutexLock( m_hMutex);

	// Create NICI Context
	
	if( !m_hContext)
	{

		if( RC_BAD( rc = CCS_CreateContext( 0, &m_hContext)))
		{
			rc = RC_SET( NE_XFLM_NICI_CONTEXT);
			m_hContext = 0;
			goto Exit;
		}
	}

	switch( m_uiEncKeySize)
	{
		case XFLM_NICI_AES128:
		{
			algorithm.algorithm = (nuint8 *)oid_aes128;
			break;
		}
		
		case XFLM_NICI_AES192:
		{
			algorithm.algorithm = (nuint8 *)oid_aes192;
			break;
		}
		
		case XFLM_NICI_AES256:
		{
			algorithm.algorithm = (nuint8 *)oid_aes256;
			break;
		}
		
		default:
		{
			rc = RC_SET( NE_XFLM_INVALID_ENC_KEY_SIZE);
			goto Exit;
		}
	}
	
	algorithm.parameterLen = sizeof(algorithm.parameter->parms[0])+
								sizeof(algorithm.parameter->count);
	algorithm.parameter = parm;
	algorithm.parameter->count = 1;
	algorithm.parameter->parms[0].parmType = NICI_P_IV;
	
	if( pucIV)
	{
		algorithm.parameter->parms[0].u.b.ptr = (nuint8 *)pucIV;
	}
	else
	{
		algorithm.parameter->parms[0].u.b.ptr = (nuint8 *)m_ucIV;
	}

	algorithm.parameter->parms[0].u.b.len = IV_SZ;

	// Init encryption

	if( RC_BAD( rc = CCS_DataEncryptInit( m_hContext, &algorithm, m_keyHandle)))
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NICI_ENC_INIT_FAILED);
		m_hContext = 0;
	 	goto Exit;
	}

	if( RC_BAD( rc = CCS_Encrypt( m_hContext, (nuint8 *)pucIn, uiInLen,
		(nuint8 *)pucOut, puiOutLen)))
	{
		rc = RC_SET( NE_XFLM_NICI_ENCRYPT_FAILED);
		m_hContext = 0;
		goto Exit;
	}

Exit:

	f_mutexUnlock( m_hMutex);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_NICICCS::decryptFromStoreAES(
	FLMBYTE *				pucIn,
	FLMUINT					uiInLen,
	FLMBYTE *				pucOut,
	FLMUINT *				puiOutLen,
	FLMBYTE *				pucIV)
{
	RCODE						rc = NE_XFLM_OK;
	NICI_ALGORITHM			algorithm;
	NICI_PARAMETER_INFO	parm[1];
	FLMBYTE					oid_aes128[] = {IDV_AES128CBC};
	FLMBYTE					oid_aes192[] = {IDV_AES192CBC};
	FLMBYTE					oid_aes256[] = {IDV_AES256CBC};

	f_mutexLock( m_hMutex);

	// Create NICI Context
	
	if( !m_hContext)
	{
		if( RC_BAD( rc = CCS_CreateContext(0, &m_hContext)))
		{
			rc = RC_SET( NE_XFLM_NICI_CONTEXT);
			m_hContext = 0;
			goto Exit;
		}
	}

	switch( m_uiEncKeySize)
	{
		case XFLM_NICI_AES128:
		{
			algorithm.algorithm = (nuint8 *)oid_aes128;
			break;
		}
		
		case XFLM_NICI_AES192:
		{
			algorithm.algorithm = (nuint8 *)oid_aes192;
			break;
		}
		
		case XFLM_NICI_AES256:
		{
			algorithm.algorithm = (nuint8 *)oid_aes256;
			break;
		}
		
		default:
		{
			rc = RC_SET( NE_XFLM_INVALID_ENC_KEY_SIZE);
			goto Exit;
		}
	}
	
	algorithm.parameterLen = sizeof(algorithm.parameter->parms[0])+
								sizeof(algorithm.parameter->count);
	algorithm.parameter = parm;
	algorithm.parameter->count = 1;
	algorithm.parameter->parms[0].parmType = NICI_P_IV;
	
	if( pucIV)
	{
		algorithm.parameter->parms[0].u.b.ptr = (nuint8 *)pucIV;
	}
	else
	{
		algorithm.parameter->parms[0].u.b.ptr = (nuint8 *)m_ucIV;
	}

	algorithm.parameter->parms[0].u.b.len = IV_SZ;

	// Init encryption

	if( RC_BAD( rc = CCS_DataDecryptInit( m_hContext, &algorithm, m_keyHandle)))
	{
		rc = RC_SET( NE_XFLM_NICI_DECRYPT_INIT_FAILED);
		m_hContext = 0;
	 	goto Exit;
	}

	if( RC_BAD( rc = CCS_Decrypt( m_hContext, (nuint8 *)pucIn, uiInLen,
		(nuint8 *)pucOut, puiOutLen)))
	{
		rc = RC_SET( NE_XFLM_NICI_DECRYPT_FAILED);
		m_hContext = 0;
		goto Exit;
	}

Exit:

	f_mutexUnlock( m_hMutex);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_NICICCS::encryptToStoreDES3(
	FLMBYTE *				pucIn,
	FLMUINT					uiInLen,
	FLMBYTE *				pucOut,
	FLMUINT *				puiOutLen,
	FLMBYTE *				pucIV)
{
	RCODE						rc = NE_XFLM_OK;
	NICI_ALGORITHM			algorithm;
	NICI_PARAMETER_INFO	parm[1];
	FLMBYTE					oid_des3[] = {IDV_DES_EDE3_CBC_IV8};

	f_mutexLock( m_hMutex);

	// Create NICI Context
	
	if( !m_hContext)
	{
		if( RC_BAD( rc = CCS_CreateContext( 0, &m_hContext)))
		{
			rc = RC_SET( NE_XFLM_NICI_CONTEXT);
			m_hContext = 0;
			goto Exit;
		}
	}

	algorithm.algorithm = (nuint8 *)oid_des3;
	algorithm.parameterLen = sizeof(algorithm.parameter->parms[0])+
								sizeof(algorithm.parameter->count);
	algorithm.parameter = parm;
	algorithm.parameter->count = 1;
	algorithm.parameter->parms[0].parmType = NICI_P_IV;
	
	if( pucIV)
	{
		algorithm.parameter->parms[0].u.b.ptr = (nuint8 *)pucIV;
	}
	else
	{
		algorithm.parameter->parms[0].u.b.ptr = (nuint8 *)m_ucIV;
	}

	algorithm.parameter->parms[0].u.b.len = IV_SZ8;

	// Init encryption

	if( RC_BAD( rc = CCS_DataEncryptInit( m_hContext, &algorithm, m_keyHandle)))
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NICI_ENC_INIT_FAILED);
		m_hContext = 0;
	 	goto Exit;
	}

	if( RC_BAD( rc = CCS_Encrypt( m_hContext, (nuint8 *)pucIn, uiInLen,
		(nuint8 *)pucOut, puiOutLen)))
	{
		rc = RC_SET( NE_XFLM_NICI_ENCRYPT_FAILED);
		m_hContext = 0;
		goto Exit;
	}

Exit:

	f_mutexUnlock( m_hMutex);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_NICICCS::decryptFromStoreDES3(
	FLMBYTE *				pucIn,
	FLMUINT					uiInLen,
	FLMBYTE *				pucOut,
	FLMUINT *				puiOutLen,
	FLMBYTE *				pucIV)
{
	RCODE						rc = NE_XFLM_OK;
	NICI_ALGORITHM			algorithm;
	NICI_PARAMETER_INFO	parm[1];
	FLMBYTE					oid_des3[] = {IDV_DES_EDE3_CBC_IV8};

	f_mutexLock( m_hMutex);

	// Create NICI Context
	
	if( !m_hContext)
	{
		if( RC_BAD( rc = CCS_CreateContext( 0, &m_hContext)))
		{
			rc = RC_SET( NE_XFLM_NICI_CONTEXT);
			m_hContext = 0;
			goto Exit;
		}
	}

	// Set up alogrithm now to do triple des decryption
	
	algorithm.algorithm = (nuint8 *)oid_des3;
	algorithm.parameterLen = sizeof(algorithm.parameter->parms[0])+
								sizeof(algorithm.parameter->count);
	algorithm.parameter = parm;
	algorithm.parameter->count = 1;
	algorithm.parameter->parms[0].parmType = NICI_P_IV;
	
	if( pucIV)
	{
		algorithm.parameter->parms[0].u.b.ptr = (nuint8 *)pucIV;
	}
	else
	{
		algorithm.parameter->parms[0].u.b.ptr = (nuint8 *)m_ucIV;
	}

	algorithm.parameter->parms[0].u.b.len = IV_SZ8;

	// Init encryption

	if( RC_BAD( rc = CCS_DataDecryptInit( m_hContext, &algorithm, m_keyHandle)))
	{
		rc = RC_SET( NE_XFLM_NICI_DECRYPT_INIT_FAILED);
		m_hContext = 0;
	 	goto Exit;
	}

	if( RC_BAD( rc = CCS_Decrypt( m_hContext, (nuint8 *)pucIn, uiInLen,
		(nuint8 *)pucOut, puiOutLen)))
	{
		rc = RC_SET( NE_XFLM_NICI_DECRYPT_FAILED);
		m_hContext = 0;
		goto Exit;
	}

Exit:

	f_mutexUnlock( m_hMutex);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_NICICCS::init(
	FLMBOOL			bKeyIsWrappingKey,
	FLMUINT			uiAlgType)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bLocked = FALSE;

	if (m_bInitCalled)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

	m_bKeyIsWrappingKey = bKeyIsWrappingKey;

	if( uiAlgType != FLM_NICI_AES && uiAlgType != FLM_NICI_DES3)
	{
		flmAssert( 0);
		rc = RC_SET( NE_XFLM_INVALID_ENC_ALGORITHM);
		goto Exit;
	}

	m_uiAlgType = uiAlgType;

	// Create a mutex to control access to the nici operations.
	
	if( RC_BAD( rc = f_mutexCreate( &m_hMutex)))
	{
		goto Exit;
	}
	
	f_mutexLock( m_hMutex);
	bLocked = TRUE;

	// Create NICI Context
	
	if( !m_hContext)
	{
		if( RC_BAD( rc = CCS_CreateContext( 0, &m_hContext)))
		{
			rc = RC_SET( NE_XFLM_NICI_CONTEXT);
			m_hContext = 0;
			goto Exit;
		}
	}
	else
	{
		flmAssert( 0);
	}

	// Generate the Random IV
	
	if( RC_BAD( rc = CCS_GetRandom( m_hContext, (nuint8 *)&m_ucRndIV, IV_SZ)))
	{
		rc = RC_SET( NE_XFLM_NICI_BAD_RANDOM);
		m_hContext = 0;
		goto Exit;
	}

	// Generate an adjustment factor for the IV
	
	if( RC_BAD( rc = CCS_GetRandom( m_hContext, (nuint8 *)&m_uiIVFactor,
		sizeof( FLMUINT))))
	{
		rc = RC_SET( NE_XFLM_NICI_BAD_RANDOM);
		m_hContext = 0;
		goto Exit;
	}

	m_bInitCalled = TRUE;

Exit:

	if (bLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_NICICCS::getWrappingKey(
	NICI_OBJECT_HANDLE *		pWrappingKeyHandle)
{
	RCODE						rc = NE_XFLM_OK;
	NICI_ATTRIBUTE			find[2];
	FLMUINT					uiCount;

	f_mutexLock( m_hMutex);

	// Create NICI context
	
	if( !m_hContext)
	{
		if( RC_BAD( rc = CCS_CreateContext( 0, &m_hContext)))
		{
			rc = RC_SET( NE_XFLM_NICI_CONTEXT);
			m_hContext = 0;
			goto Exit;
		}
	}

	find[0].type = NICI_A_GLOBAL;
	find[0].u.f.hasValue = 1;
	find[0].u.f.value = 1;
	find[0].u.f.valueInfo = 0;

	find[1].type = NICI_A_FEATURE;
	find[1].u.f.hasValue = 1;
	find[1].u.f.value = NICI_AV_STORAGE;
	find[1].u.f.valueInfo = 0;

	if( RC_BAD( rc = CCS_FindObjectsInit( m_hContext, find, 2)))
	{
		rc = RC_SET( NE_XFLM_NICI_FIND_INIT);
		m_hContext = 0;
		goto Exit;
	}

	uiCount = 1;

	if( RC_BAD( rc = CCS_FindObjects( m_hContext, pWrappingKeyHandle, &uiCount)))
	{
		rc = RC_SET( NE_XFLM_NICI_FIND_OBJECT);
		m_hContext = 0;
		goto Exit;
	}

	if( uiCount < 1)
	{
		rc = RC_SET( NE_XFLM_NICI_WRAPKEY_NOT_FOUND);
		goto Exit;
	}

Exit:

	f_mutexUnlock( m_hMutex);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_NICICCS::getKeyToStore(
	FLMBYTE **				ppucKeyInfo,
	FLMUINT32 *				pui32BufLen,
	FLMBYTE *				pszEncKeyPasswd,
	IF_CCS *					pWrappingCcs)
{
	RCODE						rc = NE_XFLM_OK;
	FLMBYTE *				pucTmp = NULL;
	FLMBYTE *				pucPtr = NULL;
	FLMUINT32				ui32PaddedLength;
	FLMBYTE *				pucWrappedKey = NULL;
	FLMUINT32				ui32WrappedKeyLen = 0;
	FLMBYTE *				pszFormattedEncKeyPasswd = NULL;
	NICI_OBJECT_HANDLE	wrappingKeyHandle = 0;

	*ppucKeyInfo = NULL;
	*pui32BufLen = 0;

	if( pWrappingCcs)
	{
		flmAssert(m_bKeyIsWrappingKey == FALSE);
		wrappingKeyHandle = pWrappingCcs->m_keyHandle;
	}
	else if( !pszEncKeyPasswd)
	{
		flmAssert( m_bKeyIsWrappingKey);
	}

	// Either extract the key or wrap the key.
	
	if( pszEncKeyPasswd && pszEncKeyPasswd[ 0])
	{
		// The password that is passed in to CCS_pbeEncrypt is NOT actually
		// unicode.  It must be treated as a sequence of bytes that that is
		// terminated with 2 nulls and has an even length.  If we treat it
		// as unicode, then we'll have endian issues if we move the database
		// to machines with different byte ordering.
		
		if( RC_BAD( rc = f_calloc( f_strlen(pszEncKeyPasswd) +
				(f_strlen(pszEncKeyPasswd) % 2) + 2, &pszFormattedEncKeyPasswd)))
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

	// The shrouded or wrapped key will be stored in m_pKey
	
	ui32PaddedLength = (ui32WrappedKeyLen + sizeof( FLMBOOL) +
								sizeof (FLMUINT32) + IV_SZ );

	// Make sure our buffer size is padded to a 16 byte boundary.
	
	if( (ui32PaddedLength % 16) != 0)
	{
		ui32PaddedLength += (16 - (ui32PaddedLength % 16));
	}

	// Add one extra byte for a NULL terminator
	
	if( RC_BAD(rc = f_alloc( ui32PaddedLength + 1, &pucTmp)))
	{
		goto Exit;
	}

	if( !m_hContext)
	{
		if( CCS_CreateContext( 0, &m_hContext))
		{
			rc = RC_SET( NE_XFLM_NICI_CONTEXT);
			m_hContext = 0;
			goto Exit;
		}
	}

	pucPtr = pucTmp;

	// Save a flag indicating whether the key is wrapped or encoded in
	// a password.
	
	UD2FBA( (pszEncKeyPasswd && pszEncKeyPasswd[0]) 
							? (FLMUINT)TRUE 
							: (FLMUINT)FALSE, pucPtr);
							
	pucPtr += sizeof(FLMBOOL);

	// Copy the key length.
	
	UD2FBA(ui32WrappedKeyLen, pucPtr);
	pucPtr += sizeof(FLMUINT32);

	// Copy the IV too.
	
	f_memcpy( pucPtr, m_ucIV, IV_SZ);
	pucPtr += IV_SZ;

	// Copy the wrapped key value
	
	f_memcpy( pucPtr, pucWrappedKey, ui32WrappedKeyLen);
	pucPtr += ui32WrappedKeyLen;

	// Fill the remainder of the buffer with random data.
	
	if( CCS_GetRandom( m_hContext, (nuint8 *)pucPtr,
			((FLMUINT)pucTmp + ui32PaddedLength) - (FLMUINT)pucPtr))
	{
		rc = RC_SET( NE_XFLM_NICI_BAD_RANDOM);
		m_hContext = 0;
		goto Exit;
	}

	pucTmp[ ui32PaddedLength] = '\0';
	*ppucKeyInfo = pucTmp;
	*pui32BufLen = ui32PaddedLength;
	pucTmp = NULL;

Exit:

	if( pucTmp)
	{
		f_free(&pucTmp);
	}

	if( pucWrappedKey)
	{
		f_free( &pucWrappedKey);
	}

	if( pszFormattedEncKeyPasswd)
	{
		f_free( &pszFormattedEncKeyPasswd);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_NICICCS::setKeyFromStore(
	FLMBYTE *				pucKeyInfo,
	FLMBYTE *				pszEncKeyPasswd,
	IF_CCS *					pWrappingCcs)
{
	RCODE						rc = NE_XFLM_OK;
	FLMBYTE *				pucTmp = pucKeyInfo;
	FLMBYTE *				pucBuffer = NULL;
	FLMBOOL					bShrouded = FALSE;
	FLMUINT32				ui32Length;
	FLMBYTE *				pucKeyBuf = NULL;
	FLMBYTE *				pszFormattedEncKeyPasswd = NULL;
	NICI_OBJECT_HANDLE	wrappingKeyHandle = 0;

	if( pWrappingCcs)
	{
		flmAssert(m_bKeyIsWrappingKey == FALSE);
		wrappingKeyHandle = pWrappingCcs->m_keyHandle;
	}

	// Extract the fields from the buffer
	
	bShrouded = FB2UD( pucTmp);
	pucTmp += sizeof(FLMUINT);

	// Actual length - note that the passed buffer is padded to 16 byte boundary.
	
	ui32Length = FB2UD( pucTmp);
	pucTmp += sizeof(FLMUINT32);

	// Get the IV
	
	f_memcpy( m_ucIV, pucTmp, IV_SZ);
	pucTmp += IV_SZ;

	// Need another temporary buffer to hold the encrypted / shrouded key.
	
	if( RC_BAD( rc = f_alloc( ui32Length, &pucBuffer)))
	{
		goto Exit;
	}

	f_memcpy( pucBuffer, pucTmp, ui32Length);

	if( bShrouded)
	{
		if( pszEncKeyPasswd == NULL || pszEncKeyPasswd[0] == '\0')
		{
			rc = RC_SET( NE_XFLM_EXPECTING_PASSWORD);
			goto Exit;
		}
		
		// The password that is passed in to CCS_pbeDecrypt is NOT actually
		// unicode.  It must be treated as a sequence of bytes that that is
		// terminated with 2 nulls and has an even length.  If we treat it
		// as unicode, then we'll have endian issues if we move the database
		// to machines with different byte ordering.
		
		if( RC_BAD( rc = f_calloc( f_strlen(pszEncKeyPasswd) +
			(f_strlen(pszEncKeyPasswd) % 2) + 2, &pszFormattedEncKeyPasswd)))
		{
			goto Exit;
		}
		
		f_strcpy( pszFormattedEncKeyPasswd, pszEncKeyPasswd);

		// Unshroud the key using the password.
		// Key handle is always kept in m_keyHandle.
		
		if( RC_BAD( rc = injectKey( pucBuffer, ui32Length,
			(FLMUNICODE *)pszFormattedEncKeyPasswd)))
		{
			goto Exit;
		}
	}
	else
	{
		if( pszEncKeyPasswd)
		{
			if( pszEncKeyPasswd[0] != '\0')
			{
				rc = RC_SET( NE_XFLM_NOT_EXPECTING_PASSWORD);
				goto Exit;
			}
		}

		// Unwrap the key.  The Key handle is always store in m_keyHandle.
		
		if( RC_BAD( rc = unwrapKey( pucBuffer, ui32Length, wrappingKeyHandle)))
		{
			goto Exit;
		}
	}

	m_bKeyVerified = TRUE;

Exit:

	if( pucBuffer)
	{
		f_free( &pucBuffer);
	}

	if( pucKeyBuf)
	{
		f_free( &pucKeyBuf);
	}
	
	if( pszFormattedEncKeyPasswd)
	{
		f_free( &pszFormattedEncKeyPasswd);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_NICICCS::extractKey(
	FLMBYTE **				ppucExtractedKey,
	FLMUINT32 *				pui32Length,
	FLMUNICODE *			puzEncKeyPasswd)
{
	RCODE						rc = NE_XFLM_OK;
	NICI_ALGORITHM			algorithm;
	NICI_ATTRIBUTE			keyAttr[2];
	NICI_ATTRIBUTE			attr[2];
	FLMBYTE					oid_sha1[] = {IDV_SHA1};
	FLMBYTE					oid_pbe[] = {IDV_pbeWithSHA1And3Key3xDES_CBC};
	FLMBYTE					ucDigest[ 20];
	FLMUINT					uiDigestLen = sizeof(ucDigest);
	FLMUINT					uiBufferSize;
	FLMBYTE *				pucKey = NULL;
	FLMBYTE *				pucFormat = NULL;
	EXTRACTED_KEY *		pExtractedKey = NULL;
	FLMUINT					uiEncLen;
	FLMBYTE *				pTemp = NULL;
	NICI_PARAMETER_INFO *	pParmInfo;
	FLMBYTE *				pucSalt;
	FLMUINT					uiAllocSize;
	FLMUINT					uiIndx;
	FLMBYTE *				pucTempPtr;

	f_mutexLock( m_hMutex);

	// Create NICI Context
	
	if( !m_hContext)
	{
		if( RC_BAD( rc = CCS_CreateContext( 0, &m_hContext)))
		{
			rc = RC_SET( NE_XFLM_NICI_CONTEXT);
			m_hContext = 0;
			goto Exit;
		}
	}

	f_memset( &attr[0], 0, sizeof(NICI_ATTRIBUTE) * 2);
	attr[0].type = NICI_A_KEY_TYPE;
	attr[1].type = NICI_A_KEY_FORMAT;

	if( RC_BAD( rc = CCS_GetAttributeValue( m_hContext, 
		m_keyHandle, &attr[0], 2)))
	{
		rc = RC_SET( NE_XFLM_NICI_ATTRIBUTE_VALUE);
		m_hContext = 0;
		goto Exit;
	}

	if( !attr[0].u.f.hasValue)
	{
		rc = RC_SET( NE_XFLM_NICI_BAD_ATTRIBUTE);
		goto Exit;
	}

	f_memset( &keyAttr[0], 0, sizeof(NICI_ATTRIBUTE) * 2);

	switch( attr[0].u.f.value)
	{
		case NICI_K_AES:
		{
			uiIndx = 0;
			keyAttr[uiIndx].type = NICI_A_KEY_VALUE;
			
			switch( m_uiEncKeySize)
			{
				case XFLM_NICI_AES128:
				{
					keyAttr[uiIndx].u.v.valueLen = 16;
					break;
				}
				
				case XFLM_NICI_AES192:
				{
					keyAttr[uiIndx].u.v.valueLen = 24;
					break;
				}
				
				case XFLM_NICI_AES256:
				{
					keyAttr[uiIndx].u.v.valueLen = 32;
					break;
				}
			}

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
		{
			uiIndx = 0;
			keyAttr[uiIndx].type = NICI_A_KEY_VALUE;
			keyAttr[uiIndx].u.v.valueLen = 8;

			uiIndx++;
			keyAttr[uiIndx].type = NICI_A_KEY_FORMAT;
			keyAttr[uiIndx].u.v.valueLen = attr[1].u.v.valueLen;
			break;
		}

		default:
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_NICI_INVALID_ALGORITHM);
			goto Exit;
		}
	}

	// Make one allocation that we can then use to hold several different things.

	uiBufferSize = sizeof( EXTRACTED_KEY) + attr[1].u.v.valueLen +
						keyAttr[0].u.v.valueLen + sizeof (ucDigest);
	
	uiAllocSize = uiBufferSize + SALT_SZ + 
						(sizeof( NICI_PARAMETER_DATA) * 2) + sizeof( FLMUINT32);

	// Make sure the allocation size is on a 8 byte boundary
	
	if( (uiAllocSize % 8) != 0)
	{
		uiAllocSize += (8 - (uiAllocSize % 8));
	}

	if( RC_BAD( rc = f_calloc( uiAllocSize, &pExtractedKey)))
	{
		goto Exit;
	}

	keyAttr[1].u.v.valuePtr = &pExtractedKey[1];
	pucFormat = (FLMBYTE *)keyAttr[1].u.v.valuePtr;
	keyAttr[0].u.v.valuePtr = pucFormat + attr[1].u.v.valueLen;
	pucKey = (FLMBYTE *)keyAttr[0].u.v.valuePtr;
	pucSalt = (FLMBYTE *)pExtractedKey + uiBufferSize;
	pParmInfo = (NICI_PARAMETER_INFO *)(pucSalt + SALT_SZ);

	// Make sure that pParmInfo is 8 byte alligned.
	
	if( (FLMUINT)pParmInfo % 8)
	{
		FLMBYTE *	pucTemp;
		
		pucTemp = (FLMBYTE *)pParmInfo + (8 - ((FLMUINT)pParmInfo % 8));
		pParmInfo = (NICI_PARAMETER_INFO *)pucTemp;
	}

	// Extracted the key value now

	if( RC_BAD( rc = CCS_ExtractKey( m_hContext, m_keyHandle, &keyAttr[0], 2)))
	{
		rc = RC_SET( NE_XFLM_EXTRACT_KEY_FAILED);
		m_hContext = 0;
		goto Exit;
	}

	// Calculate a SHA1 checksum.

	algorithm.algorithm = (nuint8 *)oid_sha1;
	algorithm.parameter = NULL;
	algorithm.parameterLen = 0;
	
	if( RC_BAD( rc = CCS_DigestInit( m_hContext, &algorithm)))
	{
		rc = RC_SET( NE_XFLM_DIGEST_INIT_FAILED);
		m_hContext = 0;
		goto Exit;
	}

	if( RC_BAD( rc = CCS_Digest( m_hContext, (nuint8 *)pucFormat,
		keyAttr[0].u.v.valueLen + attr[1].u.v.valueLen, (nuint8 *)ucDigest,
		&uiDigestLen)))
	{
		rc = RC_SET( NE_XFLM_DIGEST_FAILED);
		m_hContext = 0;
		goto Exit;
	}
	
	flmAssert( uiDigestLen == sizeof( ucDigest));

	pucTempPtr = (FLMBYTE *)pExtractedKey;

	UD2FBA( attr[0].u.f.value, pucTempPtr);
	pucTempPtr += 4;

	UD2FBA( attr[1].u.v.valueLen, pucTempPtr);
	pucTempPtr += 4;

	UD2FBA( keyAttr[0].u.v.valueLen, pucTempPtr);
	pucTempPtr += 4;
	
	UD2FBA( m_uiEncKeySize, pucTempPtr);
	
	// Point to the Digest...

	pTemp = (FLMBYTE *)&pExtractedKey[1] + attr[1].u.v.valueLen +
				keyAttr[0].u.v.valueLen;
	f_memcpy( pTemp, ucDigest, uiDigestLen);

	// Generate some salt

	if( RC_BAD( rc = CCS_GetRandom( m_hContext, (nuint8 *)pucSalt, SALT_SZ)))
	{
		rc = RC_SET( NE_XFLM_NICI_BAD_RANDOM);
		m_hContext = 0;
		pTemp = NULL;
		goto Exit;
	}

	// This buffer needs to be a separate allocation because it is returned to
	// the caller.  We will be returning the value of the SALT with the
	// encrypted key.  The call to CCS_pbeEncrypt may return an extra 8 bytes.

	if( RC_BAD( rc = f_alloc( uiBufferSize + SALT_SZ + 8, &pTemp)))
	{
		goto Exit;
	}
	
	// Now to encrypt the buffer.

	algorithm.algorithm = (nuint8 *)oid_pbe;
	pParmInfo->count = 2;
	
	pParmInfo->parms[0].parmType = NICI_P_SALT;
	pParmInfo->parms[0].u.b.len = SALT_SZ;
	pParmInfo->parms[0].u.b.ptr = (nuint8 *)pucSalt;
	
	pParmInfo->parms[1].parmType = NICI_P_COUNT;
	pParmInfo->parms[1].u.value = SALT_COUNT;

	algorithm.parameter = pParmInfo;
	algorithm.parameterLen = sizeof( NICI_PARAMETER_DATA) * 2 + sizeof( FLMUINT32);

	uiEncLen = uiBufferSize + 8;

	if( RC_BAD( rc = CCS_pbeEncrypt( m_hContext, &algorithm, puzEncKeyPasswd,
			(nuint8 *)pExtractedKey, uiBufferSize, (nuint8 *)pTemp, &uiEncLen)))
	{
		rc = RC_SET( NE_XFLM_PBE_ENCRYPT_FAILED);
		m_hContext = 0;
		goto Exit;
	}

	*ppucExtractedKey = pTemp;

	// Now add the salt to the end of the buffer.

	pTemp += uiEncLen;
	f_memcpy( pTemp, pucSalt, SALT_SZ);

	pTemp = NULL;
	*pui32Length = uiEncLen + SALT_SZ;
	
Exit:

	if( pTemp)
	{
		f_free( &pTemp);
	}

	if( pucKey)
	{
		f_free( &pExtractedKey);
	}

	f_mutexUnlock( m_hMutex);
	return(rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_NICICCS::injectKey(
	FLMBYTE *					pszExtractedKey,
	FLMUINT32					ui32Length,
	FLMUNICODE *				puzEncKeyPasswd)
{
	RCODE							rc = NE_XFLM_OK;
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

	f_mutexLock( m_hMutex);

	// Create NICI Context
	
	if( !m_hContext)
	{
		if( RC_BAD( rc = CCS_CreateContext(0, &m_hContext)))
		{
			rc = RC_SET( NE_XFLM_NICI_CONTEXT);
			m_hContext = 0;
			goto Exit;
		}
	}

	// Extract the SALT from the key buffer.
	
	pucSalt = pszExtractedKey + (ui32Length - SALT_SZ);
	ui32Length -= SALT_SZ;

	// Make one allocation and point into it for the different buffers we need.

	uiAllocSize = ui32Length + sizeof( NICI_PARAMETER_DATA) * 2 + sizeof( FLMUINT32);
	
	if( RC_BAD( rc = f_calloc( uiAllocSize, &pucBuffer)))
	{
		goto Exit;
	}
	
	pParmInfo = (NICI_PARAMETER_INFO *)(pucBuffer + ui32Length);

	// Now to decrypt the buffer.

	algorithm.algorithm = (nuint8 *)oid_pbe;
	pParmInfo->count = 2;
	
	pParmInfo->parms[0].parmType = NICI_P_SALT;
	pParmInfo->parms[0].u.b.len = SALT_SZ;
	pParmInfo->parms[0].u.b.ptr = (nuint8 *)pucSalt;
	
	pParmInfo->parms[1].parmType = NICI_P_COUNT;
	pParmInfo->parms[1].u.value = SALT_COUNT;

	algorithm.parameter = pParmInfo;
	algorithm.parameterLen = sizeof(NICI_PARAMETER_DATA) * 2 + sizeof(FLMUINT32);

	uiEncLen = ui32Length;

	if( RC_BAD( rc = CCS_pbeDecrypt( m_hContext, &algorithm, puzEncKeyPasswd,
		(nuint8 *)pszExtractedKey, ui32Length, (nuint8 *)pucBuffer, &uiEncLen)))
	{
		rc = RC_SET( NE_XFLM_PBE_DECRYPT_FAILED);
		m_hContext = 0;
		goto Exit;
	}

	// For cross platform compatibility, we need to first extract the KeyType,
	// FormatLen and KeyLen values then we will set them back again.  They are
	// stored in a specific byte order, which may not match the native order for
	// referencing integers on the local platform.

	pExtractedKey = (EXTRACTED_KEY *)pucBuffer;
	pucTempPtr = pucBuffer;

	pExtractedKey->uiKeyType = FB2UD( pucTempPtr);
	pucTempPtr += 4;

	pExtractedKey->uiFormatLen = FB2UD( pucTempPtr);
	pucTempPtr += 4;

	pExtractedKey->uiKeyLen = FB2UD( pucTempPtr);
	pucTempPtr += 4;

	m_uiEncKeySize = FB2UD( pucTempPtr);

	// Calculate a SHA1 checksum.

	algorithm.algorithm = (nuint8 *)oid_sha1;
	algorithm.parameter = NULL;
	algorithm.parameterLen = 0;
	
	if( RC_BAD( rc = CCS_DigestInit( m_hContext, &algorithm)))
	{
		rc = RC_SET( NE_XFLM_DIGEST_INIT_FAILED);
		m_hContext = 0;
		goto Exit;
	}
	
	pTemp = (FLMBYTE *)&pExtractedKey[ 1];

	if( RC_BAD( rc = CCS_Digest( m_hContext, (nuint8 *)pTemp,
		pExtractedKey->uiFormatLen + pExtractedKey->uiKeyLen, (nuint8 *)ucDigest,
		&uiDigestLen)))
	{
		rc = RC_SET( NE_XFLM_DIGEST_FAILED);
		m_hContext = 0;
		goto Exit;
	}
	
	flmAssert( uiDigestLen == sizeof( ucDigest));

	// Now compare the two digests.  They must be equal!
	
	pTemp += pExtractedKey->uiKeyLen + pExtractedKey->uiFormatLen;
	
	if( f_memcmp( pTemp, ucDigest, uiDigestLen))
	{
		rc = RC_SET( NE_XFLM_INVALID_ENCKEY_CRC);
		goto Exit;
	}

	pucFormat = (FLMBYTE *)&pExtractedKey[ 1];
	pKey = pucFormat + pExtractedKey->uiFormatLen;
	
	uiIndx = 0;
	f_memset( &keyAttr[0], 0, sizeof(NICI_ATTRIBUTE) * 7);
			
	switch( pExtractedKey->uiKeyType)
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
			keyAttr[uiIndx].u.f.value = NICI_F_WRAP | NICI_F_UNWRAP | 
												 NICI_F_KM_ENCRYPT | NICI_F_KM_DECRYPT | 
												 NICI_F_EXTRACT | NICI_F_DATA_ENCRYPT |
												 NICI_F_DATA_DECRYPT;
			keyAttr[uiIndx].u.f.valueInfo = 0;
		
			uiIndx++;
			keyAttr[uiIndx].type = NICI_A_KEY_SIZE;
			keyAttr[uiIndx].u.f.hasValue = 1;
			keyAttr[uiIndx].u.f.value = m_uiEncKeySize;
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
			keyAttr[uiIndx].u.f.value = NICI_F_WRAP | NICI_F_UNWRAP | 
												 NICI_F_KM_ENCRYPT | NICI_F_KM_DECRYPT |
												 NICI_F_EXTRACT | NICI_F_DATA_ENCRYPT | 
												 NICI_F_DATA_DECRYPT;
			keyAttr[uiIndx].u.f.valueInfo = 0;
		
			uiIndx++;
			keyAttr[uiIndx].type = NICI_A_KEY_SIZE;
			keyAttr[uiIndx].u.f.hasValue = 1;
			keyAttr[uiIndx].u.f.value = m_uiEncKeySize;
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

		default:
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_NICI_INVALID_ALGORITHM);
			goto Exit;
		}
	}

	if( RC_BAD( rc = CCS_InjectKey( m_hContext, &keyAttr[0], 7, &m_keyHandle)))
	{
		rc = RC_SET( NE_XFLM_INJECT_KEY_FAILED);
		m_hContext = 0;
		goto Exit;
	}

Exit:

	if( pucBuffer)
	{
		f_free( &pucBuffer);
	}

	f_mutexUnlock( m_hMutex);
	return( rc);
}

/****************************************************************************
Desc: getIVLen returns the correct length of the IV for the type of
		algorithm.
****************************************************************************/
FLMUINT F_NICICCS::getIVLen()
{
	switch( m_uiAlgType)
	{
		case FLM_NICI_AES:
		{
			return( IV_SZ);
		}
		
		case FLM_NICI_DES3:
		{
			return( IV_SZ8);
		}
		
		default:
		{
			return( 0);
		}
	}
}
	
/****************************************************************************
Desc: generateIV will generate a random set of bytes to be used as IV.
****************************************************************************/
RCODE F_NICICCS::generateIV(
	FLMUINT				uiIVLen,
	FLMBYTE *			pucIV)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiLoop;
	NICI_ALGORITHM		algorithm;
	FLMBYTE				oid_sha1[] = {IDV_SHA1};
	FLMBOOL				bLocked = FALSE;
	FLMBYTE *			pucIVPtr = m_ucRndIV;
	FLMBYTE				ucIVBuffer[ IV_SZ * 2];
	FLMUINT				uiIVBufferLen = sizeof(ucIVBuffer);

	if( !uiIVLen)
	{
		goto Exit;
	}

	f_mutexLock( m_hMutex);
	bLocked = TRUE;

	// Create NICI Context

	if( !m_hContext)
	{
		if( RC_BAD( rc = CCS_CreateContext( 0, &m_hContext)))
		{
			rc = RC_SET( NE_XFLM_NICI_CONTEXT);
			m_hContext = 0;
			goto Exit;
		}
	}

	// See if it is time to reinitialize the Random IV.

	if( (m_uiIVFactor & 0x07FF) == 0)
	{
		// Generate the Random IV

		if( RC_BAD( rc = CCS_GetRandom( m_hContext, (nuint8 *)&m_ucRndIV, IV_SZ)))
		{
			rc = RC_SET( NE_XFLM_NICI_BAD_RANDOM);
			m_hContext = 0;
			goto Exit;
		}

		// Generate an adjustment factor for the IV

		if( RC_BAD( rc = CCS_GetRandom( m_hContext, (nuint8 *)&m_uiIVFactor,
			sizeof( FLMUINT))))
		{
			rc = RC_SET( NE_XFLM_NICI_BAD_RANDOM);
			m_hContext = 0;
			goto Exit;
		}
	}

	// Increment each byte of the IV by the IV Factor

	for( uiLoop = 0; uiLoop < IV_SZ; uiLoop++)
	{
		(*pucIVPtr) += (FLMBYTE)m_uiIVFactor;
		pucIVPtr++;
	}

	// Now run the resulting IV through a SHA1 digest.

	algorithm.algorithm = (nuint8 *)oid_sha1;
	algorithm.parameter = NULL;
	algorithm.parameterLen = 0;
	
	if( RC_BAD( rc = CCS_DigestInit( m_hContext, &algorithm)))
	{
		rc = RC_SET( NE_XFLM_DIGEST_INIT_FAILED);
		m_hContext = 0;
		goto Exit;
	}

	if( RC_BAD( rc = CCS_Digest( m_hContext, (nuint8 *)m_ucRndIV, uiIVLen,
						  (nuint8 *)ucIVBuffer, &uiIVBufferLen)))
	{
		rc = RC_SET( NE_XFLM_DIGEST_FAILED);
		m_hContext = 0;
		goto Exit;
	}

	// Return the new IV!

	f_memcpy( pucIV, ucIVBuffer, uiIVLen);
	m_uiIVFactor++;

Exit:

	if( bLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE flmAllocCCS(
	IF_CCS **		ppCCS)
{
	RCODE				rc = NE_XFLM_OK;
	F_NICICCS *		pCCS = NULL;
	
	f_assert( (*ppCCS) == NULL);
	
	if( (pCCS = f_new F_NICICCS) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	
	*ppCCS = pCCS;
	pCCS = NULL;
		
Exit:
	
	if( pCCS)
	{
		pCCS->Release();
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
#if !defined( FLM_UNIX)
int CCSX_SetNewIV(
	int,				// MODULEID
	FLMUINT32,		// hContext
	pnuint8,			// IV
	nuint32)			// IVLen
{
	return( NICI_E_FUNCTION_NOT_SUPPORTED);
}
#endif

#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifndef FLM_USE_NICI
void f_nici_dummy( void)
{
}
#endif
