//-------------------------------------------------------------------------
// Desc:	Class definition for encryption/decryption
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

#ifndef	F_NICI_H
#define	F_NICI_H

#include "flaimsys.h"

#ifdef FLM_USE_NICI
	#ifdef FLM_NLM
		#include "nwccs.h"
	#else
		#include "ccs.h"

		N_EXTERN_LIBRARY(int)
		CCS_InjectKey (
			NICI_CC_HANDLE          hContext,
			NICI_ATTRIBUTE_PTR      attributeTemplate,
			nuint32                 attributecount,
			NICI_OBJECT_HANDLE_PTR  key);
		
		N_EXTERN_LIBRARY(int)
		CCS_ExtractKey (
			NICI_CC_HANDLE       hContext,
			NICI_OBJECT_HANDLE   key,
			NICI_ATTRIBUTE_PTR   attrTemplate,
			nuint32              attributeCount);

	#endif
#else
	#define NICI_OBJECT_HANDLE void *
#endif

#include "fpackon.h"
// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

/****************************************************************************
Definitions
****************************************************************************/

#define IV_SZ						16
#define IV_SZ8						8
#define SALT_SZ					8
#define SALT_COUNT				895

// These values are used to identify the algorithm for encryption.  They 
// map to indexes in the DDEncOpts table defined in ddprep.c.

#define FLM_NICI_AES				0
#define FLM_NICI_DES3			1
#define FLM_NICI_DES				2
#define FLM_NICI_UNDEFINED		0xFF

/****************************************************************************
A quick kludge to get around some differences between Netware and
everybody else.
****************************************************************************/

#if defined( FLM_NLM) && defined(FLM_USE_NICI)
	typedef nuint32	NICI_ULONG;
	typedef nuint8		NICI_BYTE;
	typedef nbool8		NICI_BBOOL;
#endif

/****************************************************************************
Desc:
****************************************************************************/
class IF_CCS
{
public:

	IF_CCS()
	{
	}
	
	virtual ~IF_CCS()
	{
	}

	virtual RCODE generateEncryptionKey( void) = 0;

	virtual RCODE generateWrappingKey( void) = 0;

	virtual RCODE encryptToStore(
		FLMBYTE *			pucIn,
		FLMUINT				uiInLen,
		FLMBYTE *			pucOut,
		FLMUINT *			puiOutLen) = 0;

	virtual RCODE decryptFromStore(
		FLMBYTE *				pucIn,
		FLMUINT					uiInLen,
		FLMBYTE *				pucOut,
		FLMUINT *				puiOutLen) = 0;

};

/****************************************************************************
Desc:
****************************************************************************/
class F_CCS : public IF_CCS, public F_Object
{
public:

	// Constructor & destructor
	F_CCS()
	{
		m_bInitCalled = FALSE;
		m_bKeyVerified = FALSE;
		f_memset( m_pucIV, 0, IV_SZ);
		m_bKeyIsWrappingKey = FALSE;
		m_uiAlgType = FLM_NICI_UNDEFINED;
		m_keyHandle = 0;
		m_hMutex = F_MUTEX_NULL;
	}

	~F_CCS();

	RCODE init(
		FLMBOOL			bKeyIsWrappingKey,
		FLMUINT			uiAlgType);

	RCODE generateEncryptionKey( void );

	RCODE generateWrappingKey( void );

	RCODE encryptToStore(
		FLMBYTE *			pucIn,
		FLMUINT				uiInLen,
		FLMBYTE *			pucOut,
		FLMUINT *			puiOutLen);

	RCODE decryptFromStore(
		FLMBYTE *				pucIn,
		FLMUINT					uiInLen,
		FLMBYTE *				pucOut,
		FLMUINT *				puiOutLen);

	RCODE getKeyToStore(
		FLMBYTE **			ppucKeyInfo,
		FLMUINT32 *			pui32BufLen,
		const char *		pszEncKeyPasswd = NULL,
		F_CCS *				pWrappingCcs = NULL,
		FLMBOOL				bBase64Encode = TRUE);

	RCODE setKeyFromStore(
		FLMBYTE *			pucKeyInfo,
		FLMUINT32			ui32BufLen,
		const char *		pszEncKeyPasswd = NULL,
		F_CCS *				pWrappingCcs = NULL,
		FLMBOOL				bBase64Encoded = TRUE);

	RCODE getKeyToExtract(
		FLMBYTE **			ppucKeyInfo,
		FLMUINT32 *			pui32BufLen,
		FLMUNICODE *		puzEncKeyPasswd);

	FINLINE FLMBOOL keyVerified()
	{
		return m_bKeyVerified;
	}

	FINLINE FLMUINT getEncType( void)
	{
		return m_uiAlgType;
	}

	FINLINE FLMINT FLMAPI AddRef( void)
	{
		return( f_atomicInc( &m_refCnt));
	}

	FINLINE FLMINT FLMAPI Release( void)
	{
		FLMINT		iRefCnt = f_atomicDec( &m_refCnt);
	
		if( !iRefCnt)
		{
			delete this;
		}
	
		return( iRefCnt);
	}
	
private:

	RCODE getWrappingKey(
		NICI_OBJECT_HANDLE *		pWrappingKeyHandle );

	RCODE wrapKey(
		FLMBYTE **				ppucWrappedKey,
		FLMUINT32 *				pui32Length,
		NICI_OBJECT_HANDLE	masterWrappingKey = 0 );

	RCODE unwrapKey(
		FLMBYTE *				pucWrappedKey,
		FLMUINT32				ui32WrappedKeyLength,
		NICI_OBJECT_HANDLE	masterWrappingKey = 0);

	RCODE extractKey(
		FLMBYTE **		ppucShroudedKey,
		FLMUINT32 *		pui32Length,
		FLMUNICODE *	puzEncKeyPasswd );

	RCODE injectKey(
		FLMBYTE *		pucBuffer,
		FLMUINT32		ui32Length,
		FLMUNICODE *	puzEncKeyPasswd );

	RCODE encryptToStoreAES(
		FLMBYTE *			pucIn,
		FLMUINT				uiInLen,
		FLMBYTE *			pucOut,
		FLMUINT *			puiOutLen);

	RCODE encryptToStoreDES3(
		FLMBYTE *			pucIn,
		FLMUINT				uiInLen,
		FLMBYTE *			pucOut,
		FLMUINT *			puiOutLen);

	RCODE encryptToStoreDES(
		FLMBYTE *			pucIn,
		FLMUINT				uiInLen,
		FLMBYTE *			pucOut,
		FLMUINT *			puiOutLen);

	RCODE decryptFromStoreAES(
		FLMBYTE *				pucIn,
		FLMUINT					uiInLen,
		FLMBYTE *				pucOut,
		FLMUINT *				puiOutLen);

	RCODE decryptFromStoreDES3(
		FLMBYTE *				pucIn,
		FLMUINT					uiInLen,
		FLMBYTE *				pucOut,
		FLMUINT *				puiOutLen);

	RCODE decryptFromStoreDES(
		FLMBYTE *				pucIn,
		FLMUINT					uiInLen,
		FLMBYTE *				pucOut,
		FLMUINT *				puiOutLen);

	RCODE generateEncryptionKeyAES( void );

	RCODE generateEncryptionKeyDES3( void );

	RCODE generateEncryptionKeyDES( void );

	FLMUINT					m_uiAlgType;
	FLMBOOL					m_bInitCalled;
	FLMBOOL					m_bKeyIsWrappingKey;
	FLMBOOL					m_bKeyVerified;
	NICI_OBJECT_HANDLE	m_keyHandle;		// Handle to the clear key - we don't ever get the actual key.
	F_MUTEX					m_hMutex;
	FLMBYTE 					m_pucIV[ IV_SZ];	// Used when the algorithm type is DES, 3DES or AES
};

RCODE flmDecryptBuffer(
	FLMBYTE *				pucBuffer,
	FLMUINT *				puiBufLen);
	
RCODE flmEncryptBuffer(
	FLMBYTE *				pucBuffer,
	FLMUINT *				puiBufLen);

#include "fpackoff.h"

#endif
