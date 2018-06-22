//------------------------------------------------------------------------------
// Desc:	This file contains the definitions needed for the NICI interface
//			functions.
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

#ifndef	_F_NICI_HPP
#define	_F_NICI_HPP

#ifdef FLM_USE_NICI
	#ifdef FLM_NLM
		#define N_PLAT_NLM
	#endif
	
	#include "nwccs.h"

	#ifndef IDV_NOV_AES128CBCPad
		#define IDV_NOV_AES128CBCPad              NICI_AlgorithmPrefix(1), 97 /* 0x61 */
	#endif
#else
	#define NICI_OBJECT_HANDLE void *
	#define NICI_CC_HANDLE FLMUINT32
#endif

/*--------------------------------------------------------------------------
 * Definitions
 *------------------------------------------------------------------------*/

#define IV_SZ						16
#define IV_SZ8						8
#define SALT_SZ					8
#define SALT_COUNT				895

/*-----------------------------------------------------------------------
 * CCS Interface.
 *-----------------------------------------------------------------------*/
class IF_CCS : public F_Object
{
public:

	virtual ~IF_CCS()
	{
	}

	virtual RCODE generateEncryptionKey(
		FLMUINT				uiEncKeySize) = 0;

	virtual RCODE generateWrappingKey(
		FLMUINT				uiEncKeySize) = 0;

	virtual RCODE encryptToStore(
		FLMBYTE *			pucIn,
		FLMUINT				uiInLen,
		FLMBYTE *			pucOut,
		FLMUINT *			puiOutLen,
		FLMBYTE *			pucIV = NULL) = 0;

	virtual RCODE decryptFromStore(
		FLMBYTE *			pucIn,
		FLMUINT				uiInLen,
		FLMBYTE *			pucOut,
		FLMUINT *			puiOutLen,
		FLMBYTE *			pucIV = NULL) = 0;

};  // IF_CCS



class F_CCS : public IF_CCS
{
public:

	// Constructor & destructor
	F_CCS()
	{
		m_bInitCalled = FALSE;
		m_bKeyVerified = FALSE;
		f_memset( m_ucIV, 0, IV_SZ);
		//m_bKeyIsWrappingKey = FALSE;
		//m_uiAlgType = 0xFF;
		m_keyHandle = 0;
		m_hContext = 0;
		m_uiEncKeySize = 0;
		m_hMutex = F_MUTEX_NULL;

	}

	~F_CCS();

	RCODE init(
		FLMBOOL			bKeyIsWrappingKey,
		eEncAlgorithm	eEncAlg);

	RCODE generateEncryptionKey(
		FLMUINT			uiEncKeySize);

	RCODE generateWrappingKey(
		FLMUINT			uiEncKeySize);

	RCODE encryptToStore(
		FLMBYTE *			pucIn,
		FLMUINT				uiInLen,
		FLMBYTE *			pucOut,
		FLMUINT *			puiOutLen,
		FLMBYTE *			pucIV = NULL);

	RCODE decryptFromStore(
		FLMBYTE *			pucIn,
		FLMUINT				uiInLen,
		FLMBYTE *			pucOut,
		FLMUINT *			puiOutLen,
		FLMBYTE *			pucIV = NULL);

	RCODE getKeyToStore(
		FLMBYTE **			ppucKeyInfo,
		FLMUINT32 *			pui32BufLen,
		FLMBYTE *			pzEncKeyPasswd = NULL,
		F_CCS *				pWrappingCcs = NULL);

	RCODE setKeyFromStore(
		FLMBYTE *			pucKeyInfo,
		FLMBYTE *			pszEncKeyPasswd = NULL,
		F_CCS *				pWrappingCcs = NULL);

	FINLINE FLMBOOL keyVerified()
	{
		return m_bKeyVerified;
	}

	FINLINE FLMUINT getEncType( void)
	{
		return m_uiAlgType;
	}
	
	FLMUINT getIVLen();
	
	RCODE generateIV(
		FLMUINT			uiIVLen,
		FLMBYTE *		pucIV);

private:

	RCODE getWrappingKey(
		NICI_OBJECT_HANDLE *		pWrappingKeyHandle);

	RCODE wrapKey(
		FLMBYTE **				ppucWrappedKey,
		FLMUINT32 *				pui32Length,
		NICI_OBJECT_HANDLE	masterWrappingKey = 0 );

	RCODE unwrapKey(
		FLMBYTE *				pucWrappedKey,
		FLMUINT32				ui32WrappedKeyLength,
		NICI_OBJECT_HANDLE	masterWrappingKey = 0);

	RCODE extractKey(
		FLMBYTE **			ppucShroudedKey,
		FLMUINT32 *			pui32Length,
		FLMUNICODE *		puzEncKeyPasswd );

	RCODE injectKey(
		FLMBYTE *			pucBuffer,
		FLMUINT32			ui32Length,
		FLMUNICODE *		puzEncKeyPasswd );

	RCODE encryptToStoreAES(
		FLMBYTE *			pucIn,
		FLMUINT				uiInLen,
		FLMBYTE *			pucOut,
		FLMUINT *			puiOutLen,
		FLMBYTE *			pucIV);

	RCODE encryptToStoreDES3(
		FLMBYTE *			pucIn,
		FLMUINT				uiInLen,
		FLMBYTE *			pucOut,
		FLMUINT *			puiOutLen,
		FLMBYTE *			pucIV);

	RCODE encryptToStoreDES(
		FLMBYTE *			pucIn,
		FLMUINT				uiInLen,
		FLMBYTE *			pucOut,
		FLMUINT *			puiOutLen,
		FLMBYTE *			pucIV);

	RCODE decryptFromStoreAES(
		FLMBYTE *			pucIn,
		FLMUINT				uiInLen,
		FLMBYTE *			pucOut,
		FLMUINT *			puiOutLen,
		FLMBYTE *			pucIV);

	RCODE decryptFromStoreDES3(
		FLMBYTE *			pucIn,
		FLMUINT				uiInLen,
		FLMBYTE *			pucOut,
		FLMUINT *			puiOutLen,
		FLMBYTE *			pucIV);

	RCODE decryptFromStoreDES(
		FLMBYTE *			pucIn,
		FLMUINT				uiInLen,
		FLMBYTE *			pucOut,
		FLMUINT *			puiOutLen,
		FLMBYTE *			pucIV);

	RCODE generateEncryptionKeyAES(
		FLMUINT			uiEncKeySize);

	RCODE generateEncryptionKeyDES3(
		FLMUINT			uiEncKeySize);

	RCODE generateEncryptionKeyDES(
		FLMUINT			uiEncKeySize);

	RCODE generateWrappingKeyAES(
		FLMUINT			uiEncKeySize);

	RCODE generateWrappingKeyDES3(
		FLMUINT			uiEncKeySize);

	RCODE generateWrappingKeyDES(
		FLMUINT			uiEncKeySize);

	FLMUINT					m_uiAlgType;
	FLMBOOL					m_bInitCalled;
	FLMBOOL					m_bKeyIsWrappingKey;
	FLMBOOL					m_bKeyVerified;
	NICI_OBJECT_HANDLE	m_keyHandle;			// Handle to the clear key - we don't ever get the actual key.
	FLMBYTE					m_ucIV[ IV_SZ];		// Used when the algorithm type is DES, 3DES or AES
	FLMBYTE					m_ucRndIV[ IV_SZ];	// Used when the IV is stored with the data.
	FLMUINT					m_uiIVFactor;
	NICI_CC_HANDLE			m_hContext;
	FLMUINT					m_uiEncKeySize;
	F_MUTEX					m_hMutex;

}; // F_CCS

RCODE flmDecryptBuffer(
	FLMBYTE *			pucBuffer,
	FLMUINT *			puiBufLen);
	
RCODE flmEncryptBuffer(
	FLMBYTE *			pucBuffer,
	FLMUINT *			puiBufLen);

#endif	/* _F_NICI_HPP */
