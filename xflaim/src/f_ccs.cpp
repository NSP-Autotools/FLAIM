//------------------------------------------------------------------------------
// Desc:	Controlled Cryptographic Services (CCS) interface
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

/****************************************************************************
Desc:	Controlled Cryptographic Services (CCS) interface
****************************************************************************/
#ifndef FLM_HAS_ENCRYPTION
class F_NOCCS : public IF_CCS
{
public:

	virtual ~F_NOCCS()
	{
	}

	RCODE init(
		FLMBOOL,				// bKeyIsWrappingKey,
		FLMUINT)				// uiAlgType)
	{
		return( RC_SET( NE_XFLM_ENCRYPTION_UNAVAILABLE));
	}

	RCODE generateEncryptionKey(
		FLMUINT)				// uiEncKeySize)
	{
		return( RC_SET( NE_XFLM_ENCRYPTION_UNAVAILABLE));
	}

	RCODE generateWrappingKey(
		FLMUINT)				// uiEncKeySize)
	{
		return( RC_SET( NE_XFLM_ENCRYPTION_UNAVAILABLE));
	}

	RCODE encryptToStore(
		FLMBYTE *,			// pucIn,
		FLMUINT,				// uiInLen,
		FLMBYTE *,			// pucOut,
		FLMUINT *,			// puiOutLen,
		FLMBYTE *)			// pucIV)
	{
		return( RC_SET( NE_XFLM_ENCRYPTION_UNAVAILABLE));
	}

	RCODE decryptFromStore(
		FLMBYTE *,			// pucIn,
		FLMUINT,				// uiInLen,
		FLMBYTE *,			// pucOut,
		FLMUINT *,			// puiOutLen,
		FLMBYTE *)			// pucIV)
	{
		return( RC_SET( NE_XFLM_ENCRYPTION_UNAVAILABLE));
	}

	RCODE getKeyToStore(
		FLMBYTE **,			// ppucKeyInfo,
		FLMUINT32 *,		// pui32BufLen,
		FLMBYTE *,			// pzEncKeyPasswd,
		IF_CCS *)			// pWrappingCcs)
	{
		return( RC_SET( NE_XFLM_ENCRYPTION_UNAVAILABLE));
	}

	RCODE setKeyFromStore(
		FLMBYTE *,			// pucKeyInfo,
		FLMBYTE *,			// pszEncKeyPasswd,
		IF_CCS *)			// pWrappingCcs)
	{
		return( RC_SET( NE_XFLM_ENCRYPTION_UNAVAILABLE));
	}
		
	FLMUINT getIVLen( void)
	{
		return( 0);
	}
	
	RCODE generateIV(
		FLMUINT,				// uiIVLen,
		FLMBYTE *)			// pucIV)
	{
		return( RC_SET( NE_XFLM_ENCRYPTION_UNAVAILABLE));
	}
};
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifndef FLM_HAS_ENCRYPTION
RCODE flmAllocCCS(
	IF_CCS **		ppCCS)
{
	RCODE				rc = NE_XFLM_OK;
	F_NOCCS *		pCCS = NULL;
	
	f_assert( (*ppCCS) == NULL);
	
	if( (pCCS = f_new F_NOCCS) == NULL)
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
#endif
