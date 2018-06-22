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

#ifndef F_CCS_H
#define F_CCS_H

RCODE flmAllocCCS(
	IF_CCS **		ppCCS);

/****************************************************************************
Desc:	Controlled Cryptographic Services (CCS) interface
****************************************************************************/
class IF_CCS : public F_Object
{
public:

	virtual ~IF_CCS()
	{
	}

	virtual RCODE init(
		FLMBOOL				bKeyIsWrappingKey,
		FLMUINT				uiAlgType) = 0;

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

	virtual RCODE getKeyToStore(
		FLMBYTE **			ppucKeyInfo,
		FLMUINT32 *			pui32BufLen,
		FLMBYTE *			pszEncKeyPasswd = NULL,
		IF_CCS *				pWrappingCcs = NULL) = 0;

	virtual RCODE setKeyFromStore(
		FLMBYTE *			pucKeyInfo,
		FLMBYTE *			pszEncKeyPasswd,
		IF_CCS *				pWrappingCcs) = 0;
		
	virtual FLMUINT getIVLen( void) = 0;
	
	virtual RCODE generateIV(
		FLMUINT				uiIVLen,
		FLMBYTE *			pucIV) = 0;
};

#endif // F_CCS_H
