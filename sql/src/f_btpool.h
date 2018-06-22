//------------------------------------------------------------------------------
// Desc:	Header file for the B-Tree pool
// Tabs:	3
//
// Copyright (c) 2002-2007 Novell, Inc. All Rights Reserved.
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

#ifndef F_BTPOOL_H
#define F_BTPOOL_H

#include "f_btree.h"

class F_BtPool : public F_Object
{
public:
	F_BtPool( void)
	{
		m_pBtreeList = NULL;
		m_hMutex = F_MUTEX_NULL;
		m_bInitialized = FALSE;
	}

	~F_BtPool( void)
	{
		while (m_pBtreeList)
		{
			F_Btree *	pBtree;

			pBtree = m_pBtreeList;
			m_pBtreeList = m_pBtreeList->m_pNext;

			pBtree->Release();
		}

		if (m_hMutex != F_MUTEX_NULL)
		{
			f_mutexDestroy( &m_hMutex);
		}

		m_bInitialized = FALSE;
	}

	RCODE btpInit( void);

	RCODE btpReserveBtree(
		F_Btree **		ppBtree);

	void btpReturnBtree(
		F_Btree **		ppBtree);

private:

	F_Btree *		m_pBtreeList;
	F_MUTEX			m_hMutex;
	FLMBOOL			m_bInitialized;
};

#endif



