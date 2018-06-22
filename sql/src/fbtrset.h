//------------------------------------------------------------------------------
// Desc:	This File contains routines which do certain types of verifications
// 		on objects in a FLAIM database.
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

#ifndef BTRSET_H
#define BTRSET_H

#include "f_btpool.h"
#include "f_btree.h"

class IXKeyCompare;

typedef struct BT_TABLE_XREF
{
	FLMUINT					uiKeyNum;
	F_TABLE					table;
	BT_TABLE_XREF *		pNext;
	IXKeyCompare *			pCompare;
} BT_TABLE_XREF;

#define BT_MAX_TABLE_TBL_SIZ 256

/*=============================================================================
Desc:	Result set class that uses an independant database.  The name is randomly
		generated.
=============================================================================*/
class F_BtResultSet : public F_Object
{
public:

	F_BtResultSet(
		F_Db *		pResultSetDb,
		F_BtPool *	pBtPool)
	{
		m_pBtPool = pBtPool;
		m_pResultSetDb = pResultSetDb;
		f_memset( &m_table, 0, sizeof( m_table));
		m_ppTableTbl = NULL;
	}

	~F_BtResultSet();

	// Entry Add and Sort Methods

	RCODE addEntry(						// Variable or fixed length entry coming in
		F_Db *		pSrcDb,				// Set for when we are keeping index keys
		F_INDEX *	pSrcIndex,			// Set for when we are keeping index keys
		FLMBYTE *	pucKey,				// key for sorting.
		FLMUINT		uiKeyLength,
		FLMBYTE *	pEntry,
		FLMUINT		uiEntryLength);	// If length is zero then ignore entry.

	RCODE modifyEntry(					// Modify current entry.
		F_Db *		pSrcDb,				// Set for when we are keeping index keys
		F_INDEX *	pSrcIndex,			// Set for when we are keeping index keys
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyLength,
		FLMBYTE *	pEntry,				// Points to entry buffer
		FLMUINT		uiEntryLength);

	// Methods to read entries.

	RCODE getCurrent(						// Return current entry
		F_Db *		pSrcDb,				// Set for when we are keeping index keys
		F_INDEX *	pSrcIndex,			// Set for when we are keeping index keys
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyLength,
		FLMBYTE *	pucEntry,
		FLMUINT		uiEntryLength,		// Size of Entry buffer.
		FLMUINT *	puiReturnLength);

	RCODE getNext(
		F_Db *		pSrcDb,				// Set for when we are keeping index keys
		F_INDEX *	pSrcIndex,			// Set for when we are keeping index keys
		F_Btree *	pBTree,				// Preserves the context from one call to
												// the next.  May be null if not needed.
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyBufLen,
		FLMUINT *	puiKeylen,
		FLMBYTE *	pucBuffer,
		FLMUINT		uiBufferLength,
		FLMUINT *	puiReturnLength);

	RCODE getPrev(							// Position to previous entry and return
		F_Db *		pSrcDb,				// Set for when we are keeping index keys
		F_INDEX *	pSrcIndex,			// Set for when we are keeping index keys
		F_Btree *	pBTree,				// Preserves the context from one call to
												// the next.  May be null if not needed.
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyBufLen,
		FLMUINT *	puiKeylen,
		FLMBYTE *	pucBuffer,
		FLMUINT		uiBufferLength,
		FLMUINT *	puiReturnLength);

	RCODE getFirst(						// Position to the first entry and return
		F_Db *		pSrcDb,				// Set for when we are keeping index keys
		F_INDEX *	pSrcIndex,			// Set for when we are keeping index keys
		F_Btree *	pBTree,				// Preserves the context from one call to
												// the next.  May be null if not needed.
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyBufLen,
		FLMUINT *	puiKeylen,
		FLMBYTE *	pucBuffer,
		FLMUINT		uiBufferLength,
		FLMUINT *	puiReturnLength);

	RCODE getLast(							// Position to the last entry and return
		F_Db *		pSrcDb,				// Set for when we are keeping index keys
		F_INDEX *	pSrcIndex,			// Set for when we are keeping index keys
		F_Btree *	pBTree,				// Preserves the context from one call to
												// the next.  May be null if not needed.
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyBufLen,
		FLMUINT *	puiKeylen,
		FLMBYTE *	pucBuffer,
		FLMUINT		uiBufferLength,
		FLMUINT *	puiReturnLength);

	RCODE findEntry(						// Locate an entry
		F_Db *		pSrcDb,				// Set for when we are keeping index keys
		F_INDEX *	pSrcIndex,			// Set for when we are keeping index keys
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyBufLen,
		FLMUINT *	puiKeylen,
		FLMBYTE *	pucBuffer,
		FLMUINT		uiBufferLength,
		FLMUINT *	puiReturnLength);

	RCODE deleteEntry(
		F_Db *		pSrcDb,				// Set for when we are keeping index keys
		F_INDEX *	pSrcIndex,			// Set for when we are keeping index keys
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyLength);

	// Methods for managing context

	RCODE getBTree(
		F_Db *		pSrcDb,
		F_INDEX *	pSrcIndex,
		F_Btree **	ppBtree);

	FINLINE void freeBTree(
		F_Btree **		ppBTree)
	{
		flmAssert( *ppBTree);
	
		m_pBtPool->btpReturnBtree( ppBTree);
	
		*ppBTree = NULL;
	}

private:

	F_BtPool *					m_pBtPool;
	F_Db *						m_pResultSetDb;
	F_TABLE 						m_table;
	BT_TABLE_XREF **			m_ppTableTbl;

	friend class F_DbCheck;
};
#endif
