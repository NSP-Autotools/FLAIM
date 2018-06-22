//------------------------------------------------------------------------------
// Desc:	Insert and delete keys in an index B-Tree.
// Tabs:	3
//
// Copyright (c) 1991-2007 Novell, Inc. All Rights Reserved.
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

/***************************************************************************
Desc:	Update (add or delete) a single reference
*****************************************************************************/
RCODE F_Db::refUpdate(
	LFILE *			pLFile,
	IXD *				pIxd,
	KREF_ENTRY *	pKrefEntry,
	FLMBOOL			bNormalUpdate)
{
	RCODE				rc = NE_XFLM_OK;
	F_Btree *		pbtree = NULL;
	IXKeyCompare	compareObject;

	// Get a btree

	if (RC_BAD( rc = gv_XFlmSysData.pBtPool->btpReserveBtree( &pbtree)))
	{
		goto Exit;
	}

	flmAssert( pLFile->uiRootBlk);

	compareObject.setIxInfo( this, pIxd);
	if (bNormalUpdate && pKrefEntry->bDelete)
	{
		compareObject.setOldNodeList( m_pOldNodeList);
	}
	if( RC_BAD( rc = pbtree->btOpen( this, pLFile,
		(pIxd->uiFlags & IXD_ABS_POS) ? TRUE : FALSE,
		(pIxd->pFirstData) ? TRUE : FALSE, &compareObject)))
	{
		goto Exit;
	}

	if (!pKrefEntry->bDelete)
	{
		pbtree->btResetBtree();
		if( RC_BAD( rc = pbtree->btInsertEntry(
							(FLMBYTE *)&pKrefEntry [1], pKrefEntry->ui16KeyLen,
							pKrefEntry->uiDataLen
							? ((FLMBYTE *)(&pKrefEntry [1])) + 1 + pKrefEntry->ui16KeyLen
							: NULL,
							pKrefEntry->uiDataLen, TRUE, TRUE)))
		{
			goto Exit;
		}
	}
	else
	{
		pbtree->btResetBtree();
		if (RC_BAD( rc = pbtree->btRemoveEntry(
							(FLMBYTE *)&pKrefEntry [1], pKrefEntry->ui16KeyLen)))
		{
			if (rc == NE_XFLM_NOT_FOUND)
			{
				// Already been deleted, ignore the error condition and go on.

				RC_UNEXPECTED_ASSERT( rc);
				rc = NE_XFLM_OK;
			}

			goto Exit;
		}
	}

Exit:

	if (pbtree)
	{
		gv_XFlmSysData.pBtPool->btpReturnBtree( &pbtree);
	}

	return( rc);
}
