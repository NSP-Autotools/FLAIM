//------------------------------------------------------------------------------
// Desc:	Contains the methods for the F_FixedBlk, F_BtreeRoot, F_BtreeLeaf,
//			F_BtreeNonLeaf, and F_BtreeBlk classes.
// Tabs:	3
//
// Copyright (c) 1998-2007 Novell, Inc. All Rights Reserved.
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

#include "ftksys.h"
#include "ftkdynrset.h"

// Make sure that the extension is in lower case characters. 

#define	FRSET_FILENAME_EXTENSION		"frs"

/****************************************************************************

Organization:

  These modules are orgianized by function and NOT by class.
  For example, all of the searchEntry modules are together.  
  All of the split and getFirst/next/last... modules are together.

****************************************************************************/


/****************************************************************************
					Constructors and Setup Methods
****************************************************************************/

/****************************************************************************
Desc: Set common variables
****************************************************************************/
F_FixedBlk::F_FixedBlk()
{
	m_fnCompare = NULL;
	m_pvUserData = NULL;
	m_uiPosition = DYNSSET_POSITION_NOT_SET;
	m_bDirty = FALSE;
	m_pucBlkBuf = NULL;
}

/****************************************************************************
Desc:
****************************************************************************/
F_BtreeRoot::F_BtreeRoot()
{
	int			i;

	m_pFileHdl = NULL;
	m_pszFileName = NULL;
	m_eBlkType = ACCESS_BTREE_ROOT;
	m_uiEntryOvhd = 4;
	m_uiLRUCount = 1;
	m_uiLevels = 1;
	m_uiNewBlkAddr = 0;
	m_uiHighestWrittenBlkAddr = 0;
	m_uiTotalEntries = 0;

	// Initialize the cache blocks.
	for( i = 0; i < FBTREE_CACHE_BLKS; i++)
	{
		m_CacheBlks[i].uiBlkAddr = 0xFFFFFFFF;
		m_CacheBlks[i].uiLRUValue = 0;
		m_CacheBlks[i].pBlk = NULL;
	}
}

/****************************************************************************
Desc:
****************************************************************************/
F_BtreeRoot::~F_BtreeRoot()
{
	int			i;

	closeFile();

	for( i = 0; i < FBTREE_CACHE_BLKS; i++)
	{
		if( m_CacheBlks[i].pBlk)
		{
			m_CacheBlks[i].pBlk->Release();
		}
	}
}

/****************************************************************************
Desc:	Allocate structures and set entry size.
****************************************************************************/
RCODE F_BtreeLeaf::setup(
	FLMUINT		uiEntrySize)
{
	RCODE	rc = NE_FLM_OK;

	if (RC_BAD( rc = f_calloc( DYNSSET_BLOCK_SIZE, &m_pucBlkBuf)))
	{
		goto Exit;
	}

	m_uiEntrySize = uiEntrySize;
	m_pvUserData = (void *) uiEntrySize;
	reset( ACCESS_BTREE_LEAF);
	nextBlk( FBTREE_END);
	prevBlk( FBTREE_END);
	lemBlk( FBTREE_END);
	reset( ACCESS_BTREE_LEAF);

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Allocate structures and set entry size.
****************************************************************************/
RCODE F_BtreeNonLeaf::setup(
	FLMUINT		uiEntrySize)
{
	RCODE	rc = NE_FLM_OK;

	if (RC_BAD( rc = f_calloc( DYNSSET_BLOCK_SIZE, &m_pucBlkBuf)))
	{
		goto Exit;
	}

	m_uiEntrySize = uiEntrySize;
	m_pvUserData = (void *) uiEntrySize;
	reset( ACCESS_BTREE_NON_LEAF);
	nextBlk( FBTREE_END);
	prevBlk( FBTREE_END);
	lemBlk( FBTREE_END);

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Allocate structures and set entry size.
****************************************************************************/
RCODE F_BtreeRoot::setup(
	FLMUINT		uiEntrySize,
	char *		pszFileName)
{
	RCODE			rc;

	if (RC_BAD( rc = f_calloc( DYNSSET_BLOCK_SIZE, &m_pucBlkBuf)))
	{
		goto Exit;
	}

	m_uiEntrySize = uiEntrySize;
	m_pvUserData = (void *) uiEntrySize;
	reset( ACCESS_BTREE_ROOT);
	m_pszFileName = pszFileName;
	nextBlk( FBTREE_END);
	prevBlk( FBTREE_END);
	lemBlk( FBTREE_END);

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Setup the block as a new block
****************************************************************************/
void F_BtreeBlk::reset(
	eDynRSetBlkTypes	eBlkType)
{
	m_eBlkType = eBlkType;
	
	if (eBlkType == ACCESS_BTREE_ROOT || eBlkType == ACCESS_BTREE_NON_LEAF)
	{
		m_uiEntryOvhd = 4;
	}
	else
	{
		m_uiEntryOvhd = 0;
	}
	m_uiNumSlots = (DYNSSET_BLOCK_SIZE - sizeof( FixedBlkHdr)) / 
							( m_uiEntrySize + m_uiEntryOvhd);
	entryCount( 0);
	m_uiPosition = DYNSSET_POSITION_NOT_SET;
	m_bDirty = FALSE;
}

/****************************************************************************
Desc:	Return the next entry in the result set.  If the result set
		is not positioned then the first entry will be returned.
****************************************************************************/
RCODE F_BtreeBlk::getNext(
	void *		pvEntryBuffer)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiPos = m_uiPosition;

	// Position to the next/first entry.

	if (uiPos == DYNSSET_POSITION_NOT_SET)
	{
		uiPos = 0;
	}
	else
	{
		if (++uiPos > entryCount())
		{
			rc = RC_SET( NE_FLM_EOF_HIT);
			goto Exit;
		}
	}
	f_memcpy( pvEntryBuffer, ENTRY_POS(uiPos), m_uiEntrySize);
	m_uiPosition = uiPos;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Return the last entry in the result set.
****************************************************************************/
RCODE F_BtreeBlk::getLast(
	void *		pvEntryBuffer)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiPos = entryCount();

	// Position to the next/first entry.

	if (uiPos == 0)
	{
		rc = RC_SET( NE_FLM_EOF_HIT);
		goto Exit;
	}

	uiPos--;
	f_memcpy( pvEntryBuffer, ENTRY_POS(uiPos), m_uiEntrySize);
	m_uiPosition = uiPos;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Search a btree.  Position for get* or for insert.
****************************************************************************/
RCODE F_BtreeRoot::search(
	void *	pvEntry,
	void *	pvFoundEntry)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiCurLevel = m_uiLevels - 1;	// Min 2 levels
	FLMUINT		uiBlkAddr;
	
	// Reset the stack - only needed for debugging.
	//f_memset( m_BTStack, 0, sizeof(F_BtreeBlk *) * FBTREE_MAX_LEVELS);

	// Search this root block.

	m_BTStack[ uiCurLevel] = this;
	(void) searchEntry( pvEntry, &uiBlkAddr);
	
	while( uiCurLevel--)
	{
		// Read the next block and place at uiCurLevel (backwards from FS).

		if( RC_BAD( rc = readBlk( uiBlkAddr,
					uiCurLevel ? ACCESS_BTREE_NON_LEAF : ACCESS_BTREE_LEAF,
					&m_BTStack[ uiCurLevel] )))
		{
			goto Exit;
		}

		// Set the rc - only for the leaf block, otherwise rc should be ignored.

		rc = m_BTStack[ uiCurLevel]->searchEntry( pvEntry, &uiBlkAddr, 
					uiCurLevel ? NULL : pvFoundEntry);
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Search a single block tree.  Position for get* or for insert.
		Do a binary search on all of the entries to find a match.
		If no match then position to the entry where an insert 
		will take place.
****************************************************************************/
RCODE F_BtreeBlk::searchEntry(
	void *		pvEntry,
	FLMUINT *	puiChildAddr,
	void *		pvFoundEntry)
{
	RCODE			rc = RC_SET( NE_FLM_NOT_FOUND);
	FLMUINT		uiLow;
	FLMUINT		uiMid;
	FLMUINT		uiHigh;
	FLMUINT		uiTblSize;
	FLMINT		iCompare;

	// check for zero entries.

	if (!entryCount())
	{
		uiMid = 0;
		goto Exit;
	}
	uiHigh = uiTblSize = entryCount() - 1;
	uiLow = 0;
	for(;;)
	{
		uiMid = (uiLow + uiHigh) >> 1;		// (uiLow + uiHigh) / 2

		// Use compare routine

		if (m_fnCompare)
		{
			iCompare = m_fnCompare( pvEntry, ENTRY_POS( uiMid), m_pvUserData);
		}
		else
		{
			iCompare = f_memcmp( pvEntry, ENTRY_POS( uiMid), m_uiEntrySize);
		}

		if (iCompare == 0)
		{
			if (pvFoundEntry)
			{
				f_memcpy( pvFoundEntry, ENTRY_POS( uiMid), m_uiEntrySize);
			}
			rc = NE_FLM_OK;
			goto Exit;
		}

		// Check if we are done - where wLow equals uiHigh or mid is at end.

		if (iCompare < 0)
		{
			if (uiMid == uiLow || uiLow == uiHigh)
			{
				break;
			}
			uiHigh = uiMid - 1;					// Too high
		}
		else
		{
			if (uiMid == uiHigh || uiLow == uiHigh)
			{

				// Go up one for the correct position?

				uiMid++;
				break;
			}
			uiLow = uiMid + 1;					// Too low
		}
	}

Exit:

	m_uiPosition = uiMid;
	if (puiChildAddr && blkType() != ACCESS_BTREE_LEAF)
	{
		if (uiMid == entryCount())
		{
			*puiChildAddr = lemBlk();
		}
		else
		{
			FLMBYTE *	pucChildAddr = ENTRY_POS(uiMid) + m_uiEntrySize;
			*puiChildAddr = (FLMUINT)FB2UD( pucChildAddr);
		}
	}
	return( rc);
}

/****************************************************************************
Desc:	Insert the entry into the btree - should be positioned
****************************************************************************/
RCODE F_BtreeRoot::insert(
	void *			pvEntry)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiCurLevel;
	FLMBYTE			ucEntryBuf[FBTREE_MAX_LEVELS][DYNSSET_MAX_FIXED_ENTRY_SIZE];
	FLMUINT			uiNewBlkAddr;

	if (RC_OK( rc = m_BTStack[0]->insert( pvEntry)))
	{
		goto Exit;
	}

	// Failed to input at the left level.  Do block split(s).
	// This is an iterative and NOT a recursive split algorithm.
	// The debugging, and cases to test should be lots easier this way.
	
	f_memcpy( ucEntryBuf[0], pvEntry, m_uiEntrySize);
	uiCurLevel = 0;
	uiNewBlkAddr = FBTREE_END;
	for(;;)
	{

		// Split while adding the element.

		if (RC_BAD( rc = (m_BTStack[uiCurLevel])->split( 
											this,
											ucEntryBuf[ uiCurLevel],
											uiNewBlkAddr,	
											ucEntryBuf[ uiCurLevel+1],
											&uiNewBlkAddr)))
		{
			goto Exit;
		}

		uiCurLevel++;
		flmAssert( uiCurLevel < m_uiLevels);

		if (RC_OK( rc = m_BTStack[uiCurLevel]->insertEntry( 
									ucEntryBuf[uiCurLevel], uiNewBlkAddr)))
		{
			goto Exit;
		}

		// Only returns NE_FLM_OK or FAILURE.

		// Root split?

		if (uiCurLevel + 1 == m_uiLevels)
		{
			flmAssert( m_uiLevels + 1 <= FBTREE_MAX_LEVELS);
			if (m_uiLevels + 1 > FBTREE_MAX_LEVELS)
			{
				rc = RC_SET( NE_FLM_BTREE_FULL);
				goto Exit;
			}

			// Need to split the root block.
			 rc = ((F_BtreeRoot *)m_BTStack[uiCurLevel])->split( 
						ucEntryBuf[uiCurLevel], uiNewBlkAddr );
			 break;
		}
	}

Exit:

	if (RC_OK(rc))
	{
		m_uiTotalEntries++;
	}
	return( rc);
}

/****************************************************************************
Desc:	Insert the entry into the buffer.
****************************************************************************/
RCODE F_BtreeBlk::insertEntry(
	void *		pvEntry,
	FLMUINT		uiChildAddr)
{
	RCODE			rc = NE_FLM_OK;
	FLMBYTE *	pucCurEntry;
	FLMUINT		uiShiftBytes;		// Always shift down

	if( entryCount() >= m_uiNumSlots)
	{
		rc = RC_SET( NE_FLM_FAILURE);
		goto Exit;
	}
	flmAssert( m_uiPosition != DYNSSET_POSITION_NOT_SET);
	pucCurEntry = ENTRY_POS( m_uiPosition);
	if ((uiShiftBytes = (entryCount() - m_uiPosition) * 
								(m_uiEntrySize + m_uiEntryOvhd)) != 0)
	{

		// Big hairy assert.  Finds coding bugs and corruptions.

		flmAssert( m_uiPosition * (m_uiEntrySize + m_uiEntryOvhd) + 
						uiShiftBytes < DYNSSET_BLOCK_SIZE - sizeof( FixedBlkHdr));

		f_memmove( pucCurEntry + m_uiEntrySize + m_uiEntryOvhd, 
						pucCurEntry, uiShiftBytes);
	}
	f_memcpy( pucCurEntry, pvEntry, m_uiEntrySize);
	if( m_uiEntryOvhd)
	{
		UD2FBA( (FLMUINT32)uiChildAddr, &pucCurEntry[m_uiEntrySize]);
	}
	entryCount( entryCount() + 1);
	m_uiPosition++;
	m_bDirty = TRUE;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Move first half of entries into new block.  Reset previous block
		to point to new block.  Add new last entry in new block to parent.
		Fixup prev/next linkages.
****************************************************************************/
RCODE F_BtreeBlk::split(
	F_BtreeRoot *	pRoot,
	FLMBYTE *		pucCurEntry,		// (in) Contains entry to insert
	FLMUINT			uiCurBlkAddr,		// (in) Blk addr if non-leaf
	FLMBYTE *		pucParentEntry,	// (out) Entry to insert into parent.
	FLMUINT *		puiNewBlkAddr)		// (out) New blk addr to insert into parent.
{
	RCODE				rc = NE_FLM_OK;
	F_BtreeBlk *		pPrevBlk;
	F_BtreeBlk *		pNewBlk = NULL;
	FLMBYTE *		pucEntry = NULL;
	FLMBYTE *		pucMidEntry;
	FLMBYTE *		pucChildAddr;
	FLMUINT			uiChildAddr;
	FLMUINT			uiPrevBlkAddr;
	FLMUINT			uiMid;
	FLMUINT			uiPos;
	FLMUINT			uiMoveBytes;
	FLMBOOL			bInserted = FALSE;

	// Allocate a new block for the split.

	if (RC_BAD( rc = pRoot->newBlk( &pNewBlk, blkType() )))
	{
		goto Exit;
	}
	pNewBlk->AddRef();				// Pin the block - may get flushed out.


	// the last half into the new block, but that would force the parent
	// entry to be changed.  This may take a little longer, but it is much
	// more easier to code.

	// Call search entry once just to setup for insert.  

	(void) pNewBlk->searchEntry( ENTRY_POS( 0));

	// get the count and move more then half into the new block.

	uiMid = (entryCount() + 5) >> 1;
	uiChildAddr = FBTREE_END;

	for (uiPos = 0; uiPos < uiMid; uiPos++)
	{
		pucEntry = ENTRY_POS( uiPos);
		if (blkType() != ACCESS_BTREE_LEAF)
		{
			pucChildAddr = pucEntry + m_uiEntrySize;
			uiChildAddr = (FLMUINT)FB2UD(pucChildAddr);
		}

		// m_uiPosition automatically gets incremented.

		if (RC_BAD( rc = pNewBlk->insertEntry( pucEntry, uiChildAddr)))
		{
			RC_UNEXPECTED_ASSERT( rc);
			goto Exit;
		}
	}
	
	if (m_uiPosition < uiMid)
	{

		// Insert this entry now

		bInserted = TRUE;
		(void) pNewBlk->searchEntry( pucCurEntry);
		if (RC_BAD( rc = pNewBlk->insertEntry( pucCurEntry, uiCurBlkAddr)))
		{
			goto Exit;
		}
	}

	// Let caller insert into parent entry.  This rids us of recursion.

	f_memcpy( pucParentEntry, pucEntry, m_uiEntrySize);

	// Move the rest down

	pucEntry = ENTRY_POS( 0);
	pucMidEntry = ENTRY_POS( uiMid);

	entryCount( entryCount() - uiMid);
	uiMoveBytes = entryCount() * (m_uiEntrySize + m_uiEntryOvhd);
	flmAssert( uiMoveBytes < DYNSSET_BLOCK_SIZE - sizeof( FixedBlkHdr));
	f_memmove( pucEntry, pucMidEntry, uiMoveBytes);

	if( !bInserted)
	{

		// m_uiPosition -= uiMid;

		(void) searchEntry( pucCurEntry);
		if (RC_BAD( rc = insertEntry( pucCurEntry, uiCurBlkAddr)))
		{
			goto Exit;
		}
	}

	// VISIT: Could position stack to point to current element to insert.

	// Fixup the prev/next block linkages.

	if (prevBlk() != FBTREE_END)
	{
		if (RC_BAD( rc = pRoot->readBlk( prevBlk(), blkType(), &pPrevBlk )))
		{
			goto Exit;
		}

		pPrevBlk->nextBlk( pNewBlk->blkAddr());
		uiPrevBlkAddr = pPrevBlk->blkAddr();
	}
	else
	{
		uiPrevBlkAddr = FBTREE_END;
	}
	pNewBlk->prevBlk( uiPrevBlkAddr);
	pNewBlk->nextBlk( blkAddr());
	prevBlk( pNewBlk->blkAddr());

	*puiNewBlkAddr = pNewBlk->blkAddr();

Exit:

	if (pNewBlk)
	{
		pNewBlk->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Reinsert all entries given a new root block.
		Caller will release 'this'.  Used ONLY for building the first
		ROOT and two leaves of the tree.
****************************************************************************/
RCODE F_BtreeLeaf::split(
	F_BtreeRoot *	pNewRoot)		// New Non-leaf root
{
	RCODE				rc = NE_FLM_OK;
	FLMBYTE *		pucEntry;
	FLMUINT			uiPos;
	FLMUINT			uiEntryCount = entryCount();
	FLMUINT			uiMid = (uiEntryCount + 1) >> 1;

	if (RC_BAD( rc = pNewRoot->setupTree( ENTRY_POS(uiMid), 
								ACCESS_BTREE_LEAF, NULL, NULL)))
	{
		goto Exit;
	}

	for (uiPos = 0; uiPos < uiEntryCount; uiPos++)
	{
		pucEntry = ENTRY_POS( uiPos);
		if ((rc = pNewRoot->search( pucEntry)) != NE_FLM_NOT_FOUND)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_FAILURE);
			goto Exit;
		}
		
		if (RC_BAD( rc = pNewRoot->insert( pucEntry)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Split the root block and make two new non-leaf blocks.
		The secret here is that the root block never moves (cheers!).
		This takes a little longer but is worth the work because the
		root block never goes out to disk and is not in the cache.
****************************************************************************/
RCODE F_BtreeRoot::split(
	void *			pvCurEntry,
	FLMUINT			uiCurChildAddr)
{
	RCODE				rc = NE_FLM_OK;
	FLMBYTE *		pucEntry;
	FLMBYTE *		pucChildAddr;
	F_BtreeBlk *		pLeftBlk;
	F_BtreeBlk *		pRightBlk;
	F_BtreeBlk *		pBlk;
	FLMUINT			uiChildAddr;
	FLMUINT			uiPos;
	FLMUINT			uiEntryCount = entryCount();
	FLMUINT			uiMid = (uiEntryCount + 1) >> 1;

	if (RC_BAD( rc = setupTree( NULL, ACCESS_BTREE_NON_LEAF,
											&pLeftBlk, &pRightBlk)))
	{
		goto Exit;
	}

	// Call search entry once just to setup for insert.

	(void) pLeftBlk->searchEntry( ENTRY_POS( 0));

	// Take the entries from the root block and move into leafs.

	for (uiPos = 0; uiPos <= uiMid; uiPos++)
	{
		pucEntry = ENTRY_POS( uiPos);
		pucChildAddr = pucEntry + m_uiEntrySize;
		uiChildAddr = (FLMUINT)FB2UD(pucChildAddr);

		if (RC_BAD( rc = pLeftBlk->insertEntry( pucEntry, uiChildAddr)))
		{
			goto Exit;
		}
	}

	// Call search entry once just to setup for insert.

	(void) pRightBlk->searchEntry( ENTRY_POS( 0));

	for (uiPos = uiMid + 1; uiPos < uiEntryCount; uiPos++)
	{
		pucEntry = ENTRY_POS( uiPos);
		pucChildAddr = pucEntry + m_uiEntrySize;
		uiChildAddr = (FLMUINT)FB2UD(pucChildAddr);

		if ((rc = pRightBlk->searchEntry( pucEntry )) != NE_FLM_NOT_FOUND)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_FAILURE);
			goto Exit;
		}
		
		if (RC_BAD( rc = pRightBlk->insertEntry( pucEntry, uiChildAddr)))
		{
			goto Exit;
		}
	}

	// Reset the root block and insert new midpoint.

	entryCount( 0);
	lemBlk( pRightBlk->blkAddr());	// Duplicated just in case.
	pucEntry = ENTRY_POS( uiMid);

	if ((rc = searchEntry( pucEntry )) != NE_FLM_NOT_FOUND)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_FAILURE);
		goto Exit;
	}
	
	if (RC_BAD( rc = insertEntry( pucEntry, pLeftBlk->blkAddr() )))
	{
		goto Exit;
	}

	// Insert the current entry (parameters) into the left or right blk.
	// This could be done a number of different ways.
	(void) searchEntry( pvCurEntry, &uiChildAddr);
	if (RC_BAD( rc = readBlk( uiChildAddr, ACCESS_BTREE_NON_LEAF, &pBlk)))
	{
		goto Exit;
	}
	(void) pBlk->searchEntry( pvCurEntry);
	if (RC_BAD( rc = pBlk->insertEntry( pvCurEntry, uiCurChildAddr)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Setup two child blocks for a root block.
****************************************************************************/
RCODE F_BtreeRoot::setupTree(
	FLMBYTE *		pucMidEntry,		// If !NULL entry to insert into root.
	eDynRSetBlkTypes		eBlkType,			// Leaf or non-leaf
	F_BtreeBlk **	ppLeftBlk,			// (out)
	F_BtreeBlk **	ppRightBlk)			// (out)
{
	RCODE			rc = NE_FLM_OK;
	F_BtreeBlk *	pLeftBlk = NULL;
	F_BtreeBlk *	pRightBlk = NULL;

	if (RC_BAD( rc = newBlk( &pLeftBlk, eBlkType)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = newBlk( &pRightBlk, eBlkType)))
	{
		goto Exit;
	}

	if (eBlkType == ACCESS_BTREE_NON_LEAF)
	{
		((F_BtreeNonLeaf *)pRightBlk)->lemBlk( lemBlk());
	}

	// Fix up the linkages

	pLeftBlk->nextBlk( pRightBlk->blkAddr());
	pRightBlk->prevBlk( pLeftBlk->blkAddr());
	lemBlk( pRightBlk->blkAddr());

	if (pucMidEntry)
	{

		// Add the midentry to the root block.  Search to position and insert.

		searchEntry( pucMidEntry);
		insertEntry( pucMidEntry, pLeftBlk->blkAddr());
	}
	m_uiLevels++;
	
	if (ppLeftBlk)
	{
		*ppLeftBlk = pLeftBlk;
	}
	if (ppRightBlk)
	{
		*ppRightBlk = pRightBlk;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Read in the block or get it from the cache.
****************************************************************************/
RCODE F_BtreeRoot::readBlk(
	FLMUINT			uiBlkAddr,			// Blk address to read
	eDynRSetBlkTypes		eBlkType,			// Expected access type to read
	F_BtreeBlk **	ppBlk)				// (out) Return block
{ 
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiPos;
	FLMUINT		uiLRUValue = (FLMUINT)~0;
	FLMUINT		uiLRUPos = 0;
	F_BtreeBlk *	pNewBlk;

	for (uiPos = 0; uiPos < FBTREE_CACHE_BLKS; uiPos++)
	{
		if (m_CacheBlks[uiPos].uiBlkAddr == uiBlkAddr)
		{
			goto Exit;
		}

		// The ref count is used for pinning the block.

		if (m_CacheBlks[uiPos].pBlk &&
			 m_CacheBlks[uiPos].pBlk->getRefCount() == 1 &&
			 uiLRUValue > m_CacheBlks[uiPos].uiLRUValue)
		{
			uiLRUValue = m_CacheBlks[uiPos].uiLRUValue;
			uiLRUPos = uiPos;
		}

		// There better not be a hole by this point.

		flmAssert( m_CacheBlks[uiPos].pBlk != NULL);
	}
	uiPos = uiLRUPos;

	// Read from disk?

	flmAssert( m_pFileHdl != NULL);

	if (RC_BAD( rc = newCacheBlk( uiPos, &pNewBlk, eBlkType)))
	{
		goto Exit;
	}

	// Pick the LRU block and make that object do the reading
	// so it can reset all internals and get used to being a different blk.

	pNewBlk->blkAddr( uiBlkAddr);
	m_CacheBlks[uiPos].uiBlkAddr = uiBlkAddr;
	m_CacheBlks[uiPos].uiLRUValue = m_uiLRUCount++;

	if (RC_BAD( rc = pNewBlk->readBlk( m_pFileHdl, uiBlkAddr)))
	{

		// Release the block because the reset() changed the object type.
		// May hit the assert above.

		m_CacheBlks[uiPos].pBlk->Release();
		m_CacheBlks[uiPos].pBlk = NULL;
		goto Exit;
	}

Exit:

	if (RC_OK(rc))
	{
		*ppBlk = m_CacheBlks[uiPos].pBlk;
		m_CacheBlks[uiPos].uiLRUValue = m_uiLRUCount++;
	}
	return( rc);
}

/****************************************************************************
Desc:	Get a new block using an exising or newly allocated block from
		the cache.  Initializes the block.  May be leaf or non-leaf
		but NOT the root block.
****************************************************************************/
RCODE F_BtreeRoot::newBlk(
	F_BtreeBlk **	ppBlk,
	eDynRSetBlkTypes		eBlkType)
{ 
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiLRUValue = (FLMUINT)~0;
	FLMUINT		uiPos;
	FLMUINT		uiLRUPos = 0;
	F_BtreeBlk *	pNewBlk;

	for (uiPos = 0; uiPos < FBTREE_CACHE_BLKS; uiPos++)
	{

		// The ref count is used for pinning the block.

		if (getRefCount() == 1 &&
			 uiLRUValue > m_CacheBlks[uiPos].uiLRUValue)
		{
			uiLRUValue = m_CacheBlks[uiPos].uiLRUValue;
			uiLRUPos = uiPos;
		}

		// use the first hole.

		if (m_CacheBlks[uiPos].pBlk == NULL)
		{
			uiLRUPos = uiPos;
			break;
		}
	}
	uiPos = uiLRUPos;
	if (RC_BAD( rc = newCacheBlk( uiPos, &pNewBlk, eBlkType)))
	{
		goto Exit;
	}

	pNewBlk->blkAddr( newBlkAddr());
	m_CacheBlks[uiPos].uiBlkAddr = pNewBlk->blkAddr();
	m_CacheBlks[uiPos].uiLRUValue = m_uiLRUCount++;

	pNewBlk->entryCount(0);
	pNewBlk->lemBlk(  FBTREE_END);
	pNewBlk->nextBlk( FBTREE_END);
	pNewBlk->prevBlk( FBTREE_END);
	*ppBlk = pNewBlk;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Release the existing cache block and setup and alloc new blk.
****************************************************************************/
RCODE F_BtreeRoot::newCacheBlk(
	FLMUINT			uiCachePos,
	F_BtreeBlk **	ppBlk,
	eDynRSetBlkTypes		eBlkType)
{ 
	RCODE			rc = NE_FLM_OK;
	F_BtreeBlk *	pNewBlk = NULL;

	if (m_CacheBlks[uiCachePos].pBlk)
	{
		if (m_CacheBlks[uiCachePos].pBlk->isDirty())
		{
			if (RC_BAD( rc = writeBlk( uiCachePos)))
			{
				goto Exit;
			}
		}
	}

	if (m_CacheBlks[uiCachePos].pBlk != NULL && 
		 m_CacheBlks[uiCachePos].pBlk->blkType() == eBlkType)
	{

		// If block is of the same type then reset it and use it.

		pNewBlk = m_CacheBlks[uiCachePos].pBlk;
		pNewBlk->reset( eBlkType);
		*ppBlk = pNewBlk;
		goto Exit;
	}

	if (m_CacheBlks[uiCachePos].pBlk)
	{
		m_CacheBlks[uiCachePos].pBlk->Release();
	}
	if (eBlkType == ACCESS_BTREE_LEAF)
	{
		F_BtreeLeaf * pLeafBlk;
		if ((pLeafBlk = f_new F_BtreeLeaf) == NULL)
		{
			rc = RC_SET( NE_FLM_MEM);
			goto Exit;
		}
		if (RC_BAD( rc = pLeafBlk->setup( m_uiEntrySize)))
		{
			pLeafBlk->Release();
			goto Exit;
		}
		pLeafBlk->setCompareFunc( m_fnCompare, m_pvUserData);
		pNewBlk = (F_BtreeBlk *) pLeafBlk;
	}
	else
	{
		F_BtreeNonLeaf * pNonLeafBlk;
		if ((pNonLeafBlk = f_new F_BtreeNonLeaf) == NULL)
		{
			rc = RC_SET( NE_FLM_MEM);
			goto Exit;
		}
		if (RC_BAD( rc = pNonLeafBlk->setup( m_uiEntrySize)))
		{
			pNonLeafBlk->Release();
			goto Exit;
		}
		pNonLeafBlk->setCompareFunc( m_fnCompare, m_pvUserData);
		pNewBlk = (F_BtreeBlk *) pNonLeafBlk;
	}
	m_CacheBlks[uiCachePos].pBlk = pNewBlk;
	*ppBlk = pNewBlk;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Read the block from disk.
****************************************************************************/
RCODE F_BtreeBlk::readBlk(
	IF_FileHdl *	pFileHdl,
	FLMUINT			uiBlkAddr)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiBytesRead;

	if (RC_BAD( rc = pFileHdl->read( 
						uiBlkAddr * DYNSSET_BLOCK_SIZE, DYNSSET_BLOCK_SIZE, 
						m_pucBlkBuf, &uiBytesRead)))
	{
		RC_UNEXPECTED_ASSERT( rc);
		goto Exit;
	}

	if (blkAddr() != uiBlkAddr)
	{

		// Most likely a coding error rather than an I/O error.

		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Write the block to disk.
****************************************************************************/
RCODE F_BtreeBlk::writeBlk(
	IF_FileHdl *	pFileHdl)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiBytesWritten;
	FLMUINT		uiBlkAddr = blkAddr();
			
	if (RC_BAD( rc = pFileHdl->write( 
						uiBlkAddr * DYNSSET_BLOCK_SIZE,
						DYNSSET_BLOCK_SIZE, 
						m_pucBlkBuf,
						&uiBytesWritten)))
	{
		goto Exit;
	}

	m_bDirty = FALSE;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Write all blocks that are dirty and have an addrees lower
		than the input block address and then write this block.
		Write in order so that
		we don't have to write zeros for any block.
****************************************************************************/
RCODE F_BtreeRoot::writeBlk(
	FLMUINT			uiWritePos)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiPos;
	FLMUINT		uiHighBlkAddr = m_CacheBlks[uiWritePos].uiBlkAddr;

	if (!m_pFileHdl)
	{
		if (RC_BAD( rc = openFile()))
		{
			goto Exit;
		}
	}
	for (uiPos = 0; uiPos < FBTREE_CACHE_BLKS; uiPos++)
	{
		if( (uiWritePos != uiPos) &&
			 (m_CacheBlks[uiPos].pBlk) &&
			 (m_CacheBlks[uiPos].uiBlkAddr >= m_uiHighestWrittenBlkAddr) &&
			 (m_CacheBlks[uiPos].uiBlkAddr < uiHighBlkAddr) &&
			 (m_CacheBlks[uiPos].pBlk->isDirty()) )
		{

			// Recursive call to write out lower blocks if needed.

			if (RC_BAD( rc = writeBlk( uiPos)))
			{
				goto Exit;
			}
		}
	}
	m_CacheBlks[ uiWritePos].pBlk->writeBlk( m_pFileHdl);
	if (m_CacheBlks[uiWritePos].uiBlkAddr > m_uiHighestWrittenBlkAddr)
	{
		m_uiHighestWrittenBlkAddr = m_CacheBlks[uiWritePos].uiBlkAddr;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Close the file if previously opened and creates the file.
****************************************************************************/
RCODE F_BtreeRoot::openFile( void)
{ 
	RCODE				rc = NE_FLM_OK;

	if (!m_pFileHdl)
	{
		F_FileSystem	fileSystem;

		rc = fileSystem.createUniqueFile( m_pszFileName,
								FRSET_FILENAME_EXTENSION,
								FLM_IO_RDWR | FLM_IO_CREATE_DIR, &m_pFileHdl);
	}
	return( rc);
}

/****************************************************************************
Desc:	Closes and deletes the temporary file.
****************************************************************************/
void F_BtreeRoot::closeFile( void)
{
	if (m_pFileHdl)
	{
		F_FileSystem	fileSystem;

		m_pFileHdl->closeFile();
		fileSystem.deleteFile( m_pszFileName);
		m_pFileHdl->Release();
		m_pFileHdl = NULL;
	}
}
