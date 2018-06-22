//------------------------------------------------------------------------------
// Desc:	Result set routines
// Tabs:	3
//
// Copyright (c) 1996-1998, 2003-2007 Novell, Inc. All Rights Reserved.
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

/*
** Sorting Result Sets:
**
	New algorithm 7/2/97.  This is a good one!
	Below are refinements to the existing result set code.

	1) We now have two files that are used in the merge-sort process.
		The first file is used to hold all of the blocks created when
		adding entries into the result set.  The second file is used for
		the first and thereafter odd merge steps.  The first file is then
		truncated and used for each even merge step.  At the end of the
		merge one of the files will be deleted.  Three buffers are used
		during merge and only one will remain after the merge is done.
		This is safer than the previous method and uses a little less
		disk space.  There are many small improvements that can be made.

	2) The result set code now takes a buffer and a length and has no
		knowledge of the data in the result set.  In fact, the data may
		be fixed length or variable length.  We removed the record cache
		from the result set and made a record cache manager.

  Future enhancements to consider:
	1) Do a 3, 4 or N-Way merge.
		This will greatly increase memory allocations but save a lot of time
		reading and writing when the number of entries is large.
	2) Use 3 buffers on the initial load.  This is really doing the first
		phase of the merge when adding entries.  The algorithm would add
		entries to two buffers, and when full merge to the third buffer and
		write out two sorted buffer.
	3) Don't write out the last block - use it as the first block of the
		merge when not complete.  This will save a write and read on each
		pass.  In addition, the I/O cache may be helped out.
		In addition, the last block of each phase should be used first
		on the next phase.

  Old Notes:
		Duplicate Entries:
			Duplicate entries are very difficult for a general purpose sorter
			to find.  In some cases the user would want to compare only these
			fields and not these to determine a duplicate.  This result set
			code lets the user pass in a callback routine to determine if
			two entries are the same and if one should be dropped.  The user
			could pass in NULL to cause all duplicates to be retained.
		Quick Sort Algorithm:
			This algorithm, in FRSETBLK.CPP is a great algorithm that Scott
			came up with.  It will recurse only Log(base2)N times (number of
			bits needed to represent N).  This is a breakthrough because
			all sorting algorithms I have seen will recurse N-1 times if
			the data is in order or in reverse order.  This will crash the
			stack for a production quality sort.
		Variable Length
			This sorting engine (result set) supports variable length and
			fixed length data.  There is very low overhead for variable
			length support.  This sorting engine can be used for a variety
			of tasks.

  Example:

	All numbers are logical block numbers.

	Adding		Pass 1		Pass2			Pass3				Pass4
	Phase 		File 2		File 1		File 2			File 1
	File 1		(created)	(truncated)	(truncated)		(truncated)
	=========	=========	=========	===========		====================
	1				10 (1+2)		14 (10+11)	16 (14+15)		17 Final file (16+9)
	2
	3				11 (3+4)		15 (12+13)	9
	4
	5				12 (5+6)		9
	6
	7				13 (7+8)
	8
	9				9
*/

#define FRSET_FILENAME_EXTENSION		"frs"
#define RSBLK_UNSET_FILE_POS			(~((FLMUINT64)0))
#define RS_BLOCK_SIZE					(1024 * 512)
#define RS_MAX_FIXED_ENTRY_SIZE		64
	
/*****************************************************************************
Desc:
*****************************************************************************/
typedef struct
{
	FLMUINT32	ui32Offset;
	FLMUINT32	ui32Length;
} F_VAR_HEADER;

/*****************************************************************************
Desc:
*****************************************************************************/
typedef struct
{
	FLMUINT64	ui64FilePos;
	FLMUINT		uiEntryCount;
	FLMUINT		uiBlockSize;
	FLMBOOL		bFirstBlock;
	FLMBOOL		bLastBlock;
} F_BLOCK_HEADER;

/****************************************************************************
Desc:
****************************************************************************/
class	F_ResultSetBlk : public F_Object
{
public:

	F_ResultSetBlk();

	FINLINE ~F_ResultSetBlk()
	{
		if (m_pNext)
		{
			m_pNext->m_pPrev = m_pPrev;
		}
		
		if( m_pPrev)
		{
			m_pPrev->m_pNext = m_pNext;
		}
		
		if (m_pCompare)
		{
			m_pCompare->Release();
		}
	}

	void reset( void);

	void setup(
		IF_MultiFileHdl **		ppMultiFileHdl,
		IF_ResultSetCompare *	pCompare,
		FLMUINT						uiEntrySize,
		FLMBOOL						bFirstInList,
		FLMBOOL						bDropDuplicates,
		FLMBOOL						bEntriesInOrder);

	RCODE setBuffer(
		FLMBYTE *					pBuffer,
		FLMUINT						uiBufferSize = RS_BLOCK_SIZE);

	FINLINE FLMUINT bytesUsedInBuffer( void)
	{
		if (m_bEntriesInOrder)
		{
			return( m_BlockHeader.uiBlockSize);
		}
		else
		{
			return( m_BlockHeader.uiBlockSize - m_uiLengthRemaining);
		}
	}

	RCODE addEntry(
		FLMBYTE *	pEntry,
		FLMUINT		uiEntryLength);

	RCODE modifyEntry(
		FLMBYTE *	pEntry,
		FLMUINT		uiEntryLength = 0);

	FINLINE RCODE finalize(
		FLMBOOL		bForceWrite)
	{
		return( flush( TRUE, bForceWrite));
	}

	RCODE flush(
		FLMBOOL		bLastBlockInList,
		FLMBOOL		bForceWrite);

	RCODE getCurrent(
		FLMBYTE *	pBuffer,
		FLMUINT		uiBufferLength,
		FLMUINT *	puiReturnLength);

	FINLINE RCODE getNext(
		FLMBYTE *	pucBuffer,
		FLMUINT		uiBufferLength,
		FLMUINT *	puiReturnLength)
	{
		// Are we on the last entry or past the last entry?

		if (m_iEntryPos + 1 >= (FLMINT)m_BlockHeader.uiEntryCount)
		{
			m_iEntryPos = (FLMINT) m_BlockHeader.uiEntryCount;
			return RC_SET( NE_FLM_EOF_HIT);
		}

		m_iEntryPos++;
		return( copyCurrentEntry( pucBuffer, uiBufferLength, puiReturnLength));
	}

	RCODE getNextPtr(
		FLMBYTE **	ppBuffer,
		FLMUINT *	puiReturnLength);

	RCODE getPrev(
		FLMBYTE *	pBuffer,
		FLMUINT		uiBufferLength,
		FLMUINT *	puiReturnLength);

	FINLINE FLMUINT64 getPosition( void)
	{
		return( (!m_bPositioned ||
								m_iEntryPos == -1 ||
								m_iEntryPos == (FLMINT)m_BlockHeader.uiEntryCount
								? RS_POSITION_NOT_SET
								: m_ui64BlkEntryPosition + (FLMUINT64)m_iEntryPos));
	}

	RCODE setPosition(
		FLMUINT64	ui64Position );

	RCODE	findMatch(
		FLMBYTE *	pMatchEntry,
		FLMUINT		uiMatchEntryLength,
		FLMBYTE *	pFoundEntry,
		FLMUINT *	puiFoundEntryLength,
		FLMINT *		piCompare);

	void adjustState(
		FLMUINT		uiBlkBufferSize);

	RCODE truncate(
		FLMBYTE *	pszPath);

private:

	RCODE addEntry(
		FLMBYTE *	pucEntry);

	void squeezeSpace( void);

	RCODE sortAndRemoveDups( void);

	void removeEntry(
		FLMBYTE *	pucEntry);

	RCODE quickSort(
		FLMUINT		uiLowerBounds,
		FLMUINT		uiUpperBounds);

	FINLINE RCODE entryCompare(
		FLMBYTE *	pucLeftEntry,
		FLMBYTE *	pucRightEntry,
		FLMINT *		piCompare)
	{
		RCODE			rc;

		if( m_bFixedEntrySize)
		{
			rc = m_pCompare->compare( pucLeftEntry,  m_uiEntrySize,
						pucRightEntry, m_uiEntrySize, piCompare);
		}
		else
		{
			rc = m_pCompare->compare(
						m_pucBlockBuf + ((F_VAR_HEADER *)pucLeftEntry)->ui32Offset,
						((F_VAR_HEADER *)pucLeftEntry)->ui32Length,
						m_pucBlockBuf + ((F_VAR_HEADER *)pucRightEntry)->ui32Offset,
						((F_VAR_HEADER *)pucRightEntry)->ui32Length,
						piCompare);
		}
		if (*piCompare == 0)
		{
			m_bDuplicateFound = TRUE;
		}
		
		return( rc);
	}

	RCODE copyCurrentEntry(
		FLMBYTE *	pBuffer,
		FLMUINT		uiBufferLength,
		FLMUINT *	puiReturnLength);

	RCODE compareEntry(
		FLMBYTE *	pMatchEntry,
		FLMUINT		uiMatchEntryLength,
		FLMUINT		uiEntryPos,
		FLMINT *		piCompare);

	RCODE write();
	
	RCODE read();

	F_BLOCK_HEADER					m_BlockHeader;
	IF_ResultSetCompare *		m_pCompare;
	FLMBYTE *						m_pucBlockBuf;
	FLMBYTE *						m_pucEndPoint;
	F_ResultSetBlk *				m_pNext;
	F_ResultSetBlk *				m_pPrev;
	IF_MultiFileHdl **			m_ppMultiFileHdl;
	FLMUINT64						m_ui64BlkEntryPosition;
	FLMUINT							m_uiLengthRemaining;
	FLMINT							m_iEntryPos;
	FLMUINT							m_uiEntrySize;
	FLMBOOL							m_bEntriesInOrder;
	FLMBOOL							m_bFixedEntrySize;
	FLMBOOL							m_bPositioned;
	FLMBOOL							m_bModifiedEntry;
	FLMBOOL							m_bDuplicateFound;
	FLMBOOL							m_bDropDuplicates;
	
	friend class F_ResultSet;
};

/*****************************************************************************
Desc:
*****************************************************************************/
class F_ResultSet : public IF_ResultSet
{
public:

	F_ResultSet();
	
	F_ResultSet(
		FLMUINT		uiBlkSize);

	virtual ~F_ResultSet();

	RCODE FTKAPI setupResultSet(
		const char *				pszPath,
		IF_ResultSetCompare *	pCompare,
		FLMUINT						uiEntrySize,
		FLMBOOL						bDropDuplicates = TRUE,
		FLMBOOL						bEntriesInOrder = FALSE,
		const char *				pszInputFileName = NULL);	

	FINLINE FLMUINT64 FTKAPI getTotalEntries( void)
	{
		F_ResultSetBlk	*	pBlk = m_pFirstRSBlk;
		FLMUINT64			ui64TotalEntries = 0;

		for( pBlk = m_pFirstRSBlk; pBlk; pBlk = pBlk->m_pNext)
		{
			ui64TotalEntries += pBlk->m_BlockHeader.uiEntryCount;
		}
		
		return( ui64TotalEntries);
	}

	RCODE FTKAPI addEntry(
		const void *	pvEntry,
		FLMUINT			uiEntryLength = 0);

	RCODE FTKAPI finalizeResultSet(
		IF_ResultSetSortStatus *	pSortStatus = NULL,
		FLMUINT64 *						pui64TotalEntries = NULL);

	RCODE FTKAPI getFirst(
		void *			pvEntryBuffer,
		FLMUINT			uiBufferLength = 0,
		FLMUINT *		puiEntryLength = NULL);

	RCODE FTKAPI getNext(
		void *			pvEntryBuffer,
		FLMUINT			uiBufferLength = 0,
		FLMUINT *		puiEntryLength = NULL);

	RCODE FTKAPI getLast(
		void *			pvEntryBuffer,
		FLMUINT			uiBufferLength = 0,
		FLMUINT *		puiEntryLength = NULL);

	RCODE FTKAPI getPrev(
		void *			pvEntryBuffer,
		FLMUINT			uiBufferLength = 0,
		FLMUINT *		puiEntryLength = NULL);

	RCODE FTKAPI getCurrent(
		void *			pvEntryBuffer,
		FLMUINT			uiBufferLength = 0,
		FLMUINT *		puiEntryLength = NULL);

	FINLINE RCODE FTKAPI modifyCurrent(
		const void *	pvEntry,
		FLMUINT			uiEntryLength = 0)
	{
		return( m_pCurRSBlk->modifyEntry( (FLMBYTE *)pvEntry, uiEntryLength));
	}

	FINLINE RCODE FTKAPI findMatch(
		const void *	pvMatchEntry,
		void *			pvFoundEntry)
	{
		return( findMatch( pvMatchEntry, m_uiEntrySize,
								pvFoundEntry, NULL));
	}

	RCODE FTKAPI findMatch(
		const void *	pvMatchEntry,
		FLMUINT			uiMatchEntryLength,
		void *			pvFoundEntry,
		FLMUINT *		puiFoundEntryLength);
		
	FINLINE FLMUINT64 FTKAPI getPosition( void)
	{
		return( (!m_pCurRSBlk
								? RS_POSITION_NOT_SET
								: m_pCurRSBlk->getPosition()));
	}

	RCODE FTKAPI setPosition(
		FLMUINT64		ui64Position);

	RCODE FTKAPI resetResultSet(
		FLMBOOL			bDelete = TRUE);

	RCODE FTKAPI flushToFile( void);

private:

	FINLINE FLMUINT64 numberOfBlockChains( void)
	{
		FLMUINT64			ui64Count = 0;
		F_ResultSetBlk *	pBlk = m_pFirstRSBlk;

		for (; pBlk ; pBlk = pBlk->m_pNext)
		{
			if (pBlk->m_BlockHeader.bFirstBlock)
			{
				ui64Count++;
			}
		}
		
		return( ui64Count);
	}

	RCODE mergeSort();

	RCODE getNextPtr(
		F_ResultSetBlk **			ppCurBlk,
		FLMBYTE *	*				ppBuffer,
		FLMUINT *					puiReturnLength);

	RCODE unionBlkLists(
		F_ResultSetBlk *			pLeftBlk,
		F_ResultSetBlk *			pRightBlk = NULL);

	RCODE copyRemainingItems(
		F_ResultSetBlk *			pCurBlk);

	void closeFile(
		IF_MultiFileHdl **		ppMultiFileHdl,
		FLMBOOL						bDelete = TRUE);

	RCODE openFile(
		IF_MultiFileHdl **		ppMultiFileHdl);

	F_ResultSetBlk * selectMidpoint(
		F_ResultSetBlk *			pLowBlk,
		F_ResultSetBlk *			pHighBlk,
		FLMBOOL						bPickHighIfNeighbors);

	RCODE setupFromFile( void);

	IF_ResultSetCompare *		m_pCompare;
	IF_ResultSetSortStatus *	m_pSortStatus;
	FLMUINT64						m_ui64EstTotalUnits;
	FLMUINT64						m_ui64UnitsDone;
	FLMUINT							m_uiEntrySize;
	FLMUINT64						m_ui64TotalEntries;
	F_ResultSetBlk *				m_pCurRSBlk;
	F_ResultSetBlk *				m_pFirstRSBlk;
	F_ResultSetBlk *				m_pLastRSBlk;
	char								m_szIoDefaultPath[ F_PATH_MAX_SIZE];
	char								m_szIoFilePath1[ F_PATH_MAX_SIZE];
	char								m_szIoFilePath2[ F_PATH_MAX_SIZE];
	IF_MultiFileHdl *				m_pMultiFileHdl1;
	IF_MultiFileHdl *				m_pMultiFileHdl2;
	FLMBYTE *						m_pucBlockBuf1;
	FLMBYTE *						m_pucBlockBuf2;
	FLMBYTE *						m_pucBlockBuf3;
	FLMUINT							m_uiBlockBuf1Len;
	FLMBOOL							m_bFile1Opened;
	FLMBOOL							m_bFile2Opened;
	FLMBOOL							m_bOutput2ndFile;
	FLMBOOL							m_bInitialAdding;
	FLMBOOL							m_bFinalizeCalled;
	FLMBOOL							m_bSetupCalled;
	FLMBOOL							m_bDropDuplicates;
	FLMBOOL							m_bAppAddsInOrder;
	FLMBOOL							m_bEntriesInOrder;
	FLMUINT							m_uiBlkSize;

	friend class F_ResultSetBlk;
};
	
/*****************************************************************************
Desc:
*****************************************************************************/
class F_BTreeResultSet : public IF_BTreeResultSet
{
public:

	F_BTreeResultSet( void)
	{
		m_pBTree = NULL;
	}

	virtual ~F_BTreeResultSet()
	{
		if( m_pBTree)
		{
			m_pBTree->Release();
		}
	}
	
	RCODE setupResultSet(
		IF_ResultSetCompare *	pCompare);

	RCODE FTKAPI addEntry(
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyLength,
		FLMBYTE *	pucEntry,
		FLMUINT		uiEntryLength);

	RCODE FTKAPI modifyEntry(
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyLength,
		FLMBYTE *	pucEntry,
		FLMUINT		uiEntryLength);

	RCODE FTKAPI getCurrent(
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyLength,
		FLMBYTE *	pucEntry,
		FLMUINT		uiEntryLength,
		FLMUINT *	puiReturnLength);

	RCODE FTKAPI getNext(
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyBufLen,
		FLMUINT *	puiKeylen,
		FLMBYTE *	pucBuffer,
		FLMUINT		uiBufferLength,
		FLMUINT *	puiReturnLength);

	RCODE FTKAPI getPrev(
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyBufLen,
		FLMUINT *	puiKeylen,
		FLMBYTE *	pucBuffer,
		FLMUINT		uiBufferLength,
		FLMUINT *	puiReturnLength);

	RCODE FTKAPI getFirst(
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyBufLen,
		FLMUINT *	puiKeylen,
		FLMBYTE *	pucBuffer,
		FLMUINT		uiBufferLength,
		FLMUINT *	puiReturnLength);

	RCODE FTKAPI getLast(
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyBufLen,
		FLMUINT *	puiKeylen,
		FLMBYTE *	pucBuffer,
		FLMUINT		uiBufferLength,
		FLMUINT *	puiReturnLength);

	RCODE FTKAPI findEntry(
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyLen,
		FLMBYTE *	pucBuffer,
		FLMUINT		uiBufferLength,
		FLMUINT *	puiReturnLength);

	RCODE FTKAPI deleteEntry(
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyLength);

private:

	IF_BTree *						m_pBTree;
};

/*****************************************************************************
Desc:
*****************************************************************************/
F_ResultSet::F_ResultSet()
{
	m_pCompare = NULL;
	m_pSortStatus = NULL;
	m_ui64EstTotalUnits = 0;
	m_ui64UnitsDone = 0;

	m_uiEntrySize = 0;
	m_ui64TotalEntries = 0;
	m_pCurRSBlk = NULL;
	m_pFirstRSBlk = NULL;
	m_pLastRSBlk = NULL;

	f_memset( &m_szIoDefaultPath[0], 0, F_PATH_MAX_SIZE);

	m_pucBlockBuf1 = NULL;
	m_pucBlockBuf2 = NULL;
	m_pucBlockBuf3 = NULL;
	m_uiBlockBuf1Len = 0;
	m_bFile1Opened = FALSE;
	m_bFile2Opened = FALSE;
	m_pMultiFileHdl1 = NULL;
	m_pMultiFileHdl2 = NULL;
	m_bOutput2ndFile = FALSE;
	m_bInitialAdding = TRUE;
	m_bFinalizeCalled = FALSE;
	m_bSetupCalled = FALSE;
	m_uiBlkSize = RS_BLOCK_SIZE;
}

/*****************************************************************************
Desc:
*****************************************************************************/
F_ResultSet::F_ResultSet(
	FLMUINT			uiBlkSize)
{
	m_pCompare = NULL;
	m_pSortStatus = NULL;
	m_ui64EstTotalUnits = 0;
	m_ui64UnitsDone = 0;

	m_uiEntrySize = 0;
	m_ui64TotalEntries = 0;
	m_pCurRSBlk = NULL;
	m_pFirstRSBlk = NULL;
	m_pLastRSBlk = NULL;

	f_memset( &m_szIoDefaultPath[0], 0, F_PATH_MAX_SIZE);

	m_pucBlockBuf1 = NULL;
	m_pucBlockBuf2 = NULL;
	m_pucBlockBuf3 = NULL;
	m_uiBlockBuf1Len = 0;
	m_bFile1Opened = FALSE;
	m_bFile2Opened = FALSE;
	m_pMultiFileHdl1 = NULL;
	m_pMultiFileHdl2 = NULL;
	m_bOutput2ndFile = FALSE;
	m_bInitialAdding = TRUE;
	m_bFinalizeCalled = FALSE;
	m_bSetupCalled = FALSE;
	m_uiBlkSize = uiBlkSize;
}

/*****************************************************************************
Desc:
*****************************************************************************/
F_ResultSet::~F_ResultSet()
{
	F_ResultSetBlk *	pCurRSBlk;
	F_ResultSetBlk *	pNextRSBlk;

	// Free up the result set block chain.

	for( pCurRSBlk = m_pFirstRSBlk; pCurRSBlk; pCurRSBlk = pNextRSBlk)
	{
		FLMUINT		uiCount;

		pNextRSBlk = pCurRSBlk->m_pNext;
		uiCount = pCurRSBlk->Release();
		f_assert( !uiCount);
	}

	// Set list to NULL for debugging in memory.

	m_pFirstRSBlk = NULL;
	m_pLastRSBlk = NULL;
	m_pCurRSBlk = NULL;

	// Free up all of the block buffers in the list.

	f_free( &m_pucBlockBuf1);
	f_free( &m_pucBlockBuf2);
	f_free( &m_pucBlockBuf3);

	// Close all opened files

	closeFile( &m_pMultiFileHdl1);
	closeFile( &m_pMultiFileHdl2);

	if( m_pCompare)
	{
		m_pCompare->Release();
	}

	if( m_pSortStatus)
	{
		m_pSortStatus->Release();
	}
}

/*****************************************************************************
Desc:	Reset the result set so it can be reused.
*****************************************************************************/
RCODE FTKAPI F_ResultSet::resetResultSet(
	FLMBOOL		bDelete)
{
	RCODE					rc = NE_FLM_OK;
	F_ResultSetBlk *	pCurRSBlk;
	F_ResultSetBlk *	pNextRSBlk;

	// Free up the result set block chain - except for the first one.

	for( pCurRSBlk = m_pFirstRSBlk; pCurRSBlk; pCurRSBlk = pNextRSBlk)
	{
		FLMUINT		uiCount;

		pNextRSBlk = pCurRSBlk->m_pNext;
		if( pCurRSBlk != m_pFirstRSBlk)
		{
			uiCount = pCurRSBlk->Release();
			f_assert( !uiCount);
		}
	}

	// Free up all of the block buffers in the list, except for the first one.

	f_free( &m_pucBlockBuf2);
	f_free( &m_pucBlockBuf3);

	if( !m_pucBlockBuf1 || m_uiBlockBuf1Len != m_uiBlkSize)
	{
		if( m_pucBlockBuf1)
		{
			f_free( &m_pucBlockBuf1);
		}

		if( RC_BAD( rc = f_alloc( m_uiBlkSize, &m_pucBlockBuf1)))
		{
			goto Exit;
		}

		m_uiBlockBuf1Len = m_uiBlkSize;
	}

	// Close all opened files

	closeFile( &m_pMultiFileHdl1, bDelete );
	closeFile( &m_pMultiFileHdl2 );
	m_bFile1Opened = m_bFile2Opened = FALSE;
	m_pMultiFileHdl1 = m_pMultiFileHdl2 = NULL;

	// Reset some other variables

	if( m_pSortStatus)
	{
		m_pSortStatus->Release();
		m_pSortStatus = NULL;
	}

	m_ui64EstTotalUnits = 0;
	m_ui64UnitsDone = 0;
	m_ui64TotalEntries = 0;
	m_bOutput2ndFile = FALSE;
	m_bInitialAdding = TRUE;
	m_bEntriesInOrder = m_bAppAddsInOrder;
	m_bFinalizeCalled = FALSE;

	// If we don't have a block, allocate it.  Otherwise
	// reset the one we have left.

	if( !m_pFirstRSBlk)
	{
		if( (m_pFirstRSBlk = f_new F_ResultSetBlk) == NULL)
		{
			rc = RC_SET( NE_FLM_MEM);
			goto Exit;
		}
	}
	else
	{
		m_pFirstRSBlk->reset();
	}

	m_pLastRSBlk = m_pCurRSBlk = m_pFirstRSBlk;
	(void)m_pFirstRSBlk->setup( &m_pMultiFileHdl1, m_pCompare,
		m_uiEntrySize, TRUE, m_bDropDuplicates,
		m_bEntriesInOrder);
	(void) m_pFirstRSBlk->setBuffer( m_pucBlockBuf1, m_uiBlockBuf1Len);

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Setup the result set with all of the needed input values.
		This method must only be called once.
*****************************************************************************/
RCODE FTKAPI F_ResultSet::setupResultSet(
	const char *				pszDirPath,
	IF_ResultSetCompare *	pCompare,
	FLMUINT						uiEntrySize,
	FLMBOOL						bDropDuplicates,
	FLMBOOL						bEntriesInOrder,
	const char *				pszInputFileName)
{
	RCODE			rc = NE_FLM_OK;
	FLMBOOL		bNewBlock = FALSE;
	FLMBOOL		bNewBuffer = FALSE;

	f_assert( !m_bSetupCalled );
	f_assert( uiEntrySize <= RS_MAX_FIXED_ENTRY_SIZE);

	// Perform all of the allocations first.

	m_pFirstRSBlk = m_pLastRSBlk = m_pCurRSBlk = f_new F_ResultSetBlk;

	// Allocation Error?

	if( !m_pCurRSBlk)
	{
		rc = RC_SET( NE_FLM_MEM );
		goto Exit;
	}

	bNewBlock = TRUE;
	m_pCurRSBlk->setup( &m_pMultiFileHdl1, pCompare,
			uiEntrySize, TRUE, bDropDuplicates, bEntriesInOrder);

	// Allocate only the first buffer - other buffers only used in merge.

	if( RC_BAD( rc = f_alloc( m_uiBlkSize, &m_pucBlockBuf1)))
	{
		goto Exit;
	}

	m_uiBlockBuf1Len = m_uiBlkSize;
	bNewBuffer = TRUE;
	(void) m_pCurRSBlk->setBuffer( m_pucBlockBuf1, m_uiBlockBuf1Len);

	// Set the input variables.

	if( pszDirPath)
	{
		f_strcpy( m_szIoDefaultPath, pszDirPath);
	}

	if( m_pCompare)
	{
		m_pCompare->Release();
	}

	if( (m_pCompare = pCompare) != NULL)
	{
		m_pCompare->AddRef();
	}

	m_uiEntrySize = uiEntrySize;
	m_bDropDuplicates = bDropDuplicates;
	m_bEntriesInOrder = m_bAppAddsInOrder = bEntriesInOrder;
	
	// If a filename was passed in, then we will try to open it and read whatever
	// data it holds into the result set.  If the file does not exist, it will not
	// be created at this time.

	if( pszInputFileName)
	{
		f_strcpy( m_szIoFilePath1, m_szIoDefaultPath);

		if( RC_BAD( rc = f_getFileSysPtr()->pathAppend( 
			m_szIoFilePath1, pszInputFileName)))
		{
			goto Exit;
		}

		f_strcat( m_szIoFilePath1, "." FRSET_FILENAME_EXTENSION);

		if( RC_BAD( rc = setupFromFile()))
		{
			goto Exit;
		}
	}

Exit:

	// Free allocations on any error

	if( RC_BAD(rc))
	{
		if( bNewBlock)
		{
			if( m_pCurRSBlk)
			{
				m_pCurRSBlk->Release();
				m_pFirstRSBlk = m_pLastRSBlk = m_pCurRSBlk = NULL;
			}
		}

		if( bNewBuffer)
		{
			f_free( &m_pucBlockBuf1);
			m_uiBlockBuf1Len = 0;
		}
	}
	else
	{
		m_bSetupCalled = TRUE;
	}

	return( rc);
}

/*****************************************************************************
Desc:	Attempt to establish the result set from an existing file.
*****************************************************************************/
RCODE F_ResultSet::setupFromFile( void)
{
	RCODE						rc = NE_FLM_OK;
	F_ResultSetBlk *		pNextRSBlk;
	FLMUINT					uiOffset;
	FLMUINT					uiBytesRead;
	F_BLOCK_HEADER			BlkHdr;

	f_assert( !m_bSetupCalled);
	
	if( RC_BAD( rc = FlmAllocMultiFileHdl( &m_pMultiFileHdl1)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pMultiFileHdl1->openFile( m_szIoFilePath1)))
	{
		if( rc == NE_FLM_IO_PATH_NOT_FOUND)
		{
			if( RC_BAD( rc = m_pMultiFileHdl1->createFile( m_szIoFilePath1)))
			{
				rc = NE_FLM_OK;
				m_pMultiFileHdl1->Release();
				m_pMultiFileHdl1 = NULL;
				goto Exit;
			}
		}
		else
		{
			rc = NE_FLM_OK;
			m_pMultiFileHdl1->Release();
			m_pMultiFileHdl1 = NULL;
			goto Exit;
		}
	}

	m_bFile1Opened = TRUE;

	// Release the current set of blocks.
	
	while( m_pFirstRSBlk)
	{
		m_pCurRSBlk = m_pFirstRSBlk;
		m_pFirstRSBlk = m_pFirstRSBlk->m_pNext;
		m_pCurRSBlk->Release();
	}

	m_pFirstRSBlk = m_pLastRSBlk = m_pCurRSBlk = NULL;

	// Allocate the buffer that we will use to read the data in.

	if( !m_pucBlockBuf1)
	{
		if( RC_BAD( rc = f_calloc( m_uiBlkSize, &m_pucBlockBuf1)))
		{
			goto Exit;
		}
		m_uiBlockBuf1Len = m_uiBlkSize;
	}
	else
	{
		f_memset( m_pucBlockBuf1, 0, m_uiBlkSize);
	}

	// Now read every block in the file and create a F_ResultSetBlk chain.

	f_memset( (void *)&BlkHdr, 0, sizeof(	F_BLOCK_HEADER));

	for( uiOffset = 0;;)
	{
		// Read the block header

		if( RC_BAD( rc = m_pMultiFileHdl1->read( 
			BlkHdr.ui64FilePos + BlkHdr.uiBlockSize + uiOffset,
			sizeof( F_BLOCK_HEADER), &BlkHdr, &uiBytesRead)))
		{
			if( rc == NE_FLM_EOF_HIT || rc == NE_FLM_IO_END_OF_FILE)
			{
				rc = NE_FLM_OK;
				break;
			}

			goto Exit;
		}

		// Put the previous block out of fous.

		if( m_pCurRSBlk)
		{
			if( RC_BAD( rc = m_pCurRSBlk->setBuffer( NULL, m_uiBlkSize)))
			{
				goto Exit;
			}
		}

		// Allocate a new RSBlk and link into the result block list.
		
		if( (pNextRSBlk = f_new F_ResultSetBlk) == NULL)
		{
			rc = RC_SET( NE_FLM_MEM );
			goto Exit;
		}

		if( !m_pFirstRSBlk)
		{
			m_pFirstRSBlk = m_pLastRSBlk = m_pCurRSBlk = pNextRSBlk;
		}
		else
		{
			m_pCurRSBlk->m_pNext = pNextRSBlk;
			pNextRSBlk->m_pPrev = m_pCurRSBlk;
			m_pLastRSBlk = m_pCurRSBlk = pNextRSBlk;
		}

		m_pCurRSBlk->setup( &m_pMultiFileHdl1, m_pCompare, m_uiEntrySize,
			BlkHdr.bFirstBlock, m_bDropDuplicates, !m_bInitialAdding);

		f_memcpy( (void *)&m_pCurRSBlk->m_BlockHeader, 
			(void *)&BlkHdr, sizeof(F_BLOCK_HEADER));

		// Process the block...

		if( RC_BAD( rc = m_pCurRSBlk->setBuffer( m_pucBlockBuf1, m_uiBlkSize)))
		{
			goto Exit;
		}

		m_pCurRSBlk->adjustState( m_uiBlkSize);
		uiOffset = sizeof(F_BLOCK_HEADER);
	}

	// If the file is empty or just created, we won't have a RS Block yet.

	if( !m_pCurRSBlk)
	{
		// Allocate a new RSBlk

		if( (pNextRSBlk = f_new F_ResultSetBlk) == NULL)
		{
			rc = RC_SET( NE_FLM_MEM );
			goto Exit;
		}

		if( !m_pFirstRSBlk)
		{
			m_pFirstRSBlk = m_pLastRSBlk = m_pCurRSBlk = pNextRSBlk;
		}
		else
		{
			m_pCurRSBlk->m_pNext = pNextRSBlk;
			pNextRSBlk->m_pPrev = m_pCurRSBlk;
			m_pLastRSBlk = m_pCurRSBlk = pNextRSBlk;
		}

		m_pCurRSBlk->setup(  &m_pMultiFileHdl1, m_pCompare,
				m_uiEntrySize, m_bInitialAdding, m_bDropDuplicates,
				!m_bInitialAdding );

		if( RC_BAD( rc = m_pCurRSBlk->setBuffer( m_pucBlockBuf1, m_uiBlkSize)))
		{
			goto Exit;
		}
	}
	else
	{
		// Resize the file.

		if( RC_BAD(rc = m_pCurRSBlk->truncate( (FLMBYTE *)m_szIoFilePath1)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Write the current block and close the file.  Call this function befor
		calling resetResultSet so that it can be reused.
*****************************************************************************/
RCODE FTKAPI F_ResultSet::flushToFile()
{
	RCODE		rc = NE_FLM_OK;

	f_assert( m_bFile1Opened);

	// Flush to disk what ever we have.

	if( RC_BAD( rc = m_pCurRSBlk->flush( m_bInitialAdding, TRUE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pCurRSBlk->setBuffer( NULL, m_uiBlkSize)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:		Interface to add a variable length entry to the result set.
Notes:	Public method used by application and by the internal sort
			and merge steps during finalize.  The user must never add an
			entry that is larger than the block size.
*****************************************************************************/
RCODE FTKAPI F_ResultSet::addEntry(
	const void *	pvEntry,
	FLMUINT			uiEntryLength)				// If zero then entry is fixed length
{
	RCODE		rc = NE_FLM_OK;

	f_assert( m_bSetupCalled);
	f_assert( !m_bFinalizeCalled);

	rc = m_pCurRSBlk->addEntry( (FLMBYTE *)pvEntry, uiEntryLength);

	// See if current block is full

	if( rc == NE_FLM_EOF_HIT)
	{
		F_ResultSetBlk *			pNextRSBlk;
		IF_MultiFileHdl **		ppMultiFileHdl;

		if( m_bInitialAdding && !m_bFile1Opened)
		{
			// Need to create and open the output file?
			// In a merge we may be working on the 2nd file and NOT the 1st.
			// There just isn't a better place to open the 1st file.

			if( RC_BAD(rc = openFile( &m_pMultiFileHdl1)))
			{
				goto Exit;
			}
		}

		ppMultiFileHdl = (m_bOutput2ndFile) 
									? &m_pMultiFileHdl2 
									: &m_pMultiFileHdl1;

		// Always flush to disk (TRUE) from here.

		if( RC_BAD( rc = m_pCurRSBlk->flush( m_bInitialAdding, TRUE)))
		{
			goto Exit;
		}

		(void) m_pCurRSBlk->setBuffer( NULL, m_uiBlkSize);

		// Adding the current block is complete so allocate a new
		// block object and link it into the list.
		// We must continue to use this same block buffer.

		// Allocate a new RSBlk and link into the result block list.

		if( (pNextRSBlk = f_new F_ResultSetBlk) == NULL)
		{
			rc = RC_SET( NE_FLM_MEM );
			goto Exit;
		}

		m_pCurRSBlk->m_pNext = pNextRSBlk;
		pNextRSBlk->m_pPrev = m_pCurRSBlk;
		m_pLastRSBlk = m_pCurRSBlk = pNextRSBlk;
		m_pCurRSBlk->setup(  ppMultiFileHdl, m_pCompare,
				m_uiEntrySize, m_bInitialAdding, m_bDropDuplicates,
				!m_bInitialAdding );

		// Reset all of the buffer pointers and values.

		(void)m_pCurRSBlk->setBuffer( m_pucBlockBuf1, m_uiBlockBuf1Len);

		// Make the callback only during the merge phase.

		if( !m_bInitialAdding && m_pSortStatus)
		{
			if( m_ui64EstTotalUnits <= ++m_ui64UnitsDone )
			{
				m_ui64EstTotalUnits = m_ui64UnitsDone;
			}

			if( RC_BAD( rc = m_pSortStatus->reportSortStatus( m_ui64EstTotalUnits,
									m_ui64UnitsDone)))
			{
				goto Exit;
			}
		}

		// Add the entry again.  This call should never fail because of space.
		// If it does fail then the entry is larger than the buffer size.

		if( RC_BAD( rc = m_pCurRSBlk->addEntry( 
			(FLMBYTE *)pvEntry, uiEntryLength)))
		{
			if( rc == NE_FLM_EOF_HIT)
			{
				rc = RC_SET_AND_ASSERT( NE_FLM_FAILURE);
			}

			goto Exit;
		}
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Done adding entries.  Sort all of the entries and perform a merge.
*****************************************************************************/
RCODE FTKAPI F_ResultSet::finalizeResultSet(
	IF_ResultSetSortStatus *	pSortStatus,
	FLMUINT64 *						pui64TotalEntries)
{
	RCODE				rc = NE_FLM_OK;
	FLMBOOL			bMergeSort;

	// Avoid being called more than once.

	f_assert( !m_bFinalizeCalled);
	f_assert( m_bSetupCalled);
	f_assert( !m_pSortStatus);
	
	if ((m_pSortStatus = pSortStatus) != NULL)
	{
		m_pSortStatus->AddRef();
	}

	// Not a bug - but for future possibilities just check
	// if there is more than one block and if so then
	// the while() loop merge sort needs to be called.

	bMergeSort = (m_pFirstRSBlk != m_pLastRSBlk) ? TRUE : FALSE;

	// Force the write to disk if bMergeSort is TRUE.

	if( RC_BAD(rc = m_pCurRSBlk->finalize( bMergeSort)))
	{
		goto Exit;
	}

	m_bInitialAdding = FALSE;

	// If the entries are in order fixup the block chain and we are done.

	if( m_bEntriesInOrder)
	{
		F_ResultSetBlk	*	pBlk;

		if( numberOfBlockChains() > 1)
		{
			// Entries already in order - need to fixup the blocks.

			for( pBlk = m_pFirstRSBlk; pBlk; pBlk = pBlk->m_pNext)
			{
				pBlk->m_BlockHeader.bFirstBlock = FALSE;
				pBlk->m_BlockHeader.bLastBlock = FALSE;
			}

			m_pFirstRSBlk->m_BlockHeader.bFirstBlock = TRUE;
			m_pLastRSBlk->m_BlockHeader.bLastBlock = TRUE;
			m_pCurRSBlk = NULL;
		}

		goto Exit;
	}

	// Compute total number of blocks.

	if( m_pSortStatus)
	{
		// Estimate total number of unit blocks to be written.

		FLMUINT64	ui64Units = numberOfBlockChains();
		FLMUINT64	ui64Loops;

		m_ui64EstTotalUnits = 0;
		for( ui64Loops = ui64Units; ui64Loops > 1;
			  ui64Loops = (ui64Loops + 1) / 2 )
		{
			m_ui64EstTotalUnits += ui64Units;
		}
	}

	// Do the merge sort.
	// Keep looping until we have only one block in the result set list.

	while( numberOfBlockChains() > 1)
	{
		// Allocate two more buffers.  Merge will open the 2nd file.
		// Exit will free these allocations and close one of the files.

		// Are the 2nd and 3rd buffers allocated?

		if( !m_pucBlockBuf2)
		{
			if( RC_BAD( rc = f_alloc( m_uiBlkSize, &m_pucBlockBuf2)))
			{
				goto Exit;
			}
		}

		if( !m_pucBlockBuf3)
		{
			if( RC_BAD( rc = f_alloc( m_uiBlkSize, &m_pucBlockBuf3)))
			{
				goto Exit;
			}
		}

		// Swap which file is selected as the output file.

		m_bOutput2ndFile = m_bOutput2ndFile ? FALSE : TRUE;

		// Here is the magical call that does all of the work!

		if( RC_BAD( rc = mergeSort()))
		{
			goto Exit;
		}
	}

Exit:

	// If we did a merge sort of multiple blocks then
	// free the first and second buffers and close one of the files.

	if( RC_BAD(rc))
	{
		f_free( &m_pucBlockBuf1);
		m_uiBlockBuf1Len = 0;
	}

	f_free( &m_pucBlockBuf2);
	f_free( &m_pucBlockBuf3);

	// Close the non-output opened file.  Close both on error.
	// If m_bFile2Opened then we did a merge - close one file

	if( m_bFile2Opened || RC_BAD( rc))
	{
		if( m_bOutput2ndFile || RC_BAD( rc))
		{
			if( m_bFile1Opened)
			{
				m_pMultiFileHdl1->closeFile( TRUE);
				m_bFile1Opened = FALSE;
			}

			if( m_pMultiFileHdl1)
			{
				m_pMultiFileHdl1->Release();
				m_pMultiFileHdl1 = NULL;
			}
		}

		if( !m_bOutput2ndFile || RC_BAD( rc))
		{
			if( m_bFile2Opened)
			{
				m_pMultiFileHdl2->closeFile( TRUE);
				m_bFile2Opened = FALSE;
			}

			if( m_pMultiFileHdl2)
			{
				m_pMultiFileHdl2->Release();
				m_pMultiFileHdl2 = NULL;
			}
		}
	}

	if( RC_OK(rc))
	{
		FLMUINT64			ui64Pos;
		F_ResultSetBlk *	pRSBlk;

		m_bFinalizeCalled = TRUE;
		m_bEntriesInOrder = TRUE;

		m_ui64TotalEntries = getTotalEntries();

		// Set the return value for total entries.

		if( pui64TotalEntries)
		{
			*pui64TotalEntries = m_ui64TotalEntries;
		}

		if( !m_ui64TotalEntries)
		{
			if( m_pCurRSBlk)
			{
				m_pCurRSBlk->Release();
			}

			m_pCurRSBlk = NULL;
			m_pFirstRSBlk = NULL;
			m_pLastRSBlk = NULL;
			f_free( &m_pucBlockBuf1);
			m_uiBlockBuf1Len = 0;
		}

		// Set the ui64BlkEntryPosition values in each block.

		for( ui64Pos = 0, pRSBlk = m_pFirstRSBlk;
				pRSBlk;
				pRSBlk = pRSBlk->m_pNext)
		{
			pRSBlk->m_ui64BlkEntryPosition = ui64Pos;
			ui64Pos += pRSBlk->m_BlockHeader.uiEntryCount;
		}

		// Resize the buffer to save space if only one block & in memory.

		if( m_pFirstRSBlk == m_pLastRSBlk && m_pCurRSBlk)
		{
			FLMBYTE *	pucNewBlk;
			FLMUINT		uiLen = m_pCurRSBlk->bytesUsedInBuffer();

			if( uiLen != m_uiBlockBuf1Len)
			{
				if( RC_OK( rc = f_alloc( uiLen, &pucNewBlk)))
				{
					f_memcpy( pucNewBlk, m_pucBlockBuf1, uiLen);
					f_free( &m_pucBlockBuf1);
					m_pucBlockBuf1 = pucNewBlk;
					m_uiBlockBuf1Len = uiLen;
				}
			}

			// Need to always do the SetBuffer, because it causes the
			// result set to get positioned.

			if( RC_OK( rc))
			{
				rc = m_pCurRSBlk->setBuffer( m_pucBlockBuf1, uiLen);
			}
		}
	}

	if (m_pSortStatus)
	{
		m_pSortStatus->Release();
		m_pSortStatus = NULL;
	}
	
	return( rc);
}

/*****************************************************************************
Desc:	Perform a Merge Sort on a list of result set blocks.  This new
		algorithm uses two files for the sort.  The end result may
		be one of the two files.  At the end of the sort all old result set
		block objects will be freed and only one result set block object
		will be left.  This RSBlk object will be used for reading the
		entries.  At this point there are at least 'N' result set block
		objects that will be merged into ('N'/2) block objects.
*****************************************************************************/
RCODE F_ResultSet::mergeSort( void)
{
	RCODE							rc = NE_FLM_OK;
	F_ResultSetBlk *			pBlkList = NULL;
	F_ResultSetBlk *			pTempBlk;
	F_ResultSetBlk *			pLeftBlk;
	F_ResultSetBlk *			pRightBlk;
	IF_MultiFileHdl **		ppMultiFileHdl;

	// Set output file and truncate it.

	rc = (m_bOutput2ndFile)
			? openFile( &m_pMultiFileHdl2)
			: openFile( &m_pMultiFileHdl1);

	if( RC_BAD( rc))
	{
		RC_UNEXPECTED_ASSERT( rc);
		goto Exit;
	}

	ppMultiFileHdl = ( m_bOutput2ndFile) 
								? &m_pMultiFileHdl2 
								: &m_pMultiFileHdl1;

	// Get the list to the RS blocks

	pBlkList = m_pFirstRSBlk;

	// Form an empty list to build.

	m_pFirstRSBlk = m_pLastRSBlk = m_pCurRSBlk = NULL;

	// Read and UNION one or two blocks at a time getting rid of duplicates.
	// Reading the entries when performing a union of only one block
	// is a lot of work for nothing - but it simplifies the code.

	pTempBlk = pBlkList;
	while (pTempBlk)
	{
		pLeftBlk = pTempBlk;
		pRightBlk = pTempBlk->m_pNext;

		while( pRightBlk && !pRightBlk->m_BlockHeader.bFirstBlock)
		{
			pRightBlk = pRightBlk->m_pNext;
		}

		// Allocate a new result set block list and link into the new list.

		if( (m_pCurRSBlk = f_new F_ResultSetBlk) == NULL)
		{
			rc = RC_SET( NE_FLM_MEM);
			goto Exit;
		}

		if( !m_pLastRSBlk)
		{
			// First time

			m_pFirstRSBlk = m_pLastRSBlk = m_pCurRSBlk;
		}
		else
		{
			m_pLastRSBlk->m_pNext = m_pCurRSBlk;
			m_pCurRSBlk->m_pPrev = m_pLastRSBlk;
			m_pLastRSBlk = m_pCurRSBlk;
		}

		m_pCurRSBlk->setup(  ppMultiFileHdl, m_pCompare,
				m_uiEntrySize, TRUE, m_bDropDuplicates, TRUE);

		// Output to block buffer 1

		(void)m_pCurRSBlk->setBuffer( m_pucBlockBuf1, m_uiBlkSize );
		if( RC_BAD( rc = pLeftBlk->setBuffer( m_pucBlockBuf2, m_uiBlkSize)))
		{
			goto Exit;
		}

		if( pRightBlk)
		{
			if( RC_BAD( rc = pRightBlk->setBuffer( m_pucBlockBuf3, m_uiBlkSize)))
			{
				goto Exit;
			}
		}

		// pRightBlk may be NULL - will move left block to output.
		// Output leftBlk and rightBlk to the output block (m_pCurRSBlk)

		if( RC_BAD(rc = unionBlkLists( pLeftBlk, pRightBlk)))
		{
			goto Exit;
		}

		// Setup for the next loop.

		pTempBlk = pRightBlk ? pRightBlk->m_pNext : NULL;
		while( pTempBlk && !pTempBlk->m_BlockHeader.bFirstBlock)
		{
			pTempBlk = pTempBlk->m_pNext;
		}
	}

Exit:

	// Free the working block list.

	pTempBlk = pBlkList;
	while( pTempBlk)
	{
		FLMUINT	uiTemp;

		pRightBlk = pTempBlk->m_pNext;
		uiTemp = pTempBlk->Release();
		f_assert( uiTemp == 0);
		pTempBlk = pRightBlk;
	}

	return( rc);
}

/*****************************************************************************
Desc:	Return the Current entry reference in the result set.
*****************************************************************************/
RCODE FTKAPI F_ResultSet::getCurrent(
	void *		pvBuffer,
	FLMUINT		uiBufferLength,
	FLMUINT *	puiReturnLength)
{
	RCODE		rc = NE_FLM_OK;

	f_assert( m_bFinalizeCalled);

	if( !m_pCurRSBlk)
	{
		rc = RC_SET( NE_FLM_NOT_FOUND);
	}
	else
	{
		rc = m_pCurRSBlk->getCurrent( (FLMBYTE *)pvBuffer, uiBufferLength,
										puiReturnLength );
	}

	return( rc);
}

/*****************************************************************************
Desc:	Return the next reference in the result set.  If the result set
		is not positioned then the first entry will be returned.
*****************************************************************************/
RCODE FTKAPI F_ResultSet::getNext(
	void *			pvBuffer,
	FLMUINT			uiBufferLength,
	FLMUINT *		puiReturnLength)
{
	RCODE		rc = NE_FLM_OK;

	f_assert( m_bFinalizeCalled);

	// Make sure we are positioned to a block.

	if( !m_pCurRSBlk)
	{
		m_pCurRSBlk = m_pFirstRSBlk;
		if( !m_pCurRSBlk)
		{
			rc = RC_SET( NE_FLM_EOF_HIT);
			goto Exit;
		}

		if( RC_BAD( rc = m_pCurRSBlk->setBuffer( m_pucBlockBuf1, m_uiBlkSize)))
		{
			goto Exit;
		}
	}

	rc = m_pCurRSBlk->getNext( (FLMBYTE *)pvBuffer, uiBufferLength,
										puiReturnLength );

	// Position to the next block?

	if( rc == NE_FLM_EOF_HIT)
	{
		if( m_pCurRSBlk->m_pNext)
		{
			m_pCurRSBlk->setBuffer( NULL);
			m_pCurRSBlk = m_pCurRSBlk->m_pNext;

			if( RC_BAD( rc = m_pCurRSBlk->setBuffer( m_pucBlockBuf1, m_uiBlkSize)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = m_pCurRSBlk->getNext( 
				(FLMBYTE *)pvBuffer, uiBufferLength, puiReturnLength)))
			{
				goto Exit;
			}
		}
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Return the previous reference in the result set.  If the result set
		is not positioned then the last entry will be returned.
*****************************************************************************/
RCODE FTKAPI F_ResultSet::getPrev(
	void *			pvBuffer,
	FLMUINT			uiBufferLength,
	FLMUINT *		puiReturnLength)
{
	RCODE				rc;

	f_assert( m_bFinalizeCalled);

	// Make sure we are positioned to a block.

	if( !m_pCurRSBlk)
	{
		if( (m_pCurRSBlk = m_pLastRSBlk) == NULL)
		{
			rc = RC_SET( NE_FLM_BOF_HIT);
			goto Exit;
		}

		if( RC_BAD( rc = m_pCurRSBlk->setBuffer( m_pucBlockBuf1, m_uiBlkSize)))
		{
			goto Exit;
		}
	}

	rc = m_pCurRSBlk->getPrev( (FLMBYTE *)pvBuffer, uiBufferLength,
										puiReturnLength );

	// Position to the previous block?

	if( rc == NE_FLM_BOF_HIT)
	{
		if( m_pCurRSBlk->m_pPrev)
		{
			m_pCurRSBlk->setBuffer( NULL);
			m_pCurRSBlk = m_pCurRSBlk->m_pPrev;
			if( RC_BAD( rc = m_pCurRSBlk->setBuffer( m_pucBlockBuf1, m_uiBlkSize)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = m_pCurRSBlk->getPrev( (FLMBYTE *)pvBuffer,
											uiBufferLength,
											puiReturnLength)))
			{
				goto Exit;
			}
		}
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Return the first reference in the result set.
*****************************************************************************/
RCODE FTKAPI F_ResultSet::getFirst(
	void *			pvBuffer,
	FLMUINT			uiBufferLength,
	FLMUINT *		puiReturnLength)
{
	RCODE				rc;

	f_assert( m_bFinalizeCalled);

	if( m_pCurRSBlk != m_pFirstRSBlk)
	{
		if( m_pCurRSBlk)
		{
			m_pCurRSBlk->setBuffer( NULL);
		}

		m_pCurRSBlk = m_pFirstRSBlk;

		if( RC_BAD( rc = m_pCurRSBlk->setBuffer( m_pucBlockBuf1, m_uiBlkSize)))
		{
			goto Exit;
		}
	}
	else if( !m_pCurRSBlk)
	{
		rc = RC_SET( NE_FLM_NOT_FOUND);
		goto Exit;
	}

	if( RC_BAD( rc = m_pCurRSBlk->getNext( (FLMBYTE *)pvBuffer,
		uiBufferLength, puiReturnLength)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Return the last reference in the result set.
*****************************************************************************/
RCODE FTKAPI F_ResultSet::getLast(
	void *			pvBuffer,
	FLMUINT			uiBufferLength,
	FLMUINT *		puiReturnLength)
{
	RCODE				rc = NE_FLM_OK;

	f_assert( m_bFinalizeCalled);

	if( m_pCurRSBlk != m_pLastRSBlk)
	{
		if( m_pCurRSBlk)
		{
			m_pCurRSBlk->setBuffer( NULL);
		}

		m_pCurRSBlk = m_pLastRSBlk;

		if( RC_BAD( rc = m_pCurRSBlk->setBuffer( m_pucBlockBuf1, m_uiBlkSize)))
		{
			goto Exit;
		}
	}
	else if( !m_pCurRSBlk)
	{
		rc = RC_SET( NE_FLM_EOF_HIT);
		goto Exit;
	}

	if( RC_BAD( rc = m_pCurRSBlk->getPrev( (FLMBYTE *) pvBuffer,
		uiBufferLength, puiReturnLength)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Find the matching entry in the result set using the compare routine.
		This does a binary search on the list of blocks.
*****************************************************************************/
RCODE FTKAPI F_ResultSet::findMatch(
	const void *	pvMatchEntry,			// Entry to match
	FLMUINT			uiMatchEntryLength,	// Variable length of above entry
	void *			pvFoundEntry,			// (out) Entry to return
	FLMUINT *		puiFoundEntryLength)	// (out) Length of entry returned
{
	RCODE					rc = NE_FLM_OK;
	FLMINT				iBlkCompare;		// 0 if key is/would be in block.
	F_ResultSetBlk *	pLowBlk;				// Used for locating block.
	F_ResultSetBlk *	pHighBlk;			// Low and High are exclusive.

	f_assert( m_bFinalizeCalled);

	// If not positioned anywhere, position to the midpoint.
	// Otherwise, start on the current block we are on.

	if( !m_pCurRSBlk)
	{
		// m_pFirstRSBlk will be NULL if no entries.

		if( !m_pFirstRSBlk)
		{
			rc = RC_SET( NE_FLM_NOT_FOUND);
			goto Exit;
		}

		if( m_pFirstRSBlk == m_pLastRSBlk)
		{
			m_pCurRSBlk = m_pFirstRSBlk;
		}
		else
		{
			m_pCurRSBlk = selectMidpoint( m_pFirstRSBlk, m_pLastRSBlk, FALSE);
		}

		if( RC_BAD( rc = m_pCurRSBlk->setBuffer( m_pucBlockBuf1, m_uiBlkSize)))
		{
			goto Exit;
		}
	}

	// Set the exclusive low block and high block.

	pLowBlk = m_pFirstRSBlk;
	pHighBlk = m_pLastRSBlk;

	// Loop until the correct block is found.

	for( ;;)
	{
		// Two return value returned: rc and iBlkCompare.
		// FindMatch returns NE_FLM_OK if the entry if found in the block.
		//	It returns NE_FLM_NOT_FOUND if not found in the block.
		// uiCompare returns 0 if entry would be within the block.
		// otherwise < 0 if previous blocks should be checked
		// and > 0 if next blocks should be checked.

		rc = m_pCurRSBlk->findMatch(
									(FLMBYTE *) pvMatchEntry, uiMatchEntryLength,
									(FLMBYTE *) pvFoundEntry, puiFoundEntryLength,
									&iBlkCompare );

		// Found match or should key be within the block.

		if( RC_OK(rc) || iBlkCompare == 0)
		{
			goto Exit;
		}

		if( iBlkCompare < 0)
		{
			// Done if the low block
			// Keep NE_FLM_NOT_FOUND return code

			if( m_pCurRSBlk == pLowBlk)
			{
				goto Exit;
			}

			// Set the new high block

			pHighBlk = m_pCurRSBlk->m_pPrev;
		}
		else
		{
			// Done if we are at the high block
			// Keep the NE_FLM_NOT_FOUND return code

			if( m_pCurRSBlk == pHighBlk)
			{
				goto Exit;
			}

			pLowBlk = m_pCurRSBlk->m_pNext;
		}

		if( RC_BAD( rc = m_pCurRSBlk->setBuffer( NULL)))
		{
			goto Exit;
		}

		m_pCurRSBlk = selectMidpoint( pLowBlk, pHighBlk, FALSE);

		// Need to set the working buffer.

		if( RC_BAD( rc = m_pCurRSBlk->setBuffer( m_pucBlockBuf1, m_uiBlkSize)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Select the midpoint between two different blocks in a list.
		Entries should not be the same value.
*****************************************************************************/
F_ResultSetBlk * F_ResultSet::selectMidpoint(
	F_ResultSetBlk *	pLowBlk,
	F_ResultSetBlk *	pHighBlk,
	FLMBOOL				bPickHighIfNeighbors)
{
	FLMUINT				uiCount;
	F_ResultSetBlk *	pTempBlk;

	// If the same then return.

	if( pLowBlk == pHighBlk)
	{
		pTempBlk = pLowBlk;
		goto Exit;
	}

	// Check if neighbors and use the boolean flag.

	if( pLowBlk->m_pNext == pHighBlk)
	{
		pTempBlk = (F_ResultSetBlk *)(bPickHighIfNeighbors
											 ? pHighBlk
											 : pLowBlk);
		goto Exit;
	}

	// Count the total blocks exclusive between low and high and add one.
	// Check pTempBlk against null to not crash.

	for( pTempBlk = pLowBlk, uiCount = 1;
		  pTempBlk && (pTempBlk != pHighBlk);
		  uiCount++)
	{
		pTempBlk = pTempBlk->m_pNext;
	}

	// Check for implementation error - pTempBlk is NULL and handle.

	if( !pTempBlk)
	{
		f_assert( 0);
		pTempBlk = pLowBlk;
		goto Exit;
	}

	// Loop to the middle item
	// Divide count by 2

	uiCount >>= 1;
	for( pTempBlk = pLowBlk; uiCount > 0; uiCount--)
	{
		pTempBlk = pTempBlk->m_pNext;
	}

Exit:

	return( pTempBlk);
}

/*****************************************************************************
Desc:	Set the current entry position.
*****************************************************************************/
RCODE FTKAPI F_ResultSet::setPosition(
	FLMUINT64		ui64Position)
{
	RCODE					rc = NE_FLM_OK;
	F_ResultSetBlk *	pInitialBlk = m_pCurRSBlk;

	f_assert( m_bFinalizeCalled);

	if( ui64Position == RS_POSITION_NOT_SET)
	{
		// Set out of focus

		if( m_pCurRSBlk)
		{
			if( RC_BAD( rc = m_pCurRSBlk->setBuffer( NULL)))
			{
				goto Exit;
			}
		}

		m_pCurRSBlk = NULL;
		goto Exit;
	}

	if( !m_pCurRSBlk)
	{
		m_pCurRSBlk = m_pFirstRSBlk;
	}

	// Check for empty result set.

	if( !m_pCurRSBlk)
	{
		rc = RC_SET( NE_FLM_EOF_HIT);
		goto Exit;
	}

	if( ui64Position < m_pCurRSBlk->m_ui64BlkEntryPosition)
	{
		// Go backwards looking for the correct block.

		do
		{
			m_pCurRSBlk = m_pCurRSBlk->m_pPrev;
			f_assert( m_pCurRSBlk);
		}
		while( ui64Position < m_pCurRSBlk->m_ui64BlkEntryPosition);
	}
	else if( ui64Position >= m_pCurRSBlk->m_ui64BlkEntryPosition +
								  m_pCurRSBlk->m_BlockHeader.uiEntryCount)
	{
		// Go forward looking for the correct block.

		do
		{
			if( !m_pCurRSBlk->m_pNext)
			{
				// Will set rc to EOF in SetPosition below.

				break;
			}

			m_pCurRSBlk = m_pCurRSBlk->m_pNext;
		}
		while( ui64Position >= m_pCurRSBlk->m_ui64BlkEntryPosition +
									m_pCurRSBlk->m_BlockHeader.uiEntryCount);
	}

	// Need working buffer out of focus.

	if( pInitialBlk != m_pCurRSBlk)
	{
		if( pInitialBlk)
		{
			if( RC_BAD( rc = pInitialBlk->setBuffer( NULL)))
			{
				goto Exit;
			}
		}

		// Need working buffer into focus.

		if( RC_BAD( rc = m_pCurRSBlk->setBuffer( m_pucBlockBuf1, m_uiBlkSize)))
		{
			goto Exit;
		}
	}

	// Now we are positioned to the correct block.

	if( RC_BAD( rc = m_pCurRSBlk->setPosition( ui64Position)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Return a pointer to the next entry in the list.
*****************************************************************************/
RCODE F_ResultSet::getNextPtr(
	F_ResultSetBlk **	ppCurBlk,
	FLMBYTE **			ppucBuffer,
	FLMUINT *			puiReturnLength)
{
	RCODE					rc = NE_FLM_OK;
	F_ResultSetBlk *	pCurBlk = *ppCurBlk;
	F_ResultSetBlk *	pNextBlk;
	FLMBYTE *			pucBuffer;

	f_assert( pCurBlk);

	while( RC_BAD( rc = pCurBlk->getNextPtr( ppucBuffer, puiReturnLength)))
	{
		if( rc == NE_FLM_EOF_HIT)
		{
			if( pCurBlk->m_pNext)
			{
				pNextBlk = pCurBlk->m_pNext;
				if( !pNextBlk->m_BlockHeader.bFirstBlock)
				{
					pucBuffer = pCurBlk->m_pucBlockBuf;
					pCurBlk->setBuffer( NULL );
					pCurBlk = pNextBlk;
					if( RC_BAD( rc = pCurBlk->setBuffer( pucBuffer, m_uiBlkSize)))
					{
						goto Exit;
					}
					*ppCurBlk = pCurBlk;
					continue;
				}
			}
		}

		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:		Union two block lists into a result output list.  This may be
			called to union two result sets or to perform the initial merge-sort
			on a create result set.

			Performing an N-way merge would be fast when we have over 10K
			of entries.  However, the code is more complex.
*****************************************************************************/
RCODE F_ResultSet::unionBlkLists(
	F_ResultSetBlk *	pLeftBlk,
	F_ResultSetBlk *	pRightBlk)
{
	RCODE			rc = NE_FLM_OK;
	FLMBYTE *	pucLeftEntry;
	FLMBYTE *	pucRightEntry;
	FLMUINT		uiLeftLength;
	FLMUINT		uiRightLength;

	// If no right block then copy all of the items from the left block
	// to the output block.  We could optimize this in the future.

	if( !pRightBlk)
	{
		rc = copyRemainingItems( pLeftBlk);
		goto Exit;
	}

	// Now the fun begins.  Read entries from both lists and union
	// while checking the order of the entries.

	if( RC_BAD( rc = getNextPtr( &pLeftBlk, &pucLeftEntry, &uiLeftLength)))
	{
		if( rc == NE_FLM_EOF_HIT)
		{
			rc = copyRemainingItems( pRightBlk);
		}

		goto Exit;
	}

	if( RC_BAD( rc = getNextPtr( &pRightBlk, &pucRightEntry, &uiRightLength)))
	{
		if( rc == NE_FLM_EOF_HIT)
		{
			rc = copyRemainingItems( pLeftBlk);
		}

		goto Exit;
	}

	for (;;)
	{
		FLMINT	iCompare;

		if( RC_BAD(rc = m_pCompare->compare( pucLeftEntry, uiLeftLength,
				pucRightEntry, uiRightLength, &iCompare )))
		{
			goto Exit;
		}

		if( iCompare < 0)
		{
			// Take the left item.

			if( RC_BAD(rc = addEntry( pucLeftEntry, uiLeftLength)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = getNextPtr( &pLeftBlk, 
				&pucLeftEntry, &uiLeftLength)))
			{
				if( rc != NE_FLM_EOF_HIT)
				{
					goto Exit;
				}

				if( RC_BAD( rc = addEntry( pucRightEntry, uiRightLength)))
				{
					goto Exit;
				}

				// Left entries are done - read all of the right entries.

				rc = copyRemainingItems( pRightBlk);
				goto Exit;
			}
		}
		else
		{
			// If equals then drop the right item and continue comparing left.
			// WARNING: Don't try to optimize for equals because when one
			// list runs out the remaining duplicate entries must be dropped.
			// Continuing to compare the duplicate item is the correct way.

			if( iCompare > 0 || !m_bDropDuplicates)
			{
				// Take the right item.

				if( RC_BAD(rc = addEntry( pucRightEntry, uiRightLength)))
				{
					goto Exit;
				}
			}

			if( RC_BAD( rc = getNextPtr( &pRightBlk, 
				&pucRightEntry, &uiRightLength)))
			{
				if( rc != NE_FLM_EOF_HIT)
				{
					goto Exit;
				}

				if( RC_BAD( rc = addEntry( pucLeftEntry, uiLeftLength)))
				{
					goto Exit;
				}

				// Right entries are done - read all of the left entries.

				rc = copyRemainingItems( pLeftBlk);
				goto Exit;
			}
		}
	}

Exit:

	if( RC_OK( rc))
	{
		// Flush out the output entries.

		rc = m_pCurRSBlk->finalize( TRUE );
		m_pCurRSBlk->setBuffer( NULL);
		m_pCurRSBlk = NULL;

		if( m_pSortStatus)
		{
			RCODE	rc2;
			
			++m_ui64UnitsDone;
			if( RC_BAD( rc2 = m_pSortStatus->reportSortStatus( m_ui64EstTotalUnits,
											m_ui64UnitsDone)))
			{
				if( RC_OK( rc))
				{
					rc = rc2;
				}
			}
		}
	}

	return( rc);
}

/*****************************************************************************
Desc:	Copy the remaining items from a block list to the output.
*****************************************************************************/
RCODE F_ResultSet::copyRemainingItems(
	F_ResultSetBlk *	pCurBlk)
{
	RCODE					rc;
	FLMBYTE *			pucEntry;
	FLMUINT				uiLength;

	while( RC_OK( rc = getNextPtr( &pCurBlk, &pucEntry, &uiLength)))
	{
		if( RC_BAD( rc = addEntry( pucEntry, uiLength)))
		{
			goto Exit;
		}
	}

	if( rc == NE_FLM_EOF_HIT)
	{
		rc = NE_FLM_OK;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Closes and deletes one of two files.
*****************************************************************************/
void F_ResultSet::closeFile(
	IF_MultiFileHdl **	ppMultiFileHdl,
	FLMBOOL					bDelete)
{
	if( ppMultiFileHdl == &m_pMultiFileHdl1)
	{
		if( m_bFile1Opened)
		{
			m_pMultiFileHdl1->closeFile( bDelete);
			m_bFile1Opened = FALSE;
		}

		if( m_pMultiFileHdl1)
		{
			m_pMultiFileHdl1->Release();
			m_pMultiFileHdl1 = NULL;
		}
	}
	else
	{
		if( m_bFile2Opened)
		{
			m_pMultiFileHdl2->closeFile( TRUE);
			m_bFile2Opened = FALSE;
		}

		if( m_pMultiFileHdl2)
		{
			m_pMultiFileHdl2->Release();
			m_pMultiFileHdl2 = NULL;
		}
	}
}

/*****************************************************************************
Desc:	Close the file if previously opened and creates the file.
*****************************************************************************/
RCODE F_ResultSet::openFile(
	IF_MultiFileHdl **		ppMultiFileHdl)
{
	RCODE			rc = NE_FLM_OK;
	FLMBOOL *	pbFileOpened;
	char *		pszDirPath;

	// Will close and delete if opened, else will do nothing.

	closeFile( ppMultiFileHdl);

	if( ppMultiFileHdl == &m_pMultiFileHdl1)
	{
		pbFileOpened = &m_bFile1Opened;
		pszDirPath = &m_szIoFilePath1 [0];
	}
	else
	{
		pbFileOpened = &m_bFile2Opened;
		pszDirPath = &m_szIoFilePath2 [0];
	}

	f_strcpy( pszDirPath, m_szIoDefaultPath);

	if( RC_BAD( rc = FlmAllocMultiFileHdl( ppMultiFileHdl)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = (*ppMultiFileHdl)->createUniqueFile( pszDirPath,
										FRSET_FILENAME_EXTENSION)))
	{
		(*ppMultiFileHdl)->Release();
		*ppMultiFileHdl = NULL;
		goto Exit;
	}

	*pbFileOpened = TRUE;

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
F_ResultSetBlk::F_ResultSetBlk()
{
	m_pNext = m_pPrev = NULL;
	m_pCompare = NULL;
	reset();
}

/*****************************************************************************
Desc:	Reset a block so it can be reused.
******************************************************************************/
void F_ResultSetBlk::reset( void)
{
	f_assert( !m_pNext && !m_pPrev);

	// Initialize all of the member variables
	// between this constructor, SetBuffer() and Setup().

	m_BlockHeader.ui64FilePos = RSBLK_UNSET_FILE_POS;
	m_BlockHeader.uiEntryCount = 0;
	m_ppMultiFileHdl = NULL;
	m_ui64BlkEntryPosition = RS_POSITION_NOT_SET;
	m_iEntryPos = 0;
	m_bDuplicateFound = FALSE;
	m_bPositioned = FALSE;
	m_bModifiedEntry = FALSE;
	m_pucBlockBuf = NULL;
}

/*****************************************************************************
Desc:
******************************************************************************/
void F_ResultSetBlk::setup(
	IF_MultiFileHdl **		ppMultiFileHdl,
	IF_ResultSetCompare *	pCompare,
	FLMUINT						uiEntrySize,
	FLMBOOL						bFirstInList,
	FLMBOOL						bDropDuplicates,
	FLMBOOL						bEntriesInOrder)
{
	f_assert( ppMultiFileHdl);
	m_ppMultiFileHdl = ppMultiFileHdl;

	if( m_pCompare)
	{
		m_pCompare->Release();
	}

	if( (m_pCompare = pCompare) != NULL)
	{
		m_pCompare->AddRef();
	}

	m_uiEntrySize = uiEntrySize;
	m_BlockHeader.bFirstBlock = bFirstInList;
	m_BlockHeader.bLastBlock = FALSE;
	m_bFixedEntrySize = m_uiEntrySize ? TRUE : FALSE;

	if( !m_uiEntrySize)
	{
		m_uiEntrySize = sizeof( F_VAR_HEADER);
	}

	m_bDropDuplicates = bDropDuplicates;
	m_bEntriesInOrder = bEntriesInOrder;
}

/*****************************************************************************
Desc:		The buffer is NOT allocated the by the result set block object.
			Setup the pucBuffer and associated variables.  Read in the data
			for this block if necessary.  If NULL is passed in as pucBuffer
			then this block is not the active block anymore.
Notes:	Must be called before other methods below are called.
*****************************************************************************/
RCODE F_ResultSetBlk::setBuffer(
	FLMBYTE *		pucBuffer,			// Working buffer or NULL
	FLMUINT			uiBufferLength)	// Default value is RSBLK_BLOCK_SIZE.
{
	RCODE				rc = NE_FLM_OK;

	// If a buffer is defined then read in the data from disk.

	if( pucBuffer)
	{
		m_pucBlockBuf = pucBuffer;
		if( !m_BlockHeader.uiEntryCount)
		{
			// uiBlockSize is the final block size after squeeze.
			// uiLengthRemaining is working value of bytes available.

			m_BlockHeader.uiBlockSize = uiBufferLength;
			m_uiLengthRemaining = uiBufferLength;

			if( m_bFixedEntrySize)
			{
				m_pucEndPoint = m_pucBlockBuf;
			}
			else
			{
				m_pucEndPoint = m_pucBlockBuf + uiBufferLength;
			}
		}
		else
		{
			// Read in the data if necessary.

			if( RC_BAD( rc = read()))
			{
				goto Exit;
			}
		}

		// The block is now in focus

		m_bPositioned = TRUE;
	}
	else
	{
		// Deactivating block so the buffer can be reused.
		// Check if the block has been modified

		if( m_bModifiedEntry)
		{
			// Is this a lone block?

			if( !m_BlockHeader.bLastBlock || !m_BlockHeader.bFirstBlock)
			{
				if( RC_BAD( rc = write()))
				{
					goto Exit;
				}
			}
			m_bModifiedEntry = FALSE;
		}

		// The block is now out of focus

		m_bPositioned = FALSE;
		m_pucEndPoint = m_pucBlockBuf = NULL;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:		Add a variable length entry to the result set.  If fixed length
			entry then call AddEntry for fixed length entries.
*****************************************************************************/
RCODE F_ResultSetBlk::addEntry(
	FLMBYTE *		pucEntry,
	FLMUINT			uiEntryLength)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiAlignLength;
	F_VAR_HEADER *	pEntry;

	f_assert( m_pucBlockBuf);

	// Was setup called for fixed length entries?

	if( m_bFixedEntrySize )
	{
		rc = addEntry( pucEntry );
		goto Exit;
	}

	uiAlignLength = (uiEntryLength + FLM_ALLOC_ALIGN) & (~FLM_ALLOC_ALIGN);

	// Check to see if the current buffer will overflow.

	if( m_uiLengthRemaining < uiAlignLength + sizeof( F_VAR_HEADER))
	{
		// Caller should call Flush and setup correctly what to do next.

		rc = RC_SET( NE_FLM_EOF_HIT );
		goto Exit;
	}

	// Copy entry and compute the offset value for pNextEntryPtr.

	m_pucEndPoint -= uiAlignLength;
	f_memcpy( m_pucEndPoint, pucEntry, uiEntryLength );

	pEntry = ((F_VAR_HEADER *)m_pucBlockBuf) + m_BlockHeader.uiEntryCount;
	pEntry->ui32Offset = (FLMUINT32)(m_pucEndPoint - m_pucBlockBuf);
	pEntry->ui32Length = (FLMUINT32)uiEntryLength;

	m_uiLengthRemaining -= (uiAlignLength + sizeof( F_VAR_HEADER));
	m_BlockHeader.uiEntryCount++;

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Add a fixed length entry to the result set.
*****************************************************************************/
RCODE F_ResultSetBlk::addEntry(
	FLMBYTE *	pucEntry)
{
	RCODE		rc = NE_FLM_OK;

	// Check that setup was called for fixed length entries.

	f_assert( m_bFixedEntrySize);

	// Check to see if the current buffer is full.

	if( m_uiLengthRemaining < m_uiEntrySize)
	{
		// Caller should call Flush and setup correctly what to do next.

		rc = RC_SET( NE_FLM_EOF_HIT);
		goto Exit;
	}

	f_memcpy( m_pucBlockBuf + (m_uiEntrySize * m_BlockHeader.uiEntryCount),
		pucEntry, m_uiEntrySize);
	m_BlockHeader.uiEntryCount++;
	m_pucEndPoint += m_uiEntrySize;
	m_uiLengthRemaining -= m_uiEntrySize;

Exit:

	return( rc);
}

/*****************************************************************************
Desc:		Modify the current entry being references.
Notes:	The size of each block cannot be modified.  This is to allow
			writing to the same location on disk and not waste disk memory.
*****************************************************************************/
RCODE F_ResultSetBlk::modifyEntry(
	FLMBYTE *	pucEntry,
	FLMUINT		uiEntryLength)
{
	RCODE	rc = NE_FLM_OK;

	F_UNREFERENCED_PARM( uiEntryLength);

	f_assert( m_pucBlockBuf);

	// The incoming entry MUST be the same size.

	if( m_bFixedEntrySize )
	{
		// Assert that the entry length must be zero.
		// If not - still use m_uiEntrySize;

		f_assert( !uiEntryLength);

		// Copy over the current item.

		f_memcpy( &m_pucBlockBuf [m_iEntryPos * m_uiEntrySize],
						pucEntry, m_uiEntrySize );
	}
	else
	{
		// Variable Length

		F_VAR_HEADER *	pCurEntry;

		pCurEntry = ((F_VAR_HEADER *)m_pucBlockBuf) + m_iEntryPos;

		// We cannot support changing the entry size at this time.

		f_assert( uiEntryLength == (FLMUINT)pCurEntry->ui32Length);

		f_memcpy( m_pucBlockBuf + pCurEntry->ui32Offset,
				pucEntry, uiEntryLength);
	}

	m_bModifiedEntry = TRUE;
	return( rc);
}

/*****************************************************************************
Desc:		The block is full and need to flush the block to disk.  If
			bForceWrite is FALSE then will not write block to disk.
*****************************************************************************/
RCODE F_ResultSetBlk::flush(
	FLMBOOL		bLastBlockInList,		// Last block in a block list.
	FLMBOOL		bForceWrite)			// if TRUE write out to disk.
{
	RCODE			rc = NE_FLM_OK;

	// Make sure SetBuffer was called

	f_assert( m_pucBlockBuf);
	squeezeSpace();

	if( !m_bEntriesInOrder)
	{
		// Remove duplicate entries.

		if( RC_BAD( rc = sortAndRemoveDups()))
		{
			goto Exit;
		}
	}

	m_bEntriesInOrder = TRUE;
	m_BlockHeader.bLastBlock = bLastBlockInList;

	if( bForceWrite)
	{
		if( RC_BAD( rc = write()))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	If there is length remaining, squeeze out additional space.
*****************************************************************************/
void F_ResultSetBlk::squeezeSpace( void)
{
	FLMUINT	uiPos;

	// Fixed Entry Size?

	if( m_bFixedEntrySize)
	{
		// Yes, no need to squeeze out any space.

		goto Exit;
	}

	// Is there room to shift things down?
	// Don't do if no entries or if less than 64 bytes.

	if( m_uiLengthRemaining >= 64 && m_BlockHeader.uiEntryCount)
	{
		FLMUINT			uiBytesToMoveUp;
		F_VAR_HEADER *	pEntry;

		uiBytesToMoveUp = m_uiLengthRemaining;
		m_uiLengthRemaining = 0;

		// Overlapping memory move call.

		f_assert( (m_pucBlockBuf + m_BlockHeader.uiBlockSize) > m_pucEndPoint );
		f_assert( uiBytesToMoveUp < m_BlockHeader.uiBlockSize );

		f_memmove( m_pucEndPoint - uiBytesToMoveUp, m_pucEndPoint,
			(FLMUINT) ((m_pucBlockBuf + m_BlockHeader.uiBlockSize ) - m_pucEndPoint ));

		m_BlockHeader.uiBlockSize -= uiBytesToMoveUp;
		m_pucEndPoint -= uiBytesToMoveUp;

		// Change all of the offsets for every entry.  This is expensive.

		for( uiPos = 0, pEntry = (F_VAR_HEADER *)m_pucBlockBuf;
			  uiPos < m_BlockHeader.uiEntryCount;
			  pEntry++, uiPos++)
		{
			pEntry->ui32Offset -= (FLMUINT32)uiBytesToMoveUp;
		}
	}

Exit:

	return;
}

/*****************************************************************************
Desc:	Sort the current block and remove all duplicates.
*****************************************************************************/
RCODE F_ResultSetBlk::sortAndRemoveDups( void)
{
	RCODE			rc = NE_FLM_OK;

	// Nothing to do if one or zero entries in the block.

	if( m_BlockHeader.uiEntryCount <= 1 || !m_pCompare)
	{
		goto Exit;
	}

	m_bDuplicateFound = FALSE;
	if( RC_BAD( rc = quickSort( 0, m_BlockHeader.uiEntryCount - 1)))
	{
		goto Exit;
	}

	// Some users of result sets may not have any duplicates to remove
	// or may want the side effect of having duplicates to further
	// process the entries like for sorting tracker records.  It is up
	// to the compare routine to never return 0 in this case.

	// This algorithm is tuned for the case where there are zero or few
	// duplicate records.  Removing duplicates is expensive in this design.

	if( m_bDropDuplicates && m_bDuplicateFound)
	{
		FLMUINT	uiEntriesRemaining;
		FLMINT	iCompare;

		if( m_bFixedEntrySize)
		{
			FLMBYTE *	pucEntry;
			FLMBYTE *	pucNextEntry;

			pucEntry = m_pucBlockBuf;
			for( uiEntriesRemaining = m_BlockHeader.uiEntryCount - 1
				; uiEntriesRemaining > 0
				; uiEntriesRemaining-- )
			{
				pucNextEntry = pucEntry + m_uiEntrySize;

				if( RC_BAD( rc = m_pCompare->compare( pucEntry, m_uiEntrySize,
									  pucNextEntry, m_uiEntrySize,
									  &iCompare)))
				{
					goto Exit;
				}

				if( iCompare == 0)
				{
					removeEntry( pucEntry);

					// Leave pucEntry alone - everyone will scoot down
				}
				else
				{
					pucEntry += m_uiEntrySize;
				}
			}
		}
		else
		{
			F_VAR_HEADER *	pEntry = (F_VAR_HEADER *)m_pucBlockBuf;
			F_VAR_HEADER *	pNextEntry;

			for( uiEntriesRemaining = m_BlockHeader.uiEntryCount - 1
				; uiEntriesRemaining > 0
				; uiEntriesRemaining-- )
			{
				pNextEntry = pEntry + 1;

				if( RC_BAD( rc = m_pCompare->compare( m_pucBlockBuf + pEntry->ui32Offset,
									  (FLMUINT)pEntry->ui32Length,
									  m_pucBlockBuf + pNextEntry->ui32Offset,
									  (FLMUINT)pNextEntry->ui32Length,
									  &iCompare)))
				{
					goto Exit;
				}

				if( iCompare == 0)
				{
					removeEntry( (FLMBYTE *)pEntry);

					// Leave pEntry alone - everyone will scoot down
				}
				else
				{
					pEntry++;
				}
			}
		}
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Remove the current entry from the block.
*****************************************************************************/
void F_ResultSetBlk::removeEntry(
	FLMBYTE *	pucEntry)
{
	if( m_bFixedEntrySize)
	{
		// Don't like moving zero bytes - check first.

		if( pucEntry + m_uiEntrySize < m_pucEndPoint)
		{
			// This is really easy - just memmove everyone down.

			f_memmove( pucEntry, pucEntry + m_uiEntrySize,
						(FLMUINT)(m_pucEndPoint - pucEntry) - m_uiEntrySize);
		}

		m_BlockHeader.uiEntryCount--;
		m_BlockHeader.uiBlockSize -= m_uiEntrySize;
		m_pucEndPoint -= m_uiEntrySize;
	}
	else
	{
		// Variable length entries - much harder

		// Example - remove entry  3 below...

		// [entryOfs1:len][entryOfs2:len][entryOfs3:len][entryOfs4:len]
		// [entryData1][entryData2][entryData3][entryData4]

		// Need to reduce EntryOfs1 and entryOfs2 by m_uiEntrySize+entryLen3.
		// because these entries are stored AFTER entry 3 - entries are first
		// stored going from the back of the block to the front of the block.
		// Need to reduce Ofs4 by OFFSET_SIZE.

		F_VAR_HEADER *	pEntry = (F_VAR_HEADER *)pucEntry;
		FLMUINT			uiDeletedOffset = (FLMUINT)pEntry->ui32Offset;
		FLMUINT			uiTempOffset;
		FLMUINT			uiDeletedLength = (FLMUINT)pEntry->ui32Length;
		F_VAR_HEADER *	pCurEntry;
		FLMUINT			uiPos;
		FLMUINT			uiMoveBytes;

		f_assert( m_BlockHeader.uiBlockSize >=
						(uiDeletedOffset + uiDeletedLength ));

		uiMoveBytes = (FLMUINT)
			(m_BlockHeader.uiBlockSize - (uiDeletedOffset + uiDeletedLength));

		if( uiMoveBytes)
		{

			// First move down the variable length entry data.

			f_memmove( m_pucBlockBuf + uiDeletedOffset,
						  m_pucBlockBuf + uiDeletedOffset + uiDeletedLength,
						  uiMoveBytes );
		}

		f_assert( m_BlockHeader.uiBlockSize >=
							(FLMUINT)((FLMBYTE *)(&pEntry[1]) - m_pucBlockBuf) );

		uiMoveBytes = m_BlockHeader.uiBlockSize -
							(FLMUINT)((FLMBYTE *)(&pEntry [1]) - m_pucBlockBuf);

		if( uiMoveBytes)
		{
			f_memmove( pEntry, &pEntry[1], uiMoveBytes );
		}

		m_BlockHeader.uiBlockSize -= (uiDeletedLength + sizeof( F_VAR_HEADER));

		// Adjust the offset values.

		m_BlockHeader.uiEntryCount--;

		for( uiPos = 0, pCurEntry = (F_VAR_HEADER *)m_pucBlockBuf
			; uiPos < m_BlockHeader.uiEntryCount
			; uiPos++, pCurEntry++)
		{
			// Assume that the offsets are NOT in descending order.
			// This will help in the future additional adding and deleting
			// to an existing result set.

			uiTempOffset = (FLMUINT)pCurEntry->ui32Offset;
			if (uiTempOffset > uiDeletedOffset)
			{
				uiTempOffset -= uiDeletedLength;
			}
			uiTempOffset -= sizeof( F_VAR_HEADER);
			pCurEntry->ui32Offset = (FLMUINT32)uiTempOffset;
		}
	}
}

/*****************************************************************************
Desc:		Quick sort an array of values.
Notes:	Optimized the above quicksort algorithm.  On page 559 the book
			suggests that "The worst case can sometimes be avioded by choosing
			more carefully the record for final placement at each state."
			This algorithm picks a mid point for the compare value.  Doing
			this helps the worst case where the entries are in order.  In Order
			tests went from 101 seconds down to 6 seconds!
			This helps the 'in order' sorts from worst case Order(N^^2)/2 with
			the normal quickSort to Order(NLog2 N) for the worst case.
			Also optimized the number of recursions to Log2 N from (N-2).
			Will recurse the SMALLER side and will iterate to the top of
			the routine for the LARGER side.  Follow comments below.
*****************************************************************************/
RCODE F_ResultSetBlk::quickSort(
	FLMUINT			uiLowerBounds,
	FLMUINT			uiUpperBounds)
{
	RCODE				rc = NE_FLM_OK;
	FLMBYTE *		pucEntryTbl = m_pucBlockBuf;
	FLMBYTE *		pucCurEntry;
	FLMUINT			uiLBPos;
	FLMUINT			uiUBPos;
	FLMUINT			uiMIDPos;
	FLMUINT			uiLeftItems;
	FLMUINT			uiRightItems;
	FLMINT			iCompare;
	FLMUINT			uiEntrySize = m_uiEntrySize;
	FLMBYTE			ucaSwapBuffer[ RS_MAX_FIXED_ENTRY_SIZE];

#define	RS_SWAP(pTbl,pos1,pos2)	{ \
	f_memcpy( ucaSwapBuffer, &pTbl[pos2*uiEntrySize], uiEntrySize); \
	f_memcpy( &pTbl[ pos2 * uiEntrySize ], &pTbl[ pos1 * uiEntrySize ], uiEntrySize ); \
	f_memcpy( &pTbl[ pos1 * uiEntrySize ], ucaSwapBuffer, uiEntrySize ); }

Iterate_Larger_Half:

	uiUBPos = uiUpperBounds;
	uiLBPos = uiLowerBounds;
	uiMIDPos = (uiUpperBounds + uiLowerBounds + 1) / 2;
	pucCurEntry = &pucEntryTbl[ uiMIDPos * uiEntrySize ];

	for (;;)
	{
		// Don't compare with target

		while( uiLBPos == uiMIDPos ||
				 (RC_OK( rc = entryCompare( &pucEntryTbl[ uiLBPos * uiEntrySize],
												pucCurEntry,
												&iCompare)) &&
				  iCompare < 0))
		{
			if( uiLBPos >= uiUpperBounds)
			{
				break;
			}
			uiLBPos++;
		}

		if( RC_BAD( rc))
		{
			goto Exit;
		}

		// Don't compare with target

		while( uiUBPos == uiMIDPos ||
				 (RC_OK( rc = entryCompare( pucCurEntry,
												&pucEntryTbl[uiUBPos * uiEntrySize],
												&iCompare)) &&
				  iCompare < 0))
		{
			// Check for underflow

			if( !uiUBPos)
			{
				break;
			}

			uiUBPos--;
		}

		if (RC_BAD( rc))
		{
			goto Exit;
		}

		// Interchange and continue loop

		if( uiLBPos < uiUBPos)
		{
			// Interchange [uiLBPos] with [uiUBPos].

			RS_SWAP( pucEntryTbl, uiLBPos, uiUBPos );
			uiLBPos++;						// Scan from left to right.
			uiUBPos--;						// Scan from right to left.
		}
		else
		{
			// Past each other - done

			break;
		}
	}

	// 5 cases to check.
	// 1) UB < MID < LB - Don't need to do anything.
	// 2) MID < UB < LB - swap( UB, MID )
	// 3) UB < LB < MID - swap( LB, MID )
	// 4) UB = LB < MID - swap( LB, MID ) - At first position
	// 5) MID < UB = LB - swap( UB, MID ) - At last position

	// Check for swap( LB, MID ) - cases 3 and 4

	if( uiLBPos < uiMIDPos)
	{
		// Interchange [uiLBPos] with [uiMIDPos]

		RS_SWAP( pucEntryTbl, uiMIDPos, uiLBPos );
		uiMIDPos = uiLBPos;
	}
	else if (uiMIDPos < uiUBPos)
	{
		// Cases 2 and 5
		// Interchange [uUBPos] with [uiMIDPos]

		RS_SWAP( pucEntryTbl, uiMIDPos, uiUBPos );
		uiMIDPos = uiUBPos;
	}

	// To save stack space - recurse the SMALLER Piece.  For the larger
	// piece goto the top of the routine.  Worst case will be
	// (Log2 N)  levels of recursion.

	// Don't recurse in the following cases:
	// 1) We are at an end.  Just loop to the top.
	// 2) There are two on one side.  Compare and swap.  Loop to the top.
	//		Don't swap if the values are equal.  There are many recursions
	//		with one or two entries.  This doesn't speed up any so it is
	//		commented out.

	// Check the left piece.

	uiLeftItems = (uiLowerBounds + 1 < uiMIDPos )
							? uiMIDPos - uiLowerBounds		// 2 or more
							: 0;
	uiRightItems = (uiMIDPos + 1 < uiUpperBounds )
							? uiUpperBounds - uiMIDPos 		// 2 or more
							: 0;

	if( uiLeftItems < uiRightItems)
	{
		// Recurse on the LEFT side and goto the top on the RIGHT side.

		if( uiLeftItems)
		{
			// Recursive call.

			if( RC_BAD( rc = quickSort( uiLowerBounds, uiMIDPos - 1)))
			{
				goto Exit;
			}
		}

		uiLowerBounds = uiMIDPos + 1;
		goto Iterate_Larger_Half;
	}
	else if( uiLeftItems)
	{
		// Recurse on the RIGHT side and goto the top for the LEFT side.

		if( uiRightItems)
		{
			// Recursive call.

			if( RC_BAD( rc = quickSort( uiMIDPos + 1, uiUpperBounds)))
			{
				goto Exit;
			}
		}

		uiUpperBounds = uiMIDPos - 1;
		goto Iterate_Larger_Half;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Write this block to disk.
*****************************************************************************/
RCODE F_ResultSetBlk::write( void)
{
	RCODE		rc = NE_FLM_OK;
	FLMUINT	uiBytesWritten;

	// By this time there better be something to write...
	// The file should be opened by default.

	if( m_BlockHeader.ui64FilePos == RSBLK_UNSET_FILE_POS)
	{
		if( RC_BAD(rc = (*m_ppMultiFileHdl)->size( &m_BlockHeader.ui64FilePos)))
		{
			goto Exit;
		}
	}

	// Write out the block header definition.

	if( RC_BAD( rc = (*m_ppMultiFileHdl)->write(
						 m_BlockHeader.ui64FilePos,
						 sizeof( F_BLOCK_HEADER), &m_BlockHeader,
						 &uiBytesWritten)))
	{
		goto Exit;
	}

	// Write out the data buffer

	if( RC_BAD( rc = (*m_ppMultiFileHdl)->write(
						 m_BlockHeader.ui64FilePos + sizeof( F_BLOCK_HEADER),
						 m_BlockHeader.uiBlockSize,
						 m_pucBlockBuf,
						 &uiBytesWritten)))
	{
		goto Exit;
	}

Exit:

	return rc;
}

/*****************************************************************************
Desc:	Read in the specified block into memory.
*****************************************************************************/
RCODE F_ResultSetBlk::read( void)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT				uiBytesRead;
	F_BLOCK_HEADER		BlockHeader;

	// Nothing to do?

	if (m_BlockHeader.ui64FilePos == RSBLK_UNSET_FILE_POS)
	{
		goto Exit;
	}

	// First read the block header in.

	if (RC_BAD( rc = (*m_ppMultiFileHdl)->read( m_BlockHeader.ui64FilePos,
						sizeof( F_BLOCK_HEADER ),
						(void *)&BlockHeader, &uiBytesRead)))
	{
		goto Exit;
	}

	// Verify that the block header data is the same.
	// This is the best we can do to verify that the file handle
	// is not junky.

	if (BlockHeader.ui64FilePos != m_BlockHeader.ui64FilePos ||
		 BlockHeader.uiEntryCount != m_BlockHeader.uiEntryCount)
	{
		rc = RC_SET( NE_FLM_FAILURE);
		goto Exit;
	}

	// Read in the data buffer

	if (RC_BAD( rc = (*m_ppMultiFileHdl)->read(
						m_BlockHeader.ui64FilePos + sizeof( F_BLOCK_HEADER),
						m_BlockHeader.uiBlockSize,
						m_pucBlockBuf, &uiBytesRead)))
	{
		goto Exit;
	}

Exit:

	if (RC_OK(rc))
	{
		m_bPositioned = TRUE;
		m_iEntryPos = -1;
	}

	return( rc);
}

/*****************************************************************************
Desc:	Copies the current entry into the user buffer.  Checks for overflow.
*****************************************************************************/
RCODE F_ResultSetBlk::copyCurrentEntry(
	FLMBYTE *	pucBuffer,
	FLMUINT		uiBufferLength,
	FLMUINT *	puiReturnLength)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiEntrySize;
	F_VAR_HEADER *	pEntry;
	FLMBYTE *		pucEntry;

	f_assert( pucBuffer);

	// Copy the current entry.  This is a shared routine
	// because the code to copy an entry is a little complicated.

	if( !m_bFixedEntrySize)
	{
		pEntry = ((F_VAR_HEADER *)m_pucBlockBuf) + m_iEntryPos;
		uiEntrySize = pEntry->ui32Length;
		pucEntry = m_pucBlockBuf + pEntry->ui32Offset;
	}
	else
	{
		uiEntrySize = m_uiEntrySize;
		pucEntry = &m_pucBlockBuf[ m_iEntryPos * uiEntrySize ];
	}

	if( uiBufferLength && (uiEntrySize > uiBufferLength))
	{
		uiEntrySize = uiBufferLength;
		rc = RC_SET( NE_FLM_CONV_DEST_OVERFLOW );
		// Fall through into memcpy.
	}

	f_memcpy( pucBuffer, pucEntry, uiEntrySize);

	if( puiReturnLength)
	{
		*puiReturnLength = uiEntrySize;
	}

	return( rc);
}

/*****************************************************************************
Desc:	Return the Current entry reference in the result set.
*****************************************************************************/
RCODE F_ResultSetBlk::getCurrent(
	FLMBYTE *	pBuffer,
	FLMUINT		uiBufferLength,
	FLMUINT *	puiReturnLength)
{
	RCODE			rc;

	f_assert( m_pucBlockBuf);
	if( !m_bPositioned )
	{
		rc = RC_SET( NE_FLM_NOT_FOUND);
		goto Exit;
	}

	// Check for EOF and BOF conditions - otherwise return current.

	if (m_iEntryPos >= (FLMINT) m_BlockHeader.uiEntryCount)
	{
		rc = RC_SET( NE_FLM_EOF_HIT);
		goto Exit;
	}

	if( m_iEntryPos == -1)
	{
		rc = RC_SET( NE_FLM_BOF_HIT );
		goto Exit;
	}

	rc = copyCurrentEntry( pBuffer, uiBufferLength, puiReturnLength );

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Return a pointer to the next reference in the result set.
		If the result set is not positioned then the first entry will
		be returned.
*****************************************************************************/
RCODE F_ResultSetBlk::getNextPtr(
	FLMBYTE **	ppucBuffer,
	FLMUINT *	puiReturnLength)
{
	RCODE			rc = NE_FLM_OK;

	f_assert( ppucBuffer);
	f_assert( puiReturnLength);
	f_assert( m_bPositioned);

	// Are we on the last entry or past the last entry?

	if (m_iEntryPos + 1 >= (FLMINT) m_BlockHeader.uiEntryCount)
	{
		m_iEntryPos = (FLMINT)m_BlockHeader.uiEntryCount;
		rc = RC_SET( NE_FLM_EOF_HIT );
		goto Exit;
	}

	// Position to the next entry

	m_iEntryPos++;

	if (!m_bFixedEntrySize)
	{
		F_VAR_HEADER *	pEntry;

		pEntry = ((F_VAR_HEADER *)m_pucBlockBuf) + m_iEntryPos;
		*puiReturnLength = (FLMUINT)pEntry->ui32Length;
		*ppucBuffer =  m_pucBlockBuf + pEntry->ui32Offset;
	}
	else
	{
		*puiReturnLength = m_uiEntrySize;
		*ppucBuffer = &m_pucBlockBuf[ m_iEntryPos * m_uiEntrySize];
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Return the previous reference in the result set.  If the result set
		is not positioned then the last entry will be returned.
*****************************************************************************/
RCODE F_ResultSetBlk::getPrev(
	FLMBYTE *	pucBuffer,
	FLMUINT		uiBufferLength,
	FLMUINT *	puiReturnLength)
{
	RCODE			rc = NE_FLM_OK;

	f_assert( m_bPositioned);

	// If not positioned then position past last entry.

	if (m_iEntryPos == -1)
	{
		m_iEntryPos = (FLMINT) m_BlockHeader.uiEntryCount;
	}

	// Are we on the first entry or before the first entry?

	if (m_iEntryPos == 0)
	{
		m_iEntryPos = -1;
		rc = RC_SET( NE_FLM_BOF_HIT);
		goto Exit;
	}

	m_iEntryPos--;				// position to previous entry.

	if (RC_BAD( rc = copyCurrentEntry( pucBuffer, uiBufferLength,
									puiReturnLength)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Set the current entry position for this block.
*****************************************************************************/
RCODE F_ResultSetBlk::setPosition(
	FLMUINT64	ui64Position)
{
	RCODE			rc = NE_FLM_OK;

	// Buffer must be set or SetBuffer() will set iEntryPos back to -1.

	f_assert( m_bPositioned);

	if( ui64Position == RS_POSITION_NOT_SET)
	{
		m_iEntryPos = -1;
		goto Exit;
	}
	
	f_assert( ui64Position >= m_ui64BlkEntryPosition);

	// Convert to a zero based number relative to this block.

	if (ui64Position >= m_ui64BlkEntryPosition)
	{
		ui64Position -= m_ui64BlkEntryPosition;
	}
	else
	{
		ui64Position = 0;
	}

	if (ui64Position >= m_BlockHeader.uiEntryCount)
	{
		rc = RC_SET( NE_FLM_EOF_HIT);
		m_iEntryPos = m_BlockHeader.uiEntryCount;
	}
	else
	{
		m_iEntryPos = (FLMINT)ui64Position;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:		Find the matching entry within the block using the compare routine.
			This does a binary search on entries.  Watch the (out) variable.
Out:		*piCompare = 0		-	match found OR entry would compare between
										the low and high entries in the block.
							 > 0	-	match entry is greater than
										the highest entry in the block.
							 < 0	-	match entry is less than the
										lowest entry in the block.
Notes:	One side effect is that m_iEntryPos is set to the matched
			entry or some random entry if not found is returned.
*****************************************************************************/
RCODE F_ResultSetBlk::findMatch(		// Find and return an etnry that
												// matches in this block.
	FLMBYTE *	pucMatchEntry,			// Entry to match
	FLMUINT		uiMatchEntryLength,	// Variable length of above entry
	FLMBYTE *	pucFoundEntry,			// (out) Entry to return
	FLMUINT *	puiFoundEntryLength,	// (out) Length of entry returned
	FLMINT	*	piCompare)				// See comments above.
{
	RCODE			rc = NE_FLM_OK;
	FLMINT		iCompare;				// Return from CompareEntry
	FLMUINT		uiLow;
	FLMUINT		uiHigh;
	FLMUINT		uiMid;
	FLMUINT		uiLimit;

	uiLow = 0;
	uiHigh = uiLimit = m_BlockHeader.uiEntryCount - 1;

	// Set the match entry length

	if (!uiMatchEntryLength)
	{
		uiMatchEntryLength = m_uiEntrySize;
	}

	// Check the first and last entries in the block.
	// Copy the current entry if found.

	if( RC_BAD( rc = compareEntry( pucMatchEntry, uiMatchEntryLength, uiLow,
								&iCompare)))
	{
		goto Exit;
	}

	if( iCompare <= 0)
	{
		if( iCompare < 0)
		{
			rc = RC_SET( NE_FLM_NOT_FOUND);
		}
		else
		{
			if( pucFoundEntry)
			{
				rc = copyCurrentEntry( pucFoundEntry, 0, puiFoundEntryLength);
			}
		}

		*piCompare = iCompare;
		goto Exit;
	}

	if( RC_BAD( rc =  compareEntry( pucMatchEntry, uiMatchEntryLength, uiHigh,
											&iCompare )))
	{
		goto Exit;
	}

	if (iCompare >= 0)
	{
		if (iCompare > 0)
		{
			rc = RC_SET( NE_FLM_NOT_FOUND);
		}
		else
		{
			rc = copyCurrentEntry( pucFoundEntry, 0, puiFoundEntryLength);
		}
		*piCompare = iCompare;
		goto Exit;
	}

	// Set the iCompare to equals because
	// the match entry sorts within the block somewhere.
	// Binary search the entries in the block.  May still
	// not find the matching entry.

	*piCompare = 0;
	for( ;;)
	{
		uiMid = (uiLow + uiHigh) >> 1;	// (uiLow + uiHigh) / 2

		if( RC_BAD( rc = compareEntry( pucMatchEntry, uiMatchEntryLength, uiMid,
											&iCompare)))
		{
			goto Exit;
		}

		if( iCompare == 0)
		{
			// Found Match!  All set up to return.

			if( pucFoundEntry)
			{
				rc = copyCurrentEntry( pucFoundEntry, 0, puiFoundEntryLength);
			}

			goto Exit;
		}

		// Check if we are done - where uiLow >= uiHigh.

		if( uiLow >= uiHigh)
		{
			// Done - item not found

			break;
		}

		if( iCompare < 0)
		{
			// Way too high?

			if( !uiMid)
			{
				break;
			}

			// Too high

			uiHigh = uiMid - 1;
		}
		else
		{
			if( uiMid == uiLimit)
			{
				// Done - hit the top
				break;
			}

			// Too low

			uiLow = uiMid + 1;
		}
	}

	// On break set we did not find the matching entry.

	rc = RC_SET( NE_FLM_NOT_FOUND);

Exit:

	return( rc);
}

/*****************************************************************************
Desc:		Compare the buffer entry with entry identifies by
			uiEntryPos.
Out:		*piCompare = 0	-	match found OR entry value would compare
									between the low and high entries.
						  > 0	-	match entry is greater than
									the highest entry in the block.
						  < 0	-	match entry is less than the
									lowest entry in the block.
*****************************************************************************/
RCODE F_ResultSetBlk::compareEntry(	// Compares match entry with entry
												// identified by uiEntryPos.
	FLMBYTE *	pucMatchEntry,			// Entry to match
	FLMUINT		uiMatchEntryLength,	// Variable length of pMatchEntry.
	FLMUINT		uiEntryPos,				// Position of entry in block.
	FLMINT *		piCompare)				// Return from compare.
{
	FLMBYTE *	pucEntry;
	FLMUINT		uiEntrySize;

	// Position to the entry.

	m_iEntryPos = (FLMINT) uiEntryPos;

	if (!m_bFixedEntrySize)
	{
		F_VAR_HEADER *	pEntry;

		pEntry = ((F_VAR_HEADER *)m_pucBlockBuf) + m_iEntryPos;
		uiEntrySize = (FLMUINT)pEntry->ui32Length;
		pucEntry = m_pucBlockBuf + pEntry->ui32Offset;
	}
	else
	{
		uiEntrySize = m_uiEntrySize;
		pucEntry = &m_pucBlockBuf[ m_iEntryPos * uiEntrySize ];
	}

	return( m_pCompare->compare( pucMatchEntry, uiMatchEntryLength,
							pucEntry, uiEntrySize,
							piCompare));
}

/*****************************************************************************
Desc:	Make sure the state reflects what we have in the blocks.
*****************************************************************************/
void F_ResultSetBlk::adjustState(
	FLMUINT			uiBlkBufferSize)
{
	F_VAR_HEADER *		pVarHdr;
	FLMUINT				uiTotalSize = 0;
	FLMBYTE *			pucFromPos;
	FLMBYTE *			pucToPos;
	FLMUINT				uiBytesMoved;
	FLMUINT				uiPos;

	// Are the entries in the block fixed length or variable length?

	if( m_bFixedEntrySize)
	{
		// Fixed Length.

		m_uiLengthRemaining = uiBlkBufferSize - 
										(m_BlockHeader.uiEntryCount * m_uiEntrySize);
		m_ui64BlkEntryPosition = 0;
		m_pucEndPoint = m_pucBlockBuf + (m_BlockHeader.uiEntryCount * m_uiEntrySize);
	}
	else
	{
		// Variable length Entries.
		// We may need to move the entries around.  First, determine if the block is full.

		if( m_BlockHeader.uiBlockSize < uiBlkBufferSize)
		{
			uiTotalSize = m_BlockHeader.uiBlockSize -
								(sizeof(F_VAR_HEADER) * m_BlockHeader.uiEntryCount);
			
			pucFromPos = m_pucBlockBuf + (sizeof(F_VAR_HEADER) * m_BlockHeader.uiEntryCount);

			pucToPos = (m_pucBlockBuf + uiBlkBufferSize) - uiTotalSize;

			f_memmove( pucToPos, pucFromPos, uiTotalSize);

			for( uiBytesMoved = (pucToPos - pucFromPos),
							uiPos = 0,
							pVarHdr = (F_VAR_HEADER *)m_pucBlockBuf;
				  uiPos < m_BlockHeader.uiEntryCount;
				  pVarHdr++, uiPos++)
			{
				pVarHdr->ui32Offset += (FLMUINT32)uiBytesMoved;
			}

			m_pucEndPoint = pucToPos;
			m_uiLengthRemaining = uiBlkBufferSize - m_BlockHeader.uiBlockSize;
			m_ui64BlkEntryPosition = pucToPos - m_pucBlockBuf;
		}
		else
		{
			m_uiLengthRemaining = 0;
		}
	}

	m_BlockHeader.uiBlockSize = uiBlkBufferSize;
}

/*****************************************************************************
Desc:	truncate the file to the current file position.
*****************************************************************************/
RCODE F_ResultSetBlk::truncate(
	FLMBYTE *			pszPath)
{
	RCODE				rc = NE_FLM_OK;

	if( RC_BAD( rc = (*m_ppMultiFileHdl)->truncateFile( 
		m_BlockHeader.ui64FilePos)))
	{
		goto Exit;
	}

	(*m_ppMultiFileHdl)->closeFile( FALSE);

	if( RC_BAD( rc = (*m_ppMultiFileHdl)->openFile( ( char *)pszPath)))
	{
		goto Exit;
	}

	m_BlockHeader.ui64FilePos = RSBLK_UNSET_FILE_POS;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_BTreeResultSet::setupResultSet(
	IF_ResultSetCompare *	pCompare)
{
	RCODE				rc = NE_FLM_OK;
	IF_BTree *		pBTree = NULL;
	
	
	if( m_pBTree)
	{
		m_pBTree->Release();
		m_pBTree = NULL;
	}
	
	if( RC_BAD( rc = FlmAllocBTree( NULL, &pBTree)))
	{
		goto Exit;
	}
	
	if( RC_BAD( pBTree->btCreate( 0, FALSE, TRUE, NULL, pCompare)))
	{
		goto Exit;
	}
	
	m_pBTree = pBTree;
	pBTree = NULL;
		
Exit:

	if( pBTree)
	{
		pBTree->Release();
	}

	return( rc);
}
		
/****************************************************************************
Desc:
****************************************************************************/
RCODE F_BTreeResultSet::addEntry(
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyLength,
	FLMBYTE *	pucEntry,
	FLMUINT		uiEntryLength)
{
	RCODE			rc = NE_FLM_OK;

	f_assert( uiKeyLength <= FLM_MAX_KEY_SIZE);

	if( RC_BAD( rc = m_pBTree->btInsertEntry( pucKey, uiKeyLength,
		uiKeyLength, pucEntry, uiEntryLength, TRUE, TRUE)))
	{
		if (rc == NE_FLM_NOT_UNIQUE)
		{
			rc = NE_FLM_OK;
		}
		else
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_BTreeResultSet::modifyEntry(
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyLength,
	FLMBYTE *	pucEntry,
	FLMUINT		uiEntryLength)
{
	RCODE			rc = NE_FLM_OK;

	f_assert( uiKeyLength <= FLM_MAX_KEY_SIZE);

	if( RC_BAD( rc = m_pBTree->btReplaceEntry( pucKey, uiKeyLength,
		uiKeyLength, pucEntry, uiEntryLength, TRUE, TRUE)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_BTreeResultSet::deleteEntry(
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyLength)
{
	RCODE			rc = NE_FLM_OK;

	f_assert( uiKeyLength <= FLM_MAX_KEY_SIZE);

	if (RC_BAD( rc = m_pBTree->btRemoveEntry( 
		pucKey, uiKeyLength, uiKeyLength)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_BTreeResultSet::findEntry(
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyLen,
	FLMBYTE *	pucBuffer,
	FLMUINT		uiBufferLength,
	FLMUINT *	puiReturnLength)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiLengthRV;

	f_assert( uiKeyLen <= FLM_MAX_KEY_SIZE);

	if( RC_BAD( rc = m_pBTree->btLocateEntry( pucKey, uiKeyLen, &uiKeyLen,
		FLM_EXACT, NULL, &uiLengthRV)))
	{
		goto Exit;
	}

	if( pucBuffer)
	{
		if( RC_BAD( rc = m_pBTree->btGetEntry( pucKey, uiKeyLen,
			pucBuffer, uiBufferLength, puiReturnLength)))
		{
			goto Exit;
		}
	}
	else if( puiReturnLength)
	{
		*puiReturnLength = uiLengthRV;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_BTreeResultSet::getCurrent(
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyLength,
	FLMBYTE *	pucEntry,
	FLMUINT		uiEntryLength,
	FLMUINT *	puiReturnLength)
{
	RCODE			rc = NE_FLM_OK;

	f_assert( uiKeyLength <= FLM_MAX_KEY_SIZE);

	if( RC_BAD( rc = m_pBTree->btGetEntry( pucKey, uiKeyLength,
		pucEntry, uiEntryLength, puiReturnLength)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_BTreeResultSet::getNext(
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyBufLen,
	FLMUINT *	puiKeyLen,
	FLMBYTE *	pucEntry,
	FLMUINT		uiEntryLength,
	FLMUINT *	puiReturnLength)
{
	RCODE			rc = NE_FLM_OK;

	f_assert( uiKeyBufLen <= FLM_MAX_KEY_SIZE);

	if( RC_BAD( rc = m_pBTree->btNextEntry( pucKey, uiKeyBufLen, puiKeyLen,
		puiReturnLength)))
	{
		goto Exit;
	}

	if( pucEntry)
	{
		if( RC_BAD( rc = m_pBTree->btGetEntry( pucKey, *puiKeyLen,
			pucEntry, uiEntryLength, puiReturnLength)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_BTreeResultSet::getPrev(
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyBufLen,
	FLMUINT *	puiKeyLen,
	FLMBYTE *	pucEntry,
	FLMUINT		uiEntryLength,
	FLMUINT *	puiReturnLength)
{
	RCODE			rc = NE_FLM_OK;

	f_assert( uiKeyBufLen <= FLM_MAX_KEY_SIZE);

	if( RC_BAD( rc = m_pBTree->btPrevEntry( pucKey, uiKeyBufLen, puiKeyLen,
		puiReturnLength)))
	{
		goto Exit;
	}

	if( pucEntry)
	{
		if( RC_BAD( rc = m_pBTree->btGetEntry( pucKey, *puiKeyLen,
			pucEntry, uiEntryLength, puiReturnLength)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_BTreeResultSet::getFirst(
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyBufLen,
	FLMUINT *	puiKeyLen,
	FLMBYTE *	pucEntry,
	FLMUINT		uiEntryLength,
	FLMUINT *	puiReturnLength)
{
	RCODE			rc = NE_FLM_OK;

	f_assert( uiKeyBufLen <= FLM_MAX_KEY_SIZE);

	m_pBTree->btResetBtree();
	if( RC_BAD( rc = m_pBTree->btFirstEntry( pucKey, uiKeyBufLen, puiKeyLen,
		puiReturnLength)))
	{
		goto Exit;
	}

	if( pucEntry)
	{
		if( RC_BAD( rc = m_pBTree->btGetEntry( pucKey, *puiKeyLen,
			pucEntry, uiEntryLength, puiReturnLength)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_BTreeResultSet::getLast(
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyBufLen,
	FLMUINT *	puiKeyLen,
	FLMBYTE *	pucEntry,
	FLMUINT		uiEntryLength,
	FLMUINT *	puiReturnLength)
{
	RCODE			rc = NE_FLM_OK;

	f_assert( uiKeyBufLen <= FLM_MAX_KEY_SIZE);

	if( RC_BAD( rc = m_pBTree->btLastEntry( pucKey, uiKeyBufLen, puiKeyLen,
		puiReturnLength)))
	{
		goto Exit;
	}

	if( pucEntry)
	{
		if( RC_BAD( rc = m_pBTree->btGetEntry( pucKey, *puiKeyLen,
			pucEntry, uiEntryLength, puiReturnLength)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI FlmAllocResultSet(
	IF_ResultSet **			ppResultSet)
{
	if( (*ppResultSet = f_new F_ResultSet) == NULL)
	{
		return( RC_SET( NE_FLM_MEM));
	}
	
	return( NE_FLM_OK);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI FlmAllocBTreeResultSet(
	IF_ResultSetCompare *	pCompare,
	IF_BTreeResultSet **		ppBTreeResultSet)
{
	RCODE							rc = NE_FLM_OK;
	F_BTreeResultSet *		pBTreeResultSet = NULL;
	
	if( (pBTreeResultSet = f_new F_BTreeResultSet) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pBTreeResultSet->setupResultSet( pCompare)))
	{
		goto Exit;
	}
	
	*ppBTreeResultSet = pBTreeResultSet;
	pBTreeResultSet = NULL;
	
Exit:

	if( pBTreeResultSet)
	{
		pBTreeResultSet->Release();
	}

	return( rc);
}

