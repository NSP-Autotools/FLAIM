//------------------------------------------------------------------------------
// Desc:	Class for managing a name table.
// Tabs:	3
//
// Copyright (c) 1992, 1994-2007 Novell, Inc. All Rights Reserved.
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
#if 0

#define MAX_ELEMENTS_TO_LOAD			0xFFFF
#define MAX_ATTRIBUTES_TO_LOAD		0xFFFF

typedef struct
{
	FLMUINT				uiType;
	FLMUNICODE *		puzTagName;
	FLMUINT				uiTagNum;
	FLMUINT				uiDataType;
	FLMUNICODE *		puzNamespace;
} FLM_TAG_INFO;

typedef FLMINT (*	TAG_COMPARE_FUNC)(
	FLM_TAG_INFO *			pTagInfo1,
	FLM_TAG_INFO *			pTagInfo2);

FSTATIC FLMINT tagNameCompare(
	const FLMUNICODE *	puzName1,
	const char *			pszName1,
	const FLMUNICODE *	puzName2);

FSTATIC FLMINT compareTagTypeAndName(
	FLM_TAG_INFO *			pTagInfo1,
	FLM_TAG_INFO *			pTagInfo2);

FSTATIC FLMINT compareTagTypeAndNum(
	FLM_TAG_INFO *			pTagInfo1,
	FLM_TAG_INFO *			pTagInfo2);

FSTATIC void sortTagTbl(
	FLM_TAG_INFO **		ppTagInfoTbl,
	FLMUINT					uiLowerBounds,
	FLMUINT					uiUpperBounds,
	TAG_COMPARE_FUNC		fnTagCompare);

/****************************************************************************
Desc:	Class for name/number lookup.
****************************************************************************/
class F_NameTable : public IF_NameTable
{
public:

	F_NameTable();

	virtual ~F_NameTable();

	void FTKAPI clearTable(
		FLMUINT					uiPoolBlkSize);

	RCODE FTKAPI getNextTagTypeAndNumOrder(
		FLMUINT					uiType,
		FLMUINT *				puiNextPos,
		FLMUNICODE *			puzTagName = NULL,
		char *					pszTagName = NULL,
		FLMUINT					uiNameBufSize = 0,
		FLMUINT *				puiTagNum = NULL,
		FLMUINT *				puiDataType = NULL,
		FLMUNICODE *			puzNamespace = NULL,
		FLMUINT					uiNamespaceBufSize = 0,
		FLMBOOL					bTruncatedNamesOk = TRUE);

	RCODE FTKAPI getNextTagTypeAndNameOrder(
		FLMUINT					uiType,
		FLMUINT *				puiNextPos,
		FLMUNICODE *			puzTagName = NULL,
		char *					pszTagName = NULL,
		FLMUINT					uiNameBufSize = 0,
		FLMUINT *				puiTagNum = NULL,
		FLMUINT *				puiDataType = NULL,
		FLMUNICODE *			puzNamespace = NULL,
		FLMUINT					uiNamespaceBufSize = 0,
		FLMBOOL					bTruncatedNamesOk = TRUE);

	RCODE FTKAPI getFromTagTypeAndName(
		FLMUINT					uiType,
		const FLMUNICODE *	puzTagName,
		const char *			pszTagName,
		FLMBOOL					bMatchNamespace,
		const FLMUNICODE *	puzNamespace = NULL,
		FLMUINT *				puiTagNum = NULL,
		FLMUINT *				puiDataType = NULL);

	RCODE FTKAPI getFromTagTypeAndNum(
		FLMUINT					uiType,
		FLMUINT					uiTagNum,
		FLMUNICODE *			puzTagName = NULL,
		char *					pszTagName = NULL,
		FLMUINT *				puiNameBufSize = NULL,
		FLMUINT *				puiDataType = NULL,
		FLMUNICODE *			puzNamespace = NULL,
		char *					pszNamespace = NULL,
		FLMUINT *				puiNamespaceBufSize = NULL,
		FLMBOOL					bTruncatedNamesOk = TRUE);

	RCODE FTKAPI addTag(
		FLMUINT					uiType,
		FLMUNICODE *			puzTagName,
		const char *			pszTagName,
		FLMUINT					uiTagNum,
		FLMUINT					uiDataType = 0,
		FLMUNICODE *			puzNamespace = NULL,
		FLMBOOL					bCheckDuplicates = TRUE);

	void FTKAPI removeTag(
		FLMUINT	uiType,
		FLMUINT	uiTagNum);

	RCODE FTKAPI cloneNameTable(
		IF_NameTable **		ppNewNameTable);

	FLMINT FTKAPI AddRef( void);

	FLMINT FTKAPI Release( void);
	
private:

	void sortTags( void);

	RCODE allocTag(
		FLMUINT					uiType,
		FLMUNICODE *			puzTagName,
		const char *			pszTagName,
		FLMUINT					uiTagNum,
		FLMUINT					uiDataType,
		FLMUNICODE *			puzNamespace,
		FLM_TAG_INFO **		ppTagInfo);

	RCODE reallocSortTables(
		FLMUINT					uiNewTblSize);

	RCODE copyTagName(
		FLMUNICODE *			puzDestTagName,
		char *					pszDestTagName,
		FLMUINT *				puiDestBufSize,
		FLMUNICODE *			puzSrcTagName,
		FLMBOOL					bTruncatedNamesOk);

	FLM_TAG_INFO * findTagByTypeAndNum(
		FLMUINT					uiType,
		FLMUINT					uiTagNum,
		FLMUINT *				puiInsertPos = NULL);

	FLM_TAG_INFO * findTagByTypeAndName(
		FLMUINT					uiType,
		const FLMUNICODE *	puzTagName,
		const char *			pszTagName,
		FLMBOOL					bMatchNamespace,
		const FLMUNICODE *	puzNamespace,
		FLMBOOL *				pbAmbiguous,
		FLMUINT *				puiInsertPos = NULL);

	RCODE insertTagInTables(
		FLM_TAG_INFO *			pTagInfo,
		FLMUINT					uiTagTypeAndNameTblInsertPos,
		FLMUINT					uiTagTypeAndNumTblInsertPos);

	FLMUNICODE * findNamespace(
		FLMUNICODE *			puzNamespace,
		FLMUINT *				puiInsertPos);

	RCODE insertNamespace(
		FLMUNICODE *			puzNamespace,
		FLMUINT					uiInsertPos);

	F_Pool						m_pool;
	FLMUINT						m_uiMemoryAllocated;
	FLM_TAG_INFO **			m_ppSortedByTagTypeAndName;
	FLM_TAG_INFO **			m_ppSortedByTagTypeAndNum;
	FLMUINT						m_uiTblSize;
	FLMUINT						m_uiNumTags;
	FLMBOOL						m_bTablesSorted;
	FLMUNICODE **				m_ppuzNamespaces;
	FLMUINT						m_uiNamespaceTblSize;
	FLMUINT						m_uiNumNamespaces;
	F_MUTEX						m_hRefMutex;
};

/****************************************************************************
Desc:
****************************************************************************/
F_NameTable::F_NameTable()
{
	m_pool.poolInit( 1024);
	m_uiMemoryAllocated = 0;
	m_ppSortedByTagTypeAndName = NULL;
	m_ppSortedByTagTypeAndNum = NULL;
	m_uiTblSize = 0;
	m_uiNumTags = 0;
	m_bTablesSorted = FALSE;
	m_ppuzNamespaces = NULL;
	m_uiNamespaceTblSize = 0;
	m_uiNumNamespaces = 0;
	m_hRefMutex = F_MUTEX_NULL;
}

/****************************************************************************
Desc:
****************************************************************************/
F_NameTable::~F_NameTable()
{
	clearTable( 0);
	if( m_hRefMutex)
	{
		f_mutexDestroy( &m_hRefMutex);
	}
}

/****************************************************************************
Desc:	Free everything in the table
****************************************************************************/
void FTKAPI F_NameTable::clearTable(
	FLMUINT	uiPoolBlkSize)
{
	m_pool.poolFree();
	
	if (uiPoolBlkSize)
	{
		m_pool.poolInit( uiPoolBlkSize);
	}
	
	m_uiMemoryAllocated = 0;

	// NOTE: Only one allocation is used for m_ppSortedByTagTypeAndName and
	// m_ppSortedByTagTypeAndNum - there is no
	// need to free m_ppSortedByTagTypeAndNum

	if (m_ppSortedByTagTypeAndName)
	{
		f_free( &m_ppSortedByTagTypeAndName);
		m_ppSortedByTagTypeAndNum = NULL;
		m_uiTblSize = 0;
		m_uiNumTags = 0;
	}

	if (m_ppuzNamespaces)
	{
		f_free( &m_ppuzNamespaces);
		m_ppuzNamespaces = NULL;
		m_uiNamespaceTblSize = 0;
		m_uiNumNamespaces = 0;
	}

	m_bTablesSorted = FALSE;
}

/****************************************************************************
Desc: Compare two tag names.  Name1 can be NATIVE or UNICODE.  If a non-NULL
		UNICODE string is passed, it will be used.  Otherwise, the NATIVE
		string will be used.
Note:	Comparison is case sensitive.  Either or both names may be NULL.
		Empty strings and NULL pointers are considered to be equal.
****************************************************************************/
FSTATIC FLMINT tagNameCompare(
	const FLMUNICODE *	puzName1,	// If NULL, use pszName1 for comparison
	const char *			pszName1,
	const FLMUNICODE *	puzName2)
{
	FLMUNICODE	uzChar1;
	FLMUNICODE	uzChar2;
	FLMUNICODE	uzLowerChar1;
	FLMUNICODE	uzLowerChar2;

	if (puzName1)
	{
		if (!puzName2)
		{
			if (*puzName1)
			{
				return( 1);
			}
			else
			{
				return( 0);
			}
		}
		
		for (;;)
		{
			uzChar1 = *puzName1;
			uzChar2 = *puzName2;

			if (!uzChar1)
			{
				if (!uzChar2)
				{
					return( 0);
				}
				else
				{
					return( -1);
				}
			}
			else if (!uzChar2)
			{
				return( 1);
			}
			else if (uzChar1 != uzChar2)
			{
Test_Case:
				uzLowerChar1 = f_unitolower( uzChar1);
				uzLowerChar2 = f_unitolower( uzChar2);

				if (uzLowerChar1 < uzLowerChar2)
				{
					return( -1);
				}
				else if (uzLowerChar1 > uzLowerChar2)
				{
					return( 1);
				}
				else if (uzLowerChar1 != uzChar1)
				{

					// Char1 is uppercase, char2 is lowercase
					// Uppercase sorts before lowercase

					return( -1);
				}
				else
				{

					// Char1 is lowercase, char2 is uppercase
					// Lowercase sorts after uppercase

					return( 1);
				}
			}
			puzName1++;
			puzName2++;
		}
	}
	else if (pszName1)
	{
		if (!puzName2)
		{
			if (*pszName1)
			{
				return( 1);
			}
			else
			{
				return( 0);
			}
		}
		for (;;)
		{
			uzChar1 = (FLMUNICODE)*pszName1;
			uzChar2 = *puzName2;
			if (!uzChar1)
			{
				if (!uzChar2)
				{
					return( 0);
				}
				else
				{
					return( -1);
				}
			}
			else if (!uzChar2)
			{
				return( 1);
			}
			else if (uzChar1 != uzChar2)
			{
				goto Test_Case;
			}
			pszName1++;
			puzName2++;
		}
	}
	else if (puzName2)
	{

		// Both name1's are NULL.

		if (*puzName2)
		{
			return( -1);
		}
		else
		{
			return( 0);
		}
	}
	else
	{

		// Both name1 and name2 are NULL.

		return( 0);
	}
}

/****************************************************************************
Desc: Lookup a tag by type and tag number.
****************************************************************************/
FLM_TAG_INFO * F_NameTable::findTagByTypeAndNum(
	FLMUINT		uiType,
	FLMUINT		uiTagNum,
	FLMUINT *	puiInsertPos)
{
	FLM_TAG_INFO *	pTagInfo = NULL;
	FLMUINT			uiTblSize;
	FLMUINT			uiLow;
	FLMUINT			uiMid;
	FLMUINT			uiHigh;
	FLMUINT			uiTblTagNum;
	FLMUINT			uiTblType;

	if (!m_bTablesSorted)
	{
		sortTags();
	}
	
	// Do binary search in the table

	if ((uiTblSize = m_uiNumTags) == 0)
	{
		if (puiInsertPos)
		{
			*puiInsertPos = 0;
		}
		goto Exit;
	}

	uiHigh = --uiTblSize;
	uiLow = 0;
	for (;;)
	{
		uiMid = (uiLow + uiHigh) / 2;

		uiTblType = m_ppSortedByTagTypeAndNum [uiMid]->uiType;
		uiTblTagNum = m_ppSortedByTagTypeAndNum [uiMid]->uiTagNum;
		if (uiTblType == uiType && uiTagNum == uiTblTagNum)
		{

			// Found Match

			pTagInfo = m_ppSortedByTagTypeAndNum [uiMid];
			if (puiInsertPos)
			{
				*puiInsertPos = uiMid;
			}
			goto Exit;
		}

		// Check if we are done

		if (uiLow >= uiHigh)
		{

			// Done, item not found

			if (puiInsertPos)
			{
				*puiInsertPos = (uiType < uiTblType ||
									  (uiType == uiTblType && uiTagNum < uiTblTagNum))
									 ? uiMid
									 : uiMid + 1;
			}
			goto Exit;
		}

		if (uiType < uiTblType ||
			 (uiType == uiTblType && uiTagNum < uiTblTagNum))
		{
			if (uiMid == 0)
			{
				if (puiInsertPos)
				{
					*puiInsertPos = 0;
				}
				goto Exit;
			}
			uiHigh = uiMid - 1;
		}
		else
		{
			if (uiMid == uiTblSize)
			{
				if (puiInsertPos)
				{
					*puiInsertPos = uiMid + 1;
				}
				goto Exit;
			}
			uiLow = uiMid + 1;
		}
	}

Exit:

	return( pTagInfo);
}

/****************************************************************************
Desc: Lookup a tag by tag type and tag name.  Tag name is passed in as a
		UNICODE string or a NATIVE string.  If a non-NULL UNICODE string is
		passed in, it will be used.  Otherwise, the NATIVE string will be used.
****************************************************************************/
FLM_TAG_INFO * F_NameTable::findTagByTypeAndName(
	FLMUINT					uiType,
	const FLMUNICODE *	puzTagName,
	const char *			pszTagName,
	FLMBOOL					bMatchNamespace,
	const FLMUNICODE *	puzNamespace,
	FLMBOOL *				pbAmbiguous,
	FLMUINT *				puiInsertPos)
{
	FLM_TAG_INFO *	pTagInfo = NULL;
	FLMUINT			uiTblType;
	FLMUINT			uiTblSize;
	FLMUINT			uiLow;
	FLMUINT			uiMid;
	FLMUINT			uiHigh;
	FLMINT			iCmp;

	if (!m_bTablesSorted)
	{
		sortTags();
	}
	
	// Do binary search in the table

	*pbAmbiguous = FALSE;
	if ((uiTblSize = m_uiNumTags) == 0)
	{
		if (puiInsertPos)
		{
			*puiInsertPos = 0;
		}
		goto Exit;
	}
	uiHigh = --uiTblSize;
	uiLow = 0;
	for (;;)
	{
		uiMid = (uiLow + uiHigh) / 2;
		uiTblType = m_ppSortedByTagTypeAndName [uiMid]->uiType;
		if (uiType < uiTblType)
		{
			iCmp = -1;
		}
		else if (uiType > uiTblType)
		{
			iCmp = 1;
		}
		else if ((iCmp = tagNameCompare( puzTagName, pszTagName,
						m_ppSortedByTagTypeAndName [uiMid]->puzTagName)) == 0)
		{
			if (!bMatchNamespace)
			{

				// Better not be trying to insert a new one in this case!

				f_assert( puiInsertPos == NULL);

				if ((uiMid &&
						tagNameCompare( puzTagName, pszTagName,
							m_ppSortedByTagTypeAndName [uiMid-1]->puzTagName) == 0) ||
					 (uiMid < m_uiNumTags - 1 &&
						tagNameCompare( puzTagName, pszTagName,
							m_ppSortedByTagTypeAndName [uiMid+1]->puzTagName) == 0))
				{
					*pbAmbiguous = TRUE;
				}
			}
			else
			{
				iCmp = tagNameCompare( puzNamespace, NULL,
						m_ppSortedByTagTypeAndName [uiMid]->puzNamespace);
			}

			if (iCmp == 0)
			{
				// Found Match

				pTagInfo = m_ppSortedByTagTypeAndName [uiMid];
				if (puiInsertPos)
				{
					*puiInsertPos = uiMid;
				}

				goto Exit;
			}
		}

		// Check if we are done

		if (uiLow >= uiHigh)
		{

			// Done, item not found

			if (puiInsertPos)
			{
				*puiInsertPos = (iCmp < 0)
									 ? uiMid
									 : uiMid + 1;
			}
			goto Exit;
		}

		if (iCmp < 0)
		{
			if (uiMid == 0)
			{
				if (puiInsertPos)
				{
					*puiInsertPos = 0;
				}
				goto Exit;
			}
			uiHigh = uiMid - 1;
		}
		else
		{
			if (uiMid == uiTblSize)
			{
				if (puiInsertPos)
				{
					*puiInsertPos = uiMid + 1;
				}
				goto Exit;
			}
			uiLow = uiMid + 1;
		}
	}

Exit:

	return( pTagInfo);
}

/****************************************************************************
Desc:	Copy a tag name to a UNICODE or NATIVE buffer.  Truncate if necessary.
		If a non-NULL UNICODE string is passed in, it will be populated.
		Otherwise, the NATIVE string will be populated.
****************************************************************************/
RCODE F_NameTable::copyTagName(
	FLMUNICODE *	puzDestTagName,
	char *			pszDestTagName,
	FLMUINT *		puiDestBufSize,	// Bytes, must be enough for null terminator
	FLMUNICODE *	puzSrcTagName,		// May be NULL
	FLMBOOL			bTruncatedNamesOk)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiDestCharCnt = *puiDestBufSize;
	FLMUINT			uiCharCnt;

	if (puzDestTagName)
	{

		// Decrement name buffer size by sizeof( FLMUNICODE) to allow for a
		// terminating NULL character. uiDestChars better be at least big
		// enough for a null terminating character.

		f_assert( uiDestCharCnt >= sizeof( FLMUNICODE));
		uiDestCharCnt /= sizeof( FLMUNICODE);
		uiDestCharCnt--;

		if (puzSrcTagName)
		{

			// Copy the name to the UNICODE buffer.

			uiCharCnt = 0;
			while (uiCharCnt < uiDestCharCnt && *puzSrcTagName)
			{
				*puzDestTagName++ = *puzSrcTagName;
				uiCharCnt++;
				puzSrcTagName++;
			}
			*puzDestTagName = 0;
			*puiDestBufSize = uiCharCnt;
			if (!bTruncatedNamesOk && *puzSrcTagName)
			{
				rc = RC_SET( NE_FLM_CONV_DEST_OVERFLOW);
				goto Exit;
			}
		}
		else
		{
			*puzDestTagName = 0;
			*puiDestBufSize = 0;
		}
	}
	else
	{

		// Decrement name buffer size by one to allow for a terminating
		// 0 character. uiDestCharCnt better be at list big
		// enough for a 0 terminating character.

		f_assert( uiDestCharCnt);
		uiDestCharCnt--;

		if (puzSrcTagName)
		{
			// Copy the name to the NATIVE buffer.  Non-Ascii UNICODE characters
			// will cause NE_FLM_CONV_ILLEGAL to be returned.

			uiCharCnt = 0;
			while (uiCharCnt < uiDestCharCnt && *puzSrcTagName)
			{
				if (*puzSrcTagName <= 0xFF)
				{
					*pszDestTagName++ =
						(char)f_tonative( (FLMBYTE)*puzSrcTagName);
					uiCharCnt++;
					puzSrcTagName++;				
				}
				else
				{
					rc = RC_SET( NE_FLM_CONV_ILLEGAL);
					goto Exit;
				}

			}
			*pszDestTagName = 0;
			*puiDestBufSize = uiCharCnt;
			if (!bTruncatedNamesOk && *puzSrcTagName)
			{
				rc = RC_SET( NE_FLM_CONV_DEST_OVERFLOW);
				goto Exit;
			}
		}
		else
		{
			*pszDestTagName = 0;
			*puiDestBufSize = 0;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Swap two entries in tag info table during sort.
*****************************************************************************/
FINLINE void tagInfoSwap(
	FLM_TAG_INFO * *	ppTagInfoTbl,
	FLMUINT				uiPos1,
	FLMUINT				uiPos2
	)
{
	FLM_TAG_INFO *	pTmpTagInfo = ppTagInfoTbl [uiPos1];
	ppTagInfoTbl [uiPos1] = ppTagInfoTbl [uiPos2];
	ppTagInfoTbl [uiPos2] = pTmpTagInfo;
}

/***************************************************************************
Desc:	Comparison function for sorting by tag type and name.
****************************************************************************/
FSTATIC FLMINT compareTagTypeAndName(
	FLM_TAG_INFO *		pTagInfo1,
	FLM_TAG_INFO *		pTagInfo2)
{
	FLMINT	iCmp;

	if (pTagInfo1->uiType < pTagInfo2->uiType)
	{
		return( -1);
	}
	else if (pTagInfo1->uiType > pTagInfo2->uiType)
	{
		return( 1);
	}
	else if ((iCmp = tagNameCompare( pTagInfo1->puzTagName, NULL,
						pTagInfo2->puzTagName)) != 0)
	{
		return( iCmp);
	}
	else
	{
		return( tagNameCompare( pTagInfo1->puzNamespace, NULL,
											pTagInfo2->puzNamespace));
	}
}

/***************************************************************************
Desc:	Comparison function for sorting by tag type and number.
****************************************************************************/
FSTATIC FLMINT compareTagTypeAndNum(
	FLM_TAG_INFO *	pTagInfo1,
	FLM_TAG_INFO *	pTagInfo2
	)
{
	if (pTagInfo1->uiType < pTagInfo2->uiType)
	{
		return( -1);
	}
	else if (pTagInfo1->uiType > pTagInfo2->uiType)
	{
		return( 1);
	}
	else if (pTagInfo1->uiTagNum < pTagInfo2->uiTagNum)
	{
		return( -1);
	}
	else if (pTagInfo1->uiTagNum > pTagInfo2->uiTagNum)
	{
		return( 1);
	}
	else
	{
		return( 0);
	}
}

/***************************************************************************
Desc:	Sort an array of SCACHE pointers by their block address.
****************************************************************************/
FSTATIC void sortTagTbl(
	FLM_TAG_INFO * *	ppTagInfoTbl,
	FLMUINT				uiLowerBounds,
	FLMUINT				uiUpperBounds,
	TAG_COMPARE_FUNC	fnTagCompare)
{
	FLMUINT			uiLBPos;
	FLMUINT			uiUBPos;
	FLMUINT			uiMIDPos;
	FLMUINT			uiLeftItems;
	FLMUINT			uiRightItems;
	FLM_TAG_INFO *	pCurTagInfo;
	FLMINT			iCompare;

Iterate_Larger_Half:

	uiUBPos = uiUpperBounds;
	uiLBPos = uiLowerBounds;
	uiMIDPos = (uiUpperBounds + uiLowerBounds + 1) / 2;
	pCurTagInfo = ppTagInfoTbl [uiMIDPos ];
	for (;;)
	{
		while (uiLBPos == uiMIDPos ||				// Don't compare with target
					((iCompare =
						fnTagCompare( ppTagInfoTbl [uiLBPos], pCurTagInfo)) < 0))
		{
			if (uiLBPos >= uiUpperBounds)
			{
				break;
			}
			uiLBPos++;
		}

		while (uiUBPos == uiMIDPos ||				// Don't compare with target
					(((iCompare =
						fnTagCompare( pCurTagInfo, ppTagInfoTbl [uiUBPos])) < 0)))
		{
			if (!uiUBPos)
			{
				break;
			}
			uiUBPos--;
		}

		if (uiLBPos < uiUBPos )			// Interchange and continue loop.
		{

			// Exchange [uiLBPos] with [uiUBPos].

			tagInfoSwap( ppTagInfoTbl, uiLBPos, uiUBPos);
			uiLBPos++;						// Scan from left to right.
			uiUBPos--;						// Scan from right to left.
		}
		else									// Past each other - done
		{
			break;
		}
	}

	// Check for swap( LB, MID ) - cases 3 and 4

	if( uiLBPos < uiMIDPos )
	{

		// Exchange [uiLBPos] with [uiMIDPos]

		tagInfoSwap( ppTagInfoTbl, uiMIDPos, uiLBPos);
		uiMIDPos = uiLBPos;
	}
	else if( uiMIDPos < uiUBPos )
	{

		// Exchange [uUBPos] with [uiMIDPos]

		tagInfoSwap( ppTagInfoTbl, uiMIDPos, uiUBPos);
		uiMIDPos = uiUBPos;
	}

	// Check the left piece.

	uiLeftItems = (uiLowerBounds + 1 < uiMIDPos )
							? uiMIDPos - uiLowerBounds		// 2 or more
							: 0;
	uiRightItems = (uiMIDPos + 1 < uiUpperBounds )
							? uiUpperBounds - uiMIDPos 		// 2 or more
							: 0;

	if( uiLeftItems < uiRightItems )
	{

		// Recurse on the LEFT side and goto the top on the RIGHT side.

		if (uiLeftItems )
		{
			sortTagTbl( ppTagInfoTbl, uiLowerBounds, uiMIDPos - 1, fnTagCompare);
		}
		uiLowerBounds = uiMIDPos + 1;
		goto Iterate_Larger_Half;
	}
	else if (uiLeftItems )	// Compute a truth table to figure out this check.
	{

		// Recurse on the RIGHT side and goto the top for the LEFT side.

		if (uiRightItems )
		{
			sortTagTbl( ppTagInfoTbl, uiMIDPos + 1, uiUpperBounds, fnTagCompare);
		}
		uiUpperBounds = uiMIDPos - 1;
		goto Iterate_Larger_Half;
	}
}

/****************************************************************************
Desc: Lookup a namespace - so we can reuse the memory.
****************************************************************************/
FLMUNICODE * F_NameTable::findNamespace(
	FLMUNICODE *	puzNamespace,
	FLMUINT *		puiInsertPos)
{
	FLMUNICODE *	puzFoundNamespace = NULL;
	FLMUINT			uiTblSize;
	FLMUINT			uiLow;
	FLMUINT			uiMid;
	FLMUINT			uiHigh;
	FLMINT			iCmp;

	// Do binary search in the table

	if ((uiTblSize = m_uiNumNamespaces) == 0)
	{
		if (puiInsertPos)
		{
			*puiInsertPos = 0;
		}
		goto Exit;
	}

	uiHigh = --uiTblSize;
	uiLow = 0;
	for (;;)
	{
		uiMid = (uiLow + uiHigh) / 2;

		if ((iCmp = tagNameCompare( puzNamespace, NULL,
						m_ppuzNamespaces [uiMid])) == 0)
		{

			// Found Match

			puzFoundNamespace = m_ppuzNamespaces [uiMid];
			if (puiInsertPos)
			{
				*puiInsertPos = uiMid;
			}
			goto Exit;
		}

		// Check if we are done

		if (uiLow >= uiHigh)
		{

			// Done, item not found

			if (puiInsertPos)
			{
				*puiInsertPos = (iCmp < 0)
									 ? uiMid
									 : uiMid + 1;
			}
			goto Exit;
		}

		if (iCmp < 0)
		{
			if (uiMid == 0)
			{
				if (puiInsertPos)
				{
					*puiInsertPos = 0;
				}
				goto Exit;
			}
			uiHigh = uiMid - 1;
		}
		else
		{
			if (uiMid == uiTblSize)
			{
				if (puiInsertPos)
				{
					*puiInsertPos = uiMid + 1;
				}
				goto Exit;
			}
			uiLow = uiMid + 1;
		}
	}

Exit:

	return( puzFoundNamespace);
}

/****************************************************************************
Desc:	Insert a tag info structure into the sorted tables at the specified
		positions.
****************************************************************************/
RCODE F_NameTable::insertNamespace(
	FLMUNICODE *	puzNamespace,
	FLMUINT			uiInsertPos)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiLoop;

	// See if we need to resize the table.  Add 32 each time.  There
	// should not be that many different namespaces.

	if (m_uiNumNamespaces == m_uiNamespaceTblSize)
	{
		FLMUINT			uiNewSize = m_uiNamespaceTblSize + 32;
		FLMUNICODE **	ppNewTbl;

		if( RC_BAD( rc = f_alloc( sizeof( FLMUNICODE *) * uiNewSize, &ppNewTbl)))
		{
			goto Exit;
		}

		// Copy the old tables into the new.

		if (m_uiNumNamespaces)
		{
			f_memcpy( ppNewTbl, m_ppuzNamespaces,
				sizeof( FLMUNICODE *) * m_uiNumNamespaces);
			f_free( &m_ppuzNamespaces);
		}
		m_ppuzNamespaces = ppNewTbl;
		m_uiNamespaceTblSize = uiNewSize;
	}

	// Insert the new namespace into the table.

	uiLoop = m_uiNumNamespaces;
	while (uiLoop > uiInsertPos)
	{
		m_ppuzNamespaces [uiLoop] = m_ppuzNamespaces [uiLoop - 1];
		uiLoop--;
	}
	m_ppuzNamespaces [uiInsertPos] = puzNamespace;

	// Increment the total number of namespaces

	m_uiNumNamespaces++;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Allocate a new tag info structure and set it up.
****************************************************************************/
RCODE F_NameTable::allocTag(
	FLMUINT				uiType,
	FLMUNICODE *		puzTagName,
	const char *		pszTagName,
	FLMUINT				uiTagNum,
	FLMUINT				uiDataType,
	FLMUNICODE *		puzNamespace,
	FLM_TAG_INFO **	ppTagInfo)
{
	RCODE					rc = NE_FLM_OK;
	void *				pvMark;
	FLMUINT				uiSaveMemoryAllocated;
	FLM_TAG_INFO *		pTagInfo;
	FLMUINT				uiNameSize;
	FLMUNICODE *		puzTmp;
	FLMUNICODE *		puzTblNamespace;
	FLMUINT				uiNamespaceInsertPos;

	// Create a new tag info structure.

	pvMark = m_pool.poolMark();
	uiSaveMemoryAllocated = m_uiMemoryAllocated;
	if (RC_BAD( rc = m_pool.poolCalloc( sizeof( FLM_TAG_INFO),
										(void **)&pTagInfo)))
	{
		goto Exit;
	}
	m_uiMemoryAllocated += sizeof( FLM_TAG_INFO);

	// Allocate the space for the tag name.

	if (puzTagName)
	{
		uiNameSize = (f_unilen( puzTagName) + 1) * sizeof( FLMUNICODE);
		if (RC_BAD( rc = m_pool.poolAlloc( uiNameSize,
											(void **)&pTagInfo->puzTagName)))
		{
			goto Exit;
		}
		m_uiMemoryAllocated += uiNameSize;
		f_memcpy( pTagInfo->puzTagName, puzTagName, uiNameSize);
	}
	else
	{
		uiNameSize = (f_strlen( pszTagName) + 1) * sizeof( FLMUNICODE);
		if (RC_BAD( rc = m_pool.poolAlloc( uiNameSize,
										(void **)&pTagInfo->puzTagName)))
		{
			goto Exit;
		}
		m_uiMemoryAllocated += uiNameSize;
		puzTmp = pTagInfo->puzTagName;
		while (*pszTagName)
		{
			*puzTmp++ = (FLMUNICODE)*pszTagName;
			pszTagName++;
		}
		*puzTmp = 0;
	}
	pTagInfo->uiType = uiType;
	pTagInfo->uiTagNum = uiTagNum;
	pTagInfo->uiDataType = uiDataType;

	if (puzNamespace && *puzNamespace)
	{

		// See if we have already allocated the namespace.  If so, just
		// point to it.  Otherwise, allocate it and add it to the
		// namespace table and then point to it.

		if ((puzTblNamespace = findNamespace( puzNamespace,
										&uiNamespaceInsertPos)) == NULL)
		{
			uiNameSize = (f_unilen( puzNamespace) + 1) * sizeof( FLMUNICODE);
			if (RC_BAD( rc = m_pool.poolAlloc( uiNameSize,
										(void **)&puzTblNamespace)))
			{
				goto Exit;
			}
			m_uiMemoryAllocated += uiNameSize;
			f_memcpy( puzTblNamespace, puzNamespace, uiNameSize);
			if (RC_BAD( rc = insertNamespace( puzTblNamespace,
										uiNamespaceInsertPos)))
			{
				goto Exit;
			}

			// Need to re-mark the pool after this point, because
			// we can now not afford to lose the namespace that was
			// allocated if the pool is reset at Exit due to a later
			// error.

			pvMark = m_pool.poolMark();
			uiSaveMemoryAllocated = m_uiMemoryAllocated;
		}
		
		pTagInfo->puzNamespace = puzTblNamespace;
	}

Exit:

	if (RC_BAD( rc))
	{
		m_pool.poolReset( pvMark);
		m_uiMemoryAllocated = uiSaveMemoryAllocated;
		pTagInfo = NULL;
	}
	*ppTagInfo = pTagInfo;

	return( rc);
}

/****************************************************************************
Desc:	Allocate the sort tables.
****************************************************************************/
RCODE F_NameTable::reallocSortTables(
	FLMUINT				uiNewTblSize)
{
	RCODE					rc = NE_FLM_OK;
	FLM_TAG_INFO **	ppNewTbl;

	if( RC_BAD( f_alloc( 
		sizeof( FLM_TAG_INFO *) * uiNewTblSize * 2, &ppNewTbl)))
	{
		goto Exit;
	}

	// Copy the old tables into the new.

	if (m_uiNumTags)
	{
		f_memcpy( ppNewTbl, m_ppSortedByTagTypeAndName,
						sizeof( FLM_TAG_INFO *) * m_uiNumTags);
		f_memcpy( &ppNewTbl [uiNewTblSize],
						m_ppSortedByTagTypeAndNum,
						sizeof( FLM_TAG_INFO *) * m_uiNumTags);
		f_free( &m_ppSortedByTagTypeAndName);
	}
	m_ppSortedByTagTypeAndName = ppNewTbl;
	m_ppSortedByTagTypeAndNum = &ppNewTbl [uiNewTblSize];

	m_uiTblSize = uiNewTblSize;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Get a tag name, number, etc. using type + tag number ordering.
		Tag name is returned as a UNICODE string or NATIVE string. If a
		non-NULL UNICODE string is passed in, it will be used.
		Otherwise, the NATIVE string will be used.
****************************************************************************/
RCODE FTKAPI F_NameTable::getNextTagTypeAndNumOrder(
	FLMUINT			uiType,
	FLMUINT *		puiNextPos,
	FLMUNICODE *	puzTagName,			// May be NULL
	char *			pszTagName,			// May be NULL
	FLMUINT			uiNameBufSize,
	FLMUINT *		puiTagNum,			// May be NULL
	FLMUINT *		puiDataType,		// May be NULL
	FLMUNICODE *	puzNamespace,		// May be NULL
	FLMUINT			uiNamespaceBufSize,
	FLMBOOL			bTruncatedNamesOk)
{
	RCODE				rc = NE_FLM_OK;
	FLM_TAG_INFO *	pTagInfo = NULL;
	FLMBOOL			bFound = FALSE;

	if (!m_bTablesSorted)
	{
		sortTags();
	}

	while (*puiNextPos < m_uiNumTags)
	{
		pTagInfo = m_ppSortedByTagTypeAndNum [*puiNextPos];
		if (pTagInfo->uiType == uiType)
		{
			bFound = TRUE;
			if (puiTagNum)
			{
				*puiTagNum = pTagInfo->uiTagNum;
			}
			if( puzTagName || pszTagName)
			{
				if (RC_BAD( rc = copyTagName( puzTagName, pszTagName,
									&uiNameBufSize,
									pTagInfo->puzTagName, bTruncatedNamesOk)))
				{
					goto Exit;
				}
			}

			if (puiDataType)
			{
				*puiDataType = pTagInfo->uiDataType;
			}
			
			if (puzNamespace)
			{
				if (RC_BAD( rc = copyTagName( puzNamespace, NULL,
								 &uiNamespaceBufSize,
								 pTagInfo->puzNamespace,
								 bTruncatedNamesOk)))
				{
					goto Exit;
				}
			}

			// Returned *puiNextPos should be the next one to retrieve.

			(*puiNextPos)++;
			break;
		}
		else if (pTagInfo->uiType > uiType)
		{
			break;
		}
		else
		{
			(*puiNextPos)++;
		}
	}

	if (!bFound)
	{
		rc = RC_SET( NE_FLM_EOF_HIT);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Get a tag name, number, etc. using type + tag name ordering.
		Tag name is returned as a UNICODE string or NATIVE string. If a
		non-NULL UNICODE string is passed in, it will be used.
		Otherwise, the NATIVE string will be used.
****************************************************************************/
RCODE FTKAPI F_NameTable::getNextTagTypeAndNameOrder(
	FLMUINT			uiType,
	FLMUINT *		puiNextPos,
	FLMUNICODE *	puzTagName,			// May be NULL
	char *			pszTagName,			// May be NULL
	FLMUINT			uiNameBufSize,
	FLMUINT *		puiTagNum,			// May be NULL
	FLMUINT *		puiDataType,		// May be NULL
	FLMUNICODE *	puzNamespace,		// May be NULL
	FLMUINT			uiNamespaceBufSize,
	FLMBOOL			bTruncatedNamesOk)
{
	RCODE				rc = NE_FLM_OK;
	FLM_TAG_INFO *	pTagInfo = NULL;
	FLMBOOL			bFound = FALSE;

	if (!m_bTablesSorted)
	{
		sortTags();
	}

	while (*puiNextPos < m_uiNumTags)
	{
		pTagInfo = m_ppSortedByTagTypeAndName [*puiNextPos];

		if (pTagInfo->uiType == uiType)
		{
			bFound = TRUE;
			if (puiTagNum)
			{
				*puiTagNum = pTagInfo->uiTagNum;
			}
			if (puzTagName || pszTagName)
			{
				if (RC_BAD( rc = copyTagName( puzTagName, pszTagName,
									&uiNameBufSize, pTagInfo->puzTagName,
									bTruncatedNamesOk)))
				{
					goto Exit;
				}
			}
			
			if (puiDataType)
			{
				*puiDataType = pTagInfo->uiDataType;
			}
			
			if (puzNamespace)
			{
				if (RC_BAD( rc = copyTagName( puzNamespace, NULL,
								 &uiNamespaceBufSize,
								 pTagInfo->puzNamespace,
								 bTruncatedNamesOk)))
				{
					goto Exit;
				}
			}

			// Returned *puiNextPos should be the next one to retrieve.

			(*puiNextPos)++;
			break;
		}
		else if (pTagInfo->uiType > uiType)
		{
			break;
		}
		else
		{
			(*puiNextPos)++;
		}
	}
	if (!bFound)
	{
		rc = RC_SET( NE_FLM_EOF_HIT);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Get a tag name from its tag type and number.  Tag name is returned as a
		UNICODE string or NATIVE string. If a non-NULL UNICODE string is passed
		in, it will be used.  Otherwise, the NATIVE string will be used.
****************************************************************************/
RCODE FTKAPI F_NameTable::getFromTagTypeAndNum(
	FLMUINT			uiType,
	FLMUINT			uiTagNum,
	FLMUNICODE *	puzTagName,				// May be NULL
	char *			pszTagName,				// May be NULL
	FLMUINT *		puiNameBufSize,		// May be NULL, returns # characters
	FLMUINT *		puiDataType,			// May be NULL
	FLMUNICODE *	puzNamespace,			// May be NULL
	char *			pszNamespace,			// May be NULL
	FLMUINT *		puiNamespaceBufSize,	// May be NULL, returns # characters
	FLMBOOL			bTruncatedNamesOk)
{
	RCODE				rc = NE_FLM_OK;
	FLM_TAG_INFO *	pTagInfo;
	FLMUNICODE *	puzTmpNamespace = NULL;
	FLMUNICODE *	puzTmpName = NULL;

	if ((pTagInfo = findTagByTypeAndNum( uiType, uiTagNum)) != NULL)
	{
		if( puzTagName || pszTagName)
		{
			if (RC_BAD( rc = copyTagName( puzTagName, pszTagName,
									puiNameBufSize,
									pTagInfo->puzTagName, bTruncatedNamesOk)))
			{
				goto Exit;
			}
		}
		else if (puiNameBufSize)
		{
			*puiNameBufSize = f_unilen( pTagInfo->puzTagName);
		}
		
		if (puiDataType)
		{
			*puiDataType = pTagInfo->uiDataType;
		}
		
		if (puzNamespace || pszNamespace)
		{
			if (RC_BAD( rc = copyTagName( puzNamespace, pszNamespace,
								puiNamespaceBufSize,
							 pTagInfo->puzNamespace,
							 bTruncatedNamesOk)))
			{
				goto Exit;
			}
		}
		else if (puiNamespaceBufSize)
		{
			*puiNamespaceBufSize = f_unilen( pTagInfo->puzNamespace);
		}
	}
	else
	{
		rc = RC_SET( NE_FLM_NOT_FOUND);
		goto Exit;
	}

Exit:

	if (puzTmpName)
	{
		f_free( &puzTmpName);
	}

	if (puzTmpNamespace)
	{
		f_free( &puzTmpNamespace);
	}

	return( rc);
}

/****************************************************************************
Desc:	Get a tag number from its tag name and type.  Tag name is passed
		in as a UNICODE or NATIVE string. If a non-NULL UNICODE string is
		passed in, it will be used.  Otherwise, the NATIVE string will
		be used.
****************************************************************************/
RCODE FTKAPI F_NameTable::getFromTagTypeAndName(
	FLMUINT					uiType,
	const FLMUNICODE *	puzTagName,
	const char *			pszTagName,
	FLMBOOL					bMatchNamespace,
	const FLMUNICODE *	puzNamespace,
	FLMUINT *				puiTagNum,		// May be NULL
	FLMUINT *				puiDataType)	// May be NULL
{
	RCODE						rc = NE_FLM_OK;
	FLM_TAG_INFO *			pTagInfo;
	FLMBOOL					bAmbiguous;

	if ((pTagInfo = findTagByTypeAndName( uiType,
								puzTagName, pszTagName, bMatchNamespace,
								puzNamespace,
								&bAmbiguous)) != NULL)
	{
		if (puiTagNum)
		{
			*puiTagNum = pTagInfo->uiTagNum;
		}
		
		if (puiDataType)
		{
			*puiDataType = pTagInfo->uiDataType;
		}
		
		if (bAmbiguous)
		{
			rc = RC_SET( NE_FLM_MULTIPLE_MATCHES);
			goto Exit;
		}
	}
	else
	{
		rc = RC_SET( NE_FLM_NOT_FOUND);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Insert a tag info structure into the sorted tables at the specified
		positions.
****************************************************************************/
RCODE F_NameTable::insertTagInTables(
	FLM_TAG_INFO *	pTagInfo,
	FLMUINT			uiTagTypeAndNameTblInsertPos,
	FLMUINT			uiTagTypeAndNumTblInsertPos)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiLoop;

	// See if we need to resize the tables.  Start at 256.  Double each
	// time up to 2048.  Then just add 2048 at a time.

	if (m_uiNumTags == m_uiTblSize)
	{
		FLMUINT	uiNewSize;

		if (!m_uiTblSize)
		{
			uiNewSize = 256;
		}
		else if (m_uiTblSize < 2048)
		{
			uiNewSize = m_uiTblSize * 2;
		}
		else
		{
			uiNewSize = m_uiTblSize + 2048;
		}

		if (RC_BAD( rc = reallocSortTables( uiNewSize)))
		{
			goto Exit;
		}
	}

	// Insert into the sorted-by-tag-type-and-name table

	uiLoop = m_uiNumTags;
	while (uiLoop > uiTagTypeAndNameTblInsertPos)
	{
		m_ppSortedByTagTypeAndName [uiLoop] =
			m_ppSortedByTagTypeAndName [uiLoop - 1];
		uiLoop--;
	}
	m_ppSortedByTagTypeAndName [uiTagTypeAndNameTblInsertPos] = pTagInfo;

	// Insert into the sorted-by-tag-type-and-num table

	uiLoop = m_uiNumTags;
	while (uiLoop > uiTagTypeAndNumTblInsertPos)
	{
		m_ppSortedByTagTypeAndNum [uiLoop] =
			m_ppSortedByTagTypeAndNum [uiLoop - 1];
		uiLoop--;
	}
	m_ppSortedByTagTypeAndNum [uiTagTypeAndNumTblInsertPos] = pTagInfo;

	// Increment the total number of tags

	m_uiNumTags++;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Add a tag to the table.  Tag name is passed in as a UNICODE string or
		NATIVE string. If a non-NULL UNICODE string is passed in, it will
		be used.  Otherwise, the NATIVE string will be used.
****************************************************************************/
RCODE FTKAPI F_NameTable::addTag(
	FLMUINT			uiType,
	FLMUNICODE *	puzTagName,
	const char *	pszTagName,
	FLMUINT			uiTagNum,
	FLMUINT			uiDataType,
	FLMUNICODE *	puzNamespace,
	FLMBOOL			bCheckDuplicates)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiTagTypeAndNameTblInsertPos;
	FLMUINT			uiTagTypeAndNumTblInsertPos;
	FLM_TAG_INFO *	pTagInfo;
	FLMBOOL			bAmbiguous;

	// Must have a non-NULL tag name.  Use UNICODE string if it is
	// non-NULL.  Otherwise, use NATIVE string.

	if (puzTagName && *puzTagName)
	{
		pszTagName = NULL;
	}
	else if (pszTagName && *pszTagName)
	{
		puzTagName = NULL;
	}
	else
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_INVALID_PARM);
		goto Exit;
	}

	// Tag number of zero not allowed.

	if (!uiTagNum)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_INVALID_PARM);
		goto Exit;
	}

	// Tables must be sorted in order for this to work

	if (bCheckDuplicates)
	{

		// Make sure that the tag type + name is not already used.

		if (findTagByTypeAndName( uiType, puzTagName, pszTagName,
						TRUE, puzNamespace,
						&bAmbiguous, &uiTagTypeAndNameTblInsertPos))
		{
			rc = RC_SET( NE_FLM_EXISTS);
			goto Exit;
		}

		// Make sure that the tag type + number is not already used.

		if (findTagByTypeAndNum( uiType, uiTagNum,
						&uiTagTypeAndNumTblInsertPos))
		{
			rc = RC_SET( NE_FLM_EXISTS);
			goto Exit;
		}
	}
	else
	{
		uiTagTypeAndNameTblInsertPos = uiTagTypeAndNumTblInsertPos = m_uiNumTags;
		m_bTablesSorted = FALSE;
	}

	// Create a new tag info structure.

	if (RC_BAD( rc = allocTag( uiType, puzTagName, pszTagName, uiTagNum,
								uiDataType, puzNamespace, &pTagInfo)))
	{
		goto Exit;
	}

	// Insert the tag structure into the appropriate places in the
	// sorted tables.

	if (RC_BAD( rc = insertTagInTables( pTagInfo,
							uiTagTypeAndNameTblInsertPos,
							uiTagTypeAndNumTblInsertPos)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Sort the tag tables according to their respective criteria.
****************************************************************************/
void F_NameTable::sortTags( void)
{
	if (!m_bTablesSorted && m_uiNumTags > 1)
	{
		sortTagTbl( m_ppSortedByTagTypeAndName, 0, m_uiNumTags - 1,
				compareTagTypeAndName);
		sortTagTbl( m_ppSortedByTagTypeAndNum, 0, m_uiNumTags - 1,
				compareTagTypeAndNum);
	}
	m_bTablesSorted = TRUE;
}

/****************************************************************************
Desc:	Remove a tag from the table
****************************************************************************/
void FTKAPI F_NameTable::removeTag(
	FLMUINT			uiType,
	FLMUINT			uiTagNum)
{
	FLM_TAG_INFO *	pTagInfo;
	FLMUINT			uiTagTypeAndNameTblPos;
	FLMUINT			uiTagTypeAndNumTblPos;
	FLMBOOL			bAmbiguous;
	FLMBOOL			bMatchNamespace;
	FLMUNICODE *	puzNamespace;

	if ((pTagInfo = findTagByTypeAndNum( uiType, uiTagNum,
								&uiTagTypeAndNumTblPos)) != NULL)
	{
		puzNamespace = pTagInfo->puzNamespace;
		bMatchNamespace = TRUE;
		
		if (findTagByTypeAndName( uiType, pTagInfo->puzTagName,
								NULL, bMatchNamespace,
								puzNamespace, &bAmbiguous,
								&uiTagTypeAndNameTblPos) == NULL)
		{

			// It should have been in the name table too!

			f_assert( 0);
		}

		// Shift everything in the sorted number table that is above
		// the found position down one.

		if (uiTagTypeAndNumTblPos < m_uiNumTags - 1)
		{
			f_memmove( &m_ppSortedByTagTypeAndNum [uiTagTypeAndNumTblPos],
						  &m_ppSortedByTagTypeAndNum [uiTagTypeAndNumTblPos + 1],
						  sizeof( FLM_TAG_INFO *) *
						  (m_uiNumTags - 1 - uiTagTypeAndNumTblPos));
		}

		// Shift everything in the sorted name table that is above
		// the found position down one.

		if (uiTagTypeAndNameTblPos < m_uiNumTags - 1)
		{
			f_memmove( &m_ppSortedByTagTypeAndName [uiTagTypeAndNameTblPos],
						  &m_ppSortedByTagTypeAndName [uiTagTypeAndNameTblPos + 1],
						  sizeof( FLM_TAG_INFO *) *
						  (m_uiNumTags - 1 - uiTagTypeAndNameTblPos));
		}
		m_uiNumTags--;
	}
}

/****************************************************************************
Desc:	Create a clone of this name table
****************************************************************************/
RCODE FTKAPI F_NameTable::cloneNameTable(
	IF_NameTable **		ppNewNameTable)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT				uiLoop;
	FLM_TAG_INFO *		pTagInfo;
	FLMUINT				uiPoolBlkSize;
	F_NameTable *		pNewNameTable = NULL;
	
	// Allocate a new name table 
	
	if( (pNewNameTable = f_new F_NameTable) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}

	// Set the pool size to be as optimal as possible.

	uiPoolBlkSize = m_uiMemoryAllocated / 8;
	if (uiPoolBlkSize < 1024)
	{
		uiPoolBlkSize = 1024;
	}
	else if (uiPoolBlkSize > 65536)
	{
		uiPoolBlkSize = 65536;
	}
	
	pNewNameTable->clearTable( uiPoolBlkSize);

	// Pre-allocate exactly enough table space

	if (RC_BAD( rc = pNewNameTable->reallocSortTables( m_uiNumTags)))
	{
		goto Exit;
	}

	// Add all of the tags

	for (uiLoop = 0; uiLoop < m_uiNumTags; uiLoop++)
	{
		pTagInfo = m_ppSortedByTagTypeAndNum [uiLoop];
		if (RC_BAD( rc = pNewNameTable->addTag( 
			pTagInfo->uiType, pTagInfo->puzTagName, NULL, pTagInfo->uiTagNum,
			pTagInfo->uiDataType, pTagInfo->puzNamespace, FALSE)))
		{
			goto Exit;
		}
	}

	pNewNameTable->sortTags();
	*ppNewNameTable = pNewNameTable;
	pNewNameTable = NULL;

Exit:

	if( pNewNameTable)
	{
		pNewNameTable->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Increment use count on this object.
****************************************************************************/
FLMINT FTKAPI F_NameTable::AddRef( void)
{
	return( f_atomicInc( &m_refCnt));
}

/****************************************************************************
Desc:	Decrement the use count and delete if use count goes to zero.
****************************************************************************/
FLMINT FTKAPI F_NameTable::Release( void)
{
	FLMINT		iRefCnt;

	if ((iRefCnt = f_atomicDec( &m_refCnt)) == 0)
	{
		delete this;
	}

	return( iRefCnt);
}

#else

/****************************************************************************
Desc:
****************************************************************************/
int ftkntabDummy( void)
{
	return( 0);
}

#endif
