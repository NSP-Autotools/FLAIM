//-------------------------------------------------------------------------
// Desc:	Name table class
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

#include "flaimsys.h"

typedef FLMINT (*	TAG_COMPARE_FUNC)(
	FLM_TAG_INFO *			pTagInfo1,
	FLM_TAG_INFO *			pTagInfo2);

FSTATIC FLMINT tagNameCompare(
	const FLMUNICODE *	puzName1,
	const char *			pszName1,
	const FLMUNICODE *	puzName2);

FSTATIC FLMINT compareTagNumOnly(
	FLM_TAG_INFO *			pTagInfo1,
	FLM_TAG_INFO *			pTagInfo2);

FSTATIC FLMINT compareTagNameOnly(
	FLM_TAG_INFO *			pTagInfo1,
	FLM_TAG_INFO *			pTagInfo2);

FSTATIC FLMINT compareTagTypeAndName(
	FLM_TAG_INFO *			pTagInfo1,
	FLM_TAG_INFO *			pTagInfo2);

/*
**  WARNING:	ANY CHANGES MADE TO THE FlmDictTags TABLE MUST BE REFLECTED
**					IN THE TAG DEFINES FOUND IN FLAIM.H
*/

typedef struct FlmDictTag
{
	const char *	pszTagName;
	FLMUINT			uiTagNum;
	FLMUINT			uiFieldType;
} DICT_TAG_NAME;

DICT_TAG_NAME FlmDictTags[] =
{
	{FLM_FIELD_TAG_NAME, FLM_FIELD_TAG, FLM_TEXT_TYPE},
	{FLM_INDEX_TAG_NAME, FLM_INDEX_TAG, FLM_TEXT_TYPE},
	{FLM_TYPE_TAG_NAME, FLM_TYPE_TAG, FLM_TEXT_TYPE},
	{FLM_COMMENT_TAG_NAME, FLM_COMMENT_TAG, FLM_TEXT_TYPE},
	{FLM_CONTAINER_TAG_NAME, FLM_CONTAINER_TAG, FLM_TEXT_TYPE},
	{FLM_LANGUAGE_TAG_NAME, FLM_LANGUAGE_TAG, FLM_TEXT_TYPE},
	{FLM_OPTIONAL_TAG_NAME, FLM_OPTIONAL_TAG, FLM_TEXT_TYPE},
	{FLM_UNIQUE_TAG_NAME, FLM_UNIQUE_TAG, FLM_TEXT_TYPE},
	{FLM_KEY_TAG_NAME, FLM_KEY_TAG, FLM_TEXT_TYPE},
	{FLM_REFS_TAG_NAME, FLM_REFS_TAG, FLM_TEXT_TYPE},
	{FLM_ENCDEF_TAG_NAME, FLM_ENCDEF_TAG, FLM_TEXT_TYPE},
	{FLM_DELETE_TAG_NAME, FLM_DELETE_TAG, FLM_TEXT_TYPE},
	{FLM_BLOCK_CHAIN_TAG_NAME, FLM_BLOCK_CHAIN_TAG, FLM_NUMBER_TYPE},
	{FLM_AREA_TAG_NAME, FLM_AREA_TAG, FLM_TEXT_TYPE},
	{FLM_STATE_TAG_NAME, FLM_STATE_TAG, FLM_TEXT_TYPE},
	{FLM_BLOB_TAG_NAME, FLM_BLOB_TAG, FLM_TEXT_TYPE},
	{FLM_THRESHOLD_TAG_NAME, FLM_THRESHOLD_TAG, FLM_TEXT_TYPE},
	{FLM_SUFFIX_TAG_NAME, FLM_SUFFIX_TAG, FLM_TEXT_TYPE},
	{FLM_SUBDIRECTORY_TAG_NAME, FLM_SUBDIRECTORY_TAG, FLM_TEXT_TYPE},
	{FLM_RESERVED_TAG_NAME, FLM_RESERVED_TAG, FLM_TEXT_TYPE},
	{FLM_SUBNAME_TAG_NAME, FLM_SUBNAME_TAG, FLM_TEXT_TYPE},
	{FLM_NAME_TAG_NAME, FLM_NAME_TAG, FLM_TEXT_TYPE},
	{FLM_BASE_TAG_NAME, FLM_BASE_TAG, FLM_TEXT_TYPE},
	{FLM_CASE_TAG_NAME, FLM_CASE_TAG, FLM_TEXT_TYPE},
	{FLM_COMBINATIONS_TAG_NAME, FLM_COMBINATIONS_TAG, FLM_TEXT_TYPE},
	{FLM_COUNT_TAG_NAME, FLM_COUNT_TAG, FLM_TEXT_TYPE},
	{FLM_POSITIONING_TAG_NAME, FLM_POSITIONING_TAG, FLM_TEXT_TYPE},
	{FLM_PAIRED_TAG_NAME, FLM_PAIRED_TAG, FLM_TEXT_TYPE},
	{FLM_PARENT_TAG_NAME, FLM_PARENT_TAG, FLM_TEXT_TYPE},
	{FLM_POST_TAG_NAME, FLM_POST_TAG, FLM_TEXT_TYPE},
	{FLM_REQUIRED_TAG_NAME, FLM_REQUIRED_TAG, FLM_TEXT_TYPE},
	{FLM_USE_TAG_NAME, FLM_USE_TAG, FLM_TEXT_TYPE},
	{FLM_FILTER_TAG_NAME, FLM_FILTER_TAG, FLM_TEXT_TYPE},
	{FLM_LIMIT_TAG_NAME, FLM_LIMIT_TAG, FLM_TEXT_TYPE},
	{FLM_DICT_TAG_NAME, FLM_DICT_TAG, FLM_TEXT_TYPE},
	{FLM_RECINFO_TAG_NAME, FLM_RECINFO_TAG, FLM_TEXT_TYPE},
	{FLM_DRN_TAG_NAME, FLM_DRN_TAG, FLM_TEXT_TYPE},
	{FLM_DICT_SEQ_TAG_NAME, FLM_DICT_SEQ_TAG, FLM_TEXT_TYPE},
	{FLM_LAST_CONTAINER_INDEXED_TAG_NAME, FLM_LAST_CONTAINER_INDEXED_TAG, FLM_NUMBER_TYPE},
	{FLM_LAST_DRN_INDEXED_TAG_NAME, FLM_LAST_DRN_INDEXED_TAG, FLM_NUMBER_TYPE},
	{FLM_ONLINE_TRANS_ID_TAG_NAME, FLM_ONLINE_TRANS_ID_TAG, FLM_NUMBER_TYPE},
	{NULL, 0}
};

/*
**  WARNING:	ANY CHANGES MADE TO THE FlmDictTags TABLE MUST BE REFLECTED
**					IN THE TAG DEFINES FOUND IN FLAIM.H
*/

FSTATIC void sortTagTbl(
	FLM_TAG_INFO **	ppTagInfoTbl,
	FLMUINT				uiLowerBounds,
	FLMUINT				uiUpperBounds,
	TAG_COMPARE_FUNC	fnTagCompare);

/****************************************************************************
Desc:	Constructor
****************************************************************************/
F_NameTable::F_NameTable()
{
	m_pool.poolInit( 1024);
	m_ppSortedByTagName = NULL;
	m_ppSortedByTagNum = NULL;
	m_ppSortedByTagTypeAndName = NULL;
	m_uiTblSize = 0;
	m_uiNumTags = 0;
	m_bTablesSorted = FALSE;
}

/****************************************************************************
Desc:	Destructor
****************************************************************************/
F_NameTable::~F_NameTable()
{
	clearTable();
	m_pool.poolFree();
}

/****************************************************************************
Desc:	Free everything in the table
****************************************************************************/
void F_NameTable::clearTable( void)
{
	m_pool.poolFree();
	m_pool.poolInit( 1024);

	// NOTE: Only one allocation is used for m_ppSortedByTagName,
	// m_ppSortedByTagNum, and m_ppSortedByTagTypeAndName - there is no
	// need to free m_ppSortedByTagNum and m_ppSortedByTagTypeAndName.

	if (m_ppSortedByTagName)
	{
		f_free( &m_ppSortedByTagName);
		m_ppSortedByTagNum = NULL;
		m_ppSortedByTagTypeAndName = NULL;
		m_uiTblSize = 0;
		m_uiNumTags = 0;
	}
}

/****************************************************************************
Desc: Compare two tag names.  Name1 can be NATIVE or UNICODE.  If a non-NULL
		UNICODE string is passed, it will be used.  Otherwise, the NATIVE
		string will be used.
Note:	Comparison is case insensitive for the ASCII characters A-Z.
****************************************************************************/
FSTATIC FLMINT tagNameCompare(
	const FLMUNICODE *	puzName1,	// If NULL, use pszName1 for comparison
	const char *			pszName1,
	const FLMUNICODE *	puzName2)
{
	FLMUNICODE		uzChar1;
	FLMUNICODE		uzChar2;

	if (puzName1)
	{
		for (;;)
		{
			uzChar1 = *puzName1;
			uzChar2 = *puzName2;

			// Convert to lower case for comparison.

			if (uzChar1 >= 'A' && uzChar1 <= 'Z')
			{
				uzChar1 = uzChar1 - 'A' + 'a';
			}
			if (uzChar2 >= 'A' && uzChar2 <= 'Z')
			{
				uzChar2 = uzChar2 - 'A' + 'a';
			}

			if (!uzChar1 || !uzChar2 || uzChar1 != uzChar2)
			{
				break;
			}

			puzName1++;
			puzName2++;
		}
	}
	else
	{
		for (;;)
		{
			uzChar1 = (FLMUNICODE)*pszName1;
			uzChar2 = *puzName2;

			// Convert to lower case for comparison.

			if (uzChar1 >= 'A' && uzChar1 <= 'Z')
			{
				uzChar1 = uzChar1 - 'A' + 'a';
			}
			if (uzChar2 >= 'A' && uzChar2 <= 'Z')
			{
				uzChar2 = uzChar2 - 'A' + 'a';
			}

			if (!uzChar1 || !uzChar2 || uzChar1 != uzChar2)
			{
				break;
			}

			pszName1++;
			puzName2++;
		}
	}

	if (uzChar1)
	{
		return( (FLMINT)((uzChar2 && uzChar1 < uzChar2)
								? (FLMINT)-1
								: (FLMINT)1));
	}
	else if (uzChar2)
	{
		return( -1);
	}
	else
	{
		return( 0);
	}
}

/****************************************************************************
Desc: Lookup a tag by tag name.  Tag name is passed in as a UNICODE string or
		a NATIVE string.  If a non-NULL UNICODE string is passed in, it will
		be used.  Otherwise, the NATIVE string will be used.
****************************************************************************/
FLM_TAG_INFO * F_NameTable::findTagByName(
	const FLMUNICODE *	puzTagName,
	const char *			pszTagName,
	FLMUINT *				puiInsertPos)
{
	FLM_TAG_INFO *	pTagInfo = NULL;
	FLMUINT			uiTblSize;
	FLMUINT			uiLow;
	FLMUINT			uiMid;
	FLMUINT			uiHigh;
	FLMINT			iCmp;

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
		iCmp = tagNameCompare( puzTagName, pszTagName,
						m_ppSortedByTagName [uiMid]->puzTagName);
		if (iCmp == 0)
		{

			// Found Match
			
			pTagInfo = m_ppSortedByTagName [uiMid];
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

	return( pTagInfo);
}

/****************************************************************************
Desc: Lookup a tag by tag number.
****************************************************************************/
FLM_TAG_INFO * F_NameTable::findTagByNum(
	FLMUINT		uiTagNum,
	FLMUINT *	puiInsertPos
	)
{
	FLM_TAG_INFO *	pTagInfo = NULL;
	FLMUINT			uiTblSize;
	FLMUINT			uiLow;
	FLMUINT			uiMid;
	FLMUINT			uiHigh;
	FLMUINT			uiTblTagNum;

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
		uiTblTagNum = m_ppSortedByTagNum [uiMid]->uiTagNum;
		if (uiTagNum == uiTblTagNum)
		{

			// Found Match
			
			pTagInfo = m_ppSortedByTagNum [uiMid];
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
				*puiInsertPos = (uiTagNum < uiTblTagNum)
									 ? uiMid
									 : uiMid + 1;
			}
			goto Exit;
		}

		if (uiTagNum < uiTblTagNum)
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
	const FLMUNICODE *	puzTagName,
	const char *			pszTagName,
	FLMUINT					uiType,
	FLMUINT *				puiInsertPos)
{
	FLM_TAG_INFO *	pTagInfo = NULL;
	FLMUINT			uiTblType;
	FLMUINT			uiTblSize;
	FLMUINT			uiLow;
	FLMUINT			uiMid;
	FLMUINT			uiHigh;
	FLMINT			iCmp;

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

			// Found Match
			
			pTagInfo = m_ppSortedByTagTypeAndName [uiMid];
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

	return( pTagInfo);
}

/****************************************************************************
Desc:	Copy a tag name to a UNICODE or NATIVE buffer.  Truncate if necessary.
		If a non-NULL UNICODE string is passed in, it will be populated.
		Otherwise, the NATIVE string will be populated.
****************************************************************************/
void F_NameTable::copyTagName(
	FLMUNICODE *			puzDestTagName,
	char *					pszDestTagName,
	FLMUINT					uiDestBufSize,
	const FLMUNICODE *	puzSrcTagName)
{
	if (puzDestTagName)
	{

		// Decrement name buffer size by sizeof( FLMUNICODE) to allow for a
		// terminating NULL character. uiDestBufSize better be at list big
		// enough for a null terminating character.

		flmAssert( uiDestBufSize >= sizeof( FLMUNICODE));
		uiDestBufSize -= sizeof( FLMUNICODE);

		// Copy the name to the NATIVE buffer.  Non-Ascii UNICODE characters
		// will be returned as question marks (?).

		while (uiDestBufSize >= sizeof( FLMUNICODE) && *puzSrcTagName)
		{
			*puzDestTagName++ = *puzSrcTagName;
			uiDestBufSize -= sizeof( FLMUNICODE);
			puzSrcTagName++;
		}
		*puzDestTagName = 0;
	}
	else
	{
		// Decrement name buffer size by one to allow for a terminating
		// NULL character. uiDestBufSize better be at list big
		// enough for a null terminating character.

		flmAssert( uiDestBufSize);
		uiDestBufSize--;

		// Copy the name to the NATIVE buffer.  Non-Ascii UNICODE characters
		// will be returned as question marks (?).

		while (uiDestBufSize && *puzSrcTagName)
		{
			if (*puzSrcTagName <= 127)
			{
				*pszDestTagName++ = (FLMBYTE)*puzSrcTagName;
			}
			else
			{
				*pszDestTagName++ = '?';
			}
			uiDestBufSize--;
			puzSrcTagName++;
		}
		*pszDestTagName = 0;
	}
}

/***************************************************************************
Desc:	Swap two entries in tag info table during sort.
*****************************************************************************/
FINLINE void tagInfoSwap(
	FLM_TAG_INFO **	ppTagInfoTbl,
	FLMUINT				uiPos1,
	FLMUINT				uiPos2)
{
	FLM_TAG_INFO *	pTmpTagInfo = ppTagInfoTbl [uiPos1];

	ppTagInfoTbl [uiPos1] = ppTagInfoTbl [uiPos2];
	ppTagInfoTbl [uiPos2] = pTmpTagInfo;
}

/***************************************************************************
Desc:	Comparison function for sorting by tag number.
****************************************************************************/
FSTATIC FLMINT compareTagNumOnly(
	FLM_TAG_INFO *	pTagInfo1,
	FLM_TAG_INFO *	pTagInfo2
	)
{
	if (pTagInfo1->uiTagNum < pTagInfo2->uiTagNum)
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
Desc:	Comparison function for sorting by tag name.
****************************************************************************/
FSTATIC FLMINT compareTagNameOnly(
	FLM_TAG_INFO *	pTagInfo1,
	FLM_TAG_INFO *	pTagInfo2
	)
{
	return (tagNameCompare( pTagInfo1->puzTagName, NULL,
				pTagInfo2->puzTagName));
}

/***************************************************************************
Desc:	Comparison function for sorting by tag type and name.
****************************************************************************/
FSTATIC FLMINT compareTagTypeAndName(
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
	else
	{
		return (tagNameCompare( pTagInfo1->puzTagName, NULL,
						pTagInfo2->puzTagName));
	}
}

/***************************************************************************
Desc:	Sort an array of SCACHE pointers by their block address.
****************************************************************************/
FSTATIC void sortTagTbl(
	FLM_TAG_INFO **	ppTagInfoTbl,
	FLMUINT				uiLowerBounds,
	FLMUINT				uiUpperBounds,
	TAG_COMPARE_FUNC	fnTagCompare
	)
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
Desc:	Allocate a new tag info structure and set it up.
****************************************************************************/
RCODE F_NameTable::allocTag(
	const FLMUNICODE *	puzTagName,
	const char *			pszTagName,
	FLMUINT					uiTagNum,
	FLMUINT					uiType,
	FLMUINT					uiSubType,
	FLM_TAG_INFO **		ppTagInfo)
{
	RCODE						rc = FERR_OK;
	void *					pvMark;
	FLM_TAG_INFO *			pTagInfo;
	FLMUINT					uiNameSize;
	FLMUNICODE *			puzTmp;

	// Create a new tag info structure.

	pvMark = m_pool.poolMark();
	
	if( RC_BAD( rc = m_pool.poolCalloc( sizeof( FLM_TAG_INFO), 
		(void **)&pTagInfo)))
	{
		goto Exit;
	}

	// Allocate the space for the tag name.

	if (puzTagName)
	{
		uiNameSize = (f_unilen( puzTagName) + 1) * sizeof( FLMUNICODE);
		
		if( RC_BAD( rc = m_pool.poolAlloc( uiNameSize,
			(void **)&pTagInfo->puzTagName)))
		{
			goto Exit;
		}
		
		f_memcpy( (void *)pTagInfo->puzTagName, puzTagName, uiNameSize);
	}
	else
	{
		uiNameSize = (f_strlen( pszTagName) + 1) * sizeof( FLMUNICODE);
		
		if( RC_BAD( rc = m_pool.poolAlloc( uiNameSize, 
			(void **)&pTagInfo->puzTagName)))
		{
			goto Exit;
		}
		
		puzTmp = (FLMUNICODE *)pTagInfo->puzTagName;
		
		while (*pszTagName)
		{
			*puzTmp++ = (FLMUNICODE)*pszTagName;
			pszTagName++;
		}
		*puzTmp = 0;
	}
	pTagInfo->uiTagNum = uiTagNum;
	pTagInfo->uiType = uiType;
	pTagInfo->uiSubType = uiSubType;

Exit:

	if (RC_BAD( rc))
	{
		m_pool.poolReset( pvMark);
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
	RCODE					rc = FERR_OK;
	FLM_TAG_INFO **	ppNewTbl;

	if( RC_BAD( rc = f_alloc( 
		sizeof( FLM_TAG_INFO *) * (uiNewTblSize * 3), &ppNewTbl)))
	{
		goto Exit;
	}

	// Copy the old tables into the new.

	if (m_uiNumTags)
	{
		f_memcpy( ppNewTbl, m_ppSortedByTagName,
						sizeof( FLM_TAG_INFO *) * m_uiNumTags);
		f_memcpy( &ppNewTbl [uiNewTblSize], m_ppSortedByTagNum,
						sizeof( FLM_TAG_INFO *) * m_uiNumTags);
		f_memcpy( &ppNewTbl [uiNewTblSize + uiNewTblSize],
						m_ppSortedByTagTypeAndName,
						sizeof( FLM_TAG_INFO *) * m_uiNumTags);
		f_free( &m_ppSortedByTagName);
	}
	m_ppSortedByTagName = ppNewTbl;
	m_ppSortedByTagNum = &ppNewTbl [uiNewTblSize];
	m_ppSortedByTagTypeAndName = &ppNewTbl [uiNewTblSize + uiNewTblSize];
	m_uiTblSize = uiNewTblSize;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Initialize a name table from a database.
****************************************************************************/
RCODE F_NameTable::setupFromDb(
	HFDB	hDb	// May be HFDB_NULL - means only get reserved tags.
	)
{
	RCODE				rc = FERR_OK;
	FDB *				pDb = (FDB *)hDb;
	LFILE *			pLFile;
	FLMBOOL			bStartedTrans = FALSE;
	FlmRecord *		pRec = NULL;
	FLMUINT			uiDrn;
	FLMUINT			uiLoop;
	BTSK				StackBuf [BH_MAX_LEVELS];
	FLMBOOL			bStackInitialized = FALSE;
	BTSK *			pStack;
	FLMBYTE			ucKeyBuf [8];
	FLMBYTE			ucDrnBuf [8];
	void *			pvField;
	FLMUNICODE		uzName [60];
	FLMUNICODE *	puzName = &uzName [0];
	FLMUINT			uiNameLen = sizeof( uzName);
	FLMUINT			uiLen;
	FLMUINT			uiSubType;

	// Is this a client/server handle

	if( hDb != HFDB_NULL && IsInCSMode( hDb))
	{
		fdbInitCS( pDb);

		CS_CONTEXT *	pCSContext = pDb->pCSContext;
		FCL_WIRE			Wire( pCSContext, pDb);

		// Send a request to get the name table.

		if( RC_BAD( rc = Wire.sendOp(	FCS_OPCLASS_DATABASE, 
			FCS_OP_GET_NAME_TABLE)))
		{
			goto ExitCS;
		}

		if (RC_BAD( rc = Wire.sendTerminate()))
		{
			goto Transmission_Error;
		}

		// Set the name table

		Wire.setNameTable( this);
	
		// Read the response.

		if (RC_BAD( rc = Wire.read()))
		{
			goto Transmission_Error;
		}

		if (RC_BAD( rc = Wire.getRCode()))
		{
			goto ExitCS;
		}
		goto ExitCS;

Transmission_Error:

		pCSContext->bConnectionGood = FALSE;
		goto ExitCS;

ExitCS:

		fdbExit( pDb);
		Wire.setNameTable( NULL);
		goto Exit;
	}

	// Clean out all existing tags, if any.

	clearTable();

	if (hDb != HFDB_NULL)
	{

		// Start a read transaction, if one not going.

		if (pDb->uiTransType == FLM_NO_TRANS)
		{
			if (RC_BAD( rc = FlmDbTransBegin( hDb, FLM_READ_TRANS, 0)))
			{
				goto Exit;
			}
			bStartedTrans = TRUE;
		}

		// Get the LFILE for the dictionary

		if (RC_BAD( rc = fdictGetContainer( pDb->pDict, FLM_DICT_CONTAINER,
										&pLFile)))
		{
			goto Exit;
		}

		// See what the next DRN in the dictionary container will be

		uiDrn = 0;
		if (RC_BAD( rc = FSGetNextDrn( pDb, pLFile, FALSE, &uiDrn)))
		{
			goto Exit;
		}
	}
	else
	{
		uiDrn = 0;
	}

	// Count the number of reserved tags.

	for (uiLoop = 0; FlmDictTags [uiLoop].pszTagName; uiLoop++)
	{
		;
	}

	// Preallocate space so we don't have to do it over and over.

	if (RC_BAD( rc = reallocSortTables( uiLoop + uiDrn)))
	{
		goto Exit;
	}

	// Add in all of the reserved dictionary tags.

	for (uiLoop = 0; FlmDictTags [uiLoop].pszTagName; uiLoop++)
	{
		if( RC_BAD( rc = addTag( NULL, FlmDictTags [uiLoop].pszTagName,
			FlmDictTags [uiLoop].uiTagNum, FLM_FIELD_TAG,
			FlmDictTags [uiLoop].uiFieldType, FALSE)))
		{
			goto Exit;
		}
	}

	if( hDb != HFDB_NULL)
	{

		// Set up to read through all of the records. */

		FSInitStackCache( &StackBuf [0], BH_MAX_LEVELS);
		bStackInitialized = TRUE;
		pStack = StackBuf;
		pStack->pKeyBuf = ucKeyBuf;

		// Position to the first record in the B-Tree.

		f_UINT32ToBigEndian( 0, ucDrnBuf);
		if (RC_BAD( rc = FSBtSearch( pDb, pLFile, &pStack,
								ucDrnBuf, 4, ZERO_DOMAIN)))
		{
			goto Exit;
		}

		// Read through all of the records in the B-Tree.

		while (pStack->uiCmpStatus != BT_END_OF_DATA)
		{
			if ((uiDrn = f_bigEndianToUINT32( ucKeyBuf)) == DRN_LAST_MARKER)
			{
				break;
			}

			if (RC_BAD( rc = flmRcaRetrieveRec( pDb, NULL,
				FLM_DICT_CONTAINER, uiDrn, FALSE, NULL, NULL, &pRec)))
			{
				if (rc != FERR_NOT_FOUND)
				{
					goto Exit;
				}

				// NOTE: Deliberately not bringing in to cache if not
				// found there.

				if (RC_BAD( rc = FSReadElement( pDb, &pDb->TempPool, pLFile, uiDrn,
					pStack, TRUE, &pRec, NULL, NULL)))
				{
					goto Exit;
				}
			}

			pvField = pRec->root();

			// Get the unicode name length (does not include NULL terminator)

			if (RC_BAD( rc = pRec->getUnicodeLength( pvField, &uiLen)))
			{
				goto Exit;
			}

			// Account for NULL character.

			uiLen += sizeof( FLMUNICODE);

			// See if we need a larger buffer to get the name.

			if (uiLen > uiNameLen)
			{
				FLMUNICODE *	puzTmp;

				// Add enough for 60 more unicode characters.

				uiLen += (60 * sizeof( FLMUNICODE));

				if( RC_BAD( rc = f_alloc( uiLen, &puzTmp)))
				{
					goto Exit;
				}

				if (puzName != &uzName [0])
				{
					f_free( &puzName);
				}
				puzName = puzTmp;
				uiNameLen = uiLen;
			}

			// Get the tag name.

			uiLen = uiNameLen;
			if (RC_BAD( rc = pRec->getUnicode( pvField, puzName, &uiLen)))
			{
				goto Exit;
			}

			// Get the sub-type.

			if (pRec->getFieldID( pvField) == FLM_FIELD_TAG)
			{
				void *	pvFld = pRec->find( pvField, FLM_TYPE_TAG, 1, SEARCH_TREE);

				if (!pvFld ||
					 RC_BAD( DDGetFieldType( pRec, pvFld, &uiSubType)))
				{
					uiSubType = FLM_TEXT_TYPE;
				}
			}
			else
			{
				uiSubType = 0;
			}

			// Add tag to table, without sorting yet.

			if (RC_BAD( rc = addTag( puzName, NULL, uiDrn,
									pRec->getFieldID( pvField), uiSubType, FALSE)))
			{
				goto Exit;
			}

			// Position to the next record.

			if (RC_BAD( rc = FSBtNextElm( pDb, pLFile, pStack)))
			{
				if (rc == FERR_BT_END_OF_DATA)
				{
					rc = FERR_OK;
					break;
				}
				goto Exit;
			}
		}
	}

	sortTags();

Exit:

	if (RC_BAD( rc))
	{
		clearTable();
	}

	if (bStartedTrans)
	{
		(void)FlmDbTransAbort( hDb);
	}

	if (pRec)
	{
		pRec->Release();
	}

	if (puzName != &uzName [0])
	{
		f_free( &puzName);
	}

	if (bStackInitialized)
	{
		FSReleaseStackCache( StackBuf, BH_MAX_LEVELS, FALSE);
	}
	return( rc);
}

/****************************************************************************
Desc:	Get a tag name, number, etc. using tag number ordering.
		Tag name is returned as a UNICODE string or NATIVE string. If a
		non-NULL UNICODE string is passed in, it will be used.
		Otherwise, the NATIVE string will be used.
****************************************************************************/
FLMBOOL F_NameTable::getNextTagNumOrder(
	FLMUINT *		puiNextPos,
	FLMUNICODE *	puzTagName,
	char *			pszTagName,
	FLMUINT			uiNameBufSize,
	FLMUINT *		puiTagNum,
	FLMUINT *		puiType,
	FLMUINT *		puiSubType)
{
	FLM_TAG_INFO *	pTagInfo = NULL;

	if (!m_bTablesSorted)
	{
		sortTags();
	}

	if (*puiNextPos < m_uiNumTags)
	{
		pTagInfo = m_ppSortedByTagNum [*puiNextPos];
		if (puiTagNum)
		{
			*puiTagNum = pTagInfo->uiTagNum;
		}
		if (puiType)
		{
			*puiType = pTagInfo->uiType;
		}
		if (puiSubType)
		{
			*puiSubType = pTagInfo->uiSubType;
		}

		if( puzTagName || pszTagName)
		{
			copyTagName( puzTagName, pszTagName, uiNameBufSize,
								pTagInfo->puzTagName);
		}

		// Returned *puiNextPos should be the next one to retrieve.

		(*puiNextPos)++;
	}
	else
	{

		// Nothing more in list, but initialize return variables anyway.

		if (puzTagName)
		{
			*puzTagName = 0;
		}
		if (pszTagName)
		{
			*pszTagName = 0;
		}
		if (puiTagNum)
		{
			*puiTagNum = 0;
		}
		if (puiType)
		{
			*puiType = 0;
		}
		if (puiSubType)
		{
			*puiSubType = 0;
		}
	}
	return( (FLMBOOL)(pTagInfo ? (FLMBOOL)TRUE : (FLMBOOL)FALSE));
}

/****************************************************************************
Desc:	Get a tag name, number, etc. using tag name ordering.
		Tag name is returned as a UNICODE string or NATIVE string. If a
		non-NULL UNICODE string is passed in, it will be used.
		Otherwise, the NATIVE string will be used.
****************************************************************************/
FLMBOOL F_NameTable::getNextTagNameOrder(
	FLMUINT *		puiNextPos,
	FLMUNICODE *	puzTagName,
	char *			pszTagName,
	FLMUINT			uiNameBufSize,
	FLMUINT *		puiTagNum,
	FLMUINT *		puiType,
	FLMUINT *		puiSubType)
{
	FLM_TAG_INFO *	pTagInfo = NULL;

	if (!m_bTablesSorted)
	{
		sortTags();
	}

	if (*puiNextPos < m_uiNumTags)
	{
		pTagInfo = m_ppSortedByTagName [*puiNextPos];
		if (puiTagNum)
		{
			*puiTagNum = pTagInfo->uiTagNum;
		}
		if (puiType)
		{
			*puiType = pTagInfo->uiType;
		}
		if (puiSubType)
		{
			*puiSubType = pTagInfo->uiSubType;
		}

		if( puzTagName || pszTagName)
		{
			copyTagName( puzTagName, pszTagName, uiNameBufSize,
								pTagInfo->puzTagName);
		}

		// Returned *puiNextPos should be the next one to retrieve.

		(*puiNextPos)++;
	}
	else
	{

		// Nothing more in list, but initialize return variables anyway.

		if (puzTagName)
		{
			*puzTagName = 0;
		}
		if (pszTagName)
		{
			*pszTagName = 0;
		}
		if (puiTagNum)
		{
			*puiTagNum = 0;
		}
		if (puiType)
		{
			*puiType = 0;
		}
		if (puiSubType)
		{
			*puiSubType = 0;
		}
	}
	return( (FLMBOOL)(pTagInfo ? (FLMBOOL)TRUE : (FLMBOOL)FALSE));
}

/****************************************************************************
Desc:	Get a tag name and number from type.  Tag name is returned as a
		UNICODE string or NATIVE string. If a non-NULL UNICODE string is passed
		in, it will be used.  Otherwise, the NATIVE string will be used.
****************************************************************************/
FLMBOOL F_NameTable::getFromTagType(
	FLMUINT			uiType,
	FLMUINT *		puiNextPos,
	FLMUNICODE *	puzTagName,
	char *			pszTagName,
	FLMUINT			uiNameBufSize,
	FLMUINT *		puiTagNum,
	FLMUINT *		puiSubType)
{
	FLM_TAG_INFO *	pTagInfo = NULL;

	if (!m_bTablesSorted)
	{
		sortTags();
	}
	if (*puiNextPos == 0)
	{

		// A value of zero indicates we should try to find the first
		// one.

		(void)findTagByTypeAndName( NULL, "", uiType, puiNextPos);
		
		if (*puiNextPos < m_uiNumTags &&
			 m_ppSortedByTagTypeAndName [*puiNextPos]->uiType != uiType)
		{
			(*puiNextPos)++;
		}
	}

	if (*puiNextPos < m_uiNumTags &&
		 m_ppSortedByTagTypeAndName [*puiNextPos]->uiType == uiType)
	{
		pTagInfo = m_ppSortedByTagTypeAndName [*puiNextPos];

		if (puiTagNum)
		{
			*puiTagNum = pTagInfo->uiTagNum;
		}
		if (puiSubType)
		{
			*puiSubType = pTagInfo->uiSubType;
		}

		if( puzTagName || pszTagName)
		{
			copyTagName( puzTagName, pszTagName, uiNameBufSize,
								pTagInfo->puzTagName);
		}

		// Returned *puiNextPos should be the next one to retrieve, so that
		// it is not zero.

		(*puiNextPos)++;
	}
	else
	{

		// Type was not found, but initialize return variables anyway.

		if (puzTagName)
		{
			*puzTagName = 0;
		}
		if (pszTagName)
		{
			*pszTagName = 0;
		}
		if (puiTagNum)
		{
			*puiTagNum = 0;
		}
		if (puiSubType)
		{
			*puiSubType = 0;
		}
	}

	return( (FLMBOOL)(pTagInfo ? (FLMBOOL)TRUE : (FLMBOOL)FALSE));
}

/****************************************************************************
Desc:	Get a tag name from its tag number.  Tag name is returned as a
		UNICODE string or NATIVE string. If a non-NULL UNICODE string is passed
		in, it will be used.  Otherwise, the NATIVE string will be used.
****************************************************************************/
FLMBOOL F_NameTable::getFromTagNum(
	FLMUINT			uiTagNum,
	FLMUNICODE *	puzTagName,
	char *			pszTagName,
	FLMUINT			uiNameBufSize,
	FLMUINT *		puiType,
	FLMUINT *		puiSubType)
{
	FLM_TAG_INFO *	pTagInfo;

	if (!m_bTablesSorted)
	{
		sortTags();
	}
	if ((pTagInfo = findTagByNum( uiTagNum)) != NULL)
	{
		if (puiType)
		{
			*puiType = pTagInfo->uiType;
		}
		if (puiSubType)
		{
			*puiSubType = pTagInfo->uiSubType;
		}

		if( puzTagName || pszTagName)
		{
			copyTagName( puzTagName, pszTagName, uiNameBufSize,
							pTagInfo->puzTagName);
		}
	}
	else
	{

		// Tag number was not found, but initialize return variables anyway.

		if (puzTagName)
		{
			*puzTagName = 0;
		}
		if (pszTagName)
		{
			*pszTagName = 0;
		}
		if (puiType)
		{
			*puiType = 0;
		}
		if (puiSubType)
		{
			*puiSubType = 0;
		}
	}

	return( (FLMBOOL)(pTagInfo ? (FLMBOOL)TRUE : (FLMBOOL)FALSE));
}

/****************************************************************************
Desc:	Get a tag number and type from its tag name.  Tag name is passed
		in as a UNICODE string or NATIVE string. If a non-NULL UNICODE string
		is passed in, it will be used.  Otherwise, the NATIVE string will
		be used.
****************************************************************************/
FLMBOOL F_NameTable::getFromTagName(
	const FLMUNICODE *	puzTagName,
	const char *			pszTagName,
	FLMUINT *				puiTagNum,
	FLMUINT *				puiType,
	FLMUINT *				puiSubType)
{
	FLM_TAG_INFO *	pTagInfo;

	if (!m_bTablesSorted)
	{
		sortTags();
	}
	if ((pTagInfo = findTagByName( puzTagName, pszTagName)) != NULL)
	{
		flmAssert( puiTagNum);
		*puiTagNum = pTagInfo->uiTagNum;
		if (puiType)
		{
			*puiType = pTagInfo->uiType;
		}
		if (puiSubType)
		{
			*puiSubType = pTagInfo->uiSubType;
		}
	}
	else
	{

		// Tag name was not found, but initialize return variables anyway.

		*puiTagNum = 0;
		if (puiType)
		{
			*puiType = 0;
		}
		if (puiSubType)
		{
			*puiSubType = 0;
		}
	}

	return( (FLMBOOL)(pTagInfo ? (FLMBOOL)TRUE : (FLMBOOL)FALSE));
}

/****************************************************************************
Desc:	Get a tag number from its tag name and type.  Tag name is passed
		in as a UNICODE or NATIVE string. If a non-NULL UNICODE string is
		passed in, it will be used.  Otherwise, the NATIVE string will
		be used.
****************************************************************************/
FLMBOOL F_NameTable::getFromTagTypeAndName(
	const FLMUNICODE *	puzTagName,
	const char *			pszTagName,
	FLMUINT					uiType,
	FLMUINT *				puiTagNum,
	FLMUINT *				puiSubType)
{
	FLM_TAG_INFO *	pTagInfo;

	if (!m_bTablesSorted)
	{
		sortTags();
	}
	if ((pTagInfo = findTagByTypeAndName( puzTagName, pszTagName,
									uiType)) != NULL)
	{
		flmAssert( puiTagNum);
		*puiTagNum = pTagInfo->uiTagNum;
		if (puiSubType)
		{
			*puiSubType = pTagInfo->uiSubType;
		}
	}
	else
	{

		// Tag name was not found, but initialize return variables anyway.

		*puiTagNum = 0;
		if (puiSubType)
		{
			*puiSubType = 0;
		}
	}
	return( (FLMBOOL)(pTagInfo ? (FLMBOOL)TRUE : (FLMBOOL)FALSE));
}

/****************************************************************************
Desc:	Insert a tag info structure into the sorted tables at the specified
		positions.
****************************************************************************/
RCODE F_NameTable::insertTagInTables(
	FLM_TAG_INFO *	pTagInfo,
	FLMUINT			uiTagNameTblInsertPos,
	FLMUINT			uiTagTypeAndNameTblInsertPos,
	FLMUINT			uiTagNumTblInsertPos
	)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiLoop;

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

	// Insert into the sorted-by-name table

	uiLoop = m_uiNumTags;
	while (uiLoop > uiTagNameTblInsertPos)
	{
		m_ppSortedByTagName [uiLoop] = m_ppSortedByTagName [uiLoop - 1];
		uiLoop--;
	}
	m_ppSortedByTagName [uiTagNameTblInsertPos] = pTagInfo;

	// Insert into the sorted-by-number table

	uiLoop = m_uiNumTags;
	while (uiLoop > uiTagNumTblInsertPos)
	{
		m_ppSortedByTagNum [uiLoop] = m_ppSortedByTagNum [uiLoop - 1];
		uiLoop--;
	}
	m_ppSortedByTagNum [uiTagNumTblInsertPos] = pTagInfo;

	// Insert into the sorted-by-tag-name-and-type table

	uiLoop = m_uiNumTags;
	while (uiLoop > uiTagTypeAndNameTblInsertPos)
	{
		m_ppSortedByTagTypeAndName [uiLoop] =
			m_ppSortedByTagTypeAndName [uiLoop - 1];
		uiLoop--;
	}
	m_ppSortedByTagTypeAndName [uiTagTypeAndNameTblInsertPos] = pTagInfo;

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
RCODE F_NameTable::addTag(
	const FLMUNICODE *	puzTagName,
	const char *			pszTagName,
	FLMUINT					uiTagNum,
	FLMUINT					uiType,
	FLMUINT					uiSubType,
	FLMBOOL					bCheckDuplicates)
{
	RCODE					rc = FERR_OK;
	FLMUINT				uiTagNameTblInsertPos;
	FLMUINT				uiTagTypeAndNameTblInsertPos;
	FLMUINT				uiTagNumTblInsertPos;
	FLM_TAG_INFO *		pTagInfo;

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
		flmAssert( 0);
		rc = RC_SET( FERR_INVALID_PARM);
		goto Exit;
	}

	// Tag number of zero not allowed.

	if (!uiTagNum)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_INVALID_PARM);
		goto Exit;
	}

	// Tables must be sorted in order for this to work

	if (bCheckDuplicates)
	{
		if (!m_bTablesSorted)
		{
			sortTags();
		}

		// Make sure that the tag name is not already used.

		if (findTagByName( puzTagName, pszTagName, &uiTagNameTblInsertPos))
		{
			rc = RC_SET( FERR_EXISTS);
			goto Exit;
		}

		// Make sure that the tag name + type is not already used.

		if (findTagByTypeAndName( puzTagName, pszTagName,
						uiType, &uiTagTypeAndNameTblInsertPos))
		{
			rc = RC_SET( FERR_EXISTS);
			goto Exit;
		}

		// Make sure that the tag number is not already used.

		if (findTagByNum( uiTagNum, &uiTagNumTblInsertPos))
		{
			rc = RC_SET( FERR_EXISTS);
			goto Exit;
		}
	}
	else
	{
		uiTagNameTblInsertPos =
		uiTagTypeAndNameTblInsertPos =
		uiTagNumTblInsertPos = m_uiNumTags;
		m_bTablesSorted = FALSE;
	}

	// Create a new tag info structure.

	if (RC_BAD( rc = allocTag( puzTagName, pszTagName, uiTagNum, uiType,
								uiSubType, &pTagInfo)))
	{
		goto Exit;
	}

	// Insert the tag structure into the appropriate places in the
	// sorted tables.

	if (RC_BAD( rc = insertTagInTables( pTagInfo, uiTagNameTblInsertPos,
							uiTagTypeAndNameTblInsertPos,
							uiTagNumTblInsertPos)))
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
		sortTagTbl( m_ppSortedByTagName, 0, m_uiNumTags - 1,
				compareTagNameOnly);
		sortTagTbl( m_ppSortedByTagNum, 0, m_uiNumTags - 1,
				compareTagNumOnly);
		sortTagTbl( m_ppSortedByTagTypeAndName, 0, m_uiNumTags - 1,
				compareTagTypeAndName);
	}
	m_bTablesSorted = TRUE;
}
