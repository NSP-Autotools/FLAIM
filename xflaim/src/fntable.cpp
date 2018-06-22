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

#include "flaimsys.h"

#define MAX_ELEMENTS_TO_LOAD			0xFFFF
#define MAX_ATTRIBUTES_TO_LOAD		0xFFFF

typedef FLMINT (*	TAG_COMPARE_FUNC)(
	FLM_TAG_INFO *	pTagInfo1,
	FLM_TAG_INFO *	pTagInfo2);

FSTATIC FLMINT tagNameCompare(
	const FLMUNICODE *	puzName1,
	const char *			pszName1,
	const FLMUNICODE *	puzName2);

FSTATIC FLMINT compareTagTypeAndName(
	FLM_TAG_INFO *	pTagInfo1,
	FLM_TAG_INFO *	pTagInfo2);

FSTATIC FLMINT compareTagTypeAndNum(
	FLM_TAG_INFO *	pTagInfo1,
	FLM_TAG_INFO *	pTagInfo2);

FSTATIC RCODE findTagName(
	F_Db *					pDb,
	FLMUINT					uiType,
	const FLMUNICODE *	puzTagName,
	const char *			pszTagName,
	F_DataVector *			pSrchKey,
	FLMUINT *				puiDictNum,
	FLMUINT64 *				pui64DocumentID);

FLMUNICODE gv_uzXFLAIMNamespace[] = // http://www.novell.com/XMLDatabase/Schema
{ 'h','t','t','p',':','/','/','w','w','w','.','n','o','v','e','l','l','.','c','o','m','/',
  'X','M','L','D','a','t','a','b','a','s','e','/','S','c','h','e','m','a', 0};

RESERVED_TAG_NAME FlmReservedElementTags[] =
{
	{ELM_ELEMENT_TAG_NAME, ELM_ELEMENT_TAG, XFLM_NODATA_TYPE, gv_uzXFLAIMNamespace},
	{ELM_ATTRIBUTE_TAG_NAME, ELM_ATTRIBUTE_TAG, XFLM_NODATA_TYPE, gv_uzXFLAIMNamespace},
	{ELM_INDEX_TAG_NAME, ELM_INDEX_TAG, XFLM_NODATA_TYPE, gv_uzXFLAIMNamespace},
	{ELM_ELEMENT_COMPONENT_TAG_NAME, ELM_ELEMENT_COMPONENT_TAG, XFLM_NODATA_TYPE, gv_uzXFLAIMNamespace},
	{ELM_ATTRIBUTE_COMPONENT_TAG_NAME, ELM_ATTRIBUTE_COMPONENT_TAG, XFLM_NODATA_TYPE, gv_uzXFLAIMNamespace},
	{ELM_COLLECTION_TAG_NAME, ELM_COLLECTION_TAG, XFLM_NODATA_TYPE, gv_uzXFLAIMNamespace},
	{ELM_PREFIX_TAG_NAME, ELM_PREFIX_TAG, XFLM_NODATA_TYPE, gv_uzXFLAIMNamespace},
	{ELM_NEXT_DICT_NUMS_TAG_NAME, ELM_NEXT_DICT_NUMS_TAG, XFLM_NODATA_TYPE, gv_uzXFLAIMNamespace},
	{ELM_DOCUMENT_TITLE_TAG_NAME, ELM_DOCUMENT_TITLE_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ELM_INVALID_TAG_NAME, ELM_INVALID_TAG, XFLM_NODATA_TYPE, gv_uzXFLAIMNamespace},
	{ELM_QUARANTINED_TAG_NAME, ELM_QUARANTINED_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ELM_ALL_TAG_NAME, ELM_ALL_TAG, XFLM_NODATA_TYPE, gv_uzXFLAIMNamespace},
	{ELM_ANNOTATION_TAG_NAME, ELM_ANNOTATION_TAG, XFLM_NODATA_TYPE, gv_uzXFLAIMNamespace},
	{ELM_ANY_TAG_NAME, ELM_ANY_TAG, XFLM_NODATA_TYPE, gv_uzXFLAIMNamespace},
	{ELM_ATTRIBUTE_GROUP_TAG_NAME, ELM_ATTRIBUTE_GROUP_TAG, XFLM_NODATA_TYPE, gv_uzXFLAIMNamespace},
	{ELM_CHOICE_TAG_NAME, ELM_CHOICE_TAG, XFLM_NODATA_TYPE, gv_uzXFLAIMNamespace},
	{ELM_COMPLEX_CONTENT_TAG_NAME, ELM_COMPLEX_CONTENT_TAG, XFLM_NODATA_TYPE, gv_uzXFLAIMNamespace},
	{ELM_COMPLEX_TYPE_TAG_NAME, ELM_COMPLEX_TYPE_TAG, XFLM_NODATA_TYPE, gv_uzXFLAIMNamespace},
	{ELM_DOCUMENTATION_TAG_NAME, ELM_DOCUMENTATION_TAG, XFLM_NODATA_TYPE, gv_uzXFLAIMNamespace},
	{ELM_ENUMERATION_TAG_NAME, ELM_ENUMERATION_TAG, XFLM_NODATA_TYPE, gv_uzXFLAIMNamespace},
	{ELM_EXTENSION_TAG_NAME, ELM_EXTENSION_TAG, XFLM_NODATA_TYPE, gv_uzXFLAIMNamespace},
	{ELM_DELETE_TAG_NAME, ELM_DELETE_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ELM_BLOCK_CHAIN_TAG_NAME, ELM_BLOCK_CHAIN_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ELM_ENCDEF_TAG_NAME, ELM_ENCDEF_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ELM_SWEEP_TAG_NAME, ELM_SWEEP_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{NULL, 0, 0, NULL}
};

RESERVED_TAG_NAME FlmReservedAttributeTags[] =
{
	{ATTR_DICT_NUMBER_TAG_NAME, ATTR_DICT_NUMBER_TAG, XFLM_NUMBER_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_COLLECTION_NUMBER_TAG_NAME, ATTR_COLLECTION_NUMBER_TAG, XFLM_NUMBER_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_COLLECTION_NAME_TAG_NAME, ATTR_COLLECTION_NAME_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_NAME_TAG_NAME, ATTR_NAME_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_TARGET_NAMESPACE_TAG_NAME, ATTR_TARGET_NAMESPACE_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_TYPE_TAG_NAME, ATTR_TYPE_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_STATE_TAG_NAME, ATTR_STATE_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_LANGUAGE_TAG_NAME, ATTR_LANGUAGE_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_INDEX_OPTIONS_TAG_NAME, ATTR_INDEX_OPTIONS_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_INDEX_ON_TAG_NAME, ATTR_INDEX_ON_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_REQUIRED_TAG_NAME, ATTR_REQUIRED_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_LIMIT_TAG_NAME, ATTR_LIMIT_TAG, XFLM_NUMBER_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_COMPARE_RULES_TAG_NAME, ATTR_COMPARE_RULES_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_KEY_COMPONENT_TAG_NAME, ATTR_KEY_COMPONENT_TAG, XFLM_NUMBER_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_DATA_COMPONENT_TAG_NAME, ATTR_DATA_COMPONENT_TAG, XFLM_NUMBER_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_LAST_DOC_INDEXED_TAG_NAME, ATTR_LAST_DOC_INDEXED_TAG, XFLM_NUMBER_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_NEXT_ELEMENT_NUM_TAG_NAME, ATTR_NEXT_ELEMENT_NUM_TAG, XFLM_NUMBER_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_NEXT_ATTRIBUTE_NUM_TAG_NAME, ATTR_NEXT_ATTRIBUTE_NUM_TAG, XFLM_NUMBER_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_NEXT_INDEX_NUM_TAG_NAME, ATTR_NEXT_INDEX_NUM_TAG, XFLM_NUMBER_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_NEXT_COLLECTION_NUM_TAG_NAME, ATTR_NEXT_COLLECTION_NUM_TAG, XFLM_NUMBER_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_NEXT_PREFIX_NUM_TAG_NAME, ATTR_NEXT_PREFIX_NUM_TAG, XFLM_NUMBER_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_SOURCE_TAG_NAME, ATTR_SOURCE_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_STATE_CHANGE_COUNT_TAG_NAME, ATTR_STATE_CHANGE_COUNT_TAG, XFLM_NUMBER_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_XMLNS_TAG_NAME, ATTR_XMLNS_TAG, XFLM_TEXT_TYPE, NULL},
	{ATTR_ABSTRACT_TAG_NAME, ATTR_ABSTRACT_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_BASE_TAG_NAME, ATTR_BASE_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_BLOCK_TAG_NAME, ATTR_BLOCK_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_DEFAULT_TAG_NAME, ATTR_DEFAULT_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_FINAL_TAG_NAME, ATTR_FINAL_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_FIXED_TAG_NAME, ATTR_FIXED_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_ITEM_TYPE_TAG_NAME, ATTR_ITEM_TYPE_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_MEMBER_TYPES_TAG_NAME, ATTR_MEMBER_TYPES_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_MIXED_TAG_NAME, ATTR_MIXED_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_NILLABLE_TAG_NAME, ATTR_NILLABLE_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_REF_TAG_NAME, ATTR_REF_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_USE_TAG_NAME, ATTR_USE_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_VALUE_TAG_NAME, ATTR_VALUE_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_ADDRESS_TAG_NAME, ATTR_ADDRESS_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_XMLNS_XFLAIM_TAG_NAME, ATTR_XMLNS_XFLAIM_TAG, XFLM_TEXT_TYPE, NULL},
	{ATTR_ENCRYPTION_KEY_TAG_NAME, ATTR_ENCRYPTION_KEY_TAG, XFLM_BINARY_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_TRANSACTION_TAG_NAME, ATTR_TRANSACTION_TAG, XFLM_NUMBER_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_NEXT_ENCDEF_NUM_TAG_NAME, ATTR_NEXT_ENCDEF_NUM_TAG, XFLM_NUMBER_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_ENCRYPTION_ID_TAG_NAME, ATTR_ENCRYPTION_ID_TAG, XFLM_NUMBER_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_ENCRYPTION_KEY_SIZE_TAG_NAME, ATTR_ENCRYPTION_KEY_SIZE_TAG, XFLM_NUMBER_TYPE, gv_uzXFLAIMNamespace},
	{ATTR_UNIQUE_SUB_ELEMENTS_TAG_NAME, ATTR_UNIQUE_SUB_ELEMENTS_TAG, XFLM_TEXT_TYPE, gv_uzXFLAIMNamespace},
	{NULL, 0, 0, NULL}
};

FSTATIC void sortTagTbl(
	FLM_TAG_INFO * *	ppTagInfoTbl,
	FLMUINT				uiLowerBounds,
	FLMUINT				uiUpperBounds,
	TAG_COMPARE_FUNC	fnTagCompare);

/****************************************************************************
Desc:	Constructor
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

	// The m_bLoadedAllElements will be set to FALSE if we call addTag and
	// there is no more room for elements.  Same for m_bLoadedAllAttributes
	// with respect to attributes.

	m_bLoadedAllElements = TRUE;
	m_bLoadedAllAttributes = TRUE;
	m_uiNumElementsLoaded = 0;
	m_uiNumAttributesLoaded = 0;
	m_ppuzNamespaces = NULL;
	m_uiNamespaceTblSize = 0;
	m_uiNumNamespaces = 0;
}

/****************************************************************************
Desc:	Destructor
****************************************************************************/
F_NameTable::~F_NameTable()
{
	clearTable( 0);
}

/****************************************************************************
Desc:	Free everything in the table
****************************************************************************/
void F_NameTable::clearTable(
	FLMUINT		uiPoolBlkSize)
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
	m_bLoadedAllElements = TRUE;
	m_bLoadedAllAttributes = TRUE;
	m_uiNumElementsLoaded = 0;
	m_uiNumAttributesLoaded = 0;
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
				uzLowerChar1 = f_uniToLower( uzChar1);
				uzLowerChar2 = f_uniToLower( uzChar2);

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

			// Elements and attributes need to also compare namespace

			if (uiType == ELM_ELEMENT_TAG || uiType == ELM_ATTRIBUTE_TAG)
			{

				// If the bMatchNamespace flag is FALSE, only search on
				// name.  If there are multiple elements or attributes
				// with the same name, return an ambiguous flag.

				if (!bMatchNamespace)
				{

					// Better not be trying to insert a new one in this case!

					flmAssert( puiInsertPos == NULL);

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
	FLMUNICODE *	puzSrcTagName,	// May be NULL
	FLMBOOL			bTruncatedNamesOk
	)
{
	RCODE		rc = NE_XFLM_OK;
	FLMUINT	uiDestCharCnt = *puiDestBufSize;
	FLMUINT	uiCharCnt;

	if (puzDestTagName)
	{

		// Decrement name buffer size by sizeof( FLMUNICODE) to allow for a
		// terminating NULL character. uiDestChars better be at least big
		// enough for a null terminating character.

		flmAssert( uiDestCharCnt >= sizeof( FLMUNICODE));
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
				rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
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

		flmAssert( uiDestCharCnt);
		uiDestCharCnt--;

		if (puzSrcTagName)
		{
			// Copy the name to the NATIVE buffer.  Non-Ascii UNICODE characters
			// will cause NE_XFLM_CONV_ILLEGAL to be returned.

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
					rc = RC_SET( NE_XFLM_CONV_ILLEGAL);
					goto Exit;
				}

			}
			*pszDestTagName = 0;
			*puiDestBufSize = uiCharCnt;
			if (!bTruncatedNamesOk && *puzSrcTagName)
			{
				rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
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
	FLM_TAG_INFO *	pTagInfo1,
	FLM_TAG_INFO *	pTagInfo2
	)
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
	else if (pTagInfo1->uiType == ELM_ELEMENT_TAG ||
				pTagInfo1->uiType == ELM_ATTRIBUTE_TAG)
	{
		return( tagNameCompare( pTagInfo1->puzNamespace, NULL,
											pTagInfo2->puzNamespace));
	}
	else
	{
		return( 0);
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
Desc: Lookup a namespace - so we can reuse the memory.
****************************************************************************/
FLMUNICODE * F_NameTable::findNamespace(
	FLMUNICODE *	puzNamespace,
	FLMUINT *		puiInsertPos
	)
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
	FLMUINT			uiInsertPos
	)
{
	RCODE		rc = NE_XFLM_OK;
	FLMUINT	uiLoop;

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
	RCODE				rc = NE_XFLM_OK;
	void *			pvMark;
	FLMUINT			uiSaveMemoryAllocated;
	FLM_TAG_INFO *	pTagInfo;
	FLMUINT			uiNameSize;
	FLMUNICODE *	puzTmp;
	FLMUNICODE *	puzTblNamespace;
	FLMUINT			uiNamespaceInsertPos;

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

	// If this is an element or attribute, set namespace and data type

	if (uiType == ELM_ELEMENT_TAG || uiType == ELM_ATTRIBUTE_TAG)
	{
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
	FLMUINT	uiNewTblSize
	)
{
	RCODE					rc = NE_XFLM_OK;
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
Desc:	Add the reserved dictionary tags to the name table.
****************************************************************************/
RCODE F_NameTable::addReservedDictTags( void)
{
	RCODE		rc = NE_XFLM_OK;
	FLMUINT	uiLoop;

	for (uiLoop = 0; FlmReservedElementTags [uiLoop].pszTagName; uiLoop++)
	{
		if (RC_BAD( rc = addTag( ELM_ELEMENT_TAG, NULL,
								FlmReservedElementTags [uiLoop].pszTagName,
								FlmReservedElementTags [uiLoop].uiTagNum,
								FlmReservedElementTags [uiLoop].uiDataType,
								FlmReservedElementTags [uiLoop].puzNamespace, FALSE, FALSE)))
		{
			goto Exit;
		}
	}

	for (uiLoop = 0; FlmReservedAttributeTags [uiLoop].pszTagName; uiLoop++)
	{
		if (RC_BAD( rc = addTag( ELM_ATTRIBUTE_TAG, NULL,
								FlmReservedAttributeTags [uiLoop].pszTagName,
								FlmReservedAttributeTags [uiLoop].uiTagNum,
								FlmReservedAttributeTags [uiLoop].uiDataType,
								FlmReservedAttributeTags [uiLoop].puzNamespace, FALSE, FALSE)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Get a tag name, number, etc. using type + tag number ordering.
		Tag name is returned as a UNICODE string or NATIVE string. If a
		non-NULL UNICODE string is passed in, it will be used.
		Otherwise, the NATIVE string will be used.
****************************************************************************/
RCODE F_NameTable::getNextTagTypeAndNumOrder(
	FLMUINT			uiType,
	FLMUINT *		puiNextPos,
	FLMUNICODE *	puzTagName,			// May be NULL
	char *			pszTagName,			// May be NULL
	FLMUINT			uiNameBufSize,
	FLMUINT *		puiTagNum,			// May be NULL
	FLMUINT *		puiDataType,		// May be NULL
	FLMUNICODE *	puzNamespace,		// May be NULL
	FLMUINT			uiNamespaceBufSize,
	FLMBOOL			bTruncatedNamesOk
	)
{
	RCODE				rc = NE_XFLM_OK;
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

			if (uiType == ELM_ELEMENT_TAG || uiType == ELM_ATTRIBUTE_TAG)
			{
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
			}
			else
			{
				flmAssert( !puiDataType && !puzNamespace);
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
		rc = RC_SET( NE_XFLM_EOF_HIT);
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
RCODE F_NameTable::getNextTagTypeAndNameOrder(
	FLMUINT			uiType,
	FLMUINT *		puiNextPos,
	FLMUNICODE *	puzTagName,			// May be NULL
	char *			pszTagName,			// May be NULL
	FLMUINT			uiNameBufSize,
	FLMUINT *		puiTagNum,			// May be NULL
	FLMUINT *		puiDataType,		// May be NULL
	FLMUNICODE *	puzNamespace,		// May be NULL
	FLMUINT			uiNamespaceBufSize,
	FLMBOOL			bTruncatedNamesOk
	)
{
	RCODE				rc = NE_XFLM_OK;
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
			if (uiType == ELM_ELEMENT_TAG || uiType == ELM_ATTRIBUTE_TAG)
			{
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
			}
			else
			{
				flmAssert( !puiDataType && !puzNamespace);
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
		rc = RC_SET( NE_XFLM_EOF_HIT);
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
RCODE F_NameTable::getFromTagTypeAndNum(
	F_Db *			pDb,
	FLMUINT			uiType,
	FLMUINT			uiTagNum,
	FLMUNICODE *	puzTagName,			// May be NULL
	char *			pszTagName,			// May be NULL
	FLMUINT *		puiNameBufSize,	// May be NULL, returns # characters
	FLMUINT *		puiDataType,		// May be NULL
	FLMUNICODE *	puzNamespace,		// May be NULL
	char *			pszNamespace,		// May be NULL
	FLMUINT *		puiNamespaceBufSize,	// May be NULL, returns # characters
	FLMBOOL			bTruncatedNamesOk
	)
{
	RCODE				rc = NE_XFLM_OK;
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
		if (uiType == ELM_ELEMENT_TAG || uiType == ELM_ATTRIBUTE_TAG)
		{
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
			flmAssert( !puiDataType && !puzNamespace && !pszNamespace);
		}
	}
	else
	{

		// If we do not have all of the tags cached, and this tag number is
		// outside the range of tag numbers we read in, read the database
		// definition document to get the information.

		if (pDb &&
			 ((uiType == ELM_ELEMENT_TAG && !m_bLoadedAllElements) ||
			  (uiType == ELM_ATTRIBUTE_TAG && !m_bLoadedAllAttributes)))
		{
			F_DataVector	searchKey;
			F_DataVector	foundKey;
			F_AttrElmInfo	defInfo;

			// Need to lookup the document's ID, using the tag number

			if (RC_BAD( rc = searchKey.setUINT( 0, uiType)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = searchKey.setUINT( 1, uiTagNum)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = pDb->keyRetrieve( XFLM_DICT_NUMBER_INDEX,
					&searchKey, XFLM_EXACT, &foundKey)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = pDb->getElmAttrInfo( uiType, 
				foundKey.getDocumentID(), &defInfo, TRUE, FALSE)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = defInfo.m_pNameAttr->getUnicode( 
				pDb, &puzTmpName)))
			{
				goto Exit;
			}

			if( defInfo.m_pTargetNamespaceAttr)
			{
				if( RC_BAD( rc = defInfo.m_pTargetNamespaceAttr->getUnicode( 
					pDb, &puzTmpNamespace)))
				{
					goto Exit;
				}
			}

			if( puiDataType)
			{
				*puiDataType = defInfo.m_uiDataType;
			}

			if (puzTagName || pszTagName)
			{
				if (RC_BAD( rc = copyTagName( puzTagName, pszTagName,
									puiNameBufSize,
									puzTmpName, bTruncatedNamesOk)))
				{
					goto Exit;
				}
			}
			else if (puiNameBufSize)
			{
				*puiNameBufSize = f_unilen( puzTmpName);
			}

			if (puzNamespace || pszNamespace)
			{
				if (RC_BAD( rc = copyTagName( puzNamespace, pszNamespace,
									puiNamespaceBufSize, puzTmpNamespace,
									bTruncatedNamesOk)))
				{
					goto Exit;
				}
			}
			else if (puiNamespaceBufSize)
			{
				*puiNamespaceBufSize = f_unilen( puzTmpNamespace);
			}
		}
		else
		{
			rc = RC_SET( NE_XFLM_NOT_FOUND);
			goto Exit;
		}
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
Desc:	Find a name of a particular type and determine if it is ambiguous.
****************************************************************************/
FSTATIC RCODE findTagName(
	F_Db *					pDb,
	FLMUINT					uiType,
	const FLMUNICODE *	puzTagName,
	const char *			pszTagName,
	F_DataVector *			pSrchKey,
	FLMUINT *				puiDictNum,
	FLMUINT64 *				pui64DocumentID)
{
	RCODE				rc = NE_XFLM_OK;
	F_DataVector	foundKey;
	F_DataVector	foundKey2;
	FLMUINT			uiFoundType;
	FLMUNICODE *	puzTmpName = NULL;

	// Node type and namespace are unknown, find the first name
	// that matches.

	if (RC_BAD( rc = pDb->keyRetrieve( XFLM_DICT_NAME_INDEX,
				pSrchKey, XFLM_INCL, &foundKey)))
	{
		if (rc == NE_XFLM_EOF_HIT)
		{
			rc = RC_SET( NE_XFLM_NOT_FOUND);
		}
		goto Exit;
	}

	if (RC_BAD( rc = foundKey.getUINT( 0, &uiFoundType)))
	{
		goto Exit;
	}

	// Make sure we are still on the right kind of definition.

	if (uiFoundType != uiType)
	{
		rc = RC_SET( NE_XFLM_NOT_FOUND);
		goto Exit;
	}

	// Verify that we landed on a key that has the name we
	// were searching for.  Name is in component [1] of the key.

	if (RC_BAD( rc = foundKey.getUnicode( 1, &puzTmpName)))
	{
		goto Exit;
	}
	if (tagNameCompare( puzTagName, pszTagName, puzTmpName) != 0)
	{
		rc = RC_SET( NE_XFLM_NOT_FOUND);
		goto Exit;
	}

	// Get the document ID

	*pui64DocumentID = foundKey.getDocumentID();

	// Get the dictionary number from the data part of the key

	if (RC_BAD( rc = foundKey.getUINT( 3, puiDictNum)))
	{
		goto Exit;
	}

	// Determine if there is more than one of that key.

	if (RC_BAD( rc = pDb->keyRetrieve( XFLM_DICT_NAME_INDEX, &foundKey,
								XFLM_EXCL | XFLM_MATCH_IDS, &foundKey2)))
	{
		if (rc == NE_XFLM_EOF_HIT)
		{
			rc = NE_XFLM_OK;
		}
		else
		{
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = foundKey2.getUINT( 0, &uiFoundType)))
		{
			if (rc == NE_XFLM_NOT_FOUND)
			{
				rc = NE_XFLM_OK;
			}
			goto Exit;
		}

		// Make sure we are still on the right kind of definition.

		if (uiFoundType != uiType)
		{
			goto Exit;	// will return NE_XFLM_OK
		}

		// Verify that we landed on a key that has a different name than we
		// were searching for.  Name is in component [1] of the key.

		if (RC_BAD( rc = foundKey2.getUnicode( 1, &puzTmpName)))
		{
			if (rc == NE_XFLM_NOT_FOUND)
			{
				rc = NE_XFLM_OK;
			}
			goto Exit;
		}
		if (tagNameCompare( puzTagName, pszTagName, puzTmpName) != 0)
		{
			goto Exit;	// will return NE_XFLM_OK
		}

		rc = RC_SET( NE_XFLM_MULTIPLE_MATCHES);
		goto Exit;
	}

Exit:

	if (puzTmpName)
	{
		f_free( &puzTmpName);
	}

	return( rc);
}

/****************************************************************************
Desc:	Get a tag number from its tag name and type.  Tag name is passed
		in as a UNICODE or NATIVE string. If a non-NULL UNICODE string is
		passed in, it will be used.  Otherwise, the NATIVE string will
		be used.
****************************************************************************/
RCODE F_NameTable::getFromTagTypeAndName(
	F_Db *					pDb,
	FLMUINT					uiType,
	const FLMUNICODE *	puzTagName,
	const char *			pszTagName,
	FLMBOOL					bMatchNamespace,
	const FLMUNICODE *	puzNamespace,
	FLMUINT *				puiTagNum,		// May be NULL
	FLMUINT *				puiDataType)	// May be NULL
{
	RCODE					rc = NE_XFLM_OK;
	FLM_TAG_INFO *		pTagInfo;
	FLMUINT				uiDictNum;
	FLMBOOL				bAmbiguous;
	FLMUINT				uiTmpDictNum;
	FLMUINT64			ui64DocumentID;
	F_DataVector		searchKey;
	F_DataVector		foundKey;

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
			*puiDataType = (FLMUINT)(uiType == ELM_ELEMENT_TAG ||
											 uiType == ELM_ATTRIBUTE_TAG
											 ? pTagInfo->uiDataType
											 : 0);
		}
		if (bAmbiguous)
		{
			rc = RC_SET( NE_XFLM_MULTIPLE_MATCHES);
			goto Exit;
		}
		else if (pDb && !bMatchNamespace &&
					((uiType == ELM_ELEMENT_TAG && !m_bLoadedAllElements) ||
					 (uiType == ELM_ATTRIBUTE_TAG && !m_bLoadedAllAttributes)))
		{

			// Must see if it is ambiguous - create a search key

			if (RC_BAD( rc = searchKey.setUINT( 0, uiType)))
			{
				goto Exit;
			}

			// Put name into the key.

			if (puzTagName)
			{
				if (RC_BAD( rc = searchKey.setUnicode( 1, puzTagName)))
				{
					goto Exit;
				}
			}
			else
			{
				if (RC_BAD( rc = searchKey.setUTF8( 1, (FLMBYTE *)pszTagName)))
				{
					goto Exit;
				}
			}

			if (RC_BAD( rc = findTagName( pDb, uiType, puzTagName, pszTagName,
										&searchKey, &uiTmpDictNum, &ui64DocumentID)))
			{

				// Better be found at this point because it was in the
				// name table! Error may be NE_XFLM_MULTIPLE_MATCHES.

				flmAssert( rc != NE_XFLM_NOT_FOUND);
				goto Exit;
			}

			// What we found in the index better match what we found in
			// the table!

			flmAssert( uiTmpDictNum == *puiTagNum);
		}
	}
	else
	{

		// If we do not have all of the tags cached, and we did not read in
		// all of the elements or attributes, read the dictionary name index
		// to see if we can find this tag name.  NOTE that if bMatchNamespace ==
		// FALSE we will simply find the first tag name that
		// matches and not worry about matching the namespace.

		if (pDb &&
			 ((uiType == ELM_ELEMENT_TAG && !m_bLoadedAllElements) ||
			  (uiType == ELM_ATTRIBUTE_TAG && !m_bLoadedAllAttributes)))
		{
			F_AttrElmInfo		defInfo;

			// Create a search key

			if (RC_BAD( rc = searchKey.setUINT( 0, uiType)))
			{
				goto Exit;
			}

			if (puzTagName)
			{
				if (RC_BAD( rc = searchKey.setUnicode( 1, puzTagName)))
				{
					goto Exit;
				}
			}
			else
			{
				if (RC_BAD( rc = searchKey.setUTF8( 1, (FLMBYTE *)pszTagName)))
				{
					goto Exit;
				}
			}
			if (!bMatchNamespace)
			{
				if (RC_BAD( rc = findTagName( pDb, uiType, puzTagName, pszTagName,
											&searchKey, &uiDictNum, &ui64DocumentID)))
				{
					goto Exit;
				}
			}
			else
			{
				// Add the namespace as a child node.

				if (puzNamespace)
				{
					if (RC_BAD( rc = searchKey.setUnicode( 2, puzNamespace)))
					{
						goto Exit;
					}
				}

				// Search for this exact key.

				if (RC_BAD( rc = pDb->keyRetrieve( XFLM_DICT_NAME_INDEX,
							&searchKey, XFLM_EXACT, &foundKey)))
				{
					goto Exit;
				}

				ui64DocumentID = foundKey.getDocumentID();

				// Data component [0]'s value will be the dictionary number

				if (RC_BAD( rc = foundKey.getUINT( 3, &uiDictNum)))
				{
					if (rc == NE_XFLM_NOT_FOUND)
					{
						flmAssert( 0);
						uiDictNum = 0;
						rc = NE_XFLM_OK;
					}
					else
					{
						goto Exit;
					}
				}
			}

			if (RC_BAD( rc = pDb->getElmAttrInfo( uiType, 
				ui64DocumentID, &defInfo, TRUE, FALSE)))
			{
				goto Exit;
			}

			if( puiDataType)
			{
				*puiDataType = defInfo.m_uiDataType;
			}

			if (puiTagNum)
			{
				*puiTagNum = uiDictNum;
			}
		}
		else
		{
			rc = RC_SET( NE_XFLM_NOT_FOUND);
			goto Exit;
		}
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
	FLMUINT			uiTagTypeAndNumTblInsertPos
	)
{
	RCODE		rc = NE_XFLM_OK;
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
RCODE F_NameTable::addTag(
	FLMUINT			uiType,
	FLMUNICODE *	puzTagName,
	const char *	pszTagName,
	FLMUINT			uiTagNum,
	FLMUINT			uiDataType,
	FLMUNICODE *	puzNamespace,
	FLMBOOL			bCheckDuplicates,
	FLMBOOL			bLimitNumToLoad)
{
	RCODE				rc = NE_XFLM_OK;
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
		rc = RC_SET_AND_ASSERT( NE_XFLM_INVALID_PARM);
		goto Exit;
	}

	// Tag number of zero not allowed.

	if (!uiTagNum)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_INVALID_PARM);
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
			rc = RC_SET( NE_XFLM_EXISTS);
			goto Exit;
		}

		// Make sure that the tag type + number is not already used.

		if (findTagByTypeAndNum( uiType, uiTagNum,
						&uiTagTypeAndNumTblInsertPos))
		{
			rc = RC_SET( NE_XFLM_EXISTS);
			goto Exit;
		}
	}
	else
	{
		uiTagTypeAndNameTblInsertPos =
		uiTagTypeAndNumTblInsertPos = m_uiNumTags;
		m_bTablesSorted = FALSE;
	}

	if (uiType == ELM_ELEMENT_TAG)
	{
		if (m_uiNumElementsLoaded >= MAX_ELEMENTS_TO_LOAD &&
			 bLimitNumToLoad)
		{

			// We purposely limit the number of elements that can be
			// loaded into the table.

			m_bLoadedAllElements = FALSE;
			goto Exit;	// Will return NE_XFLM_OK
		}
	}
	else if (uiType == ELM_ATTRIBUTE_TAG)
	{

		if (m_uiNumAttributesLoaded >= MAX_ATTRIBUTES_TO_LOAD &&
				bLimitNumToLoad)
		{

			// We purposely limit the number of elements that can be
			// loaded into the table.

			m_bLoadedAllAttributes = FALSE;
			goto Exit;	// Will return NE_XFLM_OK
		}
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

	if (uiType == ELM_ELEMENT_TAG)
	{
		m_uiNumElementsLoaded++;
	}
	else if (uiType == ELM_ATTRIBUTE_TAG)
	{
		m_uiNumAttributesLoaded++;
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
void F_NameTable::removeTag(
	FLMUINT	uiType,
	FLMUINT	uiTagNum)
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
		if (uiType == ELM_ELEMENT_TAG || uiType == ELM_ATTRIBUTE_TAG)
		{
			puzNamespace = pTagInfo->puzNamespace;
			bMatchNamespace = TRUE;
		}
		else
		{
			bMatchNamespace = FALSE;	// Really doesn't matter
			puzNamespace = NULL;
		}
		
		if (findTagByTypeAndName( uiType, pTagInfo->puzTagName,
								NULL, bMatchNamespace,
								puzNamespace, &bAmbiguous,
								&uiTagTypeAndNameTblPos) == NULL)
		{

			// It should have been in the name table too!

			flmAssert( 0);
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
Desc:	Clone a name table from another one
****************************************************************************/
RCODE F_NameTable::cloneNameTable(
	F_NameTable *	pSrcNameTable
	)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiLoop;
	FLM_TAG_INFO *	pTagInfo;
	FLMUINT			uiPoolBlkSize;

	// Set the pool size to be as optimal as possible.

	uiPoolBlkSize = pSrcNameTable->m_uiMemoryAllocated / 8;
	if (uiPoolBlkSize < 1024)
	{
		uiPoolBlkSize = 1024;
	}
	else if (uiPoolBlkSize > 65536)
	{
		uiPoolBlkSize = 65536;
	}
	clearTable( uiPoolBlkSize);

	// Pre-allocate exactly enough table space

	if (RC_BAD( rc = reallocSortTables( pSrcNameTable->m_uiNumTags)))
	{
		goto Exit;
	}

	// Add all of the tags

	for (uiLoop = 0; uiLoop < pSrcNameTable->m_uiNumTags; uiLoop++)
	{
		pTagInfo = pSrcNameTable->m_ppSortedByTagTypeAndNum [uiLoop];
		if (pTagInfo->uiType == ELM_ELEMENT_TAG ||
			 pTagInfo->uiType == ELM_ATTRIBUTE_TAG)
		{
			if (RC_BAD( rc = addTag( pTagInfo->uiType, pTagInfo->puzTagName, NULL,
										pTagInfo->uiTagNum,
										pTagInfo->uiDataType,
										pTagInfo->puzNamespace, FALSE)))
			{
				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = addTag( pTagInfo->uiType, pTagInfo->puzTagName, NULL,
										pTagInfo->uiTagNum, 0, NULL, FALSE)))
			{
				goto Exit;
			}
		}
	}

	sortTags();

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Copy a name table from another one.  This differs from cloneNameTable
		in that it doesn't clear the name table, it just copies the names.
		Thus, the destination name table may already have names in it, and the
		names from the source table will just be added.
****************************************************************************/
RCODE F_NameTable::importFromNameTable(
	F_NameTable *	pSrcNameTable
	)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiLoop;
	FLM_TAG_INFO *	pTagInfo;

	// Pre-allocate exactly enough table space

	if (RC_BAD( rc = reallocSortTables( m_uiNumTags +
								pSrcNameTable->m_uiNumTags)))
	{
		goto Exit;
	}

	// Add all of the tags from the source table

	for (uiLoop = 0; uiLoop < pSrcNameTable->m_uiNumTags; uiLoop++)
	{
		pTagInfo = pSrcNameTable->m_ppSortedByTagTypeAndNum [uiLoop];
		if (pTagInfo->uiType == ELM_ELEMENT_TAG ||
			 pTagInfo->uiType == ELM_ATTRIBUTE_TAG)
		{
			if (RC_BAD( rc = addTag( pTagInfo->uiType, pTagInfo->puzTagName, NULL,
										pTagInfo->uiTagNum,
										pTagInfo->uiDataType,
										pTagInfo->puzNamespace, FALSE)))
			{
				if (rc != NE_XFLM_EXISTS)
				{
					goto Exit;
				}
				else
				{
					rc = NE_XFLM_OK;
				}
			}
		}
		else
		{
			if (RC_BAD( rc = addTag( pTagInfo->uiType, pTagInfo->puzTagName, NULL,
										pTagInfo->uiTagNum, 0, NULL, FALSE)))
			{
				if (rc != NE_XFLM_EXISTS)
				{
					goto Exit;
				}
				else
				{
					rc = NE_XFLM_OK;
				}
			}
		}
	}

	sortTags();

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Increment use count on this object.
****************************************************************************/
FLMINT XFLAPI F_NameTable::AddRef( void)
{
	return( f_atomicInc( &m_refCnt));
}

/****************************************************************************
Desc:	Decrement the use count and delete if use count goes to zero.
****************************************************************************/
FLMINT XFLAPI F_NameTable::Release( void)
{
	FLMINT		iRefCnt;

	if ((iRefCnt = f_atomicDec( &m_refCnt)) == 0)
	{
		delete this;
	}

	return( iRefCnt);
}

/****************************************************************************
Desc:	Get the name table for an FDB, if any.  If it has no current
		dictionary, get the name table for the FDB's FFILE's first dictionary,
		if any.  Increment the use count on any name table returned.
		It is the caller's responsibility to release the name table when it
		is done with it.
****************************************************************************/
RCODE F_Db::getNameTable(
	F_NameTable **	ppNameTable)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bMutexLocked = FALSE;

	*ppNameTable = NULL;
	if (m_pDict)
	{
		if ((*ppNameTable = m_pDict->getNameTable()) == NULL)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_NO_NAME_TABLE);
			goto Exit;
		}
		
		(*ppNameTable)->AddRef();
	}
	else
	{
		m_pDatabase->lockMutex();
		bMutexLocked = TRUE;

		if (m_pDatabase && m_pDatabase->m_pDictList)
		{
			if ((*ppNameTable = m_pDatabase->m_pDictList->getNameTable()) == NULL)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_NO_NAME_TABLE);
				goto Exit;
			}
			
			(*ppNameTable)->AddRef();
		}
		else
		{
			rc = RC_SET( NE_XFLM_NO_NAME_TABLE);
			goto Exit;
		}
	}

Exit:

	if (bMutexLocked)
	{
		m_pDatabase->unlockMutex();
	}
	return( rc);
}

/****************************************************************************
Desc:	Get the name for a dictionary item.
****************************************************************************/
RCODE XFLAPI F_Db::getDictionaryName(
	FLMUINT					uiDictType,
	FLMUINT					uiDictNumber,
	char *					pszName,
	FLMUINT *				puiNameBufSize,
	char *					pszNamespace,
	FLMUINT *				puiNamespaceBufSize
	)
{
	RCODE				rc = NE_XFLM_OK;
	F_NameTable *	pNameTable = NULL;

	if (RC_BAD(rc = getNameTable( &pNameTable)))
	{
		goto Exit;
	}

	if (pszNamespace &&
		 (uiDictType == ELM_ELEMENT_TAG || uiDictType == ELM_ATTRIBUTE_TAG))
	{
		flmAssert( puiNamespaceBufSize);
	}
	else
	{
		flmAssert( !pszNamespace && !puiNamespaceBufSize);
		pszNamespace = NULL;
		puiNamespaceBufSize = NULL;
	}

	if (RC_BAD( rc = pNameTable->getFromTagTypeAndNum( this, uiDictType,
											uiDictNumber, NULL,
											pszName, puiNameBufSize, NULL,
											NULL, pszNamespace,
											puiNamespaceBufSize, TRUE)))
	{
		goto Exit;
	}

Exit:

	if (pNameTable)
	{
		pNameTable->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Get the name for a dictionary item.
****************************************************************************/
RCODE XFLAPI F_Db::getDictionaryName(
	FLMUINT					uiDictType,
	FLMUINT					uiDictNumber,
	FLMUNICODE *			puzName,
	FLMUINT *				puiNameBufSize,
	FLMUNICODE *			puzNamespace,
	FLMUINT *				puiNamespaceBufSize
	)
{
	RCODE				rc = NE_XFLM_OK;
	F_NameTable *	pNameTable = NULL;

	if (RC_BAD(rc = getNameTable( &pNameTable)))
	{
		goto Exit;
	}

	if (uiDictType == ELM_ELEMENT_TAG || uiDictType == ELM_ATTRIBUTE_TAG)
	{
		flmAssert( puiNamespaceBufSize);
	}
	else
	{
		flmAssert( !puiNamespaceBufSize);
		puiNamespaceBufSize = NULL;
	}

	if (RC_BAD( rc = pNameTable->getFromTagTypeAndNum( this, uiDictType,
											uiDictNumber, puzName,
											NULL, puiNameBufSize, NULL,
											puzNamespace, NULL,
											puiNamespaceBufSize, TRUE)))
	{
		goto Exit;
	}

Exit:

	if (pNameTable)
	{
		pNameTable->Release();
	}

	return( rc);
}
