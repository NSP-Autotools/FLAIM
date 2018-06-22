//------------------------------------------------------------------------------
// Desc:	F_Dict class definitions - internal object for database's
//			dictionary.
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

#ifndef FDICT_H
#define FDICT_H

#define FLM_HIGH_FIXED_ELEMENT_NUM		0xFFFF
#define FLM_LOW_EXT_ELEMENT_NUM			(FLM_HIGH_FIXED_ELEMENT_NUM + 1)

#define FLM_HIGH_FIXED_ATTRIBUTE_NUM	0xFFFF
#define FLM_LOW_EXT_ATTRIBUTE_NUM		(FLM_HIGH_FIXED_ATTRIBUTE_NUM + 1)

#define MAX_EXT_ATTR_ELM_ARRAY_SIZE		0xFFFF

struct IXD;
struct ICD;
class F_Database;
class F_AttrItem;
class IF_CCS;

/****************************************************************************
Desc:		Attribute/Element definition structure.
****************************************************************************/
typedef struct AttrElmDef
{
	// IMPORTANT NOTE: If adding pointers to this structure, be sure
	// to add code to F_Dict->clone() method to set them up properly.

	FLMUINT						uiFlags;				// Lower four bits has data type
															// Upper four bits has state
															// High bit indicates if it is an
															// attribute or not.
															// Value of zero means slot is not
															// used - allows us to memset an array
															// to zeroes and have all slots unused.
#define ATTR_ELM_DATA_TYPE_MASK	0x000F
#define ATTR_ELM_STATE_MASK		0x00F0
#define ATTR_ELM_STATE_ACTIVE		0x0010 		// Normal active attribute or element
#define ATTR_ELM_STATE_CHECKING	0x0020		// Attribute or element has been marked
															// to be checked
#define ATTR_ELM_STATE_PURGE		0x0040		// Purge this attribute or element
															// from the database.
															// And delete the dictionary definition
#define ATTR_ELM_FLAGS_MASK		0x0F00
#define ATTR_ELM_NS_DECL			0x0100		// Attribute is a namespace declaration
#define ATTR_ELM_UNIQUE_SUBELMS	0x0200		// Element's sub-elements must all have unique name ids
	ICD *		pFirstIcd;								// Points to first ICD attribute or element is
															// indexed in.  NULL if not indexed.
} ATTR_ELM_DEF;

/****************************************************************************
Desc:		Extended attribute/element definition structure.
****************************************************************************/
typedef struct ExtAttrElmDef
{
	FLMUINT			uiDictNum;
	ATTR_ELM_DEF	attrElmDef;
} EXT_ATTR_ELM_DEF;

/*****************************************************************************
Desc:		Dictionary definition info object
*****************************************************************************/
class F_AttrElmInfo : public F_Object
{
public:

	F_AttrElmInfo()
	{
		m_pDocNode = NULL;
		m_pTargetNamespaceAttr = NULL;
		m_pNameAttr = NULL;
		resetInfo();
	}

	~F_AttrElmInfo()
	{
		resetInfo();
	}

	void resetInfo( void);

	FINLINE FLMUINT getDataType( void)
	{
		return( m_uiDataType);
	}

	FINLINE FLMUINT getState( void)
	{
		return( m_uiState);
	}
	
	FINLINE FLMUINT getFlags( void)
	{
		return( m_uiFlags);
	}

private:

	FLMUINT			m_uiDictNum;
	FLMUINT			m_uiDataType;
	FLMUINT			m_uiFlags;
	FLMUINT			m_uiState;
	ICD *				m_pFirstIcd;

	IF_DOMNode *	m_pDocNode;
	IF_DOMNode *	m_pTargetNamespaceAttr;
	IF_DOMNode *	m_pNameAttr;

friend class F_Db;
friend class F_DOMNode;
friend class F_Dict;
friend class F_NameTable;
friend class F_NodeVerifier;
friend class F_Query;
};

/**************************************************************************
Desc:	Get an attribute or element's type.
**************************************************************************/
FINLINE FLMUINT attrElmGetType(
	ATTR_ELM_DEF *	pAttrElmDef)
{
	return( pAttrElmDef->uiFlags & ATTR_ELM_DATA_TYPE_MASK);
}

/**************************************************************************
Desc:	See if an attribute or element is indexed.
**************************************************************************/
FINLINE FLMBOOL attrElmIsIndexed(
	ATTR_ELM_DEF *	pAttrElmDef)
{
	return( pAttrElmDef->pFirstIcd ? TRUE : FALSE);
}

/**************************************************************************
Desc:	Get an attribute or element's state.
**************************************************************************/
FINLINE FLMUINT attrElmGetState(
	ATTR_ELM_DEF *	pAttrElmDef)
{
	return( pAttrElmDef->uiFlags & ATTR_ELM_STATE_MASK);
}

/**************************************************************************
Desc:	Set an attribute or element's state.
**************************************************************************/
FINLINE void attrElmSetState(
	ATTR_ELM_DEF *	pAttrElmDef,
	FLMUINT			uiState
	)
{
	pAttrElmDef->uiFlags =
		(pAttrElmDef->uiFlags & (~(ATTR_ELM_STATE_MASK))) |
		(uiState & ATTR_ELM_STATE_MASK);
}

/**************************************************************************
Desc:	Get an attribute or element's flags.
**************************************************************************/
FINLINE FLMUINT attrElmGetFlags(
	ATTR_ELM_DEF *	pAttrElmDef)
{
	return( pAttrElmDef->uiFlags & ATTR_ELM_FLAGS_MASK);
}

/**************************************************************************
Desc:	Set an attribute or element's flags..
**************************************************************************/
FINLINE void attrElmSetFlags(
	ATTR_ELM_DEF *	pAttrElmDef,
	FLMUINT			uiFlags)
{
	pAttrElmDef->uiFlags =
		(pAttrElmDef->uiFlags & (~(ATTR_ELM_FLAGS_MASK))) |
		(uiFlags & ATTR_ELM_FLAGS_MASK);
}

/****************************************************************************
Struct:	LFILE		(Logical File)
Desc:		This keeps track of the logical file information for an index or
			a Collection.
****************************************************************************/
typedef struct LFILE
{

	// IMPORTANT NOTE: If adding pointers to this structure, be sure
	// to add code to F_Dict->clone() method to set them up properly.

	FLMUINT   	uiRootBlk;			// Address of root block.
	FLMUINT		uiBlkAddress;		// Block address of LFile entry.
	FLMUINT		uiOffsetInBlk;		// Offset within block of entry.
	FLMUINT		uiLfNum;				// Index number or collection number.
	eLFileType	eLfType; 			// Type of logical file
	FLMUINT		uiEncId;				// Encryption Id (0 if not encrypted)
} LFILE;

/****************************************************************************
Struct:	F_COLLECTION		(Collection)
Desc:		This keeps track of collections.
****************************************************************************/
typedef struct F_COLLECTION
{
	LFILE			lfInfo;					// B-Tree information
	FLMBOOL		bNeedToUpdateNodes;	// Do we need to write out node info.
												// at commit time?
	FLMUINT64	ui64NextNodeId;		// Next Node ID
	FLMUINT64	ui64FirstDocId;		// First document ID
	FLMUINT64	ui64LastDocId;			// Last document ID
} F_COLLECTION;

/****************************************************************************
Struct:	F_PREFIX		(Prefix)
Desc:		This keeps track of Prefixes.
****************************************************************************/
typedef struct F_PREFIX
{
	FLMUINT64			ui64PrefixId;
	FLMUNICODE *		puzPrefixName;
} F_PREFIX;

/****************************************************************************
Struct:	F_ENCDEF		(Encryption Definition)
Desc:		This keeps track of encryption definitions
****************************************************************************/
typedef struct
{
	FLMUINT64			ui64EncDefId;
	FLMUINT64			ui64DocumentId;
	FLMUNICODE *		puzEncDefName;
	FLMUINT				uiEncKeySize;
	IF_CCS *				pCcs;
} F_ENCDEF;

/****************************************************************************
Struct:	IXD		(Index Definition)
Desc:		This structure holds the information for an index definition.
			There may be multiple IXDs for the same index number.
****************************************************************************/
typedef struct IXD
{

	// IMPORTANT NOTE: If adding pointers to this structure, be sure
	// to add code to F_Dict->clone() method to set them up properly.
	// ALSO, fixup indexDefsSame function in fdict.cpp

	FLMUINT		uiIndexNum;				// Index number.
	FLMUINT		uiCollectionNum;		// Collection number being indexed.
	ICD *			pIcdTree;				// Points to ICD tree
	ICD *			pFirstKey;				// Points to first key component
	ICD *			pLastKey;				// Points to last key component
	ICD *			pFirstContext;			// Points to first context-only component
	ICD *			pLastContext;			// Points to last context-only component
	ICD *			pFirstData;				// Points to first data component
	ICD *			pLastData;				// Points to last data component
	FLMUINT		uiNumIcds;				// Total ICDs for this index.
	FLMUINT		uiNumKeyComponents;	// Number of key components in the index.
	FLMUINT		uiNumDataComponents;	// Number of data components in the index.
	FLMUINT		uiNumContextComponents;	// Number of context-only components in
												// the index.
	FLMUINT		uiFlags;
		#define IXD_ABS_POS				0x00001	// Maintain absolute positioning info.
		#define IXD_HAS_SUBSTRING		0x00002
		#define IXD_OFFLINE				0x00004	// Index is offline - may or may
															// not be suspended.
		#define IXD_SUSPENDED			0x00008	// IXD_OFFLINE should also be set
		#define IXD_SINGLE_PATH			0x00010	// ICD list is a single path

	FLMUINT		uiLanguage;				// WP.LRS language number (not code!)
	FLMUINT64	ui64LastDocIndexed;	// If value is not ~0 then
												// update index with keys from a document
												// update if doc id is <= this value.
												// NOTE: This is only guaranteed to be
												// correct for update transactions.
												// This field should only be used by
												// update transactions anyway.
	LFILE			lfInfo;					// B-Tree information.
	FLMUINT64	ui64IxDefNodeId;
} IXD;

/****************************************************************************
Struct:	ICD		(Index Component Definition)
Desc:		This structure contains an index component definition.
****************************************************************************/
typedef struct ICD
{

	// IMPORTANT NOTE: If adding pointers to this structure, be sure
	// to add code to F_Dict->clone() method to set them up properly.
	// ALSO, fixup indexDefsSame function in fdict.cpp

	FLMUINT		uiIndexNum;				// Index number.
	IXD *			pIxd;						// IXD corresponding to uiIndexNum
	FLMUINT		uiDictNum;				// Attribute or element number.
	FLMUINT		uiFlags;					// The first 4 bits contain data type
												// Use FLM_XXXXX_TYPE definitions.
	FLMUINT		uiCompareRules;		// Comparison rules used for strings.												

	ICD *			pNextInChain;			// Next ICD in the chain that has this
												// attribute or element number and is
												// used in another place.
	ICD *			pParent;					// Parent ICD
	ICD *			pFirstChild;			// First Child ICD
	ICD *			pPrevSibling;			// Previous Sibling ICD
	ICD *			pNextSibling;			// Next Sibling ICD
	FLMUINT		uiCdl;					// Place in CDL list where a node matching
												// this ICD should be put.
	FLMUINT		uiKeyComponent;		// Which key component is this?  0 means
												// it is not a key component.
	ICD *			pNextKeyComponent;	// Next key component ICD.  Also used to
												// link context components
	ICD *			pPrevKeyComponent;	// Previous key component ICD.  Also used
												// to link context components
	FLMUINT		uiDataComponent;		// Which data component is this?  0 means
												// it is not a data component.
	ICD *			pNextDataComponent;	// Next data component ICD.
	ICD *			pPrevDataComponent;	// Previous data component ICD.
	FLMUINT		uiLimit;					// Zero or # of characters/bytes to limit.
#define ICD_DEFAULT_LIMIT					128
#define ICD_DEFAULT_SUBSTRING_LIMIT		48

} ICD;

#define ICD_VALUE							0x00000010	// Value agrees with parsing syntax
#define ICD_EACHWORD						0x00000020	// Index each and every word in the field
#define ICD_PRESENCE						0x00000040	// Index the tag and NOT the value
#define ICD_METAPHONE					0x00000080	// Index words of text strings using 
																// metaphone values
#define ICD_IS_ATTRIBUTE				0x00000100	// ICD is an attribute
#define ICD_REQUIRED_PIECE				0x00000200	// Required piece (not optional)
#define ICD_REQUIRED_IN_SET 			0x00000400	// Required within a set of fields.
#define ICD_SUBSTRING					0x00000800	// Index all substrings pieces
#define ICD_ESC_CHAR						0x00001000	// Placehold so that a query can parse the input
																// string and find a literal '*' or '\\'.
																// Not specified in dictionary or held in ICD
																// Only a temporary flag.
#define ICD_DESCENDING					0x00002000	// Sort in descending order.
#define ICD_MISSING_HIGH				0x00004000	// Sort missing components high instead of low.

FINLINE FLMUINT icdGetDataType(
	ICD *	pIcd)
{
	return( pIcd->uiFlags & 0x0F);
}

FINLINE void icdSetDataType(
	ICD *		pIcd,
	FLMUINT	uiDataType)
{
	pIcd->uiFlags = (pIcd->uiFlags & 0xFFFFFFF0) | (uiDataType & 0xF);
}

/****************************************************************************
Struct:	IX_ITEM		(Indexed Item - Attribute or Element)
Desc:		This structure is used to track all indexed attributes and elements
			whose numbers are greater than or equal to FLM_LOW_EXT_ELEMENT_NUM
			(for elements) or FLM_LOW_EXT_ATTRIBUTE_NUM (for attributes).
****************************************************************************/
typedef struct IndexedItem
{
	FLMUINT	uiDictNum;
	ICD *		pFirstIcd;
} IX_ITEM;

/****************************************************************************
Struct:	RESERVED_TAG_NAME
Desc:		This structure is used strictly to set up a static table (see
			fntable.cpp) which lists all reserved tag numbers
			and their types.
****************************************************************************/
typedef struct ReservedTag
{
	const char *	pszTagName;
	FLMUINT			uiTagNum;
	FLMUINT			uiDataType;
	FLMUNICODE *	puzNamespace;
} RESERVED_TAG_NAME;

/**************************************************************************
Desc:	This class is the FLAIM dictionary class.
**************************************************************************/
class F_Dict : public F_Object
{
public:

	// Constructor and destructor

	F_Dict();

	~F_Dict();
	
	void resetDict( void);

	RCODE getElement(
		F_Db *				pDb,
		FLMUINT				uiElementNum,
		F_AttrElmInfo *	pElmInfo);

	RCODE getAttribute(
		F_Db *				pDb,
		FLMUINT				uiAttributeNum,
		F_AttrElmInfo *	pAttrInfo);

	RCODE getNextElement(
		F_Db *				pDb,
		FLMUINT *			puiElementNum,
		F_AttrElmInfo *	pElmInfo);

	RCODE getNextAttribute(
		F_Db *				pDb,
		FLMUINT *			puiAttributeNum,
		F_AttrElmInfo *	pAttrInfo);

	FINLINE FLMUINT getCollectionCount(
		FLMBOOL	bCountPredefined)
	{
		FLMUINT	uiCount = m_uiHighestCollectionNum
								 ? m_uiHighestCollectionNum -
									m_uiLowestCollectionNum + 1
								 : 0;
		if (bCountPredefined)
		{

			// Add 2 for the FLM_DATA_COLLECTION and
			// FLM_DICT_COLLECTION

			uiCount += 2;
		}
		return( uiCount);
	}

	RCODE getCollection(
		FLMUINT					uiCollectionNum,
		F_COLLECTION **		ppCollection,
		FLMBOOL					bOfflineOk = FALSE);

	RCODE getPrefixId(
		F_Db *					pDb,
		const FLMUNICODE *	puzPrefix,
		FLMUINT *				puiPrefixId);

	RCODE getPrefixId(
		F_Db *					pDb,
		const char *			pszPrefix,
		FLMUINT *				puiPrefixId);

	FINLINE RCODE getPrefix(
		FLMUINT				uiPrefixId,
		FLMUNICODE *		puzPrefixBuf,
		FLMUINT				uiBufSize,
		FLMUINT *			puiCharsReturned)
	{
		return( getPrefix( TRUE, uiPrefixId, (void *)puzPrefixBuf,
			uiBufSize, puiCharsReturned));
	}

	FINLINE RCODE getPrefix(
		FLMUINT				uiPrefixId,
		char *				pszPrefixBuf,
		FLMUINT				uiBufSize,
		FLMUINT *			puiCharsReturned)
	{
		return( getPrefix( FALSE, uiPrefixId, (void *)pszPrefixBuf,
			uiBufSize, puiCharsReturned));
	}

	RCODE getPrefix(
		FLMUINT				uiPrefixId,
		F_PREFIX **			ppPrefix);

	RCODE getEncDefId(
		F_Db *					pDb,
		const FLMUNICODE *	puzEncDef,
		FLMUINT *				puiEncDefId);

	RCODE getEncDefId(
		F_Db *				pDb,
		const char *		pszEncDef,
		FLMUINT *			puiEncDefId);

	FINLINE RCODE getEncDef(
		FLMUINT				uiEncDefId,
		FLMUNICODE *		puzEncDefBuf,
		FLMUINT				uiBufSize,
		FLMUINT *			puiCharsReturned)
	{
		return getEncDef( TRUE,
								uiEncDefId,
								(void *)puzEncDefBuf,
								uiBufSize,
								puiCharsReturned);
	}

	FINLINE RCODE getEncDef(
		FLMUINT				uiEncDefId,
		char *				pszEncDefBuf,
		FLMUINT				uiBufSize,
		FLMUINT *			puiCharsReturned)
	{
		return getEncDef( FALSE,
								uiEncDefId,
								(void *)pszEncDefBuf,
								uiBufSize,
								puiCharsReturned);
	}

	RCODE getEncDef(
		FLMUINT				uiEncDefId,
		F_ENCDEF **			ppEncDef);

	RCODE getDefinitionDoc(
		F_Db *				pDb,
		FLMUINT				uiTag,
		FLMUINT				uiDictId,
		F_DOMNode **		ppDoc);

	FINLINE FLMUINT getPrefixCount(void)
	{
		FLMUINT	uiCount = m_uiHighestPrefixNum
								 ? m_uiHighestPrefixNum - m_uiLowestPrefixNum + 1
								 : 0;
		return( uiCount);
	}

	FINLINE FLMUINT getEncDefCount(void)
	{
		FLMUINT	uiCount = m_uiHighestEncDefNum
								 ? m_uiHighestEncDefNum - m_uiLowestEncDefNum + 1
								 : 0;
		return( uiCount);
	}

	FINLINE FLMUINT getIndexCount(
		FLMBOOL	bCountPredefined)
	{
		FLMUINT	uiCount = m_uiHighestIxNum
								 ? m_uiHighestIxNum - m_uiLowestIxNum + 1
								 : 0;
		if (bCountPredefined)
		{

			// Add 2 for the pre-defined indexes.

			uiCount += 2;
		}
		return( uiCount);
	}

	FINLINE FLMUINT getIxdOffset(
		FLMUINT	uiIndexNum
		)
	{
		if (uiIndexNum <= XFLM_MAX_INDEX_NUM)
		{
			return( uiIndexNum - m_uiLowestIxNum + 2);
		}
		else
		{
			switch (uiIndexNum)
			{
				case XFLM_DICT_NUMBER_INDEX:
					return( 0);
				case XFLM_DICT_NAME_INDEX:
					return( 1);
				default:
					flmAssert( 0);
					return( 0xFFFF);
			}
		}
	}

	RCODE getIndex(
		FLMUINT		uiIndexNum,
		LFILE **		ppLFile,
		IXD **		ppIxd,
		FLMBOOL		bOfflineOk = FALSE);

	IXD * getNextIndex(
		FLMUINT	uiIndexNum,
		FLMBOOL	bOkToGetPredefined);

	F_COLLECTION * getNextCollection(
		FLMUINT	uiCollectionNum,
		FLMBOOL	bOkToGetPredefined);

	FINLINE ATTR_ELM_DEF * getElementDef(
		FLMUINT	uiElementNum)
	{
		ATTR_ELM_DEF *	pElementDef = NULL;

		if (uiElementNum >= m_uiLowestElementNum &&
			 uiElementNum <= m_uiHighestElementNum)
		{
			pElementDef = &m_pElementDefTbl [uiElementNum - m_uiLowestElementNum];
			if (pElementDef && !attrElmGetState( pElementDef))
			{
				pElementDef = NULL;
			}
		}
		return( pElementDef);
	}

	FINLINE ATTR_ELM_DEF * getReservedElementDef(
		FLMUINT	uiElementNum)
	{
		ATTR_ELM_DEF *	pElementDef = &m_pReservedElementDefTbl [uiElementNum -
											XFLM_FIRST_RESERVED_ELEMENT_TAG];
		if (!attrElmGetState( pElementDef))
		{
			pElementDef = NULL;
		}
		return( pElementDef);
	}

	FINLINE ATTR_ELM_DEF * getAttributeDef(
		FLMUINT	uiAttributeNum)
	{
		ATTR_ELM_DEF *	pAttributeDef = NULL;

		if (uiAttributeNum >= m_uiLowestAttributeNum &&
			 uiAttributeNum <= m_uiHighestAttributeNum)
		{
			pAttributeDef = &m_pAttributeDefTbl [uiAttributeNum - m_uiLowestAttributeNum];
			if (pAttributeDef && !attrElmGetState( pAttributeDef))
			{
				pAttributeDef = NULL;
			}
		}
		return( pAttributeDef);
	}

	FINLINE ATTR_ELM_DEF * getReservedAttributeDef(
		FLMUINT	uiAttributeNum)
	{
		ATTR_ELM_DEF *	pAttributeDef = &m_pReservedAttributeDefTbl [uiAttributeNum -
											XFLM_FIRST_RESERVED_ATTRIBUTE_TAG];
		if (!attrElmGetState( pAttributeDef))
		{
			pAttributeDef = NULL;
		}
		return( pAttributeDef);
	}

	void linkToDatabase(
		F_Database *	pDatabase);

	void unlinkFromDatabase( void);

	RCODE linkIcdInChain(
		ICD *	pIcd);

	RCODE linkIcds(
		ICD *	pIcdTree);

	void unlinkIcdFromChain(
		ICD *	pIcd);

	void unlinkIcds(
		ICD *	pIcdTree);

	FINLINE FLMUINT getUseCount( void)
	{
		return( m_uiUseCount);
	}

	FINLINE FLMUINT decrUseCount( void)
	{
		return( --m_uiUseCount);
	}

	FINLINE void incrUseCount( void)
	{
		m_uiUseCount++;
	}

	FINLINE F_Dict * getPrev( void)
	{
		return( m_pPrev);
	}

	FINLINE F_Dict * getNext( void)
	{
		return( m_pNext);
	}

	FINLINE F_Database * getDatabase( void)
	{
		return( m_pDatabase);
	}

	RCODE copyIXD(
		IXD **		ppDestIxd,
		IXD *			pSrcIxd);

	RCODE cloneDict(
		F_Dict *	pSrcDict);

	RCODE checkElementReferences(				// was checkReferences
		FLMUINT		uiElementNum);

	RCODE checkAttributeReferences(			// was checkReferences
		FLMUINT		uiAttributeNum);

	RCODE checkCollectionReferences(
		FLMUINT		uiCollectionNum);

	RCODE updateDict(
		F_Db *			pDb,
		FLMUINT			uiDictType,
		FLMUINT64		ui64DocumentID,
		FLMUINT			uiDictNumber,
		FLMBOOL			bOpeningDict,
		FLMBOOL			bDeleting);

	RCODE setupPredefined(
		FLMUINT	uiDefaultLanguage);

	FINLINE FLMUINT getDictSeq( void)
	{
		return( m_uiDictSeq);
	}

	RCODE allocNameTable( void);

	FINLINE F_NameTable * getNameTable( void)
	{
		return( m_pNameTable);
	}

	RCODE allocElementTable(
		FLMUINT	uiLowestElementNum,
		FLMUINT	uiHighestElementNum);

	RCODE allocAttributeTable(
		FLMUINT	uiLowestAttributeNum,
		FLMUINT	uiHighestAttributeNum);

	RCODE allocIndexTable(
		FLMUINT	uiLowestIndexNum,
		FLMUINT	uiHighestIndexNum);

	RCODE allocPrefixTable(
		FLMUINT	uiLowestPrefixNum,
		FLMUINT	uiHighestPrefixNum);

	RCODE allocEncDefTable(
		FLMUINT	uiLowestEncDefNum,
		FLMUINT	uiHighestEncDefNum);

	RCODE allocCollectionTable(
		FLMUINT	uiLowestCollectionNum,
		FLMUINT	uiHighestCollectionNum);

private:

	RCODE reallocTbl(
		FLMUINT		uiNewId,
		FLMUINT		uiElementSize,
		void **		ppvTbl,
		FLMUINT *	puiLowest,
		FLMUINT *	puiHighest,
		FLMUINT		uiAdjustFactor,
		FLMUINT		uiMaxId);

	RCODE updateElementDef(
		F_Db *		pDb,
		FLMUINT64	ui64DocumentID,
		FLMUINT		uiElementNumber,
		FLMBOOL		bOpeningDict,
		FLMBOOL		bDeleting);

	RCODE updateAttributeDef(
		F_Db *		pDb,
		FLMUINT64	ui64DocumentID,
		FLMUINT		uiAttributeNumber,
		FLMBOOL		bOpeningDict,
		FLMBOOL		bDeleting);

	RCODE updateIndexDef(
		F_Db *		pDb,
		FLMUINT64	ui64DocumentID,
		FLMUINT		uiIndexNumber,
		FLMBOOL		bOpeningDict,
		FLMBOOL		bDeleting);

	RCODE updateCollectionDef(
		F_Db *			pDb,
		FLMUINT64		ui64DocumentID,
		FLMUINT			uiCollectionNumber,
		FLMBOOL			bOpeningDict,
		FLMBOOL			bDeleting);

	RCODE updatePrefixDef(
		F_Db *			pDb,
		FLMUINT64		ui64DocumentID,
		FLMUINT			uiPrefixNum,
		FLMBOOL			bOpeningDict,
		FLMBOOL			bDeleting);

	RCODE updateEncDef(
		F_Db *			pDb,
		FLMUINT64		ui64DocumentID,
		FLMUINT			uiEncDefNum,
		FLMBOOL			bOpeningDict,
		FLMBOOL			bDeleting);

	IX_ITEM * findIxItem(
		IX_ITEM *	pIxTbl,
		FLMUINT		uiNumItems,
		FLMUINT		uiTagNum,
		FLMUINT *	puiInsertPos = NULL);

	FINLINE IX_ITEM * findIxElement(
		FLMUINT		uiElementNum,
		FLMUINT *	puiInsertPos = NULL)
	{
		return( findIxItem( m_pIxElementTbl, m_uiNumIxElements,
								uiElementNum, puiInsertPos));
	}

	FINLINE IX_ITEM * findIxAttribute(
		FLMUINT		uiAttributeNum,
		FLMUINT *	puiInsertPos = NULL)
	{
		return( findIxItem( m_pIxAttributeTbl, m_uiNumIxAttributes,
								uiAttributeNum, puiInsertPos));
	}

	RCODE getExtElement(
		F_Db *				pDb,
		FLMUINT64			ui64DocumentID,
		FLMUINT				uiElementNum,
		F_AttrElmInfo *	pElmInfo);

	RCODE getExtAttribute(
		F_Db *				pDb,
		FLMUINT64			ui64DocumentID,
		FLMUINT				uiAttributeNum,
		F_AttrElmInfo *	pAttrInfo);

	void setExtElementFirstIcd(
		FLMUINT	uiElementNum,
		ICD *		pFirstIcd);

	void setExtAttributeFirstIcd(
		FLMUINT	uiAttributeNum,
		ICD *		pFirstIcd);

	FINLINE EXT_ATTR_ELM_DEF * getExtElementDef(
		FLMUINT	uiElementNum)
	{
		return( &m_pExtElementDefTbl [uiElementNum %
													m_uiExtElementDefTblSize]);
	}

	FINLINE EXT_ATTR_ELM_DEF * getExtAttributeDef(
		FLMUINT	uiAttributeNum)
	{
		return( &m_pExtAttributeDefTbl [uiAttributeNum %
													m_uiExtAttributeDefTblSize]);
	}

	RCODE getNextDictNumNodeIds(			// fdict.cpp
		F_Db *			pDb);

	RCODE createNextDictNums(				// fdict.cpp
		F_Db *			pDb);

	RCODE allocNextDictNum(					// fdict.cpp
		F_Db *			pDb,
		FLMUINT			uiDictType,
		FLMUINT *		puiDictNumber);

	RCODE setNextDictNum(					// fdict.cpp
		F_Db *			pDb,
		FLMUINT			uiDictType,
		FLMUINT			uiDictNumber);

	RCODE getPrefix(							// fdict.cpp
		FLMBOOL				bUnicode,
		FLMUINT				uiPrefixId,
		void *				pvPrefixBuf,
		FLMUINT				uiBufSize,
		FLMUINT *			puiCharsReturned);

	RCODE getEncDef(							// fdict.cpp
		FLMBOOL				bUnicode,
		FLMUINT				uiEncDefId,
		void *				pvEncDefBuf,
		FLMUINT				uiBufSize,
		FLMUINT *			puiCharsReturned);

	F_Dict *					m_pNext;			// Pointer to next F_Dict object in the list,
													// if any.  All versions of a dictionary that
													// are currently in use are linked together.
													// Usually, there will be only one local
													// dictionary in the list.
	F_Dict *					m_pPrev;			// Previous F_Dict object in the list.
	F_Database *			m_pDatabase;	// database this dictionary is associated with.
													// A null value means it is not yet linked
													// to a database.
	FLMUINT					m_uiDictSeq;	// This is the sequence number of the
													// dictionary
	F_Pool  					m_dictPool;		// Pool for all allocations except tables.

	// Fixed element definition table - used for elements whose tag numbers
	// are less than or equal to FLM_HIGH_FIXED_ELEMENT_NUM

	ATTR_ELM_DEF *			m_pElementDefTbl;
	FLMUINT					m_uiLowestElementNum;
	FLMUINT					m_uiHighestElementNum;

	// Reserved element definition table - used for elements whose tag numbers
	// are in the "reserved tag" range.

	ATTR_ELM_DEF *			m_pReservedElementDefTbl;

	// Extended element definition table - used for elements whose tag
	// numbers are greater than or equal to FLM_LOW_EXT_ELEMENT_NUM

	EXT_ATTR_ELM_DEF *	m_pExtElementDefTbl;
	FLMUINT					m_uiExtElementDefTblSize;
	F_MUTEX					m_hExtElementDefMutex;

	// Table for tracking ALL indexed elements whose tag number is
	// greater than or equal to FLM_LOW_EXT_ELEMENT_NUM.

	IX_ITEM *				m_pIxElementTbl;
	FLMUINT					m_uiIxElementTblSize;
	FLMUINT					m_uiNumIxElements;

	// Fixed attribute definition table - used for attributes whose tag numbers
	// are less than or equal to FLM_HIGH_FIXED_ATTRIBUTE_NUM

	ATTR_ELM_DEF *			m_pAttributeDefTbl;
	FLMUINT					m_uiLowestAttributeNum;
	FLMUINT					m_uiHighestAttributeNum;

	// Reserved attribute definition table - used for attributes whose tag
	// numbers are in the "reserved tag" range.

	ATTR_ELM_DEF *			m_pReservedAttributeDefTbl;

	// Extended attribute definition table - used for attributes whose tag
	// numbers are greater than or equal to FLM_LOW_EXT_ATTRIBUTE_NUM

	EXT_ATTR_ELM_DEF *	m_pExtAttributeDefTbl;
	FLMUINT					m_uiExtAttributeDefTblSize;
	F_MUTEX					m_hExtAttributeDefMutex;

	// Table for tracking ALL indexed attributes whose tag number is
	// greater than or equal to FLM_LOW_EXT_ATTRIBUTE_NUM.

	IX_ITEM *				m_pIxAttributeTbl;
	FLMUINT					m_uiIxAttributeTblSize;
	FLMUINT					m_uiNumIxAttributes;

	// Pre-defined collections

	F_COLLECTION *			m_pDictCollection;
	F_COLLECTION *			m_pDataCollection;
	F_COLLECTION *			m_pMaintCollection;

	// User defined Collections

	F_COLLECTION **		m_ppCollectionTbl;
	FLMUINT					m_uiLowestCollectionNum;
	FLMUINT					m_uiHighestCollectionNum;

	// User defined prefixes

	F_PREFIX **				m_ppPrefixTbl;
	FLMUINT					m_uiLowestPrefixNum;
	FLMUINT					m_uiHighestPrefixNum;

	// User defined encryption defs

	F_ENCDEF **				m_ppEncDefTbl;
	FLMUINT					m_uiLowestEncDefNum;
	FLMUINT					m_uiHighestEncDefNum;

	// Pre-defined indexes

	IXD *						m_pNameIndex;
	IXD *						m_pNumberIndex;

	// User defined indexes

	IXD **					m_ppIxdTbl;
	FLMUINT					m_uiLowestIxNum;
	FLMUINT					m_uiHighestIxNum;
	ICD *						m_pRootIcdList;

	FLMUINT					m_uiUseCount;	// Number of F_Db structures currently
													// pointing to this dictionary.

	// Name table

	F_NameTable *			m_pNameTable;

	// Keep track of whether or not the database is operating in limited mode.  This
	// field is copied from the database when the dictionary is created or cloned

	FLMBOOL					m_bInLimitedMode;		

friend class F_Database;
friend class F_Db;
friend class F_Query;
friend class F_DbCheck;
friend class F_BTreeInfo;
friend class F_AttrItem;
};

RCODE fdictGetDataType(						// fdict.cpp
	char *		pszDataType,
	FLMUINT *	puiDataType);

const char * fdictGetDataTypeStr(		// fdict.cpp
	FLMUINT 		uiDataType);

#endif // #ifndef FDICT_H
