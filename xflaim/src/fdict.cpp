//------------------------------------------------------------------------------
// Desc:	Routines to access anything in the dictionary
// Tabs:	3
//
// Copyright (c) 1995-2007 Novell, Inc. All Rights Reserved.
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

// Data type strings

const char * fdictDataTypes[ XFLM_NUM_OF_TYPES + 1] =
{
	XFLM_NODATA_OPTION_STR,
	XFLM_STRING_OPTION_STR,
	XFLM_INTEGER_OPTION_STR,
	XFLM_BINARY_OPTION_STR,
	NULL
};

extern RESERVED_TAG_NAME FlmReservedElementTags[];
extern RESERVED_TAG_NAME FlmReservedAttributeTags[];

FSTATIC void fdictInsertInIcdChain(
	ICD **	ppFirstIcd,
	ICD *		pIcd);

FSTATIC void fdictRemoveFromIcdChain(
	ICD **	ppFirstIcd,
	ICD *		pIcd);

FSTATIC RCODE fdictCopyCollection(
	F_Pool *				pDictPool,
	F_COLLECTION **	ppDestCollection,
	F_COLLECTION *		pSrcCollection);

FSTATIC RCODE fdictCopyPrefix(
	F_Pool *				pDictPool,
	F_PREFIX **			ppDestPrefix,
	F_PREFIX *			pSrcPrefix);

FSTATIC RCODE fdictCopyEncDef(
	F_Pool *				pDictPool,
	F_ENCDEF **			ppDestEncDef,
	F_ENCDEF *			pSrcEncDef);

FSTATIC char * fdictGetOption(
	char **	ppszSrc);

FSTATIC RCODE isIndexComponent(
	F_Db *		pDb,
	F_DOMNode *	pNode,
	FLMBOOL *	pbIsIndexComponent,
	FLMUINT *	puiElementId);

FSTATIC FLMBOOL indexDefsSame(
	IXD *	pOldIxd,
	IXD *	pNewIxd);


#define MAX_ENC_TYPES	2
// NOTE:  If you change the arrangement of the values in this array, make sure
// you search the entire codebase for references to DDEncOpts and fdictLegalEncDefTypes
// and verify that the changes won't cause problems.
const char * DDEncOpts[MAX_ENC_TYPES] = {
	XFLM_ENC_AES_OPTION_STR,				/* AES */
	XFLM_ENC_DES3_OPTION_STR				/* Triple DES */
	};

/***************************************************************************
Desc:	Constructor
***************************************************************************/
F_Dict::F_Dict()
{
	m_pNext = NULL;
	m_pPrev = NULL;
	m_pDatabase = NULL;
	m_uiDictSeq = 0;
	m_dictPool.poolInit( 1024);

	m_pElementDefTbl = NULL;
	m_uiLowestElementNum = 0;
	m_uiHighestElementNum = 0;

	m_pReservedElementDefTbl = NULL;

	m_pExtElementDefTbl = NULL;
	m_uiExtElementDefTblSize = 0;
	m_hExtElementDefMutex = F_MUTEX_NULL;

	m_pIxElementTbl = NULL;
	m_uiIxElementTblSize = 0;
	m_uiNumIxElements = 0;

	m_pAttributeDefTbl = NULL;
	m_uiLowestAttributeNum = 0;
	m_uiHighestAttributeNum = 0;

	m_pReservedAttributeDefTbl = NULL;

	m_pExtAttributeDefTbl = NULL;
	m_uiExtAttributeDefTblSize = 0;
	m_hExtAttributeDefMutex = F_MUTEX_NULL;

	m_pIxAttributeTbl = NULL;
	m_uiIxAttributeTblSize = 0;
	m_uiNumIxAttributes = 0;

	m_pDictCollection = NULL;
	m_pDataCollection = NULL;
	m_pMaintCollection = NULL;

	m_ppCollectionTbl = NULL;
	m_uiLowestCollectionNum = 0;
	m_uiHighestCollectionNum = 0;

	m_ppPrefixTbl = NULL;
	m_uiLowestPrefixNum = 0;
	m_uiHighestPrefixNum = 0;

	m_ppEncDefTbl = NULL;
	m_uiLowestEncDefNum = 0;
	m_uiHighestEncDefNum = 0;

	m_pNameIndex = NULL;
	m_pNumberIndex = NULL;

	m_ppIxdTbl = NULL;
	m_uiLowestIxNum = 0;
	m_uiHighestIxNum = 0;

	m_pNameTable = NULL;
	m_pRootIcdList = NULL;

	// Whenever an F_Dict is allocated, it is always immediately
	// used by an F_Db.

	m_uiUseCount = 1;
}

/***************************************************************************
Desc:
***************************************************************************/
F_Dict::~F_Dict()
{
	resetDict();
}

/***************************************************************************
Desc:	Clear the dictionary object so it can be reused.  NOTE: This function
		also needs to do all the freeing up that the destructor would do
		because it is called by the destructor.
***************************************************************************/
void F_Dict::resetDict( void)
{
	FLMUINT		uiLoop;

	f_free( &m_pElementDefTbl);
	m_uiLowestElementNum = 0;
	m_uiHighestElementNum = 0;

	f_free( &m_pExtElementDefTbl);
	m_uiExtElementDefTblSize = 0;
	if (m_hExtElementDefMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hExtElementDefMutex);
	}

	f_free( &m_pReservedElementDefTbl);

	f_free( &m_pIxElementTbl);
	m_uiIxElementTblSize = 0;
	m_uiNumIxElements = 0;

	f_free( &m_pAttributeDefTbl);
	m_uiLowestAttributeNum = 0;
	m_uiHighestAttributeNum = 0;

	f_free( &m_pExtAttributeDefTbl);
	m_uiExtAttributeDefTblSize = 0;
	if (m_hExtAttributeDefMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hExtAttributeDefMutex);
	}

	f_free( &m_pReservedAttributeDefTbl);

	f_free( &m_pIxAttributeTbl);
	m_uiIxAttributeTblSize = 0;
	m_uiNumIxAttributes = 0;

	f_free( &m_ppIxdTbl);
	m_uiLowestIxNum = 0;
	m_uiHighestIxNum = 0;

	m_pNameIndex = NULL;
	m_pNumberIndex = NULL;

	f_free( &m_ppCollectionTbl);
	m_uiLowestCollectionNum = 0;
	m_uiHighestCollectionNum = 0;

	f_free( &m_ppPrefixTbl);
	m_uiLowestPrefixNum = 0;
	m_uiHighestPrefixNum = 0;

	for( uiLoop = 0; 
		  uiLoop <= (m_uiHighestEncDefNum - m_uiLowestEncDefNum); uiLoop++)
	{
		if (m_ppEncDefTbl && m_ppEncDefTbl[ uiLoop] && 
			(*m_ppEncDefTbl[ uiLoop]).pCcs)
		{
			(*m_ppEncDefTbl[ uiLoop]).pCcs->Release();
		}
	}
	
	f_free( &m_ppEncDefTbl);
	m_uiLowestEncDefNum = 0;
	m_uiHighestEncDefNum = 0;

	m_pDictCollection = NULL;
	m_pDataCollection = NULL;
	m_pMaintCollection = NULL;

	m_dictPool.poolFree();
	m_dictPool.poolInit( 1024);

	if (m_pNameTable)
	{
		m_pNameTable->Release();
		m_pNameTable = NULL;
	}
	m_pRootIcdList = NULL;
}

/***************************************************************************
Desc:
***************************************************************************/
RCODE F_Dict::allocNameTable( void)
{
	if ((m_pNameTable = f_new F_NameTable) == NULL)
	{
		return( RC_SET( NE_XFLM_MEM));
	}
	return( NE_XFLM_OK);
}

/***************************************************************************
Desc:	Get the first ICD, if any for an element.  NOTE: This is only called
		for element numbers greater or equal to FLM_LOW_EXT_ELEMENT_NUM, so
		we have to look in the m_pIxElementTbl table.
***************************************************************************/
IX_ITEM * F_Dict::findIxItem(
	IX_ITEM *	pIxTbl,
	FLMUINT		uiNumItems,
	FLMUINT		uiDictNum,
	FLMUINT *	puiInsertPos
	)
{
	IX_ITEM *	pIxItem = NULL;
	FLMUINT		uiLow;
	FLMUINT		uiMid;
	FLMUINT		uiHigh;
	FLMUINT		uiTblDictNum;

	// Do binary search in the table

	if (!uiNumItems)
	{
		if (puiInsertPos)
		{
			*puiInsertPos = 0;
		}
		goto Exit;
	}

	uiHigh = --uiNumItems;
	uiLow = 0;
	for (;;)
	{
		uiMid = (uiLow + uiHigh) / 2;

		uiTblDictNum = pIxTbl [uiMid].uiDictNum;
		if (uiTblDictNum == uiDictNum)
		{

			// Found Match

			if (puiInsertPos)
			{
				*puiInsertPos = uiMid;
			}
			pIxItem = &pIxTbl [uiMid];
			goto Exit;
		}

		// Check if we are done

		if (uiLow >= uiHigh)
		{
			if (puiInsertPos)
			{
				*puiInsertPos = (uiDictNum < uiTblDictNum)
									 ? uiMid
									 : uiMid + 1;
			}
			goto Exit;
		}

		if (uiDictNum < uiTblDictNum)
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
			if (uiMid == uiNumItems)
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

	return( pIxItem);
}

/***************************************************************************
Desc:	Get information for an extended element - which are elements whose tag
		number is greater than or equal to FLM_LOW_EXT_ELEMENT_NUM
Note:	Populates data type, first Icd, and state
***************************************************************************/
RCODE F_Dict::getExtElement(
	F_Db *				pDb,
	FLMUINT64			ui64DocumentID,
	FLMUINT				uiElementNum,
	F_AttrElmInfo *	pElmInfo)
{
	RCODE						rc = NE_XFLM_OK;
	EXT_ATTR_ELM_DEF *	pExtElementDef;
	IX_ITEM *				pIxItem;
	FLMBOOL					bMutexLocked;

	// See if the element is in our extended list.  If not, bring it in.

	f_mutexLock( m_hExtElementDefMutex);
	bMutexLocked = TRUE;
	pExtElementDef = getExtElementDef( uiElementNum);
	if (pExtElementDef->uiDictNum != uiElementNum)
	{
		f_mutexUnlock( m_hExtElementDefMutex);
		bMutexLocked = FALSE;

		if (!ui64DocumentID)
		{
			F_DataVector	searchKey;
			F_DataVector	foundKey;

			// Find and read the element definition document
			// Need to lookup the document's ID, using the element number

			if (RC_BAD( rc = searchKey.setUINT( 0, ELM_ELEMENT_TAG)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = searchKey.setUINT( 1, uiElementNum)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = pDb->keyRetrieve( XFLM_DICT_NUMBER_INDEX,
					&searchKey, XFLM_EXACT, &foundKey)))
			{
				goto Exit;
			}

			ui64DocumentID = foundKey.getDocumentID();
		}

		// Must read the element definition document

		if (RC_BAD( rc = pDb->getElmAttrInfo( ELM_ELEMENT_TAG,
			ui64DocumentID, pElmInfo, TRUE, FALSE)))
		{
			goto Exit;
		}

		// See if the element is indexed.

		pIxItem = findIxElement( uiElementNum);

		// Relock the mutex and populate the extended structure
		// with the information we found.

		f_mutexLock( m_hExtElementDefMutex);
		bMutexLocked = TRUE;
		pExtElementDef->uiDictNum = uiElementNum;
		pExtElementDef->attrElmDef.uiFlags =
								(pElmInfo->m_uiDataType & ATTR_ELM_DATA_TYPE_MASK) |
								(pElmInfo->m_uiState & ATTR_ELM_STATE_MASK);
		pExtElementDef->attrElmDef.pFirstIcd = pIxItem
															? pIxItem->pFirstIcd
															: NULL;
	}

	pElmInfo->m_uiDataType = attrElmGetType( &pExtElementDef->attrElmDef);
	pElmInfo->m_pFirstIcd = pExtElementDef->attrElmDef.pFirstIcd;
	pElmInfo->m_uiState = attrElmGetState( &pExtElementDef->attrElmDef);

Exit:

	if (bMutexLocked)
	{
		f_mutexUnlock( m_hExtElementDefMutex);
	}

	return( rc);
}

/***************************************************************************
Desc:	Get information for an extended attribute - which are attributes whose
		tag number is greater than or equal to FLM_LOW_EXT_ATTRIBUTE_NUM
***************************************************************************/
RCODE F_Dict::getExtAttribute(
	F_Db *				pDb,
	FLMUINT64			ui64DocumentID,
	FLMUINT				uiAttributeNum,
	F_AttrElmInfo *	pAttrInfo)
{
	RCODE						rc = NE_XFLM_OK;
	EXT_ATTR_ELM_DEF *	pExtAttributeDef;
	IX_ITEM *				pIxItem;
	FLMBOOL					bMutexLocked;

	// See if the attribute is in our extended list.  If not, bring it in.

	f_mutexLock( m_hExtAttributeDefMutex);
	bMutexLocked = TRUE;
	pExtAttributeDef = getExtAttributeDef( uiAttributeNum);
	if (pExtAttributeDef->uiDictNum != uiAttributeNum)
	{
		f_mutexUnlock( m_hExtAttributeDefMutex);
		bMutexLocked = FALSE;

		if (!ui64DocumentID)
		{
			F_DataVector	searchKey;
			F_DataVector	foundKey;

			// Find and read the element definition document

			// Need to lookup the document's ID, using the element number

			if (RC_BAD( rc = searchKey.setUINT( 0, ELM_ATTRIBUTE_TAG)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = searchKey.setUINT( 1, uiAttributeNum)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = pDb->keyRetrieve( XFLM_DICT_NUMBER_INDEX,
					&searchKey, XFLM_EXACT, &foundKey)))
			{
				goto Exit;
			}

			ui64DocumentID = foundKey.getDocumentID();
		}

		// Must read the attribute definition document

		if (RC_BAD( rc = pDb->getElmAttrInfo( ELM_ATTRIBUTE_TAG,
			ui64DocumentID, pAttrInfo, TRUE, FALSE)))
		{
			goto Exit;
		}

		// See if the attribute is indexed.

		pIxItem = findIxAttribute( uiAttributeNum);

		// Relock the mutex and populate the extended structure
		// with the information we found.

		f_mutexLock( m_hExtAttributeDefMutex);
		bMutexLocked = TRUE;
		pExtAttributeDef->uiDictNum = uiAttributeNum;
		pExtAttributeDef->attrElmDef.uiFlags =
								(pAttrInfo->m_uiDataType & ATTR_ELM_DATA_TYPE_MASK) |
								(pAttrInfo->m_uiState & ATTR_ELM_STATE_MASK);
		pExtAttributeDef->attrElmDef.pFirstIcd = pIxItem
															? pIxItem->pFirstIcd
															: NULL;
	}

	pAttrInfo->m_uiDataType = attrElmGetType( &pExtAttributeDef->attrElmDef);
	pAttrInfo->m_pFirstIcd = pExtAttributeDef->attrElmDef.pFirstIcd;
	pAttrInfo->m_uiState = attrElmGetState( &pExtAttributeDef->attrElmDef);
	pAttrInfo->m_uiFlags = attrElmGetFlags( &pExtAttributeDef->attrElmDef);

Exit:

	if (bMutexLocked)
	{
		f_mutexUnlock( m_hExtAttributeDefMutex);
	}

	return( rc);
}

/***************************************************************************
Desc:	Get the element information.
Note:	Populates data type, first ICD, and state
***************************************************************************/
RCODE F_Dict::getElement(
	F_Db *				pDb,
	FLMUINT				uiElementNum,			// [in] Element Number to look up
	F_AttrElmInfo *	pElmInfo)
{
	RCODE				rc = NE_XFLM_OK;
	ATTR_ELM_DEF *	pElementDef;

	if (elementIsReservedTag( uiElementNum))
	{
		if ((pElementDef = getReservedElementDef( uiElementNum)) != NULL)
		{
			goto Get_Type_And_State;
		}
		else
		{
			rc = RC_SET( NE_XFLM_BAD_ELEMENT_NUM);
			goto Exit;
		}
	}
	else if (uiElementNum <= FLM_HIGH_FIXED_ELEMENT_NUM)
	{
		if ((pElementDef = getElementDef( uiElementNum)) != NULL)
		{
Get_Type_And_State:

			pElmInfo->m_uiDataType = attrElmGetType( pElementDef);
			pElmInfo->m_pFirstIcd = pElementDef->pFirstIcd;
			pElmInfo->m_uiState = attrElmGetState( pElementDef);
			pElmInfo->m_uiFlags = attrElmGetFlags( pElementDef);
		}
		else
		{
			rc = RC_SET( NE_XFLM_BAD_ELEMENT_NUM);
			goto Exit;
		}
	}
	else if (uiElementNum >= FLM_LOW_EXT_ELEMENT_NUM && m_pExtElementDefTbl)
	{
		if (RC_BAD( rc = getExtElement( pDb, 0, uiElementNum, pElmInfo)))
		{
			goto Exit;
		}
	}
	else
	{
		rc = RC_SET( NE_XFLM_BAD_ELEMENT_NUM);
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get the attribute information.
Note: Populates data type, first ICD, and state
***************************************************************************/
RCODE F_Dict::getAttribute(
	F_Db *				pDb,
	FLMUINT				uiAttributeNum,
	F_AttrElmInfo *	pAttrInfo)
{
	RCODE				rc = NE_XFLM_OK;
	ATTR_ELM_DEF *	pAttributeDef;

	if( attributeIsReservedTag( uiAttributeNum))
	{
		if ((pAttributeDef = getReservedAttributeDef( uiAttributeNum)) != NULL)
		{
			goto Get_Type_And_State;
		}
		else
		{
			rc = RC_SET( NE_XFLM_BAD_ATTRIBUTE_NUM);
			goto Exit;
		}
	}
	else if (uiAttributeNum <= FLM_HIGH_FIXED_ATTRIBUTE_NUM)
	{
		if ((pAttributeDef = getAttributeDef( uiAttributeNum)) != NULL)
		{
Get_Type_And_State:
			pAttrInfo->m_uiDataType = attrElmGetType( pAttributeDef);
			pAttrInfo->m_pFirstIcd = pAttributeDef->pFirstIcd;
			pAttrInfo->m_uiState = attrElmGetState( pAttributeDef);
			pAttrInfo->m_uiFlags = attrElmGetFlags( pAttributeDef);
		}
		else
		{
			rc = RC_SET( NE_XFLM_BAD_ATTRIBUTE_NUM);
			goto Exit;
		}
	}
	else if (uiAttributeNum >= FLM_LOW_EXT_ATTRIBUTE_NUM && m_pExtAttributeDefTbl)
	{
		if (RC_BAD( rc = getExtAttribute( pDb, 0, uiAttributeNum, pAttrInfo)))
		{
			goto Exit;
		}
	}
	else
	{
		rc = RC_SET( NE_XFLM_BAD_ATTRIBUTE_NUM);
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get the data type for any given element or attribute in the 
		current db
***************************************************************************/
RCODE F_Db::getDataType(
	FLMUINT			uiDictType,
	FLMUINT			uiNameId,
	FLMUINT *		puiDataType)
{
	RCODE				rc = NE_XFLM_OK;
	F_AttrElmInfo	defInfo;

	flmAssert( uiDictType == ELM_ELEMENT_TAG ||
				  uiDictType == ELM_ATTRIBUTE_TAG);

	if( RC_BAD( rc = 
		(uiDictType == ELM_ELEMENT_TAG 
			  ? m_pDict->getElement( this, uiNameId, &defInfo)
			  : m_pDict->getAttribute( this, uiNameId, &defInfo))))
	{
		goto Exit;
	}

	*puiDataType = defInfo.m_uiDataType;

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get the next element defined in the dictionary after the one that
		is passed in.
Note:	Populates data type, first ICD, and state
***************************************************************************/
RCODE F_Dict::getNextElement(
	F_Db *				pDb,
	FLMUINT *			puiElementNum,
	F_AttrElmInfo *	pElmInfo)
{
	RCODE					rc = NE_XFLM_OK;
	ATTR_ELM_DEF *		pElementDef = NULL;
	FLMUINT				uiElementNum = *puiElementNum;

	if (uiElementNum < m_uiLowestElementNum)
	{
		uiElementNum = m_uiLowestElementNum;
	}
	else
	{
		uiElementNum++;
	}

	while (uiElementNum >= m_uiLowestElementNum &&
			 uiElementNum <= m_uiHighestElementNum)
	{
		pElementDef = &m_pElementDefTbl [uiElementNum - m_uiLowestElementNum];
		if (attrElmGetState( pElementDef))
		{
			break;
		}
		else
		{
			pElementDef = NULL;
		}
		uiElementNum++;
	}

	if (pElementDef)
	{

		// At this point we know we have an element.

		*puiElementNum = uiElementNum;
		pElmInfo->m_uiDataType = attrElmGetType( pElementDef);
		pElmInfo->m_pFirstIcd = pElementDef->pFirstIcd;
		pElmInfo->m_uiState = attrElmGetState( pElementDef);
	}
	else
	{
		F_DataVector	searchKey;
		F_DataVector	foundKey;
		FLMUINT			uiType;

		// See if it is perchance in the extended list, if there
		// is one

		if (!m_pExtElementDefTbl)
		{
			rc = RC_SET( NE_XFLM_EOF_HIT);
			goto Exit;
		}

		searchKey.reset();
		foundKey.reset();
		if (RC_BAD( rc = searchKey.setUINT( 0, ELM_ELEMENT_TAG)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = searchKey.setUINT( 1, uiElementNum)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = pDb->keyRetrieve( XFLM_DICT_NUMBER_INDEX,
				&searchKey, XFLM_EXCL, &foundKey)))
		{
			if (rc == NE_XFLM_EOF_HIT ||
				 rc == NE_XFLM_BOF_HIT ||
				 rc == NE_XFLM_NOT_FOUND)
			{
				rc = RC_SET( NE_XFLM_EOF_HIT);
			}
			goto Exit;
		}

		// If it is not an element definition document, skip it
		// we are done - there are no more.

		if (RC_BAD( rc = foundKey.getUINT( 0, &uiType)))
		{
			if (rc == NE_XFLM_NOT_FOUND)
			{
				flmAssert( 0);
				rc = RC_SET( NE_XFLM_EOF_HIT);
			}
			goto Exit;
		}
		if (uiType != ELM_ELEMENT_TAG)
		{
			rc = RC_SET( NE_XFLM_EOF_HIT);
			goto Exit;
		}

		// Get the element number.

		if (RC_BAD( rc = foundKey.getUINT( 1, &uiElementNum)))
		{
			if (rc == NE_XFLM_NOT_FOUND)
			{
				flmAssert( 0);
				rc = RC_SET( NE_XFLM_EOF_HIT);
			}
			goto Exit;
		}

		// At this point, we know we found one, so we will populate
		// the extended element structure.

		if (RC_BAD( rc = getExtElement( pDb, foundKey.getDocumentID(),
								uiElementNum, pElmInfo)))
		{
			goto Exit;
		}
		*puiElementNum = uiElementNum;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get the next attribute defined in the dictionary after the one that
		is passed in.
Note:	Populates data type, first ICD, state, and flags
***************************************************************************/
RCODE F_Dict::getNextAttribute(
	F_Db *				pDb,
	FLMUINT *			puiAttributeNum,
	F_AttrElmInfo *	pAttrInfo)
{
	RCODE					rc = NE_XFLM_OK;
	ATTR_ELM_DEF *		pAttributeDef = NULL;
	FLMUINT				uiAttributeNum = *puiAttributeNum;

	if (uiAttributeNum < m_uiLowestAttributeNum)
	{
		uiAttributeNum = m_uiLowestAttributeNum;
	}
	else
	{
		uiAttributeNum++;
	}

	while (uiAttributeNum >= m_uiLowestAttributeNum &&
			 uiAttributeNum <= m_uiHighestAttributeNum)
	{
		pAttributeDef = &m_pAttributeDefTbl [uiAttributeNum - m_uiLowestAttributeNum];
		if (attrElmGetState( pAttributeDef))
		{
			break;
		}
		else
		{
			pAttributeDef = NULL;
		}
		uiAttributeNum++;
	}

	if (pAttributeDef)
	{

		// At this point we know we have an attribute.

		*puiAttributeNum = uiAttributeNum;
		pAttrInfo->m_uiDataType = attrElmGetType( pAttributeDef);
		pAttrInfo->m_pFirstIcd = pAttributeDef->pFirstIcd;
		pAttrInfo->m_uiState = attrElmGetState( pAttributeDef);
		pAttrInfo->m_uiFlags = attrElmGetFlags( pAttributeDef);
	}
	else
	{
		F_DataVector	searchKey;
		F_DataVector	foundKey;
		FLMUINT			uiType;

		// See if it is perchance in the extended list, if there
		// is one

		if (!m_pExtAttributeDefTbl)
		{
			rc = RC_SET( NE_XFLM_EOF_HIT);
			goto Exit;
		}

		searchKey.reset();
		foundKey.reset();
		if (RC_BAD( rc = searchKey.setUINT( 0, ELM_ATTRIBUTE_TAG)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = searchKey.setUINT( 1, uiAttributeNum)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = pDb->keyRetrieve( XFLM_DICT_NUMBER_INDEX,
				&searchKey, XFLM_EXCL, &foundKey)))
		{
			if (rc == NE_XFLM_EOF_HIT ||
				 rc == NE_XFLM_BOF_HIT ||
				 rc == NE_XFLM_NOT_FOUND)
			{
				rc = RC_SET( NE_XFLM_EOF_HIT);
			}
			goto Exit;
		}

		// If it is not an attribute definition document, skip it
		// we are done - there are no more.

		if (RC_BAD( rc = foundKey.getUINT( 0, &uiType)))
		{
			if (rc == NE_XFLM_NOT_FOUND)
			{
				flmAssert( 0);
				rc = RC_SET( NE_XFLM_EOF_HIT);
			}
			goto Exit;
		}
		if (uiType != ELM_ATTRIBUTE_TAG)
		{
			rc = RC_SET( NE_XFLM_EOF_HIT);
			goto Exit;
		}

		// Get the attribute number.

		if (RC_BAD( rc = foundKey.getUINT( 1, &uiAttributeNum)))
		{
			if (rc == NE_XFLM_NOT_FOUND)
			{
				flmAssert( 0);
				rc = RC_SET( NE_XFLM_EOF_HIT);
			}
			goto Exit;
		}

		// At this point, we know we found one, so we will populate
		// the extended attribute structure.

		if (RC_BAD( rc = getExtAttribute( pDb, foundKey.getDocumentID(),
			uiAttributeNum, pAttrInfo)))
		{
			goto Exit;
		}
		*puiAttributeNum = uiAttributeNum;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Get the collection given a collection number.
***************************************************************************/
RCODE F_Dict::getCollection(
	FLMUINT				uiCollectionNum,
	F_COLLECTION **	ppCollection,			// [out] optional
	FLMBOOL				bOfflineOk
	)
{
	RCODE					rc = NE_XFLM_OK;
	F_COLLECTION *		pCollection = NULL;

	// Try most commonly used one first - default data collection

	if( uiCollectionNum == XFLM_DATA_COLLECTION)
	{
		pCollection = m_pDataCollection;
	}
	else if( uiCollectionNum &&
				uiCollectionNum >= m_uiLowestCollectionNum &&
				uiCollectionNum <= m_uiHighestCollectionNum)
	{
		pCollection = m_ppCollectionTbl [uiCollectionNum -
											m_uiLowestCollectionNum];
	}
	else
	{
		switch (uiCollectionNum)
		{
			case XFLM_DICT_COLLECTION:
				pCollection = m_pDictCollection;
				break;
				
			case XFLM_MAINT_COLLECTION:
				pCollection = m_pMaintCollection;
				break;
		}
	}

	// If the collection is encrypted, and we are in limited mode, then it must
	// be offline.

	if (!pCollection)
	{
		if (ppCollection)
		{
			*ppCollection = NULL;
		}
		rc = RC_SET( NE_XFLM_BAD_COLLECTION);
		goto Exit;
	}
	else
	{
		if ((pCollection->lfInfo.uiEncId && m_bInLimitedMode) && !bOfflineOk)
		{
			rc = RC_SET( NE_XFLM_COLLECTION_OFFLINE);
		}
	
		if (ppCollection)
		{
			*ppCollection = pCollection;
		}
	}

Exit:

	return rc;
}

/***************************************************************************
Desc:		Returns the ID of a prefix (unicode buffer)
***************************************************************************/
RCODE F_Dict::getPrefixId(
	F_Db *					pDb,
	const FLMUNICODE *	puzPrefix,
	FLMUINT *				puiPrefixId)
{
	RCODE				rc = NE_XFLM_OK;
	F_DataVector	searchKey;
	F_DataVector	foundKey;

	// Get the prefix number - look up in the index

	if (RC_BAD( rc = searchKey.setUINT( 0, ELM_PREFIX_TAG)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = searchKey.setUnicode( 1, puzPrefix)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pDb->keyRetrieve( XFLM_DICT_NAME_INDEX,
				&searchKey, XFLM_EXACT, &foundKey)))
	{
		goto Exit;
	}

	// Data part of the retrieved key has the dictionary number
	// for this prefix

	if (RC_BAD( rc = foundKey.getUINT( 3, puiPrefixId)))
	{
		if (rc == NE_XFLM_NOT_FOUND)
		{
			flmAssert( 0);
			*puiPrefixId = 0;
			rc = NE_XFLM_OK;
		}
		else
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:		Returns the ID of a prefix (native buffer)
***************************************************************************/
RCODE F_Dict::getPrefixId(
	F_Db *				pDb,
	const char *		pszPrefix,
	FLMUINT *			puiPrefixId)
{
	RCODE				rc = NE_XFLM_OK;
	F_DataVector	searchKey;
	F_DataVector	foundKey;

	// Get the prefix number - look up in the index

	if (RC_BAD( rc = searchKey.setUINT( 0, ELM_PREFIX_TAG)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = searchKey.setUTF8( 1, (FLMBYTE *)pszPrefix)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pDb->keyRetrieve( XFLM_DICT_NAME_INDEX,
				&searchKey, XFLM_EXACT, &foundKey)))
	{
		goto Exit;
	}

	// Data part of the retrieved key has the dictionary number
	// for this prefix

	if (RC_BAD( rc = foundKey.getUINT( 3, puiPrefixId)))
	{
		if (rc == NE_XFLM_NOT_FOUND)
		{
			flmAssert( 0);
			*puiPrefixId = 0;
			rc = NE_XFLM_OK;
		}
		else
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:		Returns the ID of an encryption definition (unicode buffer)
***************************************************************************/
RCODE F_Dict::getEncDefId(
	F_Db *					pDb,
	const FLMUNICODE *	puzEncDef,
	FLMUINT *				puiEncDefId)
{
	RCODE				rc = NE_XFLM_OK;
	F_DataVector	searchKey;
	F_DataVector	foundKey;

	// Get the encdef number - look up in the name index

	if (RC_BAD( rc = searchKey.setUINT( 0, ELM_ENCDEF_TAG)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = searchKey.setUnicode( 1, puzEncDef)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pDb->keyRetrieve( XFLM_DICT_NAME_INDEX,
				&searchKey, XFLM_EXACT, &foundKey)))
	{
		goto Exit;
	}

	// Data part of the retrieved key has the dictionary number
	// for this encdef

	if (RC_BAD( rc = foundKey.getUINT( 3, puiEncDefId)))
	{
		if (rc == NE_XFLM_NOT_FOUND)
		{
			flmAssert( 0);
			*puiEncDefId = 0;
			rc = NE_XFLM_OK;
		}
		else
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:		Returns the ID of an encryption definition (native buffer)
***************************************************************************/
RCODE F_Dict::getEncDefId(
	F_Db *				pDb,
	const char *		pszEncDef,
	FLMUINT *			puiEncDefId)
{
	RCODE				rc = NE_XFLM_OK;
	F_DataVector	searchKey;
	F_DataVector	foundKey;

	// Get the encdef number - look up in the name index

	if (RC_BAD( rc = searchKey.setUINT( 0, ELM_ENCDEF_TAG)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = searchKey.setUTF8( 1, (FLMBYTE *)pszEncDef)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pDb->keyRetrieve( XFLM_DICT_NAME_INDEX,
				&searchKey, XFLM_EXACT, &foundKey)))
	{
		goto Exit;
	}

	// Data part of the retrieved key has the dictionary number
	// for this encdef

	if (RC_BAD( rc = foundKey.getUINT( 3, puiEncDefId)))
	{
		if (rc == NE_XFLM_NOT_FOUND)
		{
			flmAssert( 0);
			*puiEncDefId = 0;
			rc = NE_XFLM_OK;
		}
		else
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:		Returns the name of a prefix, either Unicode or Native.
***************************************************************************/
RCODE F_Dict::getPrefix(
	FLMBOOL				bUnicode,
	FLMUINT				uiPrefixId,
	void *				pvPrefixBuf,
	FLMUINT				uiBufSize,
	FLMUINT *			puiCharsReturned)
{
	RCODE				rc = NE_XFLM_OK;
	F_PREFIX *		pPrefix;
	FLMUNICODE *	puzName;
	FLMUINT			uiOffset = 0;
	FLMUNICODE *	puzPrefixBuf = (FLMUNICODE *)pvPrefixBuf;
	char *			pszPrefixBuf = (char *)pvPrefixBuf;
	FLMUINT			uiBufChars = (bUnicode ? uiBufSize / sizeof(FLMUNICODE) : uiBufSize);

	if( RC_BAD( rc = getPrefix( uiPrefixId, &pPrefix)))
	{
		goto Exit;
	}

	if( (puzName = pPrefix->puzPrefixName) != NULL)
	{
		if( bUnicode && puzPrefixBuf)
		{
			if( !uiBufChars)
			{
				rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
				goto Exit;
			}

			while( *puzName)
			{
				if( uiBufChars == 1)
				{
					puzPrefixBuf[ uiOffset] = 0;
					rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
					goto Exit;
				}

				puzPrefixBuf[ uiOffset++] = *puzName;
				puzName++;
				uiBufChars--;
			}

			puzPrefixBuf[ uiOffset] = 0;

		}
		else if (pszPrefixBuf)
		{
			if (!uiBufChars)
			{
				rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
				goto Exit;
			}

			while ( *puzName)
			{
				if (uiBufChars == 1)
				{
					pszPrefixBuf[ uiOffset] = 0;
					rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
					goto Exit;
				}

				// Is the character convertable?
				if (*puzName > 127)
				{
					rc = RC_SET( NE_XFLM_CONV_ILLEGAL);
					goto Exit;
				}
				else
				{
					pszPrefixBuf[ uiOffset] = (char)*puzName;
					uiOffset++;
				}
				puzName++;
			}

			pszPrefixBuf[ uiOffset] = 0;
		}
		else if( puiCharsReturned)
		{
			uiOffset = f_unilen( puzName);
		}
	}

Exit:

	if( puiCharsReturned)
	{
		*puiCharsReturned = uiOffset;
	}

	return( rc);
}

/***************************************************************************
Desc:		Returns prefix object
***************************************************************************/
RCODE F_Dict::getPrefix(
	FLMUINT				uiPrefixNum,
	F_PREFIX **			ppPrefix)
{
	F_PREFIX *		pPrefix = NULL;

	if( uiPrefixNum)
	{
		if( uiPrefixNum >= m_uiLowestPrefixNum &&
			uiPrefixNum <= m_uiHighestPrefixNum)
		{
			pPrefix = m_ppPrefixTbl[ uiPrefixNum - m_uiLowestPrefixNum];
		}
	}

	if( ppPrefix)
	{
		*ppPrefix = pPrefix;
	}

	return( pPrefix ? NE_XFLM_OK : RC_SET( NE_XFLM_BAD_PREFIX));
}

/***************************************************************************
Desc:		Returns the name of an encdef, either Unicode or Native.
***************************************************************************/
RCODE F_Dict::getEncDef(
	FLMBOOL				bUnicode,
	FLMUINT				uiEncDefId,
	void *				pvEncDefBuf,
	FLMUINT				uiBufSize,
	FLMUINT *			puiCharsReturned)
{
	RCODE				rc = NE_XFLM_OK;
	F_ENCDEF *		pEncDef;
	FLMUNICODE *	puzName;
	FLMUINT			uiOffset = 0;
	FLMUNICODE *	puzEncDefBuf = (FLMUNICODE *)pvEncDefBuf;
	char *			pszEncDefBuf = (char *)pvEncDefBuf;
	FLMUINT			uiBufChars = (bUnicode ? uiBufSize / sizeof(FLMUNICODE) : uiBufSize);

	if( RC_BAD( rc = getEncDef( uiEncDefId, &pEncDef)))
	{
		goto Exit;
	}

	if( (puzName = pEncDef->puzEncDefName) != NULL)
	{
		if( bUnicode && puzEncDefBuf)
		{
			if( !uiBufChars)
			{
				rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
				goto Exit;
			}

			while( *puzName)
			{
				if( uiBufChars == 1)
				{
					puzEncDefBuf[ uiOffset] = 0;
					rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
					goto Exit;
				}

				puzEncDefBuf[ uiOffset++] = *puzName;
				puzName++;
				uiBufChars--;
			}

			puzEncDefBuf[ uiOffset] = 0;

		}
		else if (pszEncDefBuf)
		{
			if (!uiBufChars)
			{
				rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
				goto Exit;
			}

			while ( *puzName)
			{
				if (uiBufChars == 1)
				{
					pszEncDefBuf[ uiOffset] = 0;
					rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
					goto Exit;
				}

				// Is the character convertable?
				if (*puzName > 127)
				{
					rc = RC_SET( NE_XFLM_CONV_ILLEGAL);
					goto Exit;
				}
				else
				{
					pszEncDefBuf[ uiOffset] = (char)*puzName;
					uiOffset++;
				}
				puzName++;
			}

			pszEncDefBuf[ uiOffset] = 0;
		}
		else if( puiCharsReturned)
		{
			uiOffset = f_unilen( puzName);
		}
	}

Exit:

	if( puiCharsReturned)
	{
		*puiCharsReturned = uiOffset;
	}

	return( rc);
}

/***************************************************************************
Desc:		Returns encdef object
***************************************************************************/
RCODE F_Dict::getEncDef(
	FLMUINT				uiEncDefNum,
	F_ENCDEF **			ppEncDef)
{
	F_ENCDEF *		pEncDef = NULL;

	if( uiEncDefNum)
	{
		if( uiEncDefNum >= m_uiLowestEncDefNum &&
			uiEncDefNum <= m_uiHighestEncDefNum)
		{
			pEncDef = m_ppEncDefTbl[ uiEncDefNum - m_uiLowestEncDefNum];
		}
	}

	if( ppEncDef)
	{
		*ppEncDef = pEncDef;
	}

	return( pEncDef ? NE_XFLM_OK : RC_SET( NE_XFLM_BAD_ENCDEF_NUM));
}

/***************************************************************************
Desc:		Returns the root node of the specified attribute definition
***************************************************************************/
RCODE F_Dict::getDefinitionDoc(
	F_Db *				pDb,
	FLMUINT				uiTag,
	FLMUINT				uiDictId,
	F_DOMNode **		ppDoc)
{
	RCODE				rc = NE_XFLM_OK;
	F_DataVector	searchKey;
	F_DataVector	foundKey;

	// Need to lookup the document's ID, using its dictionary number
	// attribute

	if (RC_BAD( rc = searchKey.setUINT( 0, uiTag)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = searchKey.setUINT( 1, uiDictId)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pDb->keyRetrieve( XFLM_DICT_NUMBER_INDEX,
			&searchKey, XFLM_EXACT, &foundKey)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pDb->getNode( XFLM_DICT_COLLECTION,
		foundKey.getDocumentID(), ppDoc)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:		Get the IXD and LFILE information given an index number.
***************************************************************************/
RCODE F_Dict::getIndex(
	FLMUINT		uiIndexNum,
	LFILE **		ppLFile,		// [out] optional
	IXD **		ppIxd,		// [out] optional
	FLMBOOL		bOfflineOk)
{
	RCODE	rc = NE_XFLM_OK;
	IXD *	pIxd = NULL;

	if (uiIndexNum >= m_uiLowestIxNum &&
		 uiIndexNum <= m_uiHighestIxNum)
	{
		pIxd = m_ppIxdTbl [uiIndexNum - m_uiLowestIxNum];
	}
	else
	{
		switch (uiIndexNum)
		{
			case XFLM_DICT_NUMBER_INDEX:
				pIxd = m_pNumberIndex;
				break;
			case XFLM_DICT_NAME_INDEX:
				pIxd = m_pNameIndex;
				break;
		}
	}

	if (ppIxd)
	{
		*ppIxd = pIxd;
	}

	if (!pIxd)
	{
		if (ppLFile)
		{
			*ppLFile = NULL;
		}
		rc = RC_SET( NE_XFLM_BAD_IX);
		goto Exit;
	}
	else
	{
		if (ppLFile)
		{
			*ppLFile = &pIxd->lfInfo;
		}

		// If the index is suspended the IXD_OFFLINE flag
		// will be set, so it is sufficient to just test
		// the IXD_OFFLINE for both suspended and offline
		// conditions.

		if ((pIxd->uiFlags & IXD_OFFLINE) && !bOfflineOk)
		{
			rc = RC_SET( NE_XFLM_INDEX_OFFLINE);
			goto Exit;
		}

		// An encrypted index is offline if we are in limited mode.

		if (pIxd->lfInfo.uiEncId && m_bInLimitedMode && !bOfflineOk)
		{
			rc = RC_SET( NE_XFLM_INDEX_OFFLINE);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Given an index number, returns the next index after that.  These 
		should be returned in ascending numeric order ALWAYS - there are
		routines that are depending on this.
***************************************************************************/
IXD * F_Dict::getNextIndex(
	FLMUINT	uiIndexNum,
	FLMBOOL	bOkToGetPredefined)
{
	IXD *	pIxd = NULL;

	if (uiIndexNum < m_uiLowestIxNum)
	{
		uiIndexNum = m_uiLowestIxNum;
	}
	else
	{
		uiIndexNum++;
	}
	while (uiIndexNum >= m_uiLowestIxNum &&
			 uiIndexNum <= m_uiHighestIxNum)
	{
		if ((pIxd = m_ppIxdTbl [uiIndexNum - m_uiLowestIxNum]) != NULL)
		{
			goto Exit;
		}
		uiIndexNum++;
	}

	// pIxd better be NULL at this point

	flmAssert( pIxd == NULL);

	if (bOkToGetPredefined)
	{
		if (uiIndexNum <= XFLM_DICT_NUMBER_INDEX)
		{
			pIxd = m_pNumberIndex;
		}
		else if (uiIndexNum <= XFLM_DICT_NAME_INDEX)
		{
			pIxd = m_pNameIndex;
		}
	}

Exit:

	return( pIxd);
}

/***************************************************************************
Desc:	Given a collection number, returns the next collection after that.
		These should be returned in ascending numeric order ALWAYS - there
		are routines that are depending on this.
***************************************************************************/
F_COLLECTION * F_Dict::getNextCollection(
	FLMUINT	uiCollectionNum,
	FLMBOOL	bOkToGetPredefined)
{
	F_COLLECTION *	pCollection = NULL;

	if (uiCollectionNum < m_uiLowestCollectionNum)
	{
		uiCollectionNum = m_uiLowestCollectionNum;
	}
	else
	{
		uiCollectionNum++;
	}
	while (uiCollectionNum >= m_uiLowestCollectionNum &&
			 uiCollectionNum <= m_uiHighestCollectionNum)
	{
		if ((pCollection = m_ppCollectionTbl [uiCollectionNum -
												m_uiLowestCollectionNum]) != NULL)
		{
			goto Exit;	// Will return pLFile
		}
		uiCollectionNum++;
	}

	// pCollection better be NULL at this point

	flmAssert( pCollection == NULL);

	if (bOkToGetPredefined)
	{
		if( uiCollectionNum <= XFLM_MAINT_COLLECTION)
		{
			pCollection = m_pMaintCollection;
		}
		else if( uiCollectionNum <= XFLM_DATA_COLLECTION)
		{
			pCollection = m_pDataCollection;
		}
		else if (uiCollectionNum <= XFLM_DICT_COLLECTION)
		{
			pCollection = m_pDictCollection;
		}
	}

Exit:

	return( pCollection);
}

/****************************************************************************
Desc:	Link the dictionary to an F_Database object
		NOTE: This routine assumes the global mutex is locked.
****************************************************************************/
void F_Dict::linkToDatabase(
	F_Database *	pDatabase)
{
	if ((m_pNext = pDatabase->m_pDictList) != NULL)
	{
		m_uiDictSeq = m_pNext->m_uiDictSeq + 1;
		m_pNext->m_pPrev = this;
	}
	else
	{
		m_uiDictSeq = 1;
	}
	pDatabase->m_pDictList = this;
	m_pDatabase = pDatabase;
}

/****************************************************************************
Desc:	Unlink the dictionary from its F_Database object
		NOTE: This routine assumes the database mutex is locked.
****************************************************************************/
void F_Dict::unlinkFromDatabase( void)
{

	// Unlink the local dictionary from its database - if it is connected
	// to one.

	if (m_pDatabase)
	{
		if (m_pPrev)
		{
			m_pPrev->m_pNext = m_pNext;
		}
		else
		{
			m_pDatabase->m_pDictList = m_pNext;
		}
		if (m_pNext)
		{
			m_pNext->m_pPrev = m_pPrev;
		}
	}

	// Free the local dictionary and its associated tables.

	Release();
}

/****************************************************************************
Desc:	Insert an IFD into the chain of IFDs that is headed by ppFirstIfd.
		Alter ppFirstIfd if linking at head of chain.
****************************************************************************/
FSTATIC void fdictInsertInIcdChain(
	ICD **	ppFirstIcd,
	ICD *		pIcd)
{
	ICD *	pTempIcd;
	ICD *	pPrevInChain;

	if (!(*ppFirstIcd))
	{
		*ppFirstIcd = pIcd;
	}
	else
	{

		// Follow the chain and index at the front or rear depending on
		// if the attribute or element is required within the set.

		pTempIcd = *ppFirstIcd;
		if ((pIcd->uiFlags & ICD_REQUIRED_IN_SET) ||
			 !(pTempIcd->uiFlags & ICD_REQUIRED_IN_SET))
		{
			pIcd->pNextInChain = pTempIcd;
			*ppFirstIcd = pIcd;
		}
		else
		{

			// Not required in set and first ICD is required in set.
			// Look for first not required ICD in the chain.

			pPrevInChain = pTempIcd;
			pTempIcd = pTempIcd->pNextInChain;
			for (; pTempIcd; pTempIcd = pTempIcd->pNextInChain)
			{
				if (!(pTempIcd->uiFlags & ICD_REQUIRED_IN_SET))
				{
					break;
				}
				pPrevInChain = pTempIcd;
			}
			pIcd->pNextInChain = pPrevInChain->pNextInChain;
			pPrevInChain->pNextInChain = pIcd;
		}
	}
}

/****************************************************************************
Desc:	Remove an ICD from the chain of ICDs that is headed by ppFirstIcd.
		Alter ppFirstIcd if it was at the head of the chain.
****************************************************************************/
FSTATIC void fdictRemoveFromIcdChain(
	ICD **	ppFirstIcd,
	ICD *		pIcd)
{
	ICD *		pTmpIcd;
	ICD *		pPrevInChain;

	pPrevInChain = NULL;
	pTmpIcd = *ppFirstIcd;
	while (pTmpIcd != pIcd)
	{
		pPrevInChain = pTmpIcd;
		pTmpIcd = pTmpIcd->pNextInChain;
	}

	// Better have found it!

	flmAssert( pTmpIcd == pIcd);

	// Unlink from the chain

	if (!pPrevInChain)
	{
		*ppFirstIcd = pIcd->pNextInChain;
	}
	else
	{
		pPrevInChain->pNextInChain = pIcd->pNextInChain;
	}
}

/****************************************************************************
Desc:	Set the first ICD pointer for an extended element - if the element
		is currently in memory.  Should not need to lock the mutex to do this
		because the dictionary should not be in a shared state - only one
		F_Db should be pointing to it, and it should be in the middle of
		an update transaction where an index definition is being added,
		modified, or deleted.
****************************************************************************/
void F_Dict::setExtElementFirstIcd(
	FLMUINT	uiElementNum,
	ICD *		pFirstIcd)
{
	EXT_ATTR_ELM_DEF *	pExtElementDef;

	pExtElementDef = getExtElementDef( uiElementNum);
	if (pExtElementDef->uiDictNum == uiElementNum)
	{
		pExtElementDef->attrElmDef.pFirstIcd = pFirstIcd;
	}
}

/****************************************************************************
Desc:	Set the first ICD pointer for an extended attribute - if the attribute
		is currently in memory.  Should not need to lock the mutex to do this
		because the dictionary should not be in a shared state - only one
		F_Db should be pointing to it, and it should be in the middle of
		an update transaction where an index definition is being added,
		modified, or deleted.
****************************************************************************/
void F_Dict::setExtAttributeFirstIcd(
	FLMUINT	uiAttributeNum,
	ICD *		pFirstIcd)
{
	EXT_ATTR_ELM_DEF *	pExtAttributeDef;

	pExtAttributeDef = getExtAttributeDef( uiAttributeNum);
	if (pExtAttributeDef->uiDictNum == uiAttributeNum)
	{
		pExtAttributeDef->attrElmDef.pFirstIcd = pFirstIcd;
	}
}

/****************************************************************************
Desc:	Link an ICD into its ICD chain.
****************************************************************************/
RCODE F_Dict::linkIcdInChain(
	ICD *			pIcd)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiLoop;
	FLMUINT		uiInsertPos;
	IX_ITEM *	pIxItem;

	if (pIcd->uiFlags & ICD_IS_ATTRIBUTE)
	{
		ATTR_ELM_DEF *	pAttributeDef;

		if (pIcd->uiDictNum <= FLM_HIGH_FIXED_ATTRIBUTE_NUM)
		{

			// Link the ICD into its chain, making sure that required attributes
			// are first in the chain.

			pAttributeDef = getAttributeDef( pIcd->uiDictNum);

			// Attribute had better be defined at this point.  We should have already
			// verified them.

			flmAssert( pAttributeDef != NULL);

			fdictInsertInIcdChain( &pAttributeDef->pFirstIcd, pIcd);
		}
		else if (attributeIsReservedTag( pIcd->uiDictNum))
		{

			// Link the ICD into its chain, making sure that required attributes
			// are first in the chain.

			pAttributeDef = getReservedAttributeDef( pIcd->uiDictNum);

			// Attribute had better be defined at this point.  We should have already
			// verified them.

			flmAssert( pAttributeDef);

			fdictInsertInIcdChain( &pAttributeDef->pFirstIcd, pIcd);
		}
		else
		{

			// Better be one of our extended attributes

			flmAssert( pIcd->uiDictNum >= FLM_LOW_EXT_ATTRIBUTE_NUM);

			if ((pIxItem = findIxAttribute( pIcd->uiDictNum,
										&uiInsertPos)) == NULL)
			{

				// Expand the table size if necessary - by 50 elements at a time.

				if (m_uiNumIxAttributes == m_uiIxAttributeTblSize)
				{
					FLMUINT		uiNewTblSize = m_uiIxAttributeTblSize + 50;
					IX_ITEM *	pNewTbl;

					if (RC_BAD( rc = f_calloc( sizeof( IX_ITEM) * uiNewTblSize,
											&pNewTbl)))
					{
						goto Exit;
					}
					if (m_uiIxAttributeTblSize)
					{
						f_memcpy( pNewTbl, m_pIxAttributeTbl,
								sizeof( IX_ITEM) * m_uiIxAttributeTblSize);
						f_free( &m_pIxAttributeTbl);
					}
					m_pIxAttributeTbl = pNewTbl;
					m_uiIxAttributeTblSize = uiNewTblSize;
				}

				// Insert a new IX_ITEM at the specified position.

				uiLoop = m_uiNumIxAttributes;
				while (uiLoop > uiInsertPos)
				{
					f_memcpy( &m_pIxAttributeTbl [uiLoop],
								 &m_pIxAttributeTbl [uiLoop - 1], sizeof( IX_ITEM));
					uiLoop--;
				}
				pIxItem = &m_pIxAttributeTbl [uiInsertPos];
				pIxItem->uiDictNum = pIcd->uiDictNum;
				pIxItem->pFirstIcd = NULL;
				m_uiNumIxAttributes++;
			}
			fdictInsertInIcdChain( &pIxItem->pFirstIcd, pIcd);
			setExtAttributeFirstIcd( pIcd->uiDictNum, pIxItem->pFirstIcd);
		}
	}
	else if (pIcd->uiDictNum == ELM_ROOT_TAG)
	{
		fdictInsertInIcdChain( &m_pRootIcdList, pIcd);
	}
	else
	{
		ATTR_ELM_DEF *	pElementDef;

		if (pIcd->uiDictNum <= FLM_HIGH_FIXED_ELEMENT_NUM)
		{

			// Link the ICD into its chain, making sure that required elements
			// are first in the chain.

			pElementDef = getElementDef( pIcd->uiDictNum);

			// Element had better be defined at this point.  We should have already
			// verified them.

			flmAssert( pElementDef != NULL);

			fdictInsertInIcdChain( &pElementDef->pFirstIcd, pIcd);
		}
		else if (elementIsReservedTag( pIcd->uiDictNum))
		{

			// Link the ICD into its chain, making sure that required elements
			// are first in the chain.

			pElementDef = getReservedElementDef( pIcd->uiDictNum);

			// Element had better be defined at this point.  We should have already
			// verified them.

			flmAssert( pElementDef);

			fdictInsertInIcdChain( &pElementDef->pFirstIcd, pIcd);
		}
		else
		{

			// Better be one of our extended elements

			flmAssert( pIcd->uiDictNum >= FLM_LOW_EXT_ELEMENT_NUM);

			if ((pIxItem = findIxElement( pIcd->uiDictNum,
										&uiInsertPos)) == NULL)
			{

				// Expand the table size if necessary - by 50 elements at a time.

				if (m_uiNumIxElements == m_uiIxElementTblSize)
				{
					FLMUINT		uiNewTblSize = m_uiIxElementTblSize + 50;
					IX_ITEM *	pNewTbl;

					if (RC_BAD( rc = f_calloc( sizeof( IX_ITEM) * uiNewTblSize,
											&pNewTbl)))
					{
						goto Exit;
					}
					if (m_uiIxElementTblSize)
					{
						f_memcpy( pNewTbl, m_pIxElementTbl,
								sizeof( IX_ITEM) * m_uiIxElementTblSize);
						f_free( &m_pIxElementTbl);
					}
					m_pIxElementTbl = pNewTbl;
					m_uiIxElementTblSize = uiNewTblSize;
				}

				// Insert a new IX_ITEM at the specified position.

				uiLoop = m_uiNumIxElements;
				while (uiLoop > uiInsertPos)
				{
					f_memcpy( &m_pIxElementTbl [uiLoop],
								 &m_pIxElementTbl [uiLoop - 1], sizeof( IX_ITEM));
					uiLoop--;
				}
				pIxItem = &m_pIxElementTbl [uiInsertPos];
				pIxItem->uiDictNum = pIcd->uiDictNum;
				pIxItem->pFirstIcd = NULL;
				m_uiNumIxElements++;
			}
			fdictInsertInIcdChain( &pIxItem->pFirstIcd, pIcd);
			setExtElementFirstIcd( pIcd->uiDictNum, pIxItem->pFirstIcd);
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Link all of the ICDs of a tree into their proper chains from the
		IX_ITEM array or ATTR_ELM_DEF array.
***************************************************************************/
RCODE F_Dict::linkIcds(
	ICD *			pIcdTree)
{
	RCODE	rc = NE_XFLM_OK;
	ICD *	pIcd = pIcdTree;

	while (pIcd)
	{
		if (RC_BAD( rc = linkIcdInChain( pIcd)))
		{
			goto Exit;
		}
		if (pIcd->pFirstChild)
		{
			pIcd = pIcd->pFirstChild;
		}
		else
		{
			while (pIcd && !pIcd->pNextSibling)
			{
				pIcd = pIcd->pParent;
			}
			if (!pIcd)
			{
				break;
			}
			pIcd = pIcd->pNextSibling;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Unlink an ICD from the ICD chain it is in.
****************************************************************************/
void F_Dict::unlinkIcdFromChain(
	ICD *			pIcd)
{
	IX_ITEM *	pIxItem;

	if (pIcd->uiFlags & ICD_IS_ATTRIBUTE)
	{
		ATTR_ELM_DEF *	pAttributeDef;

		// Unlink the ICD from its chain - must find its previous ICD.

		if (pIcd->uiDictNum <= FLM_HIGH_FIXED_ATTRIBUTE_NUM)
		{
			pAttributeDef = getAttributeDef( pIcd->uiDictNum);

			// Attribute had better be defined at this point.

			flmAssert( pAttributeDef != NULL && pAttributeDef->pFirstIcd);

			fdictRemoveFromIcdChain( &pAttributeDef->pFirstIcd, pIcd);
		}
		else if (attributeIsReservedTag( pIcd->uiDictNum))
		{
			pAttributeDef = getReservedAttributeDef( pIcd->uiDictNum);

			// Attribute had better be defined at this point.

			flmAssert( pAttributeDef != NULL && pAttributeDef->pFirstIcd);

			fdictRemoveFromIcdChain( &pAttributeDef->pFirstIcd, pIcd);
		}
		else
		{

			// Better be one of our extended attributes

			flmAssert( pIcd->uiDictNum >= FLM_LOW_EXT_ATTRIBUTE_NUM);

			pIxItem = findIxAttribute( pIcd->uiDictNum);

			// If there is an ICD, it better be in the IX_ITEM list!

			flmAssert( pIxItem && pIxItem->pFirstIcd);

			fdictRemoveFromIcdChain( &pIxItem->pFirstIcd, pIcd);
			setExtAttributeFirstIcd( pIcd->uiDictNum, pIxItem->pFirstIcd);
		}
	}
	else if (pIcd->uiDictNum == ELM_ROOT_TAG)
	{
		fdictRemoveFromIcdChain( &m_pRootIcdList, pIcd);
	}
	else
	{
		ATTR_ELM_DEF *	pElementDef;

		// Unlink the ICD from its chain - must find its previous ICD.

		if (pIcd->uiDictNum <= FLM_HIGH_FIXED_ELEMENT_NUM)
		{
			pElementDef = getElementDef( pIcd->uiDictNum);

			// Element had better be defined at this point.

			flmAssert( pElementDef != NULL && pElementDef->pFirstIcd);

			fdictRemoveFromIcdChain( &pElementDef->pFirstIcd, pIcd);
		}
		else if (elementIsReservedTag( pIcd->uiDictNum))
		{
			pElementDef = getReservedElementDef( pIcd->uiDictNum);

			// Element had better be defined at this point.

			flmAssert( pElementDef != NULL && pElementDef->pFirstIcd);

			fdictRemoveFromIcdChain( &pElementDef->pFirstIcd, pIcd);
		}
		else
		{

			// Better be one of our extended elements

			flmAssert( pIcd->uiDictNum >= FLM_LOW_EXT_ATTRIBUTE_NUM);

			pIxItem = findIxElement( pIcd->uiDictNum);

			// If there is an ICD, it better be in the IX_ITEM list!

			flmAssert( pIxItem && pIxItem->pFirstIcd);

			fdictRemoveFromIcdChain( &pIxItem->pFirstIcd, pIcd);
			setExtElementFirstIcd( pIcd->uiDictNum, pIxItem->pFirstIcd);
		}
	}
}

/****************************************************************************
Desc:	Unlink all ICDs in an ICD tree from the chains they are in.
****************************************************************************/
void F_Dict::unlinkIcds(
	ICD *	pIcdTree)
{
	ICD *	pIcd = pIcdTree;

	while (pIcd)
	{
		unlinkIcdFromChain( pIcd);
		if (pIcd->pFirstChild)
		{
			pIcd = pIcd->pFirstChild;
		}
		else
		{
			while (pIcd && !pIcd->pNextSibling)
			{
				pIcd = pIcd->pParent;
			}
			if (!pIcd)
			{
				break;
			}
			pIcd = pIcd->pNextSibling;
		}
	}
}

/****************************************************************************
Desc:	Copies an IXD and all of its ICDs.
****************************************************************************/
RCODE F_Dict::copyIXD(
	IXD **	ppDestIxd,
	IXD *		pSrcIxd)
{
	RCODE		rc = NE_XFLM_OK;
	IXD *		pDestIxd;
	ICD *		pSrcIcd;
	ICD *		pDestIcd;
	ICD *		pLastIcd = NULL;
	ICD *		pTmpIcd;
	FLMBOOL	bLinkAsChild = FALSE;

	if (!pSrcIxd)
	{
		*ppDestIxd = NULL;
		goto Exit;
	}

	// Allocate the IXD structure

	if (RC_BAD( rc = m_dictPool.poolAlloc( sizeof( IXD), (void **)&pDestIxd)))
	{
		goto Exit;
	}

	*ppDestIxd = pDestIxd;
	f_memcpy( pDestIxd, pSrcIxd, sizeof( IXD));

	// Null out the pointers to the ICDs.

	pDestIxd->pIcdTree = NULL;
	pDestIxd->pFirstKey = NULL;
	pDestIxd->pLastKey = NULL;
	pDestIxd->pFirstData = NULL;
	pDestIxd->pLastData = NULL;
	pDestIxd->pFirstContext = NULL;
	pDestIxd->pLastContext = NULL;

	// Copy the ICDs

	pSrcIcd = pSrcIxd->pIcdTree;
	while (pSrcIcd)
	{
		if (RC_BAD( rc = m_dictPool.poolAlloc( sizeof( ICD),
											(void **)&pDestIcd)))
		{
			goto Exit;
		}
		f_memcpy( pDestIcd, pSrcIcd, sizeof( ICD));
		pDestIcd->pNextInChain = NULL;
		pDestIcd->pIxd = pDestIxd;

		// Link into key component list

		if (pDestIcd->uiKeyComponent)
		{
			pTmpIcd = pDestIxd->pFirstKey;
			while (pTmpIcd &&
				pDestIcd->uiKeyComponent > pTmpIcd->uiKeyComponent)
			{
				pTmpIcd = pTmpIcd->pNextKeyComponent;
			}

			if ((pDestIcd->pNextKeyComponent = pTmpIcd) == NULL)
			{
				// Link at end of list.

				if ((pDestIcd->pPrevKeyComponent = pDestIxd->pLastKey) != NULL)
				{
					pDestIxd->pLastKey->pNextKeyComponent = pDestIcd;
				}
				else
				{
					pDestIxd->pFirstKey = pDestIcd;
				}
				pDestIxd->pLastKey = pDestIcd;
			}
			else
			{
				// Link in front of pTmpIcd

				flmAssert( pDestIcd->uiKeyComponent < pTmpIcd->uiKeyComponent);
				if ((pDestIcd->pPrevKeyComponent =
							pTmpIcd->pPrevKeyComponent) == NULL)
				{
					pDestIxd->pFirstKey = pDestIcd;
				}
				else
				{
					pTmpIcd->pPrevKeyComponent->pNextKeyComponent = pDestIcd;
				}
				pTmpIcd->pPrevKeyComponent = pDestIcd;
			}
		}

		// Link into the data component list

		if (pDestIcd->uiDataComponent)
		{
			pTmpIcd = pDestIxd->pFirstData;
			while (pTmpIcd && pDestIcd->uiDataComponent > pTmpIcd->uiDataComponent)
			{
				pTmpIcd = pTmpIcd->pNextDataComponent;
			}
			if ((pDestIcd->pNextDataComponent = pTmpIcd) == NULL)
			{

				// Link at end of list

				if ((pDestIcd->pPrevDataComponent = pDestIxd->pLastData) != NULL)
				{
					pDestIxd->pLastData->pNextDataComponent = pDestIcd;
				}
				else
				{
					pDestIxd->pFirstData = pDestIcd;
				}
				pDestIxd->pLastData = pDestIcd;
			}
			else
			{

				// Link in front of pTmpIcd

				flmAssert( pDestIcd->uiDataComponent != pTmpIcd->uiDataComponent);
				if ((pDestIcd->pPrevDataComponent =
							pTmpIcd->pPrevDataComponent) == NULL)
				{

					// Link at very front of list

					pDestIxd->pFirstData = pDestIcd;
				}
				else
				{
					pTmpIcd->pPrevDataComponent->pNextDataComponent = pDestIcd;
				}
				pTmpIcd->pPrevDataComponent = pDestIcd;
			}
		}

		// Link into the context component list if not a key or data component

		if (!pDestIcd->uiDataComponent && !pDestIcd->uiKeyComponent)
		{
			pTmpIcd = pDestIxd->pFirstContext;
			while (pTmpIcd && pDestIcd->uiCdl > pTmpIcd->uiCdl)
			{
				pTmpIcd = pTmpIcd->pNextKeyComponent;
			}
			if ((pDestIcd->pNextKeyComponent = pTmpIcd) == NULL)
			{

				// Link at end of list

				if ((pDestIcd->pPrevKeyComponent = pDestIxd->pLastContext) != NULL)
				{
					pDestIxd->pLastContext->pNextKeyComponent = pDestIcd;
				}
				else
				{
					pDestIxd->pFirstContext = pDestIcd;
				}
				pDestIxd->pLastContext = pDestIcd;
			}
			else
			{

				// Link in front of pTmpIcd

				flmAssert( pDestIcd->uiCdl != pTmpIcd->uiCdl);
				if ((pDestIcd->pPrevKeyComponent =
							pTmpIcd->pPrevKeyComponent) == NULL)
				{

					// Link at very front of list

					pDestIxd->pFirstContext = pDestIcd;
				}
				else
				{
					pTmpIcd->pPrevKeyComponent->pNextKeyComponent = pDestIcd;
				}
				pTmpIcd->pPrevKeyComponent = pDestIcd;
			}
		}

		// Link the ICD into the destination ICD tree

		pDestIcd->pFirstChild = NULL;
		if (!pDestIxd->pIcdTree)
		{
			pDestIxd->pIcdTree = pDestIcd;
			pDestIcd->pParent = NULL;
			pDestIcd->pPrevSibling = NULL;
			pDestIcd->pNextSibling = NULL;
		}
		else if (bLinkAsChild)
		{

			// link as child

			pLastIcd->pFirstChild = pDestIcd;
			pDestIcd->pParent = pLastIcd;
			pDestIcd->pPrevSibling = NULL;
			pDestIcd->pNextSibling = NULL;
		}
		else
		{

			// link as sibling

			pLastIcd->pNextSibling = pDestIcd;
			pDestIcd->pPrevSibling = pLastIcd;
			pDestIcd->pNextSibling = NULL;
			pDestIcd->pParent = pLastIcd->pParent;
		}

		// Go to the next source ICD

		pLastIcd = pDestIcd;
		if (pSrcIcd->pFirstChild)
		{
			pSrcIcd = pSrcIcd->pFirstChild;
			bLinkAsChild = TRUE;
		}
		else
		{
			while (pSrcIcd && !pSrcIcd->pNextSibling)
			{
				pLastIcd = pLastIcd->pParent;
				pSrcIcd = pSrcIcd->pParent;
			}
			if (!pSrcIcd)
			{
				break;
			}
			bLinkAsChild = FALSE;
			pSrcIcd = pSrcIcd->pNextSibling;
		}
	}

	// Put the ICDs into their appropriate ICD chains.

	if (RC_BAD( rc = linkIcds( pDestIxd->pIcdTree)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Copies a collection
****************************************************************************/
FSTATIC RCODE fdictCopyCollection(
	F_Pool *				pDictPool,
	F_COLLECTION **	ppDestCollection,
	F_COLLECTION *		pSrcCollection)
{
	RCODE					rc = NE_XFLM_OK;

	if (!pSrcCollection)
	{
		*ppDestCollection = NULL;
	}
	else
	{
		if (RC_BAD( rc = pDictPool->poolAlloc( sizeof( F_COLLECTION),
													(void **)ppDestCollection)))
		{
			goto Exit;
		}
		f_memcpy( *ppDestCollection, pSrcCollection, sizeof( F_COLLECTION));
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Copies a prefix
****************************************************************************/
FSTATIC RCODE fdictCopyPrefix(
	F_Pool *				pDictPool,
	F_PREFIX **			ppDestPrefix,
	F_PREFIX *			pSrcPrefix)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiPrefixLen;		

	if (!pSrcPrefix)
	{
		*ppDestPrefix = NULL;
	}
	else
	{
		if (RC_BAD( rc = pDictPool->poolAlloc( sizeof( F_PREFIX),
													(void **)ppDestPrefix)))
		{
			goto Exit;
		}

		(*ppDestPrefix)->ui64PrefixId = pSrcPrefix->ui64PrefixId;
		uiPrefixLen = f_unilen( pSrcPrefix->puzPrefixName);

		if( RC_BAD( rc = pDictPool->poolAlloc( 
			sizeof( FLMUNICODE) * (uiPrefixLen + 1),
			(void **)&(*ppDestPrefix)->puzPrefixName)))
		{
			goto Exit;
		}

		f_memcpy( (*ppDestPrefix)->puzPrefixName,
					 pSrcPrefix->puzPrefixName,
					 (uiPrefixLen + 1) *	sizeof( FLMUNICODE));
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Copies an encryption def (F_ENCDEF)
****************************************************************************/
FSTATIC RCODE fdictCopyEncDef(
	F_Pool *				pDictPool,
	F_ENCDEF **			ppDestEncDef,
	F_ENCDEF *			pSrcEncDef)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiEncDefLen;

	if (!pSrcEncDef)
	{
		*ppDestEncDef = NULL;
	}
	else
	{
		if (RC_BAD( rc = pDictPool->poolAlloc( sizeof( F_ENCDEF),
													(void **)ppDestEncDef)))
		{
			goto Exit;
		}

		(*ppDestEncDef)->ui64EncDefId = pSrcEncDef->ui64EncDefId;
		(*ppDestEncDef)->ui64DocumentId = pSrcEncDef->ui64DocumentId;
		uiEncDefLen = f_unilen( pSrcEncDef->puzEncDefName);

		if( RC_BAD( rc = pDictPool->poolAlloc( 
			sizeof( FLMUNICODE) * (uiEncDefLen + 1),
			(void **)&(*ppDestEncDef)->puzEncDefName)))
		{
			goto Exit;
		}

		f_memcpy( (*ppDestEncDef)->puzEncDefName,
					 pSrcEncDef->puzEncDefName,
					 (uiEncDefLen + 1) *	sizeof( FLMUNICODE));
	
		(*ppDestEncDef)->pCcs = pSrcEncDef->pCcs;
		(*ppDestEncDef)->pCcs->AddRef();
		(*ppDestEncDef)->uiEncKeySize = pSrcEncDef->uiEncKeySize;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Clones a dictionary from another one.
****************************************************************************/
RCODE F_Dict::cloneDict(
	F_Dict *	pSrcDict)
{
	RCODE		rc = NE_XFLM_OK;
	FLMUINT	uiCount;
	FLMUINT	uiLoop;
	FLMBOOL	bElementMutexLocked = FALSE;
	FLMBOOL	bAttributeMutexLocked = FALSE;

	resetDict();

	// Copy the element definitions.

	m_uiLowestElementNum = pSrcDict->m_uiLowestElementNum;
	m_uiHighestElementNum = pSrcDict->m_uiHighestElementNum;
	if (m_uiHighestElementNum)
	{
		uiCount = m_uiHighestElementNum - m_uiLowestElementNum + 1;
		if (RC_BAD( rc = f_alloc( uiCount * sizeof( ATTR_ELM_DEF),
					&m_pElementDefTbl)))
		{
			goto Exit;
		}
		f_memcpy( m_pElementDefTbl, pSrcDict->m_pElementDefTbl,
			uiCount * sizeof( ATTR_ELM_DEF));

		// NULL out every element's pFirstIcd pointer.  These will
		// be fixed up as indexes are added.

		for (uiLoop = 0; uiLoop < uiCount; uiLoop++)
		{
			m_pElementDefTbl[ uiLoop].pFirstIcd = NULL;
		}
	}

	// Copy the reserved element definitions.

	uiCount = XFLM_LAST_RESERVED_ELEMENT_TAG -
				 XFLM_FIRST_RESERVED_ELEMENT_TAG + 1;
	if (RC_BAD( rc = f_alloc( uiCount * sizeof( ATTR_ELM_DEF),
				&m_pReservedElementDefTbl)))
	{
		goto Exit;
	}
	f_memcpy( m_pReservedElementDefTbl, pSrcDict->m_pReservedElementDefTbl,
		uiCount * sizeof( ATTR_ELM_DEF));

	// NULL out every element's pFirstIcd pointer.  These will
	// be fixed up as indexes are added.

	for (uiLoop = 0; uiLoop < uiCount; uiLoop++)
	{
		m_pReservedElementDefTbl [uiLoop].pFirstIcd = NULL;
	}

	// Copy the extended element definitions, if any

	if (pSrcDict->m_pExtElementDefTbl)
	{

		// Source table's mutex must be locked while copying

		f_mutexLock( pSrcDict->m_hExtElementDefMutex);
		bElementMutexLocked = TRUE;

		m_uiExtElementDefTblSize = pSrcDict->m_uiExtElementDefTblSize;

		// Allocate a mutex

		if (RC_BAD( rc = f_mutexCreate( &m_hExtElementDefMutex)))
		{
			goto Exit;
		}

		// Allocate the array

		if (RC_BAD( rc = f_alloc( m_uiExtElementDefTblSize *
					sizeof( EXT_ATTR_ELM_DEF), &m_pExtElementDefTbl)))
		{
			goto Exit;
		}
		f_memcpy( m_pExtElementDefTbl, pSrcDict->m_pExtElementDefTbl,
			m_uiExtElementDefTblSize * sizeof( EXT_ATTR_ELM_DEF));

		f_mutexUnlock( pSrcDict->m_hExtElementDefMutex);
		bElementMutexLocked = FALSE;

		// NULL out every element's pFirstIcd pointer.  These will
		// be fixed up as indexes are added.

		for (uiLoop = 0; uiLoop < m_uiExtElementDefTblSize; uiLoop++)
		{
			m_pExtElementDefTbl [uiLoop].attrElmDef.pFirstIcd = NULL;
		}
	}

	// Copy the extended element IX_ITEM structures for indexing.

	if (pSrcDict->m_pIxElementTbl)
	{
		m_uiIxElementTblSize = pSrcDict->m_uiIxElementTblSize;
		m_uiNumIxElements = pSrcDict->m_uiNumIxElements;
		if (RC_BAD( rc = f_alloc( sizeof( IX_ITEM) * m_uiIxElementTblSize,
								&m_pIxElementTbl)))
		{
			goto Exit;
		}
		f_memcpy( m_pIxElementTbl, pSrcDict->m_pIxElementTbl,
			sizeof( IX_ITEM) * m_uiIxElementTblSize);

		// NULL out every element's pFirstIcd pointer.  These will
		// be fixed up as indexes are added.

		for (uiLoop = 0; uiLoop < m_uiNumIxElements; uiLoop++)
		{
			m_pIxElementTbl [uiLoop].pFirstIcd = NULL;
		}
	}

	// Copy the attribute definitions.

	m_uiLowestAttributeNum = pSrcDict->m_uiLowestAttributeNum;
	m_uiHighestAttributeNum = pSrcDict->m_uiHighestAttributeNum;
	if (m_uiHighestAttributeNum)
	{
		uiCount = m_uiHighestAttributeNum - m_uiLowestAttributeNum + 1;
		if (RC_BAD( rc = f_alloc( uiCount * sizeof( ATTR_ELM_DEF),
					&m_pAttributeDefTbl)))
		{
			goto Exit;
		}
		f_memcpy( m_pAttributeDefTbl, pSrcDict->m_pAttributeDefTbl,
			uiCount * sizeof( ATTR_ELM_DEF));

		// NULL out every attribute's pFirstIcd pointer.  These will
		// be fixed up as indexes are added.

		for (uiLoop = 0; uiLoop < uiCount; uiLoop++)
		{
			m_pAttributeDefTbl[ uiLoop].pFirstIcd = NULL;
		}
	}

	// Copy the reserved attribute definitions.

	uiCount = XFLM_LAST_RESERVED_ATTRIBUTE_TAG -
				 XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 1;
	if (RC_BAD( rc = f_alloc( uiCount * sizeof( ATTR_ELM_DEF),
				&m_pReservedAttributeDefTbl)))
	{
		goto Exit;
	}
	f_memcpy( m_pReservedAttributeDefTbl, pSrcDict->m_pReservedAttributeDefTbl,
		uiCount * sizeof( ATTR_ELM_DEF));

	// NULL out every attribute's pFirstIcd pointer.  These will
	// be fixed up as indexes are added.

	for (uiLoop = 0; uiLoop < uiCount; uiLoop++)
	{
		m_pReservedAttributeDefTbl [uiLoop].pFirstIcd = NULL;
	}

	// Copy the extended attribute definitions, if any

	if (pSrcDict->m_pExtAttributeDefTbl)
	{

		// Source table's mutex must be locked while copying

		f_mutexLock( pSrcDict->m_hExtAttributeDefMutex);
		bAttributeMutexLocked = TRUE;

		m_uiExtAttributeDefTblSize = pSrcDict->m_uiExtAttributeDefTblSize;

		// Allocate a mutex

		if (RC_BAD( rc = f_mutexCreate( &m_hExtAttributeDefMutex)))
		{
			goto Exit;
		}

		// Allocate the array

		if (RC_BAD( rc = f_alloc( m_uiExtAttributeDefTblSize *
					sizeof( EXT_ATTR_ELM_DEF),
					&m_pExtAttributeDefTbl)))
		{
			goto Exit;
		}
		f_memcpy( m_pExtAttributeDefTbl, pSrcDict->m_pExtAttributeDefTbl,
			m_uiExtAttributeDefTblSize * sizeof( EXT_ATTR_ELM_DEF));

		f_mutexUnlock( pSrcDict->m_hExtAttributeDefMutex);
		bAttributeMutexLocked = FALSE;

		// NULL out every attribute's pFirstIcd pointer.  These will
		// be fixed up as indexes are added.

		for (uiLoop = 0; uiLoop < m_uiExtAttributeDefTblSize; uiLoop++)
		{
			m_pExtAttributeDefTbl[ uiLoop].attrElmDef.pFirstIcd = NULL;
		}
	}

	// Copy the extended attribute IX_ITEM structures for indexing.

	if (pSrcDict->m_pIxAttributeTbl)
	{
		m_uiIxAttributeTblSize = pSrcDict->m_uiIxAttributeTblSize;
		m_uiNumIxAttributes = pSrcDict->m_uiNumIxAttributes;
		if (RC_BAD( rc = f_alloc( sizeof( IX_ITEM) * m_uiIxAttributeTblSize,
								&m_pIxAttributeTbl)))
		{
			goto Exit;
		}
		f_memcpy( m_pIxAttributeTbl, pSrcDict->m_pIxAttributeTbl,
			sizeof( IX_ITEM) * m_uiIxAttributeTblSize);

		// NULL out every attribute's pFirstIcd pointer.  These will
		// be fixed up as indexes are added.

		for (uiLoop = 0; uiLoop < m_uiNumIxAttributes; uiLoop++)
		{
			m_pIxAttributeTbl [uiLoop].pFirstIcd = NULL;
		}
	}

	// Copy the pre-defined collections

	if (RC_BAD( rc = fdictCopyCollection( &m_dictPool,
								&m_pDictCollection,
								pSrcDict->m_pDictCollection)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = fdictCopyCollection( &m_dictPool,
								&m_pDataCollection,
								pSrcDict->m_pDataCollection)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = fdictCopyCollection( &m_dictPool,
								&m_pMaintCollection,
								pSrcDict->m_pMaintCollection)))
	{
		goto Exit;
	}

	// Collection Table

	if ((uiCount = pSrcDict->getCollectionCount( FALSE)) > 0)
	{
		m_uiLowestCollectionNum = pSrcDict->m_uiLowestCollectionNum;
		m_uiHighestCollectionNum = pSrcDict->m_uiHighestCollectionNum;

		if (RC_BAD( rc = f_alloc( uiCount * sizeof( F_COLLECTION *),
					&m_ppCollectionTbl)))
		{
			goto Exit;
		}

		// Copy each F_COLLECTION in the table.

		for (uiLoop = 0; uiLoop < uiCount; uiLoop++)
		{
			if (RC_BAD( rc = fdictCopyCollection( &m_dictPool,
										&m_ppCollectionTbl [uiLoop],
										pSrcDict->m_ppCollectionTbl [uiLoop])))
			{
				goto Exit;
			}
		}
	}

	// Copy the user-defined prefixes

	if ((uiCount = pSrcDict->getPrefixCount()) > 0)
	{
		m_uiLowestPrefixNum = pSrcDict->m_uiLowestPrefixNum;
		m_uiHighestPrefixNum = pSrcDict->m_uiHighestPrefixNum;

		if (RC_BAD( rc = f_alloc( uiCount * sizeof( F_PREFIX *),
					&m_ppPrefixTbl)))
		{
			goto Exit;
		}

		// Copy each F_PREFIX in the table.

		for (uiLoop = 0; uiLoop < uiCount; uiLoop++)
		{
			if (RC_BAD( rc = fdictCopyPrefix( &m_dictPool,
										&m_ppPrefixTbl [uiLoop],
										pSrcDict->m_ppPrefixTbl [uiLoop])))
			{
				goto Exit;
			}
		}
	}

	// Copy the encryption definitions

	if ((uiCount = pSrcDict->getEncDefCount()) > 0)
	{
		m_uiLowestEncDefNum = pSrcDict->m_uiLowestEncDefNum;
		m_uiHighestEncDefNum = pSrcDict->m_uiHighestEncDefNum;

		if (RC_BAD( rc = f_alloc( uiCount * sizeof( F_ENCDEF *),
					&m_ppEncDefTbl)))
		{
			goto Exit;
		}

		// Copy each F_ENCDEF in the table.

		for (uiLoop = 0; uiLoop < uiCount; uiLoop++)
		{
			if (RC_BAD( rc = fdictCopyEncDef( &m_dictPool,
										&m_ppEncDefTbl [uiLoop],
										pSrcDict->m_ppEncDefTbl [uiLoop])))
			{
				goto Exit;
			}
		}
	}

	// Copy the hard-coded indexes

	if (RC_BAD( rc = copyIXD( &m_pNameIndex, pSrcDict->m_pNameIndex)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = copyIXD( &m_pNumberIndex, pSrcDict->m_pNumberIndex)))
	{
		goto Exit;
	}

	// IXD Table

	m_uiLowestIxNum = pSrcDict->m_uiLowestIxNum;
	m_uiHighestIxNum = pSrcDict->m_uiHighestIxNum;

	// Array of IXD pointers - for user defined indexes only

	if ((uiCount = pSrcDict->getIndexCount( FALSE)) > 0)
	{
		if (RC_BAD( rc = f_alloc( uiCount * sizeof( IXD *),
					&m_ppIxdTbl)))
		{
			goto Exit;
		}

		// Copy each IXD in the table.

		for (uiLoop = 0; uiLoop < uiCount; uiLoop++)
		{
			if (RC_BAD( rc = copyIXD( &m_ppIxdTbl [uiLoop],
										pSrcDict->m_ppIxdTbl [uiLoop])))
			{
				goto Exit;
			}
		}
	}

	// Copy the name table

	if (RC_BAD( rc = allocNameTable()))
	{
		goto Exit;
	}

	if (RC_BAD( rc = m_pNameTable->cloneNameTable( pSrcDict->m_pNameTable)))
	{
		goto Exit;
	}

	// See if we can get the limited mode flag from the database file instead
	// of copying it from the source dictionary.

	if (pSrcDict->m_pDatabase)
	{
		m_bInLimitedMode = pSrcDict->m_pDatabase->inLimitedMode();
	}
	else
	{
		m_bInLimitedMode = pSrcDict->m_bInLimitedMode;
	}

Exit:

	if (bElementMutexLocked)
	{
		f_mutexUnlock( pSrcDict->m_hExtElementDefMutex);
	}

	if (bAttributeMutexLocked)
	{
		f_mutexUnlock( pSrcDict->m_hExtAttributeDefMutex);
	}

	return( rc);
}

/***************************************************************************
Desc:	This routine checks to see if an element is referenced in an index
		definition.
***************************************************************************/
RCODE F_Dict::checkElementReferences(
	FLMUINT		uiElementNum
	)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiIxdCount;
	FLMUINT		uiIxdLoop;
	IXD *			pIxd;
	ICD *			pIcd;

	// Look through all ICDs in all IXDs.

	uiIxdCount = getIndexCount( FALSE);
	for (uiIxdLoop = 0; uiIxdLoop < uiIxdCount; uiIxdLoop++)
	{
		if ((pIxd = m_ppIxdTbl [uiIxdLoop]) != NULL)
		{

			// Loop through all of the index's ICDs.

			pIcd = pIxd->pIcdTree;
			while (pIcd)
			{
				if (!(pIcd->uiFlags & ICD_IS_ATTRIBUTE) &&
					 pIcd->uiDictNum == uiElementNum)
				{
					rc = RC_SET( NE_XFLM_CANNOT_DEL_ELEMENT);
					goto Exit;
				}
				if (pIcd->pFirstChild)
				{
					pIcd = pIcd->pFirstChild;
				}
				else
				{
					while (pIcd && !pIcd->pNextSibling)
					{
						pIcd = pIcd->pParent;
					}
					if (!pIcd)
					{
						break;
					}
					pIcd = pIcd->pNextSibling;
				}
			}
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	This routine checks to see if an attribute is referenced in an index
		definition.
***************************************************************************/
RCODE F_Dict::checkAttributeReferences(
	FLMUINT		uiAttributeNum
	)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiIxdCount;
	FLMUINT		uiIxdLoop;
	IXD *			pIxd;
	ICD *			pIcd;

	// Look through all ICDs in all IXDs.

	uiIxdCount = getIndexCount( FALSE);
	for (uiIxdLoop = 0; uiIxdLoop < uiIxdCount; uiIxdLoop++)
	{
		if ((pIxd = m_ppIxdTbl [uiIxdLoop]) != NULL)
		{

			// Loop through all of the index's ICDs.

			pIcd = pIxd->pIcdTree;
			while (pIcd)
			{
				if ((pIcd->uiFlags & ICD_IS_ATTRIBUTE) &&
					 pIcd->uiDictNum == uiAttributeNum)
				{
					rc = RC_SET( NE_XFLM_CANNOT_DEL_ATTRIBUTE);
					goto Exit;
				}
				if (pIcd->pFirstChild)
				{
					pIcd = pIcd->pFirstChild;
				}
				else
				{
					while (pIcd && !pIcd->pNextSibling)
					{
						pIcd = pIcd->pParent;
					}
					if (!pIcd)
					{
						break;
					}
					pIcd = pIcd->pNextSibling;
				}
			}
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	This routine checks to see if a collection is referenced in an index
		definition.
***************************************************************************/
RCODE F_Dict::checkCollectionReferences(
	FLMUINT		uiCollectionNum
	)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiIxdCount;
	FLMUINT		uiIxdLoop;
	IXD *			pIxd;

	// Look through all IXDs

	uiIxdCount = getIndexCount( FALSE);
	for (uiIxdLoop = 0; uiIxdLoop < uiIxdCount; uiIxdLoop++)
	{
		if ((pIxd = m_ppIxdTbl [uiIxdLoop]) != NULL)
		{
			if (pIxd->uiCollectionNum == uiCollectionNum)
			{
				rc = RC_SET( NE_XFLM_MUST_DELETE_INDEXES);
				goto Exit;
			}
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Reallocate a field, index, or collection table.  Note that any new
		slots in the table will be zeroed out.
***************************************************************************/
RCODE F_Dict::reallocTbl(
	FLMUINT		uiNewId,
	FLMUINT		uiElementSize,
	void **		ppvTbl,
	FLMUINT *	puiLowest,
	FLMUINT *	puiHighest,
	FLMUINT		uiAdjustFactor,
	FLMUINT		uiMaxId
	)
{
	RCODE		rc = NE_XFLM_OK;
	void *	pvNewTbl;
	FLMUINT	uiLowest = *puiLowest;
	FLMUINT	uiOldLowest = uiLowest;
	FLMUINT	uiHighest = *puiHighest;
	FLMUINT	uiOldCount;
	FLMUINT	uiNewCount;

	if (!uiHighest)
	{
		uiOldCount = 0;
		if (uiNewId <= uiAdjustFactor)
		{
			uiLowest = 1;
		}
		else
		{
			uiLowest = uiNewId - uiAdjustFactor;
		}
		if (uiNewId >= uiMaxId - uiAdjustFactor)
		{
			uiHighest = uiMaxId;
		}
		else
		{
			uiHighest = uiNewId + uiAdjustFactor;
		}
	}
	else
	{
		uiOldCount = uiHighest - uiLowest + 1;
		if (uiNewId < uiLowest)
		{
			if (uiNewId <= uiAdjustFactor)
			{
				uiLowest = 1;
			}
			else
			{
				uiLowest = uiNewId - uiAdjustFactor;
			}
		}
		else
		{
			if (uiNewId >= uiMaxId - uiAdjustFactor)
			{
				uiHighest = uiMaxId;
			}
			else
			{
				uiHighest = uiNewId + uiAdjustFactor;
			}
		}
	}

	// Reallocate the table.

	uiNewCount = uiHighest - uiLowest + 1;
	if (RC_BAD( rc = f_calloc( uiNewCount * uiElementSize,
				&pvNewTbl)))
	{
		goto Exit;
	}
	if (uiOldCount)
	{
		f_memcpy( (FLMBYTE *)pvNewTbl +
					uiElementSize * (uiOldLowest - uiLowest),
					*ppvTbl, uiOldCount * uiElementSize);
	}
	f_free( ppvTbl);
	*ppvTbl = pvNewTbl;
	*puiLowest = uiLowest;
	*puiHighest = uiHighest;

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Maps a string to an element or attribute data type.
***************************************************************************/
RCODE fdictGetDataType(
	char *		pszDataType,
	FLMUINT *	puiDataType)
{
	RCODE		rc = NE_XFLM_OK;
	FLMUINT	uiDataType;

	// Parse the type keyword - only one type allowed.

	for (uiDataType = 0; uiDataType < XFLM_NUM_OF_TYPES; uiDataType++)
	{
		if (f_stricmp( pszDataType, fdictDataTypes [uiDataType]) == 0)
		{
			*puiDataType = uiDataType;
			goto Exit;					// Will return NE_XFLM_OK
		}
	}

	rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_DATA_TYPE);

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Maps a string to an element or attribute data type.
***************************************************************************/
const char * fdictGetDataTypeStr(
	FLMUINT 		uiDataType
	)
{
	if( uiDataType > XFLM_NUM_OF_TYPES)
	{
		return( NULL);
	}

	return( fdictDataTypes[ uiDataType]);
}

/***************************************************************************
Desc:	Maps a string to an element or attribute state.
***************************************************************************/
RCODE fdictGetState(
	const char *	pszState,
	FLMUINT *		puiState)
{
	RCODE	rc = NE_XFLM_OK;

	if (f_stricmp( pszState, XFLM_CHECKING_OPTION_STR) == 0)
	{
		*puiState = ATTR_ELM_STATE_CHECKING;
	}
	else if (f_stricmp( pszState, XFLM_PURGE_OPTION_STR) == 0)
	{
		*puiState = ATTR_ELM_STATE_PURGE;
	}
	else if (f_stricmp( pszState, XFLM_ACTIVE_OPTION_STR) == 0)
	{
		*puiState = ATTR_ELM_STATE_ACTIVE;
	}
	else
	{
		rc = RC_SET( NE_XFLM_ILLEGAL_STATE);
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Maps a string to an index state.
***************************************************************************/
RCODE fdictGetIndexState(
	const char *	pszState,
	FLMUINT *		puiState)
{
	RCODE	rc = NE_XFLM_OK;

	if (f_stricmp( pszState, XFLM_INDEX_SUSPENDED_STR) == 0)
	{
		*puiState = IXD_SUSPENDED | IXD_OFFLINE;
	}
	else if (f_stricmp( pszState, XFLM_INDEX_OFFLINE_STR) == 0)
	{
		*puiState = IXD_OFFLINE;
	}
	else if (!pszState [0] ||
				f_stricmp( pszState, XFLM_INDEX_ONLINE_STR) == 0)
	{
		*puiState = 0;
	}
	else
	{
		rc = RC_SET( NE_XFLM_ILLEGAL_STATE);
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Determine if an element name is legal - formatwise
***************************************************************************/
FINLINE RCODE fdictLegalElementName(
	FLMUNICODE *	// puzElementName
	)
{
	//VISIT: Need to fill this out - could return NE_XFLM_ILLEGAL_ELEMENT_NAME
	return( NE_XFLM_OK);
}

/***************************************************************************
Desc:	Determine if an attribute name is legal - formatwise
***************************************************************************/
FINLINE RCODE fdictLegalAttributeName(
	FLMUNICODE *	puzAttributeName
	)
{
	if (*puzAttributeName == '/' && *(puzAttributeName + 1) == 0)
	{
		return( RC_SET( NE_XFLM_ILLEGAL_ATTRIBUTE_NAME));
	}
	//VISIT: Need to fill this out - could return NE_XFLM_ILLEGAL_ATTRIBUTE_NAME
	return( NE_XFLM_OK);
}

/***************************************************************************
Desc:	Determine if an element number is legal
***************************************************************************/
FINLINE RCODE fdictLegalElementNumber(
	FLMUINT	uiElementNumber,
	FLMBOOL	bAllowReserved)
{
	return( (uiElementNumber > 0 && uiElementNumber <= XFLM_MAX_ELEMENT_NUM) ||
			  (bAllowReserved &&
			   uiElementNumber >= XFLM_FIRST_RESERVED_ELEMENT_TAG &&
			   uiElementNumber <= XFLM_LAST_RESERVED_ELEMENT_TAG)
				? NE_XFLM_OK
				: RC_SET( NE_XFLM_ILLEGAL_ELEMENT_NUMBER));
}

/***************************************************************************
Desc:	Determine if an attribute number is legal
***************************************************************************/
FINLINE RCODE fdictLegalAttributeNumber(
	FLMUINT	uiAttributeNumber,
	FLMBOOL	bAllowReserved)
{
	return( (uiAttributeNumber > 0 &&
				uiAttributeNumber <= XFLM_MAX_ATTRIBUTE_NUM) ||
			  (bAllowReserved &&
			   uiAttributeNumber >= XFLM_FIRST_RESERVED_ATTRIBUTE_TAG &&
			   uiAttributeNumber <= XFLM_LAST_RESERVED_ATTRIBUTE_TAG)
				? NE_XFLM_OK
				: RC_SET( NE_XFLM_ILLEGAL_ATTRIBUTE_NUMBER));
}

/***************************************************************************
Desc:	Retrieve a dictionary.  If there is no dictionary, open it first.
***************************************************************************/
RCODE F_Db::getDictionary(
	F_Dict ** ppDict)
{
	RCODE		rc = NE_XFLM_OK;

	if (!m_pDict)
	{
		if (RC_BAD( rc = dictOpen()))
		{
			goto Exit;
		}
	}
	*ppDict = m_pDict;
Exit:

	return rc;
}

/***************************************************************************
Desc:	Change an element's or attribute's state.
***************************************************************************/
RCODE F_Db::changeItemState(
	FLMUINT				uiDictType,
	FLMUINT				uiDictNum,
	const char *		pszState)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiNewState;
	FLMBOOL				bStartedTrans = FALSE;
	F_DataVector		srchKey;
	F_DataVector		foundKey;
	F_DOMNode *			pNode = NULL;
	F_DOMNode *			pAttr = NULL;
	F_DOMNode *			pChangeCountAttr = NULL;
	F_DOMNode *			pTrackerDoc = NULL;
	FLMUINT64			ui64DocumentId;
	FLMUINT64			ui64Count;
	FLMBOOL				bMustAbortOnError = FALSE;
	F_AttrElmInfo		defInfo;

	if (uiDictType != ELM_ELEMENT_TAG && uiDictType != ELM_ATTRIBUTE_TAG)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_INVALID_PARM);
		goto Exit;
	}

	if (RC_BAD( rc = fdictGetState( pszState, &uiNewState)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = checkState( __FILE__, __LINE__)))
	{
		goto Exit;
	}

	if (m_eTransType != XFLM_NO_TRANS)
	{
		if (!okToCommitTrans())
		{
			rc = RC_SET( NE_XFLM_ABORT_TRANS);
			goto Exit;
		}
		else if (m_eTransType == XFLM_READ_TRANS)
		{
			rc = RC_SET( NE_XFLM_ILLEGAL_TRANS_OP);
			goto Exit;
		}
	}
	else
	{
		// Need to have an update transaction going.

		if (RC_BAD( rc = beginTrans( XFLM_UPDATE_TRANS)))
		{
			goto Exit;
		}
		bStartedTrans = TRUE;
	}

	// Commit any keys in the KREF buffers.

	if (RC_BAD( rc = keysCommit( FALSE)))
	{
		goto Exit;
	}

	// See if the element or attribute is defined, and get its
	// current state

	if (uiDictType == ELM_ELEMENT_TAG)
	{
		if (RC_BAD( rc = m_pDict->getElement( this, uiDictNum, &defInfo)))
		{
			goto Exit;
		}

		// Nothing to change if the state does not change.

		if (uiNewState == defInfo.m_uiState)
		{
			goto Exit;
		}

		if (elementIsReservedTag( uiDictNum))
		{
			rc = RC_SET( NE_XFLM_CANNOT_MOD_ELEMENT_STATE);
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = m_pDict->getAttribute( this, uiDictNum, &defInfo)))
		{
			goto Exit;
		}

		// Nothing to change if the state does not change.

		if (uiNewState == defInfo.m_uiState)
		{
			goto Exit;
		}

		if (attributeIsReservedTag( uiDictNum))
		{
			rc = RC_SET( NE_XFLM_CANNOT_MOD_ATTRIBUTE_STATE);
			goto Exit;
		}
	}

	// Check the legal state changes that can be made by external callers
	// of this routine

	if (!m_bItemStateUpdOk)
	{
		switch (defInfo.m_uiState)
		{
			case ATTR_ELM_STATE_ACTIVE:
			{
				if( uiNewState != ATTR_ELM_STATE_CHECKING &&
					uiNewState != ATTR_ELM_STATE_PURGE)
				{
					rc = RC_SET( NE_XFLM_ILLEGAL_STATE_CHANGE);
					goto Exit;
				}
				
				// Add a task to the tracker container so the maintenance thread
				// will sweep the database
				
				if( !(m_uiFlags & FDB_SWEEP_SCHEDULED))
				{
					if( RC_BAD( rc = createRootNode( XFLM_MAINT_COLLECTION, 
						ELM_SWEEP_TAG, ELEMENT_NODE, &pTrackerDoc)))
					{
						goto Exit;
					}
					
					if( RC_BAD( rc = pTrackerDoc->setAttributeValueUINT64( this,
						ATTR_TRANSACTION_TAG, m_ui64CurrTransID)))
					{
						goto Exit;
					}
			
					if( RC_BAD( rc = pTrackerDoc->addModeFlags( 
						this, FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
					{
						goto Exit;
					}
			
					if( RC_BAD( rc = documentDone( pTrackerDoc))) 
					{
						goto Exit;
					}
					
					f_semSignal( m_pDatabase->m_hMaintSem);
					m_uiFlags |= FDB_SWEEP_SCHEDULED;
				}
				
				break;
			}
			
			case ATTR_ELM_STATE_CHECKING:
			case ATTR_ELM_STATE_PURGE:
			{
				if (uiNewState != ATTR_ELM_STATE_ACTIVE)
				{
					rc = RC_SET( NE_XFLM_ILLEGAL_STATE_CHANGE);
					goto Exit;
				}
				break;
			}
			
			default:
			{
				// Only other old state is ATTR_ELM_STATE_UNUSED, and it
				// can be changed to any other state.

				break;
			}
		}
	}

	bMustAbortOnError = TRUE;

	// To do a state change, must create a new dictionary

	if( !(m_uiFlags & FDB_UPDATED_DICTIONARY))
	{
		if( RC_BAD( rc = dictClone()))
		{
			goto Exit;
		}
	}

	// Find the element's document

	if( RC_BAD( rc = srchKey.setUINT( 0, uiDictType)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = srchKey.setUINT( 1, uiDictNum)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = keyRetrieve( XFLM_DICT_NUMBER_INDEX,
				&srchKey, XFLM_EXACT, &foundKey)))
	{
		if( rc == NE_XFLM_NOT_FOUND)
		{
			// We should have found the thing!  It should have
			// already been indexed!

			rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
		}
		
		goto Exit;
	}

	// Read the root node of the document

	ui64DocumentId = foundKey.getDocumentID();
	if (RC_BAD( rc = getNode( XFLM_DICT_COLLECTION, ui64DocumentId, &pNode)))
	{
		if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
		{

			// We should have found the thing!  It should have
			// already been indexed!

			rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
		}
		goto Exit;
	}

	// Find the state attribute

	if (RC_BAD( rc = pNode->getAttribute( this, ATTR_STATE_TAG,
									(IF_DOMNode **)&pAttr)))
	{
		if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			goto Exit;
		}

		rc = NE_XFLM_OK;

		// Create the attribute since it was not found

		if (RC_BAD( rc = pNode->createAttribute( this, ATTR_STATE_TAG,
											(IF_DOMNode **)&pAttr)))
		{
			goto Exit;
		}
	}
	else
	{
		// Unfreeze the attribute state

		if (RC_BAD( rc = pAttr->removeModeFlags( 
			this, FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
		{
			goto Exit;
		}
	}

	// Set the attribute state

	if (RC_BAD( rc = pAttr->setUTF8( this, (FLMBYTE *)pszState)))
	{
		goto Exit;
	}

	// Freeze the attribute state

	if( RC_BAD( rc = pAttr->addModeFlags( this,
		FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
	{
		goto Exit;
	}

	// Find the state change count attribute

	if (RC_BAD( rc = pNode->getAttribute( this, ATTR_STATE_CHANGE_COUNT_TAG,
									(IF_DOMNode **)&pChangeCountAttr)))
	{
		if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			goto Exit;
		}

		rc = NE_XFLM_OK;

		// Create the attribute since it was not found

		if (RC_BAD( rc = pNode->createAttribute( this,
											ATTR_STATE_CHANGE_COUNT_TAG,
											(IF_DOMNode **)&pChangeCountAttr)))
		{
			goto Exit;
		}
		ui64Count = 0;
	}
	else
	{
		// Unfreeze the attribute state

		if (RC_BAD( rc = pChangeCountAttr->removeModeFlags( 
			this, FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = pChangeCountAttr->getUINT64( this, &ui64Count)))
		{
			goto Exit;
		}
	}

	// Increment the state change count.

	if (RC_BAD( rc = pChangeCountAttr->setUINT64( this, ui64Count + 1)))
	{
		goto Exit;
	}

	// Freeze the attribute state change count.

	if( RC_BAD( rc = pChangeCountAttr->addModeFlags( this,
		FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
	{
		goto Exit;
	}

	// Set the state in the dictionary

	if (uiDictType == ELM_ELEMENT_TAG)
	{
		if (uiDictNum <= FLM_HIGH_FIXED_ELEMENT_NUM)
		{
			ATTR_ELM_DEF *	pElementDef;

			if ((pElementDef = m_pDict->getElementDef( uiDictNum)) != NULL)
			{
				attrElmSetState( pElementDef, uiNewState);
			}
		}
		else
		{
			EXT_ATTR_ELM_DEF *	pExtElementDef;

			// Better be one of our extended elements - cannot change
			// state on reserved elements.

			flmAssert( elementIsUserDefined( uiDictNum));

			// No need to lock the mutex, because no other threads can
			// be accessing it right now.

			if (m_pDict->m_pExtElementDefTbl)
			{
				pExtElementDef = m_pDict->getExtElementDef( uiDictNum);
				if (pExtElementDef->uiDictNum == uiDictNum)
				{
					attrElmSetState( &pExtElementDef->attrElmDef, uiNewState);
				}
			}
		}
	}
	else
	{
		if (uiDictNum <= FLM_HIGH_FIXED_ATTRIBUTE_NUM)
		{
			ATTR_ELM_DEF *	pAttributeDef;

			if ((pAttributeDef = m_pDict->getAttributeDef( uiDictNum)) != NULL)
			{
				attrElmSetState( pAttributeDef, uiNewState);
			}
		}
		else
		{
			EXT_ATTR_ELM_DEF *	pExtAttributeDef;

			// Better be one of our extended attributes - cannot change
			// state on reserved attributes.

			flmAssert( attributeIsUserDefined( uiDictNum));

			// No need to lock the mutex, because no other threads can
			// be accessing it right now.

			if (m_pDict->m_pExtAttributeDefTbl)
			{
				pExtAttributeDef = m_pDict->getExtAttributeDef( uiDictNum);
				if (pExtAttributeDef->uiDictNum == uiDictNum)
				{
					attrElmSetState( &pExtAttributeDef->attrElmDef, uiNewState);
				}
			}
		}
	}

Exit:

	if( pNode)
	{
		pNode->Release();
	}

	if( pAttr)
	{
		pAttr->Release();
	}
	
	if( pTrackerDoc)
	{
		pTrackerDoc->Release();
	}

	if( pChangeCountAttr)
	{
		pChangeCountAttr->Release();
	}

	if( bStartedTrans)
	{
		rc = commitTrans( 0, FALSE);
	}
	else if( RC_BAD( rc) && bMustAbortOnError)
	{
		setMustAbortTrans( rc);
	}

	return( rc);
}

/***************************************************************************
Desc:	Retrieve an element or attribute definition - get name, data type,
		state, and namespace.
***************************************************************************/
RCODE F_Db::getElmAttrInfo(
	FLMUINT				uiType,
	FLMUINT64			ui64DocumentID,
	F_AttrElmInfo *	pDefInfo,
	FLMBOOL				bOpeningDict,
	FLMBOOL				bDeleting)
{
	RCODE				rc = NE_XFLM_OK;
	F_DOMNode *		pDocNode = NULL;
	F_DOMNode *		pAttr = NULL;
	F_CachedNode *	pCachedDocNode;
	char				szTmpBuf[ 80];
	FLMUINT			uiNameId;
	FLMUNICODE *	puzName = NULL;
	FLMBOOL			bNamespaceDecl = FALSE;
	FLMBOOL			bHadUniqueSubElementTag = FALSE;

	// Retrieve the root element of the definition

	if (RC_BAD( rc = getNode( XFLM_DICT_COLLECTION, ui64DocumentID, &pDocNode)))
	{
		goto Exit;
	}

	flmAssert( pDocNode->getNodeType() == ELEMENT_NODE);

	pDefInfo->m_pDocNode = (IF_DOMNode *)pDocNode;
	pDocNode->AddRef();
	pCachedDocNode = pDocNode->m_pCachedNode;
	uiNameId = pCachedDocNode->getNameId();

	if( uiType != uiNameId)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

	// Cycle through the attributes on the root of the definition
	
	if( pCachedDocNode->hasAttributes())
	{
		if( RC_BAD( rc = pDocNode->getFirstAttribute( 
			this, (IF_DOMNode **)&pAttr)))
		{
			goto Exit;
		}
		
		for( ;;)
		{
			if( RC_BAD( rc = pAttr->getNameId( this, &uiNameId)))
			{
				goto Exit;
			}
			
			switch( uiNameId)
			{
				case ATTR_NAME_TAG:
				{
					pDefInfo->m_pNameAttr = (IF_DOMNode *)pAttr;
					pAttr->AddRef();

					if( RC_BAD( rc = pAttr->getUnicode( this, &puzName)))
					{
						goto Exit;
					}

					if( isXMLNS( puzName) &&
						 (puzName [5] == 0 ||
						  (puzName [5] == ':' && puzName [6] != 0)))
					{
						if( uiType == ELM_ATTRIBUTE_TAG)
						{
							pDefInfo->m_uiFlags |= ATTR_ELM_NS_DECL;
							bNamespaceDecl = TRUE;
						}
						else
						{
							rc = RC_SET( NE_XFLM_ILLEGAL_ELEMENT_NAME);
							goto Exit;
						}
					}

					if (uiType == ELM_ELEMENT_TAG)
					{
						if (RC_BAD( rc = fdictLegalElementName( puzName)))
						{
							goto Exit;
						}
					}
					else
					{
						if (RC_BAD( rc = fdictLegalAttributeName( puzName)))
						{
							goto Exit;
						}
					}

					// Put a freeze on name if we are not deleting.
					// Modify is allowed, but delete is not.

					if( !bDeleting && !bOpeningDict)
					{
						if( RC_BAD( rc = pAttr->addModeFlags(
							this, FDOM_CANNOT_DELETE)))
						{
							goto Exit;
						}
					}

					break;
				}

				case ATTR_TARGET_NAMESPACE_TAG:
				{
					pDefInfo->m_pTargetNamespaceAttr = (IF_DOMNode *)pAttr;
					pAttr->AddRef();
					break;
				}

				case ATTR_TYPE_TAG:
				{
					if( RC_BAD( rc = pAttr->getUTF8( this, (FLMBYTE *)szTmpBuf, 
						sizeof( szTmpBuf), 0,  ~((FLMUINT)0))))
					{
						goto Exit;
					}

					if (RC_BAD( rc = fdictGetDataType( 
						(char *)szTmpBuf, &pDefInfo->m_uiDataType)))
					{
						goto Exit;
					}

					if (uiType == ELM_ATTRIBUTE_TAG &&
						 pDefInfo->m_uiDataType == XFLM_NODATA_TYPE)
					{
						rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_DATA_TYPE);
						goto Exit;
					}

					break;
				}

				case ATTR_STATE_TAG:
				{
					if (RC_BAD( rc = pAttr->getUTF8( 
						this, (FLMBYTE *)szTmpBuf, 
						sizeof( szTmpBuf), 0, ~((FLMUINT)0))))
					{
						goto Exit;
					}

					if (RC_BAD( rc = fdictGetState( (char *)szTmpBuf,
						&pDefInfo->m_uiState)))
					{
						goto Exit;
					}

					// Freeze the state - can only be changed by an explicit
					// call to changeItemState.

					if( !bOpeningDict)
					{
						if( RC_BAD( rc = pAttr->addModeFlags( this,
							FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
						{
							goto Exit;
						}
					}
					break;
				}

				case ATTR_DICT_NUMBER_TAG:
				{
					if (RC_BAD( rc = pAttr->getUINT( this, &pDefInfo->m_uiDictNum)))
					{
						goto Exit;
					}

					if (uiType == ELM_ELEMENT_TAG)
					{
						if (RC_BAD( rc = fdictLegalElementNumber( 
							pDefInfo->m_uiDictNum, FALSE)))
						{
							goto Exit;
						}
					}
					else
					{
						if (RC_BAD( rc = fdictLegalAttributeNumber( 
							pDefInfo->m_uiDictNum, FALSE)))
						{
							goto Exit;
						}
					}
					break;
				}
				
				case ATTR_UNIQUE_SUB_ELEMENTS_TAG:
				{
					// This one is only allowed on element definitions.
					if (uiType != ELM_ELEMENT_TAG)
					{
						rc = RC_SET( NE_XFLM_ILLEGAL_UNIQUE_SUB_ELEMENT_VALUE);
						goto Exit;
					}
					bHadUniqueSubElementTag = TRUE;
					if( RC_BAD( rc = pAttr->getUTF8( this, (FLMBYTE *)szTmpBuf,
						sizeof( szTmpBuf), 0, ~((FLMUINT)0))))
					{
						goto Exit;
					}
					if (f_stricmp( szTmpBuf, "yes") == 0 ||
						 f_stricmp( szTmpBuf, "true") == 0 ||
						 f_stricmp( szTmpBuf, "1") == 0 ||
						 f_stricmp( szTmpBuf, "on") == 0 ||
						 f_stricmp( szTmpBuf, "enable") == 0)
					{
						pDefInfo->m_uiFlags |= ATTR_ELM_UNIQUE_SUBELMS;
					}
					else if (f_stricmp( szTmpBuf, "no") != 0 &&
								f_stricmp( szTmpBuf, "false") != 0 &&
								f_stricmp( szTmpBuf, "0") != 0 &&
								f_stricmp( szTmpBuf, "off") != 0 &&
								f_stricmp( szTmpBuf, "disable") != 0)
					{
						rc = RC_SET( NE_XFLM_ILLEGAL_UNIQUE_SUB_ELEMENT_VALUE);
						goto Exit;
					}
					
					// Freeze the state - cannot be changed or deleted.

					if( !bDeleting && !bOpeningDict)
					{
						if( RC_BAD( rc = pAttr->addModeFlags( this,
							FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
						{
							goto Exit;
						}
					}
					break;
				}
				
				default:
				{
					// Ignore all other attributes

					break;
				}
			}
			
			if( RC_BAD( rc = pAttr->getNextSibling( this, (IF_DOMNode **)&pAttr)))
			{
				if( rc != NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					goto Exit;
				}
				
				rc = NE_XFLM_OK;
				break;
			}
		}
	}

	// Make sure we had both a name and number specified

	if (!pDefInfo->m_pNameAttr)
	{
		rc = (RCODE)(uiType == ELM_ELEMENT_TAG
						 ? RC_SET( NE_XFLM_MISSING_ELEMENT_NAME)
						 : RC_SET( NE_XFLM_MISSING_ATTRIBUTE_NAME));
		goto Exit;
	}

	if (!pDefInfo->m_uiDictNum)
	{
		rc = (RCODE)(uiType == ELM_ELEMENT_TAG
						 ? RC_SET( NE_XFLM_MISSING_ELEMENT_NUMBER)
						 : RC_SET( NE_XFLM_MISSING_ATTRIBUTE_NUMBER));
		goto Exit;
	}
	
	if (!bDeleting && bNamespaceDecl && 
		pDefInfo->m_uiDataType != XFLM_TEXT_TYPE)
	{
		rc = RC_SET( NE_XFLM_ILLEGAL_NAMESPACE_DECL_DATATYPE);
		goto Exit;
	}
	
	// If we didn't have a unique sub-element tag, set one - better only
	// be missing when we first add the element definition.
	
	if (!bHadUniqueSubElementTag && uiType == ELM_ELEMENT_TAG)
	{
		flmAssert( !bOpeningDict && !bDeleting);
		
		// Set attribute and freeze it if it was missing.
	
		if (RC_BAD( rc = pDocNode->createAttribute( 
			this, ATTR_UNIQUE_SUB_ELEMENTS_TAG, (IF_DOMNode **)&pAttr)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = pAttr->setUTF8( this, (FLMBYTE *)"no")))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pAttr->addModeFlags( this,
			FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
		{
			goto Exit;
		}
	}
	else if (pDefInfo->m_uiFlags & ATTR_ELM_UNIQUE_SUBELMS)
	{
		// If sub-elements must be unique, the data type has to be
		// nodata.
		
		if (pDefInfo->m_uiDataType != XFLM_NODATA_TYPE)
		{
			rc = RC_SET( NE_XFLM_DATA_TYPE_MUST_BE_NO_DATA);
			goto Exit;
		}
	}

Exit:

	if( pDocNode)
	{
		pDocNode->Release();
	}

	if( pAttr)
	{
		pAttr->Release();
	}

	if( puzName)
	{
		f_free( &puzName);
	}

	return( rc);
}

/***************************************************************************
Desc:	Add, modify, or delete an element definition in the dictionary.
***************************************************************************/
RCODE F_Dict::updateElementDef(
	F_Db *		pDb,
	FLMUINT64	ui64DocumentID,
	FLMUINT		uiElementNum,
	FLMBOOL		bOpeningDict,
	FLMBOOL		bDeleting
	)
{
	RCODE    				rc = NE_XFLM_OK;
	ATTR_ELM_DEF *			pElementDef;
	EXT_ATTR_ELM_DEF *	pExtElementDef;
	FLMUNICODE *			puzElementName = NULL;
	FLMUNICODE *			puzNamespace = NULL;
	IX_ITEM *				pIxElement;
	F_AttrElmInfo			defInfo;
	F_DOMNode *				pTmpNode = NULL;

	if (bDeleting)
	{
		flmAssert( uiElementNum);

		// NOTE: It is possible that the element may not be in the
		// element table yet, because dictDocumentDone had not been
		// called to put it in there, but we are calling dictDocumentDone
		// to remove it.

		// Remove the tag number from the name table

		m_pNameTable->removeTag( ELM_ELEMENT_TAG, uiElementNum);

		if (uiElementNum <= FLM_HIGH_FIXED_ELEMENT_NUM)
		{
			if ((pElementDef = getElementDef( uiElementNum)) != NULL)
			{
				pElementDef->uiFlags = 0;	// Unused slot
				pElementDef->pFirstIcd = NULL;
			}
		}
		else
		{

			// Better be one of our extended elements - cannot delete
			// reserved elements.

			flmAssert( elementIsUserDefined( uiElementNum));

			// No need to lock the mutex, because no other threads can
			// be accessing it right now.

			if (m_pExtElementDefTbl)
			{
				pExtElementDef = getExtElementDef( uiElementNum);
				if (pExtElementDef->uiDictNum == uiElementNum)
				{
					pExtElementDef->uiDictNum = 0;
					pExtElementDef->attrElmDef.uiFlags = 0;
					pExtElementDef->attrElmDef.pFirstIcd = NULL;
				}
			}
		}
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->getElmAttrInfo( ELM_ELEMENT_TAG,
								ui64DocumentID, &defInfo, 
								bOpeningDict, bDeleting)))
	{
		goto Exit;
	}

	if (!uiElementNum)
	{
		uiElementNum = defInfo.m_uiDictNum;
	}

	flmAssert( uiElementNum == defInfo.m_uiDictNum);

	// Get the element name

	if( RC_BAD( rc = defInfo.m_pNameAttr->getUnicode( 
		pDb, &puzElementName)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = fdictLegalElementName( puzElementName)))
	{
		goto Exit;
	}

	// Get the element's namespace

	if( defInfo.m_pTargetNamespaceAttr)
	{
		if( RC_BAD( rc = defInfo.m_pTargetNamespaceAttr->getUnicode( 
			pDb, &puzNamespace)))
		{
			goto Exit;
		}
	}

	// Add to the tag table, unless we already have our quota of
	// element names.  Remove the tag by number first, in case
	// it was renamed.

	if (!bOpeningDict)
	{
		m_pNameTable->removeTag( ELM_ELEMENT_TAG, uiElementNum);
	}

	if (RC_BAD( rc = m_pNameTable->addTag( ELM_ELEMENT_TAG,
								puzElementName, NULL,
								uiElementNum, defInfo.m_uiDataType, puzNamespace, 
								bOpeningDict ? FALSE : TRUE)))
	{
		goto Exit;
	}

	if (uiElementNum <= FLM_HIGH_FIXED_ELEMENT_NUM)
	{
		// See if we are updating an existing element, in which case the
		// data type cannot be changed.

		if ((pElementDef = getElementDef( uiElementNum)) != NULL)
		{
			if (attrElmGetType( pElementDef) != defInfo.m_uiDataType)
			{
				rc = RC_SET( NE_XFLM_CANNOT_MOD_DATA_TYPE);
				goto Exit;
			}

			// Still need to assign uiFldType because state could change.

			pElementDef->uiFlags = (defInfo.m_uiDataType & ATTR_ELM_DATA_TYPE_MASK) |
											  (defInfo.m_uiState & ATTR_ELM_STATE_MASK) |
											  (defInfo.m_uiFlags & ATTR_ELM_FLAGS_MASK);
		}
		else
		{
			// If it will fit in the range of elements, simply set it.
			// Otherwise, we will need to reallocate the array to make
			// room for it.

			if (uiElementNum < m_uiLowestElementNum ||
				 uiElementNum > m_uiHighestElementNum)
			{
				if (RC_BAD( rc = reallocTbl( uiElementNum, sizeof( ATTR_ELM_DEF),
											(void **)&m_pElementDefTbl,
											&m_uiLowestElementNum,
											&m_uiHighestElementNum, 200,
											XFLM_MAX_ELEMENT_NUM)))
				{
					goto Exit;
				}
			}

			pElementDef = &m_pElementDefTbl [uiElementNum - m_uiLowestElementNum];
			pElementDef->uiFlags = (defInfo.m_uiDataType & ATTR_ELM_DATA_TYPE_MASK) |
											  (defInfo.m_uiState & ATTR_ELM_STATE_MASK) |
											  (defInfo.m_uiFlags & ATTR_ELM_FLAGS_MASK);
		}
	}
	else
	{
		// Cannot be modifying or adding in the reserved range. Element
		// number must be user defined.

		flmAssert( elementIsUserDefined( uiElementNum));

		// See if we should increase the extended table size. Do so if:
		// 1. It has not yet been allocated, or
		// 2. It has not yet reached its maximum size and this field
		//		number would hash to a slot beyond that maximum.

		if (!m_pExtElementDefTbl ||
			 (m_uiExtElementDefTblSize < MAX_EXT_ATTR_ELM_ARRAY_SIZE &&
			  uiElementNum % MAX_EXT_ATTR_ELM_ARRAY_SIZE >
			  m_uiExtElementDefTblSize - 1))
		{
			FLMUINT					uiNewSize = uiElementNum %
												MAX_EXT_ATTR_ELM_ARRAY_SIZE + 1000;
			FLMUINT					uiOldSize = m_uiExtElementDefTblSize;
			EXT_ATTR_ELM_DEF *	pNewTbl;
			EXT_ATTR_ELM_DEF *	pOldTbl = m_pExtElementDefTbl;
			FLMUINT					uiLoop;

			if (uiNewSize > MAX_EXT_ATTR_ELM_ARRAY_SIZE)
			{
				uiNewSize = MAX_EXT_ATTR_ELM_ARRAY_SIZE;
			}

			// If we had no array, allocate a mutex too.

			if (!pOldTbl)
			{

				// Better not be a mutex allocated at this point either.

				flmAssert( m_hExtElementDefMutex == F_MUTEX_NULL);
				if (RC_BAD( rc = f_mutexCreate( &m_hExtElementDefMutex)))
				{
					goto Exit;
				}
			}

			// Allocate a new array

			if (RC_BAD( rc = f_calloc( sizeof( EXT_ATTR_ELM_DEF) * uiNewSize,
										&pNewTbl)))
			{
				goto Exit;
			}
			m_pExtElementDefTbl = pNewTbl;
			m_uiExtElementDefTblSize = uiNewSize;

			// Rehash everything from the old table into the new table and then
			// free the old one.

			if (pOldTbl)
			{
				for (uiLoop = 0; uiLoop < uiOldSize; uiLoop++)
				{
					if (pOldTbl [uiLoop].uiDictNum)
					{
						pExtElementDef = getExtElementDef(
													pOldTbl [uiLoop].uiDictNum);
						f_memcpy( pExtElementDef, &pOldTbl [uiLoop],
										sizeof( EXT_ATTR_ELM_DEF));
					}
				}
				f_free( &pOldTbl);
			}
		}

		// Put the new or modified field into the table.  No need to lock the
		// mutex to do this because no other threads can be accessing this table
		// at this time.

		pExtElementDef = getExtElementDef( uiElementNum);
		pExtElementDef->uiDictNum = uiElementNum;
		pIxElement = findIxElement( uiElementNum);
		pExtElementDef->attrElmDef.pFirstIcd = pIxElement
															? pIxElement->pFirstIcd
															: NULL;
		pExtElementDef->attrElmDef.uiFlags =
												(defInfo.m_uiDataType & ATTR_ELM_DATA_TYPE_MASK) |
												(defInfo.m_uiState & ATTR_ELM_STATE_MASK) |
												(defInfo.m_uiFlags & ATTR_ELM_FLAGS_MASK);
	}

Exit:

	if (puzElementName)
	{
		f_free( &puzElementName);
	}

	if (puzNamespace)
	{
		f_free( &puzNamespace);
	}

	if( pTmpNode)
	{
		pTmpNode->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:	Add, modify, or delete an attribute definition in the dictionary.
***************************************************************************/
RCODE F_Dict::updateAttributeDef(
	F_Db *		pDb,
	FLMUINT64	ui64DocumentID,
	FLMUINT		uiAttributeNum,
	FLMBOOL		bOpeningDict,
	FLMBOOL		bDeleting
	)
{
	RCODE    				rc = NE_XFLM_OK;
	ATTR_ELM_DEF *			pAttributeDef;
	EXT_ATTR_ELM_DEF *	pExtAttributeDef;
	FLMUNICODE *			puzAttributeName = NULL;
	FLMUNICODE *			puzNamespace = NULL;
	IX_ITEM *				pIxAttribute;
	F_AttrElmInfo			defInfo;

	if (bDeleting)
	{
		flmAssert( uiAttributeNum);

		// NOTE: It is possible that the attribute may not be in the
		// attribute table yet, because dictDocumentDone had not been
		// called to put it in there, but we are calling dictDocumentDone
		// to remove it.

		// Remove the tag number from the name table

		m_pNameTable->removeTag( ELM_ATTRIBUTE_TAG, uiAttributeNum);

		if (uiAttributeNum <= FLM_HIGH_FIXED_ATTRIBUTE_NUM)
		{
			if ((pAttributeDef = getAttributeDef( uiAttributeNum)) != NULL)
			{
				pAttributeDef->uiFlags = 0;	// Unused slot
				pAttributeDef->pFirstIcd = NULL;
			}
		}
		else
		{
			// Better be one of our extended attributes - cannot delete
			// reserved attributes.

			flmAssert( attributeIsUserDefined( uiAttributeNum));

			// No need to lock the mutex, because no other threads can
			// be accessing it right now.

			if (m_pExtAttributeDefTbl)
			{
				pExtAttributeDef = getExtAttributeDef( uiAttributeNum);
				if (pExtAttributeDef->uiDictNum == uiAttributeNum)
				{
					pExtAttributeDef->uiDictNum = 0;
					pExtAttributeDef->attrElmDef.uiFlags = 0;
					pExtAttributeDef->attrElmDef.pFirstIcd = NULL;
				}
			}
		}
		goto Exit;
	}

	if (RC_BAD( rc = pDb->getElmAttrInfo( ELM_ATTRIBUTE_TAG,
								ui64DocumentID, &defInfo, 
								bOpeningDict, bDeleting)))
	{
		goto Exit;
	}

	if (!uiAttributeNum)
	{
		uiAttributeNum = defInfo.m_uiDictNum;
	}

	flmAssert( uiAttributeNum == defInfo.m_uiDictNum);

	// Get the attribute name

	if( RC_BAD( rc = defInfo.m_pNameAttr->getUnicode( 
		pDb, &puzAttributeName)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = fdictLegalAttributeName( puzAttributeName)))
	{
		goto Exit;
	}

	// Get the attribute's namespace

	if( defInfo.m_pTargetNamespaceAttr)
	{
		if( RC_BAD( rc = defInfo.m_pTargetNamespaceAttr->getUnicode( 
			pDb, &puzNamespace)))
		{
			goto Exit;
		}
	}

	// Add to the tag table, unless we already have our quota of
	// attribute names.  Remove the tag by number first, in case
	// it was renamed.

	if (!bOpeningDict)
	{
		m_pNameTable->removeTag( ELM_ATTRIBUTE_TAG, uiAttributeNum);
	}

	if (RC_BAD( rc = m_pNameTable->addTag( ELM_ATTRIBUTE_TAG,
								puzAttributeName, NULL,
								uiAttributeNum, defInfo.m_uiDataType, puzNamespace,
								bOpeningDict ? FALSE : TRUE)))
	{
		goto Exit;
	}

	if (uiAttributeNum <= FLM_HIGH_FIXED_ATTRIBUTE_NUM)
	{
		// See if we are updating an existing attribute, in which case the
		// data type cannot be changed.

		if ((pAttributeDef = getAttributeDef( uiAttributeNum)) != NULL)
		{
			if ( attrElmGetType( pAttributeDef) != defInfo.m_uiDataType)
			{
				rc = RC_SET( NE_XFLM_CANNOT_MOD_DATA_TYPE);
				goto Exit;
			}

			// Still need to assign uiFldType because state could change.

			pAttributeDef->uiFlags = (defInfo.m_uiDataType & ATTR_ELM_DATA_TYPE_MASK) |
												 (defInfo.m_uiState & ATTR_ELM_STATE_MASK) |
												 (defInfo.m_uiFlags & ATTR_ELM_FLAGS_MASK);
		}
		else
		{

			// If it will fit in the range of attributes, simply set it.
			// Otherwise, we will need to reallocate the array to make
			// room for it.

			if (uiAttributeNum < m_uiLowestAttributeNum ||
				 uiAttributeNum > m_uiHighestAttributeNum)
			{
				if (RC_BAD( rc = reallocTbl( uiAttributeNum, sizeof( ATTR_ELM_DEF),
											(void **)&m_pAttributeDefTbl,
											&m_uiLowestAttributeNum,
											&m_uiHighestAttributeNum, 200,
											XFLM_MAX_ATTRIBUTE_NUM)))
				{
					goto Exit;
				}
			}

			pAttributeDef = &m_pAttributeDefTbl [uiAttributeNum -
															 m_uiLowestAttributeNum];
			pAttributeDef->uiFlags = (defInfo.m_uiDataType & ATTR_ELM_DATA_TYPE_MASK) |
												 (defInfo.m_uiState & ATTR_ELM_STATE_MASK) |
												 (defInfo.m_uiFlags & ATTR_ELM_FLAGS_MASK);
		}
	}
	else
	{
		// Cannot be modifying or adding in the reserved range. Attribute
		// number must be user defined.

		flmAssert( attributeIsUserDefined( uiAttributeNum));

		// See if we should increase the extended table size. Do so if:
		// 1. It has not yet been allocated, or
		// 2. It has not yet reached its maximum size and this field
		//		number would hash to a slot beyond that maximum.

		if (!m_pExtAttributeDefTbl ||
			 (m_uiExtAttributeDefTblSize < MAX_EXT_ATTR_ELM_ARRAY_SIZE &&
			  uiAttributeNum % MAX_EXT_ATTR_ELM_ARRAY_SIZE >
			  m_uiExtAttributeDefTblSize - 1))
		{
			FLMUINT					uiNewSize = uiAttributeNum %
												MAX_EXT_ATTR_ELM_ARRAY_SIZE + 1000;
			FLMUINT					uiOldSize = m_uiExtAttributeDefTblSize;
			EXT_ATTR_ELM_DEF *	pNewTbl;
			EXT_ATTR_ELM_DEF *	pOldTbl = m_pExtAttributeDefTbl;
			FLMUINT					uiLoop;

			if (uiNewSize > MAX_EXT_ATTR_ELM_ARRAY_SIZE)
			{
				uiNewSize = MAX_EXT_ATTR_ELM_ARRAY_SIZE;
			}

			// If we had no array, allocate a mutex too.

			if (!pOldTbl)
			{
				// Better not be a mutex allocated at this point either.

				flmAssert( m_hExtAttributeDefMutex == F_MUTEX_NULL);
				if (RC_BAD( rc = f_mutexCreate( &m_hExtAttributeDefMutex)))
				{
					goto Exit;
				}
			}

			// Allocate a new array

			if (RC_BAD( rc = f_calloc( sizeof( EXT_ATTR_ELM_DEF) * uiNewSize,
										&pNewTbl)))
			{
				goto Exit;
			}
			m_pExtAttributeDefTbl = pNewTbl;
			m_uiExtAttributeDefTblSize = uiNewSize;

			// Rehash everything from the old table into the new table and then
			// free the old one.

			if (pOldTbl)
			{
				for (uiLoop = 0; uiLoop < uiOldSize; uiLoop++)
				{
					if (pOldTbl [uiLoop].uiDictNum)
					{
						pExtAttributeDef = getExtAttributeDef(
														pOldTbl [uiLoop].uiDictNum);
						f_memcpy( pExtAttributeDef, &pOldTbl [uiLoop],
										sizeof( EXT_ATTR_ELM_DEF));
					}
				}
				f_free( &pOldTbl);
			}
		}

		// Put the new or modified field into the table.  No need to lock the
		// mutex to do this because no other threads can be accessing this table
		// at this time.

		pExtAttributeDef = getExtAttributeDef( uiAttributeNum);
		pExtAttributeDef->uiDictNum = uiAttributeNum;
		pIxAttribute = findIxAttribute( uiAttributeNum);
		pExtAttributeDef->attrElmDef.pFirstIcd = pIxAttribute
															? pIxAttribute->pFirstIcd
															: NULL;
		pExtAttributeDef->attrElmDef.uiFlags =
												(defInfo.m_uiDataType & ATTR_ELM_DATA_TYPE_MASK) |
												(defInfo.m_uiState & ATTR_ELM_STATE_MASK) |
												(defInfo.m_uiFlags & ATTR_ELM_FLAGS_MASK);
	}

Exit:

	if (puzAttributeName)
	{
		f_free( &puzAttributeName);
	}

	if (puzNamespace)
	{
		f_free( &puzNamespace);
	}

	return( rc);
}

/***************************************************************************
Desc:	Determine if an index name is legal - formatwise
***************************************************************************/
FINLINE RCODE fdictLegalIndexName(
	FLMUNICODE *	// puzIndexName
	)
{
	//VISIT: Need to fill this out - may need to return NE_XFLM_ILLEGAL_INDEX_NAME
	return( NE_XFLM_OK);
}

/***************************************************************************
Desc:	Determine if an index number is legal
***************************************************************************/
FINLINE RCODE fdictLegalIndexNumber(
	FLMUINT	uiIndexNumber
	)
{
	return( (uiIndexNumber > 0 && uiIndexNumber <= XFLM_MAX_INDEX_NUM)
				? NE_XFLM_OK
				: RC_SET( NE_XFLM_ILLEGAL_INDEX_NUMBER));
}

/***************************************************************************
Desc:	Determine if a collection name is legal - formatwise
***************************************************************************/
FINLINE RCODE fdictLegalCollectionName(
	FLMUNICODE *	// puzCollectionName
	)
{
	//VISIT: Need to fill this out - may need to return NE_XFLM_ILLEGAL_COLLECTION_NAME
	return( NE_XFLM_OK);
}

/***************************************************************************
Desc:	Determine if a collection number is legal
***************************************************************************/
FINLINE RCODE fdictLegalCollectionNumber(
	FLMUINT	uiCollectionNumber
	)
{
	return( ( (uiCollectionNumber > 0 &&
				uiCollectionNumber <= XFLM_MAX_COLLECTION_NUM) ||
				(uiCollectionNumber == XFLM_DATA_COLLECTION))
				? NE_XFLM_OK
				: RC_SET( NE_XFLM_ILLEGAL_COLLECTION_NUMBER));
}

/***************************************************************************
Desc:	Determine if a prefix name is legal - formatwise
***************************************************************************/
FINLINE RCODE fdictLegalPrefixName(
	FLMUNICODE *	// puzPrefixName
	)
{
	//VISIT: Need to fill this out - may need to return NE_XFLM_ILLEGAL_PREFIX_NAME
	return( NE_XFLM_OK);
}

/***************************************************************************
Desc:	Determine if a prefix number is legal
***************************************************************************/
FINLINE RCODE fdictLegalPrefixNumber(
	FLMUINT	uiPrefixNumber
	)
{
	return( (uiPrefixNumber > 0 &&
				uiPrefixNumber <= XFLM_MAX_PREFIX_NUM)
				? NE_XFLM_OK
				: RC_SET( NE_XFLM_ILLEGAL_PREFIX_NUMBER));
}

/***************************************************************************
Desc:	Determine if an encryption def name is legal - formatwise
***************************************************************************/
FINLINE RCODE fdictLegalEncDefName(
	FLMUNICODE *	// puzEncDefName
	)
{
	//VISIT: Need to fill this out - may need to return NE_XFLM_ILLEGAL_ENCDEF_NAME
	return( NE_XFLM_OK);
}

/***************************************************************************
Desc:	Determine if an encryption def number is legal
***************************************************************************/
FINLINE RCODE fdictLegalEncDefNumber(
	FLMUINT	uiEncDefNumber
	)
{
	return( (uiEncDefNumber > 0 &&
				uiEncDefNumber <= XFLM_MAX_ENCDEF_NUM)
				? NE_XFLM_OK
				: RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_ENCDEF_NUMBER));
}

/***************************************************************************
Desc:	Determine if an encryption def type (algorithm) is legal
***************************************************************************/
FINLINE RCODE fdictLegalEncDefType(
	char *			pszEncDefType,
	FLMUINT *		puiEncDefType
	)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiType;

	// Parse the type keyword - only one type allowed.

	for( uiType = 0;
		  uiType < MAX_ENC_TYPES ;
		  uiType++)
	{
		if( f_strnicmp( pszEncDefType, DDEncOpts[ uiType], f_strlen(DDEncOpts[ uiType])) == 0)
		{
			*puiEncDefType = uiType;
			goto Exit;
		}
	}

	rc = RC_SET( NE_XFLM_INVALID_ENC_ALGORITHM);

Exit:
	return( rc);
}

/***************************************************************************
Desc:	Determine if a specified encryption key size is legal
***************************************************************************/
FINLINE RCODE fdictLegalEncKeySize(
	FLMUINT			uiEncType,
	FLMUINT			uiEncKeySize)
{
	RCODE			rc = NE_XFLM_OK;
	
	switch( uiEncType)
	{
		case FLM_NICI_AES:
		{
			if (uiEncKeySize != XFLM_NICI_AES128 &&
				 uiEncKeySize != XFLM_NICI_AES192 &&
				 uiEncKeySize != XFLM_NICI_AES256)
			{
				rc = RC_SET( NE_XFLM_INVALID_ENC_KEY_SIZE);
				goto Exit;
			}
			break;
		}
		case FLM_NICI_DES3:
		{
			if (uiEncKeySize != XFLM_NICI_DES3X)
			{
				rc = RC_SET( NE_XFLM_INVALID_ENC_KEY_SIZE);
				goto Exit;
			}
			break;
		}
		default:
		{
			rc = RC_SET( NE_XFLM_INVALID_ENC_ALGORITHM);
			goto Exit;
		}
	}
	
Exit:

	return rc;

}

/***************************************************************************
Desc:	return a legal key size, based on the encryption algorithm.
***************************************************************************/
FINLINE FLMBOOL fdictGetLegalKeySize(
	FLMUINT			uiEncType,
	FLMUINT *		puiEncKeySize
	)
{
	FLMBOOL		bSizeOk = FALSE;

	// Note:  The uiEncType should have already been validated.

	switch( uiEncType)
	{
		case FLM_NICI_AES:
		{
			if ( *puiEncKeySize == 0)
			{
				*puiEncKeySize = XFLM_NICI_AES256;
				bSizeOk = TRUE;
			}
			else if ( *puiEncKeySize == XFLM_NICI_AES256)
			{
				*puiEncKeySize = XFLM_NICI_AES192;
				bSizeOk = TRUE;
			}
			else if ( *puiEncKeySize == XFLM_NICI_AES192)
			{
				*puiEncKeySize = XFLM_NICI_AES128;
				bSizeOk = TRUE;
			}
			break;
		}
		case FLM_NICI_DES3:
		{
			if ( *puiEncKeySize == 0)
			{
				*puiEncKeySize = XFLM_NICI_DES3X;
				bSizeOk = TRUE;
			}
			break;
		}
	}
	
	return bSizeOk;

}

/***************************************************************************
Desc:	Parse an option from a string.  Function returns pointer to
		beginning of option.  Parameter is moved to just after where that
		option started.
***************************************************************************/
FSTATIC char * fdictGetOption(
	char **	ppszSrc
	)
{
	char *	pszMatch = NULL;
	char *	pszSrc = *ppszSrc;

	while (*pszSrc == NATIVE_SPACE)
	{
		pszSrc++;
	}

	// See if at end of string.

	if (!(*pszSrc))
	{
		goto Exit;
	}

	pszMatch = pszSrc;
	while (*pszSrc && *pszSrc != NATIVE_SPACE)
	{
		pszSrc++;
	}
	if (*pszSrc)
	{
		*pszSrc = 0;
		pszSrc++;
	}

Exit:

	*ppszSrc = pszSrc;
	return( pszMatch);
}

/***************************************************************************
Desc:	Retrieve an index definition.
***************************************************************************/
RCODE F_Db::getIndexDef(
	FLMUINT64		ui64DocumentID,
	FLMUNICODE **	ppuzIndexName,
	FLMUINT *		puiIndexNumber,
	FLMUINT *		puiCollectionNumber,
	FLMUINT *		puiLanguage,
	FLMUINT *		puiFlags,
	FLMUINT64 *		pui64LastDocIndexed,
	FLMUINT *		puiEncId,
	F_DOMNode **	ppNode,
	FLMBOOL			bOpeningDict,
	FLMBOOL			bDeleting)
{
	RCODE				rc = NE_XFLM_OK;
	F_DOMNode *		pNode = NULL;
	F_CachedNode *	pCachedNode;
	F_DOMNode *		pAttr = NULL;
	FLMBOOL			bHadIndexNumber = FALSE;
	FLMBOOL			bHadIndexName = FALSE;
	char				szTmpBuf[ 80];
	FLMUNICODE *	puzCollectionName = NULL;
	FLMUINT			uiNameCollectionNum = 0;
	FLMUINT			uiNameId;
	FLMBOOL			bHadLastDocIndexed = FALSE;
	FLMBOOL			bHadState = FALSE;

	// Set up defaults

	*ppuzIndexName = NULL;
	*puiIndexNumber = 0;
	*puiCollectionNumber = XFLM_DATA_COLLECTION;
	*puiLanguage = m_pDatabase->m_uiDefaultLanguage;
	*puiFlags = 0;
	*puiEncId = 0;
	*ppNode = NULL;

	// Retrieve the root element of the index definition

	if (RC_BAD( rc = getNode( XFLM_DICT_COLLECTION, ui64DocumentID, &pNode)))
	{
		goto Exit;
	}
	flmAssert( pNode->getNodeType() == ELEMENT_NODE);
	pCachedNode = pNode->m_pCachedNode;

	// Cycle through the attributes
	
	if( !pCachedNode->hasAttributes())
	{
		rc = RC_SET( NE_XFLM_MISSING_INDEX_NAME);
		goto Exit;
	}
	
	if( RC_BAD( rc = pNode->getFirstAttribute( this, (IF_DOMNode **)&pAttr)))
	{
		goto Exit;
	}

	for( ;;)
	{
		if( RC_BAD( rc = pAttr->getNameId( this, &uiNameId)))
		{
			goto Exit;
		}
		
		switch (uiNameId)
		{
			case ATTR_NAME_TAG:
			{
				if (RC_BAD( rc = pAttr->getUnicode( this, ppuzIndexName)))
				{
					goto Exit;
				}
				if (RC_BAD( rc = fdictLegalIndexName( *ppuzIndexName)))
				{
					goto Exit;
				}
				bHadIndexName = TRUE;
				break;
			}

			case ATTR_DICT_NUMBER_TAG:
			{
				if (RC_BAD( rc = pAttr->getUINT( this, puiIndexNumber)))
				{
					goto Exit;
				}
			
				if (RC_BAD( rc = fdictLegalIndexNumber( *puiIndexNumber)))
				{
					goto Exit;
				}

				bHadIndexNumber = TRUE;
				break;
			}

			case ATTR_COLLECTION_NUMBER_TAG:
			{
				if (RC_BAD( rc = pAttr->getUINT( this, puiCollectionNumber)))
				{
					goto Exit;
				}

				if (RC_BAD( rc = fdictLegalCollectionNumber(
											*puiCollectionNumber)))
				{
					goto Exit;
				}

				if (uiNameCollectionNum &&
					 *puiCollectionNumber != uiNameCollectionNum)
				{
					rc = RC_SET( NE_XFLM_COLLECTION_NAME_MISMATCH);
					goto Exit;
				}
				break;
			}

			case ATTR_COLLECTION_NAME_TAG:
			{
				F_DataVector	searchKey;
				F_DataVector	dataPart;

				if (RC_BAD( rc = pAttr->getUnicode( this, &puzCollectionName)))
				{
					goto Exit;
				}

				if (RC_BAD( rc = fdictLegalCollectionName( puzCollectionName)))
				{
					goto Exit;
				}

				// Get the collection number - look up in the index

				if (RC_BAD( rc = searchKey.setUINT( 0, ELM_COLLECTION_TAG)))
				{
					goto Exit;
				}

				if (RC_BAD( rc = searchKey.setUnicode( 1, puzCollectionName)))
				{
					goto Exit;
				}

				if (RC_BAD( rc = keyRetrieve( XFLM_DICT_NAME_INDEX,
							&searchKey, XFLM_EXACT, &dataPart)))
				{
					goto Exit;
				}

				// Data part of the retrieved key has the dictionary number
				// for this collection.

				if (RC_BAD( rc = dataPart.getUINT( 3, &uiNameCollectionNum)))
				{
					if (rc == NE_XFLM_NOT_FOUND)
					{
						uiNameCollectionNum = 0;
						rc = NE_XFLM_OK;
					}
					else
					{
						goto Exit;
					}
				}

				flmAssert( uiNameCollectionNum);

				if (*puiCollectionNumber &&
					 uiNameCollectionNum != *puiCollectionNumber)
				{
					rc = RC_SET( NE_XFLM_COLLECTION_NAME_MISMATCH);
					goto Exit;
				}

				*puiCollectionNumber = uiNameCollectionNum;
				break;
			}

			case ATTR_LANGUAGE_TAG:
			{
				if( RC_BAD( rc = pAttr->getUTF8( this, (FLMBYTE *)szTmpBuf,
							sizeof( szTmpBuf), 0, ~((FLMUINT)0))))
				{
					goto Exit;
				}

				*puiLanguage = f_languageToNum( (char *)szTmpBuf);
				break;
			}

			case ATTR_INDEX_OPTIONS_TAG:
			{
				char *	pszTmp;
				char *	pszOption;

				if( RC_BAD( rc = pAttr->getUTF8( this, (FLMBYTE *)szTmpBuf,
							sizeof( szTmpBuf), 0, ~((FLMUINT)0))))
				{
					goto Exit;
				}

				pszTmp = &szTmpBuf [0];
				while ((pszOption = fdictGetOption( (char **)&pszTmp)) != NULL)
				{
					if (f_stricmp( pszOption,
									XFLM_ABS_POS_OPTION_STR) == 0)
					{
						(*puiFlags) |= IXD_ABS_POS;
					}
					else
					{
						rc = RC_SET( NE_XFLM_INVALID_INDEX_OPTION);
						goto Exit;
					}
				}
				break;
			}

			case ATTR_STATE_TAG:
			{
				FLMUINT	uiState;

				if (RC_BAD( rc = pAttr->getUTF8( this, (FLMBYTE *)szTmpBuf,
					sizeof( szTmpBuf), 0, ~((FLMUINT)0))))
				{
					goto Exit;
				}

				if (RC_BAD( rc = fdictGetIndexState( szTmpBuf, &uiState)))
				{
					goto Exit;
				}

				*puiFlags = (*puiFlags & (~(IXD_SUSPENDED | IXD_OFFLINE))) |
								uiState;

				// Freeze the state - can only be changed by an explicit
				// call to indexSuspend or indexResume.

				if (!bOpeningDict)
				{
					if( RC_BAD( rc = pAttr->addModeFlags( this,
						FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
					{
						goto Exit;
					}
				}

				bHadState = TRUE;
				break;
			}

			case ATTR_LAST_DOC_INDEXED_TAG:
			{
				if (RC_BAD( rc = pAttr->getUINT64( this, pui64LastDocIndexed)))
				{
					goto Exit;
				}

				bHadLastDocIndexed = TRUE;
				break;
			}

			case ATTR_ENCRYPTION_ID_TAG:
			{
				if (RC_BAD( rc = pAttr->getUINT( this, puiEncId)))
				{
					goto Exit;
				}
			
				if (RC_BAD( rc = fdictLegalEncDefNumber( *puiEncId)))
				{
					goto Exit;
				}

				break;
			}

			default:
			{
				// Ignore all other attributes

				break;
			}
		}
		
		if( RC_BAD( rc = pAttr->getNextSibling( this, (IF_DOMNode **)&pAttr)))
		{
			if( rc != NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				goto Exit;
			}
			
			rc = NE_XFLM_OK;
			break;
		}
	}

	if (!bHadLastDocIndexed)
	{
		*pui64LastDocIndexed =
			(FLMUINT64)(((*puiFlags) & (IXD_SUSPENDED | IXD_OFFLINE))
							? (FLMUINT64)0
							: ~((FLMUINT64)0));
	}

	// Make sure we had both a name and number specified

	if (!bHadIndexName)
	{
		rc = RC_SET( NE_XFLM_MISSING_INDEX_NAME);
		goto Exit;
	}

	if (!bHadIndexNumber)
	{
		rc = RC_SET( NE_XFLM_MISSING_INDEX_NUMBER);
		goto Exit;
	}

	// Set a state attribute and freeze it if it was missing.

	if (!bOpeningDict && !bDeleting && !bHadState)
	{
		if (RC_BAD( rc = pNode->createAttribute( this, ATTR_STATE_TAG,
											(IF_DOMNode **)&pAttr)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = pAttr->setUTF8( this, 
			(FLMBYTE *)XFLM_INDEX_ONLINE_STR)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pAttr->addModeFlags( this,
			FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
		{
			goto Exit;
		}
	}

	*ppNode = pNode;

	// Set to NULL so it will not be released at Exit.

	pNode = NULL;

Exit:

	if (pNode)
	{
		pNode->Release();
	}

	if (pAttr)
	{
		pAttr->Release();
	}

	if (puzCollectionName)
	{
		f_free( &puzCollectionName);
	}

	return( rc);
}

/***************************************************************************
Desc:	Get the information for an index component.
***************************************************************************/
RCODE F_Db::getIndexComponentDef(
	F_Dict *			pDict,
	F_DOMNode *		pElementNode,
	FLMUINT			uiElementId,
	IXD *				pIxd,
	ICD *				pIcd
	)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pAttr = NULL;
	F_CachedNode *	pCachedNode;
	ICD *				pTmpIcd;
	ICD *				pPrevIcd;
	char				szTmpBuf[ 200];
	char *			pszTmp;
	char *			pszOption;
	FLMUNICODE *	puzName = NULL;
	FLMUNICODE *	puzNamespace = NULL;
	FLMUINT			uiKeyComponent;
	FLMUINT			uiDataComponent;
	FLMUINT			uiNameDictNumber = 0;
	FLMUINT			uiDataType= 0;
	FLMBOOL			bLimitSet = FALSE;
	FLMBOOL			bRequiredSet = FALSE;
	FLMBOOL			bIndexOnSet = FALSE;
	FLMBOOL			bCompareRuleSet = FALSE;
	FLMBOOL			bIsAttr = FALSE;
	FLMBOOL			bHadDictNumber = FALSE;
	F_NameTable *	pNameTable = NULL;
	FLMUINT			uiNameId;

	if (uiElementId == ELM_ATTRIBUTE_COMPONENT_TAG)
	{
		bIsAttr = TRUE;
		pIcd->uiFlags |= ICD_IS_ATTRIBUTE;
	}
	
	pCachedNode = pElementNode->m_pCachedNode;
	
	if( !pCachedNode->hasAttributes())
	{
		rc = (RCODE)(bIsAttr
						 ? RC_SET( NE_XFLM_MISSING_ATTRIBUTE_NUMBER)
						 : RC_SET( NE_XFLM_MISSING_ELEMENT_NUMBER));
		goto Exit;
	}
	
	if( RC_BAD( rc = pElementNode->getFirstAttribute( this, &pAttr)))
	{
		goto Exit;
	}

	for( ;;)
	{
		if( RC_BAD( rc = pAttr->getNameId( this, &uiNameId)))
		{
			goto Exit;
		}
		
		switch (uiNameId)
		{
			case ATTR_DICT_NUMBER_TAG:
				if (RC_BAD( rc = pAttr->getUINT( this, &pIcd->uiDictNum)))
				{
					goto Exit;
				}
				if (bIsAttr)
				{
					if (RC_BAD( rc = fdictLegalAttributeNumber( pIcd->uiDictNum, TRUE)))
					{
						goto Exit;
					}
				}
				else
				{
					if (RC_BAD( rc = fdictLegalElementNumber( pIcd->uiDictNum, TRUE)))
					{
						goto Exit;
					}
				}
				bHadDictNumber = TRUE;
				break;

			case ATTR_NAME_TAG:
				if (RC_BAD( rc = pAttr->getUnicode( this, &puzName)))
				{
					goto Exit;
				}
				if (bIsAttr)
				{
					if (RC_BAD( rc = fdictLegalAttributeName( puzName)))
					{
						goto Exit;
					}
				}
				else
				{
					if (RC_BAD( rc = fdictLegalElementName( puzName)))
					{
						goto Exit;
					}
				}
				break;

			case ATTR_TARGET_NAMESPACE_TAG:
				if (RC_BAD( rc = pAttr->getUnicode( this, &puzNamespace)))
				{
					goto Exit;
				}
				break;

			case ATTR_INDEX_ON_TAG:
				if( RC_BAD( rc = pAttr->getUTF8( this, (FLMBYTE *)szTmpBuf,
					sizeof( szTmpBuf), 0, ~((FLMUINT)0))))
				{
					goto Exit;
				}
				if (f_stricmp( szTmpBuf, XFLM_VALUE_OPTION_STR) == 0)
				{
					pIcd->uiFlags |= ICD_VALUE;
					pIcd->uiFlags &= (~(ICD_PRESENCE | ICD_SUBSTRING |
											  ICD_EACHWORD | ICD_METAPHONE));
				}
				else if (f_stricmp( szTmpBuf, XFLM_PRESENCE_OPTION_STR) == 0)
				{
					pIcd->uiFlags |= ICD_PRESENCE;
					pIcd->uiFlags &= (~(ICD_VALUE | ICD_SUBSTRING |
											  ICD_EACHWORD | ICD_METAPHONE));
				}
				else if (f_stricmp( szTmpBuf, XFLM_SUBSTRING_OPTION_STR) == 0)
				{
					pIcd->uiFlags |= ICD_SUBSTRING;
					pIcd->uiFlags &= (~(ICD_PRESENCE | ICD_VALUE | 
											  ICD_EACHWORD | ICD_METAPHONE));
				}
				else if (f_stricmp( szTmpBuf, XFLM_EACHWORD_OPTION_STR) == 0)
				{
					pIcd->uiFlags |= ICD_EACHWORD;
					pIcd->uiFlags &= (~(ICD_PRESENCE | ICD_VALUE | 
											  ICD_SUBSTRING | ICD_METAPHONE));
				}
				else if (f_stricmp( szTmpBuf, XFLM_METAPHONE_OPTION_STR) == 0)
				{
					pIcd->uiFlags |= ICD_METAPHONE;
					pIcd->uiFlags &= (~(ICD_PRESENCE | ICD_VALUE | 
											  ICD_SUBSTRING | ICD_EACHWORD));
				}
				else
				{
					rc = RC_SET( NE_XFLM_ILLEGAL_INDEX_ON);
					goto Exit;
				}
				bIndexOnSet = TRUE;
				break;

			case ATTR_REQUIRED_TAG:
				if( RC_BAD( rc = pAttr->getUTF8( this, (FLMBYTE *)szTmpBuf,
					sizeof( szTmpBuf), 0, ~((FLMUINT)0))))
				{
					goto Exit;
				}
				if (f_stricmp( szTmpBuf, "yes") == 0 ||
					 f_stricmp( szTmpBuf, "true") == 0 ||
					 f_stricmp( szTmpBuf, "1") == 0 ||
					 f_stricmp( szTmpBuf, "on") == 0 ||
					 f_stricmp( szTmpBuf, "enable") == 0)
				{
					pIcd->uiFlags |= ICD_REQUIRED_PIECE;
				}
				else if (f_stricmp( szTmpBuf, "no") != 0 &&
							f_stricmp( szTmpBuf, "false") != 0 &&
							f_stricmp( szTmpBuf, "0") != 0 &&
							f_stricmp( szTmpBuf, "off") != 0 &&
							f_stricmp( szTmpBuf, "disable") != 0)
				{
					rc = RC_SET( NE_XFLM_ILLEGAL_REQUIRED_VALUE);
					goto Exit;
				}
				bRequiredSet = TRUE;
				break;

			case ATTR_LIMIT_TAG:
				if (RC_BAD( rc = pAttr->getUINT( this, &pIcd->uiLimit)))
				{
					goto Exit;
				}
				bLimitSet = TRUE;
				break;

			case ATTR_COMPARE_RULES_TAG:
				if (RC_BAD( rc = pAttr->getUTF8( this, (FLMBYTE *)szTmpBuf,
					sizeof( szTmpBuf), 0, ~((FLMUINT)0))))
				{
					goto Exit;
				}

				pszTmp = &szTmpBuf [0];
				while ((pszOption = fdictGetOption( &pszTmp)) != NULL)
				{
					if (f_stricmp( pszOption,
							XFLM_CASE_INSENSITIVE_OPTION_STR) == 0)
					{
						pIcd->uiCompareRules |= XFLM_COMP_CASE_INSENSITIVE;
					}
					else if (f_stricmp( pszOption,
									XFLM_MINSPACES_OPTION_STR) == 0)
					{
						pIcd->uiCompareRules |= XFLM_COMP_COMPRESS_WHITESPACE;
					}
					else if (f_stricmp( pszOption,
									XFLM_WHITESPACE_AS_SPACE_STR) == 0)
					{
						pIcd->uiCompareRules |= XFLM_COMP_WHITESPACE_AS_SPACE;
					}
					else if (f_stricmp( pszOption,
									XFLM_IGNORE_LEADINGSPACES_OPTION_STR) == 0)
					{
						pIcd->uiCompareRules |= XFLM_COMP_IGNORE_LEADING_SPACE;
					}
					else if (f_stricmp( pszOption,
									XFLM_IGNORE_TRAILINGSPACES_OPTION_STR) == 0)
					{
						pIcd->uiCompareRules |= XFLM_COMP_IGNORE_TRAILING_SPACE;
					}
					else if (f_stricmp( pszOption,
									XFLM_NOUNDERSCORE_OPTION_STR) == 0)
					{
						pIcd->uiCompareRules |= XFLM_COMP_NO_UNDERSCORES;
					}
					else if (f_stricmp( pszOption,
									XFLM_NOSPACE_OPTION_STR) == 0)
					{
						pIcd->uiCompareRules |= XFLM_COMP_NO_WHITESPACE;
					}
					else if (f_stricmp( pszOption,
									XFLM_NODASH_OPTION_STR) == 0)
					{
						pIcd->uiCompareRules |= XFLM_COMP_NO_DASHES;
					}
					else if (f_stricmp( pszOption,
									XFLM_DESCENDING_OPTION_STR) == 0)
					{
						pIcd->uiFlags |= ICD_DESCENDING;
					}
					else if (f_stricmp( pszOption,
									XFLM_MISSING_HIGH_OPTION_STR) == 0)
					{
						pIcd->uiFlags |= ICD_MISSING_HIGH;
					}
					else
					{
						rc = RC_SET( NE_XFLM_INVALID_COMPARE_RULE);
						goto Exit;
					}
				}
				bCompareRuleSet = TRUE;
				break;

			case ATTR_KEY_COMPONENT_TAG:
				if( RC_BAD( rc = pAttr->getUINT( this, &uiKeyComponent)))
				{
					goto Exit;
				}
				if (!uiKeyComponent)
				{
					rc = RC_SET( NE_XFLM_ILLEGAL_KEY_COMPONENT_NUM);
					goto Exit;
				}

				// Insert into its place in the list of key components

				pPrevIcd = NULL;
				pTmpIcd = pIxd->pFirstKey;
				while (pTmpIcd && uiKeyComponent > pTmpIcd->uiKeyComponent)
				{
					pPrevIcd = pTmpIcd;
					pTmpIcd = pTmpIcd->pNextKeyComponent;
				}

				// Make sure not already in the list

				if (pTmpIcd && pTmpIcd->uiKeyComponent == uiKeyComponent)
				{
					rc = RC_SET( NE_XFLM_DUPLICATE_KEY_COMPONENT);
					goto Exit;
				}

				// Link after pPrevIcd

				if (pPrevIcd)
				{
					if ((pIcd->pNextKeyComponent =
							pPrevIcd->pNextKeyComponent) != NULL)
					{
						pIcd->pNextKeyComponent->pPrevKeyComponent = pIcd;
					}
					else
					{
						pIxd->pLastKey = pIcd;
					}
					pPrevIcd->pNextKeyComponent = pIcd;
					pIcd->pPrevKeyComponent = pPrevIcd;
				}
				else
				{
					if ((pIcd->pNextKeyComponent = pIxd->pFirstKey) != NULL)
					{
						pIxd->pFirstKey->pPrevKeyComponent = pIcd;
					}
					else
					{
						pIxd->pLastKey = pIcd;
					}
					pIxd->pFirstKey = pIcd;
				}
				pIcd->uiKeyComponent = uiKeyComponent;
				pIxd->uiNumKeyComponents++;

				break;

			case ATTR_DATA_COMPONENT_TAG:
				if( RC_BAD( rc = pAttr->getUINT( this, &uiDataComponent)))
				{
					goto Exit;
				}
				if (!uiDataComponent)
				{
					rc = RC_SET( NE_XFLM_ILLEGAL_DATA_COMPONENT_NUM);
					goto Exit;
				}

				// Insert into its place in the list of data components

				pPrevIcd = NULL;
				pTmpIcd = pIxd->pFirstData;
				while (pTmpIcd && uiDataComponent > pTmpIcd->uiDataComponent)
				{
					pPrevIcd = pTmpIcd;
					pTmpIcd = pTmpIcd->pNextDataComponent;
				}

				// Make sure not already in the list

				if (pTmpIcd && pTmpIcd->uiDataComponent == uiDataComponent)
				{
					rc = RC_SET( NE_XFLM_DUPLICATE_DATA_COMPONENT);
					goto Exit;
				}

				// Link after pPrevIcd

				if (pPrevIcd)
				{
					if ((pIcd->pNextDataComponent =
							pPrevIcd->pNextDataComponent) != NULL)
					{
						pIcd->pNextDataComponent->pPrevDataComponent = pIcd;
					}
					else
					{
						pIxd->pLastData = pIcd;
					}
					pPrevIcd->pNextDataComponent = pIcd;
					pIcd->pPrevDataComponent = pPrevIcd;
				}
				else
				{
					if ((pIcd->pNextDataComponent = pIxd->pFirstData) != NULL)
					{
						pIxd->pFirstData->pPrevDataComponent = pIcd;
					}
					else
					{
						pIxd->pLastData = pIcd;
					}
					pIxd->pFirstData = pIcd;
				}
				pIcd->uiDataComponent = uiDataComponent;
				pIxd->uiNumDataComponents++;

				break;

			default:

				// Ignore all other attributes

				break;
		}
		
		if( RC_BAD( rc = pAttr->getNextSibling( this, &pAttr)))
		{
			if( rc != NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				goto Exit;
			}
			
			rc = NE_XFLM_OK;
			break;
		}
	}

	// If they specified a name, see if we can find it

	if (puzName)
	{
		if (*puzName == '/' && *(puzName + 1) == 0)
		{
			uiNameDictNumber = ELM_ROOT_TAG;

			// For now, we can only allow presence indexing on
			// ELM_ROOT_TAG.  That is because we don't yet
			// allow specifying of data type per ICD

			if (!(pIcd->uiFlags & ICD_PRESENCE))
			{
				rc = RC_SET( NE_XFLM_MUST_INDEX_ON_PRESENCE);
				goto Exit;
			}

			// ELM_ROOT_TAG cannot be specified as a data
			// component.

			if (pIcd->uiDataComponent)
			{
				rc = RC_SET( NE_XFLM_ILLEGAL_DATA_COMPONENT);
				goto Exit;
			}
		}
		else
		{
			F_DataVector	searchKey;
			F_DataVector	dataPart;

			// Get the dictionary number - look up in the index

			if (RC_BAD( rc = searchKey.setUINT( 0,
									uiElementId == ELM_ELEMENT_COMPONENT_TAG
									? ELM_ELEMENT_TAG
									: ELM_ATTRIBUTE_TAG)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = searchKey.setUnicode( 1, puzName)))
			{
				goto Exit;
			}
			if (puzNamespace)
			{
				if (RC_BAD( rc = searchKey.setUnicode( 2, puzNamespace)))
				{
					goto Exit;
				}
			}
			if (RC_BAD( rc = keyRetrieve( XFLM_DICT_NAME_INDEX,
						&searchKey, XFLM_EXACT, &dataPart)))
			{
				if (rc == NE_XFLM_NOT_FOUND)
				{
					// This may be a built-in type which is not indexed.

					if ( RC_BAD( rc = getNameTable( &pNameTable)))
					{
						goto Exit;
					}

					if ( RC_BAD( rc = pNameTable->getFromTagTypeAndName(
						this,
						(bIsAttr) ? ELM_ATTRIBUTE_TAG : ELM_ELEMENT_TAG,
						puzName,
						NULL,
						TRUE,
						puzNamespace,
						&uiNameDictNumber)))
					{
						if ( rc == NE_XFLM_NOT_FOUND)
						{
							rc = (RCODE)(bIsAttr
								? RC_SET( NE_XFLM_UNDEFINED_ATTRIBUTE_NAME)
								: RC_SET( NE_XFLM_UNDEFINED_ELEMENT_NAME));
						}
						goto Exit;
					}
				}
				else
				{
					goto Exit;
				}
			}
			else
			{
				// Data part of the retrieved key has the dictionary number
				// for this collection.

				if (RC_BAD( rc = dataPart.getUINT( 3, &uiNameDictNumber)))
				{
					if (rc == NE_XFLM_NOT_FOUND)
					{
						uiNameDictNumber = 0;
						rc = NE_XFLM_OK;
					}
					else
					{
						goto Exit;
					}
				}
			}

			flmAssert( uiNameDictNumber);
		}

		if (pIcd->uiDictNum &&
			 uiNameDictNumber != pIcd->uiDictNum)
		{
			rc = (RCODE)(bIsAttr
							 ? RC_SET( NE_XFLM_ATTRIBUTE_NAME_MISMATCH)
							 : RC_SET( NE_XFLM_ELEMENT_NAME_MISMATCH));
			goto Exit;
		}
		pIcd->uiDictNum = uiNameDictNumber;
		bHadDictNumber = TRUE;
	}

	// If this is a key component, make a few more checks

	if (pIcd->uiKeyComponent)
	{

		// If no limit was set, use a default.

		if (pIcd->uiFlags & ICD_SUBSTRING)
		{
			pIxd->uiFlags |= IXD_HAS_SUBSTRING;
		}
		if (!bLimitSet)
		{
			if (pIcd->uiFlags & ICD_SUBSTRING)
			{
				pIcd->uiLimit = ICD_DEFAULT_SUBSTRING_LIMIT;
			}
			else
			{
				pIcd->uiLimit = ICD_DEFAULT_LIMIT;
			}
		}
	}
	else
	{
		// There are certain things that cannot be set for data
		// and context components.  Verify that they were not set.

		if (bRequiredSet)
		{
			rc = RC_SET( NE_XFLM_CANNOT_SET_REQUIRED);
			goto Exit;
		}
		if (bLimitSet)
		{
			rc = RC_SET( NE_XFLM_CANNOT_SET_LIMIT);
			goto Exit;
		}
		if (bIndexOnSet)
		{
			rc = RC_SET( NE_XFLM_CANNOT_SET_INDEX_ON);
			goto Exit;
		}
		if (bCompareRuleSet)
		{
			rc = RC_SET( NE_XFLM_CANNOT_SET_COMPARE_RULES);
			goto Exit;
		}
		if (!pIcd->uiDataComponent)
		{

			// Insert into its place in the list of context components

			pPrevIcd = NULL;
			pTmpIcd = pIxd->pFirstContext;
			while (pTmpIcd && pIcd->uiCdl > pTmpIcd->uiCdl)
			{
				pPrevIcd = pTmpIcd;
				pTmpIcd = pTmpIcd->pNextKeyComponent;
			}

			// Link after pPrevIcd

			if (pPrevIcd)
			{
				if ((pIcd->pNextKeyComponent =
						pPrevIcd->pNextKeyComponent) != NULL)
				{
					pIcd->pNextKeyComponent->pPrevKeyComponent = pIcd;
				}
				else
				{
					pIxd->pLastContext = pIcd;
				}
				pPrevIcd->pNextKeyComponent = pIcd;
				pIcd->pPrevKeyComponent = pPrevIcd;
			}
			else
			{
				if ((pIcd->pNextKeyComponent = pIxd->pFirstContext) != NULL)
				{
					pIxd->pFirstContext->pPrevKeyComponent = pIcd;
				}
				else
				{
					pIxd->pLastContext = pIcd;
				}
				pIxd->pFirstContext = pIcd;
			}
			pIxd->uiNumContextComponents++;
		}
	}

	// Make sure we had an element or attribute number specified

	if (!bHadDictNumber)
	{
		rc = (RCODE)(bIsAttr
						 ? RC_SET( NE_XFLM_MISSING_ATTRIBUTE_NUMBER)
						 : RC_SET( NE_XFLM_MISSING_ELEMENT_NUMBER));
		goto Exit;
	}

	// Get the element or attribute's data type

	if (bIsAttr)
	{
		F_AttrElmInfo		attrInfo;

		if (RC_BAD( rc = pDict->getAttribute( this, pIcd->uiDictNum, &attrInfo)))
		{
			goto Exit;
		}

		uiDataType = attrInfo.m_uiDataType;
	}
	else
	{
		if (pIcd->uiDictNum == ELM_ROOT_TAG)
		{
			uiDataType = XFLM_NODATA_TYPE;
		}
		else
		{
			F_AttrElmInfo		elmInfo;

			if (RC_BAD( rc = pDict->getElement( this, pIcd->uiDictNum, &elmInfo)))
			{
				goto Exit;
			}

			uiDataType = elmInfo.m_uiDataType;
		}
	}
	icdSetDataType( pIcd, uiDataType);

Exit:

	if (pAttr)
	{
		pAttr->Release();
	}

	if (puzName)
	{
		f_free( &puzName);
	}

	if (puzNamespace)
	{
		f_free( &puzNamespace);
	}

	if (pNameTable)
	{
		pNameTable->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:	Determine if a NODE is an index component.
***************************************************************************/
FSTATIC RCODE isIndexComponent(
	F_Db *		pDb,
	F_DOMNode *	pNode,
	FLMBOOL *	pbIsIndexComponent,
	FLMUINT *	puiElementNum)
{
	RCODE				rc = NE_XFLM_OK;

	*pbIsIndexComponent = TRUE;
	
	if( pNode->getNodeType() != ELEMENT_NODE)
	{
		*pbIsIndexComponent = FALSE;
		goto Exit;
	}

	if( RC_BAD( rc = pNode->getNameId( pDb, puiElementNum)))
	{
		goto Exit;
	}

	switch (*puiElementNum)
	{
		case ELM_ELEMENT_COMPONENT_TAG:
		case ELM_ATTRIBUTE_COMPONENT_TAG:
			break;
		default:
			*pbIsIndexComponent = FALSE;
			goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Compare the old index definition with the new to see if anything
		changed - to determine if we really need to rebuild the index.
***************************************************************************/
FSTATIC FLMBOOL indexDefsSame(
	IXD *	pOldIxd,
	IXD *	pNewIxd
	)
{
	FLMBOOL	bSame = FALSE;
	ICD *		pOldIcd;
	ICD *		pNewIcd;

	if (pOldIxd->uiCollectionNum != pNewIxd->uiCollectionNum ||
		 pOldIxd->uiNumIcds != pNewIxd->uiNumIcds ||
		 pOldIxd->uiNumKeyComponents != pNewIxd->uiNumKeyComponents ||
		 pOldIxd->uiNumDataComponents != pNewIxd->uiNumDataComponents ||
		 ~(pOldIxd->uiFlags & (IXD_OFFLINE | IXD_SUSPENDED)) !=
		 ~(pNewIxd->uiFlags & (IXD_OFFLINE | IXD_SUSPENDED)) ||
		 pOldIxd->uiLanguage != pNewIxd->uiLanguage)
	{
		goto Exit;
	}

	// Traverse the ICDs and make sure they are the same

	pOldIcd = pOldIxd->pIcdTree;
	pNewIcd = pNewIxd->pIcdTree;

	for (;;)
	{

		// Compare the ICDs

		if (pOldIcd->uiDictNum != pNewIcd->uiDictNum ||
			 pOldIcd->uiFlags != pNewIcd->uiFlags ||
			 pOldIcd->uiCdl != pNewIcd->uiCdl ||
			 pOldIcd->uiKeyComponent != pNewIcd->uiKeyComponent ||
			 pOldIcd->uiDataComponent != pNewIcd->uiDataComponent ||
			 pOldIcd->uiLimit != pNewIcd->uiLimit)
		{
			goto Exit;
		}

		if (pOldIcd->pFirstChild)
		{
			if (!pNewIcd->pFirstChild)
			{

				// Old ICD has a child, new one doesn't, indexes are
				// different.

				goto Exit;
			}
			pOldIcd = pOldIcd->pFirstChild;
			pNewIcd = pNewIcd->pFirstChild;
			continue;
		}
		while (pOldIcd && !pOldIcd->pNextSibling)
		{

			// Old ICD has no next sibling, new one doesn't, indexes are
			// different.

			if (!pNewIcd || pNewIcd->pNextSibling)
			{
				goto Exit;
			}
			pOldIcd = pOldIcd->pParent;
			pNewIcd = pNewIcd->pParent;
		}
		if (!pOldIcd)
		{

			// Traversed back to parent ICD for old ICD, but not
			// for new ICD, indexes are different.  However, this
			// should never happen.

			if (pNewIcd)
			{
				flmAssert( 0);
				goto Exit;
			}
			break;
		}

		// OLD ICD has a sibling it can traverse to, new one
		// doesn't, indexes are different.

		if (!pNewIcd || !pNewIcd->pNextSibling)
		{
			goto Exit;
		}
		pOldIcd = pOldIcd->pNextSibling;
		pNewIcd = pNewIcd->pNextSibling;
	}

	bSame = TRUE;

Exit:

	return( bSame);
}

/***************************************************************************
Desc:	Update an index definition.
***************************************************************************/
RCODE F_Dict::updateIndexDef(
	F_Db *		pDb,
	FLMUINT64	ui64DocumentID,
	FLMUINT		uiIndexNum,
	FLMBOOL		bOpeningDict,
	FLMBOOL		bDeleting
	)
{
	RCODE				rc = NE_XFLM_OK;
	void *			pvMark = m_dictPool.poolMark();
	FLMUNICODE *	puzIndexName = NULL;
	F_DOMNode *		pNode = NULL;
	FLMUINT			uiElementId;
	FLMBOOL			bIsIndexComponent;
	FLMUINT			uiComponentNum;
	IXD *				pIxd;
	IXD *				pOldIxd;
	ICD *				pIcd;
	ICD *				pLastIcd;
	ICD *				pTmpIcd;
	FLMBOOL			bLinkAsChild;
	FLMBOOL			bHadRequired;
	FLMBOOL			bSinglePath;
	FLMBOOL			bHasChildren;
	FLMUINT			uiEncId;

	if (bOpeningDict)
	{
		flmAssert( !bDeleting);
		pOldIxd = NULL;
	}
	else
	{
		if (RC_BAD( rc = getIndex( uiIndexNum, NULL, &pOldIxd, TRUE)))
		{
			if (rc == NE_XFLM_BAD_IX)
			{
				pOldIxd = NULL;
				rc = NE_XFLM_OK;
			}
			else
			{
				goto Exit;
			}
		}
	}

	if (bDeleting)
	{
		flmAssert( uiIndexNum);

		if (pOldIxd)
		{

			// Get rid of the old b-tree

			if (RC_BAD( rc = pDb->m_pDatabase->lFileDelete( pDb, NULL, &pOldIxd->lfInfo,
							(FLMBOOL)((pOldIxd->uiFlags & IXD_ABS_POS)
										 ? (FLMBOOL)TRUE
										 : (FLMBOOL)FALSE),
							(FLMBOOL)(pOldIxd->pFirstData
										 ? (FLMBOOL)TRUE
										 : (FLMBOOL)FALSE))))
			{
				goto Exit;
			}

			// Remove from index fixup list if we are deleting an index.
			// It is impossible to be deleting an index in a background
			// thread, so if there is a fixup, it is here because the
			// index was added during this transaction (in the background).
			// If the transaction aborts the IXD will simply go away, and
			// there will be no need to fix it up.

			if (pDb->m_pIxdFixups)
			{
				IXD_FIXUP *	pIxdFixup = pDb->m_pIxdFixups;
				IXD_FIXUP *	pPrevIxdFixup = NULL;

				while (pIxdFixup && pIxdFixup->uiIndexNum != uiIndexNum)
				{
					pPrevIxdFixup = pIxdFixup;
					pIxdFixup = pIxdFixup->pNext;
				}

				if (pIxdFixup)
				{
					if (pPrevIxdFixup)
					{
						pPrevIxdFixup->pNext = pIxdFixup->pNext;
					}
					else
					{
						pDb->m_pIxdFixups = pIxdFixup->pNext;
					}
					f_free( &pIxdFixup);
				}
			}

			// On delete or modify index make sure something is in the stop list.

			if (!(pDb->m_uiFlags & FDB_REPLAYING_RFL))
			{
				if( RC_BAD( rc = pDb->addToStopList( uiIndexNum)))
				{
					goto Exit;
				}
			}

			// Unlink the old ICDs.

			unlinkIcds( pOldIxd->pIcdTree);
		}

		// NOTE: It is possible that the index may not be in the
		// index table yet, because dictDocumentDone had not been
		// called to put it in there, but we are calling dictDocumentDone
		// to remove it.

		// Remove the tag number from the name table

		m_pNameTable->removeTag( ELM_INDEX_TAG, uiIndexNum);

		if (uiIndexNum >= m_uiLowestIxNum && uiIndexNum <= m_uiHighestIxNum)
		{
			m_ppIxdTbl [uiIndexNum - m_uiLowestIxNum] = NULL;
		}
		goto Exit;
	}

	// Allocate a new IXD

	if (RC_BAD( rc = m_dictPool.poolCalloc( sizeof( IXD), (void **)&pIxd)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pDb->getIndexDef( ui64DocumentID, &puzIndexName,
									&pIxd->uiIndexNum, &pIxd->uiCollectionNum,
									&pIxd->uiLanguage, &pIxd->uiFlags,
									&pIxd->ui64LastDocIndexed,
									&uiEncId,
									&pNode, bOpeningDict, bDeleting)))
	{
		goto Exit;
	}
	if (!uiIndexNum)
	{
		uiIndexNum = pIxd->uiIndexNum;
	}
	else
	{
		flmAssert( uiIndexNum == pIxd->uiIndexNum);
	}
	pIxd->ui64IxDefNodeId = ui64DocumentID;

	// Process each sub-element, setting up the path for the index
	// Start at the first child

	if (RC_BAD( rc = pNode->getFirstChild( pDb, (IF_DOMNode **)&pNode)))
	{

		// Change not-found to illegal index definition - must be at
		// least one subordinate node.

		if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			rc = RC_SET( NE_XFLM_ILLEGAL_INDEX_DEF);
		}
		goto Exit;
	}

	pLastIcd = NULL;
	bLinkAsChild = TRUE;
	bSinglePath = TRUE;

	for (;;)
	{
		if (RC_BAD( rc = isIndexComponent( pDb, pNode, &bIsIndexComponent,
									&uiElementId)))
		{
			goto Exit;
		}
		if (bIsIndexComponent)
		{

			// Allocate an ICD and link in

			if (RC_BAD( rc = m_dictPool.poolCalloc( sizeof( ICD),
													(void **)&pIcd)))
			{
				goto Exit;
			}
			pIcd->uiCdl = pIxd->uiNumIcds;
			pIxd->uiNumIcds++;
			pIcd->pIxd = pIxd;
			pIcd->uiIndexNum = pIxd->uiIndexNum;
			if (!pIxd->pIcdTree)
			{
				pIxd->pIcdTree = pIcd;
			}
			else if (bLinkAsChild)
			{

				// link as child

				pLastIcd->pFirstChild = pIcd;
				pIcd->pParent = pLastIcd;
				if (pLastIcd->pPrevSibling)
				{
					bSinglePath = FALSE;
				}
			}
			else
			{

				// link as sibling

				pLastIcd->pNextSibling = pIcd;
				if (pLastIcd->pFirstChild)
				{
					bSinglePath = FALSE;
				}
				pIcd->pPrevSibling = pLastIcd;
				pIcd->pParent = pLastIcd->pParent;
			}
			bLinkAsChild = FALSE;
			pLastIcd = pIcd;
			if (RC_BAD( rc = pDb->getIndexComponentDef( this, pNode,
										uiElementId, pIxd, pIcd)))
			{
				goto Exit;
			}

			// The ICD with a tag of ELM_ROOT_TAG cannot be
			// linked as a child to another node.  It always
			// has to be the lone root ICD.

			if (pIcd->uiDictNum == ELM_ROOT_TAG && 
				 (pIcd->pParent || pIcd->pNextSibling || pIcd->pPrevSibling))
			{
				rc = RC_SET( NE_XFLM_BAD_USE_OF_ELM_ROOT_TAG);
				goto Exit;
			}

			// Make sure that this ICD does not have the same
			// dictionary number as any prior sibling.

			pTmpIcd = pIcd->pPrevSibling;
			while (pTmpIcd)
			{
				if (pIcd->uiDictNum == pTmpIcd->uiDictNum &&
					 ((pIcd->uiFlags & ICD_IS_ATTRIBUTE) ==
					  (pTmpIcd->uiFlags & ICD_IS_ATTRIBUTE)))
				{
					rc = RC_SET( NE_XFLM_DUP_SIBLING_IX_COMPONENTS);
					goto Exit;
				}
				pTmpIcd = pTmpIcd->pPrevSibling;
			}

			// If the node was an attribute component, we are not interested in
			// anything that might be subordinate to it - only siblings.

			if (uiElementId == ELM_ATTRIBUTE_COMPONENT_TAG)
			{
				if ( RC_BAD( rc = pNode->hasChildren( pDb, &bHasChildren)))
				{
					goto Exit;
				}

				// Attribute components should not have children

				if ( bHasChildren)
				{
					rc = RC_SET( NE_XFLM_ILLEGAL_INDEX_DEF);
					goto Exit;
				}

				goto Get_Sibling;
			}

			// See if there is a child node

			if (RC_OK( rc = pNode->getFirstChild( pDb, (IF_DOMNode **)&pNode)))
			{
				bLinkAsChild = TRUE;
				continue;
			}
			if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				goto Exit;
			}
			rc = NE_XFLM_OK;

			// Fall through to see if there are any sibling nodes

		}

Get_Sibling:

		if (RC_OK( rc = pNode->getNextSibling( pDb, (IF_DOMNode **)&pNode)))
		{
			continue;
		}

		if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			goto Exit;
		}
		rc = NE_XFLM_OK;

		// No siblings, go to the parent node and see if it
		// has a sibling

		if (!pLastIcd || (pLastIcd = pLastIcd->pParent) == NULL)
		{
			break;
		}
		bLinkAsChild = FALSE;
		if (RC_BAD( rc = pNode->getParentNode( pDb, (IF_DOMNode **)&pNode)))
		{

			// Should not be not-found - because we came down
			// through a parent node.

			flmAssert( rc != NE_XFLM_DOM_NODE_NOT_FOUND);
			goto Exit;
		}

		goto Get_Sibling;
	}

	if (bSinglePath)
	{
		pIxd->uiFlags |= IXD_SINGLE_PATH;
	}

	// Look at all of the key components.  Verify that we have all of the
	// needed components and that none are missing.  If none are set to
	// ICD_REQUIRED_PIECE, set them all to ICD_REQUIRED_IN_SET

	pIcd = pIxd->pFirstKey;
	uiComponentNum = 1;
	bHadRequired = FALSE;
	while (pIcd)
	{
		if (pIcd->uiKeyComponent != uiComponentNum)
		{
			rc = RC_SET( NE_XFLM_MISSING_KEY_COMPONENT);
			goto Exit;
		}
		if (pIcd->uiFlags & ICD_REQUIRED_PIECE)
		{
			pIcd->uiFlags |= ICD_REQUIRED_IN_SET;
			bHadRequired = TRUE;
		}
		uiComponentNum++;
		pIcd = pIcd->pNextKeyComponent;
	}

	// If we don't have at least one key component, the index
	// definition is invalid

	if (!pIxd->uiNumKeyComponents)
	{
		rc = RC_SET( NE_XFLM_ILLEGAL_INDEX_DEF);
		goto Exit;
	}

	// If none of the key components were marked as required, mark
	// them all as ICD_REQUIRED_IN_SET.

	if (!bHadRequired)
	{
		pIcd = pIxd->pFirstKey;
		while (pIcd)
		{
			pIcd->uiFlags |= ICD_REQUIRED_IN_SET;
			pIcd = pIcd->pNextKeyComponent;
		}
	}

	// Look at all of the data components.  Verify that we have all of the
	// needed components and that none are missing.

	pIcd = pIxd->pFirstData;
	uiComponentNum = 1;
	while (pIcd)
	{
		if (pIcd->uiDataComponent != uiComponentNum)
		{
			rc = RC_SET( NE_XFLM_MISSING_DATA_COMPONENT);
			goto Exit;
		}
		uiComponentNum++;
		pIcd = pIcd->pNextDataComponent;
	}

	// Look at all of the leaf ICDs.  They must be data components or
	// key components.

	pIcd = pIxd->pFirstContext;
	while (pIcd)
	{

		// Context components cannot be leaf components - only data
		// and key components can be at the leaf.

		if (!pIcd->pFirstChild)
		{
			rc = RC_SET( NE_XFLM_ILLEGAL_INDEX_COMPONENT);
			goto Exit;
		}
		pIcd = pIcd->pNextKeyComponent;
	}

	// Add to the tag table, unless we already have our quota of
	// index names.  Remove the tag by number first, in case
	// it was renamed.

	if (!bOpeningDict)
	{
		m_pNameTable->removeTag( ELM_INDEX_TAG, pIxd->uiIndexNum);
	}

	if (RC_BAD( rc = m_pNameTable->addTag( ELM_INDEX_TAG, puzIndexName, NULL,
								pIxd->uiIndexNum, 0, NULL, 0,
								bOpeningDict ? FALSE : TRUE)))
	{
		goto Exit;
	}

	// If we are not opening the dictionary and the indexes are
	// the same, no need to change the index out

	if (!bOpeningDict && pOldIxd && indexDefsSame( pOldIxd, pIxd))
	{

		// Discard the new IXD, it is not needed.

		m_dictPool.poolReset( pvMark);
	}
	else
	{
		if (!bOpeningDict)
		{
			if (pOldIxd)
			{

				// If modifying make sure something is in the stop list.

				if (!(pDb->m_uiFlags & FDB_REPLAYING_RFL))

				{
					if( RC_BAD( rc = pDb->addToStopList( uiIndexNum)))
					{
						goto Exit;
					}
				}

				// Delete the old b-tree LFILE

				if (RC_BAD( rc = pDb->m_pDatabase->lFileDelete( pDb,
								NULL, &pOldIxd->lfInfo,
								(FLMBOOL)((pOldIxd->uiFlags & IXD_ABS_POS)
											 ? (FLMBOOL)TRUE
											 : (FLMBOOL)FALSE),
								(FLMBOOL)(pOldIxd->pFirstData
											 ? (FLMBOOL)TRUE
											 : (FLMBOOL)FALSE))))
				{
					goto Exit;
				}
			}

			// Create a NEW LFILE for the index.  If this is a new index
			// definition, we have not yet created one.  If this is a
			// modified index definition, the old LFILE would have been
			// deleted up above.

			if (RC_BAD( rc = pDb->m_pDatabase->lFileCreate( pDb, &pIxd->lfInfo,
										NULL, uiIndexNum, XFLM_LF_INDEX,
										(FLMBOOL)((pIxd->uiFlags & IXD_ABS_POS)
													 ? TRUE
													 : FALSE),
										(FLMBOOL)(pIxd->pFirstData
													 ? TRUE
													 : FALSE), 
										uiEncId)))
			{
				goto Exit;
			}
		}

		// Make room in the table for the new index if necessary.

		if (pIxd->uiIndexNum < m_uiLowestIxNum ||
			 pIxd->uiIndexNum > m_uiHighestIxNum)
		{
			if (RC_BAD( rc = reallocTbl( pIxd->uiIndexNum, sizeof( IXD *),
										(void **)&m_ppIxdTbl,
										&m_uiLowestIxNum,
										&m_uiHighestIxNum, 20,
										XFLM_MAX_INDEX_NUM)))
			{
				goto Exit;
			}
		}

		// Link the new ICDs into their ICD chains and unlink the old ICDs
		// from their ICD chains.  We don't want to do either of these until
		// we are sure we are going to succeed.

		m_ppIxdTbl [pIxd->uiIndexNum - m_uiLowestIxNum] = pIxd;
		if (RC_BAD( rc = linkIcds( pIxd->pIcdTree)))
		{
			goto Exit;
		}

		if (pOldIxd)
		{

			// Unlink the old ICDs.

			unlinkIcds( pOldIxd->pIcdTree);
		}

		// Build the index, unless we are just opening the dictionary

		if (!bOpeningDict)
		{
			if (RC_BAD( rc = pDb->buildIndex( uiIndexNum, pIxd->uiFlags)))
			{
				goto Exit;
			}
		}
	}

Exit:

	if (pNode)
	{
		pNode->Release();
	}

	if (puzIndexName)
	{
		f_free( &puzIndexName);
	}

	if (RC_BAD( rc))
	{
		m_dictPool.poolReset( pvMark);
	}

	return( rc);
}

/***************************************************************************
Desc:	Retrieve a collection definition - get name and number.
***************************************************************************/
RCODE F_Db::getCollectionDef(
	FLMUINT64			ui64DocumentID,
	FLMUNICODE **		ppuzCollectionName,
	FLMUINT *			puiCollectionNumber,
	FLMUINT *			puiEncId
	)
{
	RCODE					rc = NE_XFLM_OK;
	F_DOMNode *			pNode = NULL;
	F_CachedNode *		pCachedNode;
	IF_DOMNode *		pAttr = NULL;
	FLMBOOL				bHadCollectionNumber = FALSE;
	FLMBOOL				bHadCollectionName = FALSE;
	FLMUINT				uiNameId;

	// Set up defaults

	if (ppuzCollectionName)
	{
		*ppuzCollectionName = NULL;
	}
	if (puiCollectionNumber)
	{
		*puiCollectionNumber = 0;
	}
	if (puiEncId)
	{
		*puiEncId = 0;
	}

	// Retrieve the root element of the collection definition

	if (RC_BAD( rc = getNode( XFLM_DICT_COLLECTION, ui64DocumentID, &pNode)))
	{
		goto Exit;
	}
	flmAssert( pNode->getNodeType() == ELEMENT_NODE);
	
	pCachedNode = pNode->m_pCachedNode;
	
	if( !pCachedNode->hasAttributes())
	{
		rc = RC_SET( NE_XFLM_MISSING_COLLECTION_NUMBER);
		goto Exit;
	}
	
	if( RC_BAD( rc = pNode->getFirstAttribute( this, &pAttr)))
	{
		goto Exit;
	}

	for( ;;)
	{
		if( RC_BAD( rc = pAttr->getNameId( this, &uiNameId)))
		{
			goto Exit;
		}

		switch( uiNameId)
		{
			case ATTR_NAME_TAG:
				if (ppuzCollectionName)
				{
					if (RC_BAD( rc = pAttr->getUnicode( this, ppuzCollectionName)))
					{
						goto Exit;
					}
					if (RC_BAD( rc = fdictLegalCollectionName(
												*ppuzCollectionName)))
					{
						goto Exit;
					}
				}
				bHadCollectionName = TRUE;
				break;

			case ATTR_DICT_NUMBER_TAG:
				if (puiCollectionNumber)
				{
					if (RC_BAD( rc = pAttr->getUINT( this, puiCollectionNumber)))
					{
						goto Exit;
					}
					if (RC_BAD( rc = fdictLegalCollectionNumber(
												*puiCollectionNumber)))
					{
						goto Exit;
					}
				}
				bHadCollectionNumber = TRUE;
				break;

			case ATTR_ENCRYPTION_ID_TAG:
				if (puiEncId)
				{
					if (RC_BAD( rc = pAttr->getUINT( this, puiEncId)))
					{
						goto Exit;
					}
					if (RC_BAD( rc = fdictLegalEncDefNumber(
												*puiEncId)))
					{
						goto Exit;
					}
				}
				break;

			default:

				// Ignore all other attributes

				break;
		}
		
		if( RC_BAD( rc = pAttr->getNextSibling( this, &pAttr)))
		{
			if( rc != NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				goto Exit;
			}
			
			rc = NE_XFLM_OK;
			break;
		}
	}

	// Make sure we had both a name and number specified

	if (!bHadCollectionName)
	{
		rc = RC_SET( NE_XFLM_MISSING_COLLECTION_NAME);
		goto Exit;
	}
	if (!bHadCollectionNumber)
	{
		rc = RC_SET( NE_XFLM_MISSING_COLLECTION_NUMBER);
		goto Exit;
	}

Exit:

	if (pNode)
	{
		pNode->Release();
	}

	if (pAttr)
	{
		pAttr->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:	Parse a data dictionary collection definition.
***************************************************************************/
RCODE F_Dict::updateCollectionDef(
	F_Db *			pDb,
	FLMUINT64		ui64DocumentID,
	FLMUINT			uiCollectionNum,
	FLMBOOL			bOpeningDict,
	FLMBOOL			bDeleting
	)
{
	RCODE				rc = NE_XFLM_OK;
	F_COLLECTION *	pCollection;
	F_COLLECTION *	pOldCollection;
	FLMUNICODE *	puzCollectionName = NULL;
	FLMUINT			uiTmp;
	FLMUINT			uiEncId;

	if (bOpeningDict)
	{
		flmAssert( !bDeleting);
		pOldCollection = NULL;
	}
	else
	{
		if (RC_BAD( rc = getCollection( uiCollectionNum, &pOldCollection)))
		{
			if (rc == NE_XFLM_BAD_COLLECTION)
			{
				pOldCollection = NULL;
				rc = NE_XFLM_OK;
			}
			else
			{
				goto Exit;
			}
		}
	}

	if (bDeleting)
	{
		flmAssert( uiCollectionNum);

		if (pOldCollection)
		{
			if (RC_BAD( rc = pDb->m_pDatabase->lFileDelete( pDb,
					pOldCollection, &pOldCollection->lfInfo, FALSE, TRUE)))
			{
				goto Exit;
			}
		}
		
		pDb->removeCollectionNodes( uiCollectionNum,
										pDb->m_ui64CurrTransID);

		// NOTE: It is possible that the collection may not be in the
		// collection table yet, because dictDocumentDone had not been
		// called to put it in there, but we are calling dictDocumentDone
		// to remove it.

		// Remove the tag number from the name table

		m_pNameTable->removeTag( ELM_COLLECTION_TAG, uiCollectionNum);

		if (uiCollectionNum >= m_uiLowestCollectionNum &&
			 uiCollectionNum <= m_uiHighestCollectionNum)
		{
			m_ppCollectionTbl [uiCollectionNum - m_uiLowestCollectionNum] = NULL;
		}
		goto Exit;
	}

	// Allocate a new collection

	if (RC_BAD( rc = m_dictPool.poolCalloc( sizeof( F_COLLECTION),
								(void **)&pCollection)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pDb->getCollectionDef( ui64DocumentID,
								&puzCollectionName, &uiTmp, &uiEncId)))
	{
		goto Exit;
	}
	if (!uiCollectionNum)
	{
		uiCollectionNum = uiTmp;
	}
	else
	{
		flmAssert( uiCollectionNum == uiTmp);
	}

	if (!bOpeningDict)
	{
		// If this is not a new collection, get the LFILE info
		// for new collection we just allocated.

		if (pOldCollection)
		{
			f_memcpy( pCollection, pOldCollection, sizeof( F_COLLECTION));
		}
		else
		{
			flmAssert( !bDeleting);

			// Create a NEW LFILE for the collection

			if (RC_BAD( rc = pDb->m_pDatabase->lFileCreate( pDb,
										&pCollection->lfInfo,
										pCollection, uiCollectionNum,
										XFLM_LF_COLLECTION,
										FALSE, TRUE, uiEncId)))
			{
				goto Exit;
			}
		}
	}

	// Add to the tag table, unless we already have our quota of
	// collection names.  Remove the tag by number first, in case
	// it was renamed.

	if (!bOpeningDict)
	{
		m_pNameTable->removeTag( ELM_COLLECTION_TAG, uiCollectionNum);
	}

	if (RC_BAD( rc = m_pNameTable->addTag( ELM_COLLECTION_TAG, puzCollectionName,
								NULL, uiCollectionNum, 0, NULL, 0,
								bOpeningDict ? FALSE : TRUE)))
	{
		if (rc == NE_XFLM_EXISTS)
		{
			rc = NE_XFLM_OK;
		}
		else
		{
			goto Exit;
		}
	}

	if (uiCollectionNum < m_uiLowestCollectionNum ||
		 uiCollectionNum > m_uiHighestCollectionNum)
	{
		if (RC_BAD( rc = reallocTbl( uiCollectionNum, sizeof( F_COLLECTION *),
									(void **)&m_ppCollectionTbl,
									&m_uiLowestCollectionNum,
									&m_uiHighestCollectionNum, 20,
									XFLM_MAX_COLLECTION_NUM)))
		{
			goto Exit;
		}
	}
	m_ppCollectionTbl [uiCollectionNum - m_uiLowestCollectionNum] = pCollection;

Exit:

	if (puzCollectionName)
	{
		f_free( &puzCollectionName);
	}

	return( rc);
}

/***************************************************************************
Desc:	Retrieve a prefix definition - get name and number.
***************************************************************************/
RCODE F_Db::getPrefixDef(
	F_Dict *				pDict,
	FLMUINT64			ui64DocumentID,
	FLMUNICODE **		ppuzPrefixName,
	FLMUINT *			puiPrefixNumber)
{
	RCODE					rc = NE_XFLM_OK;
	F_DOMNode *			pNode = NULL;
	F_CachedNode *		pCachedNode;
	IF_DOMNode *		pAttr = NULL;
	FLMBOOL				bHadPrefixNumber = FALSE;
	FLMBOOL				bHadPrefixName = FALSE;
	FLMUINT				uiNameId;

	// Set up defaults

	if (ppuzPrefixName)
	{
		*ppuzPrefixName = NULL;
	}
	
	if (puiPrefixNumber)
	{
		*puiPrefixNumber = 0;
	}

	// Retrieve the root element of the prefix definition

	if (RC_BAD( rc = getNode( XFLM_DICT_COLLECTION, ui64DocumentID, &pNode)))
	{
		goto Exit;
	}
	
	flmAssert( pNode->getNodeType() == ELEMENT_NODE);
	pCachedNode = pNode->m_pCachedNode;
	
	if( !pCachedNode->hasAttributes())
	{
		rc = RC_SET( NE_XFLM_MISSING_PREFIX_NUMBER);
		goto Exit;
	}
	
	if( RC_BAD( rc = pNode->getFirstAttribute( this, &pAttr)))
	{
		goto Exit;
	}

	for( ;;)
	{
		if( RC_BAD( rc = pAttr->getNameId( this, &uiNameId)))
		{
			goto Exit;
		}

		switch( uiNameId)
		{
			case ATTR_NAME_TAG:
			{
				FLMUINT		uiPrefixLen;
				FLMUINT		uiBufferSize;

				if (ppuzPrefixName)
				{
					if( RC_BAD( rc = pAttr->getUnicodeChars( this, &uiPrefixLen)))
					{
						goto Exit;
					}

					uiBufferSize = sizeof( FLMUNICODE) * (uiPrefixLen + 1);
					if( RC_BAD( rc = pDict->m_dictPool.poolAlloc( uiBufferSize,
						(void **)ppuzPrefixName)))
					{
						goto Exit;
					}

					if (RC_BAD( rc = pAttr->getUnicode( this, *ppuzPrefixName,
						uiBufferSize, 0, uiPrefixLen, NULL)))
					{
						goto Exit;
					}

					if (RC_BAD( rc = fdictLegalPrefixName( *ppuzPrefixName)))
					{
						goto Exit;
					}
				}
				bHadPrefixName = TRUE;
				break;
			}

			case ATTR_DICT_NUMBER_TAG:
			{
				if (puiPrefixNumber)
				{
					if (RC_BAD( rc = pAttr->getUINT( this, puiPrefixNumber)))
					{
						goto Exit;
					}
					if (RC_BAD( rc = fdictLegalPrefixNumber(
												*puiPrefixNumber)))
					{
						goto Exit;
					}
				}
				bHadPrefixNumber = TRUE;
				break;
			}

			default:
			{
				// Ignore all other attributes

				break;
			}
		}
		
		if( RC_BAD( rc = pAttr->getNextSibling( this, (IF_DOMNode **)&pAttr)))
		{
			if( rc != NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				goto Exit;
			}
			
			rc = NE_XFLM_OK;
			break;
		}
	}

	// Make sure we had both a name and number specified

	if (!bHadPrefixName)
	{
		rc = RC_SET( NE_XFLM_MISSING_PREFIX_NAME);
		goto Exit;
	}
	if (!bHadPrefixNumber)
	{
		rc = RC_SET( NE_XFLM_MISSING_PREFIX_NUMBER);
		goto Exit;
	}

Exit:

	if (pNode)
	{
		pNode->Release();
	}

	if (pAttr)
	{
		pAttr->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:	Parse a data dictionary prefix definition.
***************************************************************************/
RCODE F_Dict::updatePrefixDef(
	F_Db *			pDb,
	FLMUINT64		ui64DocumentID,
	FLMUINT			uiPrefixNum,
	FLMBOOL			bOpeningDict,
	FLMBOOL			bDeleting)
{
	RCODE				rc = NE_XFLM_OK;
	F_PREFIX *		pPrefix = NULL;
	F_PREFIX *		pOldPrefix = NULL;
	FLMUNICODE *	puzPrefixName = NULL;
	FLMUINT			uiTmp;
	void *			pvMark = m_dictPool.poolMark();

	if (bOpeningDict)
	{
		flmAssert( !bDeleting);
		pOldPrefix = NULL;
	}
	else
	{
		if (RC_BAD( rc = getPrefix( uiPrefixNum, &pOldPrefix)))
		{
			if (rc == NE_XFLM_BAD_PREFIX)
			{
				pOldPrefix = NULL;
				rc = NE_XFLM_OK;
			}
			else
			{
				goto Exit;
			}
		}
	}

	if (bDeleting)
	{
		flmAssert( uiPrefixNum);

		// NOTE: It is possible that the prefix may not be in the
		// collection table yet, because dictDocumentDone had not been
		// called to put it in there, but we are calling dictDocumentDone
		// to remove it.

		// Remove the tag number from the name table

		m_pNameTable->removeTag( ELM_PREFIX_TAG, uiPrefixNum);

		if (uiPrefixNum >= m_uiLowestPrefixNum &&
			 uiPrefixNum <= m_uiHighestPrefixNum)
		{
			m_ppPrefixTbl [uiPrefixNum - m_uiLowestPrefixNum] = NULL;
		}
		goto Exit;
	}

	// Allocate a new prefix

	if (RC_BAD( rc = m_dictPool.poolCalloc( sizeof( F_PREFIX),
												(void **)&pPrefix)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pDb->getPrefixDef( this, ui64DocumentID,
								&puzPrefixName, &uiTmp)))
	{
		goto Exit;
	}

	if (!uiPrefixNum)
	{
		uiPrefixNum = uiTmp;
	}
	else
	{
		flmAssert( uiPrefixNum == uiTmp);
	}

	// Add to the tag table, unless we already have our quota of
	// prefix names.  Remove the tag by number first, in case
	// it was renamed.

	if (!bOpeningDict)
	{
		m_pNameTable->removeTag( ELM_PREFIX_TAG, uiPrefixNum);
	}

	if (RC_BAD( rc = m_pNameTable->addTag( ELM_PREFIX_TAG, puzPrefixName,
								NULL, uiPrefixNum, 0, NULL, 0,
								bOpeningDict ? FALSE : TRUE)))
	{
		goto Exit;
	}

	if (uiPrefixNum < m_uiLowestPrefixNum ||
		 uiPrefixNum > m_uiHighestPrefixNum)
	{
		if (RC_BAD( rc = reallocTbl( uiPrefixNum, sizeof( F_PREFIX *),
									(void **)&m_ppPrefixTbl,
									&m_uiLowestPrefixNum,
									&m_uiHighestPrefixNum, 20,
									XFLM_MAX_PREFIX_NUM)))
		{
			goto Exit;
		}
	}

	pPrefix->ui64PrefixId = uiPrefixNum;
	pPrefix->puzPrefixName = puzPrefixName;
	m_ppPrefixTbl [uiPrefixNum - m_uiLowestPrefixNum] = pPrefix;

Exit:

	if( RC_BAD( rc))
	{
		if( pvMark)
		{
			m_dictPool.poolReset( pvMark);
		}
	}

	return( rc );
}

/***************************************************************************
Desc:	Retrieve an encDef definition - get name and number and key
***************************************************************************/
RCODE F_Db::getEncDefDef(
	F_Dict *			pDict,
	FLMUINT64		ui64DocumentID,
	FLMUNICODE **	ppuzEncDefName,
	FLMUINT *		puiEncDefNumber,
	FLMUINT *		puiEncDefKeySize,
	IF_CCS **		ppCcs)
{
	RCODE				rc = NE_XFLM_OK;
	F_DOMNode *		pNode = NULL;
	F_CachedNode *	pCachedNode;
	F_DOMNode *		pAttr = NULL;
	FLMBOOL			bHadEncDefNumber = FALSE;
	FLMBOOL			bHadEncDefName = FALSE;
	FLMBOOL			bHadEncKey = FALSE;
	FLMBOOL			bHadEncKeySize = FALSE;
	FLMBOOL			bHadAlgorithm = FALSE;
	FLMUINT			uiNameId;
	void *			pvEncKeyBuf = NULL;
	FLMUINT32		ui32EncKeyLen;
	FLMBOOL			bStartedUpdateTrans = FALSE;
	FLMBOOL			bRestartReadTrans = FALSE;
	FLMBYTE *		pszAlgorithm = NULL;
	FLMUINT			uiEncType = 0;
	FLMUINT			uiEncKeySize = 0;

	// Set up defaults

	if (ppuzEncDefName)
	{
		*ppuzEncDefName = NULL;
	}
	
	if (puiEncDefNumber)
	{
		*puiEncDefNumber = 0;
	}
	
	flmAssert( ppCcs);
	
	if (*ppCcs)
	{
		(*ppCcs)->Release();
	}
	
	*ppCcs = NULL;

	// Retrieve the root element of the encDef definition

	if (RC_BAD( rc = getNode( XFLM_DICT_COLLECTION, ui64DocumentID, &pNode)))
	{
		goto Exit;
	}
	
	flmAssert( pNode->getNodeType() == ELEMENT_NODE);
	pCachedNode = pNode->m_pCachedNode;
	
	if( !pCachedNode->hasAttributes())
	{
		rc = RC_SET( NE_XFLM_MISSING_ENCDEF_NUMBER);
		goto Exit;
	}
	
	if( RC_BAD( rc = pNode->getFirstAttribute( this, (IF_DOMNode **)&pAttr)))
	{
		goto Exit;
	}

	// Cycle through the attributes

	for( ;;)
	{
		if( RC_BAD( rc = pAttr->getNameId( this, &uiNameId)))
		{
			goto Exit;
		}

		switch( uiNameId)
		{
			case ATTR_NAME_TAG:
			{
				FLMUINT		uiEncDefLen;
				FLMUINT		uiBufferSize;

				if (ppuzEncDefName)
				{
					if( RC_BAD( rc = pAttr->getUnicodeChars( this, &uiEncDefLen)))
					{
						goto Exit;
					}

					uiBufferSize = sizeof( FLMUNICODE) * (uiEncDefLen + 1);
					if( RC_BAD( rc = pDict->m_dictPool.poolAlloc( uiBufferSize,
						(void **)ppuzEncDefName)))
					{
						goto Exit;
					}

					if (RC_BAD( rc = pAttr->getUnicode( this, *ppuzEncDefName,
						uiBufferSize, 0, uiEncDefLen, NULL)))
					{
						goto Exit;
					}

					if (RC_BAD( rc = fdictLegalEncDefName( *ppuzEncDefName)))
					{
						goto Exit;
					}
				}
				bHadEncDefName = TRUE;
				break;
			}

			case ATTR_DICT_NUMBER_TAG:
			{
				if (puiEncDefNumber)
				{
					if (RC_BAD( rc = pAttr->getUINT( this, puiEncDefNumber)))
					{
						goto Exit;
					}
					if (RC_BAD( rc = fdictLegalEncDefNumber(
												*puiEncDefNumber)))
					{
						goto Exit;
					}
				}
				bHadEncDefNumber = TRUE;
				break;
			}

			case ATTR_ENCRYPTION_KEY_TAG:
			{
				FLMUINT			uiBytesReturned;

				if( RC_BAD( rc = pAttr->getDataLength( this, (FLMUINT *)&ui32EncKeyLen)))
				{
					goto Exit;
				}

				if( !ui32EncKeyLen)
				{
					rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
					goto Exit;
				}

				if( RC_BAD( rc = f_alloc( ui32EncKeyLen, &pvEncKeyBuf)))
				{
					goto Exit;
				}

				if (RC_BAD( rc = pAttr->getBinary( this, pvEncKeyBuf, 
					0, ui32EncKeyLen, &uiBytesReturned)))
				{
					goto Exit;
				}
				bHadEncKey = TRUE;
				break;
			}

			case ATTR_ENCRYPTION_KEY_SIZE_TAG:
			{
				if (RC_BAD( rc = pAttr->getUINT( this, &uiEncKeySize)))
				{
					goto Exit;
				}
				
				// Note:  Will validate the key size when we finish the loop.
				bHadEncKeySize = TRUE;
				break;
			}

			case ATTR_TYPE_TAG:
			{
				// Get the encryption Algorithm
				
			if( RC_BAD( rc = pAttr->getUTF8( this, &pszAlgorithm)))
			{
				goto Exit;
			}
				if (RC_BAD( rc = fdictLegalEncDefType( 
					(char *)pszAlgorithm, &uiEncType)))
				{
					goto Exit;
				}

				bHadAlgorithm = TRUE;
				break;
			}

			default:
			{
				// Ignore all other attributes

				break;
			}
		}
		
		if( RC_BAD( rc = pAttr->getNextSibling( this, (IF_DOMNode **)&pAttr)))
		{
			if( rc != NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				goto Exit;
			}
			
			rc = NE_XFLM_OK;
			break;
		}
	}

	// Make sure we had both a name and number specified

	if (!bHadEncDefName)
	{
		rc = RC_SET( NE_XFLM_MISSING_ENCDEF_NAME);
		goto Exit;
	}
	if (!bHadEncDefNumber)
	{
		rc = RC_SET( NE_XFLM_MISSING_ENCDEF_NUMBER);
		goto Exit;
	}
	
	if (bHadAlgorithm && bHadEncKeySize)
	{
		if (RC_BAD( rc = fdictLegalEncKeySize( uiEncType, uiEncKeySize)))
		{
			goto Exit;
		}
	}
	else if (!bHadAlgorithm)
	{
		rc = RC_SET( NE_XFLM_MISSING_ENC_ALGORITHM);
		goto Exit;
	}

	if( bHadEncKey)
	{
		if( RC_BAD( rc = flmAllocCCS( ppCcs)))
		{
			goto Exit;
		}
		
		if (RC_BAD( rc = (*ppCcs)->init( FALSE, uiEncType)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = (*ppCcs)->setKeyFromStore(
			(FLMBYTE *)pvEncKeyBuf, NULL, m_pDatabase->m_pWrappingKey)))
		{
			goto Exit;
		}
	}
	else
	{
		if( !bHadEncKeySize)
		{
			// Pick a key size based on the encryption algorithm.
			
			(void)fdictGetLegalKeySize( uiEncType, &uiEncKeySize);

		}

		// This must be a new encryption definition that doesn't have a key yet.
		// Generate a new encryption key and save it in the DOM node document.

		// We will need an update transaction before we can proceed.

		if( getTransType() == XFLM_READ_TRANS)
		{
			// End this transaction
			
			if( RC_BAD( rc = transCommit()))
			{
				goto Exit;
			}
			
			bRestartReadTrans = TRUE;
		}

		if( getTransType() == XFLM_NO_TRANS)
		{
			if( RC_BAD( rc = transBegin( XFLM_UPDATE_TRANS)))
			{
				goto Exit;
			}
			bStartedUpdateTrans = TRUE;
		}
		
		if( RC_BAD( rc = flmAllocCCS( ppCcs)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = (*ppCcs)->init( FALSE, uiEncType)))
		{
			goto Exit;
		}

TryNewKeySize:

		if( RC_BAD( rc = (*ppCcs)->generateEncryptionKey( uiEncKeySize)))
		{
			if( !fdictGetLegalKeySize( uiEncType, &uiEncKeySize))
			{
				goto Exit;
			}
			
			rc = NE_XFLM_OK;
			goto TryNewKeySize;
		}

		if( RC_BAD( rc = (*ppCcs)->getKeyToStore(
			(FLMBYTE **)&pvEncKeyBuf, &ui32EncKeyLen, NULL,
			m_pDatabase->m_pWrappingKey)))
		{
			goto Exit;
		}

		// Set the key in the DOM node as a binary string.
		
		if( RC_BAD( rc = pNode->createAttribute( this, ATTR_ENCRYPTION_KEY_TAG,
										(IF_DOMNode **)&pAttr)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pAttr->setBinary( this, pvEncKeyBuf, ui32EncKeyLen)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pAttr->addModeFlags( this, 
			FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
		{
			goto Exit;
		}

		if( !bHadEncKeySize)
		{
			// Set the key size
			
			if( RC_BAD( rc = pNode->createAttribute( this,
				ATTR_ENCRYPTION_KEY_SIZE_TAG, (IF_DOMNode **)&pAttr)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pAttr->setUINT( this, uiEncKeySize)))
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = pNode->getAttribute( this, 
				ATTR_ENCRYPTION_KEY_SIZE_TAG, (IF_DOMNode **)&pAttr)))
			{
				goto Exit;
			}
		}

		if( RC_BAD( rc = pAttr->addModeFlags( this,
			FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
		{
			goto Exit;
		}

		// End the transaction
		
		if( bStartedUpdateTrans)
		{
			if( RC_BAD( rc = transCommit()))
			{
				goto Exit;
			}
			
			bStartedUpdateTrans = FALSE;
		}
	}
	
	if( puiEncDefKeySize)
	{
		*puiEncDefKeySize = uiEncKeySize;
	}

Exit:

	if( RC_BAD( rc))
	{
		if( *ppCcs)
		{
			(*ppCcs)->Release();
			*ppCcs = NULL;
		}
	}

	if( bStartedUpdateTrans)
	{
		if( RC_OK( rc))
		{
			// Commit the update transaction
			
			if( RC_BAD( rc = transCommit()))
			{
				(void)transAbort();
			}
		}
		else
		{
			(void)transAbort();
		}
	}

	if( bRestartReadTrans)
	{
		rc = transBegin( XFLM_READ_TRANS);
	}

	if( pNode)
	{
		pNode->Release();
	}

	if( pAttr)
	{
		pAttr->Release();
	}

	if( pvEncKeyBuf)
	{
		f_free( &pvEncKeyBuf);
	}

	if( pszAlgorithm)
	{
		f_free( &pszAlgorithm);
	}

	return( rc);
}

/***************************************************************************
Desc:	Parse a data dictionary encryption definition.
***************************************************************************/
RCODE F_Dict::updateEncDef(
	F_Db *			pDb,
	FLMUINT64		ui64DocumentID,
	FLMUINT			uiEncDefNum,
	FLMBOOL			bOpeningDict,
	FLMBOOL			bDeleting)
{
	RCODE				rc = NE_XFLM_OK;
	F_ENCDEF *		pEncDef = NULL;
	F_ENCDEF *		pOldEncDef;
	FLMUNICODE *	puzEncDefName = NULL;
	FLMUINT			uiTmp;
	void *			pvMark = m_dictPool.poolMark();
	FLMUINT			uiEncKeySize = 0;
	IF_CCS *			pCcs = NULL;

	if (bOpeningDict)
	{
		flmAssert( !bDeleting);
		pOldEncDef = NULL;
	}
	else
	{
		if (RC_BAD( rc = getEncDef( uiEncDefNum, &pOldEncDef)))
		{
			if (rc == NE_XFLM_BAD_ENCDEF_NUM)
			{
				pOldEncDef = NULL;
				rc = NE_XFLM_OK;
			}
			else
			{
				goto Exit;
			}
		}
	}

	if (bDeleting)
	{
		flmAssert( uiEncDefNum);

		// NOTE: It is possible that the encdef may not be in the
		// collection table yet, because dictDocumentDone had not been
		// called to put it in there, but we are calling dictDocumentDone
		// to remove it.

		// Remove the tag number from the name table

		// VISIT:  Before we can delete the encryption definition tag, we need
		// to be sure it is not being used.  Take a look at the collection tag
		// to see how they are controlled.

		m_pNameTable->removeTag( ELM_ENCDEF_TAG, uiEncDefNum);

		if (uiEncDefNum >= m_uiLowestEncDefNum &&
			 uiEncDefNum <= m_uiHighestEncDefNum)
		{
			m_ppEncDefTbl [uiEncDefNum - m_uiLowestEncDefNum] = NULL;
		}
		goto Exit;
	}

	if (!pOldEncDef)
	{
		// Allocate a new encdef

		if (RC_BAD( rc = m_dictPool.poolCalloc( sizeof( F_ENCDEF),
													(void **)&pEncDef)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = pDb->getEncDefDef( this, ui64DocumentID,
									&puzEncDefName, &uiTmp, &uiEncKeySize, &pCcs)))
		{
			goto Exit;
		}

		if (!uiEncDefNum)
		{
			uiEncDefNum = uiTmp;
		}
		else
		{
			flmAssert( uiEncDefNum == uiTmp);
		}

		// Add to the tag table, unless we already have our quota of
		// encryption definition names.  Remove the tag by number first, in case
		// it was renamed.

		if (!bOpeningDict)
		{
			m_pNameTable->removeTag( ELM_ENCDEF_TAG, uiEncDefNum);
		}

		if (RC_BAD( rc = m_pNameTable->addTag( ELM_ENCDEF_TAG, puzEncDefName,
									NULL, uiEncDefNum, 0, NULL, 0,
									bOpeningDict ? FALSE : TRUE)))
		{
			goto Exit;
		}

		if (uiEncDefNum < m_uiLowestEncDefNum ||
			uiEncDefNum > m_uiHighestEncDefNum)
		{
			if (RC_BAD( rc = reallocTbl( uiEncDefNum, sizeof( F_ENCDEF *),
										(void **)&m_ppEncDefTbl,
										&m_uiLowestEncDefNum,
										&m_uiHighestEncDefNum, 20,
										XFLM_MAX_ENCDEF_NUM)))
			{
				goto Exit;
			}
		}

		flmAssert(m_ppEncDefTbl [uiEncDefNum - m_uiLowestEncDefNum] == NULL);

		pEncDef->ui64EncDefId = uiEncDefNum;
		pEncDef->ui64DocumentId = ui64DocumentID;
		pEncDef->puzEncDefName = puzEncDefName;
		pEncDef->uiEncKeySize = uiEncKeySize;
		pEncDef->pCcs = pCcs;
		pEncDef->pCcs->AddRef();
		m_ppEncDefTbl [uiEncDefNum - m_uiLowestEncDefNum] = pEncDef;
	}


Exit:

	if( RC_BAD( rc))
	{
		if( pvMark)
		{
			m_dictPool.poolReset( pvMark);
		}
	}

	if( pCcs)
	{
		pCcs->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:	Update a definition in the in-memory dictionary - add, modify, or
		delete it.
***************************************************************************/
RCODE F_Dict::updateDict(
	F_Db *		pDb,
	FLMUINT		uiDictType,
	FLMUINT64	ui64DocumentID,
	FLMUINT		uiDictNumber,
	FLMBOOL		bOpeningDict,
	FLMBOOL		bDeleting)
{
	RCODE			rc = NE_XFLM_OK;

#ifdef FLM_DEBUG
	if (!bOpeningDict)
	{
		flmAssert( pDb->m_uiFlags & FDB_UPDATED_DICTIONARY);
	}
#endif

	flmAssert( !pDb->m_pDatabase->m_pRfl || 
				  !pDb->m_pDatabase->m_pRfl->isLoggingEnabled() ||
				  pDb->getTransType() == XFLM_READ_TRANS);

	// Commit any keys in the KREF buffers.

	if (RC_BAD( rc = pDb->keysCommit( FALSE)))
	{
		goto Exit;
	}

	switch (uiDictType)
	{
		case ELM_ELEMENT_TAG:
			if (RC_BAD( rc = updateElementDef( pDb, ui64DocumentID,
									uiDictNumber, bOpeningDict, bDeleting)))
			{
				goto Exit;
			}
			break;

		case ELM_ATTRIBUTE_TAG:
			if (RC_BAD( rc = updateAttributeDef( pDb, ui64DocumentID,
									uiDictNumber, bOpeningDict, bDeleting)))
			{
				goto Exit;
			}
			break;

		case ELM_INDEX_TAG:
			if (RC_BAD( rc = updateIndexDef( pDb, ui64DocumentID,
									uiDictNumber, bOpeningDict, bDeleting)))
			{
				goto Exit;
			}
			break;

		case ELM_COLLECTION_TAG:
			if (RC_BAD( rc = updateCollectionDef( pDb,
									ui64DocumentID, uiDictNumber,
									bOpeningDict, bDeleting)))
			{
				goto Exit;
			}
			break;

		case ELM_PREFIX_TAG:
			if (RC_BAD( rc = updatePrefixDef( pDb,
									ui64DocumentID, uiDictNumber,
									bOpeningDict, bDeleting)))
			{
				goto Exit;
			}
			break;

		case ELM_ENCDEF_TAG:
			if (!pDb->m_pDatabase->m_bInLimitedMode)
			{
				if (RC_BAD( rc = updateEncDef( pDb,
										ui64DocumentID, uiDictNumber,
										bOpeningDict, bDeleting)))
				{
					goto Exit;
				}
			}
			break;

		default:

			// May be other things in the dictionary that we don't care
			// about

			break;

	}

	// Commit any keys in the KREF buffers.

	if (RC_BAD( rc = pDb->keysCommit( FALSE)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Setup the predefined collections and indexes.
****************************************************************************/
RCODE F_Dict::setupPredefined(
	FLMUINT				uiDefaultLanguage)
{
	RCODE					rc = NE_XFLM_OK;
	IXD *					pIxd;
	ICD *					pIcd;
	FLMUINT				uiLoop;
	ATTR_ELM_DEF *		pElementDef;
	ATTR_ELM_DEF *		pAttributeDef;

	// Set up reserved elements table

	if (RC_BAD( rc = f_calloc( (XFLM_LAST_RESERVED_ELEMENT_TAG -
										 XFLM_FIRST_RESERVED_ELEMENT_TAG + 1) *
										 sizeof( ATTR_ELM_DEF),
				&m_pReservedElementDefTbl)))
	{
		goto Exit;
	}

	// Set up reserved attributes table

	if (RC_BAD( rc = f_calloc( (XFLM_LAST_RESERVED_ATTRIBUTE_TAG -
										 XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 1) *
										 sizeof( ATTR_ELM_DEF),
				&m_pReservedAttributeDefTbl)))
	{
		goto Exit;
	}

	// Set the data types and state to active for all reserved elements and
	// attributes.

	for (uiLoop = 0; FlmReservedElementTags [uiLoop].pszTagName; uiLoop++)
	{
		pElementDef = &m_pReservedElementDefTbl [
								FlmReservedElementTags [uiLoop].uiTagNum -
											XFLM_FIRST_RESERVED_ELEMENT_TAG];
		pElementDef->uiFlags =
					(FlmReservedElementTags [uiLoop].uiDataType &
									 ATTR_ELM_DATA_TYPE_MASK) |
									(ATTR_ELM_STATE_ACTIVE & ATTR_ELM_STATE_MASK);
	}

	for (uiLoop = 0; FlmReservedAttributeTags [uiLoop].pszTagName; uiLoop++)
	{
		FLMUINT		uiTagNum = FlmReservedAttributeTags [uiLoop].uiTagNum;

		pAttributeDef = &m_pReservedAttributeDefTbl[ uiTagNum -
											XFLM_FIRST_RESERVED_ATTRIBUTE_TAG];
		pAttributeDef->uiFlags =
					(FlmReservedAttributeTags [uiLoop].uiDataType &
									 ATTR_ELM_DATA_TYPE_MASK) |
									(ATTR_ELM_STATE_ACTIVE & ATTR_ELM_STATE_MASK);

		// Special case for ATTR_XMLNS_XFLAIM_TAG and ATTR_XMLNS_TAG.
		// These attributes allow applications to use "xmlns:xflaim" or
		// "xmlns" in dictionary definitions without requiring the need
		// to explicitly create an attribute definition for the "xmlns:xflaim"
		// or "xmlns" attributes.

		if( uiTagNum == ATTR_XMLNS_XFLAIM_TAG || uiTagNum == ATTR_XMLNS_TAG)
		{
			pAttributeDef->uiFlags |= ATTR_ELM_NS_DECL;
		}
	}

	// Allocate memory for the predefined collections

	if (RC_BAD( rc = m_dictPool.poolCalloc( sizeof( F_COLLECTION) * 3,
											(void **)&m_pDictCollection)))
	{
		goto Exit;
	}

	m_pDataCollection = &m_pDictCollection[ 1];
	m_pMaintCollection = &m_pDictCollection[ 2];


	m_pDictCollection->lfInfo.uiLfNum = XFLM_DICT_COLLECTION;
	m_pDictCollection->lfInfo.eLfType = XFLM_LF_COLLECTION;

	m_pDataCollection->lfInfo.uiLfNum = XFLM_DATA_COLLECTION;
	m_pDataCollection->lfInfo.eLfType = XFLM_LF_COLLECTION;

	m_pMaintCollection->lfInfo.uiLfNum = XFLM_MAINT_COLLECTION;
	m_pMaintCollection->lfInfo.eLfType = XFLM_LF_COLLECTION;

	// Allocate IXDs for the predefined indexes.

	if (RC_BAD( rc = m_dictPool.poolCalloc( sizeof( IXD) * 2,
											(void **)&m_pNameIndex)))
	{
		goto Exit;
	}
	m_pNumberIndex = &m_pNameIndex [1];

	// Initialize the name index IXD

	pIxd = m_pNameIndex;
	pIxd->uiIndexNum = XFLM_DICT_NAME_INDEX;
	pIxd->uiCollectionNum = XFLM_DICT_COLLECTION;
//	pIxd->pIcdTree = NULL;					// Set by poolCalloc
// pIxd->pFirstKey = NULL;					// Set by poolCalloc
//	pIxd->pLastKey = NULL;					// Set by poolCalloc
// pIxd->pFirstContext = NULL;			// Set by poolCalloc
//	pIxd->pLastContext = NULL;				// Set by poolCalloc
// pIxd->pFirstData = NULL;				// Set by poolCalloc
// pIxd->pLastData = NULL;					// Set by poolCalloc
	pIxd->uiNumIcds = 4;
	pIxd->uiNumKeyComponents = 3;
	pIxd->uiNumDataComponents = 1;
//	pIxd->uiNumContextComponents = 0;	// Set by poolCalloc
	pIxd->uiFlags = IXD_SINGLE_PATH;
	pIxd->uiLanguage = uiDefaultLanguage;
	pIxd->ui64LastDocIndexed = ~((FLMUINT64)0);
	pIxd->lfInfo.uiLfNum = XFLM_DICT_NAME_INDEX;
	pIxd->lfInfo.eLfType = XFLM_LF_INDEX;

	// Initialize the number index IXD

	pIxd = m_pNumberIndex;
	pIxd->uiIndexNum = XFLM_DICT_NUMBER_INDEX;
	pIxd->uiCollectionNum = XFLM_DICT_COLLECTION;
//	pIxd->pIcdTree = NULL;					// Set by poolCalloc
// pIxd->pFirstKey = NULL;					// Set by poolCalloc
//	pIxd->pLastKey = NULL;					// Set by poolCalloc
// pIxd->pFirstContext = NULL;			// Set by poolCalloc
//	pIxd->pLastContext = NULL;				// Set by poolCalloc
// pIxd->pFirstData = NULL;				// Set by poolCalloc
// pIxd->pLastData = NULL;					// Set by poolCalloc
	pIxd->uiNumIcds = 2;
	pIxd->uiNumKeyComponents = 2;
//	pIxd->uiNumDataComponents = 0;		// Set by poolCalloc
//	pIxd->uiNumContextComponents = 0;	// Set by poolCalloc
	pIxd->uiFlags = IXD_SINGLE_PATH;
	pIxd->uiLanguage = uiDefaultLanguage;
	pIxd->ui64LastDocIndexed = ~((FLMUINT64)0);
	pIxd->lfInfo.uiLfNum = XFLM_DICT_NUMBER_INDEX;
	pIxd->lfInfo.eLfType = XFLM_LF_INDEX;

	// Set up the ICDs for the name index

	if (RC_BAD( rc = m_dictPool.poolCalloc( sizeof( ICD) * 4,
											(void **)&m_pNameIndex->pIcdTree)))
	{
		goto Exit;
	}

	pIcd = m_pNameIndex->pIcdTree;
	m_pNameIndex->pFirstKey = pIcd;
	pIcd->uiIndexNum = m_pNameIndex->uiIndexNum;
	pIcd->pIxd = m_pNameIndex;
	pIcd->uiDictNum = ELM_ROOT_TAG;
	pIcd->uiFlags = ICD_PRESENCE | ICD_REQUIRED_PIECE;
//	pIcd->uiCompareRules = 0;				// Set by poolCalloc
// pIcd->pParent = NULL;					// Set by poolCalloc
	pIcd->pFirstChild = pIcd + 1;
//	pIcd->pPrevSibling = NULL;				// Set by poolCalloc
//	pIcd->pNextSibling = NULL;				// Set by poolCalloc
//	pIcd->pPrevKeyComponent = NULL;		// Set by poolCalloc
	pIcd->pNextKeyComponent = pIcd + 1;
//	pIcd->uiCdl = 0;							// Set by poolCalloc
	pIcd->uiKeyComponent = 1;
//	pIcd->pPrevDataComponent = NULL;		// Set by poolCalloc
//	pIcd->pNextDataComponent = NULL;		// Set by poolCalloc
// pIcd->uiDataComponent = 0;				// Set by poolCalloc
// pIcd->uiLimit = 0;						// Set by poolCalloc
	icdSetDataType( pIcd, XFLM_NODATA_TYPE);

	pIcd++;
	pIcd->uiIndexNum = m_pNameIndex->uiIndexNum;
	pIcd->pIxd = m_pNameIndex;
	pIcd->uiDictNum = ATTR_NAME_TAG;
	pIcd->uiFlags = ICD_VALUE | ICD_REQUIRED_PIECE | ICD_IS_ATTRIBUTE;
//	pIcd->uiCompareRules = 0;				// Set by poolCalloc
	pIcd->pParent = m_pNameIndex->pIcdTree;
//	pIcd->pFirstChild = NULL;				// Set by poolCalloc
//	pIcd->pPrevSibling = NULL;				// Set by poolCalloc
	pIcd->pNextSibling = pIcd + 1;
	pIcd->pPrevKeyComponent = pIcd - 1;
	pIcd->pNextKeyComponent = pIcd + 1;
	pIcd->uiCdl = 1;
	pIcd->uiKeyComponent = 2;
//	pIcd->pPrevDataComponent = NULL;		// Set by poolCalloc
//	pIcd->pNextDataComponent = NULL;		// Set by poolCalloc
// pIcd->uiDataComponent = 0;				// Set by poolCalloc
// pIcd->uiLimit = 0;						// Set by poolCalloc
	icdSetDataType( pIcd, attrElmGetType(
		getReservedAttributeDef( pIcd->uiDictNum)));

	pIcd++;
	m_pNameIndex->pLastKey = pIcd;
	pIcd->uiIndexNum = m_pNameIndex->uiIndexNum;
	pIcd->pIxd = m_pNameIndex;
	pIcd->uiDictNum = ATTR_TARGET_NAMESPACE_TAG;
	pIcd->uiFlags = ICD_VALUE | ICD_IS_ATTRIBUTE;
//	pIcd->uiCompareRules = 0;				// Set by poolCalloc
	pIcd->pParent = m_pNameIndex->pIcdTree;
//	pIcd->pFirstChild = NULL;				// Set by poolCalloc
	pIcd->pPrevSibling = pIcd - 1;
	pIcd->pNextSibling = pIcd + 1;
	pIcd->pPrevKeyComponent = pIcd - 1;
//	pIcd->pNextKeyComponent = NULL;		// Set by poolCalloc
	pIcd->uiCdl = 2;
	pIcd->uiKeyComponent = 3;
//	pIcd->pPrevDataComponent = NULL;		// Set by poolCalloc
//	pIcd->pNextDataComponent = NULL;		// Set by poolCalloc
// pIcd->uiDataComponent = 0;				// Set by poolCalloc
// pIcd->uiLimit = 0;						// Set by poolCalloc
	icdSetDataType( pIcd, attrElmGetType(
		getReservedAttributeDef( pIcd->uiDictNum)));

	pIcd++;
	m_pNameIndex->pFirstData = pIcd;
	m_pNameIndex->pLastData = pIcd;
	pIcd->uiIndexNum = m_pNameIndex->uiIndexNum;
	pIcd->pIxd = m_pNameIndex;
	pIcd->uiDictNum = ATTR_DICT_NUMBER_TAG;
	pIcd->uiFlags = ICD_VALUE | ICD_IS_ATTRIBUTE;
//	pIcd->uiCompareRules = 0;				// Set by poolCalloc
	pIcd->pParent = m_pNameIndex->pIcdTree;
//	pIcd->pFirstChild = NULL;				// Set by poolCalloc
	pIcd->pPrevSibling = pIcd - 1;
//	pIcd->pNextSibling = NULL;				// Set by poolCalloc
//	pIcd->pPrevKeyComponent = NULL;		// Set by poolCalloc
//	pIcd->pNextKeyComponent = NULL;		// Set by poolCalloc
	pIcd->uiCdl = 3;
//	pIcd->uiKeyComponent = 0;				// Set by poolCalloc
//	pIcd->pPrevDataComponent = NULL;		// Set by poolCalloc
//	pIcd->pNextDataComponent = NULL;		// Set by poolCalloc
	pIcd->uiDataComponent = 1;
//	pIcd->uiLimit = 0;						// Set by poolCalloc
	icdSetDataType( pIcd, attrElmGetType(
		getReservedAttributeDef( pIcd->uiDictNum)));

	// Set up the ICDs for the number index

	if (RC_BAD( rc = m_dictPool.poolCalloc( sizeof( ICD) * 2,
											(void **)&m_pNumberIndex->pIcdTree)))
	{
		goto Exit;
	}

	pIcd = m_pNumberIndex->pIcdTree;
	pIcd->uiIndexNum = m_pNumberIndex->uiIndexNum;
	m_pNumberIndex->pFirstKey = pIcd;
	pIcd->pIxd = m_pNumberIndex;
	pIcd->uiDictNum = ELM_ROOT_TAG;
	pIcd->uiFlags = ICD_PRESENCE | ICD_REQUIRED_PIECE;
//	pIcd->uiCompareRules = 0;				// Set by poolCalloc
//	pIcd->pParent = NULL;					// Set by poolCalloc
	pIcd->pFirstChild = pIcd + 1;
//	pIcd->pPrevSibling = NULL;				// Set by poolCalloc
//	pIcd->pNextSibling = NULL;				// Set by poolCalloc
//	pIcd->pPrevKeyComponent = NULL;		// Set by poolCalloc
	pIcd->pNextKeyComponent = pIcd + 1;
//	pIcd->uiCdl = 0;							// Set by poolCalloc
	pIcd->uiKeyComponent = 1;
//	pIcd->pPrevDataComponent = NULL;		// Set by poolCalloc
//	pIcd->pNextDataComponent = NULL;		// Set by poolCalloc
//	pIcd->uiDataComponent = 0;				// Set by poolCalloc
//	pIcd->uiLimit = 0;						// Set by poolCalloc
	icdSetDataType( pIcd, XFLM_NODATA_TYPE);

	pIcd++;
	m_pNumberIndex->pLastKey = pIcd;
	pIcd->uiIndexNum = m_pNumberIndex->uiIndexNum;
	pIcd->pIxd = m_pNumberIndex;
	pIcd->uiDictNum = ATTR_DICT_NUMBER_TAG;
	pIcd->uiFlags = ICD_VALUE | ICD_REQUIRED_PIECE | ICD_IS_ATTRIBUTE;
//	pIcd->uiCompareRules = 0;				// Set by poolCalloc
	pIcd->pParent = m_pNumberIndex->pIcdTree;
//	pIcd->pFirstChild = NULL;				// Set by poolCalloc
//	pIcd->pPrevSibling = NULL;				// Set by poolCalloc
//	pIcd->pNextSibling = NULL;				// Set by poolCalloc
	pIcd->pPrevKeyComponent = pIcd - 1;
//	pIcd->pNextKeyComponent = NULL;		// Set by poolCalloc
	pIcd->uiCdl = 1;
	pIcd->uiKeyComponent = 2;
//	pIcd->pPrevDataComponent = NULL;		// Set by poolCalloc
//	pIcd->pNextDataComponent = NULL;		// Set by poolCalloc
//	pIcd->uiDataComponent = 0;				// Set by poolCalloc
//	pIcd->uiLimit = 0;						// Set by poolCalloc
	icdSetDataType( pIcd, attrElmGetType(
		getReservedAttributeDef( pIcd->uiDictNum)));

	if (RC_BAD( rc = linkIcds( m_pNameIndex->pIcdTree)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = linkIcds( m_pNumberIndex->pIcdTree)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Allocate an element table.
****************************************************************************/
RCODE F_Dict::allocElementTable(
	FLMUINT	uiLowestElementNum,
	FLMUINT	uiHighestElementNum
	)
{
	RCODE		rc = NE_XFLM_OK;
	FLMUINT	uiCount;

	// There should not already be an element table or an extended element
	// table.

	flmAssert( m_pElementDefTbl == NULL && m_pExtElementDefTbl == NULL &&
		m_hExtElementDefMutex == F_MUTEX_NULL);

	// No need for a fixed element table if we don't have any element
	// numbers in that range.

	if (uiHighestElementNum &&
		 uiLowestElementNum <= FLM_HIGH_FIXED_ELEMENT_NUM)
	{
		m_uiLowestElementNum = uiLowestElementNum;
		if (uiHighestElementNum > FLM_HIGH_FIXED_ELEMENT_NUM)
		{
			m_uiHighestElementNum = FLM_HIGH_FIXED_ELEMENT_NUM;
		}
		else
		{
			m_uiHighestElementNum = uiHighestElementNum;
		}
		uiCount = m_uiHighestElementNum - m_uiLowestElementNum + 1;
		if (RC_BAD( rc = f_calloc( uiCount * sizeof( ATTR_ELM_DEF),
					&m_pElementDefTbl)))
		{
			goto Exit;
		}
	}

	// See if we should allocate an extended element table

	if (uiHighestElementNum >= FLM_LOW_EXT_ELEMENT_NUM)
	{
		FLMUINT	uiNewSize = uiHighestElementNum %
									MAX_EXT_ATTR_ELM_ARRAY_SIZE + 1000;

		if (uiNewSize > MAX_EXT_ATTR_ELM_ARRAY_SIZE)
		{
			uiNewSize = MAX_EXT_ATTR_ELM_ARRAY_SIZE;
		}

		// Need to allocate a mutex too.

		if (RC_BAD( rc = f_mutexCreate( &m_hExtElementDefMutex)))
		{
			goto Exit;
		}

		// Allocate a new array

		if (RC_BAD( rc = f_calloc( sizeof( EXT_ATTR_ELM_DEF) * uiNewSize,
								&m_pExtElementDefTbl)))
		{
			goto Exit;
		}
		m_uiExtElementDefTblSize = uiNewSize;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Allocate an attribute table.
****************************************************************************/
RCODE F_Dict::allocAttributeTable(
	FLMUINT	uiLowestAttributeNum,
	FLMUINT	uiHighestAttributeNum
	)
{
	RCODE		rc = NE_XFLM_OK;
	FLMUINT	uiCount;

	// There should not already be an element table or an extended element
	// table.

	flmAssert( m_pAttributeDefTbl == NULL && m_pExtAttributeDefTbl == NULL &&
		m_hExtAttributeDefMutex == F_MUTEX_NULL);

	// No need for a fixed attribute table if we don't have any attribute
	// numbers in that range.

	if (uiHighestAttributeNum &&
		 uiLowestAttributeNum <= FLM_HIGH_FIXED_ATTRIBUTE_NUM)
	{
		m_uiLowestAttributeNum = uiLowestAttributeNum;
		if (uiHighestAttributeNum > FLM_HIGH_FIXED_ELEMENT_NUM)
		{
			m_uiHighestAttributeNum = FLM_HIGH_FIXED_ELEMENT_NUM;
		}
		else
		{
			m_uiHighestAttributeNum = uiHighestAttributeNum;
		}
		uiCount = m_uiHighestAttributeNum - m_uiLowestAttributeNum + 1;
		if (RC_BAD( rc = f_calloc( uiCount * sizeof( ATTR_ELM_DEF),
					&m_pAttributeDefTbl)))
		{
			goto Exit;
		}
	}

	// See if we should allocate an extended attribute table

	if (uiHighestAttributeNum >= FLM_LOW_EXT_ATTRIBUTE_NUM)
	{
		FLMUINT	uiNewSize = uiHighestAttributeNum %
									MAX_EXT_ATTR_ELM_ARRAY_SIZE + 1000;

		if (uiNewSize > MAX_EXT_ATTR_ELM_ARRAY_SIZE)
		{
			uiNewSize = MAX_EXT_ATTR_ELM_ARRAY_SIZE;
		}

		// Need to allocate a mutex too.

		if (RC_BAD( rc = f_mutexCreate( &m_hExtAttributeDefMutex)))
		{
			goto Exit;
		}

		// Allocate a new array

		if (RC_BAD( rc = f_calloc( sizeof( EXT_ATTR_ELM_DEF) * uiNewSize,
								&m_pExtAttributeDefTbl)))
		{
			goto Exit;
		}
		m_uiExtAttributeDefTblSize = uiNewSize;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Allocate an index table.
****************************************************************************/
RCODE F_Dict::allocIndexTable(
	FLMUINT	uiLowestIndexNum,
	FLMUINT	uiHighestIndexNum
	)
{
	RCODE		rc = NE_XFLM_OK;
	FLMUINT	uiCount;

	// There should not already be an index table.

	flmAssert( m_ppIxdTbl == NULL);

	m_uiLowestIxNum = uiLowestIndexNum;
	m_uiHighestIxNum = uiHighestIndexNum;
	if ((uiCount = getIndexCount( FALSE)) > 0)
	{
		if (RC_BAD( rc = f_calloc( uiCount * sizeof( IXD *), &m_ppIxdTbl)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Allocate a prefix table.
****************************************************************************/
RCODE F_Dict::allocPrefixTable(
	FLMUINT	uiLowestPrefixNum,
	FLMUINT	uiHighestPrefixNum
	)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiCount;

	flmAssert( m_ppPrefixTbl == NULL);

	m_uiLowestPrefixNum = uiLowestPrefixNum;
	m_uiHighestPrefixNum = uiHighestPrefixNum;
	if ((uiCount = getPrefixCount()) > 0)
	{
		if (RC_BAD( rc = f_calloc( uiCount * sizeof( F_PREFIX *), &m_ppPrefixTbl)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Allocate an encdef table.
****************************************************************************/
RCODE F_Dict::allocEncDefTable(
	FLMUINT	uiLowestEncDefNum,
	FLMUINT	uiHighestEncDefNum
	)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiCount;

	flmAssert( m_ppEncDefTbl == NULL);

	m_uiLowestEncDefNum = uiLowestEncDefNum;
	m_uiHighestEncDefNum = uiHighestEncDefNum;
	if ((uiCount = getEncDefCount()) > 0)
	{
		if (RC_BAD( rc = f_calloc( uiCount * sizeof( F_ENCDEF *), &m_ppEncDefTbl)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Allocate a collection table.
****************************************************************************/
RCODE F_Dict::allocCollectionTable(
	FLMUINT	uiLowestCollectionNum,
	FLMUINT	uiHighestCollectionNum
	)
{
	RCODE		rc = NE_XFLM_OK;
	FLMUINT	uiCount;

	// There should not already be a collection table.

	flmAssert( m_ppCollectionTbl == NULL);

	m_uiLowestCollectionNum = uiLowestCollectionNum;
	m_uiHighestCollectionNum = uiHighestCollectionNum;
	if ((uiCount = getCollectionCount( FALSE)) > 0)
	{
		if (RC_BAD( rc = f_calloc( uiCount * sizeof( LFILE *),
					&m_ppCollectionTbl)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Find the node IDs where we keep the next element, next attribute,
		next index, and next collection numbers.
****************************************************************************/
RCODE F_Dict::createNextDictNums(
	F_Db *			pDb)
{
	RCODE				rc = NE_XFLM_OK;
	F_DOMNode *		pNode = NULL;
	F_DOMNode *		pAttr = NULL;

	// Create a new root element

	if (RC_BAD( rc = pDb->createRootElement( XFLM_DICT_COLLECTION,
									ELM_NEXT_DICT_NUMS_TAG, (IF_DOMNode **)&pNode)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pNode->addModeFlags( pDb,
		FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
	{
		goto Exit;
	}

	// The NODE ID of this element had better be 1, because it is the
	// first thing we create in the dictionary.  Plus, the
	// getNextDictNumNodeIds method is counting on it being one!

	if( pNode->getNodeId() != XFLM_DICTINFO_DOC_ID)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
		goto Exit;
	}

	// Create and populate the four attributes that hold next element,
	// next attribute, next index, and next collection numbers.
	// Freeze each of the nodes so that they can only be modified by
	// FLAIM.

	// Node for next element number

	if (RC_BAD( rc = pNode->createAttribute( pDb,
								ATTR_NEXT_ELEMENT_NUM_TAG, (IF_DOMNode **)&pAttr)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pAttr->setUINT( pDb, 1)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pAttr->addModeFlags( pDb,
		FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
	{
		goto Exit;
	}

	// Node for next attribute number

	if (RC_BAD( rc = pNode->createAttribute( pDb,
								ATTR_NEXT_ATTRIBUTE_NUM_TAG, (IF_DOMNode **)&pAttr)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pAttr->setUINT( pDb, 1)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pAttr->addModeFlags( pDb,
		FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
	{
		goto Exit;
	}

	// Node for next index number

	if (RC_BAD( rc = pNode->createAttribute( pDb,
								ATTR_NEXT_INDEX_NUM_TAG, (IF_DOMNode **)&pAttr)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pAttr->setUINT( pDb, 1)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pAttr->addModeFlags( pDb,
		FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
	{
		goto Exit;
	}

	// Node for next collection number

	if (RC_BAD( rc = pNode->createAttribute( pDb,
								ATTR_NEXT_COLLECTION_NUM_TAG, (IF_DOMNode **)&pAttr)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pAttr->setUINT( pDb, 1)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pAttr->addModeFlags( pDb,
		FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
	{
		goto Exit;
	}

	// Node for next prefix number

	if (RC_BAD( rc = pNode->createAttribute( pDb,
								ATTR_NEXT_PREFIX_NUM_TAG, (IF_DOMNode **)&pAttr)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pAttr->setUINT( pDb, 1)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pAttr->addModeFlags( pDb,
		FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
	{
		goto Exit;
	}

	// Node for next encdef number

	if (RC_BAD( rc = pNode->createAttribute( pDb,
								ATTR_NEXT_ENCDEF_NUM_TAG, (IF_DOMNode **)&pAttr)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pAttr->setUINT( pDb, 1)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pAttr->addModeFlags( pDb,
		FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
	{
		goto Exit;
	}

Exit:

	if (pNode)
	{
		pNode->Release();
	}

	if (pAttr)
	{
		pAttr->Release();
	}
	return( rc);
}

/****************************************************************************
Desc:	Allocate the next dictionary number for a specific dictionary type.
****************************************************************************/
RCODE F_Dict::allocNextDictNum(
	F_Db *			pDb,
	FLMUINT			uiDictType,
	FLMUINT *		puiDictNumber)
{
	RCODE				rc = NE_XFLM_OK;
	F_DOMNode *		pDoc = NULL;
	F_DOMNode *		pAttr = NULL;
	FLMUINT			uiAttrName;
	FLMUINT			uiMaxNum;

	if( RC_BAD( rc = pDb->getNode( 
		XFLM_DICT_COLLECTION, XFLM_DICTINFO_DOC_ID, &pDoc)))
	{
		if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			rc = NE_XFLM_OK;
		}
		goto Exit;
	}
	
	if( pDoc->getNodeType() != ELEMENT_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
		goto Exit;
	}
	
	switch (uiDictType)
	{
		case ELM_ELEMENT_TAG:
			uiAttrName = ATTR_NEXT_ELEMENT_NUM_TAG;
			uiMaxNum = XFLM_MAX_ELEMENT_NUM;
			break;
		case ELM_ATTRIBUTE_TAG:
			uiAttrName = ATTR_NEXT_ATTRIBUTE_NUM_TAG;
			uiMaxNum = XFLM_MAX_ATTRIBUTE_NUM;
			break;
		case ELM_INDEX_TAG:
			uiAttrName = ATTR_NEXT_INDEX_NUM_TAG;
			uiMaxNum = XFLM_MAX_INDEX_NUM;
			break;
		case ELM_COLLECTION_TAG:
			uiAttrName = ATTR_NEXT_COLLECTION_NUM_TAG;
			uiMaxNum = XFLM_MAX_COLLECTION_NUM;
			break;
		case ELM_PREFIX_TAG:
			uiAttrName = ATTR_NEXT_PREFIX_NUM_TAG;
			uiMaxNum = XFLM_MAX_PREFIX_NUM;
			break;
		case ELM_ENCDEF_TAG:
			uiAttrName = ATTR_NEXT_ENCDEF_NUM_TAG;
			uiMaxNum = XFLM_MAX_ENCDEF_NUM;
			break;
		default:

			// Nothing to allocate for other types.  *puiDictNumber will
			// return as zero.

			*puiDictNumber = 0;
			goto Exit;
	}
	
	if( RC_BAD( rc = pDoc->getAttribute( 
		pDb, uiAttrName, (IF_DOMNode **)&pAttr)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pAttr->getUINT( pDb, puiDictNumber)))
	{
		goto Exit;
	}

	// Dictionary number better be > 0

	if( !(*puiDictNumber))
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
		goto Exit;
	}

	// See if we have exceeded the limit for this type.

	if( *puiDictNumber > uiMaxNum)
	{
		*puiDictNumber = 0;
		
		switch( uiDictType)
		{
			case ELM_ELEMENT_TAG:
				rc = RC_SET( NE_XFLM_NO_MORE_ELEMENT_NUMS);
				break;
				
			case ELM_ATTRIBUTE_TAG:
				rc = RC_SET( NE_XFLM_NO_MORE_ATTRIBUTE_NUMS);
				break;
				
			case ELM_INDEX_TAG:
				rc = RC_SET( NE_XFLM_NO_MORE_INDEX_NUMS);
				break;
				
			case ELM_COLLECTION_TAG:
				rc = RC_SET( NE_XFLM_NO_MORE_COLLECTION_NUMS);
				break;
				
			case ELM_PREFIX_TAG:
				rc = RC_SET( NE_XFLM_NO_MORE_PREFIX_NUMS);
				break;
				
			case ELM_ENCDEF_TAG:
				rc = RC_SET( NE_XFLM_NO_MORE_ENCDEF_NUMS);
		}
		
		goto Exit;
	}

	// Need to increment the dictionary number for the next
	// caller.

	if( RC_BAD( rc = pAttr->removeModeFlags( 
		pDb, FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pAttr->setUINT( pDb, *puiDictNumber + 1)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pAttr->addModeFlags( pDb,
		FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
	{
		goto Exit;
	}

Exit:

	if( pAttr)
	{
		pAttr->Release();
	}
	
	if( pDoc)
	{
		pDoc->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Check and set the next dictionary number for a specific dictionary type.
****************************************************************************/
RCODE F_Dict::setNextDictNum(
	F_Db *			pDb,
	FLMUINT			uiDictType,
	FLMUINT			uiDictNumber)
{
	RCODE				rc = NE_XFLM_OK;
	F_DOMNode *		pDoc = NULL;
	F_DOMNode *		pAttr = NULL;
	FLMUINT			uiAttrName;
	FLMUINT			uiCurrDictNumber;

	if( RC_BAD( rc = pDb->getNode( 
		XFLM_DICT_COLLECTION, XFLM_DICTINFO_DOC_ID, &pDoc)))
	{
		if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			rc = NE_XFLM_OK;
		}
		goto Exit;
	}
	
	if( pDoc->getNodeType() != ELEMENT_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
		goto Exit;
	}
	
	switch( uiDictType)
	{
		case ELM_ELEMENT_TAG:
		{
			if( uiDictNumber > XFLM_MAX_ELEMENT_NUM)
			{
				rc = RC_SET( NE_XFLM_BAD_ELEMENT_NUM);
				goto Exit;
			}
			
			uiAttrName = ATTR_NEXT_ELEMENT_NUM_TAG;
			break;
		}
		
		case ELM_ATTRIBUTE_TAG:
		{
			if (uiDictNumber > XFLM_MAX_ATTRIBUTE_NUM)
			{
				rc = RC_SET( NE_XFLM_BAD_ATTRIBUTE_NUM);
				goto Exit;
			}
			
			uiAttrName = ATTR_NEXT_ATTRIBUTE_NUM_TAG;
			break;
		}
		
		case ELM_INDEX_TAG:
		{
			if (uiDictNumber > XFLM_MAX_INDEX_NUM)
			{
				rc = RC_SET( NE_XFLM_BAD_IX);
				goto Exit;
			}
			
			uiAttrName = ATTR_NEXT_INDEX_NUM_TAG;
			break;
		}
		
		case ELM_COLLECTION_TAG:
		{
			if (uiDictNumber > XFLM_MAX_COLLECTION_NUM)
			{
				rc = RC_SET( NE_XFLM_BAD_COLLECTION);
				goto Exit;
			}
			
			uiAttrName = ATTR_NEXT_COLLECTION_NUM_TAG;
			break;
		}
		
		case ELM_PREFIX_TAG:
		{
			if (uiDictNumber > XFLM_MAX_PREFIX_NUM)
			{
				rc = RC_SET( NE_XFLM_BAD_PREFIX);
				goto Exit;
			}
			
			uiAttrName = ATTR_NEXT_PREFIX_NUM_TAG;
			break;
		}
		
		case ELM_ENCDEF_TAG:
		{
			if (uiDictNumber > XFLM_MAX_ENCDEF_NUM)
			{
				rc = RC_SET( NE_XFLM_BAD_ENCDEF_NUM);
				goto Exit;
			}
			
			uiAttrName = ATTR_NEXT_ENCDEF_NUM_TAG;
			break;
		}
		
		default:
		{
			// Doesn't really matter on other dictionary types
			// because dictionary number is not used for anything.

			goto Exit;
		}
	}
	
	if( RC_BAD( rc = pDoc->getAttribute( pDb, uiAttrName, 
		(IF_DOMNode **)&pAttr)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pAttr->getUINT( pDb, &uiCurrDictNumber)))
	{
		goto Exit;
	}

	// Dictionary number better be > 0

	if( !uiCurrDictNumber)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
		goto Exit;
	}

	// If the new dictionary number >= current next dictionary
	// number, need to set the next dictionary number to one greater
	// than the new dictionary number.

	if( uiDictNumber >= uiCurrDictNumber)
	{
		if( RC_BAD( rc = pAttr->removeModeFlags( 
			pDb, FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pAttr->setUINT( pDb, uiDictNumber + 1)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pAttr->addModeFlags( pDb,
			FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
		{
			goto Exit;
		}
	}

Exit:

	if( pAttr)
	{
		pAttr->Release();
	}
	
	if( pDoc)
	{
		pDoc->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	
****************************************************************************/
void F_AttrElmInfo::resetInfo( void)
{
	m_uiDictNum = 0;
	m_uiDataType = XFLM_NODATA_TYPE;
	m_uiFlags = 0;
	m_uiState = ATTR_ELM_STATE_ACTIVE;
	m_pFirstIcd = NULL;

	if( m_pDocNode)
	{
		m_pDocNode->Release();
		m_pDocNode = NULL;
	}

	if( m_pTargetNamespaceAttr)
	{
		m_pTargetNamespaceAttr->Release();
		m_pTargetNamespaceAttr = NULL;
	}

	if( m_pNameAttr)
	{
		m_pNameAttr->Release();
		m_pNameAttr = NULL;
	}
}

/****************************************************************************
Desc:	Read in LFH headers.
****************************************************************************/
RCODE F_Db::dictReadLFH( void)
{
	RCODE					rc = NE_XFLM_OK;
	LFILE *				pLFile;
	F_COLLECTION *		pCollection;
	F_CachedBlock *	pSCache = NULL;
	FLMBOOL				bReleaseCache = FALSE;
	F_BLK_HDR *			pBlkHdr;
	FLMUINT				uiBlkAddress;
	FLMUINT				uiPos;
	FLMUINT				uiEndPos;
	FLMUINT				uiBlkSize = m_pDatabase->m_uiBlockSize;
	LFILE					TmpLFile;
	F_COLLECTION		TmpCollection;

	f_memset( &TmpLFile, 0, sizeof( LFILE));
	f_memset( &TmpCollection, 0, sizeof( F_COLLECTION));

	uiBlkAddress =
			(FLMUINT)m_pDatabase->m_lastCommittedDbHdr.ui32FirstLFBlkAddr;
	while (uiBlkAddress)
	{
		if (RC_BAD( rc = m_pDatabase->getBlock( this, NULL, 
			uiBlkAddress, NULL, &pSCache)))
		{
			goto Exit;
		}
		bReleaseCache = TRUE;

		pBlkHdr = pSCache->m_pBlkHdr;
		uiPos = SIZEOF_STD_BLK_HDR;
		uiEndPos = blkGetEnd( uiBlkSize, SIZEOF_STD_BLK_HDR, pBlkHdr);

		// Read through all of the logical file definitions in the block

		for( ; uiPos + sizeof( F_LF_HDR) <= uiEndPos; uiPos += sizeof( F_LF_HDR))
		{
			F_LF_HDR *	pLfHdr = (F_LF_HDR *)((FLMBYTE *)(pBlkHdr) + uiPos);
			eLFileType	eLfType = (eLFileType)pLfHdr->ui32LfType;

			// Have to fix up the offsets later when they are read in

			if (eLfType == XFLM_LF_INVALID)
			{
				continue;
			}

			// Populate the LFILE in the dictionary, if one has been set up.

			if (eLfType == XFLM_LF_INDEX)
			{
				FSLFileIn( (FLMBYTE *)pLfHdr,
					&TmpLFile, NULL, uiBlkAddress, uiPos);

				if (RC_OK( m_pDict->getIndex( TmpLFile.uiLfNum, &pLFile,
											NULL, TRUE)))
				{
					f_memcpy( pLFile, &TmpLFile, sizeof( LFILE));
				}

				// LFILE better have a non-zero root block.

				if (!TmpLFile.uiRootBlk)
				{
					rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
					goto Exit;
				}
			}
			else
			{

				// Better be a container

				flmAssert( eLfType == XFLM_LF_COLLECTION);

				FSLFileIn( (FLMBYTE *)pLfHdr,
					&TmpCollection.lfInfo, &TmpCollection, uiBlkAddress, uiPos);

				if (RC_OK( m_pDict->getCollection( TmpCollection.lfInfo.uiLfNum,
												&pCollection, TRUE)))
				{
					f_memcpy( pCollection, &TmpCollection, sizeof( F_COLLECTION));
				}

				// LFILE better have a non-zero root block.

				if (!TmpCollection.lfInfo.uiRootBlk)
				{
					rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
					goto Exit;
				}
			}
		}

		// Get the next block in the chain

		uiBlkAddress = (FLMUINT)pBlkHdr->ui32NextBlkInChain;
		ScaReleaseCache( pSCache, FALSE);
		bReleaseCache = FALSE;
	}

Exit:

	if (bReleaseCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	return( rc );
}

/****************************************************************************
Desc:	Read in all element, attribute, index, or collection definitions - as
		specified in uiDictType.
****************************************************************************/
RCODE F_Db::dictReadDefs(
	FLMUINT	uiDictType)
{
	RCODE				rc = NE_XFLM_OK;
	F_DataVector	key;
	LFILE *			pLFile;
	IXD *				pIxd;
	F_Btree *		pbTree = NULL;
	FLMBYTE			ucKeyBuf [XFLM_MAX_KEY_SIZE];
	FLMUINT			uiKeyLen;
	FLMUINT			uiFoundDictType;
	FLMUINT			uiLowest;
	FLMUINT			uiHighest;
	FLMUINT			uiDictNum;
	IXKeyCompare	compareObject;

	if (RC_BAD( rc = m_pDict->getIndex( XFLM_DICT_NUMBER_INDEX, &pLFile, &pIxd)))
	{
		RC_UNEXPECTED_ASSERT( rc);
		goto Exit;
	}

	// First determine the low and high field numbers.

	// If the LFILE is not yet set up, the index has not yet been
	// created, so there will be no definitions to read.  This will
	// be the case when we are first creating the dictionary.  We have
	// started a transaction, and it is trying to read in the definitions
	// but there are none.

	flmAssert( pLFile->uiRootBlk);

	// Get a btree

	if (RC_BAD( rc = gv_XFlmSysData.pBtPool->btpReserveBtree( &pbTree)))
	{
		goto Exit;
	}

	// Open the B-Tree

	compareObject.setIxInfo( this, pIxd);
	compareObject.setCompareNodeIds( FALSE);
	compareObject.setCompareDocId( FALSE);
	compareObject.setSearchKey( &key);
	
	if (RC_BAD( rc = pbTree->btOpen( this, pLFile, FALSE, FALSE,
										&compareObject)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = key.setUINT( 0, uiDictType)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = key.outputKey( pIxd, 0,
									ucKeyBuf, sizeof( ucKeyBuf), &uiKeyLen, SEARCH_KEY_FLAG)))
	{
		goto Exit;
	}

	// Position to the first key, if any

	if (RC_BAD( rc = pbTree->btLocateEntry( ucKeyBuf, sizeof( ucKeyBuf),
										&uiKeyLen, XFLM_INCL, NULL)))
	{

		// May not have found anything.

		if (rc == NE_XFLM_EOF_HIT || rc == NE_XFLM_NOT_FOUND)
		{
			rc = NE_XFLM_OK;
		}
		goto Exit;
	}

	key.reset();

	if (RC_BAD( rc = key.inputKey( pIxd, ucKeyBuf, uiKeyLen)))
	{
		goto Exit;
	}

	// See if we went past the last key of this type.

	if (RC_BAD( rc = key.getUINT( 0, &uiFoundDictType)))
	{
		goto Exit;
	}

	if (uiFoundDictType != uiDictType)
	{
		goto Exit;	// Will return NE_XFLM_OK
	}

	if (RC_BAD( rc = key.getUINT( 1, &uiLowest)))
	{
		goto Exit;
	}
	uiHighest = uiLowest;

	// Position to the end of keys of this type

	key.reset();
	if (RC_BAD( rc = key.setUINT( 0, uiDictType)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = key.setUINT( 1, 0xFFFFFFFF)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = key.outputKey( pIxd, 0,
								ucKeyBuf, sizeof( ucKeyBuf), &uiKeyLen, SEARCH_KEY_FLAG)))
	{
		goto Exit;
	}

	// Position to just past the specified key.

	if (RC_BAD( rc = pbTree->btLocateEntry( ucKeyBuf, sizeof( ucKeyBuf),
										&uiKeyLen, XFLM_EXCL, NULL)))
	{

		// May not have found anything, in which case we need to
		// position to the last key in the index.

		if (rc == NE_XFLM_EOF_HIT || rc == NE_XFLM_NOT_FOUND)
		{
			if (RC_BAD( rc = pbTree->btLastEntry( ucKeyBuf, sizeof( ucKeyBuf),
													&uiKeyLen)))
			{
				goto Exit;
			}
		}
		else
		{
			goto Exit;
		}
	}
	else
	{

		// Backup one key - since we will have gone just beyond
		// keys of this type.

		if (RC_BAD( rc = pbTree->btPrevEntry( ucKeyBuf, sizeof( ucKeyBuf),
											&uiKeyLen)))
		{
			goto Exit;
		}
	}

	// At this point we better be positioned on the last key of this type

	key.reset();

	if (RC_BAD( rc = key.inputKey( pIxd, ucKeyBuf, uiKeyLen)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = key.getUINT( 0, &uiFoundDictType)))
	{
		goto Exit;
	}

	// See if we went past the last key of this type - should not
	// be possible, unless there is a corruption.

	if (uiFoundDictType != uiDictType)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_BTREE_ERROR);
		goto Exit;
	}
	
	if (RC_BAD( rc = key.getUINT( 1, &uiHighest)))
	{
		goto Exit;
	}

	// uiHighest better be >= uiLowest or we have
	// b-tree corruption.

	if (uiHighest < uiLowest)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_BTREE_ERROR);
		goto Exit;
	}

	// Pre-allocate the tables

	if (uiDictType == ELM_ELEMENT_TAG)
	{
		if (RC_BAD( rc = m_pDict->allocElementTable( uiLowest, uiHighest)))
		{
			goto Exit;
		}
	}
	else if (uiDictType == ELM_ATTRIBUTE_TAG)
	{
		if (RC_BAD( rc = m_pDict->allocAttributeTable( uiLowest, uiHighest)))
		{
			goto Exit;
		}
	}
	else if (uiDictType == ELM_INDEX_TAG)
	{
		if (RC_BAD( rc = m_pDict->allocIndexTable( uiLowest, uiHighest)))
		{
			goto Exit;
		}
	}
	else if (uiDictType == ELM_PREFIX_TAG)
	{
		if (RC_BAD( rc = m_pDict->allocPrefixTable( uiLowest, uiHighest)))
		{
			goto Exit;
		}
	}
	else if (uiDictType == ELM_ENCDEF_TAG)
	{
		if (RC_BAD( rc = m_pDict->allocEncDefTable( uiLowest, uiHighest)))
		{
			goto Exit;
		}
	}
	else	// (uiDictType == ELM_COLLECTION_TAG)
	{
		flmAssert( uiDictType == ELM_COLLECTION_TAG);

		if (RC_BAD( rc = m_pDict->allocCollectionTable( uiLowest, uiHighest)))
		{
			goto Exit;
		}
	}

	// Position back to the first key for this type

	key.reset();
	if (RC_BAD( rc = key.setUINT( 0, uiDictType)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = key.outputKey( pIxd, 0,
											ucKeyBuf, sizeof( ucKeyBuf),
											&uiKeyLen, SEARCH_KEY_FLAG)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pbTree->btLocateEntry( ucKeyBuf, sizeof( ucKeyBuf),
										&uiKeyLen, XFLM_INCL, NULL)))
	{

		// May not have found anything.

		if (rc == NE_XFLM_EOF_HIT || rc == NE_XFLM_NOT_FOUND)
		{
			rc = NE_XFLM_OK;
		}
		goto Exit;
	}

	// Loop through all of the keys of this dictionary type

	for (;;)
	{
		key.reset();

		if (RC_BAD( rc = key.inputKey( pIxd, ucKeyBuf, uiKeyLen)))
		{
			goto Exit;
		}

		// See if we went past the last key of this type.

		if (RC_BAD( rc = key.getUINT( 0, &uiFoundDictType)))
		{
			goto Exit;
		}

		if (uiFoundDictType != uiDictType)
		{
			break;
		}

		// Get the dictionary number

		if (RC_BAD( rc = key.getUINT( 1, &uiDictNum)))
		{
			goto Exit;
		}

		// No need to process any more elements or attributes if the
		// dictionary number is in the extended range.

		if ((uiDictType == ELM_ELEMENT_TAG &&
			  uiDictNum >= FLM_LOW_EXT_ELEMENT_NUM) ||
			 (uiDictType == ELM_ATTRIBUTE_TAG &&
			  uiDictNum >= FLM_LOW_EXT_ATTRIBUTE_NUM))
		{
			if (uiDictType == ELM_ELEMENT_TAG)
			{
				m_pDict->m_pNameTable->m_bLoadedAllElements = FALSE;
			}
			else
			{
				m_pDict->m_pNameTable->m_bLoadedAllAttributes = FALSE;
			}
			break;
		}

		if (RC_BAD( rc = m_pDict->updateDict( this,
									uiDictType, key.getDocumentID(), 0,
									TRUE, FALSE)))
		{
			goto Exit;
		}

		// Go to the next key

		if (RC_BAD( rc = pbTree->btNextEntry( ucKeyBuf,
											sizeof( ucKeyBuf),
											&uiKeyLen)))
		{

			// May not have found anything.

			if (rc == NE_XFLM_EOF_HIT || rc == NE_XFLM_NOT_FOUND)
			{
				rc = NE_XFLM_OK;
				break;
			}
			goto Exit;
		}
	}

Exit:

	if (pbTree)
	{
		gv_XFlmSysData.pBtPool->btpReturnBtree( &pbTree);
	}

	return( rc);
}

/****************************************************************************
Desc:	Open a dictionary by reading in all of the dictionary tables
		from the dictionaries.
****************************************************************************/
RCODE F_Db::dictOpen( void)
{
	RCODE	rc = NE_XFLM_OK;

	// At this point, better not be pointing to a dictionary.

	flmAssert( !m_pDict);

	// Should never get here for a temporary database.

	flmAssert( !m_pDatabase->m_bTempDb);

	// Allocate a new F_Dict object for reading the dictionary
	// into memory.

	if ((m_pDict = f_new F_Dict) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	
	// Allocate the name table

	if (RC_BAD( rc = m_pDict->allocNameTable()))
	{
		goto Exit;
	}

	// Add in all of the reserved dictionary tags to the name table.

	if (RC_BAD( rc = m_pDict->getNameTable()->addReservedDictTags()))
	{
		goto Exit;
	}

	// Allocate the fixed collections and indexes and set them up

	if (RC_BAD( rc = m_pDict->setupPredefined(
										m_pDatabase->m_uiDefaultLanguage)))
	{
		goto Exit;
	}

	// Read in the LFH's for the predefined stuff.

	if (RC_BAD( rc = dictReadLFH()))
	{
		goto Exit;
	}

	// If dictionary collection is not yet set up, do nothing.

	if (m_pDict->m_pDictCollection->lfInfo.uiBlkAddress &&
		 m_pDict->m_pDictCollection->lfInfo.uiOffsetInBlk)
	{

		// Read in definitions in the following order:
		// 1) attribute definitions
		// 2) element definitions
		// 3) collection definitions
		// 4) index definitions
		// This guarantees that things will be defined by the
		// time they are referenced.

		if (RC_BAD( rc = dictReadDefs( ELM_ATTRIBUTE_TAG)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = dictReadDefs( ELM_ELEMENT_TAG)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = dictReadDefs( ELM_COLLECTION_TAG)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = dictReadDefs( ELM_INDEX_TAG)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = dictReadDefs( ELM_PREFIX_TAG)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = dictReadDefs( ELM_ENCDEF_TAG)))
		{
			goto Exit;
		}

		// Must read LFHs to get the LFILE information for the
		// collections and indexes we have just added.

		if (RC_BAD( rc = dictReadLFH()))
		{
			goto Exit;
		}
	}

	m_pDict->getNameTable()->sortTags();
	
	if (m_pDatabase)
	{
		m_pDict->m_bInLimitedMode = m_pDatabase->inLimitedMode();
	}
	// VISIT:  Should we assume limited mode if we don't have a database file ?

Exit:

	if (RC_BAD( rc) && m_pDict)
	{
		m_pDict->Release();
		m_pDict = NULL;
	}

	return( rc);
}

/****************************************************************************
Desc:		Creates a new dictionary for a database.
			This occurs on database create and on a dictionary change.
****************************************************************************/
RCODE F_Db::createNewDict( void)
{
	RCODE	rc = NE_XFLM_OK;

	// Unlink the DB from the current F_Dict object, if any.

	if (m_pDict)
	{
		m_pDatabase->lockMutex();
		unlinkFromDict();
		m_pDatabase->unlockMutex();
	}

	// Allocate a new F_Dict object for the new dictionary we
	// are going to create.

	if (RC_BAD( rc = dictOpen()))
	{
		goto Exit;
	}

	// Update the F_Db flags to indicate that the dictionary
	// was updated.

	m_uiFlags |= FDB_UPDATED_DICTIONARY;

	// Create a special document in the dictionary to hold
	// the next element, next attribute, next index, and next
	// collection numbers.

	if (RC_BAD( rc = m_pDict->createNextDictNums( this)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Add data dictionary records to the data dictionary.
****************************************************************************/
RCODE F_Db::dictCreate(
	const char *	pszDictPath,	// Name of dictionary file.  This is only
											// used if dictBuf is NULL.  If both
											// dictPath and dictBuf are NULL, the
											// database will be created with an empty
											// dictionary
	const char *	pszDictBuf)		// Buffer containing dictionary in ASCII
											// GEDCOM If NULL pszDictPath will be used
{
	RCODE    				rc = NE_XFLM_OK;
	IF_FileHdl *			pDictFileHdl = NULL;
	FLMBOOL					bFileOpen = FALSE;
	LFILE						TempLFile;
	F_COLLECTION			TempCollection;
	char *					pszXMLBuffer = NULL;
	FLMUINT64				ui64FileSize;
	FLMUINT					uiBytesRead;
	IF_BufferIStream *	pStream = NULL;

	// This should never be called for a temporary database.

	flmAssert( !m_pDatabase->m_bTempDb);

	// Create the default data collection

	if (RC_BAD(rc = m_pDatabase->lFileCreate( this, &TempCollection.lfInfo,
		&TempCollection, XFLM_DATA_COLLECTION, XFLM_LF_COLLECTION, FALSE, TRUE)))
	{
		goto Exit;
	}

	// Create the dictionary collection and indexes

	if (RC_BAD(rc = m_pDatabase->lFileCreate( this, &TempCollection.lfInfo,
		&TempCollection,
		XFLM_DICT_COLLECTION, XFLM_LF_COLLECTION, FALSE, TRUE)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = m_pDatabase->lFileCreate( this,
		&TempLFile, NULL, XFLM_DICT_NUMBER_INDEX, XFLM_LF_INDEX, FALSE, FALSE)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = m_pDatabase->lFileCreate( this,
		&TempLFile, NULL, XFLM_DICT_NAME_INDEX, XFLM_LF_INDEX, FALSE, TRUE)))
	{
		goto Exit;
	}

	// Create the maintenance collection

	if (RC_BAD(rc = m_pDatabase->lFileCreate( this, &TempCollection.lfInfo,
		&TempCollection,
		XFLM_MAINT_COLLECTION, XFLM_LF_COLLECTION, FALSE, TRUE)))
	{
		goto Exit;
	}

	// Create a new dictionary we can work with.

	if (RC_BAD( rc = createNewDict()))
	{
		goto Exit;
	}

	// If we have an XML buffer, there is no need to open the file.

	if (!pszDictBuf && pszDictPath)
	{
		if (RC_BAD( rc = gv_XFlmSysData.pFileSystem->openFile(
				pszDictPath, FLM_IO_RDONLY, &pDictFileHdl)))
		{
			goto Exit;
		}
		bFileOpen = TRUE;

		// Get the file size and allocate a buffer to hold the entire thing.

		if (RC_BAD( rc = pDictFileHdl->size( &ui64FileSize)))
		{
			goto Exit;
		}

		// Add 1 to size so we can NULL terminate the string we read.

		if (RC_BAD( rc = f_alloc( (FLMUINT)(ui64FileSize + 1), &pszXMLBuffer)))
		{
			goto Exit;
		}

		// Read the entire file into the buffer

		if (RC_BAD( rc = pDictFileHdl->read( 0, (FLMUINT)ui64FileSize, 
			pszXMLBuffer, &uiBytesRead)))
		{
			goto Exit;
		}
		pszXMLBuffer [uiBytesRead] = 0;
		pszDictBuf = pszXMLBuffer;
	}
	if (!pszDictBuf || !(*pszDictBuf))
	{

		// Neither a dictionary buffer or file were specified.

		goto Exit;
	}

	// Parse through the buffer, extracting each XML document,
	// add to the dictionary and F_Dict object.  The import method
	// reads stuff from the stream, parses it into XML documents,
	// and calls documentDone when the document is complete.
	// The documentDone method checks the dictionary syntax,
	// adds to the dictionary, etc.
	
	if( RC_BAD( rc = FlmAllocBufferIStream( &pStream)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pStream->openStream( pszDictBuf, 0)))
	{
		goto Exit;
	}

	if (RC_BAD( import( pStream, XFLM_DICT_COLLECTION)))
	{
		goto Exit;
	}

	m_pDict->getNameTable()->sortTags();

Exit:

	if( pStream)
	{
		pStream->Release();
	}

	if( bFileOpen)
	{
		pDictFileHdl->closeFile();
	}

	if( pDictFileHdl)
	{
		pDictFileHdl->Release();
	}

	if( pszXMLBuffer)
	{
		f_free( pszXMLBuffer);
	}

	return( rc);
}
