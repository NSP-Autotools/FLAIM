//------------------------------------------------------------------------------
// Desc:	Routines to perform dictionary updates.
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

typedef struct FTravTag *	F_TRAV_p;
typedef struct FTravTag
{
	F_DOMNode *		pNode;
	ICD *				pRefIcd;
	ICD *				pIcd;
	IX_CONTEXT *	pIxContext;
	F_TRAV_p			pParent;
	F_TRAV_p			pChild;
} F_TRAV;

FSTATIC void kyFreeIxContext(
	IXD *				pIxd,
	IX_CONTEXT *	pIxContext,
	IX_CONTEXT **	ppIxContextList);

/****************************************************************************
Desc:	Check a dictionary definition for duplicate names or numbers.
		If no number was assigned, assign one.  If deleting, verify that the
		delete is allowed at this point.  Freeze certain attributes so
		they cannot be changed.
****************************************************************************/
RCODE F_Db::checkDictDefInfo(
	FLMUINT64			ui64DocumentID,
	FLMBOOL				bDeleting,
	FLMUINT *			puiDictType,
	FLMUINT *			puiDictNumber)
{
	RCODE					rc = NE_XFLM_OK;
	F_DOMNode *			pNode = NULL;
	F_DOMNode *			pAttr = NULL;
	F_DOMNode *			pTmpNode = NULL;
	FLMBYTE				szTmpBuf [80];
	FLMUINT				uiNameId;
	F_DataVector		key1;
	F_DataVector		key2;
	FLMUNICODE *		puzName = NULL;
	FLMUNICODE *		puzNamespace = NULL;
	FLMUINT				uiState = 0;
	FLMBOOL				bFoundState = FALSE;
	FLMUINT				uiMaxTagNum;
	FLMBOOL				bAmbiguous;
	FLMBOOL				bHasAttrs;
	FLMUINT				uiInsertPos;
	FLM_TAG_INFO * 	pTagInfo;

	*puiDictType = 0;
	*puiDictNumber = 0;

	// Get the root node of the definition document

	if (RC_BAD( rc = getNode( XFLM_DICT_COLLECTION, ui64DocumentID, &pNode)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pNode->getNameId( this, puiDictType)))
	{
		goto Exit;
	}

	// Ignore anything that is one one of our predefined types.

	switch (*puiDictType)
	{
		case ELM_ELEMENT_TAG:
			uiMaxTagNum = XFLM_MAX_ELEMENT_NUM;
			break;
		case ELM_ATTRIBUTE_TAG:
			uiMaxTagNum = XFLM_MAX_ATTRIBUTE_NUM;
			break;
		case ELM_INDEX_TAG:
			uiMaxTagNum = XFLM_MAX_INDEX_NUM;
			break;
		case ELM_PREFIX_TAG:
			uiMaxTagNum = XFLM_MAX_PREFIX_NUM;
			break;
		case ELM_COLLECTION_TAG:
			uiMaxTagNum = XFLM_MAX_COLLECTION_NUM;
			break;
		case ELM_ENCDEF_TAG:
#ifndef FLM_HAS_ENCRYPTION
			rc = RC_SET( NE_XFLM_ENCRYPTION_UNAVAILABLE);
			goto Exit;
#else
			uiMaxTagNum = XFLM_MAX_ENCDEF_NUM;
			break;
#endif
		default:
			*puiDictType = 0;
			goto Exit;
	}
	
	if( RC_BAD( rc = pNode->hasAttributes( this, &bHasAttrs)))
	{
		goto Exit;
	}
	
	if( !bHasAttrs)
	{
		goto Exit;
	}

	if( pNode->getNodeType() != ELEMENT_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
		goto Exit;
	}
	
	if( RC_BAD( rc = pNode->getFirstAttribute( this, (IF_DOMNode **)&pAttr)))
	{
		if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			rc = RC_SET( NE_XFLM_DATA_ERROR);
		}
		
		goto Exit;
	}
	
	// Cycle through the attributes, pulling out stuff we are interested
	// in.

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
				if (RC_BAD( rc = pAttr->getUnicode( this, &puzName)))
				{
					goto Exit;
				}

				// Put a freeze on name if we are not deleting.
				// Modify is allowed, but delete is not.

				if (!bDeleting)
				{
					if (RC_BAD( rc = pAttr->addModeFlags( 
						this, FDOM_CANNOT_DELETE)))
					{
						goto Exit;
					}
				}
				break;
			}

			case ATTR_TARGET_NAMESPACE_TAG:
			{
				if (RC_BAD( rc = pAttr->getUnicode( this, &puzNamespace)))
				{
					goto Exit;
				}
				break;
			}

			case ATTR_TYPE_TAG:
			{
				// Put a freeze on data type if we are not deleting.

				if (!bDeleting)
				{
					if( RC_BAD( rc = pAttr->addModeFlags( this,
						FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
					{
						goto Exit;
					}
				}
				break;
			}

			case ATTR_STATE_TAG:
			{
				if (RC_BAD( rc = pAttr->getUTF8( this, szTmpBuf,
					sizeof( szTmpBuf), 0, ~((FLMUINT)0))))
				{
					goto Exit;
				}

				if (*puiDictType == ELM_INDEX_TAG)
				{
					if (RC_BAD( rc = fdictGetIndexState( 
						(char *)szTmpBuf, &uiState)))
					{
						goto Exit;
					}
				}
				else if (*puiDictType == ELM_ELEMENT_TAG ||
							*puiDictType == ELM_ATTRIBUTE_TAG)
				{
					if (RC_BAD( rc = fdictGetState( 
						(char *)szTmpBuf, &uiState)))
					{
						goto Exit;
					}
				}

				// Put a freeze on state if we are not deleting.

				if (!bDeleting &&
					 (*puiDictType == ELM_INDEX_TAG ||
					  *puiDictType == ELM_ELEMENT_TAG ||
					  *puiDictType == ELM_ATTRIBUTE_TAG))
				{
					if( RC_BAD( rc = pAttr->addModeFlags( this,
						FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
					{
						goto Exit;
					}
				}

				bFoundState = TRUE;
				break;
			}
			
			case ATTR_STATE_CHANGE_COUNT_TAG:
			{
				// Put a freeze on state change count if we are not deleting.

				if (!bDeleting &&
					 (*puiDictType == ELM_ELEMENT_TAG ||
					  *puiDictType == ELM_ATTRIBUTE_TAG))
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
				if (RC_BAD( rc = pAttr->getUINT( this, puiDictNumber)))
				{
					goto Exit;
				}

				// If we are deleting, no need to verify or alter
				// dictionary number.

				if (bDeleting)
				{
					break;
				}

				// If the set dictionary number was zero, allocate a new
				// one, set it, and freeze it.

				if (!(*puiDictNumber))
				{
					if (RC_BAD( rc = m_pDict->allocNextDictNum( this, *puiDictType,
											puiDictNumber)))
					{
						goto Exit;
					}

					// *puiDictNumber will be zero coming back if the
					// dictionary type does not keep track of a next
					// dictionary number (like for prefixes)

					if (*puiDictNumber)
					{
						if( RC_BAD( rc = pAttr->removeModeFlags( this,
							FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
						{
							goto Exit;
						}

						if (RC_BAD( rc = pAttr->setUINT( this, *puiDictNumber)))
						{
							goto Exit;
						}

						if( RC_BAD( rc = pAttr->addModeFlags( this,
							FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
						{
							goto Exit;
						}

					}
				}
				else
				{
					// Set the next unused dictionary number for this type - but only
					// if this number is >= the one already stored.

					if (RC_BAD( rc = m_pDict->setNextDictNum( this, *puiDictType,
											*puiDictNumber)))
					{
						goto Exit;
					}
				}
				break;
			}
			
			default:
			{
				// Ignore all other attributes for now.

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

	// If dictionary number is missing, create one and add it in.

	if (!(*puiDictNumber) && !bDeleting)
	{

		// Allocate the next unused dictionary number for this type.

		if (RC_BAD( rc = m_pDict->allocNextDictNum( this, *puiDictType,
								puiDictNumber)))
		{
			goto Exit;
		}

		// *puiDictNumber will be zero coming back if the
		// dictionary type does not keep track of a next
		// dictionary number for this type (like for prefixes)

		if (*puiDictNumber)
		{

			// Create a dict number attribute, set it to the newly
			// allocated value, and freeze it.

			if (RC_OK( rc = pNode->createAttribute( this, ATTR_DICT_NUMBER_TAG,
											(IF_DOMNode **)&pAttr)))
			{
				if (RC_OK( rc = pAttr->setUINT( this, *puiDictNumber)))
				{
					if( RC_BAD( rc = pAttr->addModeFlags( this,
						FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
					{
						goto Exit;
					}
				}
			}
		}
	}

	if (!bDeleting)
	{

		// Must have a name specified, and it must be unique
	
		if (!puzName)
		{
			switch (*puiDictType)
			{
				case ELM_ELEMENT_TAG:
					rc = RC_SET( NE_XFLM_MISSING_ELEMENT_NAME);
					break;
				case ELM_ATTRIBUTE_TAG:
					rc = RC_SET( NE_XFLM_MISSING_ATTRIBUTE_NAME);
					break;
				case ELM_INDEX_TAG:
					rc = RC_SET( NE_XFLM_MISSING_INDEX_NAME);
					break;
				case ELM_PREFIX_TAG:
					rc = RC_SET( NE_XFLM_MISSING_PREFIX_NAME);
					break;
				case ELM_COLLECTION_TAG:
					rc = RC_SET( NE_XFLM_MISSING_COLLECTION_NAME);
					break;
				case ELM_ENCDEF_TAG:
					rc = RC_SET( NE_XFLM_MISSING_ENCDEF_NAME);
					break;
				default:
				
					// Should never hit this case!
					
					flmAssert( 0);
					break;
			}
			goto Exit;	// Will return NE_XFLM_OK
		}
	
		// Verify name uniqueness
		
		key1.reset();
		if (RC_BAD( rc = key1.setUINT( 0, *puiDictType)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = key1.setUnicode( 1, puzName)))
		{
			goto Exit;
		}
		if ((*puiDictType == ELM_ELEMENT_TAG ||
			  *puiDictType == ELM_ATTRIBUTE_TAG) &&
			 puzNamespace)
		{
			if (RC_BAD( rc = key1.setUnicode( 2, puzNamespace)))
			{
				goto Exit;
			}
		}

		if (RC_BAD( rc = keyRetrieve( XFLM_DICT_NAME_INDEX,
					&key1, XFLM_EXACT, &key2)))
		{
			if (rc == NE_XFLM_NOT_FOUND)
			{
				// We should have found the thing!  It should have
				// already been indexed!

				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			}
			
			goto Exit;
		}

		// See if there is another one after this for the same
		// key.

		if (RC_BAD( rc = keyRetrieve( XFLM_DICT_NAME_INDEX,
										&key2, XFLM_EXCL | XFLM_KEY_EXACT | XFLM_MATCH_IDS,
										&key1)))
		{
			if (rc != NE_XFLM_EOF_HIT)
			{
				goto Exit;
			}
				
			rc = NE_XFLM_OK;
				
			// See if it is defined in our name table and is a number in
			// our reserved range.  In that case, it should never be
			// coming through here - it is a name conflict we should prevent.
				
			if ((pTagInfo = m_pDict->getNameTable()->findTagByTypeAndName(
									*puiDictType, puzName, NULL, TRUE,
									puzNamespace, &bAmbiguous, &uiInsertPos)) != NULL)
			{
				if (pTagInfo->uiTagNum > uiMaxTagNum)
				{
					goto Have_Name_Conflict;
				}
			}
		}
		else
		{
Have_Name_Conflict:

			// We should NOT have found another one with this same name

			switch (*puiDictType)
			{
				case ELM_ELEMENT_TAG:
					rc = RC_SET( NE_XFLM_DUPLICATE_ELEMENT_NAME);
					goto Exit;
				case ELM_ATTRIBUTE_TAG:
					rc = RC_SET( NE_XFLM_DUPLICATE_ATTRIBUTE_NAME);
					goto Exit;
				case ELM_INDEX_TAG:
					rc = RC_SET( NE_XFLM_DUPLICATE_INDEX_NAME);
					goto Exit;
				case ELM_COLLECTION_TAG:
					rc = RC_SET( NE_XFLM_DUPLICATE_COLLECTION_NAME);
					goto Exit;
				case ELM_PREFIX_TAG:
					rc = RC_SET( NE_XFLM_DUPLICATE_PREFIX_NAME);
					goto Exit;
				default:

					// VISIT: Do we care on other dictionary types that
					// we have a duplicate name?

					break;
			}
		}
		
		// If this is an attribute definition, and the name is "xmlns" or
		// begins with "xmlns:", it cannot have a target namespace.
		
		if (*puiDictType == ELM_ATTRIBUTE_TAG &&
			 puzNamespace && *puzNamespace &&
			 isXMLNS( puzName) &&
			 (puzName [5] == 0 || (puzName [5] == ':' && puzName [6])))
		{
			rc = RC_SET( NE_XFLM_NAMESPACE_NOT_ALLOWED);
			goto Exit;
		}
	}

	// Verify dictionary number uniqueness

	if (*puiDictNumber && !bDeleting)
	{
		key1.reset();
		key2.reset();
		if (RC_BAD( rc = key1.setUINT( 0, *puiDictType)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = key1.setUINT( 1, *puiDictNumber)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = keyRetrieve( XFLM_DICT_NUMBER_INDEX,
					&key1, XFLM_EXACT, &key2)))
		{
			if (rc == NE_XFLM_NOT_FOUND)
			{
				// We should have found the thing!  It should have
				// already been indexed!

				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			}
			goto Exit;
		}

		// See if there is another one after this for the same
		// key.

		if (RC_BAD( rc = keyRetrieve( XFLM_DICT_NUMBER_INDEX,
										&key2, XFLM_EXCL | XFLM_KEY_EXACT | XFLM_MATCH_IDS,
										&key1)))
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

			// We should NOT have found another one with this same number

			switch (*puiDictType)
			{
				case ELM_ELEMENT_TAG:
					rc = RC_SET( NE_XFLM_DUPLICATE_ELEMENT_NUM);
					goto Exit;
				case ELM_ATTRIBUTE_TAG:
					rc = RC_SET( NE_XFLM_DUPLICATE_ATTRIBUTE_NUM);
					goto Exit;
				case ELM_INDEX_TAG:
					rc = RC_SET( NE_XFLM_DUPLICATE_INDEX_NUM);
					goto Exit;
				case ELM_COLLECTION_TAG:
					rc = RC_SET( NE_XFLM_DUPLICATE_COLLECTION_NUM);
					goto Exit;
				default:

					// Dictionary number is not used for other types
					// so we really don't care if there is a duplicate
					// in these cases.

					break;
			}
		}
	}

	// Is it a delete?

	if (bDeleting)
	{
		if (*puiDictType == ELM_ELEMENT_TAG)
		{
			if (*puiDictNumber)
			{
				// Make sure that elements are in the right state
				// to be deleted.

				if( !m_bItemStateUpdOk)
				{
					rc = RC_SET( NE_XFLM_CANNOT_DEL_ELEMENT);
					goto Exit;
				}
			}
		}
		else if (*puiDictType == ELM_ATTRIBUTE_TAG)
		{
			if (*puiDictNumber)
			{
				// Make sure that attributes are in the right state
				// to be deleted.

				if( !m_bItemStateUpdOk)
				{
					rc = RC_SET( NE_XFLM_CANNOT_DEL_ATTRIBUTE);
					goto Exit;
				}
			}
		}
		else if (*puiDictType == ELM_COLLECTION_TAG)
		{
			if (*puiDictNumber)
			{
				if (RC_BAD( rc = m_pDict->checkCollectionReferences( *puiDictNumber)))
				{
					goto Exit;
				}
			}
		}
	}
	else
	{
		// Set a state attribute and freeze it if it was missing.
		// This makes it so that the state can only be changed by a call
		// to changeItemState.  This routine ensures that the state
		// change is legal.

		if( (*puiDictType == ELM_ATTRIBUTE_TAG ||
			  *puiDictType == ELM_ELEMENT_TAG ||
			  *puiDictType == ELM_ENCDEF_TAG) && !bFoundState)
		{
			if (RC_BAD( rc = pNode->createAttribute( 
				this, ATTR_STATE_TAG, (IF_DOMNode **)&pTmpNode)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = pTmpNode->setUTF8( 
				this, (FLMBYTE *)XFLM_ACTIVE_OPTION_STR)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pTmpNode->addModeFlags( this,
				FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
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

	if (pAttr)
	{
		pAttr->Release();
	}

	if (pTmpNode)
	{
		pTmpNode->Release();
	}

	if (puzName)
	{
		f_free( &puzName);
	}

	if (puzNamespace)
	{
		f_free( &puzNamespace);
	}

	return( rc);
}

/****************************************************************************
Desc:	This routine is called when a dictionary document is done being
		modified.  It will check the definition and then generate any
		dictionary updates that need to be done.
****************************************************************************/
RCODE F_Db::dictDocumentDone(
	FLMUINT64	ui64DocumentID,
	FLMBOOL		bDeleting,
	FLMUINT *	puiDictDefType)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiDictType;
	FLMUINT		uiDictNumber;

	if (puiDictDefType)
	{
		*puiDictDefType = 0;
	}

	// Document ID 1 is the one that is reserved for next element, next
	// attribute, next index, and next collection.

	if (ui64DocumentID == XFLM_DICTINFO_DOC_ID)
	{
		goto Exit;
	}

	// Flush any index keys before making any changes to dictionary items.
	// Dictionary changes may change index definitions, etc.

	if( RC_BAD( rc = keysCommit( FALSE)))
	{
		goto Exit;
	}

	// Clear out the cdl table in case it changes.

	krefCntrlFree();

	// Retrieve the root element of the definition

	if( RC_BAD( rc = checkDictDefInfo( ui64DocumentID, bDeleting,
								&uiDictType, &uiDictNumber)))
	{
		flmAssert( rc != NE_XFLM_DOM_NODE_NOT_FOUND);
		goto Exit;
	}

	// Was the document one of our known dictionary document types?
	// Also, had a dictionary number been defined.  If bDeleting
	// is TRUE and no dictionary number had been defined, there
	// is nothing to be done on the internal dictionary.

	if (!uiDictType || !uiDictNumber)
	{
		goto Exit;	// Will return NE_XFLM_OK
	}

	// Create a separate dictionary object if one has not already
	// been created.

	if (!(m_uiFlags & FDB_UPDATED_DICTIONARY))
	{
		if (RC_BAD( rc = dictClone()))
		{
			goto Exit;
		}
	}

	// Update the dictionary and create/drop indexes or containers
	// if it is that type of definition

	if (RC_BAD( rc = m_pDict->updateDict( this, uiDictType, ui64DocumentID,
							uiDictNumber, FALSE, bDeleting)))
	{
		goto Exit;
	}
	
	// Return the type of definition
	
	if( puiDictDefType)
	{
		*puiDictDefType = uiDictType;
	}

Exit:

	if( RC_BAD( rc))
	{
		setMustAbortTrans( rc);
	}

	return( rc );
}

/****************************************************************************
Desc:	Copies an existing dictionary to a new dictionary.
****************************************************************************/
RCODE F_Db::dictClone( void)
{
	RCODE		rc = NE_XFLM_OK;
	F_Dict *	pNewDict = NULL;

	// Allocate a new FDICT structure

	if ((pNewDict = f_new F_Dict) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	
	// Nothing to do is not a legal state.

	if (!m_pDict)
	{
		flmAssert( 0);
		m_pDict = pNewDict;
		goto Exit;
	}

	// Copy the dictionary.

	if (RC_BAD( rc = pNewDict->cloneDict( m_pDict)))
	{
		goto Exit;
	}

	m_pDatabase->lockMutex();
	unlinkFromDict();
	m_pDatabase->unlockMutex();
	m_pDict = pNewDict;
	pNewDict = NULL;
	m_uiFlags |= FDB_UPDATED_DICTIONARY;

Exit:

	if (RC_BAD( rc) && pNewDict)
	{
		pNewDict->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Build an index.
****************************************************************************/
RCODE F_Db::buildIndex(
	FLMUINT		uiIndexNum,
	FLMUINT		uiState)
{
	RCODE   		rc = NE_XFLM_OK;
	LFILE *		pIxLFile;
	IXD *			pIxd;

	// Flush any KY keys and free the tables because they may grow!

	if (RC_BAD( rc = keysCommit( TRUE)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = krefCntrlCheck()))
	{
	  	goto Exit;
	}

	if (RC_BAD(rc = m_pDict->getIndex( uiIndexNum, &pIxLFile, &pIxd, TRUE)))
	{
		goto Exit;
	}

	// NON-BLOCKING INDEX BUILD - NOTE: The IXD_SUSPENDED flag may
	// also be set, which indicates that we should NOT start the
	// background maintenance thread right now.

	if (uiState & IXD_OFFLINE)
	{
		if (RC_BAD( rc = setIxStateInfo( pIxd->uiIndexNum, 0, uiState)))
		{
			goto Exit;
		}

		// setIxStateInfo may have changed to a new dictionary, so pIxd is no
		// good after this point

		pIxd = NULL;

		// Don't schedule a maintenance thread if index is to start
		// out life in a suspended state, or if we are replaying
		// the roll-forward log.

		if (!(uiState & IXD_SUSPENDED) && !(m_uiFlags & FDB_REPLAYING_RFL))
		{
			if (RC_BAD( rc = addToStartList( uiIndexNum)))
			{
				goto Exit;
			}
		}

		// Done

		goto Exit;
	}

	// There may be "new" nodes in the node cache.
	// Need to flush them to the database so that
	// the B-Tree lookups done by the indexing code will
	// work correctly

	if( RC_BAD( rc = flushDirtyNodes()))
	{
		goto Exit;
	}

	// NORMAL INDEX BUILD - BLOCKING.  uiIndexToBeUpdated better be
	// zero at this point since we are not working in the background.

	if (RC_BAD( rc = indexSetOfDocuments( uiIndexNum, 1,
			~((FLMUINT64)0), m_pIxStatus, m_pIxClient, NULL, NULL)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Logs information about an index being built
****************************************************************************/
void flmLogIndexingProgress(
	FLMUINT		uiIndexNum,
	FLMUINT64	ui64LastDocumentId)
{
	IF_LogMessageClient *	pLogMsg = NULL;
	char							szMsg[ 128];

	if( (pLogMsg = flmBeginLogMessage( XFLM_GENERAL_MESSAGE)) != NULL)
	{
		pLogMsg->changeColor( FLM_YELLOW, FLM_BLACK);
		if (ui64LastDocumentId)
		{
			f_sprintf( (char *)szMsg,
				"Indexing progress: Index %u is offline.  Last document processed = %I64u.",
				(unsigned)uiIndexNum, ui64LastDocumentId);
		}
		else
		{
			f_sprintf( (char *)szMsg,
				"Indexing progress: Index %u is online.",
				(unsigned)uiIndexNum);
		}
		pLogMsg->appendString( szMsg);
	}
	flmEndLogMessage( &pLogMsg);
}

/****************************************************************************
Desc:	Unlink and free an IX_CONTEXT structure.
****************************************************************************/
FSTATIC void kyFreeIxContext(
	IXD *				pIxd,
	IX_CONTEXT *	pIxContext,
	IX_CONTEXT **	ppIxContextList
	)
{
	if (pIxContext->pPrev)
	{
		pIxContext->pPrev->pNext = pIxContext->pNext;
	}
	else
	{
		*ppIxContextList = pIxContext->pNext;
	}
	if (pIxContext->pNext)
	{
		pIxContext->pNext->pPrev = pIxContext->pPrev;
	}
	kyReleaseCdls( pIxd, pIxContext->pCdlTbl);
	if (pIxContext->pPool)
	{
		pIxContext->pPool->poolFree();
		pIxContext->pPool->Release();
	}
	f_free( &pIxContext);
}

/****************************************************************************
Desc:	Output the keys for a particular IX_CONTEXT structure.  Also, free
		the IX_CONTEXT structure and its associated CDLs, etc.
****************************************************************************/
RCODE F_Db::outputContextKeys(
	FLMUINT64		ui64DocumentId,
	IXD *				pIxd,
	IX_CONTEXT *	pIxContext,
	IX_CONTEXT **	ppIxContextList)
{
	RCODE	rc = NE_XFLM_OK;

	if (RC_BAD( rc = buildKeys( ui64DocumentId, pIxd,
							pIxContext->pCdlTbl, TRUE, TRUE)))
	{
		goto Exit;
	}

	// Free the IX_CONTEXT structure - unlinks from list too.

	kyFreeIxContext( pIxd, pIxContext, ppIxContextList);

	// Flush keys if over threshhold - get key count before flushing.

	if( pIxd->uiIndexNum && isKrefOverThreshold())
	{
		processDupKeys( pIxd);
		if (RC_BAD( rc = keysCommit( FALSE, FALSE)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Output the keys for a particular IX_CONTEXT structure and remove CDLs
		for the specified ICD, if any.
****************************************************************************/
RCODE F_Db::removeCdls(
	FLMUINT64		ui64DocumentId,
	IXD *				pIxd,
	IX_CONTEXT *	pIxContext,
	ICD *				pRefIcd)
{
	RCODE		rc = NE_XFLM_OK;
	CDL *		pCdl;
	CDL *		pOldCdlList;
	ICD *		pIcd;

	if (RC_BAD( rc = buildKeys( ui64DocumentId, pIxd,
							pIxContext->pCdlTbl, TRUE, TRUE)))
	{
		goto Exit;
	}

	pIcd = pRefIcd;
	while (pIcd)
	{

		// Free the CDLs for the specified ICD, if any.
		// Put the CDLs into a table for reuse.

		pCdl = pIxContext->pCdlTbl [pIcd->uiCdl].pCdlList;
		pIxContext->pCdlTbl [pIcd->uiCdl].pCdlList = NULL;
		if (pCdl)
		{
			pOldCdlList = pIxContext->pCdlList;
			pIxContext->pCdlList = pCdl;
			for (;;)
			{
				if (pCdl->pNode)
				{
					pCdl->pNode->Release();
					pCdl->pNode = NULL;
				}
				if (!pCdl->pNext)
				{
					pCdl->pNext = pOldCdlList;
					break;
				}
				pCdl = pCdl->pNext;
			}
		}
		if (pIcd == pRefIcd->pParent)
		{
			break;
		}
		if ((pIcd = pIcd->pNextSibling) == NULL)
		{

			// Also do reference ICD's parent ICD.  This is not absolutely
			// necessary, since this CDLs will be ignored when we verify
			// context, but we might as well do it to save just a little more
			// memory space.

			if ((pIcd = pRefIcd->pParent) == NULL)
			{
				break;
			}
		}
	}

	// Flush keys if over threshhold - get key count before flushing.

	if( pIxd->uiIndexNum && isKrefOverThreshold())
	{
		processDupKeys( pIxd);
		if (RC_BAD( rc = keysCommit( FALSE, FALSE)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Index a particular document for a particular index.
****************************************************************************/
RCODE F_Db::indexDocument(
	IXD *			pIxd,
	F_DOMNode *	pDocNode)
{
	RCODE				rc = NE_XFLM_OK;
	ICD *				pIcd;
	ICD *				pChildIcd;
	CDL *				pCdl;
	IX_CONTEXT *	pIxContextList = NULL;
	IX_CONTEXT *	pIxContext;
	F_DOMNode *		pTmpNode = NULL;
	F_TRAV *			pTrav = NULL;
	void *			pvMark = m_tempPool.poolMark();
	CDL_HDR *		pCdlHdr;
	FLMUINT64		ui64DocId;

	// Root of a document cannot be an attribute node
	
	flmAssert( pDocNode->getNodeType() != ATTRIBUTE_NODE);

	// Get the document ID
	
	if( RC_BAD( rc = pDocNode->getNodeId( this, &ui64DocId)))
	{
		goto Exit;
	}
				
	// If the index number is zero, we are generating keys for a query
	// but not to store in the database.  We want the kref table cleared
	// out.
	
	if (!pIxd->uiIndexNum)
	{
		if (!m_bKrefSetup)
		{
			if (RC_BAD( rc = krefCntrlCheck()))
			{
				goto Exit;
			}
		}
		else if (m_eTransType == XFLM_UPDATE_TRANS)
		{
			if (RC_BAD( rc = keysCommit( FALSE)))
			{
				goto Exit;
			}
		}
		else
		{
			// Empty the table out so that the only keys added in this
			// call are for this index, this document.

			m_pKrefPool->poolReset( NULL, TRUE);
			m_uiKrefCount = 0;
			m_uiTotalKrefBytes = 0;
		}
		flmAssert( !m_uiKrefCount && !m_uiTotalKrefBytes);
	}
	else
	{
		if (RC_BAD( rc = krefCntrlCheck()))
		{
			goto Exit;
		}
	}

	// Do an in-order traversal of the document.

	if (RC_BAD( rc = m_tempPool.poolCalloc( sizeof( F_TRAV), (void **)&pTrav)))
	{
		goto Exit;
	}
	pTrav->pNode = pDocNode;
	pTrav->pNode->AddRef();
	// pTrav->pRefIcd = NULL;		// Set by poolCalloc
	// pTrav->pIcd = NULL;			// Set by poolCalloc
	// pTrav->pIxContext = NULL;	// Set by poolCalloc
	// pTrav->pParent = NULL;		// Set by poolCalloc
	// pTrav->pChild = NULL;		// Set by poolCalloc

	for (;;)
	{
		FLMUINT			uiNameId;
		eDomNodeType	eNodeType = pTrav->pNode->getNodeType();
		
		if( RC_BAD( rc = pTrav->pNode->getNameId( this, &uiNameId)))
		{
			goto Exit;
		}

		if (uiNameId &&
			 (eNodeType == ELEMENT_NODE || eNodeType == ATTRIBUTE_NODE))
		{
			FLMBOOL	bCheckedIcdTreeRoot = FALSE;

			// See if the node has an ICD in the current context.

			if ((pIcd = pTrav->pRefIcd) == NULL)
			{
				pIcd = pIxd->pIcdTree;
				bCheckedIcdTreeRoot = TRUE;
			}
			if (pIcd->uiDictNum == ELM_ROOT_TAG)
			{
				if (pTrav->pNode->getParentId())
				{
					pIcd = NULL;
				}
			}
			else if (eNodeType == ELEMENT_NODE)
			{
				while (pIcd && (pIcd->uiDictNum != uiNameId ||
									 (pIcd->uiFlags & ICD_IS_ATTRIBUTE)))
				{
					pIcd = pIcd->pNextSibling;
				}

				// If we did not start at the root of the ICD tree,
				// need to go back there to see if we need to start
				// a new context.  However, if the root if the ICD tree
				// is ELM_ROOT_TAG, it is pointless, because the current
				// pTrev->pNode should be a child of some node if
				// bCheckedIcdTreeRoot is FALSE.

				if (!pIcd && !bCheckedIcdTreeRoot &&
					 pIxd->pIcdTree->uiDictNum != ELM_ROOT_TAG)
				{
					pIcd = pIxd->pIcdTree;
					while (pIcd && (pIcd->uiDictNum != uiNameId ||
										(pIcd->uiFlags & ICD_IS_ATTRIBUTE)))
					{
						pIcd = pIcd->pNextSibling;
					}
					if (pIcd)
					{

						// Reset these so that a new context will be
						// created below.

						pTrav->pRefIcd = NULL;
						pTrav->pIxContext = NULL;
					}
				}
			}
			else
			{
				while (pIcd && (pIcd->uiDictNum != uiNameId ||
									 !(pIcd->uiFlags & ICD_IS_ATTRIBUTE)))
				{
					pIcd = pIcd->pNextSibling;
				}

				// If we did not start at the root of the ICD tree,
				// need to go back there to see if we need to start
				// a new context.  However, if the root if the ICD tree
				// is ELM_ROOT_TAG, it is pointless, because the current
				// pTrev->pNode should be a child of some node if
				// bCheckedIcdTreeRoot is FALSE.

				if (!pIcd && !bCheckedIcdTreeRoot &&
					 pIxd->pIcdTree->uiDictNum != ELM_ROOT_TAG)
				{
					pIcd = pIxd->pIcdTree;
					while (pIcd && (pIcd->uiDictNum != uiNameId ||
										!(pIcd->uiFlags & ICD_IS_ATTRIBUTE)))
					{
						pIcd = pIcd->pNextSibling;
					}
					if (pIcd)
					{

						// Reset these so that a new context will be
						// created below.

						pTrav->pRefIcd = NULL;
						pTrav->pIxContext = NULL;
					}
				}
			}

			// If we found a matching ICD, see if we want to save the node
			// in a CDL table.

			if ((pTrav->pIcd = pIcd) != NULL)
			{

				// If there is no indexing context, start one.

				if (!pTrav->pIxContext)
				{
					flmAssert( !pTrav->pRefIcd);
					if (RC_BAD( rc = f_calloc( sizeof( IX_CONTEXT), &pIxContext)))
					{
						goto Exit;
					}
					if ((pIxContext->pNext = pIxContextList) != NULL)
					{
						pIxContextList->pPrev = pIxContext;
					}
					pIxContextList = pIxContext;
					
					if ((pIxContext->pPool = f_new F_Pool) == NULL)
					{
						rc = RC_SET( NE_XFLM_MEM);
						goto Exit;
					}

					pIxContext->pPool->poolInit( 512);

					if (RC_BAD( rc = pIxContext->pPool->poolCalloc(
											sizeof( CDL_HDR) * pIxd->uiNumIcds,
											(void **)&pIxContext->pCdlTbl)))
					{
						goto Exit;
					}
					pTrav->pIxContext = pIxContext;
					pTrav->pRefIcd = pIxd->pIcdTree;
				}
				else
				{
					pIxContext = pTrav->pIxContext;
				}

				// If this node has a parent, and the front of the
				// list is a "missing" place holder for the node,
				// replace that CDL.

				pCdlHdr = &pIxContext->pCdlTbl [pIcd->uiCdl];
				pCdl = pCdlHdr->pCdlList;
				if (pCdl && !pCdl->pNode &&
					 pTrav->pNode->getParentId() &&
					 pCdl->ui64ParentId == pTrav->pNode->getParentId())
				{
					pCdl->pNode = pTrav->pNode;
					pCdl->bInNodeSubtree = TRUE;
					pCdl->pNode->AddRef();
				}
				else
				{

					// Reuse a CDL if one is available.

					if (pIxContext->pCdlList)
					{
						pCdl = pIxContext->pCdlList;
						pIxContext->pCdlList = pCdl->pNext;

						// pCdl->pNode should have been released when it was
						// put into the CDL list!

						flmAssert( !pCdl->pNode);
					}
					else
					{
						if (RC_BAD( rc = pIxContext->pPool->poolAlloc(
													sizeof( CDL), (void **)&pCdl)))
						{
							goto Exit;
						}
					}
					pCdl->pNode = pTrav->pNode;
					pCdl->ui64ParentId = pTrav->pNode->getParentId();
					pCdl->bInNodeSubtree = TRUE;
					pCdl->pNode->AddRef();
					pCdl->pNext = pCdlHdr->pCdlList;
					pCdlHdr->pCdlList = pCdl;
				}

				// Add "missing" place-holders for any child ICDs

				pChildIcd = pIcd->pFirstChild;
				while (pChildIcd)
				{

					// Reuse a CDL if one is available.

					if (pIxContext->pCdlList)
					{
						pCdl = pIxContext->pCdlList;
						pIxContext->pCdlList = pCdl->pNext;

						// pCdl->pNode should have been released when it was
						// put into the CDL list!

						flmAssert( !pCdl->pNode);
					}
					else
					{
						if (RC_BAD( rc = pIxContext->pPool->poolAlloc(
													sizeof( CDL), (void **)&pCdl)))
						{
							goto Exit;
						}
					}
					
					if( RC_BAD( rc = pTrav->pNode->getNodeId( 
						this, &pCdl->ui64ParentId)))
					{
						goto Exit;
					}

					pCdl->pNode = NULL;
					pCdl->bInNodeSubtree = TRUE;
					pCdlHdr = &pIxContext->pCdlTbl [pChildIcd->uiCdl];
					pCdl->pNext = pCdlHdr->pCdlList;
					pCdlHdr->pCdlList = pCdl;
					pChildIcd = pChildIcd->pNextSibling;
				}
			}
		}

		// Go to the next node.

		if( eNodeType == ATTRIBUTE_NODE)
		{
			if (RC_OK( rc = pTrav->pNode->getNextSibling( this, 
				(IF_DOMNode **)&pTrav->pNode)))
			{
				pTrav->pIcd = NULL;
				continue;
			}
			else if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				goto Exit;
			}
			else
			{
				rc = NE_XFLM_OK;
			}

			// No siblings, go back to parent node and see if it has any
			// siblings.

			if (pTrav->pNode)
			{
				pTrav->pNode->Release();
				pTrav->pNode = NULL;
			}

			// If the parent has a different IX_CONTEXT (including NULL)
			// free this one before going back to the parent.

			if (pTrav->pIxContext && pTrav->pParent &&
				 pTrav->pParent->pIxContext != pTrav->pIxContext)
			{
				if (RC_BAD( rc = outputContextKeys( ui64DocId, pIxd,
											pTrav->pIxContext, &pIxContextList)))
				{
					goto Exit;
				}
				pTrav->pIxContext = NULL;
				pTrav->pRefIcd = NULL;
			}
			else if (pTrav->pIxContext && pIxd->uiFlags & IXD_SINGLE_PATH)
			{
				if (RC_BAD( rc = removeCdls( ui64DocId, pIxd, pTrav->pIxContext,
											pTrav->pRefIcd)))
				{
					goto Exit;
				}
			}

			pTrav = pTrav->pParent;

			// Has to be a parent at this point!

			flmAssert( pTrav);

			// Fall through to do element's siblings.
		}
		else if (eNodeType == ELEMENT_NODE || eNodeType == DOCUMENT_NODE)
		{
			if (RC_OK( rc = pTrav->pNode->getFirstChild( this, (IF_DOMNode **)&pTmpNode)))
			{
Setup_Child:
				if (!pTrav->pChild)
				{
					F_TRAV *	pNewTrav;

					if (RC_BAD( rc = m_tempPool.poolCalloc( sizeof( F_TRAV),
														(void **)&pNewTrav)))
					{
						goto Exit;
					}
					pNewTrav->pParent = pTrav;
					pTrav->pChild = pNewTrav;
				}
				pTrav = pTrav->pChild;
				if (pTrav->pNode)
				{
					pTrav->pNode->Release();
					pTrav->pNode = NULL;
				}
				pTrav->pNode = pTmpNode;
				pTrav->pNode->AddRef();
				pTmpNode->Release();
				pTmpNode = NULL;
				pTrav->pRefIcd = (pTrav->pParent->pIcd
										? pTrav->pParent->pIcd->pFirstChild
										: NULL);
				pTrav->pIcd = NULL;
				pTrav->pIxContext = pTrav->pParent->pIxContext;
				if (!pTrav->pRefIcd)
				{
					pTrav->pIxContext = NULL;
				}
				continue;
			}
			else if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				goto Exit;
			}
			else
			{
				rc = NE_XFLM_OK;
			}

			// See if the node has any attributes

			if( pTrav->pNode->getNodeType() == ELEMENT_NODE)
			{
				if (RC_OK( rc = pTrav->pNode->getFirstAttribute( this,
												(IF_DOMNode **)&pTmpNode)))
				{
					goto Setup_Child;
				}
				else if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					goto Exit;
				}
				else
				{
					rc = NE_XFLM_OK;
				}
			}
		}

		// Follow sibling chain.  Go to parents until we find a
		// parent that has a sibling.

		for (;;)
		{
			// If the parent has a different IX_CONTEXT
			// free this one before going back to the parent.

			if (pTrav->pIxContext && pTrav->pParent && pTrav->pParent->pIxContext &&
					pTrav->pParent->pIxContext != pTrav->pIxContext)
			{
				if (RC_BAD( rc = outputContextKeys( ui64DocId, pIxd,
											pTrav->pIxContext, &pIxContextList)))
				{
					goto Exit;
				}
				pTrav->pRefIcd = (pTrav->pParent->pIcd
										? pTrav->pParent->pIcd->pFirstChild
										: NULL);
				pTrav->pIcd = NULL;
				pTrav->pIxContext = pTrav->pParent->pIxContext;
				if (!pTrav->pRefIcd)
				{
					pTrav->pIxContext = NULL;
				}
			}
			if (RC_BAD( rc = pTrav->pNode->getNextSibling( this, (IF_DOMNode **)&pTrav->pNode)))
			{
				if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					goto Exit;
				}
				rc = NE_XFLM_OK;

				// If there is no parent node, we are done.

				if (!pTrav->pParent)
				{
					goto Done_With_Document;
				}

				// See if the parent has a a first attribute.

				if( pTrav->pParent->pNode->getNodeType() == ELEMENT_NODE)
				{
					if (RC_BAD( rc = pTrav->pParent->pNode->getFirstAttribute( this,
														(IF_DOMNode **)&pTrav->pNode)))
					{
						if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
						{
							goto Exit;
						}
						rc = NE_XFLM_OK;
					}
					else
					{

						// Will continue processing attributes as siblings.

						break;
					}
				}

				// At this point we will be going back to the parent node.

				if (pTrav->pNode)
				{
					pTrav->pNode->Release();
					pTrav->pNode = NULL;
				}
				
				// If the parent has no context, we need to output the keys we may
				// have collected so far.  NOTE: If parent's context is non-NULL and
				// just different than our context, we will already have output the
				// keys above.  pTrav->pParent must be non-NULL at this point.

				if (pTrav->pIxContext && !pTrav->pParent->pIxContext)
				{
					if (RC_BAD( rc = outputContextKeys( ui64DocId, pIxd,
												pTrav->pIxContext, &pIxContextList)))
					{
						goto Exit;
					}
				}
				else if (pTrav->pIxContext && pIxd->uiFlags & IXD_SINGLE_PATH)
				{
					if (RC_BAD( rc = removeCdls( ui64DocId, pIxd, pTrav->pIxContext,
												pTrav->pRefIcd)))
					{
						goto Exit;
					}
				}

				pTrav = pTrav->pParent;
			}
			else if (pTrav->pNode->getNodeType() == ELEMENT_NODE)
			{
				pTrav->pIcd = NULL;
				break;
			}
			else
			{

				// better not be in a list of attribute nodes at this point!

				flmAssert( pTrav->pNode->getNodeType() != ATTRIBUTE_NODE);
			}
		}
	}

Done_With_Document:

	// Need to build keys for each index context that we have.

	while (pIxContextList)
	{
		if (RC_BAD( rc = outputContextKeys( ui64DocId, pIxd,
									pIxContextList, &pIxContextList)))
		{
			goto Exit;
		}
	}

	// Flush keys - get key count before flushing.
	
	if (!pIxd->uiIndexNum)
	{
		processDupKeys( pIxd);
	}
	else
	{
		if( isKrefOverThreshold())
		{
			processDupKeys( pIxd);
			if (RC_BAD( rc = keysCommit( FALSE, FALSE)))
			{
				goto Exit;
			}
		}
	}

Exit:

	if (pTmpNode)
	{
		pTmpNode->Release();
	}

	while (pTrav && pTrav->pParent)
	{
		pTrav = pTrav->pParent;
	}
	while (pTrav)
	{
		if (pTrav->pNode)
		{
			pTrav->pNode->Release();
			pTrav->pNode = NULL;
		}
		pTrav = pTrav->pChild;
	}

	while (pIxContextList)
	{
		kyFreeIxContext( pIxd, pIxContextList, &pIxContextList);
	}

	m_tempPool.poolReset( pvMark);
	return( rc);
}

/****************************************************************************
Desc:	Index a set of documents or until time runs out.
****************************************************************************/
RCODE F_Db::indexSetOfDocuments(
	FLMUINT					uiIxNum,
	FLMUINT64				ui64StartDocumentId,
	FLMUINT64				ui64EndDocumentId,
	IF_IxStatus *			ifpIxStatus,
	IF_IxClient *			ifpIxClient,
	XFLM_INDEX_STATUS *	pIndexStatus,
	FLMBOOL *				pbHitEnd,
	IF_Thread *				pThread)
{
	RCODE						rc = NE_XFLM_OK;
	FLMUINT64				ui64DocumentId;
	FLMUINT64				ui64LastDocumentId = 0;
	IXD *						pIxd = NULL;
	F_COLLECTION *			pCollection;
	IF_LockObject *		pDatabaseLockObj = m_pDatabase->m_pDatabaseLockObj;
	FLMBOOL					bHitEnd = FALSE;
	FLMUINT					uiCurrTime;
	FLMUINT					uiLastStatusTime = 0;
	FLMUINT					uiStartTime;
	FLMUINT					uiMinTU;
	FLMUINT					uiStatusIntervalTU;
	FLMUINT64				ui64DocumentsProcessed = 0;
	FLMBOOL					bUpdateTracker = FALSE;
	FLMBOOL					bRelinquish = FALSE;
	FLMBYTE					ucKey[ FLM_MAX_NUM_BUF_SIZE];
	FLMUINT					uiKeyLen;
	void *					pvTmpPoolMark = m_tempPool.poolMark();
	F_Btree *				pbtree = NULL;
	FLMBOOL					bNeg;
	FLMUINT					uiBytesProcessed;
	F_DOMNode *				pNode = NULL;

	uiMinTU = FLM_MILLI_TO_TIMER_UNITS( 500);
	uiStatusIntervalTU = FLM_SECS_TO_TIMER_UNITS( 10);
	uiStartTime = FLM_GET_TIMER();

	if (RC_BAD( rc = krefCntrlCheck()))
	{
		goto Exit;
	}

	if (RC_BAD( rc = m_pDict->getIndex( uiIxNum, NULL, &pIxd, TRUE)))
	{
		goto Exit;
	}
	flmAssert( !(pIxd->uiFlags & IXD_SUSPENDED));

	// Get a btree

	if (RC_BAD( rc = gv_XFlmSysData.pBtPool->btpReserveBtree( &pbtree)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = m_pDict->getCollection(
								pIxd->uiCollectionNum, &pCollection)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pbtree->btOpen( this, &pCollection->lfInfo,
								FALSE, TRUE)))
	{
		goto Exit;
	}

	uiKeyLen = sizeof( ucKey);
	if (RC_BAD( rc = flmNumber64ToStorage( ui64StartDocumentId, &uiKeyLen,
									ucKey, FALSE, TRUE)))
	{
		goto Exit;
	}
	if( RC_BAD( rc = pbtree->btLocateEntry(
								ucKey, sizeof( ucKey), &uiKeyLen, XFLM_INCL)))
	{
		if (rc == NE_XFLM_EOF_HIT || rc == NE_XFLM_NOT_FOUND)
		{
			rc = NE_XFLM_OK;
			bHitEnd = TRUE;
			goto Commit_Keys;
		}

		goto Exit;
	}

	// Make sure we hit a root node.  If not, continue reading until we do
	// or until we hit the end.  Root nodes are always linked together in
	// ascending order, so if there is another document, we will find it
	// simply by searching forward from where we are.  Then we can follow
	// document links.

	for (;;)
	{
		if (RC_BAD( rc = flmCollation2Number( uiKeyLen, ucKey,
									&ui64DocumentId, &bNeg, &uiBytesProcessed)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = getNode( pIxd->uiCollectionNum, ui64DocumentId,
										&pNode)))
		{
			if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{

				// Better be able to find the node at this point!

				rc = RC_SET( NE_XFLM_DATA_ERROR);
				goto Exit;
			}
		}

		// If the node is a root node, we have a document we can
		// process.

		if (pNode->isRootNode())
		{

			// This is a root node - has no parent and is not linked
			// into orphan list.

			break;
		}

		// Need to go to the next node.

		if (RC_BAD( rc = pbtree->btNextEntry( ucKey, uiKeyLen, &uiKeyLen)))
		{
			if (rc == NE_XFLM_EOF_HIT)
			{
				rc = NE_XFLM_OK;
				bHitEnd = TRUE;
				goto Commit_Keys;
			}
			goto Exit;
		}
	}

	for (;;)
	{
		if( RC_BAD( rc = pNode->getNodeId( this, &ui64DocumentId)))
		{
			goto Exit;
		}
		
		if (ui64DocumentId > ui64EndDocumentId)
		{
			break;
		}

		if (RC_BAD( rc = indexDocument( pIxd, pNode)))
		{
			goto Exit;
		}

		// See if there is an indexing callback

		if (ifpIxClient)
		{
			if (RC_BAD( rc = ifpIxClient->doIndexing( this, uiIxNum,
								pIxd->uiCollectionNum, pNode)))
			{
				goto Exit;
			}
		}

		ui64LastDocumentId = ui64DocumentId;
		ui64DocumentsProcessed++;

		if (pIndexStatus)
		{
			pIndexStatus->ui64DocumentsProcessed++;
			pIndexStatus->ui64LastDocumentIndexed = ui64LastDocumentId;
		}

		// Get the current time

		uiCurrTime = FLM_GET_TIMER();

		// Break out if someone is waiting for an update transaction.

		if (pThread)
		{
			if (pThread->getShutdownFlag())
			{
				bRelinquish = TRUE;
				break;
			}

			if (pDatabaseLockObj->getWaiterCount())
			{
				// See if our minimum run time has elapsed

				if (FLM_ELAPSED_TIME( uiCurrTime, uiStartTime) >= uiMinTU)
				{
					if (ui64DocumentsProcessed < 50)
					{
						// If there are higher priority waiters in the lock queue,
						// we want to relinquish.

						if (pDatabaseLockObj->haveHigherPriorityWaiter(
							FLM_BACKGROUND_LOCK_PRIORITY))
						{
							bRelinquish = TRUE;
							break;
						}
					}
					else
					{
						bRelinquish = TRUE;
						break;
					}
				}
			}
			else
			{

				// Even if no one has requested a lock for a long time, we
				// still want to periodically commit our transaction so
				// we won't lose more than uiMaxCPInterval timer units worth
				// of work if we crash.  We will run until we exceed the checkpoint
				// interval and we see that someone (the checkpoint thread) is
				// waiting for the write lock.

				if (FLM_ELAPSED_TIME( uiCurrTime, uiStartTime) >
					gv_XFlmSysData.uiMaxCPInterval &&
					m_pDatabase->m_pWriteLockObj->getWaiterCount())
				{
					bRelinquish = TRUE;
					break;
				}
			}
		}

		if (FLM_ELAPSED_TIME( uiCurrTime, uiLastStatusTime) >=
					uiStatusIntervalTU)
		{
			uiLastStatusTime = uiCurrTime;
			if( ifpIxStatus)
			{
				if( RC_BAD( rc = ifpIxStatus->reportIndex( ui64LastDocumentId)))
				{
					goto Exit;
				}
			}

			// Send indexing completed event notification

			if( gv_XFlmSysData.EventHdrs[ XFLM_EVENT_UPDATES].pEventCBList)
			{
				flmDoEventCallback( XFLM_EVENT_UPDATES,
						XFLM_EVENT_INDEXING_PROGRESS, this, f_threadId(),
						0, uiIxNum, ui64LastDocumentId,
						NE_XFLM_OK);
			}

			// Log a progress message

			flmLogIndexingProgress( uiIxNum, ui64LastDocumentId);
		}

		// Need to go to the next document.

		if (RC_BAD( rc = pNode->getNextDocument( this, (IF_DOMNode **)&pNode)))
		{
			if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				rc = NE_XFLM_OK;
				bHitEnd = TRUE;
				break;
			}
		}
	}

Commit_Keys:

	if (RC_BAD( rc = keysCommit( TRUE)))
	{
		goto Exit;
	}

	// If at the end, change trans ID to the current transaction.

	if (bHitEnd)
	{
		if (RC_BAD( rc = setIxStateInfo( uiIxNum, ~((FLMUINT64)0), 0)))
		{
			goto Exit;
		}

		// setIxStateInfo may have changed to a new dictionary, so pIxd is no
		// good after this point

		pIxd = NULL;
	}
	else if (ui64DocumentsProcessed || bUpdateTracker)
	{
		if (RC_BAD( rc = setIxStateInfo( uiIxNum, ui64LastDocumentId,
										IXD_OFFLINE)))
		{
			goto Exit;
		}

		// setIxStateInfo may have changed to a new dictionary, so pIxd is no
		// good after this point

		pIxd = NULL;
	}

Exit:

	// We want to make one last call if we are in the foreground or if
	// we actually did some indexing.

	if (gv_XFlmSysData.EventHdrs[ XFLM_EVENT_UPDATES].pEventCBList)
	{
		flmDoEventCallback( XFLM_EVENT_UPDATES,
				XFLM_EVENT_INDEXING_PROGRESS, this, f_threadId(),
				0, uiIxNum,
				(FLMUINT64)(bHitEnd ? (FLMUINT64)0 : ui64LastDocumentId),
				NE_XFLM_OK);
	}

	flmLogIndexingProgress( uiIxNum,
		(FLMUINT64)(bHitEnd ? (FLMUINT64)0 : ui64LastDocumentId));

	if (ifpIxStatus)
	{
		(void) ifpIxStatus->reportIndex( ui64LastDocumentId);
	}

	if (pbHitEnd)
	{
		*pbHitEnd = bHitEnd;
	}

	krefCntrlFree();
	m_tempPool.poolReset( pvTmpPoolMark);

	if (pbtree)
	{
		gv_XFlmSysData.pBtPool->btpReturnBtree( &pbtree);
	}

	if (pNode)
	{
		pNode->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Set information in the tracker record for the index.
****************************************************************************/
RCODE F_Db::setIxStateInfo(
	FLMUINT		uiIndexNum,
	FLMUINT64	ui64LastDocIndexed,
	FLMUINT		uiState)
{
	RCODE				rc = NE_XFLM_OK;
	IXD_FIXUP *		pIxdFixup;
	IXD *				pIxd;
	F_DOMNode *		pAttr = NULL;
	F_DOMNode *		pElement = NULL;
	FLMBOOL			bMustAbortOnError = FALSE;

	// Get the IXD - even if the index is offline.

	if (RC_BAD( rc = m_pDict->getIndex( uiIndexNum, NULL, &pIxd, TRUE)))
	{
		goto Exit;
	}

	// See if this index is in our fixup list.

	pIxdFixup = m_pIxdFixups;
	while (pIxdFixup && pIxdFixup->uiIndexNum != uiIndexNum)
	{
		pIxdFixup = pIxdFixup->pNext;
	}

	if (!pIxdFixup)
	{
		if (RC_BAD( rc = f_calloc( (FLMUINT)sizeof( IXD_FIXUP), &pIxdFixup)))
		{
			goto Exit;
		}
		pIxdFixup->pNext = m_pIxdFixups;
		m_pIxdFixups = pIxdFixup;
		pIxdFixup->uiIndexNum = uiIndexNum;
		pIxdFixup->ui64LastDocIndexed = pIxd->ui64LastDocIndexed;
	}

	bMustAbortOnError = TRUE;

	// Update the last node indexed, if it changed.

	if (pIxdFixup->ui64LastDocIndexed != ui64LastDocIndexed)
	{
		pIxdFixup->ui64LastDocIndexed = ui64LastDocIndexed;

		// First, retrieve the root element of the index definition.

		if( RC_BAD( rc = getNode( XFLM_DICT_COLLECTION, pIxd->ui64IxDefNodeId,
								(F_DOMNode **)&pElement)))
		{
			if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			}

			goto Exit;
		}

		// No need to create a new node if we are just setting it
		// to the default value - see fdict.cpp, getIndexDef for where
		// the default value gets set.

		if (ui64LastDocIndexed != ~((FLMUINT64)0))
		{

			// Create a new dictionary - so that we can set the
			// ui64LastDocIndexedNodeId on the IXD.  If the transaction
			// aborts, the whole dictionary will go away.

			if (!(m_uiFlags & FDB_UPDATED_DICTIONARY))
			{
				if (RC_BAD( rc = dictClone()))
				{
					goto Exit;
				}

				// Get a pointer to the new IXD

				if (RC_BAD( rc = m_pDict->getIndex( uiIndexNum, NULL, &pIxd, TRUE)))
				{
					goto Exit;
				}
			}

			// Create a new attribute node on the index definition to hold
			// the last node indexed value.

			if (RC_BAD( rc = pElement->createAttribute( this,
										ATTR_LAST_DOC_INDEXED_TAG,
										(IF_DOMNode **)&pAttr)))
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = pElement->getAttribute( this, 
				ATTR_LAST_DOC_INDEXED_TAG, (IF_DOMNode **)&pAttr)))
			{
				if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					goto Exit;
				}

				rc = NE_XFLM_OK;
			}
		}

		if( pAttr)
		{
			if (RC_BAD( rc = pAttr->setUINT64( this, ui64LastDocIndexed)))
			{
				goto Exit;
			}
		}
	}

	// If IXD_SUSPENDED is set, then IXD_OFFLINE must also be set.
	// There are places in the code that only check for IXD_OFFLINE
	// that don't care if the index is also suspended.

	if (uiState & IXD_SUSPENDED)
	{
		uiState = IXD_SUSPENDED | IXD_OFFLINE;
	}
	else if (uiState & IXD_OFFLINE)
	{
		uiState = IXD_OFFLINE;
	}
	else
	{
		uiState = 0;
	}

	// See if we need to change state.

	if ((pIxd->uiFlags & (IXD_SUSPENDED | IXD_OFFLINE)) != uiState)
	{
		const char *	pszStateStr;

		if (uiState & IXD_SUSPENDED)
		{
			pszStateStr = XFLM_INDEX_SUSPENDED_STR;
		}
		else if (uiState & IXD_OFFLINE)
		{
			pszStateStr = XFLM_INDEX_OFFLINE_STR;
		}
		else
		{
			pszStateStr = XFLM_INDEX_ONLINE_STR;
		}

		// At this point we know we need to change the state.  That means we need
		// to create a new dictionary, if we have not already done so.

		if (!(m_uiFlags & FDB_UPDATED_DICTIONARY))
		{
			if (RC_BAD( rc = dictClone()))
			{
				goto Exit;
			}

			// Get a pointer to the new IXD

			if (RC_BAD( rc = m_pDict->getIndex( uiIndexNum, NULL, &pIxd, TRUE)))
			{
				goto Exit;
			}
		}

		// pElement may have been fetched above.  Don't need to get it
		// here if that is the case.

		if (!pElement)
		{

			// First, retrieve the root element of the index definition.

			if( RC_BAD( rc = getNode( XFLM_DICT_COLLECTION, pIxd->ui64IxDefNodeId,
									(F_DOMNode **)&pElement)))
			{
				if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
				}

				goto Exit;
			}
		}

		// Create a new attribute node on the index definition to hold
		// the last node indexed value.

		if (RC_BAD( rc = pElement->createAttribute( this,
									ATTR_STATE_TAG,
									(IF_DOMNode **)&pAttr)))
		{
			goto Exit;
		}
		
		// May need to unfreeze the state to change it.

		if( RC_BAD( rc = pAttr->removeModeFlags( this,
			FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = pAttr->setUTF8( this, (FLMBYTE *)pszStateStr)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pAttr->addModeFlags( this,
			FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
		{
			goto Exit;
		}

		// Put the state into the IXD.

		pIxd->uiFlags = (pIxd->uiFlags & (~(IXD_SUSPENDED | IXD_OFFLINE))) |
							 uiState;
	}

Exit:

	if (pAttr)
	{
		pAttr->Release();
	}

	if (pElement)
	{
		pElement->Release();
	}

	if( RC_BAD( rc) && bMustAbortOnError)
	{
		setMustAbortTrans( rc);
	}

	return( rc);
}

/****************************************************************************
Desc:		See if any IXD structures need indexing in the background.
****************************************************************************/
RCODE F_Db::startBackgroundIndexing( void)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bStartedTrans = FALSE;
	FLMUINT	uiIndexNum;
	IXD *		pIxd;

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
	}
	else
	{

		// Need to have at least a read transaction going.

		if (RC_BAD( rc = beginTrans( XFLM_READ_TRANS)))
		{
			goto Exit;
		}
		bStartedTrans = TRUE;
	}

	if (m_pDict->getIndexCount( FALSE))
	{
		uiIndexNum = 0;
		for (;;)
		{
			if ((pIxd = m_pDict->getNextIndex( uiIndexNum, FALSE)) == NULL)
			{
				break;
			}
			uiIndexNum = pIxd->uiIndexNum;

			// Restart any indexes that are off-line but not suspended

			if ((pIxd->uiFlags & (IXD_OFFLINE | IXD_SUSPENDED)) == IXD_OFFLINE)
			{
				flmAssert( flmBackgroundIndexGet( m_pDatabase,
										uiIndexNum, FALSE) == NULL);

				if (RC_BAD( rc = startIndexBuild( uiIndexNum)))
				{
					goto Exit;
				}
			}
		}
	}

Exit:

	if (bStartedTrans)
	{
		(void)abortTrans();
	}

	return( rc);
}

/****************************************************************************
Desc:	Check and set the next dictionary number for a specific dictionary type.
****************************************************************************/
RCODE F_Db::setNextDictNum(
	FLMUINT	uiDictType,
	FLMUINT	uiDictNumber
	)
{
	RCODE	rc = NE_XFLM_OK;

	// Make sure an update transaction is active

	if (m_eTransType == XFLM_NO_TRANS)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NO_TRANS_ACTIVE);
		goto Exit;
	}
	
	if (m_eTransType == XFLM_READ_TRANS)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_TRANS_OP);
		goto Exit;
	}

	// See if the transaction needs to be aborted

	if (RC_BAD( rc = m_AbortRc))
	{
		goto Exit;
	}

	// The number must be greater than 1

	if (uiDictNumber < 2)
	{
		goto Exit;
	}

	// Set the next dictionary number

	if (RC_BAD( rc = m_pDict->setNextDictNum( this, uiDictType, uiDictNumber)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:
*****************************************************************************/
RCODE F_Database::startMaintThread( void)
{
	RCODE			rc = NE_XFLM_OK;
	char			szThreadName[ F_PATH_MAX_SIZE];
	char			szBaseName[ 32];

	flmAssert( !m_pMaintThrd);
	flmAssert( m_hMaintSem == F_SEM_NULL);

	// Generate the thread name

	if( RC_BAD( rc = gv_XFlmSysData.pFileSystem->pathReduce( 
		m_pszDbPath, szThreadName, szBaseName)))
	{
		goto Exit;
	}

	f_sprintf( (char *)szThreadName, "Maintenance (%s)", (char *)szBaseName);

	// Create the maintenance semaphore

	if( RC_BAD( rc = f_semCreate( &m_hMaintSem)))
	{
		goto Exit;
	}

	// Start the thread.

	if( RC_BAD( rc = gv_XFlmSysData.pThreadMgr->createThread( &m_pMaintThrd,
		F_Database::maintenanceThread, szThreadName,
		0, 0, this, NULL, 32000)))
	{
		goto Exit;
	}

	// Signal the thread to check for any queued work

	f_semSignal( m_hMaintSem);

Exit:

	if( RC_BAD( rc))
	{
		if( m_hMaintSem != F_SEM_NULL)
		{
			f_semDestroy( &m_hMaintSem);
		}
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_Db::beginBackgroundTrans(
	IF_Thread *		pThread)
{
	RCODE		rc = NE_XFLM_OK;

RetryLock:

	// Obtain the file lock

	flmAssert( !(m_uiFlags & FDB_HAS_FILE_LOCK));

	if( RC_BAD( rc = m_pDatabase->m_pDatabaseLockObj->lock( m_hWaitSem,
		TRUE, FLM_NO_TIMEOUT, FLM_BACKGROUND_LOCK_PRIORITY,
		m_pDbStats ? &m_pDbStats->LockStats : NULL)))
	{
		if( rc == NE_XFLM_DATABASE_LOCK_REQ_TIMEOUT)
		{
			// This would only happen if we were signaled to shut down.
			// So, it's ok to exit

			flmAssert( pThread->getShutdownFlag());
		}
		goto Exit;
	}

	// The lock needs to be marked as implicit so that commitTrans
	// will unlock the database and allow the next update transaction to
	// begin before all writes are complete.

	m_uiFlags |= (FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_IMPLICIT);

	// If there are higher priority waiters in the lock queue,
	// we want to relinquish.

	if( m_pDatabase->m_pDatabaseLockObj->haveHigherPriorityWaiter(
			FLM_BACKGROUND_LOCK_PRIORITY))
	{
		if( pThread->getShutdownFlag())
		{
			rc = RC_SET( NE_XFLM_DATABASE_LOCK_REQ_TIMEOUT);
			goto Exit;
		}
		
		if( RC_BAD( rc = m_pDatabase->m_pDatabaseLockObj->unlock()))
		{
			goto Exit;
		}

		m_uiFlags &= ~(FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_IMPLICIT);
		goto RetryLock;
	}

	// If we are shutting down, relinquish and exit.

	if( pThread->getShutdownFlag())
	{
		rc = RC_SET( NE_XFLM_DATABASE_LOCK_REQ_TIMEOUT);
		goto Exit;
	}

	// Start an update transaction

	if( RC_BAD( rc = beginTrans( 
		XFLM_UPDATE_TRANS, FLM_NO_TIMEOUT, XFLM_DONT_POISON_CACHE)))
	{
		if( rc == NE_XFLM_DATABASE_LOCK_REQ_TIMEOUT)
		{
			// This would only happen if we were signaled to shut down.
			// So, it's ok to exit

			flmAssert( pThread->getShutdownFlag());
		}
		
		goto Exit;
	}

Exit:

	if( RC_BAD( rc))
	{
		if( m_uiFlags & FDB_HAS_FILE_LOCK)
		{
			(void)m_pDatabase->m_pDatabaseLockObj->unlock();
			m_uiFlags &= ~(FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_IMPLICIT);
		}
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI F_Database::maintenanceThread(
	IF_Thread *		pThread)
{
	RCODE				rc = NE_XFLM_OK;
	F_Database *	pDatabase = (F_Database *)pThread->getParm1();
	F_Db *			pDb = NULL;
	F_DOMNode *		pDoc = NULL;
	F_DOMNode *		pNextDoc = NULL;
	FLMUINT64		ui64DocId;
	FLMUINT64		ui64TmpTransId;
	FLMUINT64		ui64SweepTransId;
	FLMUINT			uiNameId;
	FLMBOOL			bStartedTrans;
	FLMBOOL			bShutdown;

Retry:
	
	bStartedTrans = FALSE;
	bShutdown = FALSE;

	pThread->setThreadStatus( FLM_THREAD_STATUS_INITIALIZING);

	if( RC_BAD( rc = gv_pXFlmDbSystem->internalDbOpen( pDatabase, &pDb)))
	{
		// If the file is being closed, this is not an error.

		if( pDatabase->getFlags() & DBF_BEING_CLOSED)
		{
			rc = NE_XFLM_OK;
			bShutdown = TRUE;
		}

		goto Exit;
	}

	for( ;;)
	{
		pThread->setThreadStatus( FLM_THREAD_STATUS_RUNNING);
		ui64DocId = 0;
		
		for( ;;)
		{
			if( RC_BAD( rc = pDb->beginBackgroundTrans( pThread)))
			{
				goto Exit;
			}
			bStartedTrans = TRUE;
				
			if( RC_BAD( pDb->getDocument( 
				XFLM_MAINT_COLLECTION, XFLM_INCL, ui64DocId, (IF_DOMNode **)&pDoc)))
			{
				break;
			}
			
			ui64DocId = pDoc->getDocumentId();

			if( RC_BAD( rc = pDoc->getNameId( pDb, &uiNameId)))
			{
				goto Exit;
			}

			if( uiNameId == ELM_DELETE_TAG)
			{
				if( RC_BAD( rc = pDb->maintBlockChainFree(
					ui64DocId, 25, 0, NULL)))
				{
					goto Exit;
				}

				bStartedTrans = FALSE;
				if( RC_BAD( rc = pDb->commitTrans( 0, FALSE)))
				{
					goto Exit;
				}
			}
			else if( uiNameId == ELM_SWEEP_TAG)
			{
				ui64SweepTransId = pDb->getTransID();
				pDb->abortTrans();
				bStartedTrans = FALSE;
				
				if( RC_BAD( rc = pDb->sweep( pThread)))
				{
					goto Exit;
				}
				
				// Delete the sweep documents from the tracker
				
				if( RC_BAD( rc = pDb->beginBackgroundTrans( pThread)))
				{
					goto Exit;
				}
				bStartedTrans = TRUE;

				for( ;;)
				{
					if( RC_BAD( rc = pDoc->getNextDocument( pDb, 
						(IF_DOMNode **)&pNextDoc)))
					{
						if( rc != NE_XFLM_DOM_NODE_NOT_FOUND)
						{
							RC_UNEXPECTED_ASSERT( rc);
							goto Exit;
						}

						rc = NE_XFLM_OK;
						break;
					}
					
					if( RC_BAD( rc = pDoc->getNameId( pDb, &uiNameId)))
					{
						goto Exit;
					}
					
					if( uiNameId == ELM_SWEEP_TAG)
					{
						if( RC_BAD( rc = pDoc->getAttributeValueUINT64( 
							pDb, ATTR_TRANSACTION_TAG, &ui64TmpTransId, 0)))
						{
							goto Exit;
						}
						
						if( ui64TmpTransId > ui64SweepTransId)
						{
							break;
						}
						
						if( RC_BAD( rc = pDoc->removeModeFlags( 
							pDb, FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
						{
							goto Exit;
						}
						
						if( RC_BAD( rc = pDoc->deleteNode( pDb)))
						{
							goto Exit;
						}
					}

					pDoc->Release();
					
					// Use the reference on pNextDoc and set it to NULL so that
					// it doesn't get released.
					
					pDoc = pNextDoc;
					pNextDoc = NULL;
				}

				bStartedTrans = FALSE;
				if( RC_BAD( rc = pDb->commitTrans( 0, FALSE)))
				{
					goto Exit;
				}
			}
			else
			{
				flmAssert( bStartedTrans);
				pDb->abortTrans();
				bStartedTrans = FALSE;
			}

			ui64DocId++;
		}

		if( bStartedTrans)
		{
			pDb->abortTrans();
			bStartedTrans = FALSE;
		}

		pThread->setThreadStatus( FLM_THREAD_STATUS_SLEEPING);
		f_semWait( pDatabase->m_hMaintSem, F_WAITFOREVER);
			
		if( pThread->getShutdownFlag())
		{
			bShutdown = TRUE;
			goto Exit;
		}
	}

Exit:

	pThread->setThreadStatus( FLM_THREAD_STATUS_TERMINATING);

	if( pDoc)
	{
		pDoc->Release();
		pDoc = NULL;
	}
	
	if( pNextDoc)
	{
		pNextDoc->Release();
		pNextDoc = NULL;
	}

	if( bStartedTrans)
	{
		pDb->abortTrans();
	}

	if( pDb)
	{
		pDb->Release();
		pDb = NULL;
	}

	if( !bShutdown)
	{
		flmAssert( RC_BAD( rc));
		f_sleep( 250);
		f_semSignal( pDatabase->m_hMaintSem);
		goto Retry;
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_Db::maintBlockChainFree(
	FLMUINT64		ui64MaintDocID,
	FLMUINT 			uiBlocksToFree,
	FLMUINT			uiExpectedEndAddr,
	FLMUINT *		puiBlocksFreed)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiTmp;
	FLMUINT			uiBlocksFreed = 0;
	FLMUINT			uiStartAddr = 0;
	FLMUINT			uiEndAddr = 0;
	F_DOMNode *		pDoc = NULL;
	F_DOMNode *		pChainNode = NULL;
	F_DOMNode *		pAddrNode = NULL;
	FLMUINT			uiRflToken = 0;

	// Make sure an update transaction is going and that a
	// non-zero number of blocks was specified

	if( getTransType() != XFLM_UPDATE_TRANS || !uiBlocksToFree)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

	// Retrieve the maintenance document

	if( RC_BAD( rc = getNode( XFLM_MAINT_COLLECTION, 
		ui64MaintDocID, XFLM_EXACT, &pDoc)))
	{
		goto Exit;
	}

	m_pDatabase->m_pRfl->disableLogging( &uiRflToken);

	while( uiBlocksFreed < uiBlocksToFree)
	{
		if( RC_BAD( rc = pDoc->getChildElement( 
			this, ELM_BLOCK_CHAIN_TAG, (IF_DOMNode **)&pChainNode)))
		{
			if( rc != NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				goto Exit;
			}

			if( RC_BAD( rc = pDoc->removeModeFlags( 
				this, FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pDoc->deleteNode( this)))
			{
				goto Exit;
			}

			break;
		}

		if( RC_BAD( rc = pChainNode->getAttributeValueUINT( 
			this, ATTR_ADDRESS_TAG, &uiStartAddr, 0)))
		{
			goto Exit;
		}

		if( !uiStartAddr)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			goto Exit;
		}

		if( RC_BAD( rc = btFreeBlockChain( 
			this, NULL, uiStartAddr, uiBlocksToFree - uiBlocksFreed, 
			&uiTmp, &uiEndAddr, NULL)))
		{
			goto Exit;
		}

		uiBlocksFreed += uiTmp;
		flmAssert( uiBlocksFreed <= uiBlocksToFree);

		if( RC_BAD( rc = pChainNode->removeModeFlags( 
			this, FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
		{
			goto Exit;
		}

		if( !uiEndAddr)
		{
			if( RC_BAD( rc = pChainNode->deleteNode( this)))
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = pChainNode->getAttribute( 
				this, ATTR_ADDRESS_TAG, (IF_DOMNode **)&pAddrNode)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pAddrNode->removeModeFlags( 
				this, FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pAddrNode->setUINT( this, uiEndAddr)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pAddrNode->addModeFlags( 
				this, FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pChainNode->addModeFlags( 
				this, FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
			{
				goto Exit;
			}
		}

		if( RC_BAD( rc = documentDone( 
			XFLM_MAINT_COLLECTION, ui64MaintDocID)))
		{
			goto Exit;
		}
	}

	if( uiExpectedEndAddr)
	{
		if( uiBlocksToFree != uiBlocksFreed ||
			uiEndAddr != uiExpectedEndAddr)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			goto Exit;
		}
	}

	if ( uiRflToken)
	{
		m_pDatabase->m_pRfl->enableLogging( &uiRflToken);
	}

	if( RC_BAD( rc = m_pDatabase->m_pRfl->logBlockChainFree( 
		this, ui64MaintDocID, uiStartAddr, uiEndAddr, uiBlocksFreed)))
	{
		goto Exit;
	}

	if( puiBlocksFreed)
	{
		*puiBlocksFreed = uiBlocksFreed;
	}

Exit:

	if ( uiRflToken)
	{
		m_pDatabase->m_pRfl->enableLogging( &uiRflToken);
	}

	if( pChainNode)
	{
		pChainNode->Release();
	}

	if( pAddrNode)
	{
		pAddrNode->Release();
	}

	if( pDoc)
	{
		pDoc->Release();
	}

	return( rc);
}
