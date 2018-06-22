//-------------------------------------------------------------------------
// Desc:	Dictionary access routiones.
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

FSTATIC RCODE fdictAddDictIndex(
	TDICT *	 		pTDict);

FSTATIC RCODE DDFieldParse(
	TDICT *			pTDict,
	DDENTRY *		pDDEntry,
	FlmRecord *		pRecord,
	FLMUINT			uiDictRecNum);
	
FSTATIC RCODE DDGetReference(
	FlmRecord *		pRecord,	
	void *			pvField,
	const char *	pszBuffer,
	FLMUINT *		puiIdRef);
	
FSTATIC RCODE DDAllocEntry(
	TDICT *			pTDict,
	FlmRecord *		pRecord,
	FLMUINT			uiDictRecNum,
	DDENTRY **		ppDDEntryRV);

FSTATIC RCODE DDIxParse(
	TDICT *			pTDict,
	DDENTRY *		pDDEntry,
	FlmRecord *		pRecord,
	void *			pvField);
	
FSTATIC RCODE DDBuildFldPath(
	TDICT *			pTDict,
	TIFD **			ppTIfd,
	FlmRecord *		pRecord,
	void *			pvField,
	FLMUINT			uiBaseNum);
	
FSTATIC FLMBOOL DDMoveWord(
	char *			pucDest,
	char *			pucSrc,
	FLMUINT			uiMaxDestLen,
	FLMUINT *		puiPos);
	
FSTATIC RCODE DDContainerParse(
	TDICT *			pTDict,
	DDENTRY *		pDDEntry,
	FlmRecord *		pRecord);

FSTATIC void DDTextToNative(
	FlmRecord *		pRecord,
	void *			pvField,
	char *			pszBuffer,
	FLMUINT			uiBufLen,
	FLMUINT *		puiBufLen);

FSTATIC RCODE DDParseStateOptions(
	FlmRecord *		pRecord,
	void *			pvField,
	FLMUINT *		puiFldInfo);

FSTATIC RCODE	DDEncDefParse(
	TDICT *			pTDict,
	DDENTRY *		pDDEntry,
	FlmRecord *		pRecord);

FSTATIC RCODE DDGetEncKey(
	TDICT *			pTDict,
	FlmRecord *		pRecord,
	void *			pvField,
	TENCDEF *		pTEncDef);
	
FSTATIC RCODE DDMakeDictIxKey(
	FDB *				pDb,
	FlmRecord *		pRecord,
	FLMBYTE *		pKeyBuf,
	FLMUINT  *		puiKeyLenRV);

FSTATIC RCODE DDCheckNameConflict(
	FDB *				pDb,
	LFILE *			pDictIxLFile,
	FlmRecord *		pNewRec,
	FLMUINT			uiDrn,
	FlmRecord *		pOldRec);
	
FSTATIC RCODE DDCheckIDConflict(
	FDB *				pDb,
	LFILE *			pDictContLFile,
	FLMUINT			uiDrn);

FSTATIC RCODE DDIxDictRecord(
	FDB *				pDb,
	LFILE *			pDictIxLFile,
	FLMUINT			uiDrn,
	FlmRecord *		pRecord,
	FLMUINT			uiFlags);

FSTATIC void fdictFixupPointers(
	FDICT *			pNewDict,
	FDICT *			pOldDict);

FSTATIC RCODE fdictReallocAllTables(
	TDICT *			pTDict);

FSTATIC RCODE fdictReallocTbl(
	FLMUINT			uiElementSize,
	FLMUINT			uiTblSize,
	FLMUINT			uiAddElements,
	void **			ppvTblRV);

FSTATIC void fdictAddItem(
	TDICT *			pTDict,
	FLMUINT			uiFieldNum,
	FLMUINT			uiFieldType);

FSTATIC RCODE	fdictAddIndex(
	TDICT *			pTDict,
	DDENTRY *		pEntry);

FSTATIC RCODE fdictFixupIfdPointers(
	FDICT *			pDict,
	FLMUINT			uiIfdStartOffset);

FSTATIC RCODE fdictAddNewCCS(
	TDICT *			pTDict,
	TENCDEF *		pTEncDef,
	FLMUINT			uiRecNum);
	
#define FDD_MAX_VALUE_SIZE 		64
#define MAX_ENC_TYPES				3

#define START_DD_INDEX_OPTS  		0
#define DD_IX_FIELD_OPT				0
#define DD_IX_COMPOUND_OPT			1
#define DD_IX_UPPER_OPT				2
#define DD_IX_EACHWORD_OPT			3
#define DD_IX_MIXED_OPT				4
#define DD_IX_CONTEXT_OPT			5
#define DD_IX_POST_OPT				6
#define MAX_DD_INDEX_OPTS    		7

// NOTE:  If you change the arrangement of the values in this array, make sure
// you search the entire codebase for references to DDEncOpts and DDGetEncType
// and verify that the changes won't cause problems.  This is particularly
// important because these values DO NOT match up exactly with the values in
// the SMEncryptionScheme enum that's used at the SMI level.

const char * DDEncOpts[ MAX_ENC_TYPES] =
{
	"aes",
	"des3",
	"des"
};

FSTATIC POOL_STATS g_TDictPoolStats = {0,0};

/*******************************************************************************
Desc:		Retrieves a dictionary item name.
Notes:	Given an item ID, this routine will search a specified shared or
		 	local dictionary for the item.  If it is found, the name
			of the item will be returned.  This routine supports version 2.0 and
			higher databases only.
*******************************************************************************/
FLMEXP RCODE FLMAPI FlmGetItemName(
	HFDB			hDb,
	FLMUINT		uiItemId,
	FLMUINT		uiNameBufSize,
	char *		pszNameBuf)
{
	RCODE				rc = FERR_OK;
	FlmRecord *		pRecord = NULL;

	*pszNameBuf = 0;
	if( RC_BAD( rc = FlmRecordRetrieve( hDb,
		FLM_DICT_CONTAINER, uiItemId, FO_EXACT, &pRecord, NULL)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pRecord->getNative( pRecord->root(), 
		pszNameBuf, &uiNameBufSize)))
	{
		goto Exit;
	}

Exit:

	if( pRecord)
	{
		pRecord->Release();
	}

	return (rc == FERR_EOF_HIT) ? RC_SET( FERR_NOT_FOUND) : rc;
}

/****************************************************************************
Desc:  	Read all data dictionary records parsing and sending to process.
			All temporary structures are off of pTDict.  pTDict must be setup.
****************************************************************************/
RCODE fdictProcessAllDictRecs(
	FDB *			pDb,
	TDICT *		pTDict)
{
	RCODE			rc;
	LFILE *		pLFile = pTDict->pLFile;
	BTSK			stackBuf[ BH_MAX_LEVELS ];	// Stack to hold b-tree variables
	BTSK *		stack = stackBuf;		 		// Points to proper stack frame
	FLMBYTE		btKeyBuf[ DRN_KEY_SIZ +8];	// Key buffer pointed to by stack
	FLMBYTE		key[4];					 		// Used for dummy empty key
	FLMUINT		uiDrn;
	FlmRecord *	pRecord = NULL;

	// Add the dictionary index to the front of TDICT.
	
	if( RC_BAD( rc = fdictAddDictIndex( pTDict)))
	{
		goto Exit;
	}

	// Position to the first of the data dictionary data records & read.
	
	FSInitStackCache( &stackBuf [0], BH_MAX_LEVELS);
	stack->pKeyBuf = btKeyBuf;
	f_UINT32ToBigEndian( 0, key);
	
	if( RC_BAD(rc = FSBtSearch( pDb, pLFile, &stack, key, DRN_KEY_SIZ, 0 )))
	{
		goto Exit;
	}

	// Special case of no records.
	if( stack->uiCmpStatus == BT_END_OF_DATA)	
		goto Exit;
	stack->uiFlags = NO_STACK;					// Fake out the stack for speed.

	do
	{
		uiDrn = f_bigEndianToUINT32( btKeyBuf);
		
		if( uiDrn == DRN_LAST_MARKER)
		{
			break;
		}
		
		// VERY IMPORTANT NOTE:
		//  	DO NOT READ FROM CACHE - THE RECORD MAY
		// 	NOT HAVE BEEN PUT INTO RECORD CACHE YET, AND WE NEED TO HAVE
		// 	THE CORRECT VERSION OF THE RECORD.

		if( RC_BAD( rc = FSReadElement( pDb, &pDb->TempPool, pLFile, 
			uiDrn, stack, TRUE, &pRecord, NULL, NULL)))
		{
			break;
		}

		if( RC_BAD(rc = fdictProcessRec( pTDict, pRecord, uiDrn)))
		{
			pDb->Diag.uiDrn = uiDrn;
			pDb->Diag.uiInfoFlags |= FLM_DIAG_DRN;
			if( pTDict->uiBadField != 0)
			{
				pDb->Diag.uiFieldNum = pTDict->uiBadField;
				pDb->Diag.uiInfoFlags |= FLM_DIAG_FIELD_NUM;
			}
			break;
		}

		// Position to the next record - SUCCESS or FERR_BT_END_OF_DATA
		rc = FSNextRecord( pDb, pLFile, stack);

	} while( RC_OK(rc));

	rc = (rc == FERR_BT_END_OF_DATA) ? FERR_OK : rc;

Exit:

	if( pRecord)
	{
		pRecord->Release();
	}

	FSReleaseStackCache( stackBuf, BH_MAX_LEVELS, FALSE);
	return( rc );
}

/****************************************************************************
Desc:		Add the dictionary index to pTDict.
****************************************************************************/
RCODE fdictAddDictIndex(
	TDICT *	 		pTDict)
{
	RCODE				rc;
	DDENTRY *		pDDEntry;
	TIXD *			pTIxd;
	TIFD *			pTIfd;
	TIFP *			pTIfp;

	if( RC_BAD( rc = DDAllocEntry( pTDict, NULL, FLM_DICT_INDEX, &pDDEntry)))
	{
		goto Exit;
	}
	pDDEntry->uiType = ITT_INDEX_TYPE;
	
	if( RC_BAD( rc = pTDict->pool.poolAlloc( sizeof( TIXD), (void **)&pTIxd)))
	{
		goto Exit;
	}

	pTDict->uiNewIxds++;
	pDDEntry->vpDef = (void *) pTIxd;
	pTIxd->uiFlags = IXD_UNIQUE;
	pTIxd->uiContainerNum = FLM_DICT_CONTAINER;
	pTIxd->uiNumFlds = 1;
	pTIxd->uiLanguage = pTDict->uiDefaultLanguage;
	pTIxd->uiEncId = 0;

	if( RC_BAD( rc = pTDict->pool.poolAlloc( sizeof( TIFD), (void **)&pTIfd)))
	{
		goto Exit;
	}
	
	pTIxd->pNextTIfd = pTIfd;
	pTDict->uiNewIfds++;
	pTIfd->pTIfp = NULL;
	pTIfd->pNextTIfd = NULL;
	pTIfd->uiFlags = (FLMUINT)(IFD_FIELD | FLM_TEXT_TYPE);
	pTIfd->uiNextFixupPos = 0;
	pTIfd->uiLimit = IFD_DEFAULT_LIMIT;
	pTIfd->uiCompoundPos = 0;

	if( RC_BAD( rc = pTDict->pool.poolAlloc( sizeof( TIFP), (void **)&pTIfp)))
	{
		goto Exit;
	}
	
	pTDict->uiNewFldPaths += 2;
	pTIfd->pTIfp = pTIfp;

	pTIfp->pNextTIfp = NULL;
	pTIfp->bFieldInThisDict = FALSE;
	pTIfp->uiFldNum = FLM_NAME_TAG;

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Process a single data dictionary record. Parse the record for syntax
			errors depending on flag value.  Only supports adding new stuff
			to pTDict.
****************************************************************************/
RCODE fdictProcessRec(
	TDICT *	 		pTDict,
	FlmRecord *		pRecord,
	FLMUINT			uiDictRecNum)
{
	RCODE      		rc = FERR_OK;
	DDENTRY *		pDDEntry;
	void *			pvField = pRecord->root();

	// Ignore items with root nodes that are in the unregistered range.

	if( pRecord->getFieldID( pvField) >= FLM_UNREGISTERED_TAGS)
	{
		goto Exit;
	}

	// Parse only on modify or add.

	switch( pRecord->getFieldID( pvField))
	{
		case FLM_FIELD_TAG:
		{
			if( RC_BAD( rc = DDAllocEntry( 
					pTDict, pRecord, uiDictRecNum, &pDDEntry)))
			{
				goto Exit;
			}

			pDDEntry->uiType = 0;				// Type of zero means field.
			if( RC_BAD( rc = DDFieldParse( pTDict, pDDEntry, 
							pRecord, uiDictRecNum)))
			{
				goto Exit;
			}
			break;
		}

		case FLM_INDEX_TAG:
		{
			if( RC_BAD( rc = DDAllocEntry( 
					pTDict, pRecord, uiDictRecNum, &pDDEntry)))
			{
				goto Exit;
			}
			pDDEntry->uiType = ITT_INDEX_TYPE;
			if( RC_BAD( rc = DDIxParse( pTDict, pDDEntry, pRecord, pvField)))
			{
				goto Exit;
			}
			pTDict->uiNewIxds++;
			break;
		}

		case FLM_CONTAINER_TAG:
		{
			if( RC_BAD( rc = DDAllocEntry( 
					pTDict, pRecord, uiDictRecNum, &pDDEntry)))
			{
				goto Exit;
			}
			pDDEntry->uiType = ITT_CONTAINER_TYPE;
			if( RC_BAD( rc = DDContainerParse( pTDict, pDDEntry, pRecord)))
			{
				goto Exit;
			}
			pTDict->uiTotalLFiles++;
			break;
		}
		case FLM_ENCDEF_TAG:
		{
			if( RC_BAD( rc = DDAllocEntry(
					pTDict, pRecord, uiDictRecNum, &pDDEntry)))
			{
				goto Exit;
			}
			pDDEntry->uiType = ITT_ENCDEF_TYPE;
			if (RC_BAD( rc = DDEncDefParse( pTDict, pDDEntry, pRecord)))
			{
				goto Exit;
			}
			break;
		}
		case FLM_AREA_TAG:
		case FLM_RESERVED_TAG:
		{
			break;
		}

		default:
		{
			// Cannot allow anything else to pass through the dictionary.
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}
	}

Exit:
	return( rc);
}

/****************************************************************************
Desc:		Allocate, check and add a name to the DDEntry structure.
****************************************************************************/
FSTATIC RCODE DDAllocEntry(
	TDICT *			pTDict,
	FlmRecord *		pRecord,
	FLMUINT			uiDictRecNum, 
	DDENTRY **		ppDDEntryRV)
{
	RCODE				rc = FERR_OK;
	DDENTRY *		pNewEntry;

	if( RC_BAD( rc = pTDict->pool.poolAlloc( sizeof( DDENTRY), 
		(void **)&pNewEntry)))
	{
		goto Exit;
	}
	
	pNewEntry->pNextEntry = NULL;
	pNewEntry->vpDef = NULL;
	pNewEntry->uiEntryNum = uiDictRecNum;
	pNewEntry->uiType = 0;

	// Zero length name NOT allowed for dictionary items.

	if( pRecord)
	{
		if( pRecord->getDataLength( pRecord->root()) == 0)
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}
	}

	if( pTDict->pLastEntry)
	{
		pTDict->pLastEntry->pNextEntry = pNewEntry;
	}
	else
	{
		pTDict->pFirstEntry = pNewEntry;
	}
	pTDict->pLastEntry = pNewEntry;
	*ppDDEntryRV = pNewEntry;

Exit:

	return( rc );
}

/****************************************************************************
Desc:		Parse field definition
****************************************************************************/
FSTATIC RCODE DDFieldParse(
	TDICT *			pTDict,
	DDENTRY *		pDDEntry,
	FlmRecord *		pRecord,
	FLMUINT			uiDictRecNum)
{
	RCODE    		rc = FERR_OK;
	TFIELD  *		pTField;
	void *			pvField = NULL;

	if( RC_BAD( rc = pTDict->pool.poolAlloc( sizeof( TFIELD), 
		(void **)&pTField)))
	{
		goto Exit;
	}
	
	pTField->uiFldNum = uiDictRecNum;
	pTField->uiFldInfo = FLM_CONTEXT_TYPE;
	pDDEntry->vpDef = (void *) pTField;

	if( (pvField = pRecord->firstChild( pRecord->root())) == NULL)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	for( ; pvField; pvField = pRecord->nextSibling( pvField))
	{
		switch( pRecord->getFieldID( pvField))
		{
			case FLM_TYPE_TAG:
			{
				rc = DDGetFieldType( pRecord, pvField, &pTField->uiFldInfo);
				break;
			}

			case FLM_STATE_TAG:
			{
				rc = DDParseStateOptions( pRecord, pvField, &pTField->uiFldInfo);
				break;
			}

			default:
			{
				if( pRecord->getFieldID( pvField) < FLM_UNREGISTERED_TAGS &&
					 pRecord->getFieldID( pvField) != FLM_COMMENT_TAG)
				{
					rc = RC_SET( FERR_SYNTAX);
				}
				break;
			}
		}
	}

Exit:

	if( RC_BAD(rc) && pvField)
	{
		pTDict->uiBadField = pRecord->getFieldID( pvField);
	}
	return( rc );
}

/****************************************************************************
Desc:		Returns the fields data type.  May be called outside of DDPREP.C
****************************************************************************/
RCODE DDGetFieldType(
	FlmRecord *		pRecord,
	void *			pvField,
	FLMUINT *		puiFldInfo)
{
	RCODE				rc = FERR_OK;
	char				szNativeBuf[ FDD_MAX_VALUE_SIZE];	

	DDTextToNative( pRecord, pvField, szNativeBuf, FDD_MAX_VALUE_SIZE, NULL );

	// Parse the type keyword - only one type allowed.

	if (f_strnicmp( szNativeBuf, "text", 4) == 0)
	{
		*puiFldInfo = FLM_TEXT_TYPE;
	}
	else if (f_strnicmp( szNativeBuf, "numb", 4) == 0)
	{
		*puiFldInfo = FLM_NUMBER_TYPE;
	}
	else if (f_strnicmp( szNativeBuf, "bina", 4) == 0)
	{
		*puiFldInfo = FLM_BINARY_TYPE;
	}
	else if (f_strnicmp( szNativeBuf, "cont", 4) == 0)
	{
		*puiFldInfo = FLM_CONTEXT_TYPE;
	}
	else if (f_strnicmp( szNativeBuf, "blob", 4) == 0)
	{
		*puiFldInfo = FLM_BLOB_TYPE;
	}
	else
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

Exit:
	return( rc);
}

/****************************************************************************
Desc:		Parses the 'state' option that is found within the 'field'
			dictionary definition.
Format: 	state [checking | unused | purge | active]
****************************************************************************/
FSTATIC RCODE DDParseStateOptions(
	FlmRecord *	pRecord,
	void *		pvField,
	FLMUINT *	puiFldInfo)
{
	RCODE			rc = FERR_OK;
	char			szNativeBuf[ FDD_MAX_VALUE_SIZE];

	DDTextToNative( pRecord, pvField, szNativeBuf, FDD_MAX_VALUE_SIZE, NULL);

	// Parse the 'state' keyword - only one type allowed

	if( f_strnicmp( szNativeBuf, "chec", 4) == 0)
	{
		// 0xFFCF is used to clear out any existing field 'state' value
		
		*puiFldInfo = (FLMUINT)((*puiFldInfo & ~ITT_FLD_STATE_MASK) | ITT_FLD_STATE_CHECKING);
	}
	else if( f_strnicmp( szNativeBuf, "unus", 4) == 0)
	{
		*puiFldInfo = (FLMUINT)((*puiFldInfo & ~ITT_FLD_STATE_MASK) | ITT_FLD_STATE_UNUSED);
	}
	else if( f_strnicmp( szNativeBuf, "purg", 4) == 0)
	{
		*puiFldInfo = (FLMUINT)((*puiFldInfo & ~ITT_FLD_STATE_MASK) | ITT_FLD_STATE_PURGE);
	}
	else if( f_strnicmp( szNativeBuf, "acti", 4) == 0)
	{
		*puiFldInfo = (FLMUINT)((*puiFldInfo & ~ITT_FLD_STATE_MASK) | ITT_FLD_STATE_ACTIVE);
	}
	else
	{
		rc = RC_SET( FERR_SYNTAX);
	}

	return( rc);
}

/****************************************************************************
Desc:		Get a number reference and set in the (OUT) parameter.
****************************************************************************/
FSTATIC RCODE DDGetReference(
	FlmRecord *		pRecord,
	void *			pvField,
	const char *	pszBuffer,
	FLMUINT *		puiIdRef)
{
	RCODE			rc = FERR_OK;

	*puiIdRef = 0;
	if( pszBuffer)
	{
		if( !(*pszBuffer))
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}

		*puiIdRef = f_atoud( pszBuffer);
	}
	else
	{
		if( RC_BAD( rc = pRecord->getUINT( pvField, puiIdRef))) 
		{
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}


/****************************************************************************
Desc:		Returns the encryption type.  May be called outside of DDPREP.C
****************************************************************************/
RCODE DDGetEncType(
	FlmRecord *		pRecord,
	void *			pvField,
	FLMUINT *		puiFldInfo)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiType;
	char				szNativeBuf[ FDD_MAX_VALUE_SIZE];	

	DDTextToNative( pRecord, pvField, szNativeBuf, FDD_MAX_VALUE_SIZE, NULL );

	// Parse the type keyword - only one type allowed.

	for( uiType = 0;
		  uiType < MAX_ENC_TYPES ;
		  uiType++)
	{
		if( f_strnicmp( szNativeBuf, DDEncOpts[ uiType],
					f_strlen(DDEncOpts[ uiType])) == 0)
		{
			*puiFldInfo = uiType;
			goto Exit;
		}
	}

	rc = RC_SET( FERR_SYNTAX);

Exit:
	return( rc);
}

/****************************************************************************
Desc:		Returns the binary key info.  May be called outside of DDPREP.C
****************************************************************************/
FSTATIC RCODE DDGetEncKey(
	TDICT *			pTDict,
	FlmRecord *		pRecord,
	void *			pvField,
	TENCDEF *		pTEncDef)
{
	RCODE				rc = FERR_OK;
	char *			pucBuffer = NULL;
	FLMUINT			uiLength;

	pTEncDef->uiLength = 0;

	if (RC_BAD( rc = pRecord->getNativeLength( pvField, &uiLength)))
	{
		goto Exit;
	}
	uiLength++;

	if( RC_BAD( rc = pTDict->pool.poolAlloc( uiLength, 
		(void **)&pucBuffer)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = pRecord->getNative( pvField, pucBuffer, &uiLength)))
	{
		goto Exit;
	}

	pTEncDef->uiLength = uiLength;
	pTEncDef->pucKeyInfo = (FLMBYTE *)pucBuffer;

Exit:

	return( rc);

}

/****************************************************************************
Desc:		Parse an data dictionary index definition for correct syntax &
			assign the correct attributes.  Build the pcode buffer for the index.
Return:	RCODE - SUCCESS or FERR_SYNTAX
Format:

0 index <psName>						# FLM_INDEX_TAG
[ 1 area [ 0 | <ID>]]				# FLM_AREA_TAG - QF files area, 0 = "same as DB"
[ 1 container {DEFAULT | <ID>}]	# FLM_CONTAINER_TAG - indexes span only one container
[ 1 count [ KEYS &| REFS]]			# FLM_COUNT_TAG - key count of keys and/or refs
[ 1 language {US | <language>}]	# FLM_LANGUAGE_TAG - for full-text parsing and/or sorting
[ 1 positioning]						# FLM_POSITIONING_TAG - full reference counts at all b-tree elements
[ 1 encdef <EncryptionDefId>]		# FLM_ENCDEF_TAG - identify the encryption definition to use

  1 key [EACHWORD]					# FLM_KEY_TAG - 'use' defaults based on type
  [ 2	base <ID>]						# FLM_BASE_TAG - base rec/field for fields below
  [ 2 combinations <below> 	 	# FLM_COMBINATIONS_TAG - how to handle repeating fields
		{ ALL | NORMALIZED}]
  [ 2 post]								# FLM_POST_TAG - case-flags post-pended to key
  [ 2	required*]						# FLM_REQUIRED_TAG - key value is required
  [ 2 unique]							# FLM_UNIQUE_TAG - key has only 1 reference
  { 2 <field> }...					# FLM_FIELD_TAG - compound key if 2 or more
	 [ 3 case mixed | upper]		# FLM_CASE_TAG - text-only, define chars case
	 [ 3 <field>]...					# FLM_FIELD_TAG - alternate field(s)
	 [ 3 paired]						# FLM_PAIRED_TAG - add field ID to key
	 [ 3 optional*						# FLM_OPTIONAL_TAG - component's value is optional
	 | 3 required]						# FLM_REQUIRED_TAG - component's value is required
	 [ 3 use eachword|value|field|minspaces|nounderscore|nospace|nodash] # FLM_USE_TAG

<field> ==
  n field <field path>				#  path identifies field -- maybe "based"
  [ m type <data type>]				# FLM_TYPE_TAG - only for ixing unregistered fields
	
Please Note:	This code only supports the minimal old 11 index format
					needed for skads databases.			
****************************************************************************/
FSTATIC RCODE DDIxParse(
	TDICT *			pTDict,
	DDENTRY *		pDDEntry,					// Points to defined entry.
	FlmRecord *		pRecord,						// Index definition record.
	void *			pvField)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiIfdFlags;
	FLMUINT			uiTempIfdFlags;
	FLMUINT			uiBaseNum;
	FLMUINT			uiNLen;
	TIXD *			pTIxd;
	TIFD * 			pLastTIfd;
	TIFD *			pTIfd;
	void *			pvTempField = NULL;
	void *			pvIfdField = NULL;
	char				szNativeBuf[ 64];
	FLMUINT			uiCompoundPos;
	FLMUINT			uiTemp;
	FLMBOOL			bHasRequiredTag = TRUE;
	FLMBOOL			bOld11Mode = FALSE;

	if( RC_BAD( rc = pTDict->pool.poolAlloc( sizeof( TIXD), 
		(void **)&pTIxd)))
	{
		goto Exit;
	}
	
	pTIxd->pNextTIfd = NULL;
	pTIxd->uiFlags = 0;
	pTIxd->uiContainerNum = FLM_DATA_CONTAINER;
	pTIxd->uiNumFlds = 0;
	pTIxd->uiLanguage = pTDict->uiDefaultLanguage;
	pTIxd->uiEncId = 0;

	if( (pvField = pRecord->firstChild( pRecord->root())) == NULL)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	pLastTIfd = NULL;
	for( ; pvField; pvField = pRecord->nextSibling( pvField))
	{
		switch( pRecord->getFieldID( pvField))
		{
			case FLM_CONTAINER_TAG:
			{
				char 		szTmpBuf [50];
				FLMUINT	uiLen = sizeof( szTmpBuf);
	
				// See if a special keyword is used - ALL or *
	
				if ((pRecord->getDataType( pvField) == FLM_TEXT_TYPE) &&
					 (RC_OK( pRecord->getNative( pvField, szTmpBuf, &uiLen))) &&
					 (f_stricmp( "ALL", szTmpBuf) == 0 ||
					  f_stricmp( "*", szTmpBuf) == 0))
				{
					if (pTDict->pDb->pFile->FileHdr.uiVersionNum < 
						 FLM_FILE_FORMAT_VER_4_50)
					{
						rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
						goto Exit;
					}
	
					// Zero will mean all containers
	
					pTIxd->uiContainerNum = 0;
				}
				else
				{
					if( RC_BAD( rc = DDGetReference(  pRecord, pvField, NULL,
												&pTIxd->uiContainerNum)))
					{
						goto Exit;
					}
					
					if( pTIxd->uiContainerNum == 0)
					{
						pTIxd->uiContainerNum = FLM_DATA_CONTAINER;
					}
				}
				
				break;
			}

			case FLM_COUNT_TAG:
			{
				pTIxd->uiFlags |= IXD_COUNT;
				break;
			}

			case FLM_LANGUAGE_TAG:
			{
				uiNLen = sizeof( szNativeBuf);
				(void) pRecord->getNative( pvField, szNativeBuf, &uiNLen);
				pTIxd->uiLanguage = f_languageToNum( szNativeBuf);
				break;
			}

			case FLM_ENCDEF_TAG:
			{
				uiNLen = sizeof( szNativeBuf);
				(void) pRecord->getNative( pvField, szNativeBuf, &uiNLen);
				pTIxd->uiEncId = f_atoud( szNativeBuf);
				flmAssert( pTIxd->uiEncId);
				break;
			}

			case FLM_TYPE_TAG:
			{
				bOld11Mode = TRUE;
				break;
			}

			case FLM_POSITIONING_TAG:
			{
				if (pTDict->pDb->pFile->FileHdr.uiVersionNum >= 
					 FLM_FILE_FORMAT_VER_4_3)
				{
					pTIxd->uiFlags |= IXD_POSITIONING;
				}
				else
				{
					rc = RC_SET( FERR_SYNTAX);
					goto Exit;
				}
				
				break;
			}

			case FLM_FIELD_TAG:
			{
				uiCompoundPos = 0;
				uiBaseNum = 0;
				uiIfdFlags = IFD_FIELD;
				bHasRequiredTag = TRUE;
				pvTempField = pvField;
				bOld11Mode = TRUE;
				goto Parse_Fields;
			}

			case FLM_KEY_TAG:
			{
				uiCompoundPos = 0;
				uiBaseNum = 0;
				uiIfdFlags = IFD_FIELD | IFD_OPTIONAL;
				bHasRequiredTag = FALSE;
	
				uiNLen = sizeof( szNativeBuf);
				(void) pRecord->getNative( pvField, szNativeBuf, &uiNLen);
	
				if( f_strnicmp( szNativeBuf, "EACH", 4) == 0)
				{
					pTIxd->uiFlags |= IXD_EACHWORD;
					uiIfdFlags = IFD_EACHWORD | IFD_OPTIONAL;
				}
	
				if( (pvTempField = pRecord->firstChild( pvField)) == NULL)
				{
					rc = RC_SET( FERR_SYNTAX);
					goto Exit;
				}
Parse_Fields:			
				for( ; pvTempField; pvTempField = pRecord->nextSibling( pvTempField))
				{
					switch( pRecord->getFieldID( pvTempField))
					{
						case FLM_BASE_TAG:
						{
							if( RC_BAD( rc = DDGetReference( pRecord, 
									pvTempField, NULL, &uiBaseNum)))
							{
								goto Exit;
							}
							break;
						}
		
						case FLM_COMBINATIONS_TAG:
						{
							rc = RC_SET( FERR_SYNTAX);
							goto Exit;
						}
		
						case FLM_POST_TAG:
						{
							pTIxd->uiFlags |= IXD_HAS_POST;
							uiIfdFlags |= IFD_POST;
							break;
						}
		
						case FLM_REQUIRED_TAG:
						{
							break;
						}
		
						case FLM_OPTIONAL_TAG:
						{
							rc = RC_SET( FERR_SYNTAX);
							goto Exit;
						}
		
						case FLM_UNIQUE_TAG:
						{
							pTIxd->uiFlags |= IXD_UNIQUE;
							uiIfdFlags |= IFD_UNIQUE_PIECE;
							break;
						}
		
						case FLM_FIELD_TAG:
						{
							pTIxd->uiNumFlds++;
		
							if( bOld11Mode)
							{
								pvField = pvTempField;
							}
		
							// Need to set IFD_COMPOUND if there is more than one field.
		
							if( pTIxd->uiNumFlds == 1 &&
								(pRecord->find( pvTempField, FLM_FIELD_TAG, 2) != NULL))
							{
								uiIfdFlags |= IFD_COMPOUND;
							}
		
							pTIfd = pLastTIfd;
							if( RC_BAD(rc = DDBuildFldPath( pTDict, &pLastTIfd, 
								pRecord, pvTempField, uiBaseNum)))
							{
								goto Exit;
							}
		
							pLastTIfd->uiCompoundPos = uiCompoundPos++;
		
							if( !pTIfd)
							{
								pTIxd->pNextTIfd = pLastTIfd;
							}
							else
							{
								pTIfd->pNextTIfd = pLastTIfd;
							}
							
							uiTempIfdFlags = uiIfdFlags;
							
							if( bOld11Mode)
							{
								uiTempIfdFlags &= ~IFD_OPTIONAL;
								uiTempIfdFlags |= (IFD_REQUIRED_PIECE | IFD_REQUIRED_IN_SET);
							}
						
							for( pvIfdField = pRecord->firstChild( pvTempField); 
								  pvIfdField; pvIfdField = pRecord->nextSibling( pvIfdField))
							{
								switch ( pRecord->getFieldID( pvIfdField))
								{
									case FLM_CASE_TAG:
									{
										uiNLen = sizeof( szNativeBuf);
										(void) pRecord->getNative( pvIfdField, szNativeBuf, &uiNLen);
			
										if( f_strnicmp( szNativeBuf, "UPPE", 4) == 0)
										{
											uiTempIfdFlags |= IFD_UPPER;
										}
										break;
									}
			
									case FLM_FIELD_TAG:
									{
										break;
									}
			
									case FLM_OPTIONAL_TAG:
									{
										if( bOld11Mode)
										{
											// Old 11 format - default for each field is required.
											uiTempIfdFlags |= IFD_OPTIONAL;
											uiTempIfdFlags &= ~(IFD_REQUIRED_PIECE | IFD_REQUIRED_IN_SET);
										}
										// New format default is optional
										break;
									}
			
									case FLM_PAIRED_TAG:
									{
										uiTempIfdFlags |= IFD_FIELDID_PAIR;
										break;
									}
			
									case FLM_POST_TAG:
									{
										uiTempIfdFlags |= IFD_POST;
										break;
									}
			
									case FLM_REQUIRED_TAG:
									{
										bHasRequiredTag = TRUE;
										uiTempIfdFlags &= ~IFD_OPTIONAL;
										uiTempIfdFlags |= (IFD_REQUIRED_PIECE | IFD_REQUIRED_IN_SET);
										break;
									}
			
									case FLM_LIMIT_TAG:
									{
										if( RC_BAD( pRecord->getUINT( pvIfdField, &uiTemp)) || 
											uiTemp > IFD_DEFAULT_LIMIT)
										{
											pLastTIfd->uiLimit = IFD_DEFAULT_LIMIT;
										}
										else
										{
											pLastTIfd->uiLimit = uiTemp;
										}
										break;
									}
			
									case FLM_UNIQUE_TAG:
									{
										uiTempIfdFlags |= IFD_UNIQUE_PIECE;
										pTIxd->uiFlags |= IXD_UNIQUE;
										break;
									}
			
									case FLM_USE_TAG:
									{
										uiNLen = sizeof( szNativeBuf);
										(void) pRecord->getNative( pvIfdField, szNativeBuf, &uiNLen);
			
										if( f_strnicmp( szNativeBuf, "EACH", 4) == 0)
										{
											uiTempIfdFlags |= IFD_EACHWORD;
											uiTempIfdFlags &= ~(IFD_VALUE|IFD_SUBSTRING);
										}
										else if( f_strnicmp( szNativeBuf, "SUBS", 4) == 0)
										{
											pTIxd->uiFlags |= IXD_HAS_SUBSTRING;
											uiTempIfdFlags |= IFD_SUBSTRING;
											uiTempIfdFlags &= ~(IFD_VALUE|IFD_EACHWORD);
											if( pLastTIfd->uiLimit == IFD_DEFAULT_LIMIT)
											{
												pLastTIfd->uiLimit = IFD_DEFAULT_SUBSTRING_LIMIT;
											}
										}
										else if( f_strnicmp( szNativeBuf, "VALU", 4) == 0)
										{
											uiTempIfdFlags |= IFD_VALUE;
											uiTempIfdFlags &= ~(IFD_EACHWORD|IFD_SUBSTRING);
										}
										else if( f_strnicmp( szNativeBuf, "FIEL", 4) == 0)
										{
											uiTempIfdFlags |= IFD_CONTEXT;
											uiTempIfdFlags &= ~(IFD_VALUE|IFD_EACHWORD|IFD_SUBSTRING);
										}
										break;
									}
											
									case FLM_FILTER_TAG:
									{
										uiNLen = sizeof( szNativeBuf);
										(void) pRecord->getNative( pvIfdField, szNativeBuf, &uiNLen);
			
										if( f_strnicmp( szNativeBuf, "MINS", 4) == 0)
										{
											uiTempIfdFlags |= IFD_MIN_SPACES;
										}
										else if( f_strnicmp( szNativeBuf, "NOUN", 4) == 0)
										{
											uiTempIfdFlags |= IFD_NO_UNDERSCORE;
										}
										else if( f_strnicmp( szNativeBuf, "NOSP", 4) == 0)
										{
											uiTempIfdFlags |= IFD_NO_SPACE;
										}
										else if( f_strnicmp( szNativeBuf, "NODA", 4) == 0)
										{
											uiTempIfdFlags |= IFD_NO_DASH;
										}
										else
										{
											rc = RC_SET( FERR_SYNTAX);
											goto Exit;
										}
										break;
									}
			
									default:
									{
										if( pRecord->getFieldID( pvIfdField) < FLM_UNREGISTERED_TAGS &&
											pRecord->getFieldID( pvIfdField) != FLM_COMMENT_TAG)
										{
											rc = RC_SET( FERR_SYNTAX);
											goto Exit;
										}
										break;
									}
								}
							}
		
							// Parse again the level 3 field definitions.  Now we
							// have the IFD uiFlags value to assign each piece that
							// will have the same compound position.
		
							pLastTIfd->uiFlags |= uiTempIfdFlags;
		
							for( pvIfdField = pRecord->firstChild( pvTempField); 
								  pvIfdField; pvIfdField = pRecord->nextSibling( pvIfdField))
							{
								if( pRecord->getFieldID( pvIfdField) == FLM_FIELD_TAG )
								{
									rc = RC_SET( FERR_SYNTAX);
									goto Exit;
								}
							}
							break;
						}
		
						default:
						{
							if( bOld11Mode)
							{
								break;
							}
		
							if( pRecord->getFieldID( pvTempField) < FLM_UNREGISTERED_TAGS &&
								pRecord->getFieldID( pvTempField) != FLM_COMMENT_TAG)
		
							{
								rc = RC_SET( FERR_SYNTAX);
								goto Exit;
							}
							
							break;
						}
					}
				}

				// Special case for optional
				
				if( !bHasRequiredTag)
				{
					// Set all of the IFD flags to IFD_REQUIRED_IN_SET
					
					for( pTIfd = pTIxd->pNextTIfd; pTIfd; pTIfd = pTIfd->pNextTIfd)
					{
						pTIfd->uiFlags |= IFD_REQUIRED_IN_SET;
					}
				}
				break;
			}

			default:
			{
				if( pRecord->getFieldID( pvField) < FLM_UNREGISTERED_TAGS &&
					pRecord->getFieldID( pvField) != FLM_COMMENT_TAG)
				{
					rc = RC_SET( FERR_SYNTAX);
					goto Exit;
				}
				break;
			}
		}
	}
	pDDEntry->vpDef = (void *) pTIxd;

Exit:

	if( RC_BAD(rc))
	{
		if( pvIfdField)
		{
			pTDict->uiBadField = pRecord->getFieldID( pvIfdField);
		}
		else if( pvTempField)
		{
			pTDict->uiBadField = pRecord->getFieldID( pvTempField);
		}
		else if( pvField)
		{
			pTDict->uiBadField = pRecord->getFieldID( pvField);
		}
	}
	else
	{
		pTDict->uiNewIxds++;
		pTDict->uiNewIfds += pTIxd->uiNumFlds;
		pTDict->uiNewLFiles++;
	}
	
	return( rc);
}

/****************************************************************************
Desc:		Build field path for each index field. This function will also
			check for the existence of the 'batch' option for QF indexes.
****************************************************************************/
FSTATIC RCODE DDBuildFldPath(
	TDICT *			pTDict,
	TIFD **			ppTIfd,
	FlmRecord *		pRecord,
	void *			pvField,
	FLMUINT			uiBaseNum)
{
	RCODE				rc = FERR_OK;
	TIFD *			pTIfd;
	TIFP *			pLastFldPath;
	TIFP *			pTIfp;
	FLMUINT			uiNumInFldPath;
	char				szNameBuf[ 32];
	char *			pszCurrent;
	char				szNativeBuf[ FDD_MAX_VALUE_SIZE];
	FLMUINT			uiBufLen;
	FLMUINT			uiPos;

	pTDict->uiTotalIfds++;
	if( RC_BAD( rc = pTDict->pool.poolAlloc( sizeof( TIFD), (void **)&pTIfd)))
	{
		goto Exit;
	}
	
	pTIfd->pTIfp = NULL;
	pTIfd->pNextTIfd = NULL;
	pTIfd->uiFlags = 0;
	pTIfd->uiNextFixupPos = 0;
	pTIfd->uiLimit = IFD_DEFAULT_LIMIT;
	pTIfd->uiCompoundPos = 0;

	pLastFldPath = NULL;
	*ppTIfd = pTIfd;

	// Build the field paths

	DDTextToNative( pRecord, pvField, szNativeBuf,
		FDD_MAX_VALUE_SIZE, &uiBufLen);

	pszCurrent = szNativeBuf;
	uiNumInFldPath = uiPos = 0;

	if( uiBaseNum)
	{
		uiNumInFldPath++;
		if( RC_BAD( rc = pTDict->pool.poolAlloc( sizeof( TIFP), (void **)&pTIfp)))
		{
			goto Exit;
		}
		
		pTIfp->pNextTIfp = NULL;
		pTIfp->bFieldInThisDict = FALSE;
		pTIfp->uiFldNum = uiBaseNum;
		pTIfd->pTIfp = pTIfp;
		pLastFldPath = pTIfp;
	}

	while( uiPos < uiBufLen)
	{
		uiNumInFldPath++;
		if( DDMoveWord( szNameBuf, pszCurrent, 
				sizeof( szNameBuf ), &uiPos ) == FALSE )
		{
			break;
		}

		if( RC_BAD( rc = pTDict->pool.poolAlloc( sizeof( TIFP), (void **)&pTIfp)))
		{
			goto Exit;
		}
		
		pTIfp->pNextTIfp = NULL;
		pTIfp->bFieldInThisDict = FALSE;

		if( pTIfd->pTIfp == NULL)
		{
			pTIfd->pTIfp = pTIfp;
		}
		else
		{
			pLastFldPath->pNextTIfp = pTIfp;
		}

		pLastFldPath = pTIfp;

		// See if there is a wildcard in the path.

		if (f_stricmp( szNameBuf, "*") == 0)
		{
			if (pTDict->pDb->pFile->FileHdr.uiVersionNum < 
				 FLM_FILE_FORMAT_VER_4_50)
			{
				rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
				goto Exit;
			}
			else
			{
				pTIfp->uiFldNum = FLM_ANY_FIELD;
			}
		}
		else
		{
			if( RC_BAD( rc = DDGetReference( NULL, NULL, szNameBuf,
									&pTIfp->uiFldNum)))
			{
				goto Exit;
			}
		}
	}

	if( uiNumInFldPath == 0)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	// Cannot have wildcard in last field of field path.

	if (pLastFldPath->uiFldNum == FLM_ANY_FIELD)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	// Single field has the field NULL terminated
	
	if( uiNumInFldPath == 1 )
	{
		pTDict->uiNewFldPaths += 2;
	}
	else
	{
		// The field paths are stored child to parent and parent to child
		// each are zero terminated.

		pTDict->uiNewFldPaths += 2 * (uiNumInFldPath + 1);
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Parse a data dictionary domain definition for correct syntax &
		assign the correct attributes.
****************************************************************************/
FSTATIC RCODE	DDContainerParse(
	TDICT *		pTDict,
	DDENTRY *	pDDEntry,
	FlmRecord *	pRecord)
{
	RCODE    	rc = FERR_OK;
	void *		pvField = NULL;

	if( pDDEntry)
	{

		if( (pvField = pRecord->firstChild( pRecord->root())) != NULL)
		{
			for( ; pvField; pvField = pRecord->nextSibling( pvField))
			{
				// Only option is unregistered fields

				if( pRecord->getFieldID( pvField) < FLM_FREE_TAG_NUMS)
				{
					rc = RC_SET( FERR_SYNTAX);
					goto Exit;
				}
			}
		}
	}

Exit:	

	if( RC_BAD(rc) && pvField)
	{
		pTDict->uiBadField = pRecord->getFieldID( pvField);
	}

	return( rc );
}

/****************************************************************************
Desc:	Parse a data dictionary domain definition for correct syntax &
		assign the correct attributes.
****************************************************************************/
FSTATIC RCODE DDEncDefParse(
	TDICT *		pTDict,
	DDENTRY *	pDDEntry,
	FlmRecord *	pRecord)
{
	RCODE    	rc = FERR_OK;
	void *		pvField = NULL;
	TENCDEF *	pTEncDef;

	// Make sure the version of the database is correct for encryption.
	
	if (pTDict->pDb->pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_60)
	{
		rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
		goto Exit;
	}

	if( RC_BAD( rc = pTDict->pool.poolAlloc( sizeof( TENCDEF), 
		(void **)&pTEncDef)))
	{
		goto Exit;
	}

	pTEncDef->uiAlgType = 0;
	pTEncDef->uiState = 0;
	pTEncDef->pucKeyInfo = NULL;
	pTEncDef->uiLength = 0;

	if( pDDEntry)
	{

		if( (pvField = pRecord->firstChild( pRecord->root())) != NULL)
		{
			for( ; pvField; pvField = pRecord->nextSibling( pvField))
			{
				switch ( pRecord->getFieldID( pvField) )
				{
					case FLM_TYPE_TAG:
					{
						// Get the encryption type.
						if (RC_BAD( rc = DDGetEncType( pRecord,
												 pvField,
												 &pTEncDef->uiAlgType)))
						{
							goto Exit;
						}
						break;
					}

					case FLM_KEY_TAG:
					{
						// Get the key information.
						if (RC_BAD( rc = DDGetEncKey( pTDict,
											   pRecord,
											   pvField,
											   pTEncDef)))
						{
							goto Exit;
						}
						break;
					}

					case FLM_STATE_TAG:
					{
						// Get the status information.
						if (RC_BAD( rc = DDParseStateOptions( pRecord,
														  pvField,
														  &pTEncDef->uiState)))
						{
							goto Exit;
						}
						break;
					}

					default:
					{
						rc = RC_SET( FERR_SYNTAX);
						goto Exit;
					}
				}
			}

			pDDEntry->vpDef = (void *)pTEncDef;

		}
		else
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}
	}

Exit:

	if (RC_BAD( rc) && pvField)
	{
		pTDict->uiBadField = pRecord->getFieldID( pvField);
	}

	return( rc );
}

/****************************************************************************
Desc:		Move word delimited by spaces from src to dest.  Used to move a
			word at a time for field path lists.
Notes:	Isolated so changes can be made to delemeting NAMES.
Visit:	Still bugs when name > buffer size - won't happen because only #'s
****************************************************************************/
FSTATIC FLMBOOL DDMoveWord(
	char *		pucDest,
	char *		pucSrc,
	FLMUINT		uiMaxDestLen,
	FLMUINT *	puiPos)
{
	FLMBOOL		bFoundWord = TRUE;
	FLMUINT		uiPos = *puiPos;
	char *		pMatch;
	FLMUINT		uiBytesToCopy;

	pucSrc += uiPos;
	while( *pucSrc == NATIVE_SPACE)
	{
		pucSrc++;
	}

	pMatch = pucSrc;
	while( *pMatch > NATIVE_SPACE)
	{
		pMatch++;
	}

	if( !*pMatch)
	{
		if( *pucSrc == '\0')
		{
			bFoundWord = FALSE;
			goto Exit;
		}
		
		uiBytesToCopy = f_strlen( pucSrc);
		
		if( uiBytesToCopy + 1 > uiMaxDestLen)
		{
			uiBytesToCopy = uiMaxDestLen - 1;
		}
		
		f_memcpy( pucDest, pucSrc, uiBytesToCopy + 1);
		*puiPos = uiPos + uiBytesToCopy + 1;
	}
	else
	{
		// Copy the bytes between pucSrc and pMatch minus one
		
		uiBytesToCopy = (FLMUINT) (pMatch - pucSrc);
		
		if( uiBytesToCopy + 1 > uiMaxDestLen)
		{
			uiBytesToCopy = uiMaxDestLen - 1;
		}

		f_memcpy( pucDest, pucSrc, uiBytesToCopy );
		pucDest[ uiBytesToCopy ] = '\0';

		// Go past consuctive spaces

		while( pucSrc[ ++uiBytesToCopy ] == NATIVE_SPACE)
		{
			uiBytesToCopy++;
		}

		*puiPos = uiPos + uiBytesToCopy;
	}
	
Exit:

	return( bFoundWord);
}

/****************************************************************************
Desc: 	Normalizes an internal string with possible formatting codes into
			a NATIVE string.  Drops all formatting codes and extended chars.
****************************************************************************/
FSTATIC void DDTextToNative(
	FlmRecord *		pRecord,
	void *			pvField,
	char *			pszBuffer,
	FLMUINT			uiBufLen,
	FLMUINT *		puiBufLen)
{
	RCODE			rc = FERR_OK;

	pszBuffer[ 0] = 0;
	
	if( pRecord->getDataLength( pvField))
	{
		if( RC_BAD( rc = pRecord->getNative( pvField, pszBuffer, &uiBufLen)))
		{
			if( rc != FERR_CONV_DEST_OVERFLOW)
			{
				pszBuffer[0] = 0;
				uiBufLen = 0;
			}
		}
	}
	else
	{
		uiBufLen = 0;
	}

	if( puiBufLen)
	{
		// Length needs to include the null byte
		
		*puiBufLen = uiBufLen + 1;
	}

	return;
}

/**************************************************************************** 
Desc:		Allocate the LFILE and read in the LFile entries.  The default
			data container and the dictionary container will be at hard coded
			slots at the first of the table.  The LFiles do not need to be in
			any numeric order.
****************************************************************************/
RCODE fdictReadLFiles(
	FDB *			pDb,
	FDICT *		pDict)
{
	RCODE			rc = FERR_OK;
	LFILE *		pLFiles = NULL;
	LFILE *		pLFile;
	SCACHE *		pSCache = NULL;
	FLMBOOL		bReleaseCache = FALSE;
	FLMBYTE *	pucBlk;
	FLMUINT		uiBlkAddress;
	FLMUINT		uiPos;
	FLMUINT		uiEndPos;
	FLMUINT		uiEstCount;
	FLMUINT		uiLFileCnt;
	FLMUINT		uiLFHCnt;
	FFILE *		pFile = pDb->pFile;
	FLMUINT		uiBlkSize = pFile->FileHdr.uiBlockSize;
	LFILE			TmpLFile;

	f_memset( &TmpLFile, 0, sizeof( LFILE));

	for( uiEstCount = 0, uiLFileCnt = 4,
			uiBlkAddress = pDb->pFile->FileHdr.uiFirstLFHBlkAddr; 
			uiBlkAddress != BT_END; )
	{
		if( RC_BAD( rc = ScaGetBlock( pDb, NULL, BHT_LFH_BLK,
										uiBlkAddress, NULL, &pSCache)))
		{
			goto Exit;
		}
		bReleaseCache = TRUE;

		pucBlk = pSCache->pucBlk;
		uiPos = BH_OVHD;
		if( (uiEndPos = (FLMUINT)FB2UW( &pucBlk[ BH_ELM_END])) <= BH_OVHD)
		{
			uiEndPos = BH_OVHD;
			uiLFHCnt = 0;
		}
		else
		{
			if( uiEndPos > uiBlkSize)
			{
				uiEndPos = uiBlkSize;
			}
			
			uiLFHCnt = (FLMUINT)((uiEndPos - BH_OVHD) / LFH_SIZE);
			uiEndPos = (FLMUINT)(BH_OVHD + uiLFHCnt * LFH_SIZE);
		}

		// May allocate too many like the inactive ones but OK for now.
		// Allocate an additional 2 for the default data and dict containers.

		if( !uiEstCount)
		{
			uiEstCount = uiLFHCnt + uiLFileCnt;
			if( uiEstCount)
			{
				if( RC_BAD( rc = f_calloc( uiEstCount * sizeof( LFILE), &pLFiles)))
				{
					goto Exit;
				}
			}
		}
		else if( uiLFHCnt)
		{
			uiEstCount += uiLFHCnt;

			if( RC_BAD(rc = f_recalloc( uiEstCount * sizeof(LFILE), &pLFiles)))
			{
				goto Exit;
			}
		}

		// Read through all of the logical file definitions in the block

		for( ; uiPos < uiEndPos; uiPos += LFH_SIZE)
		{
			FLMUINT	uiLfNum;

			// Have to fix up the offsets later when they are read in

			if( RC_BAD( rc = flmBufferToLFile( &pucBlk[ uiPos], &TmpLFile,
								uiBlkAddress, uiPos)))
			{
				goto Exit;
			}

			if( TmpLFile.uiLfType == LF_INVALID)
			{
				continue;
			}

			uiLfNum = TmpLFile.uiLfNum;

			if( uiLfNum == FLM_DATA_CONTAINER)
			{
				pLFile = pLFiles + LFILE_DATA_CONTAINER_OFFSET;
			}
			else if( uiLfNum == FLM_DICT_CONTAINER)
			{
				pLFile = pLFiles + LFILE_DICT_CONTAINER_OFFSET;
			}
			else if( uiLfNum == FLM_DICT_INDEX)
			{
				pLFile = pLFiles + LFILE_DICT_INDEX_OFFSET;
			}
			else if( uiLfNum == FLM_TRACKER_CONTAINER)
			{
				pLFile = pLFiles + LFILE_TRACKER_CONTAINER_OFFSET;
			}
			else
			{
				pLFile = pLFiles + uiLFileCnt++;
			}

			f_memcpy( pLFile, &TmpLFile, sizeof(LFILE));
		}

		// Get the next block in the chain

		uiBlkAddress = (FLMUINT)FB2UD( &pucBlk[ BH_NEXT_BLK]);
		ScaReleaseCache( pSCache, FALSE);
		bReleaseCache = FALSE;
	}

	// This routine could be called to re-read in the dictionary.

	if( pDict->pLFileTbl)
	{
		f_free( &pDict->pLFileTbl);
	}

	pDict->pLFileTbl = pLFiles;
	pDict->uiLFileCnt = uiLFileCnt;

Exit:
	
	if( bReleaseCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	if( RC_BAD(rc) && pLFiles)
	{
		f_free( &pLFiles);
	}

	return( rc);
}

/**************************************************************************** 
Desc:	Add data dictionary records to the data dictionary.
****************************************************************************/
RCODE fdictCreate(
	FDB *					pDb,
	const char *		pszDictPath,
	const char *		pDictBuf)
{
	RCODE    			rc = FERR_OK;
	IF_FileHdl *		pDictFileHdl = NULL;
	FlmRecord *			pDictRec = NULL;
	void *				pvField;
	const char *		pucGedBuf;
	LFILE *				pDictContLFile;
	LFILE *				pDictIxLFile;
	FLMUINT				uiDrn = 0;
	FLMUINT				uiCurrDictNum;
	FLMUINT				uiLFileCount;
	LFILE					DictContLFile;
	LFILE					DictIxLFile;
	LFILE					TempLFile;
	char					ucTempBuf[ 256];
	FLMUINT				uiBufLen = sizeof( ucTempBuf);
	F_NameTable 		nameTable;

	// Initialize the name table

	if( RC_BAD( rc = nameTable.setupFromDb( HFDB_NULL)))
	{
		goto Exit;
	}

	// Create Dictionary and Default Data containers

	if( RC_BAD(rc = flmLFileCreate( pDb, &DictContLFile, FLM_DICT_CONTAINER,
											 LF_CONTAINER)))
	{
		goto Exit;
	}
	uiCurrDictNum = FLM_DICT_CONTAINER;

	if( RC_BAD(rc = flmLFileCreate( pDb, &TempLFile, FLM_DATA_CONTAINER,
								LF_CONTAINER)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = flmLFileCreate( pDb, &DictIxLFile, FLM_DICT_INDEX,
							LF_INDEX)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = flmLFileCreate( pDb, &TempLFile, FLM_TRACKER_CONTAINER,
							LF_CONTAINER)))
	{
		goto Exit;
	}

	uiLFileCount = 4;

	// If we have a GEDCOM buffer, there is no need to open the file

	if( pDictBuf)
	{
		pucGedBuf = pDictBuf;
		uiBufLen = f_strlen( pDictBuf) + 1;
	}
	else if( pszDictPath)
	{
		pucGedBuf = ucTempBuf;
		if( RC_BAD( rc = gv_FlmSysData.pFileSystem->openFile( 
				pszDictPath, FLM_IO_RDONLY, &pDictFileHdl)))
		{
			goto Exit;
		}
	}
	else
	{
		// Neither a dictionary buffer or file were specified.  Create will
		// be done with an empty dictionary.

		goto Done_Getting_Dict;
	}

	// Create a new FDICT so we can write the dictionary records.
	// This FDICT is temporary and will be allocated again.

	if( RC_BAD( rc = fdictCreateNewDict( pDb)))
	{
		goto Exit;
	}

	if( (pDictRec = f_new FlmRecord) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = fdictGetContainer( pDb->pDict, 
				FLM_DICT_CONTAINER, &pDictContLFile)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = fdictGetIndex( pDb->pDict,
					pDb->pFile->bInLimitedMode,
					FLM_DICT_INDEX, &pDictIxLFile, NULL)))
	{
		goto Exit;
	}

	// Read through the dictionary records, adding them or creating dictionaries
	// as we go.

	for( ;;)
	{
		// Get records from buffer or file

		rc = ( pDictFileHdl)
			  ? pDictRec->importRecord( pDictFileHdl, &nameTable)
			  : pDictRec->importRecord( &pucGedBuf, uiBufLen, &nameTable);

		if( RC_BAD( rc))
		{
			if( rc == FERR_END || rc == FERR_EOF_HIT)
			{
				rc = FERR_OK;
				break;
			}
			else if( uiDrn)
			{
				// If an error occur then at least set the DRN of the 
				// previous record in the diagnostic information.

				pDb->Diag.uiInfoFlags |= FLM_DIAG_DRN;
				pDb->Diag.uiDrn = uiDrn;
			}
			
			goto Exit;
		}

		// See if we are switching dictionaries.
		
		pvField = pDictRec->root();
		if( pDictRec->getFieldID( pvField) == FLM_DICT_TAG)
		{
			rc = RC_SET( FERR_INVALID_TAG);
			goto Exit;
		}

		// Assign all fields a DRN value - parse for completeness.
		// If there is no DRN in the record (zero), one will be assigned
		// by FDDDictRecUpdate.

		uiDrn = pDictRec->getID();

		// Add the data dictionary record.  This also checks to see
		// if the record is already defined.

		if( RC_BAD( rc = fdictRecUpdate( pDb, pDictContLFile,
									pDictIxLFile, &uiDrn, pDictRec, NULL)))
		{
			goto Exit;
		}

		// Don't need to do the processing below if it is not a record
		// being put into the dictionary.

		if( uiCurrDictNum != FLM_DICT_CONTAINER)
		{
			continue;
		}

		// Create an LFILE for each index and container.
	
		if( pDictRec->getFieldID( pvField) == FLM_INDEX_TAG ||
			 pDictRec->getFieldID( pvField) == FLM_CONTAINER_TAG)
		{
			pvField = pDictRec->root();
			if( RC_BAD( rc = flmLFileCreate( pDb, &TempLFile, uiDrn,
									((pDictRec->getFieldID( pvField) == FLM_INDEX_TAG)
										? (FLMUINT)LF_INDEX 
										: (FLMUINT)LF_CONTAINER))))
			{
				goto Exit;
			}
			
			uiLFileCount++;
		}
	}

Done_Getting_Dict:

	// Create the FDICT again, this time with the dictionary pcode. 

	if( RC_BAD( rc = fdictCreateNewDict( pDb)))
	{
		goto Exit;
	}

Exit:

	if( pDictFileHdl)
	{
		pDictFileHdl->Release();
	}

	if( pDictRec)
	{
		pDictRec->Release();
	}

	return( rc);
}

/**************************************************************************** 
Desc:		Creates a new dictionary for a database.
			This occurs when on database create and on a dictionary change.
****************************************************************************/
RCODE fdictCreateNewDict(
	FDB *			pDb)
{
	RCODE			rc = FERR_OK;

	// Unlink the DB from the current FDICT, if any.

	if( pDb->pDict)
	{
		f_mutexLock( gv_FlmSysData.hShareMutex);
		flmUnlinkFdbFromDict( pDb);
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}

	// Allocate a new FDICT structure for the new dictionary we
	// are going to create.

	if( RC_BAD( rc = fdictRebuild( pDb)))
	{
		goto Exit;
	}

	// Update the FDB structure to indicate that the dictionary
	// was updated.

	pDb->uiFlags |= FDB_UPDATED_DICTIONARY;
	
Exit:

	// If we allocated an FDICT and there was an error, free the FDICT.

	if( (RC_BAD( rc)) && (pDb->pDict))
	{
		flmFreeDict( pDb->pDict);
		pDb->pDict = NULL;
	}
	
	return( rc);
}

/**************************************************************************** 
Desc:	Add a new field, container or index definition to the dictionary.
****************************************************************************/
RCODE flmAddRecordToDict(
	FDB *			pDb,
	FlmRecord *	pRecord,
	FLMUINT		uiDictId,
	FLMBOOL		bRereadLFiles)
{
	RCODE			rc = FERR_OK;
	TDICT			tDict;
	FLMBOOL		bTDictInitialized = FALSE;

	if( RC_BAD( rc = fdictCopySkeletonDict( pDb)))
	{
		goto Exit;
	}

	bTDictInitialized = TRUE;
	if( RC_BAD( rc = fdictInitTDict( pDb, &tDict)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = fdictProcessRec( &tDict, pRecord, uiDictId)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = fdictBuildTables( &tDict, bRereadLFiles, TRUE)))
	{
		goto Exit;
	}

	pDb->uiFlags |= FDB_UPDATED_DICTIONARY;

Exit:

	if( bTDictInitialized)
	{
		tDict.pool.poolFree();
	}

	// If we allocated an FDICT and there was an error, free the FDICT.

	if( (RC_BAD( rc)) && (pDb->pDict))
	{
		flmFreeDict( pDb->pDict);
		pDb->pDict = NULL;
	}

	return( rc);
}

/****************************************************************************
Desc:		Add an index a dictionary record to the container LFILE and the
			index LFILE.
****************************************************************************/
RCODE fdictRecUpdate(
	FDB *				pDb,
	LFILE *			pDictContLFile,
	LFILE *			pDictIxLFile,
	FLMUINT *		puiDrnRV,
	FlmRecord *		pNewRec,
	FlmRecord *		pOldRec,
	FLMBOOL			bRebuildOp)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiDrn = *puiDrnRV;
	FLMBOOL			bAllocatedID;
	void *			pvField;
	FLMBYTE *		pucKeyField = NULL;
	FLMUINT32		ui32BufLen;
	FLMUINT			uiEncType;

	bAllocatedID = FALSE;

	// Make sure we are using a valid DRN

	if( (uiDrn >= FLM_RESERVED_TAG_NUMS) &&
		 (uiDrn != 0xFFFFFFFF))
	{
		pDb->Diag.uiInfoFlags |= FLM_DIAG_DRN;
		pDb->Diag.uiDrn = uiDrn;
		rc = RC_SET( FERR_BAD_DICT_DRN);
		goto Exit;
	}

	// Allocate an unused DRN, if one has not been allocated.

	if( (pNewRec) && ((!uiDrn) || (uiDrn == 0xFFFFFFFF)))
	{
		FLMBOOL	bAllocAtEnd = (!uiDrn) ? TRUE : FALSE;

		bAllocatedID = TRUE;
		if( bAllocAtEnd)
		{
			if( RC_BAD( rc = FSGetNextDrn( pDb, pDictContLFile, FALSE, &uiDrn)))
			{
				goto Exit;
			}
		}
		else
		{
			// Scott 12/99: This must not be called any more.
			// The code merged ITT values into the table.
			
			flmAssert(0);
		}

		// Verify that we are not at our highest possible dictionary DRN.
		
		if( uiDrn >= FLM_RESERVED_TAG_NUMS)
		{
			rc = RC_SET( FERR_NO_MORE_DRNS);
			goto Exit;
		}
	}

	// The following code makes sure that the DRN and name have not already been
	// used, if adding.  It also makes sure that there is no conflict in
	// the type/name index.  It checks the entire shared dictionary
	// hierarchy if necessary - child and parent - to ensure no
	// conflicts.

	if( pNewRec)
	{
		// Check for ID conflicts in the dictionary being added to

		if( (!pOldRec) && (!bAllocatedID))
		{
			if( RC_BAD( rc = DDCheckIDConflict( pDb, pDictContLFile, uiDrn)))
			{
				if( (rc == FERR_ID_RESERVED) || (rc == FERR_DUPLICATE_DICT_REC))
				{
					pvField = pNewRec->root();
					if( (rc == FERR_DUPLICATE_DICT_REC) &&
						 (pNewRec->getFieldID( pvField) == FLM_RESERVED_TAG))
					{
						rc = RC_SET( FERR_CANNOT_RESERVE_ID);
					}
					pDb->Diag.uiInfoFlags |= FLM_DIAG_DRN;
					pDb->Diag.uiDrn = uiDrn;
				}
				goto Exit;
			}
		}

		// Check for name conflicts in the dictionary being added to

		if (pNewRec)
		{
			if (RC_BAD( rc = DDCheckNameConflict( pDb, pDictIxLFile, pNewRec,
											uiDrn, pOldRec)))
				goto Exit;
		}
	}

	if (!pOldRec && pNewRec)
	{
		// If this is an encryption definition record, we need to generate
		// a new key.
		
		if (pNewRec->getFieldID( pNewRec->root()) == FLM_ENCDEF_TAG && 
				!bRebuildOp && !(pDb->uiFlags & FDB_REPLAYING_RFL))
		{
			F_CCS			Ccs;

			// If we are running in limited mode, we will not be able to complete
			// this operation.

			if( pDb->pFile->bInLimitedMode)
			{
				rc = RC_SET( FERR_ENCRYPTION_UNAVAILABLE);
				goto Exit;
			}

			// Should not have a key yet.
			
			if( pNewRec->find(pNewRec->root(),
									FLM_KEY_TAG) != NULL)
			{
				rc = RC_SET( FERR_CANNOT_SET_KEY);
				goto Exit;
			}

			if( (pvField = pNewRec->find( pNewRec->root(),
													FLM_TYPE_TAG)) == NULL)
			{
				rc = RC_SET( FERR_MISSING_ENC_TYPE);
				goto Exit;
			}

			if( RC_BAD( rc = DDGetEncType( pNewRec, pvField, &uiEncType)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = Ccs.init( FALSE, uiEncType)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = Ccs.generateEncryptionKey()))
			{
				goto Exit;
			}

			if( RC_BAD( rc = Ccs.getKeyToStore( &pucKeyField, &ui32BufLen,
				NULL, pDb->pFile->pDbWrappingKey)))
			{
				goto Exit;
			}

			// Create the key field
			
			if( RC_BAD( rc = pNewRec->insert( pNewRec->root(), INSERT_LAST_CHILD,
				FLM_KEY_TAG, FLM_TEXT_TYPE, &pvField)))
			{
				goto Exit;
			}

			// Set the value of the new field
			
			if( RC_BAD( rc = pNewRec->setNative( pvField, 
				(const char *)pucKeyField)))
			{
				goto Exit;
			}
		}
	}

	// Delete the old record and its index entries, if any

	if( pOldRec)
	{
		// Delete the old record's index entries

		if( RC_BAD( rc = DDIxDictRecord( pDb, pDictIxLFile, uiDrn,
			pOldRec, KREF_DELETE_FLAG)))
		{
			goto Exit;
		}
		
		// Delete the old record - unless it is a modify

		if( !pNewRec)
		{
			if( RC_BAD( rc = FSRecUpdate( pDb, pDictContLFile, NULL, uiDrn,
				REC_UPD_DELETE)))
			{
				goto Exit;
			}
		}
	}

	// Add the new record, if any

	if( pNewRec)
	{
		// Add the record's index keys

		if( RC_BAD( rc = DDIxDictRecord( pDb, pDictIxLFile, uiDrn,
			pNewRec, 0)))
		{
			goto Exit;
		}

		// Add or modify the record itself

		if( RC_BAD( rc = FSRecUpdate( pDb, pDictContLFile, pNewRec,
						uiDrn, (FLMUINT)((pOldRec)
											  ? (FLMUINT)REC_UPD_MODIFY
											  : (FLMUINT)REC_UPD_ADD))))
		{
			goto Exit;
		}
	}

Exit:

	if( RC_OK( rc))
	{
		*puiDrnRV = uiDrn;
	}

	if (pucKeyField)
	{
		f_free( &pucKeyField);
	}

	return( rc);
}

/**************************************************************************** 
Desc:	Creates a collated type/name key for a dictionary record.
****************************************************************************/
FSTATIC RCODE DDMakeDictIxKey(
	FDB *				pDb,
	FlmRecord *		pRecord,
	FLMBYTE *		pKeyBuf,
	FLMUINT *		puiKeyLenRV)
{
	RCODE					rc = FERR_OK;
	FLMUINT				uiElmLen;
	FLMUINT				uiKeyLen = 0;
	void *				pvField = pRecord->root();
	const FLMBYTE *	pExportPtr;

	// Collate the name

	pExportPtr = pRecord->getDataPtr( pvField),
	uiElmLen = MAX_KEY_SIZ - uiKeyLen;

	if( RC_BAD( rc = KYCollateValue( &pKeyBuf [uiKeyLen], &uiElmLen,
		pExportPtr, pRecord->getDataLength( pvField), FLM_TEXT_TYPE, 
		uiElmLen, NULL, NULL, pDb->pFile->FileHdr.uiDefaultLanguage,
		FALSE, FALSE, FALSE, NULL)))
	{
		goto Exit;
	}

	uiKeyLen += uiElmLen;

Exit:

	*puiKeyLenRV = uiKeyLen;
	return( rc);
}

/**************************************************************************** 
Desc:	Checks to make sure a dictionary name has not already been used.
****************************************************************************/
FSTATIC RCODE DDCheckNameConflict(
	FDB *				pDb,
	LFILE *			pDictIxLFile,
	FlmRecord *		pNewRec,
	FLMUINT			uiDrn,
	FlmRecord *		pOldRec)
{
	RCODE				rc = FERR_OK;
	BTSK				StackArray[ BH_MAX_LEVELS];
	BTSK *			pStack;
	FLMBYTE			BtKeyBuf[ MAX_KEY_SIZ];
	FLMBYTE			IxKeyBuf[ MAX_KEY_SIZ];
	FLMUINT			uiKeyLen;
	void *			pvField;

	FSInitStackCache( &StackArray [0], BH_MAX_LEVELS);

	if (RC_BAD( rc = DDMakeDictIxKey( pDb, pNewRec, IxKeyBuf, &uiKeyLen)))
	{
		goto Exit;
	}
	
	StackArray[0].pKeyBuf = BtKeyBuf;
	pStack = StackArray;
	
	if (RC_BAD( rc = FSBtSearch( pDb, pDictIxLFile, &pStack,
						IxKeyBuf, uiKeyLen, 0L)))
	{
		goto Exit;
	}
	
	if (pStack->uiCmpStatus == BT_EQ_KEY)
	{
		FLMUINT		uiElmDoman;
		DIN_STATE	DinState;
		FLMUINT		uiFoundDrn;

		// If this is an ADD (!pOldRec), or the record found is different than
		// the one being updated, we have a problem.

		uiFoundDrn = FSRefFirst( pStack, &DinState, &uiElmDoman);
		if ((!pOldRec) || (uiFoundDrn != uiDrn))
		{
			pvField = pNewRec->root();
			pDb->Diag.uiInfoFlags |= FLM_DIAG_DRN;
			pDb->Diag.uiDrn = uiDrn;
			rc = (pNewRec->getFieldID( pvField) == FLM_RESERVED_TAG)
					? RC_SET( FERR_CANNOT_RESERVE_NAME)
					: RC_SET( FERR_DUPLICATE_DICT_NAME);
			goto Exit;
		}
	}
	
Exit:

	FSReleaseStackCache( StackArray, BH_MAX_LEVELS, FALSE);
	return( rc);
}

/**************************************************************************** 
Desc:	Checks to make sure a dictionary DRN has not already been used.
****************************************************************************/
FSTATIC RCODE DDCheckIDConflict(
	FDB *				pDb,
	LFILE *			pDictContLFile,
	FLMUINT			uiDrn)
{
	RCODE				rc = FERR_OK;
	FlmRecord *		pOldRec = NULL;

	// Read to see if there is an existing record.
	// NOTE: Deliberately not bringing into cache if not found.

	if( RC_BAD( rc = FSReadRecord( pDb, pDictContLFile, uiDrn,
							&pOldRec, NULL, NULL)))
	{
		if (rc == FERR_NOT_FOUND)
		{
			rc = FERR_OK;
		}
		else
		{
			goto Exit;
		}
	}

	if( pOldRec)
	{
		void * pvField = pOldRec->root();

		rc = ( pOldRec->getFieldID( pvField) == FLM_RESERVED_TAG)
			  ? RC_SET( FERR_ID_RESERVED)
			  : RC_SET( FERR_DUPLICATE_DICT_REC);
	}

Exit:

	if( pOldRec)
	{
		pOldRec->Release();
	}

	return( rc);
}

/**************************************************************************** 
Desc:	Generate a key for an index record and add or delete it from
		the index.
****************************************************************************/
FSTATIC RCODE DDIxDictRecord(
	FDB *				pDb,
	LFILE *			pDictIxLFile,
	FLMUINT			uiDrn,
	FlmRecord *		pRecord,
	FLMUINT			uiFlags)
{
	RCODE		 		rc;
	union
	{
		FLMBYTE		KeyBuf [sizeof( KREF_ENTRY) + MAX_KEY_SIZ];
		KREF_ENTRY	KrefEntry;
	};
	FLMUINT			uiKeyLen;

	flmAssert( pDictIxLFile->uiLfNum > 0 && 
		pDictIxLFile->uiLfNum < FLM_UNREGISTERED_TAGS); // Sanity check
	KrefEntry.ui16IxNum = (FLMUINT16)pDictIxLFile->uiLfNum;
	KrefEntry.uiDrn = uiDrn;
	KrefEntry.uiFlags = uiFlags;
	KrefEntry.uiTrnsSeq = 1;

	// Add or delete the key/reference

	if (RC_BAD( rc = DDMakeDictIxKey( pDb, pRecord,
												&KeyBuf [sizeof( KREF_ENTRY)], &uiKeyLen)))
	{
		goto Exit;
	}
	KrefEntry.ui16KeyLen = (FLMUINT16)uiKeyLen;

	if( RC_BAD( rc = FSRefUpdate( pDb, pDictIxLFile, &KrefEntry)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}

/***************************************************************************
Desc:		Get the field information.  Try shared first and then local.
****************************************************************************/
RCODE fdictGetField(
	FDICT *		pDict,
	FLMUINT		uiFieldNum,					// [in] Field Number to look up
	FLMUINT *	puiFieldType, 				// [out] Optional
	IFD **		ppFirstIfd,					// [out] Optional
	FLMUINT *	puiFieldState)				// [out] Optional
{
	RCODE			rc = FERR_OK;
	ITT			nonstandardItt;
	ITT *			pItt;

	if( pDict && pDict->pIttTbl && uiFieldNum < pDict->uiIttCnt)
	{
		pItt = &pDict->pIttTbl[ uiFieldNum];
		
		// Is it really a field?

		if( ! ITT_IS_FIELD( pItt))
		{
			rc = RC_SET( FERR_BAD_FIELD_NUM);
			goto Exit;
		}
	}
	else
	{
		// Check if the field is a FLAIM dictionary field.
		// Most of these fields are TEXT fields.

		if( (uiFieldNum >= FLM_DICT_FIELD_NUMS)
		&&  (uiFieldNum <= FLM_LAST_DICT_FIELD_NUM))
		{
			// Most of the dictionary fields are text type.
			// KYBUILD now doesn't verify unregistered or dictionary fields types.

			pItt = &nonstandardItt;
			nonstandardItt.uiType = FLM_TEXT_TYPE;
			nonstandardItt.pvItem = NULL;
		}
		else if( uiFieldNum >= FLM_UNREGISTERED_TAGS)
		{
			pItt = &nonstandardItt;
			nonstandardItt.uiType = FLM_TEXT_TYPE;
			nonstandardItt.pvItem = NULL;
		}
		else
		{
			rc = RC_SET( FERR_BAD_FIELD_NUM);
			goto Exit;
		}

	}
	if( puiFieldType)
	{
		*puiFieldType = ITT_FLD_GET_TYPE( pItt);
	}
	if( ppFirstIfd)
	{
		*ppFirstIfd = (IFD *)pItt->pvItem;
	}
	if( puiFieldState)
	{
		*puiFieldState = ITT_FLD_GET_STATE( pItt);
	}

Exit:
	return( rc);
}

/***************************************************************************
Desc:		Get the encryption information.
****************************************************************************/
RCODE fdictGetEncInfo(
	FDB *			pDb,
	FLMUINT		uiEncId,					// [in] Encryption definition to look up
	FLMUINT *	puiEncType,				// [out] Optional
	FLMUINT *	puiEncState				// [out] Optional
	)
{
	RCODE			rc = FERR_OK;
	ITT *			pItt;
	FDICT *		pDict = pDb->pDict;
	FlmRecord *	pRecord = NULL;
	void *		pvField = NULL;
	FLMUINT		uiEncState;
	FLMUINT		uiEncType;
	
	if ( pDb->pFile->bInLimitedMode)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_ENCRYPTION_UNAVAILABLE);
		goto Exit;
	}

	if( pDict && pDict->pIttTbl && uiEncId < pDict->uiIttCnt)
	{
		pItt = &pDict->pIttTbl[ uiEncId];

		// Is it really an encryption definition?

		if( ! ITT_IS_ENCDEF( pItt))
		{
			rc = RC_SET( FERR_BAD_ENCDEF_ID);
			goto Exit;
		}

		uiEncType = ((F_CCS *)pItt->pvItem)->getEncType();

		// Get the Encryption record and determine the state.
		if (RC_BAD( rc = FlmRecordRetrieve(	(HFDB)pDb,
														FLM_DICT_CONTAINER,
														uiEncId,
														FO_EXACT,
														&pRecord,
														NULL)))
		{
			goto Exit;
		}

		pvField = pRecord->find( pRecord->root(),
										 FLM_STATE_TAG);
		if (pvField)
		{
			const char *	pDataPtr = (const char *)pRecord->getDataPtr( pvField);

			if (f_strnicmp( pDataPtr, "chec", 4) == 0)
			{
				uiEncState = ITT_ENC_STATE_CHECKING;
			}
			else if (f_strnicmp( pDataPtr, "purg", 4) == 0)
			{
				uiEncState = ITT_ENC_STATE_PURGE;
			}
			else if (f_strnicmp( pDataPtr, "acti", 4) == 0)
			{
				uiEncState = ITT_ENC_STATE_ACTIVE;
			}
			else
			{
				uiEncState = ITT_ENC_STATE_UNUSED;
			}
		}
		else
		{
			uiEncState = ITT_ENC_STATE_UNUSED;
		}
	}
	else
	{
		rc = RC_SET( FERR_BAD_ENCDEF_ID);
		goto Exit;
	}

	if( puiEncType)
	{
		*puiEncType = uiEncType;
	}
	if( puiEncState)
	{
		*puiEncState = uiEncState;
	}

Exit:

	if (pRecord)
	{
		pRecord->Release();
	}
	return( rc);
}

/***************************************************************************
Desc:		Get the Container given a container number.
****************************************************************************/
RCODE fdictGetContainer(
	FDICT *		pDict,
	FLMUINT		uiContNum,
	LFILE **		ppLFile)
{
	ITT *			pItt;

	if( pDict && uiContNum < pDict->uiIttCnt && pDict->pIttTbl)
	{
		pItt = &pDict->pIttTbl[ uiContNum];
		
		// Is it really a container?

		if( !ITT_IS_CONTAINER( pItt))
		{
			return( RC_SET( FERR_BAD_CONTAINER));
		}
		if( ppLFile)
		{
			*ppLFile = (LFILE *) pItt->pvItem;
		}
	}
	else
	{
		// Hard coded container - data is [0], dictionary is [1].

		if( uiContNum == FLM_DATA_CONTAINER)
		{
			if( ppLFile)
			{
				*ppLFile = &pDict->pLFileTbl[ LFILE_DATA_CONTAINER_OFFSET];
			}
		}
		else if( uiContNum == FLM_DICT_CONTAINER)
		{
			if( ppLFile)
			{
				*ppLFile = &pDict->pLFileTbl[ LFILE_DICT_CONTAINER_OFFSET];
			}
		}
		else if( uiContNum == FLM_TRACKER_CONTAINER)
		{
			if( ppLFile)
			{
				*ppLFile = &pDict->pLFileTbl[ LFILE_TRACKER_CONTAINER_OFFSET];
			}
		}
		else
		{
			return( RC_SET( FERR_BAD_CONTAINER));
		}
	}

	return( FERR_OK);
}

/***************************************************************************
Desc:		Get the IXD, LFILE and IFD information given an index number.
****************************************************************************/
RCODE fdictGetIndex(
	FDICT *		pDict,
	FLMBOOL		bInLimitedMode,
	FLMUINT		uiIxNum,
	LFILE **		ppLFile,		// [out] optional
	IXD **		ppIxd,		// [out] optional
	FLMBOOL		bOfflineOk)
{
	RCODE			rc = FERR_OK;
	ITT *			pItt;
	LFILE *		pLFile;
	IXD *			pIxd;

	if( ppIxd)
	{
		*ppIxd = NULL;
	}

	if( ppLFile)
	{
		*ppLFile = NULL;
	}

	if( pDict && uiIxNum < pDict->uiIttCnt && pDict->pIttTbl)
	{
		pItt = &pDict->pIttTbl[ uiIxNum];

		// Is it really a container?

		if( !ITT_IS_INDEX( pItt))
		{
			rc = RC_SET( FERR_BAD_IX);
			goto Exit;
		}
		pLFile = (LFILE *) pItt->pvItem;
		pIxd = pLFile->pIxd;

		if( ppLFile)
		{
			*ppLFile = pLFile;
		}

		if( ppIxd)
		{
			*ppIxd = pIxd;
		}

		// If the index is suspended the IXD_OFFLINE flag
		// will be set, so it is sufficient to just test
		// the IXD_OFFLINE for both suspended and offline
		// conditions.

		if( (pIxd->uiFlags & IXD_OFFLINE) && !bOfflineOk)
		{
			rc = RC_SET( FERR_INDEX_OFFLINE);
			goto Exit;
		}

		// An encrypted index that cannot be decrypted is as good as
		// offline.
		if ( pIxd->uiEncId && bInLimitedMode && !bOfflineOk)
		{
			rc = RC_SET( FERR_INDEX_OFFLINE);
			goto Exit;
		}
	}
	else if (uiIxNum == FLM_DICT_INDEX)
	{
		pLFile = pDict->pLFileTbl + LFILE_DICT_INDEX_OFFSET;
		if( ppLFile)
		{
			*ppLFile = pLFile;
		}
		if( ppIxd)
		{
			*ppIxd = pLFile->pIxd;
		}
	}
	else
	{
		rc = RC_SET( FERR_BAD_IX);
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:		Given a IXD ID (index drn), returns the next Index Def (IXD).
****************************************************************************/
RCODE fdictGetNextIXD(
	FDICT *		pDict,
	FLMUINT		uiIndexNum,
	IXD **		ppIxd)
{
	RCODE			rc = FERR_OK;
	IXD *			pIxd = NULL;

	flmAssert( pDict && pDict->uiIxdCnt);

	for( uiIndexNum++; uiIndexNum < pDict->uiIttCnt; uiIndexNum++)
	{
		ITT * pItt = &pDict->pIttTbl[ uiIndexNum];
		if( ITT_IS_INDEX( pItt))
		{
			LFILE * pLFile = (LFILE *) pItt->pvItem;
			pIxd = pLFile->pIxd;
			break;
		}
	}

	// Special case -- return the dictionary index
	
	if( !pIxd && uiIndexNum < FLM_DICT_INDEX)
	{
		pIxd = pDict->pIxdTbl;
	}

	if( pIxd)
	{
		// Check to see if the index is offline.  Still return *ppIxd.

		if( pIxd->uiFlags & IXD_OFFLINE)
		{
			rc = RC_SET( FERR_INDEX_OFFLINE);
			goto Exit;
		}
	}
	else
	{
		rc = RC_SET( FERR_EOF_HIT);
		goto Exit;
	}

Exit:

	if( ppIxd)
	{
		*ppIxd = pIxd;
	}

	return( rc);
}

/****************************************************************************
Desc:		Rebuild the dictionary tables reading in all dictionary
			records.
****************************************************************************/
RCODE fdictRebuild(
	FDB *			pDb)
{
	RCODE			rc = FERR_OK;
	TDICT			tDict;
	FLMUINT		uiCount;
	IXD *			pIxd;
 	FLMBOOL		bTDictInitialized = FALSE;
	FLMBOOL		bSuspended;
	FLMUINT		uiOnlineTransId;

	// Allocate a new FDICT structure for reading the local dictionary
	// into memory.
	// At this point, pDb better not be pointing to a dictionary.

	flmAssert( pDb->pDict == NULL);
	if( RC_BAD( rc = flmAllocDict( &pDb->pDict)))
	{
		goto Exit;
	}

	if( !pDb->pDict->pLFileTbl)
	{
		// Read the local dictionary into memory.

		if( RC_BAD(rc = fdictReadLFiles( pDb, pDb->pDict)))
		{
			goto Exit;
		}

		// For a database create the LFiles still are not created.

		if( pDb->pDict->pLFileTbl->uiLfNum == 0)
		{
			goto Exit;
		}
	}

	bTDictInitialized = TRUE;
	if( RC_BAD( rc = fdictInitTDict( pDb, &tDict)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = fdictProcessAllDictRecs( pDb, &tDict)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = fdictBuildTables( &tDict, FALSE, FALSE)))
	{
		goto Exit;
	}

	// Loop through the IXD and set the uiLastDrnIndexed value.

	uiCount = pDb->pDict->uiIxdCnt;
	for( pIxd = pDb->pDict->pIxdTbl; uiCount--; pIxd++)
	{
		// Ignore any errors in case we are rebuilding.

		if( RC_BAD( flmGetIxTrackerInfo( pDb, pIxd->uiIndexNum,
					&pIxd->uiLastContainerIndexed,
					&pIxd->uiLastDrnIndexed, &uiOnlineTransId, &bSuspended)))
		{
			goto Exit;
		}

		if( bSuspended)
		{
			pIxd->uiFlags |= (IXD_SUSPENDED | IXD_OFFLINE);
		}
		else if( uiOnlineTransId == TRANS_ID_OFFLINE)
		{
			pIxd->uiFlags |= IXD_OFFLINE;
		}
	}

Exit:

	if( bTDictInitialized)
	{
		tDict.pool.poolFree();
	}

	return( rc );
}

/****************************************************************************
Desc:		Initializes and sets up a TDICT structure.
****************************************************************************/
RCODE fdictInitTDict(
	FDB *			pDb,
	TDICT *		pTDict)
{
	RCODE	rc = FERR_OK;

	f_memset( pTDict, 0, sizeof( TDICT));
	pTDict->pool.smartPoolInit( &g_TDictPoolStats);		

	pTDict->pDb = pDb;
	pTDict->uiVersionNum = pDb->pFile->FileHdr.uiVersionNum;
	pTDict->uiDefaultLanguage =
		pDb->pFile->FileHdr.uiDefaultLanguage;
	pTDict->pDict = pDb->pDict;


	if( RC_BAD(rc = fdictGetContainer( pDb->pDict, FLM_DICT_CONTAINER,
											  	&pTDict->pLFile )))
		goto Exit;
Exit:
	return( rc);
}

/****************************************************************************
Desc:		Build all of the dictionary tables given the temporary dictionary
			(pTDict) that was built in ddprep.
Note:		There are two ways this will be called.  The first is when
			we are building a dictionary from scratch.  The second is ONLY
			when a new field definition or container is added, or an index's
			state is changed.
****************************************************************************/
RCODE fdictBuildTables(
	TDICT *			pTDict,
	FLMBOOL			bRereadLFiles,
	FLMBOOL			bNewDict)
{
	RCODE				rc = FERR_OK;
	DDENTRY *		pEntry;
	TFIELD *			pTField;
	FLMUINT			uiEntryNum;
	TENCDEF *		pTEncDef;

	if( RC_BAD( rc = fdictReallocAllTables( pTDict)))
	{
		goto Exit;
	}

	// Go through and add each new item to the dictionary.

	for( pEntry = pTDict->pFirstEntry
		; pEntry 
		; pEntry = pEntry->pNextEntry )
	{
		uiEntryNum = pEntry->uiEntryNum;
		
		switch( pEntry->uiType)
		{
			case 0:	// Field
			{
				pTField = (TFIELD *) pEntry->vpDef;
				fdictAddItem( pTDict, uiEntryNum, pTField->uiFldInfo);
				break;
			}

			case ITT_INDEX_TYPE:
			{
				fdictAddItem( pTDict, uiEntryNum, ITT_INDEX_TYPE);
				if( RC_BAD( rc = fdictAddIndex( pTDict, pEntry )))
				{
					goto Exit;
				}
				break;
			}

			case ITT_CONTAINER_TYPE:
			{
				fdictAddItem( pTDict, uiEntryNum, ITT_CONTAINER_TYPE);
				// rc = fdictAddLFile( pTDict, pEntry ); Already done.
				break;
			}

			case ITT_ENCDEF_TYPE:
			{
				if (!pTDict->pDb->pFile)
				{
					flmAssert( 0);
					rc = RC_SET( FERR_ENCRYPTION_UNAVAILABLE);
					goto Exit;
				}
				else
				{
					fdictAddItem( pTDict, uiEntryNum, ITT_ENCDEF_TYPE);

					pTEncDef = (TENCDEF *) pEntry->vpDef;
					// Need to add a new CCS.
					if (RC_BAD( rc = fdictAddNewCCS(pTDict, pTEncDef, uiEntryNum)))
					{
						goto Exit;
					}
				}
				break;
			}

			default:
			{
				break;
			}
		}
	}

	if( pTDict->uiNewIfds || bNewDict)
	{
		if( RC_BAD( rc = fdictFixupIfdPointers( pTDict->pDict,
			bNewDict ? 0 : (pTDict->uiTotalIfds - pTDict->uiNewIfds))))
		{
			goto Exit;
		}
	}

	if( bRereadLFiles)
	{
		if( RC_BAD( rc = fdictReadLFiles( pTDict->pDb, pTDict->pDict)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = fdictFixupLFileTbl( pTDict->pDict)))
	{
		goto Exit;
	}

Exit:

	return( rc );
}

/****************************************************************************
Desc:	Fixup pointers in tables of copied dictionary.  This is called after
		a copy of one dictionary to another, or after a dictionary's tables
		have been reallocated.
****************************************************************************/
FSTATIC void fdictFixupPointers(
	FDICT *	pNewDict,
	FDICT *	pOldDict
	)
{
	FLMUINT	uiPos;
	FLMUINT	uiOffset;
	LFILE *	pOldLFile;
	LFILE *	pNewLFile;
	IFD *		pOldIfd;
	IFD *		pNewIfd;
	IXD *		pOldIxd;
	IXD *		pNewIxd;
	ITT *		pOldItt;
	ITT *		pNewItt;

	// Fixup anything that points to LFILE entries.

	if (pNewDict->pLFileTbl && pNewDict->pLFileTbl != pOldDict->pLFileTbl)
	{

		// Fixup pItt->pvItem pointers for indexes and containers

		for (uiPos = 0, pOldItt = pOldDict->pIttTbl,
					pNewItt = pNewDict->pIttTbl;
			  uiPos < pOldDict->uiIttCnt;
			  uiPos++, pOldItt++, pNewItt++)
		{
			if (ITT_IS_CONTAINER( pOldItt) || ITT_IS_INDEX( pOldItt))
			{
				if (pOldItt->pvItem)
				{
					LFILE *	pTmpLFile;

					pTmpLFile = (LFILE *)(pOldItt->pvItem);
					uiOffset = (FLMUINT)(pTmpLFile - pOldDict->pLFileTbl);
					pTmpLFile = pNewDict->pLFileTbl + uiOffset;
					pNewItt->pvItem = (void *)pTmpLFile;
				}
				else
				{
					flmAssert( pNewItt->pvItem == NULL);
				}
			}
			else if (ITT_IS_ENCDEF( pOldItt))
			{
				if (pOldItt->pvItem)
				{
					pNewItt->pvItem = pOldItt->pvItem;
					((F_CCS *)pNewItt->pvItem)->AddRef();
				}
				else
				{
					flmAssert( pNewItt->pvItem == NULL);
				}
			}
		}
	}

	// Fixup anything that points to IXD entries

	if (pNewDict->pIxdTbl && pNewDict->pIxdTbl != pOldDict->pIxdTbl)
	{

		// Fixup pLFile->pIxd pointers

		for (uiPos = 0, pOldLFile = pOldDict->pLFileTbl,
					pNewLFile = pNewDict->pLFileTbl;
			  uiPos < pOldDict->uiLFileCnt;
			  uiPos++, pOldLFile++, pNewLFile++)
		{
			if (pOldLFile->pIxd)
			{
				uiOffset = (FLMUINT)(pOldLFile->pIxd - pOldDict->pIxdTbl);
				pNewLFile->pIxd = pNewDict->pIxdTbl + uiOffset;
			}
			else
			{
				flmAssert( pNewLFile->pIxd == NULL);
			}
		}

		// Fixup pIfd->pIxd pointers

		for (uiPos = 0, pOldIfd = pOldDict->pIfdTbl,
					pNewIfd = pNewDict->pIfdTbl;
			  uiPos < pOldDict->uiIfdCnt;
			  uiPos++, pOldIfd++, pNewIfd++)
		{
			if (pOldIfd->pIxd)
			{
				uiOffset = (FLMUINT)(pOldIfd->pIxd - pOldDict->pIxdTbl);
				pNewIfd->pIxd = pNewDict->pIxdTbl + uiOffset;
			}
			else
			{
				flmAssert( pNewIfd->pIxd == NULL);
			}
		}
	}

	// Fixup anything that points to IFD entries

	if (pNewDict->pIfdTbl && pNewDict->pIfdTbl != pOldDict->pIfdTbl)
	{

		// Fixup pIfd->pNextInChain pointers

		for (uiPos = 0, pOldIfd = pOldDict->pIfdTbl,
					pNewIfd = pNewDict->pIfdTbl;
			  uiPos < pOldDict->uiIfdCnt;
			  uiPos++, pOldIfd++, pNewIfd++)
		{
			if (pOldIfd->pNextInChain)
			{
				uiOffset = (FLMUINT)(pOldIfd->pNextInChain - pOldDict->pIfdTbl);
				pNewIfd->pNextInChain = pNewDict->pIfdTbl + uiOffset;
			}
			else
			{
				flmAssert( pNewIfd->pNextInChain == NULL);
			}
		}

		// Fixup pIxd->pFirstIfd pointers

		for (uiPos = 0, pOldIxd = pOldDict->pIxdTbl,
					pNewIxd = pNewDict->pIxdTbl;
			  uiPos < pOldDict->uiIxdCnt;
			  uiPos++, pOldIxd++, pNewIxd++)
		{
			if (pOldIxd->pFirstIfd)
			{
				uiOffset = (FLMUINT)(pOldIxd->pFirstIfd - pOldDict->pIfdTbl);
				pNewIxd->pFirstIfd = pNewDict->pIfdTbl + uiOffset;
			}
			else
			{
				flmAssert( pNewIxd->pFirstIfd == NULL);
			}
		}

		// Fixup pItt->pvItem pointers

		for (uiPos = 0, pOldItt = pOldDict->pIttTbl,
					pNewItt = pNewDict->pIttTbl;
			  uiPos < pOldDict->uiIttCnt;
			  uiPos++, pOldItt++, pNewItt++)
		{
			if (ITT_IS_FIELD( pOldItt))
			{
				if (pOldItt->pvItem)
				{
					IFD *	pTmpIfd;

					pTmpIfd = (IFD *)(pOldItt->pvItem);
					uiOffset = (FLMUINT)(pTmpIfd - pOldDict->pIfdTbl);
					pTmpIfd = pNewDict->pIfdTbl + uiOffset;
					pNewItt->pvItem = (void *)pTmpIfd;
				}
				else
				{
					flmAssert( pNewItt->pvItem == NULL);
				}
			}
		}
	}

	// Fixup anything that points to field path entries

	if (pNewDict->pFldPathsTbl && pNewDict->pFldPathsTbl != pOldDict->pFldPathsTbl)
	{

		// Fixup pIfd->pFieldPathCToP and pIfd->pFieldPathPToC pointers

		for (uiPos = 0, pOldIfd = pOldDict->pIfdTbl,
					pNewIfd = pNewDict->pIfdTbl;
			  uiPos < pOldDict->uiIfdCnt;
			  uiPos++, pOldIfd++, pNewIfd++)
		{
			if (pOldIfd->pFieldPathCToP)
			{
				uiOffset = (FLMUINT)(pOldIfd->pFieldPathCToP - pOldDict->pFldPathsTbl);
				pNewIfd->pFieldPathCToP = pNewDict->pFldPathsTbl + uiOffset;
			}
			else
			{
				flmAssert( pNewIfd->pFieldPathCToP == NULL);
			}
			if (pOldIfd->pFieldPathPToC)
			{
				uiOffset = (FLMUINT)(pOldIfd->pFieldPathPToC - pOldDict->pFldPathsTbl);
				pNewIfd->pFieldPathPToC = pNewDict->pFldPathsTbl + uiOffset;
			}
			else
			{
				flmAssert( pNewIfd->pFieldPathPToC == NULL);
			}
		}

	}
}

/****************************************************************************
Desc:		Allocate all of the dictionary tables based on the counts that
			were incremented in pTDict.  Coded to add new fields, indexes or
			container, but not to modify or delete anything!
****************************************************************************/
FSTATIC RCODE fdictReallocAllTables(
	TDICT *			pTDict)
{
	RCODE				rc = FERR_OK;
	FDICT				OldDict;
	FDICT *			pDict = pTDict->pDict;

	// Save a copy of the old dictionary's pointers and counters
	// Easiest way to do this is to simply copy the structure.

	f_memcpy( &OldDict, pDict, sizeof( FDICT));

	if( pTDict->pLastEntry
	&&  pTDict->pLastEntry->uiEntryNum >= pDict->uiIttCnt
	&&  pTDict->pLastEntry->uiEntryNum < FLM_RESERVED_TAG_NUMS)
	{
		ITT *				pItt;
		FLMUINT			uiNewCount;

		uiNewCount = pTDict->pLastEntry->uiEntryNum + 1 - pDict->uiIttCnt;
		if (uiNewCount)
		{

			// Must fake out so that we don't lose the old table.

			pDict->pIttTbl = NULL;
			if( RC_BAD( rc = fdictReallocTbl( sizeof( ITT), pDict->uiIttCnt,
						uiNewCount, (void **) &pDict->pIttTbl)))
			{
				goto Exit;
			}
			pTDict->uiTotalItts = pDict->uiIttCnt + uiNewCount;

			// Copy the table to the new location (because of fake out above)

			if( OldDict.uiIttCnt)
			{
				f_memcpy( pDict->pIttTbl, OldDict.pIttTbl, 
					sizeof( ITT) * OldDict.uiIttCnt);
			}

			// Initialize the new items to empty.

			pItt = pDict->pIttTbl + pDict->uiIttCnt;
			for( ;uiNewCount--; pItt++)
			{
				pItt->uiType = ITT_EMPTY_SLOT;
				pItt->pvItem = NULL;
			}
		}
	}

	if (pTDict->uiNewIxds)
	{

		// Must fake out so that we don't lose the old table.

		pDict->pIxdTbl = NULL;
		if( RC_BAD( rc = fdictReallocTbl( sizeof( IXD), pDict->uiIxdCnt,
					pTDict->uiNewIxds, (void **)&pDict->pIxdTbl)))
		{
			goto Exit;
		}
		pTDict->uiTotalIxds = pDict->uiIxdCnt + pTDict->uiNewIxds;

		// Copy the table to the new location (because of fake out above)

		if( OldDict.uiIxdCnt)
		{
			f_memcpy( pDict->pIxdTbl, OldDict.pIxdTbl, 
				sizeof( IXD) * OldDict.uiIxdCnt);
		}
	}

	if (pTDict->uiNewIfds)
	{

		// Must fake out so that we don't lose the old table.

		pDict->pIfdTbl = NULL;
  		if( RC_BAD( rc = fdictReallocTbl( sizeof( IFD), pDict->uiIfdCnt,
					pTDict->uiNewIfds, (void **)&pDict->pIfdTbl)))
		{
			goto Exit;
		}
		pTDict->uiTotalIfds = pDict->uiIfdCnt + pTDict->uiNewIfds;

		// Copy the table to the new location (because of fake out above)

		if( OldDict.uiIfdCnt)
		{
			f_memcpy( pDict->pIfdTbl, OldDict.pIfdTbl, 
				sizeof( IFD) * OldDict.uiIfdCnt);
		}
	}

	if (pTDict->uiNewFldPaths)
	{

		// Must fake out so that we don't lose the old table.

		pDict->pFldPathsTbl = NULL;
		if( RC_BAD( rc = fdictReallocTbl( sizeof( FLMUINT), pDict->uiFldPathsCnt,
					pTDict->uiNewFldPaths, (void **)&pDict->pFldPathsTbl)))
		{
			goto Exit;
		}
		pTDict->uiTotalFldPaths = pDict->uiFldPathsCnt + pTDict->uiNewFldPaths;

		// Copy the table to the new location (because of fake out above)

		if( OldDict.uiFldPathsCnt)
		{
			f_memcpy( pDict->pFldPathsTbl, OldDict.pFldPathsTbl, 
				sizeof( FLMUINT) * OldDict.uiFldPathsCnt);
		}
	}

	fdictFixupPointers( pDict, &OldDict);

Exit:

	// Free any old tables where a new table was allocated.

	if (OldDict.pLFileTbl != pDict->pLFileTbl)
	{
		f_free( &OldDict.pLFileTbl);
	}
	if (OldDict.pIttTbl != pDict->pIttTbl)
	{
		f_free( &OldDict.pIttTbl);
	}
	if (OldDict.pIxdTbl != pDict->pIxdTbl)
	{
		f_free( &OldDict.pIxdTbl);
	}
	if (OldDict.pIfdTbl != pDict->pIfdTbl)
	{
		f_free( &OldDict.pIfdTbl);
	}
	if (OldDict.pFldPathsTbl != pDict->pFldPathsTbl)
	{
		f_free( &OldDict.pFldPathsTbl);
	}

	return( rc );
}


/****************************************************************************
Desc:		Allocate or reallocate a table.
****************************************************************************/
FSTATIC RCODE fdictReallocTbl(
	FLMUINT			uiElementSize,
	FLMUINT			uiTblSize,
	FLMUINT			uiAddElements,
	void **			ppvTblRV)
{	
	RCODE				rc = FERR_OK;

	// Does the table need to grow?

	if( uiAddElements)
	{
		if( *ppvTblRV)
		{
			if( RC_BAD( rc = f_recalloc( 
					uiElementSize * (uiTblSize + uiAddElements),
					ppvTblRV)))
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = f_calloc(	
					uiElementSize * (uiTblSize + uiAddElements),
					ppvTblRV)))
			{
				goto Exit;
			}
		}
	}

Exit:

	return( rc );
}

/****************************************************************************
Desc:		Add a new item to the item type table.
****************************************************************************/
FSTATIC void fdictAddItem(
	TDICT *			pTDict,
	FLMUINT			uiFieldNum,
	FLMUINT			uiFieldType)
{
	FDICT *			pDict = pTDict->pDict;
	ITT *				pItt;

	if( uiFieldNum < FLM_RESERVED_TAG_NUMS)
	{
		pItt = pDict->pIttTbl + uiFieldNum;
		pItt->uiType = uiFieldType;
		pItt->pvItem = NULL;

		if( uiFieldNum >= pDict->uiIttCnt)
		{
			pDict->uiIttCnt = uiFieldNum + 1;
		}
	}
}

/****************************************************************************
Desc:	Add the new IXD, IFD, field paths and LFILE for the index.
****************************************************************************/
FSTATIC RCODE fdictAddIndex(
	TDICT *			pTDict,
	DDENTRY *		pEntry)
{
	RCODE				rc = FERR_OK;
	FDICT *			pDict = pTDict->pDict;
	FLMUINT			uiIndexNum = pEntry->uiEntryNum;
	IXD *				pIxd;
	IFD *				pIfd;
	FLMUINT *		pFirstPToCFld;
	FLMUINT *		pFirstCToPFld;
	FLMUINT *		pCurFld;
	FLMUINT *		pTempFld;
	TIXD *			pTIxd;
	TIFD *			pTIfd;
	TIFP *			pTIfp;

	// The index numbers in the IXD array do not need to be in any order.
	// Just add all of the index information to the end of the table.

	pIxd = pDict->pIxdTbl + pDict->uiIxdCnt++;
	pIxd->uiIndexNum = uiIndexNum;

	pTIxd = (TIXD *) pEntry->vpDef;
	pIxd->uiContainerNum = pTIxd->uiContainerNum;
	pIxd->uiNumFlds = pTIxd->uiNumFlds;
	pIxd->uiFlags = pTIxd->uiFlags;
	pIxd->uiLanguage = pTIxd->uiLanguage;
	pIxd->uiLastContainerIndexed = 0xFFFFFFFF;
	pIxd->uiLastDrnIndexed = DRN_LAST_MARKER;
	pIxd->uiEncId = pTIxd->uiEncId;

	// Setup the IFD elements and the field paths.

	pIxd->pFirstIfd = pIfd = pDict->pIfdTbl + pDict->uiIfdCnt;
	pDict->uiIfdCnt += pIxd->uiNumFlds;

	for( pTIfd = pTIxd->pNextTIfd; pTIfd; pIfd++, pTIfd = pTIfd->pNextTIfd)
	{
		// This is a good place to set the IFD_LAST flag.
		// Could/Should be done in ddprep.c

		if( pTIfd->pNextTIfd == NULL)
			pTIfd->uiFlags |= IFD_LAST;

		pIfd->uiIndexNum = uiIndexNum;
		pIfd->pIxd = pIxd;
		pIfd->uiFlags = pTIfd->uiFlags;
		pIfd->uiLimit = pTIfd->uiLimit;
		pIfd->uiCompoundPos = pTIfd->uiCompoundPos;

		// The pTIfp->pNextTIfp are linked from parent to child.
		pTIfp = pTIfd->pTIfp;
		pCurFld = pDict->pFldPathsTbl + pDict->uiFldPathsCnt;
		pFirstPToCFld = pFirstCToPFld = pCurFld;

		pIfd->pFieldPathPToC = pFirstPToCFld;

		do
		{
			*pCurFld++ = pTIfp->uiFldNum;
			pTIfp = pTIfp->pNextTIfp;

		} while( pTIfp);

		pIfd->uiFldNum = *(pCurFld-1);
		pTempFld = pCurFld - 1;

		// Null Terminate
		*pCurFld++ = 0;

		pTIfp = pTIfd->pTIfp;
		if( pTIfp->pNextTIfp)		// If more than one field make the CToP path.
		{
			pFirstCToPFld = pCurFld;
			while( pTempFld != pFirstPToCFld)
			{
				*pCurFld++ = *pTempFld--;
			}
			*pCurFld++ = *pTempFld;
			*pCurFld++ = 0;
		}
		pIfd->pFieldPathCToP = pFirstCToPFld;
		pDict->uiFldPathsCnt += pCurFld - pFirstPToCFld;
	}

	return( rc );
}

/****************************************************************************
Desc:		Fixup the IFD chain and the pIfd->pIxd pointers.
****************************************************************************/
FSTATIC RCODE fdictFixupIfdPointers(
	FDICT *			pDict,
	FLMUINT			uiIfdStartOffset)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiCount;
	IFD *				pIfd;
	ITT *				pItt;
	ITT *				pIttTbl = pDict->pIttTbl;

	// Go through the IFD list and setup the pNextInChain pointers
	// making sure that the required fields are first.

	for( uiCount = pDict->uiIfdCnt - uiIfdStartOffset,
		pIfd = pDict->pIfdTbl + uiIfdStartOffset;
		uiCount; uiCount--, pIfd++)
	{
		IFD *				pPrevInChain;
		IFD *				pTempIfd;
		
		if( pIfd->uiFldNum >= pDict->uiIttCnt)
		{
			if( pIfd->uiFldNum < FLM_RESERVED_TAG_NUMS)
			{
				rc = RC_SET( FERR_BAD_REFERENCE);
				goto Exit;
			}
			continue;
		}
		else
		{
			pItt = pIttTbl + pIfd->uiFldNum;
			if( !ITT_IS_FIELD( pItt))
			{
				rc = RC_SET( FERR_BAD_REFERENCE);
				goto Exit;
			}
		}

		// Move the field type to the pIfd->uiFlags

		IFD_SET_FIELD_TYPE( pIfd, ITT_FLD_GET_TYPE( pItt));
		
		// Need to include 'any', 'use', 'parent' tags as valid tags.

		if( !pItt->pvItem)
		{
			pItt->pvItem = (void *) pIfd;
		}
		else
		{
			// Follow the chain and index at the front or rear depending on 
			// if the field is required within the set.

			pTempIfd = (IFD *) pItt->pvItem;
			if( (pIfd->uiFlags & IFD_REQUIRED_IN_SET)
			 || !(pTempIfd->uiFlags & IFD_REQUIRED_IN_SET))
			{
				pIfd->pNextInChain = pTempIfd;
				pItt->pvItem = (void *) pIfd;
			}
			else
			{
				// Not required in set and first IFD is required in set.
				// Look for first not required IFD in the chain.

				pPrevInChain = pTempIfd;
				pTempIfd = pTempIfd->pNextInChain;
				
				for( ; pTempIfd; pTempIfd = pTempIfd->pNextInChain)
				{
					if( !(pTempIfd->uiFlags & IFD_REQUIRED_IN_SET))
						break;
					pPrevInChain = pTempIfd;
				}
				pIfd->pNextInChain = pPrevInChain->pNextInChain;
				pPrevInChain->pNextInChain = pIfd;
			}
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Fixup the ITT pointers into the LFILE elements and all of 
			the IXD pointers in the LDICT.
****************************************************************************/
RCODE fdictFixupLFileTbl(
	FDICT *			pDict)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiCount;
	LFILE *			pLFile;
	IXD *				pIxd;
	ITT *				pItt;
	ITT *				pIttTbl = pDict->pIttTbl;
	FLMUINT			uiIttCnt = pDict->uiIttCnt;

	for( uiCount = pDict->uiLFileCnt, pLFile = pDict->pLFileTbl
		; uiCount; uiCount--, pLFile++)
	{

		if( pLFile->uiLfNum != FLM_DATA_CONTAINER
		 && pLFile->uiLfNum != FLM_DICT_CONTAINER
		 && pLFile->uiLfNum != FLM_DICT_INDEX
		 && pLFile->uiLfNum != FLM_TRACKER_CONTAINER)
		{
			pItt = pIttTbl + pLFile->uiLfNum;
			
			if( uiIttCnt <= pLFile->uiLfNum || 
				(pLFile->uiLfType == LF_CONTAINER && !ITT_IS_CONTAINER( pItt)))
			{
				rc = RC_SET( FERR_BAD_REFERENCE);
				goto Exit;
			}
			if( pLFile->uiLfType == LF_INDEX && !ITT_IS_INDEX( pItt))
			{
				rc = RC_SET( FERR_BAD_REFERENCE);
				goto Exit;
			}
			
			pItt->pvItem = pLFile;
		}
		else if( pLFile->uiLfNum == FLM_DICT_INDEX)
		{
			// The first IXD should be the dictionary index.

			if( pDict->pIxdTbl && pDict->pIxdTbl->uiIndexNum == FLM_DICT_INDEX)
			{
				pLFile->pIxd = pDict->pIxdTbl;
			}
		}
	}

	// Now that all of the indexes/containers in the ITT table point
	// to the LFILE entries, fixup the LFILE to point to the IXD entries.

	for( uiCount = pDict->uiIxdCnt, pIxd = pDict->pIxdTbl;
		  uiCount; uiCount--, pIxd++)
	{
		if( uiIttCnt <= pIxd->uiIndexNum)
		{
			if( pIxd->uiIndexNum != FLM_DICT_INDEX)
			{
				rc = RC_SET( FERR_BAD_REFERENCE);
				goto Exit;
			}
		}
		else
		{
			pItt = pIttTbl + pIxd->uiIndexNum;
			pLFile = (LFILE *) pItt->pvItem;

			if( !pLFile)
			{
				rc = RC_SET( FERR_BAD_REFERENCE);
				goto Exit;
			}
			pLFile->pIxd = pIxd;
		}

		// Verify that the pIxd->uiContainerNum is actually a container.
		// A value of 0 means that the index is on ALL containers.

		if (pIxd->uiContainerNum)
		{
			if( uiIttCnt <= pIxd->uiContainerNum)
			{
				if( pIxd->uiContainerNum != FLM_DATA_CONTAINER
				 && pIxd->uiContainerNum != FLM_DICT_CONTAINER
				 && pIxd->uiContainerNum != FLM_TRACKER_CONTAINER)
				{
					rc = RC_SET( FERR_BAD_REFERENCE);
					goto Exit;
				}
			}
			else
			{
				pItt = pIttTbl + pIxd->uiContainerNum;
				if( !ITT_IS_CONTAINER( pItt))
				{
					rc = RC_SET( FERR_BAD_REFERENCE);
					goto Exit;
				}
			}
		}
	}

Exit:
	return( rc);
}

/****************************************************************************
Desc:		Add a new CCS reference to the item type table.  If a key is included
			we can use it, otherwise we will have to generate one.
****************************************************************************/
FSTATIC RCODE fdictAddNewCCS(
	TDICT *			pTDict,
	TENCDEF *		pTEncDef,
	FLMUINT			uiRecNum)
{
	RCODE				rc = FERR_OK;
	FDICT *			pDict = pTDict->pDict;
	ITT *				pItt;
	F_CCS *			pCcs = NULL;
	FDB *				pDb = pTDict->pDb;
	F_CCS *			pDbWrappingKey;

	if( uiRecNum >= FLM_RESERVED_TAG_NUMS)
	{
		goto Exit;
	}
	
	if (!pDb->pFile->bInLimitedMode)
	{
		
		pDbWrappingKey = pDb->pFile->pDbWrappingKey;
	
		flmAssert( pTEncDef);
	
		if ((pCcs = f_new F_CCS) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
	
		// Setup the F_CCS.
		if (RC_BAD( rc = pCcs->init( FALSE, pTEncDef->uiAlgType )))
		{
			goto Exit;
		}
	
		if (!pTEncDef->uiLength)
		{
			flmAssert( 0);
			rc = RC_SET( FERR_MISSING_ENC_KEY);
			goto Exit;
		}
	
		// We need to set the key information.  This also unwraps the key and stores the
		// handle.
		
		if( RC_BAD( rc = pCcs->setKeyFromStore( pTEncDef->pucKeyInfo,
			(FLMUINT32)pTEncDef->uiLength, NULL, pDbWrappingKey)))
		{
			goto Exit;
		}
	}

	// Save the CCS object in the ITT table.
	
	pItt = pDict->pIttTbl + uiRecNum;
	pItt->pvItem = (void *)pCcs;
	pCcs = NULL;

	if( uiRecNum >= pDict->uiIttCnt)
	{
		pDict->uiIttCnt = uiRecNum + 1;
	}

Exit:

	if (pCcs)
	{
		delete pCcs;
	}

	return( rc);

}

/****************************************************************************
Desc:		Copies an existing dictionary to a new dictionary.  This does not
			fix up all of the ITT's pvItem pointers (including the
			pFirstIfd pointer of fields in the ITT table).  To clone the
			dictionary, call fdictCloneDict.
****************************************************************************/
RCODE fdictCopySkeletonDict(
	FDB *			pDb)
{
	RCODE			rc = FERR_OK;
	FDICT *		pNewDict = NULL;
	FDICT *		pOldDict = pDb->pDict;
	FLMUINT		uiTblSize;
	FLMUINT		uiPos;
	LFILE *		pLFile;
	IXD *			pIxd;
	ITT *			pItt;
	ITT *			pNewIttTbl = NULL;
	FLMUINT		uiNewIttTblLen = 0;
	LFILE *		pNewDictIndexLFile = NULL;
	FLMUINT *	pOldFieldPathsTbl = NULL;
	FLMUINT *	pNewFieldPathsTbl = NULL;

	if( RC_BAD( rc = f_calloc( (FLMUINT)sizeof( FDICT), &pNewDict)))
	{
		goto Exit;
	}

	pNewDict->pNext = pNewDict->pPrev = NULL;
	pNewDict->pFile = NULL;
	pNewDict->uiUseCount = 1;

	// Nothing to do is not a legal state.
	if( !pOldDict)
	{
		flmAssert( pOldDict != NULL);
		pDb->pDict = pNewDict;
		goto Exit;
	}

	// ITT Table

	if( (uiTblSize = pNewDict->uiIttCnt = pOldDict->uiIttCnt) != 0)
	{
		if( RC_BAD( rc = f_alloc( uiTblSize * sizeof( ITT), &pNewDict->pIttTbl)))
		{
			goto Exit;
		}
		pNewIttTbl = pNewDict->pIttTbl;
		uiNewIttTblLen = uiTblSize;
		f_memcpy( pNewDict->pIttTbl, pOldDict->pIttTbl,
			uiTblSize * sizeof( ITT));

		// Clear out all of the pointer values.
		pItt = pNewDict->pIttTbl;
		for( uiPos = 0; uiPos < uiTblSize; uiPos++, pItt++)
		{
			if ( pItt->uiType == ITT_ENCDEF_TYPE && !pDb->pFile->bInLimitedMode)
			{
				flmAssert( pItt->pvItem);
				((F_CCS *)pItt->pvItem)->AddRef();
			}
			else
			{
				pItt->pvItem = NULL;
			}
		}
	}

	// LFILE Table

	if( (uiTblSize = pNewDict->uiLFileCnt = pOldDict->uiLFileCnt) != 0)
	{
		if( RC_BAD( rc = f_alloc( uiTblSize * sizeof( LFILE),
			&pNewDict->pLFileTbl)))
		{
			goto Exit;
		}
		f_memcpy( pNewDict->pLFileTbl, pOldDict->pLFileTbl,
			uiTblSize * sizeof( LFILE));

		for( pLFile = pNewDict->pLFileTbl; uiTblSize--; pLFile++)
		{
			if( pLFile->uiLfNum < FLM_RESERVED_TAG_NUMS)
			{
				// WARNING: The code must make a new LFILE
				// before the dictionary is aware of it.

				if( pLFile->uiLfNum < uiNewIttTblLen)
				{
					pItt = pNewIttTbl + pLFile->uiLfNum;
					pItt->pvItem = (void *) pLFile;
				}
			}
			else if( pLFile->uiLfNum == FLM_DICT_INDEX)
			{
				pNewDictIndexLFile = pLFile;
			}
		}
	}

	// IXD Table

	if( (uiTblSize = pNewDict->uiIxdCnt = pOldDict->uiIxdCnt) != 0)
	{
		if( RC_BAD( rc = f_alloc(
			uiTblSize * sizeof( IXD), &pNewDict->pIxdTbl)))
		{
			goto Exit;
		}
		f_memcpy( pNewDict->pIxdTbl, pOldDict->pIxdTbl,
			uiTblSize * sizeof( IXD));

		// Fixup all of the pointers to the IXD.

		for( pIxd = pNewDict->pIxdTbl; uiTblSize--; pIxd++)
		{
			if( pIxd->uiIndexNum != FLM_DICT_INDEX)
			{
				pItt = pNewIttTbl + pIxd->uiIndexNum;
				pLFile = (LFILE *) pItt->pvItem;
				pLFile->pIxd = pIxd;
			}
			else if( pNewDictIndexLFile)
			{
				pNewDictIndexLFile->pIxd = pIxd;
			}
		}
	}

	// Field Paths Table

	if( (uiTblSize = pNewDict->uiFldPathsCnt = pOldDict->uiFldPathsCnt) != 0)
	{
		if( RC_BAD( rc = f_alloc( uiTblSize * sizeof( FLMUINT),
					&pNewDict->pFldPathsTbl)))
		{
			goto Exit;
		}
		f_memcpy( pNewDict->pFldPathsTbl, pOldDict->pFldPathsTbl,
			uiTblSize * sizeof( FLMUINT));

		pOldFieldPathsTbl = pOldDict->pFldPathsTbl;
		pNewFieldPathsTbl = pNewDict->pFldPathsTbl;
	}

	// IFD Table

	if( (uiTblSize = pNewDict->uiIfdCnt = pOldDict->uiIfdCnt) != 0)
	{
		IFD *			pIfd;
		FLMUINT		uiLastIndexNum;
		FLMUINT		uiOffset;

		if( RC_BAD( rc = f_alloc( uiTblSize * sizeof( IFD),
					&pNewDict->pIfdTbl)))
		{
			goto Exit;
		}
		f_memcpy( pNewDict->pIfdTbl, pOldDict->pIfdTbl,
			uiTblSize * sizeof( IFD));

		// Fixup all pFirstIfd pointers, backlinks to the pIxd and fldPathTbls.
		// Set all of the IfdChain values to NULL to be fixed up later.
		pIfd = pNewDict->pIfdTbl;
		uiLastIndexNum = 0;

		for( uiPos = 0; uiPos < uiTblSize; uiPos++, pIfd++)
		{
			pIfd->pNextInChain = NULL;

			if( pIfd->uiIndexNum != FLM_DICT_INDEX)
			{
				pItt = pNewIttTbl + pIfd->uiIndexNum;
				pLFile = (LFILE *) pItt->pvItem;
				pIxd = pLFile->pIxd;
			}
			else
			{
				pIxd = pNewDictIndexLFile->pIxd;
			}

			pIfd->pIxd = pIxd;
			if( uiLastIndexNum != pIfd->uiIndexNum)
			{
				pIxd->pFirstIfd = pIfd;
				uiLastIndexNum = pIfd->uiIndexNum;
			}

			// Fixup the field paths.

			flmAssert( pNewFieldPathsTbl != NULL);
			uiOffset = pIfd->pFieldPathCToP - pOldFieldPathsTbl;
			pIfd->pFieldPathCToP = pNewFieldPathsTbl + uiOffset;

			uiOffset = pIfd->pFieldPathPToC - pOldFieldPathsTbl;
			pIfd->pFieldPathPToC = pNewFieldPathsTbl + uiOffset;
		}
	}

	f_mutexLock( gv_FlmSysData.hShareMutex);
	flmUnlinkFdbFromDict( pDb);
	f_mutexUnlock( gv_FlmSysData.hShareMutex);
	pDb->pDict = pNewDict;
	pNewDict = NULL;

Exit:

	if( RC_BAD( rc) && pNewDict)
	{
		// Undo all of the allocations on the new table.
		if( pNewDict->pLFileTbl)
		{
			f_free( &pNewDict->pLFileTbl);
		}
		if( pNewDict->pIttTbl)
		{
			f_free( &pNewDict->pIttTbl);
		}
		if( pNewDict->pIxdTbl)
		{
			f_free( &pNewDict->pIxdTbl);
		}
		if( pNewDict->pIfdTbl)
		{
			f_free( &pNewDict->pIfdTbl);
		}
		if( pNewDict->pFldPathsTbl)
		{
			f_free( &pNewDict->pFldPathsTbl);
		}
		f_free( &pNewDict);
	}

	return( rc);
}

/**************************************************************************** 
Desc:	Creates a new version of the current dictionary and fixes up all
		pointers
****************************************************************************/
RCODE fdictCloneDict(
	FDB *			pDb)
{
	RCODE			rc = FERR_OK;
	TDICT			tDict;
	FLMBOOL		bTDictInitialized = FALSE;

	if( RC_BAD( rc = fdictCopySkeletonDict( pDb)))
	{
		goto Exit;
	}

	bTDictInitialized = TRUE;
	if( RC_BAD( rc = fdictInitTDict( pDb, &tDict)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = fdictBuildTables( &tDict, FALSE, TRUE)))
	{
		goto Exit;
	}

	pDb->uiFlags |= FDB_UPDATED_DICTIONARY;

Exit:

	if( bTDictInitialized)
	{
		tDict.pool.poolFree();
	}

	// If we allocated an FDICT and there was an error, free the FDICT.

	if( (RC_BAD( rc)) && (pDb->pDict))
	{
		flmFreeDict( pDb->pDict);
		pDb->pDict = NULL;
	}

	return( rc);
}
