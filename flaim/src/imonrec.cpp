//-------------------------------------------------------------------------
// Desc:	Class for editing database records via HTTP web pages.
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

#include "flaimsys.h"

/*********************************************************
Desc:	This function handles the ProcessRecord request.
**********************************************************/
RCODE F_ProcessRecordPage::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE				rc = FERR_OK;
	F_Session *		pFlmSession = m_pFlmSession;
	HFDB				hDb;
	FLMUINT			uiContainer;
	FLMUINT			uiDrn;
	char				szTmp [128];
	char *			pTmp = &szTmp[ 0];
	FLMBOOL			bReadOnly;
	char				szDbKey[ F_SESSION_DB_KEY_LEN];


	// We need to check our session.
	if (!pFlmSession)
	{
		printErrorPage( m_uiSessionRC,  TRUE, "No session available for this request");
		goto Exit;
	}

	// There are several fields on the form that we require, regardless of
	// the action that is being taken.

	// The hDb (Database File Handle)
	if (RC_BAD( rc = getDatabaseHandleParam( uiNumParams, ppszParams, pFlmSession, &hDb, szDbKey)))
	{
		printErrorPage( rc, TRUE, "Invalid Database Handle");
		goto Exit;
	}

	// ReadOnly flag
	szTmp[ 0] = '\0';
	bReadOnly = TRUE;

	// The value may be in the URL or in a form.
	if (RC_BAD( rc = ExtractParameter( uiNumParams,
												  ppszParams,
												  "ReadOnly",
												  sizeof( szTmp),
												  szTmp)))
	{
		getFormValueByName( "ReadOnly", 
				&pTmp, sizeof( szTmp), NULL);
	}

	if (szTmp[ 0])
	{
		if (f_stricmp( szTmp, "FALSE") == 0)
		{
			bReadOnly = FALSE;
		}
	}


	// The Record Drn
	szTmp[ 0] = '\0';

	if (RC_BAD( rc = ExtractParameter( uiNumParams,
												  ppszParams,
												  "DRN",
												  sizeof( szTmp),
												  szTmp)))
	{
		getFormValueByName( "DRN", 
				&pTmp, sizeof( szTmp), NULL);
	}

	if (szTmp[ 0])
	{
		uiDrn = f_atoud( szTmp);
	}
	else
	{
		rc = RC_SET( FERR_INVALID_PARM);
		printErrorPage( rc, TRUE, "Record DRN is Missing");
		goto Exit;
	}


	// The Container number
	szTmp[ 0] = '\0';

	if (RC_BAD( rc = ExtractParameter( uiNumParams,
												  ppszParams,
												  "container",
												  sizeof( szTmp),
												  szTmp)))
	{
		getFormValueByName( "container",
				&pTmp, sizeof( szTmp), NULL);
	}

	if (szTmp[ 0])
	{
		uiContainer = f_atoud( szTmp);
	}
	else
	{
		rc = RC_SET( FERR_INVALID_PARM);
		printErrorPage( rc, TRUE, "Record Container is missing");
		goto Exit;
	}

	// The Action to perform

	szTmp[ 0] = '\0';

	if (RC_BAD( rc = ExtractParameter( uiNumParams,
												  ppszParams,
												  "Action",
												  sizeof( szTmp),
												  szTmp)))
	{
		getFormValueByName( "Action", 
				&pTmp, sizeof( szTmp), NULL);
	}

	if (szTmp[ 0])
	{
		if (f_stricmp( szTmp, "Add") == 0)
		{
			addRecord( pFlmSession, hDb, szDbKey, uiDrn, uiContainer, bReadOnly);
		}
		else if (f_stricmp( szTmp, "New") == 0)
		{
			newRecord( pFlmSession, hDb, szDbKey, uiDrn, uiContainer, bReadOnly);
		}
		else if (f_stricmp( szTmp, "Delete") == 0)
		{
			deleteRecord( pFlmSession, hDb, szDbKey, uiDrn, uiContainer, bReadOnly);
		}
		else if (f_stricmp( szTmp, "Modify") == 0)
		{
			modifyRecord( pFlmSession, hDb, szDbKey, uiDrn, uiContainer, bReadOnly);
		}
		else if (f_stricmp( szTmp, "Retrieve") == 0)
		{
			retrieveRecord( pFlmSession, hDb, szDbKey, uiDrn, uiContainer, bReadOnly);
		}
		else if (f_stricmp( szTmp, "InsertSibling") == 0)
		{
			insertField( pFlmSession, hDb, szDbKey, uiDrn, uiContainer, bReadOnly, INSERT_NEXT_SIB);
		}
		else if (f_stricmp( szTmp, "InsertChild") == 0)
		{
			insertField( pFlmSession, hDb, szDbKey, uiDrn, uiContainer, bReadOnly, INSERT_FIRST_CHILD);
		}
		else if (f_stricmp( szTmp, "Copy") == 0)
		{
			copyField( pFlmSession, hDb, szDbKey, uiDrn, uiContainer, bReadOnly);
		}
		else if (f_stricmp( szTmp, "Clip") == 0)
		{
			clipField( pFlmSession, hDb, szDbKey, uiDrn, uiContainer, bReadOnly);
		}
		else
		{
			rc = RC_SET( FERR_INVALID_PARM);
			printErrorPage( rc, TRUE, "Invalid Action on Form");
			goto Exit;
		}
	}


Exit:

	fnEmit();

	return( rc);

}

/*********************************************************
Desc:	Adds the record to the database.  The fields in the
		current form identify the data to add.
**********************************************************/
void F_ProcessRecordPage::addRecord(
	F_Session *	pFlmSession,
	HFDB				hDb,
	const char *	pszDbKey,
	FLMUINT			uiDrn,
	FLMUINT			uiContainer,
	FLMBOOL			bReadOnly)
{
	RCODE						rc = FERR_OK;
	FlmRecord *				pRec = NULL;
	FLMUINT					uiAutoTrans;

	// We first need to reconsitute the record.
	if (RC_BAD( rc = constructRecord( uiDrn, uiContainer, &pRec, hDb)))
	{
		goto Exit;
	}

	uiAutoTrans = FLM_AUTO_TRANS | FLM_NO_TIMEOUT;
	if (RC_BAD( rc = FlmRecordAdd( hDb, uiContainer, &uiDrn, pRec, uiAutoTrans)))
	{
		displayRecordPage( pFlmSession, hDb, pszDbKey, pRec, bReadOnly,rc);
		goto Exit;
	}

	// Retrieve the new record to display.
	retrieveRecord( pFlmSession, hDb, pszDbKey, uiDrn, uiContainer, bReadOnly, FO_EXACT);

Exit:

	if( pRec)
	{
		pRec->Release();
	}


	return;
}


/*********************************************************
Desc:	Creates a new record.  This is a record that does not
	   exist in the database.  Do not store the record yet.
**********************************************************/
void F_ProcessRecordPage::newRecord(
	F_Session *	pFlmSession,
	HFDB				hDb,
	const char *	pszDbKey,
	FLMUINT			uiDrn,
	FLMUINT			uiContainer,
	FLMBOOL			bReadOnly)
{
	RCODE					rc = FERR_OK;
	FlmRecord *			pRec = NULL;
	char					szTmp[ 128];
	F_NameTable *		pNameTable = NULL;
	FLMUINT				uiType;
	FLMUINT				uiTagNum;
	void *				pvField = NULL;
	char *				pTmp = &szTmp[0];

	// Start with a  new record.

	if( (pRec = f_new FlmRecord) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		printErrorPage( rc, TRUE, "Failed to create new record");
		goto Exit;
	}

	pRec->setID( uiDrn);
	pRec->setContainerID( uiContainer);

	// Get the tag number for the root field.
	if (RC_BAD( rc = getFormValueByName( "fieldlist", &pTmp, sizeof( szTmp), NULL)))
	{
		printErrorPage( rc, TRUE, "Root field type could not be determined");
		goto Exit;
	}
	uiTagNum = f_atoud( szTmp);


	if (RC_BAD( rc = pFlmSession->getNameTable( hDb, &pNameTable)))
	{
		printErrorPage( rc, TRUE, "Could not get a Name Table");
		goto Exit;
	}

	// Get the field type from the name table.
	if (!pNameTable->getFromTagNum( uiTagNum, NULL, szTmp, sizeof( szTmp), NULL, &uiType))
	{
		rc = RC_SET( FERR_INVALID_PARM);
		printErrorPage( rc, TRUE, "Invalid field selected");
		goto Exit;
	}

	// Create the root field.
	if (RC_BAD( rc = pRec->insertLast( 0, uiTagNum, uiType, &pvField)))
	{
		printErrorPage( rc, TRUE, "Error occurred inserting field into record");
		goto Exit;
	}

	if (RC_BAD( rc))
	{
		goto Exit;
	}

	// Retrieve the new record to display.
	displayRecordPage( pFlmSession, hDb, pszDbKey, pRec, bReadOnly, rc);

Exit:

	if( pRec)
	{
		pRec->Release();
	}

	return;
}


/*********************************************************
Desc:	Modifies (updates) the record identified by the Drn & Container
		and presents it to the client.
**********************************************************/
void F_ProcessRecordPage::modifyRecord(
	F_Session *		pFlmSession,
	HFDB				hDb,
	const char *	pszDbKey,
	FLMUINT			uiDrn,
	FLMUINT			uiContainer,
	FLMBOOL			bReadOnly)
{
	RCODE				rc = FERR_OK;
	FlmRecord *		pRec = NULL;
	FLMUINT			uiAutoTrans;

	// We first need to reconsitute the record.

	if (RC_BAD( rc = constructRecord( uiDrn, uiContainer, &pRec, hDb)))
	{
		goto Exit;
	}

	uiAutoTrans = FLM_AUTO_TRANS | FLM_NO_TIMEOUT;
	if (RC_BAD( rc = FlmRecordModify( hDb, uiContainer, uiDrn, pRec, uiAutoTrans)))
	{
		displayRecordPage( pFlmSession, hDb, pszDbKey, pRec, bReadOnly, rc);
		goto Exit;
	}

	// Retrieve the new record to display.
	retrieveRecord( pFlmSession, hDb, pszDbKey, uiDrn, uiContainer, bReadOnly, FO_EXACT);

Exit:

	if( pRec)
	{
		pRec->Release();
	}


	return;
}

/*********************************************************
Desc:	Builds a new record from the data in the form
**********************************************************/
RCODE F_ProcessRecordPage::constructRecord(
	FLMUINT				uiDrn,
	FLMUINT				uiContainer,
	FlmRecord **		ppRec,
	HFDB					hDb)
{
	RCODE					rc = FERR_OK;
	FlmRecord *			pRec;
	char					szTmp[ 128];
	char *				pTmp;
	FLMUINT				uiFieldCount;
	FLMUINT				uiFieldCounter;
	char *				pszFldValue = NULL;

	flmAssert( ppRec);

	// Start with a  new record.

	if( (pRec = f_new FlmRecord) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	pRec->setID( uiDrn);
	pRec->setContainerID( uiContainer);

	pTmp = &szTmp[ 0];

	getFormValueByName( "FieldCount", 
			&pTmp, sizeof( szTmp), NULL);

	if (szTmp[ 0])
	{
		uiFieldCount = f_atoud( szTmp);
	}
	else
	{
		rc = RC_SET( FERR_INVALID_PARM);
		printErrorPage( rc, TRUE, "Field Count missing or invalid");
		goto Exit;
	}

	// Now we need to get each of the fields, beginning at level 0

	for ( uiFieldCounter = 0; uiFieldCounter < uiFieldCount; uiFieldCounter++)
	{
		// Retrieve the next field.
		FLMUINT				uiLevel;
		FLMUINT				uiType;
		FLMUINT				uiTag;
		void *				pvField = NULL;

		if (RC_BAD( rc = extractFieldInfo( uiFieldCounter, &pszFldValue,
								&uiLevel, &uiType, &uiTag)))
		{
			printErrorPage( rc, TRUE, "Error occurred retrieving field data from form");
			goto Exit;
		}

		// Insert the field into the record.
		if (RC_BAD( rc = pRec->insertLast( uiLevel, uiTag, uiType, &pvField)))
		{
			printErrorPage( rc, TRUE, "Error occurred inserting field into record");
			goto Exit;
		}

		// Set the data value
		switch( uiType)
		{
			case FLM_TEXT_TYPE:
				{
					// Store as unicode
					rc = storeUnicodeField( pRec, pvField, pszFldValue);
					break;
				}
			case FLM_NUMBER_TYPE:
				{
					// Store as UINT or INT
					rc = storeNumberField( pRec, pvField, pszFldValue);
					break;
				}
			case FLM_BINARY_TYPE:
				{
					// Store after converting from hex display
					rc = storeBinaryField( pRec, pvField, pszFldValue); 
					break;
				}
			case FLM_CONTEXT_TYPE:
				{
					// Store as a RecPointer
					if (pszFldValue && *pszFldValue != '\0')
					{
						rc = pRec->setRecPointer( pvField,
							(unsigned long)f_atoud( pszFldValue));
					}
					break;
				}
			case FLM_BLOB_TYPE:
				{
					// Store a Blob object...
					rc = storeBlobField( pRec, pvField, pszFldValue, hDb);
					break;
				}
		}

		if (RC_BAD( rc))
		{
			goto Exit;
		}
		f_free( &pszFldValue);
	}


Exit:

	if (RC_BAD(rc))
	{
		if (pRec)
		{
			pRec->Release();
			pRec = NULL;
		}
	}

	if (pszFldValue)
	{
		f_free( &pszFldValue);
	}

	*ppRec = pRec;


	return( rc);
}


/*********************************************************
Desc:	
**********************************************************/
RCODE F_ProcessRecordPage::extractFieldInfo(
	FLMUINT			uiFieldCounter,
	char **			ppucBuf,
	FLMUINT *		puiLevel,
	FLMUINT *		puiType,
	FLMUINT *		puiTag)
{
	RCODE				rc = FERR_OK;
	char				szField[50];
	char				szTmp[ 128];
	char *			pTmp;

	// field value first. This call will allocate a buffer for us that will
	// need to be freed later!  If we get an FERR_NOT_FOUND error,
	// then ignore it.
	
	f_sprintf( (char *)szField, "field%u", (unsigned)uiFieldCounter);
	*ppucBuf = NULL;
	
	if (RC_OK( rc = getFormValueByName( szField, ppucBuf, 0, NULL)))
	{
		fcsDecodeHttpString( *ppucBuf);
	}
	else if (rc != FERR_NOT_FOUND)
	{
		goto Exit;
	}


	pTmp = &szTmp[ 0];

	// fieldLevel next.
	f_sprintf( (char *)szField, "fieldLevel%u", (unsigned)uiFieldCounter);
	if (RC_BAD( rc = getFormValueByName( szField, &pTmp, sizeof( szTmp), NULL)))
	{
		goto Exit;
	}
	*puiLevel = f_atoud( szTmp);

	// fieldType next
	f_sprintf( (char *)szField, "fieldType%u", (unsigned)uiFieldCounter);
	if (RC_BAD( rc = getFormValueByName( szField, &pTmp, sizeof( szTmp), NULL)))
	{
		goto Exit;
	}
	*puiType = f_atoud( szTmp);

	// fieldTag next
	f_sprintf( (char *)szField, "fieldTag%u", (unsigned)uiFieldCounter);
	if (RC_BAD( rc = getFormValueByName( szField, &pTmp, sizeof( szTmp), NULL)))
	{
		goto Exit;
	}
	*puiTag = f_atoud( szTmp);

Exit:

	return( rc);

}

/*********************************************************
Desc:	Deletes the record identified by the Drn & Container
		and presents it to the client.
**********************************************************/
void F_ProcessRecordPage::deleteRecord(
	F_Session *	pFlmSession,
	HFDB				hDb,
	const char *	pszDbKey,
	FLMUINT			uiDrn,
	FLMUINT			uiContainer,
	FLMBOOL			bReadOnly)
{
	RCODE				rc = FERR_OK;
	RCODE				uiRc;
	FLMUINT			uiAutoTrans;
	FlmRecord *		pRec = NULL;

	if (RC_OK( rc = FlmRecordRetrieve(
		hDb, uiContainer, uiDrn, FO_EXACT, (FlmRecord **)&pRec, &uiDrn)))
	{
		uiAutoTrans = FLM_AUTO_TRANS | FLM_NO_TIMEOUT;
		if (RC_BAD( rc = FlmRecordDelete( hDb, uiContainer, uiDrn, uiAutoTrans)))
		{
			uiRc = rc;
			if (RC_BAD( rc = constructRecord( uiDrn, uiContainer, &pRec, hDb)))
			{
				printErrorPage( rc, TRUE, "Failed to delete record");
				goto Exit;
			}
			displayRecordPage( pFlmSession, hDb, pszDbKey, pRec, bReadOnly, uiRc);
			goto Exit;
		}
	}
	else
	{
		uiRc = rc;
		if (RC_BAD( rc = constructRecord( uiDrn, uiContainer, &pRec, hDb)))
		{
			printErrorPage( rc, TRUE, "Failed to delete record. Invalid Record");
			goto Exit;
		}
		displayRecordPage( pFlmSession, hDb, pszDbKey, pRec, bReadOnly, uiRc);
		goto Exit;
	}
	
	// This will present an empty record page since the record is no longer there.
	retrieveRecord( pFlmSession, hDb, pszDbKey, 0, uiContainer, bReadOnly);

Exit:

	if (pRec)
	{
		pRec->Release();
	}

	return;
}

/*********************************************************
Desc:	Retrieves the record identified by the Drn & Container
		and presents it to the client.
**********************************************************/
void F_ProcessRecordPage::retrieveRecord(
	F_Session *	pFlmSession,
	HFDB				hDb,
	const char *	pszDbKey,
	FLMUINT			uiDrn,
	FLMUINT			uiContainer,
	FLMBOOL			bReadOnly,
	FLMUINT			uiFlag)
{
	RCODE				rc = FERR_OK;
	FlmRecord *		pRec = NULL;
	char				szTmp[ 20];
	char *			pTmp = &szTmp[ 0];
	FLMUINT			uiFlags = FO_EXACT;

	if (uiFlag == 0xFFFFFFFF)
	{
		// Get the flags
		if (RC_OK( rc = getFormValueByName( "flags", &pTmp, sizeof( szTmp), NULL)))
		{
			uiFlags = f_atoud( szTmp);
		}
	}
	else
	{
		uiFlags = uiFlag;
	}

	rc = FlmRecordRetrieve( hDb, uiContainer, uiDrn, uiFlags, (FlmRecord **)&pRec, &uiDrn);
	if ((rc == FERR_NOT_FOUND) && (uiDrn == 0))
	{
		rc = FERR_OK;
	}

	displayRecordPage( pFlmSession, hDb, pszDbKey, pRec, bReadOnly, rc);

	if (pRec)
	{
		pRec->Release();
	}

	return;
}

/*********************************************************
Desc:	Insert a new field at the specified location.
**********************************************************/
void F_ProcessRecordPage::insertField(
	F_Session *	pFlmSession,
	HFDB				hDb,
	const char *	pszDbKey,
	FLMUINT			uiDrn,
	FLMUINT			uiContainer,
	FLMBOOL			bReadOnly,
	FLMUINT			uiInsertAt)
{
	RCODE				rc = FERR_OK;
	FlmRecord *		pRec = NULL;
	char				szTmp[ 128];
	F_NameTable *	pNameTable = NULL;
	FLMUINT			uiType;
	FLMUINT			uiTagNum;
	void *			pvField = NULL;
	char *			pTmp = &szTmp[0];
	FLMUINT			uiFldCnt;
	FLMUINT			uiSelectedField;
	FLMUINT			uiLoop;

	// Reconstruct the record.
	if (RC_BAD( rc = constructRecord( uiDrn, uiContainer, &pRec, hDb)))
	{
		goto Exit;
	}


	// Get the field count so we know whether to look for the selected field.
	if (RC_BAD( rc = getFormValueByName( "FieldCount", &pTmp, sizeof( szTmp), NULL)))
	{
		printErrorPage( rc, TRUE, "Could not retrieve the record field count");
		goto Exit;
	}
	uiFldCnt = f_atoud( szTmp);

	if (uiFldCnt == 1)
	{
		uiSelectedField = 0;
	}
	else
	{
		// We need to get the actual selected field
		if (RC_BAD( rc = getFormValueByName( "radioSel", &pTmp, sizeof( szTmp), NULL)))
		{
			printErrorPage( rc, TRUE, "Could not retrieve the selected field");
			goto Exit;
		}
		uiSelectedField = f_atoud( szTmp);
	}

	// Get the tag number for the new field.
	if (RC_BAD( rc = getFormValueByName( "fieldlist", &pTmp, sizeof( szTmp), NULL)))
	{
		printErrorPage( rc, TRUE, "Selected field type could not be determined");
		goto Exit;
	}
	uiTagNum = f_atoud( szTmp);


	if (RC_BAD( rc = pFlmSession->getNameTable( hDb, &pNameTable)))
	{
		printErrorPage( rc, TRUE, "Could not get a Name Table");
		goto Exit;
	}

	// Get the new field type from the name table.
	if (!pNameTable->getFromTagNum( uiTagNum, NULL, szTmp, sizeof( szTmp), NULL, &uiType))
	{
		rc = RC_SET( FERR_INVALID_PARM);
		printErrorPage( rc, TRUE, "Invalid field selected");
		goto Exit;
	}


	// Start at the root
	pvField = pRec->root();
	for (uiLoop = 0; uiLoop < uiSelectedField; uiLoop++)
	{
		pvField = pRec->next( pvField);		
	}
	

	// Insert the child
	if (RC_BAD( rc = pRec->insert( pvField, uiInsertAt,  uiTagNum, uiType, &pvField)))
	{
		displayRecordPage( pFlmSession, hDb, pszDbKey, pRec, bReadOnly, rc);
		goto Exit;
	}

	// Display the new record to display.
	displayRecordPage( pFlmSession, hDb, pszDbKey, pRec, bReadOnly, rc);

Exit:

	if( pRec)
	{
		pRec->Release();
	}

	return;
}

/*********************************************************
Desc:	
**********************************************************/
void F_ProcessRecordPage::copyField(
	F_Session *	pFlmSession,
	HFDB				hDb,
	const char *	pszDbKey,
	FLMUINT			uiDrn,
	FLMUINT			uiContainer,
	FLMBOOL			bReadOnly)
{
	RCODE				rc = FERR_OK;
	FlmRecord *		pRec = NULL;
	char				szTmp[ 128];
	void *			pvOrigField = NULL;
	void *			pvMarkerField = NULL;
	char *			pTmp = &szTmp[0];
	FLMUINT			uiFldCnt;
	FLMUINT			uiSelectedField;
	FLMUINT			uiLoop;

	// Reconstruct the record.
	if (RC_BAD( rc = constructRecord( uiDrn, uiContainer, &pRec, hDb)))
	{
		goto Exit;
	}


	// Get the field count so we know whether to look for the selected field.
	if (RC_BAD( rc = getFormValueByName( "FieldCount", &pTmp, sizeof( szTmp), NULL)))
	{
		displayRecordPage( pFlmSession, hDb, pszDbKey, pRec, bReadOnly, rc);
		goto Exit;
	}
	uiFldCnt = f_atoud( szTmp);

	if (uiFldCnt == 1)
	{
		uiSelectedField = 0;
	}
	else
	{
		// We need to get the actual selected field
		if (RC_BAD( rc = getFormValueByName( "radioSel", &pTmp, sizeof( szTmp), NULL)))
		{
			displayRecordPage( pFlmSession, hDb, pszDbKey, pRec, bReadOnly, rc);
			goto Exit;
		}
		uiSelectedField = f_atoud( szTmp);
	}

	// Start at the root
	pvOrigField = pRec->root();

	// Scan to the selected field.
	for ( uiLoop = 0; uiLoop < uiSelectedField; uiLoop++)
	{
		pvOrigField = pRec->next( pvOrigField);
	}

	// Now find where to begin inserting the copy of this field.
	pvMarkerField = pRec->nextSibling( pvOrigField);

	if (!pRec->isLast( pvOrigField) && pvMarkerField == NULL && !pRec->hasChild( pvOrigField))
	{
		pvMarkerField = pRec->next( pvOrigField);
	}

	// Copy/Insert fields from the OrigField to the MarkerField. If the MarkerField is
	// NULL, then add to the end of the record (i.e. after the last field).
	if (RC_BAD( rc = copyFieldsFromTo( pRec, pvOrigField, pvMarkerField)))
	{
		displayRecordPage( pFlmSession, hDb, pszDbKey, pRec, bReadOnly, rc);
		goto Exit;
	}

	// Retrieve the new record to display.
	displayRecordPage( pFlmSession, hDb, pszDbKey, pRec, bReadOnly, rc);

Exit:

	if( pRec)
	{
		pRec->Release();
	}

	return;
}



/*********************************************************
Desc:	
**********************************************************/
RCODE F_ProcessRecordPage::copyFieldsFromTo(
	FlmRecord *		pRec,
	void *			pvOrigField,
	void *			pvMarkerField)
{
	RCODE					rc = FERR_OK;
	const FLMBYTE *	pSrcData = NULL;
	FLMBYTE *			pDestData = NULL;
	void *				pvField = NULL;
	void *				pvSrcField = NULL;
	FLMUINT				uiFldCount;
	FLMUINT				uiType;
	FLMUINT				uiFldId;
	FLMUINT				uiLength;
	FLMUINT				uiSrcLevel;
	FLMUINT				uiTargetLevel;

	for( uiFldCount = 0, pvField = pvOrigField; 
			pvField != pvMarkerField; uiFldCount++)
	{
		pvField = pRec->next( pvField);
	}

	pvSrcField = pvOrigField;
	for (pvField = pvOrigField ; uiFldCount > 0; uiFldCount--)
	{
		uiFldId = pRec->getFieldID( pvSrcField);
		uiType = pRec->getDataType( pvSrcField);
		uiLength = pRec->getDataLength( pvSrcField);
		uiSrcLevel = pRec->getLevel( pvSrcField);
		uiTargetLevel = pRec->getLevel( pvField);

		if (uiSrcLevel == uiTargetLevel)
		{
			// Insert as the next sibling
			if (RC_BAD( rc = pRec->insert( pvField, INSERT_NEXT_SIB, uiFldId, uiType, &pvField)))
			{
				goto Exit;
			}
		}
		else if (uiSrcLevel > uiTargetLevel)
		{
			if (RC_BAD( rc = pRec->insert( pvField, INSERT_LAST_CHILD, uiFldId, uiType, &pvField)))
			{
				goto Exit;
			}
		}
		else // if (uiTargetLevel > uiSrcLevel)
		{
			pvField = pRec->parent( pvField);
			
			// Insert as the next sibling
			
			if( RC_BAD( rc = pRec->insert( pvField, INSERT_NEXT_SIB, 
				uiFldId, uiType, &pvField)))
			{
				goto Exit;
			}
		}
		
		pSrcData = pRec->getDataPtr( pvSrcField);
		
		if( RC_BAD( rc = pRec->allocStorageSpace( pvField, uiType, uiLength, 
			0, 0, 0, &pDestData, NULL)))
		{
			goto Exit;
		}

		f_memcpy( pDestData, pSrcData, uiLength);
		pvSrcField = pRec->next( pvSrcField);
	}

Exit:

	return( rc);
}

/*********************************************************
Desc:	
**********************************************************/
void F_ProcessRecordPage::clipField(
	F_Session *	pFlmSession,
	HFDB				hDb,
	const char *	pszDbKey,
	FLMUINT			uiDrn,
	FLMUINT			uiContainer,
	FLMBOOL			bReadOnly)
{
	RCODE				rc = FERR_OK;
	FlmRecord *		pRec = NULL;
	char				szTmp[ 128];
	void *			pvField = NULL;
	char *			pTmp = &szTmp[0];
	FLMUINT			uiFldCnt;
	FLMUINT			uiSelectedField;
	FLMUINT			uiLoop;

	// Reconstruct the record.
	if (RC_BAD( rc = constructRecord( uiDrn, uiContainer, &pRec, hDb)))
	{
		goto Exit;
	}


	// Get the field count so we know whether to look for the selected field.
	if (RC_BAD( rc = getFormValueByName( "FieldCount", &pTmp, sizeof( szTmp), NULL)))
	{
		displayRecordPage( pFlmSession, hDb, pszDbKey, pRec, bReadOnly, rc);
		goto Exit;
	}
	uiFldCnt = f_atoud( szTmp);

	if (uiFldCnt == 1)
	{
		uiSelectedField = 0;
	}
	else
	{
		// We need to get the actual selected field
		if (RC_BAD( rc = getFormValueByName( "radioSel", &pTmp, sizeof( szTmp), NULL)))
		{
			displayRecordPage( pFlmSession, hDb, pszDbKey, pRec, bReadOnly, rc);
			goto Exit;
		}
		uiSelectedField = f_atoud( szTmp);
	}

	// Start at the root
	pvField = pRec->root();

	// Scan to the selected field.
	for ( uiLoop = 0; uiLoop < uiSelectedField; uiLoop++)
	{
		pvField = pRec->next( pvField);
	}

	if (RC_BAD( rc = pRec->remove( pvField)))
	{
		displayRecordPage( pFlmSession, hDb, pszDbKey, pRec, bReadOnly, rc);
		goto Exit;
	}

	// Display the new record to display.
	displayRecordPage( pFlmSession, hDb, pszDbKey, pRec, bReadOnly, rc);

Exit:

	if (pRec)
	{
		pRec->Release();
	}

	return;
}


/*********************************************************
Desc:	Converts a string in the form XX XX XX XX (a hex
		representation of a binary stream to a binary stream
		again, then stores it in the provided field.
**********************************************************/
RCODE F_ProcessRecordPage::storeBinaryField(
	FlmRecord *		pRec,
	void *			pvField,
	const char *	pszFldValue)
{
	RCODE						rc = FERR_OK;
	F_DynamicBuffer *		pBuf = NULL;
	const char *			pszTmp;
	FLMBOOL					bHaveFirstNibble = FALSE;
	FLMBYTE					ucVal = 0;
	FLMUINT					uiNibble;

	if (pszFldValue == '\0' || *pszFldValue == '\0')
	{
		goto Exit;
	}

	if ((pBuf = f_new F_DynamicBuffer) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		printErrorPage( rc, TRUE, "Failed to allocate dynamic buffer to store binary field");
		goto Exit;
	}

	pszTmp = pszFldValue;
	while( *pszTmp)
	{
		if (*pszTmp >= '0' && *pszTmp <= '9')
		{
			uiNibble = (FLMUINT)(*pszTmp - '0');
		}
		else if (*pszTmp >= 'a' && *pszTmp <= 'f')
		{
			uiNibble = (FLMUINT)(*pszTmp - 'a' + 10);
		}
		else if (*pszTmp >= 'A' && *pszTmp <= 'F')
		{
			uiNibble = (FLMUINT)(*pszTmp - 'A' + 10);
		}
		else
		{
			pszTmp++;
			continue;
		}
		if (bHaveFirstNibble)
		{
			ucVal += (FLMBYTE)uiNibble;
			if (RC_BAD( rc = pBuf->addChar( ucVal)))
			{
				printErrorPage( rc, TRUE, "Failed to convert binary hex stream");
				goto Exit;
			}
			bHaveFirstNibble = FALSE;
		}
		else
		{
			ucVal = (FLMBYTE)(uiNibble << 4);
			bHaveFirstNibble = TRUE;
		}
		pszTmp++;
	}

	// See if we ended on an odd number of nibbles.

	if (bHaveFirstNibble)
	{
		if (RC_BAD( rc = pBuf->addChar( ucVal)))
		{
			printErrorPage( rc, TRUE, "Failed to convert binary hex stream");
			goto Exit;
		}
	}
	if (pBuf->getBufferSize())
	{
		if (RC_BAD( rc = pRec->setBinary( pvField, (void *)pBuf->printBuffer(),
					pBuf->getBufferSize())))
		{
			printErrorPage( rc, TRUE, "Failed to set BINARY value");
			goto Exit;
		}
	}

Exit:

	if (pBuf)
	{
		pBuf->Release();
	}

	return( rc);
}


/*********************************************************
Desc:	
**********************************************************/
RCODE F_ProcessRecordPage::storeUnicodeField(
	FlmRecord *		pRec,
	void *			pvField,
	const char *	pszFldValue)
{
	RCODE					rc = FERR_OK;
	FLMUNICODE *		puzBuf = NULL;
	FLMUINT				uiBufSize = 0;
	FLMUINT				uiLen;

	// If there is no data, then just return.
	if (pszFldValue == '\0' || *pszFldValue == '\0')
	{
		goto Exit;
	}

	if (RC_BAD( rc = tokenGetUnicode( pszFldValue, (void **)&puzBuf, &uiLen,
								&uiBufSize)))
	{
		printErrorPage( rc, TRUE, "Failed to parse UNICODE from ASCII buffer");
		goto Exit;
	}

	if (RC_BAD( rc = pRec->setUnicode( pvField, puzBuf)))
	{
		printErrorPage( rc, TRUE, "Failed to set UNICODE value");
		goto Exit;
	}

Exit:

	if (puzBuf)
	{
		f_free( &puzBuf);
	}

	return( rc);
}

/*********************************************************
Desc:	
**********************************************************/
RCODE F_ProcessRecordPage::storeNumberField(
	FlmRecord *		pRec,
	void *			pvField,
	const char *	pszFldValue)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiVal;
	FLMINT			iVal;

	if (pszFldValue == '\0' || *pszFldValue == '\0')
	{
		goto Exit;
	}

	// If this is a negative value, then we will store it as an INT
	if (*pszFldValue == '-')
	{
		iVal = f_atoi( pszFldValue);
		if (RC_BAD( rc = pRec->setINT( pvField, iVal)))
		{
			printErrorPage( rc, TRUE, "Failed to set INT field in record");
			goto Exit;
		}
	}
	else
	{
		uiVal = f_atoud( pszFldValue);
		if (RC_BAD( rc = pRec->setUINT( pvField, uiVal)))
		{
			printErrorPage( rc, TRUE, "Failed to set UINT field in record");
			goto Exit;
		}
	}

Exit:

	return( rc);
}


/*********************************************************
Desc:	
**********************************************************/
RCODE F_ProcessRecordPage::storeBlobField(
	FlmRecord *			pRec,
	void *				pvField,
	const char *		pszFldValue,
	HFDB					hDb)
{
	RCODE					rc = FERR_OK;
	FlmBlob *			pBlob = NULL;

	if (pszFldValue == '\0' || *pszFldValue == '\0')
	{
		goto Exit;
	}

	if ((pBlob = f_new FlmBlobImp) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		printErrorPage( rc, TRUE, "Failed to allocate new Blob object");
		goto Exit;
	}


	if(RC_BAD( rc = pBlob->referenceFile( hDb, pszFldValue, TRUE)))
	{
		printErrorPage( rc, TRUE, "Failed to create new Blob object");
		goto Exit;
	}

	if(RC_BAD( rc = pRec->setBlob( pvField, pBlob)))
	{
		printErrorPage( rc, TRUE, "Failed to store Blob object in Record");
		goto Exit;
	}

Exit:

	if (pBlob)
	{
		pBlob->Release();
		pBlob = NULL;
	}

	return( rc);
}

/*********************************************************
Desc:	
**********************************************************/
void F_ProcessRecordPage::displayRecordPage(
	F_Session *		pFlmSession,
	HFDB					hDb,
	const char *		pszDbKey,
	FlmRecord *			pRec,
	FLMBOOL				bReadOnly,
	RCODE					uiRc)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiContext = 0;
	F_NameTable *	pNameTable = NULL;
	char				szTmp[128];
	char *			pTmp = &szTmp[0];
	FLMUINT			uiTagNum = 0;
	FLMUINT			uiFlags = FO_EXACT;

	// Get the name table.
	if (RC_BAD( rc = pFlmSession->getNameTable( hDb, &pNameTable)))
	{
		printErrorPage( rc, TRUE, "Could not get a Name Table");
		goto Exit;
	}

	// Get the tag number for the currently selected field.
	if (RC_OK( rc = getFormValueByName( "fieldlist", &pTmp, sizeof( szTmp), NULL)))
	{
		uiTagNum = f_atoud( szTmp);
	}

	// Get the flags
	if (RC_OK( rc = getFormValueByName( "flags", &pTmp, sizeof( szTmp), NULL)))
	{
		uiFlags = f_atoud( szTmp);
	}

	// Begin the document.
	stdHdr();

	fnPrintf( m_pHRequest, HTML_DOCTYPE);
	fnPrintf( m_pHRequest, "<html>\n");
	fnPrintf( m_pHRequest, "<HEAD><TITLE>Database iMonitor - Record Display</TITLE>\n");
	printRecordStyle();
	printStyle();
	fnPrintf( m_pHRequest, "</HEAD>\n");
	fnPrintf( m_pHRequest, "<body>\n");


	printTableStart( "Record Manager (Traditional)", 1, 100);
	printTableEnd();

	if (RC_BAD( uiRc))
	{
		fnPrintf( m_pHRequest, "<font color=red>Return Code = 0x%04X, %s</font>\n",
				(unsigned)uiRc, FlmErrorString( uiRc));
	}

	printRecord( pszDbKey, pRec, pNameTable, &uiContext, bReadOnly, uiTagNum, uiFlags);

	fnPrintf( m_pHRequest, "</body>\n");

Exit:
	
	return;

}
