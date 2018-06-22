//-------------------------------------------------------------------------
// Desc:	Sample application for FLAIM.
// Tabs:	3
//
// Copyright (c) 2003-2007 Novell, Inc. All Rights Reserved.
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

#include "flaim.h"
#include <stdio.h>

const char * gv_pszSampleDictionary =	
	"0 @1@ field Person\n"
	" 1 type text\n"
	"0 @2@ field LastName\n"
	" 1 type text\n"
	"0 @3@ field FirstName\n"
	" 1 type text\n"
	"0 @4@ field Age\n"
	" 1 type number\n"
	"0 @5@ index LastFirst_IX\n"
	" 1 language US\n"
	" 1 key\n"
	"  2 field 2\n"
	"   3 required\n"
	"  2 field 3\n"
	"   3 required\n";

#define PERSON_TAG					1
#define LAST_NAME_TAG				2
#define FIRST_NAME_TAG				3
#define AGE_TAG						4

#define DB_NAME_STR					"sample.db"

RCODE printRecordData(
	FlmRecord *			pRec);

/***************************************************************************
Desc:	Program entry point (main)
****************************************************************************/
int main( void)
{
	HFDB					hDb = HFDB_NULL;
	HFCURSOR				hCursor = HFCURSOR_NULL;
	FLMBOOL				bTransActive = FALSE;
	FLMUINT				uiDrn;
	FlmRecord *			pDefRec = NULL;
	FlmRecord *			pRec = NULL;
	void *				pvField;
	FLMBYTE				ucTmpBuf[ 64];
	RCODE					rc = FERR_OK;

	// Initialize the FLAIM database engine.  This call
	// must be made once by the application prior to making any
	// other FLAIM calls

	if( RC_BAD( rc = FlmStartup()))
	{
		goto Exit;
	}

	// Create or open a database.  Database names in FLAIM are
	// limited to three characters.

	if( RC_BAD( rc = FlmDbCreate( DB_NAME_STR, NULL, 
		NULL, NULL, gv_pszSampleDictionary, NULL, &hDb)))
	{
		if( rc == FERR_FILE_EXISTS)
		{
			// Since the database already exists, we'll make a call
			// to FlmDbOpen to get a handle to it.

			if( RC_BAD( rc = FlmDbOpen( DB_NAME_STR, 
				NULL, NULL, 0, NULL, &hDb)))
			{
				goto Exit;
			}
			printf( "Opened database.\n\n");
		}
		else
		{
			goto Exit;
		}
	}
	else
	{
		printf( "Created database.\n\n");
	}

	// Create a record object

	if( (pDefRec = f_new FlmRecord) == NULL)
	{
		rc = FERR_MEM;
		goto Exit;
	}

	// Populate the record object with fields and values
	// The first field of a record will be inserted at
	// level zero (the first parameter of insertLast()
	// specifies the level number).  Subsequent fields
	// will be inserted at a non-zero level.

	if( RC_BAD( rc = pDefRec->insertLast( 0, PERSON_TAG,
		FLM_TEXT_TYPE, NULL)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pDefRec->insertLast( 1, FIRST_NAME_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pDefRec->setNative( pvField, "Foo")))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pDefRec->insertLast( 1, LAST_NAME_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pDefRec->setNative( pvField, "Bar")))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pDefRec->insertLast( 1, AGE_TAG,
		FLM_NUMBER_TYPE, &pvField)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pDefRec->setUINT( pvField, 32)))
	{
		goto Exit;
	}

	// Start an update transaction

	if( RC_BAD( rc = FlmDbTransBegin( hDb, FLM_UPDATE_TRANS, 15)))
	{
		goto Exit;
	}
	bTransActive = TRUE;

	// Add the record to the database.  Initialize uiDrn to 0 so that FLAIM
	// will automatically assign a unique ID to the new record.  We could
	// also have specified a specific 32-bit ID to use for the record by
	// setting uiDrn to the desired ID value.

	uiDrn = 0;
	if( RC_BAD( rc = FlmRecordAdd( hDb, FLM_DATA_CONTAINER, 
		&uiDrn, pDefRec, 0)))
	{
		goto Exit;
	}

	// Commit the transaction
	// If FlmDbTransCommit returns without an error, the changes made
	// above will be durable even if the system crashes.

	if( RC_BAD( rc = FlmDbTransCommit( hDb)))
	{
		goto Exit;
	}
	bTransActive = FALSE;

	// Retrieve the record from the database by ID

	if( RC_BAD( rc = FlmRecordRetrieve( hDb, FLM_DATA_CONTAINER, 
		uiDrn, FO_EXACT, &pRec, NULL)))
	{
		goto Exit;
	}

	// Print first name, last name, and age to stdout

	if( RC_BAD( rc = printRecordData( pRec)))
	{
		goto Exit;
	}

	// Now, build a query that retrieves the sample record.
	// First we need to initialize a cursor handle.

	if( RC_BAD( rc = FlmCursorInit( hDb, FLM_DATA_CONTAINER, &hCursor)))
	{
		goto Exit;
	}

	// We will search by first name and last name.  This will use the
	// LastFirst_IX defined in the sample dictionary for optimization.

	if( RC_BAD( rc = FlmCursorAddField( hCursor, LAST_NAME_TAG, 0)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorAddOp( hCursor, FLM_EQ_OP)))
	{
		goto Exit;
	}

	f_sprintf( (char *)ucTmpBuf, "Bar");
	if( RC_BAD( rc = FlmCursorAddValue( hCursor, FLM_STRING_VAL, 
		ucTmpBuf, 0)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorAddOp( hCursor, FLM_AND_OP)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorAddField( hCursor, FIRST_NAME_TAG, 0)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorAddOp( hCursor, FLM_EQ_OP)))
	{
		goto Exit;
	}

	f_sprintf( (char *)ucTmpBuf, "Foo");
	if( RC_BAD( rc = FlmCursorAddValue( hCursor, FLM_STRING_VAL, 
		ucTmpBuf, 0)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorFirst( hCursor, &pRec)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = printRecordData( pRec)))
	{
		goto Exit;
	}

	// Free the cursor handle

	FlmCursorFree( &hCursor);

	// Close the database

	FlmDbClose( &hDb);
	printf( "Closed database.\n");

	// FlmDbRemove will delete the database and all of its files

//	if( RC_BAD( FlmDbRemove( DB_NAME_STR, NULL, NULL, TRUE)))
//	{
//		goto Exit;
//	}
//
//	printf( "Database removed.\n");

Exit:

	// Release any record pointers we may still be holding on to

	if( pDefRec)
	{
		pDefRec->Release();
	}

	if( pRec)
	{
		pRec->Release();
	}

	// Free the cursor handle

	if( hCursor != HFCURSOR_NULL)
	{
		FlmCursorFree( &hCursor);
	}

	// If we jumped to this point with an active transaction,
	// abort it.

	if( bTransActive)
	{
		(void)FlmDbTransAbort( hDb);
	}

	// Close the database handle

	if( hDb != HFDB_NULL)
	{
		printf( "Closed database.\n");
		FlmDbClose( &hDb);
	}

	// Shut down the FLAIM database engine.  This call must be made
	// even if FlmStartup failed.  No more FLAIM calls should be made
	// by the application after calling FlmShutdown.

	FlmShutdown();

	if( RC_BAD( rc))
	{
		printf( "Error %04X -- %s\n", (unsigned)rc, 
			(char *)FlmErrorString( rc));
		return( 1);
	}


	return( 0);
}

/***************************************************************************
Desc:	Dumps data from sample records to stdout
****************************************************************************/
RCODE printRecordData(
	FlmRecord *			pRec)
{
	RCODE			rc = FERR_OK;
	void *		pvField = NULL;
	char			ucTmpBuf[ 64];
	FLMUINT		uiTmp;

	if( (pvField = pRec->find( pRec->root(), FIRST_NAME_TAG)) != NULL)
	{
		uiTmp = sizeof( ucTmpBuf);
		if( RC_BAD( rc = pRec->getNative( pvField, ucTmpBuf, &uiTmp)))
		{
			goto Exit;
		}

		printf( "First    : %s\n", ucTmpBuf);
	}

	if( (pvField = pRec->find( pRec->root(), LAST_NAME_TAG)) != NULL)
	{
		uiTmp = sizeof( ucTmpBuf);
		if( RC_BAD( rc = pRec->getNative( pvField, ucTmpBuf, &uiTmp)))
		{
			goto Exit;
		}

		printf( "Last     : %s\n", ucTmpBuf);
	}

	if( (pvField = pRec->find( pRec->root(), AGE_TAG)) != NULL)
	{
		if( RC_BAD( rc = pRec->getUINT( pvField, &uiTmp)))
		{
			goto Exit;
		}

		printf( "Age      : %u\n", (unsigned)uiTmp);
	}

	printf( "\n");

Exit:

	return( rc);
}
