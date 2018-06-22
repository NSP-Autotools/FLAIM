//-------------------------------------------------------------------------
// Desc:	BLOB read routines.
// Tabs:	3
//
// Copyright (c) 1995-2000, 2002-2007 Novell, Inc. All Rights Reserved.
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

#define 	BLOB_H_VERSION_LEN_POS		0		// 1 - version is also where _H_ ends
#define		BLOB_CODE_VERSION			28 	// BLOB Version 2.8 ends on offset 28
#define 	BLOB_H_STORAGE_TYPE_POS		1		// 1 - see byStorageType

#define 	BLOB_H_FLAGS_POS				2		// 2 - owned, referenced, ...
#define 	BLOB_H_TYPE_POS				4		// 2 - user defined type
														// Type of DATA or 0 if unknown
#define	BLOB_H_FUTURE2					6		// ZERO for now
#define 	BLOB_H_RAW_SIZE_POS			8		// 4 - for large internals
#define	BLOB_H_STORAGE_SIZE_POS		12		// 4 - for large internals
#define 	BLOB_H_MATCH_STAMP_POS		16		// 8 - match this with BLOB header
#define	BLOB_MATCH_STAMP_SIZE		8
#define	BLOB_H_RIGHT_KEY_POS 		24		// 4 - right part of encryption key

					/* Non-portable Reference BLOB Field Layout */

#define 	BLOB_R_CHARSET_POS			28		// 1=ANSI,2=UNICODE,...
#define 	BLOB_R_STRLENGTH_POS			29		// Char Length of reference path
#define 	BLOB_R_PATH_POS				30		// variable

/****************************************************************************
Desc:		Create a BLOB that references a file.
Notes:	The file will not be built by the FLAIM BLOB code, and thus the
			format of the file will not be known or controlled by FLAIM.  
****************************************************************************/
RCODE FlmBlobImp::referenceFile( 
	HFDB					hDb,
	const char *		pszFileName,
	FLMBOOL				bOwned)
{
	RCODE			rc = FERR_OK;
	char	 		szUnportablePath[ F_PATH_MAX_SIZE];
	FLMUINT		uiFlags;
	FDB *			pDb = (FDB *)hDb;

	flmAssert( !m_pHeaderBuf);

	if( RC_BAD( rc = flmCheckDatabaseState( pDb)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gv_FlmSysData.pFileSystem->pathToStorageString( 
		pszFileName, szUnportablePath)))
	{
		goto Exit;
	}

	uiFlags = (bOwned) 
					? BLOB_OWNED_REFERENCE_FLAG 
					: BLOB_UNOWNED_REFERENCE_FLAG;

	m_hDb = hDb;
	if( uiFlags & BLOB_OWNED_REFERENCE_FLAG )
	{
		m_uiStorageType = BLOB_REFERENCE_TYPE | BLOB_OWNED_TYPE;
	}
	else if( uiFlags & BLOB_UNOWNED_REFERENCE_FLAG)
	{
		m_uiStorageType = BLOB_REFERENCE_TYPE;
	}
	else
	{
		flmAssert(0);
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}
	
	m_uiFlags = uiFlags;
	m_uiAction = BLOB_CREATE_ACTION;

	if( RC_BAD( rc = buildBlobHeader( szUnportablePath)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Builds a reference blob header that will be used as 
		the data for the field.
****************************************************************************/
RCODE FlmBlobImp::buildBlobHeader(
	const char *		pszUnportablePath)
{
	RCODE       rc = FERR_OK;
	FLMBYTE *   ptr;
	FLMUINT     uiFileNameLen;

	// Determine the number of bytes to allocate.
	
	uiFileNameLen = f_strlen( pszUnportablePath) + 1;
	m_uiHeaderLen = BLOB_R_PATH_POS + uiFileNameLen;

	if( RC_BAD( rc = f_alloc( m_uiHeaderLen, &m_pHeaderBuf)))
	{
		goto Exit;
	}
	
	ptr = m_pHeaderBuf;

	ptr[ BLOB_H_VERSION_LEN_POS] = BLOB_CODE_VERSION; // 28
	ptr[ BLOB_H_STORAGE_TYPE_POS] = (FLMBYTE) m_uiStorageType;
	UW2FBA( (FLMUINT16)m_uiFlags, &ptr[ BLOB_H_FLAGS_POS ]);
	UW2FBA( BLOB_UNKNOWN_TYPE, &ptr[ BLOB_H_TYPE_POS ]);
	UW2FBA( 0, &ptr[ BLOB_H_FUTURE2 ]);
	UD2FBA( 0, &ptr[ BLOB_H_RAW_SIZE_POS ]);
	UD2FBA( 0, &ptr[ BLOB_H_STORAGE_SIZE_POS ]);
	f_memset( &ptr[ BLOB_H_MATCH_STAMP_POS ], 0, BLOB_MATCH_STAMP_SIZE );
	UD2FBA( 0, &ptr[ BLOB_H_RIGHT_KEY_POS ]);
	
	ptr[ BLOB_R_CHARSET_POS ] = 1;
	ptr[ BLOB_R_STRLENGTH_POS ] = (FLMBYTE) uiFileNameLen;
	f_memcpy( &ptr[ BLOB_R_PATH_POS ], pszUnportablePath, uiFileNameLen );

	// Watch out, the file name is NOT null terminated.
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FlmBlobImp::setupBlobFromField(
	FDB *					pDb,
	const FLMBYTE *	pBlobData,
	FLMUINT				uiBlobDataLength)
{
	RCODE		rc = FERR_OK;

	// See if the database is being forced to close

	if( RC_BAD( rc = flmCheckDatabaseState( pDb)))
	{
		goto Exit;
	}

	if( getImportDataPtr( uiBlobDataLength) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	f_memcpy( m_pHeaderBuf, pBlobData, uiBlobDataLength);

	// Check that the storage length is within reason to get information
	if( m_uiHeaderLen <= BLOB_R_PATH_POS)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	m_hDb = (HFDB)pDb;

	// Read in the data that is in the header.
	m_uiAction      = BLOB_OPEN_ACTION;
	m_uiStorageType = (FLMUINT) m_pHeaderBuf[ BLOB_H_STORAGE_TYPE_POS];
	m_uiFlags       = FB2UW( &m_pHeaderBuf[ BLOB_H_FLAGS_POS]);
	m_bReadWriteAccess  = FALSE;

Exit:
	if( RC_BAD(rc) && m_pHeaderBuf)
	{
		(void) close();
	}
	return( rc);
}

/****************************************************************************
Desc:	Closes a BLOB and frees all associated memory
****************************************************************************/
RCODE FlmBlobImp::close()						// Return value is meaningless.
{
	FlmBlobImp *	pNextBlob;
	FlmBlobImp *	pPrevBlob;
	FDB *				pDb;

	if( m_pHeaderBuf)
	{
		f_free( &m_pHeaderBuf);
		m_pHeaderBuf = NULL;
	}
	// The case of a created referenced blob that is not attached
	// will have to be deleted by the caller.

	if (m_bInDbList)
	{
		if (m_hDb != HFDB_NULL)
		{
			// Pull out of the linked list
			pPrevBlob = m_pPrevBlob;
			pNextBlob = m_pNextBlob;

			if( pPrevBlob == NULL)           /* New first blob element? */
			{
				pDb = (FDB *) m_hDb;
				pDb->pBlobList = pNextBlob;
			}
			else
			{
				pPrevBlob->setNext( pNextBlob);   /* Delete pBlob */
			}
			if( pNextBlob != NULL)
			{
				pNextBlob->setPrev( pPrevBlob);
			}
		}
		m_bInDbList = FALSE;
	}

	if( m_pFileHdl)
	{
		(void) closeFile();
	}

	return( FERR_OK);
}

/****************************************************************************
Desc:	Builds a reference blob header that will be used as 
		the data for the field.
****************************************************************************/
RCODE FlmBlobImp::closeFile()
{
	if( m_pFileHdl)
	{
		m_pFileHdl->closeFile();
		m_bFileAccessed = FALSE;

		m_pFileHdl->Release();
		m_pFileHdl = NULL;
	}
	return FERR_OK;
}

/****************************************************************************
Desc:	Opens a file given the open flags.
****************************************************************************/
RCODE FlmBlobImp::openFile()
{
	RCODE       rc = FERR_OK;
	char			szFileName[ F_PATH_MAX_SIZE];
	FDB *			pDb = (FDB *) m_hDb;

	if( !m_pFileHdl && pDb)
	{
		buildFileName( szFileName);

		if( RC_BAD( rc = gv_FlmSysData.pFileSystem->openFile( szFileName, 
					 FLM_IO_SH_DENYNONE |
							(m_bReadWriteAccess ? FLM_IO_RDWR : FLM_IO_RDONLY), 
							&m_pFileHdl)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Build a file name in the input buffer.
****************************************************************************/
RCODE FlmBlobImp::buildFileName(
	char *		pszFileName)
{
	RCODE       rc = FERR_OK;
	FLMUINT		uiNameLength;

	// Be carefull not to do a string copy - this is not null terminated.
	
	uiNameLength = m_uiHeaderLen - BLOB_R_PATH_POS;
	f_memcpy( pszFileName, &m_pHeaderBuf[ BLOB_R_PATH_POS], uiNameLength);
	pszFileName[ uiNameLength ] = '\0';

	// Override the file extension set in the gv_FlmSysData structure.

	if( gv_FlmSysData.ucBlobExt [0])
	{
		char		szBlobPath[ F_PATH_MAX_SIZE];
		char		szBlobBaseName [F_FILENAME_SIZE];
		char *	pszFileExt;

		if( RC_BAD( rc = gv_FlmSysData.pFileSystem->pathReduce( 
			pszFileName, szBlobPath, szBlobBaseName)))
		{
			goto Exit;
		}

		pszFileExt = szBlobBaseName;
		while( (*pszFileExt) && (*pszFileExt != '.'))
		{
			pszFileExt++;
		}
		
		// Add period if there was none.
		
		if( !(*pszFileExt))
		{
			*pszFileExt = '.';
		}
		
		// Get past period.
		
		pszFileExt++;
		f_strcpy( pszFileExt, (const char *)gv_FlmSysData.ucBlobExt);
		f_strcpy( pszFileName, szBlobPath);
		gv_FlmSysData.pFileSystem->pathAppend( pszFileName, szBlobBaseName);
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Compare a file name with the file name in the BLOB.  Does not
		take into account one with a full path and one without.
****************************************************************************/
FLMINT FlmBlobImp::compareFileName(
	const char *	pszFileName)
{
	char		szThisFileName[ F_PATH_MAX_SIZE];

	if( RC_BAD( buildFileName( szThisFileName)))
	{
		return( 1);
	}

#ifdef FLM_UNIX
	return( f_strcmp( szThisFileName, pszFileName));
#else
	return( f_stricmp( szThisFileName, pszFileName));
#endif
}

/****************************************************************************
Desc:	Transition the action checking for multiple referenced blobs.
****************************************************************************/
void FlmBlobImp::transitionAction(
	FLMBOOL	bDoTransition)
{
	if( bDoTransition)
	{
		/*
		Transition table:
			Operation(s)				Commit Action
			ADD							Do nothing
			DELETE						Delete blob
			ADD -> DELETE				Delete blob
			DELETE -> ADD				Do nothing
			ADD -> ADD					Multi-referenced blob - ASSERT
			DELETE -> DELETE 			Multi-referenced blob - ASSERT
		*/

		// This is the time to look for multiple-referenced blobs.

		if( m_uiCurrentAction == BLOB_ADD_ACTION)
		{
			// Two ADDs mean multi-referenced blobs.
			if( m_uiAction == BLOB_ADD_ACTION)
			{
				flmAssert(0);
			}
			m_uiAction = m_uiCurrentAction;
		}
		else if( m_uiCurrentAction == BLOB_DELETE_ACTION)
		{
			// Two DELETEs mean multi-referenced blobs.
			if( m_uiAction == BLOB_DELETE_ACTION)
			{
				flmAssert(0);
			}
			m_uiAction = m_uiCurrentAction;
		}
	}
	m_uiCurrentAction = BLOB_NO_ACTION;

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
void FlmBlobImp::setInDbList( FLMBOOL bInDbList)
{
	m_bInDbList = bInDbList;
}

/****************************************************************************
Desc:	Returns a pointer that the blob header (control) info can be copied 
		to.
****************************************************************************/
FLMBYTE * FlmBlobImp::getImportDataPtr( FLMUINT uiLength)
{
	if( m_pHeaderBuf && m_uiHeaderLen <= uiLength)
	{
		// This case handles a reuse of an existing BLOB object.
		m_uiHeaderLen = uiLength;
		return m_pHeaderBuf;
	}
	
	if( m_pHeaderBuf)
	{
		// Have a buffer, but it's not large enough
		f_free( &m_pHeaderBuf);
		m_pHeaderBuf = NULL;
		m_uiHeaderLen = 0;
	}

	m_uiHeaderLen = uiLength;
	if( RC_BAD( f_alloc( m_uiHeaderLen, &m_pHeaderBuf)))
	{
		m_pHeaderBuf = NULL;
	}

	return m_pHeaderBuf;
}


/****************************************************************************
Desc:    This code only suports referenced blobs at this time.
			Look for this blob field within the current transaction list.
			If the FILENAME has a match then transition the action to the
			current action.  If the FILENAME doesn't have a match in the list
			then create a new BLOB and add it to the list with either the
			BLOB_ADD_ACTION or the BLOB_DELETE_ACTION.

			At transaction commit time, all blobs that end with the
			BLOB_DELETE_ACTION will have their files removed.

			This code does not care about the records that the BLOB comes
			from.  So, the user can do the following:
				Add    BLOB( ABCD) in record 1
				Delete BLOB( ABCD) in record 1
				Add    BLOB( ABCD) in record 2

			The pre-ver41 code would have deleted BLOB( ABCD) because of the
			delete in record one.  Then record 2 would be pointing to a 
			non-existant blob.

			Be carefull, because this code still does not support 
			multiply-referenced blobs; both record 3 and record 4 reference
			the same blob file.  This is because the first delete will remove
			the blob file so the other reference will be corrupt.

Ret:     FERR_OK or FERR_MEM
Called:  Is only called from KyBuild().
****************************************************************************/
RCODE flmBlobPlaceInTransactionList(
	FDB *				pDb,
	FLMUINT			uiAction,
	FlmRecord * 	pRecord,
	void *			pvBlobField)
{
	RCODE					rc = FERR_OK;
	const FLMBYTE *	pBlobData;
	FLMUINT				uiBlobDataLength;
	FLMUINT				uiStorageType;
	FlmBlobImp *		pBlob;
	FlmBlobImp *		pNewBlob = NULL;
	char					szFileName[ F_PATH_MAX_SIZE];

	// Nothing to work with?
	
	if( (pBlobData = pRecord->getDataPtr( pvBlobField)) == NULL)
	{
		goto Exit;
	}

	uiBlobDataLength = pRecord->getDataLength( pvBlobField);

	// Don't have to do anything for an unowned reference
	uiStorageType = pBlobData[ BLOB_H_STORAGE_TYPE_POS];
	if( (uiStorageType & (BLOB_REFERENCE_TYPE|BLOB_OWNED_TYPE)) 
		== BLOB_REFERENCE_TYPE)
	{
		goto Exit;
	}

	// Create a temporary new blob - may or may not keep it.
	// Need to create it so we can make a file name.

	if( (pNewBlob = f_new FlmBlobImp) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	if( RC_BAD( rc = pNewBlob->setupBlobFromField( pDb,
											pBlobData, uiBlobDataLength)))
	{
		pNewBlob->Release();
		pNewBlob = NULL;
		goto Exit;
	}
	pNewBlob->setCurrentAction( uiAction);
	pNewBlob->buildFileName( szFileName);

	for( pBlob = pDb->pBlobList; pBlob; pBlob = pBlob->getNext())
	{
		if( pBlob->compareFileName( szFileName) == 0)
		{
			// Found a match!
			pBlob->transitionAction( FALSE);
			pNewBlob->Release();
			pNewBlob = NULL;
			break;
		}
	}
	if( !pBlob)
	{

		// Link to the front of the list - doesn't matter where in the list.
		pBlob = pDb->pBlobList;
		pDb->pBlobList = pNewBlob;
		pNewBlob->setNext( pBlob);
		pNewBlob->setInDbList( TRUE);
		if( pBlob)
		{
			pBlob->setPrev( pNewBlob);
		}
		// Don't delete pNewBlob - in the linked list.
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:    Commit the record operation (add,modify,delete)
****************************************************************************/
RCODE FB_OperationEnd(
	FDB *       pDb,
	RCODE			rcOfOperation)
{
	RCODE				rc = FERR_OK;
	FlmBlobImp *	pBlob;
		
	// The pDb may not be all initialized due to errors in 
	// the fdbInit().  Check for null values.
	
	if( !pDb || pDb->uiTransType == FLM_NO_TRANS)
	{
		goto Exit;
	}

	for( pBlob = pDb->pBlobList; pBlob; pBlob = pBlob->getNext())
	{
		pBlob->transitionAction( rcOfOperation == FERR_OK ? TRUE : FALSE);
	}
	
Exit:
	rc = (rcOfOperation != FERR_OK) ? rcOfOperation : rc;
	return( rc );
}

/****************************************************************************
Desc:    Called after the commit phase.
			Go through the BLOB list and delete the file of all deleted blobs.
			No longer supports deleting unattached blobs.  These were created
			blobs that were never attached to a data record field.
****************************************************************************/
void FBListAfterCommit(
	FDB *       pDb)
{
	FlmBlobImp *	pBlob;
	FlmBlobImp *	pNextBlob;
	char				szFileName[ F_PATH_MAX_SIZE];

	for( pBlob = pDb->pBlobList; pBlob; pBlob = pNextBlob )
	{
		pNextBlob = pBlob->getNext();

		if( pBlob->getAction() == BLOB_DELETE_ACTION)
		{
			// Better not be opened.  Build the file name and delete.

			if( RC_OK( pBlob->buildFileName( szFileName)))
			{
				gv_FlmSysData.pFileSystem->deleteFile( szFileName);
			}
		}

		(void) pBlob->close();
		pBlob->Release();
	}

	return;
}

/****************************************************************************
Desc:    Called after the abort command.  Cleans up the FlmBlob actions
			according to the abort rules - doing nothing to the newly added 
			blobs.
Notes:   Could be called before or during the commit call or as part of
			FlmTransAbort().  Must handle all of the cases.
****************************************************************************/
void FBListAfterAbort(
	FDB *       pDb)
{
	FlmBlobImp *	pBlob;
	FlmBlobImp *	pNextBlob;

	for( pBlob = pDb->pBlobList; pBlob; pBlob = pNextBlob )
	{
		pNextBlob = pBlob->getNext();
		(void) pBlob->close();
		pBlob->Release();
	}

	return;
}

/****************************************************************************
Desc:    Allocate a new blob object
****************************************************************************/
FLMEXP RCODE FLMAPI FlmAllocBlob(
	FlmBlob **		ppBlob)
{
	RCODE			rc = FERR_OK;

	if( (*ppBlob = f_new FlmBlobImp) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

Exit:

	return( rc);
}
