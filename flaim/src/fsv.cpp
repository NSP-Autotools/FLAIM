//-------------------------------------------------------------------------
// Desc:	FLAIM server functions
// Tabs:	3
//
// Copyright (c) 1998-2007 Novell, Inc. All Rights Reserved.
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

FSTATIC RCODE fsvIteratorParse(
	FSV_WIRE *		pWire, 
	F_Pool *			pPool);

FSTATIC RCODE fsvIteratorWhereParse(
	FSV_WIRE * 		pWire, 
	F_Pool * 		pPool);

FSTATIC RCODE fsvIteratorFromParse(
	FSV_WIRE *		pWire, 
	F_Pool * 		pPool);

FSTATIC RCODE fsvIteratorSelectParse(
	FSV_WIRE *		pWire,
	F_Pool * 		pPool);

FSTATIC RCODE fsvDbGetBlocks(
	HFDB				hDb,
	FLMUINT			uiAddress,
	FLMUINT			uiMinTransId,
	FLMUINT *		puiCount,
	FLMUINT *		puiBlocksExamined,
	FLMUINT *		puiNextBlkAddr,
	FLMUINT			uiFlags,
	F_Pool *			pPool,
	FLMBYTE **	 	ppBlocks,
	FLMUINT *		puiBytes);

FSTATIC RCODE fsvGetHandles(
	FSV_WIRE *		pWire);

FSV_SCTX *				gv_pGlobalContext = NULL;

/****************************************************************************
Desc: Initializes the server's global context.
****************************************************************************/
RCODE fsvInitGlobalContext(
	FLMUINT			uiMaxSessions,
	const char *	pszServerBasePath,
	FSV_LOG_FUNC	pLogFunc)
{
	RCODE				rc = FERR_OK;
	FSV_SCTX *		pTmpContext = NULL;

	if (gv_pGlobalContext)
	{

		// Context already initialized

		goto Exit;
	}

	if ((pTmpContext = f_new FSV_SCTX) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = pTmpContext->Setup( uiMaxSessions, pszServerBasePath,
				  pLogFunc)))
	{
		goto Exit;
	}

Exit:

	if (RC_BAD( rc))
	{
		if (pTmpContext)
		{
			pTmpContext->Release();
		}
	}
	else if (pTmpContext)
	{
		gv_pGlobalContext = pTmpContext;
	}

	return (rc);
}

/****************************************************************************
Desc: Frees any resources allocated to the global context.
****************************************************************************/
void fsvFreeGlobalContext(void)
{
	if (gv_pGlobalContext)
	{
		gv_pGlobalContext->Release();
		gv_pGlobalContext = NULL;
	}
}

/****************************************************************************
Desc: Sets the server's base (relative) path
****************************************************************************/
RCODE fsvSetBasePath(
	const char *		pszServerBasePath)
{
	RCODE 		rc = FERR_OK;

	if (!gv_pGlobalContext)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = gv_pGlobalContext->SetBasePath( pszServerBasePath)))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc: Sets the server's temporary directory
****************************************************************************/
RCODE fsvSetTempDir(
	const char *		pszTempDir)
{
	RCODE 		rc = FERR_OK;

	if (!gv_pGlobalContext)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = gv_pGlobalContext->SetTempDir( pszTempDir)))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc: Returns a pointer to the server's global context object.
****************************************************************************/
RCODE fsvGetGlobalContext(
	FSV_SCTX **	 ppGlobalContext)
{
	*ppGlobalContext = gv_pGlobalContext;
	return (FERR_OK);
}

/****************************************************************************
Desc: This is the function that processes FLAIM requests.
****************************************************************************/
RCODE fsvProcessRequest(
	FCS_DIS *	pDataIStream,
	FCS_DOS *	pDataOStream,
	F_Pool *		pScratchPool,
	FLMUINT *	puiSessionIdRV)
{
	void *		pvMark = NULL;
	FSV_WIRE 	Wire(pDataIStream, pDataOStream);
	RCODE			rc = FERR_OK;

	// Set the temporary pool

	if (pScratchPool)
	{
		pvMark = pScratchPool->poolMark();
		Wire.setPool( pScratchPool);
	}

	// Read the request

	if (RC_BAD( rc = Wire.read()))
	{
		goto Exit;
	}

	// Close the input stream.

	pDataIStream->close();
	Wire.setDIStream( NULL);

	// Get any required handles.

	if (RC_BAD( rc = fsvGetHandles( &Wire)))
	{
		goto Exit;
	}

	// Call the appropriate handler function.

	switch (Wire.getClass())
	{
		case FCS_OPCLASS_GLOBAL:
		{
			if (RC_BAD( rc = fsvOpClassGlobal( &Wire)))
			{
				goto Exit;
			}
			break;
		}

		case FCS_OPCLASS_SESSION:
		{
			if (RC_BAD( rc = fsvOpClassSession( &Wire)))
			{
				goto Exit;
			}
			break;
		}

		case FCS_OPCLASS_DATABASE:
		{
			if (RC_BAD( rc = fsvOpClassDatabase( &Wire)))
			{
				goto Exit;
			}
			break;
		}

		case FCS_OPCLASS_TRANS:
		{
			if (RC_BAD( rc = fsvOpClassTransaction( &Wire)))
			{
				goto Exit;
			}
			break;
		}

		case FCS_OPCLASS_RECORD:
		{
			if (RC_BAD( rc = fsvOpClassRecord( &Wire)))
			{
				goto Exit;
			}
			break;
		}

		case FCS_OPCLASS_ITERATOR:
		{
			if (RC_BAD( rc = fsvOpClassIterator( &Wire)))
			{
				goto Exit;
			}
			break;
		}

		case FCS_OPCLASS_BLOB:
		{
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			break;
		}

		case FCS_OPCLASS_DIAG:
		{
			if (RC_BAD( rc = fsvOpClassDiag( &Wire)))
			{
				goto Exit;
			}
			break;
		}

		case FCS_OPCLASS_FILE:
		{
			if (RC_BAD( rc = fsvOpClassFile( &Wire)))
			{
				goto Exit;
			}
			break;
		}

		case FCS_OPCLASS_ADMIN:
		{
			if (RC_BAD( rc = fsvOpClassAdmin( &Wire)))
			{
				goto Exit;
			}
			break;
		}

		case FCS_OPCLASS_INDEX:
		{
			if (RC_BAD( rc = fsvOpClassIndex( &Wire)))
			{
				goto Exit;
			}
			break;
		}

		case FCS_OPCLASS_MISC:
		{
			if (RC_BAD( rc = fsvOpClassMisc( &Wire)))
			{
				goto Exit;
			}
			break;
		}

		default:
		{
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
		}
	}

	if (puiSessionIdRV)
	{

		// Set the session ID so that the calling routine has the option of
		// performing cleanup on an error

		*puiSessionIdRV = Wire.getSessionId();
	}

Exit:

	if (RC_BAD( rc))
	{

		// If the input stream is still open, the handler never send any
		// data to the client. Close the input stream and try to send the
		// error code to the client.

		if (pDataIStream->isOpen())
		{
			(void) pDataIStream->close();
			Wire.setDIStream( NULL);
		}

		if (RC_OK( Wire.sendOpcode( Wire.getClass(), Wire.getOp())))
		{
			if (RC_OK( Wire.sendRc( rc)))
			{
				if (RC_OK( Wire.sendTerminate()))
				{
					pDataOStream->close();
				}
			}
		}
	}
	else
	{
		pDataOStream->close();
	}

	if (pScratchPool)
	{
		pScratchPool->poolReset( pvMark);
	}

	return (rc);
}

/****************************************************************************
Desc: Performs a diagnostic operation
****************************************************************************/
RCODE fsvOpClassDiag(
	FSV_WIRE *	pWire)
{
	RCODE 		opRc = FERR_OK;
	RCODE 		rc = FERR_OK;

	// Service the request.

	switch (pWire->getOp())
	{
		case FCS_OP_DIAG_HTD_ECHO:
		{
			// Simply echo the record back to the client. This is done below
			// when the response is sent to the client.

			break;
		}

		default:
		{
			opRc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto OP_EXIT;
		}
	}

OP_EXIT:

	// Send the server's response.

	if (RC_BAD( rc = pWire->sendOpcode( FCS_OPCLASS_DIAG, pWire->getOp())))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pWire->sendRc( opRc)))
	{
		goto Exit;
	}

	if (RC_OK( opRc))
	{
		switch (pWire->getOp())
		{
			case FCS_OP_DIAG_HTD_ECHO:
			{
				if (pWire->getRecord() != NULL)
				{
					if (RC_BAD( rc = pWire->sendRecord( WIRE_VALUE_HTD,
								  pWire->getRecord())))
					{
						goto Exit;
					}
				}
				break;
			}
		}
	}

	if (RC_BAD( rc = pWire->sendTerminate()))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc: Performs a file system operation
****************************************************************************/
RCODE fsvOpClassFile(
	FSV_WIRE *		pWire)
{
	RCODE				rc = FERR_OK;
	RCODE				opRc = FERR_OK;
	FSV_SCTX *		pServerContext = NULL;
	FLMUNICODE *	puzSourcePath;
	char				szSourcePath[F_PATH_MAX_SIZE];

	// Set up local variables.

	if (RC_BAD( opRc = fsvGetGlobalContext( &pServerContext)))
	{
		goto OP_EXIT;
	}

	puzSourcePath = pWire->getFilePath();
	if (puzSourcePath)
	{

		// Convert the UNICODE URL to a server path.

		if (RC_BAD( rc = pServerContext->BuildFilePath( puzSourcePath,
					  szSourcePath)))
		{
			goto Exit;
		}
	}

	// Service the request.

	switch (pWire->getOp())
	{
		case FCS_OP_FILE_EXISTS:
		{
			if (!puzSourcePath)
			{
				opRc = RC_SET( FERR_SYNTAX);
				goto OP_EXIT;
			}

			if (RC_BAD( opRc = gv_FlmSysData.pFileSystem->doesFileExist( 
				szSourcePath)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_FILE_DELETE:
		{
			if (!puzSourcePath)
			{
				opRc = RC_SET( FERR_SYNTAX);
				goto OP_EXIT;
			}

			if (RC_BAD( opRc = gv_FlmSysData.pFileSystem->deleteFile( 
				szSourcePath)))
			{
				goto OP_EXIT;
			}
			break;
		}

		default:
		{
			opRc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto OP_EXIT;
		}
	}

OP_EXIT:

	// Send the server's response.

	if (RC_BAD( rc = pWire->sendOpcode( FCS_OPCLASS_FILE, pWire->getOp())))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pWire->sendRc( opRc)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pWire->sendTerminate()))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc: Performs an administrative operation
****************************************************************************/
RCODE fsvOpClassAdmin(
	FSV_WIRE *	pWire)
{
	RCODE 		opRc = FERR_OK;
	RCODE 		rc = FERR_OK;

	opRc = RC_SET( FERR_NOT_IMPLEMENTED);
	goto OP_EXIT;

OP_EXIT:

	// Send the server's response.

	if (RC_BAD( rc = pWire->sendOpcode( FCS_OPCLASS_ADMIN, pWire->getOp())))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pWire->sendRc( opRc)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pWire->sendTerminate()))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc: Performs a global operation
****************************************************************************/
RCODE fsvOpClassGlobal(
	FSV_WIRE *	pWire)
{
	FSV_SCTX *	pServerContext;
	NODE *		pTree = NULL;
	RCODE			opRc = FERR_OK;
	RCODE			rc = FERR_OK;

	// Service the request.

	if (RC_BAD( rc = fsvGetGlobalContext( &pServerContext)))
	{
		goto Exit;
	}

	switch (pWire->getOp())
	{
		case FCS_OP_GLOBAL_STATS_START:
		{
			if (RC_BAD( opRc = FlmConfig( FLM_START_STATS, 0, 0)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_GLOBAL_STATS_STOP:
		{
			if (RC_BAD( opRc = FlmConfig( FLM_STOP_STATS, 0, 0)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_GLOBAL_STATS_RESET:
		{
			if (RC_BAD( opRc = FlmConfig( FLM_RESET_STATS, 0, 0)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_GLOBAL_MEM_INFO_GET:
		{
			FLM_MEM_INFO	memInfo;

			FlmGetMemoryInfo( &memInfo);
			if (RC_BAD( opRc = fcsBuildMemInfo( &memInfo, 
				pWire->getPool(), &pTree)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_GLOBAL_GET_THREAD_INFO:
		{
			if (RC_BAD( opRc = fcsBuildThreadInfo( pWire->getPool(), &pTree)))
			{
				goto OP_EXIT;
			}
			break;
		}

		default:
		{
			opRc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto OP_EXIT;
		}
	}

OP_EXIT:

	// Send the server's response.

	if (RC_BAD( rc = pWire->sendOpcode( FCS_OPCLASS_GLOBAL, pWire->getOp())))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pWire->sendRc( opRc)))
	{
		goto Exit;
	}

	if (RC_OK( opRc))
	{
		if (pTree)
		{
			if (RC_BAD( rc = pWire->sendHTD( WIRE_VALUE_HTD, pTree)))
			{
				goto Exit;
			}
		}
	}

	if (RC_BAD( rc = pWire->sendTerminate()))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc: Performs a session operation
****************************************************************************/
RCODE fsvOpClassSession(
	FSV_WIRE *	pWire)
{
	FLMUINT		uiSessionIdRV;
	FSV_SCTX *	pServerContext;
	FSV_SESN *	pSession = NULL;
	RCODE			opRc = FERR_OK;
	RCODE			rc = FERR_OK;

	// Service the request.

	if (RC_BAD( opRc = fsvGetGlobalContext( &pServerContext)))
	{
		goto OP_EXIT;
	}

	switch (pWire->getOp())
	{
		case FCS_OP_SESSION_OPEN:
		{
			// Create a new session.

			if (RC_BAD( opRc = pServerContext->OpenSession(
							  pWire->getClientVersion(), pWire->getFlags(),
						  &uiSessionIdRV, &pSession)))
			{
				goto OP_EXIT;
			}

			break;
		}

		case FCS_OP_SESSION_CLOSE:
		{
			// Close the session.

			if (RC_BAD( opRc = pServerContext->CloseSession( pWire->getSessionId())))
			{
				goto OP_EXIT;
			}
			break;
		}

		default:
		{
			opRc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto OP_EXIT;
		}
	}

OP_EXIT:

	// Send the response.

	if (RC_BAD( rc = pWire->sendOpcode( FCS_OPCLASS_SESSION, pWire->getOp())))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pWire->sendRc( opRc)))
	{
		goto Exit;
	}

	if (RC_OK( opRc))
	{
		if (pWire->getOp() == FCS_OP_SESSION_OPEN)
		{
			if (RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_SESSION_ID,
						  uiSessionIdRV)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_SESSION_COOKIE,
						  pSession->getCookie())))
			{
				goto Exit;
			}

			if (RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_FLAGS,
						  FCS_SESSION_GEDCOM_SUPPORT)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_FLAIM_VERSION,
						  FLM_CUR_FILE_FORMAT_VER_NUM)))
			{
				goto Exit;
			}
		}
	}

	if (RC_BAD( rc = pWire->sendTerminate()))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc: Performs a record or DRN operation
****************************************************************************/
RCODE fsvOpClassRecord(
	FSV_WIRE *		pWire)
{
	FSV_SESN *		pSession;
	HFDB				hDb;
	FLMUINT			uiContainer;
	FLMUINT			uiIndex;
	FLMUINT			uiAutoTrans;
	FLMUINT			uiDrn;
	FLMUINT			uiFlags;
	FlmRecord *		pRecord = NULL;
	FlmRecord *		pRecordRV = NULL;
	FLMUINT			uiDrnRV = 0;
	RCODE				opRc = FERR_OK;
	RCODE				rc = FERR_OK;

	// Get a pointer to the session object.

	if ((pSession = pWire->getSession()) == NULL)
	{
		opRc = RC_SET( FERR_BAD_HDL);
		goto OP_EXIT;
	}

	// Get the database handle. This is needed by all of the record
	// operations.

	if ((hDb = (HFDB) pWire->getFDB()) == HFDB_NULL)
	{
		opRc = RC_SET( FERR_BAD_HDL);
		goto OP_EXIT;
	}

	// Initialize local variables.

	uiContainer = pWire->getContainerId();
	uiIndex = pWire->getIndexId();
	uiDrn = pWire->getDrn();
	uiAutoTrans = pWire->getAutoTrans();
	uiFlags = pWire->getFlags();
	pRecord = pWire->getRecord();

	// Perform the operation.

	switch (pWire->getOp())
	{
		case FCS_OP_RECORD_RETRIEVE:
		{
			if (!uiFlags)
			{
				uiFlags = FO_EXACT;
			}

			if (pWire->getBoolean())
			{

				// Fetch the record

				if (RC_BAD( opRc = FlmRecordRetrieve( hDb, uiContainer, uiDrn,
							  uiFlags, &pRecordRV, &uiDrnRV)))
				{
					goto OP_EXIT;
				}
			}
			else
			{

				// Just get the DRN

				if (RC_BAD( opRc = FlmRecordRetrieve( hDb, uiContainer, uiDrn,
							  uiFlags, NULL, &uiDrnRV)))
				{
					goto OP_EXIT;
				}
			}
			break;
		}

		case FCS_OP_RECORD_ADD:
		{
			uiDrnRV = uiDrn;
			if (RC_BAD( opRc = FlmRecordAdd( hDb, uiContainer, &uiDrnRV, pRecord,
						  uiAutoTrans)))
			{
				goto OP_EXIT;
			}

			break;
		}

		case FCS_OP_RECORD_MODIFY:
		{
			if (RC_BAD( opRc = FlmRecordModify( hDb, uiContainer, uiDrn, pRecord,
						  uiAutoTrans)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_RECORD_DELETE:
		{
			if (RC_BAD( opRc = FlmRecordDelete( hDb, uiContainer, uiDrn,
						  uiAutoTrans)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_RESERVE_NEXT_DRN:
		{
			uiDrnRV = uiDrn;
			if (RC_BAD( opRc = FlmReserveNextDrn( hDb, uiContainer, &uiDrnRV)))
			{
				goto OP_EXIT;
			}

			break;
		}

		case FCS_OP_KEY_RETRIEVE:
		{
			if (pSession->getClientVersion() >= FCS_VERSION_1_1_1)
			{
				if (RC_BAD( opRc = FlmKeyRetrieve( hDb, uiIndex, uiContainer,
							  pRecord, uiDrn, uiFlags, &pRecordRV, &uiDrnRV)))
				{
					goto OP_EXIT;
				}
			}
			else
			{
				FLMUINT	uiKeyContainer = 0;

				if (pRecord)
				{
					uiKeyContainer = pRecord->getContainerID();
				}

				// Older clients sent index # in the container tag.

				if (RC_BAD( opRc = FlmKeyRetrieve( hDb, uiContainer, uiKeyContainer,
							  pRecord, uiDrn, uiFlags, &pRecordRV, &uiDrnRV)))
				{
					goto OP_EXIT;
				}
			}
			break;
		}

		default:
		{
			opRc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto OP_EXIT;
		}
	}

OP_EXIT:

	// Send the server's response.

	if (RC_BAD( rc = pWire->sendOpcode( FCS_OPCLASS_RECORD, pWire->getOp())))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pWire->sendRc( opRc)))
	{
		goto Exit;
	}

	if (RC_OK( opRc))
	{
		if (pRecordRV)
		{
			if (RC_BAD( rc = pWire->sendRecord( WIRE_VALUE_RECORD, pRecordRV)))
			{
				goto Exit;
			}
		}

		if (uiDrnRV)
		{
			if (RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_DRN, uiDrnRV)))
			{
				goto Exit;
			}
		}
	}

	if (RC_BAD( rc = pWire->sendTerminate()))
	{
		goto Exit;
	}

Exit:

	if (pRecordRV)
	{
		pRecordRV->Release();
	}

	return (rc);
}

/****************************************************************************
Desc: Performs a database operation.
****************************************************************************/
RCODE fsvOpClassDatabase(
	FSV_WIRE *		pWire)
{
	RCODE				rc = FERR_OK;
	RCODE				opRc = FERR_OK;
	FSV_SESN *		pSession;
	HFDB				hDb = HFDB_NULL;
	CREATE_OPTS 	CreateOptsRV;
	FLMUINT			uiBlockCountRV = 0;
	FLMUINT			uiBlocksExaminedRV = 0;
	FLMUINT			uiBlockAddrRV = 0;
	FLMUINT			uiTransIdRV;
	FLMUINT64		ui64NumValue1RV = 0;
	FLMUINT64		ui64NumValue2RV = 0;
	FLMUINT64		ui64NumValue3RV = 0;
	FLMBOOL			bBoolValueRV = FALSE;
	FLMUINT			uiItemIdRV = 0;
	char				szItemName[64];
	NODE *			pHTDRV = NULL;
	char				szPathRV[F_PATH_MAX_SIZE];
	F_NameTable 	nameTable;
	FLMBOOL			bHaveCreateOptsVal = FALSE;
	FLMBOOL			bHavePathValue = FALSE;
	FLMBYTE *		pBinary = NULL;
	FLMUINT			uiBinSize = 0;
	
	szItemName[0] = 0;

	if ((pSession = pWire->getSession()) == NULL)
	{
		opRc = RC_SET( FERR_BAD_HDL);
		goto OP_EXIT;
	}

	if (pWire->getOp() != FCS_OP_DATABASE_OPEN &&
		 pWire->getOp() != FCS_OP_DATABASE_CREATE)
	{

		// Get the database handle for all database operations other than
		// open and create.

		if ((hDb = (HFDB) pWire->getFDB()) == HFDB_NULL)
		{
			opRc = RC_SET( FERR_BAD_HDL);
			goto OP_EXIT;
		}
	}

	switch (pWire->getOp())
	{
		case FCS_OP_DATABASE_OPEN:
		{
			if (RC_BAD( opRc = pSession->OpenDatabase( pWire->getFilePath(),
						  pWire->getFilePath3(), pWire->getFilePath2(),
						  pWire->getFlags())))
			{
				goto OP_EXIT;
			}

			break;
		}

		case FCS_OP_DATABASE_CREATE:
		{
			CREATE_OPTS createOpts;
			pWire->copyCreateOpts( &createOpts);

			if (RC_BAD( opRc = pSession->CreateDatabase( pWire->getFilePath(),
						  pWire->getFilePath3(), pWire->getFilePath2(),
						  pWire->getDictPath(), pWire->getDictBuffer(), &createOpts)))
			{
				goto OP_EXIT;
			}

			break;
		}

		case FCS_OP_DATABASE_CLOSE:
		{
			if (RC_BAD( opRc = pSession->CloseDatabase()))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_DB_REDUCE_SIZE:
		{
			if (RC_BAD( opRc = FlmDbReduceSize( hDb, (FLMUINT) pWire->getCount(),
						  &uiBlockCountRV)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_GET_ITEM_NAME:
		{
			if (RC_BAD( opRc = FlmGetItemName( hDb, pWire->getItemId(),
						  sizeof(szItemName), szItemName)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_GET_NAME_TABLE:
		{
			if (RC_BAD( rc = nameTable.setupFromDb( hDb)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_GET_COMMIT_CNT:
		{
			if (RC_BAD( opRc = FlmDbGetCommitCnt( hDb, &uiBlockCountRV)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_GET_TRANS_ID:
		{
			if (RC_BAD( opRc = FlmDbGetTransId( hDb, &uiTransIdRV)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_DATABASE_GET_CONFIG:
		{
			switch ((eDbGetConfigType) pWire->getType())
			{
				case FDB_GET_VERSION:
				{
					// Doing via create opts to maintain backward compatibility.

					f_memset( &CreateOptsRV, 0, sizeof(CreateOptsRV));
					if (RC_BAD( opRc = FlmDbGetConfig( hDb, FDB_GET_VERSION,
								  (void *) &CreateOptsRV.uiVersionNum)))
					{
						goto OP_EXIT;
					}

					bHaveCreateOptsVal = TRUE;
					break;
				}
				
				case FDB_GET_BLKSIZ:
				{
					// Doing via create opts to maintain backward compatibility.

					f_memset( &CreateOptsRV, 0, sizeof(CreateOptsRV));
					if (RC_BAD( opRc = FlmDbGetConfig( hDb, FDB_GET_BLKSIZ,
								  (void *) &CreateOptsRV.uiBlockSize)))
					{
						goto OP_EXIT;
					}

					bHaveCreateOptsVal = TRUE;
					break;
				}
				
				case FDB_GET_DEFAULT_LANG:
				{
					// Doing via create opts to maintain backward compatibility.

					f_memset( &CreateOptsRV, 0, sizeof(CreateOptsRV));
					if (RC_BAD( opRc = FlmDbGetConfig( hDb, FDB_GET_DEFAULT_LANG,
								  (void *) &CreateOptsRV.uiDefaultLanguage)))
					{
						goto OP_EXIT;
					}

					bHaveCreateOptsVal = TRUE;
					break;
				}
				
				case FDB_GET_TRANS_ID:
				case FDB_GET_RFL_FILE_NUM:
				case FDB_GET_RFL_HIGHEST_NU:
				case FDB_GET_LAST_BACKUP_TRANS_ID:
				case FDB_GET_BLOCKS_CHANGED_SINCE_BACKUP:
				case FDB_GET_FILE_EXTEND_SIZE:
				case FDB_GET_APP_DATA:
				{
					FLMUINT	uiTmpValue;

					if (RC_BAD( opRc = FlmDbGetConfig( hDb,
								  (eDbGetConfigType) pWire->getType(),
								  (void *) &uiTmpValue)))
					{
						goto OP_EXIT;
					}

					ui64NumValue1RV = (FLMUINT64) uiTmpValue;
					break;
				}

				case FDB_GET_RFL_FILE_SIZE_LIMITS:
				{
					FLMUINT	uiTmpValue1;
					FLMUINT	uiTmpValue2;

					if (RC_BAD( opRc = FlmDbGetConfig( hDb,
								  FDB_GET_RFL_FILE_SIZE_LIMITS, 
								  (void *) &uiTmpValue1,
								  (void *) &uiTmpValue2)))
					{
						goto OP_EXIT;
					}

					ui64NumValue1RV = (FLMUINT64) uiTmpValue1;
					ui64NumValue2RV = (FLMUINT64) uiTmpValue2;
					break;
				}

				case FDB_GET_RFL_KEEP_FLAG:
				case FDB_GET_AUTO_TURN_OFF_KEEP_RFL_FLAG:
				case FDB_GET_KEEP_ABORTED_TRANS_IN_RFL_FLAG:
				{
					if (RC_BAD( opRc = FlmDbGetConfig( hDb,
								  (eDbGetConfigType) pWire->getType(),
								  (void *) &bBoolValueRV)))
					{
						goto OP_EXIT;
					}
					break;
				}

				case FDB_GET_PATH:
				{
					if (RC_BAD( opRc = FlmDbGetConfig( hDb, FDB_GET_PATH,
								  (void *) szPathRV)))
					{
						goto OP_EXIT;
					}

					bHavePathValue = TRUE;
					break;
				}

				case FDB_GET_CHECKPOINT_INFO:
				{
					CHECKPOINT_INFO	checkpointInfo;

					if (RC_BAD( opRc = FlmDbGetConfig( hDb, 
						FDB_GET_CHECKPOINT_INFO, (void *) &checkpointInfo)))
					{
						goto OP_EXIT;
					}

					if (RC_BAD( opRc = fcsBuildCheckpointInfo( &checkpointInfo,
								  pWire->getPool(), &pHTDRV)))
					{
						goto OP_EXIT;
					}
					break;
				}

				case FDB_GET_LOCK_HOLDER:
				{
					F_LOCK_USER		lockUser;

					if (RC_BAD( opRc = FlmDbGetConfig( hDb, FDB_GET_LOCK_HOLDER,
								  (void *) &lockUser)))
					{
						goto OP_EXIT;
					}

					if (RC_BAD( opRc = fcsBuildLockUser( &lockUser, FALSE,
								  pWire->getPool(), &pHTDRV)))
					{
						goto OP_EXIT;
					}
					break;
				}

				case FDB_GET_LOCK_WAITERS:
				{
					F_LOCK_USER *		pLockUser = NULL;

					if (RC_BAD( opRc = FlmDbGetConfig( hDb, FDB_GET_LOCK_WAITERS,
								  (void *) &pLockUser)))
					{
						if (pLockUser)
						{
							f_free( &pLockUser);
						}

						goto OP_EXIT;
					}

					if (RC_BAD( opRc = fcsBuildLockUser( pLockUser, TRUE,
								  pWire->getPool(), &pHTDRV)))
					{
						if (pLockUser)
						{
							f_free( &pLockUser);
						}

						goto OP_EXIT;
					}

					if (pLockUser)
					{
						f_free( &pLockUser);
					}
					break;
				}

				case FDB_GET_RFL_DIR:
				{
					if (RC_BAD( opRc = FlmDbGetConfig( hDb, FDB_GET_RFL_DIR,
								  (void *) szPathRV)))
					{
						goto OP_EXIT;
					}

					bHavePathValue = TRUE;
					break;
				}

				case FDB_GET_SERIAL_NUMBER:
				{
					uiBinSize = F_SERIAL_NUM_SIZE;

					if( RC_BAD( opRc = pWire->getPool()->poolAlloc( 
						uiBinSize, (void **)&pBinary)))
					{
						goto OP_EXIT;
					}
					
					if (RC_BAD( opRc = FlmDbGetConfig( hDb, 
						FDB_GET_SERIAL_NUMBER, (void *) pBinary)))
					{
						goto OP_EXIT;
					}
					break;
				}

				case FDB_GET_SIZES:
				{
					if (RC_BAD( opRc = FlmDbGetConfig( hDb, FDB_GET_SIZES,
								  (void *) &ui64NumValue1RV, 
								  (void *) &ui64NumValue2RV,
								  (void *) &ui64NumValue3RV)))
					{
						goto OP_EXIT;
					}
					break;
				}

				default:
				{
					opRc = RC_SET( FERR_NOT_IMPLEMENTED);
					goto OP_EXIT;
				}
			}
			break;
		}

		case FCS_OP_DATABASE_CONFIG:
		{
			switch ((eDbConfigType) pWire->getType())
			{
				case FDB_SET_APP_VERSION:
				case FDB_RFL_KEEP_FILES:
				case FDB_RFL_ROLL_TO_NEXT_FILE:
				case FDB_KEEP_ABORTED_TRANS_IN_RFL:
				case FDB_AUTO_TURN_OFF_KEEP_RFL:
				case FDB_SET_APP_DATA:
				{
					if (RC_BAD( opRc = FlmDbConfig( hDb,
								  (eDbConfigType) pWire->getType(),
								  (void *) ((FLMUINT) pWire->getNumber2()),
								  (void *) ((FLMUINT) pWire->getNumber3()))))
					{
						goto OP_EXIT;
					}
					break;
				}
				
				case FDB_RFL_FILE_LIMITS:
				case FDB_FILE_EXTEND_SIZE:
				{
					if (RC_BAD( opRc = FlmDbConfig( hDb,
								  (eDbConfigType) pWire->getType(),
								  (void *) ((FLMUINT) pWire->getNumber1()),
								  (void *) ((FLMUINT) pWire->getNumber2()))))
					{
						goto OP_EXIT;
					}
					break;
				}
				
				case FDB_RFL_DIR:
				{
					char *		pszPath;
					F_Pool *		pPool = pWire->getPool();
					void *		pvMark = pPool->poolMark();

					if (RC_BAD( rc = fcsConvertUnicodeToNative( pPool,
								  pWire->getFilePath(), &pszPath)))
					{
						goto Exit;
					}

					if (RC_BAD( opRc = FlmDbConfig( hDb,
								  (eDbConfigType) pWire->getType(), 
								  (void *) pszPath,
								  (void *) ((FLMUINT) pWire->getNumber3()))))
					{
						goto OP_EXIT;
					}

					pPool->poolReset( pvMark);
					break;
				}

				default:
				{
					opRc = RC_SET( FERR_NOT_IMPLEMENTED);
					goto OP_EXIT;
				}
			}
			break;
		}

		case FCS_OP_DATABASE_LOCK:
		{
			if (RC_BAD( opRc = FlmDbLock( hDb,
						  (eLockType) (FLMUINT) pWire->getNumber1(),
						  (FLMINT) pWire->getSignedValue(), 
						  (FLMUINT) pWire->getFlags())))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_DATABASE_UNLOCK:
		{
			if (RC_BAD( opRc = FlmDbUnlock( hDb)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_DATABASE_GET_BLOCK:
		{
			uiBlockCountRV = (FLMUINT) pWire->getCount();
			if (RC_BAD( opRc = fsvDbGetBlocks( hDb, pWire->getAddress(),
						  pWire->getTransId(), &uiBlockCountRV, &uiBlocksExaminedRV,
						  &uiBlockAddrRV, pWire->getFlags(), pWire->getPool(),
						  &pBinary, &uiBinSize)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_DATABASE_CHECKPOINT:
		{
			if (RC_BAD( opRc = FlmDbCheckpoint( hDb, pWire->getFlags())))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_DB_SET_BACKUP_FLAG:
		{
			FLMBOOL	bNewState = pWire->getBoolean();

			if (!IsInCSMode( hDb))
			{
				FDB *		pDb = (FDB *) hDb;

				f_mutexLock( gv_FlmSysData.hShareMutex);
				if (pDb->pFile->bBackupActive && bNewState)
				{
					f_mutexUnlock( gv_FlmSysData.hShareMutex);
					opRc = RC_SET( FERR_BACKUP_ACTIVE);
					goto OP_EXIT;
				}

				pDb->pFile->bBackupActive = bNewState;
				f_mutexUnlock( gv_FlmSysData.hShareMutex);
			}
			else
			{
				if (RC_BAD( opRc = fcsSetBackupActiveFlag( hDb, bNewState)))
				{
					goto OP_EXIT;
				}
			}
			break;
		}

		default:
		{
			opRc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto OP_EXIT;
		}
	}

OP_EXIT:

	// Send the server's response.

	if (RC_BAD( rc = pWire->sendOpcode( FCS_OPCLASS_DATABASE, pWire->getOp())))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pWire->sendRc( opRc)))
	{
		goto Exit;
	}

	switch (pWire->getOp())
	{
		case FCS_OP_DB_REDUCE_SIZE:
		case FCS_OP_GET_COMMIT_CNT:
		{

			// Return a count

			if (RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_COUNT, uiBlockCountRV)))
			{
				goto Exit;
			}
			break;
		}

		case FCS_OP_GET_NAME_TABLE:
		{

			// Return the name table.

			if (RC_OK( opRc))
			{
				if (RC_BAD( rc = pWire->sendNameTable( WIRE_VALUE_NAME_TABLE,
							  &nameTable)))
				{
					goto Exit;
				}
			}
			break;
		}

		case FCS_OP_GET_ITEM_NAME:
		{
			FLMUNICODE *	puzItemNameRV;

			if (RC_OK( opRc))
			{
				if (szItemName[0])
				{
					if (RC_BAD( rc = fcsConvertNativeToUnicode( pWire->getPool(),
								  szItemName, &puzItemNameRV)))
					{
						goto Exit;
					}

					if (RC_BAD( rc = pWire->sendString( WIRE_VALUE_ITEM_NAME,
								  puzItemNameRV)))
					{
						goto Exit;
					}
				}
			}
			break;
		}

		case FCS_OP_GET_ITEM_ID:
		{
			if (uiItemIdRV)
			{
				if (RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_ITEM_ID, uiItemIdRV)))
				{
					goto Exit;
				}
			}
			break;
		}

		case FCS_OP_GET_TRANS_ID:
		{

			// Return the transaction id for the database.

			if (RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_TRANSACTION_ID,
						  uiTransIdRV)))
			{
				goto Exit;
			}
			break;
		}

		case FCS_OP_DATABASE_GET_BLOCK:
		{

			// Return the requested block

			if (RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_COUNT, uiBlockCountRV)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_NUMBER2,
						  uiBlocksExaminedRV)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_ADDRESS, uiBlockAddrRV)))
			{
				goto Exit;
			}

			if (uiBlockCountRV)
			{
				if (RC_BAD( rc = pWire->sendBinary( WIRE_VALUE_BLOCK, pBinary,
							  uiBinSize)))
				{
					goto Exit;
				}
			}
			break;
		}

		case FCS_OP_DATABASE_GET_CONFIG:
		{
			switch (pWire->getType())
			{
				case FDB_GET_SERIAL_NUMBER:
					if (RC_BAD( rc = pWire->sendBinary( WIRE_VALUE_SERIAL_NUM, pBinary,
								  uiBinSize)))
					{
						goto Exit;
					}
					break;
				default:
					break;
			}
			break;
		}
	}

	if (bHaveCreateOptsVal)
	{
		if (RC_BAD( rc = pWire->sendCreateOpts( WIRE_VALUE_CREATE_OPTS,
					  &CreateOptsRV)))
		{
			goto Exit;
		}
	}

	if (ui64NumValue1RV)
	{
		if (RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_NUMBER1, ui64NumValue1RV)))
		{
			goto Exit;
		}
	}

	if (ui64NumValue2RV)
	{
		if (RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_NUMBER2, ui64NumValue2RV)))
		{
			goto Exit;
		}
	}

	if (ui64NumValue3RV)
	{
		if (RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_NUMBER3, ui64NumValue3RV)))
		{
			goto Exit;
		}
	}

	if (bBoolValueRV)
	{
		if (RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_BOOLEAN, bBoolValueRV)))
		{
			goto Exit;
		}
	}

	if (bHavePathValue)
	{
		FLMUNICODE *		puzPath;

		if (RC_BAD( rc = fcsConvertNativeToUnicode( pWire->getPool(), szPathRV,
					  &puzPath)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = pWire->sendString( WIRE_VALUE_FILE_PATH, puzPath)))
		{
			goto Exit;
		}
	}

	if (pHTDRV)
	{
		if (RC_BAD( rc = pWire->sendHTD( WIRE_VALUE_HTD, pHTDRV)))
		{
			goto Exit;
		}
	}

	if (RC_BAD( rc = pWire->sendTerminate()))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc: Performs an iterator (cursor) operation
****************************************************************************/
RCODE fsvOpClassIterator(
	FSV_WIRE *		pWire)
{
	RCODE				rc = FERR_OK;
	RCODE				opRc = FERR_OK;
	FSV_SESN *		pSession = NULL;
	HFCURSOR			hIterator = HFCURSOR_NULL;
	FLMBOOL			bDoDrnOp = FALSE;
	FlmRecord *		pRecordRV = NULL;
	FlmRecord *		pTmpRecord = NULL;
	FLMUINT			uiIteratorIdRV = FCS_INVALID_ID;
	FLMUINT			uiCountRV = 0;
	FLMUINT			uiDrnRV = 0;
	FLMBOOL			bFlag = FALSE;

	// Get a pointer to the session object.

	if ((pSession = pWire->getSession()) == NULL)
	{
		opRc = RC_SET( FERR_BAD_HDL);
		goto OP_EXIT;
	}

	// Get the iterator handle.

	if ((hIterator = pWire->getIteratorHandle()) == HFDB_NULL)
	{
		if (pWire->getOp() != FCS_OP_ITERATOR_INIT)
		{
			opRc = RC_SET( FERR_BAD_HDL);
			goto OP_EXIT;
		}
	}

	// Examine the wire flags for the operation.

	bDoDrnOp = (FLMBOOL)
		(
			(pWire->getFlags() & FCS_ITERATOR_DRN_FLAG) ? (FLMBOOL) TRUE :
				(FLMBOOL) FALSE
		);

	// Perform the requested operation.

	switch (pWire->getOp())
	{
		case FCS_OP_ITERATOR_INIT:
		{
			// Build the query.

			if (RC_BAD( opRc = fsvIteratorParse( pWire, pWire->getPool())))
			{
				goto OP_EXIT;
			}

			uiIteratorIdRV = pWire->getIteratorId();
			break;
		}

		case FCS_OP_ITERATOR_FREE:
		{
			// Free the iterator.

			if (RC_BAD( opRc = pSession->FreeIterator( pWire->getIteratorId())))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_ITERATOR_FIRST:
		{
			// Retrieve the first record (or DRN) in the result set.

			if (bDoDrnOp)
			{
				if (RC_BAD( opRc = FlmCursorFirstDRN( hIterator, &uiDrnRV)))
				{
					goto OP_EXIT;
				}
			}
			else
			{
				if (RC_BAD( opRc = FlmCursorFirst( hIterator, &pRecordRV)))
				{
					goto OP_EXIT;
				}
			}
			break;
		}

		case FCS_OP_ITERATOR_LAST:
		{
			// Retrieve the last record (or DRN) in the result set.

			if (bDoDrnOp)
			{
				if (RC_BAD( opRc = FlmCursorLastDRN( hIterator, &uiDrnRV)))
				{
					goto OP_EXIT;
				}
			}
			else
			{
				if (RC_BAD( opRc = FlmCursorLast( hIterator, &pRecordRV)))
				{
					goto OP_EXIT;
				}
			}
			break;
		}

		case FCS_OP_ITERATOR_NEXT:
		{
			// Retrieve the next record (or DRN) in the result set.

			if (bDoDrnOp)
			{
				if (RC_BAD( opRc = FlmCursorNextDRN( hIterator, &uiDrnRV)))
				{
					goto OP_EXIT;
				}
			}
			else
			{
				if (RC_BAD( opRc = FlmCursorNext( hIterator, &pRecordRV)))
				{
					goto OP_EXIT;
				}
			}
			break;
		}

		case FCS_OP_ITERATOR_PREV:
		{
			// Retrieve the previous record (or DRN) in the result set.

			if (bDoDrnOp)
			{
				if (RC_BAD( opRc = FlmCursorPrevDRN( hIterator, &uiDrnRV)))
				{
					goto OP_EXIT;
				}
			}
			else
			{
				if (RC_BAD( opRc = FlmCursorPrev( hIterator, &pRecordRV)))
				{
					goto OP_EXIT;
				}
			}
			break;
		}

		case FCS_OP_ITERATOR_COUNT:
		{
			// Count the number of records in the result set.

			if (RC_BAD( opRc = FlmCursorRecCount( hIterator, &uiCountRV)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_ITERATOR_TEST_REC:
		{
			if ((pTmpRecord = pWire->getRecord()) != NULL)
			{
				pTmpRecord->AddRef();

				if (RC_BAD( opRc = FlmCursorTestRec( hIterator, pTmpRecord, &bFlag)))
				{
					goto OP_EXIT;
				}

				pTmpRecord->Release();
				pTmpRecord = NULL;
			}
			else
			{
				if (RC_BAD( opRc = FlmCursorTestDRN( hIterator, pWire->getDrn(),
							  &bFlag)))
				{
					goto OP_EXIT;
				}
			}
			break;
		}

		default:
		{
			opRc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto OP_EXIT;
		}
	}

OP_EXIT:

	// Send the server's response.

	if (RC_BAD( rc = pWire->sendOpcode( FCS_OPCLASS_ITERATOR, pWire->getOp())))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pWire->sendRc( opRc)))
	{
		goto Exit;
	}

	if (RC_OK( opRc))
	{
		if (pRecordRV)
		{

			// Send the retrieved record.

			if (RC_BAD( rc = pWire->sendRecord( WIRE_VALUE_RECORD, pRecordRV)))
			{
				goto Exit;
			}
		}

		if (uiDrnRV)
		{

			// Send the record's DRN.

			if (RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_DRN, uiDrnRV)))
			{
				goto Exit;
			}
		}

		if (uiCountRV)
		{

			// Send the record count.

			if (RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_RECORD_COUNT, uiCountRV)))
			{
				goto Exit;
			}
		}

		if (uiIteratorIdRV != FCS_INVALID_ID)
		{

			// Send the iterator's ID.

			if (RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_ITERATOR_ID,
						  uiIteratorIdRV)))
			{
				goto Exit;
			}
		}

		if (bFlag)
		{
			if (RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_BOOLEAN, bFlag)))
			{
				goto Exit;
			}
		}
	}

	if (RC_BAD( rc = pWire->sendTerminate()))
	{
		goto Exit;
	}

Exit:

	if (pRecordRV)
	{
		pRecordRV->Release();
	}

	if (pTmpRecord)
	{
		pTmpRecord->Release();
		pTmpRecord = NULL;
	}

	return (rc);
}

/****************************************************************************
Desc: Performs a transaction operation
****************************************************************************/
RCODE fsvOpClassTransaction(
	FSV_WIRE *	pWire)
{
	RCODE				rc = FERR_OK;
	RCODE				opRc = FERR_OK;
	FSV_SESN *		pSession;
	HFDB				hDb;
	FLMUINT			uiTransTypeRV;
	FLMBYTE *		pBlock = NULL;
	FLMUINT			uiBlockSize = 0;
	FLMUINT			uiFlmTransFlags = 0;

	// Get a pointer to the session object.

	if ((pSession = pWire->getSession()) == NULL)
	{
		opRc = RC_SET( FERR_BAD_HDL);
		goto OP_EXIT;
	}

	// Get a handle to the database in case this is a database transaction
	// operation

	hDb = (HFDB) pWire->getFDB();

	// Perform the requested operation.

	switch (pWire->getOp())
	{
		case FCS_OP_TRANSACTION_BEGIN:
		{
			// Start a database transaction.

			if (pWire->getFlags() & FCS_TRANS_FLAG_GET_HEADER)
			{
				uiBlockSize = 2048;
				
				if( RC_BAD( rc = pWire->getPool()->poolAlloc( uiBlockSize,
					(void **)&pBlock)))
				{
					goto OP_EXIT;
				}
			}

			if (pWire->getFlags() & FCS_TRANS_FLAG_DONT_KILL)
			{
				uiFlmTransFlags |= FLM_DONT_KILL_TRANS;
			}

			if (pWire->getFlags() & FCS_TRANS_FLAG_DONT_POISON)
			{
				uiFlmTransFlags |= FLM_DONT_POISON_CACHE;
			}

			if (RC_BAD( opRc = FlmDbTransBegin( hDb,
						  pWire->getTransType() | uiFlmTransFlags,
						  pWire->getMaxLockWait(), pBlock)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_TRANSACTION_COMMIT:
		{
			// Commit a database transaction.

			if (RC_BAD( opRc = FlmDbTransCommit( hDb)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_TRANSACTION_COMMIT_EX:
		{
			// Commit a database transaction.

			if (RC_BAD( opRc = fsvDbTransCommitEx( hDb, pWire)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_TRANSACTION_ABORT:
		{
			// Abort a database transaction.

			if (RC_BAD( opRc = FlmDbTransAbort( hDb)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_TRANSACTION_GET_TYPE:
		{
			// Get the database transaction type.

			if (RC_BAD( opRc = FlmDbGetTransType( hDb, &uiTransTypeRV)))
			{
				goto OP_EXIT;
			}
			break;
		}

		default:
		{
			opRc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto OP_EXIT;
		}
	}

OP_EXIT:

	// Send the server's response.

	if (RC_BAD( rc = pWire->sendOpcode( FCS_OPCLASS_TRANS, pWire->getOp())))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pWire->sendRc( opRc)))
	{
		goto Exit;
	}

	if (pBlock)
	{
		if (RC_BAD( rc = pWire->sendBinary( WIRE_VALUE_BLOCK, 
			pBlock, uiBlockSize)))
		{
			goto Exit;
		}
	}

	if (RC_OK( opRc))
	{
		switch (pWire->getOp())
		{
			case FCS_OP_TRANSACTION_GET_TYPE:
			{
				if (RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_TRANSACTION_TYPE,
							  uiTransTypeRV)))
				{
					goto Exit;
				}
				break;
			}
		}
	}

	if (RC_BAD( rc = pWire->sendTerminate()))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc: Performs a maintenance operation.
****************************************************************************/
RCODE fsvOpClassMaintenance(
	FSV_WIRE *	pWire)
{
	FSV_SESN *	pSession;
	HFDB			hDb;
	F_Pool		pool;
	RCODE			opRc = FERR_OK;
	RCODE			rc = FERR_OK;

	// Initialize a temporary pool.

	pool.poolInit( 1024);

	// Service the request.

	if ((pSession = pWire->getSession()) == NULL)
	{
		opRc = RC_SET( FERR_BAD_HDL);
		goto OP_EXIT;
	}

	if ((hDb = (HFDB) pWire->getFDB()) == HFDB_NULL)
	{
		opRc = RC_SET( FERR_BAD_HDL);
		goto OP_EXIT;
	}

	switch (pWire->getOp())
	{
		case FCS_OP_CHECK:
			{
				if (RC_BAD( opRc = FlmDbCheck( hDb, NULL, NULL, NULL,
							  pWire->getFlags(), &pool, NULL, NULL, 0)))
				{
					goto OP_EXIT;
				}
				break;
			}

		default:
			{
				opRc = RC_SET( FERR_NOT_IMPLEMENTED);
				goto OP_EXIT;
			}
	}

OP_EXIT:

	// Send the server's response.

	if (RC_BAD( rc = pWire->sendOpcode( FCS_OPCLASS_MAINTENANCE, pWire->getOp())))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pWire->sendRc( opRc)))
	{
		goto Exit;
	}

	if (RC_OK( opRc))
	{
		switch (pWire->getOp())
		{
			case FCS_OP_CHECK:
			{
				break;
			}
		}
	}

	if (RC_BAD( rc = pWire->sendTerminate()))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc: Performs an index operation
****************************************************************************/
RCODE fsvOpClassIndex(
	FSV_WIRE *		pWire)
{
	HFDB				hDb = HFDB_NULL;
	FLMUINT			uiIndex;
	FINDEX_STATUS	indexStatus;
	F_Pool *			pTmpPool = pWire->getPool();
	RCODE				opRc = FERR_OK;
	RCODE				rc = FERR_OK;

	// Get the database handle. This is needed by all of the index
	// operations.

	if ((hDb = (HFDB) pWire->getFDB()) == HFDB_NULL)
	{
		opRc = RC_SET( FERR_BAD_HDL);
		goto OP_EXIT;
	}

	// Initialize local variables.

	uiIndex = pWire->getIndexId();

	// Service the request.

	switch (pWire->getOp())
	{
		case FCS_OP_INDEX_GET_STATUS:
		{
			if (RC_BAD( opRc = FlmIndexStatus( hDb, uiIndex, &indexStatus)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_INDEX_GET_NEXT:
		{
			if (RC_BAD( opRc = FlmIndexGetNext( hDb, &uiIndex)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_INDEX_SUSPEND:
		{
			if (RC_BAD( opRc = FlmIndexSuspend( hDb, uiIndex)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_INDEX_RESUME:
		{
			if (RC_BAD( opRc = FlmIndexResume( hDb, uiIndex)))
			{
				goto OP_EXIT;
			}
			break;
		}

		default:
		{
			opRc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto OP_EXIT;
		}
	}

OP_EXIT:

	// Send the server's response.

	if (RC_BAD( rc = pWire->sendOpcode( FCS_OPCLASS_INDEX, pWire->getOp())))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pWire->sendRc( opRc)))
	{
		goto Exit;
	}

	if (RC_OK( opRc))
	{
		switch (pWire->getOp())
		{
			case FCS_OP_INDEX_GET_STATUS:
			{
				NODE *		pStatusTree;

				if (RC_BAD( fcsBuildIndexStatus( &indexStatus, pTmpPool,
							  &pStatusTree)))
				{
					goto Exit;
				}

				if (RC_BAD( rc = pWire->sendHTD( WIRE_VALUE_HTD, pStatusTree)))
				{
					goto Exit;
				}
				break;
			}

			case FCS_OP_INDEX_GET_NEXT:
			{
				if (RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_INDEX_ID, uiIndex)))
				{
					goto Exit;
				}
				break;
			}
		}
	}

	if (RC_BAD( rc = pWire->sendTerminate()))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc: Performs a misc. operation
****************************************************************************/
RCODE fsvOpClassMisc(
	FSV_WIRE *	pWire)
{
	FLMBYTE		ucSerialNum[F_SERIAL_NUM_SIZE];
	RCODE			opRc = FERR_OK;
	RCODE			rc = FERR_OK;

	// Service the request.

	switch (pWire->getOp())
	{
		case FCS_OP_CREATE_SERIAL_NUM:
		{
			if (RC_BAD( opRc = f_createSerialNumber( ucSerialNum)))
			{
				goto OP_EXIT;
			}
			break;
		}

		default:
		{
			opRc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto OP_EXIT;
		}
	}

OP_EXIT:

	// Send the server's response.

	if (RC_BAD( rc = pWire->sendOpcode( FCS_OPCLASS_MISC, pWire->getOp())))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pWire->sendRc( opRc)))
	{
		goto Exit;
	}

	if (RC_OK( opRc))
	{
		if (pWire->getOp() == FCS_OP_CREATE_SERIAL_NUM)
		{
			if (RC_BAD( rc = pWire->sendBinary( WIRE_VALUE_SERIAL_NUM, ucSerialNum,
						  F_SERIAL_NUM_SIZE)))
			{
				goto Exit;
			}
		}
		else
		{
			flmAssert( rc == FERR_NOT_IMPLEMENTED);
		}
	}

	if (RC_BAD( rc = pWire->sendTerminate()))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc:	Configures an iterator based on from, where, select, and config
		clauses provided by the client.
****************************************************************************/
FSTATIC RCODE fsvIteratorParse(
	FSV_WIRE *		pWire,
	F_Pool *			pPool)
{
	RCODE 			rc = FERR_OK;

	// Parse the "from" clause. This contains record source information.

	if (pWire->getIteratorFrom())
	{
		if (RC_BAD( rc = fsvIteratorFromParse( pWire, pPool)))
		{
			goto Exit;
		}
	}

	if (pWire->getIteratorHandle() == HFCURSOR_NULL)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Parse the "where" clause. This contains the criteria.

	if (pWire->getIteratorWhere())
	{
		if (RC_BAD( rc = fsvIteratorWhereParse( pWire, pPool)))
		{
			goto Exit;
		}
	}

	// Parse the "select" clause. This contains customized view
	// information.

	if (pWire->getIteratorSelect())
	{
		if (RC_BAD( rc = fsvIteratorSelectParse( pWire, pPool)))
		{
			goto Exit;
		}
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc: Adds selection criteria to an iterator.
****************************************************************************/
FSTATIC RCODE fsvIteratorWhereParse(
	FSV_WIRE *		pWire,
	F_Pool *			pPool)
{
	HFCURSOR 	hIterator = pWire->getIteratorHandle();
	NODE *		pWhere = pWire->getIteratorWhere();
	NODE *		pCurNode;
	NODE *		pTmpNode;
	void *		pPoolMark;
	FLMUINT		uiTag;
	RCODE			rc = FERR_OK;

	// If no "where" clause, jump to exit.

	if (!pWhere)
	{
		goto Exit;
	}

	// Process each component of the "where" clause.

	pCurNode = GedChild( pWhere);
	while (pCurNode)
	{
		uiTag = GedTagNum( pCurNode);
		switch (uiTag)
		{
			case FCS_ITERATOR_MODE:
			{
				FLMUINT	uiFlags = 0;

				// Set the iterator's mode flags

				if (RC_BAD( rc = GedGetUINT( pCurNode, &uiFlags)))
				{
					goto Exit;
				}

				if (RC_BAD( rc = FlmCursorSetMode( hIterator, uiFlags)))
				{
					goto Exit;
				}
				break;
			}

			// Add an attribute to the criteria.

			case FCS_ITERATOR_ATTRIBUTE:
			{
				FLMUINT	uiAttrId;

				// Get the attribute ID.

				if (RC_BAD( rc = GedGetUINT( pCurNode, &uiAttrId)))
				{
					goto Exit;
				}

				// Add the attribute.

				if (uiTag == FCS_ITERATOR_ATTRIBUTE)
				{
					if (RC_BAD( rc = FlmCursorAddField( hIterator, uiAttrId, 0)))
					{
						goto Exit;
					}
				}
				else
				{

					// Sanity check.

					flmAssert( 0);
				}
				break;
			}

			// Add an attribute path to the criteria.

			case FCS_ITERATOR_ATTRIBUTE_PATH:
			{
				FLMUINT	puiPath[FCS_ITERATOR_MAX_PATH + 1];
				FLMUINT	uiAttrId;
				FLMUINT	uiPathPos = 0;
				FLMUINT	uiStartLevel;

				if ((pTmpNode = GedFind( GED_TREE, pCurNode, 
						FCS_ITERATOR_ATTRIBUTE, 1)) != NULL)
				{

					// Build the attribute path.

					uiStartLevel = GedNodeLevel( pTmpNode);
					while (pTmpNode && GedNodeLevel( pTmpNode) >= uiStartLevel)
					{
						if (GedNodeLevel( pTmpNode) == uiStartLevel &&
							 GedTagNum( pTmpNode) == FCS_ITERATOR_ATTRIBUTE)
						{
							if (RC_BAD( rc = GedGetUINT( pTmpNode, &uiAttrId)))
							{
								goto Exit;
							}

							puiPath[uiPathPos++] = uiAttrId;
							if (uiPathPos > FCS_ITERATOR_MAX_PATH)
							{
								rc = RC_SET( FERR_SYNTAX);
								goto Exit;
							}
						}

						pTmpNode = pTmpNode->next;
					}

					puiPath[uiPathPos] = 0;
				}

				// Add the attribute path.

				if (RC_BAD( rc = FlmCursorAddFieldPath( hIterator, puiPath, 0)))
				{
					goto Exit;
				}
				break;
			}

			// Add a numeric value to the criteria.

			case FCS_ITERATOR_NUMBER_VALUE:
			case FCS_ITERATOR_REC_PTR_VALUE:
			{

				// To save conversion time, cheat to determine if the number
				// is negative.

				FLMBYTE *		pucValue = (FLMBYTE *) GedValPtr( pCurNode);
				FLMBOOL			bNegative = ((*pucValue & 0xF0) == 0xB0) 
														? TRUE 
														: FALSE;

				if (bNegative)
				{
					FLMINT64	i64Value;

					if (uiTag == FCS_ITERATOR_REC_PTR_VALUE)
					{
						rc = RC_SET( FERR_SYNTAX);
						goto Exit;
					}

					if (RC_BAD( rc = GedGetINT64( pCurNode, &i64Value)))
					{
						goto Exit;
					}

					if (RC_BAD( rc = FlmCursorAddValue( hIterator, FLM_INT64_VAL,
								  &i64Value, 0)))
					{
						goto Exit;
					}
				}
				else
				{
					FLMUINT64	ui64Value;
					FLMUINT		uiValue;

					if (RC_BAD( rc = GedGetUINT64( pCurNode, &ui64Value)))
					{
						goto Exit;
					}

					if (uiTag == FCS_ITERATOR_NUMBER_VALUE)
					{
						if (RC_BAD( rc = FlmCursorAddValue( hIterator, FLM_UINT64_VAL,
									  &ui64Value, 0)))
						{
							goto Exit;
						}
					}
					else if (uiTag == FCS_ITERATOR_REC_PTR_VALUE)
					{
						uiValue = (FLMUINT)ui64Value;
						if (RC_BAD( rc = FlmCursorAddValue( hIterator,
									  FLM_REC_PTR_VAL, &uiValue, 0)))
						{
							goto Exit;
						}
					}
					else
					{

						// Sanity check.

						flmAssert( 0);
					}
				}
				break;
			}

			// Add a binary value to the criteria.

			case FCS_ITERATOR_BINARY_VALUE:
			{
				FLMBYTE *	pucValue = (FLMBYTE *) GedValPtr( pCurNode);
				FLMUINT		uiValLen = GedValLen( pCurNode);

				if (GedValType( pCurNode) != FLM_BINARY_TYPE)
				{
					rc = RC_SET( FERR_SYNTAX);
					goto Exit;
				}

				if (RC_BAD( rc = FlmCursorAddValue( hIterator, FLM_BINARY_VAL,
							  pucValue, uiValLen)))
				{
					goto Exit;
				}
				break;
			}

			// Add a UNICODE string value to the criteria.

			case FCS_ITERATOR_UNICODE_VALUE:
			{
				FLMUINT			uiLen;
				FLMUNICODE *	puzBuf;

				// Mark the pool.

				pPoolMark = pPool->poolMark();

				// Determine the length of the string.

				if (RC_BAD( rc = GedGetUNICODE( pCurNode, NULL, &uiLen)))
				{
					goto Exit;
				}

				// Allocate a temporary buffer.

				uiLen += 2;
				if( RC_BAD( rc = pPool->poolAlloc( uiLen, (void **)&puzBuf)))
				{
					goto Exit;
				}

				// Extract the string and add it to the criteria.

				if (RC_BAD( rc = GedGetUNICODE( pCurNode, puzBuf, &uiLen)))
				{
					goto Exit;
				}

				if (RC_BAD( rc = FlmCursorAddValue( hIterator, FLM_UNICODE_VAL,
							  puzBuf, 0)))
				{
					goto Exit;
				}

				pPool->poolReset( pPoolMark);
				break;
			}

			// Add a NATIVE, WP60, or Word String value to the criteria.

			case FCS_ITERATOR_NATIVE_VALUE:
			case FCS_ITERATOR_WP60_VALUE:
			case FCS_ITERATOR_WDSTR_VALUE:
			{
				FLMUINT		uiLen;
				FLMBYTE *	pucBuf;

				// Mark the pool.

				pPoolMark = pPool->poolMark();

				// Determine the length of the string.

				if (uiTag == FCS_ITERATOR_NATIVE_VALUE)
				{
					if (RC_BAD( rc = GedGetNATIVE( pCurNode, NULL, &uiLen)))
					{
						goto Exit;
					}
				}
				else
				{
					rc = RC_SET( FERR_NOT_IMPLEMENTED);
					goto Exit;
				}

				// Allocate a temporary buffer.

				uiLen += 2;
				if( RC_BAD( rc = pPool->poolAlloc( uiLen, (void **)&pucBuf)))
				{
					goto Exit;
				}

				// Extract the string and add it to the criteria.

				if (uiTag == FCS_ITERATOR_NATIVE_VALUE)
				{
					if (RC_BAD( rc = GedGetNATIVE( pCurNode, (char *) pucBuf,
							&uiLen)))
					{
						goto Exit;
					}

					if (RC_BAD( rc = FlmCursorAddValue( hIterator, FLM_STRING_VAL,
								  pucBuf, 0)))
					{
						goto Exit;
					}
				}

				pPool->poolReset( pPoolMark);
				break;
			}

			// Add a native (internal) text value

			case FCS_ITERATOR_FLM_TEXT_VALUE:
			{
				if (RC_BAD( rc = FlmCursorAddValue( hIterator, FLM_TEXT_VAL,
							  GedValPtr( pCurNode), GedValLen( pCurNode))))
				{
					goto Exit;
				}
				break;
			}

			// Add an operator to the criteria.

			case FCS_ITERATOR_OPERATOR:
			{
				FLMUINT	uiOp;
				QTYPES	eTranslatedOp;

				// Get the C/S operator ID.

				if (RC_BAD( rc = GedGetUINT( pCurNode, &uiOp)))
				{
					goto Exit;
				}

				if (!uiOp ||
					 ((uiOp - FCS_ITERATOR_OP_START) >= FCS_ITERATOR_OP_END))
				{
					rc = RC_SET( FERR_SYNTAX);
					goto Exit;
				}

				// Translate the C/S ID to a FLAIM operator ID.

				if (RC_BAD( rc = fcsTranslateQCSToQFlmOp( uiOp, &eTranslatedOp)))
				{
					goto Exit;
				}

				// Add the operator to the criteria.

				if (RC_BAD( rc = FlmCursorAddOp( hIterator, eTranslatedOp)))
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

		pCurNode = GedSibNext( pCurNode);
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc: Adds source information to an iterator.
****************************************************************************/
FSTATIC RCODE fsvIteratorFromParse(
	FSV_WIRE *		pWire,
	F_Pool *			pPool)
{
	HFDB				hDb = HFDB_NULL;
	HFCURSOR 		hIterator = pWire->getIteratorHandle();
	FLMUINT			uiIteratorId = FCS_INVALID_ID;
	NODE *			pFrom = pWire->getIteratorFrom();
	NODE *			pCurNode;
	NODE *			pCSAttrNode;
	NODE *			pTmpNode;
	RCODE				rc = FERR_OK;

	F_UNREFERENCED_PARM( pPool);

	// If no "from" clause, jump to exit.

	if (!pFrom)
	{
		goto Exit;
	}

	// Process each component of the "from" clause.

	if (hIterator == HFCURSOR_NULL)
	{
		FSV_SESN *	pSession;
		FLMUINT		uiContainerId = FLM_DATA_CONTAINER;
		FLMUINT		uiPath[4];

		uiPath[0] = FCS_ITERATOR_FROM;
		uiPath[1] = FCS_ITERATOR_CANDIDATE_SET;
		uiPath[2] = FCS_ITERATOR_RECORD_SOURCE;
		uiPath[3] = 0;
		if ((pCSAttrNode = GedPathFind( GED_TREE, pFrom, uiPath, 1)) == NULL)
		{
			rc = RC_SET( FERR_FAILURE);
			goto Exit;
		}

		// Get the database handle.

		if ((pSession = pWire->getSession()) == NULL)
		{
			rc = RC_SET( FERR_BAD_HDL);
			goto Exit;
		}

		hDb = pSession->GetDatabase();

		// Get the container ID. A default value of FLM_DATA_CONTAINER will
		// be used if a container ID is not found.

		if ((pTmpNode = GedFind( GED_TREE, pCSAttrNode, 
					FCS_ITERATOR_CONTAINER_ID, 1)) != NULL)
		{
			if (RC_BAD( rc = GedGetUINT( pTmpNode, &uiContainerId)))
			{
				goto Exit;
			}
		}

		// Initialize the cursor when we get the source - only one source is
		// allowed.

		if (RC_BAD( rc = pSession->InitializeIterator( &uiIteratorId, hDb,
					  uiContainerId, &hIterator)))
		{
			goto Exit;
		}

		// Set the iterator handle and ID so they will be available for the
		// parser to use.

		pWire->setIteratorId( uiIteratorId);
		pWire->setIteratorHandle( hIterator);
	}

	pCurNode = GedChild( pFrom);
	while (pCurNode)
	{
		switch (GedTagNum( pCurNode))
		{
			case FCS_ITERATOR_CANDIDATE_SET:
			{

				// Process record sources and indexes.

				pCSAttrNode = GedChild( pCurNode);
				while (pCSAttrNode)
				{
					switch (GedTagNum( pCSAttrNode))
					{

						// Define a record source.

						case FCS_ITERATOR_RECORD_SOURCE:
						{

							// Handled above.

							break;
						}

						// Specify a FLAIM index.

						case FCS_ITERATOR_FLAIM_INDEX:
						{
							FLMUINT	uiIndexId;

							// Get the index ID.

							if (RC_BAD( rc = GedGetUINT( pCSAttrNode, &uiIndexId)))
							{
								goto Exit;
							}

							// Add the index.

							if (RC_BAD( rc = FlmCursorConfig( hIterator,
										  FCURSOR_SET_FLM_IX, (void *) uiIndexId, 
										  (void *) 0)))
							{
								goto Exit;
							}
							break;
						}

						// Set the record type.

						case FCS_ITERATOR_RECORD_TYPE:
						{
							FLMUINT	uiRecordType;

							// Get the record type.

							if (RC_BAD( rc = GedGetUINT( pCSAttrNode, &uiRecordType)))
							{
								goto Exit;
							}

							// Add the record type.

							if (RC_BAD( rc = FlmCursorConfig( hIterator,
										  FCURSOR_SET_REC_TYPE, 
										  (void *) uiRecordType,
										  (void *) 0)))
							{
								goto Exit;
							}
							break;
						}

						case FCS_ITERATOR_OK_TO_RETURN_KEYS:
						{
							FLMUINT	uiOkToReturnKeys;

							if (RC_BAD( rc = GedGetUINT( 
								pCSAttrNode, &uiOkToReturnKeys)))
							{
								goto Exit;
							}

							if (RC_BAD( rc = FlmCursorConfig( hIterator,
										  FCURSOR_RETURN_KEYS_OK, 
										  (void *) (FLMUINT)(uiOkToReturnKeys 
																? TRUE 
																: FALSE), NULL)))
							{
								goto Exit;
							}
							break;
						}
					}

					pCSAttrNode = GedSibNext( pCSAttrNode);
				}
				break;
			}

			case FCS_ITERATOR_MODE:
			{
				FLMUINT	uiFlags;

				// Get the mode flags.

				if (RC_BAD( rc = GedGetUINT( pCurNode, &uiFlags)))
				{
					goto Exit;
				}

				if (RC_BAD( rc = FlmCursorSetMode( hIterator, uiFlags)))
				{
					goto Exit;
				}
				break;
			}
		}

		pCurNode = GedSibNext( pCurNode);
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc: Adds a view to an iterator
****************************************************************************/
FSTATIC RCODE fsvIteratorSelectParse(
	FSV_WIRE *	pWire,
	F_Pool *		pPool)
{
	NODE *		pSelect = pWire->getIteratorSelect();
	NODE *		pCurNode;
	NODE *		pView = NULL;
	FLMBOOL		bNullViewNotRec = FALSE;
	RCODE			rc = FERR_OK;

	F_UNREFERENCED_PARM( pPool);

	// If no "select" clause, jump to exit.

	if (!pSelect)
	{
		goto Exit;
	}

	pCurNode = GedChild( pSelect);
	while (pCurNode)
	{
		switch (GedTagNum( pCurNode))
		{
			case FCS_ITERATOR_VIEW_TREE:
			{
				pView = GedChild( pCurNode);
				break;
			}

			case FCS_ITERATOR_NULL_VIEW_NOT_REC:
			{
				bNullViewNotRec = TRUE;
				break;
			}
		}

		pCurNode = GedSibNext( pCurNode);
	}

	// Set the view record, if any (not supported).

	if (GedChild( pCurNode))
	{
		rc = RC_SET( FERR_NOT_IMPLEMENTED);
		goto Exit;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc: Reads blocks from the database
****************************************************************************/
FSTATIC RCODE fsvDbGetBlocks(
	HFDB			hDb,
	FLMUINT		uiAddress,
	FLMUINT		uiMinTransId,
	FLMUINT *	puiCount,
	FLMUINT *	puiBlocksExamined,
	FLMUINT *	puiNextBlkAddr,
	FLMUINT		uiFlags,
	F_Pool *		pPool,
	FLMBYTE **	ppBlocks,
	FLMUINT *	puiBytes)
{
	FDB *			pDb = (FDB *) hDb;
	FLMBOOL		bDbInitialized = FALSE;
	FLMBOOL		bTransStarted = FALSE;
	FLMUINT		uiLoop;
	FLMUINT		uiCount = *puiCount;
	SCACHE *		pSCache = NULL;
	FLMUINT		uiBlockSize;
	FLMUINT		uiMaxFileSize;
	RCODE			rc = FERR_OK;

	*ppBlocks = NULL;
	*puiCount = 0;
	*puiBlocksExamined = 0;
	*puiBytes = 0;

	if (IsInCSMode( hDb))
	{
		fdbInitCS( pDb);
		bDbInitialized = TRUE;

		CS_CONTEXT *		pCSContext = pDb->pCSContext;
		FCL_WIRE				Wire(pCSContext, pDb);

		if (!pCSContext->bConnectionGood)
		{
			rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
			goto Transmission_Error;
		}

		if (RC_BAD( rc = Wire.sendOp( FCS_OPCLASS_DATABASE,
					  FCS_OP_DATABASE_GET_BLOCK)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_ADDRESS, uiAddress)))
		{
			goto Transmission_Error;
		}

		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_TRANSACTION_ID, uiMinTransId)))
		{
			goto Transmission_Error;
		}

		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_COUNT, uiCount)))
		{
			goto Transmission_Error;
		}

		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_FLAGS, uiFlags)))
		{
			goto Transmission_Error;
		}

		if (RC_BAD( rc = Wire.sendTerminate()))
		{
			goto Transmission_Error;
		}

		// Read the response.

		if (RC_BAD( rc = Wire.read()))
		{
			goto Transmission_Error;
		}

		if (RC_BAD( rc = Wire.getRCode()))
		{
			if (rc != FERR_IO_END_OF_FILE)
			{
				goto Exit;
			}
		}

		*puiBlocksExamined = (FLMUINT) Wire.getNumber2();
		*puiCount = (FLMUINT) Wire.getCount();
		*puiNextBlkAddr = Wire.getAddress();

		if (*puiCount)
		{
			*puiBytes = Wire.getBlockSize();
			if( RC_BAD( rc = pPool->poolAlloc( *puiBytes, (void **)ppBlocks)))
			{
				goto Exit;
			}

			f_memcpy( *ppBlocks, Wire.getBlock(), *puiBytes);
		}

		goto Exit;

Transmission_Error:

		pCSContext->bConnectionGood = FALSE;
		goto Exit;
	}

	if (!uiCount)
	{
		uiCount = 1;
	}

	uiBlockSize = pDb->pFile->FileHdr.uiBlockSize;
	uiMaxFileSize = pDb->pFile->uiMaxFileSize;
	bDbInitialized = TRUE;
	if (RC_BAD( rc = fdbInit( pDb, FLM_READ_TRANS, FDB_TRANS_GOING_OK, 0,
				  &bTransStarted)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pPool->poolAlloc( uiBlockSize * uiCount,
		(void **)ppBlocks)))
	{
		goto Exit;
	}

	// Read uiCount blocks from the database starting at uiAddress. If none
	// of the blocks meet the min trans ID criteria, we will not return any
	// blocks to the reader.

	*puiNextBlkAddr = BT_END;
	for (uiLoop = 0; uiLoop < uiCount; uiLoop++)
	{
		if (!FSAddrIsBelow( FSBlkAddress( FSGetFileNumber( uiAddress),
								 FSGetFileOffset( uiAddress)), pDb->LogHdr.uiLogicalEOF))
		{
			rc = RC_SET( FERR_IO_END_OF_FILE);
			goto Exit;
		}

		if (RC_BAD( rc = ScaGetBlock( pDb, NULL, BHT_FREE, uiAddress, NULL,
					  &pSCache)))
		{
			goto Exit;
		}

		if (FB2UD( &pSCache->pucBlk[BH_TRANS_ID]) >= uiMinTransId)
		{
			f_memcpy( (*ppBlocks + ((*puiCount) * uiBlockSize)), pSCache->pucBlk,
						uiBlockSize);
			(*puiCount)++;
			(*puiBytes) += uiBlockSize;
		}
		(*puiBlocksExamined)++;

		ScaReleaseCache( pSCache, FALSE);
		pSCache = NULL;

		uiAddress += uiBlockSize;
		if (FSGetFileOffset( uiAddress) >= uiMaxFileSize)
		{
			uiAddress = FSBlkAddress( FSGetFileNumber( uiAddress) + 1, 0);
		}

		*puiNextBlkAddr = uiAddress;
	}

Exit:

	if (pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	if (bTransStarted)
	{
		RCODE rc2 = flmAbortDbTrans( pDb);
		if (RC_OK( rc))
		{
			rc = rc2;
		}
	}

	if (bDbInitialized)
	{
		fdbExit( pDb);
	}

	return (rc);
}

/****************************************************************************
Desc: Commits a database transaction and updates the log header
****************************************************************************/
RCODE fsvDbTransCommitEx(
	HFDB			hDb,
	FSV_WIRE *	pWire)
{
	RCODE				rc = FERR_OK;
	FDB *				pDb = (FDB *) hDb;
	FLMBOOL			bIgnore;
	FLMBOOL			bForceCheckpoint = FALSE;
	FLMBYTE *		pucHeader = NULL;

	if (pWire->getFlags() & FCS_TRANS_FORCE_CHECKPOINT)
	{
		bForceCheckpoint = TRUE;
	}

	pucHeader = pWire->getBlock();

	if (IsInCSMode( hDb))
	{
		fdbInitCS( pDb);

		FCL_WIRE Wire(pDb->pCSContext, pDb);

		if (!pDb->pCSContext->bConnectionGood)
		{
			rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
		}
		else
		{
			rc = Wire.doTransOp( FCS_OP_TRANSACTION_COMMIT_EX, 0, 0, 0, pucHeader,
									  bForceCheckpoint);
		}

		goto Exit;
	}

	if (RC_BAD( rc = fdbInit( pDb, FLM_NO_TRANS, FDB_TRANS_GOING_OK, 0, &bIgnore)))
	{
		goto Exit;
	}

	// If there is an invisible transaction going, it should not be
	// commitable by an application.

	if ((pDb->uiTransType == FLM_NO_TRANS) ||
		 (pDb->uiFlags & FDB_INVISIBLE_TRANS))
	{
		rc = RC_SET( FERR_NO_TRANS_ACTIVE);
		goto Exit;
	}

	// See if we have a transaction going which should be aborted.

	if (RC_BAD( pDb->AbortRc))
	{
		rc = RC_SET( FERR_ABORT_TRANS);
		goto Exit;
	}

	// Fix up the log header. Currently, only fields directly related to a
	// backup operation are updated.

	if (pucHeader)
	{
		FLMBYTE *		pLogHdr = &pucHeader[16];
		FLMBYTE *		pucUncommittedHdr = &pDb->pFile->ucUncommittedLogHdr[0];

		f_memcpy( &pucUncommittedHdr[LOG_LAST_BACKUP_TRANS_ID],
					&pLogHdr[LOG_LAST_BACKUP_TRANS_ID], 4);

		f_memcpy( &pucUncommittedHdr[LOG_BLK_CHG_SINCE_BACKUP],
					&pLogHdr[LOG_BLK_CHG_SINCE_BACKUP], 4);

		f_memcpy( &pucUncommittedHdr[LOG_INC_BACKUP_SEQ_NUM],
					&pLogHdr[LOG_INC_BACKUP_SEQ_NUM], 4);

		f_memcpy( &pucUncommittedHdr[LOG_INC_BACKUP_SERIAL_NUM],
					&pLogHdr[LOG_INC_BACKUP_SERIAL_NUM], F_SERIAL_NUM_SIZE);
	}

	// Commit the transaction

	rc = flmCommitDbTrans( pDb, 0, bForceCheckpoint);

Exit:

	flmExit( FLM_DB_TRANS_COMMIT, pDb, rc);
	return (rc);
}

/****************************************************************************
Desc: Looks up session, database, and iterator handles.
****************************************************************************/
FSTATIC RCODE fsvGetHandles(
	FSV_WIRE *	pWire)
{
	FSV_SCTX *	pServerContext = NULL;
	FSV_SESN *	pSession = NULL;
	HFCURSOR		hIterator = HFCURSOR_NULL;
	RCODE			rc = FERR_OK;

	if (RC_BAD( rc = fsvGetGlobalContext( &pServerContext)))
	{
		goto Exit;
	}

	if (pWire->getSessionId() != FCS_INVALID_ID)
	{
		if (RC_BAD( pServerContext->GetSession( pWire->getSessionId(), &pSession)))
		{
			rc = RC_SET( FERR_BAD_HDL);
			goto Exit;
		}

		if (pSession->getCookie() != pWire->getSessionCookie())
		{
			rc = RC_SET( FERR_BAD_HDL);
			goto Exit;
		}

		pWire->setSession( pSession);
	}

	if (pSession)
	{
		pWire->setFDB( (FDB *) pSession->GetDatabase());
		if (pWire->getIteratorId() != FCS_INVALID_ID)
		{
			if (RC_BAD( rc = pSession->GetIterator( pWire->getIteratorId(),
						  &hIterator)))
			{
				goto Exit;
			}

			pWire->setIteratorHandle( hIterator);
		}
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE fsvPostStreamedRequest(
	FSV_SESN *	pSession,
	FLMBYTE *		pucPacket,
	FLMUINT			uiPacketSize,
	FLMBOOL			bLastPacket,
	FCS_BIOS *		pSessionResponse)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bReleaseSession = FALSE;
	F_Pool	localPool;

	localPool.poolInit( 1024);

	if (!pSession && !bLastPacket)
	{

		// If this is a session open request, the request must be contained
		// in a single packet.

		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	if (!pSession)
	{
		FCS_BIOS biosInput;
		FCS_DIS	dataIStream;
		FCS_DOS	dataOStream;

		if (RC_BAD( rc = dataIStream.setup( &biosInput)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = dataOStream.setup( pSessionResponse)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = biosInput.write( pucPacket, uiPacketSize)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = fsvProcessRequest( &dataIStream, &dataOStream,
					  &localPool, NULL)))
		{
			goto Exit;
		}
	}
	else
	{
		FCS_BIOS *	pServerBIStream;
		FCS_BIOS *	pServerBOStream;

		// Need to add a reference to the session object so that if the
		// request closes the session, the response stream will not be
		// destructed until the response has been returned to the client.

		pSession->AddRef();
		bReleaseSession = TRUE;

		if (RC_BAD( rc = pSession->GetBIStream( &pServerBIStream)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = pSession->GetBOStream( &pServerBOStream)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = pServerBIStream->write( pucPacket, uiPacketSize)))
		{
			goto Exit;
		}

		if (bLastPacket)
		{
			FCS_DIS	dataIStream;
			FCS_DOS	dataOStream;

			if (RC_BAD( rc = dataIStream.setup( pServerBIStream)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = dataOStream.setup( pServerBOStream)))
			{
				goto Exit;
			}

			pSession->getWireScratchPool()->poolReset();
			if (RC_BAD( rc = fsvProcessRequest( &dataIStream, &dataOStream,
						  pSession->getWireScratchPool(), NULL)))
			{
				goto Exit;
			}
		}
	}

Exit:

	if (bReleaseSession)
	{
		pSession->Release();
	}

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE fsvGetStreamedResponse(
	FSV_SESN *		pSession,
	FLMBYTE *		pucPacketBuffer,
	FLMUINT			uiMaxPacketSize,
	FLMUINT *		puiPacketSize,
	FLMBOOL *		pbLastPacket)
{
	FCS_BIOS *		pServerBOStream = NULL;
	RCODE				rc = FERR_OK;

	if (RC_BAD( rc = pSession->GetBOStream( &pServerBOStream)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pServerBOStream->read( pucPacketBuffer, uiMaxPacketSize,
				  puiPacketSize)))
	{
		if (rc == FERR_EOF_HIT)
		{
			*pbLastPacket = TRUE;
			rc = FERR_OK;
		}

		goto Exit;
	}

	if (!pServerBOStream->isDataAvailable())
	{
		*pbLastPacket = TRUE;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FSV_SCTX::FSV_SCTX(void)
{
	m_uiSessionToken = 0;
	m_uiCacheSize = FSV_DEFAULT_CACHE_SIZE;
	m_bSetupCalled = FALSE;
	m_paSessions = NULL;
	m_hMutex = F_MUTEX_NULL;
	m_szServerBasePath[0] = '\0';
	m_pLogFunc = NULL;
}

/****************************************************************************
Desc:
****************************************************************************/
FSV_SCTX::~FSV_SCTX(void)
{
	FLMUINT	uiSlot;

	if (m_bSetupCalled)
	{

		// Clean up and free the session table.

		for (uiSlot = 0; uiSlot < m_uiMaxSessions; uiSlot++)
		{
			if (m_paSessions[uiSlot] != NULL)
			{
				m_paSessions[uiSlot]->Release();
			}
		}

		f_free( &m_paSessions);

		// Free the session semaphore.

		(void) f_mutexDestroy( &m_hMutex);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FSV_SCTX::Setup(
	FLMUINT			uiMaxSessions,
	const char *	pszServerBasePath,
	FSV_LOG_FUNC	pLogFunc)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiSlot;

	// Make sure that setup has not been called.

	flmAssert( m_bSetupCalled == FALSE);

	// If zero was passed as the value of uiMaxSessions, use the default.

	if (!uiMaxSessions)
	{
		m_uiMaxSessions = FSV_DEFAULT_MAX_CONNECTIONS;
	}
	else
	{
		m_uiMaxSessions = uiMaxSessions;
	}

	// Initialize the session table.

	if (RC_BAD( rc = f_alloc( sizeof(FSV_SESN *) * m_uiMaxSessions, &m_paSessions
				  )))
	{
		goto Exit;
	}

	for (uiSlot = 0; uiSlot < m_uiMaxSessions; uiSlot++)
	{
		m_paSessions[uiSlot] = NULL;
	}

	// Initialize the context mutex

	if (RC_BAD( rc = f_mutexCreate( &m_hMutex)))
	{
		goto Exit;
	}

	// Set the server's home path.

	if (pszServerBasePath)
	{
		f_strcpy( m_szServerBasePath, pszServerBasePath);
	}
	else
	{
		m_szServerBasePath[0] = '\0';
	}

	// Set the logging function.

	m_pLogFunc = pLogFunc;

	// Set the setup flag.

	m_bSetupCalled = TRUE;

Exit:

	// Clean up any allocations if an error was encountered.

	if (RC_BAD( rc))
	{
		if (m_paSessions != NULL)
		{
			f_free( &m_paSessions);
		}

		if (m_hMutex != F_MUTEX_NULL)
		{
			f_mutexDestroy( &m_hMutex);
		}
	}

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FSV_SCTX::OpenSession(
	FLMUINT		uiVersion,
	FLMUINT		uiFlags,
	FLMUINT *	puiIdRV,
	FSV_SESN **	 ppSessionRV)
{
	FLMUINT		uiSlot;
	FLMUINT		uiCurrTime;
	FLMBOOL		bLocked = FALSE;
	FSV_SESN *	pSession = NULL;
	RCODE			rc = FERR_OK;

	// Make sure that setup has been called.

	flmAssert( m_bSetupCalled == TRUE);

	// Initialize the Id

	*puiIdRV = 0;

	// Create a new session object

	if ((pSession = f_new FSV_SESN) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Allocate the session object

	if (RC_BAD( rc = pSession->Setup( this, uiVersion, uiFlags)))
	{
		goto Exit;
	}

	// Lock the context mutex

	f_mutexLock( m_hMutex);
	bLocked = TRUE;

	// Find an empty slot in the table.

	for (uiSlot = 0; uiSlot < m_uiMaxSessions; uiSlot++)
	{
		if (!m_paSessions[uiSlot])
		{
			break;
		}
	}

	if (uiSlot >= m_uiMaxSessions)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Assign the session to the table slot.

	m_paSessions[uiSlot] = pSession;

	// Increment the session token.

	m_uiSessionToken++;

	// If the session token is 0xFFFF, reset it to 1. Because
	// FSV_INVALID_ID is 0xFFFFFFFF, it is important to reset the session
	// token so that a session will never be assigned an invalid ID.

	if (m_uiSessionToken == 0xFFFF)
	{
		m_uiSessionToken = 1;
	}

	// Set the session's ID.

	*puiIdRV = uiSlot | (m_uiSessionToken << 16);
	pSession->setId( *puiIdRV);

	// Set the session's cookie using the current time.

	f_timeGetSeconds( &uiCurrTime);
	pSession->setCookie( uiCurrTime);

	// Unlock the context mutex

	f_mutexUnlock( m_hMutex);
	bLocked = FALSE;

Exit:

	if (RC_BAD( rc))
	{
		if (pSession)
		{
			pSession->Release();
			pSession = NULL;
		}
	}
	else
	{
		if (ppSessionRV)
		{
			*ppSessionRV = pSession;
		}
	}

	if (bLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FSV_SCTX::CloseSession(
	FLMUINT	uiId)
{
	FLMUINT		uiSlot = (0x0000FFFF & uiId);
	FLMBOOL		bLocked = FALSE;
	FSV_SESN *	pSession = NULL;
	RCODE			rc = FERR_OK;

	// Make sure that setup has been called.

	flmAssert( m_bSetupCalled == TRUE);

	// Lock the context mutex

	f_mutexLock( m_hMutex);
	bLocked = TRUE;

	// Make sure that the slot is valid.

	if (uiSlot >= m_uiMaxSessions)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Get a pointer to the table entry.

	if ((pSession = m_paSessions[uiSlot]) == NULL)
	{

		// Session already closed

		goto Exit;
	}

	// Verify the session ID.

	if (pSession->getId() != uiId)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Free the session.

	pSession->Release();

	// Reset the table entry.

	m_paSessions[uiSlot] = NULL;

Exit:

	if (bLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FSV_SCTX::GetSession(
	FLMUINT		uiId,
	FSV_SESN **	 ppSession)
{
	FLMUINT	uiSlot = (0x0000FFFF & uiId);
	FLMBOOL	bLocked = FALSE;
	RCODE		rc = FERR_OK;

	// Make sure that setup has been called.

	flmAssert( m_bSetupCalled == TRUE);

	// Lock the context mutex

	f_mutexLock( m_hMutex);
	bLocked = TRUE;

	// Make sure that the slot is valid.

	if (uiSlot >= m_uiMaxSessions)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Get a pointer to the entry in the session table.

	if ((*ppSession = m_paSessions[uiSlot]) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Verify the session ID.

	if ((*ppSession)->getId() != uiId)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

Exit:

	if (bLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FSV_SCTX::SetBasePath(
	const char *		pszServerBasePath)
{

	// Make sure that setup has been called.

	flmAssert( m_bSetupCalled == TRUE);

	// Lock the context mutex

	f_mutexLock( m_hMutex);

	// Set the server's base path.

	if (pszServerBasePath)
	{
		f_strcpy( m_szServerBasePath, pszServerBasePath);
	}
	else
	{
		m_szServerBasePath[0] = '\0';
	}

	// Unlock the context mutex

	f_mutexUnlock( m_hMutex);
	return (FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FSV_SCTX::GetBasePath(
	char *		pszServerBasePath)
{

	// Make sure that setup has been called.

	flmAssert( m_bSetupCalled == TRUE);

	// Lock the context mutex

	f_mutexLock( m_hMutex);

	// Copy the base path.

	f_strcpy( pszServerBasePath, m_szServerBasePath);

	// Unlock the context mutex

	f_mutexUnlock( m_hMutex);
	return (FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FSV_SCTX::BuildFilePath(
	const FLMUNICODE *		puzUrlString,
	char *						pszFilePathRV)
{
	RCODE				rc = FERR_OK;
	char				szBasePath[F_PATH_MAX_SIZE];
	FUrl				Url;
	char *			pucAsciiUrl;
	const char *	pszFile;
	F_Pool			tmpPool;

	// Initialize a temporary pool.

	tmpPool.poolInit( 256);

	// Attempt to convert the UNICODE URL to a native string

	if (RC_BAD( rc = fcsConvertUnicodeToNative( &tmpPool, puzUrlString,
				  &pucAsciiUrl)))
	{
		goto Exit;
	}

	// Parse the URL.

	if (RC_BAD( rc = Url.SetUrl( pucAsciiUrl)))
	{
		goto Exit;
	}

	pszFile = Url.GetFile();

	if (Url.GetRelative())
	{

		// Get the server's base path.

		GetBasePath( szBasePath);

		// Build the database path.

		f_strcpy( pszFilePathRV, szBasePath);
		if (RC_BAD( rc = gv_FlmSysData.pFileSystem->pathAppend( 
			pszFilePathRV, pszFile)))
		{
			goto Exit;
		}
	}
	else
	{

		// Absolute path. Use the path "as-is."

		f_strcpy( pszFilePathRV, pszFile);
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FSV_SCTX::SetTempDir(
	const char *		pszTempDir)
{
	RCODE rc = FERR_OK;

	// Make sure that setup has been called.

	flmAssert( m_bSetupCalled == TRUE);

	// Set the temporary directory. There is no need to lock the context
	// semaphore because the state of the context is not being changed.

	if (RC_BAD( rc = FlmConfig( FLM_TMPDIR, (void *) pszTempDir, 0)))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void FSV_SCTX::LogMessage(
	FSV_SESN *		pSession,
	const char *	pucMsg,
	RCODE				rc,
	FLMUINT			uiMsgSeverity)
{
	if (m_pLogFunc)
	{
		f_mutexLock( m_hMutex);
		m_pLogFunc( pucMsg, rc, uiMsgSeverity, (void *) pSession);
		f_mutexUnlock( m_hMutex);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
FSV_SESN::FSV_SESN(void)
{
	m_pServerContext = NULL;
	m_hDb = HFDB_NULL;
	m_uiSessionId = FCS_INVALID_ID;
	m_uiCookie = 0;
	m_uiFlags = 0;
	m_pBIStream = NULL;
	m_pBOStream = NULL;
	m_bSetupCalled = FALSE;
	m_uiClientProtocolVersion = 0;
	m_wireScratchPool.poolInit( 2048);
}

/****************************************************************************
Desc:
****************************************************************************/
FSV_SESN::~FSV_SESN(void)
{
	FLMUINT	uiLoop;

	if (m_bSetupCalled)
	{

		// Free iterator resources.

		for (uiLoop = 0; uiLoop < MAX_SESN_ITERATORS; uiLoop++)
		{
			if (m_IteratorList[uiLoop] != HFCURSOR_NULL)
			{
				(void) FlmCursorFree( &m_IteratorList[uiLoop]);
			}
		}

		// Close the database

		if (m_hDb != HFDB_NULL)
		{
			(void) FlmDbClose( &m_hDb);
		}

		// Free the buffer streams

		if (m_pBIStream)
		{
			m_pBIStream->Release();
		}

		if (m_pBOStream)
		{
			m_pBOStream->Release();
		}
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FSV_SESN::Setup(
	FSV_SCTX *	pServerContext,
	FLMUINT		uiVersion,
	FLMUINT		uiFlags)
{
	FLMUINT	uiLoop;
	RCODE		rc = FERR_OK;

	// Make sure that setup has not been called.

	flmAssert( m_bSetupCalled == FALSE);

	// Verify that the requested version is supported.

	if (uiVersion > FCS_VERSION_1_1_1)
	{
		rc = RC_SET( FERR_UNSUPPORTED_VERSION);
		goto Exit;
	}

	m_uiClientProtocolVersion = uiVersion;

	// Set the server context.

	m_pServerContext = pServerContext;

	// Initialize the iterator list

	for (uiLoop = 0; uiLoop < MAX_SESN_ITERATORS; uiLoop++)
	{
		m_IteratorList[uiLoop] = HFCURSOR_NULL;
	}

	m_bSetupCalled = TRUE;
	m_uiFlags = uiFlags;

Exit:

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FSV_SESN::OpenDatabase(
	FLMUNICODE *		puzDbPath,
	FLMUNICODE *		puzDataDir,
	FLMUNICODE *		puzRflDir,
	FLMUINT				uiOpenFlags)
{
	RCODE			rc = FERR_OK;
	char *		pszDbPath = NULL;
	char *		pszDataDir;
	char *		pszRflDir;

	// Make sure that setup has been called.

	flmAssert( m_bSetupCalled == TRUE);
	flmAssert( m_hDb == HFDB_NULL);

	if (RC_BAD( rc = f_alloc( F_PATH_MAX_SIZE * 3, &pszDbPath)))
	{
		goto Exit;
	}

	pszDataDir = pszDbPath + F_PATH_MAX_SIZE;
	pszRflDir = pszDataDir + F_PATH_MAX_SIZE;

	// Perform some sanity checking.

	if (!puzDbPath)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Convert the UNICODE URL to a server path.

	if (RC_BAD( rc = m_pServerContext->BuildFilePath( puzDbPath, pszDbPath)))
	{
		goto Exit;
	}

	// Convert the data directory

	if (puzDataDir)
	{
		if (RC_BAD( rc = m_pServerContext->BuildFilePath( puzDataDir, pszDataDir)))
		{
			goto Exit;
		}
	}
	else
	{
		pszDataDir = NULL;
	}

	// Convert the RFL path

	if (puzRflDir)
	{
		if (RC_BAD( rc = m_pServerContext->BuildFilePath( puzRflDir, pszRflDir)))
		{
			goto Exit;
		}
	}
	else
	{
		*pszRflDir = 0;
	}

	// Open the database.

	if (RC_BAD( rc = FlmDbOpen( pszDbPath, pszDataDir, pszRflDir, uiOpenFlags,
				  NULL, &m_hDb)))
	{
		goto Exit;
	}

Exit:

	if (pszDbPath)
	{
		f_free( &pszDbPath);
	}

	// Free resources

	if (RC_BAD( rc))
	{
		if (m_hDb != HFDB_NULL)
		{
			(void) FlmDbClose( &m_hDb);
		}
	}

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FSV_SESN::CreateDatabase(
	FLMUNICODE *		puzDbPath,
	FLMUNICODE *		puzDataDir,
	FLMUNICODE *		puzRflDir,
	FLMUNICODE *		puzDictPath,
	FLMUNICODE *		puzDictBuf,
	CREATE_OPTS *		pCreateOpts)
{
	RCODE					rc = FERR_OK;
	F_Pool				tmpPool;
	char *				pucDictBuf = NULL;
	char *				pszDbPath = NULL;
	char *				pszDataDir;
	char *				pszRflDir;
	char *				pszDictPath;

	// Initialize a temporary pool.

	tmpPool.poolInit( 1024);

	// Make sure that setup has been called.

	flmAssert( m_bSetupCalled == TRUE);
	flmAssert( m_hDb == HFDB_NULL);

	if (RC_BAD( rc = f_alloc( F_PATH_MAX_SIZE * 4, &pszDbPath)))
	{
		goto Exit;
	}

	pszDataDir = pszDbPath + F_PATH_MAX_SIZE;
	pszRflDir = pszDataDir + F_PATH_MAX_SIZE;
	pszDictPath = pszRflDir + F_PATH_MAX_SIZE;

	// Perform some sanity checking.

	if (!puzDbPath)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Convert the DB URL to a server path.

	if (RC_BAD( rc = m_pServerContext->BuildFilePath( puzDbPath, pszDbPath)))
	{
		goto Exit;
	}

	// Convert the dictionary URL to a server path.

	if (puzDictPath)
	{
		if (RC_BAD( rc = m_pServerContext->BuildFilePath( puzDictPath, pszDictPath
					  )))
		{
			goto Exit;
		}
	}
	else
	{
		pszDictPath = NULL;
	}

	// Convert the data directory

	if (puzDataDir)
	{
		if (RC_BAD( rc = m_pServerContext->BuildFilePath( puzDataDir, pszDataDir)))
		{
			goto Exit;
		}
	}
	else
	{
		pszDataDir = NULL;
	}

	// Convert the RFL path

	if (puzRflDir)
	{
		if (RC_BAD( rc = m_pServerContext->BuildFilePath( puzRflDir, pszRflDir)))
		{
			goto Exit;
		}
	}
	else
	{
		*pszRflDir = 0;
	}

	// Attempt to convert the UNICODE dictionary buffer to a native string

	if (puzDictBuf)
	{
		if (RC_BAD( rc = fcsConvertUnicodeToNative( &tmpPool, puzDictBuf,
					  &pucDictBuf)))
		{
			goto Exit;
		}
	}

	// Create the database.

	if (RC_BAD( rc = FlmDbCreate( pszDbPath, pszDataDir, pszRflDir, pszDictPath,
				  pucDictBuf, pCreateOpts, &m_hDb)))
	{
		goto Exit;
	}

Exit:

	if (pszDbPath)
	{
		f_free( &pszDbPath);
	}

	// Free resources

	if (RC_BAD( rc))
	{
		if (m_hDb != HFDB_NULL)
		{
			(void) FlmDbClose( &m_hDb);
		}
	}

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FSV_SESN::CloseDatabase(void)
{
	RCODE rc = FERR_OK;

	// Make sure that setup has been called.

	flmAssert( m_bSetupCalled == TRUE);

	// Close the database.

	if (m_hDb != HFDB_NULL)
	{
		if (RC_BAD( rc = FlmDbClose( &m_hDb)))
		{
			goto Exit;
		}
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FSV_SESN::InitializeIterator(
	FLMUINT *	puiIteratorIdRV,
	HFDB			hDb,
	FLMUINT		uiContainer,
	HFCURSOR *	phIteratorRV)
{
	HFCURSOR hIterator = HFCURSOR_NULL;
	FLMUINT	uiSlot;
	RCODE		rc = FERR_OK;

	// Make sure that setup has been called.

	flmAssert( m_bSetupCalled == TRUE);

	// Set the iterator id.

	*puiIteratorIdRV = FCS_INVALID_ID;

	// Find a slot in the session's iterator table

	for (uiSlot = 0; uiSlot < MAX_SESN_ITERATORS; uiSlot++)
	{
		if (m_IteratorList[uiSlot] == HFCURSOR_NULL)
		{
			break;
		}
	}

	// Too many open iterators

	if (uiSlot == MAX_SESN_ITERATORS)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Initialize a new iterator (cursor).

	if (RC_BAD( rc = FlmCursorInit( hDb, uiContainer, &hIterator)))
	{
		goto Exit;
	}

	// Add the iterator to the iterator list.

	m_IteratorList[uiSlot] = hIterator;
	*puiIteratorIdRV = uiSlot;

Exit:

	// Free resources

	if (RC_BAD( rc))
	{
		if (hIterator != HFCURSOR_NULL)
		{
			(void) FlmCursorFree( &hIterator);
		}
	}
	else
	{
		if (phIteratorRV)
		{
			*phIteratorRV = hIterator;
		}
	}

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FSV_SESN::FreeIterator(
	FLMUINT	uiIteratorId)
{
	HFCURSOR hIterator = HFCURSOR_NULL;
	RCODE		rc = FERR_OK;

	// Make sure that setup has been called.

	flmAssert( m_bSetupCalled == TRUE);

	// Find the iterator in the resource bag and remove it.

	if (uiIteratorId >= MAX_SESN_ITERATORS ||
		 m_IteratorList[uiIteratorId] == HFCURSOR_NULL)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	hIterator = m_IteratorList[uiIteratorId];
	m_IteratorList[uiIteratorId] = HFCURSOR_NULL;

	// Free the iterator.

	if (RC_BAD( rc = FlmCursorFree( &hIterator)))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FSV_SESN::GetIterator(
	FLMUINT		uiIteratorId,
	HFCURSOR *	phIteratorRV)
{
	RCODE rc = FERR_OK;

	// Make sure that setup has been called.

	flmAssert( m_bSetupCalled == TRUE);

	if (uiIteratorId >= MAX_SESN_ITERATORS ||
		 m_IteratorList[uiIteratorId] == HFCURSOR_NULL)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	*phIteratorRV = m_IteratorList[uiIteratorId];

Exit:

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FSV_SESN::GetBIStream(
	FCS_BIOS **	 ppBIStream)
{
	RCODE rc = FERR_OK;

	*ppBIStream = NULL;

	if (!m_pBIStream)
	{
		m_pBIStream = f_new FCS_BIOS;
		if (!m_pBIStream)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
	}

	*ppBIStream = m_pBIStream;

Exit:

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FSV_SESN::GetBOStream(
	FCS_BIOS **	 ppBOStream)
{
	RCODE rc = FERR_OK;

	*ppBOStream = NULL;

	if (!m_pBOStream)
	{
		m_pBOStream = f_new FCS_BIOS;
		if (!m_pBOStream)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
	}

	*ppBOStream = m_pBOStream;

Exit:

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE flmStreamEventDispatcher(
	FCS_BIOS	*		pStream,
	FLMUINT			uiEvent,
	void *			UserData)
{
	CS_CONTEXT *	pCSContext = (CS_CONTEXT *) UserData;
	FLMUINT			uiStreamHandlerId = FSEV_HANDLER_UNKNOWN;
	RCODE				rc = FERR_OK;

	// Determine the handler

	if (pCSContext->uiStreamHandlerId == FSEV_HANDLER_UNKNOWN)
	{
		if (f_stricmp( pCSContext->pucAddr, "DS") == 0)
		{
			uiStreamHandlerId = FSEV_HANDLER_DS;
		}
		else if (f_stricmp( pCSContext->pucAddr, "LOOPBACK") == 0)
		{
			uiStreamHandlerId = FSEV_HANDLER_LOOPBACK;
		}

		pCSContext->uiStreamHandlerId = uiStreamHandlerId;
	}
	else
	{
		uiStreamHandlerId = pCSContext->uiStreamHandlerId;
	}

	// Invoke the handler

	switch (uiStreamHandlerId)
	{
		case FSEV_HANDLER_LOOPBACK:
		{
			if (RC_BAD( rc = fsvStreamLoopback( pStream, uiEvent, UserData)))
			{
				goto Exit;
			}
			break;
		}

		default:
		{
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
		}
	}

	// Release CPU to prevent CPU hog

	f_yieldCPU();

Exit:

	if (RC_BAD( rc))
	{

		// Clear the saved handler ID in case a new handler is tried

		pCSContext->uiStreamHandlerId = FSEV_HANDLER_UNKNOWN;
	}

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE fsvStreamLoopback(
	FCS_BIOS	*		pStream,
	FLMUINT			uiEvent,
	void *			UserData)
{
	CS_CONTEXT *	pCSContext = (CS_CONTEXT *) UserData;
	FCS_DIS			dataIStream;
	FCS_DOS			dataOStream;
	RCODE				rc = FERR_OK;

	F_UNREFERENCED_PARM( pStream);

	if (uiEvent == FCS_BIOS_EOM_EVENT)
	{
		if (RC_BAD( rc = dataIStream.setup( (FCS_BIOS *) (pCSContext->pOStream))))
		{
			goto Exit;
		}

		if (RC_BAD( rc = dataOStream.setup( (FCS_BIOS *) (pCSContext->pIStream))))
		{
			goto Exit;
		}

		if (RC_BAD( rc = fsvProcessRequest( &dataIStream, &dataOStream,
					  &(pCSContext->pool), NULL)))
		{
			goto Exit;
		}
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void FSV_WIRE::reset(void)
{
	resetCommon();
	m_uiOpSeqNum = 0;
	m_uiClientVersion = 0;
	m_uiAutoTrans = 0;
	m_uiMaxLockWait = 0;
	m_puzDictPath = NULL;
	m_puzDictBuf = NULL;
	m_puzFileName = NULL;
	m_pucPassword = NULL;
	m_pDrnList = NULL;
	m_uiAreaId = 0;
	m_pIteratorSelect = NULL;
	m_pIteratorFrom = NULL;
	m_pIteratorWhere = NULL;
	m_pIteratorConfig = NULL;
	m_pSession = NULL;
	m_hIterator = HFCURSOR_NULL;
	m_uiType = 0;
	m_bSendGedcom = FALSE;
}

/****************************************************************************
Desc:
****************************************************************************/
void FSV_WIRE::setSession(
	FSV_SESN *	pSession)
{
	m_pSession = pSession;

	// See if GEDCOM is supported by the client

	if (m_pSession && (m_pSession->getFlags() & FCS_SESSION_GEDCOM_SUPPORT))
	{
		m_bSendGedcom = TRUE;
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FSV_WIRE::read(void)
{
	FLMUINT	uiTag;
	FLMUINT	uiCount = 0;
	FLMBOOL	bDone = FALSE;
	RCODE		rc = FERR_OK;

	// Read the opcode.

	if (RC_BAD( rc = readOpcode()))
	{
		goto Exit;
	}

	// Read the request / response values.

	for (;;)
	{
		if (RC_BAD( rc = readCommon( &uiTag, &bDone)))
		{
			goto Exit;
		}

		if (bDone)
		{
			goto Exit;
		}

		// uiTag will be non-zero if readCommon did not understand it.

		uiCount++;
		if (uiTag)
		{
			switch ((uiTag & WIRE_VALUE_TAG_MASK))
			{
				case WIRE_VALUE_OP_SEQ_NUM:
				{
					if (RC_BAD( rc = readNumber( uiTag, &m_uiOpSeqNum, NULL)))
					{
						goto Exit;
					}
					break;
				}

				case WIRE_VALUE_CLIENT_VERSION:
				{
					if (RC_BAD( rc = readNumber( uiTag, &m_uiClientVersion, NULL)))
					{
						goto Exit;
					}
					break;
				}

				case WIRE_VALUE_DICT_FILE_PATH:
				{
					if (RC_BAD( rc = m_pDIStream->readUTF( m_pPool, 
						&m_puzDictPath)))
					{
						goto Exit;
					}
					break;
				}

				case WIRE_VALUE_DICT_BUFFER:
				{
					if (RC_BAD( rc = m_pDIStream->readUTF( m_pPool, &m_puzDictBuf)))
					{
						goto Exit;
					}
					break;
				}

				case WIRE_VALUE_PASSWORD:
				{
					if (RC_BAD( rc = m_pDIStream->readBinary( m_pPool,
								  &m_pucPassword, NULL)))
					{
						goto Exit;
					}
					break;
				}

				case WIRE_VALUE_TYPE:
				{
					if (RC_BAD( rc = readNumber( uiTag, &m_uiType, NULL)))
					{
						goto Exit;
					}
					break;
				}

				case WIRE_VALUE_AREA_ID:
				{
					if (RC_BAD( rc = readNumber( uiTag, &m_uiAreaId, NULL)))
					{
						goto Exit;
					}
					break;
				}

				case WIRE_VALUE_FILE_NAME:
				{
					if (RC_BAD( rc = m_pDIStream->readUTF( m_pPool, &m_puzFileName)))
					{
						goto Exit;
					}
					break;
				}

				case WIRE_VALUE_AUTOTRANS:
				{
					if (RC_BAD( rc = readNumber( uiTag, &m_uiAutoTrans, NULL)))
					{
						goto Exit;
					}
					break;
				}

				case WIRE_VALUE_ITERATOR_SELECT:
				{
					if (RC_BAD( rc = m_pDIStream->readHTD( m_pPool, 0, 0,
								  &m_pIteratorSelect, NULL)))
					{
						goto Exit;
					}
					break;
				}

				case WIRE_VALUE_ITERATOR_FROM:
				{
					if (RC_BAD( rc = m_pDIStream->readHTD( m_pPool, 0, 0,
								  &m_pIteratorFrom, NULL)))
					{
						goto Exit;
					}
					break;
				}

				case WIRE_VALUE_ITERATOR_WHERE:
				{
					if (RC_BAD( rc = m_pDIStream->readHTD( m_pPool, 0, 0,
								  &m_pIteratorWhere, NULL)))
					{
						goto Exit;
					}
					break;
				}

				case WIRE_VALUE_ITERATOR_CONFIG:
				{
					if (RC_BAD( rc = m_pDIStream->readHTD( m_pPool, 0, 0,
								  &m_pIteratorConfig, NULL)))
					{
						goto Exit;
					}
					break;
				}

				case WIRE_VALUE_MAX_LOCK_WAIT:
				{
					if (RC_BAD( rc = readNumber( uiTag, &m_uiMaxLockWait, NULL)))
					{
						goto Exit;
					}
					break;
				}

				default:
				{
					if (RC_BAD( rc = skipValue( uiTag)))
					{
						goto Exit;
					}
					break;
				}
			}
		}
	}

Exit:

	return (rc);
}
