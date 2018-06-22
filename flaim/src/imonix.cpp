//-------------------------------------------------------------------------
// Desc:	Class for displaying a index keys in HTML in a web page.
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

#define INDEX_LIST_FORM_NAME	"IndexListForm"

#define REFERENCE_DRN_FIELD	"DRNField"
#define CONTAINER_FIELD			"ContainerField"
#define REFERENCES_PER_ROW		15

#define NO_CONTAINER_NUM		0xFFFF

#define MAX_RECORDS_TO_OUTPUT		100
#define KEY_LIST_INCREASE_SIZE	1024
#define REF_LIST_INCREASE_SIZE	4096

FSTATIC FLMUINT getIndexContainer(
	HFDB		hDb,
	FLMUINT	uiIndex);

FSTATIC void freeIndexListStatus(
	IXLIST_STATUS *	pIxListStatus,
	FLMBOOL				bFreeStructure);

FSTATIC void copyIndexListStatus(
	IXLIST_STATUS *	pDestIxListStatus,
	IXLIST_STATUS *	pSrcIxListStatus,
	FLMBOOL				bTransferKeyList);

FSTATIC RCODE FLMAPI imonDoIndexList(
	IF_Thread *			pThread);

/****************************************************************************
Desc:	Prints the web page for listing index keys.
****************************************************************************/
RCODE F_IndexListPage::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE				rc = FERR_OK;
	char *			pszErrType = NULL;
	RCODE				runRc = FERR_OK;
	F_Session *		pFlmSession = m_pFlmSession;
	HFDB				hDb;
	FLMUINT			uiIndex;
	FLMUINT			uiContainer;
	char				szTmp [32];
	char *			pszTmp;
	char *			pszOperation = NULL;
	F_NameTable *	pNameTable = NULL;
	FLMBOOL			bPerformIndexList = FALSE;
	FLMBOOL			bStopIndexList = FALSE;
	FLMUINT			uiIndexListThreadId;
	IXLIST_STATUS	IndexListStatus;
	char				szDbKey [F_SESSION_DB_KEY_LEN];
	FlmRecord *		pFromKey = NULL;
	FlmRecord *		pUntilKey = NULL;
	FLMBOOL			bHadFromKey = FALSE;
	FLMBOOL			bHadUntilKey = FALSE;

	f_memset( &IndexListStatus, 0, sizeof( IXLIST_STATUS));
	IndexListStatus.bIndexListRunning = FALSE;
	IndexListStatus.bHaveIndexListStatus = FALSE;

	// Acquire a FLAIM session

	if (!pFlmSession)
	{
		rc = RC_SET( m_uiSessionRC);
		goto ReportErrorExit;
	}

	// Get the database handle, if any

	if( RC_BAD( rc = getDatabaseHandleParam( uiNumParams, 
		ppszParams, pFlmSession, &hDb, szDbKey)))
	{
		goto ReportErrorExit;
	}

	if (RC_BAD( rc = pFlmSession->getNameTable( hDb, &pNameTable)))
	{
		goto ReportErrorExit;
	}

	// Get the index, if any - look in the form first - because
	// it can be in both the form and the header.  The one in the form
	// takes precedence over the one in the header.

	szTmp [0] = '\0';
	uiIndex = 0;
	pszTmp = &szTmp [0];
	if (RC_BAD( getFormValueByName( "index",
			&pszTmp, sizeof( szTmp), NULL)))
	{
		if( RC_BAD( ExtractParameter( uiNumParams, ppszParams, 
			"index", sizeof( szTmp), szTmp)))
		{
			szTmp [0] = 0;
		}
	}
	if (szTmp[ 0])
	{
		uiIndex = f_atoud( szTmp);
	}

	// Get the container, if any.

	szTmp [0] = '\0';
	uiContainer = NO_CONTAINER_NUM;
	if( RC_BAD( ExtractParameter( uiNumParams, ppszParams, 
		"container", sizeof( szTmp), szTmp)))
	{
		szTmp [0] = 0;
	}
	if (szTmp[ 0])
	{
		uiContainer = f_atoud( szTmp);
	}

	// Get the from and until keys and drns.

	bHadFromKey = getKey( hDb, uiIndex, &pFromKey, FO_FIRST);
	bHadUntilKey = getKey( hDb, uiIndex, &pUntilKey, FO_LAST);

	// Get the value of the Operation field, if present.

	getFormValueByName( "Operation",
				&pszOperation, 0, NULL);
	if( pszOperation)
	{
		if (f_stricmp( pszOperation, OPERATION_INDEX_LIST) == 0)
		{
			bPerformIndexList = TRUE;
		}
		else if (f_stricmp( pszOperation, OPERATION_STOP) == 0)
		{
			bStopIndexList = TRUE;
		}
	}

	// See if we had an index list running.  Get the index list thread ID
	// if any.

	szTmp [0] = '\0';
	uiIndexListThreadId = 0;
	if (RC_OK( ExtractParameter( uiNumParams, ppszParams, 
		"Running", sizeof( szTmp), szTmp)))
	{
		if (szTmp [0])
		{
			uiIndexListThreadId = f_atoud( szTmp);
			IndexListStatus.bIndexListRunning = TRUE;
		}
	}

	if (bPerformIndexList)
	{
		// Better not have both bIndexListRunning and bPerformIndexList set!

		flmAssert( !IndexListStatus.bIndexListRunning);
		if (bHadFromKey && bHadUntilKey)
		{

			// Run the index list.

			if (RC_BAD( runRc = runIndexList( hDb, uiIndex,
									pFromKey, pUntilKey,
									&uiIndexListThreadId)))
			{
				pszErrType = (char *)"RUNNING INDEX LIST";
			}
			else
			{
				IndexListStatus.bIndexListRunning = TRUE;
			}
		}
	}

	// Stop the index list, if requested, or get the status.

	if (IndexListStatus.bIndexListRunning)
	{

		// getIndexListStatus could change IndexListStatus.bIndexListRunning
		// to FALSE.

		getIndexListStatus( uiIndexListThreadId, bStopIndexList,
			&IndexListStatus);
	}

	// Output the web page.

	if (!IndexListStatus.bIndexListRunning && IndexListStatus.bHaveIndexListStatus)
	{

		// If we have index keys, output a page for viewing them.

		printDocStart( "Index Key Results");
		popupFrame();
	}
	else if (!IndexListStatus.bIndexListRunning)
	{
		printDocStart( "Run Index List");
		if (pszErrType)
		{
			fnPrintf( m_pHRequest,
			"<br><font color=\"Red\">ERROR %04X (%s) %s</font><br><br>\n",
			(unsigned)runRc, FlmErrorString( runRc), pszErrType);
		}
	}
	else
	{
		stdHdr();
		fnPrintf( m_pHRequest, HTML_DOCTYPE);
		fnPrintf( m_pHRequest, "<html>\n"
									  "<head>\n");
		printRecordStyle();
		printStyle();

		// Output html that will cause a refresh to occur.

		fnPrintf( m_pHRequest, 
			"<META http-equiv=\"refresh\" content=\"2; "
			"url=%s/indexlist?Running=%u&dbhandle=%s&index=%u&container=%u\">"
			"<TITLE>Index List</TITLE>\n",
			m_pszURLString,
			(unsigned)uiIndexListThreadId, szDbKey,
			(unsigned)uiIndex, (unsigned)uiContainer);

		fnPrintf( m_pHRequest, "</head>\n"
									  "<body>\n");
	}

	// Output the form for entering the index or keys
	// and index list status, if the index list is running.

	outputIndexListForm( hDb, szDbKey, uiIndex, uiContainer,
		uiIndexListThreadId, pNameTable, &IndexListStatus);

	// End the document

	printDocEnd();

Exit:

	fnEmit();

	if (pszOperation)
	{
		f_free( &pszOperation);
	}

	if (pFromKey)
	{
		pFromKey->Release();
	}

	if (pUntilKey)
	{
		pUntilKey->Release();
	}

	freeIndexListStatus( &IndexListStatus, FALSE);

	return( FERR_OK);

ReportErrorExit:

	printErrorPage( rc);
	goto Exit;
}

/****************************************************************************
Desc:	Get a from or until key from the form.
****************************************************************************/
FLMBOOL F_IndexListPage::getKey(
	HFDB				hDb,
	FLMUINT			uiIndex,
	FlmRecord **	ppKey,
	FLMUINT			uiKeyId)
{
	FDB *					pDb = NULL;
	FLMBOOL				bDummy;
	IXD *					pIxd;
	IFD *					pIfd;
	FLMUINT				uiLoop;
	FLMUINT				uiFieldCounter;
	char					szTmp [32];
	char					szFieldName [64];
	char *				pszTmp;
	FLMUINT				uiContainer;
	FLMUINT				uiRefDrn = 0;
	FLMUINT				uiValueLen;
	FlmRecord *			pKey;
	void *				pvFld;
	FLMBOOL				bHadFormData = FALSE;

	*ppKey = NULL;

	// Lookup the index definition

	pDb = (FDB *)hDb;
	if (RC_BAD( fdbInit( pDb, FLM_NO_TRANS, FDB_TRANS_GOING_OK, 0,
							&bDummy)))
	{
		goto Exit;
	}

	if (RC_BAD( fdictGetIndex(
				pDb->pDict, pDb->pFile->bInLimitedMode,
				uiIndex, NULL, &pIxd, TRUE)))
	{
		goto Exit;
	}

	// See if there is a reference DRN in the form

	pszTmp = &szTmp [0];
	szTmp [0] = 0;
	f_sprintf( (char *)szFieldName, "%s_%u",
		REFERENCE_DRN_FIELD, (unsigned)uiKeyId);
	getFormValueByName( szFieldName, &pszTmp, sizeof( szTmp), NULL);
	if (szTmp [0])
	{
		uiRefDrn = f_atoud( szTmp);
		bHadFormData = TRUE;
	}

	// See if there is a container field in the form

	uiContainer = NO_CONTAINER_NUM;
	pszTmp = &szTmp [0];
	szTmp [0] = 0;
	f_sprintf( (char *)szFieldName, "%s_%u",
		CONTAINER_FIELD, (unsigned)uiKeyId);
	getFormValueByName( szFieldName, &pszTmp, sizeof( szTmp), NULL);
	if (szTmp [0])
	{
		uiContainer = f_atoud( szTmp);
		bHadFormData = TRUE;
	}

	// Allocate a key record

	if( (pKey = f_new FlmRecord) == NULL)
	{
		goto Exit;
	}

	*ppKey = pKey;

	if (uiContainer != NO_CONTAINER_NUM)
	{
		pKey->setContainerID( uiContainer);
	}
	pKey->setID( uiRefDrn);

	if (RC_BAD( pKey->insertLast( 0, FLM_KEY_TAG, FLM_CONTEXT_TYPE, NULL)))
	{
		goto Exit;
	}

	uiFieldCounter = (FLMUINT)((uiKeyId == FO_FIRST)
										? (FLMUINT)0
										: (FLMUINT)FLM_FREE_TAG_NUMS);
	for (uiLoop = 0, pIfd = pIxd->pFirstIfd;
		  uiLoop < pIxd->uiNumFlds;
		  uiLoop++, uiFieldCounter++, pIfd++)
	{
		pszTmp = NULL;
		f_sprintf( (char *)szFieldName, "field%u", (unsigned)uiFieldCounter);
		if (RC_OK( getFormValueByName( szFieldName, &pszTmp, 0, &uiValueLen)))
		{
			fcsDecodeHttpString( pszTmp);
			bHadFormData = TRUE;
		}

		if (RC_OK( flmBuildKeyPaths( pIfd, pIfd->uiFldNum,
							IFD_GET_FIELD_TYPE( pIfd), TRUE,
							pKey, &pvFld)))
		{

			// Put the data from the form into the field in the key.

			if (pszTmp && *pszTmp)
			{
				FLMUNICODE *	puzBuf = NULL;
				FLMBYTE *		pucBuf = NULL;
				FLMUINT			uiBufSize;
				FLMINT			iVal;
				FLMUINT			uiVal;
				FLMUINT			uiLen;

				switch (IFD_GET_FIELD_TYPE( pIfd))
				{
					case FLM_TEXT_TYPE:
						uiBufSize = 0;
						if (RC_OK( tokenGetUnicode( pszTmp, (void **)&puzBuf, &uiLen,
													&uiBufSize)))
						{
							(void)pKey->setUnicode( pvFld, puzBuf);
							f_free( &puzBuf);
						}
						break;
					case FLM_NUMBER_TYPE:

						// If this is a negative value, then we will store it as an INT

						if (*pszTmp == '-')
						{
							iVal = f_atoi( pszTmp);
							(void)pKey->setINT( pvFld, iVal);
						}
						else
						{
							uiVal = f_atoud( pszTmp);
							(void)pKey->setUINT( pvFld, uiVal);
						}
						break;
					case FLM_BINARY_TYPE:
						if (RC_OK( f_alloc( f_strlen( pszTmp) / 2 + 1, &pucBuf)))
						{
							FLMBOOL		bHaveFirstNibble = FALSE;
							FLMBYTE		ucVal = 0;
							FLMUINT		uiNibble;
							char *		pszVal = pszTmp;
							FLMBYTE *	pucTmp = pucBuf;

							while (*pszVal)
							{
								if (*pszVal >= '0' && *pszVal <= '9')
								{
									uiNibble = (FLMUINT)(*pszVal - '0');
								}
								else if (*pszVal >= 'a' && *pszVal <= 'f')
								{
									uiNibble = (FLMUINT)(*pszVal - 'a' + 10);
								}
								else if (*pszVal >= 'A' && *pszVal <= 'F')
								{
									uiNibble = (FLMUINT)(*pszVal - 'A' + 10);
								}
								else
								{
									pszVal++;
									continue;
								}
								if (bHaveFirstNibble)
								{
									ucVal += (FLMBYTE)uiNibble;
									*pucTmp++ = ucVal;
									bHaveFirstNibble = FALSE;
								}
								else
								{
									ucVal = (FLMBYTE)(uiNibble << 4);
									bHaveFirstNibble = TRUE;
								}
								pszVal++;
							}

							// See if we ended on an odd number of nibbles.

							if (bHaveFirstNibble)
							{
								*pucTmp++ = ucVal;
							}
							if (pucTmp > pucBuf)
							{
								(void)pKey->setBinary( pvFld, (void *)pucBuf,
													(FLMUINT)(pucTmp - pucBuf));
							}
							f_free( &pucBuf);
						}
						break;
					case FLM_CONTEXT_TYPE:
						uiVal = f_atoud( pszTmp);
						(void)pKey->setRecPointer( pvFld, uiVal);
						break;
					default:
						break;
				}
			}
		}
		f_free( &pszTmp);
	}

Exit:

	fdbExit( pDb);
	return( bHadFormData);
}

/****************************************************************************
Desc:	Output a from or until key.
****************************************************************************/
void F_IndexListPage::outputKey(
	const char *	pszKeyName,
	HFDB				hDb,
	FLMUINT			uiIndex,
	FLMUINT			uiContainer,
	F_NameTable *	pNameTable,
	FlmRecord *		pKey,
	FLMUINT			uiRefCnt,
	FLMBOOL			bReadOnly,
	FLMUINT			uiKeyId)
{
	RCODE		rc = FERR_OK;
	FDB *		pDb = NULL;
	FLMBOOL	bDummy;
	IXD *		pIxd;
	IFD *		pIfd;
	FLMUINT	uiLoop;
	FLMUINT	uiLoop2;
	FLMBOOL	bHighlight = FALSE;
	char		szName [128];
	void *	pvFld;
	FLMUINT	uiFieldCounter;
	FLMBOOL	bAllocatedKey = FALSE;

	// Get a default key if one is not passed in.

	if (!pKey)
	{
		if (RC_BAD( rc = FlmKeyRetrieve( hDb, uiIndex, 0, pKey, 0, uiKeyId,
									&pKey, NULL)))
		{
			if (rc == FERR_EOF_HIT || rc == FERR_BOF_HIT)
			{
				pKey = NULL;
				rc = FERR_OK;
			}
			else
			{
				goto Exit;
			}
		}
		bAllocatedKey = TRUE;
	}

	// Lookup the index in the dictionary.

	pDb = (FDB *)hDb;
	if (RC_BAD( rc = fdbInit( pDb, FLM_NO_TRANS, FDB_TRANS_GOING_OK, 0,
							&bDummy)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = fdictGetIndex(
				pDb->pDict, pDb->pFile->bInLimitedMode,
				uiIndex, NULL, &pIxd, TRUE)))
	{
		goto Exit;
	}

	if (!uiRefCnt)
	{
		printStartCenter();
	}

	// Output a value for each defined field.

	printTableStart( pszKeyName, 2, 75);

	// Column headers

	printTableRowStart();
	printColumnHeading( "Field Name", JUSTIFY_LEFT, NULL, 1, 1, TRUE, 35);
	printColumnHeading( "Value", JUSTIFY_LEFT, NULL, 1, 1, TRUE, 65);
	printTableRowEnd();

	// Row for Reference (DRN) or reference count

	if (uiRefCnt)
	{
		printTableRowStart( bHighlight = !bHighlight);

		printTableDataStart( TRUE, JUSTIFY_LEFT, 35);
		fnPrintf( m_pHRequest, "Reference Count");
		printTableDataEnd();

		printTableDataStart( TRUE, JUSTIFY_LEFT, 65);
		fnPrintf( m_pHRequest, "<font color=\"0db3ae\">%lu</font>",
				(unsigned long)uiRefCnt);
		printTableDataEnd();
		printTableRowEnd();
	}
	else if (pKey && pKey->getID())
	{
		printTableRowStart( bHighlight = !bHighlight);

		printTableDataStart( TRUE, JUSTIFY_LEFT, 35);
		fnPrintf( m_pHRequest, "Reference (DRN)");
		printTableDataEnd();

		printTableDataStart( TRUE, JUSTIFY_LEFT, 65);
		if (bReadOnly)
		{
			fnPrintf( m_pHRequest, "<font color=\"0db3ae\">%lu</font>",
				(unsigned long)pKey->getID());
		}
		else
		{
			f_sprintf( szName, "%s_%u",
					REFERENCE_DRN_FIELD, (unsigned)uiKeyId);
			fnPrintf( m_pHRequest,
				"<input class=\"fieldclass\" name=\"%s\" type=\"text\" "
				"value=\"%lu\" size=\"20\">", szName,
				(unsigned long)pKey->getID());
		}
		printTableDataEnd();

		printTableRowEnd();
	}

	// Row for Container - if index is on all containers
	// uiContainer == 0

	if (!uiContainer)
	{
		printTableRowStart( bHighlight = !bHighlight);

		printTableDataStart( TRUE, JUSTIFY_LEFT, 35);
		fnPrintf( m_pHRequest, "Container");
		printTableDataEnd();

		printTableDataStart( TRUE, JUSTIFY_LEFT, 65);
		if (bReadOnly)
		{
			fnPrintf( m_pHRequest, "<font color=\"0db3ae\">%lu</font>",
				(unsigned long)pKey->getContainerID());
		}
		else
		{
			f_sprintf( szName, "%s_%u",
					CONTAINER_FIELD, (unsigned)uiKeyId);
			fnPrintf( m_pHRequest,
				"<input class=\"fieldclass\" name=\"%s\" type=\"text\" "
				"value=\"%lu\" size=\"20\">",
				szName, (unsigned long)pKey->getContainerID());
		}
		printTableDataEnd();

		printTableRowEnd();
	}

	uiFieldCounter = (FLMUINT)((uiKeyId == FO_FIRST)
										? (FLMUINT)0
										: (FLMUINT)FLM_FREE_TAG_NUMS);
	for (uiLoop = 0, pIfd = pIxd->pFirstIfd;
		  uiLoop < pIxd->uiNumFlds;
		  uiLoop++, uiFieldCounter++, pIfd++)
	{
		printTableRowStart( bHighlight = !bHighlight);
		printTableDataStart( TRUE, JUSTIFY_LEFT, 35);

		for (uiLoop2 = 0; pIfd->pFieldPathPToC [uiLoop2]; uiLoop2++)
		{
			if (uiLoop2)
			{
				fnPrintf( m_pHRequest, ".");
			}

			// Get the field name.

			if (!pNameTable ||
				 !pNameTable->getFromTagNum(
								pIfd->pFieldPathPToC [uiLoop2], NULL,
								szName,
								sizeof( szName)))
			{
				f_sprintf( szName, "TAG_%u",
					(unsigned)pIfd->pFieldPathPToC [uiLoop2]);
			}
			printEncodedString( szName, HTML_ENCODING);
			fnPrintf( m_pHRequest, "(%u)",
				(unsigned)pIfd->pFieldPathPToC [uiLoop2]);
		}
		printTableDataEnd();

		// Retrieve the field from the key.
//VISIT: Verify and find full path, not just leaf field.

		pvFld = (void *)((pKey)
								? pKey->find( pKey->root(), pIfd->uiFldNum)
								: NULL);

		printTableDataStart( TRUE, JUSTIFY_LEFT, 65);
		if (pvFld && pKey->getDataLength( pvFld))
		{
			switch (IFD_GET_FIELD_TYPE( pIfd))
			{
				case FLM_TEXT_TYPE:
					printTextField(
						pKey, pvFld, uiFieldCounter, bReadOnly);
					break;
				case FLM_NUMBER_TYPE:
					printNumberField(
						pKey, pvFld, uiFieldCounter, bReadOnly);
					break;
				case FLM_BINARY_TYPE:
					printBinaryField(
						pKey, pvFld, uiFieldCounter, bReadOnly);
					break;
				case FLM_CONTEXT_TYPE:
					printContextField(
						pKey, pvFld, uiFieldCounter, bReadOnly);
					break;
				case FLM_BLOB_TYPE:
					printBlobField(
						pKey, pvFld, uiFieldCounter, bReadOnly);
					break;
				default:
					printDefaultField(
						pKey, pvFld, uiFieldCounter, bReadOnly);
					break;
			}
		}
		else if (!bReadOnly)
		{
			fnPrintf( m_pHRequest, "<input class=\"fieldclass\" name=\"field%d\" "
										  "type=\"text\" value=\"\" size=\"20\">",
				(unsigned)uiFieldCounter);
		}
		else
		{
			printSpaces( 1);
		}
		printTableDataEnd();

		printTableRowEnd();
	}

	printTableEnd();
	if (!uiRefCnt)
	{
		printEndCenter( FALSE);
	}

Exit:

	if (pKey && bAllocatedKey)
	{
		pKey->Release();
	}

	if (RC_BAD( rc))
	{
		fnPrintf( m_pHRequest,
			"<br><font color=\"Red\">ERROR %04X (%s) outputting %s</font><br><br>\n",
			(unsigned)rc, FlmErrorString( rc), pszKeyName);
	}
	fdbExit( pDb);
}

/****************************************************************************
Desc:	Get an index's container number.
****************************************************************************/
FSTATIC FLMUINT getIndexContainer(
	HFDB		hDb,
	FLMUINT	uiIndex
	)
{
	FDB *		pDb;
	FLMBOOL	bDummy;
	IXD *		pIxd;
	FLMUINT	uiContainer = NO_CONTAINER_NUM;

	pDb = (FDB *)hDb;
	if (RC_BAD( fdbInit( pDb, FLM_NO_TRANS, FDB_TRANS_GOING_OK, 0,
							&bDummy)))
	{
		goto Exit;
	}

	if (RC_BAD( fdictGetIndex(
				pDb->pDict, pDb->pFile->bInLimitedMode,
				uiIndex, NULL, &pIxd, TRUE)))
	{
		goto Exit;
	}
	uiContainer = pIxd->uiContainerNum;
Exit:
	fdbExit( pDb);
	return( uiContainer);
}

/****************************************************************************
Desc:	Output the form for the user to input index and query keys.
****************************************************************************/
void F_IndexListPage::outputIndexListForm(
	HFDB					hDb,
	const char *		pszDbKey,
	FLMUINT				uiIndex,
	FLMUINT				uiContainer,
	FLMUINT				uiIndexListThreadId,
	F_NameTable *		pNameTable,
	IXLIST_STATUS *	pIndexListStatus)
{
	char *			pszName;
	char				szName [128];
	FLMUINT			uiLoop;
	FLMUINT			uiLoop2;
	FLMUINT			uiRefCnt;
	FLMUINT *		puiRefList;
	FlmRecord *		pKey;

	// Get the container, if we don't have it yet and we have the index.

	if (uiIndex && uiContainer == NO_CONTAINER_NUM)
	{
		uiContainer = getIndexContainer( hDb, uiIndex);
	}

	fnPrintf( m_pHRequest, "<form name=\""
		INDEX_LIST_FORM_NAME "\" type=\"submit\" "
	  "method=\"post\" action=\"%s/indexlist", m_pszURLString);
	if (pIndexListStatus->bIndexListRunning)
	{
		fnPrintf( m_pHRequest, "?Running=%u&",
			(unsigned)uiIndexListThreadId);
	}
	else
	{
		fnPrintf( m_pHRequest, "?");
	}

	fnPrintf( m_pHRequest, "dbhandle=%s&index=%u&container=%u\">\n",
		pszDbKey, (unsigned)uiIndex, (unsigned)uiContainer);

	// Output the setOperation function

	printSetOperationScript();

	// Output the database name

	printStartCenter();
	fnPrintf( m_pHRequest, "Database&nbsp;");
	printEncodedString( ((FDB *)hDb)->pFile->pszDbPath, HTML_ENCODING);
	printEndCenter( FALSE);
	fnPrintf( m_pHRequest, "<br>\n");

	// Output container name and index name if we have selected an
	// index.

	if (!uiIndex)
	{

		// Output pulldown list to select an index

		printStartCenter();
		fnPrintf( m_pHRequest, "Index&#%u;&nbsp;", (unsigned)':');
		printIndexPulldown( pNameTable, uiIndex, FALSE, FALSE, TRUE,
			"onChange='javascript:setOperation( "
			INDEX_LIST_FORM_NAME ", \"" OPERATION_INDEX_LIST "\")'");
		printEndCenter( FALSE);
		fnPrintf( m_pHRequest, "<br>\n");
	}
	else
	{

		// Output the index name

		printStartCenter();
		fnPrintf( m_pHRequest, "Index&#%u;&nbsp;", (unsigned)':');
		switch (uiIndex)
		{
			case FLM_DICT_INDEX:
				pszName = (char *)"Dictionary";
				break;
			default:
				if (!pNameTable ||
					 !pNameTable->getFromTagNum( uiIndex, NULL,
									szName,
									sizeof( szName)))
				{
					f_sprintf( szName, "IX_%u", (unsigned)uiIndex);
				}
				pszName = &szName [0];
				break;
		}
		printEncodedString( pszName, HTML_ENCODING);
		fnPrintf( m_pHRequest, " (%u)", (unsigned)uiIndex);
		printEndCenter( FALSE);
		fnPrintf( m_pHRequest, "<br>\n");

		// Output the index's container, if we were able to
		// get one.

		if (uiContainer != NO_CONTAINER_NUM)
		{
			printStartCenter();
			fnPrintf( m_pHRequest, "Index Container&#%u;&nbsp;",
				(unsigned)':');
			switch (uiContainer)
			{
				case 0:
					pszName = (char *)"All";
				case FLM_DATA_CONTAINER:
					pszName = (char *)"Data";
					break;
				case FLM_DICT_CONTAINER:
					pszName = (char *)"Dictionary";
					break;
				case FLM_TRACKER_CONTAINER:
					pszName = (char *)"Tracker";
					break;
				default:
					if (!pNameTable ||
						 !pNameTable->getFromTagNum( uiContainer, NULL,
										szName,
										sizeof( szName)))
					{
						f_sprintf( szName, "Cont_%u", (unsigned)uiContainer);
					}
					pszName = &szName [0];
					break;
			}
			printEncodedString( pszName, HTML_ENCODING);
			fnPrintf( m_pHRequest, " (%u)", (unsigned)uiContainer);
			printEndCenter( FALSE);
			fnPrintf( m_pHRequest, "<br>\n");
		}

		// Output the from and until keys

		outputKey( "From Key", hDb, uiIndex, uiContainer,
			pNameTable, pIndexListStatus->pFromKey, 0,
			pIndexListStatus->bIndexListRunning, FO_FIRST);
		fnPrintf( m_pHRequest, "<br>\n");
		outputKey( "Until Key", hDb, uiIndex, uiContainer,
			pNameTable, pIndexListStatus->pUntilKey, 0,
			pIndexListStatus->bIndexListRunning, FO_LAST);
		fnPrintf( m_pHRequest, "<br>\n");

		printStartCenter();
		if (!pIndexListStatus->bIndexListRunning)
		{

			// If we are not running an index list, add a Do Index List button

			printOperationButton( INDEX_LIST_FORM_NAME,
				"Do Index List", OPERATION_INDEX_LIST);
		}
		else
		{

			// Output a stop button

			printOperationButton( INDEX_LIST_FORM_NAME,
				"Stop Index List", OPERATION_STOP);
		}
		printEndCenter( TRUE);
	}

	// Close the form

	fnPrintf( m_pHRequest, "</form>\n");

	// Output index list status, if we have any

	if (pIndexListStatus->bHaveIndexListStatus)
	{
		printStartCenter();
		if (pIndexListStatus->bIndexListRunning)
		{
			printTableStart( "INDEX LIST PROGRESS", 2, 50);
		}
		else
		{
			printTableStart( "INDEX LIST RESULTS", 2, 50);
		}

		// Column headers

		printTableRowStart();
		printColumnHeading( "Key Count", JUSTIFY_RIGHT);
		printColumnHeading( "Reference Count", JUSTIFY_RIGHT);
		printTableRowEnd();

		printTableRowStart( TRUE);
		printTableDataStart( TRUE, JUSTIFY_RIGHT);
		fnPrintf( m_pHRequest, "%u", (unsigned)pIndexListStatus->uiKeyCount);
		printTableDataEnd();
		printTableDataStart( TRUE, JUSTIFY_RIGHT);
		fnPrintf( m_pHRequest, "%u", (unsigned)pIndexListStatus->uiRefCount);
		printTableDataEnd();
		printTableRowEnd();
		printTableEnd();
		printEndCenter( FALSE);
		fnPrintf( m_pHRequest, "<br>\n");

		if (!pIndexListStatus->bIndexListRunning && pIndexListStatus->uiKeyCount)
		{
			printTableStart( "Keys RETRIEVED", 1, 100);
			printTableEnd();
			fnPrintf( m_pHRequest, "<br>\n");

			for (uiLoop = 0; uiLoop < pIndexListStatus->uiKeyCount; uiLoop++)
			{
				uiRefCnt = pIndexListStatus->pKeyList [uiLoop].uiRefCnt;
				pKey = pIndexListStatus->pKeyList [uiLoop].pKey;
				f_sprintf( szName, "Key #%u", (unsigned)(uiLoop + 1));
				outputKey( szName, hDb, uiIndex, uiContainer, pNameTable,
					pKey, uiRefCnt, TRUE, 0);

				puiRefList = &pIndexListStatus->puiRefList [
							pIndexListStatus->pKeyList [uiLoop].uiRefStartOffset];
				for (uiLoop2 = 0; uiLoop2 < uiRefCnt; uiLoop2++, puiRefList++)
				{
					if (uiLoop2)
					{
						if (uiLoop2 % REFERENCES_PER_ROW != 0)
						{
							if (fnPrintf( m_pHRequest, ",") != 0)
							{
								goto Exit;
							}
						}
						else
						{
							if (fnPrintf( m_pHRequest, "<br>\n") != 0)
							{
								goto Exit;
							}
						}
					}
					if (fnPrintf( m_pHRequest, "<a href=\"javascript:openPopup"
								"(\'%s/ProcessRecord?dbhandle=%s&ReadOnly=TRUE&"
								"DRN=%u&container=%u&Action=Retrieve\')\">%u</a>\n",
								m_pszURLString, pszDbKey, (unsigned)(*puiRefList),
								(unsigned)pKey->getContainerID(),
								(unsigned)(*puiRefList)) != 0)
					{
						goto Exit;
					}
				}
				if (fnPrintf( m_pHRequest, "<br><br>\n") != 0)
				{
					goto Exit;
				}
			}
		}
	}
Exit:
	return;
}

/****************************************************************************
Desc:	Run an index key list.
****************************************************************************/
RCODE F_IndexListPage::runIndexList(
	HFDB			hDb,
	FLMUINT		uiIndex,
	FlmRecord *	pFromKey,
	FlmRecord *	pUntilKey,
	FLMUINT *	puiIndexListThreadId
	)
{
	RCODE					rc = FERR_OK;
	IXLIST_STATUS *	pIndexListStatus = NULL;
	IF_Thread *			pThread;
	FDB *					pDb = NULL;

	// Open the database for the thread - so it doesn't have
	// to worry about the handle going away.  The thread will close the
	// new handle when it exits.
	
	if (RC_BAD( rc = flmOpenFile( ((FDB *)hDb)->pFile, NULL, NULL, NULL,
							0, TRUE, NULL, NULL,
							(((FDB *)hDb)->pFile)->pszDbPassword, &pDb)))
	{
		goto Exit;
	}

	// Create an object to track the query.

	if (RC_BAD( rc = f_calloc( sizeof( IXLIST_STATUS), &pIndexListStatus)))
	{
		goto Exit;
	}

	// Get the index container

	pIndexListStatus->hDb = (HFDB)pDb;
	pIndexListStatus->uiIndex = uiIndex;
	if (pFromKey)
	{
		if ((pIndexListStatus->pFromKey = pFromKey->copy()) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
	}
	if (pUntilKey)
	{
		if ((pIndexListStatus->pUntilKey = pUntilKey->copy()) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
	}
	pIndexListStatus->bIndexListRunning = TRUE;
	pIndexListStatus->uiLastTimeBrowserChecked = FLM_GET_TIMER();

	// If browser does not check query status at least every 15 seconds, we will
	// assume it has gone away and the thread will terminate itself.

	pIndexListStatus->uiIndexListTimeout = FLM_SECS_TO_TIMER_UNITS( 15);

	// Start a thread to do the query.

	if( RC_BAD( rc = f_threadCreate( &pThread, imonDoIndexList,
							"WEB INDEX LIST",
							gv_uiDbThrdGrp, 1,
							(void *)pIndexListStatus, (void *)hDb)))
	{
		goto Exit;
	}

	*puiIndexListThreadId = pThread->getThreadId();
	
	// Set pIndexListStatus to NULL so it won't be freed below.  The thread
	// will free it when it stops.

	pIndexListStatus = NULL;

	// Set pDb to NULL so it won't be closed below.  The thread will
	// close it when it stops.

	pDb = NULL;

Exit:

	if (pThread)
	{
		pThread->Release();
	}

	if (pIndexListStatus)
	{
		freeIndexListStatus( pIndexListStatus, TRUE);
	}

	if (pDb)
	{
		FlmDbClose( (HFDB *)&pDb);
	}

	return( rc);
}

/****************************************************************************
Desc:	Free an IXLIST_STATUS structure.
****************************************************************************/
FSTATIC void freeIndexListStatus(
	IXLIST_STATUS *	pIxListStatus,
	FLMBOOL				bFreeStructure
	)
{
	FLMUINT	uiLoop;

	// Free the from and until keys.

	if (pIxListStatus->pFromKey)
	{
		pIxListStatus->pFromKey->Release();
	}
	if (pIxListStatus->pUntilKey)
	{
		pIxListStatus->pUntilKey->Release();
	}

	// Free the key list and reference list

	if (pIxListStatus->pKeyList)
	{
		for (uiLoop = 0; uiLoop < pIxListStatus->uiKeyCount; uiLoop++)
		{
			pIxListStatus->pKeyList [uiLoop].pKey->Release();
		}
		f_free( &pIxListStatus->pKeyList);
	}

	if (pIxListStatus->puiRefList)
	{
		f_free( &pIxListStatus->puiRefList);
	}

	// Free the structure

	if (bFreeStructure)
	{
		f_free( &pIxListStatus);
	}
}

/****************************************************************************
Desc:	Copy an IXLIST_STATUS structure's content.
****************************************************************************/
FSTATIC void copyIndexListStatus(
	IXLIST_STATUS *	pDestIxListStatus,
	IXLIST_STATUS *	pSrcIxListStatus,
	FLMBOOL				bTransferKeyList
	)
{
	f_memcpy( pDestIxListStatus, pSrcIxListStatus, sizeof( IXLIST_STATUS));
	pDestIxListStatus->pFromKey = NULL;
	pDestIxListStatus->pUntilKey = NULL;

	// Copy the from and until keys.

	if (pSrcIxListStatus->pFromKey)
	{
		pDestIxListStatus->pFromKey = pSrcIxListStatus->pFromKey->copy();
	}
	if (pSrcIxListStatus->pUntilKey)
	{
		pDestIxListStatus->pUntilKey = pSrcIxListStatus->pUntilKey->copy();
	}

	// The bTransferKeyList, if TRUE, indicates that we are transferring
	// the key list and the reference list from the source to the
	// destination, meaning we no longer want it in the source. 
	// Otherwise, it should remain in the source and we
	// need to NULL it out of the destination.

	if (bTransferKeyList)
	{
		pSrcIxListStatus->pKeyList = NULL;
		pSrcIxListStatus->puiRefList = NULL;
	}
	else
	{
		pDestIxListStatus->pKeyList = NULL;
		pDestIxListStatus->puiRefList = NULL;
	}
}

/****************************************************************************
Desc:	Output the current thread status to the web page.
****************************************************************************/
void F_IndexListPage::getIndexListStatus(
	FLMUINT				uiIndexListThreadId,
	FLMBOOL				bStopIndexList,
	IXLIST_STATUS *	pIndexListStatus)
{
	FLMUINT				uiThreadId;
	IF_Thread *			pThread = NULL;
	IXLIST_STATUS *	pThreadIndexListStatus;
	FLMBOOL				bMutexLocked = FALSE;

	flmAssert( !pIndexListStatus->bHaveIndexListStatus);

	// See if the thread is still running.

	f_mutexLock( gv_FlmSysData.hShareMutex);
	bMutexLocked = TRUE;
	uiThreadId = 0;
	for (;;)
	{
		if (RC_BAD( gv_FlmSysData.pThreadMgr->getNextGroupThread( &pThread,
						gv_uiDbThrdGrp, &uiThreadId)))
		{
			pIndexListStatus->bIndexListRunning = FALSE;
			goto Exit;
		}
		if (uiThreadId == uiIndexListThreadId)
		{

			// If the app ID is zero, the thread is on its way out or already
			// out.  Can no longer get thread status.

			if (!pThread->getThreadAppId())
			{
				pIndexListStatus->bIndexListRunning = FALSE;
				goto Exit;
			}

			// Found thread, get its query data

			pThreadIndexListStatus = (IXLIST_STATUS *)pThread->getParm1();
			pThreadIndexListStatus->uiLastTimeBrowserChecked = FLM_GET_TIMER();

			// Tell the thread to stop the query before telling it
			// to stop.  This is so we can get partial results.

			if (bStopIndexList)
			{
				pThreadIndexListStatus->bStopIndexList = TRUE;

				// Go into a while loop, waiting for the thread
				// to finish its query.

				while (pThreadIndexListStatus->bIndexListRunning)
				{
					f_mutexUnlock( gv_FlmSysData.hShareMutex);
					bMutexLocked = FALSE;
					f_sleep( 200);
					f_mutexLock( gv_FlmSysData.hShareMutex);
					bMutexLocked = TRUE;

					// If the thread app ID goes to zero, it has been
					// told to shut down, and has either already gone
					// away or is in the process of doing so, in which
					// case pThreadIndexListStatus has either already been
					// deleted, or will be - so it is not safe to access
					// it any more!

					if (!pThread->getThreadAppId())
					{
						pIndexListStatus->bIndexListRunning = FALSE;
						goto Exit;
					}
				}
			}

			break;
		}
		pThread->Release();
		pThread = NULL;
	}

	// Mutex better still be locked at this point.

	flmAssert( bMutexLocked);

	// If the query is not done, return everything except the DRN list.
	// Note that we test pThreadIndexListStatus->bIndexListRunning BEFORE
	// doing the memcpy.  This is because puiDrnList is not guaranteed
	// to be set until bIndexListRunning is FALSE.  If bIndexListRunning is TRUE,
	// we will NULL out whatever got copied into puiDrnList.

	if (!pThreadIndexListStatus->bIndexListRunning)
	{

		// Transfer the lists.

		copyIndexListStatus( pIndexListStatus, pThreadIndexListStatus, TRUE);

		// Need to unlock the mutex so that the thread can stop.

		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;
		pThread->stopThread();
	}
	else
	{

		// Don't transfer the lists.

		copyIndexListStatus( pIndexListStatus, pThreadIndexListStatus, FALSE);

		// Set bIndexListRunning to TRUE.  This takes care of a race
		// condition of pThreadIndexListStatus->bIndexListRunning getting
		// set to FALSE by the index list thread after we test it above.
		// we make the test on pThreadIndexListStatus->bIndexListRunning.  We will
		// simply get that fact next time we get status.

		pIndexListStatus->bIndexListRunning = TRUE;
	}
	pIndexListStatus->bHaveIndexListStatus = TRUE;

Exit:
	if (bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;
	}

	if (pThread)
	{
		pThread->Release();
	}
}

/****************************************************************************
Desc:	Thread to perform a query for a web page.
****************************************************************************/
FSTATIC RCODE FLMAPI imonDoIndexList(
	IF_Thread *		pThread)
{
	RCODE					rc;
	IXLIST_STATUS *	pIndexListStatus = (IXLIST_STATUS *)pThread->getParm1();
	HFDB					hDb = pIndexListStatus->hDb;
	FLMUINT				uiIndex = pIndexListStatus->uiIndex;
	FlmRecord *			pSrchKey = NULL;
	FLMUINT				uiSrchDrn;
	FLMUINT				uiSrchFlag = 0;
	FLMUINT				uiCurrTime;
	FLMUINT				uiLastTimeSetStatus = 0;
	FLMUINT				ui20SecsTime;
	FLMBOOL				bTransStarted = FALSE;
	FLMBOOL				bNewKey = FALSE;
	FLMBYTE *			pucUntilKeyBuf = NULL;
	FLMUINT				uiUntilKeyLen;
	FLMUINT				uiUntilDrn;
	FLMBYTE *			pucFoundKeyBuf = NULL;
	FLMUINT				uiFoundKeyLen;
	char *				pszEndStatus = &pIndexListStatus->szEndStatus [0];

	pThread->setThreadStatus( FLM_THREAD_STATUS_RUNNING);
	ui20SecsTime = FLM_SECS_TO_TIMER_UNITS( 20);

	// Start a transaction.

	if (RC_BAD( rc = FlmDbTransBegin( hDb, FLM_READ_TRANS, 0)))
	{
		f_sprintf( pszEndStatus, "Trans Error %04X", (unsigned)rc);
		goto Quit_List;
	}
	bTransStarted = TRUE;

	uiSrchFlag = FO_INCL;
	if (pIndexListStatus->pFromKey)
	{
		uiSrchDrn = pIndexListStatus->pFromKey->getID();
		if ((pSrchKey = pIndexListStatus->pFromKey->copy()) == NULL)
		{
			f_strcpy( pszEndStatus, "Could not copy from key");
			goto Quit_List;
		}
	}

	// Allocate key buffers for the until key and the found key so we
	// can do comparisons.

	if( RC_BAD( rc = f_alloc( MAX_KEY_SIZ * 2, &pucUntilKeyBuf)))
	{
		f_strcpy( pszEndStatus, "Could not allocate key buffers");
		goto Quit_List;
	}

	pucFoundKeyBuf = &pucUntilKeyBuf [MAX_KEY_SIZ];

	// Get the collated until key.

	if (!pIndexListStatus->pUntilKey)
	{
		f_memset( pucUntilKeyBuf, 0xFF, MAX_KEY_SIZ);
		uiUntilKeyLen = MAX_KEY_SIZ;
		uiUntilDrn = 0;
	}
	else
	{
		uiUntilDrn = pIndexListStatus->pUntilKey->getID();
		if (RC_BAD( rc = FlmKeyBuild( hDb, uiIndex,
											pIndexListStatus->pUntilKey->getContainerID(),
											pIndexListStatus->pUntilKey, 0,
											pucUntilKeyBuf, &uiUntilKeyLen)))
		{
			f_sprintf( pszEndStatus, "Until Key Build Error %04X", (unsigned)rc);
			goto Quit_List;
		}
	}

	bNewKey = TRUE;

	for (;;)
	{

		// See if we should shut down. 

		if (pThread->getShutdownFlag())
		{
			pIndexListStatus->bIndexListRunning = FALSE;

			// Transaction will be aborted below

			pThread->setThreadStatus( FLM_THREAD_STATUS_TERMINATING);
			goto Exit;
		}

		// See if the browser quit asking for status.

		uiCurrTime = FLM_GET_TIMER();
		if (FLM_ELAPSED_TIME( uiCurrTime,
				pIndexListStatus->uiLastTimeBrowserChecked) >=
						pIndexListStatus->uiIndexListTimeout)
		{
			if (pIndexListStatus->bIndexListRunning)
			{
				pThread->setThreadStatus( "Timed out, KeyCnt=%u, RefCnt=%u",
					(unsigned)pIndexListStatus->uiKeyCount,
					(unsigned)pIndexListStatus->uiRefCount);
				pIndexListStatus->bIndexListRunning = FALSE;
			}

			// Transaction will be aborted below

			goto Exit;
		}

		// If the query is not running, just pause one second at a time
		// until we are told to shut down or until we time out.

		if (!pIndexListStatus->bIndexListRunning)
		{
			pThread->sleep( 1000);
			continue;
		}

		// See if we should stop the query.

		if (pIndexListStatus->bStopIndexList)
		{
			f_sprintf( pszEndStatus, "User halted, KeyCnt=%u, RefCnt=%u",
					(unsigned)pIndexListStatus->uiKeyCount,
					(unsigned)pIndexListStatus->uiRefCount);
			goto Quit_List;
		}

		// Get the next key/reference.

		if (RC_BAD( rc = FlmKeyRetrieve( hDb, uiIndex,
									pSrchKey->getContainerID(),
									pSrchKey, uiSrchDrn, uiSrchFlag,
									&pSrchKey, &uiSrchDrn)))
		{
			if (rc == FERR_EOF_HIT)
			{
				if (bNewKey)
				{
					f_sprintf( pszEndStatus, "Index list done, KeyCnt=%u, RefCnt=%u",
						(unsigned)pIndexListStatus->uiKeyCount,
						(unsigned)pIndexListStatus->uiRefCount);
					goto Quit_List;
				}
				uiSrchFlag = FO_EXCL;
				bNewKey = TRUE;
				rc = FERR_OK;
				continue;
			}
			else
			{
				f_sprintf( pszEndStatus, "Read Error %04X, KeyCnt=%u, RefCnt=%u",
					(unsigned)rc, (unsigned)pIndexListStatus->uiKeyCount,
					(unsigned)pIndexListStatus->uiRefCount);
				goto Quit_List;
			}
		}
		pSrchKey->setID( uiSrchDrn);

		if (bNewKey)
		{
			FLMINT	iCmp;
			FLMUINT	uiCmpLen;

			// See if we have gone past the until key.

			if (RC_BAD( rc = FlmKeyBuild( hDb, uiIndex,
												pSrchKey->getContainerID(),
												pSrchKey, 0,
												pucFoundKeyBuf, &uiFoundKeyLen)))
			{
				f_sprintf( pszEndStatus, "Error Building Key Buf %04X",
					(unsigned)rc);
				goto Quit_List;
			}
			if ((uiCmpLen = uiUntilKeyLen) > uiFoundKeyLen)
			{
				uiCmpLen = uiFoundKeyLen;
			}
			iCmp = f_memcmp( pucFoundKeyBuf, pucUntilKeyBuf, uiCmpLen);
			if ((iCmp > 0) ||
				 (iCmp == 0 && uiFoundKeyLen > uiUntilKeyLen))
			{
				f_sprintf( pszEndStatus, "Index list done, KeyCnt=%u, RefCnt=%u",
						(unsigned)pIndexListStatus->uiKeyCount,
						(unsigned)pIndexListStatus->uiRefCount);
				goto Quit_List;
			}

			// Save a new key to the list.

			if (pIndexListStatus->uiKeyCount == pIndexListStatus->uiKeyListSize)
			{
				KEY_ELEMENT *	pTmpKeyList;

				if( RC_BAD( rc = f_alloc( sizeof( KEY_ELEMENT) * 
					(pIndexListStatus->uiKeyListSize + KEY_LIST_INCREASE_SIZE),
					&pTmpKeyList)))
				{
					f_strcpy( pszEndStatus, "Could not allocate key list");
					goto Quit_List;
				}

				if (pIndexListStatus->pKeyList)
				{
					f_memcpy( pTmpKeyList, pIndexListStatus->pKeyList,
						sizeof( KEY_ELEMENT) * pIndexListStatus->uiKeyCount);
					f_free( &pIndexListStatus->pKeyList);
				}
				pIndexListStatus->pKeyList = pTmpKeyList;
				pIndexListStatus->uiKeyListSize += KEY_LIST_INCREASE_SIZE;
			}
			if ((pIndexListStatus->pKeyList [pIndexListStatus->uiKeyCount].pKey =
					pSrchKey->copy()) == NULL)
			{
				f_strcpy( pszEndStatus, "Could not allocate key");
				goto Quit_List;
			}
			pIndexListStatus->pKeyList [pIndexListStatus->uiKeyCount].uiRefCnt = 0;
			pIndexListStatus->pKeyList [pIndexListStatus->uiKeyCount].uiRefStartOffset =
				pIndexListStatus->uiRefCount;
			pIndexListStatus->uiKeyCount++;
			bNewKey = FALSE;
			uiSrchFlag = FO_EXCL | FO_KEY_EXACT;
		}

		// VISIT: Need to see if we have gone past the UNTIL drn if we are on
		// the until key.

		// Save the DRN to the reference list.

		if (pIndexListStatus->uiRefCount == pIndexListStatus->uiRefListSize)
		{
			FLMUINT *	puiTmpRefList;

			if( RC_BAD( rc = f_alloc( sizeof( FLMUINT) * 
				(pIndexListStatus->uiRefListSize + REF_LIST_INCREASE_SIZE),
				&puiTmpRefList)))
			{
				f_strcpy( pszEndStatus, "Could not allocate reference list");
				goto Quit_List;
			}

			if (pIndexListStatus->puiRefList)
			{
				f_memcpy( puiTmpRefList, pIndexListStatus->puiRefList,
					sizeof( FLMUINT) * pIndexListStatus->uiRefCount);
				f_free( &pIndexListStatus->puiRefList);
			}
			pIndexListStatus->puiRefList = puiTmpRefList;
			pIndexListStatus->uiRefListSize += REF_LIST_INCREASE_SIZE;
		}
		pIndexListStatus->pKeyList [pIndexListStatus->uiKeyCount - 1].uiRefCnt++;
		pIndexListStatus->puiRefList [pIndexListStatus->uiRefCount] = uiSrchDrn;
		pIndexListStatus->uiRefCount++;

		// Update thread status every 20 seconds.  Also start a
		// new transaction.

		uiCurrTime = FLM_GET_TIMER();
		if (FLM_ELAPSED_TIME( uiCurrTime, uiLastTimeSetStatus) >=
					ui20SecsTime)
		{
			pThread->setThreadStatus( "KeyCnt=%u, RefCnt=%u", 
				(unsigned)pIndexListStatus->uiKeyCount,
				(unsigned)pIndexListStatus->uiRefCount);
			uiLastTimeSetStatus = uiCurrTime;

			bTransStarted = FALSE;
			(void)FlmDbTransCommit( hDb);
			if (RC_BAD( rc = FlmDbTransBegin( hDb, FLM_READ_TRANS, 0)))
			{
				f_sprintf( pszEndStatus, "Trans Error %04X", (unsigned)rc);
				goto Quit_List;
			}
			bTransStarted = TRUE;
		}

		continue;

Quit_List:

		pThread->setThreadStatus( pszEndStatus);

		if (bTransStarted)
		{
			bTransStarted = FALSE;

			// Only a read transaction - don't care if committed or aborted

			(void)FlmDbTransCommit( hDb);
		}

		// Close the database.

		FlmDbClose( &hDb);

		pIndexListStatus->bIndexListRunning = FALSE;

		if (pSrchKey)
		{
			pSrchKey->Release();
			pSrchKey = NULL;
		}

		if (pucUntilKeyBuf)
		{
			f_free( &pucUntilKeyBuf);
		}

		// Continue until told to shut down or until we
		// timeout.

		continue;
	}

Exit:

	// Abort the transaction if we still have one going.

	if (bTransStarted)
	{
		(void)FlmDbTransAbort( hDb);
	}

	// Close the database.

	if (hDb != HFDB_NULL)
	{
		FlmDbClose( &hDb);
	}

	if (pSrchKey)
	{
		pSrchKey->Release();
		pSrchKey = NULL;
	}

	if (pucUntilKeyBuf)
	{
		f_free( &pucUntilKeyBuf);
	}

	// Set the thread's app ID to 0, so that it will not
	// be found now that the thread is terminating (we don't
	// want getIndexListStatus() to find the thread).

	pThread->setThreadAppId( 0);

	// Free the index list status.  Must do inside mutex lock so
	// that it doesn't go away after getIndexListStatus finds the
	// thread.

	f_mutexLock( gv_FlmSysData.hShareMutex);
	freeIndexListStatus( pIndexListStatus, TRUE);
	f_mutexUnlock( gv_FlmSysData.hShareMutex);
	return( FERR_OK);
}
