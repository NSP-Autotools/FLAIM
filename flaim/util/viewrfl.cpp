//-------------------------------------------------------------------------
// Desc:	View the roll-forward log.
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
#include "sharutil.h"
#include "flm_edit.h"
#include "flmarg.h"
#include "fform.h"

#ifdef FLM_NLM
	extern "C"
	{
		FLMBOOL	gv_bSynchronized = FALSE;

		void SynchronizeStart();

		int nlm_main(
			int		ArgC,
			char **	ArgV);

		int atexit( void (*)( void ) );
	}

	FSTATIC void viewRflCleanup( void);
#endif

#define MAIN_MODULE
#include "rflread.h"

#define SRCH_LABEL_COLUMN	5
#define SRCH_ENTER_COLUMN	38

#define SRCH_PACKET_TYPE_TAG	1
#define SRCH_TRANS_ID_TAG		2
#define SRCH_CONTAINER_TAG		3
#define SRCH_INDEX_TAG			4
#define SRCH_DRN_TAG				5
#define SRCH_END_DRN_TAG		6
#define SRCH_MULTI_FILE_TAG	7

FSTATIC void viewRflFormatSerialNum(
	FLMBYTE *	pszBuf,
	FLMBYTE *	pucSerialNum);

FSTATIC RCODE viewRflShowHeader(
	F_RecEditor *		pParentEditor);

FSTATIC RCODE viewRflHeaderDispHook(
	F_RecEditor *			pRecEditor,
	NODE *					pNd,
	void *					UserData,
	DBE_DISP_COLUMN *		pDispVals,
	FLMUINT *				puiNumVals);

FSTATIC RCODE viewRflGetEOF( void);

FSTATIC RCODE rflOpenNewFile(
	F_RecEditor *		pRecEditor,
	const char *		pszFileName,
	FLMBOOL				bPosAtBOF,
	F_Pool *				pTmpPool,
	NODE **				ppNd);

/*
NetWare hooks
*/

// Local Prototypes

void UIMain(
	int			ArgC,
	char **		ArgV);

RCODE viewRflMainKeyHook(
	F_RecEditor *		pRecEditor,
	NODE *				pCurNd,
	FLMUINT				uiKeyIn,
	void *				UserData,
	FLMUINT *			puiKeyOut);

RCODE viewRflMainHelpHook(
	F_RecEditor *		pRecEditor,
	F_RecEditor *		pHelpEditor,
	F_Pool *				pPool,
	void *				UserData,
	NODE **				ppRootNd);

RCODE viewRflMainEventHook(
	F_RecEditor *		pRecEditor,
	eEventType			eEventType,
	void *				EventData,
	void *				UserData);

RCODE viewRflInspectEntry(
	F_RecEditor *		pParentEditor);

RCODE viewRflInspectEventHook(
	F_RecEditor *		pRecEditor,
	eEventType			eEventType,
	void *				EventData,
	void *				UserData);

RCODE viewRflInspectDispHook(
	F_RecEditor *			pRecEditor,
	NODE *					pNd,
	void *					UserData,
	DBE_DISP_COLUMN *		pDispVals,
	FLMUINT *				puiNumVals);

RCODE viewRflInspectKeyHook(
	F_RecEditor *		pRecEditor,
	NODE *				pCurNd,
	FLMUINT				uiKeyIn,
	void *				UserData,
	FLMUINT *			puiKeyOut);

RCODE viewRflNameTableInit(
	F_NameTable **		ppNameTable);

FSTATIC RCODE addLabel(
	FlmForm *			pForm,
	FLMUINT				uiObjectId,
	const char *		pszLabel,
	FLMUINT				uiRow);

FSTATIC FLMBOOL editSearchFormCB(
	FormEventType		eFormEvent,
	FlmForm *			pForm,
	FlmFormObject *	pFormObject,
	FLMUINT				uiKeyIn,
	FLMUINT *			puiKeyOut,
	void *				pvAppData);

FSTATIC RCODE addPullDownPacketType(
	FlmForm *		pForm,
	FLMUINT			uiPacketType,
	const char *	pszPacketTypeName);
	
FSTATIC RCODE getSearchCriteria(
	F_RecEditor *	pRecEditor,
	RFL_PACKET *	pSrchCriteria,
	FLMBOOL *		pbForward);

FSTATIC FLMBOOL rflPassesCriteria(
	RFL_PACKET *	pPacket,
	RFL_PACKET *	pSrchPacket);

/*--------------------------------------------------------
** Local (to this file only) global variables.
**-------------------------------------------------------*/
RFL_PACKET						gv_SrchCriteria;
FLMBOOL							gv_bSrchForward;
FLMBOOL							gv_bDoRefresh = TRUE;
FLMBOOL							gv_bShutdown = FALSE;
const char *					gv_pszTitle = "FLAIM RFL Viewer v1.00";
char								gv_szRflPath [F_PATH_MAX_SIZE];
static F_NameTable *			gv_pNameTable = NULL;
#ifdef FLM_NLM
	static FLMBOOL				gv_bRunning = TRUE;
#endif


/****************************************************************************
Name: main
****************************************************************************/
#if defined( FLM_UNIX)
int main(
	int			ArgC,
	char **		ArgV
	)
#elif defined( FLM_NLM)
int nlm_main(
	int			ArgC,
	char **		ArgV
	)
#else
int __cdecl main(
	int			ArgC,
	char **		ArgV
	)
#endif   
{
	int	iResCode = 0;

	if( RC_BAD( FlmStartup()))
	{
		iResCode = -1;
		goto Exit;
	}

#ifdef FLM_NLM

	/* Setup the routines to be called when the NLM exits itself */
	
	atexit( viewRflCleanup);

#endif

	UIMain( ArgC, ArgV);

Exit:

	FlmShutdown();

#ifdef FLM_NLM
	if (!gv_bSynchronized)
	{
		SynchronizeStart();
		gv_bSynchronized = TRUE;
	}
	gv_bRunning = FALSE;
#endif

	return( iResCode);
}


/****************************************************************************
Name: UIMain
****************************************************************************/
void UIMain(
	int			iArgC,
	char **		ppszArgV
	)
{
	FTX_SCREEN *		pScreen = NULL;
	FTX_WINDOW *		pTitleWin = NULL;
	F_RecEditor	*		pRecEditor = NULL;
	FLMUINT				uiTermChar;
	RCODE					rc = FERR_OK;

	gv_pRflFileHdl = NULL;
	gv_ui64RflEof = 0;
	f_memset( &gv_SrchCriteria, 0, sizeof( gv_SrchCriteria));
	gv_bSrchForward = TRUE;
	gv_SrchCriteria.uiPacketType = 0xFFFFFFFF;
	gv_SrchCriteria.uiMultiFileSearch = 1;

	if( RC_BAD( rc = FTXInit( gv_pszTitle, (FLMUINT)80, (FLMUINT)50,
		FLM_BLUE, FLM_WHITE, NULL, NULL)))
	{
		goto Exit;
	}

	FTXSetShutdownFlag( &gv_bShutdown);

	if( RC_BAD( rc = FTXScreenInit( gv_pszTitle, &pScreen)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FTXWinInit( pScreen, 0, 1, &pTitleWin)))
	{
		goto Exit;
	}

	FTXWinPaintBackground( pTitleWin, FLM_RED);

	FTXWinPrintStr( pTitleWin, gv_pszTitle);

	FTXWinSetCursorType( pTitleWin, FLM_CURSOR_INVISIBLE);

	if( RC_BAD( rc = FTXWinOpen( pTitleWin)))
	{
		goto Exit;
	}

	if( (pRecEditor = f_new F_RecEditor) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( pRecEditor->Setup( pScreen)))
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	pRecEditor->setTree( NULL);
	pRecEditor->setShutdown( &gv_bShutdown);
	pRecEditor->setKeyHook( viewRflMainKeyHook, 0);
	pRecEditor->setHelpHook( viewRflMainHelpHook, 0);
	pRecEditor->setEventHook( viewRflMainEventHook, (void *)0);

	/*
	Fire up the editor
	*/

	gv_szRflPath [0] = 0;
	if (iArgC > 1)
	{
		f_strcpy( gv_szRflPath, ppszArgV [1]);
	}

	if (!gv_szRflPath [0])
	{
		pRecEditor->requestInput(
				"Log File Name", gv_szRflPath,
				sizeof( gv_szRflPath), &uiTermChar);

		if( uiTermChar == FKB_ESCAPE)
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = pRecEditor->getFileSystem()->openFile( gv_szRflPath,
			  			FLM_IO_RDWR | FLM_IO_SH_DENYNONE, &gv_pRflFileHdl)))
	{
		pRecEditor->displayMessage( "Unable to open file", rc,
			NULL, FLM_RED, FLM_WHITE);
		rc = FERR_OK;
	}
	else
	{
		viewRflNameTableInit( &gv_pNameTable);
		pRecEditor->setTitle( gv_szRflPath);
		pRecEditor->interactiveEdit( 0, 1);
		pRecEditor->setTree( NULL);
		if( gv_pNameTable)
		{
			gv_pNameTable->Release();
			gv_pNameTable = NULL;
		}
		gv_pRflFileHdl->Release();
		gv_pRflFileHdl = NULL;
	}

Exit:

	gv_bShutdown = TRUE;

	if( pRecEditor)
	{
		pRecEditor->Release();
		pRecEditor = NULL;
	}

	if( gv_pRflFileHdl)
	{
		gv_pRflFileHdl->Release();
	}

	FTXExit();
}


#ifdef FLM_NLM
/****************************************************************************
Desc: This routine shuts down all threads in the NLM.
****************************************************************************/
void viewRflCleanup( void)
{
	gv_bShutdown = TRUE;
	while( gv_bRunning)
	{
		f_sleep( 10);
		f_yieldCPU();
	}
}
#endif


/********************************************************************
Desc: Add a label to a form.
*********************************************************************/
FSTATIC RCODE addLabel(
	FlmForm *		pForm,
	FLMUINT			uiObjectId,
	const char *	pszLabel,
	FLMUINT			uiRow)
{
	FLMUINT	uiLen = f_strlen( pszLabel);

	return( pForm->addTextObject( uiObjectId, pszLabel,
		uiLen, uiLen,
		0, TRUE, FLM_BLUE, FLM_WHITE,
		uiRow, SRCH_LABEL_COLUMN));
}

/****************************************************************************
Desc:	Callback function for search form.
*****************************************************************************/
FSTATIC FLMBOOL editSearchFormCB(
	FormEventType		eFormEvent,
	FlmForm *			pForm,
	FlmFormObject *	pFormObject,
	FLMUINT				uiKeyIn,
	FLMUINT *			puiKeyOut,
	void *				pvAppData
	)
{
	F_UNREFERENCED_PARM( pForm);
	F_UNREFERENCED_PARM( pFormObject);
	F_UNREFERENCED_PARM( puiKeyOut);
	F_UNREFERENCED_PARM( pvAppData);

	if (eFormEvent == FORM_EVENT_KEY_STROKE)
	{
		switch (uiKeyIn)
		{
			case FKB_F1:
			case FKB_F2:
			case FKB_F3:
			case FKB_F4:
			case FKB_F5:
			case FKB_F6:
			case FKB_F7:
			case FKB_F8:
			case FKB_F9:
			case FKB_F10:
			case FKB_F11:
			case FKB_F12:
				return( FALSE);
			default:
				return( TRUE);
		}
	}
	return( TRUE);
}

/********************************************************************
Desc: Add a packet type to a pulldown list
*********************************************************************/
FSTATIC RCODE addPullDownPacketType(
	FlmForm *		pForm,
	FLMUINT			uiPacketType,
	const char *	pszPacketTypeName)
{
	char	szDisplayValue [100];
	
	f_sprintf( szDisplayValue, "%s (%u)", pszPacketTypeName,
			(unsigned)uiPacketType);
	return( pForm->addPulldownItem( SRCH_PACKET_TYPE_TAG, uiPacketType,
								szDisplayValue, 0));
}

/********************************************************************
Desc: Add a label to a form.
*********************************************************************/
FSTATIC RCODE getSearchCriteria(
	F_RecEditor *	pRecEditor,
	RFL_PACKET *	pSrchCriteria,
	FLMBOOL *		pbForward
	)
{
	RCODE				rc = FERR_OK;
	FTX_SCREEN *	pScreen = pRecEditor->getScreen();
	FlmForm *		pForm = NULL;
	FLMUINT			uiRow = 1;
	FLMUINT			uiScreenCols;
	FLMUINT			uiScreenRows;
	FLMUINT			uiChar = 0;
	FLMBOOL			bValuesChanged;
	FLMUINT			uiCurrObjectId;
	const char *	pszWhat = NULL;

	if (RC_BAD( rc = FTXScreenGetSize( pScreen, &uiScreenCols, &uiScreenRows)))
	{
		pszWhat = "getting screen size";
		goto Exit;
	}

	if ((pForm = f_new FlmForm) == NULL)
	{
		pszWhat = "allocating form";
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = pForm->init( pScreen, NULL,
						"Search Criteria",
						FLM_BLUE, FLM_WHITE,
						"ESC=Quit, F1=search forward, other=search backward",
						FLM_BLUE, FLM_WHITE,
						0, 0,
						uiScreenCols - 1, uiScreenRows - 1, TRUE, TRUE,
						FLM_BLUE, FLM_LIGHTGRAY)))
	{
		pszWhat = "initializing form";
		goto Exit;
	}

	// Add the packet type selection field.

	pszWhat = "adding packet type";
	if (RC_BAD( rc = addLabel( pForm, SRCH_PACKET_TYPE_TAG + 100,
									"Packet Type", uiRow)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownObject( SRCH_PACKET_TYPE_TAG,
									20, 10,
									FLM_LIGHTGRAY, FLM_RED, uiRow, SRCH_ENTER_COLUMN)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, RFL_TRNS_BEGIN_PACKET,
								"Transaction Begin")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, RFL_TRNS_BEGIN_EX_PACKET,
								"Transaction Begin Ext")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, RFL_TRNS_COMMIT_PACKET,
								"Transaction Commit")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, RFL_TRNS_ABORT_PACKET,
								"Transaction Abort")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, RFL_ADD_RECORD_PACKET,
								"Add Record")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, RFL_MODIFY_RECORD_PACKET,
								"Modify Record")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, RFL_DELETE_RECORD_PACKET,
								"Delete Record")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, RFL_RESERVE_DRN_PACKET,
								"Reserve DRN")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, RFL_CHANGE_FIELDS_PACKET,
								"Change Fields")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, RFL_DATA_RECORD_PACKET,
								"Data Record")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, RFL_INDEX_SET_PACKET,
								"Index Set")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, RFL_INDEX_SET_PACKET_VER_2,
								"Index Set V2")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, RFL_BLK_CHAIN_FREE_PACKET,
								"Block Chain Free")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, RFL_START_UNKNOWN_PACKET,
								"Start Unknown")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, RFL_UNKNOWN_PACKET,
								"User Unknown")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, RFL_REDUCE_PACKET,
								"Reduce")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, RFL_UPGRADE_PACKET,
								"Upgrade")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, RFL_INDEX_SUSPEND_PACKET,
								"Index Suspend")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, RFL_INDEX_RESUME_PACKET,
								"Index Resume")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, RFL_ADD_RECORD_PACKET_VER_2,
								"Add Record V2")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, RFL_MODIFY_RECORD_PACKET_VER_2,
								"Modify Record V2")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, RFL_DELETE_RECORD_PACKET_VER_2,
								"Delete Record V2")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, RFL_ENC_DATA_RECORD_PACKET,
								"Data Record Encrypted")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, RFL_DATA_RECORD_PACKET_VER_3,
								"Data Record V3")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, RFL_WRAP_KEY_PACKET,
								"Wrap Encryption Key")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, RFL_ENABLE_ENCRYPTION_PACKET,
								"Enable Encryption")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, RFL_CONFIG_SIZE_EVENT_PACKET,
								"Configure RFL Size")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = addPullDownPacketType( pForm, 0xFFFFFFFF,
								"All packet types")))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectValue( SRCH_PACKET_TYPE_TAG,
										(void *)pSrchCriteria->uiPacketType, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectReturnAddress( SRCH_PACKET_TYPE_TAG,
									&pSrchCriteria->uiPacketType, NULL)))
	{
		goto Exit;
	}
	uiRow += 2;

	// Add the transaction ID field

	pszWhat = "adding transaction ID";
	if (RC_BAD( rc = addLabel( pForm, SRCH_TRANS_ID_TAG + 100,
									"Transaction ID", uiRow)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addUnsignedObject( SRCH_TRANS_ID_TAG,
					pSrchCriteria->uiTransID,
					0, 0xFFFFFFFF, 10,
					0, FALSE, FLM_LIGHTGRAY, FLM_RED, uiRow, SRCH_ENTER_COLUMN)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectReturnAddress( SRCH_TRANS_ID_TAG,
									&pSrchCriteria->uiTransID, NULL)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectHelp( SRCH_TRANS_ID_TAG,
				"0=Match any trans ID, other=Specific trans ID to find",
				NULL)))
	{
		goto Exit;
	}
	uiRow += 2;

	// Add the Container field

	pszWhat = "adding container";
	if (RC_BAD( rc = addLabel( pForm, SRCH_CONTAINER_TAG + 100,
									"Container", uiRow)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addUnsignedObject( SRCH_CONTAINER_TAG,
					pSrchCriteria->uiContainer,
					0, 0xFFFF, 5,
					0, FALSE, FLM_LIGHTGRAY, FLM_RED, uiRow, SRCH_ENTER_COLUMN)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectReturnAddress( SRCH_CONTAINER_TAG,
									&pSrchCriteria->uiContainer, NULL)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectHelp( SRCH_CONTAINER_TAG,
				"0=Match any container, other=Specific container to find",
				NULL)))
	{
		goto Exit;
	}
	uiRow += 2;

	// Add the index field

	pszWhat = "adding index";
	if (RC_BAD( rc = addLabel( pForm, SRCH_INDEX_TAG + 100,
									"Index", uiRow)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addUnsignedObject( SRCH_INDEX_TAG,
					pSrchCriteria->uiIndex,
					0, 0xFFFF, 5,
					0, FALSE, FLM_LIGHTGRAY, FLM_RED, uiRow, SRCH_ENTER_COLUMN)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectReturnAddress( SRCH_INDEX_TAG,
									&pSrchCriteria->uiIndex, NULL)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectHelp( SRCH_INDEX_TAG,
				"0=Match any index, other=Specific index to find",
				NULL)))
	{
		goto Exit;
	}
	uiRow += 2;

	// Add the DRN field

	pszWhat = "adding DRN";
	if (RC_BAD( rc = addLabel( pForm, SRCH_DRN_TAG + 100,
									"DRN", uiRow)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addUnsignedObject( SRCH_DRN_TAG,
					pSrchCriteria->uiDrn,
					0, 0xFFFFFFFF, 10,
					0, FALSE, FLM_LIGHTGRAY, FLM_RED, uiRow, SRCH_ENTER_COLUMN)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectReturnAddress( SRCH_DRN_TAG,
									&pSrchCriteria->uiDrn, NULL)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectHelp( SRCH_DRN_TAG,
				"0=Match any DRN, other=Specific DRN to find",
				NULL)))
	{
		goto Exit;
	}
	uiRow += 2;

	// Add the End DRN field

	pszWhat = "adding end DRN";
	if (RC_BAD( rc = addLabel( pForm, SRCH_END_DRN_TAG + 100,
									"End DRN", uiRow)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addUnsignedObject( SRCH_END_DRN_TAG,
					pSrchCriteria->uiEndDrn,
					0, 0xFFFFFFFF, 10,
					0, FALSE, FLM_LIGHTGRAY, FLM_RED, uiRow, SRCH_ENTER_COLUMN)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectReturnAddress( SRCH_END_DRN_TAG,
									&pSrchCriteria->uiEndDrn, NULL)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectHelp( SRCH_END_DRN_TAG,
				"0=Match any End DRN, other=Specific End DRN to find",
				NULL)))
	{
		goto Exit;
	}
	uiRow += 2;

	// Add the packet type selection field.

	pszWhat = "adding multi-file flag";
	if (RC_BAD( rc = addLabel( pForm, SRCH_MULTI_FILE_TAG + 100,
									"Search Multiple Files", uiRow)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownObject( SRCH_MULTI_FILE_TAG,
									20, 10,
									FLM_LIGHTGRAY, FLM_RED, uiRow, SRCH_ENTER_COLUMN)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownItem( SRCH_MULTI_FILE_TAG, 1,
								"Y=Yes", (FLMUINT)'Y')))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownItem( SRCH_MULTI_FILE_TAG, 2,
								"N=No", (FLMUINT)'N')))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectValue( SRCH_MULTI_FILE_TAG,
									(void *)pSrchCriteria->uiMultiFileSearch, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectReturnAddress( SRCH_MULTI_FILE_TAG,
									&pSrchCriteria->uiMultiFileSearch, NULL)))
	{
		goto Exit;
	}
	uiRow += 2;


	pForm->setFormEventCB( editSearchFormCB, NULL, TRUE);
	uiChar = pForm->interact( &bValuesChanged, &uiCurrObjectId);

	if (uiChar == FKB_ESCAPE)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
	*pbForward = (FLMBOOL)((uiChar == FKB_F1)
								  ? TRUE
								  : FALSE);

	if (RC_BAD( rc = pForm->getAllReturnData()))
	{
		pszWhat = "getting return data";
		goto Exit;
	}

Exit:
	if (RC_BAD( rc) && uiChar != FKB_ESCAPE)
	{
		char	szErrMsg [100];

		f_sprintf( (char *)szErrMsg, "Error %s", pszWhat);
		pRecEditor->displayMessage( szErrMsg, rc,
						NULL, FLM_RED, FLM_WHITE);
	}
	if (pForm)
	{
		pForm->Release();
	}
	return( rc);
}

/****************************************************************************
Desc:	See if a packet passes the search criteria.
*****************************************************************************/
FSTATIC FLMBOOL rflPassesCriteria(
	RFL_PACKET *	pPacket,
	RFL_PACKET *	pSrchPacket
	)
{
	FLMBOOL	bPasses = FALSE;

	if (pSrchPacket->uiPacketType != pPacket->uiPacketType &&
		 pSrchPacket->uiPacketType != 0xFFFFFFFF)
	{
		goto Exit;
	}
	if (pSrchPacket->uiTransID != pPacket->uiTransID &&
		 pSrchPacket->uiTransID != 0)
	{
		goto Exit;
	}
	if (pSrchPacket->uiIndex != pPacket->uiIndex &&
		 pSrchPacket->uiIndex != 0)
	{
		goto Exit;
	}
	if (pSrchPacket->uiContainer != pPacket->uiContainer &&
		 pSrchPacket->uiContainer != 0)
	{
		goto Exit;
	}
	if (pSrchPacket->uiDrn != pPacket->uiDrn &&
		 pSrchPacket->uiDrn != 0)
	{
		goto Exit;
	}
	if (pSrchPacket->uiEndDrn != pPacket->uiEndDrn &&
		 pSrchPacket->uiEndDrn != 0)
	{
		goto Exit;
	}
	bPasses = TRUE;
Exit:
	return( bPasses);
}

/***************************************************************************
Desc: Format a serial number for display.
*****************************************************************************/
FSTATIC void viewRflFormatSerialNum(
	FLMBYTE *	pszBuf,
	FLMBYTE *	pucSerialNum
	)
{
	f_sprintf( (char *)pszBuf,
			"%02X%02X%02X%02X-%02X%02X-%02X%02X-%02X%02X-%02X%02X%02X%02X%02X%02X",
			(unsigned)pucSerialNum[ 0],
			(unsigned)pucSerialNum[ 1],
			(unsigned)pucSerialNum[ 2],
			(unsigned)pucSerialNum[ 3],
			(unsigned)pucSerialNum[ 4],
			(unsigned)pucSerialNum[ 5],
			(unsigned)pucSerialNum[ 6],
			(unsigned)pucSerialNum[ 7],
			(unsigned)pucSerialNum[ 8],
			(unsigned)pucSerialNum[ 9],
			(unsigned)pucSerialNum[ 10],
			(unsigned)pucSerialNum[ 11],
			(unsigned)pucSerialNum[ 12],
			(unsigned)pucSerialNum[ 13],
			(unsigned)pucSerialNum[ 14],
			(unsigned)pucSerialNum[ 15]);
}

/****************************************************************************
Desc:
*****************************************************************************/
FSTATIC RCODE viewRflHeaderDispHook(
	F_RecEditor *			pRecEditor,
	NODE *					pNd,
	void *					UserData,
	DBE_DISP_COLUMN *		pDispVals,
	FLMUINT *				puiNumVals)
{
#define LABEL_WIDTH	32
	FLMUINT		uiCol = 0;
	FLMUINT		uiTag = 0;
	FLMUINT		uiLen;
	char *		pszTmp;
	RCODE			rc = FERR_OK;

	F_UNREFERENCED_PARM( UserData);

	if (!pNd)
	{
		goto Exit;
	}

	uiTag = GedTagNum( pNd);
	if (!pRecEditor->isSystemNode( pNd))
	{

		// Output the tag number.

		pszTmp = (char *)pDispVals [*puiNumVals].pucString;
		switch (uiTag)
		{
			case RFL_HDR_NAME_FIELD:
				f_strcpy( pszTmp, "RFL Name");
				break;
			case RFL_HDR_VERSION_FIELD:
				f_strcpy( pszTmp, "RFL Version");
				break;
			case RFL_HDR_FILE_NUMBER_FIELD:
				f_strcpy( pszTmp, "RFL File Number");
				break;
			case RFL_HDR_EOF_FIELD:
				f_strcpy( pszTmp, "File EOF");
				break;
			case RFL_HDR_DB_SERIAL_NUM_FIELD:
				f_strcpy( pszTmp, "Database Serial Number");
				break;
			case RFL_HDR_FILE_SERIAL_NUM_FIELD:
				f_strcpy( pszTmp, "RFL File Serial Number");
				break;
			case RFL_HDR_NEXT_FILE_SERIAL_NUM_FIELD:
				f_strcpy( pszTmp, "Next RFL File Serial Number");
				break;
			case RFL_HDR_KEEP_SIGNATURE_FIELD:
				f_strcpy( pszTmp, "Keep RFL Files Signature");
				break;
			default:
				f_sprintf( pszTmp, "TAG_%u", (unsigned)uiTag);
				break;
		}
		uiLen = f_strlen( pszTmp);
		if (uiLen < LABEL_WIDTH)
		{
			f_memset( &pszTmp [uiLen], '.', LABEL_WIDTH - uiLen);
		}
		pszTmp [LABEL_WIDTH] = ' ';
		pszTmp [LABEL_WIDTH + 1] = 0;
		pDispVals[ *puiNumVals].uiCol = uiCol;
		pDispVals[ *puiNumVals].foreground = FLM_WHITE;
		pDispVals[ *puiNumVals].background = FLM_BLUE;
		(*puiNumVals)++;
		uiCol += (LABEL_WIDTH + 1);

		// Output the value.

		pDispVals[ *puiNumVals].uiCol = uiCol;
		pDispVals[ *puiNumVals].foreground = FLM_YELLOW;
		pDispVals[ *puiNumVals].background = FLM_BLUE;

		(void)pRecEditor->getDisplayValue( pNd,
								F_RECEDIT_DEFAULT_TYPE,
								pDispVals[ *puiNumVals].pucString,
								sizeof( pDispVals[ *puiNumVals].pucString));
		(*puiNumVals)++;
	}

Exit:
	return( rc);
}

/****************************************************************************
Desc:	Shows the header of an RFL file.
*****************************************************************************/
FSTATIC RCODE viewRflShowHeader(
	F_RecEditor *		pParentEditor)
{
	F_RecEditor *		pRecEditor;
	NODE *				pHeaderNode;
	NODE *				pNode;
	FLMBYTE				ucHdrBuf [512];
	FLMUINT				uiBytesRead;
	FLMBYTE				szTmp [100];
	FLMUINT				uiTmp;
	F_Pool				tmpPool;
	RCODE					rc = FERR_OK;

	tmpPool.poolInit( 1024);

	if( (pRecEditor = f_new F_RecEditor) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( pRecEditor->Setup( pParentEditor->getScreen())))
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	pRecEditor->setTree( NULL);
	pRecEditor->setShutdown( &gv_bShutdown);
	pRecEditor->setDisplayHook( viewRflHeaderDispHook, 0);
	pRecEditor->setEventHook( viewRflInspectEventHook, (void *)0);
//	pRecEditor->setKeyHook( viewRflInspectKeyHook, 0);
	pRecEditor->setTitle( "RFL Header");

	// Read the header from the file.

	if (RC_BAD( rc = gv_pRflFileHdl->read( 0, 512, ucHdrBuf, &uiBytesRead)))
	{
		goto Exit;
	}

	// Create the name field

	if ((pNode = GedNodeCreate( &tmpPool, makeTagNum( RFL_HDR_NAME_FIELD),
							0, &rc)) == NULL)
	{
		goto Exit;
	}
	f_memcpy( szTmp, &ucHdrBuf [RFL_NAME_POS], RFL_NAME_LEN);
	szTmp [RFL_NAME_LEN] = 0;
	if (RC_BAD( rc = GedPutNATIVE( &tmpPool, pNode, (char *)szTmp)))
	{
		goto Exit;
	}
	pHeaderNode = pNode;

	// Create the version field

	if ((pNode = GedNodeCreate( &tmpPool,
							makeTagNum( RFL_HDR_VERSION_FIELD), 0, &rc)) == NULL)
	{
		goto Exit;
	}
	f_memcpy( szTmp, &ucHdrBuf [RFL_VERSION_POS], RFL_VERSION_LEN);
	szTmp [RFL_VERSION_LEN] = 0;
	if (RC_BAD( rc = GedPutNATIVE( &tmpPool, pNode, (char *)szTmp)))
	{
		goto Exit;
	}
	GedSibGraft( pHeaderNode, pNode, GED_LAST);

	// Create the file number field

	if ((pNode = GedNodeCreate( &tmpPool, makeTagNum( RFL_HDR_FILE_NUMBER_FIELD),
							0, &rc)) == NULL)
	{
		goto Exit;
	}
	uiTmp = (FLMUINT)FB2UD( &ucHdrBuf [RFL_FILE_NUMBER_POS]);
	if (RC_BAD( rc = GedPutUINT( &tmpPool, pNode, uiTmp)))
	{
		goto Exit;
	}
	GedSibGraft( pHeaderNode, pNode, GED_LAST);

	// Create the EOF field

	if ((pNode = GedNodeCreate( &tmpPool, makeTagNum( RFL_HDR_EOF_FIELD),
							0, &rc)) == NULL)
	{
		goto Exit;
	}
	uiTmp = (FLMUINT)FB2UD( &ucHdrBuf [RFL_EOF_POS]);
	if (RC_BAD( rc = GedPutUINT( &tmpPool, pNode, uiTmp)))
	{
		goto Exit;
	}
	GedSibGraft( pHeaderNode, pNode, GED_LAST);

	// Create the database serial number field

	if ((pNode = GedNodeCreate( &tmpPool, makeTagNum( RFL_HDR_DB_SERIAL_NUM_FIELD),
								0, &rc)) == NULL)
	{
		goto Exit;
	}
	viewRflFormatSerialNum( szTmp, &ucHdrBuf [RFL_DB_SERIAL_NUM_POS]);
	if (RC_BAD( rc = GedPutNATIVE( &tmpPool, pNode, (char *)szTmp)))
	{
		goto Exit;
	}
	GedSibGraft( pHeaderNode, pNode, GED_LAST);

	// Create the file serial number field

	if ((pNode = GedNodeCreate( &tmpPool, makeTagNum( RFL_HDR_FILE_SERIAL_NUM_FIELD),
								0, &rc)) == NULL)
	{
		goto Exit;
	}
	viewRflFormatSerialNum( szTmp, &ucHdrBuf [RFL_SERIAL_NUM_POS]);
	if (RC_BAD( rc = GedPutNATIVE( &tmpPool, pNode, (char *)szTmp)))
	{
		goto Exit;
	}
	GedSibGraft( pHeaderNode, pNode, GED_LAST);

	// Create the next file serial number field

	if ((pNode = GedNodeCreate( &tmpPool, makeTagNum( RFL_HDR_NEXT_FILE_SERIAL_NUM_FIELD),
								0, &rc)) == NULL)
	{
		goto Exit;
	}
	viewRflFormatSerialNum( szTmp, &ucHdrBuf [RFL_NEXT_FILE_SERIAL_NUM_POS]);
	if (RC_BAD( rc = GedPutNATIVE( &tmpPool, pNode, (char *)szTmp)))
	{
		goto Exit;
	}
	GedSibGraft( pHeaderNode, pNode, GED_LAST);

	// Create the next file serial number field

	if ((pNode = GedNodeCreate( &tmpPool, makeTagNum( RFL_HDR_KEEP_SIGNATURE_FIELD),
								0, &rc)) == NULL)
	{
		goto Exit;
	}

	// Null terminate just in case there is garbage in there.

	ucHdrBuf [RFL_KEEP_SIGNATURE_POS+50] = 0;
	if (RC_BAD( rc = GedPutNATIVE( &tmpPool, pNode,
								(char *)&ucHdrBuf [RFL_KEEP_SIGNATURE_POS])))
	{
		goto Exit;
	}
	GedSibGraft( pHeaderNode, pNode, GED_LAST);


	pRecEditor->setTree( pHeaderNode);
	pRecEditor->interactiveEdit( 0, 1);

Exit:

	if( pRecEditor)
	{
		pRecEditor->Release();
	}

	tmpPool.poolFree();
	return( rc);
}

/****************************************************************************
Desc:	Determine the RFL file EOF.
*****************************************************************************/
FSTATIC RCODE viewRflGetEOF( void)
{
	RCODE			rc = FERR_OK;
	NODE *		pTmpNd;
	F_Pool		tmpPool;
	FLMBYTE		ucHdrBuf [512];
	FLMUINT		uiBytesRead;
	FLMUINT		uiEof;

	tmpPool.poolInit( 4096);

	// First try to get the EOF from the file's header.

	if (RC_BAD( rc = gv_pRflFileHdl->read( 0, 512, ucHdrBuf, &uiBytesRead)))
	{
		goto Exit;
	}
	uiEof = (FLMUINT)FB2UD( &ucHdrBuf [RFL_EOF_POS]);
	if (uiEof)
	{
		gv_ui64RflEof = (FLMUINT64)uiEof;
	}
	else
	{

		// File's header had a zero for the EOF, so try to position to
		// the last node in the file - this should cause us to set
		// the EOF value.

		if (RC_BAD( rc = RflGetPrevNode( NULL, FALSE, &tmpPool, &pTmpNd)))
		{
			goto Exit;
		}

		// If we still didn't get an EOF value, set it to the file size.

		if (!gv_ui64RflEof)
		{
			if (RC_BAD( rc = gv_pRflFileHdl->size( &gv_ui64RflEof)))
			{
				goto Exit;
			}
		}
	}
Exit:
	tmpPool.poolFree();
	return( rc);
}

/****************************************************************************
Desc:	Opens a new RFL file.
*****************************************************************************/
FSTATIC RCODE rflOpenNewFile(
	F_RecEditor *		pRecEditor,
	const char *		pszFileName,
	FLMBOOL				bPosAtBOF,
	F_Pool *				pTmpPool,
	NODE **				ppNd)
{
	RCODE			rc = FERR_OK;
	IF_FileHdl *	pFileHdl = NULL;
	IF_FileHdl *	pSaveFileHdl = NULL;
	char			szPath [F_PATH_MAX_SIZE];
	char			szBaseName [F_FILENAME_SIZE];
	char			szPrefix [F_FILENAME_SIZE];
	FLMUINT		uiDbVersion = FLM_FILE_FORMAT_VER_4_3;
	FLMUINT		uiFileNum;

	// If no file name was specified, go to the next or previous file from
	// the current file.

	if (!pszFileName || !(*pszFileName))
	{
		if (RC_BAD( rc = f_pathReduce( gv_szRflPath, szPath, szPrefix)))
		{
			goto Exit;
		}

		// See if it is version 4.3 or greater first.

		uiDbVersion = FLM_FILE_FORMAT_VER_4_3;
		if (!rflGetFileNum( uiDbVersion, szPrefix, gv_szRflPath, &uiFileNum))
		{
			szPrefix [3] = 0;
			uiDbVersion = FLM_FILE_FORMAT_VER_4_0;
			if (!rflGetFileNum( uiDbVersion, szPrefix, gv_szRflPath, &uiFileNum))
			{
				rc = RC_SET( FERR_IO_PATH_NOT_FOUND);
				goto Exit;
			}
		}
		if (bPosAtBOF)
		{
			uiFileNum++;
		}
		else
		{
			uiFileNum--;
		}
		rflGetBaseFileName( uiDbVersion, szPrefix, uiFileNum, szBaseName);
		f_pathAppend( szPath, szBaseName);
		pszFileName = &szPath [0];
	}

	// See if we can open the next file.

	if( RC_BAD( rc = pRecEditor->getFileSystem()->openFile( pszFileName,
			  			FLM_IO_RDWR | FLM_IO_SH_DENYNONE, &pFileHdl)))
	{
		goto Exit;
	}
	pSaveFileHdl = gv_pRflFileHdl;
	gv_pRflFileHdl = pFileHdl;
	pFileHdl = NULL;
	if (RC_BAD( rc = viewRflGetEOF()))
	{
		goto Exit;
	}
	
	pRecEditor->setTree( NULL);
	if (bPosAtBOF)
	{
		if( RC_BAD( rc = RflGetNextNode( NULL, FALSE, pTmpPool, ppNd)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = RflGetPrevNode( NULL, FALSE, pTmpPool, ppNd)))
		{
			goto Exit;
		}
	}

	pRecEditor->setTree( *ppNd, ppNd);
	pRecEditor->setControlFlags( *ppNd,
			(F_RECEDIT_FLAG_HIDE_LEVEL | F_RECEDIT_FLAG_HIDE_TAG |
			F_RECEDIT_FLAG_READ_ONLY));
	pSaveFileHdl->Release();
	pSaveFileHdl = NULL;
	f_strcpy( gv_szRflPath, pszFileName);
	pRecEditor->setTitle( gv_szRflPath);
Exit:
	if (pFileHdl)
	{
		pFileHdl->Release();
	}
	if (pSaveFileHdl)
	{
		if (gv_pRflFileHdl)
		{
			gv_pRflFileHdl->Release();
		}
		gv_pRflFileHdl = pSaveFileHdl;
	}
	return( rc);
}

/****************************************************************************
Name:	viewRflMainKeyHook
Desc:	
*****************************************************************************/
RCODE viewRflMainKeyHook(
	F_RecEditor *		pRecEditor,
	NODE *				pCurNd,
	FLMUINT				uiKeyIn,
	void *				UserData,
	FLMUINT *			puiKeyOut)
{
	NODE *			pRootNd = NULL;
	NODE *			pTmpNd = NULL;
	NODE *			pNewNd;
	F_Pool			tmpPool;
	F_Pool			tmp2Pool;
	FTX_WINDOW *	pWindow = NULL;
	NODE *			pLastNd;
	NODE *			pFirstNd;
	RFL_PACKET *	pPacket;
	FLMBOOL			bSkipCurrent;
	RCODE				rc = FERR_OK;
	char				szResponse[ 80];
	FLMUINT			uiTermChar;
	FLMUINT			uiSrcLen;
	FLMUINT			uiOffset;

	F_UNREFERENCED_PARM( UserData);
	tmpPool.poolInit( 4096);
	tmp2Pool.poolInit( 4096);

	if( puiKeyOut)
	{
		*puiKeyOut = 0;
	}

	pRootNd = pRecEditor->getRootNode( pCurNd);
	switch( uiKeyIn)
	{
		case FKB_DOWN:
		case FKB_UP:
		case FKB_PGDN:
		case FKB_PGUP:
		case '?':
		{
			*puiKeyOut = uiKeyIn;
			break;
		}

		case FKB_END:
		{
			FLMUINT		uiLoop;

			pCurNd = NULL;
			pRecEditor->setTree( NULL);
			for( uiLoop = 0; uiLoop < 10; uiLoop++)
			{
				if( RC_BAD( rc = RflGetPrevNode( pCurNd, FALSE,
					&tmpPool, &pNewNd)))
				{
					goto Exit;
				}

				if( pNewNd)
				{
					if( !pCurNd)
					{
						pRecEditor->setTree( pNewNd, &pCurNd);
					}
					else
					{
						pRecEditor->insertRecord( pNewNd, &pCurNd);
					}
					pRecEditor->setControlFlags( pCurNd,
						(F_RECEDIT_FLAG_HIDE_LEVEL | F_RECEDIT_FLAG_HIDE_TAG |
						F_RECEDIT_FLAG_READ_ONLY));
				}
				else
				{
					break;
				}
				tmpPool.poolReset( NULL);
			}
			pRecEditor->setCurrentAtBottom();
			break;
		}

		case FKB_HOME:
		{
			pRecEditor->setTree( NULL);
			if( RC_BAD( rc = RflGetNextNode( NULL, FALSE, &tmpPool, 
				&pTmpNd)))
			{
				goto Exit;
			}

			pRecEditor->setTree( pTmpNd, &pNewNd);
			pRecEditor->setControlFlags( pNewNd,
				(F_RECEDIT_FLAG_HIDE_LEVEL | F_RECEDIT_FLAG_HIDE_TAG |
				F_RECEDIT_FLAG_READ_ONLY));
			break;
		}

		/*
		View a specific entry
		*/

		case FKB_ENTER:
		{
			viewRflInspectEntry( pRecEditor);
			break;
		}

		case 'h':
		case 'H':
			viewRflShowHeader( pRecEditor);
			break;

		case '0':
		case 'o':
			f_strcpy( szResponse, gv_szRflPath);
			pRecEditor->requestInput(
				"Log File Name",
				szResponse, sizeof( szResponse), &uiTermChar);
			if( uiTermChar == FKB_ESCAPE || !szResponse [0])
			{
				break;
			}
			
			if (RC_BAD( rc = rflOpenNewFile( pRecEditor, szResponse, TRUE,
									&tmpPool, &pTmpNd)))
			{
				pRecEditor->displayMessage( "Unable to open file", rc,
					NULL, FLM_RED, FLM_WHITE);
			}
			break;

		case 'N':
		case 'n':
			if (RC_BAD( rc = rflOpenNewFile( pRecEditor, NULL, TRUE,
									&tmpPool, &pTmpNd)))
			{
				pRecEditor->displayMessage( "Unable to open file", rc,
					NULL, FLM_RED, FLM_WHITE);
			}
			break;

		case 'P':
		case 'p':
			if (RC_BAD( rc = rflOpenNewFile( pRecEditor, NULL, FALSE,
									&tmpPool, &pTmpNd)))
			{
				pRecEditor->displayMessage( "Unable to open file", rc,
					NULL, FLM_RED, FLM_WHITE);
			}
			break;


		/*
		Goto a specific offset
		*/

		case 'G':
		case 'g':
		{
			szResponse [0] = '\0';
			pRecEditor->requestInput(
				"Offset",
				szResponse, sizeof( szResponse), &uiTermChar);

			if( uiTermChar == FKB_ESCAPE)
			{
				break;
			}
			
			if( (uiSrcLen = (FLMUINT)f_strlen( szResponse)) == 0)
			{
				uiOffset = 0;
			}
			else
			{
				if( RC_BAD( rc = pRecEditor->getNumber( szResponse, &uiOffset, NULL)))
				{
					pRecEditor->displayMessage( "Invalid offset", rc,
						NULL, FLM_RED, FLM_WHITE);
					break;
				}

				RflPositionToNode( uiOffset, FALSE, &tmpPool, &pTmpNd);
				if( pTmpNd)
				{
					pRecEditor->setTree( pTmpNd, &pNewNd);
					pRecEditor->setControlFlags( pNewNd,
						(F_RECEDIT_FLAG_HIDE_LEVEL | F_RECEDIT_FLAG_HIDE_TAG |
						F_RECEDIT_FLAG_READ_ONLY));
				}
			}
			break;
		}

		/*
		Find something in the RFL log.
		*/

		case FKB_F1:
		case FKB_F3:
			gv_bSrchForward = TRUE;
			bSkipCurrent = TRUE;
			goto Do_Search;
		case FKB_F2:
			gv_bSrchForward = FALSE;
			bSkipCurrent = TRUE;
			goto Do_Search;
		case 'F':
		case 'f':
		case 's':
		case 'S':
		{
			if (RC_BAD( rc = getSearchCriteria( pRecEditor,
										&gv_SrchCriteria, &gv_bSrchForward)))
			{
				break;
			}
			bSkipCurrent = FALSE;
Do_Search:
			if (RC_BAD( rc = pRecEditor->createStatusWindow(
				" Searching ... (press ESC to interrupt) ",
				FLM_GREEN, FLM_WHITE, NULL, NULL, &pWindow)))
			{
				goto Exit;
			}

			FTXWinOpen( pWindow);
			pLastNd = NULL;
			pCurNd = pFirstNd = pRecEditor->getCurrentNode();

			// See if we have a match in our current tree.

			for (;;)
			{
				if (!pCurNd)
				{
					break;
				}
				pPacket = (RFL_PACKET *)GedValPtr( pCurNd);
				if (rflPassesCriteria( pPacket, &gv_SrchCriteria))
				{
					if (!bSkipCurrent || pCurNd != pFirstNd)
					{
						pRecEditor->setCurrentNode( pCurNd);
						gv_bDoRefresh = FALSE;
						break;
					}
				}
				if (pWindow)
				{
					FTXWinSetCursorPos( pWindow, 0, 1);
					FTXWinPrintf( pWindow,
						"File Offset : %08X", (unsigned)pPacket->uiFileOffset);
					FTXWinClearToEOL( pWindow);
					FTXWinSetCursorPos( pWindow, 0, 2);
					FTXWinPrintf( pWindow,
						"Trans ID    : %u", (unsigned)pPacket->uiTransID);
					FTXWinClearToEOL( pWindow);

					// Test for the escape key

					if (RC_OK( FTXWinTestKB( pWindow)))
					{
						FLMUINT	uiChar;
						FTXWinInputChar( pWindow, &uiChar);
						if( uiChar == FKB_ESCAPE)
						{
							goto Exit;
						}
					}
				}

				pLastNd = pCurNd;
				if (gv_bSrchForward)
				{
					pCurNd = pRecEditor->getNextNode( pCurNd, FALSE);
				}
				else
				{
					pCurNd = pRecEditor->getPrevNode( pCurNd, FALSE);
				}
			}

			// If no match in the current tree, continue searching
			// until we find one.

			if (pCurNd)
			{
				break;
			}
			pCurNd = pLastNd;

			// If we do not have an EOF, determine one.  We don't
			// want to continue our search past this point.

			if (!gv_ui64RflEof)
			{
				if (RC_BAD( rc = viewRflGetEOF()))
				{
					goto Exit;
				}
			}

			for (;;)
			{
				tmpPool.poolReset( NULL);
				if (gv_bSrchForward)
				{
					if (RC_BAD( rc = RflGetNextNode( pLastNd, FALSE,
												&tmpPool, &pCurNd, TRUE)))
					{
						goto Exit;
					}
				}
				else
				{
					if (RC_BAD( rc = RflGetPrevNode( pLastNd, FALSE,
												&tmpPool, &pCurNd)))
					{
						goto Exit;
					}
				}
				if (!pCurNd)
				{

					// See if we can go to the next or previous file.

					if (gv_SrchCriteria.uiMultiFileSearch == 1)
					{
						if (RC_BAD( rc = rflOpenNewFile( pRecEditor, NULL,
													gv_bSrchForward,
													&tmpPool, &pCurNd)))
						{
							if (rc == FERR_IO_PATH_NOT_FOUND)
							{
								rc = FERR_OK;
								break;
							}
							goto Exit;
						}
						if (pWindow)
						{
							FTXWinSetCursorPos( pWindow, 0, 3);
							FTXWinPrintf( pWindow,
								"File Name   : %s", gv_szRflPath);
							FTXWinClearToEOL( pWindow);
						}
					}
					else
					{
						break;
					}
				}
				pPacket = (RFL_PACKET *)GedValPtr( pCurNd);
				if (rflPassesCriteria( pPacket, &gv_SrchCriteria))
				{
					pRecEditor->setTree( NULL);
					pRecEditor->setTree( pCurNd, &pNewNd);
					pRecEditor->setControlFlags( pNewNd,
							(F_RECEDIT_FLAG_HIDE_LEVEL | F_RECEDIT_FLAG_HIDE_TAG |
							F_RECEDIT_FLAG_READ_ONLY));
					break;
				}
				if (pWindow)
				{
					FTXWinSetCursorPos( pWindow, 0, 1);
					FTXWinPrintf( pWindow,
						"File Offset : %08X", (unsigned)pPacket->uiFileOffset);
					FTXWinClearToEOL( pWindow);
					FTXWinSetCursorPos( pWindow, 0, 2);
					FTXWinPrintf( pWindow,
						"Trans ID    : %u", (unsigned)pPacket->uiTransID);
					FTXWinClearToEOL( pWindow);

					// Test for the escape key

					if (RC_OK( FTXWinTestKB( pWindow)))
					{
						FLMUINT	uiChar;
						FTXWinInputChar( pWindow, &uiChar);
						if( uiChar == FKB_ESCAPE)
						{
							goto Exit;
						}
					}
				}
				tmp2Pool.poolReset( NULL);
				if ((pLastNd = GedCopy( &tmp2Pool, 1, pCurNd)) == NULL)
				{
					rc = RC_SET( FERR_MEM);
					goto Exit;
				}
			}
			if (pWindow)
			{
				FTXWinFree( &pWindow);
			}
			break;
		}

		case FKB_ALT_Q:
		case FKB_ESCAPE:
		{
			*puiKeyOut = FKB_ESCAPE;
			break;
		}
	}

Exit:
	if (pWindow)
	{
		FTXWinFree( &pWindow);
	}
	tmpPool.poolFree();
	tmp2Pool.poolFree();
	return( rc);
}


/****************************************************************************
Name:	viewRflHelpHook
Desc:	
*****************************************************************************/
RCODE viewRflMainHelpHook(
	F_RecEditor *		pRecEditor,
	F_RecEditor *		pHelpEditor,
	F_Pool *				pPool,
	void *				UserData,
	NODE **				ppRootNd)
{
	NODE *	pNewTree = NULL;
	RCODE		rc = FERR_OK;

	F_UNREFERENCED_PARM( pRecEditor);
	F_UNREFERENCED_PARM( pHelpEditor);
	F_UNREFERENCED_PARM( UserData);

	if( (pNewTree = GedNodeMake( pPool, 1, &rc)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = GedPutNATIVE( pPool, pNewTree,
		"RFL Viewer Keyboard Commands")))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		(FLMUINT)'?', (void *)"?               Help (this screen)",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		FKB_UP, (void *)"UP              Move cursor up",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		FKB_DOWN, (void *)"DOWN            Move cursor down",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		FKB_PGUP, (void *)"PG UP           Page up",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		FKB_PGDN, (void *)"PG DOWN         Page down",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		FKB_HOME, (void *)"HOME            Position to beginning of file",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		FKB_END, (void *)"END             Position to end of file",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		'S', (void *)"S or F          Search for (find) a packet",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		'O', (void *)"O               Open a new log file",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		'N', (void *)"N               Go to next log file",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		'P', (void *)"P               Go to previous log file",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		FKB_F1, (void *)"F1 or F3        Search forward (using last criteria entered)",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		FKB_F2, (void *)"F2              Search backward (using last criteria entered)",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		'G', (void *)"G               Goto an offset in the file",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		'H', (void *)"H               Show RFL Header",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		FKB_ESCAPE, (void *)"ESC, ALT-Q      Exit",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	*ppRootNd = pNewTree;

Exit:

	return( rc);
}


/****************************************************************************
Name:	viewRflMainEventHook
Desc:	
*****************************************************************************/
RCODE viewRflMainEventHook(
	F_RecEditor *		pRecEditor,
	eEventType			eEventType,
	void *				EventData,
	void *				UserData)
{
	F_Pool				tmpPool;
	NODE *				pTmpNd;
	RCODE					rc = FERR_OK;

	F_UNREFERENCED_PARM( UserData);

	tmpPool.poolInit( 4096);

	switch( eEventType)
	{
		case F_RECEDIT_EVENT_IEDIT:
		{
			NODE *		pNewNd;

			if( RC_BAD( rc = RflGetNextNode( NULL, FALSE, &tmpPool, 
										&pTmpNd)))
			{
				goto Exit;
			}

			pRecEditor->setTree( pTmpNd, &pNewNd);
			pRecEditor->setControlFlags( pNewNd,
				(F_RECEDIT_FLAG_HIDE_LEVEL | F_RECEDIT_FLAG_HIDE_TAG |
				F_RECEDIT_FLAG_READ_ONLY));
			break;
		}

		case F_RECEDIT_EVENT_REFRESH:
		{
			NODE *		pCurrentNd;
			NODE *		pNewTree;
			NODE *		pTopNd;
			NODE *		pBottomNd;
			FLMUINT		uiPriorCount;
			FLMUINT		uiNextCount;
			FLMUINT		uiCursorRow;

			if (!gv_bDoRefresh)
			{
				gv_bDoRefresh = TRUE;
				break;
			}

			/*
			Re-size the tree
			*/

			pCurrentNd = pRecEditor->getCurrentNode();
			pBottomNd = pTopNd = pCurrentNd;

			uiPriorCount = 0;
			pTmpNd = pTopNd;
			while( pTmpNd && uiPriorCount < pRecEditor->getNumRows())
			{
				pTmpNd = pRecEditor->getPrevNode( pTmpNd, FALSE);
				if( pTmpNd)
				{
					pTopNd = pTmpNd;
					uiPriorCount++;
				}
			}

			uiNextCount = 0;
			pTmpNd = pBottomNd;
			while( pTmpNd && uiNextCount < pRecEditor->getNumRows())
			{
				pBottomNd = pTmpNd;
				pTmpNd = pRecEditor->getNextNode( pTmpNd, FALSE);
				if( pTmpNd)
				{
					uiNextCount++;
				}
			}

			/*
			Clip the rest of the forest
			*/

			pTmpNd = GedSibNext( pBottomNd);
			if( pTmpNd)
			{
				pTmpNd->prior->next = NULL;
			}

			/*
			Reset the tree to the new "pruned" version
			*/

			if (pTopNd)
			{
				if( RC_BAD( rc = pRecEditor->copyBuffer( &tmpPool,
					pTopNd, &pNewTree)))
				{
					goto Exit;
				}
			}
			else
			{
				pNewTree = NULL;
			}

			/*
			Re-position the cursor
			*/

			uiCursorRow = pRecEditor->getCursorRow();
			pRecEditor->setTree( pNewTree, &pTmpNd);
			pNewTree = pTmpNd;

			if( uiPriorCount > uiCursorRow)
			{
				uiPriorCount -= uiCursorRow;
				while( uiPriorCount)
				{
					pTmpNd = pRecEditor->getNextNode( pTmpNd);
					if( pTmpNd)
					{
						pNewTree = pTmpNd;
					}
					uiPriorCount--;
				}
				pRecEditor->setCurrentNode( pNewTree);
				pRecEditor->setCurrentAtTop();
			}
			
			pTmpNd = pNewTree;
			while( uiCursorRow)
			{
				pTmpNd = pRecEditor->getNextNode( pTmpNd);
				if( pTmpNd)
				{
					pNewTree = pTmpNd;
				}
				else
				{
					break;
				}
				uiCursorRow--;
			}

			pRecEditor->setCurrentNode( pNewTree);
			break;
		}

		case F_RECEDIT_EVENT_GETDISPVAL:
		{
			DBE_VAL_INFO *		pValInfo = (DBE_VAL_INFO *)EventData;
			NODE *				pNd = pValInfo->pNd;

			RflFormatPacket( GedValPtr( pNd), (char *)pValInfo->pucBuf);
			break;
		}

		case F_RECEDIT_EVENT_GETNEXTNODE:
		{
			DBE_NODE_INFO *		pNodeInfo = (DBE_NODE_INFO *)EventData;

			pNodeInfo->pNd = pRecEditor->getNextNode( pNodeInfo->pCurNd, FALSE);
			if( !pNodeInfo->pNd)
			{
				if( RC_BAD( rc = RflGetNextNode( pNodeInfo->pCurNd,
					FALSE, &tmpPool, &pTmpNd)))
				{
					goto Exit;
				}

				if( pTmpNd)
				{
					pRecEditor->appendTree( pTmpNd, &pNodeInfo->pNd);
					pRecEditor->setControlFlags( pNodeInfo->pNd,
						(F_RECEDIT_FLAG_HIDE_LEVEL | F_RECEDIT_FLAG_HIDE_TAG |
						F_RECEDIT_FLAG_READ_ONLY));
				}
			}
			pNodeInfo->bUseNd = TRUE;
			break;
		}

		case F_RECEDIT_EVENT_GETPREVNODE:
		{
			DBE_NODE_INFO *		pNodeInfo = (DBE_NODE_INFO *)EventData;

			pNodeInfo->pNd = pRecEditor->getPrevNode( pNodeInfo->pCurNd, FALSE);
			if( !pNodeInfo->pNd)
			{
				if( RC_BAD( rc = RflGetPrevNode( pNodeInfo->pCurNd, FALSE,
					&tmpPool, &pTmpNd)))
				{
					goto Exit;
				}

				if( pTmpNd)
				{
					pRecEditor->insertRecord( pTmpNd, &pNodeInfo->pNd);
					pRecEditor->setControlFlags( pNodeInfo->pNd,
						(F_RECEDIT_FLAG_HIDE_LEVEL | F_RECEDIT_FLAG_HIDE_TAG |
						F_RECEDIT_FLAG_READ_ONLY));
				}
			}
			pNodeInfo->bUseNd = TRUE;
			break;
		}

		default:
		{
			break;
		}
	}

Exit:

	tmpPool.poolFree();
	return( rc);
}


/****************************************************************************
Name:	viewRflInspectEntry
Desc:	
*****************************************************************************/
RCODE viewRflInspectEntry(
	F_RecEditor *		pParentEditor)
{
	F_RecEditor *		pRecEditor;
	NODE *				pExpandNd;
	F_Pool				tmpPool;
	RCODE					rc = FERR_OK;

	tmpPool.poolInit( 1024);

	if( (pRecEditor = f_new F_RecEditor) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( pRecEditor->Setup( pParentEditor->getScreen())))
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	pRecEditor->setTree( NULL);
	pRecEditor->setShutdown( &gv_bShutdown);
	pRecEditor->setDisplayHook( viewRflInspectDispHook, 0);
	pRecEditor->setEventHook( viewRflInspectEventHook, (void *)0);
	pRecEditor->setKeyHook( viewRflInspectKeyHook, 0);
	pRecEditor->setTitle( "Log Entry");

	if( RC_BAD( rc = RflExpandPacket( pParentEditor->getCurrentNode(), &tmpPool,
								&pExpandNd)))
	{
		goto Exit;
	}

	pRecEditor->setTree( pExpandNd);
	pRecEditor->interactiveEdit( 0, 1);

Exit:

	if( pRecEditor)
	{
		pRecEditor->Release();
	}

	tmpPool.poolFree();
	return( rc);
}


/****************************************************************************
Name:	viewRflInspectDispHook
Desc:
*****************************************************************************/
RCODE viewRflInspectDispHook(
	F_RecEditor *			pRecEditor,
	NODE *					pNd,
	void *					UserData,
	DBE_DISP_COLUMN *		pDispVals,
	FLMUINT *				puiNumVals)
{
	FLMUINT		uiFlags;
	FLMUINT		uiCol = 0;
	FLMUINT		uiOffset;
	FLMUINT		uiTag = 0;
	FLMUINT		uiTmp;
	FLMBOOL		bBadField = FALSE;
	RCODE			rc = FERR_OK;

	F_UNREFERENCED_PARM( UserData);

	if( !pNd)
	{
		goto Exit;
	}

	uiTag = GedTagNum( pNd);
	pRecEditor->getControlFlags( pNd, &uiFlags);
	if( !pRecEditor->isSystemNode( pNd))
	{
		/*
		Output the record source
		*/

		uiOffset = 0;
		GedGetRecSource( pNd, NULL, NULL, &uiOffset);

		if( uiOffset)
		{
			f_sprintf( (char *)pDispVals[ *puiNumVals].pucString,
				"%8.8X", (unsigned)uiOffset);
			pDispVals[ *puiNumVals].uiCol = uiCol;
			pDispVals[ *puiNumVals].foreground = FLM_WHITE;
			pDispVals[ *puiNumVals].background = FLM_BLUE;
			(*puiNumVals)++;
		}
		uiCol += 10;

		/*
		Output the level
		*/

		f_sprintf( (char *)pDispVals[ *puiNumVals].pucString,
			"%u", (unsigned)GedNodeLevel( pNd));
		pDispVals[ *puiNumVals].uiCol = uiCol + (GedNodeLevel( pNd) * 2);
		pDispVals[ *puiNumVals].foreground = FLM_WHITE;
		pDispVals[ *puiNumVals].background = FLM_BLUE;
		uiCol += (FLMUINT)(f_strlen( pDispVals[ *puiNumVals].pucString) +
			(GedNodeLevel( pNd) * 2) + 1);
		(*puiNumVals)++;

		/*
		Output the tag
		*/

		if( RC_BAD( pRecEditor->getDictionaryName(
			uiTag, pDispVals[ *puiNumVals].pucString)))
		{
			f_sprintf( (char *)pDispVals[ *puiNumVals].pucString,
				"TAG_%u", (unsigned)uiTag);
		}

		/*
		Determine if the field is bad
		*/

		switch( uiTag)
		{
			case RFL_TAG_NUM_FIELD:
			case RFL_TYPE_FIELD:
			case RFL_LEVEL_FIELD:
			case RFL_DATA_LEN_FIELD:
			case RFL_DATA_FIELD:
			{
				NODE *		pParentNd = GedParent( pNd);
				FLMUINT		uiParentTag;

				if( pParentNd)
				{
					uiParentTag = GedTagNum( pParentNd);
					if( uiParentTag == RFL_INSERT_FLD_FIELD ||
						uiParentTag == RFL_MODIFY_FLD_FIELD ||
						uiParentTag == RFL_DELETE_FLD_FIELD)
					{
						break;
					}
				}
					
				bBadField = TRUE;
				break;
			}

			case RFL_PACKET_CHECKSUM_VALID_FIELD:
			{
				if( RC_OK( GedGetUINT( pNd, &uiTmp)))
				{
					if( !uiTmp)
					{
						bBadField = TRUE;
					}
				}
				break;
			}
		}

		if( bBadField)
		{
			pDispVals[ *puiNumVals].foreground = FLM_RED;
			pDispVals[ *puiNumVals].background = FLM_WHITE;
		}
		else
		{
#ifdef FLM_WIN
			pDispVals[ *puiNumVals].foreground = FLM_LIGHTGREEN;
#else
			pDispVals[ *puiNumVals].foreground = FLM_GREEN;
#endif
			pDispVals[ *puiNumVals].background = FLM_BLUE;
		}

		pDispVals[ *puiNumVals].uiCol = uiCol;
		uiCol += (FLMUINT)(f_strlen( pDispVals[ *puiNumVals].pucString) + 1);
		(*puiNumVals)++;

		/*
		Output the display value
		*/
		
		switch( uiTag)
		{
			case RFL_INSERT_FLD_FIELD:
			case RFL_MODIFY_FLD_FIELD:
			case RFL_DELETE_FLD_FIELD:
			{
				/*
				Don't output the value
				*/

				break;
			}
			default:
			{
				if( RC_BAD( rc = pRecEditor->getDisplayValue( pNd,
					F_RECEDIT_DEFAULT_TYPE, pDispVals[ *puiNumVals].pucString,
					sizeof( pDispVals[ *puiNumVals].pucString))))
				{
					goto Exit;
				}

				pDispVals[ *puiNumVals].uiCol = uiCol;
				pDispVals[ *puiNumVals].foreground = FLM_YELLOW;
				pDispVals[ *puiNumVals].background = FLM_BLUE;
				uiCol += (FLMUINT)(f_strlen( pDispVals[ *puiNumVals].pucString) + 1);
				(*puiNumVals)++;
			}
		}
	}

Exit:

	return( rc);
}


/****************************************************************************
Name:	viewRflInspectEventHook
Desc:	
*****************************************************************************/
RCODE viewRflInspectEventHook(
	F_RecEditor *		pRecEditor,
	eEventType			eEventType,
	void *				EventData,
	void *				UserData)
{
	RCODE					rc = FERR_OK;

	F_UNREFERENCED_PARM( UserData);
	F_UNREFERENCED_PARM( pRecEditor);

	switch( eEventType)
	{
		case F_RECEDIT_EVENT_NAME_TABLE:
		{
			DBE_NAME_TABLE_INFO *		pNameTableInfo = (DBE_NAME_TABLE_INFO *)EventData;

			pNameTableInfo->pNameTable = gv_pNameTable;
			pNameTableInfo->bInitialized = TRUE;
			break;
		}

		default:
		{
			break;
		}
	}

	return( rc);
}


/****************************************************************************
Name:	viewRflInspectKeyHook
Desc:	
*****************************************************************************/
RCODE viewRflInspectKeyHook(
	F_RecEditor *		pRecEditor,
	NODE *				pCurNd,
	FLMUINT				uiKeyIn,
	void *				UserData,
	FLMUINT *			puiKeyOut)
{
	RCODE			rc = FERR_OK;

	F_UNREFERENCED_PARM( UserData);
	F_UNREFERENCED_PARM( pRecEditor);
	F_UNREFERENCED_PARM( pCurNd);

	if( puiKeyOut)
	{
		*puiKeyOut = 0;
	}

	switch( uiKeyIn)
	{
		case FKB_DOWN:
		case FKB_UP:
		case FKB_PGDN:
		case FKB_PGUP:
		case FKB_ESCAPE:
		case FKB_ENTER:
		case FKB_END:
		case FKB_HOME:
		case '?':
		{
			*puiKeyOut = uiKeyIn;
			break;
		}
	}

	return( rc);
}


/****************************************************************************
Name:	viewRflNameTableInit
Desc:	
*****************************************************************************/
RCODE viewRflNameTableInit(
	F_NameTable **		ppNameTable)
{
	FLMBOOL				bOpenDb = FALSE;
	char *				pucTmp;
	char					szIoDbPath [F_PATH_MAX_SIZE];
	char					szFileName[ F_PATH_MAX_SIZE];
	HFDB					hDb = HFDB_NULL;
	F_NameTable *		pNameTable = NULL;
	FLMUINT				uiTagNum;
	RCODE					rc = FERR_OK;

	// Try to open the database

	if( RC_BAD( f_pathReduce( gv_szRflPath, szIoDbPath, szFileName)))
	{
		goto Exit;
	}

	pucTmp = f_strchr( (const char *)szFileName, '.');
	if( f_stricmp( pucTmp, ".log") == 0)
	{
		*pucTmp = 0;
		if( f_strlen( szFileName) > 5)
		{
			pucTmp = &szFileName[ f_strlen( szFileName) - 5];
			pucTmp[ 0] = '.';
			pucTmp[ 1] = 'd';
			pucTmp[ 2] = 'b';
			pucTmp[ 3] = '\0';

			if (RC_BAD( rc = f_pathAppend( szIoDbPath, szFileName)))
			{
				goto Exit;
			}
			bOpenDb = TRUE;
		}
	}

	if( bOpenDb)
	{
		if( RC_OK( FlmConfig( FLM_MAX_UNUSED_TIME, (void *)0, (void *)0)))
		{
			FlmDbOpen( szIoDbPath, NULL, NULL, // VISIT
				FO_DONT_REDO_LOG, NULL, &hDb);
		}
	}

	if( (pNameTable = f_new F_NameTable) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pNameTable->setupFromDb( hDb)))
	{
		goto Exit;
	}

	// Build the name table

	uiTagNum = 0;
	while (gv_szTagNames [uiTagNum])
	{
		if( RC_BAD( rc = pNameTable->addTag( NULL, gv_szTagNames [uiTagNum],
			uiTagNum + 32769, FLM_FIELD_TAG, 0)))
		{
			flmAssert( 0);
			goto Exit;
		}
		uiTagNum++;
	}

	*ppNameTable = pNameTable;
	pNameTable = NULL;

Exit:

	if( pNameTable)
	{
		pNameTable->Release();
	}

	if( hDb != HFDB_NULL)
	{
		FlmDbClose( &hDb);
	}

	return( rc);
}

