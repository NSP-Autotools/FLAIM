//-------------------------------------------------------------------------
// Desc: Search capabilities in the database viewer utility.
// Tabs: 3
//
// Copyright (c) 1992-2007 Novell, Inc. All Rights Reserved.
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

#include "view.h"

/********************************************************************
Desc: ?
*********************************************************************/
FLMINT ViewGetKey( void)
{
	RCODE				rc = FERR_OK;
	FlmRecord *		pKey = NULL;
	void *			pvFld;
	char				Prompt [80];
	FLMUINT			Num;
	FLMUINT			ValEntered;
	FLMUINT			Len;
	char				TempBuf [80];
	FLMUINT			NumFields;
	FLMUINT			i;
	FLMINT			GetOK;
	FLMBYTE			FieldName [80];
	FLMBYTE			FieldType [16];
	FLMINT			KeyEntered = FALSE;
	FLMBYTE			LFH[ LFH_SIZE];
	FLMUINT			FileOffset;
	LFILE *			pLFile = NULL;
	IXD *				pIxd;
	IFD *				pIfd;
	FLMUINT			uiRootBlkAddress;
	FLMBOOL			bTruncated;

	if (!gv_bViewHdrRead)
		ViewReadHdr();

	/* See if we can get dictionary information. */

	ViewGetDictInfo();
	if (gv_bViewHaveDictInfo)
	{

		/* Find the logical file */

		if ((RC_BAD( fdictGetIndex( ((FDB *)gv_hViewDb)->pDict,
				((FDB *)gv_hViewDb)->pFile->bInLimitedMode,
				gv_uiViewSearchLfNum, &pLFile, NULL))) &&
				(RC_BAD( fdictGetContainer( ((FDB *)gv_hViewDb)->pDict, 
				gv_uiViewSearchLfNum, &pLFile))))
		{
			pLFile = NULL;
		}
	}

	/* See if we have a valid logical file */

	if ((gv_uiViewSearchLfNum == FLM_DATA_CONTAINER) ||
			(gv_uiViewSearchLfNum == FLM_DICT_CONTAINER) ||
			(pLFile))
	{

		/* Get the LFH information for the logical file */

		if (!ViewGetLFH( gv_uiViewSearchLfNum, LFH, &FileOffset))
		{
			ViewShowError( "Could not get LFH for logical file");
			return( FALSE);
		}
		uiRootBlkAddress = FB2UD( &LFH [LFH_ROOT_BLK_OFFSET]);

		if (uiRootBlkAddress == 0xFFFFFFFF)
		{
			ViewShowError( "Logical file is empty");
			return( FALSE);
		}
	}
	else
	{
		ViewShowError( "Logical file not defined");
		return( FALSE);
	}

	if ((gv_uiViewSearchLfNum == FLM_DATA_CONTAINER) ||
		 (gv_uiViewSearchLfNum == FLM_DICT_CONTAINER) ||
		 ((pLFile) &&
		  (pLFile->uiLfType == LF_CONTAINER)))
	{
		if (gv_uiViewSearchLfNum == FLM_DICT_CONTAINER)
			f_strcpy( TempBuf, "Enter Dictionary Record Number: ");
		else if (gv_uiViewSearchLfNum == FLM_DATA_CONTAINER)
			f_strcpy( TempBuf, "Enter Data Container Record Number: ");
		else
			f_sprintf( (char *)TempBuf, 
				"Enter Record Number For Container %u: ", 
				(unsigned)gv_uiViewSearchLfNum);
		if ((!ViewGetNum( TempBuf, &Num, FALSE, 4,
										 0xFFFFFFFF, &ValEntered)) ||
				(!ValEntered))
			return( FALSE);
		f_UINT32ToBigEndian( (FLMUINT32)Num, gv_ucViewSearchKey);
		gv_uiViewSearchKeyLen = 4;
		return( TRUE);
	}

	/* At this point, we are dealing with an index. */

	if (gv_uiViewSearchLfNum == FLM_DICT_INDEX)
	{
		FLMUINT	 wTagType = 0;
		FLMUINT	 wElmLen;

		while (!wTagType)
		{
			if ((!ViewEditText( "Enter Type:", 
									TempBuf, sizeof( TempBuf), &ValEntered)) ||
				 (!ValEntered))
				return( FALSE);
			else if ((f_stricmp( TempBuf, "F") == 0) ||
						(f_stricmp( TempBuf, "FIELD") == 0))
			{
				wTagType = FLM_FIELD_TAG;
			}
			else if ((f_stricmp( TempBuf, "I") == 0) ||
						(f_stricmp( TempBuf, "INDEX") == 0))
			{
				wTagType = FLM_INDEX_TAG;
			}
			else if ((f_stricmp( TempBuf, "C") == 0) ||
						(f_stricmp( TempBuf, "CONTAINER") == 0))
			{
				wTagType = FLM_CONTAINER_TAG;
			}
			else if ((f_stricmp( TempBuf, "A") == 0) ||
						(f_stricmp( TempBuf, "AREA") == 0))
			{
				wTagType = FLM_AREA_TAG;
			}
			else
			{
				ViewShowError( "Illegal type, must be F)ield, I)ndex, C)ontainer, R)ecord, or A)rea");
				wTagType = 0;
			}
		}
		gv_ucViewSearchKey [0] = KY_CONTEXT_PREFIX;
		f_UINT16ToBigEndian( (FLMUINT16)wTagType, &gv_ucViewSearchKey [1]);
		gv_uiViewSearchKeyLen += KY_CONTEXT_LEN;
		gv_ucViewSearchKey [gv_uiViewSearchKeyLen++] = COMPOUND_MARKER;

		if (!ViewEditText( "Enter Name:", TempBuf, sizeof( TempBuf), &ValEntered))
			return( FALSE);

		/* Collate the name. */

		wElmLen = MAX_KEY_SIZ - gv_uiViewSearchKeyLen;
		if (RC_BAD( rc = KYCollateValue( &gv_ucViewSearchKey [gv_uiViewSearchKeyLen],
									&wElmLen,
									(const FLMBYTE *)TempBuf,
									(FLMUINT)f_strlen( TempBuf), FLM_TEXT_TYPE,
									MAX_KEY_SIZ,
									NULL, NULL,
									gv_ViewHdrInfo.FileHdr.uiDefaultLanguage,
									FALSE, FALSE, FALSE, &bTruncated)))
		{
			ViewShowRCError( "collating name", rc);
			return( FALSE);
		}
		gv_uiViewSearchKeyLen += wElmLen;
		return( TRUE);
	}
	else if (!pLFile)
	{
		ViewShowError( "Cannot get logical file information");
		return( FALSE);
	}
	else if (RC_BAD( fdictGetIndex( ((FDB *)gv_hViewDb)->pDict,
			((FDB *)gv_hViewDb)->pFile->bInLimitedMode,
			gv_uiViewSearchLfNum, &pLFile, &pIxd)))
	{
		ViewShowError( "Cannot get index field information");
		return( FALSE);
	}
	else
	{
		pIfd = pIxd->pFirstIfd;
		NumFields = pIxd->uiNumFlds;

		if (!(pIfd->uiFlags & IFD_COMPOUND))
		{
			NumFields = 1;
		}

		if( (pKey = f_new FlmRecord) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			ViewShowRCError( "creating key", rc);
			goto Exit_False;
		}

		if (RC_BAD( rc = pKey->insertLast( 0, FLM_KEY_TAG,
										FLM_CONTEXT_TYPE, &pvFld)))
		{
			ViewShowRCError( "adding key tag", rc);
			goto Exit_False;
		}

		/* Ask for data for each field and link into key tree */

		i = 0;
		while (i < NumFields)
		{

			/* Get the name of the field and its type */

			f_sprintf( (char *)FieldName, "FIELD %u", (unsigned)pIfd->uiFldNum);
			switch( IFD_GET_FIELD_TYPE( pIfd))
			{
				case FLM_TEXT_TYPE:
					f_strcpy( (char *)FieldType, "TEXT");
					break;
				case FLM_NUMBER_TYPE:
					f_strcpy( (char *)FieldType, "NUMBER");
					break;
				case FLM_BINARY_TYPE:
					f_strcpy( (char *)FieldType, "BINARY");
					break;
				case FLM_CONTEXT_TYPE:
					f_strcpy( (char *)FieldType, "CONTEXT");
					break;
				default:
					f_sprintf( (char *)FieldType, "UNK: %u!",
						(unsigned)IFD_GET_FIELD_TYPE( pIfd));
					break;
			}
			if (pIfd->uiFlags & IFD_OPTIONAL)
				f_sprintf( (char *)Prompt, "%s (%s-OPTIONAL): ", FieldName, FieldType);
			else
				f_sprintf( (char *)Prompt, "%s (%s-REQUIRED): ", FieldName, FieldType);

			switch( IFD_GET_FIELD_TYPE( pIfd))
			{
				case FLM_TEXT_TYPE:
					if (!ViewEditText( Prompt, TempBuf, sizeof( TempBuf),
								&ValEntered))
						goto Exit_False;
					break;
				case FLM_NUMBER_TYPE:
				case FLM_CONTEXT_TYPE:
					if (!ViewGetNum( Prompt, &Num, FALSE, 4, 0xFFFFFFFF,
									&ValEntered))
						goto Exit_False;
					break;
				case FLM_BINARY_TYPE:
					Len = sizeof( TempBuf);
					if (!ViewEditBinary( Prompt, TempBuf, &Len, &ValEntered))
						goto Exit_False;
					break;
			}
			if (!ValEntered)
			{
				i++;
			}
			else
			{
				FLMUINT	uiDataType;

				/* See if the entered data can be converted to the */
				/* correct type */

				uiDataType = IFD_GET_FIELD_TYPE( pIfd);
				if (RC_BAD( rc = pKey->insertLast( 1, pIfd->uiFldNum,
													uiDataType, &pvFld)))
				{
					ViewShowRCError( "creating field", rc);
				}
				else
				{
					switch( IFD_GET_FIELD_TYPE( pIfd))
					{
						case FLM_TEXT_TYPE:
							rc = pKey->setNative( pvFld, TempBuf);
							break;
						case FLM_NUMBER_TYPE:
							rc = pKey->setUINT( pvFld, Num);
							break;
						case FLM_CONTEXT_TYPE:
							rc = pKey->setRecPointer( pvFld, Num);
							break;
						case FLM_BINARY_TYPE:
							rc = pKey->setBinary( pvFld, TempBuf, Len);
							break;
					}
					if (RC_BAD( rc))
					{
						ViewShowRCError( "putting data in field", rc);
					}
				}
				if (RC_OK(rc))
				{
					i++;
					pIfd++;
					KeyEntered = TRUE;
				}
			}
		}

		// If index is on all containers, prompt for container number.

		if (!pIxd->uiContainerNum)
		{
			f_strcpy( Prompt, "CONTAINER: ");
			if (!ViewGetNum( Prompt, &Num, FALSE, sizeof( Num), 0xFFFF,
									&ValEntered))
			{
				goto Exit_False;
			}
			if (ValEntered)
			{
				pKey->setContainerID( Num);
				KeyEntered = TRUE;
			}
		}

		/* Convert the key to binary format */

		if (!KeyEntered)
			goto Exit_False;

		if ((rc = FlmKeyBuild( gv_hViewDb, gv_uiViewSearchLfNum,
									pKey->getContainerID(), pKey, 0,
									gv_ucViewSearchKey, &gv_uiViewSearchKeyLen)) != FERR_OK)
			ViewShowRCError( "building key", rc);
		else
		{
			GetOK = TRUE;
			goto Exit_GetKey;
		}
	}

Exit_False:
	GetOK = FALSE;
Exit_GetKey:
	if (pKey)
	{
		pKey->Release();
	}
	return( GetOK);
}

/********************************************************************
Desc: ?
*********************************************************************/
void ViewSearch(
	void)
{
	FLMBYTE	LFH[ LFH_SIZE];
	FLMUINT	FileOffset;
	FLMUINT	RootBlkAddress;
	BLK_EXP	BlkExp;

	if (!gv_bViewHdrRead)
		ViewReadHdr();

	/* Set flags */

	for( ;;)
	{
		gv_bViewPoppingStack = FALSE;

		/* Get the LFH information for the logical file */

		if (!ViewGetLFH( gv_uiViewSearchLfNum, LFH, &FileOffset))
		{
			ViewShowError( "Could not get LFH for logical file");
			return;
		}
		RootBlkAddress = FB2UD( &LFH [LFH_ROOT_BLK_OFFSET]);
		if (RootBlkAddress == 0xFFFFFFFF)
		{
			ViewShowError( "Logical file is empty");
			return;
		}

		BlkExp.Level = 0xFF;
		BlkExp.Type = 0xFF;
//		BlkExp.Type = BHT_NON_LEAF;
		BlkExp.NextAddr = BlkExp.PrevAddr = 0xFFFFFFFF;
		BlkExp.LfNum = gv_uiViewSearchLfNum;
		gv_bViewEnabled = FALSE;
		gv_bViewSearching = TRUE;
		ViewBlocks( RootBlkAddress, RootBlkAddress, &BlkExp);

		/* Reset Search flag before returning so everything will be back to */
		/* normal. */

		gv_bViewSearching = FALSE;

		/* If the ViewBlocks did not set up for another search, we are */
		/* done, otherwise keep-a-goin */

		if (!gv_bViewPoppingStack)
			break;
	}
}
