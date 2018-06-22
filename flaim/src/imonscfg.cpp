//-------------------------------------------------------------------------
// Desc:	Class for displaying and modifying system configuration information
//			in the monitoring web pages.
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

/****************************************************************************
Desc:	Prints the web page for system configuration parameters.
****************************************************************************/
RCODE F_SysConfigPage::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE					rc = FERR_OK;
	char					szTmp [30];
  	eFlmConfigTypes	eConfigType;

	stdHdr();

	fnPrintf( m_pHRequest, HTML_DOCTYPE);
	fnPrintf( m_pHRequest, "<html><head>\n");
	printStyle();
	fnPrintf( m_pHRequest, "</head>\n");
	fnPrintf( m_pHRequest, "<body>\n");

	printTableStart( "System Configuration", 3);
	
	// Get the Action, if any

	if (RC_BAD(rc = ExtractParameter( uiNumParams, ppszParams, 
									"Action", sizeof( szTmp), szTmp)))
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
	else
	{
		eConfigType = (eFlmConfigTypes)f_atoi( szTmp);

		if (RC_BAD( rc = doConfig( eConfigType, uiNumParams, ppszParams)))
		{
			fnPrintf( m_pHRequest,
				"<br><font color=\"Red\">ERROR %04X DOING CONFIGURATION</font><br><br>\n",
				(unsigned)rc);
		}
	}

	outputParams();

	printTableEnd();

	fnPrintf( m_pHRequest, "</body></html>\n");

Exit:

	fnEmit();

	return( rc);

}


/****************************************************************************
Desc:	Outputs a button that the user can simply press for a certain action
		to occur.  The action is described in pszParamDescription.
****************************************************************************/
void F_SysConfigPage::outputButton(
  	eFlmConfigTypes	eConfigType,
	const char *		pszParamAction,
	FLMUINT				uiValue1,
	FLMUINT				uiValue2)
{

	beginRow();
	beginInputForm( eConfigType);

	// Output value1 and value 2 as hidden values to be returned.

	fnPrintf( m_pHRequest,
		"<input name=\"Value1\" type=\"hidden\" value=\"%u\">\n"
		"<input name=\"Value2\" type=\"hidden\" value=\"%u\">\n",
		(unsigned)uiValue1, (unsigned)uiValue2);

	addSubmitButton( pszParamAction);
	endInputForm();
	endRow();
}

/****************************************************************************
Desc:	Outputs a UINT value.
****************************************************************************/
void F_SysConfigPage::outputUINT(
  	eFlmConfigTypes	eConfigType,
	const char *		pszParamDescription,
	FLMBOOL				bParamIsSettable,
	FLMBOOL				bParamIsGettable,
	FLMUINT				uiDefaultValue)
{
	FLMUINT		uiValue;
	char			szValue [40];

	beginRow();

	fnPrintf( m_pHRequest, TD_s, pszParamDescription);

	if (bParamIsGettable)
	{
		RCODE	rc;

		if (RC_BAD( rc = FlmGetConfig( eConfigType, (void *)&uiValue)))
		{
			f_sprintf( szValue, "Error %04X", (unsigned)rc);
		}
		else
		{
			f_sprintf( szValue, "%u", (unsigned)uiValue);
		}
	}
	else
	{
		f_sprintf( szValue, "%u", (unsigned)uiDefaultValue);
	}

	if (!bParamIsSettable)
	{
		fnPrintf( m_pHRequest, TD_s, szValue);
	}
	else
	{
		beginInputForm( eConfigType);
		addStrInputField( eConfigType, 10, szValue);

		// Need a submit button for some browsers.

		addSubmitButton( "Submit");
		endInputForm();
	}
	endRow();
}

/****************************************************************************
Desc:	Outputs a Boolean value.
****************************************************************************/
void F_SysConfigPage::outputBOOL(
  	eFlmConfigTypes	eConfigType,
	const char *		pszParamDescription,
	const char *		pszOnState,
	const char *		pszOffState,
	const char *		pszTurnOnAction,
	const char *		pszTurnOffAction)
{
	RCODE			rc;
	FLMBOOL		bValue;

	beginRow();

	fnPrintf( m_pHRequest, TD_s, pszParamDescription);

	if (RC_BAD( rc = FlmGetConfig( eConfigType, (void *)&bValue)))
	{
		fnPrintf( m_pHRequest, "<TD>Error %04X</TD>\n", (unsigned)rc);
		bValue = FALSE;
	}
	else
	{
		fnPrintf( m_pHRequest, TD_s, (char *)(bValue ? pszOnState : pszOffState));
	}

	beginInputForm( eConfigType);

	// Add a hidden toggle parameter to be returned.

	fnPrintf( m_pHRequest,
		"<input name=\"Toggle\" type=\"hidden\" value=\"%s\">\n",
		(char *)(bValue ? (char *)"OFF" : (char *)"ON"));

	addSubmitButton( (char *)(bValue ? pszTurnOffAction : pszTurnOnAction));
	endInputForm();
	endRow();
}

/****************************************************************************
Desc:	Outputs a string value.
****************************************************************************/
void F_SysConfigPage::outputString(
  	eFlmConfigTypes	eConfigType,
	const char *		pszParamDescription,
	FLMUINT				uiMaxStrLen,
	FLMBOOL				bParamIsSettable,
	FLMBOOL				bParamIsGettable,
	const char *		pszDefaultValue)
{
	RCODE				rc = FERR_OK;
	char *			pszValue = NULL;
	char				szErr[ 40];

	beginRow();

	fnPrintf( m_pHRequest, TD_s, pszParamDescription);

	if (RC_BAD( rc = f_alloc( uiMaxStrLen + 1, &pszValue)))
	{
		f_sprintf( (char *)szErr, "Error %04X", (unsigned)rc);
		pszValue = &szErr [0];
	}
	else
	{
		if (bParamIsGettable)
		{
			if (RC_BAD( rc = FlmGetConfig( eConfigType, (void *)pszValue)))
			{
				if (rc == FERR_IO_PATH_NOT_FOUND &&
					 eConfigType == FLM_TMPDIR)
				{
					*pszValue = 0;
				}
				else
				{
					f_sprintf( (char *)pszValue, "Error %04X", (unsigned)rc);
				}
			}
		}
		else
		{
			f_strcpy( pszValue, pszDefaultValue);
		}
	}

	if (!bParamIsSettable)
	{
		fnPrintf( m_pHRequest, TD_s, pszValue);
	}
	else
	{

		beginInputForm( eConfigType);
		addStrInputField( eConfigType, uiMaxStrLen, (char *)pszValue);

		// Need a submit button for some browsers.

		addSubmitButton( "Submit");
		endInputForm();
	}

	endRow();

	if (pszValue && pszValue != &szErr [0])
	{
		f_free( &pszValue);
	}
}

/****************************************************************************
Desc:	Prints the web page for system configuration parameters.  This function
		assumes that the table has already been started.
****************************************************************************/
void F_SysConfigPage::outputParams( void)
{
	outputButton( FLM_CLOSE_UNUSED_FILES, "Close unused file desc, free unused items");

	outputButton( FLM_CLOSE_ALL_FILES, "Close ALL file descriptors");

	outputButton( FLM_START_STATS, "Begin Statistics");

	outputButton( FLM_STOP_STATS, "End Statistics");

	outputButton( FLM_RESET_STATS, "Reset Statistics");


	outputUINT( FLM_QUERY_MAX, "Max Queries To Save");

	outputBOOL( FLM_CACHE_CHECK, "Cache Checking");

	outputBOOL( FLM_SCACHE_DEBUG, "Cache debugging");

	outputUINT( FLM_BLOCK_CACHE_PERCENTAGE, "Block Cache Percent");

	outputUINT( FLM_CACHE_LIMIT, "Cache limit (bytes)");

	outputUINT( FLM_CACHE_ADJUST_INTERVAL,
		"Dynamic Cache Adjust Interval (secs.)");

	outputUINT( FLM_CACHE_CLEANUP_INTERVAL,
		"Cache Cleanup Interval (seconds)");

	outputUINT( FLM_OPEN_THRESHOLD, "Maximum open file descriptors");

	outputUINT( FLM_OPEN_FILES, "Currently open file descriptors",
		FALSE, TRUE);

	outputUINT( FLM_MAX_CP_INTERVAL, "Checkpoint Interval (seconds)");

	outputUINT( FLM_MAX_TRANS_SECS, "Read Transaction Timeout (seconds)");

	outputUINT( FLM_MAX_UNUSED_TIME, "Unused Object Timeout (seconds)");

	outputUINT( FLM_UNUSED_CLEANUP_INTERVAL,
						"Unused Object Cleanup Interval (seconds)");

	outputString( FLM_BLOB_EXT, "BLOB Extension", 63);

	outputString( FLM_TMPDIR, "Temporary file directory",
									F_PATH_MAX_SIZE);

	outputString( FLM_CLOSE_FILE, "Force DB Close",
								F_PATH_MAX_SIZE,
								TRUE, FALSE, "nds.db");

	outputString( FLM_KILL_DB_HANDLES, "Kill DB Handles",
									F_PATH_MAX_SIZE + F_PATH_MAX_SIZE + 1,
									TRUE, FALSE, "nds.db");

}

/****************************************************************************
Desc:	Get a value from an input box on the form.
****************************************************************************/
RCODE F_SysConfigPage::getConfigValue(
  	eFlmConfigTypes	eConfigType,
	FLMUINT				uiNumParams,
	const char **		ppszParams,
	FLMUINT				uiValueLen,
	char *				pszValue)
{
	RCODE			rc = FERR_OK;
	char			szName [30];

	f_sprintf( (char *)szName, "U%u", (unsigned)eConfigType);

	if (RC_BAD(rc = ExtractParameter( uiNumParams, ppszParams,
									szName, uiValueLen, pszValue)))
	{
		if (rc == FERR_NOT_FOUND)
		{
			*pszValue = 0;
			rc = FERR_OK;
		}
		goto Exit;
	}

	fcsDecodeHttpString( pszValue);

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Get a value from an input box on the form.
****************************************************************************/
RCODE F_SysConfigPage::getConfigValue(
  	eFlmConfigTypes	eConfigType,
	FLMUINT				uiNumParams,
	const char **		ppszParams,
	char **				ppszValue,
	FLMUINT				uiMaxStrLen)
{
	RCODE				rc = FERR_OK;
	char				szName [30];
	FLMBOOL			bAllocated = FALSE;

	f_sprintf( (char *)szName, "U%u", (unsigned)eConfigType);

	// Allocate enough so that if every character is encoded with a %xx we
	// will have enough to get it.

	if (RC_BAD( rc = f_alloc( uiMaxStrLen * 3 + 1, ppszValue)))
	{
		goto Exit;
	}
	bAllocated = TRUE;

	if (RC_BAD(rc = ExtractParameter( uiNumParams, ppszParams,
									szName, ( uiMaxStrLen * 3 + 1), *ppszValue)))
	{
		if (rc == FERR_NOT_FOUND)
		{
			*(*ppszValue) = 0;
			rc = FERR_OK;
		}
		goto Exit;
	}

	fcsDecodeHttpString( *ppszValue);

Exit:

	if (RC_BAD( rc) && bAllocated)
	{
		f_free( ppszValue);
	}

	return( rc);
}

/****************************************************************************
Desc:	Configures from a button that has been pressed.
****************************************************************************/
RCODE F_SysConfigPage::configButton(
  	eFlmConfigTypes	eConfigType,
	FLMUINT				uiNumParams,
	const char **		ppszParams)
{
	RCODE			rc = FERR_OK;
	char			szTmp [20];
	FLMUINT		uiValue1;
	FLMUINT		uiValue2;

	// Get Value1 and Value2 - these will be in the parameters.

	if (RC_BAD(rc = ExtractParameter( uiNumParams, ppszParams,
									"Value1", sizeof( szTmp), szTmp)))
	{
		goto Exit;
	}

	uiValue1 = f_atoud( szTmp);
	if (RC_BAD(rc = ExtractParameter( uiNumParams, ppszParams,
									"Value2", sizeof( szTmp), szTmp)))
	{
		goto Exit;
	}

	uiValue2 = f_atoud( szTmp);

	// Do the configuration.

	if (RC_BAD( rc = FlmConfig( eConfigType, (void *)uiValue1,
							(void *)uiValue2)))
	{
		goto Exit;
	}
Exit:

	return( rc);

}

/****************************************************************************
Desc:	Configures a UINT value.
****************************************************************************/
RCODE F_SysConfigPage::configUINT(
  	eFlmConfigTypes	eConfigType,
	FLMUINT				uiNumParams,
	const char **		ppszParams)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiValue;
	char			szValue [64];

	// Get the value to configure.

	if (RC_BAD( rc = getConfigValue( eConfigType, uiNumParams, ppszParams,
												sizeof( szValue), szValue)))
	{
		goto Exit;
	}

	uiValue = f_atoud( szValue);

	// Do the configuration.

	if (RC_BAD( rc = FlmConfig( eConfigType, (void *)uiValue, (void *)0)))
	{
		goto Exit;
	}

Exit:

	return( rc);

}

/****************************************************************************
Desc:	Configures a Boolean value.
****************************************************************************/
RCODE F_SysConfigPage::configBOOL(
  	eFlmConfigTypes	eConfigType,
	FLMUINT				uiNumParams,
	const char **		ppszParams)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiValue;
	char			szToggle [20];

	// Get the toggle value - it will be in the parameters.

	if (RC_BAD(rc = ExtractParameter( uiNumParams, ppszParams,
									"Toggle", sizeof( szToggle), szToggle)))
	{
		goto Exit;
	}

	uiValue = (FLMUINT)((f_stricmp( szToggle, "OFF") == 0)
							  ? (FLMUINT)0
							  : (FLMUINT)1);

	// Do the configuration.

	if (RC_BAD( rc = FlmConfig( eConfigType, (void *)uiValue, (void *)0)))
	{
		goto Exit;
	}

Exit:

	return( rc);

}

/****************************************************************************
Desc:	Configures a string value.
****************************************************************************/
RCODE F_SysConfigPage::configString(
  	eFlmConfigTypes	eConfigType,
	FLMUINT				uiNumParams,
	const char **		ppszParams,
	FLMUINT				uiMaxStrLen)
{
	RCODE			rc = FERR_OK;
	char *		pszValue = NULL;

	// Get the value to configure.

	if (RC_BAD( rc = getConfigValue( eConfigType, uiNumParams, ppszParams,
								&pszValue, uiMaxStrLen)))
	{
		pszValue = NULL;
		goto Exit;
	}

	// Do the configuration.

	if (RC_BAD( rc = FlmConfig( eConfigType, (void *)pszValue, (void *)0)))
	{
		goto Exit;
	}

Exit:

	if (pszValue)
	{
		f_free( &pszValue);
	}

	return( rc);

}

/****************************************************************************
Desc:	Performs a FlmConfig call, as requested by the user.
****************************************************************************/
RCODE F_SysConfigPage::doConfig(
  	eFlmConfigTypes	eConfigType,
	FLMUINT				uiNumParams,
	const char **		ppszParams)
{
	RCODE				rc = FERR_OK;
	char *			pszTmp = NULL;
	char *			pszDbName;
	char *			pszPath;
	char *			pszPtr;

	switch (eConfigType)
	{
		case FLM_CLOSE_UNUSED_FILES:
		case FLM_CLOSE_ALL_FILES:
		case FLM_START_STATS:
		case FLM_STOP_STATS:
		case FLM_RESET_STATS:
			rc = configButton( eConfigType, uiNumParams, ppszParams);
			break;
		case FLM_OPEN_THRESHOLD:
		case FLM_CACHE_LIMIT:
		case FLM_MAX_CP_INTERVAL:
		case FLM_MAX_TRANS_SECS:
		case FLM_CACHE_ADJUST_INTERVAL:
		case FLM_CACHE_CLEANUP_INTERVAL:
		case FLM_UNUSED_CLEANUP_INTERVAL:
		case FLM_MAX_UNUSED_TIME:
		case FLM_BLOCK_CACHE_PERCENTAGE:
		case FLM_QUERY_MAX:
			rc = configUINT( eConfigType, uiNumParams, ppszParams);
			break;
		case FLM_SCACHE_DEBUG:
		case FLM_CACHE_CHECK:
			rc = configBOOL( eConfigType, uiNumParams, ppszParams);
			break;
		case FLM_BLOB_EXT:
			rc = configString( eConfigType, uiNumParams, ppszParams, 63);
			break;
		case FLM_TMPDIR:
		case FLM_CLOSE_FILE:
			rc = configString( eConfigType, uiNumParams, ppszParams,
								F_PATH_MAX_SIZE);
			break;
		case FLM_KILL_DB_HANDLES:

			// Get the value to configure.  The string should be a
			// database name, semicolon, path.

			pszPath = pszDbName = NULL;
			if (RC_BAD( rc = getConfigValue( eConfigType, uiNumParams,
										ppszParams, &pszTmp,
										F_PATH_MAX_SIZE + F_PATH_MAX_SIZE + 1)))
			{
				pszTmp = NULL;
				goto Exit;
			}
			pszDbName = pszPtr = pszTmp;
			while (*pszDbName && *pszDbName <= ' ')
			{
				pszDbName++;
			}
			pszPtr = pszDbName;
			if (*pszDbName)
			{
				pszPtr = pszDbName;
				while (*pszPtr && *pszPtr != ';')
				{
					pszPtr++;
				}
				if (*pszPtr == ';')
				{
					*pszPtr = 0;
					pszPath = pszPtr + 1;
					while (*pszPath && *pszPath < ' ')
					{
						pszPath++;
					}
					if (!(*pszPath))
					{
						pszPath = NULL;
					}
				}
			}
			else
			{
				pszDbName = NULL;
				pszPath = NULL;
			}

			// Do the configuration.

			if (RC_BAD( rc = FlmConfig( eConfigType, (void *)pszDbName,
										(void *)pszPath)))
			{
				goto Exit;
			}
			break;
		default:
			rc = RC_SET( FERR_INVALID_PARM);
			goto Exit;
	}

Exit:

	if (pszTmp)
	{
		f_free( &pszTmp);
	}

	return( rc);

}
