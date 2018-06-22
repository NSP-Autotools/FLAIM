//-------------------------------------------------------------------------
// Desc:	Factory class for pages created to do HTTP monitoring.
// Tabs:	3
//
// Copyright (c) 2001-2007 Novell, Inc. All Rights Reserved.
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
 Desc:	Procedure to instantiate a new WebPage object given a text string
 ****************************************************************************/
RCODE F_WebPageFactory::create(
	const char *			pszName,
	F_WebPage **			ppPage,
	HRequest *				pHRequest)
{
	RCODE						rc = FERR_OK;
	void *					pvSession = NULL;
	void *					pvUser = NULL;
	ACQUIRE_SESSION_FN	fnAcquireSession = 
									gv_FlmSysData.HttpConfigParms.fnAcquireSession;
	ACQUIRE_USER_FN		fnAcquireUser =
									gv_FlmSysData.HttpConfigParms.fnAcquireUser;
	
	flmAssert( ppPage);

	// Get the session for this user.
	if (fnAcquireSession)
	{
		if ((pvSession = fnAcquireSession( pHRequest)) == NULL)
		{
			rc = RC_SET( FERR_FAILURE);  // We should expect to succeed here.
			goto Exit;
		}
	}

	// Get the current user ...
	if (fnAcquireUser)
	{
		if ((pvUser = fnAcquireUser( pvSession, pHRequest)) == NULL)
		{
			rc = RC_SET( FERR_FAILURE);
			goto Exit;
		}
	}


	// Are we being asked for the 'home page'?
	if (*pszName == '\0')
	{
		if( (*ppPage= m_fnDefault()) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
	}
	else  // search the registry
	{
		FLMINT iEntryNum = searchRegistry ( pszName);
		if (iEntryNum == -1)
		{
			if ( (*ppPage = m_fnError()) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}
		}
		else
		{
			// Check for a secure page.  Ignore it if we don't have any session info.
			if (pvSession && isSecurePage( iEntryNum))
			{

				// Make sure:
				//	1) Security is enable and not expired. 
				// 2) The user has entered the secure password in the Nav bar.

				if ( isSecureAccessEnabled())
				{

					if ( isSecurePasswordEntered( pvSession))
					{
						if( (*ppPage = m_Registry[iEntryNum].fnCreate()) == NULL)
						{
							rc = RC_SET( FERR_MEM);
							goto Exit;
						}
					}
					else
					{
						// Return an error page
						if ( (*ppPage = m_fnSessionAccess()) == NULL)
						{
							rc = RC_SET( FERR_MEM);
							goto Exit;
						}
					}
				}
				else if ( (*ppPage = m_fnGblAccess()) == NULL)
				{
					rc = RC_SET( FERR_MEM);
					goto Exit;
				}
			}
			else if( (*ppPage = m_Registry[iEntryNum].fnCreate()) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}
		}
	}

Exit:

	if (pvSession)
	{
		gv_FlmSysData.HttpConfigParms.fnReleaseSession( pvSession);
	}

	if (pvUser)
	{
		gv_FlmSysData.HttpConfigParms.fnReleaseUser( pvUser);
	}

	return( rc);
}


/****************************************************************************
 Desc:	Tells the factory that the WebPage object is no longer needed
****************************************************************************/
void F_WebPageFactory::Release(
	F_WebPage **			ppPage)
{
	if (ppPage && *ppPage)
	{
		(*ppPage)->releaseSession();
		(*ppPage)->Release();
		*ppPage = NULL;
	}
}


/****************************************************************************
 Desc:	Takes the entries in the registry and sorts them on the name field
			(We do this because we don't trust programmers to spell or
			alphabetize..:)
****************************************************************************/
void F_WebPageFactory::sortRegistry()
{
	// We're going to use an insertion-sort algorithm here because it's simple
	// and has good performance for stuff that is already sorted or 'mostly'
	// sorted.

	FLMUINT	uiInsertionPoint;
	FLMUINT	uiCurrent;

	// First - how many entries in tmp?
	m_uiNumEntries=0; 
	while( m_Registry[m_uiNumEntries++].fnCreate != NULL)
	{
		;
	}
	m_uiNumEntries--; // The last entry in the array is NULL...

	// Basic algorithm:  As uiCurrent goes from 1 to m_uiNumEntries-1, examine
	// the entries from 0 to uiCurrent-1 and place Nth entry in it's proper place.
	// (We'll use m_Registry[uiNumEntries] as a temporary holding spot.
	// Clever, eh?)
	
	for (uiCurrent = 1; uiCurrent < m_uiNumEntries; uiCurrent++)
	{
		uiInsertionPoint = uiCurrent;
		while( (f_strcmp( m_Registry[uiCurrent].pszName,
							 m_Registry[uiInsertionPoint-1].pszName) < 0) &&
				 (uiInsertionPoint > 0) )
		{
			uiInsertionPoint--;
		}

		if (uiInsertionPoint < uiCurrent)
		{
			// Copy the entry at uiCurrent to temp space...
			f_memcpy( &m_Registry[m_uiNumEntries], &m_Registry[uiCurrent],
						 sizeof( RegistryEntry));
			
			// Move the appropriate entries up
			f_memmove( &m_Registry[uiInsertionPoint + 1],
							&m_Registry[uiInsertionPoint],
							(uiCurrent-uiInsertionPoint) * sizeof( RegistryEntry));

			//Copy the stuff in tmp to its sorted position...
			f_memcpy( &m_Registry[uiInsertionPoint], &m_Registry[m_uiNumEntries],
						 sizeof( RegistryEntry));
		}
	}

	//Reset the entry that we've been using for tmp storage
	f_memset(  &m_Registry[m_uiNumEntries], 0, sizeof( RegistryEntry));
}

/****************************************************************************
 Desc:	Returns the index into m_pRegistry for pszName, -1 if pszName is
			not in m_pRegistry  (Uses a binary search, so make sure the
			registry is in sorted order!)
****************************************************************************/
FLMINT F_WebPageFactory::searchRegistry (
	const char *		pszName)
{
	FLMBOOL			bFound = FALSE;
#define MAX_LEN 	100
	char				szPath[ MAX_LEN];
	char *			pszFirstSlash;
	FLMUINT			uiLow;
	FLMUINT			uiHigh;
	FLMUINT			uiTblSize;
	FLMUINT			uiMid;
	FLMINT			iCmp;

	// We only want the part of the string up to the first '/'.  Anything
	// after that will be handled by the web page itself...
	
	pszFirstSlash = f_strchr( pszName, '/');
	
	if (pszFirstSlash)
	{
		flmAssert( (pszFirstSlash - pszName) < MAX_LEN);
		f_strncpy( szPath, pszName, (pszFirstSlash - pszName));
		szPath[pszFirstSlash-pszName] = '\0';
	}
	else
	{
		flmAssert( f_strlen( pszName) < MAX_LEN);
		f_strcpy( szPath, pszName);
	}

	uiLow = 0;
	uiHigh = uiTblSize = m_uiNumEntries-1;
	for (;;)
	{
		uiMid = (uiLow + uiHigh) / 2;
		iCmp = f_strcmp( szPath, m_Registry[uiMid].pszName);
			
		if (iCmp == 0)
		{
			// Found Match
			bFound = TRUE;
			break;
		}

		// Check if we are done
		if (uiLow >= uiHigh)
		{
			// Done, item not found
			break;
		}

		if (iCmp < 0)
		{
			if (uiMid == 0)
			{
				break;
			}
			uiHigh = uiMid - 1;
		}
		else
		{
			if (uiMid == uiTblSize)
			{
				break;
			}
			uiLow = uiMid + 1;
		}
	}

	return( bFound ? (FLMINT)uiMid : -1);
}


/******************************************************************
Desc:	Function to test the Secure password has been entered for this
		user.  If it has, then it will be storted in the session under
		the name defined by the constant FLM_SECURE_PASSWORD.
*******************************************************************/
FLMBOOL F_WebPageFactory::isSecurePasswordEntered(
	void *		pvSession)
{
	GET_SESSION_VALUE_FN		fnGetSessionValue =
										gv_FlmSysData.HttpConfigParms.fnGetSessionValue;
	char							szData[ 21];
	FLMUINT						uiSize = sizeof( szData) - 1;
	FLMBOOL						bResult = FALSE;

	flmAssert( fnGetSessionValue);
	flmAssert( pvSession);

	if (fnGetSessionValue( pvSession, FLM_SECURE_PASSWORD, (void *)szData,
								  (FLMSIZET *)&uiSize) == 0)
	{
		szData[ uiSize] = '\0';
		bResult = isValidSecurePassword( szData);
	}

	return bResult;
}


/******************************************************************
Desc:	Function to test if the secure password entered matches the
		password stored globally.
*******************************************************************/
FLMBOOL F_WebPageFactory::isValidSecurePassword(
	const char *		pszData)
{
	GET_GBL_VALUE_FN		fnGetGblValue =
									gv_FlmSysData.HttpConfigParms.fnGetGblValue;
	char						szPassword[21];
	FLMUINT					uiSize = sizeof( szPassword) -1;
	FLMBOOL					bResult = FALSE;

	flmAssert( fnGetGblValue);

	if (fnGetGblValue( FLM_SECURE_PASSWORD,
							 szPassword,
							 (FLMSIZET *)&uiSize) == 0)
	{
		szPassword[ uiSize] = '\0';
		if (f_strcmp(pszData, szPassword) == 0)
		{
			bResult = TRUE;
		}
	}

	return bResult;
}

/******************************************************************
Desc:	Function to test if the secure password entered matches the
		password stored globally.  This function expects that the
		expiration time will be a string representation of the 
		time (i.e. FLM_GET_TIMER + some duration).
*******************************************************************/
FLMBOOL F_WebPageFactory::isSecureAccessEnabled()
{
	GET_GBL_VALUE_FN		fnGetGblValue =
									gv_FlmSysData.HttpConfigParms.fnGetGblValue;
	char						szExpiration[ 20];
	FLMUINT					uiExpSize = sizeof( szExpiration);
	FLMUINT					uiExpTime;
	FLMUINT					uiCurrTime;
	FLMBOOL					bResult = FALSE;

	flmAssert( fnGetGblValue);

	// Assuming that an error code will be returned if the global value
	// has not been set.
	if (fnGetGblValue( FLM_SECURE_EXPIRATION,
							 szExpiration,
							 (FLMSIZET *)&uiExpSize) == 0)
	{
		uiExpTime = f_atoud( szExpiration);

		f_timeGetSeconds( &uiCurrTime);

		if (uiCurrTime < uiExpTime)
		{
			bResult = TRUE;
		}
	}

	return bResult;
}

/****************************************************************************
 Desc:	Each of these functions, when called, will create a new object of
			a particular class.  They are called by WebPageFactory::Create()
****************************************************************************/
static F_WebPage * createErrorPage()
{ 
	return f_new F_ErrorPage;
}

static F_WebPage * createGblAccessPage()
{ 
	return f_new F_GblAccessPage;
}

static F_WebPage * createSessionAccessPage()
{ 
	return f_new F_SessionAccessPage;
}

static F_WebPage * createSCacheBlockPage()
{
	return f_new F_SCacheBlockPage;
}

static F_WebPage * createSCacheHashTablePage()
{
	return f_new F_SCacheHashTablePage;
}

static F_WebPage * createSCacheUseListPage()
{
	return f_new F_SCacheUseListPage;
}

static F_WebPage * createSCacheNotifyListPage()
{
	return f_new F_SCacheNotifyListPage;
}

static F_WebPage * createSCacheDataPage()
{
	return f_new F_SCacheDataPage;
}

static F_WebPage * createSCacheMgrPage()
{ 
	return f_new F_SCacheMgrPage;
}

static F_WebPage * createQueriesPage()
{ 
	return f_new F_QueriesPage;
}

static F_WebPage * createQueryPage()
{ 
	return f_new F_QueryPage;
}

static F_WebPage * createQueryStatsPage()
{ 
	return f_new F_QueryStatsPage;
}

static F_WebPage * createSysConfigPage()
{ 
	return f_new F_SysConfigPage;
}

static F_WebPage * createStatsPage()
{ 
	return f_new F_StatsPage;
}

static F_WebPage * createFlmSysDataPage()
{
	return f_new F_FlmSysDataPage;
}

static F_WebPage * createHttpConfigParmsPage()
{
	return f_new F_HttpConfigParmsPage;
}

static F_WebPage * createFlmThreadsPage()
{
	return f_new F_FlmThreadsPage;
}

static F_WebPage * createFlmIndexPage()
{
	return f_new F_FlmIndexPage;
}

static F_WebPage * createIndexListPage()
{
	return f_new F_IndexListPage;
}

static F_WebPage * createSelectPage()
{ 
	return f_new F_SelectPage;
}

static F_WebPage * createCheckDbPage()
{ 
	return f_new F_CheckDbPage;
}

static F_WebPage * serveFile()
{
	return f_new F_HttpFile;
}

static F_WebPage * createDbBackupPage()
{ 
	return f_new F_HttpDbBackup;
}

static F_WebPage * createFileHashTblPage()
{
	return f_new F_FileHashTblPage;
}

static F_WebPage * createFFilePage()
{ 
	return f_new F_FFilePage;
}

static F_WebPage * createFDBPage()
{
	return f_new F_FDBPage;
}

static F_WebPage * createRCacheMgrPage()
{
	return f_new F_RCacheMgrPage;
}

static F_WebPage * createRCachePage()
{ 
	return f_new F_RCachePage;
}

static F_WebPage * createRecordMgrPage()
{
	return f_new F_RecordMgrPage;
}

static F_WebPage * createRCHashBucketPage()
{
	return f_new F_RCHashBucketPage;
}

// Frame pages
static F_WebPage * createHeaderFrame()
{
	return f_new F_FrameHeader;
}

static F_WebPage * createMainFrame()
{
	return f_new F_FrameMain;
}

static F_WebPage * createNavFrame()
{
	return f_new F_FrameNav;
}

static F_WebPage * createWelcomeFrame()
{
	return f_new F_FrameWelcome;
}

static F_WebPage * createSecureDbAccessPage()
{
	return f_new F_SecureDbAccess;
}

static F_WebPage * createSecureDbInfoPage()
{
	return f_new F_SecureDbInfo;
}

static F_WebPage * createDatabaseConfigPage()
{
	return f_new F_DatabaseConfigPage;
}

static F_WebPage * createDatabasePage()
{
	return f_new F_DatabasePage;
}

static F_WebPage * createRecordPage()
{
	return f_new F_RecordPage;
}

static F_WebPage * createProcessRecordPage()
{
	return f_new F_ProcessRecordPage;
}

static F_WebPage * createLogHeaderPage()
{
	return f_new F_LogHeaderPage;
}

// Initialize the static variables in the class...
CREATE_FN F_WebPageFactory::m_fnDefault = createMainFrame;
CREATE_FN F_WebPageFactory::m_fnError = createErrorPage;
CREATE_FN F_WebPageFactory::m_fnGblAccess = createGblAccessPage;
CREATE_FN F_WebPageFactory::m_fnSessionAccess = createSessionAccessPage;

RegistryEntry F_WebPageFactory::m_Registry[] = {
	{"FDB", createFDBPage, FALSE},
	{"FFile", createFFilePage, FALSE},
	{"FileHashTbl", createFileHashTblPage, FALSE},
	{"FlmSysData", createFlmSysDataPage, FALSE},
	{"Header.htm", createHeaderFrame, FALSE},
	{"HttpConfigParms", createHttpConfigParmsPage, FALSE},
	{"LogHdr", createLogHeaderPage, FALSE},
	{"Nav.htm", createNavFrame, FALSE},
	{"ProcessRecord", createProcessRecordPage, TRUE},
	{"Queries", createQueriesPage, FALSE},
	{"Query", createQueryPage, FALSE},
	{"QueryStats", createQueryStatsPage, FALSE},
	{"RCHashBucket", createRCHashBucketPage, FALSE},
	{"RCache", createRCachePage, FALSE},
	{"RCacheMgr", createRCacheMgrPage, FALSE},
	{"Record", createRecordPage, TRUE},
	{"SCacheBlock", createSCacheBlockPage, FALSE},
	{"SCacheData", createSCacheDataPage, TRUE},
	{"SCacheHashTable", createSCacheHashTablePage, FALSE},
	{"SCacheMgr", createSCacheMgrPage, FALSE},
	{"SCacheNotifyList", createSCacheNotifyListPage, FALSE},
	{"SCacheUseList", createSCacheUseListPage, FALSE},
	{"SecureDbAccess", createSecureDbAccessPage, FALSE},
	{"SecureDbInfo", createSecureDbInfoPage, FALSE},
	{"Stats", createStatsPage, FALSE},
	{"SysConfig", createSysConfigPage, TRUE},
	{"Welcome.htm", createWelcomeFrame, FALSE},
	{"checkdb", createCheckDbPage, TRUE},
	{"database", createDatabasePage, TRUE},
	{"dbconfig", createDatabaseConfigPage, TRUE},
	{"dbbackup", createDbBackupPage, TRUE},
	{"file", serveFile, TRUE},
	{"index", createFlmIndexPage, TRUE},
	{"indexlist", createIndexListPage, TRUE},
	{"recordmgr", createRecordMgrPage, TRUE},
	{"select", createSelectPage, TRUE},
	{"staticfile", serveFile, FALSE},
	{"threads", createFlmThreadsPage, TRUE},
	{"", NULL, FALSE}
};
// WARNING:  Make sure that every different WebPage class that you want to
// display is listed in the array above.
