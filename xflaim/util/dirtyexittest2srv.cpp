//------------------------------------------------------------------------------
// Desc:	Dirty exit test 2
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

#include "flmunittest.h"

#if defined( FLM_NLM)
	#define DB_NAME_STR					"SYS:\\BAD.DB"
	#define REBUILD_DEST_NAME_STR		"SYS:\\BLD.DB"	
#else
	#define DB_NAME_STR					"bad.db"
	#define REBUILD_DEST_NAME_STR		"bld.db"	
#endif

/*****************************************************************************
Desc:
******************************************************************************/
class IDirtyExitTest2Impl : public TestBase
{
public:

	const char * getName( void);
	
	RCODE execute( void);
};

/*****************************************************************************
Desc:
******************************************************************************/
RCODE getTest( 
	IFlmTest **		ppTest)
{
	RCODE		rc = NE_XFLM_OK;

	if( (*ppTest = f_new IDirtyExitTest2Impl) == NULL)
	{
		rc = NE_XFLM_MEM;
		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
const char * IDirtyExitTest2Impl::getName( void)
{
	return( "Dirty Exit Test 2");
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE IDirtyExitTest2Impl::execute( void)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBOOL		bDibCreated = FALSE;
	FLMUINT64	ui64TotalNodes;
	FLMUINT64	ui64NodesRecovered;

	// Open the test state created by Dirty Exit Test 1.
	
	if ( RC_BAD( rc = openTestState( DB_NAME_STR)))
	{
		goto Exit;
	}
	
	bDibCreated = TRUE;

	beginTest( 
		"Database check test", 
		"Make sure dib was not corrupted by dirty shutdown",
		"Self explanatory",
		"No Additional Details.");

	// Make sure the database is still consistent
	
	if( RC_BAD( rc = m_pDbSystem->dbCheck( DB_NAME_STR, NULL, NULL, NULL,
		XFLM_DO_LOGICAL_CHECK, NULL, NULL)))
	{
		MAKE_FLM_ERROR_STRING("dbCheck failed", m_szDetails, rc);
		goto Exit;
	}
	
	endTest("PASS");

	beginTest( 
		"Database rebuild test", 
		"Make sure the rebuild code works",
		"Self explanatory",
		"No Additional Details.");
	
	if( RC_BAD( rc = m_pDbSystem->dbRebuild( DB_NAME_STR, NULL,
		REBUILD_DEST_NAME_STR, NULL, NULL,
		NULL, NULL, NULL, &ui64TotalNodes, &ui64NodesRecovered,
		NULL, NULL)))
	{
		MAKE_FLM_ERROR_STRING("dbRebuild failed", m_szDetails, rc);
		goto Exit;
	}

	if( ui64TotalNodes != ui64NodesRecovered)
	{
		MAKE_FLM_ERROR_STRING("dbRebuild failed", m_szDetails, NE_XFLM_FAILURE);
		goto Exit;
	}

	endTest("PASS");

Exit:

	if( RC_BAD( rc))
	{
		endTest("FAIL");
	}

	shutdownTestState( DB_NAME_STR, bDibCreated);
	return( rc);
}
