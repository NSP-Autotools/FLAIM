//------------------------------------------------------------------------------
// Desc: Utility to compare two databases for equivalence.
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

/*=============================================================================
Desc:	Define the F_DbDiff class - also note the need for the call back
		definition.
*============================================================================*/
typedef void (* DBDIFF_CALLBACK) (
	char *		pszOutput,
	void *		pvData);


class F_DbDiff
{
public:

	F_DbDiff()
	{
		m_pDb1 = m_pDb2 = NULL;
	}

	~F_DbDiff()
	{
		// Close the databases if they are open.
		if ( m_pDb1 != NULL)
		{
			m_pDb1->Release();
		}
		if ( m_pDb2 != NULL)
		{
			m_pDb2->Release();
		}
	}

	RCODE diff(
		char  *				pszDb1,
		char *				pszDb1Password,
		char *				pszDb2,
		char *				pszDb2Password,
		DBDIFF_CALLBACK	outputCallback,
		void *				pvData);

private:

	RCODE compareCollections(
		FLMUINT				uiCollection,
		DBDIFF_CALLBACK	outputCallback,
		void *				pvData);

	RCODE compareIndexes( 
		FLMUINT uiIndexNum,
		DBDIFF_CALLBACK outputCallback,
		void * pvData);

	RCODE compareNodes(
		FLMBYTE *			pszCompareInfo,
		F_DOMNode *			pNode1,
		F_DOMNode *			pNode2,
		DBDIFF_CALLBACK	outputCallback,
		void *				pvData);

	F_Db *		m_pDb1;
	F_Db *		m_pDb2;
};
