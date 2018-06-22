//-------------------------------------------------------------------------
// Desc:	Dynamic, interactive list manager - definitions.
// Tabs:	3
//
// Copyright (c) 2000, 2003, 2005-2007 Novell, Inc. All Rights Reserved.
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

#ifndef FLM_DLST_H
#define FLM_DLST_H

#include "flaimsys.h"

#ifdef __cplusplus
	class F_DynaList;
	class F_DynamicList;
	typedef F_DynaList *		F_DynaList_p;
#else
	typedef void *				F_DynaList_p;
#endif

#ifdef FLM_NLM
	#define DLIST_DUMPFILE_PATH "sys:/system/dlstdump.txt"
#else
	#define DLIST_DUMPFILE_PATH "dlstdump.txt"
#endif


typedef RCODE (* F_DLIST_DISP_HOOK)(
	FTX_WINDOW *		pWin,
	FLMBOOL				bSelected,
	FLMUINT				uiRow,
	FLMUINT				uiKey,
	void *				pvData,
	FLMUINT				uiDataLen,
	F_DynamicList*		pDynamicList);

RCODE  dlistDefaultDisplayHook(
	FTX_WINDOW *		pWin,
	FLMBOOL				bSelected,
	FLMUINT				uiRow,
	FLMUINT				uiKey,
	void *				pvData,
	FLMUINT				uiDataLen,
	F_DynamicList*		pDynamicList);

/*
Types, enums, etc.
*/

typedef struct dlist_node
{
	FLMUINT					uiKey;
	dlist_node *			pPrev;
	dlist_node *			pNext;
	void *					pvData;
	FLMUINT					uiDataLen;
	F_DLIST_DISP_HOOK		pDispHook;
} DLIST_NODE;

/*
Class definitions
*/

#ifdef __cplusplus

class	F_DynamicList : public F_Object
{
private:

	DLIST_NODE *		m_pFirst;
	DLIST_NODE *		m_pLast;
	DLIST_NODE *		m_pCur;
	FTX_WINDOW *		m_pListWin;
	FLMUINT				m_uiListRows;
	FLMUINT				m_uiListCols;
	FLMUINT				m_uiRow;
	FLMBOOL				m_bChanged;
	FLMBOOL				m_bShowHorizontalSelector;

	/*
	  Methods
	*/

	DLIST_NODE * getNode( FLMUINT uiKey);
	void freeNode( DLIST_NODE * pNode);

public:

	/*
	  Methods
	*/

	F_DynamicList( void);
	~F_DynamicList( void);

	RCODE setup( FTX_WINDOW *	pInitializedWindow);

	void refresh( void);

	RCODE insert( 
		FLMUINT					uiKey,
		F_DLIST_DISP_HOOK 	pDisplayHook,
		void *					pvData,
		FLMUINT					uiDataLen);

	RCODE update( 
		FLMUINT					uiKey,
		F_DLIST_DISP_HOOK 	pDisplayHook,
		void *					pvData,
		FLMUINT					uiDataLen);

	RCODE remove(
		FLMUINT	uiKey);

	FINLINE DLIST_NODE * getFirst( void)
	{ 
		return( m_pFirst);
	}

	FINLINE DLIST_NODE * getCurrent( void)
	{ 
		return( m_pCur);
	}

	FINLINE FTX_WINDOW * getListWin( void)
	{ 
		return( m_pListWin);
	}

	void defaultKeyAction( FLMUINT uiKey);

	void setShowHorizontalSelector( FLMBOOL bShow)
	{
		m_bShowHorizontalSelector = bShow;
		m_bChanged = TRUE;
	}
	FLMBOOL getShowHorizontalSelector() { return m_bShowHorizontalSelector;}

	RCODE dumpToFile();

	/*
	  Navigational Methods
	*/

	void cursorUp( void);
	void cursorDown( void);
	void pageUp( void);
	void pageDown( void);
	void home( void);
	void end( void);
};

#endif	// __cplusplus
#endif	// FLM_EDIT_H
