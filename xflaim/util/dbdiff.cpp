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

#include "flaimsys.h"
#include "sharutil.h"
#include "dbdiff.h"

/****************************************************************************
Name:	compareNodes
Desc:	method to compare two F_DOMNodes.
****************************************************************************/
RCODE F_DbDiff::compareNodes(
	FLMBYTE *			pszCompareInfo,	//info about the records for output
	F_DOMNode *			pNode1,
	F_DOMNode *			pNode2,
	DBDIFF_CALLBACK	outputCallback,
	void *				pvData)
{
	RCODE				rc = NE_XFLM_OK;
	FlmStringAcc	acc;
	char				szErrBuff[ 100];

	if (pNode1->compareNode( 
		pNode2,
		m_pDb1,
		m_pDb2,
		&szErrBuff[0], 
		sizeof( szErrBuff)) != 0)
	{
		acc.printf(
			"ERROR: {%s} %s\n", pszCompareInfo, szErrBuff);
		outputCallback( (char*)acc.getTEXT(), pvData);
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}

Exit:

	return rc;
}

/****************************************************************************
Name:	compareCollections
Desc:	compare two databases' Collection
****************************************************************************/
RCODE F_DbDiff::compareCollections(
	FLMUINT				uiCollection,
	DBDIFF_CALLBACK	outputCallback,
	void *				pvData)
{
	FLMUINT64		ui64NodeId1			= 0;
	FLMUINT64		ui64NodeId2			= 0;
	F_DOMNode *		pNode1				= NULL;
	F_DOMNode *		pNode2				= NULL;
	FLMBOOL			bScanFinished1 = FALSE;
	FLMBOOL			bScanFinished2 = FALSE;
	FlmStringAcc	acc;
	RCODE				rc					= NE_XFLM_OK;

	if ( uiCollection == XFLM_MAINT_COLLECTION)
	{
		// Ignore any differences in this collection
		goto Exit;
	}

	while ( (!bScanFinished1) && (!bScanFinished2))
	{
		if ( RC_BAD( rc = m_pDb1->getNextNode( uiCollection,
															ui64NodeId1,
															&pNode1)))
		{
			if ( rc == NE_XFLM_EOF_HIT)
			{
				rc = NE_XFLM_OK;
				bScanFinished1 = TRUE;
			}
			else
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = pNode1->getNodeId( m_pDb1, &ui64NodeId1)))
			{
				goto Exit;
			}
		}

		if ( RC_BAD( rc = m_pDb2->getNextNode( uiCollection,
															ui64NodeId2,
															&pNode2)))
		{
			if ( rc == NE_XFLM_EOF_HIT)
			{
				rc = NE_XFLM_OK;
				bScanFinished2 = TRUE;
			}
			else
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = pNode1->getNodeId( m_pDb2, &ui64NodeId2)))
			{
				goto Exit;
			}
		}

		if ( bScanFinished1 && bScanFinished2)
		{
			break;
		}

		//if one of them is finished and not the other one
		if (bScanFinished1 != bScanFinished2)
		{
			FlmStringAcc acc;
			acc.appendf(
				"ERROR: database1 and database2 have a different # of nodes in "
				"Collection %u\n", uiCollection);
			outputCallback( (char*)acc.getTEXT(), pvData);
			rc = RC_SET( NE_XFLM_FAILURE);
			goto Exit;
		}

		//if the nodeId's are different, then there's a problem
		if ( ui64NodeId1 != ui64NodeId2)
		{
			FlmStringAcc acc;
			acc.appendf( "ERROR: database1's nodeId %u ", ui64NodeId1);
			acc.appendf( " != database2's nodeId %u!\n", ui64NodeId2);
			acc.appendf( "ERROR: nodeId mismatch in Collection %u\n", uiCollection);
			outputCallback( (char*)acc.getTEXT(), pvData);
			rc = RC_SET( NE_XFLM_FAILURE);
			goto Exit;
		}

		if ( !bScanFinished1) //bScanFinished1==bScanFinished2 at this point
		{
			acc.printf( "NodeId %u of Collection %u", ui64NodeId1, uiCollection);
			//compare the two records for accuracy
			if ( RC_BAD( rc = compareNodes( (FLMBYTE *)acc.getTEXT(),
													  pNode1,
													  pNode2,
													  outputCallback,
													  pvData)))
			{
				goto Exit;
			}
		}
#ifdef FLM_NLM
		f_yieldCPU();
#else
		f_sleep( 0);
#endif
	}

Exit:

	if ( pNode1)
	{
		pNode1->Release();
	}
	if ( pNode2)
	{
		pNode2->Release();
	}
	return rc;
}


/****************************************************************************
Name:	flmDbDiff
Desc:	compare two databases for logical equivalence.
****************************************************************************/
RCODE F_DbDiff::diff(
	char *				pszDb1,
	char *				pszDb1Password,
	char *				pszDb2,
	char *				pszDb2Password,
	DBDIFF_CALLBACK	outputCallback,
	void *				pvData)
{
	RCODE					rc	= NE_XFLM_OK;
	RCODE					tmpRc	= NE_XFLM_OK;
	FLMBOOL				bTransActive1 = FALSE;
	FLMBOOL				bTransActive2 = FALSE;
	F_DOMNode *			pNode1 = NULL;
	F_DOMNode *			pNode2 = NULL;
	FlmVector			CollectionVec1;
	FlmVector			CollectionVec2;
	FLMUINT				uiCollectionLen1 = 0;
	FLMUINT				uiCollectionLen2 = 0;
	FLMUINT				uiIndexLen1 = 0;
	FLMUINT				uiIndexLen2 = 0;
	FLMUINT				uiLoop;
	F_COLLECTION  *	pCollection1;
	F_COLLECTION *		pCollection2;
	F_Dict *				pDict1;
	F_Dict *				pDict2;
	FLMUINT				uiCollectionNum;
	FLMUINT				uiIndexNum;
	IXD *					pIxd;
	FlmStringAcc		acc;
	FlmVector			IndexVec1;
	FlmVector			IndexVec2;

	//try to unlock any files currently open
	if ( RC_BAD( rc = dbSystem.closeUnusedFiles(0)))
	{
		goto Exit;
	}

	//can't compare a db with itself
	if ( f_strcmp( pszDb1, pszDb2) == 0)
	{
		outputCallback( "ERROR: cannot compare a dib with itself!\n", pvData);
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}

	//open up the database, and all that it entails
	if( RC_BAD( rc = dbSystem.openDb( pszDb1,
													 NULL,
													 NULL,
													 pszDb1Password,
													 XFLM_DONT_RESUME_THREADS,
													 (IF_Db **)&m_pDb1)))
	{
		acc.printf( "ERROR: cannot open dib '");
		acc.appendTEXT( pszDb1);
		acc.appendTEXT( "'!\n");
		outputCallback((char*)acc.getTEXT(), pvData);
		goto Exit;
	}
	if( RC_BAD( rc = dbSystem.openDb( pszDb2,
													 NULL,
													 NULL,
													 pszDb2Password,
													 XFLM_DONT_RESUME_THREADS,
													 (IF_Db **)&m_pDb2)))
	{
		acc.printf( "ERROR: cannot open dib '");
		acc.appendTEXT( pszDb2);
		acc.appendTEXT( "'!\n");
		outputCallback((char*)acc.getTEXT(), pvData);
		goto Exit;
	}
	if( RC_BAD( rc = m_pDb1->transBegin( XFLM_READ_TRANS)))
	{
		outputCallback( "ERROR: cannot start trans #1\n", pvData);
		goto Exit;
	}
	bTransActive1 = TRUE;
	if( RC_BAD( rc = m_pDb2->transBegin( XFLM_READ_TRANS)))
	{
		outputCallback( "ERROR: cannot start trans #2\n", pvData);
		goto Exit;
	}
	bTransActive2 = TRUE;

	if (RC_BAD( rc = m_pDb1->getDictionary( &pDict1)))
	{
		outputCallback( "ERROR: retrieving dictionary #1\n", pvData);
      goto Exit;
	}

	if (RC_BAD( rc = m_pDb2->getDictionary( &pDict2)))
	{
		outputCallback( "ERROR: retrieving dictionary #2\n", pvData);
      goto Exit;
	}

	//prepare to read Collections from the dictionary
	if (RC_BAD( rc = pDict1->getCollection( XFLM_DICT_COLLECTION,
														 &pCollection1)))
	{
		outputCallback( "ERROR: retrieving dictionary collection #1\n", pvData);
		goto Exit;
	}

	if (RC_BAD( rc = pDict2->getCollection( XFLM_DICT_COLLECTION,
														 &pCollection2)))
	{
		outputCallback( "ERROR: retrieving dictionary collection #2\n", pvData);
		goto Exit;
	}

	outputCallback( "Starting dbdiff with databases '", pvData);
	outputCallback( (char*)pszDb1, pvData);
	outputCallback( "' and '", pvData);
	outputCallback( (char*)pszDb2, pvData);
	outputCallback( "'\n", pvData);
	outputCallback( "Building index and Collection lists...", pvData);
	
	//build up a list of each Collection from the first database
	for (uiCollectionNum = 0;;)
	{
		if ((pCollection1 = pDict1->getNextCollection( uiCollectionNum,
																	  TRUE)) == NULL)
		{
			break;
		}
		uiCollectionNum = pCollection1->lfInfo.uiLfNum;
		if ( RC_BAD( rc = CollectionVec1.setElementAt( (void*)uiCollectionNum,
																	  uiCollectionLen1++)))
		{
			goto Exit;
		}
	}

	//build up a list of each Collection from the second database
	for (uiCollectionNum = 0;;)
	{
		if ((pCollection2 = pDict2->getNextCollection( uiCollectionNum,
																	  TRUE)) == NULL)
		{
			break;
		}
		uiCollectionNum = pCollection2->lfInfo.uiLfNum;
		if ( RC_BAD( rc = CollectionVec2.setElementAt( (void*)uiCollectionNum,
																	  uiCollectionLen2++)))
		{
			goto Exit;
		}
	}

	//must have same # of Collections to be the same
	if ( uiCollectionLen1 != uiCollectionLen2)
	{
		acc.printf(
			"ERROR: the two databases have a different number of Collections("
			"%u,%u)!\n", uiCollectionLen1, uiCollectionLen2);
		outputCallback( (char*)acc.getTEXT(), pvData);
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}

	//must have same # of indexes to be the same
	//build up a list of each Collection from the first database
	for (uiIndexNum = 0;;)
	{
		if ((pIxd = pDict1->getNextIndex( uiIndexNum,
													 TRUE)) == NULL)
		{
			break;
		}
		uiIndexNum = pIxd->uiIndexNum;
		IndexVec1.setElementAt(	(void*)uiIndexNum, uiIndexLen1++);
	}

	for (uiIndexNum = 0;;)
	{
		if ((pIxd = pDict2->getNextIndex( uiIndexNum,
													 TRUE)) == NULL)
		{
			break;
		}
		uiIndexNum = pIxd->uiIndexNum;
		IndexVec2.setElementAt(	(void*)uiIndexNum, uiIndexLen2++);
	}

	if ( uiIndexLen1 != uiIndexLen2)
	{
		acc.printf(
			"ERROR: the two databases have a different number of indexes("
			"%u,%u)!\n", uiIndexLen1, uiIndexLen2);
		outputCallback( (char*)acc.getTEXT(), pvData);
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}

	outputCallback( "done\n", pvData);
	outputCallback( "comparing collections\n", pvData);

	//loop through each Collection and compare the two
	for ( uiLoop = 0; uiLoop < uiCollectionLen1; uiLoop++)
	{
		FLMUINT uiCollection1 = (FLMUINT)CollectionVec1.getElementAt( uiLoop);
		FLMUINT uiCollection2 = (FLMUINT)CollectionVec2.getElementAt( uiLoop);

		if ( uiCollection1 != uiCollection2)
		{
			acc.printf( "ERROR: database1's Collection (%u) != "
				"database2's Collection (%u)!\n", uiCollection1, uiCollection2);
			outputCallback( (char*)acc.getTEXT(), pvData);
			rc = RC_SET( NE_XFLM_FAILURE);
			goto Exit;
		}
		else
		{
			acc.printf( "(%4u of %4u) processing Collection #%u\n",
				uiLoop+1, uiCollectionLen1, uiCollection1);
			outputCallback( (char*)acc.getTEXT(), pvData);
			if ( RC_BAD( rc = compareCollections( uiCollection1,
															  //uiCollection1==uiCollection2 at this point
															  outputCallback,
															  pvData)))
			{
				goto Exit;
			}
		}
	}	

	// Compare index keys

	outputCallback( "done\n", pvData);
	outputCallback( "comparing indexes\n", pvData);

	for( uiLoop = 0; uiLoop < uiIndexLen1; uiLoop++)
	{
		uiIndexNum = (FLMUINT)IndexVec1.getElementAt( uiLoop);

		if ( uiIndexNum != (FLMUINT)IndexVec2.getElementAt( uiLoop))
		{
			acc.printf( "ERROR: database1's Index (%u) != "
				"database2's Index (%u)!\n", uiIndexNum, IndexVec2.getElementAt( uiLoop));
			outputCallback( (char*)acc.getTEXT(), pvData);
			rc = RC_SET( NE_XFLM_FAILURE);
			goto Exit;
		}

		acc.printf( "(%5u of %5u) processing Index #%u\n",
				uiLoop + 1, uiIndexLen1, uiIndexNum);		
		outputCallback( acc.getTEXT(), pvData);

		if ( RC_BAD( rc = compareIndexes( uiIndexNum, outputCallback, pvData)))
		{
			goto Exit;
		}
	}

Exit:
	
	if (pNode1)
	{
		pNode1->Release();
	}
	if (pNode2)
	{
		pNode2->Release();
	}

	if ( bTransActive1)
	{
		if ( RC_OK( rc))
		{
			rc = m_pDb1->transCommit();
		}
		else
		{
			(void)m_pDb1->transAbort();
		}
	}
	if ( bTransActive2)
	{
		if ( RC_OK( rc))
		{
			rc = m_pDb2->transCommit();
		}
		else
		{
			(void)m_pDb2->transAbort();
		}
	}
	//close any files currently open.  This is so any smi open can be
	//done following this in the same process.
	tmpRc = dbSystem.closeUnusedFiles(0);
	if ( RC_OK( rc))
	{
		rc = tmpRc;
	}
	return rc;
}

RCODE F_DbDiff::compareIndexes( 
	FLMUINT uiIndexNum,
	DBDIFF_CALLBACK outputCallback,
	void * pvData)
{
	RCODE					rc = NE_XFLM_OK;
	F_DataVector		searchKey1;
	F_DataVector		searchKey2;
	FLMBOOL				bDataComp;
	FLMBOOL				bKeyComp;
	FLMUINT				uiLoop;
	FLMUINT				uiLoop2;
	FLMUINT				uiDataType;
	FLMUINT				uiDataLen1;
	FLMUINT				uiDataLen2;
	FLMBYTE *			pucVal1 = NULL;
	FLMBYTE *			pucVal2 = NULL;
	FLMUINT64			ui64Val1 = 0;
	FLMUINT64			ui64Val2 = 0;

	for( uiLoop = 0;; uiLoop++)
	{
		if ( uiLoop == 0)
		{
			if (RC_BAD( rc = m_pDb1->keyRetrieve( 
				uiIndexNum, NULL, XFLM_FIRST, &searchKey1)))
			{
				if ( rc == NE_XFLM_EOF_HIT)
				{
					// empty index. Make sure the other index is empty too.

					if (( rc = m_pDb2->keyRetrieve( 
						uiIndexNum, NULL, XFLM_FIRST, &searchKey2)) != NE_XFLM_EOF_HIT)
					{
						rc = RC_SET( NE_XFLM_DATA_ERROR);
						outputCallback( (char*)"key mismatch found.", pvData);
						goto Exit;
					}
				}
				rc = NE_XFLM_OK;
				goto Exit;
			}

			if (RC_BAD( rc = m_pDb2->keyRetrieve( 
				uiIndexNum, NULL, XFLM_FIRST, &searchKey2)))
			{
				outputCallback( (char*)"keyRetrieve failed.", pvData);
				goto Exit;
			}

			if ( searchKey1.getDocumentID() != searchKey2.getDocumentID())
			{
				// Error
				rc = RC_SET( NE_XFLM_DATA_ERROR);
				outputCallback( (char*)"index document id mismatch.", pvData);
				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = m_pDb1->keyRetrieve( 
				uiIndexNum, &searchKey1, XFLM_EXCL, &searchKey1)))
			{
				if ( rc == NE_XFLM_EOF_HIT)
				{
					// No more keys. Make sure the other index is at the end too.

					if (( rc = m_pDb2->keyRetrieve( 
						uiIndexNum, &searchKey2, XFLM_EXCL, &searchKey2)) != NE_XFLM_EOF_HIT)
					{
						rc = RC_SET( NE_XFLM_DATA_ERROR);
						outputCallback( (char*)"key mismatch found.", pvData);
						goto Exit;
					}
				}
				rc = NE_XFLM_OK;
				goto Exit;
			}

			if (RC_BAD( rc = m_pDb2->keyRetrieve( 
				uiIndexNum, &searchKey2, XFLM_EXCL, &searchKey2)))
			{
				outputCallback( (char*)"keyRetrieve failed.", pvData);
				goto Exit;
			}
		}

		for( uiLoop2 = 0;; uiLoop2++)
		{
			if ( ( bDataComp = searchKey1.isDataComponent(uiLoop2)) != 
				searchKey2.isDataComponent(uiLoop2))
			{
				rc = RC_SET( NE_XFLM_DATA_ERROR);
				outputCallback( (char*)"isDataComponent mismatch.", pvData);
				goto Exit;
			}

			if ( ( bKeyComp = searchKey1.isKeyComponent(uiLoop2)) != 
				searchKey2.isKeyComponent(uiLoop2))
			{
				rc = RC_SET( NE_XFLM_DATA_ERROR);
				outputCallback( (char*)"isKeyComponent mismatch.", pvData);
				goto Exit;
			}

			if ( !bDataComp && !bKeyComp)
			{
				// No more components to check
				break;
			}

			if ( searchKey1.isRightTruncated( uiLoop2) != 
				searchKey2.isRightTruncated( uiLoop2))
			{
				rc = RC_SET( NE_XFLM_DATA_ERROR);
				outputCallback( (char*)"isRightTruncated mismatch.", pvData);
				goto Exit;
			}

			if ( searchKey1.isLeftTruncated( uiLoop2) != 
				searchKey2.isLeftTruncated( uiLoop2))
			{
				rc = RC_SET( NE_XFLM_DATA_ERROR);
				outputCallback( (char*)"isLeftTruncated mismatch.", pvData);
				goto Exit;
			}

			if ( searchKey1.getID(uiLoop2) != searchKey2.getID(uiLoop2))
			{
				rc = RC_SET( NE_XFLM_DATA_ERROR);
				outputCallback( (char*)"node id mismatch.", pvData);
				goto Exit;
			}

			if ( searchKey1.getNameId(uiLoop2) != searchKey2.getNameId(uiLoop2))
			{
				rc = RC_SET( NE_XFLM_DATA_ERROR);
				outputCallback( (char*)"name id mismatch.", pvData);
				goto Exit;
			}

			if ( searchKey1.isAttr(uiLoop2) != searchKey2.isAttr(uiLoop2))
			{
				rc = RC_SET( NE_XFLM_DATA_ERROR);
				outputCallback( (char*)"isAttr mismatch.", pvData);
				goto Exit;
			}

			if ( ( uiDataType = searchKey1.getDataType(uiLoop2)) != 
				searchKey2.getDataType(uiLoop2))
			{
				rc = RC_SET( NE_XFLM_DATA_ERROR);
				outputCallback( (char*)"data type mismatch.", pvData);
				goto Exit;
			}

			if ( ( uiDataLen1 = searchKey1.getDataLength(uiLoop2)) != 
				( uiDataLen2 = searchKey2.getDataLength(uiLoop2)))
			{
				rc = RC_SET( NE_XFLM_DATA_ERROR);
				outputCallback( (char*)"data length mismatch.", pvData);
				goto Exit;
			}

			if ( uiDataType == XFLM_TEXT_TYPE || uiDataType == XFLM_BINARY_TYPE)
			{
				if ( pucVal1)
				{
					f_free( &pucVal1);
				}

				if ( RC_BAD( rc = f_alloc( uiDataLen1, &pucVal1)))
				{
					goto Exit;
				}

				if ( pucVal2)
				{
					f_free( &pucVal2);
				}

				if ( RC_BAD( rc = f_alloc( uiDataLen2, &pucVal2)))
				{
					goto Exit;
				}
			}

			switch( uiDataType)
			{
				case XFLM_NUMBER_TYPE:
				{
					if ( RC_BAD( rc = searchKey1.getUINT64( uiLoop2, &ui64Val1)))
					{
						outputCallback( (char*)"getUINT64 failed.", pvData);
						goto Exit;
					}

					if ( RC_BAD( rc = searchKey2.getUINT64( uiLoop2, &ui64Val2)))
					{
						outputCallback( (char*)"getUINT64 failed.", pvData);
						goto Exit;
					}

					if ( ui64Val1 != ui64Val2)
					{
						rc = RC_SET( NE_XFLM_DATA_ERROR);
						outputCallback( (char*)"UINT64 val mismatch.", pvData);
						goto Exit;
					}
					break;
				}
				case XFLM_BINARY_TYPE:
				{
					if ( RC_BAD( rc = searchKey1.getBinary( uiLoop2, pucVal1, &uiDataLen1)))
					{
						outputCallback( (char*)"getBinary failed.", pvData);
						goto Exit;
					}

					if ( RC_BAD( rc = searchKey2.getBinary( uiLoop2, pucVal2, &uiDataLen2)))
					{
						outputCallback( (char*)"getBinary failed.", pvData);
						goto Exit;
					}

					if ( f_memcmp( pucVal1, pucVal2, uiDataLen2) != 0)
					{
						rc = RC_SET( NE_XFLM_DATA_ERROR);
						outputCallback( (char*)"binary val mismatch.", pvData);
						goto Exit;
					}
					break;
				}
				case XFLM_TEXT_TYPE:
				{
					if ( RC_BAD( rc = searchKey1.getUTF8( uiLoop2, pucVal1, &uiDataLen1)))
					{
						outputCallback( (char*)"getUTF8 failed.", pvData);
						goto Exit;
					}

					if ( RC_BAD( rc = searchKey2.getUTF8( uiLoop2, pucVal2, &uiDataLen2)))
					{
						outputCallback( (char*)"getUTF8 failed.", pvData);
						goto Exit;
					}

					if ( f_strcmp( pucVal1, pucVal2) != 0)
					{
						rc = RC_SET( NE_XFLM_DATA_ERROR);
						outputCallback( (char*)"text val mismatch.", pvData);
						goto Exit;
					}
					break;
				}
				default:
					break;
			}
		}
	}
Exit:

	if ( pucVal1)
	{
		f_free( &pucVal1);
	}

	if ( pucVal2)
	{
		f_free( &pucVal2);
	}

	return rc;
}
