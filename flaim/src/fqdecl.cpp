//-------------------------------------------------------------------------
// Desc:	Various cursor/query functions
// Tabs:	3
//
// Copyright (c) 1994-2001, 2003, 2005-2007 Novell, Inc. All Rights Reserved.
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

FSTATIC RCODE flmSendCursorFrom(
	FCL_WIRE *		pWire,
	CURSOR *			pCursor);

FSTATIC RCODE flmSendCursorWhere(
	FCL_WIRE *		pWire,
	CURSOR *			pCursor);

/****************************************************************************
Desc:	Send the FROM information for the cursor to the client.
****************************************************************************/
FSTATIC RCODE flmSendCursorFrom(
	FCL_WIRE *		pWire,
	CURSOR *			pCursor)
{
	RCODE				rc = FERR_OK;
	NODE *			pRootNode;
	NODE *			pChildNode = NULL;
	NODE *			pTmp;
	F_Pool *			pPool = pWire->getPool();
	void *			pvMark = pPool->poolMark();
	FLMUINT			uiTmp;
	CS_CONTEXT *	pCSContext = pWire->getContext();

	if ((pRootNode = GedNodeMake( pPool, FCS_ITERATOR_FROM, &rc)) == NULL)
	{
		goto Exit;
	}

	uiTmp = 0;
	if (RC_BAD( rc = gedAddField( pPool, pRootNode,
								FCS_ITERATOR_CANDIDATE_SET,
								(void *)&uiTmp,
								0, FLM_NUMBER_TYPE)))
	{
		goto Exit;
	}
	pChildNode = (NODE *)((!pChildNode)
								 ? GedChild( pRootNode)
								 : GedSibNext( pChildNode));

	// Add all record sources.

	if ((pTmp = GedNodeMake( pPool, FCS_ITERATOR_RECORD_SOURCE,
						&rc)) == NULL)
	{
		goto Exit;
	}
	GedChildGraft( pChildNode, pTmp, GED_LAST);

	// Insert container number.

	if (pCursor->uiContainer != FLM_DATA_CONTAINER)
	{
		if (RC_BAD( rc = gedAddField( pPool, pTmp,
									FCS_ITERATOR_CONTAINER_ID,
									(void *)&pCursor->uiContainer,
									0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	// Add record type.

	if (pCursor->uiRecType)
	{
		if (RC_BAD( rc = gedAddField( pPool, pChildNode,
									FCS_ITERATOR_RECORD_TYPE,
									(void *)&pCursor->uiRecType,
									0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	// Add bOkToReturnKeys flag

	if( pCSContext->uiServerFlaimVer >= FLM_FILE_FORMAT_VER_4_3)
	{
		uiTmp = (FLMUINT)(pCursor->bOkToReturnKeys ? 1 : 0);
		if (RC_BAD( rc = gedAddField( pPool, pChildNode,
									FCS_ITERATOR_OK_TO_RETURN_KEYS,
									(void *)&uiTmp,
									0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	// Add index number

	if (pCursor->uiIndexNum)
	{
		if (RC_BAD( rc = gedAddField( pPool, pChildNode,
									FCS_ITERATOR_FLAIM_INDEX,
									(void *)&pCursor->uiIndexNum,
									0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if (RC_BAD( rc = gedAddField( pPool, pRootNode,
								FCS_ITERATOR_MODE,
								(void *)&pCursor->QTInfo.uiFlags,
								0, FLM_NUMBER_TYPE)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pWire->sendHTD( WIRE_VALUE_ITERATOR_FROM, pRootNode)))
	{
		pCSContext->bConnectionGood = FALSE;
		goto Exit;
	}

Exit:

	pPool->poolReset( pvMark);
	return( rc);
}

/****************************************************************************
Desc:	Send selection criteria for the cursor to the server.
****************************************************************************/
FSTATIC RCODE flmSendCursorWhere(
	FCL_WIRE *		pWire,
	CURSOR *			pCursor
	)
{
	RCODE				rc = FERR_OK;
	NODE *			pRootNode;
	NODE *			pFldNode;
	FQNODE *			pQNode;
	FLMUINT			uiOperator;
	FLMUINT			uiLastFlags = 0;
	QTYPES			eOp;
	F_Pool *			pPool = pWire->getPool();
	void *			pvMark = pPool->poolMark();
	CS_CONTEXT *	pCSContext = pWire->getContext();

	if ((pRootNode = GedNodeMake( pPool, FCS_ITERATOR_WHERE, &rc)) == NULL)
	{
		goto Exit;
	}

	if ((pQNode = pCursor->QTInfo.pTopNode) == NULL)
	{
		if ((pQNode = pCursor->QTInfo.pCurAtomNode) == NULL)
		{
			goto Exit;
		}
	}

	// Do an in-order traversal of the tree.

	for (;;)
	{
		eOp = GET_QNODE_TYPE( pQNode);

		// Skip the node if it has children and is not
		// a unary operator.  It will be output after its first child has
		// been output.

		if( pQNode->pChild &&
			(eOp != FLM_NOT_OP && eOp != FLM_NEG_OP))
		{

			// Insert a left paren.

			if( RC_BAD( rc = fcsTranslateQFlmToQCSOp(	FLM_LPAREN_OP, &uiOperator)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = gedAddField( pPool, pRootNode,
										FCS_ITERATOR_OPERATOR,
										(void *)&uiOperator,
										0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}
			pQNode = pQNode->pChild;
			continue;
		}

		// Output the node's mode flags

		if (pQNode->pQAtom &&
			 pQNode->pQAtom->uiFlags != uiLastFlags)
		{
			if (RC_BAD( rc = gedAddField( pPool, pRootNode,
										FCS_ITERATOR_MODE,
										(void *)&pQNode->pQAtom->uiFlags,
										0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}
			uiLastFlags = pQNode->pQAtom->uiFlags;
		}

		// Output the node

		if( eOp == FLM_NOT_OP || eOp == FLM_NEG_OP)
		{

			// Unary operator

			if (RC_BAD( rc = fcsTranslateQFlmToQCSOp( eOp, &uiOperator)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = gedAddField( pPool, pRootNode,
										FCS_ITERATOR_OPERATOR,
										(void *)&uiOperator,
										0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}

			// Insert a left paren.

			if (RC_BAD( rc = fcsTranslateQFlmToQCSOp(	FLM_LPAREN_OP, &uiOperator)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = gedAddField( pPool, pRootNode,
										FCS_ITERATOR_OPERATOR,
										(void *)&uiOperator,
										0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}
			
			pQNode = pQNode->pChild;
			continue;
		}
		else
		{
			// Output whatever is in the node at this point.

			if (IS_OP( eOp))
			{
				if( RC_BAD( rc = fcsTranslateQFlmToQCSOp(	eOp, &uiOperator)))
				{
					goto Exit;
				}

				if (RC_BAD( rc = gedAddField( pPool, pRootNode,
											FCS_ITERATOR_OPERATOR,
											(void *)&uiOperator,
											0, FLM_NUMBER_TYPE)))
				{
					goto Exit;
				}
			}
			else if (IS_VAL( eOp))
			{
				switch (eOp)
				{
					case FLM_UINT32_VAL:
						if (RC_BAD( rc = gedAddField( pPool, pRootNode,
													FCS_ITERATOR_NUMBER_VALUE,
													(void *)&pQNode->pQAtom->val.ui32Val,
													4, FLM_NUMBER_TYPE)))
						{
							goto Exit;
						}
						break;
					case FLM_UINT64_VAL:
						if (RC_BAD( rc = gedAddField( pPool, pRootNode,
													FCS_ITERATOR_NUMBER_VALUE,
													(void *)&pQNode->pQAtom->val.ui64Val,
													8, FLM_NUMBER_TYPE)))
						{
							goto Exit;
						}
						break;
					case FLM_INT32_VAL:
						if ((pFldNode = GedNodeMake( pPool,
												FCS_ITERATOR_NUMBER_VALUE,
												&rc)) == NULL)
						{
							goto Exit;
						}
						GedChildGraft( pRootNode, pFldNode, GED_LAST);
						if (RC_BAD( rc = GedPutINT( pPool, pFldNode,
													(FLMINT)pQNode->pQAtom->val.i32Val)))
						{
							goto Exit;
						}
						break;
					case FLM_INT64_VAL:
						if ((pFldNode = GedNodeMake( pPool,
												FCS_ITERATOR_NUMBER_VALUE,
												&rc)) == NULL)
						{
							goto Exit;
						}
						GedChildGraft( pRootNode, pFldNode, GED_LAST);
						if (RC_BAD( rc = GedPutINT64( pPool, pFldNode,
													pQNode->pQAtom->val.i64Val)))
						{
							goto Exit;
						}
						break;
					case FLM_REC_PTR_VAL:
						if (RC_BAD( rc = gedAddField( pPool, pRootNode,
													FCS_ITERATOR_REC_PTR_VALUE,
													(void *)&pQNode->pQAtom->val.ui32Val,
													4, FLM_NUMBER_TYPE)))
						{
							goto Exit;
						}
						break;
					case FLM_STRING_VAL:
						if ((pFldNode = GedNodeMake( pPool,
												FCS_ITERATOR_NATIVE_VALUE,
												&rc)) == NULL)
						{
							goto Exit;
						}
						GedChildGraft( pRootNode, pFldNode, GED_LAST);
						if (RC_BAD( rc = GedPutNATIVE( pPool, pFldNode,
							(const char *)pQNode->pQAtom->val.pucBuf)))
						{
							goto Exit;
						}
						break;
					case FLM_BINARY_VAL:
						if ((pFldNode = GedNodeMake( pPool,
												FCS_ITERATOR_BINARY_VALUE,
												&rc)) == NULL)
						{
							goto Exit;
						}
						GedChildGraft( pRootNode, pFldNode, GED_LAST);
						if (RC_BAD( rc = GedPutBINARY( pPool, pFldNode,
													pQNode->pQAtom->val.pucBuf,
													pQNode->pQAtom->uiBufLen)))
						{
							goto Exit;
						}
						break;
					case FLM_TEXT_VAL:
						if ((pFldNode = GedNodeMake( pPool,
												FCS_ITERATOR_FLM_TEXT_VALUE,
												&rc)) == NULL)
						{
							goto Exit;
						}
						GedChildGraft( pRootNode, pFldNode, GED_LAST);
						if (RC_BAD( rc = GedPutBINARY( pPool, pFldNode,
													pQNode->pQAtom->val.pucBuf,
													pQNode->pQAtom->uiBufLen)))
						{
							goto Exit;
						}
						break;
					case FLM_UNICODE_VAL:
						if ((pFldNode = GedNodeMake( pPool,
												FCS_ITERATOR_UNICODE_VALUE,
												&rc)) == NULL)
						{
							goto Exit;
						}
						GedChildGraft( pRootNode, pFldNode, GED_LAST);
						if (RC_BAD( rc = GedPutUNICODE( pPool, pFldNode,
													(FLMUNICODE *)pQNode->pQAtom->val.pucBuf)))
						{
							goto Exit;
						}
						break;
					default:
						flmAssert( 0);
						rc = RC_SET( FERR_NOT_IMPLEMENTED);
						goto Exit;
				}
			}
			else
			{
				FLMUINT *	puiPath = pQNode->pQAtom->val.QueryFld.puiFldPath;
				FLMUINT		uiPathLen = 0;

				while (*puiPath)
				{
					uiPathLen++;
					puiPath++;
				}
				if (uiPathLen == 1)
				{
					if ((pFldNode = GedNodeMake( pPool,
											FCS_ITERATOR_ATTRIBUTE,
											&rc)) == NULL)
					{
						goto Exit;
					}
					GedChildGraft( pRootNode, pFldNode, GED_LAST);
					puiPath--;
					if (RC_BAD( rc = GedPutUINT( pPool, pFldNode, *puiPath)))
					{
						goto Exit;
					}
				}
				else
				{
					if ((pFldNode = GedNodeMake( pPool,
											FCS_ITERATOR_ATTRIBUTE_PATH,
											&rc)) == NULL)
					{
						goto Exit;
					}
					GedChildGraft( pRootNode, pFldNode, GED_LAST);
					while (uiPathLen)
					{
						uiPathLen--;
						puiPath--;
						if (RC_BAD( rc = gedAddField( pPool, pFldNode,
													FCS_ITERATOR_ATTRIBUTE,
													(void *)puiPath,
													0, FLM_NUMBER_TYPE)))
						{
							goto Exit;
						}
					}
				}
			}

			// See if the node has a sibling we should traverse down.

Test_Sib:
			if (pQNode->pNextSib)
			{
				QTYPES	eParentOp = GET_QNODE_TYPE( (pQNode->pParent));

				// If we have a sibling, the parent MUST be a binary operator.
				// Output the operator

				if( RC_BAD( rc = fcsTranslateQFlmToQCSOp(	eParentOp, &uiOperator)))
				{
					goto Exit;
				}
				
				if (RC_BAD( rc = gedAddField( pPool, pRootNode,
											FCS_ITERATOR_OPERATOR,
											(void *)&uiOperator,
											0, FLM_NUMBER_TYPE)))
				{
					goto Exit;
				}
				
				pQNode = pQNode->pNextSib;
				continue;
			}

			if ((pQNode = pQNode->pParent) == NULL)
			{
				break;
			}

			// Insert a right paren.

			if (RC_BAD( rc = fcsTranslateQFlmToQCSOp(	FLM_RPAREN_OP, &uiOperator)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = gedAddField( pPool, pRootNode,
										FCS_ITERATOR_OPERATOR,
										(void *)&uiOperator,
										0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}

			goto Test_Sib;
		}
	}

	if (RC_BAD( rc = pWire->sendHTD( WIRE_VALUE_ITERATOR_WHERE, pRootNode)))
	{
		pCSContext->bConnectionGood = FALSE;
		goto Exit;
	}

Exit:

	pPool->poolReset( pvMark);
	return( rc);
}

/****************************************************************************
Desc:	Initialize a query over the client/server line.
****************************************************************************/
RCODE flmInitCurCS(
	CURSOR *	pCursor
	)
{
	RCODE				rc = FERR_OK;
	CS_CONTEXT *	pCSContext = pCursor->pCSContext;
	FCL_WIRE			Wire( pCSContext);

	if (pCursor->uiCursorId != FCS_INVALID_ID)
	{
		goto Exit;		// Returns SUCCESS;
	}

	// Send a request to create an iterator for this cursor.

	if (RC_BAD( rc = Wire.sendOp(
		FCS_OPCLASS_ITERATOR, FCS_OP_ITERATOR_INIT)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = flmSendCursorFrom( &Wire, pCursor)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = flmSendCursorWhere( &Wire, pCursor)))
	{
		goto Exit;
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
		goto Exit;
	}

	pCursor->uiCursorId = Wire.getIteratorId();

Exit:
	return( rc);

Transmission_Error:
	pCursor->pCSContext->bConnectionGood = FALSE;
	goto Exit;
}

/****************************************************************************
Desc:		Validates the selection criteria of a cursor.
Notes:	It is not necessary to explicitly validate the selection criteria
			through a call to this routine.  FLAIM will automatically attempt
			validation on the first call to any of the cursor routines which
			make use of the criteria.  Although explicit validation is 
			unnecessary, it can be convenient to identify an error in the
			selection criteria before calling cursor routines which will make
			use of it.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmCursorValidate(
	HFCURSOR		hCursor)
{
	RCODE			rc = FERR_OK;
	CURSOR *		pCursor = (CURSOR *)hCursor;

	if (!pCursor)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_INVALID_PARM);
		goto Exit;
	}
	if (pCursor->pCSContext)
	{
		rc = flmInitCurCS( pCursor);
		goto Exit2;
	}

	// Validate the query by optimizing it.

	if( !pCursor->bOptimized)
	{
		rc = flmCurPrep( pCursor);
	}

Exit:
Exit2:

	return( pCursor->rc = rc);
}
