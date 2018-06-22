//-------------------------------------------------------------------------
// Desc:	GEDCOM routines
// Tabs:	3
//
// Copyright (c) 1990-2007 Novell, Inc. All Rights Reserved.
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

extern FLMBYTE arr[];

#define BINARY_GED_HEADER_LEN 8

static FLMBYTE FlmBinaryGedHeader[BINARY_GED_HEADER_LEN] =
{
	0xFF,
	'F',
	'L',
	'M',
	'D',
	'I',
	'C',
	'T'
};

static FLMBYTE FlmBinaryRecHeader[BINARY_GED_HEADER_LEN] =
{
	0xFF,
	'F',
	'L',
	'M',
	'R',
	'E',
	'C',
	'S'
};

#define NODE_DRN_POS			0
#define NODE_CONTAINER_POS ( NODE_DRN_POS + sizeof(FLMUINT))
#define NODE_DB_POS			( NODE_CONTAINER_POS + sizeof(FLMUINT))

#define f_isdigit(c) \
		((c) < 60 ? ((((FLMBYTE) (arr[(c) >> 3])) << ((c) & 0x07)) & 0x80) : 0)

FSTATIC RCODE expWrite(
	EXP_IMP_INFO *		pExpImpInfo,
	const FLMBYTE *	pData,
	FLMUINT				uiDataLen);

FSTATIC RCODE impRead(
	EXP_IMP_INFO *		pExpImpInfo,	
	FLMBYTE *			pData,
	FLMUINT				uiDataLen,
	FLMUINT *			puiBytesReadRV);

FSTATIC RCODE tagValLenType(
	F_Pool *				pPool,
	GED_STREAM *		x,
	NODE **			 	ppNode,
	F_NameTable *		pNameTable);

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE GedGetBINARY(
	NODE *			pNode,
	void *			buffer,
	FLMUINT *		bufLenRV)
{
	RCODE				rc = FERR_OK;
	FLMBYTE *		ptr;
	FLMUINT			valLength;
	FLMUINT			outputData;
	FLMUINT			uiNodeType;

	// Check for a null node

	if (!pNode)
	{
		rc = RC_SET( FERR_CONV_NULL_SRC);
		goto Exit;
	}

	if (pNode->ui32EncId)
	{
		if (!(pNode->ui32EncFlags & FLD_HAVE_DECRYPTED_DATA))
		{
			rc = RC_SET( FERR_FLD_NOT_DECRYPTED);
			goto Exit;
		}
	}

	// If the node is not a BINARY node, return an error

	uiNodeType = GedValType( pNode);
	if (uiNodeType != FLM_BINARY_TYPE)
	{
		rc = RC_SET( FERR_CONV_ILLEGAL);
		goto Exit;
	}

	ptr = (FLMBYTE *) GedValPtr( pNode);
	valLength = GedValLen( pNode);

	// At this point we know the node is a BINARY node

	outputData = ((buffer != NULL) && (*bufLenRV));
	if ((outputData) && (valLength))
	{
		if (valLength > *bufLenRV)
		{
			rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
			goto Exit;
		}

		f_memcpy( buffer, ptr, valLength);
	}

	*bufLenRV = valLength;

Exit:

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE GedPutBINARY(
	F_Pool *			pPool,
	NODE *			pNode,
	const void *	pvData,
	FLMUINT			uiDataLen,
	FLMUINT			uiEncId,
	FLMUINT			uiEncSize)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	outPtr;

	// Check for a null node being passed in

	if (!pNode)
	{
		rc = RC_SET( FERR_CONV_NULL_DEST);
		goto Exit;
	}

	// If data pointer is NULL or length is zero, call GedAllocSpace with
	// a length of zero to set the node length to zero and node type to
	// FLM_BINARY_TYPE.

	if (pvData == NULL || !uiDataLen)
	{
		(void) GedAllocSpace( pPool, pNode, FLM_BINARY_TYPE, 
					0, uiEncId, uiEncSize);
		goto Exit;
	}

	// Allocate space in the node for the binary data

	if ((outPtr = (FLMBYTE *) GedAllocSpace( pPool, pNode, FLM_BINARY_TYPE,
											uiDataLen, uiEncId, uiEncSize)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Set the node type and copy the data into the node

	f_memcpy( outPtr, pvData, uiDataLen);

	if (pNode->ui32EncId)
	{
		pNode->ui32EncFlags = FLD_HAVE_DECRYPTED_DATA;
	}

Exit:

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE GedToTree(
	F_Pool *				pPool,
	IF_FileHdl *		pFileHdl,
	char **				pBuf,
	FLMUINT				uiBufSize,
	NODE **				ppRoot,
	F_NameTable *		pNameTable)
{
	RCODE				rc = FERR_OK;
	GED_STREAM		gedStream;
	FLMUINT			level;
	FLMUINT			levelBase = 0;
	FLMUINT			levelPrior = 0;
	FLMBYTE			nextChar;
	NODE *			nd = NULL;
	NODE *			ndPrior = NULL;
	FLMUINT64		ui64StartPos;

	gedStream.pFileHdl = pFileHdl;
	gedStream.pThis = gedStream.pBuf = *pBuf;
	gedStream.uiBufSize = uiBufSize;

	if (pFileHdl)
	{

		// Find 1st starting file position

		if (RC_OK( pFileHdl->seek( 0, FLM_IO_SEEK_CUR, 
				&gedStream.ui64FilePos)))
		{
			gedStream.pLast = gedStream.pBuf;
			gedReadChar( &gedStream, gedStream.ui64FilePos);
		}
		else
		{
			return (RC_SET( FERR_FILE_ER));
		}
	}
	else
	{
		gedStream.errorIO = 0;
		gedStream.ui64FilePos = 0;
		gedStream.pLast = gedStream.pBuf + (uiBufSize - 1);
		gedStream.thisC = f_toascii( *gedStream.pBuf);
	}

	for (;;)
	{
		gedSkipBlankLines( &gedStream);
		ui64StartPos = gedStream.ui64FilePos;

		if (f_isdigit( gedStream.thisC))
		{
			level = 0;
			do
			{
				level = gedStream.thisC - ASCII_ZERO + (level * 10);
				nextChar = (FLMBYTE) (gedNextChar( &gedStream));
			} while (f_isdigit( nextChar));

			if (!f_iswhitespace( gedStream.thisC))
			{
				rc = RC_SET( FERR_BAD_FIELD_LEVEL);
				break;
			}

			if (level > GED_MAXLVLNUM)
			{
				rc = RC_SET( FERR_GED_MAXLVLNUM);
				break;
			}

			if (ndPrior)
			{
				if (levelBase >= level)
				{
					goto successful;
				}
				else if ((levelPrior < level) && ((levelPrior + 1) != level))
				{
					rc = RC_SET( FERR_GED_SKIP_LEVEL);
					break;
				}
			}
			else
			{
				levelBase = level;
			}

			levelPrior = level;

			if( RC_OK( rc = tagValLenType( pPool, &gedStream, &nd, pNameTable)))
			{
				if (ndPrior)
				{
					ndPrior->next = nd;
				}
				else
				{
					*ppRoot = nd;
				}

				nd->prior = ndPrior;
				GedNodeLevelSet( nd, level - levelBase);
				ndPrior = nd;
				continue;
			}
		}
		else if (gedStream.thisC == '\0' || gedStream.thisC == ASCII_CTRLZ)
		{
			if (gedStream.errorIO)
			{
				rc = RC_SET( FERR_FILE_ER);
			}
			else if (ndPrior)
			{
successful:

				ndPrior->next = NULL;
				if( !pFileHdl)
				{
					*pBuf = gedStream.pThis + 
							  (FLMINT32) (ui64StartPos - gedStream.ui64FilePos);
				}

				gedStream.ui64FilePos = ui64StartPos;
				rc = FERR_OK;
			}
			else
			{
				rc = RC_SET( FERR_END);
			}
		}
		else
		{
			rc = RC_SET( FERR_BAD_FIELD_LEVEL);
		}
		
		break;
	}

	if (RC_BAD( rc))
	{
		*ppRoot = NULL;
		
		if (pFileHdl == NULL)
		{
			*pBuf = gedStream.pThis;
		}
	}

	if (pFileHdl)
	{
		pFileHdl->seek( gedStream.ui64FilePos, FLM_IO_SEEK_SET,
			&gedStream.ui64FilePos);
	}

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
FSTATIC RCODE tagValLenType(
	F_Pool *				pPool,
	GED_STREAM *		pGedStream,
	NODE **			 	newNode,
	F_NameTable *		pNameTable)
{
	FLMUINT64		ui64StartPos;
	RCODE				rc = FERR_OK;
	NODE *			nd;
	FLMUINT			drn = 0;
	FLMUINT			uiTagNum;
	char				tagBuf[ GED_MAXTAGLEN + 1];

	gedSkipWhiteSpaces( pGedStream);

	ui64StartPos = pGedStream->ui64FilePos;
	if (pGedStream->thisC == ASCII_AT)
	{
		int	badDRN;
		
		for (badDRN = 0, gedNextChar( pGedStream); 
			  pGedStream->thisC != ASCII_AT; 
			  gedNextChar( pGedStream))
		{
			FLMUINT	priorDrn = drn;

			if (!badDRN)
			{
				if (f_isdigit( pGedStream->thisC))
				{
					drn = (drn * 10) + pGedStream->thisC - ASCII_ZERO;
					badDRN = priorDrn != (drn / 10);
				}
				else
				{
					badDRN = 1;
				}
			}
		}

		if (badDRN)
		{
			drn = 0;
		}

		gedNextChar( pGedStream);
		if (f_iswhitespace( pGedStream->thisC))
		{
			gedSkipWhiteSpaces( pGedStream);
		}
		else
		{
			rc = RC_SET( FERR_GED_BAD_RECID);
			goto Exit;
		}
	}

	// Determine the Tag Number and Build the NODE

	ui64StartPos = pGedStream->ui64FilePos;

	if (!gedCopyTag( pGedStream, tagBuf))
	{
		return (RC_SET( FERR_INVALID_TAG));
	}

	if (!pNameTable->getFromTagTypeAndName( NULL, tagBuf, FLM_FIELD_TAG,
														&uiTagNum))
	{

		// See if tag is the reserved tag with the number following

		if (tagBuf[0] == f_toascii( 'T') &&
			 tagBuf[1] == f_toascii( 'A') &&
			 tagBuf[2] == f_toascii( 'G') &&
			 tagBuf[3] == f_toascii( '_'))
		{
			uiTagNum = f_atoi( &tagBuf[4]);
		}
		else
		{
			return (RC_SET( FERR_NOT_FOUND));
		}
	}

	if ((*newNode = nd = GedNodeCreate( pPool, uiTagNum, drn, &rc)) == NULL)
	{
		goto Exit;
	}

	gedSkipWhiteSpaces( pGedStream);

	ui64StartPos = pGedStream->ui64FilePos;
	if (pGedStream->thisC == ASCII_AT)
	{
		for (drn = 0; gedNextChar( pGedStream) != ASCII_AT;)
		{
			FLMUINT	priorDrn = drn;
			
			if (f_isdigit( pGedStream->thisC))
			{
				drn = (drn * 10) + pGedStream->thisC - ASCII_ZERO;
				if (priorDrn == (drn / 10))
				{
					continue;
				}
			}

			rc = RC_SET( FERR_GED_BAD_VALUE);
			goto Exit;
		}

		gedNextChar( pGedStream);
		GedPutRecPtr( pPool, nd, drn);
		if (gedCopyValue( pGedStream, NULL))
		{
			rc = RC_SET( FERR_GED_BAD_VALUE);
			goto Exit;
		}
	}
	else
	{
		FLMINT		valLength;
		FLMUINT64	ui64TempPos = pGedStream->ui64FilePos;

		if ((valLength = gedCopyValue( pGedStream, NULL)) > 0)
		{
			char *	vp = (char *) GedAllocSpace( pPool, nd, FLM_TEXT_TYPE,
															 valLength);

			if (vp)
			{
				gedReadChar( pGedStream, ui64TempPos);
				gedCopyValue( pGedStream, vp);
			}
			else
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}
		}
	}

	ui64StartPos = pGedStream->ui64FilePos;

Exit:

	gedReadChar( pGedStream, ui64StartPos);
	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
NODE * GedNodeCopy(
	F_Pool *		pPool,
	NODE *		pNode,
	NODE *		pChildList,
	NODE *		pSibList)
{
	NODE *		newNd;
	FLMUINT		bias;
	FLMBYTE *	vp;
	RCODE			rc;
	HFDB			hDb;
	FLMUINT		uiContainer;
	FLMUINT		uiRecId;

	// If the node has source information, we need to copy it

	if (RC_OK( GedGetRecSource( pNode, &hDb, &uiContainer, &uiRecId)))
	{

		// The passed in node contains record source information, so create
		// a GEDCOM record source node

		if (RC_BAD( gedCreateSourceNode( pPool, GedTagNum( pNode), hDb,
					  uiContainer, uiRecId, &newNd)))
		{
			return (NULL);
		}
	}
	else
	{

		// Create a normal (non-source) GEDCOM node

		if ((newNd = GedNodeMake( pPool, GedTagNum( pNode), &rc)) == NULL)
		{
			return (NULL);
		}
	}

	newNd->prior = NULL;
	newNd->next = pChildList;
	GedNodeLevelSet( newNd, 0);

	if ((vp = (FLMBYTE *) GedAllocSpace( pPool, newNd, GedValType( pNode),
		GedValLen( pNode), pNode->ui32EncId, GedEncLen( pNode))) != NULL)
	{
		f_memcpy( vp, GedValPtr( pNode), GedValLen( pNode));

		if (pNode->ui32EncFlags & FLD_HAVE_ENCRYPTED_DATA)
		{
			f_memcpy( GedEncPtr( newNd), GedEncPtr( pNode), GedEncLen( pNode));
		}

		newNd->ui32EncFlags = pNode->ui32EncFlags;
	}
	else
	{
		return (NULL);
	}

	if (pChildList)
	{
		pChildList->prior = newNd;
		for (bias = GedNodeLevel( pChildList) - 1;
		 	  pChildList->next; 
			  GedNodeLevelSub( pChildList, bias), pChildList = pChildList->next);
			  
		GedNodeLevelSub( pChildList, bias);
		pChildList->next = pSibList;
	}
	else
	{
		pChildList = newNd;
	}

	if (pSibList)
	{
		pSibList->prior = pChildList;
		pChildList->next = pSibList;
		for (bias = GedNodeLevel( pSibList); pSibList->next; 
			  GedNodeLevelSub( pSibList, bias), pSibList = pSibList->next);
		GedNodeLevelSub( pSibList, bias);
	}

	return (newNd);
}

/*****************************************************************************
Desc:
*****************************************************************************/
NODE * GedCopy(
	F_Pool *		pPool,
	FLMUINT		cnt,
	NODE *		tree)
{
	NODE *		oldNd;
	NODE *		newNd;
	NODE *		newRoot;
	FLMUINT		baseLevel;

	if (tree)
	{
		newRoot = newNd = GedNodeCopy( pPool, tree, NULL, NULL);
		if (newRoot)
		{
			for (baseLevel = GedNodeLevel( tree);
				  (tree = tree->next) != NULL && (GedNodeLevel( tree) > baseLevel ||
				  (GedNodeLevel( tree) == baseLevel && --cnt));)
			{
				oldNd = newNd;
				if ((newNd = GedNodeCopy( pPool, tree, NULL, NULL)) != NULL)
				{
					oldNd->next = newNd;
					newNd->prior = oldNd;
					GedNodeLevelSet( newNd, GedNodeLevel( tree) - baseLevel);
				}
				else
				{
					return (NULL);
				}
			}
		}

		return (newRoot);
	}

	return (NULL);
}

/*****************************************************************************
Desc:
*****************************************************************************/
NODE * GedClip(
	FLMUINT		treeCnt,
	NODE *		self)
{
	NODE *		next;

	if (self)
	{
		FLMUINT	oldLevel = GedNodeLevel( self);

		GedNodeLevelSet( self, 0);

		for (next = self->next; next && 
			 (GedNodeLevel( next) > oldLevel || 
			 	(GedNodeLevel( next) == oldLevel && --treeCnt));
			 GedNodeLevelSub( next, oldLevel), next = next->next)
		{
			;
		}

		if (self->prior)
		{
			self->prior->next = next;
		}

		if (next)
		{
			next->prior->next = NULL;
			next->prior = self->prior;
		}

		self->prior = NULL;
	}

	return (self);
}

/*****************************************************************************
Desc:
*****************************************************************************/
NODE * GedFind(
	FLMUINT		treeCnt,
	NODE *		nd,
	FLMUINT		tnum,
	FLMINT		nth)
{
	if (nd)
	{
		FLMUINT	strtLvl = GedNodeLevel( nd);
		do
		{
			if ((tnum == GedTagNum( nd)) && (--nth < 1))
			{
				return (nd);
			}
		} while( (nd = nd->next) != NULL && 
			(GedNodeLevel( nd) > strtLvl || 
				(--treeCnt && GedNodeLevel( nd) == strtLvl)));
	}

	return (NULL);
}

/*****************************************************************************
Desc:
*****************************************************************************/
NODE * GedPathFind(
	FLMUINT			treeCnt,
	NODE *			nd,
	FLMUINT *		puiPathArray,
	FLMINT			nth)
{
	NODE *			pNode = nd;
	NODE *			savenode;
	FLMUINT *		path;

	if (nd && puiPathArray)
	{
		FLMUINT	uiLevel = GedNodeLevel( nd);
		
		for (;;)
		{
			path = puiPathArray + (GedNodeLevel( pNode) - uiLevel);
			savenode = pNode;
			if (*path == GedTagNum( pNode))
			{
				if (*(path + 1) == 0 && (--nth < 1))
				{
					return (pNode);
				}

				if ((pNode = GedChild( pNode)) != NULL)
				{
					continue;
				}

				pNode = savenode;
			}

			do
			{
				pNode = pNode->next;
			} while (pNode != NULL && GedNodeLevel( pNode) > GedNodeLevel( savenode));

			// find next sibling/uncle/end

			if (!pNode || GedNodeLevel( pNode) < uiLevel ||
				 (GedNodeLevel( pNode) == uiLevel && !(--treeCnt)))
			{
				break;
			}
		}
	}

	return (NULL);
}

/*****************************************************************************
Desc:
*****************************************************************************/
NODE * GedChildGraft(
	NODE *		parent,
	NODE *		child,
	FLMINT		nth)
{
	NODE *		lastChildNode;

	if (parent && child)
	{
		FLMINT	level = GedNodeLevel( parent) + 1;

		if (GedChild( parent))
		{
			GedSibGraft( GedChild( parent), child, 
				(FLMINT) (nth == GED_FIRST ? GED_FIRST : nth - 1));
		}
		else
		{
			for (lastChildNode = child;
				  lastChildNode->next;
				  GedNodeLevelAdd( lastChildNode, level),
				  lastChildNode = lastChildNode->next);
				  
			child->prior = parent;
			GedNodeLevelAdd( lastChildNode, level);
			lastChildNode->next = parent->next;
			if (parent->next)
			{
				parent->next->prior = lastChildNode;
			}

			parent->next = child;
		}
	}

	return (parent);
}

/*****************************************************************************
Desc:
*****************************************************************************/
NODE * GedSibGraft(
	NODE *		self,
	NODE *		sib,
	FLMINT		nth)
{
	NODE *		lastSibNode;
	NODE *		returnNode;
	FLMINT		deltaLevel;
	FLMUINT		level;
	FLMUINT		linkAt = TRUE;

	if (!sib)
	{
		return (self);
	}

	if (!self)
	{
		return (sib);
	}

	for( level = GedNodeLevel( self), 
		  deltaLevel = (FLMINT) (level - GedNodeLevel( sib)), 
		  lastSibNode = sib; lastSibNode->next; 
		  GedNodeLevelAdd( lastSibNode, deltaLevel), 
		  lastSibNode = lastSibNode->next);

	GedNodeLevelAdd( lastSibNode, deltaLevel);
	if (nth != GED_LAST)
	{
		nth++;
	}

	if (nth <= 0)
	{
		returnNode = sib;
		while (nth)
		{
			if (self->prior)
			{
				self = self->prior;
				if (GedNodeLevel( self) > level)
				{
					continue;
				}
				else if (GedNodeLevel( self) == level)
				{
					nth++;
					continue;
				}

				self = self->next;
			}
			break;
		}
	}
	else
	{
		returnNode = self;
		while (nth)
		{
			if (self->next)
			{
				self = self->next;
				if (GedNodeLevel( self) > level)
				{
					continue;
				}
				else if (GedNodeLevel( self) == level)
				{
					nth--;
					continue;
				}

				self = self->prior;
			}

			linkAt = FALSE;
			break;
		}
	}

	if (linkAt)
	{

		// Link the sib tree AT the current self location - link before self

		sib->prior = self->prior;
		lastSibNode->next = self;
		if (self->prior)
		{
			self->prior->next = sib;
		}

		self->prior = lastSibNode;
	}
	else
	{

		// Link the sib tree AFTER the current self location

		sib->prior = self;
		lastSibNode->next = self->next;
		if (self->next)
		{
			self->next->prior = lastSibNode;
		}

		self->next = sib;
	}

	return (returnNode);
}

/*****************************************************************************
Desc:
*****************************************************************************/
NODE * GedNodeCreate(
	F_Pool *		pPool,
	FLMUINT		tagNum,
	FLMUINT		id,
	RCODE *		rc)
{
	NODE *		nd;
	
	*rc = FERR_OK;
	
	if( RC_BAD( *rc = pPool->poolAlloc( (sizeof(NODE) + (id ? sizeof(id) : 0)),
		(void **)&nd)))
	{
		goto Exit;
	}

	f_memset( nd, '\0', sizeof(NODE));

	GedValTypeSet( nd, FLM_CONTEXT_TYPE);
	GedTagNumSet( nd, tagNum);

	if (id)
	{
		FLMBYTE *		ptr;
		
		GedValTypeSetFlag( nd, HAS_REC_ID);
		ptr = ((FLMBYTE *) nd) + sizeof(NODE);
		*((FLMUINT *) (ptr + NODE_DRN_POS)) = id;
	}

Exit:
	
	return (nd);
}

/*****************************************************************************
Desc:
*****************************************************************************/
void * GedAllocSpace(
	F_Pool *			pPool,
	NODE *			pNode,
	FLMUINT			valType,
	FLMUINT			size,
	FLMUINT			uiEncId,
	FLMUINT			uiEncSize)
{
	FLMBYTE *		rPtr;
	FLMUINT			uiAllocSize = size;

	if (valType == FLM_TEXT_TYPE)
	{
		uiAllocSize++;
	}

	if (uiAllocSize <= sizeof(void *))
	{

		// If the size is less than sizeof (void *), we use the space right
		// inside value pointer itself.
		
		rPtr = (FLMBYTE *) &pNode->value;
	}
	else if (size <= GedValLen( pNode))
	{

		// If there is already allocated space, just re-use it

		rPtr = (FLMBYTE *) GedValPtr( pNode);
	}
	else
	{
		if( RC_BAD( pPool->poolAlloc( uiAllocSize, (void **)&rPtr)))
		{
			pNode->ui32Length = 0;
			pNode->value = NULL;
			return (NULL);
		}

		pNode->value = rPtr;
	}

	if (valType == FLM_TEXT_TYPE)
	{
		rPtr[size] = '\0';
	}

	// Now set the size and the data type

	pNode->ui32Length = (FLMUINT32) size;
	GedSetType( pNode, valType);

	// If passed-in enc id is zero, use the node's enc id.

	if (!uiEncId)
	{
		flmAssert( !uiEncSize);
		if (size)
		{
			uiEncId = pNode->ui32EncId;
			uiEncSize = size + (16 - (size % 16));
		}
	}
	else
	{

		// We only should have an encryption ID if size is non-zero. If
		// size is non-zero, encryption size must also be non-zero.

		flmAssert( size);
		flmAssert( uiEncSize);
	}

	if (uiEncId)
	{
		if (uiEncSize > GedEncLen( pNode))
		{
			if( RC_BAD( pPool->poolAlloc( uiEncSize, 
				(void **)&pNode->pucEncValue)))
			{
				pNode->ui32EncLength = 0;
				pNode->pucEncValue = NULL;
				return (NULL);
			}
		}

		pNode->ui32EncFlags = FLD_HAVE_DECRYPTED_DATA | FLD_HAVE_ENCRYPTED_DATA;
		pNode->ui32EncId = (FLMUINT32) uiEncId;
		pNode->ui32EncLength = (FLMUINT32) uiEncSize;
	}

	return (rPtr);
}

/*****************************************************************************
Desc:
*****************************************************************************/
void * GedValPtr(
	NODE *		nd)
{
	return( nd && nd->ui32Length 
					? GedValType( nd) == FLM_TEXT_TYPE 
							? nd->ui32Length < sizeof(void *) 
									? (void *) &nd->value 
									: (void *) nd->value 
							: nd->ui32Length > sizeof(void *)
								? (void *) nd->value
								: (void *) &nd->value
					: NULL);
}

/*****************************************************************************
Desc:
*****************************************************************************/
void * GedEncPtr(
	NODE *		nd)
{
	return( nd && nd->ui32EncLength 
					? (void *) nd->pucEncValue 
					: (void *) NULL);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE GedPutRecId(
	F_Pool *		pPool,
	NODE **	 	ppNd,
	FLMUINT		uiId)
{
	NODE *			pNewNd;
	NODE *			pOldNd = *ppNd;
	FLMBYTE *		ptr;

	if( RC_BAD( pPool->poolAlloc( sizeof(NODE) + sizeof(uiId), 
		(void **)&pNewNd))) 
	{
		*ppNd = NULL;
		return (RC_SET( FERR_MEM));
	}

	// Copy the contents of the existing node

	pNewNd->prior = pOldNd->prior;
	pNewNd->next = pOldNd->next;
	pNewNd->value = pOldNd->value;
	pNewNd->ui32Length = pOldNd->ui32Length;
	pNewNd->ui32EncId = pOldNd->ui32EncId;
	pNewNd->ui32EncLength = pOldNd->ui32EncLength;
	pNewNd->ui32EncFlags = pOldNd->ui32EncFlags;
	pNewNd->pucEncValue = pOldNd->pucEncValue;
	GedTagNumSet( pNewNd, GedTagNum( pOldNd));
	GedNodeLevelSet( pNewNd, GedNodeLevel( pOldNd));
	GedNodeTypeSet( pNewNd, (GedNodeType( pOldNd) | HAS_REC_ID));

	// Link in new node to parent and children/siblings

	if (pNewNd->prior)
	{
		pNewNd->prior->next = pNewNd;
	}

	if (pNewNd->next)
	{
		pNewNd->next->prior = pNewNd;
	}

	// Set the Ids value

	ptr = (FLMBYTE *) GedIdPtr( pNewNd);
	*((FLMUINT *) (ptr + NODE_DRN_POS)) = uiId;
	*ppNd = pNewNd;

	return (FERR_OK);
}

/*****************************************************************************
Desc:
*****************************************************************************/
void gedSetRecSource(
	NODE *	pNode,
	HFDB		hDb,
	FLMUINT	uiContainer,
	FLMUINT	uiDrn)
{
	FLMBYTE *		pucPtr;

	pucPtr = ((FLMBYTE *) pNode) + sizeof(NODE);
	if (uiDrn)
	{
		GedValTypeSetFlag( pNode, HAS_REC_ID);
		*((FLMUINT *) (pucPtr + NODE_DRN_POS)) = uiDrn;
	}

	if (uiContainer)
	{
		GedValTypeSetFlag( pNode, HAS_REC_SOURCE);
		*((FLMUINT *) (pucPtr + NODE_CONTAINER_POS)) = uiContainer;
	}

	if (hDb)
	{
		GedValTypeSetFlag( pNode, HAS_REC_SOURCE);
		*((HFDB *) (pucPtr + NODE_DB_POS)) = hDb;
	}
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE GedGetRecSource(
	NODE *			pNode,
	HFDB *			phDb,
	FLMUINT *		puiContainer,
	FLMUINT *		puiRecId)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	ptr = ((FLMBYTE *) pNode) + sizeof(NODE);

	if (GedNodeType( pNode) & HAS_REC_SOURCE)
	{
		if (phDb)
		{
			*phDb = *((HFDB *) (ptr + NODE_DB_POS));
		}

		if (puiContainer)
		{
			*puiContainer = *((FLMUINT *) (ptr + NODE_CONTAINER_POS));
		}

		if (puiRecId)
		{
			*puiRecId = *((FLMUINT *) (ptr + NODE_DRN_POS));
		}
	}
	else if (GedNodeType( pNode) & HAS_REC_ID)
	{
		if (phDb)
		{
			*phDb = NULL;
		}

		if (puiContainer)
		{
			*puiContainer = 0;
		}

		if (puiRecId)
		{
			*puiRecId = *((FLMUINT *) (ptr + NODE_DRN_POS));
		}
	}
	else
	{	
		// The record contains no record source, because the user may ignore
		// the return code lets make sure everything is set to null.
		
		if (phDb)
		{
			*phDb = NULL;
		}

		if (puiContainer)
		{
			*puiContainer = 0;
		}

		if (puiRecId)
		{
			*puiRecId = 0;
		}

		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

Exit:

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE GedPutRecPtr(
	F_Pool *		pPool,
	NODE *		nd,
	FLMUINT		drn,
	FLMUINT		uiEncId,
	FLMUINT		uiEncSize)
{
	void *		ptr;
	RCODE			rc = FERR_OK;

	// Check for a null node being passed in

	if (nd == NULL)
	{
		rc = RC_SET( FERR_CONV_NULL_DEST);
		goto Exit;
	}

	if ((ptr = GedAllocSpace( pPool, nd, FLM_CONTEXT_TYPE, sizeof(FLMUINT32),
										uiEncId, uiEncSize)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	UD2FBA( (FLMUINT32) drn, (FLMBYTE *)ptr);

	if (nd->ui32EncId)
	{
		nd->ui32EncFlags = FLD_HAVE_DECRYPTED_DATA;
	}

Exit:

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE GedGetRecPtr(
	NODE *			nd,
	FLMUINT *	 	drnRV)
{
	RCODE 			rc = FERR_OK;

	*drnRV = (FLMUINT) 0xFFFFFFFF;

	if (nd == NULL)
	{
		rc = RC_SET( FERR_CONV_NULL_SRC);
		goto Exit;
	}

	if (nd->ui32EncId)
	{
		if (!(nd->ui32EncFlags & FLD_HAVE_DECRYPTED_DATA))
		{
			rc = RC_SET( FERR_FLD_NOT_DECRYPTED);
			goto Exit;
		}
	}

	if (GedValType( nd) != FLM_CONTEXT_TYPE)
	{
		rc = RC_SET( FERR_CONV_ILLEGAL);
		goto Exit;
	}

	if (GedValLen( nd) == sizeof(FLMUINT32))
	{
		*drnRV = (FLMUINT) (FB2UD( (FLMBYTE *) GedValPtr( nd)));
	}

Exit:

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE GedWalk(
	FLMUINT				treeCnt,
	NODE *				pNode,
	GEDWALK_FUNC_p 	func,
	void *				arg)
{
	RCODE 	rc;

	if (pNode)
	{
		FLMUINT	baseLevel = GedNodeLevel( pNode);
		do
		{
			rc = (*func)((GedNodeLevel( pNode) - baseLevel), pNode, arg);
		} while(	RC_OK( rc) && (pNode = pNode->next) != NULL &&	
					(GedNodeLevel( pNode) > baseLevel || 
						(GedNodeLevel( pNode) == baseLevel && --treeCnt)));
	}
	else
	{
		rc = FERR_OK;
	}

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
NODE * GedSibNext(
	NODE *		pNode)
{
	FLMUINT	lev;

	if (pNode)
	{
		lev = GedNodeLevel( pNode);
		while( ((pNode = pNode->next) != NULL) && (GedNodeLevel( pNode) > lev));
	}

	return ((pNode && (GedNodeLevel( pNode) == lev)) ? pNode : NULL);
}

/*****************************************************************************
Desc:
*****************************************************************************/
NODE * GedParent(
	NODE *		pNode)
{
	if (pNode)
	{
		FLMUINT	lev = GedNodeLevel( pNode);
		while( ((pNode = pNode->prior) != NULL) && (GedNodeLevel( pNode) >= lev));
	}

	return (pNode);
}

/*****************************************************************************
Desc:
*****************************************************************************/
NODE * GedChild(
	NODE *		pNode)
{
	return( pNode && pNode->next && 
			 (GedNodeLevel( pNode->next) > GedNodeLevel( pNode)) 
			 		? pNode->next 
					: NULL);
}

/*****************************************************************************
Desc:
*****************************************************************************/
NODE * GedSibPrev(
	NODE *		pNode)
{
	FLMUINT	lev;

	if (pNode)
	{
		lev = GedNodeLevel( pNode);
		while( ((pNode = pNode->prior) != NULL) && (GedNodeLevel( pNode) > lev));
	}

	return ((pNode && (GedNodeLevel( pNode) == lev)) ? pNode : NULL);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE GedPutNATIVE(
	F_Pool *			pPool,
	NODE *			pNode,
	const char *	nativeString,
	FLMUINT			uiEncId,
	FLMUINT			uiEncSize)
{
	RCODE			rc = FERR_OK;
	FLMUINT		allocLength;
	FLMBYTE *	outPtr;

	// Check for a null node being passed in

	if (!pNode)
	{
		rc = RC_SET( FERR_CONV_NULL_DEST);
		goto Exit;
	}

	// If the string is NULL or empty, call GedAllocSpace with a length
	// of zero to set the node length to zero and node type to
	// FLM_TEXT_TYPE.
	
	if ((!nativeString) || (!(*nativeString)))
	{
		(void) GedAllocSpace( pPool, pNode, FLM_TEXT_TYPE, 0, uiEncId, uiEncSize);
		goto Exit;
	}

	// Determine the size of the buffer needed to store the string

	if (RC_BAD( rc = FlmNative2Storage( nativeString, 0, &allocLength, NULL)))
	{
		goto Exit;
	}

	if ((outPtr = (FLMBYTE *) GedAllocSpace( pPool, pNode, FLM_TEXT_TYPE,
												allocLength, uiEncId, uiEncSize)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Convert the string

	if (RC_BAD( rc = FlmNative2Storage( nativeString, 0, &allocLength, outPtr)))
	{
		goto Exit;
	}

	// Encrypted fields - only have decrypetd data at this point.

	if (pNode->ui32EncId)
	{
		pNode->ui32EncFlags = FLD_HAVE_DECRYPTED_DATA;
	}

Exit:

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE GedGetNATIVE(
	NODE *		pNode,
	char *		pszBuffer,
	FLMUINT *	bufLenRV)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	ptr;
	FLMUINT		valLength;
	FLMUINT		uiNodeType;

	if (!pNode)
	{
		rc = RC_SET( FERR_CONV_NULL_SRC);
		goto Exit;
	}

	if (pNode->ui32EncId)
	{
		if (!(pNode->ui32EncFlags & FLD_HAVE_DECRYPTED_DATA))
		{
			rc = RC_SET( FERR_FLD_NOT_DECRYPTED);
			goto Exit;
		}
	}

	// If the node is not a TEXT or a NUMBER node, return an error for now

	uiNodeType = (FLMBYTE) GedValType( pNode);
	if ((uiNodeType == FLM_BINARY_TYPE) || (uiNodeType == FLM_CONTEXT_TYPE))
	{
		rc = RC_SET( FERR_CONV_ILLEGAL);
		goto Exit;
	}

	ptr = (FLMBYTE *) GedValPtr( pNode);
	valLength = GedValLen( pNode);

	rc = FlmStorage2Native( uiNodeType, valLength, 
				(const FLMBYTE *) ptr, bufLenRV, pszBuffer);

Exit:

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE GedPutUINT(
	F_Pool *		pPool,
	NODE *		pNode,
	FLMUINT		uiNum,
	FLMUINT		uiEncId,
	FLMUINT		uiEncSize)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pucPtr;
	FLMBYTE		ucStorageBuf[F_MAX_NUM_BUF + 1];
	FLMUINT		uiStorageLen;
	
	if (pNode == NULL)
	{
		rc = RC_SET( FERR_CONV_NULL_DEST);
		goto Exit;
	}

	uiStorageLen = sizeof( ucStorageBuf);
	if( RC_BAD( rc = FlmUINT2Storage( uiNum, &uiStorageLen, ucStorageBuf)))
	{
		goto Exit;
	}

	// Allocate the needed space.

	if ((pucPtr = (FLMBYTE *) GedAllocSpace( pPool, pNode, FLM_NUMBER_TYPE,
			uiStorageLen, uiEncId, uiEncSize)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	f_memcpy( pucPtr, ucStorageBuf, uiStorageLen);

	if (pNode->ui32EncId)
	{
		pNode->ui32EncFlags = FLD_HAVE_DECRYPTED_DATA;
	}

Exit:

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE GedPutUINT64(
	F_Pool *		pPool,
	NODE *		pNode,
	FLMUINT64	ui64Num,
	FLMUINT		uiEncId,
	FLMUINT		uiEncSize)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pucPtr;
	FLMBYTE		ucStorageBuf[F_MAX_NUM64_BUF + 1];
	FLMUINT		uiStorageLen;
	
	if (pNode == NULL)
	{
		rc = RC_SET( FERR_CONV_NULL_DEST);
		goto Exit;
	}

	uiStorageLen = sizeof( ucStorageBuf);
	if( RC_BAD( rc = FlmUINT64ToStorage( ui64Num, &uiStorageLen, ucStorageBuf)))
	{
		goto Exit;
	}

	// Allocate the needed space.

	if ((pucPtr = (FLMBYTE *) GedAllocSpace( pPool, pNode, FLM_NUMBER_TYPE,
			uiStorageLen, uiEncId, uiEncSize)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	f_memcpy( pucPtr, ucStorageBuf, uiStorageLen);

	if (pNode->ui32EncId)
	{
		pNode->ui32EncFlags = FLD_HAVE_DECRYPTED_DATA;
	}

Exit:

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE GedPutINT(
	F_Pool *		pPool,
	NODE *		pNode,
	FLMINT		iNum,
	FLMUINT		uiEncId,
	FLMUINT		uiEncSize)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pucPtr;
	FLMBYTE		ucStorageBuf[F_MAX_NUM_BUF + 1];
	FLMUINT		uiStorageLen;

	if (!pNode)
	{
		rc = RC_SET( FERR_CONV_NULL_DEST);
		goto Exit;
	}

	uiStorageLen = sizeof( ucStorageBuf);
	if( RC_BAD( rc = FlmINT2Storage( iNum, &uiStorageLen, ucStorageBuf)))
	{
		goto Exit;
	}
	
	// Determine number of bytes required for BCD number & allocate space

	if ((pucPtr = (FLMBYTE *) GedAllocSpace( pPool, pNode, FLM_NUMBER_TYPE,
				uiStorageLen, uiEncId, uiEncSize)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	f_memcpy( pucPtr, ucStorageBuf, uiStorageLen);

	if (pNode->ui32EncId)
	{
		pNode->ui32EncFlags = FLD_HAVE_DECRYPTED_DATA;
	}

Exit:

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE GedPutINT64(
	F_Pool *		pPool,
	NODE *		pNode,
	FLMINT64		i64Num,
	FLMUINT		uiEncId,
	FLMUINT		uiEncSize)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pucPtr;
	FLMBYTE		ucStorageBuf[F_MAX_NUM64_BUF + 1];
	FLMUINT		uiStorageLen;

	if (!pNode)
	{
		rc = RC_SET( FERR_CONV_NULL_DEST);
		goto Exit;
	}

	uiStorageLen = sizeof( ucStorageBuf);
	if( RC_BAD( rc = FlmINT64ToStorage( i64Num, &uiStorageLen, ucStorageBuf)))
	{
		goto Exit;
	}
	
	// Determine number of bytes required for BCD number & allocate space

	if ((pucPtr = (FLMBYTE *) GedAllocSpace( pPool, pNode, FLM_NUMBER_TYPE,
				uiStorageLen, uiEncId, uiEncSize)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	f_memcpy( pucPtr, ucStorageBuf, uiStorageLen);

	if (pNode->ui32EncId)
	{
		pNode->ui32EncFlags = FLD_HAVE_DECRYPTED_DATA;
	}

Exit:

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE GedGetINT(
	NODE *		pNode,
	FLMINT *		piNum)
{
	RCODE	rc = FERR_OK;

	if (pNode->ui32EncId)
	{
		if (!(pNode->ui32EncFlags & FLD_HAVE_DECRYPTED_DATA))
		{
			rc = RC_SET( FERR_FLD_NOT_DECRYPTED);
			goto Exit;
		}
	}

	if (RC_BAD( rc = FlmStorage2INT( GedValType( pNode), GedValLen( pNode),
				  (const FLMBYTE *) GedValPtr( pNode), piNum)))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE GedGetINT64(
	NODE *		pNode,
	FLMINT64 *	pi64Num)
{
	RCODE	rc = FERR_OK;

	if (pNode->ui32EncId)
	{
		if (!(pNode->ui32EncFlags & FLD_HAVE_DECRYPTED_DATA))
		{
			rc = RC_SET( FERR_FLD_NOT_DECRYPTED);
			goto Exit;
		}
	}

	if (RC_BAD( rc = FlmStorage2INT64( GedValType( pNode), GedValLen( pNode),
				  (const FLMBYTE *) GedValPtr( pNode), pi64Num)))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE GedGetINT32(
	NODE *			pNode,
	FLMINT32 *		pi32Num)
{
	RCODE	rc = FERR_OK;

	if (pNode->ui32EncId)
	{
		if (!(pNode->ui32EncFlags & FLD_HAVE_DECRYPTED_DATA))
		{
			rc = RC_SET( FERR_FLD_NOT_DECRYPTED);
			goto Exit;
		}
	}

	if (RC_BAD( rc = FlmStorage2INT32( GedValType( pNode), GedValLen( pNode),
									(const FLMBYTE *)GedValPtr( pNode), pi32Num)))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE GedGetINT16(
	NODE *			pNode,
	FLMINT16 *		pi16Num)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiNum;
	FLMBOOL	bNegFlag;

	if (pNode->ui32EncId)
	{
		if (!(pNode->ui32EncFlags & FLD_HAVE_DECRYPTED_DATA))
		{
			rc = RC_SET( FERR_FLD_NOT_DECRYPTED);
			goto Exit;
		}
	}

	if (RC_OK( rc = flmBcd2Num( GedValType( pNode), GedValLen( pNode),
				 (const FLMBYTE *) GedValPtr( pNode), &uiNum, &bNegFlag)))
	{
		if (bNegFlag)
		{
			
			// We will have checked to make sure we are not less than
			// -(FLM_MAX_INT + 1), but this is smaller than
			// than -(FLM_MAX_INT16 + 1),
			// so we need to check to make sure we are not less than
			// -(FLM_MAX_INT32 + 1)
			
			if (uiNum > (FLMUINT)(FLM_MAX_INT16) + 1)
			{
				rc = RC_SET( FERR_CONV_NUM_UNDERFLOW);
				goto Exit;
			}
			*pi16Num = -((FLMINT16)uiNum);
		}
		
		// If the value is positive, we will have checked to make sure the
		// number did not overflow FLM_MAX_UINT, but not FLM_MAX_INT16.
		
		else if (uiNum > (FLMUINT)(FLM_MAX_INT16))
		{
			rc = RC_SET( FERR_CONV_NUM_OVERFLOW);
			goto Exit;
		}
		else
		{
			*pi16Num = (FLMINT16)uiNum;
		}
	}

Exit:

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE GedGetUINT(
	NODE *			pNode,
	FLMUINT *		puiNum)
{
	RCODE		rc = FERR_OK;

	if (pNode->ui32EncId)
	{
		if (!(pNode->ui32EncFlags & FLD_HAVE_DECRYPTED_DATA))
		{
			rc = RC_SET( FERR_FLD_NOT_DECRYPTED);
			goto Exit;
		}
	}

	if (RC_BAD( rc = FlmStorage2UINT( GedValType( pNode), GedValLen( pNode),
				 (const FLMBYTE *) GedValPtr( pNode), puiNum)))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE GedGetUINT8(
	NODE *			pNode,
	FLMUINT8 *		pui8Num)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiNum;
	FLMBOOL	bNegFlag;

	if (pNode->ui32EncId)
	{
		if (!(pNode->ui32EncFlags & FLD_HAVE_DECRYPTED_DATA))
		{
			rc = RC_SET( FERR_FLD_NOT_DECRYPTED);
			goto Exit;
		}
	}

	if (RC_OK( rc = flmBcd2Num( GedValType( pNode), GedValLen( pNode),
				 (const FLMBYTE *) GedValPtr( pNode), &uiNum, &bNegFlag)))
	{
		if (bNegFlag)
		{
			rc = RC_SET( FERR_CONV_NUM_UNDERFLOW);
		}
		else if (uiNum > (FLMUINT)(FLM_MAX_UINT8))
		{
			rc = RC_SET( FERR_CONV_NUM_OVERFLOW);
		}
		else
		{
			*pui8Num = (FLMUINT8)uiNum;
		}
	}

Exit:

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE GedGetUINT64(
	NODE *			pNode,
	FLMUINT64 *		pui64Num)
{
	RCODE		rc = FERR_OK;

	if (pNode->ui32EncId)
	{
		if (!(pNode->ui32EncFlags & FLD_HAVE_DECRYPTED_DATA))
		{
			rc = RC_SET( FERR_FLD_NOT_DECRYPTED);
			goto Exit;
		}
	}

	if (RC_BAD( rc = FlmStorage2UINT64( GedValType( pNode), GedValLen( pNode),
				 (const FLMBYTE *) GedValPtr( pNode), pui64Num)))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE GedGetUINT32(
	NODE *			pNode,
	FLMUINT32 *		pui32Num)
{
	RCODE	rc = FERR_OK;

	if (pNode->ui32EncId)
	{
		if (!(pNode->ui32EncFlags & FLD_HAVE_DECRYPTED_DATA))
		{
			rc = RC_SET( FERR_FLD_NOT_DECRYPTED);
			goto Exit;
		}
	}

	if (RC_BAD( rc = FlmStorage2UINT32( GedValType( pNode), GedValLen( pNode),
				 (const FLMBYTE *) GedValPtr( pNode), pui32Num)))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE GedGetUINT16(
	NODE *			pNode,
	FLMUINT16 *		pui16Num)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiNum;
	FLMBOOL	bNegFlag;

	if (pNode->ui32EncId)
	{
		if (!(pNode->ui32EncFlags & FLD_HAVE_DECRYPTED_DATA))
		{
			rc = RC_SET( FERR_FLD_NOT_DECRYPTED);
			goto Exit;
		}
	}

	if (RC_OK( rc = flmBcd2Num( GedValType( pNode), GedValLen( pNode),
				 (const FLMBYTE *) GedValPtr( pNode), &uiNum, &bNegFlag)))
	{
		if (bNegFlag)
		{
			rc = RC_SET( FERR_CONV_NUM_UNDERFLOW);
		}
		else if (uiNum > (FLMUINT)(FLM_MAX_UINT16))
		{
			rc = RC_SET( FERR_CONV_NUM_OVERFLOW);
		}
		else
		{
			*pui16Num = (FLMUINT16)uiNum;
		}
	}

Exit:

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE GedNumToText(
	const FLMBYTE *	num,
	FLMBYTE *			buffer,
	FLMUINT *			bufLenRV)
{
	FLMBYTE *			outPtr;
	FLMBYTE				c;
	FLMBYTE				c1 = 0;
	FLMUINT				bytesOutput;
	FLMUINT				outputData;
	FLMUINT				maxOutLen;
	FLMBYTE				done;
	FLMBYTE				firstNibble;
	FLMBYTE				lastChar;
	const FLMBYTE *	pExp = NULL;
	FLMBYTE				parseExponent = 0;
	FLMBYTE				firstNibbleAtExp = 0;
	FLMBYTE				firstDigit = 0;

	maxOutLen = *bufLenRV;
	outputData = ((buffer != NULL) && (maxOutLen));
	bytesOutput = 0;
	outPtr = buffer;

	// Parse through the string outputting data to the buffer
	// as we go.
	
	done = (num == NULL);						// Sets to TRUE if NULL else FALSE
	firstNibble = 1;
	lastChar = 0xFF;

	while (!done)
	{
continue_loop:

		if (firstNibble)							// Rather not do a ? : here because
		{												// of the num++ in the : portion
			c = (FLMBYTE) (*num >> 4);
		}
		else
		{
			c = (FLMBYTE) (*num++ &0x0F);
		}

		firstNibble = !firstNibble;

		if (c <= 9)									// Check common case before switch
		{
			if (parseExponent)
			{											// Exponent number?
				firstDigit++;
			}

			c1 = (FLMBYTE) (ASCII_ZERO + c); // Normal decimal value
		}
		else
		{
			switch (c)
			{
				case 0x0A:
					c1 = ASCII_DOT;
					break;
				case 0x0B:
					c1 = ASCII_DASH;
					break;
				case 0x0C:							// Ignore for now - imaginary
														// numbers not implemented
					c1 = 0;							// Set c1 to zero if no output
					break;
				case 0x0D:
					c1 = ASCII_SLASH;
					break;
				case 0x0E:

					// For real numbers the exponent appears first ;
					// This was done to make it easier for building keys
					
					if (!parseExponent)
					{
						parseExponent++;			// 1=need to output 1st digit
						pExp = num;					// Set state to reparse exponent
						if (firstNibble)
						{
							pExp--;
						}

						firstNibbleAtExp = (FLMBYTE) (firstNibble ^ 1);

						// Parse to the end of the exponent area - most 5 nibbles

						for (;;)
						{
							if (firstNibble)
							{
								if ((*num >> 4) == 0x0F)
								{
									break;
								}
							}
							else
							{
								if ((*num++ &0x0F) == 0x0F)
								{
									break;
								}
							}

							firstNibble = !firstNibble;
						}

						firstNibble = !firstNibble;	// Don't forget this!
						goto continue_loop;				// 'continue' is vauge - use
																///* a goto
					}
					else
					{
						c1 = ASCII_UPPER_E;
						parseExponent = 0;				// Clear flag
					}
					break;
				case 0x0F:
					c1 = 0;			// Set c1 to zero if no output
					if (!parseExponent)
					{					// Done if no exponent or done /w exp
						done = TRUE;
					}
					break;
			}
		}

		// If we got a character, put into output buffer (or just count)

		if (c1)
		{

			// If the last character was an exponent and the current ;
			// character is not a minus sign, insert a plus (+)
			
			if ((lastChar == ASCII_UPPER_E) && (c1 != ASCII_MINUS))
			{
				if (outputData)
				{
					if (bytesOutput < maxOutLen)
					{
						*outPtr++ = ASCII_PLUS;
					}
					else
					{
						return (RC_SET( FERR_CONV_DEST_OVERFLOW));
					}
				}

				bytesOutput++;
			}

			if (outputData)
			{
				if (bytesOutput < maxOutLen)
				{
					*outPtr++ = c1;
				}
				else
				{
					return (RC_SET( FERR_CONV_DEST_OVERFLOW));
				}
			}

			bytesOutput++;

			// If exponent (real) number output decimal place

			if (firstDigit == 1)
			{
				firstDigit++;		// Set to != 1
				if (outputData)
				{
					if (bytesOutput < maxOutLen)
					{
						*outPtr++ = ASCII_DOT;
					}
					else
					{
						return (RC_SET( FERR_CONV_DEST_OVERFLOW));
					}
				}

				bytesOutput++;
			}

			lastChar = c1;
		}
		else if (parseExponent) // Hit last trailing 'F' in num
		{
			num = pExp;				// Restore state
			firstNibble = firstNibbleAtExp;

			// Go again parsing the exponent

		}
	}

	*bufLenRV = bytesOutput;
	return (FERR_OK);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE GedTextToNum(
	FLMBYTE *		textStr,		// Pointer to buffer containing TEXT
	FLMUINT			textLen,		// Length of text (in bytes)
	FLMBYTE *		buffer,		// Pointer to buffer where number data is to be
										// returned
	FLMUINT *		bufLenRV)	// Return length -- on input contains buffer size
{
	FLMBYTE *		outPtr;
	FLMBYTE			c;
	FLMUINT			bytesProcessed;
	FLMUINT			bytesOutput;
	FLMUINT			outputData;
	FLMUINT			maxOutLen;
	FLMUINT			objType;
	FLMUINT			objLength;
	FLMBOOL			firstNibble;
	FLMBOOL			have1Num;
	FLMBOOL			insideNum;
	FLMBOOL			haveSign;

	maxOutLen = *bufLenRV;
	outputData = ((buffer != NULL) && (maxOutLen));
	bytesProcessed = bytesOutput = 0;
	outPtr = buffer;

	// Parse through the string outputting data to the buffer
	// as we go.
	
	haveSign = have1Num = insideNum = 0;
	firstNibble = 1;
	if (textStr == NULL)
	{
		textLen = 0;
	}

	for (;
		  bytesProcessed < textLen;
		  textStr += objLength, bytesProcessed += objLength)
	{

		// Determine what we are pointing at

		c = *textStr;
		objType = (FLMBYTE) flmTextObjType( c);

		if (objType == ASCII_CHAR_CODE)
		{
			objLength = 1;
			if ((c == ASCII_SPACE) ||
				 (c == ASCII_TAB) ||
				 (c == ASCII_NEWLINE) ||
				 (c == ASCII_CR))
			{
				if (insideNum)
				{
					have1Num = 1;
				}
				break;
			}

			// Code below was a break - now skips leading zeros

			if ((c == ASCII_ZERO) && (!insideNum))
			{	// Ignore leading zeroes
				continue;
			}

			if ((c >= ASCII_ZERO) && (c <= ASCII_NINE))
			{
				if (!insideNum)
				{
					insideNum = 1;
					haveSign = 1;
				}

				c -= ASCII_ZERO;
			}

			// Handle sign characters ('+', '-')

			else if (((c == ASCII_PLUS) || (c == ASCII_MINUS)) &&
						(!haveSign) &&
						(!insideNum))
			{
				haveSign = 1;
				if (c == ASCII_MINUS)
				{
					c = 0x0B;
				}
			}
			else
			{
				return (RC_SET( FERR_CONV_BAD_DIGIT));
			}

			if (outputData)
			{
				if ((firstNibble) && (bytesOutput == maxOutLen))
				{
					return (RC_SET( FERR_CONV_DEST_OVERFLOW));
				}

				if (firstNibble)
				{
					c <<= 4;
					*outPtr = c;
				}
				else
				{
					*outPtr = (FLMBYTE) (*outPtr + c);
					outPtr++;
				}
			}

			if (firstNibble)
			{
				bytesOutput++;
			}

			firstNibble = !firstNibble;
		}
		else
		{
			switch (objType)
			{
				case WHITE_SPACE_CODE:
					objLength = 1;
					break;

				// Skip the unkown codes for now

				case UNK_GT_255_CODE:
					objLength = (1 + sizeof(FLMUINT16) + FB2UW( textStr + 1));
					break;
				case UNK_LE_255_CODE:
					objLength = (2 + (FLMUINT16) * (textStr + 1));
					break;
				case UNK_EQ_1_CODE:
					objLength = 2;
					break;
				case CHAR_SET_CODE:
				case EXT_CHAR_CODE:
				case OEM_CODE:
				case UNICODE_CODE:

				// Should not hit default.

				default:
					return (RC_SET( FERR_CONV_BAD_DIGIT));
			}
		}
	}

	// Interpret empty number or all zeroes as single zero

	if ((!insideNum) && (!have1Num))
	{
		if (outputData)
		{
			if ((firstNibble) && (bytesOutput == maxOutLen))
			{
				return (RC_SET( FERR_CONV_DEST_OVERFLOW));
			}

			if (firstNibble)
			{
				*outPtr = 0x00;
			}
			else
			{
				outPtr++;
			}
		}

		if (firstNibble)
		{
			bytesOutput++;
		}

		firstNibble = !firstNibble;
	}

	// Add Terminator code to the end of the number

	if (outputData)
	{
		if ((firstNibble) && (bytesOutput == maxOutLen))
		{
			return (RC_SET( FERR_CONV_DEST_OVERFLOW));
		}

		if (firstNibble)
		{
			*outPtr = 0xFF;
		}
		else
		{
			*outPtr = (FLMBYTE) (*outPtr + 0x0F);
		}
	}

	if (firstNibble)
	{
		bytesOutput++;
	}

	*bufLenRV = bytesOutput;
	return (FERR_OK);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE GedPutUNICODE(
	F_Pool *					pPool,
	NODE *					pNode,
	const FLMUNICODE *	puzString,
	FLMUINT					uiEncId,
	FLMUINT					uiEncSize)
{
	FLMUINT		allocLength = 0;
	FLMBYTE *	outPtr;
	RCODE			rc = FERR_OK;

	// Check for a null node being passed in

	if (pNode == NULL)
	{
		rc = RC_SET( FERR_CONV_NULL_DEST);
		goto Exit;
	}

	// If the string is NULL or empty, call GedAllocSpace with a length of
	// zero to set the node length to zero and node type to FLM_TEXT_TYPE.

	if ((puzString == NULL) || (*puzString == 0))
	{
		GedAllocSpace( pPool, pNode, FLM_TEXT_TYPE, 0, uiEncId, uiEncSize);
		return (FERR_OK);
	}

	// Two passes are needed on the data. The first pass is to determine
	// the storage length The second pass is to store the string into
	// FLAIMs internal text format

	allocLength = FlmGetUnicodeStorageLength( puzString);

	if ((outPtr = (FLMBYTE *) GedAllocSpace( pPool, pNode, FLM_TEXT_TYPE,
						allocLength, uiEncId, uiEncSize)) == NULL)
	{
		return (RC_SET( FERR_MEM));
	}

	if (RC_BAD( rc = FlmUnicode2Storage( puzString, &allocLength, outPtr)))
	{
		goto Exit;
	}

	if (pNode->ui32EncId)
	{
		pNode->ui32EncFlags = FLD_HAVE_DECRYPTED_DATA;
	}

Exit:

	return (rc);
}

/*****************************************************************************
Desc:	Get Unicode data from a GEDCOM node.
*****************************************************************************/
RCODE GedGetUNICODE(
	NODE *				pNode,
	FLMUNICODE *		uniBuf,
	FLMUINT *			bufLenRV)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiNodeType;

	// Check for a null node

	if( !pNode)
	{
		rc = RC_SET( FERR_CONV_NULL_SRC);
		goto Exit;
	}

	if (pNode->ui32EncId)
	{
		if (!(pNode->ui32EncFlags & FLD_HAVE_DECRYPTED_DATA))
		{
			rc = RC_SET( FERR_FLD_NOT_DECRYPTED);
			goto Exit;
		}
	}

	// If the node is not a TEXT or a NUMBER node, return an error for now.

	uiNodeType = GedValType( pNode);

	if (uiNodeType == FLM_BINARY_TYPE || uiNodeType == FLM_CONTEXT_TYPE)
	{
		rc = RC_SET( FERR_CONV_ILLEGAL);
		goto Exit;
	}

	if( RC_BAD( rc = FlmStorage2Unicode( uiNodeType, GedValLen( pNode),
									(const FLMBYTE *) GedValPtr( pNode),
									bufLenRV, uniBuf)))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE gedCreateSourceNode(
	F_Pool *		pPool,
	FLMUINT		uiFieldNum,
	HFDB			hDb,
	FLMUINT		uiContainer,
	FLMUINT		uiRecId,
	NODE **	 	ppNode)
{
	NODE *	nd;
	RCODE		rc = FERR_OK;

	if( RC_BAD( rc = pPool->poolCalloc( 
		sizeof(NODE) + sizeof(FLMUINT) + sizeof(FLMUINT) + sizeof(HFDB), 
		(void **)&nd)))
	{
		goto Exit;
	}
		
	GedValTypeSet( nd, FLM_CONTEXT_TYPE);
	GedTagNumSet( nd, uiFieldNum);
	gedSetRecSource( nd, hDb, uiContainer, uiRecId);
	*ppNode = nd;
							
Exit:

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE expImpInit(
	IF_FileHdl *		pFileHdl,
	FLMUINT				uiFlag,
	EXP_IMP_INFO *		pExpImpInfoRV)
{
	RCODE 				rc = FERR_OK;

	f_memset( pExpImpInfoRV, 0, sizeof(EXP_IMP_INFO));
	pExpImpInfoRV->pFileHdl = pFileHdl;
	pExpImpInfoRV->bDictRecords = (uiFlag == EXPIMP_IMPORT_EXPORT_GEDCOM) 
														? FALSE 
														: TRUE;

	// Allocate a buffer for reading or writing.

	pExpImpInfoRV->uiBufSize = (uiFlag == EXPIMP_IMPORT_EXPORT_GEDCOM) 
												? (FLMUINT) 2048 
												: (FLMUINT) 32768;
	for (;;)
	{
		if (RC_BAD( rc = f_alloc( pExpImpInfoRV->uiBufSize, &pExpImpInfoRV->pBuf)))
		{
			pExpImpInfoRV->uiBufSize -= 512;
			if (pExpImpInfoRV->uiBufSize < 1024)
			{
				pExpImpInfoRV->uiBufSize = 0;
				goto Exit;
			}
		}
		else
		{
			break;
		}
	}

	// If writing, output the header data. If reading, seek past it.

	if (uiFlag == EXPIMP_EXPORT_DICTIONARY)
	{

		// Write out the header data.

		rc = expWrite( pExpImpInfoRV, FlmBinaryGedHeader, BINARY_GED_HEADER_LEN);
	}
	else if (uiFlag == EXPIMP_IMPORT_DICTIONARY)
	{
		rc = pFileHdl->seek( (FLMUINT) BINARY_GED_HEADER_LEN, FLM_IO_SEEK_SET,
								  &pExpImpInfoRV->ui64FilePos);
	}
	else
	{
		rc = expWrite( pExpImpInfoRV, FlmBinaryRecHeader, BINARY_GED_HEADER_LEN);
	}

Exit:

	if (RC_BAD( rc))
	{
		expImpFree( pExpImpInfoRV);
	}

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
void expImpFree(
	EXP_IMP_INFO *	 	pExpImpInfo)
{
	if (pExpImpInfo->pBuf)
	{
		f_free( &pExpImpInfo->pBuf);
	}

	f_memset( pExpImpInfo, 0, sizeof(EXP_IMP_INFO));
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE expFlush(
	EXP_IMP_INFO *		pExpImpInfo)
{
	RCODE					rc = FERR_OK;
	FLMUINT				uiBytesWritten;

	if ((pExpImpInfo->uiBufUsed) && (pExpImpInfo->bBufDirty))
	{
		if (RC_BAD( rc = pExpImpInfo->pFileHdl->write( pExpImpInfo->ui64FilePos,
					  pExpImpInfo->uiBufUsed, pExpImpInfo->pBuf, &uiBytesWritten)))
		{
			goto Exit;
		}

		if (uiBytesWritten < pExpImpInfo->uiBufUsed)
		{
			rc = RC_SET( FERR_IO_DISK_FULL);
			goto Exit;
		}

		pExpImpInfo->ui64FilePos += uiBytesWritten;
		pExpImpInfo->uiCurrBuffOffset = pExpImpInfo->uiBufUsed = 0;
		pExpImpInfo->bBufDirty = FALSE;
	}

Exit:

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE expImpSeek(
	EXP_IMP_INFO *	pExpImpInfo,
	FLMUINT			uiSeekPos)
{
	RCODE 			rc = FERR_OK;

	if ((uiSeekPos >= pExpImpInfo->ui64FilePos) &&
		 (uiSeekPos < pExpImpInfo->ui64FilePos + (FLMUINT) pExpImpInfo->uiBufUsed))
	{
		pExpImpInfo->uiCurrBuffOffset = 
			(FLMUINT) (uiSeekPos - pExpImpInfo->ui64FilePos);
	}
	else
	{
		if (pExpImpInfo->bBufDirty)
		{
			if (RC_BAD( rc = expFlush( pExpImpInfo)))
			{
				goto Exit;
			}
		}

		pExpImpInfo->ui64FilePos = uiSeekPos;
		pExpImpInfo->uiBufUsed = pExpImpInfo->uiCurrBuffOffset = 0;
	}

Exit:

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
FSTATIC RCODE expWrite(
	EXP_IMP_INFO *		pExpImpInfo,
	const FLMBYTE *	pData,
	FLMUINT				uiDataLen)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiCopyLen;

	while (uiDataLen)
	{
		if ((uiCopyLen = (pExpImpInfo->uiBufSize - 
				pExpImpInfo->uiCurrBuffOffset)) > uiDataLen)
		{
			uiCopyLen = uiDataLen;
		}

		f_memcpy( &pExpImpInfo->pBuf[pExpImpInfo->uiCurrBuffOffset], pData,
					uiCopyLen);
					
		pExpImpInfo->bBufDirty = TRUE;
		uiDataLen -= uiCopyLen;
		pData += uiCopyLen;
		pExpImpInfo->uiCurrBuffOffset += uiCopyLen;
		
		if (pExpImpInfo->uiCurrBuffOffset > pExpImpInfo->uiBufUsed)
		{
			pExpImpInfo->uiBufUsed = pExpImpInfo->uiCurrBuffOffset;
		}

		if (pExpImpInfo->uiCurrBuffOffset == pExpImpInfo->uiBufSize)
		{
			if (RC_BAD( rc = expFlush( pExpImpInfo)))
			{
				goto Exit;
			}
		}
	}

Exit:

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE expWriteRec(
	EXP_IMP_INFO *		pExpImpInfo,
	FlmRecord *			pRecord,
	FLMUINT				uiDrn)
{
	RCODE				rc = FERR_OK;
	FLMBYTE			TBuf[ 24];
	FLMUINT			uiLen;
	FLMUINT			uiTagNum;
	FLMUINT			uiInitLevel;
	FLMBOOL			bOutputtingRecInfo;
	FLMBOOL			bRootNode;
	FLMUINT			uiTmpLen;
	FlmRecord *		pRec = NULL;
	FlmRecord *		pRecInfoRec = NULL;
	void *			pvField;

	if (pExpImpInfo->bDictRecords)
	{

		// Create a record for the RECINFO information

		if ((pRecInfoRec = f_new FlmRecord) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		if (RC_BAD( rc = pRecInfoRec->insertLast( 0, FLM_RECINFO_TAG,
					  FLM_NUMBER_TYPE, &pvField)))
		{
			goto Exit;
		}

		// Add the record's DRN to the RECINFO information.

		if (RC_BAD( rc = flmAddField( pRecInfoRec, FLM_DRN_TAG, (void *) &uiDrn,
					  4, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}

		bOutputtingRecInfo = TRUE;

		// Output both the REC_INFO GEDCOM tree and the record's GEDCOM
		// tree.

		bRootNode = FALSE;
		pRec = pRecInfoRec;
	}
	else
	{

		// Output only the GEDCOM tree.

		bOutputtingRecInfo = FALSE;
		bRootNode = TRUE;
		pRec = pRecord;
	}

	for (;;)
	{

		// Output each node in the record.

		pvField = pRec->root();
		uiInitLevel = pRec->getLevel( pvField);
		do
		{
			uiTagNum = pRec->getFieldID( pvField);
			uiLen = pRec->getDataLength( pvField);
			UW2FBA( (FLMUINT16) uiTagNum, TBuf);
			UW2FBA( (FLMUINT16) uiLen, &TBuf[2]);
			TBuf[4] = (FLMBYTE) (pRec->getLevel( pvField) - uiInitLevel);
			TBuf[5] = (FLMBYTE) (pRec->getDataType( pvField));

			// Add on the record source information for the root node.

			uiTmpLen = 6;
			if (bRootNode)
			{
				UW2FBA( (FLMUINT16) pRec->getContainerID(), &TBuf[14]);
				UD2FBA( (FLMUINT32)pRec->getID(), &TBuf[16]);
				uiTmpLen = 20;

				bRootNode = FALSE;
			}

			if (RC_BAD( rc = expWrite( pExpImpInfo, TBuf, uiTmpLen)))
			{
				goto Exit;
			}

			if (uiLen)
			{
				const FLMBYTE *		pvData = pRec->getDataPtr( pvField);

				if (RC_BAD( rc = expWrite( pExpImpInfo, pvData, uiLen)))
				{
					goto Exit;
				}
			}

			pvField = pRec->next( pvField);
		} while (pvField && (pRec->getLevel( pvField) > uiInitLevel));

		// Output a zero tag number to indicate end of GEDCOM record.

		UW2FBA( 0, TBuf);
		if (RC_BAD( rc = expWrite( pExpImpInfo, TBuf, 2)))
		{
			goto Exit;
		}

		// Set things up to output the record after the REC_INFO.

		if (!bOutputtingRecInfo)
		{
			break;
		}

		bOutputtingRecInfo = FALSE;
		bRootNode = TRUE;
		pRec = pRecord;
	}

Exit:

	if (pRecInfoRec)
	{
		pRecInfoRec->Release();
	}

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
FSTATIC RCODE impRead(
	EXP_IMP_INFO *		pExpImpInfo,		// Export/Import information.
	FLMBYTE *			pData,				// Buffer where data is to be read into.
	FLMUINT				uiDataLen,			// Length of data to be read in.
	FLMUINT *			puiBytesReadRV)	// Returns amount of data read in.
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiCopyLen;
	FLMUINT		uiBytesRead = 0;

	while (uiDataLen)
	{

		// See if we need to read some more data into the import buffer.

		if (pExpImpInfo->uiCurrBuffOffset == pExpImpInfo->uiBufUsed)
		{

			// If we have a dirty buffer, flush it out first.

			if (pExpImpInfo->bBufDirty)
			{
				if (RC_BAD( rc = expFlush( pExpImpInfo)))
				{
					goto Exit;
				}
			}
			else
			{
				pExpImpInfo->ui64FilePos += (FLMUINT) pExpImpInfo->uiBufUsed;
				pExpImpInfo->uiBufUsed = pExpImpInfo->uiCurrBuffOffset = 0;
			}

			if (RC_BAD( rc = pExpImpInfo->pFileHdl->read( pExpImpInfo->ui64FilePos,
						  pExpImpInfo->uiBufSize, pExpImpInfo->pBuf,
						  &pExpImpInfo->uiBufUsed)))
			{
				if ((rc == FERR_IO_END_OF_FILE) && (pExpImpInfo->uiBufUsed))
				{
					rc = FERR_OK;
				}
				else
				{
					goto Exit;
				}
			}
		}

		// Copy from the import buffer to the data buffer.

		if ((
				 uiCopyLen =
				 (
				 pExpImpInfo->uiBufUsed -
			 pExpImpInfo->uiCurrBuffOffset
		 )
	 ) > uiDataLen)
			{
				uiCopyLen = uiDataLen;
		}

		f_memcpy( pData, &pExpImpInfo->pBuf[pExpImpInfo->uiCurrBuffOffset],
					uiCopyLen);
		uiDataLen -= uiCopyLen;
		uiBytesRead += uiCopyLen;
		pData += uiCopyLen;
		pExpImpInfo->uiCurrBuffOffset += uiCopyLen;
	}

Exit:

	*puiBytesReadRV = uiBytesRead;
	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE impReadRec(
	EXP_IMP_INFO *		pExpImpInfo,	// Export/Import information.
	FlmRecord **	 	ppRecordRV)		// Returns record that was read in.
{
	RCODE					rc = FERR_OK;
	FLMBYTE				TBuf[24];
	FLMUINT				uiLen;
	FLMUINT				uiTagNum;
	FLMUINT				uiRecInfoDrn = 0;
	FLMUINT				uiDictID;
	FLMBOOL				bHaveRecInfo = FALSE;
	FLMBOOL				bHaveDictID = FALSE;
	FLMUINT				uiLevel;
	FLMUINT				uiType;
	FLMBOOL				bGettingRecInfo;
	FLMUINT				uiBytesRead;
	FLMUINT				uiTmpLen;
	FlmRecord *			pRecord = NULL;
	void *				pvField;

	bGettingRecInfo = (pExpImpInfo->bDictRecords) ? TRUE : FALSE;

	// Read each node in the REC_INFO (if dictionary) and then the record.

	for (;;)
	{
		if (RC_BAD( rc = impRead( pExpImpInfo, TBuf, 2, &uiBytesRead)))
		{
			if ((rc == FERR_IO_END_OF_FILE) &&
				 (uiBytesRead == 0) &&
				 ((!bGettingRecInfo) || (!bHaveRecInfo)))
			{
				rc = RC_SET( FERR_END);
			}

			goto Exit;
		}

		// A tag number of zero means we are at the end of the record.

		uiTagNum = FB2UW( TBuf);
		if (!uiTagNum)
		{
			if (bGettingRecInfo)
			{
				bGettingRecInfo = FALSE;
				continue;
			}
			else
			{
				break;
			}
		}

		uiTmpLen = ((!bGettingRecInfo) && (!pRecord)) ? 18 : 4;
		if (RC_BAD( rc = impRead( pExpImpInfo, TBuf, uiTmpLen, &uiBytesRead)))
		{
			goto Exit;
		}

		uiLen = FB2UW( TBuf);
		uiLevel = TBuf[2];
		uiType = TBuf[3];

		if (!pRecord)
		{
			if ((pRecord = f_new FlmRecord) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}

			pRecord->setContainerID( FB2UW( &TBuf[12]));
			pRecord->setID( FB2UD( &TBuf[14]));
		}

		if (RC_BAD( rc = pRecord->insertLast( uiLevel, uiTagNum, 
			uiType, &pvField)))
		{
			goto Exit;
		}

		if (uiLen)
		{
			FLMBYTE *		pValue;

			if (RC_BAD( rc = pRecord->allocStorageSpace( pvField, uiType, 
				uiLen, 0, 0, 0, &pValue, NULL)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = impRead( pExpImpInfo, pValue, uiLen, &uiBytesRead)))
			{
				goto Exit;
			}
		}

		// Link the node into the tree.

		if (bGettingRecInfo)
		{
			switch (uiTagNum)
			{
				case FLM_RECINFO_TAG:
				{
					bHaveRecInfo = TRUE;
					break;
				}
				
				case FLM_DRN_TAG:
				{
					if (RC_BAD( rc = pRecord->getUINT( pvField, &uiRecInfoDrn)))
					{
						goto Exit;
					}
					break;
				}
				
				case FLM_DICT_SEQ_TAG:
				{
					if (RC_BAD( rc = pRecord->getUINT( pvField, &uiDictID)))
					{
						goto Exit;
					}

					bHaveDictID = TRUE;
					break;
				}
			}
		}
	}

Exit:

	if (RC_OK( rc))
	{
		*ppRecordRV = pRecord;
	}
	else
	{
		if (pRecord)
		{
			pRecord->Release();
		}

		*ppRecordRV = NULL;
	}

	return (rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE impFileIsExpImp(
	IF_FileHdl *	pFileHdl,
	FLMBOOL *		pbFileIsBinaryRV)
{
	RCODE				rc = FERR_OK;
	FLMUINT64		ui64CurrPos;
	FLMBYTE			byHeader[ BINARY_GED_HEADER_LEN];
	FLMUINT			uiBytesRead;

	*pbFileIsBinaryRV = FALSE;

	// Save current position so we can return to it.

	if (RC_BAD( rc = pFileHdl->seek( 0, FLM_IO_SEEK_CUR, &ui64CurrPos)))
	{
		goto Exit;
	}

	// Read the file's header information.

	if (RC_BAD( rc = pFileHdl->read( (FLMUINT) 0, BINARY_GED_HEADER_LEN,
				  byHeader, &uiBytesRead)))
	{
		if (rc == FERR_IO_END_OF_FILE)
		{
			uiBytesRead = 0;
			rc = FERR_OK;
		}
		else
		{
			goto Exit;
		}
	}

	if ((uiBytesRead == BINARY_GED_HEADER_LEN) && 
		((f_memcmp( byHeader, FlmBinaryGedHeader, BINARY_GED_HEADER_LEN) == 0) || 
		(f_memcmp( byHeader, FlmBinaryRecHeader, BINARY_GED_HEADER_LEN) == 0)))
	{
		*pbFileIsBinaryRV = TRUE;
	}

	// Reset the file position to where it was before.

	rc = pFileHdl->seek( ui64CurrPos, FLM_IO_SEEK_SET);

Exit:

	return (rc);
}


/****************************************************************************
Desc:		This routine adds a field to a GEDCOM tree
****************************************************************************/
RCODE gedAddField(
	F_Pool *			pPool,
	NODE *			pRecord,
	FLMUINT			uiTagNum,
	const void *	pvData,
	FLMUINT			uiDataLen,
	FLMUINT			uiDataType)
{
	RCODE			rc = FERR_OK;
	NODE *		pChildNode;
	FLMUINT		uiNum;

	if ((pChildNode = GedNodeMake( pPool, uiTagNum, &rc)) == NULL)
	{
		goto Exit;
	}

	switch( uiDataType)
	{
		case FLM_TEXT_TYPE:
		{
			rc = GedPutNATIVE( pPool, pChildNode, (const char *)pvData);
			break;
		}
		
		case FLM_NUMBER_TYPE:
		{
			switch (uiDataLen)
			{
				case 0:
					uiNum = (FLMUINT)(*((FLMUINT *)(pvData)));
					rc = GedPutUINT( pPool, pChildNode, uiNum);
					break;
				case 1:
					uiNum = (FLMUINT)(*((FLMBYTE *)(pvData)));
					rc = GedPutUINT( pPool, pChildNode, uiNum);
					break;
				case 2:
					uiNum = (FLMUINT)(*((FLMUINT16 *)(pvData)));
					rc = GedPutUINT( pPool, pChildNode, uiNum);
					break;
				case 4:
					uiNum = (FLMUINT)(*((FLMUINT32 *)(pvData)));
					rc = GedPutUINT( pPool, pChildNode, uiNum);
					break;
				case 8:
					rc = GedPutUINT64( pPool, pChildNode, *((FLMUINT64 *)(pvData)));
					break;
				default:
					flmAssert( 0);
					rc = RC_SET( FERR_INVALID_PARM);
					goto Exit;
			}
			break;
		}
		
		case FLM_BINARY_TYPE:
		{
			rc = GedPutBINARY( pPool, pChildNode, pvData, uiDataLen);
			break;
		}
	}
	
	if (RC_BAD( rc))
	{
		goto Exit;
	}
	
	GedChildGraft( pRecord, pChildNode, GED_LAST);

Exit:

	return( rc);
}
