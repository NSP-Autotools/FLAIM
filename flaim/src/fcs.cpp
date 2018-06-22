//-------------------------------------------------------------------------
// Desc: FLAIM C/S client interface
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

FSTATIC FLMBOOL flmGetNextHexPacketSlot( 
	FLMBYTE *				pucUsedMap,
	FLMUINT					uiMapSize,
	IF_RandomGenerator *	pRandGen,
	FLMUINT *				puiSlot);
	
FSTATIC RCODE flmGetNextHexPacketBytes( 
	FLMBYTE *				pucUsedMap,
	FLMUINT					uiMapSize,
	FLMBYTE *				pucPacket,
	IF_RandomGenerator *	pRandGen,
	FLMBYTE *				pucBuf,
	FLMUINT					uiCount);

/****************************************************************************
Desc:
*****************************************************************************/
FCS_BIOS::FCS_BIOS( void)
{
	m_pool.poolInit( (FCS_BIOS_BLOCK_SIZE + sizeof( FCSBIOSBLOCK)) * 2);
	m_bMessageActive = FALSE;
	m_pRootBlock = NULL;
	m_pCurrWriteBlock = NULL;
	m_pCurrReadBlock = NULL;
	m_bAcceptingData = FALSE;
	m_pEventHook = NULL;
	m_pvUserData = 0;
}

/****************************************************************************
Desc:
*****************************************************************************/
FCS_BIOS::~FCS_BIOS()
{
	m_pool.poolFree();
}

/****************************************************************************
Desc:	Clears all pending data
*****************************************************************************/
RCODE FCS_BIOS::reset( void)
{
	return( close());
}

/****************************************************************************
Desc:	Flushes any pending data and closes the stream.
*****************************************************************************/
RCODE FCS_BIOS::close( void)
{
	RCODE		rc = FERR_OK;

	m_pool.poolReset();
	m_bMessageActive = FALSE;
	m_pRootBlock = NULL;
	m_pCurrWriteBlock = NULL;
	m_pCurrReadBlock = NULL;
	m_bAcceptingData = FALSE;

	return( rc);
}

/****************************************************************************
Desc:	Writes the requested amount of data to the stream.
*****************************************************************************/
RCODE FCS_BIOS::write(
	FLMBYTE *		pucData,
	FLMUINT			uiLength)
{
	FLMUINT				uiCopySize;
	FLMUINT				uiDataPos = 0;
	FCSBIOSBLOCK *		pPrevBlock = NULL;
	RCODE					rc = FERR_OK;

	if( !m_bAcceptingData)
	{
		m_pool.poolReset();
		m_pCurrWriteBlock = NULL;
		m_pCurrReadBlock = NULL;
		m_pRootBlock = NULL;
		m_bAcceptingData = TRUE;
	}

	while( uiLength)
	{
		if( !m_pCurrWriteBlock ||
			m_pCurrWriteBlock->uiCurrWriteOffset == FCS_BIOS_BLOCK_SIZE)
		{
			pPrevBlock = m_pCurrWriteBlock;
			
			if( RC_BAD( rc = m_pool.poolCalloc( sizeof( FCSBIOSBLOCK), 
				(void **)&m_pCurrWriteBlock)))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = m_pool.poolAlloc( FCS_BIOS_BLOCK_SIZE, 
				(void **)&m_pCurrWriteBlock->pucBlock)))
			{
				goto Exit;
			}

			if( pPrevBlock)
			{
				pPrevBlock->pNextBlock = m_pCurrWriteBlock;
			}
			else
			{
				m_pRootBlock = m_pCurrWriteBlock;
				m_pCurrReadBlock = m_pCurrWriteBlock;
			}
		}

		uiCopySize = f_min( uiLength,
			(FLMUINT)(FCS_BIOS_BLOCK_SIZE -
			m_pCurrWriteBlock->uiCurrWriteOffset));

		flmAssert( uiCopySize != 0);

		f_memcpy( &(m_pCurrWriteBlock->pucBlock[
			m_pCurrWriteBlock->uiCurrWriteOffset]),
			&(pucData[ uiDataPos]), uiCopySize);
		
		m_pCurrWriteBlock->uiCurrWriteOffset += uiCopySize;
		uiDataPos += uiCopySize;
		uiLength -= uiCopySize;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Terminates the current message
*****************************************************************************/
RCODE FCS_BIOS::endMessage( void)
{
	RCODE		rc = FERR_OK;

	if( !m_bAcceptingData)
	{
		goto Exit;
	}

	if( m_pEventHook)
	{
		if( RC_BAD( rc = m_pEventHook( this,
			FCS_BIOS_EOM_EVENT, m_pvUserData)))
		{
			goto Exit;
		}
	}

Exit:

	m_bAcceptingData = FALSE;
	return( rc);
}

/****************************************************************************
Desc:	Reads the requested amount of data from the stream.
*****************************************************************************/
RCODE FCS_BIOS::read(
	FLMBYTE *		pucData,
	FLMUINT			uiLength,
	FLMUINT *		puiBytesRead)
{
	FLMUINT		uiCopySize;
	FLMUINT		uiDataPos = 0;
	RCODE			rc = FERR_OK;

	if( puiBytesRead)
	{
		*puiBytesRead = 0;
	}

	if( m_bAcceptingData)
	{
		m_bAcceptingData = FALSE;
	}
	
	while( uiLength)
	{
		if( m_pCurrReadBlock &&
			m_pCurrReadBlock->uiCurrReadOffset == 
			m_pCurrReadBlock->uiCurrWriteOffset)
		{
			m_pCurrReadBlock = m_pCurrReadBlock->pNextBlock;
		}

		if( !m_pCurrReadBlock)
		{
			m_pool.poolReset();
			rc = RC_SET( FERR_EOF_HIT);
			goto Exit;
		}

		uiCopySize = f_min( uiLength,
			m_pCurrReadBlock->uiCurrWriteOffset - 
			m_pCurrReadBlock->uiCurrReadOffset);

		f_memcpy( &(pucData[ uiDataPos]),
			&(m_pCurrReadBlock->pucBlock[ m_pCurrReadBlock->uiCurrReadOffset]),
			uiCopySize);

		m_pCurrReadBlock->uiCurrReadOffset += uiCopySize;
		uiDataPos += uiCopySize;

		if( puiBytesRead)
		{
			(*puiBytesRead) += uiCopySize;
		}
		uiLength -= uiCopySize;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	
*****************************************************************************/
FLMBOOL FCS_BIOS::isDataAvailable( void)
{
	if( m_bAcceptingData)
	{
		if( m_pRootBlock && m_pRootBlock->uiCurrWriteOffset)
		{
			return( TRUE);
		}
	}
	else if( m_pCurrReadBlock &&
		((m_pCurrReadBlock->uiCurrReadOffset <
			m_pCurrReadBlock->uiCurrWriteOffset) ||
		(m_pCurrReadBlock->pNextBlock)))
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:	Returns the amount of data available for reading
*****************************************************************************/
FLMUINT FCS_BIOS::getAvailable( void)
{
	FLMUINT				uiAvail = 0;
	FCSBIOSBLOCK *		pBlk;

	if( m_bAcceptingData)
	{
		if( m_pRootBlock && m_pRootBlock->uiCurrWriteOffset)
		{
			pBlk = m_pRootBlock;
			while( pBlk)
			{
				uiAvail += pBlk->uiCurrWriteOffset;
				pBlk = pBlk->pNextBlock;
			}
		}
	}
	else if( m_pCurrReadBlock &&
		((m_pCurrReadBlock->uiCurrReadOffset <
			m_pCurrReadBlock->uiCurrWriteOffset) ||
		(m_pCurrReadBlock->pNextBlock)))
	{
		pBlk = m_pCurrReadBlock;
		while( pBlk)
		{
			uiAvail += (pBlk->uiCurrWriteOffset -
				pBlk->uiCurrReadOffset);
			pBlk = pBlk->pNextBlock;
		}
	}

	return( uiAvail);
}

/****************************************************************************
Desc: 
****************************************************************************/
FCS_DIS::FCS_DIS( void)
{
	m_pIStream = NULL;
	m_uiBOffset = m_uiBDataSize = 0;
	m_bSetupCalled = FALSE;
}

/****************************************************************************
Desc: 
****************************************************************************/
FCS_DIS::~FCS_DIS( void)
{
	if( m_bSetupCalled)
	{
		(void)close();
	}
}

/****************************************************************************
Desc: 
****************************************************************************/
RCODE FCS_DIS::setup( 
	FCS_ISTM *		pIStream)
{
	m_pIStream = pIStream;
	m_bSetupCalled = TRUE;

	return( FERR_OK);
}

/****************************************************************************
Desc: 
****************************************************************************/
RCODE FCS_DIS::readByte( 
	FLMBYTE *		pValue)
{
	return( read( pValue, 1, NULL));
}

/****************************************************************************
Desc: 
****************************************************************************/
RCODE FCS_DIS::readShort( 
	FLMINT16 *		pValue)
{
	RCODE			rc;
	
	if( RC_OK( rc = read( (FLMBYTE *)pValue, 2, NULL)))
	{
		*pValue = f_bigEndianToINT16( (FLMBYTE *)pValue);
	}

	return( rc);
}

/****************************************************************************
Desc: 
****************************************************************************/
RCODE FCS_DIS::readUShort( 
	FLMUINT16 *		pValue)
{
	RCODE			rc;
	
	if( RC_OK( rc = read( (FLMBYTE *)pValue, 2, NULL)))
	{
		*pValue = f_bigEndianToUINT16( (FLMBYTE *)pValue);
	}

	return( rc);
}

/****************************************************************************
Desc: 
****************************************************************************/
RCODE FCS_DIS::readInt( 
	FLMINT32 *		pValue)
{
	RCODE			rc;
	
	if( RC_OK( rc = read( (FLMBYTE *)pValue, 4, NULL)))
	{
		*pValue = f_bigEndianToINT32( (FLMBYTE *)pValue);
	}
	
	return( rc);
}

/****************************************************************************
Desc: 
****************************************************************************/
RCODE FCS_DIS::readUInt( 
	FLMUINT32 *		pValue)
{
	RCODE			rc;
	
	if( RC_OK( rc = read( (FLMBYTE *)pValue, 4, NULL)))
	{
		*pValue = f_bigEndianToUINT32( (FLMBYTE *)pValue);
	}

	return( rc);
}

/****************************************************************************
Desc: 
****************************************************************************/
RCODE FCS_DIS::readInt64( 
	FLMINT64 *		pValue)
{
	RCODE			rc;
	
	if( RC_OK( rc = read( (FLMBYTE *)pValue, 8, NULL)))
	{
		*pValue = f_bigEndianToINT64( (FLMBYTE *)pValue);
	}
	
	return( rc);
}

/****************************************************************************
Desc: 
****************************************************************************/
RCODE FCS_DIS::readUInt64( 
	FLMUINT64 *		pValue)
{
	RCODE			rc;
	
	if( RC_OK( rc = read( (FLMBYTE *)pValue, 8, NULL)))
	{
		*pValue = f_bigEndianToUINT64( (FLMBYTE *)pValue);
	}

	return( rc);
}

/****************************************************************************
Desc: 
****************************************************************************/
RCODE FCS_DIS::skip( 
	FLMUINT			uiBytesToSkip)
{
	return( read( NULL, uiBytesToSkip, NULL));
}

/****************************************************************************
Desc:	Flushes any pending data and closes the DIS
****************************************************************************/
RCODE FCS_DIS::close( void)
{
	RCODE		rc = FERR_OK;

	// Verify that Setup has been called.

	flmAssert( m_bSetupCalled == TRUE);

	// Terminate and flush.
	
	if( RC_BAD( rc = endMessage()))
	{
		goto Exit;
	}

	// Reset the member variables.

	m_pIStream = NULL;

Exit:

	return( rc);
}


/****************************************************************************
Desc:	Returns the state of the stream (open == TRUE, closed == FALSE)
****************************************************************************/
FLMBOOL FCS_DIS::isOpen( void)
{
	// Verify that Setup has been called.

	flmAssert( m_bSetupCalled == TRUE);

	if( m_pIStream && m_pIStream->isOpen())
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:	Flushes and terminates the current parent stream message
****************************************************************************/
RCODE FCS_DIS::endMessage( void)
{
	RCODE		rc = FERR_OK;

	// Verify that Setup has been called.

	flmAssert( m_bSetupCalled == TRUE);

	if( !m_pIStream)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
	
	// Flush any pending data.

	if( RC_BAD( rc = flush()))
	{
		goto Exit;
	}

	// Terminate the message.

	if( RC_BAD( rc = m_pIStream->endMessage()))
	{
		goto Exit;
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:	Flushes any pending data
****************************************************************************/
RCODE FCS_DIS::flush( void)
{
	RCODE		rc = FERR_OK;

	// Verify that Setup has been called.

	flmAssert( m_bSetupCalled == TRUE);

	if( !m_pIStream)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
	
	// Flush the passed-in input stream.

	if( RC_BAD( rc = m_pIStream->flush()))
	{
		goto Exit;
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:	Reads the specified number of bytes.
****************************************************************************/
RCODE FCS_DIS::read(
	FLMBYTE *	pucData,
	FLMUINT		uiLength,
	FLMUINT *	puiBytesRead)
{
	FLMUINT		uiCopySize;
	FLMUINT		uiReadLen;
	FLMBYTE *	pucPos = NULL;
	RCODE			rc = FERR_OK;

	// Verify that Setup has been called.

	flmAssert( m_bSetupCalled == TRUE);

	if( !m_pIStream)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	if( puiBytesRead)
	{
		*puiBytesRead = uiLength;
	}

	pucPos = pucData;
	while( uiLength)
	{
		if( m_uiBOffset == m_uiBDataSize)
		{
			m_uiBOffset = m_uiBDataSize = 0;

			if( RC_BAD( rc = m_pIStream->read( m_pucBuffer,
				FCS_DIS_BUFFER_SIZE, &uiReadLen)))
			{
				if( uiReadLen)
				{
					rc = FERR_OK;
				}
				else
				{
					goto Exit;
				}
			}
			m_uiBDataSize = uiReadLen;
		}

		uiCopySize = m_uiBDataSize - m_uiBOffset;
		if( uiLength < uiCopySize)
		{
			uiCopySize = uiLength;
		}

		if( pucPos)
		{
			f_memcpy( pucPos, &(m_pucBuffer[ m_uiBOffset]), uiCopySize);
			pucPos += uiCopySize;
		}
		
		m_uiBOffset += uiCopySize;
		uiLength -= uiCopySize;
	}
	
Exit:

	if( RC_OK( rc) && uiLength)
	{
		// Unable to satisfy the read request.

		rc = RC_SET( FERR_EOF_HIT);
	}

	if( puiBytesRead)
	{
		(*puiBytesRead) -= uiLength;
	}

	return( rc);
}

/****************************************************************************
Desc:	Reads a binary token from the stream.  The token is tagged with a
		length.
****************************************************************************/
RCODE FCS_DIS::readBinary(
	F_Pool *		pPool,
	FLMBYTE **	ppValue,
	FLMUINT *	puiDataSize)
{
	FLMUINT16	ui16DataSize;
	RCODE			rc = FERR_OK;

	if( RC_BAD( rc = readUShort( &ui16DataSize)))
	{
		goto Exit;
	}

	if( pPool)
	{
		// If the data size is non-zero, allocate a buffer and
		// read the entire binary value.

		if( ui16DataSize)
		{
			if( RC_BAD( rc = pPool->poolAlloc( ui16DataSize, (void **)ppValue)))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = read( *ppValue, ui16DataSize, NULL)))
			{
				goto Exit;
			}
		}
		else
		{
			*ppValue = NULL;
		}
	}
	else
	{
		// The application is not interested in the value.  Just skip the
		// to the end of the value.

		if( RC_BAD( rc = skip( ui16DataSize)))
		{
			goto Exit;
		}
	}

Exit:

	if( puiDataSize)
	{
		*puiDataSize = ui16DataSize;
	}

	return( rc);
}


/****************************************************************************
Desc:	Reads a large binary token from the stream.  The token is tagged with a
		length.
****************************************************************************/
RCODE FCS_DIS::readLargeBinary(
	F_Pool *		pPool,
	FLMBYTE **	ppValue,
	FLMUINT *	puiDataSize)
{
	FLMUINT32	ui32DataSize;
	RCODE			rc = FERR_OK;

	if( RC_BAD( rc = readUInt( &ui32DataSize)))
	{
		goto Exit;
	}

	if( pPool)
	{
		// If the data size is non-zero, allocate a buffer and
		// read the entire binary value.

		if( ui32DataSize)
		{
			if( RC_BAD(rc = pPool->poolAlloc( ui32DataSize, (void **)ppValue)))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = read( *ppValue, ui32DataSize, NULL)))
			{
				goto Exit;
			}
		}
		else
		{
			*ppValue = NULL;
		}
	}
	else
	{
		// The application is not interested in the value.  Just skip the
		// to the end of the value.

		if( RC_BAD( rc = skip( ui32DataSize)))
		{
			goto Exit;
		}
	}

Exit:

	if( puiDataSize)
	{
		*puiDataSize = (FLMUINT)ui32DataSize;
	}

	return( rc);
}

/****************************************************************************
Desc:	Reads a UTF-8 string from the stream.
****************************************************************************/
RCODE	FCS_DIS::readUTF(
	F_Pool *			pPool,
	FLMUNICODE **	ppValue)
{
	FLMBYTE		ucByte1;
	FLMBYTE		ucByte2;
	FLMBYTE		ucByte3;
	FLMBYTE		ucLoByte;
	FLMBYTE		ucHiByte;
	FLMUINT16	ui16UTFLen;
	FLMUINT		uiOffset = 0;
	RCODE			rc = FERR_OK;
	
	// Read the data.

	if( RC_BAD( rc = readUShort( &ui16UTFLen)))
	{
		goto Exit;
	}

	// Check the size of the UTF string.  FLAIM does not support
	// strings that are larger than 32K characters.

	if( ui16UTFLen >= 32767)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	
	// Allocate space for the string.

	if( pPool)
	{
		if( RC_BAD( rc = pPool->poolAlloc(
			(FLMUINT)((FLMUINT)sizeof( FLMUNICODE) * (FLMUINT)(ui16UTFLen + 1)),
			(void **)ppValue)))
		{
			goto Exit;
		}
	}
	else if( ppValue)
	{
		*ppValue = NULL;
	}

	while( ui16UTFLen)
	{
		// Read and decode the bytes.

		if( RC_BAD( rc = read( &ucByte1, 1, NULL)))
		{
			goto Exit;
		}

		if( (ucByte1 & 0xC0) != 0xC0)
		{
			ucHiByte = 0;
			ucLoByte = ucByte1;
		}
		else
		{
			if( RC_BAD( rc = read( &ucByte2, 1, NULL)))
			{
				goto Exit;
			}

			if( (ucByte1 & 0xE0) == 0xE0)
			{
				if( RC_BAD( rc = read( &ucByte3, 1, NULL)))
				{
					goto Exit;
				}

				ucHiByte =
					(FLMBYTE)(((ucByte1 & 0x0F) << 4) | ((ucByte2 & 0x3C) >> 2));
				ucLoByte = (FLMBYTE)(((ucByte2 & 0x03) << 6) | (ucByte3 & 0x3F));
			}
			else
			{
				ucHiByte = (FLMBYTE)(((ucByte1 & 0x1C) >> 2));
				ucLoByte = (FLMBYTE)(((ucByte1 & 0x03) << 6) | (ucByte2 & 0x3F));
			}
		}

		if( pPool)
		{
			(*ppValue)[ uiOffset] = 
				(FLMUNICODE)(((((FLMUNICODE)(ucHiByte)) << 8) | 
					((FLMUNICODE)(ucLoByte))));
		}

		uiOffset++;
		ui16UTFLen--;
	}

	if( pPool)
	{
		(*ppValue)[ uiOffset] = 0;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: Reads an Hierarchical Tagged Data record from the stream.
****************************************************************************/
RCODE FCS_DIS::readHTD(
	F_Pool *			pPool,
	FLMUINT			uiContainer,
	FLMUINT			uiDrn,
	NODE **			ppNode,
	FlmRecord **	ppRecord)
{

	FLMBYTE		ucType;
	FLMBYTE		ucLevel = 0;
	FLMBYTE		ucPrevLevel = 0;
	FLMBYTE		ucDescriptor;
	FLMBYTE		ucFlags;
	FLMUINT16	ui16Tag;
	FLMBOOL		bHasValue;
	FLMBOOL		bChild;
	FLMBOOL		bSibling;
	FLMBOOL		bLeftTruncated;
	FLMBOOL		bRightTruncated;
	NODE *		pRoot = NULL;
	NODE *		pNode = NULL;
	NODE *		pPrevNode = NULL;
	void *		pField = NULL;
	void *		pvMark = NULL;
	RCODE			rc = FERR_OK;

	if( pPool)
	{
		pvMark = pPool->poolMark();
	}

	for( ;;)
	{
		// Reset variables.

		bChild = FALSE;
		bSibling = FALSE;

		// Read the attribute's tag number.

		if( RC_BAD( rc = readUShort( &ui16Tag)))
		{
			goto Exit;
		}

		// A tag number of 0 indicates that the end of the HTD data
		// stream has been reached.

		if( !ui16Tag)
		{
			break;
		}

		// Read the attribute's descriptor.

		if( RC_BAD(rc = read( &ucDescriptor, 1, NULL)))
		{
			goto Exit;
		}

		// Set the flag indicating whether or not the
		// attribute has a value.

		bHasValue = (FLMBOOL)((ucDescriptor & HTD_HAS_VALUE_FLAG)
							? (FLMBOOL)TRUE
							: (FLMBOOL)FALSE);

		// Set the value type.

		ucType = (FLMBYTE)((ucDescriptor & HTD_VALUE_TYPE_MASK));

		// Get the attribute's level.

		switch( (ucDescriptor & HTD_LEVEL_MASK) >> HTD_LEVEL_POS)
		{
			case HTD_LEVEL_SIBLING:
			{
				bSibling = TRUE;
				ucLevel = ucPrevLevel;
				break;
			}
			
			case HTD_LEVEL_CHILD:
			{
				if( ucLevel < 0xFF)
				{
					bChild = TRUE;
					ucLevel = (FLMBYTE)(ucPrevLevel + 1);
				}
				else
				{
					rc = RC_SET( FERR_BAD_FIELD_LEVEL);
					goto Exit;
				}
				break;
			}
			
			case HTD_LEVEL_BACK:
			{
				if( ucLevel > 0)
				{
					ucLevel = (FLMBYTE)(ucPrevLevel - 1);
				}
				else
				{
					rc = RC_SET( FERR_BAD_FIELD_LEVEL);
					goto Exit;
				}
				break;
			}

			case HTD_LEVEL_BACK_X:
			{
				FLMBYTE ucLevelsBack;

				if( RC_BAD(rc = read( &ucLevelsBack, 1, NULL)))
				{
					goto Exit;
				}

				if( ucPrevLevel >= ucLevelsBack)
				{
					ucLevel = (FLMBYTE)(ucPrevLevel - ucLevelsBack);
				}
				else
				{
					rc = RC_SET( FERR_BAD_FIELD_LEVEL);
					goto Exit;
				}
				break;
			}
		}

		// Allocate the record object

		if( ppRecord && ucLevel == 0)
		{
			if( *ppRecord)
			{
				if( (*ppRecord)->isReadOnly() || 
					(*ppRecord)->getRefCount() > 1)
				{
					(*ppRecord)->Release();

					if( (*ppRecord = f_new FlmRecord) == NULL)
					{
						rc = RC_SET( FERR_MEM);
						goto Exit;
					}
				}
				else
				{
					// Reuse the existing FlmRecord object.
					
					(*ppRecord)->clear();
				}
			}
			else
			{
				if( (*ppRecord = f_new FlmRecord) == NULL)
				{
					rc = RC_SET( FERR_MEM);
					goto Exit;
				}
			}
			
			(*ppRecord)->setContainerID( uiContainer);
			(*ppRecord)->setID( uiDrn);
		}

		// Allocate the attribute.

		if( pPool && ppNode)
		{
			pNode = GedNodeMake( pPool, ui16Tag, &rc);
			if( RC_BAD( rc))
			{
				goto Exit;
			}
		}

		bLeftTruncated = FALSE;
		bRightTruncated = FALSE;

		// Read the attribute's value.
		
		switch( ucType)
		{
			case HTD_TYPE_UNICODE:
			{
				FLMUNICODE * pUTF;
	
				if( pNode)
				{
					GedValTypeSet( pNode, FLM_TEXT_TYPE);
				}

				if( ppRecord)
				{
					if( RC_BAD( rc = (*ppRecord)->insertLast( ucLevel, 
						ui16Tag, FLM_TEXT_TYPE, &pField)))
					{
						goto Exit;
					}
				}

				if( !bHasValue)
				{
					break;
				}

				// Read UNICODE text in UTF-8 format.

				if( pPool)
				{
					if( RC_BAD( rc = readUTF( pPool, &pUTF)))
					{
						goto Exit;
					}

					if( pNode)
					{
						if( RC_BAD( rc = GedPutUNICODE( pPool, pNode, pUTF)))
						{
							goto Exit;
						}
					}

					if( ppRecord)
					{
						if( RC_BAD( rc = (*ppRecord)->setUnicode( pField, pUTF)))
						{
							goto Exit;
						}
					}
				}
				else
				{
					if( RC_BAD( rc = readUTF( NULL, NULL)))
					{
						goto Exit;
					}
				}
				break;
			}

			case HTD_TYPE_UINT:
			{
				FLMUINT32		ui32Value;

				if( pNode)
				{
					GedValTypeSet( pNode, FLM_NUMBER_TYPE);
				}

				if( ppRecord)
				{
					if( RC_BAD( rc = (*ppRecord)->insertLast( ucLevel, 
						ui16Tag, FLM_NUMBER_TYPE, &pField)))
					{
						goto Exit;
					}
				}

				if( !bHasValue)
				{
					break;
				}

				// Read an unsigned 32-bit integer.

				if( RC_BAD( rc = readUInt( &ui32Value)))
				{
					goto Exit;
				}

				if( pNode)
				{
					if( RC_BAD( rc = GedPutUINT( pPool, pNode, ui32Value)))
					{
						goto Exit;
					}
				}

				if( ppRecord)
				{
					if( RC_BAD( rc = (*ppRecord)->setUINT( pField, ui32Value)))
					{
						goto Exit;
					}
				}

				break;
			}
			
			case HTD_TYPE_UINT64:
			{
				FLMUINT64		ui64Value;

				if( pNode)
				{
					GedValTypeSet( pNode, FLM_NUMBER_TYPE);
				}

				if( ppRecord)
				{
					if( RC_BAD( rc = (*ppRecord)->insertLast( ucLevel, 
						ui16Tag, FLM_NUMBER_TYPE, &pField)))
					{
						goto Exit;
					}
				}

				if( !bHasValue)
				{
					break;
				}

				// Read an unsigned 64-bit integer.

				if( RC_BAD( rc = readUInt64( &ui64Value)))
				{
					goto Exit;
				}

				if( pNode)
				{
					if( RC_BAD( rc = GedPutUINT64( pPool, pNode, ui64Value)))
					{
						goto Exit;
					}
				}

				if( ppRecord)
				{
					if( RC_BAD( rc = (*ppRecord)->setUINT64( pField, ui64Value)))
					{
						goto Exit;
					}
				}

				break;
			}

			case HTD_TYPE_INT:
			{
				FLMINT32		i32Value;

				if( pNode)
				{
					GedValTypeSet( pNode, FLM_NUMBER_TYPE);
				}

				if( ppRecord)
				{
					if( RC_BAD( rc = (*ppRecord)->insertLast( ucLevel, 
						ui16Tag, FLM_NUMBER_TYPE, &pField)))
					{
						goto Exit;
					}
				}

				if( !bHasValue)
				{
					break;
				}

				// Read a signed 32-bit integer.

				if( RC_BAD( rc = readInt( &i32Value)))
				{
					goto Exit;
				}

				if( pNode)
				{
					if( RC_BAD( rc = GedPutINT( pPool, pNode, i32Value)))
					{
						goto Exit;
					}
				}

				if( ppRecord)
				{
					if( RC_BAD( rc = (*ppRecord)->setINT( pField, i32Value)))
					{
						goto Exit;
					}
				}

				break;
			}

			case HTD_TYPE_INT64:
			{
				FLMINT64		i64Value;

				if( pNode)
				{
					GedValTypeSet( pNode, FLM_NUMBER_TYPE);
				}

				if( ppRecord)
				{
					if( RC_BAD( rc = (*ppRecord)->insertLast( ucLevel, 
						ui16Tag, FLM_NUMBER_TYPE, &pField)))
					{
						goto Exit;
					}
				}

				if( !bHasValue)
				{
					break;
				}

				// Read a signed 64-bit integer.

				if( RC_BAD( rc = readInt64( &i64Value)))
				{
					goto Exit;
				}

				if( pNode)
				{
					if( RC_BAD( rc = GedPutINT64( pPool, pNode, i64Value)))
					{
						goto Exit;
					}
				}

				if( ppRecord)
				{
					if( RC_BAD( rc = (*ppRecord)->setINT64( pField, i64Value)))
					{
						goto Exit;
					}
				}

				break;
			}

			case HTD_TYPE_CONTEXT:
			{
				FLMUINT32		ui32Value;

				if( pNode)
				{
					GedValTypeSet( pNode, FLM_CONTEXT_TYPE);
				}

				if( ppRecord)
				{
					if( RC_BAD( rc = (*ppRecord)->insertLast( ucLevel, 
						ui16Tag, FLM_CONTEXT_TYPE, &pField)))
					{
						goto Exit;
					}
				}

				if( !bHasValue)
				{
					break;
				}

				// Read an unsigned 32-bit integer.

				if( RC_BAD( rc = readUInt( &ui32Value)))
				{
					goto Exit;
				}

				if( pNode)
				{
					if( RC_BAD( rc = GedPutRecPtr( pPool, pNode, ui32Value)))
					{
						goto Exit;
					}
				}

				if( ppRecord)
				{
					if( RC_BAD( rc = (*ppRecord)->setRecPointer( pField, ui32Value)))
					{
						goto Exit;
					}
				}

				break;
			}

			case HTD_TYPE_BINARY:
			{
				FLMUINT16	ui16DataSize;
				FLMBYTE *	pucData = NULL;

				if( pNode)
				{
					GedValTypeSet( pNode, FLM_BINARY_TYPE);
				}

				if( ppRecord)
				{
					if( RC_BAD( rc = (*ppRecord)->insertLast( ucLevel, 
						ui16Tag, FLM_BINARY_TYPE, &pField)))
					{
						goto Exit;
					}
				}

				if( !bHasValue)
				{
					break;
				}

				// Read a binary data stream.

				if( RC_BAD( rc = readUShort( &ui16DataSize)))
				{
					goto Exit;
				}

				if( pPool)
				{
					if( pNode)
					{
						if( (pucData = (FLMBYTE *)GedAllocSpace( pPool, pNode,
							FLM_BINARY_TYPE, ui16DataSize)) == NULL)
						{
							rc = RC_SET( FERR_MEM);
							goto Exit;
						}
					}
					else if( ppRecord)
					{
						if( RC_BAD(rc = (*ppRecord)->allocStorageSpace( pField,
							FLM_BINARY_TYPE, ui16DataSize, 0, 0, 0, &pucData, NULL)))
						{
							goto Exit;
						}
					}

					if( RC_BAD( rc = read( pucData, ui16DataSize, NULL)))
					{
						goto Exit;
					}

					if( pNode)
					{
						if( ppRecord)
						{
							if( RC_BAD( rc = (*ppRecord)->setBinary( pField, 
								pucData, ui16DataSize)))
							{
								goto Exit;
							}
						}
					}
				}
				else
				{
					if( RC_BAD( rc = skip( ui16DataSize)))
					{
						goto Exit;
					}
				}
				break;
			}

			case HTD_TYPE_DATE:
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
				goto Exit;
			}

			case HTD_TYPE_GEDCOM:
			{
				FLMBYTE		ucGedType;
				FLMUINT16	ui16DataSize;
				FLMBYTE *	pucData = NULL;

				// Read the GEDCOM data type and flags

				if( RC_BAD( rc = read( &ucGedType, 1, NULL)))
				{
					goto Exit;
				}
				ucFlags = ucGedType & 0xF0;
				ucGedType &= 0x0F;

				if( ucFlags & 0x10)
				{
					bLeftTruncated = TRUE;
				}

				if( ucFlags & 0x20)
				{
					bRightTruncated = TRUE;
				}

				if( ucGedType != FLM_TEXT_TYPE &&
					ucGedType != FLM_NUMBER_TYPE &&
					ucGedType != FLM_BINARY_TYPE &&
					ucGedType != FLM_BLOB_TYPE &&
					ucGedType != FLM_CONTEXT_TYPE)
				{
					rc = RC_SET( FERR_NOT_IMPLEMENTED);
					goto Exit;
				}

				if( pNode)
				{
					GedValTypeSet( pNode, ucGedType);
					if( bLeftTruncated)
					{
						GedSetLeftTruncated( pNode);
					}

					if( bRightTruncated)
					{
						GedSetRightTruncated( pNode);
					}
				}

				if( ppRecord)
				{
					if( RC_BAD( rc = (*ppRecord)->insertLast( ucLevel, 
						ui16Tag, ucGedType, &pField)))
					{
						goto Exit;
					}

					if( bLeftTruncated)
					{
						(*ppRecord)->setLeftTruncated( pField, TRUE);
					}

					if( bRightTruncated)
					{
						(*ppRecord)->setRightTruncated( pField, TRUE);
					}
				}

				if( !bHasValue)
				{
					break;
				}

				// Read the data size.

				if( RC_BAD( rc = readUShort( &ui16DataSize)))
				{
					goto Exit;
				}

				// Read the data value.

				if( pPool)
				{
					if( pNode)
					{
						if( (pucData = (FLMBYTE *)GedAllocSpace( pPool, pNode,
							ucGedType, ui16DataSize)) == NULL)
						{
							rc = RC_SET( FERR_MEM);
							goto Exit;
						}
					}
					else if( ppRecord)
					{
						if (RC_BAD( rc = (*ppRecord)->allocStorageSpace( pField,
							ucGedType, ui16DataSize, 0, 0, 0, &pucData, NULL)))
						{
							goto Exit;
						}
					}

					if( RC_BAD( rc = read( pucData, ui16DataSize, NULL)))
					{
						goto Exit;
					}

					if( pNode)
					{
						if( ppRecord)
						{
							if( RC_BAD( rc = (*ppRecord)->setBinary( pField, 
								pucData, ui16DataSize)))
							{
								goto Exit;
							}
						}
					}
				}
				else
				{
					if( RC_BAD( rc = skip( ui16DataSize)))
					{
						goto Exit;
					}
				}
				break;
			}

			default:
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
				goto Exit;
			}
		}

		// Set the truncation flags

		if( ucType != HTD_TYPE_GEDCOM)
		{
			if( pNode)
			{
				if( bLeftTruncated)
				{
					GedSetLeftTruncated( pNode);
				}

				if( bRightTruncated)
				{
					GedSetRightTruncated( pNode);
				}
			}
			else if( pField)
			{
				if( bLeftTruncated)
				{
					(*ppRecord)->setLeftTruncated( pField, TRUE);
				}

				if( bRightTruncated)
				{
					(*ppRecord)->setRightTruncated( pField, TRUE);
				}
			}
		}
		
		// Graft the attribute into the tree.
		
		if( pNode)
		{
			if( pRoot == NULL)
			{
				pRoot = pNode;
			}
			else
			{
				if( bSibling)
				{
					pPrevNode->next = pNode;
					pNode->prior = pPrevNode;
					GedNodeLevelSet( pNode, GedNodeLevel( pPrevNode));
				}
				else if( bChild)
				{
					pPrevNode->next = pNode;
					pNode->prior = pPrevNode;
					GedNodeLevelSet( pNode, GedNodeLevel( pPrevNode) + 1);
				}
				else
				{
					pPrevNode->next = pNode;
					pNode->prior = pPrevNode;
					GedNodeLevelSet( pNode, ucLevel);
				}
			}
		}

		ucPrevLevel = ucLevel;
		pPrevNode = pNode;

		// Reset the pool if a GEDCOM record is not going to be returned.

		if( pPool && !ppNode)
		{
			pPool->poolReset( pvMark);
		}
	}

Exit:

	if( RC_OK( rc))
	{
		if( ppNode)
		{
			*ppNode = pRoot;
		}
	}
	else
	{
		if( ppRecord && *ppRecord)
		{
			(*ppRecord)->Release();
		}
	}

	if( pPool && !ppNode)
	{
		pPool->poolReset( pvMark);
	}

	return( rc);		
}

/****************************************************************************
Desc:
****************************************************************************/
FCS_DOS::FCS_DOS( void)
{
	m_pOStream = NULL;
	m_uiBOffset = 0;
	m_tmpPool.poolInit( 512);
	m_bSetupCalled = FALSE;
}


/****************************************************************************
Desc:
****************************************************************************/
FCS_DOS::~FCS_DOS( void)
{
	if( m_bSetupCalled)
	{
		(void)close();
	}
	
	m_tmpPool.poolFree();
}

/****************************************************************************
Desc:	Writes a specified number of bytes from a buffer to the output
		stream.
****************************************************************************/
RCODE FCS_DOS::write(
	FLMBYTE *		pucData,
	FLMUINT			uiLength)
{
	RCODE		rc = FERR_OK;

	// Verify that setup has been called.

	flmAssert( m_bSetupCalled == TRUE);

	// Write the data.

Retry_Write:

	if( FCS_DOS_BUFFER_SIZE - m_uiBOffset >= uiLength)
	{
		f_memcpy( &(m_pucBuffer[ m_uiBOffset]), pucData, uiLength);
		m_uiBOffset += uiLength;
	}
	else
	{
		if( m_uiBOffset > 0)
		{
			if( RC_BAD( rc = flush()))
			{
				goto Exit;
			}
		}

		if( uiLength <= FCS_DOS_BUFFER_SIZE)
		{
			goto Retry_Write;
		}

		if( RC_BAD( rc = m_pOStream->write( pucData, uiLength)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:	Writes a UNICODE string to the stream in the UTF-8 format.
****************************************************************************/
RCODE	FCS_DOS::writeUTF(
	FLMUNICODE *	puzValue)
{
	FLMUINT			uiUTFLen;
	FLMUNICODE *	puzTmp;
	RCODE				rc = FERR_OK;
	
	// Verify that setup has been called.

	flmAssert( m_bSetupCalled == TRUE);

	// Verify pValue is valid.
	
	flmAssert( puzValue != NULL);

	// Determine the size of the string.
	
	uiUTFLen = 0;
	puzTmp = puzValue;
	while( *puzTmp)
	{
		uiUTFLen++;
		puzTmp++;
	}

	if( RC_BAD( rc = writeUShort( (FLMUINT16)uiUTFLen)))
	{
		goto Exit;
	}

	puzTmp = puzValue;
	while( *puzTmp)
	{
		if( *puzTmp <= 0x007F)
		{
			if( RC_BAD( rc = writeByte( (FLMBYTE)(*puzTmp))))
			{
				goto Exit;
			}
		}
		else if( *puzTmp >= 0x0080 && *puzTmp <= 0x07FF)
		{
			if( RC_BAD( rc = writeUShort((FLMUINT16)
				((((FLMUINT16)(0xC0 | (FLMBYTE)((*puzTmp & 0x07C0) >> 6))) << 8) |
				(FLMUINT16)(0x80 | (FLMBYTE)(*puzTmp & 0x003F))))))
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = writeUShort((FLMUINT16)
				((((FLMUINT16)(0xE0 | (FLMBYTE)((*puzTmp & 0xF000) >> 12))) << 8) |
				(FLMUINT16)(0x80 | (FLMBYTE)((*puzTmp & 0x0FC0) >> 6))))))
			{
				goto Exit;
			}

			if( RC_BAD( rc = writeByte( (0x80 | (FLMBYTE)(*puzTmp & 0x003F)))))
			{
				goto Exit;
			}
		}

		puzTmp++;
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:	Writes a binary token (including length) to the stream.
****************************************************************************/
RCODE FCS_DOS::writeBinary(
	FLMBYTE *	pucValue,
	FLMUINT		uiSize)
{
	RCODE			rc = FERR_OK;

	flmAssert( uiSize <= 0x0000FFFF);

	if( RC_BAD( rc = writeUShort( (FLMUINT16)uiSize)))
	{
		goto Exit;
	}

	if( uiSize)
	{
		if( RC_BAD( rc = write( pucValue, uiSize)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Writes a large binary token (including length) to the stream.
****************************************************************************/
RCODE FCS_DOS::writeLargeBinary(
	FLMBYTE *	pucValue,
	FLMUINT		uiSize)
{
	RCODE			rc = FERR_OK;

	if( RC_BAD( rc = writeUInt32( (FLMUINT32)uiSize)))
	{
		goto Exit;
	}

	if( uiSize)
	{
		if( RC_BAD( rc = write( pucValue, uiSize)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Writes a Hierarchical Tagged Data record to the stream.
****************************************************************************/
RCODE FCS_DOS::writeHTD(
	NODE *		pHTD,
	FlmRecord *	pRecord,
	FLMBOOL		bSendForest,
	FLMBOOL		bSendAsGedcom)
{
	FLMUINT		uiPrevLevel = 0;
	FLMUINT		uiLevelsBack = 0;
	FLMUINT		uiDescriptor = 0;
	FLMUINT		uiCurLevel = 0;
	FLMUINT		uiCurValType = 0;
	FLMUINT		uiCurDataLen = 0;
	FLMBOOL		bLeftTruncated;
	FLMBOOL		bRightTruncated;
	FLMBYTE *	pucCurData = NULL;
	FLMBYTE		pucTmpBuf[ 32];
	void *		pvMark = m_tmpPool.poolMark();
	NODE *		pCurNode = NULL;
	void *		pCurField = NULL;
	RCODE			rc = FERR_OK;

	// Verify that setup has been called.

	flmAssert( m_bSetupCalled == TRUE);

	// Set the current node or field

	if( pHTD)
	{
		pCurNode = pHTD;
	}
	else
	{
		pCurField = pRecord->root();
	}

	while( pCurNode || pCurField)
	{
		// See if we are done sending the tree/forest.

		if( pCurNode)
		{
			if( !bSendForest && (pCurNode != pHTD) &&
				(GedNodeLevel( pCurNode) == GedNodeLevel( pHTD)))
			{
				break;
			}
		}

		// Output the attribute's tag number.

		if( pCurNode)
		{
			f_UINT16ToBigEndian( (FLMUINT16)GedTagNum( pCurNode), pucTmpBuf);
		}
		else if( pCurField)
		{
			f_UINT16ToBigEndian( (FLMUINT16)pRecord->getFieldID( pCurField), pucTmpBuf);
		}

		if( RC_BAD( rc = write( pucTmpBuf, 2)))
		{
			goto Exit;
		}

		// Setup the attribute's descriptor.

		uiDescriptor = 0;
		uiLevelsBack = 0;

		if( pCurNode)
		{
			uiCurLevel = GedNodeLevel( pCurNode);
		}
		else
		{
			uiCurLevel = pRecord->getLevel( pCurField);
		}

		if( uiCurLevel == uiPrevLevel)
		{
			(void)(uiDescriptor |= (HTD_LEVEL_SIBLING << HTD_LEVEL_POS));
		}
		else if( uiCurLevel == uiPrevLevel + 1)
		{
			uiDescriptor |= (HTD_LEVEL_CHILD << HTD_LEVEL_POS);
		}
		else if( uiCurLevel == uiPrevLevel - 1)
		{
			uiDescriptor |= (HTD_LEVEL_BACK << HTD_LEVEL_POS);
		}
		else if( uiCurLevel < uiPrevLevel)
		{
			uiDescriptor |= (HTD_LEVEL_BACK_X << HTD_LEVEL_POS);
			uiLevelsBack = uiPrevLevel - uiCurLevel;
		}
		else
		{
			flmAssert( 0);
			rc = RC_SET( FERR_FAILURE);
			goto Exit;
		}

		if( pCurNode)
		{
			uiCurDataLen = GedValLen( pCurNode);
			uiCurValType = GedValType( pCurNode) & 0x0F;
			bLeftTruncated = GedIsLeftTruncated( pCurNode);
			bRightTruncated = GedIsRightTruncated( pCurNode);
			pucCurData = (FLMBYTE *)GedValPtr( pCurNode);
		}
		else
		{
			uiCurDataLen = pRecord->getDataLength( pCurField);
			uiCurValType = (FLMUINT)pRecord->getDataType( pCurField);
			bLeftTruncated = pRecord->isLeftTruncated( pCurField);
			bRightTruncated = pRecord->isRightTruncated( pCurField);
			pucCurData = (FLMBYTE *)(pRecord->getDataPtr( pCurField));
		}

		if( uiCurDataLen)
		{
			uiDescriptor |= HTD_HAS_VALUE_FLAG;
		}

		if( bSendAsGedcom)
		{
			uiDescriptor |= HTD_TYPE_GEDCOM;
		}
		else
		{
			switch( uiCurValType)
			{
				case FLM_TEXT_TYPE:
				{
					uiDescriptor |= HTD_TYPE_UNICODE;
					break;
				}

				case FLM_NUMBER_TYPE:
				{
					// To save conversion time, cheat to determine if
					// the number is negative.

					if( ((*pucCurData & 0xF0) == 0xB0))
					{
						FLMINT64		i64Value;

						if( (pCurNode && RC_BAD( rc = GedGetINT64( 
								pCurNode, &i64Value))) ||
							(pCurField && RC_BAD( rc = pRecord->getINT64( 
								pCurField, &i64Value))))
						{
							goto Exit;
						}
						if (i64Value >= (FLMINT64)(FLM_MIN_INT32) &&
							 i64Value <= (FLMINT64)(FLM_MAX_INT32))
						{
							uiDescriptor |= HTD_TYPE_INT;
						}
						else
						{
							uiDescriptor |= HTD_TYPE_INT64;
						}
					}
					else
					{
						FLMUINT64		ui64Value;

						if( (pCurNode && RC_BAD( rc = GedGetUINT64( 
								pCurNode, &ui64Value))) ||
							(pCurField && RC_BAD( rc = pRecord->getUINT64( 
								pCurField, &ui64Value))))
						{
							goto Exit;
						}
						if (ui64Value <= (FLMUINT64)(FLM_MAX_UINT32))
						{
							uiDescriptor |= HTD_TYPE_UINT;
						}
						else
						{
							uiDescriptor |= HTD_TYPE_UINT64;
						}
					}
					break;
				}

				case FLM_CONTEXT_TYPE:
				{
					uiDescriptor |= HTD_TYPE_CONTEXT;
					break;
				}

				case FLM_BINARY_TYPE:
				{
					uiDescriptor |= HTD_TYPE_BINARY;
					break;
				}

				default:
				{
					rc = RC_SET( FERR_NOT_IMPLEMENTED);
					goto Exit;
				}
			}
		}

		// Output the attribute's descriptor.

		pucTmpBuf[ 0] = (FLMBYTE)uiDescriptor;
		if( RC_BAD( rc = write( pucTmpBuf, 1)))
		{
			goto Exit;
		}

		// Output the "levels back" value (if available).

		if( uiLevelsBack)
		{
			flmAssert( uiLevelsBack <= 0xFF);
			pucTmpBuf[ 0] = (FLMBYTE)uiLevelsBack;
			if( RC_BAD( rc = write( pucTmpBuf, 1)))
			{
				goto Exit;
			}
		}

		// Output the attribute's value.

		if( bSendAsGedcom)
		{
			// Output the GEDCOM data type and flags

			pucTmpBuf[ 0] = (FLMBYTE)uiCurValType;
			if( bLeftTruncated)
			{
				pucTmpBuf[ 0] |= 0x10;
			}

			if( bRightTruncated)
			{
				pucTmpBuf[ 0] |= 0x20;
			}

			if( RC_BAD( rc = write( pucTmpBuf, 1)))
			{
				goto Exit;
			}

			if( uiCurDataLen)
			{
				// Output the data size.
				
				flmAssert( uiCurDataLen <= 0x0000FFFF);

				f_UINT16ToBigEndian( (FLMUINT16)uiCurDataLen, pucTmpBuf);
				if( RC_BAD( rc = write( pucTmpBuf, 2)))
				{
					goto Exit;
				}

				// Send the data.

				if( RC_BAD( rc = write( pucCurData, uiCurDataLen)))
				{
					goto Exit;
				}
			}
		}
		else
		{
			// Send the value.

			switch( uiCurValType)
			{
				case FLM_TEXT_TYPE:
				{
					// Extract the value.

					if( uiCurDataLen)
					{
						FLMUINT			uiBufSize;
						FLMUNICODE *	puzValue;

						// Reset the temporary pool.

						m_tmpPool.poolReset( pvMark);
						if( uiCurDataLen <= 32751)
						{
							// Allocate a buffer that is twice the size of the
							// attribute's value length.  This is necessary because the
							// UNICODE conversion will may double the size of the
							// attribute's value.  A "safety" zone of 32 bytes is added
							// to the buffer size to allow for strings that may require
							// more than 2x the attribute's size and to account for
							// null-termination bytes.

							uiBufSize = (2 * uiCurDataLen) + 32;
						}
						else
						{
							// Allocate a full 64K.

							uiBufSize = 65535;
						}
						
						if( RC_BAD( rc = m_tmpPool.poolAlloc( uiBufSize, 
							(void **)&puzValue)))
						{
							goto Exit;
						}

						// Extract UNICODE from the attribute.
						
						if( (pCurNode && RC_BAD( rc = GedGetUNICODE( 
								pCurNode, puzValue, &uiBufSize))) ||
							(pCurField && RC_BAD( rc = pRecord->getUnicode( 
								pCurField, puzValue, &uiBufSize))))
						{
							if( rc == FERR_CONV_DEST_OVERFLOW)
							{
								// Since we did not correctly guess the buffer size,
								// try again.  This time, take the slow (but safe)
								// approach of calculating the size of the UNICODE string.

								if( (pCurNode && RC_BAD( rc = GedGetUNICODE( 
										pCurNode, NULL, &uiBufSize))) ||
									(pCurField && RC_BAD( rc = pRecord->getUnicodeLength(
										pCurField, &uiBufSize))))
								{
									goto Exit;
								}

								// Add two bytes to account for null-termination.

								uiBufSize += 2;

								// Reset the pool to clear the prior allocation.

								m_tmpPool.poolReset( pvMark);

								// Allocate the new buffer.
								
								if( RC_BAD( rc = m_tmpPool.poolAlloc( uiBufSize,
									(void **)&puzValue)))
								{
									goto Exit;
								}
								
								// Extract the UNICODE string.
								
								if( (pCurNode && RC_BAD( rc = GedGetUNICODE( 
											pCurNode, puzValue, &uiBufSize))) ||
									 (pCurField && RC_BAD( rc = pRecord->getUnicode( 
									 		pCurField, puzValue, &uiBufSize))))
								{
									goto Exit;
								}
							}
							else
							{
								goto Exit;
							}
						}

						// Write the attribute's value.

						if( RC_BAD( rc = writeUTF( puzValue)))
						{
							goto Exit;
						}
					}
						
					break;
				}

				case FLM_NUMBER_TYPE:
				{
					if( uiCurDataLen)
					{	
						if( uiDescriptor & HTD_TYPE_INT64)
						{
							// Since the number is negative, extract and send it
							// as a signed 64-bit value.

							FLMINT64		i64Value;

							if( (pCurNode && RC_BAD( rc = GedGetINT64( 
									pCurNode, &i64Value))) ||
								(pCurField && RC_BAD( rc = pRecord->getINT64( 
									pCurField, &i64Value))))
							{
								goto Exit;
							}

							// Write the value.

							if( RC_BAD( rc = writeInt64( i64Value)))
							{
								goto Exit;
							}
						}
						else if( uiDescriptor & HTD_TYPE_INT)
						{
							// Since the number is negative, extract and send it
							// as a signed 32-bit value.

							FLMINT32		i32Value;

							if( (pCurNode && RC_BAD( rc = GedGetINT32( 
									pCurNode, &i32Value))) ||
								(pCurField && RC_BAD( rc = pRecord->getINT32( 
									pCurField, &i32Value))))
							{
								goto Exit;
							}

							// Write the value.

							if( RC_BAD( rc = writeInt32( i32Value)))
							{
								goto Exit;
							}
						}
						else if( uiDescriptor & HTD_TYPE_UINT64)
						{
							// The number is non-negative 64 bit

							FLMUINT64		ui64Value;

							if( (pCurNode && RC_BAD( rc = GedGetUINT64( 
									pCurNode, &ui64Value))) ||
								(pCurField && RC_BAD( rc = pRecord->getUINT64( 
									pCurField, &ui64Value))))
							{
								goto Exit;
							}

							// Write the value.

							if( RC_BAD( rc = writeUInt64( ui64Value)))
							{
								goto Exit;
							}
						}
						else
						{
							flmAssert( uiDescriptor & HTD_TYPE_UINT);
							
							// The number is non-negative 32 bit

							FLMUINT32		ui32Value;

							if( (pCurNode && RC_BAD( rc = GedGetUINT32( 
									pCurNode, &ui32Value))) ||
								(pCurField && RC_BAD( rc = pRecord->getUINT32( 
									pCurField, &ui32Value))))
							{
								goto Exit;
							}

							// Write the value.

							if( RC_BAD( rc = writeUInt32( ui32Value)))
							{
								goto Exit;
							}
						}
					}
					break;
				}

				case FLM_CONTEXT_TYPE:
				{
					// Extract the value.

					if( uiCurDataLen)
					{
						// The context node has a DRN value associated with
						// it.  Send the value as an unsigned 32-bit number.

						FLMUINT		uiDrn;

						if( (pCurNode && RC_BAD( rc = GedGetRecPtr( 
								pCurNode, &uiDrn))) ||
							(pCurField && RC_BAD( rc = pRecord->getUINT( 
								pCurField, &uiDrn))))
						{
							goto Exit;
						}

						if( RC_BAD( rc = writeUInt32( (FLMUINT32)uiDrn)))
						{
							goto Exit;
						}
					}
					break;
				}

				case FLM_BINARY_TYPE:
				{
					// Extract the value.

					if( uiCurDataLen)
					{
						if( RC_BAD( rc = writeUShort( (FLMUINT16)uiCurDataLen)))
						{
							goto Exit;
						}

						if( RC_BAD( rc = write( pucCurData, uiCurDataLen)))
						{
							goto Exit;
						}
					}
					break;
				}

				default:
				{
					rc = RC_SET( FERR_NOT_IMPLEMENTED);
					goto Exit;
				}
			}
		}

		uiPrevLevel = uiCurLevel;
		if( pCurNode)
		{
			pCurNode = GedNodeNext( pCurNode);
		}
		else
		{
			pCurField = pRecord->next( pCurField);
		}
	}

	// Write a zero tag to indicate the end of the transmission.
	
	if( RC_BAD( rc = writeUShort( 0)))
	{
		goto Exit;
	}

Exit:

	m_tmpPool.poolReset( pvMark);
	return( rc);		
}


/****************************************************************************
Desc:	Flushes any pending data and closes the stream.
****************************************************************************/
RCODE FCS_DOS::close( void)
{
	RCODE		rc = FERR_OK;

	// Verify that setup has been called.

	flmAssert( m_bSetupCalled == TRUE);

	// Flush and terminate any pending message.

	if( RC_BAD( rc = endMessage()))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Flushes pending data.
****************************************************************************/
RCODE	FCS_DOS::flush( void)
{
	// Verify that setup has been called.

	flmAssert( m_bSetupCalled == TRUE);

	// Flush the output buffer.

	if( m_uiBOffset > 0)
	{
		m_pOStream->write( m_pucBuffer, m_uiBOffset);
		m_uiBOffset = 0;
	}

	// Flush the parent stream.

	return( m_pOStream->flush());
}

/****************************************************************************
Desc:	Flushes and terminates the current parent stream message
****************************************************************************/
RCODE	FCS_DOS::endMessage( void)
{
	RCODE		rc = FERR_OK;

	// Verify that Setup has been called.

	flmAssert( m_bSetupCalled == TRUE);

	if( !m_pOStream)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
	
	// Flush any pending data.

	if( RC_BAD( rc = flush()))
	{
		goto Exit;
	}

	// Terminate the message.

	if( RC_BAD( rc = m_pOStream->endMessage()))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
FCS_FIS::FCS_FIS( void)
{
	m_pFileHdl = NULL;
	m_pucBufPos = NULL;
	m_pucBuffer = NULL;
	m_uiFileOffset = 0;
	m_uiBlockSize = 0;
	m_uiBlockEnd = 0;
}

/****************************************************************************
Desc:
*****************************************************************************/
FCS_FIS::~FCS_FIS( void)
{
	if( m_pFileHdl)
	{
		m_pFileHdl->Release();
	}

	if( m_pucBuffer)
	{
		f_free( &m_pucBuffer);
	}
}

/****************************************************************************
Desc:	Configures the input stream for use
*****************************************************************************/
RCODE FCS_FIS::setup(
	const char *	pszFilePath,
	FLMUINT			uiBlockSize)
{
	RCODE			rc = FERR_OK;
	
	flmAssert( uiBlockSize);

	if( RC_BAD( rc = close()))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gv_FlmSysData.pFileSystem->openFile( pszFilePath,
		FLM_IO_RDONLY | FLM_IO_SH_DENYNONE, &m_pFileHdl)))
	{
		goto Exit;
	}

	m_uiBlockSize = uiBlockSize;
	if( RC_BAD( rc = f_alloc( m_uiBlockSize, &m_pucBuffer)))
	{
		goto Exit;
	}
	m_pucBufPos = m_pucBuffer;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Closes the input stream and frees any resources
*****************************************************************************/
RCODE FCS_FIS::close( void)
{
	if( m_pFileHdl)
	{
		m_pFileHdl->Release();
		m_pFileHdl = NULL;
	}

	if( m_pucBuffer)
	{
		f_free( &m_pucBuffer);
	}
	
	return( FERR_OK);
}

/****************************************************************************
Desc:	Reads the requested amount of data from the stream.
*****************************************************************************/
RCODE FCS_FIS::read(
	FLMBYTE *		pucData,
	FLMUINT			uiLength,
	FLMUINT *		puiBytesRead)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiBytesRead = 0;
	FLMUINT		uiMaxSize;

	if( puiBytesRead)
	{
		*puiBytesRead = 0;
	}

	if( !m_pFileHdl)
	{
		rc = RC_SET( FERR_READING_FILE);
		goto Exit;
	}

	while( uiLength)
	{
		uiMaxSize = m_uiBlockEnd - (FLMUINT)(m_pucBufPos - m_pucBuffer);

		if( !uiMaxSize)
		{
			if( RC_BAD( rc = getNextPacket()))
			{
				goto Exit;
			}
		}
		else if( uiLength <= uiMaxSize)
		{
			f_memcpy( pucData, m_pucBufPos, uiLength);
			m_pucBufPos += uiLength;
			uiBytesRead += uiLength;
			uiLength = 0;
		}
		else
		{
			f_memcpy( pucData, m_pucBufPos, uiMaxSize);
			m_pucBufPos += uiMaxSize;
			pucData += uiMaxSize;
			uiLength -= uiMaxSize;
			uiBytesRead += uiMaxSize;
		}
	}

Exit:

	if( puiBytesRead)
	{
		*puiBytesRead = uiBytesRead;
	}

	return( rc);
}

/****************************************************************************
Desc:	Flushes any pending data.
*****************************************************************************/
RCODE FCS_FIS::flush( void)
{
	return( FERR_OK);
}

/****************************************************************************
Desc:	Flushes any pending data.
*****************************************************************************/
RCODE FCS_FIS::endMessage( void)
{
	return( FERR_OK);
}

/****************************************************************************
Desc:	Returns TRUE if the stream is open
*****************************************************************************/
FLMBOOL FCS_FIS::isOpen( void)
{
	return( TRUE);
}

/****************************************************************************
Desc:	Reads the next block from the file
*****************************************************************************/
RCODE FCS_FIS::getNextPacket( void)
{
	RCODE			rc = FERR_OK;

	if( RC_BAD( rc = m_pFileHdl->read( m_uiFileOffset, m_uiBlockSize,
		m_pucBuffer, &m_uiBlockEnd)))
	{
		if( rc == FERR_IO_END_OF_FILE)
		{
			if( !m_uiBlockEnd)
			{
				goto Exit;
			}
			else
			{
				rc = FERR_OK;
			}
		}
	}

	m_uiFileOffset += m_uiBlockEnd;
	m_pucBufPos = m_pucBuffer;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Converts a UNICODE string consisting of 7-bit ASCII characters to
		a native string.
*****************************************************************************/
RCODE fcsConvertUnicodeToNative(
	F_Pool *					pPool,
	const FLMUNICODE *	puzUnicode,
	char **					ppucNative)
{
	RCODE			rc = FERR_OK;
	char *		pucDest = NULL;
	FLMUINT		uiCount;

	uiCount = 0;
	while( puzUnicode[ uiCount])
	{
		if( puzUnicode[ uiCount] > 0x007F)
		{
			rc = RC_SET( FERR_CONV_ILLEGAL);
			goto Exit;
		}
		uiCount++;
	}
	
	if( RC_BAD( rc = pPool->poolAlloc( uiCount + 1, (void **)&pucDest)))
	{
		goto Exit;
	}

	uiCount = 0;
	while( puzUnicode[ uiCount])
	{
		pucDest[ uiCount] = f_tonative( (FLMBYTE)puzUnicode[ uiCount]);
		uiCount++;
	}

	pucDest[ uiCount] = '\0';

Exit:

	*ppucNative = pucDest;
	return( rc);
}

/****************************************************************************
Desc:	Converts a native string to a double-byte UNICODE string.
*****************************************************************************/
RCODE fcsConvertNativeToUnicode(
	F_Pool *				pPool,
	const char *		pszNative,
	FLMUNICODE **		ppuzUnicode)
{
	RCODE				rc = FERR_OK;
	FLMUNICODE *	puzDest;
	FLMUINT			uiCount;
	
	uiCount = f_strlen( pszNative);
	
	if( RC_BAD( rc = pPool->poolAlloc( uiCount + 1, (void **)&puzDest)))
	{
		goto Exit;
	}
	
	uiCount = 0;
	while( pszNative[ uiCount])
	{
		puzDest[ uiCount] = (FLMUNICODE)f_toascii( pszNative[ uiCount]);
		uiCount++;
	}

	puzDest[ uiCount] = 0;

Exit:

	*ppuzUnicode = puzDest;
	return( rc);
}


/****************************************************************************
Desc:	Initializes members of a CREATE_OPTS structure to their default values
*****************************************************************************/
void fcsInitCreateOpts(
	CREATE_OPTS *	pCreateOptsRV)
{
	f_memset( pCreateOptsRV, 0, sizeof( CREATE_OPTS));

	pCreateOptsRV->uiBlockSize = DEFAULT_BLKSIZ;
	pCreateOptsRV->uiMinRflFileSize = DEFAULT_MIN_RFL_FILE_SIZE;
	pCreateOptsRV->uiMaxRflFileSize = DEFAULT_MAX_RFL_FILE_SIZE;
	pCreateOptsRV->bKeepRflFiles = DEFAULT_KEEP_RFL_FILES_FLAG;
	pCreateOptsRV->bLogAbortedTransToRfl = DEFAULT_LOG_ABORTED_TRANS_FLAG;
	pCreateOptsRV->uiDefaultLanguage = DEFAULT_LANG;
	pCreateOptsRV->uiVersionNum = FLM_CUR_FILE_FORMAT_VER_NUM;
}

/****************************************************************************
Desc:	Converts a CHECKPOINT_INFO structure to an HTD tree
*****************************************************************************/
RCODE fcsBuildCheckpointInfo(
	CHECKPOINT_INFO *		pChkptInfo,
	F_Pool *					pPool,
	NODE **					ppTree)
{
	NODE *		pRootNd = NULL;
	void *		pMark = pPool->poolMark();
	FLMUINT		uiTmp;
	RCODE			rc = FERR_OK;

	*ppTree = NULL;

	// Build the root node of the tree.

	if( (pRootNd = GedNodeMake( pPool, FCS_CPI_CONTEXT, &rc)) == NULL)
	{
		goto Exit;
	}

	// Add fields to the tree.

	if( pChkptInfo->bRunning)
	{
		uiTmp = 1;
		if( RC_BAD( rc = gedAddField( pPool, pRootNd,
			FCS_CPI_RUNNING, (void *)&uiTmp,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pChkptInfo->uiRunningTime)
	{
		if( RC_BAD( rc = gedAddField( pPool, pRootNd,
			FCS_CPI_START_TIME, (void *)&pChkptInfo->uiRunningTime,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pChkptInfo->bForcingCheckpoint)
	{
		uiTmp = 1;
		if( RC_BAD( rc = gedAddField( pPool, pRootNd,
			FCS_CPI_FORCING_CP, (void *)&uiTmp,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pChkptInfo->uiForceCheckpointRunningTime)
	{
		if( RC_BAD( rc = gedAddField( pPool, pRootNd,
			FCS_CPI_FORCE_CP_START_TIME,
			(void *)&pChkptInfo->uiForceCheckpointRunningTime,
			4, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pChkptInfo->iForceCheckpointReason)
	{
		uiTmp = pChkptInfo->iForceCheckpointReason;
		if( RC_BAD( rc = gedAddField( pPool, pRootNd,
			FCS_CPI_FORCE_CP_REASON, (void *)&uiTmp,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pChkptInfo->bWritingDataBlocks)
	{
		uiTmp = 1;
		if( RC_BAD( rc = gedAddField( pPool, pRootNd,
			FCS_CPI_WRITING_DATA_BLOCKS, (void *)&uiTmp,
			4, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pChkptInfo->uiLogBlocksWritten)
	{
		if( RC_BAD( rc = gedAddField( pPool, pRootNd,
			FCS_CPI_LOG_BLOCKS_WRITTEN, 
			(void *)&pChkptInfo->uiLogBlocksWritten,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pChkptInfo->uiDataBlocksWritten)
	{
		if( RC_BAD( rc = gedAddField( pPool, pRootNd,
			FCS_CPI_DATA_BLOCKS_WRITTEN,
			(void *)&pChkptInfo->uiDataBlocksWritten,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pChkptInfo->uiDirtyCacheBytes)
	{
		if( RC_BAD( rc = gedAddField( pPool, pRootNd,
			FCS_CPI_DIRTY_CACHE_BYTES,
			(void *)&pChkptInfo->uiDirtyCacheBytes,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pChkptInfo->uiBlockSize)
	{
		if( RC_BAD( rc = gedAddField( pPool, pRootNd,
			FCS_CPI_BLOCK_SIZE, (void *)&pChkptInfo->uiBlockSize,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pChkptInfo->uiWaitTruncateTime)
	{
		if( RC_BAD( rc = gedAddField( pPool, pRootNd,
			FCS_CPI_WAIT_TRUNC_TIME, (void *)&pChkptInfo->uiWaitTruncateTime,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	*ppTree = pRootNd;

Exit:

	if( RC_BAD( rc))
	{
		pPool->poolReset( pMark);
	}

	return( rc);
}

/****************************************************************************
Desc:	Converts a F_LOCK_USER structure (or list of structures) to an HTD tree
*****************************************************************************/
RCODE fcsBuildLockUser(
	F_LOCK_USER *	pLockUser,
	FLMBOOL			bList,
	F_Pool *			pPool,
	NODE **			ppTree)
{
	NODE *		pRootNd = NULL;
	NODE *		pContextNd = NULL;
	void *		pMark = pPool->poolMark();
	RCODE			rc = FERR_OK;

	*ppTree = NULL;

	if( !pLockUser)
	{
		goto Exit;
	}

	// Add fields to the tree.

	for( ;;)
	{
		if( (pContextNd = GedNodeMake( pPool, FCS_LUSR_CONTEXT, &rc)) == NULL)
		{
			goto Exit;
		}

		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_LUSR_THREAD_ID, (void *)&pLockUser->uiThreadId,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_LUSR_TIME, (void *)&pLockUser->uiTime,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}

		if( pRootNd == NULL)
		{
			pRootNd = pContextNd;
		}
		else
		{
			GedSibGraft( pRootNd, pContextNd, GED_LAST);
		}

		if( !bList)
		{
			break;
		}

		pLockUser++;
		if( !pLockUser->uiTime)
		{
			break;
		}
	}

	*ppTree = pRootNd;

Exit:

	if( RC_BAD( rc))
	{
		pPool->poolReset( pMark);
	}

	return( rc);
}

/****************************************************************************
Desc:	Converts an HTD tree to a F_LOCK_USER structure (or list of structures)
*****************************************************************************/
RCODE fcsExtractLockUser(
	NODE *			pTree,
	FLMBOOL			bExtractAsList,
	void *			pvLockUser)
{
	NODE *			pTmpNd;
	FLMUINT			uiItemCount = 0;
	FLMUINT			fieldPath[ 8];
	F_LOCK_USER *	pLockUser = NULL;
	FLMUINT			uiLoop;
	RCODE				rc = FERR_OK;

	if( !pTree)
	{
		if( bExtractAsList)
		{
			*((F_LOCK_USER **)pvLockUser) = NULL;
		}
		else
		{
			f_memset( (F_LOCK_USER *)pvLockUser, 0, sizeof( F_LOCK_USER));
		}
		goto Exit;
	}

	if( bExtractAsList)
	{
		pTmpNd = pTree;
		while( pTmpNd != NULL)
		{
			if( GedTagNum( pTmpNd) == FCS_LUSR_CONTEXT)
			{
				uiItemCount++;
			}
			pTmpNd = pTmpNd->next;
		}

		if( RC_BAD( rc = f_alloc( 
			sizeof( F_LOCK_USER) * (uiItemCount + 1), &pLockUser)))
		{
			goto Exit;
		}

		*((F_LOCK_USER **)pvLockUser) = pLockUser;
	}
	else
	{
		pLockUser = (F_LOCK_USER *)pvLockUser;
		f_memset( pLockUser, 0, sizeof( F_LOCK_USER));
		uiItemCount = 1;
	}
	
	// Parse the tree and extract the values.

	for( uiLoop = 0; uiLoop < uiItemCount; uiLoop++)
	{
		fieldPath[ 0] = FCS_LUSR_CONTEXT;
		fieldPath[ 1] = FCS_LUSR_THREAD_ID;
		fieldPath[ 2] = 0;

		if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
		{
			(void) GedGetUINT( pTmpNd, &pLockUser[ uiLoop].uiThreadId);
		}

		fieldPath[ 0] = FCS_LUSR_CONTEXT;
		fieldPath[ 1] = FCS_LUSR_TIME;
		fieldPath[ 2] = 0;

		if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
		{
			(void) GedGetUINT( pTmpNd, &pLockUser[ uiLoop].uiTime);
		}

		pTree = GedSibNext( pTree);
	}

	if( bExtractAsList)
	{
		f_memset( &(pLockUser[ uiItemCount]), 0, sizeof( F_LOCK_USER));
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:	Extracts a CHECKPOINT_INFO structure from an HTD tree.
*****************************************************************************/
RCODE fcsExtractCheckpointInfo(
	NODE *					pTree,
	CHECKPOINT_INFO *		pChkptInfo)
{
	NODE *		pTmpNd;
	FLMUINT		fieldPath[ 8];
	FLMUINT		uiTmp;
	RCODE			rc = FERR_OK;

	// Initialize the structure

	f_memset( pChkptInfo, 0, sizeof( CHECKPOINT_INFO));

	// Parse the tree and extract the values.

	fieldPath[ 0] = FCS_CPI_CONTEXT;
	fieldPath[ 1] = FCS_CPI_RUNNING;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &uiTmp);
		pChkptInfo->bRunning = uiTmp ? TRUE : FALSE;
	}

	fieldPath[ 0] = FCS_CPI_CONTEXT;
	fieldPath[ 1] = FCS_CPI_START_TIME;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pChkptInfo->uiRunningTime);
	}

	fieldPath[ 0] = FCS_CPI_CONTEXT;
	fieldPath[ 1] = FCS_CPI_FORCING_CP;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &uiTmp);
		pChkptInfo->bForcingCheckpoint = uiTmp ? TRUE : FALSE;
	}

	fieldPath[ 0] = FCS_CPI_CONTEXT;
	fieldPath[ 1] = FCS_CPI_FORCE_CP_START_TIME;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pChkptInfo->uiForceCheckpointRunningTime);
	}

	fieldPath[ 0] = FCS_CPI_CONTEXT;
	fieldPath[ 1] = FCS_CPI_FORCE_CP_REASON;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetINT( pTmpNd, &pChkptInfo->iForceCheckpointReason);
	}

	fieldPath[ 0] = FCS_CPI_CONTEXT;
	fieldPath[ 1] = FCS_CPI_WRITING_DATA_BLOCKS;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &uiTmp);
		pChkptInfo->bWritingDataBlocks = uiTmp ? TRUE : FALSE;
	}

	fieldPath[ 0] = FCS_CPI_CONTEXT;
	fieldPath[ 1] = FCS_CPI_LOG_BLOCKS_WRITTEN;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pChkptInfo->uiLogBlocksWritten);
	}

	fieldPath[ 0] = FCS_CPI_CONTEXT;
	fieldPath[ 1] = FCS_CPI_DATA_BLOCKS_WRITTEN;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pChkptInfo->uiDataBlocksWritten);
	}

	fieldPath[ 0] = FCS_CPI_CONTEXT;
	fieldPath[ 1] = FCS_CPI_DIRTY_CACHE_BYTES;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pChkptInfo->uiDirtyCacheBytes);
	}

	fieldPath[ 0] = FCS_CPI_CONTEXT;
	fieldPath[ 1] = FCS_CPI_BLOCK_SIZE;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pChkptInfo->uiBlockSize);
	}

	fieldPath[ 0] = FCS_CPI_CONTEXT;
	fieldPath[ 1] = FCS_CPI_WAIT_TRUNC_TIME;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pChkptInfo->uiWaitTruncateTime);
	}

	return( rc);
}

/****************************************************************************
Desc:	Translates a FLAIM query operator to a c/s query operator
*****************************************************************************/
RCODE fcsTranslateQFlmToQCSOp(
	QTYPES			eFlmOp,
	FLMUINT *		puiCSOp)
{
	RCODE		rc = FERR_OK;

	switch( eFlmOp)
	{
		case FLM_AND_OP:
			*puiCSOp = FCS_ITERATOR_AND_OP;
			break;
		case FLM_OR_OP:
			*puiCSOp = FCS_ITERATOR_OR_OP;
			break;
		case FLM_NOT_OP:
			*puiCSOp = FCS_ITERATOR_NOT_OP;
			break;
		case FLM_EQ_OP:
			*puiCSOp = FCS_ITERATOR_EQ_OP;
			break;
		case FLM_MATCH_OP:
			*puiCSOp = FCS_ITERATOR_MATCH_OP;
			break;
		case FLM_MATCH_BEGIN_OP:
			*puiCSOp = FCS_ITERATOR_MATCH_BEGIN_OP;
			break;
		case FLM_CONTAINS_OP:
			*puiCSOp = FCS_ITERATOR_CONTAINS_OP;
			break;
		case FLM_NE_OP:
			*puiCSOp = FCS_ITERATOR_NE_OP;
			break;
		case FLM_LT_OP:
			*puiCSOp = FCS_ITERATOR_LT_OP;
			break;
		case FLM_LE_OP:
			*puiCSOp = FCS_ITERATOR_LE_OP;
			break;
		case FLM_GT_OP:
			*puiCSOp = FCS_ITERATOR_GT_OP;
			break;
		case FLM_GE_OP:
			*puiCSOp = FCS_ITERATOR_GE_OP;
			break;
		case FLM_BITAND_OP:
			*puiCSOp = FCS_ITERATOR_BITAND_OP;
			break;
		case FLM_BITOR_OP:
			*puiCSOp = FCS_ITERATOR_BITOR_OP;
			break;
		case FLM_BITXOR_OP:
			*puiCSOp = FCS_ITERATOR_BITXOR_OP;
			break;
		case FLM_MULT_OP:
			*puiCSOp = FCS_ITERATOR_MULT_OP;
			break;
		case FLM_DIV_OP:
			*puiCSOp = FCS_ITERATOR_DIV_OP;
			break;
		case FLM_MOD_OP:
			*puiCSOp = FCS_ITERATOR_MOD_OP;
			break;
		case FLM_PLUS_OP:
			*puiCSOp = FCS_ITERATOR_PLUS_OP;
			break;
		case FLM_MINUS_OP:
			*puiCSOp = FCS_ITERATOR_MINUS_OP;
			break;
		case FLM_NEG_OP:
			*puiCSOp = FCS_ITERATOR_NEG_OP;
			break;
		case FLM_LPAREN_OP:
			*puiCSOp = FCS_ITERATOR_LPAREN_OP;
			break;
		case FLM_RPAREN_OP:
			*puiCSOp = FCS_ITERATOR_RPAREN_OP;
			break;
		default:
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			break;
	}

	return( rc);
}

/****************************************************************************
Desc:	Translates a FLAIM query operator to a c/s query operator
*****************************************************************************/
RCODE fcsTranslateQCSToQFlmOp(
	FLMUINT 		uiCSOp,
	QTYPES *		peFlmOp)
{
	RCODE		rc = FERR_OK;

	switch( uiCSOp)
	{
		case FCS_ITERATOR_AND_OP:
			*peFlmOp = FLM_AND_OP;
			break;
		case FCS_ITERATOR_OR_OP:
			*peFlmOp = FLM_OR_OP;
			break;
		case FCS_ITERATOR_NOT_OP:
			*peFlmOp = FLM_NOT_OP;
			break;
		case FCS_ITERATOR_EQ_OP:
			*peFlmOp = FLM_EQ_OP;
			break;
		case FCS_ITERATOR_MATCH_OP:
			*peFlmOp = FLM_MATCH_OP;
			break;
		case FCS_ITERATOR_MATCH_BEGIN_OP:
			*peFlmOp = FLM_MATCH_BEGIN_OP;
			break;
		case FCS_ITERATOR_CONTAINS_OP:
			*peFlmOp = FLM_CONTAINS_OP;
			break;
		case FCS_ITERATOR_NE_OP:
			*peFlmOp = FLM_NE_OP;
			break;
		case FCS_ITERATOR_LT_OP:
			*peFlmOp = FLM_LT_OP;
			break;
		case FCS_ITERATOR_LE_OP:
			*peFlmOp = FLM_LE_OP;
			break;
		case FCS_ITERATOR_GT_OP:
			*peFlmOp = FLM_GT_OP;
			break;
		case FCS_ITERATOR_GE_OP:
			*peFlmOp = FLM_GE_OP;
			break;
		case FCS_ITERATOR_BITAND_OP:
			*peFlmOp = FLM_BITAND_OP;
			break;
		case FCS_ITERATOR_BITOR_OP:
			*peFlmOp = FLM_BITOR_OP;
			break;
		case FCS_ITERATOR_BITXOR_OP:
			*peFlmOp = FLM_BITXOR_OP;
			break;
		case FCS_ITERATOR_MULT_OP:
			*peFlmOp = FLM_MULT_OP;
			break;
		case FCS_ITERATOR_DIV_OP:
			*peFlmOp = FLM_DIV_OP;
			break;
		case FCS_ITERATOR_MOD_OP:
			*peFlmOp = FLM_MOD_OP;
			break;
		case FCS_ITERATOR_PLUS_OP:
			*peFlmOp = FLM_PLUS_OP;
			break;
		case FCS_ITERATOR_MINUS_OP:
			*peFlmOp = FLM_MINUS_OP;
			break;
		case FCS_ITERATOR_NEG_OP:
			*peFlmOp = FLM_NEG_OP;
			break;
		case FCS_ITERATOR_LPAREN_OP:
			*peFlmOp = FLM_LPAREN_OP;
			break;
		case FCS_ITERATOR_RPAREN_OP:
			*peFlmOp = FLM_RPAREN_OP;
			break;
		default:
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			break;
	}

	return( rc);
}

/****************************************************************************
Desc:	Converts an FINDEX_STATUS structure to an HTD tree
*****************************************************************************/
RCODE fcsBuildIndexStatus(
	FINDEX_STATUS *	pIndexStatus,
	F_Pool *				pPool,
	NODE **				ppTree)
{
	NODE *		pContextNd = NULL;
	void *		pMark = pPool->poolMark();
	FLMUINT		uiTmp;
	RCODE			rc = FERR_OK;

	*ppTree = NULL;

	if( !pIndexStatus)
	{
		goto Exit;
	}

	// Add fields to the tree.

	if( (pContextNd = GedNodeMake( pPool, FCS_IXSTAT_CONTEXT, &rc)) == NULL)
	{
		goto Exit;
	}

	if( pIndexStatus->uiIndexNum)
	{
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_IXSTAT_INDEX_NUM, (void *)&pIndexStatus->uiIndexNum,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pIndexStatus->uiStartTime)
	{
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_IXSTAT_START_TIME, (void *)&pIndexStatus->uiStartTime,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}

		// Send the "auto-online" flag for backwards compatibility

		uiTmp = 1;
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_IXSTAT_AUTO_ONLINE, 
			(void *)&uiTmp, 0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}

		// Send the priority (high) for backwards compatibility

		uiTmp = 1;
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_IXSTAT_PRIORITY,
			(void *)&uiTmp, 0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	// Set the suspended time field (for backwards compatibility)
	// if the index is suspended

	if( pIndexStatus->bSuspended)
	{
		f_timeGetSeconds( &uiTmp);
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_IXSTAT_SUSPEND_TIME, (void *)&uiTmp,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pIndexStatus->uiLastRecordIdIndexed)
	{
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_IXSTAT_LAST_REC_INDEXED, 
			(void *)&pIndexStatus->uiLastRecordIdIndexed,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pIndexStatus->uiKeysProcessed)
	{
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_IXSTAT_KEYS_PROCESSED, 
			(void *)&pIndexStatus->uiKeysProcessed,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pIndexStatus->uiRecordsProcessed)
	{
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_IXSTAT_RECS_PROCESSED, 
			(void *)&pIndexStatus->uiRecordsProcessed,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pIndexStatus->bSuspended)
	{
		uiTmp = (FLMUINT)pIndexStatus->bSuspended;
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_IXSTAT_STATE, (void *)&uiTmp,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	*ppTree = pContextNd;

Exit:

	if( RC_BAD( rc))
	{
		pPool->poolReset( pMark);
	}

	return( rc);
}

/****************************************************************************
Desc:	Extracts an FINDEX_STATUS structure from an HTD tree.
*****************************************************************************/
RCODE fcsExtractIndexStatus(
	NODE *					pTree,
	FINDEX_STATUS *		pIndexStatus)
{
	NODE *		pTmpNd;
	FLMUINT		fieldPath[ 8];
	RCODE			rc = FERR_OK;

	// Initialize the structure

	f_memset( pIndexStatus, 0, sizeof( FINDEX_STATUS));

	// Make sure pTree is non-null

	if( !pTree)
	{
		goto Exit;
	}

	// Parse the tree and extract the values.

	fieldPath[ 0] = FCS_IXSTAT_CONTEXT;
	fieldPath[ 1] = FCS_IXSTAT_INDEX_NUM;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pIndexStatus->uiIndexNum);
	}

	fieldPath[ 1] = FCS_IXSTAT_START_TIME;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pIndexStatus->uiStartTime);
	}

	fieldPath[ 1] = FCS_IXSTAT_LAST_REC_INDEXED;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pIndexStatus->uiLastRecordIdIndexed);
	}

	fieldPath[ 1] = FCS_IXSTAT_KEYS_PROCESSED;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pIndexStatus->uiKeysProcessed);
	}

	fieldPath[ 1] = FCS_IXSTAT_RECS_PROCESSED;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pIndexStatus->uiRecordsProcessed);
	}

	fieldPath[ 1] = FCS_IXSTAT_STATE;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		FLMUINT		uiTmp;
		(void)GedGetUINT( pTmpNd, &uiTmp);
		pIndexStatus->bSuspended = uiTmp ? TRUE : FALSE;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Converts an FLM_MEM_INFO structure to an HTD tree
*****************************************************************************/
RCODE fcsBuildMemInfo(
	FLM_MEM_INFO *		pMemInfo,
	F_Pool *				pPool,
	NODE **				ppTree)
{
	FLMUINT				uiTmp;
	NODE *				pContextNd = NULL;
	NODE *				pSubContext = NULL;
	void *				pMark = pPool->poolMark();
	FLM_CACHE_USAGE *	pUsage;
	RCODE					rc = FERR_OK;

	*ppTree = NULL;

	if( !pMemInfo)
	{
		goto Exit;
	}

	// Add fields to the tree.

	if( (pContextNd = GedNodeMake( pPool, 
		FCS_MEMINFO_CONTEXT, &rc)) == NULL)
	{
		goto Exit;
	}

	if( pMemInfo->bDynamicCacheAdjust)
	{
		uiTmp = 1;
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_MEMINFO_DYNA_CACHE_ADJ, (void *)&uiTmp,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pMemInfo->uiCacheAdjustPercent)
	{
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_MEMINFO_CACHE_ADJ_PERCENT, 
			(void *)&pMemInfo->uiCacheAdjustPercent,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pMemInfo->uiCacheAdjustMin)
	{
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_MEMINFO_CACHE_ADJ_MIN,
			(void *)&pMemInfo->uiCacheAdjustMin,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pMemInfo->uiCacheAdjustMax)
	{
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_MEMINFO_CACHE_ADJ_MAX,
			(void *)&pMemInfo->uiCacheAdjustMax,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pMemInfo->uiCacheAdjustMinToLeave)
	{
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_MEMINFO_CACHE_ADJ_MIN_LEAVE,
			(void *)&pMemInfo->uiCacheAdjustMinToLeave,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	pUsage = &pMemInfo->RecordCache;
	if( (pSubContext = GedNodeMake( pPool, 
		FCS_MEMINFO_RECORD_CACHE, &rc)) == NULL)
	{
		goto Exit;
	}

add_usage:

	if( pUsage->uiMaxBytes)
	{
		if( RC_BAD( rc = gedAddField( pPool, pSubContext,
			FCS_MEMINFO_MAX_BYTES,
			(void *)&pUsage->uiMaxBytes,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pUsage->uiCount)
	{
		if( RC_BAD( rc = gedAddField( pPool, pSubContext,
			FCS_MEMINFO_COUNT,
			(void *)&pUsage->uiCount,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pUsage->uiOldVerCount)
	{
		if( RC_BAD( rc = gedAddField( pPool, pSubContext,
			FCS_MEMINFO_OLD_VER_COUNT,
			(void *)&pUsage->uiOldVerCount,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pUsage->uiTotalBytesAllocated)
	{
		if( RC_BAD( rc = gedAddField( pPool, pSubContext,
			FCS_MEMINFO_TOTAL_BYTES_ALLOC,
			(void *)&pUsage->uiTotalBytesAllocated,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pUsage->uiOldVerBytes)
	{
		if( RC_BAD( rc = gedAddField( pPool, pSubContext,
			FCS_MEMINFO_OLD_VER_BYTES,
			(void *)&pUsage->uiOldVerBytes,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pUsage->uiCacheHits)
	{
		if( RC_BAD( rc = gedAddField( pPool, pSubContext,
			FCS_MEMINFO_CACHE_HITS,
			(void *)&pUsage->uiCacheHits,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pUsage->uiCacheHitLooks)
	{
		if( RC_BAD( rc = gedAddField( pPool, pSubContext,
			FCS_MEMINFO_CACHE_HIT_LOOKS,
			(void *)&pUsage->uiCacheHitLooks,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pUsage->uiCacheFaults)
	{
		if( RC_BAD( rc = gedAddField( pPool, pSubContext,
			FCS_MEMINFO_CACHE_FAULTS,
			(void *)&pUsage->uiCacheFaults,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pUsage->uiCacheFaultLooks)
	{
		if( RC_BAD( rc = gedAddField( pPool, pSubContext,
			FCS_MEMINFO_CACHE_FAULT_LOOKS,
			(void *)&pUsage->uiCacheFaultLooks,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( GedChild( pSubContext))
	{
		GedChildGraft( pContextNd, pSubContext, GED_LAST);
	}

	if( pUsage != &pMemInfo->BlockCache)
	{
		pUsage = &pMemInfo->BlockCache;
		if( (pSubContext = GedNodeMake( pPool, 
			FCS_MEMINFO_BLOCK_CACHE, &rc)) == NULL)
		{
			goto Exit;
		}
		goto add_usage;
	}

	*ppTree = pContextNd;

Exit:

	if( RC_BAD( rc))
	{
		pPool->poolReset( pMark);
	}

	return( rc);
}

/****************************************************************************
Desc:	Extracts a FLM_MEM_INFO structure from an HTD tree.
*****************************************************************************/
RCODE fcsExtractMemInfo(
	NODE *				pTree,
	FLM_MEM_INFO *		pMemInfo)
{
	NODE *					pTmpNd;
	FLMUINT					fieldPath[ 8];
	FLMUINT					uiTmp;
	FLM_CACHE_USAGE *		pUsage;
	FLMUINT					uiUsageTag;
	RCODE						rc = FERR_OK;

	// Initialize the structure

	f_memset( pMemInfo, 0, sizeof( FLM_MEM_INFO));

	// Make sure pTree is non-null

	if( !pTree)
	{
		goto Exit;
	}

	// Parse the tree and extract the values.

	fieldPath[ 0] = FCS_MEMINFO_CONTEXT;
	fieldPath[ 1] = FCS_MEMINFO_DYNA_CACHE_ADJ;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &uiTmp);
		pMemInfo->bDynamicCacheAdjust = (FLMBOOL)(uiTmp ? TRUE : FALSE);
	}

	fieldPath[ 1] = FCS_MEMINFO_CACHE_ADJ_PERCENT;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pMemInfo->uiCacheAdjustPercent);
	}

	fieldPath[ 1] = FCS_MEMINFO_CACHE_ADJ_MIN;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pMemInfo->uiCacheAdjustMin);
	}

	fieldPath[ 1] = FCS_MEMINFO_CACHE_ADJ_MAX;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pMemInfo->uiCacheAdjustMax);
	}

	fieldPath[ 1] = FCS_MEMINFO_CACHE_ADJ_MIN_LEAVE;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pMemInfo->uiCacheAdjustMinToLeave);
	}

	pUsage = &pMemInfo->RecordCache;
	uiUsageTag = FCS_MEMINFO_RECORD_CACHE;

get_usage:

	fieldPath[ 0] = FCS_MEMINFO_CONTEXT;
	fieldPath[ 1] = uiUsageTag;
	fieldPath[ 2] = FCS_MEMINFO_MAX_BYTES;
	fieldPath[ 3] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pUsage->uiMaxBytes);
	}

	fieldPath[ 2] = FCS_MEMINFO_COUNT;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pUsage->uiCount);
	}

	fieldPath[ 2] = FCS_MEMINFO_OLD_VER_COUNT;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pUsage->uiOldVerCount);
	}

	fieldPath[ 2] = FCS_MEMINFO_TOTAL_BYTES_ALLOC;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pUsage->uiTotalBytesAllocated);
	}

	fieldPath[ 2] = FCS_MEMINFO_OLD_VER_BYTES;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pUsage->uiOldVerBytes);
	}

	fieldPath[ 2] = FCS_MEMINFO_CACHE_HITS;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pUsage->uiCacheHits);
	}

	fieldPath[ 2] = FCS_MEMINFO_CACHE_HIT_LOOKS;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pUsage->uiCacheHitLooks);
	}

	fieldPath[ 2] = FCS_MEMINFO_CACHE_FAULTS;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pUsage->uiCacheFaults);
	}

	fieldPath[ 2] = FCS_MEMINFO_CACHE_FAULT_LOOKS;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pUsage->uiCacheFaultLooks);
	}

	if( pUsage != &pMemInfo->BlockCache)
	{
		pUsage = &pMemInfo->BlockCache;
		uiUsageTag = FCS_MEMINFO_BLOCK_CACHE;
		goto get_usage;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Builds a GEDCOM tree containing information on all FLAIM threads
*****************************************************************************/
RCODE fcsBuildThreadInfo(
	F_Pool *				pPool,
	NODE **				ppTree)
{
	NODE *				pContextNd = NULL;
	NODE *				pRootNd = NULL;
	void *				pMark = pPool->poolMark();
	F_THREAD_INFO *	pThreadInfo = NULL;
	FLMUINT				uiNumThreads;
	FLMUINT				uiLoop;
	RCODE					rc = FERR_OK;

	*ppTree = NULL;

	// Query FLAIM for available threads

	if( RC_BAD( rc = FlmGetThreadInfo( pPool, &pThreadInfo, &uiNumThreads)))
	{
		goto Exit;
	}

	if( (pRootNd = GedNodeMake( pPool, 
		FCS_THREAD_INFO_ROOT, &rc)) == NULL)
	{
		goto Exit;
	}

	if( RC_BAD( rc = GedPutRecPtr( pPool, pRootNd, uiNumThreads)))
	{
		goto Exit;
	}

	for( uiLoop = 0; uiLoop < uiNumThreads; uiLoop++)
	{
		// Add fields to the tree.

		if( (pContextNd = GedNodeMake( pPool, 
			FCS_THREAD_INFO_CONTEXT, &rc)) == NULL)
		{
			goto Exit;
		}

		GedChildGraft( pRootNd, pContextNd, GED_LAST);

		if( pThreadInfo->uiThreadId)
		{
			if( RC_BAD( rc = gedAddField( pPool, pContextNd,
				FCS_THREADINFO_THREAD_ID, (void *)&pThreadInfo->uiThreadId,
				0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}
		}

		if( pThreadInfo->uiThreadGroup)
		{
			if( RC_BAD( rc = gedAddField( pPool, pContextNd,
				FCS_THREADINFO_THREAD_GROUP, (void *)&pThreadInfo->uiThreadGroup,
				0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}
		}

		if( pThreadInfo->uiAppId)
		{
			if( RC_BAD( rc = gedAddField( pPool, pContextNd,
				FCS_THREADINFO_APP_ID, (void *)&pThreadInfo->uiAppId,
				0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}
		}

		if( pThreadInfo->uiStartTime)
		{
			if( RC_BAD( rc = gedAddField( pPool, pContextNd,
				FCS_THREADINFO_START_TIME, (void *)&pThreadInfo->uiStartTime,
				0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}
		}

		if( pThreadInfo->pszThreadName)
		{
			if( RC_BAD( rc = gedAddField( pPool, pContextNd,
				FCS_THREADINFO_THREAD_NAME, (void *)pThreadInfo->pszThreadName,
				0, FLM_TEXT_TYPE)))
			{
				goto Exit;
			}
		}

		if( pThreadInfo->pszThreadStatus)
		{
			if( RC_BAD( rc = gedAddField( pPool, pContextNd,
				FCS_THREADINFO_THREAD_STATUS, (void *)pThreadInfo->pszThreadStatus,
				0, FLM_TEXT_TYPE)))
			{
				goto Exit;
			}
		}

		pThreadInfo++;
	}

	*ppTree = pRootNd;

Exit:

	if( RC_BAD( rc))
	{
		pPool->poolReset( pMark);
	}

	return( rc);
}

/****************************************************************************
Desc:	Extracts a list of F_THREAD_INFO structure from an HTD tree.
*****************************************************************************/
RCODE fcsExtractThreadInfo(
	NODE *				pTree,
	F_Pool *				pPool,
	F_THREAD_INFO **	ppThreadInfo,
	FLMUINT *			puiNumThreads)
{
	NODE *				pTmpNd;
	NODE *				pContextNd;
	void *				pMark = pPool->poolMark();
	FLMUINT				uiTmp;
	F_THREAD_INFO *	pThreadInfo;
	F_THREAD_INFO *	pCurThread;
	FLMUINT				uiNumThreads;
	FLMUINT				uiLoop;
	RCODE					rc = FERR_OK;

	*ppThreadInfo = NULL;
	*puiNumThreads = 0;

	if( GedTagNum( pTree) != FCS_THREAD_INFO_ROOT)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	if( RC_BAD( rc = GedGetUINT( pTree, &uiNumThreads)))
	{
		goto Exit;
	}

	if( !uiNumThreads)
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pPool->poolAlloc( uiNumThreads * sizeof( F_THREAD_INFO),
		(void **)&pThreadInfo)))
	{
		goto Exit;
	}

	if( (pContextNd = GedFind( 1, pTree, 
		FCS_THREAD_INFO_CONTEXT, 1)) == NULL)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	for( uiLoop = 0, pCurThread = pThreadInfo; 
		uiLoop < uiNumThreads; 
		uiLoop++, pCurThread++)
	{

		if( (pTmpNd = GedFind( 1, pContextNd, 
			FCS_THREADINFO_THREAD_ID, 1)) != NULL)
		{
			(void) GedGetUINT( pTmpNd, &pCurThread->uiThreadId);
		}

		if( (pTmpNd = GedFind( 1, pContextNd, 
			FCS_THREADINFO_THREAD_GROUP, 1)) != NULL)
		{
			(void) GedGetUINT( pTmpNd, &pCurThread->uiThreadGroup);
		}

		if( (pTmpNd = GedFind( 1, pContextNd, 
			FCS_THREADINFO_APP_ID, 1)) != NULL)
		{
			(void) GedGetUINT( pTmpNd, &pCurThread->uiAppId);
		}

		if( (pTmpNd = GedFind( 1, pContextNd, 
			FCS_THREADINFO_START_TIME, 1)) != NULL)
		{
			(void) GedGetUINT( pTmpNd, &pCurThread->uiStartTime);
		}

		if( (pTmpNd = GedFind( 1, pContextNd, 
			FCS_THREADINFO_THREAD_NAME, 1)) != NULL)
		{
			if( RC_BAD( rc = GedGetNATIVE( pTmpNd, NULL, &uiTmp)))
			{
				goto Exit;
			}

			if( uiTmp)
			{
				uiTmp++;
				
				if( RC_BAD( rc = pPool->poolAlloc( uiTmp,
					(void **)&pCurThread->pszThreadName)))
				{
					goto Exit;
				}
			}

			if( RC_BAD( rc = GedGetNATIVE( pTmpNd, 
				pCurThread->pszThreadName, &uiTmp)))
			{
				goto Exit;
			}
		}

		if( (pTmpNd = GedFind( 1, pContextNd, 
			FCS_THREADINFO_THREAD_STATUS, 1)) != NULL)
		{
			if( RC_BAD( rc = GedGetNATIVE( pTmpNd, NULL, &uiTmp)))
			{
				goto Exit;
			}

			if( uiTmp)
			{
				uiTmp++;
				
				if( RC_BAD( rc = pPool->poolAlloc( uiTmp,
					(void **)pCurThread->pszThreadStatus)))
				{
					goto Exit;
				}
			}

			if( RC_BAD( rc = GedGetNATIVE( pTmpNd, 
				pCurThread->pszThreadStatus, &uiTmp)))
			{
				goto Exit;
			}
		}

		if( (pContextNd = GedSibNext( pContextNd)) != NULL)
		{
			if( GedTagNum( pContextNd) != FCS_THREAD_INFO_CONTEXT)
			{
				rc = RC_SET( FERR_SYNTAX);
				goto Exit;
			}
		}
	}

	*ppThreadInfo = pThreadInfo;
	*puiNumThreads = uiNumThreads;

Exit:

	if( RC_BAD( rc))
	{
		pPool->poolReset( pMark);
	}

	return( rc);
}

/****************************************************************************
Desc:	Reads a block from a remote database
*****************************************************************************/
RCODE fcsGetBlock(
	HFDB			hDb,
	FLMUINT		uiAddress,
	FLMUINT		uiMinTransId,
	FLMUINT *	puiCount,
	FLMUINT *	puiBlocksExamined,
	FLMUINT *	puiNextBlkAddr,
	FLMUINT		uiFlags,
	FLMBYTE *	pucBlock)
{
	FDB *				pDb = (FDB *)hDb;
	RCODE				rc = FERR_OK;

	flmAssert( IsInCSMode( hDb));

	fdbInitCS( pDb);
	CS_CONTEXT *		pCSContext = pDb->pCSContext;
	FCL_WIRE				Wire( pCSContext, pDb);

	if( !pCSContext->bConnectionGood)
	{
		rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
		goto Transmission_Error;
	}

	if( RC_BAD( rc = Wire.sendOp(
		FCS_OPCLASS_DATABASE, FCS_OP_DATABASE_GET_BLOCK)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_ADDRESS, uiAddress)))
	{
		goto Transmission_Error;
	}

	if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_TRANSACTION_ID,
		uiMinTransId)))
	{
		goto Transmission_Error;
	}

	if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_COUNT, *puiCount)))
	{
		goto Transmission_Error;
	}

	if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_FLAGS, uiFlags)))
	{
		goto Transmission_Error;
	}

	if( RC_BAD( rc = Wire.sendTerminate()))
	{
		goto Transmission_Error;
	}

	// Read the response

	if (RC_BAD( rc = Wire.read()))
	{
		goto Transmission_Error;
	}

	if( RC_BAD( rc = Wire.getRCode()))
	{
		if( rc != FERR_IO_END_OF_FILE)
		{
			goto Exit;
		}
	}

	*puiBlocksExamined = (FLMUINT)Wire.getNumber2();
	*puiCount = (FLMUINT)Wire.getCount();
	*puiNextBlkAddr = Wire.getAddress();
	if( *puiCount)
	{
		f_memcpy( pucBlock, Wire.getBlock(), Wire.getBlockSize());
	}

	goto Exit;

Transmission_Error:
	pCSContext->bConnectionGood = FALSE;
	goto Exit;

Exit:

	fdbExit( pDb);
	return( rc);
}

/****************************************************************************
Desc:	Instructs the server to generate a serial number
*****************************************************************************/
RCODE fcsCreateSerialNumber(
	void *			pvCSContext,
	FLMBYTE *		pucSerialNum)
{
	RCODE				rc = FERR_OK;
	CS_CONTEXT *	pCSContext = (CS_CONTEXT *)pvCSContext;
	FCL_WIRE			Wire( pCSContext);

	if( !pCSContext->bConnectionGood)
	{
		rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
		goto Transmission_Error;
	}

	if( RC_BAD( rc = Wire.sendOp(
		FCS_OPCLASS_MISC, FCS_OP_CREATE_SERIAL_NUM)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = Wire.sendTerminate()))
	{
		goto Transmission_Error;
	}

	// Read the response

	if (RC_BAD( rc = Wire.read()))
	{
		goto Transmission_Error;
	}

	if( RC_BAD( rc = Wire.getRCode()))
	{
		goto Exit;
	}

	if( !Wire.getSerialNum())
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	f_memcpy( pucSerialNum, Wire.getSerialNum(), F_SERIAL_NUM_SIZE);
	goto Exit;

Transmission_Error:
	pCSContext->bConnectionGood = FALSE;
	goto Exit;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Sets or clears the backup active flag for the database
Note:	This should only be called internally from the backup routines.
*****************************************************************************/
RCODE fcsSetBackupActiveFlag(
	HFDB			hDb,
	FLMBOOL		bBackupActive)
{
	FDB *				pDb = (FDB *)hDb;
	RCODE				rc = FERR_OK;

	flmAssert( IsInCSMode( hDb));

	fdbInitCS( pDb);
	CS_CONTEXT *		pCSContext = pDb->pCSContext;
	FCL_WIRE				Wire( pCSContext, pDb);

	if( !pCSContext->bConnectionGood)
	{
		rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
		goto Transmission_Error;
	}

	if( RC_BAD( rc = Wire.sendOp(
		FCS_OPCLASS_DATABASE, FCS_OP_DB_SET_BACKUP_FLAG)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_BOOLEAN, bBackupActive)))
	{
		goto Transmission_Error;
	}

	if( RC_BAD( rc = Wire.sendTerminate()))
	{
		goto Transmission_Error;
	}

	/* Read the response. */

	if (RC_BAD( rc = Wire.read()))
	{
		goto Transmission_Error;
	}

	if( RC_BAD( rc = Wire.getRCode()))
	{
		goto Exit;
	}

	goto Exit;

Transmission_Error:
	pCSContext->bConnectionGood = FALSE;
	goto Exit;

Exit:

	fdbExit( pDb);
	return( rc);
}

/****************************************************************************
Desc:	Commits an update transaction and updates the log header.
Note:	This should only be called internally from the backup routines.
*****************************************************************************/
RCODE fcsDbTransCommitEx(
	HFDB			hDb,
	FLMBOOL		bForceCheckpoint,
	FLMBYTE *	pucLogHdr)
{
	RCODE			rc = FERR_OK;
	FDB *			pDb = (FDB *)hDb;
	FLMBOOL		bInitializedFdb = FALSE;

	if( IsInCSMode( hDb))
	{
		fdbInitCS( pDb);
		bInitializedFdb = TRUE;
		FCL_WIRE Wire( pDb->pCSContext, pDb);

		if (!pDb->pCSContext->bConnectionGood)
		{
			rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
		}
		else
		{
			rc = Wire.doTransOp(
				FCS_OP_TRANSACTION_COMMIT_EX, 0, 0, 0,
				pucLogHdr, bForceCheckpoint);
		}
	}
	else
	{
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

Exit:

	if( bInitializedFdb)
	{
		fdbExit( pDb);
	}

	return( rc);
}

/****************************************************************************
Desc: Generates a hex-encoded, obfuscated string consisting of characters
		0-9, A-F from the passed-in data buffer.
*****************************************************************************/
RCODE flmGenerateHexPacket(
	FLMBYTE *		pucData,
	FLMUINT			uiDataSize,
	FLMBYTE **		ppucPacket)
{
	RCODE						rc = FERR_OK;
	FLMBYTE *				pucBinPacket = NULL;
	FLMBYTE *				pucHexPacket = NULL;
	FLMBYTE *				pucUsedMap = NULL;
	FLMUINT32				ui32Tmp;
	FLMUINT					uiLoop;
	FLMUINT					uiSlot = 0;
	FLMBYTE					ucTmp[ 32];
	FLMUINT					uiBinPacketSize;
	FLMBOOL					bTmp;
	IF_RandomGenerator *	pRandGen = NULL;

	// Determine the packet size.  Make the minimum packet size 128 bytes
	// to account for the 64-byte "header" and for the overhead of the
	// CRC bytes, etc.  Round the packet size up to the nearest 64-byte
	// boundary after adding on the data size.

	uiBinPacketSize = 128 + uiDataSize;
	if( (uiBinPacketSize % 64) != 0)
	{
		uiBinPacketSize += (64 - (uiBinPacketSize % 64));
	}

	// Allocate buffers for building the packet

	if( RC_BAD( rc = f_alloc( uiBinPacketSize, &pucBinPacket)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_calloc( uiBinPacketSize, &pucUsedMap)))
	{
		goto Exit;
	}

	// First 64-bytes of the packet are reserved as a header

	f_memset( pucUsedMap, 0xFF, 64);

	// Initialize the random number generator and seed with the current
	// time.
	
	if( RC_BAD( rc = FlmAllocRandomGenerator( &pRandGen)))
	{
		goto Exit;
	}

	// Fill the packet with random "noise"

	for( uiLoop = 0; uiLoop < uiBinPacketSize; uiLoop += 4)
	{
		ui32Tmp = pRandGen->getUINT32();
		UD2FBA( ui32Tmp, &pucBinPacket[ uiLoop]);
	}

	for( uiLoop = 0; uiLoop < 512; uiLoop++)
	{
		ui32Tmp = pRandGen->getUINT32();
		UD2FBA( ui32Tmp, &pucBinPacket[ 
			pRandGen->getUINT32( 1, (int)(uiBinPacketSize / 4)) - 1]);
	}

	// Determine a new random seed based on bytes in the
	// packet header

	if( (ui32Tmp = (FLMUINT32)FB2UD( &pucBinPacket[ 
		pRandGen->getUINT32( 1, 61) - 1])) == 0)
	{
		ui32Tmp = 1;
	}

	pRandGen->setSeed( ui32Tmp);

	// Use the CRC of the header and the also first four bytes
	// of the header as an 8-byte validation signature.  This will
	// be needed to decode the packet.

	// Initialize the CRC to 0xFFFFFFFF and then compute the 1's
	// complement of the returned CRC.  This implements the 
	// "standard" CRC used by PKZIP, etc.

	ui32Tmp = 0xFFFFFFFF;
	f_updateCRC( pucBinPacket, 64, &ui32Tmp);
	ui32Tmp = ~ui32Tmp;
	UD2FBA( ui32Tmp, &ucTmp[ 0]);
	f_memcpy( &ucTmp[ 4], pucBinPacket, 4);

	for( uiLoop = 0; uiLoop < 8; uiLoop++)
	{
		bTmp = flmGetNextHexPacketSlot( pucUsedMap, uiBinPacketSize,
			pRandGen, &uiSlot);

		flmAssert( bTmp);
		pucBinPacket[ uiSlot] = ucTmp[ uiLoop];
	}

	// Encode the data size

	UD2FBA( (FLMUINT32)uiDataSize, &ucTmp[ 0]);
	for( uiLoop = 0; uiLoop < 4; uiLoop++)
	{
		bTmp = flmGetNextHexPacketSlot( pucUsedMap, uiBinPacketSize,
			pRandGen, &uiSlot);

		flmAssert( bTmp);
		pucBinPacket[ uiSlot] = ucTmp[ uiLoop];
	}

	// Randomly dispurse the data throughout the buffer.  Obfuscate the
	// data using the first 64-bytes of the buffer.

	for( uiLoop = 0; uiLoop < uiDataSize; uiLoop++)
	{
		bTmp = flmGetNextHexPacketSlot( pucUsedMap, uiBinPacketSize,
			pRandGen, &uiSlot);

		flmAssert( bTmp);
		pucBinPacket[ uiSlot] = pucData[ uiLoop] ^ pucBinPacket[ uiLoop % 64];
	}

	// Calculate and encode the data CRC

	ui32Tmp = 0xFFFFFFFF;
	f_updateCRC( pucData, uiDataSize, &ui32Tmp);
	ui32Tmp = ~ui32Tmp;
	UD2FBA( ui32Tmp, &ucTmp[ 0]);

	for( uiLoop = 0; uiLoop < 4; uiLoop++)
	{
		bTmp = flmGetNextHexPacketSlot( pucUsedMap, uiBinPacketSize,
			pRandGen, &uiSlot);

		flmAssert( bTmp);
		pucBinPacket[ uiSlot] = ucTmp[ uiLoop];
	}

	// Hex encode the binary packet

	if( RC_BAD( rc = f_alloc(
		(uiBinPacketSize * 2) + 1, &pucHexPacket)))
	{
		goto Exit;
	}

	for( uiLoop = 0; uiLoop < uiBinPacketSize; uiLoop++)
	{
		FLMBYTE		ucLowNibble = pucBinPacket[ uiLoop] & 0x0F;
		FLMBYTE		ucHighNibble = (pucBinPacket[ uiLoop] & 0xF0) >> 4;

		pucHexPacket[ uiLoop << 1] = (ucHighNibble <= 9 
													? (ucHighNibble + '0') 
													: ((ucHighNibble - 0xA) + 'A'));

		pucHexPacket[ (uiLoop << 1) + 1] = (ucLowNibble <= 9 
													? (ucLowNibble + '0') 
													: ((ucLowNibble - 0xA) + 'A'));
	}

	pucHexPacket[ uiBinPacketSize * 2] = 0;
	*ppucPacket = pucHexPacket;
	pucHexPacket = NULL;

Exit:

	if( pucUsedMap)
	{
		f_free( &pucUsedMap);
	}

	if( pucBinPacket)
	{
		f_free( &pucBinPacket);
	}

	if( pucHexPacket)
	{
		f_free( &pucHexPacket);
	}
	
	if( pRandGen)
	{
		pRandGen->Release();
	}

	return( rc);
}

/****************************************************************************
Desc: Extracts a data buffer from the passed-in hex-encoded, obfuscated
		string.
*****************************************************************************/
RCODE flmExtractHexPacketData(
	FLMBYTE *		pucPacket,
	FLMBYTE **		ppucData,
	FLMUINT *		puiDataSize)
{
	RCODE						rc = FERR_OK;
	FLMBYTE *				pucUsedMap = NULL;
	FLMBYTE *				pucData = NULL;
	FLMBYTE *				pucBinPacket = NULL;
	FLMBYTE *				pucTmp;
	FLMUINT32				ui32Tmp;
	FLMUINT32				ui32FirstCRC;
	FLMUINT32				ui32Seed;
	FLMUINT					uiPacketSize;
	FLMUINT					uiLoop;
	FLMUINT					uiDataSize;
	FLMBYTE					ucTmp[ 32];
	FLMBYTE					ucVal = 0;
	FLMBOOL					bValid;
	IF_RandomGenerator *	pRandGen = NULL;

	if( RC_BAD( rc = FlmAllocRandomGenerator( &pRandGen)))
	{
		goto Exit;
	}
	
	// Determine the packet size, ignoring all characters except 0-9, A-F

	uiPacketSize = 0;
	pucTmp = pucPacket;
	while( *pucTmp)
	{
		if( (*pucTmp >= '0' && *pucTmp <= '9') ||
			(*pucTmp >= 'A' && *pucTmp <= 'F'))
		{
			uiPacketSize++;
		}
		pucTmp++;
	}

	if( uiPacketSize & 0x00000001 || 
		(uiPacketSize % 4) != 0 || uiPacketSize < 128)
	{
		rc = RC_SET( FERR_INVALID_CRC);
		goto Exit;
	}

	// Get the actual size of the decoded binary data by dividing
	// the packet size by 2

	uiPacketSize >>= 1;

	// Allocate a buffer and convert the data from hex ASCII to binary

	if( RC_BAD( rc = f_calloc( 
		uiPacketSize, &pucBinPacket)))
	{
		goto Exit;
	}

	uiLoop = 0;
	pucTmp = pucPacket;
	while( *pucTmp)
	{
		bValid = FALSE;
		if( *pucTmp >= '0' && *pucTmp <= '9')
		{
			ucVal = *pucTmp - '0';
			bValid = TRUE;
		}
		else if( *pucTmp >= 'A' && *pucTmp <= 'F')
		{
			ucVal = (*pucTmp - 'A') + 0x0A;
			bValid = TRUE;
		}

		if( bValid)
		{
			if( (uiLoop & 0x00000001) == 0)
			{
				ucVal <<= 4;
			}
			pucBinPacket[ uiLoop >> 1] |= ucVal;
			uiLoop++;
		}

		pucTmp++;
	}

	// Allocate the data map

	if( RC_BAD( rc = f_calloc( uiPacketSize, &pucUsedMap)))
	{
		goto Exit;
	}

	// First 64-bytes of the packet are reserved

	f_memset( pucUsedMap, 0xFF, 64);

	// Determine the CRC of the 1st 64-bytes

	ui32FirstCRC = 0xFFFFFFFF;
	f_updateCRC( pucBinPacket, 64, &ui32FirstCRC);
	ui32FirstCRC = ~ui32FirstCRC;

	// Search for the random seed within the first 64 bytes

	ui32Seed = 0;
	for( uiLoop = 0; uiLoop < 61; uiLoop++)
	{
		ui32Tmp = FB2UD( &pucBinPacket[ uiLoop]);
		pRandGen->setSeed( ui32Tmp);

		if( RC_BAD( rc = flmGetNextHexPacketBytes( pucUsedMap, uiPacketSize, 
			pucBinPacket, pRandGen, ucTmp, 8)))
		{
			goto Exit;
		}

		if( FB2UD( &ucTmp[ 0]) == ui32FirstCRC && 
			f_memcmp( &ucTmp[ 4], &pucBinPacket[ 0], 4) == 0)
		{
			ui32Seed = ui32Tmp;
			break;
		}

		// Reset the "used" map
		
		f_memset( pucUsedMap, 0, uiPacketSize);
		f_memset( pucUsedMap, 0xFF, 64);
	}

	if( !ui32Seed)
	{
		rc = RC_SET( FERR_INVALID_CRC);
		goto Exit;
	}

	// Get the data size

	if( RC_BAD( rc = flmGetNextHexPacketBytes( pucUsedMap, uiPacketSize, 
		pucBinPacket, pRandGen, ucTmp, 4)))
	{
		goto Exit;
	}

	uiDataSize = (FLMUINT)FB2UD( &ucTmp[ 0]);
	if( uiDataSize > uiPacketSize)
	{
		rc = RC_SET( FERR_INVALID_CRC);
		goto Exit;
	}

	// Allocate space for the data

	if( RC_BAD( rc = f_alloc( uiDataSize, &pucData)))
	{
		goto Exit;
	}

	// Get the data

	if( RC_BAD( rc = flmGetNextHexPacketBytes( 
		pucUsedMap, uiPacketSize, pucBinPacket, pRandGen, pucData, uiDataSize)))
	{
		goto Exit;
	}

	// Un-obfuscate the data

	for( uiLoop = 0; uiLoop < uiDataSize; uiLoop++)
	{
		pucData[ uiLoop] = pucData[ uiLoop] ^ pucBinPacket[ uiLoop % 64];
	}

	// Get the data CRC

	if( RC_BAD( rc = flmGetNextHexPacketBytes( 
		pucUsedMap, uiPacketSize, pucBinPacket, pRandGen, ucTmp, 4)))
	{
		goto Exit;
	}

	// Verify the data CRC

	ui32Tmp = 0xFFFFFFFF;
	f_updateCRC( pucData, uiDataSize, &ui32Tmp);
	ui32Tmp = ~ui32Tmp;

	if( ui32Tmp != FB2UD( &ucTmp[ 0]))
	{
		rc = RC_SET( FERR_INVALID_CRC);
		goto Exit;
	}

	*ppucData = pucData;
	pucData = NULL;
	*puiDataSize = uiDataSize;

Exit:

	if( pucUsedMap)
	{
		f_free( &pucUsedMap);
	}

	if( pucData)
	{
		f_free( &pucData);
	}

	if( pucBinPacket)
	{
		f_free( &pucBinPacket);
	}
	
	if( pRandGen)
	{
		pRandGen->Release();
	}

	return( rc);
}

/****************************************************************************
Desc: Used by flmGenerateHexPacket to find an unused byte in the packet
*****************************************************************************/
FLMBOOL flmGetNextHexPacketSlot( 
	FLMBYTE *				pucUsedMap,
	FLMUINT					uiMapSize,
	IF_RandomGenerator *	pRandGen,
	FLMUINT *				puiSlot)
{
	FLMUINT		uiLoop;
	FLMUINT		uiSlot = 0;
	FLMBOOL		bFound = FALSE;

	for( uiLoop = 0; uiLoop < 100; uiLoop++)
	{
		uiSlot = ((FLMUINT)pRandGen->getUINT32()) % uiMapSize;
		if( !pucUsedMap[ uiSlot])
		{
			bFound = TRUE;
			goto Exit;
		}
	}

	// Scan the table from the top to find an empty slot

	for( uiSlot = 0; uiSlot < uiMapSize; uiSlot++)
	{
		if( !pucUsedMap[ uiSlot])
		{
			bFound = TRUE;
			goto Exit;
		}
	}

Exit:

	if( bFound)
	{
		flmAssert( uiSlot < uiMapSize);
		*puiSlot = uiSlot;
		pucUsedMap[ uiSlot] = 0xFF;
	}

	return( bFound);
}

/****************************************************************************
Desc: Used by flmExtractHexPacket to get the next N bytes of data from the
		packet.
*****************************************************************************/
RCODE flmGetNextHexPacketBytes( 
	FLMBYTE *				pucUsedMap,
	FLMUINT					uiMapSize,
	FLMBYTE *				pucPacket,
	IF_RandomGenerator *	pRandGen,
	FLMBYTE *				pucBuf,
	FLMUINT					uiCount)
{
	FLMUINT		uiSlot;
	FLMUINT		uiLoop;
	RCODE			rc = FERR_OK;

	for( uiLoop = 0; uiLoop < uiCount; uiLoop++)
	{
		if( !flmGetNextHexPacketSlot( pucUsedMap, uiMapSize, pRandGen, &uiSlot))
		{
			rc = RC_SET( FERR_INVALID_CRC);
			goto Exit;
		}

		pucBuf[ uiLoop] = pucPacket[ uiSlot];
	}

Exit:

	return( rc);
}
	
/****************************************************************************
Desc: Decodes a string containing %XX sequences and does it in place.
		Typically, this data comes from an HTML form.
****************************************************************************/
void fcsDecodeHttpString(
	char *		pszSrc)
{
	char *		pszDest;

	pszDest = pszSrc;
	while( *pszSrc)
	{
		if( *pszSrc == '%')
		{
			pszSrc++;
			if( f_isHexChar( (FLMBYTE)pszSrc[ 0]) && 
				 f_isHexChar( (FLMBYTE)pszSrc[ 1]))
			{
				*pszDest = (f_getHexVal( (FLMBYTE)pszSrc[ 0]) << 4) |
					f_getHexVal( (FLMBYTE)pszSrc[ 1]);

				pszSrc += 2;
				pszDest++;
				continue;
			}
		}
		else if( *pszSrc == '+')
		{
			*pszDest = ' ';
			pszSrc++;
			pszDest++;
			continue;
		}

		if( pszSrc != pszDest)
		{
			*pszDest = *pszSrc;
		}
		pszSrc++;
		pszDest++;
	}

	*pszDest = 0;
}

/****************************************************************************
Desc:
*****************************************************************************/
FCS_WIRE::FCS_WIRE( FCS_DIS * pDIStream, FCS_DOS * pDOStream)
{
	m_pool.poolInit( 2048);
	m_pPool = &m_pool;
	m_pDIStream = pDIStream;
	m_pDOStream = pDOStream;
	m_pRecord = NULL;
	m_pFromKey = NULL;
	m_pUntilKey = NULL;
	m_bSendGedcom = FALSE;
	(void)resetCommon();
}

/****************************************************************************
Desc:
*****************************************************************************/
FCS_WIRE::~FCS_WIRE( void)
{
	if( m_pRecord)
	{
		m_pRecord->Release();
		m_pRecord = NULL;
	}

	if( m_pFromKey)
	{
		m_pFromKey->Release();
		m_pFromKey = NULL;
	}

	if( m_pUntilKey)
	{
		m_pUntilKey->Release();
		m_pUntilKey = NULL;
	}

	m_pool.poolFree();
}

/****************************************************************************
Desc:	Resets all member variables to their default / initial values.
*****************************************************************************/
void FCS_WIRE::resetCommon( void)
{
	if( m_pRecord)
	{
		m_pRecord->Release();
		m_pRecord = NULL;
	}

	if( m_pFromKey)
	{
		m_pFromKey->Release();
		m_pFromKey = NULL;
	}

	if( m_pUntilKey)
	{
		m_pUntilKey->Release();
		m_pUntilKey = NULL;
	}

	m_uiClass = 0;
	m_uiOp = 0;
	m_uiRCode = 0;
	m_uiDrn = 0;
	m_uiTransType = FLM_READ_TRANS;
	m_ui64Count = 0;
	m_uiItemId = 0;
	m_uiIndexId = 0;
	m_puzItemName = NULL;
	m_pHTD = NULL;
	m_uiSessionId = FCS_INVALID_ID;
	m_uiSessionCookie = 0;
	m_uiContainer = FLM_DATA_CONTAINER;
	m_uiTransId = FCS_INVALID_ID;
	m_uiIteratorId = FCS_INVALID_ID;
	m_puzFilePath = NULL;
	m_puzFilePath2 = NULL;
	m_puzFilePath3 = NULL;
	m_pucBlock = NULL;
	m_pucSerialNum = NULL;
	m_uiBlockSize = 0;
	m_bIncludesAsync = FALSE;
	fcsInitCreateOpts( &m_CreateOpts);
	m_pPool->poolReset();
	m_bFlag = FALSE;
	m_ui64Number1 = 0;
	m_ui64Number2 = 0;
	m_ui64Number3 = 0;
	m_uiAddress = 0;
	m_uiFlags = 0;
	m_uiFlaimVersion = 0;
	m_i64SignedValue = 0;
	m_pCSContext = NULL;
	m_pDb = NULL;
}

/****************************************************************************
Desc:	Reads the class and opcode for a client request or server response.
*****************************************************************************/
RCODE FCS_WIRE::readOpcode( void)
{
	FLMBYTE	ucClass;
	FLMBYTE	ucOp;
	RCODE		rc = FERR_OK;

	if( RC_BAD( rc = m_pDIStream->read( &ucClass, 1, NULL)))
	{
		goto Exit;
	}
	m_uiClass = ucClass;

	if( RC_BAD( rc = m_pDIStream->read( &ucOp, 1, NULL)))
	{
		goto Exit;
	}
	m_uiOp = ucOp;

Exit:

	return( rc);
}

	
/****************************************************************************
Desc:	Reads a client request or server response and sets the appropriate
		member variable values.
*****************************************************************************/
RCODE FCS_WIRE::readCommon(
	FLMUINT *	puiTagRV,
	FLMBOOL *	pbEndRV)
{
	FLMUINT16	ui16Tmp;
	FLMUINT		uiTag = 0;
	RCODE			rc = FERR_OK;

	*pbEndRV = FALSE;

	// Read the tag.
	
	if( RC_BAD( rc = m_pDIStream->readUShort( &ui16Tmp)))
	{
		goto Exit;
	}
	uiTag = ui16Tmp;

	// Read the request / response values.
	
	switch( (uiTag & WIRE_VALUE_TAG_MASK))
	{
		case WIRE_VALUE_RCODE:
		{
			rc = readNumber( uiTag, &m_uiRCode);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_SESSION_ID:
		{
			rc = readNumber( uiTag, &m_uiSessionId);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_SESSION_COOKIE:
		{
			rc = readNumber( uiTag, &m_uiSessionCookie);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_CONTAINER_ID:
		{
			rc = readNumber( uiTag, &m_uiContainer);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_COUNT:
		{
			rc = readNumber( uiTag, NULL, NULL, &m_ui64Count);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_DRN:
		{
			rc = readNumber( uiTag, &m_uiDrn);
			uiTag = 0;
			break;
		}
		
		case WIRE_VALUE_INDEX_ID:
		{
			rc = readNumber( uiTag,	&m_uiIndexId);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_HTD:
		{
			rc = m_pDIStream->readHTD( m_pPool, 0, 0, &m_pHTD, NULL);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_RECORD:
		{
			FlmRecord *		pRecord = m_pRecord;
			if( RC_OK( rc = receiveRecord( &pRecord)))
			{
				if( m_pRecord != pRecord)
				{
					if( m_pRecord)
					{
						m_pRecord->Release();
					}
					m_pRecord = pRecord;
				}
			}

			uiTag = 0;
			break;
		}

		case WIRE_VALUE_FROM_KEY:
		{
			FlmRecord *		pFromKey = m_pFromKey;
			if( RC_OK( rc = receiveRecord( &pFromKey)))
			{
				if( m_pFromKey != pFromKey)
				{
					if( m_pFromKey)
					{
						m_pFromKey->Release();
					}
					m_pFromKey = pFromKey;
				}
			}

			uiTag = 0;
			break;
		}

		case WIRE_VALUE_UNTIL_KEY:
		{
			FlmRecord *		pUntilKey = m_pUntilKey;
			if( RC_OK( rc = receiveRecord( &pUntilKey)))
			{
				if( m_pUntilKey != pUntilKey)
				{
					if( m_pUntilKey)
					{
						m_pUntilKey->Release();
					}
					m_pUntilKey = pUntilKey;
				}
			}

			uiTag = 0;
			break;
		}

		case WIRE_VALUE_CREATE_OPTS:
		{
			rc = receiveCreateOpts();
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_ITERATOR_ID:
		{
			rc = readNumber( uiTag, &m_uiIteratorId);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_TRANSACTION_TYPE:
		{
			rc = readNumber( uiTag, &m_uiTransType);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_TRANSACTION_ID:
		{
			rc = readNumber( uiTag, &m_uiTransId);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_ITEM_NAME:
		{
			rc = m_pDIStream->readUTF( m_pPool, &m_puzItemName);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_ITEM_ID:
		{
			rc = readNumber( uiTag, &m_uiItemId);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_BOOLEAN:
		{
			FLMUINT		uiTmp;

			if( RC_OK( rc = readNumber( uiTag, &uiTmp)))
			{
				m_bFlag = (FLMBOOL)((uiTmp) ? (FLMBOOL)TRUE : (FLMBOOL)FALSE);
			}
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_NUMBER1:
		{
			rc = readNumber( uiTag, NULL, NULL, &m_ui64Number1);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_NUMBER2:
		{
			rc = readNumber( uiTag, NULL, NULL, &m_ui64Number2);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_NUMBER3:
		{
			rc = readNumber( uiTag, NULL, NULL, &m_ui64Number3);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_ADDRESS:
		{
			rc = readNumber( uiTag, &m_uiAddress);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_SIGNED_NUMBER:
		{
			rc = readNumber( uiTag, NULL, NULL, NULL, &m_i64SignedValue);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_FILE_PATH:
		{
			rc = m_pDIStream->readUTF( m_pPool, &m_puzFilePath);
			uiTag = 0;
			break;
		}
				
		case WIRE_VALUE_FILE_PATH_2:
		{
			rc = m_pDIStream->readUTF( m_pPool, &m_puzFilePath2);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_FILE_PATH_3:
		{
			rc = m_pDIStream->readUTF( m_pPool, &m_puzFilePath3);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_BLOCK:
		{
			rc = m_pDIStream->readLargeBinary( m_pPool, 
				&m_pucBlock, &m_uiBlockSize);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_SERIAL_NUM:
		{
			FLMUINT	uiSerialLen;

			if( RC_BAD( rc = m_pDIStream->readBinary( m_pPool,
				&m_pucSerialNum, &uiSerialLen)))
			{
				goto Exit;
			}
			
			if( uiSerialLen != F_SERIAL_NUM_SIZE)
			{
				rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
				goto Exit;
			}

			uiTag = 0;
			break;
		}

		case WIRE_VALUE_START_ASYNC:
		{
			m_bIncludesAsync = TRUE;
			*pbEndRV = TRUE;
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_FLAGS:
		{
			rc = readNumber( uiTag, &m_uiFlags);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_FLAIM_VERSION:
		{
			rc = readNumber( uiTag, &m_uiFlaimVersion);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_TERMINATE:
		{
			rc = m_pDIStream->endMessage();
			*pbEndRV = TRUE;
			uiTag = 0;
			break;
		}

		default:
		{
			break;
		}
	}

Exit:

	*puiTagRV = uiTag;
	return( rc);
}

/****************************************************************************
Desc:	Copies the internal CREATE_OPTS structure into a user-supplied location
*****************************************************************************/
void FCS_WIRE::copyCreateOpts(
	CREATE_OPTS *		pCreateOptsRV)
{
	f_memcpy( pCreateOptsRV, &m_CreateOpts, sizeof( CREATE_OPTS));
}

/****************************************************************************
Desc:	Reads a numeric value from the specified data input stream.
*****************************************************************************/
RCODE FCS_WIRE::readNumber(
	FLMUINT			uiTag,
	FLMUINT *		puiNumber,
	FLMINT *			piNumber,
	FLMUINT64 *		pui64Number,
	FLMINT64 *		pi64Number)
{
	RCODE				rc = FERR_OK;

	flmAssert( !(puiNumber && piNumber));
	
	// Read the number of bytes specified by the
	// value's tag.

	switch( ((uiTag & WIRE_VALUE_TYPE_MASK) >> 
		WIRE_VALUE_TYPE_START_BIT))
	{
		case WIRE_VALUE_TYPE_GEN_0:
		{
			if( puiNumber)
			{
				*puiNumber = 0;
			}
			else if( piNumber)
			{
				*piNumber = 0;
			}
			else if( pui64Number)
			{
				*pui64Number = 0;
			}
			else if( pi64Number)
			{
				*pi64Number = 0;
			}
			break;
		}

		case WIRE_VALUE_TYPE_GEN_1:
		{
			FLMBYTE	ucValue;

			if( RC_BAD( rc = m_pDIStream->read( &ucValue, 1, NULL)))
			{
				goto Exit;
			}

			if( puiNumber)
			{
				*puiNumber = (FLMUINT)ucValue;
			}
			else if( piNumber)
			{
				*piNumber = (FLMINT)*((FLMINT8 *)&ucValue);
			}
			else if( pui64Number)
			{
				*pui64Number = (FLMUINT64)ucValue;
			}
			else if( pi64Number)
			{
				*pi64Number = (FLMINT64)*((FLMINT8 *)&ucValue);
			}
			break;
		}

		case WIRE_VALUE_TYPE_GEN_2:
		{
			if( puiNumber || pui64Number)
			{
				FLMUINT16	ui16Value;

				if( RC_BAD( rc = m_pDIStream->readUShort( &ui16Value)))
				{
					goto Exit;
				}

				if( puiNumber)
				{
					*puiNumber = (FLMUINT)ui16Value;
				}
				else if( pui64Number)
				{
					*pui64Number = (FLMUINT64)ui16Value;
				}
			}
			else if( piNumber || pi64Number)
			{
				FLMINT16		i16Value;

				if( RC_BAD( rc = m_pDIStream->readShort( &i16Value)))
				{
					goto Exit;
				}

				if( piNumber)
				{
					*piNumber = (FLMINT)i16Value;
				}
				else if( pi64Number)
				{
					*pi64Number = (FLMINT)i16Value;
				}
			}
			break;
		}

		case WIRE_VALUE_TYPE_GEN_4:
		{
			if( puiNumber || pui64Number)
			{
				FLMUINT32	ui32Value;

				if( RC_BAD( rc = m_pDIStream->readUInt( &ui32Value)))
				{
					goto Exit;
				}

				if( puiNumber)
				{
					*puiNumber = (FLMUINT)ui32Value;
				}
				else if( pui64Number)
				{
					*pui64Number = (FLMUINT64)ui32Value;
				}
			}
			else if( piNumber || pi64Number)
			{
				FLMINT32		i32Value;

				if( RC_BAD( rc = m_pDIStream->readInt( &i32Value)))
				{
					goto Exit;
				}

				if( piNumber)
				{
					*piNumber = (FLMINT)i32Value;
				}
				else if( pi64Number)
				{
					*pi64Number = (FLMINT64)i32Value;
				}
			}
			break;
		}

		case WIRE_VALUE_TYPE_GEN_8:
		{
			if( puiNumber || piNumber)
			{
				rc = RC_SET( FERR_CONV_NUM_OVERFLOW);
			}
			else
			{
				if( pui64Number)
				{
					if( RC_BAD( rc = m_pDIStream->readUInt64( pui64Number)))
					{
						goto Exit;
					}
				}
				else if( pi64Number)
				{
					if( RC_BAD( rc = m_pDIStream->readInt64( pi64Number)))
					{
						goto Exit;
					}
				}
				else
				{
					flmAssert( 0);
					rc = RC_SET( FERR_INVALID_PARM);
					goto Exit;
				}
			}
			break;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Writes an unsigned number to the specified data output stream.	
*****************************************************************************/
RCODE FCS_WIRE::writeUnsignedNumber(
	FLMUINT		uiTag,
	FLMUINT64	ui64Number)
{
	RCODE			rc = FERR_OK;

	if( ui64Number <= (FLMUINT64)0x000000FF)
	{
		uiTag |= WIRE_VALUE_TYPE_GEN_1 <<
			WIRE_VALUE_TYPE_START_BIT;

		if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)uiTag)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = m_pDOStream->writeByte( (FLMBYTE)ui64Number)))
		{
			goto Exit;
		}
	}
	else if( ui64Number <= (FLMUINT64)0x0000FFFF)
	{
		uiTag |= WIRE_VALUE_TYPE_GEN_2 <<
			WIRE_VALUE_TYPE_START_BIT;

		if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)uiTag)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)ui64Number)))
		{
			goto Exit;
		}
	}
	else if( ui64Number <= (FLMUINT64)0xFFFFFFFF)
	{
		uiTag |= WIRE_VALUE_TYPE_GEN_4 <<
			WIRE_VALUE_TYPE_START_BIT;

		if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)uiTag)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = m_pDOStream->writeUInt32( (FLMUINT32)ui64Number)))
		{
			goto Exit;
		}
	}
	else
	{
		uiTag |= WIRE_VALUE_TYPE_GEN_8 <<
			WIRE_VALUE_TYPE_START_BIT;

		if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)uiTag)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = m_pDOStream->writeUInt64( ui64Number)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Writes a signed number to the specified data output stream.	
*****************************************************************************/
RCODE FCS_WIRE::writeSignedNumber(
	FLMUINT		uiTag,
	FLMINT64		i64Number)
{
	RCODE			rc = FERR_OK;

	if( RC_BAD( rc = writeUnsignedNumber( uiTag, (FLMUINT64)i64Number)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Skips a parameter or return value in the data stream
*****************************************************************************/
RCODE FCS_WIRE::skipValue( 
	FLMUINT		uiTag)
{
	RCODE			rc = FERR_OK;

	switch( ((uiTag & WIRE_VALUE_TYPE_MASK) >> 
		WIRE_VALUE_TYPE_START_BIT))
	{
		case WIRE_VALUE_TYPE_GEN_0:
		{
			break;
		}

		case WIRE_VALUE_TYPE_GEN_1:
		{
			if( RC_BAD( rc = m_pDIStream->skip( 1)))
			{
				goto Exit;
			}
			break;
		}

		case WIRE_VALUE_TYPE_GEN_2:
		{
			if( RC_BAD( rc = m_pDIStream->skip( 2)))
			{
				goto Exit;
			}
			break;
		}

		case WIRE_VALUE_TYPE_GEN_4:
		{
			if( RC_BAD( rc = m_pDIStream->skip( 4)))
			{
				goto Exit;
			}
			break;
		}

		case WIRE_VALUE_TYPE_GEN_8:
		{
			if( RC_BAD( rc = m_pDIStream->skip( 8)))
			{
				goto Exit;
			}
			break;
		}

		case WIRE_VALUE_TYPE_BINARY:
		{
			if( RC_BAD( rc = m_pDIStream->readBinary( NULL, NULL, NULL)))
			{
				goto Exit;
			}
			break;
		}

		case WIRE_VALUE_TYPE_LARGE_BINARY:
		{
			if( RC_BAD( rc = m_pDIStream->readLargeBinary( NULL, NULL, NULL)))
			{
				goto Exit;
			}
			break;
		}

		case WIRE_VALUE_TYPE_HTD:
		{
			if( RC_BAD( rc = m_pDIStream->readHTD( NULL, 0, 0, NULL, NULL)))
			{
				goto Exit;
			}
			break;
		}

		case WIRE_VALUE_TYPE_RECORD:
		{
			if( RC_BAD( rc = receiveRecord( NULL)))
			{
				goto Exit;
			}
		}

		case WIRE_VALUE_TYPE_UTF:
		{
			if( RC_BAD( rc = m_pDIStream->readUTF( NULL, NULL)))
			{
				goto Exit;
			}
			break;
		}

		default:
		{
			rc = RC_SET( FERR_FAILURE);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Sends an opcode to the client
*****************************************************************************/
RCODE FCS_WIRE::sendOpcode(
	FLMUINT		uiClass,
	FLMUINT		uiOp)
{
	FLMBYTE		ucClass = (FLMBYTE)uiClass;
	FLMBYTE		ucOp = (FLMBYTE)uiOp;
	RCODE			rc = FERR_OK;
	
	if( RC_BAD( rc = m_pDOStream->write( &ucClass, 1)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pDOStream->write( &ucOp, 1)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:		Sends a value to the client
*****************************************************************************/
RCODE FCS_WIRE::sendTerminate( void)
{
	RCODE			rc = FERR_OK;
	
	if( RC_BAD( rc = m_pDOStream->writeUShort( 0)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pDOStream->endMessage()))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Sends a value to the client
*****************************************************************************/
RCODE FCS_WIRE::sendNumber(
	FLMUINT			uiTag,
	FLMUINT64		ui64Value,
	FLMINT64			i64Value)
{
	RCODE				rc = FERR_OK;
	
	// Send the parameter tag and value.

	switch( uiTag)
	{
		case WIRE_VALUE_AREA_ID:
		case WIRE_VALUE_OP_SEQ_NUM:
		case WIRE_VALUE_FLAGS:
		case WIRE_VALUE_CLIENT_VERSION:
		case WIRE_VALUE_SESSION_ID:
		case WIRE_VALUE_SESSION_COOKIE:
		case WIRE_VALUE_CONTAINER_ID:
		case WIRE_VALUE_INDEX_ID:
		case WIRE_VALUE_ITEM_ID:
		case WIRE_VALUE_TRANSACTION_ID:
		case WIRE_VALUE_TRANSACTION_TYPE:
		case WIRE_VALUE_DRN:
		case WIRE_VALUE_COUNT:
		case WIRE_VALUE_AUTOTRANS:
		case WIRE_VALUE_MAX_LOCK_WAIT:
		case WIRE_VALUE_RECORD_COUNT:
		case WIRE_VALUE_BOOLEAN:
		case WIRE_VALUE_ITERATOR_ID:
		case WIRE_VALUE_SHARED_DICT_ID:
		case WIRE_VALUE_PARENT_DICT_ID:
		case WIRE_VALUE_TYPE:
		case WIRE_VALUE_NUMBER1:
		case WIRE_VALUE_NUMBER2:
		case WIRE_VALUE_NUMBER3:
		case WIRE_VALUE_ADDRESS:
		case WIRE_VALUE_FLAIM_VERSION:
		{
			if( RC_BAD( rc = writeUnsignedNumber( uiTag, ui64Value)))
			{
				goto Exit;
			}
			break;
		}

		case WIRE_VALUE_SIGNED_NUMBER:
		{
			if( RC_BAD( rc = writeSignedNumber( uiTag, i64Value)))
			{
				goto Exit;
			}
			break;
		}

		default:
		{
#ifdef FLM_DEBUG
			flmAssert( 0);
#else
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
#endif
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Sends a value to the client
*****************************************************************************/
RCODE FCS_WIRE::sendBinary(
	FLMUINT			uiTag,
	FLMBYTE *		pData,
	FLMUINT			uiLength)
{
	RCODE				rc = FERR_OK;
	
	// Send the parameter tag and value.

	switch( uiTag)
	{
		case WIRE_VALUE_PASSWORD:
		case WIRE_VALUE_SERIAL_NUM:
		{
			uiTag |= WIRE_VALUE_TYPE_BINARY <<
				WIRE_VALUE_TYPE_START_BIT;

			if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)uiTag)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = m_pDOStream->writeBinary( pData, uiLength)))
			{
				goto Exit;
			}
			break;
		}

		case WIRE_VALUE_BLOCK:
		{
			uiTag |= WIRE_VALUE_TYPE_LARGE_BINARY <<
				WIRE_VALUE_TYPE_START_BIT;

			if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)uiTag)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = m_pDOStream->writeLargeBinary( pData, uiLength)))
			{
				goto Exit;
			}
			break;
		}

		default:
		{
#ifdef FLM_DEBUG
			flmAssert( 0);
#else
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
#endif
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Sends a record
*****************************************************************************/
RCODE FCS_WIRE::sendRecord(
	FLMUINT			uiTag,
	FlmRecord *		pRecord)
{
	RCODE				rc = FERR_OK;
#define RECORD_OUTPUT_BUFFER_SIZE	64
	FLMBYTE			pucBuffer[ RECORD_OUTPUT_BUFFER_SIZE];
	FLMBYTE *		pucBufPos;
	FLMBYTE			ucDescriptor;

	// Send the parameter tag and value.

	switch( uiTag)
	{
		case WIRE_VALUE_RECORD:
		case WIRE_VALUE_FROM_KEY:
		case WIRE_VALUE_UNTIL_KEY:
		{
			uiTag |= WIRE_VALUE_TYPE_RECORD <<
				WIRE_VALUE_TYPE_START_BIT;

			if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)uiTag)))
			{
				goto Exit;
			}

			// The format of a record is (X = 1 bit):
			//
			// X				X					XXXXXX			0-64 bytes		HTD
			// RESERVED		HTD_FOLLOWS		ID_LENGTH		ID_VALUE			TREE (optional)
			//
			// This sequence can repeat, terminating with a 0 byte.

			ucDescriptor = 0;
			pucBufPos = pucBuffer;
			ucDescriptor |= (FLMBYTE)RECORD_HAS_HTD_FLAG;

			// Output the descriptor.

			ucDescriptor |= (FLMBYTE)RECORD_ID_SIZE;

			*pucBufPos = ucDescriptor;
			pucBufPos++;

			// Output the ID.  Current format of a record ID is:
			//
			//		4-byte container ID, 4-byte DRN

			f_UINT32ToBigEndian( (FLMUINT32)pRecord->getContainerID(), pucBufPos);
			pucBufPos += 4;

			f_UINT32ToBigEndian( (FLMUINT32)pRecord->getID(), pucBufPos);
			pucBufPos += 4;

			// Send the descriptor and record source.

			if( RC_BAD( rc = m_pDOStream->write( pucBuffer, 
				pucBufPos - pucBuffer)))
			{
				goto Exit;
			}

			// Send the record.

			if( RC_BAD( rc = m_pDOStream->writeHTD( NULL, pRecord, 
				FALSE, m_bSendGedcom)))
			{
				goto Exit;
			}
			
			break;
		}

		default:
		{
#ifdef FLM_DEBUG
			flmAssert( 0);
#else
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
#endif
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Sends a value to the client
*****************************************************************************/
RCODE FCS_WIRE::sendDrnList(
	FLMUINT			uiTag,
	FLMUINT *		puiList)
{
	FLMUINT		uiItemCount;
	FLMUINT		uiLoop;
	FLMUINT		uiBufSize = 0;
	FLMBYTE *	pucItemBuf = NULL;
	FLMBYTE *	pucItemPos;
	RCODE			rc = FERR_OK;
	
	if( !puiList)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Send the parameter tag and value.

	switch( uiTag)
	{
		case WIRE_VALUE_DRN_LIST:
		{
			uiTag |= WIRE_VALUE_TYPE_BINARY <<
				WIRE_VALUE_TYPE_START_BIT;

			if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)uiTag)))
			{
				goto Exit;
			}

			// Count the entries in the list.  For now, support only a list of
			// 2048 elements.

			for( uiItemCount = 0; uiItemCount < 2048; uiItemCount++)
			{
				if( !puiList[ uiItemCount])
				{
					break;
				}
			}

			// Allocate a buffer for the list.

			uiBufSize = (FLMUINT)(((FLMUINT)sizeof( FLMUINT) * uiItemCount) + (FLMUINT)4);
			if( RC_BAD( rc = f_calloc( uiBufSize, &pucItemBuf)))
			{
				goto Exit;
			}
			pucItemPos = pucItemBuf;

			// Set the item count.

			UD2FBA( (FLMUINT32)uiItemCount, pucItemPos);
			pucItemPos += 4;

			// Put the items into the buffer.

			for( uiLoop = 0; uiLoop < uiItemCount; uiLoop++)
			{
				UD2FBA( (FLMUINT32)puiList[ uiLoop], pucItemPos);
				pucItemPos += 4;
			}

			// Send the list.

			if( RC_BAD( rc = m_pDOStream->writeBinary(
				pucItemBuf, uiBufSize)))
			{
				goto Exit;
			}
			break;
		}

		default:
		{
#ifdef FLM_DEBUG
			flmAssert( 0);
#else
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
#endif
		}
	}

Exit:

	if( pucItemBuf)
	{
		f_free( (void **)&pucItemBuf);
	}

	return( rc);
}

/****************************************************************************
Desc:		Sends a value to the client
*****************************************************************************/
RCODE FCS_WIRE::sendString(
	FLMUINT			uiTag,
	FLMUNICODE *	puzString)
{
	RCODE				rc = FERR_OK;
	
	// Send the parameter tag and value.

	switch( uiTag)
	{
		case WIRE_VALUE_FILE_NAME:
		case WIRE_VALUE_FILE_PATH:
		case WIRE_VALUE_FILE_PATH_2:
		case WIRE_VALUE_FILE_PATH_3:
		case WIRE_VALUE_DICT_FILE_PATH:
		case WIRE_VALUE_ITEM_NAME:
		case WIRE_VALUE_DICT_BUFFER:
		{
			uiTag |= WIRE_VALUE_TYPE_UTF <<
				WIRE_VALUE_TYPE_START_BIT;

			if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)uiTag)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = m_pDOStream->writeUTF( puzString)))
			{
				goto Exit;
			}
			break;
		}

		default:
		{
#ifdef FLM_DEBUG
			flmAssert( 0);
#else
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
#endif
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Sends a value to the client
*****************************************************************************/
RCODE FCS_WIRE::sendHTD(
	FLMUINT			uiTag,
	NODE *			pHTD)
{
	RCODE				rc = FERR_OK;
	
	// Send the parameter tag and value.

	switch( uiTag)
	{
		case WIRE_VALUE_HTD:
		case WIRE_VALUE_ITERATOR_SELECT:
		case WIRE_VALUE_ITERATOR_FROM:
		case WIRE_VALUE_ITERATOR_WHERE:
		case WIRE_VALUE_ITERATOR_CONFIG:
		{
			uiTag |= WIRE_VALUE_TYPE_HTD <<
				WIRE_VALUE_TYPE_START_BIT;

			if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)uiTag)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = m_pDOStream->writeHTD( pHTD, NULL, TRUE, m_bSendGedcom)))
			{
				goto Exit;
			}
			break;
		}

		default:
		{
#ifdef FLM_DEBUG
			flmAssert( 0);
#else
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
#endif
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Sends a value to the client
*****************************************************************************/
RCODE FCS_WIRE::sendHTD(
	FLMUINT			uiTag,
	FlmRecord *		pRecord)
{
	RCODE				rc = FERR_OK;
	
	// Send the parameter tag and value.

	switch( uiTag)
	{
		case WIRE_VALUE_HTD:
		{
			uiTag |= WIRE_VALUE_TYPE_HTD <<
				WIRE_VALUE_TYPE_START_BIT;

			if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)uiTag)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = m_pDOStream->writeHTD( NULL, pRecord, 
				FALSE, m_bSendGedcom)))
			{
				goto Exit;
			}
			break;
		}

		default:
		{
#ifdef FLM_DEBUG
			flmAssert( 0);
#else
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
#endif
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Copies the current HTD tree to the application's pool
*****************************************************************************/
RCODE FCS_WIRE::getHTD( 
	F_Pool *		pPool,
	NODE **		ppTreeRV)
{
	RCODE			rc = FERR_OK;

	if( !m_pHTD)
	{
		*ppTreeRV = NULL;
		goto Exit;
	}

	if( (*ppTreeRV = GedCopy( pPool, GED_FOREST, m_pHTD)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

Exit:

	return( rc);
}
	
/****************************************************************************
Desc:		Sends a value to the client
*****************************************************************************/
RCODE FCS_WIRE::sendCreateOpts(
	FLMUINT			uiTag,
	CREATE_OPTS *	pCreateOpts)
{
	NODE *			pRootNd = NULL;
	void *			pvMark = m_pPool->poolMark();
	RCODE				rc = FERR_OK;
	FLMUINT			uiTmp;
	
	// If no create options, goto exit.

	if( !pCreateOpts)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Send the parameter tag and value.

	switch( uiTag)
	{
		case WIRE_VALUE_CREATE_OPTS:
		{
			uiTag |= WIRE_VALUE_TYPE_HTD << WIRE_VALUE_TYPE_START_BIT;

			if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)uiTag)))
			{
				goto Exit;
			}

			// Build the root node of the CreateOpts tree.

			if( (pRootNd = GedNodeMake( m_pPool, FCS_COPT_CONTEXT, &rc)) == NULL)
			{
				goto Exit;
			}

			// Add fields to the tree.

			if( RC_BAD( rc = gedAddField( m_pPool, pRootNd,
				FCS_COPT_BLOCK_SIZE, (void *)&pCreateOpts->uiBlockSize,
				0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = gedAddField( m_pPool, pRootNd,
				FCS_COPT_MIN_RFL_FILE_SIZE, (void *)&pCreateOpts->uiMinRflFileSize,
				0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = gedAddField( m_pPool, pRootNd,
				FCS_COPT_MAX_RFL_FILE_SIZE, (void *)&pCreateOpts->uiMaxRflFileSize,
				0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}

			uiTmp = pCreateOpts->bKeepRflFiles ? 1 : 0;
			if( RC_BAD( rc = gedAddField( m_pPool, pRootNd,
				FCS_COPT_KEEP_RFL_FILES, (void *)&uiTmp,
				0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}

			uiTmp = pCreateOpts->bLogAbortedTransToRfl ? 1 : 0;
			if( RC_BAD( rc = gedAddField( m_pPool, pRootNd,
				FCS_COPT_LOG_ABORTED_TRANS, (void *)&uiTmp,
				0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = gedAddField( m_pPool, pRootNd,
				FCS_COPT_DEFAULT_LANG, (void *)&pCreateOpts->uiDefaultLanguage,
				0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = gedAddField( m_pPool, pRootNd,
				FCS_COPT_VERSION, (void *)&pCreateOpts->uiVersionNum,
				0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = gedAddField( m_pPool, pRootNd,
				FCS_COPT_APP_MAJOR_VER, (void *)&pCreateOpts->uiAppMajorVer,
				0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = gedAddField( m_pPool, pRootNd,
				FCS_COPT_APP_MINOR_VER, (void *)&pCreateOpts->uiAppMinorVer,
				0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}

			// Send the tree.

			if( RC_BAD( rc = m_pDOStream->writeHTD( pRootNd, NULL, 
				TRUE, m_bSendGedcom)))
			{
				goto Exit;
			}
			break;
		}

		default:
		{
#ifdef FLM_DEBUG
			flmAssert( 0);
#else
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
#endif
		}
	}

Exit:

	m_pPool->poolReset( pvMark);
	return( rc);
}

/****************************************************************************
Desc:		Sends a value to the client
*****************************************************************************/
RCODE FCS_WIRE::sendNameTable(
	FLMUINT			uiTag,
	F_NameTable *	pNameTable)
{
	void *			pvMark = m_pPool->poolMark();
	NODE *			pRootNd;
	NODE *			pNd;
	NODE *			pItemIdNd;
	FLMUINT			uiMaxNameChars = 1024;
	FLMUNICODE *	puzItemName = NULL;
	FLMUINT			uiId;
	FLMUINT			uiType;
	FLMUINT			uiSubType;
	FLMUINT			uiNextPos;
	RCODE				rc = FERR_OK;
	
	// If the name table pointer is invalid, goto exit.

	if( !pNameTable)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Allocate a temporary name buffer
	
	if( RC_BAD( rc = m_pPool->poolAlloc( uiMaxNameChars * sizeof( FLMUNICODE),
		(void **)&puzItemName)))
	{
		goto Exit;
	}

	// Send the parameter tag and value.

	switch( uiTag)
	{
		case WIRE_VALUE_NAME_TABLE:
		{
			uiTag |= WIRE_VALUE_TYPE_HTD <<
				WIRE_VALUE_TYPE_START_BIT;

			if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)uiTag)))
			{
				goto Exit;
			}


			// Build the root node of the name table tree.

			if( (pRootNd = GedNodeMake( m_pPool, 
				FCS_NAME_TABLE_CONTEXT, &rc)) == NULL)
			{
				goto Exit;
			}
				
			uiNextPos = 0;
			while( pNameTable->getNextTagNumOrder( &uiNextPos, puzItemName, 
				NULL, uiMaxNameChars * sizeof( FLMUNICODE), 
				&uiId, &uiType, &uiSubType))
			{
				if( (pItemIdNd = GedNodeMake( m_pPool, 
					FCS_NAME_TABLE_ITEM_ID, &rc)) == NULL)
				{
					goto Exit;
				}

				if( RC_BAD( rc = GedPutUINT( m_pPool, pItemIdNd, uiId)))
				{
					goto Exit;
				}

				if( (pNd = GedNodeMake( m_pPool, 
					FCS_NAME_TABLE_ITEM_NAME, &rc)) == NULL)
				{
					goto Exit;
				}

				if( RC_BAD( rc = GedPutUNICODE( m_pPool, pNd, puzItemName)))
				{
					goto Exit;
				}

				GedChildGraft( pItemIdNd, pNd, GED_LAST);

				if( (pNd = GedNodeMake( m_pPool, 
					FCS_NAME_TABLE_ITEM_TYPE, &rc)) == NULL)
				{
					goto Exit;
				}

				if( RC_BAD( rc = GedPutUINT( m_pPool, pNd, uiType)))
				{
					goto Exit;
				}

				GedChildGraft( pItemIdNd, pNd, GED_LAST);

				if( (pNd = GedNodeMake( m_pPool, 
					FCS_NAME_TABLE_ITEM_SUBTYPE, &rc)) == NULL)
				{
					goto Exit;
				}

				if( RC_BAD( rc = GedPutUINT( m_pPool, pNd, uiSubType)))
				{
					goto Exit;
				}

				GedChildGraft( pItemIdNd, pNd, GED_LAST);

				// Graft the item into the tree

				GedChildGraft( pRootNd, pItemIdNd, GED_LAST);

				// Release CPU to prevent CPU hog

				f_yieldCPU();
			}

			// Send the tree.

			if( RC_BAD( rc = m_pDOStream->writeHTD( pRootNd, 
				NULL, TRUE, m_bSendGedcom)))
			{
				goto Exit;
			}
			break;
		}

		default:
		{
#ifdef FLM_DEBUG
			flmAssert( 0);
#else
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
#endif
		}
	}

Exit:

	m_pPool->poolReset( pvMark);
	return( rc);
}

/****************************************************************************
Desc:	Receives a record
*****************************************************************************/
RCODE FCS_WIRE::receiveRecord(
	FlmRecord **	ppRecord)
{
	FLMBYTE					ucDescriptor = 0;
	FLMUINT					uiIdLen = 0;
	FLMUINT32				ui32Container;
	FLMUINT32				ui32Drn;
	void *					pvMark = m_pPool->poolMark();
	FLMBOOL					bHasId = FALSE;
	RCODE						rc = FERR_OK;

	// Read the record.

	if( RC_BAD( rc = m_pDIStream->read( &ucDescriptor, 1, NULL)))
	{
		goto Exit;
	}

	uiIdLen = (FLMUINT)(ucDescriptor & RECORD_ID_SIZE_MASK);

	if( uiIdLen != RECORD_ID_SIZE)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
	else if( uiIdLen)
	{
		bHasId = TRUE;
	}

	// Read the record ID.

	if( bHasId)
	{
		if( RC_BAD( rc = m_pDIStream->readUInt( &ui32Container)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = m_pDIStream->readUInt( &ui32Drn)))
		{
			goto Exit;
		}
	}

	// Read the record.

	if( (ucDescriptor & RECORD_HAS_HTD_FLAG))
	{
		if( RC_BAD( rc = m_pDIStream->readHTD( m_pPool,
			ui32Container, ui32Drn, NULL, ppRecord)))
		{
			goto Exit;
		}
	}

Exit:

	if( RC_BAD( rc) && ppRecord && *ppRecord)
	{
		(*ppRecord)->Release();
		*ppRecord = NULL;
	}

	m_pPool->poolReset( pvMark);
	return( rc);
}

/****************************************************************************
Desc:	Receives a CREATE_OPTS structure as an HTD tree.
*****************************************************************************/
RCODE FCS_WIRE::receiveCreateOpts( void)
{
	NODE *		pRootNd;
	NODE *		pTmpNd;
	void *		pPoolMark;
	FLMUINT		fieldPath[ 8];
	FLMUINT		uiTmp;
	RCODE			rc = FERR_OK;

	pPoolMark = m_pPool->poolMark();
  
	// Initialize the CREATE_OPTS structure to its default values.

	fcsInitCreateOpts( &m_CreateOpts);

	// Receive the tree.

	if( RC_BAD( rc = m_pDIStream->readHTD( m_pPool,
		0, 0, &pRootNd, NULL)))
	{
		goto Exit;
	}

	// Parse the tree and extract the values.

	fieldPath[ 0] = FCS_COPT_CONTEXT;
	fieldPath[ 1] = FCS_COPT_BLOCK_SIZE;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pRootNd, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &m_CreateOpts.uiBlockSize);
	}

	fieldPath[ 0] = FCS_COPT_CONTEXT;
	fieldPath[ 1] = FCS_COPT_MIN_RFL_FILE_SIZE;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pRootNd, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &m_CreateOpts.uiMinRflFileSize);
	}

	fieldPath[ 0] = FCS_COPT_CONTEXT;
	fieldPath[ 1] = FCS_COPT_MAX_RFL_FILE_SIZE;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pRootNd, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &m_CreateOpts.uiMaxRflFileSize);
	}

	fieldPath[ 0] = FCS_COPT_CONTEXT;
	fieldPath[ 1] = FCS_COPT_KEEP_RFL_FILES;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pRootNd, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &uiTmp);
		m_CreateOpts.bKeepRflFiles = (FLMBOOL)((uiTmp)
															? (FLMBOOL)TRUE
															: (FLMBOOL)FALSE);
	}

	fieldPath[ 0] = FCS_COPT_CONTEXT;
	fieldPath[ 1] = FCS_COPT_LOG_ABORTED_TRANS;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pRootNd, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &uiTmp);
		m_CreateOpts.bLogAbortedTransToRfl = (FLMBOOL)((uiTmp)
																	  ? (FLMBOOL)TRUE
																	  : (FLMBOOL)FALSE);
	}

	fieldPath[ 0] = FCS_COPT_CONTEXT;
	fieldPath[ 1] = FCS_COPT_DEFAULT_LANG;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pRootNd, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &m_CreateOpts.uiDefaultLanguage);
	}

	fieldPath[ 0] = FCS_COPT_CONTEXT;
	fieldPath[ 1] = FCS_COPT_VERSION;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pRootNd, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &m_CreateOpts.uiVersionNum);
	}

	fieldPath[ 0] = FCS_COPT_CONTEXT;
	fieldPath[ 1] = FCS_COPT_APP_MAJOR_VER;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pRootNd, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &m_CreateOpts.uiAppMajorVer);
	}

	fieldPath[ 0] = FCS_COPT_CONTEXT;
	fieldPath[ 1] = FCS_COPT_APP_MINOR_VER;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pRootNd, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &m_CreateOpts.uiAppMinorVer);
	}

Exit:

	m_pPool->poolReset( pPoolMark);
	return( rc);
}

/****************************************************************************
Desc:	Receives a name table.
*****************************************************************************/
RCODE FCS_WIRE::receiveNameTable(
	F_NameTable **		ppNameTable)
{
	NODE *			pRootNd;
	NODE *			pItemIdNd;
	NODE *			pNd = NULL;
	void *			pvMark = m_pPool->poolMark();
	FLMUINT			uiMaxNameChars = 1024;
	FLMUNICODE *	puzItemName;
	FLMUINT			uiItemId;
	FLMUINT			uiItemType;
	FLMUINT			uiItemSubType;
	F_NameTable *	pNameTable = NULL;
	FLMBOOL			bCreatedTable = FALSE;
	RCODE				rc = FERR_OK;

	// Allocate a temporary name buffer

	if( RC_BAD( rc = m_pPool->poolAlloc( uiMaxNameChars * sizeof( FLMUNICODE),
		(void **)&puzItemName)))
	{
		goto Exit;
	}
	
	// Initialize the name table.

	if( (pNameTable = *ppNameTable) == NULL)
	{
		if( (pNameTable = f_new F_NameTable) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
		bCreatedTable = TRUE;
	}
	else
	{
		pNameTable->clearTable();
	}

	// Receive the tree.

	if( RC_BAD( rc = m_pDIStream->readHTD( m_pPool,
		0, 0, &pRootNd, NULL)))
	{
		goto Exit;
	}

	// Parse the tree and extract the values.

	pItemIdNd = GedChild( pRootNd);
	while( pItemIdNd)
	{
		if( GedTagNum( pItemIdNd) == FCS_NAME_TABLE_ITEM_ID)
		{
			if( RC_BAD( rc = GedGetUINT( pItemIdNd, &uiItemId)))
			{
				goto Exit;
			}

			uiItemType = 0;
			uiItemSubType = 0;
			pNd = GedChild( pItemIdNd);
			while( pNd)
			{
				switch( GedTagNum( pNd))
				{
					case FCS_NAME_TABLE_ITEM_NAME:
					{
						FLMUINT		uiStrLen = uiMaxNameChars * sizeof( FLMUNICODE);

						if( RC_BAD( rc = GedGetUNICODE( pNd, puzItemName,
							&uiStrLen)))
						{
							goto Exit;
						}

						break;
					}

					case FCS_NAME_TABLE_ITEM_TYPE:
					{
						if( RC_BAD( rc = GedGetUINT( pNd, &uiItemType)))
						{
							goto Exit;
						}

						break;
					}

					case FCS_NAME_TABLE_ITEM_SUBTYPE:
					{
						if( RC_BAD( rc = GedGetUINT( pNd, &uiItemSubType)))
						{
							goto Exit;
						}

						break;
					}
				}

				pNd = GedSibNext( pNd);
			}

			if( puzItemName[ 0])
			{
				if( RC_BAD( rc = pNameTable->addTag( puzItemName, NULL, 
					uiItemId, uiItemType, uiItemSubType, FALSE)))
				{
					goto Exit;
				}
			}
		}

		pItemIdNd = GedSibNext( pItemIdNd);

		// Release CPU to prevent CPU hog

		f_yieldCPU();
	}

	pNameTable->sortTags();
	*ppNameTable = pNameTable;
	pNameTable = NULL;

Exit:

	if( pNameTable && bCreatedTable)
	{
		pNameTable->Release();
	}

	m_pPool->poolReset( pvMark);
	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
FCL_WIRE::FCL_WIRE( CS_CONTEXT * pCSContext, FDB * pDb) :
	FCS_WIRE( pCSContext != NULL ? pCSContext->pIDataStream : NULL,
			  pCSContext != NULL ? pCSContext->pODataStream : NULL)
{
	m_pCSContext = pCSContext;
	m_pDb = pDb;

	if( m_pCSContext)
	{
		m_bSendGedcom = m_pCSContext->bGedcomSupport;
	}
}

/****************************************************************************
Desc:	Sets the CS CONTEXT in FCL_WIRE and the I/O streams in FCS_WIRE
*****************************************************************************/
void FCL_WIRE::setContext(
	CS_CONTEXT *		pCSContext)
{
	m_pCSContext = pCSContext;
	m_bSendGedcom = pCSContext->bGedcomSupport;
	FCS_WIRE::setDIStream( pCSContext->pIDataStream);
	FCS_WIRE::setDOStream( pCSContext->pODataStream);
}

/****************************************************************************
Desc:	Send a client/server opcode with session id, and optionally the
		database id
****************************************************************************/
RCODE FCL_WIRE::sendOp(
	FLMUINT			uiClass,
	FLMUINT			uiOp)
{
	RCODE				rc = FERR_OK;

	if (!m_pCSContext->bConnectionGood)
	{
		rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
		goto Exit;
	}

	// Send the class and opcode

	if (RC_BAD( rc = sendOpcode( (FLMBYTE)uiClass, (FLMBYTE)uiOp)))
	{
		goto Transmission_Error;
	}

	// Send session ID

	if (RC_BAD( rc = sendNumber(
		WIRE_VALUE_SESSION_ID, m_pCSContext->uiSessionId)))
	{
		goto Transmission_Error;
	}

	// Send session cookie

	if (RC_BAD( rc = sendNumber(
		WIRE_VALUE_SESSION_COOKIE, m_pCSContext->uiSessionCookie)))
	{
		goto Transmission_Error;
	}

	// Send operation sequence number

	m_pCSContext->uiOpSeqNum++;
	if (RC_BAD( rc = sendNumber( 
		WIRE_VALUE_OP_SEQ_NUM, m_pCSContext->uiOpSeqNum)))
	{
		goto Transmission_Error;
	}

Exit:

	return( rc);

Transmission_Error:
	m_pCSContext->bConnectionGood = FALSE;
	goto Exit;
}


/****************************************************************************
Desc:	This routine instructs the server to start or end a transaction
****************************************************************************/
RCODE FCL_WIRE::doTransOp(
	FLMUINT			uiOp,
	FLMUINT			uiTransType,
	FLMUINT			uiFlags,
	FLMUINT			uiMaxLockWait,
	FLMBYTE *		pszHeader,
	FLMBOOL			bForceCheckpoint)
{
	FLMUINT			uiTransFlags = 0;
	RCODE				rc = FERR_OK;

	// Send request to server

	if( RC_BAD( rc = sendOp( FCS_OPCLASS_TRANS, uiOp)))
	{
		goto Exit;
	}

	if( uiOp == FCS_OP_TRANSACTION_BEGIN)
	{
		if (RC_BAD( rc = sendNumber(
			WIRE_VALUE_TRANSACTION_TYPE, uiTransType)))
		{
			goto Transmission_Error;
		}

		if (RC_BAD( rc = sendNumber(
			WIRE_VALUE_MAX_LOCK_WAIT, uiMaxLockWait)))
		{
			goto Transmission_Error;
		}

		if( pszHeader)
		{
			uiTransFlags |= FCS_TRANS_FLAG_GET_HEADER;
		}

		if( uiFlags & FLM_DONT_KILL_TRANS)
		{
			uiTransFlags |= FCS_TRANS_FLAG_DONT_KILL;
		}

		if( uiFlags & FLM_DONT_POISON_CACHE)
		{
			uiTransFlags |= FCS_TRANS_FLAG_DONT_POISON;
		}
	}
	else if( uiOp == FCS_OP_TRANSACTION_COMMIT_EX)
	{
		if( pszHeader)
		{
			if( RC_BAD( rc = sendBinary(
				WIRE_VALUE_BLOCK, pszHeader, F_TRANS_HEADER_SIZE)))
			{
				goto Exit;
			}
		}

		if( bForceCheckpoint)
		{
			uiTransFlags |= FCS_TRANS_FORCE_CHECKPOINT;
		}
	}

	if( uiTransFlags)
	{
		if (RC_BAD( rc = sendNumber(
			WIRE_VALUE_FLAGS, uiTransFlags)))
		{
			goto Transmission_Error;
		}
	}

	if( RC_BAD( rc = sendTerminate()))
	{
		goto Transmission_Error;
	}

	// Read the response

	if( RC_BAD( rc = read()))
	{
		goto Transmission_Error;
	}

	if (RC_BAD( rc = getRCode()))
	{
		goto Exit;
	}

	if( pszHeader)
	{
		if( getBlockSize())
		{
			f_memcpy( pszHeader, getBlock(), getBlockSize());
		}
		else
		{
			f_memset( pszHeader, 0, 2048);
		}
	}

	if (!m_pDb)
	{
		m_pCSContext->bTransActive = (FLMBOOL)((uiOp == FCS_OP_TRANSACTION_BEGIN)
													  ? (FLMBOOL)TRUE
													  : (FLMBOOL)FALSE);
	}

Exit:

	return( rc);
Transmission_Error:
	m_pCSContext->bConnectionGood = FALSE;
	goto Exit;
}

/****************************************************************************
Desc:	Reads a server response for the client.
*****************************************************************************/
RCODE	FCL_WIRE::read( void)
{
	FLMUINT	uiTag;
	FLMUINT	uiCount = 0;
	FLMBOOL	bDone = FALSE;
	RCODE		rc = FERR_OK;

	// Read the opcode.

	if( RC_BAD( rc = readOpcode()))
	{
		goto Exit;
	}
	
	// Read the request / response values.
	
	for( ;;)
	{
		if (RC_BAD( rc = readCommon( &uiTag, &bDone)))
		{
			if( rc == FERR_EOF_HIT && !uiCount)
			{
				rc = FERR_OK;
			}
			goto Exit;
		}

		if( bDone)
		{
			goto Exit;
		}

		// uiTag will be non-zero if readCommon did not understand it.

		uiCount++;
		if( uiTag)
		{
			switch( (uiTag & WIRE_VALUE_TAG_MASK))
			{
				case WIRE_VALUE_NAME_TABLE:
				{
					if( RC_BAD( rc = receiveNameTable( &m_pNameTable)))
					{
						goto Exit;
					}
					break;
				}

				default:
				{
					if( RC_BAD( rc = skipValue( uiTag)))
					{
						goto Exit;
					}
					break;
				}
			}
		}
	}

Exit:

	if( rc == FERR_EOF_HIT)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	return( rc);
}
