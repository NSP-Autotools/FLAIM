//-------------------------------------------------------------------------
// Desc:	Routines for getting RFL information for the RFL viewer utility.
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

#include "flaim.h"
#include "flaimsys.h"
#include "rflread.h"

// GEDCOM tag numbers for data to be returned

#define RFL_PACKET_FIELD							1

// Local function prototypes

FSTATIC void rflGetNumValue(
	FLMBYTE *	pucBuffer,
	FLMUINT		uiBufferLen,
	FLMUINT		uiNumOffset,
	FLMUINT		uiNumLen,
	FLMUINT *	puiNum,
	FLMUINT *	puiNumLen
	);

FSTATIC void rflFormatTransID(
	RFL_PACKET *	pRflPacket,
	char **			ppszTmp
	);

FSTATIC void rflFormatTransIDs(
	RFL_PACKET *	pRflPacket,
	char **			ppszTmp
	);

FSTATIC void rflFormatIndex(
	RFL_PACKET *	pRflPacket,
	char **			ppszTmp
	);

FSTATIC void rflFormatContainer(
	RFL_PACKET *	pRflPacket,
	char **			ppszTmp
	);

FSTATIC void rflFormatDRN(
	RFL_PACKET *	pRflPacket,
	char **			ppszTmp
	);

FSTATIC void rflFormatEndBlockAddr(
	RFL_PACKET *	pRflPacket,
	char **			ppszTmp);

FSTATIC void rflFormatDRNRange(
	RFL_PACKET *	pRflPacket,
	char **			ppszTmp);

FSTATIC FLMUINT rflFindNextPacket(
	FLMUINT			uiStartOffset,
	FLMBOOL			bFindValidPacket);

FSTATIC FLMUINT rflFindPrevPacket(
	FLMUINT			uiStartOffset,
	FLMBOOL			bGoBackMoreThanOnePckt,
	FLMBOOL			bValidStartOffset);

FSTATIC RCODE rflRetrievePacket(
	FLMUINT			uiPrevPacketAddress,
	FLMUINT			uiFileOffset,
	RFL_PACKET *	pRflPacket);

FSTATIC RCODE rflGetNextOpPacket(
	RFL_PACKET *	pRflPacket,
	FLMBOOL *		pbFoundNext);

FSTATIC RCODE rflGetPrevOpPacket(
	RFL_PACKET *	pRflPacket,
	FLMBOOL *		pbFoundPrev);

FSTATIC RCODE rflPutNum(
	F_Pool *			pPool,
	NODE *			pLinkToNode,
	FLMBOOL			bPutAsSib,
	eDispTag			eDispTag,
	FLMUINT			uiNum,
	FLMUINT			uiOffset,
	FLMUINT			uiNumExpectedBytes,
	FLMUINT			uiNumBytes,
	NODE **			ppNode);

FSTATIC RCODE rflExpandPacketHdr(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	NODE **			ppParent);

FSTATIC RCODE rflExpandTrnsPacket(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	NODE **			ppForest);

FSTATIC RCODE rflExpandStartUnknownPacket(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	NODE **			ppForest);

FSTATIC RCODE rflExpandIndexSetPacket(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	NODE **			ppForest);

FSTATIC RCODE rflExpandBlkChainFreePacket(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	NODE **			ppForest);
									 
FSTATIC RCODE rflExpandReducePacket(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	NODE **			ppForest);

FSTATIC RCODE rflExpandUpgradePacket(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	NODE **			ppForest);

FSTATIC RCODE rflExpandIndexStatePacket(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	NODE **			ppForest);

FSTATIC RCODE rflExpandDataPacket(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	FLMBOOL			bOutputPacket,
	FLMUINT			uiPacketType,
	NODE **			ppDataPacketNode,
	FLMUINT *		puiDataLen,
	FLMUINT *		puiLevel);

FSTATIC RCODE rflExpandRecordPackets(
	F_Pool *			pPool,
	FLMUINT			uiOffset,
	FLMUINT			uiPacketType,
	NODE **			ppLastPacketNode,
	FLMUINT			uiPacketOffset);

FSTATIC RCODE rflExpandChangeFieldsPacket(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	FLMBOOL			bOutputPacket,
	NODE **			ppChangeFieldsPacketNode,
	FLMUINT *		puiDataLen);

FSTATIC RCODE rflExpandRecOpPacket(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	NODE **			ppForest,
	FLMUINT			uiPacketOffset);

FSTATIC RCODE rflExpandUnkPacket(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	NODE **			ppForest);

FSTATIC RCODE rflExpandEncryptionPacket(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	NODE **			ppForest);

FSTATIC RCODE rflExpandConfigSizePacket(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	NODE **			ppForest);

FSTATIC void rflFormatCount(
	RFL_PACKET *	pRflPacket,
	char **			ppszTmp);

FSTATIC void rflFormatFlags(
	RFL_PACKET *	pRflPacket,
	char **			ppszTmp);

FSTATIC void rflFormatVersionRange(
	RFL_PACKET *	pRflPacket,
	char **			ppszTmp);

FSTATIC void rflFormatDBKeyLen(
	RFL_PACKET *	pRflPacket,
	char **			ppszTmp);

/********************************************************************
Desc: Get a value from a buffer at the specified offset.
*********************************************************************/
FSTATIC void rflGetNumValue(
	FLMBYTE *	pucBuffer,
	FLMUINT		uiBufferLen,
	FLMUINT		uiNumOffset,
	FLMUINT		uiNumLen,
	FLMUINT *	puiNum,
	FLMUINT *	puiNumLen
	)
{
	FLMBYTE		uiTmp [4];
	FLMUINT		uiValidBytes;
	FLMBYTE *	pucNumBuf;

	if (uiNumOffset + uiNumLen > uiBufferLen)
	{
		if (uiNumOffset >= uiBufferLen)
		{
			uiValidBytes = 0;
		}
		else
		{
			uiValidBytes = (FLMUINT)(uiBufferLen - uiNumOffset);
		}
		f_memset( uiTmp, 0, sizeof( uiTmp));
		if (uiValidBytes)
		{
			f_memcpy( uiTmp, &pucBuffer [uiNumOffset], uiValidBytes);
		}
		pucNumBuf = &uiTmp [0];
	}
	else
	{
		pucNumBuf = &pucBuffer [uiNumOffset];
		uiValidBytes = uiNumLen;
	}
	if (uiNumLen == 4)
	{
		*puiNum = (FLMUINT)FB2UD( pucNumBuf);
	}
	else if (uiNumLen == 2)
	{
		*puiNum = (FLMUINT)FB2UW( pucNumBuf);
	}
	else
	{
		*puiNum = *pucNumBuf;
	}
	if (puiNumLen)
	{
		*puiNumLen = uiValidBytes;
	}
}

/********************************************************************
Desc: Format a count
*********************************************************************/
FSTATIC void rflFormatCount(
	RFL_PACKET *	pRflPacket,
	char **			ppszTmp
	)
{
	char *	pszTmp = *ppszTmp;

	while (*pszTmp)
		pszTmp++;
	if (pRflPacket->uiCountBytes == 4)
		*pszTmp++ = ' ';
	else
		*pszTmp++ = '*';
	f_sprintf( pszTmp, "CNT=%-10u ", (unsigned)pRflPacket->uiCount);
	while (*pszTmp)
		pszTmp++;
	*ppszTmp = pszTmp;
}

/********************************************************************
Desc: Format flags
*********************************************************************/
FSTATIC void rflFormatFlags(
	RFL_PACKET *	pRflPacket,
	char **			ppszTmp)
{
	char *	pszTmp = *ppszTmp;

	while (*pszTmp)
	{
		pszTmp++;
	}

	if (pRflPacket->uiFlagsBytes == 4)
	{
		*pszTmp++ = ' ';
	}
	else
	{
		*pszTmp++ = '*';
	}

	f_sprintf( pszTmp, "FLAGS=%-10u ", (unsigned)pRflPacket->uiFlags);

	while (*pszTmp)
	{
		pszTmp++;
	}

	*ppszTmp = pszTmp;
}

/********************************************************************
Desc: Format a transaction ID.
*********************************************************************/
FSTATIC void rflFormatTransID(
	RFL_PACKET *	pRflPacket,
	char **			ppszTmp
	)
{
	char *	pszTmp = *ppszTmp;

	while (*pszTmp)
		pszTmp++;
	if (pRflPacket->uiTransIDBytes == 4)
		*pszTmp++ = ' ';
	else
		*pszTmp++ = '*';
	f_sprintf( pszTmp, "T=%-10u ", (unsigned)pRflPacket->uiTransID);
	while (*pszTmp)
		pszTmp++;
	*ppszTmp = pszTmp;
}

/********************************************************************
Desc: Format a transaction IDs.
*********************************************************************/
FSTATIC void rflFormatTransIDs(
	RFL_PACKET *	pRflPacket,
	char **			ppszTmp
	)
{
	char *	pszTmp = *ppszTmp;

	while (*pszTmp)
		pszTmp++;
	if (pRflPacket->uiTransIDBytes == 4)
		*pszTmp++ = ' ';
	else
		*pszTmp++ = '*';
	f_sprintf( pszTmp, "T=%-10u ", (unsigned)pRflPacket->uiTransID);
	while (*pszTmp)
		pszTmp++;

	if (pRflPacket->uiLastCommittedTransIDBytes == 4)
		*pszTmp++ = ' ';
	else
		*pszTmp++ = '*';
	f_sprintf( pszTmp, "LT=%-10u ",
		(unsigned)pRflPacket->uiLastCommittedTransID);
	while (*pszTmp)
		pszTmp++;
	*ppszTmp = pszTmp;
}

/********************************************************************
Desc: Format an index number
*********************************************************************/
FSTATIC void rflFormatIndex(
	RFL_PACKET *	pRflPacket,
	char **			ppszTmp
	)
{
	char *	pszTmp = *ppszTmp;

	while (*pszTmp)
		pszTmp++;
	if (pRflPacket->uiIndexBytes == 2)
		*pszTmp++ = ' ';
	else
		*pszTmp++ = '*';
	f_sprintf( pszTmp, "I=%-5u ", (unsigned)pRflPacket->uiIndex);
	while (*pszTmp)
		pszTmp++;
	*ppszTmp = pszTmp;
}

/********************************************************************
Desc: Format a container number
*********************************************************************/
FSTATIC void rflFormatContainer(
	RFL_PACKET *	pRflPacket,
	char **			ppszTmp
	)
{
	char *	pszTmp = *ppszTmp;

	while (*pszTmp)
		pszTmp++;
	if (pRflPacket->uiContainerBytes == 2)
		*pszTmp++ = ' ';
	else
		*pszTmp++ = '*';
	f_sprintf( pszTmp, "C=%-5u ", (unsigned)pRflPacket->uiContainer);
	while (*pszTmp)
		pszTmp++;
	*ppszTmp = pszTmp;
}

/********************************************************************
Desc: Format a DRN.
*********************************************************************/
FSTATIC void rflFormatDRN(
	RFL_PACKET *	pRflPacket,
	char **			ppszTmp
	)
{
	char *	pszTmp = *ppszTmp;

	while (*pszTmp)
		pszTmp++;
	if (pRflPacket->uiDrnBytes == 4)
		*pszTmp++ = ' ';
	else
		*pszTmp++ = '*';
	f_sprintf( pszTmp, "D=%-9u (%08X) ", (unsigned)pRflPacket->uiDrn,
							(unsigned)pRflPacket->uiDrn);
	while (*pszTmp)
		pszTmp++;
	*ppszTmp = pszTmp;
}

/********************************************************************
Desc: Format an ending block address
*********************************************************************/
FSTATIC void rflFormatEndBlockAddr(
	RFL_PACKET *	pRflPacket,
	char **			ppszTmp)
{
	char *	pszTmp = *ppszTmp;

	while (*pszTmp)
	{
		pszTmp++;
	}

	if (pRflPacket->uiEndDrnBytes == 4)
	{
		*pszTmp++ = ' ';
	}
	else
	{
		*pszTmp++ = '*';
	}

	f_sprintf( pszTmp, "B=%-9u (%08X) ", (unsigned)pRflPacket->uiEndDrn,
							(unsigned)pRflPacket->uiEndDrn);

	while (*pszTmp)
	{
		pszTmp++;
	}

	*ppszTmp = pszTmp;
}

/********************************************************************
Desc: Format version range
*********************************************************************/
FSTATIC void rflFormatVersionRange(
	RFL_PACKET *	pRflPacket,
	char **			ppszTmp
	)
{
	char *	pszTmp = *ppszTmp;

	while (*pszTmp)
		pszTmp++;
	f_sprintf( pszTmp, " OLD=");
	while (*pszTmp)
		pszTmp++;
	if (pRflPacket->uiDrnBytes != 4)
	{
		*pszTmp++ = '*';
	}
	f_sprintf( pszTmp, "%u, NEW=", (unsigned)pRflPacket->uiDrn);
	while (*pszTmp)
		pszTmp++;

	if (pRflPacket->uiEndDrnBytes != 4)
	{
		*pszTmp++ = '*';
	}
	f_sprintf( pszTmp, "%u", (unsigned)pRflPacket->uiEndDrn);
	while (*pszTmp)
		pszTmp++;
	*ppszTmp = pszTmp;
}

/********************************************************************
Desc: Format Database Key Length
*********************************************************************/
FSTATIC void rflFormatDBKeyLen(
	RFL_PACKET *	pRflPacket,
	char **			ppszTmp
	)
{
	char *	pszTmp = *ppszTmp;

	if (pRflPacket->uiEndDrn < FLM_FILE_FORMAT_VER_4_60)
	{
		return;
	}

	while (*pszTmp)
		pszTmp++;
	f_sprintf( pszTmp, " DBKeyLen=");
	while (*pszTmp)
		pszTmp++;
	if (pRflPacket->uiCountBytes != 2)
	{
		*pszTmp++ = '*';
	}
	f_sprintf( pszTmp, "%u", (unsigned)pRflPacket->uiCount);
	while (*pszTmp)
		pszTmp++;
	*ppszTmp = pszTmp;
}

/********************************************************************
Desc: Format DRN range
*********************************************************************/
FSTATIC void rflFormatDRNRange(
	RFL_PACKET *	pRflPacket,
	char **			ppszTmp
	)
{
	char *	pszTmp = *ppszTmp;

	while (*pszTmp)
		pszTmp++;
	f_sprintf( pszTmp, " D=");
	while (*pszTmp)
		pszTmp++;
	if (pRflPacket->uiDrnBytes != 4)
	{
		*pszTmp++ = '*';
	}
	f_sprintf( pszTmp, "%u to ", (unsigned)pRflPacket->uiDrn);
	while (*pszTmp)
		pszTmp++;

	if (pRflPacket->uiEndDrnBytes != 4)
	{
		*pszTmp++ = '*';
	}
	f_sprintf( pszTmp, "%u", (unsigned)pRflPacket->uiEndDrn);
	while (*pszTmp)
		pszTmp++;
	*ppszTmp = pszTmp;
}

/********************************************************************
Desc: Format a display buffer given an operation sub-tree
*********************************************************************/
void RflFormatPacket(
	void *			pPacket,
	char *			pszDispBuffer
	)
{
	RFL_PACKET *	pRflPacket = (RFL_PACKET *)pPacket;
	char *			pszTmp;

	// Format the data into our display buffer.

	pszTmp = pszDispBuffer;
	f_sprintf( pszTmp, "%08X  ", (unsigned)pRflPacket->uiFileOffset);
	while (*pszTmp)
		pszTmp++;

	// If the packet address does not match, set packet type to unknown.

	if (!pRflPacket->bValidPacketType)
	{
		if (pRflPacket->bHavePacketType)
		{
			f_sprintf( pszTmp, "Unk (%02X)      ",
						(unsigned)pRflPacket->uiPacketType);
		}
		else
		{
			f_strcpy( pszTmp, "Unk (None)    ");
		}
	}
	else
	{
		switch (pRflPacket->uiPacketType)
		{
			case RFL_TRNS_BEGIN_PACKET:
				f_strcpy( pszTmp, "BeginTrans    ");
				rflFormatTransID( pRflPacket, &pszTmp);
				break;
			case RFL_TRNS_BEGIN_EX_PACKET:
				f_strcpy( pszTmp, "BeginTransEx  ");
				rflFormatTransIDs( pRflPacket, &pszTmp);
				break;
			case RFL_TRNS_COMMIT_PACKET:
				f_strcpy( pszTmp, "CommitTrans   ");
				rflFormatTransID( pRflPacket, &pszTmp);
				break;
			case RFL_TRNS_ABORT_PACKET:
				f_strcpy( pszTmp, "AbortTrans    ");
				rflFormatTransID( pRflPacket, &pszTmp);
				break;
			case RFL_ADD_RECORD_PACKET:
				f_strcpy( pszTmp, "  AddRecord   ");
				rflFormatTransID( pRflPacket, &pszTmp);
				rflFormatContainer( pRflPacket, &pszTmp);
				rflFormatDRN( pRflPacket, &pszTmp);
				break;
			case RFL_ADD_RECORD_PACKET_VER_2:
				f_strcpy( pszTmp, "  AddRecord2  ");
				rflFormatTransID( pRflPacket, &pszTmp);
				rflFormatContainer( pRflPacket, &pszTmp);
				rflFormatDRN( pRflPacket, &pszTmp);
				rflFormatFlags( pRflPacket, &pszTmp);
				break;
			case RFL_MODIFY_RECORD_PACKET:
				f_strcpy( pszTmp, "  ModRecord   ");
				rflFormatTransID( pRflPacket, &pszTmp);
				rflFormatContainer( pRflPacket, &pszTmp);
				rflFormatDRN( pRflPacket, &pszTmp);
				break;
			case RFL_MODIFY_RECORD_PACKET_VER_2:
				f_strcpy( pszTmp, "  ModRecord   ");
				rflFormatTransID( pRflPacket, &pszTmp);
				rflFormatContainer( pRflPacket, &pszTmp);
				rflFormatDRN( pRflPacket, &pszTmp);
				rflFormatFlags( pRflPacket, &pszTmp);
				break;
			case RFL_DELETE_RECORD_PACKET:
				f_strcpy( pszTmp, "  DelRecord   ");
				rflFormatTransID( pRflPacket, &pszTmp);
				rflFormatContainer( pRflPacket, &pszTmp);
				rflFormatDRN( pRflPacket, &pszTmp);
				break;
			case RFL_DELETE_RECORD_PACKET_VER_2:
				f_strcpy( pszTmp, "  DelRecord   ");
				rflFormatTransID( pRflPacket, &pszTmp);
				rflFormatContainer( pRflPacket, &pszTmp);
				rflFormatDRN( pRflPacket, &pszTmp);
				rflFormatFlags( pRflPacket, &pszTmp);
				break;
			case RFL_RESERVE_DRN_PACKET:
				f_strcpy( pszTmp, "  ReserveDRN  ");
				rflFormatTransID( pRflPacket, &pszTmp);
				rflFormatContainer( pRflPacket, &pszTmp);
				rflFormatDRN( pRflPacket, &pszTmp);
				break;
			case RFL_CHANGE_FIELDS_PACKET:
				f_strcpy( pszTmp, "    ChgFlds   ");
				break;
			case RFL_DATA_RECORD_PACKET:
				f_strcpy( pszTmp, "    DataRec   ");
				break;
			case RFL_ENC_DATA_RECORD_PACKET:
				f_strcpy( pszTmp, "    EDataRec  ");
				break;
			case RFL_DATA_RECORD_PACKET_VER_3:
				f_strcpy( pszTmp, "    DataRec3  ");
				break;
			case RFL_INDEX_SET_PACKET:
				f_strcpy( pszTmp, "  IndexSet    ");
				rflFormatTransID( pRflPacket, &pszTmp);
				rflFormatIndex( pRflPacket, &pszTmp);
				rflFormatDRNRange( pRflPacket, &pszTmp);
				break;
			case RFL_INDEX_SET_PACKET_VER_2:
				f_strcpy( pszTmp, "  IndexSet2   ");
				rflFormatTransID( pRflPacket, &pszTmp);
				rflFormatContainer( pRflPacket, &pszTmp);
				rflFormatIndex( pRflPacket, &pszTmp);
				rflFormatDRNRange( pRflPacket, &pszTmp);
				break;
			case RFL_BLK_CHAIN_FREE_PACKET:
				f_strcpy( pszTmp, "BlkChainFree  ");
				rflFormatTransID( pRflPacket, &pszTmp);
				rflFormatDRN( pRflPacket, &pszTmp);
				rflFormatCount( pRflPacket, &pszTmp);
				rflFormatEndBlockAddr( pRflPacket, &pszTmp);
				break;
			case RFL_START_UNKNOWN_PACKET:
				f_strcpy( pszTmp, "  StartUnk    ");
				rflFormatTransID( pRflPacket, &pszTmp);
				break;
			case RFL_UNKNOWN_PACKET:
				f_strcpy( pszTmp, "  UserUnk     ");
				break;
			case RFL_REDUCE_PACKET:
				f_strcpy( pszTmp, "Reduce        ");
				rflFormatTransID( pRflPacket, &pszTmp);
				rflFormatCount( pRflPacket, &pszTmp);
				break;
			case RFL_UPGRADE_PACKET:
				f_strcpy( pszTmp, "Upgrade       ");
				rflFormatTransID( pRflPacket, &pszTmp);
				rflFormatVersionRange( pRflPacket, &pszTmp);
				rflFormatDBKeyLen( pRflPacket, &pszTmp);
				break;
			case RFL_INDEX_SUSPEND_PACKET:
				f_strcpy( pszTmp, "Index Suspend ");
				rflFormatTransID( pRflPacket, &pszTmp);
				rflFormatIndex( pRflPacket, &pszTmp);
				break;
			case RFL_INDEX_RESUME_PACKET:
				f_strcpy( pszTmp, "Index Resume  ");
				rflFormatTransID( pRflPacket, &pszTmp);
				rflFormatIndex( pRflPacket, &pszTmp);
				break;
			case RFL_WRAP_KEY_PACKET:
				f_strcpy( pszTmp, "Wrap Key      ");
				rflFormatTransID( pRflPacket, &pszTmp);
				rflFormatDBKeyLen( pRflPacket, &pszTmp);
				break;
			case RFL_ENABLE_ENCRYPTION_PACKET:
				f_strcpy( pszTmp, "Enable Enc    ");
				rflFormatTransID( pRflPacket, &pszTmp);
				rflFormatDBKeyLen( pRflPacket, &pszTmp);
				break;
			case RFL_CONFIG_SIZE_EVENT_PACKET:
				f_strcpy( pszTmp, "Config Size   ");
				break;
		}
	}
}

/********************************************************************
Desc: Find the next packet in the RFL file starting from the
		specified start offset.  This routine will NOT go more than
		a packet body length worth down.  If it cannot find what looks
		like a valid packet by then, it will just return the start
		offset that was passed in.
*********************************************************************/
FSTATIC FLMUINT rflFindNextPacket(
	FLMUINT		uiStartOffset,
	FLMBOOL		bFindValidPacket
	)
{
	RCODE			rc;
	FLMUINT		uiNextAddr;
	FLMBYTE *	pucPacketHdr = NULL;
	FLMBYTE *	pucBuffer = NULL;
	FLMUINT		uiBytesToRead;
	FLMUINT		uiBytesRead;
	RFL_PACKET	RflPacket;

	uiBytesToRead = (FLMUINT)((bFindValidPacket)
									 ? (FLMUINT)RFL_MAX_PACKET_SIZE
									 : (FLMUINT)RFL_MAX_PACKET_BODY_SIZE);
	if (RC_BAD( rc = f_calloc(
								uiBytesToRead, &pucBuffer)))
	{
		uiNextAddr = 0;
		goto Exit;
	}
	pucPacketHdr = pucBuffer;

	// Read up to a full packet body.

	rc = gv_pRflFileHdl->read( uiStartOffset,
									uiBytesToRead, pucPacketHdr, &uiBytesRead);
	if (RC_BAD( rc))
	{
		if (rc != FERR_IO_END_OF_FILE || !uiBytesRead)
		{
			if (rc == FERR_IO_END_OF_FILE)
			{
				rc = FERR_OK;
			}
			uiNextAddr = 0;
			goto Exit;
		}
	}

	// Go until we something where the packet address matches the
	// file offset.

	uiNextAddr = uiStartOffset;
	while (uiBytesRead >= 4)
	{
		if ((FLMUINT)FB2UD( pucPacketHdr) == uiNextAddr)
		{
			if (!bFindValidPacket)
				break;

			// See if this is a valid packet

			if ((RC_OK( rflRetrievePacket( 0, uiNextAddr, &RflPacket))) &&
				 (RflPacket.bValidPacketType))
			{
				break;
			}
		}
		pucPacketHdr++;
		uiNextAddr++;
		uiBytesRead--;
	}

	// If we couldn't get a matching address, simply return
	// the start address that was passed in.

	if (uiBytesRead < 4)
	{
		uiNextAddr = (FLMUINT)((bFindValidPacket)
									  ? (FLMUINT)0
									  : uiStartOffset);
	}
Exit:
	if (pucBuffer)
	{
		f_free( &pucBuffer);
	}
	return( uiNextAddr);
}

/********************************************************************
Desc: Find the previous packet in the RFL file starting from the
		specified start offset.  This routine will NOT go more than
		a packet length worth back.  If it cannot find what looks
		like a valid packet by then, it will just return the start
		offset that was passed in.
*********************************************************************/
FSTATIC FLMUINT rflFindPrevPacket(
	FLMUINT			uiStartOffset,
	FLMBOOL			bGoBackMoreThanOnePckt,
	FLMBOOL			bValidStartOffset
	)
{
	RCODE			rc;
	FLMUINT		uiPrevAddr = 0;
	FLMUINT		uiReadOffset;
	FLMBYTE *	pucPacketHdr = NULL;
	FLMBYTE *	pucBuffer = NULL;
	FLMUINT		uiBytesToRead;
	FLMUINT		uiBytesRead;
	FLMUINT		uiBestCandidate;
	RFL_PACKET	RflPacket;

Get_Prev_Packet:

	if (uiStartOffset <= 512)
	{
		uiPrevAddr = 0;
		goto Exit;
	}

	// Read up to a full packet

	uiReadOffset = (FLMUINT)((uiStartOffset >
										(FLMUINT)(RFL_MAX_PACKET_SIZE + 512))
									 ? (FLMUINT)(uiStartOffset -
													 RFL_MAX_PACKET_SIZE)
									 : (FLMUINT)512);
	if (uiStartOffset - uiReadOffset <= 4)
	{
		uiPrevAddr = uiReadOffset;
		goto Exit;
	}

	if (pucBuffer)
	{
		f_free( &pucBuffer);
	}
	uiBytesToRead = uiStartOffset - uiReadOffset;
	if (RC_BAD( rc = f_calloc( uiBytesToRead, &pucBuffer)))
	{
		goto Exit;
	}
	pucPacketHdr = pucBuffer;

	rc = gv_pRflFileHdl->read( uiReadOffset,
									uiBytesToRead, pucPacketHdr, &uiBytesRead);
	if (RC_BAD( rc))
	{
		if (rc != FERR_IO_END_OF_FILE)
		{
			uiPrevAddr = 0;
			goto Exit;
		}
	}
	else if (uiBytesRead != uiBytesToRead)
	{
		uiPrevAddr = uiStartOffset - 4;
		goto Exit;
	}

	// Go until we something where the packet address matches the
	// file offset.

	uiBestCandidate =
	uiPrevAddr = uiStartOffset - 4;
	uiBytesRead -= 4;
	pucPacketHdr += uiBytesRead;
	for (;;)
	{
		if ((FLMUINT)FB2UD( pucPacketHdr) == uiPrevAddr)
		{
			if (uiBestCandidate != uiStartOffset - 4)
			{
				uiBestCandidate = uiPrevAddr;
			}

			// See if this is a real packet whose next address is
			// the same as uiStartOffset.

			if (RC_BAD( rflRetrievePacket( 0, uiPrevAddr, &RflPacket)))
			{
				uiPrevAddr = uiStartOffset - 4;
				goto Exit;
			}

			// If we have a valid packet type and the packet's
			// next packet address is the same as the start
			// offset we passed in, we have a packet.

			if (RflPacket.bValidPacketType)
			{
				if ((!bValidStartOffset) ||
					 (RflPacket.uiNextPacketAddress == uiStartOffset))
				{
					break;
				}
				else if (RflPacket.uiNextPacketAddress < uiStartOffset)
				{
					uiPrevAddr = uiBestCandidate;
					break;
				}
			}
		}
		if (!uiBytesRead)
		{
			if ((uiBestCandidate != uiStartOffset - 4) ||
				 (!bGoBackMoreThanOnePckt))
			{
				uiPrevAddr = uiBestCandidate;
				goto Exit;
			}
			uiStartOffset = uiReadOffset;
			goto Get_Prev_Packet;
		}
		pucPacketHdr--;
		uiBytesRead--;
		uiPrevAddr--;
	}
Exit:
	if( pucBuffer)
	{
		f_free( &pucBuffer);
	}
	return( uiPrevAddr);
}

/********************************************************************
Desc: Retrieves the packet at the specified file offset.
*********************************************************************/
FSTATIC RCODE rflRetrievePacket(
	FLMUINT			uiPrevPacketAddress,
	FLMUINT			uiFileOffset,
	RFL_PACKET *	pRflPacket
	)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiBytesToRead;
	FLMUINT		uiBytesRead;
	FLMUINT		uiBytes;
	FLMUINT		uiExpectedBodyLen = 0;
	FLMBYTE *	pucPacketHdr;
	FLMBYTE *	pucPacketBody = NULL;

	f_memset( pRflPacket, 0, sizeof( RFL_PACKET));
	pRflPacket->uiFileOffset = uiFileOffset;
	pRflPacket->uiPrevPacketAddress = uiPrevPacketAddress;

	// Read the packet header.

	pucPacketHdr = &gv_rflBuffer [0];
	uiBytesToRead = RFL_PACKET_OVERHEAD;
	if (RC_BAD( rc = gv_pRflFileHdl->read( uiFileOffset,
									uiBytesToRead, pucPacketHdr, &uiBytesRead)))
	{
		if (rc != FERR_IO_END_OF_FILE)
		{
			goto Exit;
		}
		else
		{
			if (!uiBytesRead)
			{
				goto Exit;
			}
			rc = FERR_OK;
		}
	}

	// Extract values from the packet header.

	rflGetNumValue( pucPacketHdr, uiBytesRead, RFL_PACKET_ADDRESS_OFFSET,
						4, &pRflPacket->uiPacketAddress,
						&pRflPacket->uiPacketAddressBytes);

	rflGetNumValue( pucPacketHdr, uiBytesRead, RFL_PACKET_CHECKSUM_OFFSET,
						1, &pRflPacket->uiPacketChecksum, &uiBytes);
	pRflPacket->bHavePacketChecksum = (FLMBOOL)((uiBytes)
						? (FLMBOOL)TRUE : (FLMBOOL)FALSE);

	rflGetNumValue( pucPacketHdr, uiBytesRead, RFL_PACKET_TYPE_OFFSET,
						1, &pRflPacket->uiPacketType, &uiBytes);
	pRflPacket->bValidPacketType =
	pRflPacket->bHavePacketType = (FLMBOOL)((uiBytes)
						? (FLMBOOL)TRUE : (FLMBOOL)FALSE);

	rflGetNumValue( pucPacketHdr, uiBytesRead, RFL_PACKET_BODY_LENGTH_OFFSET,
						2, &pRflPacket->uiPacketBodyLength,
						&pRflPacket->uiPacketBodyLengthBytes);

	// If the packet address does not match, set bValidPacketType to FALSE

	if ((!pRflPacket->bHavePacketType) ||
		 (pRflPacket->uiPacketAddressBytes < 4) ||
		 (pRflPacket->uiPacketAddress != uiFileOffset))
	{
		pRflPacket->bValidPacketType = FALSE;
		if (uiBytesRead < RFL_PACKET_OVERHEAD)
		{
			pRflPacket->uiNextPacketAddress = 0;
		}
		else if (pRflPacket->uiPacketAddress == uiFileOffset)
		{
			pRflPacket->uiNextPacketAddress =
							rflFindNextPacket( uiFileOffset + RFL_PACKET_OVERHEAD,
												FALSE);
		}
		else
		{
			pRflPacket->uiNextPacketAddress =
							rflFindNextPacket( uiFileOffset + 1, FALSE);
		}
	}
	else
	{
		pRflPacket->bHaveTimes =
			(FLMBOOL)((pRflPacket->uiPacketType & RFL_TIME_LOGGED_FLAG)
					  ? (FLMBOOL)TRUE
					  : (FLMBOOL)FALSE);
		pRflPacket->uiPacketType &= RFL_PACKET_TYPE_MASK;
		switch (pRflPacket->uiPacketType)
		{
			case RFL_TRNS_BEGIN_PACKET:
				uiExpectedBodyLen = 8;
				if (pRflPacket->bHaveTimes)
				{
					uiExpectedBodyLen += 4;
				}
				pRflPacket->uiNextPacketAddress =
								uiFileOffset + RFL_PACKET_OVERHEAD +
								uiExpectedBodyLen;
				break;
			case RFL_TRNS_BEGIN_EX_PACKET:
				uiExpectedBodyLen = 12;
				if (pRflPacket->bHaveTimes)
				{
					uiExpectedBodyLen += 4;
				}
				pRflPacket->uiNextPacketAddress =
								uiFileOffset + RFL_PACKET_OVERHEAD +
								uiExpectedBodyLen;
				break;
			case RFL_TRNS_COMMIT_PACKET:
			case RFL_TRNS_ABORT_PACKET:
				uiExpectedBodyLen = 8;
				if (pRflPacket->bHaveTimes)
				{
					uiExpectedBodyLen += 8;
				}
				pRflPacket->uiNextPacketAddress =
											uiFileOffset + RFL_PACKET_OVERHEAD +
											uiExpectedBodyLen;
				break;
			case RFL_ADD_RECORD_PACKET:
			case RFL_MODIFY_RECORD_PACKET:
			case RFL_DELETE_RECORD_PACKET:
			case RFL_RESERVE_DRN_PACKET:
				uiExpectedBodyLen = 10;
				if (pRflPacket->bHaveTimes)
				{
					uiExpectedBodyLen += 16;
				}
				pRflPacket->uiNextPacketAddress =
											uiFileOffset + RFL_PACKET_OVERHEAD +
											uiExpectedBodyLen;
				break;
			case RFL_ADD_RECORD_PACKET_VER_2:
			case RFL_MODIFY_RECORD_PACKET_VER_2:
			case RFL_DELETE_RECORD_PACKET_VER_2:
				uiExpectedBodyLen = 11;
				if (pRflPacket->bHaveTimes)
				{
					uiExpectedBodyLen += 16;
				}
				pRflPacket->uiNextPacketAddress =
											uiFileOffset + RFL_PACKET_OVERHEAD +
											uiExpectedBodyLen;
				break;
			case RFL_CHANGE_FIELDS_PACKET:
			case RFL_DATA_RECORD_PACKET:
			case RFL_ENC_DATA_RECORD_PACKET:
			case RFL_DATA_RECORD_PACKET_VER_3:
			case RFL_UNKNOWN_PACKET:
				uiExpectedBodyLen = pRflPacket->uiPacketBodyLength;
				if (uiExpectedBodyLen & 0x03)
				{
					uiExpectedBodyLen += (4 - (uiExpectedBodyLen & 0x0003));
				}
				if (uiExpectedBodyLen > RFL_MAX_PACKET_BODY_SIZE)
				{
					uiExpectedBodyLen = RFL_MAX_PACKET_BODY_SIZE;
				}
				pRflPacket->uiNextPacketAddress =
											uiFileOffset + RFL_PACKET_OVERHEAD +
											uiExpectedBodyLen;
				break;
			case RFL_INDEX_SET_PACKET:
				uiExpectedBodyLen = 14;
				if (pRflPacket->bHaveTimes)
				{
					uiExpectedBodyLen += 16;
				}
				pRflPacket->uiNextPacketAddress =
											uiFileOffset + RFL_PACKET_OVERHEAD +
											uiExpectedBodyLen;
				break;
			case RFL_INDEX_SET_PACKET_VER_2:
				uiExpectedBodyLen = 16;
				if (pRflPacket->bHaveTimes)
				{
					uiExpectedBodyLen += 16;
				}
				pRflPacket->uiNextPacketAddress =
											uiFileOffset + RFL_PACKET_OVERHEAD +
											uiExpectedBodyLen;
				break;
			case RFL_BLK_CHAIN_FREE_PACKET:
				uiExpectedBodyLen = 16;
				if (pRflPacket->bHaveTimes)
				{
					uiExpectedBodyLen += 16;
				}
				pRflPacket->uiNextPacketAddress =
											uiFileOffset + RFL_PACKET_OVERHEAD +
											uiExpectedBodyLen;
				break;
			case RFL_START_UNKNOWN_PACKET:
				uiExpectedBodyLen = 4;
				pRflPacket->uiNextPacketAddress =
											uiFileOffset + RFL_PACKET_OVERHEAD +
											uiExpectedBodyLen;
				break;
			case RFL_REDUCE_PACKET:
				uiExpectedBodyLen = 8;
				pRflPacket->uiNextPacketAddress =
											uiFileOffset + RFL_PACKET_OVERHEAD +
											uiExpectedBodyLen;
				break;
			case RFL_UPGRADE_PACKET:
				uiExpectedBodyLen = pRflPacket->uiPacketBodyLengthBytes;
				pRflPacket->uiNextPacketAddress =
											uiFileOffset + RFL_PACKET_OVERHEAD +
											uiExpectedBodyLen;
				break;
			case RFL_INDEX_SUSPEND_PACKET:
			case RFL_INDEX_RESUME_PACKET:
				uiExpectedBodyLen = 6;
				if (pRflPacket->bHaveTimes)
				{
					uiExpectedBodyLen += 8;
				}
				pRflPacket->uiNextPacketAddress =
											uiFileOffset + RFL_PACKET_OVERHEAD +
											uiExpectedBodyLen;
				break;
			case RFL_WRAP_KEY_PACKET:
			case RFL_ENABLE_ENCRYPTION_PACKET:
				uiExpectedBodyLen = pRflPacket->uiPacketBodyLength;
				break;
			case RFL_CONFIG_SIZE_EVENT_PACKET:
				uiExpectedBodyLen = 16;
				pRflPacket->uiNextPacketAddress =
											uiFileOffset + RFL_PACKET_OVERHEAD +
											uiExpectedBodyLen;
				break;
			default:
				pRflPacket->bValidPacketType = FALSE;
				pRflPacket->uiNextPacketAddress =
										rflFindNextPacket( uiFileOffset + RFL_PACKET_OVERHEAD,
												FALSE);
				break;
		}
	}

	// Get the rest of the packet.
	// Adjust the packet body length if the packet is encrypted.

	pRflPacket->bValidChecksum = FALSE;
	if (uiBytesRead < RFL_PACKET_OVERHEAD || !pRflPacket->bValidPacketType)
	{
		uiBytesRead = 0;
	}
	else
	{
		pucPacketBody = &gv_rflBuffer [RFL_PACKET_OVERHEAD];
		if (RC_BAD( rc = gv_pRflFileHdl->read( uiFileOffset + RFL_PACKET_OVERHEAD,
										uiExpectedBodyLen, pucPacketBody, &uiBytesRead)))
		{
			if (rc != FERR_IO_END_OF_FILE)
			{
				goto Exit;
			}
		}

		pRflPacket->bValidChecksum = TRUE;

		// For change field and data record packets, if we didn't
		// read everything, or the checksum doesn't verify,
		// determine where the next packet starts, starting from
		// the packet overhead.

		if (pRflPacket->uiPacketType == RFL_CHANGE_FIELDS_PACKET ||
			 pRflPacket->uiPacketType == RFL_DATA_RECORD_PACKET ||
		 	 pRflPacket->uiPacketType == RFL_ENC_DATA_RECORD_PACKET ||
			 pRflPacket->uiPacketType == RFL_DATA_RECORD_PACKET_VER_3)
		{
			if ((uiBytesRead != uiExpectedBodyLen) ||
				 (!pRflPacket->bHavePacketChecksum) ||
				 (RflCalcChecksum( gv_rflBuffer,
							uiExpectedBodyLen) !=
							pRflPacket->uiPacketChecksum))
			{
				pRflPacket->bValidChecksum = FALSE;
				pRflPacket->uiNextPacketAddress =
								rflFindNextPacket( uiFileOffset + RFL_PACKET_OVERHEAD,
												FALSE);
			}
		}
		else
		{
			if ((uiBytesRead != uiExpectedBodyLen) ||
				 (!pRflPacket->bHavePacketChecksum) ||
				 (RflCalcChecksum( gv_rflBuffer, uiExpectedBodyLen) !=
							pRflPacket->uiPacketChecksum))
			{
				pRflPacket->bValidChecksum = FALSE;
			}
		}
	}

	// Get the packet information we want to keep

	switch (pRflPacket->uiPacketType)
	{
		case RFL_TRNS_BEGIN_PACKET:
		case RFL_TRNS_BEGIN_EX_PACKET:

			// Get transaction ID

			rflGetNumValue( pucPacketBody, uiBytesRead, 0,
					4, &pRflPacket->uiTransID,
					&pRflPacket->uiTransIDBytes);

			// Get start seconds - also serves as KEY 2 for encryption

			rflGetNumValue( pucPacketBody, uiBytesRead, 4,
					4, &pRflPacket->uiStartSeconds,
					&pRflPacket->uiStartSecondsBytes);

			// Get the last committed transaction ID

			if( pRflPacket->uiPacketType == RFL_TRNS_BEGIN_EX_PACKET)
			{
				rflGetNumValue( pucPacketBody, uiBytesRead, 8,
						4, &pRflPacket->uiLastCommittedTransID,
						&pRflPacket->uiLastCommittedTransIDBytes);
			}

			// Get start microseconds

			if (pRflPacket->bHaveTimes)
			{
				if( pRflPacket->uiPacketType == RFL_TRNS_BEGIN_EX_PACKET)
				{
					rflGetNumValue( pucPacketBody, uiBytesRead, 12,
						4, &pRflPacket->uiStartMicro,
						&pRflPacket->uiStartMicroBytes);
				}
				else
				{
					rflGetNumValue( pucPacketBody, uiBytesRead, 8,
						4, &pRflPacket->uiStartMicro,
						&pRflPacket->uiStartMicroBytes);
				}
			}
			break;
		case RFL_TRNS_COMMIT_PACKET:
		case RFL_TRNS_ABORT_PACKET:

			// Get transaction ID

			rflGetNumValue( pucPacketBody, uiBytesRead, 0,
					4, &pRflPacket->uiTransID,
					&pRflPacket->uiTransIDBytes);

			// Get transaction begin offset in file.

			rflGetNumValue( pucPacketBody, uiBytesRead, 4,
					4, &pRflPacket->uiTransStartAddr,
					&pRflPacket->uiTransStartAddrBytes);

			// Get start time and start microseconds.

			if (pRflPacket->bHaveTimes)
			{
				rflGetNumValue( pucPacketBody, uiBytesRead, 8,
						4, &pRflPacket->uiStartSeconds,
						&pRflPacket->uiStartSecondsBytes);
				rflGetNumValue( pucPacketBody, uiBytesRead, 12,
						4, &pRflPacket->uiStartMicro,
						&pRflPacket->uiStartMicroBytes);
			}
			break;
		case RFL_ADD_RECORD_PACKET:
		case RFL_MODIFY_RECORD_PACKET:
		case RFL_DELETE_RECORD_PACKET:
		case RFL_RESERVE_DRN_PACKET:

			// Get transaction ID

			rflGetNumValue( pucPacketBody, uiBytesRead, 0,
					4, &pRflPacket->uiTransID,
					&pRflPacket->uiTransIDBytes);

			// Get the container

			rflGetNumValue( pucPacketBody, uiBytesRead, 4,
					2, &pRflPacket->uiContainer,
					&pRflPacket->uiContainerBytes);

			// Get DRN

			rflGetNumValue( pucPacketBody, uiBytesRead, 6,
					4, &pRflPacket->uiDrn,
					&pRflPacket->uiDrnBytes);

			// Get start and time and microseconds.

			if (pRflPacket->bHaveTimes)
			{

				rflGetNumValue( pucPacketBody, uiBytesRead, 10,
						4, &pRflPacket->uiStartSeconds,
						&pRflPacket->uiStartSecondsBytes);
				rflGetNumValue( pucPacketBody, uiBytesRead, 14,
						4, &pRflPacket->uiStartMicro,
						&pRflPacket->uiStartMicroBytes);
				rflGetNumValue( pucPacketBody, uiBytesRead, 18,
						4, &pRflPacket->uiEndSeconds,
						&pRflPacket->uiEndSecondsBytes);
				rflGetNumValue( pucPacketBody, uiBytesRead, 22,
						4, &pRflPacket->uiEndMicro,
						&pRflPacket->uiEndMicroBytes);
			}
			break;

		case RFL_ADD_RECORD_PACKET_VER_2:
		case RFL_MODIFY_RECORD_PACKET_VER_2:
		case RFL_DELETE_RECORD_PACKET_VER_2:
			// Get transaction ID

			rflGetNumValue( pucPacketBody, uiBytesRead, 0,
					4, &pRflPacket->uiTransID,
					&pRflPacket->uiTransIDBytes);

			// Get the container

			rflGetNumValue( pucPacketBody, uiBytesRead, 4,
					2, &pRflPacket->uiContainer,
					&pRflPacket->uiContainerBytes);

			// Get DRN

			rflGetNumValue( pucPacketBody, uiBytesRead, 6,
					4, &pRflPacket->uiDrn,
					&pRflPacket->uiDrnBytes);

			// Get flags

			rflGetNumValue( pucPacketBody, uiBytesRead, 10,
					4, &pRflPacket->uiFlags,
					&pRflPacket->uiFlagsBytes);

			// Get start and time and microseconds.

			if (pRflPacket->bHaveTimes)
			{

				rflGetNumValue( pucPacketBody, uiBytesRead, 10,
						4, &pRflPacket->uiStartSeconds,
						&pRflPacket->uiStartSecondsBytes);
				rflGetNumValue( pucPacketBody, uiBytesRead, 14,
						4, &pRflPacket->uiStartMicro,
						&pRflPacket->uiStartMicroBytes);
				rflGetNumValue( pucPacketBody, uiBytesRead, 18,
						4, &pRflPacket->uiEndSeconds,
						&pRflPacket->uiEndSecondsBytes);
				rflGetNumValue( pucPacketBody, uiBytesRead, 22,
						4, &pRflPacket->uiEndMicro,
						&pRflPacket->uiEndMicroBytes);
			}
			break;

		case RFL_INDEX_SET_PACKET:

			// Get transaction ID

			rflGetNumValue( pucPacketBody, uiBytesRead, 0,
					4, &pRflPacket->uiTransID,
					&pRflPacket->uiTransIDBytes);

			// Get index number

			rflGetNumValue( pucPacketBody, uiBytesRead, 4,
					2, &pRflPacket->uiIndex,
					&pRflPacket->uiIndexBytes);

			// Get start and end drns

			rflGetNumValue( pucPacketBody, uiBytesRead, 6,
					4, &pRflPacket->uiDrn,
					&pRflPacket->uiDrnBytes);
			rflGetNumValue( pucPacketBody, uiBytesRead, 10,
					4, &pRflPacket->uiEndDrn,
					&pRflPacket->uiEndDrnBytes);

			// Get start microseconds

			if (pRflPacket->bHaveTimes)
			{
				rflGetNumValue( pucPacketBody, uiBytesRead, 14,
						4, &pRflPacket->uiStartSeconds,
						&pRflPacket->uiStartSecondsBytes);
				rflGetNumValue( pucPacketBody, uiBytesRead, 18,
						4, &pRflPacket->uiStartMicro,
						&pRflPacket->uiStartMicroBytes);
				rflGetNumValue( pucPacketBody, uiBytesRead, 22,
						4, &pRflPacket->uiEndSeconds,
						&pRflPacket->uiEndSecondsBytes);
				rflGetNumValue( pucPacketBody, uiBytesRead, 26,
						4, &pRflPacket->uiEndMicro,
						&pRflPacket->uiEndMicroBytes);
			}
			break;
		case RFL_INDEX_SET_PACKET_VER_2:

			// Get transaction ID

			rflGetNumValue( pucPacketBody, uiBytesRead, 0,
					4, &pRflPacket->uiTransID,
					&pRflPacket->uiTransIDBytes);

			// Get container number

			rflGetNumValue( pucPacketBody, uiBytesRead, 4,
					2, &pRflPacket->uiContainer,
					&pRflPacket->uiContainerBytes);

			// Get index number

			rflGetNumValue( pucPacketBody, uiBytesRead, 6,
					2, &pRflPacket->uiIndex,
					&pRflPacket->uiIndexBytes);

			// Get start and end drns

			rflGetNumValue( pucPacketBody, uiBytesRead, 8,
					4, &pRflPacket->uiDrn,
					&pRflPacket->uiDrnBytes);
			rflGetNumValue( pucPacketBody, uiBytesRead, 12,
					4, &pRflPacket->uiEndDrn,
					&pRflPacket->uiEndDrnBytes);

			// Get start microseconds

			if (pRflPacket->bHaveTimes)
			{
				rflGetNumValue( pucPacketBody, uiBytesRead, 16,
						4, &pRflPacket->uiStartSeconds,
						&pRflPacket->uiStartSecondsBytes);
				rflGetNumValue( pucPacketBody, uiBytesRead, 20,
						4, &pRflPacket->uiStartMicro,
						&pRflPacket->uiStartMicroBytes);
				rflGetNumValue( pucPacketBody, uiBytesRead, 24,
						4, &pRflPacket->uiEndSeconds,
						&pRflPacket->uiEndSecondsBytes);
				rflGetNumValue( pucPacketBody, uiBytesRead, 28,
						4, &pRflPacket->uiEndMicro,
						&pRflPacket->uiEndMicroBytes);
			}
			break;
		case RFL_BLK_CHAIN_FREE_PACKET:

			// Get transaction ID

			rflGetNumValue( pucPacketBody, uiBytesRead, 0,
					4, &pRflPacket->uiTransID,
					&pRflPacket->uiTransIDBytes);

			// Get the tracker record number

			rflGetNumValue( pucPacketBody, uiBytesRead, 4,
					4, &pRflPacket->uiDrn,
					&pRflPacket->uiDrnBytes);

			// Get the block count

			rflGetNumValue( pucPacketBody, uiBytesRead, 8,
					4, &pRflPacket->uiCount,
					&pRflPacket->uiCountBytes);

			// Get the ending block address

			rflGetNumValue( pucPacketBody, uiBytesRead, 12,
					4, &pRflPacket->uiEndDrn,
					&pRflPacket->uiEndDrnBytes);

			// Get start microseconds

			if (pRflPacket->bHaveTimes)
			{
				rflGetNumValue( pucPacketBody, uiBytesRead, 16,
						4, &pRflPacket->uiStartSeconds,
						&pRflPacket->uiStartSecondsBytes);
				rflGetNumValue( pucPacketBody, uiBytesRead, 20,
						4, &pRflPacket->uiStartMicro,
						&pRflPacket->uiStartMicroBytes);
				rflGetNumValue( pucPacketBody, uiBytesRead, 24,
						4, &pRflPacket->uiEndSeconds,
						&pRflPacket->uiEndSecondsBytes);
				rflGetNumValue( pucPacketBody, uiBytesRead, 28,
						4, &pRflPacket->uiEndMicro,
						&pRflPacket->uiEndMicroBytes);
			}
			break;
		case RFL_START_UNKNOWN_PACKET:

			// Get transaction ID

			rflGetNumValue( pucPacketBody, uiBytesRead, 0,
					4, &pRflPacket->uiTransID,
					&pRflPacket->uiTransIDBytes);
			break;

		case RFL_REDUCE_PACKET:

			// Get transaction ID and count

			rflGetNumValue( pucPacketBody, uiBytesRead, 0,
					4, &pRflPacket->uiTransID,
					&pRflPacket->uiTransIDBytes);
			rflGetNumValue( pucPacketBody, uiBytesRead, 4,
					4, &pRflPacket->uiCount,
					&pRflPacket->uiCountBytes);
			break;

		case RFL_UPGRADE_PACKET:

			// Get transaction ID

			rflGetNumValue( pucPacketBody, uiBytesRead, 0,
					4, &pRflPacket->uiTransID,
					&pRflPacket->uiTransIDBytes);

			// Get old and new version numbers

			rflGetNumValue( pucPacketBody, uiBytesRead, 4,
					4, &pRflPacket->uiDrn,
					&pRflPacket->uiDrnBytes);
			rflGetNumValue( pucPacketBody, uiBytesRead, 8,
					4, &pRflPacket->uiEndDrn,
					&pRflPacket->uiEndDrnBytes);
			if (pRflPacket->uiEndDrn >= FLM_FILE_FORMAT_VER_4_60)
			{
				// Get the size of the DB key.
				rflGetNumValue( pucPacketBody, uiBytesRead, 12,
									 2, &pRflPacket->uiCount,
									 &pRflPacket->uiCountBytes);
			}
			break;
		case RFL_INDEX_SUSPEND_PACKET:
		case RFL_INDEX_RESUME_PACKET:

			// Get transaction ID

			rflGetNumValue( pucPacketBody, uiBytesRead, 0,
					4, &pRflPacket->uiTransID,
					&pRflPacket->uiTransIDBytes);

			// Get index number

			rflGetNumValue( pucPacketBody, uiBytesRead, 4,
					2, &pRflPacket->uiIndex,
					&pRflPacket->uiIndexBytes);
			break;
		case RFL_WRAP_KEY_PACKET:
		case RFL_ENABLE_ENCRYPTION_PACKET:
			rflGetNumValue( pucPacketBody, uiBytesRead, 0, 
				4, &pRflPacket->uiTransID,
				&pRflPacket->uiTransIDBytes);
			rflGetNumValue( pucPacketBody, uiBytesRead, 4,
				2, &pRflPacket->uiCount,
				&pRflPacket->uiCountBytes);
			pRflPacket->uiNextPacketAddress =
				uiFileOffset + RFL_PACKET_OVERHEAD +
				6 + pRflPacket->uiCount;
			break;
			
		case RFL_CONFIG_SIZE_EVENT_PACKET:

			// Get transaction ID, size threshhold, time interval, and
			// size interval

			rflGetNumValue( pucPacketBody, uiBytesRead, 0,
					4, &pRflPacket->uiTransID,
					&pRflPacket->uiTransIDBytes);
			rflGetNumValue( pucPacketBody, uiBytesRead, 4,
					4, &pRflPacket->uiSizeThreshold,
					&pRflPacket->uiSizeThresholdBytes);
			rflGetNumValue( pucPacketBody, uiBytesRead, 8,
					4, &pRflPacket->uiTimeInterval,
					&pRflPacket->uiTimeIntervalBytes);
			rflGetNumValue( pucPacketBody, uiBytesRead, 12,
					4, &pRflPacket->uiSizeInterval,
					&pRflPacket->uiSizeInterval);
			break;

		default:
			break;
	}

Exit:
	return( rc);
}

/********************************************************************
Desc: Positions to the next operation packet relative to the
		packet that is passed in.
*********************************************************************/
FSTATIC RCODE rflGetNextOpPacket(
	RFL_PACKET *	pRflPacket,
	FLMBOOL *		pbFoundNext
	)
{
	RCODE		rc = FERR_OK;

	*pbFoundNext = FALSE;
	for (;;)
	{
		// Stop when we either don't have a valid packet, or it is an
		// operation packet.

		if (!pRflPacket->bValidPacketType ||
			 (pRflPacket->uiPacketType != RFL_CHANGE_FIELDS_PACKET &&
			  pRflPacket->uiPacketType != RFL_DATA_RECORD_PACKET &&
			  pRflPacket->uiPacketType != RFL_ENC_DATA_RECORD_PACKET &&
			  pRflPacket->uiPacketType != RFL_DATA_RECORD_PACKET_VER_3))
		{
			*pbFoundNext = TRUE;
			break;
		}

		// If there is no next packet, we need to break out of this loop
		// and search backwards.

		if (!pRflPacket->uiNextPacketAddress)
		{
			break;
		}

		// Get the next packet.

		if (RC_BAD( rc = rflRetrievePacket( pRflPacket->uiFileOffset,
											pRflPacket->uiNextPacketAddress,
											pRflPacket)))
		{
			goto Exit;
		}
	}
Exit:
	return( rc);
}

/********************************************************************
Desc: Retrieves the next operation in the RFL file and formats
		it into GEDCOM for display in the viewer.
*********************************************************************/
RCODE RflGetNextNode(
	NODE *		pCurrOpNode,
	FLMBOOL		bOperationsOnly,
	F_Pool *		pPool,
	NODE **		ppNextNodeRV,
	FLMBOOL		bStopAtEOF
	)
{
	RCODE				rc = FERR_OK;
	NODE *			pPacketNode = NULL;
	void *			pvMark = pPool->poolMark();
	RFL_PACKET *	pRflPacket;
	FLMUINT			uiNextPacketAddr;
	FLMUINT			uiPrevPacketAddr;

	if (!pCurrOpNode)
	{
		uiNextPacketAddr = 512;
		uiPrevPacketAddr = 0;
	}
	else
	{

		// If there is no next packet, return NULL.

		pRflPacket = (RFL_PACKET *)GedValPtr( pCurrOpNode);
		uiNextPacketAddr = pRflPacket->uiNextPacketAddress;
		if (!uiNextPacketAddr)
		{
			goto Exit;
		}
		uiPrevPacketAddr = pRflPacket->uiFileOffset;
	}
	if (bStopAtEOF && (FLMUINT64)uiNextPacketAddr > gv_ui64RflEof)
	{
		// pPacketNode should be NULL at this point.
		goto Exit;	// Should return FERR_OK;
	}

	// Create the packet node.

	if ((pPacketNode = GedNodeCreate( pPool, RFL_PACKET_FIELD,
										0, &rc)) == NULL)
	{
		goto Exit;
	}
	if ((pRflPacket = (RFL_PACKET *)GedAllocSpace( pPool, pPacketNode,
										FLM_BINARY_TYPE, sizeof( RFL_PACKET))) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	if (RC_BAD( rc = rflRetrievePacket( uiPrevPacketAddr,
										uiNextPacketAddr, pRflPacket)))
	{
		goto Exit;
	}

	// If the request is for an operation, get the next operation packet.

	if (bOperationsOnly)
	{
		FLMBOOL	bFoundNext;

		if (RC_BAD( rc = rflGetNextOpPacket( pRflPacket, &bFoundNext)))
		{
			goto Exit;
		}

		// If there is no next packet, we don't want to return anything.

		if (!bFoundNext)
		{
			pPacketNode = NULL;
			goto Exit;
		}
	}

Exit:
	if (RC_BAD( rc) || !pPacketNode)
	{
		if (rc == FERR_IO_END_OF_FILE)
		{
			rc = FERR_OK;
		}
		*ppNextNodeRV = NULL;
		pPool->poolReset( pvMark);
	}
	else
	{
		*ppNextNodeRV = pPacketNode;
	}
	return( rc);
}

/********************************************************************
Desc: Positions to the previous operation packet relative to the
		packet that was passed in.
*********************************************************************/
FSTATIC RCODE rflGetPrevOpPacket(
	RFL_PACKET *	pRflPacket,
	FLMBOOL *		pbFoundPrev
	)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bValidStartOffset;
	FLMUINT	uiPrevPacketAddress;

	*pbFoundPrev = FALSE;
	for (;;)
	{

		// Stop when we either don't have a valid packet, or it is an
		// operation packet.

		if (!pRflPacket->bValidPacketType ||
			 (pRflPacket->uiPacketType != RFL_CHANGE_FIELDS_PACKET &&
			  pRflPacket->uiPacketType != RFL_DATA_RECORD_PACKET &&
			  pRflPacket->uiPacketType != RFL_ENC_DATA_RECORD_PACKET &&
			  pRflPacket->uiPacketType != RFL_DATA_RECORD_PACKET_VER_3))
		{
			*pbFoundPrev = TRUE;
			break;
		}

		bValidStartOffset = (FLMBOOL)(((pRflPacket->uiPacketAddressBytes == 4) &&
											  (pRflPacket->uiPacketAddress ==
												pRflPacket->uiFileOffset))
											 ? (FLMBOOL)TRUE
											 : (FLMBOOL)FALSE);
		uiPrevPacketAddress = rflFindPrevPacket( pRflPacket->uiFileOffset, FALSE,
										bValidStartOffset);

		// If there is no previous packet, we are done.

		if (!uiPrevPacketAddress)
		{
			break;
		}

		// Get the previous packet.

		if (RC_BAD( rc = rflRetrievePacket( 0, uiPrevPacketAddress,
											pRflPacket)))
		{
			goto Exit;
		}
	}
Exit:
	return( rc);
}

/********************************************************************
Desc: Retrieves the previous operation in the RFL file and formats
		it into GEDCOM for display in the viewer.
*********************************************************************/
RCODE RflGetPrevNode(
	NODE *		pCurrOpNode,
	FLMBOOL		bOperationsOnly,
	F_Pool *		pPool,
	NODE **		ppNextNodeRV
	)
{
	RCODE				rc = FERR_OK;
	NODE *			pPacketNode = NULL;
	void *			pvMark = pPool->poolMark();
	RFL_PACKET *	pRflPacket;
	FLMUINT64		ui64PrevPacketAddress;
	FLMBOOL			bValidStartOffset;
	FLMBOOL			bPositioningToEOF = FALSE;

	// If pCurrOpNode is NULL, position to the last packet in the file

	if (!pCurrOpNode)
	{
		if (RC_BAD( rc = gv_pRflFileHdl->size( &ui64PrevPacketAddress)))
		{
			goto Exit;
		}
		ui64PrevPacketAddress = (FLMUINT64)rflFindPrevPacket( (FLMUINT)ui64PrevPacketAddress,
												TRUE, FALSE);
		bPositioningToEOF = TRUE;
	}
	else
	{
		pRflPacket = (RFL_PACKET *)GedValPtr( pCurrOpNode);

		// If there is no previous packet pointer, read backwards to find it.

		if ((ui64PrevPacketAddress = (FLMUINT64)pRflPacket->uiPrevPacketAddress) == 0)
		{
			bValidStartOffset = (FLMBOOL)(((pRflPacket->uiPacketAddressBytes == 4) &&
												  (pRflPacket->uiPacketAddress ==
												   pRflPacket->uiFileOffset))
												 ? (FLMBOOL)TRUE
												 : (FLMBOOL)FALSE);

			ui64PrevPacketAddress = (FLMUINT64)rflFindPrevPacket( pRflPacket->uiFileOffset,
												FALSE, bValidStartOffset);
		}
	}

	if (!ui64PrevPacketAddress)
	{
		goto Exit;
	}

	// Create the packet node.

	if ((pPacketNode = GedNodeCreate( pPool, RFL_PACKET_FIELD,
										0, &rc)) == NULL)
	{
		goto Exit;
	}
	if ((pRflPacket = (RFL_PACKET *)GedAllocSpace( pPool, pPacketNode,
										FLM_BINARY_TYPE, sizeof( RFL_PACKET))) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	if (RC_BAD( rc = rflRetrievePacket( 0, (FLMUINT)ui64PrevPacketAddress,
										pRflPacket)))
	{
		goto Exit;
	}
	if (bPositioningToEOF)
	{
		gv_ui64RflEof = (FLMUINT64)pRflPacket->uiNextPacketAddress;
	}

	// If the request is for an operation, get the previous operation packet.

	if (bOperationsOnly)
	{
		FLMBOOL	bFoundPrev;

		if (RC_BAD( rc = rflGetPrevOpPacket( pRflPacket, &bFoundPrev)))
		{
			goto Exit;
		}

		// If there is no previous packet, we don't want to return anything.

		if (!bFoundPrev)
		{
			pPacketNode = NULL;
			goto Exit;
		}
	}

Exit:
	if (RC_BAD( rc) || !pPacketNode)
	{
		if (rc == FERR_IO_END_OF_FILE)
		{
			rc = FERR_OK;
		}
		*ppNextNodeRV = NULL;
		pPool->poolReset( pvMark);
	}
	else
	{
		*ppNextNodeRV = pPacketNode;
	}
	return( rc);
}

/********************************************************************
Desc: Retrieves the node closest to the specified address.  First it
		searches for a node that follows the address.  Otherwise,
		it searches for a node prior to the address.
*********************************************************************/
RCODE RflPositionToNode(
	FLMUINT		uiFileOffset,
	FLMBOOL		bOperationsOnly,
	F_Pool *		pPool,
	NODE **		ppNodeRV
	)
{
	RCODE				rc = FERR_OK;
	void *			pvMark = pPool->poolMark();
	NODE *			pPacketNode = NULL;
	FLMUINT			uiPacketAddr;
	FLMUINT64		ui64FileSize;
	FLMBOOL			bFound;
	RFL_PACKET *	pRflPacket;

	if (RC_BAD( rc = gv_pRflFileHdl->size( &ui64FileSize)))
	{
		goto Exit;
	}

	// If the specified offset is beyond the current EOF,
	// simply position to the last packet.

	if ((FLMUINT64)uiFileOffset >= ui64FileSize)
	{
		rc = RflGetPrevNode( NULL, bOperationsOnly, pPool, &pPacketNode);
		goto Exit;
	}

	// If offset <= 512, just get the first packet.

	if (uiFileOffset <= 512)
	{
		rc = RflGetNextNode( NULL, bOperationsOnly, pPool, &pPacketNode);
		goto Exit;
	}

	// Create the packet node.

	if ((pPacketNode = GedNodeCreate( pPool, RFL_PACKET_FIELD,
								0, &rc)) == NULL)
	{
		goto Exit;
	}
	if ((pRflPacket = (RFL_PACKET *)GedAllocSpace( pPool, pPacketNode,
										FLM_BINARY_TYPE, sizeof( RFL_PACKET))) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// See if we can find a valid packet that comes on or after
	// the specified address.

	uiPacketAddr = rflFindNextPacket( uiFileOffset, TRUE);
	if (uiPacketAddr)
	{
		if (RC_BAD( rc = rflRetrievePacket( 0, uiPacketAddr, pRflPacket)))
		{
			goto Exit;
		}

		if (bOperationsOnly)
		{
			if (RC_BAD( rc = rflGetNextOpPacket( pRflPacket, &bFound)))
			{
				goto Exit;
			}

			// If we found a packet, we are done.  Otherwise, fall
			// through and try to find a previous packet.

			if (bFound)
			{
				goto Exit;
			}
		}
	}

	// At this point, we know we didn't find a packet by searching forward,
	// so we will try searching backwards.

	uiPacketAddr = rflFindPrevPacket( uiFileOffset, FALSE, FALSE);
	if ((uiPacketAddr) && (bOperationsOnly))
	{
		if (RC_BAD( rc = rflRetrievePacket( 0, uiPacketAddr,
											pRflPacket)))
		{
			goto Exit;
		}

		if (bOperationsOnly)
		{
			if (RC_BAD( rc = rflGetPrevOpPacket( pRflPacket, &bFound)))
			{
				goto Exit;
			}

			// If we found a packet, we are done.  Otherwise, fall
			// through and try to find a previous packet.

			if (bFound)
			{
				goto Exit;
			}
		}
	}

	// At this point, we know we didn't find a packet in either direction
	// by looking at only one packet worth of data, so just return
	// the unknown packet at the address that was passed in.

	if (RC_BAD( rc = rflRetrievePacket( 0, uiFileOffset, pRflPacket)))
	{
		goto Exit;
	}

Exit:
	if (RC_BAD( rc) || !pPacketNode)
	{
		if (rc == FERR_IO_END_OF_FILE)
		{
			rc = FERR_OK;
		}
		*ppNodeRV = NULL;
		pPool->poolReset( pvMark);
	}
	else
	{
		*ppNodeRV = pPacketNode;
	}
	return( rc);
}

/********************************************************************
Desc: Puts a number in as the last child of the parent node.
*********************************************************************/
FSTATIC RCODE rflPutNum(
	F_Pool *		pPool,
	NODE *		pLinkToNode,
	FLMBOOL		bPutAsSib,
	eDispTag		eDispTag,
	FLMUINT		uiNum,
	FLMUINT		uiOffset,
	FLMUINT		uiNumExpectedBytes,
	FLMUINT		uiNumBytes,
	NODE **		ppNode
	)
{
	RCODE		rc = FERR_OK;
	NODE *	pNode = NULL;
	NODE *	pNode2;

	if (uiNumBytes)
	{

		// Create the number node.

		if ((pNode = GedNodeCreate( pPool, makeTagNum( eDispTag), uiOffset, &rc)) == NULL)
		{
			goto Exit;
		}

		// Put the value into the node just created.

		if (RC_BAD( rc = GedPutUINT( pPool, pNode, uiNum)))
		{
			goto Exit;
		}

		// Graft the node in as the parent's last child.

		if (pLinkToNode)
		{
			if (bPutAsSib)
			{
				GedSibGraft( pLinkToNode, pNode, GED_LAST);
			}
			else
			{
				GedChildGraft( pLinkToNode, pNode, GED_LAST);
			}
		}

		if (uiNumBytes != uiNumExpectedBytes)
		{

			// Create the number of bytes valid node.

			if ((pNode2 = GedNodeCreate( pPool,
											makeTagNum( RFL_NUM_BYTES_VALID_FIELD),
											0, &rc)) == NULL)
			{
				goto Exit;
			}

			// Put the value into the node just created.

			if (RC_BAD( rc = GedPutUINT( pPool, pNode2, uiNumBytes)))
			{
				goto Exit;
			}

			// Graft the node in as child to the child

			GedChildGraft( pNode, pNode2, GED_LAST);
		}
	}

Exit:
	if (ppNode)
	{
		*ppNode = pNode;
	}
	return( rc);
}

/********************************************************************
Desc: Expands a packet header into multiple GEDCOM nodes for
		display.
*********************************************************************/
FSTATIC RCODE rflExpandPacketHdr(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	NODE **			ppParent
	)
{
	RCODE		rc = FERR_OK;
	eDispTag	eTagNum;
	NODE *	pParent;
	FLMUINT	uiOffset;
	NODE *	pNode;

	if (!pRflPacket->bValidPacketType)
	{
		eTagNum = RFL_UNKNOWN_PACKET_FIELD;
	}
	else
	{
		switch (pRflPacket->uiPacketType)
		{
			case RFL_TRNS_BEGIN_PACKET:
				eTagNum = RFL_TRNS_BEGIN_FIELD;
				break;
			case RFL_TRNS_BEGIN_EX_PACKET:
				eTagNum = RFL_TRNS_BEGIN_EX_FIELD;
				break;
			case RFL_TRNS_COMMIT_PACKET:
				eTagNum = RFL_TRNS_COMMIT_FIELD;
				break;
			case RFL_TRNS_ABORT_PACKET:
				eTagNum = RFL_TRNS_ABORT_FIELD;
				break;
			case RFL_ADD_RECORD_PACKET:
			case RFL_ADD_RECORD_PACKET_VER_2:
				eTagNum = RFL_RECORD_ADD_FIELD;
				break;
			case RFL_MODIFY_RECORD_PACKET:
			case RFL_MODIFY_RECORD_PACKET_VER_2:
				eTagNum = RFL_RECORD_MODIFY_FIELD;
				break;
			case RFL_DELETE_RECORD_PACKET:
			case RFL_DELETE_RECORD_PACKET_VER_2:
				eTagNum = RFL_RECORD_DELETE_FIELD;
				break;
			case RFL_RESERVE_DRN_PACKET:
				eTagNum = RFL_RESERVE_DRN_FIELD;
				break;
			case RFL_CHANGE_FIELDS_PACKET:
				eTagNum = RFL_CHANGE_FIELDS_FIELD;
				break;
			case RFL_DATA_RECORD_PACKET:
				eTagNum = RFL_DATA_RECORD_FIELD;
				break;
			case RFL_ENC_DATA_RECORD_PACKET:
				eTagNum = RFL_ENC_DATA_RECORD_FIELD;
				break;
			case RFL_DATA_RECORD_PACKET_VER_3:
				eTagNum = RFL_DATA_RECORD3_FIELD;
				break;
			case RFL_INDEX_SET_PACKET:
				eTagNum = RFL_INDEX_SET_FIELD;
				break;
			case RFL_INDEX_SET_PACKET_VER_2:
				eTagNum = RFL_INDEX_SET2_FIELD;
				break;
			case RFL_BLK_CHAIN_FREE_PACKET:
				eTagNum = RFL_BLK_CHAIN_FREE_FIELD;
				break;
			case RFL_START_UNKNOWN_PACKET:
				eTagNum = RFL_START_UNKNOWN_FIELD;
				break;
			case RFL_UNKNOWN_PACKET:
				eTagNum = RFL_UNKNOWN_USER_PACKET_FIELD;
				break;
			case RFL_REDUCE_PACKET:
				eTagNum = RFL_REDUCE_PACKET_FIELD;
				break;
			case RFL_UPGRADE_PACKET:
				eTagNum = RFL_UPGRADE_PACKET_FIELD;
				break;
			case RFL_INDEX_SUSPEND_PACKET:
				eTagNum = RFL_INDEX_SUSPEND_FIELD;
				break;
			case RFL_INDEX_RESUME_PACKET:
				eTagNum = RFL_INDEX_RESUME_FIELD;
				break;
			case RFL_WRAP_KEY_PACKET:
				eTagNum = RFL_WRAP_KEY_FIELD;
				break;
			case RFL_ENABLE_ENCRYPTION_PACKET:
				eTagNum = RFL_ENABLE_ENCRYPTION_FIELD;
				break;
			case RFL_CONFIG_SIZE_EVENT_PACKET:
				eTagNum = RFL_CONFIG_SIZE_EVENT_FIELD;
				break;
			default:
				eTagNum = RFL_UNKNOWN_PACKET_FIELD;
				break;
		}
	}

	// Create the packet node.

	if ((pParent = GedNodeCreate( pPool, makeTagNum( eTagNum),
									pRflPacket->uiFileOffset, &rc)) == NULL)
	{
		goto Exit;
	}

	// If packet type is unknown, put it into the data portion of the
	// field - if we have it.

	if ((eTagNum == RFL_UNKNOWN_PACKET_FIELD) &&
		 (pRflPacket->bHavePacketType))
	{
		if (RC_BAD( rc = GedPutUINT( pPool, pParent,
									pRflPacket->uiPacketType)))
		{
			goto Exit;
		}
	}

	// Add other fields from the packet header.

	// Output the packet address

	uiOffset = pRflPacket->uiFileOffset;
	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
								RFL_PACKET_ADDRESS_FIELD,
									pRflPacket->uiPacketAddress,
									uiOffset + RFL_PACKET_ADDRESS_OFFSET, 4,
									pRflPacket->uiPacketAddressBytes, NULL)))
	{
		goto Exit;
	}

	// Output the checksum.

	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE, 
								RFL_PACKET_CHECKSUM_FIELD,
									pRflPacket->uiPacketChecksum,
									uiOffset + RFL_PACKET_CHECKSUM_OFFSET, 1,
									(FLMUINT)((pRflPacket->bHavePacketChecksum)
												? (FLMUINT)1
												: (FLMUINT)0), &pNode)))
	{
		goto Exit;
	}
	if ((pRflPacket->bHavePacketChecksum) &&
		 (!pRflPacket->bValidChecksum))
	{
		if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
									RFL_PACKET_CHECKSUM_VALID_FIELD, 0, 0, 1, 1,
									NULL)))
		{
			goto Exit;
		}
	}
	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
								RFL_PACKET_BODY_LENGTH_FIELD,
									pRflPacket->uiPacketBodyLength,
									uiOffset + RFL_PACKET_BODY_LENGTH_OFFSET, 2,
									pRflPacket->uiPacketBodyLengthBytes, NULL)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
								RFL_NEXT_PACKET_ADDRESS_FIELD,
									pRflPacket->uiNextPacketAddress,
									0, 4, 4, NULL)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
								RFL_PREV_PACKET_ADDRESS_FIELD,
									pRflPacket->uiPrevPacketAddress,
									0, 4, 4, NULL)))
	{
		goto Exit;
	}
Exit:
	*ppParent = pParent;
	return( rc);
}

/********************************************************************
Desc: Expands a transaction packet into multiple GEDCOM nodes
		for display of all of the subcomponents.
*********************************************************************/
FSTATIC RCODE rflExpandTrnsPacket(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	NODE **			ppForest
	)
{
	RCODE		rc = FERR_OK;
	void *	pvMark = pPool->poolMark();
	NODE *	pParent = NULL;
	NODE *	pLastNode;
	FLMUINT	uiOffset;

	// Output generic packet header information.

	if (RC_BAD( rc = rflExpandPacketHdr( pRflPacket, pPool, &pParent)))
	{
		goto Exit;
	}

	// Output transaction ID

	uiOffset = pRflPacket->uiFileOffset + RFL_PACKET_OVERHEAD;
	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
									RFL_TRANS_ID_FIELD,
									pRflPacket->uiTransID, uiOffset,
									4, pRflPacket->uiTransIDBytes, &pLastNode)))
	{
		goto Exit;
	}
	uiOffset += 4;

	if( pRflPacket->uiPacketType == RFL_TRNS_BEGIN_PACKET ||
		pRflPacket->uiPacketType == RFL_TRNS_BEGIN_EX_PACKET)
	{

		// Output the start seconds

		if (RC_BAD( rc = rflPutNum( pPool, pLastNode, TRUE,
									RFL_START_SECONDS_FIELD,
									pRflPacket->uiStartSeconds,
									uiOffset,
									4, pRflPacket->uiStartSecondsBytes,
									&pLastNode)))
		{
			goto Exit;
		}
		uiOffset += 4;

		if (pRflPacket->uiPacketType == RFL_TRNS_BEGIN_EX_PACKET)
		{
			if (RC_BAD( rc = rflPutNum( pPool, pLastNode, TRUE,
										RFL_LAST_COMMITTED_TRANS_ID_FIELD,
										pRflPacket->uiLastCommittedTransID,
										uiOffset,
										4, pRflPacket->uiLastCommittedTransIDBytes,
										&pLastNode)))
			{
				goto Exit;
			}
			uiOffset += 4;
		}

		// Output the start microseconds, if present

		if (pRflPacket->bHaveTimes)
		{
			if (RC_BAD( rc = rflPutNum( pPool, pLastNode, TRUE,
										RFL_START_MSEC_FIELD,
										pRflPacket->uiStartMicro, uiOffset,
										4, pRflPacket->uiStartMicroBytes,
										&pLastNode)))
			{
				goto Exit;
			}
			uiOffset += 4;
		}
	}
	else
	{
		if (RC_BAD( rc = rflPutNum( pPool, pLastNode, TRUE,
									RFL_START_TRNS_ADDR_FIELD,
									pRflPacket->uiTransStartAddr,
									uiOffset,
									4, pRflPacket->uiTransStartAddrBytes,
									&pLastNode)))
		{
			goto Exit;
		}
		uiOffset += 4;

		if (pRflPacket->bHaveTimes)
		{

			// Output the start seconds & microseconds as
			// an END time - because it represents the time
			// the transaction ended.

			if (RC_BAD( rc = rflPutNum( pPool, pLastNode, TRUE,
										RFL_END_SECONDS_FIELD,
										pRflPacket->uiStartSeconds,
										uiOffset,
										4, pRflPacket->uiStartSecondsBytes,
										&pLastNode)))
			{
				goto Exit;
			}
			uiOffset += 4;

			if (RC_BAD( rc = rflPutNum( pPool, pLastNode, TRUE,
										RFL_END_MSEC_FIELD,
										pRflPacket->uiStartMicro,
										uiOffset,
										4, pRflPacket->uiStartMicroBytes,
										&pLastNode)))
			{
				goto Exit;
			}
			uiOffset += 4;
		}
	}
Exit:
	if (RC_BAD( rc) || !pParent)
	{
		if (rc == FERR_IO_END_OF_FILE)
		{
			rc = FERR_OK;
		}
		*ppForest = NULL;
		pPool->poolReset( pvMark);
	}
	else
	{
		*ppForest = pParent;
	}
	return( rc);
}

/********************************************************************
Desc: Expands a start unknown packet into multiple GEDCOM nodes
		for display of all of the subcomponents.
*********************************************************************/
FSTATIC RCODE rflExpandStartUnknownPacket(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	NODE **			ppForest
	)
{
	RCODE		rc = FERR_OK;
	void *	pvMark = pPool->poolMark();
	NODE *	pParent = NULL;
	NODE *	pLastNode;
	FLMUINT	uiOffset;

	// Output generic packet header information.

	if (RC_BAD( rc = rflExpandPacketHdr( pRflPacket, pPool, &pParent)))
	{
		goto Exit;
	}

	// Output transaction ID

	uiOffset = pRflPacket->uiFileOffset + RFL_PACKET_OVERHEAD;
	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
									RFL_TRANS_ID_FIELD,
									pRflPacket->uiTransID, uiOffset,
									4, pRflPacket->uiTransIDBytes, &pLastNode)))
	{
		goto Exit;
	}
	uiOffset += 4;

Exit:
	if (RC_BAD( rc) || !pParent)
	{
		if (rc == FERR_IO_END_OF_FILE)
		{
			rc = FERR_OK;
		}
		*ppForest = NULL;
		pPool->poolReset( pvMark);
	}
	else
	{
		*ppForest = pParent;
	}
	return( rc);
}

/********************************************************************
Desc: Expands an index set packet into multiple GEDCOM nodes
		for display of all of the subcomponents.
*********************************************************************/
FSTATIC RCODE rflExpandIndexSetPacket(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	NODE **			ppForest
	)
{
	RCODE		rc = FERR_OK;
	void *	pvMark = pPool->poolMark();
	NODE *	pParent = NULL;
	NODE *	pLastNode;
	FLMUINT	uiOffset;

	// Output generic packet header information.

	if (RC_BAD( rc = rflExpandPacketHdr( pRflPacket, pPool, &pParent)))
	{
		goto Exit;
	}

	// Output transaction ID

	uiOffset = pRflPacket->uiFileOffset + RFL_PACKET_OVERHEAD;
	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
									RFL_TRANS_ID_FIELD,
									pRflPacket->uiTransID, uiOffset,
									4, pRflPacket->uiTransIDBytes, &pLastNode)))
	{
		goto Exit;
	}
	uiOffset += 4;

	if (pRflPacket->uiPacketType == RFL_INDEX_SET_PACKET_VER_2)
	{

		// Output container number

		if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
										RFL_CONTAINER_FIELD,
										pRflPacket->uiContainer, uiOffset,
										2, pRflPacket->uiContainerBytes, &pLastNode)))
		{
			goto Exit;
		}
		uiOffset += 2;
	}

	// Output index number

	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
									RFL_INDEX_NUM_FIELD,
									pRflPacket->uiIndex, uiOffset,
									2, pRflPacket->uiIndexBytes, &pLastNode)))
	{
		goto Exit;
	}
	uiOffset += 2;

	// Output start DRN

	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
									RFL_START_DRN_FIELD,
									pRflPacket->uiDrn, uiOffset,
									4, pRflPacket->uiDrnBytes, &pLastNode)))
	{
		goto Exit;
	}
	uiOffset += 4;

	// Output end DRN

	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
									RFL_END_DRN_FIELD,
									pRflPacket->uiEndDrn, uiOffset,
									4, pRflPacket->uiEndDrnBytes, &pLastNode)))
	{
		goto Exit;
	}
	uiOffset += 4;

	if (pRflPacket->bHaveTimes)
	{

		// Output the start time and microseconds

		if (RC_BAD( rc = rflPutNum( pPool, pLastNode, TRUE,
									RFL_START_SECONDS_FIELD,
									pRflPacket->uiStartSeconds,
									uiOffset,
									4, pRflPacket->uiStartSecondsBytes,
									&pLastNode)))
		{
			goto Exit;
		}
		uiOffset += 4;

		if (RC_BAD( rc = rflPutNum( pPool, pLastNode, TRUE,
									RFL_START_MSEC_FIELD,
									pRflPacket->uiStartMicro,
									uiOffset,
									4, pRflPacket->uiStartMicroBytes,
									&pLastNode)))
		{
			goto Exit;
		}
		uiOffset += 4;

		// Output the end time and microseconds

		if (RC_BAD( rc = rflPutNum( pPool, pLastNode, TRUE,
									RFL_END_SECONDS_FIELD,
									pRflPacket->uiEndSeconds,
									uiOffset,
									4, pRflPacket->uiEndSecondsBytes,
									&pLastNode)))
		{
			goto Exit;
		}
		uiOffset += 4;

		if (RC_BAD( rc = rflPutNum( pPool, pLastNode, TRUE,
									RFL_END_MSEC_FIELD,
									pRflPacket->uiEndMicro,
									uiOffset,
									4, pRflPacket->uiEndMicroBytes,
									&pLastNode)))
		{
			goto Exit;
		}
		uiOffset += 4;
	}
Exit:
	if (RC_BAD( rc) || !pParent)
	{
		if (rc == FERR_IO_END_OF_FILE)
		{
			rc = FERR_OK;
		}
		*ppForest = NULL;
		pPool->poolReset( pvMark);
	}
	else
	{
		*ppForest = pParent;
	}
	return( rc);
}

/********************************************************************
Desc: Expands a block chain free packet for display of all
		of the subcomponents.
*********************************************************************/
FSTATIC RCODE rflExpandBlkChainFreePacket(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	NODE **			ppForest)
{
	RCODE		rc = FERR_OK;
	void *	pvMark = pPool->poolMark();
	NODE *	pParent = NULL;
	NODE *	pLastNode;
	FLMUINT	uiOffset;

	// Output generic packet header information.

	if (RC_BAD( rc = rflExpandPacketHdr( pRflPacket, pPool, &pParent)))
	{
		goto Exit;
	}

	// Output transaction ID

	uiOffset = pRflPacket->uiFileOffset + RFL_PACKET_OVERHEAD;
	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
									RFL_TRANS_ID_FIELD,
									pRflPacket->uiTransID, uiOffset,
									4, pRflPacket->uiTransIDBytes, &pLastNode)))
	{
		goto Exit;
	}
	uiOffset += 4;

	// Output the tracker record number

	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
									RFL_TRACKER_REC_FIELD,
									pRflPacket->uiDrn, uiOffset,
									4, pRflPacket->uiDrnBytes, &pLastNode)))
	{
		goto Exit;
	}
	uiOffset += 4;

	// Output the block count

	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
									RFL_BLOCK_COUNT_FIELD,
									pRflPacket->uiCount, uiOffset,
									4, pRflPacket->uiCountBytes, &pLastNode)))
	{
		goto Exit;
	}
	uiOffset += 4;

	// Output end block address

	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
									RFL_END_BLK_ADDR_FIELD,
									pRflPacket->uiEndDrn, uiOffset,
									4, pRflPacket->uiEndDrnBytes, &pLastNode)))
	{
		goto Exit;
	}
	uiOffset += 4;

	if (pRflPacket->bHaveTimes)
	{

		// Output the start time and microseconds

		if (RC_BAD( rc = rflPutNum( pPool, pLastNode, TRUE,
									RFL_START_SECONDS_FIELD,
									pRflPacket->uiStartSeconds,
									uiOffset,
									4, pRflPacket->uiStartSecondsBytes,
									&pLastNode)))
		{
			goto Exit;
		}
		uiOffset += 4;

		if (RC_BAD( rc = rflPutNum( pPool, pLastNode, TRUE,
									RFL_START_MSEC_FIELD,
									pRflPacket->uiStartMicro,
									uiOffset,
									4, pRflPacket->uiStartMicroBytes,
									&pLastNode)))
		{
			goto Exit;
		}
		uiOffset += 4;

		// Output the end time and microseconds

		if (RC_BAD( rc = rflPutNum( pPool, pLastNode, TRUE,
									RFL_END_SECONDS_FIELD,
									pRflPacket->uiEndSeconds,
									uiOffset,
									4, pRflPacket->uiEndSecondsBytes,
									&pLastNode)))
		{
			goto Exit;
		}
		uiOffset += 4;

		if (RC_BAD( rc = rflPutNum( pPool, pLastNode, TRUE,
									RFL_END_MSEC_FIELD,
									pRflPacket->uiEndMicro,
									uiOffset,
									4, pRflPacket->uiEndMicroBytes,
									&pLastNode)))
		{
			goto Exit;
		}
		uiOffset += 4;
	}

Exit:

	if (RC_BAD( rc) || !pParent)
	{
		if (rc == FERR_IO_END_OF_FILE)
		{
			rc = FERR_OK;
		}
		*ppForest = NULL;
		pPool->poolReset( pvMark);
	}
	else
	{
		*ppForest = pParent;
	}
	return( rc);
}

/********************************************************************
Desc: Expands a reduce packet into multiple GEDCOM nodes
		for display of all of the subcomponents.
*********************************************************************/
FSTATIC RCODE rflExpandReducePacket(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	NODE **			ppForest
	)
{
	RCODE		rc = FERR_OK;
	void *	pvMark = pPool->poolMark();
	NODE *	pParent = NULL;
	NODE *	pLastNode;
	FLMUINT	uiOffset;

	// Output generic packet header information.

	if (RC_BAD( rc = rflExpandPacketHdr( pRflPacket, pPool, &pParent)))
	{
		goto Exit;
	}

	// Output transaction ID

	uiOffset = pRflPacket->uiFileOffset + RFL_PACKET_OVERHEAD;
	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
									RFL_TRANS_ID_FIELD,
									pRflPacket->uiTransID, uiOffset,
									4, pRflPacket->uiTransIDBytes, &pLastNode)))
	{
		goto Exit;
	}
	uiOffset += 4;

	// Output the count

	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
									RFL_BLOCK_COUNT_FIELD,
									pRflPacket->uiEndDrn, uiOffset,
									4, pRflPacket->uiEndDrnBytes, &pLastNode)))
	{
		goto Exit;
	}
	uiOffset += 4;

Exit:

	if (RC_BAD( rc) || !pParent)
	{
		if (rc == FERR_IO_END_OF_FILE)
		{
			rc = FERR_OK;
		}
		*ppForest = NULL;
		pPool->poolReset( pvMark);
	}
	else
	{
		*ppForest = pParent;
	}
	return( rc);
}

/********************************************************************
Desc: Expands an upgrade packet into multiple GEDCOM nodes
		for display of all of the subcomponents.
*********************************************************************/
FSTATIC RCODE rflExpandUpgradePacket(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	NODE **			ppForest
	)
{
	RCODE		rc = FERR_OK;
	void *	pvMark = pPool->poolMark();
	NODE *	pParent = NULL;
	NODE *	pLastNode;
	FLMUINT	uiOffset;

	// Output generic packet header information.

	if (RC_BAD( rc = rflExpandPacketHdr( pRflPacket, pPool, &pParent)))
	{
		goto Exit;
	}

	// Output transaction ID

	uiOffset = pRflPacket->uiFileOffset + RFL_PACKET_OVERHEAD;
	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
									RFL_TRANS_ID_FIELD,
									pRflPacket->uiTransID, uiOffset,
									4, pRflPacket->uiTransIDBytes, &pLastNode)))
	{
		goto Exit;
	}
	uiOffset += 4;

	// Output old DB version

	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
									RFL_OLD_DB_VERSION_FIELD,
									pRflPacket->uiDrn, uiOffset,
									4, pRflPacket->uiDrnBytes, &pLastNode)))
	{
		goto Exit;
	}
	uiOffset += 4;

	// Output new DB version

	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
									RFL_NEW_DB_VERSION_FIELD,
									pRflPacket->uiEndDrn, uiOffset,
									4, pRflPacket->uiEndDrnBytes, &pLastNode)))
	{
		goto Exit;
	}
	uiOffset += 4;

Exit:

	if (RC_BAD( rc) || !pParent)
	{
		if (rc == FERR_IO_END_OF_FILE)
		{
			rc = FERR_OK;
		}
		*ppForest = NULL;
		pPool->poolReset( pvMark);
	}
	else
	{
		*ppForest = pParent;
	}
	return( rc);
}

/********************************************************************
Desc: Expands an index suspend or resume packet
*********************************************************************/
FSTATIC RCODE rflExpandIndexStatePacket(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	NODE **			ppForest
	)
{
	RCODE		rc = FERR_OK;
	void *	pvMark = pPool->poolMark();
	NODE *	pParent = NULL;
	NODE *	pLastNode;
	FLMUINT	uiOffset;

	// Output generic packet header information.

	if (RC_BAD( rc = rflExpandPacketHdr( pRflPacket, pPool, &pParent)))
	{
		goto Exit;
	}

	// Output transaction ID

	uiOffset = pRflPacket->uiFileOffset + RFL_PACKET_OVERHEAD;
	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
									RFL_TRANS_ID_FIELD,
									pRflPacket->uiTransID, uiOffset,
									4, pRflPacket->uiTransIDBytes, &pLastNode)))
	{
		goto Exit;
	}
	uiOffset += 4;

	// Output old DB version

	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
									RFL_INDEX_NUM_FIELD,
									pRflPacket->uiIndex, uiOffset,
									2, pRflPacket->uiIndexBytes, &pLastNode)))
	{
		goto Exit;
	}
	uiOffset += 4;

Exit:

	if (RC_BAD( rc) || !pParent)
	{
		if (rc == FERR_IO_END_OF_FILE)
		{
			rc = FERR_OK;
		}
		*ppForest = NULL;
		pPool->poolReset( pvMark);
	}
	else
	{
		*ppForest = pParent;
	}
	return( rc);
}

/********************************************************************
Desc: Expand a data record packet.
*********************************************************************/
FSTATIC RCODE rflExpandDataPacket(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	FLMBOOL			bOutputPacket,
	FLMUINT			uiPacketType,
	NODE **			ppDataPacketNode,
	FLMUINT *		puiDataLen,
	FLMUINT *		puiLevel
	)
{
	RCODE			rc = FERR_OK;
	void *		pvMark = pPool->poolMark();
	NODE *		pDataPacketNode = NULL;
	FLMBYTE *	pucPacketBody;
	FLMUINT		uiBytesRead;
	FLMUINT		uiOffset;
	FLMUINT		uiTagNum;
	FLMUINT		uiTagNumLen;
	FLMUINT		uiDataType;
	FLMUINT		uiDataTypeLen;
	FLMUINT		uiLevel;
	FLMUINT		uiLevelLen;
	FLMUINT		uiDataLen;
	FLMUINT		uiDataLenLen;
	NODE *		pTmpNode;
	FLMUINT		uiLastLevel = 0;
	NODE *		pLastNode;
	FLMBYTE *	pucNodeData;
	FLMUINT		uiNodeDataLen;
	FLMUINT		uiEncrypted;
	FLMUINT		uiEncryptedLen;
	FLMUINT		uiEncLen;
	FLMUINT		uiEncLenLen;
	FLMUINT		uiEncDefID;
	FLMUINT		uiEncDefIDLen;
	NODE *		pDataNode;
	NODE *		pTagNode;
	NODE *		pRootNode = NULL;

	// Output the packet header.

	if (RC_BAD( rc = rflExpandPacketHdr( pRflPacket, pPool,
								&pDataPacketNode)))
	{
		goto Exit;
	}

	// If there is no packet body, we are done.

	if (!pRflPacket->uiPacketBodyLength)
	{
		goto Exit;
	}

	// Read the packet body from disk.

	pucPacketBody = &gv_rflBuffer [0];
	f_memset( pucPacketBody, 0, pRflPacket->uiPacketBodyLength);
	uiOffset = pRflPacket->uiFileOffset + RFL_PACKET_OVERHEAD;
	rc = gv_pRflFileHdl->read( uiOffset,
										 pRflPacket->uiPacketBodyLength,
										 pucPacketBody,
										 &uiBytesRead);
	if (RC_BAD( rc))
	{
		if (rc != FERR_IO_END_OF_FILE || !uiBytesRead)
		{
			if (rc == FERR_IO_END_OF_FILE)
			{
				rc = FERR_OK;
			}
			goto Exit;
		}
	}
	if (!uiBytesRead)
	{
		goto Exit;
	}

	pLastNode = pDataPacketNode;
	while (pLastNode->next)
	{
		pLastNode = pLastNode->next;
	}

	// Go through the packet body and create GEDCOM nodes.

	while (uiBytesRead)
	{
		if (*puiDataLen)
		{

			uiDataLen = *puiDataLen;
			uiDataType = FLM_BINARY_TYPE;
			uiLastLevel = 0;
			uiLevel = *puiLevel;

			// Create the necessary number of dummy parent nodes.

			if (uiLastLevel < uiLevel)
			{
				for (;;)
				{
					if ((pTmpNode = GedNodeCreate( pPool,
									makeTagNum( RFL_MORE_DATA_FIELD), 0, &rc)) == 0)
					{
						goto Exit;
					}
					if (!pRootNode)
					{
						GedSibGraft( pLastNode, pTmpNode, GED_LAST);
						pRootNode = pTmpNode;
					}
					else
					{
						GedChildGraft( pLastNode, pTmpNode, GED_LAST);
					}
					pLastNode = pTmpNode;
					if (uiLastLevel + 1 == uiLevel)
						break;
					uiLastLevel++;
				}
			}

			// Create a GEDCOM node.

			if ((pTagNode = GedNodeCreate( pPool, makeTagNum( RFL_MORE_DATA_FIELD),
										uiOffset, &rc)) == NULL)
			{
				goto Exit;
			}
			pDataNode = pTagNode;
		}

		// Remaining length better be at least two or we
		// have an incomplete packet - we need to at least
		// be able to get the tag number at this point.

		else if (uiBytesRead < 2)
		{
			if (RC_BAD( rc = rflPutNum( pPool, NULL, TRUE,
									RFL_TAG_NUM_FIELD,
									(FLMUINT)(*pucPacketBody), uiOffset,
									2, 1, &pTagNode)))
			{
				goto Exit;
			}
			pDataNode = NULL;

			// Reset the context in case there is another packet
			// following this one.

			uiLevel = uiLastLevel;
			*puiDataLen = uiDataLen = 0;
			uiBytesRead = 0;
		}
		else if (uiBytesRead == 2)
		{
			if ((uiTagNum = (FLMUINT)FB2UW( pucPacketBody)) != 0)
			{
				if (RC_BAD( rc = rflPutNum( pPool, NULL, TRUE,
											RFL_TAG_NUM_FIELD,
											uiTagNum, uiOffset,
											2, 2, &pTagNode)))
				{
					goto Exit;
				}
				pDataNode = NULL;
				*puiDataLen = uiDataLen = 0;
				uiLevel = uiLastLevel;
				uiBytesRead = 0;
			}
			else
			{

				// Reset the context in case there is another packet
				// following this one.

				*puiDataLen = uiDataLen = 0;
				goto Exit;
			}
		}
		else
		{
			FLMBOOL	bIncompleteHdr = FALSE;

			rflGetNumValue( pucPacketBody, uiBytesRead, 0, 2, &uiTagNum, &uiTagNumLen);
			pucPacketBody += uiTagNumLen;
			uiBytesRead -= uiTagNumLen;

			rflGetNumValue( pucPacketBody, uiBytesRead, 0, 1, &uiDataType, &uiDataTypeLen);
			pucPacketBody += uiDataTypeLen;
			uiBytesRead -= uiDataTypeLen;
			
			rflGetNumValue( pucPacketBody, uiBytesRead, 0, 1, &uiLevel, &uiLevelLen);
			pucPacketBody += uiLevelLen;
			uiBytesRead -= uiLevelLen;

			if (uiPacketType == RFL_DATA_RECORD_PACKET_VER_3)
			{
				rflGetNumValue( pucPacketBody, uiBytesRead, 0, 4, &uiDataLen, &uiDataLenLen);
				if (uiDataLenLen < 4)
				{
					bIncompleteHdr = TRUE;
				}
			}
			else
			{
				rflGetNumValue( pucPacketBody, uiBytesRead, 0, 2, &uiDataLen, &uiDataLenLen);
				if (uiDataLenLen < 2)
				{
					bIncompleteHdr = TRUE;
				}
			}
			pucPacketBody += uiDataLenLen;
			uiBytesRead -= uiDataLenLen;


			// If we have an encrypted packet, handle the remaining fields here

			uiEncrypted = 0;
			uiEncryptedLen = 0;
			uiEncDefID = 0;
			uiEncDefIDLen = 0;
			uiEncLen = 0;
			uiEncLenLen = 0;
			if (uiPacketType == RFL_ENC_DATA_RECORD_PACKET ||
				 uiPacketType == RFL_DATA_RECORD_PACKET_VER_3)
			{
				rflGetNumValue( pucPacketBody, uiBytesRead, 0, 1, &uiEncrypted, &uiEncryptedLen);
				pucPacketBody += uiEncryptedLen;
				uiBytesRead -= uiEncryptedLen;

				if (!uiEncryptedLen)
				{
					bIncompleteHdr = TRUE;
				}
				else if (uiEncrypted)
				{
					rflGetNumValue( pucPacketBody, uiBytesRead, 0, 2, &uiEncDefID, &uiEncDefIDLen);
					pucPacketBody += uiEncDefIDLen;
					uiBytesRead -= uiEncDefIDLen;

					if (uiPacketType == RFL_DATA_RECORD_PACKET_VER_3)
					{
						rflGetNumValue( pucPacketBody, uiBytesRead, 0, 4, &uiEncLen, &uiEncLenLen);
						if (uiEncLenLen < 4)
						{
							bIncompleteHdr = TRUE;
						}
					}
					else
					{
						rflGetNumValue( pucPacketBody, uiBytesRead, 0, 2, &uiEncLen, &uiEncLenLen);
						if (uiEncLenLen < 2)
						{
							bIncompleteHdr = TRUE;
						}
					}
					pucPacketBody += uiEncLenLen;
					uiBytesRead -= uiEncLenLen;
				}
			}

			// If we have incomplete or bad header information,
			// output each piece of header info individually, subordinate
			// to a "tag number" field.  Then output the
			// data there too.

			if (bIncompleteHdr || !uiTagNum ||
				 (uiDataType != FLM_TEXT_TYPE &&
				  uiDataType != FLM_NUMBER_TYPE &&
				  uiDataType != FLM_BINARY_TYPE &&
				  uiDataType != FLM_CONTEXT_TYPE &&
				  uiDataType != FLM_BLOB_TYPE) ||
				 (!pRootNode && uiLevel) ||
				 (pRootNode && uiLevel > uiLastLevel + 1))
			{

				if (RC_BAD( rc = rflPutNum( pPool, NULL, TRUE, RFL_TAG_NUM_FIELD,
					uiTagNum, uiOffset, 2, uiTagNumLen, &pTagNode)))
				{
					goto Exit;
				}
				uiOffset += uiTagNumLen;

				if ( RC_BAD( rc = rflPutNum( pPool, pTagNode, FALSE, RFL_TYPE_FIELD,
					uiDataType, uiOffset, 1, uiDataTypeLen, &pTmpNode)))
				{
					goto Exit;
				}
				uiOffset += uiDataTypeLen;

				if ( RC_BAD( rc = rflPutNum( pPool, pTmpNode, TRUE, RFL_LEVEL_FIELD,
					uiLevel, uiOffset, 1, uiLevelLen, &pTmpNode)))
				{
					goto Exit;
				}
				uiOffset += uiLevelLen;

				if (uiPacketType == RFL_DATA_RECORD_PACKET_VER_3)
				{
					if ( RC_BAD( rc = rflPutNum( pPool, pTmpNode, TRUE, RFL_DATA_LEN_FIELD,
						uiDataLen, uiOffset, 4, uiDataLenLen, &pTmpNode)))
					{
						goto Exit;
					}
				}
				else
				{
					if ( RC_BAD( rc = rflPutNum( pPool, pTmpNode, TRUE, RFL_DATA_LEN_FIELD,
						uiDataLen, uiOffset, 2, uiDataLenLen, &pTmpNode)))
					{
						goto Exit;
					}
				}
				uiOffset += uiDataLenLen;

				if (uiPacketType == RFL_ENC_DATA_RECORD_PACKET ||
					 uiPacketType == RFL_DATA_RECORD_PACKET_VER_3)
				{
					if ( RC_BAD( rc = rflPutNum( pPool, pTmpNode, TRUE, RFL_ENC_FIELD,
						uiEncrypted, uiOffset, 1, uiEncryptedLen, &pTmpNode)))
					{
						goto Exit;
					}
					uiOffset += uiEncryptedLen;

					if (uiEncryptedLen && uiEncrypted)
					{
						if ( RC_BAD( rc = rflPutNum( pPool, pTmpNode, TRUE, RFL_ENC_DEF_ID_FIELD,
							uiEncDefID, uiOffset, 2, uiEncDefIDLen, &pTmpNode)))
						{
							goto Exit;
						}
						uiOffset += uiEncDefIDLen;

						if (uiPacketType == RFL_DATA_RECORD_PACKET_VER_3)
						{
							if ( RC_BAD( rc = rflPutNum( pPool, pTmpNode, TRUE, RFL_ENC_DATA_LEN_FIELD,
								uiEncLen, uiOffset, 4, uiEncLenLen, &pTmpNode)))
							{
								goto Exit;
							}
						}
						else
						{
							if ( RC_BAD( rc = rflPutNum( pPool, pTmpNode, TRUE, RFL_ENC_DATA_LEN_FIELD,
								uiEncLen, uiOffset, 2, uiEncLenLen, &pTmpNode)))
							{
								goto Exit;
							}
						}
						uiOffset += uiEncLenLen;

						if (uiEncLenLen == 2)
						{
							uiDataLen = uiEncLen;
						}
					}
				}

				if (uiDataLen && !bIncompleteHdr)
				{

					// Create a GEDCOM node for the data.

					if ((pDataNode = GedNodeCreate( pPool, makeTagNum( RFL_DATA_FIELD),
									uiOffset, &rc)) == NULL)
					{
						goto Exit;
					}
					uiDataType = FLM_BINARY_TYPE;
					GedSibGraft( pTmpNode, pDataNode, GED_LAST);
				}
				else
				{
					uiDataLen = 0;
					pDataNode = NULL;
				}

				// If we didn't get a good field, keep things at their current
				// level.

				uiLevel = uiLastLevel;
			}
			else
			{

				// Create a GEDCOM node.

				if ((pTagNode = GedNodeCreate( pPool, uiTagNum, uiOffset, &rc)) == NULL)
				{
					goto Exit;
				}
				uiOffset += 6;
				if (uiPacketType == RFL_ENC_DATA_RECORD_PACKET)
				{
					uiOffset++;
					if (uiEncrypted)
					{
						uiOffset += 4;
						uiDataLen = uiEncLen;
					}
				}
				if (uiDataLen)
				{
					pDataNode = pTagNode;
				}
				else
				{
					pDataNode = NULL;
				}
			}
		}

		// Graft the tag node relative to the last node, if any.
		// Otherwise, it becomes the root node.

		if (!pRootNode)
		{
			GedSibGraft( pLastNode, pTagNode, GED_LAST);
			pRootNode = pTagNode;
			uiLastLevel = 0;
		}
		else if (uiLevel > uiLastLevel)
		{
			GedChildGraft( pLastNode, pTagNode, GED_LAST);
			uiLastLevel = uiLevel;
		}
		else
		{
			while (uiLevel < uiLastLevel)
			{
				pLastNode = GedParent( pLastNode);
				uiLastLevel--;
			}
			GedSibGraft( pLastNode, pTagNode, GED_LAST);
			uiLastLevel = uiLevel;
		}
		pLastNode = pTagNode;

		// Allocate space for the data.  We call this even if uiDataLen is
		// zero so that the appropriate data type will be set in the node
		// as well.

		if (pDataNode)
		{
			uiNodeDataLen = (FLMUINT)((uiDataLen <= uiBytesRead)
											 ? uiDataLen
											 : uiBytesRead);
			if (((pucNodeData = (FLMBYTE *)GedAllocSpace( pPool, pDataNode,
														uiDataType,
														uiNodeDataLen)) == NULL) &&
				 (uiNodeDataLen))
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}
			if (uiNodeDataLen)
			{
				f_memcpy( pucNodeData, pucPacketBody, uiNodeDataLen);
				pucPacketBody += uiNodeDataLen;
				uiOffset += uiNodeDataLen;
				uiDataLen -= uiNodeDataLen;
				uiBytesRead -= uiNodeDataLen;
			}
		}

		*puiLevel = uiLastLevel;
		if ((*puiDataLen = uiDataLen) != 0)
		{
			break;
		}
	}

Exit:
	if (RC_BAD( rc) || !pDataPacketNode || !bOutputPacket)
	{
		if (rc == FERR_IO_END_OF_FILE)
		{
			rc = FERR_OK;
		}
		pPool->poolReset( pvMark);
	}
	else
	{
		if ((*ppDataPacketNode) && (pDataPacketNode))
		{
			GedSibGraft( *ppDataPacketNode, pDataPacketNode, GED_LAST);
		}
		if (pDataPacketNode)
		{
			*ppDataPacketNode = pDataPacketNode;
		}
	}
	return( rc);
}
										
/********************************************************************
Desc: Expand a change fields packet.
*********************************************************************/
FSTATIC RCODE rflExpandChangeFieldsPacket(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	FLMBOOL			bOutputPacket,
	NODE **			ppChangeFieldsPacketNode,
	FLMUINT *		puiDataLen
	)
{
	RCODE			rc = FERR_OK;
	void *		pvMark = pPool->poolMark();
	NODE *		pChangeFieldsPacketNode = NULL;
	NODE *		pChangeNode;
	NODE *		pLastNode;
	FLMBYTE *	pucPacketBody;
	FLMUINT		uiBytesRead;
	FLMUINT		uiOffset;
	eDispTag		eChangeTagNum;
	FLMUINT		uiDataTagNum;
	FLMUINT		uiDataLen;
	FLMUINT		uiChangeType;
	FLMBYTE *	pucNodeData;
	NODE *		pDataNode;
	FLMUINT		uiNodeDataLen;
	FLMUINT		uiTmp;
	FLMUINT		uiLen;
	FLMUINT		uiDataLenLen;

	// Output the packet header.

	if (RC_BAD( rc = rflExpandPacketHdr( pRflPacket, pPool,
								&pChangeFieldsPacketNode)))
	{
		goto Exit;
	}

	// If there is no packet body, we are done.

	if (!pRflPacket->uiPacketBodyLength)
	{
		goto Exit;
	}

	// Read the packet body from disk.

	pucPacketBody = &gv_rflBuffer [0];
	f_memset( pucPacketBody, 0, pRflPacket->uiPacketBodyLength);
	uiOffset = pRflPacket->uiFileOffset + RFL_PACKET_OVERHEAD;
	rc = gv_pRflFileHdl->read( uiOffset,
										 pRflPacket->uiPacketBodyLength,
										 pucPacketBody,
										 &uiBytesRead);
	if (RC_BAD( rc))
	{
		if (rc != FERR_IO_END_OF_FILE || !uiBytesRead)
		{
			if (rc == FERR_IO_END_OF_FILE)
			{
				rc = FERR_OK;
			}
			goto Exit;
		}
	}
	if (!uiBytesRead)
	{
		goto Exit;
	}

	pLastNode = pChangeFieldsPacketNode;
	while (pLastNode->next)
	{
		pLastNode = pLastNode->next;
	}

	// Go through the packet body and create GEDCOM nodes.

	while (uiBytesRead)
	{
		if (*puiDataLen)
		{
			uiDataTagNum = RFL_MORE_DATA_FIELD;
			uiDataLen = *puiDataLen;

			// Create a dummy change node.

			if ((pChangeNode = GedNodeCreate( pPool, makeTagNum( RFL_MORE_DATA_FIELD),
											0, &rc)) == 0)
			{
				goto Exit;
			}
			GedSibGraft( pLastNode, pChangeNode, GED_LAST);
		}
		else
		{
			uiChangeType = *pucPacketBody++;
			switch (uiChangeType)
			{
				case RFL_INSERT_FIELD:
					eChangeTagNum = RFL_INSERT_FLD_FIELD;
					break;
				case RFL_MODIFY_FIELD:
					eChangeTagNum = RFL_MODIFY_FLD_FIELD;
					break;
				case RFL_DELETE_FIELD:
					eChangeTagNum = RFL_DELETE_FLD_FIELD;
					break;
				case RFL_END_FIELD_CHANGES:
					eChangeTagNum = RFL_END_CHANGES_FIELD;
					break;
				// Added
				case RFL_INSERT_ENC_FIELD:
					eChangeTagNum = RFL_INSERT_ENC_FLD_FIELD;
					break;
				case RFL_MODIFY_ENC_FIELD:
					eChangeTagNum = RFL_MODIFY_ENC_FLD_FIELD;
					break;
				case RFL_INSERT_LARGE_FIELD:
					eChangeTagNum = RFL_INSERT_LARGE_FLD_FIELD;
					break;
				case RFL_INSERT_ENC_LARGE_FIELD:
					eChangeTagNum = RFL_INSERT_ENC_LARGE_FLD_FIELD;
					break;
				case RFL_MODIFY_LARGE_FIELD:
					eChangeTagNum = RFL_MODIFY_LARGE_FLD_FIELD;
					break;
				case RFL_MODIFY_ENC_LARGE_FIELD:
					eChangeTagNum = RFL_MODIFY_ENC_LARGE_FLD_FIELD;
					break;
				default:
					eChangeTagNum = RFL_UNKNOWN_CHANGE_TYPE_FIELD;
					break;
			}
			if (RC_BAD( rc = rflPutNum( pPool, pLastNode, TRUE,
									eChangeTagNum,
									uiChangeType, uiOffset,
									1, 1, &pChangeNode)))
			{
				goto Exit;
			}
			uiOffset++;
			uiBytesRead--;

			// Output the position field.

			rflGetNumValue( pucPacketBody, uiBytesRead, 0,
										2, &uiTmp, &uiLen);

			if (RC_BAD( rc = rflPutNum( pPool, pChangeNode, FALSE,
									RFL_POSITION_FIELD,
									uiTmp, uiOffset,
									2, uiLen, NULL)))
			{
				goto Exit;
			}
			uiOffset += uiLen;
			pucPacketBody += uiLen;
			uiBytesRead -= uiLen;

			uiDataLen = 0;
			if (uiChangeType == RFL_INSERT_FIELD ||
			 	 uiChangeType == RFL_INSERT_ENC_FIELD ||
				 uiChangeType == RFL_INSERT_LARGE_FIELD ||
				 uiChangeType == RFL_INSERT_ENC_LARGE_FIELD)
			{

				// Output tag number.

				rflGetNumValue( pucPacketBody, uiBytesRead, 0,
										2, &uiTmp, &uiLen);
				if (RC_BAD( rc = rflPutNum( pPool, pChangeNode, FALSE,
										RFL_TAG_NUM_FIELD,
										uiTmp, uiOffset,
										2, uiLen, NULL)))
				{
					goto Exit;
				}
				uiOffset += uiLen;
				pucPacketBody += uiLen;
				uiBytesRead -= uiLen;

				// Output data type

				rflGetNumValue( pucPacketBody, uiBytesRead, 0,
										1, &uiTmp, &uiLen);
				if (RC_BAD( rc = rflPutNum( pPool, pChangeNode, FALSE,
										RFL_TYPE_FIELD,
										uiTmp, uiOffset,
										1, uiLen, NULL)))
				{
					goto Exit;
				}
				uiOffset += uiLen;
				pucPacketBody += uiLen;
				uiBytesRead -= uiLen;

				// Output level

				rflGetNumValue( pucPacketBody, uiBytesRead, 0,
										1, &uiTmp, &uiLen);
				if (RC_BAD( rc = rflPutNum( pPool, pChangeNode, FALSE,
										RFL_LEVEL_FIELD,
										uiTmp, uiOffset,
										1, uiLen, NULL)))
				{
					goto Exit;
				}
				uiOffset += uiLen;
				pucPacketBody += uiLen;
				uiBytesRead -= uiLen;

				// Output data length

				uiDataLenLen = (uiChangeType == RFL_INSERT_LARGE_FIELD ||
									 uiChangeType == RFL_INSERT_ENC_LARGE_FIELD)
									? 4
									: 2;
				rflGetNumValue( pucPacketBody, uiBytesRead, 0,
										uiDataLenLen, &uiTmp, &uiLen);
				if (RC_BAD( rc = rflPutNum( pPool, pChangeNode, FALSE,
										RFL_DATA_LEN_FIELD,
										uiTmp, uiOffset,
										uiDataLenLen, uiLen, NULL)))
				{
					goto Exit;
				}
				uiOffset += uiLen;
				pucPacketBody += uiLen;
				uiBytesRead -= uiLen;
				if (uiLen == 2)
				{
					uiDataLen = uiTmp;
				}

				uiDataTagNum = RFL_DATA_FIELD;
				if (uiChangeType == RFL_INSERT_ENC_FIELD ||
					 uiChangeType == RFL_INSERT_ENC_LARGE_FIELD)
				{
					// Output the encryption definition id
					rflGetNumValue( pucPacketBody, uiBytesRead, 0, 2, &uiTmp, &uiLen);
					if ( RC_BAD( rc = rflPutNum( pPool, pChangeNode, FALSE, RFL_ENC_DEF_ID_FIELD,
						uiTmp, uiOffset, 2, uiLen, NULL)))
					{
						goto Exit;
					}
					uiOffset += uiLen;
					pucPacketBody += uiLen;
					uiBytesRead -= uiLen;

					// Output the encrypted data length
					rflGetNumValue( pucPacketBody, uiBytesRead, 0, uiDataLenLen, &uiTmp, &uiLen);
					if ( RC_BAD( rc = rflPutNum( pPool, pChangeNode, FALSE, RFL_ENC_DATA_LEN_FIELD,
						uiTmp, uiOffset, uiDataLenLen, uiLen, NULL)))
					{
						goto Exit;
					}
					uiOffset += uiLen;
					pucPacketBody += uiLen;
					uiBytesRead -= uiLen;
					if (uiLen == 2)
					{
						uiDataLen = uiTmp;
					}
				}
			}
			else if (uiChangeType == RFL_MODIFY_FIELD ||
						uiChangeType == RFL_MODIFY_ENC_FIELD ||
						uiChangeType == RFL_MODIFY_LARGE_FIELD ||
						uiChangeType == RFL_MODIFY_ENC_LARGE_FIELD)
			{

				// Output change bytes type

				rflGetNumValue( pucPacketBody, uiBytesRead, 0,
										1, &uiTmp, &uiLen);
				if ((uiTmp == RFL_REPLACE_BYTES) && (uiLen))
				{
					if (RC_BAD( rc = rflPutNum( pPool, pChangeNode, FALSE,
											RFL_REPLACE_BYTES_FIELD,
											uiTmp, uiOffset,
											1, uiLen, NULL)))
					{
						goto Exit;
					}
				}
				else
				{
					if (RC_BAD( rc = rflPutNum( pPool, pChangeNode, FALSE,
											RFL_UNKNOWN_CHANGE_BYTES_FIELD,
											uiTmp, uiOffset,
											1, uiLen, NULL)))
					{
						goto Exit;
					}
				}
				uiOffset += uiLen;
				pucPacketBody += uiLen;
				uiBytesRead -= uiLen;

				// Output data length

				uiDataLenLen = (uiChangeType == RFL_MODIFY_LARGE_FIELD ||
									 uiChangeType == RFL_MODIFY_ENC_LARGE_FIELD)
									? 4
									: 2;
				rflGetNumValue( pucPacketBody, uiBytesRead, 0,
										uiDataLenLen, &uiTmp, &uiLen);
				if (RC_BAD( rc = rflPutNum( pPool, pChangeNode, FALSE,
										RFL_DATA_LEN_FIELD,
										uiTmp, uiOffset,
										uiDataLenLen, uiLen, NULL)))
				{
					goto Exit;
				}
				uiOffset += uiLen;
				pucPacketBody += uiLen;
				uiBytesRead -= uiLen;
				if (uiLen == uiDataLenLen)
				{
					uiDataLen = uiTmp;
				}
				uiDataTagNum = RFL_DATA_FIELD;
				if (uiChangeType == RFL_MODIFY_ENC_FIELD ||
					 uiChangeType == RFL_MODIFY_ENC_LARGE_FIELD)
				{
					// Output the encryption definition id
					rflGetNumValue( pucPacketBody, uiBytesRead, 0, 2, &uiTmp, &uiLen);
					if ( RC_BAD( rc = rflPutNum( pPool, pChangeNode, FALSE, RFL_ENC_DEF_ID_FIELD,
						uiTmp, uiOffset, 2, uiLen, NULL)))
					{
						goto Exit;
					}
					uiOffset += uiLen;
					pucPacketBody += uiLen;
					uiBytesRead -= uiLen;

					// Output the encrypted data length
					rflGetNumValue( pucPacketBody, uiBytesRead, 0, uiDataLenLen, &uiTmp, &uiLen);
					if ( RC_BAD( rc = rflPutNum( pPool, pChangeNode, FALSE, RFL_ENC_DATA_LEN_FIELD,
						uiTmp, uiOffset, uiDataLenLen, uiLen, NULL)))
					{
						goto Exit;
					}
					uiOffset += uiLen;
					pucPacketBody += uiLen;
					uiBytesRead -= uiLen;
					if (uiLen == uiDataLenLen)
					{
						uiDataLen = uiTmp;
					}
				}
			}
		}

		// Create a data node, if there is data to output.

		if (uiDataLen && uiBytesRead)
		{
			// Create a GEDCOM node.

			if ((pDataNode = GedNodeCreate( pPool, uiDataTagNum,
										uiOffset, &rc)) == NULL)
			{
				goto Exit;
			}
			GedChildGraft( pChangeNode, pDataNode, GED_LAST);
			uiNodeDataLen = (FLMUINT)((uiDataLen <= uiBytesRead)
											 ? uiDataLen
											 : uiBytesRead);
			if (((pucNodeData = (FLMBYTE *)GedAllocSpace( pPool, pDataNode,
														FLM_BINARY_TYPE,
														uiNodeDataLen)) == NULL) &&
				 (uiNodeDataLen))
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}
			if (uiNodeDataLen)
			{
				f_memcpy( pucNodeData, pucPacketBody, uiNodeDataLen);
				pucPacketBody += uiNodeDataLen;
				uiOffset += uiNodeDataLen;
				uiDataLen -= uiNodeDataLen;
				uiBytesRead -= uiNodeDataLen;
			}
		}

		pLastNode = pChangeNode;
		if ((*puiDataLen = uiDataLen) != 0)
		{
			break;
		}
	}

Exit:
	if (RC_BAD( rc) || !pChangeFieldsPacketNode || !bOutputPacket)
	{
		if (rc == FERR_IO_END_OF_FILE)
		{
			rc = FERR_OK;
		}
		pPool->poolReset( pvMark);
	}
	else
	{
		if ((*ppChangeFieldsPacketNode) && (pChangeFieldsPacketNode))
		{
			GedSibGraft( *ppChangeFieldsPacketNode,
							pChangeFieldsPacketNode, GED_LAST);
		}
		if (pChangeFieldsPacketNode)
		{
			*ppChangeFieldsPacketNode = pChangeFieldsPacketNode;
		}
	}
	return( rc);
}
										
/********************************************************************
Desc: Expand the change field packets for a record
		modify operation into the appropriate GEDCOM nodes.
*********************************************************************/
FSTATIC RCODE rflExpandRecordPackets(
	F_Pool *		pPool,
	FLMUINT		uiOffset,
	FLMUINT		uiPacketType,
	NODE **		ppLastPacketNode,
	FLMUINT		uiPacketOffset
	)
{
	RCODE			rc = FERR_OK;
	RFL_PACKET	RflPacket;
	FLMUINT		uiDataLen = 0;
	FLMUINT		uiLevel = 0;
	FLMBOOL		bOutputPacket;
	NODE *		pLastPacketNode = NULL;

	if( *ppLastPacketNode)
	{
		pLastPacketNode = *ppLastPacketNode;
	}

	for (;;)
	{

		// Quit when there are no more packets.

		if (!uiOffset)
		{
			break;
		}

		// Retrieve the next packet.

		if (RC_BAD( rc = rflRetrievePacket( 0, uiOffset, &RflPacket)))
		{
			goto Exit;
		}

		if (uiPacketType == 0xFF)
		{
			if (RflPacket.uiPacketType == RFL_DATA_RECORD_PACKET ||
				 RflPacket.uiPacketType == RFL_ENC_DATA_RECORD_PACKET ||
				 RflPacket.uiPacketType == RFL_DATA_RECORD_PACKET_VER_3 ||
				 RflPacket.uiPacketType == RFL_CHANGE_FIELDS_PACKET)
			{
				uiPacketType = RflPacket.uiPacketType;
			}
			else
			{
				break;
			}
		}

		// Stop when we don't have a valid data packet.

		if (!RflPacket.bValidPacketType ||
			 RflPacket.uiPacketType != uiPacketType)
		{
			break;
		}

		bOutputPacket = (FLMBOOL)(((uiPacketOffset == 0) ||
										 (RflPacket.uiFileOffset >= uiPacketOffset))
										? (FLMBOOL)TRUE
										: (FLMBOOL)FALSE);

		if (uiPacketType == RFL_DATA_RECORD_PACKET ||
			 uiPacketType == RFL_ENC_DATA_RECORD_PACKET ||
			 uiPacketType == RFL_DATA_RECORD_PACKET_VER_3)
		{
			if (RC_BAD( rc = rflExpandDataPacket( &RflPacket, pPool,
								bOutputPacket, uiPacketType,
								&pLastPacketNode, &uiDataLen, &uiLevel)))
			{
				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = rflExpandChangeFieldsPacket( &RflPacket, pPool,
											bOutputPacket,
											&pLastPacketNode, &uiDataLen)))
			{
				goto Exit;
			}
		}

		if( !(*ppLastPacketNode))
		{
			*ppLastPacketNode = pLastPacketNode;
		}

		// If we are trying to output only a single packet, stop once we
		// have output it.

		if ((bOutputPacket) && (uiPacketOffset))
		{
			break;
		}

		// Set the address for the next packet.

		uiOffset = RflPacket.uiNextPacketAddress;
	}
Exit:
	return( rc);
}

/********************************************************************
Desc: Expands a record operation packet into multiple GEDCOM nodes
		for display of all of the subcomponents.
*********************************************************************/
FSTATIC RCODE rflExpandRecOpPacket(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	NODE **			ppForest,
	FLMUINT			uiPacketOffset
	)
{
	RCODE		rc = FERR_OK;
	void *	pvMark = pPool->poolMark();
	NODE *	pParent = NULL;
	NODE *	pLastNode;
	FLMUINT	uiOffset;

	if ((!uiPacketOffset) || (uiPacketOffset == pRflPacket->uiFileOffset))
	{

		// Output generic packet header information.

		if (RC_BAD( rc = rflExpandPacketHdr( pRflPacket, pPool, &pParent)))
		{
			goto Exit;
		}

		// Output transaction ID

		uiOffset = pRflPacket->uiFileOffset + RFL_PACKET_OVERHEAD;
		if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
										RFL_TRANS_ID_FIELD,
										pRflPacket->uiTransID, uiOffset,
										4, pRflPacket->uiTransIDBytes,
										&pLastNode)))
		{
			goto Exit;
		}
		uiOffset += 4;

		// Output the container

		if (RC_BAD( rc = rflPutNum( pPool, pLastNode, TRUE,
									RFL_CONTAINER_FIELD,
									pRflPacket->uiContainer,
									uiOffset,
									2, pRflPacket->uiContainerBytes,
									&pLastNode)))
		{
			goto Exit;
		}
		uiOffset += 2;

		// Output the DRN

		if (RC_BAD( rc = rflPutNum( pPool, pLastNode, TRUE,
									RFL_DRN_FIELD,
									pRflPacket->uiDrn,
									uiOffset,
									4, pRflPacket->uiDrnBytes,
									&pLastNode)))
		{
			goto Exit;
		}
		uiOffset += 4;

		// Output flags

		if( pRflPacket->uiPacketType == RFL_ADD_RECORD_PACKET_VER_2 ||
			pRflPacket->uiPacketType == RFL_MODIFY_RECORD_PACKET_VER_2 ||
			pRflPacket->uiPacketType == RFL_DELETE_RECORD_PACKET_VER_2)
		{
			if (RC_BAD( rc = rflPutNum( pPool, pLastNode, TRUE,
										RFL_FLAGS_FIELD,
										pRflPacket->uiFlags,
										uiOffset,
										4, pRflPacket->uiFlagsBytes,
										&pLastNode)))
			{
				goto Exit;
			}
			uiOffset += 4;
		}

		if (pRflPacket->bHaveTimes)
		{

			// Output the start time and microseconds

			if (RC_BAD( rc = rflPutNum( pPool, pLastNode, TRUE,
										RFL_START_SECONDS_FIELD,
										pRflPacket->uiStartSeconds,
										uiOffset,
										4, pRflPacket->uiStartSecondsBytes,
										&pLastNode)))
			{
				goto Exit;
			}
			uiOffset += 4;

			if (RC_BAD( rc = rflPutNum( pPool, pLastNode, TRUE,
										RFL_START_MSEC_FIELD,
										pRflPacket->uiStartMicro,
										uiOffset,
										4, pRflPacket->uiStartMicroBytes,
										&pLastNode)))
			{
				goto Exit;
			}
			uiOffset += 4;

			// Output the end time and microseconds

			if (RC_BAD( rc = rflPutNum( pPool, pLastNode, TRUE,
										RFL_END_SECONDS_FIELD,
										pRflPacket->uiEndSeconds,
										uiOffset,
										4, pRflPacket->uiEndSecondsBytes,
										&pLastNode)))
			{
				goto Exit;
			}
			uiOffset += 4;

			if (RC_BAD( rc = rflPutNum( pPool, pLastNode, TRUE,
										RFL_END_MSEC_FIELD,
										pRflPacket->uiEndMicro,
										uiOffset,
										4, pRflPacket->uiEndMicroBytes,
										&pLastNode)))
			{
				goto Exit;
			}
			uiOffset += 4;
		}
		uiPacketOffset = 0;
	}

	// Output stuff for add record and modify record

	if (pRflPacket->uiPacketType == RFL_ADD_RECORD_PACKET ||
		pRflPacket->uiPacketType == RFL_ADD_RECORD_PACKET_VER_2)
	{
		if (RC_BAD( rc = rflExpandRecordPackets( pPool,
										pRflPacket->uiNextPacketAddress,
										0xFF,
										&pParent, uiPacketOffset)))
		{
			goto Exit;
		}
	}
	else if (pRflPacket->uiPacketType == RFL_MODIFY_RECORD_PACKET ||
		pRflPacket->uiPacketType == RFL_MODIFY_RECORD_PACKET_VER_2)
	{
		if (RC_BAD( rc = rflExpandRecordPackets( pPool,
										pRflPacket->uiNextPacketAddress,
										0xFF, &pParent, uiPacketOffset)))
		{
			goto Exit;
		}
	}

Exit:
	if (RC_BAD( rc) || !pParent)
	{
		if (rc == FERR_IO_END_OF_FILE)
		{
			rc = FERR_OK;
		}
		*ppForest = NULL;
		pPool->poolReset( pvMark);
	}
	else
	{
		*ppForest = pParent;
	}
	return( rc);
}

/********************************************************************
Desc: Expand an unknown packet.
*********************************************************************/
FSTATIC RCODE rflExpandUnkPacket(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	NODE **			ppForest
	)
{
	RCODE			rc = FERR_OK;
	void *		pvMark = pPool->poolMark();
	NODE *		pForest = NULL;
	FLMBYTE *	pucPacketBody;
	FLMUINT		uiBytesRead;
	FLMUINT		uiOffset;
	NODE *		pTmpNode2;

	// Output the packet header.

	if (RC_BAD( rc = rflExpandPacketHdr( pRflPacket, pPool,
								&pForest)))
	{
		goto Exit;
	}

	// If there is no packet body, we are done.

	if (!pRflPacket->uiPacketBodyLength)
	{
		goto Exit;
	}

	// Read the packet body from disk.

	pucPacketBody = &gv_rflBuffer [0];
	f_memset( pucPacketBody, 0, pRflPacket->uiPacketBodyLength);
	uiOffset = pRflPacket->uiFileOffset + RFL_PACKET_OVERHEAD;
	rc = gv_pRflFileHdl->read( uiOffset,
										 pRflPacket->uiPacketBodyLength,
										 pucPacketBody,
										 &uiBytesRead);
	if (RC_BAD( rc))
	{
		if (rc != FERR_IO_END_OF_FILE || !uiBytesRead)
		{
			if (rc == FERR_IO_END_OF_FILE)
			{
				rc = FERR_OK;
			}
			goto Exit;
		}
	}
	if (!uiBytesRead)
	{
		goto Exit;
	}

	// Create a GEDCOM node for the data.

	if ((pTmpNode2 = GedNodeCreate( pPool, makeTagNum( RFL_DATA_FIELD),
					uiOffset, &rc)) == NULL)
	{
		goto Exit;
	}
	if (RC_BAD( rc = GedPutBINARY( pPool, pTmpNode2, pucPacketBody,
								uiBytesRead)))
	{
		goto Exit;
	}
	GedChildGraft( pForest, pTmpNode2, GED_LAST);

Exit:
	if (RC_BAD( rc) || !pForest)
	{
		if (rc == FERR_IO_END_OF_FILE)
		{
			rc = FERR_OK;
		}
		pPool->poolReset( pvMark);
		*ppForest = NULL;
	}
	else
	{
		*ppForest = pForest;
	}
	return( rc);
}

/********************************************************************
Desc: Expands an encryption packet (RFL_WRAP_KEY_PACKET or
RFL_ENABLE_ENCRYPTION_PACKET)
*********************************************************************/
FSTATIC RCODE rflExpandEncryptionPacket(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	NODE **			ppForest
	)
{
	RCODE			rc = FERR_OK;
	void *		pvMark = pPool->poolMark();
	NODE *		pParent = NULL;
	NODE *		pLastNode;
	NODE *		pDataNode;
	FLMUINT		uiOffset;
	FLMBYTE *	pucNodeData;
	FLMBYTE *	pucPacketBody;

	// Output generic packet header information.

	if (RC_BAD( rc = rflExpandPacketHdr( pRflPacket, pPool, &pParent)))
	{
		goto Exit;
	}

	// Output transaction ID

	uiOffset = pRflPacket->uiFileOffset + RFL_PACKET_OVERHEAD;
	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
									RFL_TRANS_ID_FIELD,
									pRflPacket->uiTransID, uiOffset,
									4, pRflPacket->uiTransIDBytes, &pLastNode)))
	{
		goto Exit;
	}
	uiOffset += 4;

	// Output key len

	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
									RFL_DB_KEY_LEN_FIELD,
									pRflPacket->uiCount, uiOffset,
									2, pRflPacket->uiCountBytes, &pLastNode)))
	{
		goto Exit;
	}
	uiOffset += 2;

	// Create a data node, if there is data to output.

	if ((pDataNode = GedNodeCreate( pPool, makeTagNum( RFL_DATA_FIELD),
								uiOffset, &rc)) == NULL)
	{
		goto Exit;
	}
	GedChildGraft( pParent, pDataNode, GED_LAST);
	if (((pucNodeData = (FLMBYTE *)GedAllocSpace( pPool, pDataNode,
												FLM_BINARY_TYPE,
												pRflPacket->uiCount)) == NULL) &&
			(pRflPacket->uiCount))
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	pucPacketBody = &gv_rflBuffer [0];
	rc = gv_pRflFileHdl->read( uiOffset,
										 pRflPacket->uiCount,
										 pucPacketBody,
										 NULL);

	if (pRflPacket->uiCount)
	{
		f_memcpy( pucNodeData, pucPacketBody, pRflPacket->uiCount);
	}

Exit:

	if (RC_BAD( rc) || !pParent)
	{
		if (rc == FERR_IO_END_OF_FILE)
		{
			rc = FERR_OK;
		}
		*ppForest = NULL;
		pPool->poolReset( pvMark);
	}
	else
	{
		*ppForest = pParent;
	}
	return( rc);
}

/********************************************************************
Desc: Expands a config rfl size packet (RFL_CONFIG_SIZE_EVENT_PACKET)
*********************************************************************/
FSTATIC RCODE rflExpandConfigSizePacket(
	RFL_PACKET *	pRflPacket,
	F_Pool *			pPool,
	NODE **			ppForest
	)
{
	RCODE			rc = FERR_OK;
	void *		pvMark = pPool->poolMark();
	NODE *		pParent = NULL;
	NODE *		pLastNode;
	FLMUINT		uiOffset;

	// Output generic packet header information.

	if (RC_BAD( rc = rflExpandPacketHdr( pRflPacket, pPool, &pParent)))
	{
		goto Exit;
	}
	
	// Output transaction ID

	uiOffset = pRflPacket->uiFileOffset + RFL_PACKET_OVERHEAD;
	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
									RFL_TRANS_ID_FIELD,
									pRflPacket->uiTransID, uiOffset,
									4, pRflPacket->uiTransIDBytes, &pLastNode)))
	{
		goto Exit;
	}
	uiOffset += 4;

	// Output size threshhold

	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
									RFL_SIZE_THRESHOLD_FIELD,
									pRflPacket->uiSizeThreshold, uiOffset,
									4, pRflPacket->uiSizeThresholdBytes, &pLastNode)))
	{
		goto Exit;
	}
	uiOffset += 4;

	// Output time interval

	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
									RFL_TIME_INTERVAL_FIELD,
									pRflPacket->uiTimeInterval, uiOffset,
									4, pRflPacket->uiTimeInterval, &pLastNode)))
	{
		goto Exit;
	}
	uiOffset += 4;

	// Output size interval

	if (RC_BAD( rc = rflPutNum( pPool, pParent, FALSE,
									RFL_SIZE_INTERVAL_FIELD,
									pRflPacket->uiSizeInterval, uiOffset,
									4, pRflPacket->uiSizeInterval, &pLastNode)))
	{
		goto Exit;
	}
	uiOffset += 4;

Exit:

	if (RC_BAD( rc) || !pParent)
	{
		if (rc == FERR_IO_END_OF_FILE)
		{
			rc = FERR_OK;
		}
		*ppForest = NULL;
		pPool->poolReset( pvMark);
	}
	else
	{
		*ppForest = pParent;
	}
	return( rc);
}

/********************************************************************
Desc: Expands a packet into multiple GEDCOM nodes for display of
		all of the subcomponents.
*********************************************************************/
RCODE RflExpandPacket(
	NODE *		pPacketNode,
	F_Pool *		pPool,
	NODE **		ppForest
	)
{
	RCODE				rc = FERR_OK;
	void *			pvMark = pPool->poolMark();
	NODE *			pForest = NULL;
	RFL_PACKET *	pRflPacket;
	FLMBOOL			bFoundPrev;
	FLMUINT			uiDataLen;
	FLMUINT			uiLevel;
	RFL_PACKET		tmpPacket;

	// Handle case where nothing was passed in.

	if (!pPacketNode)
	{
		goto Exit;
	}
	if ((pRflPacket = (RFL_PACKET *)GedValPtr( pPacketNode)) == NULL)
	{
		goto Exit;
	}

	if (!pRflPacket->bValidPacketType)
	{
		rc = rflExpandUnkPacket( pRflPacket, pPool, &pForest);
		goto Exit;
	}
	else
	{
		switch (pRflPacket->uiPacketType)
		{
			case RFL_TRNS_BEGIN_PACKET:
			case RFL_TRNS_BEGIN_EX_PACKET:
			case RFL_TRNS_COMMIT_PACKET:
			case RFL_TRNS_ABORT_PACKET:
				rc = rflExpandTrnsPacket( pRflPacket, pPool, &pForest);
				goto Exit;
			case RFL_ADD_RECORD_PACKET:
			case RFL_ADD_RECORD_PACKET_VER_2:
			case RFL_MODIFY_RECORD_PACKET:
			case RFL_MODIFY_RECORD_PACKET_VER_2:
			case RFL_DELETE_RECORD_PACKET:
			case RFL_DELETE_RECORD_PACKET_VER_2:
			case RFL_RESERVE_DRN_PACKET:
				rc = rflExpandRecOpPacket( pRflPacket, pPool, &pForest, 0);
				goto Exit;
			case RFL_CHANGE_FIELDS_PACKET:
			case RFL_DATA_RECORD_PACKET:
			case RFL_ENC_DATA_RECORD_PACKET:
			case RFL_DATA_RECORD_PACKET_VER_3:
				f_memcpy( &tmpPacket, pRflPacket, sizeof( RFL_PACKET));
				bFoundPrev = FALSE;
				if (RC_BAD( rc = rflGetPrevOpPacket( &tmpPacket, &bFoundPrev)))
				{
					goto Exit;
				}
				if (!bFoundPrev)
				{
					uiDataLen = 0xFFFF;
					uiLevel = 0;
					if (pRflPacket->uiPacketType == RFL_DATA_RECORD_PACKET ||
						 pRflPacket->uiPacketType == RFL_ENC_DATA_RECORD_PACKET ||
						 pRflPacket->uiPacketType == RFL_DATA_RECORD_PACKET_VER_3)
					{
						rc = rflExpandDataPacket( pRflPacket, pPool,
							TRUE, pRflPacket->uiPacketType,
							&pForest, &uiDataLen, &uiLevel);
					}
					else
					{
						rc = rflExpandChangeFieldsPacket( pRflPacket, pPool,
											TRUE, &pForest, &uiDataLen);
					}
				}
				else
				{
					rc = rflExpandRecOpPacket( &tmpPacket, pPool,
								&pForest, pRflPacket->uiFileOffset);
				}
				break;
			case RFL_INDEX_SET_PACKET:
			case RFL_INDEX_SET_PACKET_VER_2:
				rc = rflExpandIndexSetPacket( pRflPacket, pPool, &pForest);
				goto Exit;
			case RFL_BLK_CHAIN_FREE_PACKET:
				rc = rflExpandBlkChainFreePacket( pRflPacket, pPool, &pForest);
				break;
			case RFL_START_UNKNOWN_PACKET:
				rc = rflExpandStartUnknownPacket( pRflPacket, pPool, &pForest);
				goto Exit;
			case RFL_UNKNOWN_PACKET:
				rc = rflExpandUnkPacket( pRflPacket, pPool, &pForest);
				goto Exit;
			case RFL_REDUCE_PACKET:
				rc = rflExpandReducePacket( pRflPacket, pPool, &pForest);
				goto Exit;
			case RFL_UPGRADE_PACKET:
				rc = rflExpandUpgradePacket( pRflPacket, pPool, &pForest);
				goto Exit;
			case RFL_INDEX_SUSPEND_PACKET:
			case RFL_INDEX_RESUME_PACKET:
				rc = rflExpandIndexStatePacket( pRflPacket, pPool, &pForest);
				goto Exit;
			case RFL_WRAP_KEY_PACKET:
			case RFL_ENABLE_ENCRYPTION_PACKET:
				rc = rflExpandEncryptionPacket( pRflPacket, pPool, &pForest);
				goto Exit;
			case RFL_CONFIG_SIZE_EVENT_PACKET:
				rc = rflExpandConfigSizePacket( pRflPacket, pPool, &pForest);
				goto Exit;
			default:
				rc = rflExpandUnkPacket( pRflPacket, pPool, &pForest);
				goto Exit;
		}
	}

Exit:
	if (RC_BAD( rc) || !pForest)
	{
		if (rc == FERR_IO_END_OF_FILE)
		{
			rc = FERR_OK;
		}
		*ppForest = NULL;
		pPool->poolReset( pvMark);
	}
	else
	{
		*ppForest = pForest;
	}
	return( rc);
}
