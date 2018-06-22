//------------------------------------------------------------------------------
// Desc:	Streaming interface
// Tabs:	3
//
// Copyright (c) 2004-2007 Novell, Inc. All Rights Reserved.
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

#include "ftksys.h"

#define LZW_MAGIC_NUMBER							0x3482
#define LZW_END_OF_DATA								256
#define LZW_NEW_DICT									257
#define LZW_STOP_COMPRESSION						258
#define LZW_START_CODE								259
#define LZW_MAX_CODE									0xFFFF

#define MULTI_FILE_OUT_STREAM_MIN_FILE_SIZE	1048510
#define MULTI_FILE_OUT_STREAM_MAX_FILE_SIZE	2147483647

/****************************************************************************
Desc:
****************************************************************************/
FLMBYTE gv_ucBase64DecodeTable[ 256] = 
{
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,		// 0   .. 7
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,		// 8   .. 15
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,		// 16  .. 23
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,		// 24  .. 31
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,		// 32  .. 39
	0xFF, 0xFF, 0xFF, 0x3E, 0xFF, 0xFF, 0xFF, 0x3F,		// 40  .. 47
	0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B,		// 48  .. 55
	0x3C, 0x3D, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF,		// 56  .. 63
	0xFF, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,		// 64  .. 71
	0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,		// 72  .. 79
	0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,		// 80  .. 87
	0x17, 0x18, 0x19, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,		// 88  .. 95
	0xFF, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20,		// 96  .. 103
	0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,		// 104 .. 111
	0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30,		// 112 .. 119
	0x31, 0x32, 0x33, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,		// 120 .. 127
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,		// 128 .. 135
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,		// 136 .. 143
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,		// 144 .. 151 
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,		// 152 .. 159
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,		// 160 .. 167
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,		// 168 .. 175
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,		// 176 .. 183
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,		// 184 .. 191
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,		// 192 .. 199
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,		// 200 .. 207
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,		// 208 .. 215
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,		// 216 .. 223
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,		// 224 .. 231
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,		// 232 .. 239
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,		// 240 .. 247
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF		// 248 .. 255
};

/****************************************************************************
Desc:
****************************************************************************/
FLMBYTE gv_ucBase64EncodeTable[ 64] = 
{
	ASCII_UPPER_A, ASCII_UPPER_B, ASCII_UPPER_C, ASCII_UPPER_D,
	ASCII_UPPER_E, ASCII_UPPER_F, ASCII_UPPER_G, ASCII_UPPER_H,
	ASCII_UPPER_I, ASCII_UPPER_J, ASCII_UPPER_K, ASCII_UPPER_L,
	ASCII_UPPER_M, ASCII_UPPER_N, ASCII_UPPER_O, ASCII_UPPER_P,
	ASCII_UPPER_Q, ASCII_UPPER_R, ASCII_UPPER_S, ASCII_UPPER_T,
	ASCII_UPPER_U, ASCII_UPPER_V, ASCII_UPPER_W, ASCII_UPPER_X,
	ASCII_UPPER_Y, ASCII_UPPER_Z, ASCII_LOWER_A, ASCII_LOWER_B, 
	ASCII_LOWER_C, ASCII_LOWER_D, ASCII_LOWER_E, ASCII_LOWER_F,
	ASCII_LOWER_G, ASCII_LOWER_H, ASCII_LOWER_I, ASCII_LOWER_J,
	ASCII_LOWER_K, ASCII_LOWER_L, ASCII_LOWER_M, ASCII_LOWER_N,
	ASCII_LOWER_O, ASCII_LOWER_P, ASCII_LOWER_Q, ASCII_LOWER_R,
	ASCII_LOWER_S, ASCII_LOWER_T, ASCII_LOWER_U, ASCII_LOWER_V,
	ASCII_LOWER_W, ASCII_LOWER_X, ASCII_LOWER_Y, ASCII_LOWER_Z,
	ASCII_ZERO,    ASCII_ONE,     ASCII_TWO,     ASCII_THREE, 
	ASCII_FOUR,    ASCII_FIVE,    ASCII_SIX,     ASCII_SEVEN,
	ASCII_EIGHT,   ASCII_NINE,    ASCII_PLUS,    ASCII_SLASH
};

/***************************************************************************
Desc:
***************************************************************************/
class FTKEXP F_StreamPrintfClient : public IF_PrintfClient
{
public:

	F_StreamPrintfClient( IF_OStream * pOStream)		
	{
		m_pOStream = pOStream;
		m_pOStream->AddRef();
		m_rc = NE_FLM_OK;
		m_uiBufOffset = 0;
	}
	
	virtual ~F_StreamPrintfClient()
	{
		if( m_pOStream)
		{
			m_pOStream->Release();
			m_pOStream = NULL;
		}
	}

	FINLINE FLMINT FTKAPI outputChar(
		char				cChar,
		FLMUINT			uiCount)
	{
		FLMUINT			uiTmpCount;
		FLMINT			iBytesOutput = (FLMINT)uiCount;
		
		while( uiCount)
		{
			uiTmpCount = uiCount;
			
			if( m_uiBufOffset + uiTmpCount > sizeof( m_szOutBuf))
			{
				uiTmpCount = sizeof( m_szOutBuf) - m_uiBufOffset;
			}
			
			f_memset( &m_szOutBuf[ m_uiBufOffset], cChar, uiTmpCount);
			
			m_uiBufOffset += uiTmpCount;
			uiCount -= uiTmpCount;
			
			if( m_uiBufOffset == sizeof( m_szOutBuf))
			{
				flushBuffer();
			}
		}
		
		return( iBytesOutput);
	}

	FINLINE FLMINT FTKAPI outputChar(
		char				cChar)
	{
		m_szOutBuf[ m_uiBufOffset++] = cChar;
		
		if( m_uiBufOffset == sizeof( m_szOutBuf))
		{
			flushBuffer();
		}
		
		return( 1);
	}
		
	FINLINE FLMINT FTKAPI outputStr(
		const char *	pszStr,
		FLMUINT			uiLen)
	{
		FLMUINT			uiTmpLen;
		FLMINT			iBytesOutput = (FLMINT)uiLen;
		
		while( uiLen)
		{
			uiTmpLen = uiLen;
			
			if( m_uiBufOffset + uiTmpLen > sizeof( m_szOutBuf))
			{
				uiTmpLen = sizeof( m_szOutBuf) - m_uiBufOffset;
			}
			
			f_memcpy( &m_szOutBuf[ m_uiBufOffset], pszStr, uiTmpLen);
			
			m_uiBufOffset += uiTmpLen;
			uiLen -= uiTmpLen;
			pszStr += uiTmpLen;
			
			if( m_uiBufOffset == sizeof( m_szOutBuf))
			{
				flushBuffer();
			}
		}
		
		return( iBytesOutput);
	}
		
	FINLINE FLMINT FTKAPI colorFormatter(
		char,				// cFormatChar,
		eColorType,		// eColor,
		FLMUINT)			// uiFlags)
	{
		return( 0);
	}

	FINLINE RCODE flushBuffer( void)
	{
		if( RC_OK( m_rc) && m_uiBufOffset)
		{
			m_rc = m_pOStream->write( m_szOutBuf, m_uiBufOffset, NULL);
		}
		
		m_uiBufOffset = 0;
		return( m_rc);
	}
	
private:

	RCODE							m_rc;
	char							m_szOutBuf[ 256];
	FLMUINT						m_uiBufOffset;
	IF_OStream *				m_pOStream;
};

/*****************************************************************************
Desc:
******************************************************************************/
RCODE FTKAPI FlmAllocBufferIStream( 
	IF_BufferIStream **		ppIStream)
{
	if( (*ppIStream = f_new F_BufferIStream) == NULL)
	{
		return( RC_SET( NE_FLM_MEM));
	}
	
	return( NE_FLM_OK);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE FTKAPI FlmOpenBufferIStream( 
	const char *				pucBuffer,
	FLMUINT						uiLength,
	IF_PosIStream **			ppIStream)
{
	RCODE							rc = NE_FLM_OK;
	F_BufferIStream *			pIStream = NULL;
	
	if( (pIStream = f_new F_BufferIStream) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pIStream->openStream( pucBuffer, uiLength)))
	{
		goto Exit;
	}
	
	*ppIStream = pIStream;
	pIStream = NULL;
	
Exit:

	if( pIStream)
	{
		pIStream->Release();
	}
	
	return( rc);
}
	
/*****************************************************************************
Desc:
******************************************************************************/
RCODE FTKAPI FlmOpenBase64EncoderIStream(
	IF_IStream *				pSourceIStream,
	FLMBOOL						bLineBreaks,
	IF_IStream **				ppIStream)
{
	RCODE								rc = NE_FLM_OK;
	F_Base64EncoderIStream *	pIStream = NULL;
	
	if( (pIStream = f_new F_Base64EncoderIStream) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pIStream->openStream( pSourceIStream, bLineBreaks)))
	{
		goto Exit;
	}
	
	*ppIStream = pIStream;
	pIStream = NULL;
	
Exit:

	if( pIStream)
	{
		pIStream->Release();
	}
	
	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE FTKAPI FlmOpenBase64DecoderIStream(
	IF_IStream *				pSourceIStream,
	IF_IStream **				ppIStream)
{
	RCODE								rc = NE_FLM_OK;
	F_Base64DecoderIStream *	pIStream = NULL;
	
	if( (pIStream = f_new F_Base64DecoderIStream) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pIStream->openStream( pSourceIStream)))
	{
		goto Exit;
	}
	
	*ppIStream = pIStream;
	pIStream = NULL;
	
Exit:

	if( pIStream)
	{
		pIStream->Release();
	}
	
	return( rc);
}
		
/*****************************************************************************
Desc:
******************************************************************************/
RCODE FTKAPI FlmOpenFileIStream(
	const char *				pszPath,
	IF_PosIStream **			ppIStream)
{
	RCODE							rc = NE_FLM_OK;
	F_FileIStream *			pIStream = NULL;
	
	if( (pIStream = f_new F_FileIStream) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pIStream->openStream( pszPath)))
	{
		goto Exit;
	}
	
	*ppIStream = pIStream;
	pIStream = NULL;
	
Exit:

	if( pIStream)
	{
		pIStream->Release();
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI FlmOpenMultiFileIStream(
	const char *				pszDirectory,
	const char *				pszBaseName,
	IF_IStream **				ppIStream)
{
	RCODE							rc = NE_FLM_OK;
	F_MultiFileIStream *		pIStream = NULL;
	
	if( (pIStream = f_new F_MultiFileIStream) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pIStream->openStream( pszDirectory, pszBaseName)))
	{
		goto Exit;
	}
	
	*ppIStream = pIStream;
	pIStream = NULL;
	
Exit:

	if( pIStream)
	{
		pIStream->Release();
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI FlmOpenBufferedIStream(
	IF_IStream *				pSourceIStream,
	FLMUINT						uiBufferSize,
	IF_IStream **				ppIStream)
{
	RCODE							rc = NE_FLM_OK;
	F_BufferedIStream *		pIStream = NULL;
	
	if( (pIStream = f_new F_BufferedIStream) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pIStream->openStream( pSourceIStream, uiBufferSize)))
	{
		goto Exit;
	}
	
	*ppIStream = pIStream;
	pIStream = NULL;
	
Exit:

	if( pIStream)
	{
		pIStream->Release();
	}
	
	return( rc);
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI FlmOpenUncompressingIStream(
	IF_IStream *				pSourceIStream,
	IF_IStream **				ppIStream)
{
	RCODE								rc = NE_FLM_OK;
	F_UncompressingIStream *	pIStream = NULL;
	
	if( (pIStream = f_new F_UncompressingIStream) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pIStream->openStream( pSourceIStream)))
	{
		goto Exit;
	}
	
	*ppIStream = pIStream;
	pIStream = NULL;
	
Exit:

	if( pIStream)
	{
		pIStream->Release();
	}
	
	return( rc);
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI FlmOpenFileOStream(
	const char *				pszPath,
	FLMBOOL						bTruncateIfExists,
	IF_OStream **				ppOStream)
{
	RCODE							rc = NE_FLM_OK;
	F_FileOStream *			pOStream = NULL;
	
	if( (pOStream = f_new F_FileOStream) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pOStream->openStream( pszPath, bTruncateIfExists)))
	{
		goto Exit;
	}
	
	*ppOStream = pOStream;
	pOStream = NULL;
	
Exit:

	if( pOStream)
	{
		pOStream->Release();
	}
	
	return( rc);
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI FlmOpenMultiFileOStream(
	const char *				pszDirectory,
	const char *				pszBaseName,
	FLMUINT						uiMaxFileSize,
	FLMBOOL						bOkToOverwrite,
	IF_OStream **				ppOStream)
{
	RCODE							rc = NE_FLM_OK;
	F_MultiFileOStream *		pOStream = NULL;
	
	if( (pOStream = f_new F_MultiFileOStream) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pOStream->createStream( pszDirectory, pszBaseName, 
		uiMaxFileSize, bOkToOverwrite)))
	{
		goto Exit;
	}
	
	*ppOStream = pOStream;
	pOStream = NULL;
	
Exit:

	if( pOStream)
	{
		pOStream->Release();
	}
	
	return( rc);
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI FlmOpenBufferedOStream(
	IF_OStream *				pDestOStream,
	FLMUINT						uiBufferSize,
	IF_OStream **				ppOStream)
{
	RCODE							rc = NE_FLM_OK;
	F_BufferedOStream *		pOStream = NULL;
	
	if( (pOStream = f_new F_BufferedOStream) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pOStream->openStream( pDestOStream, uiBufferSize)))
	{
		goto Exit;
	}
	
	*ppOStream = pOStream;
	pOStream = NULL;
	
Exit:

	if( pOStream)
	{
		pOStream->Release();
	}
	
	return( rc);
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI FlmOpenCompressingOStream(
	IF_OStream *				pDestOStream,
	IF_OStream **				ppOStream)
{
	RCODE							rc = NE_FLM_OK;
	F_CompressingOStream *	pOStream = NULL;
	
	if( (pOStream = f_new F_CompressingOStream) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pOStream->openStream( pDestOStream)))
	{
		goto Exit;
	}
	
	*ppOStream = pOStream;
	pOStream = NULL;
	
Exit:

	if( pOStream)
	{
		pOStream->Release();
	}
	
	return( rc);
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI FlmRemoveMultiFileStream(
	const char *				pszDirectory,
	const char *				pszBaseName)
{
	RCODE						rc = NE_FLM_OK;
	F_MultiFileOStream *	pMultiStream = NULL;
	
	if( (pMultiStream = f_new F_MultiFileOStream) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pMultiStream->processDirectory( 
		pszDirectory, pszBaseName, TRUE)))
	{
		goto Exit;
	}
	
Exit:

	if( pMultiStream)
	{
		pMultiStream->Release();
	}

	return( rc);
}
		
/****************************************************************************
Desc:
*****************************************************************************/
RCODE FTKAPI F_FileIStream::openStream(
	const char *		pszFilePath)
{
	RCODE			rc = NE_FLM_OK;

	closeStream();

	if( RC_BAD( rc = f_getFileSysPtr()->openFile( (char *)pszFilePath,
		FLM_IO_RDONLY | FLM_IO_SH_DENYNONE, &m_pFileHdl)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Closes the input stream and frees any resources
*****************************************************************************/
RCODE F_FileIStream::closeStream( void)
{
	RCODE		rc = NE_FLM_OK;
	
	if( m_pFileHdl)
	{
		rc = m_pFileHdl->closeFile();
		m_pFileHdl->Release();
		m_pFileHdl = NULL;
	}

	m_ui64FileOffset = 0;
	return( rc);
}

/****************************************************************************
Desc:	
*****************************************************************************/
FLMUINT64 FTKAPI F_FileIStream::totalSize( void)
{
	FLMUINT64		ui64FileSize = 0;

	(void)m_pFileHdl->size( &ui64FileSize);
	return( ui64FileSize);
}

/****************************************************************************
Desc:	
*****************************************************************************/
FLMUINT64 FTKAPI F_FileIStream::remainingSize( void)
{
	FLMUINT64		ui64TotalSize = totalSize();
	FLMUINT64		ui64Offset = getCurrPosition();

	if( ui64TotalSize >= ui64Offset)
	{
		return( ui64TotalSize - ui64Offset);
	}

	return( 0);
}

/****************************************************************************
Desc:	
*****************************************************************************/
FLMUINT64 FTKAPI F_FileIStream::getCurrPosition( void)
{
	return( m_ui64FileOffset);
}

/****************************************************************************
Desc:	
*****************************************************************************/
RCODE FTKAPI F_FileIStream::positionTo(
	FLMUINT64			ui64Offset)
{
	m_ui64FileOffset = ui64Offset;
	return( NE_FLM_OK);
}

/****************************************************************************
Desc:	Reads the requested amount of data from the stream.
*****************************************************************************/
RCODE F_FileIStream::read(
	void *			pvBuffer,
	FLMUINT			uiBytesToRead,
	FLMUINT *		puiBytesRead)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiBytesRead = 0;

	if( !m_pFileHdl)
	{
		rc = RC_SET( NE_FLM_READING_FILE);
		goto Exit;
	}

	rc = m_pFileHdl->read( m_ui64FileOffset, uiBytesToRead,
		pvBuffer, &uiBytesRead);
	m_ui64FileOffset += uiBytesRead;

	if( RC_BAD( rc))
	{
		if( rc == NE_FLM_IO_END_OF_FILE)
		{
			rc = RC_SET( NE_FLM_EOF_HIT);
		}
		goto Exit;
	}

Exit:

	if( puiBytesRead)
	{
		*puiBytesRead = uiBytesRead;
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FTKAPI F_BufferedIStream::openStream(
	IF_IStream *	pIStream,
	FLMUINT			uiBufferSize)
{
	RCODE		rc = NE_FLM_OK;

	if( m_pIStream || !pIStream)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_ILLEGAL_OP);
		goto Exit;
	}

	m_pIStream = pIStream;
	m_pIStream->AddRef();

	m_uiBufferSize = uiBufferSize;
	m_uiBufferOffset = 0;
	m_uiBytesAvail = 0;

	if( RC_BAD( rc = f_alloc( m_uiBufferSize, &m_pucBuffer)))
	{
		goto Exit;
	}

Exit:

	if (RC_BAD( rc))
	{
		closeStream();
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FTKAPI F_BufferedIStream::read(
	void *			pvBuffer,
	FLMUINT			uiBytesToRead,
	FLMUINT *		puiBytesRead)
{
	RCODE			rc = NE_FLM_OK;
	FLMBYTE *	pucBuffer = (FLMBYTE *)pvBuffer;
	FLMUINT		uiBytesRead = 0;
	FLMUINT		uiMaxSize;

	if (!m_pIStream)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_ILLEGAL_OP);
		goto Exit;
	}

	while( uiBytesToRead)
	{
		if( (uiMaxSize = m_uiBytesAvail - m_uiBufferOffset) == 0)
		{
			if (RC_BAD( rc = m_pIStream->read( 
				m_pucBuffer, m_uiBufferSize, &m_uiBytesAvail)))
			{
				if (rc != NE_FLM_EOF_HIT || !m_uiBytesAvail)
				{
					m_uiBufferOffset = 0;
					goto Exit;
				}
			}

			f_assert( m_uiBytesAvail <= m_uiBufferSize);
			m_uiBufferOffset = 0;
		}
		else if( uiBytesToRead < uiMaxSize)
		{
			f_memcpy( pucBuffer, &m_pucBuffer[ m_uiBufferOffset], uiBytesToRead);
			m_uiBufferOffset += uiBytesToRead;
			uiBytesRead += uiBytesToRead;
			uiBytesToRead = 0;
			break;
		}
		else
		{
			f_memcpy( pucBuffer, &m_pucBuffer[ m_uiBufferOffset], uiMaxSize);
			m_uiBufferOffset += uiMaxSize;
			pucBuffer += uiMaxSize;
			uiBytesToRead -= uiMaxSize;
			uiBytesRead += uiMaxSize;
		}
	}

Exit:

	if (puiBytesRead)
	{
		*puiBytesRead = uiBytesRead;
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FTKAPI F_BufferedIStream::closeStream( void)
{
	RCODE		rc = NE_FLM_OK;
	
	if( m_pIStream)
	{
		if( m_pIStream->getRefCount() == 1)
		{
			rc = m_pIStream->closeStream();
		}

		m_pIStream->Release();
		m_pIStream = NULL;
	}

	if( m_pucBuffer)
	{
		f_free( &m_pucBuffer);
	}

	m_uiBufferSize = 0;
	m_uiBufferOffset = 0;
	m_uiBytesAvail = 0;
	
	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FTKAPI F_FileOStream::openStream(
	const char *		pszFilePath,
	FLMBOOL				bTruncateIfExists)
{
	RCODE					rc = NE_FLM_OK;
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();

	if( m_pFileHdl)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_ILLEGAL_OP);
		goto Exit;
	}

	if( bTruncateIfExists)
	{
		if( RC_BAD( rc = pFileSystem->deleteFile( (char *)pszFilePath)))
		{
			if( rc != NE_FLM_IO_PATH_NOT_FOUND)
			{
				goto Exit;
			}
		}

		if( RC_BAD( rc = pFileSystem->createFile( 
			(char *)pszFilePath, FLM_IO_RDWR, &m_pFileHdl)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = pFileSystem->openFile(
			(char *)pszFilePath, FLM_IO_RDWR, &m_pFileHdl)))
		{
			if( rc != NE_FLM_IO_PATH_NOT_FOUND)
			{
				goto Exit;
			}

			if( RC_BAD( rc = pFileSystem->createFile( 
				(char *)pszFilePath, FLM_IO_RDWR, &m_pFileHdl)))
			{
				goto Exit;
			}
		}
	}

	if( RC_BAD( rc = m_pFileHdl->size( &m_ui64FileOffset)))
	{
		goto Exit;
	}

Exit:

	if( RC_BAD( rc))
	{
		closeStream();
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FTKAPI F_FileOStream::write(
	const void *	pvBuffer,
	FLMUINT			uiBytesToWrite,
	FLMUINT *		puiBytesWritten)
{
	RCODE		rc = NE_FLM_OK;
	FLMUINT	uiBytesWritten = 0;

	if (!m_pFileHdl)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_ILLEGAL_OP);
		goto Exit;
	}

	if( RC_BAD( rc = m_pFileHdl->write( (FLMUINT)m_ui64FileOffset, 
		uiBytesToWrite, (void *)pvBuffer, &uiBytesWritten)))
	{
		goto Exit;
	}

Exit:

	m_ui64FileOffset += uiBytesWritten;

	if( puiBytesWritten)
	{
		*puiBytesWritten = uiBytesWritten;
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FTKAPI F_FileOStream::closeStream( void)
{
	RCODE		rc = NE_FLM_OK;

	if( m_pFileHdl)
	{
		m_pFileHdl->Release();
		m_pFileHdl = NULL;
	}

	m_ui64FileOffset = 0;
	return( rc);
}

/******************************************************************************
Desc:
******************************************************************************/
RCODE FTKAPI F_MultiFileIStream::openStream(
	const char *	pszDirectory,
	const char *	pszBaseName)
{
	RCODE		rc = NE_FLM_OK;

	if( m_bOpen)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_ILLEGAL_OP);
		goto Exit;
	}

	f_strcpy( m_szDirectory, pszDirectory);
	f_strcpy( m_szBaseName, pszBaseName);

	m_uiFileNum = 0xFFFFFFFF;
	m_ui64FileOffset = 0;
	m_bEndOfStream = FALSE;
	m_bOpen = TRUE;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_MultiFileIStream::rollToNextFile( void)
{
	RCODE						rc = NE_FLM_OK;
	F_FileIStream *		pFileIStream = NULL;	
	F_BufferedIStream *	pBufferedIStream = NULL;
	FLMUINT					uiNewFileNum = 0;
	char						szFilePath[ F_PATH_MAX_SIZE + 1];
	char						szFileName[ F_PATH_MAX_SIZE + 1];
	IF_FileSystem *		pFileSystem = f_getFileSysPtr();

	if( m_pIStream)
	{
		m_pIStream->Release();
		m_pIStream = NULL;
		m_ui64FileOffset = 0;
	}

	if( m_uiFileNum == 0xFFFFFFFE)
	{
		rc = RC_SET( NE_FLM_STREAM_TOO_MANY_FILES);
		goto Exit;
	}
	else if( m_uiFileNum == 0xFFFFFFFF)
	{
		f_strcpy( szFileName, m_szBaseName);
		uiNewFileNum = 0;
	}
	else
	{
		uiNewFileNum = m_uiFileNum + 1;
		f_sprintf( (char *)szFileName, "%s.%08X", m_szBaseName, uiNewFileNum);
	}

	f_strcpy( szFilePath, m_szDirectory);
	if( RC_BAD( rc = pFileSystem->pathAppend( 
		(char *)szFilePath, (char *)szFileName)))
	{
		goto Exit;
	}

	if( (pFileIStream = f_new F_FileIStream) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pFileIStream->openStream( (const char *)szFilePath)))
	{
		if (rc == NE_FLM_IO_PATH_NOT_FOUND)
		{
			m_bEndOfStream = TRUE;
			rc = RC_SET( NE_FLM_EOF_HIT);
		}
		goto Exit;
	}

	if( (pBufferedIStream = f_new F_BufferedIStream) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pBufferedIStream->openStream( pFileIStream, 16384)))
	{
		goto Exit;
	}

	m_uiFileNum = uiNewFileNum;
	m_pIStream = pBufferedIStream;
	pBufferedIStream = NULL;

Exit:

	if( pFileIStream)
	{
		pFileIStream->Release();
	}

	if( pBufferedIStream)
	{
		pBufferedIStream->Release();
	}

	return( rc);
}

/******************************************************************************
Desc:
******************************************************************************/
RCODE FTKAPI F_MultiFileIStream::read(
	void *			pvBuffer,
	FLMUINT			uiBytesToRead,
	FLMUINT *		puiBytesRead)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiTmpRead;
	FLMUINT		uiTotalRead = 0;
	FLMBOOL		bRollToNextFile = FALSE;
	FLMBYTE *	pucBuffer = (FLMBYTE *)pvBuffer;

	if( !m_bOpen)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_ILLEGAL_OP);
		goto Exit;
	}

	if (m_bEndOfStream)
	{
		rc = RC_SET( NE_FLM_EOF_HIT);
		goto Exit;
	}

	if (!m_pIStream)
	{
		bRollToNextFile = TRUE;
	}

	while (uiBytesToRead)
	{
		if (bRollToNextFile)
		{
			if (RC_BAD( rc = rollToNextFile()))
			{
				goto Exit;
			}
		}

		if (RC_BAD( rc = m_pIStream->read( 
			pucBuffer, uiBytesToRead, &uiTmpRead)))
		{
			if (rc != NE_FLM_EOF_HIT)
			{
				goto Exit;
			}

			rc = NE_FLM_OK;
			bRollToNextFile = TRUE;
			if (!uiTmpRead)
			{
				continue;
			}
		}

		pucBuffer += uiTmpRead;
		uiBytesToRead -= uiTmpRead;
		uiTotalRead += uiTmpRead;
		m_ui64FileOffset += uiTmpRead;
	}
	
Exit:

	if (puiBytesRead)
	{
		*puiBytesRead = uiTotalRead;
	}

	return( rc);
}

/******************************************************************************
Desc:
******************************************************************************/
RCODE FTKAPI F_MultiFileIStream::closeStream( void)
{
	if( m_pIStream)
	{
		m_pIStream->Release();
		m_pIStream = NULL;
	}

	m_uiFileNum = 0;
	m_ui64FileOffset = 0;
	m_szDirectory[ 0] = 0;
	m_szBaseName[ 0] = 0;
	m_bEndOfStream = FALSE;
	m_bOpen = FALSE;
	
	return( NE_FLM_OK);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_MultiFileOStream::createStream(
	const char *	pszDirectory,
	const char *	pszBaseName,
	FLMUINT			uiMaxFileSize,
	FLMBOOL			bOkToOverwrite)
{
	RCODE 			rc = NE_FLM_OK;

	if( m_bOpen)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_ILLEGAL_OP);
		goto Exit;
	}

	if( RC_BAD( rc = processDirectory( 
		pszDirectory, pszBaseName, bOkToOverwrite)))
	{
		goto Exit;
	}

	f_strcpy( m_szDirectory, pszDirectory);
	f_strcpy( m_szBaseName, pszBaseName);

	if( !uiMaxFileSize)
	{
		uiMaxFileSize = MULTI_FILE_OUT_STREAM_MAX_FILE_SIZE;
	}
	else if( uiMaxFileSize < MULTI_FILE_OUT_STREAM_MIN_FILE_SIZE)
	{
		uiMaxFileSize = MULTI_FILE_OUT_STREAM_MIN_FILE_SIZE;
	}
	else if( uiMaxFileSize > MULTI_FILE_OUT_STREAM_MAX_FILE_SIZE)
	{
		uiMaxFileSize = MULTI_FILE_OUT_STREAM_MAX_FILE_SIZE;
	}

	m_uiFileNum = 0xFFFFFFFF;
	m_ui64FileOffset = 0;
	m_ui64MaxFileSize = uiMaxFileSize;
	m_bOpen = TRUE;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_MultiFileOStream::processDirectory(
	const char *		pszDirectory,
	const char *		pszBaseName,
	FLMBOOL				bOkToDelete)
{
	RCODE					rc = NE_FLM_OK;
	IF_DirHdl *			pDirHandle = NULL;
	FLMUINT				uiBaseNameLen = f_strlen( pszBaseName);
	const char *		pszName = NULL;
	char					szSearchPattern[ F_PATH_MAX_SIZE + 1];
	char					szFilePath[ F_PATH_MAX_SIZE + 1];
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();

	f_sprintf( szSearchPattern, "%s*", pszBaseName);
	
	if (!pszDirectory || *pszDirectory == 0)
	{
		pszDirectory = ".";
	}

	if( RC_BAD( rc = pFileSystem->openDir( 
		(char *)pszDirectory, szSearchPattern, &pDirHandle)))
	{
		goto Exit;
	}

	for( ;;)
	{
		if( RC_BAD( rc = pDirHandle->next()))
		{
			if( rc != NE_FLM_IO_NO_MORE_FILES)
			{
				goto Exit;
			}

			rc = NE_FLM_OK;
			break;
		}

		// Verify that the file belongs to the stream

		pszName = pDirHandle->currentItemName();
		if( f_strcmp( pszName, pszBaseName) == 0 ||
			  (f_strncmp( pszName, pszBaseName, uiBaseNameLen) == 0 &&
				pszName[ uiBaseNameLen] == '.' && 
				f_isValidHexNum( (FLMBYTE *)&pszName[ uiBaseNameLen + 1])))
		{
			if (!bOkToDelete)
			{
				rc = RC_SET( NE_FLM_STREAM_EXISTS);
				goto Exit;
			}

			// Delete the file

			f_strcpy( szFilePath, pszDirectory);
			
			if( RC_BAD( rc = pFileSystem->pathAppend( 
				szFilePath, pszName)))
			{
				goto Exit;
			}

			if( RC_BAD( pFileSystem->deleteFile( szFilePath)))
			{
				if (rc != NE_FLM_IO_PATH_NOT_FOUND)
				{
					goto Exit;
				}

				rc = NE_FLM_OK;
			}
		}
	}

Exit:

	if( pDirHandle)
	{
		pDirHandle->Release();
		pDirHandle = NULL;
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_MultiFileOStream::rollToNextFile( void)
{
	RCODE								rc = NE_FLM_OK;
	F_FileOStream *				pFileOStream = NULL;	
	F_BufferedOStream *			pBufferedOStream = NULL;
	FLMUINT							uiNewFileNum = 0;
	char								szFilePath[ F_PATH_MAX_SIZE + 1];
	char								szFileName[ F_PATH_MAX_SIZE + 1];
	IF_FileSystem *				pFileSystem = f_getFileSysPtr();

	if( m_pOStream)
	{
		if( RC_BAD( rc = m_pOStream->closeStream()))
		{
			goto Exit;
		}

		m_pOStream->Release();
		m_pOStream = NULL;
		m_ui64FileOffset = 0;
	}

	if( m_uiFileNum == 0xFFFFFFFE)
	{
		rc = RC_SET( NE_FLM_STREAM_TOO_MANY_FILES);
		goto Exit;
	}
	else if( m_uiFileNum == 0xFFFFFFFF)
	{
		f_strcpy( szFileName, m_szBaseName);
		uiNewFileNum = 0;
	}
	else
	{
		uiNewFileNum = m_uiFileNum + 1;
		f_sprintf( (char *)szFileName, "%s.%08X", m_szBaseName, uiNewFileNum);
	}

	f_strcpy( szFilePath, m_szDirectory);

	if( RC_BAD( rc = pFileSystem->pathAppend( 
		(char *)szFilePath, (char *)szFileName)))
	{
		goto Exit;
	}

	if( (pFileOStream = f_new F_FileOStream) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pFileOStream->openStream( (const char *)szFilePath, TRUE)))
	{
		goto Exit;
	}

	if( (pBufferedOStream = f_new F_BufferedOStream) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pBufferedOStream->openStream( pFileOStream, 16384)))
	{
		goto Exit;
	}

	m_uiFileNum = uiNewFileNum;
	m_pOStream = pBufferedOStream;
	pBufferedOStream = NULL;

Exit:

	if( pFileOStream)
	{
		pFileOStream->Release();
	}

	if( pBufferedOStream)
	{
		pBufferedOStream->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FTKAPI F_MultiFileOStream::write(
	const void *	pvBuffer,
	FLMUINT			uiBytesToWrite,
	FLMUINT *		puiBytesWritten)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiMaxToWrite;
	FLMUINT		uiBytesWritten = 0;
	FLMBYTE *	pucBuffer = (FLMBYTE *)pvBuffer;

	if (!m_bOpen)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_ILLEGAL_OP);
		goto Exit;
	}

	if (!m_pOStream)
	{
		if (RC_BAD( rc = rollToNextFile()))
		{
			goto Exit;
		}
	}

	while (uiBytesToWrite)
	{
		if ((uiMaxToWrite = (FLMUINT)(m_ui64MaxFileSize - m_ui64FileOffset)) < 
			uiBytesToWrite)
		{
			if (RC_BAD( rc = m_pOStream->write( pucBuffer, uiMaxToWrite)))
			{
				goto Exit;
			}

			pucBuffer += uiMaxToWrite;
			uiBytesWritten += uiMaxToWrite;

			if (RC_BAD( rc = rollToNextFile()))
			{
				goto Exit;
			}
		}
		else
		{
			uiMaxToWrite = uiBytesToWrite;

			if (RC_BAD( rc = m_pOStream->write( pucBuffer, uiBytesToWrite)))
			{
				goto Exit;
			}
		}

		uiBytesWritten += uiBytesToWrite;
		uiBytesToWrite -= uiMaxToWrite;
		m_ui64FileOffset += uiMaxToWrite;
	}
	
Exit:

	if( puiBytesWritten)
	{
		*puiBytesWritten = uiBytesWritten;
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FTKAPI F_MultiFileOStream::closeStream( void)
{
	RCODE		rc = NE_FLM_OK;

	if( m_pOStream)
	{
		rc = m_pOStream->closeStream();
		m_pOStream->Release();
		m_pOStream = NULL;
	}

	m_uiFileNum = 0;
	m_ui64MaxFileSize = 0;
	m_ui64FileOffset = 0;
	m_szDirectory[ 0] = 0;
	m_szBaseName[ 0] = 0;
	m_bOpen = FALSE;

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FTKAPI F_BufferedOStream::openStream(
	IF_OStream *	pOStream,
	FLMUINT			uiBufferSize)
{
	RCODE		rc = NE_FLM_OK;

	if( !pOStream || m_pOStream || !uiBufferSize)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_ILLEGAL_OP);
		goto Exit;
	}

	if( RC_BAD( rc = f_alloc( uiBufferSize, &m_pucBuffer)))
	{
		goto Exit;
	}

	m_pOStream = pOStream;
	m_pOStream->AddRef();

	m_uiBufferSize = uiBufferSize;
	m_uiBufferOffset = 0;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FTKAPI F_BufferedOStream::flush( void)
{
	RCODE		rc = NE_FLM_OK;

	if( m_uiBufferOffset)
	{
		if( RC_BAD( rc = m_pOStream->write( m_pucBuffer, m_uiBufferOffset)))
		{
			goto Exit;
		}

		m_uiBufferOffset = 0;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FTKAPI F_BufferedOStream::write(
	const void *	pvBuffer,
	FLMUINT			uiBytesToWrite,
	FLMUINT *		puiBytesWritten)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiMaxToWrite;
	FLMUINT		uiBytesWritten = 0;
	FLMBYTE *	pucBuffer = (FLMBYTE *)pvBuffer;

	while( uiBytesToWrite)
	{
		uiMaxToWrite = (FLMUINT)(m_uiBufferSize - m_uiBufferOffset);
		uiMaxToWrite = uiMaxToWrite > uiBytesToWrite 
								? uiBytesToWrite 
								: uiMaxToWrite;

		f_memcpy( &m_pucBuffer[ m_uiBufferOffset], pucBuffer, uiMaxToWrite);
		pucBuffer += uiMaxToWrite;
		m_uiBufferOffset += uiMaxToWrite;
		uiBytesToWrite -= uiMaxToWrite;
		uiBytesWritten += uiMaxToWrite;

		if (m_uiBufferOffset == m_uiBufferSize)
		{
			if (RC_BAD( rc = flush()))
			{
				goto Exit;
			}
		}
	}

Exit:

	if( puiBytesWritten)
	{
		*puiBytesWritten = uiBytesWritten;
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FTKAPI F_BufferedOStream::closeStream( void)
{
	RCODE		rc = NE_FLM_OK;

	if( m_pOStream)
	{
		if( RC_OK( rc = flush()))
		{
			if( m_pOStream->getRefCount() == 1)
			{
				rc = m_pOStream->closeStream();
			}
		}

		m_pOStream->Release();
		m_pOStream = NULL;
	}

	if( m_pucBuffer)
	{
		f_free( &m_pucBuffer);
	}

	m_uiBufferSize = 0;
	m_uiBufferOffset = 0;

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
F_BufferIStream::~F_BufferIStream()
{
	closeStream();
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE FTKAPI F_BufferIStream::openStream(
	const char *	pucBuffer,
	FLMUINT			uiLength,
	char **			ppucAllocatedBuffer)
{
	RCODE				rc = NE_FLM_OK;
	
	f_assert( !m_pucBuffer);
	
	if( !pucBuffer && uiLength)
	{
		if( RC_BAD( rc = f_alloc( uiLength, &m_pucBuffer)))
		{
			goto Exit;
		}
		
		if( ppucAllocatedBuffer)
		{
			*ppucAllocatedBuffer = (char *)m_pucBuffer;
		}
		
		m_bAllocatedBuffer = TRUE;
	}
	else
	{
		m_pucBuffer = (FLMBYTE *)pucBuffer;
	}
	
	m_uiBufferLen = uiLength;
	m_uiOffset = 0;
	m_bIsOpen = TRUE;

Exit:

	return( rc);
}
		
/*****************************************************************************
Desc:
******************************************************************************/
RCODE FTKAPI F_BufferIStream::closeStream( void)
{
	if( m_bIsOpen)
	{
		if( m_bAllocatedBuffer)
		{
			if( m_pucBuffer)
			{
				f_free( &m_pucBuffer);
			}
			
			m_bAllocatedBuffer = FALSE;
		}
		else
		{
			m_pucBuffer = NULL;
		}
		
		m_bIsOpen = FALSE;
	}
	
	return( NE_FLM_OK);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE FTKAPI F_BufferIStream::read(
	void *			pvBuffer,
	FLMUINT			uiBytesToRead,
	FLMUINT *		puiBytesRead)
{
	RCODE			rc = NE_FLM_OK;
	FLMBYTE *	pucBuffer = (FLMBYTE *)pvBuffer;
	FLMUINT		uiBytesRead;
	
	f_assert( m_bIsOpen);

	uiBytesRead = uiBytesToRead < m_uiBufferLen - m_uiOffset
					  ? uiBytesToRead
					  : m_uiBufferLen - m_uiOffset;

	if (uiBytesRead)
	{
		if (pucBuffer)
		{
			f_memcpy( pucBuffer, &m_pucBuffer[ m_uiOffset], uiBytesRead);
		}

		m_uiOffset += uiBytesRead;
	}

	if (puiBytesRead)
	{
		*puiBytesRead = uiBytesRead;
	}

	if (uiBytesRead < uiBytesToRead)
	{
		rc = RC_SET( NE_FLM_EOF_HIT);
		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE FTKAPI F_Base64DecoderIStream::openStream(
	IF_IStream *	pIStream)
{
	RCODE		rc = NE_FLM_OK;

	if( m_pIStream || !pIStream)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_ILLEGAL_OP);
		goto Exit;
	}

	m_uiBufOffset = 0;
	m_uiAvailBytes = 0;
	m_pIStream = pIStream;
	pIStream->AddRef();

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Reads decoded binary from the base64 ASCII source stream.
*****************************************************************************/
RCODE FTKAPI F_Base64DecoderIStream::read(
	void *			pvBuffer,
	FLMUINT			uiBytesToRead,
	FLMUINT *		puiBytesRead)
{
	RCODE				rc = NE_FLM_OK;
	FLMBYTE *		pucOutBuf = (FLMBYTE *)pvBuffer;
	FLMBYTE			ucQuadBuffer[ 4];
	FLMUINT			uiOffset;
	FLMUINT			uiBytesToCopy;
	
	if( puiBytesRead)
	{
		*puiBytesRead = 0;
	}
	
	if( !m_pIStream)
	{
		rc = RC_SET( NE_FLM_EOF_HIT);
		goto Exit;
	}
	
	while( uiBytesToRead)
	{
		if( !m_uiAvailBytes)
		{
			m_uiBufOffset = 0;

			for( uiOffset = 0; uiOffset < 4;)
			{
				if( RC_BAD( rc = m_pIStream->read( 
					&ucQuadBuffer[ uiOffset], 1, NULL)))
				{
					if( rc != NE_FLM_EOF_HIT || !uiOffset)
					{
						goto Exit;
					}
					
					// Handle input that has not been padded correctly

					ucQuadBuffer[ uiOffset] = ASCII_EQUAL;
				}
				
				if( gv_ucBase64DecodeTable[ ucQuadBuffer[ uiOffset]] == 0xFF)
				{
					FLMBYTE	ucTmp = ucQuadBuffer[ uiOffset];

					if( ucTmp == 0 || ucTmp == ASCII_TAB || ucTmp == ASCII_SPACE ||
						ucTmp == ASCII_NEWLINE || ucTmp == ASCII_CR)
					{
						continue;
					}
					
					rc = RC_SET( NE_FLM_BAD_BASE64_ENCODING);
					goto Exit;
				}

				uiOffset++;
			}

			// If we didn't get anything except padding, assume we are at the end
			// of an incorrectly encoded buffer.
			
			if( ucQuadBuffer[ 0] == ASCII_EQUAL)
			{
				goto Exit;
			}
			
			m_ucBuffer[ 0] = 
				(gv_ucBase64DecodeTable[ ucQuadBuffer[ 0]] << 2) |
				(gv_ucBase64DecodeTable[ ucQuadBuffer[ 1]] >> 4);
			m_uiAvailBytes++;
			
			if( ucQuadBuffer[ 2] != '=')
			{
				m_ucBuffer[ 1] =
					(gv_ucBase64DecodeTable[ ucQuadBuffer[ 1]] << 4) |
					(gv_ucBase64DecodeTable[ ucQuadBuffer[ 2]] >> 2);
				m_uiAvailBytes++;
			}
			
			if( ucQuadBuffer[ 3] != '=')
			{
				m_ucBuffer[ 2] =
					(gv_ucBase64DecodeTable[ ucQuadBuffer[ 2]] << 6) |
					gv_ucBase64DecodeTable[ ucQuadBuffer[ 3]];
				m_uiAvailBytes++;
			}
		}
		
		uiBytesToCopy = f_min( m_uiAvailBytes, uiBytesToRead);

		if( pucOutBuf)
		{
			f_memcpy( pucOutBuf, &m_ucBuffer[ m_uiBufOffset], uiBytesToCopy);
		}

		uiBytesToRead -= uiBytesToCopy;
		m_uiAvailBytes -= uiBytesToCopy;
		m_uiBufOffset += uiBytesToCopy;
		pucOutBuf += uiBytesToCopy;
		
		if( puiBytesRead)
		{
			*puiBytesRead += uiBytesToCopy;
		}
	}
	
Exit:

	return( rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE FTKAPI F_Base64EncoderIStream::openStream(
	IF_IStream *	pIStream,
	FLMBOOL			bLineBreaks)
{
	RCODE		rc = NE_FLM_OK;

	if( m_pIStream || !pIStream)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_ILLEGAL_OP);
		goto Exit;
	}

	m_uiBase64Count = 0;
	m_uiBufOffset = 0;
	m_uiAvailBytes = 0;
	m_bLineBreaks = bLineBreaks;
	m_bInputExhausted = FALSE;
	m_bPriorLineEnd = FALSE;
	m_pIStream = pIStream;
	pIStream->AddRef();

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Reads ASCII base64 encoded binary from the source stream.
*****************************************************************************/
RCODE FTKAPI F_Base64EncoderIStream::read(
	void *					pvBuffer,
	FLMUINT					uiBytesToRead,
	FLMUINT *				puiBytesRead)
{
	RCODE			rc = NE_FLM_OK;
	FLMBYTE *	pucOutBuf = (FLMBYTE *)pvBuffer;
	FLMUINT		uiBytesToCopy;
	FLMUINT		uiBytesToEncode;
	FLMBYTE		ucTriBuffer[ 3];
	
	if( puiBytesRead)
	{
		*puiBytesRead = 0;
	}
	
	if( !m_pIStream)
	{
		rc = RC_SET( NE_FLM_EOF_HIT);
		goto Exit;
	}
	
	while( uiBytesToRead)
	{
		if( !m_uiAvailBytes)
		{
			m_uiBufOffset = 0;

			if( m_bInputExhausted)
			{
				rc = RC_SET( NE_FLM_EOF_HIT);
				goto Exit;
			}

			if( RC_BAD( rc = m_pIStream->read( ucTriBuffer, 
				3, &uiBytesToEncode)))
			{
				if( rc != NE_FLM_EOF_HIT)
				{
					goto Exit;
				}
			
				rc = NE_FLM_OK;
				m_bInputExhausted = TRUE;
			}

			if( uiBytesToEncode)
			{
				m_ucBuffer[ m_uiAvailBytes++] = 
					gv_ucBase64EncodeTable[ ucTriBuffer[ 0] >> 2];

				m_ucBuffer[ m_uiAvailBytes++] = 
					gv_ucBase64EncodeTable[ ((ucTriBuffer[ 0] & 0x03) << 4) |
					(ucTriBuffer[ 1] >> 4)];

				if( uiBytesToEncode >= 2)
				{
					m_ucBuffer[ m_uiAvailBytes++] = 
						gv_ucBase64EncodeTable[ ((ucTriBuffer[ 1] & 0x0F) << 2) |
						(ucTriBuffer[ 2] >> 6)];
				}
				else
				{
					m_ucBuffer[ m_uiAvailBytes++] = ASCII_EQUAL;
				}
				
				if( uiBytesToEncode == 3)
				{
					m_ucBuffer[ m_uiAvailBytes++] =
						gv_ucBase64EncodeTable[ ucTriBuffer[ 2] & 0x3F];
				}
				else
				{
					m_ucBuffer[ m_uiAvailBytes++] = ASCII_EQUAL;
				}

				m_uiBase64Count += 4;
			}

			if( m_bLineBreaks)
			{
				if( (m_uiBase64Count % 72) == 0 || 
					(m_bInputExhausted && !m_bPriorLineEnd))
				{
#ifdef FLM_UNIX
					m_ucBuffer[ m_uiAvailBytes++] = ASCII_NEWLINE;
#elif FLM_OSX
					m_ucBuffer[ m_uiAvailBytes++] = ASCII_CR;
#else
					m_ucBuffer[ m_uiAvailBytes++] = ASCII_CR;
					m_ucBuffer[ m_uiAvailBytes++] = ASCII_NEWLINE;
#endif
					m_bPriorLineEnd = TRUE;
				}
				else
				{
					m_bPriorLineEnd = FALSE;
				}
			}

			if( !m_uiAvailBytes)
			{
				rc = RC_SET( NE_FLM_EOF_HIT);
				goto Exit;
			}
		}
		
		uiBytesToCopy = f_min( m_uiAvailBytes, uiBytesToRead);

		if( pucOutBuf)
		{
			f_memcpy( pucOutBuf, &m_ucBuffer[ m_uiBufOffset], uiBytesToCopy);
		}

		pucOutBuf += uiBytesToCopy;
		uiBytesToRead -= uiBytesToCopy;
		m_uiAvailBytes -= uiBytesToCopy;
		m_uiBufOffset += uiBytesToCopy;
		
		if( puiBytesRead)
		{
			*puiBytesRead += uiBytesToCopy;
		}
	}

Exit:

	return( rc);
}

/******************************************************************************
Desc:
******************************************************************************/
RCODE FTKAPI F_CompressingOStream::openStream(
	IF_OStream *		pOStream)
{
	RCODE			rc = NE_FLM_OK;
	FLMBYTE		ucOutBuf[ 2];

	// Setup the hash table

	m_uiHashTblSize = ((2 * 1024 * 1024) / sizeof( LZWODictItem *));
	if( RC_BAD( rc = f_alloc( 
		sizeof( LZWODictItem *) * m_uiHashTblSize, &m_ppHashTbl)))
	{
		goto Exit;
	}
	
	f_memset( m_ppHashTbl, 0, sizeof( LZWODictItem *) * m_uiHashTblSize);

	// Output a magic number so the stream can be identified

	UW2FBA( LZW_MAGIC_NUMBER, ucOutBuf);
	if( RC_BAD( rc = pOStream->write( ucOutBuf, 2)))
	{
		goto Exit;
	}

	// Setup misc. member variables

	m_pOStream = pOStream;
	m_pOStream->AddRef();

	m_ui16CurrentCode = LZW_END_OF_DATA;
	m_ui16FreeCode = LZW_START_CODE;
	m_uiLastRatio = 100;
	m_uiBestRatio = 100;
	m_uiCurrentBytesIn = 0;
	m_uiTotalBytesIn = 0;
	m_uiCurrentBytesOut = 0;
	m_uiTotalBytesOut = 0;
	m_bStopCompression = FALSE;

Exit:

	if( RC_BAD( rc))
	{
		closeStream();
	}

	return( rc);
}

/******************************************************************************
Desc:
******************************************************************************/
LZWODictItem * F_CompressingOStream::findDictEntry( 
	FLMUINT16	ui16CurrentCode,
	FLMBYTE		ucChar)
{
	FLMUINT			uiHashBucket;
	LZWODictItem *	pDictItem;
	FLMUINT			uiLooks = 0;

	uiHashBucket = getHashBucket( ui16CurrentCode, ucChar);
	pDictItem = m_ppHashTbl[ uiHashBucket];

	while( pDictItem)
	{
		if( pDictItem->ui16ParentCode == ui16CurrentCode &&
			pDictItem->ucChar == ucChar)
		{
			break;
		}

		pDictItem = pDictItem->pNext;
		uiLooks++;
	}

	return( pDictItem);
}

/******************************************************************************
Desc:
******************************************************************************/
RCODE FTKAPI F_CompressingOStream::write(
	const void *	pvBuffer,
	FLMUINT			uiBytesToWrite,
	FLMUINT *		puiBytesWritten)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT				uiBucket;
	FLMUINT				uiBytesWritten = 0;
	FLMUINT				uiTmp;
	LZWODictItem *		pDictItem;
	const FLMBYTE *	pucBuffer = (const FLMBYTE *)pvBuffer;
	FLMBYTE				ucOut[ 2];

	if( !uiBytesToWrite)
	{
		goto Exit;
	}

	if( m_bStopCompression)
	{
		rc = m_pOStream->write( pucBuffer, uiBytesToWrite, &uiTmp);
		uiBytesWritten += uiTmp;
		goto Exit;
	}

	if( !m_uiTotalBytesIn)
	{
		m_ui16CurrentCode = *pucBuffer++;
		m_uiCurrentBytesIn++;
		m_uiTotalBytesIn++;
		uiBytesToWrite--;
	}

	while( uiBytesToWrite)
	{
		if( (pDictItem = findDictEntry( 
			m_ui16CurrentCode, *pucBuffer)) == NULL)
		{
			// No match.  Output the code.

			UW2FBA( m_ui16CurrentCode, ucOut);
			if( RC_BAD( rc = m_pOStream->write( ucOut, 2)))
			{
				goto Exit;
			}
			m_uiCurrentBytesOut += 2;
			m_uiTotalBytesOut += 2;
			uiBytesWritten += 2;

			// Add the new code to the dictionary

			if( m_ui16FreeCode < LZW_MAX_CODE)
			{
				uiBucket = getHashBucket( m_ui16CurrentCode, *pucBuffer);
				if( RC_BAD( rc = m_pool.poolAlloc( 
					sizeof( LZWODictItem), (void **)&pDictItem)))
				{
					goto Exit;
				}

				pDictItem->pNext = m_ppHashTbl[ uiBucket];
				m_ppHashTbl[ uiBucket] = pDictItem;

				pDictItem->ucChar = *pucBuffer;
				pDictItem->ui16Code = m_ui16FreeCode++;
				pDictItem->ui16ParentCode = m_ui16CurrentCode;
			}

			m_ui16CurrentCode = *pucBuffer;

			// May need to reset the dictionary to improve compression.
			// If compression is causing the stream to grow, we
			// need to disable it.

			if( m_uiTotalBytesIn > (10 * 1024 * 1024) && 
				m_uiTotalBytesOut > m_uiTotalBytesIn)
			{
				// Compression isn't buying us anything.  From this
				// point forward in the stream, just store the bytes
				// without compression.

				UW2FBA( LZW_STOP_COMPRESSION, ucOut);
				if (RC_BAD( rc = m_pOStream->write( ucOut, 2)))
				{
					goto Exit;
				}

				uiBytesWritten += 2;
				m_bStopCompression = TRUE;
				m_ui16CurrentCode = LZW_END_OF_DATA;

				// Finish writing out the rest of the current buffer

				if( RC_BAD( rc = m_pOStream->write( pucBuffer, uiBytesToWrite)))
				{
					goto Exit;
				}

				m_uiCurrentBytesIn = 0;
				m_uiTotalBytesIn += uiBytesToWrite;
				uiBytesWritten += uiBytesToWrite;
				uiBytesToWrite = 0;
				break;
			}
			else if( m_uiCurrentBytesIn >= 8192)
			{
				FLMUINT	uiRatio;

				uiRatio = (m_uiCurrentBytesOut * 100) / m_uiCurrentBytesIn;
				m_uiCurrentBytesIn = 0;
				m_uiCurrentBytesOut = 0;

				if( uiRatio > m_uiBestRatio)
				{
					if( uiRatio > 50 && 
						(uiRatio > 90 || uiRatio > m_uiLastRatio + 10))
					{
						// Output the dictionary reset token

						UW2FBA( LZW_NEW_DICT, ucOut);
						if (RC_BAD( rc = m_pOStream->write( ucOut, 2)))
						{
							goto Exit;
						}

						uiBytesWritten += 2;

						// Reset the dictionary

						m_pool.poolReset( NULL);
						f_memset( m_ppHashTbl, 0, sizeof( LZWODictItem *) * m_uiHashTblSize);
						m_ui16FreeCode = LZW_START_CODE;
					}
					else
					{
						m_uiBestRatio = uiRatio;
					}

					m_uiLastRatio = uiRatio;
				}

				// Good time to release the CPU

				f_yieldCPU();
			}
		}
		else
		{
			m_ui16CurrentCode = pDictItem->ui16Code;
		}

		pucBuffer++;
		m_uiCurrentBytesIn++;
		m_uiTotalBytesIn++;
		uiBytesToWrite--;
	}

Exit:

	if( puiBytesWritten)
	{
		*puiBytesWritten = uiBytesWritten;
	}

	return( rc);
}

/******************************************************************************
Desc:
******************************************************************************/
RCODE FTKAPI F_CompressingOStream::closeStream( void)
{
	RCODE			rc = NE_FLM_OK;
	FLMBYTE		ucOut[ 2];	

	if (m_pOStream)
	{
		if (RC_OK( rc) && m_ui16CurrentCode != LZW_END_OF_DATA)
		{
			UW2FBA( m_ui16CurrentCode, ucOut);
			rc = m_pOStream->write( ucOut, 2);
			m_uiCurrentBytesOut += 2;
			m_uiTotalBytesOut += 2;
		}

		// Write the end-of-data marker

		if (RC_OK( rc))
		{
			UW2FBA( LZW_END_OF_DATA, ucOut);
			rc = m_pOStream->write( ucOut, 2);
			m_uiCurrentBytesOut += 2;
			m_uiTotalBytesOut += 2;
		}

		if (m_pOStream->getRefCount() == 1)
		{
			if (RC_OK( rc))
			{
				rc = m_pOStream->closeStream();
			}
			else
			{
				m_pOStream->closeStream();
			}
		}

		m_pOStream->Release();
		m_pOStream = NULL;
	}

	if( m_ppHashTbl)
	{
		f_free( &m_ppHashTbl);
		m_uiHashTblSize = 0;
	}
	
	m_pool.poolFree();

	return( rc);
}

/******************************************************************************
Desc:
******************************************************************************/
RCODE FTKAPI F_UncompressingIStream::openStream(
	IF_IStream *		pIStream)
{
	RCODE			rc = NE_FLM_OK;
	FLMBYTE		ucInBuf[ 2];
	FLMUINT16	ui16Magic;

	// Allocate the dictonary table

	if( RC_BAD( rc = f_alloc( 
		sizeof( LZWIDictItem) * LZW_MAX_CODE, &m_pDict)))
	{
		goto Exit;
	}

	f_memset( m_pDict, 0, sizeof( LZWIDictItem) * LZW_MAX_CODE);

	// Allocate the decode buffer

	m_uiDecodeBufferSize = 4096;
	if( RC_BAD( rc = f_alloc( m_uiDecodeBufferSize, &m_pucDecodeBuffer)))
	{
		goto Exit;
	}

	// Read the magic number from the stream to ensure that this
	// is really an LZW compressed stream

	if( RC_BAD( rc = pIStream->read( ucInBuf, 2, NULL)))
	{
		goto Exit;
	}
	ui16Magic = FB2UW( ucInBuf);

	if( ui16Magic != LZW_MAGIC_NUMBER)
	{
		rc = RC_SET( NE_FLM_STREAM_NOT_COMPRESSED);
		goto Exit;
	}

	// Add a reference to the passed-in stream object

	m_pIStream = pIStream;
	m_pIStream->AddRef();

	// Setup misc. member variables

	m_ui16FreeCode = LZW_START_CODE;
	m_ui16LastCode = LZW_END_OF_DATA;
	m_uiDecodeBufferOffset = 0;
	m_bStopCompression = FALSE;
	m_bEndOfStream = FALSE;

Exit:

	if( RC_BAD( rc))
	{
		closeStream();
	}

	return( rc);
}

/******************************************************************************
Desc:
******************************************************************************/
RCODE F_UncompressingIStream::readCode(
	FLMUINT16 *		pui16Code)
{
	RCODE			rc = NE_FLM_OK;
	FLMBYTE		ucInBuf[ 2];

	if( m_bEndOfStream)
	{
		rc = RC_SET( NE_FLM_EOF_HIT);
		goto Exit;
	}

	if( RC_BAD( rc = m_pIStream->read( ucInBuf, 2, NULL)))
	{
		goto Exit;
	}
	*pui16Code = FB2UW( ucInBuf);

	if( *pui16Code == LZW_END_OF_DATA)
	{
		m_bEndOfStream = TRUE;
		rc = RC_SET( NE_FLM_EOF_HIT);
		goto Exit;
	}

Exit:

	return( rc);
}

/******************************************************************************
Desc:
******************************************************************************/
RCODE F_UncompressingIStream::decodeToBuffer(
	FLMUINT16		ui16Code)
{
	RCODE			rc = NE_FLM_OK;

	if( ui16Code >= m_ui16FreeCode || m_ui16LastCode == LZW_END_OF_DATA)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_STREAM_DECOMPRESS_ERROR);
		goto Exit;
	}

	while( ui16Code > 0x00FF)
	{
		f_assert( m_uiDecodeBufferOffset < m_uiDecodeBufferSize);
		f_assert( ui16Code < m_ui16FreeCode);

		m_pucDecodeBuffer[ m_uiDecodeBufferOffset++] = m_pDict[ ui16Code].ucChar;
		ui16Code = m_pDict[ ui16Code].ui16ParentCode;
	}
	m_pucDecodeBuffer[ m_uiDecodeBufferOffset++] = (FLMBYTE)ui16Code;

Exit:

	return( rc);
}

/******************************************************************************
Desc:
******************************************************************************/
RCODE FTKAPI F_UncompressingIStream::read(
	void *			pvBuffer,
	FLMUINT			uiBytesToRead,
	FLMUINT *		puiBytesRead)
{
	RCODE			rc = NE_FLM_OK;
	FLMBYTE *	pucBuffer = (FLMBYTE *)pvBuffer;
	FLMUINT		uiTmp;
	FLMUINT		uiBytesRead = 0;
	FLMUINT		uiSavePos;
	FLMUINT16	ui16Code;

	while( uiBytesToRead)
	{
		if( m_uiDecodeBufferOffset)
		{
			// Consume a byte from the decode buffer

			*pucBuffer++ = m_pucDecodeBuffer[ --m_uiDecodeBufferOffset];
			uiBytesToRead--;
			continue;
		}

		if( m_bStopCompression)
		{
			if (RC_BAD( rc = m_pIStream->read( pvBuffer, uiBytesToRead, &uiTmp)))
			{
				if (rc != NE_FLM_EOF_HIT)
				{
					goto Exit;
				}

				uiBytesRead += uiTmp;
			}

			break;
		}

		if( RC_BAD( rc = readCode( &ui16Code)))
		{
			goto Exit;
		}

		if( ui16Code == LZW_NEW_DICT)
		{
			m_ui16FreeCode = LZW_START_CODE;
			m_ui16LastCode = LZW_END_OF_DATA;
			continue;
		}
		else if( ui16Code == LZW_STOP_COMPRESSION)
		{
			f_assert( !m_bStopCompression);
			m_bStopCompression = TRUE;
			continue;
		}
		else
		{
			if( ui16Code >= m_ui16FreeCode)
			{
				// The code isn't in our dictionary.  There is only
				// one type of sequence that can result in this
				// condition.  The code below builds the correct
				// sequence of bytes.

				f_assert( m_ui16LastCode != LZW_END_OF_DATA);

				uiSavePos = m_uiDecodeBufferOffset++;
				if( RC_BAD( rc = decodeToBuffer( m_ui16LastCode)))
				{
					goto Exit;
				}

				m_pucDecodeBuffer[ uiSavePos] = 
					m_pucDecodeBuffer[ m_uiDecodeBufferOffset - 1];
			}
			else if( m_ui16LastCode == LZW_END_OF_DATA)
			{
				f_assert( ui16Code <= 0x00FF);
				*pucBuffer++ = (FLMBYTE)ui16Code;
				uiBytesToRead--;
				m_ui16LastCode = ui16Code;
				continue;
			}
			else
			{
				if( RC_BAD( rc = decodeToBuffer( ui16Code)))
				{
					goto Exit;
				}
			}

			if( m_ui16FreeCode < LZW_MAX_CODE)
			{
				f_assert( m_uiDecodeBufferOffset);

				m_pDict[ m_ui16FreeCode].ui16ParentCode = m_ui16LastCode;
				m_pDict[ m_ui16FreeCode].ucChar = 
							m_pucDecodeBuffer[ m_uiDecodeBufferOffset - 1];
				m_ui16FreeCode++;
			}

			m_ui16LastCode = ui16Code;
		}
	}

Exit:

	if( puiBytesRead)
	{
		*puiBytesRead = uiBytesRead;
	}

	return( rc);
}

/******************************************************************************
Desc:
******************************************************************************/
RCODE FTKAPI F_UncompressingIStream::closeStream( void)
{
	if( m_pIStream)
	{
		m_pIStream->Release();
		m_pIStream = NULL;
	}

	if( m_pDict)
	{
		f_free( &m_pDict);
	}

	if( m_pucDecodeBuffer)
	{
		f_free( &m_pucDecodeBuffer);
	}
	
	return( NE_FLM_OK);
}

/******************************************************************************
Desc: Read all data from input stream and write to the output stream.
******************************************************************************/
RCODE FTKAPI FlmWriteToOStream(
	IF_IStream *	pIStream,
	IF_OStream *	pOStream)
{
	RCODE				rc = NE_FLM_OK;
	FLMBYTE			ucBuffer[ 512];
	FLMUINT			uiBufferSize = sizeof( ucBuffer);
	FLMUINT			uiBytesRead;

	for (;;)
	{
		if( RC_BAD( rc = pIStream->read( 
			ucBuffer, uiBufferSize, &uiBytesRead)))
		{
			if( rc != NE_FLM_EOF_HIT)
			{
				if( uiBytesRead)
				{
					(void)pOStream->write( ucBuffer, uiBytesRead);					
				}
				
				goto Exit;
			}

			rc = NE_FLM_OK;

			if( !uiBytesRead)
			{
				goto Exit;
			}
		}

		if( RC_BAD( rc = pOStream->write( ucBuffer, uiBytesRead)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/******************************************************************************
Desc:
******************************************************************************/
RCODE FTKAPI FlmReadFully(
	IF_IStream *	pIStream,
	F_DynaBuf *		pDynaBuf)
{
	RCODE				rc = NE_FLM_OK;
	FLMBYTE			ucBuffer[ 512];
	FLMUINT			uiBufferSize = sizeof( ucBuffer);
	FLMUINT			uiBytesRead;

	for (;;)
	{
		if( RC_BAD( rc = pIStream->read( 
			ucBuffer, uiBufferSize, &uiBytesRead)))
		{
			if( rc != NE_FLM_EOF_HIT)
			{
				if( uiBytesRead)
				{
					(void)pDynaBuf->appendData( ucBuffer, uiBytesRead);
				}
				
				goto Exit;
			}

			rc = NE_FLM_OK;

			if( !uiBytesRead)
			{
				goto Exit;
			}
		}

		if( RC_BAD( rc = pDynaBuf->appendData( ucBuffer, uiBytesRead)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI FlmReadLine(
	IF_IStream *			pIStream,
	F_DynaBuf *				pBuffer)
{
	RCODE						rc = NE_FLM_OK;
	char						ucTmpBuf[ 32];
	
	pBuffer->truncateData( 0);
	
	for( ;;)
	{
		if( RC_BAD( rc = pIStream->read( ucTmpBuf, 1, NULL)))
		{
			goto Exit;
		}
		
		if( ucTmpBuf[ 0] != ASCII_CR)
		{
			if( RC_BAD( rc = pBuffer->appendByte( ucTmpBuf[ 0])))
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = pIStream->read( ucTmpBuf, 1, NULL)))
			{
				goto Exit;
			}
			
			if( ucTmpBuf[ 0] != ASCII_LF)
			{
				rc = RC_SET( NE_FLM_SYNTAX);
				goto Exit;
			}
			
			break;
		}
	}
	
	if( RC_BAD( rc = pBuffer->appendByte( 0)))
	{
		goto Exit;
	}
			
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI f_printf(
	IF_OStream *		pOStream,
	const char *		pszFormatStr, ...)
{
	f_va_list						args;
	F_StreamPrintfClient			printfClient( pOStream);

	f_va_start( args, pszFormatStr);
	f_vprintf( &printfClient, pszFormatStr, &args);
	f_va_end( args);
	return( printfClient.flushBuffer());
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI f_vprintf(
	IF_OStream *		pOStream,
	const char *		pszFormatStr,
	f_va_list *			args)
{
	F_StreamPrintfClient			printfClient( pOStream);

	f_vprintf( &printfClient, pszFormatStr, args);
	return( printfClient.flushBuffer());
}

