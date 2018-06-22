//------------------------------------------------------------------------------
// Desc:	HTTP support
// Tabs:	3
//
// Copyright (c) 2007 Novell, Inc. All Rights Reserved.
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

/****************************************************************************
Desc:
****************************************************************************/
class F_HTTPKeyCompare : public IF_ResultSetCompare
{
	RCODE FTKAPI compare(
		const void *			pvData1,
		FLMUINT					uiLength1,
		const void *			pvData2,
		FLMUINT					uiLength2,
		FLMINT *					piCompare)
	{
		FLMINT					iCompare;
		
		if( uiLength1 < uiLength2)
		{
			if( (iCompare = f_strnicmp( (const char *)pvData1, 
				(const char *)pvData2, uiLength1)) == 0)
			{
				iCompare = 1;
			}
		}
		else if( uiLength1 > uiLength2)
		{
			if( (iCompare = f_strnicmp( (const char *)pvData1, 
				(const char *)pvData2, uiLength2)) == 0)
			{
				iCompare = -1;
			}
		}
		else
		{
			iCompare = f_strnicmp( (const char *)pvData1, 
				(const char *)pvData2, uiLength1);
		}
		
		*piCompare = iCompare;
		return( NE_FLM_OK);
	}
};

/****************************************************************************
Desc:
****************************************************************************/
class F_HTTPHeader : public IF_HTTPHeader
{
public:

	F_HTTPHeader()
	{
		m_pResultSet = NULL;
		m_pszRequestURI = NULL;
		resetHeader();
	}
	
	virtual ~F_HTTPHeader()
	{
		resetHeader();
	}
	
	RCODE FTKAPI readRequestHeader(
		IF_IStream *				pIStream);
	
	RCODE FTKAPI readResponseHeader(
		IF_IStream *				pIStream);
		
	RCODE FTKAPI writeRequestHeader(
		IF_OStream *				pOStream);
		
	RCODE FTKAPI writeResponseHeader(
		IF_OStream *				pOStream);
		
	RCODE FTKAPI getHeaderValue(
		const char *				pszTag,
		F_DynaBuf *					pBuffer);
		
	RCODE FTKAPI setHeaderValue(
		const char *				pszTag,
		const char *				pszValue);
		
	RCODE FTKAPI getHeaderValue(
		const char *				pszTag,
		FLMUINT *					puiValue);
		
	RCODE FTKAPI setHeaderValue(
		const char *				pszTag,
		FLMUINT 						uiValue);
		
	RCODE FTKAPI setMethod(
		eHttpMethod					httpMethod);
		
	eHttpMethod getMethod( void);
		
	FLMUINT FTKAPI getStatusCode( void);
	
	RCODE FTKAPI setStatusCode(
		FLMUINT						uiStatusCode);
	
	RCODE FTKAPI setRequestURI(
		const char *				pszRequestURI);
		
	const char * FTKAPI getRequestURI( void);
		
	void FTKAPI resetHeader( void);
	
private:

	RCODE allocResultSet( void);

	RCODE readHeaderTaggedValues(
		IF_IStream *				pIStream);
	
	RCODE writeHeaderTaggedValues(
		IF_OStream *				pOStream);
	
	IF_BTreeResultSet *			m_pResultSet;
	FLMUINT							m_uiStatusCode;
	FLMUINT							m_uiContentLength;
	eHttpMethod						m_httpMethod;
	char *							m_pszRequestURI;
	char								m_szHttpVersion[ 32];
};

/*****************************************************************************
Desc:
******************************************************************************/
RCODE FTKAPI FlmAllocHTTPHeader( 
	IF_HTTPHeader **			ppHTTPHeader)
{
	if( (*ppHTTPHeader = f_new F_HTTPHeader) == NULL)
	{
		return( RC_SET( NE_FLM_MEM));
	}
	
	return( NE_FLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_HTTPHeader::allocResultSet( void)
{
	RCODE						rc = NE_FLM_OK;
	F_HTTPKeyCompare *	pHTTPCompare = NULL;
	
	f_assert( !m_pResultSet);
	
	if( (pHTTPCompare = f_new F_HTTPKeyCompare) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = FlmAllocBTreeResultSet( pHTTPCompare, &m_pResultSet)))
	{
		goto Exit;
	}
	
Exit:

	if( pHTTPCompare)
	{
		pHTTPCompare->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_HTTPHeader::readHeaderTaggedValues(
	IF_IStream *		pIStream)
{
	RCODE					rc = NE_FLM_OK;
	F_DynaBuf			lineBuf;
	F_DynaBuf			tokenBuf;
	const char *		pszLine;
	const char *		pszTagEnd;
	const char *		pszTagValue;
	
	for( ;;)
	{
		if( RC_BAD( rc = FlmReadLine( pIStream, &lineBuf)))
		{
			goto Exit;
		}
		
		if( *(pszLine = (const char *)lineBuf.getBufferPtr()) == 0)
		{
			break;
		}
		else if( (pszTagEnd = f_strstr( pszLine, ":")) != NULL)
		{
			pszTagValue = pszTagEnd + 1;
			
			while( *pszTagValue && *pszTagValue == ASCII_SPACE)
			{
				pszTagValue++;
			}
			
			if( RC_BAD( rc = m_pResultSet->addEntry( (FLMBYTE *)pszLine, 
				(FLMUINT)(pszTagEnd - pszLine), 
				(FLMBYTE *)pszTagValue, f_strlen( pszTagValue))))
			{
				goto Exit;
			}
		}
		else
		{
			rc = RC_SET( NE_FLM_BAD_HTTP_HEADER);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_HTTPHeader::readResponseHeader(
	IF_IStream *		pIStream)
{
	RCODE					rc = NE_FLM_OK;
	F_DynaBuf			lineBuf;
	F_DynaBuf			tokenBuf;
	const char * 		pszTmp;
	
	resetHeader();
	
	if( RC_BAD( rc = allocResultSet()))
	{
		goto Exit;
	}
	
	// Read the preamble
	
	if( RC_BAD( rc = FlmReadLine( pIStream, &lineBuf)))
	{
		goto Exit;
	}
	
	pszTmp = (const char *)lineBuf.getBufferPtr();
	
	// Verify the preamble
	
	if( f_strncmp( pszTmp, "HTTP", 4) != 0)
	{
		rc = RC_SET( NE_FLM_BAD_HTTP_HEADER);
		goto Exit;
	}
	pszTmp += 4;
	
	if( *pszTmp != ASCII_SLASH)
	{
		rc = RC_SET( NE_FLM_BAD_HTTP_HEADER);
		goto Exit;
	}
	pszTmp++;
	
	// Get the protocol version
	
	tokenBuf.truncateData( 0);
	while( *pszTmp && *pszTmp != ASCII_SPACE)
	{
		if( RC_BAD( rc = tokenBuf.appendByte( *pszTmp)))
		{
			goto Exit;
		}
		
		pszTmp++;
	}
	
	tokenBuf.appendByte( 0);
	
	// Skip the space
	
	if( *pszTmp)
	{
		pszTmp++;
	}

	// Get the status code

	tokenBuf.truncateData( 0);
	while( *pszTmp && *pszTmp != ASCII_SPACE)
	{
		if( RC_BAD( rc = tokenBuf.appendByte( *pszTmp)))
		{
			goto Exit;
		}
		
		pszTmp++;
	}
	
	tokenBuf.appendByte( 0);
	m_uiStatusCode = f_atoud( (const char *)tokenBuf.getBufferPtr());
	
	// Skip the space
	
	if( *pszTmp)
	{
		pszTmp++;
	}
	
	// Get the status message

	tokenBuf.truncateData( 0);
	while( *pszTmp)
	{
		if( RC_BAD( rc = tokenBuf.appendByte( *pszTmp)))
		{
			goto Exit;
		}
		
		pszTmp++;
	}
	
	tokenBuf.appendByte( 0);
	
	// Read the tag values
	
	if( RC_BAD( rc = readHeaderTaggedValues( pIStream)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_HTTPHeader::readRequestHeader(
	IF_IStream *		pIStream)
{
	RCODE					rc = NE_FLM_OK;
	F_DynaBuf			lineBuf;
	F_DynaBuf			tokenBuf;
	const char * 		pszTmp;
	
	resetHeader();
	
	if( RC_BAD( rc = allocResultSet()))
	{
		goto Exit;
	}
	
	// Read the preamble
	
	if( RC_BAD( rc = FlmReadLine( pIStream, &lineBuf)))
	{
		goto Exit;
	}
	
	pszTmp = (const char *)lineBuf.getBufferPtr();
	
	
	if( f_strncmp( pszTmp, "GET ", 4) == 0)
	{
		m_httpMethod = METHOD_GET;
		pszTmp += 4;
	}
	else if( f_strncmp( pszTmp, "PUT ", 4) == 0)
	{
		m_httpMethod = METHOD_PUT;
		pszTmp += 4;
	}
	else if( f_strncmp( pszTmp, "POST ", 5) == 0)
	{
		m_httpMethod = METHOD_POST;
		pszTmp += 5;
	}
	else
	{
		rc = RC_SET( NE_FLM_BAD_HTTP_HEADER);
		goto Exit;
	}

	// Get the request URI

	tokenBuf.truncateData( 0);
	while( *pszTmp && *pszTmp != ASCII_SPACE)
	{
		if( RC_BAD( rc = tokenBuf.appendByte( *pszTmp)))
		{
			goto Exit;
		}
		
		pszTmp++;
	}
	
	tokenBuf.appendByte( 0);
	
	if( RC_BAD( rc = setRequestURI( (const char *)tokenBuf.getBufferPtr())))
	{
		goto Exit;
	}
	
	// Skip the space
	
	if( *pszTmp)
	{
		pszTmp++;
	}

	// Get the protocol version
	
	if( f_strncmp( pszTmp, "HTTP", 4) != 0)
	{
		rc = RC_SET( NE_FLM_BAD_HTTP_HEADER);
		goto Exit;
	}
	pszTmp += 4;
	
	if( *pszTmp != ASCII_SLASH)
	{
		rc = RC_SET( NE_FLM_BAD_HTTP_HEADER);
		goto Exit;
	}
	pszTmp++;
	
	tokenBuf.truncateData( 0);
	while( *pszTmp && *pszTmp != ASCII_SPACE)
	{
		if( RC_BAD( rc = tokenBuf.appendByte( *pszTmp)))
		{
			goto Exit;
		}
		
		pszTmp++;
	}
	
	tokenBuf.appendByte( 0);
	
	// Read the tag values
	
	if( RC_BAD( rc = readHeaderTaggedValues( pIStream)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_HTTPHeader::getHeaderValue(
	const char *				pszTag,
	F_DynaBuf *					pBuffer)
{
	RCODE				rc = NE_FLM_OK;
	FLMBYTE *		pucValue;
	FLMUINT			uiTagLen = f_strlen( pszTag);
	FLMUINT			uiValueLen;
	
	if( !m_pResultSet)
	{
		rc = RC_SET( NE_FLM_NOT_FOUND);
		goto Exit;
	}
	
	if( RC_BAD( rc = m_pResultSet->findEntry( (FLMBYTE *)pszTag, uiTagLen, 
		NULL, 0, &uiValueLen)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pBuffer->allocSpace( uiValueLen + 1, (void **)&pucValue)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = m_pResultSet->getCurrent( (FLMBYTE *)pszTag, uiTagLen, 
		pucValue, uiValueLen, NULL)))
	{
		goto Exit;
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
RCODE FTKAPI F_HTTPHeader::setHeaderValue(
	const char *				pszTag,
	const char *				pszValue)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiTagLen = f_strlen( pszTag);
	FLMUINT			uiValueLen = f_strlen( pszValue);
	
	if( !m_pResultSet)
	{
		if( RC_BAD( rc = allocResultSet()))
		{
			goto Exit;
		}
	}
	
	if( RC_BAD( rc = m_pResultSet->findEntry( (FLMBYTE *)pszTag, uiTagLen, 
		NULL, 0, NULL)))
	{
		if( rc != NE_FLM_NOT_FOUND)
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = m_pResultSet->addEntry( (FLMBYTE *)pszTag, uiTagLen, 
			(FLMBYTE *)pszValue, uiValueLen)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = m_pResultSet->modifyEntry( (FLMBYTE *)pszTag, uiTagLen, 
			(FLMBYTE *)pszValue, uiValueLen)))
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
RCODE FTKAPI F_HTTPHeader::getHeaderValue(
	const char *				pszTag,
	FLMUINT *					puiValue)
{
	RCODE				rc = NE_FLM_OK;
	F_DynaBuf		valueBuf;
	
	if( RC_BAD( rc = getHeaderValue( pszTag, &valueBuf)))
	{
		goto Exit;
	}
	
	*puiValue = f_atoud( (const char *)valueBuf.getBufferPtr());
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_HTTPHeader::setHeaderValue(
	const char *				pszTag,
	FLMUINT 						uiValue)
{
	RCODE				rc = NE_FLM_OK;
	char				ucValueBuf[ 32];
	
	f_sprintf( ucValueBuf, "%u", (unsigned)uiValue);
	
	if( RC_BAD( rc = setHeaderValue( pszTag, ucValueBuf)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT FTKAPI F_HTTPHeader::getStatusCode( void)
{
	return( m_uiStatusCode);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_HTTPHeader::setStatusCode(
	FLMUINT						uiStatusCode)
{
	m_uiStatusCode = uiStatusCode;
	return( NE_FLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_HTTPHeader::setMethod(
	eHttpMethod					httpMethod)
{
	m_httpMethod = httpMethod;
	return( NE_FLM_OK);
}
		
/****************************************************************************
Desc:
****************************************************************************/
eHttpMethod F_HTTPHeader::getMethod( void)
{
	return( m_httpMethod);
}
	
/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI F_HTTPHeader::resetHeader( void)
{
	if( m_pResultSet)
	{
		m_pResultSet->Release();
		m_pResultSet = NULL;
	}
	
	if( m_pszRequestURI)
	{
		f_free( &m_pszRequestURI);
	}
	
	m_uiStatusCode = 0;
	m_uiContentLength = 0;
	m_httpMethod = METHOD_GET;
	f_strcpy( m_szHttpVersion, "1.0");
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_HTTPHeader::setRequestURI(
	const char *	pszRequestURI)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiStrLen = f_strlen( pszRequestURI);
	
	if( m_pszRequestURI)
	{
		f_free( &m_pszRequestURI);
	}

	if( RC_BAD( rc = f_alloc( uiStrLen + 1, &m_pszRequestURI)))
	{
		goto Exit;
	}
	
	f_memcpy( m_pszRequestURI, pszRequestURI, uiStrLen + 1);
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_HTTPHeader::writeRequestHeader(
	IF_OStream *				pOStream)
{
	RCODE			rc = NE_FLM_OK;
	
	if( !m_pszRequestURI)
	{
		rc = RC_SET( NE_FLM_BAD_HTTP_HEADER);
		goto Exit;
	}
	
	// Output the request
	
	switch( m_httpMethod)
	{
		case METHOD_GET:
			f_printf( pOStream, "GET ");
			break;
		
		case METHOD_POST:
			f_printf( pOStream, "POST ");
			break;
			
		case METHOD_PUT:
			f_printf( pOStream, "PUT ");
			break;
	}
	
	f_printf( pOStream, "%s ", m_pszRequestURI);
	f_printf( pOStream, "HTTP/%s\r\n", m_szHttpVersion);
	
	// Output the header fields
	
	if( RC_BAD( rc = writeHeaderTaggedValues( pOStream)))
	{
		goto Exit;
	}
	
	// Terminate

	f_printf( pOStream, "\r\n");
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_HTTPHeader::writeResponseHeader(
	IF_OStream *				pOStream)
{
	RCODE			rc = NE_FLM_OK;
	
	// Output the preamble and status
	
	f_printf( pOStream, "HTTP/%s ", m_szHttpVersion);
	f_printf( pOStream, "%u ", (unsigned)m_uiStatusCode);
	f_printf( pOStream, "%s\r\n", FlmGetHTTPStatusString( m_uiStatusCode));
	
	// Output the header fields
	
	if( RC_BAD( rc = writeHeaderTaggedValues( pOStream)))
	{
		goto Exit;
	}
	
	// Terminate

	f_printf( pOStream, "\r\n");
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_HTTPHeader::writeHeaderTaggedValues(
	IF_OStream *			pOStream)
{
	RCODE			rc = NE_FLM_OK;
	FLMBYTE		ucTag[ FLM_MAX_KEY_SIZE];
	FLMBYTE 		ucValue[ 512];
	FLMUINT		uiTagLength;
	FLMUINT		uiValueLength;
	FLMUINT		uiFieldCount;

	if( !m_pResultSet)
	{
		goto Exit;
	}
		
	uiTagLength = 0;
	uiFieldCount = 0;
	
	for( ;;)
	{
		if( !uiFieldCount)
		{
			rc = m_pResultSet->getFirst( ucTag, sizeof( ucTag), 
				&uiTagLength, NULL, 0, &uiValueLength);
		}
		else
		{
			rc = m_pResultSet->getNext( ucTag, sizeof( ucTag), 
				&uiTagLength, NULL, 0, &uiValueLength);
		}
		
		if( RC_BAD( rc))
		{
			if( rc == NE_FLM_EOF_HIT)
			{
				rc = NE_FLM_OK;
				break;
			}
			
			goto Exit;
		}
		
		if( uiValueLength >= sizeof( ucValue))
		{
			rc = RC_SET( NE_FLM_BAD_HTTP_HEADER);
			goto Exit;
		}
		
		if( RC_BAD( rc = m_pResultSet->getCurrent( ucTag, uiTagLength, 
			ucValue, uiValueLength, NULL)))
		{
			goto Exit;
		}
		
		ucTag[ uiTagLength] = 0;
		ucValue[ uiValueLength] = 0;
		
		f_printf( pOStream, "%s: %s\r\n", ucTag, ucValue);
		uiFieldCount++;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
const char * FTKAPI FlmGetHTTPStatusString( 
	FLMUINT				uiStatusCode)
{
	const char *		pszStatusCode;
	
	switch( uiStatusCode)
	{
		case FLM_HTTP_STATUS_CONTINUE:
			pszStatusCode = "Continue";
			break;
		
		case FLM_HTTP_STATUS_SWITCHING_PROTOCOLS:
			pszStatusCode = "Switching Protocols";
			break;
			
		case FLM_HTTP_STATUS_PROCESSING: 
			pszStatusCode = "Processing";
			break;
		
		case FLM_HTTP_STATUS_OK:
			pszStatusCode = "OK";
			break;
		
		case FLM_HTTP_STATUS_CREATED:
			pszStatusCode = "Created";
			break;
		
		case FLM_HTTP_STATUS_ACCEPTED:
			pszStatusCode = "Accepted";
			break;
		
		case FLM_HTTP_STATUS_NON_AUTH_INFO:
			pszStatusCode = "Non-Authoritative Information";
			break;
		
		case FLM_HTTP_STATUS_NO_CONTENT:
			pszStatusCode = "No Content";
			break;
		
		case FLM_HTTP_STATUS_RESET_CONTENT:
			pszStatusCode = "Reset Content";
			break;
		
		case FLM_HTTP_STATUS_PARTIAL_CONTENT:
			pszStatusCode = "Partial Content";
			break;
			
		case FLM_HTTP_STATUS_MULTI_STATUS:
			pszStatusCode = "Multi-Status";
			break;
		
		case FLM_HTTP_STATUS_MULTIPLE_CHOICES:
			pszStatusCode = "Multiple Choices";
			break;
		
		case FLM_HTTP_STATUS_MOVED_PERMANENTLY:
			pszStatusCode = "Moved Permanently";
			break;
		
		case FLM_HTTP_STATUS_FOUND:
			pszStatusCode = "Found";
			break;
		
		case FLM_HTTP_STATUS_SEE_OTHER:
			pszStatusCode = "See Other";
			break;
		
		case FLM_HTTP_STATUS_NOT_MODIFIED:
			pszStatusCode = "Not Modified";
			break;
		
		case FLM_HTTP_STATUS_USE_PROXY:
			pszStatusCode = "Use Proxy";
			break;
		
		case FLM_HTTP_STATUS_TEMPORARY_REDIRECT:
			pszStatusCode = "Temporary Redirect";
			break;
			
		case FLM_HTTP_STATUS_BAD_REQUEST:
			pszStatusCode = "Bad Request";
			break;
		
		case FLM_HTTP_STATUS_UNAUTHORIZED:
			pszStatusCode = "Unauthorized";
			break;
		
		case FLM_HTTP_STATUS_PAYMENT_REQUIRED:
			pszStatusCode = "Payment Required";
			break;
		
		case FLM_HTTP_STATUS_FORBIDDEN:
			pszStatusCode = "Forbidden";
			break;
		
		case FLM_HTTP_STATUS_NOT_FOUND:
			pszStatusCode = "Not Found";
			break;
		
		case FLM_HTTP_STATUS_METHOD_NOT_ALLOWED:
			pszStatusCode = "Method Not Allowed";
			break;
		
		case FLM_HTTP_STATUS_NOT_ACCEPTABLE:
			pszStatusCode = "Not Acceptable";
			break;
		
		case FLM_HTTP_STATUS_PROXY_AUTH_REQUIRED:
			pszStatusCode = "Proxy Authentication Required";
			break;
		
		case FLM_HTTP_STATUS_REQUEST_TIMEOUT:
			pszStatusCode = "Request Timeout";
			break;
		
		case FLM_HTTP_STATUS_CONFLICT:
			pszStatusCode = "Conflict";
			break;
		
		case FLM_HTTP_STATUS_GONE:
			pszStatusCode = "Gone";
			break;
		
		case FLM_HTTP_STATUS_LENGTH_REQUIRED:
			pszStatusCode = "Length Required";
			break;
		
		case FLM_HTTP_STATUS_PRECONDITION_FAILED:
			pszStatusCode = "Precondition Failed";
			break;
		
		case FLM_HTTP_STATUS_ENTITY_TOO_LARGE:
			pszStatusCode = "Request Entity Too Large";
			break;
		
		case FLM_HTTP_STATUS_URI_TOO_LONG:
			pszStatusCode = "Request-URI Too Long";
			break;
		
		case FLM_HTTP_STATUS_UNSUPPORTED_MEDIA_TYPE:
			pszStatusCode = "Unsupported Media Type";
			break;
		
		case FLM_HTTP_STATUS_RANGE_NOT_SATISFIABLE:
			pszStatusCode = "Requested Range Not Satisfiable";
			break;
		
		case FLM_HTTP_STATUS_EXPECTATION_FAILED:
			pszStatusCode = "Expectation Failed";
			break;
		
		case FLM_HTTP_STATUS_INTERNAL_SERVER_ERROR:
			pszStatusCode = "Internal Server Error";
			break;
		
		case FLM_HTTP_STATUS_NOT_IMPLEMENTED:
			pszStatusCode = "Not Implemented";
			break;
		
		case FLM_HTTP_STATUS_BAD_GATEWAY:
			pszStatusCode = "Bad Gateway";
			break;
		
		case FLM_HTTP_STATUS_SERVICE_UNAVAILABLE:
			pszStatusCode = "Service Unavailable";
			break;
		
		case FLM_HTTP_STATUS_GATEWAY_TIMEOUT:
			pszStatusCode = "Gateway Timeout";
			break;
		
		case FLM_HTTP_STATUS_VERSION_NOT_SUPPORTED:
			pszStatusCode = "HTTP Version Not Supported";
			break;
		
		default:
			pszStatusCode = "Undefined Error";
			break;
	}
	
	return( pszStatusCode);	
}

