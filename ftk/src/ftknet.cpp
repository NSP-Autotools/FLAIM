//------------------------------------------------------------------------------
// Desc:
// Tabs: 3
//
// Copyright (c) 1998, 2007 Novell, Inc. All Rights Reserved.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; version 2.1
// of the License.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the GNU
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

FSTATIC RCODE f_socketPeek(
	int					iSocket,
	FLMUINT				uiTimeoutVal,
	FLMBOOL				bPeekRead);
	
FSTATIC void f_closeSocket(
	SOCKET *				piSocket);
	
/****************************************************************************
Desc:
****************************************************************************/
class	F_TCPListener : public IF_TCPListener
{
public:

	F_TCPListener()
	{
		m_bBound = FALSE;
		m_iSocket = INVALID_SOCKET;
	}
	
	virtual ~F_TCPListener()
	{
		if( m_iSocket != INVALID_SOCKET)
		{
			f_closeSocket( &m_iSocket);
		}
	}

	RCODE	FTKAPI bind(
		FLMUINT					uiBindPort,
		FLMBYTE *				pucBindAddr = NULL);

	RCODE FTKAPI connectClient(
		IF_TCPIOStream **	ppClientStream,
		FLMUINT				uiTimeout = 3);
		
private:

	FLMBOOL			m_bBound;
	SOCKET			m_iSocket;
};

/****************************************************************************
Desc:
****************************************************************************/
class	F_TCPIOStream : public IF_TCPIOStream
{
public:

	#if defined( FLM_WIN) && _MSC_VER < 1300
		using IF_IStream::operator delete;
	#endif

	F_TCPIOStream( void);
	
	virtual ~F_TCPIOStream( void);

	RCODE FTKAPI openStream(
		const char *	pucHostAddress,
		FLMUINT			uiPort,
		FLMUINT			uiFlags,
		FLMUINT			uiConnectTimeout);

	RCODE FTKAPI openStream(
		int				iSocket,
		FLMUINT			uiFlags);
		
	RCODE FTKAPI read(
		void *			pvBuffer,
		FLMUINT			uiBytesToRead,
		FLMUINT *		puiBytesRead);
		
	RCODE FTKAPI write(
		const void *	pvBuffer,
		FLMUINT			uiBytesToWrite,
		FLMUINT *		puiBytesWritten);

	FINLINE RCODE FTKAPI socketPeekWrite(
		FLMUINT		uiTimeOut)
	{
		return( f_socketPeek( m_iSocket, uiTimeOut, FALSE));
	}

	FINLINE RCODE FTKAPI f_socketPeekRead( 
		FLMUINT		uiTimeOut)
	{
		return( f_socketPeek( m_iSocket, uiTimeOut, TRUE));
	};

	FINLINE const char * FTKAPI getLocalHostName( void)
	{
		getLocalInfo();
		return( (const char *)m_pszName);
	};

	FINLINE const char * FTKAPI getLocalHostAddress( void)
	{
		getLocalInfo();
		return( (const char *)m_pszIp);
	};

	FINLINE const char * FTKAPI getPeerHostName( void)
	{
		getRemoteInfo();
		return( (const char *)m_pszPeerName);
	};

	FINLINE const char * FTKAPI getPeerHostAddress( void)
	{
		getRemoteInfo();
		return( (const char *)m_pszPeerIp);
	};

	RCODE FTKAPI readNoWait(
		void *			pvBuffer,
		FLMUINT			uiCount,
		FLMUINT *		puiReadRead);

	RCODE FTKAPI readAll(
		void *			pvBuffer,
		FLMUINT			uiCount,
		FLMUINT *		puiBytesRead);

	void FTKAPI setIOTimeout(
		FLMUINT			uiSeconds);

	RCODE FTKAPI closeStream( void);

private:

	RCODE getLocalInfo( void);
	
	RCODE getRemoteInfo( void);

#ifndef FLM_UNIX
	WSADATA			m_wsaData;
#endif
	FLMBOOL			m_bInitialized;
	SOCKET			m_iSocket;
	FLMUINT			m_uiIOTimeout;
	FLMBOOL			m_bConnected;
	char				m_pszIp[ 256];
	char				m_pszName[ 256];
	char				m_pszPeerIp[ 256];
	char				m_pszPeerName[ 256];
	unsigned long	m_ulRemoteAddr;
};
	
/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_OPENSSL
class F_SSLIOStream : public IF_SSLIOStream
{
public:

	F_SSLIOStream()
	{
		m_pBio = NULL;
		m_pContext = NULL;
		m_pSSL = NULL;
		m_pPeerCertificate = NULL;
		m_pszPeerCertText = NULL;
		m_szPeerName[ 0] = 0;
	}
	
	virtual ~F_SSLIOStream()
	{
		closeStream();
	}
	
	RCODE FTKAPI openStream(
		const char *			pszHost,
		FLMUINT					uiPort = 443,
		FLMUINT					uiFlags = 0);
	
	RCODE FTKAPI read(
		void *					pvBuffer,
		FLMUINT					uiBytesToRead,
		FLMUINT *				puiBytesRead = NULL);
		
	RCODE FTKAPI write(
		const void *			pvBuffer,
		FLMUINT					uiBytesToWrite,
		FLMUINT *				puiBytesWritten = NULL);
		
	const char * FTKAPI getPeerCertificateText( void);
		
	RCODE FTKAPI closeStream( void);
	
private:

	BIO *							m_pBio;
	SSL_CTX *					m_pContext;
	SSL *							m_pSSL;
	X509 *						m_pPeerCertificate;
	char *						m_pszPeerCertText;
	char							m_szPeerName[ 256];
};
#endif

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE	F_TCPListener::bind(
	FLMUINT					uiBindPort,
	FLMBYTE *				pucBindAddr)
{
	RCODE						rc = NE_FLM_OK;
	struct sockaddr_in 	address;
	int						iTmp;

	if( m_bBound)
	{
		rc = RC_SET( NE_FLM_SOCKET_FAIL);
		goto Exit;
	}

	if( (m_iSocket = socket( AF_INET, 
		SOCK_STREAM, IPPROTO_TCP)) == INVALID_SOCKET)
	{
		rc = RC_SET( NE_FLM_SOCKET_FAIL);
		goto Exit;
	}

	f_memset( &address, 0, sizeof( address));
	address.sin_family = AF_INET;
	
	if( !pucBindAddr)
	{
		address.sin_addr.s_addr = htonl( INADDR_ANY);
	}
	else
	{
		address.sin_addr.s_addr = inet_addr( (char *)pucBindAddr);
	}
	address.sin_port = htons( (u_short)uiBindPort);

	// Bind to the address and port

	if( ::bind( m_iSocket, (struct sockaddr *)&address, sizeof( address)) != 0)
	{
		rc = RC_SET( NE_FLM_BIND_FAIL);
		goto Exit;
	}

	// listen() prepares a socket to accept a connection and specifies a
	// queue limit for incoming connections.  The accept() accepts the 
	// connection.  Listen returns immediatly.
	
#ifdef FLM_NLM
	if( listen( m_iSocket, 32) < 0)
#endif
	{
		if( listen( m_iSocket, 5) < 0)
		{
			rc = RC_SET( NE_FLM_LISTEN_FAIL);
			goto Exit;
		}
	}

	// Disable Nagel's algorithm

	iTmp = 1;
	if( (setsockopt( m_iSocket, IPPROTO_TCP, TCP_NODELAY, (char *)&iTmp,
		(unsigned)sizeof( iTmp) )) < 0)
	{
		rc = RC_SET( NE_FLM_SOCKET_SET_OPT_FAIL);
		goto Exit;
	}
	
	m_bBound = TRUE;

Exit:

	if( RC_BAD( rc) && m_iSocket != INVALID_SOCKET)
	{
		f_closeSocket( &m_iSocket);		
	}

	return( rc);
}
	
/*****************************************************************************
Desc:
*****************************************************************************/
RCODE F_TCPListener::connectClient(
	IF_TCPIOStream **		ppClientStream,
	FLMUINT					uiConnectTimeout)
{
	RCODE						rc = NE_FLM_OK;
	SOCKET					iSocket = INVALID_SOCKET;
	struct sockaddr_in 	address;
	F_TCPIOStream *		pClientStream = NULL;
#if defined( FLM_UNIX) && !defined( FLM_OSX)
	socklen_t				iAddrLen;
#else
	int						iAddrLen;
#endif

	if( !m_bBound)
	{
		rc = RC_SET( NE_FLM_BIND_FAIL);
		goto Exit;
	}

	if( RC_BAD( rc = f_socketPeek( m_iSocket, uiConnectTimeout, TRUE)))
	{
		goto Exit;
	}

	iAddrLen = sizeof( struct sockaddr);
	
	if( (iSocket = accept( m_iSocket, 
		(struct sockaddr *)&address, &iAddrLen)) == INVALID_SOCKET)
	{
		rc = RC_SET( NE_FLM_ACCEPT_FAIL);
		goto Exit;
	}
	
	if( (pClientStream = f_new F_TCPIOStream) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pClientStream->openStream( iSocket, 0)))
	{
		goto Exit;
	}
	
	iSocket = INVALID_SOCKET;
	*ppClientStream = pClientStream;
	pClientStream = NULL;	

Exit:

	if( pClientStream)
	{
		pClientStream->Release();
	}

	if( iSocket != INVALID_SOCKET)
	{
		f_closeSocket( &iSocket);
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE FTKAPI FlmOpenTCPIOStream(
	const char *			pszHost,
	FLMUINT					uiPort,
	FLMUINT					uiFlags,
	FLMUINT					uiConnectTimeout,
	IF_IOStream **			ppIOStream)
{
	RCODE							rc = NE_FLM_OK;
	F_TCPIOStream *			pIOStream = NULL;
	
	if( (pIOStream = f_new F_TCPIOStream) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pIOStream->openStream( pszHost, uiPort, 
		uiFlags, uiConnectTimeout)))
	{
		goto Exit;
	}
	
	*ppIOStream = pIOStream;
	pIOStream = NULL;
	
Exit:

	if( pIOStream)
	{
		pIOStream->Release();
	}
	
	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE FTKAPI FlmAllocSSLIOStream( 
	IF_IOStream **			ppIOStream)
{
#ifdef FLM_OPENSSL

	if( (*ppIOStream = f_new F_SSLIOStream) == NULL)
	{
		return( RC_SET( NE_FLM_MEM));
	}
	
	return( NE_FLM_OK);
	
#else

	F_UNREFERENCED_PARM( ppIOStream);
	return( RC_SET( NE_FLM_NOT_IMPLEMENTED));
	
#endif
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE FTKAPI FlmOpenSSLIOStream(
	const char *			pszHost,
	FLMUINT					uiPort,
	FLMUINT					uiFlags,
	IF_IOStream **			ppIOStream)
{
#ifdef FLM_OPENSSL

	RCODE							rc = NE_FLM_OK;
	F_SSLIOStream *			pIOStream = NULL;
	
	if( (pIOStream = f_new F_SSLIOStream) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pIOStream->openStream( pszHost, uiPort, uiFlags)))
	{
		goto Exit;
	}
	
	*ppIOStream = pIOStream;
	pIOStream = NULL;
	
Exit:

	if( pIOStream)
	{
		pIOStream->Release();
	}
	
	return( rc);
	
#else

	F_UNREFERENCED_PARM( pszHost);
	F_UNREFERENCED_PARM( uiPort);
	F_UNREFERENCED_PARM( uiFlags);
	F_UNREFERENCED_PARM( ppIOStream);
	
	return( RC_SET( NE_FLM_NOT_IMPLEMENTED));
	
#endif
}

/********************************************************************
Desc:
*********************************************************************/
F_TCPIOStream::F_TCPIOStream( void)
{
	m_pszIp[ 0] = 0;
	m_pszName[ 0] = 0;
	m_pszPeerIp[ 0] = 0;
	m_pszPeerName[ 0] = 0;
	m_uiIOTimeout = 30;
	m_iSocket = INVALID_SOCKET;
	m_ulRemoteAddr = 0;
	m_bInitialized = FALSE;
	m_bConnected = FALSE;

#ifndef FLM_UNIX
	if( !WSAStartup( MAKEWORD( 2, 0), &m_wsaData))
	{
		m_bInitialized = TRUE;
	}
#endif
}

/********************************************************************
Desc:
*********************************************************************/
F_TCPIOStream::~F_TCPIOStream( void)
{
	if( m_bConnected)
	{
		closeStream();
	}

#ifndef FLM_UNIX
	if( m_bInitialized)
	{
		WSACleanup();
	}
#endif
}

/********************************************************************
Desc: Opens a new connection
*********************************************************************/
RCODE F_TCPIOStream::openStream(
	const char  *		pucHostName,
	FLMUINT				uiPort,
	FLMUINT,				// uiFlags,
	FLMUINT				uiConnectTimeout)
{
	RCODE						rc = NE_FLM_OK;
	FLMINT					iSockErr;
	FLMINT    				iTries;
	FLMINT					iMaxTries = 5;
	struct sockaddr_in	address;
	struct hostent *		pHostEntry;
	unsigned long			ulIPAddr;
	int						iTmp;

	f_assert( !m_bConnected);
	m_iSocket = INVALID_SOCKET;

	if( pucHostName && pucHostName[ 0] != '\0')
	{
		ulIPAddr = inet_addr( (char *)pucHostName);
		if( ulIPAddr == (unsigned long)(-1))
		{
			pHostEntry = gethostbyname( (char *)pucHostName);

			if( !pHostEntry)
			{
				rc = RC_SET( NE_FLM_NOIP_ADDR);
				goto Exit;
			}
			else
			{
				ulIPAddr = *((unsigned long *)pHostEntry->h_addr);
			}

		}
	}
	else
	{
		ulIPAddr = inet_addr( (char *)"127.0.0.1");
	}

	// Fill in the Socket structure with family type

	f_memset( (char *)&address, 0, sizeof( struct sockaddr_in));
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = (unsigned)ulIPAddr;
	address.sin_port = htons( (unsigned short)uiPort);
	
	// Allocate a socket, then attempt to connect to it!

	if( (m_iSocket = socket( AF_INET, 
		SOCK_STREAM, IPPROTO_TCP)) == INVALID_SOCKET)
	{
		rc = RC_SET( NE_FLM_SOCKET_FAIL);
		goto Exit;
	}

	// Now attempt to connect with the specified partner host, 
	// time-out if connection doesn't complete within alloted time
	
#ifdef FLM_WIN

	if( uiConnectTimeout)
	{
		if ( uiConnectTimeout < 5 )
		{
			iMaxTries = (iMaxTries * uiConnectTimeout) / 5;
			uiConnectTimeout = 5;
		}
	}
	else
	{
		iMaxTries = 1;
	}
#endif	

	for( iTries = 0; iTries < iMaxTries; iTries++ )
	{			
		iSockErr = 0;
		if( connect( m_iSocket, (struct sockaddr *)((void *)&address),
			(unsigned)sizeof(struct sockaddr)) >= 0)
		{
			break;
		}

		#ifndef FLM_UNIX
			iSockErr = WSAGetLastError();
		#else
			iSockErr = errno;
		#endif

	#ifdef FLM_WIN

		// In WIN, we sometimes get WSAEINVAL when, if we keep
		// trying, we will eventually connect.  Therefore,
		// here we'll treat WSAEINVAL as EINPROGRESS.

		if( iSockErr == WSAEINVAL)
		{
			f_closeSocket( &m_iSocket);
			
			if( (m_iSocket = socket( AF_INET, 
				SOCK_STREAM, IPPROTO_TCP)) == INVALID_SOCKET)
			{
				rc = RC_SET( NE_FLM_SOCKET_FAIL);
				goto Exit;
			}
		#if defined( FLM_WIN) || defined( FLM_NLM)
			iSockErr = WSAEINPROGRESS;
		#else
			iSockErr = EINPROGRESS;
		#endif
			continue;
		}
	#endif

	#if defined( FLM_WIN) || defined( FLM_NLM)
		if( iSockErr == WSAEISCONN )
	#else
		if( iSockErr == EISCONN )
	#endif
		{
			break;
		}
	#if defined( FLM_WIN) || defined( FLM_NLM)
		else if( iSockErr == WSAEWOULDBLOCK)
	#else
		else if( iSockErr == EWOULDBLOCK)
	#endif
		{
			// Let's wait a split second to give the connection
         // request a chance. 

			f_sleep( 100 );
			continue;
		}
	#if defined( FLM_WIN) || defined( FLM_NLM)
		else if( iSockErr == WSAEINPROGRESS)
	#else
		else if( iSockErr == EINPROGRESS)
	#endif
		{
			if( RC_OK( rc = f_socketPeek( m_iSocket, uiConnectTimeout, FALSE)))
			{
				// Let's wait a split second to give the connection
            // request a chance. 

				f_sleep( 100 );
				continue;
			}
		}
		
		rc = RC_SET( NE_FLM_CONNECT_FAIL);
		goto Exit;
	}

	// Disable Nagel's algorithm

	iTmp = 1;
	if( (setsockopt( m_iSocket, IPPROTO_TCP, TCP_NODELAY, (char *)&iTmp,
		(unsigned)sizeof( iTmp) )) < 0)
	{
		rc = RC_SET( NE_FLM_SOCKET_SET_OPT_FAIL);
		goto Exit;
	}
	
	m_bConnected = TRUE;

Exit:

	if( RC_BAD( rc))
	{
		f_closeSocket( &m_iSocket);
	}
	
	return( rc);
}

/********************************************************************
Desc: Opens a new connection
*********************************************************************/
RCODE F_TCPIOStream::openStream(
	int					iSocket,
	FLMUINT)				// uiFlags
{
	RCODE						rc = NE_FLM_OK;
	int						iTmp;

	f_assert( !m_bConnected);
	f_assert( m_iSocket == INVALID_SOCKET);

	// Disable Nagel's algorithm

	iTmp = 1;
	if( (setsockopt( iSocket, IPPROTO_TCP, TCP_NODELAY, (char *)&iTmp,
		(unsigned)sizeof( iTmp) )) < 0)
	{
		rc = RC_SET( NE_FLM_SOCKET_SET_OPT_FAIL);
		goto Exit;
	}

	m_iSocket = iSocket;	
	m_bConnected = TRUE;

Exit:

	return( rc);
}

/********************************************************************
Desc: Gets information about the local host machine.
*********************************************************************/
RCODE F_TCPIOStream::getLocalInfo( void)
{
	RCODE						rc = NE_FLM_OK;
	struct hostent *		pHostEnt;
	FLMUINT32				ui32IPAddr;

	m_pszIp[ 0] = 0;
	m_pszName[ 0] = 0;

	if( !m_pszName[ 0])
	{
		if( gethostname( m_pszName, (unsigned)sizeof( m_pszName)))
		{
			rc = RC_SET( NE_FLM_SOCKET_FAIL);
			goto Exit;
		}
	}

	if( !m_pszIp[ 0] && (pHostEnt = gethostbyname( m_pszName)) != NULL)
	{
		ui32IPAddr = (FLMUINT32)(*((unsigned long *)pHostEnt->h_addr));
		if( ui32IPAddr != (FLMUINT32)-1)
		{
			struct in_addr			InAddr;

			InAddr.s_addr = ui32IPAddr;
			f_strcpy( m_pszIp, inet_ntoa( InAddr));
		}
	}

Exit:

	return( rc);
}

/********************************************************************
Desc: Gets information about the remote machine.
*********************************************************************/
RCODE F_TCPIOStream::getRemoteInfo( void)
{
	RCODE						rc = NE_FLM_OK;
	struct sockaddr_in 	SockAddrIn;
	char *					InetAddr = NULL;
	struct hostent	*		HostsName;

	m_pszPeerIp[ 0] = 0;
	m_pszPeerName[ 0] = 0;

	SockAddrIn.sin_addr.s_addr = (unsigned)m_ulRemoteAddr;

	InetAddr = inet_ntoa( SockAddrIn.sin_addr);
	f_strcpy( m_pszPeerIp, InetAddr);
	
	// Try to get the peer's host name by looking up his IP
	// address.

	HostsName = gethostbyaddr( (char *)&SockAddrIn.sin_addr.s_addr,
		(unsigned)sizeof( unsigned long), AF_INET );

	if( HostsName != NULL)
	{
		f_strcpy( m_pszPeerName, (char*) HostsName->h_name );
	}
	else
	{
		if( !InetAddr)
		{
			InetAddr = inet_ntoa( SockAddrIn.sin_addr);
		}
		
		f_strcpy( m_pszPeerName, InetAddr);
	}
	
	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
RCODE FTKAPI F_TCPIOStream::write(
	const void *	pucBuffer,
	FLMUINT			uiBytesToWrite,
	FLMUINT *		puiBytesWritten)
{
	RCODE				rc = NE_FLM_OK;
	FLMINT			iRetryCount = 0;
	FLMINT			iBytesWritten = 0;

	if( m_iSocket == INVALID_SOCKET)
	{
		rc = RC_SET( NE_FLM_CONNECT_FAIL);
		goto Exit;
	}

	f_assert( pucBuffer && uiBytesToWrite);

Retry:

	if( puiBytesWritten)
	{
		*puiBytesWritten = 0;
	}
	
	if( RC_OK( rc = f_socketPeek( m_iSocket, m_uiIOTimeout, FALSE)))
	{
		iBytesWritten = send( m_iSocket, 
					(char *)pucBuffer, (int)uiBytesToWrite, 0);
		
		switch( iBytesWritten)
		{
			case -1:
			{
				if( puiBytesWritten)
				{
					*puiBytesWritten = 0;
				}
				
				rc = RC_SET( NE_FLM_SOCKET_WRITE_FAIL);
				break;
			}

			case 0:
			{
				rc = RC_SET( NE_FLM_SOCKET_DISCONNECT);
				break;
			}

			default:
			{
				if( puiBytesWritten)
				{
					*puiBytesWritten = (FLMUINT)iBytesWritten;
				}
				
				break;
			}
		}
	}

	if( RC_BAD( rc) && rc != NE_FLM_SOCKET_WRITE_TIMEOUT)
	{
#ifndef FLM_UNIX
		FLMINT iSockErr = WSAGetLastError();
#else
		FLMINT iSockErr = errno;
#endif

#if defined( FLM_WIN) || defined( FLM_NLM)
		if( iSockErr == WSAECONNABORTED)
#else
		if( iSockErr == ECONNABORTED)
#endif
		{
			rc = RC_SET( NE_FLM_SOCKET_DISCONNECT);
		}
#if defined( FLM_WIN) || defined( FLM_NLM)
		else if( iSockErr == WSAEWOULDBLOCK && iRetryCount < 5)
#else
		else if( iSockErr == EWOULDBLOCK && iRetryCount < 5)
#endif
		{
			iRetryCount++;
			f_sleep( (FLMUINT)(100 * iRetryCount));
			goto Retry;
		}
	}
	
Exit:

	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
RCODE FTKAPI F_TCPIOStream::read(
	void *		pucBuffer,
   FLMUINT		uiBytesToWrite,
	FLMUINT *	puiBytesRead)
{
	RCODE			rc = NE_FLM_OK;
	FLMINT		iReadCnt = 0;

	f_assert( m_bConnected && pucBuffer && uiBytesToWrite);

	if( RC_OK( rc = f_socketPeek( m_iSocket, m_uiIOTimeout, TRUE)))
	{
		iReadCnt = (FLMINT)recv( m_iSocket, 
			(char *)pucBuffer, (int)uiBytesToWrite, 0);
			
		switch ( iReadCnt)
		{
			case -1:
			{
				iReadCnt = 0;
#if defined( FLM_WIN) || defined( FLM_NLM)
				if ( WSAGetLastError() == WSAECONNRESET)
#else
				if( errno == ECONNRESET)
#endif
				{
					rc = RC_SET( NE_FLM_SOCKET_DISCONNECT);
				}
				else
				{
					rc = RC_SET( NE_FLM_SOCKET_READ_FAIL);
				}
				break;
			}

			case 0:
			{
				rc = RC_SET( NE_FLM_SOCKET_DISCONNECT);
				break;
			}

			default:
			{
				break;
			}
		}
	}

	if( puiBytesRead)
	{
		*puiBytesRead = (FLMUINT)iReadCnt;
	}

	return( rc);
}

/********************************************************************
Desc: Reads data from the connection - Timeout valkue is zero, no error
      is generated if timeout occurs.
*********************************************************************/
RCODE F_TCPIOStream::readNoWait(
	void *			pvBuffer,
   FLMUINT			uiBytesToRead,
	FLMUINT *		puiBytesRead)
{
	RCODE			rc = NE_FLM_OK;
	FLMINT		iReadCnt = 0;

	f_assert( m_bConnected && pvBuffer && uiBytesToRead);

	if( puiBytesRead)
	{
		*puiBytesRead = 0;
	}

	if( RC_OK( rc = f_socketPeek( m_iSocket, (FLMUINT)0, TRUE)))
	{
		iReadCnt = recv( m_iSocket, (char *)pvBuffer, (int)uiBytesToRead, 0);
		switch ( iReadCnt)
		{
			case -1:
			{
				*puiBytesRead = 0;
#if defined( FLM_WIN) || defined( FLM_NLM)
				if ( WSAGetLastError() == WSAECONNRESET)
#else
				if( errno == ECONNRESET)
#endif
				{
					rc = RC_SET( NE_FLM_SOCKET_DISCONNECT);
				}
				else
				{
					rc = RC_SET( NE_FLM_SOCKET_READ_FAIL);
				}
				goto Exit;
			}

			case 0:
			{
				rc = RC_SET( NE_FLM_SOCKET_DISCONNECT);
				goto Exit;
			}

			default:
			{
				break;
			}
		}
	}
	else if (rc == NE_FLM_SOCKET_READ_TIMEOUT)
	{
		rc = NE_FLM_OK;
	}

	if( puiBytesRead)
	{
		*puiBytesRead = (FLMUINT)iReadCnt;
	}

Exit:

	return( rc);
}

/********************************************************************
Desc: Reads data and does not return until all requested data has
		been read or a timeout error has been encountered.
*********************************************************************/
RCODE F_TCPIOStream::readAll(
	void *			pvBuffer,
	FLMUINT			uiBytesToRead,
   FLMUINT *		puiBytesRead)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiToRead = 0;
	FLMUINT		uiHaveRead = 0;
	FLMUINT		uiPartialCnt;
	FLMBYTE *	pucBuffer = (FLMBYTE *)pvBuffer;

	f_assert( m_bConnected && pvBuffer && uiBytesToRead);

	uiToRead = uiBytesToRead;
	while( uiToRead)
	{
		if( RC_BAD( rc = read( pucBuffer, uiToRead, &uiPartialCnt)))
		{
			goto Exit;
		}

		pucBuffer += uiPartialCnt;
		uiHaveRead += uiPartialCnt;
		uiToRead = (FLMUINT)(uiBytesToRead - uiHaveRead);

		if( puiBytesRead)
		{
			*puiBytesRead = uiHaveRead;
		}
	}

Exit:

	return( rc);
}

/********************************************************************
Desc: Closes any open connections
*********************************************************************/
RCODE FTKAPI F_TCPIOStream::closeStream( void)
{
	if( m_iSocket == INVALID_SOCKET)
	{
		goto Exit;
	}
	
	f_closeSocket( &m_iSocket);	

Exit:

	m_bConnected = FALSE;
	return( NE_FLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI F_TCPIOStream::setIOTimeout(
	FLMUINT			uiSeconds)
{
	m_uiIOTimeout = uiSeconds;
}
	
/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_OPENSSL
RCODE FTKAPI F_SSLIOStream::openStream(
	const char *			pszHost,
	FLMUINT					uiPort,
	FLMUINT					uiFlags)
{
	RCODE						rc = NE_FLM_OK;
	char						szPort[ 32];
	X509_NAME *				pPeerName;
	EVP_PKEY *				pPublicKey = NULL;
	BIO *						pMemBIO = NULL;
	FLMUINT					uiCertTextLen;
	const char *			pszCertText;
	
	if( !uiPort)
	{
		uiPort = 443;
	}
	
	// Setup the connection to use SSLv2, SSLv3, or TLSv1 depending on
	// the capabilities of the peer
	
	if( (m_pContext = SSL_CTX_new( SSLv23_client_method())) == NULL)
	{
		rc = RC_SET( NE_FLM_CONNECT_FAIL);
		goto Exit;
	}
	
	// Set the default path for verifying certificates
	
	if( !SSL_CTX_set_default_verify_paths( m_pContext))
	{
		rc = RC_SET( NE_FLM_CONNECT_FAIL);
		goto Exit;
	}
	
	// Configure the BIO

	if( (m_pBio = BIO_new_ssl_connect( m_pContext)) == NULL)
	{
		rc = RC_SET( NE_FLM_CONNECT_FAIL);
		goto Exit;
	}
	
	BIO_get_ssl( m_pBio, &m_pSSL);
	if( m_pSSL == NULL)
	{
		rc = RC_SET( NE_FLM_CONNECT_FAIL);
		goto Exit;
	}
	
	SSL_set_mode( m_pSSL, SSL_MODE_AUTO_RETRY);
	BIO_set_conn_hostname( m_pBio, pszHost);
	
	f_sprintf( szPort, "%u", (unsigned)uiPort);
	BIO_set_conn_port( m_pBio, szPort);
	
	// Open the connection

	if( BIO_do_connect( m_pBio) <= 0)
	{
		rc = RC_SET( NE_FLM_CONNECT_FAIL);
		goto Exit;
	}

	if( SSL_get_verify_result( m_pSSL) != X509_V_OK)
	{
		rc = RC_SET( NE_FLM_CONNECT_FAIL);
		goto Exit;
	}
	
	if( (m_pPeerCertificate = SSL_get_peer_certificate( m_pSSL)) == NULL)
	{
		rc = RC_SET( NE_FLM_CONNECT_FAIL);
		goto Exit;
	}
	
	if( (pPeerName = X509_get_subject_name( m_pPeerCertificate)) == NULL)
	{
		rc = RC_SET( NE_FLM_CONNECT_FAIL);
		goto Exit;
	}
	
	if( (X509_NAME_get_text_by_NID( pPeerName, NID_commonName, 
		m_szPeerName, sizeof( m_szPeerName))) == -1)
	{
		rc = RC_SET( NE_FLM_CONNECT_FAIL);
		goto Exit;
	}
	
	if( f_stricmp( pszHost, m_szPeerName) != 0)
	{
		rc = RC_SET( NE_FLM_CONNECT_FAIL);
		goto Exit;
	}
	
	// Get the peer's certificate text
	
	if( (pMemBIO = BIO_new( BIO_s_mem())) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( (pPublicKey = X509_get_pubkey( m_pPeerCertificate)) == NULL)
	{
		rc = RC_SET( NE_FLM_CONNECT_FAIL);
		goto Exit;
	}
	
	if( pPublicKey->type != EVP_PKEY_RSA)
	{
		rc = RC_SET( NE_FLM_CONNECT_FAIL);
		goto Exit;
	}
		
	if( (PEM_write_bio_X509( pMemBIO, m_pPeerCertificate)) == 0)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( (uiCertTextLen = BIO_get_mem_data( pMemBIO, &pszCertText)) == 0)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = f_alloc( uiCertTextLen + 1, &m_pszPeerCertText)))
	{
		goto Exit;
	}
	
	f_memcpy( m_pszPeerCertText, pszCertText, uiCertTextLen + 1);
	
Exit:

	if( pMemBIO)
	{
		BIO_free( pMemBIO);
	}
	
	if( pPublicKey)
	{
		EVP_PKEY_free( pPublicKey);
	}
	
	if( RC_BAD( rc))
	{
		closeStream();
	}
	
	return( rc);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_OPENSSL
RCODE FTKAPI F_SSLIOStream::read(
	void *					pvBuffer,
	FLMUINT					uiBytesToRead,
	FLMUINT *				puiBytesRead)
{
	RCODE						rc = NE_FLM_OK;
	int						iBytesRead;
	FLMUINT					uiTotalBytesRead = 0;
	char *					pucBuffer = (char *)pvBuffer;
	
	while( uiBytesToRead)
	{
		if( (iBytesRead = BIO_read( m_pBio, &pucBuffer[ uiTotalBytesRead],
			uiBytesToRead)) <= 0)
		{
			if( !BIO_should_retry( m_pBio))
			{
				rc = RC_SET( NE_FLM_EOF_HIT);
				goto Exit;
			}
			
			continue;
		}
		
		uiTotalBytesRead += iBytesRead;
		uiBytesToRead -= iBytesRead;
	}
	
Exit:

	if( puiBytesRead)
	{
		*puiBytesRead = uiTotalBytesRead;
	}

	return( rc);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_OPENSSL
RCODE FTKAPI F_SSLIOStream::write(
	const void *			pvBuffer,
	FLMUINT					uiBytesToWrite,
	FLMUINT *				puiBytesWritten)
{
	RCODE						rc = NE_FLM_OK;
	int						iBytesWritten = 0;
	
	if( !uiBytesToWrite)
	{
		goto Exit;
	}
	
	if( (iBytesWritten = BIO_write( m_pBio, pvBuffer, uiBytesToWrite)) <= 0)
	{
		iBytesWritten = 0;
		rc = RC_SET( NE_FLM_SOCKET_WRITE_FAIL);
		goto Exit;
	}
	
Exit:

	if( puiBytesWritten)
	{
		*puiBytesWritten = (FLMUINT)iBytesWritten;
	}

	return( rc);
}
#endif
		
/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_OPENSSL
RCODE FTKAPI F_SSLIOStream::closeStream( void)
{
	if( m_pBio)
	{
		BIO_free_all( m_pBio);
		m_pBio = NULL;
		m_pSSL = NULL;
	}
	
	if( m_pPeerCertificate)
	{
		X509_free( m_pPeerCertificate);
		m_pPeerCertificate = NULL;
	}
	
	if( m_pszPeerCertText)
	{
		f_free( &m_pszPeerCertText);
	}
	
	if( m_pContext)
	{
		SSL_CTX_free( m_pContext);
		m_pContext = NULL;
	}
	
	m_szPeerName[ 0] = 0;

	return( NE_FLM_OK);
}
#endif

/******************************************************************************
Desc:
******************************************************************************/
#ifdef FLM_OPENSSL
const char * FTKAPI F_SSLIOStream::getPeerCertificateText( void)
{
	return( m_pszPeerCertText);
}
#endif

/********************************************************************
Desc:
*********************************************************************/
RCODE f_socketPeek(
	int					iSocket,
	FLMUINT				uiTimeoutVal,
	FLMBOOL				bPeekRead)
{
	RCODE					rc = NE_FLM_OK;
	struct timeval		TimeOut;
	int					iMaxDescs;
	fd_set				GenDescriptors;
	fd_set *				DescrRead;
	fd_set *				DescrWrt;

	if( iSocket != INVALID_SOCKET)
	{
		FD_ZERO( &GenDescriptors);
#ifdef FLM_WIN
		#pragma warning( push)
		#pragma warning( disable : 4127)
#endif

		FD_SET( iSocket, &GenDescriptors);
		
#ifdef FLM_WIN
		#pragma warning( pop)
#endif

		iMaxDescs = (int)(iSocket + 1);
		DescrRead = bPeekRead ? &GenDescriptors : NULL;
		DescrWrt  = bPeekRead ? NULL : &GenDescriptors;

		TimeOut.tv_sec = (long)uiTimeoutVal;
		TimeOut.tv_usec = (long)0;

		if( select( iMaxDescs, DescrRead, DescrWrt, NULL, &TimeOut) < 0 )
		{
			rc = RC_SET( NE_FLM_SELECT_ERR);
			goto Exit;
		}
		else
		{
			if( !FD_ISSET( iSocket, &GenDescriptors))
			{
				rc = bPeekRead 
					? RC_SET( NE_FLM_SOCKET_READ_TIMEOUT)
					: RC_SET( NE_FLM_SOCKET_WRITE_TIMEOUT);
			}
		}
	}
	else
	{
		rc = RC_SET( NE_FLM_CONNECT_FAIL);
	}

Exit:

	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
void f_closeSocket(
	SOCKET *				piSocket)
{
	if( *piSocket != INVALID_SOCKET)
	{
#ifndef FLM_UNIX
		closesocket( *piSocket);
#else
		::close( *piSocket);
#endif
		*piSocket = INVALID_SOCKET;
	}
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE FTKAPI FlmOpenTCPListener(
	FLMBYTE *					pucBindAddr,
	FLMUINT						uiBindPort,
	IF_TCPListener **			ppListener)
{
	RCODE							rc = NE_FLM_OK;
	F_TCPListener *			pListener = NULL;
	
	if( (pListener = f_new F_TCPListener) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pListener->bind( uiBindPort, pucBindAddr)))
	{
		goto Exit;
	}
	
	*ppListener = pListener;
	pListener = NULL;
	
Exit:

	if( pListener)
	{
		pListener->Release();
	}
	
	return( rc);
}

