//-------------------------------------------------------------------------
// Desc:	Client/server definitions.
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

#ifndef FSV_H
#define FSV_H

#include "fpackon.h"

// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

#define FSV_MAX_TCP_HANDLERS					64
#define FSV_LOG_BUFFER_SIZE					256

// Server defaults

#define FSV_DEFAULT_CACHE_SIZE				1024000
#define FSV_DEFAULT_MAX_CONNECTIONS			256
#define FSV_DEFAULT_CONOTBL_TABLE_SIZE		32
#define FSV_DEFAULT_PATH_TABLE_SIZE			32

// Server object types

#define FSV_OBJECT_SESSION						0x0001
#define FSV_OBJECT_DATABASE					0x0002
#define FSV_OBJECT_ITERATOR					0x0003
#define FSV_OBJECT_TRANSACTION				0x0004
#define FSV_OBJECT_BLOB							0x0005
#define FSV_OBJECT_PATH							0x0006
#define FSV_OBJECT_ROPS							0x0007
#define FSV_OBJECT_POOL							0x0008

typedef struct FSV_RECORD_ID
{
	FLMUINT		uiDatabaseId;
	FLMUINT		uiStore;
	FLMUINT		uiContainer;
	FLMUINT		uiDrn;
} FSV_RECORD_ID;

typedef void (* FSV_LOG_FUNC)(
	const char *	pszMsg,
	RCODE				rc,
	FLMUINT			uiSeverity,
	void *			pvUserData);
  
#define FSV_LOG_DEBUG			1
#define FSV_LOG_EVENT			2
#define FSV_LOG_ERROR			3
#define FSV_LOG_NOTHING			4

class FSV_SCTX;
class FSV_SESN;
typedef FSV_SCTX *	FSV_SCTX_p;

/****************************************************************************
Desc:
****************************************************************************/
class	FSV_SCTX : public F_Object
{
public:

	FSV_SCTX();
	
	virtual ~FSV_SCTX();

	RCODE Setup(
		FLMUINT				uiMaxSessions,
		const char *		pszServerBasePath,
		FSV_LOG_FUNC 		pLogFunc);

	RCODE OpenSession(
		FLMUINT					uiVersion,
		FLMUINT					uiFlags,
		FLMUINT *				puiIdRV,
		FSV_SESN **				ppSessionRV);

	RCODE CloseSession(
		FLMUINT					uiId);

	RCODE GetSession(
		FLMUINT					uiId,
		FSV_SESN **				ppSession);

	RCODE SetBasePath(
		const char  *			pszServerBasePath);

	RCODE GetBasePath(
		char *					pszServerBasePath);

	RCODE SetTempDir(
		const char *			pszTempDir);

	RCODE BuildFilePath(
		const FLMUNICODE *	puzUrlString,
		char *					pszFilePathRV);

	void Lock( void);

	void Unlock( void);

	void LogMessage(
		FSV_SESN *				pSession,
		const char *			pszMsg,
		RCODE						rc,
		FLMUINT					uiMsgSeverity);

private:

	FLMUINT				m_uiSessionToken;
	FLMUINT				m_uiMaxSessions;
	FLMUINT				m_uiCacheSize;
	char 					m_szServerBasePath[ F_PATH_MAX_SIZE];
	FSV_SESN **			m_paSessions;
	F_MUTEX				m_hMutex;
	FSV_LOG_FUNC 		m_pLogFunc;
	FLMBOOL				m_bSetupCalled;
	char					m_pucLogBuf[ FSV_LOG_BUFFER_SIZE];
	
friend class FSV_SESN;
};

class FSV_SESN;
typedef FSV_SESN *	FSV_SESN_p;

/****************************************************************************
Desc:
****************************************************************************/
class	FSV_SESN : public F_Object
{
public:

	FSV_SESN();
	
	virtual ~FSV_SESN();

	RCODE Setup(
		FSV_SCTX *		pServerContext,
		FLMUINT			uiVersion,
		FLMUINT			uiFlags);

	RCODE OpenDatabase(
		FLMUNICODE *	puzDbPath,
		FLMUNICODE *	puzDataDir,
		FLMUNICODE *	puzRflPath,
		FLMUINT			uiOpenFlags);

	RCODE CreateDatabase(
		FLMUNICODE *	puzDbPath,
		FLMUNICODE *	puzDataDir,
		FLMUNICODE *	puzRflPath,
		FLMUNICODE *	puzDictPath,
		FLMUNICODE *	puzDictBuf,
		CREATE_OPTS *	pCreateOpts);

	RCODE CloseDatabase( void);

	FINLINE HFDB GetDatabase( void)
	{
		return( m_hDb);
	}

	RCODE InitializeIterator(
		FLMUINT *		puiIteratorIdRV,
		HFDB				hDb,
		FLMUINT			uiContainer,
		HFCURSOR *		phIteratorRV);

	RCODE FreeIterator(
		FLMUINT			uiIteratorId);

	RCODE GetIterator(
		FLMUINT			uiIteratorId,
		HFCURSOR *		phIteratorRV);

	RCODE GetBIStream(
		FCS_BIOS ** 	ppBIStream);

	RCODE GetBOStream(
		FCS_BIOS ** 	ppBOStream);

	FINLINE void setId(
		FLMUINT			uiId)
	{
		m_uiSessionId = uiId;
	}
	
	FINLINE FLMUINT getId( void)
	{
		return( m_uiSessionId);
	}

	FINLINE void setCookie(
		FLMUINT			uiCookie)
	{
		m_uiCookie = uiCookie;
	}
	
	FINLINE FLMUINT getCookie( void)
	{
		return( m_uiCookie);
	}

	FINLINE FLMUINT getFlags( void)
	{
		return( m_uiFlags);
	}

	FINLINE F_Pool * getWireScratchPool( void)
	{
		return &m_wireScratchPool;
	}
	
	FINLINE FLMUINT getClientVersion( void)
	{
		return( m_uiClientProtocolVersion);
	}

private:

	FSV_SCTX *			m_pServerContext;
	HFDB					m_hDb;
	FLMBYTE				m_pucLogBuf[ FSV_LOG_BUFFER_SIZE];
	FLMUINT				m_uiSessionId;
	FLMUINT				m_uiCookie;
	FLMUINT				m_uiFlags;
	FLMBOOL				m_bSetupCalled;
	FLMUINT				m_uiClientProtocolVersion;
	FCS_BIOS *			m_pBIStream;
	FCS_BIOS *			m_pBOStream;
#define MAX_SESN_ITERATORS			10
	HFCURSOR				m_IteratorList[ MAX_SESN_ITERATORS];
	F_Pool				m_wireScratchPool;
};

/****************************************************************************

									Server Wire Class

****************************************************************************/

class FSV_WIRE;
typedef FSV_WIRE *	FSV_WIRE_p;

class	FSV_WIRE : public FCS_WIRE
{
private:

	FLMUINT			m_uiOpSeqNum;
	FLMUINT			m_uiClientVersion;
	FLMUINT			m_uiAutoTrans;
	FLMUINT			m_uiType;
	FLMUINT			m_uiAreaId;
	FLMUINT			m_uiMaxLockWait;
	FLMUNICODE *	m_puzDictPath;
	FLMUNICODE *	m_puzDictBuf;
	FLMUNICODE *	m_puzFileName;
	FLMBYTE *		m_pucPassword;
	FLMUINT *		m_pDrnList;
	NODE *			m_pIteratorSelect;
	NODE *			m_pIteratorFrom;
	NODE *			m_pIteratorWhere;
	NODE *			m_pIteratorConfig;
	FSV_SESN *		m_pSession;
	HFCURSOR			m_hIterator;

public:

	FINLINE FSV_WIRE( 
		FCS_DIS * 	pDIStream,
		FCS_DOS * 	pDOStream) : FCS_WIRE( pDIStream, pDOStream)
	{
		reset();
	}

	FINLINE ~FSV_WIRE()
	{
	}

	void reset( void);
	
	RCODE read( void);

	FINLINE FLMUINT getOpSeqNum( void) 
	{
		return( m_uiOpSeqNum);
	}
	
	FINLINE FLMUINT getClientVersion( void) 
	{ 
		return( m_uiClientVersion);
	}
	
	FINLINE FLMUINT getAutoTrans( void)
	{
		return( m_uiAutoTrans);
	}
	
	FINLINE FLMUINT getFlags( void)
	{
		return( m_uiFlags);
	}
	
	FINLINE FLMUINT getType( void) 
	{
		return( m_uiType);
	}
	
	FINLINE FLMUINT getAreaId( void)
	{
		return( m_uiAreaId);
	}
	
	FINLINE FLMUINT getMaxLockWait( void)
	{
		return( m_uiMaxLockWait);
	}
	
	FINLINE FLMUNICODE * getDictPath( void)
	{
		return( m_puzDictPath);
	}
	
	FINLINE FLMUNICODE * getDictBuffer( void)
	{
		return( m_puzDictBuf);
	}
	
	FINLINE FLMUNICODE * getFileName( void)
	{
		return( m_puzFileName);
	}
	
	FINLINE FLMBYTE * getPassword( void)
	{
		return( m_pucPassword);
	}
	
	FINLINE FLMUINT * getDrnList( void)
	{
		return( m_pDrnList);
	}
	
	FINLINE NODE * getIteratorSelect( void)
	{
		return( m_pIteratorSelect);
	}
	
	FINLINE NODE * getIteratorFrom( void) 
	{
		return( m_pIteratorFrom);
	}
	
	FINLINE NODE * getIteratorWhere( void)
	{
		return( m_pIteratorWhere);
	}
	
	FINLINE NODE * getIteratorConfig( void)
	{
		return( m_pIteratorSelect);
	}

	FINLINE FSV_SESN * getSession( void)
	{
		return( m_pSession);
	}
	
	FINLINE HFCURSOR getIteratorHandle( void)
	{
		return( m_hIterator);
	}

	void setSession( 
		FSV_SESN * pSession);
	
	FINLINE void setIteratorId( 
		FLMUINT		uiId)
	{ 
		m_uiIteratorId = uiId;
	}
	
	FINLINE void setIteratorHandle( 
		HFCURSOR 	hIterator) 
	{ 
		m_hIterator = hIterator;
	}
};

/****************************************************************************

								Server BLOB Class

****************************************************************************/

class FSV_BLOB;
typedef FSV_BLOB *	FSV_BLOB_p;

/****************************************************************************
Desc:
****************************************************************************/
class	FSV_BLOB : public F_Object
{
private:

	HFBLOB		m_hBlob;

public:

	FINLINE FSV_BLOB( void)
	{
		m_hBlob = HFBLOB_NULL;
	}

	virtual FINLINE ~FSV_BLOB()
	{
	}

	FINLINE HFBLOB * getFlmBlob( void)
	{ 
		return( &m_hBlob);
	}
};

RCODE fsvInitGlobalContext(
	FLMUINT				uiMaxSessions,
	const char *		pszServerBasePath,
	FSV_LOG_FUNC		pLogFunc);

void fsvFreeGlobalContext( void);

RCODE fsvGetGlobalContext(
	FSV_SCTX **			ppGlobalContext);

RCODE fsvSetBasePath(
	FLMBYTE *			pszServerBasePath);

RCODE fsvSetTempDir(
	FLMBYTE *			pszTempDir);

RCODE fsvProcessRequest(
	FCS_DIS *         pDataIStream,
	FCS_DOS *         pDataOStream,
	F_Pool *				pScratchPool,
	FLMUINT *			puiSessionIdRV);

RCODE fsvOpClassDiag(
	FSV_WIRE_p			pWire);

RCODE fsvOpClassFile(
	FSV_WIRE_p			pWire);

RCODE fsvOpClassAdmin(
	FSV_WIRE_p			pWire);

RCODE fsvOpClassGlobal(
	FSV_WIRE_p			pWire);

RCODE fsvOpClassSession(
	FSV_WIRE_p			pWire);

RCODE fsvOpClassDatabase(
	FSV_WIRE_p			pWire);

RCODE fsvOpClassTransaction(
	FSV_WIRE_p			pWire);

RCODE fsvOpClassMaintenance(
	FSV_WIRE *			pWire);

RCODE fsvOpClassRecord(
	FSV_WIRE_p			pWire);

RCODE fsvOpClassIterator(
	FSV_WIRE_p			pWire);

RCODE fsvOpClassRops(
	FSV_WIRE_p			pWire);

RCODE fsvOpClassBlob(
	FSV_WIRE_p			pWire);

RCODE fsvOpClassIndex(
	FSV_WIRE *			pWire);

RCODE fsvOpClassMisc(
	FSV_WIRE *			pWire);

RCODE  fsvDbTransCommitEx(
	HFDB					hDb,
	FSV_WIRE * 			pWire);

#ifdef FSV_LOGGING
	void fsvLogHandlerMessage(
		FSV_SESN *		pSession,
		FLMBYTE *		pucMsg,
		RCODE				rc,
		FLMUINT			uiMsgSeverity);
#endif

RCODE fsvStartTcpListener(
	FLMUINT				uiPort);

void fsvShutdownTcpListener( void);

RCODE	fsvPostStreamedRequest(
	FSV_SESN *			pSession,
	FLMBYTE *			pucPacket,
	FLMUINT				uiPacketSize,
	FLMBOOL				bLastPacket,
	FCS_BIOS *			pSessionResponse);

RCODE	fsvGetStreamedResponse(
	FSV_SESN *			pSession,
	FLMBYTE *			pucPacketBuffer,
	FLMUINT				uiMaxPacketSize,
	FLMUINT *			puiPacketSize,
	FLMBOOL *			pbLastPacket);

RCODE fsvStreamLoopback(
	FCS_BIOS *			pStream,
	FLMUINT				uiEvent,
	void *				UserData);

#include "fpackoff.h"

#endif
