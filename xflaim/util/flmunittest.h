//------------------------------------------------------------------------------
// Desc: Interface definition that all unit tests must implement
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

#ifndef FLMUNITTEST_H
#define FLMUNITTEST_H

#include "xflaim.h"

// Status codes passed to recordUnitTestResults

#if defined( FLM_WIN)
	// Inherits via dominance
	#pragma warning( disable : 4250)
#endif
	
#define PASS								"PASS"
#define FAIL								"FAIL"

#define MAX_SMALL_BUFFER_SIZE			255
#define MAX_BUFFER_SIZE					2500

#define DATA_ORDER						"Testcase Name,Owner,Description,Steps,Build,Status,Environment," \
														"ResDetails,"/*Elapsed Time,*/"Attributes,Folder\n"

#ifndef ELEMCOUNT
	#define ELEMCOUNT(a) \
		sizeof(a) / sizeof(a[0])
#endif

#define MAKE_ERR_STRING( str, buf) \
	f_sprintf( buf, str" file: %s. line: %u.", __FILE__, __LINE__); \
	flmAssert( 0)

#define MAKE_ERROR_STRING( str, buf, rcode) \
	f_sprintf( buf, str" "#rcode" == %X. file: %s. line: %u.", \
		(unsigned)rcode, __FILE__, __LINE__); \
	flmAssert( 0)

#define MAKE_FLM_ERROR_STRING( str, buf, rcode) \
	f_sprintf( buf, str" "#rcode" == 0x%04X. file: %s. line: %u.", \
		(unsigned)rcode, __FILE__, __LINE__); \
	flmAssert( 0);
	
#define MAKE_GENERIC_ERROR_STRING( str, buf, num) \
	f_sprintf( buf, str": %lX  file: %s. line: %u.", \
		(unsigned long)num, __FILE__, __LINE__); \
	flmAssert( 0);
	
#define MAKE_GENERIC_ERROR_STRING64( str, buf, num) \
	f_sprintf( buf, str": %llX  file: %s. line: %u.", \
		(unsigned long long)num, __FILE__, __LINE__); \
	flmAssert( 0);

// Error Codes

#define UNITTEST_INVALID_CSVFILE				-101
#define UNITTEST_INVALID_CONFIGFILE			-102
#define UNITTEST_CONFIGPATH_READ_FAILED	-103
#define UNITTEST_BUFFER_TOO_SMALL			-104
#define UNITTEST_INVALID_PASSWORD			-105
#define UNITTEST_INVALID_USER_NAME			-106
#define UNITTEST_INVALID_PARAM				-107
#define UNITTEST_INVALID_CONFIGPATH			-108

typedef struct KeyCompInfo
{
	void *	pvComp;
	FLMUINT	uiDataType;
	FLMUINT	uiDataSize;
} KEY_COMP_INFO;

typedef struct ElementNodeInfo
{
	void *	pvData;
	FLMUINT	uiDataType;
	FLMUINT	uiDataSize;
	FLMUINT	uiDictNum;
} ELEMENT_NODE_INFO;

typedef struct unitTestData_t
{
	char userName[ MAX_SMALL_BUFFER_SIZE];
	char buildNumber[ MAX_SMALL_BUFFER_SIZE];
	char environment[ MAX_SMALL_BUFFER_SIZE];
	char folder[ MAX_SMALL_BUFFER_SIZE];
	char attrs[ MAX_SMALL_BUFFER_SIZE];
	char csvFilename[ MAX_SMALL_BUFFER_SIZE];
} unitTestData;

RCODE createUnitTest( 
	const char *		configPath,
	const char *		buildNum, 
	const char *		user, 
	const char *		environment,
	unitTestData *		uTD);

RCODE recordUnitTestResults(
	unitTestData *		uTD,
	const char *		testName,
	const char *		testDescr, 
	const char *		steps, 
	const char *		status, 
	const char *		resultDetails,
	const char * 		elapsedTime = NULL);

/****************************************************************************
Desc:
****************************************************************************/
class IFlmTest : public F_Object
{
public:

	virtual RCODE init(
		FLMBOOL			bLog,
		const char *	pszLogfile,
		FLMBOOL			bDisplay,
		FLMBOOL			bVerboseDisplay,
		const char *	pszConfigFile,
		const char *	pszEnvironment,
		const char *	pszBuild,
		const char *	pszUser) = 0;

	virtual const char * getName( void) = 0;
	
	virtual RCODE execute( void) = 0;
};

/****************************************************************************
Desc:
****************************************************************************/
class IFlmTestDisplayer : public F_Object
{
public:

	IFlmTestDisplayer();

	RCODE init( void);
	
	void appendString( 
		const char * 		pszString);
};

/****************************************************************************
Desc:
****************************************************************************/
class ITestReporter : public F_Object
{
	unitTestData 	m_uTD;
	FLMBOOL 			m_bInitialized;

public:

	ITestReporter()
	{
		m_bInitialized = false;
		f_memset( &m_uTD, 0, sizeof( unitTestData));
	}

	virtual ~ITestReporter();
	
	RCODE init(
		const char * 		configFile,
		const char * 		buildNum,
		const char * 		environment,
		const char * 		userName);

	RCODE recordUnitTestResults(
		const char *		testName,
		const char *		testDescr,
		const char *		steps,
		const char *		status,
		const char *		resultDetails,
		const char *		elapsedTime);
};

/****************************************************************************
Desc:
****************************************************************************/
class IFlmTestLogger : public F_Object
{
private:

	FLMBOOL			m_bInitialized;
	char				m_szFilename[ 128];

public:

	IFlmTestLogger()
	{
		m_bInitialized = FALSE;
		m_szFilename[ 0] = '\0';
	}

	RCODE init( 
		const char *		pszFilename);
	
	RCODE appendString( 
		const char *		pszString);
};

/******************************************************************************
Desc: Makes iterating through the keys of an index more convenient
******************************************************************************/
class KeyIterator : public F_Object
{
public:

	KeyIterator()
	{
		m_pSearchKey = NULL;
		m_pFoundKey = NULL;
		m_pDbSystem = NULL;
		m_bFirstCall = TRUE;
		m_pDb = NULL;
		m_uiIndex = 0;
	}

	~KeyIterator();

	RCODE next( void);

	RCODE getCurrentKeyVal(
		FLMUINT		uiComponent,
		FLMBYTE * 	pszKey,
		FLMUINT 		uiBufSize,
		FLMUINT * 	puiKeyLen,
		FLMUINT64 * pui64Id = NULL);

	FINLINE FLMUINT getIndexNum( void)
	{
		return m_uiIndex;
	}

	FINLINE void setIndexNum( 
		FLMUINT		uiIndex)
	{
		m_uiIndex = uiIndex;

		// Changing the index necessitates a reset
		
		reset();
	}

	FINLINE RCODE init( 
		FLMUINT			uiIndex,
		IF_DbSystem *	pDbSystem,
		IF_Db *			pDb)
	{
		RCODE		rc = NE_XFLM_OK;

		m_pDbSystem = pDbSystem;
		m_pDb = pDb;
		
		if( RC_BAD( rc = m_pDbSystem->createIFDataVector( &m_pSearchKey)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = m_pDbSystem->createIFDataVector( &m_pFoundKey)))
		{
			goto Exit;
		}
		
		m_uiIndex = uiIndex;
		
	Exit:

		return( rc);
	}

	FINLINE void reset( void)
	{
		if( m_pSearchKey)
		{
			m_pSearchKey->reset();
		}
		
		if( m_pFoundKey)
		{
			m_pFoundKey->reset();
		}
		
		m_bFirstCall = TRUE;
	}

private:

	IF_DataVector	*	m_pSearchKey;
	IF_DataVector	*	m_pFoundKey;
	IF_DbSystem *		m_pDbSystem;
	IF_Db *				m_pDb;
	FLMBOOL				m_bFirstCall;
	FLMUINT				m_uiIndex;
};

/******************************************************************************
Desc:	Quick, dirty, and wholly inadequate subsititute for the STL BitSet.
		This only works with strings.
******************************************************************************/
class FlagSet
{
public:

	FlagSet()
	{
		m_pbFlagArray = NULL;
		m_ppucElemArray = NULL;
		m_uiNumElems = 0;
	}

	FlagSet( const FlagSet& fs);

	~FlagSet()
	{
		reset();
	}

	FlagSet crossProduct( FlagSet& fs2);
	
	FlagSet& operator=( const FlagSet& fs);

	FLMBOOL removeElem( 
		FLMBYTE *			pElem);
	
	FLMBOOL removeElemContaining(
		FLMBYTE *			pszSubString);
		
	FLMBOOL setElemFlag(
		FLMBYTE *			pElem);
		
	FINLINE FLMBOOL allElemFlagsSet()
	{
		FLMBOOL 		bAllSet = TRUE;
		
		for( FLMUINT uiLoop = 0; uiLoop < m_uiNumElems; uiLoop++)
		{
			if( m_pbFlagArray[uiLoop] == FALSE)
			{
				bAllSet = FALSE;
			}
		}
		
		return( bAllSet);
	}
	
	FINLINE FLMBOOL noElemFlagsSet( void)
	{
		FLMBOOL		bNoneSet = TRUE;
		
		for( FLMUINT uiLoop = 0; uiLoop < m_uiNumElems; uiLoop++)
		{
			if( m_pbFlagArray[uiLoop] == TRUE)
			{
				bNoneSet = FALSE;
			}
		}
		
		return( bNoneSet);
	}

	FINLINE void unsetAllFlags( void)
	{
		f_memset( m_pbFlagArray, 0, sizeof( FLMBOOL) * m_uiNumElems);
	}

	FINLINE FLMUINT getNumElements( void)
	{
		return m_uiNumElems;
	}
	
	void reset( void);
	
	void clearFlags( void)
	{
		f_memset( m_pbFlagArray, 0, sizeof( FLMBOOL) * m_uiNumElems);
	}
	
	void init( 
		FLMBYTE **	ppucElemArray, 
		FLMUINT 		uiNumElems);
		
private:

	FLMBOOL containsSubstring(
		FLMBYTE *		pszString,
		FLMBYTE *		pszSub)
	{
		FLMBYTE *	pszStringTemp = NULL;
		FLMBYTE *	pszSubTemp  = NULL;
		FLMBOOL		bContainsSub = FALSE;
 
    // First scan quickly through the two strings looking for a
    // single-character match.  When it's found, then compare the
    // rest of the substring.

		pszSubTemp = pszSub;
		if( *pszSub == 0)
		{
			goto Exit;
		}
		
		for( ; *pszString != 0; pszString++) 
		{
			if( *pszString != *pszSubTemp) 
			{
				 continue;
			}
			
			pszStringTemp = pszString;
			for( ;;) 
			{
				if( *pszSubTemp == 0) 
				{
					bContainsSub = TRUE;
					goto Exit;
				}
				 
				if( *pszStringTemp++ != *pszSubTemp++) 
				{
					break;
				}
			}
			
			pszSubTemp = pszSub;
			
		 }
		 
Exit:

		 return( bContainsSub);
	}

	FLMBOOL *	m_pbFlagArray;
	FLMBYTE **	m_ppucElemArray;
	FLMUINT		m_uiNumElems;
};

/******************************************************************************
Desc: Makes iterating through the keys of an index more convenient
******************************************************************************/
FINLINE FLMBOOL isPlaceHolder( const ELEMENT_NODE_INFO & elm)
{
	if( elm.pvData == NULL &&
		 elm.uiDataType == XFLM_NODATA_TYPE &&
		 elm.uiDataSize == 0 &&
		 elm.uiDictNum == 0)
	{
		return ( TRUE);
	}
	
	return ( FALSE);
}

/******************************************************************************
Desc: Makes iterating through the keys of an index more convenient
******************************************************************************/
class TestBase : public IFlmTest
{

public:

	TestBase()
	{
		m_bLog = FALSE;
		m_bDisplay = FALSE;
		m_bDisplayVerbose = FALSE;
		m_pLogger = NULL;
		m_pDisplayer = NULL;
		m_pReporter = NULL;
		m_pDbSystem = NULL;
		m_pDb = NULL;
		m_pInputStream = NULL;
		m_pszTestName = NULL;
		m_pszTestDesc = NULL;
		m_pszSteps = NULL;
		m_ui64StartMs = 0;
		m_ui64EndMs = 0;
	}

	virtual ~TestBase();

	RCODE init(
		FLMBOOL			bLog,
		const char *	pszLogfile,
		FLMBOOL			bDisplay,
		FLMBOOL			bVerboseDisplay,
		const char *	pszConfigFile,
		const char *	pszEnvironment,
		const char *	pszBuild,
		const char *	pszUser);

protected:

	FLMBOOL						m_bLog;
	FLMBOOL						m_bDisplay;
	FLMBOOL						m_bDisplayVerbose;
	IFlmTestLogger *			m_pLogger;
	IFlmTestDisplayer *		m_pDisplayer;
	ITestReporter *			m_pReporter;
	IF_DbSystem *				m_pDbSystem;
	IF_Db *						m_pDb;
	IF_PosIStream *			m_pInputStream;
#define DETAILS_BUF_SIZ	1024
	char							m_szDetails[ DETAILS_BUF_SIZ];
	const char *				m_pszTestName;
	const char *				m_pszTestDesc;
	const char *				m_pszSteps;
	FLMUINT64					m_ui64StartMs;
	FLMUINT64					m_ui64EndMs;

	void beginTest( 
		const char *			pszTestName, 
		const char *			pszTestDesc,
		const char *			pszTestSteps,
		const char *			pszDetails);

	void endTest( 
		const char *			pszTestResult);

	void log( 
		const char *	pszString);
		
	void display( 
		const char *	pszString);
		
	void displayLine(
		const char *	pszString);
		
	void displayTime(
		FLMUINT64		ui64Milli,
		const char * 	pszIntro = NULL);
		
	void normalizeTime(
		FLMUINT64 		ui64Milli,
		char * 			pszString);

	RCODE logTestResults(
		const char * 	pszTestResult,
		FLMUINT64 		ui64ElapsedTime = ~((FLMUINT64)0));

	RCODE displayTestResults(
		const char * 	pszTestResult,
		FLMUINT64 		ui64ElapsedTime = ~((FLMUINT64)0));

	RCODE outputAll( 
		const char * 	pszTestResult,
		FLMUINT64 		ui64ElapsedTime = ~((FLMUINT64)0));

	RCODE initCleanTestState( 
		const char  *	pszDibName);

	RCODE openTestState(
		const char *	pszDibName);

	RCODE shutdownTestState(
		const char *	pszDibName,
		FLMBOOL			bRemoveDib);

	RCODE checkQueryResults( 
		const char **	ppszResults, 
		FLMUINT			uiNumResultsExpected, 
		IF_Query *		pQuery,
		char *			pszDetails);

	RCODE doQueryTest( 
		const char *	pszQueryString,
		const char **	ppszExpectedResults,
		FLMUINT			uiNumResultsExpected,
		IF_Query *		pQuery,
		char *			pszDetails,
		FLMUINT			uiRequestedIndex = 0,
		FLMUINT *		puiIndexUsed = NULL);

	RCODE importBuffer(
		const char *	pszBuffer,
		FLMUINT			uiCollection);

	RCODE importDocument(
		const char *	pszBuffer,
		FLMUINT			uiCollection);

	RCODE importFile(
		 const char *	pszFilename,
		 FLMUINT	 		uiCollection);

	FLMINT unicmp(
		FLMUNICODE *	puzStr1,
		FLMUNICODE *	puzStr2);

	RCODE createCompoundDoc(
		ELEMENT_NODE_INFO *	pElementNodes,
		FLMUINT					uiNumElementNodes,
		FLMUINT64 *				pui64DocId);
};

/****************************************************************************
Desc:
****************************************************************************/
class ArgList
{
public:

	ArgList();
	
	~ArgList();

	const char * operator[](
		FLMUINT 			uiIndex);

	const char * getArg(
		FLMUINT 			uiIndex);

	RCODE addArg(
		const char * 	pszArg);

	FLMUINT getNumEntries( void);

	RCODE expandArgs(
		char ** 			ppszArgs,
		FLMUINT			uiNumArgs);

	RCODE expandFileArgs(
		const char * 	pszFilename);
		
private:

	enum
	{
		INIT_SIZE = 10,
		GROW_FACTOR = 2
	};

	char **		m_ppszArgs;
	FLMUINT		m_uiCapacity;
	FLMUINT		m_uiNumEntries;

	RCODE resize( void);
	
	RCODE getTokenFromFile( 
		char *			pszToken,
		IF_FileHdl * 	pFileHdl);

	FLMBOOL isWhitespace( 
		char 			c)
	{
		return (c == ' ' || c == '\t' || c == 0xd || c == 0xa)
			? TRUE
			: FALSE;
	}
};

#endif
