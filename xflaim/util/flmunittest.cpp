//------------------------------------------------------------------------------
// Desc:	Unit test driver
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

#include "flmunittest.h"

#if defined( FLM_WIN)
	#if defined( FLM_64BIT)
		#define PLATPROC_STR		"w64ia64"
	#else
		#define PLATPROC_STR		"w32x86"
	#endif
#elif defined( FLM_NLM)
	#define PLATPROC_STR			"nwx86"
#elif defined( FLM_LINUX)
	#define PLATPROC_STR			"lxx86"
#elif defined( FLM_OSX)
	#define PLATPROC_STR			"osx"
#elif defined( FLM_SOLARIS)
	#define PLATPROC_STR			"solaris"
#endif

FLMBOOL gv_bShutdown = FALSE;
extern RCODE getTest( IFlmTest ** ppTest);

#ifdef FLM_RING_ZERO_NLM
	#define main		nlm_main
#endif

struct TEST_INFO
{
	bool 				bLog;
	char 				pszLogfile[ 256];
	bool 				bDisplay;
	bool 				bVerboseDisplay;
	char 				pszEnvironment[ 32];
	char 				pszBuild[ 32];
	char 				pszUser[ 32];
	char 				pszConfig[ 256];
	TEST_INFO * 	pNext;


	TEST_INFO()
	{
		bLog = FALSE;
		bDisplay = FALSE;
		bVerboseDisplay = FALSE;
		pNext = NULL;
		pszLogfile[0] = '\0';
		f_strcpy( pszEnvironment, PLATPROC_STR);
		f_strcpy( pszBuild, __DATE__);
		f_strcpy( pszUser, "defaultUser");
		pszConfig [0] = 0;
	}
};

/****************************************************************************
Desc:
****************************************************************************/
RCODE ITestReporter::init(
	const char * 	configFile,
	const char * 	buildNum,
	const char * 	environment,
	const char * 	userName)
{
	RCODE 			rc = NE_XFLM_OK;

	if( (rc = createUnitTest( configFile, buildNum, environment, userName,
		&(this->m_uTD))) != 0)
	{
		goto Exit;
	}

	m_bInitialized = true;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
ITestReporter::~ITestReporter()
{
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE ITestReporter::recordUnitTestResults(
	const char *		testName,
	const char *		testDescr,
	const char *		steps,
	const char *		status,
	const char *		resultDetails,
	const char *		elapsedTime)
{
	return ::recordUnitTestResults( &(this->m_uTD), testName, testDescr,
		steps, status, resultDetails, elapsedTime);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestLogger::init( 
	const char *		pszFilename)
{
	f_strcpy( m_szFilename, pszFilename);
	m_bInitialized = TRUE;
	return (NE_XFLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestLogger::appendString( 
	const char *		pszString)
{
	RCODE				rc = NE_XFLM_OK;
	char *			pszTemp = NULL;

	if ( RC_BAD( rc = f_alloc( f_strlen( pszString) + 3, &pszTemp)))
	{
		goto Exit;
	}

	f_sprintf( pszTemp, "%s\n", pszString);

	if ( RC_BAD( rc = f_filecat( m_szFilename, pszString)))
	{
		goto Exit;
	}

Exit:

	if ( pszTemp)
	{
		f_free( &pszTemp);
	}

	return rc;
}

/****************************************************************************
Desc:
****************************************************************************/
IFlmTestDisplayer::IFlmTestDisplayer()
{
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestDisplayer::init( void)
{
	return( NE_XFLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
void IFlmTestDisplayer::appendString( 
	const char * 	pszString)
{
	f_printf( pszString);
}

/****************************************************************************
Desc:
****************************************************************************/
void printHelp()
{
	f_printf( "\nCommand-line usage:");
	f_printf( "\n\n[-l<log file>] [-d]");
	f_printf( "[-c<config>] [-b<build>] [-u<user>]");
	f_printf( "\n-l - Specifies a log file to print to");
	f_printf( "\n-d - Display output");
	f_printf( "\n-t - Specifies configuration file for reporting");
	f_printf( "\n-b - Specifies the build number");
	f_printf( "\n-u - Specifies the user running the unit test");
	f_printf( "\n-h - Shows this screen");
}

/****************************************************************************
Desc:
****************************************************************************/
extern "C" int main( 
	int 				argc,
	char ** 			argv)
{
	RCODE				rc = NE_XFLM_OK;
	IFlmTest *		pTest = NULL;
	unsigned int	i = 1;
	ArgList			args;
	TEST_INFO		testInfo;

	//parse the command line
	//format:
	//[-l<log file>] [-d] [-c<config>] [-b<build>] [-u<user>]
	/*
	if ( argc < 2)
	{
		f_printf("You must specify at least one test to run");
		printHelp();
		goto Exit;
	}
	*/

	if ( argc > 1)
	{
		if ( ( f_strcmp( argv[1], "--help") == 0)||
			( f_strcmp( argv[1], "-h") == 0))
		{
			printHelp();
			goto Exit;
		}
	}

	args.expandArgs( argv, argc);

	while( i < args.getNumEntries())
	{
		if ( args[i][0] != '-')
		{
			goto Exit;
		}
				
		if ( ( args[i][1] == 'l') || ( args[i][1] == 'L'))
		{
			testInfo.bLog = true;
			f_strcpy( testInfo.pszLogfile, &args[i][2]);
		}
		else if ( ( args[i][1] == 'd') || ( args[i][1] == 'D'))
		{
			testInfo.bDisplay = true;
		}
		else if ( ( args[i][1] == 'c') || ( args[i][1] == 'C'))
		{
			//config file
			f_strcpy( testInfo.pszConfig, &args[i][2]);
		}
		else if ( ( args[i][1] == 'b') || ( args[i][1] == 'B'))
		{
			//build
			f_strcpy( testInfo.pszBuild, &args[i][2]);
		}
		else if ( ( args[i][1] == 'u') || ( args[i][1] == 'U'))
		{
			//user
			f_strcpy( testInfo.pszUser, &args[i][2]);
		}
		else if ( ( args[i][1] == 'v') || ( args[i][1] == 'V'))
		{
			//verbose
			testInfo.bVerboseDisplay = TRUE;
		}
		else
		{
			f_printf( "\nInvalid parameter");
			printHelp();
			goto Exit;
		}
		i++;
	}

	f_printf("Running %s\n", argv[0]);

	if ( RC_BAD( rc = getTest( &pTest)))
	{
		f_printf( "ERROR: Unable to create test instance\n");
		goto Exit;
	}

	if ( pTest->init(
		testInfo.bLog,
		testInfo.pszLogfile,
		testInfo.bDisplay,
		testInfo.bVerboseDisplay,
		testInfo.pszConfig,
		testInfo.pszEnvironment,
		testInfo.pszBuild,
		testInfo.pszUser) != 0)
	{
		f_printf( "\nTest initialization failed");
		goto Exit;
	}

	if ( RC_BAD( rc = pTest->execute()))
	{
		// f_printf("\nTest Failed.");
		goto Exit;
	}

Exit:

	if ( pTest)
	{
		pTest->Release();
	}

	return( (int)rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE KeyIterator::next()
{
	RCODE rc = NE_XFLM_OK;
	IF_DataVector * pTemp = NULL;

	if ( m_bFirstCall)
	{
		if ( RC_BAD( rc = m_pDb->keyRetrieve(	
			m_uiIndex,
			NULL,
			XFLM_FIRST,
			m_pFoundKey)))
		{
			goto Exit;
		}
		m_bFirstCall = FALSE;
	}
	else
	{
		pTemp = m_pSearchKey;
		m_pSearchKey = m_pFoundKey;
		m_pFoundKey = pTemp;

		if ( RC_BAD( rc = m_pDb->keyRetrieve(	
			m_uiIndex,
			m_pSearchKey,
			XFLM_EXCL,
			m_pFoundKey)))
		{
			goto Exit;
		}
	}
Exit:
	return rc;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE KeyIterator::getCurrentKeyVal(
	FLMUINT uiComponent, 
	FLMBYTE * pszKey, 
	FLMUINT uiBufSize,
	FLMUINT * puiKeyLen,
	FLMUINT64 * pui64Id)
{
	RCODE		rc = NE_XFLM_OK;

	if ( RC_BAD( rc = m_pFoundKey->getUTF8( uiComponent, pszKey, &uiBufSize)))
	{
		goto Exit;
	}
	if ( puiKeyLen)
	{
		*puiKeyLen = uiBufSize;
	}
	if ( pui64Id)
	{
		*pui64Id = m_pFoundKey->getID( uiComponent);
	}

Exit:
	return rc;
}

/****************************************************************************
Desc:
****************************************************************************/
KeyIterator::~KeyIterator()
{
	if ( m_pFoundKey)
	{
		m_pFoundKey->Release();
	}

	if ( m_pSearchKey)
	{
		m_pSearchKey->Release();
	}		
}

/****************************************************************************
Desc:
****************************************************************************/
FLMBOOL FlagSet::removeElem( FLMBYTE * pElem)
{
	FLMBOOL bElemExisted = FALSE;

	for ( FLMUINT uiLoop = 0; uiLoop < m_uiNumElems; uiLoop++)
	{
		if ( f_strcmp( (char *)pElem, (char *)m_ppucElemArray[ uiLoop]) == 0)
		{
			bElemExisted = TRUE;
			if ( uiLoop < m_uiNumElems - 1)
			{
				f_free( &m_ppucElemArray[ uiLoop]);
				
				f_memmove( &m_ppucElemArray[ uiLoop], 
					&m_ppucElemArray[ uiLoop + 1], 
					(m_uiNumElems - ( uiLoop + 1)) * sizeof( FLMBYTE *));
					
				f_memmove( &m_pbFlagArray[ uiLoop], 
					&m_pbFlagArray[ uiLoop + 1], 
					(m_uiNumElems - ( uiLoop + 1)) * sizeof( FLMBOOL));
			}
			// Otherwise, we're at the end and decrementing to counter will suffice

			m_uiNumElems--;
		}
	}
	return bElemExisted;
}

/****************************************************************************
Desc:
****************************************************************************/
FLMBOOL FlagSet::removeElemContaining( FLMBYTE * pszSubString)
{
	FLMBOOL bElemExisted = FALSE;

	for ( FLMUINT uiLoop = 0; uiLoop < m_uiNumElems; )
	{
		if ( containsSubstring( m_ppucElemArray[ uiLoop], pszSubString))
		{
			bElemExisted = TRUE;
			if ( uiLoop < m_uiNumElems - 1)
			{
				f_memmove( &m_ppucElemArray[ uiLoop], 
					&m_ppucElemArray[ uiLoop + 1], 
					( m_uiNumElems - ( uiLoop + 1)) * sizeof( FLMBYTE *));
				f_memmove( &m_pbFlagArray[ uiLoop], 
					&m_pbFlagArray[ uiLoop + 1], 
					( m_uiNumElems - ( uiLoop + 1)) * sizeof( FLMBYTE *));
			}
			// Otherwise, we're at the end and decrementing to counter will suffice

			m_uiNumElems--;
		}
		else
		{
			uiLoop++;
		}
	}
	return bElemExisted;
}

/****************************************************************************
Desc:
****************************************************************************/
FLMBOOL FlagSet::setElemFlag( FLMBYTE * pElem)
{
	FLMBOOL bIsInSet = FALSE;

	for ( FLMUINT uiLoop = 0; uiLoop < m_uiNumElems; uiLoop++)
	{
		if ( f_strcmp( (char *)pElem, (char *)m_ppucElemArray[ uiLoop]) == 0 &&
			  !m_pbFlagArray [uiLoop])
		{
			m_pbFlagArray[ uiLoop] = TRUE;
			bIsInSet = TRUE;
			break;
		}
	}
	return bIsInSet;
}

/****************************************************************************
Desc:
****************************************************************************/
FlagSet FlagSet::crossProduct( FlagSet& fs2)
{
	FlagSet		fsCross;
	FLMUINT		uiLoop1;
	FLMUINT		uiCrossProductElems = this->getNumElements() * fs2.getNumElements();
	FLMBYTE **	ppszCross = NULL;
	
	
	if( RC_BAD( f_alloc( sizeof( FLMBYTE *) * uiCrossProductElems, &ppszCross)))
	{
		f_assert( 0);
	}

	for ( uiLoop1 = 0; uiLoop1 < this->getNumElements(); uiLoop1++)
	{
		for ( FLMUINT uiLoop2 = 0; uiLoop2 < fs2.getNumElements(); uiLoop2++)
		{
			FLMUINT uiIndex = (uiLoop1 * fs2.getNumElements()) + uiLoop2;

			if( RC_BAD( f_alloc(
				f_strlen( (char *)this->m_ppucElemArray[ uiLoop1]) + 
					f_strlen( (char *)fs2.m_ppucElemArray[ uiLoop2]) + 1,
					&ppszCross[ uiIndex])))
			{
				f_assert( 0);
			}
			
			f_strcpy( (char *)ppszCross[ uiIndex], (char *)this->m_ppucElemArray[ uiLoop1]);
			f_strcat( (char *)ppszCross[ uiIndex], (char *)fs2.m_ppucElemArray[ uiLoop2]);
		}
	}
	fsCross.init( ppszCross, uiCrossProductElems);

	for( uiLoop1 = 0; uiLoop1 < uiCrossProductElems; uiLoop1++)
	{
		f_free( &ppszCross[ uiLoop1]);
	}
	
	f_free( &ppszCross);
	return( fsCross);
}

/****************************************************************************
Desc:
****************************************************************************/
FlagSet& FlagSet::operator=( const FlagSet& fs)
{
	if ( this != &fs)
	{
		if ( m_ppucElemArray || m_pbFlagArray)
		{
			this->reset();
		}
		this->init( fs.m_ppucElemArray, fs.m_uiNumElems);
	}
	return *this;
}

/****************************************************************************
Desc:
****************************************************************************/
FlagSet::FlagSet( const FlagSet& fs)
{
	m_ppucElemArray = NULL;
	m_pbFlagArray = NULL;

	f_alloc( sizeof( FLMBYTE *) * fs.m_uiNumElems, &m_ppucElemArray);
	f_alloc( sizeof( FLMBOOL) * fs.m_uiNumElems, &m_pbFlagArray);
	
	f_memset( m_pbFlagArray, 0, sizeof( FLMBOOL) * fs.m_uiNumElems);

	for ( FLMUINT uiLoop = 0; uiLoop < fs.m_uiNumElems; uiLoop++)
	{
		f_alloc( f_strlen( (char *)fs.m_ppucElemArray[uiLoop]) + 1,
			&m_ppucElemArray[ uiLoop]);
		f_strcpy( (char *)m_ppucElemArray[uiLoop], (char *)fs.m_ppucElemArray[uiLoop]);
	}

	m_uiNumElems = fs.m_uiNumElems;
}

/****************************************************************************
Desc:
****************************************************************************/
void FlagSet::reset()
{
	for( FLMUINT uiLoop = 0; uiLoop < m_uiNumElems; uiLoop++)
	{
		f_free( &m_ppucElemArray[ uiLoop]);
	}
	
	f_free( &m_ppucElemArray);
	f_free( &m_pbFlagArray);

	m_uiNumElems = 0;
	m_ppucElemArray = NULL;
	m_pbFlagArray = NULL;
}

/****************************************************************************
Desc:
****************************************************************************/
void FlagSet::init( FLMBYTE ** ppucElemArray, FLMUINT uiNumElems)
{
	reset();
	
	f_alloc( sizeof( FLMBYTE *) * uiNumElems, &m_ppucElemArray);
	f_alloc( sizeof( FLMBOOL) * uiNumElems, &m_pbFlagArray);
	
	f_memset( m_pbFlagArray, 0, sizeof( FLMBOOL) * uiNumElems);
	for ( FLMUINT uiLoop = 0; uiLoop < uiNumElems; uiLoop++)
	{
		f_alloc( f_strlen( (char *)ppucElemArray[uiLoop]) + 1, 
			&m_ppucElemArray[ uiLoop]);
		f_strcpy( (char *)m_ppucElemArray[uiLoop], (char *)ppucElemArray[uiLoop]);
	}
	
	m_uiNumElems = uiNumElems;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE createUnitTest(
	const char *		configPath, 
	const char *		buildNum, 
	const char *		environment,  
	const char *		user, 
	unitTestData *		uTD)
{
	RCODE					rc = NE_XFLM_OK;
	IF_FileSystem *	pFileSystem = NULL;
	IF_FileHdl *		pConfigFileHdl = NULL;
	IF_FileHdl *		pCSVFileHdl = NULL;
	char					buffer[ MAX_BUFFER_SIZE] = "";
	FLMUINT				size = MAX_BUFFER_SIZE;
	FLMUINT64			ui64Tmp;
	char *				strPos1 = NULL;
	char *				strPos2 = NULL;

	if( !configPath || !buildNum || !environment || !uTD || !user)
	{
		flmAssert(0);
	}

	if( f_strlen(user) > MAX_SMALL_BUFFER_SIZE)
	{
		rc = RC_SET( NE_XFLM_BUFFER_OVERFLOW);
		goto Exit;
	}
	else
	{
		f_strcpy(uTD->userName, user);
	}

	if( f_strlen(environment) > MAX_SMALL_BUFFER_SIZE)
	{
		rc = RC_SET( NE_XFLM_BUFFER_OVERFLOW);
		goto Exit;
	}
	else
	{
		f_strcpy(uTD->environment, environment);
	}

	if( f_strlen(buildNum) > MAX_SMALL_BUFFER_SIZE)
	{
		rc = RC_SET( NE_XFLM_BUFFER_OVERFLOW);
		goto Exit;
	}
	else
	{
		f_strcpy(uTD->buildNumber, buildNum);
	}
	
	if( RC_BAD( rc = FlmGetFileSystem( &pFileSystem)))
	{
		goto Exit;
	}

	if( configPath [0])
	{
		if( RC_BAD( rc = pFileSystem->openFile( configPath, 
			FLM_IO_RDONLY | FLM_IO_SH_DENYNONE, &pConfigFileHdl)))
		{
			goto Exit;
		}

		if ( RC_BAD( rc = pConfigFileHdl->size( &ui64Tmp)))
		{
			goto Exit;
		}
		
		size = (FLMUINT)ui64Tmp;

		if ( RC_BAD( rc = pConfigFileHdl->read(
			0, size, buffer, &size)))
		{
			goto Exit;
		}

	#ifdef FLM_WIN
		{
			char *	temp;
			f_alloc( size, &temp);
			
			char *	tempbegin = temp;
			size_t	newsize = size;
			
			

			for( unsigned int i = 0; i < size; i++)
			{
				if ( ( ( i + 1) < size) 
					&& ( buffer[i] == 0x0D && buffer[i + 1] == 0x0A))
				{
					*temp++ = 0x0A;
					i++;
					newsize--;
				}
				else
				{
					*temp++ = buffer[i];
				}
			}
			
			f_memcpy( buffer, tempbegin, (FLMSIZET)newsize);
			size = newsize;
		}
	#endif

		// Get the FOLDER
		
		strPos1 = f_strchr(buffer, ':');
		strPos2 = f_strchr(strPos1, '\n');
		if(!strPos1 || !strPos2)
		{
			rc = RC_SET( NE_XFLM_FAILURE);
			goto Exit;
		}

		for( strPos1++; *strPos1 == ' ' || *strPos1 == '\t'; strPos1++);
		
		if(strPos2-strPos1 > MAX_SMALL_BUFFER_SIZE)
		{
			rc = RC_SET( NE_XFLM_BUFFER_OVERFLOW);
			goto Exit;
		}
		
		f_strncpy(uTD->folder, strPos1, strPos2-strPos1);
		uTD->folder[strPos2-strPos1] = '\0';

		// Get the ATTRIBUTES
		
		strPos1 = f_strchr(strPos1, ':');
		strPos2 = f_strchr(strPos1, '\n');
		
		if(!strPos1 || !strPos2)
		{
			rc = RC_SET( NE_XFLM_FAILURE);
			goto Exit;
		}

		for(strPos1++;*strPos1 == ' ' || *strPos1 == '\t';strPos1++);
		
		if( strPos2-strPos1 > MAX_SMALL_BUFFER_SIZE)
		{
			rc = RC_SET( NE_XFLM_FAILURE);
			goto Exit;
		}
		
		f_strncpy(uTD->attrs, strPos1, strPos2-strPos1);
		uTD->attrs[strPos2-strPos1] = '\0';

		// Get the CSVFILE
		
		strPos1 = f_strchr(strPos1, ':');
		strPos2 = f_strchr(strPos1, '\n');
		
		// Allow for possible \r
		
		if( *( --strPos2) != '\r')
		{
			strPos2++;
		}

		if( !strPos1 || !strPos2)
		{
			rc = RC_SET( NE_XFLM_FAILURE);
			goto Exit;
		}

		for( strPos1++;*strPos1 == ' ' || *strPos1 == '\t';strPos1++);

		if( strPos2-strPos1 > MAX_SMALL_BUFFER_SIZE)
		{
			rc = RC_SET( NE_XFLM_FAILURE);
			goto Exit;
		}

		f_strncpy(uTD->csvFilename, strPos1, strPos2-strPos1);
		uTD->csvFilename[strPos2-strPos1] = '\0';

		rc = pFileSystem->openFile( 
			uTD->csvFilename, FLM_IO_RDWR | FLM_IO_SH_DENYNONE, &pCSVFileHdl);

		if( RC_BAD( rc))
		{
			if ( rc == NE_FLM_IO_PATH_NOT_FOUND)
			{
				// Create the file and write the header
				
				if (RC_BAD( rc = f_filecat( uTD->csvFilename, DATA_ORDER)))
				{
					goto Exit;
				}
			}
		}
		else
		{
			goto Exit;
		}
	}

Exit:

	if( pConfigFileHdl)
	{
		pConfigFileHdl->Release();
	}
	
	if( pCSVFileHdl)
	{
		pCSVFileHdl->Release();
	}

	if( pFileSystem)
	{
		pFileSystem->Release();
	}
	
	return( rc);
}

/****************************************************************************
Desc:	After each unit test call this to record the unit test status
		to a CSV file.
			uTD - contains the configuration information
			testName - a Unique for this module unite test name.
			testDescr - A description of the unit test.
			steps - The steps the unit test performs.
			status - Maybe be PASS or FAIL
			resultDetails - details that explain the result status.
****************************************************************************/
RCODE recordUnitTestResults(
	unitTestData *		uTD, 
	const char *		testName, 
	const char *		testDescr, 
	const char *		steps, 
	const char *		status, 
	const char *		resultDetails, 
	const char *		elapsedTime)
{
	RCODE		rc = NE_XFLM_OK;
	char		buffer[MAX_BUFFER_SIZE];

	if( !testName || !testDescr || !steps || !status || !resultDetails || !uTD )
	{
		flmAssert(0);
	}

	//VISIT - re-enable the elapsed time reporting when the TCB can support it
	
	(void)elapsedTime;

	f_sprintf( buffer, "%s,%s,%s,%s,%s,%s,%s,%s,"/*%s,*/"%s,%s\n", 
		testName, uTD->userName, testDescr, steps, uTD->buildNumber, status, 
		uTD->environment, resultDetails, /*elapsedTime ? elapsedTime : "",*/ 
		uTD->attrs, uTD->folder);

	if( RC_BAD( rc = f_filecat( uTD->csvFilename, buffer)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE TestBase::init(
	FLMBOOL			bLog,
	const char *	pszLogfile,
	FLMBOOL			bDisplay,
	FLMBOOL			bVerboseDisplay,
	const char *	pszConfigFile,
	const char *	pszEnvironment,
	const char *	pszBuild,
	const char *	pszUser)
{
	RCODE				rc = NE_XFLM_MEM;
	
	if( RC_BAD( rc = FlmAllocDbSystem( &m_pDbSystem)))
	{
		goto Exit;
	}

	m_bLog = bLog;
	m_bDisplay = bDisplay;

	// Set up logger and displayer if true
	
	if( m_bLog)
	{
		if( ( m_pLogger = f_new IFlmTestLogger) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}

		if( RC_BAD( rc = m_pLogger->init( pszLogfile)))
		{
			goto Exit;
		}
	}
	
	if( m_bDisplay)
	{
		if( (m_pDisplayer = f_new IFlmTestDisplayer) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}
		
		if( RC_BAD( rc = m_pDisplayer->init()))
		{
			goto Exit;
		}
	}

	if( (m_pReporter = f_new ITestReporter) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = m_pReporter->init( pszConfigFile, pszBuild,
		pszEnvironment, pszUser)))
	{
		goto Exit;
	}
	m_bDisplayVerbose = bVerboseDisplay;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void TestBase::beginTest( 
	const char * 	pszTestName, 
	const char * 	pszTestDesc,
	const char * 	pszTestSteps,
	const char * 	pszDetails)
{
	char		szTemp[ 256];

	m_pszTestName = pszTestName;
	m_pszTestDesc = pszTestDesc;
	m_pszSteps = pszTestSteps;

	if( m_bDisplayVerbose)
	{
		displayLine(
			"========================================"
			"=======================================");
			
		f_sprintf( szTemp, "Test Name: %s", m_pszTestName);
		displayLine( szTemp);
		
		f_sprintf( szTemp, "Test Description: %s", m_pszTestDesc);
		displayLine( szTemp);
		
		f_sprintf( szTemp, "Steps: %s", m_pszSteps);
		displayLine( szTemp);
	}
	else
	{
		f_sprintf( szTemp, "Test Name: %s ... ", m_pszTestName);
		display( szTemp);
	}

	f_strcpy( m_szDetails, pszDetails);
	m_ui64StartMs = FLM_TIMER_UNITS_TO_SECS( FLM_GET_TIMER());
}

/****************************************************************************
Desc:
****************************************************************************/
void TestBase::endTest( 
	const char * 	pszTestResult)
{
	m_ui64EndMs = FLM_TIMER_UNITS_TO_SECS( FLM_GET_TIMER());
	outputAll( pszTestResult, m_ui64EndMs - m_ui64StartMs);
}

/****************************************************************************
Desc:
****************************************************************************/
void TestBase::log(
	const char *	pszString)
{
	if( m_bLog)
	{
		m_pLogger->appendString( pszString);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
void TestBase::display(
	const char * 	pszString)
{
	if( m_bDisplay)
	{
		m_pDisplayer->appendString( pszString);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
void TestBase::displayLine(
	const char * 	pszString)
{
	if( m_bDisplay)
	{
		m_pDisplayer->appendString( pszString);
		m_pDisplayer->appendString( "\n");
	}
}

/****************************************************************************
Desc:
****************************************************************************/
void TestBase::displayTime( 
	FLMUINT64		ui64Milli,
	const char *	pszIntro)
{
	const char *		pszDefault = "Elapsed Time: ";
	char 					szTimeBuf[ 64];
	char * 				pszTempBuf = NULL;
	
	if( pszIntro)
	{
		f_alloc( f_strlen( pszIntro) + sizeof( szTimeBuf), &pszTempBuf);
		f_strcpy( pszTempBuf, pszIntro);
	}
	else
	{
		f_alloc( f_strlen( pszDefault) + sizeof( szTimeBuf), &pszTempBuf);
		f_strcpy( pszTempBuf, pszDefault);
	}
	
	normalizeTime( ui64Milli, szTimeBuf);
	f_strcat( pszTempBuf, szTimeBuf);
	displayLine( pszTempBuf);
}

/****************************************************************************
Desc:
****************************************************************************/
void TestBase::normalizeTime( 
	FLMUINT64		ui64Milli, 
	char * 			pszString)
{
	FLMUINT64 	ui64Ms;
	FLMUINT64 	ui64S;
	FLMUINT64 	ui64M;
	FLMUINT64 	ui64H;

	ui64Ms = ui64Milli % 1000;
	ui64S = ui64Milli / 1000;

	ui64M = ui64S / 60;
	ui64S %= 60;

	ui64H = ui64M / 60;
	ui64M %= 60;

	f_sprintf( pszString, "%02I64u:%02I64u:%02I64u.%03I64u", 
		ui64H, ui64M, ui64S, ui64Ms);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE TestBase::initCleanTestState( 
	const char * 	pszDibName)
{
	RCODE						rc = NE_XFLM_OK;
	XFLM_CREATE_OPTS		createOpts;

	// Create the database
	
	f_memset( &createOpts, 0, sizeof( XFLM_CREATE_OPTS));
	
	if ( RC_BAD( rc = m_pDbSystem->dbCreate( pszDibName, NULL, NULL, NULL,
		NULL, &createOpts, &m_pDb)))
	{
		if( rc == NE_XFLM_FILE_EXISTS)
		{
			if( RC_BAD( rc = m_pDbSystem->dbRemove( pszDibName, NULL, NULL, TRUE)))
			{
				goto Exit;
			}
		}

		if( RC_BAD( rc = m_pDbSystem->dbCreate( pszDibName, NULL, NULL, NULL,
			NULL, &createOpts, &m_pDb)))
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
RCODE TestBase::openTestState(
	const char *		pszDibName)
{
	XFLM_CREATE_OPTS		createOpts;
	IF_FileSystem *		pFileSys = NULL;
	RCODE						rc = NE_XFLM_OK;

	if( RC_BAD( rc = FlmGetFileSystem( &pFileSys)))
	{
		goto Exit;
	}
	if ( RC_BAD( rc = pFileSys->doesFileExist( pszDibName)))
	{
		// Create the database

		f_memset( &createOpts, 0, sizeof( XFLM_CREATE_OPTS));
		
		if ( RC_BAD( rc = m_pDbSystem->dbCreate( pszDibName, NULL, NULL, NULL,
			NULL, &createOpts, &m_pDb)))
		{
			goto Exit;
		}
	}
	else
	{
		// Open the existing database

		if( RC_BAD( rc = m_pDbSystem->dbOpen( pszDibName, NULL, NULL, 
			NULL, FALSE, &m_pDb)))
		{
			goto Exit;
		}
	}

Exit:

	if( pFileSys)
	{
		pFileSys->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE TestBase::shutdownTestState(
	const char *	pszDibName,
	FLMBOOL			bRemoveDib)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiRefCount;

	if( bRemoveDib)
	{
		if( m_pDb)
		{
			uiRefCount = m_pDb->Release();
			flmAssert( uiRefCount == 0);
			m_pDb = NULL;
		}
		
		if( RC_BAD( rc = m_pDbSystem->dbRemove( 
			pszDibName, NULL, NULL, TRUE)))
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
RCODE TestBase::checkQueryResults( 
	const char **	ppszResults, 
	FLMUINT			uiNumResultsExpected, 
	IF_Query *		pQuery,
	char *			pszDetails)
{
	RCODE				rc = NE_XFLM_OK;
	FlagSet			flagSet;
	IF_DOMNode *	pReturn = NULL;
	FLMUINT			uiLoop;
	char				szBuffer[ 500];
	
	flagSet.init( (FLMBYTE **)ppszResults, uiNumResultsExpected);

	for( uiLoop = 0; ; uiLoop++)
	{
		if( !uiLoop)
		{
			if( RC_BAD( rc = pQuery->getFirst( m_pDb, &pReturn)))
			{
				MAKE_FLM_ERROR_STRING( "getFirst failed.", pszDetails, rc);
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = pQuery->getNext( m_pDb, &pReturn)))
			{
				if( uiLoop < uiNumResultsExpected - 1)
				{
					MAKE_FLM_ERROR_STRING( "getNext failed.", pszDetails, rc);
					goto Exit;
				}
				else
				{
					rc = NE_XFLM_OK;
					break;
				}
			}
		}

		if( RC_BAD( rc = pReturn->getUTF8( m_pDb, (FLMBYTE *)szBuffer, 
			sizeof(szBuffer), 0, sizeof( szBuffer) - 1)))
		{
			MAKE_FLM_ERROR_STRING( "getUTF8 failed.", pszDetails, rc);
			goto Exit;
		}
		
		if ( !flagSet.setElemFlag( (FLMBYTE *)szBuffer))
		{
			rc = NE_XFLM_FAILURE;
			MAKE_FLM_ERROR_STRING( "Unexpected result received.", pszDetails, rc);
			goto Exit;
		}
	}

	if( !flagSet.allElemFlagsSet())
	{
		rc = NE_XFLM_FAILURE;
		MAKE_FLM_ERROR_STRING( "Expected results not received.", 
			pszDetails, rc);
		goto Exit;
	}
	
Exit:

	if( pReturn)
	{
		pReturn->Release();
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE TestBase::doQueryTest( 
	const char *	pszQueryString,
	const char **	ppszExpectedResults,
	FLMUINT			uiNumResultsExpected,
	IF_Query *		pQuery,
	char *			pszDetails,
	FLMUINT			uiRequestedIndex,
	FLMUINT *		puiIndexUsed)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBOOL		bHaveMult;

	if( pszQueryString)
	{
		if( RC_BAD( rc = pQuery->setupQueryExpr( m_pDb, pszQueryString)))
		{
			MAKE_FLM_ERROR_STRING( "setupQueryExpr failed.", pszDetails, rc);
			goto Exit;
		}
	}

	if( uiRequestedIndex)
	{
      if( RC_BAD( rc = pQuery->setIndex( uiRequestedIndex)))
		{
			MAKE_FLM_ERROR_STRING( "setIndex failed.", pszDetails, rc);
			goto Exit;
		}
	}

	if( RC_BAD( rc = checkQueryResults( ppszExpectedResults, 
		uiNumResultsExpected, pQuery, pszDetails)))
	{
		goto Exit;
	}

	if( puiIndexUsed)
	{
		if( RC_BAD( rc = pQuery->getIndex( m_pDb, puiIndexUsed, &bHaveMult)))
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
RCODE TestBase::logTestResults(
	const char * 	pszTestResult,
	FLMUINT64 		ui64ElapsedTime)
{
	RCODE			rc = NE_XFLM_OK;
	char *		pszTemp = NULL;
	char			szTime[ 64];

	if( RC_BAD( rc = f_alloc( DETAILS_BUF_SIZ + 64, &pszTemp)))
	{
		goto Exit;
	}

	if( ui64ElapsedTime != ~((FLMUINT64)0))
	{
		normalizeTime( ui64ElapsedTime, szTime);
	}
	else
	{
		f_strcpy( szTime, "Not Recorded");
	}

	log(
			"========================================"
			"=======================================");
			
	f_sprintf( pszTemp, "Test Name: %s", m_pszTestName);
	log( pszTemp);
	
	f_sprintf( pszTemp, "Test Description: %s", m_pszTestDesc);
	log( pszTemp);
	
	f_sprintf( pszTemp, "Steps: %s", m_pszSteps);
	log( pszTemp);
	
	f_sprintf( pszTemp, "Test Result: %s", pszTestResult);
	log( pszTemp);
	
	f_sprintf( pszTemp, "Details: %s", m_szDetails);
	log( pszTemp);
	
	f_sprintf( pszTemp, "Elapsed Time: %s", szTime);
	log( pszTemp);
	
	log(
			"========================================"
			"=======================================");

Exit:

	if( pszTemp)
	{
		f_free( &pszTemp);
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE TestBase::displayTestResults(
	const char *	pszTestResult,
	FLMUINT64		ui64ElapsedTime)
{
	RCODE			rc = NE_XFLM_OK;
	char *		pszTemp = NULL;
	char			szTime[ 64];

	if( RC_BAD( rc = f_alloc( DETAILS_BUF_SIZ + 64, &pszTemp)))
	{
		goto Exit;
	}

	if( ui64ElapsedTime != ~((FLMUINT64)0))
	{
		normalizeTime( ui64ElapsedTime, szTime);
	}
	else
	{
		f_strcpy( szTime, "Not Recorded");
	}

	if( m_bDisplayVerbose)
	{
		f_sprintf( pszTemp, "Test Result: %s", pszTestResult);
		displayLine( pszTemp);
		
		f_sprintf( pszTemp, "Details: %s", m_szDetails);
		displayLine( pszTemp);
		
		f_sprintf( pszTemp, "Elapsed Time: %s", szTime);
		displayLine( pszTemp);
		
		displayLine(
			"========================================"
			"=======================================");
	}
	else
	{
		f_sprintf( pszTemp, "Result: %s", pszTestResult);
		displayLine( pszTestResult);
	}

Exit:

	if( pszTemp)
	{
		f_free( &pszTemp);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE TestBase::outputAll( 
	const char * 	pszTestResult,
	FLMUINT64 		ui64ElapsedTime)
{
	RCODE			rc = NE_XFLM_OK;
	char *		pszTime = NULL;

	if( ui64ElapsedTime != ~((FLMUINT64)0))
	{
		if( RC_BAD( rc = f_alloc( 64, &pszTime)))
		{
			goto Exit;
		}

		normalizeTime( ui64ElapsedTime, pszTime);
	} 

	if( RC_BAD( rc = displayTestResults( pszTestResult, ui64ElapsedTime)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = logTestResults( pszTestResult, ui64ElapsedTime)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pReporter->recordUnitTestResults(
		m_pszTestName, m_pszTestDesc, m_pszSteps, pszTestResult, m_szDetails,
		pszTime)))
	{
		goto Exit;
	}

Exit:

	if( pszTime)
	{
		f_free( &pszTime);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE TestBase::importBuffer(
	 const char *		pszBuffer,
	 FLMUINT				uiCollection)
{
	 RCODE 		rc = NE_XFLM_OK;

	 if( RC_BAD( rc = m_pDbSystem->openBufferIStream( pszBuffer,
		 f_strlen( pszBuffer), &m_pInputStream)))
	{
		MAKE_FLM_ERROR_STRING( "openBufferIStream failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->import( m_pInputStream, uiCollection)))
	{
		goto Exit;
	}

Exit:

	if( m_pInputStream)
	{
		m_pInputStream->Release();
		m_pInputStream = NULL;
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE TestBase::importDocument(
	 const char *		pszBuffer,
	 FLMUINT				uiCollection)
{
	 RCODE 		rc = NE_XFLM_OK;

	 if( RC_BAD( rc = m_pDbSystem->openBufferIStream( pszBuffer,
		 f_strlen( pszBuffer), &m_pInputStream)))
	{
		MAKE_FLM_ERROR_STRING( "openBufferIStream failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->importDocument( m_pInputStream, uiCollection)))
	{
		goto Exit;
	}

Exit:

	if( m_pInputStream)
	{
		m_pInputStream->Release();
		m_pInputStream = NULL;
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE TestBase::importFile(
	 const char *		pszFilename,
	 FLMUINT	 			uiCollection)
{
	RCODE 	rc = NE_XFLM_OK;
	
	if( RC_BAD( rc = m_pDbSystem->openFileIStream( pszFilename,
		&m_pInputStream)))
	{
		MAKE_FLM_ERROR_STRING( "openFileIStream failed.", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->import( m_pInputStream, uiCollection)))
	{
		MAKE_FLM_ERROR_STRING( "file import failed.", m_szDetails, rc);
		goto Exit;
	}

Exit:

	if( m_pInputStream)
	{
		m_pInputStream->Release();
		m_pInputStream = NULL;
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
TestBase::~TestBase()
{
	if( m_pLogger)
	{
		m_pLogger->Release();
	}
	
	if( m_pDisplayer)
	{
		m_pDisplayer->Release();
	}
	
	if( m_pReporter)
	{
		m_pReporter->Release();
	}
	
	if( m_pInputStream)
	{
		 m_pInputStream->Release();
	}

	if( m_pDb)
	{
		m_pDb->Release();
	}

	if( m_pDbSystem)
	{
		m_pDbSystem->Release();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
FLMINT TestBase::unicmp(
	FLMUNICODE *	puzStr1,
	FLMUNICODE *	puzStr2)
{
	while( *puzStr1 == *puzStr2 && *puzStr1)
	{
		puzStr1++;
		puzStr2++;
	}

	return( (FLMINT)*puzStr1 - (FLMINT)*puzStr2);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE TestBase::createCompoundDoc(
	ELEMENT_NODE_INFO *	pElementNodes,
	FLMUINT					uiNumElementNodes,
	FLMUINT64 *				pui64DocId)
{
	IF_DOMNode *	pDocRoot = NULL;
	IF_DOMNode *	pNode = NULL;
	FLMUINT			uiLoop;
	RCODE				rc = NE_XFLM_OK;

	if( pui64DocId)
	{
		*pui64DocId = 0;
	}
   
	if( RC_BAD( rc = m_pDb->createRootElement( 
		XFLM_DATA_COLLECTION, ELM_ANY_TAG, &pDocRoot)))
	{
		MAKE_FLM_ERROR_STRING( "createRootElement failed.", m_szDetails, rc);
		goto Exit;
	}

	for( uiLoop = 0; uiLoop < uiNumElementNodes; uiLoop++)
	{
		if( isPlaceHolder(pElementNodes[uiLoop]))
		{
			continue;
		}

		if( RC_BAD( rc = pDocRoot->createNode(
			m_pDb, ELEMENT_NODE, pElementNodes[uiLoop].uiDictNum, 
			XFLM_LAST_CHILD, &pNode)))
		{
			MAKE_FLM_ERROR_STRING( "createNode failed.", m_szDetails, rc);
			goto Exit;
		}

		switch( pElementNodes[uiLoop].uiDataType)
		{
			case XFLM_TEXT_TYPE:
			{
				if( RC_BAD( rc = pNode->setUTF8(
					m_pDb, (FLMBYTE *)pElementNodes[uiLoop].pvData)))
				{
					goto Exit;
				}
				
				break;
			}
			
			case XFLM_NUMBER_TYPE:
			{
				if ( RC_BAD( rc = pNode->setUINT(
					m_pDb, (FLMUINT)pElementNodes[uiLoop].pvData)))
				{
					goto Exit;
				}
				
				break;
			}
			
			case XFLM_BINARY_TYPE:
			{
				if ( RC_BAD( rc = pNode->setBinary(
					m_pDb, (FLMBYTE*)pElementNodes[uiLoop].pvData, 
					pElementNodes[uiLoop].uiDataSize)))
				{
					goto Exit;
				}
				break;
			}
			
			default:
			{
				flmAssert( 0);
			}
		}
	}
	
	if( RC_BAD ( rc = m_pDb->documentDone( pDocRoot)))
	{
		MAKE_FLM_ERROR_STRING( "documentDone failed.", m_szDetails, rc);
		goto Exit;
	}

	if( pui64DocId)
	{
		if( RC_BAD( rc = pDocRoot->getNodeId( m_pDb, pui64DocId)))
		{
			goto Exit;
		}
	}

Exit:

	if( pNode)
	{
		pNode->Release();
	}

	if( pDocRoot)
	{
		pDocRoot->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
ArgList::ArgList()
{
	m_uiCapacity = INIT_SIZE;
	m_uiNumEntries = 0;
	
	f_alloc( m_uiCapacity * sizeof( char *), &m_ppszArgs);
}

/****************************************************************************
Desc:
****************************************************************************/
ArgList::~ArgList()
{
	FLMUINT	uiLoop;

	if( m_ppszArgs)
	{
		for( uiLoop = 0; uiLoop < m_uiNumEntries; uiLoop++)
		{
			f_free( &( m_ppszArgs[uiLoop]));
		}
		
		f_free( &m_ppszArgs);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE ArgList::resize( void)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiLoop;
	char **		ppszTemp = NULL;

	if( RC_BAD( rc = f_alloc( m_uiCapacity * GROW_FACTOR * sizeof(char*), 
		&ppszTemp)))
	{
		goto Exit;
	}

	m_uiCapacity *= GROW_FACTOR;
	
	for( uiLoop = 0; uiLoop < m_uiNumEntries; uiLoop++)
	{
		if( RC_BAD( rc = f_alloc( f_strlen( m_ppszArgs[uiLoop]) + 1, 
			&ppszTemp[uiLoop])))
		{
			f_free( &ppszTemp);
			goto Exit;
		}
		
		f_strcpy( ppszTemp[uiLoop], m_ppszArgs[uiLoop]);
	}

	f_free( &m_ppszArgs);
	m_ppszArgs = ppszTemp;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT ArgList::getNumEntries( void)
{
	return( m_uiNumEntries);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE ArgList::addArg( 
	const char *	pszArg)
{
	RCODE		rc = NE_XFLM_OK;

	if( m_uiNumEntries >= m_uiCapacity)
	{
		if( RC_BAD( rc = resize()))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = f_alloc( f_strlen( pszArg) + 1,
		&m_ppszArgs[m_uiNumEntries])))
	{
		goto Exit;
	}

	f_strcpy( m_ppszArgs[ m_uiNumEntries++], pszArg);

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
const char * ArgList::getArg(
	FLMUINT		uiIndex)
{
	return( m_ppszArgs[ uiIndex]);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE ArgList::expandFileArgs( 
	const char *		pszFilename)
{
	RCODE					rc = NE_XFLM_OK;
	char					token[64];
	IF_FileSystem *	pFileSystem = NULL;
	IF_FileHdl *		pFileHdl = NULL;
	
	if( RC_BAD( rc = FlmGetFileSystem( &pFileSystem)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pFileSystem->openFile( pszFilename, FLM_IO_RDWR,
		&pFileHdl)))
	{
		goto Exit;
	}

	while( RC_OK( rc = getTokenFromFile(token, pFileHdl)))
	{
		if( token[0] == '@')
		{
			if( RC_BAD( rc = expandFileArgs( &token[1])))
			{
				goto Exit;
			}
		}
		else
		{
			flmAssert(*token);
			if( RC_BAD( rc = addArg( token)))
			{
				goto Exit;
			}
		}
	}
	
	if( rc == NE_FLM_IO_END_OF_FILE)
	{
		rc = NE_XFLM_OK;
	}

Exit:

	if( pFileHdl)
	{
		pFileHdl->Release();
	}
	
	if( pFileSystem)
	{
		pFileSystem->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE ArgList::expandArgs(
	char ** 		ppszArgs,
	FLMUINT 		uiNumArgs)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiLoop;

	for( uiLoop = 0; uiLoop < uiNumArgs; uiLoop++)
	{
		if( ppszArgs[uiLoop][0] == '@')
		{
			if( RC_BAD( rc = expandFileArgs( &ppszArgs[uiLoop][ 1])))
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = addArg( ppszArgs[ uiLoop])))
			{
				goto Exit;
			}
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
const char * ArgList::operator []( FLMUINT uiIndex)
{
	return( m_ppszArgs[ uiIndex]);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE ArgList::getTokenFromFile(
	char * 			pszToken,
	IF_FileHdl * 	pFileHdl)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiSize = 1;
	FLMUINT64	ui64Offset = 0;
	FLMUINT64	ui64TrueOffset = 0;
	char			c;

	for(;;)
	{
Skip_WS_And_Comments:

		if ( RC_BAD( rc = pFileHdl->read( 0, 1, &c, &uiSize)))
		{
			goto Exit;
		}

		// Skip whitespace
		
		while( isWhitespace(c))
		{
			if( RC_BAD( rc = pFileHdl->read( 0, 1, &c, &uiSize)))
			{
				goto Exit;
			}
		}

		if( c == '#')
		{
			// Skip comment

			for (;;)
			{
				if( RC_BAD( rc = pFileHdl->read( 0, 1, &c, &uiSize)))
				{
					goto Exit;
				}

#ifdef FLM_UNIX
				// On unix platforms, an EOL is indicated by an LF
				
				if( c == 0x0A)
				{
					break;
				}
#else
				// On Windows and NetWare we need to look for CR/LF
				
				if( c == 0x0D)
				{

					if( RC_BAD( rc = pFileHdl->read( 0, 1, &c, &uiSize)))
					{
						goto Exit;
					}

					// Newline found
					
					if( c == 0x0A)
					{
						break;
					}
					else
					{
						// Rewind
						
						ui64Offset = 0;
						ui64TrueOffset = 0;

						if( RC_BAD( rc = pFileHdl->tell( &ui64Offset)))
						{
							goto Exit;
						}
						
						ui64Offset--;

						if( RC_BAD( rc = pFileHdl->seek( ui64Offset, 
							FLM_IO_SEEK_SET, &ui64TrueOffset)))
						{
							goto Exit;
						}
					}
				}
#endif
			}
			
			goto Skip_WS_And_Comments;
		}

		while( !isWhitespace( c))
		{
			if( c == '#')
			{
				break;
			}

			*pszToken++ = c;
			if( RC_BAD( rc = pFileHdl->read( 0, 1, &c, &uiSize)))
			{
				goto Exit;
			}
		}

		// Put the char back
		
		if( RC_BAD( rc = pFileHdl->tell( &ui64Offset)))
		{
			goto Exit;
		}
		ui64Offset--;

		if( RC_BAD( rc = pFileHdl->seek( ui64Offset, 
			FLM_IO_SEEK_SET, &ui64TrueOffset)))
		{
			goto Exit;
		}
		break;
	}

Exit:

	*pszToken = '\0';
	return( rc);
}
