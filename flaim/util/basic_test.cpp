//-------------------------------------------------------------------------
// Desc:	Basic unit test.
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

#include "flmunittest.h"

FSTATIC const char * gv_pszSampleDictionary =	
	"0 @1@ field Person\n"
	" 1 type text\n"
	"0 @2@ field LastName\n"
	" 1 type text\n"
	"0 @3@ field FirstName\n"
	" 1 type text\n"
	"0 @4@ field Age\n"
	" 1 type number\n"
	"0 @5@ field Misc\n"
	" 1 type binary\n"
	"0 @6@ field Numbers\n"
	" 1 type number\n"
	"0 @100@ index LastFirst_IX\n"
	" 1 language US\n"
	" 1 key\n"
	"  2 field 2\n"
	"   3 required\n"
	"  2 field 3\n"
	"   3 required\n"
	"0 @101@ index NUMBER_IX\n"
	" 1 language US\n"
	" 1 key\n"
	"  2 field 6\n"
	"   3 required\n";

#define PERSON_TAG						1
#define LAST_NAME_TAG					2
#define FIRST_NAME_TAG					3
#define AGE_TAG							4
#define MISC_TAG							5
#define NUMBER_TAG						6
#define LAST_NAME_FIRST_NAME_IX		100
#define NUMBER_IX							101

#ifdef FLM_NLM
	#define DB_NAME_STR					"SYS:\\SAMPLE.DB"
	#define DB_COPY_NAME_STR			"SYS:\\SAMPLECOPY.DB"
	#define DB_RENAME_NAME_STR			"SYS:\\SAMPLERENAME.DB"
	#define DB_RESTORE_NAME_STR		"SYS:\\SAMPLERESTORE.DB"
	#define DB_REBUILD_NAME_STR		"SYS:\\SAMPLEREBUILD.DB"
	#define BACKUP_PATH					"SYS:\\SAMPLEBACKUP"
#else
	#define DB_NAME_STR					"sample.db"
	#define DB_COPY_NAME_STR			"samplecopy.db"
	#define DB_RENAME_NAME_STR			"samplerename.db"
	#define DB_RESTORE_NAME_STR		"samplerestore.db"
	#define DB_REBUILD_NAME_STR		"samplerebuild.db"
	#define BACKUP_PATH					"samplebackup"
#endif

typedef struct NUM_IX_VALUE
{
	FLMBOOL		bUnsigned;
	FLMUINT64	ui64Value;
	FLMINT64		i64Value;
} NUM_IX_VALUE;

#define NUM_NUM_KEYS	17
static NUM_IX_VALUE gv_ExpectedNumIxValues [NUM_NUM_KEYS] = 
{
	{FALSE,	0,											(FLMINT64)FLM_MIN_INT64},
	{FALSE,	0,											(FLMINT64)FLM_MIN_INT64 + 1},
	{FALSE,	0,											(FLMINT64)(FLM_MIN_INT32) - 1},
	{FALSE,	0,											(FLMINT64)(FLM_MIN_INT32)},
	{FALSE,	0,											(FLMINT64)(FLM_MIN_INT32) + 1},
	{FALSE,	0,											(FLMINT64)(-1)},
	{TRUE,	0,											0},
	{TRUE,	(FLMUINT64)FLM_MAX_INT32 - 1,		0},
	{TRUE,	(FLMUINT64)FLM_MAX_INT32,			0},
	{TRUE,	(FLMUINT64)FLM_MAX_INT32 + 1,		0},
	{TRUE,	(FLMUINT64)FLM_MAX_UINT32,			0},
	{TRUE,	(FLMUINT64)FLM_MAX_UINT32 + 1,	0},
	{TRUE,	(FLMUINT64)FLM_MAX_INT64 - 1,		0},
	{TRUE,	(FLMUINT64)FLM_MAX_INT64,			0},
	{TRUE,	(FLMUINT64)FLM_MAX_INT64 + 1,		0},
	{TRUE,	(FLMUINT64)FLM_MAX_UINT64 - 1,	0},
	{TRUE,	(FLMUINT64)FLM_MAX_UINT64,			0}
};
	
typedef struct FLMUINT_TEST
{
	FLMUINT		uiNum;
	RCODE			rcExpectedGetUINT;
	RCODE			rcExpectedGetINT;
	RCODE			rcExpectedGetUINT64;
	RCODE			rcExpectedGetINT64;
} FLMUINT_TEST;

typedef struct FLMUINT64_TEST
{
	FLMUINT64	ui64Num;
	RCODE			rcExpectedGetUINT;
	RCODE			rcExpectedGetINT;
	RCODE			rcExpectedGetUINT64;
	RCODE			rcExpectedGetINT64;
} UNSIGNED63_TEST;

typedef struct FLMINT_TEST
{
	FLMINT		iNum;
	RCODE			rcExpectedGetUINT;
	RCODE			rcExpectedGetINT;
	RCODE			rcExpectedGetUINT64;
	RCODE			rcExpectedGetINT64;
} FLMINT_TEST;

typedef struct FLMINT64_TEST
{
	FLMINT64		i64Num;
	RCODE			rcExpectedGetUINT;
	RCODE			rcExpectedGetINT;
	RCODE			rcExpectedGetUINT64;
	RCODE			rcExpectedGetINT64;
} FLMINT64_TEST;

/***************************************************************************
Desc: FLMUINT numbers to test
****************************************************************************/
#define NUM_FLMUINT_TESTS	3
static FLMUINT_TEST gv_FLMUINTTests [NUM_FLMUINT_TESTS] =
{
	// Number								GetFLMUINT RCODE				GetFLMINT RCODE			GetFLMUINT64 RCODE			GetFLMINT64 RCODE
	{0,										FERR_OK,							FERR_OK,						FERR_OK,							FERR_OK},
#ifdef FLM_64BIT
	{FLM_MAX_UINT,							FERR_OK,							FERR_CONV_NUM_OVERFLOW, FERR_OK,							FERR_CONV_NUM_OVERFLOW},
	{(FLMUINT)(FLM_MAX_UINT32),		FERR_OK,							FERR_OK,						FERR_OK,							FERR_OK},
#else
	{FLM_MAX_UINT,							FERR_OK,							FERR_CONV_NUM_OVERFLOW, FERR_OK,							FERR_OK},
	{(FLMUINT)(FLM_MAX_UINT32),		FERR_OK,							FERR_CONV_NUM_OVERFLOW, FERR_OK,							FERR_OK}
#endif
};

/***************************************************************************
Desc: FLMINT numbers to test
****************************************************************************/
#define NUM_FLMINT_TESTS	7
static FLMINT_TEST gv_FLMINTTests [NUM_FLMINT_TESTS] =
{
	// Number								GetFLMUINT RCODE				GetFLMINT RCODE			GetFLMUINT64 RCODE			GetFLMINT64 RCODE
	{-1,										FERR_CONV_NUM_UNDERFLOW,	FERR_OK,						FERR_CONV_NUM_UNDERFLOW,	FERR_OK},
	{FLM_MIN_INT,							FERR_CONV_NUM_UNDERFLOW,	FERR_OK,						FERR_CONV_NUM_UNDERFLOW,	FERR_OK},
	{FLM_MAX_INT,							FERR_OK,							FERR_OK,						FERR_OK,							FERR_OK},
	{(FLMINT)FLM_MIN_INT32,				FERR_CONV_NUM_UNDERFLOW,	FERR_OK,						FERR_CONV_NUM_UNDERFLOW,	FERR_OK},
	{(FLMINT)(FLM_MIN_INT32) + 1,		FERR_CONV_NUM_UNDERFLOW,	FERR_OK,						FERR_CONV_NUM_UNDERFLOW,	FERR_OK},
	{(FLMINT)(FLM_MAX_INT32),			FERR_OK,							FERR_OK,						FERR_OK,							FERR_OK},
	{(FLMINT)(FLM_MAX_INT32) - 1,		FERR_OK,							FERR_OK,						FERR_OK,							FERR_OK}
};

/***************************************************************************
Desc: FLMUINT64 numbers to test
****************************************************************************/
#ifndef FLM_64BIT
	#define NUM_FLMUINT64_TESTS	8
#else
	#define NUM_FLMUINT64_TESTS	7
#endif
static FLMUINT64_TEST gv_FLMUINT64Tests [NUM_FLMUINT64_TESTS] =
{
	// Number								GetFLMUINT RCODE				GetFLMINT RCODE			GetFLMUINT64 RCODE			GetFLMINT64 RCODE
	{0,										FERR_OK,							FERR_OK,						FERR_OK,							FERR_OK},
#ifndef FLM_64BIT
	{(FLMUINT64)(FLM_MAX_UINT) + 1,	FERR_CONV_NUM_OVERFLOW,		FERR_CONV_NUM_OVERFLOW,	FERR_OK,							FERR_OK},
#endif
#ifdef FLM_64BIT
	{(FLMUINT64)(FLM_MAX_UINT32) + 1,FERR_OK,							FERR_OK,						FERR_OK,							FERR_OK},
	{(FLMUINT64)(FLM_MAX_INT64) - 1,	FERR_OK,							FERR_OK,						FERR_OK,							FERR_OK},
	{(FLMUINT64)(FLM_MAX_INT64),		FERR_OK,							FERR_OK,						FERR_OK,							FERR_OK},
	{(FLMUINT64)(FLM_MAX_INT64) + 1,	FERR_OK,							FERR_CONV_NUM_OVERFLOW,	FERR_OK,							FERR_CONV_NUM_OVERFLOW},
	{FLM_MAX_UINT64 - 1,					FERR_OK,							FERR_CONV_NUM_OVERFLOW,	FERR_OK,							FERR_CONV_NUM_OVERFLOW},
	{FLM_MAX_UINT64,						FERR_OK,							FERR_CONV_NUM_OVERFLOW,	FERR_OK,							FERR_CONV_NUM_OVERFLOW},
#else
	{(FLMUINT64)(FLM_MAX_UINT32) + 1,FERR_CONV_NUM_OVERFLOW,		FERR_CONV_NUM_OVERFLOW,	FERR_OK,							FERR_OK},
	{(FLMUINT64)(FLM_MAX_INT64) - 1,	FERR_CONV_NUM_OVERFLOW,		FERR_CONV_NUM_OVERFLOW,	FERR_OK,							FERR_OK},
	{(FLMUINT64)(FLM_MAX_INT64),		FERR_CONV_NUM_OVERFLOW,		FERR_CONV_NUM_OVERFLOW,	FERR_OK,							FERR_OK},
	{(FLMUINT64)(FLM_MAX_INT64) + 1,	FERR_CONV_NUM_OVERFLOW,		FERR_CONV_NUM_OVERFLOW,	FERR_OK,							FERR_CONV_NUM_OVERFLOW},
	{FLM_MAX_UINT64 - 1,					FERR_CONV_NUM_OVERFLOW,		FERR_CONV_NUM_OVERFLOW,	FERR_OK,							FERR_CONV_NUM_OVERFLOW},
	{FLM_MAX_UINT64,						FERR_CONV_NUM_OVERFLOW,		FERR_CONV_NUM_OVERFLOW,	FERR_OK,							FERR_CONV_NUM_OVERFLOW}
#endif
};

/***************************************************************************
Desc: FLMINT64 numbers to test
****************************************************************************/
#ifndef FLM_64BIT
	#define NUM_FLMINT64_TESTS	9
#else
	#define NUM_FLMINT64_TESTS	7
#endif
static FLMINT64_TEST gv_FLMINT64Tests [NUM_FLMINT64_TESTS] =
{
	// Number								GetFLMUINT RCODE				GetFLMINT RCODE			GetFLMUINT64 RCODE			GetFLMINT64 RCODE
	{-1,										FERR_CONV_NUM_UNDERFLOW,	FERR_OK,						FERR_CONV_NUM_UNDERFLOW,	FERR_OK},
#ifndef FLM_64BIT
	{(FLMINT64)(FLM_MIN_INT) - 1,		FERR_CONV_NUM_UNDERFLOW,	FERR_CONV_NUM_UNDERFLOW,FERR_CONV_NUM_UNDERFLOW,	FERR_OK},
	{(FLMINT64)(FLM_MAX_INT) + 1,		FERR_OK,							FERR_CONV_NUM_OVERFLOW,	FERR_OK,							FERR_OK},
#endif
#ifdef FLM_64BIT
	{(FLMINT64)(FLM_MIN_INT32) - 1,	FERR_CONV_NUM_UNDERFLOW,	FERR_OK,						FERR_CONV_NUM_UNDERFLOW,	FERR_OK},
	{(FLMINT64)(FLM_MAX_INT32) + 1,	FERR_OK,							FERR_OK,						FERR_OK,							FERR_OK},
	{FLM_MIN_INT64,						FERR_CONV_NUM_UNDERFLOW,	FERR_OK,						FERR_CONV_NUM_UNDERFLOW,	FERR_OK},
	{FLM_MIN_INT64 + 1,					FERR_CONV_NUM_UNDERFLOW,	FERR_OK,						FERR_CONV_NUM_UNDERFLOW,	FERR_OK},
	{FLM_MAX_INT64 - 1,					FERR_OK,							FERR_OK,						FERR_OK,							FERR_OK},
	{FLM_MAX_INT64,						FERR_OK,							FERR_OK,						FERR_OK,							FERR_OK},
#else
	{(FLMINT64)(FLM_MIN_INT32) - 1,	FERR_CONV_NUM_UNDERFLOW,	FERR_CONV_NUM_UNDERFLOW,FERR_CONV_NUM_UNDERFLOW,	FERR_OK},
	{(FLMINT64)(FLM_MAX_INT32) + 1,	FERR_OK,							FERR_CONV_NUM_OVERFLOW,	FERR_OK,							FERR_OK},
	{FLM_MIN_INT64,						FERR_CONV_NUM_UNDERFLOW,	FERR_CONV_NUM_UNDERFLOW,FERR_CONV_NUM_UNDERFLOW,	FERR_OK},
	{FLM_MIN_INT64 + 1,					FERR_CONV_NUM_UNDERFLOW,	FERR_CONV_NUM_UNDERFLOW,FERR_CONV_NUM_UNDERFLOW,	FERR_OK},
	{FLM_MAX_INT64 - 1,					FERR_CONV_NUM_OVERFLOW,		FERR_CONV_NUM_OVERFLOW,	FERR_OK,							FERR_OK},
	{FLM_MAX_INT64,						FERR_CONV_NUM_OVERFLOW,		FERR_CONV_NUM_OVERFLOW,	FERR_OK,							FERR_OK}
#endif
};

FSTATIC const char * opToStr(
	QTYPES	eOp);
	
FSTATIC const char * addOrSubtractOne(
	FLMBOOL	bAddOne,
	FLMBOOL	bSubtractOne);
	
FSTATIC FLMBOOL nonNegResultMatchesNonNegValueExpr(
	FLMUINT64	ui64Result,
	FLMUINT64	ui64Value,
	QTYPES		eOp,
	FLMBOOL		bAddOne,
	FLMBOOL		bSubtractOne);
	
FSTATIC FLMBOOL nonNegResultMatchesNegValueExpr(
	FLMUINT64	ui64Result,
	FLMINT64		i64Value,
	QTYPES		eOp,
	FLMBOOL		bAddOne,
	FLMBOOL		bSubtractOne);
	
FSTATIC FLMBOOL negResultMatchesNonNegValueExpr(
	FLMINT64		i64Result,
	FLMUINT64	ui64Value,
	QTYPES		eOp,
	FLMBOOL		bAddOne,
	FLMBOOL		bSubtractOne);
	
FSTATIC FLMBOOL negResultMatchesNegValueExpr(
	FLMINT64		i64Result,
	FLMINT64		i64Value,
	QTYPES		eOp,
	FLMBOOL		bAddOne,
	FLMBOOL		bSubtractOne);
	
/***************************************************************************
Desc:
****************************************************************************/
class IFlmTestImpl : public TestBase
{
public:

	IFlmTestImpl()
	{
		m_hDb = HFDB_NULL;
	}
	
	virtual ~IFlmTestImpl()
	{
		if (m_hDb != HFDB_NULL)
		{
			(void)FlmDbClose( &m_hDb);
		}
	}

	FINLINE const char * getName( void)
	{
		return( "Basic Test");
	}
	
	RCODE createDbTest( void);
	
	RCODE addRecordTest(
		FLMUINT *	puiDrn);
		
	RCODE largeFieldTest( void);
	
	RCODE modifyRecordTest(
		FLMUINT	uiDrn);
	
	RCODE deleteRecordTest(
		FLMUINT	uiDrn);
		
	RCODE queryRecordTest( void);
		
	RCODE keyRetrieveTest(
		FLMUINT	uiIndex,
		FLMBOOL	bLastNameFirstNameIx);
	
	RCODE addIndexTest(
		FLMUINT *	puiIndex);
		
	RCODE addSubstringIndexTest(
		FLMUINT *	puiIndex);
		
	RCODE addPresenceIndexTest(
		FLMUINT *	puiIndex);
		
	RCODE deleteIndexTest(
		FLMUINT	uiIndex);
		
	RCODE deleteFieldTest(
		FLMUINT	uiFieldNum);
		
	RCODE suspendIndexTest(
		FLMUINT	uiIndex);
		
	RCODE resumeIndexTest(
		FLMUINT	uiIndex);
		
	RCODE sortedFieldsTest(
		FLMUINT *	puiDrn);
		
	RCODE sortedFieldsQueryTest(
		FLMUINT		uiDrn,
		FLMBOOL		bDoRootedFieldPaths);
		
	RCODE addRecWithFLMUINT(
		FLMUINT		uiNum,
		FLMUINT *	puiDrn);
		
	RCODE addRecWithFLMUINT64(
		FLMUINT64	ui64Num,
		FLMUINT *	puiDrn);
		
	RCODE addRecWithFLMINT(
		FLMINT		iNum,
		FLMUINT *	puiDrn);
		
	RCODE addRecWithFLMINT64(
		FLMINT64		i64Num,
		FLMUINT *	puiDrn);
		
	RCODE numbersTest(
		FLMUINT *	puiDrn);
		
	RCODE reopenDbTest( void);
		
	RCODE testNumField(
		FLMUINT *	puiDrn,
		FLMUINT		uiExpectedNum,
		RCODE			rcExpectedUINT,
		FLMINT		iExpectedNum,
		RCODE			rcExpectedINT,
		FLMUINT64	ui64ExpectedNum,
		RCODE			rcExpectedUINT64,
		FLMINT64		i64ExpectedNum,
		RCODE			rcExpectedINT64);
		
	RCODE numbersRetrieveTest(
		FLMUINT	uiDrn);
		
	RCODE numbersKeyRetrieveTest( void);
		
	RCODE verifyFLMUINT64ValueExpr(
		FLMUINT64	ui64Value,
		QTYPES		eOp,
		FLMBOOL		bAddOne,
		FLMBOOL		bSubtractOne,
		FlmRecord *	pRec);
		
	RCODE verifyFLMINT64ValueExpr(
		FLMINT64		i64Value,
		QTYPES		eOp,
		FLMBOOL		bAddOne,
		FLMBOOL		bSubtractOne,
		FlmRecord *	pRec);
		
	RCODE doFLMUINT64QueryTest(
		FLMUINT64	ui64Value,
		QTYPES		eOp,
		FLMBOOL		bAddOne,
		FLMBOOL		bSubtractOne);
		
	RCODE queryTestsFLMUINT64(
		FLMUINT64	ui64Value);
		
	RCODE doFLMINT64QueryTest(
		FLMINT64		i64Value,
		QTYPES		eOp,
		FLMBOOL		bAddOne,
		FLMBOOL		bSubtractOne);
		
	RCODE queryTestsFLMINT64(
		FLMINT64		i64Value);
		
	RCODE doFLMUINT32QueryTest(
		FLMUINT32	ui32Value,
		QTYPES		eOp,
		FLMBOOL		bAddOne,
		FLMBOOL		bSubtractOne);
		
	RCODE queryTestsFLMUINT32(
		FLMUINT32	ui32Value);
		
	RCODE doFLMINT32QueryTest(
		FLMINT32		i32Value,
		QTYPES		eOp,
		FLMBOOL		bAddOne,
		FLMBOOL		bSubtractOne);
		
	RCODE queryTestsFLMINT32(
		FLMINT32		i32Value);
		
	RCODE doNumQueryTests(
		NUM_IX_VALUE *	pValue);
		
	RCODE numbersQueryTest( void);
	
	RCODE backupRestoreDbTest( void);
	
	RCODE compareRecords(
		const char *	pszDb1,
		const char *	pszDb2,
		const char *	pszWhat,
		FlmRecord *		pRecord1,
		FlmRecord *		pRecord2);
		
	RCODE compareIndexes(
		const char *	pszDb1,
		const char *	pszDb2,
		HFDB				hDb1,
		HFDB				hDb2,
		FLMUINT			uiIndexNum);
		
	RCODE compareContainers(
		const char *	pszDb1,
		const char *	pszDb2,
		HFDB				hDb1,
		HFDB				hDb2,
		FLMUINT			uiContainerNum);
		
	RCODE compareDbTest(
		const char *	pszDb1,
		const char *	pszDb2);
		
	RCODE checkDbTest(
		const char *	pszDbName);
		
	RCODE rebuildDbTest(
		const char *	pszDestDbName,
		const char *	pszSrcDbName);
		
	RCODE copyDbTest(
		const char *	pszDestDbName,
		const char *	pszSrcDbName);
		
	RCODE renameDbTest(
		const char *	pszDestDbName,
		const char *	pszSrcDbName);
		
	RCODE reduceSizeTest(
		const char *	pszDbName);
		
	RCODE removeDbTest(
		const char *	pszDbName);
		
	RCODE execute( void);
	
private:

	HFDB	m_hDb;
};

const char * gv_pszFamilyNames[] =
{
	"Walton",
	"Abernathy",
	"Stillwell",
	"Anderson",
	"Armstrong",
	"Adamson",
	"Bagwell",
	"Ballard",
	"Bennett",
	"Blackman",
	"Bottoms",
	"Bradley",
	"Butterfield",
	"Cavanagh",
	"Chadwick",
	"Clark",
	"Crabtree",
	"Cunningham",
	"Darnell",
	"McClintock",
	"Davidson",
	"Dingman",
	"Doyle",
	"Eastman",
	"Ballantine",
	"Edmunds",
	"Neil",
	"Erickson",
	"Fetterman",
	"Finn",
	"Flanagan",
	"Gerber",
	"Thedford",
	"Thorman",
	"Gibson",
	"Gruszczynski",
	"Haaksman",
	"Hathaway",
	"Pernell",
	"Phillips",
	"Highsmith",
	"Hollingworth",
	"Frankenberger",
	"Hutchison",
	"Irving",
	"Weatherspoon",
	"Itaya",
	"Janiszewski",
	"Jenkins",
	"Jung",
	"Keller",
	"Jackson",
	"Kingsbury",
	"Klostermann",
	"Langley",
	"Liddle",
	"Lockhart",
	"Ludwig",
	"Kristjanson",
	"MacCormack",
	"Richards",
	"Robbins",
	"McAuliffe",
	"Merryweather",
	"Moynihan",
	"Muller",
	"Newland",
	"OCarroll",
	"Okuzawa",
	"Ortiz",
	"Pachulski",
	"Parmaksezian",
	"Peacocke",
	"Poole",
	"Prewitt",
	"Quigley",
	"Qureshi",
	"Ratcliffe",
	"Rundle",
	"Ryder",
	"Sampson",
	"Satterfield",
	"Sharkey",
	"Silverman",
	"Snedeker",
	"Goodman",
	"Spitzer",
	"Szypulski",
	"Talbott",
	"Trisko",
	"Turrubiarte",
	"Upchurch",
	"Valdez",
	"Vandenheede",
	"Volker",
	"Wilke",
	"Wojciechowski",
	"Wyndham",
	"Yamashita",
	"York",
	"Zahn",
	"Zimmermann",
	NULL
};

const char * gv_pszGivenNames[] =
{
	"Robby",
	"Agatha",
	"Anatoli",
	"Zsazsa",
	"Arlen",
	"Augusta",
	"Bambi",
	"Bee",
	"Bennie",
	"Bonni",
	"Brennan",
	"Bryon",
	"Cal",
	"Caroline",
	"Charlotte",
	"Cristine",
	"Danny",
	"Dean",
	"Desdemona",
	"Dixie",
	"Doug",
	"Ellie",
	"Zelma",
	"Elsie",
	"Ursula",
	"Ernest",
	"Fanny",
	"Francis",
	"Gailya",
	"Gertrude",
	"Gloria",
	"Greg",
	"Harriot",
	"Hennrietta",
	"Howard",
	"Ian",
	"Sherwood",
	"Xavier",
	"Ira",
	"Jacklyn",
	"Jeff",
	"Philippe",
	"Vivianne",
	"Jeremy",
	"Wendie",
	"Abbie",
	"Johnny",
	"Kerrie",
	"Lacey",
	"Lilly",
	"Lucas",
	"Magdalena",
	"Maryanne",
	"Matt",
	"Dorelle",
	"Myron",
	"Netty",
	"Nicolette",
	"Octavio",
	"Oliver",
	"Paige",
	"Parker",
	"Patti",
	"Merv",
	"Preston",
	"Quinn",
	"Randall",
	"Jean",
	"Rebekah",
	"Ricardo",
	"Rose",
	"Russell",
	"Scarlet",
	"Shannon",
	"Larry",
	"Sophie",
	"Stephen",
	"Susette",
	"Christina",
	"Ted",
	"Enrico",
	"Theresa",
	"Timothy",
	"Tony",
	"Vanna",
	"Kalli",
	"Vern",
	"Alicia",
	"Wallace",
	"Yogi",
	"Aaron",
	"Yuji",
	"Zack",
	NULL
};

/****************************************************************************
Desc:
****************************************************************************/
RCODE getTest( 
	IFlmTest **		ppTest)
{
	RCODE		rc = FERR_OK;

	if( (*ppTest = f_new IFlmTestImpl) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::createDbTest( void)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bPassed = FALSE;

	beginTest( "Create Database Test");

	for (;;)
	{
		if( RC_BAD( rc = FlmDbCreate( DB_NAME_STR, NULL, 
			NULL, NULL, gv_pszSampleDictionary, NULL, &m_hDb)))
		{
			if( rc == FERR_FILE_EXISTS)
			{
				if( RC_BAD( rc = FlmDbRemove( DB_NAME_STR, 
					NULL, NULL, TRUE)))
				{
					MAKE_ERROR_STRING( "calling FlmDbRemove", rc, m_szFailInfo);
					goto Exit;
				}
			}
			else
			{
				MAKE_ERROR_STRING( "calling FlmDbCreate", rc, m_szFailInfo);
				goto Exit;
			}
		}
		else
		{
			break;
		}
	}

	bPassed = TRUE;
	
Exit:

	endTest( bPassed);

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::addRecordTest(
	FLMUINT *		puiDrn)
{
	RCODE				rc = FERR_OK;
	FlmRecord *		pRec = NULL;
	FlmRecord *		pCopyRec = NULL;
	void *			pvField;
	FLMBOOL			bTransActive = FALSE;
	FLMBOOL			bPassed = FALSE;
	FLMUINT			uiLoop;
	FLMUINT			uiLoop2;
	FLMUINT			uiDrn2;
	FLMUINT			uiLastDrn;
	FLMUINT			uiReserveDrn;

	beginTest( "Construct Record Test");

	// Create a record object

	if( (pRec = f_new FlmRecord) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		MAKE_ERROR_STRING( "allocating FlmRecord", rc, m_szFailInfo);
		goto Exit;
	}

	// Populate the record object with fields and values
	// The first field of a record will be inserted at
	// level zero (the first parameter of insertLast()
	// specifies the level number).  Subsequent fields
	// will be inserted at a non-zero level.

	if( RC_BAD( rc = pRec->insertLast( 0, PERSON_TAG,
		FLM_TEXT_TYPE, NULL)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}

	if( RC_BAD( rc = pRec->insertLast( 1, FIRST_NAME_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}

	if( RC_BAD( rc = pRec->setNative( pvField, "Foo")))
	{
		MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
		goto Exit;
	}

	if( RC_BAD( rc = pRec->insertLast( 1, LAST_NAME_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}

	if( RC_BAD( rc = pRec->setNative( pvField, "Bar")))
	{
		MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
		goto Exit;
	}

	if( RC_BAD( rc = pRec->insertLast( 1, AGE_TAG,
		FLM_NUMBER_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}

	if( RC_BAD( rc = pRec->setUINT( pvField, 32)))
	{
		MAKE_ERROR_STRING( "calling setUINT", rc, m_szFailInfo);
		goto Exit;
	}
	endTest( TRUE);

	beginTest( "FlmReserveNextDrn Test");
	
	// Start an update transaction

	if( RC_BAD( rc = FlmDbTransBegin( m_hDb, FLM_UPDATE_TRANS, 15)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransBegin", rc, m_szFailInfo);
		goto Exit;
	}
	bTransActive = TRUE;

	// Reserve the next DRN - so it should not be used.

	if (RC_BAD( rc = FlmReserveNextDrn( m_hDb, 
		FLM_DATA_CONTAINER, &uiReserveDrn)))
	{
		MAKE_ERROR_STRING( "calling FlmReserveNextDrn", rc, m_szFailInfo);
		goto Exit;
		
	}
	if( RC_BAD( rc = FlmDbTransCommit( m_hDb)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransCommit", rc, m_szFailInfo);
		goto Exit;
	}
	bTransActive = FALSE;
	endTest( TRUE);
	
	beginTest( "FlmRecordAdd Test");
	
	// Start another update transaction

	if( RC_BAD( rc = FlmDbTransBegin( m_hDb, FLM_UPDATE_TRANS, 15)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransBegin", rc, m_szFailInfo);
		goto Exit;
	}
	bTransActive = TRUE;
	
	// Add the record to the database.

	*puiDrn = 0;
	if( RC_BAD( rc = FlmRecordAdd( m_hDb, FLM_DATA_CONTAINER, 
		puiDrn, pRec, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordAdd", rc, m_szFailInfo);
		goto Exit;
	}
	
	if (*puiDrn <= uiReserveDrn)
	{
		rc = RC_SET( FERR_FAILURE);
		f_sprintf( m_szFailInfo,
			"DRN %u returned from FlmRecordAdd must be > reserved DRN (%u)",
			(unsigned)(*puiDrn), (unsigned)uiReserveDrn);
		goto Exit;
	}
	uiLastDrn = *puiDrn;
	
	for (uiLoop = 0; gv_pszFamilyNames [uiLoop]; uiLoop++)
	{
		for (uiLoop2 = 0; gv_pszGivenNames [uiLoop2]; uiLoop2++)
		{
			if ((pCopyRec = pRec->copy()) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				MAKE_ERROR_STRING( "calling FlmRecord->copy()", rc, m_szFailInfo);
				goto Exit;
			}
			
			if ((pvField = pCopyRec->find( 
				pCopyRec->root(), FIRST_NAME_TAG)) == NULL)
			{
				rc = RC_SET( FERR_DATA_ERROR);
				MAKE_ERROR_STRING( "corruption calling FlmRecord->find()", 
					rc, m_szFailInfo);
				goto Exit;
			}
			
			if( RC_BAD( rc = pCopyRec->setNative( pvField, 
				gv_pszGivenNames [uiLoop2])))
			{
				MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
				goto Exit;
			}
			
			if ((pvField = pCopyRec->find( pCopyRec->root(), LAST_NAME_TAG)) == NULL)
			{
				rc = RC_SET( FERR_DATA_ERROR);
				MAKE_ERROR_STRING( "corruption calling FlmRecord->copy()", 
					rc, m_szFailInfo);
				goto Exit;
			}
			
			if( RC_BAD( rc = pCopyRec->setNative( pvField, 
				gv_pszFamilyNames [uiLoop])))
			{
				MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
				goto Exit;
			}
			
			uiDrn2 = 0;
			if( RC_BAD( rc = FlmRecordAdd( m_hDb, FLM_DATA_CONTAINER, 
				&uiDrn2, pCopyRec, 0)))
			{
				MAKE_ERROR_STRING( "calling FlmRecordAdd", rc, m_szFailInfo);
				goto Exit;
			}
			
			if (uiDrn2 <= uiLastDrn)
			{
				rc = RC_SET( FERR_FAILURE);
				f_sprintf( m_szFailInfo,
					"DRN %u returned from FlmRecordAdd must be > last added DRN (%u)",
					(unsigned)uiDrn2, (unsigned)uiLastDrn);
				goto Exit;
			}
			
			uiLastDrn = uiDrn2;
			pCopyRec->Release();
			pCopyRec = NULL;
		}
	}

	// Commit the transaction
	//
	// If FlmDbTransCommit returns without an error, the changes made
	// above will be durable even if the system crashes.

	if( RC_BAD( rc = FlmDbTransCommit( m_hDb)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransCommit", rc, m_szFailInfo);
		goto Exit;
	}
	bTransActive = FALSE;
	bPassed = TRUE;
	
Exit:

	if( bTransActive)
	{
		(void)FlmDbTransAbort( m_hDb);
	}

	if( pRec)
	{
		pRec->Release();
	}

	if( pCopyRec)
	{
		pCopyRec->Release();
	}

	endTest( bPassed);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::largeFieldTest( void)
{
	RCODE					rc = FERR_OK;
	FLMBYTE *			pucValue = NULL;
	FlmRecord *			pRec = NULL;
	void *				pvField;
	FLMUINT				uiDrn;
	FLMUINT				uiValueSize;
	FLMUINT				uiLoop;
	FLM_MEM_INFO		memInfo;
	FLMBOOL				bTransActive = FALSE;
	FLMBOOL				bPassed = FALSE;

	beginTest( "Large Field Test");
	
	// Generate a large binary value
	
	uiValueSize = 1024 * 1024;
	if( RC_BAD( rc = f_alloc( uiValueSize, &pucValue)))
	{
		goto Exit;
	}
	
	for( uiLoop = 0; uiLoop < uiValueSize; uiLoop++)
	{
		pucValue[ uiLoop] = (FLMBYTE)f_getRandomUINT32();
	}

	// Create a record object

	if( (pRec = f_new FlmRecord) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		MAKE_ERROR_STRING( "allocating FlmRecord", rc, m_szFailInfo);
		goto Exit;
	}

	// Populate the record object with fields and values

	if( RC_BAD( rc = pRec->insertLast( 0, MISC_TAG,
		FLM_BINARY_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}

	if( RC_BAD( rc = pRec->setBinary( pvField, pucValue, uiValueSize)))
	{
		MAKE_ERROR_STRING( "calling setBinary", rc, m_szFailInfo);
		goto Exit;
	}

	// Start an update transaction

	if( RC_BAD( rc = FlmDbTransBegin( m_hDb, FLM_UPDATE_TRANS, 15)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransBegin", rc, m_szFailInfo);
		goto Exit;
	}
	bTransActive = TRUE;
	
	// Add the record to the database.

	uiDrn = 0;
	if( RC_BAD( rc = FlmRecordAdd( m_hDb, FLM_DATA_CONTAINER, 
		&uiDrn, pRec, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordAdd", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Commit the transaction

	if( RC_BAD( rc = FlmDbTransCommit( m_hDb)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransCommit", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Clear cache
	
	FlmGetMemoryInfo( &memInfo);
	FlmConfig( FLM_CACHE_LIMIT, 0, 0);
	FlmConfig( FLM_CACHE_LIMIT, 
		(void *)(memInfo.RecordCache.uiMaxBytes + memInfo.BlockCache.uiMaxBytes), 0);

	// Make sure the record was removed from cache
		
	if( pRec->isCached())
	{
		rc = RC_SET( FERR_FAILURE);
		MAKE_ERROR_STRING( "Record is still cached", rc, m_szFailInfo);
		goto Exit;
	}
	
	pRec->Release();
	pRec = NULL;
	
	if (RC_BAD( rc = FlmRecordRetrieve( m_hDb, FLM_DATA_CONTAINER, uiDrn,
					FO_EXACT, &pRec, &uiDrn)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordRetrieve", rc, m_szFailInfo);
		goto Exit;
	}
	
	if( f_memcmp( pucValue, pRec->getDataPtr( pRec->root()), uiValueSize) != 0)
	{
		rc = RC_SET( FERR_FAILURE);
		MAKE_ERROR_STRING( "Data value did not match.", rc, m_szFailInfo);
		goto Exit;
	}
	
	bTransActive = FALSE;
	bPassed = TRUE;
	
Exit:

	if( bTransActive)
	{
		(void)FlmDbTransAbort( m_hDb);
	}

	if( pRec)
	{
		pRec->Release();
	}
	
	if( pucValue)
	{
		f_free( &pucValue);
	}

	endTest( bPassed);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::modifyRecordTest(
	FLMUINT	uiDrn)
{
	RCODE				rc = FERR_OK;
	FlmRecord *		pRec = NULL;
	FlmRecord *		pModRec = NULL;
	void *			pvField;
	FLMBOOL			bTransActive = FALSE;
	FLMBOOL			bPassed = FALSE;

	// Retrieve the record from the database by ID

	beginTest( "FlmRecordRetrieve Test");
	if( RC_BAD( rc = FlmRecordRetrieve( m_hDb, FLM_DATA_CONTAINER, 
		uiDrn, FO_EXACT, &pRec, NULL)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordRetrieve", rc, m_szFailInfo);
		goto Exit;
	}
	endTest( TRUE);

	
	beginTest( "FlmRecordModify Test");

	// Copy the record so we can modify it

	if( (pModRec = pRec->copy()) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		MAKE_ERROR_STRING( "calling FlmRecord->copy()", rc, m_szFailInfo);
		goto Exit;
	}

	// Find the first name field and change it.

	pvField = pModRec->find( pModRec->root(), FIRST_NAME_TAG);
	if( RC_BAD( rc = pModRec->setNative( pvField, "FooFoo")))
	{
		MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
		goto Exit;
	}

	// Start an update transaction

	if( RC_BAD( rc = FlmDbTransBegin( m_hDb, FLM_UPDATE_TRANS, 15)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransBegin", rc, m_szFailInfo);
		goto Exit;
	}
	bTransActive = TRUE;

	// Add the record to the database.

	if( RC_BAD( rc = FlmRecordModify( m_hDb, FLM_DATA_CONTAINER, 
		uiDrn, pModRec, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordAdd", rc, m_szFailInfo);
		goto Exit;
	}

	// Commit the transaction
	// If FlmDbTransCommit returns without an error, the changes made
	// above will be durable even if the system crashes.

	if( RC_BAD( rc = FlmDbTransCommit( m_hDb)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransCommit", rc, m_szFailInfo);
		goto Exit;
	}
	bTransActive = FALSE;

	bPassed = TRUE;
	
Exit:

	if( bTransActive)
	{
		(void)FlmDbTransAbort( m_hDb);
	}

	if( pRec)
	{
		pRec->Release();
	}

	if( pModRec)
	{
		pModRec->Release();
	}

	endTest( bPassed);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::deleteRecordTest(
	FLMUINT	uiDrn
	)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bPassed = FALSE;

	// Delete a record from the database

	beginTest( "FlmRecordDelete Test");
	if( RC_BAD( rc = FlmRecordDelete( m_hDb, FLM_DATA_CONTAINER, 
		uiDrn, FLM_AUTO_TRANS | 15)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordDelete", rc, m_szFailInfo);
		goto Exit;
	}
	bPassed = TRUE;

Exit:

	endTest( bPassed);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::queryRecordTest( void)
{
	RCODE			rc = FERR_OK;
	FlmRecord *	pRec = NULL;
	HFCURSOR		hCursor = HFCURSOR_NULL;
	FLMBYTE		ucTmpBuf[ 64];
	FLMBOOL		bPassed = FALSE;
	FLMUINT		uiIndex;
	FLMUINT		uiIndexInfo;
	
	// Now, build a query that retrieves the sample record.
	// First we need to initialize a cursor handle.

	beginTest( "Retrieve Record by query Test");

	if( RC_BAD( rc = FlmCursorInit( m_hDb, FLM_DATA_CONTAINER, &hCursor)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorInit", rc, m_szFailInfo);
		goto Exit;
	}

	// We will search by first name and last name.  This will use the
	// LastFirst_IX defined in the sample dictionary for optimization.

	if( RC_BAD( rc = FlmCursorAddField( hCursor, LAST_NAME_TAG, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddField", rc, m_szFailInfo);
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorAddOp( hCursor, FLM_EQ_OP)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddOp", rc, m_szFailInfo);
		goto Exit;
	}

	f_sprintf( (char *)ucTmpBuf, "Bar");
	if( RC_BAD( rc = FlmCursorAddValue( hCursor, FLM_STRING_VAL, 
		ucTmpBuf, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddValue", rc, m_szFailInfo);
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorAddOp( hCursor, FLM_AND_OP)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddOp failed", rc, m_szFailInfo);
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorAddField( hCursor, FIRST_NAME_TAG, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddField", rc, m_szFailInfo);
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorAddOp( hCursor, FLM_EQ_OP)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddOp", rc, m_szFailInfo);
		goto Exit;
	}

	f_sprintf( (char *)ucTmpBuf, "FooFoo");
	if( RC_BAD( rc = FlmCursorAddValue( hCursor, FLM_STRING_VAL, 
		ucTmpBuf, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddValue", rc, m_szFailInfo);
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorFirst( hCursor, &pRec)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorFirst", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Query should have optimized to use the LastName+FirstName index

	if (RC_BAD( rc = FlmCursorGetConfig( hCursor, FCURSOR_GET_FLM_IX,
								&uiIndex, &uiIndexInfo)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorGetConfig", rc, m_szFailInfo);
		goto Exit;
	}
	if (uiIndex != LAST_NAME_FIRST_NAME_IX || uiIndexInfo != HAVE_ONE_INDEX)
	{
		const char *	pszHave;
		
		switch (uiIndexInfo)
		{
			case HAVE_NO_INDEX:
				pszHave = "HAVE_NO_INDEX";
				break;
			case HAVE_ONE_INDEX:
				pszHave = "HAVE_ONE_INDEX";
				break;
			case HAVE_ONE_INDEX_MULT_PARTS:
				pszHave = "HAVE_ONE_INDEX_MULT_PARTS";
				break;
			case HAVE_MULTIPLE_INDEXES:
				pszHave = "HAVE_MULTIPLE_INDEXES";
				break;
			default:
				pszHave = "HAVE_UNKNOWN";
				break;
		}
		
		rc = RC_SET( FERR_FAILURE);
		f_sprintf( m_szFailInfo,
			"Query should have used LastName+FirstName index (%u), used %u/%s instead",
			(unsigned)LAST_NAME_FIRST_NAME_IX,
			(unsigned)uiIndex, pszHave);
		goto Exit;
	}
	bPassed = TRUE;
	
Exit:

	if (hCursor != HFCURSOR_NULL)
	{
		FlmCursorFree( &hCursor);
	}

	if( pRec)
	{
		pRec->Release();
	}

	endTest( bPassed);

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::keyRetrieveTest(
	FLMUINT	uiIndex,
	FLMBOOL	bLastNameFirstNameIx)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bPassed = FALSE;
	FLMUINT		uiFlags = FO_FIRST;
	FlmRecord *	pSearchKey = NULL;
	FLMUINT		uiSearchDrn = 0;
	FlmRecord *	pFoundKey = NULL;
	FLMUINT		uiFoundDrn = 0;
	char			szLastFirstName [100];
	char			szLastLastName [100];
	char			szCurrFirstName [100];
	char			szCurrLastName [100];
	void *		pvField;
	FLMUINT		uiLen;
	FLMINT		iLastCmp;
	FLMINT		iFirstCmp;

	if (bLastNameFirstNameIx)
	{
		beginTest( "FlmKeyRetrieve Test (Last+FirstIx)");
	}
	else
	{
		beginTest( "FlmKeyRetrieve Test (First+LastIx)");
	}
	szLastFirstName [0] = 0;
	szLastLastName [0] = 0;
	for (;;)
	{
		if (RC_BAD( rc = FlmKeyRetrieve( m_hDb, uiIndex,
								0, pSearchKey, uiSearchDrn, uiFlags,
								&pFoundKey, &uiFoundDrn)))
		{
			if (rc == FERR_EOF_HIT)
			{
				rc = FERR_OK;
				break;
			}
			else
			{
				MAKE_ERROR_STRING( "calling FlmKeyRetrieve", rc, m_szFailInfo);
				goto Exit;
			}
		}
		
		// Make sure this key is greater than the last key.
		
		if ((pvField = pFoundKey->find( pFoundKey->root(), LAST_NAME_TAG)) == NULL)
		{
			rc = RC_SET( FERR_DATA_ERROR);
			MAKE_ERROR_STRING( "corruption calling FlmRecord->find()",
				rc, m_szFailInfo);
			goto Exit;
		}
		uiLen = sizeof( szCurrLastName);
		if (RC_BAD( rc = pFoundKey->getNative( pvField, szCurrLastName, &uiLen)))
		{
			MAKE_ERROR_STRING( "calling FlmRecord->getNative()", 
				rc, m_szFailInfo);
			goto Exit;
		}
		if ((pvField = pFoundKey->find( pFoundKey->root(), FIRST_NAME_TAG)) == NULL)
		{
			rc = RC_SET( FERR_DATA_ERROR);
			MAKE_ERROR_STRING( "corruption calling FlmRecord->find()", 
				rc, m_szFailInfo);
			goto Exit;
		}
		uiLen = sizeof( szCurrFirstName);
		if (RC_BAD( rc = pFoundKey->getNative( pvField, szCurrFirstName, &uiLen)))
		{
			MAKE_ERROR_STRING( "calling FlmRecord->getNative()", rc, m_szFailInfo);
			goto Exit;
		}

		iLastCmp = f_strcmp( szCurrLastName, szLastLastName);
		iFirstCmp = f_strcmp( szCurrFirstName, szLastFirstName);
		
		if (bLastNameFirstNameIx)
		{
			if (iLastCmp < 0)
			{
				rc = RC_SET( FERR_DATA_ERROR);
				f_sprintf( m_szFailInfo, "Invalid last name order in index: "
					" %s before %s", szLastLastName, szCurrLastName);
				goto Exit;
			}
			else if (iLastCmp == 0)
			{
				if (iFirstCmp < 0)
				{
					rc = RC_SET( FERR_DATA_ERROR);
					f_sprintf( m_szFailInfo, "Invalid first name order in index: "
						" %s before %s", szLastFirstName, szCurrFirstName);
					goto Exit;
				}
			}
		}
		else
		{
			if (iFirstCmp < 0)
			{
				rc = RC_SET( FERR_DATA_ERROR);
				f_sprintf( m_szFailInfo, "Invalid first name order in index: "
					" %s before %s", szLastFirstName, szCurrFirstName);
				goto Exit;
			}
			else if (iFirstCmp == 0)
			{
				if (iLastCmp < 0)
				{
					rc = RC_SET( FERR_DATA_ERROR);
					f_sprintf( m_szFailInfo, "Invalid last name order in index: "
						" %s before %s", szLastLastName, szCurrLastName);
					goto Exit;
				}
			}
		}
		
		// Setup to get the next key.
		
		uiFlags = FO_EXCL;
		uiSearchDrn = uiFoundDrn;
		if (pSearchKey)
		{
			pSearchKey->Release();
		}
		pSearchKey = pFoundKey;
		pFoundKey = NULL;
		uiFoundDrn = 0;
		f_strcpy( szLastLastName, szCurrLastName);
		f_strcpy( szLastFirstName, szCurrFirstName);
	}
	bPassed = TRUE;

Exit:

	if (pSearchKey)
	{
		pSearchKey->Release();
	}
	if (pFoundKey)
	{
		pFoundKey->Release();
	}

	endTest( bPassed);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::addIndexTest(
	FLMUINT *	puiIndex
	)
{
	RCODE				rc = FERR_OK;
	FlmRecord *		pRec = NULL;
	void *			pvField;
	FLMBOOL			bTransActive = FALSE;
	FLMBOOL			bPassed = FALSE;
	char				szFieldNum [20];

	beginTest( "Add FirstName+LastName Index Test");

	// Create a record object

	if( (pRec = f_new FlmRecord) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		MAKE_ERROR_STRING( "allocating FlmRecord", rc, m_szFailInfo);
		goto Exit;
	}

	// 0 index FirstLast_IX
	
	if( RC_BAD( rc = pRec->insertLast( 0, FLM_INDEX_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	if( RC_BAD( rc = pRec->setNative( pvField, "FirstLast_IX")))
	{
		MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
		goto Exit;
	}

	// 1 language US

	if( RC_BAD( rc = pRec->insertLast( 1, FLM_LANGUAGE_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	if( RC_BAD( rc = pRec->setNative( pvField, "US")))
	{
		MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
		goto Exit;
	}

	// 1 key

	if( RC_BAD( rc = pRec->insertLast( 1, FLM_KEY_TAG,
		FLM_CONTEXT_TYPE, NULL)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}

	// 2 field <FIRST_NAME_TAG>

	if( RC_BAD( rc = pRec->insertLast( 2, FLM_FIELD_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	f_sprintf( szFieldNum, "%u", FIRST_NAME_TAG);
	if( RC_BAD( rc = pRec->setNative( pvField, szFieldNum)))
	{
		MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
		goto Exit;
	}
	
	// 3 required

	if( RC_BAD( rc = pRec->insertLast( 3, FLM_REQUIRED_TAG,
		FLM_TEXT_TYPE, NULL)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}

	// 2 field <LAST_NAME_TAG>

	if( RC_BAD( rc = pRec->insertLast( 2, FLM_FIELD_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	f_sprintf( szFieldNum, "%u", LAST_NAME_TAG);
	if( RC_BAD( rc = pRec->setNative( pvField, szFieldNum)))
	{
		MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
		goto Exit;
	}
	
	// 3 required
	
	if( RC_BAD( rc = pRec->insertLast( 3, FLM_REQUIRED_TAG,
		FLM_TEXT_TYPE, NULL)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Start an update transaction

	if( RC_BAD( rc = FlmDbTransBegin( m_hDb, FLM_UPDATE_TRANS, 15)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransBegin", rc, m_szFailInfo);
		goto Exit;
	}
	bTransActive = TRUE;

	// Add the record to the database.

	*puiIndex = 0;
	if( RC_BAD( rc = FlmRecordAdd( m_hDb, FLM_DICT_CONTAINER, 
		puiIndex, pRec, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordAdd", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Commit the transaction
	// If FlmDbTransCommit returns without an error, the changes made
	// above will be durable even if the system crashes.

	if( RC_BAD( rc = FlmDbTransCommit( m_hDb)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransCommit", rc, m_szFailInfo);
		goto Exit;
	}
	bTransActive = FALSE;

	bPassed = TRUE;
	
Exit:

	if( bTransActive)
	{
		(void)FlmDbTransAbort( m_hDb);
	}

	if( pRec)
	{
		pRec->Release();
	}

	endTest( bPassed);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::addSubstringIndexTest(
	FLMUINT *	puiIndex
	)
{
	RCODE				rc = FERR_OK;
	FlmRecord *		pRec = NULL;
	void *			pvField;
	FLMBOOL			bTransActive = FALSE;
	FLMBOOL			bPassed = FALSE;
	char				szFieldNum [20];

	beginTest( "Add Substring (LastName+FirstName) Index Test");

	// Create a record object

	if( (pRec = f_new FlmRecord) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		MAKE_ERROR_STRING( "allocating FlmRecord", rc, m_szFailInfo);
		goto Exit;
	}

	// 0 index FirstLast-SubString_IX
	
	if( RC_BAD( rc = pRec->insertLast( 0, FLM_INDEX_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	if( RC_BAD( rc = pRec->setNative( pvField, "FirstLast-Substring_IX")))
	{
		MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
		goto Exit;
	}

	// 1 language US

	if( RC_BAD( rc = pRec->insertLast( 1, FLM_LANGUAGE_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	if( RC_BAD( rc = pRec->setNative( pvField, "US")))
	{
		MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
		goto Exit;
	}

	// 1 key

	if( RC_BAD( rc = pRec->insertLast( 1, FLM_KEY_TAG,
		FLM_CONTEXT_TYPE, NULL)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}

	// 2 field <FIRST_NAME_TAG>

	if( RC_BAD( rc = pRec->insertLast( 2, FLM_FIELD_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	f_sprintf( szFieldNum, "%u", FIRST_NAME_TAG);
	if( RC_BAD( rc = pRec->setNative( pvField, szFieldNum)))
	{
		MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
		goto Exit;
	}
	
	// 3 required

	if( RC_BAD( rc = pRec->insertLast( 3, FLM_REQUIRED_TAG,
		FLM_TEXT_TYPE, NULL)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}

	// 3 use substring

	if( RC_BAD( rc = pRec->insertLast( 3, FLM_USE_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	if( RC_BAD( rc = pRec->setNative( pvField, "substring")))
	{
		MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
		goto Exit;
	}

	// 2 field <LAST_NAME_TAG>

	if( RC_BAD( rc = pRec->insertLast( 2, FLM_FIELD_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	f_sprintf( szFieldNum, "%u", LAST_NAME_TAG);
	if( RC_BAD( rc = pRec->setNative( pvField, szFieldNum)))
	{
		MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
		goto Exit;
	}
	
	// 3 required
	
	if( RC_BAD( rc = pRec->insertLast( 3, FLM_REQUIRED_TAG,
		FLM_TEXT_TYPE, NULL)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	
	// 3 use substring

	if( RC_BAD( rc = pRec->insertLast( 3, FLM_USE_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	if( RC_BAD( rc = pRec->setNative( pvField, "substring")))
	{
		MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
		goto Exit;
	}

	// Start an update transaction

	if( RC_BAD( rc = FlmDbTransBegin( m_hDb, FLM_UPDATE_TRANS, 15)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransBegin", rc, m_szFailInfo);
		goto Exit;
	}
	bTransActive = TRUE;

	// Add the record to the database.

	*puiIndex = 0;
	if( RC_BAD( rc = FlmRecordAdd( m_hDb, FLM_DICT_CONTAINER, 
		puiIndex, pRec, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordAdd", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Commit the transaction
	// If FlmDbTransCommit returns without an error, the changes made
	// above will be durable even if the system crashes.

	if( RC_BAD( rc = FlmDbTransCommit( m_hDb)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransCommit", rc, m_szFailInfo);
		goto Exit;
	}
	bTransActive = FALSE;

	bPassed = TRUE;
	
Exit:

	if( bTransActive)
	{
		(void)FlmDbTransAbort( m_hDb);
	}

	if( pRec)
	{
		pRec->Release();
	}

	endTest( bPassed);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::addPresenceIndexTest(
	FLMUINT *	puiIndex
	)
{
	RCODE				rc = FERR_OK;
	FlmRecord *		pRec = NULL;
	void *			pvField;
	FLMBOOL			bTransActive = FALSE;
	FLMBOOL			bPassed = FALSE;
	char				szFieldNum [20];

	beginTest( "Add Presence (LastName+FirstName) Index Test");

	// Create a record object

	if( (pRec = f_new FlmRecord) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		MAKE_ERROR_STRING( "allocating FlmRecord", rc, m_szFailInfo);
		goto Exit;
	}

	// 0 index FirstLast-Presence_IX
	
	if( RC_BAD( rc = pRec->insertLast( 0, FLM_INDEX_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	if( RC_BAD( rc = pRec->setNative( pvField, "FirstLast-Presence_IX")))
	{
		MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
		goto Exit;
	}

	// 1 language US

	if( RC_BAD( rc = pRec->insertLast( 1, FLM_LANGUAGE_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	if( RC_BAD( rc = pRec->setNative( pvField, "US")))
	{
		MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
		goto Exit;
	}

	// 1 key

	if( RC_BAD( rc = pRec->insertLast( 1, FLM_KEY_TAG,
		FLM_CONTEXT_TYPE, NULL)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}

	// 2 field <FIRST_NAME_TAG>

	if( RC_BAD( rc = pRec->insertLast( 2, FLM_FIELD_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	f_sprintf( szFieldNum, "%u", FIRST_NAME_TAG);
	if( RC_BAD( rc = pRec->setNative( pvField, szFieldNum)))
	{
		MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
		goto Exit;
	}
	
	// 3 required

	if( RC_BAD( rc = pRec->insertLast( 3, FLM_REQUIRED_TAG,
		FLM_TEXT_TYPE, NULL)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}

	// 3 use field

	if( RC_BAD( rc = pRec->insertLast( 3, FLM_USE_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	if( RC_BAD( rc = pRec->setNative( pvField, "field")))
	{
		MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
		goto Exit;
	}

	// 2 field <LAST_NAME_TAG>

	if( RC_BAD( rc = pRec->insertLast( 2, FLM_FIELD_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	f_sprintf( szFieldNum, "%u", LAST_NAME_TAG);
	if( RC_BAD( rc = pRec->setNative( pvField, szFieldNum)))
	{
		MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
		goto Exit;
	}
	
	// 3 required
	
	if( RC_BAD( rc = pRec->insertLast( 3, FLM_REQUIRED_TAG,
		FLM_TEXT_TYPE, NULL)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	
	// 3 use field

	if( RC_BAD( rc = pRec->insertLast( 3, FLM_USE_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	if( RC_BAD( rc = pRec->setNative( pvField, "field")))
	{
		MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
		goto Exit;
	}

	// Start an update transaction

	if( RC_BAD( rc = FlmDbTransBegin( m_hDb, FLM_UPDATE_TRANS, 15)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransBegin", rc, m_szFailInfo);
		goto Exit;
	}
	bTransActive = TRUE;

	// Add the record to the database.

	*puiIndex = 0;
	if( RC_BAD( rc = FlmRecordAdd( m_hDb, FLM_DICT_CONTAINER, 
		puiIndex, pRec, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordAdd", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Commit the transaction
	// If FlmDbTransCommit returns without an error, the changes made
	// above will be durable even if the system crashes.

	if( RC_BAD( rc = FlmDbTransCommit( m_hDb)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransCommit", rc, m_szFailInfo);
		goto Exit;
	}
	bTransActive = FALSE;

	bPassed = TRUE;
	
Exit:

	if( bTransActive)
	{
		(void)FlmDbTransAbort( m_hDb);
	}

	if( pRec)
	{
		pRec->Release();
	}

	endTest( bPassed);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::deleteIndexTest(
	FLMUINT	uiIndex
	)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bPassed = FALSE;

	beginTest( "Delete Index Test");

	// Delete the record from the dictionary.

	if( RC_BAD( rc = FlmRecordDelete( m_hDb, FLM_DICT_CONTAINER, uiIndex,
								 FLM_AUTO_TRANS | 15))) 
	{
		MAKE_ERROR_STRING( "calling FlmRecordDelete", rc, m_szFailInfo);
		goto Exit;
	}
	
	bPassed = TRUE;
	
Exit:

	endTest( bPassed);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::deleteFieldTest(
	FLMUINT			uiFieldNum)
{
	RCODE				rc = FERR_OK;
	FlmRecord *		pDictRec = NULL;
	FlmRecord *		pNewRec = NULL;
	void *			pvField;
	FLMUINT			uiDrn;
	FLMBOOL			bPassed = FALSE;
	FLMBOOL			bTransActive = FALSE;

	beginTest( "Delete Field Definition Test");

	// Delete the record from the dictionary.  This attempt should fail
	// because it is not properly marked.

	if( RC_BAD( rc = FlmRecordDelete( m_hDb, FLM_DICT_CONTAINER, uiFieldNum,
								 FLM_AUTO_TRANS | 15))) 
	{
		if (rc != FERR_CANNOT_DEL_ITEM)
		{
			MAKE_ERROR_STRING( "calling FlmRecordDelete", rc, m_szFailInfo);
			goto Exit;
		}
		else
		{
			rc = FERR_OK;
		}
	}
	else
	{
		rc = RC_SET( FERR_FAILURE);
		f_sprintf( m_szFailInfo, "Should not be able to delete field %u!",
				(unsigned)uiFieldNum);
		goto Exit;
	}
	
	// Retrieve the field definition record.
	
	if (RC_BAD( rc = FlmRecordRetrieve( m_hDb, FLM_DICT_CONTAINER,
								uiFieldNum, FO_EXACT, &pDictRec, &uiDrn)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordRetrieve", rc, m_szFailInfo);
		goto Exit;
	}
	
	// If it is not a field definition, we have the wrong definition record.
	
	if (pDictRec->getFieldID( pDictRec->root()) != FLM_FIELD_TAG)
	{
		rc = RC_SET( FERR_FAILURE);
		f_sprintf( m_szFailInfo, 
				"Dictionary record %u, is not a field definition!",
				(unsigned)uiFieldNum);
		goto Exit;
	}
	
	// Make a copy of the dictionary record
	
	if ((pNewRec = pDictRec->copy()) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		MAKE_ERROR_STRING( "calling FlmRecord->copy()", rc, m_szFailInfo);
		goto Exit;
	}
	
	// See if there is a state field.  If not add it.
	
	if ((pvField = pNewRec->find( pNewRec->root(), FLM_STATE_TAG)) == NULL)
	{
		if (RC_BAD( rc = pNewRec->insert( pNewRec->root(), INSERT_LAST_CHILD,
									FLM_STATE_TAG, FLM_TEXT_TYPE, &pvField)))
		{
			MAKE_ERROR_STRING( "calling FlmRecord->insert()", rc, m_szFailInfo);
			goto Exit;
		}
	}
	
	// Attempt to set the state field on the record to "unused", this should
	// fail.
	
	if (RC_BAD( rc = pNewRec->setNative( pvField, "unused")))
	{
		MAKE_ERROR_STRING( "calling FlmRecord->setNative()", rc, m_szFailInfo);
		goto Exit;
	}
	
	if( RC_BAD( rc = FlmRecordModify( m_hDb, FLM_DICT_CONTAINER, 
		uiFieldNum, pNewRec, FLM_AUTO_TRANS | 15)))
	{
		if (rc != FERR_CANNOT_MOD_FIELD_STATE)
		{
			MAKE_ERROR_STRING( "calling FlmRecordModify", rc, m_szFailInfo);
			goto Exit;
		}
		else
		{
			rc = FERR_OK;
		}
	}
	else
	{
		rc = RC_SET( FERR_FAILURE);
		f_sprintf( m_szFailInfo, 
				"Should not be able to set field %'s state to unused!",
				(unsigned)uiFieldNum);
		goto Exit;
	}
	
	// Set the state field on the record to "check", then run
	// FlmDbSweep.  The sweep should not set the field state
	// to unused.
	
	if (RC_BAD( rc = pNewRec->setNative( pvField, "check")))
	{
		MAKE_ERROR_STRING( "calling FlmRecord->setNative()", rc, m_szFailInfo);
		goto Exit;
	}
	
	if( RC_BAD( rc = FlmRecordModify( m_hDb, FLM_DICT_CONTAINER, 
		uiFieldNum, pNewRec, FLM_AUTO_TRANS | 15)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordModify", rc, m_szFailInfo);
		goto Exit;
	}
	
	if( RC_BAD( rc = FlmDbTransBegin( m_hDb, FLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
	{
		goto Exit;
	}
	
	bTransActive = TRUE;
	
	if( RC_BAD( rc = FlmDbSweep( m_hDb, SWEEP_CHECKING_FLDS, 
		EACH_CHANGE, NULL, NULL)))
	{
		MAKE_ERROR_STRING( "calling FlmDbSweep", rc, m_szFailInfo);
		goto Exit;
	}
	
	bTransActive = FALSE;
	if( RC_BAD( rc = FlmDbTransCommit( m_hDb)))
	{
		goto Exit;
	}
	
	if (pNewRec)
	{
		pNewRec->Release();
		pNewRec = NULL;
	}
	
	if (pDictRec)
	{
		pDictRec->Release();
		pDictRec = NULL;
	}
	
	// Retrieve the record again and make sure the state flag is not set
	// to unused.

	if (RC_BAD( rc = FlmRecordRetrieve( m_hDb, FLM_DICT_CONTAINER,
								uiFieldNum, FO_EXACT, &pDictRec, &uiDrn)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordRetrieve", rc, m_szFailInfo);
		goto Exit;
	}
	
	// If it is not a field definition, we have the wrong definition record.
	
	if (pDictRec->getFieldID( pDictRec->root()) != FLM_FIELD_TAG)
	{
		rc = RC_SET( FERR_FAILURE);
		f_sprintf( m_szFailInfo, 
				"Dictionary record %u, is not a field definition!",
				(unsigned)uiFieldNum);
		goto Exit;
	}
	
	// Make a copy of the dictionary record
	
	if ((pNewRec = pDictRec->copy()) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		MAKE_ERROR_STRING( "calling FlmRecord->copy()", rc, m_szFailInfo);
		goto Exit;
	}
	
	// See if there is a state field.  If not add it.
	
	if ((pvField = pNewRec->find( pNewRec->root(), FLM_STATE_TAG)) == NULL)
	{
		if (RC_BAD( rc = pNewRec->insert( pNewRec->root(), INSERT_LAST_CHILD,
									FLM_STATE_TAG, FLM_TEXT_TYPE, &pvField)))
		{
			MAKE_ERROR_STRING( "calling FlmRecord->insert()", rc, m_szFailInfo);
			goto Exit;
		}
	}
	else
	{
		char		szState [20];
		FLMUINT	uiLen = sizeof( szState);
		
		// State should be active if it is present.
		
		if (RC_BAD( rc = pNewRec->getNative( pvField, szState, &uiLen)))
		{
			MAKE_ERROR_STRING( "calling FlmRecord->getNative()", rc, m_szFailInfo);
			goto Exit;
		}
		
		if (f_strnicmp( szState, "acti", 4) != 0)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, 
				"Dictionary record %u's state should be active!",
				(unsigned)uiFieldNum);
			goto Exit;
		}
	}
	
	// Attempt to set the state field on the record to "purge", this should
	// succeed, and FlmDbSweep should get rid of the definition.
	
	if (RC_BAD( rc = pNewRec->setNative( pvField, "purge")))
	{
		MAKE_ERROR_STRING( "calling FlmRecord->setNative()", rc, m_szFailInfo);
		goto Exit;
	}
	
	if( RC_BAD( rc = FlmRecordModify( m_hDb, FLM_DICT_CONTAINER, 
		uiFieldNum, pNewRec, FLM_AUTO_TRANS | 15)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordModify", rc, m_szFailInfo);
		goto Exit;
	}
	
	if( RC_BAD( rc = FlmDbTransBegin( m_hDb, FLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
	{
		goto Exit;
	}
	
	bTransActive = TRUE;
	
	if( RC_BAD( rc = FlmDbSweep( m_hDb, SWEEP_PURGED_FLDS, 
		EACH_CHANGE, NULL, NULL)))
	{
		MAKE_ERROR_STRING( "calling FlmDbSweep", rc, m_szFailInfo);
		goto Exit;
	}
	
	bTransActive = FALSE;
	if( RC_BAD( rc = FlmDbTransCommit( m_hDb)))
	{
		goto Exit;
	}
	
	// Make sure the dictionary definition is gone now.

	if (pDictRec)
	{
		pDictRec->Release();
		pDictRec = NULL;
	}
	
	if (RC_BAD( rc = FlmRecordRetrieve( m_hDb, FLM_DICT_CONTAINER,
								uiFieldNum, FO_EXACT, &pDictRec, &uiDrn)))
	{
		if (rc != FERR_NOT_FOUND)
		{
			MAKE_ERROR_STRING( "calling FlmRecordRetrieve", rc, m_szFailInfo);
			goto Exit;
		}
		else
		{
			rc = FERR_OK;
		}
	}
	else
	{
		rc = RC_SET( FERR_FAILURE);
		f_sprintf( m_szFailInfo, "Dictionary record %u should have been purged by FlmDbSweep!",
				(unsigned)uiFieldNum);
		goto Exit;
	}
	
	bPassed = TRUE;
	
Exit:

	if( pDictRec)
	{
		pDictRec->Release();
	}
	
	if( pNewRec)
	{
		pNewRec->Release();
	}
	
	if( bTransActive)
	{
		FlmDbTransAbort( m_hDb);
	}

	endTest( bPassed);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::suspendIndexTest(
	FLMUINT	uiIndex
	)
{
	RCODE				rc = FERR_OK;
	FLMBOOL			bPassed = FALSE;
	FINDEX_STATUS	indexStatus;

	beginTest( "Suspend Index Test");

	// Delete the record from the dictionary.

	if( RC_BAD( rc = FlmIndexSuspend( m_hDb, uiIndex)))
	{
		MAKE_ERROR_STRING( "calling FlmIndexSuspend", rc, m_szFailInfo);
		goto Exit;
	}
	
	// See if the index is actually suspended.
	
	if( RC_BAD( rc = FlmIndexStatus( m_hDb, uiIndex, &indexStatus)))
	{
		MAKE_ERROR_STRING( "calling FlmIndexStatus", rc, m_szFailInfo);
		goto Exit;
	}
	
	if (!indexStatus.bSuspended)
	{
		rc = RC_SET( FERR_FAILURE);
		f_sprintf( m_szFailInfo, "FlmIndexSuspend failed to suspend index %u",
			(unsigned)uiIndex);
		goto Exit;
	}
	
	bPassed = TRUE;
	
Exit:

	endTest( bPassed);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::resumeIndexTest(
	FLMUINT	uiIndex)
{
	RCODE				rc = FERR_OK;
	FLMBOOL			bPassed = FALSE;
	FINDEX_STATUS	indexStatus;

	beginTest( "Resume Index Test");

	// Delete the record from the dictionary.

	if (RC_BAD( rc = FlmIndexResume( m_hDb, uiIndex)))
	{
		MAKE_ERROR_STRING( "calling FlmIndexResume", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Wait for the index to come on-line
	
	for (;;)
	{
	
		// See if the index is actually resumed.
		
		if( RC_BAD( rc = FlmIndexStatus( m_hDb, uiIndex, &indexStatus)))
		{
			MAKE_ERROR_STRING( "calling FlmIndexStatus", rc, m_szFailInfo);
			goto Exit;
		}
		
		if (indexStatus.bSuspended)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, "FlmIndexResume failed to resume index %u",
				(unsigned)uiIndex);
			goto Exit;
		}
		
		if (indexStatus.uiLastRecordIdIndexed == RECID_UNDEFINED)
		{
			break;
		}
		
		f_sleep( 50);
	}
	
	bPassed = TRUE;
	
Exit:

	endTest( bPassed);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::sortedFieldsTest(
	FLMUINT *	puiDrn)
{
	RCODE				rc = FERR_OK;
	FLMBOOL			bPassed = FALSE;
	FlmRecord *		pRec = NULL;
	FlmRecord *		pDataRec = NULL;
	FlmRecord *		pCopyRec = NULL;
	void *			pvField;
	void *			pvDataField;
	FLMBOOL			bTransActive = FALSE;
	char				szFieldName [100];
	FLMUINT			uiFieldId;
	FLMUINT			uiFieldPos = 0;
	FLMUINT			uiTmp;
	FLMUINT			uiCount;
	FLMUINT			uiLoop1;
	FLMUINT			uiLoop2;

	beginTest( "Sorted Fields Test");

	// Modify to keep field id table for level one fields.

	if (RC_BAD( rc = FlmDbConfig( m_hDb, FDB_ENABLE_FIELD_ID_TABLE,
							(void *)FLM_DATA_CONTAINER, (void *)TRUE)))
	{
		MAKE_ERROR_STRING( "calling FlmDbConfig", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Create a dictionary record object

	if( (pDataRec = f_new FlmRecord) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		MAKE_ERROR_STRING( "allocating FlmRecord", rc, m_szFailInfo);
		goto Exit;
	}
	if( RC_BAD( rc = pDataRec->insertLast( 0, PERSON_TAG,
		FLM_TEXT_TYPE, &pvDataField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Start an update transaction

	if( RC_BAD( rc = FlmDbTransBegin( m_hDb, FLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransBegin", rc, m_szFailInfo);
		goto Exit;
	}
	bTransActive = TRUE;

	// Create 300 IDs in the dictionary, 1001 to 1601, every other one
	
	for (uiFieldId = 1001; uiFieldId <= 1601; uiFieldId += 2)
	{
		if (pRec)
		{
			pRec->Release();
			pRec = NULL;
		}

		// Create a dictionary record object
	
		if( (pRec = f_new FlmRecord) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			MAKE_ERROR_STRING( "allocating FlmRecord", rc, m_szFailInfo);
			goto Exit;
		}
	
		// Populate the record object with fields and values
		// The first field of a record will be inserted at
		// level zero (the first parameter of insertLast()
		// specifies the level number).  Subsequent fields
		// will be inserted at a non-zero level.
	
		if( RC_BAD( rc = pRec->insertLast( 0, FLM_FIELD_TAG,
			FLM_TEXT_TYPE, &pvField)))
		{
			MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
			goto Exit;
		}
		f_sprintf( szFieldName, "Field_%u", (unsigned)uiFieldId);
		if( RC_BAD( rc = pRec->setNative( pvField, szFieldName)))
		{
			MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
			goto Exit;
		}
	
		if( RC_BAD( rc = pRec->insertLast( 1, FLM_TYPE_TAG,
			FLM_TEXT_TYPE, &pvField)))
		{
			MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
			goto Exit;
		}
		if( RC_BAD( rc = pRec->setNative( pvField, "number")))
		{
			MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
			goto Exit;
		}
		
		if( RC_BAD( rc = FlmRecordAdd( m_hDb, FLM_DICT_CONTAINER, 
			&uiFieldId, pRec, 0)))
		{
			MAKE_ERROR_STRING( "calling FlmRecordAdd", rc, m_szFailInfo);
			goto Exit;
		}
	}
	
	// Add the fields to the record in reverse order.

	for (uiFieldId = 1601; uiFieldId >= 1001; uiFieldId -= 2)
	{
		// Add three instances of each field.
		
		for (uiCount = 1; uiCount <= 3; uiCount++)
		{
			if( RC_BAD( rc = pDataRec->insertLast( 1, uiFieldId,
				FLM_NUMBER_TYPE, &pvDataField)))
			{
				MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
				goto Exit;
			}
			if (RC_BAD( rc = pDataRec->setUINT( pvDataField, uiFieldId)))
			{
				MAKE_ERROR_STRING( "calling setUINT", rc, m_szFailInfo);
				goto Exit;
			}
			
			// Under each field add 25 sub-fields, and under those add 5 sub-fields
			
			for (uiLoop1 = 0; uiLoop1 < 25; uiLoop1++)
			{
				if( RC_BAD( rc = pDataRec->insertLast( 2, uiFieldId,
					FLM_NUMBER_TYPE, &pvDataField)))
				{
					MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
					goto Exit;
				}
				if (RC_BAD( rc = pDataRec->setUINT( pvDataField, uiFieldId)))
				{
					MAKE_ERROR_STRING( "calling setUINT", rc, m_szFailInfo);
					goto Exit;
				}
				for (uiLoop2 = 0; uiLoop2 < 5; uiLoop2++)
				{
					if( RC_BAD( rc = pDataRec->insertLast( 3, uiFieldId,
						FLM_NUMBER_TYPE, &pvDataField)))
					{
						MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
						goto Exit;
					}
					if (RC_BAD( rc = pDataRec->setUINT( pvDataField, uiFieldId)))
					{
						MAKE_ERROR_STRING( "calling setUINT", rc, m_szFailInfo);
						goto Exit;
					}
				}
				
				f_yieldCPU();
			}
		}
		
		f_yieldCPU();
	}
	
	// Add the data record to the data container.

	*puiDrn = 0;	
	if( RC_BAD( rc = FlmRecordAdd( m_hDb, FLM_DATA_CONTAINER, 
		puiDrn, pDataRec, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordAdd", rc, m_szFailInfo);
		goto Exit;
	}
	
	if( RC_BAD( rc = FlmDbTransCommit( m_hDb)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransCommit", rc, m_szFailInfo);
		goto Exit;
	}
	bTransActive = FALSE;
	
	// For each of the 300 field ids, get the field from the record, inclusive
	
	for (uiFieldId = 1000; uiFieldId <= 1600; uiFieldId += 2)
	{
		pvDataField = pDataRec->findLevelOneField( uiFieldId, TRUE, &uiFieldPos);
		if (!pvDataField)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, 
				"Could not find next level one after field #%u",
				(unsigned)uiFieldId);
			goto Exit;
		}
		
		// Verify that we got the expected field ID.
		
		uiTmp = pDataRec->getFieldID( pvDataField);
		if (uiTmp != uiFieldId + 1)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, 
				"Incorrect field ID (%u) returned from level one field #%u (incl)",
				(unsigned)uiTmp, (unsigned)(uiFieldId + 1));
			goto Exit;
		}
		
		// Verify that we got the expected field value.
		
		if (RC_BAD( rc = pDataRec->getUINT( pvDataField, &uiTmp)))
		{
			MAKE_ERROR_STRING( "calling getUINT", rc, m_szFailInfo);
			goto Exit;
		}
		if (uiTmp != uiFieldId + 1)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, "Incorrect value (%u) returned from level one field #%u (incl)",
				(unsigned)uiTmp, (unsigned)(uiFieldId + 1));
			goto Exit;
		}
		
		// Get the next level one fields with the same ID.  Should be two more.
		// Set uiCount to one because we have already retrieved one of them.

		uiCount = 1;		
		for (;;)
		{
			pvDataField = pDataRec->nextLevelOneField( &uiFieldPos, TRUE);
			if (!pvDataField)
			{
				if (uiCount != 3)
				{
					rc = RC_SET( FERR_FAILURE);
					f_sprintf( m_szFailInfo, "Could not get next level one field with same ID as #%u",
										(unsigned)(uiFieldId + 1));
					goto Exit;
				}
				break;
			}
			else
			{
				uiCount++;
				if (uiCount > 3)
				{
					rc = RC_SET( FERR_FAILURE);
					f_sprintf( m_szFailInfo, 
						"Too many instances of level one fields with ID #%u",
						(unsigned)(uiFieldId + 1));
					goto Exit;
				}
				
				// Verify that we got the expected field ID.
				
				uiTmp = pDataRec->getFieldID( pvDataField);
				if (uiTmp != uiFieldId + 1)
				{
					rc = RC_SET( FERR_FAILURE);
					f_sprintf( m_szFailInfo, 
						"Incorrect field ID (%u) returned from instance #%u of level one field #%u",
						(unsigned)uiTmp, (unsigned)uiCount, (unsigned)(uiFieldId + 1));
					goto Exit;
				}
				
				// Verify that we got the expected field value.
				
				if (RC_BAD( rc = pDataRec->getUINT( pvDataField, &uiTmp)))
				{
					MAKE_ERROR_STRING( "calling getUINT", rc, m_szFailInfo);
					goto Exit;
				}
				if (uiTmp != uiFieldId + 1)
				{
					rc = RC_SET( FERR_FAILURE);
					f_sprintf( m_szFailInfo, 
						"Incorrect value (%u) returned from instance #%u of level one field #%u",
						(unsigned)uiTmp, (unsigned)uiCount, (unsigned)(uiFieldId + 1));
					goto Exit;
				}
			}
			
			f_yieldCPU();
		}
		
		f_yieldCPU();
	}
	
	// For each of the 300 field ids, get the field from the record.
	// Also, delete each one as we go, and make sure they are deleted.
	// We need to copy the record so that this can be done.
	
	if ((pCopyRec = pDataRec->copy()) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		MAKE_ERROR_STRING( "calling copy", rc, m_szFailInfo);
		goto Exit;
	}
	
	for (uiFieldId = 1001; uiFieldId <= 1601; uiFieldId += 2)
	{
		
		// Should be three instances to delete.
		
		for (uiCount = 1; uiCount <= 3; uiCount++)
		{
			pvDataField = pCopyRec->findLevelOneField( uiFieldId, FALSE, &uiFieldPos);
			if (!pvDataField)
			{
				rc = RC_SET( FERR_FAILURE);
				f_sprintf( m_szFailInfo, 
					"Could not find instance #%u of level one field #%u",
					(unsigned)uiCount, (unsigned)uiFieldId);
				goto Exit;
			}
			
			// Verify that we got the expected field ID.
			
			uiTmp = pCopyRec->getFieldID( pvDataField);
			if (uiTmp != uiFieldId)
			{
				rc = RC_SET( FERR_FAILURE);
				f_sprintf( m_szFailInfo, 
					"Incorrect field ID (%u) returned from instance #%u of level one field #%u",
					(unsigned)uiTmp, (unsigned)uiCount, (unsigned)uiFieldId);
				goto Exit;
			}
			
			// Verify that we got the expected field value.
			
			if (RC_BAD( rc = pCopyRec->getUINT( pvDataField, &uiTmp)))
			{
				MAKE_ERROR_STRING( "calling getUINT", rc, m_szFailInfo);
				goto Exit;
			}
			
			if (uiTmp != uiFieldId)
			{
				rc = RC_SET( FERR_FAILURE);
				f_sprintf( m_szFailInfo, 
					"Incorrect value (%u) returned from instance #%u of level one field #%u",
					(unsigned)uiTmp, (unsigned)uiCount, (unsigned)uiFieldId);
				goto Exit;
			}
			
			// Remove the field and make sure that the find fails.
			
			if (RC_BAD( rc = pCopyRec->remove( pvDataField)))
			{
				MAKE_ERROR_STRING( "calling remove", rc, m_szFailInfo);
				goto Exit;
			}
		}
		
		// All instances should be gone now.
		
		pvDataField = pCopyRec->findLevelOneField( uiFieldId, FALSE, &uiFieldPos);
		if (pvDataField)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, "Should NOT have found level one field #%u",
				(unsigned)uiFieldId);
			goto Exit;
		}
		
		f_yieldCPU();
	}
	
	bPassed = TRUE;
	
Exit:

	if (pRec)
	{
		pRec->Release();
	}
	
	if (pDataRec)
	{
		pDataRec->Release();
	}
	
	if (pCopyRec)
	{
		pCopyRec->Release();
	}
	
	if (bTransActive)
	{
		(void)FlmDbTransAbort( m_hDb);
	}

	endTest( bPassed);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::sortedFieldsQueryTest(
	FLMUINT		uiDrn,
	FLMBOOL		bDoRootedFieldPaths)
{
	RCODE				rc = FERR_OK;
	FLMBOOL			bPassed = FALSE;
	HFCURSOR			hCursor = HFCURSOR_NULL;
	FLMUINT			uiFieldPath [10];
	FLMUINT			uiFlags;
	FLMUINT32		ui32Value;
	FLMUINT			uiLoop;
	FlmRecord *		pRec = NULL;
	FLMBOOL			bTransActive = FALSE;
	char				szTest [100];

	f_sprintf( szTest, "Sorted Fields Query Test, %s",
		(char *)(bDoRootedFieldPaths ? (char *)"Rooted" : (char *)"Non-Rooted"));
	beginTest( szTest);
	
	// Initialize a cursor
	
	if (RC_BAD( rc = FlmCursorInit( m_hDb, FLM_DATA_CONTAINER, &hCursor)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorInit", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Add a field path - use a field number right in the middle.
	
	uiFieldPath [0] = PERSON_TAG;
	uiFieldPath [1] = 1301;
	uiFieldPath [2] = 1301;
	uiFieldPath [3] = 1301;
	uiFieldPath [4] = 0;
	uiFlags = (bDoRootedFieldPaths) ? FLM_ROOTED_PATH : 0; 
	if (RC_BAD( rc = FlmCursorAddFieldPath( hCursor, uiFieldPath, uiFlags)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddFieldPath", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Add the equals operator.
	
	if (RC_BAD( rc = FlmCursorAddOp( hCursor, FLM_EQ_OP, FALSE)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddOp", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Add the value we are comparing to.
	
	ui32Value = 1301;
	if (RC_BAD( rc = FlmCursorAddValue( hCursor, FLM_UINT32_VAL, &ui32Value, 4)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddValue", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Start a read transaction

	if( RC_BAD( rc = FlmDbTransBegin( m_hDb, FLM_READ_TRANS, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransBegin", rc, m_szFailInfo);
		goto Exit;
	}
	bTransActive = TRUE;

	// Do the query 250 times.  Get a timing to print out.

	for (uiLoop = 0; uiLoop < 250; uiLoop++)
	{
		if (RC_BAD( rc = FlmCursorFirst( hCursor, &pRec)))
		{
			MAKE_ERROR_STRING( "calling FlmCursorFirst", rc, m_szFailInfo);
			goto Exit;
		}
		if (pRec->getID() != uiDrn)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, "Got incorrect record back from query");
			goto Exit;
		}
	}
	bPassed = TRUE;
	
Exit:

	if (hCursor != HFCURSOR_NULL)
	{
		FlmCursorFree( &hCursor);
	}

	if (pRec)
	{
		pRec->Release();
	}
	
	if (bTransActive)
	{
		(void)FlmDbTransAbort( m_hDb);
	}

	endTest( bPassed);
	return( rc);
}
		
/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::addRecWithFLMUINT(
	FLMUINT		uiNum,
	FLMUINT *	puiDrn)
{
	RCODE			rc = FERR_OK;
	char			szErr [100];
	void *		pvDataField;
	FLMUINT		uiTestNum;
	FlmRecord *	pDataRec = NULL;
	FLMUINT		uiDrn;
	
	// Create a person object

	if( (pDataRec = f_new FlmRecord) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		MAKE_ERROR_STRING( "allocating FlmRecord", rc, m_szFailInfo);
		goto Exit;
	}
	if( RC_BAD( rc = pDataRec->insertLast( 0, PERSON_TAG,
		FLM_TEXT_TYPE, &pvDataField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	
	if( RC_BAD( rc = pDataRec->insertLast( 1, NUMBER_TAG,
		FLM_NUMBER_TYPE, &pvDataField)))
	{
		f_sprintf( szErr, "calling insertLast to add Unsigned %u",
			(unsigned)uiNum);
		MAKE_ERROR_STRING( szErr, rc, m_szFailInfo);
		goto Exit;
	}
	if (RC_BAD( rc = pDataRec->setUINT( pvDataField, uiNum)))
	{
		f_sprintf( szErr, "calling setUINT to add Unsigned %u",
			(unsigned)uiNum);
		MAKE_ERROR_STRING( szErr, rc, m_szFailInfo);
		goto Exit;
	}
	
	// Verify that we get back the expected data.
	// Set the test number to a different value than the passed in value to
	// ensure that we are actually changing it in the get call.
	
	uiTestNum = uiNum + 1;
	if (RC_BAD( rc = pDataRec->getUINT( pvDataField, &uiTestNum)))
	{
		f_sprintf( szErr, "calling getUINT for Unsigned value %u",
			(unsigned)uiNum);
		MAKE_ERROR_STRING( szErr, rc, m_szFailInfo);
		goto Exit;
	}
	if (uiTestNum != uiNum)
	{
		rc = RC_SET( FERR_FAILURE);
		f_sprintf( m_szFailInfo, 
			"Unsigned value set not retrieved, Set: %u, Retrieved: %u",
			(unsigned)uiNum, (unsigned)uiTestNum);
		goto Exit;
	}
	
	uiDrn = 0;	
	if( RC_BAD( rc = FlmRecordAdd( m_hDb, FLM_DATA_CONTAINER, 
		&uiDrn, pDataRec, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordAdd", rc, m_szFailInfo);
		goto Exit;
	}
	if (puiDrn)
	{
		*puiDrn = uiDrn;
	}
	
Exit:

	if (pDataRec)
	{
		pDataRec->Release();
	}
	
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::addRecWithFLMUINT64(
	FLMUINT64	ui64Num,
	FLMUINT *	puiDrn)
{
	RCODE			rc = FERR_OK;
	char			szErr [100];
	void *		pvDataField;
	FLMUINT64	ui64TestNum;
	FlmRecord *	pDataRec = NULL;
	FLMUINT		uiDrn;
	
	// Create a person object

	if( (pDataRec = f_new FlmRecord) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		MAKE_ERROR_STRING( "allocating FlmRecord", rc, m_szFailInfo);
		goto Exit;
	}
	
	if( RC_BAD( rc = pDataRec->insertLast( 0, PERSON_TAG,
		FLM_TEXT_TYPE, &pvDataField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	
	if( RC_BAD( rc = pDataRec->insertLast( 1, NUMBER_TAG,
		FLM_NUMBER_TYPE, &pvDataField)))
	{
		f_sprintf( szErr, "calling insertLast to add Unsigned64 %I64u", ui64Num);
		MAKE_ERROR_STRING( szErr, rc, m_szFailInfo);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDataRec->setUINT64( pvDataField, ui64Num)))
	{
		f_sprintf( szErr, "calling setUINT64 to add Unsigned64 %I64u", ui64Num);
		MAKE_ERROR_STRING( szErr, rc, m_szFailInfo);
		goto Exit;
	}
	
	// Verify that we get back the expected data.
	// Set the test number to a different value than the passed in value to
	// ensure that we are actually changing it in the get call.
	
	ui64TestNum = ui64Num + 1;
	if (RC_BAD( rc = pDataRec->getUINT64( pvDataField, &ui64TestNum)))
	{
		f_sprintf( szErr, "calling getUINT64 for Unsigned64 value %I64u", ui64Num);
		MAKE_ERROR_STRING( szErr, rc, m_szFailInfo);
		goto Exit;
	}
	if (ui64TestNum != ui64Num)
	{
		rc = RC_SET( FERR_FAILURE);
		f_sprintf( m_szFailInfo, 
			"Unsigned64 value set not retrieved, Set: %I64u, Retrieved: %I64u",
			ui64Num, ui64TestNum);
		goto Exit;
	}
	
	uiDrn = 0;	
	if( RC_BAD( rc = FlmRecordAdd( m_hDb, FLM_DATA_CONTAINER, 
		&uiDrn, pDataRec, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordAdd", rc, m_szFailInfo);
		goto Exit;
	}
	if (puiDrn)
	{
		*puiDrn = uiDrn;
	}
	
Exit:

	if (pDataRec)
	{
		pDataRec->Release();
	}
	
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::addRecWithFLMINT(
	FLMINT		iNum,
	FLMUINT *	puiDrn)
{
	RCODE			rc = FERR_OK;
	char			szErr [100];
	void *		pvDataField;
	FLMINT		iTestNum;
	FlmRecord *	pDataRec = NULL;
	FLMUINT		uiDrn;
	
	// Create a person object

	if( (pDataRec = f_new FlmRecord) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		MAKE_ERROR_STRING( "allocating FlmRecord", rc, m_szFailInfo);
		goto Exit;
	}
	if( RC_BAD( rc = pDataRec->insertLast( 0, PERSON_TAG,
		FLM_TEXT_TYPE, &pvDataField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	
	if( RC_BAD( rc = pDataRec->insertLast( 1, NUMBER_TAG,
		FLM_NUMBER_TYPE, &pvDataField)))
	{
		f_sprintf( szErr, "calling insertLast to add Signed %d", (int)iNum);
		MAKE_ERROR_STRING( szErr, rc, m_szFailInfo);
		goto Exit;
	}
	if (RC_BAD( rc = pDataRec->setINT( pvDataField, iNum)))
	{
		f_sprintf( szErr, "calling setINT to add Signed %d", (int)iNum);
		MAKE_ERROR_STRING( szErr, rc, m_szFailInfo);
		goto Exit;
	}
	
	// Verify that we get back the expected data.
	// Set the test number to a different value than the passed in value to
	// ensure that we are actually changing it in the get call.
	
	iTestNum = iNum + 1;
	if (RC_BAD( rc = pDataRec->getINT( pvDataField, &iTestNum)))
	{
		f_sprintf( szErr, "calling getINT for Signed value %d", (int)iNum);
		MAKE_ERROR_STRING( szErr, rc, m_szFailInfo);
		goto Exit;
	}
	if (iTestNum != iNum)
	{
		rc = RC_SET( FERR_FAILURE);
		f_sprintf( m_szFailInfo, 
			"Signed value set not retrieved, Set: %d, Retrieved: %d",
			(int)iNum, (int)iTestNum);
		goto Exit;
	}
	
	uiDrn = 0;	
	if( RC_BAD( rc = FlmRecordAdd( m_hDb, FLM_DATA_CONTAINER, 
		&uiDrn, pDataRec, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordAdd", rc, m_szFailInfo);
		goto Exit;
	}
	if (puiDrn)
	{
		*puiDrn = uiDrn;
	}
	
Exit:

	if (pDataRec)
	{
		pDataRec->Release();
	}
	
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::addRecWithFLMINT64(
	FLMINT64		i64Num,
	FLMUINT *	puiDrn)
{
	RCODE			rc = FERR_OK;
	char			szErr [100];
	void *		pvDataField;
	FLMINT64		i64TestNum;
	FlmRecord *	pDataRec = NULL;
	FLMUINT		uiDrn;
	
	// Create a person object

	if( (pDataRec = f_new FlmRecord) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		MAKE_ERROR_STRING( "allocating FlmRecord", rc, m_szFailInfo);
		goto Exit;
	}
	
	if( RC_BAD( rc = pDataRec->insertLast( 0, PERSON_TAG,
		FLM_TEXT_TYPE, &pvDataField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	
	if( RC_BAD( rc = pDataRec->insertLast( 1, NUMBER_TAG,
		FLM_NUMBER_TYPE, &pvDataField)))
	{
		f_sprintf( szErr, "calling insertLast to add Signed64 %I64d", i64Num);
		MAKE_ERROR_STRING( szErr, rc, m_szFailInfo);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDataRec->setINT64( pvDataField, i64Num)))
	{
		f_sprintf( szErr, "calling setINT64 to add Signed64 %I64d", i64Num);
		MAKE_ERROR_STRING( szErr, rc, m_szFailInfo);
		goto Exit;
	}
	
	// Verify that we get back the expected data.
	// Set the test number to a different value than the passed in value to
	// ensure that we are actually changing it in the get call.
	
	i64TestNum = i64Num + 1;
	if (RC_BAD( rc = pDataRec->getINT64( pvDataField, &i64TestNum)))
	{
		f_sprintf( szErr, "calling getINT64 for Signed64 value %I64d", i64Num);
		MAKE_ERROR_STRING( szErr, rc, m_szFailInfo);
		goto Exit;
	}
	
	if (i64TestNum != i64Num)
	{
		rc = RC_SET( FERR_FAILURE);
		f_sprintf( m_szFailInfo, 
			"Signed64 value set not retrieved, Set: %I64d, Retrieved: %I64d",
			i64Num, i64TestNum);
		goto Exit;
	}
	
	uiDrn = 0;	
	if( RC_BAD( rc = FlmRecordAdd( m_hDb, FLM_DATA_CONTAINER, 
		&uiDrn, pDataRec, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordAdd", rc, m_szFailInfo);
		goto Exit;
	}
	if (puiDrn)
	{
		*puiDrn = uiDrn;
	}
	
Exit:

	if (pDataRec)
	{
		pDataRec->Release();
	}
	
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::numbersTest(
	FLMUINT *	puiDrn)
{
	RCODE					rc = FERR_OK;
	FLMBOOL				bPassed = FALSE;
	FLMBOOL				bTransActive = FALSE;
	FLMUINT				uiLoop;
	FLMUINT_TEST *		pFLMUINTTest;
	FLMUINT64_TEST *	pFLMUINT64Test;
	FLMINT_TEST *		pFLMINTTest;
	FLMINT64_TEST *	pFLMINT64Test;

	beginTest( "64 Bit Numbers Test");

	// Start an update transaction

	if( RC_BAD( rc = FlmDbTransBegin( m_hDb, FLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransBegin", rc, m_szFailInfo);
		goto Exit;
	}
	bTransActive = TRUE;

	for (uiLoop = 0, pFLMUINTTest = &gv_FLMUINTTests [0];
		  uiLoop < NUM_FLMUINT_TESTS;
		  uiLoop++, pFLMUINTTest++)
	{
		if (RC_BAD( rc = addRecWithFLMUINT( pFLMUINTTest->uiNum, puiDrn)))
		{
			goto Exit;
		}

		// Don't want to get anything except the first DRN

		puiDrn = NULL;
	}
	for (uiLoop = 0, pFLMINTTest = &gv_FLMINTTests [0];
		  uiLoop < NUM_FLMINT_TESTS;
		  uiLoop++, pFLMINTTest++)
	{
		if (RC_BAD( rc = addRecWithFLMINT( pFLMINTTest->iNum, puiDrn)))
		{
			goto Exit;
		}
		// Don't want to get anything except the first DRN

		puiDrn = NULL;
	}
	for (uiLoop = 0, pFLMUINT64Test = &gv_FLMUINT64Tests [0];
		  uiLoop < NUM_FLMUINT64_TESTS;
		  uiLoop++, pFLMUINT64Test++)
	{
		if (RC_BAD( rc = addRecWithFLMUINT64( pFLMUINT64Test->ui64Num, puiDrn)))
		{
			goto Exit;
		}

		// Don't want to get anything except the first DRN

		puiDrn = NULL;
	}
	for (uiLoop = 0, pFLMINT64Test = &gv_FLMINT64Tests [0];
		  uiLoop < NUM_FLMINT64_TESTS;
		  uiLoop++, pFLMINT64Test++)
	{
		if (RC_BAD( rc = addRecWithFLMINT64( pFLMINT64Test->i64Num, puiDrn)))
		{
			goto Exit;
		}

		// Don't want to get anything except the first DRN

		puiDrn = NULL;
	}
	
	if( RC_BAD( rc = FlmDbTransCommit( m_hDb)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransCommit", rc, m_szFailInfo);
		goto Exit;
	}
	bTransActive = FALSE;
	
	bPassed = TRUE;
	
Exit:

	if (bTransActive)
	{
		(void)FlmDbTransAbort( m_hDb);
	}

	endTest( bPassed);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::testNumField(
	FLMUINT *	puiDrn,
	FLMUINT		uiExpectedNum,
	RCODE			rcExpectedUINT,
	FLMINT		iExpectedNum,
	RCODE			rcExpectedINT,
	FLMUINT64	ui64ExpectedNum,
	RCODE			rcExpectedUINT64,
	FLMINT64		i64ExpectedNum,
	RCODE			rcExpectedINT64)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiNum;
	FLMINT			iNum;
	FLMUINT64		ui64Num;
	FLMINT64			i64Num;
	FlmRecord *		pDataRec = NULL;
	void *			pvField;
	FLMUINT			uiDrn;
	
	if (RC_BAD( rc = FlmRecordRetrieve( m_hDb, FLM_DATA_CONTAINER, *puiDrn,
					FO_EXACT, &pDataRec, &uiDrn)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordRetrieve", rc, m_szFailInfo);
		goto Exit;
	}
	(*puiDrn)++;

	pvField = pDataRec->firstChild( pDataRec->root());
	
	// Test FLMUINT - set number different from expected number so we can verify
	// that we actually changed the value.
	
	uiNum = uiExpectedNum + 1;
	rc = pDataRec->getUINT( pvField, &uiNum);
	if (rc != rcExpectedUINT)
	{
		f_assert( 0);
		f_sprintf( m_szFailInfo, "Unexpected rc (%e) from getUINT, expected %e. Num: %u",
				rc, rcExpectedUINT, (unsigned)uiExpectedNum);
		if (rc == FERR_OK)
		{
			rc = RC_SET( FERR_FAILURE);
		}
		goto Exit;
	}
	else if (RC_OK( rc))
	{
		if (uiNum != uiExpectedNum)
		{
			rc = RC_SET_AND_ASSERT( FERR_FAILURE);
			f_sprintf( m_szFailInfo, "Unexpected UINT (%u) from getUINT, expected %u",
					(unsigned)uiNum, (unsigned)uiExpectedNum);
			goto Exit;
		}
	}
	else
	{
		// Need to set to FERR_OK, even though we expected it to fail,
		// so the caller will not exit.
		
		rc = FERR_OK;
	}
	
	// Test FLMINT - set number different from expected number so we can verify
	// that we actually changed the value.
	
	iNum = iExpectedNum + 1;
	rc = pDataRec->getINT( pvField, &iNum);
	if (rc != rcExpectedINT)
	{
		f_sprintf( m_szFailInfo, "Unexpected rc (%e) from getINT, expected %e. Num: %d",
				rc, rcExpectedINT, (int)iExpectedNum);
		if (rc == FERR_OK)
		{
			rc = RC_SET_AND_ASSERT( FERR_FAILURE);
		}
		goto Exit;
	}
	else if (RC_OK( rc))
	{
		if (iNum != iExpectedNum)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, "Unexpected INT (%d) from getINT, expected %d",
					(int)iNum, (int)iExpectedNum);
			goto Exit;
		}
	}
	else
	{
		// Need to set to FERR_OK, even though we expected it to fail,
		// so the caller will not exit.
		
		rc = FERR_OK;
	}
	
	// Test FLMUINT64 - set number different from expected number so we can verify
	// that we actually changed the value.
	
	ui64Num = ui64ExpectedNum + 1;
	rc = pDataRec->getUINT64( pvField, &ui64Num);
	if (rc != rcExpectedUINT64)
	{
		f_assert( 0);
		f_sprintf( m_szFailInfo, "Unexpected rc (%e) from getUINT64, expected %e. Num: %I64u",
				rc, rcExpectedUINT64, ui64ExpectedNum);
		if (rc == FERR_OK)
		{
			rc = RC_SET( FERR_FAILURE);
		}
		goto Exit;
	}
	else if (RC_OK( rc))
	{
		if (ui64Num != ui64ExpectedNum)
		{
			rc = RC_SET_AND_ASSERT( FERR_FAILURE);
			f_sprintf( m_szFailInfo, "Unexpected UINT64 (%I64u) from getUINT64, expected %I64u",
					ui64Num, ui64ExpectedNum);
			goto Exit;
		}
	}
	else
	{
		// Need to set to FERR_OK, even though we expected it to fail,
		// so the caller will not exit.
		
		rc = FERR_OK;
	}
	
	// Test FLMINT64 - set number different from expected number so we can verify
	// that we actually changed the value.
	
	i64Num = i64ExpectedNum + 1;
	rc = pDataRec->getINT64( pvField, &i64Num);
	if (rc != rcExpectedINT64)
	{
		f_assert( 0);
		f_sprintf( m_szFailInfo, "Unexpected rc (%e) from getINT64, expected %e. Num: %I64d",
				rc, rcExpectedINT64, i64ExpectedNum);
		if (rc == FERR_OK)
		{
			rc = RC_SET( FERR_FAILURE);
		}
		goto Exit;
	}
	else if (RC_OK( rc))
	{
		if (i64Num != i64ExpectedNum)
		{
			f_sprintf( m_szFailInfo, "Unexpected INT64 (%I64d) from getINT64, expected %I64d",
					i64Num, i64ExpectedNum);
			rc = RC_SET_AND_ASSERT( FERR_FAILURE);
			goto Exit;
		}
	}
	else
	{
		// Need to set to FERR_OK, even though we expected it to fail,
		// so the caller will not exit.
		
		rc = FERR_OK;
	}
	
Exit:

	if (pDataRec)
	{
		pDataRec->Release();
	}
	
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::numbersRetrieveTest(
	FLMUINT	uiDrn)
{
	RCODE					rc = FERR_OK;
	FLMBOOL				bPassed = FALSE;
	FLMUINT				uiLoop;
	FLMUINT_TEST *		pFLMUINTTest;
	FLMUINT64_TEST *	pFLMUINT64Test;
	FLMINT_TEST *		pFLMINTTest;
	FLMINT64_TEST *	pFLMINT64Test;

	beginTest( "64 Bit Numbers Query Test");
	
	for (uiLoop = 0, pFLMUINTTest = &gv_FLMUINTTests [0];
		  uiLoop < NUM_FLMUINT_TESTS;
		  uiLoop++, pFLMUINTTest++)
	{
		if (RC_BAD( rc = testNumField( &uiDrn,
									(FLMUINT)pFLMUINTTest->uiNum, pFLMUINTTest->rcExpectedGetUINT,
									(FLMINT)pFLMUINTTest->uiNum, pFLMUINTTest->rcExpectedGetINT,
									(FLMUINT64)pFLMUINTTest->uiNum, pFLMUINTTest->rcExpectedGetUINT64,
									(FLMINT64)pFLMUINTTest->uiNum, pFLMUINTTest->rcExpectedGetINT64)))
		{
			goto Exit;
		}
	}
	for (uiLoop = 0, pFLMINTTest = &gv_FLMINTTests [0];
		  uiLoop < NUM_FLMINT_TESTS;
		  uiLoop++, pFLMINTTest++)
	{
		if (RC_BAD( rc = testNumField( &uiDrn,
									(FLMUINT)pFLMINTTest->iNum, pFLMINTTest->rcExpectedGetUINT,
									(FLMINT)pFLMINTTest->iNum, pFLMINTTest->rcExpectedGetINT,
									(FLMUINT64)pFLMINTTest->iNum, pFLMINTTest->rcExpectedGetUINT64,
									(FLMINT64)pFLMINTTest->iNum, pFLMINTTest->rcExpectedGetINT64)))
		{
			goto Exit;
		}
	}
	for (uiLoop = 0, pFLMUINT64Test = &gv_FLMUINT64Tests [0];
		  uiLoop < NUM_FLMUINT64_TESTS;
		  uiLoop++, pFLMUINT64Test++)
	{
		if (RC_BAD( rc = testNumField( &uiDrn,
									(FLMUINT)pFLMUINT64Test->ui64Num, pFLMUINT64Test->rcExpectedGetUINT,
									(FLMINT)pFLMUINT64Test->ui64Num, pFLMUINT64Test->rcExpectedGetINT,
									(FLMUINT64)pFLMUINT64Test->ui64Num, pFLMUINT64Test->rcExpectedGetUINT64,
									(FLMINT64)pFLMUINT64Test->ui64Num, pFLMUINT64Test->rcExpectedGetINT64)))
		{
			goto Exit;
		}
	}
	for (uiLoop = 0, pFLMINT64Test = &gv_FLMINT64Tests [0];
		  uiLoop < NUM_FLMINT64_TESTS;
		  uiLoop++, pFLMINT64Test++)
	{
		if (RC_BAD( rc = testNumField( &uiDrn,
									(FLMUINT)pFLMINT64Test->i64Num, pFLMINT64Test->rcExpectedGetUINT,
									(FLMINT)pFLMINT64Test->i64Num, pFLMINT64Test->rcExpectedGetINT,
									(FLMUINT64)pFLMINT64Test->i64Num, pFLMINT64Test->rcExpectedGetUINT64,
									(FLMINT64)pFLMINT64Test->i64Num, pFLMINT64Test->rcExpectedGetINT64)))
		{
			goto Exit;
		}
	}

	bPassed = TRUE;
	
Exit:

	endTest( bPassed);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::numbersKeyRetrieveTest( void)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bPassed = FALSE;
	FLMUINT		uiFlags = FO_FIRST;
	FlmRecord *	pSearchKey = NULL;
	FlmRecord *	pFoundKey = NULL;
	FLMUINT		uiFoundDrn = 0;
	void *		pvField;
	FLMUINT		uiCurrKey = 0;
	FLMUINT64	ui64Num;
	FLMINT64		i64Num;

	beginTest( "Numbers FlmKeyRetrieve Test");
	for (;;)
	{
		if (RC_BAD( rc = FlmKeyRetrieve( m_hDb, NUMBER_IX,
								0, pSearchKey, 0, uiFlags,
								&pFoundKey, &uiFoundDrn)))
		{
			if (rc == FERR_EOF_HIT)
			{
				if (uiCurrKey != NUM_NUM_KEYS)
				{
					rc = RC_SET( FERR_DATA_ERROR);
					f_sprintf( m_szFailInfo, "Unexpected number of keys in index: %u, Expected %u",
						(unsigned)uiCurrKey, (unsigned)NUM_NUM_KEYS);
					goto Exit;
				}
				rc = FERR_OK;
				break;
			}
			else
			{
				MAKE_ERROR_STRING( "calling FlmKeyRetrieve", rc, m_szFailInfo);
				goto Exit;
			}
		}
		
		// Make sure we have not matched all of the expected keys yet.
		
		if (uiCurrKey == NUM_NUM_KEYS)
		{
			rc = RC_SET( FERR_DATA_ERROR);
			f_sprintf( m_szFailInfo, "Too many keys in index, only expecting %u",
				(unsigned)NUM_NUM_KEYS);
			goto Exit;
		}
		
		// Make sure this key is greater than the last key.
		
		if ((pvField = pFoundKey->find( pFoundKey->root(), NUMBER_TAG)) == NULL)
		{
			rc = RC_SET( FERR_DATA_ERROR);
			MAKE_ERROR_STRING( "corruption calling FlmRecord->find()",
				rc, m_szFailInfo);
			goto Exit;
		}

		if (gv_ExpectedNumIxValues [uiCurrKey].bUnsigned)
		{
			if (RC_BAD( rc = pFoundKey->getUINT64( pvField, &ui64Num)))
			{
				MAKE_ERROR_STRING( "calling FlmRecord->getUINT64", rc, m_szFailInfo);
				goto Exit;
			}
			if (ui64Num != gv_ExpectedNumIxValues [uiCurrKey].ui64Value)
			{
				rc = RC_SET( FERR_DATA_ERROR);
				f_sprintf( m_szFailInfo, "Unexpected unsigned value in index key[%u]: %I64u, Expected: %I64u",
					(unsigned)uiCurrKey, ui64Num, gv_ExpectedNumIxValues [uiCurrKey].ui64Value);
				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = pFoundKey->getINT64( pvField, &i64Num)))
			{
				MAKE_ERROR_STRING( "calling FlmRecord->getINT64", rc, m_szFailInfo);
				goto Exit;
			}
			if (i64Num != gv_ExpectedNumIxValues [uiCurrKey].i64Value)
			{
				rc = RC_SET( FERR_DATA_ERROR);
				f_sprintf( m_szFailInfo, "Unexpected signed value in index key[%u]: %I64d, Expected: %I64d",
					(unsigned)uiCurrKey, i64Num, gv_ExpectedNumIxValues [uiCurrKey].i64Value);
				goto Exit;
			}
		}
		
		// Setup to get the next key.

		uiCurrKey++;		
		uiFlags = FO_EXCL;
		if (pSearchKey)
		{
			pSearchKey->Release();
		}
		pSearchKey = pFoundKey;
		pFoundKey = NULL;
		uiFoundDrn = 0;
	}
	bPassed = TRUE;

Exit:

	if (pSearchKey)
	{
		pSearchKey->Release();
	}
	if (pFoundKey)
	{
		pFoundKey->Release();
	}

	endTest( bPassed);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
FSTATIC const char * opToStr(
	QTYPES	eOp)
{
	switch (eOp)
	{
		case FLM_EQ_OP: return( "==");
		case FLM_GT_OP: return( ">");
		case FLM_LT_OP: return( "<");
		case FLM_GE_OP: return( ">=");
		case FLM_LE_OP: return( "<=");
		case FLM_NE_OP: return( "!=");
		default:
			flmAssert( 0);
			return( "!!!");
	}
}

/***************************************************************************
Desc:
****************************************************************************/
FSTATIC const char * addOrSubtractOne(
	FLMBOOL	bAddOne,
	FLMBOOL	bSubtractOne)
{
	if (bAddOne)
	{
		return( "+ 1");
	}
	else if (bSubtractOne)
	{
		return( "- 1");
	}
	else
	{
		return( " ");
	}
}

/***************************************************************************
Desc:
****************************************************************************/
FSTATIC FLMBOOL nonNegResultMatchesNonNegValueExpr(
	FLMUINT64	ui64Result,
	FLMUINT64	ui64Value,
	QTYPES		eOp,
	FLMBOOL		bAddOne,
	FLMBOOL		bSubtractOne)
{
	FLMBOOL	bMatches = TRUE;
	
	switch (eOp)
	{
		case FLM_EQ_OP:
			if (bAddOne)
			{
				if (ui64Value == FLM_MAX_UINT64 || ui64Result != ui64Value + 1)
				{
					bMatches = FALSE;
				}
			}
			else if (bSubtractOne)
			{
				if (!ui64Value || ui64Result != ui64Value - 1)
				{
					bMatches = FALSE;
				}
			}
			else if (ui64Result != ui64Value)
			{
				bMatches = FALSE;
			}
			break;
		case FLM_GT_OP:
			if (bAddOne)
			{
				if (ui64Value == FLM_MAX_UINT64 || ui64Result <= ui64Value + 1)
				{
					bMatches = FALSE;
				}
			}
			else if (bSubtractOne)
			{
				if (ui64Value && ui64Result <= ui64Value - 1)
				{
					bMatches = FALSE;
				}
			}
			else if (ui64Result <= ui64Value)
			{
				bMatches = FALSE;
			}
			break;
		case FLM_LT_OP:
			if (bAddOne)
			{
				if (ui64Value < FLM_MAX_UINT64 && ui64Result >= ui64Value + 1)
				{
					bMatches = FALSE;
				}
			}
			else if (bSubtractOne)
			{
				if (!ui64Value || ui64Result >= ui64Value - 1)
				{
					bMatches = FALSE;
				}
			}
			else if (ui64Result >= ui64Value)
			{
				bMatches = FALSE;
			}
			break;
		case FLM_GE_OP:
			if (bAddOne)
			{
				if (ui64Value == FLM_MAX_UINT64 || ui64Result < ui64Value + 1)
				{
					bMatches = FALSE;
				}
			}
			else if (bSubtractOne)
			{
				if (ui64Value && ui64Result < ui64Value - 1)
				{
					bMatches = FALSE;
				}
			}
			else if (ui64Result < ui64Value)
			{
				bMatches = FALSE;
			}
			break;
		case FLM_LE_OP:
			if (bAddOne)
			{
				if (ui64Value < FLM_MAX_UINT64 && ui64Result > ui64Value + 1)
				{
					bMatches = FALSE;
				}
			}
			else if (bSubtractOne)
			{
				if (!ui64Value || ui64Result > ui64Value - 1)
				{
					bMatches = FALSE;
				}
			}
			else if (ui64Result > ui64Value)
			{
				bMatches = FALSE;
			}
			break;
		case FLM_NE_OP:
			if (bAddOne)
			{
				if (ui64Value == FLM_MAX_UINT64 || ui64Result == ui64Value + 1)
				{
					bMatches = FALSE;
				}
			}
			else if (bSubtractOne)
			{
				if (ui64Value && ui64Result == ui64Value - 1)
				{
					bMatches = FALSE;
				}
			}
			else if (ui64Result == ui64Value)
			{
				bMatches = FALSE;
			}
			break;
		default:
			bMatches = FALSE;
			flmAssert( 0);
			break;
	}
	
	return( bMatches);
}

/***************************************************************************
Desc:
****************************************************************************/
FSTATIC FLMBOOL nonNegResultMatchesNegValueExpr(
	FLMUINT64	ui64Result,
	FLMINT64		i64Value,
	QTYPES		eOp,
	FLMBOOL		bAddOne,
	FLMBOOL		// bSubtractOne
	)
{
	FLMBOOL	bMatches = TRUE;
	
	switch (eOp)
	{
		case FLM_EQ_OP:
		
			// Only matches if result is zero, and value is -1 + 1.
				
			if (ui64Result != 0 || i64Value != -1 || !bAddOne)
			{
				bMatches = FALSE;
			}
			break;
			
		case FLM_GT_OP:
		
			// Always true except when result is zero and value = -1 + 1
			
			if (ui64Result == 0 && i64Value == -1 && bAddOne)
			{
				bMatches = FALSE;
			}
			break;
		case FLM_LT_OP:
		
			// Result is always >= 0, and highest value is zero (-1 + 1), so result
			// should never be < value +/- anything.
			
			bMatches = FALSE;
			break;
			
		case FLM_GE_OP:
		
			// This will always be true
			
			break;
			
		case FLM_LE_OP:
		
			// This can only be true when result is zero and value is -1 + 1
			
			if (ui64Result != 0 || i64Value != -1 || !bAddOne)
			{
				bMatches = FALSE;
			}
			break;
			
		case FLM_NE_OP:
		
			// This is always true except when result is zero and value is -1 + 1
			
			if (ui64Result == 0 && i64Value == -1 && bAddOne)
			{
				bMatches = FALSE;
			}
			break;
			
		default:
			bMatches = FALSE;
			flmAssert( 0);
			break;
	}
	
	return( bMatches);
}

/***************************************************************************
Desc:
****************************************************************************/
FSTATIC FLMBOOL negResultMatchesNonNegValueExpr(
	FLMINT64		i64Result,
	FLMUINT64	ui64Value,
	QTYPES		eOp,
	FLMBOOL,		// bAddOne,
	FLMBOOL		bSubtractOne)
{
	FLMBOOL	bMatches = TRUE;
	
	flmAssert( i64Result < 0);
	
	switch (eOp)
	{
		case FLM_EQ_OP:
		
			// This will only be true in one case: result is -1, and
			// value is 0 - 1.
			
			if (i64Result != -1 || !bSubtractOne || ui64Value)
			{
				bMatches = FALSE;
			}
			break;
			
		case FLM_GT_OP:
		
			// The largest that result could be is -1, and the smallest
			// value is 0 - 1, so > will never match.
			
			bMatches = FALSE;
			break;
			
		case FLM_LT_OP:
		
			// This will always be true except for the case where
			// result is -1, and value is 0 - 1.
			
			if (i64Result == -1 && bSubtractOne && !ui64Value)
			{
				bMatches = FALSE;
			}
			break;
		case FLM_GE_OP:
		
			// This will only be true when result is -1 and value
			// is 0 - 1
			
			if (i64Result != -1 || !bSubtractOne || ui64Value)
			{
				bMatches = FALSE;
			}
			break;
			
		case FLM_LE_OP:
		
			// This will always be TRUE, as the largest result must be
			// -1, and the smallest value can be 0 - 1.
			
			break;
			
		case FLM_NE_OP:
		
			// This will always be true, except for the case where
			// result is -1, and value is 0 - 1.
			
			if (i64Result == -1 && bSubtractOne && !ui64Value)
			{
				bMatches = FALSE;
			}
			break;
			
		default:
			flmAssert( 0);
			bMatches = FALSE;
			break;
	}
	
	return( bMatches);
}

/***************************************************************************
Desc:
****************************************************************************/
FSTATIC FLMBOOL negResultMatchesNegValueExpr(
	FLMINT64		i64Result,
	FLMINT64		i64Value,
	QTYPES		eOp,
	FLMBOOL		bAddOne,
	FLMBOOL		bSubtractOne)
{
	FLMBOOL	bMatches = TRUE;
	
	// Both values should be negative.
	
	flmAssert( i64Result < 0 && i64Value < 0);
	
	switch (eOp)
	{
		case FLM_EQ_OP:
			if (bAddOne)
			{
				if (i64Result != i64Value + 1)
				{
					bMatches = FALSE;
				}
			}
			else if (bSubtractOne)
			{
				if (i64Value == FLM_MIN_INT64 || i64Result != i64Value - 1)
				{
					bMatches = FALSE;
				}
			}
			else if (i64Result != i64Value)
			{
				bMatches = FALSE;
			}
			break;
			
		case FLM_GT_OP:
			if (bAddOne)
			{
				if (i64Result <= i64Value + 1)
				{
					bMatches = FALSE;
				}
			}
			else if (bSubtractOne)
			{
				if (i64Value != FLM_MIN_INT64 && i64Result <= i64Value - 1)
				{
					bMatches = FALSE;
				}
			}
			else if (i64Result <= i64Value)
			{
				bMatches = FALSE;
			}
			break;
			
		case FLM_LT_OP:
			if (bAddOne)
			{
				if (i64Result >= i64Value + 1)
				{
					bMatches = FALSE;
				}
			}
			else if (bSubtractOne)
			{
				if (i64Value == FLM_MIN_INT64 || i64Result >= i64Value - 1)
				{
					bMatches = FALSE;
				}
			}
			else if (i64Result >= i64Value)
			{
				bMatches = FALSE;
			}
			break;
			
		case FLM_GE_OP:
			if (bAddOne)
			{
				if (i64Result < i64Value + 1)
				{
					bMatches = FALSE;
				}
			}
			else if (bSubtractOne)
			{
				if (i64Value != FLM_MIN_INT64 && i64Result < i64Value - 1)
				{
					bMatches = FALSE;
				}
			}
			else if (i64Result < i64Value)
			{
				bMatches = FALSE;
			}
			break;
			
		case FLM_LE_OP:
			if (bAddOne)
			{
				if (i64Result > i64Value + 1)
				{
					bMatches = FALSE;
				}
			}
			else if (bSubtractOne)
			{
				if (i64Value == FLM_MIN_INT64 || i64Result > i64Value - 1)
				{
					bMatches = FALSE;
				}
			}
			else if (i64Result > i64Value)
			{
				bMatches = FALSE;
			}
			break;
			
		case FLM_NE_OP:
			if (bAddOne)
			{
				if (i64Result == i64Value + 1)
				{
					bMatches = FALSE;
				}
			}
			else if (bSubtractOne)
			{
				if (i64Value != FLM_MIN_INT64 && i64Result == i64Value - 1)
				{
					bMatches = FALSE;
				}
			}
			else if (i64Result == i64Value)
			{
				bMatches = FALSE;
			}
			break;
			
		default:
			bMatches = FALSE;
			flmAssert( 0);
			break;
	}
	
	return( bMatches);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::verifyFLMUINT64ValueExpr(
	FLMUINT64	ui64Value,
	QTYPES		eOp,
	FLMBOOL		bAddOne,
	FLMBOOL		bSubtractOne,
	FlmRecord *	pRec)
{
	RCODE			rc = FERR_OK;
	void *		pvField;
	FLMUINT64	ui64Result;
	FLMINT64		i64Result;
	
	// Verify that the record has a field and that it matches the
	// criteria we specified.
	
	if ((pvField = pRec->find( pRec->root(), NUMBER_TAG)) == NULL)
	{
		rc = RC_SET( FERR_DATA_ERROR);
		MAKE_ERROR_STRING( "corruption calling FlmRecord->find()", 
			rc, m_szFailInfo);
		goto Exit;
	}
		
	if (RC_BAD( rc = pRec->getUINT64( pvField, &ui64Result)))
	{
		if (rc != FERR_CONV_NUM_UNDERFLOW)
		{
			MAKE_ERROR_STRING( "calling FlmRecord->getUINT64()", rc, m_szFailInfo);
			goto Exit;
		}
		if (RC_BAD( rc = pRec->getINT64( pvField, &i64Result)))
		{
			MAKE_ERROR_STRING( "calling FlmRecord->getINT64()", 
				rc, m_szFailInfo);
			goto Exit;
		}
		
		// Results should only be negative at this point.
		
		if (!negResultMatchesNonNegValueExpr( i64Result, ui64Value, eOp,
					bAddOne, bSubtractOne))
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, "Invalid result returned for query: "
				"%I64d %s %I64u %s",
				i64Result, opToStr( eOp), ui64Value,
				addOrSubtractOne( bAddOne, bSubtractOne));
			goto Exit;
		}
	}
	else
	{
		if (!nonNegResultMatchesNonNegValueExpr( ui64Result, ui64Value,
						eOp, bAddOne, bSubtractOne))
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, "Invalid result returned for query: "
				"%I64u %s %I64u %s",
				ui64Result, opToStr( eOp), ui64Value,
				addOrSubtractOne( bAddOne, bSubtractOne));
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::verifyFLMINT64ValueExpr(
	FLMINT64		i64Value,
	QTYPES		eOp,
	FLMBOOL		bAddOne,
	FLMBOOL		bSubtractOne,
	FlmRecord *	pRec)
{
	RCODE			rc = FERR_OK;
	void *		pvField;
	FLMUINT64	ui64Result;
	FLMINT64		i64Result;
	
	// Verify that the record has a field and that it matches the
	// criteria we specified.
	
	if ((pvField = pRec->find( pRec->root(), NUMBER_TAG)) == NULL)
	{
		rc = RC_SET( FERR_DATA_ERROR);
		MAKE_ERROR_STRING( "corruption calling FlmRecord->find()", 
			rc, m_szFailInfo);
		goto Exit;
	}
		
	if (RC_BAD( rc = pRec->getUINT64( pvField, &ui64Result)))
	{
		if (rc != FERR_CONV_NUM_UNDERFLOW)
		{
			MAKE_ERROR_STRING( "calling FlmRecord->getUINT64()", rc, m_szFailInfo);
			goto Exit;
		}
		if (RC_BAD( rc = pRec->getINT64( pvField, &i64Result)))
		{
			MAKE_ERROR_STRING( "calling FlmRecord->getINT64()", 
				rc, m_szFailInfo);
			goto Exit;
		}
		flmAssert( i64Result < 0);
		
		if (i64Value < 0)
		{
			if (!negResultMatchesNegValueExpr( i64Result, i64Value,
							eOp, bAddOne, bSubtractOne))
			{
Err_Neg_Result:
				rc = RC_SET( FERR_FAILURE);
				f_sprintf( m_szFailInfo, "Invalid result returned for query: "
					"%I64d %s %I64d %s",
					i64Result, opToStr( eOp), i64Value,
					addOrSubtractOne( bAddOne, bSubtractOne));
				goto Exit;
			}
		}
		else
		{
			if (!negResultMatchesNonNegValueExpr( i64Result, (FLMUINT64)i64Value,
							eOp, bAddOne, bSubtractOne))
			{
				goto Err_Neg_Result;
			}
		}
	}
	else
	{
		if (i64Value < 0)
		{
			if (!nonNegResultMatchesNegValueExpr( ui64Result, i64Value,
							eOp, bAddOne, bSubtractOne))
			{
Err_NonNeg_Result:
				rc = RC_SET( FERR_FAILURE);
				f_sprintf( m_szFailInfo, "Invalid result returned for query: "
					"%I64u %s %I64d %s",
					ui64Result, opToStr( eOp), i64Value,
					addOrSubtractOne( bAddOne, bSubtractOne));
				goto Exit;
			}
		}
		else
		{
			if (!nonNegResultMatchesNonNegValueExpr( ui64Result, (FLMUINT64)i64Value,
							eOp, bAddOne, bSubtractOne))
			{
				goto Err_NonNeg_Result;
			}
		}
	}
	
Exit:

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::doFLMUINT64QueryTest(
	FLMUINT64	ui64Value,
	QTYPES		eOp,
	FLMBOOL		bAddOne,
	FLMBOOL		bSubtractOne)
{
	RCODE			rc = FERR_OK;
	HFCURSOR		hCursor = HFCURSOR_NULL;
	FLMUINT32	ui32Num;
	FlmRecord *	pRec = NULL;

	if( RC_BAD( rc = FlmCursorInit( m_hDb, FLM_DATA_CONTAINER, &hCursor)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorInit", rc, m_szFailInfo);
		goto Exit;
	}
	
	// We will search on the number field.

	if (RC_BAD( rc = FlmCursorAddField( hCursor, NUMBER_TAG, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddField", rc, m_szFailInfo);
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorAddOp( hCursor, eOp)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddOp", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Add the value
	
	if( RC_BAD( rc = FlmCursorAddValue( hCursor, FLM_UINT64_VAL, 
		&ui64Value, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddValue(ui64)", rc, m_szFailInfo);
		goto Exit;
	}
	
	if (bAddOne)
	{
		if( RC_BAD( rc = FlmCursorAddOp( hCursor, FLM_PLUS_OP)))
		{
			MAKE_ERROR_STRING( "calling FlmCursorAddOp(+)", rc, m_szFailInfo);
			goto Exit;
		}
		
		ui32Num = 1;
		if (RC_BAD( rc = FlmCursorAddValue( hCursor, FLM_UINT32_VAL, &ui32Num, 0)))
		{
			MAKE_ERROR_STRING( "calling FlmCursorAddValue(ui32=+1)", rc, m_szFailInfo);
			goto Exit;
		}
	}
	else if (bSubtractOne)
	{
		if( RC_BAD( rc = FlmCursorAddOp( hCursor, FLM_MINUS_OP)))
		{
			MAKE_ERROR_STRING( "calling FlmCursorAddOp(-)", rc, m_szFailInfo);
			goto Exit;
		}
		ui32Num = 1;
		if (RC_BAD( rc = FlmCursorAddValue( hCursor, FLM_UINT32_VAL, &ui32Num, 0)))
		{
			MAKE_ERROR_STRING( "calling FlmCursorAddValue(ui32=-1)", rc, m_szFailInfo);
			goto Exit;
		}
	}
	
	// Get the results
	
	for (;;)
	{
		if (RC_BAD( rc = FlmCursorNext( hCursor, &pRec)))
		{
			if (rc == FERR_EOF_HIT)
			{
				rc = FERR_OK;
				break;
			}
			else
			{
				f_sprintf( m_szFailInfo, "Error %e (file: %s, line %u) calling FlmCursorNext for query: "
					"FLD %s %I64u %s",
					rc, __FILE__, (unsigned)__LINE__,
					opToStr( eOp), ui64Value,
					addOrSubtractOne( bAddOne, bSubtractOne));
				goto Exit;
			}
		}
		
		if (RC_BAD( rc = verifyFLMUINT64ValueExpr( ui64Value, eOp, bAddOne,
									bSubtractOne, pRec)))
		{
			goto Exit;
		}
		
	}
	
Exit:

	if (pRec)
	{
		pRec->Release();
	}

	if (hCursor != HFCURSOR_NULL)
	{
		(void)FlmCursorFree( &hCursor);
	}

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::queryTestsFLMUINT64(
	FLMUINT64	ui64Value)
{
	RCODE	rc = FERR_OK;
	
	// N == Value
	
	if (RC_BAD( rc = doFLMUINT64QueryTest( ui64Value, FLM_EQ_OP, FALSE, FALSE)))
	{
		goto Exit;
	}
	
	// N == Value + 1
	
	if (ui64Value < FLM_MAX_UINT64)
	{
		if (RC_BAD( rc = doFLMUINT64QueryTest( ui64Value, FLM_EQ_OP, TRUE, FALSE)))
		{
			goto Exit;
		}
	}
	
	// N == Value - 1
	
	if (RC_BAD( rc = doFLMUINT64QueryTest( ui64Value, FLM_EQ_OP, FALSE, TRUE)))
	{
		goto Exit;
	}
	
	// N > Value
	
	if (RC_BAD( rc = doFLMUINT64QueryTest( ui64Value, FLM_GT_OP, FALSE, FALSE)))
	{
		goto Exit;
	}
	
	// N > Value + 1
	
	if (ui64Value < FLM_MAX_UINT64)
	{
		if (RC_BAD( rc = doFLMUINT64QueryTest( ui64Value, FLM_GT_OP, TRUE, FALSE)))
		{
			goto Exit;
		}
	}
	
	// N > Value - 1
	
	if (RC_BAD( rc = doFLMUINT64QueryTest( ui64Value, FLM_GT_OP, FALSE, TRUE)))
	{
		goto Exit;
	}
	
	// N < Value
	
	if (RC_BAD( rc = doFLMUINT64QueryTest( ui64Value, FLM_LT_OP, FALSE, FALSE)))
	{
		goto Exit;
	}
	
	// N < Value + 1
	
	if (ui64Value < FLM_MAX_UINT64)
	{
		if (RC_BAD( rc = doFLMUINT64QueryTest( ui64Value, FLM_LT_OP, TRUE, FALSE)))
		{
			goto Exit;
		}
	}
	
	// N < Value - 1
	
	if (RC_BAD( rc = doFLMUINT64QueryTest( ui64Value, FLM_LT_OP, FALSE, TRUE)))
	{
		goto Exit;
	}
	
	// N >= Value
	
	if (RC_BAD( rc = doFLMUINT64QueryTest( ui64Value, FLM_GE_OP, FALSE, FALSE)))
	{
		goto Exit;
	}
	
	// N >= Value + 1
	
	if (ui64Value < FLM_MAX_UINT64)
	{
		if (RC_BAD( rc = doFLMUINT64QueryTest( ui64Value, FLM_GE_OP, TRUE, FALSE)))
		{
			goto Exit;
		}
	}
	
	// N >= Value - 1
	
	if (RC_BAD( rc = doFLMUINT64QueryTest( ui64Value, FLM_GE_OP, FALSE, TRUE)))
	{
		goto Exit;
	}
	
	// N <= Value
	
	if (RC_BAD( rc = doFLMUINT64QueryTest( ui64Value, FLM_LE_OP, FALSE, FALSE)))
	{
		goto Exit;
	}
	
	// N <= Value + 1
	
	if (ui64Value < FLM_MAX_UINT64)
	{
		if (RC_BAD( rc = doFLMUINT64QueryTest( ui64Value, FLM_LE_OP, TRUE, FALSE)))
		{
			goto Exit;
		}
	}
	
	// N <= Value - 1
	
	if (RC_BAD( rc = doFLMUINT64QueryTest( ui64Value, FLM_LE_OP, FALSE, TRUE)))
	{
		goto Exit;
	}
	
	// N != Value
	
	if (RC_BAD( rc = doFLMUINT64QueryTest( ui64Value, FLM_NE_OP, FALSE, FALSE)))
	{
		goto Exit;
	}
	
	// N != Value + 1
	
	if (ui64Value < FLM_MAX_UINT64)
	{
		if (RC_BAD( rc = doFLMUINT64QueryTest( ui64Value, FLM_NE_OP, TRUE, FALSE)))
		{
			goto Exit;
		}
	}
	
	// N != Value - 1
	
	if (RC_BAD( rc = doFLMUINT64QueryTest( ui64Value, FLM_NE_OP, FALSE, TRUE)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::doFLMINT64QueryTest(
	FLMINT64		i64Value,
	QTYPES		eOp,
	FLMBOOL		bAddOne,
	FLMBOOL		bSubtractOne)
{
	RCODE			rc = FERR_OK;
	HFCURSOR		hCursor = HFCURSOR_NULL;
	FLMUINT32	ui32Num;
	FlmRecord *	pRec = NULL;

	if( RC_BAD( rc = FlmCursorInit( m_hDb, FLM_DATA_CONTAINER, &hCursor)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorInit", rc, m_szFailInfo);
		goto Exit;
	}
	
	// We will search on the number field.

	if (RC_BAD( rc = FlmCursorAddField( hCursor, NUMBER_TAG, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddField", rc, m_szFailInfo);
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorAddOp( hCursor, eOp)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddOp", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Add the value
	
	if( RC_BAD( rc = FlmCursorAddValue( hCursor, FLM_INT64_VAL, 
		&i64Value, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddValue(i64)", rc, m_szFailInfo);
		goto Exit;
	}
	
	if (bAddOne)
	{
		if( RC_BAD( rc = FlmCursorAddOp( hCursor, FLM_PLUS_OP)))
		{
			MAKE_ERROR_STRING( "calling FlmCursorAddOp(+)", rc, m_szFailInfo);
			goto Exit;
		}
		
		ui32Num = 1;
		if (RC_BAD( rc = FlmCursorAddValue( hCursor, FLM_UINT32_VAL, &ui32Num, 0)))
		{
			MAKE_ERROR_STRING( "calling FlmCursorAddValue(ui32=+1)", rc, m_szFailInfo);
			goto Exit;
		}
	}
	else if (bSubtractOne)
	{
		if( RC_BAD( rc = FlmCursorAddOp( hCursor, FLM_MINUS_OP)))
		{
			MAKE_ERROR_STRING( "calling FlmCursorAddOp(-)", rc, m_szFailInfo);
			goto Exit;
		}
		ui32Num = 1;
		if (RC_BAD( rc = FlmCursorAddValue( hCursor, FLM_UINT32_VAL, &ui32Num, 0)))
		{
			MAKE_ERROR_STRING( "calling FlmCursorAddValue(ui32=-1)", rc, m_szFailInfo);
			goto Exit;
		}
	}
	
	// Get the results
	
	for (;;)
	{
		if (RC_BAD( rc = FlmCursorNext( hCursor, &pRec)))
		{
			if (rc == FERR_EOF_HIT)
			{
				rc = FERR_OK;
				break;
			}
			else
			{
				f_sprintf( m_szFailInfo, "Error %e (file: %s, line %u) calling FlmCursorNext for query: "
					"FLD %s %I64d %s",
					rc, __FILE__, (unsigned)__LINE__,
					opToStr( eOp), i64Value,
					addOrSubtractOne( bAddOne, bSubtractOne));
				goto Exit;
			}
		}
		
		if (RC_BAD( rc = verifyFLMINT64ValueExpr( i64Value, eOp, bAddOne,
									bSubtractOne, pRec)))
		{
			goto Exit;
		}
		
	}
	
Exit:

	if (pRec)
	{
		pRec->Release();
	}

	if (hCursor != HFCURSOR_NULL)
	{
		(void)FlmCursorFree( &hCursor);
	}

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::queryTestsFLMINT64(
	FLMINT64	i64Value)
{
	RCODE	rc = FERR_OK;
	
	// N == Value
	
	if (RC_BAD( rc = doFLMINT64QueryTest( i64Value, FLM_EQ_OP, FALSE, FALSE)))
	{
		goto Exit;
	}
	
	// N == Value + 1
	
	if (RC_BAD( rc = doFLMINT64QueryTest( i64Value, FLM_EQ_OP, TRUE, FALSE)))
	{
		goto Exit;
	}
	
	// N == Value - 1

	if (i64Value > FLM_MIN_INT64)
	{
		if (RC_BAD( rc = doFLMINT64QueryTest( i64Value, FLM_EQ_OP, FALSE, TRUE)))
		{
			goto Exit;
		}
	}
	
	// N > Value
	
	if (RC_BAD( rc = doFLMINT64QueryTest( i64Value, FLM_GT_OP, FALSE, FALSE)))
	{
		goto Exit;
	}
	
	// N > Value + 1
	
	if (RC_BAD( rc = doFLMINT64QueryTest( i64Value, FLM_GT_OP, TRUE, FALSE)))
	{
		goto Exit;
	}
	
	// N > Value - 1
	
	if (i64Value > FLM_MIN_INT64)
	{
		if (RC_BAD( rc = doFLMINT64QueryTest( i64Value, FLM_GT_OP, FALSE, TRUE)))
		{
			goto Exit;
		}
	}
	
	// N < Value
	
	if (RC_BAD( rc = doFLMINT64QueryTest( i64Value, FLM_LT_OP, FALSE, FALSE)))
	{
		goto Exit;
	}
	
	// N < Value + 1
	
	if (RC_BAD( rc = doFLMINT64QueryTest( i64Value, FLM_LT_OP, TRUE, FALSE)))
	{
		goto Exit;
	}
	
	// N < Value - 1
	
	if (i64Value > FLM_MIN_INT64)
	{
		if (RC_BAD( rc = doFLMINT64QueryTest( i64Value, FLM_LT_OP, FALSE, TRUE)))
		{
			goto Exit;
		}
	}
	
	// N >= Value
	
	if (RC_BAD( rc = doFLMINT64QueryTest( i64Value, FLM_GE_OP, FALSE, FALSE)))
	{
		goto Exit;
	}
	
	// N >= Value + 1
	
	if (RC_BAD( rc = doFLMINT64QueryTest( i64Value, FLM_GE_OP, TRUE, FALSE)))
	{
		goto Exit;
	}
	
	// N >= Value - 1
	
	if (i64Value > FLM_MIN_INT64)
	{
		if (RC_BAD( rc = doFLMINT64QueryTest( i64Value, FLM_GE_OP, FALSE, TRUE)))
		{
			goto Exit;
		}
	}
	
	// N <= Value
	
	if (RC_BAD( rc = doFLMINT64QueryTest( i64Value, FLM_LE_OP, FALSE, FALSE)))
	{
		goto Exit;
	}
	
	// N <= Value + 1
	
	if (RC_BAD( rc = doFLMINT64QueryTest( i64Value, FLM_LE_OP, TRUE, FALSE)))
	{
		goto Exit;
	}
	
	// N <= Value - 1
	
	if (i64Value > FLM_MIN_INT64)
	{
		if (RC_BAD( rc = doFLMINT64QueryTest( i64Value, FLM_LE_OP, FALSE, TRUE)))
		{
			goto Exit;
		}
	}
	
	// N != Value
	
	if (RC_BAD( rc = doFLMINT64QueryTest( i64Value, FLM_NE_OP, FALSE, FALSE)))
	{
		goto Exit;
	}
	
	// N != Value + 1
	
	if (RC_BAD( rc = doFLMINT64QueryTest( i64Value, FLM_NE_OP, TRUE, FALSE)))
	{
		goto Exit;
	}
	
	// N != Value - 1
	
	if (i64Value > FLM_MIN_INT64)
	{
		if (RC_BAD( rc = doFLMINT64QueryTest( i64Value, FLM_NE_OP, FALSE, TRUE)))
		{
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::doFLMUINT32QueryTest(
	FLMUINT32	ui32Value,
	QTYPES		eOp,
	FLMBOOL		bAddOne,
	FLMBOOL		bSubtractOne)
{
	RCODE			rc = FERR_OK;
	HFCURSOR		hCursor = HFCURSOR_NULL;
	FLMUINT32	ui32Num;
	FlmRecord *	pRec = NULL;

	if( RC_BAD( rc = FlmCursorInit( m_hDb, FLM_DATA_CONTAINER, &hCursor)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorInit", rc, m_szFailInfo);
		goto Exit;
	}
	
	// We will search on the number field.

	if (RC_BAD( rc = FlmCursorAddField( hCursor, NUMBER_TAG, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddField", rc, m_szFailInfo);
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorAddOp( hCursor, eOp)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddOp", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Add the value
	
	if( RC_BAD( rc = FlmCursorAddValue( hCursor, FLM_UINT32_VAL, 
		&ui32Value, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddValue(ui32)", rc, m_szFailInfo);
		goto Exit;
	}
	
	if (bAddOne)
	{
		if( RC_BAD( rc = FlmCursorAddOp( hCursor, FLM_PLUS_OP)))
		{
			MAKE_ERROR_STRING( "calling FlmCursorAddOp(+)", rc, m_szFailInfo);
			goto Exit;
		}
		
		ui32Num = 1;
		if (RC_BAD( rc = FlmCursorAddValue( hCursor, FLM_UINT32_VAL, &ui32Num, 0)))
		{
			MAKE_ERROR_STRING( "calling FlmCursorAddValue(ui32=+1)", rc, m_szFailInfo);
			goto Exit;
		}
	}
	else if (bSubtractOne)
	{
		if( RC_BAD( rc = FlmCursorAddOp( hCursor, FLM_MINUS_OP)))
		{
			MAKE_ERROR_STRING( "calling FlmCursorAddOp(-)", rc, m_szFailInfo);
			goto Exit;
		}
		ui32Num = 1;
		if (RC_BAD( rc = FlmCursorAddValue( hCursor, FLM_UINT32_VAL, &ui32Num, 0)))
		{
			MAKE_ERROR_STRING( "calling FlmCursorAddValue(ui32=-1)", rc, m_szFailInfo);
			goto Exit;
		}
	}
	
	// Get the results
	
	for (;;)
	{
		if (RC_BAD( rc = FlmCursorNext( hCursor, &pRec)))
		{
			if (rc == FERR_EOF_HIT)
			{
				rc = FERR_OK;
				break;
			}
			else
			{
				f_sprintf( m_szFailInfo, "Error %e (file: %s, line %u) calling FlmCursorNext for query: "
					"FLD %s %u %s",
					rc, __FILE__, (unsigned)__LINE__,
					opToStr( eOp), (unsigned)ui32Value,
					addOrSubtractOne( bAddOne, bSubtractOne));
				goto Exit;
			}
		}
		
		if (RC_BAD( rc = verifyFLMUINT64ValueExpr( (FLMUINT64)ui32Value, eOp, bAddOne,
									bSubtractOne, pRec)))
		{
			goto Exit;
		}
		
	}
	
Exit:

	if (pRec)
	{
		pRec->Release();
	}

	if (hCursor != HFCURSOR_NULL)
	{
		(void)FlmCursorFree( &hCursor);
	}

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::queryTestsFLMUINT32(
	FLMUINT32	ui32Value)
{
	RCODE	rc = FERR_OK;
	
	// N == Value
	
	if (RC_BAD( rc = doFLMUINT32QueryTest( ui32Value, FLM_EQ_OP, FALSE, FALSE)))
	{
		goto Exit;
	}
	
	// N == Value + 1
	
	if (RC_BAD( rc = doFLMUINT32QueryTest( ui32Value, FLM_EQ_OP, TRUE, FALSE)))
	{
		goto Exit;
	}
	
	// N == Value - 1
	
	if (RC_BAD( rc = doFLMUINT32QueryTest( ui32Value, FLM_EQ_OP, FALSE, TRUE)))
	{
		goto Exit;
	}
	
	// N > Value
	
	if (RC_BAD( rc = doFLMUINT32QueryTest( ui32Value, FLM_GT_OP, FALSE, FALSE)))
	{
		goto Exit;
	}
	
	// N > Value + 1
	
	if (RC_BAD( rc = doFLMUINT32QueryTest( ui32Value, FLM_GT_OP, TRUE, FALSE)))
	{
		goto Exit;
	}
	
	// N > Value - 1
	
	if (RC_BAD( rc = doFLMUINT32QueryTest( ui32Value, FLM_GT_OP, FALSE, TRUE)))
	{
		goto Exit;
	}
	
	// N < Value
	
	if (RC_BAD( rc = doFLMUINT32QueryTest( ui32Value, FLM_LT_OP, FALSE, FALSE)))
	{
		goto Exit;
	}
	
	// N < Value + 1
	
	if (RC_BAD( rc = doFLMUINT32QueryTest( ui32Value, FLM_LT_OP, TRUE, FALSE)))
	{
		goto Exit;
	}
	
	// N < Value - 1
	
	if (RC_BAD( rc = doFLMUINT32QueryTest( ui32Value, FLM_LT_OP, FALSE, TRUE)))
	{
		goto Exit;
	}
	
	// N >= Value
	
	if (RC_BAD( rc = doFLMUINT32QueryTest( ui32Value, FLM_GE_OP, FALSE, FALSE)))
	{
		goto Exit;
	}
	
	// N >= Value + 1
	
	if (RC_BAD( rc = doFLMUINT32QueryTest( ui32Value, FLM_GE_OP, TRUE, FALSE)))
	{
		goto Exit;
	}
	
	// N >= Value - 1
	
	if (RC_BAD( rc = doFLMUINT32QueryTest( ui32Value, FLM_GE_OP, FALSE, TRUE)))
	{
		goto Exit;
	}
	
	// N <= Value
	
	if (RC_BAD( rc = doFLMUINT32QueryTest( ui32Value, FLM_LE_OP, FALSE, FALSE)))
	{
		goto Exit;
	}
	
	// N <= Value + 1
	
	if (RC_BAD( rc = doFLMUINT32QueryTest( ui32Value, FLM_LE_OP, TRUE, FALSE)))
	{
		goto Exit;
	}
	
	// N <= Value - 1
	
	if (RC_BAD( rc = doFLMUINT32QueryTest( ui32Value, FLM_LE_OP, FALSE, TRUE)))
	{
		goto Exit;
	}
	
	// N != Value
	
	if (RC_BAD( rc = doFLMUINT32QueryTest( ui32Value, FLM_NE_OP, FALSE, FALSE)))
	{
		goto Exit;
	}
	
	// N != Value + 1
	
	if (RC_BAD( rc = doFLMUINT32QueryTest( ui32Value, FLM_NE_OP, TRUE, FALSE)))
	{
		goto Exit;
	}
	
	// N != Value - 1
	
	if (RC_BAD( rc = doFLMUINT32QueryTest( ui32Value, FLM_NE_OP, FALSE, TRUE)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::doFLMINT32QueryTest(
	FLMINT32		i32Value,
	QTYPES		eOp,
	FLMBOOL		bAddOne,
	FLMBOOL		bSubtractOne)
{
	RCODE			rc = FERR_OK;
	HFCURSOR		hCursor = HFCURSOR_NULL;
	FLMUINT32	ui32Num;
	FlmRecord *	pRec = NULL;

	if( RC_BAD( rc = FlmCursorInit( m_hDb, FLM_DATA_CONTAINER, &hCursor)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorInit", rc, m_szFailInfo);
		goto Exit;
	}
	
	// We will search on the number field.

	if (RC_BAD( rc = FlmCursorAddField( hCursor, NUMBER_TAG, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddField", rc, m_szFailInfo);
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorAddOp( hCursor, eOp)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddOp", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Add the value
	
	if( RC_BAD( rc = FlmCursorAddValue( hCursor, FLM_INT32_VAL, 
		&i32Value, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddValue(i32)", rc, m_szFailInfo);
		goto Exit;
	}
	
	if (bAddOne)
	{
		if( RC_BAD( rc = FlmCursorAddOp( hCursor, FLM_PLUS_OP)))
		{
			MAKE_ERROR_STRING( "calling FlmCursorAddOp(+)", rc, m_szFailInfo);
			goto Exit;
		}
		
		ui32Num = 1;
		if (RC_BAD( rc = FlmCursorAddValue( hCursor, FLM_UINT32_VAL, &ui32Num, 0)))
		{
			MAKE_ERROR_STRING( "calling FlmCursorAddValue(ui32=+1)", rc, m_szFailInfo);
			goto Exit;
		}
	}
	else if (bSubtractOne)
	{
		if( RC_BAD( rc = FlmCursorAddOp( hCursor, FLM_MINUS_OP)))
		{
			MAKE_ERROR_STRING( "calling FlmCursorAddOp(-)", rc, m_szFailInfo);
			goto Exit;
		}
		ui32Num = 1;
		if (RC_BAD( rc = FlmCursorAddValue( hCursor, FLM_UINT32_VAL, &ui32Num, 0)))
		{
			MAKE_ERROR_STRING( "calling FlmCursorAddValue(ui32=-1)", rc, m_szFailInfo);
			goto Exit;
		}
	}
	
	// Get the results
	
	for (;;)
	{
		if (RC_BAD( rc = FlmCursorNext( hCursor, &pRec)))
		{
			if (rc == FERR_EOF_HIT)
			{
				rc = FERR_OK;
				break;
			}
			else
			{
				f_sprintf( m_szFailInfo, "Error %e (file: %s, line %u) calling FlmCursorNext for query: "
					"FLD %s %d %s",
					rc, __FILE__, (unsigned)__LINE__,
					opToStr( eOp), (int)i32Value,
					addOrSubtractOne( bAddOne, bSubtractOne));
				goto Exit;
			}
		}
		
		if (RC_BAD( rc = verifyFLMINT64ValueExpr( (FLMINT64)i32Value, eOp, bAddOne,
									bSubtractOne, pRec)))
		{
			goto Exit;
		}
		
	}
	
Exit:

	if (pRec)
	{
		pRec->Release();
	}

	if (hCursor != HFCURSOR_NULL)
	{
		(void)FlmCursorFree( &hCursor);
	}

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::queryTestsFLMINT32(
	FLMINT32	i32Value)
{
	RCODE	rc = FERR_OK;
	
	// N == Value
	
	if (RC_BAD( rc = doFLMINT32QueryTest( i32Value, FLM_EQ_OP, FALSE, FALSE)))
	{
		goto Exit;
	}
	
	// N == Value + 1
	
	if (RC_BAD( rc = doFLMINT32QueryTest( i32Value, FLM_EQ_OP, TRUE, FALSE)))
	{
		goto Exit;
	}
	
	// N == Value - 1
	
	if (RC_BAD( rc = doFLMINT32QueryTest( i32Value, FLM_EQ_OP, FALSE, TRUE)))
	{
		goto Exit;
	}
	
	// N > Value
	
	if (RC_BAD( rc = doFLMINT32QueryTest( i32Value, FLM_GT_OP, FALSE, FALSE)))
	{
		goto Exit;
	}
	
	// N > Value + 1
	
	if (RC_BAD( rc = doFLMINT32QueryTest( i32Value, FLM_GT_OP, TRUE, FALSE)))
	{
		goto Exit;
	}
	
	// N > Value - 1
	
	if (RC_BAD( rc = doFLMINT32QueryTest( i32Value, FLM_GT_OP, FALSE, TRUE)))
	{
		goto Exit;
	}
	
	// N < Value
	
	if (RC_BAD( rc = doFLMINT32QueryTest( i32Value, FLM_LT_OP, FALSE, FALSE)))
	{
		goto Exit;
	}
	
	// N < Value + 1
	
	if (RC_BAD( rc = doFLMINT32QueryTest( i32Value, FLM_LT_OP, TRUE, FALSE)))
	{
		goto Exit;
	}
	
	// N < Value - 1
	
	if (RC_BAD( rc = doFLMINT32QueryTest( i32Value, FLM_LT_OP, FALSE, TRUE)))
	{
		goto Exit;
	}
	
	// N >= Value
	
	if (RC_BAD( rc = doFLMINT32QueryTest( i32Value, FLM_GE_OP, FALSE, FALSE)))
	{
		goto Exit;
	}
	
	// N >= Value + 1
	
	if (RC_BAD( rc = doFLMINT32QueryTest( i32Value, FLM_GE_OP, TRUE, FALSE)))
	{
		goto Exit;
	}
	
	// N >= Value - 1
	
	if (RC_BAD( rc = doFLMINT32QueryTest( i32Value, FLM_GE_OP, FALSE, TRUE)))
	{
		goto Exit;
	}
	
	// N <= Value
	
	if (RC_BAD( rc = doFLMINT32QueryTest( i32Value, FLM_LE_OP, FALSE, FALSE)))
	{
		goto Exit;
	}
	
	// N <= Value + 1
	
	if (RC_BAD( rc = doFLMINT32QueryTest( i32Value, FLM_LE_OP, TRUE, FALSE)))
	{
		goto Exit;
	}
	
	// N <= Value - 1
	
	if (RC_BAD( rc = doFLMINT32QueryTest( i32Value, FLM_LE_OP, FALSE, TRUE)))
	{
		goto Exit;
	}
	
	// N != Value
	
	if (RC_BAD( rc = doFLMINT32QueryTest( i32Value, FLM_NE_OP, FALSE, FALSE)))
	{
		goto Exit;
	}
	
	// N != Value + 1
	
	if (RC_BAD( rc = doFLMINT32QueryTest( i32Value, FLM_NE_OP, TRUE, FALSE)))
	{
		goto Exit;
	}
	
	// N != Value - 1
	
	if (RC_BAD( rc = doFLMINT32QueryTest( i32Value, FLM_NE_OP, FALSE, TRUE)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::doNumQueryTests(
	NUM_IX_VALUE *	pValue)
{
	RCODE	rc = FERR_OK;
	
	if (pValue->bUnsigned)
	{
		if (RC_BAD( rc = queryTestsFLMUINT64( (FLMUINT)pValue->ui64Value)))
		{
			goto Exit;
		}
		if (pValue->ui64Value <= (FLMUINT64)(FLM_MAX_INT64))
		{
			if (RC_BAD( rc = queryTestsFLMINT64( (FLMINT64)pValue->ui64Value)))
			{
				goto Exit;
			}
		}
		if (pValue->ui64Value <= (FLMUINT64)(FLM_MAX_UINT32))
		{
			if (RC_BAD( rc = queryTestsFLMUINT32( (FLMUINT32)pValue->ui64Value)))
			{
				goto Exit;
			}
		}
		if (pValue->ui64Value <= (FLMUINT64)(FLM_MAX_INT32))
		{
			if (RC_BAD( rc = queryTestsFLMINT32( (FLMINT32)pValue->ui64Value)))
			{
				goto Exit;
			}
		}
	}
	else
	{
		if (RC_BAD( rc = queryTestsFLMINT64( pValue->i64Value)))
		{
			goto Exit;
		}
		if (pValue->i64Value >= (FLMINT64)(FLM_MIN_INT32) &&
			 pValue->i64Value <= (FLMINT64)(FLM_MAX_INT32))
		{
			if (RC_BAD( rc = queryTestsFLMINT32( (FLMINT32)pValue->i64Value)))
			{
				goto Exit;
			}
		}
		if (pValue->i64Value >= 0 &&
			 pValue->i64Value <= (FLMINT64)(FLM_MAX_UINT32))
		{
			if (RC_BAD( rc = queryTestsFLMUINT32( (FLMUINT32)pValue->i64Value)))
			{
				goto Exit;
			}
		}
		if (pValue->i64Value >= 0)
		{
			if (RC_BAD( rc = queryTestsFLMUINT64( (FLMUINT64)pValue->i64Value)))
			{
				goto Exit;
			}
		}
	}
	
Exit:

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::numbersQueryTest( void)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bPassed = FALSE;
	FLMUINT		uiCurrKey;
	FLMBOOL		bTransActive = FALSE;

	beginTest( "Numbers Query Test");
	
	if( RC_BAD( rc = FlmDbTransBegin( m_hDb, FLM_READ_TRANS, 15)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransBegin", rc, m_szFailInfo);
		goto Exit;
	}
	bTransActive = TRUE;
	
	for (uiCurrKey = 0; uiCurrKey < NUM_NUM_KEYS; uiCurrKey++)
	{
		if (RC_BAD( rc = doNumQueryTests( &gv_ExpectedNumIxValues [uiCurrKey])))
		{
			goto Exit;
		}
	}
	bPassed = TRUE;

Exit:

	if (bTransActive)
	{
		(void)FlmDbTransAbort( m_hDb);
	}

	endTest( bPassed);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::reopenDbTest( void)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bPassed = FALSE;
	char		szDbName [F_PATH_MAX_SIZE];

	beginTest( "Close & Reopen Database Test");
	
	// Close the database
	
	if (RC_BAD( rc = FlmDbClose( &m_hDb)))
	{
		MAKE_ERROR_STRING( "calling FlmDbClose", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Force everything to really close.
	
	f_strcpy( szDbName, DB_NAME_STR);
	if (RC_BAD( rc = FlmConfig( FLM_CLOSE_FILE, (void *)&szDbName [0], NULL)))
	{
		MAKE_ERROR_STRING( "calling FlmConfig(FLM_CLOSE_FILE)", rc, m_szFailInfo);
		goto Exit;
	}

	// Force everything from cache.
	
	if (RC_BAD( rc = FlmConfig( FLM_CLOSE_UNUSED_FILES, (void *)0, (void *)0)))
	{
		MAKE_ERROR_STRING( "calling FlmConfig(FLM_CLOSE_UNUSED_FILES)", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Reopen the database.
	
	if( RC_BAD( rc = FlmDbOpen( DB_NAME_STR, NULL, NULL,
								FO_DONT_RESUME_BACKGROUND_THREADS, NULL, &m_hDb)))
	{
		MAKE_ERROR_STRING( "calling FlmDbOpen", rc, m_szFailInfo);
		goto Exit;
	}

	bPassed = TRUE;
	
Exit:

	endTest( bPassed);

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::backupRestoreDbTest( void)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bPassed = FALSE;
	HFBACKUP	hBackup = HFBACKUP_NULL;
	char		szTest [200];
	
	// Do the backup
	
	beginTest( "Backup Test");
	if (RC_BAD( rc = FlmDbBackupBegin( m_hDb, FLM_FULL_BACKUP, TRUE, &hBackup)))
	{
		MAKE_ERROR_STRING( "calling FlmDbBackupBegin", rc, m_szFailInfo);
		goto Exit;
	}
	if (RC_BAD( rc = FlmDbBackup( hBackup, BACKUP_PATH, NULL, NULL, NULL, NULL,
								NULL)))
	{
		MAKE_ERROR_STRING( "calling FlmDbBackup", rc, m_szFailInfo);
		goto Exit;
	}
	if (RC_BAD( rc = FlmDbBackupEnd( &hBackup)))
	{
		MAKE_ERROR_STRING( "calling FlmDbBackupEnd", rc, m_szFailInfo);
		goto Exit;
	}
	endTest( TRUE);
	
	// Do the restore
	
	f_sprintf( szTest, "Restore Backup To %s Test", DB_RESTORE_NAME_STR);
	beginTest( szTest);
	if (RC_BAD( rc = FlmDbRestore( DB_RESTORE_NAME_STR, NULL, BACKUP_PATH,
								NULL, NULL, NULL)))
	{
		MAKE_ERROR_STRING( "calling FlmDbRestore", rc, m_szFailInfo);
		goto Exit;
	}
	bPassed = TRUE;
	
Exit:

	if (hBackup != HFBACKUP_NULL)
	{
		(void)FlmDbBackupEnd( &hBackup);
	}

	endTest( bPassed);
	return( rc);
}
	
/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::compareRecords(
	const char *	pszDb1,
	const char *	pszDb2,
	const char *	pszWhat,
	FlmRecord *		pRecord1,
	FlmRecord *		pRecord2)
{
	RCODE		rc = FERR_OK;
	void *	pvField1 = pRecord1->root();
	void *	pvField2 = pRecord2->root();
	FLMUINT	uiFieldNum1;
	FLMUINT	uiFieldNum2;
	FLMUINT	uiLevel1;
	FLMUINT	uiLevel2;
	FLMUINT	uiDataType1;
	FLMUINT	uiDataType2;
	FLMUINT	uiDataLength1;
	FLMUINT	uiDataLength2;
	FLMUINT	uiEncLength1;
	FLMUINT	uiEncLength2;
	FLMUINT	uiEncId1;
	FLMUINT	uiEncId2;
	
	for (;;)
	{
		if (!pvField1)
		{
			if (pvField2)
			{
				rc = RC_SET( FERR_FAILURE);
				f_sprintf( m_szFailInfo, "%s in %s has more fields than in %s",
					pszWhat, pszDb2, pszDb1);
				goto Exit;
			}
			else
			{
				break;
			}
		}
		else if (!pvField2)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, "%s in %s has more fields than in %s",
				pszWhat, pszDb1, pszDb2);
			goto Exit;
		}
		
		// Compare the field number, data type, etc.
		
		if (RC_BAD( rc = pRecord1->getFieldInfo( pvField1, &uiFieldNum1,
												&uiLevel1, &uiDataType1, &uiDataLength1,
												&uiEncLength1, &uiEncId1)))
		{
			MAKE_ERROR_STRING( "calling FlmRecord->getFieldInfo", rc, m_szFailInfo);
			goto Exit;
		}
		if (RC_BAD( rc = pRecord2->getFieldInfo( pvField2, &uiFieldNum2,
												&uiLevel2, &uiDataType2, &uiDataLength2,
												&uiEncLength2, &uiEncId2)))
		{
			MAKE_ERROR_STRING( "calling FlmRecord->getFieldInfo", 
				rc, m_szFailInfo);
			goto Exit;
		}
		
		if (uiFieldNum1 != uiFieldNum2)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, "Field Num mismatch in %s, %s: %u, %s: %u",
				pszWhat,
				pszDb1, (unsigned)uiFieldNum1,
				pszDb2, (unsigned)uiFieldNum2);
			goto Exit;
		}
		if (uiLevel1 != uiLevel2)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, 
				"Field Level mismatch in %s, Fld: %u, %s: %u, %s: %u",
				pszWhat, (unsigned)uiFieldNum1,
				pszDb1, (unsigned)uiLevel1,
				pszDb2, (unsigned)uiLevel2);
			goto Exit;
		}
		if (uiDataLength1 != uiDataLength2)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, 
				"Field Length mismatch in %s, Fld: %u, %s: %u, %s: %u",
				pszWhat, (unsigned)uiFieldNum1,
				pszDb1, (unsigned)uiDataLength1,
				pszDb2, (unsigned)uiDataLength2);
			goto Exit;
		}

		// Data type may not match for FLAIM's reserved field numbers.  This is because
		// FLAIM does not store the field type for these fields - it just assumes that
		// they are FLM_TEXT_TYPE.  However, we also have some code where FLAIM puts
		// a FLM_CONTEXT_TYPE into the field when it is created inside a FlmRecord
		// object.  If that object remains cached, but the one we are comparing it to
		// is not cached, we would have a mismatch.  Hence, we simply ignore field
		// type for these fields.

		if (uiDataType1 != uiDataType2 && uiFieldNum1 < FLM_DICT_FIELD_NUMS)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, "Field Type mismatch in %s, Fld: %u, %s: %u, %s: %u",
				pszWhat, (unsigned)uiFieldNum1,
				pszDb1, (unsigned)uiDataType1,
				pszDb2, (unsigned)uiDataType2);
			goto Exit;
		}
		if (uiEncLength1 != uiEncLength2)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, "Field Enc. Length mismatch in %s, Fld: %u, %s: %u, %s: %u",
				pszWhat, (unsigned)uiFieldNum1,
				pszDb1, (unsigned)uiEncLength1,
				pszDb2, (unsigned)uiEncLength2);
			goto Exit;
		}
		if (uiEncId1 != uiEncId2)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, "Field Enc. Id mismatch in %s, Fld: %u, %s: %u, %s: %u",
				pszWhat, (unsigned)uiFieldNum1,
				pszDb1, (unsigned)uiEncId1,
				pszDb2, (unsigned)uiEncId2);
			goto Exit;
		}
		
		// Compare the data
		
		if (uiDataLength1)
		{
			const FLMBYTE *	pucData1 = pRecord1->getDataPtr( pvField1);
			const FLMBYTE *	pucData2 = pRecord2->getDataPtr( pvField2);
			
			if (f_memcmp( pucData1, pucData2, uiDataLength1) != 0)
			{
				rc = RC_SET( FERR_FAILURE);
				f_sprintf( m_szFailInfo, "Field Value mismatch in %s, Fld: %u",
					pszWhat, (unsigned)uiFieldNum1);
				goto Exit;
			}
		}
		
		// Go to the next field in each key.
		
		pvField1 = pRecord1->next( pvField1);
		pvField2 = pRecord2->next( pvField2);
	}
	
Exit:

	return( rc);
}
		
/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::compareIndexes(
	const char *	pszDb1,
	const char *	pszDb2,
	HFDB				hDb1,
	HFDB				hDb2,
	FLMUINT			uiIndexNum)
{
	RCODE			rc = FERR_OK;
	RCODE			rc1;
	RCODE			rc2;
	FLMUINT		uiFlags1;
	FLMUINT		uiFlags2;
	FlmRecord *	pSearchKey1 = NULL;
	FLMUINT		uiSearchDrn1 = 0;
	FlmRecord *	pSearchKey2 = NULL;
	FLMUINT		uiSearchDrn2 = 0;
	FlmRecord *	pFoundKey1 = NULL;
	FLMUINT		uiFoundDrn1 = 0;
	FlmRecord *	pFoundKey2 = NULL;
	FLMUINT		uiFoundDrn2 = 0;
	char			szWhat [40];
	FLMUINT		uiCount = 0;
	
	// Read through all keys and references in the index.  Make sure they
	// are identical.

	uiFlags1 = FO_FIRST;
	uiFlags2 = FO_FIRST;
	for (;;)
	{
		rc1 = FlmKeyRetrieve( hDb1, uiIndexNum,
								0, pSearchKey1, uiSearchDrn1, uiFlags1,
								&pFoundKey1, &uiFoundDrn1);
		rc2 = FlmKeyRetrieve( hDb2, uiIndexNum,
								0, pSearchKey2, uiSearchDrn2, uiFlags2,
								&pFoundKey2, &uiFoundDrn2);
		if (RC_BAD( rc1))
		{
			if (rc1 == FERR_EOF_HIT)
			{
				if (RC_OK( rc2))
				{
					rc = RC_SET( FERR_FAILURE);
					f_sprintf( m_szFailInfo, "%s has more keys/refs in index %u than %2",
							pszDb2, (unsigned)uiIndexNum, pszDb1);
					goto Exit;
				}
				else if (rc2 == FERR_EOF_HIT)
				{
					break;
				}
				else
				{
					rc = rc2;
					MAKE_ERROR_STRING( "calling FlmKeyRetrieve", rc2, m_szFailInfo);
					goto Exit;
				}
			}
			else
			{
				rc = rc1;
				MAKE_ERROR_STRING( "calling FlmKeyRetrieve", rc1, m_szFailInfo);
				goto Exit;
			}
		}
		else
		{
			if (rc2 == FERR_EOF_HIT)
			{
				rc = RC_SET( FERR_FAILURE);
				f_sprintf( m_szFailInfo, "%s has more keys/refs in index %u than %2",
						pszDb1, (unsigned)uiIndexNum, pszDb2);
				goto Exit;
			}
			else if (RC_BAD( rc2))
			{
				rc = rc2;
				MAKE_ERROR_STRING( "calling FlmKeyRetrieve", rc2, m_szFailInfo);
				goto Exit;
			}
		}
		
		// Compare the two keys.
		
		uiCount++;
		f_sprintf( szWhat, "Ix #%u, Key #%u", (unsigned)uiIndexNum, (unsigned)uiCount);
		if (RC_BAD( rc = compareRecords( pszDb1, pszDb2, szWhat,
									pFoundKey1, pFoundKey2)))
		{
			goto Exit;
		}
		
		// Compare the references.
		
		if (uiFoundDrn1 != uiFoundDrn2)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, "Ref DRN mismatch in %s, %s: %u, %s: %u",
					szWhat, (unsigned)uiIndexNum,
					pszDb1, (unsigned)uiFoundDrn1, pszDb2, (unsigned)uiFoundDrn2);
			goto Exit;
		}
	
		// Setup to get the next key.
		
		uiFlags1 = FO_EXCL;
		uiSearchDrn1 = uiFoundDrn1;
		if (pSearchKey1)
		{
			pSearchKey1->Release();
		}
		pSearchKey1 = pFoundKey1;
		pFoundKey1 = NULL;
		uiFoundDrn1 = 0;
		
		uiFlags2 = FO_EXCL;
		uiSearchDrn2 = uiFoundDrn2;
		if (pSearchKey2)
		{
			pSearchKey2->Release();
		}
		pSearchKey2 = pFoundKey2;
		pFoundKey2 = NULL;
		uiFoundDrn2 = 0;
	}
	
Exit:

	if (pSearchKey1)
	{
		pSearchKey1->Release();
	}
	if (pSearchKey2)
	{
		pSearchKey2->Release();
	}
	if (pFoundKey1)
	{
		pFoundKey1->Release();
	}
	if (pFoundKey2)
	{
		pFoundKey2->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::compareContainers(
	const char *	pszDb1,
	const char *	pszDb2,
	HFDB				hDb1,
	HFDB				hDb2,
	FLMUINT			uiContainerNum)
{
	RCODE			rc = FERR_OK;
	RCODE			rc1;
	RCODE			rc2;
	FlmRecord *	pRecord1 = NULL;
	FLMUINT		uiDrn1;
	FlmRecord *	pRecord2 = NULL;
	FLMUINT		uiDrn2;
	char			szWhat [40];
	
	// Read through all records in the container.  Make sure they
	// are identical.

	uiDrn1 = 1;
	uiDrn2 = 1;
	for (;;)
	{
		rc1 = FlmRecordRetrieve( hDb1, uiContainerNum, uiDrn1,
						FO_INCL, &pRecord1, &uiDrn1);
		rc2 = FlmRecordRetrieve( hDb2, uiContainerNum, uiDrn2,
						FO_INCL, &pRecord2, &uiDrn2);
		if (RC_BAD( rc1))
		{
			if (rc1 == FERR_EOF_HIT)
			{
				if (RC_OK( rc2))
				{
					rc = RC_SET( FERR_FAILURE);
					f_sprintf( m_szFailInfo, "%s has more records in container %u than %s",
							pszDb2, (unsigned)uiContainerNum, pszDb1);
					goto Exit;
				}
				else if (rc2 == FERR_EOF_HIT)
				{
					break;
				}
				else
				{
					rc = rc2;
					MAKE_ERROR_STRING( "calling FlmRecordRetrieve", rc2, m_szFailInfo);
					goto Exit;
				}
			}
			else
			{
				rc = rc1;
				MAKE_ERROR_STRING( "calling FlmRecordRetrieve", rc1, m_szFailInfo);
				goto Exit;
			}
		}
		else
		{
			if (rc2 == FERR_EOF_HIT)
			{
				rc = RC_SET( FERR_FAILURE);
				f_sprintf( m_szFailInfo, "%s has more records in container %u than %s",
						pszDb1, (unsigned)uiContainerNum, pszDb2);
				goto Exit;
			}
			else if (RC_BAD( rc2))
			{
				rc = rc2;
				MAKE_ERROR_STRING( "calling FlmRecordRetrieve", rc2, m_szFailInfo);
				goto Exit;
			}
		}
		
		// Make sure these records have the same DRN
		
		if (uiDrn1 != uiDrn2)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, "DRN mismatch in container %u, %s: %u, %s: %u",
					(unsigned)uiContainerNum,
					pszDb1, (unsigned)uiDrn1, pszDb2, (unsigned)uiDrn2);
			goto Exit;
		}
		
		// Compare the two records.
		
		f_sprintf( szWhat, "Cont #%u, Rec #%u", (unsigned)uiContainerNum,
				(unsigned)uiDrn1);
		if (RC_BAD( rc = compareRecords( pszDb1, pszDb2, szWhat,
									pRecord1, pRecord2)))
		{
			goto Exit;
		}
		
		// If we are doing the dictionary container, we will jump out
		// to check any containers or indexes that it defines.

		if (uiContainerNum == FLM_DICT_CONTAINER)
		{
			if (pRecord1->getFieldID( pRecord1->root()) == FLM_CONTAINER_TAG)
			{
				if (RC_BAD( rc = compareContainers( pszDb1, pszDb2, hDb1, hDb2, uiDrn1)))
				{
					goto Exit;
				}
			}
			else if (pRecord1->getFieldID( pRecord1->root()) == FLM_INDEX_TAG)
			{
				if (RC_BAD( rc = compareIndexes( pszDb1, pszDb2, hDb1, hDb2, uiDrn1)))
				{
					goto Exit;
				}
			}
		}

		uiDrn1++;
		uiDrn2++;
	}
	
Exit:

	if (pRecord1)
	{
		pRecord1->Release();
	}
	if (pRecord2)
	{
		pRecord2->Release();
	}
	return( rc);
}
	
/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::compareDbTest(
	const char *	pszDb1,
	const char *	pszDb2)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bPassed = FALSE;
	char			szTest [200];
	HFDB			hDb1 = HFDB_NULL;
	HFDB			hDb2 = HFDB_NULL;
	
	f_sprintf( szTest, "Compare Database Test (%s,%s)",
		pszDb1, pszDb2);
	beginTest( szTest);
	
	// Open each database.

	if( RC_BAD( rc = FlmDbOpen( pszDb1, NULL, NULL,
							FO_DONT_RESUME_BACKGROUND_THREADS, NULL, &hDb1)))
	{
		MAKE_ERROR_STRING( "calling FlmDbOpen", rc, m_szFailInfo);
		goto Exit;
	}
	if( RC_BAD( rc = FlmDbOpen( pszDb2, NULL, NULL,
							FO_DONT_RESUME_BACKGROUND_THREADS, NULL, &hDb2)))
	{
		MAKE_ERROR_STRING( "calling FlmDbOpen", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Need to compare all of the records in the default data
	// container and the tracker container
	
	if (RC_BAD( rc = compareContainers( pszDb1, pszDb2, hDb1, hDb2, FLM_DATA_CONTAINER)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = compareContainers( pszDb1, pszDb2, hDb1, hDb2, FLM_TRACKER_CONTAINER)))
	{
		goto Exit;
	}
	
	// Compare all of the keys in the dictionary index
	
	if (RC_BAD( rc = compareIndexes( pszDb1, pszDb2, hDb1, hDb2, FLM_DICT_INDEX)))
	{
		goto Exit;
	}

	// Compare the records in the dictionary container.
	// This will cause recursive calls to compareContainers for any
	// containers defined in the dictionary, as well as calls to
	// compareIndexes for an indexes defined in the dictionary.

	if (RC_BAD( rc = compareContainers( pszDb1, pszDb2, hDb1, hDb2, FLM_DICT_CONTAINER)))
	{
		goto Exit;
	}
	
	bPassed = TRUE;
	
Exit:

	if (hDb1 != HFDB_NULL)
	{
		(void)FlmDbClose( &hDb1);
	}
	if (hDb2 != HFDB_NULL)
	{
		(void)FlmDbClose( &hDb2);
	}

	endTest( bPassed);
	return( rc);
}
	
/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::checkDbTest(
	const char *			pszDbName)
{
	RCODE						rc = FERR_OK;
	FLMBOOL					bPassed = FALSE;
	char						szTest [200];
	F_Pool					pool;
	DB_CHECK_PROGRESS		checkStats;
	
	pool.poolInit( 512);
	
	f_sprintf( szTest, "Check Database Test (%s)", pszDbName);
	beginTest( szTest);

	if( RC_BAD( rc = FlmDbCheck( HFDB_NULL, pszDbName, NULL, NULL,
			FLM_CHK_INDEX_REFERENCING | FLM_CHK_FIELDS, &pool, &checkStats,
			NULL, NULL)))
	{
		MAKE_ERROR_STRING( "calling FlmDbCheck", rc, m_szFailInfo);
		goto Exit;
	}
	
	if( checkStats.bLogicalIndexCorrupt || checkStats.bPhysicalCorrupt)
	{
		rc = RC_SET( FERR_DATA_ERROR);
		MAKE_ERROR_STRING( "calling FlmDbCheck", rc, m_szFailInfo);
		goto Exit;
	}
	
	bPassed = TRUE;
	
Exit:

	endTest( bPassed);
	return( rc);
}
	
/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::rebuildDbTest(
	const char *	pszDestDbName,
	const char *	pszSrcDbName)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bPassed = FALSE;
	char		szTest [200];
	FLMUINT	uiTotalRecords;
	FLMUINT	uiRecsRecovered;
	
	f_sprintf( szTest, "Rebuild Database Test (%s --> %s)", pszSrcDbName,
			pszDestDbName);
	beginTest( szTest);

	if( RC_BAD( rc = FlmDbRebuild( pszSrcDbName, NULL, pszDestDbName, NULL, NULL,
							NULL, NULL, &uiTotalRecords, &uiRecsRecovered,
							NULL, NULL)))
	{
		MAKE_ERROR_STRING( "calling FlmDbRebuild", rc, m_szFailInfo);
		goto Exit;
	}
	bPassed = TRUE;
	
Exit:

	endTest( bPassed);
	return( rc);
}
	
/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::copyDbTest(
	const char *	pszDestDbName,
	const char *	pszSrcDbName)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bPassed = FALSE;
	char		szTest [200];
	
	// FlmDbCopy will copy a database and all of its files

	f_sprintf( szTest, "Copy Database Test (%s --> %s)",
		pszSrcDbName, pszDestDbName);
	beginTest( szTest);

	if( RC_BAD( rc = FlmDbCopy( pszSrcDbName, NULL, NULL,
										 pszDestDbName, NULL, NULL, NULL, NULL)))
	{
		MAKE_ERROR_STRING( "calling FlmDbCopy", rc, m_szFailInfo);
		goto Exit;
	}
	bPassed = TRUE;
	
Exit:

	endTest( bPassed);
	return( rc);
}
	
/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::renameDbTest(
	const char *	pszDestDbName,
	const char *	pszSrcDbName)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bPassed = FALSE;
	char		szTest [200];
	
	// FlmDbRename will rename a database and all of its files

	f_sprintf( szTest, "Rename Database Test (%s --> %s)",
		pszSrcDbName, pszDestDbName);
	beginTest( szTest);

	if( RC_BAD( rc = FlmDbRename( pszSrcDbName, NULL, NULL,
										 pszDestDbName, TRUE, NULL, NULL)))
	{
		MAKE_ERROR_STRING( "calling FlmDbRename", rc, m_szFailInfo);
		goto Exit;
	}
	bPassed = TRUE;
	
Exit:

	endTest( bPassed);
	return( rc);
}
	
/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::reduceSizeTest(
	const char *	pszDbName)
{
	RCODE			rc = FERR_OK;
	HFDB			hDb = HFDB_NULL;
	FLMBOOL		bPassed = FALSE;
	FLMBOOL		bTransActive = FALSE;
	char			szTest [200];
	FlmRecord *	pRecord = NULL;
	FLMUINT		uiDrn;
	FLMUINT		uiCount;
	
	f_sprintf( szTest, "Reduce Size Test (%s)", pszDbName);
	beginTest( szTest);
	
	if( RC_BAD( rc = FlmDbOpen( pszDbName, NULL, NULL, 0, NULL, &hDb)))
	{
		MAKE_ERROR_STRING( "calling FlmDbOpen", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Start a transaction and attempt to do the remove - should fail.

	if( RC_BAD( rc = FlmDbTransBegin( hDb, FLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransBegin", rc, m_szFailInfo);
		goto Exit;
	}
	bTransActive = TRUE;
	
	// Delete all of the records in the default data container.
	
	uiDrn = 1;
	for (;;)
	{
		if (RC_BAD( rc = FlmRecordRetrieve( hDb, FLM_DATA_CONTAINER, uiDrn,
						FO_INCL, &pRecord, &uiDrn)))
		{
			if (rc == FERR_EOF_HIT)
			{
				rc = FERR_OK;
				break;
			}
			else
			{
				MAKE_ERROR_STRING( "calling FlmRecordRetrieve", rc, m_szFailInfo);
				goto Exit;
			}
		}
		
		// Delete the record.
		
		if (RC_BAD( rc = FlmRecordDelete( hDb, FLM_DATA_CONTAINER, uiDrn, 0)))
		{
			MAKE_ERROR_STRING( "calling FlmRecordDelete", rc, m_szFailInfo);
			goto Exit;
		}
		
		// Go to the next record
		
		uiDrn++;
	}
	
	// Commit the transaction.

	bTransActive = FALSE;	
	if (RC_BAD( rc = FlmDbTransCommit( hDb)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransCommit", rc, m_szFailInfo);
		goto Exit;
	}
	
	if( RC_BAD( rc = FlmDbTransBegin( hDb, FLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransBegin", rc, m_szFailInfo);
		goto Exit;
	}
	bTransActive = TRUE;
	
	// Attempt to call reduce inside the transaction - should fail.
	
	if (RC_BAD( rc = FlmDbReduceSize( hDb, 0, &uiCount)))
	{
		if (rc == FERR_TRANS_ACTIVE)
		{
			rc = FERR_OK;
		}
		else
		{
			MAKE_ERROR_STRING( "calling FlmDbReduceSize", rc, m_szFailInfo);
			goto Exit;
		}
	}
	else
	{
		rc = RC_SET( FERR_FAILURE);
		f_strcpy( m_szFailInfo,
			"Should not be able to call FlmDbReduceSize inside of a transaction!");
		goto Exit;
	}
	
	bTransActive = FALSE;	
	if (RC_BAD( rc = FlmDbTransAbort( hDb)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransAbort", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Attempt the reduce again - should succeed this time.
	
	if (RC_BAD( rc = FlmDbReduceSize( hDb, 0, &uiCount)))
	{
		MAKE_ERROR_STRING( "calling FlmDbReduceSize", rc, m_szFailInfo);
		goto Exit;
	}
	bPassed = TRUE;
	
Exit:

	if (pRecord)
	{
		pRecord->Release();
	}

	if (bTransActive)
	{
		(void)FlmDbTransAbort( hDb);
	}

	if (hDb != HFDB_NULL)
	{
		(void)FlmDbClose( &hDb);
	}

	endTest( bPassed);
	return( rc);
}
	
/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::removeDbTest(
	const char *	pszDbName)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bPassed = FALSE;
	char		szTest [200];
	
	// FlmDbRemove will delete the database and all of its files

	f_sprintf( szTest, "Remove Database Test (%s)", pszDbName);
	beginTest( szTest);

	if( RC_BAD( rc = FlmDbRemove( pszDbName, NULL, NULL, TRUE)))
	{
		MAKE_ERROR_STRING( "calling FlmDbRemove", rc, m_szFailInfo);
		goto Exit;
	}
	bPassed = TRUE;
	
Exit:

	endTest( bPassed);
	return( rc);
}
	
/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::execute( void)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiDrn;
	FLMUINT	uiIndex;

	// Initialize the FLAIM database engine.  This call
	// must be made once by the application prior to making any
	// other FLAIM calls

	if( RC_BAD( rc = FlmStartup()))
	{
		goto Exit;
	}

	// Create database test
	
	if (RC_BAD( rc = createDbTest()))
	{
		goto Exit;
	}
	
	// FlmRecordAdd test
	
	if (RC_BAD( rc = addRecordTest( &uiDrn)))
	{
		goto Exit;
	}
	
	// FlmRecordModify test
	
	if (RC_BAD( rc = modifyRecordTest( uiDrn)))
	{
		goto Exit;
	}
	
	// Large field test
	
	if (RC_BAD( rc = largeFieldTest()))
	{
		goto Exit;
	}
	
	// Retrieve record and query tests
	
	if (RC_BAD( rc = queryRecordTest()))
	{
		goto Exit;
	}

	// FlmRecordDelete test
	
	if (RC_BAD( rc = deleteRecordTest( uiDrn)))
	{
		goto Exit;
	}
	
	// FlmKeyRetrieve test
	
	if (RC_BAD( rc = keyRetrieveTest( LAST_NAME_FIRST_NAME_IX, TRUE)))
	{
		goto Exit;
	}
	
	// Add index test
	
	if (RC_BAD( rc = addIndexTest( &uiIndex)))
	{
		goto Exit;
	}

	// FlmKeyRetrieve test
	
	if (RC_BAD( rc = keyRetrieveTest( uiIndex, FALSE)))
	{
		goto Exit;
	}
	
	// Suspend index test
	
	if (RC_BAD( rc = suspendIndexTest( uiIndex)))
	{
		goto Exit;
	}

	// Resume index test
	
	if (RC_BAD( rc = resumeIndexTest( uiIndex)))
	{
		goto Exit;
	}

	// Delete index test
	
	if (RC_BAD( rc = deleteIndexTest( uiIndex)))
	{
		goto Exit;
	}
	
	// Add sub-string index test
	
	if (RC_BAD( rc = addSubstringIndexTest( &uiIndex)))
	{
		goto Exit;
	}
	
	// Add presence index test
	
	if (RC_BAD( rc = addPresenceIndexTest( &uiIndex)))
	{
		goto Exit;
	}
	
	// Delete field test
	
	if (RC_BAD( rc = deleteFieldTest( AGE_TAG)))
	{
		goto Exit;
	}
	
	// Sorted field test

	if (RC_BAD( rc = sortedFieldsTest( &uiDrn)))
	{
		goto Exit;
	}
	
	// Sorted field query test - first time using rooted field path.
	// Second time not using rooted field path.
	
	if (RC_BAD( rc = sortedFieldsQueryTest( uiDrn, TRUE)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = sortedFieldsQueryTest( uiDrn, FALSE)))
	{
		goto Exit;
	}
	
	// Numbers test
	
	if (RC_BAD( rc = numbersTest( &uiDrn)))
	{
		goto Exit;
	}
	
	// Close and reopen the database to force everything to be written to
	// disk and to have things removed from cache.
	
	if (RC_BAD( rc = reopenDbTest()))
	{
		goto Exit;
	}
	
	// Numbers query test.
	
	if (RC_BAD( rc = numbersRetrieveTest( uiDrn)))
	{
		goto Exit;
	}
	
	// Numbers key retrieve test.
	
	if (RC_BAD( rc = numbersKeyRetrieveTest()))
	{
		goto Exit;
	}
	
	// Numbers query retrieve test.
	
	if (RC_BAD( rc = numbersQueryTest()))
	{
		goto Exit;
	}
	
	// Hot Backup/Restore test
	
	if (RC_BAD( rc = backupRestoreDbTest()))
	{
		goto Exit;
	}
	
	// Compare the restored database to the current database
	
	if (RC_BAD( rc = compareDbTest( DB_NAME_STR, DB_RESTORE_NAME_STR)))
	{
		goto Exit;
	}
	
	// Close the database

	FlmDbClose( &m_hDb);
	
	// Check database test
	
	if (RC_BAD( rc = checkDbTest( DB_NAME_STR)))
	{
		goto Exit;
	}
	
	// Rebuild database test
	
	if (RC_BAD( rc = rebuildDbTest( DB_REBUILD_NAME_STR, DB_NAME_STR)))
	{
		goto Exit;
	}
	
	// Copy database test
	
	if (RC_BAD( rc = copyDbTest( DB_COPY_NAME_STR, DB_NAME_STR)))
	{
		goto Exit;
	}
	
	// Compare the restored database to the copied database
	
	if (RC_BAD( rc = compareDbTest( DB_COPY_NAME_STR, DB_RESTORE_NAME_STR)))
	{
		goto Exit;
	}
	
	// Rename database test
	
	if (RC_BAD( rc = renameDbTest( DB_RENAME_NAME_STR, DB_COPY_NAME_STR)))
	{
		goto Exit;
	}
	
	// Reduce size test
	
	if (RC_BAD( rc = reduceSizeTest( DB_RENAME_NAME_STR)))
	{
		goto Exit;
	}

	// Remove database test
	
	if (RC_BAD( rc = removeDbTest( DB_RENAME_NAME_STR)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = removeDbTest( DB_NAME_STR)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = removeDbTest( DB_RESTORE_NAME_STR)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = removeDbTest( DB_REBUILD_NAME_STR)))
	{
		goto Exit;
	}

Exit:

	FlmShutdown();
	return( rc);
}
