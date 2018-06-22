//-------------------------------------------------------------------------
// Desc:	Command line argument parser for utilities - definitions.
// Tabs:	3
//
// Copyright (c) 2001, 2003, 2005-2007 Novell, Inc. All Rights Reserved.
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

#ifndef FLMARG_H
#define FLMARG_H

#include "sharutil.h"

typedef FLMBOOL (* FLMARG_VALIDATOR) (
	const char *	pszGivenArg,
	const char *	pszIdentifier,
	FlmStringAcc *	pOutputAccumulator,
	void *			pvUserData);

typedef void (* FLMARG_OUTPUT_CALLBACK) (
	const char *	pszOutputString,
	void *			pvUserData);

typedef void (* FLMARG_PRESS_ANY_KEY_CALLBACK) (
	const char *	pszPressAnyKeyMessage,
	void *			pvUserData);

typedef enum 
{
	FLMARG_OPTION = 0,
	FLMARG_REQUIRED_ARG,
	FLMARG_OPTIONAL_ARG,
	FLMARG_REPEATING_ARG
} FLMARG_TYPE;

typedef enum 
{
	FLMARG_CONTENT_NONE = 0,
	FLMARG_CONTENT_BOOL,
	FLMARG_CONTENT_VALIDATOR,				//use ,validator, userdata for ...
	FLMARG_CONTENT_SIGNED_INT,				//use ,MIN, MAX) for ... 
	FLMARG_CONTENT_UNSIGNED_INT,			//use ,MIN, MAX) for ... 
	FLMARG_CONTENT_ALLOWED_STRING_SET,	//use ,"foo","bar",NULL) for ...
	FLMARG_CONTENT_EXISTING_FILE,
	FLMARG_CONTENT_STRING
} FLMARG_CONTENT_TYPE;

class FlmArgSet;

class FlmArg : public F_Object
{
private:
	FlmArg(
		const char *			pszIdentifier,
		const char *			pszShortHelp,
		FLMBOOL					bCaseSensitive,
		FLMARG_TYPE				argType,
		FLMARG_CONTENT_TYPE	contentType)		
	{
		m_pszIdentifier = pszIdentifier;
		m_pszShortHelp = pszShortHelp;
		m_bCaseSensitive = bCaseSensitive;
		m_argType = argType;
		m_contentType = contentType;
			
		m_uiValueCount = 0;
		m_bIsPresent = FALSE;

		//setting optional values to smart initial values
		m_validator = NULL;
		m_uiMin = 0xFFFFFFFF;
		m_uiMax = 0xFFFFFFFF;
		m_iMin = -1;
		m_iMax = -1;
		m_uiStringSetCount = 0;
	}
	
	~FlmArg()
	{
		FLMUINT 		uiKill;
		char * 		pszStr;

		for ( uiKill = 0; uiKill < m_uiValueCount; uiKill++)
		{
			pszStr = (char *)(m_valuesVec.getElementAt( uiKill));
			
			if( pszStr)
			{
				f_free( &pszStr);
			}
		}
	}

	const char * getIdentifier()
	{
		return( m_pszIdentifier);
	}
	
	FLMBOOL isPresent()
	{
		return( m_bIsPresent);
	}
	
	FLMUINT getValueCount()
	{
		return( m_uiValueCount);
	}
	
	FLMBOOL getCaseSensitive()
	{
		return( m_bCaseSensitive);
	}
	
	const char * getShortHelp()
	{
		return( m_pszShortHelp);
	}
	
	FLMARG_TYPE getArgType()
	{
		return( m_argType);
	}
	
	FLMARG_CONTENT_TYPE getContentType()
	{
		return( m_contentType);
	}
	
	FLMARG_VALIDATOR getValidator()
	{
		return( m_validator);
	}
	
	void * getValidatorData()
	{
		return( m_pvValidatorData);
	}
	
	FlmVector * getStringSet()
	{
		return( &m_stringSet);
	}
	
	FLMUINT getStringSetLen()
	{
		return( m_uiStringSetCount);
	}
	
	void getMinMax(
		FLMUINT *		puiMin,
		FLMUINT * 		puiMax);
		
	void getMinMax( 
		FLMINT * 		puiMin,
		FLMINT * 		puiMax);

	FLMUINT getUINT(
		FLMUINT 			uiIndex);
		
	FLMINT getINT(
		FLMUINT			uiIndex);
		
	FLMBOOL getBOOL(
		FLMUINT			uiIndex);
		
	const char * getString(
		FLMUINT			uiIndex);
		
	void getString(
		char *			pszDestination,
		FLMUINT			uiDestinationBufferSize,
		FLMUINT			uiIndex);

	void setPresent()
	{
		m_bIsPresent = TRUE;
	}
	
	RCODE addValue( 
		const char *	pszVal);
		
	const char * getValue(
		FLMUINT			uiIndex);
		
	void setValidator( 
		FLMARG_VALIDATOR 		validator,
		void *					pvValidatorData)
	{
		m_validator = validator;
		m_pvValidatorData = pvValidatorData;
	}
	
	void setMinMax(
		FLMUINT			uiMin,
		FLMUINT 			uiMax)
	{
		m_uiMin = uiMin;
		m_uiMax = uiMax;
	}
	
	void setMinMax( FLMINT iMin, FLMINT iMax)
	{
		m_iMin = iMin;
		m_iMax = iMax;
	}
	
	RCODE addToStringSet(
		const char *	pszStr);

	const char *			m_pszIdentifier;
	const char *			m_pszShortHelp;
	FLMBOOL					m_bCaseSensitive;
	FLMARG_TYPE				m_argType;
	FLMARG_CONTENT_TYPE	m_contentType;
	FlmVector				m_valuesVec;
	FLMUINT					m_uiValueCount;
	FLMBOOL					m_bIsPresent;
	FLMARG_VALIDATOR		m_validator;
	void *					m_pvValidatorData;
	FLMUINT					m_uiMin;
	FLMUINT					m_uiMax;
	FLMINT					m_iMin;
	FLMINT					m_iMax;
	FlmVector				m_stringSet;
	FLMUINT					m_uiStringSetCount;
	
friend class FlmArgSet;
};

class FlmArgSet : public F_Object
{
public:
	FlmArgSet(
		char *									pszDescription,
		FLMARG_OUTPUT_CALLBACK				outputCallback,
		void *									pvOutputCallbackData,
		FLMARG_PRESS_ANY_KEY_CALLBACK		pressAnyKeyCallback,
		void *									pvPressAnyKeyCallbackData,
		FLMUINT									uiLinesPerScreen);
		
	virtual ~FlmArgSet();
	
	const char * getDescription( void)
	{
		return m_pszDescription;
	}

	RCODE addArg(
		const char *			pszIdentifier,
		const char *			pszShortHelp,
		FLMBOOL					bCaseSensitive,
		FLMARG_TYPE				argType,
		FLMARG_CONTENT_TYPE	contentType,
		...); 

	RCODE parseCommandLine(
		FLMUINT 					uiArgc,
		const char ** 			ppszArgv,
		FLMBOOL * 				pbPrintedUsage);
		
	FLMBOOL argIsPresent( 
		const char * 			pszIdentifier)
	{
		return this->getFlmArg( pszIdentifier)->isPresent();
	}
	
	FLMUINT getValueCount( 
		const char * 			pszIdentifier)
	{
		return this->getFlmArg( pszIdentifier)->getValueCount();
	}

	FLMUINT getUINT( 
		const char * 			pszIdentifier,
		FLMUINT 					uiIndex = 0)
	{
		return this->getFlmArg( pszIdentifier)->getUINT( uiIndex);
	}
	
	FLMINT getINT( 
		const char *			pszIdentifier,
		FLMUINT 					uiIndex = 0)
	{
		return this->getFlmArg( pszIdentifier)->getINT( uiIndex);
	}

	/*
		will recognize the following formats in the following order:
		TRUE				FALSE			NOTES
		----				-----			-----
		true				false			case-insensitive
		1					0
		on					off			case-insensitive
		yes				no				case-insensitive
		NULL								case-insensitive
		*									anything else is a user error
	*/
	
	FLMBOOL getBOOL(
		const char *			pszIdentifier,
		FLMUINT					uiIndex = 0)
	{
		return this->getFlmArg( pszIdentifier)->getBOOL( uiIndex);
	}

	const char * getString(
		const char * 			pszIdentifier,
		FLMUINT 					uiIndex = 0)
	{
		return this->getFlmArg( pszIdentifier)->getString( uiIndex);
	}
	
	void getString( 
		const char *			pszIdentifier,
		char *					pszDestination,
		FLMUINT 					uiDestinationBufferSize,
		FLMUINT 					uiIndex = 0)
	{
		this->getFlmArg( pszIdentifier)->getString( pszDestination,
			uiDestinationBufferSize, uiIndex);
	}

private:

	FlmArg * getFlmArg(
		const char *			pszIdentifier);
	
	RCODE printUsage( void);
	
	RCODE dump( 
		FlmVector *				pVec,
		FLMUINT					uiVecLen);
	
	void outputLines(
		const char * 			pszStr);
		
	FLMBOOL needsPreprocessing( void);
	
	RCODE preProcessParams( void);
	
	RCODE processAtParams( 
		FLMUINT					uiInsertionPoint,
		char * 					pszBuffer);
		
	RCODE displayShortHelpLines(
		FlmStringAcc *			pStringAcc,
		const char *			pszShortHelp,
		FLMUINT					uiCharsPerLine);
		
	FLMBOOL needMoreArgs( 
		FlmVector * 			pVec,
		FLMUINT 					uiVecLen);
		
	RCODE parseOption( 
		const char *			pszArg,
		FLMBOOL *				pbPrintedUsage);

	char *				m_pszDescription;
	char					m_szExecBaseName[ F_PATH_MAX_SIZE];
	FlmVector			m_flmArgVec;
	FLMUINT				m_uiFlmArgVecIndex;
	FlmVector			m_optionsVec;
	FLMUINT				m_uiOptionsVecLen;
	FlmVector			m_requiredArgsVec;
	FLMUINT				m_uiRequiredArgsVecLen;
	FlmVector			m_optionalArgsVec;
	FLMUINT				m_uiOptionalArgsVecLen;
	FlmArg *				m_pRepeatingArg;
	FLMUINT				m_uiArgc;
	FlmVector *			m_pArgv;
	FLMARG_OUTPUT_CALLBACK				m_outputCallback;
	void *									m_pvOutputCallbackData;
	FLMARG_PRESS_ANY_KEY_CALLBACK		m_pressAnyKeyCallback;
	void *									m_pvPressAnyKeyCallbackData;
	FLMUINT				m_uiOutputLines;
	FLMUINT				m_uiLinesPerScreen;
};

#endif
