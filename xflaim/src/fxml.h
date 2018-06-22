//------------------------------------------------------------------------------
// Desc:	This file contains the FLAIM XML wrapper class
// Tabs:	3
//
// Copyright (c) 1999-2000, 2002-2007 Novell, Inc. All Rights Reserved.
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

#ifndef FXML_H
#define FXML_H

/*============================================================================
Desc: 	FLAIM's XML namespace class
============================================================================*/
class F_XMLNamespace : public F_Object
{
public:

	FINLINE F_XMLNamespace()
	{
		m_puzPrefix = NULL;
		m_puzURI = NULL;
		m_pNext = NULL;
	}

	FINLINE ~F_XMLNamespace()
	{
		flmAssert( !m_pNext);

		if( m_puzPrefix)
		{
			f_free( &m_puzPrefix);
		}

		if( m_puzURI)
		{
			f_free( &m_puzURI);
		}
	}

	RCODE setPrefix(
		FLMUNICODE *		puzPrefix);

	RCODE setURI(
		FLMUNICODE *		puzURI);

	RCODE setup(
		FLMUNICODE *		puzPrefix,
		FLMUNICODE *		puzURI,
		F_XMLNamespace *	pNext);

	FINLINE FLMUNICODE * getPrefixPtr( void)
	{
		return( m_puzPrefix);
	}

	FINLINE FLMUNICODE * getURIPtr( void)
	{
		return( m_puzURI);
	}

private:

	FLMUNICODE *		m_puzPrefix;
	FLMUNICODE *		m_puzURI;
	F_XMLNamespace *	m_pNext;

friend class F_XMLNamespaceMgr;
};

/*============================================================================
Desc: 	Namespace manager class
============================================================================*/
class F_XMLNamespaceMgr : public F_Object
{
public:

	F_XMLNamespaceMgr();

	~F_XMLNamespaceMgr();

	RCODE findNamespace(
		FLMUNICODE *		puzPrefix,
		F_XMLNamespace **	ppNamespace,
		FLMUINT				uiMaxSearchSize = ~((FLMUINT)0));

	RCODE pushNamespace(
		FLMUNICODE *		puzPrefix,
		FLMUNICODE *		puzNamespaceURI);

	RCODE pushNamespace(
		F_XMLNamespace *	pNamespace);

	void popNamespaces(
		FLMUINT				uiCount);

	FLMUINT getNamespaceCount( void)
	{
		return( m_uiNamespaceCount);
	}

private:

	F_XMLNamespace *			m_pFirstNamespace;
	FLMUINT						m_uiNamespaceCount;
};

// Typedefs

typedef enum
{
	XML_STATS
} eXMLStatus;

// This callback is currently only used by the non-com utilities,
// which is why we haven't bothered to make it an interface
typedef RCODE (* XML_STATUS_HOOK)(
	eXMLStatus		eStatusType,
	void *			pvArg1,
	void *			pvArg2,
	void *			pvArg3,
	void *			pvUserData);

/*============================================================================
Desc: 	FLAIM's XML import class
============================================================================*/
class F_XMLImport : public F_XMLNamespaceMgr
{
public:

	F_XMLImport();

	~F_XMLImport();

	RCODE setup( void);

	void reset( void);

	RCODE import(
		IF_IStream *				pStream,
		F_Db *						pDb,
		FLMUINT						uiCollection,
		FLMUINT						uiFlags,
		F_DOMNode *					pNodeToLinkTo,
		eNodeInsertLoc				eInsertLoc,
		F_DOMNode **				ppNewNode,
		XFLM_IMPORT_STATS *		pImportStats);

	FINLINE void setStatusCallback(
		XML_STATUS_HOOK			fnStatus,
		void *						pvUserData)
	{
		m_fnStatus = fnStatus;
		m_pvCallbackData = pvUserData;
	}

private:

	#define F_DEFAULT_NS_DECL		0x01
	#define F_PREFIXED_NS_DECL		0x02

	typedef struct xmlattr
	{
		FLMUINT				uiLineNum;
		FLMUINT				uiLineOffset;
		FLMUINT				uiLineFilePos;	
		FLMUINT				uiLineBytes;
		FLMUINT				uiValueLineNum;
		FLMUINT				uiValueLineOffset;
		FLMUNICODE *		puzPrefix;
		FLMUNICODE *		puzLocalName;
		FLMUNICODE *		puzVal;
		FLMUINT				uiFlags;
		xmlattr *			pPrev;
		xmlattr *			pNext;
	} XML_ATTR;

	// Methods

	RCODE getFieldTagAndType(
		FLMUNICODE *	puzName,
		FLMBOOL			bOkToAdd,
		FLMUINT *		puiTagNum,
		FLMUINT *		puiDataType);

	RCODE getByte(
		FLMBYTE *	pucByte);
		
	FINLINE void ungetByte(
		FLMBYTE	ucByte)
	{
		// Can only unget a single byte.
		
		flmAssert( !m_ucUngetByte);
		m_ucUngetByte = ucByte;
		m_importStats.uiChars--;
	}
		
	RCODE getLine( void);
	
	FINLINE FLMUNICODE getChar( void)
	{
		if (m_uiCurrLineOffset == m_uiCurrLineNumChars)
		{
			return( (FLMUNICODE)0);
		}
		else
		{
			FLMUNICODE	uzChar = m_puzCurrLineBuf [m_uiCurrLineOffset++];
			return( uzChar);
		}
	}
	
	FINLINE FLMUNICODE peekChar( void)
	{
		if (m_uiCurrLineOffset == m_uiCurrLineNumChars)
		{
			return( (FLMUNICODE)0);
		}
		else
		{
			return( m_puzCurrLineBuf [m_uiCurrLineOffset]);
		}
	}
	
	FINLINE void ungetChar( void)
	{
		
		// There should never be a reason to unget past the beginning of the current
		// line.
		
		flmAssert( m_uiCurrLineOffset);
		m_uiCurrLineOffset--;
	}
	
	RCODE getName(
		FLMUINT *		puiChars);

	RCODE getQualifiedName(
		FLMUINT *		puiChars,
		FLMUNICODE **	ppuzPrefix,
		FLMUNICODE **	ppuzLocal,
		FLMBOOL *		pbNamespaceDecl,
		FLMBOOL *		pbDefaultNamespaceDecl);

	void getNmtoken(
		FLMUINT *		puiChars);

	RCODE getPubidLiteral( void);

	RCODE getSystemLiteral( void);

	RCODE getElementValue(
		FLMUNICODE *	puBuf,
		FLMUINT *		puiMaxChars,
		FLMBOOL *		pbEntity);

	RCODE processEntityValue( void);

	RCODE getEntity(
		FLMUNICODE *	puBuf,
		FLMUINT *		puiChars,
		FLMBOOL *		pbTranslated,
		FLMUNICODE *	puTransChar);

	RCODE processReference(
		FLMUNICODE *	puChar = NULL);

	RCODE processCDATA(
		F_DOMNode *		pParent,
		FLMUINT			uiSavedLineNum,
		FLMUINT     	uiSavedOffset,
		FLMUINT			uiSavedFilePos,
		FLMUINT			uiSavedLineBytes);

	RCODE processAttributeList( void);

	RCODE processComment(
		F_DOMNode *		pParent,
		FLMUINT			uiSavedLineNum,
		FLMUINT     	uiSavedOffset,
		FLMUINT			uiSavedFilePos,
		FLMUINT			uiSavedLineBytes);

	RCODE processProlog( void);

	RCODE processXMLDecl( void);

	RCODE processVersion( void);

	RCODE processEncodingDecl( void);

	RCODE processSDDecl( void);

	RCODE processMisc( void);

	RCODE processDocTypeDecl( void);

	RCODE processPI(
		F_DOMNode *		pParent,
		FLMUINT			uiSavedLineNum,
		FLMUINT     	uiSavedOffset,
		FLMUINT			uiSavedFilePos,
		FLMUINT			uiSavedLineBytes);

	RCODE processElement(
		F_DOMNode *			pNodeToLinkTo,
		eNodeInsertLoc		eInsertLoc,
		F_DOMNode **		ppNewNode);

	RCODE unicodeToNumber64(
		FLMUNICODE *		puzVal,
		FLMUINT64 *			pui64Val,
		FLMBOOL *			pbNeg);

	RCODE flushElementValue(
		F_DOMNode *			pParent,
		FLMBYTE *			pucValue,
		FLMUINT				uiValueLen);

	RCODE getBinaryVal(
		FLMUINT *			puiLength);

	RCODE fixNamingTag(
		F_DOMNode *			pNode);

	FLMBOOL lineHasToken(
		const char *	pszToken);
		
	RCODE processMarkupDecl( void);

	RCODE processPERef( void);

	RCODE processElementDecl( void);

	RCODE processEntityDecl( void);

	RCODE processNotationDecl( void);

	RCODE processAttListDecl( void);

	RCODE processContentSpec( void);

	RCODE processMixedContent( void);

	RCODE processChildContent( void);

	RCODE processAttDef( void);

	RCODE processAttType( void);

	RCODE processAttValue(
		XML_ATTR *	pAttr);

	RCODE processDefaultDecl( void);

	RCODE processID(
		FLMBOOL	bPublicId);

	RCODE processSTag(
		F_DOMNode *			pNodeToLinkTo,	
		eNodeInsertLoc		eInsertLoc,
		FLMBOOL *			pbHasContent,
		F_DOMNode **		ppElement);

	RCODE skipWhitespace(
		FLMBOOL	bRequired);

	RCODE resizeValBuffer(
		FLMUINT			uiSize);

	// Attribute management

	void resetAttrList( void)
	{
		m_pFirstAttr = NULL;
		m_pLastAttr = NULL;

		m_attrPool.poolReset( NULL);
	}

	RCODE allocAttribute(
		XML_ATTR **		ppAttr)
	{
		XML_ATTR *	pAttr = NULL;
		RCODE			rc = NE_XFLM_OK;

		if( RC_BAD( rc = m_attrPool.poolCalloc( 
			sizeof( XML_ATTR), (void **)&pAttr)))
		{
			goto Exit;
		}

		if( (pAttr->pPrev = m_pLastAttr) == NULL)
		{
			m_pFirstAttr = pAttr;
		}
		else
		{
			m_pLastAttr->pNext = pAttr;
		}

		m_pLastAttr = pAttr;

	Exit:

		*ppAttr = pAttr;
		return( rc);
	}

	RCODE setPrefix(
		XML_ATTR *		pAttr,
		FLMUNICODE *	puzPrefix)
	{
		RCODE		rc = NE_XFLM_OK;
		FLMUINT	uiStrLen;

		if( !puzPrefix)
		{
			pAttr->puzPrefix = NULL;
			goto Exit;
		}

		uiStrLen = f_unilen( puzPrefix);

		if( RC_BAD( rc = m_attrPool.poolAlloc( 
			sizeof( FLMUNICODE) * (uiStrLen + 1), (void **)&pAttr->puzPrefix)))
		{
			goto Exit;
		}

		f_memcpy( pAttr->puzPrefix, puzPrefix, 
			sizeof( FLMUNICODE) * (uiStrLen + 1));

	Exit:

		return( rc);
	}

	RCODE setLocalName(
		XML_ATTR *		pAttr,
		FLMUNICODE *	puzLocalName)
	{
		RCODE		rc = NE_XFLM_OK;
		FLMUINT	uiStrLen;

		if( !puzLocalName)
		{
			pAttr->puzLocalName = NULL;
			goto Exit;
		}

		uiStrLen = f_unilen( puzLocalName);

		if( RC_BAD( rc = m_attrPool.poolAlloc( 
			sizeof( FLMUNICODE) * (uiStrLen + 1), 
			(void **)&pAttr->puzLocalName)))
		{
			goto Exit;
		}

		f_memcpy( pAttr->puzLocalName, puzLocalName,
			sizeof( FLMUNICODE) * (uiStrLen + 1));

	Exit:

		return( rc);
	}

	RCODE setUnicode(
		XML_ATTR	*		pAttr,
		FLMUNICODE *	puzUnicode)
	{
		RCODE		rc = NE_XFLM_OK;
		FLMUINT	uiStrLen;

		if( !puzUnicode)
		{
			pAttr->puzVal = NULL;
			goto Exit;
		}

		uiStrLen = f_unilen( puzUnicode);

		if( RC_BAD( rc = m_attrPool.poolAlloc( 
			sizeof( FLMUNICODE) * (uiStrLen + 1), 
			(void **)&pAttr->puzVal)))
		{
			goto Exit;
		}

		f_memcpy( pAttr->puzVal, puzUnicode, 
			sizeof( FLMUNICODE) * (uiStrLen + 1));

	Exit:

		return( rc);
	}

	RCODE addAttributesToElement(
		F_DOMNode *			pElement);
		
	FINLINE void setErrInfo(
		FLMUINT			uiErrLineNum,
		FLMUINT			uiErrLineOffset,
		XMLParseError	eErrorType,
		FLMUINT			uiErrLineFilePos,
		FLMUINT			uiErrLineBytes)
	{
		m_importStats.uiErrLineNum = uiErrLineNum;
		m_importStats.uiErrLineOffset = uiErrLineOffset;
		m_importStats.eErrorType = eErrorType;
		m_importStats.uiErrLineFilePos = uiErrLineFilePos;
		m_importStats.uiErrLineBytes = uiErrLineBytes;
	}

	// Data

	F_Db *						m_pDb;
	FLMUINT						m_uiCollection;
	FLMBYTE						m_ucUngetByte;
	FLMUNICODE *				m_puzCurrLineBuf;
	FLMUINT						m_uiCurrLineBufMaxChars;
	FLMUINT						m_uiCurrLineNumChars;
	FLMUINT						m_uiCurrLineOffset;
	FLMUINT						m_uiCurrLineNum;
	FLMUINT						m_uiCurrLineFilePos;
	FLMUINT						m_uiCurrLineBytes;
#define FLM_XML_MAX_CHARS		128
	FLMUNICODE					m_uChars[ FLM_XML_MAX_CHARS];
	FLMBOOL						m_bSetup;
	IF_IStream *				m_pStream;
	FLMBYTE *					m_pucValBuf;
	FLMUINT						m_uiValBufSize; // Number of Unicode characters
	FLMUINT						m_uiFlags;
	FLMBOOL						m_bExtendDictionary;
	XMLEncoding					m_eXMLEncoding;
	XML_STATUS_HOOK			m_fnStatus;
	void *						m_pvCallbackData;
	XFLM_IMPORT_STATS			m_importStats;
	F_Pool						m_tmpPool;

	// Attribute management

	XML_ATTR *					m_pFirstAttr;
	XML_ATTR *					m_pLastAttr;
	F_Pool 						m_attrPool;
};

#define FLM_XML_EXTEND_DICT_FLAG				0x00000001
#define FLM_XML_COMPRESS_WHITESPACE_FLAG	0x00000002
#define FLM_XML_TRANSLATE_ESC_FLAG			0x00000004

FINLINE FLMBOOL isXMLNS(
	FLMUNICODE *	puzName)
{
	return( (puzName [0] == FLM_UNICODE_x || puzName [0] == FLM_UNICODE_X) &&
			  (puzName [1] == FLM_UNICODE_m || puzName [1] == FLM_UNICODE_M) &&
			  (puzName [2] == FLM_UNICODE_l || puzName [2] == FLM_UNICODE_L) &&
			  (puzName [3] == FLM_UNICODE_n || puzName [3] == FLM_UNICODE_N) &&
			  (puzName [4] == FLM_UNICODE_s || puzName [4] == FLM_UNICODE_S)
			  ? TRUE
			  : FALSE);
}

#endif // FXML_H
