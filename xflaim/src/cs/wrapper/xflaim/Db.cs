//------------------------------------------------------------------------------
// Desc:	Db Class
// Tabs:	3
//
// Copyright (c) 2006-2007 Novell, Inc. All Rights Reserved.
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

using System;
using System.Runtime.InteropServices;

namespace xflaim
{

	/// <summary>
	/// Predefined collections found in all XFLAIM databases.
	/// </summary>
	public enum PredefinedXFlaimCollections : uint
	{
		/// <summary>
		/// Maintenance collection
		/// </summary>
		XFLM_MAINT_COLLECTION		= 65533,
		/// <summary>
		/// Default data collection.
		/// </summary>
		XFLM_DATA_COLLECTION			= 65534,
		/// <summary>
		/// Dictionary collection.
		/// </summary>
		XFLM_DICT_COLLECTION			= 65535
	}

	/// <summary>
	/// Predefined indexes found in all XFLAIM databases.
	/// </summary>
	public enum PredefinedXFlaimIndexes : uint
	{
		/// <summary>
		/// Index on dictionary numbers
		/// </summary>
		XFLM_DICT_NUMBER_INDEX		= 65534,
		/// <summary>
		/// Index in dictionary names.
		/// </summary>
		XFLM_DICT_NAME_INDEX			= 65535
	}

//-----------------------------------------------------------------------------
// Element tags
//-----------------------------------------------------------------------------

	// IMPORTANT NOTE: These need to be kept in sync with the corresponding
	// definitions in xflaim.h
	/// <summary>
	/// Reserved dictionary tags for elements
	/// </summary>
	public enum ReservedElmTag : uint
	{
		/// <summary>
		/// First reserved element number.
		/// </summary>
		XFLM_FIRST_RESERVED_ELEMENT_TAG = 0xFFFFFE00,
		/// <summary>
		/// "element"
		/// </summary>
		ELM_ELEMENT_TAG = 0xFFFFFE00,
		/// <summary>
		/// "attribute"
		/// </summary>
		ELM_ATTRIBUTE_TAG = 0xFFFFFE01,
		/// <summary>
		/// "Index"
		/// </summary>
		ELM_INDEX_TAG = 0xFFFFFE02,
		/// <summary>
		/// "ElementComponent"
		/// </summary>
		ELM_ELEMENT_COMPONENT_TAG = 0xFFFFFE04,
		/// <summary>
		/// "AttributeComponent"
		/// </summary>
		ELM_ATTRIBUTE_COMPONENT_TAG = 0xFFFFFE05,
		/// <summary>
		/// "Collection"
		/// </summary>
		ELM_COLLECTION_TAG = 0xFFFFFE06,
		/// <summary>
		/// "Prefix"
		/// </summary>
		ELM_PREFIX_TAG = 0xFFFFFE07,
		/// <summary>
		/// "NextDictNums"
		/// </summary>
		ELM_NEXT_DICT_NUMS_TAG = 0xFFFFFE08,
		/// <summary>
		/// "DocumentTitle"
		/// </summary>
		ELM_DOCUMENT_TITLE_TAG = 0xFFFFFE09,
		/// <summary>
		/// "Invalid"
		/// </summary>
		ELM_INVALID_TAG = 0xFFFFFE0A,
		/// <summary>
		/// "Quarantined"
		/// </summary>
		ELM_QUARANTINED_TAG = 0xFFFFFE0B,
		/// <summary>
		/// "All"
		/// </summary>
		ELM_ALL_TAG = 0xFFFFFE0C,
		/// <summary>
		/// "Annotation"
		/// </summary>
		ELM_ANNOTATION_TAG = 0xFFFFFE0D,
		/// <summary>
		/// "Any"
		/// </summary>
		ELM_ANY_TAG = 0xFFFFFE0E,
		/// <summary>
		/// "AttributeGroup"
		/// </summary>
		ELM_ATTRIBUTE_GROUP_TAG = 0xFFFFFE0F,
		/// <summary>
		/// "Choice"
		/// </summary>
		ELM_CHOICE_TAG = 0xFFFFFE10,
		/// <summary>
		/// "ComplexContent"
		/// </summary>
		ELM_COMPLEX_CONTENT_TAG = 0xFFFFFE11,
		/// <summary>
		/// "ComplexType"
		/// </summary>
		ELM_COMPLEX_TYPE_TAG = 0xFFFFFE12,
		/// <summary>
		/// "Documentation"
		/// </summary>
		FLM_DOCUMENTATION_TAG = 0xFFFFFE13,
		/// <summary>
		/// "enumeration"
		/// </summary>
		ELM_ENUMERATION_TAG = 0xFFFFFE14,
		/// <summary>
		/// "extension"
		/// </summary>
		ELM_EXTENSION_TAG = 0xFFFFFE15,
		/// <summary>
		/// "Delete"
		/// </summary>
		ELM_DELETE_TAG = 0xFFFFFE16,
		/// <summary>
		/// "BlockChain"
		/// </summary>
		ELM_BLOCK_CHAIN_TAG = 0xFFFFFE17,
		/// <summary>
		/// "EncDef"
		/// </summary>
		ELM_ENCDEF_TAG = 0xFFFFFE18,
		/// <summary>
		/// "Sweep"
		/// </summary>
		ELM_SWEEP_TAG = 0xFFFFFE19,
		/// <summary>
		/// Last reserved element number
		/// </summary>
		XFLM_LAST_RESERVED_ELEMENT_TAG = 0xFFFFFE19
}

//-----------------------------------------------------------------------------
// Attribute tags
//-----------------------------------------------------------------------------

	// IMPORTANT NOTE: These need to be kept in sync with the corresponding
	// definitions in xflaim.h
	/// <summary>
	/// Reserved dictionary tags for attributes
	/// </summary>
	public enum ReservedAttrTag : uint
	{
		/// <summary>
		/// First reserved attribute number.
		/// </summary>
		XFLM_FIRST_RESERVED_ATTRIBUTE_TAG = 0xFFFFFE00,
		/// <summary>
		/// "DictNumber"
		/// </summary>
		ATTR_DICT_NUMBER_TAG = 0xFFFFFE00,
		/// <summary>
		/// "CollectionNumber"
		/// </summary>
		ATTR_COLLECTION_NUMBER_TAG = 0xFFFFFE01,
		/// <summary>
		/// "CollectionName"
		/// </summary>
		ATTR_COLLECTION_NAME_TAG = 0xFFFFFE02,
		/// <summary>
		/// "name"
		/// </summary>
		ATTR_NAME_TAG = 0xFFFFFE03,
		/// <summary>
		/// "targetNameSpace"
		/// </summary>
		ATTR_TARGET_NAMESPACE_TAG = 0xFFFFFE04,
		/// <summary>
		/// "type"
		/// </summary>
		ATTR_TYPE_TAG = 0xFFFFFE05,
		/// <summary>
		/// "State"
		/// </summary>
		ATTR_STATE_TAG = 0xFFFFFE06,
		/// <summary>
		/// "Language"
		/// </summary>
		ATTR_LANGUAGE_TAG = 0xFFFFFE07,
		/// <summary>
		/// "IndexOptions"
		/// </summary>
		ATTR_INDEX_OPTIONS_TAG = 0xFFFFFE08,
		/// <summary>
		/// "IndexOn"
		/// </summary>
		ATTR_INDEX_ON_TAG = 0xFFFFFE09,
		/// <summary>
		/// "Required"
		/// </summary>
		ATTR_REQUIRED_TAG = 0xFFFFFE0A,
		/// <summary>
		/// "Limit"
		/// </summary>
		ATTR_LIMIT_TAG = 0xFFFFFE0B,
		/// <summary>
		/// "CompareRules"
		/// </summary>
		ATTR_COMPARE_RULES_TAG = 0xFFFFFE0C,
		/// <summary>
		/// "KeyComponent"
		/// </summary>
		ATTR_KEY_COMPONENT_TAG = 0xFFFFFE0D,
		/// <summary>
		/// "DataComponent"
		/// </summary>
		ATTR_DATA_COMPONENT_TAG = 0xFFFFFE0E,
		/// <summary>
		/// "LastDocumentIndexed"
		/// </summary>
		ATTR_LAST_DOC_INDEXED_TAG = 0xFFFFFE0F,
		/// <summary>
		/// "NextElementNum"
		/// </summary>
		ATTR_NEXT_ELEMENT_NUM_TAG = 0xFFFFFE10,
		/// <summary>
		/// "NextAttributeNum"
		/// </summary>
		ATTR_NEXT_ATTRIBUTE_NUM_TAG = 0xFFFFFE11,
		/// <summary>
		/// "NextIndexNum"
		/// </summary>
		ATTR_NEXT_INDEX_NUM_TAG = 0xFFFFFE12,
		/// <summary>
		/// "NextCollectionNum"
		/// </summary>
		ATTR_NEXT_COLLECTION_NUM_TAG = 0xFFFFFE13,
		/// <summary>
		/// "NextPrefixNum"
		/// </summary>
		ATTR_NEXT_PREFIX_NUM_TAG = 0xFFFFFE14,
		/// <summary>
		/// "Source"
		/// </summary>
		ATTR_SOURCE_TAG = 0xFFFFFE15,
		/// <summary>
		/// "StateChangeCount"
		/// </summary>
		ATTR_STATE_CHANGE_COUNT_TAG = 0xFFFFFE16,
		/// <summary>
		/// "xmlns"
		/// </summary>
		ATTR_XMLNS_TAG = 0xFFFFFE17,
		/// <summary>
		/// "abstract"
		/// </summary>
		ATTR_ABSTRACT_TAG = 0xFFFFFE18,
		/// <summary>
		/// "base"
		/// </summary>
		ATTR_BASE_TAG = 0xFFFFFE19,
		/// <summary>
		/// "block"
		/// </summary>
		ATTR_BLOCK_TAG = 0xFFFFFE1A,
		/// <summary>
		/// "default"
		/// </summary>
		ATTR_DEFAULT_TAG = 0xFFFFFE1B,
		/// <summary>
		/// "final"
		/// </summary>
		ATTR_FINAL_TAG = 0xFFFFFE1C,
		/// <summary>
		/// "fixed"
		/// </summary>
		ATTR_FIXED_TAG = 0xFFFFFE1D,
		/// <summary>
		/// "itemtype"
		/// </summary>
		ATTR_ITEM_TYPE_TAG = 0xFFFFFE1E,
		/// <summary>
		/// "membertypes"
		/// </summary>
		ATTR_MEMBER_TYPES_TAG = 0xFFFFFE1F,
		/// <summary>
		/// "mixed"
		/// </summary>
		ATTR_MIXED_TAG = 0xFFFFFE20,
		/// <summary>
		/// "nillable"
		/// </summary>
		ATTR_NILLABLE_TAG = 0xFFFFFE21,
		/// <summary>
		/// "ref"
		/// </summary>
		ATTR_REF_TAG = 0xFFFFFE22,
		/// <summary>
		/// "use"
		/// </summary>
		ATTR_USE_TAG = 0xFFFFFE23,
		/// <summary>
		/// "value"
		/// </summary>
		ATTR_VALUE_TAG = 0xFFFFFE24,
		/// <summary>
		/// "address"
		/// </summary>
		ATTR_ADDRESS_TAG = 0xFFFFFE25,
		/// <summary>
		/// "xmlns:xflaim"
		/// </summary>
		ATTR_XMLNS_XFLAIM_TAG = 0xFFFFFE26,
		/// <summary>
		/// "Key"
		/// </summary>
		ATTR_ENCRYPTION_KEY_TAG = 0xFFFFFE27,
		/// <summary>
		/// "Transaction"
		/// </summary>
		ATTR_TRANSACTION_TAG = 0xFFFFFE28,
		/// <summary>
		/// "NextEncDefNum"
		/// </summary>
		ATTR_NEXT_ENCDEF_NUM_TAG = 0xFFFFFE29,
		/// <summary>
		/// "encId"
		/// </summary>
		ATTR_ENCRYPTION_ID_TAG = 0xFFFFFE2A,
		/// <summary>
		/// "keySize"
		/// </summary>
		ATTR_ENCRYPTION_KEY_SIZE_TAG = 0xFFFFFE2B,
		/// <summary>
		/// "UniqueSubElements"
		/// </summary>
		ATTR_UNIQUE_SUB_ELEMENTS_TAG = 0xFFFFFE2C,
		/// <summary>
		/// Last reserved attribute number
		/// </summary>
		XFLM_LAST_RESERVED_ATTRIBUTE_TAG = 0xFFFFFE2C
	}

//-----------------------------------------------------------------------------
// Encryption schemes
//-----------------------------------------------------------------------------

	/// <summary>
	/// Encryption schemes.
	/// </summary>
	public enum EncryptionScheme
	{
		/// <summary>AES 128 bit</summary>
		ENC_AES128	= 1,
		/// <summary>AES 192 bit</summary>
		ENC_AES192	= 2,
		/// <summary>AES 256 bit</summary>
		ENC_AES256	= 3,
		/// <summary>DES3 (168 bi)t</summary>
		ENC_DES3		= 4
	}

//-----------------------------------------------------------------------------
// Export format types.
//-----------------------------------------------------------------------------

	/// <summary>
	/// Format types for exporting XML from an XFLAIM database.
	/// </summary>
	public enum eExportFormatType : uint
	{
		/// <summary>No special formatting</summary>
		XFLM_EXPORT_NO_FORMAT =			0x00,
		/// <summary>Output a new line for each element</summary>
		XFLM_EXPORT_NEW_LINE =			0x01,
		/// <summary>
		/// Output a new line for each element and indent
		/// elements according to nesting level
		/// </summary>
		XFLM_EXPORT_INDENT =				0x02,
		/// <summary>
		/// Output a new line for each element and indent
		/// elements according to nesting level.  Also indent
		/// data for elements.
		/// </summary>
		XFLM_EXPORT_INDENT_DATA =		0x03
	}

//-----------------------------------------------------------------------------
// Change states
//-----------------------------------------------------------------------------

	/// <summary>
	/// Change states for definitions in the dictionary.
	/// </summary>
	public enum ChangeState
	{
		/// <summary>Check the definition to see if it is in use</summary>
		STATE_CHECKING = 1,
		/// <summary>Purge the definition after purging all uses of it</summary>
		STATE_PURGE		= 2,
		/// <summary>Definition is in use.</summary>
		STATE_ACTIVE	= 3
	}

//-----------------------------------------------------------------------------
// XML parse errors
//-----------------------------------------------------------------------------

	// IMPORTANT NOTE: These need to be kept in sync with the corresponding
	// definitions in xflaim.h
	/// <summary>
	/// Parse errors that can occur when importing XML into a database.
	/// </summary>
	public enum XMLParseError : uint
	{
		/// <summary>
		/// No error
		/// </summary>
		XML_NO_ERROR = 0,
		/// <summary>
		/// Invalid element name - does not start with a valid 
		/// character for element names
		/// </summary>
		XML_ERR_BAD_ELEMENT_NAME,
		/// <summary>
		/// Element names cannot be "xmlns" or have "xmlns:" as a prefix
		/// </summary>
		XML_ERR_XMLNS_IN_ELEMENT_NAME,
		/// <summary>
		/// The element begin and end tags do not match
		/// </summary>
		XML_ERR_ELEMENT_NAME_MISMATCH,
		/// <summary>
		/// The prefix for the element or attribute has not been defined with 
		/// an "xmlns:prefix=" attribute somewhere
		/// </summary>
		XML_ERR_PREFIX_NOT_DEFINED,
		/// <summary>
		/// Expecting a right angle bracket
		/// </summary>
		XML_ERR_EXPECTING_GT,
		/// <summary>
		/// Expecting a left angle bracket to begin an element name
		/// </summary>
		XML_ERR_EXPECTING_ELEMENT_LT,
		/// <summary>
		/// Expecting a '=' after the attribute name
		/// </summary>
		XML_ERR_EXPECTING_EQ,
		/// <summary>
		/// Multiple "xmlns" default namespace declarations in an element
		/// </summary>
		XML_ERR_MULTIPLE_XMLNS_DECLS,
		/// <summary>
		/// Multiple definitions for the same prefix ("xmlns:prefix=...") in an element
		/// </summary>
		XML_ERR_MULTIPLE_PREFIX_DECLS,
		/// <summary>
		/// Invalid xml declaration terminator
		/// </summary>
		XML_ERR_EXPECTING_QUEST_GT,
		/// <summary>
		/// Invalid XML markup
		/// </summary>
		XML_ERR_INVALID_XML_MARKUP,
		/// <summary>
		/// Must have at least one attr def in an ATTRLIST markup
		/// </summary>
		XML_ERR_MUST_HAVE_ONE_ATT_DEF,
		/// <summary>
		/// Expecting "NDATA" keyword
		/// </summary>
		XML_ERR_EXPECTING_NDATA,
		/// <summary>
		/// Expecting "SYSTEM" or "PUBLIC" keyword in NOTATION declaration
		/// </summary>
		XML_ERR_EXPECTING_SYSTEM_OR_PUBLIC,
		/// <summary>
		/// Expecting "("
		/// </summary>
		XML_ERR_EXPECTING_LPAREN,
		/// <summary>
		/// Expecing ")" or "|"
		/// </summary>
		XML_ERR_EXPECTING_RPAREN_OR_PIPE,
		/// <summary>
		/// Expecting a name
		/// </summary>
		XML_ERR_EXPECTING_NAME,
		/// <summary>
		/// Invalid Attr type in ATTLIST
		/// </summary>
		XML_ERR_INVALID_ATT_TYPE,
		/// <summary>
		/// Invalid default decl, expecting #FIXED, #REQUIRED, #IMPLIED, or quoted attr value
		/// </summary>
		XML_ERR_INVALID_DEFAULT_DECL,
		/// <summary>
		/// Expecting PCDATA - only PCDATA allowed after #
		/// </summary>
		XML_ERR_EXPECTING_PCDATA,
		/// <summary>
		/// Expecting "*"
		/// </summary>
		XML_ERR_EXPECTING_ASTERISK,
		/// <summary>
		/// Empty content is invalid - must be parameters between parens
		/// </summary>
		XML_ERR_EMPTY_CONTENT_INVALID,
		/// <summary>
		/// Cannot mix choice items with sequenced items.
		/// </summary>
		XML_ERR_CANNOT_MIX_CHOICE_AND_SEQ,
		/// <summary>
		/// "XML" is not a legal name for a processing instruction
		/// </summary>
		XML_ERR_XML_ILLEGAL_PI_NAME,
		/// <summary>
		/// Illegal first character in name - must be an alphabetic letter or underscore
		/// </summary>
		XML_ERR_ILLEGAL_FIRST_NAME_CHAR,
		/// <summary>
		/// Illegal second ":" found in name.  Name already has a colon.
		/// </summary>
		XML_ERR_ILLEGAL_COLON_IN_NAME,
		/// <summary>
		/// Expecting "version"
		/// </summary>
		XML_ERR_EXPECTING_VERSION,
		/// <summary>
		/// Invalid version number - only 1.0 is supported.
		/// </summary>
		XML_ERR_INVALID_VERSION_NUM,
		/// <summary>
		/// Unsupported encoding - must be "UTF-8" or "us-ascii"
		/// </summary>
		XML_ERR_UNSUPPORTED_ENCODING,
		/// <summary>
		/// Expecting "yes" or "no"
		/// </summary>
		XML_ERR_EXPECTING_YES_OR_NO,
		/// <summary>
		/// Expecting quote character - unexpected end of line
		/// </summary>
		XML_ERR_EXPECTING_QUOTE_BEFORE_EOL,
		/// <summary>
		/// Expecting ";"
		/// </summary>
		XML_ERR_EXPECTING_SEMI,
		/// <summary>
		/// Unexpected end of line in entity reference, need proper 
		/// terminating character - ";"
		/// </summary>
		XML_ERR_UNEXPECTED_EOL_IN_ENTITY,
		/// <summary>
		/// Invalid numeric character entity.  Number is either too large, or zero,
		/// or illegal characters were used in the number.
		/// </summary>
		XML_ERR_INVALID_CHARACTER_NUMBER,
		/// <summary>
		/// Unsupported predefined entity reference.
		/// </summary>
		XML_ERR_UNSUPPORTED_ENTITY,
		/// <summary>
		/// Expecting single or double quote character.
		/// </summary>
		XML_ERR_EXPECTING_QUOTE,
		/// <summary>
		/// Invalid character in public id.
		/// </summary>
		XML_ERR_INVALID_PUBLIC_ID_CHAR,
		/// <summary>
		/// Whitespace required
		/// </summary>
		XML_ERR_EXPECTING_WHITESPACE,
		/// <summary>
		/// Expecting HEX digit for binary value
		/// </summary>
		XML_ERR_EXPECTING_HEX_DIGIT,
		/// <summary>
		/// Invalid binary value for attribute
		/// </summary>
		XML_ERR_INVALID_BINARY_ATTR_VALUE,
		/// <summary>
		/// Error returned from createNode in processCDATA
		/// </summary>
		XML_ERR_CREATING_CDATA_NODE,
		/// <summary>
		/// Error returned from createNode in processComment
		/// </summary>
		XML_ERR_CREATING_COMMENT_NODE,
		/// <summary>
		/// Error returned from createNode in processPI
		/// </summary>
		XML_ERR_CREATING_PI_NODE,
		/// <summary>
		/// Error returned from createNode in processPI
		/// </summary>
		XML_ERR_CREATING_DATA_NODE,
		/// <summary>
		/// Error returned from createRootElement in processSTag
		/// </summary>
		XML_ERR_CREATING_ROOT_ELEMENT,
		/// <summary>
		/// Error returned from createNode in processSTag
		/// </summary>
		XML_ERR_CREATING_ELEMENT_NODE
	}
	
//-----------------------------------------------------------------------------
// XML encoding
//-----------------------------------------------------------------------------

	// IMPORTANT NOTE: These need to be kept in sync with the corresponding
	// definitions in xflaim.h
	/// <summary>
	/// Types of XML encoding
	/// </summary>
	public enum XMLEncoding : uint
	{
		/// <summary>
		/// UTF-8 encoding
		/// </summary>
		XFLM_XML_UTF8_ENCODING,
		/// <summary>
		/// US ASCII encoding
		/// </summary>
		XFLM_XML_USASCII_ENCODING
	}

//-----------------------------------------------------------------------------
// XML import stats
//-----------------------------------------------------------------------------

	// IMPORTANT NOTE: This structure needs to be kept in sync with the corresponding
	// definitions in Db.cpp - CS_XFLM_IMPORT_STATS.  CS_XFLM_IMPORT_STATS is
	// designed to correspond to XFLM_IMPORT_STATS in xflaim.h, but it cannot be
	// exactly the same because the enums will be a different size in C++ code.
	/// <summary>
	/// XML import statistics
	/// </summary>
	[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
	public class CS_XFLM_IMPORT_STATS
	{
		/// <summary>
		/// Lines processed
		/// </summary>
		public uint					uiLines;
		/// <summary>
		/// Characters processed
		/// </summary>
		public uint					uiChars;
		/// <summary>
		/// Attributes processed
		/// </summary>
		public uint					uiAttributes;
		/// <summary>
		/// Elements processed
		/// </summary>
		public uint					uiElements;
		/// <summary>
		/// Text nodes processed
		/// </summary>
		public uint					uiText;
		/// <summary>
		/// Documents processed
		/// </summary>
		public uint					uiDocuments;
		/// <summary>
		/// Line number where the parser encountered an error
		/// </summary>
		public uint					uiErrLineNum;
		/// <summary>
		/// Offset in the line where the parser encountered an error
		/// </summary>
		public uint					uiErrLineOffset;
		/// <summary>
		/// Type of error encountered
		/// </summary>
		public XMLParseError		eErrorType;
		/// <summary>
		/// Offset in the stream where the line containing the error starts
		/// </summary>
		public uint					uiErrLineFilePos;
		/// <summary>
		/// Offset in the stream where the error starts
		/// </summary>
		public uint					uiErrLineBytes;
		/// <summary>
		/// XML encoding (UTF-8, etc.)
		/// </summary>
		public XMLEncoding		eXMLEncoding;
	}

//-----------------------------------------------------------------------------
// Database transaction types
//-----------------------------------------------------------------------------

	// IMPORTANT NOTE: These need to be kept in sync with the corresponding
	// definitions in xflaim.h
	/// <summary>
	/// Database transaction types.
	/// </summary>
	public enum eDbTransType : uint
	{
		/// <summary>No transaction</summary>
		XFLM_NO_TRANS = 0,
		/// <summary>Read transaction</summary>
		XFLM_READ_TRANS,
		/// <summary>Update transaction</summary>
		XFLM_UPDATE_TRANS
	}

//-----------------------------------------------------------------------------
// Database transaction flags
//-----------------------------------------------------------------------------

	// IMPORTANT NOTE: These need to be kept in sync with the corresponding
	// definitions in xflaim.h
	/// <summary>
	/// Database transaction flags.
	/// </summary>
	[Flags]
	public enum DbTransFlags : uint
	{
		/// <summary>
		/// Do not terminate the transaction, even if
		/// a checkpoint is waiting to complete
		/// </summary>
		XFLM_DONT_KILL_TRANS = 0x0001,
		/// <summary>
		/// Place all blocks and nodes read during the transaction
		/// at the least-recently used positions in the cache lists.
		/// </summary>
		XFLM_DONT_POISON_CACHE = 0x0002
	}

//-----------------------------------------------------------------------------
// Database lock types
//-----------------------------------------------------------------------------

	// IMPORTANT NOTE: These need to be kept in sync with the corresponding
	// enum in ftk.h
	/// <summary>
	/// Types of locks that may be requested.
	/// </summary>
	public enum eLockType : uint
	{
		/// <summary>No lock</summary>
		FLM_LOCK_NONE = 0,
		/// <summary>Exclusive lock</summary>
		FLM_LOCK_EXCLUSIVE,
		/// <summary>Shared lock</summary>
		FLM_LOCK_SHARED
	}

	/// <summary>
	/// This object contains information about a lock holder or lock waiter.
	/// </summary>
	public class LockUser
	{
		/// <summary>
		/// Thread ID that is either holding the lock or waiting for it.
		/// </summary>
		public uint	uiThreadId;
		/// <summary>
		/// If this represents the lock holder, this is the amount of time
		/// that the thread has been holding the lock.  If it represents
		/// the lock waiter, it is the amount of time the thread has been
		/// waiting to obtain the lock.  Time is in milliseconds.
		/// </summary>
		public uint	uiTime;
	}

	// IMPORTANT NOTE: These need to be kept in sync with the corresponding
	// definitions in xflaim.h
	/// <summary>
	/// Types of locks that may be requested.
	/// </summary>
	public enum eXFlmIndexState : uint
	{
		/// <summary>Index is on-line and available for use.</summary>
		XFLM_INDEX_ONLINE = 0,
		/// <summary>Index is being built and is unavailable.</summary>
		XFLM_INDEX_BRINGING_ONLINE,
		/// <summary>Index has been suspended and is unavailable.</summary>
		XFLM_INDEX_SUSPENDED
	}

	// IMPORTANT NOTE: This structure needs to be kept in sync with the corresponding
	// definitions in xflaim.h.  It is almost exactly the same as the XFLM_INDEX_STATUS
	// structure, except that we cannot guarantee the size of eState.  In C# it is always
	// a 32 bit number.  It may not be that in C++.  That is the reason we have a
	// different structure in C# than we have in C++.
	/// <summary>
	/// Index status information.
	/// </summary>
	[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
	public class CS_XFLM_INDEX_STATUS
	{
		/// <summary>
		/// If ~0 then index is online, otherwise this is the value of the 
		/// last document ID that was indexed.
		/// </summary>
		public ulong  				ulLastDocumentIndexed;
		/// <summary>
		/// Keys processed by the background indexing thread.
		/// </summary>
		public ulong				ulKeysProcessed;
		/// <summary>
		/// Documents processed by the background indexing thread.
		/// </summary>
		public ulong				ulDocumentsProcessed;
		/// <summary>
		/// Number of transactions completed by the background indexing thread.
		/// </summary>
		public ulong				ulTransactions;
		/// <summary>
		/// ID of the index.
		/// </summary>
		public uint					uiIndexNum;
		/// <summary>
		/// Time the bacground indexing thread (if any) was started.
		/// </summary>
		public uint					uiStartTime;
		/// <summary>
		/// State of the background indexing thread (if any).
		/// </summary>
		public eXFlmIndexState	eState;
	}

//-----------------------------------------------------------------------------
// RetrieveFlags
//-----------------------------------------------------------------------------

	// IMPORTANT NOTE: These flags need to be kept in sync with the corresponding
	// definitions in xflaim.h
	/// <summary>
	/// Flags used to specify items to be retrieved from a result set.
	/// The <see cref="Db.keyRetrieve"/> method also uses these flags
	/// to specify how keys from an index are to be retrieved.
	/// </summary>
	[Flags]
	public enum RetrieveFlags : uint
	{
		/// <summary>Return item greater than or equal to the search key.</summary>
		XFLM_INCL			= 0x0010,
		/// <summary>Return item greater than the search key.</summary>
		XFLM_EXCL			= 0x0020,
		/// <summary>Return item that exactly matches the search key.</summary>
		XFLM_EXACT			= 0x0040,
		/// <summary>
		/// Used in conjunction with XFLM_EXCL.  Specifies that the item to be
		/// returned must match the key components, but the node ids may be
		/// different.
		/// </summary>
		XFLM_KEY_EXACT		= 0x0080,
		/// <summary>Retrieve the first key in the index or first item in a result set.</summary>
		XFLM_FIRST			= 0x0100,
		/// <summary>Retrieve the last key in the index or last item in a result set.</summary>
		XFLM_LAST			= 0x0200,
		/// <summary>Specifies whether to match node IDs in the search key.</summary>
		XFLM_MATCH_IDS		= 0x0400,
		/// <summary>Specifies whether to match the document ID in the search key.</summary>
		XFLM_MATCH_DOC_ID = 0x0800
	}

	// IMPORTANT NOTE: These need to be kept in sync with the corresponding
	// defines in xflaim.h.
	/// <summary>
	/// Reason checkpoint thread is forcing a checkpoint.
	/// </summary>
	public enum eCPReason : uint
	{
		/// <summary>
		/// The checkpoint interval (typically 180 seconds) has elapsed.
		/// </summary>
		XFLM_CP_TIME_INTERVAL_REASON = 1,
		/// <summary>
		/// The database is being closed.
		/// </summary>
		XFLM_CP_SHUTTING_DOWN_REASON = 2,
		/// <summary>
		/// A problem was encountered whil writing to the RFL volume.
		/// </summary>
		XFLM_CP_RFL_VOLUME_PROBLEM = 3
	}

	// IMPORTANT NOTE: This structure needs to be kept in sync with the corresponding
	// definitions in xflaim.h
	/// <summary>
	/// Checkpoint information.
	/// </summary>
	[StructLayout(LayoutKind.Sequential, Pack = 1, CharSet = CharSet.Ansi)]
	public class XFLM_CHECKPOINT_INFO
	{
		/// <summary>
		/// If true, the checkpoint is currently running
		/// </summary>
		public int					bRunning;
		/// <summary>
		/// Amount of time (seconds) that the checkpoint has been running
		/// </summary>
		public uint					uiRunningTime;
		/// <summary>
		/// If true, the checkpoint is being forced and cannot be interrupted by
		/// a foreground update transaction.
		/// </summary>
		public int					bForcingCheckpoint;
		/// <summary>
		/// Amount of time (seconds) that the checkpoint has been running
		/// in "forced" mode.
		/// </summary>
		public uint					uiForceCheckpointRunningTime;
		/// <summary>
		/// Specific reason for forcing a checkpoint.
		/// </summary>
		public eCPReason			forceCheckpointReason;
		/// <summary>
		/// If true, the checkpoint thread is currently writing data blocks
		/// </summary>
		public int					bWritingDataBlocks;
		/// <summary>
		/// Number of log blocks written to the roll-back log
		/// </summary>
		public uint					uiLogBlocksWritten;
		/// <summary>
		/// Number of data blocks written
		/// </summary>
		public uint					uiDataBlocksWritten;
		/// <summary>
		/// Amount of dirty cache
		/// </summary>
		public uint					uiDirtyCacheBytes;
		/// <summary>
		/// Database block size
		/// </summary>
		public uint					uiBlockSize;
		/// <summary>
		/// Amount of time the checkpoint has waited to truncate
		/// the roll-back log
		/// </summary>
		public uint					uiWaitTruncateTime;
	};

	/// <summary>
	/// The Db class provides a number of methods that allow C#
	/// applications to access an XFLAIM database.  A Db object
	/// is obtained by calling <see cref="DbSystem.dbCreate"/> or
	/// <see cref="DbSystem.dbOpen"/>
	/// </summary>
	public class Db
	{
		private IntPtr		m_pDb;			// Pointer to IF_Db object in unmanaged space
		private DbSystem 	m_dbSystem;

//-----------------------------------------------------------------------------
// constructor
//-----------------------------------------------------------------------------

		/// <summary>
		/// Db constructor.
		/// </summary>
		/// <param name="pDb">
		/// Reference to an IF_Db object.
		/// </param>
		/// <param name="dbSystem">
		/// DbSystem object that this Db object is associated with.
		/// </param>
		internal Db(
			IntPtr	pDb,
			DbSystem	dbSystem)
		{
			if (pDb == IntPtr.Zero)
			{
				throw new XFlaimException( "Invalid IF_Db reference");
			}
			
			m_pDb = pDb;

			if (dbSystem == null)
			{
				throw new XFlaimException( "Invalid DbSystem reference");
			}
			
			m_dbSystem = dbSystem;
			
			// Must call something inside of DbSystem.  Otherwise, the
			// m_dbSystem object gets a compiler warning on linux because
			// it is not used anywhere.  Other than that, there is really
			// no need to make the following call.
			if (m_dbSystem.getDbSystem() == IntPtr.Zero)
			{
				throw new XFlaimException( "Invalid DbSystem.IF_DbSystem object");
			}
		}

//-----------------------------------------------------------------------------
// destructor
//-----------------------------------------------------------------------------

		/// <summary>
		/// Destructor.
		/// </summary>
		~Db()
		{
			close();
		}

//-----------------------------------------------------------------------------
// getDb
//-----------------------------------------------------------------------------

		/// <summary>
		/// Return the pointer to the IF_Db object.
		/// </summary>
		/// <returns>Returns a pointer to the IF_Db object.</returns>
		internal IntPtr getDb()
		{
			return( m_pDb);
		}

//-----------------------------------------------------------------------------
// getDbSystem
//-----------------------------------------------------------------------------

		/// <summary>
		/// Return the DbSystem object associated with this Db
		/// </summary>
		/// <returns>Returns the DbSystem object associated with this Db</returns>
		internal DbSystem getDbSystem()
		{
			return m_dbSystem;
		}

//-----------------------------------------------------------------------------
// close
//-----------------------------------------------------------------------------

		/// <summary>
		/// Close this database.
		/// </summary>
		public void close()
		{
			// Release the native pDb!
		
			if (m_pDb != IntPtr.Zero)
			{
				xflaim_Db_Release( m_pDb);
				m_pDb = IntPtr.Zero;
			}
		
			// Remove our reference to the dbSystem so it can be released.
		
			m_dbSystem = null;
		}

		[DllImport("xflaim")]
		private static extern void xflaim_Db_Release(
			IntPtr	pDb);

//-----------------------------------------------------------------------------
// transBegin
//-----------------------------------------------------------------------------

		/// <summary>
		/// Starts a transaction.
		/// </summary>
		/// <param name="eTransType">
		/// The type of transaction (<see cref="eDbTransType"/>)
		/// </param>
		/// <param name="uiMaxLockWait">
		/// Specifies the amount of time to wait for lock requests occuring
		/// during the transaction to be granted.  Valid values are 0 through
		/// 255 seconds.  Zero is used to specify no-wait locks.  255 specifies
		/// that there is no timeout.
		/// </param>
		/// <param name="uiFlags">
		/// Should be a logical OR'd combination of the members of
		/// <see cref="DbTransFlags"/>
		/// </param>
		/// <returns></returns>
		public void transBegin(
			eDbTransType	eTransType,
			uint				uiMaxLockWait,
			DbTransFlags	uiFlags)
		{
			RCODE				rc;

			if ((rc = xflaim_Db_transBegin( m_pDb,
				eTransType, uiMaxLockWait, uiFlags)) != 0)
			{
				throw new XFlaimException(rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_transBegin(
			IntPtr			pDb,
			eDbTransType	eTransType,
			uint				uiMaxLockWait,
			DbTransFlags	uiFlags);

//-----------------------------------------------------------------------------
// transBegin
//-----------------------------------------------------------------------------

		/// <summary>
		/// Starts a transaction.  Transaction will be of the same type and same
		/// snapshot as the passed in Db object.  The passed in Db object should
		/// be running a read transaction.
		/// </summary>
		/// <param name="db">
		/// Database whose transaction is to be copied.
		/// </param>
		/// <returns></returns>
		public void transBegin(
			Db	db)
		{
			RCODE	rc;

			if ((rc = xflaim_Db_transBeginClone( m_pDb, db.getDb())) != 0)
			{
				throw new XFlaimException(rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_transBeginClone(
			IntPtr	pDb,
			IntPtr	pDbToClone);

//-----------------------------------------------------------------------------
// transCommit
//-----------------------------------------------------------------------------

		/// <summary>
		/// Commits an active transaction.  If no transaction is running, or the
		/// transaction commit fails, an exception will be thrown.
		/// </summary>
		/// <returns></returns>
		public void transCommit()
		{
			RCODE				rc;

			if ((rc = xflaim_Db_transCommit( m_pDb)) != 0)
			{
				throw new XFlaimException(rc);
			}
		}
		
		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_transCommit(
			IntPtr	pDb);

//-----------------------------------------------------------------------------
// transAbort
//-----------------------------------------------------------------------------

		/// <summary>
		/// Aborts an active transaction.
		/// </summary>
		/// <returns></returns>
		public void transAbort()
		{
			RCODE				rc;

			if ((rc = xflaim_Db_transAbort( m_pDb)) != 0)
			{
				throw new XFlaimException(rc);
			}
		}
		
		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_transAbort(
			IntPtr	pDb);

//-----------------------------------------------------------------------------
// getTransType
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the current transaction type.
		/// </summary>
		/// <returns><see cref="eDbTransType"/></returns>
		public eDbTransType getTransType()
		{
			return( xflaim_Db_getTransType( m_pDb));
		}

		[DllImport("xflaim")]
		private static extern eDbTransType xflaim_Db_getTransType(
			IntPtr	pDb);

//-----------------------------------------------------------------------------
// doCheckpoint
//-----------------------------------------------------------------------------

		/// <summary>
		/// Perform a checkpoint on the database.
		/// </summary>
		/// <param name="uiTimeout">
		/// Specifies the amount of time to wait for database lock.  
		/// Valid values are 0 through 255 seconds.  Zero is used to specify no-wait
		/// locks. 255 is used to specify that there is no timeout.
		/// </param>
		/// <returns></returns>
		public void doCheckpoint(
			uint				uiTimeout)
		{
			RCODE				rc;

			if ((rc = xflaim_Db_doCheckpoint( m_pDb, uiTimeout)) != 0)
			{
				throw new XFlaimException(rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_doCheckpoint(
			IntPtr	pDb,
			uint		uiTimeout);

//-----------------------------------------------------------------------------
// dbLock
//-----------------------------------------------------------------------------

		/// <summary>
		/// Lock the database. 
		/// </summary>
		/// <param name="eLckType">
		/// Type of lock being requested.
		/// </param>
		/// <param name="iPriority">
		/// Priority of lock being requested.
		/// </param>
		/// <param name="uiTimeout">
		/// Lock wait time.  Specifies the amount of time to wait for 
		/// database lock.  Valid values are 0 through 255 seconds.
		/// Zero is used to specify no-wait locks. 255 is used to specify
		/// that there is no timeout.
		/// </param>
		/// <returns></returns>
		public void dbLock(
			eLockType		eLckType,
			int				iPriority,
			uint				uiTimeout)
		{
			RCODE	rc;

			if ((rc = xflaim_Db_dbLock( m_pDb, eLckType, 
				iPriority, uiTimeout)) != 0)
			{
				throw new XFlaimException(rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_dbLock(
			IntPtr		pDb,
			eLockType	eLckType,
			int			iPriority,
			uint			uiTimeout);
	
//-----------------------------------------------------------------------------
// dbUnlock
//-----------------------------------------------------------------------------

		/// <summary>
		/// Unlocks the database.
		/// </summary>
		/// <returns></returns>
		public void dbUnlock()
		{
			RCODE	rc;

			if ((rc = xflaim_Db_dbUnlock( m_pDb)) != 0)
			{
				throw new XFlaimException(rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_dbUnlock(
			IntPtr	pDb);

//-----------------------------------------------------------------------------
// getLockType
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the type of database lock current held.
		/// </summary>
		/// <param name="eLckTyp">
		/// Type of lock is returned here.
		/// </param>
		/// <param name="bImplicit">
		/// Flag indicating whether the database was implicitly locked is 
		/// returned here.  Returns true if implicitly locked, false if
		/// explicitly locked.  Implicit lock means that the database was
		/// locked at the time the transaction was started.  Explicit lock
		/// means that the application called <see cref="dbLock"/> to
		/// obtain the lock.
		/// </param>
		public void getLockType(
			out eLockType	eLckTyp,
			out bool			bImplicit)
		{
			RCODE	rc;
			int	bImpl;

			if ((rc = xflaim_Db_getLockType( m_pDb, out eLckTyp, out bImpl)) != 0)
			{
				throw new XFlaimException( rc);
			}
			bImplicit = bImpl != 0 ? true : false;
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getLockType(
			IntPtr			pDb,
			out eLockType	peLockType,
			out int			pbImplicit);

//-----------------------------------------------------------------------------
// getLockInfo
//-----------------------------------------------------------------------------

		/// <summary>
		/// Return various pieces of lock information.
		/// </summary>
		/// <param name="iPriority">
		/// Priority to look for.  The uiPriorityCount parameter returns a count
		/// of all waiting threads with a lock priority greater than or equal to
		/// this.
		/// </param>
		/// <param name="eLckType">
		/// Returns the type of database lock current held.
		/// </param>
		/// <param name="uiThreadId">
		/// Returns the thread id of the thread that currently holds the database lock.
		/// </param>
		/// <param name="uiNumExclQueued">
		/// Returns the number of threads that are currently waiting to obtain
		/// an exclusive database lock.
		/// </param>
		/// <param name="uiNumSharedQueued">
		/// Returns the number of threads that are currently waiting to obtain
		/// a shared database lock.
		/// </param>
		/// <param name="uiPriorityCount">
		/// Returns the number of threads that are currently waiting to obtain
		/// a database lock whose priority is >= iPriority.
		/// </param>
		public void getLockInfo(
			int				iPriority,
			out eLockType	eLckType,
			out uint			uiThreadId,
			out uint			uiNumExclQueued,
			out uint			uiNumSharedQueued,
			out uint			uiPriorityCount)

		{
			RCODE	rc;
			
			if ((rc = xflaim_Db_getLockInfo( m_pDb, iPriority, out eLckType,
				out uiThreadId, out uiNumExclQueued,
				out uiNumSharedQueued, out uiPriorityCount)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getLockInfo(
			IntPtr			pDb,
			int				iPriority,
			out eLockType	eLckType,
			out uint			uiThreadId,
			out uint			uiNumExclQueued,
			out uint			uiNumSharedQueued,
			out uint			uiPriorityCount);

//-----------------------------------------------------------------------------
// indexSuspend
//-----------------------------------------------------------------------------

		/// <summary>
		/// Suspend indexing on the specified index.
		/// </summary>
		/// <param name="uiIndex">
		/// Index to be suspended.
		/// </param>
		public void indexSuspend(
			uint	uiIndex)
		{
			RCODE			rc;

			if ((rc = xflaim_Db_indexSuspend( m_pDb, uiIndex)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_indexSuspend(
			IntPtr	pDb,
			uint		uiIndex);
	
//-----------------------------------------------------------------------------
// indexResume
//-----------------------------------------------------------------------------

		/// <summary>
		/// Resume indexing on the specified index.
		/// </summary>
		/// <param name="uiIndex">
		/// Index to be resumed.
		/// </param>
		public void indexResume(
			uint	uiIndex)
		{
			RCODE			rc;

			if ((rc = xflaim_Db_indexResume( m_pDb, uiIndex)) != 0)
			{
				throw new XFlaimException(rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_indexResume(
			IntPtr	pDb,
			uint		uiIndex);
	
//-----------------------------------------------------------------------------
// indexGetNext
//-----------------------------------------------------------------------------

		/// <summary>
		/// This method provides a way to iterate through all of the indexes in the
		/// database.  It returns the index ID of the index that comes after the
		/// passed in index number.  The first index can be obtained by passing in a
		/// zero.
		/// </summary>
		/// <param name="uiCurrIndex">
		/// Current index number.  Index that comes after this one
		/// will be returned.	
		/// </param>
		/// <returns>
		/// Returns the index ID of the index that comes after uiCurrIndex.
		/// </returns>
		public uint indexGetNext(
			uint	uiCurrIndex)
		{
			RCODE	rc;

			if ((rc = xflaim_Db_indexGetNext( m_pDb, ref uiCurrIndex)) != 0)
			{
				throw new XFlaimException(rc);
			}

			return( uiCurrIndex);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_indexGetNext(
			IntPtr		pDb,
			ref uint		puiCurrIndex);

//-----------------------------------------------------------------------------
//	indexStatus
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves index status information
		/// </summary>
		/// <param name="uiIndex">
		/// Index whose status is to be returned
		/// </param>
		/// <returns>An instance of a <see cref="CS_XFLM_INDEX_STATUS"/> object.</returns>
		public CS_XFLM_INDEX_STATUS indexStatus(
			uint	uiIndex)
		{
			RCODE							rc;
			CS_XFLM_INDEX_STATUS		indexStatus = new CS_XFLM_INDEX_STATUS();

			if ((rc = xflaim_Db_indexStatus( m_pDb, uiIndex, indexStatus)) != 0)
			{
				throw new XFlaimException(rc);
			}

			return( indexStatus);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_indexStatus(
			IntPtr					pDb,
			uint						uiIndex,
			CS_XFLM_INDEX_STATUS	pIndexStatus);

//-----------------------------------------------------------------------------
// reduceSize
//-----------------------------------------------------------------------------

		/// <summary>
		/// Return unused blocks back to the file system.
		/// </summary>
		/// <param name="uiCount">
		/// Maximum number of blocks to be returned.
		/// </param>
		/// <returns>
		/// Returns the number of blocks that were actually returned to the
		/// file system.
		/// </returns>
		public uint reduceSize(
			uint	uiCount)
		{
			RCODE		rc;
			uint		uiNumReduced;

			if ((rc = xflaim_Db_reduceSize( m_pDb, uiCount, out uiNumReduced)) != 0)
			{
				throw new XFlaimException(rc);
			}

			return( uiNumReduced);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_reduceSize(
			IntPtr		pDb,
			uint			uiCount,
			out uint		uiNumReduced);

//-----------------------------------------------------------------------------
// keyRetrieve
//-----------------------------------------------------------------------------

		/// <summary>
		/// Lookup/retrieve keys in an index. 
		/// </summary>
		/// <param name="uiIndex">
		/// The index that is being searched.
		/// </param>
		/// <param name="searchKey">
		/// The search key use for the search.
		/// </param>
		/// <param name="retrieveFlags">
		/// Search flags <see cref="RetrieveFlags"/>.
		/// </param>
		/// <param name="foundKey">
		/// Data vector where found key will be returned.  If null is passed in
		/// a new data vector will be created.
		/// </param>
		/// <returns>
		/// Key that was retrieved from the index.
		/// </returns>
		public DataVector keyRetrieve(
			uint					uiIndex,
			DataVector			searchKey,
			RetrieveFlags		retrieveFlags,
			DataVector			foundKey)
		{
			RCODE		rc;
			IntPtr	pSearchKey = (searchKey == null ? IntPtr.Zero : searchKey.getDataVector());
			IntPtr	pFoundKey;

			if (foundKey == null)
			{
				foundKey = m_dbSystem.createDataVector();
			}

			pFoundKey = foundKey.getDataVector();

			if ((rc = xflaim_Db_keyRetrieve( m_pDb,
				uiIndex, pSearchKey, retrieveFlags, pFoundKey)) != 0)
			{
				throw new XFlaimException(rc);
			}
			return( foundKey);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_keyRetrieve(
			IntPtr				pDb,
			uint					uiIndex,
			IntPtr				pSearchKey,
			RetrieveFlags		retrieveFlags,
			IntPtr				pFoundKey);

//-----------------------------------------------------------------------------
// createDocument
//-----------------------------------------------------------------------------

		/// <summary>
		/// Creates a new document node. 
		/// </summary>
		/// <param name="uiCollection">
		/// The collection to store the new document in.
		/// </param>
		/// <returns>
		/// An instance of a <see cref="DOMNode"/> object.
		/// </returns>
		 public DOMNode createDocument(
	 		uint	uiCollection)
		{
			RCODE		rc;
			IntPtr	pNewNode = IntPtr.Zero;

			if ((rc = xflaim_Db_createDocument( m_pDb, uiCollection, 
				out pNewNode)) != 0)
			{
				throw new XFlaimException(rc);
			}

			return( new DOMNode( pNewNode, this));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_createDocument(
			IntPtr		pDb,
			uint			uiCollection,
			out IntPtr	ppNewNode);

//-----------------------------------------------------------------------------
// createRootElement
//-----------------------------------------------------------------------------

		/// <summary>
		/// Creates a new root element node.  This is the root node of a document
		/// in the XFLAIM database.
		/// </summary>
		/// <param name="uiCollection">
		/// The collection to store the new node in.
		/// </param>
		/// <param name="uiElementNameId">
		/// Name id of the element to be created.
		/// </param>
		/// <returns>
		/// An instance of a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode createRootElement(
			uint		uiCollection,
			uint		uiElementNameId)
		{
			RCODE		rc;
			IntPtr	pNewNode = IntPtr.Zero;

			if ((rc = xflaim_Db_createRootElement( m_pDb, uiCollection,
				uiElementNameId, out pNewNode)) != 0)
			{
				throw new XFlaimException(rc);
			}

			return( new DOMNode(pNewNode, this));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_createRootElement(
			IntPtr		pDb,
			uint			uiCollection,
			uint			uiElementNameId,
			out IntPtr	ppNewNode);

//-----------------------------------------------------------------------------
// getFirstDocument
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieve the first document in a specified collection. 
		/// </summary>
		/// <param name="uiCollection">
		/// The collection from which to retrieve the first document
		/// </param>
		/// <param name="nodeToReuse">
		/// An existing DOM node object can optionally be passed in.  It will
		/// be reused rather than allocating a new object.
		/// </param>
		/// <returns>
		/// Returns the root <see cref="DOMNode"/> of the document.
		/// </returns>
		public DOMNode getFirstDocument(
			uint			uiCollection,
			DOMNode		nodeToReuse)
		 {
			RCODE		rc;
			IntPtr	pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;
			
			if ((rc = xflaim_Db_getFirstDocument( m_pDb, uiCollection, ref pNode)) != 0)
			{
				throw new XFlaimException(rc);
			}

			if( nodeToReuse != null)
			{
				nodeToReuse.setNodePtr( pNode, this);
				return( nodeToReuse);
			}

			return( new DOMNode( pNode, this));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getFirstDocument(
			IntPtr		pDb,
			uint			uiCollection,
			ref IntPtr	pNode);

//-----------------------------------------------------------------------------
// getLastDocument
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves the last document in a specified collection. 
		/// </summary>
		/// <param name="uiCollection">
		/// The collection from which to retrieve the document
		/// </param>
		/// <param name="nodeToReuse">
		/// An existing DOM node object can optionally be passed in.  It will
		/// be reused rather than allocating a new object.
		/// </param>
		/// <returns>
		/// Returns the root <see cref="DOMNode"/> of the document.
		/// </returns>
		public DOMNode getLastDocument(
			uint		uiCollection,
			DOMNode	nodeToReuse)
		{
			RCODE		rc;
			IntPtr	pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;

			if ((rc = xflaim_Db_getLastDocument( m_pDb, uiCollection, ref pNode)) != 0)
			{
				throw new XFlaimException(rc);
			}

			if (nodeToReuse != null)
			{
				nodeToReuse.setNodePtr(pNode, this);
				return( nodeToReuse);
			}

			return( new DOMNode(pNode, this));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getLastDocument(
			IntPtr		pDb,
			uint			uiCollection,
			ref IntPtr	pNewNode);
 
//-----------------------------------------------------------------------------
// getDocument
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves a document from the specified collection. 
		/// </summary>
		/// <param name="uiCollection">
		/// The collection from which to retrieve the document
		/// </param>
		/// <param name="retrieveFlags">
		/// Search flags <see cref="RetrieveFlags"/>.
		/// </param>
		/// <param name="ulDocumentId">
		/// Document to retrieve.
		/// </param>
		/// <param name="nodeToReuse">
		/// An existing DOM node object can optionally be passed in.  It will
		/// be reused rather than allocating a new object.
		/// </param>
		/// <returns>
		/// Returns the root <see cref="DOMNode"/> of the document.
		/// </returns>
		public DOMNode getDocument(
			uint				uiCollection,
			RetrieveFlags	retrieveFlags,
			ulong				ulDocumentId,
			DOMNode			nodeToReuse)
		{
			RCODE		rc;
			IntPtr	pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;

			if ((rc = xflaim_Db_getDocument( m_pDb, uiCollection,
				retrieveFlags, ulDocumentId, ref pNode)) != 0)
			{
				throw new XFlaimException(rc);
			}

			if (nodeToReuse != null)
			{
				nodeToReuse.setNodePtr(pNode, this);
				return( nodeToReuse);
			}

			return( new DOMNode( pNode, this));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getDocument(
			IntPtr			pDb,
			uint				uiCollection,
			RetrieveFlags	retrieveFlags,
			ulong				ulDocumentId,
			ref IntPtr		pNode);

//-----------------------------------------------------------------------------
//	documentDone
//-----------------------------------------------------------------------------

		/// <summary>
		/// Indicate that modifications to a document are "done".  This allows
		/// XFLAIM to process the document as needed.
		/// </summary>
		/// <param name="uiCollection">
		/// The document's collection ID.
		/// </param>
		/// <param name="ulDocumentId">
		/// The document ID.
		/// </param>
		public void documentDone(
			uint	uiCollection,
			ulong	ulDocumentId)
		{
			RCODE rc;

			if ((rc = xflaim_Db_documentDone( m_pDb, uiCollection, ulDocumentId)) != 0)
			{
				throw new XFlaimException(rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_documentDone(
			IntPtr	pDb,
			uint		uiCollection,
			ulong		ulDocumentId);

//-----------------------------------------------------------------------------
//	documentDone
//-----------------------------------------------------------------------------

		/// <summary>
		/// Indicate that modifications to a document are "done".  This allows
		/// XFLAIM to process the document as needed.
		/// </summary>
		/// <param name="domNode">
		/// Root node of the document that the application has finished 
		/// modifying
		/// </param>
		public void documentDone(
			DOMNode	domNode)
		{
			RCODE rc;

			if ((rc = xflaim_Db_documentDone2( m_pDb, domNode.getNode())) != 0)
			{
				throw new XFlaimException(rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_documentDone2(
			IntPtr	pDb,
			IntPtr	pNode);

//-----------------------------------------------------------------------------
// createElementDef
//-----------------------------------------------------------------------------

		/// <summary>
		/// Creates a new element definition in the dictionary. 
		/// </summary>
		/// <param name="sNamespaceURI">
		/// The namespace URI that this definition should be
		/// created in.  If null, the default namespace will be used.
		/// </param>
		/// <param name="sElementName">
		/// The name of the element.
		/// </param>
		/// <param name="dataType">
		/// The data type for instances of this element.
		/// </param> 
		/// <param name="uiRequestedId">
		/// if non-zero, then xflaim will try to use this
		/// number as the name ID of the new definition.
		/// </param>
		/// <returns>
		/// Returns the name ID of the new definition.
		/// </returns>
		public uint createElementDef(
			string		sNamespaceURI,
			string		sElementName,
			FlmDataType	dataType,
			uint			uiRequestedId)
			
		{
			RCODE	rc;

			if ((rc = xflaim_Db_createElementDef( m_pDb, sNamespaceURI,
				sElementName, dataType, ref uiRequestedId)) != 0)
			{
				throw new XFlaimException(rc);
			}
			
			return( uiRequestedId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_createElementDef(
			IntPtr		pDb,
			[MarshalAs(UnmanagedType.LPWStr)]
			string		sNamespaceURI,
			[MarshalAs(UnmanagedType.LPWStr)]
			string		sElementName,
			FlmDataType	dataType,
			ref uint		puiRequestedId);

//-----------------------------------------------------------------------------
// createUniqueElmDef
//-----------------------------------------------------------------------------

		/// <summary>
		/// Create a "unique" element definition - i.e., an element definition whose
		/// child elements must all be unique.
		/// </summary>
		/// <param name="sNamespaceURI">
		/// The namespace URI that this definition should be
		/// created in.  If null, the default namespace will be used.
		/// </param>
		/// <param name="sElementName">
		/// The name of the element.
		/// </param>
		/// <param name="uiRequestedId">
		/// if non-zero, then xflaim will try to use this
		/// number as the name ID of the new definition.
		/// </param>
		/// <returns>
		/// Returns the name ID of the new definition.
		/// </returns>
		public uint createUniqueElmDef(
			string	sNamespaceURI,
			string	sElementName,
			uint		uiRequestedId)
			
		{
			RCODE	rc;

			if ((rc = xflaim_Db_createUniqueElmDef( m_pDb, sNamespaceURI,
				sElementName, ref uiRequestedId)) != 0)
			{
				throw new XFlaimException(rc);
			}

			return( uiRequestedId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_createUniqueElmDef(
			IntPtr		pDb,
			[MarshalAs(UnmanagedType.LPWStr)]
			string		sNamespaceURI,
			[MarshalAs(UnmanagedType.LPWStr)]
			string		sElementName,
			ref uint		puiRequestedId);

//-----------------------------------------------------------------------------
// getElementNameId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Gets the name id for a particular element name.
		/// </summary>
		/// <param name="sNamespaceURI">
		/// The namespace URI for the element.
		/// </param>
		/// <param name="sElementName">
		/// The name of the element.
		/// </param>
		/// <returns>
		/// Returns the name ID of the element.
		/// </returns>
		public uint getElementNameId(
			string	sNamespaceURI,
			string	sElementName)
		{
			RCODE	rc;
			uint	uiNameId;

			if ((rc = xflaim_Db_getElementNameId( m_pDb, sNamespaceURI,
				sElementName, out uiNameId)) != 0)
			{
				throw new XFlaimException(rc);
			}

			return( uiNameId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getElementNameId(
			IntPtr	pDb,
			[MarshalAs(UnmanagedType.LPWStr)]
			string	sNamespaceURI,
			[MarshalAs(UnmanagedType.LPWStr)]
			string	sElementName,
			out uint	uiNameId);

//-----------------------------------------------------------------------------
// createAttributeDef
//-----------------------------------------------------------------------------

		/// <summary>
		/// Creates a new attribute definition in the dictionary. 
		/// </summary>
		/// <param name="sNamespaceURI">
		/// The namespace URI that this definition should be
		/// created in.  If null, the default namespace will be used.
		/// </param>
		/// <param name="sAttributeName">
		/// The name of the attribute.
		/// </param>
		/// <param name="dataType">
		/// The data type for instances of this attribute.
		/// </param> 
		/// <param name="uiRequestedId">
		/// if non-zero, then XFLAIM will try to use this
		/// number as the name ID of the new definition.
		/// </param>
		/// <returns>
		/// Returns the name ID of the new definition.
		/// </returns>
		public uint createAttributeDef(
			string		sNamespaceURI,
			string		sAttributeName,
			FlmDataType	dataType,
			uint			uiRequestedId)
			
		{
			RCODE	rc;

			if ((rc = xflaim_Db_createAttributeDef( m_pDb, sNamespaceURI,
				sAttributeName, dataType, ref uiRequestedId)) != 0)
			{
				throw new XFlaimException(rc);
			}
			
			return( uiRequestedId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_createAttributeDef(
			IntPtr		pDb,
			[MarshalAs(UnmanagedType.LPWStr)]
			string		sNamespaceURI,
			[MarshalAs(UnmanagedType.LPWStr)]
			string		sAttributeName,
			FlmDataType	dataType,
			ref uint		puiRequestedId);

//-----------------------------------------------------------------------------
// getAttributeNameId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Gets the name id for a particular attribute.
		/// </summary>
		/// <param name="sNamespaceURI">
		/// The namespace URI of the attribute.
		/// </param>
		/// <param name="sAttributeName">
		/// The name of the attribute.
		/// </param>
		/// <returns>
		/// Returns the name ID of the attribute.
		/// </returns>
		public uint getAttributeNameId(
			string sNamespaceURI,
			string sAttributeName)
		{
			RCODE	rc;
			uint	uiNameId;

			if ((rc = xflaim_Db_getAttributeNameId( m_pDb, sNamespaceURI,
				sAttributeName, out uiNameId)) != 0)
			{
				throw new XFlaimException(rc);
			}

			return( uiNameId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getAttributeNameId(
			IntPtr	pDb,
			[MarshalAs(UnmanagedType.LPWStr)]
			string	sNamespaceURI,
			[MarshalAs(UnmanagedType.LPWStr)]
			string	sAttributeName,
			out uint	puiNameId);

//-----------------------------------------------------------------------------
// createPrefixDef
//-----------------------------------------------------------------------------

		/// <summary>
		/// Creates a new prefix definition in the dictionary. 
		/// </summary>
		/// <param name="sPrefixName">
		/// The name of the attribute.
		/// </param>
		/// <param name="uiRequestedId">
		/// if non-zero, then XFLAIM will try to use this
		/// number as the name ID of the new definition.
		/// </param>
		/// <returns>
		/// Returns the name ID of the new definition.
		/// </returns>
		public uint createPrefixDef(
			string	sPrefixName,
			uint		uiRequestedId)
		{
			RCODE	rc;

			if ((rc = xflaim_Db_createPrefixDef( m_pDb, sPrefixName,
				ref uiRequestedId)) != 0)
			{
				throw new XFlaimException(rc);
			}

			return( uiRequestedId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_createPrefixDef(
			IntPtr	pDb,
			[MarshalAs(UnmanagedType.LPWStr)]
			string	sPrefixName,
			ref uint	puiRequestedId);

//-----------------------------------------------------------------------------
// getPrefixId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Gets the name id for a particular prefix.
		/// </summary>
		/// <param name="sPrefixName">
		/// The name of the prefix.
		/// </param>
		/// <returns>
		/// Returns the name ID of the prefix.
		/// </returns>
		public uint getPrefixId(
			string sPrefixName)
		{
			RCODE	rc;
			uint	uiNameId;

			if ((rc = xflaim_Db_getPrefixId( m_pDb, sPrefixName,
				out uiNameId)) != 0)
			{
				throw new XFlaimException(rc);
			}

			return( uiNameId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getPrefixId(
			IntPtr	pDb,
			[MarshalAs(UnmanagedType.LPWStr)]
			string	sPrefixName,
			out uint	uiNameId);

//-----------------------------------------------------------------------------
// createEncDef
//-----------------------------------------------------------------------------

		/// <summary>
		/// Creates a new prefix definition in the dictionary. 
		/// </summary>
		/// <param name="sEncName">
		/// Encryption definition name.
		/// </param>
		/// <param name="eEncType">
		/// Encryption type.
		/// </param>
		/// <param name="uiRequestedId">
		/// If non-zero, then XFLAIM will try to use this
		/// number as the name ID of the new definition.
		/// </param>
		/// <returns>
		/// Returns the name ID of the new definition.
		/// </returns>
		public uint createEncDef(
			string				sEncName,
			EncryptionScheme	eEncType,
			uint					uiRequestedId)
		{
			RCODE		rc;
			uint		uiKeySize = 128;
			string	sEncType = "aes";

			switch (eEncType)
			{
				case EncryptionScheme.ENC_AES128:
					uiKeySize = 128;
					sEncType = "aes";
					break;
				case EncryptionScheme.ENC_AES192:
					uiKeySize = 192;
					sEncType = "aes";
					break;
				case EncryptionScheme.ENC_AES256:
					uiKeySize = 256;
					sEncType = "aes";
					break;
				case EncryptionScheme.ENC_DES3:
					uiKeySize = 168;
					sEncType = "des3";
					break;
				default:
					throw new XFlaimException( RCODE.NE_XFLM_INVALID_PARM);
			}

			if ((rc = xflaim_Db_createEncDef( m_pDb, sEncType, sEncName,
				uiKeySize, ref uiRequestedId)) != 0)
			{
				throw new XFlaimException(rc);
			}

			return( uiRequestedId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_createEncDef(
			IntPtr	pDb,
			[MarshalAs(UnmanagedType.LPWStr)]
			string	sEncName,
			[MarshalAs(UnmanagedType.LPWStr)]
			string	sEncType,
			uint		uiKeySize,
			ref uint	uiRequestedId);

//-----------------------------------------------------------------------------
// getEncDefId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Gets the ID for a particular encryption definition.
		/// </summary>
		/// <param name="sEncName">
		/// The name of the encryption definition.
		/// </param>
		/// <returns>
		/// Returns the name ID of the encryption definition.
		/// </returns>
		public uint getEncDefId(
			string	sEncName)
		{
			RCODE	rc;
			uint	uiNameId;

			if ((rc = xflaim_Db_getEncDefId( m_pDb, sEncName, out uiNameId)) != 0)
			{
				throw new XFlaimException(rc);
			}

			return( uiNameId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getEncDefId(
			IntPtr	pDb,
			[MarshalAs(UnmanagedType.LPWStr)]
			string	sEncName,
			out uint	uiNameId);

//-----------------------------------------------------------------------------
// createCollectionDef
//-----------------------------------------------------------------------------

		/// <summary>
		/// Creates a collection definition in the dictionary. 
		/// </summary>
		/// <param name="sCollectionName">
		/// The name of the collection.
		/// </param>
		/// <param name="uiEncryptionId">
		/// ID of the encryption definition that should be used
		/// to encrypt this collection.  Zero means the collection will not be encrypted.
		/// </param>
		/// <param name="uiRequestedId">
		/// if non-zero, then XFLAIM will try to use this
		/// number as the name ID of the new definition.
		/// </param>
		/// <returns>
		/// Returns the name ID of the new definition.
		/// </returns>
		public uint createCollectionDef(
			string	sCollectionName,
			uint		uiEncryptionId,
			uint		uiRequestedId)
		{
			RCODE	rc;

			if ((rc = xflaim_Db_createCollectionDef( m_pDb, sCollectionName,
				uiEncryptionId, ref uiRequestedId)) != 0)
			{
				throw new XFlaimException(rc);
			}

			return( uiRequestedId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_createCollectionDef(
			IntPtr	pDb,
			[MarshalAs(UnmanagedType.LPWStr)]
			string	sCollectionName,
			uint		uiEncryptionId,
			ref uint	puiRequestedId);

//-----------------------------------------------------------------------------
// getCollectionNumber
//-----------------------------------------------------------------------------

		/// <summary>
		/// Gets the ID for a particular collection.
		/// </summary>
		/// <param name="sCollectionName">
		/// The name of the collection.
		/// </param>
		/// <returns>
		/// Returns the ID of the collection definition.
		/// </returns>
		public uint getCollectionNumber(
			string	sCollectionName)
		{
			RCODE rc;
			uint uiNameId;

			if ((rc = xflaim_Db_getCollectionNumber( m_pDb, sCollectionName,
				out uiNameId)) != 0)
			{
				throw new XFlaimException(rc);
			}

			return( uiNameId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getCollectionNumber(
			IntPtr	pDb,
			[MarshalAs(UnmanagedType.LPWStr)]
			string	sCollectionName,
			out uint	uiNameId);

//-----------------------------------------------------------------------------
// getIndexNumber
//-----------------------------------------------------------------------------

		/// <summary>
		/// Gets the ID for a particular index.
		/// </summary>
		/// <param name="sIndexName">
		/// The name of the index.
		/// </param>
		/// <returns>
		/// Returns the ID of the index definition.
		/// </returns>
		public uint getIndexNumber(
			string	sIndexName)
		{
			RCODE rc;
			uint uiNameId;

			if ((rc = xflaim_Db_getIndexNumber( m_pDb, sIndexName,
				out uiNameId)) != 0)
			{
				throw new XFlaimException(rc);
			}

			return( uiNameId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getIndexNumber(
			IntPtr	pDb,
			[MarshalAs(UnmanagedType.LPWStr)]
			string	sIndexName,
			out uint	uiNameId);
	
//-----------------------------------------------------------------------------
// getDictionaryDef
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieve a dictionary definition document.
		/// </summary>
		/// <param name="dictType">
		/// The type of dictionary definition being retrieved.
		/// </param>
		/// <param name="uiDictNumber">
		/// The number the dictionary definition being retrieved.
		/// </param>
		/// <param name="nodeToReuse">
		/// An existing DOM node object can optionally be passed in.  It will
		/// be reused rather than allocating a new object.
		/// </param>
		/// <returns>
		/// Returns the root <see cref="DOMNode"/> of the document.
		/// </returns>
		public DOMNode getDictionaryDef(
			ReservedElmTag	dictType,
			uint				uiDictNumber,
			DOMNode			nodeToReuse)
		{
			RCODE		rc;
			IntPtr	pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;

			if ((rc = xflaim_Db_getDictionaryDef( m_pDb, dictType,
				uiDictNumber, ref pNode)) != 0)
			{
				throw new XFlaimException(rc);
			}

			if (nodeToReuse != null)
			{
				nodeToReuse.setNodePtr(pNode, this);
				return( nodeToReuse);
			}

			return( new DOMNode(pNode, this));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getDictionaryDef(
			IntPtr			pDb,
			ReservedElmTag	dictType,
			uint				uiDictNumber,
			ref IntPtr		ppNode);

//-----------------------------------------------------------------------------
// getDictionaryName
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get a dictionary definition's name.
		/// </summary>
		/// <param name="dictType">
		/// The type of dictionary definition whose name is to be returned.
		/// </param>
		/// <param name="uiDictNumber">
		/// The number of the dictionary definition.
		/// </param>
		/// <returns>
		/// Name of the dictionary item.
		/// </returns>
 		public string getDictionaryName(
			ReservedElmTag	dictType,
			uint				uiDictNumber)
		{
			RCODE		rc;
			IntPtr	puzDictName;
			string	sDictName;

			if ((rc = xflaim_Db_getDictionaryName( m_pDb, dictType, 
				uiDictNumber, out puzDictName)) != 0)
			{
				throw new XFlaimException( rc);
			}

			sDictName = Marshal.PtrToStringUni( puzDictName);
			m_dbSystem.freeUnmanagedMem( puzDictName);
			return( sDictName);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getDictionaryName(
			IntPtr			pDb,
			ReservedElmTag	dictType,
			uint				uiDictNumber,
			out IntPtr		ppuzDictName);

//-----------------------------------------------------------------------------
// getElementNamespace
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get an element definition's namespace.
		/// </summary>
		/// <param name="uiDictNumber">
		/// The number of the element definition.
		/// </param>
		/// <returns>
		/// Returns the namespace for the element definition.
		/// </returns>
		public string getElementNamespace(
			uint		uiDictNumber)
		{
			RCODE		rc;
			IntPtr	puzElmNamespace;
			string	sElmNamespace;

			if ((rc = xflaim_Db_getElementNamespace( m_pDb,
				uiDictNumber, out puzElmNamespace)) != 0)
			{
				throw new XFlaimException(rc);
			}

			sElmNamespace = Marshal.PtrToStringUni( puzElmNamespace);
			m_dbSystem.freeUnmanagedMem( puzElmNamespace);
			return( sElmNamespace);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getElementNamespace(
			IntPtr		pDb,
			uint			uiDictNumber,
			out IntPtr	ppuzElmNamespace);

//-----------------------------------------------------------------------------
// getAttributeNamespace
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get an attribute definition's namespace.
		/// </summary>
		/// <param name="uiDictNumber">
		/// The number of the attribute definition.
		/// </param>
		/// <returns>
		/// Returns the namespace for the attribute definition.
		/// </returns>
		public string getAttributeNamespace(
			uint	uiDictNumber)
		{
			RCODE		rc;
			IntPtr	puzAttrNamespace;
			string	sAttrNamespace;

			if ((rc = xflaim_Db_getAttributeNamespace( m_pDb,
				uiDictNumber, out puzAttrNamespace)) != 0)
			{
				throw new XFlaimException(rc);
			}

			sAttrNamespace = Marshal.PtrToStringUni( puzAttrNamespace);
			m_dbSystem.freeUnmanagedMem( puzAttrNamespace);
			return( sAttrNamespace);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getAttributeNamespace(
			IntPtr		pDb,
			uint			uiDictNumber,
			out IntPtr	ppuzAttrNamespace);

//-----------------------------------------------------------------------------
// getNode
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves the specified node from the specified collection. 
		/// </summary>
		/// <param name="uiCollection">
		/// The collection where the node is stored.
		/// </param>
		/// <param name="ulNodeId">
		/// The ID number of the node to be retrieved.
		/// </param>
		/// <param name="nodeToReuse">
		/// </param>
		/// <returns></returns>
		public DOMNode getNode(
			uint			uiCollection,
			ulong			ulNodeId,
			DOMNode		nodeToReuse)
		{
			RCODE		rc;
			IntPtr	pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;

			if ((rc = xflaim_Db_getNode( m_pDb, uiCollection, ulNodeId,
				ref pNode)) != 0)
			{
				throw new XFlaimException(rc);
			}

			if (nodeToReuse != null)
			{
				nodeToReuse.setNodePtr(pNode, this);
				return( nodeToReuse);
			}

			return( new DOMNode(pNode, this));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getNode(
			IntPtr		pDb,
			uint			uiCollection,
			ulong			ulNodeId,
			ref IntPtr	ppNode);

//-----------------------------------------------------------------------------
// getAttribute
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves the specified attribute node from the specified collection.
		/// </summary>
		/// <param name="uiCollection">
		/// The collection where the attribute is stored.
		/// </param>
		/// <param name="ulElementNodeId">
		/// The ID number of the element node that contains the attribute 
		/// to be retrieved.
		/// </param>
		/// <param name="uiAttrNameId">
		/// The attribute id of the attribute to be retrieved.
		/// </param>
		/// <param name="nodeToReuse">
		/// An existing DOM node object can optionally be passed in.  It will
		/// be reused rather than allocating a new object.
		/// </param>
		/// <returns>
		/// Returns the attribute node <see cref="DOMNode"/>.
		/// </returns>
		public DOMNode getAttribute(
			uint			uiCollection,
			ulong			ulElementNodeId,
			uint			uiAttrNameId,
			DOMNode		nodeToReuse)
		{
			RCODE		rc;
			IntPtr	pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;

			if ((rc = xflaim_Db_getAttribute( m_pDb, uiCollection, 
				ulElementNodeId, uiAttrNameId, ref pNode)) != 0)
			{
				throw new XFlaimException(rc);
			}

			if (nodeToReuse != null)
			{
				nodeToReuse.setNodePtr(pNode, this);
				return( nodeToReuse);
			}

			return( new DOMNode(pNode, this));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getAttribute(
			IntPtr		pDb,
			uint			uiCollection,
			ulong			ulElementNodeId,
			uint			uiAttrNameId,
			ref IntPtr	ppNode);

//-----------------------------------------------------------------------------
// getDataType
//-----------------------------------------------------------------------------

		/// <summary>
		/// Returns the data type that was specified for a particular dictionary
		/// definition.  NOTE: This really only applies to element and attribute
		/// definitions.
		/// </summary>
		/// <param name="dictType">
		/// The type of dictionary definition whose data type is to be returned.
		/// </param>
		/// <param name="uiDictNumber">
		/// The number of the dictionary definition.
		/// </param>
		/// <returns>
		/// Data type of the dictionary object.
		/// </returns>
		public FlmDataType getDataType(
			ReservedElmTag	dictType,
			uint				uiDictNumber)
		{
			RCODE			rc;
			FlmDataType	dataType;

			if ((rc = xflaim_Db_getDataType( m_pDb,
				dictType, uiDictNumber, out dataType)) != 0)
			{
				throw new XFlaimException(rc);
			}

			return( dataType);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getDataType(
			IntPtr				pDb,
			ReservedElmTag		dictType,
			uint					uiDictNumer,
			out FlmDataType	dataType);

//-----------------------------------------------------------------------------
// backupBegin
//-----------------------------------------------------------------------------

		/// <summary>
		/// Sets up a backup operation.
		/// </summary>
		/// <param name="bFullBackup">
		/// Specifies whether the backup is to be a full backup (true) or an 
		/// incremental backup (false).
		/// </param>
		/// <param name="bLockDb">
		/// Specifies whether the database should be locked during the back (a "warm" backup)
		/// or unlocked (a "hot" backup).
		/// </param>
		/// <param name="uiMaxLockWait">
		/// This parameter is only used if the bLockDb parameter is true.  
		/// It specifies the maximum number of seconds to wait to obtain a lock.
		/// </param>
		/// <returns>
		/// If successful, this method returns a <see cref="Backup"/> object which can then be used
		/// to perform the backup operation.  The database will be locked if bLockDb was specified.
		/// Otherwise, a read transaction will have been started to perform the backup.
		/// </returns>
		public Backup backupBegin(
			bool			bFullBackup,
			bool			bLockDb,
			uint			uiMaxLockWait)
		{
			RCODE		rc;
			IntPtr	pBackup;

			if ((rc = xflaim_Db_backupBegin( m_pDb, (bFullBackup ? 1 : 0),
				(bLockDb ? 1 : 0), uiMaxLockWait, out pBackup)) != 0)
			{
				throw new XFlaimException( rc);
			}

			return( new Backup( pBackup, this));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_backupBegin(
			IntPtr		pDb,
			int			bFullBackup,
			int			bLockDb,
			uint			uiMaxLockWait,
			out IntPtr	ppBackup);

//-----------------------------------------------------------------------------
// importDocument
//-----------------------------------------------------------------------------

		/// <summary>
		/// Imports an XML document into the XFlaim database.  The import requires
		/// an update transaction.
		/// </summary>
		/// <param name="istream">
		/// Input stream containing the document(s) to be imported
		/// </param>
		/// <param name="uiCollection">
		/// Destination collection for imported document(s).
		/// </param>
		/// <param name="nodeToReuse">
		/// An existing DOM node object can optionally be passed in.  It will
		/// be reused rather than allocating a new object.
		/// </param>
		/// <param name="importStats">
		/// Import statistics is returned here if a non-null value is passed in.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> that is the root of the imported document.
		/// </returns>
		public DOMNode importDocument(
			IStream					istream,
			uint						uiCollection,
			DOMNode					nodeToReuse,
			CS_XFLM_IMPORT_STATS	importStats)
		{
			RCODE		rc;
			IntPtr	pDocumentNode = (nodeToReuse != null) ? nodeToReuse.getNode() : IntPtr.Zero;

			if (importStats == null)
			{
				importStats = new CS_XFLM_IMPORT_STATS();
			}

			if ((rc = xflaim_Db_importDocument( m_pDb, istream.getIStream(),
							uiCollection, ref pDocumentNode, importStats)) != 0)
			{
				throw new XFlaimException(rc);
			}

			if( nodeToReuse != null)
			{
				nodeToReuse.setNodePtr( pDocumentNode, this);
				return( nodeToReuse);
			}

			return( new DOMNode( pDocumentNode, this));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_importDocument(
			IntPtr					pDb,
			IntPtr					pIStream,
			uint						uiCollection,
			ref IntPtr				ppDocumentNode,
			CS_XFLM_IMPORT_STATS	pImportStats);

//-----------------------------------------------------------------------------
// importIntoDocument
//-----------------------------------------------------------------------------

		/// <summary>
		/// Imports an XML fragment into a document.  The import requires
		/// an update transaction.
		/// </summary>
		/// <param name="istream">
		/// Input stream containing the nodes to be imported.
		/// </param>
		/// <param name="nodeToLinkTo">
		/// Existing node that imported nodes will link to.
		/// </param>
		/// <param name="insertLocation">
		/// Where imported XML fragment is to be linked with respect
		/// to nodeToLinkTo.
		/// </param>
		/// <returns>
		/// Returns import statistics <see cref="CS_XFLM_IMPORT_STATS"/>.
		/// </returns>
		public CS_XFLM_IMPORT_STATS importIntoDocument(
			IStream				istream,
			DOMNode				nodeToLinkTo,
			eNodeInsertLoc		insertLocation)
		{
			RCODE						rc;
			CS_XFLM_IMPORT_STATS	importStats = new CS_XFLM_IMPORT_STATS();

			if ((rc = xflaim_Db_importIntoDocument( m_pDb, 
				istream.getIStream(), nodeToLinkTo.getNode(), insertLocation,
				importStats)) != 0)
			{
				throw new XFlaimException(rc);
			}

			return( importStats);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_importIntoDocument(
			IntPtr					pDb,
			IntPtr					pIStream,
			IntPtr					pNodeToLinkTo,
			eNodeInsertLoc			insertLocation,
			CS_XFLM_IMPORT_STATS	importStats);

//-----------------------------------------------------------------------------
// changeItemState
//-----------------------------------------------------------------------------

		/// <summary>
		/// Change a dictionary definition's state.  This routine is used to determine if
		/// the dictionary item can be deleted.  It may also be used to force the
		/// definition to be deleted - once the database has determined that the
		/// definition is not in use anywhere.  This should only be used for
		/// element definitions and attribute definitions definitions.
		/// </summary>
		/// <param name="dictType">
		/// Type of dictionary definition whose state is being changed.
		/// </param>
		/// <param name="uiDictNumber">
		/// Number of element or attribute definition whose state
		/// is to be changed
		/// </param>
		/// <param name="eStateToChangeTo">
		/// State the definition is to be changed to.
		/// </param>
		public void changeItemState(
			ReservedElmTag	dictType,
			uint				uiDictNumber,
			ChangeState		eStateToChangeTo)
		{
			RCODE		rc;
			string	sState = "";

			switch (eStateToChangeTo)
			{
				case ChangeState.STATE_CHECKING:
					sState = "checking";
					break;
				case ChangeState.STATE_PURGE:
					sState = "purge";
					break;
				case ChangeState.STATE_ACTIVE:
					sState = "active";
					break;
				default:
					throw new XFlaimException( RCODE.NE_XFLM_INVALID_PARM);
			}

			if ((rc = xflaim_Db_changeItemState( m_pDb,
								dictType, uiDictNumber, sState)) != 0)
			{
				throw new XFlaimException(rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_changeItemState(
			IntPtr			pDb,
			ReservedElmTag	dictType,
			uint				uiDictNumber,
			[MarshalAs(UnmanagedType.LPStr)]
			string			sState);

//-----------------------------------------------------------------------------
// getRflFileName
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the name of a roll-forward log file.
		/// </summary>
		/// <param name="uiFileNum">
		/// Roll-forward log file number whose name is to be returned.
		/// </param>
		/// <param name="bBaseOnly">
		/// If true, only the base name of the file will be returned.
		/// Otherwise, the entire path will be returned.
		/// </param>
		/// <returns>
		/// Name of the file.
		/// </returns>
		public string getRflFileName(
			uint	uiFileNum,
			bool	bBaseOnly)
		{
			RCODE		rc;
			IntPtr	pszFileName;
			string	sFileName;

			if ((rc = xflaim_Db_getRflFileName( m_pDb, uiFileNum,
				(int)(bBaseOnly ? 1 : 0), out pszFileName)) != 0)
			{
				throw new XFlaimException(rc);
			}

			sFileName = Marshal.PtrToStringAnsi( pszFileName);
			m_dbSystem.freeUnmanagedMem( pszFileName);
			return( sFileName);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getRflFileName(
			IntPtr		pDb,
			uint			uiFileNum,
			int			bBaseOnly,
			out IntPtr	ppszName);

//-----------------------------------------------------------------------------
// setNextNodeId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Set the next node ID for a collection.  This will be the node ID for
		/// the next node that is created in the collection.  NOTE: The node ID must
		/// be greater than or equal to the current next node ID that is already
		/// set for the collection.  Otherwise, it is ignored.
		/// </summary>
		/// <param name="uiCollection">
		/// Collection whose next node ID is to be set
		/// </param>
		/// <param name="ulNextNodeId">
		/// Next node ID value
		/// </param>
		public void setNextNodeId(
			uint		uiCollection,
			ulong		ulNextNodeId)
		{
			RCODE rc;

			if ((rc = xflaim_Db_setNextNodeId( m_pDb, uiCollection, 
				ulNextNodeId)) != 0)
			{
				throw new XFlaimException(rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_setNextNodeId(
			IntPtr	pDb,
			uint		uiCollection,
			ulong		ulNextNodeId);

//-----------------------------------------------------------------------------
// setNextDictNum
//-----------------------------------------------------------------------------

		/// <summary>
		/// Set the next dictionary number that is to be assigned for a particular
		/// type if dictionary definition.  The specified "next dictionary number"	
		/// must be greater than the current "next dictionary number".  Otherwise,
		/// no action is taken.
		/// </summary>
		/// <param name="dictType">
		/// Type of dictionary definition whose "next dictionary number" is to 
		/// be changed.
		/// </param>
		/// <param name="uiDictNumber">
		/// Next dictionary number.
		/// </param>
		public void setNextDictNum(
			ReservedElmTag	dictType,
			uint				uiDictNumber)
		{
			RCODE	rc;

			if ((rc = xflaim_Db_setNextDictNum( m_pDb, dictType,
				uiDictNumber)) != 0)
			{
				throw new XFlaimException(rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_setNextDictNum(
			IntPtr			pDb,
			ReservedElmTag	dictType,
			uint				uiDictNumber);

//-----------------------------------------------------------------------------
// setRflKeepFilesFlag
//-----------------------------------------------------------------------------

		/// <summary>
		/// Specify whether the roll-forward log should keep or not keep RFL files.
		/// </summary>
		/// <param name="bKeep">
		/// Flag specifying whether to keep or not keep RFL files.
		/// </param>
		public void setRflKeepFilesFlag(
			bool		bKeep)
		{
			RCODE rc;

			if ((rc = xflaim_Db_setRflKeepFilesFlag( m_pDb,
				(int)(bKeep ? 1 : 0))) != 0)
			{
				throw new XFlaimException(rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_setRflKeepFilesFlag(
			IntPtr	pDb,
			int		bKeep);

//-----------------------------------------------------------------------------
// getRflKeepFlag
//-----------------------------------------------------------------------------

		/// <summary>
		/// Determine whether or not the roll-forward log files are being kept.
		/// </summary>
		/// <returns>
		/// Returns true if RFL files are being kept, false otherwise.
		/// </returns>
		public bool getRflKeepFlag()
		{
			RCODE	rc;
			int	bKeep;

			if ((rc = xflaim_Db_getRflKeepFlag( m_pDb, out bKeep)) != 0)
			{
				throw new XFlaimException( rc);
			}

			return( bKeep != 0 ? true : false);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getRflKeepFlag(
			IntPtr		pDb,
			out int		bKeep);

//-----------------------------------------------------------------------------
// setRflDir
//-----------------------------------------------------------------------------

		/// <summary>
		/// Set the RFL directory.
		/// </summary>
		/// <param name="sRflDir">
		/// Name of RFL directory.
		/// </param>
		public void setRflDir(
			string	sRflDir)
		{
			RCODE rc;

			if ((rc = xflaim_Db_setRflDir( m_pDb, sRflDir)) != 0)
			{
				throw new XFlaimException(rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_setRflDir(
			IntPtr	pDb,
			[MarshalAs(UnmanagedType.LPStr)]
			string	sRflDir);

//-----------------------------------------------------------------------------
// getRflDir
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the current RFL directory.
		/// </summary>
		/// <returns>
		/// Returns the current RFL directory name.
		/// </returns>
		public string getRflDir()
		{
			RCODE		rc;
			IntPtr	pszRflDir;
			string	sRflDir;

			if ((rc = xflaim_Db_getRflDir( m_pDb, out pszRflDir)) != 0)
			{
				throw new XFlaimException(rc);
			}

			sRflDir = Marshal.PtrToStringAnsi( pszRflDir);
			m_dbSystem.freeUnmanagedMem( pszRflDir);
			return( sRflDir);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getRflDir(
			IntPtr		pDb,
			out IntPtr	ppszRflDir);

//-----------------------------------------------------------------------------
//	getRflFileNum
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the current RFL file number.
		/// </summary>
		/// <returns>
		/// Returns the current RFL file number.
		/// </returns>
		public uint getRflFileNum()
		{
			RCODE	rc;
			uint	uiRflFileNum;

			if ((rc = xflaim_Db_getRflFileNum( m_pDb, out uiRflFileNum)) != 0)
			{
				throw new XFlaimException(rc);
			}

			return( uiRflFileNum);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getRflFileNum(
			IntPtr		pDb,
			out uint		puiRflFileNum);

//-----------------------------------------------------------------------------
// getHighestNotUsedRflFileNum
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the highest RFL file number that is no longer in use by XFLAIM.
		/// This RFL file can be removed from the system if needed.
		/// </summary>
		/// <returns>
		/// Returns the highest RFL file number that is no longer in use.
		/// </returns>
		public uint getHighestNotUsedRflFileNum()
		{
			RCODE	rc;
			uint	uiRflFileNum;

			if ((rc = xflaim_Db_getHighestNotUsedRflFileNum( m_pDb, 
				out uiRflFileNum)) != 0)
			{
				throw new XFlaimException(rc);
			}

			return( uiRflFileNum);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getHighestNotUsedRflFileNum(
			IntPtr	pDb,
			out uint	puiRflFileNum);

//-----------------------------------------------------------------------------
// setRflFileSizeLimits
//-----------------------------------------------------------------------------

		/// <summary>
		/// Set size limits for RFL files.
		/// </summary>
		/// <param name="uiMinRflSize">
		/// Minimum RFL file size.  Database will roll to the next RFL file when 
		/// the current RFL file reaches this size.  If possible it will complete
		/// the current transaction before rolling to the next file.
		/// </param>
		/// <param name="uiMaxRflSize">
		/// Maximum RFL file size.  Database will not allow an RFL file 
		/// to exceed this size.  Even if it is in the middle of a transaction,
		/// it will roll to the next RFL file before this size is allowed
		/// to be exceeded.  Thus, the database first looks for an opportunity to
		/// roll to the next file when the RFL file exceeds iMinRflSize.  If it can
		/// fit the current transaction in without exceeded iMaxRflSize, it will do
		/// so and then roll to the next file.  Otherwise, it will roll to the next
		/// file before iMaxRflSize is exceeded.	
		/// </param>
		public void setRflFileSizeLimits(
			uint		uiMinRflSize,
			uint		uiMaxRflSize)
		{
			RCODE rc;

			if ((rc = xflaim_Db_setRflFileSizeLimits( m_pDb, 
				uiMinRflSize, uiMaxRflSize)) != 0)
			{
				throw new XFlaimException(rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_setRflFileSizeLimits(
			IntPtr	pDb,
			uint		uiMinRflSize,
			uint		uiMaxRflSize);

//-----------------------------------------------------------------------------
// getRflFileSizeLimits
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the minimum and maximum RFL file sizes.
		/// </summary>
		/// <param name="uiMinRflSize">
		/// Returns minimum RFL file size.
		/// </param>
		/// <param name="uiMaxRflSize">
		/// Returns maximum RFL file size.
		/// </param>
		public void getRflFileSizeLimits(
			out uint	uiMinRflSize,
			out uint	uiMaxRflSize)
		{
			RCODE	rc;

			if ((rc = xflaim_Db_getRflFileSizeLimits( m_pDb,
								out uiMinRflSize, out uiMaxRflSize)) != 0)
			{
				throw new XFlaimException(rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getRflFileSizeLimits(
			IntPtr	pDb,
			out uint	puiMinRflSize,
			out uint	puiMaxRflSize);

//-----------------------------------------------------------------------------
// rflRollToNextFile
//-----------------------------------------------------------------------------

		/// <summary>
		/// Force the database to roll to the next RFL file.
		/// </summary>
		public void rflRollToNextFile()
		{
			RCODE rc;

			if ((rc = xflaim_Db_rflRollToNextFile(m_pDb)) != 0)
			{
				throw new XFlaimException(rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_rflRollToNextFile(
			IntPtr pDb);

//-----------------------------------------------------------------------------
// setKeepAbortedTransInRflFlag
//-----------------------------------------------------------------------------

		/// <summary>
		/// Specify whether the roll-forward log should keep or not keep aborted
		/// transactions.
		/// </summary>
		/// <param name="bKeep">
		/// Flag specifying whether to keep or not keep aborted transactions.
		/// </param>
		public void setKeepAbortedTransInRflFlag(
			bool	bKeep)
		{
			RCODE rc;

			if ((rc = xflaim_Db_setKeepAbortedTransInRflFlag( m_pDb,
				(int)(bKeep ? 1 : 0))) != 0)
			{
				throw new XFlaimException(rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_setKeepAbortedTransInRflFlag(
			IntPtr	pDb,
			int		bKeep);

//-----------------------------------------------------------------------------
// getKeepAbortedTransInRflFlag
//-----------------------------------------------------------------------------

		/// <summary>
		/// Determine whether or not the roll-forward log is keeping aborted
		/// transactions.
		/// </summary>
		/// <returns>
		/// Returns true if aborted transactions are being kept, false otherwise.
		/// </returns>
		public bool getKeepAbortedTransInRflFlag()
		{
			RCODE rc;
			int bKeep;


			if ((rc = xflaim_Db_getKeepAbortedTransInRflFlag( m_pDb,
				out bKeep)) != 0)
			{
				throw new XFlaimException(rc);
			}

			return( bKeep != 0 ? true : false);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getKeepAbortedTransInRflFlag(
			IntPtr	pDb,
			out int	pbKeep);

//-----------------------------------------------------------------------------
// setAutoTurnOffKeepRflFlag
//-----------------------------------------------------------------------------

		/// <summary>
		/// Specify whether the roll-forward log should automatically turn off the
		/// keeping of RFL files if the RFL volume fills up.
		/// </summary>
		/// <param name="bAutoTurnOff">
		/// Flag specifying whether to automatically turn off the
		/// keeping of RFL files if the RFL volume fills up.
		/// </param>
		public void setAutoTurnOffKeepRflFlag(
			bool	bAutoTurnOff)
		{
			RCODE rc;

			if ((rc = xflaim_Db_setAutoTurnOffKeepRflFlag( m_pDb,
				(int)(bAutoTurnOff ? 1 : 0))) != 0)
			{
				throw new XFlaimException(rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_setAutoTurnOffKeepRflFlag(
			IntPtr	pDb,
			int		bAutoTurnOff);

//-----------------------------------------------------------------------------
// getAutoTurnOffKeepRflFlag
//-----------------------------------------------------------------------------

		/// <summary>
		/// Determine whether or not keeping of RFL files will automatically be
		/// turned off if the RFL volume fills up.
		/// </summary>
		/// <returns>
		/// Returns true if the keeping of RFL files will automatically be
		/// turned off when the RFL volume fills up, false otherwise.
		/// </returns>
		public bool getAutoTurnOffKeepRflFlag()
		{
			RCODE	rc;
			int	bAutoTurnOff;

			if ((rc = xflaim_Db_getAutoTurnOffKeepRflFlag( m_pDb, out bAutoTurnOff)) != 0)
			{
				throw new XFlaimException(rc);
			}

			return( bAutoTurnOff != 0 ? true : false);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getAutoTurnOffKeepRflFlag(
			IntPtr	pDb,
			out int	pbAutoTurnOff);

//-----------------------------------------------------------------------------
// setFileExtendSize
//-----------------------------------------------------------------------------

		/// <summary>
		/// Set the file extend size for the database.  This size specifies how much
		/// to extend a database file when it needs to be extended. 
		/// </summary>
		/// <param name="uiFileExtendSize">
		/// File extend size.
		/// </param>
		public void setFileExtendSize(
			uint	uiFileExtendSize)
		{
			xflaim_Db_setFileExtendSize( m_pDb, uiFileExtendSize);
		}

		[DllImport("xflaim")]
		private static extern void xflaim_Db_setFileExtendSize(
			IntPtr	pDb,
			uint		uiFileExtendSize);

//-----------------------------------------------------------------------------
// getFileExtendSize
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the file extend size for the database.
		/// </summary>
		/// <returns>
		/// Returns file extend size.
		/// </returns>
		public uint getFileExtendSize()
		{
			return( xflaim_Db_getFileExtendSize( m_pDb));
		}

		[DllImport("xflaim")]
		private static extern uint xflaim_Db_getFileExtendSize(
			IntPtr	pDb);

//-----------------------------------------------------------------------------
// getDbVersion
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the database version for the database.  This is the version of the
		/// database, not the code.
		/// </summary>
		/// <returns>
		/// Returns database version.
		/// </returns>
		public uint getDbVersion()
		{
			return( xflaim_Db_getDbVersion( m_pDb));
		}

		[DllImport("xflaim")]
		private static extern uint xflaim_Db_getDbVersion(
			IntPtr	pDb);

//-----------------------------------------------------------------------------
// getBlockSize
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the database block size.
		/// </summary>
		/// <returns>
		/// Returns database block size.
		/// </returns>
		public uint getBlockSize()
		{
			return( xflaim_Db_getBlockSize( m_pDb));
		}

		[DllImport("xflaim")]
		private static extern uint xflaim_Db_getBlockSize(
			IntPtr	pDb);

//-----------------------------------------------------------------------------
// getDefaultLanguage
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the database default language.
		/// </summary>
		/// <returns>
		/// Returns database default language.
		/// </returns>
		public Languages getDefaultLanguage()
		{
			return( xflaim_Db_getDefaultLanguage( m_pDb));
		}

		[DllImport("xflaim")]
		private static extern Languages xflaim_Db_getDefaultLanguage(
			IntPtr	pDb);

//-----------------------------------------------------------------------------
// getTransID
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the database's current transaction ID.  If no transaction is
		/// currently running, but this Db object has an exclusive lock on the database,
		/// the transaction ID of the last committed transaction will be returned.
		/// If no transaction is running, and this Db object does not have an
		/// exclusive lock on the database, zero is returned.
		/// </summary>
		/// <returns>
		/// Transaction ID
		/// </returns>
		public ulong getTransID()
		{
			return( xflaim_Db_getTransID( m_pDb));
		}

		[DllImport("xflaim")]
		private static extern ulong xflaim_Db_getTransID(
			IntPtr	pDb);

//-----------------------------------------------------------------------------
// getDbControlFileName
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the name of the database's control file
		/// </summary>
		/// <returns>
		/// Returns control file name.
		/// </returns>
		public string getDbControlFileName()
		{
			RCODE		rc;
			IntPtr	pszFileName;
			string	sFileName;

			if ((rc = xflaim_Db_getDbControlFileName( m_pDb, out pszFileName)) != 0)
			{
				throw new XFlaimException(rc);
			}

			sFileName = Marshal.PtrToStringAnsi( pszFileName);
			m_dbSystem.freeUnmanagedMem( pszFileName);
			return( sFileName);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getDbControlFileName(
			IntPtr		pDb,
			out IntPtr	ppszFileName);

//-----------------------------------------------------------------------------
// getLastBackupTransID
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the transaction ID of the last backup that was taken on the database.
		/// </summary>
		/// <returns>
		/// Returns last backup transaction ID.
		/// </returns>
		public ulong getLastBackupTransID()
		{
			RCODE	rc;
			ulong	ulTransId;

			if ((rc = xflaim_Db_getLastBackupTransID( m_pDb, out ulTransId)) != 0)
			{
				throw new XFlaimException(rc);
			}

			return( ulTransId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getLastBackupTransID(
			IntPtr		pDb,
			out ulong	ulTransId);

//-----------------------------------------------------------------------------
// getBlocksChangedSinceBackup
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the number of blocks that have changed since the last backup was
		/// taken.
		/// </summary>
		/// <returns>
		/// Returns number of blocks that have changed.
		/// </returns>
		public uint getBlocksChangedSinceBackup()
		{
			RCODE	rc;
			uint	uiBlocksChanged;

			if ((rc = xflaim_Db_getBlocksChangedSinceBackup( m_pDb,
				out uiBlocksChanged)) != 0)
			{
				throw new XFlaimException(rc);
			}

			return( uiBlocksChanged);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getBlocksChangedSinceBackup(
			IntPtr	pDb,
			out uint	uiBlocksChanged);

//-----------------------------------------------------------------------------
// getNextIncBackupSequenceNum
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the next incremental backup sequence number for the database.
		/// </summary>
		/// <returns>
		/// Returns next incremental backup sequence number.
		/// </returns>
		public uint getNextIncBackupSequenceNum()
		{
			RCODE	rc;
			uint	uiNextIncBackupSequenceNum;

			if ((rc = xflaim_Db_getNextIncBackupSequenceNum( m_pDb,
				out uiNextIncBackupSequenceNum)) != 0)
			{
				throw new XFlaimException(rc);
			}

			return( uiNextIncBackupSequenceNum);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getNextIncBackupSequenceNum(
			IntPtr	pDb,
			out uint	uiNextIncBackupSequenceNum);

//-----------------------------------------------------------------------------
// getDiskSpaceUsage
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the disk space usage for the database.
		/// </summary>
		/// <param name="ulDataSize">
		/// Returns the amount of disk space currently being used by data files.
		/// </param>
		/// <param name="ulRollbackSize">
		/// Returns the amount of disk space currently being used by rollback files.
		/// </param>
		/// <param name="ulRflSize">
		/// Returns the amount of disk space currently being used by RFL files.
		/// </param>
		public void getDiskSpaceUsage(
			out ulong	ulDataSize,
			out ulong	ulRollbackSize,
			out ulong	ulRflSize)
		{
			RCODE	rc;

			if ((rc = xflaim_Db_getDiskSpaceUsage( m_pDb,
				out ulDataSize, out ulRollbackSize, out ulRflSize)) != 0)
			{
				throw new XFlaimException(rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getDiskSpaceUsage(
			IntPtr		pDb,
			out ulong	ulDataSize,
			out ulong	ulRollbackSize,
			out ulong	ulRflSize);

//-----------------------------------------------------------------------------
// getMustCloseRC
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get error code that caused the database to force itself to close.
		/// </summary>
		/// <returns>
		/// Returns error code that caused the "must close" condition.
		/// </returns>
		public RCODE getMustCloseRC()
		{
			return( xflaim_Db_getMustCloseRC( m_pDb));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getMustCloseRC(
			IntPtr	pDb);

//-----------------------------------------------------------------------------
// getAbortRC
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get error code that caused the current transaction to require an abort.
		/// </summary>
		/// <returns>
		/// Returns the error code that requires the transaction to abort
		/// </returns>
		public RCODE getAbortRC()
		{
			return( xflaim_Db_getAbortRC(m_pDb));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_getAbortRC(
			IntPtr	pDb);

//-----------------------------------------------------------------------------
// setMustAbortTrans
//-----------------------------------------------------------------------------

		/// <summary>
		/// Force the current transaction to abort.  This method should be called
		/// when the code should not be the code that aborts the transation, but
		/// wants to require that the transaction be aborted by whatever module has
		/// the authority to abort the transaction.  An error code may be
		/// set to indicate what error condition is causing the transaction to be
		/// aborted.
		/// </summary>
		/// <param name="rc">
		/// Error code that indicates why the transaction is aborting.
		/// </param>
		public void setMustAbortTrans(
			RCODE		rc)
		{
			xflaim_Db_setMustAbortTrans( m_pDb, rc);
		}

		[DllImport("xflaim")]
		private static extern void xflaim_Db_setMustAbortTrans(
			IntPtr	pDb,
			RCODE		rc);

//-----------------------------------------------------------------------------
// enableEncryption
//-----------------------------------------------------------------------------

		/// <summary>
		/// Enable encryption support for this database.
		/// </summary>
		public void enableEncryption()
		{
			RCODE	rc;

			if ((rc = xflaim_Db_enableEncryption( m_pDb)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_enableEncryption(
			IntPtr	pDb);

//-----------------------------------------------------------------------------
// wrapKey
//-----------------------------------------------------------------------------

		/// <summary>
		/// Wrap the database key in a password.  This method is called when it is
		/// desirable to move the database to a different machine.  Normally, the
		/// database key is wrapped in the local NICI storage key - which means that
		/// the database can only be opened and accessed on that machine. -- Once
		/// the database key is wrapped in a password, the password must be
		/// supplied to the dbOpen method to open the database.
		/// </summary>
		/// <param name="sPassword">
		/// Password the database key should be wrapped in.
		/// </param>
		public void wrapKey(
			string	sPassword)
		{
			RCODE rc;

			if ((rc = xflaim_Db_wrapKey( m_pDb, sPassword)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_wrapKey(
			IntPtr	pDb,
			[MarshalAs(UnmanagedType.LPStr)]
			string	sPassword);

//-----------------------------------------------------------------------------
// rollOverDbKey
//-----------------------------------------------------------------------------

		/// <summary>
		/// Generate a new database key.  All encryption definition keys will be
		/// re-wrapped in the new database key.
		/// </summary>
		public void rollOverDbKey()
		{
			RCODE	rc;

			if ((rc = xflaim_Db_rollOverDbKey( m_pDb)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_rollOverDbKey(
			IntPtr	pDb);

//-----------------------------------------------------------------------------
// getSerialNumber
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the database serial number. 
		/// </summary>
		/// <returns>
		/// Byte array containing the database serial number.  This number
		/// is generated and stored in the database when the database is created.
		/// </returns>
		public byte[] getSerialNumber()
		{
			byte[]	ucValue;

			ucValue = new byte[16];

			xflaim_Db_getSerialNumber( m_pDb, ucValue);
			return( ucValue);
		}

		[DllImport("xflaim")]
		private static extern void xflaim_Db_getSerialNumber(
			IntPtr	pDb,
			[MarshalAs(UnmanagedType.LPArray), Out] 
			byte[]	pucValue);

//-----------------------------------------------------------------------------
// getCheckpointInfo
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get information about the checkpoint thread.
		/// </summary>
		/// <returns>Returns information about what the checkpoint thread is doing.</returns>
		public XFLM_CHECKPOINT_INFO getCheckpointInfo()
		{
			XFLM_CHECKPOINT_INFO	checkpointInfo = new XFLM_CHECKPOINT_INFO();

			xflaim_Db_getCheckpointInfo( m_pDb, checkpointInfo);
			return( checkpointInfo);
		}

		[DllImport("xflaim")]
		private static extern void xflaim_Db_getCheckpointInfo(
			IntPtr					pDb,
			XFLM_CHECKPOINT_INFO	pCheckpointInfo);

//-----------------------------------------------------------------------------
// exportXML
//-----------------------------------------------------------------------------

		/// <summary>
		/// Export XML to a text file.
		/// </summary>
		/// <param name="startNode">
		/// The node in the XML document to export.  All of its sub-tree will be exported.
		/// </param>
		/// <param name="sFileName">
		/// File the XML is to be exported to.  File will be overwritten.
		/// </param>
		/// <param name="eFormat">
		/// Formatting to use when exporting.
		/// </param>
		public void exportXML(
			DOMNode				startNode,
			string				sFileName,
			eExportFormatType	eFormat)
		{
			RCODE		rc;
			IntPtr	pStartNode = (startNode != null) ? startNode.getNode() : IntPtr.Zero;

			if ((rc = xflaim_Db_exportXML( m_pDb, pStartNode, sFileName, eFormat)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_exportXML(
			IntPtr				pDb,
			IntPtr				pStartNode,
			[MarshalAs(UnmanagedType.LPStr)]
			string				sFileName,
			eExportFormatType	eFormat);

//-----------------------------------------------------------------------------
// exportXMLToString
//-----------------------------------------------------------------------------

		/// <summary>
		/// Export XML to a string.
		/// </summary>
		/// <param name="startNode">
		/// The node in the XML document to export.  All of its sub-tree will be exported.
		/// </param>
		/// <param name="eFormat">
		/// Formatting to use when exporting.
		/// </param>
		/// <returns>
		/// Returns a string containing the exported XML.
		/// </returns>
		public string exportXMLToString(
			DOMNode				startNode,
			eExportFormatType	eFormat)
		{
			RCODE		rc;
			IntPtr	pszStr;
			string	sXML;
			IntPtr	pStartNode = (startNode != null) ? startNode.getNode() : IntPtr.Zero;

			if ((rc = xflaim_Db_exportXMLToString( m_pDb, pStartNode,
						eFormat, out pszStr)) != 0)
			{
				throw new XFlaimException( rc);
			}
			sXML = Marshal.PtrToStringAnsi( pszStr);
			m_dbSystem.freeUnmanagedMem( pszStr);
			return( sXML);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_Db_exportXMLToString(
			IntPtr				pDb,
			IntPtr				pStartNode,
			eExportFormatType	eFormat,
			out IntPtr			ppszStr);

//-----------------------------------------------------------------------------
// getLockUsers
//-----------------------------------------------------------------------------

		/// <summary>
		/// Returns an array representing all of the threads that are either
		/// holding the database lock (this is always the zeroeth entry in the array)
		/// as well as threads waiting to obtain the database lock (entries
		/// 1 through N).
		/// </summary>
		/// <returns>Array of database lock holder and waiters.</returns>
		public LockUser [] getLockUsers()
		{
			RCODE							rc;
			LockInfoClientDelegate	lockInfoClientDelegate = new LockInfoClientDelegate();
			LockInfoClientCallback	fnLockInfoClient = new LockInfoClientCallback( lockInfoClientDelegate.funcLockInfoClient);

			if ((rc = xflaim_Db_getLockUsers( m_pDb, fnLockInfoClient)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( lockInfoClientDelegate.getLockUsers());
		}

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern RCODE xflaim_Db_getLockUsers(
			IntPtr						pDb,
			LockInfoClientCallback	fnLockInfoClient);

		private delegate int LockInfoClientCallback(
			int	bSetTotalLocks,
			uint	uiTotalLocks,
			uint	uiLockNum,
			uint	uiThreadId,
			uint	uiTime);
			
		private class LockInfoClientDelegate
		{
			LockUser []	m_lockUsers;

			public LockInfoClientDelegate()
			{
				m_lockUsers = null;
			}
			
			~LockInfoClientDelegate()
			{
			}

			public LockUser [] getLockUsers()
			{
				return( m_lockUsers);
			}
			
			public int funcLockInfoClient(
				int	bSetTotalLocks,
				uint	uiTotalLocks,
				uint	uiLockNum,
				uint	uiThreadId,
				uint	uiTime)
			{
				if (bSetTotalLocks != 0)
				{
					m_lockUsers = new LockUser [uiTotalLocks];
				}
				else
				{
					LockUser	lockUser = m_lockUsers [uiLockNum] = new LockUser();
					lockUser.uiThreadId = uiThreadId;
					lockUser.uiTime = uiTime;
				}
				return( 1);
			}
		}
	}
}
