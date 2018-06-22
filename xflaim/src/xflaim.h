//------------------------------------------------------------------------------
// Desc:	XFLAIM public definitions and interfaces
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

#ifndef XFLAIM_H
#define XFLAIM_H

	#include <flaimtk.h>
	
   // platform-specific API definitions for XFL* macros
   #if defined( FLM_WIN)
      #if defined( XFL_STATIC_LINK)
         #define XFLEXP
      #else
         #if defined( XFL_SOURCE)
            #define XFLEXP                __declspec(dllexport)
         #else
            #define XFLEXP                __declspec(dllimport)
         #endif
      #endif
		#define XFLAPI     						__stdcall
	#elif defined( FLM_NLM)
      #define XFLEXP
		#define XFLAPI     						__stdcall
	#elif defined( FLM_UNIX)
      #define XFLEXP
		#define XFLAPI
	#else
		#error Platform not supported
   #endif
	#define XFLXPC							      extern "C" XFLEXP

	#ifdef FLM_PACK_STRUCTS
		#pragma pack(push, 1)
	#endif

	/****************************************************************************
									Forward References
	****************************************************************************/
	flminterface IF_Backup;
	flminterface IF_DataVector;
	flminterface IF_Db;
	flminterface IF_DbInfo;
	flminterface IF_DirHdl;
	flminterface IF_DOMNode;
	flminterface IF_FileHdl;
	flminterface IF_FileSystem;
	flminterface IF_IStream;
	flminterface IF_PosIStream;
	flminterface IF_ResultSet;
	flminterface IF_Query;
	flminterface IF_ThreadInfo;
	flminterface IF_NodeInfo;
	flminterface IF_BTreeInfo;

	// These are interfaces that need to be implemented by
	// applications.  XFlaim uses them to report status or to do
	// callbacks of various kinds.

	flminterface IF_OStream;
	flminterface IF_BackupClient;
	flminterface IF_BackupStatus;
	flminterface IF_CommitClient;
	flminterface IF_DbCheckStatus;
	flminterface IF_DbCopyStatus;
	flminterface IF_DbRebuildStatus;
	flminterface IF_DbRenameStatus;
	flminterface IF_DeleteStatus;
	flminterface IF_EventClient;
	flminterface IF_IxClient;
	flminterface IF_IxStatus;
	flminterface IF_LoggerClient;
	flminterface IF_LogMessageClient;
	flminterface IF_OperandComparer;
	flminterface IF_RestoreClient;
	flminterface IF_RestoreStatus;
	flminterface IF_ResultSetSortStatus;
	flminterface IF_ResultSetCompare;
	flminterface IF_QueryStatus;
	flminterface IF_QueryValidator;
	flminterface IF_QueryValFunc;
	flminterface IF_QueryNodeSource;
	flminterface IF_UpgradeClient;
	flminterface IF_BTreeInfoStatus;

	/****************************************************************************
	Desc:    This structure is used as a parameter to dbCreate to specify
				the create options for a database.  It is also optionally returned
				when calling dbOpen.
				IMPORTANT NOTE: This needs to be kept in sync with the same
				structure that is defined in java and csharp modules.
	****************************************************************************/
	typedef struct
	{
		FLMUINT32	ui32BlockSize;
	#define XFLM_DEFAULT_BLKSIZ						4096
		FLMUINT32	ui32VersionNum;
	#define XFLM_VER_5_12								512
	#define XFLM_CURRENT_VERSION_NUM					XFLM_VER_5_12
	#define XFLM_CURRENT_VER_STR						"5.12"
		FLMUINT32	ui32MinRflFileSize;
	#define XFLM_DEFAULT_MIN_RFL_FILE_SIZE			(FLMUINT32)(((FLMUINT)100 * (FLMUINT)1024 * (FLMUINT)1024))
		FLMUINT32	ui32MaxRflFileSize;
	#define XFLM_DEFAULT_MAX_RFL_FILE_SIZE			FLM_MAXIMUM_FILE_SIZE
		FLMBOOL		bKeepRflFiles;
	#define XFLM_DEFAULT_KEEP_RFL_FILES_FLAG		FALSE
		FLMBOOL		bLogAbortedTransToRfl;
	#define XFLM_DEFAULT_LOG_ABORTED_TRANS_FLAG	FALSE
		FLMUINT32	ui32DefaultLanguage;
	#define XFLM_DEFAULT_LANG							(FLM_US_LANG)
	} XFLM_CREATE_OPTS, F_CREATE_OPTS;
	
	typedef enum
	{
		XML_NO_ERROR = 0,
		XML_ERR_BAD_ELEMENT_NAME,				// 1	Invalid element name - does not start with a valid character for element names
		XML_ERR_XMLNS_IN_ELEMENT_NAME,		// 2	Element names cannot be "xmlns" or have "xmlns:" as a prefix
		XML_ERR_ELEMENT_NAME_MISMATCH,		//	3	The element name inside the "</" does not match the element name in the opening "<"
		XML_ERR_PREFIX_NOT_DEFINED,			// 4	The prefix for the element or attribute has not been defined with an "xmlns:prefix=" attribute somewhere
		XML_ERR_EXPECTING_GT,					// 5	Expecting a '>'
		XML_ERR_EXPECTING_ELEMENT_LT,			// 6	Expecting a '<' to begin an element name
		XML_ERR_EXPECTING_EQ,					// 7	Expecting a '=' after the attribute name
		XML_ERR_MULTIPLE_XMLNS_DECLS,			// 8	Multiple "xmlns" default namespace declarations in an element
		XML_ERR_MULTIPLE_PREFIX_DECLS,		// 9	Multiple definitions for the same prefix ("xmlns:prefix=...") in an element
		XML_ERR_EXPECTING_QUEST_GT,			// 10	Expecting "?>" to terminate "<?xml" declaration
		XML_ERR_INVALID_XML_MARKUP,			// 11	Invalid XML markup, expecting "<?", "<!--", "<!ENTITY", "<!ELEMENT", "<!ATTLIST", or "<!NOTATION"
		XML_ERR_MUST_HAVE_ONE_ATT_DEF,		// 12	Must have at least one attr def in an <!ATTLIST markup
		XML_ERR_EXPECTING_NDATA,				// 13	Expecting "NDATA" keyword
		XML_ERR_EXPECTING_SYSTEM_OR_PUBLIC,	// 14	Expecting "SYSTEM" or "PUBLIC" keyword in <!NOTATION declaration
		XML_ERR_EXPECTING_LPAREN,				// 15	Expecting "("
		XML_ERR_EXPECTING_RPAREN_OR_PIPE,	// 16	Expecing ")" or "|"
		XML_ERR_EXPECTING_NAME,					// 17	Expecting a name
		XML_ERR_INVALID_ATT_TYPE,				// 18	Invalid Attr type in <!ATTLIST markup, expecting CDATA, ID, IDREF, IDREFS, ENTITY, ENTITIES, NMTOKEN, NMTOKENS, NOTATION, or (
		XML_ERR_INVALID_DEFAULT_DECL,			// 19 Invalid default decl, expecting #FIXED, #REQUIRED, #IMPLIED, or quoted attr value
		XML_ERR_EXPECTING_PCDATA,				// 20	Expecting PCDATA - only PCDATA allowed after #
		XML_ERR_EXPECTING_ASTERISK,			// 21	Expecting "*"
		XML_ERR_EMPTY_CONTENT_INVALID,		// 22	Empty content is invalid - must be parameters between parens
		XML_ERR_CANNOT_MIX_CHOICE_AND_SEQ,	// 23	Cannot mix choice items with sequenced items.
		XML_ERR_XML_ILLEGAL_PI_NAME,			// 24	"XML" is not a legal name for a processing instruction
		XML_ERR_ILLEGAL_FIRST_NAME_CHAR,		// 25	Illegal first character in name - must be an alphabetic letter or underscore
		XML_ERR_ILLEGAL_COLON_IN_NAME,		// 26	Illegal second ":" found in name.  Name already has a colon.
		XML_ERR_EXPECTING_VERSION,				// 27	Expecting "version"
		XML_ERR_INVALID_VERSION_NUM,			// 28	Invalid version number - only 1.0 is supported.
		XML_ERR_ENCODING_NOT_SUPPORTED,		// 29	Unsupported encoding - must be "UTF-8" or "us-ascii"
		XML_ERR_EXPECTING_YES_OR_NO,			// 30	Expecting "yes" or "no"
		XML_ERR_EXPECTING_QUOTE_BEFORE_EOL,	// 31	Expecting quote character - unexpected end of line
		XML_ERR_EXPECTING_SEMI,					// 32	Expecting ";"
		XML_ERR_UNEXPECTED_EOL_IN_ENTITY,	// 33 Unexpected end of line in entity reference, need proper terminating character - ";"
		XML_ERR_INVALID_CHARACTER_NUMBER,	// 34	Invalid numeric character entity.  Number is either too large, or zero, or illegal characters were used in the number.
		XML_ERR_UNSUPPORTED_ENTITY,			// 35	Unsupported predefined entity reference.
		XML_ERR_EXPECTING_QUOTE,				// 36	Expecting single or double quote character.
		XML_ERR_INVALID_PUBLIC_ID_CHAR,		// 37	Invalid character in public id.
		XML_ERR_EXPECTING_WHITESPACE,			// 38	Whitespace required
		XML_ERR_EXPECTING_HEX_DIGIT,			// 39	Expecting HEX digit for binary value
		XML_ERR_INVALID_BINARY_ATTR_VALUE,	// 40	Invalid binary value for attribute
		XML_ERR_CREATING_CDATA_NODE,			// 41 Error returned from createNode in processCDATA
		XML_ERR_CREATING_COMMENT_NODE,		// 42 Error returned from createNode in processComment
		XML_ERR_CREATING_PI_NODE,				// 43 Error returned from createNode in processPI
		XML_ERR_CREATING_DATA_NODE,			// 44 Error returned from createNode in processPI
		XML_ERR_CREATING_ROOT_ELEMENT,		// 45 Error returned from createRootElement in processSTag
		XML_ERR_CREATING_ELEMENT_NODE,		// 46 Error returned from createNode in processSTag
		XML_ERR_XML_PREFIX_REDEFINITION,		// 47 Cannot re-define xml namespace prefix 
		// IMPORTANT NOTE:  If new codes are added, please update gv_XMLParseErrors in fshell.cpp
		XML_NUM_ERRORS
	} XMLParseError;
	
	/****************************************************************************
	Desc:	 	XML import statistics
	****************************************************************************/
	
	typedef enum
	{
		XFLM_XML_UTF8_ENCODING,
		XFLM_XML_USASCII_ENCODING
	} XMLEncoding;

	// IMPORTANT NOTE: This needs to be kept in sync with the
	// corresponding types in java and C#.  In C# it is the
	// CS_XFLM_IMPORT_STATS, because we cannot guarantee the sizes
	// of enums in C++.
	typedef struct
	{
		FLMUINT			uiLines;
		FLMUINT			uiChars;
		FLMUINT			uiAttributes;
		FLMUINT			uiElements;
		FLMUINT			uiText;
		FLMUINT			uiDocuments;
		FLMUINT			uiErrLineNum;
		FLMUINT			uiErrLineOffset;	// NOTE: This is a zero-based offset
		XMLParseError	eErrorType;
		FLMUINT			uiErrLineFilePos;
		FLMUINT			uiErrLineBytes;
		XMLEncoding		eXMLEncoding;
	} XFLM_IMPORT_STATS;

	/****************************************************************************
	Desc:	 	Database header - on-disk format.
	****************************************************************************/
	typedef struct
	{
		// The following items can only be changed by a convert operation, which
		// always converts out to a different file.  A convert is NEVER done
		// "in-place."  Hence, once a database is opened, it can be assumed that
		// these items are stable - they will not change.  This is an important
		// thing for the read transaction, because we do not want to be copying
		// this entire structure on read transactions.  We only want to do that on
		// update transactions.  It will be safe for the read transaction to assume
		// that its copy of these items inside the FDB structure are current.

		FLMBYTE		szSignature[ 8];					// Contains the string "FLAIMDB"
	#define XFLM_DB_SIGNATURE						"FLAIMDB"
		FLMUINT8		ui8IsLittleEndian;				// Non-zero if DB is little-endian
	#ifdef FLM_BIG_ENDIAN
		#define XFLM_NATIVE_IS_LITTLE_ENDIAN	0
	#else
		#define XFLM_NATIVE_IS_LITTLE_ENDIAN	1
	#endif
		FLMUINT8		ui8DefaultLanguage;				// DB_DEFAULT_LANGUAGE
		FLMUINT16	ui16BlockSize;						// DB_BLOCK_SIZE

		// The following items can be changed by an update transaction.
		// NOTE: The following items should always be accessed from the FDB
		// because they are copied there by flmGetDbHdrInfo at the
		// beginning of a transaction:
		//		ui64CurrTransID			-->	FDB.ui64CurrTransID
		//		ui32LogicalEOF				-->	FDB.uiLogicalEOF
		//		ui32FirstAvailBlkAddr	-->	FDB.uiFirstAvailBlkAddr;
		// IMPORTANT NOTE: The following items cannot currently be changed
		// during an update transaction:
		//		ui32DbVersion
		//		ui16BlockSize
		//		ui8DefaultLanguage
		//		ui32MaxFileSize
		//		ui32FirstLFBlkAddr
		// This is because they are always accessed from pFile->lastCommittedDbHdr
		// regardless of transaction type.  If we ever want to change them in an
		// update transaction, we will need to modify flmGetDbHdrInfo to copy them
		// into the FDB, and then only access them from within the FDB.

		FLMUINT32	ui32DbVersion;					// LOG_FLAIM_VERSION
		FLMUINT8		ui8BlkChkSummingEnabled;	// Is block checksumming enabled?
															// Not currently used.
		FLMUINT8		ui8RflKeepFiles;				// LOG_KEEP_RFL_FILES
		FLMUINT8		ui8RflAutoTurnOffKeep;		// LOG_AUTO_TURN_OFF_KEEP_RFL
		FLMUINT8		ui8RflKeepAbortedTrans;		// LOG_KEEP_ABORTED_TRANS_IN_RFL
		FLMUINT32	ui32RflCurrFileNum;			// LOG_RFL_FILE_NUM
		FLMUINT64	ui64LastRflCommitID;			// LOG_LAST_RFL_COMMIT_ID
		FLMUINT32	ui32RflLastFileNumDeleted;	// LOG_LAST_RFL_FILE_DELETED
		FLMUINT32	ui32RflLastTransOffset;		// LOG_RFL_LAST_TRANS_OFFSET
		FLMUINT32	ui32RflLastCPFileNum;		// LOG_RFL_LAST_CP_FILE_NUM
		FLMUINT32	ui32RflLastCPOffset;			// LOG_RFL_LAST_CP_OFFSET
		FLMUINT64	ui64RflLastCPTransID;		// LOG_LAST_CP_TRANS_ID
		FLMUINT32	ui32RflMinFileSize;			// LOG_RFL_MIN_FILE_SIZE
		FLMUINT32	ui32RflMaxFileSize;			// LOG_RFL_MAX_FILE_SIZE
		FLMUINT64	ui64CurrTransID;				// LOG_CURR_TRANS_ID
		FLMUINT64	ui64TransCommitCnt;			// LOG_COMMIT_COUNT
		FLMUINT32	ui32RblEOF;						// LOG_ROLLBACK_EOF
		FLMUINT32	ui32RblFirstCPBlkAddr;		// LOG_PL_FIRST_CP_BLOCK_ADDR
		FLMUINT32	ui32FirstAvailBlkAddr;		// LOG_PF_AVAIL_BLKS
		FLMUINT32	ui32FirstLFBlkAddr;			// First logical file block.
		FLMUINT32	ui32LogicalEOF;				// LOG_LOGICAL_EOF
		FLMUINT32	ui32MaxFileSize;				// LOG_MAX_FILE_SIZE
		FLMUINT64	ui64LastBackupTransID;		// LOG_LAST_BACKUP_TRANS_ID
		FLMUINT32	ui32IncBackupSeqNum;			// LOG_INC_BACKUP_SEQ_NUM
		FLMUINT32	ui32BlksChangedSinceBackup;// LOG_BLK_CHG_SINCE_BACKUP

	#define XFLM_SERIAL_NUM_SIZE				16

		FLMBYTE		ucDbSerialNum[ XFLM_SERIAL_NUM_SIZE];
																// LOG_DB_SERIAL_NUM
		FLMBYTE		ucLastTransRflSerialNum[ XFLM_SERIAL_NUM_SIZE];
																// LOG_LAST_TRANS_RFL_SERIAL_NUM
		FLMBYTE		ucNextRflSerialNum[ XFLM_SERIAL_NUM_SIZE];
																// LOG_RFL_NEXT_SERIAL_NUM
		FLMBYTE		ucIncBackupSerialNum[ XFLM_SERIAL_NUM_SIZE];
																// LOG_INC_BACKUP_SERIAL_NUM
		FLMUINT32	ui32DbKeyLen;						// LOG_DATABASE_KEY_LEN
		
		// IMPORTANT NOTE: If anything is changed in here, need to make
		// corresponding changes to convertDbHdr routine and
		// flmVerifyDiskStructOffsets routine.

		FLMBYTE		ucReserved[ 64];					// Reserved for future
																// Always initialized to zero

		// Checksum should ALWAYS be last

		FLMUINT32	ui32HdrCRC;							// LOG_HDR_CHECKSUM

		// Encryption Key stuff
		
	#define XFLM_MAX_ENC_KEY_SIZE						256
	
		FLMBYTE			DbKey[ XFLM_MAX_ENC_KEY_SIZE];
																// LOG_DATABASE_KEY
																
		// Offsets of variables in the structure

	#define XFLM_DB_HDR_szSignature_OFFSET							0
	#define XFLM_DB_HDR_ui8IsLittleEndian_OFFSET					8
	#define XFLM_DB_HDR_ui8DefaultLanguage_OFFSET				9
	#define XFLM_DB_HDR_ui16BlockSize_OFFSET						10
	#define XFLM_DB_HDR_ui32DbVersion_OFFSET						12
	#define XFLM_DB_HDR_ui8BlkChkSummingEnabled_OFFSET			16
	#define XFLM_DB_HDR_ui8RflKeepFiles_OFFSET					17
	#define XFLM_DB_HDR_ui8RflAutoTurnOffKeep_OFFSET			18
	#define XFLM_DB_HDR_ui8RflKeepAbortedTrans_OFFSET			19
	#define XFLM_DB_HDR_ui32RflCurrFileNum_OFFSET				20
	#define XFLM_DB_HDR_ui64LastRflCommitID_OFFSET				24
	#define XFLM_DB_HDR_ui32RflLastFileNumDeleted_OFFSET		32
	#define XFLM_DB_HDR_ui32RflLastTransOffset_OFFSET			36
	#define XFLM_DB_HDR_ui32RflLastCPFileNum_OFFSET				40
	#define XFLM_DB_HDR_ui32RflLastCPOffset_OFFSET				44
	#define XFLM_DB_HDR_ui64RflLastCPTransID_OFFSET				48
	#define XFLM_DB_HDR_ui32RflMinFileSize_OFFSET				56
	#define XFLM_DB_HDR_ui32RflMaxFileSize_OFFSET				60
	#define XFLM_DB_HDR_ui64CurrTransID_OFFSET					64
	#define XFLM_DB_HDR_ui64TransCommitCnt_OFFSET				72
	#define XFLM_DB_HDR_ui32RblEOF_OFFSET							80
	#define XFLM_DB_HDR_ui32RblFirstCPBlkAddr_OFFSET			84
	#define XFLM_DB_HDR_ui32FirstAvailBlkAddr_OFFSET			88
	#define XFLM_DB_HDR_ui32FirstLFBlkAddr_OFFSET				92
	#define XFLM_DB_HDR_ui32LogicalEOF_OFFSET						96
	#define XFLM_DB_HDR_ui32MaxFileSize_OFFSET					100
	#define XFLM_DB_HDR_ui64LastBackupTransID_OFFSET			104
	#define XFLM_DB_HDR_ui32IncBackupSeqNum_OFFSET				112
	#define XFLM_DB_HDR_ui32BlksChangedSinceBackup_OFFSET		116
	#define XFLM_DB_HDR_ucDbSerialNum_OFFSET						120
	#define XFLM_DB_HDR_ucLastTransRflSerialNum_OFFSET			136
	#define XFLM_DB_HDR_ucNextRflSerialNum_OFFSET				152
	#define XFLM_DB_HDR_ucIncBackupSerialNum_OFFSET				168
	#define XFLM_DB_HDR_ui32DbKeyLen									184
	#define XFLM_DB_HDR_ucReserved_OFFSET							188
	#define XFLM_DB_HDR_ui32HdrCRC_OFFSET							252
	#define XFLM_DB_HDR_DbKey											256
	} XFLM_DB_HDR;
	
	// uiFlags values for keyRetrieve() method
	// IMPORTANT NOTE: If these change, please be sure to make
	// corresponding changes in java and C# code.

	#define XFLM_INCL							0x0010
	#define XFLM_EXCL							0x0020
	#define XFLM_EXACT						0x0040
	#define XFLM_KEY_EXACT					0x0080
	#define XFLM_FIRST						0x0100
	#define XFLM_LAST							0x0200
	#define XFLM_MATCH_IDS					0x0400
	#define XFLM_MATCH_DOC_ID				0x0800

	// Flags used by openDb method

	#define XFLM_ONLINE						0x0020
	#define XFLM_DONT_REDO_LOG				0x0040
	#define XFLM_DONT_RESUME_THREADS		0x0080
	#define XFLM_DO_LOGICAL_CHECK			0x0100
	#define XFLM_SKIP_DOM_LINK_CHECK		0x0200	// Used only in dbCheck.
	#define XFLM_ALLOW_LIMITED_MODE		0x0400

	// Maximum key size

	#define XFLM_MAX_KEY_SIZE				1024
	
	// "Wait forever" when starting a transaction
	
	#define FLM_NO_TIMEOUT					0xFF
	
	// Node types.  The order of these enums
	// must be preserved as the code sometimes
	// depends on the order to test ranges of
	// node types.
	// IMPORTANT NOTE: If any of these change or there
	// are additions, be sure to also modify the
	// corresponding java and C# definitions.

	typedef enum
	{
		INVALID_NODE =							0x00,
		DOCUMENT_NODE =						0x01,
		ELEMENT_NODE =							0x02,
		DATA_NODE =								0x03,
		COMMENT_NODE =							0x04,
		CDATA_SECTION_NODE =					0x05,
		ANNOTATION_NODE =						0x06,
		PROCESSING_INSTRUCTION_NODE =		0x07,
		ATTRIBUTE_NODE =						0x08,
		ANY_NODE_TYPE =						0xFFFF
	} eDomNodeType;

	// NOTE: The eNodeInsertLoc enum values cannot
	// be changed.  The RFL uses these values
	// when logging packets.
	// IMPORTANT NOTE: If any of these change or there
	// are additions, be sure to also modify the
	// corresponding java and C# definitions.
	
	typedef enum
	{
		XFLM_ROOT = 0,
		XFLM_FIRST_CHILD,
		XFLM_LAST_CHILD,
		XFLM_PREV_SIB,
		XFLM_NEXT_SIB,
		XFLM_ATTRIBUTE
	} eNodeInsertLoc;

	// NOTE: The order of the eDbTransType enum values
	// cannot be changed.  XFLAIM tests ranges of these
	// values.
	// IMPORTANT NOTE: If these are changed, be sure to make the
	// corresponding changes in java and C# code.
	
	typedef enum
	{
		XFLM_NO_TRANS = 0,
		XFLM_READ_TRANS,
		XFLM_UPDATE_TRANS
	} eDbTransType;

	// Transaction flags
	// IMPORTANT NOTE: If any of these are changed, the corresponding
	// java and C# code should also be updated.

	#define XFLM_DONT_KILL_TRANS		0x1
	#define XFLM_DONT_POISON_CACHE	0x2

	// Backup types

	typedef enum
	{
		// These values are stored in the header of the
		// backup, so do not change their values.
		XFLM_FULL_BACKUP = 0,
		XFLM_INCREMENTAL_BACKUP
	} eDbBackupType;

	// Encryption
	
	#define FLM_NICI_AES				0
	#define FLM_NICI_DES3			1
	#define FLM_NICI_UNDEFINED		0xFF

	#define XFLM_NICI_AES128		128
	#define XFLM_NICI_AES192		192
	#define XFLM_NICI_AES256		256
	#define XFLM_NICI_DES3X			168
	
	// Data types.

	#define XFLM_NODATA_TYPE			0
	#define XFLM_TEXT_TYPE				1
	#define XFLM_NUMBER_TYPE			2
	#define XFLM_BINARY_TYPE			3
	#define XFLM_NUM_OF_TYPES			4
	#define XFLM_UNKNOWN_TYPE			0xF

	#define XFLM_STRING_OPTION_STR						"string"
	#define XFLM_INTEGER_OPTION_STR						"integer"
	#define XFLM_BINARY_OPTION_STR						"binary"
	#define XFLM_NODATA_OPTION_STR						"nodata"
	#define XFLM_CHECKING_OPTION_STR						"checking"
	#define XFLM_PURGE_OPTION_STR							"purge"
	#define XFLM_ACTIVE_OPTION_STR						"active"
	#define XFLM_INDEX_SUSPENDED_STR						"suspended"
	#define XFLM_INDEX_OFFLINE_STR						"offline"
	#define XFLM_INDEX_ONLINE_STR							"online"
	#define XFLM_CASE_INSENSITIVE_OPTION_STR			"caseinsensitive"
	#define XFLM_DESCENDING_OPTION_STR					"sortdescending"
	#define XFLM_MISSING_HIGH_OPTION_STR				"sortmissinghigh"
	#define XFLM_MINSPACES_OPTION_STR					"minspaces"
	#define XFLM_WHITESPACE_AS_SPACE_STR				"whitespaceasspace"
	#define XFLM_IGNORE_LEADINGSPACES_OPTION_STR		"ignoreleadingspaces"
	#define XFLM_IGNORE_TRAILINGSPACES_OPTION_STR	"ignoretrailingspaces"
	#define XFLM_NOUNDERSCORE_OPTION_STR				"nounderscores"
	#define XFLM_NOSPACE_OPTION_STR						"nospaces"
	#define XFLM_NODASH_OPTION_STR						"nodashes"
	#define XFLM_VALUE_OPTION_STR							"value"
	#define XFLM_PRESENCE_OPTION_STR						"presence"
	#define XFLM_SUBSTRING_OPTION_STR					"substring"
	#define XFLM_EACHWORD_OPTION_STR						"eachword"
	#define XFLM_ABS_POS_OPTION_STR						"abspos"
	#define XFLM_METAPHONE_OPTION_STR					"metaphone"

	// Encryption Schemes

	#define XFLM_ENC_AES_OPTION_STR						"aes"
	#define XFLM_ENC_DES3_OPTION_STR						"des3"

	// Recovery actions

	typedef enum
	{
		XFLM_RESTORE_ACTION_CONTINUE = 0,	// Continue recovery
		XFLM_RESTORE_ACTION_STOP,				// Stop recovery
		XFLM_RESTORE_ACTION_SKIP,				// Skip operation (future)
		XFLM_RESTORE_ACTION_RETRY				// Retry the operation
	} eRestoreAction;

	// Events

	typedef enum
	{
		XFLM_EVENT_LOCKS,
		XFLM_EVENT_UPDATES,
		XFLM_MAX_EVENT_CATEGORIES				// Should always be last.
	} eEventCategory;

	typedef enum
	{
		XFLM_EVENT_LOCK_WAITING,
		XFLM_EVENT_LOCK_GRANTED,
		XFLM_EVENT_LOCK_SUSPENDED,
		XFLM_EVENT_LOCK_RESUMED,
		XFLM_EVENT_LOCK_RELEASED,
		XFLM_EVENT_LOCK_TIMEOUT,
		XFLM_EVENT_BEGIN_TRANS,
		XFLM_EVENT_COMMIT_TRANS,
		XFLM_EVENT_ABORT_TRANS,
		XFLM_EVENT_CREATE_NODE,
		XFLM_EVENT_MODIFY_NODE,
		XFLM_EVENT_DELETE_NODE,
		XFLM_EVENT_LINK_NODE,
		XFLM_EVENT_UNLINK_NODE,
		XFLM_EVENT_INDEXING_PROGRESS,
		XFLM_MAX_EVENT_TYPES			// Should always be last.
	} eEventType;

	// Logical files

	typedef enum
	{
		XFLM_LF_INVALID = 0,
		XFLM_LF_COLLECTION,
		XFLM_LF_INDEX
	} eLFileType;

	// Message logging

	typedef enum
	{
		XFLM_QUERY_MESSAGE,
		XFLM_TRANSACTION_MESSAGE,
		XFLM_GENERAL_MESSAGE,
		XFLM_NUM_MESSAGE_TYPES
	} eLogMessageType;

	typedef struct
	{
		FLMBOOL		bRunning;
		FLMBOOL		bForcingCheckpoint;
		FLMBOOL		bWritingDataBlocks;
		FLMUINT32	ui32RunningTime;
		FLMUINT32	ui32ForceCheckpointRunningTime;
		FLMUINT32	ui32ForceCheckpointReason;
			#define			XFLM_CP_TIME_INTERVAL_REASON			1
			#define			XFLM_CP_SHUTTING_DOWN_REASON			2
			#define			XFLM_CP_RFL_VOLUME_PROBLEM				3
		FLMUINT32	ui32LogBlocksWritten;
		FLMUINT32	ui32DataBlocksWritten;
		FLMUINT32	ui32DirtyCacheBytes;
		FLMUINT32	ui32BlockSize;
		FLMUINT32	ui32WaitTruncateTime;
	} XFLM_CHECKPOINT_INFO;

	typedef struct
	{
		FLMUINT				uiByteCount;
		FLMUINT				uiCount;
		FLMUINT				uiOldVerCount;
		FLMUINT				uiOldVerBytes;
		FLMUINT				uiCacheHits;
		FLMUINT				uiCacheHitLooks;
		FLMUINT				uiCacheFaults;
		FLMUINT				uiCacheFaultLooks;
		FLM_SLAB_USAGE		slabUsage;
	} XFLM_CACHE_USAGE;

	typedef struct
	{
		FLMUINT				uiMaxBytes;
		FLMUINT				uiTotalBytesAllocated;
		FLMUINT				uiCacheAdjustPercent;
		FLMUINT				uiCacheAdjustMin;
		FLMUINT				uiCacheAdjustMax;
		FLMUINT				uiCacheAdjustMinToLeave;
		FLMUINT				uiDirtyCount;
		FLMUINT				uiDirtyBytes;
		FLMUINT				uiNewCount;
		FLMUINT				uiNewBytes;
		FLMUINT				uiLogCount;
		FLMUINT				uiLogBytes;
		FLMUINT				uiFreeCount;
		FLMUINT				uiFreeBytes;
		FLMUINT				uiReplaceableCount;
		FLMUINT				uiReplaceableBytes;
		XFLM_CACHE_USAGE	BlockCache;
		XFLM_CACHE_USAGE	NodeCache;
		FLMBOOL				bDynamicCacheAdjust;
		FLMBOOL				bPreallocatedCache;
	} XFLM_CACHE_INFO;

	// IMPORTANT NOTE: This structure needs to be kept in sync with the
	// corresponding structures and classes in java and C#.
	// This structure should be like the XFLM_CACHE_USAGE structure,
	// except that it always keeps 64 bit values for some of the
	// members so that we can hold the value if we are on a 64
	// bit platforms where we can address more than 4 GB of memory.
	// We didn't want to modify the XFLM_CACHE_USAGE structure because
	// we didn't want the overhead of 64 bit arithmetic when keeping
	// track of cache statistics.  Hence, we made this "C#" structure
	// instead to copy data into when passing the information out
	// to C#.
	typedef struct
	{
		FLMUINT64			ui64ByteCount;
		FLMUINT64			ui64Count;
		FLMUINT64			ui64OldVerCount;
		FLMUINT64			ui64OldVerBytes;
		FLMUINT32			ui32CacheHits;
		FLMUINT32			ui32CacheHitLooks;
		FLMUINT32			ui32CacheFaults;
		FLMUINT32			ui32CacheFaultLooks;
		FLM_SLAB_USAGE		slabUsage;
	} CS_XFLM_CACHE_USAGE;

	// IMPORTANT NOTE: This structure needs to be kept in sync with the
	// corresponding structures and classes in java and C#.
	// This structure should be like the XFLM_CACHE_INFO structure,
	// except that it always keeps 64 bit values for some of the
	// members so that we can hold the value if we are on a 64
	// bit platforms where we can address more than 4 GB of memory.
	// We didn't want to modify the XFLM_CACHE_INFO structure because
	// we didn't want the overhead of 64 bit arithmetic when keeping
	// track of cache statistics.  Hence, we made this "C#" structure
	// instead to copy data into when passing the information out
	// to C#.
	typedef struct
	{
		FLMUINT64				ui64MaxBytes;
		FLMUINT64				ui64TotalBytesAllocated;
		FLMBOOL					bDynamicCacheAdjust;
		FLMUINT32				ui32CacheAdjustPercent;
		FLMUINT64				ui64CacheAdjustMin;
		FLMUINT64				ui64CacheAdjustMax;
		FLMUINT64				ui64CacheAdjustMinToLeave;
		FLMUINT64				ui64DirtyCount;
		FLMUINT64				ui64DirtyBytes;
		FLMUINT64				ui64NewCount;
		FLMUINT64				ui64NewBytes;
		FLMUINT64				ui64LogCount;
		FLMUINT64				ui64LogBytes;
		FLMUINT64				ui64FreeCount;
		FLMUINT64				ui64FreeBytes;
		FLMUINT64				ui64ReplaceableCount;
		FLMUINT64				ui64ReplaceableBytes;
		CS_XFLM_CACHE_USAGE	BlockCache;
		CS_XFLM_CACHE_USAGE	NodeCache;
		FLMBOOL					bPreallocatedCache;
	} CS_XFLM_CACHE_INFO;

	// IMPORTANT NOTE: This structure needs to stay in sync with
	// corresponding structures in java and C# code.
	typedef struct
	{
		FLMUINT64		ui64Count;						// Number of times read or
																// write operation was
																// performed.
		FLMUINT64		ui64TotalBytes;				// Total number of bytes
																// involved in the read or
																// write operations.
		FLMUINT64		ui64ElapMilli;					// Total elapsed time for the
																// read or write operations.
	} XFLM_DISKIO_STAT;

	// IMPORTANT NOTE: This structure needs to stay in sync with
	// corresponding structures in java and C# code.
	typedef struct
	{
		F_COUNT_TIME_STAT		CommittedTrans;		// Transactions committed
		F_COUNT_TIME_STAT		AbortedTrans;			// Transactions aborted
	} XFLM_RTRANS_STATS;

	// IMPORTANT NOTE: This structure needs to stay in sync with
	// corresponding structures in java and C# code.
	typedef struct
	{
		F_COUNT_TIME_STAT		CommittedTrans;		// Transactions committed
		F_COUNT_TIME_STAT		GroupCompletes;		// Group completes.
		FLMUINT64				ui64GroupFinished;	// Transactions finished in group
		F_COUNT_TIME_STAT		AbortedTrans;			// Transactions aborted
	} XFLM_UTRANS_STATS;

	// IMPORTANT NOTE: This structure needs to stay in sync with
	// corresponding structures in java and C# code.
	typedef struct
	{
		XFLM_DISKIO_STAT	BlockReads;					// Statistics on block reads.
		XFLM_DISKIO_STAT	BlockWrites;				// Statistics on Block writes.
		XFLM_DISKIO_STAT	OldViewBlockReads;		// Statistics on old view
																// block reads.
		FLMUINT32			ui32BlockChkErrs;			// Number of times we had
																// check errors reading
																// blocks.
		FLMUINT32			ui32OldViewBlockChkErrs;// Number of times we had
																// check errors reading an
																// old view of a block.
		FLMUINT32			ui32OldViewErrors;		// Number of times we had an
																// old view error when
																// reading.
	} XFLM_BLOCKIO_STATS;

	typedef struct
	{
		XFLM_BLOCKIO_STATS	RootBlockStats;		// Block I/O statistics for root
																// blocks.
		XFLM_BLOCKIO_STATS	MiddleBlockStats;		// Block I/O statistics for
																// blocks that are not root
																// blocks or leaf blocks.
		XFLM_BLOCKIO_STATS	LeafBlockStats;		// Block I/O statistics for leaf
																// blocks.
		FLMUINT64				ui64BlockSplits;		// Number of block splits that
																// have occurred in this logical
																// file.
		FLMUINT64				ui64BlockCombines;	// Number of block combines that
																// have occurred in this LFile
		FLMUINT					uiLFileNum;				// Logical file number.
		eLFileType				eLfType;					// Logical file type.
		FLMBOOL					bHaveStats;				// Flag indicating whether or
																// not there are statistics
																// for this LFILE.
	} XFLM_LFILE_STATS;

	typedef struct
	{
		char *						pszDbName;					// Database name - from pFile.
		XFLM_RTRANS_STATS			ReadTransStats;			// Read Transaction
																		// Statistics.
		XFLM_UTRANS_STATS			UpdateTransStats;			// Update Transaction
																		// Statistics.
		FLMUINT						uiLFileAllocSeq;			// Allocation sequence number
																		// for LFILE array.
		XFLM_LFILE_STATS *		pLFileStats;				// Pointer to LFILE statistics
																		// array.
		FLMUINT						uiLFileStatArraySize;	// Size of LFILE statistics
																		// array.
		FLMUINT						uiNumLFileStats;			// Number of elements in LFILE
																		// array currently in use.
		XFLM_BLOCKIO_STATS		LFHBlockStats;				// Block I/O statistics for
																		// LFH blocks.
		XFLM_BLOCKIO_STATS		AvailBlockStats;			// Block I/O statistics for
																		// AVAIL blocks.

		// Write statistics

		XFLM_DISKIO_STAT		DbHdrWrites;					// Statistics on DB header
																		// writes.
		XFLM_DISKIO_STAT		LogBlockWrites;				// Statistics on log block
																		// writes
		XFLM_DISKIO_STAT		LogBlockRestores;				// Statistics on log block
																		// restores

		// Read statistics.

		XFLM_DISKIO_STAT		LogBlockReads;					// Statistics on log block
																		// reads.
		FLMUINT					uiLogBlockChkErrs;			// Number of times we had
																		// checksum errors reading
																		// blocks from the rollback
																		// log.
		FLMUINT					uiReadErrors;					// Number of times we got read
																		// errors.
		FLMUINT					uiWriteErrors;					// Number of times we got write
																		// errors.
		// Lock statistics
		
		F_LOCK_STATS			LockStats;

		FLMBOOL					bHaveStats;						// Flag indicating whether or
																		// not there are statistics
																		// for this database.
	} XFLM_DB_STATS;

	typedef struct
	{
		XFLM_DB_STATS *	pDbStats;							// Pointer to array of database
																		// statistics.
		FLMUINT				uiDBAllocSeq;						// Allocation sequence number
																		// for database statistics.
		FLMUINT				uiDbStatArraySize;				// Size of the database statistics
																		// array.
		FLMUINT				uiNumDbStats;						// Number of elements in the
																		// database statistics array that
																		// are currently in use.
		FLMUINT				uiStartTime;						// Time we started collecting
																		// statistics.
		FLMUINT				uiStopTime;							// Time we stopped collecting
																		// statistics.
		FLMBOOL				bCollectingStats;					// Flag indicating whether or
																		// not we are currently
																		// collecting statistics.
	} XFLM_STATS;

	// IMPORTANT NOTE: If these change, please be sure to make
	// corresponding changes in java and C# code.
	typedef enum
	{
		XFLM_INDEX_ONLINE = 0,
		XFLM_INDEX_BRINGING_ONLINE,
		XFLM_INDEX_SUSPENDED
	} eXFlmIndexState;

	// IMPORTANT NOTE: If these change, please be sure to make
	// corresponding changes in java and C# code.
	typedef struct
	{
		FLMUINT64			ui64LastDocumentIndexed;// If ~0 then index is online,
																// otherwise this is the value of the last 
																// document ID that was indexed.
		FLMUINT64			ui64KeysProcessed;		// Keys processed for offline thread.
		FLMUINT64			ui64DocumentsProcessed;	// Documents processed for offline thread.
		FLMUINT64			ui64Transactions;			// Number of transactions started by the
																// indexing thread
		FLMUINT32			ui32IndexNum;				// Index number
		FLMUINT32			ui32StartTime;				// Start time of the offline process or zero.
		eXFlmIndexState	eState;
	} XFLM_INDEX_STATUS;

	/****************************************************************************
	Desc:	The following structures are used to pass data to the client via the
			IF_DbRebuildStatus interface
			IMPORTANT NOTE: If this structure changes, corresponding changes need
			to be made to the same structure in java and/or C# code.
	****************************************************************************/
	typedef struct
	{
		FLMINT32			i32DoingFlag;
	#define					REBUILD_GET_BLK_SIZ		1
	#define					REBUILD_RECOVER_DICT		2
	#define					REBUILD_RECOVER_DATA		3
		FLMBOOL			bStartFlag;
		FLMUINT64		ui64FileSize;
		FLMUINT64		ui64BytesExamined;
		FLMUINT64		ui64TotNodes;
		FLMUINT64		ui64NodesRecov;
		FLMUINT64		ui64DiscardedDocs;
	} XFLM_REBUILD_INFO;

	// IMPORTANT NOTE: If this structure changes, corresponding changes need to be made
	// to the same structure in java and/or C# code
	typedef struct
	{
		FLMINT32		i32ErrCode;						// Zero means no error is being reported
		FLMUINT32	ui32ErrLocale;
	#define				XFLM_LOCALE_NONE			0
	#define				XFLM_LOCALE_LFH_LIST		1
	#define				XFLM_LOCALE_AVAIL_LIST	2
	#define				XFLM_LOCALE_B_TREE		3
	#define				XFLM_LOCALE_INDEX			4
		FLMUINT32	ui32ErrLfNumber;
		FLMUINT32	ui32ErrLfType;
		FLMUINT32	ui32ErrBTreeLevel;
		FLMUINT32	ui32ErrBlkAddress;
		FLMUINT32	ui32ErrParentBlkAddress;
		FLMUINT32	ui32ErrElmOffset;
		FLMUINT64	ui64ErrNodeId;

		IF_DataVector *	ifpErrIxKey;
	} XFLM_CORRUPT_INFO;

	// IMPORTANT NOTE: If this structure changes, corresponding changes need to be made
	// to the same structure in java and/or C# code
	typedef struct
	{
		FLMINT32			i32CheckPhase;
	#define					XFLM_CHECK_LFH_BLOCKS		1
	#define					XFLM_CHECK_B_TREE				2
	#define					XFLM_CHECK_AVAIL_BLOCKS		3
	#define					XFLM_CHECK_RS_SORT			4
	#define					XFLM_CHECK_DOM_LINKS			5
		FLMBOOL			bStartFlag;
		FLMUINT64		ui64FileSize;
		FLMUINT32		ui32NumLFs;
		FLMUINT32		ui32CurrLF;
		FLMUINT32		ui32LfNumber;						// Logical file
		FLMUINT32		ui32LfType;
		FLMUINT64		ui64BytesExamined;
		FLMUINT32		ui32NumProblemsFixed;			// Number of corruptions repaired
		FLMUINT64		ui64NumDomNodes;					// in the current Lf
		FLMUINT64		ui64NumDomLinksVerified;		// in the current Lf
		FLMUINT64		ui64NumBrokenDomLinks;			// in the current Lf

		// Index check progress

		FLMUINT64		ui64NumKeys;						// Number of keys in the result set
		FLMUINT64		ui64NumDuplicateKeys;			// Number of duplicate keys generated
		FLMUINT64		ui64NumKeysExamined;				// Number of keys checked
		FLMUINT64		ui64NumKeysNotFound;				// Keys in index but missing from document
		FLMUINT64		ui64NumDocKeysNotFound;			// Keys in documents but missing from indexes
		FLMUINT64		ui64NumConflicts;					// # of non-corruption conflicts
	} XFLM_PROGRESS_CHECK_INFO;

	/**************************************************************************
	Desc:		STUFF FOR IF_Query interface
	IMPORTANT NOTE: If any of these change or new ones are added, corresponding
	changes should be made in the java and C# code.
	**************************************************************************/
	typedef enum
	{
		XFLM_UNKNOWN_OP					= 0,

		// NOTE: These operators MUST stay in this order - this order is assumed
		// by the precedence table - see fquery.cpp

		XFLM_AND_OP							= 1,
		XFLM_OR_OP							= 2,
		XFLM_NOT_OP							= 3,
		XFLM_EQ_OP							= 4,
		XFLM_NE_OP							= 5,
		XFLM_APPROX_EQ_OP					= 6,
		XFLM_LT_OP							= 7,
		XFLM_LE_OP							= 8,
		XFLM_GT_OP							= 9,
		XFLM_GE_OP							= 10,
		XFLM_BITAND_OP						= 11,
		XFLM_BITOR_OP						= 12,
		XFLM_BITXOR_OP						= 13,
		XFLM_MULT_OP						= 14,
		XFLM_DIV_OP							= 15,
		XFLM_MOD_OP							= 16,
		XFLM_PLUS_OP						= 17,
		XFLM_MINUS_OP						= 18,
		XFLM_NEG_OP							= 19,
		XFLM_LPAREN_OP						= 20,
		XFLM_RPAREN_OP						= 21,
		XFLM_COMMA_OP						= 22,
		XFLM_LBRACKET_OP					= 23,
		XFLM_RBRACKET_OP					= 24,

		// IMPORTANT NOTE: If operators are added after this point,
		// modify the isLegalOperator method below.

		// The following operators are only used internally.  They
		// may NOT be passed into the addOperator method.

		XFLM_EXISTS_OP						= 25,
		XFLM_RANGE_OP						= 26,
		XFLM_MATCH_OP						= 27
	} eQueryOperators;

	#define XFLM_FIRST_ARITH_OP		XFLM_BITAND_OP
	#define XFLM_LAST_ARITH_OP			XFLM_NEG_OP

	// Comparison rules for strings

	#define XFLM_COMP_CASE_INSENSITIVE			FLM_COMP_CASE_INSENSITIVE
	#define XFLM_COMP_COMPRESS_WHITESPACE		FLM_COMP_COMPRESS_WHITESPACE
		// Compress consecutive spaces to single space
	#define XFLM_COMP_NO_WHITESPACE				FLM_COMP_NO_WHITESPACE
		// Ignore all whitespace.  This and
		// COMP_COMPRESS_WHITESPACE cannot be used
		// together.
	#define XFLM_COMP_NO_UNDERSCORES				FLM_COMP_NO_UNDERSCORES
		// Convert underscores to whitespace.  NOTE: This
		// should be applied before COMP_COMPRESS_WHITESPACE
		// or COMP_NO_WHITESPACE
	#define XFLM_COMP_NO_DASHES					FLM_COMP_NO_DASHES
		// Remove all dashes
	#define XFLM_COMP_WHITESPACE_AS_SPACE		FLM_COMP_WHITESPACE_AS_SPACE
		// Convert tab, NL, and CR characters
		// to space
	#define XFLM_COMP_IGNORE_LEADING_SPACE		FLM_COMP_IGNORE_LEADING_SPACE
		// Ignore leading space characters
	#define XFLM_COMP_IGNORE_TRAILING_SPACE	FLM_COMP_IGNORE_TRAILING_SPACE
		// Ignore trailing space characters

	typedef enum
	{
		XFLM_QUERY_NOT_POSITIONED,
		XFLM_QUERY_AT_BOF,
		XFLM_QUERY_AT_FIRST,
		XFLM_QUERY_AT_FIRST_AND_LAST,
		XFLM_QUERY_POSITIONED,
		XFLM_QUERY_AT_LAST,
		XFLM_QUERY_AT_EOF
	} eQueryStates;

	typedef enum
	{
		XFLM_FUNC_xxx = 0
	} eQueryFunctions;

	// IMPORTANT NOTE: If any of these change, corresponding changes need to
	// be made in java and C# code.
	typedef enum
	{
		ROOT_AXIS = 0,
		CHILD_AXIS,
		PARENT_AXIS,
		ANCESTOR_AXIS,
		DESCENDANT_AXIS,
		FOLLOWING_SIBLING_AXIS,
		PRECEDING_SIBLING_AXIS,
		FOLLOWING_AXIS,
		PRECEDING_AXIS,
		ATTRIBUTE_AXIS,
		NAMESPACE_AXIS,
		SELF_AXIS,
		DESCENDANT_OR_SELF_AXIS,
		ANCESTOR_OR_SELF_AXIS,
		META_AXIS
	} eXPathAxisTypes;

	typedef enum 
	{
		XFLM_FALSE = 0,
		XFLM_TRUE,
		XFLM_UNKNOWN
	} XFlmBoolType;

	typedef enum QueryValueTypes
	{
		XFLM_MISSING_VAL = 0,
	
		// WARNING: Don't renumber below _VAL enums without 
		// re-doing gv_DoValAndDictTypesMatch table
	
		XFLM_BOOL_VAL = 1,		// 1		// XFlmBoolType
		XFLM_UINT_VAL,				// 2		// FLMUINT
		XFLM_UINT64_VAL,			// 3		// FLMUINT64
		XFLM_INT_VAL,				// 4		// FLMINT
		XFLM_INT64_VAL,			// 5		// FLMINT64
		XFLM_BINARY_VAL,			// 6		// FLMBYTE *
		XFLM_UTF8_VAL,				//	7		// FLMBYTE *
	
		// XFLM_PASSING_VAL passes all criteria.
	
		XFLM_PASSING_VAL = 0xFFFF
	} eValTypes;
	
	typedef enum
	{
		GET_FIRST_VAL = 0,
		GET_LAST_VAL,
		GET_NEXT_VAL,
		GET_PREV_VAL
	} ValIterator;

	// IMPORTANT NOTE: Changes to these should be synchronized to the
	// corresponding enums in java and C# code.
	typedef enum
	{
		XFLM_QOPT_NONE = 0,
		XFLM_QOPT_USING_INDEX,
		XFLM_QOPT_FULL_COLLECTION_SCAN,
		XFLM_QOPT_SINGLE_NODE_ID,
		XFLM_QOPT_NODE_ID_RANGE
	} eQOptTypes;

	typedef struct
	{
		eQOptTypes	eOptType;				// Type of optimization done
		FLMUINT		uiCost;					// Cost calculated for predicate
		FLMUINT64	ui64NodeId;				// Only valid if eOptType is
													// XFLM_QOPT_SINGLE_NODE_ID or
													// XFLM_QOPT_NODE_ID_RANGE
		FLMUINT64	ui64EndNodeId;			// Only valid if eOptType is
													// XFLM_QOPT_NODE_ID_RANGE
		FLMUINT		uiIxNum;					// Index used to execute query if
													// eOptType == QOPT_USING_INDEX
		FLMBYTE		szIxName[ 80];
		FLMUINT64	ui64KeysRead;
		FLMUINT64	ui64KeyHadDupDoc;
		FLMUINT64	ui64KeysPassed;
		FLMUINT64	ui64NodesRead;
		FLMUINT64	ui64NodesTested;
		FLMUINT64	ui64NodesPassed;
		FLMUINT64	ui64DocsRead;
		FLMUINT64	ui64DupDocsEliminated;
		FLMUINT64	ui64NodesFailedValidation;
		FLMUINT64	ui64DocsFailedValidation;
		FLMUINT64	ui64DocsPassed;
		FLMBOOL		bMustVerifyPath;		// Must verify node path.
		FLMBOOL		bDoNodeMatch;			// Node must be retrieved to exe
													// query.  Only valid if eOptType
													// is XFLM_QOPT_USING_INDEX.
		FLMBOOL		bCanCompareOnKey;		// Can we compare on index keys?  Only
													// valid if eOptType == XFLM_QOPT_USING_INDEX.
	} XFLM_OPT_INFO;

	/**************************************************************************
	*                  XFLAIM Dictionary Tag Numbers
	*
	* NOTES:
	*  1) These numbers cannot be changed for backward compatibility reasons.
	*  2) IF ANY NEW TAGS ARE INSERTED - Then you MUST change the database
	*   version number, because old databases will become invalid.....
	*
	***************************************************************************/

	// Special purpose collections
	// NOTE: Do not change the order of these definitions.  The
	// getNextCollection routine assumes they are in this order.

	// We have reserved from 65501 to 65535 for internal collections
	// These should be allocated starting from 65535 and going down.

	#define XFLM_MAX_COLLECTION_NUM			65500

	#define XFLM_MAINT_COLLECTION				65533
	#define XFLM_DATA_COLLECTION				65534
	#define XFLM_DICT_COLLECTION				65535

	FINLINE FLMBOOL isDictCollection(
		FLMUINT	uiCollectionNum)
	{
		return( (uiCollectionNum == XFLM_DICT_COLLECTION) ? TRUE : FALSE);
	}

	// Special purpose indexes
	// NOTE: Do not change the order of these definitions.  The
	// getNextIndex routine assumes they are in this order.

	// We have reserved from 65501 to 65535 for internal indexes
	// These should be allocated starting from 65535 and going down.

	#define XFLM_MAX_INDEX_NUM					65500
	#define XFLM_DICT_NUMBER_INDEX			65534
	#define XFLM_DICT_NAME_INDEX				65535

	// This is the reserved dictionary document that maintains
	// information about dictionary IDs 
	
	#define XFLM_DICTINFO_DOC_ID				((FLMUINT64)1)

	// Prefixes

	#define XFLM_MAX_PREFIX_NUM				65500

	// Encryption Defs

	#define XFLM_MAX_ENCDEF_NUM				65500

	/****************************************************************************
								Dictionary Identifiers
	****************************************************************************/
	#define XFLM_FIRST_RESERVED_ELEMENT_TAG		0xFFFFFE00
	// Special definitions - cannot actually be used for an element name ID, but
	// in indexing specifies the root tag
	#define ELM_ROOT_TAG									(XFLM_FIRST_RESERVED_ELEMENT_TAG - 1)

	#define ELM_ELEMENT_TAG								(XFLM_FIRST_RESERVED_ELEMENT_TAG +  0)
	#define ELM_ELEMENT_TAG_NAME						"element"
	#define ELM_ATTRIBUTE_TAG							(XFLM_FIRST_RESERVED_ELEMENT_TAG +  1)
	#define ELM_ATTRIBUTE_TAG_NAME					"attribute"
	#define ELM_INDEX_TAG								(XFLM_FIRST_RESERVED_ELEMENT_TAG +  2)
	#define ELM_INDEX_TAG_NAME							"Index"
	#define ELM_ELEMENT_COMPONENT_TAG				(XFLM_FIRST_RESERVED_ELEMENT_TAG +  4)
	#define ELM_ELEMENT_COMPONENT_TAG_NAME			"ElementComponent"
	#define ELM_ATTRIBUTE_COMPONENT_TAG				(XFLM_FIRST_RESERVED_ELEMENT_TAG +  5)
	#define ELM_ATTRIBUTE_COMPONENT_TAG_NAME		"AttributeComponent"
	#define ELM_COLLECTION_TAG							(XFLM_FIRST_RESERVED_ELEMENT_TAG +  6)
	#define ELM_COLLECTION_TAG_NAME					"Collection"
	#define ELM_PREFIX_TAG								(XFLM_FIRST_RESERVED_ELEMENT_TAG +  7)
	#define ELM_PREFIX_TAG_NAME						"Prefix"
	#define ELM_NEXT_DICT_NUMS_TAG					(XFLM_FIRST_RESERVED_ELEMENT_TAG +  8)
	#define ELM_NEXT_DICT_NUMS_TAG_NAME				"NextDictNums"
	#define ELM_DOCUMENT_TITLE_TAG					(XFLM_FIRST_RESERVED_ELEMENT_TAG +  9)
	#define ELM_DOCUMENT_TITLE_TAG_NAME				"DocumentTitle"
	#define ELM_INVALID_TAG								(XFLM_FIRST_RESERVED_ELEMENT_TAG +  10)
	#define ELM_INVALID_TAG_NAME						"Invalid"
	#define ELM_QUARANTINED_TAG						(XFLM_FIRST_RESERVED_ELEMENT_TAG +  11)
	#define ELM_QUARANTINED_TAG_NAME					"Quarantined"
	#define ELM_ALL_TAG									(XFLM_FIRST_RESERVED_ELEMENT_TAG +  12)
	#define ELM_ALL_TAG_NAME							"All"
	#define ELM_ANNOTATION_TAG							(XFLM_FIRST_RESERVED_ELEMENT_TAG +  13)
	#define ELM_ANNOTATION_TAG_NAME					"Annotation"
	#define ELM_ANY_TAG									(XFLM_FIRST_RESERVED_ELEMENT_TAG +  14)
	#define ELM_ANY_TAG_NAME							"Any"
	#define ELM_ATTRIBUTE_GROUP_TAG					(XFLM_FIRST_RESERVED_ELEMENT_TAG +  15)
	#define ELM_ATTRIBUTE_GROUP_TAG_NAME			"AttributeGroup"
	#define ELM_CHOICE_TAG								(XFLM_FIRST_RESERVED_ELEMENT_TAG +  16)
	#define ELM_CHOICE_TAG_NAME						"Choice"
	#define ELM_COMPLEX_CONTENT_TAG					(XFLM_FIRST_RESERVED_ELEMENT_TAG +  17)
	#define ELM_COMPLEX_CONTENT_TAG_NAME			"ComplexContent"
	#define ELM_COMPLEX_TYPE_TAG						(XFLM_FIRST_RESERVED_ELEMENT_TAG +  18)
	#define ELM_COMPLEX_TYPE_TAG_NAME				"ComplexType"
	#define ELM_DOCUMENTATION_TAG						(XFLM_FIRST_RESERVED_ELEMENT_TAG +  19)
	#define ELM_DOCUMENTATION_TAG_NAME				"Documentation"
	#define ELM_ENUMERATION_TAG						(XFLM_FIRST_RESERVED_ELEMENT_TAG +  20)
	#define ELM_ENUMERATION_TAG_NAME					"enumeration"
	#define ELM_EXTENSION_TAG							(XFLM_FIRST_RESERVED_ELEMENT_TAG +  21)
	#define ELM_EXTENSION_TAG_NAME					"extension"
	#define ELM_DELETE_TAG								(XFLM_FIRST_RESERVED_ELEMENT_TAG +  22)
	#define ELM_DELETE_TAG_NAME						"Delete"
	#define ELM_BLOCK_CHAIN_TAG						(XFLM_FIRST_RESERVED_ELEMENT_TAG +  23)
	#define ELM_BLOCK_CHAIN_TAG_NAME					"BlockChain"
	#define ELM_ENCDEF_TAG								(XFLM_FIRST_RESERVED_ELEMENT_TAG +  24)
	#define ELM_ENCDEF_TAG_NAME						"EncDef"
	#define ELM_SWEEP_TAG								(XFLM_FIRST_RESERVED_ELEMENT_TAG +	25)
	#define ELM_SWEEP_TAG_NAME							"Sweep"

	// IMPORTANT NOTE: Change this value whenever adding new reserved element tags!

	#define XFLM_LAST_RESERVED_ELEMENT_TAG			ELM_SWEEP_TAG


	#define XFLM_FIRST_RESERVED_ATTRIBUTE_TAG		0xFFFFFE00

	#define ATTR_DICT_NUMBER_TAG						(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG +  0)
	#define ATTR_DICT_NUMBER_TAG_NAME				"DictNumber"
	#define ATTR_COLLECTION_NUMBER_TAG				(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG +  1)
	#define ATTR_COLLECTION_NUMBER_TAG_NAME		"CollectionNumber"
	#define ATTR_COLLECTION_NAME_TAG					(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG +  2)
	#define ATTR_COLLECTION_NAME_TAG_NAME			"CollectionName"
	#define ATTR_NAME_TAG								(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG +  3)
	#define ATTR_NAME_TAG_NAME							"name"
	#define ATTR_TARGET_NAMESPACE_TAG				(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG +  4)
	#define ATTR_TARGET_NAMESPACE_TAG_NAME			"targetNameSpace"
	#define ATTR_TYPE_TAG								(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG +  5)
	#define ATTR_TYPE_TAG_NAME							"type"
	#define ATTR_STATE_TAG								(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG +  6)
	#define ATTR_STATE_TAG_NAME						"State"
	#define ATTR_LANGUAGE_TAG							(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG +  7)
	#define ATTR_LANGUAGE_TAG_NAME					"Language"
	#define ATTR_INDEX_OPTIONS_TAG					(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG +  8)
	#define ATTR_INDEX_OPTIONS_TAG_NAME				"IndexOptions"
	#define ATTR_INDEX_ON_TAG							(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG +  9)
	#define ATTR_INDEX_ON_TAG_NAME					"IndexOn"
	#define ATTR_REQUIRED_TAG							(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 10)
	#define ATTR_REQUIRED_TAG_NAME					"Required"
	#define ATTR_LIMIT_TAG								(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 11)
	#define ATTR_LIMIT_TAG_NAME						"Limit"
	#define ATTR_COMPARE_RULES_TAG					(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 12)
	#define ATTR_COMPARE_RULES_TAG_NAME				"CompareRules"
	#define ATTR_KEY_COMPONENT_TAG					(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 13)
	#define ATTR_KEY_COMPONENT_TAG_NAME				"KeyComponent"
	#define ATTR_DATA_COMPONENT_TAG					(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 14)
	#define ATTR_DATA_COMPONENT_TAG_NAME			"DataComponent"
	#define ATTR_LAST_DOC_INDEXED_TAG				(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 15)
	#define ATTR_LAST_DOC_INDEXED_TAG_NAME			"LastDocumentIndexed"
	#define ATTR_NEXT_ELEMENT_NUM_TAG				(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 16)
	#define ATTR_NEXT_ELEMENT_NUM_TAG_NAME			"NextElementNum"
	#define ATTR_NEXT_ATTRIBUTE_NUM_TAG				(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 17)
	#define ATTR_NEXT_ATTRIBUTE_NUM_TAG_NAME		"NextAttributeNum"
	#define ATTR_NEXT_INDEX_NUM_TAG					(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 18)
	#define ATTR_NEXT_INDEX_NUM_TAG_NAME			"NextIndexNum"
	#define ATTR_NEXT_COLLECTION_NUM_TAG			(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 19)
	#define ATTR_NEXT_COLLECTION_NUM_TAG_NAME		"NextCollectionNum"
	#define ATTR_NEXT_PREFIX_NUM_TAG					(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 20)
	#define ATTR_NEXT_PREFIX_NUM_TAG_NAME			"NextPrefixNum"
	#define ATTR_SOURCE_TAG								(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 21)
	#define ATTR_SOURCE_TAG_NAME						"Source"
	#define ATTR_STATE_CHANGE_COUNT_TAG				(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 22)
	#define ATTR_STATE_CHANGE_COUNT_TAG_NAME		"StateChangeCount"
	#define ATTR_XMLNS_TAG								(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 23)
	#define ATTR_XMLNS_TAG_NAME						"xmlns"
	#define ATTR_ABSTRACT_TAG							(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 24)
	#define ATTR_ABSTRACT_TAG_NAME					"abstract"
	#define ATTR_BASE_TAG								(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 25)
	#define ATTR_BASE_TAG_NAME							"base"
	#define ATTR_BLOCK_TAG								(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 26)
	#define ATTR_BLOCK_TAG_NAME						"block"
	#define ATTR_DEFAULT_TAG							(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 27)
	#define ATTR_DEFAULT_TAG_NAME						"default"
	#define ATTR_FINAL_TAG								(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 28)
	#define ATTR_FINAL_TAG_NAME						"final"
	#define ATTR_FIXED_TAG								(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 29)
	#define ATTR_FIXED_TAG_NAME						"fixed"
	#define ATTR_ITEM_TYPE_TAG							(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 30)
	#define ATTR_ITEM_TYPE_TAG_NAME					"itemtype"
	#define ATTR_MEMBER_TYPES_TAG						(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 31)
	#define ATTR_MEMBER_TYPES_TAG_NAME				"membertypes"
	#define ATTR_MIXED_TAG								(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 32)
	#define ATTR_MIXED_TAG_NAME						"mixed"
	#define ATTR_NILLABLE_TAG							(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 33)
	#define ATTR_NILLABLE_TAG_NAME					"nillable"
	#define ATTR_REF_TAG									(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 34)
	#define ATTR_REF_TAG_NAME							"ref"
	#define ATTR_USE_TAG									(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 35)
	#define ATTR_USE_TAG_NAME							"use"
	#define ATTR_VALUE_TAG								(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 36)
	#define ATTR_VALUE_TAG_NAME						"value"
	#define ATTR_ADDRESS_TAG							(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 37)
	#define ATTR_ADDRESS_TAG_NAME						"address"
	#define ATTR_XMLNS_XFLAIM_TAG						(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 38)
	#define ATTR_XMLNS_XFLAIM_TAG_NAME				"xmlns:xflaim"
	#define ATTR_ENCRYPTION_KEY_TAG					(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 39)
	#define ATTR_ENCRYPTION_KEY_TAG_NAME			"Key"
	#define ATTR_TRANSACTION_TAG						(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 40)
	#define ATTR_TRANSACTION_TAG_NAME				"Transaction"
	#define ATTR_NEXT_ENCDEF_NUM_TAG					(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 41)
	#define ATTR_NEXT_ENCDEF_NUM_TAG_NAME			"NextEncDefNum"
	#define ATTR_ENCRYPTION_ID_TAG					(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 42)
	#define ATTR_ENCRYPTION_ID_TAG_NAME				"encId"
	#define ATTR_ENCRYPTION_KEY_SIZE_TAG			(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 43)
	#define ATTR_ENCRYPTION_KEY_SIZE_TAG_NAME		"keySize"
	#define ATTR_UNIQUE_SUB_ELEMENTS_TAG			(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG + 44)
	#define ATTR_UNIQUE_SUB_ELEMENTS_TAG_NAME		"UniqueSubElements"

	// IMPORTANT NOTE: Change this value whenever adding new reserved attribute tags!

	#define XFLM_LAST_RESERVED_ATTRIBUTE_TAG		ATTR_UNIQUE_SUB_ELEMENTS_TAG

	// max element number is first reserved -2 instead of -1 because we don't want
	// anyone using ELM_ROOT_TAG either.

	#define XFLM_MAX_ELEMENT_NUM						(XFLM_FIRST_RESERVED_ELEMENT_TAG - 2)
	#define XFLM_MAX_ATTRIBUTE_NUM					(XFLM_FIRST_RESERVED_ATTRIBUTE_TAG - 1)

	// Types of metadata available on DOM nodes - can index on these.
	// Also, can search on these via the META_AXIS

	#define XFLM_META_NODE_ID							1
	#define XFLM_META_DOCUMENT_ID						2
	#define XFLM_META_PARENT_ID						3
	#define XFLM_META_FIRST_CHILD_ID					4
	#define XFLM_META_LAST_CHILD_ID					5
	#define XFLM_META_NEXT_SIBLING_ID				6
	#define XFLM_META_PREV_SIBLING_ID				7
	#define XFLM_META_VALUE								8

	#define XFLM_INI_CACHE							"cache"
	#define XFLM_INI_CACHE_ADJUST_INTERVAL		"cacheadjustinterval"
	#define XFLM_INI_CACHE_CLEANUP_INTERVAL	"cachecleanupinterval"
	#define XFLM_INI_MAX_DIRTY_CACHE				"maxdirtycache"
	#define XFLM_INI_LOW_DIRTY_CACHE				"lowdirtycache"

	// Defaults for certain other settable items in the IF_DbSystem
	
	#define XFLM_DEFAULT_MAX_CP_INTERVAL					180
	#define XFLM_DEFAULT_CACHE_ADJUST_PERCENT				70
	#define XFLM_DEFAULT_CACHE_ADJUST_MIN					(16 * 1024 * 1024)
	#define XFLM_DEFAULT_CACHE_ADJUST_MAX					0xE0000000
	#define XFLM_DEFAULT_CACHE_ADJUST_MIN_TO_LEAVE		0
	#define XFLM_DEFAULT_CACHE_ADJUST_INTERVAL			15
	#define XFLM_DEFAULT_CACHE_CLEANUP_INTERVAL			15
	#define XFLM_DEFAULT_UNUSED_CLEANUP_INTERVAL			2
	#define XFLM_DEFAULT_MAX_UNUSED_TIME					120
	#define XFLM_DEFAULT_FILE_EXTEND_SIZE					(8192 * 1024)
	#define XFLM_MIN_BLOCK_SIZE								4096
	#define XFLM_MAX_BLOCK_SIZE								8192
	#define XFLM_DEFAULT_OPEN_THRESHOLD						100		// 100 file handles to cache
	#define XFLM_DEFAULT_MAX_AVAIL_TIME						900		// 15 minutes
	#define XFLM_DEFAULT_REHASH_BACKOFF_INTERVAL			60			// 1 minute

	/**
	 * @brief The IF_DbSystem is actually an interface definition that provides public access
	 * to the XFlaim database environment.
	 *
	 * This class represents the XFlaim  database system. It extends XFLMIUnknown.
	 * The IF_DbSystem interface is used to represent the actual database system object.
	 * The database system object incorporates functionality to startup, shutdown, create,
	 * open and copy an XFlaim database.  Note that the database system when XFlaim is
	 * running is more than the files that store the data.  The database system also
	 * includes all of the in-memory structures and objects that are used, either
	 * intermittently or throughout the life of the database system.  The class id for
	 * this interface is CLSID_F_DbSystemFactory and the interface id is IID_IF_DbSystem.
	 */
	flminterface XFLEXP IF_DbSystem : public IF_Object
	{
		virtual RCODE XFLAPI updateIniFile(
			const char *	pszParamName,
			const char *	pszValue) = 0;
		
		/**
		 * @brief Return an IF_FileSystem object for performing file system operations.
		 *
		 * The getFileSystem method returns an IF_FileSystem object that can be used to
		 * perform various operations on files.
		 *
		 * @param ppFileSystem A pointer to a file system object that can 
		 * be used to perform various operations on files.
		 */
		virtual void XFLAPI getFileSystem(
			IF_FileSystem **		ppFileSystem) = 0;

		/**
		 * @brief Creates a new database.
		 *
		 * A pointer to a database object (IF_Db) is returned in the ppDb parameter. The database system
		 * engine must first be started using the  init method.  After creating a new XFlaim database,
		 * the database is open and ready to use.
		 *
		 * @param pszDbFileName Name of the control file for the database.  See the XFlaim Concepts/Database
		 * Files for a discussion on the different database files, including the control file.
		 * @param pszDataDir The directory where the data files are to be created.   If a NULL is passed in,
		 * data files will be created in the same directory as the control file (as specified by the
		 * pszDbFileName parameter).  See the XFlaim Concepts/Database Files for a discussion on the different
		 * database files.
		 * @param pszRflDir The directory where the RFL (roll forward log) files are to be located.  If a
		 * NULL is passed, roll forward log files will reside in the same directory as the control file
		 * (as specified by the pszDbFileName parameter).  See the XFlaim Concepts/Database Files for a
		 * discussion on the different database files, including the rfl (Roll-forward log) file.
		 * @param pszDictFileName Name of file containing dictionary definitions to be imported into the
		 * dictionary collection during database creation.  Note that this parameter is ignored if pzDictBuf
		 * is non-NULL.  See the XFlaim Concepts section for more information about the XFlaim Dictionary.
		 * @param pszDictBuf Null-terminated string buffer containing dictionary definitions in external
		 * XML format. If the value of this parameter is NULL, pszDictFileName will be used.  If both
		 * pszDictFileName and pszDictBuf are NULL, the database will be created with an empty dictionary.
		 * See the XFlaim Concepts section for more information about the XFlaim Dictionary.
		 * @param pCreateOpts Create options for the database. All members of the structure should be
		 * initialized to specify options desired when the database is created.  If NULL is passed as
		 * the value of this parameter, default options will be used.  See the glossary for a description
		 * of the XFLM_CREATE_OPTS structure and the default value for each field.
		 * @param ppDb A pointer to a database object that references the newly created database.
		 * @return RCODE
		 */
		virtual RCODE XFLAPI dbCreate(
			const char *			pszDbFileName,
			const char *			pszDataDir,
			const char *			pszRflDir,
			const char *			pDictFileName,
			const char *			pszDictBuf,
			XFLM_CREATE_OPTS *	pCreateOpts,
			IF_Db **					ppDb) = 0;

		/**
		 * @brief Opens an existing database.
		 *
		 * The dbOpen method opens an existing database.  A pointer to a database object (IF_Db) is returned
		 * in the ppDb parameter.  The database system engine must first be started using the  init method.
		 * After opening an XFlaim database, the database is ready to use.
		 *
		 * @param pszDbFileName Name of the control file for the database that is to be opened.  See the
		 * XFlaim Concepts/Database Files for a discussion on the different database files, including the control file.
		 * @param pszDataDir The directory where the data files are located. If a NULL is passed in,
		 * it is assumed that the data files are located in the same directory as the control file
		 * (as specified by the pszDbFileName parameter). See the XFlaim Concepts/Database Files for a
		 * discussion on the different database files.
		 * @param pszRflDir The directory where the RFL (roll forward log) files are located.  If a NULL
		 * is passed, roll forward log files are assumed to reside in the same directory as the control file
		 * (as specified by the pszDbFileName parameter).  See the XFlaim Concepts/Database Files for a
		 * discussion on the different database files, including the rfl (Roll-forward log) file.
		 * @param ppDb A pointer to a database object that references the newly created database.
		 * @return RCODE
		 */
		virtual RCODE XFLAPI dbOpen(
			const char *			pszDbFileName,
			const char *			pszDataDir,
			const char *			pszRflDir,
			const char *			pszPassword,
			FLMBOOL					bAllowLimited,
			IF_Db **					ppDb) = 0;

		/**
		 * @brief Rebuilds a database.
		 *
		 * The dbRebuild method will attempt to reconstruct a database, recovering everything that it can
		 * from the blocks of the database.
		 *
		 * @param pszSourceDbFileName Name of the source control file for the database that is to be rebuilt.
		 * See the XFlaim Concepts/Database Files for a discussion on the different database files,
		 * including the control file.
		 * @param pszSourceDataDir The directory where the data files are located. If a NULL is passed in,
		 * it is assumed that the data files are located in the same directory as the control file
		 * (as specified by the pszSourceDbFileName parameter). See the XFlaim Concepts/Database Files
		 * for a discussion on the different database files.
		 * @param pszDestDbFileName Name of the destination control file for the recovered database.
		 * See the XFlaim Concepts/Database Files for a discussion on the different database files,
		 * including the control file.
		 * @param pszDestDataDir  The directory where the data files are to be located. If a NULL is
		 * passed in, it is assumed that the data files are to be located in the same directory as
		 * the control file (as specified by the pszDestDbFileName parameter). See the XFlaim
		 * Concepts/Database Files for a discussion on the different database files.
		 * @param pszDestRflDir  The directory where the RFL (roll forward log) files of the
		 * destination database are to be created.  If a NULL is passed, roll forward log files are
		 * assumed to reside in the same directory as the control file (as specified by the
		 * pszDestDbFileName parameter).  See the XFlaim Concepts/Database Files for a discussion
		 * on the different database files, including the rfl (Roll-forward log) file.
		 * @param pszDictFileName  The name of a file containing the dictionary definitions that
		 * will be used when rebuilding the database.  A NULL may be passed in this parameter.
		 * In addition to using the definitions specified in this file, dbRebuild will attempt
		 * to recover any additional dictionary entries from the dictionary collection.
		 * See XFlaim Concepts / Dictionary for a discussion on the dictionary in XFlaim.
		 * @param pCreateOpts  Create options for the database. All members of the structure should
		 * be initialized to specify options desired when the database is created.  If NULL is passed
		 * as the value of this parameter, default options will be used.  See the glossary for a description
		 * of the XFLM_CREATE_OPTS structure and the default value for each field.
		 * @param pDbRebuild  A pointer to an application defined status reporting object.  Methods
		 * on this object are used by dbRebuild to report progress during the rebuild.  A NULL may be
		 * passed in this parameter.  This object is NOT implemented by XFlaim, but is implemented by
		 * the application.  The application must create a class that inherits from the IF_DbRebuildStatus
		 * interface and implements the pure virtual methods of that interface.
		 * @param pui64TotalNodes The total number of DOM nodes.
		 * @param pui64NodesRecovered The total number of DOM nodes recovered.
		 * @param pui64NodesDiscardedDocs The total number of documents that couldn't be recovered.
		 * @return RCODE
		 */
		virtual RCODE XFLAPI dbRebuild(
			const char *			pszSourceDbPath,
			const char *			pszSourceDataDir,
			const char *			pszDestDbPath,
			const char *			pszDestDataDir,
			const char *			pszDestRflDir,
			const char *			pszDictPath,
			const char *			pszPassword,
			XFLM_CREATE_OPTS *	pCreateOpts,
			FLMUINT64 *				pui64TotNodes,
			FLMUINT64 *				pui64NodesRecov,
			FLMUINT64 *				pui64DiscardedDocs,
			IF_DbRebuildStatus *	pDbRebuild) = 0;

		/**
		 * @brief Checks a database for corruptions.
		 *
		 * The dbCheck method will check the database for corruptions ans in certain cases it can repair them.
		 *
		 * @param pszDbFileName Name of the control file for the database that is to be checked.
		 * See the XFlaim Concepts/Database Files for a discussion on the different database files,
		 * including the control file.
		 * @param pszDataDir The directory where the data files are located.   If a NULL is passed in,
		 * it is assumed that the data files are located in the same directory as the control file
		 * (as specified by the pszDbFileName parameter). See the XFlaim Concepts/Database Files for a
		 * discussion on the different database files.
		 * @param pszRflDir  The directory where the RFL (roll forward log) files are located. If a NULL is
		 * passed, roll forward log files are assumed to reside in the same directory as the control file
		 * (as specified by the pszDbFileName parameters).  See the XFlaim Concepts/Database Files for a
		 * discussion on the different database files, including the rfl (Roll-forward log) file.
		 * @param pDbCheckStatus  A pointer to an application defined status reporting object. Methods
		 * on this object are used by dbCheck to report progress and corruptions to the calling application.
		 * A NULL may be passed in this parameter.  This object is NOT implemented by XFlaim, but is
		 * implemented by the application.  The application must create a class that inherits from the
		 * IF_DbCheckStatus  interface and implements the pure virtual methods of that interface.
		 * @param ppDbInfo If a non-NULL ppDbInfo pointer is passed, an IF_DbInfo object will be returned
		 * that contains detailed statistical information about the various B-trees in the database.
		 * The information includes things like percent utilization of various blocks at each level in
		 * a B-tree, number of keys, etc.  Methods of the  IF_DbInfo object provide for retrieval of this information.
		 * @return RCODE
		 */
		virtual RCODE XFLAPI dbCheck(
			const char *			pszDbFileName,
			const char *			pszDataDir,
			const char *			pszRflDir,
			const char *			pszPassword,			
			FLMUINT					uiFlags,
			IF_DbInfo **			ppDbInfo,
			IF_DbCheckStatus *	pDbCheckStatus) = 0;

		/**
		 * @brief Copies a database.
		 *
		 * Copies a database to a new database.  The destination database will be created if it
		 * does not exist and overwritten if it does exist.
		 *
		 * @param pszSrcDbName Name of the control file for the database that is to be copied.
		 * See the XFlaim Concepts/Database Files for a discussion on the different database files,
		 * including the control file.
		 * @param pszSrcDataDir The directory where the source database's data files are located.
		 * If a NULL is passed in, it is assumed that the source database's data files are located in
		 * the same directory as the source database's control file (as specified by the pszSrcDbName
		 * parameter).  See the XFlaim Concepts/Database Files for a discussion on the different database files.
		 * @param pszSrcRflDir The directory where the RFL (roll forward log) files of the source database
		 * are located.  If a NULL is passed, the roll forward log files of the source database are assumed
		 * to reside in the same directory as the source database's control file (as specified by the pszSrcDbName
		 * parameter).  See the XFlaim Concepts/Database Files for a discussion on the different database files,
		 * including the rfl (Roll-forward log) file.
		 * @param pszDestDbName Name of the control file for the destination database.  If the destination
		 * database already exists, it will be overwritten.  See the XFlaim Concepts/Database Files for a
		 * discussion on the different database files, including the control file.
		 * @param pszDestDataDir The directory where the destination database's data files are to be stored.
		 * If a NULL is passed in, the destination database's data files will be put in the same directory
		 * as the destination database's control file (as specified by the pszDestDbName parameter).
		 * See the XFlaim Concepts/Database Files for a discussion on the different database files.
		 * @param pszDestRflDir The directory where the RFL (roll forward log) files of the destination database
		 * are to be put.  If a NULL is passed, the roll forward log files of the destination database will be
		 * put in the same directory as the destination database's control file (as specified by the
		 * pszDestDbName parameter).  See the XFlaim Concepts/Database Files for a discussion on the different
		 * database files, including the rfl (Roll-forward log) file.
		 * @param ifpStatus A pointer to an application defined status reporting object.  Methods on this object
		 * are used by dbCopy to report progress during the copy.  A NULL may be passed in this parameter.
		 * This object is NOT implemented by XFlaim, but is implemented by the application.  The application
		 * must create a class that inherits from the IF_DbCopyStatus interface and implements the pure virtual
		 * methods of that interface.  Those methods may be called by dbCopy to report copy progress.
		 * @return RCODE
		 */
		virtual RCODE XFLAPI dbCopy(
			const char *			pszSrcDbName,
			const char *			pszSrcDataDir,
			const char *			pszSrcRflDir,
			const char *			pszDestDbName,
			const char *			pszDestDataDir,
			const char *			pszDestRflDir,
			IF_DbCopyStatus *		ifpStatus) = 0;

		/**
		 * @brief Renames a database.
		 *
		 * This method will rename an existing database to a specified new name.
		 *
		 * @param pszDbName Name of the control file for the database that is to be renamed.
		 * See the XFlaim Concepts/Database Files for a discussion on the different database files,
		 * including the control file.
		 * @param pszDataDir The directory where the database's data files are located. If a NULL
		 * is passed in, it is assumed that the data files are located in the same directory as the
		 * control file (as specified by the pszDbName parameter).  See the XFlaim Concepts/Database
		 * Files for a discussion on the different database files.
		 * @param pszRflDir The directory where the RFL (roll forward log) files of the database
		 * are located.  If a NULL is passed, the roll forward log files of the database are assumed
		 * to reside in the same directory as the control file (as specified by the pszDbName parameter).
		 * See the XFlaim Concepts/Database Files for a discussion on the different database files,
		 * including the RFL (Roll-forward log) file.
		 * @param pszNewDbName Name the control file is to be renamed to.  This name also determines
		 * what data files and RFL files will be renamed to.  See the XFlaim Concepts/Database Files
		 * for a discussion on the different database files, including the control file.  Note that the
		 * directory of the new database name must be the same as the directory specified in pszDbName.
		 * @param bOverwriteDestOk If pszNewDbName specifies the name of a file that already exists,
		 * this flag indicates whether that file should be deleted so that the rename can proceed.
		 * If FALSE, the rename will fail.
		 * @param ifpStatus A pointer to an application defined status reporting object.  Methods on
		 * this object are used by dbRename to report progress during the rename.  A NULL may be passed
		 * in this parameter.  This object is NOT implemented by XFlaim, but is implemented by the
		 * application.  The application must create a class that inherits from the IF_DbRenameStatus
		 * interface and implements the pure virtual methods of that interface.
		 * @return RCODE
		 */
		virtual RCODE XFLAPI dbRename(
			const char *			pszDbName,
			const char *			pszDataDir,
			const char *			pszRflDir,
			const char *			pszNewDbName,
			FLMBOOL					bOverwriteDestOk,
			IF_DbRenameStatus *	ifpStatus) = 0;

		/**
		 * @brief Removes a database.
		 *
		 * This method will remove (delete) an existing database.
		 *
		 * @param pszDbName Name of the control file for the database that is to be removed.
		 * See the XFlaim Concepts/Database Files for a discussion on the different database
		 * files, including the control file.
		 * @param pszDataDir The directory where the database's data files are located. If a
		 * NULL is passed in, it is assumed that the data files are located in the same directory
		 * as the control file (as specified by the pszDbName parameter).  See the XFlaim
		 * Concepts/Database Files for a discussion on the different database files.
		 * @param pszRflDir The directory where the RFL (roll forward log) files of the database are
		 * located.  If a NULL is passed, the roll forward log files of the database are assumed to
		 * reside in the same directory as the control file (as specified by the pszDbName parameter).
		 * See the XFlaim Concepts/Database Files for a discussion on the different database files,
		 * including the RFL (Roll-forward log) file.
		 * @param bRemoveRflFiles A flag that indicate whether or not the RFL file(s) should be removed as well.
		 * @return RCODE
		 */
		virtual RCODE XFLAPI dbRemove(
			const char *			pszDbName,
			const char *			pszDataDir,
			const char *			pszRflDir,
			FLMBOOL					bRemoveRflFiles) = 0;

		/**
		 * @brief Restores a database from a backup set.
		 *
		 * This method will restore a database from a backup set.  The client is responsible for
		 * providing an implementation of the IF_RestoreClient and IF_RestoreStatus interfaces.
		 *
		 * @param pszDbPath Name of the control file (including path) for the database that is to
		 * be restored.  See the XFlaim Concepts/Database Files for a discussion on the different
		 * database files, including the control file.
		 * @param pszDataDir  	The directory where the database's data files are located. If a
		 * NULL is passed in, it is assumed that the data files are located in the same directory
		 * as the control file (as specified by the pszDbName parameter).  See the XFlaim
		 * Concepts/Database Files for a discussion on the different database files.
		 * @param pszBackupPath The path to the backup file set.
		 * @param pszRflDir The directory where the RFL (roll forward log) files of the database
		 * are located.  If a NULL is passed, the roll forward log files of the database are
		 * assumed to reside in the same directory as the control file (as specified by the
		 * pszDbName parameter).  See the XFlaim Concepts/Database Files for a discussion on
		 * the different database files, including the RFL (Roll-forward log) file.
		 * @param pRestoreObj A pointer to an application defined restore client object.  Methods
		 * on this object are used by dbRestore to read the backed-up data from the location it is
		 * stored in.  The application is responsible for implementing this object.  In this way,
		 * an application can have a database restored from virtually any media.
		 * If a NULL is passed in this parameter, pszBackupPath is used to restore from.
		 * Visit: Need to say something about the default backup file set.
		 * @param pRestoreStatus A pointer to an application defined restore status object.
		 * Methods on this object are used by dbRestore to report progress during the restore.
		 * The application is responsible for implementing this object.
		 * @return RCODE
		 */
		virtual RCODE XFLAPI dbRestore(
			const char *			pszDbPath,
			const char *			pszDataDir,
			const char *			pszRflDir,
			const char *			pszBackupPath,
			const char *			pszPassword,
			IF_RestoreClient *	pRestoreObj,
			IF_RestoreStatus *	pRestoreStatus) = 0;

		/**
		 * @brief Duplicates the IF_Db interface object.
		 *
		 * @par
		 * This method will duplicate the IF_Db interface object.  It is similar to the dbOpen method,
		 * except that it doesn't take a database file name as input, but takes an already open IF_Db
		 * object as input.  The returned IF_Db object is the same as if it had been created by the dbOpen
		 * method.  However, calling dbDup is slightly more efficient than calling dbOpen.
		 * @par
		 * Since IF_Db objects cannot be shared between threads (they are not thread-safe), an application
		 * would normally have each of its threads call dbOpen to obtain an IF_Db object.  The dbDup
		 * method provides an alternative way to create IF_Db objects for multiple threads.
		 * For example, one thread could call dbOpen to obtain the first IF_Db object.  Thereafter,
		 * it could call dbDup and pass the created IF_Db objects to different threads.
		 * @par
		 * Although the most common use of dbDup is to create IF_Db objects and pass them to different
		 * threads, it is conceivable that one thread might want multiple IF_Db objects on the same
		 * database.  The uses for this are probably somewhat exotic.  For example, a single thread
		 * using multiple IF_Db objects could have multiple different transactions open simultaneously.
		 *
		 * @param pDb The database object to be duplicated.
		 * @param ppDb A new database object.
		 * @return RCODE
		 */
		virtual RCODE XFLAPI dbDup(
			IF_Db *					pDb,
			IF_Db **					ppDb) = 0;

		/**
		 * @brief Converts a corruption code to a string equivalent.
		 *
		 * This method converts a corruption code to a string equivalent that can then be printed in
		 * a diagnostic log or some other human readable output.
		 *
		 * @param iErrCode The error code to be translated.
		 * @return const char *
		 */
		virtual const char * XFLAPI checkErrorToStr(
			FLMINT	iCheckErrorCode) = 0;

		/**
		 * @brief Opens a buffered input stream.
		 *
		 * This method opens a buffered input stream.  The pucBuffer buffer holds that data that is to be
		 * streamed through the IF_PosIStream object.  NOTE: The returned IF_PosIStream object's Release()
		 * method should be called when the application is done using the object.
		 *
		 * @param pucBuffer The buffer that will be associated with the input stream.
		 * @param uiLength The size of the buffer (bytes).
		 * @param ppIStream The input stream object used to read the data in.
		 * @return RCODE
		 */
		virtual RCODE XFLAPI openBufferIStream(
			const char *			pucBuffer,
			FLMUINT					uiLength,
			IF_PosIStream **		ppIStream) = 0;

		/**
		 * @brief Opens a file input stream.
		 *
		 * This method opens a file input stream.  The pszPath parameter points to a file that holds the data
		 * that is to be streamed through the IF_PosIStream object.  NOTE: The returned IF_PosIStream object's
		 * Release() method should be called when the application is done using the object.
		 *
		 * @param pszPath The name of file whose data is to be read via the IF_PosIStream object.
		 * @param ppIStream The input stream object used to read the data in.
		 * @return RCODE
		 */
		virtual RCODE XFLAPI openFileIStream(
			const char *			pszPath,
			IF_PosIStream **		ppIStream) = 0;

		/**
		 * @brief Open a multi-file input stream.
		 *
		 * (*ppIStream)->read() will read data from the files in the directory (pszDirectory)
		 * that match the base name given (pszBaseName).  EOF is returned when there are no
		 * more files to read from.  File names start with pszBaseName, then
		 * pszBaseName.00000001, pszBaseName.00000002, etc.  The extension is a hex number.
		 */
		virtual RCODE XFLAPI openMultiFileIStream(
			const char *			pszDirectory,
			const char *			pszBaseName,
			IF_IStream **			ppIStream) = 0;
			
		/**
		 * @brief Convert an input stream (pIStream) into a buffered input stream (*ppIStream)
		 *
		 * When (*ppIStream)->read() is called, it will fill an internal buffer of
		 * uiBufferSize by reading from pIStream.  Data is returned from the buffer until
		 * the buffer is emptied, at which time another read will be done from pIStream,
		 * until pIStream has no more data to return.  This method allows any input stream
		 * to be turned into a buffered stream.
		 */
		virtual RCODE XFLAPI openBufferedIStream(
			IF_IStream *			pIStream,
			FLMUINT					uiBufferSize,
			IF_IStream **			ppIStream) = 0;

		/**
		 * @brief Read uncompressed data from a compressed stream (pIStream)
		 *
		 * When (*ppIStream)->read() is called, it will read and uncompress data from 
		 * pIStream.
		 */
		virtual RCODE XFLAPI openUncompressingIStream(
			IF_IStream *			pIStream,
			IF_IStream **			ppIStream) = 0;
			
		// METHODS FOR GETTING OUTPUT STREAM OBJECTS
			
		/**
		 * @brief Create a file output stream.
		 *
		 * Data is written out to the specified file.  The file may be created, overwritten, or
		 * appended to, depending on iAccessFlags.
		 */
		virtual RCODE XFLAPI openFileOStream(
			const char *		pszFileName,
			FLMBOOL				bTruncateIfExists,
			IF_OStream **		ppOStream) = 0;
			
		/**
		 * @brief Create a multi-file output stream.
		 *
		 * Data is written to the specified directory, creating files using the given
		 * base file name.  When a file reaches the specified size, another file will
		 * be created by appending a suffix with an incrementing HEX number.  The
		 * bOverwrite flag indicates whether to overwrite files that already exist.
		 */
		virtual RCODE XFLAPI openMultiFileOStream(
			const char *		pszDirectory,
			const char *		pszBaseName,
			FLMUINT				uiMaxFileSize,
			FLMBOOL				bOkToOverwrite,
			IF_OStream **		ppStream) = 0;
			
		/**
		 * @brief Remove a multi-file stream
		 */
		virtual RCODE XFLAPI removeMultiFileStream(
			const char *		pszDirectory,
			const char *		pszBaseName) = 0;
			
		/**
		 * @brief Convert an output stream (pOStream) into a buffered output stream (*ppOStream)
		 *
		 * As data is written to *ppOStream, it is buffered before ultimately being
		 * written to pOStream.
		 */
		virtual RCODE XFLAPI openBufferedOStream(
			IF_OStream *		pOStream,
			FLMUINT				uiBufferSize,
			IF_OStream **		ppOStream) = 0;
			
		/**
		 * @brief Convert an output stream (pOStream) into a compressed output stream (*ppOStream)
		 *
		 * As data is written to *ppOStream, it is compressed before ultimately being
		 * written to pOStream.
		 */
		virtual RCODE XFLAPI openCompressingOStream(
			IF_OStream *		pOStream,
			IF_OStream **		ppOStream) = 0;
			
		/**
		 * @brief All data is read from the input stream (pIStream) and written
		 * to the output stream (pOStream).  This goes until pIStream returns EOF.
		 */
		virtual RCODE XFLAPI writeToOStream(
			IF_IStream *		pIStream,
			IF_OStream *		pOStream) = 0;
			
		/**
		 * @brief Opens a base64 encoder stream
		 *
		 * This method opens a stream for encoding a user-supplied input stream to
		 * base64 (ASCII).
		 *
		 * @param pInputStream The stream to be encoded
		 * @param bInsertLineBreaks A line break will be inserted every 72
		 * characters
		 * @param ppEncodedStream The stream object used to read the encoded data.
		 * @return RCODE
		 */
		virtual RCODE XFLAPI openBase64Encoder(
			IF_IStream *			pInputStream,
			FLMBOOL					bInsertLineBreaks,
			IF_IStream **			ppEncodedStream) = 0;

		/**
		 * @brief Opens a base64 decoder stream
		 *
		 * This method opens a stream for decoding a user-supplied input stream from
		 * base64 (ASCII) to binary.
		 *
		 * @param pInputStream The stream to be decoded
		 * @param ppDecodedStream The stream object used to read the decoded data.
		 * @return RCODE
		 */
		virtual RCODE XFLAPI openBase64Decoder(
			IF_IStream *			pInputStream,
			IF_IStream **			ppDecodedStream) = 0;

		/**
		 * @brief Creates an IF_DataVector interface object.
		 *
		 * This method creates an IF_DataVector interface object.  The  IF_DataVector
		 * is used in index key searches.
		 *
		 * @param ifppDV The IF_DataVector object.
		 * @return RCODE
		 */
		virtual RCODE XFLAPI createIFDataVector(
			IF_DataVector **		ifppDV) = 0;

		/**
		 * @brief Creates an IF_ResultSet interface object.
		 *
		 * This method creates an  IF_ResultSet  interface object.
		 *
		 * @param ifppResultSet The IF_ResultSet object.
		 * @return RCODE
		 */
		virtual RCODE XFLAPI createIFResultSet(
			IF_ResultSet **		ifppResultSet) = 0;

		/**
		 * @brief Creates an IF_Query  interface object.
		 *
		 * This method creates an  IF_Query  interface object.
		 *
		 * @param ifppQuery The IF_Query object.
		 * @return RCODE
		 */
		virtual RCODE XFLAPI createIFQuery(
			IF_Query **				ifppQuery) = 0;

		/**
		 * @brief Frees memory for allocations that are returned from various XFlaim methods.
		 *
		 * This method frees memory that has been allocated by various methods in XFlaim.  If 
		 * a method allocates memory that needs to be freed by this method, it will be documented
		 * in that method.
		 *
		 * @param ppMem Pointer to the pointer of the memory to be freed.  When the memory is
		 * successfully freed, the pointer will be set to NULL
		 */
		virtual void XFLAPI freeMem(
			void **					ppMem) = 0;

		// Various configuration routines

		/**
		 * @brief Set a dynamic memory limit in the XFlaim database system.
		 *
		 * This method sets the dynamic memory parameters in the XFlaim database system.  When this
		 * method is called, XFlaim is put into a mode where it periodically (every 15 seconds unless
		 * otherwise specified - see setCacheAdjustInterval) adjusts its cache limit.  The parameters
		 * passed into this method are used to calculate the new limit.  The new limit remains in
		 * effect until the next adjustment is made.
		 *
		 * @param uiCacheAdjustPercent Percentage of available physical memory to set or adjust to.
		 * @param uiCacheAdjustMin Minimum bytes to adjust down to.
		 * @param uiCacheAdjustMax Maximum bytes to adjust up to.  NOTE: If this parameter is non-zero,
		 * the uiCacheAdjustMinToLeave parameter is ignored.
		 * @param uiCacheAdjustMinToLeave Minimum bytes to leave available after making adjustment.
		 * This is an alternative way to specify a maximum to adjust to.  Using this value, XFlaim will
		 * calculate the maximum by subtracting this number from the total bytes it thinks is available.
		 * That calculated number becomes the effective maximum to adjust to.
		 * @return RCODE
		 */
		virtual RCODE XFLAPI setDynamicMemoryLimit(
			FLMUINT					uiCacheAdjustPercent,
			FLMUINT					uiCacheAdjustMin,
			FLMUINT					uiCacheAdjustMax,
			FLMUINT					uiCacheAdjustMinToLeave) = 0;

		/**
		 * @brief Set a hard limit on the amount of memory that the database system can access.
		 *
		 * This method sets a hard limit on the amount of memory that the database system can access.
		 * The important difference between this routine and the setDynamicMemoryLimit routine is that
		 * the limit remains in force until a subsequent call to setHardMemoryLimit is made, or until
		 * a call to setDynamicMemoryLimit is made.  The setDynamicMemoryLimit routine, on the other
		 * hand, puts XFlaim into a mode where a new limit is automatically calculated on a preset
		 * interval (see setCacheAdjustInterval).  Thus, the purpose of the setHardMemoryLimit routine
		 * is to let the application control the limit instead of having it automatically adjusted periodically.
		 *
		 * @param uiPercent If this parameter is zero, the uiMax parameter determines the hard cache limit.
		 * Otherwise, this parameter (which must contain a number between 1 and 100) is used to determine
		 * a hard cache limit.  The hard limit will be calculated as a percentage of available physical
		 * memory on the system.
		 * @param bPercentOfAvail A flag to indicate that the percentage (uiPercent) is to be interpreted
		 * as a percentage of available memory as opposed to a percentage of all of physical memory.
		 * This parameter is only used if uiPercent is non-zero.
		 * @param uiMin Minimum bytes to set the hard cache limit to.  Note that this parameter is only used
		 * if uiPercent is non-zero and we are calculating a hard limit to set.
		 * @param uiMax Maximum bytes to set the hard limit to.  Note that if uiPercent is zero,
		 * this number contains the hard limit.
		 * @param uiMinToLeave This parameter is only used if uiPercent is non-zero and we are calculating
		 * a hard limit to set.  Instead of uiMax determining the maximum cache limit we could set, this
		 * value will determine the maximum.  This number will be subtracted from the total memory on the
		 * system or the total memory currently available (if bPercentOfAvail is TRUE) to establish a maximum.
		 * @param bPreallocate Boolean flag.  Used to indicate that the cache should be preallocated when XFlaim
		 * starts up, rather than allow it to grow as needed.  The default value to FALSE.
		 * @return RCODE
		 */
		virtual RCODE XFLAPI setHardMemoryLimit(
			FLMUINT					uiPercent,
			FLMBOOL					bPercentOfAvail,
			FLMUINT					uiMin,
			FLMUINT					uiMax,
			FLMUINT					uiMinToLeave,
			FLMBOOL					bPreallocate = FALSE) = 0;

		/**
		 * @brief Get a flag which indicates whether or not dynamic cache adjusting
		 * is allowed.
		 *
		 * This method returns a boolean TRUE or FALSE.
		 *
		 * @return FLMBOOL TRUE=supported or FALSE=not supported.
		 */
		virtual FLMBOOL XFLAPI getDynamicCacheSupported( void) = 0;
		
		/**
		 * @brief Query the database system for information regarding the current cache usage.
		 *
		 * This method is used to query the database system for information regarding the current cache usage.
		 *
		 * @param pCacheInfo The cache info structure.
		 */
		virtual void XFLAPI getCacheInfo(
			XFLM_CACHE_INFO *		pCacheInfo) = 0;

		/**
		 * @brief A method to either enable or disable cache debug mode.
		 *
		 * This is a method to either enable or disable cache debug mode.  If bDebug is TRUE (1),
		 * then cache debug will be enabled.  If bDebug is FALSE (0), cache debug will be disabled.
		 *
		 * @param bDebug A boolean to indicate whether to enable or disable cache debug mode.
		 */
		virtual void XFLAPI enableCacheDebug(
			FLMBOOL					bDebug) = 0;

		/**
		 * @brief A method to find out if cache debug mode is either enabled or disabled.
		 *
		 * This is a method to find out if cache debug mode is either enabled or disabled.
		 *
		 * @return FLMBOOL True or False
		 */
		virtual FLMBOOL XFLAPI cacheDebugEnabled( void) = 0;

		/**
		 * @brief Close all file handles (descriptors) that have not been used for a specified
		 * amount of time.
		 *
		 * This is a method to close all file handles (descriptors) that have not been used or
		 * accessed for a specified number of seconds.
		 *
		 * @param uiSeconds The number of seconds.  File handles (descriptors) that have been
		 * unused for a period of time greater than or equal to this number of seconds will be
		 * closed.  A value of zero will have the effect of closing all unused file handles
		 * (descriptors), regardless of how long they have been unused.
		 * @return RCODE
		 */
		virtual RCODE XFLAPI closeUnusedFiles(
			FLMUINT					uiSeconds) = 0;

		/**
		 * @brief Start the collection of statistics on the database system.
		 *
		 * This method starts the collection of statistics on the database system.
		 */
		virtual void XFLAPI startStats( void) = 0;

		/**
		 * @brief Stop the collection of statistics on the database system.
		 *
		 * This method stops the collection of statistics on the database system.
		 */
		virtual void XFLAPI stopStats( void) = 0;

		/**
		 * @brief Reset the statistics counters on the database system.
		 *
		 * This method resets the statistics counters on the database system.
		 */
		virtual void XFLAPI resetStats( void) = 0;

		/**
		 * @brief Retrieve the statistics from the database system.
		 *
		 * This method returns the current set of statistics from the database system.
		 *
		 * @param pFlmStats The structure where statistics are returned.
		 * @return RCODE
		 */
		virtual RCODE XFLAPI getStats(
			XFLM_STATS *			pFlmStats) = 0;

		/**
		 * @brief Free the statistic object in the database system.
		 *
		 * This method frees any memory allocations that are associated with the FLM_STATS
		 * structure.  The FLM_STATS structure will have been populated by a call to the
		 * getStats method.
		 *
		 * @param pFlmStats The statistics structure whose memory allocations are to be freed.
		 */
		virtual void XFLAPI freeStats(
			XFLM_STATS *			pFlmStats) = 0;

		/**
		 * @brief Set the directory where temporary files are created.
		 *
		 * This method sets the directory where temporary files are to be created.
		 *
		 * @param pszPath The temporary directory path.
		 * @return RCODE
		 */
		virtual RCODE XFLAPI setTempDir(
			const char *			pszPath) = 0;

		/**
		 * @brief Get the directory where temporary files are created.
		 *
		 * This method returns the directory name where temporary files are created.
		 * If no temporary directory is set, this function returns NE_FLM_IO_PATH_NOT_FOUND.
		 *
		 * @param pszPath The temporary directory path is returned here.
		 * @return RCODE
		 */
		virtual RCODE XFLAPI getTempDir(
			char *					pszPath) = 0;

		/**
		 * @brief Set the time between  checkpoints in the database.
		 *
		 * 
    	 * This method sets the maximum time between completed checkpoints in the database.
		 * This is NOT the same thing as how often a checkpoint will be performed.  Checkpoints
		 * are actually performed and completed much more frequently, depending on how much
		 * update transaction activity there is.  A background thread (often referred to as the
		 * checkpoint thread) is responsible for performing database checkpoints.  The checkpoint
		 * thread wakes up about once a second to see if there are dirty blocks in the cache
		 * that need to be written to disk.  If there are, and there is no currently active
		 * update transaction, it will begin performing a checkpoint.  If no update
		 * transactions occur while it is performing the checkpoint, it will be able to complete
		 * the checkpoint.  If the application attempts to start an update transaction while
		 * the checkpoint is being done, the checkpoint thread has the option of yielding
		 * to the update transaction and not completing the checkpoint.  The checkpoint
		 * thread may also choose to not yield and complete the checkpoint.  One of the
		 * conditions under which it will not yield is if the elapsed time since the last
		 * completed checkpoint is greater than the time specified by this method.  In that
		 * case, the update transaction is held off until the checkpoint completes.
    	 * Note that it is possible for the time between completed checkpoints to be longer than
		 * the interval specified in this routine.  The checkpoint thread cannot interrupt an
		 * active transaction when it wakes up.  If it wakes up and finds that a transaction
		 * is active, it must wait for the transaction to complete before it can start a checkpoint.
		 * Thus, if that transaction runs a long time, the time between completed checkpoints
		 * could exceed the time specified in this method.
		 *
		 * @param uiSeconds The maximum number of seconds allowed to elapse between completed
		 * checkpoints.  Default is 180 seconds.  NOTE: It is possible that the time
		 * between completed checkpoints can be greater than this value.  This is because the
		 * checkpoint thread cannot run if there is an update transaction that is currently
		 * active. If an update transaction is active and runs for a long time, the time
		 * between completed checkpoints could exceed the time specified in this method.
		 */
		virtual void XFLAPI setCheckpointInterval(
			FLMUINT					uiSeconds) = 0;

		/**
		 * @brief Get the current  checkpoint interval.
		 *
		 * This method returns the current  checkpoint interval.
		 * @return FLMUINT The current checkpoint interval (seconds).
		 */
		virtual FLMUINT XFLAPI getCheckpointInterval( void) = 0;

		/**
		 * @brief Set the time interval for dynamically adjusting the cache limit.
		 *
		 * This method sets the time interval for dynamically adjusting the cache limit.
		 *
		 * @param uiSeconds The time interval for dynamically adjusting the cache limit.
		 */
		virtual void XFLAPI setCacheAdjustInterval(
			FLMUINT					uiSeconds) = 0;

		/**
		 * @brief Get the time interval for dynamically adjusting the cache limit.
		 *
		 * This method returns the time interval (in seconds) for dynamically adjusting the cache limit.
		 *
		 * @return FLMUINT The curernt cache adjust interval (seconds).
		 */
		virtual FLMUINT XFLAPI getCacheAdjustInterval( void) = 0;

		/**
		 * @brief Set the time interval for dynamically cleaning out old cache blocks from block cache.
		 *
		 * This method sets the time interval for cleaning out old cache blocks from block cache.
		 *
		 * @param uiSeconds The time interval for dynamically cleaning out old cache blocks.
		 */
		virtual void XFLAPI setCacheCleanupInterval(
			FLMUINT					uiSeconds) = 0;

		/**
		 * @brief Get the time interval for dynamically cleaning out old cache blocks from block cache.
		 *
		 * This method returns the time interval (in seconds) for cleaning out old cache blocks from block cache.
		 *
		 * @return FLMUINT The current cache cleanup inerval (seconds).
		 */
		virtual FLMUINT XFLAPI getCacheCleanupInterval( void) = 0;

		/**
		 * @brief Set time interval for cleaning up unused resources (such as file handles).
		 *
		 * This method sets the time interval for cleaning up unused resources (such as file handles).
		 *
		 * @param uiSeconds  	The time interval for cleaning up unused resources (such as file handles).
		 */
		virtual void XFLAPI setUnusedCleanupInterval(
			FLMUINT					uiSeconds) = 0;

		/**
		 * @brief Get time interval for cleaning up unused resources (such as file handles).
		 *
		 * This method returns the time interval (in seconds) for cleaning up unused resources (such as file handles).
		 *
		 * @return FLMUINT The current unused cleanup interval (seconds).
		 */
		virtual FLMUINT XFLAPI getUnusedCleanupInterval( void) = 0;

		/**
		 * @brief Set maximum time for a resource (such as a file handle) to be unused before it is cleaned up.
		 *
		 * This method sets the maximum time for a resource (such as a file handle) to be unused before it
		 * is cleaned up.
		 *
		 * @param uiSeconds The maximum time for a resource (such as a file handle) to be unused before it is
		 * cleaned up.
		 */
		virtual void XFLAPI setMaxUnusedTime(
			FLMUINT					uiSeconds) = 0;

		/**
		 * @brief Get maximum time for a resource (such as a file handle) to be unused before it is cleaned up.
		 *
		 * This method returns the maximum time for a resource (such as a file handle) to be unused
		 * before it is cleaned up.
		 *
		 * @return FLMUINT The current maximum unused time (seconds).
		 */
		virtual FLMUINT XFLAPI getMaxUnusedTime( void) = 0;

		/**
		 * @brief Set the logger client.
		 *
		 * @param pLogger Pointer to the logger client object.
		 */
		virtual void XFLAPI setLogger(
			IF_LoggerClient *		pLogger) = 0;

		/**
		 * @brief Deactivate open database objects, forcing the database(s) to eventually be closed.
		 *
		 * This method deactivates all open database objects (IF_Db objects) for a particular
		 * database, forcing the database to eventually be closed.  Passing NULL in the pszDbFileName
		 * parameter will deactivate all active database objects for all open databases.
		 *
		 * @param pszDatabasePath Name of the control file ( including path ) for the database that
		 * is to be deactivated.  See the XFlaim Concepts/Database Files for a discussion on the
		 * different database files, including the control file.  NOTE: Passing a NULL in this
		 * parameter will cause all open databases to be deactivated.
		 * @param pszDataFilePath The directory where the data files are located.  If a NULL is passed
		 * in, it is assumed that the data files are located in the same directory as the control file
		 * (as specified by the pszDbFileName parameter). See the XFlaim Concepts/Database Files for
		 * a discussion on the different database files.
		 */
		virtual void XFLAPI deactivateOpenDb(
			const char *			pszDatabasePath,
			const char *			pszDataFilePath) = 0;

		/**
		 * @brief Set the maximum number of queries to save when statistics gathering is enabled.
		 *
		 * This method sets the maximum number of queries to save when statistics gathering is enabled.
		 *
		 * @param uiMaxToSave The maximum number of queries to save.
		 */
		virtual void XFLAPI setQuerySaveMax(
			FLMUINT					uiMaxToSave) = 0;

		/**
		 * @brief Get the maximum number of queries to save when statistics gathering is enabled.
		 *
		 * This method returns the maximum number of queries to save when statistics gathering is enabled.
		 *
		 * @return FLMUINT The maximum number of queries to save.
		 */
		virtual FLMUINT XFLAPI getQuerySaveMax( void) = 0;

		/**
		 * @brief Set the minimum and maximum dirty cache limits.
		 *
    		 * This method sets dirty cache limits - a "maximum" and a "low."  The maximum specifies the maximum
		 * dirty cache to be allowed.  When a database exceeds this amount, the checkpoint thread will kick
		 * in and write out dirty blocks until the dirty cache comes back down below the amount specified by
		 * the "low" value.  In this way, the application can control how much dirty cache builds up between
		 * forced checkpoints (see setCheckpointInterval).  The more dirty cache there is when the checkpoint
		 * forces a checkpoint, the longer it will take to complete the checkpoint.  It should be noted that
		 * the overall time that will be needed to write out dirty blocks is still the same.  It's just that
		 * the writing gets spread out more over time.  Instead of doing it all in one big chunk, it gets done
		 * in lots of little chunks.  This has both pros and cons, so this method should be used with extreme caution!
       * In a bulk load situation, where there is one thread doing the bulk loading, lowering the maximum
		 * dirty cache could actually lengthen out the overall time it takes to complete the bulk load.  This
		 * is because the bulk load will be interrupted more often by the checkpoint thread to do smaller units
		 * of writing than it otherwise would.  There will also be less piggy-backing of writes.  Piggy-backing
		 * occurs when multiple transactions write to the same data block before the block is flushed out of
		 * cache.  If the database system is flushing dirty blocks out of cache more often, it could end up
		 * writing the same block multiple times, whereas it might not not have had to if it had waited longer between flushes.
		 * The advantage to spreading out the flushing of dirty cache blocks occurs when there are multiple threads
		 * trying to do update transactions.  When the checkpoint thread forces a checkpoint, it holds back all
		 * pending update transactions.  If it has to do more writing when it forces a checkpoint, it may cause
		 * many threads to wait for a longer period of time than it would if there were less writing to do.
		 * This is less efficient to the overall throughput of the system, because it is likely that each of the
		 * threads could be doing other useful work instead of waiting.
		 *
		 * @param uiMaxDirty The maximum amount (in bytes) of dirty cache allowed.
		 * @param uiLowDirty The low threshold (in bytes) for dirty cache.
		 */
		virtual void XFLAPI setDirtyCacheLimits(
			FLMUINT					uiMaxDirty,
			FLMUINT					uiLowDirty) = 0;

		/**
		 * @brief Get the minimum and maximum dirty cache limits.
		 *
		 * This method returns the minimum and maximum dirty cache limits.
		 *
		 * @param puiMaxDirty The maximum number of dirty blocks allowed in the cache.
		 * @param puiLowDirty The low threshold for the number of dirty blocks in cache.
		 */
		virtual void XFLAPI getDirtyCacheLimits(
			FLMUINT *				puiMaxDirty,
			FLMUINT *				puiLowDirty) = 0;

		/**
		 * @brief Get an information object that can be used to get information about threads owned by the database system.
		 *
		 * This method returns a thread information object which has methods for retrieving various pieces of information
		 * about threads owned by the database system.  NOTE: When the application is done using the returned IF_ThreadInfo
		 * object, it should call its Release method.
		 *
		 * @param ifppThreadInfo A pointer to the allocated thread info object is returned here.
		 *
		 * @return RCODE
		 */
		virtual RCODE XFLAPI getThreadInfo(
			IF_ThreadInfo **		ifppThreadInfo) = 0;

		/**
		 * @brief Register a catcher object to catch database events in a particular category.
		 *
		 * This method registers an object to catch events in a particular category of database events.
		 *
		 * @param eCategory The category of events the application is registering for.
		 * @param ifpEventClient The client object whose methods are to be invoked when the event occurs.
		 *
		 * @return RCODE
		 */
		virtual RCODE XFLAPI registerForEvent(
			eEventCategory			eCategory,
			IF_EventClient *		ifpEventClient) = 0;

		/**
		 * @brief Deregister a catcher object from catching database events in a particular category.
		 *
		 * This method deregisters an object from catching events in a particular 
		 * category of database events.
		 *
		 * @param eCategory The type of event.
		 * @param ifpEventClient The client object that was passed into the registerForEvent 
		 * method.  This is necessary so that if there are multiple objects that have 
		 * registered for an event, XFlaim can know exactly which object to
		 * deregister.
		 */
		virtual void XFLAPI deregisterForEvent(
			eEventCategory			eCategory,
			IF_EventClient *		ifpEventClient) = 0;

		/**
		 * @brief Returns the metaphone codes for the next word in an input stream.
		 *
		 * This parses the next word from the input stream (ifpIStream) and returns the metaphone codes for it.  This
		 * method returns NE_XFLM_EOF_HIT when it hits the end of the input stream.  Visit: This should probably be a
		 * method on the IF_IStream interface, not the IF_DbSystem interface.
		 *
		 * @param ifpIStream Input stream object.
		 * @param puiMetaphone Primary metaphone returned for the next word in the input stream.
		 * @param puiAltMetaphone Alternate metaphone returned for the next word in the input stream.
		 *
		 * @return RCODE
		 */
		virtual RCODE XFLAPI getNextMetaphone(
			IF_IStream *			ifpIStream,
			FLMUINT *				puiMetaphone,
			FLMUINT *				puiAltMetaphone = NULL) = 0;
			
		/**
		 * @brief Compares two UTF-8 strings
		 */
		virtual RCODE XFLAPI compareUTF8Strings(
			const FLMBYTE *	pucLString,
			FLMUINT				uiLStrBytes,
			FLMBOOL				bLeftWild,
			const FLMBYTE *	pucRString,
			FLMUINT				uiRStrBytes,
			FLMBOOL				bRightWild,
			FLMUINT				uiCompareRules,
			FLMUINT				uiLanguage,
			FLMINT *				piResult) = 0;
			
		/**
		 * @brief Compares two Unicode strings
		 */
		virtual RCODE XFLAPI compareUnicodeStrings(
			const FLMUNICODE *	puzLString,
			FLMUINT					uiLStrBytes,
			FLMBOOL					bLeftWild,
			const FLMUNICODE *	puzRString,
			FLMUINT					uiRStrBytes,
			FLMBOOL					bRightWild,
			FLMUINT					uiCompareRules,
			FLMUINT					uiLanguage,
			FLMINT *					piResult) = 0;

		virtual RCODE XFLAPI utf8IsSubStr(
			const FLMBYTE *	pszString,
			const FLMBYTE *	pszSubString,
			FLMUINT				uiCompareRules,
			FLMUINT				uiLanguage,
			FLMBOOL *			pbExists) = 0;
		
		virtual FLMBOOL XFLAPI uniIsUpper(
			FLMUNICODE			uzChar) = 0;

		virtual FLMBOOL XFLAPI uniIsLower(
			FLMUNICODE			uzChar) = 0;

		virtual FLMBOOL XFLAPI uniIsAlpha(
			FLMUNICODE			uzChar) = 0;

		virtual FLMBOOL XFLAPI uniIsDecimalDigit(
			FLMUNICODE			uzChar) = 0;

		virtual FLMUNICODE XFLAPI uniToLower(
			FLMUNICODE			uzChar) = 0;
			
		// When the nextUCS2Char method is called, the UCS-2 version of the character 
		// pointed to by *ppszUTF8 is stored in *puzChar and *ppszUTF8 is updated to
		// point to the next the next character.  If *ppszUTF8 >= pszEndOfUTF8String
		// or if **ppszUTF8 == 0, *puzChar returns 0, but no error is returned.
		// Note: Remember to keep a copy of the pointer to the start of the
		// string, because whatever is passed in as ppszUTF8 will be modified.
		
		virtual RCODE XFLAPI nextUCS2Char(
			const FLMBYTE **	ppszUTF8,
			const FLMBYTE *	pszEndOfUTF8String,
			FLMUNICODE *		puzChar) = 0;
			
		virtual RCODE XFLAPI numUCS2Chars(
			const FLMBYTE *	pszUTF8,
			FLMUINT *			puiNumChars) = 0;
			
		/**
		 * @brief Waits for a specific database to close
		 *
		 * @param pszDbName Name of the control file for the database that we
		 * want to have closed.
		 *
		 * @return RCODE
		 */
		virtual RCODE XFLAPI waitToClose(
			const char *	pszDbFileName) = 0;

		/**
		 * @brief Creates an IF_NodeInfo  interface object.
		 *
		 * This method creates an  IF_NodeInfo  interface object.
		 *
		 * @param ifppNodeInfo The IF_NodeInfo object.
		 * @return RCODE
		 */
		virtual RCODE XFLAPI createIFNodeInfo(
			IF_NodeInfo **				ifppNodeInfo) = 0;
			
		/**
		 * @brief Creates an IF_BTreeInfo  interface object.
		 *
		 * This method creates an  IF_BTreeInfo  interface object.
		 *
		 * @param ifppBTreeInfo The IF_BTreeInfo object.
		 * @return RCODE
		 */
		virtual RCODE XFLAPI createIFBTreeInfo(
			IF_BTreeInfo **			ifppBTreeInfo) = 0;

		/**
		 * @brief A method to attempt to remove everything from cache
		 *
		 * This method will attempt to remove all blocks and nodes from
		 * cache.  The pDb parameter is optional.  If provided (and an update
		 * transaction is active), any dirty cache items associated with the
		 * database will be flushed.
		 *
		 * @return RCODE
		 */
		virtual RCODE XFLAPI clearCache(
			IF_Db *					pDb) = 0;
	};

	// IMPORTANT NOTE: Changes to these enums need to be synced with the
	// corresponding definitions in java and C# code.
	typedef enum
	{
		XFLM_EXPORT_NO_FORMAT =			0x00, 	// No Formatting
		XFLM_EXPORT_NEW_LINE =			0x01, 	// New Line For Each Element
		XFLM_EXPORT_INDENT =				0x02,		// Indent Elements
		XFLM_EXPORT_INDENT_DATA =		0x03		// Indent Data - this changes the data
	} eExportFormatType;

	XFLXPC RCODE XFLAPI FlmAllocDbSystem(
		IF_DbSystem **				ppDbSystem);

	/****************************************************************************
	Desc:
	****************************************************************************/
	flminterface XFLEXP IF_Db : public F_Object
	{
		virtual RCODE XFLAPI transBegin(
			eDbTransType			eTransType,
			FLMUINT					uiMaxLockWait = FLM_NO_TIMEOUT,
			FLMUINT					uiFlags = 0,
			XFLM_DB_HDR *			pDbHeader = NULL) = 0;

		virtual RCODE XFLAPI transBegin(
			IF_Db *					pDb) = 0;

		virtual RCODE XFLAPI transCommit(
			FLMBOOL *				pbEmpty = NULL) = 0;

		virtual RCODE XFLAPI transAbort( void) = 0;

		virtual eDbTransType XFLAPI getTransType( void) = 0;

		virtual RCODE XFLAPI doCheckpoint(
			FLMUINT					uiTimeout) = 0;

		virtual RCODE XFLAPI dbLock(
			eLockType				lockType,
			FLMINT					iPriority,
			FLMUINT					uiTimeout) = 0;

		virtual RCODE XFLAPI dbUnlock( void) = 0;

		virtual RCODE XFLAPI getLockType(
			eLockType *				pLockType,
			FLMBOOL *				pbImplicit) = 0;

		virtual RCODE XFLAPI getLockInfo(
			FLMINT					iPriority,
			eLockType *				pCurrLockType,
			FLMUINT *				puiThreadId,
			FLMUINT *				puiNumExclQueued,
			FLMUINT *				puiNumSharedQueued,
			FLMUINT *				puiPriorityCount) = 0;

		virtual RCODE XFLAPI indexStatus(
			FLMUINT					uiIndexNum,
			XFLM_INDEX_STATUS *	pIndexStatus) = 0;

		virtual RCODE XFLAPI indexGetNext(
			FLMUINT *				puiIndexNum) = 0;

		virtual RCODE XFLAPI indexSuspend(
			FLMUINT					uiIndexNum) = 0;

		virtual RCODE XFLAPI indexResume(
			FLMUINT					uiIndexNum) = 0;

		virtual RCODE XFLAPI keyRetrieve(
			FLMUINT					uiIndex,
			IF_DataVector *		pSearchKey,
			FLMUINT					uiFlags,
			IF_DataVector *		pFoundKey) = 0;

		virtual RCODE XFLAPI enableEncryption( void) = 0;

		virtual RCODE XFLAPI wrapKey(
			const char *	pszPassword = NULL) = 0;
		
		virtual RCODE XFLAPI rollOverDbKey( void) = 0;
			
		virtual RCODE XFLAPI changeItemState(
			FLMUINT					uiDictType,
			FLMUINT					uiDictNum,
			const char *			pszState) = 0;

		virtual RCODE XFLAPI reduceSize(
			FLMUINT					uiCount,
			FLMUINT *				puiCount) = 0;

		virtual RCODE XFLAPI upgrade(
			IF_UpgradeClient *	pUpgradeClient) = 0;

		virtual RCODE XFLAPI createRootElement(
			FLMUINT					uiCollection,
			FLMUINT					uiNameId,
			IF_DOMNode **			ppElementNode,
			FLMUINT64 *				pui64NodeId = NULL) = 0;

		virtual RCODE XFLAPI createDocument(
			FLMUINT					uiCollection,
			IF_DOMNode **			ppDocumentNode,
			FLMUINT64 *				pui64NodeId = NULL) = 0;

		virtual RCODE XFLAPI getFirstDocument(
			FLMUINT					uiCollection,
			IF_DOMNode **			ppDocumentNode) = 0;

		virtual RCODE XFLAPI getLastDocument(
			FLMUINT					uiCollection,
			IF_DOMNode **			ppDocumentNode) = 0;

		virtual RCODE XFLAPI getDocument(
			FLMUINT					uiCollection,
			FLMUINT					uiFlags,
			FLMUINT64				ui64DocumentId,
			IF_DOMNode **			ppDocumentNode) = 0;

		virtual RCODE XFLAPI documentDone(
			FLMUINT					uiCollection,
			FLMUINT64				ui64RootId) = 0;

		virtual RCODE XFLAPI documentDone(
			IF_DOMNode *			pDocNode) = 0;

		virtual RCODE XFLAPI createElementDef(
			const char *			pszNamespaceURI,
			const char *			pszElementName,
			FLMUINT					uiDataType,
			FLMUINT * 				puiElementNameId = NULL,
			IF_DOMNode **			ppDocumentNode = NULL) = 0;

		virtual RCODE XFLAPI createElementDef(
			const FLMUNICODE *	puzNamespaceURI,
			const FLMUNICODE *	puzElementName,
			FLMUINT					uiDataType,
			FLMUINT * 				puiElementNameId = NULL,
			IF_DOMNode **			ppDocumentNode = NULL) = 0;

		virtual RCODE XFLAPI createUniqueElmDef(
			const char *			pszNamespaceURI,
			const char *			pszElementName,
			FLMUINT * 				puiElementNameId = NULL,
			IF_DOMNode **			ppDocumentNode = NULL) = 0;

		virtual RCODE XFLAPI createUniqueElmDef(
			const FLMUNICODE *	puzNamespaceURI,
			const FLMUNICODE *	puzElementName,
			FLMUINT * 				puiElementNameId = NULL,
			IF_DOMNode **			ppDocumentNode = NULL) = 0;

		virtual RCODE XFLAPI getElementNameId(
			const char *			pszNamespaceURI,
			const char *			pszElementName,
			FLMUINT *				puiElementNameId) = 0;

		virtual RCODE XFLAPI getElementNameId(
			const FLMUNICODE *	puzNamespaceURI,
			const FLMUNICODE *	puzElementName,
			FLMUINT *				puiElementNameId) = 0;

		virtual RCODE XFLAPI createAttributeDef(
			const char *			pszNamespaceURI,
			const char *			pszAttributeName,
			FLMUINT					uiDataType,
			FLMUINT * 				puiAttributeNameId,
			IF_DOMNode **			ppDocumentNode = NULL) = 0;

		virtual RCODE XFLAPI createAttributeDef(
			const FLMUNICODE *	puzNamespaceURI,
			const FLMUNICODE *	puzAttributeName,
			FLMUINT					uiDataType,
			FLMUINT * 				puiAttributeNameId,
			IF_DOMNode **			ppDocumentNode = NULL) = 0;

		virtual RCODE XFLAPI getAttributeNameId(
			const char *			pszNamespaceURI,
			const char *			pszAttributeName,
			FLMUINT *				puiAttributeNameId) = 0;

		virtual RCODE XFLAPI getAttributeNameId(
			const FLMUNICODE *	puzNamespaceURI,
			const FLMUNICODE *	puzAttributeName,
			FLMUINT *				puiAttributeNameId) = 0;

		virtual RCODE XFLAPI createPrefixDef(
			const char *			pszPrefixName,
			FLMUINT * 				puiPrefixNumber) = 0;

		virtual RCODE XFLAPI createPrefixDef(
			const FLMUNICODE *	puzPrefixName,
			FLMUINT * 				puiPrefixNumber) = 0;

		virtual RCODE XFLAPI getPrefixId(
			const char *			pszPrefixName,
			FLMUINT *				puiPrefixNumber) = 0;

		virtual RCODE XFLAPI getPrefixId(
			const FLMUNICODE *	puzPrefixName,
			FLMUINT *				puiPrefixNumber) = 0;

		virtual RCODE XFLAPI createEncDef(
			const char *			pszEncType,
			const char *			pszEncName,
			FLMUINT					uiKeySize,
			FLMUINT *				puiEncDefNumber) = 0;

		virtual RCODE XFLAPI createEncDef(
			const FLMUNICODE *	puzEncType,
			const FLMUNICODE *	puzEncName,
			FLMUINT					uiKeySize,
			FLMUINT *				puiEncDefNumber) = 0;

		virtual RCODE XFLAPI getEncDefId(
			const char *			pszEncDefName,
			FLMUINT *				puiPrefixNumber) = 0;

		virtual RCODE XFLAPI getEncDefId(
			const FLMUNICODE *	puzEncDefName,
			FLMUINT *				puiEncDefNumber) = 0;

		virtual RCODE XFLAPI createCollectionDef(
			const char *			pszCollectionName,
			FLMUINT * 				puiCollectionNumber,
			FLMUINT					uiEncNumber = 0) = 0;

		virtual RCODE XFLAPI createCollectionDef(
			const FLMUNICODE *	puzCollectionName,
			FLMUINT * 				puiCollectionNumber,
			FLMUINT					uiEncNumber = 0) = 0;

		virtual RCODE XFLAPI getCollectionNumber(
			const char *			pszCollectionName,
			FLMUINT *				puiCollectionNumber) = 0;

		virtual RCODE XFLAPI getCollectionNumber(
			const FLMUNICODE *	puzCollectionName,
			FLMUINT *				puiCollectionNumber) = 0;

		virtual RCODE XFLAPI getIndexNumber(
			const char *			pszIndexName,
			FLMUINT *				puiIndexNumber) = 0;

		virtual RCODE XFLAPI getIndexNumber(
			const FLMUNICODE *	puzIndexName,
			FLMUINT *				puiIndexNumber) = 0;

		virtual RCODE XFLAPI getDictionaryDef(
			FLMUINT					uiDictType,
			FLMUINT					uiDictNumber,
			IF_DOMNode **			ppDocumentNode) = 0;

		virtual RCODE XFLAPI getDictionaryName(
			FLMUINT					uiDictType,
			FLMUINT					uiDictNumber,
			char *					pszName,
			FLMUINT *				puiNameBufSize,
			char *					pszNamespace = NULL,
			FLMUINT *				puiNamespaceBufSize = NULL) = 0;

		virtual RCODE XFLAPI getDictionaryName(
			FLMUINT					uiDictType,
			FLMUINT					uiDictNumber,
			FLMUNICODE *			puzName,
			FLMUINT *				puiNameBufSize,
			FLMUNICODE *			puzNamespace = NULL,
			FLMUINT *				puiNamespaceBufSize = NULL) = 0;

		virtual RCODE XFLAPI getNode(
			FLMUINT					uiCollection,
			FLMUINT64				ui64NodeId,
			IF_DOMNode **			ppNode) = 0;

		virtual RCODE XFLAPI getAttribute(
			FLMUINT					uiCollection,
			FLMUINT64				ui64ElementNodeId,
			FLMUINT					uiAttrNameId,
			IF_DOMNode **			ppNode) = 0;

		virtual RCODE XFLAPI getDataType(
			FLMUINT					uiDictType,
			FLMUINT					uiNameId,
			FLMUINT *				puiDataType) = 0;

		virtual RCODE XFLAPI backupBegin(
			eDbBackupType			eBackupType,
			eDbTransType			eTransType,
			FLMUINT					uiMaxLockWait,
			IF_Backup **			ppBackup) = 0;

		virtual void XFLAPI getRflFileName(
			FLMUINT					uiFileNum,
			FLMBOOL					bBaseOnly,
			char *					pszFileName,
			FLMUINT *				puiFileNameBufSize,
			FLMBOOL *				pbNameTruncated = NULL) = 0;

		virtual RCODE XFLAPI import(
			IF_IStream *			pIStream,
			FLMUINT					uiCollection,
			IF_DOMNode *			pNodeToLinkTo = NULL,
			eNodeInsertLoc			eInsertLoc = XFLM_LAST_CHILD,
			XFLM_IMPORT_STATS *	pImportStats = NULL) = 0;

		virtual RCODE XFLAPI importDocument(
			IF_IStream *			ifpStream,
			FLMUINT					uiCollection,
			IF_DOMNode **			ppDocumentNode = NULL,
			XFLM_IMPORT_STATS *	pImportStats = NULL) = 0;

		virtual RCODE XFLAPI exportXML(
			IF_DOMNode *			pStartNode,
			IF_OStream *			pOStream,
			eExportFormatType		eFormat = XFLM_EXPORT_INDENT) = 0;
			
		virtual RCODE XFLAPI setNextNodeId(
			FLMUINT					uiCollection,
			FLMUINT64				ui64NextNodeId) = 0;

		virtual RCODE XFLAPI setNextDictNum(
			FLMUINT					uiDictType,
			FLMUINT					uiDictNumber) = 0;

		// Configuration "set" and "get" methods

		virtual RCODE XFLAPI setRflKeepFilesFlag(
			FLMBOOL					bKeep) = 0;

		virtual RCODE XFLAPI getRflKeepFlag(
			FLMBOOL *				pbKeep) = 0;

		virtual RCODE XFLAPI setRflDir(
			const char *			pszNewRflDir) = 0;

		virtual void XFLAPI getRflDir(
			char *					pszRflDir) = 0;

		virtual RCODE XFLAPI getRflFileNum(
			FLMUINT *				puiRflFileNum) = 0;

		virtual RCODE XFLAPI getHighestNotUsedRflFileNum(
			FLMUINT *				puiHighestNotUsedRflFileNum) = 0;

		virtual RCODE XFLAPI setRflFileSizeLimits(
			FLMUINT					uiMinRflSize,
			FLMUINT					uiMaxRflSize) = 0;

		virtual RCODE XFLAPI getRflFileSizeLimits(
			FLMUINT *				puiRflMinFileSize,
			FLMUINT *				puiRflMaxFileSize) = 0;

		virtual RCODE XFLAPI rflRollToNextFile( void) = 0;

		virtual RCODE XFLAPI setKeepAbortedTransInRflFlag(
			FLMBOOL					bKeep) = 0;

		virtual RCODE XFLAPI getKeepAbortedTransInRflFlag(
			FLMBOOL *				pbKeep) = 0;

		virtual RCODE XFLAPI setAutoTurnOffKeepRflFlag(
			FLMBOOL					bAutoTurnOff) = 0;

		virtual RCODE XFLAPI getAutoTurnOffKeepRflFlag(
			FLMBOOL *				pbAutoTurnOff) = 0;

		virtual void XFLAPI setFileExtendSize(
			FLMUINT					uiFileExtendSize) = 0;

		virtual FLMUINT XFLAPI getFileExtendSize( void) = 0;

		virtual void XFLAPI setAppData(
			void *			pvAppData) = 0;

		virtual void * XFLAPI getAppData( void) = 0;

		virtual void XFLAPI setDeleteStatusObject(
			IF_DeleteStatus *		pDeleteStatus) = 0;

		virtual void XFLAPI setCommitClientObject(
			IF_CommitClient *		pCommitClient) = 0;

		virtual void XFLAPI setIndexingClientObject(
			IF_IxClient *			pIxClient) = 0;

		virtual void XFLAPI setIndexingStatusObject(
			IF_IxStatus *			pIxStatus) = 0;

		// Configuration information getting methods

		virtual FLMUINT XFLAPI getDbVersion( void) = 0;

		virtual FLMUINT XFLAPI getBlockSize( void) = 0;

		virtual FLMUINT XFLAPI getDefaultLanguage( void) = 0;

		virtual FLMUINT64 XFLAPI getTransID( void) = 0;

		virtual void XFLAPI getCheckpointInfo(
			XFLM_CHECKPOINT_INFO *	pCheckpointInfo) = 0;

		virtual RCODE XFLAPI getDbControlFileName(
			char *					pszControlFileName,
			FLMUINT					uiControlFileBufSize) = 0;

		virtual RCODE XFLAPI getLockWaiters(
			IF_LockInfoClient *	pLockInfo) = 0;

		virtual RCODE XFLAPI getLastBackupTransID(
			FLMUINT64 *				pui64LastBackupTransID) = 0;

		virtual RCODE XFLAPI getBlocksChangedSinceBackup(
			FLMUINT *				puiBlocksChangedSinceBackup) = 0;

		virtual RCODE XFLAPI getNextIncBackupSequenceNum(
			FLMUINT *				puiNextIncBackupSequenceNum) = 0;

		virtual void XFLAPI getSerialNumber(
			char *					pucSerialNumber) = 0;

		virtual RCODE XFLAPI getDiskSpaceUsage(
			FLMUINT64 *				pui64DataSize,
			FLMUINT64 *				pui64RollbackSize,
			FLMUINT64 *				pui64RflSize) = 0;

		virtual RCODE XFLAPI getMustCloseRC( void) = 0;

		virtual RCODE XFLAPI getAbortRC( void) = 0;

		virtual void XFLAPI setMustAbortTrans(
			RCODE						rc) = 0;
	};
	
	/****************************************************************************
	Desc:
	****************************************************************************/
	flminterface XFLEXP IF_DOMNode : public F_Object
	{
		virtual RCODE XFLAPI createNode(
			IF_Db *					pDb,
			eDomNodeType			eNodeType,
			FLMUINT					uiNameId,
			eNodeInsertLoc			eLocation,
			IF_DOMNode **			ppNewNode,
			FLMUINT64 *				pui64NodeId = NULL) = 0;

		virtual RCODE XFLAPI createChildElement(
			IF_Db *					pDb,
			FLMUINT					uiChildElementNameId,
			eNodeInsertLoc			eLocation,
			IF_DOMNode **			ppNewChildElementNode,
			FLMUINT64 *				pui64NodeId = NULL) = 0;
			
		virtual RCODE XFLAPI deleteNode(
			IF_Db *					pDb) = 0;

		virtual RCODE XFLAPI deleteChildren(
			IF_Db *					pDb,
			FLMUINT					uiNameId = 0) = 0;
			
		virtual RCODE XFLAPI createAttribute(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			IF_DOMNode **			ppAttrNode) = 0;

		virtual RCODE XFLAPI getFirstAttribute(
			IF_Db *					pDb,
			IF_DOMNode **			ppAttrNode) = 0;

		virtual RCODE XFLAPI getLastAttribute(
			IF_Db *					pDb,
			IF_DOMNode **			ppAttrNode) = 0;

		virtual RCODE XFLAPI getAttribute(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			IF_DOMNode **			ppAttrNode) = 0;

		virtual RCODE XFLAPI deleteAttribute(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId) = 0;

		virtual RCODE XFLAPI hasAttribute(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			IF_DOMNode **			ppAttrNode = NULL) = 0;

		virtual RCODE XFLAPI hasAttributes(
			IF_Db *					pDb,
			FLMBOOL *				pbHasAttrs) = 0;

		virtual RCODE XFLAPI hasNextSibling(
			IF_Db *					pDb,
			FLMBOOL *				pbHasNextSibling) = 0;

		virtual RCODE XFLAPI hasPreviousSibling(
			IF_Db *					pDb,
			FLMBOOL *				pbHasPreviousSibling) = 0;

		virtual RCODE XFLAPI hasChildren(
			IF_Db *					pDb,
			FLMBOOL *				pbHasChildren) = 0;

		virtual RCODE XFLAPI isNamespaceDecl(
			IF_Db *					pDb,
			FLMBOOL *				pbIsNamespaceDecl) = 0;

		virtual eDomNodeType XFLAPI getNodeType( void) = 0;
			
		virtual RCODE XFLAPI getNodeId(
			IF_Db *					pDb,
			FLMUINT64 *				pui64NodeId) = 0;

		virtual RCODE XFLAPI getParentId(
			IF_Db *					pDb,
			FLMUINT64 *				pui64ParentId) = 0;
			
		virtual RCODE XFLAPI getDocumentId(
			IF_Db *					pDb,
			FLMUINT64 *				pui64DocumentId) = 0;

		virtual RCODE XFLAPI getPrevSibId(
			IF_Db *					pDb,
			FLMUINT64 *				pui64PrevSibId) = 0;

		virtual RCODE XFLAPI getNextSibId(
			IF_Db *					pDb,
			FLMUINT64 *				pui64NextSibId) = 0;

		virtual RCODE XFLAPI getFirstChildId(
			IF_Db *					pDb,
			FLMUINT64 *				pui64FirstChildId) = 0;

		virtual RCODE XFLAPI getLastChildId(
			IF_Db *					pDb,
			FLMUINT64 *				pui64LastChildId) = 0;

		virtual RCODE XFLAPI getNameId(
			IF_Db *					pDb,
			FLMUINT *				puiNameId) = 0;

		virtual RCODE XFLAPI getEncDefId(
			IF_Db *					pDb,
			FLMUINT *				puiEncDefId) = 0;

		virtual RCODE XFLAPI getDataType(
			IF_Db *					pDb,
			FLMUINT *				puiDataType) = 0;

		virtual RCODE XFLAPI getDataLength(
			IF_Db *					pDb,
			FLMUINT *				puiLength) = 0;

		virtual RCODE XFLAPI getUINT32(
			IF_Db *					pDb,
			FLMUINT32 *				pui32Value) = 0;
			
		virtual RCODE XFLAPI getUINT(
			IF_Db *					pDb,
			FLMUINT *				puiValue) = 0;

		virtual RCODE XFLAPI getUINT64(
			IF_Db *					pDb,
			FLMUINT64 *				pui64Value) = 0;

		virtual RCODE XFLAPI getINT32(
			IF_Db *					pDb,
			FLMINT32 *				pi32Value) = 0;
			
		virtual RCODE XFLAPI getINT(
			IF_Db *					pDb,
			FLMINT *					piValue) = 0;

		virtual RCODE XFLAPI getINT64(
			IF_Db *					pDb,
			FLMINT64 *				pi64Value) = 0;

		virtual RCODE XFLAPI getMetaValue(
			IF_Db *					pDb,
			FLMUINT64 *				pui64Value) = 0;
			
		virtual RCODE XFLAPI getUnicodeChars(
			IF_Db *					pDb,
			FLMUINT *				puiNumChars) = 0;

		virtual RCODE XFLAPI getUnicode(
			IF_Db *					pDb,
			FLMUNICODE *			puzValueBuffer,
			FLMUINT					uiBufferSize,
			FLMUINT					uiCharOffset,
			FLMUINT					uiMaxCharsRequested,
			FLMUINT *				puiCharsReturned = NULL,
			FLMUINT *				puiBufferBytesUsed = NULL) = 0;

		virtual RCODE XFLAPI getUnicode(
			IF_Db *					pDb,
			FLMUNICODE **			ppuzUnicodeValue) = 0;

		virtual RCODE XFLAPI getUnicode(
			IF_Db *					pDb,
			F_DynaBuf *				pDynaBuf) = 0;
			
		virtual RCODE XFLAPI getUTF8(
			IF_Db *					pDb,
			FLMBYTE *				pucValueBuffer,
			FLMUINT					uiBufferSize,
			FLMUINT					uiCharOffset,
			FLMUINT					uiMaxCharsRequested,
			FLMUINT *				puiCharsReturned = NULL,
			FLMUINT *				puiBufferBytesUsed = NULL) = 0;

		virtual RCODE XFLAPI getUTF8(
			IF_Db *					pDb,
			FLMBYTE **				ppszUTF8Value) = 0;
			
		virtual RCODE XFLAPI getUTF8(
			IF_Db *					pDb,
			F_DynaBuf *				pDynaBuf) = 0;

		virtual RCODE XFLAPI getBinary(
			IF_Db *					pDb,
			void *					pvValue,
			FLMUINT					uiByteOffset,
			FLMUINT					uiBytesRequested,
			FLMUINT *				puiBytesReturned) = 0;

		virtual RCODE XFLAPI getBinary(
			IF_Db *					pDb,
			F_DynaBuf *				pBuffer) = 0;
			
		virtual RCODE XFLAPI getAttributeValueUINT32(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			FLMUINT32 *				pui32Num) = 0;

		virtual RCODE XFLAPI getAttributeValueUINT32(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			FLMUINT32 *				pui32Num,
			FLMUINT32				ui32NotFoundDefault) = 0;
			
		virtual RCODE XFLAPI getAttributeValueUINT(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			FLMUINT *				puiNum) = 0;

		virtual RCODE XFLAPI getAttributeValueUINT(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			FLMUINT *				puiNum,
			FLMUINT					uiNotFoundDefault) = 0;

		virtual RCODE XFLAPI getAttributeValueUINT64(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			FLMUINT64 *				pui64Num) = 0;

		virtual RCODE XFLAPI getAttributeValueUINT64(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			FLMUINT64 *				pui64Num,
			FLMUINT64				ui64NotFoundDefault) = 0;

		virtual RCODE XFLAPI getAttributeValueINT(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			FLMINT *					piNum) = 0;

		virtual RCODE XFLAPI getAttributeValueINT(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			FLMINT *					piNum,
			FLMINT					iNotFoundDefault) = 0;
			
		virtual RCODE XFLAPI getAttributeValueINT64(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			FLMINT64 *				pi64Num) = 0;

		virtual RCODE XFLAPI getAttributeValueINT64(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			FLMINT64 *				pi64Num,
			FLMINT64					i64NotFoundDefault) = 0;
			
		virtual RCODE XFLAPI getAttributeValueUnicode(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			FLMUNICODE *			puzValueBuffer,
			FLMUINT					uiBufferSize,
			FLMUINT *				puiCharsReturned = NULL,
			FLMUINT *				puiBufferBytesUsed = NULL) = 0;

		virtual RCODE XFLAPI getAttributeValueUnicode(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			FLMUNICODE **			ppuzValueBuffer) = 0;

		virtual RCODE XFLAPI getAttributeValueUnicode(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			F_DynaBuf *				pDynaBuf) = 0;
			
		virtual RCODE XFLAPI getAttributeValueUTF8(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			FLMBYTE *				pucValueBuffer,
			FLMUINT					uiBufferSize,
			FLMUINT *				puiCharsReturned = NULL,
			FLMUINT *				puiBufferBytesUsed = NULL) = 0;

		virtual RCODE XFLAPI getAttributeValueUTF8(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			FLMBYTE **				ppszValueBuffer) = 0;
			
		virtual RCODE XFLAPI getAttributeValueUTF8(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			F_DynaBuf *				pDynaBuf) = 0;

		virtual RCODE XFLAPI getAttributeValueBinary(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			void *					pvValueBuffer,
			FLMUINT					uiBufferSize,
			FLMUINT *				puiValueLength) = 0;

		virtual RCODE XFLAPI getAttributeValueBinary(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			F_DynaBuf *				pDynaBuf) = 0;
			
		virtual RCODE XFLAPI setUINT(
			IF_Db *					pDb,
			FLMUINT					uiValue,
			FLMUINT					uiEncDefId = 0) = 0;

		virtual RCODE XFLAPI setUINT64(
			IF_Db *					pDb,
			FLMUINT64				ui64Value,
			FLMUINT					uiEncDefId = 0) = 0;

		virtual RCODE XFLAPI setINT(
			IF_Db *					pDb,
			FLMINT					iValue,
			FLMUINT					uiEncDefId = 0) = 0;

		virtual RCODE XFLAPI setINT64(
			IF_Db *					pDb,
			FLMINT64					i64Value,
			FLMUINT					uiEncDefId = 0) = 0;

		virtual RCODE XFLAPI setMetaValue(
			IF_Db *					pDb,
			FLMUINT64				ui64Value) = 0;

		virtual RCODE XFLAPI setUnicode(
			IF_Db *					pDb,
			const FLMUNICODE *	puzValue,
			FLMUINT					uiValueLength = 0,
			FLMBOOL					bLast = TRUE,
			FLMUINT					uiEncDefId = 0) = 0;

		virtual RCODE XFLAPI setUTF8(
			IF_Db *					pDb,
			const FLMBYTE *		pszValue,
			FLMUINT					uiValueLength = 0,
			FLMBOOL					bLast = TRUE,
			FLMUINT					uiEncDefId = 0) = 0;

		virtual RCODE XFLAPI setBinary(
			IF_Db *					pDb,
			const void *			pvValue,
			FLMUINT					uiValueLength,
			FLMBOOL					bLast = TRUE,
			FLMUINT					uiEncDefId = 0) = 0;

		virtual RCODE XFLAPI setAttributeValueUINT(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			FLMUINT					uiValue,
			FLMUINT					uiEncDefId = 0) = 0;

		virtual RCODE XFLAPI setAttributeValueUINT64(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			FLMUINT64				ui64Value,
			FLMUINT					uiEncDefId = 0) = 0;

		virtual RCODE XFLAPI setAttributeValueINT(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			FLMINT					iValue,
			FLMUINT					uiEncDefId = 0) = 0;

		virtual RCODE XFLAPI setAttributeValueINT64(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			FLMINT64					i64Value,
			FLMUINT					uiEncDefId = 0) = 0;

		virtual RCODE XFLAPI setAttributeValueUnicode(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			const FLMUNICODE *	puzValue,
			FLMUINT					uiEncDefId = 0) = 0;

		virtual RCODE XFLAPI setAttributeValueUTF8(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			const FLMBYTE *		pucValue,
			FLMUINT					uiLength = 0,
			FLMUINT					uiEncDefId = 0) = 0;

		virtual RCODE XFLAPI setAttributeValueBinary(
			IF_Db *					pDb,
			FLMUINT					uiAttrNameId,
			const void *			pvValue,
			FLMUINT					uiLength,
			FLMUINT					uiEncDefId = 0) = 0;

		virtual RCODE XFLAPI getDocumentNode(
			IF_Db *					pDb,
			IF_DOMNode **			ppDocument) = 0;

		virtual RCODE XFLAPI getNextDocument(
			IF_Db *					pDb,
			IF_DOMNode **			ppNextDocument) = 0;

		virtual RCODE XFLAPI getPreviousDocument(
			IF_Db *					pDb,
			IF_DOMNode **			ppPrevDocument) = 0;

		virtual RCODE XFLAPI getParentNode(
			IF_Db *					pDb,
			IF_DOMNode **			ppParent) = 0;

		virtual RCODE XFLAPI getFirstChild(
			IF_Db *					pDb,
			IF_DOMNode **			ppFirstChild) = 0;

		virtual RCODE XFLAPI getLastChild(
			IF_Db *					pDb,
			IF_DOMNode **			ppLastChild) = 0;

		virtual RCODE XFLAPI getNextSibling(
			IF_Db *					pDb,
			IF_DOMNode **			ppNextSibling) = 0;

		virtual RCODE XFLAPI getPreviousSibling(
			IF_Db *					pDb,
			IF_DOMNode **			ppPrevSibling) = 0;

		virtual RCODE XFLAPI getChild(
			IF_Db *					pDb,
			eDomNodeType			eNodeType,
			IF_DOMNode **			ppChild) = 0;

		virtual RCODE XFLAPI getChildElement(
			IF_Db *					pDb,
			FLMUINT					uiElementNameId,
			IF_DOMNode **			ppChild,
			FLMUINT					uiFlags = 0) = 0;

		virtual RCODE XFLAPI getSiblingElement(
			IF_Db *					pDb,
			FLMUINT					uiElementNameId,
			FLMBOOL					bNext,
			IF_DOMNode **			ppSibling) = 0;

		virtual RCODE XFLAPI getAncestorElement(
			IF_Db *					pDb,
			FLMUINT					uiElementNameId,
			IF_DOMNode **			ppAncestor) = 0;
			
		virtual RCODE XFLAPI getDescendantElement(
			IF_Db *					pDb,
			FLMUINT					uiElementNameId,
			IF_DOMNode **			ppDescendant) = 0;
			
		virtual RCODE XFLAPI insertBefore(
			IF_Db *					pDb,
			IF_DOMNode *			pNewChild,
			IF_DOMNode *			pRefChild) = 0;

		virtual RCODE XFLAPI getPrefix(
			IF_Db *					pDb,
			FLMUNICODE *			puzPrefixBuffer,
			FLMUINT					uiBufferSize,
			FLMUINT *				puiCharsReturned = NULL) = 0;

		virtual RCODE XFLAPI getPrefix(
			IF_Db *					pDb,
			char *					pszPrefixBuffer,
			FLMUINT					uiBufferSize,
			FLMUINT *				puiCharsReturned = NULL) = 0;

		virtual RCODE XFLAPI getPrefixId(
			IF_Db *					pDb,
			FLMUINT *				puiPrefixId) = 0;

		virtual RCODE XFLAPI setPrefix(
			IF_Db *					pDb,
			const FLMUNICODE *	puzPrefix) = 0;

		virtual RCODE XFLAPI setPrefix(
			IF_Db *					pDb,
			const char *			pszPrefix) = 0;

		virtual RCODE XFLAPI setPrefixId(
			IF_Db *					pDb,
			FLMUINT					uiPrefixId) = 0;

		virtual RCODE XFLAPI getNamespaceURI(
			IF_Db *					pDb,
			FLMUNICODE *			puzNamespaceURIBuffer,
			FLMUINT					uiBufferSize,
			FLMUINT *				puiCharsReturned = NULL) = 0;

		virtual RCODE XFLAPI getNamespaceURI(
			IF_Db *					pDb,
			char *					pszNamespaceURIBuffer,
			FLMUINT					uiBufferSize,
			FLMUINT *				puiCharsReturned = NULL) = 0;

		virtual RCODE XFLAPI getLocalName(
			IF_Db *					pDb,
			FLMUNICODE *			puzLocalNameBuffer,
			FLMUINT					uiBufferSize,
			FLMUINT *				puiCharsReturned = NULL) = 0;

		virtual RCODE XFLAPI getLocalName(
			IF_Db *					pDb,
			char *					pszLocalNameBuffer,
			FLMUINT					uiBufferSize,
			FLMUINT *				puiCharsReturned = NULL) = 0;

		virtual RCODE XFLAPI getQualifiedName(
			IF_Db *					pDb,
			FLMUNICODE *			puzQualifiedNameBuffer,
			FLMUINT					uiBufferSize,
			FLMUINT *				puiCharsReturned = NULL) = 0;

		virtual RCODE XFLAPI getQualifiedName(
			IF_Db *					pDb,
			char *					pszQualifiedNameBuffer,
			FLMUINT					uiBufferSize,
			FLMUINT *				puiCharsReturned = NULL) = 0;

		virtual RCODE XFLAPI getCollection(
			IF_Db *					pDb,
			FLMUINT *				puiCollection) = 0;

		virtual RCODE XFLAPI createAnnotation(
			IF_Db *					pDb,
			IF_DOMNode **			ppAnnotation,
			FLMUINT64 *				pui64NodeId = NULL) = 0;

		virtual RCODE XFLAPI getAnnotation(
			IF_Db *					pDb,
			IF_DOMNode **			ppAnnotation) = 0;

		virtual RCODE XFLAPI getAnnotationId(
			IF_Db *					pDb,
			FLMUINT64 *				pui64AnnotationId) = 0;
			
		virtual RCODE XFLAPI hasAnnotation(
			IF_Db *					pDb,
			FLMBOOL *				pbHasAnnotation) = 0;

		virtual RCODE XFLAPI getIStream(
			IF_Db *					pDb,
			IF_PosIStream **		ppIStream,
			FLMUINT *				puiDataType = NULL,
			FLMUINT *				puiDataLength = NULL) = 0;

		virtual RCODE XFLAPI getTextIStream(
			IF_Db *					pDb,
			IF_PosIStream **		ppIStream,
			FLMUINT *				puiNumChars = NULL) = 0;

		virtual FLMUINT XFLAPI compareNode(
			IF_DOMNode *			pNode,
			IF_Db *					pDb1,
			IF_Db *					pDb2,
			char *					pszErrBuff,
			FLMUINT					uiErrBuffLen) = 0;

		virtual RCODE XFLAPI isDataLocalToNode(
			IF_Db *					pDb,
			FLMBOOL *				pbDataIsLocal) = 0;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	flminterface XFLEXP IF_DataVector : public F_Object
	{
		virtual void XFLAPI setDocumentID(
			FLMUINT64				ui64DocumentID) = 0;

		virtual RCODE XFLAPI setID(
			FLMUINT					uiElementNumber,
			FLMUINT64				ui64ID) = 0;

		virtual RCODE XFLAPI setNameId(
			FLMUINT					uiElementNumber,
			FLMUINT					uiNameId,
			FLMBOOL					bIsAttr,
			FLMBOOL					bIsData) = 0;

		virtual RCODE XFLAPI setINT(
			FLMUINT					uiElementNumber,
			FLMINT					iNum) = 0;

		virtual RCODE XFLAPI setINT64(
			FLMUINT					uiElementNumber,
			FLMINT64					i64Num) = 0;

		virtual RCODE XFLAPI setUINT(
			FLMUINT					uiElementNumber,
			FLMUINT					uiNum) = 0;

		virtual RCODE XFLAPI setUINT64(
			FLMUINT					uiElementNumber,
			FLMUINT64				ui64Num) = 0;

		virtual RCODE XFLAPI setUnicode(
			FLMUINT					uiElementNumber,
			const FLMUNICODE *	puzUnicode) = 0;

		virtual RCODE XFLAPI setUTF8(
			FLMUINT					uiElementNumber,
			const FLMBYTE *		pszUtf8,
			FLMUINT					uiBytesInBuffer = 0) = 0;			

		virtual RCODE XFLAPI setBinary(
			FLMUINT					uiElementNumber,
			const void *			pvBinary,
			FLMUINT					uiBinaryLen) = 0;
		
		virtual void XFLAPI setRightTruncated(
			FLMUINT					uiElementNumber) = 0;
		
		virtual void XFLAPI setLeftTruncated(
			FLMUINT					uiElementNumber) = 0;
		
		virtual void XFLAPI clearRightTruncated(
			FLMUINT					uiElementNumber) = 0;

		virtual void XFLAPI clearLeftTruncated(
			FLMUINT					uiElementNumber) = 0;

		virtual FLMBOOL XFLAPI isRightTruncated(
			FLMUINT					uiElementNumber) = 0;

		virtual FLMBOOL XFLAPI isLeftTruncated(
			FLMUINT					uiElementNumber) = 0;

		virtual FLMUINT64 XFLAPI getDocumentID( void) = 0;

		virtual FLMUINT64 XFLAPI getID(
			FLMUINT					uiElementNumber) = 0;

		virtual FLMUINT XFLAPI getNameId(
			FLMUINT					uiElementNumber) = 0;

		virtual FLMBOOL XFLAPI isAttr(
			FLMUINT					uiElementNumber) = 0;

		virtual FLMBOOL XFLAPI isDataComponent(
			FLMUINT					uiElementNumber) = 0;

		virtual FLMBOOL XFLAPI isKeyComponent(
			FLMUINT					uiElementNumber) = 0;

		virtual FLMUINT XFLAPI getDataLength(
			FLMUINT					uiElementNumber) = 0;

		virtual FLMUINT XFLAPI getDataType(
			FLMUINT					uiElementNumber) = 0;

		virtual RCODE XFLAPI getUTF8Ptr(
			FLMUINT					uiElementNumber,
			const FLMBYTE **		ppszUTF8,
			FLMUINT *				puiBufLen) = 0;

		virtual RCODE XFLAPI getINT(
			FLMUINT					uiElementNumber,
			FLMINT *					piNum) = 0;

		virtual RCODE XFLAPI getINT64(
			FLMUINT					uiElementNumber,
			FLMINT64 *				pi64Num) = 0;

		virtual RCODE XFLAPI getUINT(
			FLMUINT					uiElementNumber,
			FLMUINT *				puiNum) = 0;

		virtual RCODE XFLAPI getUINT64(
			FLMUINT					uiElementNumber,
			FLMUINT64 *				pui64Num) = 0;

		virtual RCODE XFLAPI getUnicode(
			FLMUINT					uiElementNumber,
			FLMUNICODE **			ppuzUnicode) = 0;

		virtual RCODE XFLAPI getUnicode(
			FLMUINT					uiElementNumber,
			FLMUNICODE *			puzUnicode,
			FLMUINT *				puiBufLen) = 0;

		virtual RCODE XFLAPI getUnicode(
			FLMUINT					uiElementNumber,
			F_DynaBuf *				pBuffer) = 0;
			
		virtual RCODE XFLAPI getUTF8(
			FLMUINT					uiElementNumber,
			FLMBYTE *				pszUTF8,
			FLMUINT *				puiBufLen) = 0;

		virtual RCODE XFLAPI getBinary(
			FLMUINT					uiElementNumber,
			void *					pvBuffer,
			FLMUINT *				puiBufferLen) = 0;

		virtual RCODE XFLAPI outputKey(
			IF_Db *					pDb,
			FLMUINT					uiIndexNum,
			FLMUINT					uiMatchFlags,
			FLMBYTE *				pucKeyBuf,
			FLMUINT					uiKeyBufSize,
			FLMUINT *				puiKeyLen) = 0;

		virtual RCODE XFLAPI outputData(
			IF_Db *					pDb,
			FLMUINT					uiIndexNum,
			FLMBYTE *				pucDataBuf,
			FLMUINT					uiDataBufSize,
			FLMUINT *				puiDataLen) = 0;

		virtual RCODE XFLAPI inputKey(
			IF_Db *					pDb,
			FLMUINT					uiIndexNum,
			const FLMBYTE *		pucKey,
			FLMUINT					uiKeyLen) = 0;

		virtual RCODE XFLAPI inputData(
			IF_Db *					pDb,
			FLMUINT					uiIndexNum,
			const FLMBYTE *		pucData,
			FLMUINT					uiDataLen) = 0;

		// Miscellaneous methods

		virtual void XFLAPI reset( void) = 0;

		virtual const void * XFLAPI getDataPtr(
			FLMUINT					uiElementNumber) = 0;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	flminterface XFLEXP IF_Backup : public F_Object
	{
		virtual FLMUINT64 XFLAPI getBackupTransId( void) = 0;

		virtual FLMUINT64 XFLAPI getLastBackupTransId( void) = 0;

		virtual RCODE XFLAPI backup(
			const char *			pszBackupPath,
			const char *			pszPassword,
			IF_BackupClient *		ifpClient,
			IF_BackupStatus *		ifpStatus,
			FLMUINT *				puiIncSeqNum) = 0;

		virtual RCODE XFLAPI endBackup( void) = 0;
	};

	// Note:  Any interfaces ending in Client or Status are interfaces
	// that XFlaim does not provide implementations of.  They exist to
	// allow XFlaim to pass data back to the client.  Interfaces ending in
	// Status are, generally, informational only, while interfaces ending
	// in Client exist to allow the client to modify the data or take
	// other action.
	
	/****************************************************************************
	Desc:
	****************************************************************************/
	flminterface XFLEXP IF_BackupClient : public F_Object
	{
		virtual RCODE XFLAPI WriteData(
			const void *			pvBuffer,
			FLMUINT					uiBytesToWrite) = 0;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	flminterface XFLEXP IF_BackupStatus : public F_Object
	{
		virtual RCODE XFLAPI backupStatus(
			FLMUINT64				ui64BytesToDo,
			FLMUINT64				ui64BytesDone) = 0;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	flminterface XFLEXP IF_CommitClient : public F_Object
	{
		virtual void XFLAPI commit( 
			IF_Db *					pDb) = 0;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	flminterface XFLEXP IF_EventClient : public F_Object
	{
		virtual void XFLAPI catchEvent(
			eEventType				eEvent,
			IF_Db *					pDb,
			FLMUINT					uiThreadId,
			FLMUINT64				ui64TransID,
			FLMUINT					uiIndexOrCollection,
			FLMUINT64				ui64NodeId,
			RCODE						rc) = 0;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	flminterface XFLEXP IF_IxClient : public F_Object
	{
		virtual RCODE XFLAPI doIndexing(
			IF_Db *					pDb,
			FLMUINT					uiIndexNum,
			FLMUINT					uiCollectionNum,
			IF_DOMNode *			pDocNode) = 0;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	flminterface XFLEXP IF_RestoreStatus : public F_Object
	{
		virtual RCODE XFLAPI reportProgress(
			eRestoreAction *		peAction,
			FLMUINT64				ui64BytesToDo,
			FLMUINT64				ui64BytesDone) = 0;

		virtual RCODE XFLAPI reportError(
			eRestoreAction *		peAction,
			RCODE						rcErr) = 0;

		virtual RCODE XFLAPI reportOpenRflFile(
			eRestoreAction *		peAction,
			FLMUINT					uiFileNum) = 0;

		virtual RCODE XFLAPI reportRflRead(
			eRestoreAction *		peAction,
			FLMUINT					uiFileNum,
			FLMUINT					uiBytesRead) = 0;

		virtual RCODE XFLAPI reportBeginTrans(
			eRestoreAction *		peAction,
			FLMUINT64				ui64TransId) = 0;

		virtual RCODE XFLAPI reportCommitTrans(
			eRestoreAction *		peAction,
			FLMUINT64				ui64TransId) = 0;

		virtual RCODE XFLAPI reportAbortTrans(
			eRestoreAction *		peAction,
			FLMUINT64				ui64TransId) = 0;

		virtual RCODE XFLAPI reportBlockChainFree(
			eRestoreAction *		peAction,
			FLMUINT64				ui64TransId,
			FLMUINT64				ui64MaintDocNum,
			FLMUINT					uiStartBlkAddr,
			FLMUINT					uiEndBlkAddr,
			FLMUINT					uiCount) = 0;

		virtual RCODE XFLAPI reportIndexSuspend(
			eRestoreAction *		peAction,
			FLMUINT64				ui64TransId,
			FLMUINT					uiIndexNum) = 0;

		virtual RCODE XFLAPI reportIndexResume(
			eRestoreAction *		peAction,
			FLMUINT64				ui64TransId,
			FLMUINT					uiIndexNum) = 0;

		virtual RCODE XFLAPI reportReduce(
			eRestoreAction *		peAction,
			FLMUINT64				ui64TransId,
			FLMUINT					uiCount) = 0;

		virtual RCODE XFLAPI reportUpgrade(
			eRestoreAction *		peAction,
			FLMUINT64				ui64TransId,
			FLMUINT					uiOldDbVersion,
			FLMUINT					uiNewDbVersion) = 0;

		virtual RCODE XFLAPI reportEnableEncryption(
			eRestoreAction *		peAction,
			FLMUINT64				ui64TransId) = 0;

		virtual RCODE XFLAPI reportWrapKey(
			eRestoreAction *		peAction,
			FLMUINT64				ui64TransId) = 0;
			
		virtual RCODE XFLAPI reportRollOverDbKey(
			eRestoreAction *		peAction,
			FLMUINT64				ui64TransId) = 0;
			
		virtual RCODE XFLAPI reportDocumentDone(
			eRestoreAction *		peAction,
			FLMUINT64				ui64TransId,
			FLMUINT					uiCollection,
			FLMUINT64				ui64DocumentId) = 0;
			
		virtual RCODE XFLAPI reportNodeDelete(
			eRestoreAction *		peAction,
			FLMUINT64				ui64TransId,
			FLMUINT					uiCollection,
			FLMUINT64				ui64NodeId) = 0;
			
		virtual RCODE XFLAPI reportAttributeDelete(
			eRestoreAction *		peAction,
			FLMUINT64				ui64TransId,
			FLMUINT					uiCollection,
			FLMUINT64				ui64ElementId,
			FLMUINT					uiAttrNameId) = 0;
			
		virtual RCODE XFLAPI reportNodeChildrenDelete(
			eRestoreAction *		peAction,
			FLMUINT64				ui64TransId,
			FLMUINT					uiCollection,
			FLMUINT64				ui64ParentNodeId,
			FLMUINT					uiNameId) = 0;
			
		virtual RCODE XFLAPI reportNodeCreate(
			eRestoreAction *		peAction,
			FLMUINT64				ui64TransId,
			FLMUINT					uiCollection,
			FLMUINT64				ui64RefNodeId,
			eDomNodeType			eNodeType,
			FLMUINT					uiNameId,
			eNodeInsertLoc			eLocation) = 0;
			
		virtual RCODE XFLAPI reportInsertBefore(
			eRestoreAction *		peAction,
			FLMUINT64				ui64TransId,
			FLMUINT					uiCollection,
			FLMUINT64				ui64ParentNodeId,
			FLMUINT64				ui64NewChildNodeId,
			FLMUINT64				ui64RefChildNodeId) = 0;
			
		virtual RCODE XFLAPI reportNodeUpdate(
			eRestoreAction *		peAction,
			FLMUINT64				ui64TransId,
			FLMUINT					uiCollection,
			FLMUINT64				ui64NodeId) = 0;
			
		virtual RCODE XFLAPI reportNodeSetValue(
			eRestoreAction *		peAction,
			FLMUINT64				ui64TransId,
			FLMUINT					uiCollection,
			FLMUINT64				ui64NodeId) = 0;
			
		virtual RCODE XFLAPI reportAttributeSetValue(
			eRestoreAction *		peAction,
			FLMUINT64				ui64TransId,
			FLMUINT					uiCollection,
			FLMUINT64				ui64ElementNodeId,
			FLMUINT					uiAttrNameId) = 0;
			
		virtual RCODE XFLAPI reportNodeFlagsUpdate(
			eRestoreAction *		peAction,
			FLMUINT64				ui64TransId,
			FLMUINT					uiCollection,
			FLMUINT64				ui64NodeId,
			FLMUINT					uiFlags,
			FLMBOOL					bAdd) = 0;
			
		virtual RCODE XFLAPI reportNodeSetPrefixId(
			eRestoreAction *		peAction,
			FLMUINT64				ui64TransId,
			FLMUINT					uiCollection,
			FLMUINT64				ui64NodeId,
			FLMUINT					uiAttrNameId,
			FLMUINT					uiPrefixId) = 0;
			
		virtual RCODE XFLAPI reportNodeSetMetaValue(
			eRestoreAction *		peAction,
			FLMUINT64				ui64TransId,
			FLMUINT					uiCollection,
			FLMUINT64				ui64NodeId,
			FLMUINT64				ui64MetaValue) = 0;
			
		virtual RCODE XFLAPI reportSetNextNodeId(
			eRestoreAction *		peAction,
			FLMUINT64				ui64TransId,
			FLMUINT					uiCollection,
			FLMUINT64				ui64NextNodeId) = 0;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	flminterface XFLEXP IF_RestoreClient : public F_Object
	{
		virtual RCODE XFLAPI openBackupSet( void) = 0;

		virtual RCODE XFLAPI openRflFile(					// Open an RFL file
			FLMUINT					uiFileNum) = 0;

		virtual RCODE XFLAPI openIncFile(					// Open an incremental backup file
			FLMUINT					uiFileNum) = 0;

		virtual RCODE XFLAPI read(
			FLMUINT					uiLength,					// Number of bytes to read
			void *					pvBuffer,					// Buffer to place read bytes into
			FLMUINT *				puiBytesRead) = 0;		// [out] Number of bytes read

		virtual RCODE XFLAPI close( void) = 0;			// Close the current file

		virtual RCODE XFLAPI abortFile( void) = 0;		// Abort processing the file
																		// and close file handles, etc.
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	flminterface XFLEXP IF_UpgradeClient : public F_Object
	{
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	flminterface XFLEXP IF_DbCopyStatus : public F_Object
	{
		virtual RCODE XFLAPI dbCopyStatus(
			FLMUINT64				ui64BytesToCopy,
			FLMUINT64				ui64BytesCopied,
			FLMBOOL					bNewSrcFile,
			const char *			pszSrcFileName,
			const char *			pszDestFileName) = 0;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	flminterface XFLEXP IF_DbRebuildStatus : public F_Object
	{
		virtual RCODE XFLAPI reportRebuild(
			XFLM_REBUILD_INFO *	pRebuild) = 0;
		
		virtual RCODE XFLAPI reportRebuildErr(
			XFLM_CORRUPT_INFO *	pCorruptInfo) = 0;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	flminterface XFLEXP IF_DbCheckStatus : public F_Object
	{
		virtual RCODE XFLAPI reportProgress(
			XFLM_PROGRESS_CHECK_INFO *	pProgCheck) = 0;
		
		virtual RCODE XFLAPI reportCheckErr(
			XFLM_CORRUPT_INFO *	pCorruptInfo,
			FLMBOOL *				pbFix) = 0;
			// [OUT] - If the client sets this to true, then XFlaim will
			// attempt to fix the problem.  NOTE: It is allowable for
			// XFlaim to pass in NULL here!! (This means that the client
			// doesn't have a choice regarding XFlaim's actions.)  The
			// client must check for NULL before attempting to assing a
			// value to this parameter!!
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	flminterface XFLEXP IF_DbRenameStatus : public F_Object
	{
		virtual RCODE XFLAPI dbRenameStatus(
			const char *			pszSrcFileName,
			const char *			pszDstFileName) = 0;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	flminterface XFLEXP IF_IxStatus : public F_Object
	{
		virtual RCODE XFLAPI reportIndex(
			FLMUINT64 				ui64LastDocumentId) = 0;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	flminterface XFLEXP IF_DbInfo : public F_Object
	{
		virtual FLMUINT XFLAPI getNumCollections( void) = 0;
		
		virtual FLMUINT XFLAPI getNumIndexes( void) = 0;

		virtual FLMUINT XFLAPI getNumLogicalFiles( void) = 0;

		virtual FLMUINT64 XFLAPI getFileSize( void) = 0;

		virtual const XFLM_DB_HDR * XFLAPI getDbHdr( void) = 0;

		virtual void XFLAPI getAvailBlockStats(
			FLMUINT64 *				pui64BytesUsed,
			FLMUINT *				puiBlockCount,
			FLMINT32 *				pi32LastError,
			FLMUINT *				puiNumErrors) = 0;

		virtual void XFLAPI getLFHBlockStats(
			FLMUINT64 *				pui64BytesUsed,
			FLMUINT *				puiBlockCount,
			FLMINT32 *				pi32LastError,
			FLMUINT *				puiNumErrors) = 0;

		virtual void XFLAPI getBTreeInfo(
			FLMUINT					uiNthLogicalFile,
			FLMUINT *				puiLfNum,
			eLFileType *			peLfType,
			FLMUINT *				puiRootBlkAddress,
			FLMUINT *				puiNumLevels) = 0;

		virtual void XFLAPI getBTreeBlockStats(
			FLMUINT					uiNthLogicalFile,
			FLMUINT					uiLevel,
			FLMUINT64 *				pui64KeyCount,
			FLMUINT64 *				pui64BytesUsed,
			FLMUINT64 *				pui64ElementCount,
			FLMUINT64 *				pui64ContElementCount,
			FLMUINT64 *				pui64ContElmBytes,
			FLMUINT *				puiBlockCount,
			FLMINT32 *				pi32LastError,
			FLMUINT *				puiNumErrors) = 0;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	flminterface XFLEXP IF_QueryStatus : public F_Object
	{
		virtual RCODE XFLAPI queryStatus(
			XFLM_OPT_INFO *		pOptInfo) = 0;

		virtual RCODE XFLAPI newSource(
			XFLM_OPT_INFO *		pOptInfo) = 0;
			
		virtual RCODE XFLAPI resultSetStatus(
			FLMUINT64	ui64TotalDocsRead,
			FLMUINT64	ui64TotalDocsPassed,
			FLMBOOL		bCanRetrieveDocs) = 0;
			
		virtual RCODE XFLAPI resultSetComplete(
			FLMUINT64	ui64TotalDocsRead,
			FLMUINT64	ui64TotalDocsPassed) = 0;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	flminterface XFLEXP IF_QueryValidator : public F_Object
	{
		virtual RCODE XFLAPI validateNode(
			IF_Db *					pDb,
			IF_DOMNode *			pNode,
			FLMBOOL *				pbPassed) = 0;
	};
	
	/****************************************************************************
	Desc:
	****************************************************************************/
	flminterface XFLEXP IF_QueryValFunc : public F_Object
	{
		// NOTE: pDynaBuf should only be used when returning XFLM_UTF8_VAL or
		// XFLM_BINARY_VAL.  pvVal should be used for all other types.
		// If there are no more values, return NE_XFLM_EOF_HIT or
		// NE_XFLM_BOF_HIT, depending on eValueToGet.
		
		virtual RCODE XFLAPI getValue(
			IF_Db *					pDb,
			IF_DOMNode *			pContextNode,
			ValIterator				eValueToGet,
			eValTypes *				peValType,
			FLMBOOL *				pbLastValue,
			void *					pvVal,
			F_DynaBuf *				pDynaBuf = NULL) = 0;
			
		virtual RCODE XFLAPI cloneSelf(
			IF_QueryValFunc **	ppNewObj) = 0;
	};
	
	/****************************************************************************
	Desc:	Abstract base class which provides the interface that
			XFLAIM uses to allow an application to embed a node source
			inside an XPATH component.
	****************************************************************************/
	flminterface XFLEXP IF_QueryNodeSource : public F_Object
	{
	public:

		// Method that returns the search cost of this object in providing
		// nodes for a query.

		virtual RCODE XFLAPI searchCost(
			IF_Db *					pDb,
			FLMBOOL					bNotted,
			FLMUINT *				puiCost,
			FLMBOOL *				pbMustScan) = 0;

		// Position to and return the first node that satisfies the predicate.

		virtual RCODE XFLAPI getFirst(
			IF_Db *					pDb,
			IF_DOMNode *			pContextNode,
			IF_DOMNode **			ppNode,
			FLMUINT					uiTimeLimit,			// milliseconds
			IF_QueryStatus *		pQueryStatus) = 0;

		// Position to and return the last node that satisfies the predicate.

		virtual RCODE XFLAPI getLast(
			IF_Db *					pDb,
			IF_DOMNode *			pContextNode,
			IF_DOMNode **			ppNode,
			FLMUINT					uiTimeLimit,			// milliseconds
			IF_QueryStatus *		pQueryStatus) = 0;

		// Position to and return the next node that satisfies the predicate.
		// If no prior positioning has been done,
		// position to and return the first node.

		virtual RCODE XFLAPI getNext(
			IF_Db *					pDb,
			IF_DOMNode *			pContextNode,
			IF_DOMNode **			ppNode,
			FLMUINT					uiTimeLimit,			// milliseconds
			IF_QueryStatus *		pQueryStatus) = 0;

		// Position to and return the previous node that satisfies the predicate.
		// If no prior positioning has been done,
		// position to and return the last node.

		virtual RCODE XFLAPI getPrev(
			IF_Db *					pDb,
			IF_DOMNode *			pContextNode,
			IF_DOMNode **			ppNode,
			FLMUINT					uiTimeLimit,			// milliseconds
			IF_QueryStatus *		pQueryStatus) = 0;

		// Return index being used, 0 if none.

		virtual RCODE XFLAPI getIndex(
			IF_Db *					pDb,
			FLMUINT *				puiIndex,
			FLMBOOL *				pbHaveMultiple) = 0;
			
		virtual RCODE XFLAPI getOptInfoCount(
			IF_Db *					pDb,
			FLMUINT *				puiOptInfoCount) = 0;
		
		virtual RCODE XFLAPI getOptInfo(
			IF_Db *					pDb,
			XFLM_OPT_INFO *		pOptInfoArray,
			FLMUINT					uiNumOptInfoStructsToGet) = 0;
			
		// Return a copy of the object.  Result set should be
		// emptied, score should be unset - only the predicate
		// should be preserved.
		// Returns NULL if the copy fails.

		virtual RCODE copy(
			IF_QueryNodeSource **	ppNodeSourceCopy) = 0;

		virtual void releaseResources( void) = 0;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	flminterface XFLEXP IF_OperandComparer : public F_Object
	{
		virtual RCODE XFLAPI compare(
			IF_PosIStream *		pLeftOperandStream,
			IF_PosIStream *		pRightOperandStream,
			FLMINT *					piCompare) = 0;
	};

	#define XFLM_MAX_SORT_KEYS		32
	
	/****************************************************************************
	Desc:
	****************************************************************************/
	flminterface XFLEXP IF_Query : public F_Object
	{
		virtual RCODE XFLAPI setLanguage(
			FLMUINT					uiLanguage) = 0;

		virtual RCODE XFLAPI setCollection(
			FLMUINT					uiCollection) = 0;

		virtual RCODE XFLAPI setupQueryExpr(
			IF_Db *					pDb,
			const FLMUNICODE *	puzQuery) = 0;

		virtual RCODE XFLAPI setupQueryExpr(
			IF_Db *					pDb,
			const char *			pszQueryExpr) = 0;

		virtual RCODE XFLAPI copyCriteria(
			IF_Query *				pSrcQuery) = 0;

		virtual RCODE XFLAPI addXPathComponent(
			eXPathAxisTypes		eXPathAxis,
			eDomNodeType			eNodeType,
			FLMUINT					uiNameId,
			IF_QueryNodeSource *	pNodeSource = NULL) = 0;

		virtual RCODE XFLAPI addOperator(
			eQueryOperators		eOperator,
			FLMUINT					uiCompareRules = 0,
			IF_OperandComparer *	pOpComparer = NULL) = 0;

		virtual RCODE XFLAPI addUnicodeValue(
			const FLMUNICODE *	puzVal) = 0;

		virtual RCODE XFLAPI addUTF8Value(
			const char *			pszVal,
			FLMUINT					uiUTF8Len = 0) = 0;

		virtual RCODE XFLAPI addBinaryValue(
			const void *			pvVal,
			FLMUINT					uiValLen) = 0;

		virtual RCODE XFLAPI addUINTValue(
			FLMUINT					uiVal) = 0;

		virtual RCODE XFLAPI addINTValue(
			FLMINT					iVal) = 0;

		virtual RCODE XFLAPI addUINT64Value(
			FLMUINT64				ui64Val) = 0;
			
		virtual RCODE XFLAPI addINT64Value(
			FLMINT64					i64Val) = 0;

		virtual RCODE XFLAPI addBoolean(
			FLMBOOL					bVal,
			FLMBOOL					bUnknown = FALSE) = 0;

		virtual RCODE XFLAPI addFunction(
			eQueryFunctions		eFunction) = 0;

		virtual RCODE XFLAPI addFunction(
			IF_QueryValFunc *		pFuncObj,
			FLMBOOL					bHasXPathExpr) = 0;

		virtual RCODE XFLAPI getFirst(
			IF_Db *					pDb,
			IF_DOMNode **			ppNode,
			FLMUINT					uiTimeLimit = 0) = 0;	// milliseconds

		virtual RCODE XFLAPI getLast(
			IF_Db *					pDb,
			IF_DOMNode **			ppNode,
			FLMUINT					uiTimeLimit = 0) = 0;	// milliseconds

		virtual RCODE XFLAPI getNext(
			IF_Db *					pDb,
			IF_DOMNode **			ppNode,
			FLMUINT					uiTimeLimit = 0,		// milliseconds
			FLMUINT					uiNumToSkip = 0,
			FLMUINT *				puiNumSkipped = NULL) = 0;

		virtual RCODE XFLAPI getPrev(
			IF_Db *					pDb,
			IF_DOMNode **			ppNode,
			FLMUINT					uiTimeLimit = 0,		// milliseconds
			FLMUINT					uiNumToSkip = 0,
			FLMUINT *				puiNumSkipped = NULL) = 0;

		virtual RCODE XFLAPI getCurrent(
			IF_Db *					pDb,
			IF_DOMNode **			ppNode) = 0;

		virtual void XFLAPI resetQuery( void) = 0;

		virtual RCODE XFLAPI getStatsAndOptInfo(
			FLMUINT *				puiNumOptInfos,
			XFLM_OPT_INFO **		ppOptInfo) = 0;

		virtual void XFLAPI freeStatsAndOptInfo(
			XFLM_OPT_INFO **		ppOptInfo) = 0;

		virtual void XFLAPI setDupHandling(
			FLMBOOL					bRemoveDups) = 0;

		virtual RCODE XFLAPI setIndex(
			FLMUINT					uiIndex) = 0;

		virtual RCODE XFLAPI getIndex(
			IF_Db *					pDb,
			FLMUINT *				puiIndex,
			FLMBOOL *				pbHaveMultiple) = 0;

		virtual RCODE XFLAPI addSortKey(
			void *			pvSortKeyContext,
			FLMBOOL			bChildToContext,
			FLMBOOL			bElement,
			FLMUINT			uiNameId,
			FLMUINT			uiCompareRules,
			FLMUINT			uiLimit,
			FLMUINT			uiKeyComponent,
			FLMBOOL			bSortDescending,
			FLMBOOL			bSortMissingHigh,
			void **			ppvContext) = 0;
			
		virtual RCODE XFLAPI enablePositioning( void) = 0;
		
		virtual RCODE XFLAPI positionTo(
			IF_Db *			pDb,
			IF_DOMNode **	ppNode,
			FLMUINT			uiTimeLimit,
			FLMUINT			uiPosition) = 0;
			
		virtual RCODE XFLAPI positionTo(
			IF_Db *				pDb,
			IF_DOMNode **		ppNode,
			FLMUINT				uiTimeLimit,
			IF_DataVector *	pSearchKey,
			FLMUINT				uiFlags) = 0;

		virtual RCODE XFLAPI getPosition(
			IF_Db *				pDb,
			FLMUINT *			puiPosition) = 0;
			
		virtual RCODE XFLAPI buildResultSet(
			IF_Db *	pDb,
			FLMUINT	uiTimeLimit) = 0;
			
		virtual void XFLAPI stopBuildingResultSet( void) = 0;
		
		virtual RCODE XFLAPI getCounts(
			IF_Db *		pDb,
			FLMUINT		uiTimeLimit,
			FLMBOOL		bPartialCountOk,
			FLMUINT *	puiReadCount,
			FLMUINT *	puiPassedCount,
			FLMUINT *	puiPositionableToCount,
			FLMBOOL *	pbDoneBuildingResultSet = NULL) = 0;
			
		virtual void XFLAPI enableResultSetEncryption( void) = 0;

		virtual void XFLAPI setQueryStatusObject(
			IF_QueryStatus *		pQueryStatus) = 0;
	
		virtual void XFLAPI setQueryValidatorObject(
			IF_QueryValidator *		pQueryValidator) = 0;
	};

	typedef struct XFLM_NODE_INFO_ITEM
	{
		FLMUINT64	ui64Count;
		FLMUINT64	ui64Bytes;
	} XFLM_NODE_INFO_ITEM;

	typedef struct XFLM_NODE_INFO
	{
		XFLM_NODE_INFO_ITEM		headerSize;
		XFLM_NODE_INFO_ITEM		nodeAndDataType;
		XFLM_NODE_INFO_ITEM		flags;
		XFLM_NODE_INFO_ITEM		nameId;
		XFLM_NODE_INFO_ITEM		prefixId;
		XFLM_NODE_INFO_ITEM		baseId;
		XFLM_NODE_INFO_ITEM		documentId;
		XFLM_NODE_INFO_ITEM		parentId;
		XFLM_NODE_INFO_ITEM		prevSibId;
		XFLM_NODE_INFO_ITEM		nextSibId;
		XFLM_NODE_INFO_ITEM		firstChildId;
		XFLM_NODE_INFO_ITEM		lastChildId;
		XFLM_NODE_INFO_ITEM		childElmCount;
		XFLM_NODE_INFO_ITEM		dataChildCount;
		XFLM_NODE_INFO_ITEM		attrCount;
		XFLM_NODE_INFO_ITEM		attrBaseId;
		XFLM_NODE_INFO_ITEM		attrFlags;
		XFLM_NODE_INFO_ITEM		attrPayloadLen;
		XFLM_NODE_INFO_ITEM		annotationId;
		XFLM_NODE_INFO_ITEM		metaValue;
		XFLM_NODE_INFO_ITEM		encDefId;
		XFLM_NODE_INFO_ITEM		unencDataLen;
		XFLM_NODE_INFO_ITEM		childElmNameId;
		XFLM_NODE_INFO_ITEM		childElmNodeId;
		XFLM_NODE_INFO_ITEM		encIV;
		XFLM_NODE_INFO_ITEM		encPadding;

		// Total overhead - sum of all of the above types of overhead.

		XFLM_NODE_INFO_ITEM		totalOverhead;

		// Data totals

		XFLM_NODE_INFO_ITEM		dataNodata;
		XFLM_NODE_INFO_ITEM		dataString;
		XFLM_NODE_INFO_ITEM		dataNumeric;
		XFLM_NODE_INFO_ITEM		dataBinary;

		// Summary - contains both overhead and data for each type of node

		XFLM_NODE_INFO_ITEM		attributeNode;
		XFLM_NODE_INFO_ITEM		elementNode;
		XFLM_NODE_INFO_ITEM		dataNode;
		XFLM_NODE_INFO_ITEM		commentNode;
		XFLM_NODE_INFO_ITEM		otherNode;
	} XFLM_NODE_INFO;
		
	/****************************************************************************
	Desc:	Node Info. Gatherer
	****************************************************************************/
	flminterface XFLEXP IF_NodeInfo : public F_Object
	{
		virtual void XFLAPI clearNodeInfo( void) = 0;
		
		virtual RCODE XFLAPI addNodeInfo(
			IF_Db *			pDb,
			IF_DOMNode *	pNode,
			FLMBOOL			bDoSubTree,
			FLMBOOL			bDoSelf = TRUE) = 0;
			
		virtual FLMUINT64 XFLAPI getTotalNodeCount( void) = 0;
		
		virtual void XFLAPI getNodeInfo(
			XFLM_NODE_INFO *	pNodeInfo) = 0;
	};
	
	/****************************************************************************
	Desc:	Types of information that can be gathered about a B-Tree.
	****************************************************************************/
	typedef struct XFLM_BTREE_LEVEL_INFO
	{
		FLMUINT64		ui64BlockCount;
		FLMUINT64		ui64BlockLength;
		FLMUINT64		ui64BlockFreeSpace;
		FLMUINT64		ui64ElmOffsetOverhead;
		FLMUINT64		ui64ElmCount;
		FLMUINT64		ui64ContElmCount;
		FLMUINT64		ui64ElmFlagOvhd;
		FLMUINT64		ui64ElmKeyLengthOvhd;
		FLMUINT64		ui64ElmCountsOvhd;
		FLMUINT64		ui64ElmChildAddrsOvhd;
		FLMUINT64		ui64ElmDataLenOvhd;
		FLMUINT64		ui64ElmOADataLenOvhd;
		FLMUINT64		ui64ElmKeyLength;
		FLMUINT64		ui64ElmDataLength;
		
		// The following three are how ui64ElmKeyLength is subdivided.
		// They are only applicable to index keys.
		
		FLMUINT64		ui64KeyDataSize;
		FLMUINT64		ui64KeyIdSize;
		FLMUINT64		ui64KeyComponentLengthsSize;
		
		// Data only blocks
		
		FLMUINT64		ui64DataOnlyBlockCount;
		FLMUINT64		ui64DataOnlyBlockLength;
		FLMUINT64		ui64DataOnlyBlockFreeSpace;
	} XFLM_BTREE_LEVEL_INFO;
	
	/****************************************************************************
	Desc:
	****************************************************************************/
	flminterface XFLEXP IF_BTreeInfoStatus : public F_Object
	{
		virtual RCODE XFLAPI infoStatus(
			FLMUINT		uiCurrLfNum,
			FLMBOOL		bIsCollection,
			char *		pszCurrLfName,
			FLMUINT		uiCurrLevel,
			FLMUINT64	ui64CurrLfBlockCount,
			FLMUINT64	ui64CurrLevelBlockCount,
			FLMUINT64	ui64TotalBlockCount) = 0;
	};

	/****************************************************************************
	Desc:	BTree Info. Gatherer
	****************************************************************************/
	flminterface XFLEXP IF_BTreeInfo : public F_Object
	{
		virtual void XFLAPI clearBTreeInfo( void) = 0;
		
		virtual RCODE XFLAPI collectIndexInfo(
			IF_Db *					pDb,
			FLMUINT					uiIndexNum,
			IF_BTreeInfoStatus *	pInfoStatus) = 0;
			
		virtual RCODE XFLAPI collectCollectionInfo(
			IF_Db *					pDb,
			FLMUINT					uiCollectionNum,
			IF_BTreeInfoStatus *	pInfoStatus) = 0;
			
		virtual FLMUINT XFLAPI getNumIndexes( void) = 0;
			
		virtual FLMUINT XFLAPI getNumCollections( void) = 0;

		virtual FLMBOOL XFLAPI getIndexInfo(
			FLMUINT		uiNthIndex,
			FLMUINT *	puiIndexNum,
			char **		ppszIndexName,
			FLMUINT *	puiNumLevels) = 0;
			
		virtual FLMBOOL XFLAPI getCollectionInfo(
			FLMUINT		uiNthCollection,
			FLMUINT *	puiCollectionNum,
			char **		ppszCollectionName,
			FLMUINT *	puiNumLevels) = 0;
			
		virtual FLMBOOL XFLAPI getIndexLevelInfo(
			FLMUINT						uiNthIndex,
			FLMUINT						uiBTreeLevel,
			XFLM_BTREE_LEVEL_INFO *	pLevelInfo) = 0;

		virtual FLMBOOL XFLAPI getCollectionLevelInfo(
			FLMUINT						uiNthCollection,
			FLMUINT						uiBTreeLevel,
			XFLM_BTREE_LEVEL_INFO *	pLevelInfo) = 0;
	};
	
	/****************************************************************************
	Desc:    Status and return codes
	****************************************************************************/

	// IMPORTANT NOTE: If changes are made to the RCODEs, please be sure to make
	// corresponding changes in csharp/xflaim/RCODE.cs and java/xflaim/RCODE.java.
	#define NE_XFLM_NOT_IMPLEMENTED							NE_FLM_NOT_IMPLEMENTED
	#define NE_XFLM_MEM											NE_FLM_MEM
	#define NE_XFLM_INVALID_PARM								NE_FLM_INVALID_PARM
	#define NE_XFLM_NOT_FOUND									NE_FLM_NOT_FOUND
	#define NE_XFLM_EXISTS										NE_FLM_EXISTS
	#define NE_XFLM_FAILURE										NE_FLM_FAILURE
	#define NE_XFLM_BOF_HIT										NE_FLM_BOF_HIT
	#define NE_XFLM_EOF_HIT										NE_FLM_EOF_HIT
	#define NE_XFLM_CONV_DEST_OVERFLOW						NE_FLM_CONV_DEST_OVERFLOW
	#define NE_XFLM_CONV_ILLEGAL								NE_FLM_CONV_ILLEGAL
	#define NE_XFLM_CONV_NUM_OVERFLOW						NE_FLM_CONV_NUM_OVERFLOW
	#define NE_XFLM_SYNTAX										NE_FLM_SYNTAX
	#define NE_XFLM_ILLEGAL_OP									NE_FLM_ILLEGAL_OP
	#define NE_XFLM_BAD_SEN										NE_FLM_BAD_SEN
	#define NE_XFLM_COULD_NOT_START_THREAD					NE_FLM_COULD_NOT_START_THREAD
	#define NE_XFLM_BAD_BASE64_ENCODING						NE_FLM_BAD_BASE64_ENCODING
	#define NE_XFLM_STREAM_EXISTS								NE_FLM_STREAM_EXISTS
	#define NE_XFLM_MULTIPLE_MATCHES							NE_FLM_MULTIPLE_MATCHES
	#define NE_XFLM_NOT_UNIQUE									NE_FLM_NOT_UNIQUE
	#define NE_XFLM_BTREE_ERROR								NE_FLM_BTREE_ERROR
	#define NE_XFLM_BTREE_KEY_SIZE							NE_FLM_BTREE_KEY_SIZE
	#define NE_XFLM_BTREE_FULL									NE_FLM_BTREE_FULL
	#define NE_XFLM_BTREE_BAD_STATE							NE_FLM_BTREE_BAD_STATE
	#define NE_XFLM_COULD_NOT_CREATE_MUTEX					NE_FLM_COULD_NOT_CREATE_MUTEX
	#define NE_XFLM_DATA_ERROR									NE_FLM_DATA_ERROR
	#define NE_XFLM_IO_PATH_NOT_FOUND						NE_FLM_IO_PATH_NOT_FOUND
	#define NE_XFLM_IO_END_OF_FILE							NE_FLM_IO_END_OF_FILE
	#define NE_XFLM_IO_NO_MORE_FILES							NE_FLM_IO_NO_MORE_FILES
	#define NE_XFLM_COULD_NOT_CREATE_SEMAPHORE			NE_FLM_COULD_NOT_CREATE_SEMAPHORE
	#define NE_XFLM_BAD_UTF8									NE_FLM_BAD_UTF8
	#define NE_XFLM_ERROR_WAITING_ON_SEMPAHORE			NE_FLM_ERROR_WAITING_ON_SEMPAHORE
	#define NE_XFLM_BAD_PLATFORM_FORMAT						NE_FLM_BAD_PLATFORM_FORMAT						
	
	/****************************************************************************
	Desc:		General XFLAIM errors
	****************************************************************************/
	// IMPORTANT NOTE: If changes are made to the RCODEs, please be sure to make
	// corresponding changes in csharp/xflaim/RCODE.cs and java/xflaim/RCODE.java.
	#define NE_XFLM_OK											NE_FLM_OK
	
	#define NE_XFLM_USER_ABORT									0xD100			// User or application aborted (canceled) operation.
	#define NE_XFLM_BAD_PREFIX									0xD101			// Invalid XLM namespace prefix specified.  Either a prefix name or number that was specified was not defined.
	#define NE_XFLM_ATTRIBUTE_PURGED							0xD102			// XML attribute cannot be used - it is being deleted from the database.
	#define NE_XFLM_BAD_COLLECTION							0xD103			// Invalid collection number specified.  Collection is not defined.
	#define NE_XFLM_DATABASE_LOCK_REQ_TIMEOUT				0xD104			// Request to lock the database timed out.
	#define NE_XFLM_ILLEGAL_DATA_COMPONENT					0xD105			// Cannot use ELM_ROOT_TAG as a data component in an index.
	#define NE_XFLM_MUST_INDEX_ON_PRESENCE					0xD106			// When using ELM_ROOT_TAG in an index component, must specify PRESENCE indexing only.
	#define NE_XFLM_BAD_IX										0xD107			// Invalid index number specified.  Index is not defined.
	#define NE_XFLM_BACKUP_ACTIVE								0xD108			// Operation could not be performed because a backup is currently in progress.
	#define NE_XFLM_SERIAL_NUM_MISMATCH						0xD109			// Serial number on backup file does not match the serial number that is expected.
	#define NE_XFLM_BAD_RFL_DB_SERIAL_NUM					0xD10A			// Bad database serial number in roll-forward log file header.
	#define NE_XFLM_BAD_RFL_FILE_NUMBER						0xD10B			// Bad roll-forward log file number in roll-forward log file header.
	#define NE_XFLM_CANNOT_DEL_ELEMENT						0xD10C			// Cannot delete an XML element definition in the dictionary because it is in use.
	#define NE_XFLM_CANNOT_MOD_DATA_TYPE					0xD10D			// Cannot modify the data type for an XML element or attribute definition in the dictionary.
	#define NE_XFLM_CANNOT_INDEX_DATA_TYPE					0xD10E			// Data type of XML element or attribute is not one that can be indexed.
	#define NE_XFLM_BAD_ELEMENT_NUM							0xD10F			// Bad element number specified - element not defined in dictionary.
	#define NE_XFLM_BAD_ATTRIBUTE_NUM						0xD110			// Bad attribute number specified - attribute not defined in dictionary.
	#define NE_XFLM_BAD_ENCDEF_NUM							0xD111			// Bad encryption number specified - encryption definition not defined in dictionary.
	#define NE_XFLM_INVALID_FILE_SEQUENCE					0xD112			// Incremental backup file number provided during a restore is invalid.
	#define NE_XFLM_DUPLICATE_ELEMENT_NUM					0xD113			// Element number specified in element definition is already in use.
	#define NE_XFLM_ILLEGAL_TRANS_TYPE						0xD114			// Illegal transaction type specified for transaction begin operation.
	#define NE_XFLM_UNSUPPORTED_VERSION						0xD115			// Version of database found in database header is not supported.
	#define NE_XFLM_ILLEGAL_TRANS_OP							0xD116			// Illegal operation for transaction type.
	#define NE_XFLM_INCOMPLETE_LOG							0xD117			// Incomplete rollback log.
	#define NE_XFLM_ILLEGAL_INDEX_DEF						0xD118			// Index definition document is illegal - does not conform to the expected form of an index definition document.
	#define NE_XFLM_ILLEGAL_INDEX_ON							0xD119			// The "IndexOn" attribute of an index definition has an illegal value.
	#define NE_XFLM_ILLEGAL_STATE_CHANGE					0xD11A			// Attempted an illegal state change on an element or attribute definition.
	#define NE_XFLM_BAD_RFL_SERIAL_NUM						0xD11B			// Serial number in roll-forward log file header does not match expected serial number.
	#define NE_XFLM_NEWER_FLAIM								0xD11C			// Running old code on a newer version of database.  Newer code must be used.
	#define NE_XFLM_CANNOT_MOD_ELEMENT_STATE				0xD11D			// Attempted to change state of a predefined element definition.
	#define NE_XFLM_CANNOT_MOD_ATTRIBUTE_STATE			0xD11E			// Attempted to change state of a predefined attribute definition.
	#define NE_XFLM_NO_MORE_ELEMENT_NUMS					0xD11F			// The highest element number has already been used, cannot create more element definitions.
	#define NE_XFLM_NO_TRANS_ACTIVE							0xD120			// Operation must be performed inside a database transaction.
	#define NE_XFLM_NOT_FLAIM									0xD121			// The file specified is not a FLAIM database.
	#define NE_XFLM_OLD_VIEW									0xD122			// Unable to maintain read transaction's view of the database.
	#define NE_XFLM_SHARED_LOCK								0xD123			// Attempted to perform an operation on the database that requires exclusive access, but cannot because there is a shared lock.
	#define NE_XFLM_TRANS_ACTIVE								0xD124			// Operation cannot be performed while a transaction is active.
	#define NE_XFLM_RFL_TRANS_GAP								0xD125			// A gap was found in the transaction sequence in the roll-forward log.
	#define NE_XFLM_BAD_COLLATED_KEY							0xD126			// Something in collated key is bad.
	#define NE_XFLM_MUST_DELETE_INDEXES						0xD127			// Attempting to delete a collection that has indexes defined for it.  Associated indexes must be deleted before the collection can be deleted.
	#define NE_XFLM_RFL_INCOMPLETE							0xD128			// Roll-forward log file is incomplete.
	#define NE_XFLM_CANNOT_RESTORE_RFL_FILES				0xD129			// Cannot restore roll-forward log files - not using multiple roll-forward log files.
	#define NE_XFLM_INCONSISTENT_BACKUP						0xD12A			// A problem (corruption), etc. was detected in a backup set.
	#define NE_XFLM_BLOCK_CRC									0xD12B			// CRC for database block was invalid.  May indicate problems in reading from or writing to disk.
	#define NE_XFLM_ABORT_TRANS								0xD12C			// Attempted operation after a critical error - transaction should be aborted.
	#define NE_XFLM_NOT_RFL										0xD12D			// File was not a roll-forward log file as expected.
	#define NE_XFLM_BAD_RFL_PACKET							0xD12E			// Roll-forward log file packet was bad.
	#define NE_XFLM_DATA_PATH_MISMATCH						0xD12F			// Bad data path specified to open database.  Does not match data path specified for prior opens of the database.
	#define NE_XFLM_MUST_CLOSE_DATABASE						0xD130			// Database must be closed due to a critical error.
	#define NE_XFLM_INVALID_ENCKEY_CRC						0xD131			// Encryption key CRC could not be verified.
	#define NE_XFLM_HDR_CRC										0xD132			// Database header has a bad CRC.
	#define NE_XFLM_NO_NAME_TABLE								0xD133			// No name table was set up for the database.
	#define NE_XFLM_UNALLOWED_UPGRADE						0xD134			// Cannot upgrade database from one version to another.
	#define NE_XFLM_DUPLICATE_ATTRIBUTE_NUM				0xD135			// Attribute number specified in attribute definition is already in use.
	#define NE_XFLM_DUPLICATE_INDEX_NUM						0xD136			// Index number specified in index definition is already in use.
	#define NE_XFLM_DUPLICATE_COLLECTION_NUM				0xD137			// Collection number specified in collection definition is already in use.
	#define NE_XFLM_DUPLICATE_ELEMENT_NAME					0xD138			// Element name+namespace specified in element definition is already in use.
	#define NE_XFLM_DUPLICATE_ATTRIBUTE_NAME				0xD139			// Attribute name+namespace specified in attribute definition is already in use.
	#define NE_XFLM_DUPLICATE_INDEX_NAME					0xD13A			// Index name specified in index definition is already in use.
	#define NE_XFLM_DUPLICATE_COLLECTION_NAME				0xD13B			// Collection name specified in collection definition is already in use.
	#define NE_XFLM_ELEMENT_PURGED							0xD13C			// XML element cannot be used - it is deleted from the database.
	#define NE_XFLM_TOO_MANY_OPEN_DATABASES				0xD13D			// Too many open databases, cannot open another one.
	#define NE_XFLM_DATABASE_OPEN								0xD13E			// Operation cannot be performed because the database is currently open.
	#define NE_XFLM_CACHE_ERROR								0xD13F			// Cached database block has been compromised while in cache.
	#define NE_XFLM_DB_FULL										0xD140			// Database is full, cannot create more blocks.
	#define NE_XFLM_QUERY_SYNTAX								0xD141			// Query expression had improper syntax.
	#define NE_XFLM_INDEX_OFFLINE								0xD142			// Index is offline, cannot be used in a query.
	#define NE_XFLM_RFL_DISK_FULL								0xD143			// Disk which contains roll-forward log is full.
	#define NE_XFLM_MUST_WAIT_CHECKPOINT					0xD144			// Must wait for a checkpoint before starting transaction - due to disk problems - usually in disk containing roll-forward log files.
	#define NE_XFLM_MISSING_ENC_ALGORITHM					0xD145			// Encryption definition is missing an encryption algorithm.
	#define NE_XFLM_INVALID_ENC_ALGORITHM					0xD146			// Invalid encryption algorithm specified in encryption definition.
	#define NE_XFLM_INVALID_ENC_KEY_SIZE					0xD147			// Invalid key size specified in encryption definition.
	#define NE_XFLM_ILLEGAL_DATA_TYPE						0xD148			// Data type specified for XML element or attribute definition is illegal.
	#define NE_XFLM_ILLEGAL_STATE								0xD149			// State specified for index definition or XML element or attribute definition is illegal.
	#define NE_XFLM_ILLEGAL_ELEMENT_NAME					0xD14A			// XML element name specified in element definition is illegal.
	#define NE_XFLM_ILLEGAL_ATTRIBUTE_NAME					0xD14B			// XML attribute name specified in attribute definition is illegal.
	#define NE_XFLM_ILLEGAL_COLLECTION_NAME				0xD14C			// Collection name specified in collection definition is illegal.
	#define NE_XFLM_ILLEGAL_INDEX_NAME						0xD14D			// Index name specified is illegal
	#define NE_XFLM_ILLEGAL_ELEMENT_NUMBER					0xD14E			// Element number specified in element definition or index definition is illegal.
	#define NE_XFLM_ILLEGAL_ATTRIBUTE_NUMBER				0xD14F			// Attribute number specified in attribute definition or index definition is illegal.
	#define NE_XFLM_ILLEGAL_COLLECTION_NUMBER				0xD150			// Collection number specified in collection definition or index definition is illegal.
	#define NE_XFLM_ILLEGAL_INDEX_NUMBER					0xD151			// Index number specified in index definition is illegal.
	#define NE_XFLM_ILLEGAL_ENCDEF_NUMBER					0xD152			// Encryption definition number specified in encryption definition is illegal.
	#define NE_XFLM_COLLECTION_NAME_MISMATCH				0xD153			// Collection name and number specified in index definition do not correspond to each other.
	#define NE_XFLM_ELEMENT_NAME_MISMATCH					0xD154			// Element name+namespace and number specified in index definition do not correspond to each other.
	#define NE_XFLM_ATTRIBUTE_NAME_MISMATCH				0xD155			// Attribute name+namespace and number specified in index definition do not correspond to each other.
	#define NE_XFLM_INVALID_COMPARE_RULE					0xD156			// Invalid comparison rule specified in index definition.
	#define NE_XFLM_DUPLICATE_KEY_COMPONENT				0xD157			// Duplicate key component number specified in index definition.
	#define NE_XFLM_DUPLICATE_DATA_COMPONENT				0xD158			// Duplicate data component number specified in index definition.
	#define NE_XFLM_MISSING_KEY_COMPONENT					0xD159			// Index definition is missing a key component.
	#define NE_XFLM_MISSING_DATA_COMPONENT					0xD15A			// Index definition is missing a data component.
	#define NE_XFLM_INVALID_INDEX_OPTION					0xD15B			// Invalid index option specified on index definition.
	#define NE_XFLM_NO_MORE_ATTRIBUTE_NUMS					0xD15C			// The highest attribute number has already been used, cannot create more.
	#define NE_XFLM_MISSING_ELEMENT_NAME					0xD15D			// Missing element name in XML element definition.
	#define NE_XFLM_MISSING_ATTRIBUTE_NAME					0xD15E			// Missing attribute name in XML attribute definition.
	#define NE_XFLM_MISSING_ELEMENT_NUMBER					0xD15F			// Missing element number in XML element definition.
	#define NE_XFLM_MISSING_ATTRIBUTE_NUMBER				0xD160			// Missing attribute number from XML attribute definition.
	#define NE_XFLM_MISSING_INDEX_NAME						0xD161			// Missing index name in index definition.
	#define NE_XFLM_MISSING_INDEX_NUMBER					0xD162			// Missing index number in index definition.
	#define NE_XFLM_MISSING_COLLECTION_NAME				0xD163			// Missing collection name in collection definition.
	#define NE_XFLM_MISSING_COLLECTION_NUMBER				0xD164			// Missing collection number in collection definition.
	#define NE_XFLM_MISSING_ENCDEF_NAME						0xD165			// Missing encryption definition name in encryption definition.
	#define NE_XFLM_MISSING_ENCDEF_NUMBER					0xD166			// Missing encryption definition number in encryption definition.
	#define NE_XFLM_NO_MORE_INDEX_NUMS						0xD167			// The highest index number has already been used, cannot create more.
	#define NE_XFLM_NO_MORE_COLLECTION_NUMS				0xD168			// The highest collection number has already been used, cannot create more.
	#define NE_XFLM_CANNOT_DEL_ATTRIBUTE					0xD169			// Cannot delete an XML attribute definition because it is in use.
	#define NE_XFLM_TOO_MANY_PENDING_NODES					0xD16A			// Too many documents in the pending document list.
	#define NE_XFLM_BAD_USE_OF_ELM_ROOT_TAG				0xD16B			// ELM_ROOT_TAG, if used, must be the sole root component of an index definition.
	#define NE_XFLM_DUP_SIBLING_IX_COMPONENTS				0xD16C			// Sibling components in an index definition cannot have the same XML element or attribute number.
	#define NE_XFLM_RFL_FILE_NOT_FOUND						0xD16D			// Could not open a roll-forward log file - was not found in the roll-forward log directory.
	#define NE_XFLM_ILLEGAL_KEY_COMPONENT_NUM				0xD16E			// Key component of zero in index definition is not allowed.
	#define NE_XFLM_ILLEGAL_DATA_COMPONENT_NUM			0xD16F			// Data component of zero in index definition is not allowed.
	#define NE_XFLM_ILLEGAL_PREFIX_NUMBER					0xD170			// Prefix number specified in prefix definition is illegal.
	#define NE_XFLM_MISSING_PREFIX_NAME						0xD171			// Missing prefix name in prefix definition.
	#define NE_XFLM_MISSING_PREFIX_NUMBER					0xD172			// Missing prefix number in prefix definition.
	#define NE_XFLM_UNDEFINED_ELEMENT_NAME					0xD173			// XML element name+namespace that was specified in index definition or XML document is not defined in dictionary.
	#define NE_XFLM_UNDEFINED_ATTRIBUTE_NAME				0xD174			// XML attribute name+namespace that was specified in index definition or XML document is not defined in dictionary.
	#define NE_XFLM_DUPLICATE_PREFIX_NAME					0xD175			// Prefix name specified in prefix definition is already in use.
	#define NE_XFLM_NAMESPACE_NOT_ALLOWED					0xD176			// Cannot define a namespace for XML attributes whose name begins with "xmlns:" or that is equal to "xmlns"
	#define NE_XFLM_INVALID_NAMESPACE_DECL					0xD177			// Name for namespace declaration attribute must be "xmlns" or begin with "xmlns:"
	#define NE_XFLM_ILLEGAL_NAMESPACE_DECL_DATATYPE		0xD178			// Data type for XML attributes that are namespace declarations must be text.
	#define NE_XFLM_NO_MORE_PREFIX_NUMS						0xD179			// The highest prefix number has already been used, cannot create more.
	#define NE_XFLM_NO_MORE_ENCDEF_NUMS						0xD17A			// The highest encryption definition number has already been used, cannot create more.
	#define NE_XFLM_COLLECTION_OFFLINE						0xD17B			// Collection is encrypted, cannot be accessed while in operating in limited mode.
	#define NE_XFLM_DELETE_NOT_ALLOWED						0xD17C			// Item cannot be deleted.
	#define NE_XFLM_RESET_NEEDED								0xD17D			// Used during check operations to indicate we need to reset the view.  NOTE: This is an internal error code and should not be documented.
	#define NE_XFLM_ILLEGAL_REQUIRED_VALUE					0xD17E			// An illegal value was specified for the "Required" attribute in an index definition.
	#define NE_XFLM_ILLEGAL_INDEX_COMPONENT				0xD17F			// A leaf index component in an index definition was not marked as a data component or key component.
	#define NE_XFLM_ILLEGAL_UNIQUE_SUB_ELEMENT_VALUE	0xD180			// Illegal value for the "UniqueSubElements" attribute in an element definition.
	#define NE_XFLM_DATA_TYPE_MUST_BE_NO_DATA				0xD181			// Data type for an element definition with UniqueSubElements="yes" must be nodata.
	#define NE_XFLM_CANNOT_SET_REQUIRED						0xD182			// Cannot set the "Required" attribute on a non-key index component in index definition.
	#define NE_XFLM_CANNOT_SET_LIMIT							0xD183			// Cannot set the "Limit" attribute on a non-key index component in index definition.
	#define NE_XFLM_CANNOT_SET_INDEX_ON						0xD184			// Cannot set the "IndexOn" attribute on a non-key index component in index definition.
	#define NE_XFLM_CANNOT_SET_COMPARE_RULES				0xD185			// Cannot set the "CompareRules" on a non-key index component in index definition.
	#define NE_XFLM_INPUT_PENDING								0xD186			// Attempt to set a value while an input stream is still open.
	#define NE_XFLM_INVALID_NODE_TYPE						0xD187			// Bad node type
	#define NE_XFLM_INVALID_CHILD_ELM_NODE_ID				0xD188			// Attempt to insert a unique child element that has a lower node ID than the parent element
	#define NE_XFLM_RFL_END										0xD189			// Hit the end of the RFL
	#define NE_XFLM_ILLEGAL_FLAG								0xD18A			// Illegal flag passed to getChildElement method.  Must be zero for elements that can have non-unique child elements.
	#define NE_XFLM_TIMEOUT										0xD18B			// Operation timed out.
	#define NE_XFLM_CONV_BAD_DIGIT							0xD18C			// Non-numeric digit found in text to numeric conversion.
	#define NE_XFLM_CONV_NULL_SRC								0xD18D			// Data source cannot be NULL when doing data conversion.
	#define NE_XFLM_CONV_NUM_UNDERFLOW						0xD18E			// Numeric underflow (< lower bound) converting to numeric type.
	#define NE_XFLM_UNSUPPORTED_FEATURE						0xD18F			// Attempting to use a feature for which full support has been disabled.
	#define NE_XFLM_FILE_EXISTS								0xD190			// Attempt to create a database, but the file already exists.
	#define NE_XFLM_BUFFER_OVERFLOW							0xD191			// Buffer overflow.
	#define NE_XFLM_INVALID_XML								0xD192			// Invalid XML encountered while parsing document.
	#define NE_XFLM_BAD_DATA_TYPE								0xD193			// Attempt to set/get data on an XML element or attribute using a data type that is incompatible with the data type specified in the dictionary.
	#define NE_XFLM_READ_ONLY									0xD194			// Item is read-only and cannot be updated.
	#define NE_XFLM_KEY_OVERFLOW								0xD195			// Generated index key too large.
	#define NE_XFLM_UNEXPECTED_END_OF_INPUT				0xD196			// Encountered unexpected end of input when parsing XPATH expression.
	
	/****************************************************************************
	Desc:		DOM Errors
	****************************************************************************/

	// IMPORTANT NOTE: If changes are made to the RCODEs, please be sure to make
	// corresponding changes in csharp/xflaim/RCODE.cs and java/xflaim/RCODE.java.
	#define NE_XFLM_DOM_HIERARCHY_REQUEST_ERR				0xD201			// Attempt to insert a DOM node somewhere it doesn't belong.
	#define NE_XFLM_DOM_WRONG_DOCUMENT_ERR					0xD202			// A DOM node is being used in a different document than the one that created it.
	#define NE_XFLM_DOM_DATA_ERROR							0xD203			// Links between DOM nodes in a document are corrupt.
	#define NE_XFLM_DOM_NODE_NOT_FOUND						0xD204			// The requested DOM node does not exist.
	#define NE_XFLM_DOM_INVALID_CHILD_TYPE					0xD205			// Attempting to insert a child DOM node whose type cannot be inserted as a child node.
	#define NE_XFLM_DOM_NODE_DELETED							0xD206			// DOM node being accessed has been deleted.
	#define NE_XFLM_DOM_DUPLICATE_ELEMENT					0xD207			// Node already has a child element with the given name id - this node's child nodes must all be unique.

	/****************************************************************************
	Desc:	Query Errors
	****************************************************************************/

	// IMPORTANT NOTE: If changes are made to the RCODEs, please be sure to make
	// corresponding changes in csharp/xflaim/RCODE.cs and java/xflaim/RCODE.java.
	#define NE_XFLM_Q_UNMATCHED_RPAREN						0xD301			// Query setup error: Unmatched right paren.
	#define NE_XFLM_Q_UNEXPECTED_LPAREN						0xD302			// Query setup error: Unexpected left paren.
	#define NE_XFLM_Q_UNEXPECTED_RPAREN						0xD303			// Query setup error: Unexpected right paren.
	#define NE_XFLM_Q_EXPECTING_OPERAND						0xD304			// Query setup error: Expecting an operand.
	#define NE_XFLM_Q_EXPECTING_OPERATOR					0xD305			// Query setup error: Expecting an operator.
	#define NE_XFLM_Q_UNEXPECTED_COMMA						0xD306			// Query setup error: Unexpected comma.
	#define NE_XFLM_Q_EXPECTING_LPAREN						0xD307			// Query setup error: Expecting a left paren.
	#define NE_XFLM_Q_UNEXPECTED_VALUE						0xD308			// Query setup error: Unexpected value.
	#define NE_XFLM_Q_INVALID_NUM_FUNC_ARGS				0xD309			// Query setup error: Invalid number of arguments for a function.
	#define NE_XFLM_Q_UNEXPECTED_XPATH_COMPONENT			0xD30A			// Query setup error: Unexpected XPATH componenent.
	#define NE_XFLM_Q_ILLEGAL_LBRACKET						0xD30B			// Query setup error: Illegal left bracket ([).
	#define NE_XFLM_Q_ILLEGAL_RBRACKET						0xD30C			// Query setup error: Illegal right bracket (]).
	#define NE_XFLM_Q_ILLEGAL_OPERAND						0xD30D			// Query setup error: Operand for some operator is not valid for that operator type.
	#define NE_XFLM_Q_ALREADY_OPTIMIZED						0xD30E			// Operation is illegal, cannot change certain things after query has been optimized.
	#define NE_XFLM_Q_MISMATCHED_DB							0xD30F			// Database handle passed in does not match database associated with query.
	#define NE_XFLM_Q_ILLEGAL_OPERATOR						0xD310			// Illegal operator - cannot pass this operator into the addOperator method.
	#define NE_XFLM_Q_ILLEGAL_COMPARE_RULES				0xD311			// Illegal combination of comparison rules passed to addOperator method.
	#define NE_XFLM_Q_INCOMPLETE_QUERY_EXPR				0xD312			// Query setup error: Query expression is incomplete.
	#define NE_XFLM_Q_NOT_POSITIONED							0xD313			// Query not positioned due to previous error, cannot call getNext, getPrev, or getCurrent
	#define NE_XFLM_Q_INVALID_NODE_ID_VALUE				0xD314			// Query setup error: Invalid type of value constant used for node id value comparison.
	#define NE_XFLM_Q_INVALID_META_DATA_TYPE				0xD315			// Query setup error: Invalid meta data type specified.
	#define NE_XFLM_Q_NEW_EXPR_NOT_ALLOWED					0xD316			// Query setup error: Cannot add an expression to an XPATH component after having added an expression that tests context position.
	#define NE_XFLM_Q_INVALID_CONTEXT_POS					0xD317			// Invalid context position value encountered - must be a positive number.
	#define NE_XFLM_Q_INVALID_FUNC_ARG						0xD318			// Query setup error: Parameter to user-defined functions must be a single XPATH only.
	#define NE_XFLM_Q_EXPECTING_RPAREN						0xD319			// Query setup error: Expecting right paren.
	#define NE_XFLM_Q_TOO_LATE_TO_ADD_SORT_KEYS			0xD31A			// Query setup error: Cannot add sort keys after having called getFirst, getLast, getNext, or getPrev.
	#define NE_XFLM_Q_INVALID_SORT_KEY_COMPONENT			0xD31B			// Query setup error: Invalid sort key component number specified in query.
	#define NE_XFLM_Q_DUPLICATE_SORT_KEY_COMPONENT		0xD31C			// Query setup error: Duplicate sort key component number specified in query.
	#define NE_XFLM_Q_MISSING_SORT_KEY_COMPONENT			0xD31D			// Query setup error: Missing sort key component number in sort keys that were specified for query.
	#define NE_XFLM_Q_NO_SORT_KEY_COMPONENTS_SPECIFIED	0xD31E			// Query setup error: addSortKeys was called, but no sort key components were specified.
	#define NE_XFLM_Q_SORT_KEY_CONTEXT_MUST_BE_ELEMENT	0xD31F			// Query setup error: A sort key context cannot be an XML attribute.
	#define NE_XFLM_Q_INVALID_ELEMENT_NUM_IN_SORT_KEYS 0xD320			// Query setup error: The XML element number specified for a sort key in a query is invalid - no element definition in the dictionary.
	#define NE_XFLM_Q_INVALID_ATTR_NUM_IN_SORT_KEYS 	0xD321			// Query setup error: The XML attribute number specified for a sort key in a query is invalid - no attribute definition in the dictionary.
	#define NE_XFLM_Q_NON_POSITIONABLE_QUERY				0xD322			// Attempt is being made to position in a query that is not positionable.
	#define NE_XFLM_Q_INVALID_POSITION						0xD323			// Attempt is being made to position to an invalid position in the result set.

	/****************************************************************************
	Desc:	NICI / Encryption Errors
	****************************************************************************/

	// IMPORTANT NOTE: If changes are made to the RCODEs, please be sure to make
	// corresponding changes in csharp/xflaim/RCODE.cs and java/xflaim/RCODE.java.
	#define NE_XFLM_NICI_CONTEXT								0xD401			// Error occurred while creating NICI context for encryption/decryption.
	#define NE_XFLM_NICI_ATTRIBUTE_VALUE					0xD402			// Error occurred while accessing an attribute on a NICI encryption key.
	#define NE_XFLM_NICI_BAD_ATTRIBUTE						0xD403			// Value retrieved from an attribute on a NICI encryption key was bad.
	#define NE_XFLM_NICI_WRAPKEY_FAILED						0xD404			// Error occurred while wrapping a NICI encryption key in another NICI encryption key.
	#define NE_XFLM_NICI_UNWRAPKEY_FAILED					0xD405			// Error occurred while unwrapping a NICI encryption key that is wrapped in another NICI encryption key.
	#define NE_XFLM_NICI_INVALID_ALGORITHM					0xD406			// Attempt to use invalid NICI encryption algorithm. 
	#define NE_XFLM_NICI_GENKEY_FAILED						0xD407			// Error occurred while attempting to generate a NICI encryption key.
	#define NE_XFLM_NICI_BAD_RANDOM							0xD408			// Error occurred while generating random data using NICI.
	#define NE_XFLM_PBE_ENCRYPT_FAILED						0xD409			// Error occurred while attempting to wrap a NICI encryption key in a password.
	#define NE_XFLM_PBE_DECRYPT_FAILED						0xD40A			// Error occurred while attempting to unwrap a NICI encryption key that was previously wrapped in a password.
	#define NE_XFLM_DIGEST_INIT_FAILED						0xD40B			// Error occurred while attempting to initialize the NICI digest functionality.
	#define NE_XFLM_DIGEST_FAILED								0xD40C			// Error occurred while attempting to create a NICI digest. 
	#define NE_XFLM_INJECT_KEY_FAILED						0xD40D			// Error occurred while attempting to inject an encryption key into NICI. 
	#define NE_XFLM_NICI_FIND_INIT							0xD40E			// Error occurred while attempting to initialize NICI to find information on a NICI encryption key.
	#define NE_XFLM_NICI_FIND_OBJECT							0xD40F			// Error occurred while attempting to find information on a NICI encryption key.
	#define NE_XFLM_NICI_KEY_NOT_FOUND						0xD410			// Could not find the NICI encryption key or information on the NICI encryption key.
	#define NE_XFLM_NICI_ENC_INIT_FAILED					0xD411			// Error occurred while initializing NICI to encrypt data.
	#define NE_XFLM_NICI_ENCRYPT_FAILED						0xD412			// Error occurred while encrypting data.
	#define NE_XFLM_NICI_DECRYPT_INIT_FAILED				0xD413			// Error occurred while initializing NICI to decrypt data.
	#define NE_XFLM_NICI_DECRYPT_FAILED						0xD414			// Error occurred while decrypting data.
	#define NE_XFLM_NICI_WRAPKEY_NOT_FOUND					0xD415			// Could not find the NICI encryption key used to wrap another NICI encryption key.
	#define NE_XFLM_NOT_EXPECTING_PASSWORD					0xD416			// Password supplied when none was expected.
	#define NE_XFLM_EXPECTING_PASSWORD						0xD417			// No password supplied when one was required.
	#define NE_XFLM_EXTRACT_KEY_FAILED						0xD418			// Error occurred while attempting to extract a NICI encryption key.
	#define NE_XFLM_NICI_INIT_FAILED							0xD419			// Error occurred while initializing NICI.
	#define NE_XFLM_BAD_ENCKEY_SIZE							0xD41A			// Bad encryption key size found in roll-forward log packet.
	#define NE_XFLM_ENCRYPTION_UNAVAILABLE					0xD41B			// Attempt was made to encrypt data when NICI is unavailable.

	/****************************************************************************
	Dictionary Document Definitions - below are comments that document valid 
	dictionary objects and their structure.
	****************************************************************************/

	/*
	Element Definition
	Desc: The XML syntax given below is used to define an element in the
			dictionary collection.

	<xflaim:element
		xmlns:xflaim="http://www.novell.com/XMLDatabase/Schema"
		xflaim:name="name of the element"
		xflaim:DictNumber="element number"
		xflaim:targetNameSpace="namespace of the element"
		xflaim:type="string, integer, binary, or nodata"
		xflaim:State="checking, unused, purge, or active"
		xflaim:UniqueSubElements="yes, true, on, 1, enable or no, false, off, 0, disable"
	/>

	Notes:

	  1)	If the xflaim:type attribute is missing, any type of data may be stored on the attribute.
	  2)	If the xflaim:State attribute is missing, the attribute's state is, by default, active.

	*/

	/*
	Attribute Definition
	Desc: The XML syntax given below is used to define an attribute in the
			dictionary collection.

	<xflaim:attribute
		xmlns:xflaim="http://www.novell.com/XMLDatabase/Schema"
		xflaim:name="name of the attribute"
		xflaim:DictNumber="attribute number"
		xflaim:targetNameSpace="namespace of the attribute"
		xflaim:type="string, integer, or binary"
		xflaim:State="checking, unused, purge, or active"
	/>

	Notes:

	  1)	If the xflaim:type attribute is missing, any type of data may be stored on the attribute.
	  2)	If the xflaim:State attribute is missing, the attribute's state is, by default, active.

	*/

	/*
	Collection Definition
	Desc: The XML syntax given below is used to define a collection in the
			dictionary collection.

	<xflaim:Collection
		xmlns:xflaim="http://www.novell.com/XMLDatabase/Schema"
		xflaim:name="name of the collection"
		xflaim:DictNumber="collection number"
	/>
	*/

	/*
	Prefix Definition
	Desc: The XML syntax given below is used to define a namespace prefix in the
			dictionary collection.

	<xflaim:Prefix
		xmlns:xflaim="http://www.novell.com/XMLDatabase/Schema"
		xflaim:name="name of the prefix"
		xflaim:DictNumber="prefix number"
	/>
	*/

	/*
	Index Definition
	Desc: The XML syntax given below is used to define an index in the
			dictionary collection.

	<xflaim:Index
		xmlns:xflaim="http://www.novell.com/XMLDatabase/Schema"
		xflaim:name="name of the index"
		xflaim:DictNumber="index number"
		xflaim:CollectionName="name of collection to be indexed"
		xflaim:CollectionNumber="number of collection to be indexed"
		xflaim:Language="collation language of the index"
		xflaim:IndexOptions="abspos">

		<xflaim:ElementComponent
			xflaim:name="name of element in element path to be indexed"
			xflaim:targetNameSpace="namespace of element - if element name given"
			xflaim:DictNumber="number of element in element path to be indexed"
			xflaim:KeyComponent="order of this component in the key"
			xflaim:DataComponent="order of this component in data part"
			xflaim:IndexOn="value, substring, eachword, metaphone, or presence"
			xflaim:type="string, integer, pointer, or binary"
			xflaim:Required="yes, true, on, 1, enable or no, false, off, 0, disable"
			xflaim:Limit="limit in bytes or characters for string or binary type"
			xflaim:CompareRules="comparison rules"
		> ...

		NOTE: The IndexOn, Required, Limit, and CompareRules attributes can only be set for
		key components - i.e., when the KeyComponent attribute is also specified.

		xflaim:ElementComponent elements may have one or more xflaim:ElementComponent
		or xflaim:AttributeComponent sub-elements.

		<xflaim:AttributeComponent
			xflaim:name="name of attribute in element path to be indexed"
			xflaim:targetNameSpace="namespace of attribute - if attribute name given"
			xflaim:DictNumber="number of attribute in element path to be indexed"
			xflaim:KeyComponent="order of this component in the key"
			xflaim:DataComponent="order of this component in data part"
			xflaim:IndexOn="value, substring, eachword, metaphone, or presence"
			xflaim:type="string, integer, or binary"
			xflaim:Required="yes, true, on, 1, enable or no, false, off, 0, disable"
			xflaim:Limit="limit in bytes or characters for string or binary type"
			xflaim:CompareRules="comparison rules"
		/> ...

		NOTE: The IndexOn, Required, Limit, and CompareRules attributes can only be set for
		key components - i.e., when the KeyComponent attribute is also specified.

		xflaim:AttributeComponent elements must be subordinate to a xflaim:ElementComponent
		element.  They may not have any elements subordinate to them.

	</xflaim:Index>

	Notes:

		1)	It is not valid to specify both an xflaim:CollectionName and an xflaim:CollectionNumber
			unless they refer to the same collection.  Only one of the two is needed.  If both
			are omitted, the collection that will be indexed is the default data collection.
		2)	Valid values for the IndexOptions attribute are any combination of the following:
			a)	abspos		- enables storing of absolute positioning information
		3)	For the xflaim:ElementComponent and xflaim:AttributeComponent, it is not
			valid to have both an xflaim:DictNumber and an xflaim:name attribute, unless they both specify the
			same element or attribute.  One or the other must be specified.
		4)	If the xflaim:type attribute is specified on the xflaim:ElementComponent or xflaim:AttributeComponent
			element, it indicates what type the component is to be coerced to for indexing purposes.
			This is only relevant if the xflaim:IndexOn attribute is set to "value".  For example, if a
			type of integer is specified, then the value will be coerced to an integer before putting it
			into the index.  If the value were a string of "123" it would be coerced to an integer value of
			123 for indexing purposes. If the xflaim:type attribute is omitted, then the element or attribute
			specified by the component must have a type specified in the element or attribute definition,
			and that is the type that will be used.
		5)	If the xflaim:Required attribute is missing from the xflaim:ElementComponent or xflaim:AttributeComponent
			element, or does not have a value of "yes", "on", "1", or "true", the index component is assumed to
			be optional.
		6)	The xflaim:CompareRules attribute specifies special comparison rules for the index component if the
			element or attribute is of type xflaim:string.  Rules may be any combination of the following key words:

			a) caseinsensitive - don't compare case
			b) whitespaceasspace - treat whitespace as space (must be applied before minspaces,
				ignoreleadingspaces, ignoretrailingspaces, or nospaces)
			c) minspaces - compress out extra spaces
			d)	ignoreleadingspaces - remove leading spaces
			e)	ignoretrailingspaces - remove trailing spaces
			f) nospaces - remove all whitespace
			g) nounderscore - change all underscores to spaces (must be applied before minspaces,
				ignoreleadingspaces, ignoretrailingspaces, or nospaces)
			h) nodashes - remove all dashes.
			i) sortdescending - sort in descending order.
			j) sortmissinghigh - sort missing values high.
			
			Note that if the xflaim:CompareRules attribute is omitted, the default comparison rule for xflaim:string
			values is case sensitive.  Note also that the xflaim:CompareRules attribute is ignored if the
			xflaim:IndexOn attribute is set to "presence".  In that case, comparison rules are irrelevant.
	*/


	/*
	EncDef Definition
	Desc: The XML syntax given below is used to define an encryption definition in the
			dictionary collection.

	<xflaim:EncDef 
		xmlns:xs="http://www.w3.org/2001/XMLSchema"
		xmlns:xflaim="http://www.novell.com/XMLDatabase/Schema"
		xflaim:name="Encryption description text"
		xflaim:DictNumber="Encryption definition number"
		xflaim:type="aes" | "des3"
		xflaim:keySize="256" | "192" | "128" | "168"
	/>

	Notes:

	1) Only AES and DES3 (Triple DES) encryption algorithms are supported

	2) AES supports three key lengths: 256, 192 and 128 bits in length.

	3) DES3 keys are 168 bits in length.

	4. The xflaim:keySize attribute is optional.  If it is not specified, the maximum 
		key size allowed for the chosen algorithm will be selected.  For AES, the key size
		chosen will depend on what is supported by the NICI installation.

	*/

	#ifdef FLM_PACK_STRUCTS
		#pragma pack(pop)
	#endif

#endif // XFLAIM_H
