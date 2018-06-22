//------------------------------------------------------------------------------
// Desc:	RCODEs
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

package xflaim;

/**
 * Provides enums for some of the error codes that XFlaim might return in an
 * {@link xflaim.XFlaimException XFlaimException}.
 */

public final class RCODE
{
	public static final int NE_XFLM_OK												= 0;
		
	public static final int NE_XFLM_NOT_IMPLEMENTED								= 0xC05F;
	public static final int NE_XFLM_MEM												= 0xC037;
	public static final int NE_XFLM_INVALID_PARM									= 0xC08B;
	public static final int NE_XFLM_NOT_FOUND										= 0xC006;
	public static final int NE_XFLM_EXISTS											= 0xC004;
	public static final int NE_XFLM_FAILURE										= 0xC005;
	public static final int NE_XFLM_BOF_HIT										= 0xC001;
	public static final int NE_XFLM_EOF_HIT										= 0xC002;
	public static final int NE_XFLM_CONV_DEST_OVERFLOW							= 0xC01C;
	public static final int NE_XFLM_CONV_ILLEGAL									= 0xC01D;
	public static final int NE_XFLM_CONV_NUM_OVERFLOW							= 0xC020;
	public static final int NE_XFLM_SYNTAX											= 0xC045;
	public static final int NE_XFLM_ILLEGAL_OP									= 0xC026;
	public static final int NE_XFLM_BAD_SEN										= 0xC503;
	public static final int NE_XFLM_COULD_NOT_START_THREAD					= 0xC504;
	public static final int NE_XFLM_BAD_BASE64_ENCODING						= 0xC505;
	public static final int NE_XFLM_STREAM_EXISTS								= 0xC506;
	public static final int NE_XFLM_MULTIPLE_MATCHES							= 0xC507;
	public static final int NE_XFLM_NOT_UNIQUE									= 0xC03E;
	public static final int NE_XFLM_BTREE_ERROR									= 0xC012;
	public static final int NE_XFLM_BTREE_KEY_SIZE								= 0xC508;
	public static final int NE_XFLM_BTREE_FULL									= 0xC013;
	public static final int NE_XFLM_BTREE_BAD_STATE								= 0xC509;
	public static final int NE_XFLM_COULD_NOT_CREATE_MUTEX					= 0xC50A;
	public static final int NE_XFLM_DATA_ERROR									= 0xC022;
	public static final int NE_XFLM_IO_PATH_NOT_FOUND							= 0xC209;
	public static final int NE_XFLM_IO_END_OF_FILE								= 0xC205;
	public static final int NE_XFLM_IO_NO_MORE_FILES							= 0xC20C;
	public static final int NE_XFLM_COULD_NOT_CREATE_SEMAPHORE				= 0xC500;
	public static final int NE_XFLM_BAD_UTF8										= 0xC501;
	public static final int NE_XFLM_ERROR_WAITING_ON_SEMPAHORE				= 0xC502;
	public static final int NE_XFLM_BAD_PLATFORM_FORMAT						= 0xC50B;
		
	/****************************************************************************
	Desc:		General XFLAIM errors
	****************************************************************************/

	public static final int NE_XFLM_USER_ABORT									= 0xD100; // User or application aborted (canceled) operation.
	public static final int NE_XFLM_BAD_PREFIX									= 0xD101; // Invalid XLM namespace prefix specified.  Either a prefix name or number that was specified was not defined.
	public static final int NE_XFLM_ATTRIBUTE_PURGED							= 0xD102; // XML attribute cannot be used - it is being deleted from the database.
	public static final int NE_XFLM_BAD_COLLECTION								= 0xD103; // Invalid collection number specified.  Collection is not defined.
	public static final int NE_XFLM_DATABASE_LOCK_REQ_TIMEOUT				= 0xD104; // Request to lock the database timed out.
	public static final int NE_XFLM_ILLEGAL_DATA_COMPONENT					= 0xD105; // Cannot use ELM_ROOT_TAG as a data component in an index.
	public static final int NE_XFLM_MUST_INDEX_ON_PRESENCE					= 0xD106; // When using ELM_ROOT_TAG in an index component, must specify PRESENCE indexing only.
	public static final int NE_XFLM_BAD_IX											= 0xD107; // Invalid index number specified.  Index is not defined.
	public static final int NE_XFLM_BACKUP_ACTIVE								= 0xD108; // Operation could not be performed because a backup is currently in progress.
	public static final int NE_XFLM_SERIAL_NUM_MISMATCH						= 0xD109; // Serial number on backup file does not match the serial number that is expected.
	public static final int NE_XFLM_BAD_RFL_DB_SERIAL_NUM						= 0xD10A; // Bad database serial number in roll-forward log file header.
	public static final int NE_XFLM_BAD_RFL_FILE_NUMBER						= 0xD10B; // Bad roll-forward log file number in roll-forward log file header.
	public static final int NE_XFLM_CANNOT_DEL_ELEMENT							= 0xD10C; // Cannot delete an XML element definition in the dictionary because it is in use.
	public static final int NE_XFLM_CANNOT_MOD_DATA_TYPE						= 0xD10D; // Cannot modify the data type for an XML element or attribute definition in the dictionary.
	public static final int NE_XFLM_CANNOT_INDEX_DATA_TYPE					= 0xD10E; // Data type of XML element or attribute is not one that can be indexed.
	public static final int NE_XFLM_BAD_ELEMENT_NUM								= 0xD10F; // Bad element number specified - element not defined in dictionary.
	public static final int NE_XFLM_BAD_ATTRIBUTE_NUM							= 0xD110; // Bad attribute number specified - attribute not defined in dictionary.
	public static final int NE_XFLM_BAD_ENCDEF_NUM								= 0xD111; // Bad encryption number specified - encryption definition not defined in dictionary.
	public static final int NE_XFLM_INVALID_FILE_SEQUENCE						= 0xD112; // Incremental backup file number provided during a restore is invalid.
	public static final int NE_XFLM_DUPLICATE_ELEMENT_NUM						= 0xD113; // Element number specified in element definition is already in use.
	public static final int NE_XFLM_ILLEGAL_TRANS_TYPE							= 0xD114; // Illegal transaction type specified for transaction begin operation.
	public static final int NE_XFLM_UNSUPPORTED_VERSION						= 0xD115; // Version of database found in database header is not supported.
	public static final int NE_XFLM_ILLEGAL_TRANS_OP							= 0xD116; // Illegal operation for transaction type.
	public static final int NE_XFLM_INCOMPLETE_LOG								= 0xD117; // Incomplete rollback log.
	public static final int NE_XFLM_ILLEGAL_INDEX_DEF							= 0xD118; // Index definition document is illegal - does not conform to the expected form of an index definition document.
	public static final int NE_XFLM_ILLEGAL_INDEX_ON							= 0xD119; // The "IndexOn" attribute of an index definition has an illegal value.
	public static final int NE_XFLM_ILLEGAL_STATE_CHANGE						= 0xD11A; // Attempted an illegal state change on an element or attribute definition.
	public static final int NE_XFLM_BAD_RFL_SERIAL_NUM							= 0xD11B; // Serial number in roll-forward log file header does not match expected serial number.
	public static final int NE_XFLM_NEWER_FLAIM									= 0xD11C; // Running old code on a newer version of database.  Newer code must be used.
	public static final int NE_XFLM_CANNOT_MOD_ELEMENT_STATE					= 0xD11D; // Attempted to change state of a predefined element definition.
	public static final int NE_XFLM_CANNOT_MOD_ATTRIBUTE_STATE				= 0xD11E; // Attempted to change state of a predefined attribute definition.
	public static final int NE_XFLM_NO_MORE_ELEMENT_NUMS						= 0xD11F; // The highest element number has already been used, cannot create more element definitions.
	public static final int NE_XFLM_NO_TRANS_ACTIVE								= 0xD120; // Operation must be performed inside a database transaction.
	public static final int NE_XFLM_NOT_FLAIM										= 0xD121; // The file specified is not a FLAIM database.
	public static final int NE_XFLM_OLD_VIEW										= 0xD122; // Unable to maintain read transaction's view of the database.
	public static final int NE_XFLM_SHARED_LOCK									= 0xD123; // Attempted to perform an operation on the database that requires exclusive access, but cannot because there is a shared lock.
	public static final int NE_XFLM_TRANS_ACTIVE									= 0xD124; // Operation cannot be performed while a transaction is active.
	public static final int NE_XFLM_RFL_TRANS_GAP								= 0xD125; // A gap was found in the transaction sequence in the roll-forward log.
	public static final int NE_XFLM_BAD_COLLATED_KEY							= 0xD126; // Something in collated key is bad.
	public static final int NE_XFLM_MUST_DELETE_INDEXES						= 0xD127; // Attempting to delete a collection that has indexes defined for it.  Associated indexes must be deleted before the collection can be deleted.
	public static final int NE_XFLM_RFL_INCOMPLETE								= 0xD128; // Roll-forward log file is incomplete.
	public static final int NE_XFLM_CANNOT_RESTORE_RFL_FILES					= 0xD129; // Cannot restore roll-forward log files - not using multiple roll-forward log files.
	public static final int NE_XFLM_INCONSISTENT_BACKUP						= 0xD12A; // A problem (corruption), etc. was detected in a backup set.
	public static final int NE_XFLM_BLOCK_CRC										= 0xD12B; // CRC for database block was invalid.  May indicate problems in reading from or writing to disk.
	public static final int NE_XFLM_ABORT_TRANS									= 0xD12C; // Attempted operation after a critical error - transaction should be aborted.
	public static final int NE_XFLM_NOT_RFL										= 0xD12D; // File was not a roll-forward log file as expected.
	public static final int NE_XFLM_BAD_RFL_PACKET								= 0xD12E; // Roll-forward log file packet was bad.
	public static final int NE_XFLM_DATA_PATH_MISMATCH							= 0xD12F; // Bad data path specified to open database.  Does not match data path specified for prior opens of the database.
	public static final int NE_XFLM_MUST_CLOSE_DATABASE						= 0xD130; // Database must be closed due to a critical error.
	public static final int NE_XFLM_INVALID_ENCKEY_CRC							= 0xD131; // Encryption key CRC could not be verified.
	public static final int NE_XFLM_HDR_CRC										= 0xD132; // Database header has a bad CRC.
	public static final int NE_XFLM_NO_NAME_TABLE								= 0xD133; // No name table was set up for the database.
	public static final int NE_XFLM_UNALLOWED_UPGRADE							= 0xD134; // Cannot upgrade database from one version to another.
	public static final int NE_XFLM_DUPLICATE_ATTRIBUTE_NUM					= 0xD135; // Attribute number specified in attribute definition is already in use.
	public static final int NE_XFLM_DUPLICATE_INDEX_NUM						= 0xD136; // Index number specified in index definition is already in use.
	public static final int NE_XFLM_DUPLICATE_COLLECTION_NUM					= 0xD137; // Collection number specified in collection definition is already in use.
	public static final int NE_XFLM_DUPLICATE_ELEMENT_NAME					= 0xD138; // Element name+namespace specified in element definition is already in use.
	public static final int NE_XFLM_DUPLICATE_ATTRIBUTE_NAME					= 0xD139; // Attribute name+namespace specified in attribute definition is already in use.
	public static final int NE_XFLM_DUPLICATE_INDEX_NAME						= 0xD13A; // Index name specified in index definition is already in use.
	public static final int NE_XFLM_DUPLICATE_COLLECTION_NAME				= 0xD13B; // Collection name specified in collection definition is already in use.
	public static final int NE_XFLM_ELEMENT_PURGED								= 0xD13C; // XML element cannot be used - it is deleted from the database.
	public static final int NE_XFLM_TOO_MANY_OPEN_DATABASES					= 0xD13D; // Too many open databases, cannot open another one.
	public static final int NE_XFLM_DATABASE_OPEN								= 0xD13E; // Operation cannot be performed because the database is currently open.
	public static final int NE_XFLM_CACHE_ERROR									= 0xD13F; // Cached database block has been compromised while in cache.
	public static final int NE_XFLM_DB_FULL										= 0xD140; // Database is full, cannot create more blocks.
	public static final int NE_XFLM_QUERY_SYNTAX									= 0xD141; // Query expression had improper syntax.
	public static final int NE_XFLM_INDEX_OFFLINE								= 0xD142; // Index is offline, cannot be used in a query.
	public static final int NE_XFLM_RFL_DISK_FULL								= 0xD143; // Disk which contains roll-forward log is full.
	public static final int NE_XFLM_MUST_WAIT_CHECKPOINT						= 0xD144; // Must wait for a checkpoint before starting transaction - due to disk problems - usually in disk containing roll-forward log files.
	public static final int NE_XFLM_MISSING_ENC_ALGORITHM						= 0xD145; // Encryption definition is missing an encryption algorithm.
	public static final int NE_XFLM_INVALID_ENC_ALGORITHM						= 0xD146; // Invalid encryption algorithm specified in encryption definition.
	public static final int NE_XFLM_INVALID_ENC_KEY_SIZE						= 0xD147; // Invalid key size specified in encryption definition.
	public static final int NE_XFLM_ILLEGAL_DATA_TYPE							= 0xD148; // Data type specified for XML element or attribute definition is illegal.
	public static final int NE_XFLM_ILLEGAL_STATE								= 0xD149; // State specified for index definition or XML element or attribute definition is illegal.
	public static final int NE_XFLM_ILLEGAL_ELEMENT_NAME						= 0xD14A; // XML element name specified in element definition is illegal.
	public static final int NE_XFLM_ILLEGAL_ATTRIBUTE_NAME					= 0xD14B; // XML attribute name specified in attribute definition is illegal.
	public static final int NE_XFLM_ILLEGAL_COLLECTION_NAME					= 0xD14C; // Collection name specified in collection definition is illegal.
	public static final int NE_XFLM_ILLEGAL_INDEX_NAME							= 0xD14D; // Index name specified is illegal
	public static final int NE_XFLM_ILLEGAL_ELEMENT_NUMBER					= 0xD14E; // Element number specified in element definition or index definition is illegal.
	public static final int NE_XFLM_ILLEGAL_ATTRIBUTE_NUMBER					= 0xD14F; // Attribute number specified in attribute definition or index definition is illegal.
	public static final int NE_XFLM_ILLEGAL_COLLECTION_NUMBER				= 0xD150; // Collection number specified in collection definition or index definition is illegal.
	public static final int NE_XFLM_ILLEGAL_INDEX_NUMBER						= 0xD151; // Index number specified in index definition is illegal.
	public static final int NE_XFLM_ILLEGAL_ENCDEF_NUMBER						= 0xD152; // Encryption definition number specified in encryption definition is illegal.
	public static final int NE_XFLM_COLLECTION_NAME_MISMATCH					= 0xD153; // Collection name and number specified in index definition do not correspond to each other.
	public static final int NE_XFLM_ELEMENT_NAME_MISMATCH						= 0xD154; // Element name+namespace and number specified in index definition do not correspond to each other.
	public static final int NE_XFLM_ATTRIBUTE_NAME_MISMATCH					= 0xD155; // Attribute name+namespace and number specified in index definition do not correspond to each other.
	public static final int NE_XFLM_INVALID_COMPARE_RULE						= 0xD156; // Invalid comparison rule specified in index definition.
	public static final int NE_XFLM_DUPLICATE_KEY_COMPONENT					= 0xD157; // Duplicate key component number specified in index definition.
	public static final int NE_XFLM_DUPLICATE_DATA_COMPONENT					= 0xD158; // Duplicate data component number specified in index definition.
	public static final int NE_XFLM_MISSING_KEY_COMPONENT						= 0xD159; // Index definition is missing a key component.
	public static final int NE_XFLM_MISSING_DATA_COMPONENT					= 0xD15A; // Index definition is missing a data component.
	public static final int NE_XFLM_INVALID_INDEX_OPTION						= 0xD15B; // Invalid index option specified on index definition.
	public static final int NE_XFLM_NO_MORE_ATTRIBUTE_NUMS					= 0xD15C; // The highest attribute number has already been used, cannot create more.
	public static final int NE_XFLM_MISSING_ELEMENT_NAME						= 0xD15D; // Missing element name in XML element definition.
	public static final int NE_XFLM_MISSING_ATTRIBUTE_NAME					= 0xD15E; // Missing attribute name in XML attribute definition.
	public static final int NE_XFLM_MISSING_ELEMENT_NUMBER					= 0xD15F; // Missing element number in XML element definition.
	public static final int NE_XFLM_MISSING_ATTRIBUTE_NUMBER					= 0xD160; // Missing attribute number from XML attribute definition.
	public static final int NE_XFLM_MISSING_INDEX_NAME							= 0xD161; // Missing index name in index definition.
	public static final int NE_XFLM_MISSING_INDEX_NUMBER						= 0xD162; // Missing index number in index definition.
	public static final int NE_XFLM_MISSING_COLLECTION_NAME					= 0xD163; // Missing collection name in collection definition.
	public static final int NE_XFLM_MISSING_COLLECTION_NUMBER				= 0xD164; // Missing collection number in collection definition.
	public static final int NE_XFLM_MISSING_ENCDEF_NAME						= 0xD165; // Missing encryption definition name in encryption definition.
	public static final int NE_XFLM_MISSING_ENCDEF_NUMBER						= 0xD166; // Missing encryption definition number in encryption definition.
	public static final int NE_XFLM_NO_MORE_INDEX_NUMS							= 0xD167; // The highest index number has already been used, cannot create more.
	public static final int NE_XFLM_NO_MORE_COLLECTION_NUMS					= 0xD168; // The highest collection number has already been used, cannot create more.
	public static final int NE_XFLM_CANNOT_DEL_ATTRIBUTE						= 0xD169; // Cannot delete an XML attribute definition because it is in use.
	public static final int NE_XFLM_TOO_MANY_PENDING_NODES					= 0xD16A; // Too many documents in the pending document list.
	public static final int NE_XFLM_BAD_USE_OF_ELM_ROOT_TAG					= 0xD16B; // ELM_ROOT_TAG, if used, must be the sole root component of an index definition.
	public static final int NE_XFLM_DUP_SIBLING_IX_COMPONENTS				= 0xD16C; // Sibling components in an index definition cannot have the same XML element or attribute number.
	public static final int NE_XFLM_RFL_FILE_NOT_FOUND							= 0xD16D; // Could not open a roll-forward log file - was not found in the roll-forward log directory.
	public static final int NE_XFLM_ILLEGAL_KEY_COMPONENT_NUM				= 0xD16E; // Key component of zero in index definition is not allowed.
	public static final int NE_XFLM_ILLEGAL_DATA_COMPONENT_NUM				= 0xD16F; // Data component of zero in index definition is not allowed.
	public static final int NE_XFLM_ILLEGAL_PREFIX_NUMBER						= 0xD170; // Prefix number specified in prefix definition is illegal.
	public static final int NE_XFLM_MISSING_PREFIX_NAME						= 0xD171; // Missing prefix name in prefix definition.
	public static final int NE_XFLM_MISSING_PREFIX_NUMBER						= 0xD172; // Missing prefix number in prefix definition.
	public static final int NE_XFLM_UNDEFINED_ELEMENT_NAME					= 0xD173; // XML element name+namespace that was specified in index definition or XML document is not defined in dictionary.
	public static final int NE_XFLM_UNDEFINED_ATTRIBUTE_NAME					= 0xD174; // XML attribute name+namespace that was specified in index definition or XML document is not defined in dictionary.
	public static final int NE_XFLM_DUPLICATE_PREFIX_NAME						= 0xD175; // Prefix name specified in prefix definition is already in use.
	public static final int NE_XFLM_NAMESPACE_NOT_ALLOWED						= 0xD176; // Cannot define a namespace for XML attributes whose name begins with "xmlns:" or that is equal to "xmlns"
	public static final int NE_XFLM_INVALID_NAMESPACE_DECL					= 0xD177; // Name for namespace declaration attribute must be "xmlns" or begin with "xmlns:"
	public static final int NE_XFLM_ILLEGAL_NAMESPACE_DECL_DATATYPE		= 0xD178; // Data type for XML attributes that are namespace declarations must be text.
	public static final int NE_XFLM_NO_MORE_PREFIX_NUMS						= 0xD179; // The highest prefix number has already been used, cannot create more.
	public static final int NE_XFLM_NO_MORE_ENCDEF_NUMS						= 0xD17A; // The highest encryption definition number has already been used, cannot create more.
	public static final int NE_XFLM_COLLECTION_OFFLINE							= 0xD17B; // Collection is encrypted, cannot be accessed while in operating in limited mode.
	public static final int NE_XFLM_DELETE_NOT_ALLOWED							= 0xD17C; // Item cannot be deleted.
	public static final int NE_XFLM_RESET_NEEDED									= 0xD17D; // Used during check operations to indicate we need to reset the view.  NOTE: This is an internal error code and should not be documented.
	public static final int NE_XFLM_ILLEGAL_REQUIRED_VALUE					= 0xD17E; // An illegal value was specified for the "Required" attribute in an index definition.
	public static final int NE_XFLM_ILLEGAL_INDEX_COMPONENT					= 0xD17F; // A leaf index component in an index definition was not marked as a data component or key component.
	public static final int NE_XFLM_ILLEGAL_UNIQUE_SUB_ELEMENT_VALUE		= 0xD180; // Illegal value for the "UniqueSubElements" attribute in an element definition.
	public static final int NE_XFLM_DATA_TYPE_MUST_BE_NO_DATA				= 0xD181; // Data type for an element definition with UniqueSubElements="yes" must be nodata.
	public static final int NE_XFLM_CANNOT_SET_REQUIRED						= 0xD182; // Cannot set the "Required" attribute on a non-key index component in index definition.
	public static final int NE_XFLM_CANNOT_SET_LIMIT							= 0xD183; // Cannot set the "Limit" attribute on a non-key index component in index definition.
	public static final int NE_XFLM_CANNOT_SET_INDEX_ON						= 0xD184; // Cannot set the "IndexOn" attribute on a non-key index component in index definition.
	public static final int NE_XFLM_CANNOT_SET_COMPARE_RULES					= 0xD185; // Cannot set the "CompareRules" on a non-key index component in index definition.
	public static final int NE_XFLM_INPUT_PENDING								= 0xD186; // Attempt to set a value while an input stream is still open.
	public static final int NE_XFLM_INVALID_NODE_TYPE							= 0xD187; // Bad node type
	public static final int NE_XFLM_INVALID_CHILD_ELM_NODE_ID				= 0xD188; // Attempt to insert a unique child element that has a lower node ID than the parent element
	public static final int NE_XFLM_RFL_END										= 0xD189; // Hit the end of the RFL
	public static final int NE_XFLM_ILLEGAL_FLAG									= 0xD18A; // Illegal flag passed to getChildElement method.  Must be zero for elements that can have non-unique child elements.
	public static final int NE_XFLM_TIMEOUT										= 0xD18B; // Operation timed out.
	public static final int NE_XFLM_CONV_BAD_DIGIT								= 0xD18C; // Non-numeric digit found in text to numeric conversion.
	public static final int NE_XFLM_CONV_NULL_SRC								= 0xD18D; // Data source cannot be NULL when doing data conversion.
	public static final int NE_XFLM_CONV_NUM_UNDERFLOW							= 0xD18E; // Numeric underflow (< lower bound) converting to numeric type.
	public static final int NE_XFLM_UNSUPPORTED_FEATURE						= 0xD18F; // Attempting to use a feature for which full support has been disabled.
	public static final int NE_XFLM_FILE_EXISTS									= 0xD190; // Attempt to create a database, but the file already exists.
	public static final int NE_XFLM_BUFFER_OVERFLOW								= 0xD191; // Buffer overflow.
	public static final int NE_XFLM_INVALID_XML									= 0xD192; // Invalid XML encountered while parsing document.
	public static final int NE_XFLM_BAD_DATA_TYPE								= 0xD193; // Attempt to set/get data on an XML element or attribute using a data type that is incompatible with the data type specified in the dictionary.
	public static final int NE_XFLM_READ_ONLY										= 0xD194; // Item is read-only and cannot be updated.
	public static final int NE_XFLM_KEY_OVERFLOW									= 0xD195; // Generated index key too large.
	public static final int NE_XFLM_UNEXPECTED_END_OF_INPUT					= 0xD196; // Encountered unexpected end of input when parsing XPATH expression.
		
	/****************************************************************************
	Desc:		DOM Errors
	****************************************************************************/

	public static final int NE_XFLM_DOM_HIERARCHY_REQUEST_ERR				= 0xD201; // Attempt to insert a DOM node somewhere it doesn't belong.
	public static final int NE_XFLM_DOM_WRONG_DOCUMENT_ERR					= 0xD202; // A DOM node is being used in a different document than the one that created it.
	public static final int NE_XFLM_DOM_DATA_ERROR								= 0xD203; // Links between DOM nodes in a document are corrupt.
	public static final int NE_XFLM_DOM_NODE_NOT_FOUND							= 0xD204; // The requested DOM node does not exist.
	public static final int NE_XFLM_DOM_INVALID_CHILD_TYPE					= 0xD205; // Attempting to insert a child DOM node whose type cannot be inserted as a child node.
	public static final int NE_XFLM_DOM_NODE_DELETED							= 0xD206; // DOM node being accessed has been deleted.
	public static final int NE_XFLM_DOM_DUPLICATE_ELEMENT						= 0xD207; // Node already has a child element with the given name id - this node's child nodes must all be unique.

	/****************************************************************************
	Desc:	Query Errors
	****************************************************************************/

	public static final int NE_XFLM_Q_UNMATCHED_RPAREN							= 0xD301; // Query setup error: Unmatched right paren.
	public static final int NE_XFLM_Q_UNEXPECTED_LPAREN						= 0xD302; // Query setup error: Unexpected left paren.
	public static final int NE_XFLM_Q_UNEXPECTED_RPAREN						= 0xD303; // Query setup error: Unexpected right paren.
	public static final int NE_XFLM_Q_EXPECTING_OPERAND						= 0xD304; // Query setup error: Expecting an operand.
	public static final int NE_XFLM_Q_EXPECTING_OPERATOR						= 0xD305; // Query setup error: Expecting an operator.
	public static final int NE_XFLM_Q_UNEXPECTED_COMMA							= 0xD306; // Query setup error: Unexpected comma.
	public static final int NE_XFLM_Q_EXPECTING_LPAREN							= 0xD307; // Query setup error: Expecting a left paren.
	public static final int NE_XFLM_Q_UNEXPECTED_VALUE							= 0xD308; // Query setup error: Unexpected value.
	public static final int NE_XFLM_Q_INVALID_NUM_FUNC_ARGS					= 0xD309; // Query setup error: Invalid number of arguments for a function.
	public static final int NE_XFLM_Q_UNEXPECTED_XPATH_COMPONENT			= 0xD30A; // Query setup error: Unexpected XPATH componenent.
	public static final int NE_XFLM_Q_ILLEGAL_LBRACKET							= 0xD30B; // Query setup error: Illegal left bracket ([).
	public static final int NE_XFLM_Q_ILLEGAL_RBRACKET							= 0xD30C; // Query setup error: Illegal right bracket (]).
	public static final int NE_XFLM_Q_ILLEGAL_OPERAND							= 0xD30D; // Query setup error: Operand for some operator is not valid for that operator type.
	public static final int NE_XFLM_Q_ALREADY_OPTIMIZED						= 0xD30E; // Operation is illegal, cannot change certain things after query has been optimized.
	public static final int NE_XFLM_Q_MISMATCHED_DB								= 0xD30F; // Database handle passed in does not match database associated with query.
	public static final int NE_XFLM_Q_ILLEGAL_OPERATOR							= 0xD310; // Illegal operator - cannot pass this operator into the addOperator method.
	public static final int NE_XFLM_Q_ILLEGAL_COMPARE_RULES					= 0xD311; // Illegal combination of comparison rules passed to addOperator method.
	public static final int NE_XFLM_Q_INCOMPLETE_QUERY_EXPR					= 0xD312; // Query setup error: Query expression is incomplete.
	public static final int NE_XFLM_Q_NOT_POSITIONED							= 0xD313; // Query not positioned due to previous error, cannot call getNext, getPrev, or getCurrent
	public static final int NE_XFLM_Q_INVALID_NODE_ID_VALUE					= 0xD314; // Query setup error: Invalid type of value constant used for node id value comparison.
	public static final int NE_XFLM_Q_INVALID_META_DATA_TYPE					= 0xD315; // Query setup error: Invalid meta data type specified.
	public static final int NE_XFLM_Q_NEW_EXPR_NOT_ALLOWED					= 0xD316; // Query setup error: Cannot add an expression to an XPATH component after having added an expression that tests context position.
	public static final int NE_XFLM_Q_INVALID_CONTEXT_POS						= 0xD317; // Invalid context position value encountered - must be a positive number.
	public static final int NE_XFLM_Q_INVALID_FUNC_ARG							= 0xD318; // Query setup error: Parameter to user-defined functions must be a single XPATH only.
	public static final int NE_XFLM_Q_EXPECTING_RPAREN							= 0xD319; // Query setup error: Expecting right paren.
	public static final int NE_XFLM_Q_TOO_LATE_TO_ADD_SORT_KEYS				= 0xD31A; // Query setup error: Cannot add sort keys after having called getFirst, getLast, getNext, or getPrev.
	public static final int NE_XFLM_Q_INVALID_SORT_KEY_COMPONENT			= 0xD31B; // Query setup error: Invalid sort key component number specified in query.
	public static final int NE_XFLM_Q_DUPLICATE_SORT_KEY_COMPONENT			= 0xD31C; // Query setup error: Duplicate sort key component number specified in query.
	public static final int NE_XFLM_Q_MISSING_SORT_KEY_COMPONENT			= 0xD31D; // Query setup error: Missing sort key component number in sort keys that were specified for query.
	public static final int NE_XFLM_Q_NO_SORT_KEY_COMPONENTS_SPECIFIED	= 0xD31E; // Query setup error: addSortKeys was called, but no sort key components were specified.
	public static final int NE_XFLM_Q_SORT_KEY_CONTEXT_MUST_BE_ELEMENT	= 0xD31F; // Query setup error: A sort key context cannot be an XML attribute.
	public static final int NE_XFLM_Q_INVALID_ELEMENT_NUM_IN_SORT_KEYS	= 0xD320; // Query setup error: The XML element number specified for a sort key in a query is invalid - no element definition in the dictionary.
	public static final int NE_XFLM_Q_INVALID_ATTR_NUM_IN_SORT_KEYS		= 0xD321; // Query setup error: The XML attribute number specified for a sort key in a query is invalid - no attribute definition in the dictionary.
	public static final int NE_XFLM_Q_NON_POSITIONABLE_QUERY					= 0xD322; // Attempt is being made to position in a query that is not positionable.
	public static final int NE_XFLM_Q_INVALID_POSITION							= 0xD323; // Attempt is being made to position to an invalid position in the result set.

	/****************************************************************************
	Desc:	NICI / Encryption Errors
	****************************************************************************/

	public static final int NE_XFLM_NICI_CONTEXT									= 0xD401; // Error occurred while creating NICI context for encryption/decryption.
	public static final int NE_XFLM_NICI_ATTRIBUTE_VALUE						= 0xD402; // Error occurred while accessing an attribute on a NICI encryption key.
	public static final int NE_XFLM_NICI_BAD_ATTRIBUTE							= 0xD403; // Value retrieved from an attribute on a NICI encryption key was bad.
	public static final int NE_XFLM_NICI_WRAPKEY_FAILED						= 0xD404; // Error occurred while wrapping a NICI encryption key in another NICI encryption key.
	public static final int NE_XFLM_NICI_UNWRAPKEY_FAILED						= 0xD405; // Error occurred while unwrapping a NICI encryption key that is wrapped in another NICI encryption key.
	public static final int NE_XFLM_NICI_INVALID_ALGORITHM					= 0xD406; // Attempt to use invalid NICI encryption algorithm. 
	public static final int NE_XFLM_NICI_GENKEY_FAILED							= 0xD407; // Error occurred while attempting to generate a NICI encryption key.
	public static final int NE_XFLM_NICI_BAD_RANDOM								= 0xD408; // Error occurred while generating random data using NICI.
	public static final int NE_XFLM_PBE_ENCRYPT_FAILED							= 0xD409; // Error occurred while attempting to wrap a NICI encryption key in a password.
	public static final int NE_XFLM_PBE_DECRYPT_FAILED							= 0xD40A; // Error occurred while attempting to unwrap a NICI encryption key that was previously wrapped in a password.
	public static final int NE_XFLM_DIGEST_INIT_FAILED							= 0xD40B; // Error occurred while attempting to initialize the NICI digest functionality.
	public static final int NE_XFLM_DIGEST_FAILED								= 0xD40C; // Error occurred while attempting to create a NICI digest. 
	public static final int NE_XFLM_INJECT_KEY_FAILED							= 0xD40D; // Error occurred while attempting to inject an encryption key into NICI. 
	public static final int NE_XFLM_NICI_FIND_INIT								= 0xD40E; // Error occurred while attempting to initialize NICI to find information on a NICI encryption key.
	public static final int NE_XFLM_NICI_FIND_OBJECT							= 0xD40F; // Error occurred while attempting to find information on a NICI encryption key.
	public static final int NE_XFLM_NICI_KEY_NOT_FOUND							= 0xD410; // Could not find the NICI encryption key or information on the NICI encryption key.
	public static final int NE_XFLM_NICI_ENC_INIT_FAILED						= 0xD411; // Error occurred while initializing NICI to encrypt data.
	public static final int NE_XFLM_NICI_ENCRYPT_FAILED						= 0xD412; // Error occurred while encrypting data.
	public static final int NE_XFLM_NICI_DECRYPT_INIT_FAILED					= 0xD413; // Error occurred while initializing NICI to decrypt data.
	public static final int NE_XFLM_NICI_DECRYPT_FAILED						= 0xD414; // Error occurred while decrypting data.
	public static final int NE_XFLM_NICI_WRAPKEY_NOT_FOUND					= 0xD415; // Could not find the NICI encryption key used to wrap another NICI encryption key.
	public static final int NE_XFLM_NOT_EXPECTING_PASSWORD					= 0xD416; // Password supplied when none was expected.
	public static final int NE_XFLM_EXPECTING_PASSWORD							= 0xD417; // No password supplied when one was required.
	public static final int NE_XFLM_EXTRACT_KEY_FAILED							= 0xD418; // Error occurred while attempting to extract a NICI encryption key.
	public static final int NE_XFLM_NICI_INIT_FAILED							= 0xD419; // Error occurred while initializing NICI.
	public static final int NE_XFLM_BAD_ENCKEY_SIZE								= 0xD41A; // Bad encryption key size found in roll-forward log packet.
	public static final int NE_XFLM_ENCRYPTION_UNAVAILABLE					= 0xD41B; // Attempt was made to encrypt data when NICI is unavailable.

	/****************************************************************************
	Desc:	Toolkit errors
	****************************************************************************/

	public static final int NE_FLM_BOF_HIT											= 0xC001;	///< = 0xC001 - Beginning of results encountered.
	public static final int NE_FLM_EOF_HIT											= 0xC002;	///< = 0xC002 - End of results encountered.
	public static final int NE_FLM_EXISTS											= 0xC004;	///< = 0xC004 - Object already exists.
	public static final int NE_FLM_FAILURE											= 0xC005;	///< = 0xC005 - Internal failure.
	public static final int NE_FLM_NOT_FOUND										= 0xC006;	///< = 0xC006 - An object was not found.
	public static final int NE_FLM_BTREE_ERROR									= 0xC012;	///< = 0xC012 - Corruption found in b-tree.
	public static final int NE_FLM_BTREE_FULL										= 0xC013;	///< = 0xC013 - B-tree cannot grow beyond current size.
	public static final int NE_FLM_CONV_DEST_OVERFLOW							= 0xC01C;	///< = 0xC01C - Destination buffer not large enough to hold data.
	public static final int NE_FLM_CONV_ILLEGAL									= 0xC01D;	///< = 0xC01D - Attempt to convert between data types is an unsupported conversion.
	public static final int NE_FLM_CONV_NUM_OVERFLOW							= 0xC020;	///< = 0xC020 - Numeric overflow (> upper bound) converting to numeric type.
	public static final int NE_FLM_DATA_ERROR										= 0xC022;	///< = 0xC022 - Corruption found in b-tree.
	public static final int NE_FLM_ILLEGAL_OP										= 0xC026;	///< = 0xC026 - Illegal operation
	public static final int NE_FLM_MEM												= 0xC037;	///< = 0xC037 - Attempt to allocate memory failed.
	public static final int NE_FLM_NOT_UNIQUE										= 0xC03E;	///< = 0xC03E - Non-unique key.
	public static final int NE_FLM_SYNTAX											= 0xC045;	///< = 0xC045 - Syntax error while parsing.
	public static final int NE_FLM_NOT_IMPLEMENTED								= 0xC05F;	///< = 0xC05F - Attempt was made to use a feature that is not implemented.
	public static final int NE_FLM_INVALID_PARM									= 0xC08B;	///< = 0xC08B - Invalid parameter passed into a function.

	// I/O Errors - Must be the same as they were for FLAIM.

	public static final int NE_FLM_IO_ACCESS_DENIED								= 0xC201;	///< = 0xC201 - Access to file is denied.\  Caller is not allowed access to a file.
	public static final int NE_FLM_IO_BAD_FILE_HANDLE							= 0xC202;	///< = 0xC202 - Bad file handle or file descriptor.
	public static final int NE_FLM_IO_COPY_ERR									= 0xC203;	///< = 0xC203 - Error occurred while copying a file.
	public static final int NE_FLM_IO_DISK_FULL									= 0xC204;	///< = 0xC204 - Disk full.
	public static final int NE_FLM_IO_END_OF_FILE								= 0xC205;	///< = 0xC205 - End of file reached while reading from the file.
	public static final int NE_FLM_IO_OPEN_ERR									= 0xC206;	///< = 0xC206 - Error while opening the file.
	public static final int NE_FLM_IO_SEEK_ERR									= 0xC207;	///< = 0xC207 - Error occurred while positioning (seeking) within a file.
	public static final int NE_FLM_IO_DIRECTORY_ERR								= 0xC208;	///< = 0xC208 - Error occurred while accessing or deleting a directory.
	public static final int NE_FLM_IO_PATH_NOT_FOUND							= 0xC209;	///< = 0xC209 - File not found.
	public static final int NE_FLM_IO_TOO_MANY_OPEN_FILES						= 0xC20A;	///< = 0xC20A - Too many files open.
	public static final int NE_FLM_IO_PATH_TOO_LONG								= 0xC20B;	///< = 0xC20B - File name too long.
	public static final int NE_FLM_IO_NO_MORE_FILES								= 0xC20C;	///< = 0xC20C - No more files in directory.
	public static final int NE_FLM_IO_DELETING_FILE								= 0xC20D;	///< = 0xC20D - Error occurred while deleting a file.
	public static final int NE_FLM_IO_FILE_LOCK_ERR								= 0xC20E;	///< = 0xC20E - Error attempting to acquire a byte-range lock on a file.
	public static final int NE_FLM_IO_FILE_UNLOCK_ERR							= 0xC20F;	///< = 0xC20F - Error attempting to release a byte-range lock on a file.
	public static final int NE_FLM_IO_PATH_CREATE_FAILURE						= 0xC210;	///< = 0xC210 - Error occurred while attempting to create a directory or sub-directory.
	public static final int NE_FLM_IO_RENAME_FAILURE							= 0xC211;	///< = 0xC211 - Error occurred while renaming a file.
	public static final int NE_FLM_IO_INVALID_PASSWORD							= 0xC212;	///< = 0xC212 - Invalid file password.
	public static final int NE_FLM_SETTING_UP_FOR_READ							= 0xC213;	///< = 0xC213 - Error occurred while setting up to perform a file read operation.
	public static final int NE_FLM_SETTING_UP_FOR_WRITE						= 0xC214;	///< = 0xC214 - Error occurred while setting up to perform a file write operation.
	public static final int NE_FLM_IO_CANNOT_REDUCE_PATH						= 0xC215;	///< = 0xC215 - Cannot reduce file name into more components.
	public static final int NE_FLM_INITIALIZING_IO_SYSTEM						= 0xC216;	///< = 0xC216 - Error occurred while setting up to access the file system.
	public static final int NE_FLM_FLUSHING_FILE									= 0xC217;	///< = 0xC217 - Error occurred while flushing file data buffers to disk.
	public static final int NE_FLM_IO_INVALID_FILENAME							= 0xC218;	///< = 0xC218 - Invalid file name.
	public static final int NE_FLM_IO_CONNECT_ERROR								= 0xC219;	///< = 0xC219 - Error connecting to a remote network resource.
	public static final int NE_FLM_OPENING_FILE									= 0xC21A;	///< = 0xC21A - Unexpected error occurred while opening a file.
	public static final int NE_FLM_DIRECT_OPENING_FILE							= 0xC21B;	///< = 0xC21B - Unexpected error occurred while opening a file in direct access mode.
	public static final int NE_FLM_CREATING_FILE									= 0xC21C;	///< = 0xC21C - Unexpected error occurred while creating a file.
	public static final int NE_FLM_DIRECT_CREATING_FILE						= 0xC21D;	///< = 0xC21D - Unexpected error occurred while creating a file in direct access mode.
	public static final int NE_FLM_READING_FILE									= 0xC21E;	///< = 0xC21E - Unexpected error occurred while reading a file.
	public static final int NE_FLM_DIRECT_READING_FILE							= 0xC21F;	///< = 0xC21F - Unexpected error occurred while reading a file in direct access mode.
	public static final int NE_FLM_WRITING_FILE									= 0xC220;	///< = 0xC220 - Unexpected error occurred while writing to a file.
	public static final int NE_FLM_DIRECT_WRITING_FILE							= 0xC221;	///< = 0xC221 - Unexpected error occurred while writing a file in direct access mode.
	public static final int NE_FLM_POSITIONING_IN_FILE							= 0xC222;	///< = 0xC222 - Unexpected error occurred while positioning within a file.
	public static final int NE_FLM_GETTING_FILE_SIZE							= 0xC223;	///< = 0xC223 - Unexpected error occurred while getting a file's size.
	public static final int NE_FLM_TRUNCATING_FILE								= 0xC224;	///< = 0xC224 - Unexpected error occurred while truncating a file.
	public static final int NE_FLM_PARSING_FILE_NAME							= 0xC225;	///< = 0xC225 - Unexpected error occurred while parsing a file's name.
	public static final int NE_FLM_CLOSING_FILE									= 0xC226;	///< = 0xC226 - Unexpected error occurred while closing a file.
	public static final int NE_FLM_GETTING_FILE_INFO							= 0xC227;	///< = 0xC227 - Unexpected error occurred while getting information about a file.
	public static final int NE_FLM_EXPANDING_FILE								= 0xC228;	///< = 0xC228 - Unexpected error occurred while expanding a file.
	public static final int NE_FLM_GETTING_FREE_BLOCKS							= 0xC229;	///< = 0xC229 - Unexpected error getting free blocks from file system.
	public static final int NE_FLM_CHECKING_FILE_EXISTENCE					= 0xC22A;	///< = 0xC22A - Unexpected error occurred while checking to see if a file exists.
	public static final int NE_FLM_RENAMING_FILE									= 0xC22B;	///< = 0xC22B - Unexpected error occurred while renaming a file.
	public static final int NE_FLM_SETTING_FILE_INFO							= 0xC22C;	///< = 0xC22C - Unexpected error occurred while setting a file's information.
	public static final int NE_FLM_IO_PENDING										= 0xC22D;	///< = 0xC22D - I/O has not yet completed
	public static final int NE_FLM_ASYNC_FAILED									= 0xC22E;	///< = 0xC22E - An async I/O operation failed
	public static final int NE_FLM_MISALIGNED_IO									= 0xC22F;	///< = 0xC22F - Misaligned buffer or offset encountered during I/O request
		
	// Stream Errors - These are new

	public static final int NE_FLM_STREAM_DECOMPRESS_ERROR					= 0xC400;	///< = 0xC400 - Error decompressing data stream.
	public static final int NE_FLM_STREAM_NOT_COMPRESSED						= 0xC401;	///< = 0xC401 - Attempting to decompress a data stream that is not compressed.
	public static final int NE_FLM_STREAM_TOO_MANY_FILES						= 0xC402;	///< = 0xC402 - Too many files in input stream.
		
	// Miscellaneous new toolkit errors
		
	public static final int NE_FLM_COULD_NOT_CREATE_SEMAPHORE				= 0xC500;	///< = 0xC500 - Could not create a semaphore.
	public static final int NE_FLM_BAD_UTF8										= 0xC501;	///< = 0xC501 - An invalid byte sequence was found in a UTF-8 string
	public static final int NE_FLM_ERROR_WAITING_ON_SEMAPHORE				= 0xC502;	///< = 0xC502 - Error occurred while waiting on a sempahore.
	public static final int NE_FLM_BAD_SEN											= 0xC503;	///< = 0xC503 - Invalid simple encoded number.
	public static final int NE_FLM_COULD_NOT_START_THREAD						= 0xC504;	///< = 0xC504 - Problem starting a new thread.
	public static final int NE_FLM_BAD_BASE64_ENCODING							= 0xC505;	///< = 0xC505 - Invalid base64 sequence encountered.
	public static final int NE_FLM_STREAM_EXISTS									= 0xC506;	///< = 0xC506 - Stream file already exists.
	public static final int NE_FLM_MULTIPLE_MATCHES								= 0xC507;	///< = 0xC507 - Multiple items matched but only one match was expected.
	public static final int NE_FLM_BTREE_KEY_SIZE								= 0xC508;	///< = 0xC508 - Invalid b-tree key size.
	public static final int NE_FLM_BTREE_BAD_STATE								= 0xC509;	///< = 0xC509 - B-tree operation cannot be completed.
	public static final int NE_FLM_COULD_NOT_CREATE_MUTEX						= 0xC50A;	///< = 0xC50A - Error occurred while creating or initializing a mutex.
	public static final int NE_FLM_BAD_PLATFORM_FORMAT							= 0xC50B;	///< = 0xC50B	- In-memory alignment of disk structures is incorrect
	public static final int NE_FLM_LOCK_REQ_TIMEOUT								= 0xC50C;	///< = 0xC50C	- Timeout while waiting for a lock object
	public static final int NE_FLM_WAIT_TIMEOUT									= 0xC50D;	///< = 0xC50D - Timeout while waiting on a semaphore, condition variable, or reader/writer lock
		
	// Network Errors - Must be the same as they were for FLAIM

	public static final int NE_FLM_NOIP_ADDR										= 0xC900;	///< = 0xC900 - IP address not found
	public static final int NE_FLM_SOCKET_FAIL									= 0xC901;	///< = 0xC901 - IP socket failure
	public static final int NE_FLM_CONNECT_FAIL									= 0xC902;	///< = 0xC902 - TCP/IP connection failure
	public static final int NE_FLM_BIND_FAIL										= 0xC903;	///< = 0xC903 - The TCP/IP services on your system may not be configured or installed.
	public static final int NE_FLM_LISTEN_FAIL									= 0xC904;	///< = 0xC904 - TCP/IP listen failed
	public static final int NE_FLM_ACCEPT_FAIL									= 0xC905;	///< = 0xC905 - TCP/IP accept failed
	public static final int NE_FLM_SELECT_ERR										= 0xC906;	///< = 0xC906 - TCP/IP select failed
	public static final int NE_FLM_SOCKET_SET_OPT_FAIL							= 0xC907;	///< = 0xC907 - TCP/IP socket operation failed
	public static final int NE_FLM_SOCKET_DISCONNECT							= 0xC908;	///< = 0xC908 - TCP/IP disconnected
	public static final int NE_FLM_SOCKET_READ_FAIL								= 0xC909;	///< = 0xC909 - TCP/IP read failed
	public static final int NE_FLM_SOCKET_WRITE_FAIL							= 0xC90A;	///< = 0xC90A - TCP/IP write failed
	public static final int NE_FLM_SOCKET_READ_TIMEOUT							= 0xC90B;	///< = 0xC90B - TCP/IP read timeout
	public static final int NE_FLM_SOCKET_WRITE_TIMEOUT						= 0xC90C;	///< = 0xC90C - TCP/IP write timeout
	public static final int NE_FLM_SOCKET_ALREADY_CLOSED						= 0xC90D;	///< = 0xC90D - Connection already closed
}
